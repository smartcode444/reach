//! Global emergency stop mechanism for Reach.
//!
//! Provides an immediate, non-blocking stop flag and optional background
//! Win32 hotkey listener running on a dedicated thread with its own message queue.
//!
//! When the emergency stop fires (via hotkey `Ctrl+Alt+Q` or programmatic call),
//! it sets an [`Arc<AtomicBool>`] stop flag, releases all simulated keys and
//! mouse buttons, and invokes registered cancellation and mode change callbacks.
//! Normal operation is restored by pressing `Ctrl+Alt+R` or calling [`EmergencyStop::resume`].
//!
//! # Operator vs Accessible Voice Method
//! Note: The global keyboard hotkey is intended for an assistant, caregiver,
//! operator, or helper who has physical access to the keyboard. For the user who
//! cannot operate physical input devices, the primary accessible emergency stop
//! method is the spoken voice command `"Reach stop"` (which already exists in the
//! NLU grammar and maps to setting the application mode to Sleep).

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread::{self, JoinHandle};

use windows::Win32::Foundation::{LPARAM, WPARAM};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    HOT_KEY_MODIFIERS, MOD_ALT, MOD_CONTROL, MOD_NOREPEAT, RegisterHotKey, UnregisterHotKey,
};
use windows::Win32::UI::WindowsAndMessaging::{
    GetMessageW, MSG, PostThreadMessageW, WM_HOTKEY, WM_QUIT,
};

use crate::release_all_keys;

// SAFETY: GetCurrentThreadId from kernel32 retrieves the calling thread's ID with no preconditions.
unsafe extern "system" {
    fn GetCurrentThreadId() -> u32;
}

/// Identifier for the emergency stop hotkey (`Ctrl+Alt+Q`).
pub const STOP_HOTKEY_ID: i32 = 1;

/// Identifier for the resume hotkey (`Ctrl+Alt+R`).
pub const RESUME_HOTKEY_ID: i32 = 2;

/// Errors that can occur during emergency stop registration and management.
#[derive(Debug)]
pub enum EmergencyStopError {
    /// Hotkey registration with Windows failed (e.g. key already registered).
    RegistrationFailed {
        /// Human-readable name of the hotkey combination.
        hotkey: &'static str,
        /// Detailed failure reason.
        reason: String,
    },
    /// Worker thread failed to start or died before registration finished.
    WorkerThreadDied,
}

impl fmt::Display for EmergencyStopError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RegistrationFailed { hotkey, reason } => {
                write!(
                    f,
                    "failed to register emergency stop hotkey {hotkey}: {reason}"
                )
            }
            Self::WorkerThreadDied => {
                write!(f, "emergency stop worker thread terminated unexpectedly")
            }
        }
    }
}

impl std::error::Error for EmergencyStopError {}

/// Global emergency stop controller.
///
/// Wraps an [`Arc<AtomicBool>`] flag that is checked between pipeline steps and
/// before input injection. Optionally manages a dedicated Windows thread running
/// a Win32 message loop that registers global operator hotkeys:
/// - `Ctrl+Alt+Q` to immediately trigger emergency stop.
/// - `Ctrl+Alt+R` to resume normal operation.
///
/// # Operator Hotkey Note
/// The hotkey listener is designed for an operator or helper. Voice users should
/// speak `"Reach stop"`.
pub struct EmergencyStop {
    /// The shared atomic flag signaling stop state.
    stop_flag: Arc<AtomicBool>,
    /// Windows thread ID of the dedicated message pump thread, if running.
    thread_id: Option<u32>,
    /// Thread join handle for clean shutdown.
    worker_handle: Option<JoinHandle<()>>,
    /// Optional callback invoked when emergency stop triggers.
    on_stop: Option<Arc<dyn Fn() + Send + Sync>>,
    /// Optional callback invoked when normal operation resumes.
    on_resume: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl EmergencyStop {
    /// Creates a new `EmergencyStop` controller wrapping `stop_flag` without registering hotkeys.
    ///
    /// Useful for test environments and headless execution.
    ///
    /// # Arguments
    /// * `stop_flag` - Shared atomic boolean indicating stop state.
    ///
    /// # Returns
    /// An initialized `EmergencyStop` instance.
    pub fn new(stop_flag: Arc<AtomicBool>) -> Self {
        Self {
            stop_flag,
            thread_id: None,
            worker_handle: None,
            on_stop: None,
            on_resume: None,
        }
    }

    /// Sets callbacks triggered on emergency stop and resume events.
    ///
    /// # Arguments
    /// * `on_stop` - Callback invoked when emergency stop fires.
    /// * `on_resume` - Callback invoked when resume fires.
    ///
    /// # Returns
    /// Self with configured callbacks.
    pub fn with_callbacks<S, R>(mut self, on_stop: S, on_resume: R) -> Self
    where
        S: Fn() + Send + Sync + 'static,
        R: Fn() + Send + Sync + 'static,
    {
        self.on_stop = Some(Arc::new(on_stop));
        self.on_resume = Some(Arc::new(on_resume));
        self
    }

    /// Registers global emergency stop (`Ctrl+Alt+Q`) and resume (`Ctrl+Alt+R`) hotkeys on a dedicated thread.
    ///
    /// Runs a Win32 message loop on the background thread. If registration fails
    /// (e.g. because another application owns `Ctrl+Alt+Q`), returns a typed
    /// [`EmergencyStopError::RegistrationFailed`] instead of panicking.
    ///
    /// # Arguments
    /// * `stop_flag` - Shared atomic boolean indicating stop state.
    ///
    /// # Returns
    /// An armed [`EmergencyStop`] running the hotkey listener.
    ///
    /// # Errors
    /// Returns [`EmergencyStopError`] if hotkey registration fails or worker thread aborts.
    pub fn register(stop_flag: Arc<AtomicBool>) -> Result<Self, EmergencyStopError> {
        Self::register_with_callbacks(stop_flag, None::<fn()>, None::<fn()>)
    }

    /// Registers global hotkeys with optional stop and resume notification callbacks.
    ///
    /// # Arguments
    /// * `stop_flag` - Shared atomic boolean indicating stop state.
    /// * `on_stop` - Optional callback invoked when stop hotkey triggers.
    /// * `on_resume` - Optional callback invoked when resume hotkey triggers.
    ///
    /// # Returns
    /// An armed [`EmergencyStop`] running the hotkey listener.
    ///
    /// # Errors
    /// Returns [`EmergencyStopError`] if hotkey registration fails or worker thread aborts.
    pub fn register_with_callbacks<S, R>(
        stop_flag: Arc<AtomicBool>,
        on_stop: Option<S>,
        on_resume: Option<R>,
    ) -> Result<Self, EmergencyStopError>
    where
        S: Fn() + Send + Sync + 'static,
        R: Fn() + Send + Sync + 'static,
    {
        let on_stop_arc: Option<Arc<dyn Fn() + Send + Sync>> =
            on_stop.map(|cb| Arc::new(cb) as Arc<dyn Fn() + Send + Sync>);
        let on_resume_arc: Option<Arc<dyn Fn() + Send + Sync>> =
            on_resume.map(|cb| Arc::new(cb) as Arc<dyn Fn() + Send + Sync>);

        let (ready_tx, ready_rx) = mpsc::channel::<Result<u32, EmergencyStopError>>();

        let worker_stop_flag = stop_flag.clone();
        let worker_on_stop = on_stop_arc.clone();
        let worker_on_resume = on_resume_arc.clone();

        let worker_handle = thread::Builder::new()
            .name("reach-emergency-stop".to_owned())
            .spawn(move || {
                run_hotkey_worker(worker_stop_flag, worker_on_stop, worker_on_resume, ready_tx);
            })
            .map_err(|e| EmergencyStopError::RegistrationFailed {
                hotkey: "Ctrl+Alt+Q",
                reason: format!("failed to spawn emergency stop thread: {e}"),
            })?;

        let thread_id = ready_rx
            .recv()
            .map_err(|_| EmergencyStopError::WorkerThreadDied)??;

        Ok(Self {
            stop_flag,
            thread_id: Some(thread_id),
            worker_handle: Some(worker_handle),
            on_stop: on_stop_arc,
            on_resume: on_resume_arc,
        })
    }

    /// Triggers the emergency stop.
    ///
    /// Sets the stop flag to `true`, attempts to release all simulated keys,
    /// and invokes the `on_stop` callback if registered.
    pub fn stop(&self) {
        self.stop_flag.store(true, Ordering::SeqCst);
        let _ = release_all_keys();
        if let Some(ref cb) = self.on_stop {
            cb();
        }
    }

    /// Resumes normal operation.
    ///
    /// Clears the stop flag to `false` and invokes the `on_resume` callback if registered.
    pub fn resume(&self) {
        self.stop_flag.store(false, Ordering::SeqCst);
        if let Some(ref cb) = self.on_resume {
            cb();
        }
    }

    /// Returns whether the emergency stop flag is currently active.
    ///
    /// # Returns
    /// `true` if stopped, `false` otherwise.
    pub fn is_stopped(&self) -> bool {
        self.stop_flag.load(Ordering::SeqCst)
    }

    /// Returns a clone of the underlying shared atomic stop flag.
    ///
    /// # Returns
    /// [`Arc<AtomicBool>`] handle.
    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        self.stop_flag.clone()
    }
}

impl Drop for EmergencyStop {
    /// Signals the dedicated worker thread to terminate and waits for it to exit.
    fn drop(&mut self) {
        if let Some(thread_id) = self.thread_id.take() {
            // SAFETY: PostThreadMessageW sends WM_QUIT to the dedicated thread's queue,
            // causing GetMessageW to unblock and return 0 (terminating the loop).
            unsafe {
                let _ = PostThreadMessageW(thread_id, WM_QUIT, WPARAM(0), LPARAM(0));
            }
        }
        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }
    }
}

/// Runs the dedicated Win32 message pump and hotkey listener on a worker thread.
///
/// # Arguments
/// * `stop_flag` - Shared atomic stop flag set on `Ctrl+Alt+Q` and cleared on `Ctrl+Alt+R`.
/// * `on_stop` - Optional callback invoked when emergency stop fires.
/// * `on_resume` - Optional callback invoked when resume fires.
/// * `ready_tx` - Channel used to report registration success with thread ID or failure.
fn run_hotkey_worker(
    stop_flag: Arc<AtomicBool>,
    on_stop: Option<Arc<dyn Fn() + Send + Sync>>,
    on_resume: Option<Arc<dyn Fn() + Send + Sync>>,
    ready_tx: mpsc::Sender<Result<u32, EmergencyStopError>>,
) {
    // SAFETY: GetCurrentThreadId retrieves the calling thread's ID.
    let thread_id = unsafe { GetCurrentThreadId() };

    let modifiers: HOT_KEY_MODIFIERS = MOD_CONTROL | MOD_ALT | MOD_NOREPEAT;

    // Register stop hotkey: Ctrl + Alt + Q (virtual key 0x51 / b'Q')
    // SAFETY: RegisterHotKey with None window registers the hotkey for the calling thread's message queue.
    let stop_result = unsafe { RegisterHotKey(None, STOP_HOTKEY_ID, modifiers, b'Q' as u32) };

    if let Err(err) = stop_result {
        let _ = ready_tx.send(Err(EmergencyStopError::RegistrationFailed {
            hotkey: "Ctrl+Alt+Q",
            reason: err.to_string(),
        }));
        return;
    }

    // Register resume hotkey: Ctrl + Alt + R (virtual key 0x52 / b'R')
    // SAFETY: RegisterHotKey with None window registers the hotkey for the calling thread's message queue.
    let resume_result = unsafe { RegisterHotKey(None, RESUME_HOTKEY_ID, modifiers, b'R' as u32) };

    if let Err(err) = resume_result {
        // SAFETY: Clean up previously registered stop hotkey.
        unsafe {
            let _ = UnregisterHotKey(None, STOP_HOTKEY_ID);
        }
        let _ = ready_tx.send(Err(EmergencyStopError::RegistrationFailed {
            hotkey: "Ctrl+Alt+R",
            reason: err.to_string(),
        }));
        return;
    }

    if ready_tx.send(Ok(thread_id)).is_err() {
        // SAFETY: Caller dropped the receiver, unregister both hotkeys.
        unsafe {
            let _ = UnregisterHotKey(None, STOP_HOTKEY_ID);
            let _ = UnregisterHotKey(None, RESUME_HOTKEY_ID);
        }
        return;
    }

    let mut msg = MSG::default();
    // SAFETY: GetMessageW retrieves messages from the calling thread's queue.
    // Returns 0 when WM_QUIT is retrieved, terminating the loop.
    while unsafe { GetMessageW(&mut msg, None, 0, 0).as_bool() } {
        if msg.message == WM_HOTKEY {
            let id = msg.wParam.0 as i32;
            if id == STOP_HOTKEY_ID {
                stop_flag.store(true, Ordering::SeqCst);
                let _ = release_all_keys();
                if let Some(ref cb) = on_stop {
                    cb();
                }
            } else if id == RESUME_HOTKEY_ID {
                stop_flag.store(false, Ordering::SeqCst);
                if let Some(ref cb) = on_resume {
                    cb();
                }
            }
        }
    }

    // SAFETY: Unregister thread hotkeys before thread exits.
    unsafe {
        let _ = UnregisterHotKey(None, STOP_HOTKEY_ID);
        let _ = UnregisterHotKey(None, RESUME_HOTKEY_ID);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that a newly initialized EmergencyStop reflects the initial flag value.
    #[test]
    fn flag_logic_initial_state() {
        let flag = Arc::new(AtomicBool::new(false));
        let stop = EmergencyStop::new(flag.clone());
        assert!(!stop.is_stopped());
        assert!(!flag.load(Ordering::SeqCst));
    }

    /// Verifies that calling stop() sets the flag and is_stopped reports true.
    #[test]
    fn flag_logic_stop_sets_flag() {
        let flag = Arc::new(AtomicBool::new(false));
        let stop = EmergencyStop::new(flag.clone());
        stop.stop();
        assert!(stop.is_stopped());
        assert!(flag.load(Ordering::SeqCst));
    }

    /// Verifies that calling resume() clears the flag.
    #[test]
    fn flag_logic_resume_clears_flag() {
        let flag = Arc::new(AtomicBool::new(true));
        let stop = EmergencyStop::new(flag.clone());
        stop.resume();
        assert!(!stop.is_stopped());
        assert!(!flag.load(Ordering::SeqCst));
    }

    /// Verifies multiple state transitions between stop and resume.
    #[test]
    fn flag_logic_transitions() {
        let flag = Arc::new(AtomicBool::new(false));
        let stop = EmergencyStop::new(flag.clone());

        stop.stop();
        assert!(stop.is_stopped());

        stop.resume();
        assert!(!stop.is_stopped());

        stop.stop();
        assert!(stop.is_stopped());
    }

    /// Verifies that configured callbacks are invoked on stop and resume.
    #[test]
    fn flag_logic_callbacks_invoked() {
        let flag = Arc::new(AtomicBool::new(false));
        let stop_called = Arc::new(AtomicBool::new(false));
        let resume_called = Arc::new(AtomicBool::new(false));

        let s_clone = stop_called.clone();
        let r_clone = resume_called.clone();

        let stop = EmergencyStop::new(flag).with_callbacks(
            move || s_clone.store(true, Ordering::SeqCst),
            move || r_clone.store(true, Ordering::SeqCst),
        );

        stop.stop();
        assert!(stop_called.load(Ordering::SeqCst));
        assert!(!resume_called.load(Ordering::SeqCst));

        stop.resume();
        assert!(resume_called.load(Ordering::SeqCst));
    }

    /// Verifies thread-safe concurrent access to the emergency stop flag.
    #[test]
    fn flag_logic_concurrent_access() {
        let flag = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(EmergencyStop::new(flag));

        let mut handles = Vec::new();
        for _ in 0..4 {
            let s = stop.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let _ = s.is_stopped();
                }
            }));
        }

        stop.stop();
        assert!(stop.is_stopped());
        stop.resume();
        assert!(!stop.is_stopped());

        for h in handles {
            h.join().unwrap();
        }
    }

    /// Manual test to verify Win32 hotkey registration on Windows.
    #[test]
    #[ignore = "manual desktop test for global hotkey registration"]
    fn manual_hotkey_registration() {
        let flag = Arc::new(AtomicBool::new(false));
        match EmergencyStop::register(flag.clone()) {
            Ok(controller) => {
                assert!(!controller.is_stopped());
                // Controller drop unregisters hotkeys and cleanly joins thread.
                drop(controller);
            }
            Err(EmergencyStopError::RegistrationFailed { hotkey, reason }) => {
                eprintln!("Hotkey {hotkey} already in use on this desktop: {reason}");
            }
            Err(err) => panic!("unexpected emergency stop error: {err}"),
        }
    }
}
