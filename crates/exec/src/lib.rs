//! Input injection, app and URL launching: the execution engine for Reach.
//!
//! Provides the concrete [`GlobalExecutor`] implementing [`reach_core::Executor`]
//! for keyboard presses, window scrolling, zooming, and application launching.
//! Also provides coordinate-based mouse click and string typing helpers for
//! fallback handling from the `screen` crate.
//!
//! Key injection is centralized, guarded by a [`ModifierGuard`] and panic hook
//! to ensure simulated keys are never left held down on failure or exit.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use enigo::{
    Axis, Button, Coordinate, Direction as EnigoDirection, Enigo, Key as EnigoKey, Keyboard, Mouse,
    Settings,
};
use reach_core::{Direction, Executor, Intent, Key, ZoomDir};
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};

/// Default number of scroll increments when unspecified.
pub const DEFAULT_SCROLL_STEPS: i32 = 3;

/// A sink that accepts individual key events.
///
/// Production uses [`Enigo`]; tests use a recorder so the exact sequence of
/// press, click and release events can be asserted without touching the desktop.
pub trait KeySink {
    /// Sends one key event in the requested direction.
    ///
    /// # Arguments
    /// * `key` - The key to send.
    /// * `direction` - Whether to press, release or click the key.
    ///
    /// # Errors
    /// Returns a message when the platform injector rejects the event.
    fn send_key(&mut self, key: EnigoKey, direction: EnigoDirection) -> Result<(), String>;
}

impl KeySink for Enigo {
    /// Forwards one key event to the platform injector.
    fn send_key(&mut self, key: EnigoKey, direction: EnigoDirection) -> Result<(), String> {
        Keyboard::key(self, key, direction).map_err(|e| e.to_string())
    }
}

/// Guard that ensures a held modifier key is released when dropped or on panic.
pub struct ModifierGuard<'a, S: KeySink> {
    /// Mutable reference to the key sink.
    sink: &'a mut S,
    /// The modifier key to release on drop.
    key: EnigoKey,
    /// Whether the guard is actively holding down the modifier.
    active: bool,
}

impl<'a, S: KeySink> ModifierGuard<'a, S> {
    /// Presses `key` and arms the guard to release it on drop.
    ///
    /// # Arguments
    /// * `sink` - The sink that receives the press and the eventual release.
    /// * `key` - The modifier key to press.
    ///
    /// # Returns
    /// An armed `ModifierGuard` if key press succeeded.
    ///
    /// # Errors
    /// Returns a contextualised error if the modifier press failed.
    pub fn press(sink: &'a mut S, key: EnigoKey) -> Result<Self, String> {
        sink.send_key(key, EnigoDirection::Press)
            .map_err(|e| format!("failed to press modifier key: {e}"))?;
        Ok(Self {
            sink,
            key,
            active: true,
        })
    }
}

impl<S: KeySink> Drop for ModifierGuard<'_, S> {
    /// Releases the held modifier, ignoring release errors.
    fn drop(&mut self) {
        if self.active {
            let _ = self.sink.send_key(self.key, EnigoDirection::Release);
            self.active = false;
        }
    }
}

/// Presses `key` while `modifier` is held, always releasing the modifier.
///
/// # Arguments
/// * `sink` - The sink that receives the events.
/// * `modifier` - The modifier to hold, usually Ctrl.
/// * `key` - The key to click while the modifier is held.
///
/// # Returns
/// `Ok(())` after the click and the modifier release.
///
/// # Errors
/// Returns a contextualised error if the modifier press or the key click
/// fails. The modifier is still released when the key click fails.
fn press_with_modifier<S: KeySink>(
    sink: &mut S,
    modifier: EnigoKey,
    key: EnigoKey,
) -> Result<(), String> {
    let guard = ModifierGuard::press(sink, modifier)?;
    guard
        .sink
        .send_key(key, EnigoDirection::Click)
        .map_err(|e| format!("failed to click key: {e}"))?;
    drop(guard);
    Ok(())
}

/// Flag tracking whether the global panic hook has been registered.
static PANIC_HOOK_INSTALLED: AtomicBool = AtomicBool::new(false);

/// Installs a global panic hook ensuring simulated modifier keys are released.
///
/// Safe to call multiple times; only installs once.
pub fn install_panic_key_release_hook() {
    if !PANIC_HOOK_INSTALLED.swap(true, Ordering::SeqCst) {
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            // Attempt emergency release of common modifier keys.
            if let Ok(mut enigo) = Enigo::new(&Settings::default()) {
                let _ = enigo.key(EnigoKey::Control, EnigoDirection::Release);
                let _ = enigo.key(EnigoKey::Shift, EnigoDirection::Release);
                let _ = enigo.key(EnigoKey::Alt, EnigoDirection::Release);
            }
            prev_hook(info);
        }));
    }
}

/// Maps a [`reach_core::Key`] to the corresponding [`enigo::Key`].
///
/// # Arguments
/// * `key` - The typed Reach key.
///
/// # Returns
/// The corresponding Enigo platform key.
pub fn map_key(key: Key) -> EnigoKey {
    match key {
        Key::Enter => EnigoKey::Return,
        Key::Tab => EnigoKey::Tab,
        Key::Escape => EnigoKey::Escape,
        Key::Backspace => EnigoKey::Backspace,
        Key::Delete => EnigoKey::Delete,
        Key::Up => EnigoKey::UpArrow,
        Key::Down => EnigoKey::DownArrow,
        Key::Left => EnigoKey::LeftArrow,
        Key::Right => EnigoKey::RightArrow,
        Key::PageUp => EnigoKey::PageUp,
        Key::PageDown => EnigoKey::PageDown,
        Key::Home => EnigoKey::Home,
        Key::End => EnigoKey::End,
    }
}

/// Maps a [`reach_core::Direction`] and optional step count to Enigo scroll parameters.
///
/// # Arguments
/// * `dir` - Scroll direction.
/// * `amount` - Optional step count.
///
/// # Returns
/// A tuple `(length, axis)` ready for [`enigo::Mouse::scroll`].
pub fn map_scroll(dir: Direction, amount: Option<u32>) -> (i32, Axis) {
    let steps = amount.map_or(DEFAULT_SCROLL_STEPS, |a| a as i32);
    match dir {
        Direction::Down => (steps, Axis::Vertical),
        Direction::Up => (-steps, Axis::Vertical),
        Direction::Right => (steps, Axis::Horizontal),
        Direction::Left => (-steps, Axis::Horizontal),
    }
}

/// Simulates a physical left mouse click at the given absolute screen coordinates.
///
/// # Arguments
/// * `x` - Absolute horizontal screen coordinate.
/// * `y` - Absolute vertical screen coordinate.
///
/// # Returns
/// `Ok(())` on success.
///
/// # Errors
/// Returns an error string if mouse cursor movement or button click fails.
pub fn click_at(x: i32, y: i32) -> Result<(), String> {
    let mut enigo =
        Enigo::new(&Settings::default()).map_err(|e| format!("failed to initialize enigo: {e}"))?;
    enigo
        .move_mouse(x, y, Coordinate::Abs)
        .map_err(|e| format!("failed to move mouse to ({x}, {y}): {e}"))?;
    enigo
        .button(Button::Left, EnigoDirection::Click)
        .map_err(|e| format!("failed to click left mouse button: {e}"))?;
    Ok(())
}

/// Injects text as keyboard input.
///
/// # Arguments
/// * `text` - The string to type out.
///
/// # Returns
/// `Ok(())` on success.
///
/// # Errors
/// Returns an error string if text injection fails.
pub fn type_str(text: &str) -> Result<(), String> {
    let mut enigo =
        Enigo::new(&Settings::default()).map_err(|e| format!("failed to initialize enigo: {e}"))?;
    enigo
        .text(text)
        .map_err(|e| format!("failed to type text: {e}"))?;
    Ok(())
}

/// Allowed applications for direct launching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllowedApp {
    /// Google Chrome web browser.
    Chrome,
    /// Windows Notepad text editor.
    Notepad,
    /// Windows File Explorer.
    Explorer,
}

/// Validates that an application name belongs to the safe allow-list.
///
/// Only "chrome", "notepad", and "explorer" are allowed (case-insensitive,
/// trimming whitespace and ignoring an optional `.exe` suffix).
///
/// # Arguments
/// * `name` - The spoken or typed application name.
///
/// # Returns
/// An [`AllowedApp`] variant identifying the approved executable.
///
/// # Errors
/// Returns an error message if the name is empty or not in the allow-list.
pub fn validate_app_name(name: &str) -> Result<AllowedApp, String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err("application name cannot be empty".to_owned());
    }
    let lower = trimmed.to_lowercase();
    let stripped = lower.strip_suffix(".exe").unwrap_or(&lower);
    match stripped {
        "chrome" => Ok(AllowedApp::Chrome),
        "notepad" => Ok(AllowedApp::Notepad),
        "explorer" => Ok(AllowedApp::Explorer),
        other => Err(format!(
            "unknown or disallowed application: '{other}'. Only 'chrome', 'notepad', and 'explorer' are allowed"
        )),
    }
}

/// Validates that a URL string uses an approved `http` or `https` scheme.
///
/// Rejects non-web schemes (such as `ftp://`, `file://`, `javascript:`, or `data:`)
/// and malformed or control-character-containing URLs.
///
/// # Arguments
/// * `url` - The web URL string to validate.
///
/// # Returns
/// A validated string slice referencing the trimmed URL.
///
/// # Errors
/// Returns an error message if the URL does not start with `http://` or `https://`,
/// is empty, or contains control characters.
pub fn validate_url(url: &str) -> Result<&str, String> {
    let trimmed = url.trim();
    if trimmed.is_empty() {
        return Err("URL cannot be empty".to_owned());
    }
    if trimmed.chars().any(|c| c.is_control()) {
        return Err("URL contains invalid control characters".to_owned());
    }
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        Ok(trimmed)
    } else {
        Err(format!(
            "invalid URL scheme: only http:// and https:// are allowed, got: '{trimmed}'"
        ))
    }
}

/// Returns standard file paths where Google Chrome is installed on Windows.
///
/// Checks environment variables `ProgramFiles`, `ProgramFiles(x86)`, and `LOCALAPPDATA`,
/// alongside standard drive C installation paths.
///
/// # Returns
/// A list of candidate [`PathBuf`] locations for `chrome.exe`.
pub fn standard_chrome_paths() -> Vec<PathBuf> {
    let mut candidates = Vec::new();

    if let Ok(pf) = std::env::var("ProgramFiles") {
        candidates.push(PathBuf::from(pf).join(r"Google\Chrome\Application\chrome.exe"));
    }
    if let Ok(pfx86) = std::env::var("ProgramFiles(x86)") {
        candidates.push(PathBuf::from(pfx86).join(r"Google\Chrome\Application\chrome.exe"));
    }
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        candidates.push(PathBuf::from(local).join(r"Google\Chrome\Application\chrome.exe"));
    }

    candidates.push(PathBuf::from(
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ));
    candidates.push(PathBuf::from(
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ));

    candidates
}

/// Locates the `chrome.exe` binary in standard installation directories.
///
/// # Returns
/// `Some(PathBuf)` if `chrome.exe` is found on disk, or `None`.
pub fn find_chrome_path() -> Option<PathBuf> {
    standard_chrome_paths()
        .into_iter()
        .find(|candidate| candidate.is_file())
}

/// Opens a web URL in Google Chrome with `--force-renderer-accessibility` enabled.
///
/// First validates that the URL uses an `http` or `https` scheme. If Chrome is found
/// in standard installation locations, it is launched with `--force-renderer-accessibility`
/// followed by the URL so Chrome exposes its accessibility tree to Reach.
/// If Chrome is not found on disk, falls back to the system default browser via the `open` crate.
///
/// # Arguments
/// * `url` - The web address to open.
///
/// # Returns
/// A descriptive message confirming the browser launch.
///
/// # Errors
/// Returns an error if:
/// * The URL scheme is not `http` or `https`.
/// * Spawning the browser process fails.
pub fn open_url(url: &str) -> Result<String, String> {
    let normalized = if url.starts_with("http://") || url.starts_with("https://") {
        url.trim().to_owned()
    } else if !url.contains("://") && (url.contains('.') || url.starts_with("localhost")) {
        format!("https://{}", url.trim())
    } else {
        url.trim().to_owned()
    };

    let valid = validate_url(&normalized)?;

    if let Some(chrome_path) = find_chrome_path() {
        Command::new(chrome_path)
            .arg("--force-renderer-accessibility")
            .arg(valid)
            .spawn()
            .map_err(|e| format!("failed to launch Chrome with URL: {e}"))?;
        Ok(format!("opened URL in Chrome: {valid}"))
    } else {
        open::that_detached(valid)
            .map_err(|e| format!("failed to open URL in default browser: {e}"))?;
        Ok(format!("opened URL in default browser: {valid}"))
    }
}

/// Launches an application from the allow-list using explicit argument lists.
///
/// Never passes user speech or shell strings to `cmd` or `powershell`.
/// Mapped to fixed executables:
/// - "chrome": locates `chrome.exe` in standard paths and launches with `--force-renderer-accessibility`.
/// - "notepad": launches `notepad.exe`.
/// - "explorer": launches `explorer.exe`.
///
/// # Arguments
/// * `name` - The application name to launch ("chrome", "notepad", or "explorer").
///
/// # Returns
/// A descriptive message confirming application launch.
///
/// # Errors
/// Returns an error if the name is not in the allow-list or process spawning fails.
pub fn open_app(name: &str) -> Result<String, String> {
    let app = validate_app_name(name)?;
    match app {
        AllowedApp::Chrome => {
            let mut cmd = if let Some(chrome_path) = find_chrome_path() {
                Command::new(chrome_path)
            } else {
                Command::new("chrome.exe")
            };
            cmd.arg("--force-renderer-accessibility")
                .spawn()
                .map_err(|e| format!("failed to launch Chrome: {e}"))?;
            Ok("opened app: chrome".to_owned())
        }
        AllowedApp::Notepad => {
            Command::new("notepad.exe")
                .spawn()
                .map_err(|e| format!("failed to launch Notepad: {e}"))?;
            Ok("opened app: notepad".to_owned())
        }
        AllowedApp::Explorer => {
            Command::new("explorer.exe")
                .spawn()
                .map_err(|e| format!("failed to launch Explorer: {e}"))?;
            Ok("opened app: explorer".to_owned())
        }
    }
}

/// Sends the browser zoom shortcut for `direction`.
///
/// Chrome expects Ctrl and the unshifted `=` key for zoom in. Sending
/// `Key::Unicode('+')` fails on Windows because `VkKeyScanExW` puts the Shift
/// state in the high byte of the virtual key and enigo passes that whole value
/// through as an invalid key code. On the default US layout the unshifted `=`
/// maps to `VK_OEM_PLUS` without the Shift bit, while Ctrl + `-` and Ctrl + `0`
/// need no Shift.
///
/// # Arguments
/// * `sink` - The sink that receives the key sequence.
/// * `direction` - Which zoom shortcut to send.
///
/// # Returns
/// `Ok(())` after the shortcut has been sent.
///
/// # Errors
/// Returns the sink's error if any event in the sequence fails.
fn send_zoom<S: KeySink>(sink: &mut S, direction: ZoomDir) -> Result<(), String> {
    let key = match direction {
        ZoomDir::In => EnigoKey::Unicode('='),
        ZoomDir::Out => EnigoKey::Unicode('-'),
        ZoomDir::Reset => EnigoKey::Unicode('0'),
    };
    press_with_modifier(sink, EnigoKey::Control, key)
}

/// Executor for global computer actions: key presses, scrolling, zooming, and opening apps/URLs.
pub struct GlobalExecutor {
    /// Underlying platform input injector.
    enigo: Enigo,
}

impl GlobalExecutor {
    /// Creates a new `GlobalExecutor` with default settings and registers the panic hook.
    ///
    /// # Returns
    /// An initialized `GlobalExecutor`.
    ///
    /// # Errors
    /// Returns error string if the platform input injector cannot be initialized.
    pub fn new() -> Result<Self, String> {
        install_panic_key_release_hook();
        let enigo = Enigo::new(&Settings::default())
            .map_err(|e| format!("failed to initialize enigo: {e}"))?;
        Ok(Self { enigo })
    }
}

impl Default for GlobalExecutor {
    fn default() -> Self {
        Self::new().unwrap_or_else(|e| {
            // In tests or headless environments, fallback to default settings struct.
            panic!("failed to initialize default GlobalExecutor: {e}")
        })
    }
}

impl Executor for GlobalExecutor {
    fn run(&mut self, intent: &Intent) -> Result<String, String> {
        match intent {
            Intent::PressKey { key } => {
                let enigo_key = map_key(*key);
                self.enigo
                    .key(enigo_key, EnigoDirection::Click)
                    .map_err(|e| format!("failed to press key: {e}"))?;
                Ok(format!("pressed {:?}", key))
            }
            Intent::Scroll { direction, amount } => {
                let (length, axis) = map_scroll(*direction, *amount);
                self.enigo
                    .scroll(length, axis)
                    .map_err(|e| format!("failed to scroll: {e}"))?;
                Ok(format!(
                    "scrolled {:?} by {} steps",
                    direction,
                    length.abs()
                ))
            }
            Intent::Zoom { direction } => {
                send_zoom(&mut self.enigo, *direction)?;
                Ok(match direction {
                    ZoomDir::In => "zoomed in",
                    ZoomDir::Out => "zoomed out",
                    ZoomDir::Reset => "reset zoom",
                }
                .to_owned())
            }
            Intent::OpenApp { name } => open_app(name),
            Intent::OpenUrl { url } => open_url(url),
            unhandled => Err(format!(
                "intent not handled by GlobalExecutor: {:?}",
                unhandled
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests mapping of core Key enum to Enigo keys.
    #[test]
    fn maps_all_core_keys() {
        assert_eq!(map_key(Key::Enter), EnigoKey::Return);
        assert_eq!(map_key(Key::Tab), EnigoKey::Tab);
        assert_eq!(map_key(Key::Escape), EnigoKey::Escape);
        assert_eq!(map_key(Key::Backspace), EnigoKey::Backspace);
        assert_eq!(map_key(Key::Delete), EnigoKey::Delete);
        assert_eq!(map_key(Key::Up), EnigoKey::UpArrow);
        assert_eq!(map_key(Key::Down), EnigoKey::DownArrow);
        assert_eq!(map_key(Key::Left), EnigoKey::LeftArrow);
        assert_eq!(map_key(Key::Right), EnigoKey::RightArrow);
        assert_eq!(map_key(Key::PageUp), EnigoKey::PageUp);
        assert_eq!(map_key(Key::PageDown), EnigoKey::PageDown);
        assert_eq!(map_key(Key::Home), EnigoKey::Home);
        assert_eq!(map_key(Key::End), EnigoKey::End);
    }

    /// Tests scroll direction and step mapping.
    #[test]
    fn maps_scroll_directions_and_amounts() {
        assert_eq!(map_scroll(Direction::Down, None), (3, Axis::Vertical));
        assert_eq!(map_scroll(Direction::Up, Some(5)), (-5, Axis::Vertical));
        assert_eq!(map_scroll(Direction::Right, Some(2)), (2, Axis::Horizontal));
        assert_eq!(map_scroll(Direction::Left, None), (-3, Axis::Horizontal));
    }

    /// Records every key event so the zoom sequence can be asserted.
    #[derive(Debug, Default)]
    struct RecordingKeySink {
        /// Events in the order they were sent.
        events: Vec<(EnigoKey, EnigoDirection)>,
    }

    impl KeySink for RecordingKeySink {
        /// Records one key event.
        fn send_key(&mut self, key: EnigoKey, direction: EnigoDirection) -> Result<(), String> {
            self.events.push((key, direction));
            Ok(())
        }
    }

    /// Zoom in must send Ctrl plus the unshifted equals key, then release Ctrl.
    #[test]
    fn zoom_in_sends_ctrl_and_unshifted_equals() {
        let mut sink = RecordingKeySink::default();
        send_zoom(&mut sink, ZoomDir::In).unwrap();
        assert_eq!(
            sink.events,
            vec![
                (EnigoKey::Control, EnigoDirection::Press),
                (EnigoKey::Unicode('='), EnigoDirection::Click),
                (EnigoKey::Control, EnigoDirection::Release),
            ]
        );
    }

    /// Zoom out must keep Ctrl plus minus, then release Ctrl.
    #[test]
    fn zoom_out_sends_ctrl_and_minus() {
        let mut sink = RecordingKeySink::default();
        send_zoom(&mut sink, ZoomDir::Out).unwrap();
        assert_eq!(
            sink.events,
            vec![
                (EnigoKey::Control, EnigoDirection::Press),
                (EnigoKey::Unicode('-'), EnigoDirection::Click),
                (EnigoKey::Control, EnigoDirection::Release),
            ]
        );
    }

    /// Zoom reset must keep Ctrl plus zero, then release Ctrl.
    #[test]
    fn zoom_reset_sends_ctrl_and_zero() {
        let mut sink = RecordingKeySink::default();
        send_zoom(&mut sink, ZoomDir::Reset).unwrap();
        assert_eq!(
            sink.events,
            vec![
                (EnigoKey::Control, EnigoDirection::Press),
                (EnigoKey::Unicode('0'), EnigoDirection::Click),
                (EnigoKey::Control, EnigoDirection::Release),
            ]
        );
    }

    /// Manual check that the real shortcuts change Chrome's zoom level.
    ///
    /// Open Chrome, note the page zoom, then run:
    /// `cargo test -p exec zoom_shortcuts_manually -- --ignored --nocapture`
    /// and watch the page zoom in, out and reset.
    #[test]
    #[ignore = "touches the real desktop; run manually with Chrome focused"]
    fn zoom_shortcuts_manually() {
        let mut exec = GlobalExecutor::new().expect("failed to initialize executor");
        for direction in [ZoomDir::In, ZoomDir::Out, ZoomDir::Reset] {
            let message = exec
                .run(&Intent::Zoom { direction })
                .expect("zoom shortcut failed");
            println!("{message}");
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
    }

    /// Tests executor rejects unhandled element intents.
    #[test]
    fn executor_rejects_element_intent() {
        let mut exec = GlobalExecutor::new().expect("failed to init executor");
        let intent = Intent::ClickElement {
            target: "Submit".to_owned(),
        };
        let result = exec.run(&intent);
        assert!(result.is_err());
    }

    /// Tests executor rejects invalid URL protocols.
    #[test]
    fn executor_rejects_invalid_url_protocol() {
        let mut exec = GlobalExecutor::new().expect("failed to init executor");
        let intent = Intent::OpenUrl {
            url: "ftp://example.com".to_owned(),
        };
        let result = exec.run(&intent);
        assert!(result.is_err());
    }

    /// Tests panic hook registers idempotently without panicking.
    #[test]
    fn panic_hook_installs_idempotently() {
        install_panic_key_release_hook();
        install_panic_key_release_hook();
        assert!(PANIC_HOOK_INSTALLED.load(Ordering::SeqCst));
    }

    /// Tests allow-list application name validation accepts allowed apps.
    #[test]
    fn validate_app_name_accepts_allowed_apps() {
        assert_eq!(validate_app_name("chrome"), Ok(AllowedApp::Chrome));
        assert_eq!(validate_app_name("Chrome"), Ok(AllowedApp::Chrome));
        assert_eq!(validate_app_name("chrome.exe"), Ok(AllowedApp::Chrome));
        assert_eq!(validate_app_name(" notepad "), Ok(AllowedApp::Notepad));
        assert_eq!(validate_app_name("NOTEPAD.EXE"), Ok(AllowedApp::Notepad));
        assert_eq!(validate_app_name("explorer"), Ok(AllowedApp::Explorer));
        assert_eq!(validate_app_name("explorer.exe"), Ok(AllowedApp::Explorer));
    }

    /// Tests allow-list application name validation rejects unapproved apps.
    #[test]
    fn validate_app_name_rejects_unapproved_apps() {
        assert!(validate_app_name("cmd").is_err());
        assert!(validate_app_name("powershell").is_err());
        assert!(validate_app_name("calc").is_err());
        assert!(validate_app_name("firefox").is_err());
        assert!(validate_app_name("").is_err());
        assert!(validate_app_name("   ").is_err());
        assert!(validate_app_name("malicious_app").is_err());
    }

    /// Tests URL validation accepts valid http and https URLs.
    #[test]
    fn validate_url_accepts_http_and_https() {
        assert!(validate_url("http://example.com").is_ok());
        assert!(validate_url("https://example.com").is_ok());
        assert!(validate_url("https://example.com/register?foo=bar#section").is_ok());
        assert!(validate_url("http://localhost:8080/demo/form.html").is_ok());
    }

    /// Tests URL validation rejects non-http/https protocols.
    #[test]
    fn validate_url_rejects_disallowed_schemes() {
        assert!(validate_url("ftp://example.com").is_err());
        assert!(validate_url("file:///C:/Windows").is_err());
        assert!(validate_url("javascript:alert(1)").is_err());
        assert!(validate_url("mailto:test@example.com").is_err());
        assert!(validate_url("data:text/html,<h1>test</h1>").is_err());
        assert!(validate_url("").is_err());
        assert!(validate_url("   ").is_err());
        assert!(validate_url("not a url").is_err());
    }

    /// Tests that standard Chrome paths includes typical Windows locations.
    #[test]
    fn standard_chrome_paths_has_candidates() {
        let paths = standard_chrome_paths();
        assert!(!paths.is_empty());
        assert!(
            paths
                .iter()
                .any(|p| p.to_string_lossy().contains("chrome.exe"))
        );
    }

    /// Manual test launching Notepad.
    #[test]
    #[ignore = "manual launch test for notepad; run with: cargo test -p exec -- --ignored --nocapture"]
    fn manual_launch_notepad() {
        let res = open_app("notepad");
        assert!(res.is_ok(), "failed to launch notepad: {:?}", res);
    }

    /// Manual test launching Chrome with accessibility and URL.
    #[test]
    #[ignore = "manual launch test for Chrome; run with: cargo test -p exec -- --ignored --nocapture"]
    fn manual_launch_chrome_url() {
        let res = open_url("https://example.com");
        assert!(res.is_ok(), "failed to launch url: {:?}", res);
    }
}
