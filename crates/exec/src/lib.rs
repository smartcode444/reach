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
use std::sync::atomic::{AtomicBool, Ordering};

/// Default number of scroll increments when unspecified.
pub const DEFAULT_SCROLL_STEPS: i32 = 3;

/// Guard that ensures a held modifier key is released when dropped or on panic.
pub struct ModifierGuard<'a> {
    /// Mutable reference to the Enigo input injector.
    enigo: &'a mut Enigo,
    /// The modifier key to release on drop.
    key: EnigoKey,
    /// Whether the guard is actively holding down the modifier.
    active: bool,
}

impl<'a> ModifierGuard<'a> {
    /// Presses `key` and arms the guard to release it on drop.
    ///
    /// # Arguments
    /// * `enigo` - Mutable reference to the Enigo input injector.
    /// * `key` - The modifier key to press.
    ///
    /// # Returns
    /// An armed `ModifierGuard` if key press succeeded.
    ///
    /// # Errors
    /// Returns error string if input injection failed.
    pub fn press(enigo: &'a mut Enigo, key: EnigoKey) -> Result<Self, String> {
        enigo
            .key(key, EnigoDirection::Press)
            .map_err(|e| format!("failed to press modifier key: {e}"))?;
        Ok(Self {
            enigo,
            key,
            active: true,
        })
    }
}

impl Drop for ModifierGuard<'_> {
    fn drop(&mut self) {
        if self.active {
            let _ = self.enigo.key(self.key, EnigoDirection::Release);
            self.active = false;
        }
    }
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

    /// Presses a key with a modifier held down, guaranteed to release the modifier.
    fn press_with_modifier(&mut self, modifier: EnigoKey, key: EnigoKey) -> Result<(), String> {
        let guard = ModifierGuard::press(&mut self.enigo, modifier)?;
        guard
            .enigo
            .key(key, EnigoDirection::Click)
            .map_err(|e| format!("failed to click key: {e}"))?;
        drop(guard);
        Ok(())
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
            Intent::Zoom { direction } => match direction {
                ZoomDir::In => {
                    self.press_with_modifier(EnigoKey::Control, EnigoKey::Unicode('+'))?;
                    Ok("zoomed in".to_owned())
                }
                ZoomDir::Out => {
                    self.press_with_modifier(EnigoKey::Control, EnigoKey::Unicode('-'))?;
                    Ok("zoomed out".to_owned())
                }
                ZoomDir::Reset => {
                    self.press_with_modifier(EnigoKey::Control, EnigoKey::Unicode('0'))?;
                    Ok("reset zoom".to_owned())
                }
            },
            Intent::OpenApp { name } => {
                // Never pass user text to cmd/powershell. Launch safely via open crate.
                open::that_detached(name)
                    .map_err(|e| format!("failed to launch application: {e}"))?;
                Ok(format!("opened app: {}", name))
            }
            Intent::OpenUrl { url } => {
                if !url.starts_with("http://") && !url.starts_with("https://") {
                    return Err("invalid URL protocol: only http and https are allowed".to_owned());
                }
                open::that_detached(url).map_err(|e| format!("failed to open web URL: {e}"))?;
                Ok(format!("opened URL: {}", url))
            }
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
}
