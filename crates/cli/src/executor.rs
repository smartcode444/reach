//! Composite executor routing between screen target resolution and input execution.

use exec::GlobalExecutor;
use reach_core::{Executor, Intent};
use screen::ActionOutcome;

/// Composite executor routing intents to screen target resolution and global input injection.
pub struct CompositeExecutor {
    /// Underlying global executor for key, scroll, zoom, and application intents.
    global: GlobalExecutor,
}

impl CompositeExecutor {
    /// Creates a new `CompositeExecutor` with initialized platform injectors.
    ///
    /// # Returns
    /// An initialized `CompositeExecutor`.
    ///
    /// # Errors
    /// Returns error message if input injection cannot be initialized.
    pub fn new() -> Result<Self, String> {
        let global = GlobalExecutor::new()?;
        Ok(Self { global })
    }
}

impl Default for CompositeExecutor {
    fn default() -> Self {
        Self::new()
            .unwrap_or_else(|e| panic!("failed to initialize default CompositeExecutor: {e}"))
    }
}

impl Executor for CompositeExecutor {
    fn run(&mut self, intent: &Intent) -> Result<String, String> {
        match intent {
            Intent::ClickElement { target } => {
                let outcome = screen::click_element(target)
                    .map_err(|e| format!("click element failed: {e}"))?;
                match outcome {
                    ActionOutcome::Done(method) => {
                        Ok(format!("clicked element '{target}' via {method:?}"))
                    }
                    ActionOutcome::NeedsChoice(names) => {
                        Ok(format!("ambiguous choice among: {}", names.join(", ")))
                    }
                    ActionOutcome::NotFound => Err(format!("element not found: {target}")),
                    ActionOutcome::NeedsClickAt { x, y } => {
                        exec::click_at(x, y)?;
                        Ok(format!("clicked at ({x}, {y}) for '{target}'"))
                    }
                    ActionOutcome::NeedsTypeText { text } => {
                        exec::type_str(&text)?;
                        Ok(format!("typed text for '{target}'"))
                    }
                }
            }
            Intent::FocusField { target } => {
                let outcome =
                    screen::focus_field(target).map_err(|e| format!("focus field failed: {e}"))?;
                match outcome {
                    ActionOutcome::Done(method) => {
                        Ok(format!("focused field '{target}' via {method:?}"))
                    }
                    ActionOutcome::NeedsChoice(names) => {
                        Ok(format!("ambiguous choice among: {}", names.join(", ")))
                    }
                    ActionOutcome::NotFound => Err(format!("field not found: {target}")),
                    ActionOutcome::NeedsClickAt { x, y } => {
                        exec::click_at(x, y)?;
                        Ok(format!("focused at ({x}, {y}) for '{target}'"))
                    }
                    ActionOutcome::NeedsTypeText { text } => {
                        exec::type_str(&text)?;
                        Ok(format!("typed text for '{target}'"))
                    }
                }
            }
            Intent::TypeText { text, field } => {
                let outcome = screen::type_text(text, field.as_deref())
                    .map_err(|e| format!("type text failed: {e}"))?;
                match outcome {
                    ActionOutcome::Done(method) => Ok(format!("typed text via {method:?}")),
                    ActionOutcome::NeedsChoice(names) => Ok(format!(
                        "ambiguous field choice among: {}",
                        names.join(", ")
                    )),
                    ActionOutcome::NotFound => Err(format!("target field not found: {:?}", field)),
                    ActionOutcome::NeedsClickAt { x, y } => {
                        exec::click_at(x, y)?;
                        exec::type_str(text)?;
                        Ok(format!("clicked at ({x}, {y}) and typed text"))
                    }
                    ActionOutcome::NeedsTypeText { text: to_type } => {
                        exec::type_str(&to_type)?;
                        Ok("typed text directly".to_owned())
                    }
                }
            }
            // Global intents are routed to GlobalExecutor:
            global_intent @ (Intent::PressKey { .. }
            | Intent::Scroll { .. }
            | Intent::Zoom { .. }
            | Intent::OpenApp { .. }
            | Intent::OpenUrl { .. }) => self.global.run(global_intent),
            unrecognized => Err(format!(
                "intent not recognised by executor: {:?}",
                unrecognized
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests that the composite executor rejects intents not supported in this phase.
    #[test]
    fn composite_executor_rejects_unsupported_intents() {
        let mut exec = CompositeExecutor::new().expect("failed to init CompositeExecutor");
        let intent = Intent::Cancel;
        assert!(exec.run(&intent).is_err());
    }

    /// Tests that the composite executor rejects invalid URL schemes without touching desktop.
    #[test]
    fn composite_executor_rejects_invalid_url() {
        let mut exec = CompositeExecutor::new().expect("failed to init CompositeExecutor");
        let intent = Intent::OpenUrl {
            url: "ftp://example.com".to_owned(),
        };
        assert!(exec.run(&intent).is_err());
    }

    /// Tests routing of global intents to the underlying global executor on real desktop.
    #[test]
    #[ignore = "touches real desktop input injection (blocked by UIPI in headless); run manually"]
    fn composite_executor_routes_global_scroll() {
        let mut exec = CompositeExecutor::new().expect("failed to init CompositeExecutor");
        let intent = Intent::Scroll {
            direction: reach_core::Direction::Down,
            amount: Some(1),
        };
        let _ = exec.run(&intent);
    }
}
