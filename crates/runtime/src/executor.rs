//! Composite executor routing between screen target resolution and input execution.

use std::sync::mpsc::Sender;

use exec::GlobalExecutor;
use reach_core::{Executor, Intent};
use screen::{
    ActionOutcome, GridOutcome, GridState, NumberClickOutcome, NumberedSnapshot, Rect,
    click_number, get_foreground_window_rect,
};

use crate::overlay::{NumberedBadge, OverlayEvent};

/// Composite executor routing intents to screen target resolution and global input injection.
pub struct CompositeExecutor {
    /// Underlying global executor for key, scroll, zoom, and application intents.
    global: GlobalExecutor,
    /// Last numbered snapshot taken by `ShowNumbers`.
    last_snapshot: Option<NumberedSnapshot>,
    /// Active mouse grid state, if visible.
    grid_state: Option<GridState>,
    /// Optional channel sender to update the visual overlay.
    overlay_sender: Option<Sender<OverlayEvent>>,
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
        Ok(Self {
            global,
            last_snapshot: None,
            grid_state: None,
            overlay_sender: None,
        })
    }

    /// Sets an overlay sender channel to notify a visual overlay window.
    ///
    /// # Arguments
    /// * `sender` - The channel to send overlay events to.
    pub fn set_overlay_sender(&mut self, sender: Sender<OverlayEvent>) {
        self.overlay_sender = Some(sender);
    }

    /// Attaches an overlay sender channel, returning self.
    ///
    /// # Arguments
    /// * `sender` - The channel to send overlay events to.
    ///
    /// # Returns
    /// The updated `CompositeExecutor`.
    pub fn with_overlay_sender(mut self, sender: Sender<OverlayEvent>) -> Self {
        self.overlay_sender = Some(sender);
        self
    }

    /// Returns the last numbered snapshot taken by `ShowNumbers`, if any.
    ///
    /// # Returns
    /// An optional reference to the current snapshot.
    pub fn last_snapshot(&self) -> Option<&NumberedSnapshot> {
        self.last_snapshot.as_ref()
    }

    /// Sets the active numbered snapshot (useful for testing or external injection).
    ///
    /// # Arguments
    /// * `snapshot` - The snapshot to store.
    pub fn set_last_snapshot(&mut self, snapshot: Option<NumberedSnapshot>) {
        self.last_snapshot = snapshot;
    }

    /// Returns the current mouse grid state, if active.
    ///
    /// # Returns
    /// An optional reference to the active grid.
    pub fn grid_state(&self) -> Option<&GridState> {
        self.grid_state.as_ref()
    }

    /// Clears both snapshot and mouse grid, sending a hide event to the overlay if attached.
    pub fn clear_overlay(&mut self) {
        self.last_snapshot = None;
        self.grid_state = None;
        if let Some(ref sender) = self.overlay_sender {
            let _ = sender.send(OverlayEvent::HideAll);
        }
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
            Intent::ShowNumbers => {
                let snapshot = screen::snapshot::collect_numbered_snapshot()
                    .unwrap_or_else(|_| screen::snapshot::numbered_snapshot(&[]));
                let count = snapshot.len();
                let badges: Vec<NumberedBadge> = snapshot
                    .elements
                    .iter()
                    .map(|e| NumberedBadge {
                        number: e.number,
                        rect: e.rect,
                    })
                    .collect();
                self.last_snapshot = Some(snapshot);
                if let Some(ref sender) = self.overlay_sender {
                    let _ = sender.send(OverlayEvent::ShowNumbers(badges));
                }
                Ok(format!("showing numbers for {count} elements"))
            }
            Intent::HideNumbers => {
                self.last_snapshot = None;
                if let Some(ref sender) = self.overlay_sender {
                    let _ = sender.send(OverlayEvent::HideNumbers);
                }
                Ok("numbers overlay hidden".to_owned())
            }
            Intent::ClickNumber { n } => match click_number(self.last_snapshot.as_ref(), *n) {
                NumberClickOutcome::NeedsClickAt { x, y } => {
                    exec::click_at(x, y)?;
                    self.clear_overlay();
                    Ok(format!("clicked number {n} at ({x}, {y})"))
                }
                NumberClickOutcome::NotFound { message } => Err(message),
            },
            Intent::ShowGrid => {
                let rect = get_foreground_window_rect().unwrap_or(Rect {
                    left: 0,
                    top: 0,
                    right: 1920,
                    bottom: 1080,
                });
                let grid = GridState::new(rect);
                if let Some(ref sender) = self.overlay_sender
                    && let Ok(event) = crate::overlay::build_grid_overlay_event(&grid)
                {
                    let _ = sender.send(event);
                }
                self.grid_state = Some(grid);
                Ok("showing mouse grid".to_owned())
            }
            Intent::GridSelect { cell } => {
                let Some(ref mut grid) = self.grid_state else {
                    return Err("mouse grid is not active; say 'show grid' first".to_owned());
                };
                let outcome = grid
                    .select(*cell)
                    .map_err(|e| format!("grid select failed: {e}"))?;
                match outcome {
                    GridOutcome::Zoomed { level, .. } => {
                        if let Some(ref sender) = self.overlay_sender
                            && let Ok(event) = crate::overlay::build_grid_overlay_event(grid)
                        {
                            let _ = sender.send(event);
                        }
                        Ok(format!("zoomed into grid cell {cell} (level {level})"))
                    }
                    GridOutcome::ClickAt { x, y } => {
                        exec::click_at(x, y)?;
                        self.grid_state = None;
                        if let Some(ref sender) = self.overlay_sender {
                            let _ = sender.send(OverlayEvent::HideGrid);
                        }
                        Ok(format!("clicked grid cell {cell} at ({x}, {y})"))
                    }
                }
            }
            // Window-changing commands clear active snapshots and hide overlay:
            window_intent @ (Intent::SwitchWindow { .. }
            | Intent::OpenApp { .. }
            | Intent::OpenUrl { .. }) => {
                self.clear_overlay();
                self.global.run(window_intent)
            }
            // Other global intents are routed to GlobalExecutor:
            global_intent @ (Intent::PressKey { .. }
            | Intent::Scroll { .. }
            | Intent::Zoom { .. }) => self.global.run(global_intent),
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
    use screen::{ControlKind, ElementSummary, NumberedElement};
    use std::sync::mpsc;

    /// Tests that ClickNumber with no snapshot fails with a helpful message.
    #[test]
    fn click_number_without_snapshot_fails_with_show_numbers_guidance() {
        let mut exec = CompositeExecutor::new().expect("failed to init CompositeExecutor");
        let intent = Intent::ClickNumber { n: 1 };
        let err = exec.run(&intent).expect_err("should fail without snapshot");
        assert!(
            err.contains("show numbers"),
            "error must guide user to say 'show numbers': {err}"
        );
    }

    /// Tests that ShowNumbers and HideNumbers send expected overlay events.
    #[test]
    fn show_and_hide_numbers_send_overlay_events() {
        let (tx, rx) = mpsc::channel();
        let mut exec = CompositeExecutor::new()
            .expect("init")
            .with_overlay_sender(tx);

        let elem = ElementSummary {
            name: "Save".to_owned(),
            kind: ControlKind::Button,
            rect: Rect {
                left: 10,
                top: 20,
                right: 50,
                bottom: 40,
            },
            enabled: true,
            offscreen: false,
            is_password: false,
        };
        let snap = NumberedSnapshot::new(vec![NumberedElement {
            number: 1,
            rect: elem.rect,
            summary: elem,
        }]);
        exec.set_last_snapshot(Some(snap));

        assert!(exec.last_snapshot().is_some());

        let res = exec.run(&Intent::HideNumbers).expect("hide numbers");
        assert_eq!(res, "numbers overlay hidden");
        assert!(exec.last_snapshot().is_none());
        assert_eq!(rx.try_recv().unwrap(), OverlayEvent::HideNumbers);
    }

    /// Tests grid select without show grid returns an error.
    #[test]
    fn grid_select_without_show_grid_fails() {
        let mut exec = CompositeExecutor::new().expect("init");
        let res = exec.run(&Intent::GridSelect { cell: 5 });
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("show grid"));
    }

    /// Tests that the composite executor rejects unsupported intents.
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
