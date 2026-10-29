//! Reach status window.
//!
//! This is deliberately a small always-on-top HUD. It reports what the voice
//! loop is doing without ever taking keyboard focus from the application the
//! user is controlling.
//!
//! ## Interim input
//!
//! `audio` and `stt` are doc-only crates at the moment, so the app reads one
//! utterance per line from the console that launched it. The reader sends the
//! same `StatusEvent` channel the microphone path will use; only the input
//! source changes when audio and speech-to-text land.
//!
//! The window is configured with `focusable: false`, `alwaysOnTop: true`,
//! `transparent: true` and click-through enabled, so it can become the basis of
//! the P9 overlays without changing the window model.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::collections::VecDeque;
use std::io::{self, BufRead};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use reach_runtime::{
    CompositeExecutor, ErrorKind, OverlayEvent, Pipeline, PipelineModel, Rect, StatusEvent,
    run_worker,
};
use serde::Serialize;
use tauri::{Emitter, Manager, PhysicalPosition, WebviewWindow};

/// The model file used when `REACH_MODEL` is not set and the file exists.
///
/// TODO(config): read the path from the TOML configuration once that crate
/// exists, instead of probing the working directory.
const DEFAULT_MODEL_PATH: &str = "models/needle3.cact";

/// How long the worker pauses before executing a typed command.
///
/// This delay gives the user time to click the target window after typing a
/// command into the console and pressing Enter.
const TYPED_COMMAND_DELAY: Duration = Duration::from_secs(3);

/// The event name the status window listens for.
const STATUS_EVENT: &str = "reach-status";

/// The event name the overlay window listens for.
const OVERLAY_EVENT: &str = "reach-overlay";

/// How many recent status envelopes are kept for a late-loading front end.
const STATUS_HISTORY: usize = 8;

/// Starts the status window and its voice-loop threads.
///
/// # Errors
/// Prints the failure kind and exits with status 1 when Tauri cannot start.
fn main() {
    if let Err(error) = run() {
        eprintln!("reach-app: {error}");
        std::process::exit(1);
    }
}

/// Builds and runs the Tauri application.
///
/// # Returns
/// `Ok(())` after the Tauri event loop exits.
///
/// # Errors
/// Returns a message when the status or overlay window is missing, click-through cannot
/// be enabled, or Tauri fails to start.
fn run() -> Result<(), String> {
    let (line_sender, line_receiver) = mpsc::channel::<String>();
    let (status_sender, status_receiver) = mpsc::channel::<StatusEvent>();
    let (overlay_sender, overlay_receiver) = mpsc::channel::<OverlayEvent>();
    let history = Arc::new(Mutex::new(StatusHistory::new()));

    tauri::Builder::default()
        .manage(LatestStatus(Arc::clone(&history)))
        .invoke_handler(tauri::generate_handler![current_status])
        .setup(move |app| {
            let window = app
                .get_webview_window("status")
                .ok_or_else(|| "status window was not created".to_owned())?;
            window
                .set_ignore_cursor_events(true)
                .map_err(|error| format!("click-through failed: {error}"))?;
            place_bottom_right(&window)?;
            window
                .show()
                .map_err(|error| format!("show failed: {error}"))?;

            let overlay = app
                .get_webview_window("overlay")
                .ok_or_else(|| "overlay window was not created".to_owned())?;
            overlay
                .set_ignore_cursor_events(true)
                .map_err(|error| format!("overlay click-through failed: {error}"))?;
            place_overlay(&overlay)?;

            // The forwarders start before the worker so events reach the windows.
            spawn_status_forwarder(window, status_receiver, history);
            spawn_overlay_forwarder(overlay, overlay_receiver);
            spawn_worker(line_receiver, status_sender, overlay_sender);
            spawn_console_reader(line_sender);
            Ok(())
        })
        .run(tauri::generate_context!())
        .map_err(|error| format!("tauri event loop failed: {error}"))
}

/// Returns the recent status events so a late-loading front end can catch up.
///
/// Events emitted before `status.js` registers its listener would otherwise be
/// lost. The front end listens first, then calls this command and replays the
/// snapshot. Sequence numbers let it ignore any event it has already applied.
///
/// # Arguments
/// * `latest` - The shared history maintained by the status forwarder.
///
/// # Returns
/// The recent envelopes in emission order, oldest first. The list is empty
/// before the first event.
#[tauri::command]
fn current_status(latest: tauri::State<'_, LatestStatus>) -> Vec<WireEnvelope> {
    latest
        .0
        .lock()
        .map(|guard| guard.snapshot())
        .unwrap_or_default()
}

/// A shared, bounded history of status events.
struct LatestStatus(Arc<Mutex<StatusHistory>>);

/// A status event plus the sequence number the front end uses for ordering.
#[derive(Debug, Clone, Serialize)]
struct WireEnvelope {
    /// Increases by one for every event the forwarder emits.
    seq: u64,
    /// The status itself.
    status: WireStatus,
}

/// A bounded ring of recent envelopes and the next sequence number.
struct StatusHistory {
    /// The sequence number the next envelope will receive.
    next_seq: u64,
    /// Recent envelopes, oldest first.
    events: VecDeque<WireEnvelope>,
}

impl StatusHistory {
    /// Creates an empty history whose first envelope will be sequence zero.
    ///
    /// # Returns
    /// An empty [`StatusHistory`].
    fn new() -> Self {
        Self {
            next_seq: 0,
            events: VecDeque::with_capacity(STATUS_HISTORY),
        }
    }

    /// Assigns the next sequence number, records the event and returns the
    /// envelope to emit.
    ///
    /// # Arguments
    /// * `status` - The translated status event.
    ///
    /// # Returns
    /// The envelope carrying the new sequence number.
    fn push(&mut self, status: WireStatus) -> WireEnvelope {
        let envelope = WireEnvelope {
            seq: self.next_seq,
            status,
        };
        self.next_seq = self.next_seq.saturating_add(1);
        if self.events.len() == STATUS_HISTORY {
            self.events.pop_front();
        }
        self.events.push_back(envelope.clone());
        envelope
    }

    /// Returns a copy of the retained envelopes, oldest first.
    ///
    /// # Returns
    /// Up to [`STATUS_HISTORY`] recent envelopes.
    fn snapshot(&self) -> Vec<WireEnvelope> {
        self.events.iter().cloned().collect()
    }
}

/// A status event in the shape the status window expects.
///
/// This adapter exists so `reach-runtime` stays free of serde and so the
/// front-end contract is visible in one place. Error text is never included;
/// only [`ErrorKind::as_str`] travels across the boundary.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireStatus {
    /// The listening window is open.
    Listening,
    /// The loop is waiting for the wake word.
    Asleep,
    /// A transcript was heard.
    Heard {
        /// The text the user said.
        text: String,
    },
    /// An action finished.
    Action {
        /// A short description of the action.
        label: String,
    },
    /// A risky action is waiting for a yes or no.
    Confirming {
        /// A short description of the action.
        label: String,
    },
    /// The utterance was not understood.
    NotUnderstood,
    /// The loop stopped.
    Stopped,
    /// A failure occurred.
    Error {
        /// The failure category.
        kind: &'static str,
    },
}

impl From<StatusEvent> for WireStatus {
    /// Converts a runtime event into the front-end payload.
    ///
    /// # Arguments
    /// * `event` - The event emitted by the shared voice loop.
    ///
    /// # Returns
    /// The serializable wire form.
    fn from(event: StatusEvent) -> Self {
        match event {
            StatusEvent::Listening => Self::Listening,
            StatusEvent::Asleep => Self::Asleep,
            StatusEvent::Heard { text } => Self::Heard { text },
            StatusEvent::Action { label } => Self::Action { label },
            StatusEvent::Confirming { label } => Self::Confirming { label },
            StatusEvent::NotUnderstood => Self::NotUnderstood,
            StatusEvent::Stopped => Self::Stopped,
            StatusEvent::Error { kind } => Self::Error {
                kind: kind.as_str(),
            },
        }
    }
}

/// Builds the voice loop's model.
///
/// `REACH_MODEL` wins when set. Otherwise the default model is used only if it
/// exists, so a clean checkout runs the rule grammar offline instead of failing
/// at startup.
///
/// # Returns
/// The loaded model, or the disabled model when no file is configured.
///
/// # Errors
/// Returns [`ErrorKind::Model`] when an explicitly configured model cannot be
/// loaded. The worker reports the kind and stops; the error text is not shown.
fn load_model() -> Result<PipelineModel, ErrorKind> {
    let configured = std::env::var_os("REACH_MODEL").map(PathBuf::from);
    let path = configured.or_else(|| {
        let default = PathBuf::from(DEFAULT_MODEL_PATH);
        default.exists().then_some(default)
    });

    match path {
        Some(path) => PipelineModel::load(path).map_err(|_| ErrorKind::Model),
        None => Ok(PipelineModel::disabled()),
    }
}

/// Starts the worker thread that owns the shared pipeline.
///
/// # Arguments
/// * `lines` - Transcripts from the console until audio/STT land.
/// * `status` - Sink for status events.
/// * `overlay` - Sink for overlay visual events.
fn spawn_worker(
    lines: Receiver<String>,
    status: Sender<StatusEvent>,
    overlay: Sender<OverlayEvent>,
) {
    thread::spawn(move || {
        let model = match load_model() {
            Ok(model) => model,
            Err(kind) => {
                let _ = status.send(StatusEvent::Error { kind });
                let _ = status.send(StatusEvent::Stopped);
                return;
            }
        };
        let executor = match CompositeExecutor::new() {
            Ok(executor) => executor.with_overlay_sender(overlay),
            Err(_) => {
                let _ = status.send(StatusEvent::Error {
                    kind: ErrorKind::Internal,
                });
                let _ = status.send(StatusEvent::Stopped);
                return;
            }
        };

        let mut pipeline =
            Pipeline::new(model, executor, false).with_execution_delay(TYPED_COMMAND_DELAY);
        let _ = run_worker(&mut pipeline, &lines, &status);
    });
}

/// Reads typed utterances from the console.
///
/// This is the temporary input source, not the production path. It sends one
/// trimmed line per utterance to the same channel the microphone will use.
///
/// TODO(audio/stt): replace this reader with the microphone -> VAD ->
/// whisper-rs utterance channel once those crates are implemented. The worker
/// and the `StatusEvent` channel must not change.
///
/// # Arguments
/// * `lines` - Sender consumed by the worker's input channel.
fn spawn_console_reader(lines: Sender<String>) {
    thread::spawn(move || {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let Ok(line) = line else {
                break;
            };
            let utterance = line.trim();
            if utterance.is_empty() {
                continue;
            }
            if lines.send(utterance.to_owned()).is_err() {
                break;
            }
        }
    });
}

/// Forwards status envelopes to the window and keeps a snapshot for late loads.
///
/// # Arguments
/// * `window` - The status window.
/// * `statuses` - Events from the worker thread.
/// * `history` - Bounded history updated before each event is emitted.
fn spawn_status_forwarder(
    window: WebviewWindow,
    statuses: Receiver<StatusEvent>,
    history: Arc<Mutex<StatusHistory>>,
) {
    thread::spawn(move || {
        for status in statuses {
            let wire = WireStatus::from(status);
            let Ok(mut guard) = history.lock() else {
                // Without the sequence counter the front end cannot order
                // events, so stop instead of showing an unknown state.
                break;
            };
            let envelope = guard.push(wire);
            drop(guard);
            if window.emit(STATUS_EVENT, envelope).is_err() {
                break;
            }
        }
    });
}

/// Moves `window` to the bottom-right corner of the primary monitor.
///
/// # Arguments
/// * `window` - The status window, created hidden by `tauri.conf.json`.
///
/// # Returns
/// `Ok(())` after the window has been repositioned.
///
/// # Errors
/// Returns a message when the monitor, window size or position cannot be read.
fn place_bottom_right(window: &WebviewWindow) -> Result<(), String> {
    let monitor = window
        .primary_monitor()
        .map_err(|error| format!("monitor lookup failed: {error}"))?
        .ok_or_else(|| "no primary monitor".to_owned())?;
    let size = window
        .outer_size()
        .map_err(|error| format!("window size failed: {error}"))?;
    let monitor_size = monitor.size();
    let monitor_position = monitor.position();
    let margin = 24i32;
    let x = monitor_position.x + monitor_size.width as i32 - size.width as i32 - margin;
    let y = monitor_position.y + monitor_size.height as i32 - size.height as i32 - margin;
    window
        .set_position(PhysicalPosition::new(x, y))
        .map_err(|error| format!("window position failed: {error}"))
}

/// Sizes and positions the overlay window to cover the primary monitor.
///
/// # Arguments
/// * `overlay` - The overlay window to size.
///
/// # Returns
/// `Ok(())` on success.
///
/// # Errors
/// Returns an error message if monitor querying fails.
fn place_overlay(overlay: &WebviewWindow) -> Result<(), String> {
    if let Ok(Some(monitor)) = overlay.primary_monitor() {
        let size = monitor.size();
        let _ = overlay.set_size(*size);
        let _ = overlay.set_position(PhysicalPosition::new(0, 0));
    }
    Ok(())
}

/// Forwards overlay updates from the voice loop to the transparent overlay window.
///
/// # Arguments
/// * `window` - The overlay window.
/// * `events` - Channel receiver carrying overlay commands.
fn spawn_overlay_forwarder(window: WebviewWindow, events: Receiver<OverlayEvent>) {
    thread::spawn(move || {
        for event in events {
            match event {
                OverlayEvent::ShowNumbers(badges) => {
                    let wire_badges = badges
                        .into_iter()
                        .map(|b| WireBadge {
                            number: b.number,
                            rect: b.rect.into(),
                        })
                        .collect();
                    let payload = WireOverlayPayload::ShowNumbers {
                        badges: wire_badges,
                    };
                    let _ = window.emit(OVERLAY_EVENT, payload);
                    let _ = window.show();
                }
                OverlayEvent::HideNumbers => {
                    let _ = window.emit(OVERLAY_EVENT, WireOverlayPayload::HideNumbers);
                }
                OverlayEvent::ShowGrid {
                    bounds,
                    level,
                    cells,
                } => {
                    let wire_cells = cells
                        .into_iter()
                        .map(|(c, r)| WireCell {
                            cell: c,
                            rect: r.into(),
                        })
                        .collect();
                    let payload = WireOverlayPayload::ShowGrid {
                        bounds: bounds.into(),
                        level,
                        cells: wire_cells,
                    };
                    let _ = window.emit(OVERLAY_EVENT, payload);
                    let _ = window.show();
                }
                OverlayEvent::HideGrid => {
                    let _ = window.emit(OVERLAY_EVENT, WireOverlayPayload::HideGrid);
                }
                OverlayEvent::HideAll => {
                    let _ = window.emit(OVERLAY_EVENT, WireOverlayPayload::Clear);
                    let _ = window.hide();
                }
            }
        }
        let _ = window.hide();
    });
}

/// Wire representation of a screen rectangle for the overlay front end.
#[derive(Debug, Clone, Copy, Serialize)]
struct WireRect {
    /// Left horizontal coordinate.
    left: i32,
    /// Top vertical coordinate.
    top: i32,
    /// Right horizontal coordinate.
    right: i32,
    /// Bottom vertical coordinate.
    bottom: i32,
}

impl From<Rect> for WireRect {
    /// Converts a screen [`Rect`] into its serializable wire representation.
    ///
    /// # Arguments
    /// * `r` - The bounding rectangle to convert.
    ///
    /// # Returns
    /// An equivalent [`WireRect`].
    fn from(r: Rect) -> Self {
        Self {
            left: r.left,
            top: r.top,
            right: r.right,
            bottom: r.bottom,
        }
    }
}

/// Wire representation of a numbered badge for the overlay front end.
#[derive(Debug, Clone, Copy, Serialize)]
struct WireBadge {
    /// 1-based number to display.
    number: u32,
    /// Bounding rectangle of the target element.
    rect: WireRect,
}

/// Wire representation of a 3x3 grid cell for the overlay front end.
#[derive(Debug, Clone, Copy, Serialize)]
struct WireCell {
    /// 1-based cell index (1 to 9).
    cell: u32,
    /// Sub-rectangle for this cell.
    rect: WireRect,
}

/// Wire payload sent to the overlay window over the `reach-overlay` event.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireOverlayPayload {
    /// Display numbered badges on screen.
    ShowNumbers {
        /// Numbered badges to display.
        badges: Vec<WireBadge>,
    },
    /// Hide numbered badges.
    HideNumbers,
    /// Display or update the mouse grid.
    ShowGrid {
        /// Overall grid bounding box.
        bounds: WireRect,
        /// Current zoom level.
        level: u32,
        /// The 9 sub-cells.
        cells: Vec<WireCell>,
    },
    /// Hide the mouse grid.
    HideGrid,
    /// Clear all overlay visuals.
    Clear,
}
