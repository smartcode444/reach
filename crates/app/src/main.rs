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
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::collections::VecDeque;
use std::io::{self, BufRead};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use reach_runtime::{
    CompositeExecutor, ConfigOverrides, ErrorKind, InputSource, OverlayEvent, PausingExecutor,
    PhrasesConfig, Pipeline, PipelineModel, ReachConfig, Rect, StatusEvent, VoiceInput,
    run_worker_source,
};
use serde::Serialize;
use tauri::{Emitter, Manager, PhysicalPosition, WebviewWindow};

/// Command line options accepted by reach-app.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct AppArgs {
    /// Select typed input from stdin instead of microphone.
    text: bool,
    /// Path to Whisper GGML model container.
    stt_model: Option<String>,
    /// Path to config/reach.toml.
    config_path: Option<String>,
    /// Whether to force rules-only (offline).
    no_model: bool,
    /// Path to Needle model container.
    model_path: Option<String>,
    /// Whether to explicitly enable Needle model.
    model: bool,
    /// Print route decisions instead of executing.
    dry_run: bool,
}

/// Parses command line arguments for reach-app.
///
/// # Arguments
/// * `args` - Raw command-line arguments (including argv[0]).
///
/// # Returns
/// The parsed [`AppArgs`].
///
/// # Errors
/// Returns an error message for unknown flags or missing arguments.
fn parse_app_args(args: &[String]) -> Result<AppArgs, String> {
    let mut parsed = AppArgs::default();
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--text" => parsed.text = true,
            "--no-model" => parsed.no_model = true,
            "--model" => {
                parsed.model = true;
                if let Some(next) = args.get(index + 1).filter(|next| {
                    !next.starts_with("--")
                        && (next.ends_with(".cact")
                            || next.ends_with(".bin")
                            || next.contains('/')
                            || next.contains('\\'))
                }) {
                    parsed.model_path = Some(next.clone());
                    index += 1;
                }
            }
            "--dry-run" => parsed.dry_run = true,
            "--stt-model" => {
                index += 1;
                let path = args.get(index).ok_or("missing path after --stt-model")?;
                parsed.stt_model = Some(path.clone());
            }
            "--config" => {
                index += 1;
                let path = args.get(index).ok_or("missing path after --config")?;
                parsed.config_path = Some(path.clone());
            }
            unknown => return Err(format!("unknown option: {unknown}")),
        }
        index += 1;
    }
    Ok(parsed)
}

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
    let raw_args: Vec<String> = std::env::args().collect();
    let args = parse_app_args(&raw_args)?;

    let config_path = args.config_path.as_deref().unwrap_or("config/reach.toml");
    let mut config = ReachConfig::load_or_default(config_path)?;
    config.apply_overrides(&ConfigOverrides {
        model: args.model,
        no_model: args.no_model,
        no_voice: false,
        model_path: args.model_path.clone(),
        stt_model: args.stt_model.clone(),
        start_threshold: None,
        min_speech_ms: None,
    });

    let phrases = PhrasesConfig::load_or_default("config/phrases.toml").unwrap_or_default();

    let (status_sender, status_receiver) = mpsc::channel::<StatusEvent>();
    let (overlay_sender, overlay_receiver) = mpsc::channel::<OverlayEvent>();
    let history = Arc::new(Mutex::new(StatusHistory::new()));

    let app_args = args;
    let app_config = config;

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
            spawn_worker(status_sender, overlay_sender, app_args, app_config, phrases);
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

/// Builds the pipeline's intent model based on configuration and overrides.
///
/// # Arguments
/// * `config` - Loaded reach configuration.
///
/// # Returns
/// A [`PipelineModel`] instance.
///
/// # Errors
/// Returns [`ErrorKind::Model`] if model file is specified but cannot be loaded.
fn load_model(config: &ReachConfig) -> Result<PipelineModel, ErrorKind> {
    if config.no_model {
        return Ok(PipelineModel::disabled());
    }
    PipelineModel::load(&config.needle_model).map_err(|_| ErrorKind::Model)
}

/// Starts the worker thread that owns the shared pipeline and input source.
///
/// # Arguments
/// * `status` - Sink for status events.
/// * `overlay` - Sink for overlay visual events.
/// * `args` - Application command-line arguments.
/// * `config` - Reach configuration settings.
/// * `phrases` - Custom phrase rewrites.
fn spawn_worker(
    status: Sender<StatusEvent>,
    overlay: Sender<OverlayEvent>,
    args: AppArgs,
    config: ReachConfig,
    phrases: PhrasesConfig,
) {
    thread::spawn(move || {
        let model = match load_model(&config) {
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

        let wake_config = config.to_wake_config();

        if args.text {
            let (line_sender, line_receiver) = mpsc::channel::<String>();
            spawn_console_reader(line_sender);

            let mut pipeline =
                Pipeline::with_wake_config(model, executor, args.dry_run, wake_config)
                    .with_phrases(phrases)
                    .with_execution_delay(TYPED_COMMAND_DELAY);

            let mut input = InputSource::Text(line_receiver);
            let _ = run_worker_source(&mut pipeline, &mut input, &status);
        } else {
            let (voice, control) = match VoiceInput::create_default(
                &config.stt_model,
                config.whisper_threads,
                config.endpoint.to_endpoint_config(),
            ) {
                Ok(pair) => pair,
                Err(error) => {
                    eprintln!("reach-app: voice capture failed: {error}");
                    let _ = status.send(StatusEvent::Error {
                        kind: ErrorKind::Internal,
                    });
                    let _ = status.send(StatusEvent::Stopped);
                    return;
                }
            };

            let pausing_executor = PausingExecutor::new(executor, Box::new(control));

            let mut pipeline =
                Pipeline::with_wake_config(model, pausing_executor, args.dry_run, wake_config)
                    .with_phrases(phrases);

            let mut input = InputSource::Voice(Box::new(voice));
            let _ = run_worker_source(&mut pipeline, &mut input, &status);
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests that default arguments set expected flags.
    #[test]
    fn parse_default_app_args() {
        let args = vec!["reach-app".to_owned()];
        let parsed = parse_app_args(&args).expect("valid args");
        assert!(!parsed.text);
        assert!(!parsed.no_model);
        assert!(!parsed.model);
        assert!(!parsed.dry_run);
        assert_eq!(parsed.stt_model, None);
        assert_eq!(parsed.config_path, None);
    }

    /// Tests that text input flag is recognized.
    #[test]
    fn parse_text_flag() {
        let args = vec!["reach-app".to_owned(), "--text".to_owned()];
        let parsed = parse_app_args(&args).expect("valid args");
        assert!(parsed.text);
    }

    /// Tests that explicit config and model paths are parsed.
    #[test]
    fn parse_explicit_paths() {
        let args = vec![
            "reach-app".to_owned(),
            "--config".to_owned(),
            "custom.toml".to_owned(),
            "--stt-model".to_owned(),
            "whisper.bin".to_owned(),
            "--model".to_owned(),
            "models/needle.cact".to_owned(),
            "--dry-run".to_owned(),
        ];
        let parsed = parse_app_args(&args).expect("valid args");
        assert_eq!(parsed.config_path.as_deref(), Some("custom.toml"));
        assert_eq!(parsed.stt_model.as_deref(), Some("whisper.bin"));
        assert_eq!(parsed.model_path.as_deref(), Some("models/needle.cact"));
        assert!(parsed.model);
        assert!(parsed.dry_run);
    }

    /// Tests that unknown options return an error.
    #[test]
    fn parse_unknown_option_fails() {
        let args = vec!["reach-app".to_owned(), "--invalid".to_owned()];
        assert!(parse_app_args(&args).is_err());
    }
}
