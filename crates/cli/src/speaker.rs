//! Spoken feedback engine and phrases for Reach.
//!
//! Spoken feedback uses a closed list of [`Phrase`] variants with short,
//! fixed wording. Arbitrary model output and user transcripts are never spoken.
//!
//! Output is driven by the [`Speaker`] trait with two concrete implementations:
//! - [`TtsSpeaker`]: Uses the platform text-to-speech engine via the `tts` crate.
//! - [`NullSpeaker`]: A silent recording speaker for tests and `--no-voice` mode.

use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The maximum duration speech is allowed to run before timing out.
pub const SPEECH_TIMEOUT: Duration = Duration::from_secs(8);

/// How often the completion of speech is polled.
pub const SPEECH_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Typed errors that can occur during speech synthesis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpeakerError {
    /// The speech synthesizer backend failed to initialize.
    Initialization(String),
    /// Speech output timed out before completing.
    Timeout,
    /// An error occurred during speech synthesis or status querying.
    Backend(String),
}

impl fmt::Display for SpeakerError {
    /// Formats the error for user display.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Initialization(err) => write!(f, "speech initialization failed: {err}"),
            Self::Timeout => write!(f, "speech output timed out after 8 seconds"),
            Self::Backend(err) => write!(f, "speech backend error: {err}"),
        }
    }
}

impl std::error::Error for SpeakerError {}

/// A short, fixed label describing an action requiring user confirmation.
///
/// Element names must never be included in confirmation labels to protect privacy
/// and prevent accidental speech feedback leaks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionLabel {
    /// Generic or unspecified computer action.
    Action,
    /// Clicking an element.
    Click,
    /// Submitting a form.
    Submit,
    /// Deleting an item.
    Delete,
    /// Sending a message or data.
    Send,
    /// Making a payment or purchase.
    Pay,
    /// Custom fixed label.
    Other(String),
}

impl ActionLabel {
    /// Returns the wording for the action label.
    ///
    /// # Returns
    /// The lowercase slice representation of the action label.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Action => "action",
            Self::Click => "click",
            Self::Submit => "submit",
            Self::Delete => "delete",
            Self::Send => "send",
            Self::Pay => "payment",
            Self::Other(label) => label.as_str(),
        }
    }
}

impl fmt::Display for ActionLabel {
    /// Formats the action label into text.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl From<&str> for ActionLabel {
    /// Converts a string slice into an [`ActionLabel`].
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "click" => Self::Click,
            "submit" => Self::Submit,
            "delete" | "remove" => Self::Delete,
            "send" => Self::Send,
            "pay" | "buy" | "purchase" => Self::Pay,
            "action" => Self::Action,
            other => Self::Other(other.to_owned()),
        }
    }
}

impl From<&reach_core::Intent> for ActionLabel {
    /// Maps a known intent to a confirmation action label.
    fn from(intent: &reach_core::Intent) -> Self {
        match intent {
            reach_core::Intent::ClickElement { target } => {
                let lower = target.to_lowercase();
                if lower.contains("submit") {
                    Self::Submit
                } else if lower.contains("delete") || lower.contains("remove") {
                    Self::Delete
                } else if lower.contains("send") {
                    Self::Send
                } else if lower.contains("pay")
                    || lower.contains("buy")
                    || lower.contains("purchase")
                {
                    Self::Pay
                } else {
                    Self::Click
                }
            }
            _ => Self::Action,
        }
    }
}

/// Closed list of spoken feedback phrases.
///
/// Spoken feedback uses short, fixed wording. Raw user transcripts and arbitrary
/// free text are never spoken.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Phrase {
    /// Acknowledgment that Reach is awake and listening.
    Listening,
    /// Acknowledgment that a command was successfully executed.
    Done,
    /// Request to rephrase or repeat an unrecognized or low-confidence utterance.
    DidntCatch,
    /// Prompt asking for spoken confirmation of a risky action.
    Confirming(ActionLabel),
    /// Acknowledgment that a pending action was cancelled or denied.
    Cancelled,
    /// Notification that the requested target element could not be found.
    NotFound,
    /// Disambiguation prompt when multiple candidate elements match.
    WhichOne(Vec<String>),
}

impl Phrase {
    /// Formats the phrase into its short, fixed spoken representation.
    ///
    /// Element names are spoken only for [`Phrase::WhichOne`].
    ///
    /// # Returns
    /// The exact text to speak.
    pub fn to_speech(&self) -> String {
        match self {
            Self::Listening => "Listening".to_owned(),
            Self::Done => "Done".to_owned(),
            Self::DidntCatch => "I didn't catch that".to_owned(),
            Self::Confirming(action) => format!("Confirm {}?", action.as_str()),
            Self::Cancelled => "Cancelled".to_owned(),
            Self::NotFound => "Not found".to_owned(),
            Self::WhichOne(names) => {
                if names.is_empty() {
                    "Which one?".to_owned()
                } else {
                    format!("Which one? {}", names.join(", "))
                }
            }
        }
    }
}

/// A mechanism for speaking structured phrases to the user.
pub trait Speaker: Send {
    /// Speaks `phrase`, blocking until speech finishes or times out.
    ///
    /// # Arguments
    /// * `phrase` - The closed phrase to speak.
    ///
    /// # Errors
    /// Returns [`SpeakerError`] if speech cannot start, backend errors occur,
    /// or speech output exceeds the 8-second timeout.
    fn speak(&mut self, phrase: Phrase) -> Result<(), SpeakerError>;
}

/// A silent speaker that records phrases for testing and silent mode.
#[derive(Debug, Default, Clone)]
pub struct NullSpeaker {
    /// Thread-safe log of phrases spoken.
    phrases: Arc<Mutex<Vec<Phrase>>>,
}

impl NullSpeaker {
    /// Creates a new empty [`NullSpeaker`].
    ///
    /// # Returns
    /// An initialized [`NullSpeaker`] with no phrases recorded.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a copy of the recorded phrases in order.
    ///
    /// # Returns
    /// A vector of phrases spoken through this speaker.
    pub fn phrases(&self) -> Vec<Phrase> {
        self.phrases.lock().map(|p| p.clone()).unwrap_or_default()
    }

    /// Clears the list of recorded phrases.
    pub fn clear(&self) {
        if let Ok(mut lock) = self.phrases.lock() {
            lock.clear();
        }
    }
}

impl Speaker for NullSpeaker {
    /// Records `phrase` silently.
    ///
    /// # Arguments
    /// * `phrase` - The phrase to record.
    ///
    /// # Errors
    /// Never returns an error.
    fn speak(&mut self, phrase: Phrase) -> Result<(), SpeakerError> {
        if let Ok(mut lock) = self.phrases.lock() {
            lock.push(phrase);
        }
        Ok(())
    }
}

/// Text-to-speech speaker using the platform audio backend.
pub struct TtsSpeaker {
    /// Underlying TTS engine handle.
    tts: tts::Tts,
}

impl TtsSpeaker {
    /// Initializes a new [`TtsSpeaker`] using the default system speech synthesizer.
    ///
    /// # Returns
    /// An initialized [`TtsSpeaker`] ready to output speech.
    ///
    /// # Errors
    /// Returns [`SpeakerError::Initialization`] if the TTS subsystem fails to load.
    pub fn new() -> Result<Self, SpeakerError> {
        let tts = tts::Tts::default().map_err(|e| SpeakerError::Initialization(e.to_string()))?;
        Ok(Self { tts })
    }
}

impl Speaker for TtsSpeaker {
    /// Speaks `phrase`, blocking until speech completes or 8 seconds elapse.
    ///
    /// Polling is conducted every 50 ms. If the timeout is reached, speech is
    /// stopped and [`SpeakerError::Timeout`] is returned.
    ///
    /// # Arguments
    /// * `phrase` - The phrase to synthesize and speak.
    ///
    /// # Returns
    /// `Ok(())` once speech playback has finished.
    ///
    /// # Errors
    /// Returns [`SpeakerError::Timeout`] on timeout or [`SpeakerError::Backend`] on error.
    fn speak(&mut self, phrase: Phrase) -> Result<(), SpeakerError> {
        let text = phrase.to_speech();
        self.tts
            .speak(&text, false)
            .map_err(|e| SpeakerError::Backend(e.to_string()))?;

        let start = Instant::now();
        std::thread::sleep(SPEECH_POLL_INTERVAL);

        loop {
            if start.elapsed() >= SPEECH_TIMEOUT {
                let _ = self.tts.stop();
                return Err(SpeakerError::Timeout);
            }

            match self.tts.is_speaking() {
                Ok(true) => std::thread::sleep(SPEECH_POLL_INTERVAL),
                Ok(false) => return Ok(()),
                Err(err) => {
                    let _ = self.tts.stop();
                    return Err(SpeakerError::Backend(err.to_string()));
                }
            }
        }
    }
}

/// Extracts a [`Phrase`] from a single line of printed feedback text, if any.
///
/// Converts fixed console announcements into their corresponding typed phrase
/// variant for speech output. Captions, latency measurements, and unrecognised
/// debug text return `None`.
///
/// # Arguments
/// * `line` - A single line of feedback text emitted by Reach.
///
/// # Returns
/// `Some(phrase)` if the text matches a known feedback event, or `None`.
pub fn phrase_from_line(line: &str) -> Option<Phrase> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }

    if trimmed == "listening" {
        return Some(Phrase::Listening);
    }

    if trimmed == "cancelled" {
        return Some(Phrase::Cancelled);
    }

    if trimmed.contains("I didn't catch that") {
        return Some(Phrase::DidntCatch);
    }

    if let Some(rest) = trimmed.strip_prefix("confirm:") {
        if let Ok(intent) = serde_json::from_str::<reach_core::Intent>(rest.trim()) {
            return Some(Phrase::Confirming(ActionLabel::from(&intent)));
        }
        return Some(Phrase::Confirming(ActionLabel::Action));
    }

    if trimmed.contains("element not found") {
        return Some(Phrase::NotFound);
    }

    if let Some(rest) = trimmed.strip_prefix("ambiguous choice among:") {
        let names = rest
            .split(',')
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
            .collect();
        return Some(Phrase::WhichOne(names));
    }

    // Success actions that complete an operation
    if trimmed.starts_with("scrolled")
        || trimmed.starts_with("clicked")
        || trimmed.starts_with("typed")
        || trimmed.starts_with("zoomed")
        || trimmed.starts_with("reset zoom")
        || trimmed.starts_with("pressed")
        || trimmed.starts_with("opened")
        || trimmed.starts_with("executed")
    {
        return Some(Phrase::Done);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests exact phrase to spoken text conversions.
    #[test]
    fn phrase_to_speech_conversions() {
        assert_eq!(Phrase::Listening.to_speech(), "Listening");
        assert_eq!(Phrase::Done.to_speech(), "Done");
        assert_eq!(Phrase::DidntCatch.to_speech(), "I didn't catch that");
        assert_eq!(
            Phrase::Confirming(ActionLabel::Submit).to_speech(),
            "Confirm submit?"
        );
        assert_eq!(
            Phrase::Confirming(ActionLabel::Delete).to_speech(),
            "Confirm delete?"
        );
        assert_eq!(
            Phrase::Confirming(ActionLabel::Action).to_speech(),
            "Confirm action?"
        );
        assert_eq!(Phrase::Cancelled.to_speech(), "Cancelled");
        assert_eq!(Phrase::NotFound.to_speech(), "Not found");
        assert_eq!(
            Phrase::WhichOne(vec!["Submit".to_owned(), "Cancel".to_owned()]).to_speech(),
            "Which one? Submit, Cancel"
        );
        assert_eq!(Phrase::WhichOne(vec![]).to_speech(), "Which one?");
    }

    /// Tests ActionLabel conversions from strings and intents.
    #[test]
    fn action_label_conversions() {
        assert_eq!(ActionLabel::from("submit"), ActionLabel::Submit);
        assert_eq!(ActionLabel::from("delete"), ActionLabel::Delete);
        assert_eq!(ActionLabel::from("remove"), ActionLabel::Delete);
        assert_eq!(ActionLabel::from("send"), ActionLabel::Send);
        assert_eq!(ActionLabel::from("pay"), ActionLabel::Pay);
        assert_eq!(ActionLabel::from("click"), ActionLabel::Click);
        assert_eq!(ActionLabel::from("action"), ActionLabel::Action);
        assert_eq!(
            ActionLabel::from("custom"),
            ActionLabel::Other("custom".to_owned())
        );

        let intent = reach_core::Intent::ClickElement {
            target: "Submit Form".to_owned(),
        };
        assert_eq!(ActionLabel::from(&intent), ActionLabel::Submit);

        let generic = reach_core::Intent::Scroll {
            direction: reach_core::Direction::Down,
            amount: None,
        };
        assert_eq!(ActionLabel::from(&generic), ActionLabel::Action);
    }

    /// Tests NullSpeaker records phrases in order and can be cleared.
    #[test]
    fn null_speaker_records_phrases() {
        let mut speaker = NullSpeaker::new();
        assert!(speaker.phrases().is_empty());

        speaker.speak(Phrase::Listening).unwrap();
        speaker.speak(Phrase::Done).unwrap();
        assert_eq!(speaker.phrases(), vec![Phrase::Listening, Phrase::Done]);

        speaker.clear();
        assert!(speaker.phrases().is_empty());
    }

    /// Tests that phrase_from_line detects all supported output lines.
    #[test]
    fn phrase_from_line_parsing() {
        assert_eq!(phrase_from_line("listening"), Some(Phrase::Listening));
        assert_eq!(phrase_from_line("cancelled"), Some(Phrase::Cancelled));
        assert_eq!(
            phrase_from_line("I didn't catch that, could you say it another way?"),
            Some(Phrase::DidntCatch)
        );
        assert_eq!(
            phrase_from_line("confirm: {\"action\":\"click_element\",\"target\":\"submit\"}"),
            Some(Phrase::Confirming(ActionLabel::Submit))
        );
        assert_eq!(
            phrase_from_line("execution failed: element not found: search"),
            Some(Phrase::NotFound)
        );
        assert_eq!(
            phrase_from_line("ambiguous choice among: Name, First Name"),
            Some(Phrase::WhichOne(vec![
                "Name".to_owned(),
                "First Name".to_owned()
            ]))
        );
        assert_eq!(
            phrase_from_line("scrolled down 3 lines"),
            Some(Phrase::Done)
        );
        assert_eq!(
            phrase_from_line("clicked element 'Button'"),
            Some(Phrase::Done)
        );

        // Lines that must NOT produce speech
        assert_eq!(phrase_from_line("ignored"), None);
        assert_eq!(phrase_from_line("heard: reach scroll down"), None);
        assert_eq!(phrase_from_line("latency: 20 ms"), None);
        assert_eq!(phrase_from_line("Run this action? (yes/no) "), None);
        assert_eq!(phrase_from_line("rejected: LowConfidence"), None);
    }

    /// Manual verification test that speaks every phrase once using the real TTS backend.
    #[test]
    #[ignore = "touches audio output; run manually to verify TTS pronunciation"]
    fn manual_speak_every_phrase() {
        let mut speaker = TtsSpeaker::new().expect("TTS engine must initialize");
        let phrases = vec![
            Phrase::Listening,
            Phrase::Done,
            Phrase::DidntCatch,
            Phrase::Confirming(ActionLabel::Submit),
            Phrase::Confirming(ActionLabel::Delete),
            Phrase::Cancelled,
            Phrase::NotFound,
            Phrase::WhichOne(vec!["Full Name".to_owned(), "First Name".to_owned()]),
        ];
        for phrase in phrases {
            println!("Speaking phrase: {:?}", phrase);
            speaker.speak(phrase).expect("speech must succeed");
        }
    }
}
