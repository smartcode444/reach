//! Wake-word gate and listening window.
//!
//! The gate runs before the rule grammar and the Needle model. It decides
//! whether an utterance should be handled at all, whether it was only the wake
//! word, or whether it already contains a command. The wake word counts only as
//! the first word: "I can't reach the top" is not a wake event, while "Reach,
//! scroll down" is.
//!
//! The gate is intentionally lexical and cheap. It cannot tell the noun or verb
//! "reach" from the wake word, so "Reach for the stars" is accepted as
//! `Command("for the stars")`. That is a known false-trigger limitation:
//! downstream grounding must reject a command whose target does not exist on
//! screen. Keeping the gate small also keeps it available when the model is not.
//!
//! The listening window is a separate state machine with an injected clock.
//! After a bare wake word the caller opens it; the next utterance is a command
//! even without the wake word, and the window then closes. "stop" and "sleep"
//! close it early. No method calls [`Instant::now`] itself, so tests can drive
//! time deterministically.

use std::time::{Duration, Instant};

/// The result of evaluating one transcript against the wake word.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateResult {
    /// The utterance does not start with the wake word and is not a command.
    Ignored,
    /// The utterance contained only the wake word.
    WakeOnly,
    /// The wake word was followed by a command, preserved as spoken.
    Command(String),
}

/// Configuration for the wake word and the listening window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WakeConfig {
    /// The primary wake word, compared case-insensitively.
    pub wake_word: String,
    /// Additional words treated exactly like the wake word.
    pub aliases: Vec<String>,
    /// How long a bare wake word keeps the listening window open.
    pub window: Duration,
}

impl Default for WakeConfig {
    /// Returns the default wake word, aliases and five-second window.
    fn default() -> Self {
        Self {
            wake_word: "reach".to_owned(),
            aliases: vec!["rich".to_owned()],
            window: Duration::from_secs(5),
        }
    }
}

/// Matches the configured wake word at the start of a transcript.
#[derive(Debug, Clone)]
pub struct WakeGate {
    /// The configuration this gate was built from.
    config: WakeConfig,
    /// Lowercased wake words and aliases, with empty entries removed.
    wake_words: Vec<String>,
}

impl WakeGate {
    /// Builds a gate from `config`.
    ///
    /// # Arguments
    /// * `config` - The wake word, aliases and listening-window length.
    pub fn new(config: WakeConfig) -> Self {
        let wake_words = normalized_wake_words(&config);
        Self { config, wake_words }
    }

    /// Returns the configuration this gate uses.
    pub fn config(&self) -> &WakeConfig {
        &self.config
    }

    /// Classifies one transcript.
    ///
    /// # Arguments
    /// * `transcript` - The recognised text, with its original casing and
    ///   punctuation.
    ///
    /// # Returns
    /// [`GateResult::Ignored`] when the first word is not the wake word,
    /// [`GateResult::WakeOnly`] when only the wake word was spoken, and
    /// [`GateResult::Command`] with the remaining text otherwise.
    pub fn evaluate(&self, transcript: &str) -> GateResult {
        let Some(rest) = self.match_wake(transcript) else {
            return GateResult::Ignored;
        };
        let command = clean_command(rest);
        if command.is_empty() {
            GateResult::WakeOnly
        } else {
            GateResult::Command(command.to_owned())
        }
    }

    /// Returns `"stop"` or `"sleep"` when the transcript is exactly one of
    /// those words, optionally after the wake word.
    ///
    /// # Arguments
    /// * `transcript` - The recognised text.
    ///
    /// # Returns
    /// The canonical stop word, or `None` when the utterance is not an exact
    /// stop or sleep command.
    pub fn stop_word(&self, transcript: &str) -> Option<&'static str> {
        let candidate = match self.match_wake(transcript) {
            Some(rest) => clean_command(rest),
            None => prepare_text(transcript).trim(),
        };
        stop_word_exact(candidate)
    }

    /// Returns the text after the wake word, or `None` when the first word is
    /// not one of the configured wake words.
    ///
    /// # Arguments
    /// * `transcript` - The recognised text.
    fn match_wake<'a>(&self, transcript: &'a str) -> Option<&'a str> {
        let text = prepare_text(transcript);
        let (first_word, rest) = split_first_word(text)?;
        let normalized = normalize_word(first_word);
        self.wake_words
            .iter()
            .any(|wake| wake == &normalized)
            .then_some(rest)
    }
}

impl Default for WakeGate {
    /// Builds a gate with [`WakeConfig::default`].
    fn default() -> Self {
        Self::new(WakeConfig::default())
    }
}

/// Classifies one transcript with a temporary [`WakeGate`].
///
/// # Arguments
/// * `transcript` - The recognised text.
/// * `config` - The wake word, aliases and listening-window length.
///
/// # Returns
/// The same result as [`WakeGate::evaluate`].
pub fn gate(transcript: &str, config: &WakeConfig) -> GateResult {
    WakeGate::new(config.clone()).evaluate(transcript)
}

/// A one-command listening window with an injected clock.
#[derive(Debug, Clone)]
pub struct ListenWindow {
    /// How long the window stays open after [`Self::open`].
    window: Duration,
    /// When the window was opened, or `None` when it is closed.
    opened_at: Option<Instant>,
}

impl ListenWindow {
    /// Creates a closed window with the given lifetime.
    ///
    /// # Arguments
    /// * `window` - How long the window stays open after a wake word.
    pub fn new(window: Duration) -> Self {
        Self {
            window,
            opened_at: None,
        }
    }

    /// Returns the configured window lifetime.
    pub fn window(&self) -> Duration {
        self.window
    }

    /// Opens the window at `now`, replacing any previous open time.
    ///
    /// # Arguments
    /// * `now` - The current time, supplied by the caller.
    pub fn open(&mut self, now: Instant) {
        self.opened_at = Some(now);
    }

    /// Reports whether the window is open at `now`.
    ///
    /// # Arguments
    /// * `now` - The current time, supplied by the caller.
    ///
    /// # Returns
    /// `true` before `open_time + window`, `false` at or after the deadline.
    pub fn is_open(&self, now: Instant) -> bool {
        self.opened_at
            .is_some_and(|opened| now.saturating_duration_since(opened) < self.window)
    }

    /// Consumes the window if it is open, closing it.
    ///
    /// # Arguments
    /// * `now` - The current time, supplied by the caller.
    ///
    /// # Returns
    /// `true` when a command consumed an open window, `false` otherwise.
    pub fn consume(&mut self, now: Instant) -> bool {
        let open = self.is_open(now);
        // Clear an expired timestamp as well, so the state does not linger.
        self.opened_at = None;
        open
    }

    /// Closes the window when `transcript` is an exact stop or sleep phrase.
    ///
    /// # Arguments
    /// * `gate` - The wake-word matcher used to recognise an optional wake
    ///   word before "stop" or "sleep".
    /// * `transcript` - The recognised text.
    /// * `now` - The current time, supplied by the caller.
    ///
    /// # Returns
    /// The canonical stop word when an open window was closed, or `None` when
    /// the window was closed already or the utterance is not an exact stop
    /// phrase.
    pub fn close_on_stop(
        &mut self,
        gate: &WakeGate,
        transcript: &str,
        now: Instant,
    ) -> Option<&'static str> {
        if !self.is_open(now) {
            return None;
        }
        let word = gate.stop_word(transcript)?;
        self.close();
        Some(word)
    }

    /// Closes the window without waiting for a command.
    pub fn close(&mut self) {
        self.opened_at = None;
    }
}

/// The wake gate and its listening window combined.
///
/// This is the small orchestration layer a caller can use directly: a bare
/// wake word opens the window, and the next non-empty utterance is returned as
/// a command and consumes the window.
#[derive(Debug, Clone)]
pub struct WakeSession {
    /// The lexical wake-word matcher.
    gate: WakeGate,
    /// The one-command window opened by a bare wake word.
    window: ListenWindow,
}

impl WakeSession {
    /// Creates a session from `config`.
    ///
    /// # Arguments
    /// * `config` - The wake word, aliases and listening-window length.
    pub fn new(config: WakeConfig) -> Self {
        let window = ListenWindow::new(config.window);
        let gate = WakeGate::new(config);
        Self { gate, window }
    }

    /// Classifies `transcript` at `now`, updating the listening window.
    ///
    /// # Arguments
    /// * `transcript` - The recognised text.
    /// * `now` - The current time, supplied by the caller.
    ///
    /// # Returns
    /// [`GateResult::Ignored`] when nothing should run, [`GateResult::WakeOnly`]
    /// when the window was opened or refreshed, or [`GateResult::Command`] with
    /// the text to act on.
    pub fn evaluate(&mut self, transcript: &str, now: Instant) -> GateResult {
        if self.window.is_open(now) {
            let cleaned = clean_window_utterance(transcript);
            if cleaned.is_empty() {
                return GateResult::Ignored;
            }
            if let Some(word) = self.window.close_on_stop(&self.gate, transcript, now) {
                return GateResult::Command(word.to_owned());
            }
            return match self.gate.evaluate(transcript) {
                GateResult::Command(command) => {
                    self.window.close();
                    GateResult::Command(command)
                }
                GateResult::Ignored => {
                    self.window.close();
                    GateResult::Command(cleaned.to_owned())
                }
                // A repeated wake word refreshes the window instead of
                // consuming it.
                GateResult::WakeOnly => {
                    self.window.open(now);
                    GateResult::WakeOnly
                }
            };
        }

        let result = self.gate.evaluate(transcript);
        if result == GateResult::WakeOnly {
            self.window.open(now);
        }
        result
    }

    /// Reports whether the listening window is open at `now`.
    ///
    /// # Arguments
    /// * `now` - The current time, supplied by the caller.
    pub fn is_open(&self, now: Instant) -> bool {
        self.window.is_open(now)
    }
}

impl Default for WakeSession {
    /// Creates a session with [`WakeConfig::default`].
    fn default() -> Self {
        Self::new(WakeConfig::default())
    }
}

/// Returns the normalized wake words and aliases for `config`.
///
/// Empty entries are dropped so a blank alias cannot match every utterance,
/// and duplicates are removed so the same word is not checked twice.
///
/// # Arguments
/// * `config` - The wake word and aliases to normalize.
fn normalized_wake_words(config: &WakeConfig) -> Vec<String> {
    let mut words = Vec::with_capacity(1 + config.aliases.len());
    for word in std::iter::once(&config.wake_word).chain(config.aliases.iter()) {
        let normalized = normalize_word(word);
        if !normalized.is_empty() && !words.contains(&normalized) {
            words.push(normalized);
        }
    }
    words
}

/// Lowercases a word and removes punctuation attached to its edges.
///
/// The middle of the word is left intact, so a possessive such as "reach's"
/// does not become the wake word "reach".
///
/// # Arguments
/// * `word` - The word or configured wake word to normalize.
fn normalize_word(word: &str) -> String {
    word.trim_matches(|character: char| !character.is_alphanumeric())
        .to_lowercase()
}

/// Returns the transcript with leading noise tags and punctuation removed.
///
/// # Arguments
/// * `transcript` - The recognised text.
fn prepare_text(transcript: &str) -> &str {
    let text = strip_noise_tags(transcript);
    text.trim_start_matches(|character: char| !character.is_alphanumeric())
}

/// Removes leading `[...]` noise tags and the whitespace after them.
///
/// Whisper sometimes emits tags such as `[BLANK_AUDIO]` before the words. A tag
/// without a closing bracket is left alone so a stray `[` cannot swallow the
/// rest of the utterance.
///
/// # Arguments
/// * `text` - The text to clean.
fn strip_noise_tags(mut text: &str) -> &str {
    text = text.trim_start();
    while let Some(rest) = text.strip_prefix('[') {
        let Some(close) = rest.find(']') else {
            break;
        };
        text = rest[close + 1..].trim_start();
    }
    text
}

/// Splits the first word from the rest of the text.
///
/// # Arguments
/// * `text` - Text whose first word has already been cleaned.
///
/// # Returns
/// The first word and the remainder, or `None` when the text does not start
/// with a word.
fn split_first_word(text: &str) -> Option<(&str, &str)> {
    let mut end = 0;
    for (index, character) in text.char_indices() {
        if is_word_character(character) {
            end = index + character.len_utf8();
        } else {
            break;
        }
    }
    if end == 0 {
        return None;
    }
    Some((&text[..end], &text[end..]))
}

/// Reports whether `character` can appear inside the first word.
///
/// Apostrophes and hyphens stay inside the word so "Reach's" and
/// "Reach-scroll" are not accepted as the wake word.
///
/// # Arguments
/// * `character` - The character to test.
fn is_word_character(character: char) -> bool {
    character.is_alphanumeric()
        || matches!(character, '\'' | '\u{2018}' | '\u{2019}' | '\u{02BC}' | '-')
}

/// Removes the punctuation and whitespace between the wake word and command.
///
/// # Arguments
/// * `rest` - The text immediately after the first word.
fn clean_command(rest: &str) -> &str {
    let rest = rest.trim_start_matches(is_wake_trailing_punctuation);
    let rest = rest.trim_start();
    let rest = strip_one_command_punctuation(rest);
    rest.trim()
}

/// Reports whether `character` is punctuation attached to the wake word.
///
/// # Arguments
/// * `character` - The character to test.
fn is_wake_trailing_punctuation(character: char) -> bool {
    matches!(
        character,
        ',' | '.'
            | '!'
            | '?'
            | ';'
            | ':'
            | '…'
            | '"'
            | '\''
            | '\u{2018}'
            | '\u{2019}'
            | '\u{201C}'
            | '\u{201D}'
            | ')'
            | ']'
            | '}'
            | '»'
    )
}

/// Strips at most one leading comma, period or colon from a command.
///
/// # Arguments
/// * `text` - The command candidate.
fn strip_one_command_punctuation(text: &str) -> &str {
    text.strip_prefix(',')
        .or_else(|| text.strip_prefix('.'))
        .or_else(|| text.strip_prefix(':'))
        .unwrap_or(text)
}

/// Recognises an exact "stop" or "sleep" utterance.
///
/// # Arguments
/// * `text` - The candidate text, already stripped of the optional wake word.
///
/// # Returns
/// The canonical word, or `None` when the text is not exactly one of them.
fn stop_word_exact(text: &str) -> Option<&'static str> {
    match normalize_word(text).as_str() {
        "stop" => Some("stop"),
        "sleep" => Some("sleep"),
        _ => None,
    }
}

/// Returns the transcript with noise tags removed and outer whitespace trimmed.
///
/// # Arguments
/// * `transcript` - The recognised text.
fn clean_window_utterance(transcript: &str) -> &str {
    strip_noise_tags(transcript).trim()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// A wake word followed by a command yields the command; without the wake
    /// word the same words are ignored.
    #[test]
    fn wake_word_then_command() {
        let gate = WakeGate::default();
        assert_eq!(
            gate.evaluate("Reach scroll down"),
            GateResult::Command("scroll down".to_owned())
        );
        assert_eq!(gate.evaluate("scroll down"), GateResult::Ignored);
    }

    /// The wake word only counts as the first word of the utterance.
    #[test]
    fn ignores_reach_later_in_the_sentence() {
        let gate = WakeGate::default();
        assert_eq!(gate.evaluate("I can't reach the top"), GateResult::Ignored);
        assert_eq!(gate.evaluate("let's reach out to him"), GateResult::Ignored);
        assert_eq!(
            gate.evaluate("The reach of the network"),
            GateResult::Ignored
        );
    }

    /// "Reach for the stars" is a known false trigger. The gate cannot tell the
    /// verb from the wake word, so downstream grounding must reject the command.
    #[test]
    fn documents_the_for_the_stars_false_trigger() {
        let gate = WakeGate::default();
        assert_eq!(
            gate.evaluate("Reach for the stars"),
            GateResult::Command("for the stars".to_owned())
        );
    }

    /// A bare wake word is distinct from a command, and trailing punctuation is
    /// stripped.
    #[test]
    fn wake_only_and_punctuation() {
        let gate = WakeGate::default();
        assert_eq!(gate.evaluate("Reach."), GateResult::WakeOnly);
        assert_eq!(gate.evaluate("Reach!"), GateResult::WakeOnly);
        assert_eq!(gate.evaluate("Reach,"), GateResult::WakeOnly);
        assert_eq!(
            gate.evaluate("  reach,  Scroll Down "),
            GateResult::Command("Scroll Down".to_owned())
        );
        assert_eq!(
            gate.evaluate("REACH click Submit"),
            GateResult::Command("click Submit".to_owned())
        );
    }

    /// The default alias works, but "teach" and "each" only work when the user
    /// adds them explicitly.
    #[test]
    fn aliases_are_explicit() {
        let gate = WakeGate::default();
        assert_eq!(
            gate.evaluate("Rich scroll down"),
            GateResult::Command("scroll down".to_owned())
        );
        assert_eq!(gate.evaluate("rich."), GateResult::WakeOnly);
        assert_eq!(gate.evaluate("teach scroll down"), GateResult::Ignored);
        assert_eq!(gate.evaluate("each scroll down"), GateResult::Ignored);

        let configured = WakeGate::new(WakeConfig {
            aliases: vec!["teach".to_owned(), "each".to_owned()],
            ..WakeConfig::default()
        });
        assert_eq!(
            configured.evaluate("teach scroll down"),
            GateResult::Command("scroll down".to_owned())
        );
        assert_eq!(
            configured.evaluate("each scroll down"),
            GateResult::Command("scroll down".to_owned())
        );
    }

    /// Empty and whitespace-only transcripts are ignored.
    #[test]
    fn ignores_empty_transcripts() {
        let gate = WakeGate::default();
        assert_eq!(gate.evaluate(""), GateResult::Ignored);
        assert_eq!(gate.evaluate("   \t\n "), GateResult::Ignored);
    }

    /// A leading Whisper noise tag is ignored.
    #[test]
    fn ignores_leading_noise_tags() {
        let gate = WakeGate::default();
        assert_eq!(
            gate.evaluate("[BLANK_AUDIO] Reach scroll down"),
            GateResult::Command("scroll down".to_owned())
        );
        assert_eq!(gate.evaluate(" [noise]  Reach."), GateResult::WakeOnly);
    }

    /// Unicode quotes and apostrophes do not break the first-word check.
    #[test]
    fn handles_unicode_quotes_and_apostrophes() {
        let gate = WakeGate::default();
        assert_eq!(
            gate.evaluate("“Reach,” scroll down"),
            GateResult::Command("scroll down".to_owned())
        );
        assert_eq!(
            gate.evaluate("’Reach scroll down"),
            GateResult::Command("scroll down".to_owned())
        );
        assert_eq!(gate.evaluate("I can’t reach the top"), GateResult::Ignored);
        assert_eq!(gate.evaluate("Reach’s scroll down"), GateResult::Ignored);
    }

    /// The command keeps its original casing and inner punctuation.
    #[test]
    fn preserves_command_casing_and_inner_punctuation() {
        let gate = WakeGate::default();
        assert_eq!(
            gate.evaluate("Reach, type John Okafor"),
            GateResult::Command("type John Okafor".to_owned())
        );
        assert_eq!(
            gate.evaluate("Reach, type hello, world"),
            GateResult::Command("type hello, world".to_owned())
        );
        assert_eq!(
            gate.evaluate("Reach  scroll  down"),
            GateResult::Command("scroll  down".to_owned())
        );
    }

    /// The documented defaults are what the app ships with.
    #[test]
    fn default_configuration() {
        let config = WakeConfig::default();
        assert_eq!(config.wake_word, "reach");
        assert_eq!(config.aliases, vec!["rich".to_owned()]);
        assert_eq!(config.window, Duration::from_secs(5));
    }

    /// Stop and sleep are exact whole utterances, optionally after the wake word.
    #[test]
    fn recognises_stop_and_sleep() {
        let gate = WakeGate::default();
        assert_eq!(gate.stop_word("stop"), Some("stop"));
        assert_eq!(gate.stop_word("Stop."), Some("stop"));
        assert_eq!(gate.stop_word("Reach stop"), Some("stop"));
        assert_eq!(gate.stop_word("Reach, sleep."), Some("sleep"));
        assert_eq!(gate.stop_word("stop now"), None);
        assert_eq!(gate.stop_word("Reach"), None);
    }

    /// The window stays open until its configured deadline.
    #[test]
    fn window_expires_at_the_deadline() {
        let start = Instant::now();
        let mut window = ListenWindow::new(Duration::from_secs(5));
        window.open(start);
        assert!(window.is_open(start + Duration::from_millis(4900)));
        assert!(!window.is_open(start + Duration::from_secs(5)));
        assert!(!window.is_open(start + Duration::from_millis(5100)));
    }

    /// One command consumes the window, and an expired window cannot be
    /// consumed.
    #[test]
    fn one_command_consumes_the_window() {
        let start = Instant::now();
        let mut window = ListenWindow::new(Duration::from_secs(5));
        window.open(start);
        assert!(window.consume(start + Duration::from_secs(1)));
        assert!(!window.is_open(start + Duration::from_secs(1)));
        assert!(!window.consume(start + Duration::from_secs(2)));

        window.open(start);
        assert!(!window.consume(start + Duration::from_millis(5100)));
    }

    /// The window itself closes on an exact stop or sleep phrase, with or
    /// without the wake word.
    #[test]
    fn window_closes_on_stop_phrase() {
        let start = Instant::now();
        let gate = WakeGate::default();
        let mut window = ListenWindow::new(Duration::from_secs(5));

        window.open(start);
        assert_eq!(
            window.close_on_stop(&gate, "stop", start + Duration::from_secs(1)),
            Some("stop")
        );
        assert!(!window.is_open(start + Duration::from_secs(1)));

        window.open(start);
        assert_eq!(
            window.close_on_stop(&gate, "Reach sleep.", start + Duration::from_secs(2)),
            Some("sleep")
        );
        assert!(!window.is_open(start + Duration::from_secs(2)));

        window.open(start);
        assert_eq!(
            window.close_on_stop(&gate, "stop now", start + Duration::from_secs(3)),
            None
        );
        assert!(window.is_open(start + Duration::from_secs(3)));
    }

    /// While the window is open, one command needs no wake word and then closes
    /// the window.
    #[test]
    fn session_accepts_one_command_without_the_wake_word() {
        let start = Instant::now();
        let mut session = WakeSession::new(WakeConfig::default());
        assert_eq!(session.evaluate("Reach", start), GateResult::WakeOnly);
        assert!(session.is_open(start + Duration::from_secs(1)));
        assert_eq!(
            session.evaluate("scroll down", start + Duration::from_secs(1)),
            GateResult::Command("scroll down".to_owned())
        );
        assert!(!session.is_open(start + Duration::from_secs(1)));
        assert_eq!(
            session.evaluate("scroll up", start + Duration::from_secs(2)),
            GateResult::Ignored
        );
    }

    /// Stop and sleep close an open window early.
    #[test]
    fn session_stop_and_sleep_close_the_window() {
        let start = Instant::now();
        let mut session = WakeSession::new(WakeConfig::default());
        assert_eq!(session.evaluate("Reach", start), GateResult::WakeOnly);
        assert_eq!(
            session.evaluate("stop", start + Duration::from_secs(1)),
            GateResult::Command("stop".to_owned())
        );
        assert!(!session.is_open(start + Duration::from_secs(1)));

        assert_eq!(
            session.evaluate("Reach", start + Duration::from_secs(2)),
            GateResult::WakeOnly
        );
        assert_eq!(
            session.evaluate("sleep", start + Duration::from_secs(3)),
            GateResult::Command("sleep".to_owned())
        );
        assert!(!session.is_open(start + Duration::from_secs(3)));

        assert_eq!(
            session.evaluate("Reach", start + Duration::from_secs(4)),
            GateResult::WakeOnly
        );
        assert_eq!(
            session.evaluate("Reach stop.", start + Duration::from_secs(5)),
            GateResult::Command("stop".to_owned())
        );
        assert!(!session.is_open(start + Duration::from_secs(5)));
    }

    /// A repeated wake word inside an open window is stripped before the
    /// command is returned.
    #[test]
    fn session_strips_a_repeated_wake_word() {
        let start = Instant::now();
        let mut session = WakeSession::new(WakeConfig::default());
        assert_eq!(session.evaluate("Reach", start), GateResult::WakeOnly);
        assert_eq!(
            session.evaluate("Reach scroll down", start + Duration::from_secs(1)),
            GateResult::Command("scroll down".to_owned())
        );
        assert!(!session.is_open(start + Duration::from_secs(1)));
    }

    /// The free function mirrors [`WakeGate::evaluate`].
    #[test]
    fn free_gate_function_matches() {
        assert_eq!(
            gate("Reach scroll down", &WakeConfig::default()),
            GateResult::Command("scroll down".to_owned())
        );
        assert_eq!(
            gate("scroll down", &WakeConfig::default()),
            GateResult::Ignored
        );
    }
}
