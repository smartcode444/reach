//! Configuration loading and phrase rewriting for Reach.
//!
//! Loads `config\reach.toml` for settings and `config\phrases.toml` for custom
//! spoken phrase mappings. If files are missing, documented defaults are used.
//! If files exist but are invalid, clear errors naming the key and file are returned.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

use audio::EndpointConfig;
use serde::{Deserialize, Serialize};

/// Default primary wake word.
pub const DEFAULT_WAKE_WORD: &str = "reach";
/// Default wake word aliases.
pub const DEFAULT_ALIASES: &[&str] = &["rich"];
/// Default listening window in seconds.
pub const DEFAULT_WINDOW_SECONDS: u32 = 15;
/// Default Whisper model container path.
pub const DEFAULT_STT_MODEL: &str = "models/ggml-tiny.en.bin";
/// Default Whisper decoding threads (0 = auto).
pub const DEFAULT_WHISPER_THREADS: usize = 0;
/// Default Needle model container path.
pub const DEFAULT_NEEDLE_MODEL: &str = "models/needle3.cact";
/// Default Needle transformer layer depth.
pub const DEFAULT_MODEL_LAYERS: u32 = 20;
/// Default rules-only flag (Needle off by default).
pub const DEFAULT_NO_MODEL: bool = true;
/// Default voice feedback flag.
pub const DEFAULT_NO_VOICE: bool = false;

/// Returns the default primary wake word.
fn default_wake_word() -> String {
    DEFAULT_WAKE_WORD.to_owned()
}

/// Returns the default wake word aliases.
fn default_aliases() -> Vec<String> {
    DEFAULT_ALIASES.iter().map(|&s| s.to_owned()).collect()
}

/// Returns the default listening window in seconds.
fn default_window_seconds() -> u32 {
    DEFAULT_WINDOW_SECONDS
}

/// Returns the default speech-to-text model path.
fn default_stt_model() -> String {
    DEFAULT_STT_MODEL.to_owned()
}

/// Returns the default Whisper thread count.
fn default_whisper_threads() -> usize {
    DEFAULT_WHISPER_THREADS
}

/// Returns the default Needle model path.
fn default_needle_model() -> String {
    DEFAULT_NEEDLE_MODEL.to_owned()
}

/// Returns the default Needle layer depth.
fn default_model_layers() -> u32 {
    DEFAULT_MODEL_LAYERS
}

/// Returns the default rules-only flag.
fn default_no_model() -> bool {
    DEFAULT_NO_MODEL
}

/// Returns the default voice feedback disabling flag.
fn default_no_voice() -> bool {
    DEFAULT_NO_VOICE
}

/// Default endpoint start threshold RMS.
pub const DEFAULT_START_THRESHOLD: f32 = 0.03;
/// Default endpoint end threshold RMS.
pub const DEFAULT_END_THRESHOLD: f32 = 0.008;
/// Default endpoint minimum speech duration in milliseconds.
pub const DEFAULT_MIN_SPEECH_MS: u32 = 400;
/// Default endpoint trailing silence duration in milliseconds.
pub const DEFAULT_END_SILENCE_MS: u32 = 700;
/// Default endpoint pre-roll audio duration in milliseconds.
pub const DEFAULT_PRE_ROLL_MS: u32 = 200;
/// Default endpoint maximum utterance duration in milliseconds.
pub const DEFAULT_MAX_UTTERANCE_MS: u32 = 6000;

/// Returns default start threshold.
fn default_start_threshold() -> f32 {
    DEFAULT_START_THRESHOLD
}

/// Returns default end threshold.
fn default_end_threshold() -> f32 {
    DEFAULT_END_THRESHOLD
}

/// Returns default min speech duration.
fn default_min_speech_ms() -> u32 {
    DEFAULT_MIN_SPEECH_MS
}

/// Returns default end silence duration.
fn default_end_silence_ms() -> u32 {
    DEFAULT_END_SILENCE_MS
}

/// Returns default pre-roll duration.
fn default_pre_roll_ms() -> u32 {
    DEFAULT_PRE_ROLL_MS
}

/// Returns default max utterance duration.
fn default_max_utterance_ms() -> u32 {
    DEFAULT_MAX_UTTERANCE_MS
}

/// Settings in the `[endpoint]` section of `reach.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EndpointSection {
    /// RMS energy threshold to trigger voice detection from silence.
    #[serde(default = "default_start_threshold")]
    pub start_threshold: f32,
    /// RMS energy threshold below which speech transitions towards silence.
    #[serde(default = "default_end_threshold")]
    pub end_threshold: f32,
    /// Minimum voiced audio needed to keep an utterance in milliseconds.
    #[serde(default = "default_min_speech_ms")]
    pub min_speech_ms: u32,
    /// Continuous silence that ends an utterance in milliseconds.
    #[serde(default = "default_end_silence_ms")]
    pub end_silence_ms: u32,
    /// Audio kept before voiced onset in milliseconds.
    #[serde(default = "default_pre_roll_ms")]
    pub pre_roll_ms: u32,
    /// Maximum length of one utterance in milliseconds.
    #[serde(default = "default_max_utterance_ms")]
    pub max_utterance_ms: u32,
}

impl Default for EndpointSection {
    /// Returns default endpoint settings.
    fn default() -> Self {
        Self {
            start_threshold: DEFAULT_START_THRESHOLD,
            end_threshold: DEFAULT_END_THRESHOLD,
            min_speech_ms: DEFAULT_MIN_SPEECH_MS,
            end_silence_ms: DEFAULT_END_SILENCE_MS,
            pre_roll_ms: DEFAULT_PRE_ROLL_MS,
            max_utterance_ms: DEFAULT_MAX_UTTERANCE_MS,
        }
    }
}

impl EndpointSection {
    /// Builds an [`audio::EndpointConfig`] from these settings.
    ///
    /// # Returns
    /// An initialized [`audio::EndpointConfig`].
    pub fn to_endpoint_config(&self) -> EndpointConfig {
        EndpointConfig {
            start_threshold_rms: self.start_threshold,
            end_threshold_rms: self.end_threshold,
            min_speech_ms: self.min_speech_ms,
            end_silence_ms: self.end_silence_ms,
            pre_roll_ms: self.pre_roll_ms,
            max_utterance_ms: self.max_utterance_ms,
            ..EndpointConfig::default()
        }
    }
}

/// Top-level configuration for Reach, loaded from `config\reach.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReachConfig {
    /// Primary wake word.
    #[serde(default = "default_wake_word")]
    pub wake_word: String,
    /// Alternative wake word aliases.
    #[serde(default = "default_aliases")]
    pub aliases: Vec<String>,
    /// Duration of listening window in seconds.
    #[serde(default = "default_window_seconds")]
    pub window_seconds: u32,
    /// Path to Whisper GGML model.
    #[serde(default = "default_stt_model")]
    pub stt_model: String,
    /// Decoding threads for Whisper (0 = auto).
    #[serde(default = "default_whisper_threads")]
    pub whisper_threads: usize,
    /// Path to Needle model container.
    #[serde(default = "default_needle_model")]
    pub needle_model: String,
    /// Transformer layers for Needle (1 to 20).
    #[serde(default = "default_model_layers")]
    pub model_layers: u32,
    /// Rules-only mode without Needle inference.
    #[serde(default = "default_no_model")]
    pub no_model: bool,
    /// Disable voice output.
    #[serde(default = "default_no_voice")]
    pub no_voice: bool,
    /// Audio endpointing settings.
    #[serde(default)]
    pub endpoint: EndpointSection,
}

impl Default for ReachConfig {
    /// Returns default Reach configuration.
    fn default() -> Self {
        Self {
            wake_word: default_wake_word(),
            aliases: default_aliases(),
            window_seconds: default_window_seconds(),
            stt_model: default_stt_model(),
            whisper_threads: default_whisper_threads(),
            needle_model: default_needle_model(),
            model_layers: default_model_layers(),
            no_model: default_no_model(),
            no_voice: default_no_voice(),
            endpoint: EndpointSection::default(),
        }
    }
}

impl ReachConfig {
    /// Applies command-line argument overrides onto this configuration.
    ///
    /// Command-line flags take precedence over configuration file values.
    ///
    /// # Arguments
    /// * `args` - Parsed arguments from the command line.
    pub fn apply_cli_overrides(&mut self, args: &crate::listen::ListenArgs) {
        if args.model {
            self.no_model = false;
        }
        if args.no_model {
            self.no_model = true;
        }
        if args.no_voice {
            self.no_voice = true;
        }
        if let Some(ref path) = args.model_path {
            self.needle_model = path.clone();
        }
        if let Some(ref path) = args.stt_model {
            self.stt_model = path.clone();
        }
        if let Some(start) = args.start_threshold {
            self.endpoint.start_threshold = start;
        }
        if let Some(min_speech) = args.min_speech_ms {
            self.endpoint.min_speech_ms = min_speech;
        }
    }

    /// Loads the configuration from `path`, or returns documented defaults if the file is missing.
    ///
    /// If the file is missing, defaults are used and no file is created. If the file exists
    /// but is invalid, returns an error naming the file, key, and reason.
    ///
    /// # Arguments
    /// * `path` - Path to `reach.toml`.
    ///
    /// # Returns
    /// The loaded [`ReachConfig`] or default settings.
    ///
    /// # Errors
    /// Returns an error string if reading or parsing the file fails.
    pub fn load_or_default(path: impl AsRef<Path>) -> Result<Self, String> {
        let path_ref = path.as_ref();
        if !path_ref.exists() {
            return Ok(Self::default());
        }
        let content = std::fs::read_to_string(path_ref).map_err(|err| {
            format!(
                "failed to read configuration file '{}': {err}",
                path_ref.display()
            )
        })?;
        Self::parse(&content, path_ref)
    }

    /// Parses and validates a configuration string from `content`.
    ///
    /// Checks that all keys are known, values fall within permissible ranges,
    /// and audio endpointing relationships hold.
    ///
    /// # Arguments
    /// * `content` - TOML formatted configuration string.
    /// * `path` - Path of the file being parsed, used for error diagnostics.
    ///
    /// # Returns
    /// The validated [`ReachConfig`].
    ///
    /// # Errors
    /// Returns an error message naming the file and invalid key if parsing or validation fails.
    pub fn parse(content: &str, path: impl AsRef<Path>) -> Result<Self, String> {
        let path_ref = path.as_ref();
        let config: ReachConfig = toml::from_str(content).map_err(|err| {
            format!(
                "failed to parse configuration file '{}': {err}",
                path_ref.display()
            )
        })?;

        if config.window_seconds < 3 || config.window_seconds > 120 {
            return Err(format!(
                "failed to validate '{}': 'window_seconds' ({}) must be between 3 and 120",
                path_ref.display(),
                config.window_seconds
            ));
        }

        if config.model_layers < 1 || config.model_layers > 20 {
            return Err(format!(
                "failed to validate '{}': 'model_layers' ({}) must be between 1 and 20",
                path_ref.display(),
                config.model_layers
            ));
        }

        if config.endpoint.start_threshold <= config.endpoint.end_threshold {
            return Err(format!(
                "failed to validate '{}': 'start_threshold' ({}) must be greater than 'end_threshold' ({})",
                path_ref.display(),
                config.endpoint.start_threshold,
                config.endpoint.end_threshold
            ));
        }

        if !(0.0..=1.0).contains(&config.endpoint.start_threshold) {
            return Err(format!(
                "failed to validate '{}': 'start_threshold' ({}) must be between 0.0 and 1.0",
                path_ref.display(),
                config.endpoint.start_threshold
            ));
        }

        if !(0.0..=1.0).contains(&config.endpoint.end_threshold) {
            return Err(format!(
                "failed to validate '{}': 'end_threshold' ({}) must be between 0.0 and 1.0",
                path_ref.display(),
                config.endpoint.end_threshold
            ));
        }

        if config.endpoint.pre_roll_ms > config.endpoint.max_utterance_ms {
            return Err(format!(
                "failed to validate '{}': 'pre_roll_ms' ({}) cannot exceed 'max_utterance_ms' ({})",
                path_ref.display(),
                config.endpoint.pre_roll_ms,
                config.endpoint.max_utterance_ms
            ));
        }

        if config.endpoint.min_speech_ms >= config.endpoint.max_utterance_ms {
            return Err(format!(
                "failed to validate '{}': 'min_speech_ms' ({}) must be less than 'max_utterance_ms' ({})",
                path_ref.display(),
                config.endpoint.min_speech_ms,
                config.endpoint.max_utterance_ms
            ));
        }

        Ok(config)
    }
}

/// Custom phrase rewrite configuration loaded from `config\phrases.toml`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhrasesConfig {
    /// Table mapping spoken phrases to canonical command strings.
    #[serde(default)]
    pub phrases: HashMap<String, String>,
}

impl PhrasesConfig {
    /// Loads phrases mapping from `path`, or returns empty mapping if the file is missing.
    ///
    /// # Arguments
    /// * `path` - Path to `phrases.toml`.
    ///
    /// # Returns
    /// The loaded [`PhrasesConfig`] or default empty configuration.
    ///
    /// # Errors
    /// Returns an error message if reading or parsing the file fails.
    pub fn load_or_default(path: impl AsRef<Path>) -> Result<Self, String> {
        let path_ref = path.as_ref();
        if !path_ref.exists() {
            return Ok(Self::default());
        }
        let content = std::fs::read_to_string(path_ref).map_err(|err| {
            format!(
                "failed to read phrases file '{}': {err}",
                path_ref.display()
            )
        })?;
        Self::parse(&content, path_ref)
    }

    /// Parses and validates phrase mappings from a TOML string.
    ///
    /// Normalises mapping keys and rejects any mapping targeting "yes", "no", "confirm", or "cancel".
    /// To preserve privacy, spoken phrases are never logged or exposed in error messages.
    ///
    /// # Arguments
    /// * `content` - TOML formatted phrases table.
    /// * `path` - Path of the phrases file for error diagnostics.
    ///
    /// # Returns
    /// The validated [`PhrasesConfig`].
    ///
    /// # Errors
    /// Returns an error message if parsing fails or any phrase targets forbidden actions.
    pub fn parse(content: &str, path: impl AsRef<Path>) -> Result<Self, String> {
        let path_ref = path.as_ref();
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct RawPhrases {
            #[serde(default)]
            phrases: HashMap<String, String>,
        }

        let raw: RawPhrases = toml::from_str(content).map_err(|err| {
            format!(
                "failed to parse phrases file '{}': {err}",
                path_ref.display()
            )
        })?;

        let mut normalized_phrases = HashMap::with_capacity(raw.phrases.len());
        for (phrase, canonical) in raw.phrases {
            let target = canonical.trim().to_owned();
            let target_lower = target.to_ascii_lowercase();
            if target_lower == "yes"
                || target_lower == "no"
                || target_lower == "confirm"
                || target_lower == "cancel"
            {
                return Err(format!(
                    "failed to validate '{}': phrase targets forbidden action '{}' (confirm/deny/cancel phrases cannot be aliased)",
                    path_ref.display(),
                    target
                ));
            }
            let key = normalize_phrase(&phrase);
            normalized_phrases.insert(key, target);
        }

        Ok(Self {
            phrases: normalized_phrases,
        })
    }

    /// Rewrites a spoken command to its canonical form if configured.
    ///
    /// Normalises `command` and checks against configured phrases. Unknown phrases pass through unchanged.
    ///
    /// # Arguments
    /// * `command` - The spoken command string after wake word removal.
    ///
    /// # Returns
    /// The canonical command string if matched, or the original command.
    pub fn rewrite<'a>(&'a self, command: &'a str) -> Cow<'a, str> {
        let normalized = normalize_phrase(command);
        if let Some(target) = self.phrases.get(&normalized) {
            Cow::Borrowed(target.as_str())
        } else {
            Cow::Borrowed(command)
        }
    }
}

/// Normalises spoken phrase text for exact matching.
///
/// Trims leading and trailing whitespace, strips trailing punctuation, collapses
/// consecutive whitespace characters to a single space, and converts to lowercase.
///
/// # Arguments
/// * `text` - The input phrase or command to normalise.
///
/// # Returns
/// The normalised string.
pub fn normalize_phrase(text: &str) -> String {
    let trimmed = text.trim();
    let without_punct = trimmed.trim_end_matches(|c: char| c.is_ascii_punctuation());
    without_punct
        .split_whitespace()
        .map(|word| word.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Defaults load when the file is missing.
    fn defaults_load_when_the_file_is_missing() {
        let missing = Path::new("config/non_existent_reach.toml");
        let config =
            ReachConfig::load_or_default(missing).expect("missing file should yield defaults");
        assert_eq!(config, ReachConfig::default());

        let missing_phrases = Path::new("config/non_existent_phrases.toml");
        let phrases = PhrasesConfig::load_or_default(missing_phrases)
            .expect("missing phrases file should yield default");
        assert_eq!(phrases, PhrasesConfig::default());
    }

    #[test]
    /// A full valid file parses to the expected values.
    fn full_valid_file_parses_to_expected_values() {
        let toml_str = r#"
wake_word = "reach"
aliases = ["rich", "computer"]
window_seconds = 20
stt_model = "models/custom_stt.bin"
whisper_threads = 4
needle_model = "models/custom_needle.cact"
model_layers = 18
no_model = false
no_voice = true

[endpoint]
start_threshold = 0.05
end_threshold = 0.01
min_speech_ms = 500
end_silence_ms = 800
pre_roll_ms = 300
max_utterance_ms = 7000
"#;
        let config =
            ReachConfig::parse(toml_str, Path::new("reach.toml")).expect("valid toml should parse");
        assert_eq!(config.wake_word, "reach");
        assert_eq!(
            config.aliases,
            vec!["rich".to_owned(), "computer".to_owned()]
        );
        assert_eq!(config.window_seconds, 20);
        assert_eq!(config.stt_model, "models/custom_stt.bin");
        assert_eq!(config.whisper_threads, 4);
        assert_eq!(config.needle_model, "models/custom_needle.cact");
        assert_eq!(config.model_layers, 18);
        assert!(!config.no_model);
        assert!(config.no_voice);
        assert_eq!(config.endpoint.start_threshold, 0.05);
        assert_eq!(config.endpoint.end_threshold, 0.01);
        assert_eq!(config.endpoint.min_speech_ms, 500);
        assert_eq!(config.endpoint.end_silence_ms, 800);
        assert_eq!(config.endpoint.pre_roll_ms, 300);
        assert_eq!(config.endpoint.max_utterance_ms, 7000);
    }

    #[test]
    /// Each invalid case gives an error naming the key.
    fn each_invalid_case_gives_an_error_naming_the_key() {
        let path = Path::new("reach.toml");

        // Unknown key
        let err = ReachConfig::parse("typo_key = 123", path).unwrap_err();
        assert!(
            err.contains("typo_key"),
            "error must name unknown key: {err}"
        );

        // window_seconds out of range (low)
        let err = ReachConfig::parse("window_seconds = 2", path).unwrap_err();
        assert!(
            err.contains("window_seconds"),
            "error must name window_seconds: {err}"
        );

        // window_seconds out of range (high)
        let err = ReachConfig::parse("window_seconds = 121", path).unwrap_err();
        assert!(
            err.contains("window_seconds"),
            "error must name window_seconds: {err}"
        );

        // model_layers 0
        let err = ReachConfig::parse("model_layers = 0", path).unwrap_err();
        assert!(
            err.contains("model_layers"),
            "error must name model_layers: {err}"
        );

        // model_layers 21
        let err = ReachConfig::parse("model_layers = 21", path).unwrap_err();
        assert!(
            err.contains("model_layers"),
            "error must name model_layers: {err}"
        );

        // start_threshold below end_threshold
        let err = ReachConfig::parse(
            "[endpoint]\nstart_threshold = 0.005\nend_threshold = 0.01",
            path,
        )
        .unwrap_err();
        assert!(
            err.contains("start_threshold") || err.contains("end_threshold"),
            "error must name threshold key: {err}"
        );

        // pre_roll_ms above max_utterance_ms
        let err = ReachConfig::parse(
            "[endpoint]\npre_roll_ms = 8000\nmax_utterance_ms = 6000",
            path,
        )
        .unwrap_err();
        assert!(
            err.contains("pre_roll_ms"),
            "error must name pre_roll_ms: {err}"
        );

        // wrong type
        let err = ReachConfig::parse("window_seconds = \"fifteen\"", path).unwrap_err();
        assert!(
            err.contains("window_seconds"),
            "error must name wrong type key: {err}"
        );
    }

    #[test]
    /// CLI flags override the file, which overrides defaults.
    fn cli_flags_override_file_which_overrides_defaults() {
        use crate::listen::parse_listen_args;

        let file_toml = r#"
stt_model = "models/file_model.bin"
no_voice = true
no_model = true
[endpoint]
start_threshold = 0.05
min_speech_ms = 500
"#;
        let mut config =
            ReachConfig::parse(file_toml, Path::new("reach.toml")).expect("valid file");
        assert_eq!(config.stt_model, "models/file_model.bin");
        assert!(config.no_voice);
        assert!(config.no_model);
        assert_eq!(config.endpoint.start_threshold, 0.05);
        assert_eq!(config.endpoint.min_speech_ms, 500);

        // CLI flags override the file
        let args = vec![
            "--stt-model".to_owned(),
            "models/cli_model.bin".to_owned(),
            "--model".to_owned(),
            "--start-threshold".to_owned(),
            "0.07".to_owned(),
            "--min-speech-ms".to_owned(),
            "600".to_owned(),
        ];
        let listen_args = parse_listen_args(&args).expect("must parse args");
        config.apply_cli_overrides(&listen_args);

        assert_eq!(config.stt_model, "models/cli_model.bin");
        assert!(!config.no_model); // --model set no_model to false
        assert_eq!(config.endpoint.start_threshold, 0.07);
        assert_eq!(config.endpoint.min_speech_ms, 600);
        assert!(config.no_voice); // kept from file
    }

    #[test]
    /// Phrase rewrite maps a phrase to canonical form; unknown phrases pass through; yes/cancel are rejected.
    fn phrase_rewrite_and_forbidden_targets() {
        let valid_phrases = r#"
[phrases]
"scroll the page down" = "scroll down"
"please click submit" = "click submit"
"#;
        let phrases =
            PhrasesConfig::parse(valid_phrases, Path::new("phrases.toml")).expect("valid phrases");
        assert_eq!(phrases.rewrite("scroll the page down"), "scroll down");
        assert_eq!(phrases.rewrite("Scroll The Page Down!"), "scroll down");
        assert_eq!(phrases.rewrite("unknown phrase"), "unknown phrase");

        // Forbidden target "yes"
        let forbidden_yes = r#"
[phrases]
"sure thing" = "yes"
"#;
        let err = PhrasesConfig::parse(forbidden_yes, Path::new("phrases.toml")).unwrap_err();
        assert!(
            err.contains("yes") || err.contains("Confirm"),
            "error must mention forbidden target: {err}"
        );

        // Forbidden target "cancel"
        let forbidden_cancel = r#"
[phrases]
"stop that" = "cancel"
"#;
        let err = PhrasesConfig::parse(forbidden_cancel, Path::new("phrases.toml")).unwrap_err();
        assert!(
            err.contains("cancel") || err.contains("Deny"),
            "error must mention forbidden target: {err}"
        );
    }

    #[test]
    /// The shipped config\reach.toml and config\phrases.toml parse successfully.
    fn shipped_config_files_parse_successfully() {
        let reach_path = Path::new("../../config/reach.toml");
        let phrases_path = Path::new("../../config/phrases.toml");

        let reach_content = std::fs::read_to_string(reach_path)
            .or_else(|_| std::fs::read_to_string("config/reach.toml"))
            .expect("config/reach.toml must exist");
        let reach_config =
            ReachConfig::parse(&reach_content, reach_path).expect("shipped reach.toml must parse");
        assert_eq!(reach_config.wake_word, "reach");
        assert_eq!(reach_config.window_seconds, 15);
        assert_eq!(reach_config.model_layers, 20);

        let phrases_content = std::fs::read_to_string(phrases_path)
            .or_else(|_| std::fs::read_to_string("config/phrases.toml"))
            .expect("config/phrases.toml must exist");
        let phrases_config = PhrasesConfig::parse(&phrases_content, phrases_path)
            .expect("shipped phrases.toml must parse");
        assert!(phrases_config.phrases.is_empty());
    }
}
