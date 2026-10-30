//! Configuration loading and phrase rewriting for Reach.
//!
//! Loads `config\reach.toml` for settings and `config\phrases.toml` for custom
//! spoken phrase mappings. If files are missing, documented defaults are used.
//! If files exist but are invalid, clear errors naming the key and file are returned.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use audio::EndpointConfig;
use nlu::wake::WakeConfig;
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

/// Returns the default no-voice feedback flag.
fn default_no_voice() -> bool {
    DEFAULT_NO_VOICE
}

/// Voice activity detection and utterance endpointing parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EndpointSettings {
    /// Energy threshold to start an utterance (0.0 to 1.0).
    #[serde(default = "default_start_threshold")]
    pub start_threshold: f32,
    /// Energy threshold to transition towards silence (0.0 to 1.0).
    #[serde(default = "default_end_threshold")]
    pub end_threshold: f32,
    /// Minimum voiced speech duration in milliseconds.
    #[serde(default = "default_min_speech_ms")]
    pub min_speech_ms: u32,
    /// Continuous trailing silence in milliseconds to end an utterance.
    #[serde(default = "default_end_silence_ms")]
    pub end_silence_ms: u32,
    /// Audio pre-roll buffer in milliseconds before speech start.
    #[serde(default = "default_pre_roll_ms")]
    pub pre_roll_ms: u32,
    /// Hard maximum utterance duration in milliseconds.
    #[serde(default = "default_max_utterance_ms")]
    pub max_utterance_ms: u32,
}

/// Returns default start threshold.
fn default_start_threshold() -> f32 {
    0.03
}
/// Returns default end threshold.
fn default_end_threshold() -> f32 {
    0.008
}
/// Returns default minimum speech milliseconds.
fn default_min_speech_ms() -> u32 {
    200
}
/// Returns default end silence milliseconds.
fn default_end_silence_ms() -> u32 {
    700
}
/// Returns default pre-roll milliseconds.
fn default_pre_roll_ms() -> u32 {
    200
}
/// Returns default maximum utterance milliseconds.
fn default_max_utterance_ms() -> u32 {
    6000
}

impl Default for EndpointSettings {
    fn default() -> Self {
        Self {
            start_threshold: default_start_threshold(),
            end_threshold: default_end_threshold(),
            min_speech_ms: default_min_speech_ms(),
            end_silence_ms: default_end_silence_ms(),
            pre_roll_ms: default_pre_roll_ms(),
            max_utterance_ms: default_max_utterance_ms(),
        }
    }
}

impl EndpointSettings {
    /// Converts these endpoint settings into the audio crate's [`EndpointConfig`].
    ///
    /// # Returns
    /// An instance of [`EndpointConfig`].
    pub fn to_endpoint_config(&self) -> EndpointConfig {
        EndpointConfig {
            frame_samples: audio::FRAME_SAMPLES,
            start_threshold_rms: self.start_threshold,
            end_threshold_rms: self.end_threshold,
            min_speech_ms: self.min_speech_ms,
            end_silence_ms: self.end_silence_ms,
            pre_roll_ms: self.pre_roll_ms,
            max_utterance_ms: self.max_utterance_ms,
        }
    }

    /// Validates field ranges.
    ///
    /// # Errors
    /// Returns error string naming invalid field and reason.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.start_threshold) {
            return Err("start_threshold must be between 0.0 and 1.0".to_owned());
        }
        if !(0.0..=1.0).contains(&self.end_threshold) {
            return Err("end_threshold must be between 0.0 and 1.0".to_owned());
        }
        if self.start_threshold <= self.end_threshold {
            return Err("start_threshold must be greater than end_threshold".to_owned());
        }
        if self.min_speech_ms == 0 {
            return Err("min_speech_ms must be greater than zero".to_owned());
        }
        if self.max_utterance_ms < self.min_speech_ms {
            return Err("max_utterance_ms cannot be less than min_speech_ms".to_owned());
        }
        Ok(())
    }
}

/// Top-level settings loaded from `config\reach.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReachConfig {
    /// The primary wake word used to activate Reach.
    #[serde(default = "default_wake_word")]
    pub wake_word: String,
    /// Alternative phonetically similar words accepted as the wake word.
    #[serde(default = "default_aliases")]
    pub aliases: Vec<String>,
    /// Duration in seconds the listening window stays open after a bare wake word.
    #[serde(default = "default_window_seconds")]
    pub window_seconds: u32,
    /// Path to the Whisper speech-to-text GGML model container.
    #[serde(default = "default_stt_model")]
    pub stt_model: String,
    /// Number of Whisper decoding threads (0 = auto).
    #[serde(default)]
    pub whisper_threads: usize,
    /// Path to the Needle 3 in-process intent model.
    #[serde(default = "default_needle_model")]
    pub needle_model: String,
    /// Number of transformer layers used by Needle (1 to 20).
    #[serde(default = "default_model_layers")]
    pub model_layers: u32,
    /// When true, runs in rules-only mode without loading or querying Needle.
    #[serde(default = "default_no_model")]
    pub no_model: bool,
    /// When true, disables spoken voice feedback.
    #[serde(default = "default_no_voice")]
    pub no_voice: bool,
    /// Voice activity detection and endpointing settings.
    #[serde(default)]
    pub endpoint: EndpointSettings,
}

impl Default for ReachConfig {
    fn default() -> Self {
        Self {
            wake_word: default_wake_word(),
            aliases: default_aliases(),
            window_seconds: default_window_seconds(),
            stt_model: default_stt_model(),
            whisper_threads: DEFAULT_WHISPER_THREADS,
            needle_model: default_needle_model(),
            model_layers: default_model_layers(),
            no_model: default_no_model(),
            no_voice: default_no_voice(),
            endpoint: EndpointSettings::default(),
        }
    }
}

/// Command-line overrides that can be applied to [`ReachConfig`].
#[derive(Debug, Default, Clone, PartialEq)]
pub struct ConfigOverrides {
    /// Explicit flag to enable model.
    pub model: bool,
    /// Explicit flag to disable model.
    pub no_model: bool,
    /// Explicit flag to disable voice feedback.
    pub no_voice: bool,
    /// Override model path.
    pub model_path: Option<String>,
    /// Override STT model path.
    pub stt_model: Option<String>,
    /// Override start threshold.
    pub start_threshold: Option<f32>,
    /// Override min speech duration.
    pub min_speech_ms: Option<u32>,
}

impl ReachConfig {
    /// Applies CLI argument overrides to this configuration.
    ///
    /// # Arguments
    /// * `overrides` - Parsed CLI options.
    pub fn apply_overrides(&mut self, overrides: &ConfigOverrides) {
        if overrides.model {
            self.no_model = false;
        }
        if overrides.no_model {
            self.no_model = true;
        }
        if overrides.no_voice {
            self.no_voice = true;
        }
        if let Some(ref path) = overrides.model_path {
            self.needle_model = path.clone();
        }
        if let Some(ref path) = overrides.stt_model {
            self.stt_model = path.clone();
        }
        if let Some(start) = overrides.start_threshold {
            self.endpoint.start_threshold = start;
        }
        if let Some(min_speech) = overrides.min_speech_ms {
            self.endpoint.min_speech_ms = min_speech;
        }
    }

    /// Converts this configuration into a [`WakeConfig`].
    ///
    /// # Returns
    /// An instance of [`WakeConfig`].
    pub fn to_wake_config(&self) -> WakeConfig {
        WakeConfig {
            wake_word: self.wake_word.clone(),
            aliases: self.aliases.clone(),
            window: Duration::from_secs(u64::from(self.window_seconds)),
        }
    }

    /// Loads the configuration from `path`, or returns documented defaults if absent.
    ///
    /// # Arguments
    /// * `path` - Filesystem path to the TOML configuration.
    ///
    /// # Returns
    /// An instance of [`ReachConfig`].
    ///
    /// # Errors
    /// Returns error naming the file and key if syntax or semantics are invalid.
    pub fn load_or_default<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let p = path.as_ref();
        if !p.exists() {
            return Ok(Self::default());
        }
        let content = std::fs::read_to_string(p)
            .map_err(|e| format!("failed to read configuration file '{}': {e}", p.display()))?;
        let config: Self = toml::from_str(&content)
            .map_err(|e| format!("invalid configuration file '{}': {e}", p.display()))?;
        config.validate(p)?;
        Ok(config)
    }

    /// Validates value ranges and field invariants.
    ///
    /// # Arguments
    /// * `path` - The path the configuration was loaded from.
    ///
    /// # Errors
    /// Returns error message naming key and reason.
    fn validate(&self, path: &Path) -> Result<(), String> {
        if self.wake_word.trim().is_empty() {
            return Err(format!(
                "in '{}': key 'wake_word' cannot be empty",
                path.display()
            ));
        }
        if !(3..=120).contains(&self.window_seconds) {
            return Err(format!(
                "in '{}': key 'window_seconds' must be between 3 and 120 (found {})",
                path.display(),
                self.window_seconds
            ));
        }
        if !(1..=20).contains(&self.model_layers) {
            return Err(format!(
                "in '{}': key 'model_layers' must be between 1 and 20 (found {})",
                path.display(),
                self.model_layers
            ));
        }
        self.endpoint
            .validate()
            .map_err(|e| format!("in '{}': {e}", path.display()))?;
        Ok(())
    }
}

/// Spoken phrase rewrite mappings loaded from `config\phrases.toml`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhrasesConfig {
    /// Exact full-utterance rewrites.
    #[serde(default)]
    pub rewrites: HashMap<String, String>,
    /// Substring replacements applied left-to-right.
    #[serde(default)]
    pub replacements: Vec<SubstringReplacement>,
    /// Forbidden target words that must not be clicked, focused, or typed.
    #[serde(default)]
    pub forbidden_targets: Vec<String>,
}

/// An individual substring replacement rule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SubstringReplacement {
    /// String to search for.
    pub from: String,
    /// String to replace it with.
    pub to: String,
}

impl PhrasesConfig {
    /// Loads phrase mappings from `path`, or returns empty defaults if missing.
    ///
    /// # Arguments
    /// * `path` - Path to the phrases TOML file.
    ///
    /// # Returns
    /// An instance of [`PhrasesConfig`].
    ///
    /// # Errors
    /// Returns error message if TOML syntax is invalid.
    pub fn load_or_default<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let p = path.as_ref();
        if !p.exists() {
            return Ok(Self::default());
        }
        let content = std::fs::read_to_string(p)
            .map_err(|e| format!("failed to read phrases file '{}': {e}", p.display()))?;
        let config: Self = toml::from_str(&content)
            .map_err(|e| format!("invalid phrases file '{}': {e}", p.display()))?;
        Ok(config)
    }

    /// Rewrites an utterance using full-utterance mappings and substring rules.
    ///
    /// # Arguments
    /// * `command` - The command string after the wake gate.
    ///
    /// # Returns
    /// The rewritten command string.
    pub fn rewrite<'a>(&'a self, command: &'a str) -> Cow<'a, str> {
        let lower = command.to_lowercase();
        if let Some(target) = self.rewrites.get(&lower) {
            return Cow::Owned(target.clone());
        }

        let mut current = Cow::Borrowed(command);
        for rule in &self.replacements {
            if current.contains(&rule.from) {
                let replaced = current.replace(&rule.from, &rule.to);
                current = Cow::Owned(replaced);
            }
        }
        current
    }

    /// Checks whether a target word or phrase is on the forbidden list.
    ///
    /// # Arguments
    /// * `target` - Target element name or phrase.
    ///
    /// # Returns
    /// True if forbidden, false otherwise.
    pub fn is_forbidden_target(&self, target: &str) -> bool {
        let trimmed = target.trim().to_lowercase();
        self.forbidden_targets
            .iter()
            .any(|f| f.to_lowercase() == trimmed)
    }
}
