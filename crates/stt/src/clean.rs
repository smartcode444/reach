//! Cleanup for raw Whisper output.
//!
//! Whisper sometimes returns noise annotations such as `[BLANK_AUDIO]`,
//! `(music)` or `*noise*`, and on silence it can hallucinate short phrases
//! such as "you" or "thank you". This module removes those artefacts before
//! the text reaches the command grammar. It never logs or stores transcripts.

/// Words that describe non-speech audio inside a bracketed annotation.
///
/// An annotation is removed only when every word in it belongs to this list,
/// so `(music)` and `[background noise]` are dropped while `(please)` is kept.
const NOISE_WORDS: &[&str] = &[
    "applause",
    "applauding",
    "audio",
    "background",
    "beep",
    "beeping",
    "beeps",
    "blank",
    "breathing",
    "clicking",
    "coughing",
    "effect",
    "effects",
    "hum",
    "humming",
    "inaudible",
    "instrumental",
    "laughter",
    "music",
    "noise",
    "playing",
    "rustling",
    "sigh",
    "sighing",
    "sighs",
    "silence",
    "soft",
    "sound",
    "sounds",
    "static",
    "upbeat",
    "wind",
];

/// Phrases Whisper is known to hallucinate when the input has no speech.
///
/// Matching is exact and case-insensitive after trimming surrounding
/// punctuation. The list deliberately excludes "yes", "no", "okay" and
/// "stop", because those are meaningful answers or commands.
const SILENCE_HALLUCINATIONS: &[&str] = &[
    "applause",
    "blank audio",
    "bye",
    "bye bye",
    "bye-bye",
    "goodbye",
    "inaudible",
    "like and subscribe",
    "music",
    "noise",
    "please like and subscribe",
    "please subscribe",
    "see you",
    "see you in the next video",
    "see you next time",
    "silence",
    "static",
    "subscribe",
    "thank you",
    "thank you for watching",
    "thank you for watching everyone",
    "thank-you",
    "thanks",
    "thanks for watching",
    "thanks for watching everyone",
    "the end",
    "you",
    "you are welcome",
    "you're welcome",
    "you’re welcome",
];

/// Cleans one raw Whisper transcript.
///
/// The function removes known noise annotations, collapses all whitespace to
/// single spaces, and rejects output that is empty or is a known silence
/// hallucination. Matching against the hallucination list is case-insensitive
/// and applies to the whole utterance only, so "Reach thank you" is kept.
///
/// # Arguments
/// * `raw` - The text exactly as Whisper returned it.
///
/// # Returns
/// The cleaned transcript, or `None` when nothing usable was heard.
pub fn clean_transcript(raw: &str) -> Option<String> {
    let without_noise = strip_noise_annotations(raw);
    let collapsed = collapse_whitespace(&without_noise);
    if collapsed.is_empty() {
        return None;
    }
    if is_silence_hallucination(&collapsed) {
        return None;
    }
    Some(collapsed)
}

/// Returns the delimiter that closes `opening`, if it starts an annotation.
///
/// # Arguments
/// * `opening` - A character that may start a bracketed annotation.
///
/// # Returns
/// The matching closing character, or `None` for ordinary text.
fn closing_delimiter(opening: char) -> Option<char> {
    match opening {
        '[' => Some(']'),
        '(' => Some(')'),
        '*' => Some('*'),
        _ => None,
    }
}

/// Removes bracketed annotations that describe non-speech audio.
///
/// Recognised annotations use `[...]`, `(...)` or `*...*` delimiters. Text
/// that is not a noise description, such as `(please)`, is left untouched.
///
/// # Arguments
/// * `raw` - The raw transcript.
///
/// # Returns
/// The transcript with noise annotations removed.
fn strip_noise_annotations(raw: &str) -> String {
    let mut cleaned = String::with_capacity(raw.len());
    let mut characters = raw.char_indices().peekable();

    while let Some((_, character)) = characters.next() {
        let Some(closing) = closing_delimiter(character) else {
            cleaned.push(character);
            continue;
        };
        let Some(content_start) = characters.peek().map(|(index, _)| *index) else {
            cleaned.push(character);
            continue;
        };

        let lookahead = characters.clone();
        let mut closing_index = None;
        for (index, candidate) in lookahead {
            if candidate == closing {
                closing_index = Some(index);
                break;
            }
        }

        let Some(closing_index) = closing_index else {
            cleaned.push(character);
            continue;
        };
        let content = &raw[content_start..closing_index];
        if is_noise_annotation(content) {
            // Consume the annotation's contents and its closing delimiter.
            for (index, _) in characters.by_ref() {
                if index == closing_index {
                    break;
                }
            }
        } else {
            cleaned.push(character);
        }
    }

    cleaned
}

/// Reports whether annotation `content` consists only of noise words.
///
/// # Arguments
/// * `content` - The text between the annotation delimiters.
///
/// # Returns
/// `true` for an empty annotation or one made entirely of noise words.
fn is_noise_annotation(content: &str) -> bool {
    let lowered = content.trim().to_lowercase();
    let words: Vec<&str> = lowered
        .split(|character: char| !character.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .collect();
    if words.is_empty() {
        return true;
    }
    words.iter().all(|word| NOISE_WORDS.contains(word))
}

/// Collapses every run of whitespace into one space and trims the ends.
///
/// # Arguments
/// * `text` - Text that may contain newlines, tabs or repeated spaces.
///
/// # Returns
/// The collapsed text.
fn collapse_whitespace(text: &str) -> String {
    let mut collapsed = String::with_capacity(text.len());
    for word in text.split_whitespace() {
        if !collapsed.is_empty() {
            collapsed.push(' ');
        }
        collapsed.push_str(word);
    }
    collapsed
}

/// Reports whether the whole cleaned utterance is a silence hallucination.
///
/// Surrounding punctuation is ignored, but internal words are not, so
/// "thank you" matches while "Reach thank you" does not.
///
/// # Arguments
/// * `text` - A cleaned, whitespace-collapsed transcript.
///
/// # Returns
/// `true` when the utterance is empty or exactly matches a known phrase.
fn is_silence_hallucination(text: &str) -> bool {
    let trimmed = text.trim_matches(|character: char| !character.is_alphanumeric());
    if trimmed.is_empty() {
        return true;
    }

    let normalized = trimmed
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    SILENCE_HALLUCINATIONS.contains(&normalized.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// A table of real transcripts and the expected cleaned result.
    fn cleans_transcripts() {
        let cases: &[(&str, Option<&str>)] = &[
            ("", None),
            ("   ", None),
            ("[BLANK_AUDIO]", None),
            ("(music)", None),
            ("*noise*", None),
            ("[BLANK_AUDIO] (music) *noise*", None),
            ("[BLANK_AUDIO] Reach scroll down", Some("Reach scroll down")),
            ("(music) click submit (applause)", Some("click submit")),
            ("  Reach   scroll \n down  ", Some("Reach scroll down")),
            ("you", None),
            ("You.", None),
            ("THANK YOU", None),
            ("thanks for watching!", None),
            ("bye", None),
            ("Reach thank you", Some("Reach thank you")),
            ("thank you very much", Some("thank you very much")),
            ("please subscribe", None),
            ("the end", None),
            ("music", None),
            (".", None),
            ("click submit", Some("click submit")),
            ("open Chrome", Some("open Chrome")),
            ("(please) click submit", Some("(please) click submit")),
            ("[inaudible] scroll up", Some("scroll up")),
            ("thank you, Reach", Some("thank you, Reach")),
            ("*noise* Reach scroll down", Some("Reach scroll down")),
            ("Reach (music) scroll down", Some("Reach scroll down")),
            ("Reach *noise* scroll down", Some("Reach scroll down")),
            ("[BLANK_AUDIO] you", None),
            ("[]", None),
            ("()", None),
            ("**", None),
            ("Reach [] scroll down", Some("Reach scroll down")),
            ("See you next time!", None),
            ("goodbye", None),
            ("Thanks for watching everyone.", None),
        ];

        for (index, (raw, expected)) in cases.iter().enumerate() {
            assert_eq!(
                clean_transcript(raw).as_deref(),
                *expected,
                "cleaning case {index} produced the wrong result"
            );
        }
    }

    #[test]
    /// A command that contains a hallucination phrase is kept intact.
    fn keeps_reach_thank_you() {
        assert_eq!(
            clean_transcript("Reach thank you"),
            Some("Reach thank you".to_owned())
        );
    }

    #[test]
    /// Hallucination matching ignores case and trailing punctuation.
    fn drops_hallucinations_case_insensitively() {
        for raw in ["You", "THANK YOU.", "Thanks For Watching!", "Bye"] {
            assert_eq!(clean_transcript(raw), None, "case {raw:?}");
        }
    }

    #[test]
    /// Parenthetical text that is not a known noise description is kept.
    fn keeps_unknown_parenthetical_text() {
        assert_eq!(
            clean_transcript("(please) click submit"),
            Some("(please) click submit".to_owned())
        );
    }

    #[test]
    /// Whitespace is trimmed and collapsed.
    fn collapses_whitespace() {
        assert_eq!(
            clean_transcript("  Reach \t  scroll\n\ndown  "),
            Some("Reach scroll down".to_owned())
        );
    }

    #[test]
    /// Punctuation-only output carries no speech and is rejected.
    fn drops_punctuation_only_output() {
        assert_eq!(clean_transcript(" ... !!! "), None);
    }
}
