//! The fast-path rule grammar.
//!
//! The grammar recognises common command phrasing before the Needle model is
//! consulted. It is deliberately conservative: when a phrase is ambiguous or
//! the target is unclear it returns `None`, which lets the caller ask for a
//! clarification instead of guessing.

use reach_core::{Direction, Intent, Key, Mode, ZoomDir};

use crate::numbers::{parse_number, parse_ordinal};
use crate::routing::CLARIFY_QUESTION;

/// Words that carry no command meaning and are removed before matching.
const FILLERS: &[&str] = &["um", "uh", "erm", "er", "ah", "hmm", "please", "kindly"];

/// Wake words and politeness phrases removed from the start of an utterance.
///
/// Longer phrases come first so that "hey reach" is removed as a unit rather
/// than leaving "hey" behind.
const PREFIX_PHRASES: &[&str] = &[
    "i would like you to",
    "i would like to",
    "i'd like you to",
    "i'd like to",
    "i want you to",
    "i want to",
    "hey reach",
    "okay reach",
    "ok reach",
    "hey rich",
    "okay rich",
    "ok rich",
    "hey ridge",
    "okay ridge",
    "ok ridge",
    "go ahead and",
    "can you",
    "could you",
    "would you",
    "will you",
    "please",
    "kindly",
    "reach",
    "rich",
    "ridge",
    "just",
];

/// Generic nouns removed from the end of a named click target.
const CLICK_NOUNS: &[&str] = &["button", "link", "icon"];

/// Generic nouns removed from the end of a field name.
const FIELD_NOUNS: &[&str] = &["field", "box", "input", "textarea", "area"];

/// Words that may follow a scroll amount, such as "times" or "pages".
const SCROLL_UNITS: &[&str] = &[
    "time", "times", "step", "steps", "page", "pages", "notch", "notches", "percent", "%",
];

/// The largest scroll amount the rule grammar will act on.
///
/// A misheard number must not turn into a very long scroll on a low-end
/// laptop, so anything above this is left for clarification.
const MAX_SCROLL_AMOUNT: u32 = 50;

/// Top-level domains accepted for a bare spoken web address.
///
/// The list is deliberately finite: it keeps a dotted local file name from
/// being mistaken for a web address and handed to the shell.
const WEB_TLDS: &[&str] = &[
    "ai", "app", "biz", "blog", "cloud", "co", "com", "dev", "edu", "gov", "info", "io", "live",
    "me", "mobi", "name", "net", "news", "ng", "online", "org", "pro", "shop", "site", "space",
    "store", "tech", "tv", "uk", "website", "world", "xyz",
];

/// Application names the rule grammar is allowed to launch.
///
/// The executor enforces the same list. Rejecting an unknown name here keeps a
/// misheard sentence from being treated as an application launch at all.
const ALLOWED_APPS: &[&str] = &["chrome", "notepad", "explorer"];

/// The largest number of words accepted in a named click target.
///
/// Element names are short. A longer phrase is usually ordinary speech such
/// as "the link in the email I sent", so it is left for clarification.
const MAX_CLICK_TARGET_WORDS: usize = 4;

/// Words that introduce a click command, used to restore spoken casing.
const CLICK_TRIGGERS: &[&str] = &["click", "tap", "press", "hit", "choose", "select"];

/// Words that introduce a focus command, used to restore spoken casing.
const FOCUS_TRIGGERS: &[&str] = &[
    "focus", "move", "go", "click", "select", "tap", "press", "hit",
];

/// Words that introduce a window-switch command, used to restore spoken casing.
const SWITCH_TRIGGERS: &[&str] = &["switch", "bring", "activate"];

/// The result of trying to read a key press from an utterance.
enum KeyAttempt {
    /// The words do not ask for a key press.
    NotAKey,
    /// The words name a key that `reach_core::Key` cannot represent.
    Unsupported,
    /// The words name a supported key.
    Key(Key),
}

/// The result of trying to read a numbered click.
enum NumberAttempt {
    /// The words are not a numbered click.
    NotNumbered,
    /// The words name a number outside the valid overlay range.
    Invalid,
    /// The words name a valid overlay number.
    Number(u32),
}

/// The result of trying to read a click command.
enum ClickAttempt {
    /// The words do not ask for a click.
    NotClick,
    /// The words ask for a click but the target is invalid.
    Invalid,
    /// A valid click.
    Click(Intent),
}

/// Parses one spoken utterance with the fast rule grammar.
///
/// # Arguments
/// * `utterance` - The recognised text, with its original casing and
///   punctuation. The parser lowercases and cleans it internally.
///
/// # Returns
/// The matching [`Intent`], or `None` when no rule applies. `None` means the
/// caller should try the model or ask the user to rephrase; it never means
/// "do nothing".
pub fn parse_rule(utterance: &str) -> Option<Intent> {
    let cleaned = strip_prefixes(utterance);
    if cleaned.is_empty() {
        return None;
    }

    // Typing keeps the original casing, so it is parsed before tokenisation.
    if let Some(intent) = parse_type_text(cleaned) {
        return Some(intent);
    }

    let tokens = tokenize(cleaned);
    if tokens.is_empty() {
        return None;
    }

    if let Some(intent) = parse_mode(&tokens) {
        return Some(intent);
    }
    if let Some(intent) = parse_control(&tokens) {
        return Some(intent);
    }
    if let Some(intent) = parse_screen(&tokens) {
        return Some(intent);
    }
    if let Some(intent) = parse_zoom(&tokens) {
        return Some(intent);
    }
    if let Some(intent) = parse_scroll(&tokens) {
        return Some(intent);
    }
    if let Some(intent) = parse_focus(&tokens) {
        return Some(restore_intent_case(cleaned, intent));
    }
    if let Some(intent) = parse_switch(&tokens) {
        return Some(restore_intent_case(cleaned, intent));
    }

    match parse_key(&tokens) {
        KeyAttempt::Key(key) => return Some(Intent::PressKey { key }),
        KeyAttempt::Unsupported => return None,
        KeyAttempt::NotAKey => {}
    }

    match parse_click(&tokens) {
        ClickAttempt::Click(intent) => return Some(restore_intent_case(cleaned, intent)),
        ClickAttempt::Invalid => return None,
        ClickAttempt::NotClick => {}
    }

    parse_open(&tokens)
}

/// Restores the spoken casing of a target that was matched case-insensitively.
///
/// # Arguments
/// * `text` - The utterance after wake words and politeness phrases are
///   removed, with its original casing.
/// * `intent` - The intent produced by the lowercase parser.
///
/// # Returns
/// The same intent, with any named click, field or window target replaced by
/// the words the user actually spoke. Other intents are unchanged.
fn restore_intent_case(text: &str, intent: Intent) -> Intent {
    match intent {
        Intent::ClickElement { target } => Intent::ClickElement {
            target: restore_spoken_case(text, CLICK_TRIGGERS, &target),
        },
        Intent::FocusField { target } => Intent::FocusField {
            target: restore_spoken_case(text, FOCUS_TRIGGERS, &target),
        },
        Intent::SwitchWindow {
            target: Some(target),
        } => Intent::SwitchWindow {
            target: Some(restore_spoken_case(text, SWITCH_TRIGGERS, &target)),
        },
        other => other,
    }
}

/// Finds `target` in `text` after one of `triggers` and returns its spoken form.
///
/// The parser matches lowercase tokens, so this helper walks the original
/// tokens to recover casing. If the target cannot be found, the lowercase
/// value is returned unchanged rather than losing the command.
///
/// # Arguments
/// * `text` - The utterance with its original casing.
/// * `triggers` - The command words that introduce the target.
/// * `target` - The lowercase target produced by the parser.
///
/// # Returns
/// The matching words joined with spaces, or `target` when no match is found.
fn restore_spoken_case(text: &str, triggers: &[&str], target: &str) -> String {
    let original = tokenize_preserving_case(text);
    let target_words: Vec<String> = target.split_whitespace().map(str::to_lowercase).collect();
    if target_words.is_empty() {
        return target.to_owned();
    }
    let search_start = original
        .iter()
        .position(|token| {
            let lower = token.to_lowercase();
            triggers.iter().any(|trigger| lower == *trigger)
        })
        .map_or(0, |index| index + 1);
    let Some(last_start) = original.len().checked_sub(target_words.len()) else {
        return target.to_owned();
    };
    for index in search_start..=last_start {
        let Some(window) = original.get(index..index + target_words.len()) else {
            continue;
        };
        let matches = window
            .iter()
            .zip(target_words.iter())
            .all(|(spoken, expected)| spoken.to_lowercase() == *expected);
        if matches {
            return window.join(" ");
        }
    }
    target.to_owned()
}

/// Removes wake words and politeness phrases from the start of `text`.
///
/// # Arguments
/// * `text` - The raw utterance, with its original casing.
///
/// # Returns
/// The remaining text with surrounding whitespace trimmed. Repeated phrases
/// are removed, so "please can you click" becomes "click".
fn strip_prefixes(text: &str) -> &str {
    let mut rest = text.trim();
    'outer: loop {
        for phrase in PREFIX_PHRASES {
            if let Some(after) = strip_prefix_ascii_case(rest, phrase) {
                rest = after.trim_start_matches(|c: char| c.is_whitespace() || c == ',');
                continue 'outer;
            }
        }
        break;
    }
    rest
}

/// Strips an ASCII `prefix` from `text` when it ends on a word boundary.
///
/// # Arguments
/// * `text` - The text to inspect.
/// * `prefix` - A lowercase ASCII phrase.
///
/// # Returns
/// The text after the prefix, including any separating space, or `None` when
/// the prefix does not match. A comma or colon also counts as a boundary so
/// that "Reach, click" and "type: hello" are recognised.
fn strip_prefix_ascii_case<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    let head = text.get(..prefix.len())?;
    if !head.eq_ignore_ascii_case(prefix) {
        return None;
    }
    let tail = text.get(prefix.len()..)?;
    match tail.chars().next() {
        None => Some(tail),
        Some(c) if c.is_whitespace() || c == ',' || c == ':' => Some(tail),
        _ => None,
    }
}

/// Splits `text` into lowercase tokens with surrounding punctuation removed.
///
/// # Arguments
/// * `text` - The cleaned utterance.
///
/// # Returns
/// Tokens with filler words removed. Punctuation inside a token is kept so
/// that web addresses survive intact.
fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(clean_token)
        .filter(|token| !token.is_empty() && !FILLERS.contains(&token.as_str()))
        .collect()
}

/// Splits `text` into cleaned tokens without changing their casing.
///
/// Matching still uses [`tokenize`]; this variant exists so a target or field
/// that was matched case-insensitively can be returned as the user spoke it.
///
/// # Arguments
/// * `text` - The text to split.
///
/// # Returns
/// Tokens with surrounding punctuation removed and filler words dropped.
fn tokenize_preserving_case(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(clean_token_preserving_case)
        .filter(|token| {
            if token.is_empty() {
                return false;
            }
            let lower = token.to_lowercase();
            !FILLERS.contains(&lower.as_str())
        })
        .collect()
}

/// Cleans one token for matching.
///
/// # Arguments
/// * `token` - One whitespace-separated word.
///
/// # Returns
/// The lowercase token with typographic quotes normalised and leading or
/// trailing ASCII punctuation removed.
fn clean_token(token: &str) -> String {
    clean_token_preserving_case(token).to_lowercase()
}

/// Cleans one token without lowercasing it.
///
/// # Arguments
/// * `token` - One whitespace-separated word.
///
/// # Returns
/// The token with typographic quotes normalised and leading or trailing ASCII
/// punctuation removed. Internal punctuation is kept.
fn clean_token_preserving_case(token: &str) -> String {
    token
        .replace(['\u{2018}', '\u{2019}'], "'")
        .replace(['\u{201c}', '\u{201d}'], "\"")
        .trim_matches(|c: char| c.is_ascii_punctuation())
        .to_owned()
}

/// Reports whether `tokens` is exactly `phrase`.
///
/// # Arguments
/// * `tokens` - Cleaned utterance tokens.
/// * `phrase` - The lowercase words to compare against.
///
/// # Returns
/// `true` when the lengths and every word match.
fn is_phrase(tokens: &[String], phrase: &[&str]) -> bool {
    tokens.len() == phrase.len()
        && tokens
            .iter()
            .zip(phrase.iter())
            .all(|(token, word)| token.as_str() == *word)
}

/// Removes `phrase` from the start of `tokens`.
///
/// # Arguments
/// * `tokens` - Cleaned utterance tokens.
/// * `phrase` - The lowercase words to remove.
///
/// # Returns
/// The remaining tokens, or `None` when `tokens` does not start with `phrase`.
fn strip_phrase_prefix<'a>(tokens: &'a [String], phrase: &[&str]) -> Option<&'a [String]> {
    if tokens.len() < phrase.len() {
        return None;
    }
    let head = tokens.get(..phrase.len())?;
    if head
        .iter()
        .zip(phrase.iter())
        .all(|(token, word)| token.as_str() == *word)
    {
        tokens.get(phrase.len()..)
    } else {
        None
    }
}

/// Removes `phrase` from the end of `tokens`.
///
/// # Arguments
/// * `tokens` - Cleaned utterance tokens.
/// * `phrase` - The lowercase words to remove.
///
/// # Returns
/// The remaining tokens, or `None` when `tokens` does not end with `phrase`.
fn strip_phrase_suffix<'a>(tokens: &'a [String], phrase: &[&str]) -> Option<&'a [String]> {
    let start = tokens.len().checked_sub(phrase.len())?;
    let tail = tokens.get(start..)?;
    if tail
        .iter()
        .zip(phrase.iter())
        .all(|(token, word)| token.as_str() == *word)
    {
        tokens.get(..start)
    } else {
        None
    }
}

/// Removes politeness and conversational fillers from the end of `tokens`.
///
/// Strips "please", "kindly", "for me", and "now" repeatedly.
///
/// # Arguments
/// * `tokens` - Cleaned tokens to inspect.
///
/// # Returns
/// A subslice of `tokens` with trailing politeness removed.
fn strip_trailing_politeness(mut tokens: &[String]) -> &[String] {
    loop {
        let mut stripped = false;
        for suffix in [
            &["please"][..],
            &["kindly"][..],
            &["for", "me"][..],
            &["now"][..],
        ] {
            if let Some(rest) = strip_phrase_suffix(tokens, suffix) {
                tokens = rest;
                stripped = true;
                break;
            }
        }
        if !stripped {
            break;
        }
    }
    tokens
}

/// Removes filler words and leading articles from on-screen targets.
///
/// Strips trailing "please", "kindly", "for me", "a little", "a bit", and "now",
/// as well as any leading "the".
///
/// # Arguments
/// * `tokens` - Cleaned tokens representing a candidate target.
///
/// # Returns
/// A subslice of `tokens` with fillers and leading articles stripped.
fn strip_target_fillers(mut tokens: &[String]) -> &[String] {
    loop {
        let mut stripped = false;
        for suffix in [
            &["please"][..],
            &["kindly"][..],
            &["for", "me"][..],
            &["now"][..],
            &["a", "bit"][..],
            &["a", "little"][..],
        ] {
            if let Some(rest) = strip_phrase_suffix(tokens, suffix) {
                tokens = rest;
                stripped = true;
                break;
            }
        }
        if !stripped {
            break;
        }
    }
    while let Some(rest) = strip_phrase_prefix(tokens, &["the"]) {
        tokens = rest;
    }
    tokens
}

/// Finds the first case-insensitive occurrence of ASCII `needle` in `text`.
///
/// # Arguments
/// * `haystack` - The text to search.
/// * `needle` - A lowercase ASCII phrase.
///
/// # Returns
/// The byte index where `needle` starts, or `None` when it is absent. The
/// search never splits a multi-byte character, so it is safe on any input.
fn find_ascii_case_insensitive(haystack: &str, needle: &str) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    haystack.char_indices().find_map(|(index, _)| {
        // Skip byte ranges that would split a multi-byte character.
        let end = index.checked_add(needle.len())?;
        let candidate = haystack.get(index..end)?;
        candidate.eq_ignore_ascii_case(needle).then_some(index)
    })
}

/// Strips one of `triggers` from the start of `text`.
///
/// # Arguments
/// * `text` - The text after wake words and politeness phrases are removed.
/// * `triggers` - Lowercase ASCII trigger words such as "type" or "write".
///
/// # Returns
/// The text after the trigger with leading whitespace and punctuation
/// removed, or `None` when no trigger matches.
fn strip_trigger<'a>(text: &'a str, triggers: &[&str]) -> Option<&'a str> {
    triggers
        .iter()
        .find_map(|trigger| strip_prefix_ascii_case(text, trigger))
        .map(|rest| rest.trim_start_matches(|c: char| c.is_whitespace() || c == ':' || c == ','))
}

/// Removes any of `phrases` repeatedly from the start of `text`.
///
/// # Arguments
/// * `text` - The text to clean.
/// * `phrases` - Lowercase ASCII phrases to remove.
///
/// # Returns
/// The remaining text, trimmed at the start.
fn strip_leading_phrases<'a>(text: &'a str, phrases: &[&str]) -> &'a str {
    let mut rest = text.trim_start();
    'outer: loop {
        for phrase in phrases {
            if let Some(after) = strip_prefix_ascii_case(rest, phrase) {
                rest =
                    after.trim_start_matches(|c: char| c.is_whitespace() || c == ':' || c == ',');
                continue 'outer;
            }
        }
        return rest;
    }
}

/// Removes a leading "out" only when it is a filler before dictated text.
///
/// # Arguments
/// * `text` - The text after the typing trigger.
///
/// # Returns
/// The text without "out" for "type out hello", but unchanged for "type out
/// of office", where "out" is part of the message.
fn strip_out_prefix(text: &str) -> &str {
    let Some(after) = strip_prefix_ascii_case(text, "out") else {
        return text;
    };
    let trimmed = after.trim_start_matches(|c: char| c.is_whitespace() || c == ':' || c == ',');
    if trimmed == "of" || trimmed.starts_with("of ") {
        return text;
    }
    trimmed
}

/// Strips an ASCII `suffix` from `text` when it starts on a word boundary.
///
/// # Arguments
/// * `text` - The text to inspect.
/// * `suffix` - A lowercase ASCII phrase.
///
/// # Returns
/// The text before the suffix, or `None` when it does not match.
fn strip_suffix_ascii_case<'a>(text: &'a str, suffix: &str) -> Option<&'a str> {
    let start = text.len().checked_sub(suffix.len())?;
    let tail = text.get(start..)?;
    if !tail.eq_ignore_ascii_case(suffix) {
        return None;
    }
    let head = text.get(..start)?;
    match head.chars().next_back() {
        Some(c) if c.is_whitespace() || c == ',' => Some(head),
        _ => None,
    }
}

/// Parses a typing command while preserving the spoken text's casing.
///
/// # Arguments
/// * `text` - The utterance after wake words and politeness phrases are
///   removed.
///
/// # Returns
/// A [`Intent::TypeText`] when the utterance starts with a typing trigger and
/// contains some text, otherwise `None`.
fn parse_type_text(text: &str) -> Option<Intent> {
    let rest = strip_trigger(text, &["type", "write", "dictate", "enter", "put"])?;
    let rest = strip_leading_phrases(rest, &["the word", "the text"]);
    let rest = strip_out_prefix(rest);
    let (text_part, field_part) = split_field(rest);
    let spoken = clean_typed_text(text_part);
    if spoken.is_empty() || is_bare_of_phrase(&spoken) {
        return None;
    }
    let field = field_part.and_then(clean_field);
    Some(Intent::TypeText {
        text: spoken,
        field,
    })
}

/// Reports whether dictated text is really the start of a noun phrase.
///
/// "type of blood" is ordinary speech, not a command. A user who really wants
/// those words can say "type the words of blood", which this check accepts.
///
/// # Arguments
/// * `text` - The text after the typing trigger.
///
/// # Returns
/// `true` when the text is exactly "of" or begins with "of ".
fn is_bare_of_phrase(text: &str) -> bool {
    let lower = text.to_lowercase();
    lower == "of" || lower.starts_with("of ")
}

/// Splits typed text from an optional "into the ... field" suffix.
///
/// # Arguments
/// * `text` - The text after the typing trigger.
///
/// # Returns
/// The text to type and, when the suffix really names a field, the field
/// phrase. A suffix such as "into the store" is left inside the text because
/// it does not end with a field noun.
fn split_field(text: &str) -> (&str, Option<&str>) {
    for marker in [" into the ", " into ", " in the ", " in "] {
        if let Some(index) = find_ascii_case_insensitive(text, marker) {
            let field_start = index.checked_add(marker.len()).unwrap_or(index);
            if let Some(field) = text.get(field_start..)
                && field_has_noun(field)
            {
                return (text.get(..index).unwrap_or(text), Some(field));
            }
        }
    }
    (text, None)
}

/// Reports whether the last word of `field` is a generic field noun.
///
/// # Arguments
/// * `field` - The candidate field phrase.
///
/// # Returns
/// `true` when the phrase ends with a word from [`FIELD_NOUNS`].
fn field_has_noun(field: &str) -> bool {
    tokenize(field)
        .last()
        .is_some_and(|last| FIELD_NOUNS.contains(&last.as_str()))
}

/// Cleans a field phrase into a spoken target.
///
/// # Arguments
/// * `text` - The field phrase, for example "the search box".
///
/// # Returns
/// The remaining words, or `None` when only an article or generic noun is
/// present.
fn clean_field(text: &str) -> Option<String> {
    clean_field_tokens(&tokenize_preserving_case(text))
}

/// Cleans already-tokenised field words into a spoken target.
///
/// # Arguments
/// * `tokens` - Cleaned field tokens. Their casing is preserved.
///
/// # Returns
/// The remaining words joined with spaces, or `None` when nothing is left.
fn clean_field_tokens(tokens: &[String]) -> Option<String> {
    let mut start = 0;
    if tokens.first().is_some_and(|token| {
        let lower = token.to_lowercase();
        matches!(lower.as_str(), "the" | "a" | "an")
    }) {
        start = 1;
    }
    let mut end = tokens.len();
    while end > start {
        let Some(last) = tokens.get(end - 1) else {
            break;
        };
        let lower = last.to_lowercase();
        if !FIELD_NOUNS.contains(&lower.as_str()) {
            break;
        }
        end -= 1;
    }
    if start >= end {
        return None;
    }
    let words = tokens.get(start..end)?;
    let stripped = strip_target_fillers(words);
    if stripped.is_empty() {
        return None;
    }
    Some(stripped.join(" "))
}

/// Cleans the text that will be typed.
///
/// # Arguments
/// * `text` - The spoken text after the typing trigger.
///
/// # Returns
/// The text with surrounding quotes and trailing politeness removed. Internal
/// casing and punctuation are preserved because they may be intentional.
fn clean_typed_text(text: &str) -> String {
    let mut cleaned = strip_surrounding_quotes(text.trim()).trim();
    cleaned = strip_leading_phrases(cleaned, &["please", "kindly", "just"]);
    loop {
        let mut changed = false;
        for suffix in ["please", "kindly", "for me", "now"] {
            if let Some(head) = strip_suffix_ascii_case(cleaned, suffix) {
                cleaned = head.trim_end_matches(|c: char| c.is_whitespace() || c == ',');
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    cleaned.trim().to_owned()
}

/// Removes one pair of matching quotes around `text`.
///
/// # Arguments
/// * `text` - The text to inspect.
///
/// # Returns
/// The text without its outer quote pair, or the original text when no pair
/// surrounds the whole value.
fn strip_surrounding_quotes(text: &str) -> &str {
    if let Some(inner) = text
        .strip_prefix('"')
        .and_then(|inner| inner.strip_suffix('"'))
    {
        return inner;
    }
    if let Some(inner) = text
        .strip_prefix('\'')
        .and_then(|inner| inner.strip_suffix('\''))
    {
        return inner;
    }
    if let Some(inner) = text
        .strip_prefix('\u{201c}')
        .and_then(|inner| inner.strip_suffix('\u{201d}'))
    {
        return inner;
    }
    if let Some(inner) = text
        .strip_prefix('\u{2018}')
        .and_then(|inner| inner.strip_suffix('\u{2019}'))
    {
        return inner;
    }
    text
}

/// Strips leading and trailing wake words and wake-word aliases.
///
/// Removes "reach", "rich", and "ridge" from the front and back of control tokens.
///
/// # Arguments
/// * `tokens` - Cleaned utterance tokens.
///
/// # Returns
/// A subslice of `tokens` with wake words removed.
fn strip_wake_tokens(mut tokens: &[String]) -> &[String] {
    loop {
        let mut stripped = false;
        for wake in ["reach", "rich", "ridge"] {
            if let Some(rest) = strip_phrase_prefix(tokens, &[wake]) {
                tokens = rest;
                stripped = true;
                break;
            }
            if let Some(rest) = strip_phrase_suffix(tokens, &[wake]) {
                tokens = rest;
                stripped = true;
                break;
            }
        }
        if !stripped {
            break;
        }
    }
    tokens
}

/// Parses yes/no/cancel/undo answers.
///
/// # Arguments
/// * `tokens` - Cleaned utterance tokens.
///
/// # Returns
/// The matching control intent, or `None` when the words are not an answer.
fn parse_control(tokens: &[String]) -> Option<Intent> {
    let tokens = strip_wake_tokens(tokens);
    let tokens = strip_trailing_politeness(tokens);
    let tokens = strip_phrase_suffix(tokens, &["a", "bit"])
        .or_else(|| strip_phrase_suffix(tokens, &["a", "little"]))
        .unwrap_or(tokens);
    let tokens = strip_trailing_politeness(tokens);
    match tokens {
        [word] => match word.as_str() {
            "yes" | "yeah" | "yep" | "sure" | "okay" | "ok" | "confirm" | "correct"
            | "affirmative" | "proceed" | "continue" => Some(Intent::Confirm),
            "no" | "nope" | "nah" | "deny" | "negative" | "dont" | "don't" => Some(Intent::Deny),
            "cancel" | "stop" | "abort" | "nevermind" => Some(Intent::Cancel),
            "undo" => Some(Intent::Undo),
            _ => None,
        },
        [first, second] => match (first.as_str(), second.as_str()) {
            ("do", "it") | ("go", "ahead") | ("that's", "right") => Some(Intent::Confirm),
            ("yes", "please") | ("yes", "confirm") | ("yes", "do") => Some(Intent::Confirm),
            ("no", "thanks") => Some(Intent::Deny),
            ("never", "mind") | ("forget", "it") | ("cancel", "that") => Some(Intent::Cancel),
            ("undo", "that") | ("undo", "last") => Some(Intent::Undo),
            _ => None,
        },
        [first, second, third] => match (first.as_str(), second.as_str(), third.as_str()) {
            ("yes", "do", "it") | ("yes", "go", "ahead") => Some(Intent::Confirm),
            ("no", "thank", "you") | ("don't", "do", "it") | ("dont", "do", "it") => {
                Some(Intent::Deny)
            }
            _ => None,
        },
        _ => None,
    }
}

/// Parses a listening-mode change.
///
/// # Arguments
/// * `tokens` - Cleaned utterance tokens.
///
/// # Returns
/// A [`Intent::SetMode`] when the words name a mode, otherwise `None`.
fn parse_mode(tokens: &[String]) -> Option<Intent> {
    let tokens = strip_trailing_politeness(tokens);
    let tokens = strip_phrase_suffix(tokens, &["a", "bit"])
        .or_else(|| strip_phrase_suffix(tokens, &["a", "little"]))
        .unwrap_or(tokens);
    let tokens = strip_trailing_politeness(tokens);
    let mode = mode_phrase(tokens)
        .or_else(|| strip_phrase_prefix(tokens, &["set", "mode", "to"]).and_then(mode_phrase))
        .or_else(|| strip_phrase_prefix(tokens, &["switch", "to"]).and_then(mode_phrase))
        .or_else(|| strip_phrase_prefix(tokens, &["start"]).and_then(mode_phrase))
        .or_else(|| strip_phrase_prefix(tokens, &["go", "to"]).and_then(mode_phrase))?;
    Some(Intent::SetMode { mode })
}

/// Maps a short phrase to a listening mode.
///
/// # Arguments
/// * `tokens` - Cleaned words that should name a mode.
///
/// # Returns
/// The mode, or `None` when the phrase is not a mode name.
fn mode_phrase(tokens: &[String]) -> Option<Mode> {
    match tokens {
        [word] => match word.as_str() {
            "command" => Some(Mode::Command),
            "dictation" => Some(Mode::Dictation),
            "sleep" => Some(Mode::Sleep),
            "wake" => Some(Mode::Command),
            _ => None,
        },
        [first, second] => match (first.as_str(), second.as_str()) {
            ("command", "mode") | ("wake", "up") | ("stop", "dictation") => Some(Mode::Command),
            ("dictation", "mode") => Some(Mode::Dictation),
            ("sleep", "mode") | ("stop", "listening") => Some(Mode::Sleep),
            _ => None,
        },
        [first, second, third] => match (first.as_str(), second.as_str(), third.as_str()) {
            ("go", "to", "sleep") => Some(Mode::Sleep),
            _ => None,
        },
        _ => None,
    }
}

/// Parses overlay and mouse-grid commands.
///
/// # Arguments
/// * `tokens` - Cleaned utterance tokens.
///
/// # Returns
/// The matching screen intent, or `None` when no screen command applies.
fn parse_screen(tokens: &[String]) -> Option<Intent> {
    let tokens = strip_trailing_politeness(tokens);
    let tokens = strip_phrase_suffix(tokens, &["a", "bit"])
        .or_else(|| strip_phrase_suffix(tokens, &["a", "little"]))
        .unwrap_or(tokens);
    let tokens = strip_trailing_politeness(tokens);
    if let Some(rest) = strip_phrase_prefix(tokens, &["show"]) {
        let rest = strip_phrase_prefix(rest, &["me"]).unwrap_or(rest);
        let rest = strip_phrase_prefix(rest, &["the"]).unwrap_or(rest);
        let rest = strip_phrase_suffix(rest, &["now"]).unwrap_or(rest);
        if is_phrase(rest, &["numbers"]) || is_phrase(rest, &["click", "numbers"]) {
            return Some(Intent::ShowNumbers);
        }
        if is_phrase(rest, &["grid"]) || is_phrase(rest, &["mouse", "grid"]) {
            return Some(Intent::ShowGrid);
        }
    }
    if let Some(rest) = strip_phrase_prefix(tokens, &["number"])
        && is_phrase(rest, &["the", "screen"])
    {
        return Some(Intent::ShowNumbers);
    }
    if let Some(rest) =
        strip_phrase_prefix(tokens, &["hide"]).or_else(|| strip_phrase_prefix(tokens, &["remove"]))
    {
        let rest = strip_phrase_prefix(rest, &["the"]).unwrap_or(rest);
        let rest = strip_phrase_suffix(rest, &["now"]).unwrap_or(rest);
        if is_phrase(rest, &["numbers"]) || is_phrase(rest, &["click", "numbers"]) {
            return Some(Intent::HideNumbers);
        }
    }
    if is_phrase(tokens, &["mouse", "grid"]) {
        return Some(Intent::ShowGrid);
    }
    for trigger in [
        &["grid"][..],
        &["cell"][..],
        &["select", "cell"][..],
        &["click", "cell"][..],
        &["choose", "cell"][..],
        &["grid", "cell"][..],
    ] {
        if let Some(rest) = strip_phrase_prefix(tokens, trigger) {
            let rest = strip_phrase_prefix(rest, &["number"]).unwrap_or(rest);
            let rest = strip_phrase_suffix(rest, &["now"]).unwrap_or(rest);
            let Some((cell, _)) = trailing_number(rest) else {
                continue;
            };
            if (1..=9).contains(&cell) {
                return Some(Intent::GridSelect { cell });
            }
        }
    }
    None
}

/// Parses a number or ordinal that fills the whole token slice.
///
/// # Arguments
/// * `tokens` - Cleaned words that should contain only a number.
///
/// # Returns
/// The value and consumed count when every token belongs to the number,
/// otherwise `None`.
fn trailing_number(tokens: &[String]) -> Option<(u32, usize)> {
    let tokens = strip_trailing_politeness(tokens);
    let tokens = strip_phrase_suffix(tokens, &["a", "bit"])
        .or_else(|| strip_phrase_suffix(tokens, &["a", "little"]))
        .unwrap_or(tokens);
    let tokens = strip_trailing_politeness(tokens);
    if let Some((value, consumed)) = parse_ordinal(tokens)
        && consumed == tokens.len()
    {
        return Some((value, consumed));
    }
    if let Some((value, consumed)) = parse_number(tokens)
        && consumed == tokens.len()
    {
        return Some((value, consumed));
    }
    None
}

/// Parses zoom commands.
///
/// # Arguments
/// * `tokens` - Cleaned utterance tokens.
///
/// # Returns
/// A [`Intent::Zoom`], or `None` when the words are not a zoom command.
fn parse_zoom(tokens: &[String]) -> Option<Intent> {
    let tokens = strip_trailing_politeness(tokens);
    let tokens = strip_phrase_suffix(tokens, &["a", "bit"])
        .or_else(|| strip_phrase_suffix(tokens, &["a", "little"]))
        .unwrap_or(tokens);
    let tokens = strip_trailing_politeness(tokens);
    if is_phrase(tokens, &["reset", "zoom"])
        || is_phrase(tokens, &["zoom", "reset"])
        || is_phrase(tokens, &["zoom", "normal"])
    {
        return Some(Intent::Zoom {
            direction: ZoomDir::Reset,
        });
    }
    if is_phrase(tokens, &["make", "it", "bigger"])
        || is_phrase(tokens, &["make", "the", "page", "bigger"])
        || is_phrase(tokens, &["make", "it", "larger"])
        || is_phrase(tokens, &["make", "the", "page", "larger"])
        || is_phrase(tokens, &["bigger"])
        || is_phrase(tokens, &["enlarge"])
        || is_phrase(tokens, &["enlarge", "it"])
        || is_phrase(tokens, &["enlarge", "the", "page"])
        || is_phrase(tokens, &["magnify"])
        || is_phrase(tokens, &["magnify", "it"])
        || is_phrase(tokens, &["magnify", "the", "page"])
        || is_phrase(tokens, &["bring", "it", "closer"])
    {
        return Some(Intent::Zoom {
            direction: ZoomDir::In,
        });
    }
    if is_phrase(tokens, &["make", "it", "smaller"])
        || is_phrase(tokens, &["make", "the", "page", "smaller"])
        || is_phrase(tokens, &["smaller"])
        || is_phrase(tokens, &["shrink"])
        || is_phrase(tokens, &["shrink", "it"])
        || is_phrase(tokens, &["shrink", "the", "page"])
    {
        return Some(Intent::Zoom {
            direction: ZoomDir::Out,
        });
    }
    let rest = strip_phrase_prefix(tokens, &["zoom"])?;
    let rest = strip_phrase_suffix(rest, &["now"]).unwrap_or(rest);
    let direction = match rest {
        [word] => match word.as_str() {
            "in" => ZoomDir::In,
            "out" => ZoomDir::Out,
            "reset" | "normal" => ZoomDir::Reset,
            _ => return None,
        },
        [first, second, third]
            if matches!(
                (first.as_str(), second.as_str(), third.as_str()),
                ("in", "a", "bit") | ("in", "a", "little")
            ) =>
        {
            ZoomDir::In
        }
        [first, second, third]
            if matches!(
                (first.as_str(), second.as_str(), third.as_str()),
                ("out", "a", "bit") | ("out", "a", "little")
            ) =>
        {
            ZoomDir::Out
        }
        _ => return None,
    };
    Some(Intent::Zoom { direction })
}

/// Parses scroll commands with a direction and optional bounded amount.
///
/// # Arguments
/// * `tokens` - Cleaned utterance tokens.
///
/// # Returns
/// A [`Intent::Scroll`], or `None` when the direction is missing or the
/// amount is invalid. Amounts above [`MAX_SCROLL_AMOUNT`] are rejected so a
/// misheard number cannot cause a very long scroll.
fn parse_scroll(tokens: &[String]) -> Option<Intent> {
    let tokens = strip_trailing_politeness(tokens);
    let (direction, rest) =
        if let Some(rest) = strip_phrase_prefix(tokens, &["scroll", "the", "page", "down"]) {
            (Direction::Down, rest)
        } else if let Some(rest) = strip_phrase_prefix(tokens, &["scroll", "the", "page", "up"]) {
            (Direction::Up, rest)
        } else if let Some(rest) = strip_phrase_prefix(tokens, &["push", "the", "page", "down"]) {
            (Direction::Down, rest)
        } else if let Some(rest) = strip_phrase_prefix(tokens, &["push", "the", "page", "up"]) {
            (Direction::Up, rest)
        } else if let Some(rest) = strip_phrase_prefix(tokens, &["scroll", "to", "the", "top"])
            .or_else(|| strip_phrase_prefix(tokens, &["scroll", "to", "top"]))
        {
            (Direction::Up, rest)
        } else if let Some(rest) = strip_phrase_prefix(tokens, &["scroll", "to", "the", "bottom"])
            .or_else(|| strip_phrase_prefix(tokens, &["scroll", "to", "bottom"]))
        {
            (Direction::Down, rest)
        } else if let Some(rest) = strip_phrase_prefix(tokens, &["scroll", "down"]) {
            (Direction::Down, rest)
        } else if let Some(rest) = strip_phrase_prefix(tokens, &["scroll", "up"]) {
            (Direction::Up, rest)
        } else if let Some(rest) = strip_phrase_prefix(tokens, &["scroll", "left"]) {
            (Direction::Left, rest)
        } else if let Some(rest) = strip_phrase_prefix(tokens, &["scroll", "right"]) {
            (Direction::Right, rest)
        } else if let Some(rest) = strip_phrase_prefix(tokens, &["go", "down"]) {
            (Direction::Down, rest)
        } else if let Some(rest) = strip_phrase_prefix(tokens, &["go", "up"]) {
            (Direction::Up, rest)
        } else if let Some(rest) = strip_phrase_prefix(tokens, &["move", "down"]) {
            (Direction::Down, rest)
        } else if let Some(rest) = strip_phrase_prefix(tokens, &["move", "up"]) {
            (Direction::Up, rest)
        } else if let Some(rest) = strip_phrase_prefix(tokens, &["scroll", "page", "down"]) {
            (Direction::Down, rest)
        } else if let Some(rest) = strip_phrase_prefix(tokens, &["scroll", "page", "up"]) {
            (Direction::Up, rest)
        } else if strip_phrase_prefix(tokens, &["down", "a", "bit"])
            .or_else(|| strip_phrase_prefix(tokens, &["down", "a", "little"]))
            .is_some()
        {
            return Some(Intent::Scroll {
                direction: Direction::Down,
                amount: Some(1),
            });
        } else if strip_phrase_prefix(tokens, &["up", "a", "bit"])
            .or_else(|| strip_phrase_prefix(tokens, &["up", "a", "little"]))
            .is_some()
        {
            return Some(Intent::Scroll {
                direction: Direction::Up,
                amount: Some(1),
            });
        } else {
            return None;
        };

    let rest = strip_trailing_politeness(rest);
    if rest.is_empty() {
        return Some(Intent::Scroll {
            direction,
            amount: None,
        });
    }
    if is_phrase(rest, &["the", "page"]) || is_phrase(rest, &["page"]) {
        return Some(Intent::Scroll {
            direction,
            amount: None,
        });
    }
    if is_phrase(rest, &["a", "bit"]) || is_phrase(rest, &["a", "little"]) {
        return Some(Intent::Scroll {
            direction,
            amount: Some(1),
        });
    }
    if is_phrase(rest, &["a", "lot"]) {
        return Some(Intent::Scroll {
            direction,
            amount: Some(5),
        });
    }

    let amount_tokens = strip_phrase_prefix(rest, &["by"]).unwrap_or(rest);
    if is_phrase(amount_tokens, &["a", "lot"]) {
        return Some(Intent::Scroll {
            direction,
            amount: Some(5),
        });
    }
    if is_phrase(amount_tokens, &["a", "bit"]) || is_phrase(amount_tokens, &["a", "little"]) {
        return Some(Intent::Scroll {
            direction,
            amount: Some(1),
        });
    }
    // Homophones are useful for numbered overlay clicks but dangerous here:
    // "scroll down to" must not scroll twice.
    if is_homophone(amount_tokens.first()) {
        return None;
    }
    if let Some((amount, consumed)) = parse_number(amount_tokens) {
        if amount == 0 || amount > MAX_SCROLL_AMOUNT {
            return None;
        }
        let trailing = amount_tokens.get(consumed..)?;
        if trailing.is_empty() || is_all_scroll_units(trailing) {
            return Some(Intent::Scroll {
                direction,
                amount: Some(amount),
            });
        }
    }
    None
}

/// Reports whether every token is a scroll unit word.
///
/// # Arguments
/// * `tokens` - Words that follow a scroll amount.
///
/// # Returns
/// `true` when the slice is non-empty and every word is in [`SCROLL_UNITS`].
fn is_all_scroll_units(tokens: &[String]) -> bool {
    !tokens.is_empty()
        && tokens
            .iter()
            .all(|token| SCROLL_UNITS.contains(&token.as_str()))
}

/// Parses focus commands.
///
/// # Arguments
/// * `tokens` - Cleaned utterance tokens.
///
/// # Returns
/// A [`Intent::FocusField`], or `None` when the words do not name a field.
fn parse_focus(tokens: &[String]) -> Option<Intent> {
    let tokens = strip_trailing_politeness(tokens);
    let (rest, require_noun) = if let Some(rest) = strip_phrase_prefix(tokens, &["focus"]) {
        (strip_phrase_prefix(rest, &["on"]).unwrap_or(rest), false)
    } else if let Some(rest) = strip_phrase_prefix(tokens, &["move", "to"]) {
        (rest, true)
    } else if let Some(rest) = strip_phrase_prefix(tokens, &["go", "to"]) {
        (rest, true)
    } else if let Some(rest) = strip_phrase_prefix(tokens, &["click", "on"]) {
        (rest, true)
    } else if let Some(rest) = strip_phrase_prefix(tokens, &["click"]) {
        (rest, true)
    } else if let Some(rest) = strip_phrase_prefix(tokens, &["select"]) {
        (rest, true)
    } else if let Some(rest) = strip_phrase_prefix(tokens, &["press"]) {
        (rest, true)
    } else if let Some(rest) = strip_phrase_prefix(tokens, &["hit"]) {
        (rest, true)
    } else {
        let rest = strip_phrase_prefix(tokens, &["tap"])?;
        (strip_phrase_prefix(rest, &["on"]).unwrap_or(rest), true)
    };

    let rest = strip_phrase_prefix(rest, &["the"])
        .or_else(|| strip_phrase_prefix(rest, &["a"]))
        .or_else(|| strip_phrase_prefix(rest, &["an"]))
        .unwrap_or(rest);
    let rest = strip_target_fillers(rest);
    if rest.is_empty() {
        return None;
    }
    if require_noun && !ends_with_field_noun(rest) {
        return None;
    }
    let target = clean_field_tokens(rest)?;
    Some(Intent::FocusField { target })
}

/// Reports whether the last token is a generic field noun.
///
/// # Arguments
/// * `tokens` - Cleaned words to inspect.
///
/// # Returns
/// `true` when the final word is in [`FIELD_NOUNS`].
fn ends_with_field_noun(tokens: &[String]) -> bool {
    tokens
        .last()
        .is_some_and(|last| FIELD_NOUNS.contains(&last.as_str()))
}

/// Parses window-switching commands.
///
/// # Arguments
/// * `tokens` - Cleaned utterance tokens.
///
/// # Returns
/// A [`Intent::SwitchWindow`] for a named window or for the next window, or
/// `None` when the words do not ask to switch. A request for the previous
/// window is rejected because the core contract cannot express it.
fn parse_switch(tokens: &[String]) -> Option<Intent> {
    let tokens = strip_trailing_politeness(tokens);
    let tokens = strip_phrase_suffix(tokens, &["a", "bit"])
        .or_else(|| strip_phrase_suffix(tokens, &["a", "little"]))
        .unwrap_or(tokens);
    let tokens = strip_trailing_politeness(tokens);
    if is_phrase(tokens, &["next", "window"]) {
        return Some(Intent::SwitchWindow { target: None });
    }
    if let Some(rest) = strip_phrase_prefix(tokens, &["switch"]) {
        let rest_after_to = strip_phrase_prefix(rest, &["to"]).unwrap_or(rest);
        let rest_after_the = strip_phrase_prefix(rest_after_to, &["the"]).unwrap_or(rest_after_to);
        if rest_after_the.is_empty() {
            return Some(Intent::Clarify {
                question: CLARIFY_QUESTION.to_owned(),
            });
        }
        if is_phrase(rest, &["window"])
            || is_phrase(rest, &["next", "window"])
            || is_phrase(rest_after_to, &["next", "window"])
            || is_phrase(rest_after_the, &["next", "window"])
        {
            return Some(Intent::SwitchWindow { target: None });
        }
        if is_phrase(rest, &["previous", "window"])
            || is_phrase(rest_after_to, &["previous", "window"])
            || is_phrase(rest_after_the, &["previous", "window"])
        {
            return None;
        }
        let target = strip_phrase_prefix(rest, &["window", "to"]).unwrap_or(rest);
        let target = strip_phrase_prefix(target, &["window"]).unwrap_or(target);
        let target = strip_phrase_prefix(target, &["to"]).unwrap_or(target);
        let target = strip_phrase_prefix(target, &["the"]).unwrap_or(target);
        let target = strip_target_fillers(target);
        let target = strip_phrase_suffix(target, &["window"]).unwrap_or(target);
        let target = strip_target_fillers(target);
        if target.is_empty() {
            return Some(Intent::Clarify {
                question: CLARIFY_QUESTION.to_owned(),
            });
        }
        return Some(Intent::SwitchWindow {
            target: Some(target.join(" ")),
        });
    }
    if let Some(rest) = strip_phrase_prefix(tokens, &["bring"]) {
        let target = strip_phrase_suffix(rest, &["to", "the", "front"])?;
        let target = strip_target_fillers(target);
        let target = strip_phrase_suffix(target, &["window"]).unwrap_or(target);
        let target = strip_target_fillers(target);
        if target.is_empty() {
            return Some(Intent::Clarify {
                question: CLARIFY_QUESTION.to_owned(),
            });
        }
        return Some(Intent::SwitchWindow {
            target: Some(target.join(" ")),
        });
    }
    if let Some(rest) = strip_phrase_prefix(tokens, &["activate"]) {
        let target = strip_target_fillers(rest);
        let target = strip_phrase_suffix(target, &["window"]).unwrap_or(target);
        let target = strip_target_fillers(target);
        if target.is_empty() {
            return Some(Intent::Clarify {
                question: CLARIFY_QUESTION.to_owned(),
            });
        }
        return Some(Intent::SwitchWindow {
            target: Some(target.join(" ")),
        });
    }
    if let Some(rest) = strip_phrase_prefix(tokens, &["go", "to"])
        && (is_phrase(rest, &["the", "next", "window"]) || is_phrase(rest, &["next", "window"]))
    {
        return Some(Intent::SwitchWindow { target: None });
    }
    None
}

/// Parses a key-press request, distinguishing unsupported keys from non-keys.
///
/// # Arguments
/// * `tokens` - Cleaned utterance tokens.
///
/// # Returns
/// A [`KeyAttempt`] describing a supported key, an unsupported key, or words
/// that are not a key request at all.
fn parse_key(tokens: &[String]) -> KeyAttempt {
    if let Some(key) = match_key(tokens) {
        return KeyAttempt::Key(key);
    }
    let Some(rest) = strip_phrase_prefix(tokens, &["press"])
        .or_else(|| strip_phrase_prefix(tokens, &["hit"]))
        .or_else(|| strip_phrase_prefix(tokens, &["tap"]))
        .or_else(|| strip_phrase_prefix(tokens, &["push"]))
    else {
        return KeyAttempt::NotAKey;
    };
    let rest = strip_phrase_prefix(rest, &["the"]).unwrap_or(rest);
    let rest = strip_phrase_suffix(rest, &["key"]).unwrap_or(rest);
    let rest = strip_phrase_suffix(rest, &["now"]).unwrap_or(rest);
    if rest.is_empty() {
        return KeyAttempt::NotAKey;
    }
    if let Some(key) = match_key(rest) {
        return KeyAttempt::Key(key);
    }
    if is_unsupported_key(rest) {
        return KeyAttempt::Unsupported;
    }
    KeyAttempt::NotAKey
}

/// Maps key words to a supported [`Key`].
///
/// # Arguments
/// * `tokens` - Cleaned words that should name one key.
///
/// # Returns
/// The key, or `None` when the words are not a supported key name.
fn match_key(tokens: &[String]) -> Option<Key> {
    match tokens {
        [word] => match word.as_str() {
            "enter" | "return" => Some(Key::Enter),
            "tab" => Some(Key::Tab),
            "escape" | "esc" => Some(Key::Escape),
            "backspace" => Some(Key::Backspace),
            "delete" | "del" => Some(Key::Delete),
            "up" => Some(Key::Up),
            "down" => Some(Key::Down),
            "left" => Some(Key::Left),
            "right" => Some(Key::Right),
            "home" => Some(Key::Home),
            "end" => Some(Key::End),
            "pageup" => Some(Key::PageUp),
            "pagedown" => Some(Key::PageDown),
            _ => None,
        },
        [first, second] => match (first.as_str(), second.as_str()) {
            ("page", "up") => Some(Key::PageUp),
            ("page", "down") => Some(Key::PageDown),
            ("up", "arrow") | ("arrow", "up") => Some(Key::Up),
            ("down", "arrow") | ("arrow", "down") => Some(Key::Down),
            ("left", "arrow") | ("arrow", "left") => Some(Key::Left),
            ("right", "arrow") | ("arrow", "right") => Some(Key::Right),
            ("back", "space") => Some(Key::Backspace),
            _ => None,
        },
        _ => None,
    }
}

/// Reports whether the words name a key the core contract cannot represent.
///
/// # Arguments
/// * `tokens` - Cleaned words after the key trigger.
///
/// # Returns
/// `true` for keys such as space, shift and function keys. These are rejected
/// rather than being mistaken for a click target.
fn is_unsupported_key(tokens: &[String]) -> bool {
    match tokens {
        [word] => {
            matches!(
                word.as_str(),
                "space"
                    | "shift"
                    | "control"
                    | "ctrl"
                    | "alt"
                    | "windows"
                    | "win"
                    | "caps"
                    | "insert"
                    | "ins"
                    | "print"
                    | "screen"
                    | "menu"
            ) || is_function_key(word)
        }
        [first, second] => matches!(
            (first.as_str(), second.as_str()),
            ("caps", "lock") | ("print", "screen") | ("function", "key")
        ),
        _ => false,
    }
}

/// Reports whether a token names a function key such as `f5`.
///
/// # Arguments
/// * `token` - A single lowercase token.
///
/// # Returns
/// `true` for `f1` through `f24`.
fn is_function_key(token: &str) -> bool {
    let Some(digits) = token.strip_prefix('f') else {
        return false;
    };
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return false;
    }
    digits
        .parse::<u8>()
        .is_ok_and(|number| (1..=24).contains(&number))
}

/// Parses a click command, including numbered overlay clicks.
///
/// # Arguments
/// * `tokens` - Cleaned utterance tokens.
///
/// # Returns
/// A [`ClickAttempt`] describing a valid click, an invalid target, or words
/// that are not a click command.
fn parse_click(tokens: &[String]) -> ClickAttempt {
    let tokens = strip_trailing_politeness(tokens);
    let Some(rest) = strip_phrase_prefix(tokens, &["click"])
        .or_else(|| strip_phrase_prefix(tokens, &["tap"]))
        .or_else(|| strip_phrase_prefix(tokens, &["press"]))
        .or_else(|| strip_phrase_prefix(tokens, &["hit"]))
        .or_else(|| strip_phrase_prefix(tokens, &["choose"]))
        .or_else(|| strip_phrase_prefix(tokens, &["select"]))
    else {
        return ClickAttempt::NotClick;
    };
    let rest = strip_phrase_prefix(rest, &["on"]).unwrap_or(rest);
    let rest = strip_phrase_prefix(rest, &["the"]).unwrap_or(rest);
    let rest = strip_phrase_prefix(rest, &["a"]).unwrap_or(rest);
    let rest = strip_phrase_prefix(rest, &["an"]).unwrap_or(rest);
    let rest = strip_target_fillers(rest);
    if rest.is_empty() {
        return ClickAttempt::Invalid;
    }

    match parse_click_number(rest) {
        NumberAttempt::Number(n) => return ClickAttempt::Click(Intent::ClickNumber { n }),
        NumberAttempt::Invalid => return ClickAttempt::Invalid,
        NumberAttempt::NotNumbered => {}
    }

    let mut target = rest;
    while target.len() > 1 {
        let Some(last) = target.last() else {
            break;
        };
        if !CLICK_NOUNS.contains(&last.as_str()) {
            break;
        }
        let Some(shorter) = target.get(..target.len() - 1) else {
            break;
        };
        target = shorter;
    }
    let target = strip_target_fillers(target);
    if target.is_empty() || is_phrase(target, &["number"]) {
        return ClickAttempt::Invalid;
    }
    if target.len() > MAX_CLICK_TARGET_WORDS {
        return ClickAttempt::Invalid;
    }
    let target = target.join(" ");
    if target.is_empty() {
        return ClickAttempt::Invalid;
    }
    ClickAttempt::Click(Intent::ClickElement { target })
}

/// Parses a numbered overlay click from the words after the click trigger.
///
/// # Arguments
/// * `tokens` - Cleaned words after the click trigger.
///
/// # Returns
/// A [`NumberAttempt`]. A number followed by extra words is invalid rather
/// than being treated as an element name, so "click 4 times" is clarified
/// instead of clicking an element called "4 times". Homophones such as "to"
/// only count as a number when they stand alone.
fn parse_click_number(tokens: &[String]) -> NumberAttempt {
    let (candidate, explicit_number) = if let Some(rest) = strip_phrase_prefix(tokens, &["number"])
    {
        let rest = strip_phrase_prefix(rest, &["the"]).unwrap_or(rest);
        (rest, true)
    } else if let Some(rest) = strip_phrase_prefix(tokens, &["item"]) {
        (rest, true)
    } else if let Some(rest) = strip_phrase_prefix(tokens, &["the"]) {
        (rest, true)
    } else {
        (tokens, false)
    };
    if candidate.is_empty() {
        return NumberAttempt::Invalid;
    }

    if let Some((value, consumed)) = parse_ordinal(candidate) {
        if consumed == candidate.len() {
            return if value == 0 {
                NumberAttempt::Invalid
            } else {
                NumberAttempt::Number(value)
            };
        }
        return NumberAttempt::Invalid;
    }
    if let Some((value, consumed)) = parse_number(candidate) {
        if consumed == candidate.len() {
            return if value == 0 {
                NumberAttempt::Invalid
            } else {
                NumberAttempt::Number(value)
            };
        }
        if !explicit_number && is_homophone(candidate.first()) {
            return NumberAttempt::NotNumbered;
        }
        return NumberAttempt::Invalid;
    }
    NumberAttempt::NotNumbered
}

/// Reports whether a token is a common homophone for a number.
///
/// # Arguments
/// * `token` - The optional first token of a numbered click.
///
/// # Returns
/// `true` for words such as "to", "for" and "ate".
fn is_homophone(token: Option<&String>) -> bool {
    token.is_some_and(|token| {
        matches!(
            token.as_str(),
            "to" | "too" | "for" | "fore" | "ate" | "won" | "oh"
        )
    })
}

/// Parses open/launch/visit commands into app or URL intents.
///
/// # Arguments
/// * `tokens` - Cleaned utterance tokens.
///
/// # Returns
/// An [`Intent::OpenUrl`] for a web address, an [`Intent::OpenApp`] for an
/// application name, or `None` when the target is empty or is a dotted name
/// that is not a web address. Non-web schemes are rejected.
fn parse_open(tokens: &[String]) -> Option<Intent> {
    let tokens = strip_trailing_politeness(tokens);
    let rest = strip_phrase_prefix(tokens, &["open", "up"])
        .or_else(|| strip_phrase_prefix(tokens, &["open"]))
        .or_else(|| strip_phrase_prefix(tokens, &["launch"]))
        .or_else(|| strip_phrase_prefix(tokens, &["start"]))
        .or_else(|| strip_phrase_prefix(tokens, &["run"]))
        .or_else(|| strip_phrase_prefix(tokens, &["visit"]))
        .or_else(|| strip_phrase_prefix(tokens, &["browse"]))
        .or_else(|| strip_phrase_prefix(tokens, &["go", "to"]))?;
    let rest = strip_phrase_prefix(rest, &["the"]).unwrap_or(rest);
    let rest = strip_phrase_prefix(rest, &["a"]).unwrap_or(rest);
    let rest = strip_phrase_prefix(rest, &["an"]).unwrap_or(rest);
    let rest = strip_phrase_prefix(rest, &["app"]).unwrap_or(rest);
    let rest = strip_phrase_prefix(rest, &["called"]).unwrap_or(rest);
    let rest = strip_target_fillers(rest);
    let rest = strip_phrase_suffix(rest, &["app"]).unwrap_or(rest);
    let rest = strip_phrase_suffix(rest, &["application"]).unwrap_or(rest);
    let rest = strip_phrase_suffix(rest, &["program"]).unwrap_or(rest);
    let rest = strip_target_fillers(rest);
    if rest.is_empty() {
        return None;
    }
    if is_phrase(rest, &["window"])
        || is_phrase(rest, &["next", "window"])
        || is_phrase(rest, &["previous", "window"])
    {
        return None;
    }

    let candidate = spoken_candidate(rest);
    if candidate.is_empty() || candidate.contains('@') {
        return None;
    }
    if candidate.ends_with(".exe") {
        let name = candidate.strip_suffix(".exe")?.trim();
        return allowed_app_name(name).map(|name| Intent::OpenApp { name });
    }
    if looks_like_web_url(&candidate) {
        return Some(Intent::OpenUrl { url: candidate });
    }
    if candidate.contains('.') || candidate.contains(':') {
        // A dotted local name or an explicit non-web scheme has no matching
        // intent in the core contract.
        return None;
    }

    let name = rest.join(" ");
    allowed_app_name(&name).map(|name| Intent::OpenApp { name })
}

/// Returns the canonical allow-listed application name.
///
/// # Arguments
/// * `name` - The spoken application name.
///
/// # Returns
/// The lowercase name when it is in [`ALLOWED_APPS`], otherwise `None`.
fn allowed_app_name(name: &str) -> Option<String> {
    let canonical = name.trim().to_lowercase();
    ALLOWED_APPS
        .contains(&canonical.as_str())
        .then_some(canonical)
}

/// Joins words into an address, expanding spoken "dot", "slash" and "colon".
///
/// # Arguments
/// * `tokens` - Cleaned words after the open trigger.
///
/// # Returns
/// The address with spoken punctuation converted and normal word spacing
/// kept elsewhere.
fn spoken_candidate(tokens: &[String]) -> String {
    let mut candidate = String::new();
    for token in tokens {
        match token.as_str() {
            "dot" => candidate.push('.'),
            "slash" => candidate.push('/'),
            "colon" => candidate.push(':'),
            _ => {
                if !candidate.is_empty()
                    && !candidate.ends_with('.')
                    && !candidate.ends_with('/')
                    && !candidate.ends_with(':')
                {
                    candidate.push(' ');
                }
                candidate.push_str(token);
            }
        }
    }
    candidate
}

/// Reports whether an address is a web URL the executor may open.
///
/// The check is deliberately strict. It accepts a bare `name.tld`, a
/// `www.` name, or an explicit `http`/`https` address, and then validates the
/// host labels and the characters in the path. Spaces, control characters,
/// empty labels, bad percent escapes and non-web schemes are rejected so a
/// loose spoken phrase cannot become an arbitrary URL.
///
/// # Arguments
/// * `candidate` - The joined address.
///
/// # Returns
/// `true` for a well-formed web address, `false` otherwise.
fn looks_like_web_url(candidate: &str) -> bool {
    if candidate.is_empty() || !candidate.is_ascii() {
        return false;
    }
    if candidate
        .chars()
        .any(|character| character.is_whitespace() || character.is_control())
    {
        return false;
    }
    let rest = if let Some(rest) = candidate.strip_prefix("https://") {
        rest
    } else if let Some(rest) = candidate.strip_prefix("http://") {
        rest
    } else if let Some(rest) = candidate.strip_prefix("www.") {
        rest
    } else {
        candidate
    };
    let (host, tail) = split_host_and_tail(rest);
    is_valid_host(host) && is_valid_url_tail(tail)
}

/// Splits an address into its host and the path, query or fragment.
///
/// # Arguments
/// * `address` - The address after any scheme or `www.` prefix is removed.
///
/// # Returns
/// The host and the remainder, which is empty when there is no path.
fn split_host_and_tail(address: &str) -> (&str, &str) {
    match address.find(['/', '?', '#']) {
        Some(index) => (
            address.get(..index).unwrap_or(address),
            address.get(index..).unwrap_or(""),
        ),
        None => (address, ""),
    }
}

/// Reports whether `host` is a valid dotted host name with a known web TLD.
///
/// # Arguments
/// * `host` - The host part of the address, without a scheme or path.
///
/// # Returns
/// `true` when every label is non-empty, hyphen-safe and the last label is in
/// [`WEB_TLDS`].
fn is_valid_host(host: &str) -> bool {
    if host.is_empty() || host.len() > 253 || !host.is_ascii() {
        return false;
    }
    let mut labels = 0;
    let mut last = "";
    for label in host.split('.') {
        if label.is_empty() || label.len() > 63 {
            return false;
        }
        if label.starts_with('-') || label.ends_with('-') {
            return false;
        }
        if !label
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        {
            return false;
        }
        labels += 1;
        last = label;
    }
    labels >= 2 && WEB_TLDS.iter().any(|tld| tld.eq_ignore_ascii_case(last))
}

/// Reports whether the path, query or fragment of an address is safe.
///
/// # Arguments
/// * `tail` - The text after the host, or an empty string.
///
/// # Returns
/// `true` when the tail is empty or starts with `/`, `?` or `#` and contains
/// only permitted URL characters and valid percent escapes.
fn is_valid_url_tail(tail: &str) -> bool {
    if tail.is_empty() {
        return true;
    }
    let Some(first) = tail.chars().next() else {
        return true;
    };
    if !matches!(first, '/' | '?' | '#') {
        return false;
    }
    let bytes = tail.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        let Some(byte) = bytes.get(index) else {
            return false;
        };
        if *byte == b'%' {
            let Some(high) = bytes.get(index + 1) else {
                return false;
            };
            let Some(low) = bytes.get(index + 2) else {
                return false;
            };
            if !high.is_ascii_hexdigit() || !low.is_ascii_hexdigit() {
                return false;
            }
            index += 3;
            continue;
        }
        if !is_url_tail_byte(*byte) {
            return false;
        }
        index += 1;
    }
    true
}

/// Reports whether `byte` may appear in a URL path, query or fragment.
///
/// # Arguments
/// * `byte` - One ASCII byte from the address tail.
///
/// # Returns
/// `true` for unreserved characters and common path separators. Colons and
/// backslashes are excluded so a path cannot look like another scheme.
fn is_url_tail_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric()
        || matches!(
            byte,
            b'-' | b'.'
                | b'_'
                | b'~'
                | b'/'
                | b'?'
                | b'&'
                | b'='
                | b'#'
                | b'+'
                | b'!'
                | b'$'
                | b'\''
                | b'('
                | b')'
                | b'*'
                | b','
                | b';'
                | b'@'
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Asserts that `utterance` parses to the intent encoded in `json`.
    fn assert_parses(utterance: &str, json: &str) {
        let expected: Intent = serde_json::from_str(json).expect("test intent JSON must be valid");
        assert_eq!(
            parse_rule(utterance),
            Some(expected),
            "utterance did not parse as expected: {utterance}"
        );
    }

    /// Asserts that `utterance` produces no rule match.
    fn assert_no_match(utterance: &str) {
        assert_eq!(
            parse_rule(utterance),
            None,
            "utterance should not have matched: {utterance}"
        );
    }

    /// Yes-like answers confirm the pending action.
    #[test]
    fn parses_confirmation() {
        assert_parses("yes", r#"{"action":"confirm"}"#);
        assert_parses("yes please", r#"{"action":"confirm"}"#);
        assert_parses("go ahead", r#"{"action":"confirm"}"#);
        assert_parses("reach yes", r#"{"action":"confirm"}"#);
        assert_parses("Reach, yes", r#"{"action":"confirm"}"#);
        assert_parses("reach, yes.", r#"{"action":"confirm"}"#);
        assert_parses("rich yes", r#"{"action":"confirm"}"#);
        assert_parses("ridge yes", r#"{"action":"confirm"}"#);
        assert_parses("yes reach", r#"{"action":"confirm"}"#);
        assert_parses("proceed", r#"{"action":"confirm"}"#);
        assert_parses("continue", r#"{"action":"confirm"}"#);
        assert_parses("yes do it", r#"{"action":"confirm"}"#);
        assert_parses("reach do it", r#"{"action":"confirm"}"#);
    }

    /// No-like answers deny the pending action.
    #[test]
    fn parses_denial() {
        assert_parses("no", r#"{"action":"deny"}"#);
        assert_parses("no thank you", r#"{"action":"deny"}"#);
        assert_parses("reach no", r#"{"action":"deny"}"#);
        assert_parses("Reach, no", r#"{"action":"deny"}"#);
        assert_parses("rich no", r#"{"action":"deny"}"#);
        assert_parses("no reach", r#"{"action":"deny"}"#);
    }

    /// Cancel and undo are separate from deny.
    #[test]
    fn parses_cancel_and_undo() {
        assert_parses("stop", r#"{"action":"cancel"}"#);
        assert_parses("never mind", r#"{"action":"cancel"}"#);
        assert_parses("undo that", r#"{"action":"undo"}"#);
    }

    /// Mode phrases switch listening behaviour.
    #[test]
    fn parses_mode_changes() {
        assert_parses(
            "switch to dictation",
            r#"{"action":"set_mode","mode":"dictation"}"#,
        );
        assert_parses("go to sleep", r#"{"action":"set_mode","mode":"sleep"}"#);
        assert_parses("wake up", r#"{"action":"set_mode","mode":"command"}"#);
    }

    /// Overlay and grid commands are recognised.
    #[test]
    fn parses_screen_commands() {
        assert_parses("show numbers", r#"{"action":"show_numbers"}"#);
        assert_parses("hide the numbers", r#"{"action":"hide_numbers"}"#);
        assert_parses("show the mouse grid", r#"{"action":"show_grid"}"#);
        assert_parses("select cell three", r#"{"action":"grid_select","cell":3}"#);
    }

    /// Zoom commands map to the three zoom directions.
    #[test]
    fn parses_zoom() {
        assert_parses("zoom in", r#"{"action":"zoom","direction":"in"}"#);
        assert_parses("zoom out", r#"{"action":"zoom","direction":"out"}"#);
        assert_parses("reset zoom", r#"{"action":"zoom","direction":"reset"}"#);
    }

    /// Scroll direction and optional amount are parsed together.
    #[test]
    fn parses_scroll() {
        assert_parses(
            "scroll down",
            r#"{"action":"scroll","direction":"down","amount":null}"#,
        );
        assert_parses(
            "scroll up three times",
            r#"{"action":"scroll","direction":"up","amount":3}"#,
        );
        assert_parses(
            "scroll to the top",
            r#"{"action":"scroll","direction":"up","amount":null}"#,
        );
    }

    /// Single key presses are recognised before the click fallback.
    #[test]
    fn parses_key_presses() {
        assert_parses("press enter", r#"{"action":"press_key","key":"enter"}"#);
        assert_parses(
            "press the page down key",
            r#"{"action":"press_key","key":"page_down"}"#,
        );
        assert_parses("press escape", r#"{"action":"press_key","key":"escape"}"#);
    }

    /// A known key name decides a key press; an unknown name is a click target.
    #[test]
    fn known_keys_beat_clicks() {
        assert_parses("press enter", r#"{"action":"press_key","key":"enter"}"#);
        assert_parses(
            "press submit",
            r#"{"action":"click_element","target":"submit"}"#,
        );
    }

    /// Keys that the core contract cannot represent are rejected.
    #[test]
    fn rejects_unsupported_keys() {
        assert_no_match("press space");
        assert_no_match("press control");
    }

    /// Numbered clicks accept digits, words and ordinals.
    #[test]
    fn parses_numbered_clicks() {
        assert_parses("click number 4", r#"{"action":"click_number","n":4}"#);
        assert_parses("click fourteen", r#"{"action":"click_number","n":14}"#);
        assert_parses("click 14", r#"{"action":"click_number","n":14}"#);
        assert_parses("tap number fourteen", r#"{"action":"click_number","n":14}"#);
        assert_parses(
            "click the fourteenth",
            r#"{"action":"click_number","n":14}"#,
        );
    }

    /// Out-of-range and ambiguous numeric clicks are rejected.
    #[test]
    fn rejects_invalid_numbered_clicks() {
        assert_no_match("click zero");
        assert_no_match("click number 0");
        assert_no_match("click 4 times");
        assert_no_match("click number");
    }

    /// Named clicks strip articles and generic element nouns.
    #[test]
    fn parses_named_clicks() {
        assert_parses(
            "click the submit button",
            r#"{"action":"click_element","target":"submit"}"#,
        );
        assert_parses(
            "click on the login link",
            r#"{"action":"click_element","target":"login"}"#,
        );
    }

    /// Named targets keep the casing the user spoke.
    #[test]
    fn preserves_target_casing() {
        assert_parses(
            "click Submit",
            r#"{"action":"click_element","target":"Submit"}"#,
        );
        assert_parses(
            "Um, click Submit.",
            r#"{"action":"click_element","target":"Submit"}"#,
        );
        assert_parses(
            "focus on the Email field",
            r#"{"action":"focus_field","target":"Email"}"#,
        );
        assert_parses(
            "switch to the Chrome window",
            r#"{"action":"switch_window","target":"Chrome"}"#,
        );
    }

    /// Long phrases are usually ambient speech, not element names.
    #[test]
    fn rejects_overlong_click_targets() {
        assert_no_match("please click on the link in the email I sent");
        assert_no_match("click the link in the email I sent");
        assert_parses(
            "click the save and continue button",
            r#"{"action":"click_element","target":"save and continue"}"#,
        );
    }

    /// Typing preserves the spoken text and extracts an optional field.
    #[test]
    fn parses_typing() {
        assert_parses(
            "type Hello World",
            r#"{"action":"type_text","text":"Hello World","field":null}"#,
        );
        assert_parses(
            "type hello into the search box",
            r#"{"action":"type_text","text":"hello","field":"search"}"#,
        );
        assert_parses(
            "type hello into the Search Box",
            r#"{"action":"type_text","text":"hello","field":"Search"}"#,
        );
    }

    /// "type of blood" is a noun phrase, not a dictation command.
    #[test]
    fn rejects_ambient_type_of_blood() {
        assert_no_match("type of blood");
        assert_no_match("type of");
    }

    /// Text containing "into the" is not mistaken for a field.
    #[test]
    fn keeps_into_inside_typed_text() {
        assert_parses(
            "type I went into the store",
            r#"{"action":"type_text","text":"I went into the store","field":null}"#,
        );
    }

    /// Focus commands strip generic field nouns.
    #[test]
    fn parses_focus() {
        assert_parses(
            "focus on the email field",
            r#"{"action":"focus_field","target":"email"}"#,
        );
    }

    /// Web addresses become URL intents and app names become app intents.
    #[test]
    fn parses_open_commands() {
        assert_parses(
            "open google dot com",
            r#"{"action":"open_url","url":"google.com"}"#,
        );
        assert_parses(
            "open the chrome app",
            r#"{"action":"open_app","name":"chrome"}"#,
        );
        assert_parses(
            "open explorer",
            r#"{"action":"open_app","name":"explorer"}"#,
        );
    }

    /// Only allow-listed application names are accepted by the rule parser.
    #[test]
    fn rejects_unknown_apps() {
        assert_no_match("open calculator");
        assert_no_match("open spotify");
        assert_no_match("open the window it's hot");
        assert_no_match("open the door");
    }

    /// Non-web schemes are rejected rather than handed to the executor.
    #[test]
    fn rejects_non_web_urls() {
        assert_no_match("open file:///C:/Windows");
        assert_no_match("open javascript:alert(1)");
    }

    /// Loose dotted phrases and addresses with spaces are not URLs.
    #[test]
    fn rejects_loose_web_addresses() {
        assert_no_match("open hello world dot com");
        assert_no_match("open example dot com slash hello world");
        assert_no_match("open foo dot com colon 8080");
        assert_no_match("open example dot com slash javascript colon alert(1)");
        assert_no_match("open https://example.com/path with space");
    }

    /// Window switching distinguishes a named window from the next one.
    #[test]
    fn parses_window_switching() {
        assert_parses(
            "switch to the chrome window",
            r#"{"action":"switch_window","target":"chrome"}"#,
        );
        assert_parses("next window", r#"{"action":"switch_window","target":null}"#);
    }

    /// Wake words and politeness phrases are ignored.
    #[test]
    fn strips_wake_words_and_politeness() {
        assert_parses(
            "Hey Reach, click submit",
            r#"{"action":"click_element","target":"submit"}"#,
        );
        assert_parses(
            "Could you scroll down please?",
            r#"{"action":"scroll","direction":"down","amount":null}"#,
        );
    }

    /// Empty and unrelated utterances do not match.
    #[test]
    fn rejects_empty_and_unrelated_utterances() {
        assert_no_match("");
        assert_no_match("   ");
        assert_no_match("banana");
        assert_no_match("hello there");
    }

    /// Extreme inputs are rejected without panicking or splitting a character.
    #[test]
    fn handles_extreme_input_without_panicking() {
        let long_ascii = "a".repeat(10_000);
        let long_unicode = "é".repeat(10_000);
        let long_command = format!("click {}", "word ".repeat(5_000));
        let inputs = [
            "",
            "   ",
            "\t\n",
            long_ascii.as_str(),
            long_unicode.as_str(),
            long_command.as_str(),
            "🙂🙂",
            "é",
            "Ｒｅａｃｈ",
        ];
        for input in inputs {
            let _ = parse_rule(input);
        }
    }

    /// Builds owned tokens for a short test phrase.
    fn tokens(words: &[&str]) -> Vec<String> {
        words.iter().map(|word| (*word).to_owned()).collect()
    }

    /// Prefix stripping requires a word boundary.
    #[test]
    fn strips_prefixes_only_on_word_boundaries() {
        assert_eq!(
            strip_prefix_ascii_case("Reach, click", "reach"),
            Some(", click")
        );
        assert_eq!(strip_prefix_ascii_case("reach", "reach"), Some(""));
        assert_eq!(strip_prefix_ascii_case("reachable", "reach"), None);
    }

    /// Token cleaning keeps punctuation that is inside a token.
    #[test]
    fn cleans_tokens_without_losing_internal_punctuation() {
        assert_eq!(clean_token("OK."), "ok");
        assert_eq!(clean_token("(hello)"), "hello");
        assert_eq!(
            clean_token("https://example.com/path?q=1,"),
            "https://example.com/path?q=1"
        );
        assert_eq!(clean_token("don't"), "don't");
    }

    /// Case-insensitive phrase search reports byte positions safely.
    #[test]
    fn finds_phrases_case_insensitively() {
        assert_eq!(
            find_ascii_case_insensitive("Type Hello into the box", " into "),
            Some(10)
        );
        assert_eq!(find_ascii_case_insensitive("hello", "zzz"), None);
    }

    /// "out" is only stripped when it is a filler before typed text.
    #[test]
    fn strips_out_only_when_it_is_a_filler() {
        assert_eq!(strip_out_prefix("out hello"), "hello");
        assert_eq!(strip_out_prefix("out of office"), "out of office");
        assert_eq!(strip_out_prefix("outlook"), "outlook");
    }

    /// Typed phrases that begin with "out of" keep every word.
    #[test]
    fn types_out_of_phrase() {
        assert_parses(
            "type out of office",
            r#"{"action":"type_text","text":"out of office","field":null}"#,
        );
    }

    /// Scroll amounts never use number homophones.
    #[test]
    fn rejects_scroll_homophones() {
        assert_no_match("scroll down to");
        assert_no_match("scroll down for");
        assert_no_match("scroll down won");
    }

    /// Scroll amounts are capped before they reach the executor.
    #[test]
    fn caps_scroll_amounts() {
        assert_parses(
            "scroll down 50",
            r#"{"action":"scroll","direction":"down","amount":50}"#,
        );
        assert_no_match("scroll down 51");
    }

    /// Dotted executable names are apps, but other dotted names are rejected.
    #[test]
    fn parses_dotted_app_names() {
        assert_parses(
            "open notepad.exe",
            r#"{"action":"open_app","name":"notepad"}"#,
        );
        assert_no_match("open readme.txt");
        assert_no_match("open someone@example.com");
    }

    /// The core contract cannot express a previous-window request.
    #[test]
    fn rejects_previous_window_requests() {
        assert_no_match("switch to the previous window");
        assert_no_match("go to the previous window");
    }

    /// A trailing "now" is ignored by every command family.
    #[test]
    fn accepts_trailing_now() {
        assert_parses("sleep now", r#"{"action":"set_mode","mode":"sleep"}"#);
        assert_parses("show grid now", r#"{"action":"show_grid"}"#);
        assert_parses(
            "make it bigger now",
            r#"{"action":"zoom","direction":"in"}"#,
        );
        assert_parses(
            "switch to chrome now",
            r#"{"action":"switch_window","target":"chrome"}"#,
        );
    }

    /// Field cleaning removes articles and generic field nouns.
    #[test]
    fn strips_field_nouns() {
        assert_eq!(clean_field("the search box"), Some("search".to_owned()));
        assert_eq!(clean_field("the field"), None);
    }

    /// A trailing number must consume every word to be accepted.
    #[test]
    fn trailing_number_requires_full_consumption() {
        assert_eq!(trailing_number(&tokens(&["twenty", "one"])), Some((21, 2)));
        assert_eq!(trailing_number(&tokens(&["4", "times"])), None);
    }

    /// Function keys are recognised so they can be rejected explicitly.
    #[test]
    fn recognises_function_keys() {
        assert!(is_function_key("f5"));
        assert!(is_function_key("f24"));
        assert!(!is_function_key("f25"));
        assert!(!is_function_key("five"));
    }

    /// Spoken URL punctuation is joined without stray spaces.
    #[test]
    fn builds_spoken_urls() {
        assert_eq!(
            spoken_candidate(&tokens(&["google", "dot", "com", "slash", "maps"])),
            "google.com/maps"
        );
        assert_eq!(
            spoken_candidate(&tokens(&[
                "http", "colon", "slash", "slash", "example", "dot", "com"
            ])),
            "http://example.com"
        );
    }

    /// Only web-like addresses are accepted as URLs.
    #[test]
    fn recognises_web_addresses() {
        assert!(looks_like_web_url("example.com"));
        assert!(looks_like_web_url("www.example.com"));
        assert!(looks_like_web_url("https://example.com/path"));
        assert!(looks_like_web_url("example.com/path?q=1"));
        assert!(looks_like_web_url("https://example.com/path%20name"));
        assert!(!looks_like_web_url("file:///C:/Windows"));
        assert!(!looks_like_web_url("readme.txt"));
        assert!(!looks_like_web_url(".com"));
        assert!(!looks_like_web_url("hello world.com"));
        assert!(!looks_like_web_url("example..com"));
        assert!(!looks_like_web_url("-example.com"));
        assert!(!looks_like_web_url("https://example.com:8080"));
        assert!(!looks_like_web_url("https://example.com/%zz"));
    }

    /// Ambient sentences that must be present in the shared corpus.
    const REQUIRED_AMBIENT: &[&str] = &[
        "the volume of sales went up",
        "I'll call you in five minutes",
        "I can't reach the top shelf",
        "let's start the meeting",
        "please click on the link in the email I sent",
        "scroll through the report when you have time",
        "he pressed the button yesterday",
        "open the window it's hot",
        "zoom is down again",
        "type of blood",
        "tell me a joke",
        "what is the capital of France",
        "hello how are you today",
    ];

    /// Loads the shared utterance corpus.
    fn load_corpus() -> Vec<serde_json::Value> {
        serde_json::from_str(include_str!("../../../tests/utterances.json"))
            .expect("corpus JSON must be valid")
    }

    /// Returns the stable id for a corpus case, or its 1-based position.
    fn corpus_case_id(case: &serde_json::Value, index: usize) -> String {
        case.get("id")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| format!("case-{}", index + 1))
    }

    /// Reports whether a corpus case is marked as ambient speech.
    fn is_ambient(case: &serde_json::Value) -> bool {
        case.get("ambient")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
    }

    /// Formats one corpus failure with its id, text, expected and actual intent.
    fn format_corpus_failure(
        case_id: &str,
        utterance: &str,
        expected: &Option<Intent>,
        actual: &Option<Intent>,
    ) -> String {
        format!("{case_id}: {utterance:?}: expected {expected:?}, got {actual:?}")
    }

    /// The shared corpus must contain at least one hundred cases.
    #[test]
    fn utterance_corpus_has_enough_cases() {
        let cases = load_corpus();
        assert!(
            cases.len() >= 100,
            "corpus must hold at least 100 cases, found {}",
            cases.len()
        );
    }

    /// The corpus holds at least 25 ambient sentences, all expecting `None`.
    #[test]
    fn ambient_corpus_has_at_least_25_cases() {
        let cases = load_corpus();
        let ambient: Vec<_> = cases
            .iter()
            .enumerate()
            .filter(|(_, case)| is_ambient(case))
            .collect();
        assert!(
            ambient.len() >= 25,
            "corpus must hold at least 25 ambient cases, found {}",
            ambient.len()
        );
        for (index, case) in ambient {
            let case_id = corpus_case_id(case, index);
            let expected = case
                .get("intent")
                .expect("every ambient case needs an intent field");
            assert!(
                expected.is_null(),
                "{case_id}: ambient case must expect null, got {expected}"
            );
        }
    }

    /// Every named ambient sentence from the audit is present in the corpus.
    #[test]
    fn required_ambient_sentences_are_present() {
        let cases = load_corpus();
        let utterances: Vec<&str> = cases
            .iter()
            .filter(|case| is_ambient(case))
            .filter_map(|case| case.get("utterance").and_then(serde_json::Value::as_str))
            .collect();
        for required in REQUIRED_AMBIENT {
            assert!(
                utterances.contains(required),
                "missing ambient case: {required}"
            );
        }
    }

    /// Ambient speech never produces an intent, and failures name the case.
    #[test]
    fn ambient_corpus_returns_no_intent() {
        let cases = load_corpus();
        let mut failures = Vec::new();
        for (index, case) in cases.iter().enumerate() {
            if !is_ambient(case) {
                continue;
            }
            let case_id = corpus_case_id(case, index);
            let utterance = case
                .get("utterance")
                .and_then(serde_json::Value::as_str)
                .expect("every ambient case needs an utterance");
            if let Some(intent) = parse_rule(utterance) {
                failures.push(format_corpus_failure(
                    &case_id,
                    utterance,
                    &None,
                    &Some(intent),
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "{} ambient false positives:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }

    /// Failure text includes the case id and the spoken text.
    #[test]
    fn corpus_failure_message_includes_case_id_and_text() {
        let actual = Some(Intent::ClickElement {
            target: "link".to_owned(),
        });
        let message =
            format_corpus_failure("ambient-05", "please click on the link", &None, &actual);
        assert!(message.contains("ambient-05"));
        assert!(message.contains("please click on the link"));
    }

    /// Every case in the shared corpus parses as expected.
    #[test]
    fn utterance_corpus_matches_rule_grammar() {
        let cases = load_corpus();
        let mut failures = Vec::new();
        for (index, case) in cases.iter().enumerate() {
            let case_id = corpus_case_id(case, index);
            let utterance = case
                .get("utterance")
                .and_then(serde_json::Value::as_str)
                .expect("every corpus case needs an utterance");
            let expected_value = case
                .get("intent")
                .expect("every corpus case needs an intent field");
            let expected = if expected_value.is_null() {
                None
            } else {
                Some(
                    serde_json::from_value::<Intent>(expected_value.clone())
                        .expect("corpus intents must deserialise"),
                )
            };
            let actual = parse_rule(utterance);
            if actual != expected {
                failures.push(format_corpus_failure(
                    &case_id, utterance, &expected, &actual,
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "{} corpus failures:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }

    /// Forgiving rule grammar parses synonyms, filler words, and number words.
    #[test]
    fn forgiving_synonyms_fillers_and_numbers() {
        // Zoom in synonyms (9 cases)
        assert_parses("make it bigger", r#"{"action":"zoom","direction":"in"}"#);
        assert_parses(
            "make the page bigger",
            r#"{"action":"zoom","direction":"in"}"#,
        );
        assert_parses("bigger", r#"{"action":"zoom","direction":"in"}"#);
        assert_parses("enlarge", r#"{"action":"zoom","direction":"in"}"#);
        assert_parses("magnify", r#"{"action":"zoom","direction":"in"}"#);
        assert_parses("bring it closer", r#"{"action":"zoom","direction":"in"}"#);
        assert_parses(
            "please make the page bigger",
            r#"{"action":"zoom","direction":"in"}"#,
        );
        assert_parses(
            "can you enlarge for me",
            r#"{"action":"zoom","direction":"in"}"#,
        );
        assert_parses(
            "magnify the page please",
            r#"{"action":"zoom","direction":"in"}"#,
        );

        // Zoom out synonyms (6 cases)
        assert_parses("make it smaller", r#"{"action":"zoom","direction":"out"}"#);
        assert_parses(
            "make the page smaller",
            r#"{"action":"zoom","direction":"out"}"#,
        );
        assert_parses("smaller", r#"{"action":"zoom","direction":"out"}"#);
        assert_parses("shrink", r#"{"action":"zoom","direction":"out"}"#);
        assert_parses(
            "would you shrink the page please",
            r#"{"action":"zoom","direction":"out"}"#,
        );
        assert_parses("shrink for me", r#"{"action":"zoom","direction":"out"}"#);

        // Scroll down synonyms and amount words (12 cases)
        assert_parses(
            "go down",
            r#"{"action":"scroll","direction":"down","amount":null}"#,
        );
        assert_parses(
            "move down",
            r#"{"action":"scroll","direction":"down","amount":null}"#,
        );
        assert_parses(
            "scroll page down",
            r#"{"action":"scroll","direction":"down","amount":null}"#,
        );
        assert_parses(
            "push the page down",
            r#"{"action":"scroll","direction":"down","amount":null}"#,
        );
        assert_parses(
            "scroll the page down",
            r#"{"action":"scroll","direction":"down","amount":null}"#,
        );
        assert_parses(
            "down a bit",
            r#"{"action":"scroll","direction":"down","amount":1}"#,
        );
        assert_parses(
            "scroll down a little",
            r#"{"action":"scroll","direction":"down","amount":1}"#,
        );
        assert_parses(
            "scroll down a lot",
            r#"{"action":"scroll","direction":"down","amount":5}"#,
        );
        assert_parses(
            "scroll down ten",
            r#"{"action":"scroll","direction":"down","amount":10}"#,
        );
        assert_parses(
            "scroll down twenty percent",
            r#"{"action":"scroll","direction":"down","amount":20}"#,
        );
        assert_parses(
            "could you please go down a bit",
            r#"{"action":"scroll","direction":"down","amount":1}"#,
        );
        assert_parses(
            "just move down a lot",
            r#"{"action":"scroll","direction":"down","amount":5}"#,
        );

        // Scroll up synonyms and amount words (9 cases)
        assert_parses(
            "go up",
            r#"{"action":"scroll","direction":"up","amount":null}"#,
        );
        assert_parses(
            "move up",
            r#"{"action":"scroll","direction":"up","amount":null}"#,
        );
        assert_parses(
            "scroll page up",
            r#"{"action":"scroll","direction":"up","amount":null}"#,
        );
        assert_parses(
            "push the page up",
            r#"{"action":"scroll","direction":"up","amount":null}"#,
        );
        assert_parses(
            "scroll the page up",
            r#"{"action":"scroll","direction":"up","amount":null}"#,
        );
        assert_parses(
            "up a bit",
            r#"{"action":"scroll","direction":"up","amount":1}"#,
        );
        assert_parses(
            "scroll up a lot",
            r#"{"action":"scroll","direction":"up","amount":5}"#,
        );
        assert_parses(
            "scroll up twenty percent",
            r#"{"action":"scroll","direction":"up","amount":20}"#,
        );
        assert_parses(
            "can you push the page up for me",
            r#"{"action":"scroll","direction":"up","amount":null}"#,
        );

        // Open app synonyms (6 cases)
        assert_parses("launch chrome", r#"{"action":"open_app","name":"chrome"}"#);
        assert_parses("start notepad", r#"{"action":"open_app","name":"notepad"}"#);
        assert_parses("open up chrome", r#"{"action":"open_app","name":"chrome"}"#);
        assert_parses("run explorer", r#"{"action":"open_app","name":"explorer"}"#);
        assert_parses(
            "go ahead and launch notepad please",
            r#"{"action":"open_app","name":"notepad"}"#,
        );
        assert_parses(
            "please run chrome for me",
            r#"{"action":"open_app","name":"chrome"}"#,
        );

        // Click synonyms (9 cases)
        assert_parses(
            "press submit",
            r#"{"action":"click_element","target":"submit"}"#,
        );
        assert_parses(
            "hit submit",
            r#"{"action":"click_element","target":"submit"}"#,
        );
        assert_parses(
            "tap submit",
            r#"{"action":"click_element","target":"submit"}"#,
        );
        assert_parses(
            "select submit",
            r#"{"action":"click_element","target":"submit"}"#,
        );
        assert_parses(
            "click submit for me",
            r#"{"action":"click_element","target":"submit"}"#,
        );
        assert_parses(
            "press the save button please",
            r#"{"action":"click_element","target":"save"}"#,
        );
        assert_parses(
            "hit the submit button now",
            r#"{"action":"click_element","target":"submit"}"#,
        );
        assert_parses(
            "click the banana",
            r#"{"action":"click_element","target":"banana"}"#,
        );
        assert_parses(
            "just tap the login button",
            r#"{"action":"click_element","target":"login"}"#,
        );

        // Focus field synonyms (7 cases)
        assert_parses(
            "go to the email field",
            r#"{"action":"focus_field","target":"email"}"#,
        );
        assert_parses(
            "select the username box",
            r#"{"action":"focus_field","target":"username"}"#,
        );
        assert_parses(
            "focus search",
            r#"{"action":"focus_field","target":"search"}"#,
        );
        assert_parses(
            "click on the search field",
            r#"{"action":"focus_field","target":"search"}"#,
        );
        assert_parses(
            "click on the full name field",
            r#"{"action":"focus_field","target":"full name"}"#,
        );
        assert_parses(
            "select the search box for me",
            r#"{"action":"focus_field","target":"search"}"#,
        );
        assert_parses(
            "click the username field",
            r#"{"action":"focus_field","target":"username"}"#,
        );

        // Type text synonyms (6 cases)
        assert_parses(
            "type Jinwoo",
            r#"{"action":"type_text","text":"Jinwoo","field":null}"#,
        );
        assert_parses(
            "write Reach rocks",
            r#"{"action":"type_text","text":"Reach rocks","field":null}"#,
        );
        assert_parses(
            "enter Jinwoo Sung",
            r#"{"action":"type_text","text":"Jinwoo Sung","field":null}"#,
        );
        assert_parses(
            "put Jinwoo in the full name field",
            r#"{"action":"type_text","text":"Jinwoo","field":"full name"}"#,
        );
        assert_parses(
            "put Hello in the search box",
            r#"{"action":"type_text","text":"Hello","field":"search"}"#,
        );
        assert_parses(
            "enter test into the email field",
            r#"{"action":"type_text","text":"test","field":"email"}"#,
        );

        // Switch window exact cases and empty targets (5 cases)
        assert_parses(
            "switch window to banana",
            r#"{"action":"switch_window","target":"banana"}"#,
        );
        assert_parses(
            "switch to chrome",
            r#"{"action":"switch_window","target":"chrome"}"#,
        );
        assert_parses(
            "switch window to the banana",
            r#"{"action":"switch_window","target":"banana"}"#,
        );
        assert_parses(
            "switch to",
            r#"{"action":"clarify","question":"I didn't catch that, could you say it another way?"}"#,
        );
        assert_parses(
            "switch window to",
            r#"{"action":"clarify","question":"I didn't catch that, could you say it another way?"}"#,
        );
    }

    /// Negative cases that must not match a command.
    #[test]
    fn negative_utterances_do_not_match_command() {
        assert_no_match("what's the weather");
        assert_no_match("I'll call you in five minutes");
        assert_no_match("the volume of sales went up");
        assert_no_match("hello how are you");
        assert_no_match("Okay reach dance good");
        assert_no_match("good morning reach");
        assert_no_match("why is the sky blue");
        assert_no_match("tell me a joke");
        assert_no_match("how do I bake a cake");
        assert_no_match("this is not a command");
        assert_no_match("can you believe what happened yesterday");
        assert_no_match("the price went down by ten dollars");
        assert_no_match("we need to zoom our camera");
        assert_no_match("she loves to read books");
        assert_no_match("open the door for me");
        assert_no_match("I want to eat an apple");
    }

    /// Synonyms never produce Confirm, Deny or Cancel.
    #[test]
    fn synonyms_never_produce_confirm_or_deny() {
        let commands = [
            "bigger",
            "smaller",
            "enlarge",
            "shrink",
            "go down",
            "go up",
            "launch chrome",
            "press submit",
            "focus search",
        ];
        for cmd in commands {
            let res = parse_rule(cmd);
            assert_ne!(res, Some(Intent::Confirm));
            assert_ne!(res, Some(Intent::Deny));
            assert_ne!(res, Some(Intent::Cancel));
        }
    }
}
