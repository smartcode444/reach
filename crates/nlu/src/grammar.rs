//! The fast-path rule grammar.
//!
//! The grammar recognises common command phrasing before the Needle model is
//! consulted. It is deliberately conservative: when a phrase is ambiguous or
//! the target is unclear it returns `None`, which lets the caller ask for a
//! clarification instead of guessing.

use reach_core::{Direction, Intent, Key, Mode, ZoomDir};

use crate::numbers::{parse_number, parse_ordinal};

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
    "go ahead and",
    "can you",
    "could you",
    "would you",
    "will you",
    "please",
    "kindly",
    "reach",
];

/// Generic nouns removed from the end of a named click target.
const CLICK_NOUNS: &[&str] = &["button", "link", "icon"];

/// Generic nouns removed from the end of a field name.
const FIELD_NOUNS: &[&str] = &["field", "box", "input", "textarea", "area"];

/// Words that may follow a scroll amount, such as "times" or "pages".
const SCROLL_UNITS: &[&str] = &[
    "time", "times", "step", "steps", "page", "pages", "notch", "notches",
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
        return Some(intent);
    }
    if let Some(intent) = parse_switch(&tokens) {
        return Some(intent);
    }

    match parse_key(&tokens) {
        KeyAttempt::Key(key) => return Some(Intent::PressKey { key }),
        KeyAttempt::Unsupported => return None,
        KeyAttempt::NotAKey => {}
    }

    match parse_click(&tokens) {
        ClickAttempt::Click(intent) => return Some(intent),
        ClickAttempt::Invalid => return None,
        ClickAttempt::NotClick => {}
    }

    parse_open(&tokens)
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

/// Cleans one token for matching.
///
/// # Arguments
/// * `token` - One whitespace-separated word.
///
/// # Returns
/// The lowercase token with typographic quotes normalised and leading or
/// trailing ASCII punctuation removed.
fn clean_token(token: &str) -> String {
    let lowered = token
        .to_lowercase()
        .replace(['\u{2018}', '\u{2019}'], "'")
        .replace(['\u{201c}', '\u{201d}'], "\"");
    lowered
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
    let rest = strip_trigger(text, &["type", "write", "dictate"])?;
    let rest = strip_leading_phrases(rest, &["the word", "the text"]);
    let rest = strip_out_prefix(rest);
    let (text_part, field_part) = split_field(rest);
    let spoken = clean_typed_text(text_part);
    if spoken.is_empty() {
        return None;
    }
    let field = field_part.and_then(clean_field);
    Some(Intent::TypeText {
        text: spoken,
        field,
    })
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
    for marker in [" into ", " in the "] {
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
    clean_field_tokens(&tokenize(text))
}

/// Cleans already-tokenised field words into a spoken target.
///
/// # Arguments
/// * `tokens` - Cleaned field tokens.
///
/// # Returns
/// The remaining words joined with spaces, or `None` when nothing is left.
fn clean_field_tokens(tokens: &[String]) -> Option<String> {
    let mut start = 0;
    if tokens
        .first()
        .is_some_and(|token| matches!(token.as_str(), "the" | "a" | "an"))
    {
        start = 1;
    }
    let mut end = tokens.len();
    while end > start {
        let Some(last) = tokens.get(end - 1) else {
            break;
        };
        if !FIELD_NOUNS.contains(&last.as_str()) {
            break;
        }
        end -= 1;
    }
    if start >= end {
        return None;
    }
    tokens.get(start..end).map(|words| words.join(" "))
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
    cleaned = strip_leading_phrases(cleaned, &["please", "kindly"]);
    loop {
        let mut changed = false;
        for suffix in ["please", "kindly"] {
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

/// Parses yes/no/cancel/undo answers.
///
/// # Arguments
/// * `tokens` - Cleaned utterance tokens.
///
/// # Returns
/// The matching control intent, or `None` when the words are not an answer.
fn parse_control(tokens: &[String]) -> Option<Intent> {
    let tokens = strip_phrase_suffix(tokens, &["now"]).unwrap_or(tokens);
    match tokens {
        [word] => match word.as_str() {
            "yes" | "yeah" | "yep" | "sure" | "okay" | "ok" | "confirm" | "correct"
            | "affirmative" => Some(Intent::Confirm),
            "no" | "nope" | "nah" | "deny" | "negative" => Some(Intent::Deny),
            "cancel" | "stop" | "abort" | "nevermind" => Some(Intent::Cancel),
            "undo" => Some(Intent::Undo),
            _ => None,
        },
        [first, second] => match (first.as_str(), second.as_str()) {
            ("do", "it") | ("go", "ahead") | ("that's", "right") => Some(Intent::Confirm),
            ("no", "thanks") => Some(Intent::Deny),
            ("never", "mind") | ("forget", "it") | ("cancel", "that") => Some(Intent::Cancel),
            ("undo", "that") | ("undo", "last") => Some(Intent::Undo),
            _ => None,
        },
        [first, second, third] => match (first.as_str(), second.as_str(), third.as_str()) {
            ("no", "thank", "you") => Some(Intent::Deny),
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
    let tokens = strip_phrase_suffix(tokens, &["now"]).unwrap_or(tokens);
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
    let tokens = strip_phrase_suffix(tokens, &["now"]).unwrap_or(tokens);
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
    let tokens = strip_phrase_suffix(tokens, &["now"]).unwrap_or(tokens);
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
    let tokens = strip_phrase_suffix(tokens, &["now"]).unwrap_or(tokens);
    if is_phrase(tokens, &["reset", "zoom"]) {
        return Some(Intent::Zoom {
            direction: ZoomDir::Reset,
        });
    }
    if is_phrase(tokens, &["make", "it", "bigger"]) || is_phrase(tokens, &["make", "it", "larger"])
    {
        return Some(Intent::Zoom {
            direction: ZoomDir::In,
        });
    }
    if is_phrase(tokens, &["make", "it", "smaller"]) {
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
    let rest = strip_phrase_prefix(tokens, &["scroll"])?;
    let rest = strip_phrase_suffix(rest, &["now"]).unwrap_or(rest);
    let (direction, rest) = if let Some(rest) = strip_phrase_prefix(rest, &["to", "the", "top"]) {
        (Direction::Up, rest)
    } else if let Some(rest) = strip_phrase_prefix(rest, &["to", "the", "bottom"]) {
        (Direction::Down, rest)
    } else if let Some(rest) = strip_phrase_prefix(rest, &["to", "top"]) {
        (Direction::Up, rest)
    } else if let Some(rest) = strip_phrase_prefix(rest, &["to", "bottom"]) {
        (Direction::Down, rest)
    } else if let Some(rest) = strip_phrase_prefix(rest, &["page", "down"]) {
        (Direction::Down, rest)
    } else if let Some(rest) = strip_phrase_prefix(rest, &["page", "up"]) {
        (Direction::Up, rest)
    } else if let Some(rest) = strip_phrase_prefix(rest, &["down"]) {
        (Direction::Down, rest)
    } else if let Some(rest) = strip_phrase_prefix(rest, &["up"]) {
        (Direction::Up, rest)
    } else if let Some(rest) = strip_phrase_prefix(rest, &["left"]) {
        (Direction::Left, rest)
    } else {
        let rest = strip_phrase_prefix(rest, &["right"])?;
        (Direction::Right, rest)
    };

    let rest = strip_phrase_suffix(rest, &["now"]).unwrap_or(rest);
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

    let amount_tokens = strip_phrase_prefix(rest, &["by"]).unwrap_or(rest);
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
    let (rest, require_noun) = if let Some(rest) = strip_phrase_prefix(tokens, &["focus"]) {
        (strip_phrase_prefix(rest, &["on"]).unwrap_or(rest), false)
    } else if let Some(rest) = strip_phrase_prefix(tokens, &["move", "to"]) {
        (rest, true)
    } else {
        let rest = strip_phrase_prefix(tokens, &["go", "to"])?;
        (rest, true)
    };

    let rest = strip_phrase_prefix(rest, &["the"])
        .or_else(|| strip_phrase_prefix(rest, &["a"]))
        .or_else(|| strip_phrase_prefix(rest, &["an"]))
        .unwrap_or(rest);
    let rest = strip_phrase_suffix(rest, &["now"]).unwrap_or(rest);
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
    let tokens = strip_phrase_suffix(tokens, &["now"]).unwrap_or(tokens);
    if is_phrase(tokens, &["next", "window"]) {
        return Some(Intent::SwitchWindow { target: None });
    }
    if let Some(rest) = strip_phrase_prefix(tokens, &["switch"]) {
        let rest = strip_phrase_prefix(rest, &["to"]).unwrap_or(rest);
        let rest = strip_phrase_prefix(rest, &["the"]).unwrap_or(rest);
        if rest.is_empty() {
            return None;
        }
        if is_phrase(rest, &["window"]) || is_phrase(rest, &["next", "window"]) {
            return Some(Intent::SwitchWindow { target: None });
        }
        if is_phrase(rest, &["previous", "window"]) {
            return None;
        }
        let target = strip_phrase_suffix(rest, &["window"]).unwrap_or(rest);
        let target = strip_phrase_prefix(target, &["the"]).unwrap_or(target);
        if target.is_empty() {
            return None;
        }
        return Some(Intent::SwitchWindow {
            target: Some(target.join(" ")),
        });
    }
    if let Some(rest) = strip_phrase_prefix(tokens, &["bring"]) {
        let target = strip_phrase_suffix(rest, &["to", "the", "front"])?;
        let target = strip_phrase_prefix(target, &["the"]).unwrap_or(target);
        let target = strip_phrase_suffix(target, &["window"]).unwrap_or(target);
        if target.is_empty() {
            return None;
        }
        return Some(Intent::SwitchWindow {
            target: Some(target.join(" ")),
        });
    }
    if let Some(rest) = strip_phrase_prefix(tokens, &["activate"]) {
        let target = strip_phrase_prefix(rest, &["the"]).unwrap_or(rest);
        if target.is_empty() {
            return None;
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
    let rest = strip_phrase_suffix(rest, &["now"]).unwrap_or(rest);
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
    if target.is_empty() || is_phrase(target, &["number"]) {
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
    let rest = strip_phrase_prefix(tokens, &["open"])
        .or_else(|| strip_phrase_prefix(tokens, &["launch"]))
        .or_else(|| strip_phrase_prefix(tokens, &["start"]))
        .or_else(|| strip_phrase_prefix(tokens, &["visit"]))
        .or_else(|| strip_phrase_prefix(tokens, &["browse"]))
        .or_else(|| strip_phrase_prefix(tokens, &["go", "to"]))?;
    let rest = strip_phrase_prefix(rest, &["the"]).unwrap_or(rest);
    let rest = strip_phrase_prefix(rest, &["a"]).unwrap_or(rest);
    let rest = strip_phrase_prefix(rest, &["an"]).unwrap_or(rest);
    let rest = strip_phrase_prefix(rest, &["app"]).unwrap_or(rest);
    let rest = strip_phrase_prefix(rest, &["called"]).unwrap_or(rest);
    let rest = strip_phrase_suffix(rest, &["app"]).unwrap_or(rest);
    let rest = strip_phrase_suffix(rest, &["application"]).unwrap_or(rest);
    let rest = strip_phrase_suffix(rest, &["program"]).unwrap_or(rest);
    let rest = strip_phrase_suffix(rest, &["now"]).unwrap_or(rest);
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
        if name.is_empty() {
            return None;
        }
        return Some(Intent::OpenApp {
            name: name.to_owned(),
        });
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
    if name.is_empty() {
        None
    } else {
        Some(Intent::OpenApp { name })
    }
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
/// # Arguments
/// * `candidate` - The joined address.
///
/// # Returns
/// `true` for `http`/`https` addresses, `www.` names and bare names whose
/// last label is in [`WEB_TLDS`].
fn looks_like_web_url(candidate: &str) -> bool {
    if candidate.starts_with("http://") {
        return candidate.len() > "http://".len();
    }
    if candidate.starts_with("https://") {
        return candidate.len() > "https://".len();
    }
    if candidate.starts_with("www.") {
        return candidate.len() > "www.".len();
    }
    let host = candidate
        .split('/')
        .next()
        .unwrap_or("")
        .split('?')
        .next()
        .unwrap_or("")
        .split('#')
        .next()
        .unwrap_or("");
    let first = host.split('.').next().unwrap_or("");
    let last = host.rsplit('.').next().unwrap_or("");
    !first.is_empty() && !last.is_empty() && host.contains('.') && WEB_TLDS.contains(&last)
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
    }

    /// No-like answers deny the pending action.
    #[test]
    fn parses_denial() {
        assert_parses("no", r#"{"action":"deny"}"#);
        assert_parses("no thank you", r#"{"action":"deny"}"#);
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
    }

    /// Non-web schemes are rejected rather than handed to the executor.
    #[test]
    fn rejects_non_web_urls() {
        assert_no_match("open file:///C:/Windows");
        assert_no_match("open javascript:alert(1)");
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
        assert!(!looks_like_web_url("file:///C:/Windows"));
        assert!(!looks_like_web_url("readme.txt"));
        assert!(!looks_like_web_url(".com"));
    }

    /// The shared corpus must contain at least one hundred cases.
    #[test]
    fn utterance_corpus_has_enough_cases() {
        let cases: Vec<serde_json::Value> =
            serde_json::from_str(include_str!("../../../tests/utterances.json"))
                .expect("corpus JSON must be valid");
        assert!(
            cases.len() >= 100,
            "corpus must hold at least 100 cases, found {}",
            cases.len()
        );
    }

    /// Every case in the shared corpus parses as expected.
    #[test]
    fn utterance_corpus_matches_rule_grammar() {
        let cases: Vec<serde_json::Value> =
            serde_json::from_str(include_str!("../../../tests/utterances.json"))
                .expect("corpus JSON must be valid");
        let mut failures = Vec::new();
        for case in cases {
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
                failures.push(format!(
                    "{utterance:?}: expected {expected:?}, got {actual:?}"
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
}
