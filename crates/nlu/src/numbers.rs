//! Spoken number and ordinal parsing for the rule grammar.
//!
//! Speech recognisers frequently return number words instead of digits, and
//! small words are easily confused ("to" for "two", "for" for "four"). This
//! module keeps that handling in one place so the command grammar can treat
//! numbers uniformly.

/// The largest value the word-based number grammar accepts.
///
/// Spoken commands use small numbers ("click fourteen", "scroll down three").
/// Capping word-based parsing keeps a misheard sentence from becoming a huge
/// scroll count, while digit input is still parsed separately.
const MAX_WORD_NUMBER: u32 = 999;

/// Maps one cardinal word, including common homophones, to its value.
///
/// # Arguments
/// * `word` - A single lowercase token.
///
/// # Returns
/// The numeric value, or `None` when the token is not a number word.
fn word_value(word: &str) -> Option<u32> {
    Some(match word {
        "zero" | "oh" => 0,
        "one" | "won" => 1,
        "two" | "to" | "too" => 2,
        "three" => 3,
        "four" | "for" | "fore" => 4,
        "five" => 5,
        "six" => 6,
        "seven" => 7,
        "eight" | "ate" => 8,
        "nine" => 9,
        "ten" => 10,
        "eleven" => 11,
        "twelve" => 12,
        "thirteen" => 13,
        "fourteen" => 14,
        "fifteen" => 15,
        "sixteen" => 16,
        "seventeen" => 17,
        "eighteen" => 18,
        "nineteen" => 19,
        "twenty" => 20,
        "thirty" => 30,
        "forty" => 40,
        "fifty" => 50,
        "sixty" => 60,
        "seventy" => 70,
        "eighty" => 80,
        "ninety" => 90,
        _ => return None,
    })
}

/// Parses a token made only of ASCII digits.
///
/// # Arguments
/// * `token` - A single lowercase token.
///
/// # Returns
/// The parsed value, or `None` when the token is not a valid `u32`.
fn digit_value(token: &str) -> Option<u32> {
    if token.is_empty() || !token.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    token.parse().ok()
}

/// Maps one ordinal word to its value.
///
/// # Arguments
/// * `word` - A single lowercase token.
///
/// # Returns
/// The numeric value, or `None` when the token is not an ordinal word.
fn ordinal_value(word: &str) -> Option<u32> {
    Some(match word {
        "first" => 1,
        "second" => 2,
        "third" => 3,
        "fourth" => 4,
        "fifth" => 5,
        "sixth" => 6,
        "seventh" => 7,
        "eighth" => 8,
        "ninth" => 9,
        "tenth" => 10,
        "eleventh" => 11,
        "twelfth" => 12,
        "thirteenth" => 13,
        "fourteenth" => 14,
        "fifteenth" => 15,
        "sixteenth" => 16,
        "seventeenth" => 17,
        "eighteenth" => 18,
        "nineteenth" => 19,
        "twentieth" => 20,
        "thirtieth" => 30,
        "fortieth" => 40,
        "fiftieth" => 50,
        "sixtieth" => 60,
        "seventieth" => 70,
        "eightieth" => 80,
        "ninetieth" => 90,
        "hundredth" => 100,
        _ => return None,
    })
}

/// Parses an ordinal written as digits with a suffix, such as `4th`.
///
/// # Arguments
/// * `token` - A single lowercase token.
///
/// # Returns
/// The numeric value when both the digits and the suffix are valid, otherwise
/// `None`. Suffixes are checked so that "2th" is rejected rather than guessed.
fn ordinal_digit_value(token: &str) -> Option<u32> {
    let (digits, suffix) = token
        .strip_suffix("st")
        .map(|digits| (digits, "st"))
        .or_else(|| token.strip_suffix("nd").map(|digits| (digits, "nd")))
        .or_else(|| token.strip_suffix("rd").map(|digits| (digits, "rd")))
        .or_else(|| token.strip_suffix("th").map(|digits| (digits, "th")))?;
    let value = digit_value(digits)?;
    let expected = if (11..=13).contains(&(value % 100)) {
        "th"
    } else {
        match value % 10 {
            1 => "st",
            2 => "nd",
            3 => "rd",
            _ => "th",
        }
    };
    (suffix == expected).then_some(value)
}

/// Parses a number at the start of `tokens`.
///
/// # Arguments
/// * `tokens` - Lowercase words from an utterance, already cleaned of
///   punctuation and filler words.
///
/// # Returns
/// The value and the number of tokens consumed, or `None` when the first
/// token is not a number word or digit. A successful parse may consume fewer
/// tokens than `tokens` contains, so callers must check the consumed count
/// before treating the rest of the utterance as empty.
pub fn parse_number(tokens: &[String]) -> Option<(u32, usize)> {
    let first = tokens.first()?;
    if let Some(value) = digit_value(first) {
        return Some((value, 1));
    }

    let mut index = 0;
    let mut current: u32 = 0;
    let mut saw_value = false;
    let mut saw_hundred = false;

    // "a hundred" is a common spoken form of "one hundred".
    if first == "a" {
        if tokens.get(1).map(String::as_str) == Some("hundred") {
            index = 1;
        } else {
            return None;
        }
    }

    while let Some(token) = tokens.get(index) {
        if token == "and" {
            let next_is_number = tokens
                .get(index + 1)
                .and_then(|next| word_value(next))
                .is_some();
            if !saw_hundred || !next_is_number {
                break;
            }
            index += 1;
            continue;
        }
        if token == "hundred" {
            if current == 0 {
                current = 1;
            }
            current = current.checked_mul(100)?;
            saw_value = true;
            saw_hundred = true;
            index += 1;
            continue;
        }
        let Some(value) = word_value(token) else {
            break;
        };
        if value >= 20 && value % 10 == 0 {
            if !current.is_multiple_of(100) {
                break;
            }
            current = current.checked_add(value)?;
        } else if value < 20 {
            if !current.is_multiple_of(10) {
                break;
            }
            current = current.checked_add(value)?;
        } else {
            break;
        }
        saw_value = true;
        index += 1;
    }

    if !saw_value || current > MAX_WORD_NUMBER {
        return None;
    }
    Some((current, index))
}

/// Parses an ordinal at the start of `tokens`.
///
/// # Arguments
/// * `tokens` - Lowercase words from an utterance, already cleaned of
///   punctuation and filler words.
///
/// # Returns
/// The value and the number of tokens consumed, or `None` when the first
/// token is not an ordinal word or a digit with an ordinal suffix.
pub fn parse_ordinal(tokens: &[String]) -> Option<(u32, usize)> {
    let first = tokens.first()?;

    if let Some(value) = ordinal_digit_value(first) {
        return Some((value, 1));
    }

    // Hyphenated compounds such as "twenty-first" arrive as one token.
    if let Some((tens, unit)) = first.split_once('-') {
        let tens = word_value(tens)?;
        let unit = ordinal_value(unit)?;
        if tens >= 20 && tens % 10 == 0 && (1..=9).contains(&unit) {
            return Some((tens + unit, 1));
        }
        return None;
    }

    if let Some(value) = ordinal_value(first) {
        return Some((value, 1));
    }

    // Spoken compounds such as "twenty first" arrive as two tokens.
    let tens = word_value(first)?;
    if tens < 20 || tens % 10 != 0 {
        return None;
    }
    let unit = ordinal_value(tokens.get(1)?)?;
    if !(1..=9).contains(&unit) {
        return None;
    }
    Some((tens + unit, 2))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds owned tokens from a short test phrase.
    fn tokens(words: &[&str]) -> Vec<String> {
        words.iter().map(|word| (*word).to_owned()).collect()
    }

    /// Digits are accepted as a single token.
    #[test]
    fn parses_digits() {
        assert_eq!(parse_number(&tokens(&["14"])), Some((14, 1)));
        assert_eq!(parse_number(&tokens(&["0"])), Some((0, 1)));
    }

    /// Simple number words from zero through nineteen are accepted.
    #[test]
    fn parses_simple_words() {
        assert_eq!(parse_number(&tokens(&["zero"])), Some((0, 1)));
        assert_eq!(parse_number(&tokens(&["one"])), Some((1, 1)));
        assert_eq!(parse_number(&tokens(&["fourteen"])), Some((14, 1)));
        assert_eq!(parse_number(&tokens(&["nineteen"])), Some((19, 1)));
    }

    /// Tens and units combine into a single value.
    #[test]
    fn parses_compound_tens() {
        assert_eq!(parse_number(&tokens(&["twenty"])), Some((20, 1)));
        assert_eq!(parse_number(&tokens(&["twenty", "one"])), Some((21, 2)));
        assert_eq!(parse_number(&tokens(&["ninety", "nine"])), Some((99, 2)));
    }

    /// Hundreds combine with tens and units, with or without "and".
    #[test]
    fn parses_hundreds() {
        assert_eq!(parse_number(&tokens(&["hundred"])), Some((100, 1)));
        assert_eq!(parse_number(&tokens(&["a", "hundred"])), Some((100, 2)));
        assert_eq!(
            parse_number(&tokens(&["one", "hundred", "and", "five"])),
            Some((105, 4))
        );
        assert_eq!(
            parse_number(&tokens(&["nine", "hundred", "ninety", "nine"])),
            Some((999, 4))
        );
    }

    /// Common speech-recognition homophones map to their intended numbers.
    #[test]
    fn parses_homophones() {
        assert_eq!(parse_number(&tokens(&["to"])), Some((2, 1)));
        assert_eq!(parse_number(&tokens(&["too"])), Some((2, 1)));
        assert_eq!(parse_number(&tokens(&["for"])), Some((4, 1)));
        assert_eq!(parse_number(&tokens(&["fore"])), Some((4, 1)));
        assert_eq!(parse_number(&tokens(&["ate"])), Some((8, 1)));
        assert_eq!(parse_number(&tokens(&["won"])), Some((1, 1)));
    }

    /// Parsing stops at the first token that is not part of the number.
    #[test]
    fn reports_consumed_tokens() {
        assert_eq!(
            parse_number(&tokens(&["twenty", "one", "steps"])),
            Some((21, 2))
        );
        assert_eq!(
            parse_number(&tokens(&["five", "times", "please"])),
            Some((5, 1))
        );
    }

    /// A word that is not a number produces no value.
    #[test]
    fn rejects_unknown_words() {
        assert_eq!(parse_number(&tokens(&["banana"])), None);
        assert_eq!(parse_number(&[]), None);
    }

    /// Repeated hundreds are rejected instead of overflowing the counter.
    #[test]
    fn rejects_overflowing_word_numbers() {
        assert_eq!(
            parse_number(&tokens(&[
                "hundred", "hundred", "hundred", "hundred", "hundred"
            ])),
            None
        );
    }

    /// Ordinal words are accepted.
    #[test]
    fn parses_ordinal_words() {
        assert_eq!(parse_ordinal(&tokens(&["first"])), Some((1, 1)));
        assert_eq!(parse_ordinal(&tokens(&["fourth"])), Some((4, 1)));
        assert_eq!(parse_ordinal(&tokens(&["fourteenth"])), Some((14, 1)));
        assert_eq!(parse_ordinal(&tokens(&["twentieth"])), Some((20, 1)));
    }

    /// Digits with an ordinal suffix are accepted.
    #[test]
    fn parses_ordinal_digits() {
        assert_eq!(parse_ordinal(&tokens(&["1st"])), Some((1, 1)));
        assert_eq!(parse_ordinal(&tokens(&["2nd"])), Some((2, 1)));
        assert_eq!(parse_ordinal(&tokens(&["3rd"])), Some((3, 1)));
        assert_eq!(parse_ordinal(&tokens(&["4th"])), Some((4, 1)));
        assert_eq!(parse_ordinal(&tokens(&["21st"])), Some((21, 1)));
    }

    /// A cardinal tens word followed by an ordinal unit forms a compound.
    #[test]
    fn parses_compound_ordinals() {
        assert_eq!(parse_ordinal(&tokens(&["twenty", "first"])), Some((21, 2)));
        assert_eq!(parse_ordinal(&tokens(&["ninety", "ninth"])), Some((99, 2)));
    }

    /// An ordinal parse consumes only the ordinal tokens.
    #[test]
    fn ordinal_reports_consumed_tokens() {
        assert_eq!(
            parse_ordinal(&tokens(&["twenty", "first", "item"])),
            Some((21, 2))
        );
    }

    /// Non-ordinal words are rejected.
    #[test]
    fn rejects_unknown_ordinals() {
        assert_eq!(parse_ordinal(&tokens(&["banana"])), None);
        assert_eq!(parse_ordinal(&tokens(&["twenty"])), None);
    }
}
