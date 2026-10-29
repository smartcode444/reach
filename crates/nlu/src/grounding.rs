//! Lexical grounding for model calls.
//!
//! Needle can name a declared tool for speech that is not a command. Grounding
//! checks that the utterance contains at least one word or phrase associated
//! with the tool the model chose, and that every string or numeric argument the
//! model filled in was actually spoken. It is a cheap filter, not proof of
//! intent: the wake gate and the target resolver still have to do their part.

use reach_core::Intent;

use crate::numbers::{parse_number, parse_ordinal};
use crate::tools::NO_ACTION;

/// Words that ground each exposed tool in the user's utterance.
///
/// This table is deliberately separate from the trigger tables in
/// [`crate::grammar`]. The grammar's tables drive parsing and case restoration;
/// this table only decides whether a model call is supported by the words the
/// user actually said. The two may drift, and that is acceptable: the coverage
/// test below catches a tool with no trigger, and the utterance corpus catches
/// a missing synonym. Keeping them separate means a model-side synonym can be
/// added without changing the rule parser.
///
/// A trigger is matched as a whole word or a contiguous phrase, ignoring case
/// and punctuation. Short, generic words are avoided on purpose: "up" must not
/// ground a scroll, and "window" must not ground a window switch, because those
/// words appear in ordinary speech.
///
/// Cardinal number words are deliberately absent from `click_number`: "I'll
/// call you in five minutes" is ordinary speech. The click verbs ground the
/// tool, and [`arguments_are_grounded`] then requires the spoken number itself,
/// so "click fourteen" grounds while "click the big one" does not.
const TOOL_TRIGGERS: &[(&str, &[&str])] = &[
    // The router handles `no_action` before grounding; this entry keeps the
    // coverage test honest if that order ever changes.
    (NO_ACTION, &["nothing", "never mind", "no action"]),
    (
        "click_element",
        &["click", "tap", "press", "hit", "choose", "select"],
    ),
    (
        "click_number",
        &[
            "click", "tap", "choose", "select", "number", "item", "first", "second", "third",
            "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
        ],
    ),
    ("type_text", &["type", "write", "dictate"]),
    ("focus_field", &["focus", "move to", "go to", "tab to"]),
    ("press_key", &["press", "push", "keyboard"]),
    (
        "scroll",
        &[
            "scroll",
            "page up",
            "page down",
            "top of the page",
            "bottom of the page",
        ],
    ),
    (
        "zoom",
        &[
            "zoom",
            "magnify",
            "make it bigger",
            "make it larger",
            "make it smaller",
        ],
    ),
    ("open_app", &["open", "launch", "start", "run"]),
    (
        "switch_window",
        &[
            "switch",
            "bring to the front",
            "activate",
            "next window",
            "previous window",
        ],
    ),
    (
        "show_numbers",
        &[
            "show numbers",
            "show the numbers",
            "display numbers",
            "numbered overlay",
            "numbers on screen",
        ],
    ),
    (
        "hide_numbers",
        &[
            "hide numbers",
            "hide the numbers",
            "remove numbers",
            "remove the numbers",
            "hide overlay",
            "remove overlay",
        ],
    ),
    ("show_grid", &["grid", "mouse grid"]),
    (
        "grid_select",
        &["cell", "grid cell", "select cell", "click cell"],
    ),
];

/// Reports whether `utterance` contains a trigger for `tool`.
///
/// # Arguments
/// * `utterance` - The recognised text, with its original casing and punctuation.
/// * `tool` - The tool name exactly as it appears in [`crate::tools::tool_specs`].
///
/// # Returns
/// `true` when at least one trigger phrase appears as whole words. An unknown
/// tool, or a tool with no table entry, returns `false`.
pub fn is_grounded(utterance: &str, tool: &str) -> bool {
    let Some((_, triggers)) = TOOL_TRIGGERS.iter().find(|(name, _)| *name == tool) else {
        return false;
    };
    let spoken = words(utterance);
    triggers
        .iter()
        .any(|trigger| contains_phrase(&spoken, trigger))
}

/// Reports whether every argument of `intent` is supported by `utterance`.
///
/// Numeric arguments must appear as a spoken digit, number word or ordinal.
/// String arguments must appear as a normalised contiguous phrase.
///
/// # Arguments
/// * `utterance` - The recognised text, with its original casing and punctuation.
/// * `intent` - The typed intent produced from the model call.
///
/// # Returns
/// `true` when every argument that names something is present in the
/// utterance. Arguments that are absent, such as `Scroll { amount: None }`,
/// impose no extra requirement.
pub fn arguments_are_grounded(utterance: &str, intent: &Intent) -> bool {
    let spoken = words(utterance);

    // The match is exhaustive on purpose: a new `Intent` variant with a
    // string or numeric argument must make a grounding decision here instead
    // of silently inheriting "true".
    match intent {
        Intent::ClickElement { target } | Intent::FocusField { target } => {
            contains_phrase(&spoken, target)
        }
        Intent::ClickNumber { n } => number_is_spoken(&spoken, *n),
        Intent::TypeText { text, field } => {
            contains_phrase(&spoken, text)
                && match field {
                    Some(field) => contains_phrase(&spoken, field),
                    None => true,
                }
        }
        Intent::PressKey { .. } => true,
        Intent::Scroll { amount, .. } => match amount {
            Some(amount) => number_is_spoken(&spoken, *amount),
            None => true,
        },
        Intent::Zoom { .. } => true,
        Intent::OpenApp { name } => contains_phrase(&spoken, name),
        Intent::OpenUrl { url } => contains_phrase(&spoken, url),
        Intent::SwitchWindow { target } => match target {
            Some(target) => contains_phrase(&spoken, target),
            None => true,
        },
        Intent::ShowNumbers | Intent::HideNumbers | Intent::ShowGrid => true,
        Intent::GridSelect { cell } => number_is_spoken(&spoken, *cell),
        Intent::SetMode { .. } => true,
        Intent::Confirm | Intent::Deny | Intent::Undo | Intent::Cancel => true,
        Intent::Clarify { .. } => true,
    }
}

/// Reports whether `tool` and every argument of `intent` are grounded.
///
/// This is the check the router uses: [`is_grounded`] proves the tool trigger
/// is present, and [`arguments_are_grounded`] proves the model did not invent
/// a target, text, field, name or number.
///
/// # Arguments
/// * `utterance` - The recognised text that produced the model call.
/// * `tool` - The call name exactly as the model emitted it.
/// * `intent` - The typed intent converted from that call.
///
/// # Returns
/// `true` only when both the tool trigger and every argument are grounded.
pub fn is_intent_grounded(utterance: &str, tool: &str, intent: &Intent) -> bool {
    is_grounded(utterance, tool) && arguments_are_grounded(utterance, intent)
}

/// Reports whether `value` is spoken anywhere in `spoken`.
///
/// Cardinal numbers, ordinals and digits are all accepted, so "fourteen",
/// "14" and "fourteenth" all ground `ClickNumber { n: 14 }`. Homophones such as
/// "to" and "for" are accepted because [`parse_number`] accepts them for the
/// rule grammar, and speech recognition makes the same substitution.
///
/// # Arguments
/// * `spoken` - The lowercased words of the utterance.
/// * `value` - The number taken from the model call.
///
/// # Returns
/// `true` when a cardinal or ordinal parse starting at any word equals
/// `value`.
fn number_is_spoken(spoken: &[String], value: u32) -> bool {
    (0..spoken.len()).any(|start| {
        let rest = &spoken[start..];
        parse_number(rest).is_some_and(|(parsed, _)| parsed == value)
            || parse_ordinal(rest).is_some_and(|(parsed, _)| parsed == value)
    })
}

/// Splits `text` into lowercase words, dropping punctuation.
///
/// # Arguments
/// * `text` - The text to split.
///
/// # Returns
/// The words in order. Apostrophes, hyphens and other punctuation are treated
/// as separators, so "I'll" becomes "i" and "ll".
fn words(text: &str) -> Vec<String> {
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Reports whether `needle` appears as a contiguous phrase in `haystack`.
///
/// # Arguments
/// * `haystack` - The lowercased words of the utterance.
/// * `needle` - The trigger phrase, which may contain spaces.
///
/// # Returns
/// `true` when every word of `needle` appears in order and adjacent.
fn contains_phrase(haystack: &[String], needle: &str) -> bool {
    let needle_words = words(needle);
    if needle_words.is_empty() || needle_words.len() > haystack.len() {
        return false;
    }
    haystack
        .windows(needle_words.len())
        .any(|window| window.iter().eq(needle_words.iter()))
}

/// Unit tests for grounding and the trigger table.
#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use reach_core::Direction;

    use super::*;
    use crate::tools::tool_specs;

    /// Every tool the model can call has a non-empty trigger list.
    #[test]
    fn every_exposed_tool_has_at_least_one_trigger() {
        for spec in tool_specs() {
            let triggers = TOOL_TRIGGERS
                .iter()
                .find(|(name, _)| *name == spec.name)
                .map(|(_, triggers)| *triggers);
            assert!(
                triggers.is_some_and(|triggers| !triggers.is_empty()),
                "{} has no trigger words",
                spec.name
            );
        }
    }

    /// The table contains no duplicates and no tool outside `tool_specs()`.
    #[test]
    fn trigger_table_matches_the_exposed_tool_set() {
        let specs = tool_specs();
        let mut seen = HashSet::new();
        for (name, triggers) in TOOL_TRIGGERS {
            assert!(!triggers.is_empty(), "{name} has an empty trigger list");
            assert!(seen.insert(*name), "duplicate trigger entry for {name}");
            assert!(
                specs.iter().any(|spec| spec.name == *name),
                "trigger table names unknown tool {name}"
            );
        }
    }

    /// Common spoken commands ground the tool the model should choose.
    #[test]
    fn real_commands_are_grounded() {
        for (utterance, tool) in [
            ("scroll down", "scroll"),
            ("click submit", "click_element"),
            ("click number four", "click_number"),
            ("type hello", "type_text"),
            ("focus the search field", "focus_field"),
            ("press enter", "press_key"),
            ("zoom in", "zoom"),
            ("open chrome", "open_app"),
            ("switch to notepad", "switch_window"),
            ("show numbers", "show_numbers"),
            ("hide numbers", "hide_numbers"),
            ("show the grid", "show_grid"),
            ("select cell five", "grid_select"),
        ] {
            assert!(
                is_grounded(utterance, tool),
                "{utterance:?} should ground {tool}"
            );
        }
    }

    /// The captured ambient hallucinations do not ground their tools.
    #[test]
    fn ambient_speech_does_not_ground_the_captured_hallucinations() {
        for (utterance, tool) in [
            ("the volume of sales went up", "hide_numbers"),
            ("hello how are you today", "hide_numbers"),
            ("I'll call you in five minutes", "hide_numbers"),
            ("tell me a joke", "open_app"),
        ] {
            assert!(
                !is_grounded(utterance, tool),
                "{utterance:?} must not ground {tool}"
            );
        }
    }

    /// Triggers match whole words, not fragments inside longer words.
    #[test]
    fn whole_words_do_not_match_inside_longer_words() {
        assert!(!is_grounded("he pressed the button yesterday", "press_key"));
        assert!(!is_grounded("the volume of sales went up", "scroll"));
        assert!(!is_grounded("she opened the door", "open_app"));
    }

    /// Multi-word triggers only match when the words are adjacent and in order.
    #[test]
    fn phrases_match_only_as_contiguous_words() {
        assert!(is_grounded("never mind", NO_ACTION));
        assert!(!is_grounded("never really mind", NO_ACTION));
        assert!(is_grounded("page down", "scroll"));
        assert!(!is_grounded("page after down", "scroll"));
    }

    /// Case and punctuation do not affect grounding.
    #[test]
    fn punctuation_and_case_are_ignored() {
        assert!(is_grounded("Please, SCROLL down!", "scroll"));
        assert!(is_grounded("Hide the numbers.", "hide_numbers"));
    }

    /// An unknown tool is never grounded.
    #[test]
    fn unknown_tool_is_not_grounded() {
        assert!(!is_grounded("open example.com", "open_url"));
        assert!(!is_grounded("confirm", "confirm"));
        assert!(!is_grounded("scroll down", "not_a_tool"));
    }

    /// Cardinal numbers alone do not ground a numbered click.
    #[test]
    fn cardinal_numbers_do_not_ground_click_number() {
        assert!(!is_grounded(
            "I'll call you in five minutes",
            "click_number"
        ));
        assert!(is_grounded("click number five", "click_number"));
    }

    /// Numeric arguments must be spoken, not inferred from context.
    #[test]
    fn numeric_arguments_must_be_spoken() {
        let click = Intent::ClickNumber { n: 14 };
        assert!(!arguments_are_grounded("click the big one", &click));
        assert!(arguments_are_grounded("click fourteen", &click));
        assert!(arguments_are_grounded("click 14", &click));
        assert!(arguments_are_grounded("click the fourteenth item", &click));

        let scroll = Intent::Scroll {
            direction: Direction::Down,
            amount: Some(40),
        };
        assert!(!arguments_are_grounded("scroll down", &scroll));
        assert!(arguments_are_grounded("scroll down forty", &scroll));
        assert!(arguments_are_grounded("scroll forty down", &scroll));

        let no_amount = Intent::Scroll {
            direction: Direction::Down,
            amount: None,
        };
        assert!(arguments_are_grounded("scroll down", &no_amount));
    }

    /// String arguments must appear as normalised contiguous phrases.
    #[test]
    fn string_arguments_must_be_spoken() {
        let click = Intent::ClickElement {
            target: "submit".to_owned(),
        };
        assert!(!arguments_are_grounded("click the thing", &click));
        assert!(arguments_are_grounded("Please, click SUBMIT!", &click));

        let typed = Intent::TypeText {
            text: "hello world".to_owned(),
            field: Some("search".to_owned()),
        };
        assert!(!arguments_are_grounded("type hello world", &typed));
        assert!(arguments_are_grounded(
            "type hello world in the search field",
            &typed
        ));

        let focus = Intent::FocusField {
            target: "search".to_owned(),
        };
        assert!(!arguments_are_grounded("focus the box", &focus));
        assert!(arguments_are_grounded("focus the search box", &focus));

        let app = Intent::OpenApp {
            name: "chrome".to_owned(),
        };
        assert!(!arguments_are_grounded("open notepad", &app));
        assert!(arguments_are_grounded("open chrome", &app));

        let window = Intent::SwitchWindow {
            target: Some("notepad".to_owned()),
        };
        assert!(!arguments_are_grounded("switch window", &window));
        assert!(arguments_are_grounded("switch to notepad", &window));
    }

    /// The full check combines the tool trigger and the argument checks.
    #[test]
    fn full_grounding_checks_triggers_and_arguments() {
        let click_number = Intent::ClickNumber { n: 14 };
        assert!(!is_intent_grounded(
            "click the big one",
            "click_number",
            &click_number
        ));
        assert!(is_intent_grounded(
            "click fourteen",
            "click_number",
            &click_number
        ));

        let click_element = Intent::ClickElement {
            target: "submit".to_owned(),
        };
        assert!(!is_intent_grounded(
            "click the thing",
            "click_element",
            &click_element
        ));
        assert!(is_intent_grounded(
            "click submit",
            "click_element",
            &click_element
        ));

        let scroll = Intent::Scroll {
            direction: Direction::Down,
            amount: Some(40),
        };
        assert!(!is_intent_grounded("scroll down", "scroll", &scroll));
        assert!(is_intent_grounded("scroll down forty", "scroll", &scroll));
    }
}
