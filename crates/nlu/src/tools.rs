//! The tool list the Needle model is allowed to call.
//!
//! Only the intents in [`tool_specs`] are visible to the model. Confirmation,
//! cancellation, mode changes and URL opening stay in the rule grammar, so a
//! model mistake can never confirm a risky action on the user's behalf.

use serde_json::{Map, Value, json};

use reach_core::Intent;

/// The name of the abstention tool.
///
/// The model calls `no_action` when the speech is not a computer command. It is
/// deliberately not an [`Intent`] variant, so callers must check for it before
/// calling [`call_to_intent`].
pub const NO_ACTION: &str = "no_action";

/// A JSON Schema type used by one tool parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolParamKind {
    /// A JSON string.
    String,
    /// A JSON integer.
    Integer,
}

impl ToolParamKind {
    /// Returns the JSON Schema name for this type.
    fn as_str(self) -> &'static str {
        match self {
            ToolParamKind::String => "string",
            ToolParamKind::Integer => "integer",
        }
    }
}

/// One argument accepted by a tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolParam {
    /// The JSON key the model must use.
    pub name: &'static str,
    /// The JSON type of the value.
    pub kind: ToolParamKind,
    /// Whether the model must include the key.
    pub required: bool,
}

/// A tool the Needle model may call.
///
/// Tool names are the snake_case names of [`Intent`] variants, except for
/// [`NO_ACTION`]. Descriptions are one short sentence because Needle 3 receives
/// the full list on every request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolSpec {
    /// The tool name the model emits.
    pub name: &'static str,
    /// A one-sentence description of when to call the tool.
    pub description: &'static str,
    /// The arguments the tool accepts.
    pub params: &'static [ToolParam],
}

/// Returns the complete tool list sent to the model.
///
/// The list is the closed set of safe actions plus [`NO_ACTION`]. Confirmation,
/// denial, cancellation, undo, mode changes, clarification and URL opening are
/// absent on purpose: those remain rule-only.
///
/// [`NO_ACTION`] is deliberately first. Needle 3 shows a strong positional bias
/// with a long tool list; putting the abstention tool first was the only order
/// that produced correct calls for both "scroll down" and "click submit" in the
/// manual integration run.
pub fn tool_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: NO_ACTION,
            description: "Use when the speech is not a computer command.",
            params: &[],
        },
        ToolSpec {
            name: "click_element",
            description: "Click the on-screen element whose name best matches the target.",
            params: &[ToolParam {
                name: "target",
                kind: ToolParamKind::String,
                required: true,
            }],
        },
        ToolSpec {
            name: "click_number",
            description: "Click the element labelled with a number from the numbered overlay.",
            params: &[ToolParam {
                name: "n",
                kind: ToolParamKind::Integer,
                required: true,
            }],
        },
        ToolSpec {
            name: "type_text",
            description: "Type text, optionally into the field named by the user.",
            params: &[
                ToolParam {
                    name: "text",
                    kind: ToolParamKind::String,
                    required: true,
                },
                ToolParam {
                    name: "field",
                    kind: ToolParamKind::String,
                    required: false,
                },
            ],
        },
        ToolSpec {
            name: "focus_field",
            description: "Move keyboard focus to the named input field.",
            params: &[ToolParam {
                name: "target",
                kind: ToolParamKind::String,
                required: true,
            }],
        },
        ToolSpec {
            name: "press_key",
            description: "Press one keyboard key such as enter, tab or page down.",
            params: &[ToolParam {
                name: "key",
                kind: ToolParamKind::String,
                required: true,
            }],
        },
        ToolSpec {
            name: "scroll",
            description: "Scroll the active window up, down, left or right.",
            params: &[
                ToolParam {
                    name: "direction",
                    kind: ToolParamKind::String,
                    required: true,
                },
                ToolParam {
                    name: "amount",
                    kind: ToolParamKind::Integer,
                    required: false,
                },
            ],
        },
        ToolSpec {
            name: "zoom",
            description: "Zoom the active window in, out or back to the default.",
            params: &[ToolParam {
                name: "direction",
                kind: ToolParamKind::String,
                required: true,
            }],
        },
        ToolSpec {
            name: "open_app",
            description: "Open an allowed application such as chrome, notepad or explorer.",
            params: &[ToolParam {
                name: "name",
                kind: ToolParamKind::String,
                required: true,
            }],
        },
        ToolSpec {
            name: "switch_window",
            description: "Bring another window to the front, or switch to the next window.",
            params: &[ToolParam {
                name: "target",
                kind: ToolParamKind::String,
                required: false,
            }],
        },
        ToolSpec {
            name: "show_numbers",
            description: "Draw numbers over every clickable element on screen.",
            params: &[],
        },
        ToolSpec {
            name: "hide_numbers",
            description: "Remove the numbered overlay from the screen.",
            params: &[],
        },
        ToolSpec {
            name: "show_grid",
            description: "Show the 3x3 mouse grid.",
            params: &[],
        },
        ToolSpec {
            name: "grid_select",
            description: "Pick a cell number from the 3x3 mouse grid.",
            params: &[ToolParam {
                name: "cell",
                kind: ToolParamKind::Integer,
                required: true,
            }],
        },
    ]
}

/// Serialises `tools` as the compact JSON array Needle expects.
///
/// The output is a JSON Schema per tool with `name`, `description` and
/// `parameters`. Needle's constrained decoder reads the argument keys from
/// `parameters.properties`.
pub fn tools_json(tools: &[ToolSpec]) -> String {
    let entries: Vec<Value> = tools.iter().map(tool_json).collect();
    Value::Array(entries).to_string()
}

/// Builds the JSON Schema object for one tool.
///
/// # Arguments
/// * `tool` - The spec to describe.
///
/// # Returns
/// An object with `name`, `description` and a `parameters` JSON Schema whose
/// `properties` list every accepted argument.
fn tool_json(tool: &ToolSpec) -> Value {
    let mut properties = Map::new();
    let mut required = Vec::new();
    for param in tool.params {
        properties.insert(
            param.name.to_owned(),
            json!({ "type": param.kind.as_str() }),
        );
        if param.required {
            required.push(Value::String(param.name.to_owned()));
        }
    }

    let mut parameters = Map::new();
    parameters.insert("type".to_owned(), Value::String("object".to_owned()));
    parameters.insert("properties".to_owned(), Value::Object(properties));
    if !required.is_empty() {
        parameters.insert("required".to_owned(), Value::Array(required));
    }

    json!({
        "name": tool.name,
        "description": tool.description,
        "parameters": Value::Object(parameters),
    })
}

/// Converts one model call into a typed [`Intent`].
///
/// The tool name becomes the internal `action` tag, and `arguments` supplies the
/// remaining fields. A non-object `arguments` value is treated as an empty
/// object. An unknown tool name, or [`NO_ACTION`], fails deserialisation; the
/// caller must reject the reply rather than repair it.
///
/// # Arguments
/// * `name` - The tool name exactly as the model emitted it.
/// * `arguments` - The call's arguments, usually a JSON object.
///
/// # Returns
/// The typed intent, or a deserialisation error when the name or fields do not
/// match an [`Intent`] variant.
///
/// # Errors
/// Returns [`serde_json::Error`] if the combined object is not a valid intent.
pub fn call_to_intent(name: &str, arguments: Value) -> Result<Intent, serde_json::Error> {
    let mut object = match arguments {
        Value::Object(object) => object,
        _ => Map::new(),
    };
    object.insert("action".to_owned(), Value::String(name.to_owned()));
    serde_json::from_value(Value::Object(object))
}

/// Unit tests for the tool table and the call conversion.
#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use reach_core::{Direction, Key, ZoomDir};

    /// Returns the JSON action tag for `intent`.
    fn action_name(intent: &Intent) -> String {
        serde_json::to_value(intent).unwrap()["action"]
            .as_str()
            .unwrap()
            .to_owned()
    }

    /// Returns one sample call per exposed intent, in tool-list order.
    fn sample_calls() -> Vec<(&'static str, Value, Intent)> {
        vec![
            (
                "click_element",
                json!({ "target": "submit" }),
                Intent::ClickElement {
                    target: "submit".to_owned(),
                },
            ),
            (
                "click_number",
                json!({ "n": 4 }),
                Intent::ClickNumber { n: 4 },
            ),
            (
                "type_text",
                json!({ "text": "hello", "field": "search" }),
                Intent::TypeText {
                    text: "hello".to_owned(),
                    field: Some("search".to_owned()),
                },
            ),
            (
                "focus_field",
                json!({ "target": "email" }),
                Intent::FocusField {
                    target: "email".to_owned(),
                },
            ),
            (
                "press_key",
                json!({ "key": "page_down" }),
                Intent::PressKey { key: Key::PageDown },
            ),
            (
                "scroll",
                json!({ "direction": "down", "amount": 3 }),
                Intent::Scroll {
                    direction: Direction::Down,
                    amount: Some(3),
                },
            ),
            (
                "zoom",
                json!({ "direction": "in" }),
                Intent::Zoom {
                    direction: ZoomDir::In,
                },
            ),
            (
                "open_app",
                json!({ "name": "chrome" }),
                Intent::OpenApp {
                    name: "chrome".to_owned(),
                },
            ),
            (
                "switch_window",
                json!({ "target": "notepad" }),
                Intent::SwitchWindow {
                    target: Some("notepad".to_owned()),
                },
            ),
            ("show_numbers", json!({}), Intent::ShowNumbers),
            ("hide_numbers", json!({}), Intent::HideNumbers),
            ("show_grid", json!({}), Intent::ShowGrid),
            (
                "grid_select",
                json!({ "cell": 5 }),
                Intent::GridSelect { cell: 5 },
            ),
        ]
    }

    /// Every tool spec has a matching sample and round-trips through the
    /// conversion with its typed intent.
    #[test]
    fn every_intent_tool_round_trips() {
        let cases = sample_calls();
        let specs = tool_specs();

        for (name, arguments, expected) in &cases {
            assert!(
                specs.iter().any(|spec| spec.name == *name),
                "no tool spec for {name}"
            );
            let intent = call_to_intent(name, arguments.clone()).unwrap();
            assert_eq!(&intent, expected, "round trip failed for {name}");
        }

        for spec in specs.iter().filter(|spec| spec.name != NO_ACTION) {
            assert!(
                cases.iter().any(|(name, _, _)| *name == spec.name),
                "no round-trip sample for {}",
                spec.name
            );
        }
    }

    /// The tool table is derived from an exhaustive decision over `Intent`, so
    /// a new variant cannot be exposed or hidden by accident.
    #[test]
    fn tool_table_matches_an_exhaustive_intent_decision() {
        /// Decides whether the model may emit `intent`.
        fn is_exposed(intent: &Intent) -> bool {
            match intent {
                Intent::ClickElement { .. } => true,
                Intent::ClickNumber { .. } => true,
                Intent::TypeText { .. } => true,
                Intent::FocusField { .. } => true,
                Intent::PressKey { .. } => true,
                Intent::Scroll { .. } => true,
                Intent::Zoom { .. } => true,
                Intent::OpenApp { .. } => true,
                Intent::OpenUrl { .. } => false,
                Intent::SwitchWindow { .. } => true,
                Intent::ShowNumbers => true,
                Intent::HideNumbers => true,
                Intent::ShowGrid => true,
                Intent::GridSelect { .. } => true,
                Intent::SetMode { .. } => false,
                Intent::Confirm => false,
                Intent::Deny => false,
                Intent::Undo => false,
                Intent::Cancel => false,
                Intent::Clarify { .. } => false,
            }
        }

        let samples = vec![
            Intent::ClickElement {
                target: "submit".to_owned(),
            },
            Intent::ClickNumber { n: 1 },
            Intent::TypeText {
                text: "hello".to_owned(),
                field: None,
            },
            Intent::FocusField {
                target: "email".to_owned(),
            },
            Intent::PressKey { key: Key::Enter },
            Intent::Scroll {
                direction: Direction::Down,
                amount: None,
            },
            Intent::Zoom {
                direction: ZoomDir::Reset,
            },
            Intent::OpenApp {
                name: "notepad".to_owned(),
            },
            Intent::OpenUrl {
                url: "https://example.com".to_owned(),
            },
            Intent::SwitchWindow { target: None },
            Intent::ShowNumbers,
            Intent::HideNumbers,
            Intent::ShowGrid,
            Intent::GridSelect { cell: 9 },
            Intent::SetMode {
                mode: reach_core::Mode::Sleep,
            },
            Intent::Confirm,
            Intent::Deny,
            Intent::Undo,
            Intent::Cancel,
            Intent::Clarify {
                question: "Which one?".to_owned(),
            },
        ];

        let mut exposed: Vec<String> = samples
            .iter()
            .filter(|intent| is_exposed(intent))
            .map(action_name)
            .collect();
        let mut declared: Vec<String> = tool_specs()
            .iter()
            .filter(|spec| spec.name != NO_ACTION)
            .map(|spec| spec.name.to_owned())
            .collect();
        exposed.sort();
        declared.sort();
        assert_eq!(exposed, declared);

        for hidden in [
            Intent::OpenUrl {
                url: "https://example.com".to_owned(),
            },
            Intent::SetMode {
                mode: reach_core::Mode::Command,
            },
            Intent::Confirm,
            Intent::Deny,
            Intent::Undo,
            Intent::Cancel,
            Intent::Clarify {
                question: "Which one?".to_owned(),
            },
        ] {
            assert!(!is_exposed(&hidden), "{hidden:?} must stay rule-only");
        }

        assert!(
            tool_specs().iter().any(|spec| spec.name == NO_ACTION),
            "the abstention tool must always be declared"
        );
    }

    /// The JSON form describes every spec, including required arguments.
    #[test]
    fn tools_json_describes_every_spec() {
        let specs = tool_specs();
        let parsed: Value = serde_json::from_str(&tools_json(&specs)).unwrap();
        let entries = parsed.as_array().unwrap();
        assert_eq!(entries.len(), specs.len());

        let mut names = HashSet::new();
        for (entry, spec) in entries.iter().zip(&specs) {
            assert_eq!(entry["name"], spec.name);
            assert_eq!(entry["description"], spec.description);
            assert!(names.insert(spec.name), "duplicate tool {}", spec.name);

            let parameters = entry["parameters"].as_object().unwrap();
            assert_eq!(parameters["type"], "object");
            let properties = parameters["properties"].as_object().unwrap();
            assert_eq!(properties.len(), spec.params.len());
            for param in spec.params {
                assert_eq!(properties[param.name]["type"], param.kind.as_str());
            }

            let required: Vec<&str> = parameters
                .get("required")
                .and_then(Value::as_array)
                .map(|values| values.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            let expected: Vec<&str> = spec
                .params
                .iter()
                .filter(|param| param.required)
                .map(|param| param.name)
                .collect();
            assert_eq!(required, expected, "required mismatch for {}", spec.name);
        }

        assert!(names.contains(NO_ACTION));
    }

    #[test]
    /// Prints the tool count and tool list length in characters.
    fn reports_tool_count_and_characters() {
        let tools = tool_specs();
        let json = tools_json(&tools);
        println!("TOOL_COUNT={} TOOL_CHARS={}", tools.len(), json.len());
        assert!(!tools.is_empty());
    }

    /// Descriptions are short enough for the full tool list to fit the context.
    #[test]
    fn descriptions_are_one_short_sentence() {
        for spec in tool_specs() {
            let description = spec.description;
            assert!(!description.is_empty(), "{} has no description", spec.name);
            assert!(
                description.len() <= 100,
                "{} description is too long",
                spec.name
            );
            assert!(
                description.ends_with('.'),
                "{} must end with a period",
                spec.name
            );
            assert!(
                !description[..description.len() - 1].contains('.'),
                "{} must be one sentence",
                spec.name
            );
            assert!(
                !description.contains('\n'),
                "{} must be a single line",
                spec.name
            );
        }
    }

    /// The abstention tool has the required text and accepts no arguments.
    #[test]
    fn no_action_has_no_parameters() {
        let spec = tool_specs()
            .into_iter()
            .find(|spec| spec.name == NO_ACTION)
            .unwrap();
        assert!(spec.params.is_empty());
        assert_eq!(
            spec.description,
            "Use when the speech is not a computer command."
        );
    }

    /// Optional arguments may be omitted and become `None` in the intent.
    #[test]
    fn optional_arguments_may_be_omitted() {
        assert_eq!(
            call_to_intent("switch_window", json!({})).unwrap(),
            Intent::SwitchWindow { target: None }
        );
        assert_eq!(
            call_to_intent("scroll", json!({ "direction": "up" })).unwrap(),
            Intent::Scroll {
                direction: Direction::Up,
                amount: None
            }
        );
        assert_eq!(
            call_to_intent("type_text", json!({ "text": "hello" })).unwrap(),
            Intent::TypeText {
                text: "hello".to_owned(),
                field: None
            }
        );
    }

    /// Non-object arguments are ignored rather than rejected.
    #[test]
    fn non_object_arguments_become_an_empty_object() {
        assert_eq!(
            call_to_intent("show_numbers", Value::Null).unwrap(),
            Intent::ShowNumbers
        );
        assert_eq!(
            call_to_intent("show_numbers", json!("ignore me")).unwrap(),
            Intent::ShowNumbers
        );
    }

    /// The tool name wins over an `action` key smuggled into the arguments.
    #[test]
    fn tool_name_overrides_arguments_action() {
        let intent = call_to_intent(
            "show_numbers",
            json!({ "action": "open_url", "url": "https://example.com" }),
        )
        .unwrap();
        assert_eq!(intent, Intent::ShowNumbers);
    }

    /// Unknown names and the abstention tool never become an `Intent`.
    #[test]
    fn rejects_unknown_tools_and_no_action() {
        assert!(call_to_intent("format_disk", json!({})).is_err());
        assert!(call_to_intent(NO_ACTION, json!({})).is_err());
    }
}
