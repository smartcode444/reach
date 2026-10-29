//! On-screen element representation, collection limits, and high-level element actions.
//!
//! Provides the [`LiveElement`] type which pairs an [`ElementSummary`] with
//! its platform automation handle, limits for tree traversal, and high-level
//! actions for clicking, focusing, and typing.

use crate::error::ScreenError;
use crate::label::{ControlKind, ElementSummary, MatchConfig, Resolution, rank_matches, resolve};
use std::time::Duration;

#[cfg(windows)]
use uiautomation::core::{UIAutomation, UIElement, UITreeWalker};
#[cfg(windows)]
use uiautomation::patterns::{UIInvokePattern, UIValuePattern};
#[cfg(windows)]
use uiautomation::types::{ControlType, Handle};

/// Control kinds that can accept typed text when a field name is supplied.
const FIELD_CONTROL_KINDS: [ControlKind; 2] = [ControlKind::Edit, ControlKind::ComboBox];

/// Limits for walking the on-screen UI tree to avoid excessive latency or memory use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CollectLimits {
    /// Maximum depth in the accessibility hierarchy to traverse.
    pub max_depth: usize,
    /// Maximum total number of elements to collect before stopping.
    pub max_elements: usize,
    /// Maximum duration allowed for tree collection.
    pub timeout: Duration,
}

impl Default for CollectLimits {
    /// Documented defaults: max depth 12, max 150 elements, 1500 ms timeout.
    fn default() -> Self {
        Self {
            max_depth: 12,
            max_elements: 150,
            timeout: Duration::from_millis(1500),
        }
    }
}

impl CollectLimits {
    /// Limits for web page document and fallback window collection:
    /// max depth 30, max 400 elements, 3000 ms timeout.
    pub const DOCUMENT: Self = Self {
        max_depth: 30,
        max_elements: 400,
        timeout: Duration::from_millis(3000),
    };

    /// Returns the document-specific collection limits (depth 30, 400 elements, 3000 ms).
    pub fn document() -> Self {
        Self::DOCUMENT
    }
}

/// The chosen collection root, indicating whether a Document was found or fallback occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectionRoot<T> {
    /// A Document control was found; collection starts here.
    Document(T),
    /// No Document control was found; collection falls back to the whole window.
    FallbackWindow(T),
}

impl<T> CollectionRoot<T> {
    /// Returns a reference to the underlying element or handle.
    ///
    /// # Returns
    /// A reference to the contained element.
    pub fn element(&self) -> &T {
        match self {
            Self::Document(elem) | Self::FallbackWindow(elem) => elem,
        }
    }

    /// Consumes the wrapper, returning the underlying element or handle.
    ///
    /// # Returns
    /// The contained element.
    pub fn into_element(self) -> T {
        match self {
            Self::Document(elem) | Self::FallbackWindow(elem) => elem,
        }
    }

    /// Whether a Document control was selected as the root.
    ///
    /// # Returns
    /// `true` if this root is a Document, `false` if it is a fallback window.
    pub fn is_document(&self) -> bool {
        matches!(self, Self::Document(_))
    }
}

/// Decides whether to start element collection at a Document control or fall back to the window.
///
/// If `document` is `Some`, collection starts at the Document control.
/// If `document` is `None`, collection falls back to `window`.
/// In both cases, the collection uses document limits (depth 30, 400 elements, 3000 ms).
///
/// # Arguments
/// * `window` - The handle or element representing the foreground window.
/// * `document` - The optional Document control found inside the window.
///
/// # Returns
/// A [`CollectionRoot`] wrapping either `document` if present, or `window` as fallback,
/// along with the [`CollectLimits`] to apply.
pub fn decide_collection_root<T>(
    window: T,
    document: Option<T>,
) -> (CollectionRoot<T>, CollectLimits) {
    let limits = CollectLimits::DOCUMENT;
    match document {
        Some(doc) => (CollectionRoot::Document(doc), limits),
        None => (CollectionRoot::FallbackWindow(window), limits),
    }
}

/// The successful method used to act on an on-screen element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionMethod {
    /// Triggered via the UI Automation Invoke pattern.
    InvokePattern,
    /// Updated via the UI Automation Value pattern.
    ValuePattern,
    /// Focused via UI Automation SetFocus.
    SetFocus,
}

/// The outcome of an element-targeted action (click, focus, type).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionOutcome {
    /// Action completed successfully via an in-process accessibility pattern.
    Done(ActionMethod),
    /// The target name was ambiguous among multiple on-screen elements.
    ///
    /// Contains candidate element names (up to 3) for user disambiguation.
    NeedsChoice(Vec<String>),
    /// No matching element was found on screen.
    NotFound,
    /// UI Automation patterns were unsupported; physical mouse click is required at `(x, y)`.
    NeedsClickAt {
        /// X coordinate on screen.
        x: i32,
        /// Y coordinate on screen.
        y: i32,
    },
    /// Direct text setting was unsupported; physical keyboard input is required.
    NeedsTypeText {
        /// Text string to be injected via keyboard.
        text: String,
    },
}

/// A live on-screen element with its metadata summary and backend action handle.
#[derive(Debug)]
pub struct LiveElement {
    /// Metadata summary (name, kind, bounding rect, state flags).
    pub summary: ElementSummary,
    /// Depth of the element relative to collection root.
    pub depth: usize,
    /// Platform UI Automation element handle on Windows.
    #[cfg(windows)]
    pub(crate) uia_element: Option<UIElement>,
}

impl LiveElement {
    /// Creates a new `LiveElement` from an [`ElementSummary`] without a platform handle.
    ///
    /// Sets `depth` to 0. Useful for testing pure logic without Windows accessibility dependencies.
    ///
    /// # Arguments
    /// * `summary` - The element summary metadata.
    ///
    /// # Returns
    /// A new `LiveElement` wrapping `summary`.
    pub fn from_summary(summary: ElementSummary) -> Self {
        Self {
            summary,
            depth: 0,
            #[cfg(windows)]
            uia_element: None,
        }
    }

    /// Creates a new `LiveElement` from an [`ElementSummary`] and traversal depth.
    ///
    /// # Arguments
    /// * `summary` - The element summary metadata.
    /// * `depth` - Traversal depth relative to collection root.
    ///
    /// # Returns
    /// A new `LiveElement` wrapping `summary` with the specified `depth`.
    pub fn from_summary_with_depth(summary: ElementSummary, depth: usize) -> Self {
        Self {
            summary,
            depth,
            #[cfg(windows)]
            uia_element: None,
        }
    }

    /// Attempts to click this element using the Invoke pattern, falling back to mouse coordinates.
    ///
    /// # Returns
    /// An [`ActionOutcome`] indicating whether Invoke succeeded or physical click is needed.
    pub fn click(&self) -> ActionOutcome {
        #[cfg(windows)]
        if let Some(elem) = &self.uia_element
            && let Ok(invoke) = elem.get_pattern::<UIInvokePattern>()
            && invoke.invoke().is_ok()
        {
            return ActionOutcome::Done(ActionMethod::InvokePattern);
        }

        let (x, y) = self.summary.rect.centre();
        ActionOutcome::NeedsClickAt { x, y }
    }

    /// Attempts to focus this element via SetFocus, falling back to a click at its centre.
    ///
    /// # Returns
    /// An [`ActionOutcome`] indicating whether SetFocus succeeded or physical click is needed.
    pub fn focus(&self) -> ActionOutcome {
        #[cfg(windows)]
        if let Some(elem) = &self.uia_element
            && elem.set_focus().is_ok()
        {
            return ActionOutcome::Done(ActionMethod::SetFocus);
        }

        let (x, y) = self.summary.rect.centre();
        ActionOutcome::NeedsClickAt { x, y }
    }

    /// Attempts to set text directly via the Value pattern, falling back to typing.
    ///
    /// Never stores or logs values of password fields.
    ///
    /// # Arguments
    /// * `text` - The text string to set or type.
    ///
    /// # Returns
    /// An [`ActionOutcome`] indicating whether Value pattern succeeded or typing is needed.
    pub fn type_text(&self, text: &str) -> ActionOutcome {
        #[cfg(windows)]
        if let Some(elem) = &self.uia_element
            && let Ok(val) = elem.get_pattern::<UIValuePattern>()
            && val.set_value(text).is_ok()
        {
            return ActionOutcome::Done(ActionMethod::ValuePattern);
        }

        // Focus first, then ask for physical keyboard typing.
        let _ = self.focus();
        ActionOutcome::NeedsTypeText {
            text: text.to_owned(),
        }
    }
}

/// The result of collecting on-screen elements from a window hierarchy.
#[derive(Debug)]
pub struct CollectResult {
    /// Collected elements with summaries and action handles.
    pub elements: Vec<LiveElement>,
    /// Whether collection stopped early due to depth, element count, or timeout limits.
    pub truncated: bool,
}

/// Strategy for locating the active foreground window handle.
pub trait ForegroundFinder {
    /// Obtains the handle of the foreground window as a numeric address.
    ///
    /// # Errors
    /// Returns [`ScreenError::NoForegroundWindow`] if no active window is available.
    fn get_foreground_hwnd(&self) -> Result<isize, ScreenError>;
}

/// Default Windows foreground window locator using Win32 `GetForegroundWindow`.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemForegroundFinder;

impl ForegroundFinder for SystemForegroundFinder {
    fn get_foreground_hwnd(&self) -> Result<isize, ScreenError> {
        #[cfg(windows)]
        {
            let hwnd = unsafe { windows::Win32::UI::WindowsAndMessaging::GetForegroundWindow() };
            if hwnd.0.is_null() {
                Err(ScreenError::NoForegroundWindow)
            } else {
                Ok(hwnd.0 as isize)
            }
        }
        #[cfg(not(windows))]
        {
            Err(ScreenError::NoForegroundWindow)
        }
    }
}

/// Maps a Windows UI Automation [`ControlType`] to Reach's [`ControlKind`].
#[cfg(windows)]
fn map_control_type(ct: ControlType) -> ControlKind {
    match ct {
        ControlType::Button | ControlType::SplitButton => ControlKind::Button,
        ControlType::Hyperlink => ControlKind::Link,
        ControlType::Edit => ControlKind::Edit,
        ControlType::CheckBox => ControlKind::CheckBox,
        ControlType::ComboBox => ControlKind::ComboBox,
        ControlType::MenuItem => ControlKind::MenuItem,
        ControlType::TabItem => ControlKind::TabItem,
        ControlType::ListItem => ControlKind::ListItem,
        ControlType::RadioButton => ControlKind::RadioButton,
        ControlType::Document => ControlKind::Document,
        _ => ControlKind::Other,
    }
}

/// Searches descendant elements of `root` for a Document control (the web page).
///
/// Uses a depth-bounded search with zero timeout to check if the window contains a web page
/// without blocking if none is present.
///
/// # Arguments
/// * `auto` - UI Automation instance.
/// * `root` - Root UI element of the foreground window.
///
/// # Returns
/// The Document [`UIElement`] if found, or `None` if the window contains no Document control.
#[cfg(windows)]
fn find_document_control(auto: &UIAutomation, root: &UIElement) -> Option<UIElement> {
    if let Ok(ct) = root.get_control_type()
        && ct == ControlType::Document
    {
        return Some(root.clone());
    }

    auto.create_matcher()
        .from_ref(root)
        .control_type(ControlType::Document)
        .timeout(0)
        .depth(15)
        .find_first()
        .ok()
}

/// Collects live elements from the active foreground window using a custom foreground finder.
///
/// If the foreground window contains a Document control (the web page), collection starts at
/// that Document with its own limits (depth 30, 400 elements, 3000 ms). If no Document exists,
/// it falls back to the whole window with the same limits.
///
/// # Arguments
/// * `finder` - Strategy for obtaining the foreground window handle.
/// * `limits` - Traversal budget constraints.
///
/// # Returns
/// A [`CollectResult`] containing live elements and whether collection was truncated.
///
/// # Errors
/// Returns [`ScreenError::NoForegroundWindow`] if no active window is found,
/// or [`ScreenError::Automation`] if UI Automation initialization fails.
pub fn collect_foreground_with<F: ForegroundFinder>(
    finder: &F,
    limits: CollectLimits,
) -> Result<CollectResult, ScreenError> {
    #[cfg(windows)]
    {
        let hwnd_val = finder.get_foreground_hwnd()?;
        let auto = UIAutomation::new().map_err(|e| ScreenError::Automation(e.to_string()))?;
        let root = auto
            .element_from_handle(Handle::from(hwnd_val))
            .map_err(|e| ScreenError::Automation(e.to_string()))?;
        let walker = auto
            .create_tree_walker()
            .map_err(|e| ScreenError::Automation(e.to_string()))?;

        let doc = find_document_control(&auto, &root);
        let (collection_root, chosen_limits) = decide_collection_root(root, doc);
        let effective_limits = if limits == CollectLimits::default() {
            chosen_limits
        } else {
            limits
        };

        collect_tree(collection_root.element(), &walker, effective_limits)
    }
    #[cfg(not(windows))]
    {
        let _ = (finder, limits);
        Err(ScreenError::Automation(
            "UI Automation is only available on Windows".to_owned(),
        ))
    }
}

/// Collects live elements from the active foreground window using the system default finder.
///
/// If the foreground window contains a Document control (the web page), collection starts at
/// that Document with its own limits (depth 30, 400 elements, 3000 ms). If no Document exists,
/// it falls back to the whole window with the same limits.
///
/// # Arguments
/// * `limits` - Traversal budget constraints. Defaults to [`CollectLimits::DOCUMENT`].
///
/// # Returns
/// A [`CollectResult`] containing live elements and truncation status.
///
/// # Errors
/// Returns [`ScreenError`] if window retrieval or UI tree traversal fails.
pub fn collect_foreground(limits: CollectLimits) -> Result<CollectResult, ScreenError> {
    collect_foreground_with(&SystemForegroundFinder, limits)
}

/// Recursively traverses a UI Automation subtree within depth, count, and time bounds.
#[cfg(windows)]
fn collect_tree(
    root: &UIElement,
    walker: &UITreeWalker,
    limits: CollectLimits,
) -> Result<CollectResult, ScreenError> {
    let start_time = std::time::Instant::now();
    let mut elements = Vec::new();
    let mut truncated = false;

    // Stack holds (element, current_depth)
    let mut stack = vec![(root.clone(), 0usize)];

    while let Some((current, depth)) = stack.pop() {
        if start_time.elapsed() >= limits.timeout {
            truncated = true;
            break;
        }

        if elements.len() >= limits.max_elements {
            truncated = true;
            break;
        }

        // Read element properties (safely handling errors with defaults).
        let name = current.get_name().unwrap_or_default();
        let ct = current.get_control_type().unwrap_or(ControlType::Custom);
        let kind = map_control_type(ct);

        let uia_rect = current.get_bounding_rectangle().unwrap_or_default();
        let rect = crate::label::Rect {
            left: uia_rect.get_left(),
            top: uia_rect.get_top(),
            right: uia_rect.get_right(),
            bottom: uia_rect.get_bottom(),
        };

        let enabled = current.is_enabled().unwrap_or(true);
        let offscreen = current.is_offscreen().unwrap_or(false);
        let is_password = current.is_password().unwrap_or(false);

        let summary = ElementSummary {
            name,
            kind,
            rect,
            enabled,
            offscreen,
            is_password,
        };

        elements.push(LiveElement {
            summary,
            depth,
            uia_element: Some(current.clone()),
        });

        // Collect children if within depth limits.
        if depth < limits.max_depth {
            if let Some(children) = walker.get_children(&current) {
                // Push children in reverse order so first child is popped first.
                for child in children.into_iter().rev() {
                    stack.push((child, depth + 1));
                }
            }
        } else {
            // Reached depth limit; if children existed, mark truncated.
            if walker.get_first_child(&current).is_ok() {
                truncated = true;
            }
        }
    }

    Ok(CollectResult {
        elements,
        truncated,
    })
}

/// Resolves a click target against candidate elements and executes the click action.
///
/// # Arguments
/// * `label` - Spoken or typed target element label.
/// * `elements` - Candidate live elements.
/// * `cfg` - Match configuration thresholds.
///
/// # Returns
/// An [`ActionOutcome`] indicating result of the resolution and click attempt.
pub fn click_element_in(label: &str, elements: &[LiveElement], cfg: &MatchConfig) -> ActionOutcome {
    let summaries: Vec<ElementSummary> = elements.iter().map(|e| e.summary.clone()).collect();
    let matches = rank_matches(label, &summaries, None);
    match resolve(&matches, cfg) {
        Resolution::NotFound => ActionOutcome::NotFound,
        Resolution::Ambiguous(indices) => {
            let names = indices
                .into_iter()
                .filter_map(|i| elements.get(i).map(|e| e.summary.name.clone()))
                .collect();
            ActionOutcome::NeedsChoice(names)
        }
        Resolution::Unique(index) => match elements.get(index) {
            Some(elem) => elem.click(),
            None => ActionOutcome::NotFound,
        },
    }
}

/// Resolves a focus target against candidate elements and executes the focus action.
///
/// Only [`ControlKind::Edit`] and [`ControlKind::ComboBox`] are considered, so
/// a button, link or menu item with the same name can never receive focus.
///
/// # Arguments
/// * `label` - Spoken or typed target element label.
/// * `elements` - Candidate live elements.
/// * `cfg` - Match configuration thresholds.
///
/// # Returns
/// An [`ActionOutcome`] indicating result of the resolution and focus attempt.
pub fn focus_field_in(label: &str, elements: &[LiveElement], cfg: &MatchConfig) -> ActionOutcome {
    let summaries: Vec<ElementSummary> = elements.iter().map(|e| e.summary.clone()).collect();
    // Field targets must be editable; a same-named button or link must not win.
    let matches = rank_matches(label, &summaries, Some(&FIELD_CONTROL_KINDS));
    match resolve(&matches, cfg) {
        Resolution::NotFound => ActionOutcome::NotFound,
        Resolution::Ambiguous(indices) => {
            let names = indices
                .into_iter()
                .filter_map(|i| elements.get(i).map(|e| e.summary.name.clone()))
                .collect();
            ActionOutcome::NeedsChoice(names)
        }
        Resolution::Unique(index) => match elements.get(index) {
            Some(elem) => elem.focus(),
            None => ActionOutcome::NotFound,
        },
    }
}

/// Types text, optionally resolving and focusing a named field first.
///
/// If `field` is provided, filters eligible controls to [`ControlKind::Edit`] and
/// [`ControlKind::ComboBox`].
///
/// # Arguments
/// * `text` - Text to type into the field.
/// * `field` - Optional spoken name of the target field to focus first.
/// * `elements` - Candidate live elements.
/// * `cfg` - Match configuration thresholds.
///
/// # Returns
/// An [`ActionOutcome`] indicating result of typing or field resolution.
pub fn type_text_in(
    text: &str,
    field: Option<&str>,
    elements: &[LiveElement],
    cfg: &MatchConfig,
) -> ActionOutcome {
    if let Some(field_name) = field {
        let summaries: Vec<ElementSummary> = elements.iter().map(|e| e.summary.clone()).collect();
        // Filtering strictly to editable controls.
        let matches = rank_matches(field_name, &summaries, Some(&FIELD_CONTROL_KINDS));
        match resolve(&matches, cfg) {
            Resolution::NotFound => ActionOutcome::NotFound,
            Resolution::Ambiguous(indices) => {
                let names = indices
                    .into_iter()
                    .filter_map(|i| elements.get(i).map(|e| e.summary.name.clone()))
                    .collect();
                ActionOutcome::NeedsChoice(names)
            }
            Resolution::Unique(index) => match elements.get(index) {
                Some(elem) => elem.type_text(text),
                None => ActionOutcome::NotFound,
            },
        }
    } else {
        // No specific field targeted; request direct text typing into whatever is active.
        ActionOutcome::NeedsTypeText {
            text: text.to_owned(),
        }
    }
}

/// High-level function: clicks the element whose name best matches `label`.
///
/// Collects elements from the active foreground window, resolves `label`, and executes.
///
/// # Arguments
/// * `label` - Spoken target label.
///
/// # Returns
/// The outcome of the action.
///
/// # Errors
/// Returns [`ScreenError`] if foreground window collection fails.
pub fn click_element(label: &str) -> Result<ActionOutcome, ScreenError> {
    let collected = collect_foreground(CollectLimits::DOCUMENT)?;
    Ok(click_element_in(
        label,
        &collected.elements,
        &MatchConfig::default(),
    ))
}

/// High-level function: focuses the field whose name best matches `label`.
///
/// # Arguments
/// * `label` - Spoken target label.
///
/// # Returns
/// The outcome of the action.
///
/// # Errors
/// Returns [`ScreenError`] if foreground window collection fails.
pub fn focus_field(label: &str) -> Result<ActionOutcome, ScreenError> {
    let collected = collect_foreground(CollectLimits::DOCUMENT)?;
    Ok(focus_field_in(
        label,
        &collected.elements,
        &MatchConfig::default(),
    ))
}

/// High-level function: types `text`, focusing the field named `field` first if provided.
///
/// # Arguments
/// * `text` - Text to type.
/// * `field` - Optional spoken name of the field.
///
/// # Returns
/// The outcome of the action.
///
/// # Errors
/// Returns [`ScreenError`] if foreground window collection fails.
pub fn type_text(text: &str, field: Option<&str>) -> Result<ActionOutcome, ScreenError> {
    let collected = collect_foreground(CollectLimits::DOCUMENT)?;
    Ok(type_text_in(
        text,
        field,
        &collected.elements,
        &MatchConfig::default(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::label::Rect;

    /// Helper to create a dummy live element for testing.
    fn make_live(name: &str, kind: ControlKind, x: i32, y: i32) -> LiveElement {
        LiveElement::from_summary(ElementSummary {
            name: name.to_owned(),
            kind,
            rect: Rect {
                left: x,
                top: y,
                right: x + 100,
                bottom: y + 40,
            },
            enabled: true,
            offscreen: false,
            is_password: false,
        })
    }

    /// Tests default collection limits values.
    #[test]
    fn collect_limits_defaults() {
        let limits = CollectLimits::default();
        assert_eq!(limits.max_depth, 12);
        assert_eq!(limits.max_elements, 150);
        assert_eq!(limits.timeout, Duration::from_millis(1500));
    }

    /// Tests click_element_in resolves unique match to fallback coordinates when unhandled.
    #[test]
    fn click_element_unique_resolves_to_centre_coordinates() {
        let elements = vec![
            make_live("Submit", ControlKind::Button, 100, 200),
            make_live("Cancel", ControlKind::Button, 300, 200),
        ];
        let outcome = click_element_in("submit", &elements, &MatchConfig::default());
        assert_eq!(outcome, ActionOutcome::NeedsClickAt { x: 150, y: 220 });
    }

    /// Tests click_element_in returns NeedsChoice for ambiguous names.
    #[test]
    fn click_element_ambiguous_returns_needs_choice() {
        let elements = vec![
            make_live("Option A", ControlKind::Button, 10, 10),
            make_live("Option B", ControlKind::Button, 10, 60),
        ];
        let outcome = click_element_in("Option", &elements, &MatchConfig::default());
        match outcome {
            ActionOutcome::NeedsChoice(names) => {
                assert_eq!(names.len(), 2);
                assert!(names.contains(&"Option A".to_string()));
                assert!(names.contains(&"Option B".to_string()));
            }
            other => panic!("expected NeedsChoice, got {:?}", other),
        }
    }

    /// Tests click_element_in returns NotFound when no match meets threshold.
    #[test]
    fn click_element_unmatched_returns_not_found() {
        let elements = vec![make_live("Submit", ControlKind::Button, 100, 200)];
        let outcome = click_element_in("xyz", &elements, &MatchConfig::default());
        assert_eq!(outcome, ActionOutcome::NotFound);
    }

    /// Tests focus_field_in resolves unique field to centre click coordinates when SetFocus is stubbed.
    #[test]
    fn focus_field_unique_resolves_to_click() {
        let elements = vec![make_live("Username", ControlKind::Edit, 50, 50)];
        let outcome = focus_field_in("username", &elements, &MatchConfig::default());
        assert_eq!(outcome, ActionOutcome::NeedsClickAt { x: 100, y: 70 });
    }

    /// Tests type_text_in without field returns direct NeedsTypeText.
    #[test]
    fn type_text_without_field_returns_needs_type_text() {
        let elements = vec![make_live("Username", ControlKind::Edit, 50, 50)];
        let outcome = type_text_in("hello world", None, &elements, &MatchConfig::default());
        assert_eq!(
            outcome,
            ActionOutcome::NeedsTypeText {
                text: "hello world".to_owned()
            }
        );
    }

    /// Tests type_text_in with named field focuses field and returns NeedsTypeText.
    #[test]
    fn type_text_with_field_resolves_field() {
        let elements = vec![
            make_live("Search", ControlKind::Edit, 200, 100),
            make_live("Search", ControlKind::Button, 400, 100),
        ];
        // Searching for "Search" with edit filter only picks the Edit control.
        let outcome = type_text_in(
            "query text",
            Some("Search"),
            &elements,
            &MatchConfig::default(),
        );
        assert_eq!(
            outcome,
            ActionOutcome::NeedsTypeText {
                text: "query text".to_owned()
            }
        );
    }

    /// focus_field_in must ignore buttons, links and menu items even when their
    /// names match the spoken field name.
    #[test]
    fn focus_field_ignores_non_editable_controls() {
        let elements = vec![
            make_live("Email", ControlKind::Button, 10, 10),
            make_live("Email", ControlKind::Link, 20, 20),
            make_live("Email", ControlKind::MenuItem, 30, 30),
            make_live("Email", ControlKind::Edit, 50, 50),
        ];
        let outcome = focus_field_in("email", &elements, &MatchConfig::default());
        assert_eq!(outcome, ActionOutcome::NeedsClickAt { x: 100, y: 70 });
    }

    /// focus_field_in must report NotFound when only non-editable controls match.
    #[test]
    fn focus_field_without_editable_control_returns_not_found() {
        let elements = vec![
            make_live("Email", ControlKind::Button, 10, 10),
            make_live("Email", ControlKind::Link, 20, 20),
            make_live("Email", ControlKind::MenuItem, 30, 30),
        ];
        let outcome = focus_field_in("email", &elements, &MatchConfig::default());
        assert_eq!(outcome, ActionOutcome::NotFound);
    }

    /// type_text_in with a field name must ignore buttons, links and menu items.
    #[test]
    fn type_text_with_field_ignores_non_editable_controls() {
        let elements = vec![
            make_live("Email", ControlKind::Button, 10, 10),
            make_live("Email", ControlKind::Link, 20, 20),
            make_live("Email", ControlKind::MenuItem, 30, 30),
            make_live("Email", ControlKind::Edit, 50, 50),
        ];
        let outcome = type_text_in(
            "student@example.com",
            Some("email"),
            &elements,
            &MatchConfig::default(),
        );
        assert_eq!(
            outcome,
            ActionOutcome::NeedsTypeText {
                text: "student@example.com".to_owned()
            }
        );
    }

    /// type_text_in with a field name must report NotFound when only
    /// non-editable controls match.
    #[test]
    fn type_text_with_field_without_editable_control_returns_not_found() {
        let elements = vec![
            make_live("Email", ControlKind::Button, 10, 10),
            make_live("Email", ControlKind::Link, 20, 20),
            make_live("Email", ControlKind::MenuItem, 30, 30),
        ];
        let outcome = type_text_in("text", Some("email"), &elements, &MatchConfig::default());
        assert_eq!(outcome, ActionOutcome::NotFound);
    }

    /// Combo boxes are valid field targets alongside Edit controls.
    #[test]
    fn combo_box_is_a_valid_field_target() {
        let elements = vec![
            make_live("Level", ControlKind::Button, 10, 10),
            make_live("Level", ControlKind::ComboBox, 50, 50),
        ];
        assert_eq!(
            focus_field_in("level", &elements, &MatchConfig::default()),
            ActionOutcome::NeedsClickAt { x: 100, y: 70 }
        );
        assert_eq!(
            type_text_in("200", Some("level"), &elements, &MatchConfig::default()),
            ActionOutcome::NeedsTypeText {
                text: "200".to_owned()
            }
        );
    }

    /// The demo form's field labels resolve to their Edit controls, and the
    /// Chrome Home button and Managed bookmarks folder are not candidates.
    #[test]
    fn focus_field_resolves_demo_form_fields() {
        let elements = vec![
            make_live("Home", ControlKind::Button, 0, 0),
            make_live("Managed bookmarks", ControlKind::MenuItem, 0, 40),
            make_live("Full Name", ControlKind::Edit, 100, 100),
            make_live("Email", ControlKind::Edit, 100, 200),
            make_live("Matric Number", ControlKind::Edit, 100, 300),
            make_live("Submit", ControlKind::Button, 100, 400),
        ];
        assert_eq!(
            focus_field_in("full name", &elements, &MatchConfig::default()),
            ActionOutcome::NeedsClickAt { x: 150, y: 120 }
        );
        assert_eq!(
            focus_field_in("email", &elements, &MatchConfig::default()),
            ActionOutcome::NeedsClickAt { x: 150, y: 220 }
        );
        assert_eq!(
            focus_field_in("matric number", &elements, &MatchConfig::default()),
            ActionOutcome::NeedsClickAt { x: 150, y: 320 }
        );
    }

    /// Tests mock foreground finder returning an error.
    #[derive(Debug)]
    struct FailingFinder;

    impl ForegroundFinder for FailingFinder {
        fn get_foreground_hwnd(&self) -> Result<isize, ScreenError> {
            Err(ScreenError::NoForegroundWindow)
        }
    }

    /// Tests collect_foreground_with propagates NoForegroundWindow.
    #[test]
    fn collect_foreground_handles_failing_finder() {
        let finder = FailingFinder;
        let res = collect_foreground_with(&finder, CollectLimits::default());
        assert!(matches!(res, Err(ScreenError::NoForegroundWindow)));
    }

    /// Tests document-specific collection limits match depth 30, 400 elements, 3000 ms timeout.
    #[test]
    fn collect_limits_document() {
        let limits = CollectLimits::DOCUMENT;
        assert_eq!(limits.max_depth, 30);
        assert_eq!(limits.max_elements, 400);
        assert_eq!(limits.timeout, Duration::from_millis(3000));
        assert_eq!(CollectLimits::document(), limits);
    }

    /// Tests that the fallback decision selects the Document root with document limits when a Document exists.
    #[test]
    fn fallback_decision_prefers_document_when_present() {
        let window = "window_handle";
        let document = Some("document_handle");
        let (root, limits) = decide_collection_root(window, document);
        assert_eq!(root, CollectionRoot::Document("document_handle"));
        assert!(root.is_document());
        assert_eq!(root.into_element(), "document_handle");
        assert_eq!(limits, CollectLimits::DOCUMENT);
    }

    /// Tests that the fallback decision falls back to the whole window with the same limits when no Document exists.
    #[test]
    fn fallback_decision_falls_back_to_window_when_document_missing() {
        let window = "window_handle";
        let document: Option<&str> = None;
        let (root, limits) = decide_collection_root(window, document);
        assert_eq!(root, CollectionRoot::FallbackWindow("window_handle"));
        assert!(!root.is_document());
        assert_eq!(root.into_element(), "window_handle");
        assert_eq!(limits, CollectLimits::DOCUMENT);
    }

    /// Tests that LiveElement correctly retains depth metadata.
    #[test]
    fn live_element_depth_tracking() {
        let summary = ElementSummary {
            name: "Test Button".to_owned(),
            kind: ControlKind::Button,
            rect: Rect {
                left: 0,
                top: 0,
                right: 50,
                bottom: 20,
            },
            enabled: true,
            offscreen: false,
            is_password: false,
        };
        let elem = LiveElement::from_summary_with_depth(summary, 7);
        assert_eq!(elem.depth, 7);
    }

    /// Manual integration test to run against a real Windows desktop or demo form.
    #[test]
    #[ignore = "run manually on the demo form with: cargo test -p screen -- --ignored --nocapture"]
    fn manual_windows_collect_foreground() {
        let limits = CollectLimits::default();
        let result = collect_foreground(limits);
        match result {
            Ok(collected) => {
                println!(
                    "Collected {} elements (truncated: {})",
                    collected.elements.len(),
                    collected.truncated
                );
                assert!(!collected.elements.is_empty());
            }
            Err(e) => {
                println!("Collection failed: {e}");
            }
        }
    }
}
