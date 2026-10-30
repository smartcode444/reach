//! Pure functions and types for resilient on-screen element label matching.
//!
//! Provides text normalisation, fuzzy scoring (exact equality, whole-word
//! containment, Levenshtein, and Jaro-Winkler), control kind filtering, and
//! resolution into unique, ambiguous, or missing matches.
//!
//! No operating system calls or logging of element names take place here.

/// A rectangular bounding box representing an element's position on screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    /// The x-coordinate of the left edge.
    pub left: i32,
    /// The y-coordinate of the top edge.
    pub top: i32,
    /// The x-coordinate of the right edge.
    pub right: i32,
    /// The y-coordinate of the bottom edge.
    pub bottom: i32,
}

impl Rect {
    /// Computes the centre coordinate of this rectangle.
    ///
    /// # Returns
    /// An `(x, y)` tuple representing the geometric centre point.
    pub fn centre(&self) -> (i32, i32) {
        ((self.left + self.right) / 2, (self.top + self.bottom) / 2)
    }
}

/// The kind of UI control represented by an on-screen element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlKind {
    /// A clickable button.
    Button,
    /// A clickable hyperlink.
    Link,
    /// An editable text input field.
    Edit,
    /// A toggleable check box.
    CheckBox,
    /// A drop-down combo box.
    ComboBox,
    /// An item in a context or application menu.
    MenuItem,
    /// A tab header in a tabbed interface.
    TabItem,
    /// An item in a list or tree view.
    ListItem,
    /// A radio selection button.
    RadioButton,
    /// Any other control type not specifically categorised.
    Other,
}

/// Summary of an on-screen element used for label matching and targeting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElementSummary {
    /// The accessible name or visual text label of the element.
    pub name: String,
    /// The UI control type.
    pub kind: ControlKind,
    /// The bounding box on the screen.
    pub rect: Rect,
    /// Whether the element is currently enabled for user interaction.
    pub enabled: bool,
    /// Whether the element is scrolled or positioned off-screen.
    pub offscreen: bool,
    /// Whether the element is a password/sensitive input field.
    pub is_password: bool,
}

/// A candidate element match with its similarity score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Match {
    /// Index of the candidate element in the input slice.
    pub index: usize,
    /// Similarity score between 0.0 and 1.0 (higher is better).
    pub score: f32,
}

/// Configuration thresholds for resolving ambiguous or low-scoring matches.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatchConfig {
    /// Minimum score required for a match to be considered valid.
    pub min_score: f32,
    /// Score difference between the top two matches below which the result is ambiguous.
    pub ambiguity_margin: f32,
}

impl Default for MatchConfig {
    /// Default thresholds: minimum score of 0.70 and ambiguity margin of 0.05.
    ///
    /// 0.70 rejects fuzzy noise such as "name" against "Home" (0.67) or
    /// "Managed bookmarks" (0.66), while still accepting a whole-word
    /// containment match such as "name" against "Full Name" (0.82).
    fn default() -> Self {
        Self {
            min_score: 0.70,
            ambiguity_margin: 0.05,
        }
    }
}

/// The outcome of resolving candidate matches against configuration thresholds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Resolution {
    /// Exactly one candidate clearly matched above threshold.
    Unique(usize),
    /// Multiple candidate elements matched with scores within the ambiguity margin.
    ///
    /// Contains up to the top 3 candidate indices.
    Ambiguous(Vec<usize>),
    /// No candidates met the minimum score threshold.
    NotFound,
}

/// Normalises a speech or UI text label for resilient matching.
///
/// Converts text to lowercase, replaces punctuation characters with whitespace,
/// collapses multiple whitespace sequences into a single space, and strips
/// leading English articles ("the").
///
/// # Arguments
/// * `label` - The raw input label from speech or an on-screen element.
///
/// # Returns
/// A cleaned, normalised string.
pub fn normalise_label(label: &str) -> String {
    let lower = label.to_lowercase();
    let cleaned: String = lower
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect();
    let words: Vec<&str> = cleaned.split_whitespace().collect();
    let remaining_words = match words.as_slice() {
        ["the", rest @ ..] => rest,
        all => all,
    };
    remaining_words.join(" ")
}

/// Checks if `needle_words` appears as a consecutive subslice of `haystack_words`.
///
/// # Arguments
/// * `needle_words` - Sequence of words to search for.
/// * `haystack_words` - Sequence of words to search within.
///
/// # Returns
/// `true` if `needle_words` is non-empty and is a contiguous subslice of `haystack_words`.
fn is_word_subslice(needle_words: &[&str], haystack_words: &[&str]) -> bool {
    if needle_words.is_empty() || needle_words.len() > haystack_words.len() {
        return false;
    }
    haystack_words
        .windows(needle_words.len())
        .any(|window| window == needle_words)
}

/// Calculates the similarity score between a normalised query and a normalised element name.
///
/// Evaluates:
/// 1. Exact equality (1.0).
/// 2. Whole-word containment (base 0.75 + length ratio bonus).
/// 3. Fuzzy matching using normalised Levenshtein and Jaro-Winkler.
///
/// # Arguments
/// * `norm_query` - Cleaned query string.
/// * `norm_element` - Cleaned element label.
///
/// # Returns
/// Similarity score between 0.0 and 1.0.
fn score_match(norm_query: &str, norm_element: &str) -> f32 {
    if norm_query == norm_element {
        return 1.0;
    }

    let q_words: Vec<&str> = norm_query.split_whitespace().collect();
    let e_words: Vec<&str> = norm_element.split_whitespace().collect();

    let is_containment =
        is_word_subslice(&q_words, &e_words) || is_word_subslice(&e_words, &q_words);

    let jaro = strsim::jaro_winkler(norm_query, norm_element) as f32;
    let lev = strsim::normalized_levenshtein(norm_query, norm_element) as f32;
    let fuzzy = jaro.max(lev);

    if is_containment {
        let shorter = norm_query.len().min(norm_element.len());
        let longer = norm_query.len().max(norm_element.len());
        let ratio = if longer > 0 {
            shorter as f32 / longer as f32
        } else {
            0.0
        };
        let containment = 0.75 + 0.15 * ratio;
        containment.max(fuzzy * 0.90)
    } else {
        fuzzy
    }
}

/// Ranks candidate elements by similarity to the query string.
///
/// Evaluates exact equality, whole-word containment, and fuzzy edit distances
/// (Levenshtein and Jaro-Winkler). Skips disabled, offscreen, and empty-named
/// elements. Optionally filters candidate elements by [`ControlKind`].
///
/// # Arguments
/// * `query` - Spoken or typed search query.
/// * `elements` - Candidate UI elements collected from the screen.
/// * `kinds` - Optional slice of allowed control types; if `None`, all control
///   types are eligible.
///
/// # Returns
/// A list of candidate matches with scores between 0.0 and 1.0, sorted best first.
pub fn rank_matches(
    query: &str,
    elements: &[ElementSummary],
    kinds: Option<&[ControlKind]>,
) -> Vec<Match> {
    let norm_query = normalise_label(query);
    if norm_query.is_empty() {
        return Vec::new();
    }

    let mut matches = Vec::new();

    for (index, elem) in elements.iter().enumerate() {
        // Skip disabled or offscreen elements.
        if !elem.enabled || elem.offscreen {
            continue;
        }

        // Apply control kind filter if specified.
        if let Some(allowed) = kinds
            && !allowed.contains(&elem.kind)
        {
            continue;
        }

        let norm_name = normalise_label(&elem.name);
        // Skip elements with empty normalised names.
        if norm_name.is_empty() {
            continue;
        }

        let score = score_match(&norm_query, &norm_name);
        if score > 0.0 {
            matches.push(Match { index, score });
        }
    }

    // Sort descending by score, breaking ties by original element index.
    matches.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.index.cmp(&b.index))
    });

    matches
}

/// Resolves ranked matches into a single outcome according to threshold configuration.
///
/// If no candidates meet `min_score`, returns [`Resolution::NotFound`].
/// If the top two scores differ by less than or equal to `ambiguity_margin`,
/// returns [`Resolution::Ambiguous`] containing up to the top 3 candidate indices.
/// Otherwise, returns [`Resolution::Unique`] with the best candidate index.
///
/// # Arguments
/// * `matches` - Candidate matches, sorted descending by score.
/// * `cfg` - Match threshold configuration.
///
/// # Returns
/// The resolved outcome: `Unique`, `Ambiguous`, or `NotFound`.
pub fn resolve(matches: &[Match], cfg: &MatchConfig) -> Resolution {
    let top = match matches.first() {
        Some(m) if m.score >= cfg.min_score => m,
        _ => return Resolution::NotFound,
    };

    if matches.len() == 1 {
        return Resolution::Unique(top.index);
    }

    // Check if the runner-up is within the ambiguity margin.
    if let Some(second) = matches.get(1)
        && (top.score - second.score) <= cfg.ambiguity_margin
    {
        let ambiguous: Vec<usize> = matches
            .iter()
            .filter(|m| m.score >= cfg.min_score && (top.score - m.score) <= cfg.ambiguity_margin)
            .take(3)
            .map(|m| m.index)
            .collect();
        return Resolution::Ambiguous(ambiguous);
    }

    Resolution::Unique(top.index)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to construct a standard active element for tests.
    fn make_element(name: &str, kind: ControlKind) -> ElementSummary {
        ElementSummary {
            name: name.to_owned(),
            kind,
            rect: Rect {
                left: 10,
                top: 10,
                right: 100,
                bottom: 50,
            },
            enabled: true,
            offscreen: false,
            is_password: false,
        }
    }

    /// Tests centre calculation of a bounding box.
    #[test]
    fn rect_centre_calculates_midpoint() {
        let rect = Rect {
            left: 10,
            top: 20,
            right: 30,
            bottom: 60,
        };
        assert_eq!(rect.centre(), (20, 40));
    }

    /// Tests normalisation of speech and UI labels.
    #[test]
    fn normalise_label_cleans_text() {
        assert_eq!(normalise_label("SUBMIT"), "submit");
        assert_eq!(normalise_label("Submit, please!"), "submit please");
        assert_eq!(normalise_label("  First   Name  "), "first name");
        assert_eq!(normalise_label("the submit button"), "submit button");
        assert_eq!(normalise_label("The Window"), "window");
        assert_eq!(normalise_label("the theme"), "theme");
        assert_eq!(normalise_label(""), "");
    }

    /// Tests exact query matching one element out of distinct options.
    #[test]
    fn submit_vs_submit_and_cancel_resolves_unique() {
        let elements = vec![
            make_element("Submit", ControlKind::Button),
            make_element("Cancel", ControlKind::Button),
        ];
        let matches = rank_matches("submit", &elements, None);
        let resolution = resolve(&matches, &MatchConfig::default());
        assert_eq!(resolution, Resolution::Unique(0));
    }

    /// Tests exact query prefers the exact element over one with extra words.
    #[test]
    fn submit_vs_submit_form_and_submit_resolves_exact() {
        let elements = vec![
            make_element("Submit form", ControlKind::Button),
            make_element("Submit", ControlKind::Button),
        ];
        let matches = rank_matches("submit", &elements, None);
        let resolution = resolve(&matches, &MatchConfig::default());
        assert_eq!(resolution, Resolution::Unique(1));
    }

    /// Tests query matching a common word in multiple elements produces an ambiguous resolution.
    #[test]
    fn name_vs_first_and_last_name_resolves_ambiguous() {
        let elements = vec![
            make_element("First name", ControlKind::Edit),
            make_element("Last name", ControlKind::Edit),
        ];
        let matches = rank_matches("name", &elements, None);
        let resolution = resolve(&matches, &MatchConfig::default());
        match resolution {
            Resolution::Ambiguous(indices) => {
                assert!(indices.contains(&0));
                assert!(indices.contains(&1));
                assert_eq!(indices.len(), 2);
            }
            other => panic!("expected Ambiguous, got {:?}", other),
        }
    }

    /// Tests slight typos in spoken input resolve to the intended target via fuzzy matching.
    #[test]
    fn emial_typo_vs_email_resolves_unique() {
        let elements = vec![make_element("Email", ControlKind::Edit)];
        let matches = rank_matches("emial", &elements, None);
        let resolution = resolve(&matches, &MatchConfig::default());
        assert_eq!(resolution, Resolution::Unique(0));
    }

    /// Tests completely unrelated input resolves to NotFound.
    #[test]
    fn zzz_resolves_not_found() {
        let elements = vec![
            make_element("Submit", ControlKind::Button),
            make_element("Cancel", ControlKind::Button),
        ];
        let matches = rank_matches("zzz", &elements, None);
        let resolution = resolve(&matches, &MatchConfig::default());
        assert_eq!(resolution, Resolution::NotFound);
    }

    /// Tests disabled and offscreen elements are skipped even on exact text match.
    #[test]
    fn disabled_and_offscreen_elements_are_skipped() {
        let mut disabled = make_element("Submit", ControlKind::Button);
        disabled.enabled = false;
        let mut offscreen = make_element("Submit", ControlKind::Button);
        offscreen.offscreen = true;

        let elements = vec![disabled, offscreen];
        let matches = rank_matches("submit", &elements, None);
        assert!(matches.is_empty());
        let resolution = resolve(&matches, &MatchConfig::default());
        assert_eq!(resolution, Resolution::NotFound);
    }

    /// Tests elements with empty or whitespace-only names are skipped.
    #[test]
    fn empty_element_names_are_skipped() {
        let elements = vec![
            make_element("", ControlKind::Button),
            make_element("   ", ControlKind::Button),
        ];
        let matches = rank_matches("submit", &elements, None);
        assert!(matches.is_empty());
    }

    /// Tests empty query resolves to NotFound.
    #[test]
    fn empty_query_resolves_not_found() {
        let elements = vec![make_element("Submit", ControlKind::Button)];
        let matches = rank_matches("", &elements, None);
        assert!(matches.is_empty());
        let resolution = resolve(&matches, &MatchConfig::default());
        assert_eq!(resolution, Resolution::NotFound);
    }

    /// Tests filtering by kind ignores elements with mismatched ControlKind.
    #[test]
    fn kinds_filter_excludes_unmatched_kind() {
        let elements = vec![make_element("Name", ControlKind::Button)];
        let matches = rank_matches("Name", &elements, Some(&[ControlKind::Edit]));
        assert!(matches.is_empty());
    }

    /// Tests kinds = None allows any ControlKind to match.
    #[test]
    fn kinds_none_allows_any_kind() {
        let elements = vec![make_element("Name", ControlKind::Button)];
        let matches = rank_matches("Name", &elements, None);
        assert!(!matches.is_empty());
        assert_eq!(matches[0].index, 0);
    }

    /// "name" must not fuzzy-match Chrome's Home button.
    #[test]
    fn name_does_not_match_home() {
        let elements = vec![make_element("Home", ControlKind::Button)];
        let matches = rank_matches("name", &elements, None);
        let resolution = resolve(&matches, &MatchConfig::default());
        assert_eq!(resolution, Resolution::NotFound, "matches: {matches:?}");
    }

    /// "name" must not fuzzy-match Chrome's Managed bookmarks folder.
    #[test]
    fn name_does_not_match_managed_bookmarks() {
        let elements = vec![make_element("Managed bookmarks", ControlKind::MenuItem)];
        let matches = rank_matches("name", &elements, None);
        let resolution = resolve(&matches, &MatchConfig::default());
        assert_eq!(resolution, Resolution::NotFound, "matches: {matches:?}");
    }

    /// On a Chrome-shaped screen, "name" must only match the Full Name field,
    /// never the Home button or the Managed bookmarks folder.
    #[test]
    fn name_matches_only_the_full_name_field_on_the_demo_form() {
        let elements = vec![
            make_element("Home", ControlKind::Button),
            make_element("Managed bookmarks", ControlKind::MenuItem),
            make_element("Full Name", ControlKind::Edit),
            make_element("Email", ControlKind::Edit),
            make_element("Matric Number", ControlKind::Edit),
            make_element("Submit", ControlKind::Button),
        ];
        let matches = rank_matches("name", &elements, None);
        let cfg = MatchConfig::default();
        let accepted: Vec<usize> = matches
            .iter()
            .filter(|m| m.score >= cfg.min_score)
            .map(|m| m.index)
            .collect();
        assert_eq!(
            accepted,
            vec![2],
            "Home or Managed bookmarks passed the threshold: {matches:?}"
        );
        assert_eq!(resolve(&matches, &cfg), Resolution::Unique(2));
    }

    /// The demo form's field labels resolve to their own fields.
    #[test]
    fn demo_form_labels_resolve_to_their_fields() {
        let elements = vec![
            make_element("Home", ControlKind::Button),
            make_element("Managed bookmarks", ControlKind::MenuItem),
            make_element("Full Name", ControlKind::Edit),
            make_element("Email", ControlKind::Edit),
            make_element("Matric Number", ControlKind::Edit),
            make_element("Submit", ControlKind::Button),
        ];
        for (query, expected) in [("full name", 2), ("email", 3), ("matric number", 4)] {
            let matches = rank_matches(query, &elements, None);
            assert_eq!(
                resolve(&matches, &MatchConfig::default()),
                Resolution::Unique(expected),
                "query {query:?} matches: {matches:?}"
            );
        }
    }
}
