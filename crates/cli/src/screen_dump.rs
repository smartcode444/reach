//! Dev-only screen-dump utility for dumping on-screen elements to stdout.
//!
//! # Warning
//! This module prints live on-screen element information (including text labels,
//! control types, and coordinates) to stdout. It is intended for development and
//! debugging only, and should only be used on the demo form, never when sensitive
//! personal or private data is visible on screen.

use screen::{CollectLimits, ControlKind, LiveElement, Rect, collect_foreground};
use std::time::Duration;

/// Parses command-line arguments for the `screen-dump` subcommand.
///
/// Accepts `--delay <seconds>` or `--delay=<seconds>`. If omitted, defaults to 0 seconds.
///
/// # Arguments
/// * `args` - Arguments following the `screen-dump` subcommand.
///
/// # Returns
/// The parsed [`Duration`] to sleep before capturing the screen.
///
/// # Errors
/// Returns an error message if an argument is unknown, `--delay` has no value,
/// or the delay cannot be parsed as a non-negative number of seconds.
pub fn parse_screen_dump_args(args: &[String]) -> Result<Duration, String> {
    let mut delay_secs = 0.0f64;
    let mut iter = args.iter().peekable();

    while let Some(arg) = iter.next() {
        if arg == "--delay" {
            let val = iter
                .next()
                .ok_or_else(|| "screen-dump: --delay requires a duration in seconds".to_owned())?;
            delay_secs = val
                .parse::<f64>()
                .map_err(|e| format!("screen-dump: invalid delay '{val}': {e}"))?;
            if delay_secs < 0.0 {
                return Err("screen-dump: delay must be non-negative".to_owned());
            }
        } else if let Some(val) = arg.strip_prefix("--delay=") {
            delay_secs = val
                .parse::<f64>()
                .map_err(|e| format!("screen-dump: invalid delay '{val}': {e}"))?;
            if delay_secs < 0.0 {
                return Err("screen-dump: delay must be non-negative".to_owned());
            }
        } else {
            return Err(format!("screen-dump: unknown option '{arg}'"));
        }
    }

    Ok(Duration::from_secs_f64(delay_secs))
}

/// Formats a single collected element into a display line.
///
/// # Arguments
/// * `depth` - Traversal hierarchy depth.
/// * `kind` - Control type kind.
/// * `name` - Accessible text label or name.
/// * `rect` - On-screen bounding rectangle.
///
/// # Returns
/// A formatted string representing the element.
pub fn format_element_line(depth: usize, kind: ControlKind, name: &str, rect: Rect) -> String {
    format!("{depth}, {kind:?}, {name:?}, {rect:?}")
}

/// Formats the collection results into lines for printing, skipping password fields.
///
/// # Arguments
/// * `elements` - Collected live elements from the screen.
/// * `truncated` - Whether collection hit depth, count, or time limits.
///
/// # Returns
/// A vector of output lines including elements and the final truncation indicator.
pub fn format_screen_dump_output(elements: &[LiveElement], truncated: bool) -> Vec<String> {
    let mut lines = Vec::new();
    for elem in elements {
        // Skip password fields entirely to prevent leaking sensitive UI controls.
        if elem.summary.is_password {
            continue;
        }
        lines.push(format_element_line(
            elem.depth,
            elem.summary.kind,
            &elem.summary.name,
            elem.summary.rect,
        ));
    }
    lines.push(format!("truncated: {truncated}"));
    lines
}

/// Runs the dev-only `screen-dump` subcommand.
///
/// Waits for the specified delay, then collects elements from the active foreground window
/// and prints each element to standard output, skipping password fields entirely.
///
/// # Warning
/// This function prints live window contents (including element names and coordinates)
/// directly to stdout. It is intended for development and debugging only, and must only
/// be used on the demo form, never with sensitive personal or private data on screen.
///
/// # Arguments
/// * `args` - Arguments following the `screen-dump` subcommand (e.g. `["--delay", "2"]`).
///
/// # Returns
/// `Ok(())` on successful collection and printing.
///
/// # Errors
/// Returns an error message if argument parsing or screen collection fails.
pub fn run_screen_dump(args: &[String]) -> Result<(), String> {
    let delay = parse_screen_dump_args(args)?;
    if !delay.is_zero() {
        std::thread::sleep(delay);
    }

    let collected = collect_foreground(CollectLimits::DOCUMENT)
        .map_err(|e| format!("screen-dump failed to collect foreground: {e}"))?;

    let lines = format_screen_dump_output(&collected.elements, collected.truncated);
    for line in lines {
        println!("{line}");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use screen::ElementSummary;

    /// Tests that parse_screen_dump_args parses standard --delay <seconds>.
    #[test]
    fn parse_screen_dump_args_delay_space() {
        let args = vec!["--delay".to_string(), "2".to_string()];
        let delay = parse_screen_dump_args(&args).unwrap();
        assert_eq!(delay, Duration::from_secs(2));
    }

    /// Tests that parse_screen_dump_args parses --delay=<seconds> with fractional seconds.
    #[test]
    fn parse_screen_dump_args_delay_equals_float() {
        let args = vec!["--delay=1.5".to_string()];
        let delay = parse_screen_dump_args(&args).unwrap();
        assert_eq!(delay, Duration::from_millis(1500));
    }

    /// Tests that parse_screen_dump_args defaults to zero duration when no args passed.
    #[test]
    fn parse_screen_dump_args_default_zero() {
        let args: Vec<String> = vec![];
        let delay = parse_screen_dump_args(&args).unwrap();
        assert_eq!(delay, Duration::ZERO);
    }

    /// Tests that parse_screen_dump_args errors when --delay has no value.
    #[test]
    fn parse_screen_dump_args_missing_value() {
        let args = vec!["--delay".to_string()];
        let res = parse_screen_dump_args(&args);
        assert!(res.is_err());
    }

    /// Tests that parse_screen_dump_args errors on invalid number format.
    #[test]
    fn parse_screen_dump_args_invalid_number() {
        let args = vec!["--delay".to_string(), "abc".to_string()];
        let res = parse_screen_dump_args(&args);
        assert!(res.is_err());
    }

    /// Tests that parse_screen_dump_args errors on negative delay.
    #[test]
    fn parse_screen_dump_args_negative_delay() {
        let args = vec!["--delay".to_string(), "-1".to_string()];
        let res = parse_screen_dump_args(&args);
        assert!(res.is_err());
    }

    /// Tests that format_screen_dump_output skips password fields and includes truncation status.
    #[test]
    fn format_screen_dump_output_skips_passwords_and_prints_truncated() {
        let normal_summary = ElementSummary {
            name: "Username".to_owned(),
            kind: ControlKind::Edit,
            rect: Rect {
                left: 10,
                top: 20,
                right: 110,
                bottom: 50,
            },
            enabled: true,
            offscreen: false,
            is_password: false,
        };
        let password_summary = ElementSummary {
            name: "Password".to_owned(),
            kind: ControlKind::Edit,
            rect: Rect {
                left: 10,
                top: 60,
                right: 110,
                bottom: 90,
            },
            enabled: true,
            offscreen: false,
            is_password: true,
        };

        let elements = vec![
            LiveElement::from_summary_with_depth(normal_summary, 2),
            LiveElement::from_summary_with_depth(password_summary, 2),
        ];

        let output = format_screen_dump_output(&elements, false);

        // Password field must be skipped: only 1 element line + 1 truncation line = 2 lines.
        assert_eq!(output.len(), 2);
        assert!(output[0].contains("Username"));
        assert!(output[0].contains("Edit"));
        assert_eq!(output[1], "truncated: false");
    }

    /// Tests that format_screen_dump_output reports truncated: true when collection truncated.
    #[test]
    fn format_screen_dump_output_reports_truncated_true() {
        let elements = vec![];
        let output = format_screen_dump_output(&elements, true);
        assert_eq!(output, vec!["truncated: true"]);
    }
}
