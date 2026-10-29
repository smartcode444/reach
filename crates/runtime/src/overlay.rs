//! Overlay event types for drawing numbered badges and 3x3 mouse grid.
//!
//! Provides data types sent from the runtime executor to the visual overlay window.
//! No desktop injection or UI automation calls take place here.

use screen::{GridError, GridState, Rect, grid_cells};

/// A numbered badge item to display on the overlay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NumberedBadge {
    /// 1-based number to display.
    pub number: u32,
    /// Bounding rectangle of the target element.
    pub rect: Rect,
}

/// Commands and updates sent to the overlay renderer window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OverlayEvent {
    /// Show or replace the numbered badges on screen.
    ShowNumbers(Vec<NumberedBadge>),
    /// Hide the numbered badges.
    HideNumbers,
    /// Show or zoom the 3x3 mouse grid.
    ShowGrid {
        /// Current bounding box of the grid.
        bounds: Rect,
        /// Current zoom level (1..=3).
        level: u32,
        /// The 9 sub-rectangles for cells 1..=9.
        cells: Vec<(u32, Rect)>,
    },
    /// Hide the mouse grid.
    HideGrid,
    /// Clear all overlay visuals (badges and grid).
    HideAll,
}

/// Helper to calculate grid cells and construct an [`OverlayEvent::ShowGrid`].
///
/// # Arguments
/// * `grid` - The active grid state.
///
/// # Returns
/// An [`OverlayEvent::ShowGrid`] containing the current bounds, level, and 9 cell rects.
///
/// # Errors
/// Returns [`GridError`] if the current bounds are degenerate.
pub fn build_grid_overlay_event(grid: &GridState) -> Result<OverlayEvent, GridError> {
    let cells = grid_cells(&grid.current_rect)?;
    Ok(OverlayEvent::ShowGrid {
        bounds: grid.current_rect,
        level: grid.level,
        cells,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies build_grid_overlay_event builds 9 cells for a valid rect.
    #[test]
    fn build_grid_overlay_event_constructs_nine_cells() {
        let grid = GridState::new(Rect {
            left: 0,
            top: 0,
            right: 300,
            bottom: 300,
        });
        let event = build_grid_overlay_event(&grid).expect("valid grid event");
        match event {
            OverlayEvent::ShowGrid {
                bounds,
                level,
                cells,
            } => {
                assert_eq!(bounds, grid.current_rect);
                assert_eq!(level, 1);
                assert_eq!(cells.len(), 9);
            }
            _ => panic!("expected ShowGrid event"),
        }
    }
}
