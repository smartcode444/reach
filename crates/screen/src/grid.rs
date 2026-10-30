//! Pure functions and geometry for the 3x3 recursive mouse grid.
//!
//! Provides coordinate subdivision, cell rect calculation, and recursive 3-level
//! zooming terminating in absolute center click coordinates.
//!
//! No operating system calls or logging take place here.

use crate::error::ScreenError;
use crate::label::Rect;

/// Maximum zoom depth before a cell selection clicks the centre.
pub const MAX_GRID_LEVEL: u32 = 3;

/// An error encountered in grid calculations or selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GridError {
    /// The specified cell number was outside the valid 1..=9 range.
    InvalidCell(u32),
    /// The grid rectangle has degenerate (zero or negative) dimensions.
    EmptyRect,
}

impl std::fmt::Display for GridError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCell(cell) => {
                write!(f, "invalid grid cell {cell}; must be between 1 and 9")
            }
            Self::EmptyRect => write!(f, "grid rectangle has empty or degenerate dimensions"),
        }
    }
}

impl std::error::Error for GridError {}

/// The outcome of selecting a cell in the mouse grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GridOutcome {
    /// The grid zoomed into a sub-rect for the next level.
    Zoomed {
        /// The new zoom level (2 or 3).
        level: u32,
        /// The sub-rect for the new level.
        rect: Rect,
    },
    /// The final level (3) was reached; the centre of the chosen cell must be clicked.
    ClickAt {
        /// Absolute X coordinate.
        x: i32,
        /// Absolute Y coordinate.
        y: i32,
    },
}

/// The state of an active 3x3 recursive mouse grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GridState {
    /// Bounding rectangle of the current zoom level.
    pub current_rect: Rect,
    /// Current zoom level, starting at 1 up to [`MAX_GRID_LEVEL`].
    pub level: u32,
}

impl GridState {
    /// Initializes a new 3x3 grid over `rect` at level 1.
    ///
    /// # Arguments
    /// * `rect` - Bounding rectangle of the grid (typically the foreground window).
    ///
    /// # Returns
    /// An initialized [`GridState`].
    pub fn new(rect: Rect) -> Self {
        Self {
            current_rect: rect,
            level: 1,
        }
    }

    /// Selects a cell (1 to 9), either zooming into the cell or triggering the final click.
    ///
    /// At levels 1 and 2, this narrows `current_rect` to the selected cell and increments
    /// `level`. At level 3, this computes the centre coordinate of the selected cell and
    /// returns [`GridOutcome::ClickAt`].
    ///
    /// # Arguments
    /// * `cell` - 1-based cell index between 1 and 9.
    ///
    /// # Returns
    /// A [`GridOutcome`] indicating whether the grid zoomed or reached the final click point.
    ///
    /// # Errors
    /// Returns [`GridError::InvalidCell`] if `cell` is not in 1..=9, or [`GridError::EmptyRect`]
    /// if the active rectangle has become degenerate.
    pub fn select(&mut self, cell: u32) -> Result<GridOutcome, GridError> {
        let sub_rect = cell_rect(&self.current_rect, cell)?;

        if self.level < MAX_GRID_LEVEL {
            self.level += 1;
            self.current_rect = sub_rect;
            Ok(GridOutcome::Zoomed {
                level: self.level,
                rect: self.current_rect,
            })
        } else {
            let (x, y) = sub_rect.centre();
            Ok(GridOutcome::ClickAt { x, y })
        }
    }
}

/// Calculates the sub-rectangle for cell `cell` (1 to 9) within `rect`.
///
/// Cells are numbered 1 to 9 in reading order:
/// - 1, 2, 3: top row (left, middle, right)
/// - 4, 5, 6: middle row (left, middle, right)
/// - 7, 8, 9: bottom row (left, middle, right)
///
/// # Arguments
/// * `rect` - The parent rectangle.
/// * `cell` - The 1-based cell index.
///
/// # Returns
/// The bounding [`Rect`] of the cell.
///
/// # Errors
/// Returns [`GridError::InvalidCell`] if `cell` is not in 1..=9, or [`GridError::EmptyRect`]
/// if the dimensions are degenerate.
pub fn cell_rect(rect: &Rect, cell: u32) -> Result<Rect, GridError> {
    if !(1..=9).contains(&cell) {
        return Err(GridError::InvalidCell(cell));
    }
    let width = rect.right - rect.left;
    let height = rect.bottom - rect.top;
    if width <= 0 || height <= 0 {
        return Err(GridError::EmptyRect);
    }

    let col = ((cell - 1) % 3) as i32;
    let row = ((cell - 1) / 3) as i32;

    let left = rect.left + (col * width) / 3;
    let right = rect.left + ((col + 1) * width) / 3;
    let top = rect.top + (row * height) / 3;
    let bottom = rect.top + ((row + 1) * height) / 3;

    Ok(Rect {
        left,
        top,
        right,
        bottom,
    })
}

/// Returns the 9 sub-rectangles for a 3x3 subdivision of `rect`, paired with their 1-based numbers.
///
/// # Arguments
/// * `rect` - The parent rectangle.
///
/// # Returns
/// A vector of 9 `(u32, Rect)` pairs for cells 1 to 9.
///
/// # Errors
/// Returns [`GridError::EmptyRect`] if the dimensions are degenerate.
pub fn grid_cells(rect: &Rect) -> Result<Vec<(u32, Rect)>, GridError> {
    let mut cells = Vec::with_capacity(9);
    for cell in 1..=9 {
        let r = cell_rect(rect, cell)?;
        cells.push((cell, r));
    }
    Ok(cells)
}

/// Maps an absolute `(x, y)` point to its 1-based cell number (1 to 9) within `rect`.
///
/// # Arguments
/// * `rect` - The parent bounding box.
/// * `x` - Absolute horizontal coordinate.
/// * `y` - Absolute vertical coordinate.
///
/// # Returns
/// `Some(cell)` if `(x, y)` is within `rect`, or `None` if outside.
pub fn point_to_cell(rect: &Rect, x: i32, y: i32) -> Option<u32> {
    if x < rect.left || x >= rect.right || y < rect.top || y >= rect.bottom {
        return None;
    }
    let width = rect.right - rect.left;
    let height = rect.bottom - rect.top;
    if width <= 0 || height <= 0 {
        return None;
    }

    let col = ((x - rect.left) * 3) / width;
    let row = ((y - rect.top) * 3) / height;

    let col = col.clamp(0, 2);
    let row = row.clamp(0, 2);

    Some((row * 3 + col + 1) as u32)
}

/// Obtains the bounding rectangle of the current foreground window on Windows.
///
/// # Errors
/// Returns [`ScreenError::NoForegroundWindow`] if no window is active or its rect
/// cannot be obtained.
pub fn get_foreground_window_rect() -> Result<Rect, ScreenError> {
    use crate::element::{ForegroundFinder, SystemForegroundFinder};
    let finder = SystemForegroundFinder;
    let hwnd_val = finder.get_foreground_hwnd()?;
    #[cfg(windows)]
    {
        use windows::Win32::Foundation::{HWND, RECT};
        use windows::Win32::UI::WindowsAndMessaging::GetWindowRect;
        let mut rect = RECT::default();
        let hwnd = HWND(hwnd_val as *mut core::ffi::c_void);
        // SAFETY: GetWindowRect is called with a valid or non-null HWND and a valid pointer to a stack-allocated RECT.
        let ok = unsafe { GetWindowRect(hwnd, &mut rect) };
        if ok.is_ok() {
            Ok(Rect {
                left: rect.left,
                top: rect.top,
                right: rect.right,
                bottom: rect.bottom,
            })
        } else {
            Err(ScreenError::NoForegroundWindow)
        }
    }
    #[cfg(not(windows))]
    {
        let _ = hwnd_val;
        Err(ScreenError::NoForegroundWindow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::label::Rect;

    /// Numbering and rect-to-cell geometry.
    #[test]
    fn numbering_and_rect_to_cell_geometry() {
        let bounds = Rect {
            left: 0,
            top: 0,
            right: 300,
            bottom: 300,
        };
        let cells = grid_cells(&bounds).expect("valid rect");
        assert_eq!(cells.len(), 9);

        let expected = [
            (
                1,
                Rect {
                    left: 0,
                    top: 0,
                    right: 100,
                    bottom: 100,
                },
            ),
            (
                2,
                Rect {
                    left: 100,
                    top: 0,
                    right: 200,
                    bottom: 100,
                },
            ),
            (
                3,
                Rect {
                    left: 200,
                    top: 0,
                    right: 300,
                    bottom: 100,
                },
            ),
            (
                4,
                Rect {
                    left: 0,
                    top: 100,
                    right: 100,
                    bottom: 200,
                },
            ),
            (
                5,
                Rect {
                    left: 100,
                    top: 100,
                    right: 200,
                    bottom: 200,
                },
            ),
            (
                6,
                Rect {
                    left: 200,
                    top: 100,
                    right: 300,
                    bottom: 200,
                },
            ),
            (
                7,
                Rect {
                    left: 0,
                    top: 200,
                    right: 100,
                    bottom: 300,
                },
            ),
            (
                8,
                Rect {
                    left: 100,
                    top: 200,
                    right: 200,
                    bottom: 300,
                },
            ),
            (
                9,
                Rect {
                    left: 200,
                    top: 200,
                    right: 300,
                    bottom: 300,
                },
            ),
        ];

        for (num, exp_rect) in expected {
            let got_rect = cell_rect(&bounds, num).expect("valid cell");
            assert_eq!(got_rect, exp_rect, "mismatch for cell {num}");
            let found = cells.iter().find(|(n, _)| *n == num).map(|(_, r)| *r);
            assert_eq!(found, Some(exp_rect), "grid_cells mismatch for cell {num}");

            let (cx, cy) = exp_rect.centre();
            assert_eq!(point_to_cell(&bounds, cx, cy), Some(num));
        }
    }

    /// Grid subdivision maths and the final click point.
    #[test]
    fn grid_subdivision_maths_and_the_final_click_point() {
        let bounds = Rect {
            left: 0,
            top: 0,
            right: 270,
            bottom: 270,
        };
        let mut grid = GridState::new(bounds);
        assert_eq!(grid.level, 1);

        // Level 1 -> select cell 5 (centre of 270x270: 90..180 x 90..180)
        let outcome1 = grid.select(5).expect("valid select");
        assert_eq!(
            outcome1,
            GridOutcome::Zoomed {
                level: 2,
                rect: Rect {
                    left: 90,
                    top: 90,
                    right: 180,
                    bottom: 180,
                },
            }
        );
        assert_eq!(grid.level, 2);

        // Level 2 -> select cell 1 (top-left of 90..180: 90..120 x 90..120)
        let outcome2 = grid.select(1).expect("valid select");
        assert_eq!(
            outcome2,
            GridOutcome::Zoomed {
                level: 3,
                rect: Rect {
                    left: 90,
                    top: 90,
                    right: 120,
                    bottom: 120,
                },
            }
        );
        assert_eq!(grid.level, 3);

        // Level 3 -> final select cell 9 (bottom-right of 90..120: 110..120 x 110..120)
        // Centre is (115, 115)
        let outcome3 = grid.select(9).expect("valid select");
        assert_eq!(outcome3, GridOutcome::ClickAt { x: 115, y: 115 });
    }

    /// Invalid cell number produces an error.
    #[test]
    fn invalid_cell_number_returns_error() {
        let bounds = Rect {
            left: 0,
            top: 0,
            right: 100,
            bottom: 100,
        };
        assert_eq!(cell_rect(&bounds, 0), Err(GridError::InvalidCell(0)));
        assert_eq!(cell_rect(&bounds, 10), Err(GridError::InvalidCell(10)));

        let mut grid = GridState::new(bounds);
        assert_eq!(grid.select(0), Err(GridError::InvalidCell(0)));
        assert_eq!(grid.select(10), Err(GridError::InvalidCell(10)));
    }

    /// Degenerate rect produces EmptyRect error.
    #[test]
    fn degenerate_rect_returns_empty_rect_error() {
        let zero_rect = Rect {
            left: 10,
            top: 10,
            right: 10,
            bottom: 10,
        };
        assert_eq!(cell_rect(&zero_rect, 1), Err(GridError::EmptyRect));
        assert_eq!(grid_cells(&zero_rect), Err(GridError::EmptyRect));
    }
}
