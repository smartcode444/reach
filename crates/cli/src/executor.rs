//! Composite executor re-exported from the shared runtime.
//!
//! The CLI and the desktop app must inject input through exactly the same
//! executor, so the implementation lives in `reach-runtime`.

pub use reach_runtime::executor::CompositeExecutor;
