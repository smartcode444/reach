//! Build script for the Reach status window.
//!
//! Tauri needs a build step to validate `tauri.conf.json`, embed the Windows
//! resource file and expose the generated context to `generate_context!`.

/// Runs Tauri's build helper.
///
/// # Panics
/// Panics when the configuration is invalid or the Windows icon is missing;
/// both are build-time mistakes that must stop the build.
fn main() {
    tauri_build::build();
}
