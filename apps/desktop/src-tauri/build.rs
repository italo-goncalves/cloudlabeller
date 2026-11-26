// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

/// Every command the page may call. Each gets an `allow-<name>` permission,
/// which capabilities/default.json grants to the main window. A command
/// missing here can't be called from the page.
const COMMANDS: &[&str] = &[
    "environment_ensure",
    "environment_cancel",
    "environment_current",
    "folder_pick",
    "project_open",
    "project_close",
    "engine_call",
    "engine_cancel",
    "array_read",
    "array_write",
    "photo_read",
    "file_pick",
    "file_save",
    "file_read",
    "engine_path",
    "recent_list",
    "recent_add",
    "keep_awake",
    "job_run",
    "job_cancel",
    "gpu_lease",
    "update_check",
    "update_install",
];

fn main() {
    tauri_build::try_build(
        tauri_build::Attributes::new()
            .app_manifest(tauri_build::AppManifest::new().commands(COMMANDS)),
    )
    .expect("tauri build script failed");
}
