// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

//! CloudLabeller's desktop core (ADR-0002, ADR-0003, ADR-0004).
//!
//! The page talks only to this core. The core owns three things:
//! - the Environment, installed from the app's own Recipe with geokit;
//! - the Engine, one Python process per open Project, over JSON-RPC on its
//!   stdin/stdout;
//! - the Jobs: it starts each one the Engine plans, holds the GPU Lease for
//!   GPU Jobs, and stages a Workspace for COLMAP Jobs in WSL.

pub mod engine;
pub mod environment;
pub mod jobs;
pub mod project;
pub mod updates;

use geokit_env::process::Processes;
use tauri::Manager;

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .manage(Processes::default())
        .manage(project::Open::default())
        .manage(updates::Pending::default())
        .setup(|app| {
            environment::manage(app.handle())?;
            jobs::manage(app.handle());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            environment::environment_ensure,
            environment::environment_cancel,
            environment::environment_current,
            project::folder_pick,
            project::project_open,
            project::project_close,
            project::engine_call,
            project::engine_cancel,
            project::array_read,
            project::array_write,
            jobs::job_run,
            jobs::job_cancel,
            jobs::gpu_lease,
            updates::update_check,
            updates::update_install,
        ])
        .build(tauri::generate_context!())
        .expect("error while building the CloudLabeller desktop app")
        .run(|app, event| {
            if let tauri::RunEvent::Exit = event {
                // The Engine saves and stops; every process tree the core
                // started (installs, Jobs) ends with the app.
                if let Some(engine) = app.state::<project::Open>().engine.lock().expect("engine lock").take() {
                    engine.shutdown(std::time::Duration::from_secs(20));
                }
                app.state::<Processes>().end_all();
            }
        });
}
