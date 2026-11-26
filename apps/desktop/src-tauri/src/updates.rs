// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

//! Updates: the signed installer and `latest.json` attached to each GitHub
//! Release of italo-goncalves/cloudlabeller (tauri.conf.json's endpoint).
//!
//! The page asks once at start (`update_check`) and offers what it finds,
//! with Later; `update_install` downloads it, checks its signature against
//! the public key in tauri.conf.json, installs it and restarts. While the
//! public key is empty (no signing key made yet), nothing is checked.

use std::sync::Mutex;

use geokit_env::Problem;
use serde::Serialize;
use tauri::ipc::Channel;
use tauri::{AppHandle, Manager};
use tauri_plugin_updater::{Update, UpdaterExt};

#[derive(Default)]
pub struct Pending(Mutex<Option<Update>>);

#[derive(Clone, Debug, Serialize)]
pub struct Offered {
    pub version: String,
    pub current: String,
    pub notes: Option<String>,
    pub date: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Downloading {
    Progress { done: u64, total: Option<u64> },
    Installing,
}

fn problem(e: impl std::fmt::Display) -> Problem {
    Problem::new("update.failed", "The update could not be checked or installed.")
        .with("reason", e.to_string())
}

fn signing_configured(app: &AppHandle) -> bool {
    app.config()
        .plugins
        .0
        .get("updater")
        .and_then(|u| u.get("pubkey"))
        .and_then(|k| k.as_str())
        .is_some_and(|k| !k.trim().is_empty())
}

/// A newer version, if there is one (none while signing isn't set up).
#[tauri::command]
pub async fn update_check(app: AppHandle) -> Result<Option<Offered>, Problem> {
    if !signing_configured(&app) || cfg!(debug_assertions) {
        return Ok(None);
    }
    let found = app.updater().map_err(problem)?.check().await.map_err(problem)?;
    let offered = found.as_ref().map(|u| Offered {
        version: u.version.clone(),
        current: u.current_version.clone(),
        notes: u.body.clone(),
        date: u.date.map(|d| d.to_string()),
    });
    *app.state::<Pending>().0.lock().expect("update lock") = found;
    Ok(offered)
}

/// Download and install the update `update_check` found, then restart.
#[tauri::command]
pub async fn update_install(app: AppHandle, on: Channel<Downloading>) -> Result<(), Problem> {
    let update = app
        .state::<Pending>()
        .0
        .lock()
        .expect("update lock")
        .take()
        .ok_or_else(|| Problem::new("update.none", "There is no update to install."))?;
    let mut done = 0u64;
    let progress = on.clone();
    update
        .download_and_install(
            move |chunk, total| {
                done += chunk as u64;
                let _ = progress.send(Downloading::Progress { done, total });
            },
            move || {
                let _ = on.send(Downloading::Installing);
            },
        )
        .await
        .map_err(problem)?;
    app.restart();
}
