// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

//! The open Project, as the page reaches it: picking its folder, its
//! Engine, and its Array files.
//!
//! A Project folder is reachable only once the user has picked it in the
//! core's own dialog (or created it there). The page reads and writes only
//! Array files, only inside that folder's `.arrays/`.

use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use geokit_env::envs::Environments;
use geokit_env::Problem;
use serde_json::{json, Value};
use tauri::ipc::{Channel, InvokeBody, Request, Response};
use tauri::{AppHandle, Manager, State};
use tauri_plugin_dialog::DialogExt;

use crate::engine::{Engine, Notify};
use crate::environment;

/// Methods the page may not call through `engine_call`: the core does them
/// itself (opening and closing, Jobs).
pub const RESERVED: &[&str] = &[
    "project.open",
    "project.create",
    "project.close",
    "engine.shutdown",
    "jobs.plan",
    "jobs.finish",
];

#[derive(Default)]
pub struct Open {
    pub engine: Mutex<Option<Arc<Engine>>>,
    picked: Mutex<HashSet<PathBuf>>,
}

impl Open {
    pub fn engine(&self) -> Result<Arc<Engine>, Problem> {
        self.engine
            .lock()
            .expect("engine lock")
            .clone()
            .filter(|e| e.running())
            .ok_or_else(|| Problem::new("project.none_open", "No Project is open."))
    }

    fn picked(&self, root: &Path) -> bool {
        self.picked.lock().expect("picked lock").contains(&key(root))
    }

    /// Whether the user picked this folder (or opened it as a recent Project).
    pub fn was_picked(&self, p: &Path) -> bool {
        self.picked(p)
    }

    /// A folder the user chose before (a recent Project), openable as if picked now.
    pub fn remember(&self, p: &Path) {
        self.picked.lock().expect("picked lock").insert(key(p));
    }
}

fn key(p: &Path) -> PathBuf {
    let s = environment::plain(p).replace('/', "\\");
    PathBuf::from(if cfg!(windows) { s.to_lowercase() } else { s })
}

/// `rel` as a path inside a Project's `.arrays/` folder: relative, `/`
/// separated, no `..`, no empty or reserved part (Windows device names,
/// trailing dots or spaces).
pub fn array_path(root: &Path, rel: &str) -> Result<PathBuf, Problem> {
    let bad = || Problem::new("array.bad_path", "Not an Array file path.").with("path", rel.to_string());
    let rest = rel.strip_prefix(".arrays/").ok_or_else(bad)?;
    let mut path = root.join(".arrays");
    for part in rest.split('/') {
        let reserved = {
            let stem = part.split('.').next().unwrap_or("").to_ascii_uppercase();
            matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
                || (stem.len() == 4
                    && (stem.starts_with("COM") || stem.starts_with("LPT"))
                    && stem.as_bytes()[3].is_ascii_digit())
        };
        if part.is_empty()
            || part == "."
            || part == ".."
            || part.ends_with('.')
            || part.ends_with(' ')
            || reserved
            || part.chars().any(|c| matches!(c, '<' | '>' | ':' | '"' | '\\' | '|' | '?' | '*') || (c as u32) < 0x20)
        {
            return Err(bad());
        }
        path.push(part);
    }
    if path.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(bad());
    }
    Ok(path)
}

fn page_notify(on: Channel<Value>) -> Notify {
    Arc::new(move |v: Value| {
        let _ = on.send(v);
    })
}

/// Ask the user for a Project folder (an existing one, or where a new one
/// goes). Only folders picked here can be opened.
#[tauri::command]
pub async fn folder_pick(app: AppHandle, title: String) -> Option<String> {
    let picked = app.dialog().file().set_title(title).blocking_pick_folder()?;
    let path = picked.into_path().ok()?;
    app.state::<Open>().picked.lock().expect("picked lock").insert(key(&path));
    Some(environment::plain(&path))
}

/// Open (or, with `create`, make) the Project at `root` in a new Engine,
/// closing the one before. The Engine's notifications arrive on `on`.
#[tauri::command]
pub async fn project_open(
    app: AppHandle,
    root: String,
    create: bool,
    classes: Option<Value>,
    on: Channel<Value>,
) -> Result<Value, Problem> {
    let open = app.state::<Open>();
    let root = PathBuf::from(root);
    let picked_parent = root.parent().is_some_and(|p| open.picked(p));
    if !open.picked(&root) && !(create && picked_parent) {
        return Err(Problem::new("project.not_picked", "Pick the Project's folder first."));
    }
    let kept = environment::current(&app.state::<Environments>())
        .ok_or_else(|| Problem::new("environment.not_ready", "The Environment isn't ready."))?;
    let log = app
        .path()
        .app_log_dir()
        .map_err(|e| Problem::new("engine.log", e.to_string()))?
        .join("engine.log");
    let worker = app.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let open = worker.state::<Open>();
        close_engine(&open);
        let engine = Engine::start(&kept.environment, &root, &log, page_notify(on))?;
        let method = if create { "project.create" } else { "project.open" };
        let mut params = json!({"root": engine.root_env});
        if let Some(classes) = classes {
            params["classes"] = classes;
        }
        match engine.call(method, params) {
            Ok(state) => {
                open.picked.lock().expect("picked lock").insert(key(&root));
                *open.engine.lock().expect("engine lock") = Some(engine);
                Ok(state)
            }
            Err(problem) => {
                engine.shutdown(Duration::from_secs(10));
                Err(problem)
            }
        }
    })
    .await
    .map_err(|e| Problem::new("engine.crashed", e.to_string()))?
}

fn close_engine(open: &Open) {
    if let Some(engine) = open.engine.lock().expect("engine lock").take() {
        if engine.running() {
            let _ = engine.call("project.close", json!({}));
            engine.shutdown(Duration::from_secs(20));
        }
    }
}

/// Save and close the Project, and stop its Engine.
#[tauri::command]
pub async fn project_close(app: AppHandle) -> Result<(), Problem> {
    let worker = app.clone();
    tauri::async_runtime::spawn_blocking(move || close_engine(&worker.state::<Open>()))
        .await
        .map_err(|e| Problem::new("engine.crashed", e.to_string()))
}

/// Any other Engine method (docs/engine.md), answered when it ends. With a
/// `ticket`, its `request.progress` notifications carry the ticket and
/// `engine_cancel` can cancel it.
#[tauri::command]
pub async fn engine_call(
    app: AppHandle,
    method: String,
    params: Value,
    ticket: Option<String>,
) -> Result<Value, Problem> {
    if RESERVED.contains(&method.as_str()) {
        return Err(Problem::new("engine.reserved", "The core makes that call itself.")
            .with("method", method));
    }
    let engine = app.state::<Open>().engine()?;
    tauri::async_runtime::spawn_blocking(move || match ticket {
        Some(t) => engine.call_ticketed(&method, params, &t),
        None => engine.call(&method, params),
    })
    .await
    .map_err(|e| Problem::new("engine.crashed", e.to_string()))?
}

/// Cancel the long request made under `ticket`.
#[tauri::command]
pub async fn engine_cancel(app: AppHandle, ticket: String) -> Result<bool, Problem> {
    let engine = app.state::<Open>().engine()?;
    tauri::async_runtime::spawn_blocking(move || engine.cancel_ticket(&ticket))
        .await
        .map_err(|e| Problem::new("engine.crashed", e.to_string()))?
}

/// An Array file of the open Project, as raw bytes.
#[tauri::command]
pub fn array_read(open: State<'_, Open>, path: String) -> Result<Response, Problem> {
    let engine = open.engine()?;
    let file = array_path(&engine.root, &path)?;
    let bytes = std::fs::read(&file).map_err(|e| {
        Problem::new("array.read", "The Array file can't be read.")
            .with("path", path.clone())
            .with("reason", e.to_string())
    })?;
    Ok(Response::new(bytes))
}

/// Write an inbound Array file: the raw body, at the `x-path` header.
#[tauri::command]
pub fn array_write(open: State<'_, Open>, request: Request<'_>) -> Result<(), Problem> {
    let engine = open.engine()?;
    let rel = request
        .headers()
        .get("x-path")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| Problem::new("array.bad_path", "No x-path header."))?
        .to_string();
    let file = array_path(&engine.root, &rel)?;
    let InvokeBody::Raw(bytes) = request.body() else {
        return Err(Problem::new("array.body", "The Array file's body must be raw bytes."));
    };
    if let Some(dir) = file.parent() {
        std::fs::create_dir_all(dir).map_err(|e| Problem::new("array.write", e.to_string()))?;
    }
    let part = file.with_extension("bin.part");
    std::fs::write(&part, bytes)
        .and_then(|_| std::fs::rename(&part, &file))
        .map_err(|e| {
            Problem::new("array.write", "The Array file can't be written.")
                .with("path", rel.clone())
                .with("reason", e.to_string())
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_paths_stay_inside_arrays() {
        let root = Path::new(r"C:\p.clproj");
        assert_eq!(
            array_path(root, ".arrays/mask/IMG_1.JPG.bin").unwrap(),
            root.join(".arrays").join("mask").join("IMG_1.JPG.bin")
        );
        for bad in [
            "labels/cloud.npy",
            ".arrays/../project.json",
            ".arrays/a//b.bin",
            ".arrays/CON.bin",
            ".arrays/com1",
            ".arrays/x.bin.",
            ".arrays/a\\b.bin",
            ".arrays/c:x.bin",
            ".arrays/",
        ] {
            assert!(array_path(root, bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_core_keeps_opening_and_jobs() {
        for m in ["project.open", "jobs.plan", "jobs.finish", "engine.shutdown"] {
            assert!(RESERVED.contains(&m));
        }
        assert!(!RESERVED.contains(&"transfer.images_to_cloud"));
    }
}
