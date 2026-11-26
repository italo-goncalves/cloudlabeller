// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

//! Files the page reaches besides Array files: the open Project's photos,
//! files the user picked (a mask to import, where an export goes), the
//! recent Projects, and keeping the computer awake through a long pipeline.
//!
//! As with Project folders, the page names a file outside the Project only
//! once the user has chosen it in the core's own dialog.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Sender};
use std::sync::Mutex;

use geokit_env::Problem;
use serde::Deserialize;
use tauri::ipc::Response;
use tauri::{AppHandle, Manager, State};
use tauri_plugin_dialog::DialogExt;

use crate::engine::env_path;
use crate::environment;
use crate::project::Open;

/// The file a Project keeps its schema and settings in: a folder holding
/// one is a Project.
pub const MANIFEST: &str = "project.json";
/// How many recent Projects are kept.
pub const RECENT_KEPT: usize = 12;

#[derive(Default)]
pub struct Files {
    chosen: Mutex<HashSet<PathBuf>>,
    awake: Mutex<Option<Sender<()>>>,
}

impl Files {
    fn chose(&self, p: &Path) {
        self.chosen.lock().expect("files lock").insert(key(p));
    }

    fn chosen(&self, p: &Path) -> bool {
        self.chosen.lock().expect("files lock").contains(&key(p))
    }
}

fn key(p: &Path) -> PathBuf {
    let s = environment::plain(p).replace('/', "\\");
    PathBuf::from(if cfg!(windows) { s.to_lowercase() } else { s })
}

/// A Survey's id: lowercase ASCII letters, digits and `-` (`s1`, `s12`).
fn survey_id_ok(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 32
        && id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

/// `name` as a photo of Survey `survey` of the Project at `root`: one plain
/// file name in `surveys/<survey>/images/`.
pub fn photo_path(root: &Path, survey: &str, name: &str) -> Result<PathBuf, Problem> {
    if !survey_id_ok(survey) {
        return Err(Problem::new("photo.bad_survey", "Not a Survey's id.").with("survey", survey.to_string()));
    }
    let ok = !name.is_empty()
        && name != "."
        && name != ".."
        && !name.ends_with('.')
        && !name.ends_with(' ')
        && !name.chars().any(|c| matches!(c, '/' | '\\' | ':' | '<' | '>' | '"' | '|' | '?' | '*') || (c as u32) < 0x20);
    if ok {
        Ok(root.join("surveys").join(survey).join("images").join(name))
    } else {
        Err(Problem::new("photo.bad_name", "Not a photo's name.").with("name", name.to_string()))
    }
}

/// A file filter for the dialogs: its name and extensions without the dot.
#[derive(Debug, Clone, Deserialize)]
pub struct Filter {
    pub name: String,
    pub extensions: Vec<String>,
}

/// A photo of a Survey of the open Project, as its file's bytes.
#[tauri::command]
pub fn photo_read(open: State<'_, Open>, survey: String, name: String) -> Result<Response, Problem> {
    let engine = open.engine()?;
    let file = photo_path(&engine.root, &survey, &name)?;
    std::fs::read(&file).map(Response::new).map_err(|e| {
        Problem::new("photo.read", "The photo can't be read.")
            .with("name", name.clone())
            .with("reason", e.to_string())
    })
}

/// Ask the user for a file to read (a mask to import).
#[tauri::command]
pub async fn file_pick(app: AppHandle, title: String, filters: Vec<Filter>) -> Option<String> {
    let mut d = app.dialog().file().set_title(title);
    for f in &filters {
        let ext: Vec<&str> = f.extensions.iter().map(String::as_str).collect();
        d = d.add_filter(&f.name, &ext);
    }
    let path = d.blocking_pick_file()?.into_path().ok()?;
    app.state::<Files>().chose(&path);
    Some(environment::plain(&path))
}

/// Ask the user where a file goes (an export).
#[tauri::command]
pub async fn file_save(app: AppHandle, title: String, name: String, filters: Vec<Filter>) -> Option<String> {
    let mut d = app.dialog().file().set_title(title).set_file_name(name);
    for f in &filters {
        let ext: Vec<&str> = f.extensions.iter().map(String::as_str).collect();
        d = d.add_filter(&f.name, &ext);
    }
    let path = d.blocking_save_file()?.into_path().ok()?;
    app.state::<Files>().chose(&path);
    Some(environment::plain(&path))
}

/// A file the user picked, as its bytes.
#[tauri::command]
pub fn file_read(files: State<'_, Files>, path: String) -> Result<Response, Problem> {
    let p = PathBuf::from(&path);
    if !files.chosen(&p) {
        return Err(Problem::new("file.not_picked", "Pick the file first.").with("path", path));
    }
    std::fs::read(&p).map(Response::new).map_err(|e| {
        Problem::new("file.read", "The file can't be read.").with("path", path.clone()).with("reason", e.to_string())
    })
}

/// A folder or file the user picked, as the open Project's Engine sees it
/// (`/mnt/c/…` in WSL): what the page hands the Engine for an import or an export.
#[tauri::command]
pub fn engine_path(app: AppHandle, path: String) -> Result<String, Problem> {
    let open = app.state::<Open>();
    let engine = open.engine()?;
    let p = PathBuf::from(&path);
    if !app.state::<Files>().chosen(&p) && !open.was_picked(&p) {
        return Err(Problem::new("file.not_picked", "Pick it first.").with("path", path));
    }
    env_path(&engine.environment, &p)
}

// -- recent Projects ------------------------------------------------------------

fn recent_file(app: &AppHandle) -> Result<PathBuf, Problem> {
    Ok(app
        .path()
        .app_local_data_dir()
        .map_err(|e| Problem::new("recent.dir", e.to_string()))?
        .join("recent.json"))
}

/// The list in `file`, newest first; a missing or unreadable one is empty.
pub fn read_recent(file: &Path) -> Vec<String> {
    std::fs::read_to_string(file)
        .ok()
        .and_then(|s| serde_json::from_str::<Vec<String>>(&s).ok())
        .unwrap_or_default()
}

/// `root` put first in the list, once, the list cut to `RECENT_KEPT`.
pub fn with_recent(list: &[String], root: &str) -> Vec<String> {
    let k = key(Path::new(root));
    let mut out = vec![root.to_string()];
    out.extend(list.iter().filter(|r| key(Path::new(r)) != k).cloned());
    out.truncate(RECENT_KEPT);
    out
}

/// Those of `list` that still hold a Project.
pub fn still_there(list: &[String]) -> Vec<String> {
    list.iter().filter(|r| Path::new(r).join(MANIFEST).is_file()).cloned().collect()
}

/// The recent Projects that still exist, newest first. Each becomes openable
/// as if picked: the user chose it before.
#[tauri::command]
pub fn recent_list(app: AppHandle) -> Result<Vec<String>, Problem> {
    let list = still_there(&read_recent(&recent_file(&app)?));
    let open = app.state::<Open>();
    for r in &list {
        open.remember(Path::new(r));
    }
    Ok(list)
}

/// Put `root` first among the recent Projects.
#[tauri::command]
pub fn recent_add(app: AppHandle, root: String) -> Result<(), Problem> {
    let file = recent_file(&app)?;
    let list = with_recent(&read_recent(&file), &root);
    if let Some(dir) = file.parent() {
        std::fs::create_dir_all(dir).map_err(|e| Problem::new("recent.write", e.to_string()))?;
    }
    let text = serde_json::to_string_pretty(&list).map_err(|e| Problem::new("recent.write", e.to_string()))?;
    std::fs::write(&file, text).map_err(|e| Problem::new("recent.write", e.to_string()))
}

// -- keeping awake ---------------------------------------------------------------

/// Keep the computer from sleeping (the display may still turn off) until
/// told otherwise or the app ends. Windows ties the request to a thread, so
/// one thread holds it while it lasts.
#[tauri::command]
pub fn keep_awake(files: State<'_, Files>, on: bool) {
    let mut awake = files.awake.lock().expect("awake lock");
    if let Some(stop) = awake.take() {
        let _ = stop.send(());
    }
    if on {
        let (tx, rx) = mpsc::channel::<()>();
        std::thread::spawn(move || {
            hold_awake(true);
            let _ = rx.recv();
            hold_awake(false);
        });
        *awake = Some(tx);
    }
}

#[cfg(windows)]
fn hold_awake(on: bool) {
    use windows_sys::Win32::System::Power::{SetThreadExecutionState, ES_CONTINUOUS, ES_SYSTEM_REQUIRED};
    // SAFETY: a plain Win32 call with flag arguments.
    unsafe {
        SetThreadExecutionState(if on { ES_CONTINUOUS | ES_SYSTEM_REQUIRED } else { ES_CONTINUOUS });
    }
}

#[cfg(not(windows))]
fn hold_awake(_on: bool) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn photos_are_plain_names_in_a_surveys_images() {
        let root = Path::new(r"C:\p");
        assert_eq!(
            photo_path(root, "s1", "DJI_0001.JPG").unwrap(),
            root.join("surveys").join("s1").join("images").join("DJI_0001.JPG")
        );
        for bad in ["", "..", "a/b.jpg", r"a\b.jpg", "c:x.jpg", "x.jpg.", "x\u{1}.jpg"] {
            assert!(photo_path(root, "s1", bad).is_err(), "{bad:?}");
        }
        for bad in ["", "..", "S1", "s1/x", r"s1\x", "s 1"] {
            assert!(photo_path(root, bad, "a.jpg").is_err(), "{bad:?}");
        }
    }

    #[test]
    fn the_recent_list_puts_the_newest_first_once_and_is_cut() {
        let list: Vec<String> = (0..RECENT_KEPT).map(|i| format!(r"C:\p{i}")).collect();
        let next = with_recent(&list, r"C:\P3");
        assert_eq!(next[0], r"C:\P3");
        assert_eq!(next.len(), RECENT_KEPT);
        assert_eq!(next.iter().filter(|r| r.to_lowercase() == r"c:\p3").count(), 1);
        let more = with_recent(&list, r"C:\new");
        assert_eq!(more.len(), RECENT_KEPT);
        assert!(!more.contains(&format!(r"C:\p{}", RECENT_KEPT - 1)));
    }

    #[test]
    fn the_recent_list_keeps_only_projects_that_exist() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a");
        std::fs::create_dir_all(&p).unwrap();
        std::fs::write(p.join(MANIFEST), "{}").unwrap();
        let gone = dir.path().join("gone").display().to_string();
        let kept = still_there(&[p.display().to_string(), gone]);
        assert_eq!(kept, vec![p.display().to_string()]);
        let file = dir.path().join("recent.json");
        assert!(read_recent(&file).is_empty());
        std::fs::write(&file, "not json").unwrap();
        assert!(read_recent(&file).is_empty());
    }
}
