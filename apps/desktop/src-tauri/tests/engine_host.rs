// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

//! The Engine host against the real Engine, in the developer's Python
//! (`CLOUDLABELLER_DEV_PYTHON`, else the dev env CLAUDE.md names). Skipped
//! where there is none.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cloudlabeller_desktop_lib::engine::Engine;
use geokit_env::envs::{Environment, Machine, Python};
use serde_json::{json, Value};

pub fn dev_python() -> Option<PathBuf> {
    let p = std::env::var("CLOUDLABELLER_DEV_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(r"C:\Python_envs\cloudlabeller\python.exe"));
    p.is_file().then_some(p)
}

fn dev_environment() -> Option<Environment> {
    let path = dev_python()?.display().to_string();
    // The core starts this Python as the developer's (this checkout on PYTHONPATH).
    std::env::set_var("CLOUDLABELLER_DEV_PYTHON", &path);
    Some(Environment { machine: Machine::Windows, distro: String::new(), python: Python::Path { path } })
}

#[test]
fn the_engine_answers_and_notifies() {
    let Some(env) = dev_environment() else {
        eprintln!("skipped: no developer Python");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("p.clproj");
    let seen: Arc<Mutex<Vec<Value>>> = Arc::default();
    let notify = {
        let seen = seen.clone();
        Arc::new(move |v: Value| seen.lock().unwrap().push(v))
    };
    let engine = Engine::start(&env, &root, &dir.path().join("engine.log"), notify).unwrap();

    let hello = engine.call("engine.hello", json!({})).unwrap();
    assert_eq!(hello["protocol"], 1);

    let state = engine
        .call("project.create", json!({"root": engine.root_env, "classes": [{"name": "rock"}]}))
        .unwrap();
    assert_eq!(state["schema"][0]["name"], "rock");
    engine.call("schema.add", json!({"name": "soil"})).unwrap();

    let problem = engine.call("image.mask", json!({"name": "nope.jpg"})).unwrap_err();
    assert_eq!(problem.code, "image.not_found");

    let methods: Vec<String> = seen
        .lock()
        .unwrap()
        .iter()
        .map(|n| n["method"].as_str().unwrap_or("").to_string())
        .collect();
    assert!(methods.contains(&"engine.ready".to_string()));
    assert!(methods.contains(&"project.changed".to_string()));

    assert_eq!(engine.native(&engine.root_env), root);
    engine.shutdown(Duration::from_secs(20));
    assert!(!engine.running());
    assert!(root.join("project.json").is_file());
    // the Engine saved the schema on its way out
    let manifest = std::fs::read_to_string(root.join("project.json")).unwrap();
    assert!(manifest.contains("soil"));
}
