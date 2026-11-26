// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

//! The Job runner with geokit's scripted spawner and a stand-in Engine: no
//! Python, no WSL, no GPU.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cloudlabeller_desktop_lib::jobs::{
    progress_line, result_line, run_job, stage_script, staged_args, JobSaid, Jobs, Planner, Tell,
};
use geokit_env::envs::{Environment, Machine, Python};
use geokit_env::process::scripted::{Script, Scripted};
use geokit_env::process::Processes;
use geokit_env::Problem;
use serde_json::{json, Value};

struct FakeEngine {
    plan: Value,
    calls: Mutex<Vec<(String, Value)>>,
    env: Environment,
    root_native: PathBuf,
}

impl FakeEngine {
    fn new(plan: Value, machine: Machine, root_native: PathBuf) -> Self {
        let wsl = matches!(machine, Machine::Wsl);
        let distro = if wsl { "Ubuntu".into() } else { String::new() };
        let python = if wsl { "/env/bin/python" } else { r"C:\env\Scripts\python.exe" };
        FakeEngine {
            plan,
            calls: Mutex::default(),
            env: Environment {
                machine,
                distro,
                python: Python::Path { path: python.into() },
            },
            root_native,
        }
    }
    fn finished(&self) -> Value {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .find(|(m, _)| m == "jobs.finish")
            .map(|(_, p)| p.clone())
            .expect("jobs.finish was called")
    }
}

impl Planner for FakeEngine {
    fn call(&self, method: &str, params: Value) -> Result<Value, Problem> {
        self.calls.lock().unwrap().push((method.into(), params.clone()));
        match method {
            "jobs.plan" => Ok(self.plan.clone()),
            "jobs.finish" => Ok(json!({"plan": params["plan"], "ok": params["exit"] == 0})),
            _ => Err(Problem::new("fake", "unexpected")),
        }
    }
    fn native(&self, path: &str) -> PathBuf {
        match path.strip_prefix("/proj") {
            Some(rest) => self.root_native.join(rest.trim_start_matches('/')),
            None => PathBuf::from(path),
        }
    }
    fn root_env(&self) -> String {
        "/proj".into()
    }
    fn environment(&self) -> Environment {
        self.env.clone()
    }
}

fn plan(gpu: bool, stop_file: Option<&str>, stage: Value) -> Value {
    json!({
        "plan": 7, "kind": "sfm", "group": "reconstruction", "gpu": gpu,
        "module": "cloudlabeller.photogrammetry.run_cli",
        "args": ["/proj/images", "/proj/reconstruction", "--gpu"],
        "log": "/proj/reconstruction/sfm.log",
        "stop_file": stop_file, "stage": stage, "project": "/proj",
    })
}

fn collect() -> (Tell, Arc<Mutex<Vec<JobSaid>>>) {
    let said: Arc<Mutex<Vec<JobSaid>>> = Arc::default();
    let s = said.clone();
    (Arc::new(move |m| s.lock().unwrap().push(m)), said)
}

fn kinds(said: &[JobSaid]) -> Vec<String> {
    said.iter()
        .map(|s| serde_json::to_value(s).unwrap()["kind"].as_str().unwrap().to_string())
        .collect()
}

#[test]
fn a_job_runs_and_its_outcome_goes_to_the_engine() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("reconstruction")).unwrap();
    let engine = FakeEngine::new(plan(false, None, Value::Null), Machine::Windows, dir.path().into());
    let spawner = Arc::new(Scripted::always(
        Script::ok()
            .says("PROGRESS 0.500 matching")
            .says("a log line")
            .complains("a warning")
            .says("RESULT 120 4"),
    ));
    let procs = Processes::new(spawner.clone());
    let (tell, said) = collect();
    run_job(&engine, &procs, &Jobs::default(), "sfm", json!({}), tell);

    let said = said.lock().unwrap();
    let k = kinds(&said);
    assert_eq!(k.first().map(String::as_str), Some("planned"));
    assert_eq!(k.last().map(String::as_str), Some("finished"));
    assert!(k.contains(&"started".into()) && k.contains(&"progress".into()));
    let done = engine.finished();
    assert_eq!(done["exit"], 0);
    assert_eq!(done["results"], json!({"RESULT": "120 4"}));
    assert_eq!(done["cancelled"], false);
    let log = std::fs::read_to_string(dir.path().join("reconstruction/sfm.log")).unwrap();
    assert!(log.contains("a log line") && log.contains("a warning") && log.starts_with("$ "));
    let job = &spawner.spawned()[0];
    assert!(job.args.iter().any(|a| a == "cloudlabeller.photogrammetry.run_cli"));
}

#[test]
fn a_failed_job_reports_its_exit() {
    let engine = FakeEngine::new(plan(false, None, Value::Null), Machine::Windows, PathBuf::from("."));
    let procs = Processes::new(Arc::new(Scripted::always(Script::ok().complains("boom").exits(3))));
    let (tell, _) = collect();
    run_job(&engine, &procs, &Jobs::default(), "sfm", json!({}), tell);
    assert_eq!(engine.finished()["exit"], 3);
}

#[test]
fn a_gpu_job_waits_for_the_lease_and_can_give_up() {
    let jobs = Arc::new(Jobs::default());
    let held = jobs.lease.try_hold("training").expect("the Lease is free");
    let engine = Arc::new(FakeEngine::new(plan(true, None, Value::Null), Machine::Windows, ".".into()));
    let procs = Arc::new(Processes::new(Arc::new(Scripted::always(Script::ok()))));
    let (tell, said) = collect();
    let worker = {
        let (jobs, engine, procs) = (jobs.clone(), engine.clone(), procs.clone());
        std::thread::spawn(move || run_job(engine.as_ref(), &procs, &jobs, "mvs", json!({}), tell))
    };
    // it waits, first in line
    for _ in 0..100 {
        if kinds(&said.lock().unwrap()).contains(&"waiting".into()) {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(matches!(said.lock().unwrap().get(1), Some(JobSaid::Waiting { place: 1 })));
    assert!(jobs.cancel(&procs, 7));
    worker.join().unwrap();
    drop(held);
    let done = engine.finished();
    assert_eq!((done["exit"].clone(), done["cancelled"].clone()), (Value::Null, json!(true)));
    assert!(!kinds(&said.lock().unwrap()).contains(&"started".into()));
}

#[test]
fn the_lease_comes_when_the_holder_ends() {
    let jobs = Arc::new(Jobs::default());
    let held = jobs.lease.try_hold("training").unwrap();
    let engine = Arc::new(FakeEngine::new(plan(true, None, Value::Null), Machine::Windows, ".".into()));
    let procs = Arc::new(Processes::new(Arc::new(Scripted::always(Script::ok()))));
    let (tell, said) = collect();
    let worker = {
        let (jobs, engine, procs) = (jobs.clone(), engine.clone(), procs.clone());
        std::thread::spawn(move || run_job(engine.as_ref(), &procs, &jobs, "mvs", json!({}), tell))
    };
    std::thread::sleep(Duration::from_millis(200));
    drop(held);
    worker.join().unwrap();
    assert_eq!(engine.finished()["exit"], 0);
    assert!(kinds(&said.lock().unwrap()).contains(&"started".into()));
    assert_eq!(jobs.lease.state().holder, None, "the Lease is given back");
}

#[test]
fn cancel_kills_a_job_without_a_stop_file() {
    let jobs = Arc::new(Jobs::default());
    let engine = Arc::new(FakeEngine::new(plan(false, None, Value::Null), Machine::Windows, ".".into()));
    let procs = Arc::new(Processes::new(Arc::new(Scripted::always(Script::ok().until_killed()))));
    let (tell, said) = collect();
    let worker = {
        let (jobs, engine, procs) = (jobs.clone(), engine.clone(), procs.clone());
        std::thread::spawn(move || run_job(engine.as_ref(), &procs, &jobs, "sfm", json!({}), tell))
    };
    for _ in 0..100 {
        if kinds(&said.lock().unwrap()).contains(&"started".into()) {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(jobs.cancel(&procs, 7));
    worker.join().unwrap();
    let done = engine.finished();
    assert_eq!(done["cancelled"], true);
    assert_ne!(done["exit"], 0);
}

#[test]
fn cancel_asks_a_job_with_a_stop_file_to_wind_down() {
    let dir = tempfile::tempdir().unwrap();
    let jobs = Arc::new(Jobs::default());
    let stop = dir.path().join("train.stop");
    let engine = Arc::new(FakeEngine::new(
        plan(false, Some("/proj/train.stop"), Value::Null),
        Machine::Windows,
        dir.path().into(),
    ));
    let procs = Arc::new(Processes::new(Arc::new(Scripted::always(
        Script::ok().runs_for(Duration::from_millis(600)).says("RESULT {\"epochs_run\": 2}"),
    ))));
    let (tell, said) = collect();
    let worker = {
        let (jobs, engine, procs) = (jobs.clone(), engine.clone(), procs.clone());
        std::thread::spawn(move || run_job(engine.as_ref(), &procs, &jobs, "train", json!({}), tell))
    };
    for _ in 0..100 {
        if kinds(&said.lock().unwrap()).contains(&"started".into()) {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(jobs.cancel(&procs, 7));
    assert!(stop.exists(), "the stop file was made, not a kill");
    worker.join().unwrap();
    let done = engine.finished();
    assert_eq!((done["exit"].clone(), done["cancelled"].clone()), (json!(0), json!(true)));
}

#[test]
fn a_wsl_colmap_job_runs_in_a_workspace() {
    let stage = json!({"in": ["images", "reconstruction"], "out": ["reconstruction"]});
    let engine = FakeEngine::new(plan(false, None, stage), Machine::Wsl, ".".into());
    let spawner = Arc::new(Scripted::new(|p| {
        let all = p.args.join(" ");
        if all.contains("$HOME") {
            Script::ok().says("/home/u")
        } else {
            Script::ok()
        }
    }));
    let procs = Processes::new(spawner.clone());
    let (tell, said) = collect();
    run_job(&engine, &procs, &Jobs::default(), "sfm", json!({}), tell);

    assert_eq!(engine.finished()["exit"], 0);
    let spawned: Vec<String> = spawner.spawned().iter().map(|p| p.args.join(" ")).collect();
    assert_eq!(spawned.len(), 4, "{spawned:#?}");
    let ws = "/home/u/.local/share/cloudlabeller/workspaces/7";
    assert!(spawned[1].contains("cp -a") && spawned[1].contains(ws), "{}", spawned[1]);
    assert!(spawned[2].contains(&format!("{ws}/images")), "{}", spawned[2]);
    assert!(!spawned[2].contains("/proj/images"), "{}", spawned[2]);
    assert!(spawned[3].contains("rm -rf") && spawned[3].contains("/proj/reconstruction"));
    let k = kinds(&said.lock().unwrap());
    assert_eq!(k.iter().filter(|s| *s == "staging").count(), 2);
}

#[test]
fn line_protocol_and_staging_helpers() {
    assert_eq!(progress_line("PROGRESS 0.250 Matching"), Some((0.25, "Matching".into())));
    assert_eq!(progress_line("PROGRESS x"), None);
    assert_eq!(result_line("RESULT {\"a\": 1}"), Some(("RESULT".into(), "{\"a\": 1}".into())));
    assert_eq!(result_line("RESULT_MESH 3 1"), Some(("RESULT_MESH".into(), "3 1".into())));
    assert_eq!(result_line("nothing"), None);
    let args: Vec<String> = ["/proj/images", "/projector", "--x", "/proj"].iter().map(|s| s.to_string()).collect();
    assert_eq!(staged_args(&args, "/proj", "/ws"), vec!["/ws/images", "/projector", "--x", "/ws"]);
    assert!(stage_script("/proj", "/ws", &["../etc".into()], &[], true).is_err());
    assert!(stage_script("/proj", "/ws", &["images".into()], &["products".into()], true).is_ok());
}
