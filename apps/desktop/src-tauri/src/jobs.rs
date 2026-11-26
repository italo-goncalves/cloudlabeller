// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

//! Jobs: the Engine plans them, the core runs them (ADR-0004).
//!
//! `job_run`:
//! 1. `jobs.plan` asks the Engine whether the Job can run and how;
//! 2. a GPU Job waits for the one GPU Lease (ADR-0003), telling the page
//!    its place in the queue ("waiting for GPU");
//! 3. a Job with a `stage` hint, in a WSL Environment, is staged: its input
//!    folders are copied into a Workspace on WSL's own disk and its command
//!    line pointed there;
//! 4. it runs with geokit's `Processes` (its whole tree ended on cancel and
//!    on exit), the Lease held until it is gone; its `PROGRESS` lines go to
//!    the page, every line to the plan's log, its `RESULT…` lines are kept;
//! 5. a staged Job that ended well has its output folders copied back, and
//!    its Workspace is removed;
//! 6. `jobs.finish` hands the outcome to the Engine, which updates the
//!    Project; its answer is the Job's last message to the page.
//!
//! `job_cancel` creates the plan's stop file when it has one (training
//! winds down and still saves), kills the Job otherwise, and gives up the
//! wait for the Lease.

use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use geokit_env::envs::{Environment, Machine};
use geokit_env::lease::GpuLease;
use geokit_env::process::{How, Processes, Said};
use geokit_env::programs::{sh, wsl, Program};
use geokit_env::Problem;
use serde::Serialize;
use serde_json::{json, Value};
use tauri::ipc::Channel;
use tauri::{AppHandle, Manager, State};

use crate::engine::Engine;
use crate::environment::python_program;
use crate::project::Open;

/// How long copying a Workspace in or out may take.
const STAGE_TIMEOUT: Duration = Duration::from_secs(6 * 3600);

/// What a Job tells the page, in order.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum JobSaid {
    /// The Engine's plan (`jobs.plan`'s answer).
    Planned { plan: Value },
    /// Waiting for the GPU Lease: `place` 1 is next.
    Waiting { place: usize },
    /// Copying the Workspace `in` (to WSL's disk) or `out` (back).
    Staging { direction: &'static str },
    Started,
    Progress { fraction: f64, message: String },
    Line { line: String, err: bool },
    /// `jobs.finish`'s answer: `{ok, result | error, stopped}`.
    Finished { outcome: Value },
    /// The Job couldn't be planned or run at all.
    Failed { problem: Problem },
}

/// The Engine as the Job runner uses it (the Engine itself in the app; a
/// stand-in in tests).
pub trait Planner: Send + Sync {
    fn call(&self, method: &str, params: Value) -> Result<Value, Problem>;
    /// A path the Engine named, as the core sees it.
    fn native(&self, path: &str) -> PathBuf;
    /// The Project folder as the Environment sees it.
    fn root_env(&self) -> String;
    fn environment(&self) -> Environment;
}

impl Planner for Engine {
    fn call(&self, method: &str, params: Value) -> Result<Value, Problem> {
        Engine::call(self, method, params)
    }
    fn native(&self, path: &str) -> PathBuf {
        Engine::native(self, path)
    }
    fn root_env(&self) -> String {
        self.root_env.clone()
    }
    fn environment(&self) -> Environment {
        self.environment.clone()
    }
}

/// Where a Job's messages go.
pub type Tell = Arc<dyn Fn(JobSaid) + Send + Sync>;

/// A Job the core is running (or waiting to run), by plan number.
#[derive(Default)]
struct Running {
    process: Option<u32>,
    stop_file: Option<PathBuf>,
    cancelled: Arc<AtomicBool>,
}

#[derive(Default)]
pub struct Jobs {
    pub lease: GpuLease,
    running: Mutex<HashMap<u64, Running>>,
}

impl Jobs {
    /// Cancel plan `plan`: see the module docs. False if it isn't running.
    pub fn cancel(&self, procs: &Processes, plan: u64) -> bool {
        let running = self.running.lock().expect("jobs lock");
        let Some(job) = running.get(&plan) else { return false };
        job.cancelled.store(true, Ordering::SeqCst);
        match (&job.stop_file, job.process) {
            (Some(stop), Some(_)) => {
                let _ = std::fs::File::create(stop);
            }
            (None, Some(id)) => procs.kill(id),
            _ => {} // still waiting for the Lease: it gives up
        }
        true
    }
}

/// The `RESULT…` line kept under its first word, the rest of the line after.
pub fn result_line(line: &str) -> Option<(String, String)> {
    if !line.starts_with("RESULT") {
        return None;
    }
    let (key, rest) = line.split_once(' ').unwrap_or((line, ""));
    Some((key.to_string(), rest.trim().to_string()))
}

/// `PROGRESS <fraction> <message>`.
pub fn progress_line(line: &str) -> Option<(f64, String)> {
    let rest = line.strip_prefix("PROGRESS ")?;
    let (frac, msg) = rest.split_once(' ').unwrap_or((rest, ""));
    Some((frac.parse().ok()?, msg.to_string()))
}

/// The plan's command line with the Project folder replaced by the Workspace.
pub fn staged_args(args: &[String], root_env: &str, workspace: &str) -> Vec<String> {
    args.iter()
        .map(|a| match a.strip_prefix(root_env) {
            Some(rest) if rest.is_empty() || rest.starts_with('/') => format!("{workspace}{rest}"),
            _ => a.clone(),
        })
        .collect()
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array()
        .map(|a| a.iter().filter_map(|s| s.as_str().map(str::to_string)).collect())
        .unwrap_or_default()
}

/// A folder name from a `stage` hint: one plain relative segment path.
fn stage_dir(d: &str) -> Result<&str, Problem> {
    let ok = !d.is_empty()
        && !d.starts_with('/')
        && d.split('/').all(|p| !p.is_empty() && p != "." && p != "..")
        && d.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '_' | '-' | '.'));
    if ok {
        Ok(d)
    } else {
        Err(Problem::new("job.bad_stage", "The plan's stage folder isn't a plain folder.")
            .with("folder", d.to_string()))
    }
}

/// The bash script that copies the Workspace in (or out).
pub fn stage_script(root_env: &str, workspace: &str, dirs: &[String], outs: &[String], into: bool) -> Result<String, Problem> {
    let mut s = String::from("set -e\n");
    if into {
        s.push_str(&format!("rm -rf {ws}\nmkdir -p {ws}\n", ws = sh(workspace)));
        for d in dirs {
            let d = stage_dir(d)?;
            let src = sh(&format!("{root_env}/{d}"));
            let dst = sh(&format!("{workspace}/{d}"));
            s.push_str(&format!("if [ -e {src} ]; then mkdir -p \"$(dirname {dst})\"; cp -a {src} {dst}; fi\n"));
        }
        for d in outs {
            let d = stage_dir(d)?;
            s.push_str(&format!("mkdir -p {}\n", sh(&format!("{workspace}/{d}"))));
        }
    } else {
        for d in outs {
            let d = stage_dir(d)?;
            let src = sh(&format!("{workspace}/{d}"));
            let dst = sh(&format!("{root_env}/{d}"));
            s.push_str(&format!("mkdir -p {dst}\ncp -a {src}/. {dst}/\n"));
        }
        s.push_str(&format!("rm -rf {}\n", sh(workspace)));
    }
    Ok(s)
}

fn run_script(procs: &Processes, distro: &str, script: &str) -> Result<(), Problem> {
    let c = procs.output(&wsl(distro, script), STAGE_TIMEOUT)?;
    if c.success() {
        Ok(())
    } else {
        Err(Problem::new("job.stage_failed", "Copying the Workspace failed.")
            .with("stderr", c.stderr.join("\n")))
    }
}

/// `$HOME` in WSL, where the Workspaces live.
fn wsl_home(procs: &Processes, distro: &str) -> Result<String, Problem> {
    let c = procs.output(&wsl(distro, "printf '%s\\n' \"$HOME\""), Duration::from_secs(60))?;
    c.lines
        .iter()
        .rev()
        .find(|l| l.starts_with('/'))
        .cloned()
        .ok_or_else(|| Problem::new("job.no_home", "WSL didn't say where its home is."))
}

/// Run one Job from plan to finish (see the module docs). The last message
/// to `tell` is `finished` or `failed`.
pub fn run_job(
    planner: &dyn Planner,
    procs: &Processes,
    jobs: &Jobs,
    kind: &str,
    options: Value,
    tell: Tell,
) {
    if let Err(problem) = run_job_inner(planner, procs, jobs, kind, options, &tell) {
        tell(JobSaid::Failed { problem });
    }
}

fn run_job_inner(
    planner: &dyn Planner,
    procs: &Processes,
    jobs: &Jobs,
    kind: &str,
    options: Value,
    tell: &Tell,
) -> Result<(), Problem> {
    let plan = planner.call("jobs.plan", json!({"kind": kind, "options": options}))?;
    tell(JobSaid::Planned { plan: plan.clone() });
    let token = plan["plan"].as_u64().ok_or_else(|| Problem::new("job.bad_plan", "The plan has no number."))?;
    let finish = |exit: Option<i32>, results: &BTreeMap<String, String>, cancelled: bool| {
        planner.call(
            "jobs.finish",
            json!({"plan": token, "exit": exit, "results": results, "cancelled": cancelled}),
        )
    };

    let cancelled = Arc::new(AtomicBool::new(false));
    let stop_file = plan["stop_file"].as_str().map(|p| planner.native(p));
    jobs.running.lock().expect("jobs lock").insert(
        token,
        Running { process: None, stop_file: stop_file.clone(), cancelled: cancelled.clone() },
    );
    let forget = || {
        jobs.running.lock().expect("jobs lock").remove(&token);
    };

    // 2. the GPU Lease
    let held = if plan["gpu"].as_bool().unwrap_or(false) {
        let name = format!("{kind} #{token}");
        match jobs.lease.hold(&name, || cancelled.load(Ordering::SeqCst), |place| {
            tell(JobSaid::Waiting { place })
        }) {
            Some(h) => Some(h),
            None => {
                forget();
                let outcome = finish(None, &BTreeMap::new(), true)?;
                tell(JobSaid::Finished { outcome });
                return Ok(());
            }
        }
    } else {
        None
    };

    // 3. the Workspace
    let env = planner.environment();
    let root_env = planner.root_env();
    let mut args = strings(&plan["args"]);
    let stage = &plan["stage"];
    let staged = if env.machine == Machine::Wsl && stage.is_object() {
        let (ins, outs) = (strings(&stage["in"]), strings(&stage["out"]));
        let home = wsl_home(procs, &env.distro)?;
        let workspace = format!("{home}/.local/share/{}/workspaces/{token}", crate::environment::APP);
        tell(JobSaid::Staging { direction: "in" });
        if let Err(p) = run_script(procs, &env.distro, &stage_script(&root_env, &workspace, &ins, &outs, true)?) {
            forget();
            drop(held);
            let outcome = finish(None, &BTreeMap::new(), false)?;
            tell(JobSaid::Failed { problem: p });
            tell(JobSaid::Finished { outcome });
            return Ok(());
        }
        args = staged_args(&args, &root_env, &workspace);
        Some((workspace, outs))
    } else {
        None
    };

    // 4. the process
    let module = plan["module"].as_str().ok_or_else(|| Problem::new("job.bad_plan", "The plan names no module."))?;
    let mut all: Vec<&str> = vec!["-u", "-m", module];
    all.extend(args.iter().map(String::as_str));
    let mut program: Program = python_program(&env, &all)?;
    program.env.insert("TF_CPP_MIN_LOG_LEVEL".into(), "2".into());

    let log_path = plan["log"].as_str().map(|p| planner.native(p));
    let log = Arc::new(Mutex::new(log_path.as_ref().and_then(|p| std::fs::File::create(p).ok())));
    let results: Arc<Mutex<BTreeMap<String, String>>> = Arc::default();
    let sink = {
        let log = log.clone();
        let results = results.clone();
        let tell = tell.clone();
        Arc::new(move |said: Said| {
            match said {
                Said::Out { line } | Said::Err { line } if line.is_empty() => {}
                Said::Out { line } => {
                    if let Some((fraction, message)) = progress_line(&line) {
                        tell(JobSaid::Progress { fraction, message });
                    } else if let Some((k, v)) = result_line(&line) {
                        results.lock().expect("results lock").insert(k, v);
                    }
                    if let Some(f) = log.lock().expect("log lock").as_mut() {
                        let _ = writeln!(f, "{line}");
                    }
                    tell(JobSaid::Line { line, err: false });
                }
                Said::Err { line } => {
                    if let Some(f) = log.lock().expect("log lock").as_mut() {
                        let _ = writeln!(f, "{line}");
                    }
                    tell(JobSaid::Line { line, err: true });
                }
                Said::Exit { .. } => {}
            }
            true
        }) as geokit_env::process::Sink
    };
    if let Some(f) = log.lock().expect("log lock").as_mut() {
        let _ = writeln!(f, "$ {} {}", program.program, program.args.join(" "));
    }
    if cancelled.load(Ordering::SeqCst) && stop_file.is_none() {
        forget();
        drop(held);
        let outcome = finish(None, &BTreeMap::new(), true)?;
        tell(JobSaid::Finished { outcome });
        return Ok(());
    }
    let (id, ended) = match procs.start_holding(&program, sink, held) {
        Ok(x) => x,
        Err(p) => {
            forget();
            let outcome = finish(None, &BTreeMap::new(), false)?;
            tell(JobSaid::Failed { problem: p });
            tell(JobSaid::Finished { outcome });
            return Ok(());
        }
    };
    if let Some(job) = jobs.running.lock().expect("jobs lock").get_mut(&token) {
        job.process = Some(id);
    }
    tell(JobSaid::Started);
    // a cancel that came between the Lease and the start
    if cancelled.load(Ordering::SeqCst) {
        jobs.cancel(procs, token);
    }
    let exit = ended.recv().unwrap_or(geokit_env::process::Exit { code: None, how: How::Lost });
    forget();
    drop(log);
    let was_cancelled = cancelled.load(Ordering::SeqCst);

    // 5. back from the Workspace
    if let Some((workspace, outs)) = staged {
        if exit.success() {
            tell(JobSaid::Staging { direction: "out" });
            if let Err(p) = run_script(procs, &env.distro, &stage_script(&root_env, &workspace, &[], &outs, false)?) {
                let outcome = finish(None, &BTreeMap::new(), was_cancelled)?;
                tell(JobSaid::Failed { problem: p });
                tell(JobSaid::Finished { outcome });
                return Ok(());
            }
        } else {
            let _ = run_script(procs, &env.distro, &format!("rm -rf {}", sh(&workspace)));
        }
    }

    // 6. the Engine applies it
    let code = match exit.how {
        How::Ended => exit.code,
        How::Killed | How::Lost => None,
    };
    let code = if exit.how == How::Killed { Some(code.unwrap_or(-1)) } else { code };
    let results = results.lock().expect("results lock").clone();
    let outcome = finish(code, &results, was_cancelled)?;
    tell(JobSaid::Finished { outcome });
    Ok(())
}

pub fn manage(app: &AppHandle) {
    app.manage(Jobs::default());
}

/// Run a Job of `kind` with `options` in the open Project; everything it
/// says arrives on `on`, the last message `finished` or `failed`.
#[tauri::command]
pub async fn job_run(app: AppHandle, kind: String, options: Value, on: Channel<JobSaid>) -> Result<(), Problem> {
    let engine = app.state::<Open>().engine()?;
    let worker = app.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let tell: Tell = Arc::new(move |s: JobSaid| {
            let _ = on.send(s);
        });
        run_job(
            engine.as_ref(),
            &worker.state::<Processes>(),
            &worker.state::<Jobs>(),
            &kind,
            options,
            tell,
        );
    })
    .await
    .map_err(|e| Problem::new("job.crashed", e.to_string()))
}

/// Cancel a running Job by its plan number; false if it isn't running.
#[tauri::command]
pub fn job_cancel(jobs: State<'_, Jobs>, procs: State<'_, Processes>, plan: u64) -> bool {
    jobs.cancel(&procs, plan)
}

/// Who holds the GPU Lease and who waits for it.
#[tauri::command]
pub fn gpu_lease(jobs: State<'_, Jobs>) -> Value {
    serde_json::to_value(jobs.lease.state()).unwrap_or(Value::Null)
}
