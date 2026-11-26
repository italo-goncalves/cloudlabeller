// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

//! The Engine host: one `python -m cloudlabeller.engine` per open Project
//! (ADR-0002, contract in docs/engine.md).
//!
//! geokit's spawner closes a child's input, and the Engine reads JSON-RPC on
//! stdin, so the core starts it itself: the command line comes from
//! geokit's `programs::python` (native, conda or WSL alike), the process
//! from `std::process`. It starts no processes of its own (Jobs are the
//! core's, ADR-0004), so ending it ends nothing else.
//!
//! A reader thread routes each response to the call waiting on its `id`,
//! and each notification to `on_notify` (the page's Channel).

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use geokit_env::envs::{Environment, Machine};
use geokit_env::programs::{wsl_path, Program};
use geokit_env::Problem;
use serde_json::{json, Value};

/// How long the Engine may take to say `engine.ready`.
pub const READY_TIMEOUT: Duration = Duration::from_secs(120);

type Pending = Arc<Mutex<HashMap<u64, Sender<Value>>>>;
/// The page's ticket for each pending request it may cancel or follow.
type Tickets = Arc<Mutex<HashMap<u64, String>>>;
pub type Notify = Arc<dyn Fn(Value) + Send + Sync>;

pub struct Engine {
    child: Mutex<Child>,
    stdin: Mutex<ChildStdin>,
    pending: Pending,
    tickets: Tickets,
    next: AtomicU64,
    /// The Project's folder as the core and the page see it.
    pub root: PathBuf,
    /// The same folder as the Engine (and its Jobs) see it: `/mnt/c/…` in WSL.
    pub root_env: String,
    pub environment: Environment,
}

/// A Project folder as `env` sees it.
pub fn env_path(env: &Environment, native: &Path) -> Result<String, Problem> {
    let s = crate::environment::plain(native);
    match env.machine {
        Machine::Wsl => wsl_path(&s, &env.distro),
        _ => Ok(s),
    }
}

/// The command that starts the Engine in `env`.
pub fn engine_program(env: &Environment) -> Result<Program, Problem> {
    let mut p = crate::environment::python_program(env, &["-m", "cloudlabeller.engine"])?;
    // TensorFlow is never imported by the Engine, but quiet it if a module does.
    p.env.insert("TF_CPP_MIN_LOG_LEVEL".into(), "2".into());
    Ok(p)
}

fn command(p: &Program) -> Command {
    let mut c = Command::new(&p.program);
    c.args(&p.args).envs(&p.env);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        c.creation_flags(CREATE_NO_WINDOW);
    }
    c
}

/// A JSON-RPC error as a Problem: a Problem's own `data` when the Engine
/// sent one (-32000, -32602), else the code and message.
pub fn problem_of(error: &Value) -> Problem {
    let data = &error["data"];
    if let (Some(code), Some(english)) = (data["code"].as_str(), data["english"].as_str()) {
        let mut p = Problem::new(code, english);
        if let Some(params) = data["params"].as_object() {
            for (k, v) in params {
                p = p.with(k, v.to_string());
            }
        }
        return p;
    }
    Problem::new("engine.error", error["message"].as_str().unwrap_or("The Engine failed."))
        .with("rpc", error["code"].to_string())
}

impl Engine {
    /// Start the Engine in `env` for the Project at `root`; its stderr goes
    /// to `log`. Returns once it has said `engine.ready`.
    pub fn start(
        env: &Environment,
        root: &Path,
        log: &Path,
        on_notify: Notify,
    ) -> Result<Arc<Engine>, Problem> {
        let root_env = env_path(env, root)?;
        let program = engine_program(env)?;
        if let Some(dir) = log.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let stderr = std::fs::File::create(log).map_err(|e| {
            Problem::new("engine.log", "The Engine's log can't be written.")
                .with("path", log.display().to_string())
                .with("reason", e.to_string())
        })?;
        let mut child = command(&program)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(stderr))
            .spawn()
            .map_err(|e| {
                Problem::new("engine.start", "The Engine could not be started.")
                    .with("program", program.program.clone())
                    .with("reason", e.to_string())
            })?;
        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");
        let pending: Pending = Arc::new(Mutex::new(HashMap::new()));
        let tickets: Tickets = Arc::new(Mutex::new(HashMap::new()));
        let (ready_tx, ready_rx) = mpsc::channel::<Value>();

        {
            let pending = pending.clone();
            let tickets = tickets.clone();
            std::thread::Builder::new()
                .name("engine-stdout".into())
                .spawn(move || read(stdout, pending, tickets, on_notify, ready_tx))
                .expect("engine reader thread");
        }

        let engine = Arc::new(Engine {
            child: Mutex::new(child),
            stdin: Mutex::new(stdin),
            pending,
            tickets,
            next: AtomicU64::new(1),
            root: root.to_path_buf(),
            root_env,
            environment: env.clone(),
        });
        match ready_rx.recv_timeout(READY_TIMEOUT) {
            Ok(_) => Ok(engine),
            Err(_) => {
                engine.kill();
                Err(Problem::new("engine.not_ready", "The Engine didn't start in time.")
                    .with("log", log.display().to_string()))
            }
        }
    }

    /// Call `method` and wait for its answer (a long request answers when
    /// its work ends; its progress arrives as notifications meanwhile).
    pub fn call(&self, method: &str, params: Value) -> Result<Value, Problem> {
        let (id, rx) = self.send(method, params)?;
        self.wait(id, rx)
    }

    /// Send a request; its answer arrives on the receiver.
    pub fn send(&self, method: &str, params: Value) -> Result<(u64, Receiver<Value>), Problem> {
        let id = self.next.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = mpsc::channel();
        self.pending.lock().expect("pending lock").insert(id, tx);
        let line = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        let mut stdin = self.stdin.lock().expect("stdin lock");
        let written = writeln!(stdin, "{line}").and_then(|_| stdin.flush());
        if let Err(e) = written {
            self.pending.lock().expect("pending lock").remove(&id);
            return Err(gone().with("reason", e.to_string()));
        }
        Ok((id, rx))
    }

    /// Call `method` under the page's `ticket`: its progress notifications
    /// carry the ticket, and `cancel_ticket` cancels it.
    pub fn call_ticketed(&self, method: &str, params: Value, ticket: &str) -> Result<Value, Problem> {
        let (id, rx) = self.send(method, params)?;
        self.tickets.lock().expect("tickets lock").insert(id, ticket.to_string());
        let out = self.wait(id, rx);
        self.tickets.lock().expect("tickets lock").remove(&id);
        out
    }

    /// Cancel the pending request under `ticket`; false if there is none.
    pub fn cancel_ticket(&self, ticket: &str) -> Result<bool, Problem> {
        let id = self
            .tickets
            .lock()
            .expect("tickets lock")
            .iter()
            .find_map(|(id, t)| (t == ticket).then_some(*id));
        match id {
            Some(id) => Ok(self.cancel(id)?["cancelled"].as_bool().unwrap_or(false)),
            None => Ok(false),
        }
    }

    pub fn wait(&self, _id: u64, rx: Receiver<Value>) -> Result<Value, Problem> {
        let msg = rx.recv().map_err(|_| gone())?;
        if let Some(error) = msg.get("error") {
            return Err(problem_of(error));
        }
        Ok(msg.get("result").cloned().unwrap_or(Value::Null))
    }

    /// Cancel a pending long request (`engine.cancel`).
    pub fn cancel(&self, id: u64) -> Result<Value, Problem> {
        self.call("engine.cancel", json!({"id": id}))
    }

    /// A path the Engine named, as the core sees it: the Project's own
    /// prefix put back (WSL) and `/` turned into the system's separator.
    pub fn native(&self, path: &str) -> PathBuf {
        match path.strip_prefix(&self.root_env) {
            Some(rest) => {
                let mut p = self.root.clone();
                for part in rest.split(['/', '\\']).filter(|s| !s.is_empty()) {
                    p.push(part);
                }
                p
            }
            None => PathBuf::from(path),
        }
    }

    /// Save, stop and wait for the Engine; killed if it doesn't end in time.
    pub fn shutdown(&self, timeout: Duration) {
        let _ = self.send("engine.shutdown", json!({}));
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if let Ok(Some(_)) = self.child.lock().expect("child lock").try_wait() {
                return;
            }
            if std::time::Instant::now() >= deadline {
                self.kill();
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    pub fn kill(&self) {
        let mut child = self.child.lock().expect("child lock");
        let _ = child.kill();
        let _ = child.wait();
    }

    pub fn running(&self) -> bool {
        matches!(self.child.lock().expect("child lock").try_wait(), Ok(None))
    }
}

fn gone() -> Problem {
    Problem::new("engine.gone", "The Engine has stopped.")
}

fn read(
    stdout: std::process::ChildStdout,
    pending: Pending,
    tickets: Tickets,
    on_notify: Notify,
    ready: Sender<Value>,
) {
    for line in BufReader::new(stdout).lines() {
        let Ok(line) = line else { break };
        let Ok(msg) = serde_json::from_str::<Value>(&line) else {
            log::warn!("engine: not JSON-RPC: {line}");
            continue;
        };
        if msg.get("method").is_some() {
            let mut msg = msg;
            if msg["method"] == "engine.ready" {
                let _ = ready.send(msg["params"].clone());
            }
            if msg["method"] == "request.progress" {
                let id = msg["params"]["id"].as_u64();
                if let Some(t) = id.and_then(|id| tickets.lock().expect("tickets lock").get(&id).cloned()) {
                    msg["params"]["ticket"] = Value::String(t);
                }
            }
            on_notify(msg);
        } else if let Some(id) = msg.get("id").and_then(Value::as_u64) {
            if let Some(tx) = pending.lock().expect("pending lock").remove(&id) {
                let _ = tx.send(msg);
            }
        }
    }
    // The Engine is gone: every call still waiting fails (`engine.gone`).
    pending.lock().expect("pending lock").clear();
    on_notify(json!({"jsonrpc": "2.0", "method": "engine.exited", "params": {}}));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn problems_keep_the_engines_code() {
        let p = problem_of(&json!({
            "code": -32000, "message": "No Project is open.",
            "data": {"code": "project.none_open", "english": "No Project is open.", "params": {}}
        }));
        assert_eq!(p.code, "project.none_open");
        let p = problem_of(&json!({"code": -32601, "message": "no method 'x'"}));
        assert_eq!(p.code, "engine.error");
    }

    #[test]
    fn wsl_paths_for_the_engine() {
        let env = Environment {
            machine: Machine::Wsl,
            distro: "Ubuntu".into(),
            python: geokit_env::envs::Python::Default,
        };
        assert_eq!(env_path(&env, Path::new(r"C:\data\p.clproj")).unwrap(), "/mnt/c/data/p.clproj");
    }
}
