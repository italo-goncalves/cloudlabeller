// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

//! The Environment: the Python the Engine and the Jobs run in (ADR-0003).
//!
//! On start the page asks for it with `environment_ensure`. The app's own
//! Environment is made from its Recipe with geokit:
//!
//! - none yet: probe where it should go (WSL when a distribution reaches the
//!   GPU with `nvidia-smi`, else native Windows on the CPU) and install it;
//! - one from another build of the app (its recorded version is not this
//!   build's): install over it, in place;
//! - this build's: run its check, and repair it from nothing if it fails.
//!
//! The app never installs WSL itself (that changes system settings and may
//! need a reboot); the page links to Microsoft's instructions instead.
//!
//! In a debug build, `CLOUDLABELLER_DEV_PYTHON` names a Python to use as the
//! Environment as it is (the developer's own), with nothing installed.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use geokit_env::envs::{Environment, Environments, Kept, Machine, Python};
use geokit_env::install::{self, Health, Progress, Request};
use geokit_env::process::Processes;
use geokit_env::programs::{gpu_found, wsl_gpu_probe, Os, Own, Recipe, Shipped, DEFAULT_PYTHON};
use geokit_env::{detect, Problem};
use serde::Serialize;
use tauri::ipc::Channel;
use tauri::{AppHandle, Manager, State};

/// The Recipe's name: its folder in the data folder.
pub const RECIPE_NAME: &str = "engine";
/// CloudLabeller's folder in WSL, under `~/.local/share`.
pub const APP: &str = "cloudlabeller";

const CHECK_TIMEOUT: Duration = Duration::from_secs(180);
const PROBE_TIMEOUT: Duration = Duration::from_secs(60);

/// Where the Environment goes, and why, for the page to say.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Why {
    /// A WSL distribution reaches the GPU: the GPU Jobs run there.
    Gpu,
    /// WSL has distributions, but none reaches a GPU.
    NoGpuInWsl,
    /// WSL is missing, or has no distribution.
    NoWsl,
    /// Not Windows: the system's own.
    NotWindows,
}

#[derive(Clone, Debug, Serialize)]
pub struct Choice {
    /// `{machine: "windows"}` or `{machine: "wsl", distro}`, as geokit reads it.
    #[serde(serialize_with = "own_json")]
    pub own: Own,
    pub why: Why,
    /// The GPU found, as `nvidia-smi -L` names it.
    pub gpu: Option<String>,
    pub distros: Vec<String>,
}

/// geokit 0.2's `Own` is only Deserialize: written here in its own shape.
fn own_json<S: serde::Serializer>(own: &Own, s: S) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeMap;
    let mut m = s.serialize_map(None)?;
    match own {
        Own::Windows => m.serialize_entry("machine", "windows")?,
        Own::Wsl { distro } => {
            m.serialize_entry("machine", "wsl")?;
            m.serialize_entry("distro", distro)?;
        }
        Own::Posix => m.serialize_entry("machine", "posix")?,
    }
    m.end()
}

/// What `environment_ensure` tells the page as it goes.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Said {
    /// Where a new Environment goes, before it is installed.
    Chose { choice: Choice },
    /// What happens next: `check`, `install`, `upgrade` or `repair`.
    Doing { what: String },
    Step { i: usize, n: usize, what: String },
    Out { line: String },
    Err { line: String },
    /// The Environment is ready.
    Ready { kept: Kept },
    Failed { problem: Problem },
}

/// Whether a setup runs, and how to cancel it.
#[derive(Default)]
pub struct Setup {
    busy: AtomicBool,
    cancel: Arc<AtomicBool>,
}

/// What the app ships, from its resource folder.
pub struct Resources {
    pub uv: PathBuf,
    pub uv_linux: PathBuf,
    pub wheel: PathBuf,
    pub lockfile: Option<PathBuf>,
    /// The wheel's own version and a digest of it, recorded as `installed`.
    pub build: String,
}

/// Windows' `\\?\` prefix, which uv and WSL don't read.
pub fn plain(p: &Path) -> String {
    let s = p.to_string_lossy();
    s.strip_prefix(r"\\?\").unwrap_or(&s).to_string()
}

impl Resources {
    pub fn find(dir: &Path) -> Result<Self, Problem> {
        let engine = dir.join("engine");
        let wheel = std::fs::read_dir(&engine)
            .ok()
            .and_then(|it| {
                it.filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .find(|p| p.extension().is_some_and(|x| x == "whl"))
            })
            .ok_or_else(|| {
                Problem::new("environment.no_wheel", "The app's wheel is missing from its install.")
                    .with("folder", plain(&engine))
            })?;
        let lock = engine.join("requirements.lock");
        let bytes = std::fs::read(&wheel).map_err(|e| {
            Problem::new("environment.no_wheel", "The app's wheel can't be read.")
                .with("reason", e.to_string())
        })?;
        let version = wheel_version(&wheel).unwrap_or_else(|| "0".into());
        let lock_bytes = std::fs::read(&lock).unwrap_or_default();
        Ok(Resources {
            uv: dir.join("uv").join("windows").join("uv.exe"),
            uv_linux: dir.join("uv").join("linux").join("uv"),
            wheel,
            lockfile: lock.is_file().then_some(lock),
            build: format!("{version}+{:016x}", fnv1a(&[&bytes, &lock_bytes])),
        })
    }

    /// The Recipe: the lockfile first, then the wheel without its
    /// dependencies. With no lockfile (before one is committed), the wheel is
    /// installed with its dependencies, unpinned.
    pub fn recipe(&self) -> Recipe {
        let wheel = plain(&self.wheel);
        let (packages, wheels) = match &self.lockfile {
            Some(_) => (vec![], vec![wheel]),
            None => (vec![wheel], vec![]),
        };
        Recipe {
            name: RECIPE_NAME.into(),
            app: APP.into(),
            python: DEFAULT_PYTHON.into(),
            lockfile: self.lockfile.as_deref().map(plain),
            packages,
            wheels,
            check: check_code(&self.build),
        }
    }

    pub fn shipped(&self, data: &Path) -> Shipped {
        Shipped {
            data: plain(data),
            uv: plain(&self.uv),
            uv_linux: Some(plain(&self.uv_linux)),
        }
    }
}

/// The Recipe's check: the Engine imports, and its last line is this
/// build's stamp, so an Environment from another build reads as Stale.
pub fn check_code(build: &str) -> String {
    format!(
        "import cloudlabeller, cloudlabeller.engine.methods, numpy, scipy, pycolmap\n\
         print('cloudlabeller', cloudlabeller.__version__, 'pycolmap', pycolmap.__version__)\n\
         print({build:?})"
    )
}

/// `cloudlabeller-0.2.1-py3-none-any.whl` → `0.2.1`.
pub fn wheel_version(wheel: &Path) -> Option<String> {
    let name = wheel.file_name()?.to_str()?;
    name.split('-').nth(1).map(str::to_string)
}

/// A stable 64-bit digest (FNV-1a) of some byte strings.
pub fn fnv1a(parts: &[&[u8]]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for part in parts {
        for b in *part {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        h ^= 0xff;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Where a new Environment goes, from what WSL says.
pub fn choose(os: Os, distros: &[String], gpus: &[(String, Option<String>)]) -> Choice {
    if os != Os::Windows {
        return Choice { own: Own::Posix, why: Why::NotWindows, gpu: None, distros: vec![] };
    }
    if distros.is_empty() {
        return Choice { own: Own::Windows, why: Why::NoWsl, gpu: None, distros: vec![] };
    }
    match gpus.iter().find_map(|(d, g)| g.clone().map(|g| (d.clone(), g))) {
        Some((distro, gpu)) => Choice {
            own: Own::Wsl { distro },
            why: Why::Gpu,
            gpu: Some(gpu),
            distros: distros.to_vec(),
        },
        None => Choice {
            own: Own::Windows,
            why: Why::NoGpuInWsl,
            gpu: None,
            distros: distros.to_vec(),
        },
    }
}

fn probe(procs: &Processes) -> Choice {
    let os = Os::this();
    if os != Os::Windows {
        return choose(os, &[], &[]);
    }
    let distros = detect::wsl_distros(procs).unwrap_or_default();
    let gpus: Vec<(String, Option<String>)> = distros
        .iter()
        .map(|d| {
            let found = wsl_gpu_probe(d)
                .ok()
                .and_then(|p| procs.output(&p, PROBE_TIMEOUT).ok())
                .and_then(|c| gpu_found(&c.lines));
            (d.clone(), found)
        })
        .collect();
    choose(os, &distros, &gpus)
}

/// This checkout's root, where a debug build's developer Python finds the
/// `cloudlabeller` package (the dev env doesn't install it).
fn checkout() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("..")
}

/// How the core starts Python `args` in `env`: geokit's isolated command
/// line, except for a debug build's developer Python, which is started
/// without `-I` and with this checkout on `PYTHONPATH`.
pub fn python_program(env: &Environment, args: &[&str]) -> Result<geokit_env::programs::Program, Problem> {
    if let (Some(dev), Python::Path { path }) = (dev_environment(), &env.python) {
        if let Python::Path { path: dev_path } = &dev.environment.python {
            if dev_path == path {
                let mut all = vec!["-X", "utf8"];
                all.extend_from_slice(args);
                let mut p = geokit_env::programs::program(path, &all);
                p.env.insert("PYTHONPATH".into(), plain(&checkout()));
                p.env.insert("PYTHONIOENCODING".into(), "utf-8".into());
                return Ok(p);
            }
        }
    }
    geokit_env::programs::python(env, args)
}

/// The developer's own Python, in a debug build only.
pub fn dev_environment() -> Option<Kept> {
    if !cfg!(debug_assertions) {
        return None;
    }
    let path = std::env::var("CLOUDLABELLER_DEV_PYTHON").ok()?;
    Some(Kept {
        environment: Environment {
            machine: Machine::Windows,
            distro: String::new(),
            python: Python::Path { path },
        },
        own: false,
        installed: None,
    })
}

/// The Environment the Engine runs in now, if one is ready.
pub fn current(envs: &Environments) -> Option<Kept> {
    if let Some(dev) = dev_environment() {
        return Some(dev);
    }
    let env = envs.default_environment()?;
    envs.get(&env).filter(|k| k.own)
}

fn resources(app: &AppHandle) -> Result<(Resources, PathBuf), Problem> {
    let dir = app
        .path()
        .resource_dir()
        .map_err(|e| Problem::new("environment.no_resources", e.to_string()))?;
    let data = app
        .path()
        .app_local_data_dir()
        .map_err(|e| Problem::new("environment.no_data", e.to_string()))?;
    Ok((Resources::find(&dir)?, data))
}

/// Make the Environment ready: check, upgrade, repair or install it (see
/// the module docs). `reinstall` probes again and installs from nothing.
fn ensure(
    app: &AppHandle,
    reinstall: bool,
    tell: &dyn Fn(Said) -> bool,
    cancelled: &dyn Fn() -> bool,
) -> Result<Kept, Problem> {
    if let Some(dev) = dev_environment() {
        return Ok(dev);
    }
    let procs = app.state::<Processes>();
    let envs = app.state::<Environments>();
    let (res, data) = resources(app)?;
    let recipe = res.recipe();
    let shipped = res.shipped(&data);
    let events: install::Events = {
        let tell_page = {
            let app = app.clone();
            move |s: Said| send(&app, s)
        };
        Arc::new(move |p: Progress| {
            tell_page(match p {
                Progress::Step { i, n, what } => Said::Step { i, n, what },
                Progress::Out { line } => Said::Out { line },
                Progress::Err { line } => Said::Err { line },
            })
        })
    };

    let existing = current(&envs).filter(|_| !reinstall);
    let own = match &existing {
        Some(kept) => own_of(&kept.environment),
        None => {
            let choice = probe(&procs);
            let own = choice.own.clone();
            tell(Said::Chose { choice });
            own
        }
    };
    let req = |fresh| Request { own: &own, shipped: &shipped, recipe: &recipe, fresh };

    if let Some(kept) = existing {
        if kept.installed.as_deref() != Some(res.build.as_str()) {
            tell(Said::Doing { what: "upgrade".into() });
            return install::install(&req(false), &procs, &envs, events, cancelled);
        }
        tell(Said::Doing { what: "check".into() });
        return match install::verify(&kept, &recipe, &procs, CHECK_TIMEOUT) {
            Health::Healthy => Ok(kept),
            Health::Stale | Health::Broken { .. } => {
                tell(Said::Doing { what: "repair".into() });
                install::repair(&req(true), &procs, &envs, events, cancelled)
            }
        };
    }
    tell(Said::Doing { what: "install".into() });
    install::install(&req(reinstall), &procs, &envs, events, cancelled)
}

/// Which `Own` made an Environment of the app's own.
fn own_of(env: &Environment) -> Own {
    match env.machine {
        Machine::Windows => Own::Windows,
        Machine::Wsl => Own::Wsl { distro: env.distro.clone() },
        Machine::Posix => Own::Posix,
    }
}

struct Channels(std::sync::Mutex<Option<Channel<Said>>>);

fn send(app: &AppHandle, s: Said) -> bool {
    let channels = app.state::<Channels>();
    let guard = channels.0.lock().expect("environment channel lock");
    guard.as_ref().is_some_and(|c| c.send(s).is_ok())
}

pub fn manage(app: &AppHandle) -> Result<(), Box<dyn std::error::Error>> {
    let data = app.path().app_local_data_dir()?;
    std::fs::create_dir_all(&data)?;
    app.manage(Environments::load(&data));
    app.manage(Setup::default());
    app.manage(Channels(std::sync::Mutex::new(None)));
    Ok(())
}

/// Make the Environment ready, telling the page each step on `on`; the
/// last message is `ready` or `failed`.
#[tauri::command]
pub async fn environment_ensure(
    app: AppHandle,
    reinstall: bool,
    on: Channel<Said>,
) -> Result<(), Problem> {
    let setup = app.state::<Setup>();
    if setup.busy.swap(true, Ordering::SeqCst) {
        return Err(Problem::new("environment.busy", "The Environment is already being set up."));
    }
    setup.cancel.store(false, Ordering::SeqCst);
    *app.state::<Channels>().0.lock().expect("environment channel lock") = Some(on);
    let worker = app.clone();
    let outcome = tauri::async_runtime::spawn_blocking(move || {
        let cancel = worker.state::<Setup>().cancel.clone();
        let tell = |s: Said| send(&worker, s);
        let result = ensure(&worker, reinstall, &tell, &|| cancel.load(Ordering::SeqCst));
        match result {
            Ok(kept) => tell(Said::Ready { kept }),
            Err(problem) => tell(Said::Failed { problem }),
        };
    })
    .await;
    setup.busy.store(false, Ordering::SeqCst);
    outcome.map_err(|e| Problem::new("environment.crashed", e.to_string()))
}

/// Stop a setup that is running: the running step is killed.
#[tauri::command]
pub fn environment_cancel(setup: State<'_, Setup>) {
    setup.cancel.store(true, Ordering::SeqCst);
}

/// The Environment ready now, if any.
#[tauri::command]
pub fn environment_current(envs: State<'_, Environments>) -> Option<Kept> {
    current(&envs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn wsl_with_a_gpu_wins() {
        let c = choose(
            Os::Windows,
            &s(&["Debian", "Ubuntu"]),
            &[("Debian".into(), None), ("Ubuntu".into(), Some("GPU 0: RTX".into()))],
        );
        assert_eq!(c.own, Own::Wsl { distro: "Ubuntu".into() });
        assert_eq!(c.why, Why::Gpu);
        assert_eq!(c.gpu.as_deref(), Some("GPU 0: RTX"));
    }

    #[test]
    fn no_gpu_or_no_wsl_means_native() {
        let c = choose(Os::Windows, &s(&["Ubuntu"]), &[("Ubuntu".into(), None)]);
        assert_eq!((c.own, c.why), (Own::Windows, Why::NoGpuInWsl));
        let c = choose(Os::Windows, &[], &[]);
        assert_eq!((c.own, c.why), (Own::Windows, Why::NoWsl));
    }

    #[test]
    fn the_check_prints_the_build_last() {
        let code = check_code("0.2.1+00ff");
        assert!(code.trim_end().ends_with("print(\"0.2.1+00ff\")"));
    }

    #[test]
    fn recipe_pins_with_a_lockfile_and_falls_back_without() {
        let mut res = Resources {
            uv: "uv.exe".into(),
            uv_linux: "uv".into(),
            wheel: PathBuf::from(r"C:\r\engine\cloudlabeller-0.2.1-py3-none-any.whl"),
            lockfile: None,
            build: "0.2.1+1".into(),
        };
        let r = res.recipe();
        assert_eq!(r.packages.len(), 1);
        assert!(r.wheels.is_empty() && r.lockfile.is_none());
        res.lockfile = Some(PathBuf::from(r"C:\r\engine\requirements.lock"));
        let r = res.recipe();
        assert!(r.packages.is_empty());
        assert_eq!(r.wheels.len(), 1);
        assert_eq!(r.name, "engine");
        assert_eq!(r.app, "cloudlabeller");
        assert!(r.problem().is_none(), "{:?}", r.problem());
    }

    #[test]
    fn wheel_version_and_digest() {
        let w = Path::new("cloudlabeller-0.2.1-py3-none-any.whl");
        assert_eq!(wheel_version(w).as_deref(), Some("0.2.1"));
        assert_ne!(fnv1a(&[b"a", b"b"]), fnv1a(&[b"ab"]));
        assert_eq!(fnv1a(&[b"x"]), fnv1a(&[b"x"]));
    }

    #[test]
    fn plain_drops_the_verbatim_prefix() {
        assert_eq!(plain(Path::new(r"\\?\C:\x")), r"C:\x");
        assert_eq!(plain(Path::new(r"C:\x")), r"C:\x");
    }
}
