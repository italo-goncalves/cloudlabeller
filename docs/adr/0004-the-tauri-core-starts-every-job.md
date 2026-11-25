# The Tauri core starts every Job; the Engine plans them and applies their results

ADR-0003 gives the one GPU Lease to the Tauri core, because only the core
starts Jobs on both sides of the Windows/WSL boundary. ADR-0002's Engine,
as first built (2026-11-25), started Jobs itself. So the core couldn't see
or hold the Lease for them. Now the core starts every Job, GPU or not. The
Engine keeps the two parts that need the open Project:

1. **Plan** (`jobs.plan`): it checks that the Job can run, saves what the
   child will read, and answers with the command line, whether the Job
   wants the GPU, its log file, and how to stop it gracefully.
2. **Apply** (`jobs.finish`): the core reports how the Job ended and the
   `RESULT…` lines it printed. If it ended well, the Engine folds the
   products into the Project (reload products, record the new frame, mark
   predicted photos) and answers with the summary.

Between the two steps the core runs the process: it takes the Lease, stages
the Workspace, streams progress and the log to the page, and cancels the
Job.

## Considered Options

- **The Engine starts Jobs and asks the core for the Lease over the pipe**:
  this needs requests from the Engine to the core, as well as from the core
  to the Engine. A Job would also die with its Engine, and the core still
  couldn't stage a Workspace for a Job it didn't start.
- **The Engine starts CPU Jobs, the core GPU Jobs**: two ways to run one
  thing, and `sfm` is a GPU Job only when its options say so.

## Consequences

- One place runs processes: the core, through geokit's `process::Spawner`
  (a Windows Job Object or a Posix process group per Job, the WSL side
  ended too). The Engine runs no processes at all.
- The `*_cli.py` line protocol (`PROGRESS`, `RESULT…`) is now read by the
  core. The core passes the `RESULT…` lines back to the Engine unparsed.
- A Job outlives its Engine. If the Engine restarts, a Job's plan is lost,
  and the core reopens the Project once the Job ends so the Engine reloads
  what it wrote.
- The Engine still refuses a second plan of the same group (reconstruction
  or ml) until the first one is finished. So does closing the Project.
  This guards the Project's files, whatever the GPU Lease decides.
- The Glossary's "Engine … starts its Jobs" becomes "plans its Jobs".
