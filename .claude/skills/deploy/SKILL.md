---
name: deploy
description: Update the installed CloudLabeller distribution (C:\Programas\CloudLabeller) with the latest code from this working tree. Use when the user says "deploy", "update the distribution", "update the installed app", or similar.
---

# Deploy to the installed distribution

The installed app at `C:\Programas\CloudLabeller` is an embedded Python
3.10 bundle with the package at `app\cloudlabeller` and its own `colmap\`
folder. Deploying = mirroring the pure-Python `cloudlabeller/` package into
it. The embedded Python, its site-packages and the COLMAP bundle are NEVER
touched.

## Steps

1. Make sure the test suite is green first (`tests/`, per CLAUDE.md). Do not
   deploy on top of failing tests without telling the user.
2. Run the script (Bash tool):

   ```bash
   bash .claude/skills/deploy/deploy.sh
   ```

   It performs, in order:
   - **Delete-safety check** — lists files that exist only in the
     distribution and aborts (mirror would delete them). Inspect the list;
     if they are genuinely obsolete, re-run with `--force`. If you don't
     know what they are, ask the user.
   - **Running-process note** — syncing `.py` files while the app runs is
     safe (loaded code is in memory; `colmap.exe` never reads them), but the
     GUI needs a restart to pick the changes up, and any subprocess stage it
     launches from now on already runs the new code. Relay this note.
   - **Mirror sync** — `robocopy /MIR` excluding `__pycache__`.
   - **Verification** — byte-compiles the package and runs an import smoke
     test with the distribution's own interpreter.

3. Report: how many files robocopy copied, that verification passed, and
   remind the user to restart the installed app when convenient.

## Failure modes

- "no distribution at …" — this machine has no installed copy (e.g. the
  OneDrive-synced repo on the other machine); nothing to do.
- Byte-compilation / import failure — the distribution is now in a mixed
  state; fix the error and re-deploy before the user relaunches the app.
- New third-party dependencies are NOT handled by this script: if the code
  gained one, install it into the embedded Python first
  (`C:\Programas\CloudLabeller\python.exe -m pip install <pkg>`), then
  deploy.
- License hygiene: CloudLabeller is GPL-3.0-or-later (dual-licensed with a
  commercial option — see README.md). GPL dependencies (`plyfile`, the
  COLMAP bundle) are compatible with the public GPL build; they only need
  stripping from a future COMMERCIAL build (plyfile is unused by the app —
  hylite imports it lazily in a code path CloudLabeller never calls). After
  any pip change in the embedded Python, regenerate the notices:
  `C:\Programas\CloudLabeller\python.exe tools\gen_third_party_licenses.py
  --out C:\Programas\CloudLabeller\THIRD_PARTY_LICENSES.txt`. Public
  releases ship WITHOUT `colmap\` (smaller; the app downloads it via
  Photogrammetry → Download COLMAP…) — use `scripts/build_portable.py
  --skip-colmap`.
