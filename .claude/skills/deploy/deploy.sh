#!/usr/bin/env bash
# Deploy the cloudlabeller package into the installed distribution
# (C:\Programas\CloudLabeller). Pure-Python sync: the embedded Python,
# site-packages and the bundled colmap/ are never touched.
#
# Usage: deploy.sh [--force]
#   --force   proceed even if the distribution contains package files that
#             do not exist in the source tree (they WILL be deleted).
set -u

# Repo root = three levels above this script (.claude/skills/deploy/).
here="$(cd "$(dirname "$0")" && pwd)"
src="$(cd "$here/../../.." && pwd)/cloudlabeller"
dist_root="/c/Programas/CloudLabeller"
dest="$dist_root/app/cloudlabeller"
dist_py="$dist_root/python.exe"

fail() { echo "DEPLOY FAILED: $1"; exit 1; }

[ -d "$src" ] || fail "source package not found at $src"
[ -d "$dest" ] || fail "no distribution at $dest (nothing to update on this machine)"
[ -x "$dist_py" ] || fail "distribution python not found at $dist_py"

# --- safety: what would a mirror delete? -----------------------------------
tmp="$(mktemp -d)"
(cd "$src" && find . -name __pycache__ -prune -o -type f -print | sort) > "$tmp/src.txt"
(cd "$dest" && find . -name __pycache__ -prune -o -type f -print | sort) > "$tmp/dist.txt"
extras="$(comm -13 "$tmp/src.txt" "$tmp/dist.txt")"
if [ -n "$extras" ] && [ "${1:-}" != "--force" ]; then
    echo "Files exist ONLY in the distribution (a mirror sync would DELETE them):"
    echo "$extras"
    fail "re-run with --force to delete them, or reconcile first"
fi

# --- informational: is the installed app running? ---------------------------
# Syncing .py files is safe while it runs (imports are already in memory;
# colmap.exe never reads them) — but the GUI needs a restart to pick up the
# new code, and stages launched from now on already use it.
running="$(powershell -NoProfile -Command \
    "Get-Process python, pythonw, colmap -ErrorAction SilentlyContinue | Select-Object -ExpandProperty ProcessName" 2>/dev/null | sort -u | tr '\n' ' ')"
[ -n "$running" ] && echo "NOTE: running processes detected ($running) — sync is safe; restart the app to load the new code."

# --- sync -------------------------------------------------------------------
robocopy "$(cygpath -w "$src")" "$(cygpath -w "$dest")" //MIR //XD __pycache__ //NDL //NJH //NP
code=$?
[ $code -le 7 ] || fail "robocopy exit $code"

# --- verify with the distribution's own interpreter -------------------------
cd "$dist_root/app" || fail "cannot cd to $dist_root/app"
"$dist_py" -m compileall -q cloudlabeller || fail "byte-compilation failed on the distribution Python"
"$dist_py" - <<'EOF' || fail "import smoke test failed"
import cloudlabeller
from cloudlabeller.core.project import Project
from cloudlabeller.photogrammetry import adjacency, meshing, mvs, pipeline, sfm
from cloudlabeller.ui import status
print("import smoke test OK (cloudlabeller", cloudlabeller.__version__ + ")")
EOF

echo "DEPLOY OK: robocopy exit $code (see counts above); distribution verified."
