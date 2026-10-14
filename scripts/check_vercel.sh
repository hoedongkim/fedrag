#!/usr/bin/env bash
# Run the public demo's Vercel function on this machine, as Vercel would build it, and call every
# endpoint the frontend uses:
#
#   scripts/check_vercel.sh [ENV_FILE]       (or: make vercel-check [ENV_FILE=...])
#
# 1. A clean virtualenv with nothing but requirements.txt (what Vercel installs), in
#    deploy/build/vercel/venv (git-ignored), rebuilt when requirements.txt changes.
# 2. The function's files as the next push would hold them: what git tracks or would add (not
#    ignored) under api/, backend/app/ and data/, copied to deploy/build/vercel/bundle. So
#    backend/.env, data/chroma and the models are never in it, and a file the demo reads that git
#    ignores is missing here as it would be on Vercel.
# 3. backend/tests/vercel_entry_probe.py on that copy with that interpreter, in an empty environment
#    except DATABASE_URL: from ENV_FILE (a file with a line DATABASE_URL=..., read without running
#    it through the shell) or else from the environment. Without one the database routes must
#    answer 503; with the read-only URL (docs/deploy.md, step 1) they must answer 200:
#
#      make vercel-check ENV_FILE=~/.fedrag-space.env
#
#    The URL reaches the probe through its environment only and is never printed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/deploy/build/vercel"
PYTHON="${PYTHON:-python3}"
ENV_FILE="${1:-}"

fail() { echo "check_vercel: $*" >&2; exit 1; }

database_url="${DATABASE_URL:-}"
if [ -n "$ENV_FILE" ]; then
  [ -r "$ENV_FILE" ] || fail "cannot read $ENV_FILE."
  database_url="$(sed -n 's/^DATABASE_URL=//p' "$ENV_FILE" | tail -n 1)"
  [ -n "$database_url" ] || fail "$ENV_FILE has no DATABASE_URL= line."
fi

want="$(tr -d '[:space:]' < "$ROOT/.python-version")"
have="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [ "$have" != "$want" ]; then
  echo "check_vercel: $PYTHON is Python $have; Vercel runs $want (.python-version). Continuing." >&2
fi

# 1. The virtualenv.
digest="$("$PYTHON" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$ROOT/requirements.txt")"
if [ ! -x "$OUT/venv/bin/python" ] || [ "$(cat "$OUT/venv/requirements.sha256" 2>/dev/null)" != "$digest" ]; then
  echo "Creating deploy/build/vercel/venv from requirements.txt..."
  rm -rf "$OUT/venv"
  mkdir -p "$OUT"
  "$PYTHON" -m venv "$OUT/venv"
  "$OUT/venv/bin/python" -m pip install --quiet --disable-pip-version-check -r "$ROOT/requirements.txt"
  echo "$digest" > "$OUT/venv/requirements.sha256"
fi

# 2. The function's files.
rm -rf "$OUT/bundle"
mkdir -p "$OUT/bundle" "$OUT/home" "$OUT/tmp"
(cd "$ROOT" && git ls-files -z --cached --others --exclude-standard -- api backend/app data) \
  | "$PYTHON" -c '
import shutil, sys
from pathlib import Path

root, bundle = Path(sys.argv[1]), Path(sys.argv[2])
for name in sys.stdin.buffer.read().decode().split("\0"):
    source = root / name
    if name and source.is_file():  # a tracked file deleted in the working tree is skipped
        (bundle / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, bundle / name)
' "$ROOT" "$OUT/bundle"

# 3. The probe, then a summary of its JSON.
status=0
env -i PATH=/usr/bin:/bin HOME="$OUT/home" TMPDIR="$OUT/tmp" \
  ${database_url:+"DATABASE_URL=$database_url"} \
  "$OUT/venv/bin/python" -I -B "$ROOT/backend/tests/vercel_entry_probe.py" "$OUT/bundle" \
  > "$OUT/probe.json" || status=$?
[ -s "$OUT/probe.json" ] || fail "the probe printed nothing (exit $status); see the error above."
site_packages="$("$OUT/venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
"$PYTHON" - "$OUT/probe.json" "$(du -sk "$OUT/bundle" | cut -f1)" "$(du -sk "$site_packages" | cut -f1)" <<'PY'
import json
import sys

result = json.loads(open(sys.argv[1], encoding="utf-8").read().strip().splitlines()[-1])
bundle_mb, packages_mb = int(sys.argv[2]) / 1024, int(sys.argv[3]) / 1024
database = "DATABASE_URL set" if result["with_database"] else "no database (DATABASE_URL not set)"
print(f"api/index.py -> {result['app_class']} from {result['app_package']}, with {database}")
print(f"{'call':<22}{'status':>7}{'bytes':>10}{'seconds':>9}  detail")
for name, call in result["calls"].items():
    detail = call.get("analysis_status") or call.get("passages_available") or call.get("items") or ""
    print(f"{name:<22}{call['status']:>7}{call['bytes']:>10,}{call['seconds']:>9.3f}  {detail}")
print(
    f"Import {result['import_seconds']} s, peak memory {result['peak_rss_mb']} MB; "
    f"{len(result['distributions'])} distributions imported; heavy modules: "
    f"{', '.join(result['heavy']) or 'none'}."
)
print(
    f"Files: {bundle_mb:.1f} MB of code and data, {packages_mb:.0f} MB of packages "
    "(this machine's wheels; Vercel's 500 MB limit counts the Linux ones)."
)
for problem in result["problems"]:
    print(f"PROBLEM: {problem}")
print("vercel-check: OK" if not result["problems"] else "vercel-check: FAILED")
PY
exit "$status"
