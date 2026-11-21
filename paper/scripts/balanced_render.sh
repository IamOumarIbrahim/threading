#!/usr/bin/env bash
# Balanced check, step 2: draw every scene with every algorithm at 2,000-20,000 threads.
# Usage (repo root): paper/scripts/balanced_render.sh <work_dir> [parallel_jobs]
# Writes <work_dir>/r/<threads>/<scene>/<scene>_<algo>.png and the threader log with
# per-algorithm drawing times in <work_dir>/log/<threads>_<scene>.txt. SVGs are removed.
set -euo pipefail
WORK="$1"; JOBS="${2:-6}"
EXE="$(cd "$(dirname "$0")/../.." && pwd)/build/threader"
mkdir -p "$WORK/log"
for t in ${THREADS:-2000 4000 6000 8000 10000 12000 14000 16000 18000 20000}; do
  for f in "$WORK"/src/*.png; do echo "$t $f"; done
done | xargs -P "$JOBS" -L 1 bash -c '
  t=$0; f=$1; s=$(basename "$f" .png); out="'"$WORK"'/r/$t/$s"
  [ -f "'"$WORK"'/log/${t}_${s}.txt" ] && exit 0
  "'"$EXE"'" "$f" --threads "$t" --seed 7 --out "$out" > "'"$WORK"'/log/${t}_${s}.tmp" 2>/dev/null
  rm -f "$out"/*.svg
  mv "'"$WORK"'/log/${t}_${s}.tmp" "'"$WORK"'/log/${t}_${s}.txt"'
echo "done: $(ls "$WORK"/log/*.txt | wc -l) jobs"
