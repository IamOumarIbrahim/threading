#!/usr/bin/env bash
# Fresh claude session per image, same prompt. Resumable. Sandboxed to images/ only.
# Optional env: ORDER=<id list file>  OUT=<predictions csv>  (defaults: order.txt, predictions.csv)
cd "$(dirname "$0")" || exit 1
ORDER="${ORDER:-order.txt}"; OUT="${OUT:-predictions.csv}"
export CLAUDE_CODE_DISABLE_AUTO_MEMORY=1
MODEL=claude-opus-5-5
# Free-form names: lowercase words (letters, spaces, hyphens, apostrophes), "|"-separated, max 6 words each.
valid() {
  [[ -z "$1" ]] && return 1
  [[ "$1" == "NONE" ]] && return 0
  local IFS='|' part
  for part in $1; do
    [[ "$part" =~ ^[a-z][-a-z\'\ ]*$ ]] || return 1
    (( $(wc -w <<< "$part") <= 6 )) || return 1
  done
}
ask() {
  claude -p "$1" \
    --model "$MODEL" \
    --tools Read \
    --permission-mode dontAsk \
    --allowedTools "Read(./images/**)" \
    --disallowedTools "Read(./*.csv)" "Read(./*.txt)" "Read(./run.sh)" "Read(./PROMPT.md)" \
    --setting-sources "" \
    --strict-mcp-config \
    --disable-slash-commands \
    --no-session-persistence < /dev/null 2>/dev/null
}
[ "$1" = "--probe" ] && { ask "$2"; exit; }
[ -f "$OUT" ] || echo "identifier,prediction" > "$OUT"
echo "$(date -u +%FT%TZ) model=$MODEL order=$ORDER out=$OUT" >> run_log.txt
while read -r id <&3; do
  [ -z "$id" ] && continue
  grep -q "^$id," "$OUT" && continue
  raw=$(ask "$(sed "s/{ID}/$id/g" PROMPT.md)"); rc=$?
  # Stop the run (don't skip ahead) on any CLI error or usage-limit message; rerun later to resume.
  if (( rc != 0 )) || grep -qiE "usage limit|rate limit|hit your limit|limit reached|resets at|resets in|api error|try again later|overloaded|unavailable" <<< "$raw"; then
    echo "STOPPED at $id (claude exit $rc): $(tr '\n' ' ' <<< "$raw" | cut -c1-200)" >&2
    echo "Nothing was written for $id. Rerun this script later to resume." >&2
    exit 1
  fi
  out=$(tr -d '\r' <<< "$raw" | sed '/^[[:space:]]*$/d' | tail -n1 | sed -E 's/^[[:space:]]+|[[:space:]]+$//g')
  out=$(sed -E 's/ *\| */|/g' <<< "$out")
  [[ "$out" == "NONE" ]] || out="${out,,}"
  if valid "$out"; then
    echo "$id,$out" >> "$OUT"; echo "$id -> $out"
  else
    echo "INVALID for $id (not written): $out" >&2
  fi
done 3< "$ORDER"
