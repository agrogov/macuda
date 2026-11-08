#!/bin/sh
# prereq-check.sh <ID>... — a task's notes file is the hand-off: its first line must be exactly "STATUS: PASS".
#   sh scripts/prereq-check.sh J01
set -u
[ $# -ge 1 ] || { echo "usage: sh scripts/prereq-check.sh <ID>..." >&2; exit 2; }
JEFF=$(cd "$(dirname "$0")/.." && pwd)
rc=0
for id in "$@"; do
  f="$JEFF/notes/$id.md"
  if [ ! -f "$f" ]; then
    echo "prereq $id: FAIL (no $f)" >&2
    rc=1
    continue
  fi
  first=$(sed -n '1p' "$f")
  if [ "$first" != "STATUS: PASS" ]; then
    echo "prereq $id: FAIL (first line is '$first', want 'STATUS: PASS')" >&2
    rc=1
  fi
done
[ "$rc" -eq 0 ] && echo "prereq: $* PASS"
exit "$rc"
