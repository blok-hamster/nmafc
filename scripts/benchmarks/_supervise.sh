#!/usr/bin/env bash
# Keep a resumable benchmark running until it finishes.
#
# The in-process retry budget covers an outage the process survives. It cannot
# cover the process not surviving: a power cut, an OOM kill, a hibernate that
# tears down the interpreter. Those need something outside the process, and the
# only reason that is cheap here is that the run resumes -- every answer is
# banked to --out as it lands, and a restart skips what is already there.
#
# Stops on success, on a provider refusal it cannot get past, or when two
# restarts in a row bank nothing. That last guard matters: without it a run
# that fails instantly on every attempt would spin here forever, looking busy.
#
# Usage:
#   scripts/benchmarks/_supervise.sh <results.json> <logfile> <command...>

set -u

RESULTS="$1"; shift
LOGFILE="$1"; shift

banked() {
    python -c "
import json,sys
try:
    print(len(json.load(open(sys.argv[1]))))
except Exception:
    print(0)
" "$RESULTS" 2>/dev/null || echo 0
}

attempt=0
stalled=0
previous=$(banked)

while true; do
    attempt=$((attempt + 1))
    echo "=== supervisor: attempt $attempt, $previous answers banked ===" >> "$LOGFILE"

    "$@" >> "$LOGFILE" 2>&1
    status=$?

    current=$(banked)
    echo "=== supervisor: exit $status, banked $previous -> $current ===" >> "$LOGFILE"

    if [ "$status" -eq 0 ]; then
        echo "=== supervisor: finished after $attempt attempt(s) ===" >> "$LOGFILE"
        exit 0
    fi

    if [ "$current" -le "$previous" ]; then
        stalled=$((stalled + 1))
        if [ "$stalled" -ge 2 ]; then
            echo "=== supervisor: two restarts with no progress, giving up ===" >> "$LOGFILE"
            exit 1
        fi
    else
        stalled=0
    fi

    previous=$current
    # Long enough for a network stack to come back after a wake, short enough
    # not to idle through it.
    sleep 30
done
