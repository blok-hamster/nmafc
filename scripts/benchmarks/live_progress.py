"""Append a live progress line for a running benchmark.

The runner only prints when a whole conversation finishes, which for the memory
arms is a 30-45 minute gap of apparent silence. This polls the signals that do
move in the meantime -- per-worker memory-event counts and the runner log -- and
appends one timestamped line per interval to a file you can tail or keep open in
an editor.

    python -u scripts/benchmarks/live_progress.py \
        --log scripts/benchmarks/results/pilot.log \
        --out scripts/benchmarks/results/live.log
"""

from __future__ import annotations

import argparse
import glob
import os
import sqlite3
import tempfile
import time
from datetime import datetime

TEMP_GLOB = "nmafc_bench_*"


def event_counts() -> list[tuple[str, int]]:
    """Per-worker memory-event counts, newest storage dirs first."""
    counts = []
    dirs = glob.glob(os.path.join(tempfile.gettempdir(), TEMP_GLOB))
    dirs.sort(key=lambda d: os.path.getmtime(d), reverse=True)
    for d in dirs[:8]:
        db = os.path.join(d, "cold.db")
        if not os.path.exists(db):
            continue
        try:
            # read-only so we never contend with the writer holding the WAL
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2)
            n = conn.execute("select count(*) from memory_event_log").fetchone()[0]
            conn.close()
        except Exception:
            continue
        counts.append((os.path.basename(d)[-8:], n))
    return counts


def current_arm(log_path: str) -> tuple[str, int]:
    """Most recent 'ARM:' header in the runner log, and the log's line count."""
    if not os.path.exists(log_path):
        return "(no log yet)", 0
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception:
        return "(unreadable)", 0
    arm = "(none yet)"
    for line in reversed(lines):
        if "ARM:" in line:
            arm = line.split("ARM:")[1].strip()
            break
    return arm, len(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="runner log to watch")
    ap.add_argument("--out", required=True, help="file to append progress lines to")
    ap.add_argument("--interval", type=int, default=20, help="seconds between samples")
    args = ap.parse_args()

    prev: dict[str, int] = {}
    started = time.time()

    with open(args.out, "a", encoding="utf-8") as out:
        out.write(f"\n=== watching {args.log} @ {datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
        out.flush()

        while True:
            arm, log_lines = current_arm(args.log)
            counts = event_counts()

            parts = []
            for name, n in counts:
                delta = n - prev.get(name, n)
                parts.append(f"{name}={n}{f' (+{delta})' if delta else ''}")
                prev[name] = n

            mins = (time.time() - started) / 60
            workers = "  ".join(parts) if parts else "no memory workers active"
            out.write(
                f"[{datetime.now():%H:%M:%S}] +{mins:5.1f}m  arm={arm:<18} "
                f"log={log_lines:>3}L  {workers}\n"
            )
            out.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
