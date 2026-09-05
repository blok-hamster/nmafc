"""What the extractor actually classified, read from persisted benchmark stores.

The 82% CoreAnchor figure in extractor.py is attributed to the permissive prompt
and was measured by _probe_extraction.py on 25 exchanges of one conversation.
This reads every record the full runs actually wrote, so the split is measured
on the same data the reported accuracy came from, at full scale and free.

Reads Cold ROM (SQLite) rather than Hot RAM (LanceDB): the write log carries the
same entity_name, fact_content and memory_type, and needs no pyarrow, which
Application Control blocks on this machine.

Also flags CoreAnchors that look misfiled: entities named after a speech act
(the failure mode the tiered prompt was written to stop), and entities whose
names read as present-tense state rather than identity or milestone.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

SPEECH_ACT = re.compile(
    r"(^|_)(told|tells?|said|says?|shares?|shared|asks?|asked|mentions?|mentioned"
    r"|agrees?|agreed|reacts?|reacted|discusses?|discussed|expresses?|expressed"
    r"|acknowledges?|acknowledged|suggests?|suggested|responds?|responded"
    r"|greet\w*|compliments?|replies|replied|thanks?|wishes|invites?|offers?"
    r"|conversation|chat|talk|talked|advice|support|encourage\w*|sympath\w*)($|_)",
    re.IGNORECASE,
)

# Names describing a situation that can change: ActiveContext by the prompt's
# own rule, not something permanent.
STATEFUL = re.compile(
    r"(^|_)(current|currently|upcoming|plan|plans|planning|goal|goals|schedule"
    r"|scheduled|ongoing|progress|next|todo|intends?|intention|wants?|hopes?"
    r"|looking|recent|recently|now|today|feeling|mood)($|_)",
    re.IGNORECASE,
)


def audit(run_dir: Path, dump: int) -> None:
    dbs = sorted((run_dir / "stores").glob("*/cold.db"))
    if not dbs:
        print("no cold.db under", run_dir, "\n")
        return

    tiers: Counter[str] = Counter()
    speech: list[tuple[str, str]] = []
    stateful: list[tuple[str, str]] = []
    other: list[tuple[str, str]] = []

    for db in dbs:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        rows = con.execute(
            "SELECT entity_name, fact_content, memory_type FROM memory_event_log"
        ).fetchall()
        con.close()
        for name, fact, tier in rows:
            tiers[tier] += 1
            if tier != "CoreAnchor":
                continue
            pair = (name, (fact or "").strip())
            if SPEECH_ACT.search(name or ""):
                speech.append(pair)
            elif STATEFUL.search(name or ""):
                stateful.append(pair)
            else:
                other.append(pair)

    total = sum(tiers.values())
    print("=" * 74)
    print("%s  --  %d conversations, %d records" % (run_dir.name, len(dbs), total))
    print("=" * 74)
    for tier, n in tiers.most_common():
        print("  %-16s %6d  %5.1f%%" % (tier, n, 100.0 * n / total))

    core = tiers.get("CoreAnchor", 0)
    if not core:
        print()
        return

    def show(label: str, items: list[tuple[str, str]]) -> None:
        print("\n  %s: %d  (%.1f%% of CoreAnchors)"
              % (label, len(items), 100.0 * len(items) / core))
        seen = set()
        for name, fact in items:
            if name in seen:
                continue
            seen.add(name)
            print("      %-42s %s" % (name[:42], fact[:70]))
            if len(seen) >= dump:
                break

    show("CoreAnchors named after a speech act", speech)
    show("CoreAnchors naming a changeable state", stateful)
    show("other CoreAnchors (sample)", other)
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", default=["full_v2", "full_v3"])
    ap.add_argument("--dump", type=int, default=12)
    args = ap.parse_args()
    base = Path(__file__).resolve().parent / "results"
    for run in args.runs:
        audit(base / run, args.dump)
    sys.exit(0)
