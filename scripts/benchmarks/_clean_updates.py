"""Cut the mined updates down to the ones that can actually measure anything.

`_mine_updates.py` produced 211 candidates and only 101 were used. The filter
that made that cut was run inline and never written down, which left the whole
mylocomoeval result resting on a file nobody could regenerate. This is that
filter, reconstructed against the three saved files and reproducing both stages
exactly: 211 -> 128 -> 101, with zero items in dispute apart from the two named
below.

Three things get dropped, and each is dropped because keeping it would make a
measurement meaningless rather than merely noisy.

ABSENCE       The `stale` field has to be a concrete earlier value, because the
              measurement it feeds is "did the superseded fact reach the
              prompt". Roughly forty percent of what the miner returned has a
              stale of "no pet mentioned" or "not specified before turn 31" --
              the thing that changed is that something went from unsaid to
              said. There is no earlier fact in the store to retrieve, so
              stale-in-prompt can only ever read false and every such question
              silently inflates the score.

TURN-ANCHORED "How many turtles does Nate have as of turn 302?" cannot be
              answered by either arm. Both are asked once, after the last turn,
              with no notion of standing at turn 302; and the phrasing leaks a
              turn number that nothing in either prompt can interpret. The
              question is well formed and unanswerable, which is worse than
              being hard.

REPEATED      Six questions in the raw set are the same question written twice,
              and several more are different wordings of one update: three ask
              which game James is playing, all answering "Cyberpunk 2077" over
              "The Witcher 3". Left in, one update carries three votes and the
              conversation with the most verbose miner output dominates the
              total. Deduplication is on (conversation, gold, stale) rather
              than on the question text, since it is the update being tested
              that must be distinct, not the sentence asking about it.

Two items are held out by name rather than by rule. "natural" and "previous
job" are stale values that name no value: the first is arguably a real prior
state and the second is a placeholder the miner failed to fill. They are listed
explicitly so the judgement is visible instead of being buried in a regex that
would have to be bent to catch them.

No API calls. Reads JSON, writes JSON.

Usage:
    python scripts/benchmarks/_clean_updates.py --check
    python scripts/benchmarks/_clean_updates.py --src C:/nmafc_ab/updates.json \
        --out C:/nmafc_ab/updates_final.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# A stale value that reports the absence of a value. Anchored at the start as
# well as searched, because "no dog" and "not on a diet" are absences while
# "nobody's business" and "notary" are not.
ABSENCE = re.compile(
    r"^(no|none|nothing|n/?a|unknown|unspecified)\b"
    r"|^not\b"
    r"|\b(not (specified|mentioned|stated|given|known|established|discussed"
    r"|revealed)|unknown|unspecified|none mentioned)\b",
    re.I,
)

# Turn-anchored questions. The miner writes "as of turn N" and "in turn N".
TURN_ANCHORED = re.compile(r"\b(as of|in|by|at) turn \d+", re.I)

# Named exclusions. See the module docstring: these are judgements, not rules.
PLACEHOLDER_STALE = {"natural", "previous job"}


def usable(item: dict) -> bool:
    """Does this question have a superseded value that could be retrieved?"""
    stale = str(item.get("stale", "")).strip()
    if not stale or ABSENCE.search(stale) or stale.lower() in PLACEHOLDER_STALE:
        return False
    return not TURN_ANCHORED.search(item.get("question", ""))


def clean(items: list[dict]) -> list[dict]:
    kept: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        if not usable(item):
            continue
        key = (item["conv"], str(item["gold"]).lower().strip(),
               str(item["stale"]).lower().strip())
        if key in seen:
            continue
        seen.add(key)
        kept.append(item)
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="C:/nmafc_ab/updates.json")
    ap.add_argument("--out", default="C:/nmafc_ab/updates_final.json")
    ap.add_argument("--check", action="store_true",
                    help="Compare against the saved file instead of writing.")
    args = ap.parse_args()

    raw = json.loads(Path(args.src).read_text(encoding="utf-8"))
    kept = clean(raw)
    dropped = len(raw) - len(kept)
    print(f"{len(raw)} mined -> {len(kept)} usable ({dropped} dropped)")

    absent = sum(1 for i in raw if ABSENCE.search(str(i.get("stale", "")).strip()))
    anchored = sum(1 for i in raw if TURN_ANCHORED.search(i.get("question", "")))
    print(f"  absence stale   {absent}")
    print(f"  turn-anchored   {anchored}")
    print(f"  repeated update {dropped - absent - anchored - len(PLACEHOLDER_STALE)}"
          " (approximate: the categories overlap)")

    target = Path(args.out)
    if args.check:
        if not target.is_file():
            raise SystemExit(f"nothing to check against at {target}")
        saved = json.loads(target.read_text(encoding="utf-8"))
        mine = {(i["conv"], i["question"]) for i in kept}
        theirs = {(i["conv"], i["question"]) for i in saved}
        print(f"\nsaved {len(saved)}, reconstructed {len(kept)}, "
              f"disagreements {len(mine ^ theirs)}")
        for conv, question in sorted(mine ^ theirs):
            side = "only reconstructed" if (conv, question) in mine else "only saved"
            print(f"  {side}: [{conv}] {question[:80]}")
        return

    target.write_text(json.dumps(kept, indent=2), encoding="utf-8")
    print(f"\n-> {target}")


if __name__ == "__main__":
    main()
