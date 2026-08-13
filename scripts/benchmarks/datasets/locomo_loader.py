"""LoCoMo dataset loader.

Loads the real LoCoMo (Long-term Conversational Memory) dataset from HuggingFace.
Source: KimmoZZZ/locomo (locomo10.json)

Dataset contains 10 multi-session conversations with ~200 QA pairs each,
spanning 5 reasoning categories used in the original paper (Maharana et al., 2024).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from huggingface_hub import hf_hub_download

REPO_ID = "KimmoZZZ/locomo"
FILENAME = "locomo10.json"

CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial",
}


def _session_index(key: str) -> int:
    """Sort key for 'session_<n>' keys, ordering them numerically.

    Sorting these as plain text is wrong once a conversation has ten or more
    sessions: 'session_10' sorts before 'session_2'.
    """
    part = key.split("_")[1] if "_" in key else ""
    return int(part) if part.isdigit() else 0


@dataclass
class LoCoMoQA:
    question: str
    answer: str
    evidence: list[str]
    category: int

    @property
    def category_name(self) -> str:
        return CATEGORY_NAMES.get(self.category, f"unknown-{self.category}")


@dataclass
class LoCoMoConversation:
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: list[list[dict]]
    session_dates: list[str]
    qa_pairs: list[LoCoMoQA]
    event_summary: dict = field(default_factory=dict)
    observations: dict = field(default_factory=dict)
    session_summaries: dict = field(default_factory=dict)

    @property
    def total_turns(self) -> int:
        return sum(len(s) for s in self.sessions)

    @property
    def num_sessions(self) -> int:
        return len(self.sessions)

    def get_flat_history(self, up_to_session: int | None = None) -> list[dict]:
        """Flatten sessions into a single turn list.

        Each turn carries the speaker's real name and the timestamp of the
        session it belongs to, alongside the role mapping.

        Both matter for scoring. LoCoMo's temporal category asks questions like
        "When did Jon lose his job?" against gold answers such as
        "19 January, 2023" — answerable only from the session timestamps, which
        this loader previously parsed into `session_dates` and then discarded
        here. Every arm received an undated transcript, so that entire category
        was unanswerable by construction. Speaker names matter for the same
        reason: the questions refer to people by name, while a bare
        role/content turn renders as "User"/"Assistant".

        Consumers that only want role/content can keep ignoring the extra keys.
        """
        sessions = self.sessions[:up_to_session] if up_to_session else self.sessions
        dates = (
            self.session_dates[:up_to_session] if up_to_session else self.session_dates
        )
        turns = []
        for idx, session in enumerate(sessions):
            date = dates[idx] if idx < len(dates) else ""
            for turn in session:
                role = "user" if turn["speaker"] == self.speaker_a else "assistant"
                turns.append({
                    "role": role,
                    "content": turn["text"],
                    "speaker": turn["speaker"],
                    "date": date,
                    "session": idx + 1,
                })
        return turns


def load_locomo(cache_dir: str | None = None) -> list[LoCoMoConversation]:
    """Download and parse the full LoCoMo dataset from HuggingFace."""
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        repo_type="dataset",
        cache_dir=cache_dir,
    )

    with open(path) as f:
        raw_data = json.load(f)

    conversations = []
    for item in raw_data:
        conv_data = item["conversation"]

        session_keys = sorted(
            (
                k for k in conv_data.keys()
                if k.startswith("session_") and not k.endswith("_date_time")
            ),
            key=_session_index,
        )

        # Look each timestamp up from its own session key rather than building a
        # second sorted list. Sorting by text put the sessions in dictionary
        # order -- 1, 10, 11, ... 19, 2, 20 -- which alone told the story out of
        # chronological order. Worse, the date keys carry a `_date_time` suffix,
        # so text order placed 'session_10_date_time' *before*
        # 'session_1_date_time' while 'session_1' still sorted before
        # 'session_10'. The two lists disagreed and every session was stamped
        # with a different session's date, making the temporal question category
        # unanswerable for every arm.
        sessions = [conv_data[k] for k in session_keys]
        session_dates = [conv_data.get(f"{k}_date_time", "") for k in session_keys]

        qa_pairs = []
        for qa in item.get("qa", []):
            answer = qa.get("answer") or qa.get("adversarial_answer", "")
            qa_pairs.append(LoCoMoQA(
                question=qa["question"],
                answer=str(answer),
                evidence=qa.get("evidence", []),
                category=qa["category"],
            ))

        conversations.append(LoCoMoConversation(
            sample_id=item.get("sample_id", f"conv-{len(conversations)}"),
            speaker_a=conv_data.get("speaker_a", "Speaker A"),
            speaker_b=conv_data.get("speaker_b", "Speaker B"),
            sessions=sessions,
            session_dates=session_dates,
            qa_pairs=qa_pairs,
            event_summary=item.get("event_summary", {}),
            observations=item.get("observation", {}),
            session_summaries=item.get("session_summary", {}),
        ))

    return conversations


def iter_qa_by_category(
    conversations: list[LoCoMoConversation],
    category: int | None = None,
) -> list[tuple[LoCoMoConversation, LoCoMoQA]]:
    """Yield (conversation, qa) pairs filtered by category."""
    results = []
    for conv in conversations:
        for qa in conv.qa_pairs:
            if category is None or qa.category == category:
                results.append((conv, qa))
    return results


def get_dataset_stats(conversations: list[LoCoMoConversation]) -> dict:
    """Return summary statistics for the loaded dataset."""
    total_qa = sum(len(c.qa_pairs) for c in conversations)
    total_turns = sum(c.total_turns for c in conversations)
    total_sessions = sum(c.num_sessions for c in conversations)

    category_counts: dict[str, int] = {}
    for conv in conversations:
        for qa in conv.qa_pairs:
            name = qa.category_name
            category_counts[name] = category_counts.get(name, 0) + 1

    return {
        "conversations": len(conversations),
        "total_qa_pairs": total_qa,
        "total_turns": total_turns,
        "total_sessions": total_sessions,
        "qa_by_category": category_counts,
    }
