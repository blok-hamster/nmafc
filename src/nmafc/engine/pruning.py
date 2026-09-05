from __future__ import annotations

from typing import TYPE_CHECKING

from nmafc.schemas.memory import DecayConfig, MemoryRecord, MemoryStateUpdate
from nmafc.storage.cold_base import ColdStorageBase
from nmafc.storage.hot import HotStorage

if TYPE_CHECKING:
    from nmafc.storage.event_log import EventLog


def detect_override(
    new_update: MemoryStateUpdate,
    existing: list[MemoryRecord],
) -> list[MemoryRecord]:
    """Find existing records that the new update contradicts.

    Detection rules:
    1. Explicit: new_update.overrides_entity matches record.entity_name
    2. Implicit: same entity_name (the new fact replaces the old one)
    """
    targets = []
    for record in existing:
        if new_update.overrides_entity and record.entity_name == new_update.overrides_entity:
            targets.append(record)
        elif record.entity_name == new_update.entity_name:
            targets.append(record)
    return targets


def apply_suppression(record: MemoryRecord, gamma: float) -> MemoryRecord:
    """Apply synaptic suppression multiplier to a contradicted record.

    Legacy path — kept for backward compatibility with tests and benchmarks
    that explicitly set exclude_invalidated=False.
    """
    new_weight = record.weight * gamma
    return record.model_copy(update={"weight": new_weight})


def invalidate_record(record: MemoryRecord, current_turn: int) -> MemoryRecord:
    """Mark a contradicted record as temporally invalidated.

    The record's weight is frozen (not multiplied by gamma) and invalid_at is
    set. The record remains in Hot RAM but is excluded from default searches.
    """
    return record.model_copy(update={"invalid_at": current_turn})


def create_suppression_event(
    old_record: MemoryRecord,
    new_weight: float,
    suppressed_by: str,
    turn: int,
) -> MemoryEvent:
    """Create a SUPPRESSION event for a contradicted record.

    Call this after apply_suppression to log the override for the Web UI.
    """
    from nmafc.schemas.events import EventType, MemoryEvent

    return MemoryEvent(
        event_type=EventType.SUPPRESSION,
        turn=turn,
        record_id=old_record.id,
        entity_name=old_record.entity_name,
        old_weight=old_record.weight,
        new_weight=new_weight,
        suppressed_by=suppressed_by,
        old_memory_type=old_record.memory_type.value,
    )


def identify_prunable(records: list[MemoryRecord], w_prune: float) -> list[str]:
    """Return IDs of records whose weight has fallen below the prune threshold."""
    return [r.id for r in records if r.weight <= w_prune]


def _reachable_survivors(
    start: str,
    outgoing: dict[str, list[str]],
    doomed: set[str],
    max_depth: int,
) -> list[str]:
    """Entity names still reachable from a doomed entity once it is gone.

    Walks forward from `start` along `related_entities`, passing straight
    through other doomed entities and collecting the surviving ones it lands on.
    The chain matters: small talk decays in runs, so a link into a doomed node
    very often leads to another doomed node, and stopping at depth one would
    leave most rewiring with nothing to attach to.

    Breadth-first and depth-capped rather than a full closure. Each extra level
    reaches a fact one degree further from anything actually said together, so
    an uncapped walk would eventually connect a conversation to itself and hand
    every retrieval the entire store back.
    """
    survivors: list[str] = []
    seen = {start}
    frontier = [start]

    for _ in range(max_depth):
        following: list[str] = []
        for entity in frontier:
            for target in outgoing.get(entity, ()):
                key = target.lower()
                if key in seen:
                    continue
                seen.add(key)
                if key in doomed:
                    following.append(key)
                else:
                    survivors.append(target)
        if not following:
            break
        frontier = following

    return survivors


def plan_rewiring(
    records: list[MemoryRecord],
    doomed_ids: set[str],
    max_links: int,
    max_depth: int = 3,
) -> list[tuple[str, list[str]]]:
    """Reroute links around records that are about to be deleted.

    Returns (record_id, new_related_entities) for every surviving record whose
    links change, ready for HotStorage.apply_relation_updates.

    The problem this solves was measured rather than assumed. On the ten-store
    LoCoMo run the extractor wrote 1.44 links per fact with 10% of facts
    unlinked; the live stores afterwards held 0.92 links per fact with 31%
    unlinked. Decay removes 55% of everything written -- correctly, it is
    small talk -- and the dead-pointer sweep in MemoryConsolidator then removes
    every link that pointed at it, taking 36% of the graph with it. Spreading
    Activation was not failing to traverse. It was traversing a graph that
    forgetting had disconnected, which is the two signature mechanisms of the
    system quietly cancelling each other out.

    Severing is the wrong response to a dead node when the node was only ever a
    waypoint: "Maria mentioned the puppy" is worth forgetting, but the path from
    Maria to the puppy is not. Rewiring keeps the path and drops the waypoint,
    which is also what the biology suggests -- consolidation is usually
    described as the gist outliving the episode that carried it, not as the gist
    being lost along with it.

    Two rules keep this from degenerating into linking everything to everything:

    * Direct links are never displaced. A surviving record keeps every link it
      already had to a surviving entity, and rewired links are only added in
      whatever room is left under `max_links`. An inferred connection must never
      cost a stated one.
    * Fan-out is capped. A doomed hub with twelve neighbours would otherwise
      hand twelve links to each of its twelve inbound neighbours, and the
      retrieval that follows would return the conversation.
    """
    if not doomed_ids or max_links <= 0:
        return []

    doomed_entities = {
        r.entity_name.lower() for r in records if r.id in doomed_ids
    }
    if not doomed_entities:
        return []

    # Forward adjacency for the doomed nodes only: they are the only ones a
    # walk ever passes *through*, since a surviving node is an endpoint.
    outgoing: dict[str, list[str]] = {}
    for record in records:
        key = record.entity_name.lower()
        if key in doomed_entities:
            outgoing.setdefault(key, []).extend(record.related_entities)

    # Memoised per doomed entity: several records commonly point at the same
    # one, and the walk is the expensive part of this pass.
    resolved: dict[str, list[str]] = {}

    updates: list[tuple[str, list[str]]] = []
    for record in records:
        if record.id in doomed_ids or not record.related_entities:
            continue

        self_key = record.entity_name.lower()
        kept: list[str] = []
        rewired: list[str] = []
        seen = {self_key}
        touched = False

        for link in record.related_entities:
            key = link.lower()
            if key not in doomed_entities:
                if key not in seen:
                    seen.add(key)
                    kept.append(link)
                continue

            touched = True
            if key not in resolved:
                resolved[key] = _reachable_survivors(
                    key, outgoing, doomed_entities, max_depth
                )
            for target in resolved[key]:
                target_key = target.lower()
                if target_key not in seen:
                    seen.add(target_key)
                    rewired.append(target)

        if not touched:
            continue

        new_links = kept + rewired[: max(0, max_links - len(kept))]
        if new_links != list(record.related_entities):
            updates.append((record.id, new_links))

    return updates


def prune_cycle(
    hot: HotStorage,
    cold: ColdStorageBase,
    w_prune: float,
    current_turn: int,
    event_logger: EventLog | None = None,
    config: DecayConfig | None = None,
) -> int:
    """Evict or invalidate below-threshold records from Hot RAM.

    Behavior depends on memory type:
    - EphemeralState below w_prune: physically deleted (genuine expiry)
    - ActiveContext below w_prune: temporally invalidated (invalid_at set,
      excluded from default search but retained for temporal queries)
    - Already-invalidated records below 0.01: physically deleted to prevent
      indefinite accumulation

    Returns the count of records removed or invalidated.

    `config` enables link rewiring around the deleted records (see
    plan_rewiring). It is optional so that existing callers keep the old
    sever-the-link behaviour, which is also what the ablation arm needs: the
    finding that forgetting disconnects the graph is only demonstrable if the
    unrewired variant can still be run.

    When `event_logger` is provided, PRUNE events are emitted for each
    affected record.
    """
    from nmafc.schemas.memory import MemoryType

    all_records = hot.get_all()

    delete_ids: list[str] = []
    invalidate_updates: list[tuple[str, int]] = []

    for rec in all_records:
        if rec.weight > w_prune:
            if rec.invalid_at is not None and rec.weight < 0.01:
                delete_ids.append(rec.id)
            continue

        if rec.invalid_at is not None:
            delete_ids.append(rec.id)
        elif rec.memory_type == MemoryType.EPHEMERAL_STATE:
            delete_ids.append(rec.id)
        else:
            invalidate_updates.append((rec.id, current_turn))

    if event_logger is not None and (delete_ids or invalidate_updates):
        from nmafc.schemas.events import EventType, MemoryEvent

        affected = set(delete_ids) | {uid for uid, _ in invalidate_updates}
        for rec in all_records:
            if rec.id in affected:
                event_logger.log(
                    MemoryEvent(
                        event_type=EventType.PRUNE,
                        turn=current_turn,
                        record_id=rec.id,
                        entity_name=rec.entity_name,
                        old_weight=rec.weight,
                        old_memory_type=rec.memory_type.value,
                    )
                )

    # Rewire before deleting, not after: the walk needs the doomed records'
    # own links to know where their inbound neighbours should be pointed
    # instead, and those links are gone the moment the rows are.
    if config is not None and config.rewire_pruned_links and delete_ids:
        rewiring = plan_rewiring(
            all_records,
            set(delete_ids),
            max_links=config.rewire_max_links,
            max_depth=config.rewire_max_depth,
        )
        hot.apply_relation_updates(rewiring)

    hot.delete_many(delete_ids)
    # Hot only, and deliberately. Supersession propagates to the archive
    # (see MemoryWrapper's override path) because a superseded fact is wrong.
    # A pruned fact is merely weak, and is still true, so it keeps its place in
    # Cold ROM: recall has failed while recognition survives, which is the whole
    # point of having two tiers.
    hot.set_invalid_at_many(invalidate_updates)

    return len(delete_ids) + len(invalidate_updates)
