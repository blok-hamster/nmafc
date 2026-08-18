"""Entity Graph endpoints — adjacency map, clustering coefficients, node details."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from nmafc.engine.decay import build_entity_graph, clustering_coefficient
from nmafc.web.deps import get_tenant_memory
from nmafc.wrapper import NeuromorphicMemory

router = APIRouter(prefix="/api/graph", tags=["graph"])

MemDep = Annotated[NeuromorphicMemory, Depends(get_tenant_memory())]


@router.get("")
def get_entity_graph(mem: MemDep):
    """Full entity graph: nodes with metadata, edges from related_entities."""
    records = mem._hot.get_all()
    if not records:
        return {"nodes": [], "edges": []}

    adj = build_entity_graph(records)

    entity_records: dict[str, list] = {}
    for r in records:
        key = r.entity_name.lower()
        if key not in entity_records:
            entity_records[key] = []
        entity_records[key].append(r)

    nodes = []
    for entity, neighbors in adj.items():
        recs = entity_records.get(entity, [])
        types = {}
        total_weight = 0.0
        for r in recs:
            t = r.memory_type.value
            types[t] = types.get(t, 0) + 1
            total_weight += r.weight
        avg_weight = total_weight / len(recs) if recs else 0.0
        cc = clustering_coefficient(adj, entity)
        nodes.append({
            "id": entity,
            "record_count": len(recs),
            "types": types,
            "avg_weight": round(avg_weight, 4),
            "clustering_coefficient": round(cc, 4),
            "related_entities": list(neighbors),
        })

    edge_set: set[tuple[str, str]] = set()
    for entity, neighbors in adj.items():
        for n in neighbors:
            pair = tuple(sorted([entity, n]))
            edge_set.add(pair)

    edges = [{"source": s, "target": t} for s, t in edge_set]

    return {"nodes": nodes, "edges": edges}


@router.get("/entity/{entity_name}")
def get_entity_detail(entity_name: str, mem: MemDep):
    """Single entity: records, clustering coefficient, neighbors."""
    records = mem._hot.get_all()
    if not records:
        return {"error": "No records in Hot RAM"}

    adj = build_entity_graph(records)
    key = entity_name.lower()

    recs = [r for r in records if r.entity_name.lower() == key]
    if not recs:
        return {"error": f"Entity '{entity_name}' not found"}

    neighbors = list(adj.get(key, set()))
    cc = clustering_coefficient(adj, key)

    return {
        "entity_name": entity_name,
        "records": [r.model_dump() for r in recs],
        "clustering_coefficient": round(cc, 4),
        "neighbors": neighbors,
        "degree": len(neighbors),
    }
