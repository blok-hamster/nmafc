export type MemoryType = "CoreAnchor" | "ActiveContext" | "EphemeralState";

export type EventType =
  | "weight_update"
  | "override"
  | "suppression"
  | "prune"
  | "consolidation"
  | "ltp"
  | "retrieval";

export interface MemoryRecord {
  id: string;
  entity_name: string;
  fact_content: string;
  memory_type: MemoryType;
  weight: number;
  consolidation_index: number;
  created_at_turn: number;
  last_reinforced_turn: number;
  is_active: boolean;
  related_entities: string[];
}

export interface MemoryEvent {
  event_type: EventType;
  agent_id: string;
  conversation_id: string;
  turn: number;
  timestamp: string;
  record_id: string;
  entity_name: string;
  old_weight: number | null;
  new_weight: number | null;
  old_memory_type: string | null;
  new_memory_type: string | null;
  suppressed_by: string | null;
  old_k: number | null;
  new_k: number | null;
  retrieval_score: number | null;
  hops: number | null;
  metadata: Record<string, unknown>;
}

export interface GraphNode {
  id: string;
  record_count: number;
  types: Record<string, number>;
  avg_weight: number;
  clustering_coefficient: number;
  related_entities: string[];
}

export interface GraphEdge {
  source: string;
  target: string;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface DecayPoint {
  turn: number;
  weight: number;
}

export interface DecayCurve {
  record_id: string;
  entity_name: string;
  memory_type: string;
  current_weight: number;
  consolidation_index: number;
  points: DecayPoint[];
}

export interface DecayCurvesResponse {
  current_turn: number;
  prune_threshold: number;
  curves: DecayCurve[];
}

export interface HotStats {
  count: number;
  avg_weight: number;
  types: Record<string, number>;
}

export interface ColdStats {
  total_events: number;
  active_events: number;
}

export interface EventStats {
  total_events: number;
  by_type: Record<string, number>;
}

export interface AllStats {
  hot: HotStats;
  cold: ColdStats;
  events: EventStats;
  current_turn: number;
}

export interface SearchResult {
  record: MemoryRecord;
  score: number;
  hops: number;
}

export interface TimelinePoint {
  turn: number;
  event_type: string;
  count: number;
}

export interface DecayParams {
  lambda_core_anchor: number;
  lambda_active_context: number;
  lambda_ephemeral: number;
  eta: number;
  gamma: number;
  w_prune: number;
  theta: number;
  top_k: number;
  fallback_keyword_limit: number;
  max_hops: number;
  cold_semantic_fallback: boolean;
  beta: number;
  auto_consolidate_turns: number;
}

export interface EntityDetail {
  entity_name: string;
  records: MemoryRecord[];
  clustering_coefficient: number;
  neighbors: string[];
  degree: number;
}

export interface WSMessage {
  type: string;
  turn?: number;
  response?: string;
  stats?: HotStats;
  count?: number;
  restored?: number;
}
