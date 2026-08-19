import type {
  AllStats,
  DecayCurvesResponse,
  DecayParams,
  EntityDetail,
  GraphData,
  MemoryEvent,
  MemoryRecord,
  SearchResult,
  WSMessage,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

/* ── Tenant header state (set by TenantProvider) ── */

let _tenantHeaders: () => Record<string, string> = () => ({});

export function setTenantHeaderProvider(fn: () => Record<string, string>) {
  _tenantHeaders = fn;
}

async function fetchJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {
    ..._tenantHeaders(),
    ...((init?.headers as Record<string, string>) || {}),
  };
  const res = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

/* ── Health / Config ── */

export const health = () => fetchJSON<{ status: string }>("/api/health");
export const getConfig = () => fetchJSON<Record<string, unknown>>("/api/config");
export const getStats = () => fetchJSON<AllStats>("/api/stats");

/* ── Memory ── */

export const getAllRecords = () =>
  fetchJSON<MemoryRecord[]>("/api/memory/all");
export const getMutableRecords = () =>
  fetchJSON<MemoryRecord[]>("/api/memory/mutable");
export const getRecord = (id: string) =>
  fetchJSON<MemoryRecord>(`/api/memory/${id}`);
export const getEntityRecords = (name: string) =>
  fetchJSON<MemoryRecord[]>(`/api/memory/entity/${encodeURIComponent(name)}`);
export const searchMemory = (q: string, topK = 10) =>
  fetchJSON<SearchResult[]>(
    `/api/memory/search?q=${encodeURIComponent(q)}&top_k=${topK}`
  );

/* ── Graph ── */

export const getGraph = () => fetchJSON<GraphData>("/api/graph");
export const getEntityDetail = (name: string) =>
  fetchJSON<EntityDetail>(`/api/graph/entity/${encodeURIComponent(name)}`);

/* ── Events ── */

export const getEvents = (params?: {
  turn_from?: number;
  turn_to?: number;
  event_type?: string;
  entity_name?: string;
  limit?: number;
  offset?: number;
}) => {
  const sp = new URLSearchParams();
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null) sp.set(k, String(v));
    }
  }
  const qs = sp.toString();
  return fetchJSON<MemoryEvent[]>(`/api/events${qs ? `?${qs}` : ""}`);
};
export const getTimeline = (limit = 100) =>
  fetchJSON<{ turn: number; event_type: string; count: number }[]>(
    `/api/events/timeline?limit=${limit}`
  );
export const getEventStats = () =>
  fetchJSON<{ total_events: number; by_type: Record<string, number> }>(
    "/api/events/stats"
  );
export const getOverrideEvents = (limit = 50) =>
  fetchJSON<MemoryEvent[]>(`/api/events/overrides?limit=${limit}`);
export const getConsolidationEvents = (limit = 50) =>
  fetchJSON<MemoryEvent[]>(`/api/events/consolidations?limit=${limit}`);
export const getPruneEvents = (limit = 50) =>
  fetchJSON<MemoryEvent[]>(`/api/events/prunes?limit=${limit}`);
export const getLTPEvents = (limit = 50) =>
  fetchJSON<MemoryEvent[]>(`/api/events/ltp?limit=${limit}`);
export const getEntityHistory = (name: string, limit = 50) =>
  fetchJSON<MemoryEvent[]>(
    `/api/events/entity/${encodeURIComponent(name)}?limit=${limit}`
  );

/* ── Decay ── */

export const getDecayCurves = (turnsAhead = 50) =>
  fetchJSON<DecayCurvesResponse>(`/api/decay/curves?turns_ahead=${turnsAhead}`);
export const getDecayParams = () => fetchJSON<DecayParams>("/api/decay/params");

/* ── Process ── */

export const processTurn = (userMsg: string) =>
  fetchJSON<{ response: string; turn: number }>("/api/process", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_msg: userMsg }),
  });

export const ingestUpdates = (
  updates: {
    entity_name: string;
    fact_content: string;
    memory_type: string;
    related_entities?: string[];
  }[]
) =>
  fetchJSON<{ status: string; turn: number }>("/api/ingest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ updates }),
  });

export const consolidate = () =>
  fetchJSON<{ consolidated: number }>("/api/consolidate", { method: "POST" });

export const rollback = (toTurn: number) =>
  fetchJSON<{ restored: number; current_turn: number }>("/api/rollback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ to_turn: toTurn }),
  });

/* ── WebSocket ── */

export function createWebSocket(
  onMessage: (msg: WSMessage) => void,
  agentId = "default",
  conversationId = "default",
): WebSocket {
  let wsUrl: string;
  const params = new URLSearchParams({
    agent_id: agentId,
    conversation_id: conversationId,
  });
  if (API_BASE) {
    const wsBase = API_BASE.replace(/^http/, "ws");
    wsUrl = `${wsBase}/ws/live?${params}`;
  } else {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    wsUrl = `${proto}//${window.location.host}/ws/live?${params}`;
  }
  const ws = new WebSocket(wsUrl);
  ws.onmessage = (e) => {
    try {
      onMessage(JSON.parse(e.data));
    } catch {
      // ignore malformed
    }
  };
  return ws;
}
