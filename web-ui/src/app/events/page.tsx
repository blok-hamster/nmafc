"use client";

import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { getEvents, getEventStats, getTimeline } from "@/lib/api";
import { useWS } from "@/components/layout/WebSocketProvider";
import { useTenant } from "@/components/TenantProvider";
import type { MemoryEvent } from "@/lib/types";

const EVENT_COLORS: Record<string, string> = {
  weight_update: "#3b82f6",
  override: "#ef4444",
  suppression: "#f59e0b",
  prune: "#dc2626",
  consolidation: "#22c55e",
  ltp: "#8b5cf6",
  retrieval: "#06b6d4",
};

type Tab = "all" | "timeline" | "overrides" | "consolidations" | "prunes" | "ltp";

export default function EventsPage() {
  const [tab, setTab] = useState<Tab>("all");
  const [events, setEvents] = useState<MemoryEvent[]>([]);
  const [stats, setStats] = useState<{ total_events: number; by_type: Record<string, number> } | null>(null);
  const [timelineData, setTimelineData] = useState<{ turn: number; event_type: string; count: number }[]>([]);
  const [filterEntity, setFilterEntity] = useState("");
  const { lastMessage } = useWS();
  const { tenantVersion } = useTenant();

  async function load() {
    try {
      const [s, t] = await Promise.all([getEventStats(), getTimeline()]);
      setStats(s);
      setTimelineData(t);
    } catch { /* */ }
  }

  async function loadTab() {
    try {
      const params: Record<string, unknown> = { limit: 200 };
      if (filterEntity) params.entity_name = filterEntity;

      let data: MemoryEvent[];
      switch (tab) {
        case "overrides":
          data = await getEvents({ ...params, event_type: "suppression" } as Parameters<typeof getEvents>[0]);
          break;
        case "consolidations":
          data = await getEvents({ ...params, event_type: "consolidation" } as Parameters<typeof getEvents>[0]);
          break;
        case "prunes":
          data = await getEvents({ ...params, event_type: "prune" } as Parameters<typeof getEvents>[0]);
          break;
        case "ltp":
          data = await getEvents({ ...params, event_type: "ltp" } as Parameters<typeof getEvents>[0]);
          break;
        default:
          data = await getEvents(params as Parameters<typeof getEvents>[0]);
      }
      setEvents(data);
    } catch { /* */ }
  }

  useEffect(() => { load(); }, [tenantVersion]);
  useEffect(() => { loadTab(); }, [tab, filterEntity, tenantVersion]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (lastMessage) { load(); loadTab(); } }, [lastMessage]); // eslint-disable-line react-hooks/exhaustive-deps

  const chartData = (() => {
    if (timelineData.length === 0) return [];
    const turns = [...new Set(timelineData.map((d) => d.turn))].sort((a, b) => a - b);
    const types = [...new Set(timelineData.map((d) => d.event_type))];
    return turns.map((turn) => {
      const row: Record<string, number> = { turn };
      for (const t of types) {
        const match = timelineData.find((d) => d.turn === turn && d.event_type === t);
        row[t] = match?.count ?? 0;
      }
      return row;
    });
  })();

  const chartTypes = [...new Set(timelineData.map((d) => d.event_type))];

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-xl font-bold text-zinc-100">Event Timeline</h1>
        <p className="text-sm text-zinc-500 mt-1">
          {stats?.total_events ?? 0} cognitive events logged
        </p>
      </div>

      {stats && (
        <div className="grid grid-cols-3 lg:grid-cols-7 gap-3">
          {Object.entries(EVENT_COLORS).map(([type, color]) => (
            <div
              key={type}
              className="rounded border border-zinc-800 bg-zinc-900/50 px-3 py-2"
            >
              <div className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
                <span className="text-[10px] text-zinc-500 uppercase">{type.replace("_", " ")}</span>
              </div>
              <p className="text-lg font-mono text-zinc-200 mt-1">
                {stats.by_type[type] ?? 0}
              </p>
            </div>
          ))}
        </div>
      )}

      {chartData.length > 0 && (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
          <h2 className="text-sm font-semibold text-zinc-300 mb-3">Events per Turn</h2>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis dataKey="turn" stroke="#52525b" fontSize={11} tick={{ fill: "#71717a" }} />
              <YAxis stroke="#52525b" fontSize={11} tick={{ fill: "#71717a" }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#18181b",
                  border: "1px solid #3f3f46",
                  borderRadius: 6,
                  fontSize: 12,
                }}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {chartTypes.map((type) => (
                <Bar
                  key={type}
                  dataKey={type}
                  stackId="a"
                  fill={EVENT_COLORS[type] || "#71717a"}
                  name={type.replace("_", " ")}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="flex items-center gap-4">
        <div className="flex gap-1 bg-zinc-900 rounded-lg p-1">
          {(["all", "timeline", "overrides", "consolidations", "prunes", "ltp"] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`px-3 py-1.5 rounded text-xs capitalize transition-colors ${
                tab === t
                  ? "bg-zinc-700 text-zinc-100"
                  : "text-zinc-500 hover:text-zinc-300"
              }`}
            >
              {t}
            </button>
          ))}
        </div>
        <input
          type="text"
          placeholder="Filter by entity..."
          value={filterEntity}
          onChange={(e) => setFilterEntity(e.target.value)}
          className="px-3 py-1.5 rounded-md bg-zinc-900 border border-zinc-700 text-xs text-zinc-300 placeholder:text-zinc-600 focus:outline-none focus:border-zinc-500"
        />
      </div>

      <div className="rounded-lg border border-zinc-800 overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-zinc-900 border-b border-zinc-800">
              <th className="px-4 py-2.5 text-left text-xs font-medium text-zinc-400 uppercase">Turn</th>
              <th className="px-4 py-2.5 text-left text-xs font-medium text-zinc-400 uppercase">Type</th>
              <th className="px-4 py-2.5 text-left text-xs font-medium text-zinc-400 uppercase">Entity</th>
              <th className="px-4 py-2.5 text-left text-xs font-medium text-zinc-400 uppercase">Weight</th>
              <th className="px-4 py-2.5 text-left text-xs font-medium text-zinc-400 uppercase">Details</th>
            </tr>
          </thead>
          <tbody>
            {events.map((ev, i) => (
              <tr key={i} className="border-b border-zinc-800/50 hover:bg-zinc-900/50">
                <td className="px-4 py-2 font-mono text-zinc-300">{ev.turn}</td>
                <td className="px-4 py-2">
                  <span
                    className="inline-block px-2 py-0.5 rounded text-xs border"
                    style={{
                      color: EVENT_COLORS[ev.event_type] || "#a1a1aa",
                      borderColor: (EVENT_COLORS[ev.event_type] || "#52525b") + "40",
                      backgroundColor: (EVENT_COLORS[ev.event_type] || "#52525b") + "15",
                    }}
                  >
                    {ev.event_type}
                  </span>
                </td>
                <td className="px-4 py-2 font-mono text-zinc-400 text-xs">{ev.entity_name}</td>
                <td className="px-4 py-2 font-mono text-zinc-400 text-xs">
                  {ev.old_weight !== null && ev.new_weight !== null
                    ? `${ev.old_weight.toFixed(3)} -> ${ev.new_weight.toFixed(3)}`
                    : "-"}
                </td>
                <td className="px-4 py-2 text-zinc-500 text-xs">
                  {ev.retrieval_score !== null && `score: ${ev.retrieval_score.toFixed(3)}`}
                  {ev.hops !== null && ev.hops > 0 && ` hops: ${ev.hops}`}
                  {ev.suppressed_by && ` by: ${ev.suppressed_by}`}
                  {ev.old_memory_type && ev.new_memory_type && `${ev.old_memory_type} -> ${ev.new_memory_type}`}
                </td>
              </tr>
            ))}
            {events.length === 0 && (
              <tr>
                <td colSpan={5} className="px-4 py-8 text-center text-zinc-600">
                  No events found
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
