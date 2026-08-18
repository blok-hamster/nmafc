"use client";

import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { getDecayCurves, getDecayParams } from "@/lib/api";
import { useWS } from "@/components/layout/WebSocketProvider";
import { useTenant } from "@/components/TenantProvider";
import type { DecayCurvesResponse, DecayParams } from "@/lib/types";

const COLORS = [
  "#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#8b5cf6",
  "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#6366f1",
];

export default function DecayPage() {
  const [curves, setCurves] = useState<DecayCurvesResponse | null>(null);
  const [params, setParams] = useState<DecayParams | null>(null);
  const [turnsAhead, setTurnsAhead] = useState(50);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const { lastMessage } = useWS();
  const { tenantVersion } = useTenant();

  async function load() {
    try {
      const [c, p] = await Promise.all([getDecayCurves(turnsAhead), getDecayParams()]);
      setCurves(c);
      setParams(p);
    } catch { /* */ }
  }

  useEffect(() => { load(); }, [turnsAhead, tenantVersion]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (lastMessage) load(); }, [lastMessage]); // eslint-disable-line react-hooks/exhaustive-deps

  const displayed = curves
    ? selectedIds.size > 0
      ? curves.curves.filter((c) => selectedIds.has(c.record_id))
      : curves.curves
    : [];

  // Merge all curves into a single data array for recharts
  const chartData = (() => {
    if (!curves || displayed.length === 0) return [];
    const maxLen = Math.max(...displayed.map((c) => c.points.length));
    const rows = [];
    for (let i = 0; i < maxLen; i++) {
      const row: Record<string, number> = { turn: displayed[0]?.points[i]?.turn ?? i };
      for (const curve of displayed) {
        row[curve.entity_name] = curve.points[i]?.weight ?? 0;
      }
      rows.push(row);
    }
    return rows;
  })();

  function toggleRecord(id: string) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-xl font-bold text-zinc-100">Decay Curves</h1>
        <p className="text-sm text-zinc-500 mt-1">
          Projected weight trajectories for {curves?.curves.length ?? 0} mutable records
        </p>
      </div>

      {params && (
        <div className="grid grid-cols-4 lg:grid-cols-8 gap-3">
          {([
            ["λ Core", params.lambda_core_anchor],
            ["λ Active", params.lambda_active_context],
            ["λ Ephemeral", params.lambda_ephemeral],
            ["η", params.eta],
            ["γ", params.gamma],
            ["w_prune", params.w_prune],
            ["θ", params.theta],
            ["β", params.beta],
          ] as const).map(([label, value]) => (
            <div key={label} className="rounded border border-zinc-800 bg-zinc-900/50 px-3 py-2">
              <p className="text-[10px] text-zinc-500 uppercase">{label}</p>
              <p className="text-sm font-mono text-zinc-200">{value}</p>
            </div>
          ))}
        </div>
      )}

      <div className="flex items-center gap-3">
        <label className="text-xs text-zinc-500">Turns ahead:</label>
        <input
          type="range"
          min={10}
          max={200}
          value={turnsAhead}
          onChange={(e) => setTurnsAhead(Number(e.target.value))}
          className="w-40"
        />
        <span className="text-xs font-mono text-zinc-400">{turnsAhead}</span>
      </div>

      <div className="flex gap-4 min-h-0">
        {/* Chart */}
        <div className="flex-1 rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                <XAxis
                  dataKey="turn"
                  stroke="#52525b"
                  fontSize={11}
                  tick={{ fill: "#71717a" }}
                />
                <YAxis
                  domain={[0, 1]}
                  stroke="#52525b"
                  fontSize={11}
                  tick={{ fill: "#71717a" }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#18181b",
                    border: "1px solid #3f3f46",
                    borderRadius: 6,
                    fontSize: 12,
                  }}
                  labelStyle={{ color: "#a1a1aa" }}
                />
                <Legend
                  wrapperStyle={{ fontSize: 11, color: "#a1a1aa" }}
                />
                <ReferenceLine
                  y={curves?.prune_threshold ?? 0.1}
                  stroke="#ef4444"
                  strokeDasharray="6 3"
                  label={{
                    value: "prune threshold",
                    position: "right",
                    fill: "#ef4444",
                    fontSize: 10,
                  }}
                />
                {displayed.map((curve, i) => (
                  <Line
                    key={curve.record_id}
                    type="monotone"
                    dataKey={curve.entity_name}
                    stroke={COLORS[i % COLORS.length]}
                    strokeWidth={2}
                    dot={false}
                    name={`${curve.entity_name} (${curve.memory_type})`}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-[400px] text-zinc-600">
              No decay curves to display
            </div>
          )}
        </div>

        {/* Legend / selector */}
        <div className="w-56 shrink-0 rounded-lg border border-zinc-800 bg-zinc-900 p-3 space-y-1 max-h-[450px] overflow-auto">
          <p className="text-xs text-zinc-500 uppercase mb-2">Records</p>
          {curves?.curves.map((c, i) => (
            <button
              key={c.record_id}
              onClick={() => toggleRecord(c.record_id)}
              className={`w-full text-left px-2 py-1.5 rounded text-xs flex items-center gap-2 transition-colors ${
                selectedIds.size === 0 || selectedIds.has(c.record_id)
                  ? "bg-zinc-800/50 text-zinc-300"
                  : "text-zinc-600 hover:text-zinc-400"
              }`}
            >
              <span
                className="h-2 w-2 rounded-full shrink-0"
                style={{
                  backgroundColor:
                    selectedIds.size === 0 || selectedIds.has(c.record_id)
                      ? COLORS[i % COLORS.length]
                      : "#52525b",
                }}
              />
              <span className="truncate font-mono">{c.entity_name}</span>
              <span className="ml-auto text-zinc-600">{c.memory_type.slice(0, 3)}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
