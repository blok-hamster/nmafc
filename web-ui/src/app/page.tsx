"use client";

import { useEffect, useState } from "react";
import { getStats } from "@/lib/api";
import { useWS } from "@/components/layout/WebSocketProvider";
import { useTenant } from "@/components/TenantProvider";
import type { AllStats } from "@/lib/types";

export default function DashboardPage() {
  const [stats, setStats] = useState<AllStats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { lastMessage } = useWS();
  const { tenantVersion } = useTenant();

  async function load() {
    try {
      setStats(await getStats());
      setError(null);
    } catch {
      setError("Cannot reach backend");
    }
  }

  useEffect(() => { load(); }, [tenantVersion]);
  useEffect(() => { if (lastMessage) load(); }, [lastMessage]);

  if (error) {
    return (
      <div className="p-8 text-center text-zinc-500">
        <p className="text-lg">{error}</p>
        <p className="text-sm mt-2">Start the backend with <code className="text-zinc-400">nmafc-web</code></p>
      </div>
    );
  }

  if (!stats) {
    return <div className="p-8 text-zinc-500">Loading...</div>;
  }

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-xl font-bold text-zinc-100">Dashboard</h1>
        <p className="text-sm text-zinc-500 mt-1">System overview at a glance</p>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Turn" value={stats.current_turn} />
        <StatCard label="Hot RAM Records" value={stats.hot.count} />
        <StatCard label="Cold ROM Events" value={stats.cold.total_events} />
        <StatCard label="Cognitive Events" value={stats.events.total_events} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Panel title="Hot RAM by Type">
          {Object.keys(stats.hot.types).length === 0 ? (
            <p className="text-zinc-600 text-sm">No records</p>
          ) : (
            <div className="space-y-2">
              {Object.entries(stats.hot.types).map(([type, count]) => (
                <TypeBar key={type} label={type} count={count} total={stats.hot.count} />
              ))}
            </div>
          )}
        </Panel>

        <Panel title="Events by Type">
          {Object.keys(stats.events.by_type).length === 0 ? (
            <p className="text-zinc-600 text-sm">No events</p>
          ) : (
            <div className="space-y-2">
              {Object.entries(stats.events.by_type).map(([type, count]) => (
                <TypeBar key={type} label={type} count={count} total={stats.events.total_events} />
              ))}
            </div>
          )}
        </Panel>
      </div>

      <Panel title="Hot RAM Health">
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <p className="text-2xl font-mono text-zinc-100">
              {stats.hot.avg_weight.toFixed(3)}
            </p>
            <p className="text-xs text-zinc-500 mt-1">Avg Weight</p>
          </div>
          <div>
            <p className="text-2xl font-mono text-zinc-100">
              {stats.cold.active_events}
            </p>
            <p className="text-xs text-zinc-500 mt-1">Active Cold Events</p>
          </div>
          <div>
            <p className="text-2xl font-mono text-zinc-100">
              {stats.hot.count > 0 ? (stats.hot.avg_weight * 100).toFixed(0) + "%" : "-"}
            </p>
            <p className="text-xs text-zinc-500 mt-1">Avg Retention</p>
          </div>
        </div>
      </Panel>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
      <p className="text-xs text-zinc-500 uppercase tracking-wider">{label}</p>
      <p className="text-2xl font-mono font-bold text-zinc-100 mt-1">{value}</p>
    </div>
  );
}

function Panel({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
      <h2 className="text-sm font-semibold text-zinc-300 mb-3">{title}</h2>
      {children}
    </div>
  );
}

function TypeBar({
  label,
  count,
  total,
}: {
  label: string;
  count: number;
  total: number;
}) {
  const pct = total > 0 ? (count / total) * 100 : 0;
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-zinc-400">{label}</span>
        <span className="text-zinc-500 font-mono">
          {count} ({pct.toFixed(0)}%)
        </span>
      </div>
      <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
        <div
          className="h-full bg-zinc-500 rounded-full transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
