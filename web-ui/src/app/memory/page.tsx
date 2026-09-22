"use client";

import { useEffect, useState } from "react";
import { getAllRecords, getRecordSource, searchMemory } from "@/lib/api";
import { useWS } from "@/components/layout/WebSocketProvider";
import { useTenant } from "@/components/TenantProvider";
import { IconX, IconArrowUp, IconArrowDown } from "@/components/icons";
import type { MemoryRecord, RecordSource } from "@/lib/types";

type SortKey = "entity_name" | "memory_type" | "weight" | "consolidation_index" | "created_at_turn";

export default function MemoryPage() {
  const [records, setRecords] = useState<MemoryRecord[]>([]);
  const [sortKey, setSortKey] = useState<SortKey>("weight");
  const [sortAsc, setSortAsc] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<{ record: MemoryRecord; score: number; hops: number }[] | null>(null);
  const [selected, setSelected] = useState<MemoryRecord | null>(null);
  const { lastMessage } = useWS();
  const { tenantVersion } = useTenant();

  async function load() {
    try {
      setRecords(await getAllRecords());
    } catch { /* */ }
  }

  useEffect(() => { load(); }, [tenantVersion]);
  useEffect(() => { if (lastMessage) load(); }, [lastMessage]);

  async function handleSearch() {
    if (!searchQuery.trim()) {
      setSearchResults(null);
      return;
    }
    try {
      setSearchResults(await searchMemory(searchQuery));
    } catch { /* */ }
  }

  const sorted = [...records].sort((a, b) => {
    const av = a[sortKey];
    const bv = b[sortKey];
    if (typeof av === "string") return sortAsc ? av.localeCompare(bv as string) : (bv as string).localeCompare(av);
    return sortAsc ? (av as number) - (bv as number) : (bv as number) - (av as number);
  });

  const display = searchResults ? searchResults.map((r) => r.record) : sorted;

  function toggleSort(key: SortKey) {
    if (sortKey === key) setSortAsc(!sortAsc);
    else { setSortKey(key); setSortAsc(false); }
  }

  const sortIndicator = (key: SortKey) =>
    sortKey === key ? (sortAsc ? <IconArrowUp className="w-3 h-3 inline ml-1" /> : <IconArrowDown className="w-3 h-3 inline ml-1" />) : null;

  return (
    <div className="p-6 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-zinc-100">Memory Explorer</h1>
          <p className="text-sm text-zinc-500 mt-1">{records.length} records in Hot RAM</p>
        </div>
      </div>

      <div className="flex gap-2">
        <input
          type="text"
          placeholder="Search by meaning..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSearch()}
          className="flex-1 px-3 py-2 rounded-md bg-zinc-900 border border-zinc-700 text-sm text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-zinc-500"
        />
        <button
          onClick={handleSearch}
          className="px-4 py-2 rounded-md bg-zinc-800 text-sm text-zinc-300 hover:bg-zinc-700 transition-colors"
        >
          Search
        </button>
        {searchResults && (
          <button
            onClick={() => { setSearchResults(null); setSearchQuery(""); }}
            className="px-4 py-2 rounded-md bg-zinc-800 text-sm text-zinc-400 hover:bg-zinc-700"
          >
            Clear
          </button>
        )}
      </div>

      {searchResults && (
        <p className="text-xs text-zinc-500">
          {searchResults.length} results (vector similarity)
        </p>
      )}

      <div className="rounded-lg border border-zinc-800 overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-zinc-900 border-b border-zinc-800">
              {([
                ["entity_name", "Entity"],
                ["memory_type", "Type"],
                ["weight", "Weight"],
                ["consolidation_index", "Consolidation"],
                ["created_at_turn", "Created"],
              ] as const).map(([key, label]) => (
                <th
                  key={key}
                  onClick={() => toggleSort(key)}
                  className="px-4 py-2.5 text-left text-xs font-medium text-zinc-400 uppercase tracking-wider cursor-pointer hover:text-zinc-200 select-none"
                >
                  {label}
                  <span className="text-zinc-600">{sortIndicator(key)}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {display.map((rec) => (
              <tr
                key={rec.id}
                onClick={() => setSelected(rec)}
                className={`border-b border-zinc-800/50 cursor-pointer transition-colors ${
                  selected?.id === rec.id
                    ? "bg-zinc-800"
                    : "hover:bg-zinc-900/50"
                }`}
              >
                <td className="px-4 py-2.5 font-mono text-zinc-200">{rec.entity_name}</td>
                <td className="px-4 py-2.5">
                  <TypeBadge type={rec.memory_type} />
                </td>
                <td className="px-4 py-2.5 font-mono text-zinc-300">{rec.weight.toFixed(4)}</td>
                <td className="px-4 py-2.5 font-mono text-zinc-400">{rec.consolidation_index}</td>
                <td className="px-4 py-2.5 font-mono text-zinc-500">
                  {rec.created_at_turn}
                  {rec.created_date && (
                    <span className="block text-xs text-zinc-600">{rec.created_date}</span>
                  )}
                </td>
              </tr>
            ))}
            {display.length === 0 && (
              <tr>
                <td colSpan={5} className="px-4 py-8 text-center text-zinc-600">
                  No records found
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {selected && (
        <RecordDetail record={selected} onClose={() => setSelected(null)} />
      )}
    </div>
  );
}

function TypeBadge({ type }: { type: string }) {
  const colors: Record<string, string> = {
    CoreAnchor: "bg-amber-900/40 text-amber-300 border-amber-800",
    ActiveContext: "bg-blue-900/40 text-blue-300 border-blue-800",
    EphemeralState: "bg-zinc-800 text-zinc-400 border-zinc-700",
  };
  return (
    <span className={`inline-block px-2 py-0.5 rounded text-xs border ${colors[type] || colors.EphemeralState}`}>
      {type}
    </span>
  );
}

function RecordDetail({
  record,
  onClose,
}: {
  record: MemoryRecord;
  onClose: () => void;
}) {
  const [source, setSource] = useState<RecordSource | null>(null);
  const [sourceLoading, setSourceLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setSourceLoading(true);
    getRecordSource(record.id)
      .then((s) => { if (!cancelled) setSource(s); })
      .catch(() => { if (!cancelled) setSource(null); })
      .finally(() => { if (!cancelled) setSourceLoading(false); });
    return () => { cancelled = true; };
  }, [record.id]);

  const validity = source?.validity;

  const validLabel =
    validity?.valid_at_text ||
    (validity?.valid_date
      ? validity.valid_date
      : validity?.valid_at != null
        ? `Turn ${validity.valid_at}`
        : record.valid_at_text || `Turn ${record.valid_at ?? record.created_at_turn}`);

  const invalidLabel =
    validity?.invalid_date || (validity?.invalid_at != null
      ? `Turn ${validity.invalid_at}`
      : record.invalid_at != null
        ? `Turn ${record.invalid_at}`
        : null);

  const sourcedTurns = source?.source_turns.filter((t) => t.text) ?? [];

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4 space-y-3">
      <div className="flex justify-between items-start">
        <h3 className="font-mono font-bold text-zinc-100">{record.entity_name}</h3>
        <button onClick={onClose} className="text-zinc-600 hover:text-zinc-300">
          <IconX className="w-4 h-4" />
        </button>
      </div>
      <p className="text-sm text-zinc-300">{record.fact_content}</p>
      <div className="grid grid-cols-4 gap-4 text-xs">
        <div>
          <span className="text-zinc-500">Type</span>
          <p><TypeBadge type={record.memory_type} /></p>
        </div>
        <div>
          <span className="text-zinc-500">Weight</span>
          <p className="font-mono text-zinc-200">{record.weight.toFixed(4)}</p>
        </div>
        <div>
          <span className="text-zinc-500">Consolidation</span>
          <p className="font-mono text-zinc-200">{record.consolidation_index}</p>
        </div>
        <div>
          <span className="text-zinc-500">Created</span>
          <p className="font-mono text-zinc-200">
            Turn {record.created_at_turn}
            {record.created_date && (
              <span className="block text-zinc-400">{record.created_date}</span>
            )}
          </p>
        </div>
      </div>

      <div>
        <span className="text-xs text-zinc-500">Validity</span>
        <div className="mt-1 space-y-1 text-xs">
          <p className="font-mono text-zinc-200">
            Valid from {validLabel}
          </p>
          {invalidLabel && (
            <p className="font-mono text-zinc-400">Superseded {invalidLabel}</p>
          )}
          {validity?.has_dates === false && !record.valid_at_text && (
            <p className="text-zinc-600">
              Store has no calendar dates — backfill with{" "}
              <code className="text-zinc-500">_backfill_turn_dates.py</code> for dated facts.
            </p>
          )}
        </div>
      </div>

      <div>
        <span className="text-xs text-zinc-500">Source turns</span>
        <div className="mt-1 max-h-48 overflow-auto space-y-2">
          {sourceLoading ? (
            <p className="text-xs text-zinc-600">Loading source…</p>
          ) : sourcedTurns.length > 0 ? (
            sourcedTurns.map((t) => (
              <blockquote key={t.turn} className="border-l-2 border-zinc-600 pl-3 text-xs">
                <p className="font-mono text-zinc-400">
                  Turn {t.turn}
                  {t.timestamp && <span> · {t.timestamp}</span>}
                </p>
                <p className="text-zinc-300 mt-0.5 whitespace-pre-wrap">{t.text}</p>
              </blockquote>
            ))
          ) : (
            <p className="text-xs text-zinc-600">
              No source text stored — run{" "}
              <code className="text-zinc-500">_backfill_turn_text.py</code> to restore verbatim
              turns for this store.
            </p>
          )}
        </div>
      </div>

      {record.related_entities.length > 0 && (
        <div>
          <span className="text-xs text-zinc-500">Related Entities</span>
          <div className="flex flex-wrap gap-1 mt-1">
            {record.related_entities.map((e) => (
              <span key={e} className="px-2 py-0.5 rounded bg-zinc-800 text-xs text-zinc-400 border border-zinc-700">
                {e}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
