"use client";

import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { getGraph, getEntityDetail } from "@/lib/api";
import { useWS } from "@/components/layout/WebSocketProvider";
import { useTenant } from "@/components/TenantProvider";
import { IconX } from "@/components/icons";
import type { GraphData, GraphNode, EntityDetail } from "@/lib/types";

interface SimNode extends GraphNode, d3.SimulationNodeDatum {}
interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  source: string | SimNode;
  target: string | SimNode;
}

// Mirrors the TypeBadge palette on the Memory Explorer so the two pages agree.
const TYPE_COLORS: Record<string, string> = {
  CoreAnchor: "#b45309",
  ActiveContext: "#3b82f6",
  EphemeralState: "#52525b",
};

const TYPE_LABELS: Record<string, string> = {
  CoreAnchor: "Core Anchor",
  ActiveContext: "Active Context",
  EphemeralState: "Ephemeral",
};

function dominantType(types: Record<string, number>): string {
  let best = "EphemeralState";
  let count = 0;
  for (const [t, c] of Object.entries(types)) {
    if (c > count) { count = c; best = t; }
  }
  return best;
}

function nodeRadius(d: GraphNode): number {
  const degree = d.related_entities.length;
  return Math.min(26, 5 + Math.sqrt(degree) * 4);
}

export default function GraphPage() {
  const svgRef = useRef<SVGSVGElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const gRef = useRef<any>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const zoomRef = useRef<any>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const nodeSelRef = useRef<any>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const linkSelRef = useRef<any>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const labelSelRef = useRef<any>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const nodesRef = useRef<any[]>([]);

  const [data, setData] = useState<GraphData | null>(null);
  const [selected, setSelected] = useState<EntityDetail | null>(null);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [focusNode, setFocusNode] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [tooltip, setTooltip] = useState<{ x: number; y: number; node: SimNode } | null>(null);
  const { lastMessage } = useWS();
  const { tenantVersion } = useTenant();

  async function load() {
    try {
      setData(await getGraph());
    } catch { /* */ }
  }

  useEffect(() => { load(); }, [tenantVersion]);
  useEffect(() => { if (lastMessage) load(); }, [lastMessage]);

  function focusTransform(n: SimNode) {
    const svg = d3.select(svgRef.current);
    const zoom = zoomRef.current;
    if (!svg.node() || !zoom || n.x == null || n.y == null) return;
    const width = svgRef.current?.clientWidth ?? 600;
    const height = svgRef.current?.clientHeight ?? 400;
    const scale = 1.6;
    const t = d3.zoomIdentity
      .translate(width / 2 - n.x * scale, height / 2 - n.y * scale)
      .scale(scale);
    svg.call(zoom.transform, t);
  }

  // Build the simulation once per data load. Hover/focus styling is applied by
  // the effect below without rebuilding, so moving the mouse never restarts
  // the layout.
  useEffect(() => {
    if (!data || !svgRef.current || data.nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight || 500;

    svg.selectAll("*").remove();
    const g = svg.append("g");
    gRef.current = g.node();

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 5])
      .on("zoom", (e) => g.attr("transform", e.transform));
    svg.call(zoom);
    zoomRef.current = zoom;

    const simNodes: SimNode[] = data.nodes.map((n) => ({ ...n }));
    const simLinks: SimLink[] = data.edges.map((e) => ({ ...e }));
    nodesRef.current = simNodes;

    const simulation = d3
      .forceSimulation(simNodes)
      .force(
        "link",
        d3.forceLink<SimNode, SimLink>(simLinks)
          .id((d) => d.id)
          .distance(90)
      )
      .force("charge", d3.forceManyBody().strength(-250))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide<SimNode>().radius((d) => nodeRadius(d) + 10));

    const link = g
      .append("g")
      .selectAll("line")
      .data(simLinks)
      .join("line")
      .attr("stroke", "#3f3f46")
      .attr("stroke-width", 1)
      .attr("stroke-opacity", 0.6);
    linkSelRef.current = link;

    const node = g
      .append("g")
      .selectAll("circle")
      .data(simNodes)
      .join("circle")
      .attr("r", (d) => nodeRadius(d))
      .attr("fill", (d) => TYPE_COLORS[dominantType(d.types)] ?? TYPE_COLORS.EphemeralState)
      .attr("stroke", "#18181b")
      .attr("stroke-width", 1.5)
      .attr("cursor", "pointer")
      .on("click", async (_e, d) => {
        try {
          setSelected(await getEntityDetail(d.id));
        } catch { /* */ }
      })
      .on("mouseenter", (e, d) => {
        setHoveredNode(d.id);
        const rect = wrapRef.current?.getBoundingClientRect();
        setTooltip({
          x: e.clientX - (rect?.left ?? 0),
          y: e.clientY - (rect?.top ?? 0),
          node: d,
        });
      })
      .on("mousemove", (e) => {
        const rect = wrapRef.current?.getBoundingClientRect();
        setTooltip((t) =>
          t
            ? { ...t, x: e.clientX - (rect?.left ?? 0), y: e.clientY - (rect?.top ?? 0) }
            : t
        );
      })
      .on("mouseleave", () => {
        setHoveredNode(null);
        setTooltip(null);
      });
    nodeSelRef.current = node;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (node as any).call(
      d3.drag<SVGCircleElement, SimNode>()
        .on("start", (e, d) => {
          if (!e.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x; d.fy = d.y;
        })
        .on("drag", (e, d) => { d.fx = e.x; d.fy = e.y; })
        .on("end", (e, d) => {
          if (!e.active) simulation.alphaTarget(0);
          d.fx = null; d.fy = null;
        })
    );

    const labels = g
      .append("g")
      .selectAll("text")
      .data(simNodes)
      .join("text")
      .text((d) => d.id)
      .attr("font-size", 10)
      .attr("fill", "#a1a1aa")
      .attr("dx", 14)
      .attr("dy", 4);
    labelSelRef.current = labels;

    simulation.on("tick", () => {
      link
        .attr("x1", (d) => (d.source as SimNode).x!)
        .attr("y1", (d) => (d.source as SimNode).y!)
        .attr("x2", (d) => (d.target as SimNode).x!)
        .attr("y2", (d) => (d.target as SimNode).y!);
      node
        .attr("cx", (d) => d.x!)
        .attr("cy", (d) => d.y!);
      labels
        .attr("x", (d) => d.x!)
        .attr("y", (d) => d.y!);
    });

    simulation.on("end", () => {
      if (focusNode) {
        const n = nodesRef.current.find((n) => n.id === focusNode);
        if (n) focusTransform(n);
      }
    });

    return () => { simulation.stop(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  // Hover/focus emphasis: without a target everything is full opacity; with
  // one, the target and its direct neighbours stay, everything else fades.
  useEffect(() => {
    if (!data || !nodeSelRef.current) return;
    const active = hoveredNode ?? focusNode;
    const activeExists = active ? data.nodes.some((n) => n.id === active) : false;
    const relevant = (d: { id: string }) => {
      if (!active || !activeExists) return true;
      if (d.id === active) return true;
      return data.edges.some(
        (l) =>
          (l.source === active && l.target === d.id) ||
          (l.target === active && l.source === d.id)
      );
    };

    nodeSelRef.current.attr("opacity", (d: { id: string }) => (active ? (relevant(d) ? 1 : 0.15) : 1));
    labelSelRef.current?.attr("opacity", (d: { id: string }) => (active ? (relevant(d) ? 1 : 0.15) : 1));
    linkSelRef.current
      ?.attr("stroke-opacity", (d: SimLink) => {
        if (!active) return 0.6;
        const s = (d.source as SimNode).id;
        const t = (d.target as SimNode).id;
        return s === active || t === active ? 1 : 0.06;
      })
      .attr("stroke", (d: SimLink) => {
        const s = (d.source as SimNode).id;
        const t = (d.target as SimNode).id;
        return s === active || t === active ? "#818cf8" : "#3f3f46";
      });
  }, [data, hoveredNode, focusNode]);

  const matches =
    data && searchQuery.trim()
      ? data.nodes
          .filter((n) => n.id.toLowerCase().includes(searchQuery.toLowerCase()))
          .slice(0, 8)
      : [];

  function selectFocus(id: string) {
    setFocusNode(id);
    setSearchQuery("");
    // Wait for the layout to be roughly settled before centring on it.
    window.setTimeout(() => {
      const n = nodesRef.current.find((node) => node.id === id);
      if (n) focusTransform(n);
    }, 400);
  }

  return (
    <div className="p-6 h-full flex flex-col">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-zinc-100">Entity Graph</h1>
          <p className="text-sm text-zinc-500 mt-1">
            {data ? `${data.nodes.length} nodes, ${data.edges.length} edges` : "Loading..."}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && matches[0]) selectFocus(matches[0].id);
            }}
            placeholder="Find an entity…"
            className="w-56 px-3 py-1.5 rounded-md bg-zinc-900 border border-zinc-700 text-sm text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-zinc-500"
          />
          {focusNode && (
            <button
              onClick={() => setFocusNode(null)}
              className="px-2 py-1.5 rounded-md bg-zinc-800 text-xs text-zinc-300 hover:bg-zinc-700"
            >
              Clear focus
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 flex gap-4 min-h-0">
        <div
          ref={wrapRef}
          className="relative flex-1 rounded-lg border border-zinc-800 bg-zinc-900/50 overflow-hidden"
        >
          {data && data.nodes.length > 0 ? (
            <svg ref={svgRef} className="w-full h-full" />
          ) : (
            <div className="flex items-center justify-center h-full text-zinc-600">
              No entities to display
            </div>
          )}

          {searchQuery && matches.length > 0 && (
            <div className="absolute z-20 top-2 left-2 w-64 rounded-md border border-zinc-700 bg-zinc-900 shadow-lg overflow-hidden">
              {matches.map((n) => (
                <button
                  key={n.id}
                  onClick={() => selectFocus(n.id)}
                  className="block w-full text-left px-3 py-1.5 text-xs text-zinc-300 hover:bg-zinc-800 font-mono"
                >
                  {n.id}
                </button>
              ))}
            </div>
          )}

          {tooltip && (
            <div
              className="pointer-events-none absolute z-10 rounded-md border border-zinc-700 bg-zinc-900/95 px-3 py-2 text-xs shadow-lg max-w-64"
              style={{ left: tooltip.x + 12, top: tooltip.y + 12 }}
            >
              <p className="font-mono font-semibold text-zinc-100">{tooltip.node.id}</p>
              <div className="mt-1 space-y-0.5 text-zinc-400">
                <p>
                  {tooltip.node.record_count} records ·{" "}
                  {tooltip.node.related_entities.length} connections
                </p>
                <p>
                  <span
                    className="mr-1 inline-block h-2 w-2 rounded-full"
                    style={{
                      backgroundColor:
                        TYPE_COLORS[dominantType(tooltip.node.types)] ??
                        TYPE_COLORS.EphemeralState,
                    }}
                  />
                  {TYPE_LABELS[dominantType(tooltip.node.types)] ?? dominantType(tooltip.node.types)}
                </p>
                <p>avg weight {tooltip.node.avg_weight.toFixed(2)}</p>
              </div>
            </div>
          )}
        </div>

        {selected && (
          <div className="w-72 rounded-lg border border-zinc-800 bg-zinc-900 p-4 space-y-3 shrink-0">
            <div className="flex justify-between items-start">
              <h3 className="font-mono font-bold text-zinc-100">{selected.entity_name}</h3>
              <button onClick={() => setSelected(null)} className="text-zinc-600 hover:text-zinc-300">
                <IconX className="w-4 h-4" />
              </button>
            </div>
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div>
                <span className="text-zinc-500">Records</span>
                <p className="font-mono text-zinc-200">{selected.records.length}</p>
              </div>
              <div>
                <span className="text-zinc-500">Degree</span>
                <p className="font-mono text-zinc-200">{selected.degree}</p>
              </div>
              <div>
                <span className="text-zinc-500">Clustering Coeff</span>
                <p className="font-mono text-zinc-200">{selected.clustering_coefficient}</p>
              </div>
            </div>
            <div>
              <span className="text-xs text-zinc-500">Neighbors</span>
              <div className="flex flex-wrap gap-1 mt-1">
                {selected.neighbors.map((n) => (
                  <span key={n} className="px-2 py-0.5 rounded bg-zinc-800 text-xs text-zinc-400 border border-zinc-700">
                    {n}
                  </span>
                ))}
              </div>
            </div>
            <div>
              <span className="text-xs text-zinc-500">Facts</span>
              <div className="mt-1 space-y-2 max-h-48 overflow-auto">
                {selected.records.map((r) => (
                  <div key={r.id} className="text-xs text-zinc-400 border-l-2 border-zinc-700 pl-2">
                    {r.fact_content}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="mt-2 flex flex-wrap items-center gap-x-5 gap-y-1 text-xs text-zinc-500">
        <span className="text-zinc-400 font-medium">Legend</span>
        {Object.entries(TYPE_LABELS).map(([t, label]) => (
          <span key={t} className="inline-flex items-center gap-1.5">
            <span
              className="h-2.5 w-2.5 rounded-full"
              style={{ backgroundColor: TYPE_COLORS[t] }}
            />
            {label}
          </span>
        ))}
        <span className="text-zinc-600">Node size = connections</span>
      </div>
      <div className="mt-1 text-xs text-zinc-600">
        Click a node for its facts · Hover to trace connections · Drag to rearrange · Scroll to zoom
      </div>
    </div>
  );
}