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

export default function GraphPage() {
  const svgRef = useRef<SVGSVGElement>(null);
  const [data, setData] = useState<GraphData | null>(null);
  const [selected, setSelected] = useState<EntityDetail | null>(null);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const { lastMessage } = useWS();
  const { tenantVersion } = useTenant();

  async function load() {
    try {
      setData(await getGraph());
    } catch { /* */ }
  }

  useEffect(() => { load(); }, [tenantVersion]);
  useEffect(() => { if (lastMessage) load(); }, [lastMessage]);

  useEffect(() => {
    if (!data || !svgRef.current || data.nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight || 500;

    svg.selectAll("*").remove();

    const g = svg.append("g");

    svg.call(
      d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.1, 5])
        .on("zoom", (e) => g.attr("transform", e.transform))
    );

    const simNodes: SimNode[] = data.nodes.map((n) => ({ ...n }));
    const simLinks: SimLink[] = data.edges.map((e) => ({ ...e }));

    const simulation = d3
      .forceSimulation(simNodes)
      .force(
        "link",
        d3.forceLink<SimNode, SimLink>(simLinks).id((d) => d.id).distance(100)
      )
      .force("charge", d3.forceManyBody().strength(-200))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(30));

    const weightColor = d3.scaleSequential(d3.interpolateViridis).domain([0, 1]);

    const link = g
      .append("g")
      .selectAll("line")
      .data(simLinks)
      .join("line")
      .attr("stroke", "#3f3f46")
      .attr("stroke-width", 1)
      .attr("stroke-opacity", 0.6);

    const node = g
      .append("g")
      .selectAll("circle")
      .data(simNodes)
      .join("circle")
      .attr("r", (d) => 8 + d.record_count * 2)
      .attr("fill", (d) => weightColor(d.avg_weight))
      .attr("stroke", "#18181b")
      .attr("stroke-width", 1.5)
      .attr("cursor", "pointer")
      .on("click", async (_e, d) => {
        try {
          setSelected(await getEntityDetail(d.id));
        } catch { /* */ }
      })
      .on("mouseenter", (_e, d) => setHoveredNode(d.id))
      .on("mouseleave", () => setHoveredNode(null));

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

    node.attr("opacity", (d) => {
      if (!hoveredNode) return 1;
      if (d.id === hoveredNode) return 1;
      const connected = simLinks.some(
        (l) =>
          (l.source as SimNode).id === hoveredNode && (l.target as SimNode).id === d.id ||
          (l.target as SimNode).id === hoveredNode && (l.source as SimNode).id === d.id
      );
      return connected ? 1 : 0.2;
    });

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

    return () => { simulation.stop(); };
  }, [data, hoveredNode]);

  return (
    <div className="p-6 h-full flex flex-col">
      <div className="mb-4">
        <h1 className="text-xl font-bold text-zinc-100">Entity Graph</h1>
        <p className="text-sm text-zinc-500 mt-1">
          {data ? `${data.nodes.length} nodes, ${data.edges.length} edges` : "Loading..."}
        </p>
      </div>

      <div className="flex-1 flex gap-4 min-h-0">
        <div className="flex-1 rounded-lg border border-zinc-800 bg-zinc-900/50 overflow-hidden">
          {data && data.nodes.length > 0 ? (
            <svg ref={svgRef} className="w-full h-full" />
          ) : (
            <div className="flex items-center justify-center h-full text-zinc-600">
              No entities to display
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

      <div className="mt-2 flex items-center gap-4 text-xs text-zinc-500">
        <span>Node size = record count</span>
        <span>Color = avg weight (viridis)</span>
        <span>Drag to rearrange, scroll to zoom, click for details</span>
      </div>
    </div>
  );
}
