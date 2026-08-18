"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { useTenant } from "@/components/TenantProvider";
import { useWS } from "@/components/layout/WebSocketProvider";
import {
  IconDashboard,
  IconMemory,
  IconGraph,
  IconDecay,
  IconEvents,
  IconDocs,
} from "@/components/icons";

const NAV = [
  { href: "/", label: "Dashboard", Icon: IconDashboard },
  { href: "/memory", label: "Memory Explorer", Icon: IconMemory },
  { href: "/graph", label: "Entity Graph", Icon: IconGraph },
  { href: "/decay", label: "Decay Curves", Icon: IconDecay },
  { href: "/events", label: "Event Timeline", Icon: IconEvents },
  { href: "/docs", label: "Documentation", Icon: IconDocs },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-56 shrink-0 border-r border-zinc-800 bg-zinc-950 flex flex-col">
      <div className="px-4 py-5 border-b border-zinc-800">
        <h1 className="text-sm font-bold tracking-widest text-zinc-100 uppercase">
          NMAFC
        </h1>
        <p className="text-[10px] text-zinc-500 mt-0.5">Memory Explorer</p>
      </div>
      <nav className="flex-1 py-2">
        {NAV.map(({ href, label, Icon }) => {
          const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={`flex items-center gap-3 px-4 py-2.5 text-sm transition-colors ${
                active
                  ? "bg-zinc-800 text-zinc-100"
                  : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900"
              }`}
            >
              <Icon className="w-4 h-4 shrink-0" />
              {label}
            </Link>
          );
        })}
      </nav>
      <div className="px-4 py-3 border-t border-zinc-800 space-y-3">
        <TenantSelector />
        <StatusIndicator />
      </div>
    </aside>
  );
}

function TenantSelector() {
  const { agentId, conversationId, setAgentId, setConversationId } = useTenant();
  const [agentInput, setAgentInput] = useState(agentId);
  const [convInput, setConvInput] = useState(conversationId);

  function apply() {
    setAgentId(agentInput.trim() || "default");
    setConversationId(convInput.trim() || "default");
  }

  return (
    <div className="space-y-2">
      <div>
        <label className="text-[10px] text-zinc-600 uppercase tracking-wider">Agent</label>
        <input
          type="text"
          value={agentInput}
          onChange={(e) => setAgentInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && apply()}
          className="w-full mt-0.5 px-2 py-1 rounded bg-zinc-900 border border-zinc-800 text-xs text-zinc-300 placeholder:text-zinc-600 focus:outline-none focus:border-zinc-600"
        />
      </div>
      <div>
        <label className="text-[10px] text-zinc-600 uppercase tracking-wider">Conversation</label>
        <input
          type="text"
          value={convInput}
          onChange={(e) => setConvInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && apply()}
          className="w-full mt-0.5 px-2 py-1 rounded bg-zinc-900 border border-zinc-800 text-xs text-zinc-300 placeholder:text-zinc-600 focus:outline-none focus:border-zinc-600"
        />
      </div>
      <button
        onClick={apply}
        className="w-full px-2 py-1 rounded bg-zinc-800 text-xs text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700 transition-colors"
      >
        Switch
      </button>
    </div>
  );
}

function StatusIndicator() {
  const { connected } = useWS();
  return (
    <div className="flex items-center gap-2 text-xs text-zinc-500">
      <span
        className={`h-2 w-2 rounded-full ${connected ? "bg-emerald-500" : "bg-red-500"}`}
      />
      {connected ? "Backend connected" : "Backend offline"}
    </div>
  );
}
