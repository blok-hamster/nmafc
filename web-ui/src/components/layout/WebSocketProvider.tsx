"use client";

import { createContext, useContext, useEffect } from "react";
import { useWebSocket, type UseWebSocketReturn } from "@/hooks/useWebSocket";
import { useTenant } from "@/components/TenantProvider";
import { setTenantHeaderProvider } from "@/lib/api";

const WSContext = createContext<UseWebSocketReturn>({
  connected: false,
  lastMessage: null,
});

export function WebSocketProvider({ children }: { children: React.ReactNode }) {
  const { agentId, conversationId } = useTenant();

  // Wire the API client to send tenant headers on every request
  useEffect(() => {
    setTenantHeaderProvider(() => ({
      "X-Agent-ID": agentId,
      "X-Conversation-ID": conversationId,
    }));
  }, [agentId, conversationId]);

  const ws = useWebSocket();
  return <WSContext.Provider value={ws}>{children}</WSContext.Provider>;
}

export function useWS() {
  return useContext(WSContext);
}
