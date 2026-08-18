"use client";

import { useEffect, useRef, useState } from "react";
import { createWebSocket } from "@/lib/api";
import { useTenant } from "@/components/TenantProvider";
import type { WSMessage } from "@/lib/types";

export interface UseWebSocketReturn {
  connected: boolean;
  lastMessage: WSMessage | null;
}

export function useWebSocket(): UseWebSocketReturn {
  const [connected, setConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WSMessage | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const { agentId, conversationId, tenantVersion } = useTenant();

  useEffect(() => {
    let reconnectTimer: ReturnType<typeof setTimeout>;

    function connect() {
      const ws = createWebSocket((msg) => {
        setLastMessage(msg);
      }, agentId, conversationId);
      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        reconnectTimer = setTimeout(connect, 3000);
      };
      ws.onerror = () => ws.close();
      wsRef.current = ws;
    }

    wsRef.current?.close();
    connect();

    return () => {
      clearTimeout(reconnectTimer);
      wsRef.current?.close();
    };
  }, [agentId, conversationId, tenantVersion]);

  return { connected, lastMessage };
}
