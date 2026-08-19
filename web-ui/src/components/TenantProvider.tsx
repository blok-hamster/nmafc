"use client";

import { createContext, useCallback, useContext, useState } from "react";

interface TenantState {
  agentId: string;
  conversationId: string;
  setAgentId: (id: string) => void;
  setConversationId: (id: string) => void;
  /** Bumped on every tenant change so pages know to re-fetch. */
  tenantVersion: number;
}

const TenantContext = createContext<TenantState>({
  agentId: "default",
  conversationId: "default",
  setAgentId: () => {},
  setConversationId: () => {},
  tenantVersion: 0,
});

export function TenantProvider({ children }: { children: React.ReactNode }) {
  const [agentId, setAgentIdRaw] = useState("default");
  const [conversationId, setConversationIdRaw] = useState("default");
  const [version, setVersion] = useState(0);

  const setAgentId = useCallback((id: string) => {
    setAgentIdRaw(id || "default");
    setVersion((v) => v + 1);
  }, []);

  const setConversationId = useCallback((id: string) => {
    setConversationIdRaw(id || "default");
    setVersion((v) => v + 1);
  }, []);

  return (
    <TenantContext.Provider
      value={{
        agentId,
        conversationId,
        setAgentId,
        setConversationId,
        tenantVersion: version,
      }}
    >
      {children}
    </TenantContext.Provider>
  );
}

export function useTenant() {
  return useContext(TenantContext);
}
