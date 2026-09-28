import type { NextConfig } from "next";

const backendUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// NMAFC_EXPORT=1 builds a fully static dashboard (web-ui/out) that FastAPI
// serves itself — the one-container story. In that mode there is no Next
// server, so rewrites are impossible; the exported app talks to the same
// origin (/api and /ws/live), which the client already handles when
// NEXT_PUBLIC_API_URL is unset. Without NMAFC_EXPORT, dev keeps its proxy.
const isExport = process.env.NMAFC_EXPORT === "1";

const nextConfig: NextConfig = {
  ...(isExport ? { output: "export" as const, trailingSlash: true } : {}),
  ...(isExport
    ? {}
    : {
        async rewrites() {
          return [
            {
              source: "/api/:path*",
              destination: `${backendUrl}/api/:path*`,
            },
            {
              source: "/ws/:path*",
              destination: `${backendUrl}/ws/:path*`,
            },
          ];
        },
      }),
};

export default nextConfig;
