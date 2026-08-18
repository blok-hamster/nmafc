import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Sidebar } from "@/components/layout/Sidebar";
import { WebSocketProvider } from "@/components/layout/WebSocketProvider";
import { TenantProvider } from "@/components/TenantProvider";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "NMAFC Memory Explorer",
  description: "Neuromorphic Memory Architecture - visual memory explorer",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex bg-zinc-950 text-zinc-100">
        <TenantProvider>
          <WebSocketProvider>
            <Sidebar />
            <main className="flex-1 overflow-auto">{children}</main>
          </WebSocketProvider>
        </TenantProvider>
      </body>
    </html>
  );
}
