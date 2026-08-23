"use client";

import { usePathname } from "next/navigation";
import { useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { HamburgerButton, MobileDrawer } from "./Sidebar";

/* ── Route → human title map ─────────────────────── */
const ROUTE_TITLES: Record<string, string> = {
  "/": "Overview",
  "/overview": "Research Workspace",
  "/upload": "New Research Run",
  "/timeline": "Research Timeline",
  "/experiments": "Experiment Matrix",
  "/knowledge": "Hypotheses & Knowledge Base",
  "/recommendation": "Executive Report & Recommendations",
};


function getPageTitle(pathname: string): string {
  // Exact match
  if (ROUTE_TITLES[pathname]) return ROUTE_TITLES[pathname];
  // Prefix match for dynamic [jobId] routes
  for (const [prefix, title] of Object.entries(ROUTE_TITLES)) {
    if (pathname.startsWith(prefix + "/")) return title;
  }
  return "Evidra";
}

/* ── Topbar ──────────────────────────────────────── */
export function Topbar() {
  const pathname = usePathname();
  const [drawerOpen, setDrawerOpen] = useState(false);
  const title = getPageTitle(pathname);

  return (
    <>
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:px-3 focus:py-1.5 focus:bg-brand-500 focus:text-[#052620] focus:font-bold focus:rounded-md focus:shadow-md"
      >
        Skip to main content
      </a>
      <header className="
        sticky top-0 z-10 flex items-center justify-between
        h-14 px-4 md:px-6
        bg-surface-1/80 border-b border-border-subtle
        backdrop-blur-sm shrink-0
      ">

        {/* Left: hamburger (mobile) + title */}
        <div className="flex items-center gap-3">
          <HamburgerButton onClick={() => setDrawerOpen(true)} />

          {/* Mobile logo (visible only when sidebar is hidden) */}
          <Link href="/overview" className="md:hidden flex items-center gap-2">
            <Image
              src="/evidra-icon-v2.png"
              alt="Evidra"
              width={200}
              height={200}
              className="w-8 h-8 rounded-lg shrink-0 object-contain shadow-sm"
            />
          </Link>

          <h1 className="font-semibold text-sm text-text tracking-tight">
            {title}
          </h1>
        </div>

      </header>

      {/* Mobile drawer */}
      <MobileDrawer open={drawerOpen} onClose={() => setDrawerOpen(false)} />
    </>
  );
}
