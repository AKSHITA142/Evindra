"use client";

import { usePathname } from "next/navigation";
import { Sidebar, MobileBottomNav } from "@/components/shell/Sidebar";
import { Topbar } from "@/components/shell/Topbar";

/* Routes that should NOT get the app shell (e.g., the standalone landing page) */
const NO_SHELL_ROUTES = new Set(["/"]);


export function ShellLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const showShell = !NO_SHELL_ROUTES.has(pathname);

  if (!showShell) {
    return <>{children}</>;
  }

  return (
    <div className="dash-shell-bg flex h-screen overflow-hidden">
      {/* Desktop sidebar */}
      <Sidebar />

      {/* Main area */}
      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
        {/* Topbar */}
        <Topbar />

        {/* Scrollable page content */}
        <main
          id="main-content"
          className="flex-1 overflow-y-auto pb-20 md:pb-6"
        >
          {children}
        </main>
      </div>

      {/* Mobile bottom nav */}
      <MobileBottomNav />
    </div>
  );
}
