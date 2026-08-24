"use client";

import Link from "next/link";
import Image from "next/image";
import { usePathname, useRouter } from "next/navigation";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload,
  Activity,
  FlaskConical,
  BookOpen,
  Lightbulb,
  ChevronLeft,
  ChevronRight,
  Menu,
  X,
  LayoutDashboard,
  Lock,
  AlertCircle,
  Info,
} from "lucide-react";
import { useDashboard } from "@/hooks/useResearch";

/* ── Base Nav Items ────────────────────────────── */
interface NavItemSpec {
  baseHref: string;
  icon: React.ElementType;
  label: string;
  dynamic?: boolean;
}

const PRIMARY_ITEMS: NavItemSpec[] = [
  { baseHref: "/overview", icon: LayoutDashboard, label: "Overview" },
  { baseHref: "/upload", icon: Upload, label: "New Run" },
  { baseHref: "/about", icon: Info, label: "What We Serve" },
];

const MISSION_ITEMS: NavItemSpec[] = [
  { baseHref: "/timeline", icon: Activity, label: "Timeline", dynamic: true },
  { baseHref: "/experiments", icon: FlaskConical, label: "Experiments", dynamic: true },
  { baseHref: "/knowledge", icon: BookOpen, label: "Knowledge", dynamic: true },
  { baseHref: "/recommendation", icon: Lightbulb, label: "Recommendation", dynamic: true },
];

/* ── Single Nav Link Component ───────────────────── */
function NavButton({
  item,
  activeJobId,
  collapsed,
  active,
  onDisabledClick,
}: {
  item: NavItemSpec;
  activeJobId: string | null;
  collapsed: boolean;
  active: boolean;
  onDisabledClick: (label: string) => void;
}) {
  const Icon = item.icon;
  const isMissionItem = !!item.dynamic;
  const isDisabled = isMissionItem && !activeJobId;
  const href = item.dynamic && activeJobId ? `${item.baseHref}/${activeJobId}` : item.baseHref;

  if (isDisabled) {
    return (
      <div className="relative group">
        <button
          type="button"
          onClick={() => onDisabledClick(item.label)}
          className={`
            w-full relative flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium
            transition-all duration-150 select-none cursor-not-allowed
            text-text-muted/40 hover:text-text-muted/60 hover:bg-surface-2/40
          `}
        >
          <div className="relative shrink-0">
            <Icon className="w-4 h-4 text-text-muted/40" />
            <Lock className="w-2.5 h-2.5 text-text-muted/60 absolute -bottom-1 -right-1" />
          </div>

          <AnimatePresence>
            {!collapsed && (
              <motion.div
                initial={{ opacity: 0, width: 0 }}
                animate={{ opacity: 1, width: "auto" }}
                exit={{ opacity: 0, width: 0 }}
                transition={{ duration: 0.18 }}
                className="flex items-center justify-between flex-1 overflow-hidden whitespace-nowrap min-w-0"
              >
                <span className="truncate">{item.label}</span>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-surface-3/80 text-text-muted/60 font-mono text-[9px] border border-border-subtle/50">
                  Locked
                </span>
              </motion.div>
            )}
          </AnimatePresence>
        </button>

        {/* Tooltip when collapsed */}
        {collapsed && (
          <div className="
            pointer-events-none absolute left-full ml-3 px-2.5 py-1.5 rounded-lg
            bg-surface-3 border border-border text-xs text-text-muted whitespace-nowrap
            opacity-0 group-hover:opacity-100 transition-opacity duration-150 z-50 shadow-xl
            flex items-center gap-1.5
          ">
            <Lock className="w-3 h-3 text-warning-400" />
            <span>{item.label} (Upload dataset first)</span>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="relative group">
      <Link
        href={href}
        className={`
          relative flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium
          transition-all duration-150 select-none cursor-pointer
          ${active
            ? "bg-brand-500/10 text-brand-400 border border-brand-500/20 shadow-sm"
            : "text-text-secondary hover:text-text hover:bg-surface-3"
          }
        `}
      >
        {/* Active indicator bar */}
        {active && (
          <span className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-5 rounded-r-full bg-brand-500 shadow-[0_0_8px_rgba(118,255,3,0.6)]" />
        )}

        <Icon
          className={`w-4 h-4 shrink-0 transition-colors ${active ? "text-brand-400" : "text-text-muted group-hover:text-text-secondary"
            }`}
        />

        <AnimatePresence>
          {!collapsed && (
            <motion.span
              initial={{ opacity: 0, width: 0 }}
              animate={{ opacity: 1, width: "auto" }}
              exit={{ opacity: 0, width: 0 }}
              transition={{ duration: 0.18 }}
              className="overflow-hidden whitespace-nowrap font-medium"
            >
              {item.label}
            </motion.span>
          )}
        </AnimatePresence>
      </Link>

      {/* Tooltip when collapsed */}
      {collapsed && (
        <span className="
          pointer-events-none absolute left-full ml-3 px-2.5 py-1.5 rounded-lg
          bg-surface-3 border border-border text-xs text-text whitespace-nowrap
          opacity-0 group-hover:opacity-100 transition-opacity duration-150 z-50 shadow-xl
        ">
          {item.label}
        </span>
      )}
    </div>
  );
}

/* ── Main Desktop Sidebar ────────────────────────── */
export function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const [collapsed, setCollapsed] = useState(false);
  const [disabledToast, setDisabledToast] = useState<string | null>(null);
  const { data: dash } = useDashboard();

  // Extract active jobId from URL if on a job route (e.g. /timeline/abc-123)
  const pathParts = pathname.split("/");
  const isJobRoute = ["timeline", "experiments", "knowledge", "recommendation"].includes(pathParts[1]);
  const urlJobId = isJobRoute && pathParts[2] ? pathParts[2] : null;

  // Fallback to most recent job from backend if on overview/upload page
  const latestJobId = dash?.recent_jobs?.[0]?.job_id ?? null;
  const activeJobId = urlJobId || latestJobId;

  const isActive = (spec: NavItemSpec) => {
    if (spec.baseHref === "/overview") return pathname === "/overview";
    return pathname.startsWith(spec.baseHref);
  };

  const handleDisabledClick = (label: string) => {
    setDisabledToast(`Please upload a dataset first to view ${label} and research progress.`);
    setTimeout(() => {
      setDisabledToast(null);
    }, 4500);
  };

  return (
    <>
      <motion.aside
        animate={{ width: collapsed ? 68 : 255 }}
        transition={{ duration: 0.22, ease: "easeInOut" }}
        className="
          hidden md:flex flex-col shrink-0
          bg-surface-1 border-r border-border-subtle
          h-screen sticky top-0 overflow-hidden z-20 select-none
        "
      >
        {/* Logo & Expand Toggle Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-border-subtle h-18 shrink-0">
          <Link href="/overview" className="flex items-center justify-start min-w-0">
            {collapsed ? (
              <Image
                src="/evidra-icon-v2.png"
                alt="Evidra"
                width={200}
                height={200}
                priority
                className="w-9 h-9 rounded-lg shrink-0 object-contain shadow-md shadow-brand-500/20"
              />
            ) : (
              <Image
                src="/evidra-logo-v2.png"
                alt="Evidra"
                width={947}
                height={380}
                priority
                className="h-13 w-auto max-w-[190px] object-contain object-left shrink-0 drop-shadow-sm"
              />
            )}
          </Link>

          {/* Quick collapse icon button */}
          {!collapsed && (
            <button
              onClick={() => setCollapsed(true)}
              className="p-1.5 rounded-md text-text-muted hover:text-text hover:bg-surface-3 transition-colors cursor-pointer"
              title="Collapse sidebar"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Nav links */}
        <nav className="flex-1 px-2.5 py-4 flex flex-col gap-1 overflow-y-auto overflow-x-hidden">
          {/* Primary Navigation */}
          <div className="space-y-1">
            {!collapsed && (
              <p className="px-2.5 text-[10px] font-bold uppercase tracking-widest text-text-muted/70 mb-1.5">
                Workspace
              </p>
            )}
            {PRIMARY_ITEMS.map((item) => (
              <NavButton
                key={item.baseHref}
                item={item}
                activeJobId={activeJobId}
                collapsed={collapsed}
                active={isActive(item)}
                onDisabledClick={handleDisabledClick}
              />
            ))}
          </div>

          {/* Visible Gap & Section Divider */}
          <div className="my-3 border-t border-border-subtle/80" />

          {/* Active Mission Execution Routes */}
          <div className="space-y-1">
            {!collapsed && (
              <div className="px-2.5 flex items-center justify-between mb-1.5">
                <p className="text-[10px] font-bold uppercase tracking-widest text-text-muted/70">
                  Research Mission
                </p>
                {activeJobId && (
                  <span className="w-1.5 h-1.5 rounded-full bg-success-400 animate-pulse" />
                )}
              </div>
            )}
            {MISSION_ITEMS.map((item) => (
              <NavButton
                key={item.baseHref}
                item={item}
                activeJobId={activeJobId}
                collapsed={collapsed}
                active={isActive(item)}
                onDisabledClick={handleDisabledClick}
              />
            ))}
          </div>
        </nav>

        {/* Bottom Expand / Collapse Toggle Bar */}
        <div className="p-2.5 border-t border-border-subtle bg-surface-2/40 shrink-0">
          <button
            onClick={() => setCollapsed((c) => !c)}
            className="
              w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg
              bg-surface-3/60 hover:bg-surface-3 text-text-secondary hover:text-text
              border border-border-subtle hover:border-border
              transition-all text-xs font-semibold cursor-pointer shadow-sm
            "
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {collapsed ? (
              <>
                <ChevronRight className="w-4 h-4 text-brand-400" />
                <span className="sr-only">Expand</span>
              </>
            ) : (
              <>
                <ChevronLeft className="w-4 h-4" />
                <span>Collapse Sidebar</span>
              </>
            )}
          </button>
        </div>
      </motion.aside>

      {/* Floating Disabled Notice Toast */}
      <AnimatePresence>
        {disabledToast && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 15, scale: 0.95 }}
            className="
              fixed bottom-6 left-6 md:left-24 z-50 max-w-md
              bg-surface-3/95 backdrop-blur-md border border-brand-500/30 rounded-xl p-4
              shadow-2xl shadow-black/60 flex items-start gap-3
            "
          >
            <div className="w-8 h-8 rounded-lg bg-warning-500/10 border border-warning-500/25 flex items-center justify-center shrink-0 mt-0.5">
              <AlertCircle className="w-4 h-4 text-warning-400" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-xs font-semibold text-text">Dataset Required</p>
              <p className="text-xs text-text-muted mt-0.5 leading-relaxed">
                {disabledToast}
              </p>
              <button
                onClick={() => {
                  setDisabledToast(null);
                  router.push("/upload");
                }}
                className="mt-2.5 inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-brand-500 hover:bg-brand-400 text-[#052620] text-xs font-bold transition-colors cursor-pointer"
              >
                <Upload className="w-3 h-3" />
                Upload Dataset
              </button>
            </div>
            <button
              onClick={() => setDisabledToast(null)}
              className="p-1 rounded-md text-text-muted hover:text-text transition-colors cursor-pointer"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

/* ── Mobile Drawer ──────────────────────────────── */
export function MobileDrawer({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const pathname = usePathname();
  const router = useRouter();
  const [disabledToast, setDisabledToast] = useState<string | null>(null);
  const { data: dash } = useDashboard();

  const pathParts = pathname.split("/");
  const isJobRoute = ["timeline", "experiments", "knowledge", "recommendation"].includes(pathParts[1]);
  const urlJobId = isJobRoute && pathParts[2] ? pathParts[2] : null;

  const latestJobId = dash?.recent_jobs?.[0]?.job_id ?? null;
  const activeJobId = urlJobId || latestJobId;

  const isActive = (spec: NavItemSpec) => {
    if (spec.baseHref === "/") return pathname === "/";
    return pathname.startsWith(spec.baseHref);
  };

  const handleDisabledClick = (label: string) => {
    setDisabledToast(`Please upload a dataset first to view ${label} and research progress.`);
    setTimeout(() => setDisabledToast(null), 4500);
  };

  return (
    <AnimatePresence>
      {open && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18 }}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40 md:hidden"
            onClick={onClose}
          />

          {/* Drawer panel */}
          <motion.aside
            initial={{ x: "-100%" }}
            animate={{ x: 0 }}
            exit={{ x: "-100%" }}
            transition={{ duration: 0.22, ease: "easeInOut" }}
            className="
              fixed top-0 left-0 bottom-0 w-72
              bg-surface-1 border-r border-border z-50 md:hidden
              flex flex-col
            "
          >
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-border-subtle h-18 shrink-0">
              <Link href="/overview" onClick={onClose} className="flex items-center">
                <Image
                  src="/evidra-logo-v2.png"
                  alt="Evidra"
                  width={947}
                  height={380}
                  className="h-12 w-auto max-w-[190px] object-contain shrink-0 drop-shadow-sm"
                />
              </Link>
              <button
                onClick={onClose}
                className="p-1.5 rounded-md text-text-muted hover:text-text hover:bg-surface-3 transition-colors cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Nav links */}
            <nav className="flex-1 px-3 py-4 flex flex-col gap-1 overflow-y-auto">
              <p className="px-3 text-[10px] font-bold uppercase tracking-widest text-text-muted/70 mb-1.5">
                Workspace
              </p>
              {PRIMARY_ITEMS.map((item) => (
                <div key={item.baseHref} onClick={onClose}>
                  <NavButton
                    item={item}
                    activeJobId={activeJobId}
                    collapsed={false}
                    active={isActive(item)}
                    onDisabledClick={handleDisabledClick}
                  />
                </div>
              ))}

              <div className="my-3 border-t border-border-subtle" />

              <p className="px-3 text-[10px] font-bold uppercase tracking-widest text-text-muted/70 mb-1.5">
                Research Mission
              </p>
              {MISSION_ITEMS.map((item) => (
                <div key={item.baseHref} onClick={activeJobId ? onClose : undefined}>
                  <NavButton
                    item={item}
                    activeJobId={activeJobId}
                    collapsed={false}
                    active={isActive(item)}
                    onDisabledClick={handleDisabledClick}
                  />
                </div>
              ))}
              {/* Disabled Toast in Mobile Drawer */}
              {disabledToast && (
                <div className="mx-3 my-2 p-3 rounded-lg bg-surface-2 border border-warning-500/30 flex items-start gap-2.5">
                  <AlertCircle className="w-4 h-4 text-warning-400 shrink-0 mt-0.5" />
                  <div className="flex-1 min-w-0">
                    <p className="text-[11px] text-text-muted leading-tight">{disabledToast}</p>
                    <button
                      onClick={() => {
                        setDisabledToast(null);
                        onClose();
                        router.push("/upload");
                      }}
                      className="mt-2 text-[11px] font-bold text-brand-400 hover:underline inline-flex items-center gap-1 cursor-pointer"
                    >
                      <Upload className="w-3 h-3" /> Upload Dataset
                    </button>
                  </div>
                  <button onClick={() => setDisabledToast(null)} className="text-text-muted hover:text-text cursor-pointer">
                    <X className="w-3 h-3" />
                  </button>
                </div>
              )}
            </nav>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}

/* ── Mobile Bottom Navigation Bar ────────────────── */
export function MobileBottomNav() {
  const pathname = usePathname();
  const { data: dash } = useDashboard();

  const pathParts = pathname.split("/");
  const isJobRoute = ["timeline", "experiments", "knowledge", "recommendation"].includes(pathParts[1]);
  const urlJobId = isJobRoute && pathParts[2] ? pathParts[2] : null;

  const latestJobId = dash?.recent_jobs?.[0]?.job_id ?? null;
  const activeJobId = urlJobId || latestJobId;

  const navItems = [
    { baseHref: "/overview", icon: LayoutDashboard, label: "Overview" },
    { baseHref: "/upload", icon: Upload, label: "New Run" },
    { baseHref: "/timeline", icon: Activity, label: "Timeline", dynamic: true },
    { baseHref: "/experiments", icon: FlaskConical, label: "Experiments", dynamic: true },
    { baseHref: "/recommendation", icon: Lightbulb, label: "Report", dynamic: true },
  ];

  return (
    <nav className="
      fixed bottom-0 left-0 right-0 z-30 md:hidden
      bg-surface-1/95 backdrop-blur-md border-t border-border-subtle
      flex items-center justify-around px-2 py-2 safe-area-bottom
    ">
      {navItems.map((item) => {
        const isActive = item.baseHref === "/overview" ? pathname === "/overview" : pathname.startsWith(item.baseHref);
        const isDisabled = item.dynamic && !activeJobId;
        const href = item.dynamic && activeJobId ? `${item.baseHref}/${activeJobId}` : item.baseHref;
        const Icon = item.icon;

        if (isDisabled) {
          return (
            <div
              key={item.baseHref}
              className="flex flex-col items-center gap-1 px-2 py-1 text-text-muted/40 cursor-not-allowed select-none"
            >
              <Icon className="w-4 h-4" />
              <span className="text-[10px] font-medium">{item.label}</span>
            </div>
          );
        }

        return (
          <Link
            key={item.baseHref}
            href={href}
            className={`
              flex flex-col items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-medium transition-colors
              ${isActive ? "text-brand-400 font-bold" : "text-text-muted hover:text-text"}
            `}
          >
            <Icon className={`w-4 h-4 ${isActive ? "text-brand-400" : ""}`} />
            <span className="text-[10px]">{item.label}</span>
          </Link>
        );
      })}
    </nav>
  );
}

/* ── Mobile Hamburger Button ─────────────────────── */
export function HamburgerButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="md:hidden p-1.5 rounded-md text-text-muted hover:text-text hover:bg-surface-3 transition-colors cursor-pointer"
      aria-label="Open navigation drawer"
    >
      <Menu className="w-5 h-5" />
    </button>
  );
}
