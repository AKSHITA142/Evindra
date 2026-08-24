"use client";

import { motion } from "framer-motion";
import { cn } from "@/utils/cn";
import type { ReactNode } from "react";

interface MetricCardProps {
  label: string;
  value: string | number;
  icon?: ReactNode;
  subtext?: string;
  trend?: "up" | "down" | "neutral";
  trendValue?: string;
  className?: string;
  /** Semantic, not decorative — pick the one that matches what the metric means */
  accent?: "brand" | "success" | "warning" | "error" | "neutral";
}

const accentMap = {
  brand: {
    icon: "text-brand-400",
    bg: "bg-brand-500/10",
    border: "border-brand-500/25",
    value: "text-text",
  },
  success: {
    icon: "text-success-400",
    bg: "bg-success-500/10",
    border: "border-success-500/25",
    value: "text-text",
  },
  warning: {
    icon: "text-warning-400",
    bg: "bg-warning-500/10",
    border: "border-warning-500/25",
    value: "text-text",
  },
  error: {
    icon: "text-error-400",
    bg: "bg-error-500/10",
    border: "border-error-500/25",
    value: "text-text",
  },
  neutral: {
    icon: "text-text-secondary",
    bg: "bg-surface-3",
    border: "border-border",
    value: "text-text",
  },
};

export function MetricCard({
  label,
  value,
  icon,
  subtext,
  trend,
  trendValue,
  className,
  accent = "neutral",
}: MetricCardProps) {
  const colors = accentMap[accent];
  const isString = typeof value === "string";
  const strLen = isString ? value.length : 0;
  const isLong = strLen > 15;
  const isVeryLong = strLen > 24;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.28, ease: "easeOut" }}
      whileHover={{ y: -2 }}
      className={cn("card p-4 sm:p-5 flex items-start gap-3 sm:gap-4 depth-hover min-w-0 h-full", className)}
    >
      {icon && (
        <div
          className={cn(
            "w-9 h-9 sm:w-10 sm:h-10 rounded-lg flex items-center justify-center shrink-0 border mt-0.5",
            colors.bg,
            colors.border
          )}
        >
          <span className={cn("w-4 h-4 sm:w-5 sm:h-5", colors.icon)}>{icon}</span>
        </div>
      )}
      <div className="min-w-0 flex-1">
        <p className="text-[11px] sm:text-xs text-text-secondary font-medium mb-1 uppercase tracking-wider truncate">
          {label}
        </p>
        <p
          className={cn(
            "font-bold",
            isVeryLong
              ? "text-xs sm:text-sm leading-snug break-words"
              : isLong
              ? "text-sm sm:text-base leading-snug break-words"
              : "text-2xl leading-none tabular-nums",
            colors.value
          )}
          title={isString ? value : undefined}
        >
          {value}
        </p>
        {subtext && (
          <p className="text-xs text-text-muted mt-1.5 leading-tight break-words">{subtext}</p>
        )}
        {trend && trendValue && (
          <p
            className={cn(
              "text-xs font-medium mt-1.5",
              trend === "up" && "text-success-400",
              trend === "down" && "text-error-400",
              trend === "neutral" && "text-text-secondary"
            )}
          >
            {trend === "up" ? "↑" : trend === "down" ? "↓" : "→"} {trendValue}
          </p>
        )}
      </div>
    </motion.div>
  );
}
