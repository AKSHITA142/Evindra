/** Format a number as a percentage string */
export function formatPercent(value: number | null | undefined, decimals = 1): string {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return `${Number(value).toFixed(decimals)}%`;
}

/** Format bytes to human-readable size */
export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined || bytes <= 0 || Number.isNaN(Number(bytes))) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

/** Format a duration in seconds to human-readable */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) return "—";
  const secNum = Number(seconds);
  if (secNum < 60) return `${secNum.toFixed(1)}s`;
  const mins = Math.floor(secNum / 60);
  const secs = Math.round(secNum % 60);
  return `${mins}m ${secs}s`;
}

/** Format a metric value (auto-detect float/percent) */
export function formatMetric(value: number | null | undefined, isPercent = false): string {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  const num = Number(value);
  if (isPercent) return formatPercent(num * 100);
  if (Math.abs(num) < 1) return num.toFixed(4);
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(num);
}

/** Format a number with thousand separators */
export function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "0";
  return new Intl.NumberFormat().format(Number(value));
}

/** Capitalize first letter */
export function capitalize(str: string | null | undefined): string {
  if (!str) return "";
  return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
}

/** Convert snake_case to Title Case */
export function snakeToTitle(str: string | null | undefined): string {
  if (!str) return "";
  return str
    .split("_")
    .map((word) => capitalize(word))
    .join(" ");
}

/** Format ISO date string to readable */
export function formatDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return "—";
    return d.toLocaleString("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "—";
  }
}

