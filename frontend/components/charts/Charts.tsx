"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  ScatterChart,
  Scatter,
  Legend,
} from "recharts";

// ── Shared chart colours ──────────────────────
// Coherent analytical palette anchored on the brand teal, not a rainbow.
// Use CHART_COLORS for neutral categorical series (e.g. pie/donut segments
// with no inherent severity meaning). For severity-coded data, use the
// semantic tokens directly (success/warning/error) instead of this array.
export const CHART_COLORS = [
  "#76FF03", // brand-500 (lime primary)
  "#a6ff5c", // brand-300 (light lime)
  "#4fae00", // brand-700 (deep lime)
  "#3892F6", // info (neutral steel-blue)
  "#FFC107", // warning-500 (attention)
  "#A1A4AA", // text-secondary (other/neutral)
  "#8CFF20", // brand-400 (bright lime)
  "#00E676", // success-500 (only if a "healthy" segment applies)
];

const tooltipStyle = {
  backgroundColor: "var(--surface-3)",
  border: "1px solid var(--border)",
  borderRadius: "10px",
  color: "var(--text)",
  fontSize: "12px",
};

const axisTick = { fontSize: 11, fill: "#70757C" };
const gridStroke = "rgba(255,255,255,0.05)";
const brandCursor = { fill: "rgba(118,255,3,0.08)" };

type TooltipValue = number | string | ReadonlyArray<number | string> | undefined;
type TooltipName = number | string | undefined;

function formatVal(v: TooltipValue): string {
  if (typeof v === "number") return v.toFixed(4);
  if (typeof v === "string") return v;
  if (Array.isArray(v)) return v.join(", ");
  return "";
}

// ── Vertical Bar Chart ────────────────────────
interface BarItem {
  name: string;
  value: number;
}

interface AppBarChartProps {
  data: BarItem[];
  label?: string;
  color?: string;
  height?: number;
}

export function AppBarChart({
  data,
  label = "Value",
  color = "#76FF03",
  height = 240,
}: AppBarChartProps) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
        <XAxis
          dataKey="name"
          tick={axisTick}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          tick={axisTick}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          contentStyle={tooltipStyle}
          cursor={brandCursor}
          formatter={(v: TooltipValue) => [formatVal(v), label]}
        />
        <Bar dataKey="value" fill={color} radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

// ── Horizontal Bar Chart (Feature Importance) ──
export function HorizontalBarChart({
  data,
  label = "Importance",
  height = 240,
}: AppBarChartProps) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart
        data={data}
        layout="vertical"
        margin={{ top: 4, right: 20, left: 4, bottom: 0 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} horizontal={false} />
        <XAxis
          type="number"
          tick={axisTick}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          dataKey="name"
          type="category"
          tick={axisTick}
          axisLine={false}
          tickLine={false}
          width={100}
        />
        <Tooltip
          contentStyle={tooltipStyle}
          cursor={brandCursor}
          formatter={(v: TooltipValue) => [formatVal(v), label]}
        />
        <Bar dataKey="value" fill="#76FF03" radius={[0, 4, 4, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

// Semantic Job Status Color Palette
export const STATUS_COLORS: Record<string, string> = {
  completed: "#00E676", // success green
  success: "#00E676",
  running: "#76FF03",   // lime (active)
  queued: "#FFC107",    // warning amber
  failed: "#FF3D00",    // error red
  error: "#FF3D00",
  cancelled: "#70757C", // muted grey
};

// ── Pie / Donut Chart ─────────────────────────
interface PieItem {
  name: string;
  value: number;
  color?: string;
}

interface AppPieChartProps {
  data: PieItem[];
  height?: number;
  innerRadius?: number;
}

export function AppPieChart({
  data,
  height = 220,
  innerRadius = 55,
}: AppPieChartProps) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <PieChart margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          innerRadius={innerRadius}
          outerRadius={innerRadius + 32}
          paddingAngle={4}
          dataKey="value"
          stroke="var(--surface-2)"
          strokeWidth={2}
        >
          {data.map((entry, i) => {
            const key = entry.name.toLowerCase();
            const fill = entry.color || STATUS_COLORS[key] || CHART_COLORS[i % CHART_COLORS.length];
            return <Cell key={i} fill={fill} />;
          })}
        </Pie>
        <Tooltip
          contentStyle={tooltipStyle}
          formatter={(v: TooltipValue, name: TooltipName) => [
            `${v} jobs`,
            String(name).charAt(0).toUpperCase() + String(name).slice(1),
          ]}
        />
      </PieChart>
    </ResponsiveContainer>
  );
}

// ── Scatter Chart (Runtime vs Accuracy) ───────
interface ScatterItem {
  x: number;
  y: number;
  name: string;
}

interface AppScatterChartProps {
  data: ScatterItem[];
  xLabel?: string;
  yLabel?: string;
  height?: number;
}

export function AppScatterChart({
  data,
  xLabel = "Runtime (s)",
  yLabel = "Score",
  height = 260,
}: AppScatterChartProps) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
        <XAxis
          type="number"
          dataKey="x"
          name={xLabel}
          tick={axisTick}
          axisLine={false}
          tickLine={false}
          label={{ value: xLabel, position: "insideBottom", offset: -4, fill: "#70757C", fontSize: 11 }}
        />
        <YAxis
          type="number"
          dataKey="y"
          name={yLabel}
          tick={axisTick}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          contentStyle={tooltipStyle}
          cursor={{ strokeDasharray: "3 3", stroke: "rgba(118,255,3,0.4)" }}
          formatter={(v: TooltipValue, name: TooltipName) => [
            formatVal(v),
            name ?? "",
          ]}
        />
        <Scatter data={data} fill="#76FF03" opacity={0.85} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}
