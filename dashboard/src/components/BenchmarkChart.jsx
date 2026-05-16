// Normalised performance chart: portfolio NAV vs benchmarks,
// each rebased to 100 at the first shared trading day.

import {
    CartesianGrid,
    Legend,
    Line,
    LineChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from "recharts";
import { palette, tooltipStyle } from "../theme.js";

const benchmarkColors = {
    portfolio: "#3b82f6",
    SPY: "#10b981",
    DIA: "#f59e0b",
    VT: "#a78bfa",
};

export default function BenchmarkChart({ series, labels }) {
    if (!series || series.length === 0) {
        return (
            <div
                style={{
                    color: palette.textMuted,
                    fontSize: 13,
                    padding: 16,
                }}
            >
                Waiting for performance series…
            </div>
        );
    }
    return (
        <ResponsiveContainer width="100%" height={320}>
            <LineChart data={series} margin={{ left: 10, right: 10 }}>
                <CartesianGrid
                    stroke={palette.border}
                    strokeDasharray="3 3"
                    vertical={false}
                />
                <XAxis
                    dataKey="date"
                    tick={{ fill: palette.textFaint, fontSize: 10 }}
                    minTickGap={48}
                />
                <YAxis
                    tick={{ fill: palette.textFaint, fontSize: 10 }}
                    tickFormatter={(v) => `${Math.round(v)}`}
                />
                <Tooltip
                    contentStyle={tooltipStyle}
                    formatter={(v) => (v ? v.toFixed(2) : "—")}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line
                    type="monotone"
                    dataKey="portfolio"
                    stroke={benchmarkColors.portfolio}
                    strokeWidth={2.5}
                    dot={false}
                    name="Portfolio"
                />
                {labels.map((label) => (
                    <Line
                        key={label}
                        type="monotone"
                        dataKey={label}
                        stroke={benchmarkColors[label] || palette.textMuted}
                        strokeWidth={2}
                        dot={false}
                        name={label}
                    />
                ))}
            </LineChart>
        </ResponsiveContainer>
    );
}
