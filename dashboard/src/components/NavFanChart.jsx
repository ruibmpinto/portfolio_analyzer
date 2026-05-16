// Percentile fan chart of NAV over the simulated horizon.

import {
    Area,
    AreaChart,
    CartesianGrid,
    Line,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from "recharts";
import { colorFor, fmtCHF, palette, tooltipStyle } from "../theme.js";

export default function NavFanChart({ strategy }) {
    if (!strategy?.fan_chart?.length) return null;
    const main = colorFor(strategy.name).main;
    return (
        <ResponsiveContainer width="100%" height={340}>
            <AreaChart
                data={strategy.fan_chart}
                margin={{ left: 20, right: 10 }}
            >
                <CartesianGrid
                    stroke={palette.border}
                    strokeDasharray="3 3"
                />
                <XAxis
                    dataKey="month"
                    tick={{ fill: palette.textFaint, fontSize: 11 }}
                    tickFormatter={(v) => `M${v}`}
                />
                <YAxis
                    tick={{ fill: palette.textFaint, fontSize: 11 }}
                    tickFormatter={(v) => `${Math.round(v / 1000)}k`}
                    domain={["dataMin - 5000", "dataMax + 5000"]}
                />
                <Tooltip
                    contentStyle={tooltipStyle}
                    formatter={(v) => fmtCHF(v)}
                    labelFormatter={(v) => `Month ${v}`}
                />
                <Area
                    type="monotone"
                    dataKey="p95"
                    stroke="none"
                    fill={main}
                    fillOpacity={0.1}
                />
                <Area
                    type="monotone"
                    dataKey="p75"
                    stroke="none"
                    fill={main}
                    fillOpacity={0.15}
                />
                <Area
                    type="monotone"
                    dataKey="p25"
                    stroke="none"
                    fill={palette.bg}
                    fillOpacity={0.5}
                />
                <Area
                    type="monotone"
                    dataKey="p5"
                    stroke="none"
                    fill={palette.bg}
                    fillOpacity={0.7}
                />
                <Line
                    type="monotone"
                    dataKey="p50"
                    stroke={main}
                    strokeWidth={2.5}
                    dot={false}
                />
            </AreaChart>
        </ResponsiveContainer>
    );
}
