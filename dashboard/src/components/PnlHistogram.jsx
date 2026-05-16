// P&L histogram for one strategy + an optional overlay line of
// every strategy's distribution.

import {
    Bar,
    BarChart,
    CartesianGrid,
    Cell,
    Legend,
    Line,
    LineChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from "recharts";
import { colorFor, palette, tooltipStyle } from "../theme.js";

export function PnlHistogram({ strategy }) {
    if (!strategy?.pnl_histogram?.length) return null;
    const main = colorFor(strategy.name).main;
    return (
        <ResponsiveContainer width="100%" height={260}>
            <BarChart data={strategy.pnl_histogram} margin={{ left: 10 }}>
                <CartesianGrid
                    stroke={palette.border}
                    strokeDasharray="3 3"
                    vertical={false}
                />
                <XAxis
                    dataKey="pnl"
                    tick={{ fill: palette.textFaint, fontSize: 10 }}
                    tickFormatter={(v) => `${v.toFixed(0)}%`}
                    interval={3}
                />
                <YAxis
                    tick={{ fill: palette.textFaint, fontSize: 10 }}
                    tickFormatter={(v) => `${v.toFixed(1)}%`}
                />
                <Tooltip
                    contentStyle={tooltipStyle}
                    formatter={(v) => `${v.toFixed(2)}%`}
                />
                <Bar dataKey="freq" radius={[2, 2, 0, 0]}>
                    {strategy.pnl_histogram.map((entry, i) => (
                        <Cell
                            key={i}
                            fill={entry.pnl < 0 ? palette.danger : main}
                            fillOpacity={entry.pnl < 0 ? 1 : 0.85}
                        />
                    ))}
                </Bar>
            </BarChart>
        </ResponsiveContainer>
    );
}

export function PnlOverlay({ strategies }) {
    if (!strategies?.length) return null;
    return (
        <ResponsiveContainer width="100%" height={240}>
            <LineChart margin={{ left: 10 }}>
                <CartesianGrid
                    stroke={palette.border}
                    strokeDasharray="3 3"
                    vertical={false}
                />
                <XAxis
                    dataKey="pnl"
                    type="number"
                    tick={{ fill: palette.textFaint, fontSize: 10 }}
                    tickFormatter={(v) => `${v.toFixed(0)}%`}
                />
                <YAxis
                    tick={{ fill: palette.textFaint, fontSize: 10 }}
                    tickFormatter={(v) => `${v.toFixed(1)}%`}
                />
                <Tooltip
                    contentStyle={tooltipStyle}
                    formatter={(v) => `${v.toFixed(2)}%`}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {strategies.map((s) => {
                    const c = colorFor(s.name);
                    return (
                        <Line
                            key={s.name}
                            data={s.pnl_histogram}
                            dataKey="freq"
                            stroke={c.main}
                            strokeWidth={2}
                            dot={false}
                            name={c.label}
                        />
                    );
                })}
            </LineChart>
        </ResponsiveContainer>
    );
}
