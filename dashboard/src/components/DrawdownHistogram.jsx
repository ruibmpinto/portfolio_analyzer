// Max-drawdown histogram + overlay variant.

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

export function DrawdownHistogram({ strategy }) {
    if (!strategy?.dd_histogram?.length) return null;
    const main = colorFor(strategy.name).main;
    return (
        <ResponsiveContainer width="100%" height={260}>
            <BarChart data={strategy.dd_histogram} margin={{ left: 10 }}>
                <CartesianGrid
                    stroke={palette.border}
                    strokeDasharray="3 3"
                    vertical={false}
                />
                <XAxis
                    dataKey="dd"
                    tick={{ fill: palette.textFaint, fontSize: 10 }}
                    tickFormatter={(v) => `${v.toFixed(0)}%`}
                    interval={2}
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
                    {strategy.dd_histogram.map((entry, i) => (
                        <Cell
                            key={i}
                            fill={
                                entry.dd < -15
                                    ? palette.danger
                                    : entry.dd < -10
                                      ? palette.warn
                                      : main
                            }
                            fillOpacity={0.85}
                        />
                    ))}
                </Bar>
            </BarChart>
        </ResponsiveContainer>
    );
}

export function DrawdownOverlay({ strategies }) {
    if (!strategies?.length) return null;
    return (
        <ResponsiveContainer width="100%" height={220}>
            <LineChart margin={{ left: 10 }}>
                <CartesianGrid
                    stroke={palette.border}
                    strokeDasharray="3 3"
                    vertical={false}
                />
                <XAxis
                    dataKey="dd"
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
                            data={s.dd_histogram}
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
