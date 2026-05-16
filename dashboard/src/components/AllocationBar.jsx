// Stacked allocation bar — shows target_weights buckets per
// strategy. Buckets are derived from the ticker -> category map
// when available (cash bucket comes from the CASH pseudo-ticker).

import {
    Bar,
    BarChart,
    Legend,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from "recharts";
import { palette, tooltipStyle } from "../theme.js";

const bucketColors = {
    cash: "#f59e0b",
    low_vol: "#10b981",
    etf: "#6366f1",
    equity: "#ef4444",
    other: palette.textMuted,
};

function bucketise(targetWeights, categories) {
    const buckets = { cash: 0, low_vol: 0, etf: 0, equity: 0, other: 0 };
    for (const [ticker, weight] of Object.entries(targetWeights)) {
        if (ticker === "CASH") {
            buckets.cash += weight * 100;
            continue;
        }
        const cat = (categories?.[ticker] || "equity").toLowerCase();
        if (cat.includes("low") || cat.includes("vol")) {
            buckets.low_vol += weight * 100;
        } else if (cat.includes("etf")) {
            buckets.etf += weight * 100;
        } else if (cat.includes("equity") || cat === "growth") {
            buckets.equity += weight * 100;
        } else {
            buckets.other += weight * 100;
        }
    }
    return buckets;
}

export default function AllocationBar({ strategies, categories }) {
    if (!strategies?.length) return null;
    const data = strategies.map((s) => ({
        name: s.name,
        ...bucketise(s.target_weights, categories || {}),
    }));
    return (
        <ResponsiveContainer width="100%" height={130}>
            <BarChart layout="vertical" data={data} margin={{ left: 110 }}>
                <XAxis
                    type="number"
                    tick={{ fill: palette.textFaint, fontSize: 11 }}
                    tickFormatter={(v) => `${v}%`}
                    domain={[0, 100]}
                />
                <YAxis
                    type="category"
                    dataKey="name"
                    tick={{ fill: palette.textMuted, fontSize: 11 }}
                    width={105}
                />
                <Tooltip
                    contentStyle={tooltipStyle}
                    formatter={(v) => `${v.toFixed(1)}%`}
                />
                <Bar
                    dataKey="cash"
                    stackId="a"
                    fill={bucketColors.cash}
                    name="Cash"
                />
                <Bar
                    dataKey="low_vol"
                    stackId="a"
                    fill={bucketColors.low_vol}
                    name="Low-Vol"
                />
                <Bar
                    dataKey="etf"
                    stackId="a"
                    fill={bucketColors.etf}
                    name="ETFs"
                />
                <Bar
                    dataKey="equity"
                    stackId="a"
                    fill={bucketColors.equity}
                    name="Equity"
                />
                <Bar
                    dataKey="other"
                    stackId="a"
                    fill={bucketColors.other}
                    name="Other"
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
            </BarChart>
        </ResponsiveContainer>
    );
}
