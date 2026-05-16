// Strategy risk-return radar — carried over from the legacy
// hardcoded App.jsx. Now driven by /api/strategies output.

import {
    Legend,
    PolarAngleAxis,
    PolarGrid,
    PolarRadiusAxis,
    Radar,
    RadarChart,
    ResponsiveContainer,
} from "recharts";
import { colorFor, palette } from "../theme.js";

// Maps each radar axis to a normaliser over the strategies'
// mc_summary objects. Higher is better — for "loss-like" axes
// we invert the metric before normalising.
const axes = [
    {
        key: "return",
        label: "Return",
        pick: (s) => s.mc_summary.mu_ann_pct,
        invert: false,
    },
    {
        key: "pgain20",
        label: "P(Gain>20%)",
        pick: (s) => s.mc_summary.p_gain_20pct,
        invert: false,
    },
    {
        key: "lowloss",
        label: "Low P(Loss)",
        pick: (s) => s.mc_summary.prob_loss * 100.0,
        invert: true,
    },
    {
        key: "lowdd",
        label: "Low MaxDD",
        pick: (s) => Math.abs(s.mc_summary.max_drawdown_p50 * 100.0),
        invert: true,
    },
    {
        key: "lowdd15",
        label: "Low DD>15%",
        pick: (s) => s.mc_summary.p_dd_over_15,
        invert: true,
    },
    {
        key: "retvol",
        label: "Return/Vol",
        pick: (s) =>
            s.mc_summary.sigma_ann_pct
                ? s.mc_summary.mu_ann_pct / s.mc_summary.sigma_ann_pct
                : 0,
        invert: false,
    },
];

export default function RadarPanel({ strategies }) {
    if (!strategies?.length) return null;
    const data = axes.map(({ label, pick, invert }) => {
        const row = { metric: label };
        const raw = strategies.map(pick);
        const display = invert ? raw.map((v) => -v) : raw;
        const max = Math.max(...display);
        const min = Math.min(...display);
        const span = max - min || 1.0;
        strategies.forEach((s, idx) => {
            row[s.name] = ((display[idx] - min) / span) * 100;
        });
        return row;
    });
    return (
        <ResponsiveContainer width="100%" height={320}>
            <RadarChart data={data}>
                <PolarGrid stroke={palette.border} />
                <PolarAngleAxis
                    dataKey="metric"
                    tick={{ fill: palette.textMuted, fontSize: 11 }}
                />
                <PolarRadiusAxis
                    tick={{ fill: palette.textFaint, fontSize: 9 }}
                    domain={[0, 100]}
                />
                {strategies.map((s) => {
                    const c = colorFor(s.name);
                    return (
                        <Radar
                            key={s.name}
                            name={c.label}
                            dataKey={s.name}
                            stroke={c.main}
                            fill={c.main}
                            fillOpacity={0.15}
                        />
                    );
                })}
                <Legend wrapperStyle={{ fontSize: 11 }} />
            </RadarChart>
        </ResponsiveContainer>
    );
}
