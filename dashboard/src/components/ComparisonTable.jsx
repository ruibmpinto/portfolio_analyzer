// Head-to-head metrics table — port of the legacy "Comparison"
// panel. Best value per row is highlighted in green.

import { fmtCHF, fmtPct, palette } from "../theme.js";

const rows = [
    { label: "Ann. Return", pick: (s) => s.mc_summary.mu_ann_pct, unit: "%" },
    {
        label: "Ann. Volatility",
        pick: (s) => s.mc_summary.sigma_ann_pct,
        unit: "%",
        invert: true,
    },
    {
        label: "Median terminal NAV",
        pick: (s) => s.mc_summary.terminal_p50,
        unit: "chf",
    },
    {
        label: "P(Loss)",
        pick: (s) => s.mc_summary.prob_loss * 100.0,
        unit: "%",
        invert: true,
    },
    { label: "VaR 5%", pick: (s) => s.mc_summary.var_5_pct, unit: "%" },
    { label: "CVaR 5%", pick: (s) => s.mc_summary.cvar_5_pct, unit: "%" },
    {
        label: "Median MaxDD",
        pick: (s) => s.mc_summary.max_drawdown_p50 * 100.0,
        unit: "%",
        invert: true,
    },
    {
        label: "P(DD>15%)",
        pick: (s) => s.mc_summary.p_dd_over_15,
        unit: "%",
        invert: true,
    },
    {
        label: "P(Gain>20%)",
        pick: (s) => s.mc_summary.p_gain_20pct,
        unit: "%",
    },
    {
        label: "3Y CAGR forecast",
        pick: (s) => s.cagr_forecast_pct,
        unit: "%",
    },
];

export default function ComparisonTable({ strategies }) {
    if (!strategies?.length) return null;
    return (
        <div
            style={{
                border: `1px solid ${palette.border}`,
                borderRadius: 8,
                overflow: "hidden",
            }}
        >
            <table
                style={{
                    width: "100%",
                    fontSize: 12,
                    borderCollapse: "collapse",
                }}
            >
                <thead>
                    <tr style={{ background: palette.bg }}>
                        <th
                            style={{
                                textAlign: "left",
                                padding: "8px 12px",
                                color: palette.textFaint,
                                fontWeight: 500,
                            }}
                        >
                            Metric
                        </th>
                        {strategies.map((s) => (
                            <th
                                key={s.name}
                                style={{
                                    textAlign: "left",
                                    padding: "8px 12px",
                                    color: palette.textFaint,
                                    fontWeight: 500,
                                }}
                            >
                                {s.name}
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {rows.map((row, idx) => {
                        const values = strategies.map(row.pick);
                        const best = row.invert
                            ? Math.min(...values)
                            : Math.max(...values);
                        return (
                            <tr
                                key={row.label}
                                style={{
                                    background:
                                        idx % 2 === 0
                                            ? palette.card
                                            : palette.cardAlt,
                                    borderTop: `1px solid ${palette.bg}`,
                                }}
                            >
                                <td
                                    style={{
                                        padding: "6px 12px",
                                        color: palette.textMuted,
                                    }}
                                >
                                    {row.label}
                                </td>
                                {values.map((v, i) => {
                                    const isBest = v === best;
                                    const display =
                                        row.unit === "chf"
                                            ? fmtCHF(v)
                                            : fmtPct(v);
                                    return (
                                        <td
                                            key={i}
                                            style={{
                                                padding: "6px 12px",
                                                fontFamily: "monospace",
                                                color: isBest
                                                    ? palette.success
                                                    : palette.text,
                                                fontWeight: isBest
                                                    ? 700
                                                    : 400,
                                            }}
                                        >
                                            {display}
                                        </td>
                                    );
                                })}
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
}
