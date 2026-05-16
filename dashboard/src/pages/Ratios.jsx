// Ratios page — return KPIs (Day P&L, YTD, 1Y, 3Y) and the
// four risk metrics (Sharpe, Sortino, MaxDD, CAGR) rendered as
// a single table for compact scanning.

import { useEndpoint } from "../hooks/useEndpoint.js";
import { fmtNum, fmtPct, palette } from "../theme.js";

export default function Ratios() {
    const { data, loading, error } = useEndpoint("/api/overview");
    if (loading) {
        return (
            <div style={{ color: palette.textMuted, padding: 24 }}>
                Loading ratios…
            </div>
        );
    }
    if (error) {
        return (
            <div
                style={{
                    color: palette.danger,
                    fontFamily: "monospace",
                    padding: 16,
                }}
            >
                {error.message}
            </div>
        );
    }
    const d = data || {};
    const r = d.risk || {};
    const rows = [
        {
            label: "Day P&L",
            value: fmtPct(d.day_pnl_pct),
            color: (d.day_pnl_pct ?? 0) >= 0 ? palette.success : palette.danger,
            desc: "Most recent trading-day return",
        },
        {
            label: "Year-to-date",
            value: fmtPct(d.return_ytd_pct),
            color:
                (d.return_ytd_pct ?? 0) >= 0
                    ? palette.success
                    : palette.danger,
            desc: "Cumulative return since Jan 1",
        },
        {
            label: "1Y return",
            value: fmtPct(d.return_1y_pct),
            color:
                (d.return_1y_pct ?? 0) >= 0
                    ? palette.success
                    : palette.danger,
            desc: "Cumulative return over the trailing 252 trading days",
        },
        {
            label: "3Y return",
            value: fmtPct(d.return_3y_pct),
            color:
                (d.return_3y_pct ?? 0) >= 0
                    ? palette.success
                    : palette.danger,
            desc: "Cumulative return over the trailing 756 trading days",
        },
        {
            label: "Sharpe ratio",
            value: fmtNum(r.sharpe),
            color:
                (r.sharpe ?? 0) >= 1
                    ? palette.success
                    : (r.sharpe ?? 0) >= 0
                      ? palette.warn
                      : palette.danger,
            desc: "Annualised excess return per unit of total volatility (rf=0)",
        },
        {
            label: "Sortino ratio",
            value: fmtNum(r.sortino),
            color:
                (r.sortino ?? 0) >= 1
                    ? palette.success
                    : (r.sortino ?? 0) >= 0
                      ? palette.warn
                      : palette.danger,
            desc: "Like Sharpe but penalises only downside deviation",
        },
        {
            label: "Maximum drawdown",
            value: fmtPct(r.max_drawdown_pct),
            color: palette.danger,
            desc: "Worst peak-to-trough decline in portfolio value",
        },
        {
            label: "CAGR",
            value: fmtPct(r.cagr_pct),
            color:
                (r.cagr_pct ?? 0) >= 0
                    ? palette.success
                    : palette.danger,
            desc: "Compound annual growth rate since the first transaction",
        },
    ];

    return (
        <div
            style={{
                background: palette.card,
                borderRadius: 10,
                padding: 16,
            }}
        >
            <div style={{ marginBottom: 12 }}>
                <div
                    style={{
                        color: palette.textMuted,
                        fontWeight: 600,
                        fontSize: 14,
                    }}
                >
                    Performance & risk ratios
                </div>
                <div style={{ color: palette.textFaint, fontSize: 11 }}>
                    Computed from the backcast daily-return series
                </div>
            </div>
            <div
                style={{
                    border: `1px solid ${palette.border}`,
                    borderRadius: 8,
                    overflow: "hidden",
                }}
            >
                <table style={{ width: "100%", fontSize: 13 }}>
                    <thead>
                        <tr style={{ background: palette.bg }}>
                            <th
                                style={{
                                    textAlign: "left",
                                    padding: "10px 14px",
                                    color: palette.textFaint,
                                    fontWeight: 500,
                                    width: "30%",
                                }}
                            >
                                Metric
                            </th>
                            <th
                                style={{
                                    textAlign: "right",
                                    padding: "10px 14px",
                                    color: palette.textFaint,
                                    fontWeight: 500,
                                    width: "15%",
                                }}
                            >
                                Value
                            </th>
                            <th
                                style={{
                                    textAlign: "left",
                                    padding: "10px 14px",
                                    color: palette.textFaint,
                                    fontWeight: 500,
                                }}
                            >
                                Description
                            </th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows.map((row, i) => (
                            <tr
                                key={row.label}
                                style={{
                                    background:
                                        i % 2 === 0
                                            ? palette.card
                                            : palette.cardAlt,
                                    borderTop: `1px solid ${palette.bg}`,
                                }}
                            >
                                <td
                                    style={{
                                        padding: "10px 14px",
                                        color: palette.text,
                                        fontWeight: 600,
                                    }}
                                >
                                    {row.label}
                                </td>
                                <td
                                    style={{
                                        padding: "10px 14px",
                                        fontFamily: "monospace",
                                        textAlign: "right",
                                        fontWeight: 700,
                                        fontSize: 15,
                                        color: row.color,
                                    }}
                                >
                                    {row.value}
                                </td>
                                <td
                                    style={{
                                        padding: "10px 14px",
                                        color: palette.textMuted,
                                        fontSize: 12,
                                    }}
                                >
                                    {row.desc}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
