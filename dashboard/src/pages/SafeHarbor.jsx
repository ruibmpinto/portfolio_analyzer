// Safe-Harbor page — Swiss 5x-of-Jan-1-NAV turnover tracker plus
// sub-6-month holding-period violations. Consumes /api/safe_harbor.

import { useEndpoint } from "../hooks/useEndpoint.js";
import KpiCard from "../components/KpiCard.jsx";
import { fmtCHF, fmtDate, fmtNum, palette } from "../theme.js";


// Status -> chip colour + human label.
const _statusStyles = {
    ok: { color: palette.success, label: "Within safe harbor" },
    warning: {
        color: palette.warn,
        label: "Approaching turnover limit",
    },
    breach: { color: palette.danger, label: "Safe harbor breached" },
    no_baseline: {
        color: palette.textFaint,
        label: "No Jan-1 baseline",
    },
};


export default function SafeHarbor() {
    const { data, loading, error } = useEndpoint("/api/safe_harbor");
    if (loading)
        return (
            <div style={{ color: palette.textMuted, padding: 24 }}>
                Loading safe-harbor tracker…
            </div>
        );
    if (error)
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
    const violations = data?.holding_period_violations || [];
    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <StatusCard payload={data} />
            <KpiStrip payload={data} />
            <ViolationsCard violations={violations} />
        </div>
    );
}


function StatusCard({ payload }) {
    const style = _statusStyles[payload?.status] || _statusStyles.no_baseline;
    return (
        <div
            style={{
                background: palette.card,
                borderRadius: 10,
                padding: 16,
                borderLeft: `4px solid ${style.color}`,
            }}
        >
            <div
                style={{
                    color: palette.textFaint,
                    fontSize: 11,
                    textTransform: "uppercase",
                    letterSpacing: 0.5,
                    marginBottom: 4,
                }}
            >
                Status
            </div>
            <div
                style={{
                    color: style.color,
                    fontSize: 22,
                    fontWeight: 700,
                }}
            >
                {style.label}
            </div>
            <div
                style={{
                    color: palette.textMuted,
                    fontSize: 12,
                    marginTop: 6,
                    lineHeight: 1.5,
                }}
            >
                Swiss safe harbor requires (1) annual turnover under 5× the
                portfolio value at year start and (2) every closed position
                held ≥ 6 months. Either rule broken loses tax-free capital
                gains status for the year.
            </div>
        </div>
    );
}


function KpiStrip({ payload }) {
    const ratio = payload?.ytd_turnover_ratio;
    const ratioLabel =
        ratio === null || ratio === undefined
            ? "—"
            : `${ratio.toFixed(2)}× / ${payload?.threshold_ratio?.toFixed(1) ?? "5.0"}×`;
    const ratioColor =
        ratio === null || ratio === undefined
            ? palette.textFaint
            : ratio >= (payload?.threshold_ratio || 5.0)
                ? palette.danger
                : ratio >= 4.0
                    ? palette.warn
                    : palette.success;
    return (
        <div
            style={{
                display: "grid",
                gridTemplateColumns: "repeat(4, 1fr)",
                gap: 10,
            }}
        >
            <KpiCard
                label="Jan-1 NAV"
                value={fmtCHF(payload?.jan1_nav_chf)}
                color={palette.accent}
            />
            <KpiCard
                label="YTD turnover"
                value={fmtCHF(payload?.ytd_turnover_chf)}
                sub="buy + sell notional across all currencies"
                color={palette.textMuted}
            />
            <KpiCard
                label="Turnover ratio"
                value={ratioLabel}
                color={ratioColor}
            />
            <KpiCard
                label="YTD realized"
                value={fmtCHF(payload?.ytd_realized_chf)}
                sub="FIFO-matched closed lots this year"
                color={
                    (payload?.ytd_realized_chf || 0) >= 0
                        ? palette.success
                        : palette.danger
                }
            />
        </div>
    );
}


function ViolationsCard({ violations }) {
    return (
        <div
            style={{
                background: palette.card,
                borderRadius: 10,
                padding: 16,
                borderLeft:
                    violations.length > 0
                        ? `3px solid ${palette.danger}`
                        : `3px solid ${palette.success}`,
            }}
        >
            <div style={{ marginBottom: 10 }}>
                <div style={{ color: palette.textMuted, fontWeight: 600 }}>
                    Holding-period violations
                </div>
                <div style={{ color: palette.textFaint, fontSize: 11 }}>
                    Closed positions held less than 180 days (6-month rule)
                </div>
            </div>
            {violations.length === 0 ? (
                <div style={{ color: palette.textFaint, padding: 8 }}>
                    None. Every closed lot this year cleared the 6-month
                    minimum hold.
                </div>
            ) : (
                <ViolationsTable violations={violations} />
            )}
        </div>
    );
}


function ViolationsTable({ violations }) {
    return (
        <div
            style={{
                border: `1px solid ${palette.border}`,
                borderRadius: 8,
                overflow: "hidden",
            }}
        >
            <table style={{ width: "100%", fontSize: 12 }}>
                <thead>
                    <tr style={{ background: palette.bg }}>
                        {[
                            "Ticker", "Buy date", "Sell date",
                            "Days held", "Shares", "Realized (CHF)",
                        ].map((h) => (
                            <th
                                key={h}
                                style={{
                                    textAlign: "left",
                                    padding: "8px 12px",
                                    color: palette.textFaint,
                                    fontWeight: 500,
                                }}
                            >
                                {h}
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {violations.map((v, i) => (
                        <tr
                            key={`${v.ticker}-${v.sell_date}-${i}`}
                            style={{
                                background:
                                    i % 2 === 0
                                        ? palette.card
                                        : palette.cardAlt,
                            }}
                        >
                            <Cell mono bold color={palette.danger}>
                                {v.ticker}
                            </Cell>
                            <Cell>{fmtDate(v.buy_date)}</Cell>
                            <Cell>{fmtDate(v.sell_date)}</Cell>
                            <Cell>{v.days_held}</Cell>
                            <Cell>{fmtNum(v.shares, 2)}</Cell>
                            <Cell>{fmtCHF(v.realized_chf)}</Cell>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}


function Cell({ children, mono, bold, color }) {
    return (
        <td
            style={{
                padding: "6px 12px",
                color: color || palette.textMuted,
                fontFamily: mono ? "monospace" : "inherit",
                fontWeight: bold ? 600 : 400,
            }}
        >
            {children}
        </td>
    );
}
