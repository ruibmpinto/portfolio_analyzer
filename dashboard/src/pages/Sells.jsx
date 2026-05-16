// Sells page — tickers held >= 6 months are listed as
// eligible; tickers still in the lock-up appear in a separate
// muted table with the unlock date.

import { useEndpoint } from "../hooks/useEndpoint.js";
import { fmtDate, fmtNum, palette } from "../theme.js";

export default function Sells() {
    const { data, loading, error } = useEndpoint("/api/sells");
    if (loading)
        return (
            <div style={{ color: palette.textMuted, padding: 24 }}>
                Loading sell candidates…
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
    const eligible = data?.eligible || [];
    const locked = data?.locked || [];
    return (
        <div>
            <div
                style={{
                    background: palette.card,
                    borderRadius: 10,
                    padding: 16,
                    marginBottom: 16,
                }}
            >
                <Header
                    title="Sellable today"
                    sub={`Tickers whose oldest open lot is ≥ ${data?.min_hold_days || 183} days old`}
                />
                <Table rows={eligible} muted={false} />
            </div>
            <div
                style={{
                    background: palette.card,
                    borderRadius: 10,
                    padding: 16,
                }}
            >
                <Header
                    title="Locked"
                    sub="Positions still inside the 6-month hold window"
                />
                <Table rows={locked} muted />
            </div>
        </div>
    );
}

function Header({ title, sub }) {
    return (
        <div style={{ marginBottom: 10 }}>
            <div style={{ color: palette.textMuted, fontWeight: 600 }}>
                {title}
            </div>
            <div style={{ color: palette.textFaint, fontSize: 11 }}>
                {sub}
            </div>
        </div>
    );
}

function Table({ rows, muted }) {
    if (!rows.length) {
        return (
            <div style={{ color: palette.textFaint, padding: 8 }}>
                None.
            </div>
        );
    }
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
                            "Ticker",
                            "Shares",
                            "Earliest lot",
                            "Days held",
                            muted ? "Unlocks" : "Recommendation",
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
                    {rows.map((r, i) => (
                        <tr
                            key={r.ticker}
                            style={{
                                background:
                                    i % 2 === 0
                                        ? palette.card
                                        : palette.cardAlt,
                                opacity: muted ? 0.7 : 1,
                                borderTop: `1px solid ${palette.bg}`,
                            }}
                        >
                            <td
                                style={{
                                    padding: "6px 12px",
                                    fontFamily: "monospace",
                                    fontWeight: 600,
                                    color: palette.text,
                                }}
                            >
                                {r.ticker}
                            </td>
                            <td
                                style={{
                                    padding: "6px 12px",
                                    color: palette.textMuted,
                                }}
                            >
                                {fmtNum(r.shares, 0)}
                            </td>
                            <td
                                style={{
                                    padding: "6px 12px",
                                    color: palette.textMuted,
                                }}
                            >
                                {fmtDate(r.earliest_lot_date)}
                            </td>
                            <td
                                style={{
                                    padding: "6px 12px",
                                    color: palette.textMuted,
                                }}
                            >
                                {r.days_held}
                            </td>
                            <td
                                style={{
                                    padding: "6px 12px",
                                    color: muted
                                        ? palette.textFaint
                                        : palette.text,
                                }}
                            >
                                {muted
                                    ? fmtDate(r.unlock_date)
                                    : recsForRow(r)}
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}

function recsForRow(r) {
    if (!r.recommendations?.length) {
        return (
            <span style={{ color: palette.textFaint }}>
                no strategy flags this
            </span>
        );
    }
    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {r.recommendations.map((rec, i) => (
                <div key={i} style={{ fontSize: 11 }}>
                    <span
                        style={{
                            background: "#7f1d1d20",
                            color: palette.danger,
                            padding: "1px 6px",
                            borderRadius: 4,
                            marginRight: 6,
                            fontWeight: 600,
                        }}
                    >
                        {rec.action} {rec.shares}
                    </span>
                    <span style={{ color: palette.textMuted }}>
                        {rec.strategy} — {rec.note}
                    </span>
                </div>
            ))}
        </div>
    );
}
