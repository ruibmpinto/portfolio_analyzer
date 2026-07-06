// Stops page — per-holding breakeven gap plus suggested stop-loss
// and rebuy price levels. Consumes /api/stops.

import { useEndpoint } from "../hooks/useEndpoint.js";
import { fmtNum, fmtPct, palette } from "../theme.js";


// Colour thresholds for the breakeven column. Anything above 40 bps
// is expensive (UK stamp territory); under 20 bps is cheap (US).
const _cheap_bps = 20.0;
const _expensive_bps = 40.0;


function _bpsColor(bps) {
    if (bps === null || bps === undefined) return palette.textFaint;
    if (bps >= _expensive_bps) return palette.danger;
    if (bps >= _cheap_bps) return palette.warn;
    return palette.success;
}


export default function Stops() {
    const { data, loading, error } = useEndpoint("/api/stops");
    if (loading)
        return (
            <div style={{ color: palette.textMuted, padding: 24 }}>
                Computing per-holding breakeven gaps…
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
    const rows = data?.rows || [];
    return (
        <div
            style={{
                background: palette.card,
                borderRadius: 10,
                padding: 16,
            }}
        >
            <div style={{ marginBottom: 10 }}>
                <div style={{ color: palette.textMuted, fontWeight: 600 }}>
                    Suggested stops and rebuy targets
                </div>
                <div style={{ color: palette.textFaint, fontSize: 11 }}>
                    Trigger sits 2× breakeven below current. Rebuy sits one
                    more breakeven below the trigger — after fees + spread,
                    the round-trip breaks even.
                </div>
            </div>
            {rows.length === 0 ? (
                <div style={{ color: palette.textFaint, padding: 8 }}>
                    No open positions.
                </div>
            ) : (
                <StopsTable rows={rows} />
            )}
        </div>
    );
}


function StopsTable({ rows }) {
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
                            "Ticker", "Weight",
                            "Current", "Sell bps", "Buy bps",
                            "Breakeven bps", "Trigger price",
                            "Rebuy target",
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
                            }}
                        >
                            <Cell mono bold color={palette.accent}>
                                {r.ticker}
                            </Cell>
                            <Cell>{fmtPct(r.weight_pct, 1)}</Cell>
                            <Cell>
                                {r.currency} {fmtNum(r.current_price_native, 2)}
                            </Cell>
                            <Cell>{fmtNum(r.f_s_bps, 1)}</Cell>
                            <Cell>{fmtNum(r.f_b_bps, 1)}</Cell>
                            <Cell bold color={_bpsColor(r.breakeven_bps)}>
                                {fmtNum(r.breakeven_bps, 1)}
                            </Cell>
                            <Cell>
                                {r.currency}{" "}
                                {fmtNum(r.suggested_trigger_price_native, 2)}
                            </Cell>
                            <Cell>
                                {r.currency}{" "}
                                {fmtNum(r.suggested_rebuy_price_native, 2)}
                            </Cell>
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
