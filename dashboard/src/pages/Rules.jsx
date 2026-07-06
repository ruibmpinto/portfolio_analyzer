// Rules page — sell-and-rebuy breakeven formula plus four
// per-market example rows. Consumes /api/rules.

import { useEndpoint } from "../hooks/useEndpoint.js";
import { fmtNum, palette } from "../theme.js";

export default function Rules() {
    const { data, loading, error } = useEndpoint("/api/rules");
    if (loading)
        return (
            <div style={{ color: palette.textMuted, padding: 24 }}>
                Loading rules…
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
    const formula = data?.formula;
    const examples = data?.examples || [];
    const notes = data?.operational_notes || [];
    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <FormulaCard formula={formula} />
            <ExamplesCard examples={examples} />
            <NotesCard notes={notes} />
        </div>
    );
}

function FormulaCard({ formula }) {
    if (!formula) return null;
    return (
        <div
            style={{
                background: palette.card,
                borderRadius: 10,
                padding: 16,
            }}
        >
            <SectionHeader
                title="Breakeven formula"
                sub="Minimum price gap for a sell-and-rebuy round-trip to net zero on fees + spread"
            />
            <div style={{ color: palette.textMuted, fontSize: 13, lineHeight: 1.5 }}>
                {formula.summary}
            </div>
            <div
                style={{
                    marginTop: 12,
                    marginBottom: 12,
                    fontFamily: "monospace",
                    fontSize: 13,
                    color: palette.accent,
                    background: palette.bg,
                    padding: "10px 14px",
                    borderRadius: 6,
                    border: `1px solid ${palette.border}`,
                }}
            >
                {formula.expression}
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {(formula.components || []).map((c) => (
                    <div
                        key={c.name}
                        style={{
                            display: "grid",
                            gridTemplateColumns: "160px 1fr",
                            gap: 12,
                            fontSize: 12,
                        }}
                    >
                        <code
                            style={{
                                color: palette.accent,
                                fontFamily: "monospace",
                            }}
                        >
                            {c.name}
                        </code>
                        <span style={{ color: palette.textMuted }}>
                            {c.meaning}
                        </span>
                    </div>
                ))}
            </div>
        </div>
    );
}

function ExamplesCard({ examples }) {
    return (
        <div
            style={{
                background: palette.card,
                borderRadius: 10,
                padding: 16,
            }}
        >
            <SectionHeader
                title="Per-market examples"
                sub="Illustrative round numbers. Breakeven bps are venue-specific and independent of the specific price / share count."
            />
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
                                "Ticker", "Market", "Currency",
                                "Shares", "Price",
                                "Sell bps", "Buy bps", "Spread bps",
                                "Breakeven bps", "Trigger bps (2×)",
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
                        {examples.map((ex, i) => (
                            <tr
                                key={ex.ticker}
                                style={{
                                    background:
                                        i % 2 === 0
                                            ? palette.card
                                            : palette.cardAlt,
                                }}
                            >
                                <Cell mono bold color={palette.accent}>
                                    {ex.ticker}
                                </Cell>
                                <Cell>{ex.market_label}</Cell>
                                <Cell>{ex.currency}</Cell>
                                <Cell>{fmtNum(ex.shares, 0)}</Cell>
                                <Cell>{fmtNum(ex.price_native, 2)}</Cell>
                                <Cell>{fmtNum(ex.f_s_bps, 1)}</Cell>
                                <Cell>{fmtNum(ex.f_b_bps, 1)}</Cell>
                                <Cell>{fmtNum(ex.half_spread_bps, 1)}</Cell>
                                <Cell bold>
                                    {fmtNum(ex.breakeven_bps, 1)}
                                </Cell>
                                <Cell>
                                    {fmtNum(ex.suggested_trigger_bps, 1)}
                                </Cell>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

function NotesCard({ notes }) {
    if (!notes.length) return null;
    return (
        <div
            style={{
                background: palette.card,
                borderRadius: 10,
                padding: 16,
                borderLeft: `3px solid ${palette.warn}`,
            }}
        >
            <SectionHeader
                title="Operational notes"
                sub="IBKR configuration to make the strategy behave correctly"
            />
            <ul
                style={{
                    margin: 0,
                    paddingLeft: 20,
                    color: palette.textMuted,
                    fontSize: 13,
                    lineHeight: 1.6,
                }}
            >
                {notes.map((note, i) => (
                    <li key={i}>{note}</li>
                ))}
            </ul>
        </div>
    );
}

function SectionHeader({ title, sub }) {
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
