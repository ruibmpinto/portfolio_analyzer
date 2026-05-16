// Holdings page — per-position table with native + CHF prices,
// fees footer, and annual-returns comparison bar chart.

import {
    Bar,
    BarChart,
    CartesianGrid,
    Legend,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from "recharts";
import { useEndpoint } from "../hooks/useEndpoint.js";
import {
    fmtCHF,
    fmtNum,
    fmtPct,
    palette,
    tooltipStyle,
} from "../theme.js";

export default function Holdings() {
    const { data, loading, error } = useEndpoint("/api/holdings");

    if (loading) {
        return (
            <div style={{ color: palette.textMuted, padding: 24 }}>
                Loading holdings…
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
    const payload = data || {};
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
                    label="Current Holdings"
                    sub={`Total: ${fmtCHF(payload.total_value_chf)}`}
                />
                <HoldingsTable rows={payload.rows || []} />
            </div>

            <div
                style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: 16,
                    marginBottom: 16,
                }}
            >
                <FeesCard fees={payload.fees} />
                <IncomeCard
                    dividends={payload.dividends}
                    taxes={payload.taxes}
                />
            </div>

            <AnnualReturnsCard rows={payload.annual_returns || []} />
        </div>
    );
}

function Header({ label, sub }) {
    return (
        <div style={{ marginBottom: 10 }}>
            <div
                style={{
                    color: palette.textMuted,
                    fontSize: 13,
                    fontWeight: 600,
                }}
            >
                {label}
            </div>
            <div style={{ color: palette.textFaint, fontSize: 11 }}>
                {sub}
            </div>
        </div>
    );
}

function HoldingsTable({ rows }) {
    if (!rows.length) {
        return (
            <div style={{ color: palette.textFaint, padding: 12 }}>
                No open positions.
            </div>
        );
    }
    return (
        <div
            style={{
                border: `1px solid ${palette.border}`,
                borderRadius: 8,
                overflow: "auto",
                maxHeight: 420,
            }}
        >
            <table style={{ width: "100%", fontSize: 12 }}>
                <thead>
                    <tr style={{ background: palette.bg }}>
                        {[
                            "Ticker",
                            "Shares",
                            "CCY",
                            "Price (native)",
                            "Price (CHF)",
                            "Value (CHF)",
                            "Weight",
                            "Sector",
                        ].map((h) => (
                            <th
                                key={h}
                                style={{
                                    textAlign: "left",
                                    padding: "8px 12px",
                                    color: palette.textFaint,
                                    fontWeight: 500,
                                    position: "sticky",
                                    top: 0,
                                    background: palette.bg,
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
                                {Number.isInteger(r.shares)
                                    ? fmtNum(r.shares, 0)
                                    : fmtNum(r.shares, 3)}
                            </td>
                            <td
                                style={{
                                    padding: "6px 12px",
                                    color: palette.textFaint,
                                }}
                            >
                                {r.currency}
                            </td>
                            <td
                                style={{
                                    padding: "6px 12px",
                                    fontFamily: "monospace",
                                    color: palette.textMuted,
                                }}
                            >
                                {fmtNum(r.price_native)}
                            </td>
                            <td
                                style={{
                                    padding: "6px 12px",
                                    fontFamily: "monospace",
                                    color: palette.textMuted,
                                }}
                            >
                                {fmtNum(r.price_chf)}
                            </td>
                            <td
                                style={{
                                    padding: "6px 12px",
                                    fontFamily: "monospace",
                                    color: palette.text,
                                }}
                            >
                                {fmtCHF(r.value_chf)}
                            </td>
                            <td
                                style={{
                                    padding: "6px 12px",
                                    color: palette.textMuted,
                                }}
                            >
                                {fmtPct(r.weight_pct)}
                            </td>
                            <td
                                style={{
                                    padding: "6px 12px",
                                    color: palette.textFaint,
                                }}
                            >
                                {r.sector}
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}

function FeesCard({ fees }) {
    if (!fees) return null;
    return (
        <div
            style={{
                background: palette.card,
                borderRadius: 10,
                padding: 16,
            }}
        >
            <Header
                label="Cumulative Fees (lifetime)"
                sub="Broker commissions + auto-FX fees per trade currency"
            />
            <CurrencyBreakdownTable
                bucket={fees}
                amountLabel="Native amount"
                totalColor={palette.danger}
                totalLabel="Total fees paid (CHF)"
            />
        </div>
    );
}


function IncomeCard({ dividends, taxes }) {
    if (!dividends && !taxes) return null;
    const net =
        (dividends?.total_chf || 0) + (taxes?.total_chf || 0);
    return (
        <div
            style={{
                background: palette.card,
                borderRadius: 10,
                padding: 16,
            }}
        >
            <Header
                label="Lifetime Income"
                sub="Dividends received and withholding tax paid"
            />

            <SectionLabel color={palette.success}>
                Dividends received
            </SectionLabel>
            <CurrencyBreakdownTable
                bucket={dividends}
                amountLabel="Native amount"
                totalColor={palette.success}
                totalLabel="Dividends (CHF)"
            />

            <div style={{ height: 14 }} />

            <SectionLabel color={palette.danger}>
                Withholding tax
            </SectionLabel>
            <CurrencyBreakdownTable
                bucket={taxes}
                amountLabel="Native amount"
                totalColor={palette.danger}
                totalLabel="Taxes (CHF)"
            />

            <div
                style={{
                    marginTop: 14,
                    padding: "10px 12px",
                    borderTop: `1px solid ${palette.border}`,
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                }}
            >
                <span
                    style={{
                        color: palette.text,
                        fontWeight: 600,
                    }}
                >
                    Net income (CHF)
                </span>
                <span
                    style={{
                        fontFamily: "monospace",
                        color:
                            net >= 0 ? palette.success : palette.danger,
                        fontWeight: 700,
                        fontSize: 15,
                    }}
                >
                    {fmtCHF(net)}
                </span>
            </div>
        </div>
    );
}


function SectionLabel({ color, children }) {
    return (
        <div
            style={{
                color: color,
                fontSize: 12,
                fontWeight: 600,
                marginBottom: 6,
                textTransform: "uppercase",
                letterSpacing: 0.5,
            }}
        >
            {children}
        </div>
    );
}


// Renders one row per currency. Each row makes the meaning of
// every number explicit: currency code, native amount, FX rate
// used, and the CHF-equivalent. This replaces the previous
// 2-column grid where "USD 478.12 CHF 10.20" wrapped on a
// single line and read like a conversion.
function CurrencyBreakdownTable({
    bucket,
    amountLabel,
    totalColor,
    totalLabel,
}) {
    const entries = Object.entries(bucket?.by_currency || {});
    const fxRates = bucket?.fx_rates_used || {};
    if (entries.length === 0) {
        return (
            <div
                style={{
                    color: palette.textFaint,
                    fontSize: 12,
                    padding: "4px 0 8px",
                }}
            >
                none
            </div>
        );
    }
    return (
        <div
            style={{
                border: `1px solid ${palette.border}`,
                borderRadius: 6,
                overflow: "hidden",
                fontSize: 12,
            }}
        >
            <table style={{ width: "100%" }}>
                <thead>
                    <tr style={{ background: palette.bg }}>
                        <th
                            style={{
                                textAlign: "left",
                                padding: "6px 10px",
                                color: palette.textFaint,
                                fontWeight: 500,
                            }}
                        >
                            Currency
                        </th>
                        <th
                            style={{
                                textAlign: "right",
                                padding: "6px 10px",
                                color: palette.textFaint,
                                fontWeight: 500,
                            }}
                        >
                            {amountLabel}
                        </th>
                        <th
                            style={{
                                textAlign: "right",
                                padding: "6px 10px",
                                color: palette.textFaint,
                                fontWeight: 500,
                            }}
                        >
                            FX → CHF
                        </th>
                        <th
                            style={{
                                textAlign: "right",
                                padding: "6px 10px",
                                color: palette.textFaint,
                                fontWeight: 500,
                            }}
                        >
                            CHF equivalent
                        </th>
                    </tr>
                </thead>
                <tbody>
                    {entries.map(([ccy, amount], i) => {
                        const rate = fxRates[ccy] ?? 1.0;
                        return (
                            <tr
                                key={ccy}
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
                                        padding: "6px 10px",
                                        color: palette.textMuted,
                                        fontWeight: 600,
                                    }}
                                >
                                    {ccy}
                                </td>
                                <td
                                    style={{
                                        padding: "6px 10px",
                                        fontFamily: "monospace",
                                        textAlign: "right",
                                        color: palette.text,
                                    }}
                                >
                                    {fmtNum(amount)}
                                </td>
                                <td
                                    style={{
                                        padding: "6px 10px",
                                        fontFamily: "monospace",
                                        textAlign: "right",
                                        color: palette.textFaint,
                                    }}
                                >
                                    {ccy === "CHF"
                                        ? "—"
                                        : fmtNum(rate, 4)}
                                </td>
                                <td
                                    style={{
                                        padding: "6px 10px",
                                        fontFamily: "monospace",
                                        textAlign: "right",
                                        color: palette.text,
                                    }}
                                >
                                    {fmtCHF(amount * rate)}
                                </td>
                            </tr>
                        );
                    })}
                    <tr
                        style={{
                            background: palette.bg,
                            borderTop: `2px solid ${palette.border}`,
                        }}
                    >
                        <td
                            colSpan={3}
                            style={{
                                padding: "6px 10px",
                                color: palette.textMuted,
                                fontWeight: 600,
                            }}
                        >
                            {totalLabel}
                        </td>
                        <td
                            style={{
                                padding: "6px 10px",
                                fontFamily: "monospace",
                                textAlign: "right",
                                color: totalColor,
                                fontWeight: 700,
                            }}
                        >
                            {fmtCHF(bucket?.total_chf || 0)}
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
    );
}


function AnnualReturnsCard({ rows }) {
    if (!rows.length) {
        return (
            <div
                style={{
                    background: palette.card,
                    borderRadius: 10,
                    padding: 16,
                    color: palette.textFaint,
                }}
            >
                No annual return data.
            </div>
        );
    }
    const benchmarkKeys = Object.keys(rows[0]).filter(
        (k) => k.endsWith("_pct") && k !== "portfolio_pct",
    );
    return (
        <div
            style={{
                background: palette.card,
                borderRadius: 10,
                padding: 16,
            }}
        >
            <Header
                label="Annual returns"
                sub="Portfolio vs benchmarks, calendar-year close-to-close"
            />
            <ResponsiveContainer width="100%" height={240}>
                <BarChart data={rows} margin={{ left: 0, right: 8 }}>
                    <CartesianGrid stroke={palette.border} strokeDasharray="3 3" vertical={false} />
                    <XAxis
                        dataKey="year"
                        tick={{ fill: palette.textFaint, fontSize: 11 }}
                    />
                    <YAxis
                        tick={{ fill: palette.textFaint, fontSize: 10 }}
                        tickFormatter={(v) => `${v.toFixed(0)}%`}
                    />
                    <Tooltip
                        contentStyle={tooltipStyle}
                        formatter={(v) =>
                            v === null || v === undefined ? "—" : `${v.toFixed(1)}%`
                        }
                    />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Bar
                        dataKey="portfolio_pct"
                        fill={palette.accent}
                        name="Portfolio"
                    />
                    {benchmarkKeys.map((k, i) => (
                        <Bar
                            key={k}
                            dataKey={k}
                            fill={i === 0 ? palette.success : palette.warn}
                            name={k.replace("_pct", "")}
                        />
                    ))}
                </BarChart>
            </ResponsiveContainer>
        </div>
    );
}
