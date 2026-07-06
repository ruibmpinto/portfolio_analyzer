// Strategies page — carries over the four legacy panels
// (radar, comparison, allocation bar, fan chart, P&L / DD
// distributions, sells+buys) but rebuilds them off live
// /api/strategies output.

import { useState } from "react";
import AllocationBar from "../components/AllocationBar.jsx";
import ComparisonTable from "../components/ComparisonTable.jsx";
import {
    DrawdownHistogram,
    DrawdownOverlay,
} from "../components/DrawdownHistogram.jsx";
import KpiCard from "../components/KpiCard.jsx";
import NavFanChart from "../components/NavFanChart.jsx";
import { PnlHistogram, PnlOverlay } from "../components/PnlHistogram.jsx";
import RadarPanel from "../components/RadarPanel.jsx";
import { useEndpoint } from "../hooks/useEndpoint.js";
import {
    colorFor,
    fmtCHF,
    fmtPct,
    palette,
} from "../theme.js";

const tabs = [
    { id: "overview", label: "Comparison" },
    { id: "pnl", label: "P&L Distribution" },
    { id: "drawdown", label: "Drawdown" },
    { id: "paths", label: "NAV Projection" },
    { id: "actions", label: "Actions" },
];

export default function Strategies() {
    const { data, loading, error } = useEndpoint("/api/strategies");
    const [selected, setSelected] = useState(null);
    const [tab, setTab] = useState("overview");

    if (loading)
        return (
            <div style={{ color: palette.textMuted, padding: 24 }}>
                Running Monte Carlo per strategy… this may take a
                few seconds on first load.
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
    const strategies = data?.strategies || [];
    if (!strategies.length) {
        return (
            <div
                style={{
                    color: palette.textFaint,
                    padding: 24,
                }}
            >
                No strategies produced output. Errors:{" "}
                {JSON.stringify(data?.errors)}
            </div>
        );
    }
    const current = strategies.find((s) => s.name === selected) || strategies[0];
    const cur = colorFor(current.name);

    return (
        <div>
            {/* Strategy selector */}
            <div
                style={{
                    display: "flex",
                    gap: 8,
                    marginBottom: 12,
                    flexWrap: "wrap",
                }}
            >
                {strategies.map((s) => {
                    const c = colorFor(s.name);
                    const active = s.name === current.name;
                    return (
                        <button
                            key={s.name}
                            onClick={() => setSelected(s.name)}
                            style={{
                                background: active ? c.main : palette.card,
                                color: active ? "white" : palette.textMuted,
                                border: active
                                    ? "none"
                                    : `1px solid ${palette.border}`,
                                padding: "8px 14px",
                                borderRadius: 8,
                                fontSize: 13,
                                fontWeight: 600,
                                cursor: "pointer",
                            }}
                        >
                            {c.label}
                        </button>
                    );
                })}
            </div>

            {/* Per-strategy KPI strip */}
            <div
                style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(6, 1fr)",
                    gap: 10,
                    marginBottom: 12,
                }}
            >
                <KpiCard
                    label="Median terminal NAV"
                    value={fmtCHF(current.mc_summary.terminal_p50)}
                    sub={`± ${fmtCHF(current.mc_summary.terminal_se_chf)} SE`}
                    color={cur.main}
                />
                <KpiCard
                    label="Mean P&L"
                    value={fmtPct(current.mc_summary.mu_ann_pct)}
                    sub={`± ${fmtPct(current.mc_summary.mu_se_pct)} SE`}
                    color={palette.success}
                />
                <KpiCard
                    label="P(Loss)"
                    value={fmtPct(current.mc_summary.prob_loss * 100)}
                    sub={`± ${fmtPct(current.mc_summary.prob_loss_se * 100)} SE`}
                    color={
                        current.mc_summary.prob_loss > 0.25
                            ? palette.danger
                            : palette.success
                    }
                />
                <KpiCard
                    label="VaR 5%"
                    value={fmtPct(current.mc_summary.var_5_pct)}
                    color={palette.danger}
                />
                <KpiCard
                    label="Median MaxDD"
                    value={fmtPct(
                        current.mc_summary.max_drawdown_p50 * 100,
                    )}
                    color={palette.warn}
                />
                <KpiCard
                    label="MC paths"
                    value={(current.mc_summary.n_paths || 0).toLocaleString()}
                    sub={`${current.mc_summary.horizon_days} trading days`}
                    color={palette.textMuted}
                />
            </div>

            {/* Tab bar */}
            <div
                style={{
                    display: "flex",
                    gap: 4,
                    padding: 4,
                    background: palette.card,
                    borderRadius: 8,
                    marginBottom: 12,
                }}
            >
                {tabs.map((t) => (
                    <button
                        key={t.id}
                        onClick={() => setTab(t.id)}
                        style={{
                            background:
                                tab === t.id ? palette.accent : "transparent",
                            color:
                                tab === t.id ? "white" : palette.textMuted,
                            border: "none",
                            padding: "8px 14px",
                            borderRadius: 6,
                            fontWeight: 600,
                            cursor: "pointer",
                            fontSize: 12,
                        }}
                    >
                        {t.label}
                    </button>
                ))}
            </div>

            {tab === "overview" && (
                <ComparisonTab strategies={strategies} />
            )}
            {tab === "pnl" && <PnlTab strategies={strategies} current={current} />}
            {tab === "drawdown" && (
                <DrawdownTab strategies={strategies} current={current} />
            )}
            {tab === "paths" && <PathsTab current={current} />}
            {tab === "actions" && (
                <ActionsTab current={current} />
            )}

            <ReasoningCard strategy={current} />
            <MethodologyCard mc={data?.mc} current={current} />
        </div>
    );
}

function MethodologyCard({ mc, current }) {
    if (!mc?.methodology) return null;
    const m = mc.methodology;
    const cagrPct = current?.cagr_forecast_pct;
    const coverage = current?.cagr_forecast_coverage;
    return (
        <div
            style={{
                background: palette.card,
                borderRadius: 10,
                padding: 16,
                marginTop: 12,
                color: palette.textMuted,
                fontSize: 12,
                lineHeight: 1.5,
                borderLeft: `3px solid ${palette.accent}`,
            }}
        >
            <div
                style={{
                    color: palette.textFaint,
                    fontSize: 11,
                    marginBottom: 8,
                    textTransform: "uppercase",
                    letterSpacing: 0.5,
                }}
            >
                Monte Carlo methodology
            </div>
            <div style={{ marginBottom: 8 }}>
                <strong>Run config:</strong>{" "}
                {(mc.n_paths || 0).toLocaleString()} paths ·{" "}
                {mc.horizon_days} trading days ·{" "}
                CHF {mc.monthly_contribution_chf}/month contribution
                {cagrPct !== undefined && (
                    <>
                        {" · "}
                        3Y CAGR forecast {fmtPct(cagrPct)}
                        {coverage !== undefined && (
                            <> (coverage {fmtPct(coverage * 100, 0)})</>
                        )}
                    </>
                )}
            </div>
            <div style={{ marginBottom: 8 }}>{m.summary}</div>
            <div style={{ marginBottom: 4 }}>
                <strong>Distributions</strong>
            </div>
            <ul style={{ margin: 0, paddingLeft: 18, marginBottom: 8 }}>
                {m.distributions.map(([label, eq], i) => (
                    <li key={i}>
                        {label}:{" "}
                        <code style={{ color: palette.textFaint }}>{eq}</code>
                    </li>
                ))}
            </ul>
            <div style={{ marginBottom: 4 }}>
                <strong>Equations</strong>
            </div>
            <ul style={{ margin: 0, paddingLeft: 18, marginBottom: 8 }}>
                {m.equations.map(([label, eq], i) => (
                    <li key={i}>
                        {label}:{" "}
                        <code style={{ color: palette.textFaint }}>{eq}</code>
                    </li>
                ))}
            </ul>
            <div style={{ marginBottom: 4 }}>
                <strong>Parameter clips</strong>
            </div>
            <ul style={{ margin: 0, paddingLeft: 18 }}>
                <li>
                    Student-t df:{" "}
                    [{m.parameters.student_t_df_clip[0]},{" "}
                    {m.parameters.student_t_df_clip[1]}]
                </li>
                <li>
                    Clayton theta:{" "}
                    [{m.parameters.clayton_theta_clip[0]},{" "}
                    {m.parameters.clayton_theta_clip[1]}]
                </li>
                <li>
                    Kendall tau:{" "}
                    [{m.parameters.kendall_tau_clip[0]},{" "}
                    {m.parameters.kendall_tau_clip[1]}]
                </li>
                <li>
                    Min history to fit:{" "}
                    {m.parameters.min_history_for_fit_days} days
                </li>
                <li>
                    Trading days / year:{" "}
                    {m.parameters.trading_days_per_year}
                </li>
            </ul>
        </div>
    );
}

function ComparisonTab({ strategies }) {
    return (
        <div
            style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 12,
            }}
        >
            <Panel title="Risk-Return Profile">
                <RadarPanel strategies={strategies} />
            </Panel>
            <Panel title="Head-to-Head Metrics">
                <ComparisonTable strategies={strategies} />
            </Panel>
            <div style={{ gridColumn: "1 / span 2" }}>
                <Panel title="Allocation per strategy">
                    <AllocationBar strategies={strategies} />
                </Panel>
            </div>
        </div>
    );
}

function PnlTab({ strategies, current }) {
    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <Panel title={`P&L distribution — ${colorFor(current.name).label}`}>
                <PnlHistogram strategy={current} />
            </Panel>
            <Panel title="All strategies overlaid">
                <PnlOverlay strategies={strategies} />
            </Panel>
        </div>
    );
}

function DrawdownTab({ strategies, current }) {
    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <Panel title={`Drawdown distribution — ${colorFor(current.name).label}`}>
                <DrawdownHistogram strategy={current} />
            </Panel>
            <Panel title="All strategies overlaid">
                <DrawdownOverlay strategies={strategies} />
            </Panel>
        </div>
    );
}

function PathsTab({ current }) {
    return (
        <Panel
            title={`NAV fan chart — ${colorFor(current.name).label}`}
            sub="Shaded bands: 5–95 and 25–75 percentiles. Solid: median."
        >
            <NavFanChart strategy={current} />
        </Panel>
    );
}

function ActionsTab({ current }) {
    const actions = current.actions || [];
    const buys = actions.filter((a) => a.action === "BUY");
    const reduces = actions.filter((a) => a.action === "REDUCE");
    return (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <ActionsTable title="BUYS" rows={buys} color={palette.success} />
            <ActionsTable
                title="REDUCES"
                rows={reduces}
                color={palette.danger}
            />
        </div>
    );
}

function ActionsTable({ title, rows, color }) {
    return (
        <Panel title={title}>
            {rows.length === 0 && (
                <div style={{ color: palette.textFaint, padding: 8 }}>
                    None.
                </div>
            )}
            {rows.length > 0 && (
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
                                    "Cost (CHF)",
                                    "Current %",
                                    "Target %",
                                    "Achievable %",
                                    "Note",
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
                                    <td
                                        style={{
                                            padding: "6px 12px",
                                            fontFamily: "monospace",
                                            fontWeight: 600,
                                            color: color,
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
                                        {r.shares}
                                    </td>
                                    <td
                                        style={{
                                            padding: "6px 12px",
                                            fontFamily: "monospace",
                                            color: palette.textMuted,
                                        }}
                                    >
                                        {fmtCHF(r.est_cost_chf)}
                                    </td>
                                    <td
                                        style={{
                                            padding: "6px 12px",
                                            color: palette.textMuted,
                                        }}
                                    >
                                        {fmtPct(r.current_wt_pct)}
                                    </td>
                                    <td
                                        style={{
                                            padding: "6px 12px",
                                            color: palette.textMuted,
                                        }}
                                    >
                                        {fmtPct(r.target_wt_pct)}
                                    </td>
                                    <td
                                        style={{
                                            padding: "6px 12px",
                                            color: palette.textMuted,
                                        }}
                                    >
                                        {fmtPct(r.achievable_wt_pct)}
                                    </td>
                                    <td
                                        style={{
                                            padding: "6px 12px",
                                            color: palette.textFaint,
                                        }}
                                    >
                                        {r.note}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </Panel>
    );
}

function ReasoningCard({ strategy }) {
    if (!strategy?.reasoning) return null;
    return (
        <div
            style={{
                background: palette.card,
                borderRadius: 10,
                padding: 16,
                marginTop: 12,
                color: palette.textMuted,
                fontSize: 13,
                lineHeight: 1.5,
                borderLeft: `3px solid ${colorFor(strategy.name).main}`,
            }}
        >
            <div
                style={{
                    color: palette.textFaint,
                    fontSize: 11,
                    marginBottom: 4,
                    textTransform: "uppercase",
                    letterSpacing: 0.5,
                }}
            >
                Strategy reasoning
            </div>
            {strategy.reasoning}
        </div>
    );
}

function Panel({ title, sub, children }) {
    return (
        <div
            style={{
                background: palette.card,
                borderRadius: 10,
                padding: 16,
            }}
        >
            <div style={{ marginBottom: 8 }}>
                <div
                    style={{
                        color: palette.textMuted,
                        fontSize: 13,
                        fontWeight: 600,
                    }}
                >
                    {title}
                </div>
                {sub && (
                    <div style={{ color: palette.textFaint, fontSize: 11 }}>
                        {sub}
                    </div>
                )}
            </div>
            {children}
        </div>
    );
}
