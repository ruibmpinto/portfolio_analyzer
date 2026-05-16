// Overview page — KPI cards, benchmark performance chart,
// rebalance reminder, and screener-cache freshness footer.

import BenchmarkChart from "../components/BenchmarkChart.jsx";
import KpiCard from "../components/KpiCard.jsx";
import RebalanceReminder from "../components/RebalanceReminder.jsx";
import { useEndpoint } from "../hooks/useEndpoint.js";
import {
    fmtCHF,
    fmtDate,
    palette,
} from "../theme.js";

export default function Overview() {
    const overview = useEndpoint("/api/overview");
    const meta = useEndpoint("/api/meta");

    if (overview.loading || meta.loading) {
        return <Loading />;
    }
    if (overview.error) {
        return <ErrorPanel error={overview.error} />;
    }

    const data = overview.data || {};
    return (
        <div>
            <RebalanceReminder
                meta={meta.data}
                onMarked={meta.reload}
            />

            <div style={{ marginBottom: 16 }}>
                <KpiCard
                    label="Total NAV"
                    value={fmtCHF(data.total_value_chf)}
                    color={palette.accent}
                    sub="See the Ratios tab for return and risk metrics"
                />
            </div>

            <div
                style={{
                    background: palette.card,
                    borderRadius: 10,
                    padding: 16,
                }}
            >
                <div
                    style={{
                        color: palette.textMuted,
                        fontSize: 13,
                        marginBottom: 8,
                    }}
                >
                    Portfolio vs Benchmarks (rebased to 100)
                </div>
                <BenchmarkChart
                    series={data.performance_series}
                    labels={data.benchmark_labels || []}
                />
            </div>

            <div
                style={{
                    color: palette.textFaint,
                    fontSize: 11,
                    marginTop: 12,
                    textAlign: "right",
                }}
            >
                Screener cache last refreshed:{" "}
                {fmtDate(meta.data?.screener?.latest_updated_utc)}
            </div>
        </div>
    );
}

function Loading() {
    return (
        <div style={{ color: palette.textMuted, padding: 24 }}>
            Loading portfolio data…
        </div>
    );
}

function ErrorPanel({ error }) {
    return (
        <div
            style={{
                background: "#7f1d1d20",
                color: palette.danger,
                padding: 16,
                borderRadius: 8,
                fontFamily: "monospace",
                fontSize: 13,
            }}
        >
            Error: {error.message}
        </div>
    );
}
