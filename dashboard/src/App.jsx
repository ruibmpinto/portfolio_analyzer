// Top-level dashboard shell — routes between Overview, Ratios,
// Holdings, Strategies, and Sells. Drives the API-port handshake
// on mount.

import { useState } from "react";
import Holdings from "./pages/Holdings.jsx";
import Overview from "./pages/Overview.jsx";
import Ratios from "./pages/Ratios.jsx";
import Rules from "./pages/Rules.jsx";
import SafeHarbor from "./pages/SafeHarbor.jsx";
import Sells from "./pages/Sells.jsx";
import Stops from "./pages/Stops.jsx";
import Strategies from "./pages/Strategies.jsx";
import { useApiPort } from "./hooks/useApiPort.js";
import { apiPost } from "./api/client.js";
import { palette } from "./theme.js";

const pages = [
    { id: "overview", label: "Overview", Component: Overview },
    { id: "ratios", label: "Ratios", Component: Ratios },
    { id: "holdings", label: "Holdings", Component: Holdings },
    { id: "strategies", label: "Strategies", Component: Strategies },
    { id: "sells", label: "Sells", Component: Sells },
    { id: "stops", label: "Stops", Component: Stops },
    { id: "rules", label: "Rules", Component: Rules },
    { id: "safe_harbor", label: "Safe Harbor", Component: SafeHarbor },
];

export default function App() {
    const port = useApiPort();
    const [active, setActive] = useState("overview");
    const [refreshing, setRefreshing] = useState(false);
    const Page = pages.find((p) => p.id === active)?.Component || Overview;

    const onRefresh = async () => {
        setRefreshing(true);
        try {
            await apiPost("/api/refresh");
            // Force the active page to remount so every hook
            // fetches fresh data after the cache flush.
            setActive((prev) => prev);
            window.location.reload();
        } finally {
            setRefreshing(false);
        }
    };

    return (
        <div
            style={{
                minHeight: "100vh",
                background: palette.bg,
                color: palette.text,
                fontFamily:
                    "system-ui, -apple-system, 'SF Pro Text', sans-serif",
            }}
        >
            <div style={{ maxWidth: 1320, margin: "0 auto", padding: 20 }}>
                <header
                    style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        marginBottom: 16,
                    }}
                >
                    <div>
                        <h1
                            style={{
                                margin: 0,
                                fontSize: 22,
                                color: palette.text,
                            }}
                        >
                            Portfolio
                        </h1>
                        <div
                            style={{
                                color: palette.textFaint,
                                fontSize: 11,
                                marginTop: 2,
                            }}
                        >
                            Sidecar:{" "}
                            {port
                                ? `127.0.0.1:${port}`
                                : "not connected (using fallback)"}
                        </div>
                    </div>
                    <button
                        onClick={onRefresh}
                        disabled={refreshing}
                        style={{
                            background: palette.card,
                            color: palette.text,
                            border: `1px solid ${palette.border}`,
                            borderRadius: 6,
                            padding: "8px 14px",
                            fontSize: 12,
                            fontWeight: 600,
                            cursor: refreshing ? "wait" : "pointer",
                        }}
                    >
                        {refreshing ? "Refreshing…" : "Refresh data"}
                    </button>
                </header>

                <nav
                    style={{
                        display: "flex",
                        gap: 4,
                        background: palette.card,
                        padding: 4,
                        borderRadius: 8,
                        marginBottom: 16,
                    }}
                >
                    {pages.map((p) => (
                        <button
                            key={p.id}
                            onClick={() => setActive(p.id)}
                            style={{
                                background:
                                    active === p.id
                                        ? palette.accent
                                        : "transparent",
                                color:
                                    active === p.id
                                        ? "white"
                                        : palette.textMuted,
                                border: "none",
                                padding: "8px 16px",
                                borderRadius: 6,
                                fontSize: 13,
                                fontWeight: 600,
                                cursor: "pointer",
                            }}
                        >
                            {p.label}
                        </button>
                    ))}
                </nav>

                <Page />
            </div>
        </div>
    );
}
