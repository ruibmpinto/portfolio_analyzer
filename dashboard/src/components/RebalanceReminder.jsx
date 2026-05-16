// Banner that appears on the Overview page when the configured
// rebalance window has elapsed. Clicking "Mark rebalanced" hits
// the /api/rebalance-done endpoint and reloads the meta payload.

import { useState } from "react";
import { apiPost } from "../api/client.js";
import { palette } from "../theme.js";

export default function RebalanceReminder({ meta, onMarked }) {
    const [busy, setBusy] = useState(false);
    const [strategy, setStrategy] = useState("mvo");

    if (!meta?.rebalance_due) return null;

    const days = meta.days_since_rebalance;
    const reason =
        days === null
            ? "No rebalance has ever been logged."
            : `Last rebalance was ${days} days ago — ` +
              `over the ${meta.rebalance_due_threshold_days}-day threshold.`;

    const markDone = async () => {
        setBusy(true);
        try {
            await apiPost("/api/rebalance-done", {
                strategy_name: strategy,
            });
            if (onMarked) await onMarked();
        } finally {
            setBusy(false);
        }
    };

    return (
        <div
            style={{
                background: "#7f1d1d20",
                border: `1px solid ${palette.danger}`,
                color: palette.text,
                borderRadius: 10,
                padding: "12px 16px",
                marginBottom: 16,
                display: "flex",
                gap: 12,
                alignItems: "center",
                justifyContent: "space-between",
                flexWrap: "wrap",
            }}
        >
            <div>
                <div
                    style={{
                        fontWeight: 600,
                        color: palette.danger,
                        marginBottom: 2,
                    }}
                >
                    Rebalance due
                </div>
                <div style={{ color: palette.textMuted, fontSize: 13 }}>
                    {reason}
                </div>
            </div>
            <div
                style={{
                    display: "flex",
                    gap: 8,
                    alignItems: "center",
                }}
            >
                <select
                    value={strategy}
                    onChange={(e) => setStrategy(e.target.value)}
                    style={{
                        background: palette.bg,
                        color: palette.text,
                        border: `1px solid ${palette.border}`,
                        borderRadius: 6,
                        padding: "6px 10px",
                    }}
                >
                    <option value="mvo">MVO</option>
                    <option value="equal_weight">Equal-Weight</option>
                    <option value="min_variance">Min-Variance</option>
                </select>
                <button
                    onClick={markDone}
                    disabled={busy}
                    style={{
                        background: palette.danger,
                        color: "white",
                        border: "none",
                        borderRadius: 6,
                        padding: "8px 14px",
                        fontWeight: 600,
                        cursor: busy ? "wait" : "pointer",
                    }}
                >
                    {busy ? "Saving…" : "Mark rebalanced"}
                </button>
            </div>
        </div>
    );
}
