// Reusable KPI tile used by Overview and the strategy header.

import { palette } from "../theme.js";

export default function KpiCard({ label, value, color, sub }) {
    const accent = color || palette.accent;
    return (
        <div
            style={{
                background: palette.card,
                borderLeft: `3px solid ${accent}`,
                borderRadius: 8,
                padding: "12px 14px",
            }}
        >
            <div style={{ color: palette.textMuted, fontSize: 11 }}>
                {label}
            </div>
            <div
                style={{
                    color: accent,
                    fontSize: 20,
                    fontWeight: 700,
                    marginTop: 4,
                }}
            >
                {value}
            </div>
            {sub && (
                <div
                    style={{
                        color: palette.textFaint,
                        fontSize: 11,
                        marginTop: 2,
                    }}
                >
                    {sub}
                </div>
            )}
        </div>
    );
}
