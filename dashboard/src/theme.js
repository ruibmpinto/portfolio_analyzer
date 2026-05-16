// Shared design tokens for the dashboard.
//
// Kept as plain JS so existing inline-style components can keep
// using them without pulling in CSS-in-JS or tailwind. The
// scenario colour table is preserved from the legacy hardcoded
// App.jsx so the Strategies tab carries over identically.

export const palette = {
    bg: "#0f172a",
    card: "#1e293b",
    cardAlt: "#172033",
    border: "#334155",
    text: "#e2e8f0",
    textMuted: "#94a3b8",
    textFaint: "#64748b",
    accent: "#3b82f6",
    success: "#10b981",
    warn: "#f59e0b",
    danger: "#ef4444",
};

export const tooltipStyle = {
    background: palette.bg,
    border: `1px solid ${palette.border}`,
    borderRadius: 8,
    color: palette.text,
};

export const strategyColors = {
    mvo: { main: "#3b82f6", light: "#93c5fd", label: "MVO" },
    equal_weight: {
        main: "#10b981",
        light: "#6ee7b7",
        label: "Equal-Weight",
    },
    min_variance: {
        main: "#ef4444",
        light: "#fca5a5",
        label: "Min-Variance",
    },
    inverse_vol: { main: "#f59e0b", light: "#fcd34d", label: "Inverse-Vol" },
    max_growth: { main: "#ef4444", light: "#fca5a5", label: "Max-Growth" },
    min_risk: { main: "#10b981", light: "#6ee7b7", label: "Min-Risk" },
    risk_adjusted: {
        main: "#3b82f6",
        light: "#93c5fd",
        label: "Risk-Adjusted",
    },
};

export function colorFor(strategyName) {
    return (
        strategyColors[strategyName] || {
            main: palette.accent,
            light: "#bfdbfe",
            label: strategyName,
        }
    );
}

export function fmtCHF(value) {
    if (value === null || value === undefined || Number.isNaN(value)) {
        return "—";
    }
    return `CHF ${Math.round(value).toLocaleString("de-CH")}`;
}

export function fmtPct(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(value)) {
        return "—";
    }
    return `${value.toFixed(digits)}%`;
}

export function fmtNum(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(value)) {
        return "—";
    }
    return value.toLocaleString("de-CH", {
        maximumFractionDigits: digits,
        minimumFractionDigits: digits,
    });
}

export function fmtDate(value) {
    if (!value) return "—";
    return value.slice(0, 10);
}
