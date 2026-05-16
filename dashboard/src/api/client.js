// Typed fetch helpers for the dashboard sidecar.
//
// In production the Tauri shell pushes the chosen port via the
// `api-port-ready` event (see hooks/useApiPort.js). In `pnpm
// dev`, the dev server falls back to the DASHBOARD_API_PORT env
// var or the default 8000. Every fetch goes through `apiGet` /
// `apiPost` so we centralise the base-URL composition.

let cachedBase = null;

export function setApiPort(port) {
    cachedBase = `http://127.0.0.1:${port}`;
}

export function getApiBase() {
    if (cachedBase) return cachedBase;
    const envPort = import.meta.env.VITE_DASHBOARD_API_PORT;
    if (envPort) return `http://127.0.0.1:${envPort}`;
    return "http://127.0.0.1:8000";
}

export async function apiGet(path) {
    const res = await fetch(`${getApiBase()}${path}`);
    if (!res.ok) {
        throw new Error(
            `GET ${path} failed: ${res.status} ${res.statusText}`,
        );
    }
    return res.json();
}

export async function apiPost(path, body = {}) {
    const res = await fetch(`${getApiBase()}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    if (!res.ok) {
        throw new Error(
            `POST ${path} failed: ${res.status} ${res.statusText}`,
        );
    }
    return res.json();
}
