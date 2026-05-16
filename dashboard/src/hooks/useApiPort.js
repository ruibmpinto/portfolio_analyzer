// Resolves the sidecar's listening port.
//
// In production the Rust shell emits `api-port-ready` once the
// PyInstaller child has printed its chosen port. In dev (when
// the browser is just hitting `vite`) we fall back to a static
// VITE_DASHBOARD_API_PORT env var or 8000.

import { useEffect, useState } from "react";
import { setApiPort } from "../api/client.js";

export function useApiPort() {
    const [port, setPort] = useState(() => {
        const env = import.meta.env.VITE_DASHBOARD_API_PORT;
        if (env) {
            setApiPort(Number(env));
            return Number(env);
        }
        return null;
    });

    useEffect(() => {
        let unlisten = null;
        let cancelled = false;
        // Tauri event API is only available when running inside
        // the shell; in plain Vite this import fails silently.
        // The `@vite-ignore` annotation tells Vite's static
        // analyzer to leave the dynamic import alone so a
        // missing @tauri-apps/api package does not break dev.
        const eventModule = "@tauri-apps/api/event";
        const tauriModule = "@tauri-apps/api/tauri";
        import(/* @vite-ignore */ eventModule)
            .then(({ listen }) =>
                listen("api-port-ready", (evt) => {
                    if (cancelled) return;
                    const next = Number(evt.payload?.port);
                    if (!Number.isFinite(next)) return;
                    setApiPort(next);
                    setPort(next);
                }),
            )
            .then((u) => {
                if (typeof u === "function") unlisten = u;
            })
            .catch(() => {
                /* not running under Tauri; ignore */
            });

        // Also poll the IPC command once on mount, in case the
        // event fired before listeners registered.
        import(/* @vite-ignore */ tauriModule)
            .then(({ invoke }) => invoke("get_api_port"))
            .then((p) => {
                if (cancelled || !p) return;
                setApiPort(Number(p));
                setPort(Number(p));
            })
            .catch(() => {
                /* dev mode */
            });

        return () => {
            cancelled = true;
            if (unlisten) unlisten();
        };
    }, []);

    return port;
}
