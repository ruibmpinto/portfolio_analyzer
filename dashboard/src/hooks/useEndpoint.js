// Generic data-fetching hook for the sidecar endpoints.
//
// Returns `{ data, error, loading, reload }`. The reload helper
// is used by the manual refresh button on the Overview page.

import { useCallback, useEffect, useState } from "react";
import { apiGet } from "../api/client.js";

export function useEndpoint(path, deps = []) {
    const [data, setData] = useState(null);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(true);

    const reload = useCallback(() => {
        let cancelled = false;
        setLoading(true);
        setError(null);
        apiGet(path)
            .then((payload) => {
                if (!cancelled) setData(payload);
            })
            .catch((err) => {
                if (!cancelled) setError(err);
            })
            .finally(() => {
                if (!cancelled) setLoading(false);
            });
        return () => {
            cancelled = true;
        };
    }, [path]);

    useEffect(reload, deps); // eslint-disable-line react-hooks/exhaustive-deps

    return { data, error, loading, reload };
}
