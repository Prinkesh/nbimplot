"use client";

import { useEffect } from "react";

export default function DemoLoader() {
  useEffect(() => {
    let disposed = false;
    const basePath = process.env.NEXT_PUBLIC_BASE_PATH || "";
    const version = process.env.NEXT_PUBLIC_DEMO_VERSION || "local";
    import(/* webpackIgnore: true */ `${basePath}/demo.js?v=${encodeURIComponent(version)}`).catch((error) => {
      if (disposed) return;
      console.error("Failed to load nbimplot demo module", error);
      const mode = document.querySelector("#mode");
      if (mode) mode.textContent = "failed";
    });
    return () => {
      disposed = true;
      window.__nbimplotExamplesDemo?.dispose?.();
    };
  }, []);

  return null;
}
