"use client";

import { useCallback, useRef } from "react";
import { useAnalysisStore } from "@/stores/analysis-store";
import { analyzeImage } from "@/lib/api";
import { ANALYSIS_STEPS } from "@/lib/constants";

export function useAnalysis() {
  const store = useAnalysisStore();
  const abortRef = useRef<AbortController | null>(null);

  // Extract stable setter functions (these don't change between renders)
  const { setStatus, setProgress, setResult, setError, reset: storeReset } =
    useAnalysisStore.getState();

  const simulateProgress = useCallback(
    (startProgress: number, endProgress: number, stepIndex: number) => {
      return new Promise<void>((resolve) => {
        const step = ANALYSIS_STEPS[stepIndex];
        if (!step) {
          resolve();
          return;
        }

        const duration = 300 + Math.random() * 200;
        const increment = (endProgress - startProgress) / 10;
        let current = startProgress;
        let ticks = 0;

        const interval = setInterval(() => {
          ticks++;
          current = Math.min(current + increment, endProgress);
          setProgress(Math.round(current), step.label);
          if (ticks >= 10) {
            clearInterval(interval);
            resolve();
          }
        }, duration / 10);
      });
    },
    [setProgress],
  );

  const analyze = useCallback(
    async (file: File) => {
      // Abort any previous in-flight analysis
      abortRef.current?.abort();
      abortRef.current = new AbortController();

      // Clear any previous result before starting
      setStatus("uploading");
      setProgress(0, "Uploading");

      try {
        // Simulate upload progress
        await simulateProgress(0, 10, 0);

        setStatus("analyzing");

        // Start actual API call (runs in parallel with simulated progress)
        const analysisPromise = analyzeImage(file, {
          includeMetadata: true,
          includeExplainability: true,
          includeForensics: true,
        });

        // Simulate progress through steps while waiting for API
        for (let i = 1; i < ANALYSIS_STEPS.length - 1; i++) {
          const start = 10 + ((i - 1) * 80) / (ANALYSIS_STEPS.length - 2);
          const end = 10 + (i * 80) / (ANALYSIS_STEPS.length - 2);
          await simulateProgress(start, end, i);
        }

        // Wait for actual result
        const result = await analysisPromise;
        setResult(result);
      } catch (err) {
        if (err instanceof Error && err.name === "AbortError") return;
        setError(
          err instanceof Error ? err.message : "Analysis failed",
        );
      }
    },
    [setStatus, setProgress, setResult, setError, simulateProgress],
  );

  const reset = useCallback(() => {
    abortRef.current?.abort();
    storeReset();
  }, [storeReset]);

  return {
    ...store,
    analyze,
    reset,
  };
}
