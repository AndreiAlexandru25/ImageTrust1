"use client";

import { create } from "zustand";
import type { ComprehensiveAnalysisResult } from "@/lib/types";

interface AnalysisStore {
  status: "idle" | "uploading" | "analyzing" | "complete" | "error";
  progress: number;
  currentStep: string;
  result: ComprehensiveAnalysisResult | null;
  imagePreview: string | null;
  imageFile: File | null;
  error: string | null;

  setFile: (file: File, preview: string) => void;
  setStatus: (status: AnalysisStore["status"]) => void;
  setProgress: (progress: number, step: string) => void;
  setResult: (result: ComprehensiveAnalysisResult) => void;
  setError: (error: string) => void;
  reset: () => void;
}

export const useAnalysisStore = create<AnalysisStore>((set) => ({
  status: "idle",
  progress: 0,
  currentStep: "",
  result: null,
  imagePreview: null,
  imageFile: null,
  error: null,

  setFile: (file, preview) =>
    set({
      imageFile: file,
      imagePreview: preview,
      error: null,
      result: null,
      status: "idle",
      progress: 0,
      currentStep: "",
    }),

  setStatus: (status) => set({ status }),

  setProgress: (progress, step) =>
    set({ progress, currentStep: step }),

  setResult: (result) =>
    set({ result, status: "complete", progress: 100, currentStep: "Complete" }),

  setError: (error) =>
    set({ error, status: "error" }),

  reset: () =>
    set({
      status: "idle",
      progress: 0,
      currentStep: "",
      result: null,
      imagePreview: null,
      imageFile: null,
      error: null,
    }),
}));
