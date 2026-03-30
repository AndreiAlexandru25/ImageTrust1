import type { Verdict, Confidence } from "./types";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const VERDICT_CONFIG: Record<
  Verdict,
  { label: string; color: string; bgClass: string; textClass: string; icon: string }
> = {
  real: {
    label: "Authentic",
    color: "#22C55E",
    bgClass: "bg-verdict-real/10",
    textClass: "text-verdict-real",
    icon: "shield-check",
  },
  ai_generated: {
    label: "AI-Generated",
    color: "#EF4444",
    bgClass: "bg-verdict-ai/10",
    textClass: "text-verdict-ai",
    icon: "bot",
  },
  uncertain: {
    label: "Uncertain",
    color: "#F59E0B",
    bgClass: "bg-verdict-uncertain/10",
    textClass: "text-verdict-uncertain",
    icon: "help-circle",
  },
  manipulated: {
    label: "Manipulated",
    color: "#F97316",
    bgClass: "bg-verdict-manipulated/10",
    textClass: "text-verdict-manipulated",
    icon: "alert-triangle",
  },
  screenshot: {
    label: "Screenshot / Capture",
    color: "#6366F1",
    bgClass: "bg-primary/10",
    textClass: "text-primary",
    icon: "monitor",
  },
};

export const CONFIDENCE_CONFIG: Record<
  Confidence,
  { label: string; color: string }
> = {
  very_low: { label: "Very Low", color: "#EF4444" },
  low: { label: "Low", color: "#F97316" },
  medium: { label: "Medium", color: "#F59E0B" },
  high: { label: "High", color: "#22C55E" },
  very_high: { label: "Very High", color: "#15803D" },
};

export const SEVERITY_CONFIG: Record<
  string,
  { color: string; bgClass: string; borderClass: string }
> = {
  info: {
    color: "#6366F1",
    bgClass: "bg-blue-500/10",
    borderClass: "border-l-blue-500",
  },
  warning: {
    color: "#F59E0B",
    bgClass: "bg-amber-500/10",
    borderClass: "border-l-amber-500",
  },
  critical: {
    color: "#EF4444",
    bgClass: "bg-red-500/10",
    borderClass: "border-l-red-500",
  },
};

export const ANALYSIS_STEPS = [
  {
    id: "upload",
    label: "Upload",
    description: "Uploading image to server",
  },
  {
    id: "preprocessing",
    label: "Preprocessing",
    description: "Preparing image for analysis",
  },
  {
    id: "ml_detection",
    label: "ML Detection",
    description: "Running AI detection models",
  },
  {
    id: "signal_analysis",
    label: "Signal Analysis",
    description: "Analyzing frequency, texture, noise patterns",
  },
  {
    id: "calibration",
    label: "Calibration",
    description: "Calibrating probabilities",
  },
  {
    id: "explainability",
    label: "Explainability",
    description: "Generating Grad-CAM heatmaps",
  },
  {
    id: "localization",
    label: "Localization",
    description: "Patch-level AI region detection",
  },
  {
    id: "forensics",
    label: "Forensics",
    description: "Running forensic analysis",
  },
  {
    id: "metadata",
    label: "Metadata",
    description: "Extracting image metadata",
  },
  {
    id: "complete",
    label: "Complete",
    description: "Analysis complete",
  },
];

export const THRESHOLDS = {
  AI_HIGH: 0.75,
  AI_LOW: 0.35,
  UNCERTAIN_LOW: 0.35,
  UNCERTAIN_HIGH: 0.65,
  CONFIDENCE_HIGH: 0.8,
};

export const MAX_FILE_SIZE_MB = 50;
export const ACCEPTED_FILE_TYPES = ["image/jpeg", "image/png", "image/webp"];
