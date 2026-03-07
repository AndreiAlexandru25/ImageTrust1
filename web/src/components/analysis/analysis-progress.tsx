"use client";

import { Check, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { ANALYSIS_STEPS } from "@/lib/constants";
import { Progress } from "@/components/ui/progress";

interface AnalysisProgressProps {
  progress: number;
  currentStep: string;
  status: "uploading" | "analyzing" | "complete" | "error";
}

export function AnalysisProgress({
  progress,
  currentStep,
  status,
}: AnalysisProgressProps) {
  const activeIndex = ANALYSIS_STEPS.findIndex((s) => s.label === currentStep);

  return (
    <div className="space-y-6 animate-[fade-in_0.3s_ease-out]">
      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span className="font-medium">{currentStep || "Preparing..."}</span>
          <span className="text-muted-foreground font-mono">{progress}%</span>
        </div>
        <Progress value={progress} className="h-2" />
      </div>

      <div className="space-y-1">
        {ANALYSIS_STEPS.map((step, i) => {
          const isComplete = i < activeIndex || status === "complete";
          const isActive = i === activeIndex && status !== "complete";

          return (
            <div
              key={step.id}
              className={cn(
                "flex items-center gap-3 py-1.5 px-2 rounded-md text-sm transition-colors",
                isActive && "bg-primary/5",
              )}
            >
              <div className="shrink-0">
                {isComplete ? (
                  <div className="flex items-center justify-center h-5 w-5 rounded-full bg-verdict-real/20">
                    <Check className="h-3 w-3 text-verdict-real" />
                  </div>
                ) : isActive ? (
                  <div className="flex items-center justify-center h-5 w-5">
                    <Loader2 className="h-4 w-4 text-primary animate-spin" />
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-5 w-5 rounded-full border border-muted-foreground/30">
                    <div className="h-1.5 w-1.5 rounded-full bg-muted-foreground/30" />
                  </div>
                )}
              </div>
              <span
                className={cn(
                  "transition-colors",
                  isComplete
                    ? "text-muted-foreground"
                    : isActive
                      ? "text-foreground font-medium"
                      : "text-muted-foreground/50",
                )}
              >
                {step.label}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
