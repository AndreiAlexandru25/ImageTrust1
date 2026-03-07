"use client";

import { Clock, Zap } from "lucide-react";
import type { TimingBreakdown as TimingType } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface TimingBreakdownProps {
  timing: TimingType;
  totalMs: number;
}

export function TimingBreakdown({ timing, totalMs }: TimingBreakdownProps) {
  const entries = [
    { label: "ML Models", ms: timing.ml_models_ms, color: "bg-primary" },
    {
      label: "CNN Ensemble",
      ms: timing.calibrated_ensemble_ms,
      color: "bg-indigo-400",
    },
    {
      label: "Frequency Analysis",
      ms: timing.frequency_ms,
      color: "bg-blue-400",
    },
    { label: "Noise Analysis", ms: timing.noise_ms, color: "bg-cyan-400" },
    {
      label: "Texture Analysis",
      ms: timing.texture_ms,
      color: "bg-teal-400",
    },
    { label: "Edge Analysis", ms: timing.edge_ms, color: "bg-emerald-400" },
    { label: "Color Analysis", ms: timing.color_ms, color: "bg-green-400" },
    {
      label: "Ensemble",
      ms: timing.ensemble_ms,
      color: "bg-amber-400",
    },
    ...(timing.gradcam_ms
      ? [{ label: "Grad-CAM", ms: timing.gradcam_ms, color: "bg-orange-400" }]
      : []),
    ...(timing.metadata_ms
      ? [
          {
            label: "Metadata",
            ms: timing.metadata_ms,
            color: "bg-violet-400",
          },
        ]
      : []),
    ...(timing.localization_ms
      ? [
          {
            label: "Localization",
            ms: timing.localization_ms,
            color: "bg-rose-400",
          },
        ]
      : []),
    ...(timing.screenshot_ms
      ? [
          {
            label: "Screenshot Det.",
            ms: timing.screenshot_ms,
            color: "bg-pink-400",
          },
        ]
      : []),
  ].filter((e) => e.ms >= 1);

  const maxMs = Math.max(...entries.map((e) => e.ms), 1);

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Clock className="h-4 w-4 text-muted-foreground" />
            <CardTitle className="text-base">Performance</CardTitle>
          </div>
          <div className="flex items-center gap-1.5 text-sm">
            <Zap className="h-3.5 w-3.5 text-verdict-uncertain" />
            <span className="font-mono font-bold">
              {totalMs.toFixed(0)}ms
            </span>
            <span className="text-muted-foreground text-xs">total</span>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {entries.map((entry) => (
            <div
              key={entry.label}
              className="flex items-center gap-3"
            >
              <span className="text-xs text-muted-foreground w-32 shrink-0 truncate">
                {entry.label}
              </span>
              <div className="flex-1 h-4 rounded bg-muted overflow-hidden">
                <div
                  className={cn(
                    "h-full rounded transition-all duration-500",
                    entry.color,
                  )}
                  style={{ width: `${(entry.ms / maxMs) * 100}%` }}
                />
              </div>
              <span className="text-xs font-mono text-muted-foreground w-16 text-right">
                {entry.ms.toFixed(0)}ms
              </span>
            </div>
          ))}
        </div>

        <div className="grid grid-cols-3 gap-4 mt-4 pt-4 border-t">
          <div className="text-center">
            <p className="text-[10px] text-muted-foreground uppercase">
              Total
            </p>
            <p className="text-lg font-bold font-mono">
              {(totalMs / 1000).toFixed(2)}s
            </p>
          </div>
          <div className="text-center">
            <p className="text-[10px] text-muted-foreground uppercase">
              Throughput
            </p>
            <p className="text-lg font-bold font-mono">
              {(1000 / totalMs).toFixed(1)}
              <span className="text-xs text-muted-foreground ml-0.5">
                img/s
              </span>
            </p>
          </div>
          <div className="text-center">
            <p className="text-[10px] text-muted-foreground uppercase">
              Components
            </p>
            <p className="text-lg font-bold font-mono">{entries.length}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
