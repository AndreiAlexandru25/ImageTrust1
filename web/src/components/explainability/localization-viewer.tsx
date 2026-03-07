"use client";

import { useState } from "react";
import { MapPin, AlertTriangle, Shield, Grid3X3 } from "lucide-react";
import type { LocalizationData } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

interface LocalizationViewerProps {
  localization: LocalizationData;
  originalImage: string;
}

export function LocalizationViewer({
  localization,
  originalImage,
}: LocalizationViewerProps) {
  const [view, setView] = useState<"overlay" | "heatmap" | "side-by-side">(
    "side-by-side",
  );

  const hotCount = localization.hot_regions.length;
  const criticalCount = localization.hot_regions.filter(
    (r) => r.severity === "critical",
  ).length;

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base flex items-center gap-2">
            <Grid3X3 className="h-4 w-4" />
            Patch-Level AI Localization
          </CardTitle>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-[10px]">
              {localization.grid_shape[0]}×{localization.grid_shape[1]} grid
            </Badge>
            <Badge variant="outline" className="text-[10px]">
              {localization.n_patches} patches
            </Badge>
            <Badge variant="outline" className="text-[10px]">
              {localization.n_models_used} models
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Summary stats */}
        <div className="grid grid-cols-3 gap-3">
          <div className="text-center p-2 rounded-lg bg-muted/50">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">
              Mean AI Prob
            </p>
            <p
              className={`text-lg font-mono font-bold ${
                localization.mean_ai_prob > 0.5
                  ? "text-verdict-ai"
                  : "text-verdict-real"
              }`}
            >
              {(localization.mean_ai_prob * 100).toFixed(1)}%
            </p>
          </div>
          <div className="text-center p-2 rounded-lg bg-muted/50">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">
              Max AI Prob
            </p>
            <p
              className={`text-lg font-mono font-bold ${
                localization.max_ai_prob > 0.5
                  ? "text-verdict-ai"
                  : localization.max_ai_prob > 0.35
                    ? "text-verdict-uncertain"
                    : "text-verdict-real"
              }`}
            >
              {(localization.max_ai_prob * 100).toFixed(1)}%
            </p>
          </div>
          <div className="text-center p-2 rounded-lg bg-muted/50">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">
              Hot Regions
            </p>
            <p
              className={`text-lg font-mono font-bold ${
                hotCount > 0 ? "text-verdict-ai" : "text-verdict-real"
              }`}
            >
              {hotCount}
              {criticalCount > 0 && (
                <span className="text-xs ml-1">
                  ({criticalCount} critical)
                </span>
              )}
            </p>
          </div>
        </div>

        {/* View controls */}
        <div className="flex gap-1">
          {(["side-by-side", "overlay", "heatmap"] as const).map((v) => (
            <Button
              key={v}
              variant={view === v ? "default" : "ghost"}
              size="sm"
              className="text-xs h-7"
              onClick={() => setView(v)}
            >
              {v === "side-by-side"
                ? "Compare"
                : v === "overlay"
                  ? "Overlay"
                  : "Heatmap"}
            </Button>
          ))}
        </div>

        {/* Images */}
        <div
          className={
            view === "side-by-side"
              ? "grid grid-cols-1 md:grid-cols-2 gap-3"
              : ""
          }
        >
          {(view === "side-by-side" || view === "overlay") && (
            <div className="relative rounded-lg overflow-hidden bg-black/5 dark:bg-white/5">
              <img
                src={originalImage}
                alt="Original"
                className="w-full h-auto object-contain max-h-[350px]"
              />
              {view === "overlay" && localization.overlay_base64 && (
                <img
                  src={`data:image/png;base64,${localization.overlay_base64}`}
                  alt="Localization overlay"
                  className="absolute inset-0 w-full h-full object-contain opacity-60"
                />
              )}
              {view === "side-by-side" && (
                <p className="absolute bottom-2 left-2 text-[10px] bg-background/80 backdrop-blur-sm px-2 py-0.5 rounded text-muted-foreground">
                  Original
                </p>
              )}
            </div>
          )}

          {(view === "side-by-side" || view === "heatmap") && (
            <div className="relative rounded-lg overflow-hidden bg-black/5 dark:bg-white/5">
              {localization.overlay_base64 ? (
                <img
                  src={`data:image/png;base64,${localization.overlay_base64}`}
                  alt="AI Localization Map"
                  className="w-full h-auto object-contain max-h-[350px]"
                />
              ) : (
                <div className="flex items-center justify-center h-[200px] text-muted-foreground text-sm">
                  Localization map not available
                </div>
              )}
              <p className="absolute bottom-2 left-2 text-[10px] bg-background/80 backdrop-blur-sm px-2 py-0.5 rounded text-muted-foreground">
                AI Localization
              </p>
            </div>
          )}
        </div>

        {/* Hot regions list */}
        {localization.hot_regions.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              Detected AI Regions
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {localization.hot_regions.slice(0, 8).map((region, i) => (
                <div
                  key={i}
                  className={`flex items-start gap-2 p-2.5 rounded-lg text-xs border ${
                    region.severity === "critical"
                      ? "border-verdict-ai/30 bg-verdict-ai/5"
                      : region.severity === "warning"
                        ? "border-verdict-uncertain/30 bg-verdict-uncertain/5"
                        : "border-border bg-muted/50"
                  }`}
                >
                  {region.severity === "critical" ? (
                    <AlertTriangle className="h-3.5 w-3.5 text-verdict-ai shrink-0 mt-0.5" />
                  ) : (
                    <MapPin className="h-3.5 w-3.5 text-verdict-uncertain shrink-0 mt-0.5" />
                  )}
                  <div>
                    <p className="font-medium">
                      Patch ({region.row}, {region.col}) &mdash;{" "}
                      {(region.ai_probability * 100).toFixed(0)}% AI
                    </p>
                    <p className="text-muted-foreground mt-0.5">
                      Position: ({region.x}, {region.y}) &bull;{" "}
                      {region.width}×{region.height}px
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {localization.hot_regions.length === 0 && (
          <div className="flex items-center justify-center gap-2 py-4 text-sm text-verdict-real">
            <Shield className="h-4 w-4" />
            <span>No AI-generated regions detected at patch level</span>
          </div>
        )}

        {/* Legend */}
        <div className="flex items-center justify-center gap-4 pt-2">
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-2 rounded-sm bg-green-500" />
            <span className="text-[10px] text-muted-foreground">
              Authentic
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-2 rounded-sm bg-yellow-500" />
            <span className="text-[10px] text-muted-foreground">
              Uncertain
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-2 rounded-sm bg-red-500" />
            <span className="text-[10px] text-muted-foreground">
              AI-Generated
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
