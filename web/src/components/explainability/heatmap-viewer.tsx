"use client";

import { useState } from "react";
import { Eye, EyeOff, ZoomIn } from "lucide-react";
import type { GradCAMData } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";

interface HeatmapViewerProps {
  gradcam: GradCAMData;
  originalImage: string;
}

export function HeatmapViewer({ gradcam, originalImage }: HeatmapViewerProps) {
  const [opacity, setOpacity] = useState(70);
  const [showOverlay, setShowOverlay] = useState(true);
  const [view, setView] = useState<"overlay" | "heatmap" | "side-by-side">(
    "side-by-side",
  );

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Grad-CAM Heatmap</CardTitle>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-[10px]">
              Layer: {gradcam.layer_name}
            </Badge>
            <Badge variant="outline" className="text-[10px]">
              Score: {(gradcam.activation_score * 100).toFixed(1)}%
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* View controls */}
        <div className="flex items-center justify-between gap-4">
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

          {view === "overlay" && (
            <div className="flex items-center gap-3 flex-1 max-w-[200px]">
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7"
                onClick={() => setShowOverlay(!showOverlay)}
              >
                {showOverlay ? (
                  <Eye className="h-3.5 w-3.5" />
                ) : (
                  <EyeOff className="h-3.5 w-3.5" />
                )}
              </Button>
              <Slider
                min={0}
                max={100}
                value={opacity}
                onChange={(e) =>
                  setOpacity(Number(e.target.value))
                }
                disabled={!showOverlay}
              />
              <span className="text-xs text-muted-foreground font-mono w-8">
                {opacity}%
              </span>
            </div>
          )}
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
              {view === "overlay" && showOverlay && gradcam.overlay_base64 && (
                <img
                  src={`data:image/png;base64,${gradcam.overlay_base64}`}
                  alt="Heatmap overlay"
                  className="absolute inset-0 w-full h-full object-contain heatmap-overlay transition-opacity duration-300"
                  style={{ opacity: opacity / 100 }}
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
              {gradcam.heatmap_base64 ? (
                <img
                  src={`data:image/png;base64,${gradcam.heatmap_base64}`}
                  alt="Grad-CAM Heatmap"
                  className="w-full h-auto object-contain max-h-[350px]"
                />
              ) : gradcam.overlay_base64 ? (
                <img
                  src={`data:image/png;base64,${gradcam.overlay_base64}`}
                  alt="Grad-CAM Overlay"
                  className="w-full h-auto object-contain max-h-[350px]"
                />
              ) : (
                <div className="flex items-center justify-center h-[200px] text-muted-foreground text-sm">
                  Heatmap not available
                </div>
              )}
              <p className="absolute bottom-2 left-2 text-[10px] bg-background/80 backdrop-blur-sm px-2 py-0.5 rounded text-muted-foreground">
                Grad-CAM
              </p>
            </div>
          )}
        </div>

        {/* Highlighted regions */}
        {gradcam.highlighted_regions.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              Highlighted Regions
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {gradcam.highlighted_regions.map((region, i) => (
                <div
                  key={i}
                  className="flex items-start gap-2 p-2.5 rounded-lg bg-muted/50 text-xs"
                >
                  <ZoomIn className="h-3.5 w-3.5 text-primary shrink-0 mt-0.5" />
                  <div>
                    <p className="font-medium">
                      Region {i + 1} &mdash;{" "}
                      {(region.activation * 100).toFixed(0)}% activation
                    </p>
                    <p className="text-muted-foreground mt-0.5">
                      {region.description}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Legend */}
        <div className="flex items-center justify-center gap-4 pt-2">
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-2 rounded-sm bg-blue-500" />
            <span className="text-[10px] text-muted-foreground">Low</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-2 rounded-sm bg-green-500" />
            <span className="text-[10px] text-muted-foreground">Medium</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-2 rounded-sm bg-yellow-500" />
            <span className="text-[10px] text-muted-foreground">High</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-2 rounded-sm bg-red-500" />
            <span className="text-[10px] text-muted-foreground">
              Very High
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
