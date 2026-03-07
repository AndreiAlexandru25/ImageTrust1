"use client";

import type { IndividualResult } from "@/lib/types";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

interface ModelResultsGridProps {
  results: IndividualResult[];
}

export function ModelResultsGrid({ results }: ModelResultsGridProps) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
      {results.map((result, i) => (
        <div key={i} className="animate-[fade-in_0.3s_ease-out]">
          <Card className="h-full hover:border-primary/20 transition-colors">
            <CardContent className="p-4 space-y-3">
              <div className="flex items-start justify-between gap-2">
                <h4 className="text-sm font-medium leading-tight">
                  {result.method}
                </h4>
                <Badge
                  variant={
                    result.ai_probability > 0.65
                      ? "danger"
                      : result.ai_probability < 0.35
                        ? "success"
                        : "warning"
                  }
                  className="shrink-0 text-[10px]"
                >
                  {result.ai_probability > 0.5 ? "AI" : "Real"}
                </Badge>
              </div>

              <div className="space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">AI Score</span>
                  <span className="font-mono">
                    {(result.ai_probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="h-2 rounded-full bg-muted overflow-hidden">
                  <div
                    className={cn(
                      "h-full rounded-full transition-all duration-600",
                      result.ai_probability > 0.65
                        ? "bg-verdict-ai"
                        : result.ai_probability < 0.35
                          ? "bg-verdict-real"
                          : "bg-verdict-uncertain",
                    )}
                    style={{ width: `${result.ai_probability * 100}%` }}
                  />
                </div>
              </div>

              <div className="flex justify-between text-[11px] text-muted-foreground">
                <span>
                  Confidence: {((result.confidence ?? 0) * 100).toFixed(0)}%
                </span>
                <span>Weight: {((result.weight ?? 0) * 100).toFixed(1)}%</span>
              </div>
            </CardContent>
          </Card>
        </div>
      ))}
    </div>
  );
}
