"use client";

import { AlertTriangle, Info } from "lucide-react";
import type { IndividualResult, Verdict, DetectionSummary } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

interface ModelVotingChartProps {
  results: IndividualResult[];
  votes: { ai: number; real: number; total: number };
  finalVerdict?: Verdict;
  overrideApplied?: boolean;
  detectionSummary?: DetectionSummary;
}

export function ModelVotingChart({
  results,
  votes,
  finalVerdict,
  overrideApplied,
  detectionSummary,
}: ModelVotingChartProps) {
  const total = votes.total || 1;

  // Split results into categories for clearer display
  const cnnModels = results.filter(
    (r) =>
      r.method.includes("calibrated") && !r.method.includes("Ensemble"),
  );
  const cnnEnsemble = results.filter(
    (r) => r.method.includes("Ensemble"),
  );
  const hfModels = results.filter(
    (r) =>
      r.method.startsWith("ML:") &&
      !r.method.includes("calibrated") &&
      !r.method.includes("Ensemble") &&
      !r.method.includes("Custom Trained"),
  );
  const signals = results.filter((r) =>
    ["Frequency", "Texture", "Noise", "Edge", "Color"].some((s) =>
      r.method.includes(s),
    ),
  );
  const otherML = results.filter(
    (r) =>
      r.method.includes("Custom Trained"),
  );

  // Check if model votes disagree with final verdict
  const voteMajorityIsAI = votes.ai > votes.real;
  const verdictIsAI = finalVerdict === "ai_generated" || finalVerdict === "manipulated";
  const disagreement = overrideApplied && (
    (verdictIsAI && !voteMajorityIsAI) ||
    (!verdictIsAI && voteMajorityIsAI && finalVerdict !== "uncertain" && finalVerdict !== "screenshot")
  );

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Model Agreement</CardTitle>
      </CardHeader>
      <CardContent className="space-y-5">
        {/* Vote bar */}
        <div className="space-y-2">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Real ({votes.real})</span>
            <span>AI ({votes.ai})</span>
          </div>
          <div className="h-6 rounded-full bg-muted flex overflow-hidden">
            <div
              className="bg-verdict-real/80 h-full transition-all duration-700"
              style={{ width: `${(votes.real / total) * 100}%` }}
            />
            <div
              className="bg-verdict-ai/80 h-full transition-all duration-700"
              style={{ width: `${(votes.ai / total) * 100}%` }}
            />
          </div>
        </div>

        {/* Disagreement notice */}
        {disagreement && (
          <div className="flex items-start gap-2 p-2.5 rounded-lg border border-amber-500/30 bg-amber-500/5">
            <AlertTriangle className="h-3.5 w-3.5 text-amber-500 shrink-0 mt-0.5" />
            <p className="text-[11px] text-muted-foreground leading-relaxed">
              <span className="font-medium text-amber-600 dark:text-amber-400">
                Model votes ({votes.ai} AI / {votes.real} Real) differ from the final verdict
              </span>
              {" "}&mdash; the verdict incorporates additional evidence
              (patch localization, forensic analysis) beyond
              individual model scores.
            </p>
          </div>
        )}

        {/* CNN Calibrated Models */}
        {cnnModels.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              CNN Models (Calibrated)
            </p>
            <div className="space-y-1.5">
              {cnnModels.map((model, i) => (
                <ModelBar key={i} result={model} />
              ))}
            </div>
          </div>
        )}

        {/* CNN Ensemble */}
        {cnnEnsemble.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              CNN Ensemble
            </p>
            <div className="space-y-1.5">
              {cnnEnsemble.map((model, i) => (
                <ModelBar key={i} result={model} />
              ))}
            </div>
          </div>
        )}

        {/* HuggingFace Models */}
        {hfModels.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              HuggingFace Detectors
            </p>
            <div className="space-y-1.5">
              {hfModels.map((model, i) => (
                <ModelBar key={i} result={model} />
              ))}
            </div>
          </div>
        )}

        {/* Other ML models */}
        {otherML.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              Custom Trained
            </p>
            <div className="space-y-1.5">
              {otherML.map((model, i) => (
                <ModelBar key={i} result={model} />
              ))}
            </div>
          </div>
        )}

        {/* Signal Analysis */}
        {signals.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              Signal Analysis
            </p>
            <div className="space-y-1.5">
              {signals.map((signal, i) => (
                <ModelBar key={i} result={signal} />
              ))}
            </div>
          </div>
        )}

        {/* Interpretation footer */}
        {detectionSummary && (
          <div className="flex items-start gap-2 pt-3 border-t">
            <Info className="h-3.5 w-3.5 text-muted-foreground shrink-0 mt-0.5" />
            <p className="text-[10px] text-muted-foreground leading-relaxed italic">
              CNN models are trained on your dataset with temperature-scaled calibration.
              HuggingFace detectors are pre-trained public models.
              Signal analysis uses frequency, texture, noise, edge, and color forensics.
              All contribute to the weighted ensemble score.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function ModelBar({ result }: { result: IndividualResult }) {
  const isAI = result.ai_probability > 0.5;
  const prob = result.ai_probability * 100;

  const shortName = result.method
    .replace("ML: CNN ", "")
    .replace("ML: ", "")
    .replace("AI Image Detector", "AI Detector")
    .replace("Analysis (FFT)", "(FFT)")
    .replace("Pattern Analysis", "Pattern")
    .replace(" Analysis", "")
    .replace(" Coherence", "")
    .replace(" Distribution", "")
    .replace(" (calibrated)", " (cal)")
    .replace(" (min)", " Ensemble")
    .trim();

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div className="group cursor-default">
          <div className="flex items-center gap-2">
            <span className="text-xs w-36 truncate text-muted-foreground group-hover:text-foreground transition-colors">
              {shortName}
            </span>
            <div className="flex-1 h-4 rounded bg-muted overflow-hidden">
              <div
                className={cn(
                  "h-full rounded transition-all duration-600",
                  isAI ? "bg-verdict-ai/70" : "bg-verdict-real/70",
                )}
                style={{ width: `${prob}%` }}
              />
            </div>
            <span
              className={cn(
                "text-xs font-mono w-12 text-right",
                isAI ? "text-verdict-ai" : "text-verdict-real",
              )}
            >
              {prob.toFixed(1)}%
            </span>
          </div>
        </div>
      </TooltipTrigger>
      <TooltipContent>
        <p className="text-xs">
          {result.method}: {(result.ai_probability * 100).toFixed(2)}% AI
          (weight: {(result.weight ?? 0).toFixed(3)})
        </p>
      </TooltipContent>
    </Tooltip>
  );
}
