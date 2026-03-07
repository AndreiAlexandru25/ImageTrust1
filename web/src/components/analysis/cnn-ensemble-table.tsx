"use client";

import { AlertTriangle } from "lucide-react";
import type { CalibratedEnsemble, MetaClassifierResult, ConformalPrediction } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

interface CNNEnsembleTableProps {
  ensemble?: CalibratedEnsemble;
  metaClassifier?: MetaClassifierResult;
  conformal?: ConformalPrediction;
  finalVerdict?: string;
  overrideApplied?: boolean;
}

const STRATEGY_LABELS: Record<string, string> = {
  min: "Conservative (min)",
  mean: "Average (mean)",
  max: "Aggressive (max)",
  median: "Median",
};

export function CNNEnsembleTable({
  ensemble,
  metaClassifier,
  conformal,
  finalVerdict,
  overrideApplied,
}: CNNEnsembleTableProps) {
  // Check if CNN ensemble verdict differs from final verdict
  const ensembleDisagrees = overrideApplied && ensemble && (
    (ensemble.verdict === "real" && (finalVerdict === "ai_generated" || finalVerdict === "manipulated")) ||
    (ensemble.verdict === "ai_generated" && finalVerdict === "real")
  );
  return (
    <div className="space-y-4">
      {/* Calibrated Ensemble */}
      {ensemble && (
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">
                Calibrated CNN Ensemble
              </CardTitle>
              <Badge variant="outline" className="text-[10px]">
                {STRATEGY_LABELS[ensemble.strategy] ?? ensemble.strategy}
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-xs text-muted-foreground">
                    <th className="text-left py-2 font-medium">Model</th>
                    <th className="text-right py-2 font-medium">Raw</th>
                    <th className="text-right py-2 font-medium">Calibrated</th>
                    <th className="text-right py-2 font-medium">Verdict</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(ensemble.raw_probs).map(
                    ([model, rawProb]) => {
                      const calProb =
                        ensemble.calibrated_probs[model] ?? rawProb;
                      const isAI = calProb > 0.5;
                      return (
                        <tr
                          key={model}
                          className="border-b last:border-0"
                        >
                          <td className="py-2.5 text-xs">{model}</td>
                          <td className="py-2.5 text-right font-mono text-xs text-muted-foreground">
                            {(rawProb * 100).toFixed(1)}%
                          </td>
                          <td
                            className={cn(
                              "py-2.5 text-right font-mono text-xs font-medium",
                              isAI
                                ? "text-verdict-ai"
                                : "text-verdict-real",
                            )}
                          >
                            {(calProb * 100).toFixed(1)}%
                          </td>
                          <td className="py-2.5 text-right">
                            <Badge
                              variant={isAI ? "danger" : "success"}
                              className="text-[10px]"
                            >
                              {isAI ? "AI" : "Real"}
                            </Badge>
                          </td>
                        </tr>
                      );
                    },
                  )}
                </tbody>
                <tfoot>
                  <tr className="border-t bg-muted/30">
                    <td className="py-2.5 text-xs font-medium">
                      Ensemble Average
                    </td>
                    <td />
                    <td
                      className={cn(
                        "py-2.5 text-right font-mono text-xs font-bold",
                        ensemble.ensemble_avg_prob > 0.5
                          ? "text-verdict-ai"
                          : "text-verdict-real",
                      )}
                    >
                      {(ensemble.ensemble_avg_prob * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 text-right">
                      <Badge
                        variant={
                          ensemble.ensemble_avg_prob > 0.5
                            ? "danger"
                            : "success"
                        }
                        className="text-[10px]"
                      >
                        {ensemble.verdict_text}
                      </Badge>
                    </td>
                  </tr>
                </tfoot>
              </table>
            </div>

            <div className="grid grid-cols-3 gap-4 mt-4 pt-4 border-t">
              <div className="text-center">
                <p className="text-[10px] text-muted-foreground uppercase">
                  Agreement
                </p>
                <p className="text-lg font-bold font-mono">
                  {((ensemble.model_agreement ?? 0) * 100).toFixed(0)}%
                </p>
              </div>
              <div className="text-center">
                <p className="text-[10px] text-muted-foreground uppercase">
                  Std Dev
                </p>
                <p className="text-lg font-bold font-mono">
                  {(ensemble.ensemble_std ?? 0).toFixed(4)}
                </p>
              </div>
              <div className="text-center">
                <p className="text-[10px] text-muted-foreground uppercase">
                  Min Prob
                </p>
                <p className="text-lg font-bold font-mono">
                  {((ensemble.ensemble_min_prob ?? 0) * 100).toFixed(1)}%
                </p>
              </div>
            </div>

            {/* Uncertain region info */}
            {ensemble.uncertain_low != null && (
              <p className="text-[10px] text-muted-foreground mt-3 pt-3 border-t italic">
                Abstain region: [{(ensemble.uncertain_low * 100).toFixed(0)}% &ndash; {(ensemble.uncertain_high * 100).toFixed(0)}%].
                Predictions within this range are flagged as uncertain.
              </p>
            )}

            {/* Disagreement notice */}
            {ensembleDisagrees && (
              <div className="flex items-start gap-2 mt-3 pt-3 border-t">
                <AlertTriangle className="h-3.5 w-3.5 text-amber-500 shrink-0 mt-0.5" />
                <p className="text-[11px] text-muted-foreground leading-relaxed">
                  <span className="font-medium text-amber-600 dark:text-amber-400">
                    CNN Ensemble predicts &ldquo;{ensemble.verdict_text}&rdquo;
                  </span>
                  {" "}but the final verdict was adjusted to &ldquo;{finalVerdict?.replace(/_/g, " ")}&rdquo;
                  based on additional forensic evidence. CNN models are trained on
                  fully AI-generated images and may not detect partial edits or
                  AI-assisted modifications.
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Meta-Classifier */}
      {metaClassifier && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">
              Phase 2: Meta-Classifier (MLP)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div>
                <p className="text-[10px] text-muted-foreground uppercase">
                  AI Probability
                </p>
                <p
                  className={cn(
                    "text-xl font-bold font-mono",
                    metaClassifier.ai_probability > 0.5
                      ? "text-verdict-ai"
                      : "text-verdict-real",
                  )}
                >
                  {(metaClassifier.ai_probability * 100).toFixed(1)}%
                </p>
              </div>
              <div>
                <p className="text-[10px] text-muted-foreground uppercase">
                  Confidence
                </p>
                <p className="text-xl font-bold font-mono">
                  {(metaClassifier.confidence * 100).toFixed(1)}%
                </p>
              </div>
              <div>
                <p className="text-[10px] text-muted-foreground uppercase">
                  Raw Logit
                </p>
                <p className="text-xl font-bold font-mono">
                  {metaClassifier.raw_logit.toFixed(3)}
                </p>
              </div>
              <div>
                <p className="text-[10px] text-muted-foreground uppercase">
                  Uncertain
                </p>
                <Badge
                  variant={metaClassifier.is_uncertain ? "warning" : "success"}
                  className="mt-1"
                >
                  {metaClassifier.is_uncertain ? "Yes" : "No"}
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Conformal Prediction */}
      {conformal && conformal.prediction_set && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">
              Conformal Prediction
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div>
                <p className="text-[10px] text-muted-foreground uppercase">
                  Prediction Set
                </p>
                <div className="flex gap-1 mt-1">
                  {(conformal.prediction_set ?? []).map((pred) => (
                    <Badge
                      key={pred}
                      variant={pred === "ai_generated" ? "danger" : "success"}
                      className="text-[10px]"
                    >
                      {pred === "ai_generated" ? "AI" : "Real"}
                    </Badge>
                  ))}
                </div>
              </div>
              <div>
                <p className="text-[10px] text-muted-foreground uppercase">
                  Set Size
                </p>
                <p className="text-xl font-bold font-mono">
                  {conformal.set_size ?? 0}
                </p>
              </div>
              <div>
                <p className="text-[10px] text-muted-foreground uppercase">
                  Coverage
                </p>
                <p className="text-xl font-bold font-mono">
                  {((conformal.coverage_level ?? 0) * 100).toFixed(0)}%
                </p>
              </div>
              <div>
                <p className="text-[10px] text-muted-foreground uppercase">
                  Uncertain
                </p>
                <Badge
                  variant={conformal.is_uncertain ? "warning" : "success"}
                  className="mt-1"
                >
                  {conformal.is_uncertain ? "Yes" : "No"}
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Status message when conformal is not calibrated */}
      {conformal && !conformal.prediction_set && (
        <Card>
          <CardContent className="p-6 text-center text-sm text-muted-foreground">
            Conformal predictor requires calibration data.
            Run <code className="text-xs bg-muted px-1.5 py-0.5 rounded">calibrate_conformal()</code> with
            a held-out calibration set to enable prediction intervals.
          </CardContent>
        </Card>
      )}
    </div>
  );
}
