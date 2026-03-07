"use client";

import {
  ShieldCheck,
  Bot,
  HelpCircle,
  AlertTriangle,
  Monitor,
  Info,
  ArrowRight,
  Layers,
} from "lucide-react";
import type { Verdict, Confidence, DetectionSummary } from "@/lib/types";
import { VERDICT_CONFIG, CONFIDENCE_CONFIG } from "@/lib/constants";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

interface VerdictCardProps {
  verdict: Verdict;
  aiProbability: number;
  confidence: number;
  confidenceLevel: Confidence;
  votes: { ai: number; real: number; total: number };
  overrideApplied?: boolean;
  overrideReason?: string;
  rawVerdict?: Verdict;
  rawAiProbability?: number;
  detectionSummary?: DetectionSummary;
}

const iconMap = {
  real: ShieldCheck,
  ai_generated: Bot,
  uncertain: HelpCircle,
  manipulated: AlertTriangle,
  screenshot: Monitor,
};

export function VerdictCard({
  verdict,
  aiProbability,
  confidence,
  confidenceLevel,
  votes,
  overrideApplied,
  overrideReason,
  rawVerdict,
  rawAiProbability,
  detectionSummary,
}: VerdictCardProps) {
  const config = VERDICT_CONFIG[verdict] ?? VERDICT_CONFIG.uncertain;
  const confConfig = CONFIDENCE_CONFIG[confidenceLevel] ?? CONFIDENCE_CONFIG.medium;
  const Icon = iconMap[verdict] ?? HelpCircle;
  const ds = detectionSummary;

  // Determine if there's a significant disagreement between models and verdict
  const modelsDisagree = overrideApplied && ds && !ds.models_agree_with_verdict;

  return (
    <div
      className="rounded-xl border-2 p-6 space-y-4 animate-[fade-in_0.4s_ease-out]"
      style={{ borderColor: `${config.color}40` }}
    >
      {/* Final verdict header */}
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div
            className="flex items-center justify-center h-12 w-12 rounded-xl"
            style={{ backgroundColor: `${config.color}15` }}
          >
            <Icon
              className="h-6 w-6"
              style={{ color: config.color }}
            />
          </div>
          <div>
            <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">
              Final Verdict
            </p>
            <h2
              className="text-2xl font-bold"
              style={{ color: config.color }}
            >
              {config.label}
            </h2>
          </div>
        </div>
        <Badge
          variant="outline"
          className="text-xs"
          style={{ borderColor: confConfig.color, color: confConfig.color }}
        >
          {confConfig.label} Confidence
        </Badge>
      </div>

      {/* Key metrics */}
      <div className="grid grid-cols-3 gap-4">
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground">AI Probability</p>
          <p className="text-xl font-bold font-mono">
            {(aiProbability * 100).toFixed(1)}%
          </p>
        </div>
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground">Confidence</p>
          <p className="text-xl font-bold font-mono">
            {(confidence * 100).toFixed(1)}%
          </p>
        </div>
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground">Model Votes</p>
          <p className="text-xl font-bold font-mono">
            <span className={cn(
              votes.ai > votes.real ? "text-verdict-ai" : "text-verdict-real"
            )}>
              {votes.ai}
            </span>
            <span className="text-muted-foreground mx-1">/</span>
            <span className="text-muted-foreground">{votes.total}</span>
            <span className="text-xs text-muted-foreground ml-1">AI</span>
          </p>
        </div>
      </div>

      {/* Probability bar */}
      <div className="space-y-1.5">
        <div className="flex justify-between text-[10px] text-muted-foreground">
          <span>Real</span>
          <span>AI-Generated</span>
        </div>
        <div className="h-3 rounded-full bg-muted overflow-hidden relative">
          <div
            className="h-full rounded-full transition-all duration-1000 ease-out"
            style={{
              width: `${aiProbability * 100}%`,
              background: `linear-gradient(90deg, #22C55E, ${
                aiProbability > 0.5 ? "#EF4444" : "#F59E0B"
              })`,
            }}
          />
          <div
            className="absolute top-0 h-full w-px bg-foreground/30"
            style={{ left: "50%" }}
          />
        </div>
      </div>

      {/* Override explanation banner — shown when verdict was adjusted */}
      {overrideApplied && overrideReason && (
        <div className="rounded-lg border border-verdict-uncertain/40 bg-verdict-uncertain/5 p-3 space-y-2.5">
          <div className="flex items-start gap-2.5">
            <Info className="h-4 w-4 text-verdict-uncertain shrink-0 mt-0.5" />
            <div className="space-y-0.5">
              <p className="text-xs font-semibold text-verdict-uncertain">
                Verdict Override Applied
              </p>
              <p className="text-xs text-muted-foreground leading-relaxed">
                {overrideReason}
              </p>
            </div>
          </div>

          {/* Show raw vs final when override changed the verdict */}
          {rawVerdict && rawAiProbability != null && (
            <div className="flex items-center gap-2 pt-1 border-t border-verdict-uncertain/20">
              <Layers className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
              <div className="flex items-center gap-1.5 text-[11px]">
                <span className="text-muted-foreground">Model prediction:</span>
                <Badge
                  variant={rawVerdict === "real" ? "success" : rawVerdict === "ai_generated" ? "danger" : "warning"}
                  className="text-[9px] h-4 px-1.5"
                >
                  {VERDICT_CONFIG[rawVerdict]?.label ?? rawVerdict}
                </Badge>
                <span className="font-mono text-muted-foreground">
                  ({(rawAiProbability * 100).toFixed(1)}% AI)
                </span>
                <ArrowRight className="h-3 w-3 text-muted-foreground" />
                <span className="text-muted-foreground">Adjusted to:</span>
                <Badge
                  variant={verdict === "real" ? "success" : verdict === "ai_generated" ? "danger" : "warning"}
                  className="text-[9px] h-4 px-1.5"
                >
                  {config.label}
                </Badge>
                <span className="font-mono text-muted-foreground">
                  ({(aiProbability * 100).toFixed(1)}% AI)
                </span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Model disagreement warning */}
      {modelsDisagree && ds && (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3">
          <div className="flex items-start gap-2.5">
            <AlertTriangle className="h-4 w-4 text-amber-500 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <p className="text-xs font-semibold text-amber-600 dark:text-amber-400">
                Model Disagreement Detected
              </p>
              <p className="text-xs text-muted-foreground leading-relaxed">
                {ds.cnn_total > 0 && (
                  <>CNN Ensemble: {ds.cnn_ai_count}/{ds.cnn_total} vote AI
                  (avg {(ds.cnn_avg * 100).toFixed(1)}%). </>
                )}
                {ds.hf_total > 0 && (
                  <>HF Models: {ds.hf_ai_count}/{ds.hf_total} vote AI
                  (avg {(ds.hf_avg * 100).toFixed(1)}%). </>
                )}
                {ds.signal_total > 0 && (
                  <>Signals: {ds.signal_ai_count}/{ds.signal_total} flag AI
                  (avg {(ds.signal_avg * 100).toFixed(1)}%). </>
                )}
                {ds.loc_critical_count > 0 && (
                  <>Localization: {ds.loc_critical_count} critical region(s)
                  (max {(ds.loc_max_ai_prob * 100).toFixed(0)}% AI,
                  z-score {ds.loc_max_zscore.toFixed(1)}). </>
                )}
                The final verdict incorporates evidence beyond model scores.
              </p>
            </div>
          </div>
        </div>
      )}

      {verdict === "screenshot" && !overrideApplied && (
        <div className="flex items-start gap-2.5 rounded-lg border border-muted-foreground/20 bg-muted/50 p-3">
          <Monitor className="h-4 w-4 text-muted-foreground shrink-0 mt-0.5" />
          <p className="text-xs text-muted-foreground">
            Screenshot/screen capture detected via resolution and metadata heuristics.
          </p>
        </div>
      )}
    </div>
  );
}
