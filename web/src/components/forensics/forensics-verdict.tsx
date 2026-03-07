"use client";

import { Shield, AlertTriangle, Info, ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";
import type { ForensicsData, EvidenceItem, DetectionSummary } from "@/lib/types";
import { VERDICT_CONFIG, SEVERITY_CONFIG } from "@/lib/constants";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { Verdict } from "@/lib/types";

interface ForensicsVerdictProps {
  forensics: ForensicsData;
  detectionSummary?: DetectionSummary;
  overrideApplied?: boolean;
}

function formatVerdict(verdict: string): string {
  const config = VERDICT_CONFIG[verdict as Verdict];
  return config?.label ?? verdict.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function getVerdictColor(verdict: string): string {
  const config = VERDICT_CONFIG[verdict as Verdict];
  return config?.color ?? "#F59E0B";
}

export function ForensicsVerdict({
  forensics,
  detectionSummary,
  overrideApplied,
}: ForensicsVerdictProps) {
  const [expandedEvidence, setExpandedEvidence] = useState(true);

  const scoreColor =
    forensics.authenticity_score > 0.7
      ? "text-verdict-real"
      : forensics.authenticity_score > 0.4
        ? "text-verdict-uncertain"
        : "text-verdict-ai";

  // Categorize evidence by severity
  const criticalEvidence = forensics.evidence.filter(
    (e) => e.severity === "critical",
  );
  const warningEvidence = forensics.evidence.filter(
    (e) => e.severity === "warning",
  );
  const infoEvidence = forensics.evidence.filter(
    (e) => e.severity === "info",
  );

  const criticalCount = criticalEvidence.length;
  const warningCount = warningEvidence.length;
  const infoCount = infoEvidence.length;
  const totalEvidence = forensics.evidence.length;

  // Compute a consistency assessment
  const aiEvidenceCount = criticalCount + warningCount;
  const realEvidenceCount = infoCount;

  return (
    <div className="space-y-4">
      {/* Summary Card */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Forensic Analysis</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div>
              <p className="text-[10px] text-muted-foreground uppercase">
                Verdict
              </p>
              <p
                className="text-sm font-semibold mt-1"
                style={{ color: getVerdictColor(forensics.primary_verdict) }}
              >
                {formatVerdict(forensics.primary_verdict)}
              </p>
            </div>
            <div>
              <p className="text-[10px] text-muted-foreground uppercase">
                Authenticity
              </p>
              <p className={cn("text-xl font-bold font-mono", scoreColor)}>
                {(forensics.authenticity_score * 100).toFixed(0)}%
              </p>
            </div>
            <div>
              <p className="text-[10px] text-muted-foreground uppercase">
                Copy-Move
              </p>
              <Badge
                variant={
                  forensics.copy_move_detected ? "danger" : "success"
                }
                className="mt-1 text-[10px]"
              >
                {forensics.copy_move_detected ? "Detected" : "None"}
              </Badge>
            </div>
            <div>
              <p className="text-[10px] text-muted-foreground uppercase">
                Splicing
              </p>
              <Badge
                variant={
                  forensics.splicing_detected ? "danger" : "success"
                }
                className="mt-1 text-[10px]"
              >
                {forensics.splicing_detected ? "Detected" : "None"}
              </Badge>
            </div>
          </div>

          {/* Evidence summary bar */}
          <div className="flex items-center gap-3 pt-3 border-t">
            {criticalCount > 0 && (
              <div className="flex items-center gap-1.5 text-xs text-verdict-ai">
                <AlertTriangle className="h-3.5 w-3.5" />
                {criticalCount} critical
              </div>
            )}
            {warningCount > 0 && (
              <div className="flex items-center gap-1.5 text-xs text-verdict-uncertain">
                <AlertTriangle className="h-3.5 w-3.5" />
                {warningCount} warning{warningCount > 1 ? "s" : ""}
              </div>
            )}
            {infoCount > 0 && (
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Info className="h-3.5 w-3.5" />
                {infoCount} authentic
              </div>
            )}
            <div className="flex-1" />
            <span className="text-[10px] text-muted-foreground">
              {totalEvidence} indicators analyzed
            </span>
          </div>

          {/* Consistency assessment */}
          {detectionSummary && overrideApplied && (
            <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3">
              <div className="flex items-start gap-2">
                <AlertTriangle className="h-3.5 w-3.5 text-amber-500 shrink-0 mt-0.5" />
                <div className="space-y-1">
                  <p className="text-[11px] font-medium text-amber-600 dark:text-amber-400">
                    Verdict Override Active
                  </p>
                  <p className="text-[11px] text-muted-foreground leading-relaxed">
                    The forensic verdict was adjusted based on contextual evidence.
                    {" "}CNN models predict {(detectionSummary.cnn_avg * 100).toFixed(1)}% AI probability,
                    while signal analysis averages {(detectionSummary.signal_avg * 100).toFixed(1)}%.
                    {aiEvidenceCount > realEvidenceCount ? (
                      <> The majority of forensic indicators ({aiEvidenceCount}/{totalEvidence}) support the AI detection.</>
                    ) : (
                      <> Most individual detectors ({realEvidenceCount}/{totalEvidence}) show low AI probability,
                      but the override was triggered by additional evidence.</>
                    )}
                  </p>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Evidence List */}
      {forensics.evidence.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <button
              className="flex items-center justify-between w-full"
              onClick={() => setExpandedEvidence(!expandedEvidence)}
            >
              <CardTitle className="text-base">Evidence Details</CardTitle>
              {expandedEvidence ? (
                <ChevronUp className="h-4 w-4 text-muted-foreground" />
              ) : (
                <ChevronDown className="h-4 w-4 text-muted-foreground" />
              )}
            </button>
          </CardHeader>
          {expandedEvidence && (
            <CardContent className="space-y-4">
              {/* Critical evidence first */}
              {criticalEvidence.length > 0 && (
                <div className="space-y-2">
                  <p className="text-xs font-medium text-verdict-ai uppercase tracking-wider">
                    Critical Findings
                  </p>
                  {criticalEvidence.map((item, i) => (
                    <EvidenceCard key={`critical-${i}`} item={item} />
                  ))}
                </div>
              )}

              {/* Warning evidence */}
              {warningEvidence.length > 0 && (
                <div className="space-y-2">
                  <p className="text-xs font-medium text-verdict-uncertain uppercase tracking-wider">
                    Warnings
                  </p>
                  {warningEvidence.map((item, i) => (
                    <EvidenceCard key={`warning-${i}`} item={item} />
                  ))}
                </div>
              )}

              {/* Info evidence */}
              {infoEvidence.length > 0 && (
                <div className="space-y-2">
                  <p className="text-xs font-medium text-verdict-real uppercase tracking-wider">
                    Authentic Indicators
                  </p>
                  {infoEvidence.map((item, i) => (
                    <EvidenceCard key={`info-${i}`} item={item} />
                  ))}
                </div>
              )}
            </CardContent>
          )}
        </Card>
      )}
    </div>
  );
}

function EvidenceCard({ item }: { item: EvidenceItem }) {
  const config = SEVERITY_CONFIG[item.severity] || SEVERITY_CONFIG.info;
  const Icon =
    item.severity === "critical"
      ? AlertTriangle
      : item.severity === "warning"
        ? Info
        : Shield;

  return (
    <div
      className={cn(
        "flex gap-3 p-3 rounded-lg border-l-4",
        config.bgClass,
        config.borderClass,
      )}
    >
      <Icon
        className="h-4 w-4 shrink-0 mt-0.5"
        style={{ color: config.color }}
      />
      <div className="space-y-0.5">
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium">{item.type}</span>
          <Badge
            variant="outline"
            className="text-[9px] h-4 px-1.5"
            style={{ borderColor: config.color, color: config.color }}
          >
            {item.severity}
          </Badge>
        </div>
        <p className="text-xs text-muted-foreground">{item.description}</p>
        {item.details && (
          <p className="text-[11px] text-muted-foreground/70 mt-1">
            {item.details}
          </p>
        )}
      </div>
    </div>
  );
}
