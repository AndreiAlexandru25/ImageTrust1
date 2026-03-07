"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { ArrowLeft, ImageIcon } from "lucide-react";
import { useAnalysisStore } from "@/stores/analysis-store";
import { ErrorBoundary } from "@/components/shared/error-boundary";
import { VerdictCard } from "@/components/analysis/verdict-card";
import { ConfidenceGauge } from "@/components/analysis/confidence-gauge";
import { ModelVotingChart } from "@/components/analysis/model-voting-chart";
import { ModelResultsGrid } from "@/components/analysis/model-results-grid";
import { CNNEnsembleTable } from "@/components/analysis/cnn-ensemble-table";
import { TimingBreakdown } from "@/components/analysis/timing-breakdown";
import { HeatmapViewer } from "@/components/explainability/heatmap-viewer";
import { LocalizationViewer } from "@/components/explainability/localization-viewer";
import { ForensicsVerdict } from "@/components/forensics/forensics-verdict";
import { MetadataPanel } from "@/components/metadata/metadata-panel";
import { ExportPanel } from "@/components/export/export-panel";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function AnalysisPage() {
  const router = useRouter();
  const { result, imagePreview, reset } = useAnalysisStore();

  useEffect(() => {
    if (!result) {
      router.push("/");
    }
  }, [result, router]);

  if (!result) return null;

  const handleNewAnalysis = () => {
    reset();
    router.push("/");
  };

  return (
    <div className="container mx-auto px-4 py-6">
      <div className="max-w-7xl mx-auto space-y-6 animate-[fade-in_0.3s_ease-out]">
        {/* Top bar */}
        <div className="flex items-center justify-between">
          <Button
            variant="ghost"
            className="gap-2"
            onClick={handleNewAnalysis}
          >
            <ArrowLeft className="h-4 w-4" />
            New Analysis
          </Button>
          <p className="text-xs text-muted-foreground font-mono">
            ID: {result.analysis_id}
          </p>
        </div>

        {/* Main layout: Sidebar + Content */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left sidebar */}
          <div className="lg:col-span-4 space-y-4">
            {/* Image preview */}
            {imagePreview && (
              <Card>
                <CardContent className="p-3">
                  <div className="relative rounded-lg overflow-hidden bg-black/5 dark:bg-white/5">
                    <img
                      src={imagePreview}
                      alt="Analyzed image"
                      className="w-full max-h-[280px] object-contain"
                    />
                    <div className="absolute bottom-2 right-2 flex items-center gap-1 text-[10px] bg-background/80 backdrop-blur-sm px-2 py-0.5 rounded text-muted-foreground">
                      <ImageIcon className="h-3 w-3" />
                      {result.image_dimensions?.[0] ?? "?"} x{" "}
                      {result.image_dimensions?.[1] ?? "?"}
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Verdict */}
            <ErrorBoundary>
              <VerdictCard
                verdict={result.verdict}
                aiProbability={result.ai_probability}
                confidence={result.confidence}
                confidenceLevel={result.confidence_level}
                votes={result.votes}
                overrideApplied={result.override_applied}
                overrideReason={result.override_reason}
                rawVerdict={result.raw_verdict}
                rawAiProbability={result.raw_ai_probability}
                detectionSummary={result.detection_summary}
              />
            </ErrorBoundary>

            {/* Gauge */}
            <ErrorBoundary>
              <Card>
                <CardContent className="p-4 flex justify-center">
                  <ConfidenceGauge value={result.ai_probability} />
                </CardContent>
              </Card>
            </ErrorBoundary>

            {/* Export */}
            <ErrorBoundary>
              <ExportPanel result={result} />
            </ErrorBoundary>
          </div>

          {/* Main content */}
          <div className="lg:col-span-8">
            <Tabs defaultValue="overview" className="space-y-4">
              <TabsList className="w-full justify-start flex-wrap h-auto gap-1 p-1">
                <TabsTrigger value="overview" className="text-xs">
                  Overview
                </TabsTrigger>
                <TabsTrigger value="ensemble" className="text-xs">
                  CNN Ensemble
                </TabsTrigger>
                <TabsTrigger value="explainability" className="text-xs">
                  Explainability
                </TabsTrigger>
                <TabsTrigger value="forensics" className="text-xs">
                  Forensics
                </TabsTrigger>
                <TabsTrigger value="metadata" className="text-xs">
                  Metadata
                </TabsTrigger>
                <TabsTrigger value="performance" className="text-xs">
                  Performance
                </TabsTrigger>
              </TabsList>

              {/* Overview Tab */}
              <TabsContent value="overview" className="space-y-4">
                <ErrorBoundary>
                  <ModelVotingChart
                    results={result.individual_results}
                    votes={result.votes}
                    finalVerdict={result.verdict}
                    overrideApplied={result.override_applied}
                    detectionSummary={result.detection_summary}
                  />
                </ErrorBoundary>
                <ErrorBoundary>
                  <ModelResultsGrid results={result.individual_results} />
                </ErrorBoundary>
              </TabsContent>

              {/* CNN Ensemble Tab */}
              <TabsContent value="ensemble">
                <ErrorBoundary>
                  <CNNEnsembleTable
                    ensemble={result.calibrated_ensemble}
                    metaClassifier={result.meta_classifier}
                    conformal={result.conformal_prediction}
                    finalVerdict={result.verdict}
                    overrideApplied={result.override_applied}
                  />
                </ErrorBoundary>
              </TabsContent>

              {/* Explainability Tab */}
              <TabsContent value="explainability" className="space-y-4">
                <ErrorBoundary>
                  {result.localization && imagePreview && (
                    <LocalizationViewer
                      localization={result.localization}
                      originalImage={imagePreview}
                    />
                  )}
                  {result.gradcam && imagePreview ? (
                    <HeatmapViewer
                      gradcam={result.gradcam}
                      originalImage={imagePreview}
                    />
                  ) : !result.localization ? (
                    <Card>
                      <CardContent className="p-8 text-center text-muted-foreground text-sm">
                        Explainability data not available for this analysis.
                      </CardContent>
                    </Card>
                  ) : null}
                </ErrorBoundary>
              </TabsContent>

              {/* Forensics Tab */}
              <TabsContent value="forensics">
                <ErrorBoundary>
                  {result.forensics ? (
                    <ForensicsVerdict
                      forensics={result.forensics}
                      detectionSummary={result.detection_summary}
                      overrideApplied={result.override_applied}
                    />
                  ) : (
                    <Card>
                      <CardContent className="p-8 text-center text-muted-foreground text-sm">
                        Forensic analysis data not available.
                      </CardContent>
                    </Card>
                  )}
                </ErrorBoundary>
              </TabsContent>

              {/* Metadata Tab */}
              <TabsContent value="metadata">
                <ErrorBoundary>
                  {result.metadata || result.provenance ? (
                    <MetadataPanel
                      metadata={result.metadata}
                      provenance={result.provenance}
                    />
                  ) : (
                    <Card>
                      <CardContent className="p-8 text-center text-muted-foreground text-sm">
                        Metadata extraction not available for this image.
                        <br />
                        <span className="text-xs">
                          EXIF, provenance, and AI indicator data will appear
                          here when available.
                        </span>
                      </CardContent>
                    </Card>
                  )}
                </ErrorBoundary>
              </TabsContent>

              {/* Performance Tab */}
              <TabsContent value="performance">
                <ErrorBoundary>
                  {result.timing_breakdown ? (
                    <TimingBreakdown
                      timing={result.timing_breakdown}
                      totalMs={result.processing_time_ms}
                    />
                  ) : (
                    <Card>
                      <CardContent className="p-8 text-center text-muted-foreground text-sm">
                        Timing data not available.
                        <br />
                        <span className="text-xs">
                          Total processing time:{" "}
                          {result.processing_time_ms?.toFixed(0) ?? "?"}ms
                        </span>
                      </CardContent>
                    </Card>
                  )}
                </ErrorBoundary>
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </div>
    </div>
  );
}
