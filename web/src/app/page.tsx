"use client";

import { useCallback, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import {
  Shield,
  Cpu,
  Eye,
  FileSearch,
  ArrowRight,
} from "lucide-react";
import { useAnalysisStore } from "@/stores/analysis-store";
import { useAnalysis } from "@/hooks/use-analysis";
import { ImageUploader } from "@/components/upload/image-uploader";
import { AnalysisProgress } from "@/components/analysis/analysis-progress";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

const features = [
  {
    icon: Cpu,
    title: "Ensemble ML Detection",
    description:
      "4 HuggingFace models + 5 signal analyzers with calibrated probabilities",
  },
  {
    icon: Eye,
    title: "Visual Explainability",
    description:
      "Grad-CAM heatmaps highlight suspicious regions with activation maps",
  },
  {
    icon: FileSearch,
    title: "Forensic Analysis",
    description:
      "EXIF metadata, C2PA provenance, copy-move and splicing detection",
  },
  {
    icon: Shield,
    title: "Calibrated Confidence",
    description:
      "Temperature scaling, conformal prediction, and UNCERTAIN abstain regions",
  },
];

export default function HomePage() {
  const router = useRouter();
  const { analyze, status, progress, currentStep, imagePreview, reset } =
    useAnalysis();
  const { imageFile, setFile, result } = useAnalysisStore();

  const handleFileSelect = useCallback(
    (file: File) => {
      const preview = URL.createObjectURL(file);
      setFile(file, preview);
    },
    [setFile],
  );

  const handleAnalyze = useCallback(async () => {
    if (!imageFile) return;
    analysisInitiated.current = true;
    await analyze(imageFile);
  }, [imageFile, analyze]);

  const handleClear = useCallback(() => {
    reset();
  }, [reset]);

  // Track whether analysis was initiated from this page visit
  const analysisInitiated = useRef(false);

  // Reset stale "complete" state when returning to home page
  // (e.g., via browser back button without clicking New Analysis)
  useEffect(() => {
    if (status === "complete" && result && !analysisInitiated.current) {
      reset();
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Navigate to results when analysis completes (only if we started it)
  useEffect(() => {
    if (status === "complete" && result && analysisInitiated.current) {
      analysisInitiated.current = false;
      router.push("/analysis");
    }
  }, [status, result, router]);

  const isProcessing = status === "uploading" || status === "analyzing";

  return (
    <div className="container mx-auto px-4 py-8 md:py-16">
      <div className="max-w-4xl mx-auto space-y-12 animate-[fade-in_0.4s_ease-out]">
        <div className="text-center space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border bg-muted/50 text-xs text-muted-foreground">
            <Shield className="h-3 w-3 text-primary" />
            AI-Generated Image Forensic Detection
          </div>
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
            Verify Image{" "}
            <span className="text-primary">Authenticity</span>
          </h1>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Upload an image to detect AI-generated content using ensemble machine
            learning models, signal analysis, and forensic techniques.
          </p>
        </div>

        <div>
          <Card className="overflow-hidden">
            <CardContent className="p-6 md:p-8">
              <div className="space-y-6">
                <ImageUploader
                  onFileSelect={handleFileSelect}
                  preview={imagePreview}
                  onClear={handleClear}
                  disabled={isProcessing}
                />

                {isProcessing && (
                  <AnalysisProgress
                    progress={progress}
                    currentStep={currentStep}
                    status={status}
                  />
                )}

                {status === "error" && (
                  <div className="text-sm text-destructive bg-destructive/10 rounded-lg px-4 py-3">
                    Analysis failed. Please check the backend is running and try again.
                  </div>
                )}

                {imageFile && !isProcessing && status !== "complete" && (
                  <div className="flex justify-center animate-[fade-in_0.3s_ease-out]">
                    <Button
                      size="lg"
                      onClick={handleAnalyze}
                      className="gap-2 px-8"
                    >
                      <ArrowRight className="h-4 w-4" />
                      Analyze Image
                    </Button>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {features.map((feature) => (
            <Card
              key={feature.title}
              className="group hover:border-primary/30 transition-colors"
            >
              <CardContent className="p-5 flex gap-4">
                <div className="flex items-center justify-center h-10 w-10 rounded-lg bg-primary/10 shrink-0 group-hover:bg-primary/15 transition-colors">
                  <feature.icon className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <h3 className="font-semibold text-sm">{feature.title}</h3>
                  <p className="text-xs text-muted-foreground mt-1">
                    {feature.description}
                  </p>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}
