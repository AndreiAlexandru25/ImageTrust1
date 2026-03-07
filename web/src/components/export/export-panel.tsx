"use client";

import { useCallback } from "react";
import { Download, FileJson, FileText } from "lucide-react";
import type { ComprehensiveAnalysisResult } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

interface ExportPanelProps {
  result: ComprehensiveAnalysisResult;
}

export function ExportPanel({ result }: ExportPanelProps) {
  const exportJSON = useCallback(() => {
    const data = JSON.stringify(result, null, 2);
    const blob = new Blob([data], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `imagetrust-${result.analysis_id}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [result]);

  const exportCSV = useCallback(() => {
    const rows = [
      ["Field", "Value"],
      ["Analysis ID", result.analysis_id],
      ["Verdict", result.verdict],
      ["AI Probability", result.ai_probability.toFixed(4)],
      ["Confidence", result.confidence.toFixed(4)],
      ["Confidence Level", result.confidence_level],
      ["Votes AI", result.votes.ai.toString()],
      ["Votes Real", result.votes.real.toString()],
      ["Processing Time (ms)", result.processing_time_ms.toFixed(0)],
    ];

    result.individual_results.forEach((r) => {
      rows.push([
        `Model: ${r.method}`,
        `${(r.ai_probability * 100).toFixed(2)}%`,
      ]);
    });

    const csv = rows.map((row) => row.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `imagetrust-${result.analysis_id}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [result]);

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center gap-2">
          <Download className="h-4 w-4 text-muted-foreground" />
          <CardTitle className="text-base">Export Results</CardTitle>
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col sm:flex-row gap-2">
          <Button
            variant="outline"
            className="flex-1 gap-2"
            onClick={exportJSON}
          >
            <FileJson className="h-4 w-4" />
            Export JSON
          </Button>
          <Button
            variant="outline"
            className="flex-1 gap-2"
            onClick={exportCSV}
          >
            <FileText className="h-4 w-4" />
            Export CSV
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
