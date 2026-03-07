"use client";

import {
  Camera,
  AlertCircle,
  Check,
  Shield,
} from "lucide-react";
import type { MetadataInfo, ProvenanceInfo } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";

interface MetadataPanelProps {
  metadata?: MetadataInfo;
  provenance?: ProvenanceInfo;
}

export function MetadataPanel({ metadata, provenance }: MetadataPanelProps) {
  return (
    <div className="space-y-4">
      {/* Image Info */}
      {metadata && (
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">Image Information</CardTitle>
              <Badge
                variant={metadata.has_exif ? "success" : "outline"}
                className="text-[10px]"
              >
                {metadata.has_exif ? "EXIF Present" : "No EXIF"}
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
              {metadata.file_name && (
                <InfoItem label="File Name" value={metadata.file_name} />
              )}
              {metadata.file_size && (
                <InfoItem
                  label="File Size"
                  value={formatFileSize(metadata.file_size)}
                />
              )}
              <InfoItem
                label="Dimensions"
                value={`${metadata.width} x ${metadata.height}`}
              />
              {metadata.format && (
                <InfoItem label="Format" value={metadata.format} />
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* EXIF Data */}
      {metadata?.exif && metadata.has_exif && (
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <Camera className="h-4 w-4 text-muted-foreground" />
              <CardTitle className="text-base">EXIF Metadata</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
              {metadata.exif.make && (
                <InfoItem label="Camera Make" value={metadata.exif.make} />
              )}
              {metadata.exif.model && (
                <InfoItem label="Camera Model" value={metadata.exif.model} />
              )}
              {metadata.exif.software && (
                <InfoItem label="Software" value={metadata.exif.software} />
              )}
              {metadata.exif.datetime_original && (
                <InfoItem
                  label="Date Taken"
                  value={metadata.exif.datetime_original}
                />
              )}
              {metadata.exif.exposure_time && (
                <InfoItem
                  label="Exposure"
                  value={metadata.exif.exposure_time}
                />
              )}
              {metadata.exif.f_number && (
                <InfoItem
                  label="F-Number"
                  value={`f/${metadata.exif.f_number}`}
                />
              )}
              {metadata.exif.iso && (
                <InfoItem
                  label="ISO"
                  value={metadata.exif.iso.toString()}
                />
              )}
              {metadata.exif.focal_length && (
                <InfoItem
                  label="Focal Length"
                  value={`${metadata.exif.focal_length}mm`}
                />
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* AI Indicators */}
      {metadata && metadata.ai_indicators.length > 0 && (
        <Card className="border-verdict-ai/30">
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <AlertCircle className="h-4 w-4 text-verdict-ai" />
              <CardTitle className="text-base text-verdict-ai">
                AI Indicators Detected
              </CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {metadata.ai_indicators.map((indicator, i) => (
                <Badge
                  key={i}
                  variant="danger"
                  className="text-[10px]"
                >
                  {indicator}
                </Badge>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Anomalies */}
      {metadata && metadata.anomalies.length > 0 && (
        <Card className="border-verdict-uncertain/30">
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <AlertCircle className="h-4 w-4 text-verdict-uncertain" />
              <CardTitle className="text-base">Anomalies</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <ul className="space-y-1.5">
              {metadata.anomalies.map((anomaly, i) => (
                <li
                  key={i}
                  className="flex items-start gap-2 text-xs text-muted-foreground"
                >
                  <AlertCircle className="h-3 w-3 text-verdict-uncertain shrink-0 mt-0.5" />
                  {anomaly}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {/* Provenance */}
      {provenance && (
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Shield className="h-4 w-4 text-muted-foreground" />
                <CardTitle className="text-base">Provenance</CardTitle>
              </div>
              <Badge
                variant={
                  provenance.status === "verified"
                    ? "success"
                    : provenance.status === "tampered"
                      ? "danger"
                      : "outline"
                }
                className="text-[10px] capitalize"
              >
                {provenance.status === "missing"
                  ? "Not Available"
                  : provenance.status}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              {provenance.claimed_source && (
                <InfoItem
                  label="Claimed Source"
                  value={provenance.claimed_source}
                />
              )}
              {provenance.claimed_creator && (
                <InfoItem
                  label="Creator"
                  value={provenance.claimed_creator}
                />
              )}
              {provenance.creation_date && (
                <InfoItem
                  label="Creation Date"
                  value={provenance.creation_date}
                />
              )}
              <InfoItem
                label="Confidence"
                value={`${(provenance.confidence_score * 100).toFixed(0)}%`}
              />
            </div>

            {(provenance.trust_indicators.length > 0 ||
              provenance.warning_indicators.length > 0) && (
              <>
                <Separator />
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {provenance.trust_indicators.length > 0 && (
                    <div className="space-y-2">
                      <p className="text-xs font-medium text-verdict-real">
                        Trust Indicators
                      </p>
                      <ul className="space-y-1">
                        {provenance.trust_indicators.map((ind, i) => (
                          <li
                            key={i}
                            className="flex items-start gap-1.5 text-[11px] text-muted-foreground"
                          >
                            <Check className="h-3 w-3 text-verdict-real shrink-0 mt-0.5" />
                            {ind}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {provenance.warning_indicators.length > 0 && (
                    <div className="space-y-2">
                      <p className="text-xs font-medium text-verdict-uncertain">
                        Warning Indicators
                      </p>
                      <ul className="space-y-1">
                        {provenance.warning_indicators.map((ind, i) => (
                          <li
                            key={i}
                            className="flex items-start gap-1.5 text-[11px] text-muted-foreground"
                          >
                            <AlertCircle className="h-3 w-3 text-verdict-uncertain shrink-0 mt-0.5" />
                            {ind}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function InfoItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="space-y-0.5">
      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">
        {label}
      </p>
      <p className="text-sm font-medium truncate" title={value}>
        {value}
      </p>
    </div>
  );
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
