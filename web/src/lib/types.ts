export type Verdict = "real" | "ai_generated" | "manipulated" | "uncertain" | "screenshot";

export type Confidence = "very_low" | "low" | "medium" | "high" | "very_high";

export type ProvenanceStatus =
  | "verified"
  | "partial"
  | "missing"
  | "tampered"
  | "unknown";

export interface IndividualResult {
  method: string;
  ai_probability: number;
  confidence: number;
  weight: number;
  details: Record<string, unknown>;
}

export interface CalibratedEnsemble {
  raw_probs: Record<string, number>;
  calibrated_probs: Record<string, number>;
  ensemble_avg_prob: number;
  ensemble_min_prob: number;
  verdict: string;
  verdict_text: string;
  strategy: string;
  uncertain_low: number;
  uncertain_high: number;
  ensemble_std: number;
  model_agreement: number;
}

export interface UncertaintyInfo {
  score: number;
  method: string;
  should_abstain: boolean;
  confidence_level: string;
  ensemble_std: number;
}

export interface MetaClassifierResult {
  ai_probability: number;
  confidence: number;
  is_uncertain: boolean;
  raw_logit: number;
  feature_importances?: Record<string, number>;
}

export interface ConformalPrediction {
  prediction_set: string[];
  set_size: number;
  coverage_level: number;
  is_uncertain: boolean;
  threshold: number;
  conformity_scores: number[];
}

export interface HighlightedRegion {
  bbox: [number, number, number, number];
  center: [number, number];
  activation: number;
  area: number;
  description: string;
}

export interface GradCAMData {
  heatmap_base64: string;
  overlay_base64: string;
  highlighted_regions: HighlightedRegion[];
  activation_score: number;
  layer_name: string;
}

export interface EvidenceItem {
  type: string;
  severity: "info" | "warning" | "critical";
  description: string;
  details?: string;
}

export interface ForensicsData {
  primary_verdict: string;
  authenticity_score: number;
  evidence: EvidenceItem[];
  copy_move_detected: boolean;
  splicing_detected: boolean;
}

export interface EXIFData {
  make?: string;
  model?: string;
  software?: string;
  datetime_original?: string;
  exposure_time?: string;
  f_number?: number;
  iso?: number;
  focal_length?: number;
  gps_latitude?: number;
  gps_longitude?: number;
  has_camera_info: boolean;
}

export interface MetadataInfo {
  has_exif: boolean;
  exif?: EXIFData;
  ai_indicators: string[];
  anomalies: string[];
  file_name?: string;
  file_size?: number;
  width: number;
  height: number;
  format?: string;
}

export interface ProvenanceInfo {
  status: ProvenanceStatus;
  trust_indicators: string[];
  warning_indicators: string[];
  claimed_source?: string;
  claimed_creator?: string;
  creation_date?: string;
  confidence_score: number;
}

export interface ScreenshotData {
  is_screenshot: boolean;
  probability: number;
  confidence: number;
  indicators: string[];
}

export interface HotRegion {
  row: number;
  col: number;
  x: number;
  y: number;
  width: number;
  height: number;
  ai_probability: number;
  severity: "info" | "warning" | "critical";
}

export interface LocalizationData {
  heatmap_base64: string;
  overlay_base64: string;
  grid_shape: [number, number];
  hot_regions: HotRegion[];
  mean_ai_prob: number;
  max_ai_prob: number;
  n_patches: number;
  n_models_used: number;
}

export interface TimingBreakdown {
  ml_models_ms: number;
  frequency_ms: number;
  noise_ms: number;
  texture_ms: number;
  edge_ms: number;
  color_ms: number;
  calibrated_ensemble_ms: number;
  ensemble_ms: number;
  gradcam_ms?: number;
  metadata_ms?: number;
  localization_ms?: number;
  screenshot_ms?: number;
  total_ms: number;
}

export interface DetectionSummary {
  raw_verdict: string;
  raw_ai_probability: number;
  cnn_avg: number;
  cnn_verdict: string;
  hf_avg: number;
  hf_verdict: string;
  hf_ai_count: number;
  hf_total: number;
  signal_avg: number;
  signal_ai_count: number;
  signal_total: number;
  cnn_ai_count: number;
  cnn_total: number;
  loc_max_ai_prob: number;
  loc_max_zscore: number;
  loc_critical_count: number;
  loc_warning_count: number;
  models_agree_with_verdict: boolean;
}

export interface ComprehensiveAnalysisResult {
  analysis_id: string;
  verdict: Verdict;
  ai_probability: number;
  confidence: number;
  confidence_level: Confidence;
  override_applied?: boolean;
  override_reason?: string;
  raw_verdict?: Verdict;
  raw_ai_probability?: number;
  detection_summary?: DetectionSummary;
  votes: {
    ai: number;
    real: number;
    total: number;
  };
  individual_results: IndividualResult[];
  calibrated_ensemble?: CalibratedEnsemble;
  uncertainty?: UncertaintyInfo;
  meta_classifier?: MetaClassifierResult;
  conformal_prediction?: ConformalPrediction;
  gradcam?: GradCAMData;
  forensics?: ForensicsData;
  metadata?: MetadataInfo;
  provenance?: ProvenanceInfo;
  localization?: LocalizationData;
  screenshot?: ScreenshotData;
  timing_breakdown?: TimingBreakdown;
  processing_time_ms: number;
  image_dimensions: [number, number];
}

export interface AnalysisState {
  status: "idle" | "uploading" | "analyzing" | "complete" | "error";
  progress: number;
  currentStep: string;
  result: ComprehensiveAnalysisResult | null;
  imagePreview: string | null;
  imageFile: File | null;
  error: string | null;
}

export type AnalysisStep = {
  id: string;
  label: string;
  description: string;
  status: "pending" | "active" | "complete" | "error";
};
