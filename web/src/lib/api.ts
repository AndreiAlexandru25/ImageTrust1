import type { ComprehensiveAnalysisResult } from "./types";

// Use Next.js rewrites in dev (proxies /api/* to localhost:8000),
// or explicit API_BASE_URL for production deployments
const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "";

export class APIError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "APIError";
  }
}

export async function analyzeImage(
  file: File,
  options: {
    includeMetadata?: boolean;
    includeExplainability?: boolean;
    includeForensics?: boolean;
  } = {},
): Promise<ComprehensiveAnalysisResult> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append(
    "include_metadata",
    String(options.includeMetadata ?? true),
  );
  formData.append(
    "include_explainability",
    String(options.includeExplainability ?? true),
  );
  formData.append(
    "include_forensics",
    String(options.includeForensics ?? true),
  );

  const response = await fetch(
    `${API_BASE_URL}/api/v1/analyze/comprehensive`,
    {
      method: "POST",
      body: formData,
    },
  );

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Unknown error" }));
    throw new APIError(response.status, error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

export async function healthCheck(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.ok;
  } catch {
    return false;
  }
}
