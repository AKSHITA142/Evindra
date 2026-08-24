import type {
  Dataset,
  Job,
  JobLogEntry,
  StartJobResponse,
  UploadResponse,
  ExperimentResult,
  Report,
  DashboardData,
} from "@/types/api";

const rawBackendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8000";
const backendUrl = rawBackendUrl.replace("//localhost", "//127.0.0.1");
const apiPrefix = process.env.NEXT_PUBLIC_API_PREFIX || "/api/v1";

// In browser, communicate directly with FastAPI backend to avoid Next.js 10MB proxy buffer and socket hang-ups
const BASE_URL = typeof window !== "undefined" && backendUrl
  ? `${backendUrl}${apiPrefix}`
  : apiPrefix;

// ── Unique client ID for per-browser session isolation ────────────────
function getClientId(): string {
  if (typeof window === "undefined") return "ssr";
  let id = localStorage.getItem("datapilot-client-id");
  if (!id) {
    id = `client_${crypto.randomUUID?.() ?? Math.random().toString(36).slice(2)}`;
    localStorage.setItem("datapilot-client-id", id);
  }
  return id;
}

// ── Error class ───────────────────────────────────────────────────────

class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string,
    public errorCode?: string
  ) {
    super(detail);
    this.name = "ApiError";
  }
}

// ── Core request helper ───────────────────────────────────────────────

/**
 * Core fetch wrapper that:
 * 1. Adds X-Client-Id header for session isolation
 * 2. Unwraps backend `{ data: ..., meta: {} }` envelopes
 * 3. Maps backend error format `{ error_code, message }` to ApiError
 */
async function request<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      "X-Client-Id": getClientId(),
      ...(options?.headers ?? {}),
    },
    ...options,
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    let errorCode: string | undefined;
    try {
      const body = await res.json();
      // Backend returns { error_code, message, details } on errors
      detail = body.message ?? body.detail ?? detail;
      errorCode = body.error_code;
    } catch {
      // ignore parse errors
    }
    throw new ApiError(res.status, detail, errorCode);
  }

  const json = await res.json();

  // Unwrap SuccessResponse envelope { data: T, meta: {} }
  if (json !== null && typeof json === "object" && "data" in json) {
    return json.data as T;
  }

  // Fallback: return as-is if no envelope
  return json as T;
}

// ── Dataset APIs ──────────────────────────────────────────────────────

export async function uploadDataset(
  file: File,
  mission: string,
  taskType: string = "general"
): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  if (mission) {
    formData.append("mission", mission);
  }
  formData.append("task_type", taskType);

  const res = await fetch(`${BASE_URL}/upload`, {
    method: "POST",
    headers: {
      "X-Client-Id": getClientId(),
    },
    body: formData,
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body.message ?? body.detail ?? detail;
    } catch {
      // ignore
    }
    throw new ApiError(res.status, detail);
  }

  const json = await res.json();
  // Unwrap envelope
  if (json !== null && typeof json === "object" && "data" in json) {
    return json.data as UploadResponse;
  }
  return json;
}

export async function getDataset(datasetId: string): Promise<Dataset> {
  return request<Dataset>(`/datasets/${datasetId}`);
}

export async function listDatasets(skip = 0, limit = 50): Promise<Dataset[]> {
  return request<Dataset[]>(`/datasets?skip=${skip}&limit=${limit}`);
}

// ── Job APIs ──────────────────────────────────────────────────────────

export async function startJob(
  datasetId: string,
  mission: string,
  taskType: string = "general"
): Promise<StartJobResponse> {
  return request<StartJobResponse>("/jobs/start", {
    method: "POST",
    // Send user_goal, mission, and task_type so backend receives explicit task_type selection
    body: JSON.stringify({ dataset_id: datasetId, user_goal: mission, mission, task_type: taskType }),
  });
}

export async function getJob(jobId: string): Promise<Job> {
  return request<Job>(`/jobs/${jobId}`);
}

export async function getJobLogs(jobId: string): Promise<JobLogEntry[]> {
  return request<JobLogEntry[]>(`/jobs/${jobId}/logs`);
}

export async function cancelJob(jobId: string): Promise<void> {
  await request(`/jobs/${jobId}/cancel`, { method: "POST" });
}

// ── Experiment APIs ───────────────────────────────────────────────────

export async function getExperiments(
  jobId: string
): Promise<ExperimentResult[]> {
  return request<ExperimentResult[]>(`/experiments/${jobId}`);
}

export async function getExperiment(
  experimentId: string
): Promise<ExperimentResult> {
  return request<ExperimentResult>(`/experiments/detail/${experimentId}`);
}

// ── Report APIs ───────────────────────────────────────────────────────

export async function getReport(jobId: string): Promise<Report> {
  return request<Report>(`/reports/${jobId}`);
}

export async function downloadReport(
  reportId: string,
  format: "html" | "markdown" = "markdown"
): Promise<Blob> {
  const res = await fetch(
    `${BASE_URL}/reports/${reportId}/download?format=${format}`,
    {
      headers: { "X-Client-Id": getClientId() },
    }
  );
  if (!res.ok) throw new ApiError(res.status, "Download failed");
  return res.blob();
}

// ── Dashboard APIs ────────────────────────────────────────────────────

export async function getDashboard(): Promise<DashboardData> {
  return request<DashboardData>("/dashboard");
}

export { ApiError };
