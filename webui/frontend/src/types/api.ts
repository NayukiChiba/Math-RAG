export interface TaskInfo {
  taskId: string;
  command: string;
  status: "pending" | "running" | "succeeded" | "failed" | "cancelled";
  createdAt: string;
  startedAt?: string | null;
  finishedAt?: string | null;
  progress?: number | null;
  message?: string | null;
  args?: Record<string, unknown> | null;
  errorMessage?: string | null;
  logTail: string[];
}

export interface TaskRef {
  taskId: string;
}

export interface IndexStatus {
  corpusExists: boolean;
  corpusDocCount?: number | null;
  bm25Exists: boolean;
  bm25plusExists: boolean;
  vectorExists: boolean;
  retrievalDir: string;
}

export interface RawPdfInfo {
  name: string;
  sizeBytes: number;
  modifiedAt: string;
}

export interface ProcessedInfo {
  ocrBooks: string[];
  termsBooks: string[];
  chunkBooks: string[];
}

export interface ReportRunInfo {
  runId: string;
  path: string;
  hasFinalReport: boolean;
  hasComparison: boolean;
  hasFullEval: boolean;
  createdAt: string;
}

export interface FigureInfo {
  relPath: string;
  sizeBytes: number;
  modifiedAt: string;
}

export interface RagRetrievalItem {
  rank?: number;
  term: string;
  subject?: string;
  source?: string;
  page?: number | null;
  score: number;
  text?: string;
}

export interface RagSource {
  source: string;
  page?: number | null;
}

export interface RagLatency {
  retrieval_ms: number;
  generation_ms: number;
  total_ms: number;
}

export interface RagQueryResult {
  query: string;
  answer: string;
  retrievedTerms: RagRetrievalItem[];
  sources: RagSource[];
  latency: RagLatency;
}

export type WsRagEvent =
  | { type: "status"; message: string }
  | {
      type: "retrieval";
      retrievedTerms: RagRetrievalItem[];
      sources: RagSource[];
    }
  | { type: "token"; delta: string }
  | { type: "done"; latency: RagLatency }
  | { type: "error"; error: string }
  | { type: "ping" };

export type WsTaskEvent =
  | { type: "log"; stream: "stdout" | "stderr"; line: string }
  | { type: "status"; status: TaskInfo["status"]; message?: string }
  | { type: "progress"; progress: number; message?: string | null }
  | { type: "done"; status: TaskInfo["status"]; result?: unknown; error?: string }
  | { type: "ping" };
