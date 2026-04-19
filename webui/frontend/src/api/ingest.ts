import { http } from "./http";
import type { TaskRef } from "@/types/api";

export interface IngestPayload {
  pdf: string;
  ocrStartPage?: number;
  extractStartPage?: number;
  generateStartPage?: number;
  skipGeneration?: boolean;
  skipIndex?: boolean;
  rebuildIndex?: boolean;
  skipBm25?: boolean;
  skipBm25plus?: boolean;
  skipVector?: boolean;
  vectorModel?: string;
  batchSize?: number;
}

export const ingestApi = {
  upload(file: File): Promise<{ name: string; sizeBytes: number; path: string }> {
    const form = new FormData();
    form.append("file", file);
    return http
      .post("/ingest/upload", form, {
        headers: { "Content-Type": "multipart/form-data" },
      })
      .then((r) => r.data);
  },
  start(payload: IngestPayload): Promise<TaskRef> {
    return http.post<TaskRef>("/ingest", payload).then((r) => r.data);
  },
};
