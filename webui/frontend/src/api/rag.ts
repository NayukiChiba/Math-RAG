import { http } from "./http";
import type { RagQueryResult, TaskRef } from "@/types/api";

export interface RagQueryPayload {
  query: string;
  useRag: boolean;
  temperature?: number;
  topP?: number;
  maxNewTokens?: number;
}

export const ragApi = {
  query(payload: RagQueryPayload): Promise<RagQueryResult> {
    return http.post<RagQueryResult>("/rag/query", payload).then((r) => r.data);
  },
  batch(
    queries: string[],
    options: Omit<RagQueryPayload, "query"> = { useRag: true },
  ): Promise<TaskRef> {
    return http
      .post<TaskRef>("/rag/batch", { queries, ...options })
      .then((r) => r.data);
  },
};
