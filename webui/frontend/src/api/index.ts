import { http } from "./http";
import type { IndexStatus, TaskRef } from "@/types/api";

export interface BuildIndexPayload {
  rebuild?: boolean;
  skipBm25?: boolean;
  skipBm25plus?: boolean;
  skipVector?: boolean;
  vectorModel?: string;
  batchSize?: number;
}

export const indexApi = {
  status(): Promise<IndexStatus> {
    return http.get<IndexStatus>("/index/status").then((r) => r.data);
  },
  build(payload: BuildIndexPayload): Promise<TaskRef> {
    return http.post<TaskRef>("/index/build", payload).then((r) => r.data);
  },
};
