import { http } from "./http";
import type { FigureInfo, ReportRunInfo } from "@/types/api";

export const reportsApi = {
  runs(): Promise<ReportRunInfo[]> {
    return http.get<ReportRunInfo[]>("/reports").then((r) => r.data);
  },
  runTree(runId: string): Promise<{
    runId: string;
    path: string;
    files: { relPath: string; sizeBytes: number; modifiedAt: string }[];
  }> {
    return http.get(`/reports/${runId}/tree`).then((r) => r.data);
  },
  runFileUrl(runId: string, path: string): string {
    return `/api/reports/${runId}/file?path=${encodeURIComponent(path)}`;
  },
  runFileText(runId: string, path: string): Promise<string> {
    return http
      .get(`/reports/${runId}/file`, {
        params: { path },
        responseType: "text",
        transformResponse: [(data) => data],
      })
      .then((r) => r.data as string);
  },
  publishedTree(): Promise<{
    path: string;
    files: { relPath: string; sizeBytes: number; modifiedAt: string }[];
  }> {
    return http.get("/reports-published/tree").then((r) => r.data);
  },
  publishedFileUrl(path: string): string {
    return `/api/reports-published/file?path=${encodeURIComponent(path)}`;
  },
  figures(): Promise<FigureInfo[]> {
    return http.get<FigureInfo[]>("/figures").then((r) => r.data);
  },
  figureUrl(path: string): string {
    return `/api/figures/file?path=${encodeURIComponent(path)}`;
  },
};
