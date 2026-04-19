import { http } from "./http";

export const statsApi = {
  get(): Promise<{
    available: boolean;
    statsDir: string;
    files?: Record<string, unknown>;
  }> {
    return http.get("/stats").then((r) => r.data);
  },
  figures(): Promise<{ label: string; filename: string; relPath: string }[]> {
    return http.get("/stats/figures").then((r) => r.data);
  },
};
