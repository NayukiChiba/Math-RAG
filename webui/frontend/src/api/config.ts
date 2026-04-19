import { http } from "./http";

export type EngineName = "local" | "api";

export interface EnginesState {
  ocr: EngineName;
  dataGen: EngineName;
  rag: EngineName;
}

export const configApi = {
  get(): Promise<Record<string, unknown>> {
    return http.get<Record<string, unknown>>("/config").then((r) => r.data);
  },
  patch(section: string, updates: Record<string, unknown>): Promise<unknown> {
    return http.patch("/config", { section, updates }).then((r) => r.data);
  },
  getEngines(): Promise<EnginesState> {
    return http.get<EnginesState>("/config/engines").then((r) => r.data);
  },
  patchEngines(payload: Partial<EnginesState>): Promise<EnginesState> {
    return http
      .patch<EnginesState>("/config/engines", payload)
      .then((r) => r.data);
  },
};
