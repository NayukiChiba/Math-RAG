import { http } from "./http";
import type { ProcessedInfo, RawPdfInfo } from "@/types/api";

export const dataApi = {
  raw(): Promise<RawPdfInfo[]> {
    return http.get<RawPdfInfo[]>("/data/raw").then((r) => r.data);
  },
  processed(): Promise<ProcessedInfo> {
    return http.get<ProcessedInfo>("/data/processed").then((r) => r.data);
  },
};
