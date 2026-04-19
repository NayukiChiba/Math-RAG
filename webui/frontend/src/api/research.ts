import { http } from "./http";
import type { TaskRef } from "@/types/api";

export const researchApi = {
  listCommands(): Promise<Record<string, string>> {
    return http
      .get<Record<string, string>>("/research/commands")
      .then((r) => r.data);
  },
  run(command: string, args: string[] = []): Promise<TaskRef> {
    return http
      .post<TaskRef>(`/research/${command}`, { args })
      .then((r) => r.data);
  },
};
