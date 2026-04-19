import { http } from "./http";
import type { TaskInfo } from "@/types/api";

export const tasksApi = {
  list(): Promise<TaskInfo[]> {
    return http.get<TaskInfo[]>("/tasks").then((r) => r.data);
  },
  get(taskId: string): Promise<TaskInfo> {
    return http.get<TaskInfo>(`/tasks/${taskId}`).then((r) => r.data);
  },
  cancel(taskId: string): Promise<void> {
    return http.delete(`/tasks/${taskId}`).then(() => undefined);
  },
};
