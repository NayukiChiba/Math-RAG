import { defineStore } from "pinia";
import { ref } from "vue";
import { tasksApi } from "@/api/tasks";
import type { TaskInfo } from "@/types/api";

export const useTaskStore = defineStore("tasks", () => {
  const tasks = ref<TaskInfo[]>([]);
  const loading = ref(false);

  async function refresh(): Promise<void> {
    loading.value = true;
    try {
      tasks.value = await tasksApi.list();
    } finally {
      loading.value = false;
    }
  }

  async function cancel(taskId: string): Promise<void> {
    await tasksApi.cancel(taskId);
    await refresh();
  }

  return { tasks, loading, refresh, cancel };
});
