<template>
  <el-card class="task-panel" shadow="never">
    <template #header>
      <div class="task-head">
        <div>
          <span class="label">任务：</span>
          <span class="value">{{ command }}</span>
        </div>
        <div class="task-head-right">
          <el-tag :type="statusTagType">{{ statusText }}</el-tag>
          <el-button
            v-if="status === 'running' || status === 'pending'"
            size="small"
            type="danger"
            plain
            @click="cancel"
          >取消</el-button>
        </div>
      </div>
    </template>

    <el-progress
      v-if="progress !== null"
      :percentage="Math.round(progress * 100)"
      :status="progressStatus"
    />

    <LogViewer :lines="lines" />

    <div v-if="errorMessage" class="error-message">
      <el-alert
        type="error"
        :title="errorMessage"
        :closable="false"
        show-icon
      />
    </div>
  </el-card>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import { ElMessage } from "element-plus";
import { useWebSocket } from "@/composables/useWebSocket";
import LogViewer, { type LogLine } from "./LogViewer.vue";
import { tasksApi } from "@/api/tasks";
import type { TaskInfo, WsTaskEvent } from "@/types/api";

const props = defineProps<{
  taskId: string;
  command?: string;
}>();

const emit = defineEmits<{
  done: [status: TaskInfo["status"]];
}>();

const command = computed(() => props.command ?? props.taskId);
const status = ref<TaskInfo["status"]>("pending");
const progress = ref<number | null>(null);
const errorMessage = ref<string | null>(null);
const lines = ref<LogLine[]>([]);

const statusText = computed(() => {
  return (
    {
      pending: "等待中",
      running: "运行中",
      succeeded: "已完成",
      failed: "失败",
      cancelled: "已取消",
    } as Record<TaskInfo["status"], string>
  )[status.value];
});

const statusTagType = computed(() => {
  return (
    {
      pending: "info",
      running: "warning",
      succeeded: "success",
      failed: "danger",
      cancelled: "info",
    } as Record<TaskInfo["status"], string>
  )[status.value] as "info" | "warning" | "success" | "danger";
});

const progressStatus = computed(() => {
  if (status.value === "failed") return "exception" as const;
  if (status.value === "succeeded") return "success" as const;
  return undefined;
});

const { connect, disconnect } = useWebSocket<WsTaskEvent>(`/ws/tasks/${props.taskId}`);

function handleEvent(event: WsTaskEvent): void {
  if (event.type === "log") {
    lines.value.push({ stream: event.stream, text: event.line });
    if (lines.value.length > 2000) {
      lines.value = lines.value.slice(-2000);
    }
    return;
  }
  if (event.type === "status") {
    status.value = event.status;
    return;
  }
  if (event.type === "progress") {
    progress.value = event.progress;
    return;
  }
  if (event.type === "done") {
    status.value = event.status;
    if (event.error) errorMessage.value = event.error;
    emit("done", event.status);
    return;
  }
  if ((event.type as string) === "error") {
    errorMessage.value = (event as { error?: string }).error || "未知错误";
  }
}

async function cancel(): Promise<void> {
  try {
    await tasksApi.cancel(props.taskId);
    ElMessage.success("已发送取消请求");
  } catch (e) {
    ElMessage.error((e as Error).message);
  }
}

watch(
  () => props.taskId,
  async (newId) => {
    disconnect();
    if (!newId) return;
    lines.value = [];
    status.value = "pending";
    progress.value = null;
    errorMessage.value = null;
    await connect({ onMessage: handleEvent });
  },
  { immediate: true },
);

onBeforeUnmount(disconnect);
</script>

<style scoped>
.task-panel {
  margin-top: 16px;
}

.task-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.task-head .label {
  color: #909399;
}

.task-head .value {
  font-family: "SFMono-Regular", Consolas, monospace;
  color: #303133;
  margin-right: 12px;
}

.task-head-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

.error-message {
  margin-top: 12px;
}
</style>
