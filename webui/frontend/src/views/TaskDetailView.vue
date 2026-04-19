<template>
  <div>
    <el-page-header @back="$router.push('/tasks')" :content="taskId" />

    <el-card v-if="info" style="margin-top: 12px" shadow="never">
      <el-descriptions :column="2" border>
        <el-descriptions-item label="命令">{{ info.command }}</el-descriptions-item>
        <el-descriptions-item label="状态">
          <el-tag :type="tagType(info.status)">{{ info.status }}</el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="创建时间">{{ info.createdAt }}</el-descriptions-item>
        <el-descriptions-item label="开始时间">{{ info.startedAt ?? "—" }}</el-descriptions-item>
        <el-descriptions-item label="结束时间">{{ info.finishedAt ?? "—" }}</el-descriptions-item>
        <el-descriptions-item label="进度">
          {{ info.progress != null ? `${Math.round(info.progress * 100)}%` : "—" }}
        </el-descriptions-item>
        <el-descriptions-item label="参数" :span="2">
          <pre class="args-pre">{{ JSON.stringify(info.args, null, 2) }}</pre>
        </el-descriptions-item>
        <el-descriptions-item v-if="info.errorMessage" label="错误" :span="2">
          <pre class="args-pre error">{{ info.errorMessage }}</pre>
        </el-descriptions-item>
      </el-descriptions>
    </el-card>

    <TaskPanel :task-id="taskId" :command="info?.command" />
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import TaskPanel from "@/components/TaskPanel.vue";
import { tasksApi } from "@/api/tasks";
import type { TaskInfo } from "@/types/api";

const route = useRoute();
const taskId = computed(() => route.params.taskId as string);
const info = ref<TaskInfo | null>(null);

function tagType(s: TaskInfo["status"]) {
  return (
    {
      running: "warning",
      succeeded: "success",
      failed: "danger",
      cancelled: "info",
      pending: "info",
    } as Record<TaskInfo["status"], "warning" | "success" | "danger" | "info">
  )[s];
}

onMounted(async () => {
  try {
    info.value = await tasksApi.get(taskId.value);
  } catch (_) {
    /* ignore */
  }
});
</script>

<style scoped>
.args-pre {
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 12px;
  background-color: #f5f7fa;
  padding: 8px;
  border-radius: 4px;
  margin: 0;
  max-height: 240px;
  overflow: auto;
}

.args-pre.error {
  color: #d9534f;
}
</style>
