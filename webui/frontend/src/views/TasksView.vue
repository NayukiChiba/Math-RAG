<template>
  <el-card shadow="never">
    <template #header>
      <div class="card-title">
        <span>任务中心</span>
        <el-button size="small" @click="taskStore.refresh()" :loading="taskStore.loading">
          刷新
        </el-button>
      </div>
    </template>

    <el-table :data="taskStore.tasks" size="small" stripe>
      <el-table-column prop="command" label="命令" width="240" />
      <el-table-column label="状态" width="110">
        <template #default="scope">
          <el-tag :type="tagType(scope.row.status)" size="small">
            {{ scope.row.status }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="createdAt" label="创建时间" width="220" />
      <el-table-column prop="startedAt" label="开始时间" width="220" />
      <el-table-column prop="finishedAt" label="结束时间" width="220" />
      <el-table-column label="操作" width="160">
        <template #default="scope">
          <el-button
            size="small"
            type="primary"
            link
            @click="$router.push(`/tasks/${scope.row.taskId}`)"
          >详情</el-button>
          <el-button
            v-if="scope.row.status === 'running' || scope.row.status === 'pending'"
            size="small"
            type="danger"
            link
            @click="cancel(scope.row.taskId)"
          >取消</el-button>
        </template>
      </el-table-column>
    </el-table>
  </el-card>
</template>

<script setup lang="ts">
import { onMounted } from "vue";
import { ElMessage, ElMessageBox } from "element-plus";
import { useTaskStore } from "@/stores/taskStore";
import type { TaskInfo } from "@/types/api";

defineOptions({ name: "TasksView" });

const taskStore = useTaskStore();

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

async function cancel(id: string): Promise<void> {
  try {
    await ElMessageBox.confirm("确定取消这个任务吗？", "确认");
    await taskStore.cancel(id);
    ElMessage.success("已取消");
  } catch {
    /* user cancelled */
  }
}

onMounted(() => taskStore.refresh());
</script>

<style scoped>
.card-title {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
</style>
