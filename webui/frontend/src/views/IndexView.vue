<template>
  <el-row :gutter="16">
    <el-col :span="14">
      <el-card shadow="never">
        <template #header>
          <div class="card-title">
            <el-icon><DataLine /></el-icon>
            <span>索引状态</span>
          </div>
        </template>
        <el-descriptions :column="1" border v-if="status">
          <el-descriptions-item label="语料文件">
            <el-tag :type="status.corpusExists ? 'success' : 'info'">
              {{ status.corpusExists ? "已构建" : "未构建" }}
            </el-tag>
            <span v-if="status.corpusDocCount != null" style="margin-left: 8px">
              共 {{ status.corpusDocCount }} 条文档
            </span>
          </el-descriptions-item>
          <el-descriptions-item label="BM25 索引">
            <el-tag :type="status.bm25Exists ? 'success' : 'info'">
              {{ status.bm25Exists ? "已构建" : "未构建" }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="BM25+ 索引">
            <el-tag :type="status.bm25plusExists ? 'success' : 'info'">
              {{ status.bm25plusExists ? "已构建" : "未构建" }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="向量索引">
            <el-tag :type="status.vectorExists ? 'success' : 'info'">
              {{ status.vectorExists ? "已构建" : "未构建" }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="索引目录">
            <code>{{ status.retrievalDir }}</code>
          </el-descriptions-item>
        </el-descriptions>
        <el-button style="margin-top: 12px" @click="refresh">刷新</el-button>
      </el-card>
    </el-col>

    <el-col :span="10">
      <el-card shadow="never">
        <template #header>
          <div class="card-title">
            <el-icon><Refresh /></el-icon>
            <span>构建 / 重建索引</span>
          </div>
        </template>
        <el-form label-width="120px" :disabled="!!currentTaskId">
          <el-form-item label="重建全部">
            <el-switch v-model="form.rebuild" />
          </el-form-item>
          <el-form-item label="跳过 BM25">
            <el-switch v-model="form.skipBm25" />
          </el-form-item>
          <el-form-item label="跳过 BM25+">
            <el-switch v-model="form.skipBm25plus" />
          </el-form-item>
          <el-form-item label="跳过向量">
            <el-switch v-model="form.skipVector" />
          </el-form-item>
          <el-form-item label="向量模型">
            <el-input v-model="form.vectorModel" placeholder="默认使用 config" />
          </el-form-item>
          <el-form-item label="批次大小">
            <el-input-number v-model="form.batchSize" :min="1" :max="128" />
          </el-form-item>
          <el-form-item>
            <el-button type="primary" @click="start">启动构建</el-button>
          </el-form-item>
        </el-form>
      </el-card>
    </el-col>
  </el-row>

  <TaskPanel
    v-if="currentTaskId"
    :task-id="currentTaskId"
    command="cli.build-index"
    @done="onDone"
  />
</template>

<script setup lang="ts">
import { onMounted, reactive, ref } from "vue";
import { ElMessage } from "element-plus";
import { DataLine, Refresh } from "@element-plus/icons-vue";
import TaskPanel from "@/components/TaskPanel.vue";
import { indexApi } from "@/api/index";
import type { IndexStatus, TaskInfo } from "@/types/api";

const status = ref<IndexStatus | null>(null);
const currentTaskId = ref<string | null>(null);

const form = reactive({
  rebuild: false,
  skipBm25: false,
  skipBm25plus: false,
  skipVector: false,
  vectorModel: "",
  batchSize: 32,
});

async function refresh(): Promise<void> {
  status.value = await indexApi.status();
}

async function start(): Promise<void> {
  try {
    const res = await indexApi.build({
      rebuild: form.rebuild,
      skipBm25: form.skipBm25,
      skipBm25plus: form.skipBm25plus,
      skipVector: form.skipVector,
      vectorModel: form.vectorModel || undefined,
      batchSize: form.batchSize,
    });
    currentTaskId.value = res.taskId;
    ElMessage.success("任务已提交");
  } catch (e) {
    ElMessage.error((e as Error).message);
  }
}

async function onDone(st: TaskInfo["status"]): Promise<void> {
  if (st === "succeeded") ElMessage.success("索引构建完成");
  else if (st === "failed") ElMessage.error("索引构建失败");
  await refresh();
}

onMounted(refresh);
</script>

<style scoped>
.card-title {
  display: flex;
  align-items: center;
  gap: 6px;
}
</style>
