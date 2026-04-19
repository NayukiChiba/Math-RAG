<template>
  <div class="home-view">
    <el-row :gutter="16">
      <el-col :span="8">
        <el-card shadow="never">
          <template #header>
            <div class="card-title">
              <el-icon><DataBoard /></el-icon>
              <span>语料 & 索引</span>
            </div>
          </template>
          <template v-if="status">
            <div class="stat-item">
              <span>语料文档数</span>
              <strong>{{ status.corpusDocCount ?? "—" }}</strong>
            </div>
            <div class="stat-item">
              <span>BM25 索引</span>
              <el-tag :type="status.bm25Exists ? 'success' : 'info'" size="small">
                {{ status.bm25Exists ? "已构建" : "未构建" }}
              </el-tag>
            </div>
            <div class="stat-item">
              <span>BM25+ 索引</span>
              <el-tag :type="status.bm25plusExists ? 'success' : 'info'" size="small">
                {{ status.bm25plusExists ? "已构建" : "未构建" }}
              </el-tag>
            </div>
            <div class="stat-item">
              <span>向量索引</span>
              <el-tag :type="status.vectorExists ? 'success' : 'info'" size="small">
                {{ status.vectorExists ? "已构建" : "未构建" }}
              </el-tag>
            </div>
          </template>
          <el-button
            style="margin-top: 12px"
            type="primary"
            link
            @click="$router.push('/index')"
          >前往索引管理 →</el-button>
        </el-card>
      </el-col>

      <el-col :span="8">
        <el-card shadow="never">
          <template #header>
            <div class="card-title">
              <el-icon><Folder /></el-icon>
              <span>数据资产</span>
            </div>
          </template>
          <div class="stat-item">
            <span>原始 PDF</span>
            <strong>{{ rawCount }}</strong>
          </div>
          <div class="stat-item">
            <span>OCR 已处理书目</span>
            <strong>{{ processed?.ocrBooks.length ?? 0 }}</strong>
          </div>
          <div class="stat-item">
            <span>术语抽取书目</span>
            <strong>{{ processed?.termsBooks.length ?? 0 }}</strong>
          </div>
          <div class="stat-item">
            <span>结构化生成书目</span>
            <strong>{{ processed?.chunkBooks.length ?? 0 }}</strong>
          </div>
          <el-button
            style="margin-top: 12px"
            type="primary"
            link
            @click="$router.push('/ingest')"
          >前往 PDF 入库 →</el-button>
        </el-card>
      </el-col>

      <el-col :span="8">
        <el-card shadow="never">
          <template #header>
            <div class="card-title">
              <el-icon><List /></el-icon>
              <span>最近任务</span>
            </div>
          </template>
          <el-empty
            v-if="taskStore.tasks.length === 0"
            description="暂无任务"
            :image-size="60"
          />
          <div v-else>
            <div
              v-for="t in taskStore.tasks.slice(0, 5)"
              :key="t.taskId"
              class="task-row"
              @click="$router.push(`/tasks/${t.taskId}`)"
            >
              <span class="cmd">{{ t.command }}</span>
              <el-tag size="small" :type="statusType(t.status)">
                {{ t.status }}
              </el-tag>
            </div>
          </div>
          <el-button
            style="margin-top: 12px"
            type="primary"
            link
            @click="$router.push('/tasks')"
          >查看全部 →</el-button>
        </el-card>
      </el-col>
    </el-row>

    <el-card style="margin-top: 16px" shadow="never">
      <template #header>
        <div class="card-title">快捷入口</div>
      </template>
      <div class="shortcut-grid">
        <ShortcutCard
          title="RAG 问答"
          desc="流式数学术语问答 + 引用依据"
          icon="ChatDotSquare"
          to="/chat"
        />
        <ShortcutCard
          title="研究线"
          desc="14 个评测 / 实验 / 报告命令"
          icon="Histogram"
          to="/research"
        />
        <ShortcutCard
          title="报告中心"
          desc="浏览 outputs/log 和 outputs/reports"
          icon="Document"
          to="/reports"
        />
        <ShortcutCard
          title="术语统计"
          desc="书籍 / 学科分布、字段覆盖率"
          icon="DataAnalysis"
          to="/stats"
        />
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import {
  ChatDotSquare,
  DataAnalysis,
  DataBoard,
  Document,
  Folder,
  Histogram,
  List,
} from "@element-plus/icons-vue";
import { indexApi } from "@/api/index";
import { dataApi } from "@/api/data";
import { useTaskStore } from "@/stores/taskStore";
import type { IndexStatus, ProcessedInfo, TaskInfo } from "@/types/api";
import ShortcutCard from "@/components/ShortcutCard.vue";

defineOptions({ name: "HomeView" });

const status = ref<IndexStatus | null>(null);
const processed = ref<ProcessedInfo | null>(null);
const rawCount = ref(0);
const taskStore = useTaskStore();

function statusType(s: TaskInfo["status"]) {
  return (
    { running: "warning", succeeded: "success", failed: "danger" } as Record<
      string,
      "warning" | "success" | "danger" | "info"
    >
  )[s] || "info";
}

onMounted(async () => {
  try {
    [status.value, processed.value, rawCount.value] = await Promise.all([
      indexApi.status(),
      dataApi.processed(),
      dataApi.raw().then((r) => r.length),
    ]);
  } catch (_) {
    /* ignore */
  }
  taskStore.refresh();
});

computed(() => status.value);
</script>

<style scoped>
.card-title {
  display: flex;
  align-items: center;
  gap: 6px;
  font-weight: 500;
}

.stat-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 0;
  border-bottom: 1px dashed #ebeef5;
}

.stat-item:last-child {
  border-bottom: none;
}

.task-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 4px;
  cursor: pointer;
  border-radius: 4px;
}

.task-row:hover {
  background-color: #f5f7fa;
}

.cmd {
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 13px;
  color: #606266;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 70%;
}

.shortcut-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
}
</style>
