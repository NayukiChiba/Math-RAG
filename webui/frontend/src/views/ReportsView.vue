<template>
  <el-card shadow="never">
    <template #header>
      <div class="card-title">
        <span>报告中心</span>
        <el-button size="small" @click="refresh" :loading="loading">刷新</el-button>
      </div>
    </template>

    <el-tabs v-model="tab">
      <el-tab-pane name="runs" label="历史运行 (outputs/log)">
        <el-table :data="runs" size="small" stripe>
          <el-table-column prop="runId" label="运行 ID" width="220" />
          <el-table-column label="成果">
            <template #default="scope">
              <el-space wrap>
                <el-tag v-if="scope.row.hasFinalReport" type="success" size="small">
                  final_report.md
                </el-tag>
                <el-tag v-if="scope.row.hasComparison" type="warning" size="small">
                  comparison
                </el-tag>
                <el-tag v-if="scope.row.hasFullEval" size="small">
                  full_eval
                </el-tag>
              </el-space>
            </template>
          </el-table-column>
          <el-table-column prop="createdAt" label="创建时间" width="220" />
          <el-table-column label="操作" width="130">
            <template #default="scope">
              <el-button
                size="small"
                type="primary"
                link
                @click="$router.push(`/reports/${scope.row.runId}`)"
              >进入</el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>

      <el-tab-pane name="published" label="定稿 (outputs/reports)">
        <el-table :data="publishedFiles" size="small" stripe>
          <el-table-column prop="relPath" label="文件" />
          <el-table-column
            prop="sizeBytes"
            label="大小"
            :formatter="(row: any) => formatSize(row.sizeBytes)"
            width="100"
          />
          <el-table-column prop="modifiedAt" label="修改时间" width="220" />
          <el-table-column label="操作" width="160">
            <template #default="scope">
              <el-button
                size="small"
                type="primary"
                link
                @click="open(publishedUrl(scope.row.relPath))"
              >打开</el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>
    </el-tabs>
  </el-card>
</template>

<script setup lang="ts">
import { onMounted, ref } from "vue";
import { reportsApi } from "@/api/reports";
import type { ReportRunInfo } from "@/types/api";

const tab = ref("runs");
const runs = ref<ReportRunInfo[]>([]);
const publishedFiles = ref<{ relPath: string; sizeBytes: number; modifiedAt: string }[]>([]);
const loading = ref(false);

async function refresh(): Promise<void> {
  loading.value = true;
  try {
    runs.value = await reportsApi.runs();
    const pub = await reportsApi.publishedTree();
    publishedFiles.value = pub.files;
  } finally {
    loading.value = false;
  }
}

function formatSize(b: number): string {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / (1024 * 1024)).toFixed(1)} MB`;
}

function publishedUrl(path: string): string {
  return reportsApi.publishedFileUrl(path);
}

function open(url: string): void {
  window.open(url, "_blank");
}

onMounted(refresh);
</script>

<style scoped>
.card-title {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
</style>
