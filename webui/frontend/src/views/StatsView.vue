<template>
  <el-card shadow="never">
    <template #header>
      <div class="card-title">
        <span>术语统计</span>
        <el-button size="small" @click="refresh" :loading="loading">刷新</el-button>
      </div>
    </template>

    <el-alert
      v-if="data && !data.available"
      type="warning"
      :closable="false"
      show-icon
      :title="`未检测到统计输出目录：${data.statsDir}`"
      description="请先在 /research 页面运行 stats 命令生成统计数据。"
    />

    <template v-if="data?.available">
      <el-collapse>
        <el-collapse-item
          v-for="(content, name) in data.files"
          :key="name"
          :title="name"
          :name="name"
        >
          <pre class="json-pre">{{ JSON.stringify(content, null, 2) }}</pre>
        </el-collapse-item>
      </el-collapse>
    </template>

    <el-divider />

    <div v-if="figures.length > 0">
      <h3>可视化图表</h3>
      <el-row :gutter="12">
        <el-col :span="8" v-for="fig in figures" :key="fig.relPath">
          <el-card shadow="hover">
            <img :src="figUrl(fig.relPath)" style="width: 100%" />
            <div class="fig-caption">{{ fig.filename }}</div>
          </el-card>
        </el-col>
      </el-row>
    </div>
  </el-card>
</template>

<script setup lang="ts">
import { onMounted, ref } from "vue";
import { statsApi } from "@/api/stats";
import { reportsApi } from "@/api/reports";

interface StatsResponse {
  available: boolean;
  statsDir: string;
  files?: Record<string, unknown>;
}

const data = ref<StatsResponse | null>(null);
const figures = ref<{ label: string; filename: string; relPath: string }[]>([]);
const loading = ref(false);

async function refresh(): Promise<void> {
  loading.value = true;
  try {
    data.value = await statsApi.get();
    figures.value = await statsApi.figures();
  } finally {
    loading.value = false;
  }
}

function figUrl(relPath: string): string {
  return reportsApi.figureUrl(relPath);
}

onMounted(refresh);
</script>

<style scoped>
.card-title {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.json-pre {
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 12px;
  background-color: #f5f7fa;
  padding: 12px;
  border-radius: 4px;
  max-height: 400px;
  overflow: auto;
}

.fig-caption {
  margin-top: 6px;
  text-align: center;
  color: #606266;
  font-size: 12px;
}
</style>
