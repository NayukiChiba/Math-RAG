<template>
  <el-card shadow="never">
    <template #header>
      <div class="card-title">
        <span>图表库（outputs/figures）</span>
        <el-button size="small" @click="refresh" :loading="loading">刷新</el-button>
      </div>
    </template>

    <el-empty v-if="figures.length === 0" description="暂无图表" />
    <el-row v-else :gutter="12">
      <el-col :span="6" v-for="fig in figures" :key="fig.relPath">
        <el-card shadow="hover" class="fig-card" @click="preview(fig.relPath)">
          <img :src="url(fig.relPath)" class="thumb" />
          <div class="fig-name">{{ fig.relPath }}</div>
        </el-card>
      </el-col>
    </el-row>

    <el-dialog v-model="previewVisible" width="80%">
      <img v-if="previewPath" :src="url(previewPath)" style="max-width: 100%" />
    </el-dialog>
  </el-card>
</template>

<script setup lang="ts">
import { onMounted, ref } from "vue";
import { reportsApi } from "@/api/reports";
import type { FigureInfo } from "@/types/api";

const figures = ref<FigureInfo[]>([]);
const loading = ref(false);
const previewVisible = ref(false);
const previewPath = ref<string | null>(null);

async function refresh(): Promise<void> {
  loading.value = true;
  try {
    figures.value = await reportsApi.figures();
  } finally {
    loading.value = false;
  }
}

function url(path: string): string {
  return reportsApi.figureUrl(path);
}

function preview(path: string): void {
  previewPath.value = path;
  previewVisible.value = true;
}

onMounted(refresh);
</script>

<style scoped>
.card-title {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.fig-card {
  margin-bottom: 12px;
  cursor: pointer;
}

.thumb {
  width: 100%;
  aspect-ratio: 16 / 10;
  object-fit: contain;
  background-color: #fafafa;
}

.fig-name {
  margin-top: 6px;
  font-size: 12px;
  color: #606266;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
