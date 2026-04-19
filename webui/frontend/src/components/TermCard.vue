<template>
  <el-card class="term-card" shadow="hover">
    <div class="term-head">
      <div class="rank">#{{ item.rank ?? "-" }}</div>
      <div class="term">{{ item.term || "（未知术语）" }}</div>
      <el-tag v-if="item.subject" size="small">{{ item.subject }}</el-tag>
      <el-tag type="info" size="small">得分 {{ item.score.toFixed(3) }}</el-tag>
    </div>
    <div class="term-source" v-if="item.source">
      来源：{{ shortSource }}
      <span v-if="item.page"> · 第 {{ item.page }} 页</span>
    </div>
    <div class="term-text" v-if="item.text">
      <MarkdownView :source="item.text" />
    </div>
  </el-card>
</template>

<script setup lang="ts">
import { computed } from "vue";
import MarkdownView from "./MarkdownView.vue";
import type { RagRetrievalItem } from "@/types/api";

const props = defineProps<{ item: RagRetrievalItem }>();

const shortSource = computed(() => {
  const src = props.item.source || "";
  return src.split("(")[0].split("（")[0].trim() || src;
});
</script>

<style scoped>
.term-card {
  margin-bottom: 12px;
}

.term-head {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}

.rank {
  color: #909399;
  font-weight: 600;
  min-width: 32px;
}

.term {
  font-size: 15px;
  font-weight: 600;
  color: #303133;
}

.term-source {
  color: #909399;
  font-size: 12px;
  margin-bottom: 8px;
}

.term-text {
  border-top: 1px dashed #ebeef5;
  padding-top: 8px;
  font-size: 13px;
  color: #606266;
  max-height: 220px;
  overflow-y: auto;
}
</style>
