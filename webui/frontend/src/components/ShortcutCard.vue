<template>
  <el-card class="shortcut" shadow="hover" @click="go">
    <div class="shortcut-head">
      <el-icon :size="20">
        <component :is="iconComp" />
      </el-icon>
      <span class="title">{{ title }}</span>
    </div>
    <div class="desc">{{ desc }}</div>
  </el-card>
</template>

<script setup lang="ts">
import { computed } from "vue";
import { useRouter } from "vue-router";
import * as icons from "@element-plus/icons-vue";

const props = defineProps<{
  title: string;
  desc: string;
  icon: string;
  to: string;
}>();

const router = useRouter();

const iconComp = computed(
  () => (icons as unknown as Record<string, any>)[props.icon],
);

function go(): void {
  router.push(props.to);
}
</script>

<style scoped>
.shortcut {
  cursor: pointer;
  transition: transform 0.2s;
}

.shortcut:hover {
  transform: translateY(-2px);
}

.shortcut-head {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
  color: #409eff;
}

.shortcut-head .title {
  font-size: 15px;
  font-weight: 600;
  color: #303133;
}

.desc {
  color: #909399;
  font-size: 13px;
}
</style>
