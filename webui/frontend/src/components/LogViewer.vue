<template>
  <div class="log-box" ref="boxRef">
    <div
      v-for="(line, idx) in lines"
      :key="idx"
      :class="['code-line', line.stream === 'stderr' ? 'stream-stderr' : 'stream-stdout']"
    >{{ line.text }}</div>
    <div v-if="lines.length === 0" class="empty">暂无日志</div>
  </div>
</template>

<script setup lang="ts">
import { nextTick, ref, watch } from "vue";

export interface LogLine {
  stream: "stdout" | "stderr";
  text: string;
}

const props = defineProps<{ lines: LogLine[]; autoScroll?: boolean }>();

const boxRef = ref<HTMLDivElement | null>(null);

watch(
  () => props.lines.length,
  async () => {
    if (props.autoScroll === false) return;
    await nextTick();
    if (boxRef.value) {
      boxRef.value.scrollTop = boxRef.value.scrollHeight;
    }
  },
);
</script>

<style scoped>
.log-box {
  background-color: #1e1e1e;
  color: #d4d4d4;
  padding: 12px;
  border-radius: 6px;
  height: 360px;
  overflow-y: auto;
}

.empty {
  color: #888;
  font-style: italic;
  padding: 12px;
}

.stream-stdout {
  color: #d4d4d4;
}

.stream-stderr {
  color: #ff8a80;
}
</style>
