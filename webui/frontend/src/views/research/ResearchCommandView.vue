<template>
  <div>
    <el-page-header @back="$router.push('/research')" :content="`research ${command}`" />

    <el-card shadow="never" style="margin-top: 12px">
      <template #header>
        <div>参数（等价于 python main.py research {{ command }} [参数]）</div>
      </template>

      <div v-if="preset" class="preset">
        <div class="preset-title">常用参数：</div>
        <el-space wrap>
          <el-button
            v-for="(p, i) in preset"
            :key="i"
            size="small"
            @click="applyPreset(p)"
          >{{ p.label }}</el-button>
        </el-space>
      </div>

      <div class="args-editor">
        <div
          v-for="(item, idx) in argList"
          :key="idx"
          class="arg-row"
        >
          <el-input
            v-model="item.value"
            placeholder="--flag 或 值"
            :disabled="!!currentTaskId"
          />
          <el-button
            type="danger"
            link
            :disabled="!!currentTaskId"
            @click="remove(idx)"
          >删除</el-button>
        </div>
        <el-button :disabled="!!currentTaskId" @click="add">+ 添加参数</el-button>
      </div>

      <el-divider />

      <div class="actions">
        <el-input
          v-model="rawCmd"
          placeholder="或直接粘贴命令行参数字符串，例如 --limit 10 --visualize"
          :disabled="!!currentTaskId"
          @keydown.enter="start"
        >
          <template #append>
            <el-button @click="start" :disabled="!!currentTaskId" type="primary">
              启动
            </el-button>
          </template>
        </el-input>
      </div>
    </el-card>

    <TaskPanel
      v-if="currentTaskId"
      :task-id="currentTaskId"
      :command="`research.${command}`"
      @done="onDone"
    />
  </div>
</template>

<script setup lang="ts">
import { computed, reactive, ref, watch } from "vue";
import { useRoute } from "vue-router";
import { ElMessage } from "element-plus";
import TaskPanel from "@/components/TaskPanel.vue";
import { researchApi } from "@/api/research";
import type { TaskInfo } from "@/types/api";

const route = useRoute();
const command = computed(() => route.params.command as string);

const argList = reactive<{ value: string }[]>([]);
const rawCmd = ref("");
const currentTaskId = ref<string | null>(null);

interface Preset {
  label: string;
  args: string[];
}

const PRESETS: Record<string, Preset[]> = {
  experiments: [
    { label: "全部实验组", args: ["--groups", "norag", "bm25", "vector", "hybrid"] },
    { label: "限制 5 条（调试）", args: ["--limit", "5"] },
    { label: "包含 hybrid-rrf", args: ["--groups", "norag", "bm25", "vector", "hybrid", "hybrid-rrf"] },
  ],
  "eval-retrieval": [
    { label: "生成图表", args: ["--visualize"] },
  ],
  "full-reports": [
    { label: "仅检索", args: ["--retrieval-only"] },
    { label: "遇错继续", args: ["--continue-on-error"] },
  ],
  "quick-eval": [
    { label: "基础方法", args: ["--mode", "basic"] },
    { label: "全部方法", args: ["--mode", "all"] },
  ],
  "publish-reports": [
    { label: "指定 run-id", args: ["--run-id", ""] },
  ],
};

const preset = computed(() => PRESETS[command.value] || null);

function add(): void {
  argList.push({ value: "" });
}

function remove(i: number): void {
  argList.splice(i, 1);
}

function applyPreset(p: Preset): void {
  argList.splice(0, argList.length);
  for (const a of p.args) argList.push({ value: a });
}

async function start(): Promise<void> {
  let args: string[];
  if (rawCmd.value.trim()) {
    args = rawCmd.value.trim().split(/\s+/);
  } else {
    args = argList.map((x) => x.value).filter((v) => v !== "");
  }

  try {
    const res = await researchApi.run(command.value, args);
    currentTaskId.value = res.taskId;
    ElMessage.success("任务已提交");
  } catch (e) {
    ElMessage.error((e as Error).message);
  }
}

function onDone(s: TaskInfo["status"]): void {
  if (s === "succeeded") ElMessage.success("命令执行完成");
  else if (s === "failed") ElMessage.error("命令执行失败");
}

watch(
  command,
  () => {
    argList.splice(0, argList.length);
    rawCmd.value = "";
    currentTaskId.value = null;
  },
  { immediate: true },
);
</script>

<style scoped>
.preset {
  margin-bottom: 12px;
}

.preset-title {
  color: #909399;
  margin-bottom: 8px;
}

.args-editor {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.arg-row {
  display: flex;
  gap: 8px;
}

.actions {
  margin-top: 12px;
}
</style>
