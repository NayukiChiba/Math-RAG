<template>
  <el-card shadow="never" style="margin-bottom: 16px">
    <template #header>
      <div class="card-title">
        <span>引擎快捷切换</span>
        <el-button size="small" @click="loadEngines" :loading="enginesLoading">
          刷新
        </el-button>
      </div>
    </template>
    <el-alert
      type="info"
      :closable="false"
      show-icon
      title="三处使用 LLM 的环节可独立切换本地 / API。修改后立即生效（RAG 单例会被自动重置）。"
      style="margin-bottom: 12px"
    />

    <el-descriptions :column="1" border size="default">
      <el-descriptions-item label="OCR（PDF → Markdown）">
        <el-radio-group v-model="engines.ocr" @change="saveEngines">
          <el-radio-button value="local">本地 (pix2text)</el-radio-button>
          <el-radio-button value="api">API (多模态 Vision)</el-radio-button>
        </el-radio-group>
      </el-descriptions-item>
      <el-descriptions-item label="数据处理（术语结构化生成）">
        <el-radio-group v-model="engines.dataGen" disabled>
          <el-radio-button value="local">本地</el-radio-button>
          <el-radio-button value="api">API</el-radio-button>
        </el-radio-group>
        <span class="hint-inline">当前仅支持 API 引擎</span>
      </el-descriptions-item>
      <el-descriptions-item label="RAG 回答（问答生成）">
        <el-radio-group v-model="engines.rag" @change="saveEngines">
          <el-radio-button value="local">本地 (HuggingFace)</el-radio-button>
          <el-radio-button value="api">API (OpenAI 兼容)</el-radio-button>
        </el-radio-group>
      </el-descriptions-item>
    </el-descriptions>
  </el-card>

  <el-card shadow="never">
    <template #header>
      <div class="card-title">
        <span>config.toml 分段编辑</span>
        <el-button size="small" @click="load" :loading="loading">重新加载</el-button>
      </div>
    </template>

    <el-alert
      type="warning"
      show-icon
      :closable="false"
      title="修改配置会立即写回 config.toml 文件；保留注释与顺序。仅允许修改已存在的 key 类型。"
      style="margin-bottom: 12px"
    />

    <el-tabs v-model="activeSection" type="card">
      <el-tab-pane
        v-for="section in visibleSections"
        :key="section"
        :name="section"
        :label="section"
      >
        <el-form label-width="220px">
          <el-form-item
            v-for="(value, key) in (data[section] as Record<string, unknown>)"
            :key="key"
            :label="key"
          >
            <el-input
              v-if="typeof value === 'string' && (value as string).length > 60"
              v-model="editable[section][key]"
              type="textarea"
              :rows="4"
            />
            <el-input-number
              v-else-if="typeof value === 'number'"
              v-model="editable[section][key]"
              controls-position="right"
            />
            <el-switch
              v-else-if="typeof value === 'boolean'"
              v-model="editable[section][key]"
            />
            <el-input
              v-else-if="typeof value === 'string'"
              v-model="editable[section][key]"
            />
            <el-input
              v-else
              :model-value="JSON.stringify(editable[section][key])"
              disabled
            />
          </el-form-item>
        </el-form>

        <div style="text-align: right">
          <el-button type="primary" @click="save(section)">保存 {{ section }}</el-button>
        </div>
      </el-tab-pane>
    </el-tabs>
  </el-card>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref } from "vue";
import { ElMessage } from "element-plus";
import { configApi, type EnginesState } from "@/api/config";

const data = ref<Record<string, any>>({});
const editable = reactive<Record<string, any>>({});
const loading = ref(false);
const activeSection = ref("generation");

const engines = reactive<EnginesState>({
  ocr: "local",
  dataGen: "api",
  rag: "api",
});
const enginesLoading = ref(false);

const visibleSections = computed(() =>
  Object.keys(data.value).filter((k) => {
    const v = data.value[k];
    return v && typeof v === "object" && !Array.isArray(v);
  }),
);

async function loadEngines(): Promise<void> {
  enginesLoading.value = true;
  try {
    const state = await configApi.getEngines();
    engines.ocr = state.ocr;
    engines.dataGen = state.dataGen;
    engines.rag = state.rag;
  } catch (e) {
    ElMessage.error((e as Error).message);
  } finally {
    enginesLoading.value = false;
  }
}

async function saveEngines(): Promise<void> {
  try {
    const state = await configApi.patchEngines({
      ocr: engines.ocr,
      rag: engines.rag,
    });
    engines.ocr = state.ocr;
    engines.rag = state.rag;
    ElMessage.success("引擎已切换");
    await load();
  } catch (e) {
    ElMessage.error((e as Error).message);
  }
}

async function load(): Promise<void> {
  loading.value = true;
  try {
    data.value = await configApi.get();
    for (const section of Object.keys(data.value)) {
      const sectionData = data.value[section];
      if (
        sectionData &&
        typeof sectionData === "object" &&
        !Array.isArray(sectionData)
      ) {
        editable[section] = reactive({ ...sectionData });
      }
    }
  } finally {
    loading.value = false;
  }
}

async function save(section: string): Promise<void> {
  try {
    const updates: Record<string, unknown> = {};
    const current = data.value[section] as Record<string, unknown>;
    for (const key of Object.keys(current)) {
      const original = current[key];
      const edited = editable[section][key];
      if (isPrimitive(original) && original !== edited) {
        updates[key] = edited;
      }
    }
    if (Object.keys(updates).length === 0) {
      ElMessage.info("没有修改可保存");
      return;
    }
    await configApi.patch(section, updates);
    ElMessage.success(`已保存 ${Object.keys(updates).length} 项修改`);
    await load();
  } catch (e) {
    ElMessage.error((e as Error).message);
  }
}

function isPrimitive(v: unknown): boolean {
  return (
    typeof v === "string" ||
    typeof v === "number" ||
    typeof v === "boolean" ||
    Array.isArray(v)
  );
}

onMounted(async () => {
  await Promise.all([load(), loadEngines()]);
});
</script>

<style scoped>
.card-title {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.hint-inline {
  margin-left: 12px;
  color: #909399;
  font-size: 12px;
}
</style>
