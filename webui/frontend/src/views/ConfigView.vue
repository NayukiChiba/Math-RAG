<template>
  <div class="config-view">
    <el-card shadow="never" style="margin-bottom: 16px">
      <template #header>
        <div class="page-title">
          <span>模型引擎配置</span>
          <el-button size="small" @click="load" :loading="loading">
            重新加载
          </el-button>
        </div>
      </template>
      <el-alert
        type="warning"
        show-icon
        :closable="false"
        title="修改保存后立即写回 config.toml。切换引擎会自动重置 RAG 单例。"
      />
    </el-card>

    <ModelEngineCard
      title="OCR · PDF 转 Markdown"
      description="负责把上传的 PDF 每页识别为 Markdown + LaTeX 公式。"
      :values="ocrValues"
      :local-fields="OCR_LOCAL_FIELDS"
      :api-fields="OCR_API_FIELDS"
      @save="onSaveOcr"
    />

    <ModelEngineCard
      title="术语结构化生成"
      description="把 OCR 上下文转换为结构化术语条目 JSON。"
      :values="termsValues"
      :local-fields="TERMS_LOCAL_FIELDS"
      :api-fields="TERMS_API_FIELDS"
      @save="onSaveTerms"
    />

    <ModelEngineCard
      title="RAG 回答生成"
      description="检索后由该引擎生成最终答案，支持流式输出。"
      :values="ragValues"
      :local-fields="RAG_LOCAL_FIELDS"
      :api-fields="RAG_API_FIELDS"
      @save="onSaveRag"
    />

    <el-card shadow="never" class="simple-card">
      <template #header>
        <div class="card-title">
          <div class="card-title-main">
            <span class="title">检索 · Embedding 向量</span>
            <el-tag type="info" size="small">本地</el-tag>
          </div>
          <div class="card-title-actions">
            <el-button
              size="small"
              type="primary"
              :disabled="embeddingModel === embeddingInitial"
              @click="onSaveEmbedding"
            >
              保存
            </el-button>
          </div>
        </div>
      </template>
      <el-alert
        type="info"
        :closable="false"
        show-icon
        title="文档与查询向量化。使用本地 SentenceTransformer 加载 HuggingFace 模型，首次会自动下载。"
        style="margin-bottom: 12px"
      />
      <el-form label-width="160px">
        <el-form-item label="模型">
          <el-input
            v-model="embeddingModel"
            placeholder="BAAI/bge-base-zh-v1.5"
          />
          <div class="field-hint">
            HuggingFace 模型名（例：BAAI/bge-base-zh-v1.5）或本地模型目录。
          </div>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card shadow="never" class="simple-card">
      <template #header>
        <div class="card-title">
          <div class="card-title-main">
            <span class="title">检索 · Reranker 重排</span>
            <el-tag type="info" size="small">本地</el-tag>
          </div>
          <div class="card-title-actions">
            <el-button
              size="small"
              type="primary"
              :disabled="rerankerModel === rerankerInitial"
              @click="onSaveReranker"
            >
              保存
            </el-button>
          </div>
        </div>
      </template>
      <el-alert
        type="info"
        :closable="false"
        show-icon
        title="对召回结果二次排序。使用本地 SentenceTransformer 加载 HuggingFace 模型，首次会自动下载。"
        style="margin-bottom: 12px"
      />
      <el-form label-width="160px">
        <el-form-item label="模型">
          <el-input
            v-model="rerankerModel"
            placeholder="BAAI/bge-reranker-v2-mixed"
          />
          <div class="field-hint">
            HuggingFace 模型名（例：BAAI/bge-reranker-v2-mixed）或本地模型目录。
          </div>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { onMounted, reactive, ref } from "vue";
import { ElMessage } from "element-plus";
import ModelEngineCard, {
  type FieldDef,
} from "@/components/ModelEngineCard.vue";
import { configApi } from "@/api/config";

// ── 字段定义 ──────────────────────────────────────────────────────

const OCR_LOCAL_FIELDS: FieldDef[] = [
  { key: "device", label: "device", type: "text", hint: "cuda / cpu" },
  { key: "render_dpi", label: "render_dpi", type: "number", min: 72 },
  { key: "resized_shape", label: "resized_shape", type: "number", min: 64 },
  { key: "mfr_batch_size", label: "mfr_batch_size", type: "number", min: 1 },
  { key: "text_contain_formula", label: "text_contain_formula", type: "switch" },
  { key: "batch_pages", label: "batch_pages", type: "number", min: 0 },
  { key: "ocr_workers", label: "ocr_workers", type: "number", min: 0 },
  { key: "page_start", label: "page_start", type: "number", min: 0 },
  {
    key: "page_end",
    label: "page_end",
    type: "number",
    min: -1,
    hint: "-1 表示不限制",
  },
  { key: "skip_existing", label: "skip_existing", type: "switch" },
];

const OCR_API_FIELDS: FieldDef[] = [
  { key: "api_base", label: "api_base", type: "text" },
  { key: "model", label: "model", type: "text" },
  {
    key: "api_key_env",
    label: "api_key_env",
    type: "text",
    hint: "从环境变量 / .env 中读取对应名称的密钥",
  },
  { key: "max_tokens", label: "max_tokens", type: "number", min: 1 },
  {
    key: "temperature",
    label: "temperature",
    type: "number",
    step: 0.1,
    min: 0,
    max: 2,
  },
  { key: "prompt", label: "prompt", type: "textarea", rows: 6 },
];

const TERMS_LOCAL_FIELDS: FieldDef[] = [
  {
    key: "local_model_dir",
    label: "local_model_dir",
    type: "text",
    hint: "本地 HuggingFace 模型目录，支持相对项目根或绝对路径",
    placeholder: "../Qwen3.5-4B",
  },
  { key: "max_tokens", label: "max_tokens", type: "number", min: 1 },
  {
    key: "temperature",
    label: "temperature",
    type: "number",
    step: 0.1,
    min: 0,
    max: 2,
  },
  { key: "top_p", label: "top_p", type: "number", step: 0.05, min: 0, max: 1 },
  { key: "max_attempts", label: "max_attempts", type: "number", min: 1 },
  { key: "request_timeout", label: "request_timeout", type: "number", min: 1 },
];

const TERMS_API_FIELDS: FieldDef[] = [
  { key: "api_base", label: "api_base", type: "text" },
  { key: "model", label: "model", type: "text" },
  { key: "api_key_env", label: "api_key_env", type: "text" },
  { key: "max_tokens", label: "max_tokens", type: "number", min: 1 },
  {
    key: "temperature",
    label: "temperature",
    type: "number",
    step: 0.1,
    min: 0,
    max: 2,
  },
  { key: "top_p", label: "top_p", type: "number", step: 0.05, min: 0, max: 1 },
  { key: "stream", label: "stream", type: "switch" },
  { key: "request_timeout", label: "request_timeout", type: "number", min: 1 },
];

const RAG_LOCAL_FIELDS: FieldDef[] = [
  {
    key: "local_model_dir",
    label: "local_model_dir",
    type: "text",
    hint: "本地 HuggingFace 模型目录",
    placeholder: "../Qwen3.5-4B",
  },
  {
    key: "temperature",
    label: "temperature",
    type: "number",
    step: 0.1,
    min: 0,
    max: 2,
  },
  { key: "top_p", label: "top_p", type: "number", step: 0.05, min: 0, max: 1 },
  { key: "max_new_tokens", label: "max_new_tokens", type: "number", min: 1 },
  { key: "max_context_chars", label: "max_context_chars", type: "number", min: 1 },
  {
    key: "max_chars_per_term",
    label: "max_chars_per_term",
    type: "number",
    min: 1,
  },
];

const RAG_API_FIELDS: FieldDef[] = [
  { key: "api_base", label: "api_base", type: "text" },
  { key: "api_model", label: "api_model", type: "text" },
  { key: "api_key_env", label: "api_key_env", type: "text" },
  { key: "api_stream", label: "api_stream", type: "switch" },
  {
    key: "temperature",
    label: "temperature",
    type: "number",
    step: 0.1,
    min: 0,
    max: 2,
  },
  { key: "top_p", label: "top_p", type: "number", step: 0.05, min: 0, max: 1 },
  { key: "max_new_tokens", label: "max_new_tokens", type: "number", min: 1 },
  { key: "max_context_chars", label: "max_context_chars", type: "number", min: 1 },
];

// ── 状态 ──────────────────────────────────────────────────────────

const loading = ref(false);

const ocrValues = reactive<Record<string, unknown>>({ engine: "api" });
const termsValues = reactive<Record<string, unknown>>({ engine: "api" });
const ragValues = reactive<Record<string, unknown>>({ engine: "api" });

const embeddingModel = ref("");
const embeddingInitial = ref("");
const rerankerModel = ref("");
const rerankerInitial = ref("");

type SectionValues = Record<string, unknown>;

function writeReactive(target: Record<string, unknown>, next: SectionValues): void {
  for (const key of Object.keys(target)) {
    delete target[key];
  }
  Object.assign(target, next);
}

async function load(): Promise<void> {
  loading.value = true;
  try {
    const data = (await configApi.get()) as Record<string, SectionValues>;

    // OCR 需要把 [ocr] 和 [ocr.api] 合并（engine + local 字段 + prompt/api_* 字段）
    const ocrLocal = (data.ocr ?? {}) as SectionValues;
    const ocrApiSec = (ocrLocal.api ?? {}) as SectionValues;
    const ocrMerged: SectionValues = {
      engine: ocrLocal.engine ?? "api",
      device: ocrLocal.device ?? "cuda",
      render_dpi: ocrLocal.render_dpi ?? 200,
      resized_shape: ocrLocal.resized_shape ?? 512,
      mfr_batch_size: ocrLocal.mfr_batch_size ?? 8,
      text_contain_formula: ocrLocal.text_contain_formula ?? true,
      batch_pages: ocrLocal.batch_pages ?? 10,
      ocr_workers: ocrLocal.ocr_workers ?? 0,
      page_start: ocrLocal.page_start ?? 0,
      page_end: ocrLocal.page_end ?? -1,
      skip_existing: ocrLocal.skip_existing ?? true,
      api_base: ocrApiSec.api_base ?? "",
      model: ocrApiSec.model ?? "",
      api_key_env: ocrApiSec.api_key_env ?? "API-KEY-OCR",
      max_tokens: ocrApiSec.max_tokens ?? 4000,
      temperature: ocrApiSec.temperature ?? 0.1,
      prompt: ocrApiSec.prompt ?? "",
    };
    writeReactive(ocrValues, ocrMerged);

    writeReactive(termsValues, (data.terms_gen ?? {}) as SectionValues);
    writeReactive(ragValues, (data.rag_gen ?? {}) as SectionValues);

    const retrieval = (data.retrieval ?? {}) as SectionValues;
    const embModel = String(
      retrieval.default_vector_model ?? "BAAI/bge-base-zh-v1.5",
    );
    const rerankModel = String(
      retrieval.default_reranker_model ?? "BAAI/bge-reranker-v2-mixed",
    );
    embeddingModel.value = embModel;
    embeddingInitial.value = embModel;
    rerankerModel.value = rerankModel;
    rerankerInitial.value = rerankModel;
  } catch (e) {
    ElMessage.error((e as Error).message);
  } finally {
    loading.value = false;
  }
}

// ── 保存回调 ─────────────────────────────────────────────────────

function pickKeys(
  src: Record<string, unknown>,
  keys: string[],
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const k of keys) {
    if (k in src) out[k] = src[k];
  }
  return out;
}

async function onSaveOcr(payload: Record<string, unknown>): Promise<void> {
  // 拆两次 PATCH：本地字段 + engine → [ocr]；API 字段 → [ocr.api]
  const localUpdates = {
    engine: payload.engine,
    ...pickKeys(
      payload,
      OCR_LOCAL_FIELDS.map((f) => f.key),
    ),
  };
  const apiUpdates = pickKeys(
    payload,
    OCR_API_FIELDS.map((f) => f.key),
  );
  try {
    await configApi.patch("ocr", localUpdates);
    await configApi.patch("ocr.api", apiUpdates);
    ElMessage.success("OCR 配置已保存");
    await load();
  } catch (e) {
    ElMessage.error((e as Error).message);
  }
}

async function saveSection(
  section: string,
  payload: Record<string, unknown>,
  hint: string,
): Promise<void> {
  try {
    await configApi.patch(section, payload);
    ElMessage.success(`${hint} 已保存`);
    await load();
  } catch (e) {
    ElMessage.error((e as Error).message);
  }
}

async function onSaveTerms(payload: Record<string, unknown>): Promise<void> {
  await saveSection("terms_gen", payload, "术语生成");
}

async function onSaveRag(payload: Record<string, unknown>): Promise<void> {
  await saveSection("rag_gen", payload, "RAG 回答");
}

async function onSaveEmbedding(): Promise<void> {
  await saveSection(
    "retrieval",
    { default_vector_model: embeddingModel.value },
    "Embedding 模型",
  );
}

async function onSaveReranker(): Promise<void> {
  await saveSection(
    "retrieval",
    { default_reranker_model: rerankerModel.value },
    "Reranker 模型",
  );
}

onMounted(async () => {
  await load();
});
</script>

<style scoped>
.config-view {
  max-width: 960px;
  margin: 0 auto;
}

.page-title {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: 600;
  font-size: 16px;
}

.simple-card {
  margin-bottom: 16px;
}

.card-title {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card-title-main {
  display: flex;
  align-items: center;
  gap: 10px;
}

.card-title-main .title {
  font-weight: 600;
  font-size: 15px;
}

.card-title-actions {
  display: flex;
  gap: 8px;
}

.field-hint {
  color: #909399;
  font-size: 12px;
  margin-top: 2px;
  line-height: 1.4;
}
</style>
