<template>
  <el-row :gutter="16">
    <el-col :span="14">
      <el-card shadow="never">
        <template #header>
          <div class="card-title">
            <el-icon><Upload /></el-icon>
            <span>上传 PDF</span>
          </div>
        </template>
        <el-upload
          drag
          :auto-upload="true"
          :http-request="uploadPdf"
          :show-file-list="false"
          accept=".pdf"
        >
          <el-icon class="el-icon--upload"><upload-filled /></el-icon>
          <div class="el-upload__text">
            拖拽 PDF 到此处，或 <em>点击选择文件</em>
          </div>
          <template #tip>
            <div class="el-upload__tip">
              上传后文件保存到 <code>data/raw/</code>，再在右侧启动入库流水线。
            </div>
          </template>
        </el-upload>

        <el-divider>raw 目录下已有 PDF</el-divider>
        <el-table :data="rawPdfs" size="small" max-height="260">
          <el-table-column prop="name" label="文件名" />
          <el-table-column
            prop="sizeBytes"
            label="大小"
            width="100"
            :formatter="(row: any) => formatSize(row.sizeBytes)"
          />
          <el-table-column label="操作" width="120">
            <template #default="scope">
              <el-button
                size="small"
                type="primary"
                plain
                @click="pickPdf(scope.row.name)"
              >选择</el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-card>
    </el-col>

    <el-col :span="10">
      <el-card shadow="never">
        <template #header>
          <div class="card-title">
            <el-icon><Operation /></el-icon>
            <span>入库参数</span>
          </div>
        </template>
        <el-form label-width="140px" :disabled="!!currentTaskId">
          <el-form-item label="PDF 文件">
            <el-input v-model="form.pdf" placeholder="文件名或绝对路径" />
          </el-form-item>
          <el-form-item label="OCR 起始页">
            <el-input-number
              v-model="form.ocrStartPage"
              :min="1"
              controls-position="right"
            />
          </el-form-item>
          <el-form-item label="抽取起始页">
            <el-input-number
              v-model="form.extractStartPage"
              :min="1"
              controls-position="right"
            />
          </el-form-item>
          <el-form-item label="生成起始页">
            <el-input-number
              v-model="form.generateStartPage"
              :min="1"
              controls-position="right"
            />
          </el-form-item>
          <el-form-item label="跳过">
            <el-checkbox v-model="form.skipGeneration">跳过结构化生成</el-checkbox>
            <el-checkbox v-model="form.skipIndex">跳过索引构建</el-checkbox>
          </el-form-item>
          <el-form-item label="索引选项">
            <el-checkbox v-model="form.rebuildIndex">强制重建</el-checkbox>
            <el-checkbox v-model="form.skipBm25">跳过 BM25</el-checkbox>
            <el-checkbox v-model="form.skipBm25plus">跳过 BM25+</el-checkbox>
            <el-checkbox v-model="form.skipVector">跳过向量</el-checkbox>
          </el-form-item>
          <el-form-item label="向量模型">
            <el-input v-model="form.vectorModel" placeholder="默认使用 config 中的模型" />
          </el-form-item>
          <el-form-item label="批次大小">
            <el-input-number
              v-model="form.batchSize"
              :min="1"
              :max="128"
              controls-position="right"
            />
          </el-form-item>
          <el-form-item>
            <el-button type="primary" :disabled="!form.pdf" @click="start">
              启动入库任务
            </el-button>
          </el-form-item>
        </el-form>
      </el-card>
    </el-col>
  </el-row>

  <TaskPanel
    v-if="currentTaskId"
    :task-id="currentTaskId"
    command="cli.ingest"
    @done="onDone"
  />
</template>

<script setup lang="ts">
import { onMounted, reactive, ref } from "vue";
import { ElMessage, type UploadRequestOptions } from "element-plus";
import { Operation, Upload, UploadFilled } from "@element-plus/icons-vue";
import TaskPanel from "@/components/TaskPanel.vue";
import { dataApi } from "@/api/data";
import { ingestApi } from "@/api/ingest";
import type { RawPdfInfo, TaskInfo } from "@/types/api";

const rawPdfs = ref<RawPdfInfo[]>([]);
const currentTaskId = ref<string | null>(null);

const form = reactive({
  pdf: "",
  ocrStartPage: undefined as number | undefined,
  extractStartPage: undefined as number | undefined,
  generateStartPage: undefined as number | undefined,
  skipGeneration: false,
  skipIndex: false,
  rebuildIndex: false,
  skipBm25: false,
  skipBm25plus: false,
  skipVector: false,
  vectorModel: "",
  batchSize: 32,
});

async function refreshRaw(): Promise<void> {
  rawPdfs.value = await dataApi.raw();
}

async function uploadPdf(options: UploadRequestOptions): Promise<unknown> {
  try {
    const res = await ingestApi.upload(options.file as File);
    ElMessage.success(`已上传：${res.name}`);
    form.pdf = res.name;
    await refreshRaw();
    return res;
  } catch (e) {
    ElMessage.error((e as Error).message);
    throw e;
  }
}

function pickPdf(name: string): void {
  form.pdf = name;
}

function formatSize(b: number): string {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / (1024 * 1024)).toFixed(1)} MB`;
}

async function start(): Promise<void> {
  try {
    const payload = {
      pdf: form.pdf,
      ocrStartPage: form.ocrStartPage,
      extractStartPage: form.extractStartPage,
      generateStartPage: form.generateStartPage,
      skipGeneration: form.skipGeneration,
      skipIndex: form.skipIndex,
      rebuildIndex: form.rebuildIndex,
      skipBm25: form.skipBm25,
      skipBm25plus: form.skipBm25plus,
      skipVector: form.skipVector,
      vectorModel: form.vectorModel || undefined,
      batchSize: form.batchSize,
    };
    const res = await ingestApi.start(payload);
    currentTaskId.value = res.taskId;
    ElMessage.success("任务已提交");
  } catch (e) {
    ElMessage.error((e as Error).message);
  }
}

function onDone(status: TaskInfo["status"]): void {
  if (status === "succeeded") {
    ElMessage.success("入库完成");
  } else if (status === "failed") {
    ElMessage.error("入库失败");
  }
}

onMounted(refreshRaw);
</script>

<style scoped>
.card-title {
  display: flex;
  align-items: center;
  gap: 6px;
}
</style>
