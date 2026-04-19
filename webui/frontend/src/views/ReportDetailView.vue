<template>
  <div class="report-detail">
    <el-page-header @back="$router.push('/reports')" :content="runId" />

    <el-row :gutter="12" style="margin-top: 12px">
      <el-col :span="7">
        <el-card shadow="never">
          <template #header>
            <div>文件列表</div>
          </template>
          <el-tree
            :data="tree"
            :props="{ label: 'label', children: 'children' }"
            node-key="path"
            @node-click="onClick"
            default-expand-all
          />
        </el-card>
      </el-col>
      <el-col :span="17">
        <el-card shadow="never">
          <template #header>
            <div class="card-title">
              <span>{{ currentFile || "选择左侧文件查看" }}</span>
              <el-button
                v-if="currentFile"
                size="small"
                @click="open(fileUrl(currentFile))"
              >下载 / 在新窗口打开</el-button>
            </div>
          </template>
          <div v-if="!currentFile" class="hint">未选择文件</div>
          <div v-else-if="loading" class="hint">加载中...</div>
          <div v-else>
            <MarkdownView v-if="isMarkdown" :source="content" />
            <img
              v-else-if="isImage"
              :src="fileUrl(currentFile)"
              style="max-width: 100%"
            />
            <pre v-else-if="isText" class="pre-text">{{ content }}</pre>
            <div v-else class="hint">
              该文件类型不支持预览，请下载查看。
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import MarkdownView from "@/components/MarkdownView.vue";
import { reportsApi } from "@/api/reports";

interface TreeNode {
  label: string;
  path?: string;
  children?: TreeNode[];
}

const route = useRoute();
const runId = computed(() => route.params.runId as string);

const tree = ref<TreeNode[]>([]);
const currentFile = ref<string | null>(null);
const content = ref<string>("");
const loading = ref(false);

const isMarkdown = computed(() => currentFile.value?.toLowerCase().endsWith(".md") ?? false);
const isImage = computed(() =>
  /\.(png|jpe?g|svg|gif|webp)$/i.test(currentFile.value || ""),
);
const isText = computed(() =>
  /\.(json|jsonl|txt|md|log|yaml|yml|toml|csv|py)$/i.test(currentFile.value || ""),
);

function buildTree(files: { relPath: string }[]): TreeNode[] {
  const root: TreeNode = { label: "/", children: [] };
  for (const file of files) {
    const parts = file.relPath.split("/");
    let node = root;
    for (let i = 0; i < parts.length; i++) {
      const isLast = i === parts.length - 1;
      const name = parts[i];
      if (!node.children) node.children = [];
      let existing = node.children.find((c) => c.label === name);
      if (!existing) {
        existing = isLast
          ? { label: name, path: file.relPath }
          : { label: name, children: [] };
        node.children.push(existing);
      }
      node = existing;
    }
  }
  return root.children || [];
}

async function refresh(): Promise<void> {
  const data = await reportsApi.runTree(runId.value);
  tree.value = buildTree(data.files);
}

async function onClick(node: TreeNode): Promise<void> {
  if (!node.path) return;
  currentFile.value = node.path;
  content.value = "";
  if (!isText.value) return;
  loading.value = true;
  try {
    content.value = await reportsApi.runFileText(runId.value, node.path);
  } finally {
    loading.value = false;
  }
}

function fileUrl(path: string): string {
  return reportsApi.runFileUrl(runId.value, path);
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

.hint {
  color: #909399;
  padding: 20px;
  text-align: center;
}

.pre-text {
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 12px;
  background-color: #f5f7fa;
  padding: 12px;
  border-radius: 4px;
  max-height: 600px;
  overflow: auto;
  white-space: pre-wrap;
}
</style>
