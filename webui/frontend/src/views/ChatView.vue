<template>
  <div class="chat-view">
    <div class="chat-main">
      <div class="chat-scroll" ref="scrollRef">
        <div v-if="messages.length === 0" class="chat-empty">
          <el-empty description="还没有对话，在下方输入问题开始体验流式 RAG 问答。" />
        </div>

        <div
          v-for="(msg, idx) in messages"
          :key="idx"
          :class="['chat-msg', msg.role]"
        >
          <div class="chat-avatar">{{ msg.role === "user" ? "我" : "AI" }}</div>
          <div class="chat-bubble">
            <MarkdownView :source="msg.content" />
            <div class="chat-meta" v-if="msg.role === 'assistant' && msg.latency">
              <el-tag size="small" effect="plain">
                检索 {{ msg.latency.retrieval_ms }} ms
              </el-tag>
              <el-tag size="small" effect="plain">
                生成 {{ msg.latency.generation_ms }} ms
              </el-tag>
              <el-tag size="small" effect="plain">
                总计 {{ msg.latency.total_ms }} ms
              </el-tag>
            </div>
            <div
              class="chat-citations"
              v-if="msg.role === 'assistant' && msg.retrievedTerms?.length"
            >
              <el-divider content-position="left">参考依据</el-divider>
              <TermCard
                v-for="(item, i) in msg.retrievedTerms"
                :key="i"
                :item="item"
              />
            </div>
          </div>
        </div>
      </div>

      <div class="chat-input-bar">
        <el-input
          v-model="input"
          type="textarea"
          :rows="3"
          :autosize="{ minRows: 2, maxRows: 6 }"
          placeholder="示例：什么是一致收敛？请给出 ε-δ 定义并举例。"
          :disabled="streaming"
          @keydown.ctrl.enter="submit"
        />
        <div class="chat-input-actions">
          <el-button
            type="primary"
            :loading="streaming"
            :disabled="!input.trim()"
            @click="submit"
          >发送 (Ctrl+Enter)</el-button>
          <el-button :disabled="streaming" @click="reset">清空对话</el-button>
        </div>
      </div>
    </div>

    <el-card class="chat-settings" shadow="never">
      <template #header><span>生成参数</span></template>
      <div class="settings-row">
        <el-checkbox v-model="useRag">启用 RAG 检索增强</el-checkbox>
      </div>
      <div class="settings-row">
        <div class="label">温度 {{ temperature.toFixed(2) }}</div>
        <el-slider v-model="temperature" :min="0" :max="1" :step="0.05" />
      </div>
      <div class="settings-row">
        <div class="label">Top-P {{ topP.toFixed(2) }}</div>
        <el-slider v-model="topP" :min="0" :max="1" :step="0.05" />
      </div>
      <div class="settings-row">
        <div class="label">最大生成 token {{ maxNewTokens }}</div>
        <el-slider
          v-model="maxNewTokens"
          :min="64"
          :max="2048"
          :step="64"
        />
      </div>
      <el-alert
        type="info"
        show-icon
        :closable="false"
        title="检索阈值和检索策略在 /config 页面调整"
        style="margin-top: 12px"
      />
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { nextTick, ref } from "vue";
import { ElMessage } from "element-plus";
import MarkdownView from "@/components/MarkdownView.vue";
import TermCard from "@/components/TermCard.vue";
import { useWebSocket } from "@/composables/useWebSocket";
import type {
  RagLatency,
  RagRetrievalItem,
  RagSource,
  WsRagEvent,
} from "@/types/api";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  retrievedTerms?: RagRetrievalItem[];
  sources?: RagSource[];
  latency?: RagLatency;
}

defineOptions({ name: "ChatView" });

const input = ref("");
const useRag = ref(true);
const temperature = ref(0.7);
const topP = ref(0.8);
const maxNewTokens = ref(512);

const messages = ref<ChatMessage[]>([]);
const streaming = ref(false);
const scrollRef = ref<HTMLDivElement | null>(null);

const { connect, send, disconnect } = useWebSocket<WsRagEvent>("/ws/rag");

async function scrollBottom(): Promise<void> {
  await nextTick();
  if (scrollRef.value) {
    scrollRef.value.scrollTop = scrollRef.value.scrollHeight;
  }
}

async function submit(): Promise<void> {
  const query = input.value.trim();
  if (!query || streaming.value) return;

  messages.value.push({ role: "user", content: query });
  const aiMsg: ChatMessage = { role: "assistant", content: "" };
  messages.value.push(aiMsg);
  input.value = "";
  streaming.value = true;
  await scrollBottom();

  try {
    await connect({
      onMessage: (event) => handleEvent(event, aiMsg),
      onClose: () => {
        streaming.value = false;
      },
      onError: () => {
        streaming.value = false;
        ElMessage.error("WebSocket 连接失败");
      },
    });
    send({
      query,
      useRag: useRag.value,
      temperature: temperature.value,
      topP: topP.value,
      maxNewTokens: maxNewTokens.value,
    });
  } catch (e) {
    streaming.value = false;
    ElMessage.error((e as Error).message || "连接失败");
  }
}

function handleEvent(event: WsRagEvent, msg: ChatMessage): void {
  if (event.type === "retrieval") {
    msg.retrievedTerms = event.retrievedTerms;
    msg.sources = event.sources;
    return;
  }
  if (event.type === "token") {
    msg.content += event.delta;
    scrollBottom();
    return;
  }
  if (event.type === "done") {
    msg.latency = event.latency;
    streaming.value = false;
    disconnect();
    return;
  }
  if (event.type === "error") {
    msg.content += `\n\n**错误：** ${event.error}`;
    streaming.value = false;
    disconnect();
  }
}

function reset(): void {
  messages.value = [];
}
</script>

<style scoped>
.chat-view {
  display: grid;
  grid-template-columns: 1fr 320px;
  gap: 16px;
  height: calc(100vh - 140px);
}

.chat-main {
  display: flex;
  flex-direction: column;
  background-color: #ffffff;
  border-radius: 6px;
  border: 1px solid #ebeef5;
  overflow: hidden;
}

.chat-scroll {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
}

.chat-empty {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.chat-msg {
  display: flex;
  gap: 12px;
  margin-bottom: 20px;
}

.chat-msg.user {
  flex-direction: row-reverse;
}

.chat-avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background-color: #409eff;
  color: #ffffff;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  flex-shrink: 0;
}

.chat-msg.user .chat-avatar {
  background-color: #67c23a;
}

.chat-bubble {
  max-width: 75%;
  padding: 12px 16px;
  background-color: #f4f7fb;
  border-radius: 8px;
}

.chat-msg.user .chat-bubble {
  background-color: #ecf5ff;
}

.chat-meta {
  margin-top: 8px;
  display: flex;
  gap: 6px;
}

.chat-citations {
  margin-top: 12px;
}

.chat-input-bar {
  border-top: 1px solid #ebeef5;
  padding: 12px 16px;
  background-color: #fafafa;
}

.chat-input-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin-top: 8px;
}

.chat-settings {
  height: fit-content;
}

.settings-row {
  margin-bottom: 12px;
}

.settings-row .label {
  color: #606266;
  margin-bottom: 4px;
  font-size: 13px;
}
</style>
