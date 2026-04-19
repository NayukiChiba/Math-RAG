<template>
  <el-card shadow="never" class="engine-card">
    <template #header>
      <div class="card-title">
        <div class="card-title-main">
          <span class="title">{{ title }}</span>
          <el-tag
            :type="editable.engine === 'api' ? 'success' : 'info'"
            size="small"
          >
            {{ engineLabel }}
          </el-tag>
        </div>
        <div class="card-title-actions">
          <el-button size="small" @click="reset" :disabled="!dirty">
            重置
          </el-button>
          <el-button
            size="small"
            type="primary"
            :loading="saving"
            :disabled="!dirty"
            @click="save"
          >
            保存
          </el-button>
        </div>
      </div>
    </template>

    <el-alert
      v-if="description"
      type="info"
      :closable="false"
      show-icon
      :title="description"
      style="margin-bottom: 12px"
    />

    <div class="engine-switch">
      <span class="switch-label">引擎</span>
      <el-radio-group v-model="editable.engine" @change="onEngineChange">
        <el-radio-button value="local">本地</el-radio-button>
        <el-radio-button value="api">API</el-radio-button>
      </el-radio-group>
      <span v-if="engineHint" class="hint-inline">{{ engineHint }}</span>
    </div>

    <el-form label-width="160px" class="engine-form">
      <el-form-item
        v-for="field in activeFields"
        :key="field.key"
        :label="field.label"
      >
        <el-input
          v-if="field.type === 'textarea'"
          v-model="editable[field.key] as string"
          type="textarea"
          :rows="field.rows ?? 4"
          :placeholder="field.placeholder"
        />
        <el-input-number
          v-else-if="field.type === 'number'"
          v-model="editable[field.key] as number"
          controls-position="right"
          :step="field.step ?? 1"
          :min="field.min"
          :max="field.max"
        />
        <el-switch
          v-else-if="field.type === 'switch'"
          v-model="editable[field.key] as boolean"
        />
        <el-input
          v-else
          v-model="editable[field.key] as string"
          :placeholder="field.placeholder"
          :type="field.type === 'password' ? 'password' : 'text'"
          :show-password="field.type === 'password'"
        />
        <div v-if="field.hint" class="field-hint">{{ field.hint }}</div>
      </el-form-item>
    </el-form>
  </el-card>
</template>

<script setup lang="ts">
import { computed, reactive, ref, watch } from "vue";
import { ElMessage } from "element-plus";

export type FieldType = "text" | "number" | "switch" | "textarea" | "password";

export interface FieldDef {
  key: string;
  label: string;
  type: FieldType;
  placeholder?: string;
  hint?: string;
  step?: number;
  min?: number;
  max?: number;
  rows?: number;
}

const props = defineProps<{
  title: string;
  description?: string;
  engineHint?: string;
  values: Record<string, unknown>;
  localFields: FieldDef[];
  apiFields: FieldDef[];
}>();

const emit = defineEmits<{
  (e: "save", payload: Record<string, unknown>): Promise<void> | void;
}>();

const editable = reactive<Record<string, unknown>>({ ...props.values });
const saving = ref(false);

watch(
  () => props.values,
  (next) => {
    for (const key of Object.keys(editable)) {
      delete editable[key];
    }
    Object.assign(editable, next);
  },
  { deep: true },
);

const activeFields = computed<FieldDef[]>(() => {
  return editable.engine === "api" ? props.apiFields : props.localFields;
});

const engineLabel = computed(() =>
  editable.engine === "api" ? "API" : "本地",
);

const dirty = computed(() => {
  return JSON.stringify(editable) !== JSON.stringify(props.values);
});

function onEngineChange(): void {
  // 切换引擎时仅更新本地状态；等用户点「保存」才写回
}

function reset(): void {
  for (const key of Object.keys(editable)) {
    delete editable[key];
  }
  Object.assign(editable, props.values);
}

async function save(): Promise<void> {
  saving.value = true;
  try {
    await emit("save", { ...editable });
  } catch (e) {
    ElMessage.error((e as Error).message);
  } finally {
    saving.value = false;
  }
}
</script>

<style scoped>
.engine-card {
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

.engine-switch {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
  padding: 10px 12px;
  background: #f5f7fa;
  border-radius: 6px;
}

.switch-label {
  font-weight: 500;
  color: #606266;
}

.hint-inline {
  color: #909399;
  font-size: 12px;
}

.engine-form {
  padding-top: 4px;
}

.field-hint {
  color: #909399;
  font-size: 12px;
  margin-top: 2px;
  line-height: 1.4;
}
</style>
