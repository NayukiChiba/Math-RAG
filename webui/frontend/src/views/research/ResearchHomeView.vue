<template>
  <el-card shadow="never">
    <template #header>
      <div>研究线 CLI 命令</div>
    </template>
    <el-alert
      type="info"
      :closable="false"
      show-icon
      title="每个命令对应 python main.py research <command> 的底层 runner。点击进入后可透传参数并实时查看日志。"
    />

    <el-row :gutter="12" style="margin-top: 16px">
      <el-col :span="8" v-for="cmd in commands" :key="cmd.name">
        <el-card class="cmd-card" shadow="hover" @click="goto(cmd.name)">
          <div class="cmd-head">
            <el-icon><component :is="cmd.icon" /></el-icon>
            <span class="name">{{ cmd.name }}</span>
          </div>
          <div class="desc">{{ cmd.desc }}</div>
          <div class="module">
            <el-tag size="small" type="info">{{ cmd.module }}</el-tag>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </el-card>
</template>

<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useRouter } from "vue-router";
import {
  Aim,
  Compass,
  DataAnalysis,
  Document,
  Files,
  Histogram,
  MagicStick,
  Medal,
  Notification,
  PieChart,
  Promotion,
  Stamp,
  Star,
  TrendCharts,
} from "@element-plus/icons-vue";
import { researchApi } from "@/api/research";

interface CmdMeta {
  name: string;
  desc: string;
  module: string;
  icon: any;
}

const router = useRouter();

const DESC: Record<string, { desc: string; icon: any }> = {
  "generate-queries": { desc: "生成评测查询集（queries.jsonl）", icon: MagicStick },
  "build-term-mapping": { desc: "构建评测术语映射表", icon: Compass },
  "eval-retrieval": { desc: "正式检索评测（BM25/Vector/Hybrid 等对比）", icon: Aim },
  experiments: { desc: "端到端对比实验（norag/bm25/vector/hybrid）", icon: TrendCharts },
  "eval-generation": { desc: "生成质量评测", icon: Medal },
  "eval-generation-comparison": { desc: "生成质量对比", icon: Histogram },
  "significance-test": { desc: "统计显著性检验（配对 t / Bootstrap）", icon: DataAnalysis },
  report: { desc: "生成最终 Markdown 评测报告与图表", icon: Document },
  "full-reports": { desc: "全量评测总控（日志 + 定稿）", icon: Files },
  "publish-reports": { desc: "从历史运行发布定稿到 outputs/reports/", icon: Promotion },
  "quick-eval": { desc: "快速检索评测（调试用）", icon: Star },
  "defense-figures": { desc: "生成答辩演示图表", icon: PieChart },
  "add-missing-terms": { desc: "分析并补充语料缺失术语", icon: Notification },
  stats: { desc: "术语与语料统计可视化", icon: Stamp },
};

const commands = ref<CmdMeta[]>([]);

async function load(): Promise<void> {
  const map = await researchApi.listCommands();
  const list: CmdMeta[] = [];
  for (const [name, module] of Object.entries(map)) {
    list.push({
      name,
      module,
      desc: DESC[name]?.desc || "",
      icon: DESC[name]?.icon || Document,
    });
  }
  // 手动加入 stats（后端未列在 _COMMAND_MODULES 中）
  if (!map["stats"]) {
    list.push({
      name: "stats",
      module: "research.dataStat.run_statistics",
      desc: DESC["stats"].desc,
      icon: DESC["stats"].icon,
    });
  }
  commands.value = list;
}

function goto(name: string): void {
  router.push(`/research/${name}`);
}

onMounted(load);
</script>

<style scoped>
.cmd-card {
  margin-bottom: 12px;
  cursor: pointer;
}

.cmd-head {
  display: flex;
  align-items: center;
  gap: 6px;
  font-weight: 600;
  font-size: 15px;
  margin-bottom: 6px;
}

.cmd-head .name {
  font-family: "SFMono-Regular", Consolas, monospace;
  color: #409eff;
}

.desc {
  color: #606266;
  font-size: 13px;
  min-height: 36px;
}

.module {
  margin-top: 6px;
}
</style>
