import { createRouter, createWebHashHistory } from "vue-router";
import type { RouteRecordRaw } from "vue-router";

const routes: RouteRecordRaw[] = [
  {
    path: "/",
    component: () => import("@/layouts/DefaultLayout.vue"),
    children: [
      {
        path: "",
        name: "home",
        component: () => import("@/views/HomeView.vue"),
        meta: { title: "概览" },
      },
      {
        path: "chat",
        name: "chat",
        component: () => import("@/views/ChatView.vue"),
        meta: { title: "RAG 问答" },
      },
      {
        path: "ingest",
        name: "ingest",
        component: () => import("@/views/IngestView.vue"),
        meta: { title: "PDF 入库" },
      },
      {
        path: "index",
        name: "index",
        component: () => import("@/views/IndexView.vue"),
        meta: { title: "检索索引" },
      },
      {
        path: "tasks",
        name: "tasks",
        component: () => import("@/views/TasksView.vue"),
        meta: { title: "任务中心" },
      },
      {
        path: "tasks/:taskId",
        name: "taskDetail",
        component: () => import("@/views/TaskDetailView.vue"),
        meta: { title: "任务详情" },
      },
      {
        path: "research",
        name: "research",
        component: () => import("@/views/research/ResearchHomeView.vue"),
        meta: { title: "研究线" },
      },
      {
        path: "research/:command",
        name: "researchCommand",
        component: () => import("@/views/research/ResearchCommandView.vue"),
        meta: { title: "研究线命令" },
      },
      {
        path: "reports",
        name: "reports",
        component: () => import("@/views/ReportsView.vue"),
        meta: { title: "报告中心" },
      },
      {
        path: "reports/:runId",
        name: "reportDetail",
        component: () => import("@/views/ReportDetailView.vue"),
        meta: { title: "报告详情" },
      },
      {
        path: "figures",
        name: "figures",
        component: () => import("@/views/FiguresView.vue"),
        meta: { title: "图表库" },
      },
      {
        path: "stats",
        name: "stats",
        component: () => import("@/views/StatsView.vue"),
        meta: { title: "术语统计" },
      },
      {
        path: "config",
        name: "config",
        component: () => import("@/views/ConfigView.vue"),
        meta: { title: "配置" },
      },
    ],
  },
];

const router = createRouter({
  history: createWebHashHistory(),
  routes,
});

export default router;
