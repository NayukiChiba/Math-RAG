<template>
  <el-container class="layout-root">
    <el-aside width="220px" class="layout-aside">
      <div class="brand">
        <span class="brand-title">Math-RAG</span>
        <span class="brand-sub">控制台</span>
      </div>
      <el-menu
        :default-active="activeMenu"
        router
        class="layout-menu"
        background-color="#001529"
        text-color="#c6d5e8"
        active-text-color="#ffffff"
      >
        <el-menu-item index="/">
          <el-icon><Odometer /></el-icon>
          <span>概览</span>
        </el-menu-item>
        <el-menu-item index="/chat">
          <el-icon><ChatDotSquare /></el-icon>
          <span>RAG 问答</span>
        </el-menu-item>
        <el-sub-menu index="product">
          <template #title>
            <el-icon><Box /></el-icon>
            <span>产品线</span>
          </template>
          <el-menu-item index="/ingest">PDF 入库</el-menu-item>
          <el-menu-item index="/index">检索索引</el-menu-item>
        </el-sub-menu>
        <el-menu-item index="/research">
          <el-icon><Histogram /></el-icon>
          <span>研究线</span>
        </el-menu-item>
        <el-menu-item index="/reports">
          <el-icon><Document /></el-icon>
          <span>报告中心</span>
        </el-menu-item>
        <el-menu-item index="/figures">
          <el-icon><PictureFilled /></el-icon>
          <span>图表库</span>
        </el-menu-item>
        <el-menu-item index="/stats">
          <el-icon><DataAnalysis /></el-icon>
          <span>术语统计</span>
        </el-menu-item>
        <el-menu-item index="/tasks">
          <el-icon><List /></el-icon>
          <span>任务中心</span>
        </el-menu-item>
        <el-menu-item index="/config">
          <el-icon><Setting /></el-icon>
          <span>配置</span>
        </el-menu-item>
      </el-menu>
    </el-aside>
    <el-container>
      <el-header class="layout-header">
        <div class="layout-title">
          {{ (route.meta?.title as string) || "Math-RAG" }}
        </div>
        <div class="layout-actions">
          <el-tag size="small" type="success">API 正常</el-tag>
        </div>
      </el-header>
      <el-main>
        <router-view v-slot="{ Component }">
          <keep-alive :include="['TasksView', 'HomeView']">
            <component :is="Component" />
          </keep-alive>
        </router-view>
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup lang="ts">
import { computed } from "vue";
import { useRoute } from "vue-router";
import {
  Box,
  ChatDotSquare,
  DataAnalysis,
  Document,
  Histogram,
  List,
  Odometer,
  PictureFilled,
  Setting,
} from "@element-plus/icons-vue";

const route = useRoute();

const activeMenu = computed(() => {
  const path = route.path;
  if (path.startsWith("/research")) return "/research";
  if (path.startsWith("/reports")) return "/reports";
  if (path.startsWith("/tasks")) return "/tasks";
  return path;
});
</script>

<style scoped>
.layout-root {
  height: 100vh;
}

.layout-aside {
  background-color: #001529;
  color: #ffffff;
  overflow-y: auto;
}

.brand {
  padding: 18px 20px 12px;
  display: flex;
  flex-direction: column;
  border-bottom: 1px solid #112a44;
}

.brand-title {
  font-size: 18px;
  font-weight: 600;
  color: #ffffff;
}

.brand-sub {
  font-size: 12px;
  color: #8ca7c6;
  margin-top: 4px;
}

.layout-menu {
  border-right: none;
}

.layout-header {
  background-color: #ffffff;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid #e4e7ed;
}

.layout-title {
  font-size: 16px;
  font-weight: 500;
}

.layout-actions {
  display: flex;
  gap: 8px;
}
</style>
