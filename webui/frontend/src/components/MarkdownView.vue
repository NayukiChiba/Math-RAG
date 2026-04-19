<template>
  <div class="markdown-body" v-html="html"></div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import { renderMarkdown } from "@/composables/useMarkdown";

const props = defineProps<{ source: string }>();

const html = computed(() => renderMarkdown(props.source));
</script>
