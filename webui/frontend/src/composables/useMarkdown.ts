import { marked } from "marked";
import hljs from "highlight.js";
import katex from "katex";

// 数学公式占位符策略：先用标签把 LaTeX 抽出，交给 marked 处理，再替换回渲染后的 HTML。
interface MathPlaceholder {
  token: string;
  html: string;
}

function replaceMath(source: string): { text: string; placeholders: MathPlaceholder[] } {
  const placeholders: MathPlaceholder[] = [];
  let counter = 0;

  const wrap = (tex: string, display: boolean): string => {
    const token = `@@MATH_${counter++}_${display ? "D" : "I"}@@`;
    let html: string;
    try {
      html = katex.renderToString(tex, {
        displayMode: display,
        throwOnError: false,
        output: "html",
      });
    } catch {
      html = `<code class="math-error">${escapeHtml(tex)}</code>`;
    }
    placeholders.push({ token, html });
    return token;
  };

  // 行间 $$...$$ 优先
  let text = source.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => wrap(tex.trim(), true));
  // 行内 $...$
  text = text.replace(
    /(^|[^\\])\$([^$\n]+?)\$/g,
    (_match, prefix: string, tex: string) => `${prefix}${wrap(tex.trim(), false)}`,
  );
  return { text, placeholders };
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

marked.setOptions({
  breaks: true,
  gfm: true,
});

export function renderMarkdown(raw: string): string {
  if (!raw) return "";
  const { text, placeholders } = replaceMath(raw);
  let html = marked.parse(text, { async: false }) as string;

  for (const ph of placeholders) {
    html = html.split(ph.token).join(ph.html);
  }

  // 代码高亮（轻量：扫描 <pre><code class="language-xxx">）
  html = html.replace(
    /<pre><code class="language-(\w+)">([\s\S]*?)<\/code><\/pre>/g,
    (_m, lang: string, code: string) => {
      try {
        const decoded = code
          .replace(/&amp;/g, "&")
          .replace(/&lt;/g, "<")
          .replace(/&gt;/g, ">")
          .replace(/&quot;/g, '"');
        const highlighted = hljs.highlight(decoded, { language: lang }).value;
        return `<pre><code class="hljs language-${lang}">${highlighted}</code></pre>`;
      } catch {
        return `<pre><code class="language-${lang}">${code}</code></pre>`;
      }
    },
  );

  return html;
}
