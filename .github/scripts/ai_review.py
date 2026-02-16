#!/usr/bin/env python3
"""
AI Code Review Script
调用 LLM API 对 PR diff 进行代码审查
"""

import os

from openai import OpenAI

SYSTEM_PROMPT = """你是一个资深的代码审查专家。请审查以下 Pull Request 的代码变更。

你需要关注以下方面：
1. **潜在的 Bug 和边缘情况** - 未处理的异常、空值检查、边界条件等
2. **安全漏洞** - SQL注入、XSS、敏感信息泄露、不安全的依赖等
3. **性能问题** - 不必要的循环、内存泄漏、N+1查询等
4. **代码质量** - 可读性、命名规范、重复代码、过于复杂的逻辑
5. **最佳实践** - 是否遵循该语言/框架的最佳实践

请用中文回复，格式如下：
- 如果发现问题，列出具体的问题和建议，引用具体的代码行
- 如果代码质量良好，简单说明即可
- 不要过度挑剔，只关注真正重要的问题

回复格式：
### 🤖 AI Code Review

**审查的提交:** `{commit_sha}`

#### 发现的问题

（如果有问题，按严重程度列出）

#### 总结

（简短总结代码质量）

---
<details>
<summary>ℹ️ 关于此审查</summary>

此审查由 AI 自动生成，仅供参考。如有误报请忽略。

</details>
"""


def get_diff_content() -> str:
    """读取 PR diff 内容"""
    diff_file = os.environ.get("diff_file", "pr_diff.txt")
    if os.path.exists(diff_file):
        with open(diff_file, encoding="utf-8", errors="ignore") as f:
            return f.read()

    # 备用：直接读取
    if os.path.exists("pr_diff.txt"):
        with open("pr_diff.txt", encoding="utf-8", errors="ignore") as f:
            return f.read()

    return ""


def truncate_diff(diff: str, max_chars: int = 60000) -> str:
    """截断过长的 diff，避免超出 token 限制"""
    if len(diff) <= max_chars:
        return diff

    return diff[:max_chars] + "\n\n... (diff 过长，已截断)"


def main():
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        print("Error: LLM_API_KEY not set")
        return

    base_url = os.environ.get("LLM_BASE_URL")  # 可选
    model = os.environ.get("LLM_MODEL")

    pr_title = os.environ.get("PR_TITLE", "")
    pr_body = os.environ.get("PR_BODY", "")

    # 获取 commit SHA
    commit_sha = os.environ.get("GITHUB_SHA", "unknown")[:10]

    diff_content = get_diff_content()
    if not diff_content:
        print("No diff content found, skipping review")
        return

    diff_content = truncate_diff(diff_content)

    # 构造用户消息
    user_message = f"""## Pull Request 信息

**标题:** {pr_title}

**描述:**
{pr_body or "无描述"}

## 代码变更 (diff)

```diff
{diff_content}
```

请审查以上代码变更。"""

    # 初始化 OpenAI 客户端（兼容其他 API）
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(commit_sha=commit_sha),
                },
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=2000,
        )

        review_content = response.choices[0].message.content

        # 写入结果文件
        with open("review_result.md", "w", encoding="utf-8") as f:
            f.write(review_content)

        print("Review completed successfully!")
        print(review_content)

    except Exception as e:
        print(f"Error calling LLM API: {e}")
        # 写入错误信息（可选择不发评论）
        # with open("review_result.md", "w") as f:
        #     f.write(f"⚠️ AI 审查失败: {str(e)}")


if __name__ == "__main__":
    main()
