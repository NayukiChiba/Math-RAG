"""
使用 OpenAI 兼容接口在线生成数学名词数据（JSON）。
说明：不使用命令行参数，直接运行即可。
"""

import json
import os
import re
import sys
import time
from pathlib import Path

# 规范模块搜索路径，保证能定位项目根目录
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_toml(path):
    """读取 TOML 配置。"""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)


def _default_prompts():
    """默认提示词模板（建议放在 config.toml 中显式配置）。"""
    return {
        "prompt_system": (
            "你是数学术语编纂助手。请为指定术语生成标准化 JSON。"
            "必须严格输出 JSON 对象，不要输出多余解释。"
        ),
        "prompt_user": (
            "术语：{{term}}\n"
            "学科：{{subject_label}}\n"
            "要求：\n"
            "1) 输出字段必须包含：id, term, aliases, sense_id, subject, definitions, notation, formula, "
            "usage, applications, disambiguation, related_terms, search_keys, lang, confidence。\n"
            "2) definitions 至少包含 2 条（strict + alternative），必要时再补充 informal。\n"
            "3) 数学符号与公式必须使用 LaTeX。\n"
            "4) related_terms 至少 3 个。\n"
            "5) 仅输出 JSON 对象。\n"
            "6) 内容必须更具体，避免空泛表述。\n"
            "示例（格式参考，内容需针对当前术语生成）：\n"
            "{{example}}"
        ),
        "prompt_example": (
            "{\n"
            '  "id": "ma-柯西列",\n'
            '  "term": "柯西列",\n'
            '  "aliases": ["Cauchy 列"],\n'
            '  "sense_id": "1",\n'
            '  "subject": "数学分析",\n'
            '  "definitions": [\n'
            "    {\n"
            '      "type": "strict",\n'
            '      "text": "在度量空间 $(X,d)$ 中，序列 $\\{x_n\\}$ 称为柯西列，若对任意 $\\epsilon>0$，存在 $N$ 使得当 $m,n\\ge N$ 时，有 $d(x_m,x_n)<\\epsilon$。",\n'
            '      "conditions": "$(X,d)$ 为度量空间",\n'
            '      "notation": "$d(x_m,x_n)$"\n'
            "    }\n"
            "  ],\n"
            '  "notation": "$d(x_m,x_n)$",\n'
            '  "formula": ["\\\\forall\\\\epsilon>0\\\\,\\\\exists N\\\\,\\\\forall m,n\\\\ge N:\\\\,d(x_m,x_n)<\\\\epsilon"],\n'
            '  "usage": "用于刻画序列在度量空间中的内在收敛性，不依赖极限点是否存在。",\n'
            '  "applications": "用于证明完备性与收敛性等价性质；在函数空间中用于一致收敛的判别。",\n'
            '  "disambiguation": "不同于一般收敛，柯西列只要求后项彼此接近。",\n'
            '  "related_terms": ["完备性", "收敛", "度量空间"],\n'
            '  "search_keys": ["柯西列", "cauchy列", "cauchy"],\n'
            '  "lang": "zh",\n'
            '  "confidence": "medium"\n'
            "}\n"
        ),
        "prompt_repair_system": (
            "你是数学术语编纂助手。请修复并补全 JSON。"
            "必须严格输出 JSON 对象，不要输出多余解释。"
        ),
        "prompt_repair_user": (
            "术语：{{term}}\n"
            "学科：{{subject_label}}\n"
            "问题：{{reason}}\n"
            "请在保持结构完整的前提下补全内容，至少包含 1 条 strict 定义，"
            "并补齐公式、usage、applications、related_terms。\n"
            "仅输出修复后的 JSON 对象。\n"
            "当前 JSON：\n"
            "{{bad_json}}\n"
        ),
    }


def _load_config():
    """从 config.toml 读取配置。"""
    root_dir = Path(__file__).resolve().parent.parent
    config_path = root_dir / "config.toml"
    data = _load_toml(str(config_path))
    paths = data.get("paths", {})
    model_cfg = data.get("model", {})

    processed_dir = os.path.abspath(
        os.path.join(root_dir, paths.get("processed_dir", ""))
    )
    output_json = os.path.abspath(
        os.path.join(root_dir, paths.get("model_output_json", ""))
    )

    cfg = {
        "root_dir": str(root_dir),
        "processed_dir": processed_dir,
        "output_json": output_json,
        "api_base": model_cfg.get("api_base", "").rstrip("/"),
        "model": model_cfg.get("model", ""),
        "api_key_env": model_cfg.get("api_key_env", "API-KEY"),
        "subject_label": model_cfg.get("subject_label", ""),
        "seed_terms": model_cfg.get("seed_terms", []),
        "max_tokens": model_cfg.get("max_tokens", 900),
        "temperature": model_cfg.get("temperature", 0.3),
        "top_p": model_cfg.get("top_p", 0.9),
        "max_retries": model_cfg.get("max_retries", 2),
        "min_definition_chars": model_cfg.get("min_definition_chars", 40),
        "min_usage_chars": model_cfg.get("min_usage_chars", 20),
        "min_applications_chars": model_cfg.get("min_applications_chars", 20),
        "min_formula_count": model_cfg.get("min_formula_count", 1),
        "endpoint": model_cfg.get("endpoint", "/chat/completions"),
        "request_timeout": model_cfg.get("request_timeout", 60),
        # 不配置 rpm/tpm 视为不限速
        "rpm": model_cfg.get("rpm", 0),
        "tpm": model_cfg.get("tpm", 0),
        "retry_wait_seconds": model_cfg.get("retry_wait_seconds", 2),
        "clear_context_each_request": model_cfg.get("clear_context_each_request", True),
        "stream": model_cfg.get("stream", False),
        "start_index": model_cfg.get("start_index", 1),
        "prompt_system": model_cfg.get("prompt_system", ""),
        "prompt_user": model_cfg.get("prompt_user", ""),
        "prompt_example": model_cfg.get("prompt_example", ""),
        "prompt_repair_system": model_cfg.get("prompt_repair_system", ""),
        "prompt_repair_user": model_cfg.get("prompt_repair_user", ""),
    }

    defaults = _default_prompts()
    for key, value in defaults.items():
        if not cfg.get(key):
            cfg[key] = value
    return cfg


def _load_env_value(root_dir, key):
    """从环境变量或 .env 读取密钥（支持含 '-' 的键名）。"""
    if key in os.environ:
        return os.environ.get(key)

    env_path = os.path.join(root_dir, ".env")
    if not os.path.isfile(env_path):
        return None

    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            if k != key:
                continue
            v = v.strip().strip('"').strip("'")
            return v
    return None


def _ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_key(text):
    if not text:
        return ""
    return re.sub(r"\s+", "", str(text)).lower()


def _estimate_tokens(text):
    """粗略估算 token 数量（按 2 字符≈1 token）。"""
    if not text:
        return 0
    return max(1, len(text) // 2)


def _render_prompt(template, **kwargs):
    """用 {{key}} 形式替换模板占位符。"""
    if not template:
        return ""
    text = template
    for key, value in kwargs.items():
        text = text.replace(f"{{{{{key}}}}}", str(value))
    return text


def _build_id(term):
    key = _normalize_key(term)
    if not key:
        return "ma-unknown"
    return f"ma-{key}"


def _extract_json_block(text):
    """从模型输出中提取 JSON 对象。"""
    if not text:
        return None

    code_block = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    if code_block:
        return code_block.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _quality_check(record, cfg):
    """检查输出质量，决定是否需要重试。"""
    if not isinstance(record, dict):
        return False, "输出不是 JSON 对象"

    definitions = _ensure_list(record.get("definitions"))
    if not definitions:
        return False, "definitions 为空"

    strict_defs = [
        d for d in definitions if isinstance(d, dict) and d.get("type") == "strict"
    ]
    if not strict_defs:
        return False, "缺少 strict 定义"

    alt_defs = [
        d for d in definitions if isinstance(d, dict) and d.get("type") == "alternative"
    ]
    if not alt_defs:
        return False, "缺少 alternative 定义"

    strict_text = strict_defs[0].get("text", "")
    if len(strict_text) < cfg["min_definition_chars"]:
        return False, "strict 定义过短"

    formulas = _ensure_list(record.get("formula"))
    if (
        len([f for f in formulas if isinstance(f, str) and f.strip()])
        < cfg["min_formula_count"]
    ):
        return False, "公式数量不足"

    usage = record.get("usage") or ""
    applications = record.get("applications") or ""
    if len(usage) < cfg["min_usage_chars"]:
        return False, "usage 过短"
    if len(applications) < cfg["min_applications_chars"]:
        return False, "applications 过短"

    related_terms = _ensure_list(record.get("related_terms"))
    if len([t for t in related_terms if isinstance(t, str) and t.strip()]) < 3:
        return False, "related_terms 过少"

    return True, ""


def _normalize_record(record, term, model_id, subject_label):
    """补齐字段，确保符合 plan.md 规定的格式。"""
    record = record or {}

    record_id = record.get("id") or _build_id(term)
    aliases = _ensure_list(record.get("aliases"))
    definitions = _ensure_list(record.get("definitions"))

    if not definitions:
        definitions = [
            {
                "type": "informal",
                "text": f"{term} 是数学中的一个概念。",
                "conditions": "",
                "notation": "",
            }
        ]

    fixed_definitions = []
    for item in definitions:
        if not isinstance(item, dict):
            continue
        fixed_definitions.append(
            {
                "type": item.get("type") or "informal",
                "text": item.get("text") or "",
                "conditions": item.get("conditions") or "",
                "notation": item.get("notation") or "",
            }
        )

    keys = set()
    if term:
        keys.add(term)
        keys.add(_normalize_key(term))
    for a in aliases:
        if a:
            keys.add(a)
            keys.add(_normalize_key(a))
    search_keys = [k for k in keys if k]

    return {
        "id": record_id,
        "term": record.get("term") or term,
        "aliases": aliases,
        "sense_id": record.get("sense_id") or "1",
        "subject": record.get("subject") or subject_label,
        "definitions": fixed_definitions,
        "notation": record.get("notation") or "",
        "formula": _ensure_list(record.get("formula")),
        "usage": record.get("usage") or "",
        "applications": record.get("applications") or "",
        "disambiguation": record.get("disambiguation") or "",
        "related_terms": _ensure_list(record.get("related_terms")),
        "search_keys": search_keys,
        "lang": record.get("lang") or "zh",
        "confidence": record.get("confidence") or "low",
    }


def _build_prompt(cfg, term):
    """从配置构造提示词。"""
    example = cfg["prompt_example"]
    system = _render_prompt(
        cfg["prompt_system"],
        term=term,
        subject_label=cfg["subject_label"],
        example=example,
    )
    user = _render_prompt(
        cfg["prompt_user"],
        term=term,
        subject_label=cfg["subject_label"],
        example=example,
    )
    return system, user


def _build_repair_prompt(cfg, term, bad_json, reason):
    """从配置构造修复提示词。"""
    system = _render_prompt(
        cfg["prompt_repair_system"],
        term=term,
        subject_label=cfg["subject_label"],
        reason=reason,
        bad_json=bad_json,
    )
    user = _render_prompt(
        cfg["prompt_repair_user"],
        term=term,
        subject_label=cfg["subject_label"],
        reason=reason,
        bad_json=bad_json,
    )
    return system, user


def _call_model(cfg, api_key, messages):
    """调用 OpenAI 兼容接口。"""
    import requests

    url = f"{cfg['api_base']}{cfg['endpoint']}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": cfg["temperature"],
        "top_p": cfg["top_p"],
        "max_tokens": cfg["max_tokens"],
        "stream": cfg["stream"],
    }
    resp = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=cfg["request_timeout"],
        stream=cfg["stream"],
    )
    if resp.status_code in (429, 503):
        raise RuntimeError(f"HTTP {resp.status_code}")
    resp.raise_for_status()

    if not cfg["stream"]:
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    parts = []
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data:"):
            data_str = line[len("data:") :].strip()
        else:
            data_str = line.strip()
        if not data_str or data_str == "[DONE]":
            continue
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if content:
            parts.append(content)
    return "".join(parts)


def main():
    cfg = _load_config()
    if not cfg["api_base"]:
        print("config.toml 中未配置 model.api_base。")
        return
    if not cfg["model"]:
        print("config.toml 中未配置 model.model。")
        return

    api_key = _load_env_value(cfg["root_dir"], cfg["api_key_env"])
    if not api_key:
        print("未找到 API Key，请检查 .env 或环境变量。")
        return

    os.makedirs(cfg["processed_dir"], exist_ok=True)

    terms = list(cfg["seed_terms"])
    if not terms:
        print("model.seed_terms 为空，请先在 config.toml 中填写种子术语。")
        return

    try:
        start_index = int(cfg["start_index"])
    except (TypeError, ValueError):
        start_index = 1
    if start_index < 1:
        start_index = 1
    if start_index > len(terms):
        print("start_index 超出术语数量，已无可处理项。")
        return
    terms = terms[start_index - 1 :]

    success_terms = []
    failed_terms = []
    failed_reasons = {}

    first_item = True
    output_dir = os.path.dirname(cfg["output_json"])
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    out_f = open(cfg["output_json"], "w", encoding="utf-8")
    out_f.write("[\n")

    total = len(terms)
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    iterator = tqdm(terms, desc="生成术语") if tqdm else terms
    last_request_at = 0.0
    current_term = None
    interrupted = False

    try:
        for idx, term in enumerate(iterator, start=1):
            current_term = term
            if not tqdm:
                print(f"正在生成 {idx}/{total}：{term}")

            record = None
            last_json = ""
            last_reason = ""
            ok = False

            for attempt in range(cfg["max_retries"] + 1):
                if attempt == 0:
                    system, user = _build_prompt(cfg, term)
                else:
                    system, user = _build_repair_prompt(
                        cfg, term, last_json, last_reason
                    )

                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]

                try:
                    now = time.time()
                    min_interval = 0.0
                    if cfg["rpm"] and cfg["rpm"] > 0:
                        min_interval = 60.0 / cfg["rpm"]

                    prompt_tokens = _estimate_tokens(system + user)
                    token_budget = prompt_tokens + cfg["max_tokens"]
                    if cfg["tpm"] and cfg["tpm"] > 0:
                        min_interval_tpm = 60.0 * token_budget / cfg["tpm"]
                    else:
                        min_interval_tpm = 0.0

                    min_interval = max(min_interval, min_interval_tpm)
                    sleep_for = min_interval - (now - last_request_at)
                    if sleep_for > 0:
                        time.sleep(sleep_for)

                    gen_text = _call_model(cfg, api_key, messages)
                    last_request_at = time.time()
                except Exception as e:
                    err = str(e)
                    last_reason = f"请求失败：{err}"
                    if "HTTP 429" in err or "HTTP 503" in err:
                        time.sleep(max(cfg["retry_wait_seconds"], 10))
                    else:
                        time.sleep(cfg["retry_wait_seconds"])
                    continue

                json_block = _extract_json_block(gen_text)
                if json_block:
                    last_json = json_block
                    try:
                        record = json.loads(json_block)
                    except json.JSONDecodeError:
                        record = None
                else:
                    record = None
                    last_json = gen_text

                ok, reason = _quality_check(record, cfg)
                if ok:
                    break
                last_reason = reason
                time.sleep(cfg["retry_wait_seconds"])

            item = _normalize_record(record, term, cfg["model"], cfg["subject_label"])
            if not first_item:
                out_f.write(",\n")
            out_f.write(json.dumps(item, ensure_ascii=False, indent=2))
            out_f.flush()
            first_item = False

            if ok:
                success_terms.append(term)
            else:
                failed_terms.append(term)
                failed_reasons[term] = last_reason or "质量未达标"

            if not tqdm:
                print(f"完成 {idx}/{total}：{term}")
    except KeyboardInterrupt:
        interrupted = True
        print(f"已中断，当前词条：{current_term}")
    except Exception as e:
        interrupted = True
        print(f"异常中断：{e}，当前词条：{current_term}")
    finally:
        out_f.write("\n]\n")
        out_f.close()

    print(f"生成完成：{cfg['output_json']}")
    print(f"成功数量：{len(success_terms)}")
    print(f"失败数量：{len(failed_terms)}")
    if failed_terms:
        print("失败词条：")
        for t in failed_terms:
            print(f"- {t}（原因：{failed_reasons.get(t, '未知')}）")
    if interrupted:
        print(f"中断位置：{current_term}")


if __name__ == "__main__":
    main()
