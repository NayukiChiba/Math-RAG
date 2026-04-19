"""webui.backend.routes.configRoutes 的 TOML 写入工具测试。

重点覆盖：
- 多行字符串（\"\"\"...\"\"\"）值写入时要保持多行格式
- 替换 section 内某个 key 时不会破坏其它多行值
- 子表 header（如 [retrieval.embedding]）可以精确匹配
"""

from __future__ import annotations


def _import_module():
    from webui.backend.routes import configRoutes

    return configRoutes


def test_tomlValue_single_line_string():
    mod = _import_module()
    assert mod._tomlValue("hello") == '"hello"'


def test_tomlValue_quote_escaping():
    mod = _import_module()
    assert mod._tomlValue('he said "hi"') == '"he said \\"hi\\""'


def test_tomlValue_backslash_escaping():
    mod = _import_module()
    assert mod._tomlValue("a\\b") == '"a\\\\b"'


def test_tomlValue_multiline_uses_triple_quotes():
    mod = _import_module()
    out = mod._tomlValue("line1\nline2")
    assert out.startswith('"""')
    assert out.endswith('"""')
    assert "line1\nline2" in out


def test_tomlValue_bool_and_number():
    mod = _import_module()
    assert mod._tomlValue(True) == "true"
    assert mod._tomlValue(False) == "false"
    assert mod._tomlValue(42) == "42"
    assert mod._tomlValue(1.5) == "1.5"


def test_tomlValue_list():
    mod = _import_module()
    assert mod._tomlValue([1, 2, 3]) == "[1, 2, 3]"
    assert mod._tomlValue(["a", "b"]) == '["a", "b"]'


def test_replaceInSection_preserves_multiline_value_when_other_key_changes():
    """替换 section 内某个 key 时，多行字符串值必须原样保留（只有一份）。"""
    mod = _import_module()
    lines = [
        "[ocr.api]\n",
        'model = "old"\n',
        'prompt = """first line\n',
        "second line\n",
        'third line"""\n',
        "temperature = 0.1\n",
        "[next]\n",
        "k = 1\n",
    ]
    newLines, remaining = mod._replaceInSection(
        lines, "ocr.api", {"model": "new-model"}
    )
    text = "".join(newLines)

    assert remaining == {}
    assert 'model = "new-model"' in text
    # prompt 多行值整体保留且只出现一次
    assert text.count('prompt = """') == 1
    assert text.count('third line"""') == 1
    assert "second line" in text
    # 其它字段未变
    assert "temperature = 0.1" in text
    # 其它 section 不受影响
    assert "[next]" in text


def test_replaceInSection_updates_multiline_value():
    """把多行值替换为新的多行内容：旧多行应被整块清除。"""
    mod = _import_module()
    lines = [
        "[ocr.api]\n",
        'prompt = """old line 1\n',
        "old line 2\n",
        'old line 3"""\n',
        'model = "x"\n',
    ]
    newLines, remaining = mod._replaceInSection(
        lines, "ocr.api", {"prompt": "new line A\nnew line B"}
    )
    text = "".join(newLines)

    assert remaining == {}
    # 旧的三行都被清除
    assert "old line 1" not in text
    assert "old line 2" not in text
    assert "old line 3" not in text
    # 新的多行值以三引号包裹
    assert text.count('"""') == 2
    assert "new line A" in text
    assert "new line B" in text
    # 其它 key 不变
    assert 'model = "x"' in text


def test_replaceInSection_matches_subtable_header():
    """[retrieval.embedding] 等带点号的子表 header 应能精确匹配。"""
    mod = _import_module()
    lines = [
        "[retrieval]\n",
        "bm25_default_weight = 0.7\n",
        "[retrieval.embedding]\n",
        'engine = "local"\n',
        "[retrieval.reranker]\n",
        'engine = "local"\n',
    ]
    # 修改父表的 key，不应触碰子表
    newLines, remaining = mod._replaceInSection(
        lines, "retrieval", {"bm25_default_weight": 0.5}
    )
    assert remaining == {}
    text = "".join(newLines)
    assert "bm25_default_weight = 0.5" in text
    # 子表仍然保留
    assert "[retrieval.embedding]" in text
    assert "[retrieval.reranker]" in text

    # 精确替换子表字段
    newLines2, remaining2 = mod._replaceInSection(
        lines, "retrieval.embedding", {"engine": "api"}
    )
    assert remaining2 == {}
    text2 = "".join(newLines2)
    # 子表内已改为 api
    subtableIdx = text2.index("[retrieval.embedding]")
    nextHeaderIdx = text2.index("[retrieval.reranker]")
    subSection = text2[subtableIdx:nextHeaderIdx]
    assert 'engine = "api"' in subSection
    # reranker 未被波及，仍是 local
    rerankSection = text2[nextHeaderIdx:]
    assert 'engine = "local"' in rerankSection


def test_replaceInSection_appends_missing_key_to_remaining():
    mod = _import_module()
    lines = [
        "[ocr]\n",
        'device = "cuda"\n',
    ]
    newLines, remaining = mod._replaceInSection(
        lines, "ocr", {"device": "cpu", "new_field": 42}
    )
    assert remaining == {"new_field": 42}
    assert 'device = "cpu"' in "".join(newLines)


def test_roundtrip_preserves_ocr_prompt():
    """完整回合：读当前 config -> PATCH model 字段 -> 写回 -> tomllib 可再次解析。"""
    mod = _import_module()
    # 模拟包含多行 prompt 的 config.toml 内容
    original = (
        "[ocr.api]\n"
        'model = "deepseek-vl2"\n'
        'api_base = "https://example"\n'
        'prompt = """Line A\n'
        "Line B\n"
        "Line C with 公式 $x$\n"
        'End"""\n'
        "temperature = 0.1\n"
    )
    # PATCH model
    lines = original.splitlines(keepends=True)
    newLines, _ = mod._replaceInSection(lines, "ocr.api", {"model": "deepseek-chat"})
    newText = "".join(newLines)

    # 新文本应能被 tomllib 解析（构造完整文件）
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore

    parsed = tomllib.loads(newText)
    assert parsed["ocr"]["api"]["model"] == "deepseek-chat"
    assert parsed["ocr"]["api"]["api_base"] == "https://example"
    assert "Line A\nLine B\nLine C with 公式 $x$\nEnd" == parsed["ocr"]["api"]["prompt"]
    assert parsed["ocr"]["api"]["temperature"] == 0.1
