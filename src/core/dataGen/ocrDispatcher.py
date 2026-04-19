"""
OCR 引擎分派：根据 config.toml [ocr].engine 决定用 pix2text 本地还是 API 多模态。

入口保持与 pix2text_ocr.main() 相同的 CLI 签名，方便被 CLI handlers 和 Web UI 任务复用。
"""

from __future__ import annotations

from core import config


def _getEngine() -> str:
    """读取 [ocr].engine，默认 local。"""
    data = {}
    try:
        data = config._get_config_data()
    except Exception:
        data = {}
    ocrCfg = data.get("ocr", {}) if isinstance(data, dict) else {}
    engine = str(ocrCfg.get("engine", "local")).strip().lower()
    return engine if engine in ("local", "api") else "local"


def main() -> None:
    engine = _getEngine()
    if engine == "api":
        print("[OCR Dispatcher] 选用引擎: api (多模态 AI)")
        from core.dataGen.apiOcr import main as _apiMain

        _apiMain()
    else:
        print("[OCR Dispatcher] 选用引擎: local (pix2text)")
        from core.dataGen.pix2text_ocr import main as _localMain

        _localMain()


if __name__ == "__main__":
    main()
