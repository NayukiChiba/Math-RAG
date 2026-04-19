"""
多模态 AI OCR：调用 OpenAI 兼容的 Vision API 把 PDF 页面转成 Markdown。

使用方法（与 pix2text_ocr.py 对齐）：
    python -m core.dataGen.apiOcr                     # 按顺序处理所有 PDF
    python -m core.dataGen.apiOcr "书名.pdf"           # 只处理指定的 PDF
    python -m core.dataGen.apiOcr "书名.pdf" 136       # 从第 136 页开始

配置项在 config.toml [ocr] 与 [ocr.api] 段。
"""

import base64
import io
import os
import sys
import time

from core import config

_LOADER = None


def _getLoader():
    global _LOADER
    if _LOADER is None:
        from core.utils import getFileLoader

        _LOADER = getFileLoader()
    return _LOADER


def _getOcrApiConfig() -> dict:
    """读取 [ocr.api] 子段配置，提供默认值。"""
    data = (
        _getLoader().toml(config.CONFIG_TOML)
        if os.path.isfile(config.CONFIG_TOML)
        else {}
    )
    ocr_cfg = data.get("ocr", {}) if isinstance(data, dict) else {}
    api_cfg = ocr_cfg.get("api", {}) if isinstance(ocr_cfg, dict) else {}
    defaults = {
        "api_base": "https://api.deepseek.com/v1",
        "model": "deepseek-vl2",
        "api_key_env": "API-KEY",
        "max_tokens": 4000,
        "temperature": 0.1,
        "prompt": (
            "请将图片中的内容完整转录为 Markdown 格式。要求：\n"
            "1. 保留原始段落结构\n"
            "2. 所有数学公式使用 LaTeX（行内 $...$，行间 $$...$$）\n"
            "3. 表格使用 Markdown 表格语法\n"
            "4. 不要输出其它解释，只输出 Markdown 内容"
        ),
    }
    result = dict(defaults)
    result.update({k: v for k, v in api_cfg.items() if v is not None})
    return result


def _resolveApiKey(envName: str) -> str:
    """优先从环境变量读取 API Key，回退到 .env 文件。"""
    key = os.environ.get(envName)
    if key:
        return key
    envPath = os.path.join(config.PROJECT_ROOT, ".env")
    if not os.path.isfile(envPath):
        raise RuntimeError(f"未找到 API 密钥，请在环境或 .env 中配置 {envName}")
    with open(envPath, encoding="utf-8") as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                if k.strip() == envName:
                    return v.strip().strip("'").strip('"')
    raise RuntimeError(f"未找到 API 密钥，请在环境或 .env 中配置 {envName}")


def _renderPageToImage(pdfPath: str, pageIdx: int, dpi: int):
    """使用 PyMuPDF 渲染单页为 PIL Image。"""
    import fitz
    from PIL import Image

    doc = fitz.open(pdfPath)
    page = doc.load_page(pageIdx)
    pix = page.get_pixmap(dpi=dpi)
    imgData = pix.tobytes(output="jpg", jpg_quality=95)
    image = Image.open(io.BytesIO(imgData)).convert("RGB")
    doc.close()
    return image


def _imageToBase64(image) -> str:
    """PIL Image → base64 JPEG data URL。"""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _callVisionApi(
    client, model: str, imageB64: str, prompt: str, maxTokens: int, temperature: float
) -> str:
    """调用 OpenAI 兼容的 Vision Chat Completion 接口。"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{imageB64}"},
                    },
                ],
            }
        ],
        max_tokens=maxTokens,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def _collectPdfs() -> list[str]:
    """扫描 raw/ 目录，返回按文件名排序的 PDF 路径列表。"""
    if not os.path.isdir(config.RAW_DIR):
        return []
    return [
        os.path.join(config.RAW_DIR, name)
        for name in sorted(os.listdir(config.RAW_DIR))
        if name.lower().endswith(".pdf")
    ]


def _getPageCount(pdfPath: str) -> int | None:
    """获取 PDF 总页数。"""
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    try:
        return len(PdfReader(pdfPath).pages)
    except Exception:
        return None


def _processSinglePdf(pdfPath: str, pageStartOverride: int | None = None) -> None:
    """处理单个 PDF 文件（多模态 AI OCR 版本）。"""
    from openai import OpenAI

    ocrCfg = config.get_ocr_config()
    apiCfg = _getOcrApiConfig()

    pageStart = ocrCfg.get("page_start", 0)
    pageEnd = ocrCfg.get("page_end")
    skipExisting = ocrCfg.get("skip_existing", True)
    renderDpi = ocrCfg.get("render_dpi", 200)

    if pageStartOverride is not None:
        pageStart = max(0, pageStartOverride - 1)

    pdfName = os.path.basename(pdfPath)
    bookName = os.path.splitext(pdfName)[0]
    bookDir = os.path.join(config.OCR_DIR, bookName)
    pagesDir = os.path.join(bookDir, "pages")
    os.makedirs(pagesDir, exist_ok=True)

    totalPages = _getPageCount(pdfPath)
    if totalPages is None:
        print(f"无法读取 PDF 页数: {pdfPath}")
        return

    end = min(pageEnd if pageEnd is not None else totalPages - 1, totalPages - 1)
    if end < pageStart:
        print("未找到可处理的页码范围。")
        return

    apiKey = _resolveApiKey(apiCfg["api_key_env"])
    client = OpenAI(api_key=apiKey, base_url=apiCfg["api_base"])

    print(f"\n{'=' * 60}")
    print(f"[API OCR] 处理: {pdfName}  页范围 [{pageStart + 1}, {end + 1}]")
    print(f"  模型: {apiCfg['model']}  端点: {apiCfg['api_base']}")
    print(f"{'=' * 60}")

    t0 = time.time()
    processed = 0
    skipped = 0

    for pageIdx in range(pageStart, end + 1):
        pageNo = pageIdx + 1
        pageFile = os.path.join(pagesDir, f"page_{pageNo:04d}.md")

        if skipExisting and os.path.isfile(pageFile):
            skipped += 1
            continue

        try:
            image = _renderPageToImage(pdfPath, pageIdx, renderDpi)
            b64 = _imageToBase64(image)
            content = _callVisionApi(
                client,
                apiCfg["model"],
                b64,
                apiCfg["prompt"],
                int(apiCfg.get("max_tokens", 4000)),
                float(apiCfg.get("temperature", 0.1)),
            )
            if content.strip():
                with open(pageFile, "w", encoding="utf-8") as f:
                    f.write(f"<!-- page: {pageNo} -->\n\n")
                    f.write(content)
                processed += 1
                print(f"  页 {pageNo} 完成")
            else:
                print(f"  页 {pageNo} 返回为空，跳过")
        except Exception as e:
            print(f"  页 {pageNo} 失败: {e}")

    elapsed = time.time() - t0
    print(
        f"\n[API OCR] 完成: 处理 {processed} 页, 跳过 {skipped} 页, 耗时 {elapsed:.1f}s"
    )


def main() -> None:
    pageStartOverride = None
    pdfList: list[str]

    if len(sys.argv) > 1:
        pdfName = sys.argv[1]
        if not pdfName.lower().endswith(".pdf"):
            pdfName += ".pdf"
        pdfPath = os.path.join(config.RAW_DIR, pdfName)
        if not os.path.isfile(pdfPath):
            print(f"未找到 PDF: {pdfPath}")
            return
        pdfList = [pdfPath]
        if len(sys.argv) > 2:
            try:
                pageStartOverride = int(sys.argv[2])
            except ValueError:
                print(f"无效的页码参数: {sys.argv[2]}")
                return
    else:
        pdfList = _collectPdfs()
        if not pdfList:
            print(f"raw 目录下未找到 PDF: {config.RAW_DIR}")
            return

    for pdfPath in pdfList:
        _processSinglePdf(pdfPath, pageStartOverride)
        pageStartOverride = None

    print(f"\n全部 API OCR 完成，共 {len(pdfList)} 本书")


if __name__ == "__main__":
    main()
