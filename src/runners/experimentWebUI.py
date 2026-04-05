"""实验 WebUI 入口。"""

from __future__ import annotations

import argparse

from runners.experiments import experimentWebUI as webui


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Math-RAG 对比实验 WebUI")
    parser.add_argument("--port", type=int, default=7860, help="服务端口")
    parser.add_argument("--share", action="store_true", help="生成公网链接")
    args = parser.parse_args(argv)

    print("=" * 60)
    print(" Math-RAG 对比实验 WebUI")
    print("=" * 60)

    demo = webui.createUI()
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
