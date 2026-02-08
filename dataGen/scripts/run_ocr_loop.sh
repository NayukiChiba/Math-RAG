#!/bin/bash

echo "============================================================"
echo "Math-RAG OCR 自动重启脚本（pix2text_ocr.py）"
echo "============================================================"
echo ""

# 设置 conda 环境名称
CONDA_ENV="MathRag"

# 设置重启等待时间（秒）
WAIT_SECONDS=60

# 可选：指定书名和起始页码（留空则处理全部）
# 示例: BOOK_NAME="数学分析(第5版) 上 (华东师范大学数学系).pdf"
# 示例: START_PAGE=136
BOOK_NAME=""
START_PAGE=""

# 激活 conda 环境
echo "正在激活 conda 环境: $CONDA_ENV"

# 初始化 conda（如果需要）
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    eval "$(conda shell.bash hook 2>/dev/null)"
fi

conda activate "$CONDA_ENV"
if [ $? -ne 0 ]; then
    echo "错误: 无法激活 conda 环境 $CONDA_ENV"
    echo "请确保已安装 conda 并创建了该环境"
    read -p "按 Enter 键退出..."
    exit 1
fi
echo "conda 环境已激活"
echo ""

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动 pix2text OCR..."
    echo ""

    # 切换到 dataGen 目录（脚本位于 scripts/ 子目录下）
    cd "$(dirname "$0")/.."

    # 构建命令参数
    if [ -n "$BOOK_NAME" ]; then
        if [ -n "$START_PAGE" ]; then
            echo "处理指定书籍: $BOOK_NAME，起始页: $START_PAGE"
            python pix2text_ocr.py "$BOOK_NAME" "$START_PAGE"
        else
            echo "处理指定书籍: $BOOK_NAME"
            python pix2text_ocr.py "$BOOK_NAME"
        fi
    else
        echo "处理所有 PDF"
        python pix2text_ocr.py
    fi

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] OCR 全部完成！"
        break
    fi

    # 错误退出（包括内存不足、GPU 错误等）
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 进程异常退出 (错误码: $EXIT_CODE)，等待 $WAIT_SECONDS 秒后重启..."
    echo "已处理的页面会自动跳过，从断点继续。"
    sleep "$WAIT_SECONDS"
    echo ""

    # 重启后不再指定起始页码（依赖 skip_existing 自动跳过）
    START_PAGE=""
done

echo ""
echo "脚本结束。"
read -p "按 Enter 键退出..."
