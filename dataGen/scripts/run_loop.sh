#!/bin/bash

echo "============================================================"
echo "Math-RAG Pipeline 自动重启脚本"
echo "============================================================"
echo ""

# 设置 conda 环境名称
CONDA_ENV="MathRag"

# 设置每次运行最大处理页数（根据内存情况调整，0=不限制）
MAX_PAGES=50

# 设置重启等待时间（秒）
WAIT_SECONDS=10

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
    # 尝试直接使用 conda 命令
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
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动 pipeline..."
    echo ""
    
    # 切换到 dataGen 目录（脚本位于 scripts/ 子目录下）
    cd "$(dirname "$0")/.."
    
    python -m pipeline.run --max-pages "$MAX_PAGES"
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 所有任务已完成！"
        break
    fi
    
    if [ $EXIT_CODE -eq 2 ]; then
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 达到页数限制，等待 $WAIT_SECONDS 秒后重启以释放内存..."
        sleep "$WAIT_SECONDS"
        echo ""
        continue
    fi
    
    # 其他错误（包括内存不足导致的崩溃）
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 进程异常退出 (错误码: $EXIT_CODE)，等待 $WAIT_SECONDS 秒后重启..."
    sleep "$WAIT_SECONDS"
    echo ""
done

echo ""
echo "脚本结束。"
read -p "按 Enter 键退出..."
