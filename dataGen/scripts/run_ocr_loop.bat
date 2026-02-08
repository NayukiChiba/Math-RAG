@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ============================================================
echo Math-RAG OCR 自动重启脚本（pix2text_ocr.py）
echo ============================================================
echo.

REM 设置 conda 环境名称
set CONDA_ENV=MathRag

REM 设置重启等待时间（秒）
set WAIT_SECONDS=60

REM 可选：指定书名和起始页码（留空则处理全部）
REM 示例: set BOOK_NAME="数学分析(第5版) 上 (华东师范大学数学系).pdf"
REM 示例: set START_PAGE=136
set BOOK_NAME=
set START_PAGE=

REM 激活 conda 环境
echo 正在激活 conda 环境: %CONDA_ENV%
call conda activate %CONDA_ENV%
if errorlevel 1 (
    echo 错误: 无法激活 conda 环境 %CONDA_ENV%
    echo 请确保已安装 conda 并创建了该环境
    pause
    exit /b 1
)
echo conda 环境已激活
echo.

:loop
echo [%date% %time%] 启动 pix2text OCR...
echo.

REM 切换到 dataGen 目录（脚本位于 scripts\ 子目录下）
cd /d %~dp0\..

REM 构建命令参数
if defined BOOK_NAME (
    if defined START_PAGE (
        echo 处理指定书籍: %BOOK_NAME%，起始页: %START_PAGE%
        python pix2text_ocr.py %BOOK_NAME% %START_PAGE%
    ) else (
        echo 处理指定书籍: %BOOK_NAME%
        python pix2text_ocr.py %BOOK_NAME%
    )
) else (
    echo 处理所有 PDF
    python pix2text_ocr.py
)

set EXIT_CODE=%ERRORLEVEL%

if %EXIT_CODE%==0 (
    echo.
    echo [%date% %time%] OCR 全部完成！
    goto end
)

REM 错误退出（包括内存不足、GPU 错误等）
echo.
echo [%date% %time%] 进程异常退出 (错误码: %EXIT_CODE%)，等待 %WAIT_SECONDS% 秒后重启...
echo 已处理的页面会自动跳过，从断点继续。
timeout /t %WAIT_SECONDS% /nobreak
echo.

REM 重启后不再指定起始页码（依赖 skip_existing 自动跳过）
set START_PAGE=
goto loop

:end
echo.
echo 脚本结束。
pause
