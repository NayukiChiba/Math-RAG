@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ============================================================
echo Math-RAG Pipeline 自动重启脚本
echo ============================================================
echo.

REM 设置 conda 环境名称
set CONDA_ENV=MathRag

REM 设置每次运行最大处理页数（根据内存情况调整，0=不限制）
set MAX_PAGES=10

REM 设置重启等待时间（秒）
set WAIT_SECONDS=60

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
echo [%date% %time%] 启动 pipeline...
echo.

REM 切换到 dataGen 目录（脚本位于 scripts\ 子目录下）
cd /d %~dp0\..
python -m pipeline.run --max-pages %MAX_PAGES%

set EXIT_CODE=%ERRORLEVEL%

if %EXIT_CODE%==0 (
    echo.
    echo [%date% %time%] 所有任务已完成！
    goto end
)

if %EXIT_CODE%==2 (
    echo.
    echo [%date% %time%] 达到页数限制，等待 %WAIT_SECONDS% 秒后重启以释放内存...
    timeout /t %WAIT_SECONDS% /nobreak
    echo.
    goto loop
)

REM 其他错误（包括内存不足导致的崩溃）
echo.
echo [%date% %time%] 进程异常退出 (错误码: %EXIT_CODE%)，等待 %WAIT_SECONDS% 秒后重启...
timeout /t %WAIT_SECONDS% /nobreak
echo.
goto loop

:end
echo.
echo 脚本结束。
pause
