@echo off
chcp 65001 >nul
title AIGC检测器

echo ========================================
echo    AIGC检测器 - 一键启动
echo ========================================
echo.

cd /d "%~dp0"

REM 检查虚拟环境是否存在
if not exist "venv_web\Scripts\python.exe" (
    echo [1/3] 创建虚拟环境...
    python -m venv venv_web
    if errorlevel 1 (
        echo 错误: 虚拟环境创建失败
        pause
        exit /b 1
    )
)

REM 安装依赖
echo [2/3] 安装依赖（首次可能需要几分钟）...
call venv_web\Scripts\pip.exe install -r requirements_web.txt -q
if errorlevel 1 (
    echo 错误: 依赖安装失败
    pause
    exit /b 1
)

REM 清理旧进程
echo [3/3] 启动服务...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5000 ^| findstr LISTENING') do (
    echo 关闭旧进程 %%a ...
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5173 ^| findstr LISTENING') do (
    echo 关闭旧进程 %%a ...
    taskkill /F /PID %%a >nul 2>&1
)

REM 启动后端
start "AIGC后端" cmd /c "venv_web\Scripts\python.exe app.py"

REM 等待后端启动
echo 等待后端启动...
timeout /t 10 /nobreak >nul

REM 检查后端是否启动成功
curl -s http://localhost:5000/api/system-info >nul 2>&1
if errorlevel 1 (
    echo 警告: 后端可能未正常启动，请检查日志
) else (
    echo ✓ 后端已启动 (http://localhost:5000)
)

REM 启动前端
cd frontend
if not exist "node_modules" (
    echo 安装前端依赖...
    call npm install
)
start "AIGC前端" cmd /c "npm run dev"
cd ..

echo.
echo ========================================
echo ✓ 启动完成！
echo ========================================
echo.
echo 访问地址:
echo   前端: http://localhost:5173
echo   后端: http://localhost:5000
echo.
echo 按任意键退出此窗口（服务继续在后台运行）...
pause >nul
