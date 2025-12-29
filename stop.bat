@echo off
REM 停止所有服務

echo ==========================================
echo 停止印鑑比對系統服務
echo ==========================================

REM 檢查 Docker 是否運行
docker info >nul 2>&1
if errorlevel 1 (
    echo 錯誤: Docker 未運行，請先啟動 Docker Desktop
    pause
    exit /b 1
)

echo.
echo 正在停止服務...
docker compose stop

echo.
echo ==========================================
echo 服務已停止！
echo ==========================================
echo.
pause

