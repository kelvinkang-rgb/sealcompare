@echo off
REM 重啟後端服務（用於後端代碼更改後）

echo ==========================================
echo 重啟後端服務
echo ==========================================

REM 檢查 Docker 是否運行
docker info >nul 2>&1
if errorlevel 1 (
    echo 錯誤: Docker 未運行，請先啟動 Docker Desktop
    pause
    exit /b 1
)

echo.
echo 正在重啟後端服務...
docker compose restart backend

echo.
echo ==========================================
echo 後端服務已重啟！
echo ==========================================
echo.
pause

