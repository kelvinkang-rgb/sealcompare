@echo off
REM 重新構建並重啟前端服務（用於前端代碼或配置更改後）

echo ==========================================
echo 重新構建並重啟前端服務
echo ==========================================

REM 檢查 Docker 是否運行
docker info >nul 2>&1
if errorlevel 1 (
    echo 錯誤: Docker 未運行，請先啟動 Docker Desktop
    pause
    exit /b 1
)

echo.
echo 正在重新構建前端服務...
docker compose up -d --build frontend

echo.
echo ==========================================
echo 前端服務重新構建完成！
echo ==========================================
echo.
pause

