@echo off
REM 重啟所有服務（不重新構建）

echo ==========================================
echo 重啟印鑑比對系統服務
echo ==========================================

REM 檢查 Docker 是否運行
docker info >nul 2>&1
if errorlevel 1 (
    echo 錯誤: Docker 未運行，請先啟動 Docker Desktop
    pause
    exit /b 1
)

echo.
echo 正在重啟服務...
docker compose restart

echo.
echo ==========================================
echo 服務已重啟！
echo ==========================================
echo.
echo 服務訪問地址：
echo   - 前端: http://localhost:3000
echo   - 後端 API: http://localhost:8000
echo   - API 文檔: http://localhost:8000/docs
echo.
pause

