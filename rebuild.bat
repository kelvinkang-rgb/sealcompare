@echo off
REM 重新構建並啟動服務（用於代碼更改後）

echo ==========================================
echo 重新構建並啟動服務
echo ==========================================

REM 檢查 Docker 是否運行
docker info >nul 2>&1
if errorlevel 1 (
    echo 錯誤: Docker 未運行，請先啟動 Docker Desktop
    pause
    exit /b 1
)

echo.
echo 1. 停止現有服務...
docker compose down

echo.
echo 2. 重新構建並啟動服務...
docker compose up -d --build

echo.
echo 3. 等待服務啟動...
timeout /t 10 /nobreak >nul

echo.
echo 4. 檢查服務狀態...
docker compose ps

echo.
echo ==========================================
echo 重新構建完成！
echo ==========================================
echo.
echo 服務訪問地址：
echo   - 前端: http://localhost:3000
echo   - 後端 API: http://localhost:8000
echo   - API 文檔: http://localhost:8000/docs
echo.
pause

