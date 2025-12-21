@echo off
REM 印鑑比對系統部署腳本 (Windows)

echo ==========================================
echo 開始部署印鑑比對系統
echo ==========================================

REM 檢查 Docker 是否安裝
where docker >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo 錯誤: Docker 未安裝，請先安裝 Docker Desktop
    exit /b 1
)

echo.
echo 1. 停止現有服務...
docker-compose down

echo.
echo 2. 構建並啟動服務...
docker-compose up -d --build

echo.
echo 3. 等待服務啟動...
timeout /t 10 /nobreak >nul

echo.
echo 4. 檢查服務狀態...
docker-compose ps

echo.
echo ==========================================
echo 部署完成！
echo ==========================================
echo.
echo 服務訪問地址：
echo   - 前端: http://localhost:3000
echo   - 後端 API: http://localhost:8000
echo   - API 文檔: http://localhost:8000/docs
echo.
echo 查看日誌：
echo   docker-compose logs -f
echo.
echo 停止服務：
echo   docker-compose down
echo.

pause

