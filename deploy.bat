@echo off
REM 印鑑比對系統部署腳本 (Windows 批處理)

setlocal enabledelayedexpansion

echo ==========================================
echo 開始部署印鑑比對系統
echo ==========================================
echo.

REM 檢查 Docker 是否安裝
echo 檢查 Docker...
where docker >nul 2>&1
if errorlevel 1 (
    echo 錯誤: Docker 未安裝，請先安裝 Docker Desktop
    exit /b 1
)
docker --version
if errorlevel 1 (
    echo 錯誤: Docker 未正確安裝
    exit /b 1
)
echo   ✓ Docker 已安裝
echo.

REM 檢查 Docker Compose 是否安裝
echo 檢查 Docker Compose...
set DOCKER_COMPOSE=
docker compose version >nul 2>&1
if errorlevel 1 (
    docker-compose --version >nul 2>&1
    if errorlevel 1 (
        echo 錯誤: Docker Compose 未安裝，請先安裝 Docker Compose
        exit /b 1
    ) else (
        set DOCKER_COMPOSE=docker-compose
        echo   ✓ Docker Compose (standalone) 已安裝
    )
) else (
    set DOCKER_COMPOSE=docker compose
    echo   ✓ Docker Compose (plugin) 已安裝
)
echo.

echo 1. 停止現有服務...
%DOCKER_COMPOSE% down
if errorlevel 1 (
    echo   警告: 停止服務時發生錯誤（可能是服務尚未啟動）
)

echo.
echo 2. 構建並啟動服務...
%DOCKER_COMPOSE% up -d --build
if errorlevel 1 (
    echo 錯誤: 構建或啟動服務失敗
    exit /b 1
)

echo.
echo 3. 等待服務啟動...
timeout /t 10 /nobreak >nul

echo.
echo 4. 檢查服務狀態...
%DOCKER_COMPOSE% ps

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
echo   %DOCKER_COMPOSE% logs -f
echo.
echo 停止服務：
echo   %DOCKER_COMPOSE% down
echo.

endlocal

