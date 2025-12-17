@echo off
echo 啟動印鑑比對系統...

REM 檢查 Docker 是否運行
docker info >nul 2>&1
if errorlevel 1 (
    echo 錯誤: Docker 未運行，請先啟動 Docker
    exit /b 1
)

REM 構建並啟動服務
docker-compose up -d --build

echo 服務已啟動！
echo 前端: http://localhost:3000
echo 後端 API: http://localhost:8000
echo API 文檔: http://localhost:8000/docs

pause

