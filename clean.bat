@echo off
REM 清理服務（停止並刪除容器、網絡，但保留數據卷）

echo ==========================================
echo 清理服務
echo ==========================================
echo.
echo 警告：此操作將停止並刪除所有容器和網絡
echo 但會保留數據卷（資料庫數據不會丟失）
echo.
set /p confirm="確定要繼續嗎？(Y/N): "
if /i not "%confirm%"=="Y" (
    echo 操作已取消
    pause
    exit /b 0
)

REM 檢查 Docker 是否運行
docker info >nul 2>&1
if errorlevel 1 (
    echo 錯誤: Docker 未運行，請先啟動 Docker Desktop
    pause
    exit /b 1
)

echo.
echo 正在清理服務...
docker compose down

echo.
echo ==========================================
echo 清理完成！
echo ==========================================
echo.
pause

