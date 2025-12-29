@echo off
REM 查看服務日誌

if "%1"=="" (
    echo ==========================================
    echo 查看所有服務日誌
    echo ==========================================
    echo.
    echo 使用方式：
    echo   logs.bat              - 查看所有服務日誌
    echo   logs.bat backend      - 查看後端日誌
    echo   logs.bat frontend     - 查看前端日誌
    echo   logs.bat postgres     - 查看資料庫日誌
    echo   logs.bat redis        - 查看 Redis 日誌
    echo.
    docker compose logs -f
) else (
    echo ==========================================
    echo 查看 %1 服務日誌
    echo ==========================================
    echo.
    docker compose logs -f %1
)

