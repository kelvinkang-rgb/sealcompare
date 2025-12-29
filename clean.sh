#!/bin/bash

# 清理服務（停止並刪除容器、網絡，但保留數據卷）

echo "=========================================="
echo "清理服務"
echo "=========================================="
echo ""
echo "警告：此操作將停止並刪除所有容器和網絡"
echo "但會保留數據卷（資料庫數據不會丟失）"
echo ""

read -p "確定要繼續嗎？(Y/N): " confirm
if [ "$confirm" != "Y" ] && [ "$confirm" != "y" ]; then
    echo "操作已取消"
    exit 0
fi

# 檢查 Docker 是否運行
if ! docker info > /dev/null 2>&1; then
    echo "錯誤: Docker 未運行，請先啟動 Docker"
    exit 1
fi

# 使用 docker compose 或 docker-compose
if docker compose version > /dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

echo ""
echo "正在清理服務..."
$DOCKER_COMPOSE down

echo ""
echo "=========================================="
echo "清理完成！"
echo "=========================================="
echo ""

