#!/bin/bash

# 重啟後端服務（用於後端代碼更改後）

echo "=========================================="
echo "重啟後端服務"
echo "=========================================="

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
echo "正在重啟後端服務..."
$DOCKER_COMPOSE restart backend

echo ""
echo "=========================================="
echo "後端服務已重啟！"
echo "=========================================="
echo ""

