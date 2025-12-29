#!/bin/bash

# 停止所有服務

echo "=========================================="
echo "停止印鑑比對系統服務"
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
echo "正在停止服務..."
$DOCKER_COMPOSE stop

echo ""
echo "=========================================="
echo "服務已停止！"
echo "=========================================="
echo ""

