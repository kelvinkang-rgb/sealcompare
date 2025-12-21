#!/bin/bash

# 印鑑比對系統部署腳本

set -e

echo "=========================================="
echo "開始部署印鑑比對系統"
echo "=========================================="

# 檢查 Docker 是否安裝
if ! command -v docker &> /dev/null; then
    echo "錯誤: Docker 未安裝，請先安裝 Docker"
    exit 1
fi

# 檢查 Docker Compose 是否安裝
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "錯誤: Docker Compose 未安裝，請先安裝 Docker Compose"
    exit 1
fi

# 使用 docker compose 或 docker-compose
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

echo ""
echo "1. 停止現有服務..."
$DOCKER_COMPOSE down

echo ""
echo "2. 構建並啟動服務..."
$DOCKER_COMPOSE up -d --build

echo ""
echo "3. 等待服務啟動..."
sleep 10

echo ""
echo "4. 檢查服務狀態..."
$DOCKER_COMPOSE ps

echo ""
echo "=========================================="
echo "部署完成！"
echo "=========================================="
echo ""
echo "服務訪問地址："
echo "  - 前端: http://localhost:3000"
echo "  - 後端 API: http://localhost:8000"
echo "  - API 文檔: http://localhost:8000/docs"
echo ""
echo "查看日誌："
echo "  $DOCKER_COMPOSE logs -f"
echo ""
echo "停止服務："
echo "  $DOCKER_COMPOSE down"
echo ""

