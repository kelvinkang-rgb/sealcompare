#!/bin/bash

# 重新構建並啟動服務（用於代碼更改後）

echo "=========================================="
echo "重新構建並啟動服務"
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
echo "1. 停止現有服務..."
$DOCKER_COMPOSE down

echo ""
echo "2. 重新構建並啟動服務..."
$DOCKER_COMPOSE up -d --build

echo ""
echo "3. 等待服務啟動..."
sleep 10

echo ""
echo "4. 檢查服務狀態..."
$DOCKER_COMPOSE ps

echo ""
echo "=========================================="
echo "重新構建完成！"
echo "=========================================="
echo ""
echo "服務訪問地址："
echo "  - 前端: http://localhost:3000"
echo "  - 後端 API: http://localhost:8000"
echo "  - API 文檔: http://localhost:8000/docs"
echo ""

