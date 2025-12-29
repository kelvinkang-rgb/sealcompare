#!/bin/bash

# 重啟所有服務（不重新構建）

echo "=========================================="
echo "重啟印鑑比對系統服務"
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
echo "正在重啟服務..."
$DOCKER_COMPOSE restart

echo ""
echo "=========================================="
echo "服務已重啟！"
echo "=========================================="
echo ""
echo "服務訪問地址："
echo "  - 前端: http://localhost:3000"
echo "  - 後端 API: http://localhost:8000"
echo "  - API 文檔: http://localhost:8000/docs"
echo ""

