#!/bin/bash

# 查看服務日誌

# 使用 docker compose 或 docker-compose
if docker compose version > /dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

if [ -z "$1" ]; then
    echo "=========================================="
    echo "查看所有服務日誌"
    echo "=========================================="
    echo ""
    echo "使用方式："
    echo "  ./logs.sh              - 查看所有服務日誌"
    echo "  ./logs.sh backend      - 查看後端日誌"
    echo "  ./logs.sh frontend     - 查看前端日誌"
    echo "  ./logs.sh postgres     - 查看資料庫日誌"
    echo "  ./logs.sh redis        - 查看 Redis 日誌"
    echo ""
    $DOCKER_COMPOSE logs -f
else
    echo "=========================================="
    echo "查看 $1 服務日誌"
    echo "=========================================="
    echo ""
    $DOCKER_COMPOSE logs -f "$1"
fi

