#!/bin/bash

# 查看服務狀態

# 使用 docker compose 或 docker-compose
if docker compose version > /dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

echo "=========================================="
echo "服務狀態"
echo "=========================================="
echo ""

$DOCKER_COMPOSE ps

echo ""
echo "=========================================="
echo ""

