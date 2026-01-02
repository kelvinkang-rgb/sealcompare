#!/bin/bash

################################################################################
# 印鑑比對系統 - 萬用部署腳本
#
# 版本: 2.0
# 作者: SealCompare Team
# 說明: 支援多種部署操作（init/deploy/rebuild/restart/stop/status）
#
# 使用方式:
#   ./deploy.sh [command]        - 執行指定命令
#   ./deploy.sh                  - 顯示互動式選單
#   ./deploy.sh --help           - 顯示幫助訊息
#
# 支援命令:
#   init     - 初始化部署（首次使用，含資料庫遷移）
#   deploy   - 快速更新（僅重啟服務，不重建）
#   rebuild  - 完全重建（重建 Docker images，保留資料）
#   restart  - 重啟所有服務
#   stop     - 停止所有服務
#   status   - 查看服務狀態
################################################################################

set -e  # 遇到錯誤立即退出

# ============================================================================
# 全域變數
# ============================================================================

SCRIPT_VERSION="2.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_COMPOSE=""

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ============================================================================
# 工具函數 - 顏色輸出
# ============================================================================

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

header() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
}

# ============================================================================
# 工具函數 - 環境檢查
# ============================================================================

check_docker() {
    if ! command -v docker &> /dev/null; then
        error "Docker 未安裝，請先安裝 Docker"
        echo "  安裝指南: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # 檢查 Docker daemon 是否運行
    if ! docker info &> /dev/null; then
        error "Docker daemon 未運行，請啟動 Docker"
        exit 1
    fi
    
    success "Docker 已就緒"
}

check_docker_compose() {
    # 優先使用 docker compose (plugin)
    if docker compose version &> /dev/null; then
        DOCKER_COMPOSE="docker compose"
    elif command -v docker-compose &> /dev/null; then
        DOCKER_COMPOSE="docker-compose"
    else
        error "Docker Compose 未安裝，請先安裝 Docker Compose"
        echo "  安裝指南: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    success "Docker Compose 已就緒 (使用: $DOCKER_COMPOSE)"
}

check_compose_file() {
    if [ ! -f "$SCRIPT_DIR/docker-compose.yml" ]; then
        error "找不到 docker-compose.yml 文件"
        exit 1
    fi
}

check_environment() {
    info "檢查環境..."
    check_docker
    check_docker_compose
    check_compose_file
    echo ""
}

# ============================================================================
# 工具函數 - 服務管理
# ============================================================================

wait_for_service() {
    local service=$1
    local max_attempts=30
    local attempt=1
    
    info "等待服務 $service 就緒..."
    
    while [ $attempt -le $max_attempts ]; do
        if $DOCKER_COMPOSE ps | grep -q "$service.*running"; then
            # 額外檢查健康狀態
            local health=$($DOCKER_COMPOSE ps | grep "$service" | grep -o "healthy\|unhealthy\|starting" || echo "unknown")
            
            if [ "$health" = "healthy" ]; then
                success "服務 $service 已就緒"
                return 0
            elif [ "$health" = "unhealthy" ]; then
                error "服務 $service 健康檢查失敗"
                return 1
            fi
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo ""
    warning "服務 $service 等待超時（可能仍在啟動中）"
    return 1
}

show_service_info() {
    header "服務資訊"
    
    echo -e "${CYAN}服務訪問地址：${NC}"
    echo "  - 前端:      http://localhost:3000"
    echo "  - 後端 API:  http://localhost:8000"
    echo "  - API 文檔:  http://localhost:8000/docs"
    echo ""
    
    echo -e "${CYAN}常用命令：${NC}"
    echo "  - 查看日誌:  $DOCKER_COMPOSE logs -f"
    echo "  - 查看狀態:  ./deploy.sh status"
    echo "  - 停止服務:  ./deploy.sh stop"
    echo ""
}

# ============================================================================
# 命令實作 - init（初始化部署）
# ============================================================================

cmd_init() {
    header "初始化部署"
    
    # 檢查是否已有運行的服務
    if $DOCKER_COMPOSE ps | grep -q "Up"; then
        warning "檢測到運行中的服務"
        read -p "是否要停止現有服務並重新初始化? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            info "取消初始化"
            exit 0
        fi
        
        info "停止現有服務..."
        $DOCKER_COMPOSE down
        echo ""
    fi
    
    info "構建並啟動所有服務..."
    $DOCKER_COMPOSE up -d --build
    echo ""
    
    info "等待服務健康檢查..."
    wait_for_service "postgres"
    wait_for_service "redis"
    wait_for_service "backend"
    wait_for_service "frontend"
    echo ""
    
    info "執行資料庫遷移..."
    if $DOCKER_COMPOSE exec -T backend alembic upgrade head 2>/dev/null; then
        success "資料庫遷移完成"
    else
        warning "資料庫遷移失敗或已是最新版本"
    fi
    echo ""
    
    success "初始化完成！"
    show_service_info
}

# ============================================================================
# 命令實作 - deploy（快速更新）
# ============================================================================

cmd_deploy() {
    header "快速更新部署"
    
    info "檢查服務狀態..."
    if ! $DOCKER_COMPOSE ps | grep -q "Up"; then
        warning "服務未運行，建議使用 'init' 命令初始化"
        read -p "是否要啟動服務? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cmd_init
            return
        else
            exit 0
        fi
    fi
    echo ""
    
    info "重啟 backend 和 frontend 服務..."
    $DOCKER_COMPOSE restart backend frontend
    echo ""
    
    info "等待服務就緒..."
    sleep 5
    echo ""
    
    success "更新完成！"
    show_service_info
}

# ============================================================================
# 命令實作 - rebuild（完全重建）
# ============================================================================

cmd_rebuild() {
    header "完全重建"
    
    warning "此操作將重建所有 Docker images（約需 2-5 分鐘）"
    warning "資料庫和上傳檔案將會保留"
    read -p "確定要繼續? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        info "取消重建"
        exit 0
    fi
    echo ""
    
    info "停止所有服務..."
    $DOCKER_COMPOSE down
    echo ""
    
    info "重建並啟動服務（保留 volumes）..."
    $DOCKER_COMPOSE up -d --build --force-recreate
    echo ""
    
    info "等待服務健康檢查..."
    wait_for_service "postgres"
    wait_for_service "redis"
    wait_for_service "backend"
    wait_for_service "frontend"
    echo ""
    
    success "重建完成！"
    show_service_info
}

# ============================================================================
# 命令實作 - restart（重啟服務）
# ============================================================================

cmd_restart() {
    header "重啟所有服務"
    
    info "重啟服務..."
    $DOCKER_COMPOSE restart
    echo ""
    
    info "等待服務就緒..."
    sleep 5
    echo ""
    
    success "重啟完成！"
    show_service_info
}

# ============================================================================
# 命令實作 - stop（停止服務）
# ============================================================================

cmd_stop() {
    header "停止服務"
    
    info "停止所有容器..."
    $DOCKER_COMPOSE stop
    echo ""
    
    success "所有服務已停止"
    echo ""
    info "提示："
    echo "  - 容器、volumes 和 images 已保留"
    echo "  - 重新啟動: ./deploy.sh restart"
    echo "  - 完全清理: docker compose down -v (會刪除資料)"
    echo ""
}

# ============================================================================
# 命令實作 - status（查看狀態）
# ============================================================================

cmd_status() {
    header "服務狀態"
    
    echo -e "${CYAN}容器狀態：${NC}"
    $DOCKER_COMPOSE ps
    echo ""
    
    echo -e "${CYAN}資源使用：${NC}"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" \
        $(docker ps --filter "name=sealcompare-" --format "{{.Names}}") 2>/dev/null || \
        warning "無法獲取資源統計（服務可能未運行）"
    echo ""
    
    show_service_info
}

# ============================================================================
# 互動式選單
# ============================================================================

show_menu() {
    clear
    header "印鑑比對系統部署工具 v${SCRIPT_VERSION}"
    
    echo "請選擇操作："
    echo ""
    echo "  1) init     - 初始化部署（首次使用，含資料庫遷移）"
    echo "  2) deploy   - 快速更新（僅重啟服務，不重建）"
    echo "  3) rebuild  - 完全重建（重建 Docker images，保留資料）"
    echo "  4) restart  - 重啟所有服務"
    echo "  5) stop     - 停止所有服務"
    echo "  6) status   - 查看服務狀態"
    echo "  7) exit     - 退出"
    echo ""
    echo -n "請輸入選項 [1-7]: "
    
    read -r choice
    echo ""
    
    case $choice in
        1) cmd_init ;;
        2) cmd_deploy ;;
        3) cmd_rebuild ;;
        4) cmd_restart ;;
        5) cmd_stop ;;
        6) cmd_status ;;
        7) info "退出"; exit 0 ;;
        *) error "無效選項: $choice"; exit 1 ;;
    esac
}

# ============================================================================
# 幫助訊息
# ============================================================================

show_help() {
    cat << EOF
印鑑比對系統部署工具 v${SCRIPT_VERSION}

使用方式:
  ./deploy.sh [command]        - 執行指定命令
  ./deploy.sh                  - 顯示互動式選單
  ./deploy.sh --help           - 顯示此幫助訊息

支援命令:
  init       初始化部署（首次使用，含資料庫遷移）
  deploy     快速更新（僅重啟服務，不重建）
  rebuild    完全重建（重建 Docker images，保留資料）
  restart    重啟所有服務
  stop       停止所有服務
  status     查看服務狀態

範例:
  ./deploy.sh init              # 初始化系統
  ./deploy.sh deploy            # 更新程式碼後快速部署
  ./deploy.sh rebuild           # 修改 Dockerfile 後重建
  ./deploy.sh status            # 查看服務狀態

注意事項:
  - init 和 rebuild 會保留資料庫和上傳檔案
  - deploy 適合小幅度程式碼更新（最快）
  - rebuild 適合 Dockerfile 或依賴變更（較慢）
  - 完全清理請使用: docker compose down -v（會刪除所有資料）

EOF
}

# ============================================================================
# 主程式
# ============================================================================

main() {
    # 解析命令列參數
    local command="${1:-}"
    
    # 顯示幫助
    if [ "$command" = "--help" ] || [ "$command" = "-h" ]; then
        show_help
        exit 0
    fi
    
    # 檢查環境（除了 help 和 status 以外都需要）
    if [ "$command" != "status" ]; then
        check_environment
    fi
    
    # 切換到腳本目錄（確保 docker-compose.yml 可以被找到）
    cd "$SCRIPT_DIR"
    
    # 命令分發
    case "$command" in
        "")
            # 無參數，顯示互動式選單
            show_menu
            ;;
        init)
            cmd_init
            ;;
        deploy)
            cmd_deploy
            ;;
        rebuild)
            cmd_rebuild
            ;;
        restart)
            cmd_restart
            ;;
        stop)
            cmd_stop
            ;;
        status)
            check_environment
            cmd_status
            ;;
        *)
            error "未知命令: $command"
            echo ""
            echo "使用 './deploy.sh --help' 查看可用命令"
            exit 1
            ;;
    esac
}

# 執行主程式
main "$@"
