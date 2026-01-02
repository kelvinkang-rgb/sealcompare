# 印鑑比對系統部署腳本 (Windows PowerShell)

$ErrorActionPreference = "Stop"

Write-Host "=========================================="
Write-Host "開始部署印鑑比對系統"
Write-Host "=========================================="
Write-Host ""

# 檢查 Docker 是否安裝
Write-Host "檢查 Docker..."
try {
    $dockerVersion = docker --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Docker 未安裝"
    }
    Write-Host "  ✓ Docker 已安裝: $dockerVersion"
} catch {
    Write-Host "錯誤: Docker 未安裝，請先安裝 Docker Desktop" -ForegroundColor Red
    exit 1
}

# 檢查 Docker Compose 是否安裝
Write-Host "檢查 Docker Compose..."
$dockerCompose = $null
try {
    $composeVersion = docker compose version 2>&1
    if ($LASTEXITCODE -eq 0) {
        $dockerCompose = "docker compose"
        Write-Host "  ✓ Docker Compose (plugin) 已安裝: $composeVersion"
    } else {
        throw "Docker Compose plugin 未找到"
    }
} catch {
    try {
        $composeVersion = docker-compose --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $dockerCompose = "docker-compose"
            Write-Host "  ✓ Docker Compose (standalone) 已安裝: $composeVersion"
        } else {
            throw "Docker Compose 未安裝"
        }
    } catch {
        Write-Host "錯誤: Docker Compose 未安裝，請先安裝 Docker Compose" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "1. 停止現有服務..."
& $dockerCompose.Split(' ') down
if ($LASTEXITCODE -ne 0) {
    Write-Host "  警告: 停止服務時發生錯誤（可能是服務尚未啟動）" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "2. 構建並啟動服務..."
& $dockerCompose.Split(' ') up -d --build
if ($LASTEXITCODE -ne 0) {
    Write-Host "錯誤: 構建或啟動服務失敗" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "3. 等待服務啟動..."
Start-Sleep -Seconds 10

Write-Host ""
Write-Host "4. 檢查服務狀態..."
& $dockerCompose.Split(' ') ps

Write-Host ""
Write-Host "=========================================="
Write-Host "部署完成！" -ForegroundColor Green
Write-Host "=========================================="
Write-Host ""
Write-Host "服務訪問地址："
Write-Host "  - 前端: http://localhost:3000"
Write-Host "  - 後端 API: http://localhost:8000"
Write-Host "  - API 文檔: http://localhost:8000/docs"
Write-Host ""
Write-Host "查看日誌："
Write-Host "  $dockerCompose logs -f"
Write-Host ""
Write-Host "停止服務："
Write-Host "  $dockerCompose down"
Write-Host ""

