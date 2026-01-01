# 快速開始（Makefile 唯一入口）

本專案以 **`make`** 作為唯一操作入口（啟動、日誌、測試）。請勿直接使用零散腳本或手動拼 docker compose 指令，避免與專案設定漂移。

## 1) 啟動

```bash
make up
```

## 2) 開啟 UI（唯一主入口）

- `http://localhost:3000/multi-seal-test`

## 3) 常用指令

```bash
make logs
make test-backend
make test-e2e
make down
```

## 4) 後端健康檢查與 API 文件

- `http://localhost:8000/health`
- `http://localhost:8000/docs`


