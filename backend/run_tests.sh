#!/bin/bash
# 運行測試腳本

echo "運行單元測試..."
pytest tests/ -v --cov=app --cov=core --cov-report=term-missing --cov-report=html

echo "測試完成！覆蓋率報告已生成在 htmlcov/ 目錄"

