# 使用官方 Python 運行時作為基礎映像
FROM python:3.11-slim

# 設置工作目錄
WORKDIR /app

# 安裝系統依賴（OpenCV 需要）以及中文字體
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    fonts-wqy-zenhei \
    fonts-wqy-microhei \
    && rm -rf /var/lib/apt/lists/*

# 複製依賴文件
COPY requirements.txt .

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式代碼
COPY . .

# 設置環境變數
ENV PYTHONUNBUFFERED=1

# 設置預設入口點為 main.py，允許在 Docker Desktop UI 中通過參數覆蓋
# 在 Docker Desktop UI 中，只需在參數欄位輸入圖像路徑即可
# 例如：test_images/seal_original_1.jpg test_images/seal_similar.jpg --verbose
ENTRYPOINT ["python", "main.py"]
