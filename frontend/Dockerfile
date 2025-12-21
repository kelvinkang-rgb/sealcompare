FROM node:18-alpine as build

WORKDIR /app

# 複製 package 文件
COPY package*.json ./

# 安裝依賴
RUN npm install

# 複製源代碼
COPY . .

# 設置構建時環境變數（如果未提供，使用默認值）
ARG VITE_API_BASE_URL=/api/v1
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

# 構建應用
RUN npm run build

# 生產環境
FROM nginx:alpine

# 複製構建結果
COPY --from=build /app/dist /usr/share/nginx/html

# 複製 nginx 配置
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]

