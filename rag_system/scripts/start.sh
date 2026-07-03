#!/bin/bash
set e

echo "🚀 使用本地Ollama启动RAG系统"
echo "================================"

# 1. 检查Ollama
echo "1️⃣ 检查Ollama状态..."
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "⚠️ Ollama未运行，正在启动..."
    ollama serve &
    sleep 5
fi

# 2. 检查模型
echo "2️⃣ 检查模型..."
if ! ollama list | grep -q "qwen2.5:7b"; then
    echo "📥 下载 qwen2.5:7b..."
    ollama pull qwen2.5:7b
fi

if ! ollama list | grep -q "nomic-embed-text-v2-moe"; then
    echo "📥 下载 nomic-embed-text-v2-moe..."
    ollama pull nomic-embed-text-v2-moe
fi
# 3.检查Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker未安装，请先安装Docker Desktop"
    exit 1
fi

if ! command -v docker compose &> /dev/null; then
    echo "❌ docker compose未安装"
    exit 1
fi
# 创建必要目录
mkdir -p RAG_files chroma_db logs eval_results
# 4.检查是否有文档
if [ -z "$(ls -A RAG_files 2>/dev/null)" ]; then
    echo "⚠️  RAG_files目录为空，请放入文档后再使用"
fi
# 5. 构建并启动Docker
echo "3️⃣ 构建Docker镜像..."
docker compose build

echo "4️⃣ 启动服务..."
docker compose up -d

# 4. 等待服务就绪
echo "5️⃣ 等待服务就绪..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ 服务已就绪!"
        break
    fi
    echo "   等待中... ($i/30)"
    sleep 2
done

echo ""
echo "============================================================"
echo "  ✅ 部署完成!"
echo "  🌐 Web界面: http://localhost:8000"
echo "  📚 API文档: http://localhost:8000/docs"
echo "  🔍 Ollama:  http://localhost:11434"
echo "============================================================"
