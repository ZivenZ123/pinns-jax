#!/bin/bash

# PINNs-JAX MCP Server 安装脚本

echo "🚀 开始设置 PINNs-JAX MCP Server..."

# 检查Node.js是否安装
if ! command -v node &> /dev/null; then
    echo "❌ 错误: 未找到Node.js，请先安装Node.js 18+"
    exit 1
fi

# 检查npm是否安装
if ! command -v npm &> /dev/null; then
    echo "❌ 错误: 未找到npm"
    exit 1
fi

# 创建MCP Server目录
echo "📁 创建MCP Server目录..."
mkdir -p mcp_server
cd mcp_server

# 初始化npm项目（如果package.json不存在）
if [ ! -f "package.json" ]; then
    echo "📦 初始化npm项目..."
    npm init -y
fi

# 安装依赖
echo "📦 安装依赖..."
npm install @modelcontextprotocol/sdk zod
npm install --save-dev @types/node typescript

# 创建目录结构
echo "📁 创建目录结构..."
mkdir -p src/tools src/resources src/prompts build

# 编译TypeScript
echo "🔨 编译TypeScript..."
npm run build

# 检查编译是否成功
if [ $? -eq 0 ]; then
    echo "✅ MCP Server 设置完成！"
    echo ""
    echo "📋 下一步："
    echo "1. 在AI客户端中配置MCP服务器"
    echo "2. 使用 'npm run inspector' 测试服务器"
    echo "3. 开始使用PINN功能！"
    echo ""
    echo "🔧 配置示例 (Claude Desktop):"
    echo "{"
    echo "  \"mcpServers\": {"
    echo "    \"pinnsjax\": {"
    echo "      \"command\": \"node\","
    echo "      \"args\": [\"$(pwd)/build/index.js\"],"
    echo "      \"cwd\": \"$(dirname $(pwd))\""
    echo "    }"
    echo "  }"
    echo "}"
else
    echo "❌ 编译失败，请检查错误信息"
    exit 1
fi 