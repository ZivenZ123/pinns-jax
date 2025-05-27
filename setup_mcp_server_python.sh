#!/bin/bash

# PINNs-JAX MCP Server (Python版本) 安装脚本

echo "🚀 开始设置 PINNs-JAX MCP Server (Python版本)..."

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到Python3，请先安装Python 3.8+"
    exit 1
fi

# 检查UV是否安装
if ! command -v uv &> /dev/null; then
    echo "❌ 错误: 未找到UV包管理器"
    echo "请安装UV: curl -sSf https://install.python-uv.org | bash"
    exit 1
fi

# 创建MCP Server目录
echo "📁 创建Python MCP Server目录..."
mkdir -p mcp_server_python
cd mcp_server_python

# 检查是否已有pyproject.toml
if [ ! -f "pyproject.toml" ]; then
    echo "❌ 错误: 未找到pyproject.toml文件"
    echo "请确保您在正确的目录中运行此脚本"
    exit 1
fi

# 安装MCP依赖
echo "📦 安装MCP依赖..."
uv add "mcp[cli]"

# 同步项目依赖
echo "📦 同步项目依赖..."
uv sync

# 测试安装
echo "🧪 测试安装..."
if uv run python -c "import mcp; print('MCP安装成功')"; then
    echo "✅ MCP依赖安装成功"
else
    echo "❌ MCP依赖安装失败"
    exit 1
fi

# 测试服务器
echo "🧪 测试服务器..."
if uv run python -c "from pinnsjax_mcp_server.server import mcp; print('服务器模块加载成功')"; then
    echo "✅ 服务器模块加载成功"
else
    echo "❌ 服务器模块加载失败，请检查代码"
    exit 1
fi

echo "✅ Python MCP Server 设置完成！"
echo ""
echo "📋 下一步："
echo "1. 在AI客户端中配置MCP服务器"
echo "2. 使用 'uv run mcp-inspector pinnsjax-mcp-server' 测试服务器"
echo "3. 开始使用PINN功能！"
echo ""
echo "🔧 配置示例 (Claude Desktop):"
echo "{"
echo "  \"mcpServers\": {"
echo "    \"pinnsjax\": {"
echo "      \"command\": \"uv\","
echo "      \"args\": [\"run\", \"pinnsjax-mcp-server\"],"
echo "      \"cwd\": \"$(dirname $(pwd))\""
echo "    }"
echo "  }"
echo "}"
echo ""
echo "🔧 配置示例 (Cursor):"
echo "{"
echo "  \"name\": \"pinnsjax\","
echo "  \"command\": \"uv\","
echo "  \"args\": [\"run\", \"pinnsjax-mcp-server\"],"
echo "  \"cwd\": \"$(dirname $(pwd))\""
echo "}"
echo ""
echo "💡 提示: 确保工作目录(cwd)指向包含pinnsjax库的项目根目录" 