#!/bin/zsh

# PINNs-JAX MCP服务器安装脚本 (基于 uv) - ZSH版本

echo "🚀 开始设置PINNs-JAX MCP服务器..."

# 检查Python版本
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
required_version="3.10"

if [[ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]]; then
    echo "❌ 错误: 需要Python 3.10或更高版本，当前版本: $python_version"
    echo "uv 会自动下载并使用正确的 Python 版本"
fi

echo "✅ Python版本检查: $python_version"

# 检查UV是否安装
if ! command -v uv &> /dev/null; then
    echo "📦 UV未安装, 正在安装..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
else
    echo "✅ UV已安装"
fi

# 检查虚拟环境是否存在
if [[ -d ".venv" ]]; then
    echo "✅ 发现现有虚拟环境, 将使用现有环境"
    # 检查虚拟环境中的Python版本
    venv_python_version=$(.venv/bin/python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "unknown")
    echo "📋 虚拟环境Python版本: $venv_python_version"
    
    # 询问是否要重新创建虚拟环境
    echo "🤔 是否要重新创建虚拟环境？(y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "🗑️  删除现有虚拟环境..."
        rm -rf .venv
        echo "🔧 创建新的 Python 3.10 虚拟环境..."
        uv venv --python 3.10
    fi
else
    echo "🔧 创建新的 Python 3.10 虚拟环境..."
    uv venv --python 3.10
fi

# 激活虚拟环境
echo "🔄 激活虚拟环境..."
source .venv/bin/activate

# 同步依赖
echo "📦 安装项目依赖..."
uv sync

# 安装开发依赖
echo "📦 安装开发依赖..."
uv sync --dev

# 测试MCP服务器
echo "🧪 测试MCP服务器..."
if python -c "from pinnsjax_mcp_server.server import main; print('MCP服务器导入成功')"; then
    echo "✅ MCP服务器设置成功!"
else
    echo "❌ MCP服务器设置失败"
    exit 1
fi

echo ""
echo "🎉 安装完成!"
echo ""
echo "📋 下一步:"
echo "1. 激活虚拟环境: source .venv/bin/activate"
echo "2. 在您的Cline中配置MCP服务器"
echo "3. 使用以下配置:"
echo ""
echo "   {"
echo "     \"mcpServers\": {"
echo "       \"pinnsjax\": {"
echo "         \"disabled\": false,"
echo "         \"timeout\": 3600,"
echo "         \"command\": \"/Users/zivenzhong/.local/bin/uv\","
echo "         \"args\": ["
echo "           \"--directory\","
echo "           \"$(pwd)/pinnsjax_mcp_server\","
echo "           \"run\","
echo "           \"start_server.py\""
echo "         ],"
echo "         \"env\": {"
echo "           \"PINNSJAX_ROOT\": \"$(pwd)\""
echo "         },"
echo "         \"transportType\": \"stdio\""
echo "       }"
echo "     }"
echo "   }"
echo ""
echo "4. 重启Cline"
echo "5. 测试命令: '显示所有可用的PINN配置模板'"
echo ""
echo "🔧 常用 uv 命令:"
echo "  uv sync                    # 同步所有依赖"
echo "  uv sync --dev              # 同步包括开发依赖"
echo "  uv add <package>           # 添加新依赖"
echo "  uv remove <package>        # 移除依赖"
echo "  uv run <command>           # 在虚拟环境中运行命令"
echo "  uv --directory pinnsjax_mcp_server run start_server.py # 运行MCP服务器"
echo "  uv run mcp-inspector pinnsjax_mcp_server/start_server.py  # 使用inspector测试" 