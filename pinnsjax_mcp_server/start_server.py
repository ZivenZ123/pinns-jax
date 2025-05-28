#!/usr/bin/env python3
"""
PINNs-JAX MCP Server 启动脚本

这个脚本确保 MCP server 能够在任何目录下正确运行，
通过设置正确的环境变量和 Python 路径。
"""

import os
import sys
from pathlib import Path


def setup_environment():
    """设置运行环境"""
    # 获取当前脚本所在目录
    script_dir = Path(__file__).parent.absolute()

    # 设置项目根目录（上一级目录）
    project_root = script_dir.parent

    # 设置环境变量
    os.environ["PINNSJAX_ROOT"] = str(project_root)
    os.environ["PYTHONPATH"] = str(project_root)

    # 添加到 Python 路径
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print(f"🔧 设置项目根目录: {project_root}")
    print(f"🔧 设置 PYTHONPATH: {project_root}")


def main():
    """主函数"""
    # 设置环境
    setup_environment()

    # 导入并启动 server
    try:
        from server import main as server_main

        print("🚀 启动 PINNs-JAX MCP Server...")
        server_main()
    except ImportError as e:
        print(f"❌ 无法导入 server 模块: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
