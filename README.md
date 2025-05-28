# PINNs-JAX: 物理信息神经网络与MCP服务器

这是一个基于JAX的物理信息神经网络(Physics-Informed Neural Networks, PINNs)实现，集成了Model Context Protocol (MCP) 服务器，可以与AI助手（如Cline等）无缝协作。

## 🚀 特性

- **JAX加速**: 利用JAX的自动微分和JIT编译实现高性能计算
- **MCP集成**: 内置MCP服务器，支持AI助手直接调用
- **多种PDE**: 支持椭圆型、抛物型、双曲型方程和Navier-Stokes方程
- **灵活配置**: 基于Hydra的配置管理系统
- **可视化**: 内置训练过程和结果可视化功能
- **现代包管理**: 使用uv实现极速依赖管理

## 📦 快速开始

### 1. 安装 uv

uv 是一个用 Rust 编写的极速 Python 包管理器，比 pip 快 10-100 倍：

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 或者使用 Homebrew (macOS)
brew install uv
```

### 2. 设置项目

```bash
# 克隆仓库
git clone <your-repo-url>
cd pinns-jax

# 运行安装脚本（推荐，适用于zsh用户）
chmod +x setup_mcp.zsh
./setup_mcp.zsh

# 或者手动设置
uv venv --python 3.10      # 创建虚拟环境
source .venv/bin/activate   # 激活虚拟环境
uv sync                     # 安装所有依赖
uv sync --dev              # 安装开发依赖
```

## 🔧 uv 使用指南

### 核心概念

- **项目管理**: 基于 `pyproject.toml` 的现代 Python 项目
- **虚拟环境**: 自动创建和管理 `.venv` 目录
- **锁定文件**: `uv.lock` 确保依赖版本一致性

### 常用命令

#### 依赖管理
```bash
# 添加依赖
uv add numpy scipy pandas          # 运行时依赖
uv add --dev pytest black isort    # 开发依赖
uv add "numpy>=1.20,<2.0"         # 指定版本范围

# 移除依赖
uv remove package-name
uv remove pytest --dev

# 更新依赖
uv sync --upgrade                  # 更新所有依赖
uv lock --upgrade                  # 更新锁定文件
```

#### 运行命令
```bash
# 在虚拟环境中运行
uv run python script.py           # 运行Python脚本
uv --directory pinnsjax_mcp_server run start_server.py  # 运行MCP服务器
uv run pytest                     # 运行测试
uv run black .                    # 格式化代码
```

#### 环境管理
```bash
# 创建虚拟环境
uv venv --python 3.10             # 指定Python版本
uv venv .venv-py311               # 指定目录

# Python版本管理
uv python list                     # 列出可用版本
uv python install 3.11            # 安装Python版本
```

### uv vs 其他工具对比

| 特性 | uv | pip | conda |
|------|-----|-----|-------|
| 速度 | ⚡ 极快 (10-100x) | 🐌 慢 | 🐢 较慢 |
| 依赖解析 | ✅ 智能并行 | ⚠️ 基础 | ✅ 完整 |
| Python管理 | ✅ 自动下载 | ❌ 需要预装 | ✅ 包含 |
| 锁文件 | ✅ uv.lock | ❌ 无 | ✅ environment.yml |
| 标准兼容 | ✅ 完全遵循PEP | ✅ 标准 | ❌ 自定义 |

## 🔧 MCP服务器配置

### Cline插件配置

在Cline的MCP设置中添加：

```json
{
  "mcpServers": {
    "pinnsjax": {
      "disabled": false,
      "timeout": 3600,
      "command": "/Users/zivenzhong/.local/bin/uv",
      "args": [
        "--directory",
        "/path/to/your/pinns-jax/pinnsjax_mcp_server",
        "run",
        "start_server.py"
      ],
      "env": {
        "PINNSJAX_ROOT": "/path/to/your/pinns-jax"
      },
      "transportType": "stdio"
    }
  }
}
```

**注意**: 请将 `/path/to/your/pinns-jax` 替换为您的实际项目路径。

## 🛠️ MCP工具和功能

### 可用工具

1. **train_pinn_model**: 训练PINN模型
2. **validate_model**: 验证模型性能
3. **predict_solution**: 使用模型进行预测
4. **visualize_results**: 生成可视化图表
5. **list_available_configs**: 查看可用配置
6. **get_training_status**: 获取训练状态

### 可用资源

1. **training_history**: 训练历史数据
2. **model_configs**: 模型配置信息
3. **experiment_results**: 实验结果数据

### 专家提示

- **pinn_expert_guidance**: 获取PINN建模和训练的专家指导

## 📚 使用示例

### 通过AI助手训练模型

```
"请使用heat_equation.yaml配置训练一个PINN模型，训练1000轮，学习率设为0.001"
```

### 查看可用配置

```
"显示所有可用的PINN配置模板"
```

### 运行示例

#### Burgers 方程示例

```bash
# 进入示例目录
cd examples/burgers_continuous_forward

# 运行训练（完整训练）
uv run python train.py

# 运行训练（快速测试，只训练 10 个 epoch）
uv run python train.py trainer.max_epochs=10

# 查看可用的配置选项
uv run python train.py --help
```

## 📁 项目结构

```
pinns-jax/
├── pinnsjax/                    # 核心PINNs库
│   ├── conf/                    # 配置文件
│   ├── models/                  # 模型定义
│   ├── trainer/                 # 训练器
│   ├── utils/                   # 工具函数
│   └── train.py                 # 训练脚本
├── pinnsjax_mcp_server/         # MCP服务器
│   ├── __init__.py
│   └── server.py                # 服务器主文件
├── examples/                    # 示例代码
├── tutorials/                   # 教程
├── data/                        # 数据文件
├── pyproject.toml               # 项目配置和依赖
├── uv.lock                      # 依赖锁定文件
├── .python-version              # Python版本
├── setup_mcp.zsh               # ZSH安装脚本
└── README.md                    # 本文件
```

## 📦 项目依赖

### 核心依赖
- **jax[cpu]** - 自动微分和JIT编译
- **numpy** - 数值计算
- **scipy** - 科学计算
- **matplotlib** - 可视化

### 深度学习
- **flax** - 神经网络库
- **optax** - 优化器

### 配置管理
- **hydra-core** - 配置管理
- **omegaconf** - 配置处理

### MCP服务器
- **mcp** - Model Context Protocol
- **fastapi** - Web框架
- **uvicorn** - ASGI服务器

### 工具库
- **tqdm** - 进度条
- **rich** - 美化输出
- **rootutils** - 项目根目录管理
- **pyDOE** - 实验设计

所有依赖都在 `pyproject.toml` 中定义，使用 `uv sync` 自动安装。

## 🧪 测试MCP服务器

使用MCP Inspector测试服务器：

```bash
# 使用UV运行inspector
uv run mcp-inspector pinnsjax_mcp_server/start_server.py

# 或者直接运行服务器
uv --directory pinnsjax_mcp_server run start_server.py
```

## 🔍 故障排除

### 常见问题

1. **Python版本不匹配**
   ```bash
   # 检查可用版本
   uv python list
   
   # 安装需要的版本
   uv python install 3.10
   
   # 重新创建环境
   rm -rf .venv
   uv venv --python 3.10
   ```

2. **依赖冲突**
   ```bash
   # 查看依赖树
   uv tree
   
   # 强制更新锁定文件
   uv lock --upgrade
   
   # 清除缓存
   uv cache clean
   ```

3. **MCP模块找不到**
   ```bash
   uv add "mcp>=1.2.0"
   ```

4. **JAX安装问题**
   ```bash
   # CPU版本
   uv add "jax[cpu]"
   
   # GPU版本（CUDA）
   uv add "jax[cuda12]"
   ```

5. **网络问题**
   ```bash
   # 使用国内镜像
   export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/
   ```

### 调试技巧

- 使用MCP Inspector测试各个功能
- 检查AI客户端的开发者控制台
- 查看服务器的stderr输出

## 📖 支持的PDE类型

- **椭圆型方程**: 泊松方程、拉普拉斯方程
- **抛物型方程**: 热传导方程、扩散方程  
- **双曲型方程**: 波动方程、对流方程
- **Navier-Stokes方程**: 流体力学问题
- **非线性PDE**: 薛定谔方程、KdV方程等

## 🚀 性能优势

使用uv带来的性能提升：
- **安装速度**: 比 pip 快 10-100 倍
- **依赖解析**: 并行解析，智能缓存
- **磁盘使用**: 全局缓存，避免重复下载
- **内存效率**: Rust 实现，内存占用低

## 💡 最佳实践

### 1. 版本控制
```bash
# 提交到 git
git add pyproject.toml uv.lock .python-version
git commit -m "Add uv configuration"

# .gitignore 添加
.venv/
__pycache__/
*.pyc
```

### 2. CI/CD 配置
```yaml
# GitHub Actions 示例
- name: Set up uv
  run: curl -LsSf https://astral.sh/uv/install.sh | sh

- name: Install dependencies
  run: uv sync --dev

- name: Run tests
  run: uv run pytest
```

## ❓ 常见问题

**Q: 如何切换 Python 版本？**
A: 使用 `uv venv --python 3.11` 创建新环境。

**Q: 如何导出依赖列表？**
A: 使用 `uv pip freeze > requirements.txt`。

**Q: 如何在 CI/CD 中使用？**
A: 参考上面的 GitHub Actions 配置示例。

**Q: 从 conda 如何迁移？**
A: 手动转换 environment.yaml 到 pyproject.toml，然后运行 `uv sync`。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

MIT License

## 🔗 相关链接

- [JAX官方文档](https://jax.readthedocs.io/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Hydra配置管理](https://hydra.cc/)
- [uv包管理器](https://github.com/astral-sh/uv)
- [uv官方文档](https://docs.astral.sh/uv/)
