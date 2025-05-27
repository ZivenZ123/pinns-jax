# PINNs-JAX MCP Server

这是一个为PINNs-JAX库构建的Model Context Protocol (MCP) Server，它为AI助手提供了与物理信息神经网络(Physics-Informed Neural Networks)交互的能力。

## 功能特性

### 🛠️ 工具 (Tools)
- **train_pinn_model**: 训练PINN模型，支持各种PDE问题
- **validate_model**: 验证已训练模型的性能
- **predict_solution**: 使用训练好的模型进行预测
- **visualize_results**: 生成训练和预测结果的可视化图表
- **list_available_configs**: 列出所有可用的配置模板
- **get_training_status**: 获取训练任务的状态和进度

### 📊 资源 (Resources)
- **training_history**: 存储训练历史和指标数据
- **model_configs**: 管理模型配置模板和参数
- **experiment_results**: 存储实验结果和性能数据

### 💡 提示 (Prompts)
- **pinn_expert_guidance**: 提供PINN建模和训练的专家指导

## 安装和设置

### 1. 克隆和安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd mcp_server

# 安装依赖
npm install

# 编译TypeScript
npm run build
```

### 2. 配置AI客户端

#### Claude Desktop 配置

在Claude Desktop的设置中添加MCP服务器配置：

```json
{
  "mcpServers": {
    "pinnsjax": {
      "command": "node",
      "args": ["/path/to/mcp_server/build/index.js"],
      "cwd": "/path/to/your/pinnsjax/project"
    }
  }
}
```

#### Cursor 配置

在Cursor的MCP设置中添加：

```json
{
  "name": "pinnsjax",
  "command": "node",
  "args": ["/path/to/mcp_server/build/index.js"],
  "cwd": "/path/to/your/pinnsjax/project"
}
```

### 3. 验证安装

使用MCP Inspector测试服务器：

```bash
npm run inspector
```

## 使用示例

### 训练PINN模型

```typescript
// 通过AI助手调用
"请使用heat_equation.yaml配置训练一个PINN模型，训练1000轮"

// 对应的工具调用
train_pinn_model({
  config_path: "configs/heat_equation.yaml",
  experiment_name: "heat_eq_experiment",
  epochs: 1000,
  learning_rate: 0.001
})
```

### 查看训练历史

```typescript
// AI助手可以访问训练历史资源
"显示最近的训练历史"

// 访问 training_history 资源
```

### 获取专家指导

```typescript
// 使用专家提示
"我在训练Navier-Stokes方程的PINN时遇到收敛问题，请提供建议"

// 触发 pinn_expert_guidance 提示
```

## 支持的PDE类型

- **椭圆型方程**: 泊松方程、拉普拉斯方程
- **抛物型方程**: 热传导方程、扩散方程  
- **双曲型方程**: 波动方程、对流方程
- **Navier-Stokes方程**: 流体力学问题
- **非线性PDE**: 薛定谔方程、KdV方程等

## 项目结构

```
mcp_server/
├── src/
│   ├── index.ts          # 主入口文件
│   ├── tools/            # 工具实现
│   │   └── index.ts
│   ├── resources/        # 资源实现
│   │   └── index.ts
│   └── prompts/          # 提示实现
│       └── index.ts
├── build/                # 编译输出
├── package.json
├── tsconfig.json
└── README.md
```

## 开发和调试

### 开发模式

```bash
# 监听文件变化并自动编译
npm run dev
```

### 调试

```bash
# 使用MCP Inspector进行调试
npm run inspector
```

### 日志

服务器会在stderr输出日志信息，可以通过AI客户端的开发者工具查看。

## 配置要求

### 环境要求
- Node.js 18+
- Python 3.8+ (用于运行PINNs-JAX)
- JAX和相关依赖

### PINNs-JAX项目结构
确保您的PINNs-JAX项目具有以下结构：
```
your_project/
├── pinnsjax/           # PINNs-JAX库
├── configs/            # 配置文件
├── data/              # 数据文件
├── outputs/           # 输出结果
└── mcp_server/        # 本MCP服务器
```

## 故障排除

### 常见问题

1. **模块找不到错误**
   - 确保已正确安装所有依赖
   - 检查工作目录是否正确设置

2. **Python命令执行失败**
   - 确保Python环境中安装了PINNs-JAX
   - 检查配置文件路径是否正确

3. **权限错误**
   - 确保MCP服务器有读写项目目录的权限

### 调试技巧

- 使用MCP Inspector测试各个功能
- 检查AI客户端的开发者控制台
- 查看服务器的stderr输出

## 贡献

欢迎提交Issue和Pull Request来改进这个MCP服务器！

## 许可证

MIT License 