#!/usr/bin/env python3
"""
PINNs-JAX MCP Server

这个MCP Server为PINNs-JAX库提供AI助手接口，支持：
- 训练物理信息神经网络模型
- 模型验证和测试
- 结果预测和可视化
- 实验管理和配置
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base

# 设置项目根目录和Python路径
PINNSJAX_ROOT = os.environ.get(
    "PINNSJAX_ROOT",
    "/Users/zivenzhong/Documents/科研/金融数学/2 文献阅读笔记/"
    "PINN（第一部分）：非线性偏微分方程的数据驱动解 (2017)/pinns-jax",
)

# 确保 pinnsjax 包在 Python 路径中
if PINNSJAX_ROOT not in sys.path:
    sys.path.insert(0, PINNSJAX_ROOT)

# 尝试导入pinnsjax模块
try:
    from pinnsjax import train

    PINNSJAX_AVAILABLE = True
    print(
        f"✅ 成功导入 pinnsjax 模块，项目根目录: {PINNSJAX_ROOT}",
        file=sys.stderr,
    )
except ImportError as e:
    PINNSJAX_AVAILABLE = False
    print(
        f"警告: 无法导入pinnsjax模块 ({e})，将使用subprocess方式调用",
        file=sys.stderr,
    )

# 创建FastMCP实例
mcp = FastMCP("PINNs-JAX", dependencies=["jax", "jaxlib", "hydra-core"])

# ==================== 工具 (Tools) ====================


@mcp.tool()
def train_pinn_model(
    config_path: str,
    experiment_name: Optional[str] = None,
    epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    override_params: Optional[Dict[str, Any]] = None,
) -> str:
    """训练物理信息神经网络模型。支持各种PDE问题的求解，包括热传导、波动方程、Navier-Stokes等。

    Args:
        config_path: 配置文件路径，例如: pinnsjax/conf/heat_equation.yaml
        experiment_name: 实验名称，用于标识此次训练
        epochs: 训练轮数，默认使用配置文件中的设置
        learning_rate: 学习率，默认使用配置文件中的设置
        override_params: 覆盖配置参数的字典

    Returns:
        训练结果的描述信息
    """
    try:
        if PINNSJAX_AVAILABLE:
            # 直接调用pinnsjax训练函数
            try:
                # 构建配置覆盖参数
                overrides = []
                if experiment_name:
                    overrides.append(f"experiment_name={experiment_name}")
                if epochs:
                    overrides.append(f"trainer.max_epochs={epochs}")
                if learning_rate:
                    overrides.append(
                        f"model.optimizer.learning_rate={learning_rate}"
                    )
                if override_params:
                    for key, value in override_params.items():
                        overrides.append(f"{key}={value}")

                # 调用训练函数
                result = train.main_train(
                    config_path=config_path, overrides=overrides
                )

                return f"""✅ PINN模型训练完成！

**训练配置:**
- 配置文件: {config_path}
- 实验名称: {experiment_name or '默认'}
- 训练轮数: {epochs or '配置文件默认'}
- 学习率: {learning_rate or '配置文件默认'}

**训练结果:**
{result}"""
            except Exception:
                # 如果直接调用失败，回退到subprocess方式
                return _train_with_subprocess(
                    config_path,
                    experiment_name,
                    epochs,
                    learning_rate,
                    override_params,
                )
        else:
            # 使用subprocess方式
            return _train_with_subprocess(
                config_path,
                experiment_name,
                epochs,
                learning_rate,
                override_params,
            )

    except Exception as e:
        return f"❌ 训练失败: {str(e)}"


def _train_with_subprocess(
    config_path, experiment_name, epochs, learning_rate, override_params
):
    """使用subprocess方式训练模型的后备方法"""
    try:
        # 构建训练命令
        cmd = [sys.executable, "-m", "pinnsjax.train"]
        cmd.append(f"--config-path={config_path}")

        if experiment_name:
            cmd.append(f"experiment_name={experiment_name}")

        if epochs:
            cmd.append(f"trainer.max_epochs={epochs}")

        if learning_rate:
            cmd.append(f"model.optimizer.learning_rate={learning_rate}")

        # 添加其他覆盖参数
        if override_params:
            for key, value in override_params.items():
                cmd.append(f"{key}={value}")

        # 执行训练
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600  # 1小时超时
        )

        if result.returncode == 0:
            warning_section = ""
            if result.stderr:
                warning_section = f"**警告/错误:**\n{result.stderr}"

            return f"""✅ PINN模型训练完成！

**训练配置:**
- 配置文件: {config_path}
- 实验名称: {experiment_name or '默认'}

**训练输出:**
{result.stdout}

{warning_section}"""
        else:
            return f"❌ 训练失败: {result.stderr}"

    except subprocess.TimeoutExpired:
        return "❌ 训练超时（超过1小时）"
    except Exception as e:
        return f"❌ 训练失败: {str(e)}"


@mcp.tool()
def validate_model(
    model_path: str, config_path: str, validation_data: Optional[str] = None
) -> str:
    """验证已训练的PINN模型性能，计算验证集上的误差指标。

    Args:
        model_path: 模型检查点路径
        config_path: 对应的配置文件路径
        validation_data: 验证数据路径，如果不同于训练时使用的数据

    Returns:
        验证结果的描述信息
    """
    try:
        cmd = [sys.executable, "-m", "pinnsjax.validate"]
        cmd.append(f"--config-path={config_path}")
        cmd.append(f"--model-path={model_path}")
        cmd.append("val=true")

        if validation_data:
            cmd.append(f"val_dataset.data_path={validation_data}")

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )

        if result.returncode == 0:
            note_section = ""
            if result.stderr:
                note_section = f"**注意事项:**\n{result.stderr}"

            return f"""📊 模型验证完成！

**验证结果:**
{result.stdout}

{note_section}"""
        else:
            return f"❌ 验证失败: {result.stderr}"

    except Exception as e:
        return f"❌ 验证失败: {str(e)}"


@mcp.tool()
def predict_solution(
    model_path: str,
    config_path: str,
    prediction_points: str,
    output_path: Optional[str] = None,
) -> str:
    """使用训练好的PINN模型预测PDE解。可以在新的时空点上进行预测。

    Args:
        model_path: 训练好的模型路径
        config_path: 模型配置文件路径
        prediction_points: 预测点的数据文件路径或坐标描述
        output_path: 预测结果保存路径

    Returns:
        预测结果的描述信息
    """
    try:
        cmd = [sys.executable, "-m", "pinnsjax.predict"]
        cmd.extend(
            [
                f"--config-path={config_path}",
                f"--model-path={model_path}",
                f"--prediction-points={prediction_points}",
                "test=false",
                "train=false",
            ]
        )

        if output_path:
            cmd.append(f"--output-path={output_path}")

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )

        if result.returncode == 0:
            return f"""🔮 预测完成！

**预测设置:**
- 模型: {model_path}
- 预测点: {prediction_points}
- 输出路径: {output_path or '默认路径'}

**预测结果:**
{result.stdout}"""
        else:
            return f"❌ 预测失败: {result.stderr}"

    except Exception as e:
        return f"❌ 预测失败: {str(e)}"


@mcp.tool()
def visualize_results(
    experiment_path: str,
    plot_types: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> str:
    """生成PINN训练和预测结果的可视化图表，包括损失曲线、解的分布等。

    Args:
        experiment_path: 实验结果路径
        plot_types: 绘图类型列表，如: ['loss', 'solution', 'error']
        save_path: 图片保存路径

    Returns:
        可视化结果的描述信息
    """
    try:
        cmd = [sys.executable, "-m", "pinnsjax.visualize"]
        cmd.append(f"--experiment-path={experiment_path}")

        if plot_types:
            cmd.append(f"--plot-types={','.join(plot_types)}")

        if save_path:
            cmd.append(f"--save-path={save_path}")

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )

        if result.returncode == 0:
            plot_type_str = ", ".join(plot_types) if plot_types else "默认"
            return f"""📈 可视化完成！

**可视化设置:**
- 实验路径: {experiment_path}
- 图表类型: {plot_type_str}
- 保存路径: {save_path or '默认路径'}

**生成结果:**
{result.stdout}"""
        else:
            return f"❌ 可视化失败: {result.stderr}"

    except Exception as e:
        return f"❌ 可视化失败: {str(e)}"


@mcp.tool()
def list_available_configs() -> str:
    """列出所有可用的PINN配置模板，包括不同PDE问题的预设配置。

    Returns:
        可用配置列表的描述信息
    """
    try:
        configs_path = Path(
            "/Users/zivenzhong/Documents/科研/金融数学/2 文献阅读笔记/PINN（第一部分）：非线性偏微分方程的数据驱动解 (2017)/pinns-jax/pinnsjax/conf"
        )
        if not configs_path.exists():
            return "❌ 配置目录不存在: pinnsjax/conf"

        config_files = []
        for config_file in configs_path.rglob("*.yaml"):
            relative_path = config_file.relative_to(configs_path)
            config_files.append(f"- {relative_path}")

        if not config_files:
            return "📋 未找到配置文件"

        config_list = "\n".join(config_files)
        return f"""📋 可用的PINN配置模板:

{config_list}

💡 使用这些配置文件可以快速开始训练不同类型的PDE问题。"""

    except Exception as e:
        return f"❌ 获取配置列表失败: {str(e)}"


@mcp.tool()
def get_training_status(experiment_path: Optional[str] = None) -> str:
    """获取当前或最近训练任务的状态和进度信息。

    Args:
        experiment_path: 实验路径，如果不提供则查找最近的实验

    Returns:
        训练状态信息
    """
    try:
        if experiment_path:
            status_path = Path(experiment_path)
        else:
            # 查找最近的实验
            outputs_path = Path("outputs")
            if not outputs_path.exists():
                return "❌ 输出目录不存在: outputs"

            experiments = [d for d in outputs_path.iterdir() if d.is_dir()]
            if not experiments:
                return "📊 未找到任何实验"

            # 按修改时间排序，获取最新的
            status_path = max(experiments, key=lambda x: x.stat().st_mtime)

        # 读取训练状态信息
        log_files = list(status_path.glob("*.log"))
        metric_files = list(status_path.glob("metrics.json"))

        status_info = f"📊 训练状态信息:\n\n实验路径: {status_path}\n"

        if metric_files:
            with open(metric_files[0], "r") as f:
                metrics = json.load(f)
            status_info += f"最佳损失: {metrics.get('best_loss', 'N/A')}\n"
            status_info += (
                f"验证误差: {metrics.get('validation_error', 'N/A')}\n"
            )

        if log_files:
            status_info += f"日志文件: {len(log_files)} 个\n"

        return status_info

    except Exception as e:
        return f"❌ 获取训练状态失败: {str(e)}"


# ==================== 资源 (Resources) ====================


@mcp.resource("file://training_history.json")
def get_training_history() -> str:
    """存储PINN模型的训练历史和指标数据，包括损失曲线、验证误差等。"""
    try:
        history_path = Path("data/training_history.json")

        if history_path.exists():
            with open(history_path, "r") as f:
                history = json.load(f)
        else:
            history = {
                "experiments": [],
                "last_updated": "未知",
                "total_experiments": 0,
            }

        return f"# PINN训练历史\n\n```json\n{json.dumps(history, indent=2, ensure_ascii=False)}\n```"

    except Exception as e:
        return f"❌ 无法读取训练历史: {str(e)}"


@mcp.resource("file://model_configs.yaml")
def get_model_configs() -> str:
    """管理PINN模型的配置模板和参数设置，支持不同PDE问题的配置。"""
    try:
        configs_path = Path("pinnsjax/conf")
        configs_info = "# PINN模型配置\n\n## 可用配置模板\n\n"

        if not configs_path.exists():
            return configs_info + "❌ 配置目录不存在"

        for config_file in configs_path.rglob("*.yaml"):
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config_content = yaml.safe_load(f)

                relative_path = config_file.relative_to(configs_path)
                configs_info += f"### {relative_path}\n"
                configs_info += f"- 路径: {config_file}\n"
                key_list = (
                    list(config_content.keys()) if config_content else ["无"]
                )
                configs_info += f"- 主要参数: {key_list}\n\n"

            except Exception:
                continue

        return configs_info

    except Exception as e:
        return f"❌ 无法读取模型配置: {str(e)}"


@mcp.resource("file://experiment_results.json")
def get_experiment_results() -> str:
    """存储PINN实验的结果数据，包括模型性能、预测精度等指标。"""
    try:
        outputs_path = Path("outputs")
        results_info = "# PINN实验结果\n\n## 最近实验\n\n"

        if not outputs_path.exists():
            return results_info + "❌ 输出目录不存在"

        experiments = [d for d in outputs_path.iterdir() if d.is_dir()]
        experiments.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        for exp_dir in experiments[:10]:  # 只显示最近10个实验
            results_info += f"### 实验: {exp_dir.name}\n"

            # 读取指标文件
            metrics_file = exp_dir / "metrics.json"
            if metrics_file.exists():
                try:
                    with open(metrics_file, "r") as f:
                        metrics = json.load(f)
                    results_info += (
                        f"- 最佳损失: {metrics.get('best_loss', 'N/A')}\n"
                    )
                    results_info += f"- 验证误差: {metrics.get('validation_error', 'N/A')}\n"
                except Exception:
                    results_info += "- 状态: 指标文件损坏\n"
            else:
                results_info += "- 状态: 未完成\n"

            results_info += f"- 模型路径: {exp_dir / 'checkpoints'}\n\n"

        return results_info

    except Exception as e:
        return f"❌ 无法读取实验结果: {str(e)}"


# ==================== 提示 (Prompts) ====================


@mcp.prompt()
def pinn_expert_guidance() -> List[base.Message]:
    """提供物理信息神经网络(PINN)建模和训练的专家指导，包括最佳实践和问题解决方案。"""

    prompt_text = """# PINNs-JAX 专家指导系统

你是一位物理信息神经网络(Physics-Informed Neural Networks, PINNs)的专家，专门使用JAX框架进行科学计算和机器学习。你的任务是帮助用户有效地使用PINNs-JAX库来解决偏微分方程(PDE)问题。

## 你的专业领域

### 1. 物理信息神经网络理论
- **PINN基础原理**: 理解如何将物理定律(PDE)嵌入神经网络损失函数
- **损失函数设计**: 平衡数据拟合项、PDE残差项和边界条件项
- **网络架构选择**: 针对不同PDE问题选择合适的网络结构
- **训练策略**: 自适应权重、课程学习、多尺度训练等高级技术

### 2. JAX生态系统
- **JAX特性**: 自动微分、JIT编译、向量化、并行计算
- **Optax优化器**: 选择和配置适合的优化算法
- **函数式编程**: JAX的函数式编程范式和最佳实践

### 3. PDE问题类型
- **椭圆型方程**: 泊松方程、拉普拉斯方程
- **抛物型方程**: 热传导方程、扩散方程
- **双曲型方程**: 波动方程、对流方程
- **Navier-Stokes方程**: 流体力学问题
- **非线性PDE**: 薛定谔方程、KdV方程等

## 可用工具和资源

你可以使用以下MCP工具来帮助用户：

### 工具 (Tools)
1. **train_pinn_model**: 训练PINN模型
2. **validate_model**: 验证模型性能
3. **predict_solution**: 使用模型进行预测
4. **visualize_results**: 生成可视化图表
5. **list_available_configs**: 查看可用配置
6. **get_training_status**: 获取训练状态

### 资源 (Resources)
1. **training_history**: 训练历史数据
2. **model_configs**: 模型配置信息
3. **experiment_results**: 实验结果数据

## 指导原则

### 1. 问题诊断
当用户遇到问题时，按以下步骤诊断：
- **理解物理问题**: 确认PDE类型、边界条件、初始条件
- **检查数据质量**: 验证训练数据的完整性和正确性
- **分析网络架构**: 评估网络深度、宽度是否适合问题复杂度
- **审查训练配置**: 检查学习率、批次大小、损失权重等参数

### 2. 性能优化
- **损失平衡**: 调整PDE损失、边界损失、数据损失的权重
- **网络初始化**: 使用Xavier或He初始化，考虑物理约束
- **学习率调度**: 使用余弦退火、指数衰减等策略
- **正则化技术**: 应用dropout、权重衰减等防止过拟合

### 3. 实用建议
- **从简单开始**: 先用简单配置验证方法可行性
- **渐进式训练**: 逐步增加问题复杂度
- **可视化验证**: 定期检查解的物理合理性
- **实验记录**: 保持详细的实验日志和参数记录

## 交互方式

### 回答用户问题时：
1. **理解需求**: 仔细分析用户的具体问题和目标
2. **提供解决方案**: 给出具体的配置建议和代码示例
3. **使用工具**: 主动使用可用的MCP工具来获取信息或执行操作
4. **解释原理**: 说明建议背后的物理和数学原理
5. **预防问题**: 提醒可能遇到的常见陷阱和解决方法

### 代码建议格式：
- 提供完整的配置文件示例
- 包含详细的参数说明
- 给出预期的训练时间和性能指标
- 建议验证和测试步骤

记住：你的目标是帮助用户成功地使用PINNs-JAX解决实际的科学计算问题，既要保证数值精度，也要确保物理意义的正确性。"""

    return [
        base.Message(
            role="user",
            content=base.TextContent(type="text", text=prompt_text),
        )
    ]


def main():
    """启动MCP服务器的主函数"""
    mcp.run()


if __name__ == "__main__":
    main()
