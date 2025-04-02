"""Burgers方程连续前向传播训练脚本。

这个模块实现了使用PINN(Physics-Informed Neural Networks)来求解Burgers方程的
连续前向传播训练过程。它包含了数据加载、PDE定义和模型训练的主要功能。
"""

from typing import Dict, Optional

import hydra
import numpy as np
import jax

from omegaconf import DictConfig

import pinnsjax


def read_data_fn(root_path):
    """从指定的根路径读取并预处理数据。

    :param root_path: 包含数据的根目录。
    :return: 处理后的数据将在Mesh类中使用。
    """

    # * 从指定路径加载数据文件
    data = pinnsjax.utils.load_data(root_path, "burgers_shock.mat")
    # * 获取精确解的实部
    exact_u = np.real(data["usol"])
    return {"u": exact_u}


def pde_fn(functional_model,
           params,
           outputs: Dict[str, jax.Array],
           x: jax.Array,
           t: jax.Array):
    """定义偏微分方程(PDEs)。"""

    # * 计算u对x和t的一阶偏导数
    u_x, u_t = pinnsjax.utils.gradient(
        functional_model,
        argnums=(1, 2),
        order=1
    )(params, x, t, 'u')

    # * 计算u对x的二阶偏导数
    u_xx = pinnsjax.utils.gradient(
        functional_model,
        argnums=1,
        order=2
    )(params, x, t, 'u')[0]

    # ! Burgers方程：u_t + u*u_x - (0.01/π)*u_xx = 0
    outputs["f"] = u_t + outputs["u"] * u_x - (0.01 / np.pi) * u_xx

    return outputs


@hydra.main(
    version_base="1.3",
    config_path="configs",
    config_name="config.yaml"
)
def main(cfg: DictConfig) -> Optional[float]:
    """训练的主入口点。

    :param cfg: 由Hydra组成的DictConfig配置。
    :return: 包含优化指标值的Optional[float]。
    """

    # * 应用额外的工具
    # * （例如，如果cfg中没有提供标签，则请求标签，打印cfg树等）
    pinnsjax.utils.extras(cfg)

    # * 训练模型
    metric_dict, _ = pinnsjax.train(
        cfg, read_data_fn=read_data_fn, pde_fn=pde_fn, output_fn=None
    )

    # * 安全地检索基于hydra的超参数优化的指标值
    metric_value = pinnsjax.utils.get_metric_value(
        metric_dict=metric_dict, metric_names=cfg.get("optimized_metric")
    )

    # * 返回优化后的指标
    return metric_value


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
