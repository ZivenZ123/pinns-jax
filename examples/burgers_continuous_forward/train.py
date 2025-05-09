"""Burgers方程连续前向传播训练脚本。

这个模块实现了使用PINN(Physics-Informed Neural Networks)来求解Burgers方程的
连续前向传播训练过程。它包含了数据加载、PDE定义和模型训练的主要功能。
"""

from typing import Optional, Callable, Any

import hydra
import numpy as np
from numpy.typing import NDArray
from jax import Array
from omegaconf import DictConfig

from pinnsjax import utils, train


def read_data_fn(root_path: str) -> dict[str, NDArray]:
    """从指定的根路径读取并预处理数据。

    参数:
        root_path: 包含数据的根目录。

    返回:
        处理后的数据将在Mesh类中使用。
        在这里 exact_u 是一个 256*100 的 NDArray, 表示Burgers方程的精确解。
    """

    # * 从指定路径加载数据文件
    data: dict[str, NDArray] = utils.load_data(
        root_path, "burgers_shock.mat"
    )
    # * 获取精确解的实部
    exact_u: NDArray = np.real(data["usol"])
    return {"u": exact_u}


def pde_fn(
    functional_model: Callable[..., Any],
    params: dict[str, Array],
    outputs: dict[str, Array],
    x: Array,
    t: Array
) -> dict[str, Array]:
    """定义Burgers方程的物理约束。

    该函数实现了Burgers方程的物理信息神经网络(PINN)约束：
    u_t + u*u_x - (0.01/π)*u_xx = 0

    参数:
        functional_model: 神经网络模型的函数形式
        params: 神经网络的参数, 使用jax.Array类型
        outputs: 包含神经网络输出的字典，其中包含键"u"对应的解
        x: 空间坐标输入, 使用jax.Array类型
        t: 时间坐标输入, 使用jax.Array类型

    返回:
        更新后的outputs字典, 包含物理残差"f"
    """

    # * 计算u对x和t的一阶偏导数
    u_x, u_t = utils.gradient(
        functional_model,
        argnums=(1, 2),
        order=1
    )(params, x, t, 'u')

    # * 计算u对x的二阶偏导数
    u_xx = utils.gradient(
        functional_model,
        argnums=1,
        order=2
    )(params, x, t, 'u')[0]

    # ! Burgers方程：u_t + u*u_x - (0.01/π)*u_xx = 0
    outputs["f"] = u_t + outputs["u"]*u_x - (0.01/np.pi)*u_xx

    return outputs


@hydra.main(
    version_base="1.3",
    config_path="configs",
    config_name="config"
)
def main(cfg: DictConfig) -> Optional[float]:
    """训练的主入口点。

    参数:
        cfg: 由Hydra组成的DictConfig配置。

    返回:
        包含优化指标值的Optional[float]。
    """

    # * 应用额外的工具
    # * （例如，如果cfg中没有提供标签，则请求标签，打印cfg树等）
    utils.extras(cfg)

    # * 训练模型
    metric_dict, _ = train(
        cfg, read_data_fn=read_data_fn, pde_fn=pde_fn, output_fn=None
    )

    # * 安全地检索基于hydra的超参数优化的指标值
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_names=cfg.get("optimized_metric")
    )

    # * 返回优化后的指标
    return metric_value


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
