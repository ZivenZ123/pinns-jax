"""实现隐式龙格-库塔方法求解微分方程。"""

import logging

from typing import Dict, List, Union, Optional

import jax.numpy as jnp

from pinnsjax.utils import load_data_txt

log = logging.getLogger(__name__)


class RungeKutta(object):
    """实现隐式龙格-库塔方法求解微分方程的类。"""

    def __init__(
        self,
        root_dir,
        t1: int,
        t2: int,
        time_domain,
        q: int = None,
        dtype: str = 'float32'
    ):
        """初始化一个用于求解微分方程的RungeKutta对象, 使用隐式龙格-库塔方法。

        :param root_dir: 存储权重数据的根目录
        :param t1: 时间域的起始索引
        :param t2: 时间域的结束索引
        :param time_domain: 表示时间域的TimeDomain类
        :param q: 隐式龙格-库塔方法的阶数。如果未提供，将自动计算
        """

        self.np_dtype = jnp.dtype(dtype)

        time_diff = time_domain[t2] - time_domain[t1]
        self.dt = jnp.array(time_diff).astype(self.np_dtype)

        if q is None:
            eps = jnp.finfo(float).eps
            q = int(jnp.ceil(0.5 * jnp.log(eps) / jnp.log(self.dt)))

        self.load_irk_weights(root_dir, q)

    def load_irk_weights(self, root_dir, q: int) -> None:
        """加载隐式龙格-库塔方法的权重和系数，并保存在字典中。

        :param root_dir: 存储权重数据的根目录
        :param q: 隐式龙格-库塔方法的阶数
        """
        file_name = "Butcher_IRK%d.txt" % q
        tmp = load_data_txt(root_dir, file_name)

        weights = jnp.reshape(
            tmp[0: q**2 + q],
            (q + 1, q)
        ).astype(self.np_dtype)

        self.alpha = jnp.array(weights[0:-1, :].T, dtype=self.np_dtype)
        self.beta = jnp.array(weights[-1:, :].T, dtype=self.np_dtype)
        self.weights = jnp.array(weights.T, dtype=self.np_dtype)
        self.IRK_times = tmp[q**2 + q:]

    def __call__(
        self,
        outputs,
        mode: str,
        solution_names: List[str],
        collection_points_names: List[str]
    ):
        """使用龙格-库塔方法执行前向步骤来求解微分方程。

        :param outputs: 包含解张量和其他变量的字典
        :param mode: 前向步骤的模式，例如："inverse_discrete_1"、
            "inverse_discrete_2"、"forward_discrete"
        :param solution_names: 解变量的键列表
        :param collection_points_names: 收集点变量的键列表
        :return: 前向步骤后更新了解张量的字典
        """

        for sol_name, col_name in zip(solution_names, collection_points_names):
            if mode == "inverse_discrete_1":
                outputs[sol_name] = (
                    outputs[sol_name] -
                    self.dt * jnp.matmul(outputs[col_name], self.alpha)
                )

            elif mode == "inverse_discrete_2":
                outputs[sol_name] = (
                    outputs[sol_name] +
                    self.dt * jnp.matmul(
                        outputs[col_name],
                        (self.beta - self.alpha)
                    )
                )

            elif mode == "forward_discrete":
                outputs[sol_name] = (
                    outputs[sol_name] -
                    self.dt * jnp.matmul(outputs[col_name], self.weights)
                )

        return outputs


if __name__ == "__main__":
    _ = RungeKutta(None, None, None, None)
