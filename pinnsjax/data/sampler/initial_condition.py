"""初始条件采样器模块。

这个模块实现了用于采样偏微分方程初始条件的功能。它提供了初始条件的生成、采样以及损失计算等功能。
"""

import numpy as np
import jax.numpy as jnp

from .sampler_base import SamplerBase


class InitialCondition(SamplerBase):
    """初始条件采样器类。

    这个类用于表示和操作偏微分方程的初始条件, 提供了初始条件的生成、采样以及损失计算等功能。
    """

    def __init__(
        self,
        mesh,
        num_sample=None,
        solution=None,
        initial_fun=None,
        dtype: str = 'float32'
    ):
        """初始化一个InitialCondition对象。

        参数:
            mesh: 包含空间和时间域信息的网格对象。
            num_sample: 要采样的样本数量。如果为None, 则使用所有点。
            solution: 解变量名称列表。
            initial_fun: 生成初始条件的函数（可选）。
            dtype: 数据类型, 默认为'float32'。
        """
        super().__init__(dtype)

        self.solution_names = solution

        (self.spatial_domain, self.time_domain, self.solution) = (
            mesh.on_initial_boundary(self.solution_names)
        )

        if initial_fun:
            self.solution = initial_fun(self.spatial_domain)

        (
            self.spatial_domain_sampled,
            self.time_domain_sampled,
            self.solution_sampled,
        ) = self.sample_mesh(
            num_sample,
            (self.spatial_domain, self.time_domain, self.solution)
        )

        self.spatial_domain_sampled = jnp.split(
            self.spatial_domain_sampled,
            indices_or_sections=self.spatial_domain_sampled.shape[1],
            axis=1
        )
        self.solution_sampled = jnp.split(
            self.solution_sampled,
            indices_or_sections=self.solution_sampled.shape[1],
            axis=1
        )

    def sample_mesh(self, num_sample, flatten_mesh):
        """对网格数据进行采样。

        参数:
            num_sample: 要采样的样本数量。如果为None, 则使用所有点。
            flatten_mesh: 扁平化的网格数据, 包含空间域、时间域和解数据。

        返回:
            采样后的空间域、时间域和解数据。
        """
        flatten_mesh = list(flatten_mesh)
        concatenated_solutions = [
            flatten_mesh[2][solution_name]
            for solution_name in self.solution_names
        ]
        flatten_mesh[2] = np.concatenate(concatenated_solutions, axis=-1)

        if num_sample is None:
            return self.convert_to_tensor(flatten_mesh)
        else:
            idx = np.random.choice(
                range(flatten_mesh[0].shape[0]),
                num_sample,
                replace=False
            )
            return self.convert_to_tensor(
                (
                    flatten_mesh[0][idx, :],
                    flatten_mesh[1][idx, :],
                    flatten_mesh[2][idx, :]
                )
            )

    def loss_fn(self, params, inputs, loss, functions):
        """计算初始条件的损失函数。

        参数:
            params: 模型参数。
            inputs: 输入数据, 包含空间坐标、时间坐标和真实解。
            loss: 损失变量。
            functions: 包含前向传播和损失计算函数的字典。

        返回:
            损失值和前向传播的输出字典。
        """

        x, t, u = inputs
        outputs = functions["forward"](params, x, t)
        loss = functions["loss_fn"](
            loss, outputs, u, keys=self.solution_names
        )

        return loss, None
