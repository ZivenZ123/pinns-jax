"""边界条件模块。

这个模块实现了用于表示和操作边界条件的功能。它提供了狄利克雷边界条件和周期性边界条件的实现。
"""

import numpy as np
import jax.numpy as jnp
import jax

from pinnsjax import utils
from .sampler_base import SamplerBase


class DirichletBoundaryCondition(SamplerBase):
    """狄利克雷边界条件类。

    这个类用于表示和操作狄利克雷边界条件, 提供了边界数据的采样和损失计算功能。
    """

    def __init__(
        self,
        mesh,
        solution,
        num_sample: int = None,
        idx_t: int = None,
        boundary_fun=None,
        discrete: bool = False,
        dtype: str = 'float32'
    ):
        """初始化一个狄利克雷边界条件对象。

        参数:
            mesh: 用于采样的网格实例。
            solution: 解输出的名称。
            num_sample: 要生成的样本数量。如果为None, 则使用所有可用点。
            idx_t: 离散模式下的时间步索引。
            boundary_fun: 可应用于边界数据的函数。
            discrete: 当问题是离散的时为True的布尔值。
            dtype: 数据类型, 默认为'float32'。
        """

        super().__init__(dtype)

        self.solution_names = solution
        self.discrete = discrete

        spatial_upper_bound, time_upper_bound, solution_upper_bound = (
            mesh.on_upper_boundary(self.solution_names)
        )
        spatial_lower_bound, time_lower_bound, solution_lower_bound = (
            mesh.on_lower_boundary(self.solution_names)
        )

        spatial_bound = np.vstack([spatial_upper_bound, spatial_lower_bound])
        time_bound = np.vstack([time_upper_bound, time_lower_bound])

        solution_bound = {}
        for solution_name in self.solution_names:
            solution_bound[solution_name] = np.vstack([
                solution_upper_bound[solution_name],
                solution_lower_bound[solution_name]
            ])

        if boundary_fun:
            solution_bound = boundary_fun(time_bound)

        self.idx_t = idx_t

        (
            self.spatial_domain_sampled,
            self.time_domain_sampled,
            self.solution_sampled,
        ) = self.sample_mesh(
            num_sample,
            (spatial_bound, time_bound, solution_bound)
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
        """对网格数据进行采样用于训练。

        如果定义了idx_t, 则仅选择该时间点上的点。
        如果未定义num_sample, 则选择所有点。

        参数:
            num_sample: 要生成的样本数量。
            flatten_mesh: 扁平化的网格数据。

        返回:
            采样的空间、时间和解数据。
        """

        flatten_mesh = self.concatenate_solutions(flatten_mesh)

        if self.discrete:
            t_points = len(flatten_mesh[0]) // 2
            flatten_mesh = [
                np.vstack(
                    (
                        flatten_mesh_[self.idx_t:self.idx_t + 1, :],
                        flatten_mesh_[
                            self.idx_t + t_points:self.idx_t + t_points + 1, :
                        ],
                    )
                )
                for flatten_mesh_ in flatten_mesh
            ]

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
        """计算基于输入和函数的损失函数。

        参数:
            params: 模型参数。
            inputs: 用于计算损失的输入数据。
            loss: 损失变量。
            functions: 损失计算所需的额外函数。

        返回:
            损失变量和前向传播的输出字典。
        """

        x, t, u = inputs

        # 在离散模式下, 我们不使用时间.
        if self.discrete:
            t = None

        outputs = functions["forward"](params, x, t)

        loss = functions["loss_fn"](loss, outputs, u, keys=self.solution_names)

        return loss, outputs


class PeriodicBoundaryCondition(SamplerBase):
    """周期性边界条件类。

    这个类用于表示和操作周期性边界条件, 提供了边界数据的采样和损失计算功能。
    """

    def __init__(
        self,
        mesh,
        solution,
        idx_t: int = None,
        num_sample: int = None,
        derivative_order: int = 0,
        discrete: bool = False,
        dtype: str = 'float32'
    ):
        """初始化一个周期性边界条件对象。

        参数:
            mesh: 用于采样的网格实例。
            solution: 解输出的名称。
            idx_t: 离散模式下的时间步索引。
            num_sample: 要生成的样本数量。如果为None, 则使用所有可用点。
            derivative_order: 导数的阶数。
            discrete: 当问题是离散的时为True的布尔值。
            dtype: 数据类型, 默认为'float32'。
        """
        super().__init__(dtype)

        self.derivative_order = derivative_order
        self.idx_t = idx_t
        self.solution_names = solution

        spatial_upper_bound, time_upper_bound, _ = mesh.on_upper_boundary(
            self.solution_names
        )
        spatial_lower_bound, time_lower_bound, _ = mesh.on_lower_boundary(
            self.solution_names
        )

        self.discrete = discrete

        sampled_data = self.sample_mesh(
            num_sample,
            (
                spatial_upper_bound,
                time_upper_bound,
                spatial_lower_bound,
                time_lower_bound
            ),
        )
        self.spatial_domain_sampled, self.time_domain_sampled = sampled_data

        self.mid = len(self.time_domain_sampled) // 2

    def sample_mesh(self, num_sample, flatten_mesh):
        """对训练的网格数据进行采样。

        参数:
            num_sample: 要生成的样本数量。
            flatten_mesh: 扁平化的网格数据。

        返回:
            采样的空间、时间和解数据。
        """

        if self.discrete:
            flatten_mesh = [
                flatten_mesh_[self.idx_t:self.idx_t + 1, :]
                for flatten_mesh_ in flatten_mesh
            ]

        if num_sample is None:
            return self.convert_to_tensor(
                (
                    np.vstack((flatten_mesh[0], flatten_mesh[2])),
                    np.vstack((flatten_mesh[1], flatten_mesh[3])),
                )
            )
        else:
            idx = np.random.choice(
                range(flatten_mesh[0].shape[0]),
                num_sample,
                replace=False
            )
            return self.convert_to_tensor(
                (
                    np.vstack((
                        flatten_mesh[0][idx, :],
                        flatten_mesh[2][idx, :]
                    )),
                    np.vstack((
                        flatten_mesh[1][idx, :],
                        flatten_mesh[3][idx, :]
                    )),
                )
            )

    def loss_fn(self, params, inputs, loss, functions):
        """基于输入和函数计算损失函数。

        参数:
            params: 模型参数。
            inputs: 用于计算损失的输入数据。
            loss: 损失变量。
            functions: 损失计算所需的额外函数。

        返回:
            损失变量和前向传播的输出字典。
        """

        x, t, _ = inputs

        # 在离散模式下, 我们不使用时间.
        if self.discrete:
            t = None

        outputs = functions["forward"](params, x, t)

        if self.derivative_order > 0:
            for solution_name in self.solution_names:
                if self.discrete:
                    outputs[f"{solution_name}-tmp"] = jax.vmap(
                        utils.fwd_gradient(
                            functions['functional_net'],
                            argnums=1,
                            order=1
                        ),
                        in_axes=functions['functional_net'].in_axes_discrete
                    )(params, *x, None, solution_name)
                else:
                    outputs[f"{solution_name}-tmp"] = jax.vmap(
                        utils.gradient(
                            functions['functional_net'],
                            argnums=1,
                            order=1
                        ),
                        in_axes=functions['functional_net'].in_axes_gard
                    )(params, *x, t, solution_name)

                loss = functions["loss_fn"](
                    loss,
                    outputs,
                    keys=[f"{solution_name}-tmp"],
                    mid=self.mid
                )
        loss = functions["loss_fn"](
            loss,
            outputs,
            keys=self.solution_names,
            mid=self.mid
        )
        return loss, outputs
