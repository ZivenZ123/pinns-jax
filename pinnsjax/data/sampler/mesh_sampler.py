"""网格采样器模块。

这个模块实现了用于从连续和离散模式的网格中采样的功能。它提供了两种采样器:
MeshSampler用于连续模式, DiscreteMeshSampler用于离散模式。
"""

from typing import List

import jax.numpy as jnp
from .sampler_base import SamplerBase


class MeshSampler(SamplerBase):
    """连续模式网格采样器类。

    这个类用于从连续模式的网格中采样, 提供了网格数据的采样、转换和损失计算功能。
    支持在特定时间步或所有时间步进行采样, 并可以处理解输出和采集点。
    """

    def __init__(
        self,
        mesh,
        idx_t: int = None,
        num_sample: int = None,
        solution: List = None,
        collection_points: List = None,
        use_lhs: bool = True,
        dtype: str = 'float32'
    ):
        """初始化网格采样器, 用于从网格中采样训练数据。

        参数:
            mesh: 用于采样的网格实例
            idx_t: 时间步索引, None表示采样所有时间步
            num_sample: 要采样的点数
            solution: 要采样的解的名称列表, 如['u']表示采样u(x,t)
            collection_points: 采集点模式的名称列表
            use_lhs: 是否使用拉丁超立方采样
            dtype: 数据类型, 默认为float32
        """
        super().__init__(dtype)

        # 保存初始化参数
        self.idx_t = idx_t
        self.solution_names = solution
        self.collection_points_names = collection_points

        # 情况1: 需要采样解的值（用于训练数据）
        if self.solution_names:
            if self.idx_t:  # 在特定时间步采样, 如t=0.1时刻
                flatten_mesh = mesh.on_initial_boundary(
                    self.solution_names,
                    self.idx_t
                )
            else:  # 在所有时间步采样
                flatten_mesh = mesh.flatten_mesh(self.solution_names)

            # 从网格中随机采样num_sample个点
            # 返回: 空间坐标、时间坐标、解的值
            sampled_data = self.sample_mesh(num_sample, flatten_mesh)
            self.spatial_domain_sampled = sampled_data[0]
            self.time_domain_sampled = sampled_data[1]
            self.solution_sampled = sampled_data[2]

            # 将解的值按列分割, 便于处理多个解
            # 例如: 如果有[u,v]两个解, 则分开存储
            self.solution_sampled = jnp.split(
                self.solution_sampled,
                indices_or_sections=self.solution_sampled.shape[1],
                axis=1
            )

        # 情况2: 只采样空间和时间点（用于PDE约束）
        else:
            # 使用拉丁超立方采样生成采集点
            points = mesh.collection_points(num_sample, use_lhs)
            # 转换为张量格式
            sampled_points = self.convert_to_tensor(points)
            self.spatial_domain_sampled = sampled_points[0]
            self.time_domain_sampled = sampled_points[1]
            self.solution_sampled = None

        # 将空间坐标按维度分割
        # 例如: 2D问题(x,y)会分成x和y两个数组
        self.spatial_domain_sampled = jnp.split(
            self.spatial_domain_sampled,
            indices_or_sections=self.spatial_domain_sampled.shape[1],
            axis=1
        )

    def loss_fn(self, params, inputs, loss, functions):
        """基于输入和函数计算损失函数。

        参数:
            params: 模型参数。
            inputs: 用于计算损失的输入数据, 包含空间坐标、时间坐标和解值。
            loss: 损失变量。
            functions: 损失计算所需的额外函数字典, 包含前向传播、PDE函数和损失函数。

        返回:
            损失变量和前向传播的输出字典。
        """

        x, t, u = inputs

        outputs = functions["forward"](params, x, t)

        if self.collection_points_names:
            outputs = functions["pde_fn"](
                functions["functional_net"],
                params,
                outputs,
                *x,
                t
            )

        loss = functions["loss_fn"](
            loss,
            outputs,
            keys=self.collection_points_names
        )
        loss = functions["loss_fn"](
            loss,
            outputs,
            u,
            keys=self.solution_names
        )

        return loss, outputs


class DiscreteMeshSampler(SamplerBase):
    """离散模式网格采样器类。

    这个类用于从离散模式的网格中采样, 专门处理离散时间步的采样需求。
    支持在特定时间步进行采样, 并可以处理解输出和采集点。
    """

    def __init__(
        self,
        mesh,
        idx_t: int,
        num_sample: int = None,
        solution: List = None,
        collection_points: List = None,
        dtype: str = 'float32'
    ):
        """初始化一个用于在离散模式下收集训练数据的网格采样器。

        参数:
            mesh: 用于采样的网格实例。
            idx_t: 离散模式下的时间步索引。
            num_sample: 要生成的样本数量。
            solution: 解输出的名称列表。
            collection_points: 采集点模式的名称列表。
            dtype: 数据类型, 默认为'float32'。
        """
        super().__init__(dtype)

        self.solution_names = solution
        self.collection_points_names = collection_points
        self.idx_t = idx_t
        self._mode = None

        flatten_mesh = mesh.on_initial_boundary(
            self.solution_names,
            self.idx_t
        )

        (
            self.spatial_domain_sampled,
            self.time_domain_sampled,
            self.solution_sampled,
        ) = self.sample_mesh(num_sample, flatten_mesh)

        self.spatial_domain_sampled = jnp.split(
            self.spatial_domain_sampled,
            indices_or_sections=self.spatial_domain_sampled.shape[1],
            axis=1
        )
        self.time_domain_sampled = None
        self.solution_sampled = jnp.split(
            self.solution_sampled,
            indices_or_sections=self.solution_sampled.shape[1],
            axis=1
        )

    @property
    def mode(self):
        """获取RungeKutta类的当前模式。

        返回:
            当前模式值。
        """
        return self._mode

    @mode.setter
    def mode(self, value):
        """设置RungeKutta类的模式值。

        参数:
            value: 要设置的模式值。
        """
        self._mode = value

    def loss_fn(self, params, inputs, loss, functions):
        """基于输入和函数计算损失函数。

        参数:
            params: 模型参数。
            inputs: 用于计算损失的输入数据, 包含空间坐标、时间坐标和解值。
            loss: 损失变量。
            functions: 损失计算所需的额外函数字典, 包含前向传播、PDE函数、RungeKutta方法和损失函数。

        返回:
            损失变量和前向传播的输出字典。

        注意:
            _mode在PINNDataModule类中分配, 可以是以下值之一:
            - 'inverse_discrete_1'
            - 'inverse_discrete_2'
            - 'forward_discrete'
        """

        x, t, u = inputs

        outputs = functions["forward"](params, x, t)

        if self._mode:
            outputs = functions["pde_fn"](
                functions["functional_net"],
                params,
                outputs,
                *x
            )
            outputs = functions["runge_kutta"](
                outputs,
                mode=self._mode,
                solution_names=self.solution_names,
                collection_points_names=self.collection_points_names,
            )
        loss = functions["loss_fn"](loss, outputs, u, keys=self.solution_names)

        return loss, outputs
