"""网格采样器模块。

这个模块实现了用于从连续和离散模式的网格中采样的功能。它提供了两种采样器:
MeshSampler用于连续模式, DiscreteMeshSampler用于离散模式。
"""

from typing import Optional

from numpy.typing import NDArray
import jax.numpy as jnp
from jax import Array

from pinnsjax.data import MeshBase
from .sampler_base import SamplerBase


class MeshSampler(SamplerBase):
    """连续模式网格采样器类。

    这个类用于从连续模式的网格中采样, 提供了网格数据的采样、转换和损失计算功能。
    支持在特定时间步或所有时间步进行采样, 并可以处理解输出和采集点。
    """

    def __init__(
        self,
        mesh: MeshBase,
        **kwargs
    ):
        """初始化网格采样器, 用于从网格中采样训练数据。

        参数:
            mesh: 用于采样的网格实例
            **kwargs: 关键字参数, 可包含:
                num_sample: 要采样的样本数量。默认为None, 表示使用所有点
                solution: 解变量名称列表
                dtype: 数据类型, 默认为'float32'
                seed: 随机种子, 默认为0
                idx_t: 时间步索引, None表示采样所有时间步
                collection_points: 采集点模式的名称列表
                use_lhs: 是否使用拉丁超立方采样

        属性说明:
            solution_names: Optional[list[str]]
                解变量名称列表
            idx_t: Optional[int]
                时间步索引, None表示采样所有时间步
            collection_points_names: Optional[list[str]]
                采样点模式的名称列表
            spatial_domain_sampled: list[Array]
                采样后的空间坐标列表
                为一个list, 每个元素的形状为(采样点数, 1)
                例如: 如果空间维度为2, 形状为(N,2)的数组会被存储为两个形状为(N,1)的数组
            time_domain_sampled: Array
                采样后的时间坐标, 形状为(采样点数, 1)
            solution_sampled: Optional[list[Array]]
                采样后的解数据列表, 为None表示不需要解的值(用于PDE约束时)
                # ! 如果solution_names为None, 则solution_sampled为None
                为一个list, 每个元素的形状为(采样点数, 1)
                例如: 如果有多个解变量(u,v), 形状为(N,2)的数组会被存储为两个形状为(N,1)的数组
        """
        # ---------- 关键字参数的初始化 ----------
        num_sample: Optional[int] = kwargs.get('num_sample', None)
        solution: Optional[list[str]] = kwargs.get('solution', None)
        dtype: str = kwargs.get('dtype', 'float32')
        seed: int = kwargs.get('seed', 0)
        idx_t: Optional[int] = kwargs.get('idx_t', None)
        collection_points: Optional[list[str]] = (
            kwargs.get('collection_points', None)
        )
        use_lhs: bool = kwargs.get('use_lhs', True)

        super().__init__(dtype)

        self.solution_names: Optional[list[str]] = solution
        self.idx_t: Optional[int] = idx_t
        self.collection_points_names: Optional[list[str]] = collection_points

        # 情况1: 需要采样解的值(用于训练数据)
        if self.solution_names is not None:
            if self.idx_t is not None:  # 在特定时间步采样, 如t=0.1时刻
                flatten_mesh: tuple[NDArray, NDArray, dict[str, NDArray]] = (
                    mesh.on_initial_boundary(self.solution_names, self.idx_t)
                )
            else:  # 在所有时间步采样
                flatten_mesh: tuple[NDArray, NDArray, dict[str, NDArray]] = (
                    mesh.flatten_mesh(self.solution_names)
                )

            # 从网格中随机采样num_sample个点
            sampled_data: list[Array] = self.sample_mesh(
                num_sample,
                flatten_mesh,
                seed=seed
            )
            self.spatial_domain_sampled: Array = sampled_data[0]
            self.time_domain_sampled: Array = sampled_data[1]
            self.solution_sampled: Array = sampled_data[2]

            # 将解变量数据按维度分割成多个数组, 返回一个list
            # 例如: 如果有多个解变量(u,v), 形状为(N,2)的数组会被分割成两个形状为(N,1)的数组
            self.solution_sampled: list[Array] = jnp.split(
                self.solution_sampled,
                indices_or_sections=self.solution_sampled.shape[1],
                axis=1
            )

        # 情况2: 只采样空间和时间点(用于PDE约束)
        else:
            # 使用拉丁超立方采样生成采集点
            points: tuple[NDArray, NDArray] = (
                mesh.collection_points(num_sample, use_lhs)
            )
            # 转换为张量格式
            sampled_points: list[Array] = self.convert_to_tensor(points)
            self.spatial_domain_sampled: Array = sampled_points[0]
            self.time_domain_sampled: Array = sampled_points[1]
            self.solution_sampled = None

        # 将空间域数据按维度分割成多个数组, 返回一个list
        # 例如: 如果空间维度为2, 形状为(N,2)的数组会被分割成两个形状为(N,1)的数组
        self.spatial_domain_sampled: list[Array] = jnp.split(
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
        solution: list = None,
        collection_points: list = None,
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
