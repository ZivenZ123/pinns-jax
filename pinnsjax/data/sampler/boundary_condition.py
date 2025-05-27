"""边界条件模块。

这个模块实现了用于表示和操作边界条件的功能。它提供了狄利克雷边界条件和周期性边界条件的实现。
"""

from typing import Optional, Callable

import numpy as np
from numpy.typing import NDArray
import jax.numpy as jnp
import jax
from jax import Array

from pinnsjax import utils
from pinnsjax.data import MeshBase
from .sampler_base import SamplerBase


# todo: 添加详细类型提示
class DirichletBoundaryCondition(SamplerBase):
    """狄利克雷边界条件类。
    这个类用于表示和操作狄利克雷边界条件, 提供了边界数据的采样和损失计算功能。
    """

    def __init__(
        self,
        mesh: MeshBase,
        **kwargs
    ):
        """初始化一个狄利克雷边界条件对象。

        参数:
            mesh: 用于采样的网格实例
            **kwargs: 关键字参数, 可包含:
                num_sample: 要采样的样本数量。默认为None, 表示使用所有点
                solution: 解变量名称列表
                boundary_fun: 如果不为None, 表示生成边界条件的闭形式函数
                dtype: 数据类型, 默认为'float32'
                seed: 随机种子, 默认为0
                discrete: 当问题是离散时为True, 默认为False
                idx_t: 离散模式下的时间步索引

        属性:
            solution_names: Optional[list[str]]
                解变量名称列表
            discrete: bool
                当问题是离散时为True, 否则为False
            idx_t: Optional[int]
                离散模式下的时间步索引
            spatial_domain_sampled: list[Array]
                采样后的空间坐标列表
                采样点分布在上下边界, 不要求上下边界对应采样点相同
                为一个list, 每个元素的形状为(采样点数, 1)
                例如: 如果空间维度为2, 形状为(N,2)的数组会被存储为两个形状为(N,1)的数组
            time_domain_sampled: Array
                采样后的时间坐标, 形状为(采样点数, 1)
                采样点分布在上下边界, 不要求上下边界对应采样点相同
            solution_sampled: list[Array]
                采样后的解数据列表
                采样点分布在上下边界, 不要求上下边界对应采样点相同
                为一个list, 每个元素的形状为(采样点数, 1)
                例如: 如果有多个解变量(u,v), 形状为(N,2)的数组会被存储为两个形状为(N,1)的数组
        """
        # ---------- 关键字参数的初始化, 父类属性的继承 ----------
        num_sample: Optional[int] = kwargs.get('num_sample', None)
        solution: Optional[list[str]] = kwargs.get('solution', None)
        boundary_fun: Optional[Callable] = kwargs.get('boundary_fun', None)
        dtype: str = kwargs.get('dtype', 'float32')
        seed: int = kwargs.get('seed', 0)
        discrete: bool = kwargs.get('discrete', False)
        idx_t: Optional[int] = kwargs.get('idx_t', None)

        super().__init__(dtype)

        # ---------- 获取边界上的网格信息 ----------
        self.solution_names: Optional[list[str]] = solution
        self.discrete: bool = discrete  # todo 还没用上
        self.idx_t: Optional[int] = idx_t  # todo 还没用上

        (
            _spatial_upper_bound,  # 形状为(时间点数, 空间维度)
            _time_upper_bound,  # 形状为(时间点数, 1)
            _solution_upper_bound  # 字典, 每个键值的形状为(时间点数, 1)
        ) = mesh.on_upper_boundary(self.solution_names)
        _spatial_upper_bound: NDArray
        _time_upper_bound: NDArray
        _solution_upper_bound: dict[str, NDArray]
        (
            _spatial_lower_bound,  # 形状为(时间点数, 空间维度)
            _time_lower_bound,  # 形状为(时间点数, 1)
            _solution_lower_bound  # 字典, 每个键值的形状为(时间点数, 1)
        ) = mesh.on_lower_boundary(self.solution_names)
        _spatial_lower_bound: NDArray
        _time_lower_bound: NDArray
        _solution_lower_bound: dict[str, NDArray]

        _spatial_domain: NDArray = np.vstack(
            [_spatial_upper_bound, _spatial_lower_bound]
        )  # 形状为(2*时间点数, 空间维度)
        _time_domain: NDArray = np.vstack(
            [_time_upper_bound, _time_lower_bound]
        )  # 形状为(2*时间点数, 1)

        # ! 采样点的解值可能直接来自数据, 也可能来自闭形式的求值
        if boundary_fun is not None:
            # 如果提供了boundary_fun, 直接使用它计算边界条件
            _solution: dict[str, NDArray] = boundary_fun(
                _time_domain)  # 字典, 每个键值的形状为(2*时间点数, 1)
        else:
            # 如果没有提供boundary_fun, 从mesh中获取边界条件
            _solution: dict[str, NDArray] = {
                solution_name: np.vstack([
                    _solution_upper_bound[solution_name],
                    _solution_lower_bound[solution_name]
                ]) for solution_name in self.solution_names
            }  # 字典, 每个键值的形状为(2*时间点数, 1)

        # ---------- 随机采样后更新的网格信息, 放入实例属性 ----------
        if num_sample is not None:  # 表示需要采样
            (
                self.spatial_domain_sampled,  # 形状为(num_sample, 空间维度)
                self.time_domain_sampled,  # 形状为(num_sample, 1)
                self.solution_sampled,  # 形状为(num_sample, 解的名称数)
            ) = self.sample_mesh(
                num_sample,
                flatten_mesh=(_spatial_domain, _time_domain, _solution),
                seed=seed
            )
        else:  # 如果不需要采样
            (
                self.spatial_domain_sampled,  # 形状为(2*时间点数, 空间维度)
                self.time_domain_sampled,  # 形状为(2*时间点数, 1)
                self.solution_sampled,  # 形状为(2*时间点数, 解的名称数)
            ) = self.concatenate_solutions(
                flatten_mesh=(_spatial_domain, _time_domain, _solution)
            )
        self.spatial_domain_sampled: Array
        self.time_domain_sampled: Array
        self.solution_sampled: Array

        # ---------- 数据后处理, 进行split操作 ----------
        # 将空间域数据按维度分割成多个数组, 返回一个list
        # 例如: 如果空间维度为2, 形状为(N,2)的数组会被分割成两个形状为(N,1)的数组
        self.spatial_domain_sampled: list[Array] = jnp.split(
            self.spatial_domain_sampled,
            indices_or_sections=self.spatial_domain_sampled.shape[1],
            axis=1
        )

        # 将解变量数据按维度分割成多个数组, 返回一个list
        # 例如: 如果有多个解变量(u,v), 形状为(N,2)的数组会被分割成两个形状为(N,1)的数组
        self.solution_sampled: list[Array] = jnp.split(
            self.solution_sampled,
            indices_or_sections=self.solution_sampled.shape[1],
            axis=1
        )

    # ? 这个是用来干嘛, 现在还没有用上
    def _process_discrete_boundary(
        self,
        flatten_mesh: tuple[Array, Array, Array]
    ) -> tuple[Array, Array, Array]:
        """处理离散模式下的边界条件数据。

        参数:
            flatten_mesh: 包含上下边界的空间、时间和解数据的元组
                1. 空间域, 形状为(2*时间点数, 空间维度)
                2. 时间域, 形状为(2*时间点数, 1)
                3. 解, 形状为(2*时间点数, 解的名称数)
        返回:
            处理后的边界数据元组，只包含指定时间步的数据:
                1. 空间域, 形状为(2, 空间维度)，包含上下边界在指定时间步的空间坐标
                2. 时间域, 形状为(2, 1)，包含上下边界在指定时间步的时间坐标
                3. 解, 形状为(2, 解的名称数)，包含上下边界在指定时间步的解值
        """
        t_points: int = len(flatten_mesh[0]) // 2  # 获取时间点数
        return tuple(
            jnp.vstack((
                # 把上下边界对应的时间点拼接起来
                _mesh[self.idx_t: self.idx_t+1, :],
                _mesh[self.idx_t+t_points: self.idx_t+t_points+1, :],
            ))
            for _mesh in flatten_mesh
        )

    # def sample_mesh(
    #     self,
    #     num_sample: Optional[int],
    #     key: Optional[Array] = None
    # ):
    #     """对网格数据进行采样用于训练。
    #     如果定义了idx_t, 则仅选择该时间点上的点
    #     如果未定义num_sample, 则选择所有点

    #     参数:
    #         num_sample: 采样点数量。如果为None, 则返回所有网格点数据
    #                 这里的采样点数量包含了上边界与下边界采样点之和
    #                 并不要求上边界和下边界在对应相同的时间点上采样
    #         flatten_mesh: 扁平化的网格数据

    #     返回:
    #         采样的空间、时间和解数据
    #     """

    #     # ---------- 先进行数据处理, 得到 flatten_mesh ----------
    #     # 将不同解变量的数据在最后一个维度上拼接
    #     # 例如: 如果有u和v两个解变量, 每个形状为(N,1), 则拼接后形状为(N,2)
    #     _concatenated_solutions: Array = jnp.concatenate([
    #         self.solution[solution_name]
    #         for solution_name in self.solution_names
    #     ], axis=-1)  # 形状为(2*时间点数, 解的名称数)

    #     # 每个元素都是jax.Array类型, 确保数据类型一致
    #     _dt = self.dtype
    #     flatten_mesh: tuple[Array, Array, Array] = (
    #         jnp.array(self.spatial_domain, dtype=_dt),  # 形状为(2*时间点数, 空间维度)
    #         jnp.array(self.time_domain, dtype=_dt),  # 形状为(2*时间点数, 1)
    #         _concatenated_solutions  # 形状为(2*时间点数, 解的名称数)
    #     )

    #     # 处理离散模式下的边界条件
    #     if self.discrete:
    #         flatten_mesh = self._process_discrete_boundary(flatten_mesh)

    #     # 如果不需要采样, 直接返回所有时间点上的数据
    #     if num_sample is None:
    #         return flatten_mesh

    #     # ---------- 再进行随机采样, 得到采样后的 flatten_mesh ----------
    #     # 如果没有提供随机种子, 使用默认种子
    #     if key is None:
    #         key: Array = jr.PRNGKey(0)

    #     # 生成随机索引, 用于从所有数据点中采样
    #     idx: Array = jr.choice(
    #         key,  # 随机数生成器的密钥
    #         flatten_mesh[0].shape[0],  # 获取2*时间点数, 表示在范围[0, 2*时间点数-1]内采样
    #         shape=(num_sample,),  # 返回的形状为(num_sample,)
    #         replace=False  # 确保不会重复采样同一个点
    #     )  # 形状为(num_sample,)

    #     # 使用生成的索引对每个数据数组进行采样
    #     return tuple(mesh[idx] for mesh in flatten_mesh)

    # todo: 没改过loss_fn
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


# todo: 没改过PeriodicBoundaryCondition
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
