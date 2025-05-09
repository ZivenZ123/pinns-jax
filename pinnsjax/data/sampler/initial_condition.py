"""初始条件采样器模块。

这个模块实现了用于采样偏微分方程初始条件的功能。它提供了初始条件的生成、采样以及损失计算等功能。
"""

from typing import Optional, Callable

from numpy.typing import NDArray
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike

from pinnsjax.data import MeshBase
from .sampler_base import SamplerBase


class InitialCondition(SamplerBase):
    """初始条件采样器类。
    这个类用于表示和操作偏微分方程的初始条件, 提供了初始条件的生成、采样以及损失计算等功能。
    """

    def __init__(
        self,
        mesh: MeshBase,
        **kwargs
    ):
        """初始化一个InitialCondition对象。

        参数:
            mesh: 用于采样的网格实例
            **kwargs: 关键字参数, 可包含:
                num_sample: 要采样的样本数量。默认为None, 表示使用所有点
                solution: 解变量名称列表
                initial_fun: 如果不为None, 表示生成初始条件的闭形式函数
                dtype: 数据类型, 默认为'float32'
                seed: 随机种子, 默认为0

        属性说明:
            solution_names: Optional[list[str]]
                解变量名称列表
            # ? 这个设计很奇怪, 下面三个中, 两个是list, 一个不是
            spatial_domain_sampled: list[Array]
                采样后的空间坐标列表
                为一个list, 每个元素的形状为(采样点数, 1)
                例如: 如果空间维度为2, 形状为(N,2)的数组会被存储为两个形状为(N,1)的数组
            time_domain_sampled: Array
                采样后的时间坐标, 形状为(采样点数, 1)
            solution_sampled: list[Array]
                采样后的解数据列表
                为一个list, 每个元素的形状为(采样点数, 1)
                例如: 如果有多个解变量(u,v), 形状为(N,2)的数组会被存储为两个形状为(N,1)的数组
        """
        # ---------- 关键字参数的初始化, 父类属性的继承 ----------
        num_sample: Optional[int] = kwargs.get('num_sample', None)
        solution: Optional[list[str]] = kwargs.get('solution', None)
        initial_fun: Optional[
            Callable[[NDArray], dict[str, NDArray]]
        ] = kwargs.get('initial_fun', None)
        dtype: DTypeLike = kwargs.get('dtype', 'float32')
        seed: int = kwargs.get('seed', 0)

        super().__init__(dtype)

        # ---------- 获取初始边界上的网格信息, 放入实例属性 ----------
        self.solution_names: Optional[list[str]] = solution
        (
            _spatial_domain,  # 形状为(空间点数, 空间维度)
            _time_domain,  # 形状为(空间点数, 1)
            _solution  # 字典, 每个键值的形状为(空间点数, 1)
        ) = mesh.on_initial_boundary(self.solution_names)
        _spatial_domain: NDArray
        _time_domain: NDArray
        _solution: dict[str, NDArray]

        # 如果提供了initial_fun, 直接使用它计算初始条件
        if initial_fun is not None:
            _solution: dict[str, NDArray] = initial_fun(
                _spatial_domain
            )  # 字典, 每个键值的形状为(空间点数, 1)

        # ---------- 随机采样后更新的网格信息, 放入实例属性 ----------
        (
            self.spatial_domain_sampled,  # 形状为(num_sample, 空间维度)
            self.time_domain_sampled,  # 形状为(num_sample, 1)
            self.solution_sampled,  # 形状为(num_sample, 解的名称数)
        ) = self.sample_mesh(
            num_sample,
            flatten_mesh=(_spatial_domain, _time_domain, _solution),
            seed=seed
        )
        self.spatial_domain_sampled: Array
        self.time_domain_sampled: Array
        self.solution_sampled: Array

        # ---------- 数据后处理, 进行split操作 ----------
        # ? 为什么要做split操作
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

# todo: 没改过loss_fn
    def loss_fn(self, params, inputs, loss, functions):
        """计算初始条件的损失函数。

        参数:
            params: 模型参数
            inputs: 输入数据, 包含空间坐标、时间坐标和真实解
            loss: 损失变量
            functions: 包含前向传播和损失计算函数的字典

        返回:
            损失值和前向传播的输出字典
        """

        x, t, u = inputs
        outputs = functions["forward"](params, x, t)
        loss = functions["loss_fn"](
            loss, outputs, u, keys=self.solution_names
        )

        return loss, None
