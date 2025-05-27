"""采样器基类模块。

这个模块实现了用于数据采样的基础功能。它提供了网格数据的采样、转换以及基本的访问操作。
"""

from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike


class SamplerBase:
    """采样器基类。
    这个类提供了用于训练数据的采样、转换和访问的基础功能。它支持空间域、时间域和解数据的
    采样和操作, 并提供了数据打乱、均值计算等实用功能。
    """

    def __init__(self, dtype: DTypeLike = 'float32'):
        """初始化一个SamplerBase对象。
        一些属性将在子类中被具体实现和赋值。

        参数:
            dtype: 用于指定采样后数据的数值类型

        属性:
            solution_names: 解变量名称列表
            spatial_domain_sampled: 采样后的空间坐标
            time_domain_sampled: 采样后的时间坐标
            solution_sampled: 采样后的解数据
            dtype: 用于指定采样后数据的数值类型
            rng: # todo
        """
        self.solution_names: Optional[list[str]] = None
        self.spatial_domain_sampled = None
        self.time_domain_sampled = None
        self.solution_sampled = None
        self.dtype: DTypeLike = dtype
        self.rng = jax.random.PRNGKey(seed=0)

    def sample_mesh(
        self,
        num_sample: Optional[int],
        flatten_mesh: tuple[NDArray, NDArray, dict[str, NDArray]],
        seed: int = 0
    ) -> list[Array]:
        """对网格数据进行采样。
        从给定的网格点中随机采样指定数量的点。采样过程确保不会重复选择同一个点,
        并保持空间域、时间域和解变量之间的对应关系。

        参数:
            num_sample: 采样点数量。如果为None, 则返回所有网格点数据
            flatten_mesh: 包含三个元素的元组
                1. 空间域数据, 形状为(网格点数, 空间维度)
                2. 时间域数据, 形状为(网格点数, 1)
                3. 解数据字典, 每个键值的形状为(网格点数, 1)
            seed: 随机数生成种子, 默认为0, 用于随机采样

        返回:
            包含三个元素的list:
                - 采样后的空间域数据, 形状为(num_sample, 空间维度)
                - 采样后的时间域数据, 形状为(num_sample, 1)
                - 采样后的解数据, 形状为(num_sample, 解的名称数)
        """

        flatten_mesh: list[Array] = self.concatenate_solutions(flatten_mesh)

        # 如果不需要采样, 直接返回所有网格点上的数据
        if num_sample is None:
            return flatten_mesh

        rng = np.random.default_rng(seed)  # 创建随机数生成器
        idx: NDArray = rng.choice(
            range(flatten_mesh[0].shape[0]),  # 获取网格点数, 表示在范围[0, 网格点数-1]内采样
            size=num_sample,  # 返回的形状为(num_sample,)
            replace=False  # 确保不会重复采样同一个点
        )  # 形状为(num_sample,)

        _sampled_mesh: list[Array] = [mesh[idx] for mesh in flatten_mesh]
        return _sampled_mesh

    def concatenate_solutions(
        self,
        flatten_mesh: tuple[NDArray, NDArray, dict[str, NDArray]]
    ) -> list[Array]:
        """拼接解数据解数据。
        将 flatten_mesh 进行处理, 并将 flatten_mesh 里面的数据转换为tensor。

        参数:
            flatten_mesh: 包含三个元素的元组
                1. 空间域数据, 形状为(网格点数, 空间维度)
                2. 时间域数据, 形状为(网格点数, 1)
                3. 解数据字典, 每个键值的形状为(网格点数, 1)

        返回:
            包含三个元素的list:
                - 空间域数据与 flatten_mesh 的相同
                - 时间域数据与 flatten_mesh 的相同
                - 解数据, 形状为(网格点数, 解的名称数)
        """

        # 将不同解变量的数据在最后一个维度上拼接
        # 例如: 如果有u和v两个解变量, 每个形状为(N,1), 则拼接后形状为(N,2)
        _concatenated_solutions: NDArray = np.concatenate([
            flatten_mesh[-1][solution_name]
            for solution_name in self.solution_names
        ], axis=-1)  # 形状为(网格点数, 解的名称数)

        flatten_mesh = list(flatten_mesh)
        flatten_mesh[-1] = _concatenated_solutions
        return self.convert_to_tensor(flatten_mesh)  # 转化为JAX张量

    def convert_to_tensor(
        self,
        arrays: Union[tuple[NDArray, ...], list[NDArray]]
    ) -> list[Array]:
        """将NumPy数组转换为JAX张量。
        将输入的NumPy数组转换为JAX张量, 并确保数据类型的一致性。

        参数:
            arrays: 要转换的NumPy数组集合, 可以是tuple或list类型

        返回:
            转换后的JAX张量列表
        """
        return [jnp.asarray(array, dtype=self.dtype) for array in arrays]

    # def loss_fn(self, inputs, loss, **functions):
    #     """计算损失函数。

    #     根据给定的输入和函数计算损失值。这是一个抽象方法, 需要子类实现具体的
    #     损失计算逻辑。

    #     参数:
    #         inputs: 用于计算损失的输入数据。
    #         loss: 损失变量。
    #         functions: 计算损失所需的额外函数。
    #     """
    #     pass

    def requires_grad(self, x, t, enable_grad=True):
        """设置张量的梯度计算属性。

        为输入列表中的张量设置requires_grad属性, 控制是否计算梯度。

        参数:
            x: 要修改requires_grad属性的张量列表
            t: 要修改requires_grad属性的张量
            enable_grad: 布尔值, 表示是否启用梯度计算

        返回:
            修改后的张量列表和张量。
        """
        if t is not None:
            t = t.requires_grad_(enable_grad)
        x = [x_.requires_grad_(enable_grad) for x_ in x]

        return x, t

    @property
    def mean(self):
        """计算输入数据的均值。

        计算沿着每列连接输入数据的均值, 用于数据标准化。

        返回:
            包含沿每列均值的numpy数组
        """

        x, t, _ = self[:]
        inputs = np.concatenate((*x, t), 1).astype(np.float32)

        return inputs.mean(0, keepdims=True)

    @property
    def std(self):
        """计算输入数据的标准差。

        计算沿着每列连接输入数据的标准差, 用于数据标准化。

        返回:
            包含沿每列标准差值的numpy数组
        """

        x, t, _ = self[:]
        inputs = np.concatenate((*x, t), 1).astype(np.float32)

        return inputs.std(0, keepdims=True)

    def shuffle(self):
        """打乱采样数据。

        随机打乱空间域、时间域和解数据的顺序, 用于训练数据的随机化。
        这个方法会修改实例的内部状态。
        """
        # 获取随机排列索引
        random_indices = jax.random.permutation(
            self.rng, jnp.arange(len(self.spatial_domain_sampled[0]))
        )

        # 打乱spatial_domain_sampled
        self.spatial_domain_sampled = [
            jnp.take(spatial_domain, random_indices, axis=0)
            for spatial_domain in self.spatial_domain_sampled
        ]

        # 如果time_domain_sampled不为None, 则打乱它
        if self.time_domain_sampled is not None:
            self.time_domain_sampled = jnp.take(
                self.time_domain_sampled, random_indices, axis=0
            )

        # 如果solution_sampled不为None, 则打乱它
        if self.solution_sampled is not None:
            self.solution_sampled = [
                jnp.take(solution, random_indices, axis=0)
                for solution in self.solution_sampled
            ]

    def __len__(self) -> int:
        """获取采样数据点的数量。

        返回:
            采样数据点的数量
        """

        return len(self.spatial_domain_sampled[0])

    def __getitem__(
        self, idx: Union[int, slice, NDArray]
    ) -> tuple[
        Optional[list[Array]],
        Optional[Array],
        Optional[dict[str, Array]]
    ]:
        """使用索引获取特定的采样数据点。
        根据索引获取空间域、时间域和解数据。

        #! 在某些情况下, 可能没有time_domain和solution_domain。
        例如, 在周期性边界条件中, 没有solution_domain。

        参数:
            idx: 所需数据点的索引。可以是整数、切片或布尔/整数数组。
                - 整数: 返回单个数据点
                - 切片: 返回指定范围的数据点
                - 数组: 返回由数组指定的数据点

        返回:
            包含索引点的空间、时间和解数据的元组:
                1. spatial_domain: list, 每个元素形状为(选中点数, 1)
                2. time_domain: 形状为(选中点数, 1)
                3. solution_domain: dict, 长度为解的名称数, 每个键值形状为(选中点数, 1)
        """

        spatial_domain: Optional[list[Array]] = None
        if self.spatial_domain_sampled is not None:
            spatial_domain = [
                spatial_domain[idx]
                for spatial_domain in self.spatial_domain_sampled
            ]

        time_domain: Optional[Array] = None
        if self.time_domain_sampled is not None:
            time_domain = self.time_domain_sampled[idx]

        solution_domain: Optional[dict[str, Array]] = None
        if self.solution_sampled is not None:
            solution_domain = {
                solution_name: self.solution_sampled[i][idx]
                for i, solution_name in enumerate(self.solution_names)
            }

        return spatial_domain, time_domain, solution_domain
