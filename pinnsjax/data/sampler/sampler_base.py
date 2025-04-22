"""采样器基类模块。

这个模块实现了用于数据采样的基础功能。它提供了网格数据的采样、转换以及基本的访问操作。
"""

import jax
import jax.numpy as jnp
import numpy as np


class SamplerBase:
    """采样器基类。

    这个类提供了用于训练数据的采样、转换和访问的基础功能。它支持空间域、时间域和解数据的
    采样和操作, 并提供了数据打乱、均值计算等实用功能。
    """

    def __init__(self, dtype):
        """初始化一个SamplerBase对象。

        参数:
            dtype: 数据类型, 用于指定采样数据的数值类型。
        """
        self.time_domain_sampled = None
        self.spatial_domain_sampled = None
        self.solution_sampled = None
        self.solution_names = None
        dtype = 'float32'
        self.np_dtype = np.dtype(dtype)
        self.rng = jax.random.PRNGKey(0)

    def concatenate_solutions(self, flatten_mesh):
        """连接采样解数据。

        将多个解数据连接成一个数组。这个方法主要用于处理多个解变量的情况,
        将它们合并成一个统一的数组以便于后续处理。

        参数:
            flatten_mesh: 包含空间域、时间域和解数据的扁平化网格数据。

        返回:
            带有连接解的扁平化网格数据。
        """
        flatten_mesh = list(flatten_mesh)
        concatenated_solutions = [
            flatten_mesh[2][solution_name]
            for solution_name in self.solution_names
        ]
        flatten_mesh[2] = np.concatenate(concatenated_solutions, axis=-1)

        return flatten_mesh

    def sample_mesh(self, num_sample, flatten_mesh):
        """对网格数据进行采样。

        从给定的网格数据中随机抽取指定数量的样本点。如果num_sample为None,
        则返回所有数据点。

        参数:
            num_sample: 要生成的样本数量。如果为None, 则返回所有数据点。
            flatten_mesh: 扁平化的网格数据。

        返回:
            采样的空间、时间和解数据。
        """

        flatten_mesh = self.concatenate_solutions(flatten_mesh)

        if num_sample is None:
            return self.convert_to_tensor(flatten_mesh)
        else:
            idx = np.random.choice(
                range(flatten_mesh[0].shape[0]),
                num_sample,
                replace=False
            )
            return self.convert_to_tensor((
                flatten_mesh[0][idx, :],
                flatten_mesh[1][idx, :],
                flatten_mesh[2][idx, :]
            ))

    def convert_to_tensor(self, arrays):
        """将NumPy数组转换为JAX张量。

        将输入的NumPy数组转换为JAX张量, 并确保数据类型的一致性。

        参数:
            arrays: 要转换的NumPy数组列表。

        返回:
            转换后的JAX张量列表。
        """

        return [jnp.array(array.astype(self.np_dtype)) for array in arrays]

    def loss_fn(self, inputs, loss, **functions):
        """计算损失函数。

        根据给定的输入和函数计算损失值。这是一个抽象方法, 需要子类实现具体的
        损失计算逻辑。

        参数:
            inputs: 用于计算损失的输入数据。
            loss: 损失变量。
            functions: 计算损失所需的额外函数。
        """
        pass

    def requires_grad(self, x, t, enable_grad=True):
        """设置张量的梯度计算属性。

        为输入列表中的张量设置requires_grad属性, 控制是否计算梯度。

        参数:
            x: 要修改requires_grad属性的张量列表。
            t: 要修改requires_grad属性的张量。
            enable_grad: 布尔值, 表示是否启用梯度计算。

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
            包含沿每列均值的numpy数组。
        """

        x, t, _ = self[:]
        inputs = np.concatenate((*x, t), 1).astype(np.float32)

        return inputs.mean(0, keepdims=True)

    @property
    def std(self):
        """计算输入数据的标准差。

        计算沿着每列连接输入数据的标准差, 用于数据标准化。

        返回:
            包含沿每列标准差值的numpy数组。
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

    def __len__(self):
        """获取采样数据点的数量。

        返回:
            采样数据点的数量。
        """

        return len(self.spatial_domain_sampled[0])

    def __getitem__(self, idx):
        """使用索引获取特定的采样数据点。

        根据索引获取空间域、时间域和解数据。在某些情况下, 可能没有
        time_domain和solution_domain。例如, 在周期性边界条件中, 没有
        solution_domain。

        参数:
            idx: 所需数据点的索引。

        返回:
            包含索引点的空间、时间和解数据的元组。
        """

        spatial_domain = [
            spatial_domain[idx]
            for spatial_domain in self.spatial_domain_sampled
        ]

        time_domain = None
        if self.time_domain_sampled is not None:
            time_domain = self.time_domain_sampled[idx]

        solution_domain = None
        if self.solution_sampled is not None:
            solution_domain = {
                solution_name: self.solution_sampled[i][idx]
                for i, solution_name in enumerate(self.solution_names)
            }

        return (spatial_domain, time_domain, solution_domain)
