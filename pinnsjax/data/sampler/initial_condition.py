import jax
import numpy as np
import jax.numpy as jnp

from .sampler_base import SamplerBase


class InitialCondition(SamplerBase):
    """初始化初始边界条件。"""

    def __init__(self, mesh, num_sample=None, solution=None, initial_fun=None, dtype: str = 'float32'):
        """初始化一个用于采样初始条件数据的InitialCondition对象。

        :param mesh: 包含空间和时间域信息的网格对象。
        :param num_sample: 样本数量。
        :param solution: 解变量名称列表。
        :param initial_fun: 生成初始条件的函数（可选）。
        """
        super().__init__(dtype)

        self.solution_names = solution

        (self.spatial_domain, self.time_domain, self.solution) = mesh.on_initial_boundary(
            self.solution_names
        )

        if initial_fun:
            self.solution = initial_fun(self.spatial_domain)

        (
            self.spatial_domain_sampled,
            self.time_domain_sampled,
            self.solution_sampled,
        ) = self.sample_mesh(num_sample, (self.spatial_domain, self.time_domain, self.solution))

        self.spatial_domain_sampled = jnp.split(self.spatial_domain_sampled,
                                                indices_or_sections=self.spatial_domain_sampled.shape[1],
                                                axis=1)
        self.solution_sampled = jnp.split(self.solution_sampled, 
                                        indices_or_sections=self.solution_sampled.shape[1], 
                                        axis=1)

    def sample_mesh(self, num_sample, flatten_mesh):
        """对网格数据进行采样用于训练。如果未定义num_sample，则将选择所有点。

        :param num_sample: 要生成的样本数量。
        :param flatten_mesh: 扁平化的网格数据。
        :return: 采样后的空间、时间和解数据。
        """
        flatten_mesh = list(flatten_mesh)
        concatenated_solutions = [
            flatten_mesh[2][solution_name] for solution_name in self.solution_names
        ]
        flatten_mesh[2] = np.concatenate(concatenated_solutions, axis=-1)

        if num_sample is None:
            return self.convert_to_tensor(flatten_mesh)
        else:
            idx = np.random.choice(range(flatten_mesh[0].shape[0]), num_sample, replace=False)
            return self.convert_to_tensor(
                (flatten_mesh[0][idx, :], flatten_mesh[1][idx, :], flatten_mesh[2][idx, :])
            )

    def loss_fn(self, params, inputs, loss, functions):
        """基于输入和函数计算损失函数。

        :param params: 模型参数。
        :param inputs: 用于计算损失的输入数据。
        :param loss: 损失变量。
        :param functions: 损失计算所需的额外函数。
        :return: 损失变量和前向传播的输出字典。
        """

        x, t, u = inputs
        outputs = functions["forward"](params, x, t)
        loss = functions["loss_fn"](loss, outputs, u, keys=self.solution_names)

        return loss, None
