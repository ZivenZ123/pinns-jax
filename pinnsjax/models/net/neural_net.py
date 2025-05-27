"""
全连接神经网络模块, 用于物理信息神经网络(PINN)的实现。

该模块提供了一个全连接神经网络(FCN)的实现, 专为解决物理信息神经网络问题而设计。
它支持连续和离散问题的求解, 并提供了权重归一化等特性。
"""

from numpy.typing import NDArray
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike


class FCN:
    """全连接神经网络类, 用于物理信息神经网络(PINN)的实现。

    该类实现了一个带有权重归一化的全连接神经网络, 可用于解决连续或离散的物理问题。
    网络使用SWISH激活函数(sigmoid线性单元), 并支持多输出。
    """

    def __init__(
        self,
        layers: list[int],
        lb: NDArray,
        ub: NDArray,
        output_names: list[str],
        discrete: bool = False,
        dtype: DTypeLike = 'float32'
    ):
        """初始化一个 `FCN` 模块。

        参数:
            layers: 表示每层神经元数量的列表
            lb: 域的下界, 形状为(空间维度+1,)
                例如, 对于2D空间: [x_min, y_min, t_min]
            ub: 域的上界, 形状为(空间维度+1,)
                例如, 对于2D空间: [x_max, y_max, t_max]
            output_names: 输出变量的名称列表
            discrete: 问题是否为离散的
            dtype: 数据类型，默认为'float32'

        属性:
            lb: 域的下界, 形状为(空间维度+1,)
                例如, 对于2D空间: [x_min, y_min, t_min]
            ub: 域的上界, 形状为(空间维度+1,)
                例如, 对于2D空间: [x_max, y_max, t_max]
            n_dim: 输入维度
            params: 网络参数，包括权重和偏置
            output_names: 输出变量的名称列表
            discrete: 是否为离散问题
        """
        super().__init__()

        self.lb: Array = jnp.array(lb, dtype=dtype)
        self.ub: Array = jnp.array(ub, dtype=dtype)
        self.n_dim: int = len(self.lb)

        key = jax.random.PRNGKey(1234)

        self.params = self.initialize_net(key, layers)
        self.output_names: list[str] = output_names
        self.discrete: bool = discrete
        # self.forward = jax.vmap(self.forward, in_axes=(None, 0))

    def _xavier_init(
        self,
        key: Array,
        size: tuple[int, int]
    ) -> Array:
        """使用 Xavier 初始化方法初始化权重矩阵。

        Xavier初始化(也称为Glorot初始化)的核心思想是保持每一层输入和输出的方差一致,
        防止深层网络中的梯度消失或爆炸问题。当网络层数较深时, 如果使用标准正态分布初始化权重,
        信号在前向传播过程中可能会迅速衰减或放大, 导致训练困难。

        Xavier初始化通过将权重的方差设置为2/(输入维度+输出维度), 使得每层输出的方差保持相对稳定。
        这样可以确保信号既不会在网络中消失也不会爆炸, 从而使深层网络能够更有效地学习。

        参数:
            key: JAX 随机数生成器的密钥
            size: 包含输入维度和输出维度的元组(in_dim, out_dim)

        返回:
            初始化后的权重矩阵, 形状为(in_dim, out_dim)
        """
        in_dim, out_dim = size
        xavier_stddev = jnp.sqrt(2.0 / (in_dim + out_dim))

        return jax.random.normal(key, shape=(in_dim, out_dim)) * xavier_stddev

    def initialize_net(
        self,
        key: Array,
        layers: list[int]
    ) -> dict[str, list[Array]]:
        """初始化整个神经网络的参数。

        参数:
            key: JAX 随机数生成器的密钥
            layers: 表示每层神经元数量的列表

        返回:
            包含网络参数的字典, 包括权重和偏置
        """
        num_layers = len(layers)
        weights = []
        biases = []
        for i in range(num_layers-1):
            _, layer_key = jax.random.split(key)
            W = self._xavier_init(
                layer_key,
                (layers[i], layers[i + 1])
            )
            b = jnp.zeros((1, layers[i + 1]))
            weights.append(W)
            biases.append(b)
        params = {'weights': weights,
                  'biases': biases}
        return params

    def forward(
        self,
        params: dict[str, list[Array]],
        X: Array
    ) -> Array:
        """执行神经网络的前向传播计算。

        该方法首先对输入数据进行归一化处理，将其映射到[-1,1]区间，然后通过神经网络的各层进行前向计算。
        对于中间层使用tanh激活函数, 最后一层不使用激活函数(线性输出)。

        参数:
            params: 包含网络参数的字典，包括权重('weights')和偏置('biases')
            X: 输入数据，形状为(batch_size, input_dim)

        返回:
            神经网络的输出，形状为(batch_size, output_dim)
        """
        if self.discrete:
            H = 2.0 * (X - self.lb[:-1]) / (self.ub[:-1] - self.lb[:-1]) - 1.0
        else:
            H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0

        for i, (w, b) in enumerate(zip(params['weights'], params['biases'])):
            if i == len(params) - 1:
                H = jnp.dot(H, w) + b
            else:
                H = jax.nn.tanh(jnp.dot(H, w) + b)

        return H

    def __call__(self, params, spatial, time):
        """执行网络的单次前向传播。

        参数:
            spatial: 输入空间张量的列表
            time: 表示时间的输入张量
        返回:
            解的张量
        """

        if self.discrete is False:
            z = jnp.hstack([*spatial, time])
        else:
            z = jnp.hstack([*spatial])

        z = self.forward(params, z)

        # 离散模式
        if self.discrete:
            outputs_dict = {
                name: z for i, name in enumerate(self.output_names)
            }

        # 连续模式
        else:
            outputs_dict = {
                name: z[:, i:i + 1] for i, name in enumerate(self.output_names)
            }

        return outputs_dict


class NetHFM:
    """
    在这个模型中, 均值和标准差将用于输入数据的归一化。同时, 也将进行权重归一化。
    """

    def __init__(
        self,
        mean,
        std,
        layers: list,
        output_names: list[str],
        discrete=False
    ):
        """初始化一个 `NetHFM` 模块。

        参数:
            mean: 输入数据的均值
            std: 输入数据的标准差
            layers: 表示每层神经元数量的列表
            output_names: 网络输出的名称列表
            discrete: 是否为离散问题, 默认为False
        """
        super().__init__()
        self.num_layers = len(layers)
        self.output_names = output_names
        self.trainable_variables = []

        key = jax.random.PRNGKey(0)

        self.discrete = discrete

        self.X_mean = jnp.array(mean, dtype=jnp.float32)
        self.X_std = jnp.array(std, dtype=jnp.float32)
        print(mean[0])
        self.n_dim = len(mean[0])

        self.params = self.initalize_net(key, layers)

    def initalize_net(self, key, layers: list) -> None:
        """初始化神经网络的权重、偏置和伽马参数。

        参数:
            key: 随机数生成器的密钥
            layers: 表示每层神经元数量的列表
        """

        weights = []
        biases = []
        gammas = []

        for i in range(0, self.num_layers-1):
            in_dim = layers[i]
            out_dim = layers[i+1]
            key, layer_key = jax.random.split(key)
            W = jax.random.normal(layer_key, (in_dim, out_dim))
            b = jnp.zeros([1, out_dim])
            g = jnp.ones([1, out_dim])

            weights.append(W)
            biases.append(b)
            gammas.append(g)

        params = {'weights': weights,
                  'biases': biases,
                  'gammas': gammas}
        return params

    def forward(self, params, H):
        """执行前向传播计算。

        参数:
            params: 网络参数
            H: 输入数据
        返回:
            处理后的输出
        """
        H = (H - self.X_mean) / self.X_std

        for i, (W, b, g) in enumerate(
            zip(
                params['weights'],
                params['biases'],
                params['gammas']
            )
        ):
            # 权重归一化
            V = W / jnp.linalg.norm(W, axis=0, keepdims=True)

            # 矩阵乘法
            H = jnp.dot(H, V)

            # 添加偏置
            H = g * H + b

            # 激活函数
            if i < self.num_layers - 2:
                H = H * jax.nn.sigmoid(H)
        return H

    def __call__(
        self,
        params,
        spatial: list[Array],
        time: Array
    ) -> dict[str, Array]:
        """执行网络的前向传播。

        参数:
            params: 网络参数
            spatial: 输入空间张量的列表
            time: 表示时间的输入张量
        返回:
            一个字典, 键为输出名称, 值为对应的输出张量
        """

        if self.discrete is False:
            H = jnp.hstack([*spatial, time])
        else:
            H = jnp.hstack([*spatial])

        H = self.forward(params, H)

        outputs_dict = {
            name: H[:, i:i+1] for i, name in enumerate(self.output_names)
        }

        return outputs_dict


if __name__ == "__main__":
    _ = FCN()
    _ = NetHFM()
