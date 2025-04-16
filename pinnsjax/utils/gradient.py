"""梯度计算工具模块, 提供各种微分运算的实现。"""

from typing import Any, Callable, Union, Sequence
from jax import jacfwd, jacrev, hessian, vmap

__all__ = ['gradient', 'fwd_gradient', 'hessian']


def gradient(
    functional_model: Callable[..., Any],
    argnums: Union[int, Sequence[int]],
    order: int = 1
) -> Callable[..., Any]:
    """计算函数的梯度(反向模式)。

    参数:
        functional_model: 需要计算梯度的函数
        argnums: 指定要对哪些参数计算导数。可以是单个整数, 表示要对第几个参数求导;
                也可以是整数序列, 表示要对多个参数求导。例如: 0表示对第一个参数求导,
                (0,1)表示对前两个参数求导。
        order: 梯度的阶数, 默认为1

    返回:
        计算得到的梯度函数
    """
    grad_functional_model = functional_model
    for _ in range(order):
        grad_functional_model = jacrev(
            grad_functional_model, argnums=argnums
        )
    if functional_model.discrete:
        grad_functional_model = vmap(
            grad_functional_model,
            in_axes=functional_model.in_axes_discrete
        )
    else:
        grad_functional_model = vmap(
            grad_functional_model,
            in_axes=functional_model.in_axes_gard
        )
    return grad_functional_model


def fwd_gradient(
    functional_model: Callable[..., Any],
    argnums: Union[int, Sequence[int]],
    order: int
) -> Callable[..., Any]:
    """计算函数的梯度(前向模式)。

    参数:
        functional_model: 需要计算梯度的函数
        argnums: 指定要对哪些参数计算导数。可以是单个整数, 表示要对第几个参数求导;
                也可以是整数序列, 表示要对多个参数求导。例如: 0表示对第一个参数求导,
                (0,1)表示对前两个参数求导。
        order: 梯度的阶数

    返回:
        计算得到的前向梯度函数
    """
    grad_functional_model = functional_model
    for _ in range(order):
        grad_functional_model = jacfwd(
            grad_functional_model, argnums=argnums
        )
    if functional_model.discrete:
        grad_functional_model = vmap(
            grad_functional_model,
            in_axes=functional_model.in_axes_discrete
        )

    return grad_functional_model
