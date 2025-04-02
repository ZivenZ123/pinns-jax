"""梯度计算工具模块，提供各种微分运算的实现。"""

import jax


def gradient(functional_model, argnums, order=1):
    """计算函数模型的梯度。

    Args:
        functional_model: 需要计算梯度的函数模型
        argnums: 计算梯度的参数索引
        order: 梯度的阶数, 默认为1

    Returns:
        计算得到的梯度函数
    """
    grad_functional_model = functional_model
    for _ in range(order):
        grad_functional_model = jax.jacrev(
            grad_functional_model, argnums=argnums
        )
    if functional_model.discrete:
        grad_functional_model = jax.vmap(
            grad_functional_model,
            in_axes=functional_model.in_axes_discrete
        )
    else:
        grad_functional_model = jax.vmap(
            grad_functional_model,
            in_axes=functional_model.in_axes_gard
        )
    return grad_functional_model


def hessian(functional_model, argnums):
    """计算函数模型的Hessian矩阵。

    Args:
        functional_model: 需要计算Hessian的函数模型
        argnums: 计算Hessian的参数索引

    Returns:
        计算得到的Hessian函数
    """
    return jax.hessian(functional_model, argnums=argnums)


def jacrev(functional_model, argnums):
    """计算函数模型的Jacobian矩阵(反向模式)。

    Args:
        functional_model: 需要计算Jacobian的函数模型
        argnums: 计算Jacobian的参数索引

    Returns:
        计算得到的Jacobian函数
    """
    return jax.jacrev(functional_model, argnums=argnums)


def jacfwd(functional_model, argnums):
    """计算函数模型的Jacobian矩阵(前向模式)。

    Args:
        functional_model: 需要计算Jacobian的函数模型
        argnums: 计算Jacobian的参数索引

    Returns:
        计算得到的Jacobian函数
    """
    return jax.jacfwd(functional_model, argnums=argnums)


def fwd_gradient(functional_model, argnums, order):
    """计算函数模型的前向梯度。

    Args:
        functional_model: 需要计算梯度的函数模型
        argnums: 计算梯度的参数索引
        order: 梯度的阶数

    Returns:
        计算得到的前向梯度函数
    """
    grad_functional_model = functional_model
    for _ in range(order):
        grad_functional_model = jax.jacfwd(
            grad_functional_model, argnums=argnums
        )
    if functional_model.discrete:
        grad_functional_model = jax.vmap(
            grad_functional_model,
            in_axes=functional_model.in_axes_discrete
        )

    return grad_functional_model
