import functools
import jax
import jax.numpy as jnp

from typing import Dict, List, Union, Optional, Tuple


def sse(loss,
        preds,
        target=None,
        keys=None,
        mid=None):
    """计算给定预测值和可选目标值的平方和误差 (SSE) 损失.

    :param loss: 损失变量.
    :param preds: 包含不同键的预测张量的字典.
    :param target: 包含目标张量的字典 (可选).
    :param keys: 需要计算 SSE 损失的键列表 (可选).
    :param mid: 用于中点计算的预测值分隔索引 (可选).
    :return: 计算得到的 SSE 损失.
    """

    if keys is None:
        return loss

    for key in keys:
        if target is None and mid is None:
            loss = loss + jnp.sum(jnp.square(preds[key]))
        elif target is None and mid is not None:
            diff = preds[key][:mid] - preds[key][mid:]
            loss = loss + jnp.sum(jnp.square(diff))
        elif target is not None:
            loss = loss + jnp.sum(jnp.square(preds[key] - target[key]))

    return loss


def mse(loss,
        preds,
        target=None,
        keys=None,
        mid=None):
    """计算给定预测值和可选目标值的均方误差 (MSE) 损失.

    :param loss: 损失变量.
    :param preds: 包含不同键的预测张量的字典.
    :param target: 包含目标张量的字典 (可选).
    :param keys: 需要计算 SSE 损失的键列表 (可选).
    :param mid: 用于中点计算的预测值分隔索引 (可选).
    :return: 计算得到的 MSE 损失.
    """

    if keys is None:
        return loss

    for key in keys:
        if target is None and mid is None:
            loss = loss + jnp.mean(jnp.square(preds[key]))
        elif target is None and mid is not None:
            diff = preds[key][:mid] - preds[key][mid:]
            loss = loss + jnp.mean(jnp.square(diff))
        elif target is not None:
            loss = loss + jnp.mean(jnp.square(preds[key] - target[key]))

    return loss


def relative_l2_error(preds, target):
    """计算预测张量和目标张量之间的相对 L2 误差.

    :param preds: 预测张量.
    :param target: 目标张量.
    :return: 相对 L2 误差值.
    """

    numerator = jnp.mean(jnp.square(preds - target))
    denominator = jnp.mean(jnp.square(target))
    return jnp.sqrt(numerator/denominator)


def fix_extra_variables(trainable_variables, extra_variables):
    """将额外变量转换为带有梯度跟踪的 tf 张量. 这些变量在反问题中是可训练的.

    :param extra_variables: 需要转换的额外变量字典.
    :return: 转换后的额外变量字典, 作为带有梯度的 tf 张量.
    """

    if extra_variables is None:
        return trainable_variables, None
    extra_variables_dict = {}
    for key in extra_variables:
        variable = jnp.array(extra_variables[key])
        extra_variables_dict[key] = variable
        trainable_variables[key] = variable
    return trainable_variables, extra_variables_dict


def make_functional(net, params, n_dim, discrete, output_fn):
    """根据维度数使模型具有函数式特性.

    :param net: 神经网络模型.
    :param params: 模型参数.
    :param n_dim: 维度数.
    :param output_fn: 应用于输出的输出函数.
    :return: 函数式模型和轴形状.
    """

    def functional_model_1d(params, x, time, output_c=None):
        return _execute_model(net, params, [x], time, output_c)

    def functional_model_2d(params, x, y, time, output_c=None):
        return _execute_model(net, params, [x, y], time, output_c)

    def functional_model_3d(params, x, y, z, time, output_c=None):
        return _execute_model(net, params, [x, y, z], time, output_c)

    functional_model_1d.discrete = discrete
    functional_model_2d.discrete = discrete
    functional_model_3d.discrete = discrete

    if discrete:
        functional_model_1d.in_axes_discrete = (None, 0, None, None)
        functional_model_1d.in_axes = (None, None, 0, 0)
        functional_model_2d.in_axes_discrete = (None, 0, 0, None, None)
        functional_model_2d.in_axes = (None, None, 0, 0, 0)
        functional_model_3d.in_axes_discrete = (None, 0, 0, 0, None, None)
        functional_model_3d.in_axes = (None, None, 0, 0, 0, 0)
    else:
        functional_model_1d.in_axes = (None, None, 0, 0, 0)
        functional_model_1d.in_axes_gard = (None, 0, 0, None)
        functional_model_2d.in_axes = (None, None, 0, 0, 0, 0)
        functional_model_2d.in_axes_gard = (None, 0, 0, 0, None)
        functional_model_3d.in_axes = (None, None, 0, 0, 0, 0, 0)
        functional_model_3d.in_axes_gard = (None, 0, 0, 0, 0, None)

    def _execute_model(net, params, inputs, time, output_c):
        outputs_dict = net(params, inputs, time)

        if output_c is None:
            return outputs_dict
        else:
            return outputs_dict[output_c].squeeze()

    models = {
        2: functional_model_1d,
        3: functional_model_2d,
        4: functional_model_3d,
    }

    functional_model_fun = models[n_dim]

    try:
        return models[n_dim]
    except KeyError:
        raise ValueError(f"{n_dim} 维度数不受支持.")
    '''

    #output_fn = (output_fn if output_fn is None 
    #            else functools.partial(jax.vmap,
    #                                  in_axes=functional_model_fun.in_axes)(output_fn))

    def functional_model(params, x, y, z, time, output_c=None):
        outputs = functional_model_fun(params, x, y, z, time)

        #return outputs
        #if output_fn:
        #    outputs = output_fn(functional_model_fun, params, outputs, x, time)

        if output_c is None:
            return outputs
        else:
            return [outputs[output_].squeeze() for output_ in output_c]

    if discrete:
        functional_model.in_axes = functional_model_fun.in_axes
        functional_model.in_axes_discrete = (
            functional_model_fun.in_axes_discrete)
        functional_model.discrete = discrete
    else:
        functional_model.in_axes = functional_model_fun.in_axes
        functional_model.in_axes_gard = functional_model_fun.in_axes_gard
        functional_model.discrete = discrete

    return functional_model
    '''
