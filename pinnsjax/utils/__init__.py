"""工具函数模块，提供梯度计算、损失函数、数据处理和可视化等功能。"""

from .gradient import (
    gradient,
    hessian,
    jacrev,
    jacfwd,
    fwd_gradient
)
from .module_fn import (
    fix_extra_variables,
    sse,
    mse,
    relative_l2_error,
    make_functional
)
from .utils import (
    extras,
    get_metric_value,
    load_data,
    load_data_txt,
    task_wrapper,
)
from .pylogger import get_pylogger
from .plotting import (
    plot_ac,
    plot_burgers_continuous_forward,
    plot_burgers_continuous_inverse,
    plot_burgers_discrete_forward,
    plot_burgers_discrete_inverse,
    plot_kdv,
    plot_navier_stokes,
    plot_schrodinger,
)

__all__ = [
    'gradient', 'hessian', 'jacrev', 'jacfwd', 'fwd_gradient',
    'fix_extra_variables', 'sse', 'mse', 'relative_l2_error',
    'make_functional', 'extras', 'get_metric_value', 'load_data',
    'load_data_txt', 'task_wrapper', 'get_pylogger',
    'plot_ac', 'plot_burgers_continuous_forward',
    'plot_burgers_continuous_inverse', 'plot_burgers_discrete_forward',
    'plot_burgers_discrete_inverse', 'plot_kdv', 'plot_navier_stokes',
    'plot_schrodinger'
]
