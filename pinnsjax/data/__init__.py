"""PINNs数据模块。

这个模块提供了物理信息神经网络(PINNs)所需的各种数据结构和工具, 包括:
- 空间域和时间域的定义
- 网格数据的表示和处理
- 边界条件和初始条件的实现
- 数据采样和预处理功能
"""

# 点云数据模块，用于处理非结构化数据点
from .domains.point_cloud import PointCloudData
# 空间域模块，定义了各种空间区域：一维区间、二维矩形和三维长方体
from .domains.spatial import Interval, Rectangle, RectangularPrism
# 时间域模块，用于定义时间相关的计算域
from .domains.time import TimeDomain
# 网格模块，包含结构化网格和点云表示
from .mesh.mesh import Mesh, PointCloud
# PINN数据模块，用于处理物理信息神经网络的数据
from .pinn_datamodule import PINNDataModule
# 边界条件模块，包含狄利克雷边界条件和周期性边界条件
from .sampler.boundary_condition import (
    DirichletBoundaryCondition,
    PeriodicBoundaryCondition,
)
# 初始条件模块，用于指定微分方程的初始状态
from .sampler.initial_condition import InitialCondition
# 网格采样器模块，用于在计算域上采样点
from .sampler.mesh_sampler import DiscreteMeshSampler, MeshSampler

__all__ = [
    'PointCloudData',
    'Interval',
    'Rectangle',
    'RectangularPrism',
    'TimeDomain',
    'Mesh',
    'PointCloud',
    'PINNDataModule',
    'DirichletBoundaryCondition',
    'PeriodicBoundaryCondition',
    'InitialCondition',
    'DiscreteMeshSampler',
    'MeshSampler',
]
