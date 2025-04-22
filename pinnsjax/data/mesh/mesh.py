"""网格模块。

这个模块实现了用于表示和操作网格的功能。它提供了网格的初始化、边界条件生成以及数据收集等操作。
"""

from typing import Callable, List, Union

import numpy as np
from pyDOE import lhs

from pinnsjax.data import Interval, Rectangle, RectangularPrism, TimeDomain


class MeshBase:
    """网格基类。

    这个类作为Mesh和PointCloud类的基类, 提供了网格数据的基本操作和边界条件生成功能。
    它定义了网格的基本属性和方法, 包括空间域网格、时间域网格、空间维度、解数据等。
    """

    def __init__(self):
        """初始化MeshBase对象。

        初始化网格的基本属性, 包括空间域网格、时间域网格、空间维度、解数据等。
        这些属性将在子类中被具体实现和赋值。
        """
        self.spatial_domain_mesh = None
        self.time_domain_mesh = None
        self.spatial_dim = None
        self.solution = None
        self.lb = None
        self.ub = None

    def domain_bounds(self):
        """计算网格的域边界。

        根据生成的空间和时间域网格计算域的下界和上界。

        返回:
            包含两个元素的元组:
            - lb: 域的下界, 表示每个维度的最小值
            - ub: 域的上界, 表示每个维度的最大值
        """
        mesh = np.concatenate(
            (
                self.spatial_domain_mesh,
                self.time_domain_mesh,
            ),
            axis=-1,
        )

        ub = mesh.max(axis=(0, 1))
        lb = mesh.min(axis=(0, 1))
        return lb, ub

    def on_lower_boundary(self, solution_names: List):
        """生成下边界点的数据。

        参数:
            solution_names: 需要获取的解数据的名称列表。

        返回:
            包含三个元素的元组:
            - spatial_domain: 下边界上的空间坐标
            - time_domain: 下边界上的时间坐标
            - solution_domain: 下边界上的解数据, 以字典形式返回
        """
        spatial_domain = (
            np.ones((self.time_domain_mesh.shape[1], self.spatial_dim))
            * self.lb[0:-1]
        )
        time_domain = self.time_domain_mesh[0, :]
        solution_domain = {
            solution_name: self.solution[solution_name][0, :][:, None]
            for solution_name in solution_names
        }

        return spatial_domain, time_domain, solution_domain

    def on_upper_boundary(self, solution_names: List):
        """生成上边界点的数据。

        参数:
            solution_names: 需要获取的解数据的名称列表。

        返回:
            包含三个元素的元组:
            - spatial_domain: 上边界上的空间坐标
            - time_domain: 上边界上的时间坐标
            - solution_domain: 上边界上的解数据, 以字典形式返回
        """
        spatial_domain = (
            np.ones((self.time_domain_mesh.shape[1], self.spatial_dim))
            * self.ub[0:-1]
        )
        time_domain = self.time_domain_mesh[-1, :]
        solution_domain = {
            solution_name: self.solution[solution_name][-1, :][:, None]
            for solution_name in solution_names
        }
        return spatial_domain, time_domain, solution_domain

    def on_initial_boundary(self, solution_names: List, idx: int = 0):
        """生成初始边界点的数据。

        参数:
            solution_names: 需要获取的解数据的名称列表。
            idx: 时间步的索引, 默认为0。

        返回:
            包含三个元素的元组:
            - spatial_domain: 初始边界上的空间坐标
            - time_domain: 初始边界上的时间坐标
            - solution_domain: 初始边界上的解数据, 以字典形式返回
        """
        spatial_domain = np.squeeze(
            self.spatial_domain_mesh[:, idx:idx + 1, :], axis=-2
        )
        time_domain = self.time_domain_mesh[:, idx]
        solution_domain = {
            solution_name: self.solution[solution_name][:, idx:idx + 1]
            for solution_name in solution_names
        }

        return spatial_domain, time_domain, solution_domain

    def collection_points(self, n_f: int, use_lhs: bool = True):
        """生成用于数据收集的点集。

        参数:
            n_f: 要收集的点数。
            use_lhs: 是否使用拉丁超立方采样, 默认为True。

        返回:
            包含两个元素的元组:
            - spatial_domain: 空间域中的点集
            - time_domain: 时间域中的点集
        """
        if use_lhs:
            f = self.lb + (self.ub - self.lb) * lhs(self.spatial_dim + 1, n_f)
            spatial_domain = f[:, 0:self.spatial_dim]
            time_domain = f[:, self.spatial_dim:self.spatial_dim + 1]
        else:
            spatial_domain, time_domain, _ = self.flatten_mesh(None)
        return spatial_domain, time_domain

    def flatten_mesh(self, solution_names: List):
        """将网格数据展平以进行训练。

        参数:
            solution_names: 需要展平的解数据的名称列表。

        返回:
            包含三个元素的元组:
            - spatial_domain: 展平后的空间坐标
            - time_domain: 展平后的时间坐标
            - solution_domain: 展平后的解数据, 以字典形式返回
        """
        time_domain = self.time_domain_mesh.flatten()[:, None]
        spatial_domain = np.zeros(
            (len(time_domain), self.spatial_domain_mesh.shape[-1])
        )
        for i in range(self.spatial_domain_mesh.shape[-1]):
            spatial_domain[:, i] = self.spatial_domain_mesh[:, :, i].flatten()

        solution_domain = {}
        if solution_names is not None:
            for solution_name in solution_names:
                solution_domain[solution_name] = (
                    self.solution[solution_name][:, :].flatten()[:, None]
                )

        return spatial_domain, time_domain, solution_domain


class Mesh(MeshBase):
    """网格类。

    这个类用于表示和操作规则网格, 需要已经定义了SpatialDomain和TimeDomain类。
    如果网格的维度未确定, 建议使用PointCloud类。
    """

    def __init__(
        self,
        spatial_domain: Union[Interval, Rectangle, RectangularPrism],
        time_domain: TimeDomain,
        root_dir: str,
        read_data_fn: Callable,
        ub: List = None,
        lb: List = None,
    ):
        """初始化Mesh对象。

        根据空间和时间域生成网格, 并加载解数据。

        参数:
            spatial_domain: SpatialDomain类的实例, 表示空间域。
            time_domain: TimeDomain类的实例, 表示时间域。
            root_dir: 解数据的根目录。
            read_data_fn: 读取解数据的函数。
            ub: 域的上界, 如果为None则自动计算。
            lb: 域的下界, 如果为None则自动计算。
        """
        super().__init__()
        self.solution = read_data_fn(root_dir)

        # 从self.solution字典中获取第一个键值的形状
        # spatial_points: 空间点的数量（行数）
        # t_points: 时间点的数量（列数）
        # ? 为什么用的是solution的shape而不是spatial_domain的shape
        spatial_points, t_points = list(self.solution.values())[0].shape

        self.spatial_domain, self.time_domain = spatial_domain, time_domain

        # 获取网格点上的空间坐标
        self.spatial_domain_mesh = spatial_domain.generate_mesh(t_points)
        # 获取网格点上的时间坐标
        self.time_domain_mesh = time_domain.generate_mesh(spatial_points)

        self.spatial_dim = spatial_domain.spatial_dim

        if ub is None and lb is None:
            self.lb, self.ub = self.domain_bounds()
        else:
            self.lb, self.ub = np.array(lb), np.array(ub)


class PointCloud(MeshBase):
    """点云类。

    这个类用于表示和操作点云网格, 需要定义空间域、时间域和解的网格。
    与Mesh类不同, PointCloud类可以处理不规则网格。
    """

    def __init__(
        self,
        root_dir: str,
        read_data_fn: Callable,
        ub: List = None,
        lb: List = None,
    ):
        """初始化PointCloud对象。

        生成点云网格并从文件中加载数据。

        参数:
            root_dir: 数据的根目录。
            read_data_fn: 读取空间、时间和解数据的函数。
            ub: 域的上界, 如果为None则自动计算。
            lb: 域的下界, 如果为None则自动计算。
        """
        super().__init__()
        data = read_data_fn(root_dir)
        self.spatial_domain, self.time_domain, self.solution = (
            data.spatial,
            data.time,
            data.solution,
        )

        if not isinstance(self.solution, dict):
            raise ValueError("解数据的输出不是字典。")

        if isinstance(self.time_domain, list):
            if len(self.time_domain) == 1:
                self.time_domain = self.time_domain[0]

        if not isinstance(self.spatial_domain, list):
            self.spatial_domain = [self.spatial_domain]

        spatial_num_points, time_num_points = list(
            self.solution.values()
        )[0].shape

        self.spatial_dim = len(self.spatial_domain)
        self.time_dim = 1
        self.solution_dim = len(self.solution.keys())

        # 生成空间和时间域的网格
        self.spatial_domain_mesh = np.zeros(
            (spatial_num_points, time_num_points, self.spatial_dim)
        )

        for i, interval in enumerate(self.spatial_domain):
            self.spatial_domain_mesh[:, :, i] = np.tile(
                interval, (1, time_num_points)
            )

        self.time_domain_mesh = np.tile(
            self.time_domain, (1, spatial_num_points)
        ).T[:, :, None]

        if ub is None and lb is None:
            self.lb, self.ub = self.domain_bounds()
        else:
            self.lb, self.ub = np.array(lb), np.array(ub)
