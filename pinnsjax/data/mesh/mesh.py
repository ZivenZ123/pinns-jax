from typing import Callable, List, Union

import numpy as np
from pyDOE import lhs

from pinnsjax.data import Interval, Rectangle, RectangularPrism, TimeDomain


class MeshBase:
    """该辅助类由Mesh和PointCloud类使用。"""

    def __init__(self):
        """生成网格数据和边界条件的基类。"""
        self.spatial_domain_mesh = None
        self.time_domain_mesh = None
        self.spatial_dim = None
        self.solution = None
        self.lb = None
        self.ub = None

    def domain_bounds(self):
        """根据生成的空间和时间域网格计算域边界。

        :return: 域的下界和上界。
        """
        mesh = np.concatenate(
            (
                self.spatial_domain_mesh,
                self.time_domain_mesh,
            ),
            axis = -1,
        )

        ub = mesh.max(axis = (0, 1))
        lb = mesh.min(axis = (0, 1))
        return lb, ub

    def on_lower_boundary(self, solution_names: List):
        """生成下边界点的数据。

        :param solution_names: 解输出的名称列表。
        :return: 下边界上的空间、时间和解数据。
        """
        spatial_domain = (
            np.ones((self.time_domain_mesh.shape[1], self.spatial_dim)) * self.lb[0:-1]
        )
        time_domain = self.time_domain_mesh[0, :]
        solution_domain = {
            solution_name: self.solution[solution_name][0, :][:, None]
            for solution_name in solution_names
        }

        return spatial_domain, time_domain, solution_domain

    def on_upper_boundary(self, solution_names: List):
        """生成上边界点的数据。

        :param solution_names: 解输出的名称列表。
        :return: 上边界上的空间、时间和解数据。
        """
        spatial_domain = (
            np.ones((self.time_domain_mesh.shape[1], self.spatial_dim)) * self.ub[0:-1]
        )
        time_domain = self.time_domain_mesh[-1, :]
        solution_domain = {
            solution_name: self.solution[solution_name][-1, :][:, None]
            for solution_name in solution_names
        }
        return spatial_domain, time_domain, solution_domain

    def on_initial_boundary(self, solution_names: List, idx: int = 0):
        """生成初始边界点的数据。

        :param solution_names: 解输出的名称列表。
        :param idx: 时间步的索引。
        :return: 初始边界上的空间、时间和解数据。
        """

        spatial_domain = np.squeeze(self.spatial_domain_mesh[:, idx : idx + 1, :], axis=-2)
        time_domain = self.time_domain_mesh[:, idx]
        solution_domain = {
            solution_name: self.solution[solution_name][:, idx : idx + 1]
            for solution_name in solution_names
        }

        return spatial_domain, time_domain, solution_domain

    def collection_points(self, N_f: int, use_lhs: bool = True):
        """生成用于数据收集的点集。

        :param N_f: 要收集的点数。
        :return: 空间域中的点集。
        """
        if use_lhs:
            f = self.lb + (self.ub - self.lb) * lhs(self.spatial_dim + 1, N_f)
            spatial_domain = f[:, 0 : self.spatial_dim]
            time_domain = f[:, self.spatial_dim : self.spatial_dim + 1]
        else:
            spatial_domain, time_domain, _ = self.flatten_mesh(None)
        return spatial_domain, time_domain

    def flatten_mesh(self, solution_names: List):
        """将网格数据展平以进行训练。

        :param solution_names: 解输出的名称列表。
        :return: 展平的空间、时间和解数据。
        """
        time_domain = self.time_domain_mesh.flatten()[:, None]
        spatial_domain = np.zeros((len(time_domain), self.spatial_domain_mesh.shape[-1]))
        for i in range(self.spatial_domain_mesh.shape[-1]):
            spatial_domain[:, i] = self.spatial_domain_mesh[:, :, i].flatten()

        solution_domain = {}
        if solution_names is not None:
            for solution_name in solution_names:
                solution_domain[solution_name] = self.solution[solution_name][:, :].flatten()[
                    :, None
                ]

        return spatial_domain, time_domain, solution_domain


class Mesh(MeshBase):
    """要使用此类, 您应定义SpatialDomain和TimeDomain类。

    如果网格的维度未确定, 最好使用PointCloud。
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
        """根据空间和时间域生成网格, 并加载解数据。

        :param spatial_domain: SpatialDomain类的实例。
        :param time_domain: TimeDomain类的实例。
        :param root_dir: 解数据的根目录。
        :param read_data_fn: 读取解数据的函数。
        :param ub: 域的上界。
        :param lb: 域的下界。
        """

        super().__init__()
        self.solution = read_data_fn(root_dir)
        spatial_points, t_points = list(self.solution.values())[0].shape

        self.spatial_domain, self.time_domain = spatial_domain, time_domain

        # 生成空间和时间域的网格
        self.spatial_domain_mesh = spatial_domain.generate_mesh(t_points)  # 获取网格点上的空间坐标
        self.time_domain_mesh = time_domain.generate_mesh(spatial_points)  # 获取网格点上的时间坐标

        self.spatial_dim = self.spatial_domain_mesh.shape[-1]

        if ub is None and lb is None:
            self.lb, self.ub = self.domain_bounds()
        else:
            self.lb, self.ub = np.array(lb), np.array(ub)


class PointCloud(MeshBase):
    """要使用此类, 您应定义空间域、时间域和解的网格。"""

    def __init__(self, root_dir: str, read_data_fn: Callable, ub: List = None, lb: List = None):
        """生成点云网格并从文件中加载数据。

        :param root_dir: 数据的根目录。
        :param read_data_fn: 读取空间、时间和解数据的函数。
        :param ub: 域的上界。
        :param lb: 域的下界。
        """

        super().__init__()
        data = read_data_fn(root_dir)
        self.spatial_domain, self.time_domain, self.solution = (
            data.spatial,
            data.time,
            data.solution,
        )

        if not isinstance(self.solution, dict):
            raise "解数据的输出不是字典。"

        if isinstance(self.time_domain, list):
            if len(self.time_domain) == 1:
                self.time_domain = self.time_domain[0]

        if not isinstance(self.spatial_domain, list):
            self.spatial_domain = [self.spatial_domain]

        spatial_num_points, time_num_points = list(self.solution.values())[0].shape

        self.spatial_dim = len(self.spatial_domain)
        self.time_dim = 1
        self.solution_dim = len(self.solution.keys())

        # 生成空间和时间域的网格
        self.spatial_domain_mesh = np.zeros(
            (spatial_num_points, time_num_points, self.spatial_dim)
        )

        for i, interval in enumerate(self.spatial_domain):
            self.spatial_domain_mesh[:, :, i] = np.tile(interval, (1, time_num_points))

        self.time_domain_mesh = np.tile(self.time_domain, (1, spatial_num_points)).T[:, :, None]

        if ub is None and lb is None:
            self.lb, self.ub = self.domain_bounds()
        else:
            self.lb, self.ub = np.array(lb), np.array(ub)
