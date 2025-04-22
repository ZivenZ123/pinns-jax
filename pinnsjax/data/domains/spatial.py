"""空间域模块。

这个模块实现了用于表示和操作空间域的功能。它提供了不同维度（一维、二维和三维）空间域的初始化、网格生成以及基本的访问操作。"""

__all__ = ["Interval", "Rectangle", "RectangularPrism"]

from typing import Sequence, List, Union
import numpy as np


class Interval:
    """一维空间区间类。

    这个类用于表示和操作一维空间区间，提供了一维空间区间的初始化、网格生成以及基本的访问操作。
    """

    def __init__(
        self,
        x_interval: Sequence[float],
        shape: int
    ):
        """初始化一个Interval对象以表示一维空间区间。

        参数:
            x_interval: 表示空间区间[起始x, 结束x]的序列。
                可以是任何支持索引访问的序列类型, 如列表、元组、numpy数组等。
            shape: 区间的空间点数量。
        """
        self.x_interval = x_interval
        self.shape = shape
        self.x = np.linspace(
            self.x_interval[0],
            self.x_interval[1],
            num=self.shape
        )
        self.spatial_dim = 1

    def generate_mesh(self, t_points: int) -> np.ndarray:
        """基于时间点数量生成空间网格。

        参数:
            t_points: 时间点的数量。

        返回:
            在(空间点索引, 时间点索引, 1)网格上对应的空间坐标。
            返回的是一个广播视图(broadcast view),
            这意味着它共享原始空间数组的内存。由于是只读视图, 不能直接修改返回值。
            如果需要修改网格数据, 应该先使用 copy() 方法创建副本。

        注意:
            由于返回的是只读视图, 任何尝试直接修改返回值的操作都会引发 ValueError。
            如果需要修改网格数据, 应该先创建副本:
            >>> mesh = interval.generate_mesh(t_points)
            >>> mesh_copy = mesh.copy()  # 创建可修改的副本
        """
        mesh = np.broadcast_to(
            self.x[:, np.newaxis, np.newaxis],
            (self.shape, t_points, self.spatial_dim)
        )
        return mesh

    def __len__(self) -> int:
        """获取空间区间的长度。

        返回:
            空间区间中的空间点数量。
        """
        return self.shape

    def __getitem__(
        self,
        idx: Union[int, slice, List[int]]
    ) -> Union[float, np.ndarray]:
        """使用索引从空间区间获取特定的空间点。

        参数:
            idx: 所需空间点的索引。

        返回:
            指定索引处的空间值。
        """
        return self.x[idx]


# todo 需要修改: 类型提示, generate_mesh的逻辑以及返回值的维度
class Rectangle:
    """二维空间矩形类。

    这个类用于表示和操作二维空间矩形，提供了二维空间矩形的初始化、网格生成以及基本的访问操作。"""

    def __init__(self, x_interval, y_interval, shape):
        """初始化一个Rectangle对象以表示二维空间矩形。

        参数:
            x_interval: 表示x轴区间[起始x, 结束x]的元组。
            y_interval: 表示y轴区间[起始y, 结束y]的元组。
            shape: 矩形中每个轴上的点数。
        """

        self.x_interval = x_interval
        self.y_interval = y_interval
        self.shape = shape
        self.spatial_dim = 2

    def generate_mesh(self, t_points):
        """为二维空间矩形生成网格。

        参数:
            t_points: 网格中的时间点数量。

        返回:
            二维空间矩形的网格。
        """

        x = np.linspace(
            self.x_interval[0],
            self.x_interval[1],
            num=self.shape[0]
        )
        y = np.linspace(
            self.y_interval[0],
            self.y_interval[1],
            num=self.shape[1]
        )

        self.xx, self.yy = np.meshgrid(x, y)
        self.spatial_mesh = np.stack((self.xx.flatten(), self.yy.flatten()), 1)

        # 创建形状为 (空间点数, 1, 2) 的数组
        spatial_expanded = self.spatial_mesh[:, np.newaxis, :]
        # 使用广播机制广播到所有时间点
        self.mesh = np.broadcast_to(
            spatial_expanded,
            (np.prod(self.shape), t_points, self.spatial_dim)
        )

        return self.mesh

    def __len__(self):
        """获取矩形网格的长度。

        返回:
            矩形网格中的点数。
        """
        return len(self.mesh)

    def __getitem__(self, idx):
        """使用索引从矩形网格获取特定点。

        参数:
            idx: 所需点的索引。

        返回:
            指定索引处的点值。
        """

        return self.mesh[idx, 0]


# todo 需要修改: 类型提示, generate_mesh的逻辑以及返回值的维度
class RectangularPrism:
    """三维空间立方体类。

    这个类用于表示和操作三维空间立方体，提供了三维空间立方体的初始化、网格生成以及基本的访问操作。"""

    def __init__(self, x_interval, y_interval, z_interval, shape):
        """初始化一个RectangularPrism对象以表示三维空间立方体。

        参数:
            x_interval: 表示x轴区间[起始x, 结束x]的元组或列表。
            y_interval: 表示y轴区间[起始y, 结束y]的元组或列表。
            z_interval: 表示z轴区间[起始z, 结束z]的元组或列表。
            shape: 立方体中每个轴上的点数。
        """

        self.x_interval = x_interval
        self.y_interval = y_interval
        self.z_interval = z_interval
        self.shape = shape
        self.spatial_dim = 3

    def generate_mesh(self, t_points):
        """为三维空间立方体生成网格。

        参数:
            t_points: 网格中的时间点数量。

        返回:
            三维空间立方体的网格。
        """

        x = np.linspace(
            self.x_interval[0],
            self.x_interval[1],
            num=self.shape[0]
        )
        y = np.linspace(
            self.y_interval[0],
            self.y_interval[1],
            num=self.shape[1]
        )
        z = np.linspace(
            self.z_interval[0],
            self.z_interval[1],
            num=self.shape[2]
        )

        self.xx, self.yy = np.meshgrid(x, y)
        self.spatial_mesh = np.stack((self.xx.flatten(), self.yy.flatten()), 1)

        # 创建形状为 (空间点数, 1, 3) 的数组
        spatial_expanded = self.spatial_mesh[:, np.newaxis, :]
        # 使用广播机制广播到所有时间点
        self.mesh = np.broadcast_to(
            spatial_expanded,
            (np.prod(self.shape), t_points, self.spatial_dim)
        )

        return self.mesh

    def __len__(self):
        """获取立方体网格的长度。

        返回:
            立方体网格中的点数。
        """
        return len(self.mesh)

    def __getitem__(self, idx):
        """使用索引从立方体网格获取特定点。

        参数:
            idx: 所需点的索引。

        返回:
            指定索引处的点值。
        """
        return self.mesh[idx, 0]
