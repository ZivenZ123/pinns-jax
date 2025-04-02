"""空间域模块，提供一维、二维和三维空间网格的生成功能。"""

import numpy as np


class Interval:
    """初始化一个一维空间区间。"""

    def __init__(self, x_interval, shape):
        """初始化一个Interval对象, 表示一维空间区间。

        :param x_interval: 表示空间区间[start_x, end_x]的元组或列表。
        :param shape: 空间区间中的点数。
        """

        self.x_interval = x_interval
        self.shape = shape

    def generate_mesh(self, t_points: int):
        """为一维空间区间生成网格。

        :param t_points: 网格中的时间点数量。
        :return: 一维空间区间的网格。
        """

        x = np.linspace(
            self.x_interval[0],
            self.x_interval[1],
            num=self.shape[0]
        )
        # 使用广播机制代替 np.tile
        self.mesh = x[:, np.newaxis, np.newaxis]  # 形状变为 (shape[0], 1, 1)
        # 广播到所有时间点
        self.mesh = np.broadcast_to(self.mesh, (self.shape[0], t_points, 1))

        return self.mesh

    def __len__(self):
        """获取区间网格的长度。

        :return: 区间网格中的点数。
        """

        return len(self.mesh)

    def __getitem__(self, idx):
        """使用索引从区间网格获取特定点。

        :param idx: 所需点的索引。
        :return: 指定索引处的点值。
        """

        return self.mesh[idx, 0]


class Rectangle:
    """初始化一个二维空间域。"""

    def __init__(self, x_interval, y_interval, shape):
        """初始化一个Rectangle对象, 表示二维空间矩形。

        :param x_interval: 表示x轴区间[start_x, end_x]的元组。
        :param y_interval: 表示y轴区间[start_y, end_y]的元组。
        :param shape: 矩形中每个轴上的点数。
        """

        self.x_interval = x_interval
        self.y_interval = y_interval
        self.shape = shape

    def generate_mesh(self, t_points):
        """为二维空间矩形生成网格。

        :param t_points: 网格中的时间点数量。
        :return: 二维空间矩形的网格。
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
            (np.prod(self.shape), t_points, 2)
        )

        return self.mesh

    def __len__(self):
        """获取矩形网格的长度。

        :return: 矩形网格中的点数。
        """
        return len(self.mesh)

    def __getitem__(self, idx):
        """使用索引从矩形网格获取特定点。

        :param idx: 所需点的索引。
        :return: 指定索引处的点值。
        """

        return self.mesh[idx, 0]


class RectangularPrism:
    """初始化一个三维空间域。"""

    def __init__(self, x_interval, y_interval, z_interval, shape):
        """初始化一个Rectangular Prism对象, 表示三维形状。

        :param x_interval: 表示x轴区间[start_x, end_x]的元组或列表。
        :param y_interval: 表示y轴区间[start_y, end_y]的元组或列表。
        :param z_interval: 表示z轴区间[start_z, end_z]的元组或列表。
        :param shape: 立方体中每个轴上的点数。
        """

        self.x_interval = x_interval
        self.y_interval = y_interval
        self.z_interval = z_interval
        self.shape = shape

    def generate_mesh(self, t_points):
        """为三维空间立方体生成网格。

        :param t_points: 网格中的时间点数量。
        :return: 三维空间立方体的网格。
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
            (np.prod(self.shape), t_points, 3)
        )

        return self.mesh

    def __len__(self):
        """获取立方体网格的长度。

        :return: 立方体网格中的点数。
        """
        return len(self.mesh)

    def __getitem__(self, idx):
        """使用索引从立方体网格获取特定点。

        :param idx: 所需点的索引。
        :return: 指定索引处的点值。
        """
        return self.mesh[idx, 0]
