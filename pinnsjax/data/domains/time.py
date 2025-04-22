"""时间域模块。

这个模块实现了用于表示和操作时间域的功能。它提供了时间域的初始化、网格生成以及基本的访问操作。
"""

from typing import Sequence, List, Union
import numpy as np


class TimeDomain:
    """时间域类。

    这个类用于表示和操作时间域, 提供了时间域的初始化、网格生成以及基本的访问操作。
    """

    def __init__(
        self,
        t_interval: Sequence[float],
        t_points: int
    ):
        """初始化一个TimeDomain对象以表示时间域。

        参数:
            t_interval: 表示时间区间[起始时间, 结束时间]的序列。
                可以是任何支持索引访问的序列类型, 如列表、元组、numpy数组等。
            t_points: 区间的时间点数量。
        """
        self.time_interval = t_interval
        self.t_points = t_points
        self.time = np.linspace(
            self.time_interval[0],
            self.time_interval[1],
            num=t_points
        )

    def generate_mesh(self, spatial_points: int) -> np.ndarray:
        """基于空间点数量生成时间网格。

        参数:
            spatial_points: 空间点的数量。

        返回:
            在(空间点索引, 时间点索引, 1)网格上对应的时间坐标。
            返回的是一个广播视图(broadcast view),
            这意味着它共享原始时间数组的内存。由于是只读视图, 不能直接修改返回值。
            如果需要修改网格数据, 应该先使用 copy() 方法创建副本。

        注意:
            由于返回的是只读视图, 任何尝试直接修改返回值的操作都会引发 ValueError。
            如果需要修改网格数据, 应该先创建副本:
            >>> mesh = time_domain.generate_mesh(spatial_points)
            >>> mesh_copy = mesh.copy()  # 创建可修改的副本
        """
        mesh = np.broadcast_to(
            self.time[np.newaxis, :, np.newaxis],
            (spatial_points, self.t_points, 1)
        )
        return mesh

    def __len__(self) -> int:
        """获取时间域的长度。

        返回:
            时间域中的时间点数量。
        """
        return self.t_points

    def __getitem__(
        self,
        idx: Union[int, slice, List[int]]
    ) -> Union[float, np.ndarray]:
        """使用索引从时间域获取特定的时间点。

        参数:
            idx: 所需时间点的索引。

        返回:
            指定索引处的时间值。
        """
        return self.time[idx]
