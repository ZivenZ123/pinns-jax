import numpy as np


class TimeDomain:
    """初始化一个时间域。"""

    def __init__(self, t_interval, t_points: int):
        """初始化一个TimeDomain对象以表示时间域。

        :param t_interval: 表示时间区间[起始时间, 结束时间]的元组或列表。
        :param t_points: 用于离散化区间的时间点数量。
        """
        self.time_interval = t_interval
        self.time = np.linspace(self.time_interval[0], self.time_interval[1], num=t_points)

    def generate_mesh(self, spatial_points):
        """基于空间点数量生成时间网格（使用广播而非 tile）。
        
        :param spatial_points: 空间点的数量。
        :return: 时间和空间点的网格。
        """
        # self.time.shape == (t_points,) 假设有 t_points 个时间点
        # 先在前面和后面各添加一个尺寸为 1 的新轴，变为 (1, t_points, 1)
        # 再通过 broadcast_to 逻辑扩展到 (spatial_points, t_points, 1)
        mesh = np.broadcast_to(self.time[None, :, None], (spatial_points, self.time.size, 1))
        return mesh

    def __len__(self):
        """获取时间域的长度。

        :return: 时间域中的时间点数量。
        """
        return len(self.time)

    def __getitem__(self, idx):
        """使用索引从时间域获取特定的时间点。

        :param idx: 所需时间点的索引。
        :return: 指定索引处的时间值。
        """
        return self.time[idx]
