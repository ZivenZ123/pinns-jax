"""网格模块。

这个模块实现了用于表示和操作网格的功能。它提供了网格的初始化、边界条件生成以及数据收集等操作。
"""

from typing import (
    Callable, List, Union, Sequence, Tuple, Dict, Optional
)

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

        初始化网格的基本属性, 这些属性将在子类中被具体实现和赋值。
        """
        self.spatial_domain_mesh: Optional[np.ndarray] = None
        self.time_domain_mesh: Optional[np.ndarray] = None
        self.spatial_dim: Optional[int] = None
        self.solution: Optional[Dict[str, np.ndarray]] = None
        self.lb: Optional[np.ndarray] = None
        self.ub: Optional[np.ndarray] = None

    def domain_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """计算网格的上下边界。

        返回:
            包含两个元素的元组:
            - lb: 域的下界, 表示每个维度的最小值
            - ub: 域的上界, 表示每个维度的最大值

        示例:
            假设有3个空间点, 有2个时间点, 空间维度为2:

            全网格上的空间坐标示例(spatial_domain_mesh):
            [
                [[x1,y1], [x1,y1]],  # 点1在两个时间点的坐标
                [[x2,y2], [x2,y2]],  # 点2在两个时间点的坐标
                [[x3,y3], [x3,y3]]   # 点3在两个时间点的坐标
            ]  # 形状: (3, 2, 2)

            全网格上的时间坐标示例(time_domain_mesh):
            [
                [[t1], [t2]],  # 点1在两个时间点的时间
                [[t1], [t2]],  # 点2在两个时间点的时间
                [[t1], [t2]]   # 点3在两个时间点的时间
            ]  # 形状: (3, 2, 1)

            连接后的结果(mesh):
            [
                [[x1,y1,t1], [x1,y1,t2]],  # 点1的时空坐标
                [[x2,y2,t1], [x2,y2,t2]],  # 点2的时空坐标
                [[x3,y3,t1], [x3,y3,t2]]   # 点3的时空坐标
            ]  # 形状: (3, 2, 3)

            max(axis=(0,1))是沿着前两个维度压缩,
            对于第三个维度的每个位置, 取所有前两个维度的最大值,
            上面例子中最后结果的形状: (3,)
            max_x = max(x1, x1, x2, x2, x3, x3)  # 第一个位置上的最大值
            max_y = max(y1, y1, y2, y2, y3, y3)  # 第二个位置上的最大值
            max_t = max(t1, t2, t1, t2, t1, t2)  # 第三个位置上的最大值

            最终结果:
            ub = [max_x, max_y, max_t]

            min(axis=(0,1))的计算过程类似, 只是取最小值而不是最大值。
        """
        mesh = np.concatenate(
            (
                self.spatial_domain_mesh,  # 形状: (空间点数, 时间点数, 空间维度)
                self.time_domain_mesh,     # 形状: (空间点数, 时间点数, 1)
            ),
            axis=-1,  # 在最后一个维度上进行连接
        )  # 形状: (空间点数, 时间点数, 空间维度+1)

        ub = mesh.max(axis=(0, 1))  # 形状: (空间维度+1,)
        lb = mesh.min(axis=(0, 1))  # 形状: (空间维度+1,)
        return lb, ub

    def on_lower_boundary(
        self, solution_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """生成下边界点的数据。

        这个方法用于获取网格下边界上的所有数据点。下边界指的是空间维度上的最小值边界。
        对于每个时间点, 我们获取下边界上的空间坐标、时间坐标和解数据。

        参数:
            solution_names: 需要获取的解数据的名称列表。

        返回:
            包含三个元素的元组:
            - spatial_domain: 下边界上的空间坐标, 形状为(时间点数, 空间维度)
                例如, 对于2D空间和3个时间点:
                [[x_min, y_min],  # 第一个时间点的每个轴的下边界
                 [x_min, y_min],  # 第二个时间点的每个轴的下边界
                 [x_min, y_min]]  # 第三个时间点的每个轴的下边界
                # ! 注意(x_min, y_min)代表了每个轴的下边界, 但不一定是域的下边界,
                # ! 因为(x_min, y_min)不一定在域中,
                # ! 但是如果域是(超)立方体, 那么(x_min, y_min)一定在域中
            - time_domain: 下边界上的时间坐标, 形状为(时间点数, 1)
                例如, 对于3个时间点:
                [[t1],  # 第一个时间点
                 [t2],  # 第二个时间点
                 [t3]]  # 第三个时间点
            - solution_domain: 下边界上的解数据字典, 每个解的形状为(时间点数, 1)
                例如, 对于解u和3个时间点:
                {'u': [[u1],  # 第一个时间点的解
                       [u2],  # 第二个时间点的解
                       [u3]]} # 第三个时间点的解
        """
        # 获取时间点的数量
        _num_time_points = self.time_domain_mesh.shape[1]

        # 创建下边界上的空间坐标
        # 使用空间维度的下界值self.lb[0:-1] (即[x_min, y_min]) 填充所有时间点
        spatial_domain = np.tile(self.lb[0:-1], (_num_time_points, 1))

        # 获取所有时间点坐标(time_domain_mesh的每行是相同的), 形状转化为(时间点数, 1)
        time_domain = self.time_domain_mesh[0, :, :]

        # 获取下边界上的解数据
        solution_domain = {}
        for solution_name in solution_names:
            # 获取第一个空间点的所有时间点的解数据
            # ! 注意: self.solution[solution_name] 是读取数据得到的
            # ! 应该保证它的第一行(索引0)对应的是下边界上的解
            # ! 这是因为在网格生成时, 下边界上的点被放在了第一行
            _solution_values = self.solution[solution_name][0, :]
            # 将一维数组转换为列向量, 保持与spatial_domain和time_domain的形状一致
            solution_domain[solution_name] = _solution_values.reshape(-1, 1)

        return spatial_domain, time_domain, solution_domain

    def on_upper_boundary(
        self, solution_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """生成上边界点的数据。

        这个方法用于获取网格上边界上的所有数据点。上边界指的是空间维度上的最大值边界。
        对于每个时间点, 我们获取上边界上的空间坐标、时间坐标和解数据。

        参数:
            solution_names: 需要获取的解数据的名称列表。

        返回:
            包含三个元素的元组:
            - spatial_domain: 上边界上的空间坐标, 形状为(时间点数, 空间维度)
                例如, 对于2D空间和3个时间点:
                [[x_max, y_max],  # 第一个时间点的每个轴的上边界
                 [x_max, y_max],  # 第二个时间点的每个轴的上边界
                 [x_max, y_max]]  # 第三个时间点的每个轴的上边界
            - time_domain: 上边界上的时间坐标, 形状为(时间点数, 1)
                例如, 对于3个时间点:
                [[t1],  # 第一个时间点
                 [t2],  # 第二个时间点
                 [t3]]  # 第三个时间点
            - solution_domain: 上边界上的解数据字典, 每个解的形状为(时间点数, 1)
                例如, 对于解u和3个时间点:
                {'u': [[u1],  # 第一个时间点的解
                       [u2],  # 第二个时间点的解
                       [u3]]} # 第三个时间点的解
        """
        # 获取时间点的数量
        _num_time_points = self.time_domain_mesh.shape[1]

        # 创建上边界上的空间坐标
        # 使用空间维度的上界值self.ub[0:-1] (即[x_max, y_max]) 填充所有时间点
        spatial_domain = np.tile(self.ub[0:-1], (_num_time_points, 1))

        # 获取所有时间点坐标(time_domain_mesh的每行是相同的), 形状转化为(时间点数, 1)
        time_domain = self.time_domain_mesh[0, :, :]

        # 获取上边界上的解数据
        solution_domain = {}
        for solution_name in solution_names:
            # 获取最后一个空间点的所有时间点的解数据
            # ! 注意: self.solution[solution_name] 是读取数据得到的
            # ! 应该保证它的最后一行(索引-1)对应的是上边界上的解
            # ! 这是因为在网格生成时, 上边界上的点被放在了最后一行
            _solution_values = self.solution[solution_name][-1, :]
            # 将一维数组转换为列向量, 保持与spatial_domain和time_domain的形状一致
            solution_domain[solution_name] = _solution_values.reshape(-1, 1)

        return spatial_domain, time_domain, solution_domain

    def on_initial_boundary(
        self, solution_names: List[str], idx: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """生成初始边界点的数据。

        这个方法用于获取网格初始时间边界上的所有数据点。初始边界指的是时间维度上的第一个时间点。
        对于每个空间点, 我们获取初始时间点上的空间坐标、时间坐标和解数据。

        参数:
            solution_names: 需要获取的解数据的名称列表。
            idx: 时间步的索引, 默认为0, 表示初始时间点。

        返回:
            包含三个元素的元组:
            - spatial_domain: 初始边界上的空间坐标, 形状为(空间点数, 空间维度)
                例如, 对于2D空间和3个空间点:
                [[x1, y1],  # 第一个空间点的坐标
                 [x2, y2],  # 第二个空间点的坐标
                 [x3, y3]]  # 第三个空间点的坐标
            - time_domain: 初始边界上的时间坐标, 形状为(空间点数, 1)
                例如, 对于3个空间点:
                [[t0],  # 第一个空间点的时间
                 [t0],  # 第二个空间点的时间
                 [t0]]  # 第三个空间点的时间
            - solution_domain: 初始边界上的解数据字典, 每个解的形状为(空间点数, 1)
                例如, 对于解u和3个空间点:
                {'u': [[u1],  # 第一个空间点的解
                       [u2],  # 第二个空间点的解
                       [u3]]} # 第三个空间点的解
        """
        # 获取初始时间点上的空间坐标
        # 使用索引操作直接获取指定时间点的空间坐标
        spatial_domain = self.spatial_domain_mesh[:, idx, :]

        # 获取初始时间点上的时间坐标
        # 所有空间点在初始时间点的时间坐标都相同
        time_domain = self.time_domain_mesh[:, idx, :]

        # 获取初始时间点上的解数据
        solution_domain = {}
        for solution_name in solution_names:
            # 获取所有空间点在初始时间点的解数据
            _solution_values = self.solution[solution_name][:, idx]
            # 将一维数组转换为列向量, 保持与spatial_domain和time_domain的形状一致
            solution_domain[solution_name] = _solution_values.reshape(-1, 1)

        return spatial_domain, time_domain, solution_domain

    def flatten_mesh(
        self, solution_names: Optional[List[str]]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """将网格数据展平以进行训练。

        这个方法将三维的网格数据展平成一列二维数组, 以便用于神经网络的训练。
        展平后的数据保持了空间点和时间点的对应关系, 每一行代表一个时空点。

        参数:
            solution_names: 需要展平的解数据的名称列表。如果为None, 则只展平空间和时间域数据。

        返回:
            包含三个元素的元组:
            - spatial_domain: 展平后的空间坐标, 形状为(空间点数×时间点数, 空间维度)
                例如, 对于2D空间、3个空间点和2个时间点:
                [[x1,y1],  # 第一个空间点的第一个时间点
                 [x1,y1],  # 第一个空间点的第二个时间点
                 [x2,y2],  # 第二个空间点的第一个时间点
                 [x2,y2],  # 第二个空间点的第二个时间点
                 [x3,y3],  # 第三个空间点的第一个时间点
                 [x3,y3]]  # 第三个空间点的第二个时间点
            - time_domain: 展平后的时间坐标, 形状为(空间点数×时间点数, 1)
                例如, 对于3个空间点和2个时间点:
                [[t1],  # 第一个空间点的第一个时间点
                 [t2],  # 第一个空间点的第二个时间点
                 [t1],  # 第二个空间点的第一个时间点
                 [t2],  # 第二个空间点的第二个时间点
                 [t1],  # 第三个空间点的第一个时间点
                 [t2]]  # 第三个空间点的第二个时间点
            - solution_domain: 展平后的解数据字典, 每个解的形状为(空间点数×时间点数, 1)
                例如, 对于解u、3个空间点和2个时间点:
                {'u': [[u1],  # 第一个空间点的第一个时间点的解
                       [u2],  # 第一个空间点的第二个时间点的解
                       [u3],  # 第二个空间点的第一个时间点的解
                       [u4],  # 第二个空间点的第二个时间点的解
                       [u5],  # 第三个空间点的第一个时间点的解
                       [u6]]} # 第三个空间点的第二个时间点的解
        """
        # 展平时间域数据, 将三维数组(time_domain_mesh)转换为二维数组
        # 形状从(空间点数, 时间点数, 1)变为(空间点数×时间点数, 1)
        time_domain = self.time_domain_mesh.reshape(-1, 1)

        # 展平空间域数据, 将三维数组(spatial_domain_mesh)转换为二维数组
        # 形状从(空间点数, 时间点数, 空间维度)变为(空间点数×时间点数, 空间维度)
        spatial_domain = self.spatial_domain_mesh.reshape(
            -1, self.spatial_dim
        )

        # 展平解数据
        # 形状从(空间点数, 时间点数)变为(空间点数×时间点数, 1)
        solution_domain = {}
        if solution_names is not None:
            for solution_name in solution_names:
                solution_domain[solution_name] = (
                    self.solution[solution_name].reshape(-1, 1)
                )

        return spatial_domain, time_domain, solution_domain

    def collection_points(
        self, n_f: int, use_lhs: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """生成配置点数据集。

        这个方法用于生成用于训练神经网络的配置点。配置点可以用于计算PDE的残差项。
        可以通过拉丁超立方采样(lhs)或使用所有网格点来生成配置点。

        参数:
            n_f: 要收集的配置点数量。
            use_lhs: 是否使用拉丁超立方采样, 默认为True。
                如果为True, 则在空间和时间域内随机采样n_f个点;
                如果为False, 则使用网格上的所有点, 共(空间点数×时间点数)个点。

        返回:
            包含两个元素的元组:
            - spatial_domain: 配置点的空间坐标, 形状为(n_f, 空间维度)
                例如, 对于2D空间和3个配置点:
                [[x1,y1],  # 第一个配置点的空间坐标
                 [x2,y2],  # 第二个配置点的空间坐标
                 [x3,y3]]  # 第三个配置点的空间坐标
            - time_domain: 配置点的时间坐标, 形状为(n_f, 1)
                例如, 对于3个配置点:
                [[t1],  # 第一个配置点的时间坐标
                 [t2],  # 第二个配置点的时间坐标
                 [t3]]  # 第三个配置点的时间坐标
        """
        if use_lhs:
            # 获取空间维度
            _sd = self.spatial_dim
            # 使用拉丁超立方采样生成形状为(n_f, 空间维度+1)的采样点矩阵
            # lhs函数返回的矩阵中每一列都是在[0,1]之间的均匀分布
            # ! 如果区域是超立方体, 那么采样点都在区域内, 否则要小心
            _sampled_points = self.lb + (self.ub - self.lb) * lhs(_sd+1, n_f)
            # 提取空间坐标(前_sd列)
            spatial_domain = _sampled_points[:, 0:_sd]
            # 提取时间坐标(最后一列)
            time_domain = _sampled_points[:, _sd:_sd+1]
        else:
            # 如果不使用lhs, 则使用网格上的所有点
            spatial_domain, time_domain, _ = self.flatten_mesh(None)
        return spatial_domain, time_domain


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
        ub: Union[Sequence, np.ndarray] = None,
        lb: Union[Sequence, np.ndarray] = None,
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
        # spatial_points: 空间点的数量(行数)
        # t_points: 时间点的数量(列数)
        # ? 为什么用的是solution的shape而不是spatial_domain的shape
        spatial_points, t_points = list(self.solution.values())[0].shape

        self.spatial_domain, self.time_domain = spatial_domain, time_domain

        # 获取网格点上的空间坐标
        self.spatial_domain_mesh = spatial_domain.generate_mesh(t_points)
        # 获取网格点上的时间坐标
        self.time_domain_mesh = time_domain.generate_mesh(spatial_points)

        self.spatial_dim = spatial_domain.spatial_dim

        # ? 需要手动传入lb和ub吗, 感觉自动计算就可以了
        if ub is not None and lb is not None:
            self.lb, self.ub = np.array(lb), np.array(ub)
        else:
            self.lb, self.ub = self.domain_bounds()


class PointCloud(MeshBase):
    """点云类。

    这个类用于表示和操作点云网格, 需要定义空间域、时间域和解的网格。
    与Mesh类不同, PointCloud类可以处理不规则网格。
    """

    def __init__(
        self,
        root_dir: str,
        read_data_fn: Callable,
        ub: Union[Sequence, np.ndarray] = None,
        lb: Union[Sequence, np.ndarray] = None,
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

        if ub is not None and lb is not None:
            self.lb, self.ub = np.array(lb), np.array(ub)
        else:
            self.lb, self.ub = self.domain_bounds()
