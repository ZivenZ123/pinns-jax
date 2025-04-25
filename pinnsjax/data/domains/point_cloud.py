"""点云数据相关的数据类定义。

该模块提供了用于存储和处理点云数据的数据结构。
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class PointCloudData:
    """用于存储点云数据的数据类, 包括空间点、时间值和解数据。

    这是一个替代空间和时间类的类, 应该与 `PointCloud` 类一起使用。

    属性:
        spatial: 空间点网格的数组列表
        time: 时间点网格的数组列表
        solution: 包含解数据的字典
    """

    spatial: List[np.ndarray]
    time: List[np.ndarray]
    solution: Dict[str, np.ndarray]
