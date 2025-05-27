import sys
import time
import numpy as np


# 情况1：不带参数或只指定分布参数
# 返回单个浮点数
single_value = np.random.normal()
print(f"类型: {type(single_value)}")  # <class 'numpy.float64'>

# 指定均值和标准差，仍返回单个值
single_value_2 = np.random.normal(loc=100, scale=15)
print(f"类型: {type(single_value_2)}")  # <class 'numpy.float64'>

# 情况2：指定size参数
# 返回NumPy数组
array_1d = np.random.normal(size=5)
print(f"类型: {type(array_1d)}")     # <class 'numpy.ndarray'>
print(f"数据类型: {array_1d.dtype}")  # float64

# 返回多维数组
array_2d = np.random.normal(size=(3, 4))
print(f"类型: {type(array_2d)}")     # <class 'numpy.ndarray'>



# 创建两种类型的数值
py_float = 3.14159
np_float = np.float64(3.14159)

# 第一个区别：类型本身
print("=== 类型对比 ===")
print(f"Python float 类型: {type(py_float)}")
print(f"NumPy float64 类型: {type(np_float)}")
print(f"它们是同一类型吗? {type(py_float) == type(np_float)}")

# 精度和表示
print("\n=== 精度对比 ===")
# 两者实际上有相同的精度（都是IEEE 754双精度）
large_py = float(1e308)
large_np = np.float64(1e308)

print(f"Python float 最大值: {sys.float_info.max}")
print(f"NumPy float64 最大值: {np.finfo(np.float64).max}")
print(f"它们相等吗? {sys.float_info.max == np.finfo(np.float64).max}")

print("\n=== 方法和属性对比 ===")
# 查看可用的方法
py_methods = dir(py_float)
np_methods = dir(np_float)

# NumPy特有的方法
np_only_methods = set(np_methods) - set(py_methods)
print(f"NumPy float64 独有的方法数量: {len(np_only_methods)}")
print("一些例子:", list(np_only_methods)[:10])

# 实际使用这些方法
print("\n=== 使用NumPy特有方法 ===")
np_value = np.float64(3.7)
print(f"转为整数（NumPy方法）: {np_value.astype(int)}")
print(f"获取字节表示: {np_value.tobytes()}")
print(f"查看形状属性: {np_value.shape}")  # 标量的形状是空元组


print("\n=== 数组操作行为 ===")
# 创建数组时的差异
arr_from_py = np.array([1.1, 2.2, 3.3], dtype=float)
arr_from_np = np.array([1.1, 2.2, 3.3], dtype=np.float64)

print(f"从Python float创建: {arr_from_py.dtype}")
print(f"从NumPy float64创建: {arr_from_np.dtype}")

# 数学运算的差异
py_val = 5.0
np_val = np.float64(5.0)

# Python float与数组运算
result_py = arr_from_np + py_val
print(f"\n数组 + Python float: {result_py}")
print(f"结果类型: {type(result_py[0])}")

# NumPy float与数组运算
result_np = arr_from_np + np_val
print(f"数组 + NumPy float: {result_np}")
print(f"结果类型: {type(result_np[0])}")




print("\n=== 性能对比 ===")
n = 1000000

# Python float列表操作
py_list = [float(i) for i in range(n)]
start = time.time()
py_sum = sum(py_list)
py_time = time.time() - start

# NumPy数组操作
np_array = np.arange(n, dtype=np.float64)
start = time.time()
np_sum = np.sum(np_array)
np_time = time.time() - start

print(f"Python float列表求和时间: {py_time:.4f}秒")
print(f"NumPy float64数组求和时间: {np_time:.4f}秒")
print(f"NumPy快了约 {py_time/np_time:.1f} 倍")


print("\n=== 内存使用对比 ===")
# 单个值的内存使用
py_float = 3.14159
np_float = np.float64(3.14159)

print(f"Python float 内存大小: {sys.getsizeof(py_float)} 字节")
print(f"NumPy float64 内存大小: {sys.getsizeof(np_float)} 字节")

# 在数组中的内存效率
n = 1000
py_list = [float(i) for i in range(n)]
np_array = np.arange(n, dtype=np.float64)

print(f"\n包含{n}个元素:")
print(f"Python列表内存: {sys.getsizeof(py_list)} 字节")
print(f"NumPy数组内存: {np_array.nbytes} 字节（只计算数据）")
print(f"NumPy数组总内存: {sys.getsizeof(np_array)} 字节（包括开销）")



print("\n=== 类型转换 ===")
# NumPy到Python
np_val = np.float64(2.718)
py_val = float(np_val)
print(f"NumPy转Python: {py_val}, 类型: {type(py_val)}")

# Python到NumPy
py_val2 = 3.14159
np_val2 = np.float64(py_val2)
print(f"Python转NumPy: {np_val2}, 类型: {type(np_val2)}")

# 自动转换示例
mixed_calc = np_val + py_val  # NumPy float64 + Python float
print(f"混合计算结果: {mixed_calc}, 类型: {type(mixed_calc)}")



# 场景1：简单的数学计算 - 使用Python float
def calculate_circle_area(radius):
    """计算圆的面积 - 简单计算用Python float"""
    return 3.14159 * radius ** 2

# 场景2：科学计算和数据分析 - 使用NumPy
def analyze_data(data):
    """数据分析 - 使用NumPy以获得更好的性能"""
    np_data = np.array(data, dtype=np.float64)
    return {
        'mean': np_data.mean(),
        'std': np_data.std(),
        'max': np_data.max(),
        'min': np_data.min()
    }

# 测试对比
radius = 5.0
area_py = calculate_circle_area(radius)

data = [np.random.normal(100, 15) for _ in range(1000)]
stats = analyze_data(data)

print(f"\n=== 实际应用示例 ===")
print(f"圆面积（Python float）: {area_py}")
print(f"数据统计（NumPy）: {stats}")



# 一个综合示例展示两者的协作
def scientific_calculation(x, y):
    """展示两种float类型的协同工作"""
    # 开始时可能是Python float
    if isinstance(x, float):
        print("输入是Python float")
    
    # 转换为NumPy进行复杂计算
    np_x = np.float64(x)
    np_y = np.float64(y)
    
    # 使用NumPy的高级功能
    result = np.sqrt(np_x**2 + np_y**2)  # 勾股定理
    
    # 如果需要，可以转回Python float
    return float(result) if isinstance(x, float) else result

# 测试
print("\n=== 协同工作示例 ===")
print(scientific_calculation(3.0, 4.0))  # Python float输入
print(scientific_calculation(np.float64(3), np.float64(4)))  # NumPy输入