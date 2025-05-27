import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# |---------- 实验1 ----------|
# 创建初始密钥
master_key = jax.random.PRNGKey(2025)
print(f"master_key的值: {master_key}")

# 生成多个独立的正态分布样本
N_SAMPLES = 10000

# 1. 标准正态分布
key, subkey = jax.random.split(master_key)
print(f"key的值: {key}")
print(f"subkey的值: {subkey}")
standard_normal = jax.random.normal(subkey, shape=(N_SAMPLES,))

# 2. 自定义正态分布（均值=100，标准差=15）
key, subkey = jax.random.split(key)
mu, sigma = 100.0, 15.0
custom_normal = mu + sigma * jax.random.normal(subkey, shape=(N_SAMPLES,))

# 3. 多维正态分布
key, subkey = jax.random.split(key)
multi_dim = jax.random.normal(subkey, shape=(1000, 3))  # 1000个3维向量

# 可视化
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(standard_normal, bins=50, density=True, alpha=0.7)
plt.title('Standard Normal Distribution')

plt.subplot(1, 3, 2)
plt.hist(custom_normal, bins=50, density=True, alpha=0.7)
plt.title('Custom Normal Distribution(μ=100, σ=15)')

plt.subplot(1, 3, 3)
plt.scatter(multi_dim[:100, 0], multi_dim[:100, 1], alpha=0.5)
plt.title('2D Normal Distribution(First 100 Points)')

plt.tight_layout()
plt.show()

# |---------- 实验2: PRNGKey的内部结构 ----------|
key = jax.random.PRNGKey(42)
print(f"Key的类型: {type(key)}")
print(f"Key的形状: {key.shape}")
print(f"Key的值: {key}")

# PRNGKey实际上是一个形状为(2,)的uint32数组
print(f"Key的数据类型: {key.dtype}")


# ---------- 实验3: 实际应用中的PRNGKey管理 ----------|
def initialize_neural_network(key, layer_sizes):
    """初始化一个神经网络的权重"""
    keys = jax.random.split(key, len(layer_sizes) - 1)

    weights = []
    for i in range(len(layer_sizes) - 1):
        # Xavier初始化
        fan_in = layer_sizes[i]
        fan_out = layer_sizes[i + 1]
        std = jnp.sqrt(2.0 / (fan_in + fan_out))

        w = jax.random.normal(keys[i], shape=(fan_in, fan_out)) * std
        weights.append(w)

    return weights


# 使用示例
master_key = jax.random.PRNGKey(2025)
network_weights = initialize_neural_network(master_key, [784, 256, 128, 10])

# 每次使用相同的master_key会得到相同的初始化
weights_again = initialize_neural_network(master_key, [784, 256, 128, 10])

# 验证确实相同
for i, (w1, w2) in enumerate(zip(network_weights, weights_again)):
    print(f"层{i}: 权重相同 = {jnp.allclose(w1, w2)}")
