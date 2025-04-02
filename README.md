# PINNs-JAX 项目

<div align="center">

<img src="http://drive.google.com/uc?export=view&id=1jMpe_5_XZpozJviP7BNO9k1VuSOyfhxR" width="400">
</br>
</br>

<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rezaakb/pinns-jax/blob/main/tutorials/0-Burgers.ipynb)

<a href="https://openreview.net/pdf?id=BPFzolSSrI">[论文]</a> - <a href="https://github.com/rezaakb/pinns-torch">[PyTorch版本]</a> - <a href="https://github.com/rezaakb/pinns-tf2">[TensorFlow v2版本]</a> - <a href="https://github.com/maziarraissi/PINNs">[TensorFlow v1版本]</a>
</div>

## 项目描述

本文介绍了"PINNs-JAX"，这是一个创新的实现，利用JAX框架来发挥XLA编译器的独特功能。这种方法旨在提高PINN应用中的计算效率和灵活性。

<div align="center">
<img src="http://drive.google.com/uc?export=view&id=1bhiyum1xh2KnLOnMeTjevBgOA8m4Qkel" width="1000">
</br>
<em>每个子图对应一个问题，其迭代次数显示在顶部。对数x轴显示相对于原始TensorFlow v1代码的加速因子，y轴显示平均相对误差。</em>
</div>
</br>

更多信息，请参考我们的论文：

<a href="https://openreview.net/pdf?id=BPFzolSSrI">比较不同框架下的PINNs：JAX、TensorFlow和PyTorch。</a> Reza Akbarian Bafghi, 和 Maziar Raissi. AI4DiffEqtnsInSci, ICLR, 2024.

## 安装

PINNs-JAX需要安装以下依赖：

- [JAX](https://jax.readthedocs.io/en/latest/installation.html) >= 0.4.16
- [Hydra](https://hydra.cc/docs/intro/) >= 1.3

然后，您可以通过[pip]安装PINNs-JAX：

```bash
pip install pinnsjax
```

如果您打算引入新功能或修改代码，我们建议复制仓库并设置本地安装：

```bash
git clone https://github.com/rezaakb/pinns-jax
cd pinns-jax

# [可选] 创建conda环境
conda create -n myenv python=3.9
conda activate myenv

# 安装包
pip install -e .
```

## 快速开始

在[examples](examples)文件夹中探索各种已实现的示例。要运行特定代码，例如Allen Cahn PDE的代码，您可以使用：

```bash
python examples/ac/train.py
```

您可以使用指定的配置文件来训练模型，例如[examples/ac/configs/config.yaml](examples/ac/configs/config.yaml)中的配置。参数可以直接从命令行覆盖。例如：

```bash
python examples/ac/train.py trainer.max_epochs=20
```

使用我们的包有两种主要方式：

- 使用Hydra实现您的训练结构，如我们提供的示例所示。
- 直接使用我们的包来解决您的自定义问题。

关于直接使用我们的包来解决Burgers PDE连续前向问题的实用指南，请参考我们的教程：[tutorials/0-Burgers.ipynb](tutorials/0-Burgers.ipynb)。

## 数据

数据位于服务器上，运行每个示例时会自动下载。

## 贡献

由于这是我们的第一个版本，可能还有改进和bug修复的空间。我们非常重视社区贡献。如果您在使用这个库时发现任何问题、缺失功能或异常行为，请随时在GitHub上提出issue或提交pull request。如有任何问题、建议或反馈，请发送给[Reza Akbarian Bafghi](https://www.linkedin.com/in/rezaakbarian/)：[reza.akbarianbafghi@colorado.edu](mailto:reza.akbarianbafghi@colorado.edu)。

## 许可证

根据[MIT]许可证条款分发，"pinnsjax"是免费的开源软件。

## 资源

我们使用[这个模板](https://github.com/ashleve/lightning-hydra-template)来开发这个包，借鉴了其结构和设计原则。为了更深入的理解，我们建议访问他们的GitHub仓库。

## 引用

```
@inproceedings{
bafghi2024comparing,
title={Comparing {PINN}s Across Frameworks: {JAX}, TensorFlow, and PyTorch},
author={Reza Akbarian Bafghi and Maziar Raissi},
booktitle={ICLR 2024 Workshop on AI4DifferentialEquations In Science},
year={2024},
url={https://openreview.net/forum?id=BPFzolSSrI}
}
```

# PINNs-JAX 项目

## 环境设置

本项目提供了两种环境配置，分别适用于 GPU 和 CPU 环境：

### GPU 环境（适用于有 CUDA 支持的机器）

使用 pip：
```bash
pip install -r requirements-gpu.txt
```

使用 conda：
```bash
conda env create -f environment-gpu.yaml
conda activate pinnsgpu
```

### CPU 环境（适用于 Mac 和没有 GPU 的机器）

使用 pip：
```bash
pip install -r requirements-cpu.txt
```

使用 conda：
```bash
conda env create -f environment-cpu.yaml
conda activate pinnscpu
```

### Apple Silicon (M系列) Mac 用户

如果您使用的是 M 系列芯片的 Mac，可以通过取消 `requirements-cpu.txt` 或 `environment-cpu.yaml` 中的相关注释来启用 Metal 加速支持。

## 注意事项

- JAX 库同时支持 CPU 和 GPU 计算，本项目配置会根据您的环境自动选择合适的后端
- 对于 GPU 用户，请确保已正确安装 CUDA 驱动程序
