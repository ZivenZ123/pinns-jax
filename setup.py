#!/usr/bin/env python

"""PINNs-JAX 包的安装配置文件。"""

from setuptools import find_packages, setup

setup(
    name="pinnsjax",
    version="0.0.2",
    description="An implementation of PINNs in JAX.",
    author="Reza Akbarian Bafghi",
    author_email="reza.akbarianbafghi@coloardo.edu",
    url="https://github.com/rezaakb/pinns-jax",
    install_requires=[
        "hydra-core",
        "scipy",
        "pyDOE",
        "matplotlib",
        "rootutils",
        "rich",
        "tqdm",
        "requests"
    ],
    packages=find_packages(),
    # 在安装包后，用于自定义在终端中可用的全局命令
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
