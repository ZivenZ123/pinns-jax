"""训练模块, 提供模型训练、验证、测试和预测的主要功能。"""

from typing import Any, Callable, Dict, Optional, Tuple, Union

import time
import hydra
import rootutils
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from pinnsjax.trainer import Trainer
from pinnsjax import utils
from pinnsjax.models import PINNModule
from pinnsjax.data import (
    Interval,
    Rectangle,
    RectangularPrism,
    TimeDomain,
    PINNDataModule,
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = utils.get_pylogger(__name__)

OmegaConf.register_new_resolver("eval", eval)


@utils.task_wrapper
def train(
    cfg: DictConfig,
    read_data_fn: Callable,
    pde_fn: Callable,
    output_fn: Callable = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """训练物理信息神经网络(PINN)模型。

    此函数是训练过程的主要入口, 负责:
    1. 根据配置初始化时间域、空间域和网格
    2. 创建训练、验证和测试数据集
    3. 实例化神经网络模型和训练器
    4. 执行训练、验证和测试过程
    5. 可选地进行预测和可视化

    此方法被可选的@task_wrapper装饰器包装, 该装饰器提供了以下功能:
    - 控制训练失败时的行为
    - 支持多次运行
    - 保存崩溃信息
    - 提供实验跟踪功能

    参数:
        cfg: 由Hydra组合的DictConfig配置对象, 包含所有训练相关的配置参数
        read_data_fn: 用于读取数据的函数
        pde_fn: 偏微分方程函数
        output_fn: 可选的输出处理函数

    返回:
        包含以下内容的元组:
        - 训练和测试过程中的指标字典
        - 包含所有实例化对象的字典, 包括:
            * cfg: 配置对象
            * datamodule: 数据模块
            * model: 神经网络模型
            * trainer: 训练器
    """

    # ==================== 0. 初始化随机种子 ====================
    np.random.seed(cfg.seed)

    # ==================== 1. 初始化域和网格 ====================
    # 1.1 初始化时间域
    log.info("实例化时间域 <%s>", cfg.time_domain.get("_target_"))
    td: TimeDomain = instantiate(cfg.time_domain)

    # 1.2 初始化空间域
    log.info("实例化空间域 <%s>", cfg.spatial_domain.get("_target_"))
    sd: Union[Interval, Rectangle, RectangularPrism] = (
        instantiate(cfg.spatial_domain)
    )

    # 1.3 初始化网格
    log.info("实例化网格 <%s>", cfg.mesh.get("_target_"))
    mesh_type = cfg.mesh.get("_target_")

    def create_mesh():
        return instantiate(
            cfg.mesh,
            time_domain=td,
            spatial_domain=sd,
            read_data_fn=read_data_fn
        )

    def create_point_cloud():
        return instantiate(
            cfg.mesh,
            read_data_fn=read_data_fn
        )

    mesh_types = {
        "pinnsjax.data.Mesh": create_mesh,
        "pinnsjax.data.PointCloud": create_point_cloud
    }

    mesh = mesh_types[mesh_type]()

    # ==================== 2. 创建数据集 ====================
    # 2.1 创建训练数据集
    train_datasets = []
    for dataset_name, dataset in cfg.train_datasets.items():
        dataset_type = dataset.get("_target_", "pinnsjax.data.MeshSampler")
        log.info("实例化训练数据集 <%s>: <%s>", dataset_name, dataset_type)
        train_datasets.append(
            instantiate(dataset)(
                mesh=mesh,
                dtype=cfg.dtype
            )
        )

    # 2.2 创建验证数据集
    val_dataset = None
    if cfg.get("val_dataset"):
        for dataset_name, dataset in cfg.val_dataset.items():
            dataset_type = dataset.get("_target_", "pinnsjax.data.MeshSampler")
            log.info("实例化验证数据集 <%s>: <%s>", dataset_name, dataset_type)
            val_dataset = instantiate(dataset)(
                mesh=mesh,
                dtype=cfg.dtype
            )

    # 2.3 创建测试数据集
    test_dataset = None
    if cfg.get("test_dataset"):
        for dataset_name, dataset in cfg.test_dataset.items():
            dataset_type = dataset.get("_target_", "pinnsjax.data.MeshSampler")
            log.info("实例化测试数据集 <%s>: <%s>", dataset_name, dataset_type)
            test_dataset = instantiate(dataset)(
                mesh=mesh,
                dtype=cfg.dtype
            )

    # 2.4 创建预测数据集
    pred_dataset = None
    if cfg.get("pred_dataset"):
        for dataset_name, dataset in cfg.pred_dataset.items():
            dataset_type = dataset.get("_target_", "pinnsjax.data.MeshSampler")
            log.info("实例化预测数据集 <%s>: <%s>", dataset_name, dataset_type)
            pred_dataset = instantiate(dataset)(
                mesh=mesh,
                dtype=cfg.dtype
            )

    # 2.5 创建数据模块
    data_type = cfg.data.get("_target_", "pinnsjax.data.PINNDataModule")
    log.info("实例化数据模块 <%s>", data_type)
    datamodule: PINNDataModule = instantiate(
        cfg.data,
        train_datasets=train_datasets,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        pred_dataset=pred_dataset,
        batch_size=cfg.get("batch_size"),
    )

    # ==================== 3. 初始化模型 ====================
    # 3.1 初始化神经网络
    net_type = cfg.net.get("_target_", "pinnsjax.models.FCN")
    log.info("实例化神经网络 <%s>", net_type)

    if net_type == "pinnsjax.models.FCN":
        net = instantiate(cfg.net)(
            lb=mesh.lb,
            ub=mesh.ub,
            dtype=cfg.dtype
        )
    elif net_type == "pinnsjax.models.NetHFM":
        net = instantiate(cfg.net)(
            mean=train_datasets[0].mean,
            std=train_datasets[0].std
        )
    else:
        raise ValueError(f"不支持的神经网络类型: {net_type}")

    # 3.2 初始化PINN模型
    model_type = cfg.model.get("_target_", "pinnsjax.models.PINNModule")
    log.info("实例化模型 <%s>", model_type)

    model: PINNModule = instantiate(cfg.model)(
        net=net,
        pde_fn=pde_fn,
        output_fn=output_fn
    )

    # 3.3 初始化训练器
    trainer_type = cfg.trainer.get("_target_", "pinnsjax.trainer.Trainer")
    log.info("实例化训练器 <%s>", trainer_type)
    trainer: Trainer = instantiate(cfg.trainer)

    # 3.4 创建对象字典
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "trainer": trainer,
    }

    # ==================== 4. 训练过程 ====================
    # 4.1 执行训练
    if cfg.get("train"):
        log.info("开始训练！")
        start_time = time.time()
        try:
            trainer.fit(model=model, datamodule=datamodule)
        except KeyboardInterrupt:
            print("训练停止。")
        log.info("耗时: %f", time.time() - start_time)
    log.info(
        "平均时间: %f - %f - %f",
        np.median(trainer.time_list[-5:]),
        np.median(trainer.time_list[-100:-10]),
        np.median(trainer.time_list)
    )

    # 4.2 执行验证
    if cfg.get("val"):
        log.info("开始验证！")
        trainer.validate(model=model, datamodule=datamodule)

    train_metrics = trainer.callback_metrics

    # 4.3 执行测试
    if cfg.get("test"):
        log.info("开始测试！")
        ckpt_path = None  # 检查点回调中保存的最佳模型路径
        trainer = instantiate(cfg.trainer)
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    test_metrics = trainer.callback_metrics

    # ==================== 5. 预测和可视化 ====================
    if cfg.get("plotting"):
        try:
            log.info("正在绘图")
            preds_dict = trainer.predict(
                model=model, datamodule=datamodule
            )
            for sol_key, pred in preds_dict.items():
                preds_dict[sol_key] = pred
            instantiate(
                cfg.plotting,
                mesh=mesh,
                preds=preds_dict,
                train_datasets=train_datasets,
                val_dataset=val_dataset,
                file_name=cfg.paths.output_dir,
            )()
        except KeyboardInterrupt:
            print("绘图停止。")

    # ==================== 6. 返回结果 ====================
    # 合并训练和测试指标
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="train.yaml"
)
def main(cfg: DictConfig) -> Optional[float]:
    """物理信息神经网络(PINN)训练的主入口函数。

    此函数是使用Hydra配置系统启动训练过程的主要入口点。它负责:
    1. 应用额外的工具和配置
    2. 调用train函数进行模型训练
    3. 获取并返回优化后的指标值

    此函数被@hydra.main装饰器包装, 该装饰器提供了以下功能:
    - 自动加载配置文件
    - 管理配置路径
    - 处理命令行参数

    参数:
        cfg: 由Hydra组合的DictConfig配置对象, 包含所有训练相关的配置参数

    返回:
        可选的float类型数值, 表示优化后的指标值。如果配置中未指定优化指标, 则返回None
    """
    # 应用额外工具
    # (例如: 如果 cfg 中未提供 tag, 则询问; 打印 cfg 树; 等等)
    utils.extras(cfg)

    # 训练模型
    metric_dict, _ = train(cfg)

    # 安全地获取用于 Hydra 超参数优化的指标数值
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # 返回优化后的指标
    return metric_value


if __name__ == "__main__":
    main()
