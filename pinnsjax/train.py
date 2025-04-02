"""训练模块，提供模型训练、验证、测试和预测的主要功能。"""

from typing import Any, Callable, Dict, Optional, Tuple, Union

import time
import hydra
import rootutils
import numpy as np
from omegaconf import DictConfig, OmegaConf

from pinnsjax.trainer import Trainer
from pinnsjax import utils
from pinnsjax.models import PINNModule
from pinnsjax.data import (
    Interval,
    Mesh,
    PointCloud,
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
    """训练模型。还可以使用训练过程中获得的最佳权重对测试集进行评估。

    此方法被可选的@task_wrapper装饰器包装, 该装饰器控制失败时的行为。
    对于多次运行、保存崩溃信息等非常有用。

    :param cfg: 由Hydra组合的DictConfig配置。
    :return: 包含指标和所有实例化对象的字典的元组。
    """

    # 为pytorch、numpy和python.random中的随机数生成器设置种子
    if cfg.get("seed"):
        np.random.seed(cfg.seed)

    if cfg.get("time_domain"):
        log.info("实例化时间域 <%s>", OmegaConf.get_type(cfg.time_domain))
        td: TimeDomain = hydra.utils.instantiate(cfg.time_domain)

    if cfg.get("spatial_domain"):
        log.info("实例化空间域 <%s>", OmegaConf.get_type(cfg.spatial_domain))
        sd: Union[Interval, Rectangle, RectangularPrism] = (
            hydra.utils.instantiate(cfg.spatial_domain)
        )

    log.info("实例化网格 <%s>", OmegaConf.get_type(cfg.mesh))
    mesh_type = cfg.mesh.get("_target_", "pinnsjax.data.Mesh")
    if mesh_type == "pinnsjax.data.Mesh":
        mesh: Mesh = hydra.utils.instantiate(
            cfg.mesh,
            time_domain=td,
            spatial_domain=sd,
            read_data_fn=read_data_fn
        )
    elif mesh_type == "pinnsjax.data.PointCloud":
        mesh: PointCloud = hydra.utils.instantiate(
            cfg.mesh,
            read_data_fn=read_data_fn
        )
    else:
        raise ValueError(
            f"网格应在配置文件中定义，但找到：{mesh_type}"
        )

    train_datasets = []
    for i, (dataset_dic) in enumerate(cfg.train_datasets):
        for key, dataset in dataset_dic.items():
            dataset_type = dataset.get("_target_", "pinnsjax.data.MeshSampler")
            log.info("实例化训练数据集 %d: <%s>", i+1, dataset_type)
            train_datasets.append(
                hydra.utils.instantiate(dataset)(
                    mesh=mesh,
                    dtype=cfg.dtype
                )
            )

    val_dataset = None
    if cfg.get("val_dataset"):
        for i, dataset_dic in enumerate(cfg.val_dataset):
            for _, dataset in dataset_dic.items():
                dataset_type = dataset.get(
                    "_target_",
                    "pinnsjax.data.MeshSampler"
                )
                log.info("实例化验证数据集 %d: <%s>", i+1, dataset_type)
                val_dataset = hydra.utils.instantiate(dataset)(
                    mesh=mesh,
                    dtype=cfg.dtype
                )

    test_dataset = None
    if cfg.get("test_dataset"):
        for i, dataset_dic in enumerate(cfg.test_dataset):
            for _, dataset in dataset_dic.items():
                dataset_type = dataset.get(
                    "_target_",
                    "pinnsjax.data.MeshSampler"
                )
                log.info("实例化测试数据集 %d: <%s>", i+1, dataset_type)
                test_dataset = hydra.utils.instantiate(dataset)(
                    mesh=mesh,
                    dtype=cfg.dtype
                )

    pred_dataset = None
    if cfg.get("pred_dataset"):
        for i, dataset_dic in enumerate(cfg.pred_dataset):
            for _, dataset in dataset_dic.items():
                dataset_type = dataset.get(
                    "_target_",
                    "pinnsjax.data.MeshSampler"
                )
                log.info("实例化预测数据集 %d: <%s>", i+1, dataset_type)
                pred_dataset = hydra.utils.instantiate(dataset)(
                    mesh=mesh,
                    dtype=cfg.dtype
                )

    data_type = cfg.data.get("_target_", "pinnsjax.data.PINNDataModule")
    log.info("实例化数据模块 <%s>", data_type)
    datamodule: PINNDataModule = hydra.utils.instantiate(
        cfg.data,
        train_datasets=train_datasets,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        pred_dataset=pred_dataset,
        batch_size=cfg.get("batch_size"),
    )

    # 初始化网络
    net_type = cfg.net.get("_target_", "pinnsjax.models.FCN")
    log.info("实例化神经网络 <%s>", net_type)

    if net_type == "pinnsjax.models.FCN":
        net = hydra.utils.instantiate(cfg.net)(
            lb=mesh.lb,
            ub=mesh.ub,
            dtype=cfg.dtype
        )
    elif net_type == "pinnsjax.models.NetHFM":
        net = hydra.utils.instantiate(cfg.net)(
            mean=train_datasets[0].mean,
            std=train_datasets[0].std
        )
    else:
        raise ValueError(f"不支持的神经网络类型：{net_type}")

    model_type = cfg.model.get("_target_", "pinnsjax.models.PINNModule")
    log.info("实例化模型 <%s>", model_type)

    model: PINNModule = hydra.utils.instantiate(cfg.model)(
        net=net,
        pde_fn=pde_fn,
        output_fn=output_fn
    )

    trainer_type = cfg.trainer.get("_target_", "pinnsjax.trainer.Trainer")
    log.info("实例化训练器 <%s>", trainer_type)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "trainer": trainer,
    }

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

    if cfg.get("val"):
        log.info("开始验证！")
        trainer.validate(model=model, datamodule=datamodule)

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("开始测试！")
        ckpt_path = None  # 检查点回调中保存的最佳模型路径
        trainer = hydra.utils.instantiate(cfg.trainer)
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    test_metrics = trainer.callback_metrics

    if cfg.get("plotting"):
        try:
            log.info("正在绘图")
            preds_dict = trainer.predict(
                model=model, datamodule=datamodule
            )
            for sol_key, pred in preds_dict.items():
                preds_dict[sol_key] = pred
            hydra.utils.instantiate(
                cfg.plotting,
                mesh=mesh,
                preds=preds_dict,
                train_datasets=train_datasets,
                val_dataset=val_dataset,
                file_name=cfg.paths.output_dir,
            )()
        except KeyboardInterrupt:
            print("绘图停止。")

    # 合并训练和测试指标
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="train.yaml"
)
def main(cfg: DictConfig) -> Optional[float]:
    """主要的训练入口。

    :param cfg: 由 Hydra 组合的 DictConfig 配置。
    :return: 可选的 float 类型数值，表示优化后的指标。
    """
    # 应用额外工具
    # (例如：如果 cfg 中未提供 tag，则询问；打印 cfg 树；等等)
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
