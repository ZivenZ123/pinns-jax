"""工具函数模块，提供各种通用功能支持。"""

import os
import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, Tuple

import numpy as np
import requests
import scipy
from omegaconf import DictConfig, open_dict

from pinnsjax.utils import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)


def extras(cfg: DictConfig) -> None:
    """在任务开始前应用可选工具。

    工具:
        - 忽略python警告
        - 从命令行设置标签
        - 使用Rich库打印配置

    :param cfg: 包含配置树的DictConfig对象。
    """
    # 如果没有`extras`配置则返回
    if not cfg.get("extras"):
        log.warning("未找到Extras配置! <cfg.extras=null>")
        return

    # 禁用python警告
    if cfg.extras.get("ignore_warnings"):
        log.info("正在禁用Python警告! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # 如果配置中没有提供标签，提示用户从命令行输入标签
    if cfg.extras.get("enforce_tags"):
        log.info("正在强制使用标签! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # 使用Rich库美观打印配置树
    if cfg.extras.get("print_config"):
        log.info("正在使用Rich打印配置树! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """控制任务函数执行失败行为的可选装饰器。

    此包装器可用于：
        - 确保即使任务函数引发异常，日志记录器也会被关闭（防止多次运行失败）
        - 将异常保存到`.log`文件
        - 在`logs/`文件夹中用专用文件标记运行失败（以便我们稍后可以找到并重新运行）
        - 等等（根据您的需求进行调整）

    示例:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: 要包装的任务函数。

    :return: 包装后的任务函数。
    """

    def wrap(
        cfg: DictConfig,
        read_data_fn: Callable,
        pde_fn: Callable,
        output_fn: Callable
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # 执行任务
        try:
            metric_dict, object_dict = task_func(
                cfg=cfg,
                read_data_fn=read_data_fn,
                pde_fn=pde_fn,
                output_fn=output_fn
            )

        # 发生异常时要做的事情
        except Exception as ex:
            # 将异常保存到`.log`文件
            log.exception("")

            # 某些超参数组合可能无效或导致内存溢出错误
            # 因此，当使用如Optuna之类的超参数搜索插件时，您可能需要禁用
            # 引发以下异常，以避免多次运行失败
            raise ex

        # 无论成功还是异常都要执行的操作
        finally:
            # 在终端显示输出目录路径
            log.info(f"输出目录: {cfg.paths.output_dir}")

            # 始终关闭wandb运行（即使发生异常，也不会导致多次运行失败）
            if find_spec("wandb"):  # 检查是否安装了wandb
                import wandb

                if wandb.run:
                    log.info("正在关闭wandb! ")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_names: list) -> float:
    """安全获取在LightningModule中记录的指标值。

    :param metric_dict: 包含指标值的字典。
    :param metric_name: 要获取的指标名称。
    :return: 指标的值。
    """
    for type_metric, list_metrics in metric_names.items():
        if type_metric == "extra_variables":
            prefix = ""
        elif type_metric == "error":
            prefix = "val/error_"

        for metric_name in list_metrics:
            metric_name = f"{prefix}{metric_name}"

            if not metric_name:
                log.info("指标名称为空！跳过指标值获取...")
                continue

            if metric_name not in metric_dict:
                log.info(
                    f"未找到指标值！<metric_name={metric_name}>\n"
                    "确保在LightningModule中记录的指标名称正确！\n"
                    "确保`hparams_search`配置中的`optimized_metric`名称正确！"
                )
            else:
                metric_value = metric_dict[metric_name].item()
                log.info(f"获取到指标值! <{metric_name}={metric_value}>")

    return metric_value


def download_file(path, folder_name, filename):
    """从给定URL下载文件并保存到指定路径。

    :param path: 文件应保存的路径。
    :param folder_name: 服务器上包含文件的文件夹名称。
    :param filename: 要下载的文件名。
    """

    url = f"https://storage.googleapis.com/pinns/{folder_name}/{filename}"
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as file:
            file.write(response.content)
        log.info("文件下载成功。")
    else:
        FileNotFoundError("文件下载失败。")


def load_data_txt(root_path, file_name):
    """从文件加载文本数据，如果文件不存在则下载。

    :param root_path: 数据文件应该位于的根目录。
    :param file_name: 数据文件的名称。
    :return: 作为numpy数组加载的数据。
    """
    path = os.path.join(root_path, file_name)
    if os.path.exists(path):
        log.info("权重文件可用。")
    else:
        download_file(path, "irk_weights", file_name)

    return np.float32(np.loadtxt(path, ndmin=2))


def load_data(root_path, file_name):
    """从MATLAB .mat文件加载数据, 如果文件不存在则下载。

    :param root_path: 数据文件应该位于的根目录。
    :param file_name: 数据文件的名称。
    :return: 使用scipy.io.loadmat函数加载的数据。
    """

    path = os.path.join(root_path, file_name)
    if os.path.exists(path):
        log.info("数据可用。")
    else:
        download_file(path, "data", file_name)

    return scipy.io.loadmat(path)
