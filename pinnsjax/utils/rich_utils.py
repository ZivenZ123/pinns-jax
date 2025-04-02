"""用于配置树打印和标签管理的工具函数."""

from pathlib import Path
from typing import Sequence

import rich
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.prompt import Prompt

from pinnsjax.utils import pylogger

log = pylogger.get_pylogger(__name__)


def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "data",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """使用 Rich 库以树形结构打印 DictConfig 的内容.

    :param cfg: 由 Hydra 组成的 DictConfig.
    :param print_order: 决定配置组件的打印顺序. 默认为 ``("data", "model",
    "callbacks", "logger", "trainer", "paths", "extras")``.
    :param resolve: 是否解析 DictConfig 的引用字段. 默认为 ``False``.
    :param save_to_file: 是否将配置导出到 hydra 输出文件夹. 默认为 ``False``.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # 将 `print_order` 中的字段添加到队列
    for field in print_order:
        if field in cfg:
            queue.append(field)
        else:
            log.warning(
                f"配置中未找到字段 '{field}'. 跳过 '{field}' 配置打印..."
            )

    # 将所有其他字段添加到队列 (未在 `print_order` 中指定)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # 从队列生成配置树
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # 打印配置树
    rich.print(tree)

    # 将配置树保存到文件
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """如果配置中未提供标签，则提示用户从命令行输入标签.

    :param cfg: 由 Hydra 组成的 DictConfig.
    :param save_to_file: 是否将标签导出到 hydra 输出文件夹. 默认为 ``False``.
    """
    if not cfg.get("tags"):
        if "id" in HydraConfig().cfg.hydra.job:
            raise ValueError("在启动多运行之前请指定标签!")

        log.warning("配置中未提供标签. 提示用户输入标签...")
        tags = Prompt.ask(
            "输入以逗号分隔的标签列表",
            default="dev"
        )
        tags = [t.strip() for t in tags.split(",") if t != ""]

        with open_dict(cfg):
            cfg.tags = tags

        log.info(f"标签: {cfg.tags}")

    if save_to_file:
        with open(Path(cfg.paths.output_dir, "tags.log"), "w") as file:
            rich.print(cfg.tags, file=file)
