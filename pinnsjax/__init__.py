"""PINNs-JAX 包的主要接口。"""

from .train import train
from .trainer import Trainer

__all__ = ['train', 'Trainer']
