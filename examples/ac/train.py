from typing import Any, Dict, List, Optional, Tuple

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import rootutils
from omegaconf import DictConfig

import pinnsjax


def read_data_fn(root_path):
    """Read and preprocess data from the specified root path.

    :param root_path: The root directory containing the data.
    :return: Processed data will be used in Mesh class.
    """

    data = pinnsjax.utils.load_data(root_path, "AC.mat")
    exact_u = np.real(data["uu"])
    return {"u": exact_u}


def pde_fn(
    functional_model, params, outputs: Dict[str, jax.Array], x: jax.Array
):
    """Define the partial differential equations (PDEs)."""

    u = outputs["u"]
    u_xx = pinnsjax.utils.fwd_gradient(functional_model, argnums=1, order=2)(
        params, x, None, "u"
    )[0]

    outputs["f"] = 5.0 * u[:-1] - 5.0 * (u[:-1] ** 3) + 0.0001 * u_xx[:-1]

    return outputs


@hydra.main(
    version_base="1.3", config_path="configs", config_name="config.yaml"
)
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    pinnsjax.utils.extras(cfg)

    # train the model
    metric_dict, _ = pinnsjax.train(
        cfg, read_data_fn=read_data_fn, pde_fn=pde_fn, output_fn=None
    )

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = pinnsjax.utils.get_metric_value(
        metric_dict=metric_dict, metric_names=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
