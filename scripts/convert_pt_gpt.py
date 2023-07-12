from typing import Tuple, Dict, List
import yaml
import os
from pathlib import Path

import lightning as L
from lightning.pytorch.tuner import Tuner
import torch
import hydra
from omegaconf import DictConfig
import optuna

from dlearn import utils

log = utils.get_pylogger(__name__)




def convert(cfg: DictConfig) -> Tuple[dict, dict]:
    """Demo function.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info(f"Loaded Model!")

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    object_dict = {
        "cfg": cfg,
        "model": model,
    }
    
    out_path = Path(hydra.utils.get_original_cwd())  / 'outputs'

    log.info("Scripting Model...")
    scripted_model = model.to("cpu").to_torchscript(method="script")
    path = f"{out_path}/model_gpt.script.pt"
    torch.jit.save(scripted_model, path)
    log.info("Successfully Scripted Model...")

    log.info("Tracing Model...")
    traced_model = model.to("cpu").to_torchscript(method="trace", example_inputs=(torch.randint(0, 128, (1, model.block_size)), torch.randint(0, 128, (1, model.block_size))))
    path = f"{out_path}/model_gpt.trace.pt"
    torch.jit.save(traced_model, path)
    log.info("Successfully Traced Model...")

    return None, object_dict


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="convert.yaml"
)
def main(cfg: DictConfig) -> None:
    convert(cfg)

if __name__ == "__main__":
    main()