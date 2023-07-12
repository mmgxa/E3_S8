from typing import List, Tuple

import torch
import hydra
import gradio as gr
from omegaconf import DictConfig
from torchvision import transforms

from dlearn import utils

log = utils.get_pylogger(__name__)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def demo(cfg: DictConfig) -> Tuple[dict, dict]:
    """Demo function.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info("Running Demo")

    log.info(f"Instantiating traced model <{cfg.ckpt_path}>")
    loaded_model = torch.jit.load(cfg.ckpt_path)

    log.info(f"Loaded Model!")

    def predict(image):
        image = transforms.ToTensor()(image.convert('RGB')).unsqueeze(0)
        if image is None:
            return None
        # image = torch.tensor(image[None, None, ...], dtype=torch.float32)
        preds = loaded_model.forward_jit(image)
        preds = preds[0].tolist()
        return {classes[i]: preds[i] for i in range(10)}

    demo = gr.Interface(fn=predict, # mapping function from input to output
                    inputs=gr.Image(type="pil"), # what are the inputs?
                    outputs=[gr.Label(num_top_classes=10, label="Predictions")],
                    title='EMLO-ViT - 🐱 🐶',
                    description='ViT trained on CIFAR10',
                    article='For EMLO-S8',
                    )
    
    demo.launch(server_port=8080)
    


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="demo.yaml"
)
def main(cfg: DictConfig) -> None:
    demo(cfg)

if __name__ == "__main__":
    main()