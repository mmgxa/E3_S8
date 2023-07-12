from typing import List, Tuple

import torch
import hydra
import gradio as gr
import tiktoken
from omegaconf import DictConfig

from dlearn import utils

log = utils.get_pylogger(__name__)


# tokenizer
cl100k_base = tiktoken.get_encoding("cl100k_base")

# In production, load the arguments directly instead of accessing private attributes
# See openai_public.py for examples of arguments for specific encodings
enc = tiktoken.Encoding(
    # If you're changing the set of special tokens, make sure to use a different name
    # It should be clear from the name what behaviour to expect.
    name="cl100k_im",
    pat_str=cl100k_base._pat_str,
    mergeable_ranks=cl100k_base._mergeable_ranks,
    special_tokens={
        **cl100k_base._special_tokens,
        "<|im_start|>": 100264,
        "<|im_end|>": 100265,
    }
)

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

    def regen_text(chatbot, temperature, max_tokens) -> str:
        _, chatbot = text_completion(chatbot[-1][0], chatbot, temperature, max_tokens)
        return "", chatbot

    def text_completion(text: str, chatbot, temperature, max_tokens) -> str:
        input_enc = torch.tensor(enc.encode(text))
        with torch.inference_mode():
            out_gen = loaded_model.model.generate(input_enc.unsqueeze(0).long(), 
                                                  max_new_tokens=max_tokens,
                                                  temperature=temperature)
        decoded = enc.decode(out_gen[0].cpu().numpy().tolist())
        chatbot.append((text, decoded))
        return "", chatbot


    with gr.Blocks(title="EMLO-S8", theme=gr.themes.Base()) as grdemo:
        gr.Markdown("A Custom GPT Model trained on Harry Potter's dataset.")

        with gr.Column(scale=6):
            chatbot = gr.Chatbot(elem_id="chatbot", label="EMLO Chatbot", visible=True, height=550)
            with gr.Row(equal_height=True):
                with gr.Column(scale=6):
                    textbox = gr.Textbox(show_label=False,
                        placeholder="Enter text and press ENTER", container=False)
                with gr.Column(scale=1, min_width=60):
                    submit_btn = gr.Button(value="Submit", visible=True)
            with gr.Row(visible=True) as button_row:
                regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=True)
                clear_btn = gr.ClearButton([textbox, chatbot], value="🗑️  Clear chat")

            with gr.Accordion("Parameters", open=False, visible=True) as parameter_row:
                temperature = gr.Slider(minimum=0.0, maximum=1.0, value=1, step=0.1, interactive=True, label="Temperature")
                max_tokens = gr.Slider(minimum=0, maximum=256, value=32, step=4, interactive=True, label="Max output tokens")

            submit_btn.click(text_completion, inputs=[textbox, chatbot, temperature, max_tokens], outputs=[textbox, chatbot])
            regenerate_btn.click(regen_text, inputs=[chatbot, temperature, max_tokens], outputs=[textbox, chatbot])
            
        
    grdemo.launch()

@hydra.main(
    version_base="1.3", config_path="../configs", config_name="demo.yaml"
)
def main(cfg: DictConfig) -> None:
    demo(cfg)

if __name__ == "__main__":
    main()