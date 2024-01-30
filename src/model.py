from transformers import AutoModelForCausalLM
from lora import LoRAConfig, apply_lora
from omegaconf import DictConfig

def load_model(cfg: DictConfig) -> AutoModelForCausalLM:
    """
    Load the Mistral 7B model and apply LoRA configuration.

    Parameters:
    cfg (DictConfig): The configuration object from Hydra.

    Returns:
    AutoModelForCausalLM: The loaded and configured Mistral 7B model.
    """
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name)
    lora_config = LoRAConfig(lora_alpha=cfg.model.lora_alpha, lora_r=cfg.model.lora_r)
    model = apply_lora(model, lora_config)
    return model