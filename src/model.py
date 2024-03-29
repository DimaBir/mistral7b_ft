from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM


def load_model(cfg):
    """
    Load the model and wrap it with PEFTModel for LoRA training.

    Parameters:
    cfg: Configuration object with model settings.

    Returns:
    The wrapped model with LoRA configuration applied.
    """
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(cfg.model.name)

    # Create a LoRAConfig
    lora_config = LoraConfig.from_pretrained(
        cfg.model.name,
        lora_alpha=cfg.model.lora_alpha,
        lora_r=cfg.model.lora_r
    )

    # Wrap the base model as a PEFTModel for LoRA training
    model = get_peft_model(base_model, lora_config)

    return model
