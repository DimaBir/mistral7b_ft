from omegaconf import DictConfig


def generate_code(prompt, model, tokenizer, cfg: DictConfig):
    """
    Generate code based on a given prompt using the fine-tuned model.

    Args:
        prompt (str): The input prompt for code generation.
        model (Model): The fine-tuned model.
        tokenizer (Tokenizer): The tokenizer corresponding to the model.
        cfg (DictConfig): The configuration object from Hydra.

    Returns:
        str: The generated code sequence.
    """
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(inputs['input_ids'], max_length=cfg.model.max_length)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error in generating code: {e}")
        return ""
