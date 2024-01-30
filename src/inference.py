def generate_code(prompt, model, tokenizer, max_length=512):
    """
    Generate code based on a given prompt using the fine-tuned model.

    Parameters:
    prompt (str): The input prompt for code generation.
    model (Model): The fine-tuned Mistral 7B model.
    tokenizer (Tokenizer): The tokenizer corresponding to the Mistral 7B model.
    max_length (int): Maximum length for the generated sequence.

    Returns:
    str: The generated code sequence.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
