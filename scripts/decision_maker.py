import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def make_decision(input_text: str, model_path: str):
    """
    Generates a decision based on input text using a pre-trained causal language model.

    Args:
        input_text (str): Input string containing the relevant trading data.
        model_path (str): Path to the pre-trained model on Hugging Face repository.

    Returns:
        str: The decision text generated by the model.
    """
    try:
        print(f"Loading model from Hugging Face Hub: {model_path}")
        # Load model and tokenizer with Hugging Face Hub
        model = AutoModelForCausalLM.from_pretrained(model_path, use_auth_token=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=True)
    except Exception as e:
        raise ValueError(f"Error loading model from '{model_path}': {e}")

    # Tokenize the input text
    try:
        inputs = tokenizer(input_text, return_tensors="pt")
    except Exception as e:
        raise RuntimeError(f"Error during tokenization: {e}")

    # Generate output using the model
    try:
        outputs = model.generate(**inputs, max_length=50)
        decision = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        raise RuntimeError(f"Error during model inference: {e}")

    return decision

if __name__ == "__main__":
    # Example input data
    input_text = "Open: 0.6200, Close: 0.6250"

    # Using the specific Hugging Face model repository
    model_path = "mrzlab630/lora-alpaca-trading-candles"

    try:
        decision = make_decision(input_text, model_path)
        print("Decision:", decision)
    except ValueError as e:
        print(f"Model Error: {e}")
    except RuntimeError as e:
        print(f"Inference Error: {e}")
