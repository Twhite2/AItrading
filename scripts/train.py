from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import pandas as pd
import torch

def fine_tune_model(data_path: str, output_dir: str):
    # Load data
    df = pd.read_csv(data_path)
    training_data = [{"text": f"Open: {row['open']}, Close: {row['close']}"} for _, row in df.iterrows()]
    
    # Load pretrained model and tokenizer
    model_name = "mrzlab630/lora-alpaca-trading-candles"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize data
    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)

    tokenized_data = tokenize(training_data)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_steps=1000,
        save_total_limit=2,
    )

    # Fine-tune the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
    )
    trainer.train()

    # Save fine-tuned model
    model.save_pretrained(output_dir)
    print(f"Fine-tuned model saved to {output_dir}")

if __name__ == "__main__":
    fine_tune_model("data/processed/nzdusd_processed.csv", "mrzlab630/lora-alpaca-trading-candles")
