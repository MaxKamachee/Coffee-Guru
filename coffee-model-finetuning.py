import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

def prepare_dataset(file_path, tokenizer):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=256)
    return dataset

def fine_tune_model(model, tokenizer, train_dataset):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    training_args = TrainingArguments(
        output_dir="./coffee_model",
        overwrite_output_dir=True,
        num_train_epochs=30,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        learning_rate=5e-5,
        warmup_steps=200,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()
    
    model.save_pretrained("./coffee_model")
    tokenizer.save_pretrained("./coffee_model")

def main():
    model_name = "gpt2"  # Using the smallest GPT-2 model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = prepare_dataset("coffee_recipes.txt", tokenizer)
    fine_tune_model(model, tokenizer, train_dataset)

    print("Fine-tuning complete. Model and tokenizer saved in ./coffee_model")

if __name__ == "__main__":
    main()