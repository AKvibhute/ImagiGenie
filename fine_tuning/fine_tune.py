
import os
import math
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

# ✅ Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n✅ Using device: {device}")

# ✅ Paths
model_path = "./ai_engine/fine_tuned_model"
data_path = "data/processed/stories.txt"
os.makedirs(model_path, exist_ok=True)

# ✅ Load tokenizer and add <|endofstory|>
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
tokenizer.add_special_tokens({"eos_token": "<|endofstory|>"})
tokenizer.pad_token = tokenizer.eos_token

# ✅ Load and resize model
model = GPT2LMHeadModel.from_pretrained("distilgpt2")
model.resize_token_embeddings(len(tokenizer))
model.to(device)

# ✅ Load dataset
dataset = load_dataset("text", data_files={"train": data_path})

# ✅ Add <|endofstory|> to each sample (if not already present)
def add_end_token(example):
    text = example["text"].strip()
    if not text.endswith("<|endofstory|>"):
        text += " <|endofstory|>"
    return {"text": text}

dataset = dataset.map(add_end_token)

# ✅ Tokenize dataset
def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512  # A good fit for ~300 token outputs
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# ✅ Data collator for causal LM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ✅ Training arguments
training_args = TrainingArguments(
    num_train_epochs=5,
    per_device_train_batch_size=1,  # Good for low VRAM
    gradient_accumulation_steps=4,
    logging_strategy="epoch",
    save_strategy="no",
    evaluation_strategy="no",
    report_to="none",
    fp16=False  # Disable mixed precision (e.g., for MX330 or older GPUs)
)

# ✅ Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

# ✅ Train
trainer.train()

# ✅ Save model and tokenizer
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)

# ✅ Evaluate with Perplexity
def compute_perplexity(model, tokenized_subset, tokenizer, batch_size=2):
    model.eval()
    dataloader = DataLoader(tokenized_subset, batch_size=batch_size, collate_fn=data_collator)
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(0)
            total_tokens += input_ids.size(0)

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

# ✅ Perplexity evaluation
print("\n📊 Evaluating model on sample subset...")
eval_subset = tokenized_dataset["train"].select(range(100))
eval_subset.set_format(type="torch", columns=["input_ids", "attention_mask"])
perplexity = compute_perplexity(model, eval_subset, tokenizer)
print(f"✅ Perplexity: {perplexity:.2f}")