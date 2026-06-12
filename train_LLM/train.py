from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments, Trainer

# Load dataset
dataset = load_dataset("json", data_files="dataset.json")

# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",  # or llava-v1.5-7b if multimodal
    max_seq_length=1024,
    load_in_4bit=True,
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj","v_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    use_gradient_checkpointing=True,
    use_rslora=True,
)

# Training config
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    save_strategy="epoch",
    fp16=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)

trainer.train()

# Save model
model.save_pretrained("my_llm")
tokenizer.save_pretrained("my_llm")
