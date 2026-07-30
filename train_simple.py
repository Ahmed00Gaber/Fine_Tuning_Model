#!/usr/bin/env python3
"""
Simplified Fine-Tuning Script for Gemma 4B

This is a streamlined version of the fine-tuning script with minimal configuration.
Perfect for quick testing and experimentation.
"""

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import torch
import json
import os

# ============================================================
# CONFIGURATION - Edit these values
# ============================================================

MODEL_NAME = "google/gemma-1.1-4b-it"  # Base model
DATA_DIR = "data"                     # Data directory
OUTPUT_DIR = "outputs"                # Output directory
EPOCHS = 3                            # Training epochs
BATCH_SIZE = 1                        # Batch size per GPU
LEARNING_RATE = 2e-4                 # Learning rate

# ============================================================
# Load and Prepare Data
# ============================================================

print("Loading data...")

# Collect all JSONL files
data_files = [
    "welcoming_gemma_clean.jsonl",
    "query_question_gemma_clean.jsonl",
    "rule_lookup_gemma_clean.jsonl",
    "out_of_scope_gemma_clean.jsonl",
]

questions = []
answers = []

for filename in data_files:
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            questions.append(item.get('question', ''))
            output = item.get('output', {})
            if isinstance(output, dict):
                answers.append(output.get('answer', str(output)))
            else:
                answers.append(str(output))

# Create dataset
dataset = Dataset.from_dict({'question': questions, 'answer': answers})
print(f"Loaded {len(dataset)} examples")

# ============================================================
# Format Data for Gemma
# ============================================================

def format_for_gemma(example):
    """Format examples in Gemma's chat format"""
    text = f"<start_of_turn>user\n{example['question']}<end_of_turn>\n<start_of_turn>model\n{example['answer']}<end_of_turn>"
    return {'text': text}

dataset = dataset.map(format_for_gemma)
print("Data formatted for Gemma")

# ============================================================
# Load Tokenizer and Model
# ============================================================

print("Loading tokenizer and model...")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Quantization config for QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
model.config.use_cache = False

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================================================
# Training Configuration
# ============================================================

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    learning_rate=LEARNING_RATE,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    optim="paged_adamw_8bit",
    max_seq_length=512,
    report_to="none",
    packing=True,
)

# ============================================================
# Train!
# ============================================================

print("Starting training...")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text",
    tokenizer=tokenizer,
)

trainer.train()

# ============================================================
# Save Results
# ============================================================

print("Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n✅ Training complete! Model saved to {OUTPUT_DIR}")
