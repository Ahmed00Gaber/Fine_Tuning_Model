# ============================================================
# Fine-Tuning Script for Gemma 4B Model
# Adjusted from Llama-3.2-1B-Instruct to Google Gemma 4B
# ============================================================

# Import necessary libraries
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import torch
import os
import json

# ============================================================
# Configuration
# ============================================================

# Model Configuration
MODEL_NAME = "google/gemma-1.1-4b-it"  # Changed from Llama to Gemma 4B

# Data Configuration - Load all JSONL files from the data directory
DATA_DIR = "data"
DATA_FILES = [
    "welcoming_gemma_clean.jsonl",
    "query_question_gemma_clean.jsonl", 
    "rule_lookup_gemma_clean.jsonl",
    "out_of_scope_gemma_clean.jsonl",
]

# Output Configuration
OUTPUT_DIR = "outputs"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Quantization Configuration (QLoRA)
# ============================================================

# QLoRA loads the pretrained model in 4-bit precision
# This reduces GPU memory usage while maintaining good performance

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                  # Load model weights in 4-bit precision
    bnb_4bit_quant_type="nf4",         # Use NF4 (NormalFloat4)
    bnb_4bit_compute_dtype=torch.float16, # Perform computations in FP16
    bnb_4bit_use_double_quant=True,     # Apply second quantization step
)

# ============================================================
# Load Tokenizer
# ============================================================

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

# Gemma uses <bos> and <eos> tokens
# For chat models, we need to handle padding properly
tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# Load Base Model
# ============================================================

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,     # Load in 4-bit using QLoRA
    device_map="auto",                 # Auto device placement
    torch_dtype=torch.float16,         # FP16 computations
    trust_remote_code=True,
)

# Disable cache during training
model.config.use_cache = False

# ============================================================
# Print Model Information
# ============================================================

print("=" * 60)
print("Model Loaded Successfully")
print(f"Model: {MODEL_NAME}")
print(f"Device: {model.device}")
print("=" * 60)

# ============================================================
# Custom Dataset Loading Function
# ============================================================

def load_jsonl_files(directory, files):
    """Load multiple JSONL files and combine them into a single dataset"""
    all_data = []
    
    for file in files:
        filepath = os.path.join(directory, file)
        print(f"Loading {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    all_data.append(data)
                except json.JSONDecodeError:
                    continue
    
    return all_data

# Load all data
data_list = load_jsonl_files(DATA_DIR, DATA_FILES)
print(f"Total examples loaded: {len(data_list)}")

# ============================================================
# Convert Dataset to Hugging Face Dataset
# ============================================================

dataset_dict = {
    'question': [],
    'answer': [],
    'type': []
}

for item in data_list:
    question = item.get('question', '')
    output = item.get('output', {})
    
    # Extract answer based on output type
    if isinstance(output, dict):
        answer = output.get('answer', '')
        if not answer:
            # For rule_lookup and query_question, use the whole output or specific fields
            answer = str(output)
    else:
        answer = str(output)
    
    data_type = output.get('type', 'unknown') if isinstance(output, dict) else 'unknown'
    
    dataset_dict['question'].append(question)
    dataset_dict['answer'].append(answer)
    dataset_dict['type'].append(data_type)

# Create Hugging Face Dataset
dataset = Dataset.from_dict(dataset_dict)

print(f"Dataset created with {len(dataset)} examples")
print(f"Types distribution: {set(dataset['type'])}")

# ============================================================
# Convert Dataset into Gemma Chat Format
# ============================================================

def format_gemma_example(example):
    """
    Format examples for Gemma fine-tuning.
    Gemma uses a specific chat format with <start_of_turn> and <end_of_turn> tokens.
    """
    # For Gemma, we use the chat template approach
    # Format: <start_of_turn>user
    # {question}<end_of_turn>
    # <start_of_turn>model
    # {answer}<end_of_turn>
    
    # Some Gemma versions use different tokens, so we check what's available
    # Common approach: use the tokenizer's chat template
    
    # Alternative format that works with most Gemma versions:
    text = f"<start_of_turn>user\n{example['question']}<end_of_turn>\n<start_of_turn>model\n{example['answer']}<end_of_turn>"
    
    return {"text": text}

# Apply formatting
dataset = dataset.map(format_gemma_example)

# Verify results
print("\nSample formatted example:")
print(dataset[0]["text"][:500])

# ============================================================
# LoRA Configuration
# ============================================================

# LoRA (Low-Rank Adaptation) configuration for efficient fine-tuning
lora_config = LoraConfig(
    r=16,                                    # Rank of LoRA matrices
    lora_alpha=32,                           # Scaling factor
    target_modules=[                        # Target modules for LoRA
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,                       # Dropout rate
    bias="none",                            # No bias training
    task_type="CAUSAL_LM",                  # Causal language modeling
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()

# ============================================================
# Training Configuration
# ============================================================

# Training arguments for SFT (Supervised Fine-Tuning)
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,                  # Output directory
    num_train_epochs=3,                    # Number of training epochs
    per_device_train_batch_size=1,        # Batch size per device
    gradient_accumulation_steps=4,         # Gradient accumulation steps
    learning_rate=2e-4,                   # Learning rate
    logging_steps=10,                      # Log every 10 steps
    save_strategy="epoch",                # Save after each epoch
    save_total_limit=2,                   # Keep only 2 checkpoints
    fp16=True,                             # Use FP16 mixed precision
    optim="paged_adamw_8bit",             # Memory-efficient optimizer
    max_seq_length=512,                   # Maximum sequence length
    report_to="none",                     # Disable external logging
    packing=True,                          # Enable sequence packing
    # Additional settings for better training
    warmup_ratio=0.1,                     # Learning rate warmup
    weight_decay=0.01,                    # Weight decay
    lr_scheduler_type="cosine",           # Learning rate scheduler
)

# ============================================================
# Create the Trainer and Start Fine-Tuning
# ============================================================

print("\nCreating SFTTrainer...")

# Create the SFT trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text",            # Use 'text' field from dataset
    tokenizer=tokenizer,
)

print("Starting training...")

# Start the fine-tuning process
trainer.train()

# ============================================================
# Save the Fine-Tuned Model
# ============================================================

print("\nSaving model...")
trainer.save_model(OUTPUT_DIR)

# Also save the tokenizer
tokenizer.save_pretrained(OUTPUT_DIR)

# Save training configuration
with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w") as f:
    json.dump({
        "model_name": MODEL_NAME,
        "training_epochs": training_args.num_train_epochs,
        "batch_size": training_args.per_device_train_batch_size,
        "learning_rate": training_args.learning_rate,
        "max_seq_length": training_args.max_seq_length,
        "lora_config": {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "target_modules": lora_config.target_modules,
            "lora_dropout": lora_config.lora_dropout,
        },
        "dataset_files": DATA_FILES,
        "total_examples": len(dataset),
    }, f, indent=2)

print("=" * 60)
print("Fine-tuning completed successfully!")
print(f"Model saved to: {OUTPUT_DIR}")
print("=" * 60)
