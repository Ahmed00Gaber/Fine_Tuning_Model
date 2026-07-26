# Fine_Tuning_Model - Gemma 4B Fine-Tuning

This repository contains code and data for fine-tuning the **Google Gemma 1.1 4B** model on Arabic legal/regulatory question-answering data.

## 📁 Repository Structure

```
Fine_Tuning_Model/
├── fine_tune_gemma_4b.py      # Main fine-tuning script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── data/
    ├── welcoming_gemma_clean.jsonl     # Welcome/greeting conversations
    ├── query_question_gemma_clean.jsonl # Query-based questions
    ├── rule_lookup_gemma_clean.jsonl    # Legal rule lookup data
    ├── out_of_scope_gemma_clean.jsonl   # Out-of-scope questions
    └── validation_report_gemma.json      # Data validation report
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Install PyTorch with CUDA support (choose appropriate version)
# For CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 2. Run Fine-Tuning

```bash
python fine_tune_gemma_4b.py
```

### 3. Output

The fine-tuned model will be saved in the `outputs/` directory, including:
- Model weights (LoRA adapters)
- Tokenizer configuration
- Training configuration

## 📊 Data Overview

The dataset contains Arabic legal/regulatory conversations categorized into:

| Type | Count | Description |
|------|-------|-------------|
| rule_lookup | ~6,718 | Legal rule/article lookup requests |
| query_question | ~7,294 | General legal queries and questions |
| metadata_search | ~2,067 | Metadata search requests |
| welcoming | ~328 | Welcome and greeting conversations |
| out_of_scope | ~75 | Out-of-scope questions |

**Total: ~16,482 examples**

## ⚙️ Configuration

### Model Settings
- **Base Model**: `google/gemma-1.1-4b-it`
- **Quantization**: 4-bit (NF4) with QLoRA
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)

### LoRA Configuration
- Rank (r): 16
- Alpha: 32
- Dropout: 0.05
- Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### Training Configuration
- Epochs: 3
- Batch Size: 1 (with gradient accumulation of 4)
- Learning Rate: 2e-4
- Max Sequence Length: 512
- Optimizer: paged_adamw_8bit
- Mixed Precision: FP16

## 🎯 Features

- **QLoRA**: 4-bit quantization for memory-efficient training
- **LoRA**: Parameter-efficient fine-tuning
- **Multi-file Support**: Loads all JSONL files from the data directory
- **Automatic Formatting**: Converts data to Gemma's chat format
- **Checkpointing**: Saves model after each epoch

## 🔧 Customization

### Change Model
Edit `MODEL_NAME` in `fine_tune_gemma_4b.py`:
```python
MODEL_NAME = "google/gemma-1.1-4b-it"  # Change to any Gemma variant
# Options: google/gemma-2b-it, google/gemma-7b-it, google/gemma-1.1-2b-it, etc.
```

### Adjust Training Parameters
Modify the `training_args` in the script:
```python
training_args = SFTConfig(
    num_train_epochs=3,
    per_device_train_batch_size=1,
    learning_rate=2e-4,
    # ... other parameters
)
```

### Use Different Data
Edit `DATA_FILES` list to include/exclude specific files:
```python
DATA_FILES = [
    "welcoming_gemma_clean.jsonl",
    "query_question_gemma_clean.jsonl",
    # Add or remove files as needed
]
```

## 💡 Tips

1. **GPU Requirements**: For 4-bit QLoRA, you need at least **12-16GB VRAM** for Gemma 4B
2. **Memory Issues**: Reduce batch size or use gradient accumulation if you encounter OOM errors
3. **Training Time**: Expect several hours for 3 epochs depending on your GPU
4. **Monitoring**: Use `watch -n 10 nvidia-smi` to monitor GPU usage

## 📚 References

- [Gemma Models](https://huggingface.co/collections/google/gemma-release-6552432667522261079)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [PEFT Library](https://github.com/huggingface/peft)
- [TRL Library](https://github.com/huggingface/trl)

## 🤝 Contributing

Feel free to open issues or pull requests for improvements!

---

**Last Updated**: 2026-07-26
