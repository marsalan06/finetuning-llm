# LoRA Fine-Tuning Pipeline

A modular Python implementation for fine-tuning language models using LoRA (Low-Rank Adaptation) technique. This project provides a complete pipeline for fine-tuning models on custom datasets with efficient parameter updates.

## Features

- **Modular Design**: Well-organized code structure with separate modules for different functionalities
- **LoRA Implementation**: Efficient fine-tuning using Low-Rank Adaptation
- **Configurable**: Centralized configuration for easy hyperparameter tuning
- **Evaluation Tools**: Built-in comparison between base and fine-tuned models
- **CSV Export**: Save model predictions for manual analysis

## Project Structure

```
finetuning-llm/
├── config.py              # Configuration and hyperparameters
├── data_loader.py         # Dataset loading and preprocessing
├── model_setup.py         # Model initialization and LoRA setup
├── trainer.py             # Training orchestration
├── generation.py          # Text generation and evaluation
├── main.py               # Main pipeline script
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── lower_rank_adaption_fine_tuning.ipynb  # Original notebook
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd finetuning-llm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Full Pipeline

Run the complete fine-tuning pipeline:

```bash
python main.py
```

This will:
1. Load and preprocess the dataset
2. Initialize the base model
3. Apply LoRA configuration
4. Train the model
5. Save the fine-tuned model
6. Compare base and fine-tuned models
7. Save results to CSV

### Evaluation Only

If you already have a trained model, run only the evaluation:

```bash
python main.py --evaluate
```

## Configuration

All hyperparameters are centralized in `config.py`. Key configurations include:

### Model Configuration
- `MODEL_CHECKPOINT`: Base model to fine-tune (default: "distilgpt2")
- `MAX_LENGTH`: Maximum sequence length (default: 512)

### LoRA Configuration
- `r`: Rank of LoRA adaptation (default: 8)
- `lora_alpha`: Scaling factor (default: 32)
- `lora_dropout`: Dropout rate (default: 0.1)
- `target_modules`: Layers to apply LoRA to

### Training Configuration
- `learning_rate`: Learning rate (default: 5e-5)
- `per_device_train_batch_size`: Batch size (default: 16)
- `num_train_epochs`: Number of training epochs (default: 3)

## Dataset

The pipeline uses the "iamtarun/code_instructions_120k_alpaca" dataset, which contains:
- Instruction: Programming task description
- Input: Additional context or requirements
- Output: Expected code solution

## Output Files

- `./fine_tuned_model/`: Saved fine-tuned model and tokenizer
- `./results/`: Training logs and checkpoints
- `./logs/`: Training logs
- `generated_outputs.csv`: Model comparison results

## Key Components

### Data Loader (`data_loader.py`)
- Handles dataset loading from Hugging Face
- Preprocesses data by combining instruction and input
- Tokenizes text for model training

### Model Setup (`model_setup.py`)
- Initializes base model and tokenizer
- Applies LoRA configuration
- Provides model information utilities

### Trainer (`trainer.py`)
- Sets up training configuration
- Handles model training and saving
- Loads fine-tuned models for evaluation

### Generation (`generation.py`)
- Generates text from models
- Compares base and fine-tuned model outputs
- Saves results to CSV format

## LoRA Benefits

1. **Parameter Efficiency**: Only updates a small number of parameters
2. **Memory Efficient**: Requires less GPU memory
3. **Fast Training**: Faster convergence compared to full fine-tuning
4. **Modular**: Can be easily applied to different base models

## Customization

### Using Different Models
Change the model checkpoint in `config.py`:
```python
MODEL_CHECKPOINT = "gpt2"  # or any other model
```

### Using Different Datasets
Modify the dataset name in `config.py`:
```python
DATASET_NAME = "your-dataset-name"
```

### Adjusting LoRA Parameters
Modify LoRA configuration in `config.py`:
```python
LORA_CONFIG = {
    "r": 16,  # Higher rank for more capacity
    "lora_alpha": 64,  # Higher alpha for stronger adaptation
    # ... other parameters
}
```

## Troubleshooting

### GPU Memory Issues
- **CUDA Out of Memory**: Reduce batch size in `config.py`
- **MPS Memory Issues** (macOS): Reduce batch size or use CPU fallback
- Use a smaller model checkpoint
- Reduce sequence length

### macOS Specific
- **MPS Not Available**: Ensure you're using PyTorch 2.0+ and macOS 12.3+
- **Slow Training**: MPS may be slower than CUDA for some operations
- **Memory Issues**: Apple Silicon has unified memory - monitor Activity Monitor

### Training Too Slow
- Increase batch size if memory allows
- Reduce number of epochs
- Use a smaller dataset for testing

### Poor Results
- Increase LoRA rank (`r`)
- Adjust learning rate
- Try different target modules

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.53+
- PEFT 0.15+
- GPU acceleration (recommended):
  - **NVIDIA GPU**: CUDA-compatible GPU
  - **Apple Silicon**: MPS (Metal Performance Shaders) - automatically detected
  - **CPU**: Fallback option (slower training)

## License

This project is open source and available under the MIT License. 