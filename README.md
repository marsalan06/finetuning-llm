# LoRA Fine-tuning Pipeline with Checkpoint Support

A comprehensive pipeline for fine-tuning language models using LoRA (Low-Rank Adaptation) with advanced checkpoint management and resource optimization for MacBook GPU training.

## Features

- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning
- **Checkpoint Management**: Automatic checkpoint saving and resumption
- **Resource Optimization**: Optimized for MacBook GPU (MPS) training
- **Memory Management**: Gradient checkpointing, mixed precision, and batch size optimization
- **System Monitoring**: Real-time resource usage monitoring
- **Flexible Training**: Resume training from any checkpoint

## Resource Optimizations for MacBook GPU

The pipeline includes several optimizations specifically designed for MacBook GPU training:

### Memory Optimizations
- **Reduced Batch Size**: `per_device_train_batch_size=4` (reduced from 16)
- **Gradient Accumulation**: `gradient_accumulation_steps=8` to simulate larger batches
- **Mixed Precision**: FP16 training enabled for memory efficiency
- **Gradient Checkpointing**: Trades compute for memory
- **Frequent Checkpoints**: Save every 500 steps instead of every epoch

### Training Optimizations
- **Cosine Learning Rate Scheduling**: Smooth learning rate decay
- **Gradient Clipping**: Prevents gradient explosion
- **Warmup Steps**: Gradual learning rate warmup
- **Multi-worker Data Loading**: Parallel data loading

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd finetuning-llm
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

Run the full training pipeline:
```bash
python main.py
```

The script will:
- Check system resources
- Load and preprocess data
- Set up the model with LoRA
- Train with automatic checkpointing
- Evaluate and compare results

### Resume Training

If training was interrupted, resume from the latest checkpoint:
```bash
python main.py --resume
```

### Evaluation Only

Run evaluation on a previously trained model:
```bash
python main.py --evaluate
```

### System Check

Analyze system resources and get recommendations:
```bash
python system_check.py
```

### Checkpoint Management

List all available checkpoints:
```bash
python checkpoint_manager.py list
```

Test latest checkpoint against base model:
```bash
python checkpoint_manager.py test
```

Test specific checkpoint:
```bash
python checkpoint_manager.py test ./results/checkpoint-1000
```

Clean up old checkpoints (keep only the latest 3):
```bash
python checkpoint_manager.py cleanup 3
```

Delete a specific checkpoint:
```bash
python checkpoint_manager.py delete ./results/checkpoint-1000
```

## Configuration

All settings are centralized in `config.py`:

### Training Configuration
```python
TRAINING_ARGS = {
    "per_device_train_batch_size": 4,  # Reduced for memory efficiency
    "gradient_accumulation_steps": 8,  # Simulate larger batch size
    "fp16": True,  # Mixed precision
    "save_steps": 500,  # Save every 500 steps
    "eval_steps": 500,  # Evaluate every 500 steps
    "warmup_steps": 100,  # Learning rate warmup
    "lr_scheduler_type": "cosine",  # Cosine scheduling
    "max_grad_norm": 1.0,  # Gradient clipping
    "resume_from_checkpoint": True,  # Enable checkpoint resumption
}
```

### Memory Optimization
```python
MEMORY_OPTIMIZATION = {
    "use_gradient_checkpointing": True,
    "use_8bit_optimizer": False,  # Enable if you have bitsandbytes
    "use_4bit_quantization": False,  # Enable if you have bitsandbytes
    "max_memory_MB": 8000,  # Adjust based on your MacBook
}
```

## Checkpoint System

### Automatic Checkpointing
- Checkpoints are saved every 500 steps
- Only the latest 3 checkpoints are kept to save disk space
- Training automatically resumes from the latest checkpoint if interrupted

### Checkpoint Information
Each checkpoint contains:
- Model weights
- Optimizer state
- Learning rate scheduler state
- Training history
- Evaluation metrics

### Manual Checkpoint Management
Use the checkpoint manager for advanced operations:
```bash
# List all checkpoints with details
python checkpoint_manager.py list

# Test latest checkpoint against base model
python checkpoint_manager.py test

# Test specific checkpoint
python checkpoint_manager.py test ./results/checkpoint-1000

# Clean up old checkpoints
python checkpoint_manager.py cleanup 3

# Check if resuming is possible
python checkpoint_manager.py resume
```

## System Requirements

### Minimum Requirements
- macOS with Apple Silicon (M1/M2/M3) or Intel Mac
- 8GB RAM (16GB recommended)
- 10GB free disk space

### Recommended Setup
- 16GB+ RAM
- 20GB+ free disk space
- Fast SSD storage

## Troubleshooting

### Memory Issues
If you encounter memory issues:
1. Reduce `per_device_train_batch_size` to 2
2. Increase `gradient_accumulation_steps` to 16
3. Enable `use_8bit_optimizer` if you have bitsandbytes installed
4. Reduce `max_length` in the configuration

### Training Interruption
If training is interrupted:
1. The system will automatically resume from the latest checkpoint
2. Use `python main.py --resume` to manually resume
3. Check available checkpoints with `python checkpoint_manager.py list`
4. Test checkpoint performance with `python checkpoint_manager.py test`

### Performance Optimization
For better performance:
1. Ensure you're using MPS (Apple Silicon GPU)
2. Close other applications to free up memory
3. Use an external SSD for faster I/O
4. Consider using 8-bit optimization if available

### System Analysis
Run system check to get recommendations:
```bash
python system_check.py
```

## File Structure

```
finetuning-llm/
├── config.py              # Configuration settings
├── data_loader.py         # Data loading and preprocessing
├── model_setup.py         # Model and LoRA setup
├── trainer.py             # Training with checkpoint support
├── generation.py          # Text generation and evaluation
├── checkpoint_manager.py  # Checkpoint management utilities
├── system_check.py        # System resource analysis
├── main.py               # Main training pipeline
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Output Files

After training, you'll find:
- `./results/` - Training checkpoints and logs
- `./fine_tuned_model/` - Final trained model
- `./logs/` - Training logs
- `generated_outputs.csv` - Comparison results
- `checkpoint_XXXX_comparison.csv` - Checkpoint test results

## Advanced Usage

### Custom Configuration
Modify `config.py` to adjust:
- Model checkpoint
- Dataset
- LoRA parameters
- Training hyperparameters
- Memory optimization settings

### Custom Datasets
Replace the dataset in `config.py`:
```python
DATASET_NAME = "your-dataset-name"
```

### Different Models
Change the base model:
```python
MODEL_CHECKPOINT = "gpt2"  # or any other model
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License. 