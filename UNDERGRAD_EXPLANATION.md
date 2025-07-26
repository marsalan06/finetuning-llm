# LoRA Fine-Tuning Pipeline: Comprehensive Undergraduate Guide

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Core Concepts](#core-concepts)
4. [Data Flow](#data-flow)
5. [Important Checks & Validations](#important-checks--validations)
6. [Optimization Techniques](#optimization-techniques)
7. [Key Parameters](#key-parameters)
8. [Edge Cases & Error Handling](#edge-cases--error-handling)
9. [Training Process](#training-process)
10. [Evaluation & Comparison](#evaluation--comparison)

## Project Overview

This project implements a **LoRA (Low-Rank Adaptation)** fine-tuning pipeline for language models, specifically optimized for MacBook GPU training. The system uses **QLoRA (Quantized LoRA)** for memory efficiency and includes comprehensive checkpoint management.

### Key Features:
- **Parameter-Efficient Fine-tuning**: LoRA reduces trainable parameters by 99%
- **Memory Optimization**: 4-bit quantization for MacBook GPU compatibility
- **Checkpoint Management**: Automatic saving and resumption capabilities
- **Resource Monitoring**: Real-time system resource analysis
- **Comparative Evaluation**: Base vs fine-tuned model comparison

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Loading  │───▶│  Model Setup    │───▶│   Training      │
│   & Preprocessing│    │   & LoRA Config │    │   & Checkpoints │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Tokenization   │    │  QLoRA Setup    │    │  Model Saving   │
│  & Formatting   │    │  & Quantization │    │  & Evaluation   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Code Files & Functions Overview

### 📁 Core Configuration Files

#### `config.py` - Central Configuration Management
**Purpose**: Centralizes all hyperparameters and settings for easy modification.

**Key Functions**:
- `get_device()`: Automatically detects best available device (CUDA > MPS > CPU)
- `create_directories()`: Creates necessary directories for training outputs

**Configuration Classes**:
- `LORA_CONFIG`: LoRA/QLoRA parameters (rank, alpha, dropout, target modules)
- `TRAINING_ARGS`: Training hyperparameters (batch size, learning rate, etc.)
- `MEMORY_OPTIMIZATION`: Memory management settings
- `GENERATION_CONFIG`: Text generation parameters

**Device-Specific Logic**:
```python
@staticmethod
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
```

### 📁 Data Processing Files

#### `data_loader.py` - Dataset Management
**Purpose**: Handles dataset loading, preprocessing, and tokenization.

**Key Functions**:

**`load_data()`**
- Loads dataset from Hugging Face Hub
- Splits into train/validation sets (80/20)
- Returns DatasetDict with train/test splits

**`preprocess_data(dataset, tokenizer)`**
- Combines instruction and input fields
- Tokenizes inputs and outputs
- Applies padding and truncation
- Creates labels for training

**`setup_tokenizer(model_checkpoint=None)`**
- Initializes tokenizer with proper padding token
- Handles missing padding token by using EOS token

**Data Flow**:
```python
# Example preprocessing
combined_input = f"Instruction: {instruction}\nInput: {input_data}"
tokenized_inputs = tokenizer(combined_input, max_length=512)
tokenized_labels = tokenizer(examples['output'], max_length=512)
```

### 📁 Model Setup Files

#### `model_setup.py` - Model Initialization & LoRA Configuration
**Purpose**: Sets up base model, applies LoRA, and handles device-specific optimizations.

**Key Functions**:

**`setup_model(model_checkpoint=None)`**
- Loads base model with device-specific optimizations
- Handles QLoRA quantization for CUDA
- Falls back to regular model for MPS
- Sets up tokenizer with padding

**`setup_lora(model, tokenizer)`**
- Applies LoRA configuration to base model
- Creates PEFT model with trainable adapters
- Prints model information and parameter counts

**`print_model_info(model, stage="Base Model")`**
- Displays model architecture information
- Shows total vs trainable parameters
- Prints LoRA-specific information if applicable

**Device-Specific Handling**:
```python
if Config.DEVICE.type == "mps":
    print("⚠ MPS detected, using regular LoRA")
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
else:
    # Configure 4-bit quantization for CUDA
    quantization_config = BitsAndBytesConfig(...)
```

### 📁 Training Files

#### `trainer.py` - Training Orchestration
**Purpose**: Manages the training process, checkpoint handling, and model saving.

**Key Functions**:

**`setup_trainer(model, tokenizer, train_dataset, eval_dataset)`**
- Creates TrainingArguments with MacBook GPU optimizations
- Applies memory optimizations (gradient checkpointing)
- Sets up Trainer with datasets and configuration

**`train_model(trainer, resume_from_checkpoint=None)`**
- Executes training with checkpoint support
- Handles automatic resumption from latest checkpoint
- Returns training results and metrics

**`save_model(model, tokenizer, save_path=None)`**
- Saves fine-tuned model and tokenizer
- Creates model directory structure
- Handles model serialization

**`load_fine_tuned_model(model_path=None)`**
- Loads saved fine-tuned model
- Moves model to appropriate device
- Returns model and tokenizer pair

**Checkpoint Management Functions**:
- `find_latest_checkpoint()`: Locates most recent checkpoint
- `get_checkpoint_info()`: Returns checkpoint metadata
- `cleanup_old_checkpoints(keep_latest=3)`: Manages disk space
- `resume_training_from_checkpoint()`: Handles training resumption

**Training Configuration**:
```python
training_args = TrainingArguments(
    per_device_train_batch_size=4,      # Memory-optimized
    gradient_accumulation_steps=8,      # Simulate larger batches
    learning_rate=5e-5,                 # Conservative learning rate
    fp16=False,                         # MPS compatibility
    save_steps=500,                     # Frequent checkpoints
    eval_steps=500                      # Regular evaluation
)
```

### 📁 Generation & Evaluation Files

#### `generation.py` - Text Generation & Model Comparison
**Purpose**: Handles text generation and comparative evaluation between models.

**Key Functions**:

**`generate_text(model, tokenizer, input_text, max_length=None)`**
- Tokenizes input text
- Generates text using model
- Handles device placement
- Returns decoded generated text

**`compare_models(base_model, base_tokenizer, fine_tuned_model, fine_tuned_tokenizer, inputs)`**
- Generates outputs from both base and fine-tuned models
- Handles multiple input samples
- Returns parallel outputs for comparison

**`print_comparison(inputs, base_outputs, fine_tuned_outputs)`**
- Displays side-by-side comparison
- Shows input and both model outputs
- Formats output for readability

**`save_predictions_to_csv(inputs, base_outputs, fine_tuned_outputs, file_path=None)`**
- Saves comparison results to CSV
- Creates structured output file
- Handles file encoding and formatting

**`get_sample_inputs()`**
- Returns predefined test cases
- Includes programming-related prompts
- Used for model evaluation

**Generation Process**:
```python
# Tokenize input
inputs = tokenizer(input_text, return_tensors="pt", max_length=512)
inputs = {key: value.to(Config.DEVICE) for key, value in inputs.items()}

# Generate text
with torch.no_grad():
    outputs = model.generate(inputs['input_ids'], max_length=100)
    
# Decode output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### 📁 Main Orchestration Files

#### `main.py` - Pipeline Orchestration
**Purpose**: Main entry point that orchestrates the entire fine-tuning pipeline.

**Key Functions**:

**`main()`** - Primary training pipeline
1. **System Check**: Validates resources and device
2. **Data Loading**: Loads and preprocesses dataset
3. **Model Setup**: Initializes base model and LoRA
4. **Trainer Setup**: Configures training environment
5. **Checkpoint Check**: Looks for existing checkpoints
6. **Training**: Executes training with resumption support
7. **Cleanup**: Removes old checkpoints
8. **Model Saving**: Saves fine-tuned model
9. **Evaluation**: Compares base vs fine-tuned models
10. **Results**: Saves comparison to CSV

**`check_system_resources()`**
- Checks available RAM and memory usage
- Validates device compatibility (MPS/CUDA/CPU)
- Provides recommendations for optimization
- Warns about low memory situations

**`run_evaluation_only()`**
- Loads pre-trained models
- Runs comparative evaluation
- Saves results without training

**`resume_training()`**
- Checks for existing checkpoints
- Resumes training from latest checkpoint
- Handles training interruption scenarios

**Command Line Interface**:
```bash
python main.py              # Run full training pipeline
python main.py --evaluate   # Run evaluation only
python main.py --resume     # Resume training from checkpoint
```

### 📁 Utility Files

#### `system_check.py` - System Analysis
**Purpose**: Analyzes system resources and provides optimization recommendations.

**Key Functions**:
- **Memory Analysis**: Checks RAM availability and usage
- **GPU Detection**: Identifies available GPU (MPS/CUDA)
- **Performance Testing**: Benchmarks system capabilities
- **Optimization Recommendations**: Suggests parameter adjustments

#### `checkpoint_manager.py` - Advanced Checkpoint Management
**Purpose**: Provides advanced checkpoint management utilities.

**Key Functions**:
- **`list_checkpoints()`**: Lists all available checkpoints with details
- **`test_checkpoint()`**: Tests checkpoint performance against base model
- **`cleanup_checkpoints()`**: Removes old checkpoints to save space
- **`delete_checkpoint()`**: Removes specific checkpoint
- **`resume_checkpoint()`**: Validates checkpoint for resumption

**Usage Examples**:
```bash
python checkpoint_manager.py list          # List all checkpoints
python checkpoint_manager.py test          # Test latest checkpoint
python checkpoint_manager.py cleanup 3     # Keep only latest 3
python checkpoint_manager.py delete ./results/checkpoint-1000
```

### 📁 Jupyter Notebook

#### `lower_rank_adaption_fine_tuning.ipynb` - Interactive Tutorial
**Purpose**: Interactive notebook for learning and experimentation.

**Contents**:
- **Interactive Code Cells**: Step-by-step execution
- **Visualizations**: Training curves and model comparisons
- **Explanations**: Detailed comments and markdown
- **Experimentation**: Parameter tuning examples

### 📁 Configuration & Dependencies

#### `requirements.txt` - Dependencies
**Purpose**: Lists all required Python packages.

**Key Dependencies**:
- `torch`: PyTorch for deep learning
- `transformers`: Hugging Face transformers library
- `peft`: Parameter-Efficient Fine-Tuning
- `datasets`: Hugging Face datasets
- `bitsandbytes`: 4-bit quantization (CUDA only)
- `accelerate`: Training acceleration
- `psutil`: System resource monitoring

#### `server_info` - System Information
**Purpose**: Stores system-specific information and configurations.

### 📁 Output Directories

#### `./results/` - Training Outputs
- **Checkpoints**: Model snapshots during training
- **Logs**: Training logs and metrics
- **Configurations**: Training configuration files

#### `./logs/` - Training Logs
- **TensorBoard Logs**: Training visualization data
- **Text Logs**: Detailed training progress

#### `./checkpoints/` - Checkpoint Storage
- **Model Snapshots**: Complete model states
- **Optimizer States**: Training optimizer states
- **Scheduler States**: Learning rate scheduler states

#### `./fine_tuned_model/` - Final Model
- **Model Weights**: Fine-tuned model parameters
- **Tokenizer**: Model tokenizer
- **Configuration**: Model configuration files

## File Interaction Flow

```
main.py
├── config.py (loads configuration)
├── data_loader.py (loads & preprocesses data)
├── model_setup.py (sets up model & LoRA)
├── trainer.py (handles training)
├── generation.py (evaluates models)
└── system_check.py (validates system)
```

## Function Call Hierarchy

```
main()
├── check_system_resources()
├── load_data() → data_loader.py
├── setup_tokenizer() → data_loader.py
├── preprocess_data() → data_loader.py
├── setup_model() → model_setup.py
├── setup_lora() → model_setup.py
├── setup_trainer() → trainer.py
├── train_model() → trainer.py
├── save_model() → trainer.py
├── load_fine_tuned_model() → trainer.py
├── compare_models() → generation.py
└── save_predictions_to_csv() → generation.py
```

This modular architecture ensures:
- **Separation of Concerns**: Each file has a specific responsibility
- **Reusability**: Functions can be used independently
- **Maintainability**: Easy to modify individual components
- **Testability**: Each module can be tested separately
- **Scalability**: Easy to extend with new features

## Core Concepts

### 1. LoRA (Low-Rank Adaptation)
**What it is**: A parameter-efficient fine-tuning technique that adds small trainable matrices to existing layers.

**Why it's important**: 
- Reduces memory usage by 99%
- Maintains model performance
- Enables fine-tuning on consumer hardware

**How it works**:
```python
# Original layer: W = W₀
# LoRA modification: W = W₀ + ΔW
# Where ΔW = BA (low-rank decomposition)
# B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), where r << min(d,k)
```

### 2. QLoRA (Quantized LoRA)
**What it is**: LoRA combined with 4-bit quantization for extreme memory efficiency.

**Key benefits**:
- 4-bit quantization reduces memory by ~75%
- Maintains model performance
- Enables training on 8GB RAM systems

**⚠️ MPS Compatibility Issue**: QLoRA (4-bit quantization) does NOT work with Apple Silicon MPS. The system automatically falls back to regular LoRA when MPS is detected.

### 3. Checkpoint Management
**Purpose**: Save training progress to resume interrupted training.

**Implementation**:
- Automatic saving every 500 steps
- Keep only latest 3 checkpoints (disk space optimization)
- Automatic resumption capability

## Data Flow

### Step 1: Data Loading (`data_loader.py`)
```python
# Load dataset from Hugging Face
dataset = load_dataset("iamtarun/code_instructions_120k_alpaca")

# Split into train/validation (80/20 split)
dataset = dataset["train"].train_test_split(test_size=0.2)
```

**Important checks**:
- Dataset availability
- Memory requirements
- Data format validation

### Step 2: Preprocessing (`data_loader.py`)
```python
def preprocess_function(examples):
    # Combine instruction and input
    combined_input = f"Instruction: {instruction}\nInput: {input_data}"
    
    # Tokenize inputs and outputs
    tokenized_inputs = tokenizer(combined_input, max_length=512)
    tokenized_labels = tokenizer(examples['output'], max_length=512)
    
    return {'input_ids': tokenized_inputs, 'labels': tokenized_labels}
```

**Key techniques**:
- **Padding**: Ensures uniform sequence length
- **Truncation**: Prevents memory overflow
- **Label creation**: Creates targets for training

### Step 3: Model Setup (`model_setup.py`)
```python
# Load base model with quantization
model = AutoModelForCausalLM.from_pretrained(
    "distilgpt2",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
)

# Apply LoRA configuration
lora_config = LoraConfig(
    r=8,                    # Rank (smaller = less memory)
    lora_alpha=32,          # Scaling factor
    lora_dropout=0.1,       # Regularization
    target_modules=["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
)
```

### Step 4: Training (`trainer.py`)
```python
# Training arguments optimized for MacBook GPU
training_args = TrainingArguments(
    per_device_train_batch_size=4,      # Small batch for memory
    gradient_accumulation_steps=8,      # Simulate larger batch
    learning_rate=5e-5,                 # Conservative learning rate
    fp16=False,                         # Disabled for MPS compatibility
    save_steps=500,                     # Frequent checkpoints
    eval_steps=500,                     # Regular evaluation
    warmup_steps=100,                   # Gradual warmup
    lr_scheduler_type="cosine"          # Smooth decay
)
```

## Important Checks & Validations

### 1. System Resource Check (`main.py`)
```python
def check_system_resources():
    # Check available memory
    memory = psutil.virtual_memory()
    print(f"Available RAM: {memory.available / (1024**3):.2f} GB")
    
    # Check device compatibility
    if Config.DEVICE.type == "mps":
        print("✓ Using Apple Silicon GPU (MPS)")
    elif Config.DEVICE.type == "cuda":
        print("✓ Using NVIDIA GPU")
    else:
        print("⚠ Using CPU (training will be slow)")
```

**Critical validations**:
- **Memory availability**: Minimum 8GB recommended
- **GPU compatibility**: MPS for Apple Silicon, CUDA for NVIDIA
- **Disk space**: 10GB+ for checkpoints and models

### 2. Model Compatibility Check
```python
# Check if bitsandbytes is available for quantization
try:
    import bitsandbytes as bnb
    # Enable 4-bit quantization
except ImportError:
    print("⚠ bitsandbytes not installed, using regular model")
    # Fall back to full precision
```

### 3. Checkpoint Validation
```python
def get_checkpoint_info():
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    return {
        "total_checkpoints": len(checkpoint_dirs),
        "checkpoints": [{"name": name, "step": int(name.split("-")[-1])} 
                       for name in checkpoint_dirs]
    }
```

## Optimization Techniques

### 1. Memory Optimizations

**Gradient Accumulation**:
```python
gradient_accumulation_steps=8  # Accumulate gradients over 8 steps
# Effective batch size = 4 × 8 = 32
```

**Mixed Precision Training**:
```python
fp16=True  # Use 16-bit precision (disabled for MPS)
```

**Gradient Checkpointing**:
```python
model.gradient_checkpointing_enable()  # Trade compute for memory
```

### 2. Training Optimizations

**Learning Rate Scheduling**:
```python
lr_scheduler_type="cosine"  # Smooth cosine decay
warmup_steps=100           # Gradual warmup
```

**Gradient Clipping**:
```python
max_grad_norm=1.0  # Prevent gradient explosion
```

**Batch Size Optimization**:
```python
per_device_train_batch_size=4  # Reduced for memory efficiency
per_device_eval_batch_size=8   # Larger for evaluation
```

### 3. Checkpoint Optimizations

**Frequent Saving**:
```python
save_steps=500  # Save every 500 steps (not every epoch)
```

**Disk Space Management**:
```python
save_total_limit=3  # Keep only latest 3 checkpoints
```

## MPS & QLoRA Configuration Analysis

### Device Detection & Configuration
```python
@staticmethod
def get_device():
    """Get the best available device for the current system."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
```

**Device Priority**: CUDA > MPS > CPU

### QLoRA Configuration (4-bit Quantization)
```python
LORA_CONFIG = {
    "load_in_4bit": True,                    # 4-bit quantization
    "bnb_4bit_compute_dtype": "float16",     # Compute in float16
    "bnb_4bit_use_double_quant": True,       # Nested quantization
    "bnb_4bit_quant_type": "nf4",           # NormalFloat4 quantization
    "r": 8,                                  # LoRA rank
    "lora_alpha": 32,                        # Scaling factor
    "lora_dropout": 0.1,                     # Regularization
    "target_modules": [                      # Target layers for LoRA
        "attn.c_attn", "attn.c_proj", 
        "mlp.c_fc", "mlp.c_proj"
    ]
}
```

### MPS Compatibility Issues & Solutions

**❌ Problem**: QLoRA (4-bit quantization) does NOT work with Apple Silicon MPS
```python
# In model_setup.py
if Config.DEVICE.type == "mps":
    print("⚠ MPS detected, using regular LoRA (bitsandbytes not compatible with MPS)")
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
    model.to(Config.DEVICE)
else:
    # Configure 4-bit quantization for CUDA only
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
```

**Why QLoRA doesn't work with MPS**:
1. **bitsandbytes library**: Not compatible with Apple Silicon
2. **CUDA-specific optimizations**: 4-bit quantization relies on CUDA kernels
3. **Memory management**: Different memory architecture on Apple Silicon

**✅ Solution**: Automatic fallback to regular LoRA
- **Memory impact**: ~4x more memory usage on MPS
- **Performance**: Still efficient due to LoRA's parameter reduction
- **Compatibility**: Full MPS support maintained

### MPS-Optimized Training Configuration
```python
TRAINING_ARGS = {
    "fp16": False,                           # ❌ Disabled for MPS compatibility
    "per_device_train_batch_size": 4,        # ✅ Reduced for memory
    "gradient_accumulation_steps": 8,        # ✅ Simulate larger batches
    "use_gradient_checkpointing": False,     # ❌ Disabled for MPS
    "dataloader_num_workers": 2,             # ✅ Multi-worker loading
}
```

**MPS-Specific Optimizations**:
- **No FP16**: Mixed precision disabled (MPS limitation)
- **Smaller batches**: 4 instead of 16 for memory efficiency
- **Gradient accumulation**: 8 steps to simulate batch size of 32
- **No gradient checkpointing**: Not compatible with MPS

### Memory Optimization for MPS
```python
MEMORY_OPTIMIZATION = {
    "use_gradient_checkpointing": False,     # MPS incompatible
    "use_8bit_optimizer": True,              # ✅ Works with MPS
    "use_4bit_quantization": True,           # ❌ Falls back to regular LoRA
    "max_memory_MB": 8000,                   # Limit memory usage
}
```

## Key Parameters

### LoRA Parameters
| Parameter | Value | Purpose | Impact |
|-----------|-------|---------|---------|
| `r` (rank) | 8 | Low-rank dimension | Lower = less memory, less capacity |
| `lora_alpha` | 32 | Scaling factor | Higher = stronger adaptation |
| `lora_dropout` | 0.1 | Regularization | Prevents overfitting |
| `target_modules` | ["attn.c_attn", ...] | Target layers | Which layers to adapt |

### Training Parameters
| Parameter | Value | Purpose | Impact |
|-----------|-------|---------|---------|
| `learning_rate` | 5e-5 | Step size | Lower = stable, slower convergence |
| `batch_size` | 4 | Samples per step | Lower = less memory, more noise |
| `gradient_accumulation` | 8 | Effective batch | Simulates larger batches |
| `warmup_steps` | 100 | Gradual start | Prevents early instability |
| `max_grad_norm` | 1.0 | Gradient clipping | Prevents explosion |

### Memory Parameters
| Parameter | Value | Purpose | Impact |
|-----------|-------|---------|---------|
| `load_in_4bit` | True | Quantization | 75% memory reduction (CUDA only) |
| `bnb_4bit_use_double_quant` | True | Nested quantization | Additional 10% reduction (CUDA only) |
| `max_memory_MB` | 8000 | Memory limit | Prevents OOM errors |
| `fp16` | False | Mixed precision | Disabled for MPS compatibility |
| `use_gradient_checkpointing` | False | Memory trade-off | Disabled for MPS compatibility |

## Edge Cases & Error Handling

### 1. Memory Issues
**Symptoms**: Out of Memory (OOM) errors
**Solutions**:
```python
# Reduce batch size
per_device_train_batch_size=2

# Increase gradient accumulation
gradient_accumulation_steps=16

# Enable 8-bit optimizer
use_8bit_optimizer=True
```

### 2. Training Interruption
**Symptoms**: Training stops unexpectedly
**Solutions**:
```python
# Automatic resumption
resume_from_checkpoint=True

# Manual resumption
latest_checkpoint = find_latest_checkpoint()
trainer.train(resume_from_checkpoint=latest_checkpoint)
```

### 3. Device Compatibility
**MPS (Apple Silicon) Issues**:
```python
# bitsandbytes doesn't work with MPS
if Config.DEVICE.type == "mps":
    print("⚠ MPS detected, using regular LoRA (bitsandbytes not compatible with MPS)")
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
    model.to(Config.DEVICE)
else:
    # Configure 4-bit quantization for CUDA only
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
```

**What actually happens**:
- **On CUDA**: Full QLoRA with 4-bit quantization (75% memory reduction)
- **On MPS**: Regular LoRA without quantization (still 99% parameter reduction)
- **On CPU**: Regular LoRA (slow but functional)

### 4. Dataset Issues
**Memory overflow during tokenization**:
```python
# Reduce max_length
max_length=256  # Instead of 512

# Use smaller dataset for testing
dataset = dataset.select(range(1000))  # Test with subset
```

## Training Process

### Phase 1: Initialization
1. **System Check**: Validate resources and device
2. **Data Loading**: Load and preprocess dataset
3. **Model Setup**: Load base model with LoRA
4. **Trainer Configuration**: Set up training arguments

### Phase 2: Training Loop
1. **Forward Pass**: Generate predictions
2. **Loss Calculation**: Compute training loss
3. **Backward Pass**: Compute gradients
4. **Gradient Accumulation**: Accumulate over multiple steps
5. **Parameter Update**: Update LoRA parameters
6. **Checkpointing**: Save progress every 500 steps

### Phase 3: Evaluation
1. **Validation**: Evaluate on test set
2. **Metrics**: Compute loss and accuracy
3. **Model Selection**: Save best model

### Phase 4: Finalization
1. **Model Saving**: Save fine-tuned model
2. **Cleanup**: Remove old checkpoints
3. **Comparison**: Compare with base model

## Evaluation & Comparison

### Model Comparison (`generation.py`)
```python
def compare_models(base_model, fine_tuned_model, inputs):
    base_outputs = [generate_text(base_model, input) for input in inputs]
    fine_tuned_outputs = [generate_text(fine_tuned_model, input) for input in inputs]
    return base_outputs, fine_tuned_outputs
```

### Sample Test Cases
```python
sample_inputs = [
    "Write a Python function to reverse a string",
    "Write a function to check if a number is prime",
    "Create a function to find the maximum element in a list",
    "Write a function to calculate the factorial of a number"
]
```

### Output Analysis
- **Base Model**: Generic responses, may not follow instructions
- **Fine-tuned Model**: Task-specific, instruction-following responses
- **Quantitative Metrics**: Loss reduction, accuracy improvement
- **Qualitative Assessment**: Human evaluation of output quality

## Best Practices for Undergraduate Studies

### 1. Experimentation
- **Start Small**: Use small datasets for initial testing
- **Monitor Resources**: Watch memory and GPU usage
- **Save Checkpoints**: Always enable checkpointing
- **Document Changes**: Keep track of parameter modifications
- **Device Testing**: Test on both MPS and CUDA if available

### 5. MPS-Specific Considerations
- **QLoRA Limitation**: Understand that 4-bit quantization won't work on MPS
- **Memory Management**: Use smaller batch sizes (4 instead of 16)
- **Performance Monitoring**: Monitor MPS memory usage closely
- **Fallback Strategy**: Have CPU fallback for critical experiments

### 2. Debugging
- **System Check**: Always run system_check.py first
- **Memory Monitoring**: Watch for OOM errors
- **Gradient Monitoring**: Check for gradient explosion
- **Loss Tracking**: Monitor training and validation loss

### 3. Optimization
- **Batch Size Tuning**: Start small, increase gradually
- **Learning Rate Tuning**: Use conservative values initially
- **LoRA Rank Tuning**: Experiment with different r values
- **Checkpoint Frequency**: Balance between safety and disk space

### 4. Evaluation
- **Multiple Metrics**: Use both quantitative and qualitative measures
- **Comparative Analysis**: Always compare with base model
- **Error Analysis**: Understand where model fails
- **Human Evaluation**: Include human assessment of outputs

This comprehensive guide covers all aspects of the LoRA fine-tuning pipeline, providing undergraduate students with a complete understanding of the system's architecture, implementation details, and best practices for experimentation and optimization. 


-----------------improvements--------------------------------
## **📋 Parameter Changes & Their Benefits:**

### **🔧 Training Parameters (config.py):**

**Learning Rate:**
- **Before:** `2e-5` → **After:** `1e-5`
- **Benefit:** Prevents overfitting, more stable training

**Batch Sizes:**
- **Train:** `4` → `2` | **Eval:** `8` → `4`
- **Benefit:** Reduces memory usage, better for MacBook GPU

**Gradient Accumulation:**
- **Before:** `8` → **After:** `16`
- **Benefit:** Maintains effective batch size (2×16=32) while using less memory

**Training Duration:**
- **Epochs:** `4` → `2` | **Warmup:** `300` → `100`
- **Benefit:** Faster training, less overfitting

**Gradient Norm:**
- **Before:** `1.0` → **After:** `0.5`
- **Benefit:** More stable gradients, prevents exploding gradients

### **🎯 Generation Parameters (config.py):**

**Max Length:**
- **Before:** `100` → **After:** `200`
- **Benefit:** Allows longer code generation

**Sampling Parameters:**
- **Temperature:** `0.7` (controls randomness)
- **Top-p:** `0.9` (nucleus sampling)
- **Top-k:** `50` (top-k sampling)
- **Repetition Penalty:** `1.2` (prevents repetition)
- **No-repeat-ngram:** `3` (prevents n-gram repetition)
- **Benefit:** Better quality, less repetitive outputs

**Removed:**
- **`early_stopping`** (not compatible with this model)
- **Benefit:** No more generation warnings

### **🔧 Generation Function (generation.py):**

**Max Length Fix:**
- **Before:** `max_length` → **After:** `max_new_tokens`
- **Benefit:** No more "input length > max_length" errors

**Attention Mask:**
- **Added:** Explicit attention mask handling
- **Benefit:** No more attention mask warnings

### **📁 Folder Structure:**

**Consolidated:**
- **Before:** Scattered across `./results/`, `./checkpoints/`, `./logs/`
- **After:** Everything in `./results/`
- **Benefit:** Clean organization, easier to manage

### **🔄 Resume Logic:**

**Fixed:**
- **Before:** Always found latest checkpoint even when user said "n"
- **After:** Truly starts fresh when user says "n"
- **Benefit:** User control over training behavior

### **📊 Overall Impact:**

✅ **Stability:** Reduced overfitting, more stable training
✅ **Memory:** Better MacBook GPU compatibility
✅ **Quality:** Better generation parameters
✅ **Reliability:** Fixed generation errors and warnings
✅ **Organization:** Clean folder structure
✅ **Control:** Proper resume functionality

**Result:** Training should be more stable, faster, and produce better outputs!
-----------------improvements--------------------------------
