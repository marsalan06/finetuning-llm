"""
Configuration file for LoRA fine-tuning parameters and settings.
Centralizes all hyperparameters to make the code more maintainable.
"""

import torch
import os

class Config:
    """Configuration class containing all hyperparameters and settings."""
    
    # Device configuration - optimized for Docker GPU (CUDA preferred)
    @staticmethod
    def get_device():
        """Get the best available device for the current system. CUDA preferred for Docker/RunPod."""
        if torch.cuda.is_available():
            print("[Config] Using CUDA device (Docker/RunPod GPU)")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("[Config] Using MPS device (Apple Silicon)")
            return torch.device("mps")
        else:
            print("[Config] Using CPU device")
            return torch.device("cpu")
    
    DEVICE = get_device()
    
    # Model configuration
    MODEL_CHECKPOINT = "distilgpt2"
    MAX_LENGTH = 512
    
    # Dataset configuration
    DATASET_NAME = "iamtarun/code_instructions_120k_alpaca"
    TRAIN_TEST_SPLIT = 0.2  # 20% for validation
    
    # QLoRA configuration - 4-bit quantization for memory efficiency
    LORA_CONFIG = {
        "task_type": "CAUSAL_LM",
        "inference_mode": False,
        "r": 8,  # Rank
        "lora_alpha": 32,  # Scaling factor
        "lora_dropout": 0.1,  # Dropout rate
        "target_modules": [
            "attn.c_attn",  # Attention layer: query, key, value weights
            "attn.c_proj",  # Attention layer: projection weights
            "mlp.c_fc",     # MLP layer: fully connected layer
            "mlp.c_proj"    # MLP layer: projection layer
        ],
        "load_in_4bit": True,  # 4-bit quantization
        "bnb_4bit_compute_dtype": "float16",  # Compute dtype for 4-bit
        "bnb_4bit_use_double_quant": True,  # Nested quantization
        "bnb_4bit_quant_type": "nf4",  # NormalFloat4 quantization
    }
    
    # Training configuration - OPTIMIZED FOR DOCKER GPU (RunPod)
    TRAINING_ARGS = {
        "output_dir": "./results",
        "eval_strategy": "steps",  # Changed from "epoch" to "steps" for more frequent checkpoints
        "eval_steps": 500,  # Evaluate every 500 steps
        "save_strategy": "steps",  # Changed from "epoch" to "steps" for more frequent saves
        "save_steps": 500,  # Save every 500 steps
        "save_total_limit": 3,  # Keep only the last 3 checkpoints to save disk space
        "learning_rate": 1e-5,  # REDUCED from 2e-5 to 1e-5 to prevent overfitting
        "per_device_train_batch_size": 2,  # REDUCED from 4 to 2 for better stability
        "per_device_eval_batch_size": 4,   # REDUCED from 8 to 4
        "gradient_accumulation_steps": 16,  # INCREASED from 8 to 16 to maintain effective batch size
        "num_train_epochs": 2,  # REDUCED from 4 to 2 to prevent overfitting
        "weight_decay": 0.01,
        "logging_dir": "./results/logs",  # Store logs inside results folder
        "logging_steps": 100,  # REDUCED from 500 for more frequent logging
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",  # Changed from "accuracy" to "eval_loss"
        "report_to": "none",
        "fp16": False,  # Disabled for MPS compatibility
        "dataloader_num_workers": 2,  # NEW: Use multiple workers for data loading
        "remove_unused_columns": False,  # NEW: Required for some datasets
        "warmup_steps": 100,  # REDUCED from 300 for shorter warmup
        "lr_scheduler_type": "cosine",  # NEW: Cosine learning rate scheduling
        "max_grad_norm": 0.5,  # REDUCED from 1.0 for more stable training
        "prediction_loss_only": True,  # Only compute loss during evaluation
    }
    
    # Generation configuration
    GENERATION_CONFIG = {
        "max_length": 200,  # INCREASED from 100 to allow for longer code generation
        "num_return_sequences": 1,
        "do_sample": True,
        "temperature": 0.7,  # Control randomness
        "top_p": 0.9,  # Nucleus sampling
        "top_k": 50,  # Top-k sampling
        "repetition_penalty": 1.2,  # Prevent repetition
        "no_repeat_ngram_size": 3,  # Prevent n-gram repetition
    }
    
    # File paths - ALL CONSOLIDATED IN RESULTS FOLDER
    MODEL_SAVE_PATH = "./results/fine_tuned_model"
    PREDICTIONS_CSV_PATH = "./results/generated_outputs.csv"
    
    # NEW: Memory optimization settings
    MEMORY_OPTIMIZATION = {
        "use_gradient_checkpointing": False,  # Disabled for MPS compatibility
        "use_8bit_optimizer": True,  # Enable 8-bit optimizer for memory efficiency
        "use_4bit_quantization": True,  # Enable 4-bit quantization
        "max_memory_MB": 8000,  # Limit memory usage (adjust based on your MacBook)
    }
    
    @staticmethod
    def create_directories():
        """Create necessary directories for the training process."""
        directories = [
            Config.TRAINING_ARGS["output_dir"],  # ./results
            Config.TRAINING_ARGS["logging_dir"],  # ./results/logs
            os.path.dirname(Config.MODEL_SAVE_PATH)  # ./results (for fine_tuned_model)
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}") 