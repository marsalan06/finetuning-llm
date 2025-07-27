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
    
    # LoRA configuration - FIXED for DistilGPT2 and MPS compatibility
    LORA_CONFIG = {
        "task_type": "CAUSAL_LM",
        "inference_mode": False,
        "r": 8,  # Rank
        "lora_alpha": 32,  # Scaling factor
        "lora_dropout": 0.1,  # Dropout rate
        # FIXED: Correct target modules for DistilGPT2
        # "target_modules": [
        #     "q_lin",    # Query linear layer
        #     "k_lin",    # Key linear layer  
        #     "v_lin",    # Value linear layer
        #     "out_lin",  # Output linear layer
        #     "fc",       # Feed-forward layer
        #     "proj"      # Projection layer
        # ],
        "target_modules": ["c_attn", "c_proj"],

        # REMOVED: QLoRA configs for MPS compatibility
        "load_in_4bit": False,  # Disabled for MPS
        "bnb_4bit_compute_dtype": None,  # Removed
        "bnb_4bit_use_double_quant": False,  # Removed
        "bnb_4bit_quant_type": None,  # Removed
    }
    
    # Training configuration - OPTIMIZED FOR BETTER RESULTS
    TRAINING_ARGS = {
        "output_dir": "./results",
        "eval_strategy": "steps",  # Evaluate by steps for more frequent checkpoints
        "eval_steps": 1000,  # INCREASED: Evaluate every 1000 steps (was 500)
        "save_strategy": "steps",  # Save by steps for more frequent saves
        "save_steps": 1000,  # INCREASED: Save every 1000 steps (was 500)
        "save_total_limit": 3,  # Keep only the last 3 checkpoints to save disk space
        "learning_rate": 1e-5,  # Good learning rate
        "per_device_train_batch_size": 2,  # Good batch size for stability
        "per_device_eval_batch_size": 4,   # Good eval batch size
        "gradient_accumulation_steps": 16,  # Good effective batch size
        "num_train_epochs": 3,  # INCREASED: Train for 3 epochs for better results
        "weight_decay": 0.01,
        "logging_dir": "./results/logs",  # Store logs inside results folder
        "logging_steps": 100,  # Good logging frequency
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",  # Use eval_loss for best model selection
        "report_to": "none",
        "fp16": False,  # Disabled for MPS compatibility
        "dataloader_num_workers": 2,  # Use multiple workers for data loading
        "remove_unused_columns": False,  # Required for some datasets
        "warmup_steps": 100,  # Good warmup steps
        "lr_scheduler_type": "cosine",  # Cosine learning rate scheduling
        "max_grad_norm": 0.5,  # Good gradient clipping
        "prediction_loss_only": True,  # Only compute loss during evaluation
    }
    
    # Generation configuration
    GENERATION_CONFIG = {
        "max_length": 200,  # Good max length
        "num_return_sequences": 1,
        "do_sample": True,
        "temperature": 0.8,  # Good temperature for creativity
        "top_p": 0.95,  # Good top-p for diversity
        "top_k": 50,  # Good top-k for diversity
        "repetition_penalty": 1.0,  # Good repetition penalty
        "no_repeat_ngram_size": 1,  # Good n-gram size
    }
    
    # File paths - ALL CONSOLIDATED IN RESULTS FOLDER
    MODEL_SAVE_PATH = "./results/fine_tuned_model"
    PREDICTIONS_CSV_PATH = "./results/generated_outputs.csv"
    
    # Memory optimization settings - UPDATED for MPS
    MEMORY_OPTIMIZATION = {
        "use_gradient_checkpointing": False,  # Disabled for MPS compatibility
        "use_8bit_optimizer": True,  # Enable 8-bit optimizer for memory efficiency
        "use_4bit_quantization": False,  # DISABLED: 4-bit quantization not compatible with MPS
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