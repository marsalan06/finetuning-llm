"""
Configuration file for LoRA fine-tuning parameters and settings.
Centralizes all hyperparameters to make the code more maintainable.
"""

import torch

class Config:
    """Configuration class containing all hyperparameters and settings."""
    
    # Device configuration - optimized for macOS with MPS support
    @staticmethod
    def get_device():
        """Get the best available device for the current system."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    DEVICE = get_device()
    
    # Model configuration
    MODEL_CHECKPOINT = "distilgpt2"
    MAX_LENGTH = 512
    
    # Dataset configuration
    DATASET_NAME = "iamtarun/code_instructions_120k_alpaca"
    TRAIN_TEST_SPLIT = 0.2  # 20% for validation
    
    # LoRA configuration
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
        ]
    }
    
    # Training configuration
    TRAINING_ARGS = {
        "output_dir": "./results",
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 64,
        "num_train_epochs": 3,
        "weight_decay": 0.01,
        "logging_dir": "./logs",
        "logging_steps": 500,
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "report_to": "none"
    }
    
    # Generation configuration
    GENERATION_CONFIG = {
        "max_length": 100,
        "num_return_sequences": 1,
        "do_sample": True
    }
    
    # File paths
    MODEL_SAVE_PATH = "./fine_tuned_model"
    PREDICTIONS_CSV_PATH = "generated_outputs.csv" 