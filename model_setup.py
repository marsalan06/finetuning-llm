"""
Model setup module for initializing the base model and applying LoRA configuration.
Contains functions for setting up the model, tokenizer, and LoRA adapters.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from config import Config


def setup_model(model_checkpoint=None):
    """
    Sets up the base model and tokenizer with QLoRA quantization.
    
    Args:
        model_checkpoint (str, optional): Model checkpoint to use. 
                                       Defaults to Config.MODEL_CHECKPOINT.
    
    Returns:
        tuple: (model, tokenizer) - The configured model and tokenizer
    """
    if model_checkpoint is None:
        model_checkpoint = Config.MODEL_CHECKPOINT
        
    print(f"Setting up model with QLoRA: {model_checkpoint}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Import bitsandbytes for quantization
    try:
        import bitsandbytes as bnb
        from transformers.utils.quantization_config import BitsAndBytesConfig
        
        # Check if MPS is being used (bitsandbytes doesn't work well with MPS)
        if Config.DEVICE.type == "mps":
            print("⚠ MPS detected, using regular LoRA (bitsandbytes not compatible with MPS)")
            model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
            model.to(Config.DEVICE)
        else:
            # Configure 4-bit quantization for CUDA
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Initialize model with quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_checkpoint,
                quantization_config=quantization_config,
                device_map="auto"  # Automatically handle device placement
            )
            
            print("✓ QLoRA 4-bit quantization enabled")
        
    except ImportError:
        print("⚠ bitsandbytes not installed, falling back to regular model")
        model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
        model.to(Config.DEVICE)
    
    print(f"Base model {model_checkpoint} loaded successfully.")
    print(f"Using device: {Config.DEVICE}")
    
    return model, tokenizer


def setup_lora(model, tokenizer):
    """
    Sets up QLoRA (Quantized LoRA) for fine-tuning.
    
    Args:
        model: The base model to apply LoRA to
        tokenizer: The tokenizer (not used in LoRA setup but kept for consistency)
    
    Returns:
        PeftModel: Model with QLoRA configuration applied
    """
    print("Setting up QLoRA configuration...")
    
    # Create LoRA configuration (quantization is handled in model loading)
    lora_config = LoraConfig(
        task_type=Config.LORA_CONFIG["task_type"],
        inference_mode=Config.LORA_CONFIG["inference_mode"],
        r=Config.LORA_CONFIG["r"],
        lora_alpha=Config.LORA_CONFIG["lora_alpha"],
        lora_dropout=Config.LORA_CONFIG["lora_dropout"],
        target_modules=Config.LORA_CONFIG["target_modules"]
    )
    
    # Apply LoRA configuration to the model
    model = get_peft_model(model, lora_config)
    
    print("QLoRA configuration applied successfully.")
    print(f"QLoRA parameters:")
    print(f"  - Rank (r): {Config.LORA_CONFIG['r']}")
    print(f"  - Alpha: {Config.LORA_CONFIG['lora_alpha']}")
    print(f"  - Dropout: {Config.LORA_CONFIG['lora_dropout']}")
    print(f"  - Target modules: {Config.LORA_CONFIG['target_modules']}")
    print(f"  - 4-bit quantization: {Config.LORA_CONFIG.get('load_in_4bit', False)}")
    
    return model


def print_model_info(model, stage="Base Model"):
    """
    Prints information about the model structure.
    
    Args:
        model: The model to print information about
        stage (str): Stage description (e.g., "Base Model", "LoRA Model")
    """
    print(f"\n{stage} Information:")
    print(f"Model type: {type(model).__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Print LoRA-specific information if it's a PEFT model
    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters() 