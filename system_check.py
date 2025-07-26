"""
System Check Script for LoRA Fine-tuning
Analyzes system resources and provides recommendations for optimal training configuration.
"""

import torch
import psutil
import platform
import os
from config import Config


def check_system_resources():
    """
    Comprehensive system resource check and recommendations.
    """
    print("="*80)
    print("SYSTEM RESOURCE ANALYSIS")
    print("="*80)
    
    # System Information
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Python Version: {platform.python_version()}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Memory Analysis
    memory = psutil.virtual_memory()
    print(f"\nMEMORY ANALYSIS:")
    print(f"  Total RAM: {memory.total / (1024**3):.2f} GB")
    print(f"  Available RAM: {memory.available / (1024**3):.2f} GB")
    print(f"  Memory Usage: {memory.percent}%")
    print(f"  Used RAM: {memory.used / (1024**3):.2f} GB")
    
    # Disk Space
    disk = psutil.disk_usage('.')
    print(f"\nDISK SPACE:")
    print(f"  Total: {disk.total / (1024**3):.2f} GB")
    print(f"  Available: {disk.free / (1024**3):.2f} GB")
    print(f"  Usage: {disk.percent}%")
    
    # GPU/Device Analysis
    print(f"\nDEVICE ANALYSIS:")
    print(f"  Configured Device: {Config.DEVICE}")
    
    if torch.cuda.is_available():
        print(f"  CUDA Available: ✓")
        print(f"  CUDA Version: Available")
        gpu_count = torch.cuda.device_count()
        print(f"  GPU Count: {gpu_count}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"    GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    elif torch.backends.mps.is_available():
        print(f"  MPS Available: ✓")
        print(f"  Using Apple Silicon GPU")
        
        # Try to get MPS memory info (if available)
        try:
            # Use a more compatible approach for MPS device info
            print(f"  MPS Device: Apple Silicon GPU")
        except Exception:
            print(f"  MPS Device: Apple Silicon GPU")
    
    else:
        print(f"  GPU Not Available: ⚠")
        print(f"  Using CPU (training will be very slow)")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    
    # Memory recommendations
    if memory.available < 4 * (1024**3):  # Less than 4GB
        print(f"  ⚠ CRITICAL: Very low memory ({memory.available / (1024**3):.1f} GB available)")
        print(f"    - Reduce batch size to 2")
        print(f"    - Increase gradient accumulation to 16")
        print(f"    - Consider using CPU training")
        print(f"    - Close other applications")
    elif memory.available < 8 * (1024**3):  # Less than 8GB
        print(f"  ⚠ LOW MEMORY: {memory.available / (1024**3):.1f} GB available")
        print(f"    - Reduce batch size to 4")
        print(f"    - Enable gradient checkpointing")
        print(f"    - Consider 8-bit optimization")
    elif memory.available < 16 * (1024**3):  # Less than 16GB
        print(f"  ✓ MODERATE MEMORY: {memory.available / (1024**3):.1f} GB available")
        print(f"    - Current configuration should work")
        print(f"    - Monitor memory usage during training")
    else:
        print(f"  ✓ GOOD MEMORY: {memory.available / (1024**3):.1f} GB available")
        print(f"    - Can increase batch size if needed")
        print(f"    - Consider larger models")
    
    # Disk recommendations
    if disk.free < 10 * (1024**3):  # Less than 10GB
        print(f"  ⚠ LOW DISK SPACE: {disk.free / (1024**3):.1f} GB available")
        print(f"    - Clean up old checkpoints")
        print(f"    - Reduce checkpoint frequency")
        print(f"    - Use external storage if available")
    else:
        print(f"  ✓ SUFFICIENT DISK SPACE: {disk.free / (1024**3):.1f} GB available")
    
    # Device recommendations
    if Config.DEVICE.type == "mps":
        print(f"  ✓ Using Apple Silicon GPU (MPS)")
        print(f"    - Good for training on MacBook")
        print(f"    - Monitor Activity Monitor for memory usage")
    elif Config.DEVICE.type == "cuda":
        print(f"  ✓ Using NVIDIA GPU")
        print(f"    - Optimal for training")
    else:
        print(f"  ⚠ Using CPU")
        print(f"    - Training will be very slow")
        print(f"    - Consider using cloud GPU if available")
    
    # Configuration recommendations
    print(f"\nCONFIGURATION RECOMMENDATIONS:")
    
    current_batch_size = Config.TRAINING_ARGS["per_device_train_batch_size"]
    current_grad_accum = Config.TRAINING_ARGS["gradient_accumulation_steps"]
    effective_batch = current_batch_size * current_grad_accum
    
    print(f"  Current effective batch size: {effective_batch}")
    print(f"  Current batch size: {current_batch_size}")
    print(f"  Current gradient accumulation: {current_grad_accum}")
    
    if memory.available < 4 * (1024**3):
        print(f"  Recommended batch size: 2")
        print(f"  Recommended gradient accumulation: 16")
    elif memory.available < 8 * (1024**3):
        print(f"  Recommended batch size: 4")
        print(f"  Recommended gradient accumulation: 8")
    else:
        print(f"  Current configuration is appropriate")
    
    print("="*80)


def check_training_readiness():
    """
    Check if the system is ready for training.
    """
    print("\nTRAINING READINESS CHECK:")
    
    # Check if required directories exist
    required_dirs = [
        Config.TRAINING_ARGS["output_dir"],
        Config.TRAINING_ARGS["logging_dir"]
    ]
    
    missing_dirs = []
    for directory in required_dirs:
        if not os.path.exists(directory):
            missing_dirs.append(directory)
    
    if missing_dirs:
        print(f"  ⚠ Missing directories: {missing_dirs}")
        print(f"    - Will be created automatically during training")
    else:
        print(f"  ✓ All required directories exist")
    
    # Check PyTorch installation
    try:
        import transformers
        print(f"  ✓ Transformers installed: {transformers.__version__}")
    except ImportError:
        print(f"  ⚠ Transformers not installed")
    
    try:
        import peft
        print(f"  ✓ PEFT installed: {peft.__version__}")
    except ImportError:
        print(f"  ⚠ PEFT not installed")
    
    try:
        import datasets
        print(f"  ✓ Datasets installed: {datasets.__version__}")
    except ImportError:
        print(f"  ⚠ Datasets not installed")
    
    # Check if we can load the model
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_CHECKPOINT)
        model = AutoModelForCausalLM.from_pretrained(Config.MODEL_CHECKPOINT)
        print(f"  ✓ Model loading test successful")
        del model, tokenizer  # Clean up
    except Exception as e:
        print(f"  ⚠ Model loading test failed: {e}")
    
    print("\nREADY TO TRAIN!" if Config.DEVICE.type != "cpu" else "\nREADY TO TRAIN (but will be slow on CPU)")


def main():
    """
    Main function to run system checks.
    """
    check_system_resources()
    check_training_readiness()
    
    print("\n" + "="*80)
    print("SYSTEM CHECK COMPLETE")
    print("="*80)
    
    # Final recommendations
    print("\nNEXT STEPS:")
    print("1. If all checks pass, run: python main.py")
    print("2. If you have memory issues, adjust config.py")
    print("3. Monitor training with: python checkpoint_manager.py list")
    print("4. Resume interrupted training with: python main.py --resume")


if __name__ == "__main__":
    main() 