"""
Test script to verify device detection and MPS availability on macOS.
Run this to check if your system is properly configured for GPU acceleration.
"""

import torch
from config import Config


def test_device_detection():
    """Test device detection and print system information."""
    print("="*50)
    print("DEVICE DETECTION TEST")
    print("="*50)
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    
    # Check MPS availability (macOS)
    mps_available = torch.backends.mps.is_available()
    print(f"MPS available: {mps_available}")
    
    # Check CPU
    print(f"CPU available: True")
    
    # Show selected device
    print(f"\nSelected device: {Config.DEVICE}")
    
    # Test tensor operations on selected device
    print(f"\nTesting tensor operations on {Config.DEVICE}...")
    try:
        test_tensor = torch.randn(3, 3).to(Config.DEVICE)
        result = torch.mm(test_tensor, test_tensor)
        print(f"✓ Tensor operations successful on {Config.DEVICE}")
        print(f"  Test tensor shape: {test_tensor.shape}")
        print(f"  Result tensor shape: {result.shape}")
    except Exception as e:
        print(f"✗ Error with tensor operations on {Config.DEVICE}: {e}")
    
    print("\n" + "="*50)
    print("RECOMMENDATIONS:")
    
    if cuda_available:
        print("✓ CUDA detected - optimal for training")
    elif mps_available:
        print("✓ MPS detected - good for training on Apple Silicon")
        print("  Note: MPS may be slower than CUDA for some operations")
    else:
        print("⚠ Only CPU available - training will be slow")
        print("  Consider using a machine with GPU acceleration")
    
    print("="*50)


def test_model_loading():
    """Test if we can load a small model on the selected device."""
    print("\n" + "="*50)
    print("MODEL LOADING TEST")
    print("="*50)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("Loading small test model...")
        model_name = "distilgpt2"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(Config.DEVICE)
        
        print(f"✓ Model loaded successfully on {Config.DEVICE}")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test generation
        print("Testing text generation...")
        input_text = "Hello, world"
        inputs = tokenizer(input_text, return_tensors="pt").to(Config.DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(inputs['input_ids'], max_length=20)
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ Generation successful: {generated_text}")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
    
    print("="*50)


if __name__ == "__main__":
    test_device_detection()
    test_model_loading() 