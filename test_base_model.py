"""
Test script for using the base model (without fine-tuning).
This script loads the base model and tests text generation capabilities.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import Config
from generation import generate_text, get_sample_inputs


def load_base_model():
    """
    Load the base model and tokenizer.
    
    Returns:
        tuple: (model, tokenizer) - The base model and tokenizer
    """
    print("Loading base model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_CHECKPOINT)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(Config.MODEL_CHECKPOINT)
    model.to(Config.DEVICE)
    
    print(f"✓ Base model loaded successfully on {Config.DEVICE}")
    print(f"  Model: {Config.MODEL_CHECKPOINT}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer


def test_simple_generation(model, tokenizer):
    """Test simple text generation."""
    print("\n" + "="*50)
    print("SIMPLE GENERATION TEST")
    print("="*50)
    
    test_inputs = [
        "Hello, how are you?",
        "The weather is",
        "Python is a programming language",
        "Write a function to"
    ]
    
    for i, input_text in enumerate(test_inputs, 1):
        print(f"\nTest {i}:")
        print(f"Input: {input_text}")
        
        try:
            generated_text = generate_text(model, tokenizer, input_text, max_length=50)
            print(f"Generated: {generated_text}")
        except Exception as e:
            print(f"Error: {e}")
        
        print("-" * 40)


def test_code_generation(model, tokenizer):
    """Test code generation capabilities."""
    print("\n" + "="*50)
    print("CODE GENERATION TEST")
    print("="*50)
    
    code_prompts = [
        "Write a Python function to reverse a string",
        "Create a function to check if a number is prime",
        "Write a function to find the maximum element in a list",
        "Create a function to calculate factorial"
    ]
    
    for i, prompt in enumerate(code_prompts, 1):
        print(f"\nCode Test {i}:")
        print(f"Prompt: {prompt}")
        
        try:
            generated_code = generate_text(model, tokenizer, prompt, max_length=150)
            print(f"Generated Code:\n{generated_code}")
        except Exception as e:
            print(f"Error: {e}")
        
        print("-" * 50)


def test_model_capabilities(model, tokenizer):
    """Test various model capabilities."""
    print("\n" + "="*50)
    print("MODEL CAPABILITIES TEST")
    print("="*50)
    
    # Test different generation parameters
    test_input = "The future of artificial intelligence"
    
    print(f"Base input: {test_input}")
    
    # Test with different max_lengths
    for max_len in [20, 50, 100]:
        print(f"\nMax length: {max_len}")
        try:
            generated = generate_text(model, tokenizer, test_input, max_length=max_len)
            print(f"Output: {generated}")
        except Exception as e:
            print(f"Error: {e}")
    
    # Test temperature-like behavior (if supported)
    print(f"\nTesting generation with different parameters...")
    try:
        # Generate multiple times to see variation
        for i in range(3):
            generated = generate_text(model, tokenizer, test_input, max_length=30)
            print(f"Variation {i+1}: {generated}")
    except Exception as e:
        print(f"Error: {e}")


def interactive_mode(model, tokenizer):
    """Interactive mode for testing the model."""
    print("\n" + "="*50)
    print("INTERACTIVE MODE")
    print("="*50)
    print("Type your prompts (type 'quit' to exit):")
    
    while True:
        try:
            user_input = input("\nEnter prompt: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Exiting interactive mode...")
                break
            
            if not user_input:
                continue
            
            # Get max length from user
            try:
                max_len = int(input("Max length (default 100): ") or "100")
            except ValueError:
                max_len = 100
            
            print(f"\nGenerating with max_length={max_len}...")
            generated = generate_text(model, tokenizer, user_input, max_length=max_len)
            print(f"Generated: {generated}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function to run all tests."""
    print("Base Model Test Script")
    print("="*50)
    
    # Load the base model
    model, tokenizer = load_base_model()
    
    # Run tests
    # test_simple_generation(model, tokenizer)
    # test_code_generation(model, tokenizer)
    # test_model_capabilities(model, tokenizer)
    
    # Ask if user wants interactive mode
    try:
        choice = input("\nWould you like to enter interactive mode? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            interactive_mode(model, tokenizer)
    except KeyboardInterrupt:
        print("\nExiting...")
    
    print("\nBase model testing completed!")


if __name__ == "__main__":
    main() 