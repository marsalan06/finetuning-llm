#!/usr/bin/env python3
"""
Interactive Testing Mode for Model Comparison
Allows real-time testing of base model vs fine-tuned model
UPDATED: Now works with new Alpaca-style prompt format
"""

import torch
import sys
import os
from config import Config
from model_setup import setup_model
from trainer import load_fine_tuned_model
from generation import generate_text


def load_models():
    """Load both base and fine-tuned models."""
    print("🔄 Loading models...")
    
    # Load base model
    print("📥 Loading base model...")
    base_model, base_tokenizer = setup_model()
    print("✅ Base model loaded successfully!")
    
    # Load fine-tuned model
    print("📥 Loading fine-tuned model...")
    try:
        fine_tuned_model, fine_tuned_tokenizer = load_fine_tuned_model()
        print("✅ Fine-tuned model loaded successfully!")
        return base_model, base_tokenizer, fine_tuned_model, fine_tuned_tokenizer
    except FileNotFoundError as e:
        print(f"❌ Error loading fine-tuned model: {e}")
        print("💡 Make sure you've trained the model first with: python main.py")
        return base_model, base_tokenizer, None, None


def test_single_input(base_model, base_tokenizer, fine_tuned_model, fine_tuned_tokenizer, user_input):
    """Test a single input on both models."""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    # Clean up the input - extract just the instruction if it's in the full format
    clean_input = user_input
    if "### Instruction:" in user_input:
        # Extract just the instruction part
        instruction_start = user_input.find("### Instruction:") + len("### Instruction:")
        instruction_end = user_input.find("### Input:") if "### Input:" in user_input else len(user_input)
        clean_input = user_input[instruction_start:instruction_end].strip()
        print(f"🔧 Cleaned input: {clean_input}")
    
    # Generate base model output
    print("🤖 Generating base model output...")
    try:
        base_output = generate_text(base_model, base_tokenizer, clean_input)
        print(f"✅ Base model generated {len(base_output)} characters")
    except Exception as e:
        base_output = f"Error: {e}"
        print(f"❌ Base model error: {e}")
    
    # Generate fine-tuned model output
    if fine_tuned_model is not None:
        print("🎯 Generating fine-tuned model output...")
        try:
            fine_tuned_output = generate_text(fine_tuned_model, fine_tuned_tokenizer, clean_input)
            print(f"✅ Fine-tuned model generated {len(fine_tuned_output)} characters")
        except Exception as e:
            fine_tuned_output = f"Error: {e}"
            print(f"❌ Fine-tuned model error: {e}")
    else:
        fine_tuned_output = "Fine-tuned model not available"
    
    # Print comparison
    print("\n📊 RESULTS:")
    print("-" * 40)
    print(f"📝 Input: {clean_input}")
    print(f"🔵 Base Model: {base_output}")
    print(f"🟢 Fine-tuned Model: {fine_tuned_output}")
    print("-" * 40)


def interactive_mode():
    """Run interactive testing mode."""
    print("🚀 Interactive Model Testing Mode")
    print("=" * 50)
    print("📝 UPDATED: Now uses Alpaca-style prompt format")
    print("🎯 Target modules:", Config.LORA_CONFIG["target_modules"])
    
    # Load models
    base_model, base_tokenizer, fine_tuned_model, fine_tuned_tokenizer = load_models()
    
    if fine_tuned_model is None:
        print("\n⚠️  Fine-tuned model not found!")
        print("💡 To train a fine-tuned model, run: python main.py")
        print("💡 To resume training, run: python main.py --resume")
        print("\n🔄 Continuing with base model only...")
    
    print("\n🎯 Available Commands:")
    print("  'test <your input>' - Test a custom input")
    print("  'help'             - Show this help")
    print("  'quit' or 'exit'   - Exit the program")
    print("  'clear'            - Clear the screen")
    print("\n💡 Example inputs:")
    print("  'Write a Python function to reverse a string'")
    print("  'Create a function to check if a number is prime'")
    print("  'Generate a Python code for crawling a website'")
    
    while True:
        try:
            user_input = input("\n🎯 Enter command: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            elif user_input.lower() == 'help':
                print("\n🎯 Available Commands:")
                print("  'test <your input>' - Test a custom input")
                print("  'help'             - Show this help")
                print("  'quit' or 'exit'   - Exit the program")
                print("  'clear'            - Clear the screen")
                print("\n💡 Example inputs:")
                print("  'Write a Python function to reverse a string'")
                print("  'Create a function to check if a number is prime'")
                print("  'Generate a Python code for crawling a website'")
            
            elif user_input.lower() == 'clear':
                os.system('clear' if os.name == 'posix' else 'cls')
                print("🚀 Interactive Model Testing Mode")
                print("=" * 50)
                print("📝 UPDATED: Now uses Alpaca-style prompt format")
            
            elif user_input.lower().startswith('test'):
                if len(user_input.split()) < 2:
                    print("❌ Please provide an input: test <your input>")
                else:
                    test_input = ' '.join(user_input.split()[1:])
                    test_single_input(base_model, base_tokenizer, fine_tuned_model, fine_tuned_tokenizer, test_input)
            
            elif user_input.strip() == "":
                print("❌ Please provide an input. Type 'help' for available commands.")
            else:
                # Treat as a direct input
                test_single_input(base_model, base_tokenizer, fine_tuned_model, fine_tuned_tokenizer, user_input)
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def quick_test():
    """Run a quick test with sample inputs."""
    print("⚡ Quick Test Mode")
    print("=" * 30)
    print("📝 UPDATED: Now uses Alpaca-style prompt format")
    
    # Load models
    base_model, base_tokenizer, fine_tuned_model, fine_tuned_tokenizer = load_models()
    
    # Sample inputs - updated for better testing
    sample_inputs = [
        "Write a Python function to reverse a string",
        "Create a function to check if a number is prime",
        "Write a function to find the maximum element in a list"
    ]
    
    print(f"\n🧪 Testing {len(sample_inputs)} sample inputs...")
    
    for i, input_text in enumerate(sample_inputs, 1):
        print(f"\n📊 Test {i}/{len(sample_inputs)}")
        test_single_input(base_model, base_tokenizer, fine_tuned_model, fine_tuned_tokenizer, input_text)
    
    print("\n✅ Quick test completed!")


def main():
    """Main function."""
    if len(sys.argv) > 1:
        if sys.argv[1] == '--quick':
            quick_test()
        else:
            print("Usage: python interactive_test.py [--quick]")
            print("  --quick: Run quick test with sample inputs")
            print("  No args: Start interactive mode")
    else:
        interactive_mode()


if __name__ == "__main__":
    main() 