"""
Example usage script demonstrating how to use the modular components.
This script shows how to use each module independently for custom workflows.
UPDATED: Now works with new Alpaca-style prompt format and updated target modules
"""

from config import Config
from data_loader import load_data, preprocess_data, setup_tokenizer
from model_setup import setup_model, setup_lora, print_model_info
from trainer import setup_trainer, train_model, save_model
from generation import generate_text, get_sample_inputs


def example_data_loading():
    """Example of loading and preprocessing data."""
    print("=== Data Loading Example ===")
    
    # Load dataset
    dataset = load_data()
    
    # Setup tokenizer
    tokenizer = setup_tokenizer()
    
    # Preprocess data
    tokenized_dataset = preprocess_data(dataset, tokenizer)
    
    print(f"Dataset shape: {tokenized_dataset}")
    print(f"Sample data: {tokenized_dataset['train'][0]}")
    
    return tokenized_dataset, tokenizer


def example_model_setup():
    """Example of setting up model with LoRA."""
    print("\n=== Model Setup Example ===")
    
    # Setup base model
    model, tokenizer = setup_model()
    print_model_info(model, "Base Model")
    
    # Apply LoRA
    model = setup_lora(model, tokenizer)
    print_model_info(model, "LoRA Model")
    
    return model, tokenizer


def example_training(model, tokenizer, train_dataset, eval_dataset):
    """Example of training the model."""
    print("\n=== Training Example ===")
    
    # Setup trainer
    trainer = setup_trainer(model, tokenizer, train_dataset, eval_dataset)
    
    # Train model
    results = train_model(trainer)
    
    # Save model
    save_model(model, tokenizer)
    
    return results


def example_generation(model, tokenizer):
    """Example of text generation."""
    print("\n=== Generation Example ===")
    
    # Get sample inputs
    sample_inputs = get_sample_inputs()
    
    # Generate text for each input
    for i, input_text in enumerate(sample_inputs, 1):
        print(f"\nTest {i}:")
        print(f"Input: {input_text}")
        
        generated_text = generate_text(model, tokenizer, input_text)
        print(f"Generated: {generated_text}")
        print("-" * 50)


def example_custom_config():
    """Example of using custom configuration."""
    print("\n=== Custom Configuration Example ===")
    
    # You can modify Config class attributes or create custom config
    # UPDATED: Now shows the correct target modules for DistilGPT2
    custom_config = {
        "MODEL_CHECKPOINT": "distilgpt2",
        "MAX_LENGTH": 256,  # Shorter sequences for faster training
        "LORA_CONFIG": {
            "r": 4,  # Lower rank for faster training
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            # UPDATED: Correct target modules for DistilGPT2
            "target_modules": ["c_attn", "c_proj"]  # Simplified version
        }
    }
    
    print("Custom configuration:")
    for key, value in custom_config.items():
        print(f"  {key}: {value}")
    
    print("\n📝 Note: Using simplified target modules for faster training")
    print("🎯 Current target modules:", Config.LORA_CONFIG["target_modules"])


def example_prompt_formatting():
    """Example of the new prompt formatting."""
    print("\n=== Prompt Formatting Example ===")
    
    # Show the new Alpaca-style format
    sample_prompts = [
        "Write a Python function to reverse a string",
        "Create a function to check if a number is prime",
        "Generate a Python code for crawling a website"
    ]
    
    print("New Alpaca-style prompt format:")
    for prompt in sample_prompts:
        formatted = f"{prompt}\n\n### Response:\n"
        print(f"Input: {prompt}")
        print(f"Formatted: {formatted}")
        print("-" * 30)
    
    print("✅ This format works better for generation quality")


def main():
    """Main example function."""
    print("LoRA Fine-Tuning - Example Usage")
    print("="*50)
    print("📝 UPDATED: Now uses Alpaca-style prompt format")
    print("🎯 Target modules:", Config.LORA_CONFIG["target_modules"])
    
    # Example 1: Data loading
    tokenized_dataset, tokenizer = example_data_loading()
    
    # Example 2: Model setup
    model, tokenizer = example_model_setup()
    
    # Example 3: Custom configuration
    example_custom_config()
    
    # Example 4: Prompt formatting
    example_prompt_formatting()
    
    # Example 5: Training (commented out to avoid long execution)
    # Uncomment the following lines to run training
    # results = example_training(model, tokenizer, 
    #                          tokenized_dataset["train"], 
    #                          tokenized_dataset["test"])
    
    # Example 6: Generation
    example_generation(model, tokenizer)
    
    print("\nExample usage completed!")
    print("\n💡 Key Updates:")
    print("  - Alpaca-style prompt format: prompt\\n\\n### Response:\\noutput")
    print("  - Updated target modules for DistilGPT2")
    print("  - MPS-compatible LoRA configuration")


if __name__ == "__main__":
    main() 