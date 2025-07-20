"""
Main script for LoRA fine-tuning of language models.
This script orchestrates the entire fine-tuning process from data loading to model evaluation.
Includes checkpoint functionality and resource management for MacBook GPU training.
"""

import os
import sys
from config import Config

# Import our custom modules
from data_loader import load_data, preprocess_data, setup_tokenizer
from model_setup import setup_model, setup_lora, print_model_info
from trainer import (
    setup_trainer, train_model, save_model, load_fine_tuned_model,
    find_latest_checkpoint, get_checkpoint_info, cleanup_old_checkpoints
)
from generation import compare_models, print_comparison, save_predictions_to_csv, get_sample_inputs


def check_system_resources():
    """
    Check system resources and provide recommendations.
    """
    print("="*60)
    print("System Resource Check")
    print("="*60)
    
    import torch
    import psutil
    
    # Check available memory
    memory = psutil.virtual_memory()
    print(f"Available RAM: {memory.available / (1024**3):.2f} GB")
    print(f"Total RAM: {memory.total / (1024**3):.2f} GB")
    print(f"Memory usage: {memory.percent}%")
    
    # Check device
    print(f"Using device: {Config.DEVICE}")
    
    if Config.DEVICE.type == "mps":
        print("✓ Using Apple Silicon GPU (MPS)")
    elif Config.DEVICE.type == "cuda":
        print("✓ Using NVIDIA GPU")
    else:
        print("⚠ Using CPU (training will be slow)")
    
    # Recommendations
    if memory.available < 8 * (1024**3):  # Less than 8GB available
        print("⚠ Low memory detected. Consider:")
        print("  - Reducing batch size further")
        print("  - Using gradient checkpointing")
        print("  - Enabling 8-bit optimization")
    
    print("="*60)


def main():
    """
    Main function that orchestrates the entire fine-tuning process.
    """
    print("="*60)
    print("LoRA Fine-Tuning Pipeline with Checkpoint Support")
    print("="*60)
    
    # Check system resources
    check_system_resources()
    
    # Step 1: Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    dataset = load_data()
    tokenizer = setup_tokenizer()
    tokenized_dataset = preprocess_data(dataset, tokenizer)
    
    # Step 2: Setup base model and tokenizer
    print("\n2. Setting up base model...")
    model, tokenizer = setup_model()
    print_model_info(model, "Base Model")
    
    # Step 3: Apply LoRA configuration
    print("\n3. Applying LoRA configuration...")
    model = setup_lora(model, tokenizer)
    print_model_info(model, "LoRA Model")
    
    # Step 4: Setup trainer
    print("\n4. Setting up trainer...")
    trainer = setup_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"]
    )
    
    # Step 5: Check for existing checkpoints
    print("\n5. Checking for existing checkpoints...")
    checkpoint_info = get_checkpoint_info()
    if checkpoint_info["total_checkpoints"] > 0:
        print(f"Found {checkpoint_info['total_checkpoints']} existing checkpoints:")
        for checkpoint in checkpoint_info["checkpoints"]:
            print(f"  - {checkpoint['name']} (step {checkpoint['step']})")
        
        # Ask user if they want to resume
        response = input("\nDo you want to resume from the latest checkpoint? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            latest_checkpoint = find_latest_checkpoint()
            print(f"Resuming from checkpoint: {latest_checkpoint}")
        else:
            latest_checkpoint = None
            print("Starting training from scratch")
    else:
        latest_checkpoint = None
        print("No existing checkpoints found. Starting training from scratch.")
    
    # Step 6: Train the model
    print("\n6. Training the model...")
    training_results = train_model(trainer, resume_from_checkpoint=latest_checkpoint)
    
    # Step 7: Clean up old checkpoints
    print("\n7. Cleaning up old checkpoints...")
    cleanup_old_checkpoints(keep_latest=3)
    
    # Step 8: Save the fine-tuned model
    print("\n8. Saving the fine-tuned model...")
    save_model(model, tokenizer)
    
    # Step 9: Load the fine-tuned model for evaluation
    print("\n9. Loading fine-tuned model for evaluation...")
    fine_tuned_model, fine_tuned_tokenizer = load_fine_tuned_model()
    
    # Step 10: Generate sample inputs
    print("\n10. Preparing sample inputs for evaluation...")
    sample_inputs = get_sample_inputs()
    
    # Step 11: Compare base and fine-tuned models
    print("\n11. Comparing base and fine-tuned models...")
    base_model, base_tokenizer = setup_model()  # Load fresh base model
    
    base_outputs, fine_tuned_outputs = compare_models(
        base_model=base_model,
        base_tokenizer=base_tokenizer,
        fine_tuned_model=fine_tuned_model,
        fine_tuned_tokenizer=fine_tuned_tokenizer,
        inputs=sample_inputs
    )
    
    # Step 12: Print comparison and save results
    print("\n12. Printing comparison and saving results...")
    print_comparison(sample_inputs, base_outputs, fine_tuned_outputs)
    save_predictions_to_csv(sample_inputs, base_outputs, fine_tuned_outputs)
    
    print("\n" + "="*60)
    print("Fine-tuning pipeline completed successfully!")
    print("="*60)
    print(f"Results saved to: {Config.PREDICTIONS_CSV_PATH}")
    print(f"Model saved to: {Config.MODEL_SAVE_PATH}")
    print(f"Checkpoints saved to: {Config.TRAINING_ARGS['output_dir']}")


def run_evaluation_only():
    """
    Run only the evaluation part (useful when model is already trained).
    """
    print("="*60)
    print("Model Evaluation Only")
    print("="*60)
    
    # Load models
    print("\n1. Loading models...")
    base_model, base_tokenizer = setup_model()
    fine_tuned_model, fine_tuned_tokenizer = load_fine_tuned_model()
    
    # Get sample inputs
    print("\n2. Preparing sample inputs...")
    sample_inputs = get_sample_inputs()
    
    # Compare models
    print("\n3. Comparing models...")
    base_outputs, fine_tuned_outputs = compare_models(
        base_model=base_model,
        base_tokenizer=base_tokenizer,
        fine_tuned_model=fine_tuned_model,
        fine_tuned_tokenizer=fine_tuned_tokenizer,
        inputs=sample_inputs
    )
    
    # Print results
    print("\n4. Printing results...")
    print_comparison(sample_inputs, base_outputs, fine_tuned_outputs)
    save_predictions_to_csv(sample_inputs, base_outputs, fine_tuned_outputs)
    
    print("\nEvaluation completed!")


def resume_training():
    """
    Resume training from the latest checkpoint.
    """
    print("="*60)
    print("Resume Training from Checkpoint")
    print("="*60)
    
    from trainer import find_latest_checkpoint, get_checkpoint_info
    
    # Check for checkpoints
    checkpoint_info = get_checkpoint_info()
    if checkpoint_info["total_checkpoints"] == 0:
        print("No checkpoints found. Please run the full training first.")
        return
    
    print(f"Found {checkpoint_info['total_checkpoints']} checkpoints:")
    for checkpoint in checkpoint_info["checkpoints"]:
        print(f"  - {checkpoint['name']} (step {checkpoint['step']})")
    
    # Load data and setup
    print("\n1. Loading data and setting up models...")
    dataset = load_data()
    tokenizer = setup_tokenizer()
    tokenized_dataset = preprocess_data(dataset, tokenizer)
    
    model, tokenizer = setup_model()
    model = setup_lora(model, tokenizer)
    
    trainer = setup_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"]
    )
    
    # Resume training
    print("\n2. Resuming training...")
    latest_checkpoint = find_latest_checkpoint()
    training_results = train_model(trainer, resume_from_checkpoint=latest_checkpoint)
    
    print("\nTraining resumed and completed successfully!")


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--evaluate":
            run_evaluation_only()
        elif sys.argv[1] == "--resume":
            resume_training()
        else:
            print("Usage:")
            print("  python main.py              # Run full training pipeline")
            print("  python main.py --evaluate   # Run evaluation only")
            print("  python main.py --resume     # Resume training from checkpoint")
    else:
        main() 