"""
Main script for LoRA fine-tuning of language models.
This script orchestrates the entire fine-tuning process from data loading to model evaluation.
"""

import os
import sys
from config import Config

# Import our custom modules
from data_loader import load_data, preprocess_data, setup_tokenizer
from model_setup import setup_model, setup_lora, print_model_info
from trainer import setup_trainer, train_model, save_model, load_fine_tuned_model
from generation import compare_models, print_comparison, save_predictions_to_csv, get_sample_inputs


def main():
    """
    Main function that orchestrates the entire fine-tuning process.
    """
    print("="*60)
    print("LoRA Fine-Tuning Pipeline")
    print("="*60)
    
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
    
    # Step 5: Train the model
    print("\n5. Training the model...")
    training_results = train_model(trainer)
    
    # Step 6: Save the fine-tuned model
    print("\n6. Saving the fine-tuned model...")
    save_model(model, tokenizer)
    
    # Step 7: Load the fine-tuned model for evaluation
    print("\n7. Loading fine-tuned model for evaluation...")
    fine_tuned_model, fine_tuned_tokenizer = load_fine_tuned_model()
    
    # Step 8: Generate sample inputs
    print("\n8. Preparing sample inputs for evaluation...")
    sample_inputs = get_sample_inputs()
    
    # Step 9: Compare base and fine-tuned models
    print("\n9. Comparing base and fine-tuned models...")
    base_model, base_tokenizer = setup_model()  # Load fresh base model
    
    base_outputs, fine_tuned_outputs = compare_models(
        base_model=base_model,
        base_tokenizer=base_tokenizer,
        fine_tuned_model=fine_tuned_model,
        fine_tuned_tokenizer=fine_tuned_tokenizer,
        inputs=sample_inputs
    )
    
    # Step 10: Print comparison and save results
    print("\n10. Printing comparison and saving results...")
    print_comparison(sample_inputs, base_outputs, fine_tuned_outputs)
    save_predictions_to_csv(sample_inputs, base_outputs, fine_tuned_outputs)
    
    print("\n" + "="*60)
    print("Fine-tuning pipeline completed successfully!")
    print("="*60)
    print(f"Results saved to: {Config.PREDICTIONS_CSV_PATH}")
    print(f"Model saved to: {Config.MODEL_SAVE_PATH}")


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


if __name__ == "__main__":
    # Check if evaluation-only mode is requested
    if len(sys.argv) > 1 and sys.argv[1] == "--evaluate":
        run_evaluation_only()
    else:
        main() 