"""
Generation module for text generation and evaluation.
Contains functions for generating text and comparing model outputs.
"""

import torch
import csv
from config import Config


def generate_text(model, tokenizer, input_text, max_length=None):
    """
    Generate text from the model based on input text.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer to use
        input_text (str): Input text to generate from
        max_length (int, optional): Maximum length of generated text. 
                                  Defaults to Config.GENERATION_CONFIG['max_length'].
    
    Returns:
        str: Generated text
    """
    if max_length is None:
        max_length = Config.GENERATION_CONFIG['max_length']
    
    # Tokenize input
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=Config.MAX_LENGTH,
        pad_to_multiple_of=8  # Ensure proper padding
    )
    
    # Move inputs to device
    inputs = {key: value.to(Config.DEVICE) for key, value in inputs.items()}
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'], 
            max_length=max_length, 
            num_return_sequences=Config.GENERATION_CONFIG['num_return_sequences'],
            do_sample=Config.GENERATION_CONFIG['do_sample']
        )
    
    # Decode generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def compare_models(base_model, base_tokenizer, fine_tuned_model, fine_tuned_tokenizer, inputs):
    """
    Compare outputs from base model and fine-tuned model.
    
    Args:
        base_model: The base model
        base_tokenizer: The base model's tokenizer
        fine_tuned_model: The fine-tuned model
        fine_tuned_tokenizer: The fine-tuned model's tokenizer
        inputs (list): List of input texts to test
        
    Returns:
        tuple: (base_outputs, fine_tuned_outputs) - Lists of outputs from both models
    """
    print("Generating outputs from base model...")
    base_model_outputs = [
        generate_text(base_model, base_tokenizer, input_text) 
        for input_text in inputs
    ]
    
    print("Generating outputs from fine-tuned model...")
    fine_tuned_outputs = [
        generate_text(fine_tuned_model, fine_tuned_tokenizer, input_text) 
        for input_text in inputs
    ]
    
    return base_model_outputs, fine_tuned_outputs


def print_comparison(inputs, base_outputs, fine_tuned_outputs):
    """
    Print comparison between base and fine-tuned model outputs.
    
    Args:
        inputs (list): List of input texts
        base_outputs (list): List of base model outputs
        fine_tuned_outputs (list): List of fine-tuned model outputs
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    for i, (input_text, base_output, fine_tuned_output) in enumerate(
        zip(inputs, base_outputs, fine_tuned_outputs), 1
    ):
        print(f"\nTest Case {i}:")
        print(f"Input: {input_text}")
        print(f"Base Model Output: {base_output}")
        print(f"Fine-Tuned Model Output: {fine_tuned_output}")
        print("-" * 50)


def save_predictions_to_csv(inputs, base_outputs, fine_tuned_outputs, file_path=None):
    """
    Save the input and model outputs into a CSV file.
    
    Args:
        inputs (list): List of input texts
        base_outputs (list): List of base model outputs
        fine_tuned_outputs (list): List of fine-tuned model outputs
        file_path (str, optional): Path to save the CSV file. 
                                 Defaults to Config.PREDICTIONS_CSV_PATH.
    """
    if file_path is None:
        file_path = Config.PREDICTIONS_CSV_PATH
    
    print(f"Saving predictions to: {file_path}")
    
    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Input", "Base Model Output", "Fine-Tuned Model Output"])
        
        for input_text, base_output, fine_tuned_output in zip(
            inputs, base_outputs, fine_tuned_outputs
        ):
            writer.writerow([input_text, base_output, fine_tuned_output])
    
    print(f"Predictions saved successfully to '{file_path}'")


def get_sample_inputs():
    """
    Get sample inputs for testing the models.
    
    Returns:
        list: List of sample input texts
    """
    return [
        "Generate a Python code for crawling a website for a specific type of data.",
        "Write a Python function to reverse a string",
        "Write a function to check if a number is prime",
        "Create a function to find the maximum element in a list",
        "Write a function to calculate the factorial of a number"
    ] 