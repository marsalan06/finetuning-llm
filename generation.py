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
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize input with proper attention mask
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=Config.MAX_LENGTH,
        pad_to_multiple_of=8,  # Ensure proper padding
        return_attention_mask=True  # Explicitly request attention mask
    )
    
    # Move inputs to device
    inputs = {key: value.to(Config.DEVICE) for key, value in inputs.items()}
    
    # Generate text with proper parameters
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],  # Pass attention mask
            max_new_tokens=max_length - inputs['input_ids'].shape[1],  # Use max_new_tokens instead of max_length
            num_return_sequences=Config.GENERATION_CONFIG['num_return_sequences'],
            do_sample=Config.GENERATION_CONFIG['do_sample'],
            temperature=Config.GENERATION_CONFIG['temperature'],
            top_p=Config.GENERATION_CONFIG['top_p'],
            top_k=Config.GENERATION_CONFIG['top_k'],
            repetition_penalty=Config.GENERATION_CONFIG['repetition_penalty'],
            no_repeat_ngram_size=Config.GENERATION_CONFIG['no_repeat_ngram_size'],
            pad_token_id=tokenizer.pad_token_id,  # Explicitly set pad token
            eos_token_id=tokenizer.eos_token_id  # Explicitly set eos token
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


def evaluate_on_dataset(model, tokenizer, dataset, num_samples=5):
    """
    Evaluate model predictions against actual dataset values.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer to use
        dataset: The dataset to evaluate on
        num_samples (int): Number of samples to evaluate
        
    Returns:
        list: List of evaluation results
    """
    print(f"\n🔍 Evaluating model on {num_samples} dataset samples...")
    print("=" * 80)
    
    evaluation_results = []
    
    # Get random samples from the dataset
    import random
    random.seed(42)  # For reproducible results
    
    # Convert dataset to list if it's a Dataset object
    if hasattr(dataset, 'select'):
        # For HuggingFace datasets
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        samples = [dataset[i] for i in indices]
    else:
        # For other dataset types
        samples = random.sample(list(dataset), min(num_samples, len(dataset)))
    
    for i, sample in enumerate(samples, 1):
        print(f"\n📊 Sample {i}/{num_samples}")
        print("-" * 50)
        
        # Extract prompt and expected output
        if isinstance(sample, dict):
            prompt = sample.get('prompt', '')
            expected_output = sample.get('output', '')
        else:
            # Handle different dataset formats
            prompt = str(sample[0]) if len(sample) > 0 else ''
            expected_output = str(sample[1]) if len(sample) > 1 else ''
        
        print(f"📝 Prompt: {prompt}")
        print(f"✅ Expected: {expected_output[:100]}{'...' if len(expected_output) > 100 else ''}")
        
        # Generate prediction
        try:
            predicted_output = generate_text(model, tokenizer, prompt)
            print(f"🤖 Predicted: {predicted_output[:100]}{'...' if len(predicted_output) > 100 else ''}")
            
            # Calculate similarity metrics
            similarity_score = calculate_similarity(expected_output, predicted_output)
            print(f"📈 Similarity Score: {similarity_score:.2f}")
            
            # Store result
            result = {
                'sample_id': i,
                'prompt': prompt,
                'expected': expected_output,
                'predicted': predicted_output,
                'similarity': similarity_score
            }
            evaluation_results.append(result)
            
        except Exception as e:
            print(f"❌ Error generating prediction: {e}")
            result = {
                'sample_id': i,
                'prompt': prompt,
                'expected': expected_output,
                'predicted': f"ERROR: {e}",
                'similarity': 0.0
            }
            evaluation_results.append(result)
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("📊 EVALUATION SUMMARY")
    print("=" * 80)
    
    valid_results = [r for r in evaluation_results if not r['predicted'].startswith("ERROR:")]
    if valid_results:
        avg_similarity = sum(r['similarity'] for r in valid_results) / len(valid_results)
        good_predictions = sum(1 for r in valid_results if r['similarity'] >= 0.5)
        
        print(f"Average Similarity Score: {avg_similarity:.2f}")
        print(f"Successful Predictions: {len(valid_results)}/{len(evaluation_results)}")
        print(f"Good Predictions (≥0.5): {good_predictions}/{len(valid_results)} ({good_predictions/len(valid_results)*100:.1f}%)")
        
        # Show best and worst predictions
        best_result = max(valid_results, key=lambda x: x['similarity'])
        worst_result = min(valid_results, key=lambda x: x['similarity'])
        
        print(f"\n🏆 Best Prediction (Score: {best_result['similarity']:.2f}):")
        print(f"Prompt: {best_result['prompt'][:50]}...")
        print(f"Expected: {best_result['expected'][:50]}...")
        print(f"Predicted: {best_result['predicted'][:50]}...")
        
        print(f"\n⚠️  Worst Prediction (Score: {worst_result['similarity']:.2f}):")
        print(f"Prompt: {worst_result['prompt'][:50]}...")
        print(f"Expected: {worst_result['expected'][:50]}...")
        print(f"Predicted: {worst_result['predicted'][:50]}...")
    
    return evaluation_results


def calculate_similarity(expected, predicted):
    """
    Calculate similarity between expected and predicted outputs.
    
    Args:
        expected (str): Expected output
        predicted (str): Predicted output
        
    Returns:
        float: Similarity score between 0 and 1
    """
    import difflib
    
    # Clean the texts
    expected_clean = expected.strip().lower()
    predicted_clean = predicted.strip().lower()
    
    # Use difflib for sequence matching
    similarity = difflib.SequenceMatcher(None, expected_clean, predicted_clean).ratio()
    
    return similarity


def compare_models_with_dataset_evaluation(base_model, base_tokenizer, fine_tuned_model, fine_tuned_tokenizer, dataset, inputs):
    """
    Compare outputs from base model and fine-tuned model with dataset evaluation.
    
    Args:
        base_model: The base model
        base_tokenizer: The base model's tokenizer
        fine_tuned_model: The fine-tuned model
        fine_tuned_tokenizer: The fine-tuned model's tokenizer
        dataset: The dataset to evaluate on
        inputs (list): List of input texts to test
        
    Returns:
        tuple: (base_outputs, fine_tuned_outputs, evaluation_results)
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
    
    # Evaluate fine-tuned model on dataset
    evaluation_results = evaluate_on_dataset(fine_tuned_model, fine_tuned_tokenizer, dataset)
    
    return base_model_outputs, fine_tuned_outputs, evaluation_results 