"""
Data loader module for handling dataset loading and preprocessing.
Contains functions for loading datasets and preparing them for training.
"""

from datasets import load_dataset
from transformers import AutoTokenizer
from config import Config


def load_data():
    """
    Loads the dataset from Hugging Face and splits it into train/validation sets.
    
    Returns:
        DatasetDict: Dataset split into train and test sets
    """
    print(f"Loading dataset: {Config.DATASET_NAME}")
    dataset = load_dataset(Config.DATASET_NAME)
    print(f"Dataset loaded: {dataset}")

    # Split the dataset into train and validation sets
    dataset = dataset["train"].train_test_split(test_size=Config.TRAIN_TEST_SPLIT)
    print(f"Dataset after splitting: {dataset}")

    return dataset


def preprocess_data(dataset, tokenizer):
    """
    Preprocess dataset for GPT2-style decoder-only fine-tuning using HuggingFace Datasets.
    - Uses the dataset's existing 'prompt' field as input.
    - Concatenates 'prompt' + 'output' as the training input.
    - Labels are same as input_ids, but prompt tokens are masked with -100 (ignored during loss).
    - IMPROVED: Uses consistent Alpaca-style formatting for better generation quality.
    """
    def preprocess_function(examples):
        prompts = examples["prompt"]
        outputs = examples["output"]
        
        # Clean and format the data
        full_texts = []
        valid_prompts = []
        
        for prompt, output in zip(prompts, outputs):
            # Clean up the prompt and output
            prompt = prompt.strip()
            output = output.strip()
            
            # Skip empty outputs
            if not output.strip():
                continue
                
            # IMPROVED: Use consistent Alpaca-style formatting
            # This format works better for generation and is more consistent
            full_text = f"{prompt}\n\n### Response:\n{output}"
            
            full_texts.append(full_text)
            valid_prompts.append(prompt)

        # If no valid examples, return empty batch
        if not full_texts:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        # Tokenize full (prompt + output)
        tokenized = tokenizer(
            full_texts,
            padding="max_length",
            truncation=True,
            max_length=Config.MAX_LENGTH,
            return_tensors="pt"
        )

        # Tokenize prompt separately to find how much to mask
        # Include the separator in the prompt part for masking
        prompt_texts = [f"{p.strip()}\n\n### Response:\n" for p in valid_prompts]
        if prompt_texts:
            prompt_tokenized = tokenizer(
                prompt_texts,
                padding="max_length",
                truncation=True,
                max_length=Config.MAX_LENGTH,
                return_tensors="pt"
            )

            # Mask the prompt part in the labels with -100
            labels = tokenized["input_ids"].clone()
            for i, prompt_len in enumerate(prompt_tokenized["attention_mask"].sum(dim=1).tolist()):
                if i < len(labels):
                    labels[i][:prompt_len] = -100  # Mask prompt tokens

            tokenized["labels"] = labels
        else:
            # If no prompts, just use the input_ids as labels
            tokenized["labels"] = tokenized["input_ids"].clone()

        # Convert tensors to Python lists (HF Datasets requires this)
        return {k: v.tolist() for k, v in tokenized.items()}

    # Apply tokenization to the dataset with smaller batch size for stability
    try:
        tokenized_dataset = dataset.map(
            preprocess_function, 
            batched=True, 
            batch_size=32,  # Smaller batch size to avoid Arrow errors
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing dataset"
        )
    except Exception as e:
        print(f"⚠️  Error with batch_size=32, trying with batch_size=16: {e}")
        tokenized_dataset = dataset.map(
            preprocess_function, 
            batched=True, 
            batch_size=16,  # Even smaller batch size as fallback
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing dataset (fallback)"
        )
    
    print("✅ Tokenization complete.")
    return tokenized_dataset


def setup_tokenizer(model_checkpoint=None):
    """
    Sets up the tokenizer with proper padding token.
    
    Args:
        model_checkpoint (str, optional): Model checkpoint to use. 
                                       Defaults to Config.MODEL_CHECKPOINT.
    
    Returns:
        AutoTokenizer: Configured tokenizer
    """
    if model_checkpoint is None:
        model_checkpoint = Config.MODEL_CHECKPOINT
        
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    # Ensure a padding token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return tokenizer 