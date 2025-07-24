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
    """
    def preprocess_function(examples):
        prompts = examples["prompt"]
        outputs = examples["output"]
        full_texts = [prompt + output for prompt, output in zip(prompts, outputs)]

        # Tokenize full (prompt + output)
        tokenized = tokenizer(
            full_texts,
            padding="max_length",
            truncation=True,
            max_length=Config.MAX_LENGTH,
            return_tensors="pt"
        )

        # Tokenize prompt separately to find how much to mask
        prompt_tokenized = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=Config.MAX_LENGTH,
            return_tensors="pt"
        )

        # Mask the prompt part in the labels with -100
        labels = tokenized["input_ids"].clone()
        for i, prompt_len in enumerate(prompt_tokenized["attention_mask"].sum(dim=1).tolist()):
            labels[i][:prompt_len] = -100  # Mask prompt tokens

        tokenized["labels"] = labels

        # Convert tensors to Python lists (HF Datasets requires this)
        return {k: v.tolist() for k, v in tokenized.items()}

    # Apply tokenization to the dataset
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
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