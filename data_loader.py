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
    Preprocesses the dataset by combining instruction and input as input prompt.
    
    Args:
        dataset: The dataset to preprocess
        tokenizer: The tokenizer to use for tokenization
        
    Returns:
        DatasetDict: Tokenized dataset ready for training
    """
    def preprocess_function(examples):
        """
        Preprocess function that combines instruction and input, then tokenizes.
        
        Args:
            examples: Batch of examples from the dataset
            
        Returns:
            dict: Tokenized inputs with labels
        """
        # Combine the instruction and input to form a prompt
        combined_input = [
            f"Instruction: {instruction}\nInput: {input_data}" 
            for instruction, input_data in zip(examples['instruction'], examples['input'])
        ]

        # Tokenize the combined input as prompt
        tokenized_inputs = tokenizer(
            combined_input, 
            padding="max_length", 
            truncation=True, 
            max_length=Config.MAX_LENGTH
        )

        # Tokenize the output (target text) - this will be the label
        tokenized_labels = tokenizer(
            examples['output'], 
            padding="max_length", 
            truncation=True, 
            max_length=Config.MAX_LENGTH
        )

        # Return the tokenized inputs and outputs
        tokenized_inputs['labels'] = tokenized_labels['input_ids']
        return tokenized_inputs

    # Apply the tokenization function to the dataset
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    print(f"Dataset after tokenization: {tokenized_dataset}")
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