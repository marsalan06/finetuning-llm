"""
Trainer module for handling the fine-tuning process.
Contains functions for setting up the trainer and executing training.
"""

from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from config import Config


def setup_trainer(model, tokenizer, train_dataset, eval_dataset):
    """
    Sets up the trainer with specified configuration.
    
    Args:
        model: The model to train
        tokenizer: The tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        
    Returns:
        Trainer: Configured trainer instance
    """
    print("Setting up trainer...")
    
    # Create training arguments
    training_args = TrainingArguments(**Config.TRAINING_ARGS)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    print("Trainer setup complete.")
    print(f"Training configuration:")
    print(f"  - Learning rate: {Config.TRAINING_ARGS['learning_rate']}")
    print(f"  - Batch size: {Config.TRAINING_ARGS['per_device_train_batch_size']}")
    print(f"  - Epochs: {Config.TRAINING_ARGS['num_train_epochs']}")
    print(f"  - Output directory: {Config.TRAINING_ARGS['output_dir']}")
    
    return trainer


def train_model(trainer):
    """
    Trains the model using the provided trainer.
    
    Args:
        trainer: The configured trainer instance
        
    Returns:
        dict: Training results
    """
    print("Starting model training...")
    
    # Train the model
    training_results = trainer.train()
    
    print("Training completed successfully!")
    print(f"Training loss: {training_results.training_loss:.4f}")
    
    return training_results


def save_model(model, tokenizer, save_path=None):
    """
    Saves the trained model and tokenizer.
    
    Args:
        model: The trained model to save
        tokenizer: The tokenizer to save
        save_path (str, optional): Path to save the model. 
                                 Defaults to Config.MODEL_SAVE_PATH.
    """
    if save_path is None:
        save_path = Config.MODEL_SAVE_PATH
        
    print(f"Saving model to: {save_path}")
    
    # Save the fine-tuned model
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"Model saved successfully to '{save_path}'")


def load_fine_tuned_model(model_path=None):
    """
    Loads a fine-tuned model from disk.
    
    Args:
        model_path (str, optional): Path to the saved model. 
                                  Defaults to Config.MODEL_SAVE_PATH.
    
    Returns:
        tuple: (model, tokenizer) - The loaded model and tokenizer
    """
    if model_path is None:
        model_path = Config.MODEL_SAVE_PATH
        
    print(f"Loading fine-tuned model from: {model_path}")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load the fine-tuned model
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(Config.DEVICE)
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("Fine-tuned model loaded successfully.")
    
    return model, tokenizer 