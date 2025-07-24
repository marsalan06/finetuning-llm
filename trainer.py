"""
Trainer module for handling the fine-tuning process.
Contains functions for setting up the trainer and executing training with checkpoint support.
"""

import os
import glob
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from config import Config


def setup_trainer(model, tokenizer, train_dataset, eval_dataset):
    """
    Sets up the trainer with specified configuration optimized for MacBook GPU.
    
    Args:
        model: The model to train
        tokenizer: The tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        
    Returns:
        Trainer: Configured trainer instance
    """
    print("Setting up trainer with MacBook GPU optimizations...")
    
    # Create directories
    Config.create_directories()
    
    # Apply memory optimizations
    if Config.MEMORY_OPTIMIZATION["use_gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled")
    
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
    print(f"  - Gradient accumulation steps: {Config.TRAINING_ARGS['gradient_accumulation_steps']}")
    print(f"  - Effective batch size: {Config.TRAINING_ARGS['per_device_train_batch_size'] * Config.TRAINING_ARGS['gradient_accumulation_steps']}")
    print(f"  - Epochs: {Config.TRAINING_ARGS['num_train_epochs']}")
    print(f"  - Mixed precision (FP16): {Config.TRAINING_ARGS['fp16']}")
    print(f"  - Output directory: {Config.TRAINING_ARGS['output_dir']}")
    print(f"  - Checkpoint directory: {Config.CHECKPOINT_DIR}")
    
    return trainer


def find_latest_checkpoint():
    """
    Finds the latest checkpoint in the output directory.
    
    Returns:
        str or None: Path to the latest checkpoint, or None if no checkpoint found
    """
    output_dir = Config.TRAINING_ARGS["output_dir"]
    
    # Look for checkpoint directories
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    
    if not checkpoint_dirs:
        return None
    
    # Sort by checkpoint number and return the latest
    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
    latest_checkpoint = checkpoint_dirs[-1]
    
    print(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint


def train_model(trainer, resume_from_checkpoint=None):
    """
    Trains the model using the provided trainer with checkpoint support.
    
    Args:
        trainer: The configured trainer instance
        resume_from_checkpoint (str, optional): Path to checkpoint to resume from
        
    Returns:
        dict: Training results
    """
    print("Starting model training...")
    
    # Check for existing checkpoints if no specific checkpoint provided
    if resume_from_checkpoint is None and Config.TRAINING_ARGS["resume_from_checkpoint"]:
        resume_from_checkpoint = find_latest_checkpoint()
    
    if resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
    else:
        print("Starting training from scratch")
    
    # Train the model
    training_results = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
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
                                  If None, loads from the latest checkpoint.
    
    Returns:
        tuple: (model, tokenizer) - The loaded model and tokenizer
    """
    if model_path is None:
        # Find the latest checkpoint
        latest_checkpoint = find_latest_checkpoint()
        if latest_checkpoint is None:
            raise FileNotFoundError("No fine-tuned model checkpoints found. Please train the model first.")
        model_path = latest_checkpoint
        
    print(f"Loading fine-tuned model from: {model_path}")
    
    # Check if the model directory exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Fine-tuned model not found at: {model_path}")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load the base model first
    base_model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_CHECKPOINT,
        trust_remote_code=True
    )
    base_model.to(Config.DEVICE)
    
    # Load the LoRA adapters from the checkpoint
    from peft import PeftModel
    fine_tuned_model = PeftModel.from_pretrained(base_model, model_path)
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        Config.MODEL_CHECKPOINT,
        trust_remote_code=True
    )
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Fine-tuned model loaded successfully.")
    
    return fine_tuned_model, tokenizer


def get_checkpoint_info():
    """
    Gets information about available checkpoints.
    
    Returns:
        dict: Information about checkpoints
    """
    output_dir = Config.TRAINING_ARGS["output_dir"]
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    
    checkpoint_info = {
        "total_checkpoints": len(checkpoint_dirs),
        "checkpoints": []
    }
    
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_name = os.path.basename(checkpoint_dir)
        checkpoint_info["checkpoints"].append({
            "name": checkpoint_name,
            "path": checkpoint_dir,
            "step": int(checkpoint_name.split("-")[-1])
        })
    
    return checkpoint_info


def cleanup_old_checkpoints(keep_latest=3):
    """
    Cleans up old checkpoints, keeping only the latest ones.
    
    Args:
        keep_latest (int): Number of latest checkpoints to keep
    """
    output_dir = Config.TRAINING_ARGS["output_dir"]
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    
    if len(checkpoint_dirs) <= keep_latest:
        return
    
    # Sort by checkpoint number
    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
    
    # Remove old checkpoints
    checkpoints_to_remove = checkpoint_dirs[:-keep_latest]
    
    for checkpoint_dir in checkpoints_to_remove:
        import shutil
        shutil.rmtree(checkpoint_dir)
        print(f"Removed old checkpoint: {checkpoint_dir}")


def resume_training_from_checkpoint(checkpoint_path=None):
    """
    Resumes training from a specific checkpoint.
    
    Args:
        checkpoint_path (str, optional): Path to the checkpoint. 
                                      If None, uses the latest checkpoint.
    
    Returns:
        bool: True if training was resumed successfully, False otherwise
    """
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()
    
    if checkpoint_path is None:
        print("No checkpoint found to resume from.")
        return False
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return False
    
    print(f"Resuming training from checkpoint: {checkpoint_path}")
    return True 