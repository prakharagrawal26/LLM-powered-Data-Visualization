from functools import partial
import torch
import os
import gc
import logging
import pandas as pd
from typing import Dict, Any, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TrainerCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset

# Import helpers from model_handler
from .model_handler import (
    load_base_model_and_tokenizer,
    get_bitsandbytes_config,
    get_lora_config,
    get_training_args,
    format_for_sft # Import the formatting function
)

logger = logging.getLogger(__name__)

# Optional: Callback to free memory during training
class FreeMemoryCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        logger.info("Epoch end: Clearing CUDA cache.")
        torch.cuda.empty_cache()
        gc.collect()

def train_model(config: Dict[str, Any], training_df: pd.DataFrame) -> bool:
    """
    Sets up and runs the SFTTrainer fine-tuning process.

    Returns:
        True if training was successful and adapter saved, False otherwise.
    """
    logger.info("--- Starting Model Fine-tuning Process ---")

    # 1. Load Configs
    bnb_config = get_bitsandbytes_config(config)
    lora_config = get_lora_config(config)
    training_args = get_training_args(config)
    adapter_output_path = config['paths']['adapter_output_dir']

    if not all([bnb_config, lora_config, training_args]):
        logger.error("Failed to load necessary configurations (BnB, LoRA, TrainingArgs). Cannot train.")
        return False

    # Ensure adapter output directory exists
    os.makedirs(adapter_output_path, exist_ok=True)

    # Initialize resources
    model = None
    tokenizer = None
    trainer = None

    try:
        # 2. Load Base Model and Tokenizer
        model, tokenizer = load_base_model_and_tokenizer(config['base_model_name'], bnb_config)
        if model is None or tokenizer is None:
            return False # Error already logged in load function

        # 3. Prepare Model for QLoRA
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        logger.info("Model prepared for K-bit training.")

        # 4. Apply LoRA
        model = get_peft_model(model, lora_config)
        logger.info("Applied LoRA configuration.")
        model.print_trainable_parameters()

        # 5. Prepare Dataset
        if not isinstance(training_df, pd.DataFrame) or training_df.empty:
             logger.error("Invalid or empty training dataframe provided.")
             return False
        if not all(col in training_df.columns for col in ['instruction', 'output']):
             logger.error("Training DataFrame must contain 'instruction' and 'output' columns.")
             return False
        training_df['instruction'] = training_df['instruction'].astype(str)
        training_df['output'] = training_df['output'].astype(str)
        dataset = Dataset.from_pandas(training_df[['instruction', 'output']]).shuffle(seed=42)
        logger.info(f"Prepared training dataset with {len(dataset)} samples.")

        # 6. Initialize SFTTrainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=lora_config,
            formatting_func=partial(format_for_sft, tokenizer=tokenizer), # Use partial correctly
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=tokenizer.model_max_length or 1024, # Use tokenizer max length or default
            packing=False,
            callbacks=[FreeMemoryCallback()] # Add callback
        )
        logger.info("SFTTrainer initialized.")

        # 7. Train
        logger.info("Starting training...")
        gc.collect()
        torch.cuda.empty_cache()
        train_result = trainer.train()
        logger.info("--- Training Finished ---")

        # 8. Save Model (Adapter) and Tokenizer
        logger.info(f"Saving adapter model and tokenizer to {adapter_output_path}...")
        # Use save_pretrained for PEFT model to save adapter correctly
        model.save_pretrained(adapter_output_path)
        tokenizer.save_pretrained(adapter_output_path)

        # Save training metrics if needed
        metrics = train_result.metrics
        logger.info(f"Training Metrics: {metrics}")
        try:
             trainer.log_metrics("train", metrics)
             trainer.save_metrics("train", metrics)
             trainer.save_state() # Saves trainer state like optimizer, scheduler etc.
        except Exception as e:
             logger.warning(f"Could not save metrics/state: {e}")


        logger.info("Adapter and tokenizer saved successfully.")
        return True

    except Exception as e:
        logger.error(f"Error during training process: {e}", exc_info=True)
        return False
    finally:
        # Cleanup
        logger.info("Cleaning up training resources...")
        del model
        del tokenizer
        del trainer
        if 'dataset' in locals(): del dataset
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Training cleanup complete.")