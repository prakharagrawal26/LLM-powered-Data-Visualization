import torch
import os
import gc
import re
import json
import logging
from typing import Optional, Dict, Any, Tuple
from functools import partial

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from trl import SFTTrainer
from datasets import Dataset

from .tools import tool_registry, PlotRequest, PlotType

logger = logging.getLogger(__name__)

# --- Model/Tokenizer Loading ---

def load_base_model_and_tokenizer(model_name: str, bnb_config: BitsAndBytesConfig) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
    """Loads the base model and tokenizer."""
    try:
        logger.info(f"Loading base model: {model_name} with BitsAndBytes config...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set tokenizer pad_token to eos_token")
        tokenizer.padding_side = 'left' # Important for generation

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        # Ensure model pad token id is set (important for training and generation)
        if model.config.pad_token_id is None:
             model.config.pad_token_id = tokenizer.pad_token_id

        logger.info("Base model and tokenizer loaded.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load base model/tokenizer {model_name}: {e}", exc_info=True)
        return None, None

def load_finetuned_model_for_inference(base_model_name: str, adapter_path: str, bnb_config: BitsAndBytesConfig) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
    """Loads the base model, applies the adapter, merges it, and loads the tokenizer."""
    if not os.path.exists(adapter_path):
        logger.error(f"Adapter path not found: {adapter_path}")
        return None, None

    base_model = None # To ensure it's cleaned up
    try:
        logger.info(f"Loading base model {base_model_name} for adapter merging...")
        # Load base model with quantization
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        logger.info("Base model loaded. Applying and merging adapter...")
        # Load the PEFT model and merge
        merged_model = PeftModel.from_pretrained(base_model, adapter_path)
        merged_model = merged_model.merge_and_unload()
        merged_model.eval() # Set to eval mode

        logger.info(f"Loading tokenizer from adapter path: {adapter_path}")
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set tokenizer pad_token to eos_token")
        tokenizer.padding_side = 'left'

        logger.info("Fine-tuned model merged and tokenizer loaded successfully.")
        return merged_model, tokenizer

    except Exception as e:
        logger.error(f"Failed to load/merge fine-tuned model from {adapter_path}: {e}", exc_info=True)
        return None, None
    finally:
        # Clean up the original base model explicitly after merge/unload or failure
        del base_model
        gc.collect()
        torch.cuda.empty_cache()


# --- Training Setup ---

def get_bitsandbytes_config(config: Dict[str, Any]) -> Optional[BitsAndBytesConfig]:
    """Creates BitsAndBytesConfig from config."""
    try:
        bnb_params = config['bnb']
        compute_dtype = bnb_params.get('bnb_4bit_compute_dtype', torch.bfloat16) # Default if needed
        
        # If compute_dtype is still a string, convert it (should be handled by config loader now)
        if isinstance(compute_dtype, str):
            logger.warning(f"Compute dtype '{compute_dtype}' is a string, attempting conversion.")
            compute_dtype = getattr(torch, compute_dtype.split('.')[-1])

        return BitsAndBytesConfig(
            load_in_4bit=bnb_params['load_in_4bit'],
            bnb_4bit_quant_type=bnb_params['bnb_4bit_quant_type'],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=bnb_params['bnb_4bit_use_double_quant']
        )
    except KeyError as e:
        logger.error(f"Missing required BitsAndBytes config key: {e}")
        return None
    except Exception as e:
        logger.error(f"Error creating BitsAndBytesConfig: {e}")
        return None

def get_lora_config(config: Dict[str, Any]) -> Optional[LoraConfig]:
    """Creates LoraConfig from config."""
    try:
        lora_params = config['lora']
        return LoraConfig(
            r=lora_params['r'],
            lora_alpha=lora_params['lora_alpha'],
            target_modules=lora_params['target_modules'],
            lora_dropout=lora_params['lora_dropout'],
            bias=lora_params['bias'],
            task_type=lora_params['task_type']
        )
    except KeyError as e:
        logger.error(f"Missing required LoRA config key: {e}")
        return None
    except Exception as e:
        logger.error(f"Error creating LoraConfig: {e}")
        return None

def get_training_args(config: Dict[str, Any]) -> Optional[TrainingArguments]:
    """Creates TrainingArguments from config."""
    try:
        trainer_params = config['trainer']
        output_dir = config['paths']['adapter_output_dir'] # Use the correct path
        logging_dir = os.path.join(output_dir, "training_logs")

        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=trainer_params['per_device_train_batch_size'],
            gradient_accumulation_steps=trainer_params['gradient_accumulation_steps'],
            learning_rate=float(trainer_params['learning_rate']),
            num_train_epochs=trainer_params['num_train_epochs'],
            logging_dir=logging_dir,
            logging_steps=trainer_params['logging_steps'],
            save_strategy=trainer_params['save_strategy'],
            save_total_limit=trainer_params.get('save_total_limit', 1), # Default if missing
            fp16=trainer_params['fp16'],
            max_grad_norm=trainer_params['max_grad_norm'],
            warmup_ratio=trainer_params['warmup_ratio'],
            lr_scheduler_type=trainer_params['lr_scheduler_type'],
            report_to="none",
            # dataloader_num_workers=1 # Keep low for simplicity
        )
    except KeyError as e:
        logger.error(f"Missing required Trainer config key: {e}")
        return None
    except Exception as e:
        logger.error(f"Error creating TrainingArguments: {e}")
        return None

# --- Formatting Function for SFTTrainer ---

def get_system_prompt() -> str:
    """Builds the system prompt using the tool registry."""
    tool_descs_json = tool_registry.get_tool_descriptions_json()
    tool_names = tool_registry.get_tool_names()
    return f"""You are an expert climate data visualization assistant. Analyze the user request and respond ONLY with a single valid JSON object specifying the plotting tool and parameters.

Available Tools:
{tool_descs_json}

Respond ONLY with a single valid JSON object with keys "plot_type" (one of {tool_names}) and "parameters" (dict with values for city, country, start_year, end_year, title).

User Request:"""

# Store system prompt globally after first generation
_SYSTEM_PROMPT = None

def format_for_sft(sample: Dict[str, str], tokenizer) -> list[str]:
    """Formats a data sample for SFTTrainer."""
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        _SYSTEM_PROMPT = get_system_prompt()

    instruction = sample['instruction']
    output_json = sample['output']
    # Format: System Prompt + Instruction + Separator + JSON Output + EOS Token
    text = f"{_SYSTEM_PROMPT}\n{instruction}\n\nJSON Output:\n{output_json}{tokenizer.eos_token}"
    return [text]


# --- Inference Logic ---

def generate_json_response(model, tokenizer, query: str, config: dict) -> Optional[str]:
    """Generates the JSON response string from the fine-tuned model."""
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        _SYSTEM_PROMPT = get_system_prompt() # Ensure prompt is generated

    full_prompt = f"{_SYSTEM_PROMPT}\n{query}\n\nJSON Output:\n"
    logger.debug(f"Sending prompt to model:\n{full_prompt[:300]}...")

    try:
        inputs = tokenizer(full_prompt, return_tensors="pt", return_attention_mask=True).to(model.device)

        inference_params = config.get('inference', {})
        max_new_tokens = inference_params.get('max_new_tokens', 200)
        do_sample = inference_params.get('do_sample', False)

        logger.info("Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        # Decode only generated tokens
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        logger.info(f"Model Raw Output:\n>>>\n{generated_text}\n<<<")
        return generated_text.strip()

    except Exception as e:
        logger.error(f"Error during model generation: {e}", exc_info=True)
        return None

def extract_json_from_text(text: str) -> Optional[str]:
    """Extracts JSON object from text, prioritizing markdown blocks."""
    if not text: return None
    logger.debug(f"Attempting to extract JSON from: {text}")
    # Try markdown block first
    match_markdown = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if match_markdown:
        json_str = match_markdown.group(1).strip()
        logger.info("Extracted JSON using markdown markers.")
        return json_str

    # Try finding first complete {} block
    start_brace = text.find('{')
    if start_brace != -1:
        level = 0
        for i, char in enumerate(text[start_brace:]):
            if char == '{': level += 1
            elif char == '}': level -= 1
            if level == 0:
                json_str = text[start_brace : start_brace + i + 1].strip()
                logger.info("Extracted JSON using brace matching.")
                return json_str
        logger.warning("Found '{' but no matching '}'")

    # Fallback: If text starts with { and ends with }, assume it's JSON
    if text.startswith('{') and text.endswith('}'):
        logger.warning("Using fallback: Assuming raw text is JSON.")
        return text

    logger.error("Could not extract JSON object from text.")
    return None

def parse_json_to_plot_request(json_str: str) -> Optional[PlotRequest]:
    """Parses JSON string into a validated PlotRequest object."""
    if not json_str: return None
    try:
        data = json.loads(json_str)
        if not isinstance(data, dict) or "plot_type" not in data or "parameters" not in data:
            raise ValueError("JSON missing required 'plot_type' or 'parameters' keys.")
        params = data["parameters"]
        if not isinstance(params, dict):
             raise ValueError("'parameters' must be a dictionary.")
        required_params = ["city", "country", "start_year", "end_year"]
        if not all(k in params for k in required_params):
            missing = [k for k in required_params if k not in params]
            raise ValueError(f"Missing required parameters: {missing}")

        plot_type_enum = PlotType(data["plot_type"]) # Raises ValueError if invalid type

        request = PlotRequest(
            plot_type=plot_type_enum,
            city=str(params["city"]),
            country=str(params["country"]),
            start_year=int(params["start_year"]),
            end_year=int(params["end_year"]),
            title=params.get("title") # Optional
        )
        request.validate() # Internal validation (e.g., year range)
        logger.info(f"Successfully parsed JSON into PlotRequest: {request}")
        return request
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.error(f"Failed parsing/validating JSON to PlotRequest: {e}. JSON was: {json_str}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing JSON: {e}", exc_info=True)
        return None