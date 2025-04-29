import yaml
import os
import logging
import torch

logger = logging.getLogger(__name__)

# Hardcoded paths relative to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
LOG_FILE = os.path.join(DATA_DIR, "pipeline.log")
PROCESSED_DATA_FILENAME = "filtered_climate_data.csv"
ADAPTER_DIR = os.path.join(DATA_DIR, "phi2_finetuned_adapter") # Simplified name

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def load_config() -> dict:
    """Loads the YAML configuration file."""
    if not os.path.exists(CONFIG_PATH):
        logger.error(f"Configuration file not found at: {CONFIG_PATH}")
        raise FileNotFoundError(f"Configuration file not found at: {CONFIG_PATH}")
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded successfully from {CONFIG_PATH}")

        # Convert string representations of torch dtypes back to actual dtypes
        if 'bnb' in config and 'bnb_4bit_compute_dtype' in config['bnb']:
            dtype_str = config['bnb']['bnb_4bit_compute_dtype']
            try:
                config['bnb']['bnb_4bit_compute_dtype'] = getattr(torch, dtype_str.split('.')[-1])
            except AttributeError:
                logger.error(f"Invalid torch dtype specified in config: {dtype_str}")
                raise ValueError(f"Invalid torch dtype specified in config: {dtype_str}")

        # Add derived/fixed paths to the config dict for easy access
        config['paths'] = {
            'data_dir': DATA_DIR,
            'log_file': LOG_FILE,
            'processed_data_path': os.path.join(DATA_DIR, PROCESSED_DATA_FILENAME),
            'adapter_output_dir': ADAPTER_DIR, # For trainer output and final adapter
            'source_csv_path': os.path.join(DATA_DIR, config.get('source_csv_filename', 'source_data.csv'))
        }

        return config
    except (yaml.YAMLError, ValueError, FileNotFoundError) as e:
        logger.error(f"Error loading or processing config file {CONFIG_PATH}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading config: {e}")
        raise

# Load config once when module is imported
try:
    CONFIG = load_config()
except Exception as e:
     # Log critical failure if config cannot be loaded
     logging.basicConfig(level=logging.ERROR) # Basic config for this message
     logger = logging.getLogger(__name__)
     logger.critical(f"Failed to load configuration on startup: {e}. Exiting.")
     # Optionally exit here, or let subsequent code fail
     CONFIG = {} # Assign empty dict to avoid NameError later, though program likely won't run