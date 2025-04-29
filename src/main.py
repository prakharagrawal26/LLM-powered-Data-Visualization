import os
import gc
import torch
import logging
import sys
import matplotlib.pyplot as plt

# Import project modules using relative paths
from .config import CONFIG, DATA_DIR, LOG_FILE, ADAPTER_DIR # Import config and derived paths
from .data_handler import (
    prepare_filtered_data,
    generate_training_data,
    load_processed_data_for_inference,
    get_yearly_avg_temps
)
from .model_handler import (
    load_finetuned_model_for_inference,
    generate_json_response,
    extract_json_from_text,
    parse_json_to_plot_request,
    get_bitsandbytes_config # Need this for loading inference model
)
from .tools import tool_registry # Import the registry instance
from . import tools # Import the module to ensure tools are registered
from .training import train_model

# --- Basic Logging Setup ---
def setup_logging():
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # Ensure data dir exists for log file
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    # Basic config with file and console handlers
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout) # Log to console as well
        ]
    )
    # Suppress overly verbose messages from libraries like transformers, datasets
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    logging.info("Logging setup complete.")
    logging.info(f"Log file: {LOG_FILE}")

# --- Main Execution Logic ---
def run():
    # Setup logging first
    setup_logging()
    logger = logging.getLogger(__name__) # Get logger after setup

    logger.info("="*40)
    logger.info("Starting LLM Fine-tuning Pipeline (Simplified)")
    logger.info("="*40)

    if not CONFIG: # Check if config loaded successfully in config.py
         logger.critical("Configuration could not be loaded. Cannot proceed.")
         return

    # --- Step 1: Prepare Data ---
    logger.info("--- Step 1: Preparing Filtered Data ---")
    filtered_data_path = prepare_filtered_data(CONFIG)
    if not filtered_data_path:
        logger.critical("Failed to prepare filtered data. Exiting.")
        return
    logger.info(f"Filtered data ready at: {filtered_data_path}")

    # --- Step 2: Generate Training Data ---
    logger.info("\n--- Step 2: Generating Training Data ---")
    training_df = generate_training_data(
        num_examples=CONFIG['num_training_examples'],
        countries_list=CONFIG['filter_countries'],
        year_start=CONFIG['filter_year_start'],
        year_end=CONFIG['filter_year_end']
    )
    if training_df is None or training_df.empty:
        logger.critical("Failed to generate training data. Exiting.")
        return
    logger.info(f"Generated {len(training_df)} training examples.")

    # --- Step 3: Fine-tune Model ---
    logger.info("\n--- Step 3: Fine-tuning Model ---")
    adapter_path = CONFIG['paths']['adapter_output_dir']
    # Check if essential adapter files exist
    adapter_model_file = os.path.join(adapter_path, 'adapter_model.safetensors')
    tokenizer_config_file = os.path.join(adapter_path, 'tokenizer_config.json')

    if not os.path.exists(adapter_model_file) or not os.path.exists(tokenizer_config_file):
        logger.info(f"Adapter not found at {adapter_path}. Starting fine-tuning...")
        training_success = train_model(CONFIG, training_df)
        if not training_success:
            logger.critical("Fine-tuning failed. Exiting.")
            del training_df; gc.collect()
            return
        logger.info(f"Fine-tuning successful. Adapter saved to: {adapter_path}")
    else:
        logger.info(f"Found existing adapter at {adapter_path}. Skipping fine-tuning.")

    # Clean up training data now
    del training_df
    gc.collect()
    torch.cuda.empty_cache()

    # --- Step 4: Prepare for Inference ---
    logger.info("\n--- Step 4: Preparing for Inference ---")
    # Load the processed data into memory
    if not load_processed_data_for_inference(filtered_data_path):
        logger.critical("Failed to load processed data for inference. Exiting.")
        return

    # Load the fine-tuned model
    logger.info("Loading fine-tuned model for inference...")
    bnb_config_inf = get_bitsandbytes_config(CONFIG) # Get config for loading
    if not bnb_config_inf:
        logger.critical("Failed to get BitsAndBytes config for inference loading. Exiting.")
        return

    model, tokenizer = load_finetuned_model_for_inference(
        CONFIG['base_model_name'],
        adapter_path,
        bnb_config_inf
    )
    if model is None or tokenizer is None:
        logger.critical("Failed to load fine-tuned model for inference. Exiting.")
        return
    logger.info("Inference model loaded.")

    # --- Step 5: Run Inference Tests ---
    logger.info("\n" + "="*20 + " Step 5: Running Inference Tests " + "="*20)
    test_queries = [
        "Create a line plot for Paris, France from 1960 to 2000.",
        "Generate a scatter plot with trend for Tokyo, Japan, 1970-2010.",
        "Show Berlin, Germany temps 1980-2005 as a line graph.",
        "Scatter plot for Sydney, Australia from 1955 to 1995.",
        # Add more or failing cases if desired
        "Plot temperature for Nowhere Land 2000-2005" # Should fail data lookup
    ]

    for i, query in enumerate(test_queries):
        logger.info(f"\n--- Processing Query {i+1}: '{query}' ---")
        fig = None
        plot_request = None # Reset for each query

        # 1. Get JSON from LLM
        raw_response = generate_json_response(model, tokenizer, query, CONFIG)
        if not raw_response:
            logger.error("❌ Failed: Model did not generate a response.")
            continue

        # 2. Extract JSON
        json_str = extract_json_from_text(raw_response)
        if not json_str:
            logger.error(f"❌ Failed: Could not extract JSON from response: {raw_response}")
            continue

        # 3. Parse JSON to PlotRequest
        plot_request = parse_json_to_plot_request(json_str)
        if not plot_request:
            logger.error("❌ Failed: Could not parse JSON into a valid PlotRequest.")
            continue

        # 4. Execute Request (Get data -> Call tool)
        logger.info(f"Executing request: {plot_request}")
        try:
            # Get data using the handler function
            yearly_data = get_yearly_avg_temps(
                plot_request.city, plot_request.country,
                plot_request.start_year, plot_request.end_year
            )

            if yearly_data is None or yearly_data.empty:
                logger.error(f"❌ Failed: No data found for the request parameters.")
                continue # Skip to next query

            # Get the plotting tool function
            plot_function = tool_registry.get_tool(plot_request.plot_type.value)
            if plot_function is None:
                logger.error(f"❌ Failed: Plotting tool '{plot_request.plot_type.value}' not found in registry.")
                continue

            # Prepare title and call the tool
            plot_title = plot_request.title or f"Avg Temp: {plot_request.city} ({plot_request.start_year}-{plot_request.end_year})"
            fig = plot_function(yearly_data, plot_title)

            if fig:
                logger.info(f"✅ Success: Plot generated for query {i+1}.")
                # Save the plot
                plot_filename = f"plot_query_{i+1}_{plot_request.city.replace(' ','_')}_{plot_request.plot_type.name}.png"
                plot_save_path = os.path.join(DATA_DIR, plot_filename) # Save in ./data/
                try:
                    fig.savefig(plot_save_path)
                    logger.info(f"Saved plot to {plot_save_path}")
                except Exception as save_e:
                    logger.error(f"Failed to save plot {plot_save_path}: {save_e}")
                plt.close(fig) # Close figure after saving/showing
            else:
                logger.error(f"❌ Failed: Plotting function did not return a figure for query {i+1}.")

        except Exception as exec_e:
            logger.error(f"❌ Failed: Unexpected error during request execution: {exec_e}", exc_info=True)
            if fig: plt.close(fig) # Ensure closure on error


    # --- Cleanup ---
    logger.info("\n--- Pipeline Finished ---")
    logger.info("Unloading inference model...")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Resources cleaned up.")
    logger.info("="*40 + " END OF PIPELINE " + "="*40 + "\n")


# --- Script Entry Point ---
if __name__ == "__main__":
    run() # Execute the main function