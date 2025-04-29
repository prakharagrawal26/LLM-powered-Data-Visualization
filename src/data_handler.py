import pandas as pd
import numpy as np
import os
import json
import logging
from typing import List, Optional, Dict

from .tools import PlotType # Import from tools module

logger = logging.getLogger(__name__)

# --- Initial Data Loading and Filtering ---

def create_dummy_data(output_path: str):
    """Creates a small dummy CSV file."""
    try:
        logger.info(f"Creating dummy data at {output_path}...")
        dummy_data = {
            'dt': pd.to_datetime(['1900-01-01','1950-01-01','2000-01-01','2010-01-01']*8 + ['1965-01-01', '1985-01-01']*4),
            'AverageTemperature': [5, 7, 9, 10]*8 + [15, 18]*4,
            'City': ['Berlin']*4 + ['Abiko']*4 + ['Munich']*4 + ['Tokyo']*4 + ['Hamburg']*4 + ['Paris']*4 + ['London']*4 + ['New York']*4 + ['Cairo']*2 + ['Sydney']*2 + ['Moscow']*2 + ['Beijing']*2,
            'Country': ['Germany']*4 + ['Japan']*4 + ['Germany']*4 + ['Japan']*4 + ['Germany']*4 + ['France']*4 + ['United Kingdom']*4 + ['USA']*4 + ['Egypt']*2 + ['Australia']*2 + ['Russia']*2 + ['China']*2
        }
        df_dummy = pd.DataFrame(dummy_data)
        df_dummy.to_csv(output_path, index=False)
        logger.info(f"Dummy data created at {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create dummy data at {output_path}: {e}")
        return False

def prepare_filtered_data(config: dict) -> Optional[str]:
    """
    Loads source CSV, filters it based on config, saves the result,
    and returns the path to the filtered file or None on failure.
    Uses dummy data if source is missing and config allows.
    """
    source_csv_path = config['paths']['source_csv_path']
    output_csv_path = config['paths']['processed_data_path']
    countries = config['filter_countries']
    year_start = config['filter_year_start']
    year_end = config['filter_year_end']
    create_dummy = config.get('create_dummy_if_missing', True)

    logger.info(f"Preparing filtered data. Output target: {output_csv_path}")

    if os.path.exists(output_csv_path):
        logger.warning(f"Filtered data file '{output_csv_path}' already exists. Skipping creation.")
        return output_csv_path

    effective_source_path = source_csv_path
    if not os.path.exists(source_csv_path):
        logger.warning(f"Source CSV '{source_csv_path}' not found.")
        if create_dummy:
            if create_dummy_data(source_csv_path): # Try creating dummy in the source location
                 effective_source_path = source_csv_path
            else:
                 logger.error("Failed to create dummy data. Cannot proceed.")
                 return None
        else:
            logger.error(f"Source file missing and dummy creation disabled. Cannot proceed.")
            return None

    try:
        df = pd.read_csv(effective_source_path)
        logger.info(f"Loaded {len(df)} records from {effective_source_path}.")

        # Basic cleaning and type conversion
        df = df.dropna(subset=['AverageTemperature'])
        required_cols = ['dt', 'AverageTemperature', 'City', 'Country']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Source CSV missing required columns: {missing}")

        df_filtered = df[required_cols].copy()
        df_filtered['dt'] = pd.to_datetime(df_filtered['dt'], errors='coerce')
        df_filtered = df_filtered.dropna(subset=['dt'])
        df_filtered['year'] = df_filtered['dt'].dt.year

        # Perform filtering
        country_mask = df_filtered['Country'].isin(countries)
        year_mask = (df_filtered['year'] >= year_start) & (df_filtered['year'] <= year_end)
        df_filtered = df_filtered[country_mask & year_mask]

        if df_filtered.empty:
            logger.error("No data found after applying filters. Check config and source data.")
            return None

        # Save the filtered data
        df_filtered.to_csv(output_csv_path, index=False)
        logger.info(f"Filtered data ({len(df_filtered)} records) saved to {output_csv_path}.")
        return output_csv_path

    except Exception as e:
        logger.error(f"Error during data filtering/saving: {e}", exc_info=True)
        return None

# --- Training Data Generation ---

def generate_training_data(num_examples: int, countries_list: List[str], year_start: int, year_end: int) -> Optional[pd.DataFrame]:
    """Generates synthetic training data (Instruction -> JSON)."""
    logger.info(f"Generating {num_examples} function call training samples...")

    city_country_pairs = {
        'Berlin': 'Germany', 'Tokyo': 'Japan', 'Munich': 'Germany', 'London': 'United Kingdom',
        'Paris': 'France', 'Cairo': 'Egypt', 'Sydney': 'Australia', 'New York': 'USA',
        'Moscow': 'Russia', 'Beijing': 'China', 'Rome': 'Italy', 'Madrid': 'Spain'
    }
    valid_pairs = {city: country for city, country in city_country_pairs.items() if country in countries_list}
    if not valid_pairs:
        logger.error("No valid city/country pairs found for training data generation based on config.")
        return None
    cities = list(valid_pairs.keys())

    plot_type_descs_line = ['line plot', 'line graph', 'temperature trend']
    plot_type_descs_scatter = ['scatter plot', 'scatter with trend', 'regression plot']
    all_plot_type_descs = plot_type_descs_line + plot_type_descs_scatter

    start_years = list(range(year_start, max(year_start + 1, year_end - 10), 5))
    end_years = list(range(min(year_end, year_start + 10), year_end + 1, 5))
    if not start_years or not end_years:
        logger.error(f"Invalid year range [{year_start}-{year_end}] for generating training data years.")
        return None

    training_samples = []
    attempts = 0
    max_attempts = num_examples * 5

    while len(training_samples) < num_examples and attempts < max_attempts:
        attempts += 1
        try:
            city_idx = np.random.randint(len(cities))
            city = cities[city_idx]
            country = valid_pairs[city]
            plot_desc = np.random.choice(all_plot_type_descs)
            start = np.random.choice(start_years)
            valid_ends = [y for y in end_years if y > start]
            if not valid_ends: continue
            end = np.random.choice(valid_ends)

            instruction = f"Show a {plot_desc} for {city}, {country} from {start} to {end}."
            if plot_desc in plot_type_descs_line: plot_type = PlotType.LINE.value
            else: plot_type = PlotType.SCATTER.value

            # output_dict = {
            #     "plot_type": plot_type,
            #     "parameters": {"city": city, "country": country, "start_year": start, "end_year": end, "title": f"{city} Temp {start}-{end}"}
            # }
            # output_json = json.dumps(output_dict, indent=2)
            # training_samples.append({"instruction": instruction, "output": output_json})
            try:
                # Explicitly convert numpy int types to standard Python int
                start_py_int = int(start)
                end_py_int = int(end)

                output_dict = {
                    "plot_type": plot_type,
                    "parameters": {
                        "city": city,
                        "country": country,
                        "start_year": start_py_int, # Use the Python int version
                        "end_year": end_py_int,     # Use the Python int version
                        "title": f"{city} Temp {start}-{end}" # Title string is fine
                    }
                }
                # Now json.dumps should work correctly
                output_json = json.dumps(output_dict, indent=2)
                training_samples.append({"instruction": instruction, "output": output_json})

                if len(training_samples) % 200 == 0:
                    logger.info(f"  ...generated {len(training_samples)} samples.")

            except TypeError as json_err:
                # Add specific logging if the conversion itself fails (less likely)
                logger.error(f"JSON serialization error for sample: {json_err}. Data was: start={start}, end={end}, type_start={type(start)}, type_end={type(end)}")
                continue # Skip this problematic sample
            except Exception as e:
                logger.warning(f"Error generating sample (after int conversion attempt): {e}", exc_info=False)
                continue            
                        
            if len(training_samples) % 200 == 0:
                 logger.info(f"  ...generated {len(training_samples)} samples.")

        except Exception as e:
            logger.warning(f"Error generating sample: {e}", exc_info=False)
            continue

    if len(training_samples) < num_examples:
         logger.warning(f"Generated only {len(training_samples)}/{num_examples} samples after {max_attempts} attempts.")

    if not training_samples:
        logger.error("Failed to generate any training samples.")
        return None

    logger.info(f"Finished generating {len(training_samples)} training samples.")
    return pd.DataFrame(training_samples)


# --- Data Access for Inference ---
_loaded_data: Optional[pd.DataFrame] = None

def load_processed_data_for_inference(processed_data_path: str) -> bool:
    """Loads the filtered data into memory for inference use."""
    global _loaded_data
    if _loaded_data is not None:
        return True # Already loaded
    if not os.path.exists(processed_data_path):
        logger.error(f"Processed data file not found: {processed_data_path}")
        return False
    try:
        df = pd.read_csv(processed_data_path)
        required_cols = ['dt', 'AverageTemperature', 'City', 'Country', 'year']
        if not all(c in df.columns for c in required_cols):
            raise ValueError(f"Processed data missing columns: {required_cols}")
        df['dt'] = pd.to_datetime(df['dt'], errors='coerce')
        df['year'] = df['year'].astype(int) # Ensure year is int
        df = df.dropna(subset=['dt', 'year', 'AverageTemperature'])
        _loaded_data = df
        logger.info(f"Loaded {len(_loaded_data)} records from processed data for inference.")
        return True
    except Exception as e:
        logger.error(f"Failed to load processed data from {processed_data_path}: {e}")
        _loaded_data = None
        return False

def get_yearly_avg_temps(city: str, country: str, start_year: int, end_year: int) -> Optional[pd.Series]:
    """Filters in-memory data and calculates yearly averages."""
    if _loaded_data is None:
        logger.error("Processed data not loaded. Cannot get yearly averages.")
        return None
    try:
        mask = (_loaded_data['City'].str.lower() == city.lower()) & \
               (_loaded_data['Country'].str.lower() == country.lower()) & \
               (_loaded_data['year'] >= start_year) & \
               (_loaded_data['year'] <= end_year)
        filtered = _loaded_data.loc[mask]

        if filtered.empty:
            logger.warning(f"No data found for {city}, {country} ({start_year}-{end_year}) in loaded data.")
            return None

        yearly_avg = filtered.groupby('year')['AverageTemperature'].mean().dropna()
        if yearly_avg.empty:
            logger.warning(f"Yearly averages are empty for {city} ({start_year}-{end_year}).")
            return None

        logger.debug(f"Calculated {len(yearly_avg)} yearly avg points for {city}.")
        return yearly_avg
    except Exception as e:
        logger.error(f"Error getting yearly avg temps for {city}: {e}")
        return None