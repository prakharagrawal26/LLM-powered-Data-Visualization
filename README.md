# LLM Fine-Tuning for Climate Data Visualization

This project demonstrates a complete pipeline for fine-tuning a large language model (LLM) to act as a specialized tool-using agent. The goal is to create a model that can understand natural language queries about climate data and generate the appropriate JSON output to call a plotting function.

## Project Overview

The core of this project is a Python-based pipeline that performs the following steps:

1.  **Data Preparation**: Loads a dataset of global land temperatures, filters it for specific countries and a date range, and prepares it for training.
2.  **Synthetic Data Generation**: Creates a synthetic dataset of instruction-response pairs, where the instructions are natural language queries and the responses are JSON objects.
3.  **Fine-Tuning**: Fine-tunes the `microsoft/phi-2` model using QLoRA on the synthetic dataset.
4.  **Inference**: Uses the fine-tuned model to generate JSON responses to user queries and then executes the corresponding plotting function to create a visualization.

## Setup and Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the data**:
    - The project expects the `GlobalLandTemperaturesByCity.csv` file to be present in the `data` directory.
    - If the file is not found, the script will automatically create a smaller, dummy version for testing purposes.

## Usage

To run the entire pipeline, simply execute the `main.py` script:

```bash
python -m src.main
```

The script will perform the following actions:

1.  **Prepare the data**: If the filtered data file does not exist, it will be created.
2.  **Generate training data**: A synthetic training dataset will be generated.
3.  **Fine-tune the model**: If a fine-tuned adapter is not already present, the script will fine-tune the `microsoft/phi-2` model.
4.  **Run inference tests**: A series of test queries will be run through the fine-tuned model, and the resulting plots will be saved in the `data` directory.

### Example Queries

The following are examples of the types of queries the model is trained to handle:

-   "Create a line plot for Paris, France from 1960 to 2000."
-   "Generate a scatter plot with trend for Tokyo, Japan, 1970-2010."
-   "Show Berlin, Germany temps 1980-2005 as a line graph."
-   "Scatter plot for Sydney, Australia from 1955 to 1995."