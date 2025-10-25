import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from typing import Dict, Callable, Any, List, Optional
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# --- Schemas ---

class PlotType(Enum):
    """Enumeration of the available plot types."""
    LINE = "line_plot"
    SCATTER = "scatter_with_trend"

@dataclass
class PlotRequest:
    """Represents the parameters needed to generate a plot."""
    city: str
    country: str
    start_year: int
    end_year: int
    plot_type: PlotType
    title: Optional[str] = None

    def validate(self):
        """Validates the plot request.

        Raises:
            ValueError: If the year range is invalid, the city or country
                are empty, or the plot type is invalid.
        """
        if not isinstance(self.start_year, int) or not isinstance(self.end_year, int) or self.start_year >= self.end_year:
            raise ValueError(f"Invalid year range: {self.start_year}-{self.end_year}")
        if not self.city or not self.country:
            raise ValueError("City and Country cannot be empty")
        if not isinstance(self.plot_type, PlotType):
            raise ValueError(f"Invalid plot type: {self.plot_type}")
        logger.debug("PlotRequest validated.")

# --- Tool Registry ---

class ToolRegistry:
    """Simple registry for plotting tools."""
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._descriptions: List[Dict[str, Any]] = []
        logger.info("ToolRegistry initialized.")

    def register(self, name: str, description: str, input_params: Dict[str, str]):
        """Decorator to register a plotting function.

        Args:
            name: The name of the tool.
            description: A description of what the tool does.
            input_params: A dictionary describing the input parameters
                for the tool.
        """
        def decorator(func: Callable):
            self._tools[name] = func
            tool_info = { "name": name, "description": description, "required_parameters_from_query": input_params }
            self._descriptions = [d for d in self._descriptions if d['name'] != name] # Avoid duplicates
            self._descriptions.append(tool_info)
            logger.info(f"Tool '{name}' registered.")
            return func
        return decorator

    def get_tool(self, name: str) -> Optional[Callable]:
        """Retrieves a tool from the registry.

        Args:
            name: The name of the tool to retrieve.

        Returns:
            The tool function, or None if the tool is not found.
        """
        tool = self._tools.get(name)
        if tool is None:
            logger.warning(f"Tool '{name}' not found.")
        return tool

    def get_tool_descriptions(self) -> List[Dict[str, Any]]:
        """Returns the descriptions of all registered tools."""
        return self._descriptions

    def get_tool_names(self) -> List[str]:
        """Returns the names of all registered tools."""
        return list(self._tools.keys())

    def get_tool_descriptions_json(self) -> str:
        """Returns the tool descriptions as a JSON string."""
        try:
            return json.dumps(self._descriptions, indent=2)
        except Exception as e:
            logger.error(f"Error generating tool descriptions JSON: {e}")
            return "[]"

# Global instance
tool_registry = ToolRegistry()

# --- Plotting Tools (Registered) ---

@tool_registry.register(
    name="line_plot",
    description="Generates a line plot of temperature trends.",
    input_params={ "city": "string", "country": "string", "start_year": "integer", "end_year": "integer", "title": "string (optional)" }
)
def plot_line(data: pd.Series, title: str) -> Optional[plt.Figure]:
    """Generates a line plot.

    Args:
        data: A pandas Series with the yearly average temperatures.
        title: The title of the plot.

    Returns:
        A matplotlib Figure object, or None if an error occurs.
    """
    if not isinstance(data, pd.Series) or len(data) < 2:
        logger.warning("plot_line: Not enough data points.")
        return None
    fig = None
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data.index, data.values, marker='o')
        ax.set_title(title); ax.set_xlabel("Year"); ax.set_ylabel("Avg Temp (°C)"); ax.grid(True)
        plt.tight_layout()
        logger.info(f"Line plot created: '{title}'")
        return fig
    except Exception as e:
        logger.error(f"Error in plot_line for '{title}': {e}", exc_info=True)
        if fig: plt.close(fig)
        return None

@tool_registry.register(
    name="scatter_with_trend",
    description="Generates a scatter plot with a linear trend line.",
    input_params={ "city": "string", "country": "string", "start_year": "integer", "end_year": "integer", "title": "string (optional)" }
)
def plot_scatter(data: pd.Series, title: str) -> Optional[plt.Figure]:
    """Generates a scatter plot with a linear trend line.

    Args:
        data: A pandas Series with the yearly average temperatures.
        title: The title of the plot.

    Returns:
        A matplotlib Figure object, or None if an error occurs.
    """
    if not isinstance(data, pd.Series) or len(data) < 2:
        logger.warning("plot_scatter: Not enough data points.")
        return None
    fig = None
    try:
        years = data.index.values; temps = data.values
        if np.all(years == years[0]): # Handle single year case for linregress
             logger.warning("plot_scatter: Cannot calculate trend for single year data.")
             fig, ax = plt.subplots(figsize=(10, 5)); ax.scatter(years, temps, label='Data')
        else:
            slope, intercept, r_value, _, _ = stats.linregress(years, temps)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(years, temps, label='Data Points')
            ax.plot(years, intercept + slope * years, 'r', label=f'Trend (R²={r_value**2:.2f})')
            ax.legend()

        ax.set_title(title); ax.set_xlabel("Year"); ax.set_ylabel("Avg Temp (°C)"); ax.grid(True)
        plt.tight_layout()
        logger.info(f"Scatter plot created: '{title}'")
        return fig
    except Exception as e:
        logger.error(f"Error in plot_scatter for '{title}': {e}", exc_info=True)
        if fig: plt.close(fig)
        return None