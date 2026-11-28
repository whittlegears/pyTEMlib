"""
MCP Server for pyTEMlib - Transmission Electron Microscopy Data Analysis

This server provides MCP tools for analyzing TEM data using the pyTEMlib library.
It exposes key functionalities for file I/O, data analysis, and visualization.
"""

from fastmcp import FastMCP
import functools
import os
import json
from typing import List
import traceback

# Import pyTEMlib modules
try:
    import pyTEMlib as pytem
    from pyTEMlib import utilities
    PYTEMLIB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: pyTEMlib not available: {e}")
    PYTEMLIB_AVAILABLE = False

# Create MCP server
mcp = FastMCP("pyTEMlib-MCP")

# Global state for loaded datasets: file name as given -> {key: sidpy.Dataset}.
# It is the registry of pytemlib_mcp.analysis, so a file opened by either set of tools is known to both.
if PYTEMLIB_AVAILABLE:
    from pytemlib_mcp import analysis as mcp_analysis
    loaded_datasets = mcp_analysis.loaded_datasets
else:
    mcp_analysis = None
    loaded_datasets = {}

@mcp.tool()
def get_pytemlib_status() -> str:
    """Check if pyTEMlib is properly loaded and available."""
    if not PYTEMLIB_AVAILABLE:
        return "pyTEMlib is not available. Please ensure it's properly installed."

    try:
        version = getattr(pytem, '__version__', 'unknown')
        return f"pyTEMlib v{version} is loaded and ready. Available modules: file_tools, utilities, image_tools, eels_tools, eds_tools, crystal_tools, diffraction_tools, probe_tools, graph_tools"
    except Exception as e:
        return f"pyTEMlib loaded but error checking version: {str(e)}"

@mcp.tool()
def calculate_electron_wavelength(voltage: float, unit: str = "nm") -> str:
    """Calculate the relativistic de Broglie wavelength of electrons.

    Parameters:
    - voltage: Acceleration voltage in volts (e.g., 200000 for 200kV)
    - unit: Unit for wavelength ('m', 'mm', 'μm', 'nm', 'A', 'Å', 'pm')

    Returns the wavelength in the specified unit.
    """
    if not PYTEMLIB_AVAILABLE:
        return "pyTEMlib not available"

    try:
        wavelength = utilities.get_wavelength(voltage, unit)
        return f"Electron wavelength at {voltage/1000:.0f}kV: {wavelength:.3f} {unit}"
    except Exception as e:
        return f"Error calculating wavelength: {str(e)}"

@mcp.tool()
def list_supported_file_formats() -> str:
    """List the file formats supported by pyTEMlib for reading TEM data."""
    formats = [
        ".dm3, .dm4 - Digital Micrograph files",
        ".emd - EMD files",
        ".hf5 - HDF5/pyNSID files",
        ".ndata, .h5 - Nion data files",
        ".mrc - MRC image files",
        ".rto - Bruker files",
        ".npy - numpy arrays (1-D spectrum, 2-D image, 3-D image stack)",
        ".csv, .txt, .tsv, .dat, .xy - text spectra (one or two numeric columns)",
        ".msa, .emsa - EMSA/MAS spectra"
    ]
    return "Supported file formats:\n" + "\n".join(f"- {fmt}" for fmt in formats)

@mcp.tool()
def open_tem_file(filename: str) -> str:
    """Open a TEM data file and load its datasets.

    Parameters:
    - filename: Path to the TEM data file

    Returns the keys, titles, types and shapes of the datasets in the file.
    open_dataset returns the same as structured data.
    """
    if not PYTEMLIB_AVAILABLE:
        return "pyTEMlib not available"

    try:
        if not os.path.exists(filename):
            return f"File not found: {filename}"

        datasets = mcp_analysis.load_datasets(filename)

        result = f"Successfully opened {filename}\n"
        result += f"Found {len(datasets)} datasets:\n"

        for key, dataset in datasets.items():
            result += f"- {key}: {dataset.title} ({dataset.data_type.name})\n"
            if hasattr(dataset, 'shape'):
                result += f"  Shape: {dataset.shape}\n"
        return result
    except mcp_analysis.AnalysisError as e:
        return f"Error opening file: {e.message}"
    except Exception as e:
        return f"Error opening file: {str(e)}\n{traceback.format_exc()}"

@mcp.tool()
def list_loaded_datasets() -> str:
    """List all currently loaded datasets from opened files."""
    if not loaded_datasets:
        return "No datasets currently loaded. Use open_tem_file() to load data."

    result = "Loaded datasets:\n"
    for filename, datasets in loaded_datasets.items():
        result += f"\nFile: {filename}\n"
        for key, dataset in datasets.items():
            result += f"  - {key}: {dataset.title}\n"

    return result

@mcp.tool()
def get_dataset_info(filename: str, dataset_key: str) -> str:
    """Get detailed information about a specific dataset.

    Parameters:
    - filename: The file containing the dataset
    - dataset_key: The key/name of the dataset
    """
    if filename not in loaded_datasets:
        return f"File {filename} not loaded. Use open_tem_file() first."

    datasets = loaded_datasets[filename]
    if dataset_key not in datasets:
        available_keys = list(datasets.keys())
        return f"Dataset '{dataset_key}' not found. Available: {available_keys}"

    dataset = datasets[dataset_key]

    result = f"Dataset: {dataset_key}\n"
    result += f"Title: {dataset.title}\n"
    result += f"Data type: {dataset.data_type.name}\n"

    if hasattr(dataset, 'shape'):
        result += f"Shape: {dataset.shape}\n"

    if hasattr(dataset, 'dimensions'):
        result += "Dimensions:\n"
        for i, dim in enumerate(dataset.dimensions):
            result += f"  {i}: {dim.name} ({dim.units})\n"

    if hasattr(dataset, 'metadata') and dataset.metadata:
        result += f"Metadata keys: {list(dataset.metadata.keys())}\n"

    return result

@mcp.tool()
def calculate_probe_size(voltage: float, convergence_angle_mrad: float) -> str:
    """Calculate electron probe size using pyTEMlib probe_tools.

    Parameters:
    - voltage: Acceleration voltage in volts
    - convergence_angle_mrad: Convergence angle in milliradians
    """
    if not PYTEMLIB_AVAILABLE:
        return "pyTEMlib not available"

    try:
        # Convert mrad to radians
        convergence_angle_rad = convergence_angle_mrad / 1000.0

        # This is a simplified calculation - actual implementation would use probe_tools
        wavelength = utilities.get_wavelength(voltage)
        probe_size = 0.61 * wavelength / convergence_angle_rad  # Airy disk approximation

        return f"Estimated probe size at {voltage/1000:.0f}kV with {convergence_angle_mrad}mrad convergence: {probe_size*1e9:.1f} nm"

    except Exception as e:
        return f"Error calculating probe size: {str(e)}"

@mcp.tool()
def analyze_diffraction_pattern(filename: str, dataset_key: str) -> str:
    """Analyze a diffraction pattern dataset.

    Parameters:
    - filename: File containing the diffraction data
    - dataset_key: Key of the diffraction dataset
    """
    if not PYTEMLIB_AVAILABLE:
        return "pyTEMlib not available"

    if filename not in loaded_datasets:
        return f"File {filename} not loaded."

    datasets = loaded_datasets[filename]
    if dataset_key not in datasets:
        return f"Dataset {dataset_key} not found."

    try:
        dataset = datasets[dataset_key]

        # Basic analysis - check if it's a 2D image
        if len(dataset.shape) != 2:
            return f"Dataset {dataset_key} is not a 2D diffraction pattern (shape: {dataset.shape})"

        result = f"Diffraction pattern analysis for {dataset_key}:\n"
        result += f"Shape: {dataset.shape}\n"
        result += f"Data range: {dataset.min():.3f} to {dataset.max():.3f}\n"

        # This would be extended with actual diffraction analysis functions
        result += "\nNote: Full diffraction analysis requires additional parameters (camera length, voltage, etc.)"

        return result

    except Exception as e:
        return f"Error analyzing diffraction pattern: {str(e)}"

@mcp.tool()
def create_visualization_script(filename: str, dataset_key: str) -> str:
    """Generate Python code that reopens a loaded dataset and plots it with sidpy.

    Parameters:
    - filename: File containing the dataset
    - dataset_key: Key of the dataset to visualize
    """
    if filename not in loaded_datasets:
        return f"File {filename} not loaded."
    if dataset_key not in loaded_datasets[filename]:
        return f"Dataset {dataset_key} not found."
    return (
        "import matplotlib.pyplot as plt\n"
        "from pytemlib_mcp import analysis\n\n"
        f"dataset = analysis.resolve({f'{filename}#{dataset_key}'!r})\n"
        "dataset.plot()\n"
        "plt.show()\n"
    )

# The first tools of this server, in the order they have always been listed.
BASIC_TOOLS = [
    "get_pytemlib_status - Check if pyTEMlib is loaded",
    "calculate_electron_wavelength - Calculate electron wavelength",
    "list_supported_file_formats - Show supported file formats",
    "open_tem_file - Open TEM data files",
    "list_loaded_datasets - List currently loaded datasets",
    "get_dataset_info - Get detailed dataset information",
    "calculate_probe_size - Calculate electron probe size",
    "analyze_diffraction_pattern - Analyze diffraction patterns",
    "create_visualization_script - Generate visualization code",
    "get_available_tools - List all available tools",
]

# Tools that only read and compute. The analysis tools below are all of this kind.
READ_ONLY_HINT = {"readOnlyHint": True}
ANALYSIS_TOOLS: List[str] = []


def _first_line(function) -> str:
    return (function.__doc__ or "").strip().split("\n")[0].rstrip(".")


def _register_analysis_tool(function, annotations):
    """Register one pytemlib_mcp.analysis function as an MCP tool with its own name, signature and docstring.

    A result that carries an image (key ``_png``) becomes its dict plus a PNG image content block.
    """
    from fastmcp.tools import ToolResult
    from fastmcp.utilities.types import Image

    @functools.wraps(function)
    def run(*args, **kwargs):
        result = function(*args, **kwargs)
        if isinstance(result, dict) and "_png" in result:
            png = result.pop("_png")
            return ToolResult(content=[json.dumps(result), Image(data=png, format="png")], structured_content=result)
        return result

    mcp.tool(annotations=annotations)(run)
    ANALYSIS_TOOLS.append(f"{function.__name__} - {_first_line(function)}")


if PYTEMLIB_AVAILABLE:
    # Dataset reading, reference data, EDS, EELS, peak fitting, image and probe tools. Each takes a dataset
    # reference ("path" or "path#key"), returns a dict, and refuses with {"error": {"code", "message"}}.
    for _function in mcp_analysis.READ_ONLY_TOOLS:
        _register_analysis_tool(_function, READ_ONLY_HINT)
    for _function in mcp_analysis.STORING_TOOLS:
        _register_analysis_tool(_function, None)


@mcp.tool()
def get_available_tools() -> str:
    """List all available MCP tools for pyTEMlib."""
    return "Available pyTEMlib MCP Tools:\n" + "\n".join(f"- {tool}" for tool in BASIC_TOOLS + ANALYSIS_TOOLS)


def main(argv=None):
    """Run the server. Without arguments: SSE on 127.0.0.1:8003 at /mcp, as it always has."""
    import argparse
    parser = argparse.ArgumentParser(description="pyTEMlib MCP server")
    parser.add_argument("--transport", default="sse", choices=("sse", "streamable-http", "http", "stdio"))
    parser.add_argument("--host", default="127.0.0.1", help='"127.0.0.1" for local-only, "0.0.0.0" to expose on your network')
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--path", default="/mcp")
    args = parser.parse_args(argv)
    if args.transport == "stdio":
        mcp.run(transport="stdio")
        return
    print("Starting pyTEMlib MCP Server...")
    mcp.run(transport=args.transport, host=args.host, port=args.port, path=args.path)


# Run the server
if __name__ == "__main__":
    main()
