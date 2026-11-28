"""MCP server exposing pyTEMlib file reading and reference data."""

import argparse
import os

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

import pyTEMlib
from pyTEMlib import file_tools, utilities

mcp = FastMCP("pyTEMlib-MCP")

loaded_datasets = {}


def _dataset(filename: str, dataset_key: str):
    if filename not in loaded_datasets:
        raise ToolError(f"{filename} is not loaded. Call open_tem_file first.")
    datasets = loaded_datasets[filename]
    if dataset_key not in datasets:
        raise ToolError(f"No dataset '{dataset_key}' in {filename}. Available: {list(datasets)}")
    return datasets[dataset_key]


@mcp.tool()
def get_pytemlib_status() -> str:
    """Report the installed pyTEMlib version."""
    return f"pyTEMlib {pyTEMlib.__version__}"


@mcp.tool()
def calculate_electron_wavelength(voltage: float, unit: str = "nm") -> str:
    """Relativistic electron wavelength.

    voltage: acceleration voltage in volts (e.g. 200000 for 200 kV).
    unit: one of m, mm, μm, nm, A, Å, pm.
    """
    if unit not in utilities.units_spatial:
        raise ToolError(f"Unknown unit '{unit}'. Use one of {sorted(utilities.units_spatial)}.")
    wavelength = utilities.get_wavelength(voltage, unit)
    return f"Electron wavelength at {voltage / 1000:g} kV: {wavelength:.5g} {unit}"


@mcp.tool()
def open_tem_file(filename: str) -> str:
    """Open a TEM data file with pyTEMlib and keep its datasets loaded.

    Formats read by pyTEMlib.file_tools.open_file: .dm3, .dm4, .emi, .emd,
    .hf5, .h5, .ndata, .rto, .mrc, and EDAX .h5 files.
    """
    if not os.path.exists(filename):
        raise ToolError(f"File not found: {filename}")
    datasets = file_tools.open_file(filename)
    loaded_datasets[filename] = datasets
    lines = [f"Opened {filename}: {len(datasets)} dataset(s)"]
    for key, dataset in datasets.items():
        lines.append(f"- {key}: {dataset.title} ({dataset.data_type.name}, shape {dataset.shape})")
    return "\n".join(lines)


@mcp.tool()
def list_loaded_datasets() -> str:
    """List datasets from files opened with open_tem_file."""
    if not loaded_datasets:
        return "No datasets loaded. Use open_tem_file to load a file."
    lines = []
    for filename, datasets in loaded_datasets.items():
        lines.append(f"{filename}:")
        lines.extend(f"  - {key}: {dataset.title}" for key, dataset in datasets.items())
    return "\n".join(lines)


@mcp.tool()
def get_dataset_info(filename: str, dataset_key: str) -> str:
    """Title, type, shape, axes and metadata keys of one loaded dataset."""
    dataset = _dataset(filename, dataset_key)
    lines = [
        f"Dataset: {dataset_key}",
        f"Title: {dataset.title}",
        f"Data type: {dataset.data_type.name}",
        f"Shape: {dataset.shape}",
        "Dimensions:",
    ]
    for i in range(dataset.ndim):
        dim = getattr(dataset, f"dim_{i}")
        lines.append(f"  {i}: {dim.name} ({dim.units}, {dim.dimension_type.name}), {len(dim)} points")
    if dataset.metadata:
        lines.append(f"Metadata keys: {list(dataset.metadata)}")
    return "\n".join(lines)


@mcp.tool()
def calculate_probe_size(voltage: float, convergence_angle_mrad: float) -> str:
    """Diffraction-limited probe radius 0.61·λ/α (first Airy minimum).

    Ignores aberrations and source size, so it is a lower bound on the real probe.
    """
    if convergence_angle_mrad <= 0:
        raise ToolError("convergence_angle_mrad must be positive")
    radius = 0.61 * utilities.get_wavelength(voltage, "nm") / (convergence_angle_mrad / 1000.0)
    return (
        f"Diffraction-limited probe radius at {voltage / 1000:g} kV and "
        f"{convergence_angle_mrad:g} mrad: {radius:.4g} nm (aberrations not included)"
    )


@mcp.tool()
def get_element_info(element: str) -> str:
    """Atomic number, atomic weight and ionization edge onsets (eV) of an element.

    element: chemical symbol (e.g. "Fe") or atomic number (e.g. "26").
    """
    try:
        z = utilities.get_z(element)
    except ValueError:
        raise ToolError(f"Unknown element '{element}'") from None
    if not 0 < z < len(utilities.elements):
        raise ToolError(f"Unknown element '{element}'")
    data = utilities.get_x_sections(z)
    lines = [f"Element: {utilities.elements[z]} (Z = {z})"]
    if "atomic_weight" in data:
        lines.append(f"Atomic weight: {data['atomic_weight']}")
    edges = {key: value["onset"] for key, value in data.items() if isinstance(value, dict) and "onset" in value}
    if edges:
        lines.append("Ionization edge onsets:")
        lines.extend(f"  {edge}: {onset:.1f} eV" for edge, onset in sorted(edges.items(), key=lambda e: e[1]))
    return "\n".join(lines)


@mcp.tool()
def create_visualization_script(filename: str, dataset_key: str) -> str:
    """Python script that opens a loaded dataset with pyTEMlib and plots it."""
    _dataset(filename, dataset_key)
    return (
        "from pyTEMlib import file_tools\n"
        "import matplotlib.pyplot as plt\n\n"
        f"datasets = file_tools.open_file({filename!r})\n"
        f"dataset = datasets[{dataset_key!r}]\n"
        "dataset.plot()\n"
        "plt.show()\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=["http", "sse", "stdio"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=args.transport, host=args.host, port=args.port, path="/mcp")


if __name__ == "__main__":
    main()
