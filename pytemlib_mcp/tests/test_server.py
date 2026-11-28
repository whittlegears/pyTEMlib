import asyncio
from pathlib import Path

import pytest
from fastmcp import Client

from pytemlib_mcp.server import loaded_datasets, mcp

example_data = Path(__file__).resolve().parents[2] / "example_data"
eds_file = str(example_data / "EDS-STO.emd")


def call(name, arguments=None):
    async def run():
        async with Client(mcp) as client:
            return await client.call_tool(name, arguments or {}, raise_on_error=False)

    return asyncio.run(run())


def text(result):
    return result.content[0].text


def test_tools_listed():
    async def run():
        async with Client(mcp) as client:
            return {tool.name for tool in await client.list_tools()}

    assert asyncio.run(run()) == {
        "get_pytemlib_status",
        "calculate_electron_wavelength",
        "open_tem_file",
        "list_loaded_datasets",
        "get_dataset_info",
        "calculate_probe_size",
        "get_element_info",
        "create_visualization_script",
    }


def test_wavelength_at_200kv():
    result = call("calculate_electron_wavelength", {"voltage": 200000, "unit": "pm"})
    assert "2.5079 pm" in text(result)


def test_unknown_wavelength_unit_is_an_error():
    assert call("calculate_electron_wavelength", {"voltage": 200000, "unit": "furlong"}).is_error


def test_probe_size_is_airy_radius():
    result = call("calculate_probe_size", {"voltage": 200000, "convergence_angle_mrad": 30})
    assert "0.05099 nm" in text(result)


def test_element_info_reports_real_edges():
    body = text(call("get_element_info", {"element": "Ti"}))
    assert "Z = 22" in body
    assert "L3: 455.5 eV" in body
    assert "K1: 4966.4 eV" in body


def test_unknown_element_is_an_error():
    assert call("get_element_info", {"element": "Xx"}).is_error


@pytest.mark.skipif(not Path(eds_file).exists(), reason="example_data not present")
def test_open_file_and_inspect_dataset():
    loaded_datasets.clear()
    opened = call("open_tem_file", {"filename": eds_file})
    assert not opened.is_error
    assert "SuperX" in text(opened)

    info = text(call("get_dataset_info", {"filename": eds_file, "dataset_key": "SuperX"}))
    assert "SPECTRUM" in info
    assert "energy_scale (eV, SPECTRAL), 2048 points" in info

    script = text(call("create_visualization_script", {"filename": eds_file, "dataset_key": "SuperX"}))
    assert "open_file(" in script and "random" not in script


def test_missing_file_is_an_error():
    assert call("open_tem_file", {"filename": "/nonexistent.dm3"}).is_error


def test_dataset_info_requires_open_file():
    loaded_datasets.clear()
    assert call("get_dataset_info", {"filename": "x.dm3", "dataset_key": "a"}).is_error
