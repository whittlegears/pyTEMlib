"""pyTEMlib's analysis tools on Tiled keys: acquisitions as asyncroscopy writes them, in a real Tiled server."""

import asyncio

import numpy as np
import pytest
from fastmcp import Client

mcp_analysis = pytest.importorskip("pyTEMlib.mcp_analysis")
from tiled.client import from_uri  # noqa: E402
from tiled.server import SimpleTiledServer  # noqa: E402

from pyTEMlib.analysis_server import create_server  # noqa: E402
from pyTEMlib.analysis_server.tiled_datasets import install, to_sidpy  # noqa: E402
from pyTEMlib.analysis_server import stored  # noqa: E402


def gaussian(x, centre, fwhm, height):
    return height * np.exp(-0.5 * ((x - centre) / (fwhm / 2.3548)) ** 2)


@pytest.fixture(scope="module")
def catalogue(tmp_path_factory):
    server = SimpleTiledServer(directory=tmp_path_factory.mktemp("tiled"))
    client = from_uri(server.uri)
    energy = 100.0 + 10.0 * np.arange(1024)
    spectrum = 5.0 + gaussian(energy, 4510.0, 130.0, 800.0) + gaussian(energy, 8040.0, 150.0, 400.0)
    client.write_array(spectrum[np.newaxis, :], key="eds_point", metadata={
        "acquisition_type": "spectrum", "detector": "eds", "energy_offset_ev": 100.0, "energy_dispersion_ev": 10.0,
        "acceleration_voltage_v": 200000.0})
    cube = np.tile(spectrum, (4, 5, 1)) * np.arange(1, 21).reshape(4, 5, 1)
    client.write_array(cube, key="eds_map", metadata={
        "acquisition_type": "spectrum_image", "energy_offset_ev": 100.0, "energy_dispersion_ev": 10.0,
        "spectrum_image_axes": "rows=+y, columns=+x, channels"})
    lattice = np.cos(2 * np.pi * np.arange(64) / 8.0)
    run = client.create_container("run_1", metadata={"acquisition_type": "image"})
    run.write_array(np.add.outer(lattice, lattice) + 2.0, key="haadf", metadata={"pixel_size_m": 2e-10})
    client.write_array(np.zeros((3, 16, 16)), key="stack", metadata={"pixel_size_m": 1e-9})
    client.write_array(np.ones(50), key="profile", metadata={})
    yield client
    server.close()


def described(catalogue, reference):
    key, _, path = reference.partition("#")
    return to_sidpy(stored.pick(catalogue[key], key, path or None))


def test_a_streamed_spectrum_of_one_row_is_a_spectrum_on_its_recorded_energy_scale(catalogue):
    dataset = described(catalogue, "eds_point")
    assert dataset.data_type.name == "SPECTRUM" and dataset.shape == (1024,)
    axis = dataset._axes[0]
    assert axis.units == "eV" and axis.dimension_type.name == "SPECTRAL"
    assert axis.values[0] == pytest.approx(100.0) and axis.values[1] - axis.values[0] == pytest.approx(10.0)
    assert dataset.metadata["experiment"]["acceleration_voltage"] == pytest.approx(200000.0)


def test_a_spectrum_image_has_positions_then_channels(catalogue):
    dataset = described(catalogue, "eds_map")
    assert dataset.data_type.name == "SPECTRAL_IMAGE" and dataset.shape == (4, 5, 1024)
    assert [dataset._axes[i].dimension_type.name for i in range(3)] == ["SPATIAL", "SPATIAL", "SPECTRAL"]


def test_an_image_in_a_run_is_in_nm_and_a_stack_has_frames_first(catalogue):
    image = described(catalogue, "run_1#haadf")
    assert image.data_type.name == "IMAGE" and image._axes[0].units == "nm"
    assert image._axes[1].values[1] == pytest.approx(0.2)
    stack = described(catalogue, "stack")
    assert stack.data_type.name == "IMAGE_STACK" and stack._axes[0].dimension_type.name == "TEMPORAL"


def test_a_profile_without_an_energy_scale_is_refused_not_guessed(catalogue):
    with pytest.raises(mcp_analysis.AnalysisError) as refused:
        described(catalogue, "profile")
    assert refused.value.code == "uncalibrated"


def test_pytemlib_tools_take_tiled_keys(catalogue, tmp_path):
    install(lambda: catalogue, [tmp_path])
    try:
        peaks = mcp_analysis.find_spectrum_peaks("eds_point", max_peaks=2)
        assert "error" not in peaks, peaks
        assert sorted(round(p["position"], -1) for p in peaks["peaks"]) == [4510.0, 8040.0]
        summed = mcp_analysis.find_spectrum_peaks("eds_map", max_peaks=1)
        assert summed["summed_spectra"] == 20
        fft = mcp_analysis.image_fft("run_1#haadf", include_array=True)
        assert "error" not in fft and fft["reciprocal_units"][0] == "1/nm"
        assert mcp_analysis.image_fft("profile")["error"]["code"] == "uncalibrated"
        assert mcp_analysis.image_fft("nothing_here")["error"]["code"] == "path_not_allowed"
        assert mcp_analysis.image_fft(str(tmp_path / "nothing.dm3"))["error"]["code"] == "file_not_found"
    finally:
        install(lambda: None, None)


def test_an_hdf5_file_asyncroscopy_wrote_is_read_with_the_same_rules(tmp_path):
    import h5py

    energy = 100.0 + 10.0 * np.arange(1024)
    path = tmp_path / "eds.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs["instrument_id"] = "twin"
        spectrum = h5.create_dataset("spectrum", data=(5.0 + gaussian(energy, 4510.0, 130.0, 800.0))[np.newaxis, :])
        spectrum.attrs.update({"acquisition_type": "spectrum", "energy_offset_ev": 100.0, "energy_dispersion_ev": 10.0})
    plain = tmp_path / "other.h5"
    with h5py.File(plain, "w") as h5:
        h5.create_dataset("data", data=np.zeros(8))
    install(lambda: None, [tmp_path])
    try:
        peaks = mcp_analysis.find_spectrum_peaks(f"{path}#spectrum", max_peaks=1)
        assert "error" not in peaks, peaks
        assert round(peaks["peaks"][0]["position"], -1) == 4510.0
        assert mcp_analysis.find_spectrum_peaks(str(path), max_peaks=1)["peaks"][0]["position"] == peaks["peaks"][0]["position"]
        assert mcp_analysis.find_spectrum_peaks(f"{path}#nothing")["error"]["code"] == "unknown_key"
        assert "error" in mcp_analysis.find_spectrum_peaks(str(plain))
    finally:
        install(lambda: None, None)
    install(lambda: None, [tmp_path / "elsewhere"])
    try:
        assert mcp_analysis.find_spectrum_peaks(f"{path}#spectrum")["error"]["code"] == "path_not_allowed"
    finally:
        install(lambda: None, None)


def test_the_analysis_server_offers_pytemlibs_tools_marked_read_only(tmp_path):
    mcp, _ = create_server(local_roots=[tmp_path])

    async def listed():
        async with Client(mcp) as client:
            return {tool.name: tool for tool in await client.list_tools()}

    tools = asyncio.run(listed())
    for function in mcp_analysis.READ_ONLY_TOOLS:
        assert tools[function.__name__].annotations.read_only_hint, function.__name__
    assert "open_dataset" not in tools and "eels_elemental_ratio" not in tools
