"""The MCP server itself (pyTEMlib.mcp_server), called through a real MCP client over the in-memory transport."""

import asyncio
import json
import os

import pytest

fastmcp = pytest.importorskip('fastmcp')

from pyTEMlib import mcp_analysis, mcp_server  # noqa: E402

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'example_data')
EDS_STO = os.path.join(EXAMPLES, 'EDS-STO.emd')
IMAGE = os.path.join(EXAMPLES, 'p1-3-hr3.dm3')

BASIC = {'get_pytemlib_status', 'calculate_electron_wavelength', 'list_supported_file_formats', 'open_tem_file',
         'list_loaded_datasets', 'get_dataset_info', 'calculate_probe_size', 'get_element_info',
         'analyze_diffraction_pattern', 'create_visualization_script', 'get_available_tools'}


def call(name, arguments=None):
    async def run():
        async with fastmcp.Client(mcp_server.mcp) as client:
            return await client.call_tool(name, arguments or {}, raise_on_error=False)
    return asyncio.run(run())


def listed():
    async def run():
        async with fastmcp.Client(mcp_server.mcp) as client:
            return {tool.name: tool for tool in await client.list_tools()}
    return asyncio.run(run())


def test_the_original_tools_are_still_there_and_every_analysis_function_is_a_tool():
    tools = listed()
    analysis = {function.__name__ for function in mcp_analysis.READ_ONLY_TOOLS}
    assert BASIC <= set(tools) and analysis <= set(tools)
    assert len(analysis) == 27 and set(tools) == BASIC | analysis
    assert not {'open_dataset', 'read_array', 'eels_elemental_ratio'} & set(tools)


def test_analysis_tools_are_marked_read_only_and_keep_their_arguments():
    tools = listed()
    for function in mcp_analysis.READ_ONLY_TOOLS:
        assert tools[function.__name__].annotations.readOnlyHint is True, function.__name__
        assert tools[function.__name__].description, function.__name__
    schema = tools['eels_fit_power_law_background'].inputSchema
    assert set(schema['required']) == {'dataset', 'fit_start_ev', 'fit_end_ev'}
    assert schema['properties']['include_curves']['default'] is False


def test_the_tool_list_names_every_tool():
    text = call('get_available_tools').content[0].text
    for name in listed():
        assert f'- {name} - ' in text, name


def test_open_tem_file_reports_the_datasets_instead_of_failing():
    text = call('open_tem_file', {'filename': EDS_STO}).content[0].text
    assert text.startswith('Successfully opened') and 'SuperX: EDS_SuperX (SPECTRUM)' in text and 'Error' not in text
    assert 'File not found' in call('open_tem_file', {'filename': os.path.join(EXAMPLES, 'missing.dm3')}).content[0].text
    info = call('get_dataset_info', {'filename': EDS_STO, 'dataset_key': 'SuperX'}).content[0].text
    assert 'Data type: SPECTRUM' in info  # a file opened by either set of tools is known to both


def test_a_file_read_by_an_analysis_tool_is_known_to_the_old_ones():
    assert 'error' not in call('image_fft', {'dataset': IMAGE, 'max_side': 64}).structured_content
    assert IMAGE in call('list_loaded_datasets').content[0].text


def test_a_tool_returns_structured_content_and_the_same_as_json_text():
    result = call('xray_absorption_edges', {'element': 'Ti'})
    assert result.structured_content['edges'][0]['shell'] == 'K1'
    assert json.loads(result.content[0].text) == result.structured_content


def test_a_refusal_is_an_error_object_not_a_protocol_error():
    result = call('eels_find_zero_loss_peak', {'dataset': os.path.join(EXAMPLES, 'EELS_STO.dm3')})
    assert not result.is_error and result.structured_content['error']['code'] == 'no_zero_loss'


def test_an_image_result_is_a_dict_and_a_png_block():
    result = call('image_fft', {'dataset': IMAGE, 'max_side': 128})
    assert result.structured_content['width'] == 2048 and '_png' not in result.structured_content
    kinds = [block.type for block in result.content]
    assert kinds == ['text', 'image'] and result.content[1].mimeType == 'image/png'
    probe = call('simulate_probe', {'voltage_kv': 200, 'aperture_mrad': 30})
    assert [block.type for block in probe.content] == ['text'] and probe.structured_content['wavelength_nm'] > 0


def test_curves_are_only_sent_when_asked_for():
    arguments = {'dataset': EDS_STO, 'elements': ['Sr', 'Ti', 'O']}
    assert 'curves' not in call('eds_quantify', arguments).structured_content
    curves = call('eds_quantify', {**arguments, 'include_curves': True}).structured_content['curves']
    assert len(curves['model']['x']) == 2048
