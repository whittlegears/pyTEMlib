"""Each test gets pyTEMlib's analysis tools as it found them: create_server installs its resolvers and file roots
process-wide, and the other test modules expect none."""

import pytest

from pyTEMlib import mcp_analysis


@pytest.fixture(autouse=True)
def analysis_tools_as_found():
    resolvers, roots = list(mcp_analysis._resolvers), mcp_analysis._allowed_roots
    yield
    mcp_analysis._resolvers[:] = resolvers
    mcp_analysis._allowed_roots = roots
