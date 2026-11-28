"""
Optional machine-learning tools

These need extra packages and are not imported by ``import pyTEMlib``:

- ``bayesian_optimization``: ``pip install pyTEMlib[ml]`` (torch, gpytorch, botorch)
- ``sam2_segmentation``: additionally SAM2,
  ``pip install "git+https://github.com/facebookresearch/sam2.git" huggingface-hub``

Import the submodule you need, e.g. ``from pyTEMlib.ml_tools import bayesian_optimization``.
"""
