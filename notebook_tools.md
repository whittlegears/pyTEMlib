# Analysis tools extracted from the asyncroscopy notebooks

The analysis code in the asyncroscopy notebooks (and the AFM segmentation
notebook in SPM_agent_simple) is now available as instrument-independent pyTEMlib
functions. Each one also has an MCP tool in `pyTEMlib/mcp_analysis_tools.py`, which
`pyTEMlib/mcp_server.py` registers automatically.

The notebook versions were tied to one instrument, one detector and one sample.
The library versions take any 2D image or N-D array, read physical calibration from
parameters instead of constants, and leave instrument control to the caller.

| Notebook(s) | Library functions | MCP tool(s) |
| :-- | :-- | :-- |
| 12 / 12-twin aberration hill climbing, 14 stage-z hill climbing | `image_tools.image_metrics`: `image_contrast`, `image_entropy`, `high_frequency_power`, `edge_sharpness`, `image_quality_metrics` (also per frame of a stack) | `compute_image_metrics` |
| same, plus 18 polycrystalline gold (defocus parabola) | `optimization_tools`: `fit_quadratic_response` (any number of parameters), `propose_next_point`, `TrustRegionOptimizer` (ask/tell), `parabola_vertex` | `propose_next_setting`, `fit_parabola_vertex` |
| 09 hackathon digital twin (segmentation and particle statistics), SPM 6 segmentation agent | `image_tools.image_segmentation`: `otsu_threshold`, `threshold_image` (otsu / percentile / absolute, bright or dark), `clean_mask`, `remove_small_regions`, `segment_particles`, `particle_statistics`, `radial_path`, `gradient_magnitude` | `segment_image` |
| 09 hackathon (lattice parameter from nano-diffraction) | `diffraction_tools.ring_analysis`: `find_direct_beam`, `deconvolve_direct_beam`, `radial_profile`, `find_profile_peaks`, `analyze_ring_pattern`, `angle_to_d_spacing`, `cubic_lattice_parameter` | `analyze_diffraction_rings` |
| 09 hackathon (centre-to-edge lattice profile) | `optimization_tools.linear_trend`, `smoothstep` | `fit_linear_trend` |
| 10 compressed sensing, 09 hackathon (active sampling) | `image_tools.sparse_reconstruction`: `make_sampling_mask`, `sparse_reconstruct` (nearest / DCT / TV; L2 / L1 / Huber), `reconstruct_with_uncertainty`, `next_measurement_locations` | `plan_sampling_mask`, `reconstruct_sparse_data`, `suggest_next_measurements` |
| 16 alpha-tilt diffraction map (drift tracking) | `image_tools.image_alignment`: `cross_correlation_shift`, `fourier_shift`, `align_image_stack`, `align_and_average`, `stage_drift_correction`; `diffraction_tools.four_d_stem.tilt_series_map` | `measure_image_shift`, `align_image_series` |
| 15 MAPED | `diffraction_tools.four_d_stem`: `maped_average`, `virtual_image`, `mean_diffraction_pattern`, `center_of_mass_shifts`, `align_cube_patterns`, `align_cube_scan` | `maped_average_cubes`, `four_d_stem_virtual_image` |

## What changed from the notebook versions

- **Diffraction rings.** The notebook found only the first ring, assumed it was FCC
  {200} and fixed the voltage at 200 kV. Now every ring is reported with its
  d-spacing. You can assign reflections (`hkl`) to get cubic lattice parameters, and
  voltage, calibration and search range are parameters. The radial profile used to
  label bin *i* (radii [i, i+1)) as radius *i*, which made every ring angle 0.5 px too
  small, about 1 % on the lattice parameter in the tests. It now uses bin centres.
  The default minimum ring separation is 1 mrad instead of 2 mrad, because gold's
  {111} and {200} rings are 1.6 mrad apart at 200 kV.
- **Sparse reconstruction.** Data no longer have to lie in [0, 1]. Values are
  normalised by the measured range and returned in the original units. Extra axes
  (colour channels, spectra, diffraction patterns) are reconstructed together.
- **Hill climbing.** The 1D and 2D quadratic fits are now one fit that works in any
  number of dimensions. The optimiser has no instrument calls: the caller sets the
  parameter, measures, and calls `tell`.
- **Alignment.** This replaces `quantem.cross_correlation_shift`. Sub-pixel accuracy
  comes from upsampled-DFT refinement (Guizar-Sicairos et al. 2008). Plain
  cross-correlation is the default because whitened phase correlation was off by up
  to 0.6 px on smooth images in testing.

## Not converted

- **Acquisition-only notebooks** (00, 02, 03, 04, 05, 07, 12 aperture test, 13, 17,
  HACK stage focus, 11 MCP/agent tests) set instrument parameters and read data.
  They contain no analysis to extract. The loops in 07, 13, 14 and 16 stay
  instrument code, which the new tools can drive.
- **20 BlobNet** needs the external `blobnet` package and trained weights.
  Wrapping it would add a heavy optional dependency, so it is left out for now.
- **01 Aberrations / AberrationsBO** are an ipywidgets GUI and a BoTorch
  multi-objective study. Their reusable parts are the image metrics and the
  optimiser above.
- **SPM_agent_simple** notebooks other than 6 (LangGraph agents, PFM control,
  importance-map plots) are agent and instrument code, not image analysis.

## Tests

`tests/test_notebook_tools.py` (30 tests) builds synthetic data with a known answer
for each tool: known shifts, known disk sizes and ellipse axes, a gold ring pattern
at a = 4.078 Å, and a known focus optimum. `tests/test_mcp_analysis_tools.py`
(7 tests) calls every MCP tool through a FastMCP client.
