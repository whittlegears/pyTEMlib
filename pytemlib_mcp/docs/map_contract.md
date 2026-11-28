# Map data contract (as built today)

Sections 1 to 6 record the map data that existing code writes and reads (mapping redesign, step S2). Section 7
records the view descriptors the analysis server's tools give clients (step S6).

The machine-readable part is [`../schemas/stitched_map.schema.json`](../schemas/stitched_map.schema.json).
It covers the stitched map record, the level nodes, the point of interest (POI) and the point store.
The run layouts are HDF5 arrays, so they are described here only, in tables.

## Sources

Every row cites the code that writes or reads the field.

| Short name | What it is |
| :-- | :-- |
| `runs.py`, `store.py`, `pyramid.py`, `poi.py`, `map_tools.py` | asyncroscopy `asyncroscopy/analysis_server/map_store/{runs,store,pyramid,poi}.py` and `asyncroscopy/analysis_server/map_tools.py`, at commit `bdc9552` of `feature/agentic-mapping`. |
| `run_file.py`, `mapper.py`, `record.py` | asyncroscopy `asyncroscopy/jobs/mapping/run_file.py`, `asyncroscopy/jobs/mapping/mapper.py`, `asyncroscopy/data/record.py` at `ccdb5c3` (the branch tip). |
| `tiled.ts` | Angstrom `src/lib/tiled.ts` at `05b23ed` (`claude_dev`). |
| `image_mosaic.py` | pyTEMlib `pyTEMlib/image_tools/image_mosaic.py` on `feature/mcp-analysis-tools`. |

The map store now lives in this package as `pytemlib_mcp.analysis_server` (`src/pytemlib_mcp/analysis_server/`).
The line numbers here are from asyncroscopy commit `bdc9552`, the last commit that held it there; the code was
moved without changing those lines.

## 1. Runs

A run is the set of tiles one map is stitched from. A run is one HDF5 file or one Tiled container
with a `tiles` child. The reader picks the layout from `run_layout`. The value 2 or more means
layout 2. A missing value means layout 1 (`store.py:139`, `store.py:149`, `store.py:173`, `store.py:179`).

### 1.1 Run-level metadata

These are HDF5 file attributes, or Tiled container metadata.

| Field | Type | Written by | Read by |
| :-- | :-- | :-- | :-- |
| `run_layout` | int: 1 or 2 | `collect_run` writes 1 (`map_tools.py:379`). The Mapper writes 2 (`run_file.py:29`, `run_file.py:107`). | `store.py:139`, `store.py:149` |
| `plan` | JSON string of an object | `map_tools.py:379`, `mapper.py:291` | `store.py:138`, `store.py:148`. A non-object becomes `{}` (`store.py:158`). |
| `kind` | `"map_run"` | `map_tools.py:379`. The Mapper writes it through `record_fields(..., "map_run", ...)` (`mapper.py:301`, `record.py:85`). | not read by the map store |
| `id` | hex string | `map_tools.py:379`, `record.py:83` | not read by the map store |
| `sample_id`, `sample_load_id` | string | copied from the first image (`map_tools.py:380`), or from the SAMPLE device (`record.py:86`) | `store.py:232-257`. They are copied onto the map (`store.py:210`). |
| `tiles` | int | `map_tools.py:380` | not read by the map store |
| `created` | float, Unix seconds | `map_tools.py:380`, `record.py:88` | not read by the map store |
| `source` | `"workflow"` | `map_tools.py:379` | not read by the map store |
| `spectra_at` | `[x, y]` fractions of the tile image, x from the left, y from the top | `collect_run`, only with `spectrum_keys` (`MapTools._spectra` checks both) | `eds_pois` through `TileSource.attribute`. Without it the spectra are at the tile centre, where the Mapper takes them (asyncroscopy `map_profile.py`, `place_beam([0.5, 0.5])`). |
| `instrument_id`, `code_version`, `sample_error` | string | `record.py:82-90`. Null values are left out (`record.py:90`). | not read by the map store |
| `schema_version`, `run_id`, `plan_id`, `grid` (JSON `{"rows","cols"}`), `axes`, `microscope`, `instrument_profile`, `trace_offset`, `run` | various | Mapper only (`mapper.py:288-300`) | not read by the map store |

### 1.2 Layout 1: one dataset per tile

Layout 1 is for small runs (`__init__.py:13-14`). `collect_map_run` writes it
(`map_tools.py:394-409` for Tiled, `map_tools.py:417-438` for a local file).

- `tiles/tile_NNNNN`: one 2-D array per tile. The name is `tile_{k:05d}` (`map_tools.py:400`, `map_tools.py:431`).
  The reader sorts the names (`store.py:143`, `store.py:156`). A tile with fewer than 2 dimensions is refused (`runs.py:172-173`).
- Each tile's attributes (local) or metadata (Tiled):

| Field | Type | Written | Read |
| :-- | :-- | :-- | :-- |
| `index` | int | `map_tools.py:401`, `map_tools.py:432` | `runs.py:174`. The default is the position in the sorted list. |
| `stage_reported_m` | `[x, y]` metres, tile centre | `map_tools.py:401`, `map_tools.py:433` | `runs.py:169`. Required, unless `commanded_xy_m` is there. |
| `commanded_xy_m` | `[x, y]` metres | not written by `collect_run` | `runs.py:169`, used only when `stage_reported_m` is missing |
| `pixel_size_m` | float | `map_tools.py:401`, `map_tools.py:434` | `runs.py:170`. Required. |
| `source_key` | string | `map_tools.py:402`, `map_tools.py:435` | not read by the map store |
| `acquired` | float seconds | not written by `collect_run` | `runs.py:174`. The run has times only when every tile has one (`runs.py:182`). |

Beside `tiles`, a layout 1 run that `collect_run` gathered with `spectrum_keys` also has:

| Dataset | Shape, type | Written | Read |
| :-- | :-- | :-- | :-- |
| `spectrum_keys` | `[n]` string, one per image in `image_keys` order, `""` where no spectrum was taken | `collect_run`: an HDF5 string dataset (local) or a unicode array child of the run container (Tiled) | `eds_pois` through `TileSource.dataset`, as for layout 2. Entry k belongs to the tile whose `index` is k. |

From Tiled, a layout 1 run of at most `V1_EXPORT_MAX_TILES` (20 000) tiles is read as one HDF5
export. A larger run is read tile by tile (`runs.py:24`, `runs.py:255-262`).

### 1.3 Layout 2: one chunked array for all tiles

Layout 2 is for maps of any size (`__init__.py:7-12`). The Mapper's `RunFile` writes it (`run_file.py:110-158`).

| Dataset | Shape, type | Written | Read |
| :-- | :-- | :-- | :-- |
| `tiles` | `[n, rows, cols]`, the first detector's dtype, chunks `(1, rows, cols)` | `run_file.py:129-136` | `runs.py:191`, `runs.py:247` |
| `tiles` attributes | `detector`, `fov_m`, `pixel_size_m = fov_m / imsize` | `run_file.py:135` | `pixel_size_m` is required (`runs.py:189-190`) |
| `channels/<DETECTOR>` | `[n, rows, cols]`, each further detector | `run_file.py:131` | not read by the map store |
| `written` | `[n]` bool | `run_file.py:123`, `run_file.py:157` | `runs.py:192`. Only written tiles are placed. All tiles count as written if it is missing. |
| `stage_reported_m` | `[n, 5]` float64, NaN until written | `run_file.py:124`, `run_file.py:137` | `runs.py:194`. Only columns 0 and 1 (x, y) are read. |
| `commanded_xy_m` | `[n, 2]` float64 | `run_file.py:125`, `run_file.py:138` | not read by the map store |
| `acquired_s` | `[n]` float64, the instrument `clock_s` | `run_file.py:126`, `run_file.py:139`; meaning `run_file.py:10-11` | `runs.py:195`, `runs.py:201` |
| `row_col` | `[n, 2]` int32, -1 until written | `run_file.py:127`, `run_file.py:140` | `runs.py:196`, `runs.py:202` |
| `source_keys` | `[n]` string | `run_file.py:128`, `run_file.py:141` | not read by the map store |
| `spectra`, `spectrum_keys` | `[n, channels]`, `[n]` string | `run_file.py:142-151` | `spectrum_keys` is read by `eds_pois` (`map_tools.py:108`), in either layout |
| `spectrum_image`, `spectrum_image_keys` | `[n, points, points, channels]`, `[n]` string | `run_file.py:160-178` | not read by the map store |
| further per-tile arrays (for example `drift_correction_m`) | `[n, ...]` float64, NaN-filled | `run_file.py:152-156` | not read by the map store |
| `trace` | string array; attributes `trace_file`, `trace_offset`, `events_not_copied` | `run_file.py:192-196` | not read by the map store |

While tiles are being written, the file is named `.h5.partial`. `finish` renames it to `.h5`
(`run_file.py:3-4`, `run_file.py:204-229`).

## 2. Stitched map

### 2.1 Where a map is stored, and its versions

- Key: `<run stem>_map_<YYYYmmddTHHMMSSmmm>Z_<4 hex>` (`store.py:94-101`, `store.py:194`). The run stem is the
  run's file name without `.h5` or `.hdf5` (`store.py:86-91`). A local map is that name plus `.h5`, in the
  run's folder (`store.py:196`). Maps stitched before versions existed keep the key `<run stem>_map` (`__init__.py:27-28`).
- Local file: the levels are datasets in a temporary `.<8 hex>.tmp` file. The JSON record is the file attribute
  `metadata`. The file is renamed to its final name when the map is done (`pyramid.py:24-44`).
- Tiled: a container created with `{"kind": "stitching", "run": run}` (`store.py:197`). While it is being written, its
  metadata is the record with `kind` = `"stitching"`. The `swap` step then sets `kind` to `"stitched_map"` (`pyramid.py:86-93`).
  A failed write deletes the container (`pyramid.py:95-108`).
- `list_maps` finds versions by `kind == "stitched_map"` and `run == <run>`. It sorts them by `stitched_at`, newest first
  (`store.py:103-127`). Angstrom lists maps by `kind == "stitched_map"` alone (`tiled.ts:611-634`). The `list_maps` tool
  now also lists every map when it is given no run (section 7.1), so a client need not search Tiled itself.

### 2.2 The record

The schema root is this record. The record is made at `store.py:210` from
`{kind, run, axes, id, parents, version, **sample_of(run), **metadata}`. Here `metadata` is the dict that
`map_tools.py:246-266` builds.

| Field | Type | Source | Angstrom reads |
| :-- | :-- | :-- | :-- |
| `kind` | `"stitched_map"` | `store.py:210` | `tiled.ts:617-618` (search) |
| `run` | string, the run key or path | `store.py:210` | `tiled.ts:341`, `tiled.ts:639-640` |
| `axes` | `"rows=+y, columns=+x; stage coordinates in metres"` | `store.py:19`, `store.py:210` | no |
| `id` | 32 hex characters (`uuid4().hex`) | `store.py:210` | no |
| `parents` | `[run]` | `store.py:210` | no |
| `version` | the writer's name: the local path, or the Tiled key | `store.py:210`, `pyramid.py:26`, `pyramid.py:61` | no |
| `sample_id`, `sample_load_id` | non-empty string, only when the run records them | `store.py:210`, `store.py:232-257` | no |
| `levels` | int, from `pyramid_levels(canvas_shape)` | `map_tools.py:238`, `map_tools.py:247`; `image_mosaic.py:1052-1059` | `tiled.ts:327` (default 1) |
| `downsample` | `[2**k for k < levels]` | `map_tools.py:248` | `tiled.ts:328-331` (default `2**k`) |
| `pixel_size_m` | float > 0, level 0 | `map_tools.py:249` (`Placement.pixel_size_m`) | `tiled.ts:317-320` (required) |
| `origin_xy_m` | `[x, y]`, the stage position of the centre of level-0 pixel (0, 0) | `map_tools.py:250`; meaning `image_mosaic.py:186` | `tiled.ts:318`, `tiled.ts:321-323` (required) |
| `canvas_shape` | `[rows, cols]` of level 0 | `map_tools.py:251` | `tiled.ts:319`, `tiled.ts:324-326` (required) |
| `level_shapes` | `[[rows, cols], ...]`, one per level | `map_tools.py:252`; `image_mosaic.py:1062-1068` | no |
| `tiles` | int, the tiles placed | `map_tools.py:253` | no |
| `stitched_at` | float, Unix seconds | `map_tools.py:254` | no |
| `refined` | bool | `map_tools.py:255` | no |
| `stage_sigma_m` | float. If the plan gives none, it is 5 % of the median tile width | `map_tools.py:220-222`, `map_tools.py:256` | no |
| `inputs_sha256` | 64 hex characters | `map_tools.py:158-165`, `map_tools.py:257` | no |
| `pytemlib_version` | string | `map_tools.py:258` | no |
| the `stitch_quality` fields (2.3) | see 2.3 | `map_tools.py:245`, `map_tools.py:259` | no |
| `drift_unmeasured` | string `DRIFT_UNMEASURED` | `map_tools.py:125`, `map_tools.py:139-146` | no |
| `tile_positions_px` | `[[row, col], ...]` top-left on level 0, rounded to 3 places | only when tiles <= 10 000 (`map_tools.py:261-262`) | no |
| `tile_gains`, `tile_offsets` | float list or null | `map_tools.py:264-265` (same condition) | no |
| `scan_linear_applied` | 2 x 2 | `map_tools.py:266` (same condition) | no |

The record has `drift` or `drift_unmeasured`, never both. `measured_drift_only` keeps `drift` only when
`matches_used >= 2 * len(drift.times_s)`. Otherwise it removes `drift` and adds `drift_unmeasured`
(`map_tools.py:128-146`).

`finite_or_none` replaces every non-finite float in the quality fields with null before storing
(`map_tools.py:147-155`, `map_tools.py:245`).

Old maps can have fields that the current writer does not write. These fields are read, but they
are not in the schema:

- `pois`: points kept inside the map (`poi.py:168`, `poi.py:211`, `tiled.ts:332`).
- `poi_layer`: the layer name, `"poi"` when missing (`poi.py:138-139`, `tiled.ts:344`).
- `level_shapes` can be missing. Then `level_shape_of` computes the level shapes from `canvas_shape` (`pyramid.py:115-128`).

### 2.3 `stitch_quality` fields

These fields are returned by `image_mosaic.stitch_quality` (`image_mosaic.py:1267-1312`). The schema
defines them in `$defs/stitch_quality_fields`. `$defs/stitch_quality` is the output on its own, with
no other field allowed.

| Field | When | Type |
| :-- | :-- | :-- |
| `matches_used`, `matches_found` | always | int |
| `median_residual_px`, `p95_residual_px` | when a match was used | float, 3 places |
| `fallback_tiles`, `unplaced_tiles` | when `refine` was on (`Placement.fallback` is set) | int; at most `UNPLACED_LISTED` indices |
| `scan_linear` | always, from `register_tiles` | `{applied: bool, matrix: 2x2, stderr: 2x2 or null}`. `stderr` is null when fewer than `LINEAR_MIN_TILES` tiles matched (`image_mosaic.py:959-962`). |
| `tile_gain` | when gains were solved | `{min, max}` |
| `solver` | when a solve ran | `{method: "direct", converged}` or `{method: "lsqr", converged, iterations, istop}` (`image_mosaic.py:801-810`) |
| `drift` | when there were times and 2 or more knots | `{times_s, xy_m, total_nm, mean_rate_nm_per_s}` (`image_mosaic.py:210-222`) |

### 2.4 Pyramid levels

- Datasets or Tiled arrays named `level_0` to `level_<levels-1>`. Each level is the 2 x 2 average of the level before.
  Each level's shape is the shape of the level before, halved and rounded down, and at least 1 x 1
  (`image_mosaic.py:1062-1068`).
- Local: chunks are `(min(block, rows), min(block, cols))`, and the datasets have no attributes (`pyramid.py:30-31`).
- Tiled: chunks are `block` along each axis, with a smaller last chunk. The node metadata is `{"downsample": 2**k}`
  (`pyramid.py:65-81`). This is `$defs/level_node`.
- The block side is `BLOCK` = 512 (`map_tools.py:71`, `map_tools.py:241`).
- Angstrom finds levels by the node name `level_(\d+)`. It takes the shape, `data_type` and `chunks` from the node's
  structure. It takes the factor from the node's `downsample`, then from the map's `downsample[k]`, then `2**k`
  (`tiled.ts:686-708`).

### 2.5 Pixel to stage transform

The map is placed by its origin and pixel size alone. The axes are aligned with the stage: rows go along +y
and columns along +x (`store.py:19`). Level-k pixel (r, c) averages the level-0 pixels
`[r*2^k, (r+1)*2^k) x [c*2^k, (c+1)*2^k)` (`pyramid.py:147-156`):

```
x = origin_xy_m[0] + (c * 2^k + (2^k - 1) / 2) * pixel_size_m
y = origin_xy_m[1] + (r * 2^k + (2^k - 1) / 2) * pixel_size_m
```

- At level 0 this is the same as `Placement.pixel_to_stage` (`image_mosaic.py:204-205`) and Angstrom's `pixelToStage`
  (`tiled.ts:419-424`, level-0 coordinates).
- The map's stage box runs from pixel (-0.5, -0.5) to pixel (rows - 0.5, cols - 0.5) (`map_tools.py:487-488`).
  `stage_box_to_pixels` works out the inverse for a box (`pyramid.py:131-144`).
- `scan_linear_applied` is not part of this transform. It is the 2 x 2 map from canvas pixels to tile pixels, used
  only while drawing (`image_mosaic.py:192-194`).

## 3. Points of interest

### 3.1 A point

Every point is the object `$defs/poi`. Three writers make points, and each writes all nine
required fields:

| Writer | `kind` | `source` | Other fields |
| :-- | :-- | :-- | :-- |
| `eds_pois` (`map_tools.py:103-122`) | `"eds_point"` | `"run"` | One per non-empty entry of the run's `spectrum_keys`. `id` = `<run name>:eds:<tile>`, `links` = `[spectrum key]`, `score` null. `x_m`, `y_m`: the point of the stitched tile where the spectrum was taken, the run's `spectra_at` (tile centre without it). A tile offset `(f - 0.5) * shape` in (row, col) pixels from the tile centre is turned into map pixels with the inverse of `scan_linear_applied`, then into stage metres by 2.5. |
| `score_map` (`map_tools.py:467-472`) | `"candidate"` | `"scorer:<name>"` | `score`, `scorer_version`, `window_m`; `tag` when given; `id` = `<tag>:region:<k>` or 8 hex |
| `add_poi` (`map_tools.py:769`) | the caller's, default `"note"` | `"agent"` | `id` given or 8 hex |

Angstrom reads `id`, `x_m`, `y_m`, `kind`, `source`, `score`, `scorer_version`, `reason` and `links`. It skips a point
whose `x_m` or `y_m` is not finite (`tiled.ts:374-394`). It does not read `window_m` or `tag`.

### 3.2 The point store

Points belong to the run, not to a map version. Every version of a run's map uses one point store.

- Key: `<run stem>_points` in Tiled, or `<run stem>_points.h5` beside a local run (`poi.py:26-29`). Angstrom derives
  the same key from `run` (`tiled.ts:351-354`).
- Metadata (`$defs/poi_store`): first `{kind: "poi_store", run, grid, poi_layer: "poi_<8 hex>", layers: [], ready: false}`,
  with `migrated_from` when it copied an old map's points (`poi.py:101-103`). `ready` is set true once the copy is
  complete (`poi.py:115-116`). `update_highlights` adds `highlights` and `replaced_highlights` (`poi.py:336-341`).
  `update_pois` adds entries to `layers` and starts a new `poi_layer` (`poi.py:308-314`). A local store keeps
  this metadata as the JSON attribute `metadata` (`poi.py:43-57`).
- Grid (`$defs/poi_grid`): `x0 = origin_x - pixel / 2`, `y0 = origin_y - pixel / 2`,
  `cell_m = max(rows, cols) * pixel / 64`, `n` = `POI_CELLS` = 64 (`poi.py:18`, `poi.py:119-129`).
  Cell (i, j) is `floor((y - y0) / cell_m)`, `floor((x - x0) / cell_m)`, clipped to the grid (`poi.py:131-136`).
- Layout: `<store>/<poi_layer>/<group>/<batch>`. `eds_point` points go in group `c_<i>_<j>`. All other points go
  in the current highlights group: `highlights`, or the store's `highlights` value once replaced (`poi.py:19-20`, `poi.py:141-150`).
  - Tiled: the layer node metadata is `{kind: "poi_layer", grid}` (`poi.py:287`, `poi.py:365`), the group node metadata
    is `{count}` (`poi.py:290`, `poi.py:295`), and the batch node `b_<20-digit ns>_<6 hex>` has metadata `{pois: [...]}` (`poi.py:277`, `poi.py:294`).
  - Local: a group has the attribute `count`. A batch is a UTF-8 string dataset that holds the JSON array (`poi.py:278-284`).
- Angstrom reads the store's `kind`, `run`, `ready`, `grid` and `poi_layer` (`tiled.ts:356-372`). It reads the
  highlights from the literal group `highlights` (`tiled.ts:664-666`), the cells from `c_<i>_<j>` (`tiled.ts:668-670`),
  the cell counts from the `count` metadata of `c_*` children (`tiled.ts:672-684`), and the points from each batch's
  `pois` (`tiled.ts:650-662`). A map with no store gets the grid from the map's own metadata, by the same formula (`tiled.ts:333`, `tiled.ts:345`).

## 4. pyTEMlib's own map record

`image_mosaic.stitch_mosaic` returns a sidpy dataset. It holds its placement in `metadata['mosaic']`
(`image_mosaic.py:1381-1391`): `positions_px`, `stage_positions_px`, `origin_xy_m`, `pixel_size_m`, `sources`,
`tile_gains`, `tile_offsets`, `scan_linear_applied` and the `stitch_quality` fields. The map's centre is in
`metadata['experiment']['stage_position']` (`image_mosaic.py:1379`). This record is not stored by the map store,
and it is not validated by the schema.

## 5. Tests

`pytemlib_mcp/tests/test_map_contract.py` checks that the schema is valid draft 2020-12. It also validates real
`stitch_quality` output from `register_tiles` against `$defs/stitch_quality`: a 2 x 2 map stitched without
refinement, and a refined 4 x 4 map with times and drift.

The whole stitched map record is not validated. No code in pyTEMlib writes that record: it is put together
in asyncroscopy's `MapTools._assemble` and `MapStore.write_map`.

## 6. Open questions

These are listed, not resolved.

1. **Affine or origin plus pixel size (D6).** The map transform is axis-aligned (2.5). The stitcher measures the scan's
   stretch, rotation and shear (`scan_linear`), but uses them only to draw the tiles. Should the measured matrix go into
   the map transform?
2. **Two names for a tile's time.** Layout 1 reads `acquired` (`runs.py:174`). Layout 2 reads `acquired_s` (`runs.py:195`).
   `Tile` uses `acquired_s`.
3. **Two names for tile positions.** The map store writes `tile_positions_px` (`map_tools.py:262`). `stitch_mosaic` writes
   `positions_px` (`image_mosaic.py:1382`).
4. **Two halving rules.** For maps with no `level_shapes`, `level_shape_of` rounds rows up and columns down
   (`pyramid.py:125-127`). The current `level_shapes` rounds both down (`image_mosaic.py:1062-1068`). Which maps were
   written by the first rule is not known.
5. **Highlights after a rescore.** After `update_highlights`, the store's current group is `highlights_<8 hex>`
   (`poi.py:336`, `poi.py:341`). Angstrom still reads the literal `highlights` group (`tiled.ts:664-666`), so it shows the
   replaced points. The `list_pois` tool reads the current group; `test_analysis_map_live_tiled.py` checks this against a
   local Tiled server. A client that reads points through `list_pois` (section 7.3) does not have this problem.
6. **Stored null stderr.** A map stored before the fix to `estimate_scan_linear` has `scan_linear.stderr` =
   `[[null, null], [null, null]]` (inf through `finite_or_none`). The schema refuses that. Should old maps be accepted?
7. **The meaning of `stage_reported_m` columns 2 to 4** in layout 2 is not stated by `run_file.py`. The map store reads only x and y.
8. **Mapper runs in Tiled.** The map store reads a layout 2 run from Tiled through node metadata and child arrays
   (`store.py:146-152`). This page does not show how a Mapper run file is served by Tiled. That was not verified.
9. **Schema ownership and strictness.** The root schema refuses fields that the current writer does not write. Old maps
   with the inline `pois`/`poi_layer` fields fail it. Should the schema have a separate "accepted on read" form?

## 7. View descriptors (analysis server tools)

These are tool replies, not stored fields. They let a client draw a map, its points and an image without the
conventions in sections 2 to 3: the level node names, the point store key, the grid formula and the highlights group.
The JSON shapes are `$defs` in the schema; the tests validate the replies against them
(`tests/analysis_server/test_analysis_map_views.py` on local HDF5 maps, `test_analysis_map_live_tiled.py` on a local
Tiled server).

A stitched map record no longer carries `poi_kinds` (commit `6c9d735` wrote it). A record is written once, so it
cannot list kinds added later, and the schema refused the field. `map_info` gives the kinds instead (7.2).

### 7.1 `list_maps(run=None, limit=20)`

`run` is now optional. Without it, the tool lists every stitched map: those in Tiled, and the files named `*_map*.h5`
under the server's local roots whose `metadata` has `kind == "stitched_map"`. Each entry of `maps` is
`$defs/map_listing_entry`: the earlier `map`, `id`, `stitched_at`, `inputs_sha256` and `tiles`, plus `run` and the
fields of `$defs/map_view`:

| Field | Meaning |
| :-- | :-- |
| `title` | `<run stem>, stitched <YYYY-mm-dd HH:MM:SS> UTC`. Without a run, the map's own name. Without `stitched_at`, no time. |
| `canvas_shape`, `pixel_size_m`, `origin_xy_m`, `axes` | From the record (2.2). This is the transform of decision D6, option A. |
| `stage_box_m` | `[x_min, x_max, y_min, y_max]`, from pixel (-0.5, -0.5) to pixel (rows - 0.5, cols - 0.5) of level 0 (2.5). |
| `local` | True for a map file, false for a Tiled key. |
| `pyramid` | One `$defs/level_descriptor` per level, read from the stored array, not from the record. |

A level descriptor:

| Field | Meaning |
| :-- | :-- |
| `level` | k |
| `key` | The Tiled key `<map>/level_<k>`, or `<file>#level_<k>` for a local map (the `path#dataset` form the dataset resolver reads). |
| `shape` | `[rows, cols]` of the stored array. |
| `dtype` | numpy's dtype string with byte order, for example `<f4`. |
| `chunks` | Chunk sizes along each axis, as Tiled's array structure gives them. A local dataset's chunk shape is expanded to the same form. |
| `downsample` | `2**k`, the factor `pixel_to_stage` uses. |
| `pixel_size_m`, `origin_xy_m` | The level's own transform: pixel (r, c) of level k is centred at `(origin_xy_m[0] + c * pixel_size_m, origin_xy_m[1] + r * pixel_size_m)`. This is the formula of 2.5 worked out per level: `pixel_size_m = 2**k * p`, `origin_xy_m = origin + (2**k - 1) / 2 * p`. |

### 7.2 `map_info(map)`

The earlier fields stay (`levels` is still the number of levels). It adds the `$defs/map_view` fields of 7.1 and
`poi_kinds`: the kinds of point on the map, sorted, each `$defs/poi_kind` `{kind, label, in_cells}`.

- `label` is the server's label for a kind it writes itself: `eds_point` "EDS point spectrum" (`eds_pois`), `candidate`
  "Scored region" (`score_map`), `note` "Note" (the default kind of `add_poi`). A kind that a caller named itself (for
  example `revisit`) has the label null.
- `in_cells` is true for the kinds kept in spatial cells (`CELL_KINDS`). Only those can come back as counts (7.3).

### 7.3 `list_pois(map, kind=None, box_m=None, limit=200, counts_above=None)`

Without `counts_above` the reply is as before. Points always come from the run's point store and its current
highlights group (`highlights` or `highlights_<8 hex>`), so a client never walks the store.

With `counts_above` = n, if more than n points kept in cells are in the cells the box touches (the whole map without a
box), those points are not read. The reply then has:

| Field | Meaning |
| :-- | :-- |
| `cells` | `[{cell: "<i>_<j>", count, box_m}]`, `$defs/poi_cell_count`, for each touched cell that holds points. `count` is the stored count of the whole cell, so a cell partly inside the box counts all its points. `box_m` is the cell's stage box. |
| `cell_kinds` | The kinds the counts are of (`CELL_KINDS`). A count is of every one of these kinds. |
| `pois` | The other points that match: the highlights, filtered by kind and box. |
| `total` | `len(pois)` plus the sum of the counts. |

A query for highlight kinds only (for example `kind="candidate"`) never gives counts. A query for cell kinds only gives
counts and no points.

With a box, only the stored cells that the box touches are read (one listing of the layer, then those cells). Before,
every cell the box touched was asked for, stored or not, which against Tiled took minutes for half a map.

### 7.4 `image_info(dataset)`

New tool. It says which array of a dataset is its image, and where that image sits on the stage. Its reply is
`$defs/image_info`:

| Field | Meaning |
| :-- | :-- |
| `kind` | `"image"` or `"map"` |
| `image` | The array's key. For an acquisition: the key itself when it is an array, else the first array of its `image` group (one that records `pixel_size_m` first), else the first array in it, the rule `frames.image_node` uses. For a map: its level 0 descriptor's key. No detector name is read. |
| `shape`, `dtype`, `chunks` | `shape` is `[rows, cols]` (the last two axes); `dtype` and `chunks` are the array's. |
| `frame` | `{pixel_size_m, origin_xy_m, centre_xy_m, axes}`. Pixel (r, c) is centred at `(origin_xy_m[0] + c * pixel_size_m, origin_xy_m[1] + r * pixel_size_m)`. For an image, `origin_xy_m` is pyTEMlib's `pixel_to_stage(dataset, 0, 0)` and `centre_xy_m` is `get_image_centre_m` (2.5 does not apply). |
| `frame_refused` | With `frame` null: why the image cannot be placed (the refusals of `frame_of`: no position or pixel size, other axes, part of the scan field, tilt). |
| `note` | When the image records no image shift and is placed as if it had none. |
| `sample_id`, `sample_load_id` | When the dataset records them. |

An unknown key gives the `unknown_dataset` error with the newest keys. A container with no array gives the error code
`no_image`.


### 7.5 `read_map_block(map, level, block)`

New read-only tool. It gives the pixels of one block of one pyramid level, for a map in Tiled or in a local HDF5
file, so a client draws every map the same way. A local map is not served by Tiled, so this tool is the only way a
client reads its pixels. For a Tiled map a client can still read the block from Tiled; the two give the same array.

- `block` is `[i, j]`: chunk `i` along rows and chunk `j` along columns of the level descriptor's `chunks` (7.1).
  These are the blocks Tiled's `read_block((i, j))` gives for the same level, and a local level's chunks are expanded
  to the same form. A reply is never larger than one chunk.
- The block covers rows `sum(chunks[0][:i])` to `sum(chunks[0][:i + 1])` and the same for columns.

Its reply is `$defs/map_block`:

| Field | Meaning |
| :-- | :-- |
| `map`, `level`, `block` | As given. |
| `shape` | `[rows, cols]`: `[chunks[0][i], chunks[1][j]]`. |
| `dtype` | The level descriptor's `dtype`, numpy's string with byte order (for example `<u2`). |
| `data` | The block's bytes, in C (row-major) order and the byte order `dtype` names, base64-encoded. |

The whole reply is the tool's structured content (`structuredContent`). Its text content is the same JSON without
`data`: a 512 x 512 block of `<u2` is about 700 000 base64 characters, more than a client that hands tool text to a
model will keep (Angstrom cuts a tool's text at 100 000 characters and saves the rest as a file). A client reads the
pixels from the structured content.

A level outside `0` to `levels - 1` gives the error code `bad_level`. A block that is not two integers inside the
level's chunk grid gives `bad_block`. Both errors name the valid range. An unknown map is refused the same way
`map_info` refuses it.

Tests: `test_analysis_map_views.py` reads blocks of a local map with 32-pixel chunks and compares them with h5py,
and checks that the text content has no `data`;
`test_analysis_map_live_tiled.py` compares the first and last block of each level with Tiled's own `read_block`.
