# WRE real-world preview — Notre-Dame de Paris

This directory is a **real-photograph sparse 3D reconstruction demo** of the west facade of Notre-Dame de Paris.

It is separate from the deterministic synthetic regression fixture. The inputs here are ten real Wikimedia Commons photographs whose source pages, authors, licenses, downloaded hashes, and original-file metadata are recorded in `resolved_sources.json`.

## What the successful run produced

The materialized demo was produced with WRE's real L3.2-L3.6 pipeline and exact `pycolmap==4.2.0` / COLMAP 4.2.0:

- 10 real input photographs;
- 45 attempted exhaustive image pairs;
- 45 raw matched pairs;
- 45 geometrically verified pairs;
- 1 reconstructed sparse model;
- **10 / 10 registered images**;
- **3,039 reconstructed 3D points**;
- 1 model imported into WRE Estimated Geometry;
- local scale remains **UNRESOLVED** (the coordinates are not yet metric/geographic world coordinates).

See `generated/summary.json` for the machine-readable result.

## See the real input photographs

Open `input_images/`.

`resolved_sources.json` maps every `image-XX.jpg` to its Wikimedia Commons source page and records the author, license, license URL, downloaded SHA-256, byte length, and original image metadata.

The source request asks Wikimedia for a 1024 px derivative. Wikimedia's image API returned its standard cached 1280 px derivatives for this set; the exact downloaded URLs and hashes are recorded rather than inferred.

## Fastest way to see the reconstruction

### Option A — standalone browser viewer

Download this branch (or clone it), then open:

`demo/real_world_preview/generated/viewer.html`

The viewer is self-contained: the reconstructed point data is embedded in the HTML and it does not need a Python server or an external JavaScript library.

Controls:

- drag: rotate;
- Shift + drag: pan;
- mouse wheel or W/S: zoom;
- A/D/Q/E: pan;
- R: reset view.

White points are reconstructed 3D points. Red points are reconstructed camera centers.

### Option B — CloudCompare

Open:

`generated/reconstruction_wre_ascii.ply`

Steps:

1. Install CloudCompare.
2. `File` -> `Open`.
3. Select `reconstruction_wre_ascii.ply`.
4. Use the mouse to orbit, pan, and zoom around the point cloud.
5. Increase point size if the cloud looks too sparse on a high-DPI display.

The WRE PLY contains 3,039 reconstructed scene points plus the reconstructed camera centers. Scene points are light gray/white; camera centers are red.

### Option C — MeshLab

1. Open MeshLab.
2. `File` -> `Import Mesh`.
3. Select `generated/reconstruction_wre_ascii.ply`.
4. Orbit/zoom around the point cloud.

### Option D — COLMAP GUI

For the solver-native reconstruction, use the files in:

`generated/colmap_text_model/`

or open `generated/reconstruction_colmap.ply` in a PLY-capable viewer.

## Static views

Four pre-rendered views are committed for quick inspection:

- `generated/screenshots/front.png`
- `generated/screenshots/side.png`
- `generated/screenshots/top.png`
- `generated/screenshots/oblique.png`

These are projections of the **actual reconstructed sparse geometry**, not illustrations of the cathedral.

## Important limitation

This is the current WRE **sparse 3D** result, not yet a dense textured model. It represents reconstructed feature points and estimated camera positions. Dense point clouds, mesh generation, and texturing belong to later roadmap work (L14).

Likewise, the current local model has unresolved absolute scale and no world/geographic anchoring yet. WRE deliberately does not invent those quantities.

## Re-run the demo

From the repository root on this branch:

```bash
uv sync --frozen --group dev
uv pip install --python .venv/bin/python "numpy==2.5.3" "pycolmap==4.2.0"
.venv/bin/python demo/real_world_preview/download_inputs.py
.venv/bin/python demo/real_world_preview/run_demo.py --force
```

The downloader refuses unreviewed license families and records the resolved Commons metadata. The reconstruction script verifies downloaded hashes before treating the files as WRE observations.

## Why only ten images?

The current L3 baseline intentionally uses deterministic CPU/single-thread exhaustive matching. Ten images imply 45 pairs and complete in a practical GitHub Actions window. The earlier 18-image experiment implied 153 pairs and hit the workflow timeout during matching. Candidate retrieval and adaptive routing are being developed in L4 specifically to avoid exhaustive all-pairs work as collections grow.
