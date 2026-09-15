# WRE V1 technical 3D preview

This folder is a **temporary user-facing preview** of the first working sparse-3D baseline validated by milestone M1 / L3.

It lives on the dedicated `demo-v1-preview` branch so the canonical development history on `main` is not polluted by generated demonstration assets.

## What is in this folder

- `scene.json` — the exact L3.7 synthetic scene specification (`seed = 7301`).
- `render_inputs.py` — reproduces the exact binary P5 PGM inputs used by the L3.7 renderer and verifies their identities.
- `input_images/` — the 12 exact deterministic PGM views after materialization.
- `input_previews/` — PNG copies of the same pixels for easy viewing in a browser.
- `manifest.json` — SHA-256 and byte length for every PGM input.
- `reference/reference_scene_with_cameras.ply` — **synthetic reference geometry**, not a solver result. White points are known scene points; red points are known camera centers.
- `run_demo.py` — reruns the real WRE L3 pipeline against the committed inputs with exact `pycolmap==4.2.0`.
- `generated/reconstruction_wre_ascii.ply` — actual sparse geometry reconstructed by WRE/COLMAP and re-exported from the WRE imported estimate.
- `generated/summary.json` — observed reconstruction metrics and solver identity.
- `generated/colmap_text_model/` — actual native COLMAP sparse model in human-readable text form.

## What the current version already does

The real pipeline exercised here is:

```text
12 image observations
  -> WRE content/provenance identity
  -> COLMAP 4.2.0 SIFT extraction
  -> raw pair matching
  -> two-view geometric verification
  -> incremental sparse reconstruction
  -> WRE solver-independent estimated-geometry import
  -> PLY export for human inspection
```

That is a real working **sparse 3D reconstruction baseline**.

It is not yet the finished MONDE/WRE world engine:

- scale remains `UNRESOLVED`;
- coordinates are local, not geographic/world coordinates;
- accepted `SpatialFragment` lifecycle begins in L6;
- fragment merge/world graph/uncertainty are later lots;
- dense reconstruction/mesh/texture are L14;
- historical/4D world reconstruction is L15.

So treat this as **V1 technical sparse reconstruction**, not as the final product.

## Reproduce the exact input images

From the repository root:

```bash
uv sync --frozen --group dev
.venv/bin/python demo/v1_preview/render_inputs.py
```

The renderer checks the exact SHA-256 + byte length in `manifest.json`. If any generated PGM differs from the expected L3.7 deterministic input, it fails.

## Run the real reconstruction locally

Install the exact reference solver environment and run:

```bash
uv pip install --python .venv/bin/python "numpy==2.5.3" "pycolmap==4.2.0"
.venv/bin/python demo/v1_preview/run_demo.py --force
```

Expected outputs:

```text
demo/v1_preview/generated/
├── reconstruction_wre_ascii.ply
├── reconstruction_colmap.ply
├── summary.json
├── colmap_text_model/
└── work/
```

Only the compact human-inspection outputs are committed to this preview branch. Databases/native transient work files remain reproducible rather than versioned.

## Which 3D file should I open?

Open:

```text
demo/v1_preview/generated/reconstruction_wre_ascii.ply
```

It contains:

- white vertices = 3D points **actually reconstructed** and imported into WRE;
- red vertices = reconstructed camera centers;
- local coordinates;
- unresolved metric/world scale.

The separate file:

```text
demo/v1_preview/reference/reference_scene_with_cameras.ply
```

is the known synthetic reference scene. It is useful for comparison but is **not** the reconstructed answer.

## How to visualize it

### CloudCompare

1. Open CloudCompare.
2. `File -> Open`.
3. Select `reconstruction_wre_ascii.ply`.
4. Rotate/zoom with the mouse.

### MeshLab

1. Open MeshLab.
2. `File -> Import Mesh`.
3. Select the PLY.

### COLMAP GUI

For the full sparse camera/point model:

1. run `run_demo.py`;
2. open COLMAP;
3. `File -> Import model`;
4. select `demo/v1_preview/generated/colmap_text_model/`.

## Images

The easiest files to inspect are:

```text
demo/v1_preview/input_previews/view-000.png
...
demo/v1_preview/input_previews/view-011.png
```

The matching `input_images/*.pgm` files are the exact P5 raster inputs used by the reconstruction demo.

## Video and 4D status

WRE already supports deterministic **video ingestion + keyframe extraction** from L2, but the validated L3 reconstruction fixture uses still synthetic views.

Video trajectory/rig/VIO work is L11.

The historical/4D engine is **not implemented yet**; it is L15. This preview therefore contains no fake “4D result”.
