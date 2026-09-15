from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import zlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from wre.domain.estimated_geometry import LocalScaleStatus
from wre.domain.observations import ImageObservation, MediaAssetRef, ObservationId, SourceId, SourceRef
from wre.domain.runs import ProducerRef, ReconstructionRun, ReconstructionRunId
from wre.ingestion.hashing import hash_file_content
from wre.reconstruction import (
    COLMAP_IMPORTER_VERSION,
    ColmapFeatureExtractionConfig,
    ColmapFeatureExtractionRequest,
    ColmapFeatureInput,
    ColmapGeometricVerificationConfig,
    ColmapGeometricVerificationRequest,
    ColmapIncrementalReconstructionConfig,
    ColmapIncrementalReconstructionRequest,
    ColmapPairMatchingConfig,
    ColmapPairMatchingRequest,
    ColmapReconstructionImportConfig,
    ColmapReconstructionImportRequest,
    ColmapReconstructionInput,
    extract_colmap_features,
    import_colmap_reconstruction,
    match_colmap_pairs,
    reconstruct_colmap_incrementally,
    verify_colmap_geometry,
)

HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE / "input_images"
RESOLVED_SOURCES = HERE / "resolved_sources.json"


def _run(*, value: str, implementation: str, version: str, revision: str | None,
         observation_ids: tuple[ObservationId, ...], configuration_sha256: Any,
         minute: int) -> ReconstructionRun:
    return ReconstructionRun(
        run_id=ReconstructionRunId(value),
        producer=ProducerRef(implementation=implementation, version=version, revision=revision),
        input_observation_ids=observation_ids,
        started_at=datetime(2026, 9, 15, 16, 0, tzinfo=UTC) + timedelta(minutes=minute),
        configuration_sha256=configuration_sha256,
    )


def _load_sources() -> tuple[dict[str, object], ...]:
    raw = json.loads(RESOLVED_SOURCES.read_text(encoding="utf-8"))
    files = raw.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("resolved_sources.json does not contain a non-empty files list")
    items = tuple(item for item in files if isinstance(item, dict))
    if len(items) != len(files):
        raise RuntimeError("resolved_sources.json contains an invalid file entry")
    return items


def _observations(sources: tuple[dict[str, object], ...]) -> tuple[tuple[ImageObservation, Path], ...]:
    rows: list[tuple[ImageObservation, Path]] = []
    for index, source in enumerate(sources):
        local_name = source.get("local_name")
        source_page = source.get("source_page")
        if not isinstance(local_name, str) or not isinstance(source_page, str):
            raise RuntimeError("resolved source is missing local_name/source_page")
        path = INPUT_DIR / local_name
        content = hash_file_content(path)
        expected_sha = source.get("downloaded_sha256")
        expected_size = source.get("downloaded_byte_length")
        if content.sha256.value != expected_sha or content.byte_length != expected_size:
            raise RuntimeError(f"committed real-world input changed: {local_name}")
        observation = ImageObservation(
            observation_id=ObservationId(f"obs:demo:notre-dame:{index:03d}"),
            asset=MediaAssetRef(
                uri=path.resolve().as_uri(),
                sha256=content.sha256,
                byte_length=content.byte_length,
                mime_type="image/jpeg",
            ),
            source=SourceRef(
                source_id=SourceId("demo:wikimedia-commons:notre-dame"),
                locator=source_page,
            ),
            received_at=datetime(2026, 9, 15, 16, 0, tzinfo=UTC),
        )
        rows.append((observation, path))
    return tuple(rows)


def _camera_center(rotation_matrix: Any, translation_xyz: Any) -> tuple[float, float, float]:
    return tuple(
        -sum(rotation_matrix[row][col] * translation_xyz[row] for row in range(3))
        for col in range(3)
    )


def _write_ascii_ply(path: Path, estimate: Any) -> None:
    vertices: list[tuple[float, float, float, int, int, int]] = []
    for point in estimate.points3d:
        x, y, z = point.position_xyz
        vertices.append((x, y, z, 235, 235, 235))
    for pose in estimate.camera_poses:
        x, y, z = _camera_center(pose.rotation_matrix, pose.translation_xyz)
        vertices.append((x, y, z, 255, 64, 64))
    lines = [
        "ply",
        "format ascii 1.0",
        "comment WRE real-world sparse reconstruction",
        "comment white points = reconstructed 3D points; red points = reconstructed camera centers",
        "comment local scale is UNRESOLVED; coordinates are not geographic coordinates",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    lines.extend(f"{x:.9f} {y:.9f} {z:.9f} {r} {g} {b}" for x, y, z, r, g, b in vertices)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def _write_png(path: Path, width: int, height: int, pixels: bytearray) -> None:
    rows = bytearray()
    stride = width * 3
    for y in range(height):
        rows.append(0)
        rows.extend(pixels[y * stride:(y + 1) * stride])
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += _png_chunk(b"IDAT", zlib.compress(bytes(rows), 9))
    payload += _png_chunk(b"IEND", b"")
    path.write_bytes(payload)


def _normalized_geometry(estimate: Any) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray([point.position_xyz for point in estimate.points3d], dtype=np.float64)
    cameras = np.asarray(
        [_camera_center(pose.rotation_matrix, pose.translation_xyz) for pose in estimate.camera_poses],
        dtype=np.float64,
    )
    if len(points) < 3:
        raise RuntimeError("not enough reconstructed points for visualization")
    center = np.median(points, axis=0)
    centered = points - center
    covariance = np.cov(centered, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    basis = vectors[:, np.argsort(values)[::-1]]
    points_pca = centered @ basis
    cameras_pca = (cameras - center) @ basis
    scale = float(np.percentile(np.linalg.norm(points_pca, axis=1), 95))
    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError("invalid visualization scale")
    return points_pca / scale, cameras_pca / scale


def _render_view(path: Path, points: np.ndarray, cameras: np.ndarray, matrix: np.ndarray) -> None:
    width, height, margin = 1200, 900, 60
    rotated = points @ matrix.T
    rotated_cameras = cameras @ matrix.T
    x = rotated[:, 0]
    y = rotated[:, 1]
    keep = (
        (x >= np.percentile(x, 1))
        & (x <= np.percentile(x, 99))
        & (y >= np.percentile(y, 1))
        & (y <= np.percentile(y, 99))
    )
    visible = rotated[keep]
    x_min, y_min = np.min(visible[:, :2], axis=0)
    x_max, y_max = np.max(visible[:, :2], axis=0)
    span_x = max(float(x_max - x_min), 1e-9)
    span_y = max(float(y_max - y_min), 1e-9)
    scale = min((width - 2 * margin) / span_x, (height - 2 * margin) / span_y)
    pixels = bytearray([18, 20, 24]) * (width * height)

    def draw(px: int, py: int, color: tuple[int, int, int], radius: int) -> None:
        for dy in range(-radius, radius + 1):
            yy = py + dy
            if yy < 0 or yy >= height:
                continue
            for dx in range(-radius, radius + 1):
                xx = px + dx
                if xx < 0 or xx >= width:
                    continue
                offset = (yy * width + xx) * 3
                pixels[offset:offset + 3] = bytes(color)

    depth = visible[:, 2]
    d_min, d_max = float(np.min(depth)), float(np.max(depth))
    d_span = max(d_max - d_min, 1e-9)
    for vx, vy, vz in visible:
        px = int(margin + (float(vx) - x_min) * scale)
        py = int(height - margin - (float(vy) - y_min) * scale)
        shade = int(130 + 110 * (float(vz) - d_min) / d_span)
        draw(px, py, (shade, shade, shade), 1)
    for vx, vy, _ in rotated_cameras:
        px = int(margin + (float(vx) - x_min) * scale)
        py = int(height - margin - (float(vy) - y_min) * scale)
        draw(px, py, (255, 64, 64), 4)
    _write_png(path, width, height, pixels)


def _write_screenshots(directory: Path, estimate: Any) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    points, cameras = _normalized_geometry(estimate)
    identity = np.eye(3)
    side = np.asarray(((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)))
    top = np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)))
    angle_y = math.radians(35)
    angle_x = math.radians(-20)
    ry = np.asarray(((math.cos(angle_y), 0.0, math.sin(angle_y)), (0.0, 1.0, 0.0), (-math.sin(angle_y), 0.0, math.cos(angle_y))))
    rx = np.asarray(((1.0, 0.0, 0.0), (0.0, math.cos(angle_x), -math.sin(angle_x)), (0.0, math.sin(angle_x), math.cos(angle_x))))
    for name, matrix in (("front", identity), ("side", side), ("top", top), ("oblique", rx @ ry)):
        _render_view(directory / f"{name}.png", points, cameras, matrix)


def _write_viewer(path: Path, estimate: Any) -> None:
    points, cameras = _normalized_geometry(estimate)
    vertices: list[list[float | int]] = []
    for x, y, z in points:
        vertices.append([float(x), float(y), float(z), 0.92, 0.92, 0.92])
    for x, y, z in cameras:
        vertices.append([float(x), float(y), float(z), 1.0, 0.18, 0.18])
    data = json.dumps(vertices, separators=(",", ":"))
    html = f"""<!doctype html>
<meta charset=\"utf-8\"><title>WRE Notre-Dame sparse 3D preview</title>
<style>html,body{{margin:0;height:100%;background:#111;color:#eee;font:14px system-ui}}#c{{width:100%;height:100%;display:block}}#help{{position:fixed;left:12px;top:12px;background:#000a;padding:10px 12px;border-radius:8px;max-width:420px}}code{{color:#fff}}</style>
<canvas id=\"c\"></canvas><div id=\"help\"><b>WRE real-world sparse reconstruction — Notre-Dame</b><br>Drag: rotate · Shift+drag: pan · Wheel/W/S: zoom · A/D/Q/E: pan · R: reset<br>White = reconstructed 3D points · Red = reconstructed cameras<br><small>Local scale unresolved; this is sparse geometry, not a textured mesh.</small></div>
<script>
const V={data}; const canvas=document.getElementById('c'); const gl=canvas.getContext('webgl2');
const vs=`#version 300 es\nin vec3 p; in vec3 c; uniform float yaw,pitch,zoom; uniform vec2 pan; out vec3 col; void main(){{float cy=cos(yaw),sy=sin(yaw),cx=cos(pitch),sx=sin(pitch); mat3 ry=mat3(cy,0.,-sy,0.,1.,0.,sy,0.,cy); mat3 rx=mat3(1.,0.,0.,0.,cx,sx,0.,-sx,cx); vec3 q=rx*ry*p; float z=zoom/(2.7-q.z); gl_Position=vec4(q.x*z+pan.x,q.y*z+pan.y,0.,1.); gl_PointSize=c.r>.98&&c.g<.5?8.:2.6; col=c;}}`;
const fs=`#version 300 es\nprecision mediump float; in vec3 col; out vec4 outColor; void main(){{outColor=vec4(col,1.);}}`;
function sh(t,s){{let x=gl.createShader(t);gl.shaderSource(x,s);gl.compileShader(x);return x}} let pr=gl.createProgram();gl.attachShader(pr,sh(gl.VERTEX_SHADER,vs));gl.attachShader(pr,sh(gl.FRAGMENT_SHADER,fs));gl.linkProgram(pr);gl.useProgram(pr);
const flat=new Float32Array(V.flat()); const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,flat,gl.STATIC_DRAW); const stride=24; for(const [n,o] of [['p',0],['c',12]]){{let l=gl.getAttribLocation(pr,n);gl.enableVertexAttribArray(l);gl.vertexAttribPointer(l,3,gl.FLOAT,false,stride,o)}}
let yaw=.45,pitch=-.2,zoom=1.3,pan=[0,0],drag=false,last=[0,0],shift=false;
function resize(){{const d=devicePixelRatio||1;canvas.width=innerWidth*d;canvas.height=innerHeight*d;gl.viewport(0,0,canvas.width,canvas.height)}} addEventListener('resize',resize);resize();
canvas.onpointerdown=e=>{{drag=true;last=[e.clientX,e.clientY];shift=e.shiftKey;canvas.setPointerCapture(e.pointerId)}};canvas.onpointerup=()=>drag=false;canvas.onpointermove=e=>{{if(!drag)return;let dx=e.clientX-last[0],dy=e.clientY-last[1];last=[e.clientX,e.clientY];if(shift){{pan[0]+=dx/500;pan[1]-=dy/500}}else{{yaw+=dx/180;pitch+=dy/180}}}};canvas.onwheel=e=>{{e.preventDefault();zoom*=Math.exp(-e.deltaY*.001)}},{{passive:false}};
addEventListener('keydown',e=>{{if(e.key==='w'||e.key==='W')zoom*=1.08;if(e.key==='s'||e.key==='S')zoom/=1.08;if(e.key==='a'||e.key==='A')pan[0]-=.04;if(e.key==='d'||e.key==='D')pan[0]+=.04;if(e.key==='q'||e.key==='Q')pan[1]+=.04;if(e.key==='e'||e.key==='E')pan[1]-=.04;if(e.key==='r'||e.key==='R'){{yaw=.45;pitch=-.2;zoom=1.3;pan=[0,0]}}}});
function frame(){{gl.clearColor(.067,.067,.075,1);gl.clear(gl.COLOR_BUFFER_BIT);for(const [n,v] of [['yaw',yaw],['pitch',pitch],['zoom',zoom]])gl.uniform1f(gl.getUniformLocation(pr,n),v);gl.uniform2f(gl.getUniformLocation(pr,'pan'),pan[0],pan[1]);gl.drawArrays(gl.POINTS,0,V.length);requestAnimationFrame(frame)}}frame();
</script>"""
    path.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the WRE sparse pipeline on real Notre-Dame photographs.")
    parser.add_argument("--output", type=Path, default=HERE / "generated")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.force:
            raise SystemExit(f"{output} already exists; use --force")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    try:
        import pycolmap
    except ImportError as exc:
        raise SystemExit("Install pycolmap==4.2.0 and numpy==2.5.3 first") from exc
    if str(pycolmap.__version__) != "4.2.0":
        raise SystemExit(f"expected pycolmap 4.2.0, got {pycolmap.__version__}")

    sources = _load_sources()
    rows = _observations(sources)
    observations = tuple(item[0] for item in rows)
    image_paths = tuple(item[1] for item in rows)
    observation_ids = tuple(item.observation_id for item in observations)
    revision = str(pycolmap.COLMAP_build)
    work = output / "work"
    work.mkdir()

    feature_config = ColmapFeatureExtractionConfig(max_image_size=2000)
    features = extract_colmap_features(
        ColmapFeatureExtractionRequest(
            run=_run(value="run:demo:notre-dame:features", implementation="pycolmap.extract_features", version="4.2.0", revision=revision, observation_ids=observation_ids, configuration_sha256=feature_config.sha256, minute=0),
            inputs=tuple(ColmapFeatureInput(observation=o, source_path=p) for o, p in rows),
            database_path=work / "features.db",
            config=feature_config,
        ), module=pycolmap,
    )
    matching_config = ColmapPairMatchingConfig()
    matching = match_colmap_pairs(
        ColmapPairMatchingRequest(
            run=_run(value="run:demo:notre-dame:matching", implementation="pycolmap.match_exhaustive", version="4.2.0", revision=revision, observation_ids=observation_ids, configuration_sha256=matching_config.sha256, minute=1),
            features=features, database_path=work / "matches.db", config=matching_config,
        ), module=pycolmap,
    )
    verification_config = ColmapGeometricVerificationConfig()
    verification = verify_colmap_geometry(
        ColmapGeometricVerificationRequest(
            run=_run(value="run:demo:notre-dame:verification", implementation="pycolmap.geometric_verification", version="4.2.0", revision=revision, observation_ids=observation_ids, configuration_sha256=verification_config.sha256, minute=2),
            matching=matching, database_path=work / "verified.db", config=verification_config,
        ), module=pycolmap,
    )
    names = {item.observation_id: item.image_name for item in features.images}
    reconstruction_inputs = tuple(
        ColmapReconstructionInput(observation=o, source_path=p, image_name=names[o.observation_id])
        for o, p in rows
    )
    reconstruction_config = ColmapIncrementalReconstructionConfig()
    reconstruction = reconstruct_colmap_incrementally(
        ColmapIncrementalReconstructionRequest(
            run=_run(value="run:demo:notre-dame:reconstruction", implementation="pycolmap.incremental_mapping", version="4.2.0", revision=revision, observation_ids=observation_ids, configuration_sha256=reconstruction_config.sha256, minute=3),
            verification=verification, inputs=reconstruction_inputs,
            output_path=work / "sparse", config=reconstruction_config,
        ), module=pycolmap,
    )
    if not reconstruction.models:
        raise SystemExit("No sparse reconstruction model was produced from the real photographs")
    import_config = ColmapReconstructionImportConfig()
    imported = import_colmap_reconstruction(
        ColmapReconstructionImportRequest(
            run=_run(value="run:demo:notre-dame:import", implementation="wre.colmap_reconstruction_importer", version=COLMAP_IMPORTER_VERSION, revision=None, observation_ids=observation_ids, configuration_sha256=import_config.sha256, minute=4),
            reconstruction=reconstruction, features=features, config=import_config,
        ), module=pycolmap,
    )

    largest = max(reconstruction.models, key=lambda m: (m.num_registered_images, m.num_points3d, -m.model_index))
    imported_largest = next(m for m in imported.models if m.source_model_index == largest.model_index)
    estimate = imported_largest.estimate
    if estimate.observation_count < 6 or estimate.point_count < 100:
        raise SystemExit(
            f"Real-world result is too weak for a useful demo: {estimate.observation_count} registered images, {estimate.point_count} points"
        )

    model_path = reconstruction.output_path / largest.relative_path
    native = pycolmap.Reconstruction(str(model_path))
    native.export_PLY(str(output / "reconstruction_colmap.ply"))
    text_dir = output / "colmap_text_model"
    text_dir.mkdir()
    native.write_text(str(text_dir))
    _write_ascii_ply(output / "reconstruction_wre_ascii.ply", estimate)
    _write_screenshots(output / "screenshots", estimate)
    _write_viewer(output / "viewer.html", estimate)

    registered_names = sorted(
        str(native.image(image_id).name) for image_id in native.reg_image_ids()
    )
    summary = {
        "dataset": "Notre-Dame de Paris west facade — Wikimedia Commons real photographs",
        "input_image_count": len(observations),
        "pycolmap_version": str(pycolmap.__version__),
        "colmap_version": str(pycolmap.COLMAP_version),
        "colmap_build": str(pycolmap.COLMAP_build),
        "scale_status": "unresolved",
        "metrics": {
            "attempted_pair_count": matching.attempted_pair_count,
            "raw_pair_count": len(matching.pairs),
            "verified_pair_count": len(verification.geometries),
            "reconstructed_model_count": reconstruction.model_count,
            "largest_model_index": largest.model_index,
            "largest_registered_image_count": estimate.observation_count,
            "largest_point_count": estimate.point_count,
            "imported_model_count": imported.model_count,
            "unresolved_scale_model_count": sum(m.estimate.scale_status is LocalScaleStatus.UNRESOLVED for m in imported.models),
        },
        "registered_colmap_image_names": registered_names,
        "files": {
            "wre_ascii_ply": "reconstruction_wre_ascii.ply",
            "colmap_ply": "reconstruction_colmap.ply",
            "interactive_viewer": "viewer.html",
            "screenshots": "screenshots/",
            "colmap_text_model": "colmap_text_model/",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
