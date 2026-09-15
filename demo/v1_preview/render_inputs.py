from __future__ import annotations

import hashlib
import json
import random
import struct
import zlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCENE_PATH = HERE / "scene.json"
MANIFEST_PATH = HERE / "manifest.json"
INPUT_DIR = HERE / "input_images"
PREVIEW_DIR = HERE / "input_previews"


def _patch(seed: int, size: int) -> tuple[int, ...]:
    rng = random.Random(seed)
    values: list[int] = []
    for y in range(size):
        for x in range(size):
            if x in (0, size - 1) or y in (0, size - 1):
                values.append(224 if (x + y + seed) % 2 else 24)
            else:
                values.append(240 if rng.getrandbits(1) else 16)
    return tuple(values)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return (
        struct.pack(">I", len(payload))
        + body
        + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    )


def _grayscale_png(width: int, height: int, pixels: bytes) -> bytes:
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        start = y * width
        rows.extend(pixels[start : start + width])
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(
                b"IHDR",
                struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0),
            ),
            _png_chunk(b"IDAT", zlib.compress(bytes(rows), level=9)),
            _png_chunk(b"IEND", b""),
        )
    )


def _load_scene() -> dict[str, object]:
    raw = json.loads(SCENE_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("scene.json must contain an object")
    return raw


def render() -> None:
    scene = _load_scene()
    width = int(scene["width"])
    height = int(scene["height"])
    focal = float(scene["focal_length_px"])
    patch_size = int(scene["patch_size"])
    background = int(scene["background_value"])
    scene_seed = int(scene["seed"])
    grid_u = [int(item) for item in scene["grid_u"]]
    grid_v = [int(item) for item in scene["grid_v"]]
    depths = [float(item) for item in scene["depths"]]
    camera_x = [float(item) for item in scene["camera_centers_x"]]
    camera_y = [float(item) for item in scene["camera_centers_y"]]

    points: list[tuple[int, int, float, tuple[int, ...]]] = []
    point_index = 0
    for v in grid_v:
        for u in grid_u:
            depth = depths[point_index % len(depths)]
            points.append(
                (
                    u,
                    v,
                    depth,
                    _patch(scene_seed + point_index * 104_729, patch_size),
                )
            )
            point_index += 1

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    half = patch_size // 2
    generated: dict[str, tuple[str, int]] = {}

    for camera_index, (center_x, center_y) in enumerate(
        zip(camera_x, camera_y, strict=True)
    ):
        pixels = bytearray([background]) * (width * height)
        for base_u, base_v, depth, patch in points:
            center_u = round(base_u - focal * center_x / depth)
            center_v = round(base_v - focal * center_y / depth)
            for patch_y in range(patch_size):
                image_y = center_v + patch_y - half
                if image_y < 0 or image_y >= height:
                    continue
                for patch_x in range(patch_size):
                    image_x = center_u + patch_x - half
                    if image_x < 0 or image_x >= width:
                        continue
                    pixels[image_y * width + image_x] = patch[
                        patch_y * patch_size + patch_x
                    ]

        name = f"view-{camera_index:03d}.pgm"
        pgm = f"P5\n{width} {height}\n255\n".encode() + bytes(pixels)
        (INPUT_DIR / name).write_bytes(pgm)
        (PREVIEW_DIR / name.replace(".pgm", ".png")).write_bytes(
            _grayscale_png(width, height, bytes(pixels))
        )
        generated[f"input_images/{name}"] = (
            hashlib.sha256(pgm).hexdigest(),
            len(pgm),
        )

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = {
        item["path"]: (item["sha256"], item["byte_length"])
        for item in manifest["images"]
    }
    if generated != expected:
        raise RuntimeError(
            "rendered image identities do not match the committed L3.7 demo manifest"
        )

    print(f"Rendered and verified {len(generated)} exact PGM inputs.")
    print(f"PNG convenience previews: {PREVIEW_DIR}")


if __name__ == "__main__":
    render()
