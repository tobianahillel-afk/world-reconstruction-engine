from __future__ import annotations

import hashlib
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES_PATH = HERE / "sources.json"
INPUT_DIR = HERE / "input_images"
RESOLVED_PATH = HERE / "resolved_sources.json"
API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "world-reconstruction-engine-demo/1.0 (GitHub public demo)"


def _plain(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html.unescape(value))).strip()


def _metadata_value(metadata: dict[str, object], key: str) -> str:
    raw = metadata.get(key)
    if not isinstance(raw, dict):
        return ""
    value = raw.get("value")
    return _plain(value if isinstance(value, str) else "")


def _open_with_backoff(request: urllib.request.Request, *, timeout: int):
    for attempt in range(7):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 502, 503, 504} or attempt == 6:
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else min(2 ** attempt, 30)
            print(f"Wikimedia HTTP {exc.code}; retrying in {delay:.0f}s")
            time.sleep(delay)
    raise AssertionError("unreachable")


def _query_file(title: str, width: int) -> dict[str, object]:
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "titles": f"File:{title}",
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|sha1|size|mime",
            "iiurlwidth": str(width),
        }
    )
    request = urllib.request.Request(f"{API}?{params}", headers={"User-Agent": USER_AGENT})
    with _open_with_backoff(request, timeout=60) as response:
        payload = json.load(response)
    pages = payload.get("query", {}).get("pages", [])
    if len(pages) != 1 or "missing" in pages[0]:
        raise RuntimeError(f"Wikimedia Commons file not found: {title}")
    info = pages[0].get("imageinfo", [])
    if len(info) != 1:
        raise RuntimeError(f"missing imageinfo for Commons file: {title}")
    return info[0]


def _download(url: str, path: Path) -> tuple[str, int]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    digest = hashlib.sha256()
    size = 0
    with _open_with_backoff(request, timeout=120) as response, path.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def main() -> int:
    spec = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    titles = spec["files"]
    width = int(spec["download_policy"]["thumbnail_width_px"])
    if not isinstance(titles, list) or not titles:
        raise RuntimeError("sources.json must contain a non-empty files list")

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    for existing in INPUT_DIR.glob("*"):
        if existing.is_file():
            existing.unlink()

    resolved: list[dict[str, object]] = []
    for index, title_raw in enumerate(titles):
        if not isinstance(title_raw, str) or not title_raw.strip():
            raise RuntimeError("Commons file titles must be non-empty strings")
        title = title_raw.strip()
        info = _query_file(title, width)
        metadata = info.get("extmetadata", {})
        if not isinstance(metadata, dict):
            raise RuntimeError(f"missing extmetadata for {title}")
        license_short = _metadata_value(metadata, "LicenseShortName")
        if not (
            license_short.startswith("CC BY")
            or license_short == "CC0"
            or license_short.lower().startswith("public domain")
        ):
            raise RuntimeError(
                f"refusing source with unreviewed license {license_short!r}: {title}"
            )
        artist = _metadata_value(metadata, "Artist")
        credit = _metadata_value(metadata, "Credit")
        date_time = _metadata_value(metadata, "DateTimeOriginal") or _metadata_value(
            metadata, "DateTime"
        )
        description = _metadata_value(metadata, "ImageDescription")
        license_url = _metadata_value(metadata, "LicenseUrl")
        page_url = "https://commons.wikimedia.org/wiki/File:" + urllib.parse.quote(
            title.replace(" ", "_"), safe="()_',.-"
        )
        thumb_url = info.get("thumburl") or info.get("url")
        original_url = info.get("url")
        if not isinstance(thumb_url, str) or not isinstance(original_url, str):
            raise RuntimeError(f"Commons did not return usable URLs for {title}")
        suffix = Path(urllib.parse.urlparse(original_url).path).suffix.lower() or ".jpg"
        local_name = f"image-{index:02d}{suffix}"
        local_path = INPUT_DIR / local_name
        sha256, byte_length = _download(thumb_url, local_path)
        resolved.append(
            {
                "index": index,
                "local_name": local_name,
                "commons_title": title,
                "source_page": page_url,
                "downloaded_url": thumb_url,
                "original_file_url": original_url,
                "license": license_short,
                "license_url": license_url,
                "artist": artist,
                "credit": credit,
                "date": date_time,
                "description": description,
                "downloaded_sha256": sha256,
                "downloaded_byte_length": byte_length,
                "commons_sha1_original": str(info.get("sha1", "")),
                "commons_original_width": int(info.get("width", 0)),
                "commons_original_height": int(info.get("height", 0)),
                "commons_mime": str(info.get("mime", "")),
            }
        )
        print(f"[{index + 1:02d}/{len(titles):02d}] {local_name}: {title} [{license_short}]")
        time.sleep(1.25)

    document = {
        "dataset_id": spec["dataset_id"],
        "provider": "Wikimedia Commons",
        "thumbnail_width_px": width,
        "files": resolved,
    }
    RESOLVED_PATH.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Downloaded and license-checked {len(resolved)} real photographs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
