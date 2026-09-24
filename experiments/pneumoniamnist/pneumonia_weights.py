"""Fetch the PneumoniaMNIST weight files used by the inference example.

The Zenodo archive is stored uncompressed. Only the entries named below are
downloaded, by HTTP range, so the 94 MB ResNet-50 checkpoint stays remote.
"""
from pathlib import Path
import struct
import time
import urllib.request

WEIGHTS = Path(__file__).resolve().parent / "data" / "weights"
WEIGHTS_URL = "https://zenodo.org/api/records/7782114/files/weights_pneumoniamnist.zip/content"
# Local-header offset and size. The archive is immutable
# (Zenodo md5 01ee733fc0e3263e0b809e2b0182594d).
ZIP_ENTRIES = {
    "resnet18_28_1.pth": (19054427, 44737899),
    "autokeras_1/variables.data": (0, 116925),
    "autokeras_1/variables.index": (117022, 805),
    "autokeras_1/saved_model.pb": (117910, 137799),
    "autokeras_2/variables.data": (255781, 490804),
    "autokeras_2/variables.index": (746682, 1332),
    "autokeras_2/saved_model.pb": (748097, 241769),
    "autokeras_3/variables.data": (989938, 116925),
    "autokeras_3/variables.index": (1106960, 805),
    "autokeras_3/saved_model.pb": (1107848, 137799),
    "automl_vision_1/model.tflite": (1245719, 5852211),
    "automl_vision_1/labels.txt": (7098020, 17),
    "automl_vision_2/model.tflite": (7213569, 5852235),
    "automl_vision_2/labels.txt": (13065894, 17),
    "automl_vision_3/model.tflite": (13177314, 5876491),
    "automl_vision_3/labels.txt": (19053895, 17),
}


def _read_range(url, start, end, timeout=60):
    request = urllib.request.Request(url, headers={
        "User-Agent": "ssme-lite-pneumonia",
        "Range": f"bytes={start}-{end}",
    })
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _stream_range(url, start, end, destination):
    expected = end - start + 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == expected:
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    downloaded = partial.stat().st_size if partial.exists() else 0
    if downloaded > expected:
        partial.unlink()
        downloaded = 0
    failures = 0
    while downloaded < expected:
        downloaded = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={
            "User-Agent": "ssme-lite-pneumonia",
            "Range": f"bytes={start + downloaded}-{end}",
        })
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 206:
                    raise RuntimeError(f"expected HTTP 206, got {response.status}")
                with partial.open("ab" if downloaded else "wb") as handle:
                    while downloaded < expected:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
            failures = 0
        except Exception as error:
            failures += 1
            if failures > 8:
                raise RuntimeError(f"{destination.name} stalled at {downloaded} bytes") from error
            print(f"retry {destination.name} at {downloaded / 1e6:.1f} MB ({error})", flush=True)
            time.sleep(min(2 * failures, 10))
    if downloaded != expected:
        raise RuntimeError(f"{destination.name}: expected {expected} bytes, got {downloaded}")
    partial.replace(destination)
    return destination


def _zip_data_start(entry):
    header = _read_range(WEIGHTS_URL, entry, entry + 64)
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError("weights zip entry is missing a local header")
    name_len, extra_len = struct.unpack_from("<HH", header, 26)
    return entry + 30 + name_len + extra_len


def download_weights():
    """Return paths for the seven checkpoints, downloading any that are missing."""
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, (offset, size) in ZIP_ENTRIES.items():
        destination = WEIGHTS / name
        paths[name] = destination
        if destination.exists() and destination.stat().st_size == size:
            continue
        print(f"downloading {name} ({size / 1e6:.1f} MB)", flush=True)
        data_start = _zip_data_start(offset)
        _stream_range(WEIGHTS_URL, data_start, data_start + size - 1, destination)
    return paths
