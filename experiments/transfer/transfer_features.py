"""Frozen ImageNet ResNet18 features for the PneumoniaMNIST transfer experiment.

The backbone is pretrained on ImageNet and never updated. Pneumonia labels are
not used here. This writes ``data/resnet18_features.npz`` for the sklearn heads.

    python experiments/transfer/transfer_features.py
"""
from pathlib import Path
import hashlib
import shutil
import time
import urllib.request
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FEATURES = DATA / "resnet18_features.npz"
DATASET = DATA / "pneumoniamnist.npz"
SHARED_DATASET = HERE.parents[1] / "data" / "pneumoniamnist" / "pneumoniamnist.npz"
DATASET_URLS = [
    "https://zenodo.org/records/10519652/files/pneumoniamnist.npz?download=1",
    "https://zenodo.org/records/6496656/files/pneumoniamnist.npz?download=1",
]
DATASET_MD5 = "28209eda62fecd6e6a2d98b1501bb15f"
# torchvision ResNet18 ImageNet-1k V1, hosted as timm's resnet18.tv_in1k.
# download.pytorch.org is too slow from this network, so the file is fetched
# in HTTP ranges from the mirror and checked by size.
BACKBONE_URL = "https://hf-mirror.com/timm/resnet18.tv_in1k/resolve/main/pytorch_model.bin"
BACKBONE = DATA / "resnet18_tv_in1k.pth"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _valid_dataset(path):
    return path.exists() and hashlib.md5(path.read_bytes()).hexdigest() == DATASET_MD5


def download_dataset():
    """Return the official MedMNIST npz, downloading it into this experiment."""
    if _valid_dataset(DATASET):
        return DATASET
    DATA.mkdir(parents=True, exist_ok=True)
    if _valid_dataset(SHARED_DATASET):
        shutil.copy2(SHARED_DATASET, DATASET)
        return DATASET
    last_error = None
    for url in DATASET_URLS:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ssme-lite-transfer"})
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = response.read()
        except Exception as error:
            last_error = error
            continue
        if hashlib.md5(payload).hexdigest() != DATASET_MD5:
            last_error = RuntimeError("pneumoniamnist.npz md5 did not match")
            continue
        DATASET.write_bytes(payload)
        return DATASET
    raise RuntimeError(f"could not download pneumoniamnist.npz: {last_error}")


def load_images():
    with np.load(download_dataset()) as archive:
        return (archive["train_images"], archive["train_labels"].astype(int).reshape(-1),
                archive["test_images"], archive["test_labels"].astype(int).reshape(-1))


def _stream_range(url, start, end, destination):
    expected = end - start + 1
    request = urllib.request.Request(url, headers={
        "User-Agent": "ssme-lite-transfer",
        "Range": f"bytes={start}-{end}",
    })
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206:
            raise RuntimeError(f"expected HTTP 206, got {response.status}")
        payload = response.read()
    if len(payload) != expected:
        raise RuntimeError(f"range {start}-{end} returned {len(payload)} bytes")
    with destination.open("ab") as handle:
        handle.write(payload)


def download_backbone():
    """Return the frozen ImageNet ResNet18 checkpoint."""
    if BACKBONE.exists() and BACKBONE.stat().st_size > 40_000_000:
        return BACKBONE
    DATA.mkdir(parents=True, exist_ok=True)
    partial = BACKBONE.with_suffix(".pth.part")
    probe = urllib.request.Request(BACKBONE_URL, headers={
        "User-Agent": "ssme-lite-transfer",
        "Range": "bytes=0-0",
    })
    with urllib.request.urlopen(probe, timeout=30) as response:
        total = int(response.headers["Content-Range"].split("/")[-1])
    downloaded = partial.stat().st_size if partial.exists() else 0
    if downloaded > total:
        partial.unlink()
        downloaded = 0
    print(f"downloading ResNet18 ({total / 1e6:.1f} MB)", flush=True)
    failures = 0
    while downloaded < total:
        end = min(downloaded + (1 << 20) - 1, total - 1)
        try:
            _stream_range(BACKBONE_URL, downloaded, end, partial)
        except Exception as error:
            failures += 1
            if failures > 8:
                raise RuntimeError(f"ResNet18 stalled at {downloaded} bytes") from error
            print(f"retry at {downloaded / 1e6:.1f} MB ({error})", flush=True)
            time.sleep(min(2 * failures, 10))
            continue
        failures = 0
        downloaded = end + 1
        print(f"  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB", flush=True)
    partial.replace(BACKBONE)
    return BACKBONE


def extract_features(images, batch_size=64, device="auto"):
    """Map uint8 images to 512-d features. The classifier layer is removed."""
    import torch
    import torch.nn.functional as functional
    from torchvision.models import resnet18
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    backbone = resnet18(weights=None)
    state = torch.load(download_backbone(), map_location="cpu", weights_only=True)
    state = {key: value for key, value in state.items() if not key.startswith("fc.")}
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    if unexpected or set(missing) != {"fc.weight", "fc.bias"}:
        raise RuntimeError(f"ResNet18 checkpoint does not match torchvision: missing={missing} unexpected={unexpected}")
    backbone.fc = torch.nn.Identity()
    backbone.eval()
    backbone.requires_grad_(False)
    backbone.to(device)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    print(f"feature extraction device: {device}", flush=True)
    columns = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(images[start:start + batch_size])).float()
            if batch.ndim == 3:
                batch = batch.unsqueeze(1)
            batch = batch.repeat(1, 3, 1, 1).to(device) / 255.0
            batch = functional.interpolate(batch, size=224, mode="bilinear", align_corners=False)
            columns.append(backbone((batch - mean) / std).cpu().numpy())
    return np.concatenate(columns)


def cached_features(batch_size=64, device="auto"):
    if FEATURES.exists():
        with np.load(FEATURES, allow_pickle=False) as archive:
            return archive["X_train"], archive["y_train"], archive["X_test"], archive["y_test"]
    train_images, y_train, test_images, y_test = load_images()
    X_train = extract_features(train_images, batch_size=batch_size, device=device)
    X_test = extract_features(test_images, batch_size=batch_size, device=device)
    FEATURES.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FEATURES, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    return X_train, y_train, X_test, y_test


def main():
    X_train, y_train, X_test, y_test = cached_features()
    print(FEATURES)
    print(f"train {X_train.shape} test {X_test.shape} labels {np.bincount(y_train).tolist()} / {np.bincount(y_test).tolist()}")


if __name__ == "__main__":
    main()
