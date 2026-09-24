"""Load the seven released PneumoniaMNIST checkpoints.

The short package demo, from inference through the report, is
``pneumonia_ssme.py``. This module only builds the sklearn-style models.

ResNet-18 @ 28 and the three AutoKeras graphs are checked against the published
test CSV. The three AutoML Vision files are TFLite; their AUC matches the
published files, while individual scores still depend on the image resizer.
"""
from pathlib import Path
import json
import numpy as np

from pneumonia_weights import WEIGHTS, download_weights

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
CSV_DIR = ROOT / "data" / "pneumoniamnist" / "official_test_csv"
SCORES = DATA / "inferred_test_scores.npz"
MODELS = (
    "resnet18_28_1",
    "autokeras_1", "autokeras_2", "autokeras_3",
    "automl_vision_1", "automl_vision_2", "automl_vision_3",
)
EXACT_OFFICIAL = {"resnet18_28_1", "autokeras_1", "autokeras_2", "autokeras_3"}


def _varint(blob, index):
    value, shift = 0, 0
    while True:
        byte = blob[index]
        index += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, index
        shift += 7


def _checkpoint_entries(index_bytes):
    def handles(footer):
        cursor = 0
        decoded = []
        for _ in range(2):
            offset, cursor = _varint(footer, cursor)
            size, cursor = _varint(footer, cursor)
            decoded.append((offset, size))
        return decoded

    meta_offset, _ = handles(index_bytes[-48:])[0]
    body = index_bytes[:meta_offset][:-5]
    restarts = int(np.frombuffer(body[-4:], dtype="<u4")[0])
    stop = len(body) - 4 - restarts * 4
    cursor, key, rows = 0, b"", []
    while cursor < stop:
        shared, cursor = _varint(body, cursor)
        unshared, cursor = _varint(body, cursor)
        value_len, cursor = _varint(body, cursor)
        key = key[:shared] + body[cursor:cursor + unshared]
        cursor += unshared
        rows.append((key.decode(), body[cursor:cursor + value_len]))
        cursor += value_len
    return rows


def _tensor_shape(blob):
    dims, cursor = [], 0
    while cursor < len(blob):
        key, cursor = _varint(blob, cursor)
        field, wire = key >> 3, key & 7
        if wire == 2:
            length, cursor = _varint(blob, cursor)
            chunk = blob[cursor:cursor + length]
            cursor += length
            if field == 2 and chunk:
                dim_key, dim_cursor = _varint(chunk, 0)
                if dim_key >> 3 == 1 and dim_key & 7 == 0:
                    dim, _ = _varint(chunk, dim_cursor)
                    dims.append(dim)
        elif wire == 0:
            _, cursor = _varint(blob, cursor)
        elif wire == 5:
            cursor += 4
        else:
            break
    return dims


def _bundle_tensor(entry, data):
    cursor, dtype, shape, offset, size = 0, None, [], 0, None
    while cursor < len(entry):
        key, cursor = _varint(entry, cursor)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, cursor = _varint(entry, cursor)
            if field == 1:
                dtype = value
            elif field == 4:
                offset = value
            elif field == 5:
                size = value
        elif wire == 2:
            length, cursor = _varint(entry, cursor)
            chunk = entry[cursor:cursor + length]
            cursor += length
            if field == 2:
                shape = _tensor_shape(chunk)
        elif wire == 5:
            cursor += 4
        else:
            raise RuntimeError("unsupported checkpoint protobuf wire type")
    kind = {1: np.float32, 3: np.int32, 9: np.int64}.get(dtype)
    if kind is None or size is None:
        raise RuntimeError(f"unsupported checkpoint tensor dtype {dtype}")
    array = np.frombuffer(data, dtype=kind, count=size // np.dtype(kind).itemsize, offset=offset)
    return array.reshape(shape).copy() if shape else array.reshape(()).copy()


def _saved_model_layers(path):
    text = Path(path).read_bytes().decode("latin1")
    best, start, needle = None, 0, '{"class_name": "Functional"'
    while True:
        at = text.find(needle, start)
        if at < 0:
            break
        depth = 0
        for index, char in enumerate(text[at:]):
            depth += 1 if char == "{" else -1 if char == "}" else 0
            if depth == 0:
                try:
                    layers = json.loads(text[at:at + index + 1])["config"]["layers"]
                except (json.JSONDecodeError, KeyError):
                    break
                if best is None or len(layers) > len(best):
                    best = layers
                break
        start = at + len(needle)
    if not best:
        raise RuntimeError(f"no Keras graph in {path}")
    return best


def load_autokeras_run(directory):
    directory = Path(directory)
    data = (directory / "variables.data").read_bytes()
    tensors = {}
    for name, entry in _checkpoint_entries((directory / "variables.index").read_bytes()):
        if not name.endswith("VARIABLE_VALUE") or name.startswith("keras_api/metrics"):
            continue
        tensors[name.split("/.ATTRIBUTES")[0]] = _bundle_tensor(entry, data)
    ids = sorted(int(key.split("/")[0].rsplit("-", 1)[1]) for key in tensors if key.endswith("/kernel"))
    return {
        "layers": _saved_model_layers(directory / "saved_model.pb"),
        "mean": tensors["layer_with_weights-0/mean"],
        "variance": tensors["layer_with_weights-0/variance"],
        "weights": [(tensors[f"layer_with_weights-{i}/kernel"], tensors[f"layer_with_weights-{i}/bias"]) for i in ids],
    }


class _PositiveModel:
    """sklearn-style view consumed by PredictionMatrixGenerator."""

    classes_ = np.array([0, 1])

    def __init__(self, positive_fn):
        self.positive_fn = positive_fn

    def predict_proba(self, images):
        positive = np.asarray(self.positive_fn(images), dtype=float).reshape(-1)
        if len(positive) != len(images) or not np.isfinite(positive).all():
            raise RuntimeError("inference did not return one finite positive score per image")
        self.positive_ = positive
        return np.column_stack([1.0 - positive, positive])


def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _positive_from_logits(model, batch, device):
    import torch
    import torch.nn.functional as functional
    columns = []
    with torch.inference_mode():
        for start in range(0, len(batch), 64):
            logits = model(batch[start:start + 64].to(device))
            columns.append(functional.softmax(logits, dim=1)[:, 1].detach().cpu())
    return torch.cat(columns).numpy()


def _small_resnet():
    import torch.nn as nn
    import torch.nn.functional as functional

    class BasicBlock(nn.Module):
        def __init__(self, in_planes, planes, stride):
            super().__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes),
                )

        def forward(self, x):
            out = self.bn2(self.conv2(functional.relu(self.bn1(self.conv1(x)))))
            return functional.relu(out + self.shortcut(x))

    class ResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_planes = 64
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(64, 2, 1)
            self.layer2 = self._make_layer(128, 2, 2)
            self.layer3 = self._make_layer(256, 2, 2)
            self.layer4 = self._make_layer(512, 2, 2)
            self.linear = nn.Linear(512, 2)

        def _make_layer(self, planes, blocks, stride):
            layers, strides = [], [stride] + [1] * (blocks - 1)
            for item in strides:
                layers.append(BasicBlock(self.in_planes, planes, item))
                self.in_planes = planes
            return nn.Sequential(*layers)

        def forward(self, x):
            x = functional.relu(self.bn1(self.conv1(x)))
            x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
            return self.linear(functional.adaptive_avg_pool2d(x, 1).flatten(1))

    return ResNet()


def resnet18_28(images, path, device):
    import torch
    network = _small_resnet()
    network.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["net"])
    network.eval().to(device)
    batch = torch.from_numpy(np.ascontiguousarray(images)).float().unsqueeze(1).repeat(1, 3, 1, 1)
    return _positive_from_logits(network, (batch / 255.0 - 0.5) / 0.5, device)


def autokeras_positive(images, run, device):
    import torch
    import torch.nn.functional as functional
    mean = float(np.asarray(run["mean"]).reshape(()))
    scale = float(np.sqrt(np.asarray(run["variance"]).reshape(())))
    pending = iter(run["weights"])
    x = (torch.from_numpy(np.ascontiguousarray(images)).float().to(device) - mean) / scale
    x = x.unsqueeze(1)
    with torch.inference_mode():
        for layer in run["layers"]:
            kind, config = layer["class_name"], layer["config"]
            if kind in ("InputLayer", "Custom>CastToFloat32", "Custom>ExpandLastDim", "Normalization", "Dropout"):
                continue
            if kind == "Conv2D":
                kernel, bias = next(pending)
                weight = torch.from_numpy(np.ascontiguousarray(kernel.transpose(3, 2, 0, 1))).to(device)
                bias = torch.from_numpy(np.ascontiguousarray(bias)).to(device)
                padding = "same" if config["padding"] == "same" else 0
                x = functional.relu(functional.conv2d(x, weight, bias, padding=padding))
            elif kind == "MaxPooling2D":
                x = functional.max_pool2d(x, config["pool_size"]["items"][0], stride=config["strides"]["items"][0])
            elif kind == "Flatten":
                x = x.permute(0, 2, 3, 1).reshape(len(images), -1)
            elif kind == "Dense":
                kernel, bias = next(pending)
                weight = torch.from_numpy(np.ascontiguousarray(kernel)).to(device)
                bias = torch.from_numpy(np.ascontiguousarray(bias)).to(device)
                x = x @ weight + bias
            elif kind == "Activation" and config["activation"] == "sigmoid":
                x = torch.sigmoid(x)
            else:
                raise RuntimeError(f"unsupported AutoKeras layer {kind}")
        if next(pending, None) is not None:
            raise RuntimeError("AutoKeras checkpoint has unused weights")
        return x.flatten().cpu().numpy()


def automl_positive(images, model_path, label_path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError as exc:
        raise ImportError("AutoML Vision weights are TFLite; pip install ai-edge-litert") from exc
    from PIL import Image
    labels = [line.strip().lower() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    pneumonia = labels.index("pneumonia")
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    scores = np.empty(len(images), dtype=np.float64)
    for index, image in enumerate(images):
        picture = Image.fromarray(np.uint8(image)).resize((224, 224), Image.BICUBIC).convert("RGB")
        interpreter.set_tensor(input_index, np.expand_dims(np.asarray(picture), 0))
        interpreter.invoke()
        quantized = np.asarray(interpreter.get_tensor(output_index)).reshape(-1).astype(np.float32) / 255.0
        scores[index] = quantized[pneumonia]
    return scores


def build_models(paths):
    device = _device()
    print(f"inference device: {device}", flush=True)
    functions = {"resnet18_28_1": lambda images: resnet18_28(images, paths["resnet18_28_1.pth"], device)}
    for run in (1, 2, 3):
        autokeras = load_autokeras_run(WEIGHTS / f"autokeras_{run}")
        functions[f"autokeras_{run}"] = lambda images, network=autokeras: autokeras_positive(images, network, device)
        functions[f"automl_vision_{run}"] = lambda images, run=run: automl_positive(
            images, paths[f"automl_vision_{run}/model.tflite"], paths[f"automl_vision_{run}/labels.txt"])
    return {name: _PositiveModel(functions[name]) for name in MODELS}


def load_test_images():
    import sys
    examples = str(ROOT / "examples")
    if examples not in sys.path:
        sys.path.insert(0, examples)
    from pneumonia_official import download_dataset
    with np.load(download_dataset()) as archive:
        return archive["test_images"], archive["test_labels"].astype(int).reshape(-1)


def _official_positive(name):
    import pandas as pd
    matches = sorted(CSV_DIR.glob(f"pneumoniamnist_test_*@{name}.csv"))
    if len(matches) != 1:
        return None
    frame = pd.read_csv(matches[0], header=None, index_col=0).sort_index()
    return frame.to_numpy(dtype=float)[:, -1]


def check_against_official(models, y):
    from sklearn.metrics import roc_auc_score
    failed = []
    for name, model in models.items():
        published = _official_positive(name)
        if published is None:
            continue
        score = model.positive_
        gap = float(np.max(np.abs(score - published)))
        auc, official_auc = float(roc_auc_score(y, score)), float(roc_auc_score(y, published))
        print(f"{name}: max|Δ|={gap:.3e} auc {auc:.3f} vs {official_auc:.3f}", flush=True)
        if name in EXACT_OFFICIAL and (gap > 1e-3 or abs(auc - official_auc) > 0.0015):
            failed.append(name)
    if failed:
        raise RuntimeError(f"inference does not reproduce published test scores: {failed}")


def infer_scores(images=None):
    """Download anything missing, then return ``(scores, y, model_names)``."""
    from ssme_lite import PredictionMatrixGenerator
    if images is None:
        images, y = load_test_images()
    else:
        _, y = load_test_images()
    paths = download_weights()
    models = build_models(paths)
    generator = PredictionMatrixGenerator(models)
    scores = generator.fit_transform(images)
    if list(generator.model_names_) != list(MODELS):
        raise RuntimeError("prediction matrix columns are not the predeclared model order")
    check_against_official(models, y)
    return scores, y, generator.model_names_


def save_scores(scores, y, names, destination=SCORES):
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, scores=scores, y=y, model_names=np.asarray(names))
    note = destination.with_name("inferred_models.json")
    note.write_text(json.dumps({
        "dataset": "PneumoniaMNIST",
        "pool": "official_test",
        "weights": "https://doi.org/10.5281/zenodo.7782114 weights_pneumoniamnist.zip",
        "models": list(names),
        "omitted": "autosklearn snapshots were not released; resnet50_224_1 was not downloaded",
    }, indent=2), encoding="utf-8")
    return destination


def main():
    scores, y, names = infer_scores()
    path = save_scores(scores, y, names)
    print(path)


if __name__ == "__main__":
    main()
