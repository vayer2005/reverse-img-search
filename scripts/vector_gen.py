"""
DINOv2 image embeddings (ViT-B/14, 768-d CLS) for Postgres pgvector / Elasticsearch.

Uses Hugging Face `facebook/dinov2-base` — same preprocessing as `AutoImageProcessor`.
Vectors are L2-normalized by default (cosine / dot-product search on normalized vectors).

Run as a script: encodes every image file under `IMG_PATH` (from `.env` or the
environment), one JSON object per line (`path` + `vector`) on stdout.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from typing import BinaryIO, Iterable, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# schema: dinov2-base = ViT-B/14, dim 768.
MODEL_ID = "facebook/dinov2-base"
EMBEDDING_DIM = 768

_img = os.getenv("IMG_PATH", "").strip()
IMAGE_DIR = Path(_img).expanduser() if _img else None


ImageInput = Union[str, Path, Image.Image, bytes, BinaryIO]


def _to_pil(image: ImageInput) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(Path(image)).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, bytes):
        from io import BytesIO

        return Image.open(BytesIO(image)).convert("RGB")
    return Image.open(image).convert("RGB")


class Dinov2Vectorizer:
    """Loads DINOv2 once; encodes images to numpy vectors (float32)."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: torch.device | None = None,
        l2_normalize: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.l2_normalize = l2_normalize
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id)
        self._model.eval()
        self._model.to(self.device)

    @torch.inference_mode()
    def encode_pil(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        inputs = self._processor(images=list(images), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self._model(**inputs)
        # CLS token = index 0 (DINOv2 ViT).
        x = out.last_hidden_state[:, 0, :].float()
        if self.l2_normalize:
            x = F.normalize(x, p=2, dim=-1)
        return x.cpu().numpy().astype(np.float32)

    def encode_one(self, image: ImageInput) -> np.ndarray:
        pil = _to_pil(image)
        return self.encode_pil([pil])[0]

    def encode_paths(self, paths: Iterable[Union[str, Path]]) -> np.ndarray:
        pils = [_to_pil(p) for p in paths]
        return self.encode_pil(pils)


def encode_image(
    image: ImageInput,
    *,
    model_id: str = MODEL_ID,
    l2_normalize: bool = True,
    device: torch.device | None = None,
) -> np.ndarray:
    """Single-image convenience API (loads model every call — prefer `Dinov2Vectorizer` for batches)."""
    v = Dinov2Vectorizer(model_id=model_id, device=device, l2_normalize=l2_normalize)
    return v.encode_one(image)


def _iter_image_files(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.iterdir()
        if p.is_file()
    )


def main() -> None:
    if IMAGE_DIR is None:
        print(
            "Set IMG_PATH in .env (project root) or the environment, e.g. IMG_PATH=/path/to/images",
            file=sys.stderr,
        )
        sys.exit(1)
    if not IMAGE_DIR.is_dir():
        print(f"Not a directory: {IMAGE_DIR}", file=sys.stderr)
        sys.exit(1)
    paths = _iter_image_files(IMAGE_DIR)
    if not paths:
        print(f"No image files in {IMAGE_DIR}", file=sys.stderr)
        return

    vec = Dinov2Vectorizer()
    for path in paths:
        v = vec.encode_one(path)
        print(json.dumps({"path": str(path), "vector": len(v.tolist())}))


if __name__ == "__main__":
    main()
