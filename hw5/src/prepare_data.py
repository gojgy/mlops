"""Воспроизводимая выборка из тензоров ДЗ 4, без повторной токенизации."""

import hashlib
import random
from pathlib import Path

import torch

from src.config import load_params


def main():
    params = load_params()
    cfg = params["sample"]
    ids = {}
    for split in ("train", "val"):
        source = Path(cfg[f"source_{split}"])
        blob = torch.load(source, weights_only=False)
        examples = random.Random(cfg["seed"]).sample(blob["examples"], cfg[f"{split}_size"])
        ids[split] = {e["id"] for e in examples}
        blob["examples"] = examples
        blob["selection"] = {
            "source": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "seed": cfg["seed"],
            "size": len(examples),
        }
        out = Path(params["data"][split])
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(blob, out)
        print(f"{split}: {len(examples)} примеров → {out}")
    if ids["train"] & ids["val"]:
        raise ValueError("train и val пересекаются по id")


if __name__ == "__main__":
    main()
