"""Чтение сохранённых измерений без распаковки в рабочие файлы Git."""

import hashlib
from pathlib import Path
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def source():
    if any(p.is_dir() for p in RESULTS.glob("load-*")):
        return RESULTS
    archive = RESULTS / "raw.tar.gz"
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    parent = ROOT / "context/results"
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / digest
    if not target.is_dir():
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            with tarfile.open(archive, "r:gz") as data:
                data.extractall(temporary, filter="data")
            Path(temporary).rename(target)
    return target


if __name__ == "__main__":
    print(source())
