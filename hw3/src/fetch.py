"""Стадия fetch: скачать файл-источник и сверить sha256.

Контрольная сумма живёт в params.yaml: если зеркало перезальёт файл,
стадия упадёт, а не соберёт молча другой датасет.
"""

import hashlib
import shutil
import urllib.request
from pathlib import Path

from src.config import load_params

CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    src = load_params()["collect"]["source"]
    out = Path(src["path"])
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and sha256_of(out) == src["sha256"]:
        print(f"fetch: {out} уже на месте, sha256 совпал")
        return

    tmp = out.with_suffix(out.suffix + ".part")
    print(f"fetch: скачиваю {src['url']}")
    with urllib.request.urlopen(src["url"]) as resp, tmp.open("wb") as fh:
        shutil.copyfileobj(resp, fh, CHUNK)

    got = sha256_of(tmp)
    if got != src["sha256"]:
        tmp.unlink()
        raise SystemExit(
            f"fetch: sha256 не совпал — источник изменился.\n"
            f"  ожидался {src['sha256']}\n  получен  {got}"
        )
    tmp.replace(out)
    print(f"fetch: {out}, {out.stat().st_size / 1e6:.0f} МБ, sha256 совпал")


if __name__ == "__main__":
    main()
