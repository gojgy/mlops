"""Чтение params.yaml — единственная точка правды о конфигурации."""

from pathlib import Path

import yaml


def load_params(path: str = "params.yaml") -> dict:
    """Загрузить параметры запуска."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def source_file(params: dict) -> Path:
    """Файл-источник. Его кладёт стадия fetch; collect только читает."""
    path = Path(params["collect"]["source"]["path"])
    if not path.exists():
        raise SystemExit(
            f"стадия collect не нашла источник: {path}\n"
            "Его скачивает стадия fetch: запустите `dvc repro` целиком\n"
            "или `dvc repro fetch` отдельно."
        )
    return path


def version_window(params: dict) -> tuple[int, int]:
    """Окно лет текущей версии датасета, границы включительно.

    Версия живёт в params, а не в аргументах командной строки: иначе
    dvc.lock не запомнит, из чего собран артефакт.
    """
    version = params["collect"]["version"]
    versions = params["collect"]["versions"]
    if version not in versions:
        raise SystemExit(
            f"collect.version = {version!r}, но в collect.versions "
            f"есть только {sorted(versions)}"
        )
    first, last = versions[version]["years"]
    return first, last
