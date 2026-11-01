"""Проверка и атомарная публикация выпуска. Один писатель."""
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from urllib.parse import unquote, urlsplit


BINDINGS = ('model_id', 'model_sha256', 'protocol_sha256',
            'golden_sha256', 'runtime_sha256')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def cache_key(data_path: Path, config: dict) -> str:
    return digest(canonical({'data_sha256': digest(Path(data_path).read_bytes()),
                             'config': config}))


def _sha256(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def validate_release(manifest: dict, report: dict) -> None:
    if not isinstance(manifest, dict) or not isinstance(report, dict):
        raise ValueError('Нужны manifest и report')
    if type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1:
        raise ValueError('Неверная schema_version')
    if not all(_text(manifest.get(k)) for k in ('model_id', 'artifact_uri')):
        raise ValueError('Пустой model_id или artifact_uri')
    if not all(_sha256(manifest.get(k)) for k in BINDINGS[1:]):
        raise ValueError('Неверный SHA-256')
    runtime = manifest.get('runtime')
    if not isinstance(runtime, dict) or not runtime:
        raise ValueError('Пустой runtime')
    try:
        runtime_hash = digest(canonical(runtime))
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError('Неверный runtime') from error
    if runtime_hash != manifest['runtime_sha256']:
        raise ValueError('Изменён runtime')
    if (report.get('status') != 'complete' or report.get('passed') is not True
            or not _text(report.get('evaluation_run_id'))):
        raise ValueError('Оценка не завершена или кандидат отклонён')
    if any(report.get(k) != manifest[k] for k in BINDINGS):
        raise ValueError('Отчёт другого выпуска')
    metrics = report.get('metrics')
    if not isinstance(metrics, dict):
        raise ValueError('Отсутствуют metrics')
    for name in ('accuracy', 'format_rate'):
        value = metrics.get(name)
        if type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value):
            raise ValueError(f'Неверная метрика {name}')


def verify_artifact(manifest: dict, root: Path) -> None:
    """Проверить закреплённые файлы. Ссылки HF на blobs разрешены."""
    if not isinstance(manifest, dict):
        raise ValueError('Нужен manifest')
    files = manifest.get('file_hashes')
    if not isinstance(files, dict) or not files:
        raise ValueError('Пустой file_hashes')
    for name, expected in files.items():
        if (not isinstance(name, str) or not name or '\\' in name
                or PurePosixPath(name).is_absolute()
                or any(part in ('', '.', '..') for part in name.split('/'))
                or not _sha256(expected)):
            raise ValueError('Неверный file_hashes')
    if digest(canonical(files)) != manifest.get('model_sha256'):
        raise ValueError('Неверный model_sha256')
    uri = manifest.get('artifact_uri')
    if not _text(uri):
        raise ValueError('Пустой artifact_uri')
    parsed = urlsplit(uri)
    if parsed.scheme:
        if (parsed.scheme != 'file' or parsed.netloc not in ('', 'localhost')
                or parsed.query or parsed.fragment or not parsed.path.startswith('/')):
            raise ValueError('Нужен локальный file URI')
        artifact = Path(unquote(parsed.path))
    else:
        artifact = Path(uri)
        if not artifact.is_absolute():
            artifact = Path(root) / artifact
    if not artifact.exists():
        raise ValueError('Артефакт отсутствует')
    base = artifact if artifact.is_dir() else artifact.parent
    for name, expected in files.items():
        path = base / name
        try:
            with path.open('rb') as handle:
                actual = hashlib.file_digest(handle, 'sha256').hexdigest()
        except (OSError, ValueError) as error:
            raise ValueError(f'Файл недоступен: {name}') from error
        if actual != expected:
            raise ValueError(f'Изменён файл: {name}')


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write(target, payload):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + '.',
                                         suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        sync_directory(target.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def publish(manifest: dict, report: dict, target: Path) -> None:
    validate_release(manifest, report)
    atomic_write(target, canonical(manifest))
