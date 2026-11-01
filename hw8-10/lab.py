"""Подготовка и выпуск. initialize создаёт только синтетические данные."""
import argparse
import hashlib
import json
import uuid
from pathlib import Path
import release


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    release.atomic_write(Path(path), canonical(value))


def initialize(root):
    if root.exists():
        raise FileExistsError('Рабочий каталог уже существует. Выберите новый --workspace.')
    root.mkdir(parents=True)
    runtime={'base_revision':'fixture-only','tokenizer_sha256':'a'*64,
             'template_sha256':'b'*64,'generation':{'max_new_tokens':32,'do_sample':False},'dtype':'float32'}
    for version in ['A','B']:
        artifact=root/f'model-{version}.txt';artifact.write_text(f'STUB ONLY — version {version}\n')
        files={artifact.name:digest(artifact.read_bytes())}
        m={'schema_version':1,'model_id':f'fixture-{version}',
           'artifact_uri':artifact.name,'model_sha256':digest(canonical(files)),
           'protocol_sha256':'d'*64,'golden_sha256':'e'*64,
           'runtime_sha256':digest(canonical(runtime)),'runtime':runtime,'file_hashes':files}
        report={k:m[k] for k in ['model_id','model_sha256','protocol_sha256','golden_sha256','runtime_sha256']}
        report.update(status='complete',passed=True,evaluation_run_id=f'fixture-eval-{version}',metrics={'accuracy':.8,'format_rate':1.0},synthetic=True)
        write(root/f'manifest-{version}.json',m);write(root/f'report-{version}.json',report)
    write(root/'accepted.json',read(root/'manifest-A.json'))
    report=read(root/'report-B.json');report['status']='running';report['passed']='false'
    write(root/'report-incomplete.json',report)
    report=read(root/'report-B.json');report['passed']=False
    write(root/'report-reject.json',report)
    write(root/'config.json',{'tokenizer_revision':'r1','template_sha256':'b'*64,'max_length':128,'code_revision':'prepare-v1'})
    (root/'input.txt').write_text('Первый пример\nВторой пример\n')
    print('Создан синтетический стенд:',root)


def _row_count(payload, format_name, schema=None):
    if format_name == 'text':
        return len(payload.splitlines())
    if format_name != 'jsonl':
        raise ValueError('Неизвестный format')
    rows = [json.loads(line) for line in payload.decode('utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError('Пустые данные')
    ids = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Нужна запись JSON')
        row_id = row.get('id')
        if not isinstance(row_id, str) or not row_id.strip() or row_id in ids:
            raise ValueError('Нужны уникальные непустые id')
        ids.add(row_id)
        if not isinstance(row.get('input'), str) or not row['input'].strip():
            raise ValueError('Пустой input')
        choices = row.get('choices')
        if (not isinstance(choices, (list, dict)) or not choices
                or any(not isinstance(label, str) or not label.strip() for label in choices)):
            raise ValueError('Неверные choices')
        expected = row.get('expected')
        if not isinstance(expected, str) or expected not in choices:
            raise ValueError('Неверный expected')
        if schema == 'golden':
            if not isinstance(choices, list):
                raise ValueError('Нужен список choices')
            for name in ('group_id', 'source'):
                if not isinstance(row.get(name), str) or not row[name].strip():
                    raise ValueError(f'Пустой {name}')
            slices = row.get('slices')
            if (not isinstance(slices, list) or not slices
                    or any(not isinstance(item, str) or not item.strip() for item in slices)):
                raise ValueError('Неверные slices')
    return len(rows)


def prepare(root):
    root = Path(root)
    config = read(root / 'config.json')
    if not isinstance(config, dict):
        raise ValueError('Нужен config')
    payload = (root / 'input.txt').read_bytes()
    input_sha256 = digest(payload)
    key = digest(canonical({'data_sha256': input_sha256, 'config': config}))
    cache = root / 'cache'
    cache.mkdir(exist_ok=True)
    entry = cache / (key + '.json')
    if entry.exists():
        try:
            cached = read(entry)
            result = cached['result']
            if (cached['cache_key'] == key and result['input_sha256'] == input_sha256
                    and canonical(result['config']) == canonical(config)
                    and type(result['rows']) is int and result['rows'] >= 0
                    and cached['result_sha256'] == digest(canonical(result))):
                print('CACHE_HIT', input_sha256)
                return result
        except (ValueError, TypeError, OSError, KeyError):
            pass
    result = {'input_sha256': input_sha256,
              'rows': _row_count(payload, config.get('format', 'text'), config.get('schema')),
              'config': config}
    write(entry, {'cache_key': key, 'result': result,
                  'result_sha256': digest(canonical(result))})
    print('PREPARED', result['input_sha256'])
    return result


def _events(root):
    path = root / 'publications.jsonl'
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def _record_event(root, event):
    events = _events(root)
    if any(item['publication_id'] == event['publication_id'] for item in events):
        return
    events.append(event)
    release.atomic_write(root / 'publications.jsonl',
                         b''.join(canonical(item) + b'\n' for item in events))


def _recover_publication(root):
    pending = root / '.publication-pending.json'
    if not pending.exists():
        return
    transaction = read(pending)
    target = root / 'accepted.json'
    if target.exists() and read(target) == transaction['manifest']:
        _record_event(root, transaction['event'])
    pending.unlink()
    release.sync_directory(root)


def publish_once(root, manifest, report, *, fail_after_publish=False):
    """Публикация с восстановлением журнала. Один писатель."""
    root = Path(root)
    _recover_publication(root)
    release.validate_release(manifest, report)
    release.verify_artifact(manifest, root)
    publication_id = digest(canonical({'manifest': manifest, 'report': report}))
    event = {'publication_id': publication_id, 'model_id': manifest['model_id'],
             'model_sha256': manifest['model_sha256'],
             'runtime_sha256': manifest['runtime_sha256'],
             'evaluation_run_id': report['evaluation_run_id']}
    target = root / 'accepted.json'
    description = {**event, 'target': str(target)}
    if (target.exists() and read(target) == manifest
            and any(item['publication_id'] == publication_id for item in _events(root))):
        return description
    pending = root / '.publication-pending.json'
    write(pending, {'manifest': manifest, 'event': event})
    release.publish(manifest, report, target)
    if fail_after_publish:
        raise RuntimeError('Инъекция сбоя после публикации')
    _record_event(root, event)
    pending.unlink()
    release.sync_directory(root)
    return description


def run(root, report_name, fail_after_prepare=False, fail_after_publish=False,
        manifest_name='manifest-B.json'):
    root = Path(root)
    attempt_id = uuid.uuid4().hex
    attempt_path = root / 'attempts' / (attempt_id + '.json')
    attempt = {'attempt_id': attempt_id, 'status': 'running',
               'manifest': str(manifest_name), 'report': str(report_name)}
    write(attempt_path, attempt)
    try:
        _recover_publication(root)
        prepared = prepare(root)
        attempt['prepared'] = prepared['input_sha256']
        if fail_after_prepare:
            raise RuntimeError('Инъекция сбоя после подготовки')
        manifest = read(root / manifest_name)
        report = read(root / report_name)
        description = publish_once(root, manifest, report,
                                   fail_after_publish=fail_after_publish)
        attempt.update(status='complete', publication=description)
        write(attempt_path, attempt)
        print('PUBLISHED', manifest['model_id'])
        return description
    except BaseException as error:
        attempt.update(status='failed', error=str(error))
        write(attempt_path, attempt)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=Path('.lab'))
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('init')
    run_parser = sub.add_parser('run')
    run_parser.add_argument('--report', default='report-B.json')
    run_parser.add_argument('--manifest', default='manifest-B.json')
    run_parser.add_argument('--fail-after-prepare', action='store_true')
    run_parser.add_argument('--fail-after-publish', action='store_true')
    args = parser.parse_args()
    if args.command == 'init':
        initialize(args.workspace)
    else:
        run(args.workspace, args.report, args.fail_after_prepare,
            args.fail_after_publish, args.manifest)


if __name__ == '__main__':
    main()
