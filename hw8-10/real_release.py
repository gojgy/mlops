"""Связь выпуска с сохранённой оценкой ДЗ 6–7. Модель не запускается.

prepare сохраняет исходную резервную версию, а не принимает её заново.
Для чтения MLmodel нужен PyYAML; MLflow нужен только с --tracking-uri.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
HW67 = HERE.parent / 'hw6-7'
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'vocab.json',
                   'merges.txt', 'special_tokens_map.json', 'added_tokens.json', 'chat_template.jinja')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f'Повторное поле JSON: {key}')
            obj[key] = value
        return obj
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def evaluation_modules(source):
    modules = []
    for name in ('contract', 'solution'):
        spec = importlib.util.spec_from_file_location('hw67_' + name, source / 'evaluation' / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


def verify_evaluation(source=HW67):
    """Проверить прежние хеши и заново вычислить score из исходных ответов."""
    source = Path(source).resolve()
    contract, solution = evaluation_modules(source)
    config = read(source / 'evaluation/params.json')
    rows = contract.read_rows(source / config['golden'])
    behavior = contract.read_rows(source / config['behavioral'])
    training = contract.read_rows(source / config['training_index'])
    proof = contract.validate_golden(rows, training, behavior)
    baseline = read(source / 'evaluation/baseline.json')
    candidate = read(source / 'metrics/candidate-model.json')
    protocol = dict(config=config, golden=rows, behavioral=behavior, training_index=training,
                    evaluation_code={p.name: p.read_text() for p in sorted((source / 'evaluation').glob('*.py'))})
    expected = dict(protocol_sha256=contract.fingerprint(protocol),
                    golden_sha256=contract.fingerprint(rows), baseline_spec_sha256=contract.fingerprint(baseline))
    # Сверяем с прежним приложением; это проверка целостности, не цифровая подпись.
    archive = read(source / 'context/submission/manifest.json')['files']
    names = ['evaluation/params.json', 'evaluation/baseline.json', 'metrics/candidate-model.json']
    names += [f'metrics/{role}-{suffix}.json' for role in ('baseline', 'candidate')
              for suffix in ('eval', 'outputs')]
    for name in names:
        if file_hash(source / name) != archive[name]:
            raise ValueError('Файл отличается от прежнего приложения: ' + name)
    reports = {}
    for role, model in [('baseline', baseline), ('candidate', candidate)]:
        old = read(source / f'metrics/{role}-eval.json')
        outputs = read(source / f'metrics/{role}-outputs.json')
        if old['status'] != 'complete' or old['model'] != model or old['leakage'] != proof:
            raise ValueError('Неполный или чужой исходный отчёт: ' + role)
        if any(old[key] != value for key, value in expected.items()):
            raise ValueError('Изменился протокол, данные или база: ' + role)
        if contract.fingerprint(outputs) != old['outputs_sha256']:
            raise ValueError('Изменены исходные ответы: ' + role)
        by_id = {o['id']: o for o in outputs}
        if len(by_id) != len(outputs) or set(by_id) != {r['id'] for r in rows + behavior}:
            raise ValueError('Неверные id ответов: ' + role)
        measured = solution.score(rows, [by_id[r['id']] for r in rows])
        measured['behavioral'] = solution.score(behavior, [by_id[r['id']] for r in behavior])
        if any(old[key] != measured[key] for key in measured):
            raise ValueError('Исходные метрики не соответствуют ответам: ' + role)
        if not isinstance(old.get('evaluation_run_id'), str) or not old['evaluation_run_id']:
            raise ValueError('Нет evaluation run ID')
        reports[role] = {**old, **measured}
    if reports['baseline']['evaluation_run_id'] != reports['candidate']['evaluation_run_id']:
        raise ValueError('Разные evaluation run ID')
    return config, rows, reports


def gate_result(baseline, candidate, critical, source):
    _, solution = evaluation_modules(source)
    try:
        result = solution.quality_gate(baseline, candidate, critical)
        if not isinstance(result, dict) or result.get('passed') is not True:
            raise ValueError('quality_gate не подтвердил passed=True')
        return result
    except ValueError as exc:
        return {'passed': False, 'reason': str(exc)}


def hashes(root):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink() and not path.exists():
            raise ValueError('Повреждённая ссылка: ' + str(path))
        if path.is_file():
            result[path.relative_to(root).as_posix()] = file_hash(path)
    if not result:
        raise ValueError('Пустой артефакт: ' + str(root))
    return result


def snapshot_path(baseline, snapshot=None):
    if baseline['name'] != 'Qwen/Qwen3-1.7B' or not re.fullmatch('[0-9a-f]{40}', baseline['revision']):
        raise ValueError('Нужна закреплённая Qwen3-1.7B')
    hub = Path(os.environ.get('HF_HUB_CACHE', Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface')) / 'hub'))
    root = Path(snapshot) if snapshot else hub / 'models--Qwen--Qwen3-1.7B/snapshots' / baseline['revision']
    root = root.resolve()
    if root.name != baseline['revision']:
        raise ValueError('Имя snapshot не соответствует revision')
    index = read(root / 'model.safetensors.index.json')
    for name in set(index['weight_map'].values()) | {'config.json', 'tokenizer.json', 'tokenizer_config.json', 'generation_config.json'}:
        if Path(name).is_absolute() or '..' in Path(name).parts or not (root / name).is_file():
            raise ValueError('Нет файла snapshot: ' + name)
    return root


def mlmodel_adapter(path, run_id):
    import yaml
    path = Path(path).resolve()
    if path.is_dir():
        path = path / 'MLmodel'
    info = yaml.safe_load(path.read_text())
    if info.get('run_id') != run_id or info.get('artifact_path') != 'model':
        raise ValueError('MLmodel другого training run: ' + str(path))
    root = path.parent
    adapter = (root / info['flavors']['python_function']['artifacts']['adapter']['path']).resolve()
    if not adapter.is_relative_to(root):
        raise ValueError('Путь адаптера вне MLmodel')
    for name in ('adapter_model.safetensors', 'adapter_config.json', 'manifest.json', 'tokenizer.json', 'tokenizer_config.json'):
        if not (adapter / name).is_file():
            raise ValueError('Нет файла адаптера: ' + name)
    return path, adapter


def obtain_training_artifact(spec, candidate_model=None, tracking_uri=None):
    """Чтение готового training run, без обучения и загрузки модели."""
    paths = []
    if tracking_uri:
        from mlflow import MlflowClient
        client = MlflowClient(tracking_uri=tracking_uri)
        version = client.get_model_version(spec['name'], str(spec['version']))
        if version.run_id != spec['run_id']:
            raise ValueError('Registry version ссылается на другой run')
        paths = [Path(client.download_artifacts(spec['run_id'], 'model')) / 'MLmodel']
    elif candidate_model:
        paths = [Path(candidate_model)]
    else:
        temp = Path(tempfile.gettempdir())
        paths = sorted(set(temp.glob('tmp*/MLmodel')) | set(temp.glob('tmp*/model/MLmodel')))
    found = []
    for path in paths:
        try:
            found.append(mlmodel_adapter(path, spec['run_id']))
        except ValueError:
            if candidate_model or tracking_uri:
                raise
    if not found:
        raise FileNotFoundError('Нет MLmodel выбранного run; задайте --candidate-model или --tracking-uri')
    signatures = [hashes(adapter) for _, adapter in found]
    if any(value != signatures[0] for value in signatures):
        raise ValueError('Копии артефакта одного run различаются')
    model_file, adapter = found[0]
    return model_file, adapter, {'training_run_id': spec['run_id'], 'registry_uri': spec['uri'],
                                'mlmodel_uri': model_file.as_uri(), 'mlmodel_sha256': file_hash(model_file),
                                'matching_local_copies': len(found), 'training_repeated': False}


def runtime_for(base, tokenizer, baseline, config):
    tokenizer = Path(tokenizer)
    files = {name: file_hash(tokenizer / name) for name in TOKENIZER_FILES if (tokenizer / name).is_file()}
    cfg = read(tokenizer / 'tokenizer_config.json')
    template_file = tokenizer / 'chat_template.jinja'
    template = template_file.read_text() if template_file.exists() else cfg.get('chat_template')
    if not isinstance(template, str) or not template:
        raise ValueError('Нет единственного шаблона диалога')
    return dict(base_name=baseline['name'], base_revision=baseline['revision'], base_path=str(base),
                tokenizer_path=str(tokenizer), tokenizer_sha256=digest(files),
                template_sha256=hashlib.sha256(template.encode()).hexdigest(), dtype='bfloat16', device='mps',
                generation=dict(system=config['system'], max_new_tokens=config['max_new_tokens'],
                                enable_thinking=config['enable_thinking'], do_sample=False, seed=config['seed'],
                                tokenize=False, add_generation_prompt=True, skip_special_tokens=True,
                                pad_token_id_policy='pad_or_eos'))


def make_manifest(model_id, kind, artifact, runtime, report, provenance, base_hashes):
    files = hashes(artifact)
    return dict(schema_version=1, model_id=model_id, kind=kind, artifact_uri=artifact.as_uri(),
                file_hashes=files, model_sha256=digest(files), runtime=runtime, runtime_sha256=digest(runtime),
                protocol_sha256=report['protocol_sha256'], golden_sha256=report['golden_sha256'],
                base_file_hashes=base_hashes, provenance=provenance)


def source_references(source, root):
    names = ['context/submission/manifest.json', 'evaluation/params.json', 'evaluation/baseline.json',
             'metrics/candidate-model.json', 'metrics/sweep.json', 'metrics/release-decision.json',
             'metrics/quality-gate.json']
    names += [f'metrics/{role}-{suffix}.json' for role in ('baseline', 'candidate') for suffix in ('eval', 'outputs')]
    names += ['evaluation/' + p.name for p in sorted((source / 'evaluation').glob('*.py'))]
    config = read(source / 'evaluation/params.json')
    names += [config[key] for key in ('golden', 'behavioral', 'training_index')]
    refs = {}
    for name in names:
        target = root / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
        refs[name] = {'path': str(target.relative_to(root)), 'sha256': file_hash(target),
                      'original_uri': (source / name).as_uri()}
    return refs


def release_report(manifest, baseline, candidate, config, refs, source, role='candidate'):
    report = baseline if role == 'baseline' else candidate
    decision = gate_result(baseline, report, config['critical_slices'], source)
    result = {key: manifest[key] for key in ('model_id', 'model_sha256', 'runtime_sha256', 'protocol_sha256', 'golden_sha256')}
    result.update(status='complete', passed=decision['passed'], evaluation_run_id=report['evaluation_run_id'],
                  metrics={key: report['overall'][key] for key in ('accuracy', 'format_rate')},
                  quality_gate=decision, sources=refs, source_verified=True,
                  behavioral=report['behavioral']['overall'],
                  failed_behavior=[item['id'] for item in report['behavioral']['per_item'] if not item['correct']])
    if role == 'baseline':
        # Наличие исходного указателя не является решением quality_gate.
        result.update(passed=False, disposition='bootstrap_existing_baseline', newly_accepted=False)
    else:
        result['disposition'] = 'accepted' if result['passed'] else 'rejected'
    return result


def preparation_config(runtime):
    """Настройки кеша с хешем кода реальной стадии подготовки."""
    code_files = {name: file_hash(HERE / name) for name in ('lab.py', 'release.py')}
    code_sha256 = digest(code_files)
    return dict(format='jsonl', schema='golden', tokenizer_revision=runtime['base_revision'],
                tokenizer_sha256=runtime['tokenizer_sha256'], template_sha256=runtime['template_sha256'],
                max_length=32768, code_revision=code_sha256, preparation_code_sha256=code_sha256,
                preparation_code_files=code_files)


def assess(workspace):
    """Вернуть заново проверенный отчёт кандидата; passed с диска не используется."""
    root = Path(workspace).resolve()
    manifest = read(root / 'manifest-candidate.json')
    saved = root / 'sources'
    source_root = manifest['provenance'].get('source_root')
    source = Path(source_root) if source_root else None
    check_originals = source is not None and source.is_dir()
    previous = read(root / 'report-candidate.json')
    refs = previous['sources']
    required = {'MLmodel', 'evaluation/params.json', 'evaluation/baseline.json',
                'evaluation/contract.py', 'evaluation/solution.py', 'evaluation/cli.py',
                'context/submission/manifest.json', 'metrics/candidate-model.json'}
    required |= {f'metrics/{role}-{suffix}.json' for role in ('baseline', 'candidate') for suffix in ('eval', 'outputs')}
    if not required <= refs.keys():
        raise ValueError('Неполный список исходных отчётов')
    for name, ref in refs.items():
        path = (root / ref['path']).resolve()
        expected_path = (saved / name).resolve()
        if (not path.is_relative_to(saved) or path != expected_path
                or file_hash(path) != ref['sha256']):
            raise ValueError('Изменён сохранённый исходный файл: ' + name)
        # original_uri — сведения о происхождении, а не обязательный путь.
        # Для MLmodel достаточно сохранённой копии: исходник мог быть временным.
        if check_originals and name != 'MLmodel' and file_hash(source / name) != ref['sha256']:
            raise ValueError('Изменён исходный файл: ' + name)
    config = read(saved / 'evaluation/params.json')
    for key in ('golden', 'behavioral', 'training_index'):
        name = config[key]
        if name not in refs or not (saved / name).resolve().is_relative_to(saved):
            raise ValueError('Нет сохранённых данных оценки: ' + key)
    # Пересчёт по полному сохранённому набору не зависит от source_root.
    config, _, reports = verify_evaluation(saved)
    baseline = read(root / 'manifest-baseline.json')
    if (manifest['model_id'] != reports['candidate']['model']['run_id']
            or baseline['model_id'] != 'Qwen3-1.7B-' + reports['baseline']['model']['revision']):
        raise ValueError('Чужой model_id')
    for item in (baseline, manifest):
        from urllib.parse import unquote, urlparse
        uri = urlparse(item['artifact_uri'])
        if uri.scheme != 'file' or uri.netloc:
            raise ValueError('Нужен локальный артефакт')
        if item is manifest and Path(unquote(uri.path)).resolve() != (root / 'artifacts/candidate').resolve():
            raise ValueError('Нужна локальная копия кандидата')
        actual = hashes(Path(unquote(uri.path)))
        if actual != item['file_hashes'] or digest(actual) != item['model_sha256']:
            raise ValueError('Изменились файлы артефакта')
        runtime = runtime_for(Path(item['runtime']['base_path']), Path(item['runtime']['tokenizer_path']),
                              reports['baseline']['model'], config)
        if (Path(runtime['tokenizer_path']).resolve() != Path(unquote(uri.path)).resolve()
                or Path(runtime['base_path']).name != runtime['base_revision']):
            raise ValueError('Чужой путь токенизатора или базы')
        if runtime != item['runtime'] or digest(runtime) != item['runtime_sha256']:
            raise ValueError('Runtime отличается от исходной оценки')
        if hashes(Path(runtime['base_path'])) != item['base_file_hashes']:
            raise ValueError('Изменились файлы базы')
        for key in ('protocol_sha256', 'golden_sha256'):
            if item[key] != reports['candidate'][key]:
                raise ValueError('Изменился ' + key)
    import yaml
    # Проверяем сохранённый MLmodel: исходный временный каталог может быть удалён.
    info = yaml.safe_load((root / 'sources/MLmodel').read_text())
    if (info['run_id'] != manifest['model_id'] or info['artifact_path'] != 'model'
            or file_hash(root / 'sources/MLmodel') != manifest['provenance']['mlmodel_sha256']):
        raise ValueError('MLmodel другого training run')
    result = release_report(manifest, reports['baseline'], reports['candidate'], config, refs, saved)
    write(root / 'report-candidate.json', result)
    return result


def prepare(workspace, source=HW67, snapshot=None, candidate_model=None, tracking_uri=None):
    root, source = Path(workspace).resolve(), Path(source).resolve()
    config, rows, reports = verify_evaluation(source)
    if config['max_new_tokens'] != 32 or config['enable_thinking'] is not False:
        raise ValueError('Условия генерации отличаются от прежней оценки')
    baseline_spec = reports['baseline']['model']
    base = snapshot_path(baseline_spec, snapshot)
    base_files = hashes(base)
    mlmodel, adapter, training = obtain_training_artifact(reports['candidate']['model'], candidate_model, tracking_uri)
    adapter_manifest = read(adapter / 'manifest.json')
    adapter_config = read(adapter / 'adapter_config.json')
    if (adapter_manifest['model']['name'] != baseline_spec['name']
            or adapter_manifest['model']['revision'] != baseline_spec['revision']
            or adapter_manifest['model']['dtype'] != 'bfloat16'
            or adapter_config['base_model_name_or_path'] != baseline_spec['name']):
        raise ValueError('Адаптер обучен с другой базой или dtype')
    git_sha = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    provenance = dict(source='hw6-7', source_root=str(source), git_sha=git_sha,
                      baseline_spec_uri=(source / 'evaluation/baseline.json').as_uri(),
                      baseline_spec_sha256=file_hash(source / 'evaluation/baseline.json'),
                      evaluation_reused=True, runtime_evidence='evaluation/cli.py + adapter manifest; device=mps по исходному запуску')
    baseline_runtime = runtime_for(base, base, baseline_spec, config)
    baseline = make_manifest('Qwen3-1.7B-' + baseline_spec['revision'], 'base', base, baseline_runtime,
                             reports['baseline'], {**provenance, 'bootstrap': 'bootstrap_existing_baseline',
                                                   'operation': 'bootstrap_existing_baseline', 'newly_accepted': False}, base_files)
    accepted = root / 'accepted.json'
    if accepted.exists() and read(accepted) != baseline:
        raise ValueError('prepare не заменяет существующую принятую версию')
    dest = root / 'artifacts/candidate'
    original_hashes = hashes(adapter)
    if dest.exists():
        if hashes(dest) != original_hashes:
            raise ValueError('Существующая копия кандидата отличается')
    else:
        shutil.copytree(adapter, dest)
    if hashes(dest) != original_hashes:
        raise ValueError('Ошибка копирования кандидата')
    candidate_runtime = runtime_for(base, dest, baseline_spec, config)
    candidate = make_manifest(reports['candidate']['model']['run_id'], 'adapter', dest, candidate_runtime,
                              reports['candidate'], {**provenance, **training}, base_files)
    refs = source_references(source, root)
    shutil.copyfile(mlmodel, root / 'sources/MLmodel')
    refs['MLmodel'] = dict(path='sources/MLmodel', sha256=file_hash(mlmodel), original_uri=mlmodel.as_uri())
    for role, manifest in [('baseline', baseline), ('candidate', candidate)]:
        write(root / f'manifest-{role}.json', manifest)
        write(root / f'report-{role}.json', release_report(manifest, reports['baseline'], reports['candidate'], config, refs, source, role))
    for name in ('input.jsonl', 'input.txt'):
        shutil.copyfile(source / config['golden'], root / name)
    write(root / 'config.json', preparation_config(baseline_runtime))
    notes = (
        'Сохранена исходная резервная версия Qwen3-1.7B из evaluation/baseline.json ДЗ 6–7.\n'
        f'Git HEAD при подготовке: {git_sha}. Хеш исходного baseline.json: {provenance["baseline_spec_sha256"]}.\n'
        'provenance: bootstrap_existing_baseline. Это не новая приёмка и не оценка PASS.\n'
        'quality_gate вызван повторно после пересчёта score по исходным outputs.\n'
        'База: accuracy 49/50; behavior-refusal не пройден. report-baseline.passed=false.\n'
        'Кандидат: accuracy 48/50; ухудшение mathematics; behavior-refusal не пройден.\n'
        'Пороги не менялись. accepted.json сохраняет базу; кандидат не публикуется.\n'
        'Runtime: bfloat16, MPS, system из evaluation/params.json, max_new_tokens=32,\n'
        'enable_thinking=false, do_sample=false; не generation max40 из training manifest.\n'
        'Исходный eval не сохранял отдельный runtime hash: конфигурация восстановлена из\n'
        'закреплённого cli.py, params.json и manifest адаптера; отдельного доказательства\n'
        'переменных HW6_DEVICE/HW6_DTYPE прежнего процесса нет. Модель повторно не запускалась.\n'
        'Git HEAD фиксирует состояние при подготовке, не доказывает чистоту прежнего запуска.\n'
        'Старые хеши проверены по submission/manifest.json (целостность, не цифровая подпись).\n'
        'preflight-adapter не использован: preflight.py обучал один шаг, выбранный run — четыре.\n'
        f'Источник кандидата: {training["mlmodel_uri"]}; run_id={training["training_run_id"]}.\n'
        'MLmodel и полные исходные отчёты/ответы скопированы в sources; веса — artifacts/candidate.\n'
        'Хеши не являются историческим хешем весов на момент eval: прежний отчёт его не содержал.\n'
        'Связь с весами подтверждается run_id в MLmodel и совпадением сохранённых копий.\n'
    )
    (root.parent / 'real-notes.txt').write_text(notes, encoding='utf-8')
    if not accepted.exists():
        write(accepted, baseline)
    return dict(workspace=str(root), baseline_id=baseline['model_id'], candidate_id=candidate['model_id'],
                candidate_passed=read(root / 'report-candidate.json')['passed'], baseline_newly_accepted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'assess'])
    parser.add_argument('--workspace', type=Path, default=HERE / 'context/real')
    parser.add_argument('--source', type=Path, default=HW67)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--candidate-model', type=Path, help='Каталог MLmodel выбранного training run')
    parser.add_argument('--tracking-uri', help='Явно разрешить чтение MLflow; по умолчанию только локальные файлы')
    args = parser.parse_args()
    result = (prepare(args.workspace, args.source, args.snapshot, args.candidate_model, args.tracking_uri)
              if args.command == 'prepare' else assess(args.workspace))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
