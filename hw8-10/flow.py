"""Подготовка, получение готового артефакта, допуск и выпуск."""
import argparse
import json
import inspect
from pathlib import Path

from prefect import flow, task
from prefect.cache_policies import NO_CACHE

import lab
import release


def preparation_key(context, parameters):
    root = Path(parameters['workspace'])
    config = dict(lab.read(root / 'config.json'))
    config['preparation_code_sha256'] = release.digest(
        (inspect.getsource(lab.prepare) + inspect.getsource(lab._row_count)).encode())
    return release.cache_key(root / 'input.txt', config)


@task(persist_result=True, cache_key_fn=preparation_key)
def prepare(workspace):
    root = Path(workspace)
    result = lab.prepare(root)
    return {'sha256': result['input_sha256'], 'rows': result['rows']}


@task(persist_result=True, cache_policy=NO_CACHE)
def obtain(workspace, manifest_name):
    root = Path(workspace)
    manifest = lab.read(root / manifest_name)
    release.verify_artifact(manifest, root)
    return {'uri': str(root / manifest_name), 'manifest_sha256': release.digest(release.canonical(manifest)), 'model_id': manifest['model_id'],
            'model_sha256': manifest['model_sha256']}


@task(persist_result=True, cache_policy=NO_CACHE)
def assess(workspace, artifact, report_name):
    root = Path(workspace)
    manifest = lab.read(artifact['uri'])
    if release.digest(release.canonical(manifest)) != artifact['manifest_sha256']:
        raise ValueError('Manifest изменён после проверки')
    if manifest['runtime'].get('base_revision') != 'fixture-only':
        import real_release
        report = real_release.assess(root)
    else:
        report = lab.read(root / report_name)
    release.validate_release(manifest, report)
    return {'report': report, 'evaluation_run_id': report['evaluation_run_id']}


@task(persist_result=True, cache_policy=NO_CACHE)
def publish(workspace, artifact, decision):
    root = Path(workspace)
    manifest = lab.read(artifact['uri'])
    if release.digest(release.canonical(manifest)) != artifact['manifest_sha256']:
        raise ValueError('Manifest изменён до публикации')
    return lab.publish_once(root, manifest, decision['report'])


@flow(name='model-release', persist_result=True)
def run(workspace, manifest_name='manifest-B.json', report_name='report-B.json', fail_after_prepare=False):
    root = str(Path(workspace).resolve())
    preparation = prepare.submit(root)
    prepared = preparation.result()
    print('PREPARATION_STATE', preparation.state.name, flush=True)
    if fail_after_prepare:
        raise RuntimeError('Инъекция сбоя после подготовки')
    artifact = obtain(root, manifest_name)
    decision = assess(root, artifact, report_name)
    result = publish(root, artifact, decision)
    return {'prepared': prepared, 'artifact': artifact, 'result': result}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--manifest', default='manifest-B.json')
    parser.add_argument('--report', default='report-B.json')
    parser.add_argument('--fail-after-prepare', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(str(args.workspace), args.manifest, args.report, args.fail_after_prepare), ensure_ascii=False))
