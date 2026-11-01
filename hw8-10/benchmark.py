"""Восемь одинаковых по составу HTTP-запросов при параллельности 1 и 2.

Сервер должен быть загружен и прогрет пользователем до запуска команды.
Все попытки, включая ошибки и тайм-ауты, сохраняются без повторов.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler
import uuid


HERE = Path(__file__).resolve().parent


def request_json(url, timeout, payload=None):
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
    request = Request(url, data=body, headers={'Content-Type': 'application/json'},
                      method='GET' if body is None else 'POST')
    # Локальные запросы не должны зависеть от системного HTTP proxy.
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            status, raw = response.status, response.read()
    except HTTPError as exc:
        status, raw = exc.code, exc.read()
    text = raw.decode('utf-8', errors='replace')
    try:
        value = json.loads(text)
    except ValueError:
        value = {'unparsed_body': text}
    return status, value


def finite_seconds(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def attempt(url, timeout, row, limit, concurrency, index, run_id, expected):
    payload = dict(prompt=row['input'], max_new_tokens=limit)
    result = dict(run_id=run_id, attempt_id=f'{run_id}-c{concurrency}-{index:02d}',
                  input_id=row['id'], concurrency=concurrency, index=index,
                  started_at=datetime.now(timezone.utc).isoformat(),
                  request=payload, status=None, success=False, error=None,
                  request_id=None, model_id=None, runtime_sha256=None, output_tokens=None,
                  generation_seconds=None, queue_seconds=None, server_seconds=None)
    started = time.perf_counter()
    try:
        status, body = request_json(url + '/generate', timeout, payload)
        result.update(status=status, response=body)
        if not isinstance(body, dict):
            raise ValueError('Ответ API не является объектом JSON')
        for key in ('request_id', 'model_id', 'runtime_sha256', 'output_tokens',
                    'generation_seconds', 'queue_seconds', 'server_seconds'):
            result[key] = body.get(key)
        if status != 200:
            raise ValueError(f'HTTP {status}: {body}')
        for key in ('request_id', 'model_id', 'runtime_sha256'):
            if not isinstance(body.get(key), str) or not body[key]:
                raise ValueError('Нет поля ответа: ' + key)
        answer = body.get('answer', body.get('text'))
        if not isinstance(answer, str) or 'STUB ONLY' in answer:
            raise ValueError('Нет настоящего текстового ответа')
        if type(body.get('output_tokens')) is not int or not 0 <= body['output_tokens'] <= limit:
            raise ValueError('Некорректное число токенов')
        if not finite_seconds(body.get('generation_seconds')):
            raise ValueError('Нет длительности генерации')
        for key in ('queue_seconds', 'server_seconds'):
            if body.get(key) is not None and not finite_seconds(body[key]):
                raise ValueError('Некорректная длительность: ' + key)
        for key in ('model_id', 'runtime_sha256'):
            if body[key] != expected[key]:
                raise ValueError('API загрузил другую версию: ' + key)
        result['success'] = True
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        result['client_seconds'] = time.perf_counter() - started
    return result


def summarize(attempts, concurrency, duration, run_id):
    good = [item for item in attempts if item['success']]
    bad = [item for item in attempts if not item['success']]
    latencies = [item['client_seconds'] for item in good]
    errors = [item['client_seconds'] for item in bad]
    return dict(run_id=run_id, concurrency=concurrency, n=len(attempts), successes=len(good),
                errors=len(bad), batch_seconds=duration, successful_requests_per_second=len(good) / duration,
                success_mean_client_seconds=statistics.mean(latencies) if latencies else None,
                success_max_client_seconds=max(latencies) if latencies else None,
                error_mean_client_seconds=statistics.mean(errors) if errors else None,
                error_max_client_seconds=max(errors) if errors else None,
                success_mean_generation_seconds=statistics.mean([item['generation_seconds'] for item in good]) if good else None,
                output_tokens=sum(item['output_tokens'] for item in good),
                model_ids=json.dumps(sorted({item['model_id'] for item in good})),
                runtime_sha256=json.dumps(sorted({item['runtime_sha256'] for item in good})))


def run(url, inputs, manifest_path, output, timeout=180):
    if not finite_seconds(timeout) or timeout == 0:
        raise ValueError('timeout должен быть положительным конечным числом')
    inputs, manifest_path, output = Path(inputs), Path(manifest_path), Path(output)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('synthetic') or manifest['runtime']['generation']['do_sample'] is not False:
        raise ValueError('Нужна реальная модель с прежней greedy-генерацией')
    limit = manifest['runtime']['generation']['max_new_tokens']
    if type(limit) is not int or limit != 32:
        raise ValueError('Нужен прежний лимит 32 токена')
    rows = [json.loads(line) for line in inputs.read_text().splitlines() if line.strip()][:8]
    if len(rows) != 8 or len({row['id'] for row in rows}) != 8:
        raise ValueError('Нужны восемь запросов с уникальными id')
    if any(not isinstance(row.get('input'), str) or not row['input'].strip() for row in rows):
        raise ValueError('Пустой input')
    status, ready = request_json(url.rstrip('/') + '/ready', min(timeout, 10))
    if status != 200 or not isinstance(ready, dict) or ready.get('ready') is not True:
        raise ValueError('Сервис ещё не готов; сначала завершите загрузку и прогрев')
    # Новая папка исключает случайное стирание предыдущих измерений.
    output.mkdir(parents=True, exist_ok=False)
    run_id = uuid.uuid4().hex
    metadata = dict(run_id=run_id, url=url, inputs=str(inputs.resolve()),
                    input_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),
                    manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                    model_id=manifest['model_id'], runtime_sha256=manifest['runtime_sha256'],
                    request_ids=[row['id'] for row in rows], max_new_tokens=limit,
                    timeout_seconds=timeout, concurrency=[1, 2], retries=0,
                    started_at=datetime.now(timezone.utc).isoformat(),
                    note='Полная клиентская задержка; TTFT, ITL и устойчивый p95 не измеряются.')
    (output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    summaries = []
    with (output / 'attempts.jsonl').open('w', encoding='utf-8', buffering=1) as journal:
        with (output / 'summary.csv').open('w', encoding='utf-8', newline='') as table:
            writer = None
            for concurrency in (1, 2):
                measured = []
                started = time.perf_counter()
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    pending = [pool.submit(attempt, url.rstrip('/'), timeout, row, limit,
                                           concurrency, i, run_id, manifest) for i, row in enumerate(rows, 1)]
                    for future in as_completed(pending):
                        result = future.result()
                        measured.append(result)
                        journal.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                        journal.flush()
                summary = summarize(measured, concurrency, time.perf_counter() - started, run_id)
                if writer is None:
                    writer = csv.DictWriter(table, fieldnames=list(summary))
                    writer.writeheader()
                writer.writerow(summary)
                table.flush()
                summaries.append(summary)
                print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8091')
    parser.add_argument('--input', type=Path, default=HERE / 'context/real/input.jsonl')
    parser.add_argument('--manifest', type=Path, default=HERE / 'context/real/accepted.json')
    parser.add_argument('--output', type=Path, default=HERE / 'context/benchmark')
    parser.add_argument('--timeout', type=float, default=180)
    args = parser.parse_args()
    summaries = run(args.url, args.input, args.manifest, args.output, args.timeout)
    if any(row['errors'] for row in summaries):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
