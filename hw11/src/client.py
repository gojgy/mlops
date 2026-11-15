"""Открытая нагрузка из клиентского Pod. Одна строка на каждую попытку."""
import asyncio
import json
import socket
import sys
import time

import httpx


async def attempt(client, item, config, origin):
    planned = origin + item['offset']
    await asyncio.sleep(max(0, planned - time.monotonic()))
    started = time.monotonic()
    sent_utc = time.time()
    row = dict(run=config['run'], logical_id=item['id'], request_id=config['run'] + '-' + item['id'],
               client=config['client'], client_ip=socket.gethostbyname(socket.gethostname()),
               policy=config.get('policy'), traffic=config.get('traffic'), repeat=config.get('repeat'),
               scheduled_monotonic=planned, sent_monotonic=started, sent_utc=sent_utc,
               scheduled_utc=config['start_epoch'] + item['offset'],
               send_lateness_seconds=started - planned, first_monotonic=None, end_monotonic=None,
               status_code=None, outcome='error', error=None, pod=None, model_id=None,
               backend_mode=None, parts=0, chunk_times=[], texts=[], cache_hit=None,
               prompt_chars=len(item['body']['prompt']), requested_parts=item['body'].get('parts', 4))
    async def stream():
        async with client.stream('POST', config['url'] + '/generate', json=item['body'],
                                 headers={'X-Request-ID': row['request_id'], 'Connection': 'close'}) as response:
            row['status_code'] = response.status_code
            if response.status_code != 200:
                try:
                    body = json.loads(await response.aread())
                    for key in ('pod', 'model_id', 'backend_mode'):
                        row[key] = body.get(key)
                except (ValueError, TypeError):
                    pass
                row['outcome'] = 'rejected' if response.status_code == 429 else 'http_error'
                return
            ended = False
            async for line in response.aiter_lines():
                data = json.loads(line)
                if data['request_id'] != row['request_id']:
                    raise ValueError('Чужой request_id')
                for key in ('pod', 'model_id', 'backend_mode'):
                    if row[key] not in (None, data[key]):
                        raise ValueError('Метаданные изменились внутри ответа')
                    row[key] = data[key]
                if data['type'] == 'chunk':
                    now = time.monotonic()
                    if row['first_monotonic'] is None:
                        row['first_monotonic'] = now
                    row['chunk_times'].append(now)
                    row['texts'].append(data['text'])
                    row['parts'] += 1
                    row['cache_hit'] = data['cache_hit']
                elif data['type'] == 'end':
                    ended = True
            row['outcome'] = 'success' if ended and row['parts'] == row['requested_parts'] else 'aborted'
    try:
        await asyncio.wait_for(stream(), timeout=config.get('timeout', 15))
    except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
        row.update(outcome='timeout', error=type(exc).__name__)
    except Exception as exc:
        row.update(outcome='aborted' if row['status_code'] == 200 else 'error',
                   error=f'{type(exc).__name__}: {exc}')
    row['end_monotonic'] = time.monotonic()
    row['end_utc'] = time.time()
    row['total_seconds'] = row['end_monotonic'] - started
    row['ttfc_seconds'] = None if row['first_monotonic'] is None else row['first_monotonic'] - started
    times = row['chunk_times']
    row['inter_chunk_seconds'] = None if len(times) < 2 else (times[-1] - times[0]) / (len(times) - 1)
    print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
    return row


async def run(config):
    # Сдвиг монотонных часов сохраняет опоздание запуска самого генератора.
    origin = time.monotonic() + config['start_epoch'] - time.time()
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=128, max_keepalive_connections=0),
                                 timeout=httpx.Timeout(15, connect=3), trust_env=False) as client:
        return await asyncio.gather(*(attempt(client, item, config, origin) for item in config['requests']))


if __name__ == '__main__':
    asyncio.run(run(json.load(sys.stdin)))
