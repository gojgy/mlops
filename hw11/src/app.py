"""HTTP-контракт ДЗ 8–10 с выдачей ответа частями и метриками."""
import asyncio
from contextlib import asynccontextmanager
import json
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

from src.backend import Settings, SimulatedBackend

BUCKETS = (.025, .05, .1, .2, .4, .8, 1.6, 3.2, 6.4, 12.8, 25.6, 60)
REQUESTS = Counter('inference_requests', 'Законченные попытки', ['status'])
FIRST = Histogram('inference_first_chunk_seconds', 'До первой части успешного ответа', buckets=BUCKETS)
TOTAL = Histogram('inference_total_seconds', 'Полный успешный ответ', buckets=BUCKETS)
QUEUE = Histogram('inference_queue_seconds', 'Ожидание успешного запроса', buckets=BUCKETS)
RUNNING = Gauge('inference_running', 'Выполняемые запросы')
WAITING = Gauge('inference_waiting', 'Ожидающие запросы')
CACHE = Counter('inference_cache', 'Обращения к имитируемому кешу', ['result'])
log = logging.getLogger('uvicorn.error')


class Message(BaseModel):
    model_config = ConfigDict(extra='forbid')
    role: StrictStr
    content: StrictStr = Field(max_length=4096)


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prompt: StrictStr = Field(min_length=1, max_length=8192)
    prefix: StrictStr = Field(default='default', max_length=128)
    history: list[Message] = Field(default_factory=list, max_length=16)
    parts: StrictInt = Field(default=4, ge=1, le=256)
    chunk_seconds: float | None = Field(default=None, ge=0, le=.5)


settings = Settings()
backend = SimulatedBackend(settings)


def metadata(request_id):
    return dict(request_id=request_id, pod=settings.pod, model_id=backend.model_id,
                backend_mode='simulated')


def event(name, request_id, **fields):
    log.info(json.dumps(dict(event=name, time=time.time(), **metadata(request_id), **fields), ensure_ascii=False))


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(backend.warmup())
    yield
    if not task.done():
        task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get('/health')
def health():
    return dict(alive=True, backend_mode='simulated')


@app.get('/ready')
def ready():
    return JSONResponse(dict(ready=backend.ready, backend_mode='simulated'),
                        status_code=200 if backend.ready else 503)


@app.get('/metrics')
def metrics():
    RUNNING.set(backend.running)
    WAITING.set(backend.waiting)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post('/admin/cache/reset')
def reset():
    if backend.running or backend.waiting:
        raise HTTPException(409, 'Есть активные запросы')
    backend.cache.clear()
    return dict(cache_size=0, pod=settings.pod)


@app.post('/generate')
async def generate(body: GenerateRequest, request: Request):
    request_id = request.headers.get('X-Request-ID') or str(uuid.uuid4())
    if not body.prompt.strip():
        raise HTTPException(422, 'Пустой prompt')
    started = time.monotonic()
    event('received', request_id, warmed=backend.ready)
    error = backend.admit()
    if error:
        REQUESTS.labels('unready' if error == 503 else 'rejected').inc()
        event('rejected', request_id, status=error, warmed=backend.ready)
        return JSONResponse(dict(error='not ready' if error == 503 else 'queue full', **metadata(request_id)), status_code=error)

    async def stream():
        acquired = False
        first_at = None
        queued = None
        complete = False
        try:
            queued = await backend.acquire()
            acquired = True
            hit, work = await backend.prepare(body.prefix)
            CACHE.labels('hit' if hit else 'miss').inc()
            text = ' | '.join([m.content for m in body.history] + [body.prompt])
            size = max(1, (len(text) + body.parts - 1) // body.parts)
            delay = settings.chunk_seconds if body.chunk_seconds is None else body.chunk_seconds
            for i in range(body.parts):
                if await request.is_disconnected():
                    break
                chunk = text[i * size:(i + 1) * size] or '.'
                if first_at is None:
                    first_at = time.monotonic() - started
                yield json.dumps(dict(type='chunk', text=chunk, sequence=i, cache_hit=hit,
                                      **metadata(request_id))) + '\n'
                if i < body.parts - 1:
                    await asyncio.sleep(delay)
            else:
                complete = True
                yield json.dumps(dict(type='end', parts=body.parts, **metadata(request_id))) + '\n'
        finally:
            if acquired:
                backend.release()
            else:
                backend.waiting -= 1
            status = 'ok' if complete else 'aborted'
            REQUESTS.labels(status).inc()
            if complete:
                FIRST.observe(first_at)
                TOTAL.observe(time.monotonic() - started)
                QUEUE.observe(max(0, queued))
            event('finished', request_id, status=status, first_seconds=first_at,
                  total_seconds=time.monotonic() - started)

    return StreamingResponse(stream(), media_type='application/x-ndjson')
