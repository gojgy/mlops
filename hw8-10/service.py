"""FastAPI: закреплённая версия, прогрев и проверка до вычисления."""
import argparse
import copy
import hashlib
import json
import logging
import math
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from urllib.parse import unquote, urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, ValidationError
from starlette.concurrency import run_in_threadpool

import release


MAX_BODY_BYTES = 65536
MAX_PROMPT_CHARS = 8192
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt',
                   'special_tokens_map.json', 'added_tokens.json', 'chat_template.jinja')


def verify_tokenizer(path, runtime):
    files = {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
             for name in TOKENIZER_FILES if (path / name).is_file()}
    canonical = json.dumps(files, sort_keys=True, separators=(',', ':'),
                           ensure_ascii=False, allow_nan=False).encode('utf-8')
    if not files or hashlib.sha256(canonical).hexdigest() != runtime['tokenizer_sha256']:
        raise ValueError('Изменены файлы токенизатора')
    if (path / 'chat_template.jinja').is_file():
        template = (path / 'chat_template.jinja').read_bytes().decode('utf-8')
    else:
        template = json.loads((path / 'tokenizer_config.json').read_text(encoding='utf-8')).get('chat_template')
    if not isinstance(template, str):
        raise ValueError('Нужен единственный шаблон токенизатора')
    if hashlib.sha256(template.encode('utf-8')).hexdigest() != runtime['template_sha256']:
        raise ValueError('Изменён шаблон токенизатора')


def local_path(value, root):
    """Относительный путь считается от root; file:// разрешён."""
    parsed = urlparse(str(value))
    if parsed.scheme:
        if (parsed.scheme != 'file' or parsed.netloc not in ('', 'localhost')
                or parsed.query or parsed.fragment or not parsed.path.startswith('/')):
            raise ValueError('Нужен локальный путь')
        path = Path(unquote(parsed.path))
    else:
        path = Path(value)
    return (path if path.is_absolute() else Path(root) / path).resolve()


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prompt: Annotated[StrictStr, Field(min_length=1, max_length=MAX_PROMPT_CHARS)]
    max_new_tokens: Annotated[StrictInt, Field(ge=1, le=32)] | None = None


class StubBackend:
    """Имитатор только для синтетических опытов."""
    def __init__(self, manifest):
        self.loaded_id = manifest['model_id']
        self.runtime_sha256 = manifest['runtime_sha256']
        self.calls = 0

    def warmup(self):
        pass

    def generate(self, prompt, limit):
        self.calls += 1
        time.sleep(.01)
        return dict(answer=f'STUB ONLY: {self.loaded_id}', output_tokens=1)


class TransformersBackend:
    """База Qwen3 и необязательный PEFT-адаптер из manifest."""
    def __init__(self, manifest, root):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.loaded_id = manifest['model_id']
        self.runtime_sha256 = manifest['runtime_sha256']
        self.calls = 0
        self.torch = torch
        runtime = manifest['runtime']
        self.generation = copy.deepcopy(runtime['generation'])
        revision = runtime['base_revision']
        if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise ValueError('Нужна точная ревизия базы')
        for key in ('system', 'max_new_tokens', 'enable_thinking', 'do_sample'):
            if key not in self.generation:
                raise ValueError(f'Нет generation.{key}')
        if not isinstance(self.generation['system'], str):
            raise ValueError('Неверная системная инструкция')
        if type(self.generation['enable_thinking']) is not bool or self.generation['do_sample'] is not False:
            raise ValueError('Поддерживается do_sample=false')
        for key, expected in (('tokenize', False), ('add_generation_prompt', True),
                              ('skip_special_tokens', True)):
            if key in self.generation and self.generation[key] is not expected:
                raise ValueError(f'Несовместимый generation.{key}')
        if self.generation.get('pad_token_id_policy', 'pad_or_eos') != 'pad_or_eos':
            raise ValueError('Несовместимый pad_token_id_policy')
        dtype = runtime['dtype']
        if dtype not in ('float32', 'float16', 'bfloat16'):
            raise ValueError('Неверный dtype')
        device = runtime.get('device', 'auto')
        if device == 'auto':
            device = ('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')
        self.device = torch.device(device)
        artifact = local_path(manifest['artifact_uri'], root)
        tokenizer_path = local_path(runtime['tokenizer_path'], artifact)
        base_path = None
        if runtime.get('base_path'):
            base_path = local_path(runtime['base_path'], root)
            if base_path.name != revision:
                raise ValueError('Путь базы не соответствует ревизии')
        verify_tokenizer(tokenizer_path, runtime)
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
        template = self.tokenizer.chat_template
        if not isinstance(template, str):
            raise ValueError('Нужен единственный шаблон токенизатора')
        if hashlib.sha256(template.encode('utf-8')).hexdigest() != runtime['template_sha256']:
            raise ValueError('Изменён шаблон токенизатора')
        if 'seed' in self.generation:
            from transformers import set_seed
            if type(self.generation['seed']) is not int:
                raise ValueError('Неверный seed')
            set_seed(self.generation['seed'])
        source = runtime['base_name']
        options = dict(revision=revision, dtype=getattr(torch, dtype))
        if base_path is not None:
            source = str(base_path)
            options['local_files_only'] = True
        model = AutoModelForCausalLM.from_pretrained(source, **options)
        if (artifact / 'adapter_config.json').is_file():
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(artifact), local_files_only=True)
        self.model = model.to(self.device).eval()

    def _generate(self, prompt, limit):
        text = self.tokenizer.apply_chat_template(
            [{'role': 'system', 'content': self.generation['system']},
             {'role': 'user', 'content': prompt}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=self.generation['enable_thinking'])
        batch = self.tokenizer(text, return_tensors='pt').to(self.device)
        input_tokens = batch['input_ids'].shape[1]
        context_length = getattr(self.model.config, 'max_position_embeddings', None)
        if isinstance(context_length, int) and input_tokens + limit > context_length:
            raise ValueError('Превышена длина контекста')
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        with self.torch.inference_mode():
            output = self.model.generate(**batch, max_new_tokens=limit, do_sample=False,
                                         pad_token_id=pad_token_id)
        tokens = output[0, input_tokens:]
        return dict(answer=self.tokenizer.decode(tokens, skip_special_tokens=True),
                    output_tokens=len(tokens))

    def warmup(self):
        self._generate('Привет', self.generation['max_new_tokens'])

    def generate(self, prompt, limit):
        self.calls += 1
        return self._generate(prompt, limit)


class ServiceState:
    def __init__(self, root, warmup_seconds=5, backend_factory=None, manifest_path=None,
                 backend_kind='auto'):
        if not math.isfinite(warmup_seconds) or warmup_seconds < 0:
            raise ValueError('Неверное время прогрева')
        self.root = Path(root).resolve()
        self.warmup_seconds = warmup_seconds
        self.backend_factory = backend_factory
        self.manifest_path = manifest_path
        self.backend_kind = backend_kind
        self.backend = None
        self.manifest = None
        self.error = None
        self.limit = 32
        self.fixed_limit = False
        self.warmup_finished = threading.Event()
        self.stopped = threading.Event()
        self.compute = threading.Lock()
        self.slots = threading.BoundedSemaphore(2)

    @property
    def ready(self):
        return self.warmup_finished.is_set() and not self.stopped.is_set()

    def load(self):
        try:
            path = self.manifest_path
            if path is None:
                path = self.root / 'accepted.json'
                if not path.exists():
                    path = self.root / 'baseline-manifest.json'
            else:
                path = local_path(path, self.root)
            manifest = json.loads(path.read_text(encoding='utf-8'))
            runtime_bytes = json.dumps(manifest['runtime'], sort_keys=True, separators=(',', ':'),
                                       ensure_ascii=False, allow_nan=False).encode()
            if hashlib.sha256(runtime_bytes).hexdigest() != manifest['runtime_sha256']:
                raise ValueError('Изменён runtime')
            release.verify_artifact(manifest, self.root)
            self.manifest = copy.deepcopy(manifest)
            self.limit = manifest['runtime']['generation']['max_new_tokens']
            if type(self.limit) is not int or not 1 <= self.limit <= 32:
                raise ValueError('Неверный лимит протокола')
            kind = self.backend_kind
            if kind == 'auto':
                kind = ('stub' if manifest['runtime']['base_revision'] == 'fixture-only'
                        else 'transformers')
            self.fixed_limit = kind == 'transformers'
            if self.backend_factory is not None:
                self.backend = self.backend_factory(copy.deepcopy(manifest))
            elif kind == 'stub':
                self.backend = StubBackend(manifest)
            else:
                self.backend = TransformersBackend(manifest, self.root)
            warmup = getattr(self.backend, 'warmup', None)
            if warmup is not None:
                warmup()
            if not self.stopped.wait(self.warmup_seconds):
                self.warmup_finished.set()
        except Exception as exc:
            self.error = type(exc).__name__
            logging.exception('Не удалось загрузить модель')

    def generate(self, request, received_at=None):
        received_at = time.perf_counter() if received_at is None else received_at
        if not request.prompt.strip():
            raise HTTPException(422, 'Пустой prompt')
        limit = request.max_new_tokens if request.max_new_tokens is not None else self.limit
        if limit > self.limit or (self.fixed_limit and limit != self.limit):
            raise HTTPException(422, 'Лимит отличается от протокола')
        if not self.ready:
            raise HTTPException(503, 'Модель не готова')
        if not self.slots.acquire(blocking=False):
            raise HTTPException(429, 'Модель занята')
        try:
            with self.compute:
                if not self.ready:
                    raise HTTPException(503, 'Модель не готова')
                started = time.perf_counter()
                try:
                    result = self.backend.generate(request.prompt, limit)
                except ValueError as exc:
                    raise HTTPException(422, str(exc)) from exc
                finished = time.perf_counter()
                return dict(result, model_id=self.manifest['model_id'],
                            runtime_sha256=self.manifest['runtime_sha256'],
                            request_id=str(uuid.uuid4()),
                            queue_seconds=started - received_at,
                            generation_seconds=finished - started,
                            server_seconds=finished - received_at)
        finally:
            self.slots.release()


def create_app(workspace=Path('.lab'), warmup_seconds=5, backend_factory=None,
               manifest_path=None, backend_kind='auto', debug=False):
    """backend_factory(manifest) позволяет внедрить StubBackend без весов."""
    state = ServiceState(workspace, warmup_seconds, backend_factory, manifest_path, backend_kind)

    @asynccontextmanager
    async def lifespan(app):
        thread = threading.Thread(target=state.load, daemon=True)
        thread.start()
        try:
            yield
        finally:
            state.stopped.set()
            await run_in_threadpool(thread.join, 1)

    app = FastAPI(lifespan=lifespan)
    app.state.service = state

    @app.get('/health')
    def health():
        return dict(alive=True)

    @app.get('/ready')
    def ready():
        return JSONResponse(dict(ready=state.ready), status_code=200 if state.ready else 503)

    @app.post('/generate')
    async def generate(request: Request):
        received_at = time.perf_counter()
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_BODY_BYTES:
                raise HTTPException(413, 'Слишком большой запрос')
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeDecodeError) as exc:
            raise HTTPException(400, 'Неверный JSON') from exc
        try:
            validated = GenerateRequest.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(422, 'Неверный запрос') from exc
        return await run_in_threadpool(state.generate, validated, received_at)

    if debug:
        @app.get('/debug')
        def diagnostics():
            return dict(loaded_id=state.manifest['model_id'] if state.backend else None,
                        calls=getattr(state.backend, 'calls', 0),
                        load_failed=state.error is not None,
                        warmup_finished=state.warmup_finished.is_set())

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=Path('.lab'))
    parser.add_argument('--port', type=int, default=8091)
    parser.add_argument('--warmup-seconds', type=float, default=5)
    parser.add_argument('--backend', choices=('auto', 'stub', 'transformers'), default='auto')
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.warmup_seconds) or args.warmup_seconds < 0:
        parser.error('Неверное время прогрева')
    # CLI сохраняет диагностику для опытов; create_app её не включает.
    app = create_app(args.workspace, args.warmup_seconds, manifest_path=args.manifest,
                     backend_kind=args.backend, debug=True)
    uvicorn.run(app, host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
