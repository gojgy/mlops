"""Проверки сервиса через отдельный процесс и локальный HTTP."""
import hashlib
import json
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class ServiceHTTPTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifests = {}
        for version in ('A', 'B'):
            artifact = self.root / f'model-{version}.txt'
            artifact.write_text(f'STUB ONLY: {version}\n')
            files = {artifact.name: hashlib.sha256(artifact.read_bytes()).hexdigest()}
            runtime = dict(base_revision='fixture-only', tokenizer_sha256='a' * 64,
                           template_sha256='b' * 64, dtype='float32',
                           generation=dict(max_new_tokens=32, do_sample=False))
            # Разные runtime проверяют также соответствие метаданных ответу.
            runtime['tokenizer_sha256'] = ('a' if version == 'A' else 'c') * 64
            self.manifests[version] = dict(
                schema_version=1, model_id=f'fixture-{version}', artifact_uri=artifact.name,
                file_hashes=files, model_sha256=digest(files), runtime=runtime,
                runtime_sha256=digest(runtime), protocol_sha256='d' * 64,
                golden_sha256='e' * 64)
        self.accept('A')

    def accept(self, version):
        temporary = self.root / 'accepted.tmp'
        temporary.write_text(json.dumps(self.manifests[version]))
        temporary.replace(self.root / 'accepted.json')

    @contextmanager
    def server(self, warmup=0):
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            port = reservation.getsockname()[1]
        self.url = f'http://127.0.0.1:{port}'
        with tempfile.TemporaryFile(mode='w+') as log:
            process = subprocess.Popen(
                [sys.executable, str(PROJECT / 'service.py'), '--workspace', str(self.root),
                 '--port', str(port), '--warmup-seconds', str(warmup)],
                stdout=log, stderr=subprocess.STDOUT, cwd=PROJECT)
            try:
                deadline = time.monotonic() + 15
                while True:
                    if process.poll() is not None:
                        log.seek(0)
                        self.fail(f'Сервис завершился: {log.read()}')
                    try:
                        if self.request('/health')[0] == 200:
                            break
                    except (URLError, TimeoutError):
                        pass
                    if time.monotonic() > deadline:
                        log.seek(0)
                        self.fail(f'Сервис не запустился: {log.read()}')
                    time.sleep(.02)
                yield
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    def request(self, path, payload=None, raw=None):
        body = raw if raw is not None else (json.dumps(payload).encode() if payload is not None else None)
        request = Request(self.url + path, data=body,
                          headers={'Content-Type': 'application/json'})
        try:
            response = urlopen(request, timeout=5)
        except HTTPError as error:
            response = error
        with response:
            return response.status, json.load(response)

    def wait_ready(self):
        deadline = time.monotonic() + 10
        while self.request('/ready')[0] != 200:
            self.assertLess(time.monotonic(), deadline, 'Прогрев не завершился')
            time.sleep(.02)

    def assert_version(self, version):
        status, result = self.request('/generate', dict(prompt='Привет', max_new_tokens=32))
        self.assertEqual(status, 200, result)
        self.assertEqual(result['answer'], f'STUB ONLY: fixture-{version}')
        self.assertEqual(result['model_id'], f'fixture-{version}')
        self.assertEqual(result['runtime_sha256'], self.manifests[version]['runtime_sha256'])
        self.assertTrue(result['request_id'])
        self.assertGreaterEqual(result['generation_seconds'], 0)
        self.assertEqual(result['output_tokens'], 1)

    def test_readiness_blocks_generation_until_warmup(self):
        with self.server(warmup=2):
            self.assertEqual(self.request('/health')[0], 200)
            self.assertEqual(self.request('/ready')[0], 503)
            before = self.request('/debug')[1]
            self.assertFalse(before['warmup_finished'])
            self.assertEqual(self.request('/generate', {'prompt': 'Привет'})[0], 503)
            self.assertEqual(self.request('/debug')[1]['calls'], before['calls'])
            self.wait_ready()
            self.assertTrue(self.request('/debug')[1]['warmup_finished'])
            self.assert_version('A')

    def test_loaded_version_a_b_a_changes_only_after_restart(self):
        with self.server():
            self.wait_ready()
            self.assert_version('A')
            self.accept('B')
            self.assert_version('A')
        with self.server():
            self.wait_ready()
            self.assert_version('B')
            self.accept('A')
            self.assert_version('B')
        with self.server():
            self.wait_ready()
            self.assert_version('A')

    def test_invalid_requests_do_not_call_backend(self):
        with self.server():
            self.wait_ready()
            before = self.request('/debug')[1]['calls']
            invalid = [dict(prompt=''), dict(prompt='  '), dict(prompt='x', max_new_tokens=0),
                       dict(prompt='x', max_new_tokens=100000),
                       dict(prompt='x', max_new_tokens='32'),
                       dict(prompt='x', max_new_tokens=True),
                       dict(prompt='x', max_new_tokens=1.5), dict(prompt=123),
                       dict(prompt='x' * 8193), {}, [], None]
            for payload in invalid:
                with self.subTest(payload=str(payload)[:80]):
                    status, _ = self.request('/generate', raw=json.dumps(payload).encode())
                    self.assertTrue(400 <= status < 500, status)
                    self.assertEqual(self.request('/debug')[1]['calls'], before)
            self.assertEqual(self.request('/generate', raw=b'{')[0], 400)
            self.assertEqual(self.request('/generate', raw=b' ' * 65537)[0], 413)
            self.assertEqual(self.request('/debug')[1]['calls'], before)
            self.assert_version('A')
            self.assertEqual(self.request('/debug')[1]['calls'], before + 1)

    def test_corrupt_weights_never_become_ready(self):
        (self.root / 'model-A.txt').write_text('Повреждённые веса')
        with self.server():
            deadline = time.monotonic() + 5
            while True:
                debug = self.request('/debug')[1]
                if debug.get('load_failed') or debug['warmup_finished']:
                    break
                self.assertLess(time.monotonic(), deadline, 'Загрузка не завершилась')
                time.sleep(.02)
            self.assertTrue(debug.get('load_failed'))
            self.assertEqual(self.request('/health')[0], 200)
            self.assertEqual(self.request('/ready')[0], 503)
            self.assertEqual(self.request('/generate', {'prompt': 'Привет'})[0], 503)
            self.assertEqual(self.request('/debug')[1]['calls'], 0)


class ServiceBackendTests(unittest.TestCase):
    def setUp(self):
        # Отдельный импорт: три исходных HTTP-теста не зависят от API create_app.
        sys.path.insert(0, str(PROJECT))
        self.addCleanup(sys.path.remove, str(PROJECT))
        import service
        self.service = service

    def test_two_requests_serialize_and_third_is_rejected(self):
        from fastapi import HTTPException

        entered = threading.Event()
        finish = threading.Event()

        class BlockingBackend:
            calls = 0

            def generate(self, prompt, limit):
                self.calls += 1
                entered.set()
                if not finish.wait(5):
                    raise RuntimeError('Проверка не завершила запрос')
                return dict(answer=prompt, output_tokens=1)

        state = self.service.ServiceState(PROJECT, 0)
        state.backend = BlockingBackend()
        state.manifest = dict(model_id='A', runtime_sha256='a' * 64)
        state.warmup_finished.set()
        request = self.service.GenerateRequest(prompt='Привет')
        # Наблюдаем занятие второго места без зависимости от скорости потоков.
        second_accepted = threading.Event()
        slots = state.slots

        class ObservedSlots:
            def acquire(self, blocking):
                result = slots.acquire(blocking=blocking)
                if result and entered.is_set():
                    second_accepted.set()
                return result

            def release(self):
                slots.release()

        state.slots = ObservedSlots()
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(state.generate, request)
            try:
                self.assertTrue(entered.wait(2))
                second = pool.submit(state.generate, request)
                self.assertTrue(second_accepted.wait(2))
                with self.assertRaises(HTTPException) as rejected:
                    state.generate(request)
                self.assertEqual(rejected.exception.status_code, 429)
                self.assertEqual(state.backend.calls, 1)
                time.sleep(.02)
            finally:
                finish.set()
            one, two = first.result(), second.result()
        self.assertEqual(state.backend.calls, 2)
        self.assertGreater(two['queue_seconds'], .01)
        for result in (one, two):
            self.assertAlmostEqual(result['server_seconds'],
                                   result['queue_seconds'] + result['generation_seconds'])

    def test_runtime_hash_mismatch_does_not_construct_backend(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'accepted.json').write_text(json.dumps(dict(
                runtime={'dtype': 'float32'}, runtime_sha256='a' * 64)))
            factory = unittest.mock.Mock()
            state = self.service.ServiceState(root, 0, backend_factory=factory)
            with self.assertLogs(level='ERROR'):
                state.load()
            self.assertFalse(state.ready)
            factory.assert_not_called()

    def test_tokenizer_template_and_revision_checked_before_loading(self):
        from unittest.mock import Mock

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            revision = 'a' * 40
            base = root / revision
            base.mkdir()
            config = base / 'tokenizer_config.json'
            original = json.dumps(dict(chat_template='Шаблон')).encode()
            config.write_bytes(original)
            runtime = dict(base_name='Qwen/Qwen3-1.7B', base_revision=revision,
                           base_path=str(base), tokenizer_path=str(base), device='cpu',
                           dtype='float32', tokenizer_sha256=digest({config.name: hashlib.sha256(original).hexdigest()}),
                           template_sha256=hashlib.sha256('Шаблон'.encode()).hexdigest(),
                           generation=dict(system='Инструкция', max_new_tokens=32,
                                           enable_thinking=False, do_sample=False))
            model_loader, tokenizer_loader, adapter_loader = Mock(), Mock(), Mock()
            tokenizer_loader.from_pretrained.return_value.chat_template = 'Шаблон'
            dependencies = dict(
                torch=SimpleNamespace(device=lambda name: name, float32='float32'),
                transformers=SimpleNamespace(AutoModelForCausalLM=model_loader,
                                             AutoTokenizer=tokenizer_loader),
                peft=SimpleNamespace(PeftModel=adapter_loader))
            manifest = dict(model_id='base', runtime_sha256=digest(runtime),
                            runtime=runtime, artifact_uri=base.as_uri())
            for defect in ('tokenizer', 'template', 'revision'):
                with self.subTest(defect=defect), patch.dict(sys.modules, dependencies):
                    modified = json.loads(json.dumps(manifest))
                    if defect == 'tokenizer':
                        config.write_text('{}')
                    elif defect == 'template':
                        modified['runtime']['template_sha256'] = '0' * 64
                    else:
                        modified['runtime']['base_path'] = str(root)
                    try:
                        with self.assertRaises(ValueError):
                            self.service.TransformersBackend(modified, root)
                        model_loader.from_pretrained.assert_not_called()
                        tokenizer_loader.from_pretrained.assert_not_called()
                    finally:
                        config.write_bytes(original)
            with patch.dict(sys.modules, dependencies):
                self.service.TransformersBackend(manifest, root)
                model_loader.from_pretrained.assert_called_once_with(
                    str(base), revision=revision, dtype='float32', local_files_only=True)
                tokenizer_loader.from_pretrained.assert_called_once_with(str(base), local_files_only=True)
                adapter_loader.from_pretrained.assert_not_called()
                adapter = root / 'adapter'
                adapter.mkdir()
                (adapter / 'adapter_config.json').write_text('{}')
                manifest['artifact_uri'] = adapter.as_uri()
                self.service.TransformersBackend(manifest, root)
                adapter_loader.from_pretrained.assert_called_once_with(
                    model_loader.from_pretrained.return_value, str(adapter), local_files_only=True)

    def test_factory_warmup_and_baseline_without_accepted_manifest(self):
        from unittest.mock import Mock

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / 'model.txt'
            weights.write_text('STUB ONLY')
            files = {weights.name: hashlib.sha256(weights.read_bytes()).hexdigest()}
            runtime = dict(base_revision='fixture-only', generation=dict(max_new_tokens=32))
            manifest = dict(model_id='base', runtime=runtime, runtime_sha256=digest(runtime),
                            model_sha256=digest(files), file_hashes=files, artifact_uri=weights.as_uri())
            (root / 'baseline-manifest.json').write_text(json.dumps(manifest))
            backend = Mock()
            factory = Mock(return_value=backend)
            state = self.service.ServiceState(root, 0, backend_factory=factory)
            state.load()
            self.assertTrue(state.ready)
            factory.assert_called_once_with(manifest)
            backend.warmup.assert_called_once_with()
            backend.generate.assert_not_called()
            self.assertFalse((root / 'accepted.json').exists())
            state.fixed_limit = True
            with self.assertRaises(self.service.HTTPException) as rejected:
                state.generate(self.service.GenerateRequest(prompt='Привет', max_new_tokens=16))
            self.assertEqual(rejected.exception.status_code, 422)
            backend.generate.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
