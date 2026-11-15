"""Управляемый CPU-имитатор с ограниченным локальным кешем."""
import asyncio
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time


@dataclass(frozen=True)
class Settings:
    pod: str = os.getenv('POD_NAME', 'local')
    model_path: str = os.getenv('MODEL_PATH', '/models/A.json')
    model_sha256: str = os.getenv('MODEL_SHA256', '')
    warmup_seconds: float = float(os.getenv('WARMUP_SECONDS', '20'))
    never_ready: bool = os.getenv('NEVER_READY', 'false') == 'true'
    concurrency: int = int(os.getenv('CONCURRENCY', '1'))
    queue_size: int = int(os.getenv('QUEUE_SIZE', '4'))
    cache_size: int = int(os.getenv('CACHE_SIZE', '4'))
    hit_cpu_seconds: float = float(os.getenv('HIT_CPU_SECONDS', '0.08'))
    miss_cpu_seconds: float = float(os.getenv('MISS_CPU_SECONDS', '0.12'))
    chunk_seconds: float = float(os.getenv('CHUNK_SECONDS', '0.05'))


def consume_cpu(seconds):
    started = time.thread_time()
    block = b'controlled-cpu-work-' * 512
    iterations = 0
    while time.thread_time() - started < seconds:
        hashlib.sha256(block).digest()
        iterations += 1
    return iterations


class SimulatedBackend:
    def __init__(self, settings):
        self.settings = settings
        payload = Path(settings.model_path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != settings.model_sha256:
            raise ValueError('Хеш артефакта не совпал')
        self.model_id = json.loads(payload)['model_id']
        self.ready = False
        self.cache = OrderedDict()
        self.running = 0
        self.waiting = 0
        self.slots = asyncio.Semaphore(settings.concurrency)

    async def warmup(self):
        await asyncio.sleep(self.settings.warmup_seconds)
        self.ready = not self.settings.never_ready

    def admit(self):
        if not self.ready:
            return 503
        if self.running + self.waiting >= self.settings.concurrency + self.settings.queue_size:
            return 429
        self.waiting += 1
        return None

    async def acquire(self):
        started = time.monotonic()
        await self.slots.acquire()
        self.waiting -= 1
        self.running += 1
        return time.monotonic() - started

    async def prepare(self, prefix):
        hit = prefix in self.cache
        if hit:
            self.cache.move_to_end(prefix)
        else:
            self.cache[prefix] = True
            if len(self.cache) > self.settings.cache_size:
                self.cache.popitem(last=False)
        work = self.settings.hit_cpu_seconds if hit else self.settings.miss_cpu_seconds
        iterations = await asyncio.to_thread(consume_cpu, work)
        return hit, iterations

    def release(self):
        self.running -= 1
        self.slots.release()
