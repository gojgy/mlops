"""Воспроизводимые опыты в проверенном учебном kind-кластере."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import cluster

ROOT = cluster.ROOT
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
PYTHON = sys.executable
LOCK = threading.Lock()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def call(tool, *args, check=True, timeout=180, input=None):
    cmd = cluster.command(tool, *map(str, args))
    r = subprocess.run(
        cmd, input=input, capture_output=True, text=True, timeout=timeout
    )
    with LOCK:
        with (RESULTS / "commands.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    dict(
                        time=time.time(),
                        tool=tool,
                        args=list(map(str, args)),
                        code=r.returncode,
                        stderr=r.stderr[-4000:],
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )
    if check and r.returncode:
        raise RuntimeError(f"{tool} {args}: {r.stderr}")
    return r


def k(*args, **kwargs):
    return call("kubectl", "-n", "hw11", *args, **kwargs)


def get(resource, *args):
    return json.loads(k("get", resource, *args, "-o", "json").stdout)


def wait_clients():
    k("wait", "--for=condition=Ready", "pod", "-l", "app=hw11-client", "--timeout=120s")
    clients = get("pods", "-l", "app=hw11-client")["items"]
    assert len(clients) == 4 and len({p["status"]["podIP"] for p in clients}) == 4
    return sorted(p["metadata"]["name"] for p in clients)


def api_pods():
    return get("pods", "-l", "app=hw11-api")["items"]


def upgrade(**values):
    args = []
    for key, value in values.items():
        args += [
            "--set",
            f"{key}={str(value).lower() if type(value) is bool else value}",
        ]
    return call(
        "helm",
        "upgrade",
        "hw11",
        str(ROOT / "chart"),
        "--namespace",
        "hw11",
        "--reuse-values",
        *args,
    )


def rollout(timeout="150s"):
    k("rollout", "status", "deployment/hw11-api", "--timeout=" + timeout, timeout=180)


def inside(code, client="hw11-client-0"):
    return json.loads(k("exec", client, "--", "python", "-c", code).stdout)


def http(url, body=None, client="hw11-client-0", timeout=10):
    code = (
        "import httpx,json; "
        f"r=httpx.request({'GET' if body is None else 'POST'!r},{url!r},json={body!r},timeout={timeout},trust_env=False); "
        'print(json.dumps({"status":r.status_code,"body":r.text}))'
    )
    return inside(code, client)


def clear_cache():
    for pod in api_pods():
        if pod["metadata"].get("deletionTimestamp"):
            continue
        r = http("http://" + pod["status"]["podIP"] + ":8000/admin/cache/reset", {})
        assert r["status"] == 200, r


def make_plan(traffic, duration=120, rate=6.4, parts=4, chunk_seconds=None):
    requests = []
    for i in range(int(duration * rate)):
        client = (
            i % 4
            if traffic == "even"
            else (0 if i % 10 < 8 else 1 + ((i // 10 + i % 10) % 3))
        )
        body = dict(
            prompt=f"client={client}; item={i:04d}; " + "x" * (24 + i % 4 * 16),
            prefix=f"client-{client}-doc-{(i // 4) % 2}",
            parts=parts,
        )
        if chunk_seconds is not None:
            body["chunk_seconds"] = chunk_seconds
        requests.append(dict(id=f"r{i:05d}", client=client, offset=i / rate, body=body))
    return requests


def start_load(
    name, plan, policy="plain", traffic="even", repeat=0, lead=4, timeout=15
):
    target = RESULTS / name
    target.mkdir(parents=True, exist_ok=False)
    write(target / "plan.json", plan)
    start = time.time() + lead
    metadata = dict(
        name=name,
        policy=policy,
        traffic=traffic,
        repeat=repeat,
        start_epoch=start,
        duration=(
            max(x["offset"] for x in plan) + (plan[1]["offset"] - plan[0]["offset"])
            if len(plan) > 1
            else 0
        ),
        plan_sha256=hashlib.sha256(
            json.dumps(plan, sort_keys=True).encode()
        ).hexdigest(),
        keep_alive=False,
        retries=0,
        backend_mode="simulated",
    )
    write(target / "metadata.json", metadata)
    processes = []
    for number in range(4):
        rows = [x for x in plan if x["client"] == number]
        if not rows:
            continue
        config = dict(
            run=name,
            client=number,
            policy=policy,
            traffic=traffic,
            repeat=repeat,
            start_epoch=start,
            timeout=timeout,
            url=f"http://hw11-{policy}:8000",
            requests=rows,
        )
        cmd = cluster.command(
            "kubectl",
            "-n",
            "hw11",
            "exec",
            "-i",
            f"hw11-client-{number}",
            "--",
            "python",
            "-m",
            "src.client",
        )
        out = (target / f"client-{number}.jsonl").open("w")
        err = (target / f"client-{number}.stderr").open("w")
        p = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=out, stderr=err, text=True
        )
        p.stdin.write(json.dumps(config))
        p.stdin.close()
        processes.append((p, out, err))
    return target, processes


def finish_load(load):
    target, processes = load
    for process, out, err in processes:
        code = process.wait(timeout=240)
        out.close()
        err.close()
        if code:
            raise RuntimeError(f"Клиент завершился: {target}, код {code}")
    rows = []
    for p in sorted(target.glob("client-*.jsonl")):
        rows += [json.loads(line) for line in p.read_text().splitlines()]
    expected = json.loads((target / "plan.json").read_text())
    assert len(rows) == len(expected), (target, len(rows), len(expected))
    assert len({r["logical_id"] for r in rows}) == len(rows)
    write(
        target / "counts.json",
        dict(
            planned=len(expected),
            sent=len(rows),
            outcomes={
                s: sum(r["outcome"] == s for r in rows)
                for s in sorted({r["outcome"] for r in rows})
            },
        ),
    )
    return rows


@contextmanager
def observe(name):
    stop = threading.Event()
    path = RESULTS / name
    path.mkdir(exist_ok=True, parents=True)

    def collect():
        with (path / "timeline.jsonl").open("w") as out:
            while not stop.is_set():
                try:
                    row = dict(
                        time=time.time(),
                        pods=api_pods(),
                        endpoints=get(
                            "endpointslices",
                            "-l",
                            "kubernetes.io/service-name=hw11-plain",
                        )["items"],
                        hpa=get("hpa")["items"],
                    )
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                except Exception as e:
                    out.write(json.dumps(dict(time=time.time(), error=str(e))) + "\n")
                    out.flush()
                stop.wait(2)

    thread = threading.Thread(target=collect)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=30)
        for p in api_pods():
            name = p["metadata"]["name"]
            r = k("logs", name, "--timestamps", check=False)
            (path / (name + ".log")).write_text(r.stdout + r.stderr)
