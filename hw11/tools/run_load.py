"""Два вида трафика × две политики × три прогона по 120 секунд."""

import json
import time
import threading
from urllib.parse import urlencode
from urllib.request import urlopen

import experiments as ex


def prometheus(query):
    with urlopen(
        "http://127.0.0.1:19090/api/v1/query?" + urlencode({"query": query}), timeout=10
    ) as r:
        result = json.load(r)
    if result["status"] != "success":
        raise ValueError(result)
    return result["data"]["result"]


def range_query(query, start, end):
    params = urlencode(dict(query=query, start=start, end=end, step=2))
    with urlopen(
        "http://127.0.0.1:19090/api/v1/query_range?" + params, timeout=15
    ) as response:
        return json.load(response)["data"]["result"]


def snapshots(pods):
    selector = '{pod=~"' + "|".join(pods) + '"}'
    return {
        name: prometheus(name + selector)
        for name in [
            "inference_first_chunk_seconds_bucket",
            "inference_first_chunk_seconds_count",
            "inference_total_seconds_bucket",
            "inference_cache_total",
            "inference_requests_total",
        ]
    }


def network_trace(folder, pods):
    code = """import json,socket,struct,time
seen=[]
for _ in range(60):
 for line in open('/proc/net/tcp').read().splitlines()[1:]:
  fields=line.split()
  if int(fields[1].split(':')[1],16)==8000 and fields[3]=='01':
   source=socket.inet_ntoa(struct.pack('<I',int(fields[2].split(':')[0],16)))
   seen.append({'time':time.time(),'peer':source})
 time.sleep(.1)
print(json.dumps(seen))
"""
    for pod in pods:
        result = ex.k("exec", pod, "--", "python", "-c", code)
        ex.write(folder / ("connections-" + pod + ".json"), json.loads(result.stdout))


def main():
    ex.wait_clients()
    ex.upgrade(
        image="mlops-hw11:fixed",
        clientImage="mlops-hw11:fixed",
        model="A",
        neverReady=False,
        warmupSeconds=20,
        replicas=2,
        concurrency=1,
        queueSize=4,
        cacheSize=4,
        hitCPU=0.08,
        missCPU=0.12,
        chunkSeconds=0.05,
        hpa=False,
        monitoring=True,
    )
    ex.rollout()
    configs = dict(
        backend_mode="simulated",
        warmup_seconds=20,
        concurrency=1,
        queue_size=4,
        cache_size=4,
        hit_cpu_seconds=0.08,
        miss_cpu_seconds=0.12,
        chunk_seconds=0.05,
        duration=120,
        rate=6.4,
        keep_alive=False,
        initial_cache="empty",
        repeats=3,
        client_pods=4,
    )
    ex.write(ex.RESULTS / "load-config.json", configs)
    ex.write(ex.RESULTS / "services.json", ex.get("services"))
    ex.write(ex.RESULTS / "clients.json", ex.get("pods", "-l", "app=hw11-client"))
    for traffic in ["even", "skew"]:
        plan = ex.make_plan(traffic)
        for repeat in range(1, 4):
            # Чередование порядка ограничивает влияние порядка опытов.
            for policy in (
                ["plain", "affinity"] if repeat % 2 else ["affinity", "plain"]
            ):
                name = f"load-{traffic}-{policy}-{repeat}"
                ex.clear_cache()
                time.sleep(3)
                pods = sorted(
                    p["metadata"]["name"]
                    for p in ex.api_pods()
                    if not p["metadata"].get("deletionTimestamp")
                )
                assert len(pods) == 2
                before = snapshots(pods)
                with ex.observe(name):
                    load = ex.start_load(
                        name + "/requests", plan, policy, traffic, repeat
                    )
                    trace = None
                    if traffic == "even" and repeat == 1:
                        trace = threading.Thread(
                            target=network_trace, args=(ex.RESULTS / name, pods)
                        )
                        trace.start()
                    rows = ex.finish_load(load)
                    if trace is not None:
                        trace.join(timeout=30)
                    time.sleep(3)
                    after = snapshots(pods)
                meta = json.loads((load[0] / "metadata.json").read_text())
                series = {
                    metric: range_query(
                        metric, meta["start_epoch"], max(r["end_utc"] for r in rows)
                    )
                    for metric in [
                        "inference_running",
                        "inference_waiting",
                        'sum by (pod)(rate(container_cpu_usage_seconds_total{namespace="hw11",container="api"}[30s]))',
                        'sum by (pod)(container_memory_working_set_bytes{namespace="hw11",container="api"})',
                    ]
                }
                ex.write(ex.RESULTS / name / "metrics-range.json", series)
                ex.write(
                    ex.RESULTS / name / "prometheus.json",
                    dict(pods=pods, before=before, after=after),
                )
                print(
                    name,
                    len(rows),
                    {
                        s: sum(r["outcome"] == s for r in rows)
                        for s in sorted({r["outcome"] for r in rows})
                    },
                    flush=True,
                )


if __name__ == "__main__":
    main()
