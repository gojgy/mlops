"""Масштабирование и переходы алертов в учебном кластере."""

import json
import threading
import time
from urllib.request import urlopen

import experiments as ex


def record_alerts(stop, path):
    with path.open("w") as f:
        while not stop.is_set():
            try:
                with urlopen("http://127.0.0.1:19090/api/v1/alerts", timeout=5) as r:
                    body = json.load(r)
                f.write(json.dumps(dict(time=time.time(), **body)) + "\n")
                f.flush()
            except Exception as e:
                f.write(json.dumps(dict(time=time.time(), error=str(e))) + "\n")
                f.flush()
            stop.wait(2)


def desired():
    d = ex.get("deployment", "hw11-api")
    return d["spec"]["replicas"], d.get("status", {}).get("readyReplicas", 0)


def wait_for(predicate, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(3)
    raise TimeoutError("Ожидаемый переход HPA не произошёл")


def main():
    ex.upgrade(
        image="mlops-hw11:fixed",
        model="A",
        neverReady=False,
        concurrency=1,
        queueSize=4,
        hpa=True,
        hitCPU=0.08,
        missCPU=0.12,
    )
    ex.rollout()
    wait_for(lambda: desired() == (2, 2), 120)
    ex.write(
        ex.RESULTS / "hpa-resource-metrics.json",
        json.loads(
            ex.k(
                "get", "--raw", "/apis/metrics.k8s.io/v1beta1/namespaces/hw11/pods"
            ).stdout
        ),
    )
    print("HPA: две реплики готовы; запускаю CPU-нагрузку", flush=True)
    with ex.observe("hpa"):
        load = ex.start_load(
            "hpa/requests", ex.make_plan("even", duration=150, rate=10), lead=4
        )
        wait_for(lambda: desired() == (3, 3), 120)
        ex.write(
            ex.RESULTS / "hpa-scaled-up.json",
            dict(
                time=time.time(),
                deployment=ex.get("deployment", "hw11-api"),
                hpa=ex.get("hpa"),
            ),
        )
        print("HPA: третья реплика готова", flush=True)
        ex.finish_load(load)
        ended = time.time()
        wait_for(lambda: desired() == (2, 2), 180)
        ex.write(
            ex.RESULTS / "hpa-scaled-down.json",
            dict(
                time=time.time(), since_load_end=time.time() - ended, hpa=ex.get("hpa")
            ),
        )
    print("HPA: возврат к двум репликам; повтор ClientIP", flush=True)
    # Та же привязка клиента после появления третьей реплики.
    ex.clear_cache()
    with ex.observe("hpa-hot"):
        load = ex.start_load(
            "hpa-hot/requests",
            ex.make_plan("skew", duration=120, rate=6.4),
            policy="affinity",
            traffic="skew",
        )
        rows = ex.finish_load(load)
        ex.write(
            ex.RESULTS / "hpa-hot/distribution.json",
            {p: sum(r["pod"] == p for r in rows) for p in {r["pod"] for r in rows}},
        )
    wait_for(lambda: desired() == (2, 2), 180)
    time.sleep(65)
    print("Алерты: исходное состояние, перегрузка, восстановление", flush=True)
    with ex.observe("alerts"):
        stop = threading.Event()
        thread = threading.Thread(
            target=record_alerts, args=(stop, ex.RESULTS / "alerts/states.jsonl")
        )
        thread.start()
        try:
            time.sleep(6)
            load = ex.start_load(
                "alerts/requests", ex.make_plan("even", duration=40, rate=30), lead=3
            )
            ex.finish_load(load)
            time.sleep(80)
        finally:
            stop.set()
            thread.join(timeout=10)
    ex.write(ex.RESULTS / "hpa-final.json", ex.get("hpa"))


if __name__ == "__main__":
    main()
