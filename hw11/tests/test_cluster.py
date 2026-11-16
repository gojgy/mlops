"""Проверки настоящего локального Kubernetes. Запуск: HW11_INTEGRATION=1."""

import json
import os
from pathlib import Path
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import experiments as ex


@unittest.skipUnless(os.getenv("HW11_INTEGRATION") == "1", "Нужен учебный kind-кластер")
class ClusterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = os.getenv("HW11_IMAGE", "mlops-hw11:fixed")
        cls.label = os.getenv("HW11_RUN", "checks")
        ex.wait_clients()

    def stable_a(self):
        ex.upgrade(
            image=self.image,
            model="A",
            neverReady=False,
            warmupSeconds=20,
            hpa=False,
            concurrency=1,
        )
        ex.rollout()

    def test_warmup(self):
        plan = ex.make_plan("even", duration=35, rate=4)
        name = self.label + "-warmup"
        with ex.observe(name):
            load = ex.start_load(name + "/requests", plan)
            ex.upgrade(
                image=self.image,
                model="B",
                neverReady=False,
                warmupSeconds=20,
                hpa=False,
                concurrency=1,
                queueSize=4,
                experiment=str(time.time_ns()),
            )
            rows = ex.finish_load(load)
            ex.rollout()
        errors = [r for r in rows if r["outcome"] != "success"]
        ex.write(
            ex.RESULTS / name / "verdict.json",
            dict(
                passed=not errors,
                errors=len(errors),
                codes={
                    str(c): sum(r["status_code"] == c for r in rows)
                    for c in {r["status_code"] for r in rows}
                },
            ),
        )
        self.assertEqual(len(errors), 0, "Запросы не обслужены при прогреве; см. JSONL")

    def test_rollout_completes_streams(self):
        self.stable_a()
        ex.upgrade(concurrency=3)
        ex.rollout()
        name = self.label + "-rollout"
        plan = ex.make_plan("even", duration=68, rate=1)
        # Два длинных потока начинаются до rollout и сохраняются при замене старого Pod.
        plan[0]["body"].update(parts=200, chunk_seconds=0.2)
        plan[1]["body"].update(parts=200, chunk_seconds=0.2)
        with ex.observe(name):
            load = ex.start_load(name + "/requests", plan, timeout=80)
            time.sleep(6)
            ex.upgrade(
                image=self.image,
                model="B",
                warmupSeconds=20,
                experiment=str(time.time_ns()),
            )
            ex.rollout()
            rows = ex.finish_load(load)
        self.assertTrue(all(r["outcome"] == "success" for r in rows))
        self.assertTrue(all(r["status_code"] == 200 for r in rows))
        self.assertTrue(any(r["model_id"] == "sim-B" for r in rows))
        ex.write(
            ex.RESULTS / name / "verdict.json",
            dict(
                passed=True,
                requests=len(rows),
                long_streams=[r for r in rows if r["requested_parts"] == 200],
            ),
        )

    def test_rollback_excludes_unready_version(self):
        self.stable_a()
        history = json.loads(
            ex.call("helm", "history", "hw11", "-n", "hw11", "-o", "json").stdout
        )
        revision = history[-1]["revision"]
        name = self.label + "-rollback"
        with ex.observe(name):
            load = ex.start_load(
                name + "/requests", ex.make_plan("even", duration=32, rate=2)
            )
            ex.upgrade(
                image=self.image,
                model="B",
                neverReady=True,
                experiment=str(time.time_ns()),
            )
            time.sleep(25)
            rows = ex.finish_load(load)
            self.assertTrue(
                all(
                    r["outcome"] == "success" and r["model_id"] == "sim-A" for r in rows
                )
            )
            ex.call("helm", "rollback", "hw11", str(revision), "-n", "hw11")
            ex.rollout()
            check = ex.http(
                "http://hw11-plain:8000/generate", dict(prompt="rollback", parts=2)
            )
        self.assertEqual(check["status"], 200)
        parts = [json.loads(line) for line in check["body"].splitlines()]
        self.assertTrue(all(p["model_id"] == "sim-A" for p in parts))
        ex.write(
            ex.RESULTS / name / "verdict.json",
            dict(passed=True, revision=revision, response=check),
        )

    def test_pod_replacement_restores_version_and_history(self):
        self.stable_a()
        name = self.label + "-replacement"
        first = ex.http(
            "http://hw11-plain:8000/generate", dict(prompt="CONTROL-FIRST", parts=4)
        )
        chunks = [json.loads(line) for line in first["body"].splitlines()]
        pod = chunks[0]["pod"]
        previous = {p["metadata"]["name"] for p in ex.api_pods()}
        with ex.observe(name):
            ex.k("delete", "pod", pod, "--wait=false")
            ex.rollout()
            replacement = next(
                p for p in ex.api_pods() if p["metadata"]["name"] not in previous
            )
            response = ex.http(
                "http://" + replacement["status"]["podIP"] + ":8000/generate",
                dict(
                    prompt="CONTROL-SECOND",
                    history=[dict(role="user", content="CONTROL-FIRST")],
                    parts=4,
                ),
            )
        self.assertEqual(response["status"], 200)
        output = [json.loads(line) for line in response["body"].splitlines()]
        self.assertTrue(all(p["model_id"] == "sim-A" for p in output))
        self.assertNotIn(pod, {p["metadata"]["name"] for p in ex.api_pods()})
        self.assertIn("CONTROL-FIRST", "".join(p.get("text", "") for p in output))
        ex.write(
            ex.RESULTS / name / "verdict.json",
            dict(passed=True, deleted=pod, response=response),
        )

    def test_forced_failure_counts_interrupted_streams(self):
        self.stable_a()
        name = self.label + "-forced"
        plan = ex.make_plan("even", duration=12, rate=2, parts=120, chunk_seconds=0.1)
        with ex.observe(name):
            load = ex.start_load(name + "/requests", plan, timeout=35)
            time.sleep(7)
            pods = [
                p for p in ex.api_pods() if not p["metadata"].get("deletionTimestamp")
            ]
            victim = None
            for pod in pods:
                metrics = ex.http("http://" + pod["status"]["podIP"] + ":8000/metrics")[
                    "body"
                ]
                if any(
                    line.startswith("inference_running ")
                    and float(line.split()[-1]) > 0
                    for line in metrics.splitlines()
                ):
                    victim = pod
                    break
            self.assertIsNotNone(
                victim, "Нет активного потока для принудительного сбоя"
            )
            container_id = victim["status"]["containerStatuses"][0][
                "containerID"
            ].split("://", 1)[1]
            import subprocess
            import cluster

            cluster.validate()
            result = subprocess.run(
                cluster.docker(
                    "exec", cluster.NODE, "crictl", "stop", "--timeout=0", container_id
                ),
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            ex.write(
                ex.RESULTS / name / "kill.json",
                dict(
                    container_id=container_id,
                    code=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                ),
            )
            victim = victim["metadata"]["name"]
            rows = ex.finish_load(load)
            ex.k(
                "wait",
                "--for=condition=Ready",
                "pod",
                "-l",
                "app=hw11-api",
                "--timeout=120s",
            )
        self.assertTrue(any(r["outcome"] == "aborted" for r in rows))
        self.assertTrue(any(r["outcome"] == "rejected" for r in rows))
        self.assertEqual(len(rows), len(plan))
        probe = ex.http(
            "http://hw11-plain:8000/generate", dict(prompt="recovered", parts=2)
        )
        self.assertEqual(probe["status"], 200)
        ex.write(
            ex.RESULTS / name / "verdict.json",
            dict(
                passed=True,
                victim=victim,
                outcomes={
                    s: sum(r["outcome"] == s for r in rows)
                    for s in {r["outcome"] for r in rows}
                },
            ),
        )

    def test_dialogues_are_isolated(self):
        self.stable_a()
        outputs = {}
        for marker in ("DIALOGUE-RED-728", "DIALOGUE-BLUE-419"):
            r = ex.http(
                "http://hw11-plain:8000/generate",
                dict(
                    prompt="continue",
                    history=[dict(role="user", content=marker)],
                    parts=4,
                ),
            )
            self.assertEqual(r["status"], 200)
            outputs[marker] = "".join(
                json.loads(x).get("text", "") for x in r["body"].splitlines()
            )
        self.assertIn("DIALOGUE-RED-728", outputs["DIALOGUE-RED-728"])
        self.assertNotIn("DIALOGUE-BLUE-419", outputs["DIALOGUE-RED-728"])
        self.assertIn("DIALOGUE-BLUE-419", outputs["DIALOGUE-BLUE-419"])
        self.assertNotIn("DIALOGUE-RED-728", outputs["DIALOGUE-BLUE-419"])
        ex.write(
            ex.RESULTS / (self.label + "-dialogues.json"),
            dict(passed=True, outputs=outputs),
        )
