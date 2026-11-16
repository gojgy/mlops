"""Проверка полноты измерений без исключения неуспешных попыток."""

from collections import Counter, defaultdict
import csv
import json

from saved_results import RESULTS, source

R = source()


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def requests(folder):
    return [row for p in folder.glob("client-*.jsonl") for row in lines(p)]


def main():
    findings = []
    plans = defaultdict(set)
    for folder in sorted(p for p in R.glob("load-*") if p.is_dir()):
        raw = requests(folder / "requests")
        plan = read(folder / "requests/plan.json")
        meta = read(folder / "requests/metadata.json")
        assert len(raw) == len(plan) == 768, folder
        assert len({r["logical_id"] for r in raw}) == 768, folder
        assert {r["logical_id"] for r in raw} == {p["id"] for p in plan}, folder
        plans[meta["traffic"]].add(meta["plan_sha256"])
        assert len({r["client_ip"] for r in raw}) == 4
        assert all(
            r["backend_mode"] == "simulated" and r["model_id"] == "sim-A" for r in raw
        )
        span = max(r["end_utc"] for r in raw) - min(r["scheduled_utc"] for r in raw)
        assert span >= 120, (folder, span)
        for r in raw:
            assert r["end_monotonic"] >= r["sent_monotonic"]
            assert r["send_lateness_seconds"] >= -0.01
            assert r["parts"] == len(r["chunk_times"])
            assert sorted(r["chunk_times"]) == r["chunk_times"]
            if r["outcome"] == "success":
                assert r["parts"] == r["requested_parts"]
                assert r["sent_monotonic"] <= r["first_monotonic"] <= r["end_monotonic"]
        findings.append(
            dict(
                run=folder.name,
                n=len(raw),
                seconds=span,
                outcomes=dict(Counter(r["outcome"] for r in raw)),
            )
        )
    with (RESULTS / "requests.csv").open() as stream:
        exported = list(csv.DictReader(stream))
    originals = [row for folder in R.glob("*/requests") for row in requests(folder)]
    assert Counter(row["run"] for row in exported) == Counter(
        row["run"] for row in originals
    )
    assert Counter(row["outcome"] for row in exported) == Counter(
        row["outcome"] for row in originals
    )
    assert len(findings) == 12
    assert all(len(hashes) == 1 for hashes in plans.values()), "Расписания различаются"
    comparison = read(RESULTS / "histogram-comparison.json")
    assert all(c["client_success"] == c["server_completed"] for c in comparison)
    assert read(R / "before-warmup/verdict.json")["errors"] > 0
    assert read(R / "verified-warmup/verdict.json")["errors"] == 0
    assert read(R / "verified-rollout/requests/counts.json")["outcomes"] == {
        "success": 68
    }
    for name in ["verified-rollback", "verified-replacement", "verified-forced"]:
        assert read(R / name / "verdict.json")["passed"] is True
    assert read(R / "verified-dialogues.json")["passed"] is True
    assert (R / "hpa-scaled-up.json").is_file() and (
        R / "hpa-scaled-down.json"
    ).is_file()
    samples = lines(R / "alerts/states.jsonl")
    states = defaultdict(list)
    for sample in samples:
        if "data" not in sample:
            continue
        active = {
            a["labels"]["alertname"]: a["state"] for a in sample["data"]["alerts"]
        }
        for name in ["FirstChunkSlow", "RejectionsHigh"]:
            states[name].append(active.get(name, "inactive"))
    for name, values in states.items():
        assert {"inactive", "pending", "firing"} <= set(values), (name, set(values))
        assert values[-1] == "inactive", (name, values[-1])
    assert len(states) == 2
    result = dict(
        passed=True,
        load_runs=findings,
        identical_plans=True,
        matching_server_client_counts=True,
        alerts={k: sorted(set(v)) for k, v in states.items()},
    )
    (RESULTS / "verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print("Полнота данных, расписания, счётчики и переходы алертов проверены")


if __name__ == "__main__":
    main()
