"""Подсчёт результатов и четыре графика из сохранённых запросов и метрик."""

from collections import Counter, defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from saved_results import RESULTS, source

ROOT = Path(__file__).resolve().parents[1]
R = source()
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
    }
)


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [
        json.loads(line)
        for p in sorted(path.glob("client-*.jsonl"))
        for line in p.read_text().splitlines()
    ]


def quantile(values, q):
    return None if not values else float(np.quantile(values, q, method="linear"))


def distribution(data):
    good = [r for r in data if r["outcome"] == "success"]
    counts = Counter(r["outcome"] for r in data)
    return dict(
        n=len(data),
        successful=len(good),
        rejected=counts["rejected"],
        aborted=counts["aborted"],
        timeout=counts["timeout"],
        errors=len(data) - len(good),
        error_fraction=1 - len(good) / len(data) if data else None,
        cache_hit_fraction=sum(r["cache_hit"] is True for r in good) / len(good)
        if good
        else None,
        ttfc_p50=quantile([r["ttfc_seconds"] for r in good], 0.5),
        ttfc_p95=quantile([r["ttfc_seconds"] for r in good], 0.95),
        total_p50=quantile([r["total_seconds"] for r in good], 0.5),
        total_p95=quantile([r["total_seconds"] for r in good], 0.95),
        max_send_lateness=max((r["send_lateness_seconds"] for r in data), default=None),
    )


def histogram_quantile(buckets, q):
    pairs = sorted((float(le), count) for le, count in buckets.items())
    n = pairs[-1][1]
    if n <= 0:
        return None
    target = n * q
    previous_le, previous_count = 0.0, 0.0
    for le, count in pairs:
        if count >= target:
            if not np.isfinite(le):
                return previous_le
            return (
                previous_le
                + (le - previous_le)
                * (target - previous_count)
                / (count - previous_count)
                if count > previous_count
                else le
            )
        previous_le, previous_count = le, count
    return None


def metric_map(series):
    return {tuple(sorted(s["metric"].items())): float(s["value"][1]) for s in series}


def prom_comparison(snapshot):
    before = metric_map(snapshot["before"]["inference_first_chunk_seconds_bucket"])
    after = metric_map(snapshot["after"]["inference_first_chunk_seconds_bucket"])
    buckets = defaultdict(float)
    for labels, value in after.items():
        label = dict(labels)
        delta = value - before.get(labels, 0)
        if delta < 0:
            raise ValueError("Счётчик сбросился внутри сравнения")
        buckets[label["le"]] += delta
    return dict(
        server_completed=buckets.get("+Inf", 0),
        server_histogram_p95=histogram_quantile(buckets, 0.95),
        buckets=dict(buckets),
    )


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / (name + ".png"), bbox_inches="tight")
    plt.close(fig)


def warmup_figure():
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    for ax, folder, title in zip(
        axes,
        ["before-warmup", "verified-warmup", "verified-rollout"],
        ["Исходная готовность", "Исправленная готовность", "Штатное обновление A → B"],
    ):
        path = R / folder
        data = rows(path / "requests")
        t0 = min(r["scheduled_utc"] for r in data)
        timeline = [
            json.loads(x)
            for x in (path / "timeline.jsonl").read_text().splitlines()
            if x.strip()
        ]
        samples = [t for t in timeline if "pods" in t]
        for model in ["A", "B"]:
            values = []
            for sample in samples:
                count = 0
                for pod in sample["pods"]:
                    env = {
                        e["name"]: e.get("value")
                        for e in pod["spec"]["containers"][0]["env"]
                    }
                    if env.get("MODEL_PATH") != f"/models/{model}.json":
                        continue
                    active = not pod["metadata"].get("deletionTimestamp")
                    ready = any(
                        c["type"] == "Ready" and c["status"] == "True"
                        for c in pod["status"].get("conditions", [])
                    )
                    count += int(active and ready)
                values.append(count)
            ax.step(
                [t["time"] - t0 for t in samples],
                values,
                where="post",
                label="Ready " + model,
            )
        endpoints = [
            sum(
                e.get("conditions", {}).get("ready") is True
                for group in sample["endpoints"]
                for e in group.get("endpoints", [])
            )
            for sample in samples
        ]
        ax.step(
            [t["time"] - t0 for t in samples],
            endpoints,
            where="post",
            linestyle=":",
            color="gray",
            label="Ready Endpoints",
        )
        bad = [r for r in data if r["outcome"] != "success"]
        if bad:
            ax.scatter(
                [r["sent_utc"] - t0 for r in bad],
                [0.15] * len(bad),
                c="#c43c39",
                s=12,
                label="Неуспешный запрос",
            )
        for r in data:
            if r["requested_parts"] == 200:
                ax.plot(
                    [r["sent_utc"] - t0, r["end_utc"] - t0],
                    [2.8, 2.8],
                    linewidth=3,
                    alpha=0.5,
                    color="#348b63",
                )
        ax.set(
            title=f"{title}: {len(data) - len(bad)}/{len(data)} успешных",
            ylabel="Реплики, шт.",
            xlabel="Время относительно начала нагрузки, с",
            ylim=(-0.1, 3.2),
        )
        ax.legend(loc="lower right")
        ax.grid(alpha=0.2)
    save(fig, "01-warmup-rollout")


def queue_figure():
    folder = R / "load-skew-affinity-1"
    data = rows(folder / "requests")
    t0 = read(folder / "requests/metadata.json")["start_epoch"]
    metrics = read(folder / "metrics-range.json")
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    pods = sorted(
        {r["pod"] for r in data if r["pod"]}
        | {s["metric"]["pod"] for s in metrics["inference_running"]}
    )
    colors = {p: plt.get_cmap("tab10")(i) for i, p in enumerate(pods)}
    for pod in pods:
        good = [r for r in data if r["pod"] == pod and r["outcome"] == "success"]
        axes[0].scatter(
            [r["sent_utc"] - t0 for r in good],
            [r["ttfc_seconds"] for r in good],
            s=8,
            alpha=0.6,
            label=pod,
            color=colors[pod],
        )
    for metric, ax in [("inference_waiting", axes[1]), ("inference_running", axes[2])]:
        for s in metrics[metric]:
            pod = s["metric"].get("pod")
            if pod in colors:
                ax.step(
                    [v[0] - t0 for v in s["values"]],
                    [float(v[1]) for v in s["values"]],
                    where="post",
                    label=pod,
                    color=colors[pod],
                )
        ax.set_ylabel(
            "Ожидают, шт." if metric == "inference_waiting" else "Выполняются, шт."
        )
        ax.grid(alpha=0.2)
    axes[0].set(
        title="ClientIP, 80% запросов через одного клиента · прогон 1", ylabel="TTFC, с"
    )
    axes[0].legend(fontsize=8)
    axes[2].set_xlabel("Время нагрузки, с")
    save(fig, "02-hot-pod")


def policies_figure(summary):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for j, traffic in enumerate(["even", "skew"]):
        for policy, color in [("plain", "#2673aa"), ("affinity", "#c06827")]:
            group = sorted(
                [
                    r
                    for r in summary
                    if r["traffic"] == traffic and r["policy"] == policy
                ],
                key=lambda x: x["repeat"],
            )
            axes[0, j].plot(
                [r["repeat"] for r in group],
                [r["ttfc_p95"] for r in group],
                marker="o",
                color=color,
                label=policy,
            )
            axes[1, j].plot(
                [r["repeat"] for r in group],
                [100 * r["error_fraction"] for r in group],
                marker="o",
                color=color,
                label=policy,
            )
        axes[0, j].set(
            title="Равномерно" if traffic == "even" else "80% через одного клиента",
            ylabel="p95 TTFC успешных, с",
        )
        axes[1, j].set(xlabel="Повтор", ylabel="Неуспешные попытки, %")
        for ax in axes[:, j]:
            ax.set_xticks([1, 2, 3])
            ax.grid(alpha=0.2)
            ax.legend()
    fig.suptitle(
        "6.4 запроса/с · 120 с · один исполнитель на Pod · одинаковое расписание"
    )
    save(fig, "03-policies")


def hpa_figure():
    samples = [
        json.loads(x) for x in (R / "hpa/timeline.jsonl").read_text().splitlines()
    ]
    samples = [s for s in samples if "pods" in s]
    t0 = samples[0]["time"]
    initial = {p["metadata"]["name"] for p in samples[0]["pods"]}
    load_start = read(R / "hpa/requests/metadata.json")["start_epoch"]
    for sample in samples:
        new = [p for p in sample["pods"] if p["metadata"]["name"] not in initial]
        ready_new = [
            p
            for p in new
            if any(
                c["type"] == "Ready" and c["status"] == "True"
                for c in p["status"].get("conditions", [])
            )
        ]
        if ready_new:
            pod = ready_new[0]
            ready_time = next(
                c["lastTransitionTime"]
                for c in pod["status"]["conditions"]
                if c["type"] == "Ready"
            )
            timing = dict(
                pod=pod["metadata"]["name"],
                created_after_start=datetime.fromisoformat(
                    pod["metadata"]["creationTimestamp"].replace("Z", "+00:00")
                ).timestamp()
                - load_start,
                ready_after_start=datetime.fromisoformat(
                    ready_time.replace("Z", "+00:00")
                ).timestamp()
                - load_start,
            )
            (R / "hpa-transition.json").write_text(json.dumps(timing, indent=2) + "\n")
            break
    else:
        raise ValueError("Не найден новый готовый Pod при масштабировании")
    data = rows(R / "hpa/requests")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    times = [s["time"] - t0 for s in samples]
    ready = [
        sum(
            not p["metadata"].get("deletionTimestamp")
            and any(
                c["type"] == "Ready" and c["status"] == "True"
                for c in p["status"].get("conditions", [])
            )
            for p in s["pods"]
        )
        for s in samples
    ]
    desired = [
        s["hpa"][0].get("status", {}).get("desiredReplicas", 2) if s["hpa"] else 2
        for s in samples
    ]
    axes[0].step(times, ready, where="post", label="Ready")
    axes[0].step(times, desired, where="post", label="Желаемые", linestyle="--")
    axes[0].set(
        ylabel="Реплики, шт.",
        ylim=(1.5, 3.5),
        title="HPA: CPU target 60%, requests 200m, стабилизация вниз 30 с",
    )
    axes[0].legend()
    bins = np.arange(
        min(r["sent_utc"] for r in data) - t0,
        max(r["end_utc"] for r in data) - t0 + 4,
        2,
    )
    count, edges = np.histogram([r["sent_utc"] - t0 for r in data], bins=bins)
    axes[1].step(edges[:-1], count / 2, where="post", label="Отправлено запросов/с")
    axes[1].set(ylabel="Запросы/с", xlabel="Время опыта, с")
    cpu = []
    for s in samples:
        h = s["hpa"][0].get("status", {}) if s["hpa"] else {}
        metrics = h.get("currentMetrics", [])
        value = next(
            (
                m["resource"]["current"].get("averageUtilization")
                for m in metrics
                if m.get("resource", {}).get("name") == "cpu"
            ),
            None,
        )
        cpu.append(np.nan if value is None else value)
    twin = axes[1].twinx()
    twin.plot(times, cpu, color="#c06827", label="Средняя CPU, % requests")
    twin.set_ylabel("CPU, % requests")
    twin.axhline(60, linestyle=":", color="#c06827")
    for ax in axes:
        ax.grid(alpha=0.2)
    save(fig, "04-hpa")


def load_analysis():
    summary = []
    per_pod = []
    comparisons = []
    for folder in sorted(R.glob("load-*")):
        if not folder.is_dir():
            continue
        data = rows(folder / "requests")
        meta = read(folder / "requests/metadata.json")
        row = dict(
            run=folder.name,
            policy=meta["policy"],
            traffic=meta["traffic"],
            repeat=meta["repeat"],
            **distribution(data),
        )
        summary.append(row)
        for pod in sorted({r["pod"] for r in data if r["pod"]}):
            subset = [r for r in data if r["pod"] == pod]
            per_pod.append(dict(run=folder.name, pod=pod, **distribution(subset)))
        comparisons.append(
            dict(
                run=folder.name,
                client_success=row["successful"],
                client_p95=row["ttfc_p95"],
                **prom_comparison(read(folder / "prometheus.json")),
            )
        )
    for name, data in [("summary.csv", summary), ("per-pod.csv", per_pod)]:
        with (RESULTS / name).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)
    (RESULTS / "histogram-comparison.json").write_text(
        json.dumps(comparisons, ensure_ascii=False, indent=2) + "\n"
    )
    return summary


def hot_analysis():
    samples = [
        json.loads(x) for x in (R / "hpa-hot/timeline.jsonl").read_text().splitlines()
    ]
    active = lambda p: "deletionTimestamp" not in p["metadata"]
    initial = {p["metadata"]["name"] for p in samples[0]["pods"] if active(p)}
    pods = {
        p["metadata"]["name"] for sample in samples for p in sample["pods"] if active(p)
    }
    data = rows(R / "hpa-hot/requests")
    result = dict(
        new_pods=sorted(pods - initial),
        requests_by_pod={
            pod: sum(r["pod"] == pod for r in data) for pod in sorted(pods)
        },
        **distribution(data),
    )
    (R / "hpa-hot/summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )


def export_requests():
    requests = [row for folder in sorted(R.glob("*/requests")) for row in rows(folder)]
    fields = list(dict.fromkeys(key for row in requests for key in row))
    with (RESULTS / "requests.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in requests:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


def main():
    export_requests()
    summary = load_analysis()
    warmup_figure()
    queue_figure()
    policies_figure(summary)
    hpa_figure()
    hot_analysis()
    print("Четыре графика и таблицы сохранены")


if __name__ == "__main__":
    main()
