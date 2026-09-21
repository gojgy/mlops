"""Стадия split: разбиение на train/val/test по группам, с проверкой контаминации."""

import json
import random
import time
from collections import Counter
from pathlib import Path

from src.config import load_params
from src.contamination import is_clean, report
from src.schema import Example, dump, iter_examples
from src.textnorm import normalize_group


def group_split(
    sizes: dict[str, int], ratios: dict[str, float], seed: int, max_eval_group_share: float
) -> dict[str, str]:
    """Раздать метки сплита группам: группа целиком уходит в один сплит.

    Группы идут в случайном порядке (seed), каждая — в сплит с наибольшим
    относительным недобором доли. Группа крупнее max_eval_group_share от целевого
    объёма val/test идёт только в основной сплит: иначе одна категория занимает
    больше половины test, и метрика на нём — метрика одной категории.
    """
    total = sum(sizes.values())
    target = {name: total * share for name, share in ratios.items()}
    filled = {name: 0 for name in ratios}
    main_split = max(ratios, key=lambda n: ratios[n])
    order = sorted(sizes)
    random.Random(seed).shuffle(order)
    labels: dict[str, str] = {}
    for key in order:
        allowed = [
            n for n in ratios
            if n == main_split or sizes[key] <= max_eval_group_share * target[n]
        ]
        name = max(allowed, key=lambda n: (target[n] - filled[n]) / target[n])
        labels[key] = name
        filled[name] += sizes[key]
    return labels


def main() -> None:
    params = load_params()
    paths = params["paths"]
    cfg = params["split"]
    started = time.perf_counter()

    examples: list[Example] = list(iter_examples(paths["clean"]))
    if cfg["group_key"] != "topic":
        raise SystemExit(f"неизвестный split.group_key: {cfg['group_key']!r}")

    sizes: dict[str, int] = {}
    for ex in examples:
        key = normalize_group(ex.topic)
        sizes[key] = sizes.get(key, 0) + 1

    labels = group_split(sizes, cfg["ratios"], cfg["seed"], cfg["max_eval_group_share"])
    buckets: dict[str, list[Example]] = {name: [] for name in cfg["ratios"]}
    for ex in examples:
        buckets[labels[normalize_group(ex.topic)]].append(ex)

    empty = [name for name, rows in buckets.items() if not rows]
    if empty:
        raise SystemExit(
            f"split: пустые сплиты {empty} — групп слишком мало или они слишком крупные "
            f"для max_eval_group_share = {cfg['max_eval_group_share']}"
        )

    for name, rows in buckets.items():
        out = Path(paths[name])
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for ex in rows:
                fh.write(dump(ex) + "\n")

    nd = params["clean"]["near_dup"]
    rep = report(
        buckets["train"],
        buckets["test"],
        shingle_words=nd["shingle_words"],
        num_perm=nd["num_perm"],
        threshold=params["contamination"]["threshold"],
    )

    metrics = {
        "version": params["collect"]["version"],
        "seed": cfg["seed"],
        "group_key": cfg["group_key"],
        "groups_total": len(sizes),
        "max_eval_group_share": cfg["max_eval_group_share"],
        "sizes": {name: len(rows) for name, rows in buckets.items()},
        "groups": {
            name: len({normalize_group(ex.topic) for ex in rows}) for name, rows in buckets.items()
        },
        "largest_group_share": {
            name: round(Counter(normalize_group(ex.topic) for ex in rows).most_common(1)[0][1] / len(rows), 4)
            for name, rows in buckets.items()
        },
        "ratios_actual": {
            name: round(len(rows) / len(examples), 4) for name, rows in buckets.items()
        },
        "contamination": rep,
        "seconds": round(time.perf_counter() - started, 2),
    }
    mpath = Path(paths["metrics_split"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        "split: "
        + ", ".join(f"{name} {len(rows)}" for name, rows in buckets.items())
        + f" (групп {len(sizes)}, {metrics['seconds']} с)"
    )

    if not is_clean(rep):
        raise SystemExit(
            "split: КОНТАМИНАЦИЯ train/test — "
            + ", ".join(f"{k} = {rep[k]}" for k in ("id_overlap", "text_overlap", "group_overlap", "near_dup_pairs"))
        )


if __name__ == "__main__":
    main()
