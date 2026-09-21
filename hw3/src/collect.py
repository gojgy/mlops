"""Стадия collect: выгрузка метаданных arXiv → data/raw.jsonl.

Задача — «аннотация статьи → заголовок»: user — аннотация, assistant — заголовок,
topic — основная категория arXiv. Стадия не перекладывает parquet один в один,
каждое действие видно числом в metrics/collect.json:

  1. сужение — архивы (collect.archives) и окно лет версии;
  2. проверка — отозванные статьи, аннотации-заглушки, производные заголовки,
     заголовок, дословно стоящий в аннотации;
  3. починка текста — склейка строк, разорванных переносами выгрузки;
  4. инструкция, которой в источнике нет, — в нескольких вариантах.
"""

import hashlib
import json
import re
import time
from pathlib import Path

import pyarrow.parquet as pq

from src.config import load_params, source_file, version_window
from src.textnorm import normalize_text

COLUMNS = ["id", "title", "abstract", "categories"]
BATCH = 20000
SAMPLE_BUCKETS = 10_000

NEW_STYLE_ID = re.compile(r"^(\d{2})(\d{2})\.\d{4,5}$")
WITHDRAWN = re.compile(
    r"\b(?:has|have|had) been withdrawn\b|\b(?:is|was|are) withdrawn\b|^\W*withdrawn\b",
    re.IGNORECASE,
)
DERIVATIVE_TITLE = re.compile(
    r"\bpart\s+(?:[ivx]+|\d+)\b"
    r"|^\W*(?:erratum|corrigendum|addendum|publisher'?s note)\b"
    r"|^\W*(?:a\s+)?(?:comments?|reply|response|rejoinder)\s+(?:on|to)\b",
    re.IGNORECASE,
)


def stable_hash(text: str) -> int:
    """sha1, а не hash(): тот солится на каждый запуск, и raw.jsonl не воспроизводился бы."""
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest(), 16)


def pick_prompt(example_id: str, variants: list[str]) -> str:
    """Детерминированно выбрать вариант инструкции по id примера."""
    return variants[stable_hash(example_id) % len(variants)]


def in_sample(example_id: str, share: float) -> bool:
    """Попадает ли статья в выборку. Зависит только от id: попавшее в v1 попадёт и в v2."""
    return stable_hash("sample:" + example_id) % SAMPLE_BUCKETS < share * SAMPLE_BUCKETS


def unwrap(text: str) -> str:
    """Склеить текст, разорванный переносами строк выгрузки, в одну строку."""
    return " ".join(text.split())


def primary_category(categories: list[str]) -> str:
    """Основная категория — первая в списке: 'cs.IT math.IT' → 'cs.IT'."""
    joined = " ".join(categories or []).split()
    return joined[0] if joined else ""


def submission_year(arxiv_id: str) -> int | None:
    """Год подачи из id нового образца (YYMM.NNNNN). Старые id — None."""
    match = NEW_STYLE_ID.match(arxiv_id)
    return 2000 + int(match.group(1)) if match else None


def main() -> None:
    params = load_params()
    cfg = params["collect"]
    paths = params["paths"]
    variants = cfg["system_prompts"]
    if not variants:
        raise SystemExit("collect.system_prompts пуст: инструкцию брать неоткуда")
    archives = set(cfg["archives"])
    first_year, last_year = version_window(params)
    src = source_file(params)

    out = Path(paths["raw"])
    out.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    dropped = {
        "archive_filter": 0,
        "year_filter": 0,
        "withdrawn": 0,
        "short_abstract": 0,
        "derivative_title": 0,
        "title_in_abstract": 0,
        "not_sampled": 0,
    }
    scanned = written = titles_unwrapped = 0
    prompts_used: set[str] = set()
    topics: set[str] = set()

    with out.open("w", encoding="utf-8") as fh:
        for batch in pq.ParquetFile(src).iter_batches(batch_size=BATCH, columns=COLUMNS):
            for row in batch.to_pylist():
                scanned += 1

                topic = primary_category(row["categories"])
                if topic.split(".")[0] not in archives:
                    dropped["archive_filter"] += 1
                    continue
                year = submission_year(row["id"])
                if year is None or not (first_year <= year <= last_year):
                    dropped["year_filter"] += 1
                    continue

                # строка попадает ровно в один счётчик
                title = unwrap(row["title"] or "")
                abstract = unwrap(row["abstract"] or "")
                if cfg["drop_withdrawn"] and WITHDRAWN.search(abstract):
                    dropped["withdrawn"] += 1
                    continue
                if len(abstract) < cfg["min_abstract_chars"] or not title:
                    dropped["short_abstract"] += 1
                    continue
                if cfg["drop_derivative_titles"] and DERIVATIVE_TITLE.search(title):
                    dropped["derivative_title"] += 1
                    continue
                if cfg["drop_title_in_abstract"] and normalize_text(title) in normalize_text(abstract):
                    dropped["title_in_abstract"] += 1
                    continue

                # выборка после фильтров: счётчики выше описывают весь срез
                if not in_sample(row["id"], cfg["sample_share"]):
                    dropped["not_sampled"] += 1
                    continue

                # аннотации разорваны переносами все, поэтому счётчик — по заголовкам
                if "\n" in (row["title"] or "").strip():
                    titles_unwrapped += 1

                example_id = f"arxiv:{row['id']}"
                prompt = pick_prompt(example_id, variants)
                prompts_used.add(prompt)
                topics.add(topic)
                record = {
                    "id": example_id,
                    "topic": topic,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": abstract},
                        {"role": "assistant", "content": title},
                    ],
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    metrics = {
        "version": cfg["version"],
        "years": [first_year, last_year],
        "archives": sorted(archives),
        "rows_scanned": scanned,
        "rows_written": written,
        **{f"dropped_{name}": count for name, count in dropped.items()},
        "titles_unwrapped": titles_unwrapped,
        "topics": len(topics),
        "system_prompt_variants": len(prompts_used),
        "seconds": round(time.perf_counter() - started, 2),
    }
    mpath = Path(paths["metrics_collect"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"collect: версия {cfg['version']} ({first_year}–{last_year}), просмотрено {scanned}, "
        f"записано {written}: архивы -{dropped['archive_filter']}, годы -{dropped['year_filter']}, "
        f"отозванные -{dropped['withdrawn']}, короткие -{dropped['short_abstract']}, "
        f"производные заголовки -{dropped['derivative_title']}, "
        f"заголовок в аннотации -{dropped['title_in_abstract']}, вне выборки -{dropped['not_sampled']}; "
        f"категорий {len(topics)}, вариантов инструкции {len(prompts_used)}, "
        f"{metrics['seconds']} с → {out}"
    )


if __name__ == "__main__":
    main()
