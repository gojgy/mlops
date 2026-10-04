"""Стадия train: LoRA-дообучение на выходе стадии tokenize.

    python -m src.train --variant all_layers          # полный прогон варианта
    python -m src.train --variant all_layers --max-steps 4 --out /tmp/x   # smoke

Цикл обучения написан руками, а не через Trainer: так видно всё, что
обычно прячется, — где считается val loss, как копятся градиенты, что
сохраняется рядом с адаптером.

Что сохраняется в models/adapter_<variant>/: адаптер.
В metrics/train_<variant>.json — кривые train/val loss, время, пиковая память,
число обучаемых параметров, вес адаптера и отпечаток входов.
"""

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

# Потолок памяти Metal — до импорта torch. Без него mps занимает сколько дадут,
# и на ноутбуке с 16–32 ГБ система уходит в своп вместо внятной ошибки.
from src.config import load_params

memory_cfg = load_params()["model"]
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", str(memory_cfg["mps_high_watermark"]))
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", str(memory_cfg["mps_low_watermark"]))   # нижний порог не выше верхнего

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from src.config import load_params
from src.data import LABEL_PAD_ID, batches, load_split
from src.runtime import allocated_bytes, memory_metric, resolve_device, resolve_dtype, set_seed


TRAIN_CODE = ("src/train.py", "src/data.py", "src/runtime.py", "src/config.py")
TRAIN_PARAMS = ("model", "data", "lora", "train", "variants")


def inputs_fingerprint(params: dict) -> str:
    """Отпечаток кода обучения и секций конфига, от которых зависит адаптер.

    check.sh сверяет его с текущим: правили train.py или lr, а адаптер
    остался от прошлого прогона — проверять его бессмысленно.
    """
    h = hashlib.sha256()
    for name in TRAIN_CODE:
        h.update(name.encode())
        h.update(Path(name).read_bytes())
    h.update(json.dumps({k: params.get(k) for k in TRAIN_PARAMS}, sort_keys=True).encode())
    for name in ("train", "val"):
        with Path(params["data"][name]).open("rb") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:12]


def lora_config(params: dict, n_layers: int, freeze_first: int) -> LoraConfig:
    cfg = params["lora"]
    if not 0 <= freeze_first < n_layers:
        raise ValueError("freeze_first должен быть в пределах числа слоёв")
    return LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["alpha"],
        lora_dropout=cfg["dropout"],
        target_modules=cfg["target_modules"],
        modules_to_save=cfg.get("modules_to_save"),
        layers_to_transform=list(range(freeze_first, n_layers)),
        task_type="CAUSAL_LM",
    )


@torch.no_grad()
def evaluate(model, examples, pad_id, device, batch_size: int, track_memory=None) -> float:
    """Средний лосс на токен по всему val-сплиту.

    Среднее по батчам нельзя: в батчах разное число токенов под маской.
    Поэтому сумма лоссов, взвешенная числом токенов, делённая на их сумму.
    """
    model.eval()
    total, count = 0.0, 0
    for batch in batches(examples, batch_size, pad_id, shuffle=False, seed=0):
        batch = {k: v.to(device) for k, v in batch.items()}
        n = int((batch["labels"][:, 1:] != LABEL_PAD_ID).sum())
        if n == 0:
            continue
        loss = model(**batch).loss
        if track_memory:
            track_memory()
        total += loss.item() * n
        count += n
    model.train()
    if device.type == "mps":
        torch.mps.empty_cache()   # логиты оценки не должны висеть в кэше до конца обучения
    if not count:
        raise ValueError("В val нет токенов ответа")
    return total / count


def dir_size_mb(path: Path) -> float:
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1048576, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all_layers")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out", default=None, help="куда писать адаптер и метрики (smoke-тесты)")
    ap.add_argument("--val-limit", type=int, default=None, help="оценивать на первых N примерах val (smoke)")
    args = ap.parse_args()

    params = load_params()
    variants = {v["name"]: v for v in params["variants"]}
    if args.variant not in variants:
        raise SystemExit(f"нет варианта {args.variant!r}, есть: {sorted(variants)}")
    variant = variants[args.variant]
    tcfg = params["train"]
    max_steps = args.max_steps if args.max_steps is not None else tcfg.get("max_steps")

    set_seed(tcfg["seed"])
    device = resolve_device(params["model"]["device"])
    dtype = resolve_dtype(params["model"]["dtype"])

    train_blob = load_split(params["data"]["train"])
    val_blob = load_split(params["data"]["val"])
    if args.val_limit:
        val_blob["examples"] = val_blob["examples"][:args.val_limit]
    pad_id = train_blob["pad_token_id"]
    for blob in (train_blob, val_blob):
        if blob.get("model") != params["model"]["name"]:
            raise ValueError("Модель тензоров не совпадает с model.name")

    tokenizer = AutoTokenizer.from_pretrained(params["model"]["name"])
    model = AutoModelForCausalLM.from_pretrained(params["model"]["name"], dtype=dtype).to(device)
    n_layers = model.config.num_hidden_layers
    if params["train"].get("gradient_checkpointing"):
        # Активации 28 слоёв не храним, а пересчитываем на обратном проходе:
        # памяти в разы меньше, шаг примерно на треть дольше.
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, lora_config(params, n_layers, variant["freeze_first"]))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    model.train()

    examples = train_blob["examples"]
    micro_per_epoch = math.ceil(len(examples) / tcfg["batch_size"])
    steps_per_epoch = math.ceil(micro_per_epoch / tcfg["grad_accum"])
    total_steps = steps_per_epoch * tcfg["epochs"]
    if max_steps:
        total_steps = min(total_steps, max_steps)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=tcfg["lr"], weight_decay=tcfg["weight_decay"],
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(total_steps * tcfg["warmup_ratio"])), total_steps
    )

    eval_bs = tcfg.get("eval_batch_size", tcfg["batch_size"])
    peak = allocated_bytes(device)

    def track_memory():
        nonlocal peak
        peak = max(peak, allocated_bytes(device))

    eval_started = time.perf_counter()
    base_val = evaluate(model, val_blob["examples"], pad_id, device, eval_bs, track_memory)
    base_eval_seconds = time.perf_counter() - eval_started
    print(f"  шаг 0: val {base_val:.4f} ({base_eval_seconds:.1f} с)", flush=True)
    curve_train: list[list[float]] = []
    curve_val: list[list[float]] = [[0, round(base_val, 4)]]
    print(f"[{args.variant}] устройство {device}, обучаемых {trainable:,} из {total:,} "
          f"({trainable / total:.3%}); шагов {total_steps}")

    started = time.perf_counter()
    step, accum_loss, diverged = 0, 0.0, False
    eval_seconds = base_eval_seconds
    train_eval_seconds = 0.0
    seen_tokens = 0
    for epoch in range(tcfg["epochs"]):
        for micro_index, batch in enumerate(batches(examples, tcfg["batch_size"], pad_id, shuffle=True, seed=tcfg["seed"] + epoch)):
            group_start = (micro_index // tcfg["grad_accum"]) * tcfg["grad_accum"]
            group_size = min(tcfg["grad_accum"], micro_per_epoch - group_start)
            seen_tokens += int(batch["attention_mask"].sum())
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / group_size
            loss.backward()
            accum_loss += loss.item()
            peak = max(peak, allocated_bytes(device))
            if (micro_index + 1) % tcfg["grad_accum"] and micro_index + 1 != micro_per_epoch:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            curve_train.append([step, round(accum_loss, 4)])
            if not math.isfinite(accum_loss):
                diverged = True
                print(f"  шаг {step}: лосс {accum_loss} — обучение разошлось, останавливаюсь")
                break
            accum_loss = 0.0
            if step % tcfg["eval_every"] == 0 or step == total_steps:
                eval_started = time.perf_counter()
                val_loss = evaluate(model, val_blob["examples"], pad_id, device, eval_bs, track_memory)
                elapsed_eval = time.perf_counter() - eval_started
                eval_seconds += elapsed_eval
                train_eval_seconds += elapsed_eval
                curve_val.append([step, round(val_loss, 4)])
                print(f"  шаг {step}/{total_steps}: train {curve_train[-1][1]:.4f}, val {val_loss:.4f}", flush=True)
            elif args.max_steps:
                print(f"  шаг {step}/{total_steps}: train {curve_train[-1][1]:.4f}", flush=True)
            if step >= total_steps:
                break
        if diverged or step >= total_steps:
            break
    seconds = time.perf_counter() - started - train_eval_seconds   # чистое обучение, без замеров val

    out_root = Path(args.out) if args.out else Path(params["paths"]["models"])
    adapter_dir = out_root / f"adapter_{args.variant}"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    metrics = {
        "variant": args.variant,
        "freeze_first": variant["freeze_first"],
        "model": params["model"]["name"],
        "device": device.type,
        "dtype": params["model"]["dtype"],
        "seed": tcfg["seed"],
        "lr": tcfg["lr"],
        "effective_batch": tcfg["batch_size"] * tcfg["grad_accum"],
        "steps": step,
        "train_examples": len(examples),
        "val_examples": len(val_blob["examples"]),
        "processed_tokens": seen_tokens,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_share": round(trainable / total, 6),
        "base_val_loss": base_val,
        "final_val_loss": curve_val[-1][1] if curve_val else None,
        "diverged": diverged,
        "curve_train": curve_train,
        "curve_val": curve_val,
        "seconds": round(seconds, 1),
        "eval_seconds": round(eval_seconds, 1),
        "seconds_per_step": round(seconds / max(step, 1), 3),
        "train_tokens_per_sec": round(seen_tokens / seconds, 1) if seconds else 0,
        "peak_memory_mb": round(peak / 1048576, 1),
        "memory_metric": memory_metric(device),
        "adapter_dir": str(adapter_dir),
        "adapter_size_mb": dir_size_mb(adapter_dir),
        "inputs_fingerprint": inputs_fingerprint(params),
    }
    mdir = out_root / "metrics" if args.out else Path(params["paths"]["metrics"])
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / f"train_{args.variant}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[{args.variant}] {step} шагов за {seconds:.0f} с; "
          f"пик памяти {metrics['peak_memory_mb']:.0f} МБ; адаптер {metrics['adapter_size_mb']} МБ -> {adapter_dir}")


if __name__ == "__main__":
    main()
