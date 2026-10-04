"""Итоговый прогон на 3072 примерах с отдельной конфигурацией."""

import argparse
import sys

from src.config import load_params


def load_final_params() -> dict:
    params = load_params()
    for section, overrides in load_params("params.final.yaml").items():
        params[section] = {**params[section], **overrides}
    return params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "train", "compare"))
    args, remaining = parser.parse_known_args()
    if args.stage == "prepare":
        from src import prepare_data as stage
    elif args.stage == "train":
        from src import train as stage
    else:
        from src import compare as stage
    params = load_final_params()
    stage.load_params = lambda: params
    sys.argv = [sys.argv[0], *remaining]
    stage.main()


if __name__ == "__main__":
    main()
