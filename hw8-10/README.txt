ДЗ 8–10: выпуск модели и API

Python 3.12. Команды из hw8-10:
  uv sync --frozen
  make check

Для настоящей модели:
  uv sync --frozen --extra model
  make prepare
  make flow
  make serve

make flow завершается отказом: кандидат ДЗ 6–7 не прошёл проверку качества.
API обслуживает исходную базовую Qwen3-1.7B; нового допуска ей не присваивали.
make prepare использует локальные материалы hw6-7 и сохранённый MLflow-артефакт.
Параметры путей: .venv/bin/python real_release.py prepare --help.
База: https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e

Синтетический опыт без весов:
  .venv/bin/python lab.py --workspace .lab-demo init
  PREFECT_HOME="$PWD/context/prefect" .venv/bin/python flow.py --workspace .lab-demo

Замер после готовности API, результат — в новом каталоге:
  make benchmark BENCHMARK_OUTPUT=context/benchmark-new

Читать REPORT.txt, defects.csv и reports/. В reports сохранены исходные оценки,
ответы, измерения и журналы. context содержит рабочие материалы и не входит в Git.

ci.example.yml — шаблон. Удалённые проверки и защита слияния не настраивались.
