# ДЗ 11: Kubernetes под нагрузкой

CPU-имитатор продолжает HTTP-контракт ДЗ 8–10: `/health`, `/ready`, `/generate`. Ответ выдаётся частями; `backend_mode=simulated`. Настоящие веса не используются.

## Запуск

Нужны Docker Desktop, kind 0.33.0, Helm 4.3.1, kubectl и Python 3.12. Из `hw11`:

```sh
make install
make init
make build
make up
make check
make integration
```

Кластер называется `mlops-hw11`. Все команды Kubernetes и Helm выполняются через `tools/cluster.py`: он проверяет отдельный kubeconfig, контекст, локальный порт Docker и сертификат kind. Общий `~/.kube/config` не используется. Переопределение адреса или контекста запрещено.

```sh
python3 tools/cluster.py kubectl -n hw11 get pods
python3 tools/cluster.py kubectl -n hw11 port-forward --address 127.0.0.1 svc/hw11-prometheus 19090:9090
python3 tools/cluster.py kubectl -n hw11 port-forward --address 127.0.0.1 svc/hw11-grafana 13000:3000
```

Prometheus: `http://127.0.0.1:19090`; Grafana: `http://127.0.0.1:13000/d/hw11`. Последние две команды работают в отдельных терминалах.

## Опыты

```sh
make load
make hpa
make report
```

Сырые запросы — `results/requests.csv`, сводные таблицы — `summary.csv` и `per-pod.csv`. Полные журналы, состояния Kubernetes, расписания и исходные метрики собраны в `results/raw.tar.gz`. `make report` автоматически распаковывает архив в игнорируемую `context/` и заново строит отчёт и графики; кластер для этого не нужен.

Сравнение политик: 12 прогонов по 120 секунд. Расписание, запросы и результаты сохраняются в `results/`. Имена прогонов уникальны: существующие результаты не перезаписываются. Для нового полного запуска используйте копию проекта с пустым `results/`.

Исходный дефект готовности сохранён в коммите `ba29870`. Воспроизведение с текущим набором проверок:

```sh
python3 tools/setup.py build_before
HW11_INTEGRATION=1 HW11_IMAGE=mlops-hw11:before HW11_RUN=before-new .venv/bin/python -m unittest tests.test_cluster.ClusterTests.test_warmup -v
```

Ожидается неуспешная проверка: непрогретые реплики получают запросы. Исправленный образ проверяется той же функцией с `HW11_IMAGE=mlops-hw11:fixed`.

## Удаление

```sh
make down
```

Удаляются только кластер `mlops-hw11` и собственные теги `mlops-hw11:before/fixed`. Общий кеш Docker и другие проекты сохраняются. Prometheus ограничен 256 MB и двумя часами истории, его том — 512 MiB; том Grafana — 512 MiB. Конфиги доступа, лекция и рабочие материалы находятся в игнорируемой `context/`.

Результаты и объяснения — [report.html](report.html); дашборд — `chart/files/monitoring/dashboard.json`.
