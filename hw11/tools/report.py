"""Краткий HTML-отчёт по фактическим результатам опытов."""

import csv
from html import escape
import json
from pathlib import Path
import statistics

from saved_results import RESULTS, source

ROOT = Path(__file__).resolve().parents[1]
R = source()


def read(path):
    return json.loads(Path(path).read_text())


def f(value):
    return "—" if value is None else f"{value:.3f}"


def table(headers, rows):
    return (
        "<table><thead><tr>"
        + "".join("<th>" + escape(str(v)) + "</th>" for v in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join("<td>" + escape(str(v)) + "</td>" for v in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table>"
    )


def figure(name, caption):
    return (
        '<figure><img src="figures/'
        + name
        + '.png" alt="'
        + escape(caption)
        + '"><figcaption>'
        + escape(caption)
        + "</figcaption></figure>"
    )


def main():
    summary = list(csv.DictReader((RESULTS / "summary.csv").open()))
    if len(summary) != 12:
        raise ValueError("Нужны все двенадцать прогонов")
    environment = read(R / "environment.json")
    resources = read(R / "resources-final.json")
    before = read(R / "before-warmup/verdict.json")
    forced = read(R / "verified-forced/verdict.json")["outcomes"]
    comparisons = read(RESULTS / "histogram-comparison.json")
    transition = read(R / "hpa-transition.json")
    down = read(R / "hpa-scaled-down.json")
    hot = read(R / "hpa-hot/summary.json")
    original_counts = " и ".join(
        str(n)
        for pod, n in hot["requests_by_pod"].items()
        if pod not in hot["new_pods"]
    )
    alert_lines = [
        json.loads(x) for x in (R / "alerts/states.jsonl").read_text().splitlines()
    ]
    transitions = []
    previous = {}
    for sample in alert_lines:
        if "data" not in sample:
            continue
        states = {
            a["labels"]["alertname"]: a["state"] for a in sample["data"]["alerts"]
        }
        for name in ["FirstChunkSlow", "RejectionsHigh"]:
            state = states.get(name, "inactive")
            if previous.get(name) != state:
                transitions.append(
                    (name, round(sample["time"] - alert_lines[0]["time"], 1), state)
                )
                previous[name] = state
    rows = []
    for r in summary:
        rows.append(
            [
                r["traffic"],
                r["policy"],
                r["repeat"],
                r["n"],
                r["successful"],
                r["errors"],
                f(float(r["ttfc_p50"])),
                f(float(r["ttfc_p95"])),
                f(float(r["total_p95"])),
            ]
        )
    group = {}
    for traffic in ["even", "skew"]:
        for policy in ["plain", "affinity"]:
            values = [
                r for r in summary if r["traffic"] == traffic and r["policy"] == policy
            ]
            group[traffic, policy] = dict(
                p95=statistics.mean(float(r["ttfc_p95"]) for r in values),
                cache=statistics.mean(float(r["cache_hit_fraction"]) for r in values),
                errors=sum(int(r["errors"]) for r in values),
                n=sum(int(r["n"]) for r in values),
            )
    delta = (1 - group["skew", "plain"]["p95"] / group["skew", "affinity"]["p95"]) * 100
    max_lateness = max(float(r["max_send_lateness"]) for r in summary)
    client_difference = max(
        abs(c["client_success"] - c["server_completed"]) for c in comparisons
    )
    hist_rows = [
        [
            c["run"],
            c["client_success"],
            int(c["server_completed"]),
            f(c["client_p95"]),
            f(c["server_histogram_p95"]),
        ]
        for c in comparisons
    ]
    limits = table(
        ["Контейнер", "CPU requests / limit", "RAM requests / limit"],
        [
            ["API", "200m / 1 CPU", "96 / 192 MiB"],
            ["Клиент", "25m / 500m", "40 / 128 MiB"],
            ["Prometheus", "100m / 500m", "192 / 512 MiB"],
            ["Grafana", "100m / 500m", "128 / 384 MiB"],
            ["Экспортёр Ready", "25m / 200m", "40 / 96 MiB"],
            ["Metrics Server", "50m / 500m", "128 / 256 MiB"],
        ],
    )
    html = """<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>ДЗ 11 · Kubernetes под нагрузкой</title><style>
body{font:17px/1.55 system-ui,sans-serif;color:#172637;max-width:1100px;margin:45px auto;padding:0 25px;background:#fff}h1{font-size:34px}h2{font-size:25px;margin-top:38px}h3{font-size:20px}p{max-width:1000px}table{border-collapse:collapse;font-size:14px;width:100%;margin:18px 0}th,td{text-align:left;border-bottom:1px solid #d8e1e9;padding:8px}th{background:#edf3f8}img{width:100%;height:auto}figure{margin:25px 0}figcaption{color:#526578;font-size:14px}code{font-size:.9em;background:#eef3f7;padding:2px 4px}a{color:#0969a4}.note{border-left:4px solid #2673aa;padding-left:16px}@media print{body{margin:0;font-size:11pt}h2{break-after:avoid}figure,table{break-inside:avoid}}</style>
<h1>ДЗ 11. Kubernetes под нагрузкой</h1>"""
    html += f"""<p class="note">Развёрнут CPU-имитатор с очередью, локальным кешем и выдачей ответа частями. Исправлена преждевременная готовность; проверены обновление, откат, сбой и масштабирование. Сравнены обычный Service и привязка ClientIP.</p>
<h2>Стенд</h2><p>Docker Desktop: {environment["docker"]["NCPU"]} CPU, {environment["docker"]["MemTotal"] / 2**30:.2f} GiB RAM. kind 0.33.0, Kubernetes 1.37.0, один локальный узел arm64, две реплики API. Helm 4.3.1. Отдельный kubeconfig; адрес API и сертификат сверяются с контейнером kind перед каждой командой. Запуск и удаление — в <a href="README.md">README</a>.</p>
{limits}<p>Исходный дефект сохранён в коммите <code>ba29870</code>. Образы, зависимости, ConfigMap и параметры закреплены в репозитории. Маленькие артефакты <code>A.json</code>/<code>B.json</code> проверяются по SHA-256 при загрузке; версия в ответе берётся из загруженного файла. История диалога передаётся клиентом. Локальный кеш — только ускорение, а не хранилище истории.</p>
<h2>Опыт A: готовность, обновление, восстановление</h2><p><b>Гипотеза:</b> работающий HTTP-процесс ещё не готов обслуживать запросы. В исходнике startup/readiness получали 200 до завершения 20-секундного прогрева. Под нагрузкой возникло {before["errors"]} ошибок из 140 запросов; в журнале есть обращения с <code>warmed=false</code>. После исправления тот же сценарий дал 140 ответов HTTP 200. Ready и EndpointSlice исключают новые реплики до окончания прогрева.</p>
<p>При штатном обновлении A→B прошли все 68 запросов: ноль HTTP 5xx и незавершённых потоков. Два 40-секундных потока начались до обновления. Для этого опыта предусмотрены три одновременных ответа на Pod; в сравнении политик — один. Старые ответы завершились в пределах grace period 40 с; preStop — 3 с, ожидание завершения uvicorn — 30 с.</p>
<p>Заведомо неготовая B не получила рабочий трафик. После управляемого отката ответы подтвердили sim-A. Замена Pod сохранила ту же версию, а продолжение диалога проверено непосредственно на новой реплике.</p>
<p>При отдельном принудительном завершении контейнера результаты иные: успешных ответов — {forced.get("success", 0)}, оборванных — {forced.get("aborted", 0)}, отказов — {forced.get("rejected", 0)}, тайм-аутов — {forced.get("timeout", 0)}, прочих HTTP-ошибок — {forced.get("http_error", 0)}. Все 24 попытки сохранены. Kubernetes обнаружил отказ, перезапустил контейнер и дождался прогрева. Контрольный запрос после восстановления успешен; автоматических клиентских повторов нет.</p>"""
    html += figure(
        "01-warmup-rollout",
        "Готовность и результаты запросов. Зелёные отрезки в нижнем опыте — длительные ответы; исходные данные: before-warmup и verified-* в архиве results/raw.tar.gz.",
    )
    html += """<h2>Опыт B: привязка клиента и очередь</h2><p><b>Гипотеза:</b> привязка к реплике сохраняет локальность кеша, но может перегрузить одну очередь. Сравнены два Service над теми же двумя Pod. Четыре клиента имеют разные Pod IP; keep-alive отключён, повторов нет. Сеть — kube-proxy iptables, masqueradeAll=false. Источники TCP-соединений на API подтверждены трассой.</p>
<p>Каждая комбинация выполнена трижды по 120 секунд: 6.4 запроса/с, 768 попыток. При равномерном трафике клиенты отправляют поровну; при перекосе один клиент отправляет 616 запросов, то есть 80.2% после округления. Тела, длины, лимиты и расписание совпадают между политиками. Начальный кеш пуст, прогрев завершён. Порядок политик чередуется.</p>
<p>Параметры имитатора: один исполнитель и четыре места ожидания на Pod; кеш на четыре префикса; вытесняется самый давно использованный. Попадание в кеш требует 0.08 с процессорного времени, промах — 0.12 с. Ответ содержит четыре части с интервалом 0.05 с. Это заданная модель нагрузки, а не производительность языковой модели.</p>"""
    html += table(
        [
            "Трафик",
            "Политика",
            "Повтор",
            "n",
            "Успехи",
            "Ошибки",
            "p50 TTFC, с",
            "p95 TTFC, с",
            "p95 полного, с",
        ],
        rows,
    )
    html += f"""<p>В сценарии с перекосом обычный Service снизил среднее по трём прогонам p95 TTFC успешных ответов на {delta:.1f}% относительно ClientIP. Ошибки: {group["skew", "plain"]["errors"]}/{group["skew", "plain"]["n"]} и {group["skew", "affinity"]["errors"]}/{group["skew", "affinity"]["n"]} соответственно. Решение — отказаться от обязательной ClientIP-привязки. При равномерном трафике доля попаданий среди успешных ответов уменьшилась с {group["even", "affinity"]["cache"]:.1%} до {group["even", "plain"]["cache"]:.1%}. При перекосе она почти одинакова: {group["skew", "affinity"]["cache"]:.1%} и {group["skew", "plain"]["cache"]:.1%}. Это наблюдения по завершённым ответам: отказавшие запросы кеш не проверяют.</p>"""
    html += figure(
        "02-hot-pod",
        "Перекос 80.2%, ClientIP, первый прогон: задержка успешных запросов и running/waiting по каждой реплике.",
    )
    html += figure(
        "03-policies",
        "Три прогона каждого варианта. Показаны перцентили успешных ответов вместе с долей всех неуспешных попыток.",
    )
    html += f"""<h2>Опыт C: HPA и оповещения</h2><p><b>Гипотеза:</b> вычислительная нагрузка вызовет масштабирование по CPU, но новая реплика не устранит жёсткую привязку клиента. Настроен HPA autoscaling/v2: минимум 2, максимум 3, цель CPU 60% от requests 200m. Доступность Resource Metrics API проверена. Нагрузка создаёт настоящий расход CPU через SHA-256, а не только ожидание.</p>
<p>Новый Pod создан через {transition["created_after_start"]:.1f} с после начала нагрузки и стал Ready через {transition["ready_after_start"]:.1f} с. После окончания нагрузки кластер вернулся к двум готовым репликам за {down["since_load_end"]:.1f} с; окно стабилизации вниз — 30 с. В повторе ClientIP из {hot["n"]} попыток успешны {hot["successful"]}, отказов {hot["rejected"]}. На исходные реплики пришлось {original_counts} запросов; новая реплика получила {sum(hot["requests_by_pod"][p] for p in hot["new_pods"])}. Данные сохранены в hpa-hot внутри архива измерений. HPA использует среднюю загрузку: горячая очередь сама по себе не гарантирует превышения порога CPU и не переносит закреплённые запросы.</p>"""
    html += figure(
        "04-hpa",
        "Масштабирование 2→3→2 вместе с фактической отправкой запросов и средней CPU-загрузкой.",
    )
    html += """<p>Два правила Prometheus: p95 до первой части выше 0.4 с при минимум десяти успешных ответах за минуту; доля отказов выше 10% при минимум двадцати попытках за минуту. Условие удерживается 10 с. Управляемая нагрузка и последующее снижение дали следующие переходы:</p>"""
    html += table(["Правило", "С начала опыта, с", "Состояние"], transitions)
    html += f"""<h2>Метод измерения и проверки</h2><p>Клиент использует монотонные часы. TTFC = время первой содержательной части − время отправки; средний inter-chunk latency = (время последней части − время первой)/(N−1), при N=1 значение отсутствует. Отдельно сохранены плановое время, фактическая отправка, завершение, статус, версия, Pod и число частей. Ошибки и тайм-ауты не включаются в перцентили успешных ответов, но входят в общий учёт попыток. Максимальное опоздание генератора: {max_lateness:.3f} с.</p>
<p>Клиентские p50/p95 рассчитаны линейной интерполяцией. Prometheus хранит classic Histogram; серверный p95 получен из суммарных bucket-счётчиков с сохранением <code>le</code>. До и после каждого прогона сохранены счётчики после паузы на сбор метрик. Это тот же набор завершённых ответов; клиентское время дополнительно включает транспорт.</p>"""
    html += table(
        [
            "Прогон",
            "Успехи клиента",
            "Завершения сервера",
            "p95 клиента, с",
            "p95 Histogram, с",
        ],
        hist_rows,
    )
    html += f"""<p>Максимальное расхождение количества завершений: {client_difference:g}. Оценка Histogram приближённая: линейная интерполяция внутри широкого bucket может заметно отличаться от точного клиентского перцентиля. PromQL и исходные счётчики сохранены; мониторинг собирается каждые 2 с.</p>
<p>Шесть проверок через Kubernetes пройдены: прогрев, штатное обновление, откат, восстановление версии и истории на новой реплике, учёт принудительного обрыва, изоляция двух диалогов. Ещё четыре проверки защищают выбор локального кластера. Исходный дефект обнаружен той же проверкой прогрева.</p>
<h2>Ревью: закрепить диалог за Pod навсегда?</h2>
<p>При очереди 3 с на реплике A и пересчёте префикса за 0.8 с на свободной реплике B новый запрос следует направить B: при прочих равных он начнёт обслуживаться раньше. Перед допуском нужно проверить свободный слот, ограничение очереди и остаток клиентского тайм-аута. Для выбора нужны замеры стоимости пересчёта по длине контекста, времени передачи истории, вероятности попадания в кеш, очередей и ошибок. Сравнивать следует ожидаемое время до первого ответа, а не только наличие кеша.</p>
<p>История должна быть у клиента либо во внешнем хранилище. При потере A новый Pod получает историю и восстанавливает префикс вычислением; локальный кеш может исчезнуть. Уже начатый поток остаётся на обслуживающей реплике до завершения в пределах grace period. Его нельзя незаметно перенаправить B: повтор может дублировать вывод. При обрыве нужен явный результат ошибки и согласованное правило новой попытки.</p>
<p>Это соответствует <a href="https://cloud.google.com/blog/products/containers-kubernetes/how-gke-inference-gateway-improved-latency-for-vertex-ai">кейсу Google Cloud / Vertex AI со слайда 70 лекции</a>: маршрутизация учитывала и кеш префикса, и очередь; увеличение веса очереди позволяло обходить перегруженные реплики даже при попадании в кеш. Наш обычный Service решает более узкую задачу распределения. Совместный учёт кеша и очереди потребует отдельного сравнения на настоящем движке.</p>
<h2>Ограничения и ресурсы</h2>
<p>Один узел и общий CPU: добавление реплики не добавляет физические процессоры. Имитация кеша не воспроизводит KV-cache GPU. Измерены части ответа; TTFT, TPOT и tokens/s модели не заявляются. Три повтора показывают изменчивость этого стенда, а не гарантии для производственной системы.</p>
<p>Prometheus: история не более двух часов и 256 MB, том 512 MiB. Grafana: том 512 MiB. Стартовый лимит Grafana 128 MiB оказался ниже фактического объёма около 171 MiB и был увеличен до нагрузочных сравнений. На учебном узле containerd занимает {resources["kind_volume_bytes"]["containerd"] / 2**30:.2f} GiB, данные kubelet — {resources["kind_volume_bytes"]["kubelet"] / 2**20:.0f} MiB; они находятся в отдельном Docker-томе. Учтён также записываемый слой контейнера. Размеры образов и итоговый замер сохранены в environment.json и resources-final.json внутри архива измерений. Команды удаления собственного стенда приведены в README.</p>
<p><a href="results/requests.csv">Сырые запросы</a> · <a href="results/raw.tar.gz">Журналы и измерения</a> · <a href="results/summary.csv">Сводная таблица</a> · <a href="results/per-pod.csv">По репликам</a> · <a href="results/histogram-comparison.json">Счётчики Histogram</a> · <a href="chart/files/monitoring/dashboard.json">Дашборд Grafana</a></p></html>"""
    (ROOT / "report.html").write_text(html)
    print("report.html создан из сохранённых результатов")


if __name__ == "__main__":
    main()
