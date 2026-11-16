"""Ready реплик учебного Deployment из Kubernetes API."""
from pathlib import Path
import threading
import time

import httpx
from prometheus_client import Gauge, start_http_server

READY = Gauge('hw11_pod_ready', 'Ready по состоянию Kubernetes', ['pod'])
DESIRED = Gauge('hw11_desired_replicas', 'Желаемое число реплик')
ERROR = Gauge('hw11_observer_error', 'Ошибка чтения Kubernetes API')


def collect():
    service_account = Path('/var/run/secrets/kubernetes.io/serviceaccount')
    token = (service_account / 'token').read_text()
    namespace = (service_account / 'namespace').read_text()
    with httpx.Client(base_url='https://kubernetes.default.svc',
                      verify=str(service_account / 'ca.crt'), trust_env=False,
                      headers={'Authorization': 'Bearer ' + token}, timeout=5) as client:
        while True:
            try:
                pods = client.get(f'/api/v1/namespaces/{namespace}/pods', params={'labelSelector': 'app=hw11-api'})
                pods.raise_for_status()
                deployment = client.get(f'/apis/apps/v1/namespaces/{namespace}/deployments/hw11-api')
                deployment.raise_for_status()
                READY.clear()
                for pod in pods.json()['items']:
                    ready = (not pod['metadata'].get('deletionTimestamp') and
                             any(c['type'] == 'Ready' and c['status'] == 'True'
                                 for c in pod['status'].get('conditions', [])))
                    READY.labels(pod['metadata']['name']).set(int(ready))
                DESIRED.set(deployment.json()['spec']['replicas'])
                ERROR.set(0)
            except (httpx.HTTPError, KeyError):
                ERROR.set(1)
            time.sleep(2)


if __name__ == '__main__':
    start_http_server(9101)
    threading.Thread(target=collect, daemon=True).start()
    threading.Event().wait()
