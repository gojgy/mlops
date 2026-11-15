"""Команды только для локального kind-кластера mlops-hw11."""
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'context/kubeconfig'
NAME = 'mlops-hw11'
CONTEXT = 'kind-' + NAME
NODE = NAME + '-control-plane'
DOCKER_CONTEXT = 'desktop-linux'


def docker(*args):
    return ['docker', '--context', DOCKER_CONTEXT, *args]


def output(args):
    return subprocess.check_output(args, timeout=15, text=True)


def validate():
    if not CONFIG.is_file():
        raise RuntimeError('Нет отдельного учебного kubeconfig')
    endpoint = json.loads(output(['docker', 'context', 'inspect', DOCKER_CONTEXT]))[0]['Endpoints']['docker']['Host']
    if not endpoint.startswith('unix://'):
        raise RuntimeError('Docker Desktop должен использовать локальный Unix socket')
    node = json.loads(output(docker('inspect', NODE)))[0]
    if node['Config']['Labels'].get('io.x-k8s.kind.cluster') != NAME:
        raise RuntimeError('Контейнер не принадлежит учебному kind-кластеру')
    config = json.loads(output(['kubectl', '--kubeconfig', str(CONFIG), 'config', 'view', '--raw', '-o', 'json']))
    if config.get('current-context') != CONTEXT:
        raise RuntimeError('В учебном файле выбран другой контекст')
    if len(config.get('contexts', [])) != 1 or len(config.get('clusters', [])) != 1:
        raise RuntimeError('В учебном файле должны быть только один контекст и кластер')
    context = config['contexts'][0]
    if context['name'] != CONTEXT or context['context']['cluster'] != CONTEXT:
        raise RuntimeError('Другой кластер в учебном контексте')
    if len(config.get('users', [])) != 1 or context['context'].get('user') != CONTEXT or config['users'][0]['name'] != CONTEXT:
        raise RuntimeError('Другой пользователь в учебном файле')
    credentials = config['users'][0]['user']
    if set(credentials) != {'client-certificate-data', 'client-key-data'}:
        raise RuntimeError('Разрешён только локальный сертификат kind')
    selected = config['clusters'][0]
    if selected['name'] != CONTEXT:
        raise RuntimeError('Другой кластер в учебном файле')
    cluster = selected['cluster']
    if cluster.get('proxy-url'):
        raise RuntimeError('Прокси для учебного API запрещён')
    url = urlsplit(cluster['server'])
    ports = node['NetworkSettings']['Ports'].get('6443/tcp') or []
    if (url.scheme != 'https' or url.hostname != '127.0.0.1'
            or not any(p['HostIp'] == '127.0.0.1' and int(p['HostPort']) == url.port for p in ports)):
        raise RuntimeError('Адрес API не совпадает с локальным портом kind')
    actual_ca = output(docker('exec', NODE, 'cat', '/etc/kubernetes/pki/ca.crt')).encode()
    if base64.b64decode(cluster.get('certificate-authority-data', '')) != actual_ca:
        raise RuntimeError('Сертификат API не совпадает с учебным kind-контейнером')
    if cluster.get('insecure-skip-tls-verify'):
        raise RuntimeError('Проверка сертификата должна быть включена')
    return cluster['server']


def command(tool, *args):
    for arg in args:
        if any(arg == flag or arg.startswith(flag + '=') for flag in
               ('--kubeconfig', '--context', '--kube-context', '--server', '-s', '--cluster', '--user', '--kube-apiserver', '--kube-token', '--kube-ca-file', '--insecure-skip-tls-verify')):
            raise ValueError('Переопределение учебного кластера запрещено')
    validate()
    if tool == 'kubectl':
        return ['kubectl', '--kubeconfig', str(CONFIG), '--context', CONTEXT, *args]
    if tool == 'helm':
        return ['helm', '--kubeconfig', str(CONFIG), '--kube-context', CONTEXT, *args]
    raise ValueError(tool)


if __name__ == '__main__':
    tool, *args = sys.argv[1:]
    if tool == 'verify':
        print(CONTEXT, validate())
    else:
        os.execvp(tool, command(tool, *args))
