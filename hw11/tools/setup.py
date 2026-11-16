"""Создание и удаление только отдельного учебного kind-кластера."""

import argparse
import io
import tarfile
import json
import os
import shutil
import subprocess
import sys

import cluster

ROOT = cluster.ROOT
NODE_IMAGE = "kindest/node:v1.37.0@sha256:a1ed56cfb0e7b93589bdf97c8cd566405a265939e3620fc4f5de89adff580ae5"
IMAGES = [
    "prom/prometheus:v3.16.0@sha256:e04987641b57704f408e31f6373ba4138ecf27cb4a165d24aabb6c38fddd0f06",
    "grafana/grafana:13.3.1@sha256:0615ec4fbd4072004b7d86b46ded63013a74b7d6eb1119273dd89e8fa25c13cf",
    "registry.k8s.io/metrics-server/metrics-server:v0.9.1@sha256:9b0ba1926e1d3089673f45461c3c1416109dea33b3ba3434048defd380df329b",
]


def run(args, **kwargs):
    print("$", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True, **kwargs)


def environment():
    env = {
        **os.environ,
        "DOCKER_CONTEXT": cluster.DOCKER_CONTEXT,
        "KIND_EXPERIMENTAL_PROVIDER": "docker",
        "KUBECONFIG": str(cluster.CONFIG),
    }
    env.pop("DOCKER_HOST", None)
    return env


def init():
    ROOT.joinpath("context").mkdir(exist_ok=True)
    endpoint = json.loads(
        cluster.output(["docker", "context", "inspect", cluster.DOCKER_CONTEXT])
    )[0]["Endpoints"]["docker"]["Host"]
    if not endpoint.startswith("unix://"):
        raise RuntimeError("Docker Desktop должен быть локальным")
    if shutil.disk_usage(ROOT).free < 12 * 2**30:
        raise RuntimeError("Для учебного стенда нужно минимум 12 GiB свободного места")
    if cluster.CONFIG.exists():
        cluster.validate()
    else:
        run(
            [
                "kind",
                "create",
                "cluster",
                "--name",
                cluster.NAME,
                "--image",
                NODE_IMAGE,
                "--config",
                ROOT / "kind.yaml",
                "--kubeconfig",
                cluster.CONFIG,
                "--wait",
                "180s",
            ],
            env=environment(),
        )
        cluster.validate()
    run([sys.executable, ROOT / "tools/cluster.py", "verify"])


def public_docker():
    endpoint = json.loads(
        cluster.output(["docker", "context", "inspect", cluster.DOCKER_CONTEXT])
    )[0]["Endpoints"]["docker"]["Host"]
    if not endpoint.startswith("unix://"):
        raise RuntimeError("Разрешён только локальный Docker Desktop")
    config = ROOT / "context/docker-public"
    config.mkdir(exist_ok=True)
    (config / "config.json").write_text(
        json.dumps(
            {
                "cliPluginsExtraDirs": [
                    "/Applications/Docker.app/Contents/Resources/cli-plugins"
                ]
            }
        )
    )
    return ["docker", "--config", str(config), "--host", endpoint]


def build_before():
    cluster.validate()
    target = ROOT / "context/before-source"
    target.mkdir(exist_ok=True)
    archive = subprocess.check_output(
        ["git", "archive", "ba29870", "hw11"], cwd=ROOT.parent
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
        tf.extractall(target, filter="data")
    run(public_docker() + ["build", "-t", "mlops-hw11:before", str(target / "hw11")])
    run(
        ["kind", "load", "docker-image", "mlops-hw11:before", "--name", cluster.NAME],
        env=environment(),
    )


def build():
    cluster.validate()
    run(public_docker() + ["build", "-t", "mlops-hw11:fixed", str(ROOT)])
    run(
        ["kind", "load", "docker-image", "mlops-hw11:fixed", "--name", cluster.NAME],
        env=environment(),
    )


def install():
    cluster.validate()
    for image in IMAGES:
        run(public_docker() + ["pull", image])
        # Импорт только arm64: manifest list не требует отсутствующих amd64-слоёв.
        save = subprocess.Popen(
            public_docker() + ["image", "save", "--platform", "linux/arm64", image],
            stdout=subprocess.PIPE,
        )
        try:
            run(
                cluster.docker(
                    "exec",
                    "-i",
                    cluster.NODE,
                    "ctr",
                    "--namespace=k8s.io",
                    "images",
                    "import",
                    "--platform",
                    "linux/arm64",
                    "--digests",
                    "-",
                ),
                stdin=save.stdout,
            )
        finally:
            save.stdout.close()
        if save.wait():
            raise RuntimeError("Не удалось экспортировать образ")
    run(cluster.command("kubectl", "apply", "-f", str(ROOT / "metrics-server.yaml")))
    run(
        cluster.command(
            "helm",
            "upgrade",
            "--install",
            "hw11",
            str(ROOT / "chart"),
            "-n",
            "hw11",
            "--create-namespace",
            "--set",
            "image=mlops-hw11:fixed",
            "--set",
            "clientImage=mlops-hw11:fixed",
            "--set",
            "monitoring=true",
            "--wait",
            "--timeout",
            "180s",
        )
    )


def down():
    cluster.validate()
    run(["kind", "delete", "cluster", "--name", cluster.NAME], env=environment())
    # Публичные образы могут использоваться другими проектами; общий кеш сохраняется.
    for image in ["mlops-hw11:before", "mlops-hw11:fixed"]:
        exists = subprocess.run(
            public_docker() + ["image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if exists.returncode == 0:
            run(public_docker() + ["image", "rm", image])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "action", choices=["init", "build", "build_before", "install", "down"]
    )
    args = p.parse_args()
    globals()[args.action]()
