"""Отказ до обращения к API при несовпадении учебного кластера."""

import base64
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import cluster


class GuardTests(unittest.TestCase):
    def config(self):
        return dict(
            **{"current-context": cluster.CONTEXT},
            contexts=[
                dict(
                    name=cluster.CONTEXT,
                    context={"cluster": cluster.CONTEXT, "user": cluster.CONTEXT},
                )
            ],
            users=[
                dict(
                    name=cluster.CONTEXT,
                    user={"client-certificate-data": "x", "client-key-data": "y"},
                )
            ],
            clusters=[
                dict(
                    name=cluster.CONTEXT,
                    cluster={
                        "server": "https://127.0.0.1:12345",
                        "certificate-authority-data": base64.b64encode(
                            b"lab-ca\n"
                        ).decode(),
                    },
                )
            ],
        )

    def validate(self, config, node=None):
        with tempfile.NamedTemporaryFile() as f:
            answers = [
                json.dumps(
                    [{"Endpoints": {"docker": {"Host": "unix:///local-docker.sock"}}}]
                ),
                json.dumps(
                    [
                        node
                        or {
                            "Config": {
                                "Labels": {"io.x-k8s.kind.cluster": cluster.NAME}
                            },
                            "NetworkSettings": {
                                "Ports": {
                                    "6443/tcp": [
                                        {"HostIp": "127.0.0.1", "HostPort": "12345"}
                                    ]
                                }
                            },
                        }
                    ]
                ),
                json.dumps(config),
                "lab-ca\n",
            ]
            with (
                patch.object(cluster, "CONFIG", Path(f.name)),
                patch.object(cluster, "output", side_effect=answers),
            ):
                return cluster.validate()

    def test_only_matching_local_cluster_allowed(self):
        self.assertEqual(self.validate(self.config()), "https://127.0.0.1:12345")

    def test_wrong_context_and_remote_server_rejected(self):
        config = self.config()
        config["current-context"] = "production"
        with self.assertRaises(RuntimeError):
            self.validate(config)
        config = self.config()
        config["clusters"][0]["cluster"]["server"] = "https://example.invalid:6443"
        with self.assertRaises(RuntimeError):
            self.validate(config)

    def test_other_ca_or_credentials_rejected(self):
        config = self.config()
        config["clusters"][0]["cluster"]["certificate-authority-data"] = (
            base64.b64encode(b"other").decode()
        )
        with self.assertRaises(RuntimeError):
            self.validate(config)
        config = self.config()
        config["users"][0]["user"] = {"exec": {"command": "cloud-cli"}}
        with self.assertRaises(RuntimeError):
            self.validate(config)

    def test_cannot_override_target_with_arguments(self):
        for flag in [
            "--kubeconfig",
            "--context",
            "--kube-context",
            "--server",
            "-s",
            "-shttps://example.invalid",
            "--kube-apiserver",
            "--token",
        ]:
            with self.subTest(flag=flag), patch.object(cluster, "validate") as validate:
                with self.assertRaises(ValueError):
                    cluster.command("kubectl", flag, "elsewhere")
                validate.assert_not_called()
