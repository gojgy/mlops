"""Проверки отдельных процессов на синтетическом стенде."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import lab
import release


PROJECT = Path(__file__).resolve().parents[1]


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'workspace'
        self.cli('init')

    def process(self, command, success=True):
        result = subprocess.run(
            [sys.executable, *command], cwd=PROJECT, capture_output=True,
            text=True, timeout=15, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
        )
        print('$', ' '.join(command), '\n', result.stdout, result.stderr, sep='', flush=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def cli(self, *args, success=True):
        return self.process(['lab.py', '--workspace', str(self.root), *args], success)

    def read(self, name):
        return json.loads((self.root / name).read_text())

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value, ensure_ascii=False))

    def events(self):
        path = self.root / 'publications.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def test_rejected_incomplete_and_foreign_reports_preserve_pointer(self):
        before = (self.root / 'accepted.json').read_bytes()
        for report in ['report-reject.json', 'report-incomplete.json', 'report-A.json']:
            with self.subTest(report=report):
                self.cli('run', '--report', report, success=False)
                after = (self.root / 'accepted.json').read_bytes()
                print('accepted SHA-256:', hashlib.sha256(before).hexdigest(),
                      hashlib.sha256(after).hexdigest(), flush=True)
                self.assertEqual(before, after)
                self.assertEqual(self.events(), [])

    def test_prepare_failure_cache_reuse_and_invalidation(self):
        first = self.cli('run', '--fail-after-prepare', success=False)
        self.assertIn('PREPARED', first.stdout)
        second = self.cli('run', '--fail-after-prepare', success=False)
        self.assertIn('CACHE_HIT', second.stdout)
        (self.root / 'input.txt').write_text('Изменённые данные\n')
        changed = self.cli('run', '--fail-after-prepare', success=False)
        self.assertIn('PREPARED', changed.stdout, 'Изменённые байты использовали старый кеш')
        for key, value in [('tokenizer_revision', 'r2'), ('template_sha256', 'f' * 64),
                           ('max_length', 256), ('code_revision', 'prepare-v2')]:
            config = self.read('config.json')
            config[key] = value
            self.write('config.json', config)
            changed = self.cli('run', '--fail-after-prepare', success=False)
            self.assertIn('PREPARED', changed.stdout, key)
        self.assertEqual(self.events(), [])
        attempts = [json.loads(path.read_text()) for path in (self.root / 'attempts').glob('*.json')]
        self.assertEqual(len(attempts), 7)
        self.assertEqual(len({attempt['attempt_id'] for attempt in attempts}), 7)
        self.assertTrue(all(attempt['status'] == 'failed' for attempt in attempts))

    def test_repeat_recovers_pointer_journal_gap(self):
        # Сбой после настоящей публикации: работает и на исходном CLI.
        script = '''import sys
import lab
original = lab.release.publish
def fail(*args, **kwargs):
    original(*args, **kwargs)
    raise RuntimeError('Сбой между указателем и журналом')
lab.release.publish = fail
sys.argv = ['lab.py', '--workspace', sys.argv[1], 'run']
lab.main()
'''
        self.process(['-c', script, str(self.root)], success=False)
        self.assertEqual(self.read('accepted.json'), self.read('manifest-B.json'))
        self.assertEqual(self.events(), [])
        self.cli('run')
        self.cli('run')
        self.assertEqual(len(self.events()), 1, 'Повтор создал дубликат публикации')

    def test_corrupt_cache_is_rebuilt(self):
        self.cli('run', '--fail-after-prepare', success=False)
        entry, = (self.root / 'cache').glob('*.json')
        entry.write_text('{broken')
        result = self.cli('run')
        self.assertIn('PREPARED', result.stdout)
        self.assertEqual(self.read('accepted.json'), self.read('manifest-B.json'))

    def test_changed_artifact_cannot_be_published(self):
        before = (self.root / 'accepted.json').read_bytes()
        (self.root / 'model-B.txt').write_text('Подменённый файл\n')
        self.cli('run', success=False)
        self.assertEqual((self.root / 'accepted.json').read_bytes(), before)
        self.assertEqual(self.events(), [])

    def test_cli_crash_flag_and_manifest_selection(self):
        self.cli('run', '--fail-after-publish', success=False)
        self.assertEqual(self.read('accepted.json'), self.read('manifest-B.json'))
        self.assertEqual(self.events(), [])
        self.cli('run')
        self.cli('run')
        self.assertEqual(len(self.events()), 1)
        self.cli('run', '--manifest', 'manifest-A.json', '--report', 'report-A.json')
        self.assertEqual(self.read('accepted.json'), self.read('manifest-A.json'))
        self.assertEqual(len(self.events()), 2)
        self.cli('run')
        self.assertEqual(self.read('accepted.json'), self.read('manifest-B.json'))
        self.assertEqual(len(self.events()), 2)

    def test_new_evaluation_and_runtime_have_distinct_publications(self):
        self.cli('run')
        report = self.read('report-B.json')
        report['evaluation_run_id'] = 'another-evaluation'
        self.write('report-B.json', report)
        self.cli('run')
        manifest = self.read('manifest-B.json')
        manifest['runtime']['dtype'] = 'float16'
        manifest['runtime_sha256'] = object_hash(manifest['runtime'])
        report['runtime_sha256'] = manifest['runtime_sha256']
        self.write('manifest-B.json', manifest)
        self.write('report-B.json', report)
        self.cli('run')
        self.cli('run')
        self.assertEqual(len(self.events()), 3)
        self.assertEqual(len({event['publication_id'] for event in self.events()}), 3)

    def test_file_uri_snapshot_symlink_and_missing_file(self):
        snapshot = self.root / 'snapshot space'
        snapshot.mkdir()
        blob = self.root / 'blob'
        blob.write_bytes(b'real artifact bytes')
        (snapshot / 'weights.bin').symlink_to(blob)
        manifest = self.read('manifest-B.json')
        manifest['artifact_uri'] = snapshot.as_uri()
        manifest['file_hashes'] = {'weights.bin': hashlib.sha256(blob.read_bytes()).hexdigest()}
        manifest['model_sha256'] = object_hash(manifest['file_hashes'])
        report = self.read('report-B.json')
        report['model_sha256'] = manifest['model_sha256']
        self.write('manifest-B.json', manifest)
        self.write('report-B.json', report)
        self.cli('run')
        blob.unlink()
        self.cli('run', success=False)
        self.assertEqual(len(self.events()), 1)

    def test_jsonl_preparation_rejects_invalid_rows(self):
        config = self.read('config.json')
        config['format'] = 'jsonl'
        self.write('config.json', config)
        rows = [dict(id='1', input='Вопрос', choices=['A', 'B'], expected='A'),
                dict(id='2', input='Второй вопрос', choices={'A': 'Ответ'}, expected='A')]
        path = self.root / 'input.txt'
        path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
        self.cli('run')
        entry, = (self.root / 'cache').glob('*.json')
        cached = json.loads(entry.read_text())
        prepared = cached.get('result', cached)
        self.assertEqual(prepared['rows'], 2)
        self.assertEqual(prepared['input_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertIn('CACHE_HIT', self.cli('run').stdout)
        for changes in [dict(id=''), dict(input=' '), dict(choices=[]),
                        dict(choices='A'), dict(expected='other')]:
            with self.subTest(changes=changes):
                path.write_text(json.dumps({**rows[0], **changes}) + '\n')
                self.cli('run', success=False)
        path.write_text('\n'.join([json.dumps(rows[0])] * 2))
        self.cli('run', success=False)
        self.assertEqual(len(self.events()), 1)

    def test_verify_artifact_rejects_bad_inventory(self):
        manifest = self.read('manifest-B.json')
        for files in [{}, {'../model-B.txt': 'a' * 64}, {'/model-B.txt': 'a' * 64},
                      {'model-B.txt': 'invalid'}, {'model-B.txt': 'a' * 64}]:
            with self.subTest(files=files), self.assertRaises(ValueError):
                release.verify_artifact({**manifest, 'file_hashes': files}, self.root)

    def test_cache_hit_skips_preparation(self):
        prepared = lab.prepare(self.root)
        with patch.object(lab, '_row_count', side_effect=AssertionError('Повтор подготовки')):
            self.assertEqual(lab.prepare(self.root), prepared)
        entry, = (self.root / 'cache').glob('*.json')
        cached = json.loads(entry.read_text())
        cached['result']['rows'] = 900
        entry.write_text(json.dumps(cached))
        with patch.object(lab, '_row_count', wraps=lab._row_count) as count:
            self.assertEqual(lab.prepare(self.root), prepared)
            count.assert_called_once()

    def test_nan_and_disk_failure_preserve_pointer_hash(self):
        manifest, report = self.read('manifest-B.json'), self.read('report-B.json')
        before = (self.root / 'accepted.json').read_bytes()
        for name, metric, expected_error in [('NaN', float('nan'), ValueError),
                                              ('Ошибка диска', .8, OSError)]:
            with self.subTest(name=name):
                (self.root / 'accepted.json').write_bytes(before)
                report['metrics']['accuracy'] = metric
                caught = None
                with patch('os.replace', side_effect=OSError('Ошибка диска')):
                    try:
                        release.publish(manifest, report, self.root / 'accepted.json')
                    except (ValueError, OSError) as error:
                        caught = error
                after = (self.root / 'accepted.json').read_bytes()
                print(name, 'SHA-256 до/после:', hashlib.sha256(before).hexdigest(),
                      hashlib.sha256(after).hexdigest(), flush=True)
                self.assertIsInstance(caught, expected_error)
                self.assertEqual(before, after)
                self.assertEqual(list(self.root.glob('*.tmp')), [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
