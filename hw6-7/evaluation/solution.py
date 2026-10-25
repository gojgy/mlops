"""Оценка ответов и проверка условий выпуска."""

import json
import math


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Повторное поле JSON')
        result[key] = value
    return result


def _counts(items):
    n = len(items)
    correct = sum(item['correct'] for item in items)
    valid = sum(item['valid'] for item in items)
    errors = sum(item['error'] is not None for item in items)
    return dict(n=n, correct=correct, valid=valid, errors=errors,
                accuracy=correct / n, format_rate=valid / n)


def _by_id(items):
    result = {}
    for item in items:
        key = item.get('id')
        if not isinstance(key, str) or not key or key in result:
            raise ValueError('Отсутствующий или повторный id')
        result[key] = item
    if not result:
        raise ValueError('Пустой набор')
    return result


def score(rows, outputs):
    expected, actual = _by_id(rows), _by_id(outputs)
    if expected.keys() != actual.keys():
        raise ValueError('Не совпадают id заданий и ответов')
    items = []
    for key, row in expected.items():
        out = actual[key]
        raw, error = out.get('raw', ''), out.get('error')
        parsed, valid = None, False
        if error is None:
            try:
                obj = json.loads(raw, object_pairs_hook=_unique_object)
                valid = (isinstance(obj, dict) and set(obj) == {'answer'}
                         and isinstance(obj['answer'], str) and obj['answer'] in row['choices'])
                if valid:
                    parsed = obj['answer']
            except (ValueError, TypeError):
                pass
        items.append(dict(id=key, raw=raw, parsed=parsed, valid=valid,
                          correct=valid and parsed == row['expected'], error=error,
                          slices=row['slices']))
    names = sorted({name for item in items for name in item['slices']})
    return dict(overall=_counts(items), per_item=items,
                slices={name: _counts([item for item in items if name in item['slices']])
                        for name in names})


def _validate_score(report):
    items = report['per_item']
    by_id = _by_id(items)
    for item in items:
        if (type(item['valid']) is not bool or type(item['correct']) is not bool
                or item['correct'] and not item['valid']
                or item['error'] is not None and (item['valid'] or item['correct'])
                or not isinstance(item['slices'], list) or not item['slices']
                or any(not isinstance(s, str) or not s for s in item['slices'])):
            raise ValueError('Некорректный результат примера')
    names = {s for item in items for s in item['slices']}
    if names != set(report['slices']):
        raise ValueError('Не совпадают срезы и результаты примеров')
    groups = [(report['overall'], items)]
    groups.extend((report['slices'][s], [item for item in items if s in item['slices']]) for s in names)
    for measured, members in groups:
        reference = _counts(members)
        for key in ('n', 'correct', 'valid', 'errors'):
            if type(measured[key]) is not int or measured[key] != reference[key]:
                raise ValueError('Счётчики не соответствуют результатам примеров')
        for key in ('accuracy', 'format_rate'):
            value = measured[key]
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not math.isclose(value, reference[key], rel_tol=0, abs_tol=1e-12)):
                raise ValueError('Метрика не соответствует счётчикам')
    if report['overall']['errors']:
        raise ValueError('Есть ошибки генерации')
    return by_id


def quality_gate(baseline, candidate, critical_slices):
    try:
        for key in ('protocol_sha256', 'golden_sha256', 'baseline_spec_sha256'):
            if not isinstance(baseline[key], str) or not baseline[key] or baseline[key] != candidate[key]:
                raise ValueError('Несовместимые версии протокола, данных или базы')
        checked = []
        for report in (baseline, candidate):
            if report['status'] != 'complete' or not isinstance(report['model'], dict) or not report['model']:
                raise ValueError('Неполный отчёт или отсутствует модель')
            items = _validate_score(report)
            behavior = _validate_score(report['behavioral'])
            if len(items) < 50 or len(report['slices']) < 3 or len(behavior) < 3:
                raise ValueError('Недостаточно примеров, срезов или проверок поведения')
            checked.append((items, behavior))
        for a, b in zip(*checked):
            if a.keys() != b.keys() or any(a[key]['slices'] != b[key]['slices'] for key in a):
                raise ValueError('Разные id или состав срезов')
        if set(baseline['slices']) != set(candidate['slices']):
            raise ValueError('Разные срезы')
        for key in ('accuracy', 'format_rate'):
            if candidate['overall'][key] < baseline['overall'][key]:
                raise ValueError(f'Ухудшение общей метрики {key}')
        if not critical_slices:
            raise ValueError('Не определены критические срезы')
        for name in critical_slices:
            if candidate['slices'][name]['accuracy'] < baseline['slices'][name]['accuracy']:
                raise ValueError(f'Ухудшение критического среза {name}')
        if candidate['behavioral']['overall']['correct'] != candidate['behavioral']['overall']['n']:
            raise ValueError('Не пройдены обязательные проверки поведения')
    except (KeyError, TypeError, IndexError, AttributeError, ZeroDivisionError) as exc:
        raise ValueError('Неполный или некорректный отчёт') from exc
    return dict(passed=True, critical_slices=list(critical_slices),
                baseline_accuracy=baseline['overall']['accuracy'],
                candidate_accuracy=candidate['overall']['accuracy'])
