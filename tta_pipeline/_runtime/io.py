"""Small, dependency-free helpers for portable inference workers."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temporary, path)


def resolve(repo, record, verify=True):
    relative = record['path'] if isinstance(record, dict) else record
    path = Path(repo) / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    if verify and isinstance(record, dict) and record.get('sha256') and sha256(path) != record['sha256']:
        raise ValueError('SHA256 mismatch: ' + str(path))
    return path.resolve()


def select_entries(repo, clips=None):
    value = read(Path(repo) / 'configs/evaluation.json')
    entries = value if isinstance(value, list) else value['entries']
    if clips:
        selected = set(clips.split(','))
        if selected - {entry['clip'] for entry in entries}:
            raise ValueError('Unknown requested clip')
        entries = [entry for entry in entries if entry['clip'] in selected]
    if not entries or len({e['clip'] for e in entries}) != len(entries):
        raise ValueError('Expected nonempty unique clips')
    return entries


def model_paths(repo):
    config = read(Path(repo) / 'configs/models.json')
    return {key: resolve(repo, value, verify=False) for key, value in config['paths'].items()}


def identifier(value):
    if not value or Path(value).name != value or value in ('.', '..'):
        raise ValueError('Unsafe identifier')
    return value
