"""Unpack exact Linux x86_64 environments offline into this repository."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

REPO = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', choices=['generation', 'deepspeech', 'evaluation'])
    parser.add_argument('--recreate', action='store_true', help='Replace only packaged env directories after moving the repository')
    args = parser.parse_args()
    if not (REPO / 'environments/archives.json').is_file():
        raise SystemExit('Private environment archives are not included in Git. See docs/ENVIRONMENT.md; the complete local TTA installation retains them separately.')
    records = json.loads((REPO / 'environments/archives.json').read_text())
    for record in records:
        name = record['name']
        if args.only and name != args.only:
            continue
        target = REPO / 'envs' / name
        archive = REPO / record['archive']
        marker = target / '.tta-environment.json'
        expected = {'prefix': str(target), 'archive_sha256': record['sha256']}
        if marker.is_file() and json.loads(marker.read_text()) == expected and not args.recreate:
            print(name + ': already installed at this location', flush=True)
            continue
        if target.exists():
            if not args.recreate:
                raise RuntimeError(str(target) + ' exists without a matching location marker. Use --recreate to reinstall this packaged environment.')
            if target.is_symlink():
                raise RuntimeError('Refusing to delete an environment symlink')
            shutil.rmtree(target)
        print(name + ': checking archive, then unpacking', flush=True)
        if archive.stat().st_size != record['size'] or digest(archive) != record['sha256']:
            raise RuntimeError('Environment archive checksum mismatch: ' + name)
        target.mkdir(parents=True)
        with tarfile.open(archive, 'r:gz') as tar:
            # All archives are locally produced and hash-pinned; reject path escapes.
            for member in tar.getmembers():
                if member.name.startswith('/') or '..' in Path(member.name).parts:
                    raise RuntimeError('Archive path escape')
            tar.extractall(target)
        env = dict(os.environ)
        env.pop('PYTHONHOME', None)
        env.pop('PYTHONPATH', None)
        env['PATH'] = str(target / 'bin') + os.pathsep + env.get('PATH', '')
        subprocess.run([str(target / 'bin/python'), str(target / 'bin/conda-unpack')], env=env, check=True)
        if name == 'generation':
            for site in target.glob('lib/python*/site-packages'):
                for filename, vendor in [
                    ('__editable__.flexavatar-0.0.1.pth', 'third_party/FlexAvatar/src'),
                    ('__editable__.modnet-0.0.1.pth', 'third_party/FlexAvatar_MODNet/src'),
                    ('__editable__.stylegan_generators-0.0.4.pth', 'third_party/FlexAvatar_stylegan_generators/src')]:
                    source = REPO / vendor
                    if not source.is_dir():
                        raise RuntimeError('Missing vendored editable source: ' + vendor)
                    (site / filename).write_text(os.path.relpath(source, site) + '\n')
        marker.write_text(json.dumps(expected, indent=2) + '\n')
        subprocess.run([str(target / 'bin/python'), '-c', 'import sys; print(sys.version); print(sys.prefix)'], env=env, check=True)
        print(name + ': ready', flush=True)

if __name__ == '__main__':
    main()
