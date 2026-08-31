"""Record the exact candidate environment and its importable package roots."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pkgutil
import sys
import sysconfig


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def environment_lock(requirements: str) -> dict:
    packages = {
        distribution.metadata['Name']: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get('Name')
    }
    mapping = importlib.metadata.packages_distributions()
    import_roots = {
        root for root, distributions in mapping.items()
        if root and distributions and root.isidentifier()
    }
    # Several modern binary wheels omit ``top_level.txt``, so
    # packages_distributions() alone misses roots such as numpy/lightgbm/sklearn.
    # Enumerating only this venv's site-packages (not the stdlib) provides the
    # actual candidate import surface while keeping the lock deterministic.
    site_directories = {
        path for key, path in sysconfig.get_paths().items()
        if key in {'purelib', 'platlib'} and path and os.path.isdir(path)
    }
    for module in pkgutil.iter_modules(sorted(site_directories)):
        if module.name.isidentifier():
            import_roots.add(module.name)
    import_roots = sorted(import_roots)
    return {
        'schema_version': 'candidate-env-2.1',
        'python': sys.version.split()[0],
        'requirements_file': os.path.basename(requirements),
        'requirements_sha256': sha256(requirements),
        'packages': dict(sorted(packages.items(), key=lambda item: item[0].lower())),
        'import_roots': import_roots,
        'policy': {
            'runtime_installation': False,
            'network': False,
            'filesystem': 'read_only_pinned_environment',
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--requirements', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    payload = environment_lock(args.requirements)
    with open(args.out, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')
    print(f"  locked {len(payload['packages'])} distributions / "
          f"{len(payload['import_roots'])} import roots -> {args.out}")


if __name__ == '__main__':
    main()
