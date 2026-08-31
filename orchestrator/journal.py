"""Append-only JSONL journal with fsync for crash-safe experiment accounting."""
import json
import os


def append(path, event):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(event, ensure_ascii=False, sort_keys=True)
    with open(path, 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')
        fh.flush()
        os.fsync(fh.fileno())


def write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(value, fh, indent=2, ensure_ascii=False, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
