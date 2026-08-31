"""Apply structured block replacements without executing LLM-provided commands."""
import os


def parse_blocks(source):
    lines = source.splitlines()
    starts = {}
    spans = {}
    for idx, line in enumerate(lines):
        text = line.strip()
        if text.startswith('# <<<BLOCK:') and text.endswith('>>>'):
            name = text[len('# <<<BLOCK:'):-3]
            if name in starts or name in spans:
                raise ValueError(f'重复 block: {name}')
            starts[name] = idx
        elif text.startswith('# <<<END:') and text.endswith('>>>'):
            name = text[len('# <<<END:'):-3]
            if name not in starts:
                raise ValueError(f'没有起点的 END: {name}')
            spans[name] = (starts.pop(name), idx)
    if starts:
        raise ValueError(f'未闭合 block: {sorted(starts)}')
    return lines, spans


def apply_replacements(parent_source, replacements):
    lines, spans = parse_blocks(parent_source)
    by_block = {item['block']: item['code'].strip('\n').splitlines()
                for item in replacements}
    for block in sorted(by_block, key=lambda name: spans[name][0], reverse=True):
        lo, hi = spans[block]
        lines[lo + 1:hi] = by_block[block]
    return '\n'.join(lines) + '\n'


def write_candidate(parent_path, patch_obj, out_path):
    with open(parent_path, encoding='utf-8') as fh:
        parent = fh.read()
    candidate = apply_replacements(parent, patch_obj['replacements'])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(candidate)
    return out_path
