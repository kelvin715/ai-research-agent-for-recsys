"""Curated research corpus used only by the explicit offline control.

为什么这是必需的而不是可选变体：官方任务要求 2 明写 agent 必须
"autonomously draw on established methods from both industry and academia"，
Innovation & Problem Insight（20%）直接评 "Originality in drawing on published methods,
papers, or public solutions — rewarding agents that go beyond naive baseline tweaks"。

正式 live 模式由 trusted orchestrator 在候选生成前调用 hosted WebSearchTool，并把完整来源
轨迹冻结到 run snapshot；候选沙箱始终断网。这里的 M01-M08 只用于 offline 消融或回归测试，
不再是 live run 的默认或强制输入。replay 模式可重用 live snapshot，得到固定研究条件。

方法卡的内容边界（计划 §6.3）：只写机制、映射到哪个 block、需要哪个 API、numpy 可行性、出处。
**不写预期 delta，不排优先级** —— 排序必须由 agent 自己基于证据做出，否则就成了人工路线。
"""
import os
import re

LIBRARY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'research_library')

FRONTMATTER_RE = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.S)


def _parse_card(path):
    with open(path, encoding='utf-8') as fh:
        raw = fh.read()
    meta, body = {}, raw
    m = FRONTMATTER_RE.match(raw)
    if m:
        body = raw[m.end():]
        for line in m.group(1).splitlines():
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            value = value.strip()
            if value.startswith('[') and value.endswith(']'):
                value = [x.strip() for x in value[1:-1].split(',') if x.strip()]
            meta[key.strip()] = value
    meta['body'] = body.strip()
    meta.setdefault('id', os.path.splitext(os.path.basename(path))[0])
    meta.setdefault('title', meta['id'])
    meta.setdefault('blocks', [])
    return meta


def load_library(library_dir=LIBRARY_DIR):
    if not os.path.isdir(library_dir):
        return []
    return [_parse_card(os.path.join(library_dir, name))
            for name in sorted(os.listdir(library_dir)) if name.endswith('.md')]
