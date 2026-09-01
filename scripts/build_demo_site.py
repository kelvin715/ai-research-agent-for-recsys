#!/usr/bin/env python3
"""Freeze one or more experiment runs into a static dashboard site for GitHub Pages.

The live dashboard reads two dynamic endpoints from `orchestrator/graph_view.py`:
`/api/state` and `/artifact/<path>`.  GitHub Pages serves files only, so this script
calls `build_graph_state()` once per run, writes the result to `state.json`, copies
every artifact that state references, and emits an `index.html` whose `<html>` carries
`data-static-base` — the flag `assets/app.js` uses to read the snapshot instead of the
controller.  The same `app.js` and `styles.css` drive both modes.

    python3 scripts/build_demo_site.py

Output lands in `docs/` so Pages can serve the repository's `/docs` folder directly.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "orchestrator"))

import graph_view  # noqa: E402  (flat imports, see CLAUDE.md)

SITE = ROOT / "docs"
DEMO = SITE / "demo"
STATIC_SOURCE = ROOT / "visualization" / "experiment-graph"

# Run directories to freeze. The submitted run ships inside the repository; the two
# extended searches are read from the local workspace because their full seed
# workspaces are far too large to commit.
RUNS = [
    {
        "slug": "submitted-run",
        "title": "Submitted run · GPT-5.4",
        "blurb": "The challenge-compliant run behind the submission: warm start reproduced, "
                 "then three new experiments before the N=3 stopping rule fired.",
        "source": ROOT / "artifacts" / "experiment-records",
    },
    {
        "slug": "long-run-gpt54",
        "title": "Extended search · GPT-5.4",
        "blurb": "13 rounds with the convergence stop lifted. Includes the strongest "
                 "challenger of either run, rejected on matched-seed evidence.",
        "source": ROOT.parent / "runs" / "pilot-15round-gpt54-20260831-003",
    },
    {
        "slug": "long-run-gpt56sol",
        "title": "Extended search · GPT-5.6-sol",
        "blurb": "15 rounds to the declared iteration cap, same controller on a different "
                 "model. Same parent rotation, same verdicts.",
        "source": ROOT.parent / "runs" / "pilot-15round-gpt56sol-20260831-001",
    },
]


def collect_artifact_paths(state: dict) -> set[str]:
    """Every run-relative path the frozen state can link to."""
    found: set[str] = set()

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "artifacts" and isinstance(item, dict):
                    found.update(str(v) for v in item.values() if isinstance(v, str))
                elif key == "artifact" and isinstance(item, str):
                    found.add(item)
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(state)
    return found


def freeze_run(run: dict) -> dict | None:
    source = run["source"]
    if not (source / "frontier.json").is_file():
        print(f"skip {run['slug']}: no frontier.json under {source}")
        return None

    target = DEMO / run["slug"]
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    state = graph_view.build_graph_state(source)
    (target / "state.json").write_text(json.dumps(state, indent=1, sort_keys=True) + "\n",
                                       encoding="utf-8")

    copied, skipped, total_bytes = 0, 0, 0
    for relative in sorted(collect_artifact_paths(state)):
        origin = (source / relative).resolve()
        try:
            origin.relative_to(source.resolve())
        except ValueError:
            skipped += 1
            continue
        if not origin.is_file() or origin.suffix.lower() not in graph_view.TEXT_ARTIFACT_SUFFIXES:
            skipped += 1
            continue
        destination = target / "artifact" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, destination)
        copied += 1
        total_bytes += destination.stat().st_size

    index = (STATIC_SOURCE / "index.html").read_text(encoding="utf-8")
    # Absolute asset paths only work at a server root; the site lives under /demo/<slug>/.
    index = index.replace('href="/assets/', 'href="../assets/')
    index = index.replace('src="/assets/', 'src="../assets/')
    if 'data-static-base' not in index:
        index = index.replace("<html ", '<html data-static-base="." ', 1)
    banner = (
        '<div style="position:fixed;left:0;right:0;bottom:0;z-index:99;padding:7px 14px;'
        'font:600 11px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#70809a;'
        'background:rgba(255,255,255,.93);border-top:1px solid #e4eaf3;text-align:center">'
        f'Frozen snapshot of <code>{html.escape(run["slug"])}</code> — read-only replay. '
        '<a href="../../" style="color:#6688d9">All runs</a> · '
        '<a href="https://github.com/kelvin715/ai-research-agent-for-recsys" '
        'style="color:#6688d9">Repository</a></div>'
    )
    index = index.replace("</body>", banner + "\n</body>", 1)
    (target / "index.html").write_text(index, encoding="utf-8")

    summary = {**run, "nodes": len(state.get("nodes", [])), "artifacts": copied,
               "bytes": total_bytes}
    print(f"{run['slug']}: {summary['nodes']} nodes, {copied} artifacts "
          f"({total_bytes / 1024:.0f} KB), {skipped} skipped")
    return summary


def write_landing(frozen: list[dict]) -> None:
    cards = "\n".join(
        f"""      <a class="card" href="demo/{html.escape(run['slug'])}/">
        <h2>{html.escape(run['title'])}</h2>
        <p>{html.escape(run['blurb'])}</p>
        <span class="meta">{run['nodes']} experiment nodes · {run['artifacts']} linked evidence files</span>
      </a>"""
        for run in frozen
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Experiment graphs · AI Research Agent for Recommendation</title>
<style>
  :root {{ color-scheme: light; --ink:#24334d; --muted:#70809a; --line:#e4eaf3; --blue:#6688d9; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; padding:56px 24px 72px; background:#f5f8fd; color:var(--ink);
         font:400 15px/1.65 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }}
  main {{ max-width: 880px; margin: 0 auto; }}
  h1 {{ font-size: 30px; margin: 0 0 10px; letter-spacing: -0.02em; }}
  .lede {{ color: var(--muted); margin: 0 0 8px; }}
  .note {{ color: var(--muted); font-size: 13px; margin: 0 0 34px; }}
  .grid {{ display: grid; gap: 16px; }}
  .card {{ display:block; padding:22px 24px; background:#fff; border:1px solid var(--line);
          border-radius:14px; text-decoration:none; color:inherit;
          transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease; }}
  .card:hover {{ transform: translateY(-2px); border-color:#ccd8ea;
                box-shadow: 0 8px 22px rgba(73,94,131,.09); }}
  .card h2 {{ font-size: 17px; margin: 0 0 7px; }}
  .card p {{ margin: 0 0 12px; color: var(--muted); font-size: 14px; }}
  .meta {{ font-size: 11.5px; font-weight: 700; color: #98a5b8;
          text-transform: uppercase; letter-spacing: .06em; }}
  footer {{ margin-top: 38px; font-size: 13px; color: var(--muted); }}
  a.plain {{ color: var(--blue); }}
</style>
</head>
<body>
  <main>
    <h1>Experiment graphs</h1>
    <p class="lede">Read-only replays of the search graphs saved by the research agent.
      Select a node to see its hypothesis, parent, three-seed scores, paired interval,
      effect on the deployed combination, and the exact evidence files.</p>
    <p class="note">These are frozen snapshots. The identical dashboard runs live against
      an in-progress experiment with
      <code>python3 scripts/serve_experiment_graph.py --run-dir runs/&lt;id&gt;</code>.</p>
    <div class="grid">
{cards}
    </div>
    <footer>
      <a class="plain" href="https://github.com/kelvin715/ai-research-agent-for-recsys">
        Source repository</a> · TikTok TechJam 2026 Track 2
    </footer>
  </main>
</body>
</html>
"""
    (SITE / "index.html").write_text(page, encoding="utf-8")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    DEMO.mkdir(parents=True, exist_ok=True)
    shared = DEMO / "assets"
    if shared.exists():
        shutil.rmtree(shared)
    shared.mkdir(parents=True)
    for name in ("styles.css", "app.js"):
        shutil.copy2(STATIC_SOURCE / "assets" / name, shared / name)

    # Tell GitHub Pages to serve the tree verbatim instead of running Jekyll over it.
    (SITE / ".nojekyll").touch()

    frozen = [summary for run in RUNS if (summary := freeze_run(run))]
    if not frozen:
        print("no runs frozen", file=sys.stderr)
        return 1
    write_landing(frozen)
    print(f"wrote {(SITE / 'index.html').relative_to(ROOT)} with {len(frozen)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
