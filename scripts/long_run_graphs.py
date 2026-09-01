#!/usr/bin/env python3
"""Export and render the extended (stop-rule-lifted) search runs.

Two modes:

  export   read a full ``runs/<run-id>/`` workspace and distil the decision record
           into ``artifacts/long-run-records/<run-id>.json``
  render   read the distilled records and draw the experiment-graph figures under
           ``docs/assets/``

The distilled record contains exactly the numbers the README quotes, so every claim
about these runs can be checked without shipping the multi-gigabyte seed workspaces.

    python3 scripts/long_run_graphs.py export --run-dir ../runs/<run-id>
    python3 scripts/long_run_graphs.py render
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "artifacts" / "long-run-records"
ASSETS = ROOT / "docs" / "assets"

# Same visual language as visualization/experiment-graph/assets/styles.css.
COLORS = {
    "BASELINE": "#86a7ed",
    "ACCEPT": "#7bc8aa",
    "UNCERTAIN": "#e7b26f",
    "ROLLBACK": "#e29aa5",
    "REJECT": "#aeb8c6",
    "ORCHESTRATOR_ERROR": "#aeb8c6",
    "PLANNING_ERROR": "#aeb8c6",
}
INK = "#24334d"
MUTED = "#70809a"
FAINT = "#99a8bd"
LINE = "#dfe6f1"
PORTFOLIO = "#51a988"
EDGE = "#b3bfd1"


# --------------------------------------------------------------------------- export


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def export(run_dir: Path) -> Path:
    frontier = _load(run_dir / "frontier.json")
    if frontier is None:
        raise SystemExit(f"no frontier.json under {run_dir}")
    config = _load(run_dir / "config.json") or {}
    summary = _load(run_dir / "summary.json")
    journal = [
        json.loads(line)
        for line in (run_dir / "journal.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    nodes = []
    for node in frontier["nodes"]:
        nodes.append(
            {
                key: node[key]
                for key in (
                    "node_id",
                    "parent_node_id",
                    "children",
                    "decision",
                    "status",
                    "execution_mode",
                    "mechanism",
                    "operator_stack",
                    "selection_primary",
                    "pipeline_sha256",
                    "times_selected_as_parent",
                )
                if key in node
            }
        )
    by_node = {n["node_id"]: n for n in nodes}

    journal_by_iter = {entry["iter"]: entry for entry in journal}
    iterations = []
    for iter_dir in sorted(run_dir.glob("iter-*")):
        number = int(iter_dir.name.split("-")[1])
        proposal = _load(iter_dir / "proposal.json") or {}
        metrics = _load(iter_dir / "metrics.json")
        selection = _load(iter_dir / "ensemble-selection.json")
        reflection = _load(iter_dir / "reflection.json") or {}
        entry = journal_by_iter.get(number, {})
        outcome = entry.get("outcome") or {}
        node_id = f"n{number:03d}"

        record = {
            "iteration": number,
            "node_id": node_id if node_id in by_node else None,
            # No journal entry means the process was killed inside this iteration.
            "decision": outcome.get("decision") if entry else "TERMINATED",
            "status": entry.get("status") if entry else "TERMINATED",
            "execution_mode": proposal.get("execution_mode"),
            "mechanism": proposal.get("mechanism") or proposal.get("hypothesis"),
            "primary_block": proposal.get("primary_block"),
            "basis_type": proposal.get("basis_type"),
            "parent_node_id": (by_node.get(node_id) or {}).get("parent_node_id"),
            "implementation_attempts": len(list(iter_dir.glob("attempt-*"))),
            "planning_attempts": len(list(iter_dir.glob("planning-attempt-*"))),
            "reflection_result": reflection.get("result"),
            "next_lesson": reflection.get("next_lesson"),
            "wall_s": entry.get("wall_s"),
            "usage": entry.get("usage"),
        }
        if outcome.get("reason"):
            record["failure_reason"] = outcome["reason"]

        if metrics:
            paired = metrics.get("paired_vs_incumbent") or {}
            record.update(
                {
                    "standalone_primary": metrics.get("selection_primary"),
                    "GAUC_mean": metrics.get("GAUC_mean"),
                    "nDCG@5_mean": metrics.get("nDCG@5_mean"),
                    "paired_delta_vs_parent": paired.get("delta_primary"),
                    "paired_ci95_vs_parent": paired.get("paired_ci95"),
                    "paired_excludes_zero": paired.get("excludes_zero"),
                }
            )
        if selection:
            comparison = selection.get("incumbent_comparison") or {}
            entered = bool(comparison.get("candidate_entered"))
            record.update(
                {
                    "deployed_primary_after": selection.get("selection_primary"),
                    "promotion_gate": selection.get("promotion_gate"),
                    "candidate_entered_challenger": entered,
                    # A challenger that the candidate never entered is a carry-over
                    # from an earlier iteration; recording it here would misattribute
                    # that older portfolio delta to this experiment.
                    "challenger_primary": comparison.get("challenger_primary") if entered else None,
                    "challenger_delta_vs_incumbent": comparison.get("delta_primary") if entered else None,
                    "matched_seed_deltas": comparison.get("matched_seed_deltas") if entered else None,
                    "mean_matched_seed_delta": comparison.get("mean_matched_seed_delta") if entered else None,
                    "challenger_paired_ci95": comparison.get("paired_ci95") if entered else None,
                    "promoted": comparison.get("promoted"),
                }
            )
        iterations.append(record)

    usage = {
        "calls": sum(e["usage"]["calls"] for e in journal),
        "prompt_tokens": sum(e["usage"].get("prompt_tokens", 0) for e in journal),
        "completion_tokens": sum(e["usage"].get("completion_tokens", 0) for e in journal),
        "total_tokens": sum(e["usage"]["total_tokens"] for e in journal),
        # Run wall clock when the run finished; otherwise the sum of logged iterations.
        "wall_s": (summary or {}).get("wall_s") or sum(e["wall_s"] for e in journal),
    }

    counts: dict[str, int] = {}
    for item in iterations:
        counts[item["decision"] or "UNKNOWN"] = counts.get(item["decision"] or "UNKNOWN", 0) + 1

    record = {
        "run_id": config.get("run_id", run_dir.name),
        "model": config.get("model"),
        "role": config.get("role"),
        "iterations_cap": config.get("iterations_cap"),
        "convergence_stop_enabled": bool((config.get("convergence") or {}).get("stopping_enabled", True)),
        "seeds": config.get("seeds"),
        "journal_iterations": len(journal),
        "terminated_early": summary is None,
        "stop_reason": (summary or {}).get("stop_reason", "process terminated before summary"),
        "iteration_decision_counts": counts,
        # The seed-ensemble score of the reproduced official baseline node, i.e. the
        # same quantity the graph plots for every other node.
        "baseline_selection_primary": next(
            (n["selection_primary"] for n in nodes if n["decision"] == "BASELINE"), None
        ),
        "deployed_portfolio_primary": (summary or {}).get("best_selection_primary")
        or next(
            (i["deployed_primary_after"] for i in reversed(iterations) if i.get("deployed_primary_after")),
            None,
        ),
        "best_new_single_model_primary": max(
            (n["selection_primary"] for n in nodes if n["decision"] not in ("BASELINE",) and not n["node_id"].startswith("w") and n["selection_primary"]),
            default=None,
        ),
        "usage": usage,
        "summary": {
            key: (summary or {}).get(key)
            for key in (
                "iterations_executed",
                "accepted",
                "uncertain",
                "rolled_back",
                "failed",
                "planning_errors",
                "manual_interventions",
                "test_labels_used_for_selection",
                "validation_labels_exposed_to_candidate",
                "post_run_integrity_gate",
                "convergence",
            )
            if summary
        },
        "label_boundary": {
            "test_labels_used_anywhere": any(
                (_load(p) or {}).get("test_labels_used") for p in run_dir.glob("iter-*/ensemble-selection.json")
            ),
            "submission_written": (run_dir / "submission.json").exists(),
            "note": "Validation-only pilot: no submission was produced and no hidden-test "
            "label entered any selection decision.",
        },
        "nodes": nodes,
        "iterations": iterations,
        "provenance": {
            "source_run_dir": str(run_dir),
            "note": "Distilled from the full run workspace: decisions, metrics and gate "
            "outcomes only. Seed prediction arrays and pipeline sources stay local.",
        },
    }

    RECORDS.mkdir(parents=True, exist_ok=True)
    out = RECORDS / f"{record['run_id']}.json"
    out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return out


# --------------------------------------------------------------------------- render

# A single shared score window keeps the two figures directly comparable. Nodes that
# fall outside it (one collapsed objective at 0.4776) and nodes that never produced a
# score are drawn in their own lanes below the window, with the real value in the label.
Y_WINDOW = (0.5950, 0.6068)
MEMBER_LABEL_OFFSETS = {"w001": (-36, 13), "w002": (32, -20), "w003": (-6, -21)}


def _decision_color(decision: str | None) -> str:
    return COLORS.get(decision or "", COLORS["REJECT"])


def render(record_path: Path, out_path: Path, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    record = json.loads(record_path.read_text(encoding="utf-8"))
    nodes = {n["node_id"]: n for n in record["nodes"]}
    iterations = record["iterations"]
    baseline = record["baseline_selection_primary"]
    deployed = record["deployed_portfolio_primary"]
    max_iter = max(i["iteration"] for i in iterations)

    span = Y_WINDOW[1] - Y_WINDOW[0]
    lane_off = Y_WINDOW[0] - 0.22 * span
    lane_none = Y_WINDOW[0] - 0.40 * span

    # x position: the three warm-start members share iteration 1, spread for legibility.
    members = sorted(n for n in nodes if n.startswith("w"))
    x_of = {"n000": 0.0}
    for index, member in enumerate(members):
        x_of[member] = 0.66 + 0.34 * index
    for item in iterations:
        if item.get("node_id"):
            x_of[item["node_id"]] = float(item["iteration"])

    def y_of(node_id: str) -> float:
        value = nodes[node_id].get("selection_primary")
        if not value:
            return lane_none
        return value if value >= Y_WINDOW[0] else lane_off

    fig = plt.figure(figsize=(14.0, 8.4), dpi=170)
    fig.patch.set_facecolor("#ffffff")
    grid = fig.add_gridspec(
        2, 1, height_ratios=[3.0, 1.0], hspace=0.34,
        left=0.068, right=0.985, top=0.868, bottom=0.085,
    )
    ax = fig.add_subplot(grid[0])
    ax2 = fig.add_subplot(grid[1])

    for axis in (ax, ax2):
        axis.set_facecolor("#fbfcfe")
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(LINE)
        axis.tick_params(colors=MUTED, labelsize=8.5, length=3, color=LINE)
        axis.set_axisbelow(True)

    # keep a clean right margin for the reference-line labels
    right_margin = 3.1
    ax.set_xlim(-0.62, max_iter + right_margin)
    ax.set_ylim(lane_none - 0.10 * span, Y_WINDOW[1] + 0.05 * span)
    ax.grid(True, axis="y", color=LINE, linewidth=0.7, alpha=0.85)

    # --- lanes and reference levels ---------------------------------------
    ax.axhspan(lane_none - 0.10 * span, Y_WINDOW[0] - 0.06 * span, color="#f2f5fa", zorder=0)
    ax.axhline(deployed, color=PORTFOLIO, linewidth=1.7, linestyle=(0, (2, 3)), zorder=2)
    ax.axhline(baseline, color=COLORS["BASELINE"], linewidth=1.4, linestyle=(0, (1, 3)), zorder=2)
    label_x = max_iter + 0.42
    ax.text(
        label_x, deployed, f"  deployed portfolio\n  {deployed:.6f}",
        color=PORTFOLIO, fontsize=8.6, fontweight="bold", va="center", ha="left",
    )
    ax.text(
        label_x, baseline, f"  official FM baseline\n  {baseline:.6f}",
        color=COLORS["BASELINE"], fontsize=8.6, fontweight="bold", va="center", ha="left",
    )
    has_off_scale = any(
        0 < (n.get("selection_primary") or 0) < Y_WINDOW[0] for n in nodes.values()
    )
    lanes = [(lane_none, "no score\ngate or planning stop")]
    if has_off_scale:
        lanes.append((lane_off, "off scale"))
    for lane_y, lane_text in lanes:
        ax.text(
            label_x, lane_y, f"  {lane_text}",
            color=FAINT, fontsize=8.0, fontweight="bold", va="center", ha="left",
        )

    # --- edges ------------------------------------------------------------
    for node in nodes.values():
        parent = node.get("parent_node_id")
        if not parent or parent not in nodes:
            continue
        ax.annotate(
            "",
            xy=(x_of[node["node_id"]], y_of(node["node_id"])),
            xytext=(x_of[parent], y_of(parent)),
            arrowprops=dict(
                arrowstyle="-|>", color=EDGE, linewidth=0.85, alpha=0.6,
                connectionstyle="arc3,rad=0.16", shrinkA=9, shrinkB=9,
            ),
            zorder=3,
        )

    # --- nodes ------------------------------------------------------------
    for index, (node_id, node) in enumerate(nodes.items()):
        x, y = x_of[node_id], y_of(node_id)
        value = node.get("selection_primary")
        color = _decision_color(node["decision"])
        anchor = node_id.startswith("w") or node_id == "n000"
        ax.scatter(
            [x], [y], s=330 if anchor else 205, c=color,
            edgecolors="#ffffff", linewidths=1.9, zorder=5,
            marker="o" if value else "X",
        )
        short = f"{node_id[0]}{int(node_id[1:])}"
        if value:
            ax.text(x, y, short, color="#ffffff", fontsize=6.7, fontweight="bold",
                    ha="center", va="center", zorder=6)
        else:
            ax.annotate(short, (x, y), textcoords="offset points", xytext=(0, -15),
                        ha="center", fontsize=7.2, color=FAINT, fontweight="bold", zorder=6)
        if not value:
            continue
        # the three warm-start members sit close together, so stagger their labels
        dx, dy = MEMBER_LABEL_OFFSETS.get(node_id, (0, 13))
        text = f"{value:.6f}" if value >= Y_WINDOW[0] else f"{value:.6f}  (off scale)"
        ax.annotate(
            text, (x, y), textcoords="offset points", xytext=(dx, dy),
            ha="center", fontsize=7.1, color=MUTED, fontweight="bold", zorder=6,
        )

    # Iterations that never produced a node at all (proposal-schema or controller
    # failures) would otherwise be invisible; place them in the no-score lane.
    for item in iterations:
        if item.get("node_id") or item["decision"] == "WARMSTART_VERIFIED":
            continue
        x = float(item["iteration"])
        ax.scatter([x], [lane_none], s=205, c=COLORS["REJECT"], edgecolors="#ffffff",
                   linewidths=1.9, zorder=5, marker="X", alpha=0.75)
        ax.annotate(
            f"it{item['iteration']}", (x, lane_none), textcoords="offset points",
            xytext=(0, -15), ha="center", fontsize=7.2, color=FAINT,
            fontweight="bold", zorder=6,
        )

    ticks = [round(Y_WINDOW[0] + i * span / 4, 4) for i in range(5)]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:.4f}" for t in ticks])
    ax.set_ylabel("standalone validation primary", color=INK, fontsize=9.5, fontweight="bold")
    ax.set_xticks([0] + list(range(1, max_iter + 1)))
    ax.set_xticklabels(["warm\nstart"] + [str(i) for i in range(1, max_iter + 1)], fontsize=8)

    legend = [
        Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=COLORS["BASELINE"], markeredgecolor="w", label="baseline"),
        Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=COLORS["ACCEPT"], markeredgecolor="w", label="accepted"),
        Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=COLORS["UNCERTAIN"], markeredgecolor="w", label="uncertain"),
        Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=COLORS["ROLLBACK"], markeredgecolor="w", label="rolled back"),
        Line2D([], [], marker="X", linestyle="", markersize=8, markerfacecolor=COLORS["REJECT"], markeredgecolor="w", label="stopped before scoring"),
    ]
    ax.legend(
        handles=legend, loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=5, frameon=False,
        fontsize=8.4, labelcolor=MUTED, handletextpad=0.35, columnspacing=1.6,
    )

    # --- lower panel: effect on the deployed portfolio --------------------
    entered = [i for i in iterations if i.get("candidate_entered_challenger")]
    ax2.set_xlim(-0.62, max_iter + right_margin)
    ax2.grid(True, axis="y", color=LINE, linewidth=0.7, alpha=0.85)
    ax2.axhline(0.0, color=PORTFOLIO, linewidth=1.5, linestyle=(0, (2, 3)), zorder=2)
    ax2.text(
        max_iter + 0.42, 0.0, "  deployed portfolio\n  (unchanged all run)",
        color=PORTFOLIO, fontsize=8.2, fontweight="bold", va="center", ha="left",
    )
    for item in entered:
        x = item["iteration"]
        delta = item["challenger_delta_vs_incumbent"]
        seeds = item.get("matched_seed_deltas") or []
        ax2.bar(
            [x], [delta], width=0.40,
            color=COLORS["UNCERTAIN"] if delta > 0 else COLORS["ROLLBACK"],
            edgecolor="white", linewidth=1.0, zorder=4, alpha=0.92,
        )
        ax2.scatter(
            [x] * len(seeds), seeds, s=27, facecolor="white",
            edgecolor=INK, linewidths=1.0, zorder=6,
        )
        top = max([delta] + seeds)
        ax2.annotate(
            f"n{x:03d}  {delta:+.6f}\nnot promoted",
            (x, top), textcoords="offset points", xytext=(0, 14), ha="center",
            fontsize=7.2, color=MUTED, fontweight="bold", zorder=7,
        )
    ax2.set_xticks(list(range(1, max_iter + 1)))
    ax2.set_xticklabels([str(i) for i in range(1, max_iter + 1)], fontsize=8)
    ax2.set_xlabel("iteration", color=INK, fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("challenger − deployed", color=INK, fontsize=9.5, fontweight="bold")
    ax2.text(
        0.0, 1.09,
        "portfolio effect, shown only for candidates that actually entered the best challenger"
        "  ·  white dots = the three matched-seed deltas",
        transform=ax2.transAxes, fontsize=8.2, color=MUTED, fontweight="bold",
    )
    if entered:
        reach = max(
            abs(v) for i in entered
            for v in ([i["challenger_delta_vs_incumbent"]] + (i.get("matched_seed_deltas") or []))
        )
        ax2.set_ylim(-1.9 * reach, 3.2 * reach)

    fig.text(0.068, 0.955, title, color=INK, fontsize=15.0, fontweight="bold", ha="left")
    subtitle = (
        f"{record['journal_iterations']} logical iterations · convergence stop lifted · "
        f"{record['usage']['calls']} LLM calls · {record['usage']['total_tokens']:,} tokens · "
        f"{record['usage']['wall_s'] / 60:.0f} min · 3 fixed seeds per scored node · "
        "0 human interventions"
    )
    fig.text(0.068, 0.921, subtitle, color=MUTED, fontsize=9.0, ha="left")

    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    return out_path


FIGURES = {
    "pilot-15round-gpt54-20260831-003": (
        "experiment-graph-long-run-gpt54.png",
        "Extended search · GPT-5.4 · stop rule lifted",
    ),
    "pilot-15round-gpt56sol-20260831-001": (
        "experiment-graph-long-run-gpt56sol.png",
        "Extended search · GPT-5.6-sol · stop rule lifted",
    ),
}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    exporter = sub.add_parser("export")
    exporter.add_argument("--run-dir", required=True, type=Path)
    sub.add_parser("render")
    args = parser.parse_args(argv)

    if args.mode == "export":
        print(f"wrote {export(args.run_dir).relative_to(ROOT)}")
        return 0

    for run_id, (filename, title) in FIGURES.items():
        source = RECORDS / f"{run_id}.json"
        if not source.exists():
            print(f"skip {run_id}: no distilled record")
            continue
        print(f"wrote {render(source, ASSETS / filename, title).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
