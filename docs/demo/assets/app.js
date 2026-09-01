(() => {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";
  const NODE_W = 208;
  const NODE_H = 104;
  const X_GAP = 286;
  const Y_GAP = 142;
  const PADDING = 70;
  const REFRESH_MS = 2000;

  // Exported static snapshots set data-static-base on <html>.  When it is present the
  // dashboard reads a frozen state.json and pre-copied artifacts instead of the live
  // controller endpoints, so one codebase drives both the local server and the demo site.
  const STATIC_BASE = document.documentElement.dataset.staticBase || null;

  const palette = {
    baseline: { fill: "#f3f6ff", stroke: "#9bb0df", accent: "#7d98d5", halo: "#dce6fb", text: "#6178ad" },
    accepted: { fill: "#f0faf6", stroke: "#8fcbb4", accent: "#65b291", halo: "#d9f0e7", text: "#43866c" },
    uncertain: { fill: "#fff9ed", stroke: "#e4bd83", accent: "#dba15b", halo: "#f6e7ca", text: "#9c723c" },
    rollback: { fill: "#fff4f5", stroke: "#dda4ad", accent: "#ca7b88", halo: "#f5dfe3", text: "#a65e69" },
    failed: { fill: "#f6f8fa", stroke: "#bbc4d0", accent: "#98a4b3", halo: "#e4e8ed", text: "#6f7c8e" },
    planning_failed: { fill: "#f8f8fa", stroke: "#c6ccd5", accent: "#a7b0bd", halo: "#eaedf1", text: "#747f8f" },
    running: { fill: "#f2f6ff", stroke: "#9bb5ee", accent: "#7397e1", halo: "#dce7ff", text: "#5f78b4" },
    paused: { fill: "#fff7f0", stroke: "#dfb898", accent: "#cd9367", halo: "#f7e5d6", text: "#936b4c" },
    portfolio: { fill: "#eefaf5", stroke: "#7fc1a8", accent: "#52a783", halo: "#d3eee3", text: "#418069" },
    recorded: { fill: "#f6f8fb", stroke: "#b9c3d2", accent: "#95a3b6", halo: "#e2e8f0", text: "#6f7f95" }
  };

  const state = {
    data: null,
    nodeMap: new Map(),
    positions: new Map(),
    selectedId: null,
    edgeFilters: new Set(["execution", "reference", "portfolio"]),
    transform: { x: 40, y: 40, scale: 1 },
    fitted: false,
    dragging: false,
    dragOrigin: null,
    refreshTimer: null,
    inFlight: false,
    firstLoad: true
  };

  const el = id => document.getElementById(id);
  const svg = el("graph-svg");
  const scene = el("graph-scene");
  const nodeLayer = el("node-layer");
  const edgeLayer = el("edge-layer");
  const viewport = el("graph-viewport");

  function svgElement(name, attrs = {}) {
    const node = document.createElementNS(SVG_NS, name);
    Object.entries(attrs).forEach(([key, value]) => {
      if (value !== undefined && value !== null) node.setAttribute(key, String(value));
    });
    return node;
  }

  function cleanText(value, fallback = "") {
    if (value === undefined || value === null) return fallback;
    return String(value).replace(/\s+/g, " ").trim();
  }

  function formatScore(value, digits = 6) {
    return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
  }

  function formatDelta(value, digits = 6) {
    if (typeof value !== "number" || !Number.isFinite(value)) return "—";
    return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
  }

  function formatCompact(value) {
    if (value === undefined || value === null) return "—";
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    if (Math.abs(number) >= 1_000_000) return `${(number / 1_000_000).toFixed(1)}M`;
    if (Math.abs(number) >= 1_000) return `${(number / 1_000).toFixed(1)}k`;
    return number.toFixed(0);
  }

  function deltaClass(value) {
    if (typeof value !== "number" || Math.abs(value) < 1e-12) return "neutral";
    return value > 0 ? "positive" : "negative";
  }

  function wrapLines(text, maxChars = 28, maxLines = 2) {
    const words = cleanText(text, "Untitled experiment").split(" ");
    const lines = [];
    let line = "";
    for (const word of words) {
      const candidate = line ? `${line} ${word}` : word;
      if (candidate.length > maxChars && line) {
        lines.push(line);
        line = word;
        if (lines.length === maxLines - 1) break;
      } else {
        line = candidate;
      }
    }
    const usedWords = lines.join(" ").split(" ").filter(Boolean).length;
    if (line && lines.length < maxLines) {
      const remaining = words.length > usedWords + line.split(" ").length;
      lines.push(remaining ? `${line.replace(/[.,;:]$/, "")}…` : line);
    }
    return lines.slice(0, maxLines);
  }

  function computeDepths(nodes, edges) {
    const map = new Map(nodes.map(node => [node.id, node]));
    const executionParents = new Map();
    edges.filter(edge => edge.type === "execution").forEach(edge => executionParents.set(edge.target, edge.source));
    const memo = new Map();
    const visiting = new Set();

    function depth(id) {
      if (memo.has(id)) return memo.get(id);
      if (visiting.has(id)) return 0;
      visiting.add(id);
      const node = map.get(id);
      let value = 0;
      if (node?.kind === "portfolio") {
        value = Math.max(0, ...nodes.filter(item => item.kind !== "portfolio").map(item => depth(item.id))) + 1;
      } else {
        const parent = executionParents.get(id);
        value = parent && map.has(parent) ? depth(parent) + 1 : 0;
      }
      visiting.delete(id);
      memo.set(id, value);
      return value;
    }
    nodes.forEach(node => depth(node.id));

    // Long runs remain readable by placing each three-round group in a new time column.
    // The true execution parent is still shown by the solid edge; this only affects layout.
    const timedNodes = nodes
      .filter(node => ["experiment", "planning", "active"].includes(node.kind) && Number.isFinite(node.iteration))
      .sort((a, b) => a.iteration - b.iteration);
    const firstRound = timedNodes.length ? timedNodes[0].iteration : 2;
    timedNodes.forEach(node => {
      const parent = executionParents.get(node.id);
      const parentDepth = parent && memo.has(parent) ? memo.get(parent) : 1;
      const timeDepth = 2 + Math.floor(Math.max(0, node.iteration - firstRound) / 3);
      memo.set(node.id, Math.max(parentDepth + 1, timeDepth));
    });
    const portfolio = nodes.find(node => node.kind === "portfolio");
    if (portfolio) {
      memo.set(portfolio.id, Math.max(0, ...nodes
        .filter(node => node.kind !== "portfolio")
        .map(node => memo.get(node.id) || 0)) + 1);
    }
    return memo;
  }

  function layoutGraph(data) {
    const nodes = data.nodes || [];
    const depths = computeDepths(nodes, data.edges || []);
    const executionParents = new Map(
      (data.edges || []).filter(edge => edge.type === "execution")
        .map(edge => [edge.target, edge.source]));
    const branchMemo = new Map();
    function branchRoot(id) {
      if (branchMemo.has(id)) return branchMemo.get(id);
      let current = id;
      const visited = new Set();
      while (executionParents.has(current) && !visited.has(current)) {
        visited.add(current);
        const parent = executionParents.get(current);
        if (parent === "n000") break;
        current = parent;
      }
      branchMemo.set(id, current);
      return current;
    }
    const branchOrder = new Map(
      nodes.filter(node => node.kind === "warmstart")
        .sort((a, b) => a.id.localeCompare(b.id))
        .map((node, index) => [node.id, index]));
    const columns = new Map();
    nodes.forEach(node => {
      const d = depths.get(node.id) || 0;
      if (!columns.has(d)) columns.set(d, []);
      columns.get(d).push(node);
    });
    columns.forEach(column => column.sort((a, b) => {
      const kindOrder = { baseline: 0, warmstart: 1, experiment: 2, active: 3, planning: 4, portfolio: 5 };
      const branchA = branchOrder.get(branchRoot(a.id)) ?? 99;
      const branchB = branchOrder.get(branchRoot(b.id)) ?? 99;
      return branchA - branchB ||
        (kindOrder[a.kind] ?? 9) - (kindOrder[b.kind] ?? 9) ||
        (a.iteration ?? 999) - (b.iteration ?? 999) || a.id.localeCompare(b.id);
    }));

    const maxRows = Math.max(1, ...Array.from(columns.values()).map(column => column.length));
    const canvasHeight = Math.max(590, maxRows * Y_GAP + PADDING * 2);
    const positions = new Map();
    columns.forEach((column, depth) => {
      const columnHeight = (column.length - 1) * Y_GAP + NODE_H;
      const startY = Math.max(PADDING, (canvasHeight - columnHeight) / 2);
      column.forEach((node, index) => positions.set(node.id, {
        x: PADDING + depth * X_GAP,
        y: startY + index * Y_GAP,
        depth
      }));
    });
    const maxDepth = Math.max(0, ...Array.from(depths.values()));
    return {
      positions,
      width: PADDING * 2 + maxDepth * X_GAP + NODE_W,
      height: canvasHeight
    };
  }

  function edgePath(source, target, type) {
    const sx = source.x + NODE_W;
    const sy = source.y + NODE_H / 2;
    const tx = target.x;
    const ty = target.y + NODE_H / 2;
    const distance = Math.max(55, (tx - sx) * 0.5);
    if (type === "reference") {
      const bend = sy <= ty ? -24 : 24;
      return `M ${sx} ${sy} C ${sx + distance} ${sy + bend}, ${tx - distance} ${ty + bend}, ${tx} ${ty}`;
    }
    return `M ${sx} ${sy} C ${sx + distance} ${sy}, ${tx - distance} ${ty}, ${tx} ${ty}`;
  }

  function renderEdges(data) {
    edgeLayer.replaceChildren();
    (data.edges || []).forEach(edge => {
      if (!state.edgeFilters.has(edge.type)) return;
      const source = state.positions.get(edge.source);
      const target = state.positions.get(edge.target);
      if (!source || !target) return;
      const group = svgElement("g", { class: `edge-group edge-${edge.type}` });
      const pathData = edgePath(source, target, edge.type);
      group.append(svgElement("path", { class: `graph-edge ${edge.type}`, d: pathData }));
      if (edge.type === "portfolio" && typeof edge.weight === "number") {
        const mx = (source.x + NODE_W + target.x) / 2;
        const my = (source.y + target.y + NODE_H) / 2 - 6;
        const label = svgElement("text", { class: "edge-label", x: mx, y: my, "text-anchor": "middle" });
        label.textContent = `${Math.round(edge.weight * 100)}%`;
        group.append(label);
      }
      edgeLayer.append(group);
    });
  }

  function statusMeta(node) {
    return palette[node.status_key] || palette.recorded;
  }

  function renderNode(node) {
    const position = state.positions.get(node.id);
    const colors = statusMeta(node);
    const group = svgElement("g", {
      class: `graph-node ${node.id === state.selectedId ? "selected" : ""}`,
      transform: `translate(${position.x},${position.y})`,
      tabindex: "0",
      role: "button",
      "aria-label": `${node.label}, ${node.status_label}`,
      "data-node-id": node.id
    });

    group.append(svgElement("rect", {
      class: "node-halo", x: -4, y: -4, width: NODE_W + 8, height: NODE_H + 8,
      rx: 19, stroke: colors.halo
    }));
    group.append(svgElement("rect", {
      class: "node-shell", x: 0, y: 0, width: NODE_W, height: NODE_H,
      rx: 16, fill: colors.fill, stroke: colors.stroke
    }));
    group.append(svgElement("rect", {
      class: "node-accent", x: 0, y: 0, width: 5, height: NODE_H,
      rx: 3, fill: colors.accent
    }));

    if (node.kind === "active") {
      group.append(svgElement("rect", {
        class: "running-ring", x: 5, y: 5, width: NODE_W - 10, height: NODE_H - 10,
        rx: 13, stroke: colors.accent
      }));
    }

    const badgeWidth = Math.min(106, Math.max(48, cleanText(node.status_label).length * 5.1 + 17));
    group.append(svgElement("rect", {
      class: "node-badge", x: NODE_W - badgeWidth - 10, y: 10,
      width: badgeWidth, height: 19, rx: 9.5, fill: colors.halo
    }));
    const statusText = svgElement("text", {
      class: "node-status-text", x: NODE_W - badgeWidth / 2 - 10,
      y: 23, "text-anchor": "middle", fill: colors.text
    });
    statusText.textContent = cleanText(node.status_label).toUpperCase();
    group.append(statusText);

    const meta = svgElement("text", { class: "node-meta", x: 17, y: 22 });
    meta.textContent = node.kind === "portfolio" ? "FINAL RESULT" :
      node.kind === "warmstart" ? `WARM START · ${node.id.toUpperCase()}` :
      node.kind === "baseline" ? "STARTING POINT" :
      node.iteration !== null && node.iteration !== undefined ? `ROUND ${node.iteration} · ${node.id.toUpperCase()}` : node.id.toUpperCase();
    group.append(meta);

    const lines = wrapLines(node.label, 29, 2);
    const title = svgElement("text", { class: "node-title", x: 17, y: 44 });
    lines.forEach((line, index) => {
      const tspan = svgElement("tspan", { x: 17, dy: index === 0 ? 0 : 14 });
      tspan.textContent = line;
      title.append(tspan);
    });
    group.append(title);

    const scoreLabel = svgElement("text", { class: "node-score-label", x: 17, y: 85 });
    scoreLabel.textContent = node.kind === "portfolio" ? "COMBINED SCORE" : "VALIDATION";
    group.append(scoreLabel);
    const score = svgElement("text", { class: "node-score", x: 17, y: 99 });
    score.textContent = formatScore(node.score);
    group.append(score);

    if (node.is_portfolio_member && node.kind !== "portfolio") {
      const weight = svgElement("text", {
        class: "node-meta", x: NODE_W - 12, y: 94, "text-anchor": "end", fill: colors.text
      });
      weight.textContent = typeof node.portfolio_weight === "number" ?
        `${Math.round(node.portfolio_weight * 100)}% in best combination` : "BEST COMBINATION";
      group.append(weight);
    } else if (node.kind === "portfolio") {
      const crown = svgElement("path", {
        class: "portfolio-crown", d: "M178 81l5-8 5 8 6-8 2 13h-20z", stroke: colors.accent
      });
      group.append(crown);
    }

    const select = () => selectNode(node.id);
    group.addEventListener("click", event => { event.stopPropagation(); select(); });
    group.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); select(); }
    });
    return group;
  }

  function renderNodes(data) {
    nodeLayer.replaceChildren();
    (data.nodes || []).forEach(node => nodeLayer.append(renderNode(node)));
  }

  function applyTransform() {
    scene.setAttribute("transform", `translate(${state.transform.x} ${state.transform.y}) scale(${state.transform.scale})`);
  }

  function fitGraph(layout, animate = false) {
    const rect = viewport.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const scale = Math.min(1.05, Math.max(0.28, Math.min(
      (rect.width - 40) / Math.max(layout.width, 1),
      (rect.height - 40) / Math.max(layout.height, 1)
    )));
    state.transform = {
      scale,
      x: (rect.width - layout.width * scale) / 2,
      y: (rect.height - layout.height * scale) / 2
    };
    if (animate) scene.style.transition = "transform 240ms ease";
    applyTransform();
    if (animate) window.setTimeout(() => { scene.style.transition = ""; }, 260);
  }

  function renderGraph(data, preserveView = true) {
    state.nodeMap = new Map((data.nodes || []).map(node => [node.id, node]));
    const layout = layoutGraph(data);
    state.positions = layout.positions;
    renderEdges(data);
    renderNodes(data);
    if (!preserveView || !state.fitted) {
      requestAnimationFrame(() => fitGraph(layout));
      state.fitted = true;
    } else {
      applyTransform();
    }
    el("graph-loading").classList.add("hidden");
  }

  function section(id, value) {
    const container = el(`${id}-section`);
    const content = el(`detail-${id}`);
    if (cleanText(value)) {
      content.textContent = cleanText(value);
      container.classList.remove("hidden");
    } else {
      container.classList.add("hidden");
    }
  }

  function renderConfidence(interval) {
    const box = el("confidence-box");
    if (!Array.isArray(interval) || interval.length !== 2 || !interval.every(Number.isFinite)) {
      box.classList.add("hidden");
      return;
    }
    box.classList.remove("hidden");
    el("confidence-label").textContent = `[${formatDelta(interval[0])}, ${formatDelta(interval[1])}]`;
    const span = Math.max(Math.abs(interval[0]), Math.abs(interval[1]), 0.000001);
    const left = ((interval[0] + span) / (2 * span)) * 100;
    const right = ((interval[1] + span) / (2 * span)) * 100;
    const range = el("confidence-range");
    range.style.left = `${Math.max(0, Math.min(100, left))}%`;
    range.style.width = `${Math.max(1, Math.min(100, right) - Math.max(0, left))}%`;
    el("confidence-zero").style.left = "50%";
  }

  function selectNode(nodeId) {
    const node = state.nodeMap.get(nodeId);
    if (!node) return;
    state.selectedId = nodeId;
    document.querySelectorAll(".graph-node").forEach(item =>
      item.classList.toggle("selected", item.dataset.nodeId === nodeId));
    el("empty-detail").classList.add("hidden");
    el("detail-content").classList.remove("hidden");

    el("detail-node-id").textContent = node.kind === "portfolio" ? "CURRENT BEST" :
      node.iteration !== null && node.iteration !== undefined ? `Round ${node.iteration} · ${node.id}` : node.id;
    const status = el("detail-status");
    status.textContent = node.status_label;
    status.className = `status-chip ${node.status_key}`;
    el("detail-title").textContent = cleanText(node.label, node.id);
    const stage = el("detail-stage");
    if (node.stage) { stage.textContent = node.stage; stage.classList.remove("hidden"); }
    else stage.classList.add("hidden");

    el("detail-score-label").textContent = node.kind === "portfolio" ? "Combination score" : "Standalone score";
    el("detail-score").textContent = formatScore(node.score);
    const scoreDelta = el("detail-score-delta");
    if (node.kind === "portfolio" && typeof node.score_delta === "number") {
      scoreDelta.textContent = `${formatDelta(node.score_delta)} vs best model`;
      scoreDelta.className = deltaClass(node.score_delta);
    } else if (typeof node.score_delta === "number") {
      scoreDelta.textContent = `${formatDelta(node.score_delta)} vs parent`;
      scoreDelta.className = deltaClass(node.score_delta);
    } else {
      scoreDelta.textContent = node.score === null ? "No trusted score yet" : "Starting or saved score";
      scoreDelta.className = "neutral";
    }

    const combinationDelta = el("detail-combination-delta");
    const combinationNote = el("detail-combination-note");
    if (typeof node.combination_delta === "number") {
      combinationDelta.textContent = formatDelta(node.combination_delta);
      combinationDelta.className = deltaClass(node.combination_delta);
      combinationNote.textContent = node.candidate_entered_combination ? "Candidate entered comparison" : "Not selected";
    } else if (node.is_portfolio_member) {
      combinationDelta.textContent = typeof node.portfolio_weight === "number" ? `${Math.round(node.portfolio_weight * 100)}%` : "Member";
      combinationDelta.className = "positive";
      combinationNote.textContent = "Used in current best result";
    } else {
      combinationDelta.textContent = "—";
      combinationDelta.className = "neutral";
      combinationNote.textContent = node.kind === "portfolio" ? "Selected result" : "Did not enter best comparison";
    }
    renderConfidence(node.score_ci95);
    section("hypothesis", node.hypothesis);
    section("why", node.why);
    section("result", node.result);
    section("lesson", node.lesson);

    const tags = [];
    if (node.execution_mode) tags.push(cleanText(node.execution_mode).replaceAll("_", " "));
    if (node.operator_id) tags.push(cleanText(node.operator_id).replaceAll("_", " "));
    (node.patch_scope || []).forEach(item => tags.push(`${item} change`));
    if (node.times_selected_as_parent) tags.push(`chosen as parent ${node.times_selected_as_parent}×`);
    el("detail-tags").replaceChildren(...tags.map(tag => {
      const span = document.createElement("span");
      span.className = "detail-tag";
      span.textContent = tag;
      return span;
    }));

    const artifactEntries = Object.entries(node.artifacts || {});
    const artifactSection = el("artifact-section");
    if (artifactEntries.length) {
      artifactSection.classList.remove("hidden");
      el("artifact-links").replaceChildren(...artifactEntries.map(([label, path]) => {
        const link = document.createElement("a");
        link.className = "artifact-link";
        const encoded = path.split("/").map(encodeURIComponent).join("/");
        link.href = STATIC_BASE ? `${STATIC_BASE}/artifact/${encoded}` : `/artifact/${encoded}`;
        link.target = "_blank";
        link.rel = "noopener";
        link.textContent = label;
        link.title = path;
        return link;
      }));
    } else {
      artifactSection.classList.add("hidden");
    }
  }

  function nodeStatusForTimeline(item) {
    const decision = cleanText(item.decision).toUpperCase();
    if (decision === "ACCEPT" || decision === "WARMSTART_VERIFIED") return "accepted";
    if (decision === "UNCERTAIN") return "uncertain";
    if (decision === "ROLLBACK") return "rollback";
    if (decision === "IN_PROGRESS") return cleanText(item.status).toUpperCase() === "PAUSED" ? "paused" : "running";
    if (decision.includes("PLANNING") || decision === "ORCHESTRATOR_ERROR") return "planning_failed";
    if (decision === "REJECT" || cleanText(item.status).toUpperCase() === "FAILED") return "failed";
    return "recorded";
  }

  function renderTimeline(data) {
    const list = el("timeline-list");
    const rows = (data.timeline || []).sort((a, b) => (a.iteration ?? 0) - (b.iteration ?? 0));
    list.replaceChildren(...rows.map(item => {
      const status = nodeStatusForTimeline(item);
      const row = document.createElement("div");
      row.className = "timeline-row";
      const marker = document.createElement("span");
      marker.className = `timeline-marker ${status}`;
      marker.textContent = item.iteration ?? "·";
      const copy = document.createElement("div");
      copy.className = "timeline-copy";
      const title = document.createElement("strong");
      title.textContent = cleanText(item.label, "Recorded event");
      const meta = document.createElement("span");
      const parts = [cleanText(item.decision || item.status).replaceAll("_", " ")];
      if (typeof item.tokens === "number") parts.push(`${formatCompact(item.tokens)} tokens`);
      if (typeof item.wall_s === "number") parts.push(`${Math.round(item.wall_s)}s`);
      meta.textContent = parts.filter(Boolean).join(" · ");
      copy.append(title, meta);
      const score = document.createElement("span");
      score.className = "timeline-score";
      score.textContent = formatScore(item.score);
      row.append(marker, copy, score);
      return row;
    }));
  }

  function renderCombination(data) {
    const container = el("combination-content");
    const selection = data.selection || {};
    if (!selection.members?.length) {
      container.innerHTML = '<div class="mini-empty">No model combination has been selected yet.</div>';
      return;
    }
    container.replaceChildren();
    const score = document.createElement("div");
    score.className = "combination-score";
    const scoreLabel = document.createElement("span");
    scoreLabel.textContent = "Validation score";
    const scoreValue = document.createElement("strong");
    scoreValue.textContent = formatScore(selection.score);
    score.append(scoreLabel, scoreValue);
    container.append(score);

    const members = document.createElement("div");
    members.className = "combination-members";
    selection.members.forEach(member => {
      const row = document.createElement("div");
      row.className = "combination-member";
      const swatch = document.createElement("i");
      swatch.className = "member-swatch";
      const copy = document.createElement("div");
      copy.className = "member-copy";
      const title = document.createElement("strong");
      const graphNode = state.nodeMap.get(member.node_id);
      title.textContent = cleanText(member.mechanism || graphNode?.label, member.node_id);
      const nodeId = document.createElement("span");
      nodeId.textContent = `${member.node_id} · standalone ${formatScore(member.standalone_primary ?? graphNode?.score)}`;
      copy.append(title, nodeId);
      const weight = document.createElement("span");
      weight.className = "member-weight";
      weight.textContent = typeof member.weight === "number" ? `${Math.round(member.weight * 100)}%` : "member";
      row.append(swatch, copy, weight);
      row.addEventListener("click", () => { if (graphNode) selectNode(member.node_id); });
      members.append(row);
    });
    container.append(members);

    if (selection.context_router?.routes?.length) {
      const note = document.createElement("div");
      note.className = "router-note";
      note.textContent = `The saved router uses ${selection.context_router.routes.length} fixed tab-specific weight sets. No test label is used to choose a route.`;
      container.append(note);
    }
  }

  function renderHeader(data) {
    const run = data.run || {};
    const stats = data.stats || {};
    el("run-name").textContent = cleanText(run.id, "Unnamed experiment");
    const subtitle = [run.model, run.role, run.current_iteration ? `round ${run.current_iteration}` : null]
      .filter(Boolean).join(" · ");
    el("run-subtitle").textContent = subtitle || "Recorded experiment";
    const pill = el("live-pill");
    pill.className = `live-pill ${run.status || "waiting"}`;
    el("live-label").textContent = cleanText(run.status_label, "Waiting");
    el("stat-nodes").textContent = formatCompact(stats.graph_nodes);
    el("stat-measured").textContent = formatCompact(stats.measured_models);
    el("stat-standalone").textContent = formatScore(stats.best_standalone);
    el("stat-combination").textContent = formatScore(stats.best_combination);
    document.title = `${cleanText(run.id, "Experiment")} · Experiment Graph`;
  }

  function renderWarnings(data) {
    const banner = el("warning-banner");
    if (data.warnings?.length) {
      banner.textContent = `Some files changed while the dashboard was reading them: ${data.warnings.join(" · ")}`;
      banner.classList.remove("hidden");
    } else banner.classList.add("hidden");
  }

  function showToast(message, duration = 1800) {
    const toast = el("toast");
    toast.textContent = message;
    toast.classList.remove("hidden");
    window.setTimeout(() => toast.classList.add("hidden"), duration);
  }

  function render(data) {
    const previousNodeCount = state.data?.nodes?.length || 0;
    state.data = data;
    renderHeader(data);
    renderGraph(data, !state.firstLoad);
    renderTimeline(data);
    renderCombination(data);
    renderWarnings(data);
    if (!state.selectedId) {
      const defaultNode = (data.nodes || []).find(node => node.kind === "active") ||
        (data.nodes || []).find(node => node.kind === "portfolio") ||
        (data.nodes || []).find(node => node.is_best_standalone);
      if (defaultNode) state.selectedId = defaultNode.id;
    }
    if (state.selectedId && state.nodeMap.has(state.selectedId)) selectNode(state.selectedId);
    if (!state.firstLoad && data.nodes.length > previousNodeCount) showToast("A new experiment node was added");
    state.firstLoad = false;
  }

  async function refresh(manual = false) {
    if (state.inFlight) return;
    state.inFlight = true;
    const button = el("refresh-button");
    if (manual) button.disabled = true;
    try {
      const response = STATIC_BASE
        ? await fetch(`${STATIC_BASE}/state.json`, { cache: "no-store" })
        : await fetch(`/api/state?t=${Date.now()}`, { cache: "no-store" });
      const data = await response.json();
      if (!response.ok || data.error) throw new Error(data.error || `HTTP ${response.status}`);
      render(data);
      if (manual) showToast("Graph refreshed");
    } catch (error) {
      const pill = el("live-pill");
      pill.className = "live-pill paused";
      el("live-label").textContent = "Connection lost";
      if (manual || state.firstLoad) showToast(`Could not refresh: ${error.message}`, 3500);
    } finally {
      state.inFlight = false;
      button.disabled = false;
    }
  }

  function zoomAt(factor, cx, cy) {
    const oldScale = state.transform.scale;
    const nextScale = Math.max(0.2, Math.min(2.3, oldScale * factor));
    const worldX = (cx - state.transform.x) / oldScale;
    const worldY = (cy - state.transform.y) / oldScale;
    state.transform.x = cx - worldX * nextScale;
    state.transform.y = cy - worldY * nextScale;
    state.transform.scale = nextScale;
    applyTransform();
  }

  viewport.addEventListener("wheel", event => {
    event.preventDefault();
    const rect = viewport.getBoundingClientRect();
    zoomAt(event.deltaY < 0 ? 1.1 : 0.9, event.clientX - rect.left, event.clientY - rect.top);
  }, { passive: false });

  viewport.addEventListener("pointerdown", event => {
    if (event.target.closest?.(".graph-node")) return;
    state.dragging = true;
    state.dragOrigin = { x: event.clientX, y: event.clientY, tx: state.transform.x, ty: state.transform.y };
    viewport.classList.add("dragging");
    viewport.setPointerCapture(event.pointerId);
  });
  viewport.addEventListener("pointermove", event => {
    if (!state.dragging || !state.dragOrigin) return;
    state.transform.x = state.dragOrigin.tx + event.clientX - state.dragOrigin.x;
    state.transform.y = state.dragOrigin.ty + event.clientY - state.dragOrigin.y;
    applyTransform();
  });
  viewport.addEventListener("pointerup", event => {
    state.dragging = false;
    state.dragOrigin = null;
    viewport.classList.remove("dragging");
    try { viewport.releasePointerCapture(event.pointerId); } catch (_) { /* no-op */ }
  });

  el("zoom-in").addEventListener("click", () => zoomAt(1.15, viewport.clientWidth / 2, viewport.clientHeight / 2));
  el("zoom-out").addEventListener("click", () => zoomAt(0.87, viewport.clientWidth / 2, viewport.clientHeight / 2));
  el("fit-graph").addEventListener("click", () => {
    if (!state.data) return;
    fitGraph(layoutGraph(state.data), true);
  });
  el("refresh-button").addEventListener("click", () => refresh(true));
  el("edge-filter").addEventListener("click", event => {
    const button = event.target.closest("[data-edge]");
    if (!button || !state.data) return;
    const type = button.dataset.edge;
    if (state.edgeFilters.has(type)) state.edgeFilters.delete(type);
    else state.edgeFilters.add(type);
    button.classList.toggle("active", state.edgeFilters.has(type));
    renderEdges(state.data);
  });

  window.addEventListener("resize", () => {
    if (!state.data) return;
    window.clearTimeout(window.__graphResizeTimer);
    window.__graphResizeTimer = window.setTimeout(() => fitGraph(layoutGraph(state.data), true), 140);
  });

  refresh();
  if (!STATIC_BASE) {
    state.refreshTimer = window.setInterval(() => refresh(false), REFRESH_MS);
  }
})();
