"""Persistent execution frontier with real parent re-selection.

Unlike linear incumbent accounting, every executed candidate remains a node.  The policy
creates several baseline-rooted drafts, then may expand any successful ACCEPT or UNCERTAIN
node.  Failed/rollback nodes remain in the graph for audit and ensemble diversity analysis,
but are not executable parents.
"""
from __future__ import annotations

import json
import math
import os


ELIGIBLE_DECISIONS = {'BASELINE', 'ACCEPT', 'UNCERTAIN'}


class Frontier:
    def __init__(self, run_dir, draft_count=2, exploration=0.00025):
        self.run_dir = os.path.abspath(run_dir)
        self.path = os.path.join(self.run_dir, 'frontier.json')
        self.draft_count = int(draft_count)
        self.exploration = float(exploration)
        self.nodes = {}
        self.selections = []

    def _relative(self, path):
        return os.path.relpath(os.path.abspath(path), self.run_dir) if path else None

    def resolve(self, path):
        return os.path.join(self.run_dir, path) if path else None

    def initialize(self, pipeline_path, prediction_paths, metrics_path, metrics,
                   pipeline_sha256=None, operator_stack=None):
        if self.nodes:
            raise ValueError('frontier already initialized')
        self.nodes['n000'] = {
            'node_id': 'n000', 'parent_node_id': None, 'children': [],
            'decision': 'BASELINE', 'status': 'COMPLETE',
            'pipeline_path': self._relative(pipeline_path),
            'prediction_paths': [self._relative(path) for path in prediction_paths],
            'metrics_path': self._relative(metrics_path),
            'selection_primary': float(metrics['selection_primary']),
            'pipeline_sha256': pipeline_sha256, 'operator_stack': operator_stack,
            'execution_mode': 'baseline', 'mechanism': 'official FM baseline',
            'times_selected_as_parent': 0,
        }
        self.save()

    def add_node(self, node):
        node = dict(node)
        node_id = node['node_id']
        if node_id in self.nodes:
            raise ValueError(f'duplicate frontier node: {node_id}')
        parent_id = node.get('parent_node_id')
        if parent_id not in self.nodes:
            raise ValueError(f'unknown frontier parent: {parent_id}')
        node.setdefault('children', [])
        node.setdefault('times_selected_as_parent', 0)
        for key in ('pipeline_path', 'metrics_path'):
            if node.get(key) and os.path.isabs(node[key]):
                node[key] = self._relative(node[key])
        node['prediction_paths'] = [
            self._relative(path) if os.path.isabs(path) else path
            for path in node.get('prediction_paths', [])]
        self.nodes[node_id] = node
        self.nodes[parent_id]['children'].append(node_id)
        self.save()
        return node

    def successful_nodes(self):
        return [node for node in self.nodes.values()
                if node.get('decision') in ELIGIBLE_DECISIONS
                and node.get('pipeline_path') and node.get('metrics_path')
                and node.get('prediction_paths')]

    def update_node(self, node_id, **updates):
        """Update controller-owned outcome metadata after portfolio evaluation."""
        if node_id not in self.nodes:
            raise ValueError(f'unknown frontier node: {node_id}')
        immutable = {'node_id', 'parent_node_id', 'pipeline_path', 'prediction_paths'}
        overlap = immutable & set(updates)
        if overlap:
            raise ValueError(f'cannot update immutable frontier fields: {sorted(overlap)}')
        self.nodes[node_id].update(updates)
        self.save()
        return dict(self.nodes[node_id])

    def select_parent(self, iteration, portfolio_selection=None):
        """Select a pipeline parent, explicitly rotating through portfolio members.

        A model can be a valuable ensemble specialist even when its standalone score was a
        rollback.  Once trusted portfolio selection assigns it non-zero weight it becomes a legal
        refinement parent.  Prefer the least-expanded current members before falling back to the
        ordinary frontier score.
        """
        nonbaseline = len(self.nodes) - 1
        reason = 'frontier_score'
        selected = None
        portfolio_weights = {}
        if portfolio_selection:
            portfolio_weights = {
                item['node_id']: float(item.get('weight', 0.0))
                for item in portfolio_selection.get('members', [])
                if item.get('node_id') in self.nodes
            }
            members = [self.nodes[node_id] for node_id in portfolio_weights
                       if self.nodes[node_id].get('status') == 'COMPLETE'
                       and self.nodes[node_id].get('pipeline_path')
                       and self.nodes[node_id].get('metrics_path')
                       and self.nodes[node_id].get('prediction_paths')]
            if members:
                minimum_visits = min(int(node.get('times_selected_as_parent', 0))
                                     for node in members)
                least_expanded = [node for node in members
                                  if int(node.get('times_selected_as_parent', 0))
                                  == minimum_visits]
                selected = max(
                    least_expanded,
                    key=lambda node: (
                        portfolio_weights[node['node_id']],
                        -float(node['selection_primary']),
                        node['node_id']))
                reason = 'portfolio_member_rotation'
        if selected is None and nonbaseline < self.draft_count:
            selected = self.nodes['n000']
            reason = 'independent_baseline_draft'
        elif selected is None:
            candidates = [node for node in self.successful_nodes()
                          if node['node_id'] != 'n000']
            if not candidates:
                candidates = [self.nodes['n000']]
                reason = 'baseline_fallback'
            total_selections = max(1, len(self.selections))

            def utility(node):
                visits = int(node.get('times_selected_as_parent', 0))
                bonus = self.exploration * math.sqrt(
                    math.log(total_selections + 2) / (visits + 1))
                return (float(node['selection_primary']) + bonus,
                        node.get('decision') == 'ACCEPT',
                        -visits, node['node_id'])

            selected = max(candidates, key=utility)
        selected['times_selected_as_parent'] = int(
            selected.get('times_selected_as_parent', 0)) + 1
        record = {
            'iteration': int(iteration), 'node_id': selected['node_id'],
            'reason': reason, 'selection_primary': selected['selection_primary'],
            'decision': selected['decision'],
            'times_selected_as_parent': selected['times_selected_as_parent'],
            'portfolio_weight': portfolio_weights.get(selected['node_id']),
        }
        self.selections.append(record)
        self.save()
        return dict(selected), record

    def best_validation(self):
        return max(self.successful_nodes(), key=lambda node: node['selection_primary'])

    def best_stable(self):
        candidates = [node for node in self.successful_nodes()
                      if node['decision'] in {'BASELINE', 'ACCEPT'}]
        return max(candidates, key=lambda node: node['selection_primary'])

    def save(self):
        payload = {
            'schema_version': 'frontier-1.0',
            'policy': {
                'draft_count': self.draft_count,
                'exploration': self.exploration,
                'eligible_decisions': sorted(ELIGIBLE_DECISIONS),
                'uncertain_nodes_expandable': True,
                'rollback_nodes_preserved_but_not_expandable': True,
                'selected_portfolio_members_expandable': True,
                'portfolio_parent_policy': 'least-expanded member, then weight and opportunity',
            },
            'nodes': [self.nodes[key] for key in sorted(self.nodes)],
            'parent_selections': self.selections,
        }
        tmp = self.path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, self.path)
