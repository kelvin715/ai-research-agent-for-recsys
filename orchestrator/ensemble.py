"""Validation-only heterogeneous ensemble selection over retained frontier nodes."""
from __future__ import annotations

import copy
import os
from itertools import combinations, product

import numpy as np

import evaluator


MIN_CONSISTENT_SEED_GAIN = 1e-4
MAX_MATCHED_SEED_REGRESSION = 5e-5
WEIGHT_GRID_UNITS = 10
CONTEXT_MIN_ROWS = 1000
CONTEXT_MIN_MEAN_SEED_GAIN = 1.5e-4


def within_user_rank_average(predictions, user_ids):
    user_ids = np.asarray(user_ids)
    if not predictions:
        raise ValueError('at least one prediction is required')
    order = np.argsort(user_ids, kind='stable')
    sorted_users = user_ids[order]
    boundaries = np.r_[0, np.flatnonzero(
        sorted_users[1:] != sorted_users[:-1]) + 1, len(order)]
    output = np.zeros(len(user_ids), dtype=np.float64)
    for prediction in predictions:
        prediction = np.asarray(prediction)
        if prediction.shape != user_ids.shape:
            raise ValueError('prediction and user ids have different shapes')
        ranks = np.zeros(len(user_ids), dtype=np.float64)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            indices = order[start:end]
            local_order = np.argsort(prediction[indices], kind='stable')
            values = prediction[indices][local_order]
            tie_starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
            tie_ends = np.r_[tie_starts[1:], len(values)]
            denominator = max(len(values) - 1, 1)
            for tie_start, tie_end in zip(tie_starts, tie_ends):
                rank = ((tie_start + tie_end - 1) / 2.0) / denominator
                ranks[indices[local_order[tie_start:tie_end]]] = rank
        output += ranks
    return (output / len(predictions)).astype(np.float32)


def _node_prediction(node, run_dir, users):
    paths = [os.path.join(run_dir, path) for path in node['prediction_paths']]
    return within_user_rank_average(
        [np.load(path, allow_pickle=False) for path in paths], users)


def _node_seed_predictions(node, run_dir):
    return [np.load(os.path.join(run_dir, path), allow_pickle=False)
            for path in node['prediction_paths']]


def _weighted_prediction(predictions, member_ids, base_weights,
                         context_router=None, context_values=None):
    """Combine already rank-normalized member predictions, optionally by context."""
    weights = np.asarray(base_weights, dtype=np.float64)
    if (weights.shape != (len(predictions),) or not np.isfinite(weights).all()
            or np.any(weights < 0) or float(weights.sum()) <= 0):
        raise ValueError('invalid member weights')
    weights /= weights.sum()
    output = np.sum([
        weight * np.asarray(prediction, dtype=np.float64)
        for weight, prediction in zip(weights, predictions)
    ], axis=0)
    if not context_router:
        return output.astype(np.float32)
    if context_values is None:
        raise ValueError('context values are required by contextual router')
    if context_router.get('feature') != 'tab':
        raise ValueError('only the legal inference-time tab router is supported')
    context_values = np.asarray(context_values)
    if context_values.shape != output.shape:
        raise ValueError('context and prediction shapes differ')
    for route in context_router.get('routes', []):
        route_map = route.get('member_weights') or {}
        route_weights = np.asarray(
            [float(route_map.get(node_id, 0.0)) for node_id in member_ids],
            dtype=np.float64)
        if (not np.isfinite(route_weights).all() or np.any(route_weights < 0)
                or float(route_weights.sum()) <= 0):
            raise ValueError('invalid contextual route weights')
        route_weights /= route_weights.sum()
        mask = context_values == route['value']
        if mask.any():
            output[mask] = np.sum([
                weight * np.asarray(prediction, dtype=np.float64)[mask]
                for weight, prediction in zip(route_weights, predictions)
            ], axis=0)
    return output.astype(np.float32)


def combine_member_predictions(predictions, user_ids, member_ids, weights,
                               context_router=None, context_values=None):
    """Exact validation/submission combiner shared by selection and test inference."""
    if len(predictions) != len(member_ids):
        raise ValueError('predictions and member ids have different lengths')
    if not predictions:
        raise ValueError('at least one member prediction is required')
    if len(predictions) == 1:
        ranked = [np.asarray(predictions[0])]
    else:
        ranked = [within_user_rank_average([prediction], user_ids)
                  for prediction in predictions]
    return _weighted_prediction(
        ranked, member_ids, weights, context_router, context_values)


def promotion_decision(current_primary, candidate_primary, paired,
                       per_seed_deltas, min_consistent_seed_gain=None):
    """Use CI for stable evidence, with a matched-seed fallback for small blends.

    The fixed validation set makes paired user bootstrap uncertainty much larger than
    seed-to-seed training noise. A small heterogeneous blend can therefore be robust across
    matched seeds yet fail the CI gate. Such a blend is retained when at least two of three
    fixed seeds improve, the mean gain is material, and the worst regression is tightly bounded.
    """
    threshold = (MIN_CONSISTENT_SEED_GAIN if min_consistent_seed_gain is None
                 else float(min_consistent_seed_gain))
    ci = paired.get('paired_ci95') or [float('nan'), float('nan')]
    positive_ci = bool(
        paired.get('excludes_zero') and len(ci) == 2 and np.isfinite(ci).all()
        and float(ci[0]) > 0)
    deltas = np.asarray(per_seed_deltas, dtype=np.float64)
    seed_robust = bool(
        len(deltas) >= 3 and np.isfinite(deltas).all()
        and int(np.sum(deltas > 0)) >= len(deltas) - 1
        and float(deltas.min()) >= -MAX_MATCHED_SEED_REGRESSION
        and float(deltas.mean()) >= threshold)
    improves = bool(float(candidate_primary) > float(current_primary))
    if improves and positive_ci:
        return True, 'positive_paired_ci95'
    if improves and seed_robust:
        return True, 'bounded_matched_seed_robustness'
    return False, 'insufficient_promotion_evidence'


def _candidate_nodes(frontier, max_pool, required_node_ids=None):
    required_node_ids = set(required_node_ids or ())
    candidates = [node for node in frontier.nodes.values()
                  if node.get('status') == 'COMPLETE'
                  and node.get('prediction_paths') and node.get('pipeline_path')
                  and node.get('selection_primary') is not None]
    candidates.sort(key=lambda node: node['selection_primary'], reverse=True)
    distinct, seen = [], set()
    for node in candidates:
        identity = node.get('pipeline_sha256') or node['pipeline_path']
        if identity in seen:
            continue
        seen.add(identity)
        distinct.append(node)
        if (len(distinct) >= max_pool
                and required_node_ids <= {item['node_id'] for item in distinct}):
            break
    protected = [node for node in candidates if node['node_id'] in required_node_ids]
    by_id = {node['node_id']: node for node in protected}
    for node in distinct:
        by_id.setdefault(node['node_id'], node)
    result = sorted(by_id.values(), key=lambda node: node['selection_primary'], reverse=True)
    if len(result) > max_pool:
        protected_ids = set(required_node_ids)
        keep = [node for node in result if node['node_id'] in protected_ids]
        keep.extend(node for node in result if node['node_id'] not in protected_ids
                    and len(keep) < max_pool)
        result = sorted(keep, key=lambda node: node['selection_primary'], reverse=True)
    return result


def selection_prediction(selection, frontier, parsed_dir, seed_index=None):
    """Reconstruct the exact validation prediction represented by a selection record."""
    users = np.load(os.path.join(parsed_dir, 'valid.user_id.npy'), allow_pickle=False)
    members = selection.get('members') or []
    if not members:
        raise ValueError('selection has no members')
    weights = np.asarray([float(item.get('weight', 0.0)) for item in members],
                         dtype=np.float64)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
        raise ValueError('selection weights are invalid')
    weights /= weights.sum()
    predictions = []
    member_ids = []
    for item in members:
        node = frontier.nodes.get(item['node_id'])
        if node is None:
            raise ValueError(f"selection references unknown node {item['node_id']}")
        if seed_index is None:
            prediction = _node_prediction(node, frontier.run_dir, users)
        else:
            seeds = _node_seed_predictions(node, frontier.run_dir)
            if seed_index >= len(seeds):
                raise ValueError('selection members do not share the requested seed')
            prediction = seeds[seed_index]
        predictions.append(np.asarray(prediction, dtype=np.float64))
        member_ids.append(item['node_id'])
    router = selection.get('context_router')
    context_values = None
    if router:
        context_values = np.load(
            os.path.join(parsed_dir, f"valid.{router['feature']}.npy"),
            allow_pickle=False)
    return combine_member_predictions(
        predictions, users, member_ids, weights, router, context_values)


def compare_selections(incumbent, challenger, frontier, parsed_dir, n_boot=1000,
                       candidate_node_id=None):
    """Measure a proposed portfolio change against the actually deployed portfolio."""
    incumbent_prediction = selection_prediction(
        incumbent, frontier, parsed_dir)
    challenger_prediction = selection_prediction(
        challenger, frontier, parsed_dir)
    incumbent_primary = evaluator.score(
        incumbent_prediction, 'valid', n_boot=0)['primary']
    challenger_primary = evaluator.score(
        challenger_prediction, 'valid', n_boot=0)['primary']
    paired = evaluator.compare(
        challenger_prediction, incumbent_prediction, 'valid', n=n_boot)
    incumbent_ids = {item['node_id'] for item in incumbent.get('members', [])}
    challenger_ids = {item['node_id'] for item in challenger.get('members', [])}
    candidate_entered = bool(
        candidate_node_id and candidate_node_id in challenger_ids
        and candidate_node_id not in incumbent_ids)

    seed_counts = []
    for selection in (incumbent, challenger):
        counts = [len(_node_seed_predictions(
            frontier.nodes[item['node_id']], frontier.run_dir))
            for item in selection.get('members', [])]
        seed_counts.append(min(counts) if counts and len(set(counts)) == 1 else 0)
    matched_seed_count = min(seed_counts)
    per_seed_deltas = []
    if matched_seed_count >= 3:
        for seed_index in range(matched_seed_count):
            inc_seed = selection_prediction(
                incumbent, frontier, parsed_dir, seed_index=seed_index)
            cha_seed = selection_prediction(
                challenger, frontier, parsed_dir, seed_index=seed_index)
            per_seed_deltas.append(float(
                evaluator.score(cha_seed, 'valid', n_boot=0)['primary']
                - evaluator.score(inc_seed, 'valid', n_boot=0)['primary']))
    promotes, reason = promotion_decision(
        incumbent_primary, challenger_primary, paired, per_seed_deltas)
    return {
        'incumbent_primary': float(incumbent_primary),
        'challenger_primary': float(challenger_primary),
        'delta_primary': float(challenger_primary - incumbent_primary),
        'paired_ci95': paired.get('paired_ci95'),
        'paired_excludes_zero': paired.get('excludes_zero'),
        'matched_seed_deltas': per_seed_deltas,
        'mean_matched_seed_delta': (
            float(np.mean(per_seed_deltas)) if per_seed_deltas else None),
        'promoted': bool(promotes),
        'promotion_reason': reason,
        'candidate_node_id': candidate_node_id,
        'candidate_entered': candidate_entered,
        'incumbent_member_ids': sorted(incumbent_ids),
        'challenger_member_ids': sorted(challenger_ids),
    }


def _extreme_directional_slices(report, limit=3):
    rows = []
    for item in report.get('tab', []):
        rows.append({'slice': f"tab={item['tab']}", **item})
    for dimension in ('train_user_history_quartile',
                      'train_item_popularity_quartile'):
        for index, item in enumerate(
                (report.get(dimension) or {}).get('slices', []), start=1):
            rows.append({'slice': f'{dimension}:q{index}', **item})
    rows.sort(key=lambda item: item['delta_primary'], reverse=True)
    return {
        'member_advantages': rows[:limit],
        'member_disadvantages': list(reversed(rows[-limit:])),
    }


def portfolio_diagnostics(frontier, selection, parsed_dir):
    """Compact trusted diagnostics for portfolio-aware proposal generation.

    Only aggregate validation measurements leave this function.  Row-level labels and residuals
    stay in the trusted evaluator, while the planner receives member contribution, redundancy and
    fixed-slice specialization signals.
    """
    members = selection.get('members') or []
    if not members:
        return {'status': 'NO_PORTFOLIO'}
    users = np.load(os.path.join(parsed_dir, 'valid.user_id.npy'), allow_pickle=False)
    portfolio_prediction = selection_prediction(selection, frontier, parsed_dir)
    portfolio_primary = evaluator.score(
        portfolio_prediction, 'valid', n_boot=0)['primary']
    ranked = {}
    for item in members:
        node = frontier.nodes[item['node_id']]
        ranked[item['node_id']] = within_user_rank_average(
            [_node_prediction(node, frontier.run_dir, users)], users)

    member_rows = []
    for item in members:
        node_id = item['node_id']
        others = [other for other in members if other['node_id'] != node_id]
        removal_primary = None
        if others:
            removal = {'members': copy.deepcopy(others)}
            if selection.get('context_router'):
                router = copy.deepcopy(selection['context_router'])
                kept_routes = []
                for route in router.get('routes', []):
                    route['member_weights'].pop(node_id, None)
                    if sum(float(value) for value in
                           route['member_weights'].values()) > 0:
                        kept_routes.append(route)
                router['routes'] = kept_routes
                if kept_routes:
                    removal['context_router'] = router
            removal_prediction = selection_prediction(
                removal, frontier, parsed_dir)
            removal_primary = evaluator.score(
                removal_prediction, 'valid', n_boot=0)['primary']
        correlation = float(np.corrcoef(
            ranked[node_id], portfolio_prediction)[0, 1])
        directional = evaluator.directional_compare(
            ranked[node_id], portfolio_prediction, 'valid', parsed_dir)
        member_rows.append({
            'node_id': node_id,
            'mechanism': item.get('mechanism'),
            'operator_stack': item.get('operator_stack'),
            'weight': float(item.get('weight', 0.0)),
            'standalone_primary': float(item['standalone_primary']),
            'leave_one_out_primary': (
                float(removal_primary) if removal_primary is not None else None),
            'marginal_contribution': (
                float(portfolio_primary - removal_primary)
                if removal_primary is not None else None),
            'rank_correlation_with_portfolio': correlation,
            'slice_specialization': _extreme_directional_slices(directional),
        })

    pairwise = []
    for left, right in combinations(members, 2):
        correlation = float(np.corrcoef(
            ranked[left['node_id']], ranked[right['node_id']])[0, 1])
        pairwise.append({
            'left': left['node_id'], 'right': right['node_id'],
            'within_user_rank_correlation': correlation,
        })
    return {
        'status': 'READY',
        'selection_primary': float(portfolio_primary),
        'objective': ('improve marginal portfolio primary; a candidate may succeed through '
                      'standalone quality or robust complementary ordering'),
        'members': member_rows,
        'pairwise_diversity': pairwise,
        'labels_exposed_to_candidate': False,
        'diagnostic_policy': 'aggregate fixed-slice feedback only',
    }


def _contextual_weight_refinement(
        selected, selected_weights, ranked_predictions, ranked_seed_predictions,
        users, parsed_dir, current_prediction, current_metrics, matched_seed_count,
        n_boot):
    """Fit a bounded tab-aware mixture and require unusually strong seed robustness.

    The router is intentionally a trusted post-model component: ``tab`` is available at inference,
    the candidate pipelines never see validation labels, and only coarse 0.1-grid weights are
    considered.  We optimize each sufficiently large tab independently, then test the combined
    policy once.  Because this adds several validation-selected degrees of freedom, its matched-seed
    gate is stricter than ordinary blending: all three seeds must improve and their mean gain must
    reach ``CONTEXT_MIN_MEAN_SEED_GAIN`` unless the paired user CI is already positive.
    """
    status = {
        'status': 'NOT_SELECTED',
        'feature': 'tab',
        'min_rows': CONTEXT_MIN_ROWS,
        'grid_units': WEIGHT_GRID_UNITS,
        'regularization': ('coarse weights; route only tabs with >= min_rows; all matched seeds '
                           f'positive with mean >= {CONTEXT_MIN_MEAN_SEED_GAIN:g}, or positive CI'),
        'tab_trials': [],
    }
    if len(selected) < 2:
        status['status'] = 'NOT_APPLICABLE'
        return None, current_prediction, current_metrics, status
    tabs = np.load(os.path.join(parsed_dir, 'valid.tab.npy'), allow_pickle=False)
    member_ids = [node['node_id'] for node in selected]
    member_predictions = [ranked_predictions[node_id] for node_id in member_ids]
    grid = [
        np.asarray(integer_weights, dtype=np.float64) / WEIGHT_GRID_UNITS
        for integer_weights in product(
            range(WEIGHT_GRID_UNITS + 1), repeat=len(selected))
        if sum(integer_weights) == WEIGHT_GRID_UNITS
    ]
    routes = []
    values, counts = np.unique(tabs, return_counts=True)
    for value, count in zip(values, counts):
        if int(count) < CONTEXT_MIN_ROWS:
            continue
        mask = tabs == value
        best_primary = float(current_metrics['primary'])
        best_weights = np.asarray(selected_weights, dtype=np.float64)
        for weights in grid:
            candidate = np.asarray(current_prediction).copy()
            candidate[mask] = np.sum([
                weight * np.asarray(prediction, dtype=np.float64)[mask]
                for weight, prediction in zip(weights, member_predictions)
            ], axis=0).astype(np.float32)
            primary = evaluator.score(candidate, 'valid', n_boot=0)['primary']
            if primary > best_primary:
                best_primary = float(primary)
                best_weights = weights
        trial = {
            'value': int(value), 'rows': int(count),
            'best_weights': [float(weight) for weight in best_weights],
            'best_primary_when_routed_alone': best_primary,
            'delta_when_routed_alone': float(best_primary - current_metrics['primary']),
            'evaluated_weight_vectors': len(grid),
        }
        status['tab_trials'].append(trial)
        if best_primary > current_metrics['primary']:
            routes.append({
                'value': int(value), 'rows': int(count),
                'member_weights': {
                    node_id: float(weight)
                    for node_id, weight in zip(member_ids, best_weights)},
            })
    if not routes:
        status['status'] = 'NO_POINT_IMPROVING_ROUTES'
        return None, current_prediction, current_metrics, status

    router = {
        'feature': 'tab',
        'fallback': 'global member weights',
        'min_rows': CONTEXT_MIN_ROWS,
        'weight_grid_step': 1.0 / WEIGHT_GRID_UNITS,
        'routes': routes,
        'test_label_free': True,
    }
    combined = _weighted_prediction(
        member_predictions, member_ids, selected_weights, router, tabs)
    combined_metrics = evaluator.score(combined, 'valid', n_boot=n_boot)
    paired = evaluator.compare(combined, current_prediction, 'valid', n=n_boot)
    per_seed_deltas = []
    if matched_seed_count >= 3:
        for seed_index in range(matched_seed_count):
            seed_members = [
                ranked_seed_predictions[node_id][seed_index]
                for node_id in member_ids]
            base_seed = _weighted_prediction(
                seed_members, member_ids, selected_weights)
            routed_seed = _weighted_prediction(
                seed_members, member_ids, selected_weights, router, tabs)
            per_seed_deltas.append(float(
                evaluator.score(routed_seed, 'valid', n_boot=0)['primary']
                - evaluator.score(base_seed, 'valid', n_boot=0)['primary']))
    ci = paired.get('paired_ci95') or [float('nan'), float('nan')]
    positive_ci = bool(
        paired.get('excludes_zero') and len(ci) == 2 and np.isfinite(ci).all()
        and float(ci[0]) > 0)
    deltas = np.asarray(per_seed_deltas, dtype=np.float64)
    strict_seed_robust = bool(
        len(deltas) >= 3 and np.isfinite(deltas).all()
        and bool(np.all(deltas > 0))
        and float(deltas.mean()) >= CONTEXT_MIN_MEAN_SEED_GAIN)
    improves = bool(combined_metrics['primary'] > current_metrics['primary'])
    promoted = bool(improves and (positive_ci or strict_seed_robust))
    status.update({
        'status': 'CONTEXT_ROUTER_SELECTED' if promoted else 'EVIDENCE_REJECTED',
        'combined_primary': float(combined_metrics['primary']),
        'delta_vs_global_weights': float(
            combined_metrics['primary'] - current_metrics['primary']),
        'paired_ci95': paired.get('paired_ci95'),
        'paired_excludes_zero': paired.get('excludes_zero'),
        'matched_seed_deltas': per_seed_deltas,
        'mean_matched_seed_delta': (
            float(deltas.mean()) if len(deltas) else None),
        'promotion_reason': (
            'positive_paired_ci95' if promoted and positive_ci
            else 'strict_all_seed_robustness' if promoted
            else 'insufficient_contextual_promotion_evidence'),
        'router': router,
    })
    if not promoted:
        return None, current_prediction, current_metrics, status
    return router, combined, combined_metrics, status


def select(frontier, parsed_dir, n_boot=1000, max_members=3, max_pool=8,
           incumbent_selection=None, candidate_node_id=None,
           contextual_refinement=False):
    """Select the best legal subset and bounded weights without greedy path dependence.

    Rollback nodes remain eligible as *members*: a weak standalone model can carry
    complementary ordering information.  All subsets up to ``max_members`` are first ranked
    by the exact validation point estimate.  The highest-scoring subset that beats the best
    single model and either has a positive paired CI95 lower bound or passes the bounded
    matched-seed robustness rule becomes the final checkpoint. Evidence
    is checked against the best single model, so a useful triple is not blocked by a weak
    intermediate pair.
    """
    users = np.load(os.path.join(parsed_dir, 'valid.user_id.npy'), allow_pickle=False)
    protected_ids = {
        item['node_id'] for item in (incumbent_selection or {}).get('members', [])}
    if candidate_node_id:
        protected_ids.add(candidate_node_id)
    pool = _candidate_nodes(frontier, max_pool, required_node_ids=protected_ids)
    if not pool:
        return {'status': 'NO_CANDIDATES', 'selected': False, 'members': [], 'trials': []}

    predictions = {
        node['node_id']: _node_prediction(node, frontier.run_dir, users)
        for node in pool
    }
    seed_predictions = {
        node['node_id']: _node_seed_predictions(node, frontier.run_dir)
        for node in pool
    }
    seed_counts = {len(items) for items in seed_predictions.values()}
    matched_seed_count = seed_counts.pop() if len(seed_counts) == 1 else 0
    ranked_predictions = {
        node_id: within_user_rank_average([prediction], users)
        for node_id, prediction in predictions.items()
    }
    ranked_seed_predictions = {
        node_id: [within_user_rank_average([prediction], users)
                  for prediction in items]
        for node_id, items in seed_predictions.items()
    }

    single_best = pool[0]
    single_best_id = single_best['node_id']
    single_prediction = predictions[single_best_id]
    single_metrics = evaluator.score(single_prediction, 'valid', n_boot=n_boot)
    single_seed_primaries = [
        evaluator.score(prediction, 'valid', n_boot=0)['primary']
        for prediction in seed_predictions[single_best_id]
    ]
    trials = []
    candidates = []
    for member_count in range(2, min(max_members, len(pool)) + 1):
        for members in combinations(pool, member_count):
            member_ids = [node['node_id'] for node in members]
            combined = np.mean(
                [ranked_predictions[node_id] for node_id in member_ids],
                axis=0, dtype=np.float64).astype(np.float32)
            primary = evaluator.score(combined, 'valid', n_boot=0)['primary']
            trial = {
                'members': member_ids,
                'primary': float(primary),
                'delta_vs_single_best': float(
                    primary - single_metrics['primary']),
                'evidence_evaluated': False,
                'promoted': False,
            }
            trials.append(trial)
            candidates.append({
                'members': list(members), 'member_ids': member_ids,
                'prediction': combined, 'primary': float(primary),
                'trial': trial,
            })

    candidates.sort(key=lambda item: item['primary'], reverse=True)
    selected = [single_best]
    current_prediction = single_prediction
    current_metrics = single_metrics
    for candidate in candidates:
        if candidate['primary'] <= single_metrics['primary']:
            break
        per_seed_deltas = []
        if matched_seed_count >= 3:
            for seed_index in range(matched_seed_count):
                combined_seed = np.mean([
                    ranked_seed_predictions[node_id][seed_index]
                    for node_id in candidate['member_ids']
                ], axis=0, dtype=np.float64).astype(np.float32)
                primary = evaluator.score(
                    combined_seed, 'valid', n_boot=0)['primary']
                per_seed_deltas.append(float(
                    primary - single_seed_primaries[seed_index]))
        paired = evaluator.compare(
            candidate['prediction'], single_prediction, 'valid', n=n_boot)
        promotes, promotion_reason = promotion_decision(
            single_metrics['primary'], candidate['primary'],
            paired, per_seed_deltas)
        candidate['trial'].update(
            evidence_evaluated=True,
            paired_ci95=paired.get('paired_ci95'),
            paired_excludes_zero=paired.get('excludes_zero'),
            matched_seed_deltas=per_seed_deltas,
            mean_matched_seed_delta=(
                float(np.mean(per_seed_deltas)) if per_seed_deltas else None),
            promoted=promotes,
            promotion_reason=promotion_reason,
        )
        if promotes:
            selected = candidate['members']
            current_prediction = candidate['prediction']
            current_metrics = evaluator.score(
                current_prediction, 'valid', n_boot=n_boot)
            break

    trials.sort(key=lambda trial: trial['primary'], reverse=True)
    selected_ids = [node['node_id'] for node in selected]
    selected_weights = np.ones(len(selected), dtype=np.float64) / len(selected)
    weight_search = {
        'status': 'NOT_APPLICABLE' if len(selected) == 1 else 'EQUAL_WEIGHT_RETAINED',
        'grid_units': WEIGHT_GRID_UNITS,
        'trials': [],
    }
    if len(selected) > 1:
        equal_prediction = current_prediction
        equal_seed_primaries = []
        if matched_seed_count >= 3:
            for seed_index in range(matched_seed_count):
                equal_seed = np.mean([
                    ranked_seed_predictions[node_id][seed_index]
                    for node_id in selected_ids
                ], axis=0, dtype=np.float64).astype(np.float32)
                equal_seed_primaries.append(evaluator.score(
                    equal_seed, 'valid', n_boot=0)['primary'])
        weighted_candidates = []
        for integer_weights in product(
                range(1, WEIGHT_GRID_UNITS + 1), repeat=len(selected)):
            if sum(integer_weights) != WEIGHT_GRID_UNITS:
                continue
            weights = np.asarray(integer_weights, dtype=np.float64) / WEIGHT_GRID_UNITS
            prediction = np.sum([
                weight * ranked_predictions[node_id]
                for node_id, weight in zip(selected_ids, weights)
            ], axis=0).astype(np.float32)
            primary = evaluator.score(prediction, 'valid', n_boot=0)['primary']
            trial = {
                'weights': [float(weight) for weight in weights],
                'primary': float(primary),
                'delta_vs_equal_weight': float(
                    primary - current_metrics['primary']),
                'promoted': False,
            }
            weight_search['trials'].append(trial)
            weighted_candidates.append({
                'weights': weights, 'prediction': prediction,
                'primary': float(primary), 'trial': trial,
            })
        if weighted_candidates:
            best_weighted = max(
                weighted_candidates, key=lambda item: item['primary'])
            if best_weighted['primary'] > current_metrics['primary']:
                per_seed_deltas = []
                if equal_seed_primaries:
                    for seed_index in range(matched_seed_count):
                        weighted_seed = np.sum([
                            weight * ranked_seed_predictions[node_id][seed_index]
                            for node_id, weight in zip(
                                selected_ids, best_weighted['weights'])
                        ], axis=0).astype(np.float32)
                        primary = evaluator.score(
                            weighted_seed, 'valid', n_boot=0)['primary']
                        per_seed_deltas.append(float(
                            primary - equal_seed_primaries[seed_index]))
                paired = evaluator.compare(
                    best_weighted['prediction'], equal_prediction,
                    'valid', n=n_boot)
                promotes, reason = promotion_decision(
                    current_metrics['primary'], best_weighted['primary'],
                    paired, per_seed_deltas)
                best_weighted['trial'].update(
                    paired_ci95=paired.get('paired_ci95'),
                    paired_excludes_zero=paired.get('excludes_zero'),
                    matched_seed_deltas=per_seed_deltas,
                    mean_matched_seed_delta=(
                        float(np.mean(per_seed_deltas))
                        if per_seed_deltas else None),
                    promoted=promotes,
                    promotion_reason=reason,
                )
                if promotes:
                    selected_weights = best_weighted['weights']
                    current_prediction = best_weighted['prediction']
                    current_metrics = evaluator.score(
                        current_prediction, 'valid', n_boot=n_boot)
                    weight_search['status'] = 'WEIGHTED_SELECTED'
        weight_search['trials'].sort(
            key=lambda trial: trial['primary'], reverse=True)

    context_router = None
    context_search = {
        'status': 'DISABLED_DURING_ITERATIVE_SEARCH',
        'reason': ('context routing is fit only once at final portfolio designation to bound '
                   'validation search degrees of freedom and runtime'),
    }
    if contextual_refinement:
        context_router, current_prediction, current_metrics, context_search = (
            _contextual_weight_refinement(
                selected, selected_weights, ranked_predictions,
                ranked_seed_predictions, users, parsed_dir,
                current_prediction, current_metrics, matched_seed_count, n_boot))

    heterogeneous = len({
        (tuple(node.get('operator_stack') or ()), node.get('mechanism'))
        for node in selected}) > 1
    result = {
        'status': 'SELECTED' if len(selected) > 1 else 'SINGLE_BEST',
        'selected': len(selected) > 1,
        'members': [{
            'node_id': node['node_id'],
            'pipeline_path': node['pipeline_path'],
            'pipeline_sha256': node.get('pipeline_sha256'),
            'operator_stack': node.get('operator_stack', []),
            'mechanism': node.get('mechanism'),
            'standalone_primary': node['selection_primary'],
            'weight': float(selected_weights[index]),
        } for index, node in enumerate(selected)],
        'combination': (
            'exhaustive best-subset within-user rank average over node seed ensembles with '
            'bounded 0.1-step global weights' +
            (' and trusted coarse tab-aware routing' if context_router else '')),
        'heterogeneous': heterogeneous,
        'selection_primary': float(current_metrics['primary']),
        'single_best_primary': float(single_metrics['primary']),
        'delta_vs_single_best': float(
            current_metrics['primary'] - single_metrics['primary']),
        'promotion_gate': ('positive paired CI95 lower bound OR >=2/3 matched seeds positive, '
                           f'mean gain >= {MIN_CONSISTENT_SEED_GAIN:g}, worst seed >= '
                           f'-{MAX_MATCHED_SEED_REGRESSION:g}'),
        'weight_search': weight_search,
        'context_search': context_search,
        'pool_node_ids': [node['node_id'] for node in pool],
        'trials': trials,
        'test_labels_used': False,
    }
    if context_router is not None:
        result['context_router'] = context_router
    if incumbent_selection is not None:
        comparison = compare_selections(
            incumbent_selection, result, frontier, parsed_dir,
            n_boot=n_boot, candidate_node_id=candidate_node_id)
        result['incumbent_comparison'] = comparison
        candidate_credit = (candidate_node_id is None
                            or comparison['candidate_entered'])
        if not (comparison['promoted'] and candidate_credit):
            # The globally best challenger subset is not necessarily the best substrate for a
            # contextual router.  At final designation, do not discard a robust contextual
            # refinement of the deployed portfolio merely because another subset had a slightly
            # higher global point estimate before routing.
            if (contextual_refinement
                    and not incumbent_selection.get('context_router')
                    and len(incumbent_selection.get('members') or []) > 1):
                incumbent_ids = [
                    item['node_id'] for item in incumbent_selection['members']]
                incumbent_nodes = [frontier.nodes[node_id]
                                   for node_id in incumbent_ids]
                incumbent_weights = [
                    float(item.get('weight', 0.0))
                    for item in incumbent_selection['members']]
                incumbent_prediction = selection_prediction(
                    incumbent_selection, frontier, parsed_dir)
                incumbent_metrics = evaluator.score(
                    incumbent_prediction, 'valid', n_boot=n_boot)
                incumbent_router, _, refined_metrics, incumbent_context_search = (
                    _contextual_weight_refinement(
                        incumbent_nodes, incumbent_weights, ranked_predictions,
                        ranked_seed_predictions, users, parsed_dir,
                        incumbent_prediction, incumbent_metrics,
                        matched_seed_count, n_boot))
                if incumbent_router is not None:
                    refined = copy.deepcopy(incumbent_selection)
                    refined.update(
                        status='INCUMBENT_CONTEXT_REFINED',
                        selection_primary=float(refined_metrics['primary']),
                        context_router=incumbent_router,
                        context_search=incumbent_context_search,
                        combination=(
                            incumbent_selection.get('combination', 'weighted portfolio')
                            + ' with trusted coarse tab-aware routing'),
                        challenger={
                            'status': result['status'],
                            'members': result['members'],
                            'selection_primary': result['selection_primary'],
                            'candidate_entered': comparison['candidate_entered'],
                            'promotion_reason': comparison['promotion_reason'],
                        },
                        pool_node_ids=result['pool_node_ids'],
                        trials=result['trials'],
                        weight_search=result['weight_search'],
                        test_labels_used=False,
                    )
                    refined['incumbent_comparison'] = compare_selections(
                        incumbent_selection, refined, frontier, parsed_dir,
                        n_boot=n_boot, candidate_node_id=None)
                    return refined
            retained = copy.deepcopy(incumbent_selection)
            retained.update(
                status='INCUMBENT_RETAINED',
                incumbent_comparison=comparison,
                challenger={
                    'status': result['status'],
                    'members': result['members'],
                    'selection_primary': result['selection_primary'],
                    'candidate_entered': comparison['candidate_entered'],
                    'promotion_reason': comparison['promotion_reason'],
                },
                pool_node_ids=result['pool_node_ids'],
                trials=result['trials'],
                weight_search=result['weight_search'],
                context_search=result['context_search'],
                test_labels_used=False,
            )
            return retained
    return result
