"""Regression tests for deterministic controller behavior."""
import json
import math
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'trusted'))
sys.path.insert(0, os.path.join(ROOT, 'orchestrator'))
sys.path.insert(0, os.path.join(ROOT, 'task_spec'))

import agent
import core
import ensemble
import evaluator
import external_research
import frontier
import llm
import memory
import operators
import prior_store
import schemas
import gates
import stable_ops


def proposal(persona, hypothesis, block):
    evidence_id = {
        'optimizer': 'E001', 'architecture': 'E002', 'reward': 'E003',
    }[persona]
    return {
        'persona': persona,
        'execution_mode': 'custom_patch',
        'operator_id': None,
        'basis_type': 'external_research',
        'mechanism': f'{persona} control mechanism',
        'mechanism_tags': [persona, 'structural'],
        'parent_references': [],
        'implementation_plan': f'vectorized implementation in {block}',
        'hypothesis': hypothesis,
        'observation': 'a concrete observation',
        'justification': 'a falsifiable justification',
        'evidence': [{'type': 'current_code', 'ref': 'pipeline.py'},
                     {'type': 'external_research',
                      'ref': f'{evidence_id} persona-specific method'}],
        'evidence_adaptation': {
            'evidence_id': evidence_id,
            'source_mechanism': 'a concrete source mechanism',
            'implementation_mapping': f'map vectorized arrays into {block}',
            'protocol_caveat': 'none',
        },
        'research_nonuse_reason': None,
        'expected_observation': 'primary increases',
        'experiment_contract': {
            'claim': f'{persona} mechanism changes validation ordering',
            'required_components': [f'executable {block} change'],
            'forbidden_shortcuts': ['comment-only or unused configuration'],
            'expected_observable': 'predictions differ from parent',
            'falsification_condition': 'predictions are identical or primary declines',
            'portfolio_role': 'standalone_improvement',
        },
        'primary_block': block,
        'patch_scope': [block],
        'estimated_cost_s': 10,
        'risk': [],
        'fallback': 'rollback',
    }


class AgentCoreTest(unittest.TestCase):
    LIVE_KINDS = {f'E{index:03d}': 'external_research'
                  for index in range(1, 4)}
    LIVE_PERSONAS = {
        'E001': 'optimizer', 'E002': 'architecture', 'E003': 'reward',
    }

    def test_gpt5_models_use_modern_completion_parameters(self):
        self.assertTrue(llm.uses_modern_completion_params('gpt-5.4'))
        self.assertTrue(llm.uses_modern_completion_params('gpt-5.6-sol'))
        self.assertFalse(llm.uses_modern_completion_params('gpt-4o'))

    def test_convergence_requires_three_consecutive_small_gains(self):
        counter = 0
        counter, gain = agent.convergence_update(counter, 0.50, 0.501)
        self.assertEqual((counter, gain), (1, 0.0010000000000000009))
        counter, _ = agent.convergence_update(counter, 0.501, 0.504)
        self.assertEqual(counter, 0)
        for _ in range(3):
            counter, _ = agent.convergence_update(counter, 0.504, 0.504)
        self.assertEqual(counter, 3)
        self.assertTrue(agent.convergence_should_stop(True, stopping_enabled=True))
        self.assertFalse(agent.convergence_should_stop(True, stopping_enabled=False))

    def test_positive_noise_is_uncertain_but_significant_gain_is_accepted(self):
        unresolved = {'paired_ci95': [-0.0001, 0.0002], 'excludes_zero': False}
        positive = {'paired_ci95': [0.00001, 0.0003], 'excludes_zero': True}
        self.assertEqual(agent.classify_candidate(0.00003, unresolved), 'UNCERTAIN')
        self.assertEqual(agent.classify_candidate(0.00003, positive), 'ACCEPT')
        self.assertEqual(agent.classify_candidate(-0.00003, positive), 'ROLLBACK')

    def test_portfolio_gain_can_accept_a_standalone_rollback(self):
        comparison = {
            'candidate_entered': True,
            'delta_primary': 0.00031,
            'promoted': True,
        }
        decision, channel = agent.portfolio_aware_decision(
            'ROLLBACK', comparison)
        self.assertEqual(decision, 'ACCEPT')
        self.assertEqual(channel, 'portfolio_marginal')
        comparison['promoted'] = False
        decision, channel = agent.portfolio_aware_decision(
            'ROLLBACK', comparison)
        self.assertEqual(decision, 'UNCERTAIN')
        self.assertEqual(channel, 'portfolio_marginal_unconfirmed')

    def test_experiment_contract_audit_is_fail_closed(self):
        required = ['sample weights reach gradient', 'predictions can change']
        validate = schemas.implementation_audit_validator(required)
        passing = {
            'status': 'PASS',
            'component_evidence': [
                {'component': required[0], 'code_evidence': 'loss multiplies g by weights'},
                {'component': required[1], 'code_evidence': 'updated parameters feed predict'},
            ],
            'missing_components': [],
            'forbidden_shortcuts_found': [],
            'changed_runtime_path': True,
            'analysis': 'connected dataflow',
        }
        self.assertIs(validate(passing), passing)
        invalid = dict(passing, status='PASS', changed_runtime_path=False)
        with self.assertRaisesRegex(ValueError, 'PASS'):
            validate(invalid)

    def test_ensemble_promotes_small_gain_when_all_matched_seeds_improve(self):
        unresolved = {'paired_ci95': [-0.0006, 0.0012], 'excludes_zero': False}
        promoted, reason = ensemble.promotion_decision(
            0.60475, 0.60501, unresolved,
            [0.00048, 0.00079, 0.00003])
        self.assertTrue(promoted)
        self.assertEqual(reason, 'bounded_matched_seed_robustness')

    def test_ensemble_allows_one_tightly_bounded_seed_regression(self):
        unresolved = {'paired_ci95': [-0.0003, 0.0006], 'excludes_zero': False}
        promoted, reason = ensemble.promotion_decision(
            0.60567, 0.60583, unresolved,
            [-0.000036, 0.000188, 0.000295])
        self.assertTrue(promoted)
        self.assertEqual(reason, 'bounded_matched_seed_robustness')

    def test_context_router_uses_only_declared_row_context_and_member_weights(self):
        users = np.array([10, 10, 20, 20])
        tabs = np.array([1, 6, 1, 6])
        left = np.array([0.0, 1.0, 0.0, 1.0])
        right = np.array([1.0, 0.0, 1.0, 0.0])
        router = {
            'feature': 'tab',
            'routes': [{
                'value': 6,
                'member_weights': {'left': 1.0, 'right': 0.0},
            }],
        }
        combined = ensemble.combine_member_predictions(
            [left, right], users, ['left', 'right'], [0.5, 0.5],
            context_router=router, context_values=tabs)
        np.testing.assert_allclose(combined, [0.5, 1.0, 0.5, 1.0])
        with self.assertRaisesRegex(ValueError, 'context values'):
            ensemble.combine_member_predictions(
                [left, right], users, ['left', 'right'], [0.5, 0.5],
                context_router=router)

    def test_context_router_requires_and_accepts_strict_all_seed_evidence(self):
        selected = [{'node_id': 'left'}, {'node_id': 'right'}]
        ranked = {
            'left': np.ones(4, dtype=np.float32),
            'right': np.zeros(4, dtype=np.float32),
        }
        ranked_seeds = {
            node_id: [prediction.copy() for _ in range(3)]
            for node_id, prediction in ranked.items()
        }
        current = np.full(4, 0.5, dtype=np.float32)

        def score(prediction, split, n_boot=0):
            self.assertEqual(split, 'valid')
            return {'primary': float(np.mean(prediction))}

        with tempfile.TemporaryDirectory() as tmp:
            np.save(os.path.join(tmp, 'valid.tab.npy'),
                    np.zeros(4, dtype=np.int32))
            with mock.patch.object(ensemble, 'CONTEXT_MIN_ROWS', 1), \
                    mock.patch.object(ensemble.evaluator, 'score', side_effect=score), \
                    mock.patch.object(
                        ensemble.evaluator, 'compare',
                        return_value={
                            'paired_ci95': [-0.1, 0.6],
                            'excludes_zero': False,
                        }):
                router, combined, metrics, status = (
                    ensemble._contextual_weight_refinement(
                        selected, [0.5, 0.5], ranked, ranked_seeds,
                        np.array([1, 1, 2, 2]), tmp, current,
                        {'primary': 0.5}, matched_seed_count=3, n_boot=10))
        self.assertIsNotNone(router)
        self.assertEqual(status['status'], 'CONTEXT_ROUTER_SELECTED')
        self.assertEqual(status['promotion_reason'], 'strict_all_seed_robustness')
        self.assertEqual(status['matched_seed_deltas'], [0.5, 0.5, 0.5])
        self.assertAlmostEqual(metrics['primary'], 1.0)
        np.testing.assert_array_equal(combined, np.ones(4, dtype=np.float32))

    def test_final_selector_can_context_refine_incumbent_after_global_challenger_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            parsed = os.path.join(tmp, 'parsed')
            os.makedirs(parsed)
            np.save(os.path.join(parsed, 'valid.user_id.npy'),
                    np.array([1, 1, 2, 2], dtype=np.int32))
            np.save(os.path.join(parsed, 'valid.tab.npy'),
                    np.array([0, 1, 0, 1], dtype=np.int32))
            nodes = {}
            for node_id, prediction, primary in (
                    ('left', [0.0, 1.0, 0.0, 1.0], 0.60),
                    ('right', [1.0, 0.0, 1.0, 0.0], 0.59)):
                path = f'{node_id}.npy'
                np.save(os.path.join(tmp, path), np.asarray(prediction, dtype=np.float32))
                nodes[node_id] = {
                    'node_id': node_id, 'status': 'COMPLETE',
                    'pipeline_path': f'{node_id}.py',
                    'pipeline_sha256': node_id,
                    'prediction_paths': [path],
                    'selection_primary': primary,
                    'mechanism': node_id,
                }
            graph = types.SimpleNamespace(nodes=nodes, run_dir=tmp)
            incumbent = {
                'status': 'SELECTED',
                'selection_primary': 0.60,
                'combination': 'global weighted portfolio',
                'members': [
                    {'node_id': 'left', 'weight': 0.5},
                    {'node_id': 'right', 'weight': 0.5},
                ],
            }
            calls = []

            def context_refinement(*args):
                calls.append(args)
                current_prediction, current_metrics = args[6], args[7]
                if len(calls) == 1:
                    return (None, current_prediction, current_metrics,
                            {'status': 'EVIDENCE_REJECTED'})
                router = {
                    'feature': 'tab', 'test_label_free': True,
                    'routes': [{'value': 1, 'member_weights': {
                        'left': 0.8, 'right': 0.2}}],
                }
                return (router, np.full(4, 0.61, dtype=np.float32),
                        {'primary': 0.61},
                        {'status': 'CONTEXT_ROUTER_SELECTED'})

            rejected = {
                'promoted': False, 'candidate_entered': False,
                'promotion_reason': 'insufficient_promotion_evidence',
            }
            accepted = {
                'promoted': True, 'candidate_entered': False,
                'promotion_reason': 'strict_all_seed_robustness',
            }
            with mock.patch.object(
                    ensemble.evaluator, 'score',
                    side_effect=lambda prediction, split, n_boot=0: {
                        'primary': float(np.mean(prediction))}), \
                    mock.patch.object(
                        ensemble, '_contextual_weight_refinement',
                        side_effect=context_refinement), \
                    mock.patch.object(
                        ensemble, 'compare_selections',
                        side_effect=[rejected, accepted]):
                result = ensemble.select(
                    graph, parsed, n_boot=0, max_members=2,
                    incumbent_selection=incumbent,
                    contextual_refinement=True)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result['status'], 'INCUMBENT_CONTEXT_REFINED')
        self.assertAlmostEqual(result['selection_primary'], 0.61)
        self.assertTrue(result['context_router']['test_label_free'])

    def test_empirical_portfolio_verification_is_fail_closed(self):
        portfolio = {
            'portfolio_id': 'validated_rank_portfolio_v1',
            'operators': ['legal_rank_stack_v1', 'item_lambdarank'],
            'weights': [0.7, 0.3],
            'validation_selection_primary': 0.61,
        }
        selection = {
            'selection_primary': 0.61,
            'members': [
                {'operator_stack': ['legal_rank_stack_v1'], 'weight': 0.7},
                {'operator_stack': ['item_lambdarank'], 'weight': 0.3},
            ],
        }
        verified = agent.verify_empirical_portfolio_selection(
            portfolio, selection)
        self.assertEqual(verified['status'], 'VERIFIED')
        selection['members'][0]['weight'] = 0.6
        with self.assertRaisesRegex(ValueError, 'weights changed'):
            agent.verify_empirical_portfolio_selection(portfolio, selection)

    def test_empirical_context_router_is_materialized_without_refitting(self):
        portfolio = {
            'portfolio_id': 'validated-router',
            'operators': ['left_op', 'right_op'],
            'weights': [0.6, 0.4],
            'global_validation_selection_primary': 0.60,
            'validation_selection_primary': 0.61,
            'context_router': {
                'feature': 'tab', 'fallback': 'global operator weights',
                'min_rows': 1000, 'weight_grid_step': 0.1,
                'inference_label_free': True,
                'promotion_reason': 'strict_all_seed_robustness',
                'matched_seed_deltas': [0.01, 0.01, 0.01],
                'routes': [{
                    'value': 6, 'rows': 1200,
                    'operator_weights': {'left_op': 0.2, 'right_op': 0.8},
                }],
            },
        }
        selection = {
            'selection_primary': 0.60,
            'combination': 'global weighted portfolio',
            'members': [
                {'node_id': 'w001', 'operator_stack': ['left_op'], 'weight': 0.6},
                {'node_id': 'w002', 'operator_stack': ['right_op'], 'weight': 0.4},
            ],
        }

        def selection_prediction(current, *_args, **_kwargs):
            value = 0.61 if current.get('context_router') else 0.60
            return np.full(4, value, dtype=np.float64)

        with mock.patch.object(
                agent.heterogeneous_ensemble, 'selection_prediction',
                side_effect=selection_prediction), mock.patch.object(
                    agent.evaluator, 'score',
                    side_effect=lambda prediction, *_args, **_kwargs: {
                        'primary': float(np.mean(prediction))}), mock.patch.object(
                    agent.evaluator, 'compare', return_value={
                        'paired_ci95': [-0.01, 0.02], 'excludes_zero': False}):
            routed, verification = agent.apply_empirical_context_router(
                portfolio, selection, object(), '/unused', n_boot=10)

        self.assertEqual(routed['status'], 'EMPIRICAL_CONTEXT_ROUTER_VERIFIED')
        self.assertEqual(
            routed['context_router']['routes'][0]['member_weights'],
            {'w001': 0.2, 'w002': 0.8})
        self.assertFalse(routed['context_search']['fit_during_run'])
        self.assertAlmostEqual(routed['selection_primary'], 0.61)
        self.assertEqual(verification['status'], 'VERIFIED')

    def test_empirical_warmstart_is_one_logical_iteration(self):
        portfolio = {
            'portfolio_id': 'atomic-test',
            'evidence_id': 'E001',
            'operators': ['legal_rank_stack_v1', 'item_lambdarank'],
            'weights': [0.7, 0.3],
            'validation_selection_primary': 0.61,
        }
        baseline_metrics = {'selection_primary': 0.59}
        selection = {
            'selection_primary': 0.61,
            'members': [
                {'operator_stack': ['legal_rank_stack_v1'], 'weight': 0.7},
                {'operator_stack': ['item_lambdarank'], 'weight': 0.3},
            ],
        }
        candidate_metrics = [
            {'selection_primary': 0.60, 'paired_vs_incumbent': {
                'paired_ci95': [0.001, 0.003], 'excludes_zero': True}},
            {'selection_primary': 0.595, 'paired_vs_incumbent': {
                'paired_ci95': [0.0001, 0.002], 'excludes_zero': True}},
        ]

        class PassingGate:
            ok = True

            @staticmethod
            def as_event():
                return {'ok': True}

        def materialize(_source, _stack, operator_id):
            spec = operators.SPECS[operator_id]
            source = f'# candidate {operator_id}\n'
            return {
                'operator_id': operator_id,
                'operator_stack': [operator_id],
                'logical_scope': list(spec.logical_scope),
                'materialized_scope': list(spec.logical_scope),
                'config': {},
                'patch': {'source': source},
                'source': source,
            }

        def write_candidate(_parent, patch, destination):
            with open(destination, 'w', encoding='utf-8') as handle:
                handle.write(patch['source'])

        frontier_mock = mock.Mock()
        frontier_mock.add_node.side_effect = lambda node: node
        with tempfile.TemporaryDirectory() as tmp:
            baseline_path = os.path.join(tmp, 'baseline.py')
            with open(baseline_path, 'w', encoding='utf-8') as handle:
                handle.write('# baseline\n')
            with (mock.patch.object(agent.operators, 'materialize', side_effect=materialize),
                  mock.patch.object(agent.patching, 'write_candidate',
                                    side_effect=write_candidate),
                  mock.patch.object(agent.gates, 'run_static_gates',
                                    return_value=PassingGate()),
                  mock.patch.object(agent, 'smoke_test',
                                    return_value=(True, {'ok': True})),
                  mock.patch.object(agent, 'run_seeds', return_value=[
                      {'ok': True, 'pred': 's0.npy'},
                      {'ok': True, 'pred': 's1.npy'},
                      {'ok': True, 'pred': 's2.npy'}]),
                  mock.patch.object(agent, 'score_prediction_set',
                                    side_effect=candidate_metrics),
                  mock.patch.object(agent.heterogeneous_ensemble, 'select',
                                    return_value=selection),
                  mock.patch('builtins.print')):
                state = agent.run_empirical_portfolio_warmstart(
                    portfolio, baseline_path,
                    ['b0.npy', 'b1.npy', 'b2.npy'], baseline_metrics,
                    frontier_mock, tmp, 10, 1, float('inf'), 10)

            self.assertEqual(len(state['events']), 1)
            self.assertEqual(state['events'][0]['iter'], 1)
            self.assertEqual(state['events'][0]['measured_subexperiments'], 2)
            self.assertEqual(len(state['subexperiments']), 2)
            self.assertTrue(all(
                not item['counts_as_logical_iteration']
                for item in state['subexperiments']))
            self.assertEqual(
                [entry['memory_id'] for entry in state['memories']],
                ['wm001', 'wm002'])
            self.assertEqual(
                [call.args[0]['node_id'] for call in frontier_mock.add_node.call_args_list],
                ['w001', 'w002'])
            self.assertTrue(os.path.isdir(os.path.join(tmp, 'iter-001')))
            self.assertFalse(os.path.exists(os.path.join(tmp, 'iter-002')))
            self.assertEqual(state['events'][0]['convergence'][
                'consecutive_small_gain'], 0)

    def test_ensemble_mixed_seed_gain_still_requires_positive_ci(self):
        unresolved = {'paired_ci95': [-0.0001, 0.0004], 'excludes_zero': False}
        promoted, reason = ensemble.promotion_decision(
            0.6047, 0.6049, unresolved,
            [0.0005, 0.0003, -0.0001])
        self.assertFalse(promoted)
        self.assertEqual(reason, 'insufficient_promotion_evidence')

    def test_ensemble_subset_search_can_select_triple_past_uncertain_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            parsed = os.path.join(tmp, 'parsed')
            os.makedirs(parsed)
            np.save(os.path.join(parsed, 'valid.user_id.npy'), np.arange(3))
            vectors = {
                'n000': np.array([1.0, 0.0, 0.0], dtype=np.float32),
                'n001': np.array([0.0, 1.0, 0.0], dtype=np.float32),
                'n002': np.array([0.0, 0.0, 1.0], dtype=np.float32),
            }
            nodes = {}
            for index, (node_id, vector) in enumerate(vectors.items()):
                paths = []
                for seed in range(3):
                    name = f'{node_id}-s{seed}.npy'
                    np.save(os.path.join(tmp, name), vector)
                    paths.append(name)
                nodes[node_id] = {
                    'node_id': node_id, 'status': 'COMPLETE',
                    'pipeline_path': f'{node_id}.py',
                    'pipeline_sha256': node_id,
                    'prediction_paths': paths,
                    'selection_primary': [0.60, 0.59, 0.58][index],
                    'operator_stack': [node_id], 'mechanism': node_id,
                }
            graph = types.SimpleNamespace(nodes=nodes, run_dir=tmp)
            point_scores = {
                (1_000_000, 0, 0): 0.6000,
                (0, 1_000_000, 0): 0.5900,
                (0, 0, 1_000_000): 0.5800,
                (500_000, 500_000, 0): 0.6002,
                (500_000, 0, 500_000): 0.5990,
                (0, 500_000, 500_000): 0.5980,
            }

            def fake_score(prediction, split, n_boot=0):
                values = np.asarray(prediction, dtype=np.float64)
                if np.all(values > 0):
                    target = np.array([0.4, 0.4, 0.2])
                    return {'primary': float(
                        0.6015 - 0.01 * np.square(values - target).sum())}
                key = tuple(np.rint(values * 1_000_000).astype(np.int64))
                return {'primary': point_scores[key]}

            unresolved = {
                'delta_primary': 0.001, 'paired_ci95': [-0.001, 0.002],
                'excludes_zero': False,
            }
            average = lambda predictions, users: np.mean(
                predictions, axis=0, dtype=np.float64).astype(np.float32)
            with mock.patch.object(ensemble, 'within_user_rank_average', average), \
                    mock.patch.object(ensemble.evaluator, 'score', side_effect=fake_score), \
                    mock.patch.object(ensemble.evaluator, 'compare',
                                      return_value=unresolved):
                selected = ensemble.select(
                    graph, parsed, n_boot=10, max_members=3, max_pool=3)
            self.assertEqual(selected['status'], 'SELECTED')
            self.assertEqual(
                [member['node_id'] for member in selected['members']],
                ['n000', 'n001', 'n002'])
            self.assertAlmostEqual(selected['selection_primary'], 0.6015)
            self.assertEqual(
                [member['weight'] for member in selected['members']],
                [0.4, 0.4, 0.2])

    def test_persona_set_is_exact(self):
        obj = {'candidates': [
            proposal('optimizer', 'change optimization', 'train'),
            proposal('architecture', 'change representation', 'model'),
            proposal('reward', 'change supervision', 'target'),
        ]}
        self.assertIs(agent.validate_proposal_set(
            obj, self.LIVE_KINDS, self.LIVE_PERSONAS), obj)
        obj['candidates'][2]['persona'] = 'optimizer'
        with self.assertRaises(ValueError):
            agent.validate_proposal_set(obj, self.LIVE_KINDS, self.LIVE_PERSONAS)

    def test_diagnostic_evidence_must_be_numeric(self):
        obj = {'candidates': [
            proposal('optimizer', 'change optimization', 'train'),
            proposal('architecture', 'change representation', 'model'),
            proposal('reward', 'change supervision', 'target'),
        ]}
        obj['candidates'][2]['evidence'] = [
            {'type': 'external_research', 'ref': 'E003 auxiliary targets'},
            {'type': 'diagnostic', 'ref': 'is_like correlation is strong'}]
        with self.assertRaises(ValueError):
            agent.validate_proposal_set(obj, self.LIVE_KINDS, self.LIVE_PERSONAS)

    def test_research_derived_candidate_requires_real_research_evidence(self):
        obj = {'candidates': [
            proposal('optimizer', 'change optimization', 'train'),
            proposal('architecture', 'change representation', 'model'),
            proposal('reward', 'change supervision', 'target'),
        ]}
        obj['candidates'][0]['evidence'] = [{'type': 'current_code', 'ref': 'pipeline.py'}]
        with self.assertRaises(ValueError):
            agent.validate_proposal_set(obj, self.LIVE_KINDS, self.LIVE_PERSONAS)

    def test_nonresearch_tuning_candidate_does_not_require_e_id(self):
        obj = {'candidates': [
            proposal('optimizer', 'tune learning rate', 'train'),
            proposal('architecture', 'change representation', 'model'),
            proposal('reward', 'change supervision', 'target'),
        ]}
        tuning = obj['candidates'][0]
        tuning.update(
            basis_type='current_code', evidence_adaptation=None,
            research_nonuse_reason=('The hypothesis is a bounded change to the current learning '
                                    'rate; no retrieved source identifies its dataset-specific value.'),
            evidence=[{'type': 'current_code', 'ref': "HP['lr'] = 0.001 in train"}])
        self.assertIs(agent.validate_proposal_set(
            obj, self.LIVE_KINDS, self.LIVE_PERSONAS), obj)

    def test_cross_persona_research_is_allowed_but_remains_auditable(self):
        item = proposal('reward', 'change supervision', 'target')
        item['evidence'] = [
            {'type': 'external_research', 'ref': 'E001 optimizer-only evidence'}]
        item['evidence_adaptation']['evidence_id'] = 'E001'
        self.assertIs(agent._validate_candidate_research(
            item, self.LIVE_KINDS, self.LIVE_PERSONAS), item)

    def test_unknown_external_evidence_id_is_still_rejected(self):
        item = proposal('optimizer', 'change optimization', 'loss')
        item['evidence'] = [
            {'type': 'external_research', 'ref': 'E999 nonexistent evidence'}]
        item['evidence_adaptation']['evidence_id'] = 'E999'
        with self.assertRaisesRegex(ValueError, '不存在'):
            agent._validate_candidate_research(
                item, self.LIVE_KINDS, self.LIVE_PERSONAS)

    def test_all_nonresearch_candidates_require_set_level_explanation(self):
        candidates = [
            proposal('optimizer', 'tune optimizer', 'train'),
            proposal('architecture', 'add safe feature', 'features'),
            proposal('reward', 'adjust target transform', 'target'),
        ]
        for item in candidates:
            item['basis_type'] = 'current_code'
            item['evidence'] = [{'type': 'current_code', 'ref': 'specific pipeline block'}]
            item['evidence_adaptation'] = None
            item['research_nonuse_reason'] = 'No retrieved mechanism directly addresses this gap.'
        obj = {'candidates': candidates}
        with self.assertRaisesRegex(ValueError, '顶层|解释'):
            agent.validate_proposal_set(obj, self.LIVE_KINDS, self.LIVE_PERSONAS)
        obj['research_nonuse_reason'] = (
            'All retrieved methods require incompatible features; these are bounded code hypotheses.')
        self.assertIs(agent.validate_proposal_set(
            obj, self.LIVE_KINDS, self.LIVE_PERSONAS), obj)

    def test_rank_average_is_within_user_and_tie_aware(self):
        users = np.array([1, 1, 1, 2, 2])
        first = np.array([0.1, 0.2, 0.3, 8.0, 2.0])
        second = np.array([3.0, 2.0, 1.0, 1.0, 1.0])
        got = agent.within_user_rank_average([first, second], users)
        np.testing.assert_allclose(got, [0.5, 0.5, 0.5, 0.75, 0.25])

    def test_selection_filters_memory_before_fallback_selection(self):
        candidates = [
            proposal('optimizer', 'change optimization', 'train'),
            proposal('architecture', 'change representation', 'model'),
            proposal('reward', 'repeat failed target blend', 'target'),
        ]
        candidates[0].update(
            basis_type='current_code', evidence_adaptation=None,
            research_nonuse_reason='This is a bounded current optimizer setting.',
            evidence=[{'type': 'current_code', 'ref': "HP['lr'] current value"}])
        candidates[1]['evidence'] = [
            {'type': 'external_research', 'ref': 'E001 cross-persona mechanism'}]
        candidates[1]['evidence_adaptation']['evidence_id'] = 'E001'

        class FakeClient:
            def call(self, phase, system, prompt, max_tokens):
                if phase == 'draft_candidates':
                    return {'candidates': candidates}, {'phase': phase, 'total_tokens': 10}
                if phase == 'select_candidate':
                    # Candidate 2 was removed; select viable index 1 (original index 1).
                    return {'selected_index': 1, 'selection_rationale': 'higher upside'}, {
                        'phase': phase, 'total_tokens': 5}
                raise AssertionError(phase)

        def screen(_client, _source, _metrics, item, _memories, _bundle,
                   _evidence_kinds, _usages, evidence_personas=None,
                   parent_operator_stack=None):
            blocker = ({'kind': 'memory_reject'} if item['persona'] == 'reward' else None)
            return item, [{'action': 'reject' if blocker else 'proceed'}], blocker

        bundle = {
            'mode': 'offline',
            'sources': [{'source_id': 'S001', 'kind': 'url',
                         'url': 'https://example.test', 'title': 'source'}],
            'evidence': [
                {'evidence_id': evidence_id, 'kind': 'external_research',
                 'persona': persona, 'summary': 'method evidence',
                 'source_ids': ['S001']}
                for evidence_id, persona in self.LIVE_PERSONAS.items()
            ],
        }
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
                agent, 'screen_candidate_memory', side_effect=screen):
            prior = {'memory_id': 'm001', 'kind': 'failure', 'primary_block': 'target',
                     'hypothesis': 'failed target blend', 'conclusion': 'rolled back'}
            selected, traces, blocker = agent.select_experiment(
                FakeClient(), 'source', {'selection_primary': 0.6}, [prior], bundle,
                '{}', '{}', '(none)', tmp, [], 1)
            self.assertIsNone(blocker)
            self.assertEqual(selected['persona'], 'architecture')
            self.assertEqual(traces[-1]['selected_index'], 1)
            self.assertEqual(traces[-1]['eligible_original_indices'], [0, 1])
            import json
            with open(os.path.join(tmp, 'research-citations.json'), encoding='utf-8') as fh:
                audit = json.load(fh)
            self.assertEqual(audit[0]['basis_type'], 'current_code')
            self.assertEqual(audit[0]['evidence_ids'], [])
            self.assertIsNone(audit[0]['primary_evidence_persona_match'])
            self.assertFalse(audit[1]['primary_evidence_persona_match'])

    def test_prediction_set_selects_on_exact_submission_ensemble(self):
        seed_metrics = {'primary_mean': 0.51,
                        'paired_vs_incumbent': {'delta_primary': 0.01}}
        ensembles = [np.array([0.2, 0.8]), np.array([0.1, 0.9])]
        with mock.patch.object(agent.evaluator, 'score_seeds', return_value=seed_metrics), \
                mock.patch.object(agent, 'prediction_ensemble', side_effect=ensembles), \
                mock.patch.object(agent.evaluator, 'score', return_value={'primary': 0.62}), \
                mock.patch.object(agent.evaluator, 'compare',
                                  return_value={'delta_primary': 0.02}), \
                mock.patch.object(agent.evaluator, 'directional_compare', return_value={}):
            got = agent.score_prediction_set(['a', 'b'], 'valid', ['c', 'd'], n_boot=0)
        self.assertEqual(got['selection_primary'], 0.62)
        self.assertEqual(got['paired_vs_incumbent']['delta_primary'], 0.02)
        self.assertEqual(got['per_seed_paired_vs_incumbent']['delta_primary'], 0.01)

    def test_zero_bootstrap_is_a_supported_point_estimate_mode(self):
        stats = [(np.array([1, 2]), np.array([1.0, 1.0]),
                  np.array([0.5, 0.6]), np.array([0.7, 0.8]))]
        ci, sd = evaluator.bootstrap_seed_mean(stats, n=0)
        self.assertTrue(all(math.isnan(value) for value in ci))
        self.assertTrue(math.isnan(sd))
        paired = evaluator.paired_bootstrap_seed_mean(stats, stats, n=0)
        self.assertEqual(paired['delta_primary'], 0.0)
        self.assertFalse(paired['excludes_zero'])

    def test_failed_duplicate_requires_memory_review(self):
        prior = {'memory_id': 'm001', 'kind': 'failure', 'primary_block': 'train',
                 'patch_scope': ['train'],
                 'hypothesis': 'Increase learning rate from 0.001 to 0.01',
                 'conclusion': 'rolled back'}
        current = proposal('optimizer', 'Set lr from 0.001 to 0.01', 'train')
        matches = memory.retrieve([prior], current)
        self.assertTrue(memory.requires_memory_review(matches))

    def test_implementation_failure_does_not_hard_block_a_new_implementation(self):
        first = proposal('optimizer', 'use pairwise BPR', 'loss')
        first['mechanism'] = 'Bayesian personalized ranking'
        first['implementation_plan'] = 'nested user pair loops in loss'
        entry = memory.build_entry(1, first,
            {'decision': 'REJECT', 'reason': {'kind': 'smoke'}},
            {'next_lesson': 'vectorize', 'result': 'failed', 'analysis': 'timeout'})
        second = proposal('optimizer', 'use bounded pairwise BPR', 'loss')
        second['mechanism'] = 'Bayesian personalized ranking'
        second['implementation_plan'] = 'bounded vectorized pairs in loss'
        matches = memory.retrieve([entry], second)
        self.assertTrue(matches[0]['mechanism_similarity'] > 0.5)
        self.assertFalse(matches[0]['same_implementation'])

    def test_candidate_view_physically_withholds_validation_and_test_labels(self):
        parsed = os.path.join(ROOT, 'views', 'agent', 'parsed')
        if not os.path.isdir(parsed):
            self.skipTest('generated views are intentionally excluded from the submission package')
        self.assertTrue(os.path.exists(os.path.join(parsed, 'train.label.npy')))
        self.assertFalse(os.path.exists(os.path.join(parsed, 'valid.label.npy')))
        self.assertFalse(os.path.exists(os.path.join(parsed, 'test.label.npy')))

    def test_gates_reject_label_access_outside_target_and_nested_loss_loops(self):
        baseline = os.path.join(ROOT, 'candidate', 'pipeline.py')
        with open(baseline, encoding='utf-8') as fh:
            source = fh.read()
        with tempfile.TemporaryDirectory() as tmp:
            leaked = source.replace(
                "    Xtr = Xs['train'][train_idx]",
                "    Xtr = Xs['train'][train_idx]\n    stolen = splits['valid'].label", 1)
            leaked_path = os.path.join(tmp, 'leaked.py')
            with open(leaked_path, 'w', encoding='utf-8') as fh:
                fh.write(leaked)
            self.assertFalse(gates.g2_lineage(leaked_path).ok)

            quadratic = source.replace(
                '    B = len(y)',
                '    B = len(y)\n    for i in range(B):\n        for j in range(B):\n            pass',
                1)
            quadratic_path = os.path.join(tmp, 'quadratic.py')
            with open(quadratic_path, 'w', encoding='utf-8') as fh:
                fh.write(quadratic)
            result = gates.g3_code(quadratic_path)
            self.assertFalse(result.ok)
        self.assertIn('O(B^2)', ' '.join(result.violations))

    def test_import_gate_uses_locked_model_roots_but_keeps_dangerous_roots_denied(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = os.path.join(tmp, 'env.lock.json')
            with open(lock_path, 'w', encoding='utf-8') as fh:
                json.dump({'import_roots': [
                    'lightgbm', 'torch', 'httpx', 'pip', 'not-a-module',
                ]}, fh)
            allowed = gates.import_allow(lock_path)
        self.assertIn('lightgbm', allowed)
        self.assertIn('torch', allowed)
        self.assertNotIn('httpx', allowed)
        self.assertNotIn('pip', allowed)

    def test_pairwise_patch_prompt_pins_gradient_direction(self):
        item = proposal('optimizer', 'same-user BPR', 'loss')
        prompt = core.patch_prompt('', item)
        self.assertIn('g_pos = (sigmoid(diff) - 1) / pair_count', prompt)
        self.assertIn('g_pos` must be\nnegative', prompt)
        self.assertIn('full train view before batching', prompt)

    def test_fidelity_prompt_allows_equivalent_legal_context_access(self):
        item = proposal('reward', 'tab calibration', 'predict')
        prompt = core.implementation_audit_prompt(
            'parent', 'candidate', item,
            operators.backend_context(['legal_rank_stack_v1']))
        self.assertIn('DV.load(split).tab', prompt)
        self.assertIn('Semantically\nequivalent legal data paths', prompt)

    def test_same_user_bpr_proposal_requires_full_train_pairing(self):
        item = proposal('optimizer', 'same-user BPR improves ranking', 'loss')
        item['mechanism'] = 'same-user BPR'
        item['implementation_plan'] = (
            'Pair within-batch rows with g_pos=(sigmoid(diff)-1)/pair_count.')
        with self.assertRaisesRegex(ValueError, 'loss 和 train'):
            agent._validate_candidate_research(
                item, {'E001': 'external_research'}, {'E001': 'optimizer'})

        item['patch_scope'] = ['loss', 'train']
        with self.assertRaisesRegex(ValueError, 'mini-batch'):
            agent._validate_candidate_research(
                item, {'E001': 'external_research'}, {'E001': 'optimizer'})

        item['implementation_plan'] = (
            'Build full train per-user pools before batching; use '
            'g_pos=(sigmoid(diff)-1)/pair_count and g_neg=-g_pos.')
        self.assertIs(
            agent._validate_candidate_research(
                item, {'E001': 'external_research'}, {'E001': 'optimizer'}),
            item)

    def test_deepfm_delegate_rejects_incompatible_objective_switch(self):
        item = proposal('optimizer', 'Switch DeepFM to BPR', 'train')
        item['mechanism'] = 'bpr objective through DeepFM backend'
        item['implementation_plan'] = (
            "Pass objective='bpr' to SO.train so the delegated backend consumes it.")
        item['patch_scope'] = ['loss', 'train']
        item['experiment_contract']['required_components'] = [
            'delegated BPR objective']
        got = agent._validate_candidate_research(
            item, {'E001': 'external_research'}, {'E001': 'optimizer'},
            parent_operator_stack=['deepfm_engagement_mtl_v1'],
            allow_planning_blockers=True)
        self.assertEqual(
            got['_planning_blocker']['kind'], 'incompatible_delegated_objective')

    def test_deepfm_native_objective_is_not_misread_as_plain_pointwise(self):
        item = proposal('optimizer', 'Retune native DeepFM MTL', 'train')
        item['mechanism'] = 'preserve pointwise_engagement_mtl through DeepFM backend'
        item['implementation_plan'] = (
            'Keep the pointwise_engagement_mtl objective consumed by SO.train and change only '
            'the explicitly declared optimizer schedule.')
        item['experiment_contract']['required_components'] = [
            'native pointwise_engagement_mtl objective']
        got = agent._validate_candidate_research(
            item, {'E001': 'external_research'}, {'E001': 'optimizer'},
            parent_operator_stack=['deepfm_engagement_mtl_v1'],
            allow_planning_blockers=True)
        self.assertNotIn('_planning_blocker', got)

    def test_invalid_bpr_draft_blocks_only_that_candidate(self):
        optimizer = proposal('optimizer', 'same-user BPR improves ranking', 'loss')
        optimizer['mechanism'] = 'same-user BPR'
        optimizer['implementation_plan'] = 'Change only the loss function.'
        architecture = proposal('architecture', 'add legal interactions', 'features')
        reward = proposal('reward', 'adjust legal auxiliary target', 'target')
        draft = {'candidates': [optimizer, architecture, reward],
                 'research_nonuse_reason': None}
        evidence_kinds = {
            'E001': 'external_research',
            'E002': 'external_research',
            'E003': 'external_research',
        }
        validated = agent.validate_proposal_set(
            draft, evidence_kinds,
            {'E001': 'optimizer', 'E002': 'architecture', 'E003': 'reward'})
        self.assertEqual(validated['candidates'][0]['_planning_blocker']['kind'],
                         'invalid_custom_bpr_scope')
        screened, traces, blocker = agent.screen_candidate_memory(
            None, '', {}, validated['candidates'][0], [], {}, evidence_kinds, [])
        self.assertIs(screened, validated['candidates'][0])
        self.assertEqual(blocker['kind'], 'invalid_custom_bpr_scope')
        self.assertEqual(traces[0]['action'], 'reject')
        self.assertNotIn('_planning_blocker', validated['candidates'][1])
        self.assertNotIn('_planning_blocker', validated['candidates'][2])

    def test_measured_correct_bpr_is_exhausted_without_new_knowledge(self):
        item = proposal('optimizer', 'same-user BPR improves ranking', 'loss')
        item.update(
            mechanism='same-user sampled BPR',
            patch_scope=['loss', 'train'],
            implementation_plan=(
                'Build full train per-user pools before batching; use '
                'g_pos=(sigmoid(diff)-1)/pair_count and g_neg=-g_pos.'),
            research_evidence=[{'knowledge_id': 'K0001'}])
        match = {
            'memory_id': 'm001',
            'failure_class': 'statistical_inconclusive',
            'mechanism': 'same-user sampled BPR',
            'mechanism_similarity': 1.0,
            'patch_scope': ['loss', 'train'],
            'implementation_plan': item['implementation_plan'],
            'research_knowledge_ids': ['K0001'],
        }
        blocked = agent.deterministic_mechanism_exhaustion(item, [match])
        self.assertEqual(blocked['kind'], 'mechanism_exhausted')
        item['research_evidence'].append({'knowledge_id': 'K0007'})
        self.assertIsNone(agent.deterministic_mechanism_exhaustion(item, [match]))

    def test_bpr_exhaustion_uses_typed_family_not_lexical_similarity(self):
        item = proposal('optimizer', 'BPR correctness hardening', 'train')
        item.update(
            mechanism='BPR gradient-sign sanity hardening for FM',
            patch_scope=['train', 'loss'],
            implementation_plan=(
                'Build same-user positive and negative pools over all train rows before batching; '
                'run a sanity step with g_pos=(sigmoid(diff)-1)/pair_count.'),
            research_evidence=[{'knowledge_id': 'K0001'}])
        differently_worded_match = {
            'memory_id': 'm001',
            'failure_class': 'statistical_inconclusive',
            'mechanism': 'same-user BPR FM training',
            # Deliberately below the old threshold; this mirrors the observed formal run.
            'mechanism_similarity': 0.31,
            'patch_scope': ['loss', 'train'],
            'implementation_plan': (
                'Precompute full train per-user blocks before batching; use '
                'g_pos=(sigmoid(diff)-1)/pair_count and g_neg=-g_pos.'),
            'research_knowledge_ids': ['K0001'],
        }
        blocked = agent.deterministic_mechanism_exhaustion(
            item, [differently_worded_match])
        self.assertEqual(blocked['kind'], 'mechanism_exhausted')
        self.assertEqual(blocked['memory_id'], 'm001')

    def test_only_cited_research_enters_patch_context(self):
        evidence = [{'evidence_id': 'E001'}, {'evidence_id': 'E002'}]
        item = proposal('optimizer', 'use ranking', 'loss')
        item['evidence'] = [{'type': 'external_research', 'ref': 'E002 ranking'}]
        self.assertEqual(external_research.cited_evidence(item, evidence),
                         [{'evidence_id': 'E002'}])

    def test_research_query_plan_requires_one_query_per_persona(self):
        plan = {'queries': [
            {'persona': 'optimizer', 'query': 'ranking loss numpy', 'reason': 'loss gap'},
            {'persona': 'architecture', 'query': 'feature crossing recsys',
             'reason': 'interaction gap'},
            {'persona': 'reward', 'query': 'watch time multitask recsys',
             'reason': 'signal gap'},
        ]}
        self.assertIs(external_research.validate_query_plan(plan), plan)
        plan['queries'][2]['persona'] = 'optimizer'
        with self.assertRaises(ValueError):
            external_research.validate_query_plan(plan)

    def test_research_query_plan_rejects_broad_overlong_query(self):
        plan = {'queries': [
            {'persona': 'optimizer', 'query': 'x' * 221, 'reason': 'loss gap'},
            {'persona': 'architecture', 'query': 'feature crossing recsys',
             'reason': 'interaction gap'},
            {'persona': 'reward', 'query': 'watch time multitask recsys',
             'reason': 'signal gap'},
        ]}
        with self.assertRaisesRegex(ValueError, '220 characters'):
            external_research.validate_query_plan(plan)

    def test_first_research_iteration_bootstraps_exact_dataset_priors(self):
        prompt = external_research.query_plan_prompt('', {}, '{}', '', '', iteration=1)
        self.assertIn('bootstrap iteration', prompt)
        self.assertIn('KuaiRand-Pure', prompt)
        self.assertIn('dataset variant, target, date split, candidate set, metrics', prompt)
        later = external_research.query_plan_prompt('', {}, '{}', '', '', iteration=2)
        self.assertNotIn('bootstrap iteration', later)

    def test_query_plan_receives_cached_prior_without_treating_it_as_instruction(self):
        prompt = external_research.query_plan_prompt(
            '', {}, '{}', '', '', iteration=2,
            prior_text='[K0001] persona=optimizer; summary=BPR already acquired')
        self.assertIn('K0001', prompt)
        self.assertIn('avoid duplicating already acquired knowledge', prompt)

    def test_unbounded_search_still_uses_run_deadline(self):
        with mock.patch.object(external_research.time, 'monotonic', return_value=100.0):
            self.assertEqual(
                external_research._bounded_research_timeout(None, 250.0), 150.0)
            self.assertIsNone(external_research._bounded_research_timeout(None, None))

    def test_live_normalization_keeps_raw_sources_and_assigns_evidence_ids(self):
        searches = [{
            'search_id': 'Q01', 'persona': 'optimizer', 'query': 'query',
            'reason': 'reason', 'status': 'complete', 'final_output': 'memo',
            'raw_responses': [{'response_id': 'resp_1'}],
            'source_candidates': [{'url': 'https://arxiv.org/abs/1', 'title': 'Paper'}],
        }]
        sources, evidence = external_research._normalize_live(searches)
        self.assertEqual(sources[0]['source_id'], 'S001')
        self.assertEqual(evidence[0]['evidence_id'], 'E001')
        self.assertEqual(evidence[0]['source_ids'], ['S001'])
        self.assertEqual(searches[0]['raw_responses'][0]['response_id'], 'resp_1')

    def test_incomplete_live_search_never_becomes_evidence(self):
        searches = [{
            'search_id': 'Q01', 'persona': 'optimizer', 'query': 'query',
            'reason': 'reason', 'status': 'incomplete', 'final_output': '',
            'raw_responses': [{'response_id': 'resp_1'}],
            'source_candidates': [{'url': 'https://arxiv.org/abs/1', 'title': 'Paper'}],
        }]
        sources, evidence = external_research._normalize_live(searches)
        self.assertEqual(len(sources), 1)
        self.assertEqual(evidence, [])
        self.assertEqual(searches[0]['status'], 'incomplete')

    def test_live_coverage_identifies_and_retries_missing_personas(self):
        evidence = [{'kind': 'external_research', 'persona': 'optimizer'}]
        self.assertEqual(
            external_research._missing_personas(evidence), ['architecture', 'reward'])
        plan = {'queries': [
            {'persona': persona, 'query': f'{persona} query', 'reason': 'gap'}
            for persona in external_research.PERSONAS
        ]}
        retries = external_research._retry_queries(
            plan, external_research._missing_personas(evidence))
        self.assertEqual([item['persona'] for item in retries],
                         ['architecture', 'reward'])

    def test_offline_is_explicit_and_replay_is_hash_pinned(self):
        card = {'id': 'M99', 'title': 'control method', 'body': 'mechanism',
                'source': 'paper citation', 'blocks': ['loss']}
        with tempfile.TemporaryDirectory() as tmp:
            original_dir = os.path.join(tmp, 'original', 'iter-001')
            replay_dir = os.path.join(tmp, 'replay', 'iter-001')
            os.makedirs(original_dir)
            os.makedirs(replay_dir)
            original = external_research.acquire(
                None, '', {}, '', '', '', original_dir, [], mode='offline',
                offline_cards=[card])
            replayed = external_research.acquire(
                None, '', {}, '', '', '', replay_dir, [], mode='replay',
                replay_path=os.path.join(tmp, 'original'), iteration=1)
        self.assertEqual(original['mode'], 'offline')
        self.assertEqual(original['evidence'][0]['evidence_id'], 'E001')
        self.assertEqual(replayed['mode'], 'replay')
        self.assertEqual(len(replayed['replayed_from']['sha256']), 64)

    def test_prior_store_seeds_retrieves_and_merges_with_provenance(self):
        def evidence(evidence_id, persona, source_id, summary):
            return {
                'evidence_id': evidence_id, 'kind': 'external_research',
                'persona': persona, 'query': f'{persona} query', 'reason': 'gap',
                'summary': summary, 'source_ids': [source_id],
            }

        seed = {
            'schema_version': 'external-research-1.1', 'mode': 'live',
            'status': 'complete', 'stage': 'pre_draft', 'query_plan': {},
            'review': {}, 'searches': [],
            'sources': [
                {'source_id': f'S{i:03d}', 'kind': 'url',
                 'url': f'https://seed.test/{persona}', 'title': persona}
                for i, persona in enumerate(prior_store.PERSONAS, start=1)],
            'evidence': [
                evidence(f'E{i:03d}', persona, f'S{i:03d}', f'{persona} mechanism')
                for i, persona in enumerate(prior_store.PERSONAS, start=1)],
        }
        live = {
            'schema_version': 'external-research-1.1', 'mode': 'live',
            'status': 'complete', 'stage': 'pre_draft', 'query_plan': {},
            'review': {},
            'searches': [{'search_id': 'Q01', 'source_ids': ['S001']}],
            'sources': [{'source_id': 'S001', 'kind': 'url',
                         'url': 'https://live.test/new', 'title': 'new'}],
            'evidence': [evidence('E001', 'optimizer', 'S001', 'new optimizer method')],
        }
        with tempfile.TemporaryDirectory() as tmp:
            seed_path = os.path.join(tmp, 'seed.json')
            store_path = os.path.join(tmp, 'prior', 'store.json')
            live_path = os.path.join(tmp, 'live.json')
            import json
            with open(seed_path, 'w', encoding='utf-8') as fh:
                json.dump(seed, fh)
            with open(live_path, 'w', encoding='utf-8') as fh:
                json.dump(live, fh)
            prior_store.initialize(store_path, seed_path)
            selected = prior_store.retrieve(store_path, 'optimizer architecture reward')
            self.assertEqual(len(selected), 3)
            seeded = prior_store.seed_bundle(selected, store_path)
            self.assertFalse(seeded['live_search_performed'])
            known = {item['knowledge_id'] for item in prior_store.entries(store_path)}
            mapping = prior_store.ingest_bundle(store_path, live, live_path)
            merged = prior_store.merge_live_bundle(
                selected[1:], live, mapping, known_knowledge_ids=known)
            external_research._validate_bundle(merged)
            self.assertEqual(merged['searches'][0]['source_ids'], ['S003'])
            self.assertEqual(merged['evidence'][-1]['knowledge_origin'], 'live')
            self.assertEqual(merged['new_knowledge_ids'], [mapping['E001']])

            known_after = {item['knowledge_id'] for item in prior_store.entries(store_path)}
            refreshed_cache = prior_store.retrieve(
                store_path, 'new optimizer method', per_persona=1)
            mapping_again = prior_store.ingest_bundle(store_path, live, live_path)
            refreshed = prior_store.merge_live_bundle(
                refreshed_cache, live, mapping_again,
                known_knowledge_ids=known_after)
            knowledge_ids = [item['knowledge_id'] for item in refreshed['evidence']]
            self.assertEqual(len(knowledge_ids), len(set(knowledge_ids)))
            self.assertEqual(refreshed['new_knowledge_ids'], [])
            self.assertEqual(refreshed['refreshed_knowledge_ids'], [mapping['E001']])
            used = {selected[0]['knowledge_id']}
            unused = prior_store.retrieve(store_path, 'optimizer', used_knowledge_ids=used)
            self.assertNotIn(selected[0]['knowledge_id'],
                             {item['knowledge_id'] for item in unused})

    def test_empirical_prior_is_curated_validation_only_and_auditable(self):
        import json

        sources = [
            {'source_id': f'S{i:03d}', 'kind': 'local_code',
             'citation': f'local-valid-result-{i}'}
            for i in range(1, 4)]
        empirical = {
            'schema_version': 'external-research-1.1', 'mode': 'curated',
            'status': 'complete', 'stage': 'pre_draft',
            'provenance': {
                'pre_run_curated_prior': True,
                'counts_as_runtime_intervention': False,
                'heldout_metrics_excluded': True,
                'leaky_configs_excluded_from_positive_prior': True,
                'selection_data': 'official_validation_and_random_exposure_validation_only',
            },
            'sources': sources,
            'operator_portfolio': {
                'portfolio_id': 'validated_rank_portfolio_v1',
                'evidence_id': 'E001',
                'activation': 'verify_before_llm_search',
                'operators': ['legal_rank_stack_v1', 'item_lambdarank'],
                'weights': [0.7, 0.3],
                'validation_selection_primary': 0.61,
                'counts_as_one_logical_iteration': True,
                'counts_as_measured_subexperiments': True,
            },
            'evidence': [{
                'evidence_id': f'E{i:03d}', 'kind': 'curated_research',
                'persona': persona, 'query': f'{persona} mechanism',
                'reason': 'manual validation result',
                'summary': f'{persona} validated implementation detail',
                'source_ids': [f'S{i:03d}'],
            } for i, persona in enumerate(prior_store.PERSONAS, start=1)],
        }
        with tempfile.TemporaryDirectory() as tmp:
            empirical_path = os.path.join(tmp, 'empirical.json')
            store_path = os.path.join(tmp, 'prior', 'store.json')
            with open(empirical_path, 'w', encoding='utf-8') as fh:
                json.dump(empirical, fh)
            store = prior_store.initialize(
                store_path, empirical_snapshot=empirical_path)
            self.assertFalse(store['policy']['counts_as_runtime_intervention'])
            self.assertTrue(store['policy']['human_curated_prior'])
            self.assertEqual(len(store['empirical_snapshots']), 1)
            selected = prior_store.retrieve(
                store_path, 'optimizer architecture reward implementation')
            bundle = prior_store.cache_bundle(selected, store_path)
            external_research._validate_bundle(bundle)
            self.assertEqual(
                {item['kind'] for item in bundle['evidence']}, {'curated_research'})
            self.assertFalse(bundle['live_search_performed'])
            portfolio = prior_store.load_empirical_portfolio(empirical_path)
            self.assertEqual(
                portfolio['operators'],
                ['legal_rank_stack_v1', 'item_lambdarank'])
            self.assertTrue(portfolio['counts_as_one_logical_iteration'])
            self.assertTrue(portfolio['counts_as_measured_subexperiments'])
            self.assertNotIn('counts_as_measured_iterations', portfolio)

            empirical['evidence'][0]['test_primary'] = 0.9
            with open(empirical_path, 'w', encoding='utf-8') as fh:
                json.dump(empirical, fh)
            with self.assertRaisesRegex(ValueError, 'test-prefixed'):
                prior_store.initialize(
                    os.path.join(tmp, 'bad', 'store.json'),
                    empirical_snapshot=empirical_path)

    def test_locked_prior_surfaces_exact_post_warmstart_negatives(self):
        empirical_path = os.path.join(
            ROOT, 'empirical_priors', 'track2-solutions-valid-only-v1.json')
        with tempfile.TemporaryDirectory() as tmp:
            store_path = os.path.join(tmp, 'store.json')
            prior_store.initialize(store_path, empirical_snapshot=empirical_path)
            selected = prior_store.retrieve(
                store_path,
                'legal_rank_stack_v1 selected_user_profile aux_weight 0.1 0.2 stable_ops',
                per_persona=2)
        negative_queries = {
            item['query'] for item in selected
            if item.get('record_type') == 'negative_result'}
        self.assertTrue(any('selected_user_profile' in query
                            for query in negative_queries))
        self.assertTrue(any('aux_weight' in query for query in negative_queries))
        self.assertTrue(any('optimizer-side proposal' in query
                            for query in negative_queries))

    def test_empty_prior_store_and_used_evidence_remains_retrievable(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_path = os.path.join(tmp, 'prior', 'store.json')
            store = prior_store.initialize(store_path)
            self.assertEqual(store['entries'], [])

            seeded = prior_store._new_store()
            seeded['entries'] = [{
                'knowledge_id': 'K0001', 'content_sha256': 'a' * 64,
                'kind': 'agent_cached_prior', 'persona': 'optimizer',
                'query': 'BPR ranking', 'reason': 'pairwise loss',
                'summary': 'vectorized BPR pairwise ranking loss',
                'sources': [{'url': 'https://example.test/bpr'}], 'origins': [],
            }]
            journal_path = store_path
            import journal
            journal.write_json(journal_path, seeded)
            selected = prior_store.retrieve(
                store_path, 'BPR pairwise ranking', used_knowledge_ids=['K0001'])
            self.assertEqual([item['knowledge_id'] for item in selected], ['K0001'])
            self.assertTrue(selected[0]['retrieval']['previously_used'])
            self.assertEqual(selected[0]['retrieval']['use_count'], 1)

    def test_gap_gate_allows_cache_reuse_and_bounds_live_queries(self):
        cached = [
            {'knowledge_id': 'K0001', 'persona': 'optimizer'},
            {'knowledge_id': 'K0002', 'persona': 'architecture'},
            {'knowledge_id': 'K0003', 'persona': 'reward'},
        ]
        decision = {'coverage': [
            {'persona': 'optimizer', 'decision': 'use_cache',
             'knowledge_ids': ['K0003'], 'gap_type': 'none',
             'gap': None, 'query': None, 'priority': 0},
            {'persona': 'architecture', 'decision': 'web_search',
             'knowledge_ids': [], 'gap_type': 'implementation_detail',
             'gap': 'need a sparse implementation', 'query': 'sparse FM implementation',
             'priority': 2},
            {'persona': 'reward', 'decision': 'web_search',
             'knowledge_ids': ['K0003'], 'gap_type': 'conflicting_evidence',
             'gap': 'resolve censoring conflict', 'query': 'watch time censoring regression',
             'priority': 3},
        ]}
        self.assertIs(external_research.gap_plan_validator(cached)(decision), decision)
        queries, suppressed = external_research.select_gap_queries(decision, 1)
        self.assertEqual(queries[0]['persona'], 'reward')
        self.assertEqual(suppressed, ['architecture'])

        invalid = {'coverage': [dict(item) for item in decision['coverage']]}
        invalid['coverage'][0]['knowledge_ids'] = ['K9999']
        with self.assertRaises(ValueError):
            external_research.gap_plan_validator(cached)(invalid)

    def test_cache_bundle_performs_no_web_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_path = os.path.join(tmp, 'prior', 'store.json')
            prior_store.initialize(store_path)
            cached = [{
                'knowledge_id': 'K0001', 'persona': 'optimizer',
                'query': 'query', 'reason': 'reason', 'summary': 'method summary',
                'sources': [{'url': 'https://example.test/method'}],
            }]
            with mock.patch.object(external_research, '_run_live_searches') as web:
                bundle = prior_store.cache_bundle(cached, store_path, gap_decision={})
            web.assert_not_called()
            external_research._validate_bundle(bundle)
            self.assertFalse(bundle['live_search_performed'])
            self.assertEqual(bundle['executed_live_query_count'], 0)

    def test_selective_acquire_searches_only_planned_gap_without_reviewer(self):
        planned = [{
            'persona': 'architecture', 'query': 'sparse field interaction implementation',
            'reason': 'cached evidence lacks implementation detail',
            'gap_type': 'implementation_detail', 'priority': 3,
        }]
        search_result = [{
            'search_id': 'Q01', **planned[0], 'status': 'complete',
            'latency_s': 0.1, 'final_output': 'mechanism https://example.test/code',
            'raw_responses': [],
            'source_candidates': [{'url': 'https://example.test/code', 'title': 'code'}],
            'usage': {'phase': 'web_search', 'total_tokens': 10},
        }]
        with tempfile.TemporaryDirectory() as tmp:
            usages = []
            with mock.patch.object(
                    external_research, '_run_live_searches',
                    return_value=search_result) as web:
                with mock.patch.object(core, 'ask_validated') as ask:
                    bundle = external_research.acquire(
                        object(), '', {}, '', '', '', tmp, usages, mode='live',
                        planned_queries=planned, gap_decision={'policy': 'prior_first'},
                        max_followups=0)
            web.assert_called_once()
            ask.assert_not_called()
            self.assertEqual(bundle['planned_live_query_count'], 1)
            self.assertEqual(bundle['executed_live_query_count'], 1)
            self.assertEqual(bundle['evidence'][0]['persona'], 'architecture')
            self.assertEqual(bundle['review']['policy'], 'prior_first_no_followup')

    def test_timeout_recovery_requires_asymptotic_change(self):
        guidance = core.recovery_guidance({
            'kind': 'smoke',
            'gate': {'info': {'error_class': 'TIMEOUT'}},
        })
        self.assertIn('O(B^2)', guidance)
        self.assertIn('bounded sampling', guidance)

    def test_operator_proposal_materializes_without_llm_patch(self):
        item = proposal('optimizer', 'same-user BPR improves ranking', 'train')
        item.update(
            execution_mode='operator', operator_id='same_user_bpr',
            mechanism='same-user BPR', patch_scope=['loss', 'train'],
            implementation_plan='Use the trusted full-train same-user pair sampler.')
        self.assertIs(agent._validate_candidate_research(
            item, {'E001': 'external_research'}, {'E001': 'optimizer'}, []), item)

        with open(os.path.join(ROOT, 'candidate', 'pipeline.py'), encoding='utf-8') as fh:
            source = fh.read()
        materialized = operators.materialize(source, [], 'same_user_bpr')
        self.assertEqual(materialized['logical_scope'], ['loss', 'train'])
        self.assertEqual(set(materialized['materialized_scope']), set(schemas.BLOCKS))
        self.assertIn('"objective": "bpr"', materialized['source'])
        with tempfile.TemporaryDirectory() as tmp:
            parent_path = os.path.join(tmp, 'parent.py')
            candidate_path = os.path.join(tmp, 'candidate.py')
            with open(parent_path, 'w', encoding='utf-8') as fh:
                fh.write(source)
            with open(candidate_path, 'w', encoding='utf-8') as fh:
                fh.write(materialized['source'])
            self.assertTrue(gates.g3_code(
                candidate_path, materialized['materialized_scope'],
                parent_path=parent_path, primary_block='train',
                max_patch_blocks=len(schemas.BLOCKS)).ok)
            self.assertTrue(gates.g2_lineage(candidate_path).ok)

    def test_operator_composition_changes_only_adapter_configuration(self):
        with open(os.path.join(ROOT, 'candidate', 'pipeline.py'), encoding='utf-8') as fh:
            source = fh.read()
        first = operators.materialize(source, [], 'same_user_bpr')
        second = operators.materialize(
            first['source'], first['operator_stack'], 'legal_temporal_context')
        self.assertEqual(second['operator_stack'],
                         ['same_user_bpr', 'legal_temporal_context'])
        self.assertEqual(second['logical_scope'], ['features'])
        self.assertEqual(second['materialized_scope'], ['data_view'])
        self.assertIn('hour', second['config']['features'])
        self.assertIn('user_gap', second['config']['features'])

    def test_compound_legal_rank_stack_is_root_only_and_fully_configured(self):
        with open(os.path.join(ROOT, 'candidate', 'pipeline.py'), encoding='utf-8') as fh:
            source = fh.read()
        catalog = {item['operator_id']: item for item in operators.catalog([])}
        self.assertEqual(
            catalog['legal_rank_stack_v1']['search_priority'],
            'first_round_exploit')
        self.assertEqual(
            catalog['legal_rank_stack_v1']['compound_components'],
            ['same_user_bpr', 'censored_watch_time',
             'legal_temporal_context', 'tuned_k32'])

        materialized = operators.materialize(source, [], 'legal_rank_stack_v1')
        self.assertEqual(materialized['operator_stack'], ['legal_rank_stack_v1'])
        self.assertEqual(materialized['config']['objective'], 'bpr_censored_watch')
        self.assertEqual(materialized['config']['features'], ['hour', 'user_gap'])
        self.assertEqual(
            {key: materialized['config']['hp'][key]
             for key in ('k', 'lr', 'n_neg', 'epochs', 'aux_weight')},
            {'k': 32, 'lr': 0.0005, 'n_neg': 4,
             'epochs': 3, 'aux_weight': 0.1})
        self.assertEqual(set(materialized['materialized_scope']), set(schemas.BLOCKS))
        self.assertIn('selected_user_profile', operators.applicable(
            ['legal_rank_stack_v1']))
        self.assertNotIn('same_user_bpr', operators.applicable(
            ['legal_rank_stack_v1']))
        self.assertNotIn('legal_rank_stack_v1', operators.applicable(
            ['same_user_bpr']))

    def test_standalone_lambdarank_operator_is_library_gated_and_not_composable(self):
        with mock.patch.object(gates, 'locked_import_roots', return_value=set()):
            self.assertNotIn('item_lambdarank', operators.applicable([]))
        with mock.patch.object(
                gates, 'locked_import_roots', return_value={'lightgbm'}):
            self.assertIn('item_lambdarank', operators.applicable([]))
            config = operators.config_for(['item_lambdarank'])
            self.assertEqual(config['model_family'], 'lightgbm_rank')
            self.assertEqual(operators.applicable(['item_lambdarank']), [])
            with self.assertRaisesRegex(ValueError, 'standalone'):
                operators.config_for(['same_user_bpr', 'item_lambdarank'])

    def test_deepfm_engagement_mtl_is_cpu_compound_root_and_library_gated(self):
        with mock.patch.object(gates, 'locked_import_roots', return_value=set()):
            self.assertNotIn('deepfm_engagement_mtl_v1', operators.applicable([]))
        with mock.patch.object(gates, 'locked_import_roots', return_value={'torch'}):
            catalog = {item['operator_id']: item for item in operators.catalog([])}
            item = catalog['deepfm_engagement_mtl_v1']
            self.assertEqual(item['search_priority'], 'first_round_exploit')
            self.assertEqual(
                item['compound_components'], ['deepfm_cpu', 'engagement_aux_heads_4'])
            config = operators.config_for(['deepfm_engagement_mtl_v1'])
            self.assertEqual(config['model_family'], 'torch_deepfm_mtl')
            self.assertEqual(config['objective'], 'pointwise_engagement_mtl')
            self.assertEqual(config['hp']['epochs'], 12)
            self.assertEqual(config['hp']['hidden'], [128, 64])
            self.assertEqual(config['hp']['l2'], 0.0001)
            self.assertEqual(config['hp']['torch_threads'], 1)
            refinements = operators.applicable(['deepfm_engagement_mtl_v1'])
            self.assertEqual(
                refinements, ['legal_temporal_context', 'selected_user_profile'])
            temporal = operators.config_for([
                'deepfm_engagement_mtl_v1', 'legal_temporal_context'])
            self.assertEqual(temporal['features'], ['hour', 'user_gap'])
            profiled = operators.config_for([
                'deepfm_engagement_mtl_v1', 'selected_user_profile'])
            self.assertEqual(profiled['features'], ['user_core'])
            context = operators.backend_context(['deepfm_engagement_mtl_v1'])
            self.assertIn('two-dimensional NumPy array',
                          context['feature_api']['return'])
            self.assertEqual(
                context['applicable_trusted_refinements'], refinements)
            self.assertIn('user_gap',
                          context['feature_api']['supported_config_keys']['features'])
            self.assertEqual(
                context['target_api']['supported_objectives_for_this_family'],
                ['pointwise_engagement_mtl'])
            deep_catalog = {
                item['operator_id']: item
                for item in operators.catalog(['deepfm_engagement_mtl_v1'])}
            self.assertEqual(
                deep_catalog['legal_temporal_context']['search_priority'],
                'first_round_exploit')
            with self.assertRaisesRegex(ValueError, 'standalone'):
                operators.config_for(['same_user_bpr', 'deepfm_engagement_mtl_v1'])
            with self.assertRaisesRegex(ValueError, 'supported refinement'):
                operators.config_for([
                    'deepfm_engagement_mtl_v1', 'same_user_bpr'])

    def test_lambdarank_backend_contract_pins_untested_training_parameters(self):
        context = operators.backend_context(['item_lambdarank'])
        fixed = context['train_api']['fixed_training_parameters']
        self.assertEqual(fixed['learning_rate'], 0.05)
        self.assertEqual(fixed['num_leaves'], 63)
        self.assertEqual(fixed['min_data_in_leaf'], 50)
        self.assertEqual(fixed['feature_fraction'], 0.85)
        self.assertEqual(fixed['bagging_fraction'], 0.8)
        self.assertEqual(fixed['lambda_l2'], 1.0)
        prompt = core.implementation_audit_prompt(
            'parent', 'candidate', proposal('optimizer', 'weighted LambdaRank', 'train'),
            context)
        self.assertIn('every listed fixed parameter', prompt)
        self.assertIn('"learning_rate": 0.05', prompt)

    def test_engagement_auxiliary_targets_are_train_only_and_ordered(self):
        train = types.SimpleNamespace(
            label=np.array([0, 1, 0], dtype=np.float32))
        values = {
            name: np.array([index % 2, (index + 1) % 2, 1], dtype=np.float32)
            for index, name in enumerate(stable_ops.ENGAGEMENT_AUX_TARGETS)
        }
        config = {'objective': 'pointwise_engagement_mtl'}
        with mock.patch.object(stable_ops.DV, 'train_targets', return_value=values):
            target = stable_ops.build_target(
                {'train': train}, np.array([2, 0], dtype=np.int64), config)
        np.testing.assert_array_equal(target['main'], [0, 0])
        np.testing.assert_array_equal(
            target['engagement_aux'],
            np.column_stack([values[name][[2, 0]]
                             for name in stable_ops.ENGAGEMENT_AUX_TARGETS]))
        self.assertEqual(target['engagement_aux'].shape, (2, 4))

    def test_group_target_statistics_are_leave_one_out_for_training_rows(self):
        splits = {
            'train': types.SimpleNamespace(
                video_id=np.array([10, 10, 20], dtype=np.int64)),
            'valid': types.SimpleNamespace(
                video_id=np.array([10, 30], dtype=np.int64)),
        }
        labels = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        features = stable_ops._group_features(
            splits, np.arange(3), 'video_id', labels,
            global_mean=2.0 / 3.0, prior=1.0)
        np.testing.assert_array_equal(features['train'][0], [1.0, 1.0, 0.0])
        self.assertAlmostEqual(features['train'][1][0], 1.0 / 3.0)
        self.assertAlmostEqual(features['train'][1][1], 5.0 / 6.0)
        self.assertAlmostEqual(features['valid'][1][0], 5.0 / 9.0)
        self.assertAlmostEqual(features['valid'][1][1], 2.0 / 3.0)

    def test_causal_group_statistics_exclude_same_and_future_dates(self):
        values = {
            'train': np.array([10, 10, 10, 20], dtype=np.int64),
            'valid': np.array([10, 20, 30], dtype=np.int64),
        }
        dates = np.array([1, 1, 2, 1], dtype=np.int32)
        labels = np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32)
        features = stable_ops._causal_group_features_from_values(
            values, dates, np.arange(4), labels,
            global_mean=0.75, prior=1.0)
        np.testing.assert_array_equal(features['train'][0], [0.0, 0.0, 2.0, 0.0])
        self.assertAlmostEqual(features['train'][1][0], 0.75)
        self.assertAlmostEqual(features['train'][1][1], 0.75)
        self.assertAlmostEqual(features['train'][1][2], 1.75 / 3.0)
        self.assertAlmostEqual(features['valid'][1][0], 2.75 / 4.0)
        self.assertAlmostEqual(features['valid'][1][1], 1.75 / 2.0)

    def test_same_operator_and_parent_stack_is_deterministically_deduplicated(self):
        first = proposal('optimizer', 'try same-user ranking loss', 'train')
        first.update(
            execution_mode='operator', operator_id='same_user_bpr',
            mechanism='same-user BPR', patch_scope=['loss', 'train'],
            parent_operator_stack=[])
        prior = memory.build_entry(
            1, first,
            {'decision': 'ROLLBACK', 'delta_primary': -0.001,
             'candidate_metrics': {'paired_vs_incumbent': {}}},
            {'next_lesson': 'do not repeat', 'result': 'not_supported',
             'analysis': 'negative'})
        repeated = dict(first)
        repeated['hypothesis'] = 'a differently worded pairwise objective'
        repeated['implementation_plan'] = 'Registry-owned deterministic BPR implementation.'
        selected, traces, blocker = agent.screen_candidate_memory(
            None, '', {}, repeated, [prior], {'evidence': []}, {}, [],
            parent_operator_stack=[])
        self.assertIs(selected, repeated)
        self.assertEqual(blocker['kind'], 'operator_already_measured')
        self.assertTrue(traces[0]['deterministic_override'])

    def test_frontier_runs_independent_root_drafts_then_reselects_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            def artifact(stem, score):
                pipeline = os.path.join(tmp, f'{stem}.py')
                prediction = os.path.join(tmp, f'{stem}.npy')
                metrics = os.path.join(tmp, f'{stem}.json')
                with open(pipeline, 'w', encoding='utf-8') as fh:
                    fh.write(f'# {stem}\n')
                np.save(prediction, np.array([score], dtype=np.float32))
                with open(metrics, 'w', encoding='utf-8') as fh:
                    json.dump({'selection_primary': score}, fh)
                return pipeline, prediction, metrics

            base = artifact('base', 0.60)
            graph = frontier.Frontier(tmp, draft_count=2, exploration=0.0)
            graph.initialize(base[0], [base[1]], base[2],
                             {'selection_primary': 0.60})
            first, first_reason = graph.select_parent(1)
            self.assertEqual(first['node_id'], 'n000')
            self.assertEqual(first_reason['reason'], 'independent_baseline_draft')

            good = artifact('good', 0.61)
            graph.add_node({
                'node_id': 'n001', 'parent_node_id': 'n000',
                'decision': 'UNCERTAIN', 'status': 'COMPLETE',
                'pipeline_path': good[0], 'prediction_paths': [good[1]],
                'metrics_path': good[2], 'selection_primary': 0.61,
                'operator_stack': ['same_user_bpr'], 'mechanism': 'BPR',
            })
            second, _ = graph.select_parent(2)
            self.assertEqual(second['node_id'], 'n000')

            weak = artifact('weak', 0.59)
            graph.add_node({
                'node_id': 'n002', 'parent_node_id': 'n000',
                'decision': 'ROLLBACK', 'status': 'COMPLETE',
                'pipeline_path': weak[0], 'prediction_paths': [weak[1]],
                'metrics_path': weak[2], 'selection_primary': 0.59,
                'operator_stack': None, 'mechanism': 'weak control',
            })
            third, third_reason = graph.select_parent(3)
            self.assertEqual(third['node_id'], 'n001')
            self.assertEqual(third_reason['reason'], 'frontier_score')
            self.assertEqual(graph.best_validation()['node_id'], 'n001')
            self.assertEqual(
                [node['node_id'] for node in ensemble._candidate_nodes(graph, 8)],
                ['n001', 'n000', 'n002'])

            portfolio = {'members': [
                {'node_id': 'n001', 'weight': 0.8},
                {'node_id': 'n002', 'weight': 0.2},
            ]}
            specialist, specialist_reason = graph.select_parent(
                4, portfolio_selection=portfolio)
            self.assertEqual(specialist['node_id'], 'n002')
            self.assertEqual(
                specialist_reason['reason'], 'portfolio_member_rotation')
            self.assertEqual(specialist['decision'], 'ROLLBACK')

    def test_ensemble_pool_protects_low_scoring_portfolio_specialist(self):
        graph = types.SimpleNamespace(nodes={})
        for index, score in enumerate([0.9, 0.8, 0.7, 0.1]):
            node_id = f'n{index:03d}'
            graph.nodes[node_id] = {
                'node_id': node_id, 'status': 'COMPLETE',
                'pipeline_path': f'{node_id}.py',
                'prediction_paths': [f'{node_id}.npy'],
                'selection_primary': score,
                'pipeline_sha256': node_id,
            }
        pool = ensemble._candidate_nodes(
            graph, 3, required_node_ids={'n003'})
        self.assertEqual(
            [node['node_id'] for node in pool], ['n000', 'n001', 'n003'])

    def test_stable_bpr_gradient_increases_pair_margin(self):
        model = stable_ops.FM(dim=4, k=2, lr=0.01, l2=0.0, seed=0)
        positive = np.array([[0, 2]], dtype=np.int32)
        negative = np.array([[1, 3]], dtype=np.int32)
        before = float(model.logits(positive)[0][0]
                       - model.logits(negative)[0][0])
        model.step_bpr(positive, negative)
        after = float(model.logits(positive)[0][0]
                      - model.logits(negative)[0][0])
        self.assertGreater(after, before)

if __name__ == '__main__':
    unittest.main()
