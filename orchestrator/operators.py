"""Trusted operator registry for the mixed operator/custom-patch workflow."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import gates
import patching


BASE_CONFIG = {
    'objective': 'pointwise',
    'features': [],
    'dur_buckets': 10,
    'hp': {
        'k': 16, 'lr': 0.001, 'l2': 1e-6,
        'epochs': 8, 'batch': 8192, 'n_neg': 1, 'aux_weight': 0.2,
    },
}


@dataclass(frozen=True)
class OperatorSpec:
    operator_id: str
    title: str
    description: str
    logical_scope: tuple[str, ...]
    primary_block: str
    requires: tuple[str, ...] = ()
    libraries: tuple[str, ...] = ()
    search_priority: str = 'standard'


SPECS = {
    'legal_rank_stack_v1': OperatorSpec(
        'legal_rank_stack_v1', 'Validated legal ranking stack',
        'Compound first-round recipe: same-user BPR, a train-only censored watch-time '
        'auxiliary task, legal hour/user-gap context, and the robust k=32 schedule. Its '
        'components were separately ablated before registration; use it to test their '
        'cumulative effect within the short convergence window.',
        ('features', 'target', 'model', 'loss', 'train'), 'train',
        search_priority='first_round_exploit'),
    'same_user_bpr': OperatorSpec(
        'same_user_bpr', 'Same-user BPR',
        'Vectorized full-train user pools, one sampled same-user negative per positive, '
        'correct normalized pairwise gradient.',
        ('loss', 'train'), 'train'),
    'censored_watch_time': OperatorSpec(
        'censored_watch_time', 'Censored watch-time auxiliary task',
        'Adds a train-only log watch-time head; completed plays are right-censored and use '
        'a one-sided loss. Shares FM interactions with BPR.',
        ('target', 'model', 'loss', 'train'), 'target',
        requires=('same_user_bpr',)),
    'legal_temporal_context': OperatorSpec(
        'legal_temporal_context', 'Legal temporal context',
        'Adds hour and strictly-earlier exposure-gap buckets without evaluation feedback.',
        ('features',), 'features'),
    'selected_user_profile': OperatorSpec(
        'selected_user_profile', 'Selected static user profile',
        'Adds seven low-cardinality pre-impression user profile fields; FM crosses them with '
        'candidate item and context fields.',
        ('features',), 'features'),
    'tuned_k32': OperatorSpec(
        'tuned_k32', 'Robust BPR capacity schedule',
        'Uses k=32, lr=5e-4 and four negatives under a fixed three-epoch budget. '
        'A 0.1 auxiliary weight keeps ranking dominant. The shorter schedule avoids the '
        'measured late-epoch regression of the denser pair sampler without exposing '
        'validation labels to candidate code.',
        ('train',), 'train', requires=('same_user_bpr',)),
    'item_lambdarank': OperatorSpec(
        'item_lambdarank', 'Item/context LambdaRank',
        'Standalone LightGBM LambdaRank scorer using context plus strictly-prior-date '
        'train-only item, author, tag, music, and auxiliary-action rates. It is intended as '
        'an error-diverse ensemble member, not as an FM stack mutation.',
        ('features', 'target', 'model', 'train'), 'model', libraries=('lightgbm',)),
    'deepfm_engagement_mtl_v1': OperatorSpec(
        'deepfm_engagement_mtl_v1', 'CPU DeepFM + four engagement heads',
        'Standalone compound primitive: a DeepFM main scorer plus train-only '
        'is_like/is_follow/is_comment/is_forward auxiliary heads sharing its deep trunk. '
        'Only the long_view main logit is emitted. Uses the validated fixed 12-epoch CPU '
        'schedule because candidate processes cannot inspect validation labels for checkpointing.',
        ('target', 'model', 'loss', 'train'), 'model', libraries=('torch',),
        search_priority='first_round_exploit'),
}

STANDALONE_MODEL_OPERATORS = {'item_lambdarank', 'deepfm_engagement_mtl_v1'}
POST_STANDALONE_COMPATIBILITY = {
    # These primitives only extend the categorical field list consumed by
    # stable_ops.build_features, so they are valid refinements of DeepFM too.
    'deepfm_engagement_mtl_v1': {
        'legal_temporal_context',
        'selected_user_profile',
    },
    # The LambdaRank branch owns a separate dense feature builder.  Fail closed
    # until a refinement has an explicit implementation for that representation.
    'item_lambdarank': set(),
}
COMPOUND_OPERATOR_COMPONENTS = {
    'legal_rank_stack_v1': (
        'same_user_bpr',
        'censored_watch_time',
        'legal_temporal_context',
        'tuned_k32',
    ),
}
OPERATOR_BUILDING_BLOCKS = {
    **COMPOUND_OPERATOR_COMPONENTS,
    'deepfm_engagement_mtl_v1': ('deepfm_cpu', 'engagement_aux_heads_4'),
}


def _expanded_stack(stack):
    expanded = []
    for operator_id in stack:
        expanded.extend(COMPOUND_OPERATOR_COMPONENTS.get(operator_id, (operator_id,)))
    return expanded


def _libraries_available(spec):
    return set(spec.libraries) <= gates.locked_import_roots()


def _apply_config(config, operator_id):
    out = copy.deepcopy(config)
    if operator_id == 'same_user_bpr':
        out['objective'] = 'bpr'
    elif operator_id == 'censored_watch_time':
        out['objective'] = 'bpr_censored_watch'
    elif operator_id == 'legal_temporal_context':
        for name in ('hour', 'user_gap'):
            if name not in out['features']:
                out['features'].append(name)
    elif operator_id == 'selected_user_profile':
        if 'user_core' not in out['features']:
            out['features'].append('user_core')
    elif operator_id == 'tuned_k32':
        out['hp'].update(
            k=32, lr=0.0005, n_neg=4, epochs=3, aux_weight=0.1)
    elif operator_id == 'item_lambdarank':
        out['model_family'] = 'lightgbm_rank'
        out['hp'].update(num_boost_round=100)
    elif operator_id == 'deepfm_engagement_mtl_v1':
        out['model_family'] = 'torch_deepfm_mtl'
        out['objective'] = 'pointwise_engagement_mtl'
        out['hp'].update(
            k=16, hidden=[128, 64], lr=0.001, l2=0.0001,
            epochs=12, batch=8192, aux_weight=0.2, torch_threads=1,
        )
    else:
        raise KeyError(f'unknown operator: {operator_id}')
    return out


def config_for(stack):
    config = copy.deepcopy(BASE_CONFIG)
    seen = set()
    for operator_id in _expanded_stack(stack):
        if operator_id in seen:
            raise ValueError(f'duplicate operator in stack: {operator_id}')
        spec = SPECS.get(operator_id)
        if spec is None:
            raise ValueError(f'unknown operator in stack: {operator_id}')
        if operator_id in STANDALONE_MODEL_OPERATORS and seen:
            raise ValueError(f'{operator_id} is a standalone model operator')
        standalone = seen & STANDALONE_MODEL_OPERATORS
        if standalone:
            parent = next(iter(standalone))
            if operator_id not in POST_STANDALONE_COMPATIBILITY.get(parent, set()):
                raise ValueError(
                    f'{operator_id} is not a supported refinement after {parent}')
        missing = set(spec.requires) - seen
        if missing:
            raise ValueError(f'{operator_id} requires {sorted(missing)}')
        config = _apply_config(config, operator_id)
        seen.add(operator_id)
    return config


def applicable(stack):
    if stack is None:
        return []
    seen = set(_expanded_stack(stack))
    applicable_ids = []
    for operator_id, spec in SPECS.items():
        components = set(_expanded_stack([operator_id]))
        if (operator_id in stack or components & seen or not _libraries_available(spec)
                or (operator_id in STANDALONE_MODEL_OPERATORS and seen)):
            continue
        try:
            config_for(list(stack) + [operator_id])
        except ValueError:
            continue
        applicable_ids.append(operator_id)
    return applicable_ids


def backend_context(stack):
    """Return the exact trusted adapter contract visible to proposal/patch audits.

    Operator pipelines intentionally delegate tested primitives to ``stable_ops``.  Without this
    contract an LLM sees only the thin adapter and has to guess whether the backend consumes a
    dict, dense matrix, categorical ids, or a configuration key.  The result is either a semantic
    no-op or an over-strict fidelity rejection.  Keep this representation compact and fail closed:
    only keys and shapes explicitly listed here may be treated as supported behavior.
    """
    if not stack:
        return {
            'kind': 'direct_pipeline',
            'delegated_backend': False,
            'guidance': 'All executable dataflow needed for a custom patch is in the pipeline.',
        }
    config = config_for(stack)
    family = config.get('model_family', 'fm')
    context = {
        'kind': 'trusted_stable_ops_adapter',
        'delegated_backend': True,
        'parent_operator_stack': list(stack),
        'exact_parent_config': config,
        'feature_api': {
            'call': 'SO.build_features(splits, train_idx, config)',
            'return': ('(Xs, dimension), where Xs[split] is a two-dimensional NumPy array; '
                       'it is not a dict of x_cat/x_dense tensors'),
            'categorical_path': (
                'for FM/DeepFM, every column is a globally offset categorical id and dimension '
                'is the embedding-table size'),
            'supported_config_keys': {
                'features': {
                    'hour': 'appends hourmin//100 as a categorical field',
                    'user_gap': ('appends a strictly-earlier exposure-gap bucket per split; '
                                 'quantile edges are fit on selected train rows only'),
                    'user_core': 'appends seven static pre-impression user profile fields',
                },
                'dur_buckets': 'number of train-quantile duration buckets',
            },
        },
        'target_api': {
            'call': 'SO.build_target(splits, train_idx, config)',
            'supported_objectives_for_this_family': [
                'pointwise', 'bpr', 'bpr_censored_watch'],
        },
        'train_api': {
            'call': 'SO.train(splits, train_idx, Xs, dimension, target, seed, config, verbose)',
            'consumes': ('Xs[train][train_idx] directly; FM treats the globally offset integer '
                         'columns as sparse categorical fields'),
            'supported_hp_keys': sorted(config.get('hp', {}).keys()),
            'unknown_keys': 'ignored unless the custom patch implements their consumer visibly',
        },
        'prediction_api': {
            'call': 'SO.predict(model, Xs, split)',
            'consumes': 'the same Xs[split] representation returned by build_features',
            'legal_context_access': (
                'The pipeline predict signature has no splits argument. A custom predict block '
                'may call DV.load(split) to read inference-time arrays such as tab; this is '
                'semantically the same field as splits[split].tab and exposes no label.'),
        },
        'family': family,
        'applicable_trusted_refinements': applicable(stack),
        'audit_rule': (
            'The calls and supported keys above are visible code evidence. Any other delegated '
            'behavior remains opaque and must be implemented inside the declared patch scope.'),
    }
    if family == 'lightgbm_rank':
        context['target_api']['supported_objectives_for_this_family'] = ['pointwise']
        context['feature_api'].update({
            'return': ('(Xs, feature_count), where Xs[split] is a float32 dense NumPy matrix '
                       'with 25 fixed context/group-statistic columns'),
            'categorical_path': 'column 5 (tab) is declared categorical to LightGBM',
            'supported_config_keys': {},
        })
        context['train_api'].update({
            'consumes': ('rows sorted by user and date with LambdaRank groups; fixed params plus '
                         'hp.num_boost_round'),
            'supported_hp_keys': ['num_boost_round'],
            'fixed_training_parameters': {
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'eval_at': [5],
                'lambdarank_truncation_level': 10,
                'learning_rate': 0.05,
                'num_leaves': 63,
                'min_data_in_leaf': 50,
                'feature_fraction': 0.85,
                'bagging_fraction': 0.8,
                'bagging_freq': 1,
                'lambda_l2': 1.0,
                'num_threads': 1,
                'force_row_wise': True,
                'seed_fields': [
                    'seed', 'bagging_seed', 'feature_fraction_seed'],
            },
            'grouping': 'stable sort by (user_id, date), group on both user and date changes',
            'fidelity_constraint': (
                'A custom replacement of SO.train must reproduce every fixed parameter and '
                'grouping rule above exactly unless the accepted experiment contract explicitly '
                'names that parameter as the sole tested mechanism.'),
        })
    elif family == 'torch_deepfm_mtl':
        context['train_api']['consumes'] = (
            'Xs[train][train_idx] directly; DeepFM embeds every categorical-id column and uses '
            'Xs[train].shape[1] as field_count')
        context['target_api']['supported_objectives_for_this_family'] = [
            'pointwise_engagement_mtl']
        context['target_api']['fidelity_constraint'] = (
            'The current DeepFM trainer requires target[main] plus target[engagement_aux]. '
            'Changing objective to bpr or bpr_censored_watch is incompatible unless model, loss, '
            'target, and train are all replaced by a visible custom implementation.')
    return context


def catalog(stack):
    """Compact prompt/audit representation for the selected frontier parent."""
    items = []
    for operator_id in applicable(stack):
        priority = SPECS[operator_id].search_priority
        if (list(stack or []) == ['deepfm_engagement_mtl_v1']
                and operator_id == 'legal_temporal_context'):
            priority = 'first_round_exploit'
        items.append({
        'operator_id': operator_id,
        'title': SPECS[operator_id].title,
        'description': SPECS[operator_id].description,
        'primary_block': SPECS[operator_id].primary_block,
        'patch_scope': list(SPECS[operator_id].logical_scope),
        'required_libraries': list(SPECS[operator_id].libraries),
        'search_priority': priority,
        'compound_components': list(OPERATOR_BUILDING_BLOCKS.get(operator_id, ())),
        'implementation': 'trusted_registry_no_llm_patch',
        })
    return items


def validate_proposal(proposal, parent_stack):
    mode = proposal.get('execution_mode')
    operator_id = proposal.get('operator_id')
    if mode == 'custom_patch':
        if operator_id is not None:
            raise ValueError('custom_patch proposal must set operator_id to null')
        return proposal
    if mode != 'operator':
        raise ValueError('execution_mode must be operator|custom_patch')
    if parent_stack is None:
        raise ValueError('selected custom-patch parent is not operator-compatible')
    if operator_id not in applicable(parent_stack):
        raise ValueError(
            f'operator {operator_id!r} is not applicable; choose from {applicable(parent_stack)}')
    spec = SPECS[operator_id]
    if proposal.get('patch_scope') != list(spec.logical_scope):
        raise ValueError(
            f'operator {operator_id} patch_scope must equal {list(spec.logical_scope)}')
    if proposal.get('primary_block') != spec.primary_block:
        raise ValueError(
            f'operator {operator_id} primary_block must equal {spec.primary_block}')
    return proposal


def _adapter_replacements(config):
    rendered = json.dumps(config, ensure_ascii=True, sort_keys=True)
    return {
        'data_view': f'''# Stable operator adapter. Configuration is controller-generated.
import stable_ops as SO
OP_CONFIG = {rendered}


def build_data_view():
    return SO.build_data_view()''',
        'features': '''def build_features(splits, train_idx):
    return SO.build_features(splits, train_idx, OP_CONFIG)''',
        'target': '''def build_target(splits, train_idx):
    return SO.build_target(splits, train_idx, OP_CONFIG)''',
        'model': '''sigmoid = SO.sigmoid
FM = SO.FM''',
        'loss': '''loss_and_step = SO.loss_and_step''',
        'train': '''HP = dict(OP_CONFIG['hp'])


def train(splits, train_idx, Xs, dim, y, seed, verbose=False):
    config = dict(OP_CONFIG)
    config['hp'] = dict(HP)
    return SO.train(splits, train_idx, Xs, dim, y, seed, config, verbose)''',
        'predict': '''def predict(model, Xs, split):
    return SO.predict(model, Xs, split)''',
    }


def is_adapter(source):
    return 'import stable_ops as SO' in source and 'OP_CONFIG = ' in source


def materialize(parent_source, parent_stack, operator_id):
    """Return a deterministic patch plus the new operator stack.

    The first operator replaces all seven mutable blocks with a thin adapter.  Once the
    adapter exists, composition changes only its controller-owned configuration block.
    """
    if operator_id not in applicable(parent_stack):
        raise ValueError(f'operator {operator_id!r} is not applicable to {parent_stack}')
    if parent_stack and not is_adapter(parent_source):
        raise ValueError('operator stack provenance exists but parent is not a stable adapter')
    new_stack = list(parent_stack) + [operator_id]
    replacements = _adapter_replacements(config_for(new_stack))
    if is_adapter(parent_source):
        replacements = {'data_view': replacements['data_view']}
    patch = {
        'replacements': [{'block': block, 'code': code}
                         for block, code in replacements.items()],
        'notes': f'trusted operator {operator_id}; stack={new_stack}',
    }
    candidate = patching.apply_replacements(parent_source, patch['replacements'])
    return {
        'source': candidate,
        'patch': patch,
        'operator_id': operator_id,
        'operator_stack': new_stack,
        'logical_scope': list(SPECS[operator_id].logical_scope),
        'materialized_scope': list(replacements),
        'config': config_for(new_stack),
    }
