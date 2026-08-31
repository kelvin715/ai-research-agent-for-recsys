"""Run-local, provenance-preserving cache for Agent-discovered web research.

The store contains evidence summaries and source metadata, never external training
examples or model weights.  Raw WebSearch responses remain in immutable snapshots;
this module builds a small retrieval layer for later iterations.
"""
import collections
import copy
import hashlib
import json
import math
import os
import shutil

import journal
import memory


SCHEMA_VERSION = 'agent-prior-store-1.0'
PERSONAS = ('optimizer', 'architecture', 'reward')


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _entry_digest(persona, summary, sources, evidence_kind='external_research'):
    payload = {
        'evidence_kind': evidence_kind,
        'persona': persona,
        'summary': ' '.join(summary.split()),
        'sources': sorted(
            source.get('url') or source.get('citation') or '' for source in sources),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _sources_for_evidence(bundle, evidence):
    source_map = {item['source_id']: item for item in bundle.get('sources', [])}
    sources = []
    for source_id in evidence['source_ids']:
        source = copy.deepcopy(source_map[source_id])
        source.pop('source_id', None)
        source.pop('search_ids', None)
        sources.append(source)
    return sources


def _validate_seed(bundle):
    if not isinstance(bundle, dict) or bundle.get('status') != 'complete':
        raise ValueError('prior seed must be a complete research snapshot')
    if bundle.get('mode') not in {'live', 'replay', 'warm_start'}:
        raise ValueError('prior seed must originate from Agent live research')
    evidence = [item for item in bundle.get('evidence', [])
                if item.get('kind') == 'external_research']
    if not evidence:
        raise ValueError('prior seed contains no evidence')
    personas = {item.get('persona') for item in evidence}
    if not set(PERSONAS) <= personas:
        raise ValueError('prior seed must cover optimizer, architecture, and reward')
    return bundle


def _contains_forbidden_test_key(value):
    if isinstance(value, dict):
        return any(str(key).casefold().startswith('test')
                   or _contains_forbidden_test_key(item)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_forbidden_test_key(item) for item in value)
    return False


def _validate_empirical_seed(bundle):
    """Validate a pre-run curated, validation-only empirical prior.

    This is intentionally separate from ``_validate_seed`` so a curated
    result table can never masquerade as Agent-acquired live research.
    """
    if not isinstance(bundle, dict) or bundle.get('status') != 'complete':
        raise ValueError('empirical prior must be a complete research snapshot')
    if bundle.get('mode') != 'curated':
        raise ValueError('empirical prior mode must be curated')
    if _contains_forbidden_test_key(bundle):
        raise ValueError('empirical prior must not contain test-prefixed fields')
    provenance = bundle.get('provenance') or {}
    required = {
        'pre_run_curated_prior': True,
        'counts_as_runtime_intervention': False,
        'heldout_metrics_excluded': True,
        'leaky_configs_excluded_from_positive_prior': True,
        'selection_data': 'official_validation_and_random_exposure_validation_only',
    }
    if any(provenance.get(key) != expected for key, expected in required.items()):
        raise ValueError('empirical prior provenance policy is incomplete')
    evidence = bundle.get('evidence')
    if (not isinstance(evidence, list) or not evidence
            or any(item.get('kind') != 'curated_research' for item in evidence)):
        raise ValueError('empirical prior may contain only curated_research evidence')
    record_types = {'mechanism', 'tuning', 'negative_result', 'guardrail'}
    if any(item.get('record_type', 'mechanism') not in record_types for item in evidence):
        raise ValueError('empirical prior contains an invalid record_type')
    for item in evidence:
        requirements = item.get('requires_success_mechanisms', [])
        if (not isinstance(requirements, list)
                or any(not isinstance(value, str) or not value.strip()
                       for value in requirements)):
            raise ValueError('empirical prior success requirements must be strings')
    personas = {item.get('persona') for item in evidence}
    if not set(PERSONAS) <= personas:
        raise ValueError('empirical prior must cover optimizer, architecture, and reward')
    source_ids = {item.get('source_id') for item in bundle.get('sources', [])}
    if (None in source_ids or any(not item.get('source_ids')
                                  or not set(item['source_ids']) <= source_ids
                                  for item in evidence)):
        raise ValueError('empirical prior has missing source provenance')
    portfolio = bundle.get('operator_portfolio')
    if portfolio is not None:
        if not isinstance(portfolio, dict):
            raise ValueError('empirical operator_portfolio must be an object')
        required_portfolio = {
            'activation': 'verify_before_llm_search',
            'counts_as_one_logical_iteration': True,
            'counts_as_measured_subexperiments': True,
        }
        if any(portfolio.get(key) != value
               for key, value in required_portfolio.items()):
            raise ValueError('empirical operator_portfolio policy is incomplete')
        portfolio_id = portfolio.get('portfolio_id')
        operators = portfolio.get('operators')
        weights = portfolio.get('weights')
        expected_primary = portfolio.get('validation_selection_primary')
        evidence_ids = {item.get('evidence_id') for item in evidence}
        if not isinstance(portfolio_id, str) or not portfolio_id.strip():
            raise ValueError('empirical operator_portfolio requires portfolio_id')
        if (not isinstance(operators, list) or len(operators) < 2
                or len(set(operators)) != len(operators)
                or any(not isinstance(value, str) or not value.strip()
                       for value in operators)):
            raise ValueError('empirical operator_portfolio operators are invalid')
        if (not isinstance(weights, list) or len(weights) != len(operators)
                or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                       or value <= 0 for value in weights)
                or not math.isclose(sum(weights), 1.0)):
            raise ValueError('empirical operator_portfolio weights are invalid')
        if (not isinstance(expected_primary, (int, float))
                or not math.isfinite(expected_primary)):
            raise ValueError(
                'empirical operator_portfolio validation_selection_primary is invalid')
        if portfolio.get('evidence_id') not in evidence_ids:
            raise ValueError('empirical operator_portfolio evidence_id is missing')
        context_router = portfolio.get('context_router')
        if context_router is not None:
            global_primary = portfolio.get('global_validation_selection_primary')
            if (not isinstance(global_primary, (int, float))
                    or not math.isfinite(global_primary)
                    or global_primary >= expected_primary):
                raise ValueError(
                    'empirical context router requires a lower finite global primary')
            if (not isinstance(context_router, dict)
                    or context_router.get('feature') != 'tab'
                    or context_router.get('fallback') != 'global operator weights'
                    or context_router.get('inference_label_free') is not True
                    or context_router.get('min_rows') != 1000
                    or not math.isclose(
                        context_router.get('weight_grid_step', float('nan')), 0.1)):
                raise ValueError('empirical context router policy is invalid')
            routes = context_router.get('routes')
            if not isinstance(routes, list) or not routes:
                raise ValueError('empirical context router requires routes')
            route_values = [route.get('value') for route in routes]
            if (any(not isinstance(value, int) for value in route_values)
                    or len(set(route_values)) != len(route_values)):
                raise ValueError('empirical context router route values are invalid')
            for route in routes:
                route_weights = route.get('operator_weights')
                if (not isinstance(route_weights, dict)
                        or set(route_weights) != set(operators)
                        or any(not isinstance(value, (int, float))
                               or not math.isfinite(value) or value < 0
                               for value in route_weights.values())
                        or not math.isclose(sum(route_weights.values()), 1.0)
                        or not isinstance(route.get('rows'), int)
                        or route['rows'] < context_router['min_rows']):
                    raise ValueError(
                        'empirical context router operator weights are invalid')
            matched_seed_deltas = context_router.get('matched_seed_deltas')
            if (not isinstance(matched_seed_deltas, list)
                    or len(matched_seed_deltas) < 3
                    or any(not isinstance(value, (int, float))
                           or not math.isfinite(value) or value <= 0
                           for value in matched_seed_deltas)
                    or sum(matched_seed_deltas) / len(matched_seed_deltas) < 1.5e-4
                    or context_router.get('promotion_reason')
                    != 'strict_all_seed_robustness'):
                raise ValueError(
                    'empirical context router robustness evidence is invalid')
    return bundle


def load_empirical_portfolio(path):
    """Load the optional structured operator portfolio from a validated snapshot."""
    if not path:
        return None
    with open(path, encoding='utf-8') as handle:
        bundle = json.load(handle)
    _validate_empirical_seed(bundle)
    portfolio = bundle.get('operator_portfolio')
    return copy.deepcopy(portfolio) if portfolio is not None else None


def _new_store():
    return {
        'schema_version': SCHEMA_VERSION,
        'policy': {
            'source': 'agent_cached_prior',
            'human_curated_prior': False,
            'counts_as_runtime_intervention': False,
            'external_training_data': False,
            'expected_validation_delta_stored': False,
        },
        'seed_snapshots': [],
        'empirical_snapshots': [],
        'entries': [],
    }


def _load(path):
    with open(path, encoding='utf-8') as fh:
        store = json.load(fh)
    if store.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('unsupported prior-store schema')
    return store


def _ingest(store, bundle, origin, allowed_kinds=('external_research',)):
    by_digest = {item['content_sha256']: item for item in store['entries']}
    evidence_to_knowledge = {}
    for evidence in bundle.get('evidence', []):
        evidence_kind = evidence.get('kind')
        if evidence_kind not in allowed_kinds:
            continue
        persona = evidence.get('persona')
        if persona not in PERSONAS:
            continue
        sources = _sources_for_evidence(bundle, evidence)
        digest = _entry_digest(persona, evidence['summary'], sources, evidence_kind)
        entry = by_digest.get(digest)
        evidence_origin = {
            **origin,
            'evidence_id': evidence['evidence_id'],
            'search_id': evidence.get('search_id'),
        }
        if entry is None:
            entry = {
                'knowledge_id': f"K{len(store['entries']) + 1:04d}",
                'content_sha256': digest,
                'kind': 'agent_cached_prior',
                'evidence_kind': evidence_kind,
                'persona': persona,
                'query': evidence.get('query', ''),
                'reason': evidence.get('reason', ''),
                'summary': evidence['summary'],
                'record_type': evidence.get('record_type', 'mechanism'),
                'requires_success_mechanisms': evidence.get(
                    'requires_success_mechanisms', []),
                'sources': sources,
                'origins': [evidence_origin],
            }
            store['entries'].append(entry)
            by_digest[digest] = entry
        elif evidence_origin not in entry['origins']:
            entry['origins'].append(evidence_origin)
        evidence_to_knowledge[evidence['evidence_id']] = entry['knowledge_id']
    return evidence_to_knowledge


def initialize(store_path, seed_snapshot=None, empirical_snapshot=None):
    """Create a run-local store and copy/hash-lock any explicitly supplied priors."""
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    store = _new_store()
    for snapshot, validator, directory, record_key, acquisition, allowed_kinds in (
            (seed_snapshot, _validate_seed, 'seeds', 'seed_snapshots', 'seed',
             ('external_research',)),
            (empirical_snapshot, _validate_empirical_seed, 'empirical',
             'empirical_snapshots', 'curated_empirical_prior', ('curated_research',))):
        if snapshot is None:
            continue
        snapshot = os.path.abspath(snapshot)
        with open(snapshot, encoding='utf-8') as fh:
            bundle = validator(json.load(fh))
        digest = _file_sha256(snapshot)
        seed_dir = os.path.join(os.path.dirname(store_path), directory)
        os.makedirs(seed_dir, exist_ok=True)
        copied = os.path.join(seed_dir, digest + '.json')
        if not os.path.exists(copied):
            shutil.copy2(snapshot, copied)
        seed_record = {
            'sha256': digest,
            'path': os.path.relpath(copied, os.path.dirname(store_path)),
            'original_mode': bundle.get('mode'),
            'source_path': snapshot,
        }
        store[record_key].append(seed_record)
        _ingest(store, bundle, {
            'snapshot_sha256': digest,
            'snapshot_path': seed_record['path'],
            'original_mode': bundle.get('mode'),
            'acquisition': acquisition,
        }, allowed_kinds=allowed_kinds)
        if acquisition == 'curated_empirical_prior':
            store['policy'].update(
                human_curated_prior=True,
                counts_as_runtime_intervention=False,
                expected_validation_delta_stored=True)
    journal.write_json(store_path, store)
    return store


def ingest_bundle(store_path, bundle, snapshot_path, snapshot_label=None):
    """Add a completed live bundle and return its E### -> K#### mapping."""
    store = _load(store_path)
    digest = _file_sha256(snapshot_path)
    mapping = _ingest(store, bundle, {
        'snapshot_sha256': digest,
        'snapshot_path': snapshot_label or os.path.abspath(snapshot_path),
        'original_mode': bundle.get('mode'),
        'acquisition': 'live',
    })
    journal.write_json(store_path, store)
    return mapping


def entries(store_path):
    return _load(store_path)['entries']


def retrieve(store_path, context, used_knowledge_ids=None, per_persona=1,
             successful_mechanisms=None):
    """Retrieve relevant evidence, preferring unused records without exhausting the cache.

    A cited record remains available: using evidence once does not make its mechanism false.
    ``use_count`` is instead a deterministic diversity penalty and an audit field.
    """
    use_counts = collections.Counter(used_knowledge_ids or [])
    successful_text = memory.normalized(' '.join(successful_mechanisms or []))
    query_tokens = memory.tokens(context)
    selected = []
    for persona in PERSONAS:
        ranked = []
        for entry in entries(store_path):
            if entry['persona'] != persona:
                continue
            record_type = entry.get('record_type', 'mechanism')
            requirements = entry.get('requires_success_mechanisms', [])
            if (record_type == 'tuning'
                    and any(memory.normalized(requirement) not in successful_text
                            for requirement in requirements)):
                continue
            text = ' '.join((entry.get('query', ''), entry.get('reason', ''),
                             entry.get('summary', '')))
            entry_tokens = memory.tokens(text)
            overlap = len(query_tokens & entry_tokens)
            containment = overlap / max(1, min(len(query_tokens), len(entry_tokens)))
            use_count = use_counts[entry['knowledge_id']]
            # A modest penalty favors novel evidence while keeping a highly relevant
            # previously cited mechanism ahead of an unrelated unused record.
            type_penalty = {
                'mechanism': 0.0,
                'tuning': 0.04,
                'negative_result': 0.0,
                'guardrail': 0.30,
            }.get(record_type, 0.15)
            retrieval_score = (containment - min(0.24, 0.08 * use_count)
                               - type_penalty)
            ranked.append((retrieval_score, containment, overlap,
                           entry['knowledge_id'], use_count, entry))
        ranked.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
        for score, containment, overlap, _, use_count, entry in ranked[:per_persona]:
            item = copy.deepcopy(entry)
            item['retrieval'] = {
                'score': round(score, 6),
                'lexical_containment': round(containment, 6),
                'token_overlap': overlap,
                'previously_used': use_count > 0,
                'use_count': use_count,
                'record_type': entry.get('record_type', 'mechanism'),
            }
            selected.append(item)
    return selected


def render_catalog(selected, max_chars=900):
    if not selected:
        return '(no cached prior matched this iteration)'
    rendered = []
    for entry in selected:
        summary = ' '.join(entry['summary'].split())
        if len(summary) > max_chars:
            summary = summary[:max_chars] + ' ...[truncated]'
        urls = [source.get('url') for source in entry['sources'] if source.get('url')]
        rendered.append(
            f"[{entry['knowledge_id']}] persona={entry['persona']}; "
            f"kind={entry.get('evidence_kind', 'external_research')}; "
            f"record_type={entry.get('record_type', 'mechanism')}; "
            f"retrieval={json.dumps(entry.get('retrieval', {}), sort_keys=True)}; "
            f"query={entry.get('query', '')}; sources={urls[:4]}; summary={summary}")
    return '\n'.join(rendered)


def _bundle_from_records(records, mode, metadata=None, searches=None, query_plan=None,
                         review=None, search_context_size=None):
    sources, source_by_key, evidence = [], {}, []
    for record in records:
        source_ids = []
        for source in record['sources']:
            key = source.get('url') or source.get('citation')
            if not key:
                continue
            if key not in source_by_key:
                source_id = f'S{len(sources) + 1:03d}'
                source_by_key[key] = source_id
                normalized = copy.deepcopy(source)
                normalized['source_id'] = source_id
                sources.append(normalized)
            source_ids.append(source_by_key[key])
        item = {
            'evidence_id': f'E{len(evidence) + 1:03d}',
            'kind': record.get('evidence_kind', 'external_research'),
            'persona': record['persona'],
            'query': record.get('query', ''),
            'reason': record.get('reason', ''),
            'summary': record['summary'],
            'source_ids': sorted(set(source_ids)),
            'knowledge_id': record['knowledge_id'],
            'knowledge_origin': record.get('knowledge_origin', 'cached'),
            'record_type': record.get('record_type', 'mechanism'),
            'requires_success_mechanisms': record.get(
                'requires_success_mechanisms', []),
        }
        if record.get('origin_evidence_id'):
            item['origin_evidence_id'] = record['origin_evidence_id']
        evidence.append(item)
    bundle = {
        'schema_version': 'external-research-1.1',
        'mode': mode,
        'status': 'complete',
        'stage': 'pre_draft',
        'query_plan': query_plan,
        'review': review,
        'searches': copy.deepcopy(searches or []),
        'sources': sources,
        'evidence': evidence,
    }
    if search_context_size is not None:
        bundle['search_context_size'] = search_context_size
    if metadata:
        bundle.update(copy.deepcopy(metadata))
    return bundle


def cache_bundle(selected, store_path, gap_decision=None):
    """Build a complete research bundle entirely from the local prior."""
    store = _load(store_path)
    records = [{**entry, 'knowledge_origin': 'cached'} for entry in selected]
    return _bundle_from_records(records, 'warm_start', {
        'cache_policy': 'prior_first_cache_hit_no_web_search',
        'seed_snapshots': store['seed_snapshots'],
        'empirical_snapshots': store.get('empirical_snapshots', []),
        'cached_knowledge_ids': [entry['knowledge_id'] for entry in selected],
        'new_knowledge_ids': [],
        'live_search_performed': False,
        'planned_live_query_count': 0,
        'executed_live_query_count': 0,
        'gap_decision': copy.deepcopy(gap_decision),
    })


def seed_bundle(selected, store_path):
    """Backward-compatible alias for cache-only warm starts."""
    return cache_bundle(selected, store_path)


def merge_live_bundle(cached, live_bundle, evidence_to_knowledge,
                      known_knowledge_ids=None):
    """Combine retrieved cache entries with new live evidence under fresh local IDs."""
    known = set(known_knowledge_ids or [])
    records_by_id = {
        entry['knowledge_id']: {**entry, 'knowledge_origin': 'cached'}
        for entry in cached}
    source_map = {item['source_id']: item for item in live_bundle['sources']}
    for item in live_bundle['evidence']:
        sources = []
        for source_id in item['source_ids']:
            source = copy.deepcopy(source_map[source_id])
            source.pop('source_id', None)
            source.pop('search_ids', None)
            sources.append(source)
        knowledge_id = evidence_to_knowledge[item['evidence_id']]
        # Prefer the current live record when it refreshes an already retrieved K####;
        # the immutable live.json still retains the original response provenance.
        records_by_id[knowledge_id] = {
            'knowledge_id': knowledge_id,
            'evidence_kind': item.get('kind', 'external_research'),
            'persona': item['persona'],
            'query': item.get('query', ''),
            'reason': item.get('reason', ''),
            'summary': item['summary'],
            'record_type': item.get('record_type', 'mechanism'),
            'requires_success_mechanisms': item.get(
                'requires_success_mechanisms', []),
            'sources': sources,
            'knowledge_origin': 'live',
            'origin_evidence_id': item['evidence_id'],
        }
    records = list(records_by_id.values())
    mapped_ids = set(evidence_to_knowledge.values())
    bundle = _bundle_from_records(
        records, 'warm_start', {
            'cache_policy': 'retrieved_prior_plus_live_gap_search',
            'cached_knowledge_ids': [entry['knowledge_id'] for entry in cached],
            'new_knowledge_ids': sorted(mapped_ids - known),
            'refreshed_knowledge_ids': sorted(mapped_ids & known),
            'live_search_performed': True,
            'live_mode': live_bundle.get('mode'),
            'planned_live_query_count': live_bundle.get(
                'planned_live_query_count', len(live_bundle.get('query_plan', {}).get('queries', []))),
            'executed_live_query_count': len(live_bundle.get('searches', [])),
            'gap_decision': copy.deepcopy(live_bundle.get('gap_decision')),
        }, searches=live_bundle.get('searches'),
        query_plan=live_bundle.get('query_plan'), review=live_bundle.get('review'),
        search_context_size=live_bundle.get('search_context_size'))
    # Cached sources are emitted first, so live S### identifiers generally shift.
    # Remap search-level provenance instead of leaving identifiers pointing at the
    # wrong source in the merged bundle.  The untouched response is also retained
    # separately as research/live.json by the caller.
    new_by_key = {
        item.get('url') or item.get('citation'): item['source_id']
        for item in bundle['sources']
    }
    old_by_id = {item['source_id']: item for item in live_bundle['sources']}
    for search in bundle['searches']:
        remapped = []
        for source_id in search.get('source_ids', []):
            source = old_by_id.get(source_id, {})
            key = source.get('url') or source.get('citation')
            if key in new_by_key:
                remapped.append(new_by_key[key])
        search['source_ids'] = sorted(set(remapped))
        for source_id in search['source_ids']:
            source = next(item for item in bundle['sources']
                          if item['source_id'] == source_id)
            source.setdefault('search_ids', [])
            if search.get('search_id') not in source['search_ids']:
                source['search_ids'].append(search.get('search_id'))
    return bundle
