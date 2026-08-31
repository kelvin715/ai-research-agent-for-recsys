"""Typed experiment memory and deterministic lexical retrieval for V1+ agents."""
import json
import hashlib
import re


TOKEN_RE = re.compile(r"[a-z_][a-z0-9_]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]+", re.I)
NUMBER_RE = re.compile(r"(?<![a-z_])\d+(?:\.\d+)?", re.I)
PARAMETER_PATTERNS = {
    'lr': (r'learning\s+rate', r'\blr\b'),
    'k': (r'embedding\s+(?:dimension|size)', r'latent\s+factors?', r'\bk\b'),
    'duration_buckets': (r'duration\s+buckets?', r'n_dur_buckets'),
    'hourmin_buckets': (r'hourmin\s+buckets?', r'n_hourmin_buckets'),
    'date_buckets': (r'date\s+buckets?', r'n_date_buckets'),
    'batch_size': (r'batch\s+size', r'\bbatch\b'),
    'epochs': (r'\bepochs?\b',),
}
STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'better', 'by', 'can', 'capture',
    'current', 'due', 'for', 'from', 'improve', 'improves', 'improving', 'in',
    'introducing', 'is', 'it', 'lead', 'leading', 'may', 'model', 'more', 'of',
    'on', 'performance', 'potentially', 'set', 'that', 'the', 'this', 'to',
    'validation', 'will', 'with',
}


def tokens(text):
    return {token.lower() for token in TOKEN_RE.findall(text or '')
            if token.lower() not in STOPWORDS and len(token) > 1}


def normalized(text):
    return ' '.join(TOKEN_RE.findall((text or '').lower()))


def lexical_similarity(a, b):
    """Containment-weighted overlap; robust to boilerplate and small paraphrases."""
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb)
    containment = overlap / min(len(ta), len(tb))
    jaccard = overlap / len(ta | tb)
    return round(0.7 * containment + 0.3 * jaccard, 6)


def parameter_signature(text):
    """Extract parameter family and an explicit from/to pair when present."""
    value = (text or '').lower().replace('→', ' to ').replace('->', ' to ')
    value = re.sub(r'ndcg\s*@\s*\d+', 'ndcg', value)
    parameters = sorted(name for name, patterns in PARAMETER_PATTERNS.items()
                        if any(re.search(pattern, value) for pattern in patterns))
    transition = None
    match = re.search(
        r'from\s+(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?)', value)
    if match:
        transition = [match.group(1), match.group(2)]
    return {'parameters': parameters, 'transition': transition}


def same_parameter_transition(a, b):
    sa, sb = parameter_signature(a), parameter_signature(b)
    return bool(set(sa['parameters']) & set(sb['parameters']) and
                sa['transition'] is not None and
                sa['transition'] == sb['transition'])


def proposal_text(proposal):
    return ' '.join(str(proposal.get(key, '')) for key in
                    ('hypothesis', 'observation', 'justification'))


def mechanism_text(proposal):
    return str(proposal.get('mechanism') or proposal.get('hypothesis') or '')


def implementation_signature(proposal):
    """Stable signature for one proposed implementation, not the broader mechanism."""
    payload = {
        'execution_mode': proposal.get('execution_mode'),
        'operator_id': proposal.get('operator_id'),
        'parent_operator_stack': proposal.get('parent_operator_stack'),
        'mechanism': normalized(mechanism_text(proposal)),
        'implementation_plan': normalized(str(proposal.get('implementation_plan') or '')),
        'patch_scope': sorted(proposal.get('patch_scope') or []),
        'primary_block': proposal.get('primary_block'),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def retrieve(entries, proposal, top_k=4):
    """Retrieve by exact primary block plus hypothesis/content token overlap."""
    query = proposal_text(proposal)
    query_hypothesis = proposal.get('hypothesis', '')
    query_mechanism = mechanism_text(proposal)
    query_implementation = implementation_signature(proposal)
    block = proposal.get('primary_block')
    ranked = []
    for entry in entries:
        context_sim = lexical_similarity(query, entry.get('hypothesis', '') + ' '
                                         + entry.get('conclusion', ''))
        hypothesis_sim = lexical_similarity(query_hypothesis, entry.get('hypothesis', ''))
        same_block = entry.get('primary_block') == block
        same_scope = set(entry.get('patch_scope') or []) == set(proposal.get('patch_scope') or [])
        exact = normalized(proposal.get('hypothesis')) == normalized(entry.get('hypothesis'))
        signature_match = same_parameter_transition(
            query_hypothesis, entry.get('hypothesis', ''))
        mechanism_sim = lexical_similarity(
            query_mechanism, entry.get('mechanism') or entry.get('hypothesis', ''))
        same_implementation = (
            query_implementation == entry.get('implementation_signature'))
        same_operator = bool(
            proposal.get('execution_mode') == 'operator'
            and proposal.get('operator_id') == entry.get('operator_id')
            and proposal.get('parent_operator_stack')
            == entry.get('parent_operator_stack'))
        score = (hypothesis_sim + 0.15 * context_sim
                 + (0.20 if same_block else 0.0)
                 + (1.0 if exact else 0.0)
                 + 0.5 * mechanism_sim
                 + (1.2 if same_operator else 0.0)
                 + (1.2 if same_implementation else 0.0)
                 + (0.8 if signature_match else 0.0))
        ranked.append((score, {
            'memory_id': entry['memory_id'],
            'kind': entry['kind'],
            'failure_class': entry.get('failure_class'),
            'mechanism': entry.get('mechanism'),
            'mechanism_similarity': mechanism_sim,
            'implementation_plan': entry.get('implementation_plan'),
            'implementation_signature': entry.get('implementation_signature'),
            'execution_mode': entry.get('execution_mode'),
            'operator_id': entry.get('operator_id'),
            'parent_operator_stack': entry.get('parent_operator_stack'),
            'same_operator': same_operator,
            'research_knowledge_ids': entry.get('research_knowledge_ids', []),
            'same_implementation': same_implementation,
            'primary_block': entry['primary_block'],
            'patch_scope': entry.get('patch_scope'),
            'hypothesis': entry['hypothesis'],
            'conclusion': entry['conclusion'],
            'delta_primary': entry.get('delta_primary', entry.get('delta_primary_mean')),
            'delta_primary_mean': entry.get('delta_primary_mean', entry.get('delta_primary')),
            'lexical_similarity': hypothesis_sim,
            'hypothesis_similarity': hypothesis_sim,
            'context_similarity': context_sim,
            'parameter_signature': parameter_signature(query_hypothesis),
            'memory_parameter_signature': parameter_signature(entry.get('hypothesis', '')),
            'parameter_signature_match': signature_match,
            'same_block': same_block,
            'same_patch_scope': same_scope,
            'exact_hypothesis': exact,
        }))
    ranked.sort(key=lambda pair: (-pair[0], pair[1]['memory_id']))
    return [item for score, item in ranked[:top_k] if score > 0]


def requires_memory_review(matches, similarity_threshold=0.58):
    """Review overlaps without treating one bad implementation as a dead mechanism."""
    return any(match.get('same_implementation') or match.get('same_operator') or
               match['exact_hypothesis'] or
               match.get('parameter_signature_match') or
               (match.get('mechanism_similarity', 0) >= similarity_threshold) or
               (match['same_block'] and
                match['hypothesis_similarity'] >= similarity_threshold)
               for match in matches)


def failure_class(decision, outcome):
    if decision == 'ACCEPT':
        return None
    if decision == 'UNCERTAIN':
        return 'statistical_inconclusive'
    if decision == 'ROLLBACK':
        return 'statistical_negative'
    if decision == 'NO_OP':
        return 'no_op'
    if decision == 'REJECT':
        reason = outcome.get('reason') or {}
        kind = reason.get('kind') if isinstance(reason, dict) else None
        return 'policy_or_static' if kind == 'static_gate' else 'implementation_or_runtime'
    if decision in {'PLANNING_ERROR', 'ORCHESTRATOR_ERROR', 'DEADLINE'}:
        return 'controller_or_budget'
    return 'planning'


def build_entry(iteration, proposal, outcome, reflection):
    decision = outcome['decision']
    standalone_delta = outcome.get('delta_primary', outcome.get('delta_primary_mean'))
    portfolio_delta = outcome.get('portfolio_delta_primary')
    delta = (portfolio_delta
             if outcome.get('acceptance_channel') in {
                 'portfolio_marginal', 'standalone_and_portfolio',
                 'portfolio_marginal_unconfirmed'} and portfolio_delta is not None
             else standalone_delta)
    candidate = outcome.get('candidate_metrics') or {}
    paired = candidate.get('paired_vs_incumbent') or {}
    fail_class = failure_class(decision, outcome)
    if decision == 'ACCEPT':
        kind = 'success'
        conclusion = (f"accepted: delta={delta:+.9f}; paired_ci95="
                      f"{json.dumps(paired.get('paired_ci95'))}; "
                      f"lesson={reflection.get('next_lesson', '')}")
    elif decision == 'UNCERTAIN':
        kind = 'inconclusive'
        conclusion = (f"uncertain: delta={delta:+.9f}; paired_ci95="
                      f"{json.dumps(paired.get('paired_ci95'))}; "
                      f"lesson={reflection.get('next_lesson', '')}")
    elif decision == 'NO_OP':
        kind = 'failure'
        conclusion = ('no-op: predictions were identical to the incumbent for all seeds; '
                      f"lesson={reflection.get('next_lesson', '')}")
    elif decision in {'ROLLBACK', 'REJECT', 'MEMORY_REJECT'}:
        kind = 'failure'
        if delta is None:
            conclusion = (f"{decision.lower()}: {json.dumps(outcome.get('reason'), ensure_ascii=False)}; "
                          f"lesson={reflection.get('next_lesson', '')}")
        else:
            conclusion = (f"rolled back: delta={delta:+.9f}; paired_ci95="
                          f"{json.dumps(paired.get('paired_ci95'))}; "
                          f"lesson={reflection.get('next_lesson', '')}")
    else:
        kind = 'failure'
        conclusion = f"{decision.lower()}: lesson={reflection.get('next_lesson', '')}"
    return {
        'memory_id': f'm{iteration:03d}',
        'iteration': iteration,
        'kind': kind,
        'failure_class': fail_class,
        'decision': decision,
        'primary_block': proposal['primary_block'],
        'patch_scope': proposal['patch_scope'],
        'persona': proposal.get('persona'),
        'basis_type': proposal.get('basis_type'),
        'execution_mode': proposal.get('execution_mode'),
        'operator_id': proposal.get('operator_id'),
        'parent_operator_stack': proposal.get('parent_operator_stack'),
        'research_nonuse_reason': proposal.get('research_nonuse_reason'),
        'mechanism': mechanism_text(proposal),
        'mechanism_tags': proposal.get('mechanism_tags', []),
        'parent_references': proposal.get('parent_references', []),
        'experiment_contract': proposal.get('experiment_contract'),
        'implementation_plan': proposal.get('implementation_plan'),
        'implementation_signature': implementation_signature(proposal),
        'candidate_pipeline_sha256': outcome.get('candidate_pipeline_sha256'),
        'research_evidence_ids': [item.get('evidence_id') for item in
                                  proposal.get('research_evidence', [])],
        'research_knowledge_ids': sorted(set(
            item.get('knowledge_id') for item in proposal.get('research_evidence', [])
            if item.get('knowledge_id'))),
        'hypothesis': proposal['hypothesis'],
        'conclusion': conclusion,
        'delta_primary': delta,
        'delta_primary_mean': delta,
        'standalone_delta_primary': standalone_delta,
        'portfolio_delta_primary': portfolio_delta,
        'acceptance_channel': outcome.get('acceptance_channel'),
    }
