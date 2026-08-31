"""Small deterministic validators for structured Agent outputs."""

BLOCKS = {'data_view', 'features', 'target', 'model', 'loss', 'train', 'predict'}
PERSONAS = {'optimizer', 'architecture', 'reward'}
EVIDENCE_BASES = {
    'external_research', 'curated_research', 'diagnostic', 'current_code',
    'journal', 'task_spec', 'prior_knowledge',
}
RESEARCH_BASES = {'external_research', 'curated_research'}
PORTFOLIO_ROLES = {
    'standalone_improvement', 'member_refinement',
    'error_diversifier', 'mechanism_fusion',
}


def require_string(obj, key):
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{key} 必须是非空字符串')
    return value.strip()


def validate_proposal(obj):
    if not isinstance(obj, dict):
        raise ValueError('proposal 必须是 JSON object')
    for key in ('mechanism', 'implementation_plan', 'hypothesis', 'observation',
                'justification', 'expected_observation', 'primary_block', 'fallback'):
        require_string(obj, key)
    basis_type = require_string(obj, 'basis_type')
    if basis_type not in EVIDENCE_BASES:
        raise ValueError(f'basis_type 必须属于 {sorted(EVIDENCE_BASES)}')
    execution_mode = require_string(obj, 'execution_mode')
    if execution_mode not in {'operator', 'custom_patch'}:
        raise ValueError('execution_mode 必须是 operator|custom_patch')
    operator_id = obj.get('operator_id')
    if execution_mode == 'operator':
        if not isinstance(operator_id, str) or not operator_id.strip():
            raise ValueError('operator proposal 必须提供 operator_id')
        scope_limit = len(BLOCKS)
    else:
        if operator_id is not None:
            raise ValueError('custom_patch proposal 的 operator_id 必须为 null')
        scope_limit = 3
    scope = obj.get('patch_scope')
    if (not isinstance(scope, list) or not 1 <= len(scope) <= scope_limit
            or any(block not in BLOCKS for block in scope)
            or len(scope) != len(set(scope))):
        raise ValueError(f'patch_scope 必须含 1–{scope_limit} 个不同的合法 block')
    if obj['primary_block'] not in scope:
        raise ValueError('primary_block 必须属于 patch_scope')
    evidence = obj.get('evidence')
    if not isinstance(evidence, list) or not evidence:
        raise ValueError('evidence 必须是非空数组')
    for item in evidence:
        if not isinstance(item, dict):
            raise ValueError('evidence item 必须是 object')
        require_string(item, 'type')
        require_string(item, 'ref')
    if not any(item['type'] == basis_type for item in evidence):
        raise ValueError('evidence 至少一项的 type 必须等于 basis_type')
    adaptation = obj.get('evidence_adaptation')
    if basis_type in RESEARCH_BASES:
        if not isinstance(adaptation, dict):
            raise ValueError(
                'research-derived proposal 的 evidence_adaptation 必须是 object，'
                '形如 {"evidence_id":"E###","source_mechanism":"...",'
                '"implementation_mapping":"...","protocol_caveat":"..."}；不能为 null 或 string')
        for key in ('evidence_id', 'source_mechanism', 'implementation_mapping',
                    'protocol_caveat'):
            require_string(adaptation, key)
        if obj['primary_block'] not in adaptation['implementation_mapping']:
            raise ValueError('evidence_adaptation.implementation_mapping 必须点名 primary_block')
    elif adaptation is not None:
        raise ValueError('非 research-derived proposal 的 evidence_adaptation 必须为 null')
    nonuse = obj.get('research_nonuse_reason')
    if basis_type not in RESEARCH_BASES:
        if not isinstance(nonuse, str) or not nonuse.strip():
            raise ValueError('非 research-derived proposal 必须解释 research_nonuse_reason')
    elif nonuse not in (None, ''):
        raise ValueError('research-derived proposal 的 research_nonuse_reason 应为 null')
    risk = obj.get('risk')
    if not isinstance(risk, list) or any(not isinstance(x, str) for x in risk):
        raise ValueError('risk 必须是字符串数组')
    cost = obj.get('estimated_cost_s')
    if not isinstance(cost, (int, float)) or cost <= 0:
        raise ValueError('estimated_cost_s 必须为正数')
    tags = obj.get('mechanism_tags')
    if (not isinstance(tags, list) or not tags or len(tags) > 8
            or any(not isinstance(tag, str) or not tag.strip() for tag in tags)):
        raise ValueError('mechanism_tags 必须含 1–8 个非空字符串')
    parents = obj.get('parent_references')
    if (not isinstance(parents, list)
            or any(not isinstance(parent, str) or not parent.strip()
                   for parent in parents)):
        raise ValueError('parent_references 必须是字符串数组')
    contract = obj.get('experiment_contract')
    if not isinstance(contract, dict):
        raise ValueError('experiment_contract 必须是 object')
    for key in ('claim', 'expected_observable', 'falsification_condition'):
        require_string(contract, key)
    for key in ('required_components', 'forbidden_shortcuts'):
        values = contract.get(key)
        if (not isinstance(values, list) or not values
                or any(not isinstance(value, str) or not value.strip()
                       for value in values)):
            raise ValueError(f'experiment_contract.{key} 必须是非空字符串数组')
    role = require_string(contract, 'portfolio_role')
    if role not in PORTFOLIO_ROLES:
        raise ValueError(
            f'experiment_contract.portfolio_role 必须属于 {sorted(PORTFOLIO_ROLES)}')
    if role == 'mechanism_fusion' and len(parents) < 2:
        raise ValueError('mechanism_fusion 必须引用至少两个 parent_references')
    return obj


def validate_persona_proposal(obj):
    validate_proposal(obj)
    persona = require_string(obj, 'persona')
    if persona not in PERSONAS:
        raise ValueError(f'persona 必须属于 {sorted(PERSONAS)}')
    return obj


def validate_patch(obj, patch_scope):
    if not isinstance(obj, dict):
        raise ValueError('patch 必须是 JSON object')
    replacements = obj.get('replacements')
    if not isinstance(replacements, list):
        raise ValueError('replacements 必须是数组')
    blocks = []
    for item in replacements:
        if not isinstance(item, dict):
            raise ValueError('replacement 必须是 object')
        block = require_string(item, 'block')
        code = require_string(item, 'code')
        if block not in BLOCKS:
            raise ValueError(f'未知 block: {block}')
        if '<<<BLOCK:' in code or '<<<END:' in code or '```' in code:
            raise ValueError('replacement code 不得包含 block 哨兵或 markdown fence')
        blocks.append(block)
    if len(blocks) != len(set(blocks)) or set(blocks) != set(patch_scope):
        raise ValueError(f'replacement blocks {blocks} 必须精确等于 patch_scope {patch_scope}')
    require_string(obj, 'notes')
    return obj


def validate_reflection(obj):
    if not isinstance(obj, dict):
        raise ValueError('reflection 必须是 JSON object')
    result = require_string(obj, 'result')
    if result not in {'supported', 'not_supported', 'inconclusive', 'failed'}:
        raise ValueError('reflection.result 非法')
    require_string(obj, 'analysis')
    require_string(obj, 'next_lesson')
    return obj


def implementation_audit_validator(required_components):
    """Build a fail-closed validator for the proposal-to-code fidelity audit."""
    required_components = list(required_components)

    def validate(obj):
        if not isinstance(obj, dict):
            raise ValueError('implementation audit 必须是 JSON object')
        status = require_string(obj, 'status')
        if status not in {'PASS', 'REVISE'}:
            raise ValueError('implementation audit.status 必须是 PASS|REVISE')
        require_string(obj, 'analysis')
        evidence = obj.get('component_evidence')
        if not isinstance(evidence, list):
            raise ValueError('component_evidence 必须是数组')
        observed = []
        for item in evidence:
            if not isinstance(item, dict):
                raise ValueError('component_evidence item 必须是 object')
            component = require_string(item, 'component')
            require_string(item, 'code_evidence')
            observed.append(component)
        if sorted(observed) != sorted(required_components):
            raise ValueError('component_evidence 必须逐项覆盖 required_components')
        missing = obj.get('missing_components')
        shortcuts = obj.get('forbidden_shortcuts_found')
        for key, values in (('missing_components', missing),
                            ('forbidden_shortcuts_found', shortcuts)):
            if (not isinstance(values, list)
                    or any(not isinstance(value, str) for value in values)):
                raise ValueError(f'{key} 必须是字符串数组')
        changed = obj.get('changed_runtime_path')
        if not isinstance(changed, bool):
            raise ValueError('changed_runtime_path 必须是 boolean')
        if status == 'PASS' and (missing or shortcuts or not changed):
            raise ValueError('PASS 要求无缺失/捷径且运行时路径实际改变')
        if status == 'REVISE' and not (missing or shortcuts or not changed):
            raise ValueError('REVISE 必须给出具体缺失、捷径或未改变的运行时路径')
        return obj

    return validate


def validate_memory_review(obj):
    if not isinstance(obj, dict):
        raise ValueError('memory review 必须是 JSON object')
    action = require_string(obj, 'action')
    if action not in {'proceed', 'revise', 'reject'}:
        raise ValueError('memory review.action 必须是 proceed|revise|reject')
    require_string(obj, 'analysis')
    require_string(obj, 'novelty_or_new_evidence')
    return obj
