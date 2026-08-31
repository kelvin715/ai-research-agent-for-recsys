"""Minimal trusted GPT JSON client with token/latency accounting."""
import json
import os
import ssl
import time

import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def uses_modern_completion_params(model):
    """Return whether a model rejects legacy temperature/max_tokens fields."""
    return (model == 'gpt-5' or model.startswith('gpt-5.')
            or model.startswith('gpt-5-'))


def load_env(path=os.path.join(ROOT, '.env')):
    env = {}
    with open(path, encoding='utf-8') as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            env[key.strip()] = value.strip().strip('"\'')
    required = ['OPENAI_API_KEY', 'OPENAI_BASE_URL', 'LLM_MODEL']
    missing = [key for key in required if not env.get(key)]
    if missing:
        raise RuntimeError(f'.env 缺少配置: {missing}')
    return env


class JsonLLM:
    def __init__(self, timeout_s=180, retries=3, model=None):
        env = load_env()
        self.model = model or env['LLM_MODEL']
        self.modern_completion_params = uses_modern_completion_params(self.model)
        self.temperature = None if self.modern_completion_params else 0
        self.web_search_tool_choice = ('auto' if self.modern_completion_params else 'required')
        self.web_search_max_tokens = (12000 if self.model == 'gpt-5.4'
                                      else (8000 if self.modern_completion_params else 1800))
        self.web_search_reasoning_effort = ('low' if self.modern_completion_params else None)
        self.web_search_verbosity = ('low' if self.modern_completion_params else None)
        self.base_url = env['OPENAI_BASE_URL'].rstrip('/')
        self.api_key = env['OPENAI_API_KEY']
        self.verify_ssl = env.get('VERIFY_SSL', '1') != '0'
        self.timeout_s = timeout_s
        self.retries = retries
        self.deadline = None

    def set_deadline(self, deadline):
        """Set an absolute monotonic deadline shared by retries and backoff."""
        self.deadline = deadline

    def _remaining(self):
        if self.deadline is None:
            return None
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('LLM call skipped: run wall-clock deadline reached')
        return remaining

    @property
    def endpoint(self):
        if self.base_url.endswith('/v1'):
            return self.base_url + '/chat/completions'
        return self.base_url + '/v1/chat/completions'

    def call(self, phase, system_prompt, user_prompt, max_tokens=4000):
        payload = {
            'model': self.model,
            'response_format': {'type': 'json_object'},
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
        }
        if self.temperature is not None:
            payload['temperature'] = self.temperature
        token_limit_key = ('max_completion_tokens' if self.modern_completion_params
                           else 'max_tokens')
        payload[token_limit_key] = max_tokens
        headers = {'Authorization': f'Bearer {self.api_key}',
                   'Content-Type': 'application/json'}
        last_error = None
        for attempt in range(1, self.retries + 1):
            remaining = self._remaining()
            call_timeout = self.timeout_s if remaining is None else min(self.timeout_s, remaining)
            t0 = time.time()
            try:
                with httpx.Client(verify=self.verify_ssl, timeout=call_timeout) as client:
                    response = client.post(self.endpoint, headers=headers, json=payload)
                response.raise_for_status()
                body = response.json()
                content = body['choices'][0]['message']['content']
                if isinstance(content, list):
                    content = ''.join(item.get('text', '') for item in content
                                      if isinstance(item, dict))
                text = str(content).strip()
                if text.startswith('```'):
                    text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
                obj = json.loads(text)
                usage = body.get('usage') or {}
                return obj, {
                    'phase': phase,
                    'model': self.model,
                    'attempt': attempt,
                    'latency_s': round(time.time() - t0, 3),
                    'prompt_tokens': int(usage.get('prompt_tokens') or 0),
                    'completion_tokens': int(usage.get('completion_tokens') or 0),
                    'total_tokens': int(usage.get('total_tokens') or 0),
                    'response_id': body.get('id'),
                }
            except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError) as exc:
                last_error = f'{type(exc).__name__}: {exc}'
                if attempt < self.retries:
                    delay = 2 ** (attempt - 1)
                    remaining = self._remaining()
                    if remaining is not None:
                        delay = min(delay, remaining)
                    time.sleep(delay)
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError(f'LLM {phase} reached run deadline: {last_error}')
        raise RuntimeError(f'LLM {phase} 连续 {self.retries} 次失败: {last_error}')
