"""Faithful multi-turn text loop matching the original MedAgentBench protocol.

The agent outputs raw text in one of three formats:
  GET url?param1=value1&param2=value2...
  POST url
  {json payload}
  FINISH([answer1, answer2, ...])

The loop parses the text, executes the action, appends the result to the
message history, and repeats up to max_round times.
"""

import json
import time

from litellm import completion, completion_cost
from secretagent import config, record
from secretagent.llm_util import echo_boxed

import fhir_tools


def _completion_with_backoff(**kw):
    """Retry completion with exponential backoff for rate limits."""
    for attempt in range(5):
        try:
            return completion(**kw)
        except Exception as e:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)

# Exact system prompt from the original MedAgentBench paper
_PROMPT_TEMPLATE = """You are an expert in using FHIR functions to assist medical professionals. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.

1. If you decide to invoke a GET function, you MUST put it in the format of
GET url?param_name1=param_value1&param_name2=param_value2...

2. If you decide to invoke a POST function, you MUST put it in the format of
POST url
[your payload data in JSON format]

3. If you have got answers for all the questions and finished all the requested tasks, you MUST call to finish the conversation in the format of (make sure the list is JSON loadable.)
FINISH([answer1, answer2, ...])

Your response must be in the format of one of the three cases, and you can call only one function each time. You SHOULD NOT include any other text in the response.

Here is a list of functions in JSON format that you can invoke. Note that you should use {api_base} as the api_base.
{functions}

Context: {context}
Question: {question}"""


def medagent_loop(instruction: str, context: str,
                   allowed_actions=('GET', 'POST', 'FINISH'),
                   max_round=None, return_raw=False) -> list[str]:
    """Multi-turn text loop replicating the MedAgentBench paper protocol.

    Args:
        allowed_actions: which actions the agent can use (for read_act phases)
        max_round: override max rounds (default from config)
        return_raw: if True, return the raw conversation text instead of parsed FINISH answers
    """
    model = config.require('llm.model')
    if max_round is None:
        max_round = int(config.get('fhir.max_round', 8))
    max_tokens = int(config.get('llm.max_tokens', 2048))
    fhir_base = config.get('fhir.api_base', 'http://localhost:8080/fhir/')

    # Load FHIR function definitions
    from pathlib import Path
    funcs_file = Path(__file__).parent / 'data' / 'funcs_v1.json'
    funcs = json.loads(funcs_file.read_text())

    # Extract raw task context from the enriched context
    # (enriched context has FHIR base + API defs prepended by load_dataset;
    # the original prompt template embeds those itself, so we extract just
    # the task-specific context)
    task_context = ''
    marker = 'Task context: '
    if marker in context:
        task_context = context[context.index(marker) + len(marker):]

    # Build the initial prompt using the paper's exact template
    system_prompt = _PROMPT_TEMPLATE.format(
        api_base=fhir_base,
        functions=json.dumps(funcs),
        context=task_context,
        question=instruction,
    )

    messages = [{"role": "user", "content": system_prompt}]

    if config.get('echo.llm_input'):
        echo_boxed(system_prompt, 'llm_input (round 0)')

    # Aggregate stats across all rounds
    total_stats = dict(input_tokens=0, output_tokens=0, latency=0.0, cost=0.0)
    history = []  # for recording

    for round_idx in range(max_round):
        # Call LLM
        start = time.time()
        response = _completion_with_backoff(model=model, messages=messages, max_tokens=max_tokens)
        latency = time.time() - start

        raw = response.choices[0].message.content or ''
        total_stats['input_tokens'] += response.usage.prompt_tokens
        total_stats['output_tokens'] += response.usage.completion_tokens
        total_stats['latency'] += latency
        try:
            total_stats['cost'] += completion_cost(completion_response=response)
        except Exception:
            pass

        if config.get('echo.llm_output'):
            echo_boxed(raw, f'llm_output (round {round_idx})')

        # Strip markdown wrappers (Gemini quirk from original code)
        r = raw.strip().replace('```tool_code', '').replace('```', '').strip()

        # Parse action (check allowed_actions for read_act phases)
        action = 'GET' if r.startswith('GET') else 'POST' if r.startswith('POST') else 'FINISH' if r.startswith('FINISH(') else None
        if action and action not in allowed_actions:
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"{action} is not available in this phase. Use: {', '.join(allowed_actions)}"})
            continue

        if r.startswith('GET'):
            url = r[3:].strip() + '&_format=json'
            get_res = fhir_tools.fhir_get(url.replace('&_format=json&_format=json', '&_format=json'))
            # fhir_get already appends _format=json, so use raw URL
            # Actually, replicate original exactly: original does url + '&_format=json'
            # and fhir_get also adds it. Let's use the raw send_get_request instead.
            raw_res = fhir_tools._send_get_request_raw(r[3:].strip() + '&_format=json')
            if 'data' in raw_res:
                feedback = (f"Here is the response from the GET request:\n{raw_res['data']}. "
                            "Please call FINISH if you have got answers for all the questions "
                            "and finished all the requested tasks")
            else:
                feedback = f"Error in sending the GET request: {raw_res['error']}"
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": feedback})
            history.append({'round': round_idx, 'action': 'GET', 'url': r[3:].strip()})

        elif r.startswith('POST'):
            lines = r.split('\n')
            post_url = lines[0][4:].strip()
            payload_text = '\n'.join(lines[1:])
            try:
                payload = json.loads(payload_text)
                # Log the POST for refsol grading (same as fhir_tools.fhir_post)
                fhir_tools.log_post(post_url, payload)
                feedback = ("POST request accepted and executed successfully. "
                            "Please call FINISH if you have got answers for all the questions "
                            "and finished all the requested tasks")
            except Exception:
                feedback = "Invalid POST request"
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": feedback})
            history.append({'round': round_idx, 'action': 'POST', 'url': post_url})

        elif r.startswith('FINISH('):
            # Extract answer list
            result_str = r[len('FINISH('):-1]
            try:
                answers = json.loads(result_str)
                if not isinstance(answers, list):
                    answers = [answers]
            except (json.JSONDecodeError, TypeError):
                answers = [result_str]
            history.append({'round': round_idx, 'action': 'FINISH', 'answers': answers})
            record.record(
                func='solve_medical_task', args=(instruction, context),
                kw={}, output=answers, stats=total_stats,
                step_info={'history': history, 'rounds': round_idx + 1})
            return answers

        else:
            # Invalid action — stop (matching original behavior)
            history.append({'round': round_idx, 'action': 'INVALID', 'content': r[:200]})
            break

    # Reached max rounds without FINISH
    if return_raw:
        # For read phase: return all GET responses concatenated
        raw_data = '\n'.join(
            m['content'] for m in messages
            if m['role'] == 'user' and 'response from the GET' in m.get('content', ''))
        return raw_data or '(no data retrieved)'
    record.record(
        func='solve_medical_task', args=(instruction, context),
        kw={}, output='**max rounds reached**', stats=total_stats,
        step_info={'history': history, 'rounds': max_round, 'status': 'TASK_LIMIT_REACHED'})
    return []


# ──────────────────────────────────────────────────────────────────────
# CodeAct: iterative code generation with error feedback
# ──────────────────────────────────────────────────────────────────────

_CODEACT_PROMPT = """You are an expert in using FHIR functions to assist medical professionals.
Write Python code to solve the task below. You have these tools available as functions:

- fhir_get(url: str) -> str: Send a GET request to the FHIR server. Returns JSON string.
- fhir_post(url: str, payload: str) -> str: Send a POST request. payload is a JSON string.
- final_answer(result): Call this with your final answer to return it.

You also have: json, re, and all standard Python builtins.

FHIR API base URL: {api_base}

Context: {context}
Question: {question}

Write Python code that solves this task. Call final_answer(result) as the last line.
Output ONLY a ```python``` code block."""


def codeact_loop(instruction: str, context: str) -> list:
    """CodeAct: iterative code generation with error feedback, up to 8 passes."""
    import re as re_mod
    from smolagents.local_python_executor import LocalPythonExecutor, BASE_PYTHON_TOOLS

    model = config.require('llm.model')
    max_passes = int(config.get('fhir.max_round', 8))
    max_tokens = int(config.get('llm.max_tokens', 4096))
    fhir_base = config.get('fhir.api_base', 'http://localhost:8080/fhir/')

    # Extract task context
    task_context = ''
    marker = 'Task context: '
    if marker in context:
        task_context = context[context.index(marker) + len(marker):]

    # Set up sandbox
    executor = LocalPythonExecutor(additional_authorized_imports=['json', 're'])
    executor.custom_tools = {
        'fhir_get': fhir_tools.fhir_get,
        'fhir_post': fhir_tools.fhir_post,
        'instruction': instruction,
        'context': context,
    }
    executor.static_tools = {**BASE_PYTHON_TOOLS, 'final_answer': lambda x: x}

    prompt = _CODEACT_PROMPT.format(
        api_base=fhir_base, context=task_context, question=instruction)
    messages = [{"role": "user", "content": prompt}]

    total_stats = dict(input_tokens=0, output_tokens=0, latency=0.0, cost=0.0)

    for attempt in range(max_passes):
        start = time.time()
        response = _completion_with_backoff(model=model, messages=messages, max_tokens=max_tokens)
        latency = time.time() - start

        raw = response.choices[0].message.content or ''
        total_stats['input_tokens'] += response.usage.prompt_tokens
        total_stats['output_tokens'] += response.usage.completion_tokens
        total_stats['latency'] += latency
        try:
            total_stats['cost'] += completion_cost(completion_response=response)
        except Exception:
            pass

        if config.get('echo.llm_output'):
            echo_boxed(raw, f'codeact (pass {attempt})')

        # Extract code from ```python``` block
        match = re_mod.search(r'```python\n(.*?)\n```', raw, re_mod.DOTALL)
        if not match:
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content":
                "No ```python``` code block found. Please output ONLY a ```python``` code block."})
            continue

        code = match.group(1)
        try:
            result = executor(code)
            answer = result.output
            if not isinstance(answer, list):
                answer = [answer]
            record.record(
                func='solve_medical_task', args=(instruction, context),
                kw={}, output=answer, stats=total_stats,
                step_info={'passes': attempt + 1, 'code': code})
            return answer
        except Exception as ex:
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content":
                f"Code execution error:\n{ex}\n\nFix the code and try again."})

    # All passes exhausted
    record.record(
        func='solve_medical_task', args=(instruction, context),
        kw={}, output='**max passes reached**', stats=total_stats,
        step_info={'passes': max_passes, 'status': 'EXHAUSTED'})
    return []
