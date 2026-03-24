"""Access an LLM model, and monitor cost, latency, etc.
"""

import re
import time
from typing import Any

from secretagent import config
from secretagent.cache_util import cached
from litellm import completion, completion_cost

def echo_boxed(text: str, tag:str = ''):
    """Echo some text in a pretty box."""
    lines = text.split('\n')
    width = max(len(line) for line in lines)
    print('┌' + tag.center(width+2, '─') + '┐')
    for line in lines:
        print('│ ' + line.ljust(width) + ' │')
    print('└' + '─' * (width + 2) + '┘')

def _llm_impl(prompt: str, model: str) -> tuple[str, dict[str, Any]]:
  """Use an LLM model.

  Returns result as a string plus a dictionary of measurements,
  including # input_tokens, # output_tokens, latency in seconds, and cost.
  """
  if config.get('echo.model'):
    print(f'calling model {model}')

  if config.get('echo.llm_input'):
    echo_boxed(prompt, 'llm_input')

  messages = [dict(role='user', content=prompt)]
  extra_kw = {}
  max_tokens = config.get('llm.max_tokens', None)
  if max_tokens:
    extra_kw['max_tokens'] = int(max_tokens)
  start_time = time.time()

  response = completion(model=model, messages=messages, **extra_kw)
  latency = time.time() - start_time

  msg = response.choices[0].message
  content = msg.content or ''
  reasoning = getattr(msg, 'reasoning_content', None) or ''
  # Thinking models (e.g. Qwen 3.5) sometimes put <answer> tags in
  # reasoning_content instead of content. Prefer content; fall back
  # to reasoning_content only if content lacks the expected tags.
  if content and '<answer>' in content:
    model_output = content
  elif reasoning and '<answer>' in reasoning:
    # Extract only the LAST <answer>...</answer> block from reasoning,
    # since earlier ones are the model thinking about the format.
    matches = re.findall(r'<answer>(.*?)</answer>', reasoning, re.DOTALL)
    if matches:
      last_answer = matches[-1].strip()
      model_output = f'<answer>{last_answer}</answer>'
    else:
      model_output = reasoning
  else:
    model_output = content or reasoning

  if config.get('echo.llm_output'):
    echo_boxed(model_output, 'llm_output')

  stats = dict(
    input_tokens=response.usage.prompt_tokens,
    output_tokens=response.usage.completion_tokens,
    latency=latency,
    cost=completion_cost(completion_response=response),
  )
  return model_output, stats

def llm(prompt: str, model: str) -> tuple[str, dict[str, Any]]:
  """Use an LLM model, with optional cachier caching via config.

  See cache_util.py for why this weird process is necessary.
  """
  return cached(_llm_impl)(prompt, model)
