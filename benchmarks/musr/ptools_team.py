"""Interfaces for MUSR team allocation.

Single-ptool approach: answer_question bound to simulate with thinking
outperforms decomposed workflows (84% vs 40-67%) on this task.
The LLM reasons better about trade-offs holistically than through
lossy intermediate extraction.
"""

from secretagent.core import interface
from ptools_common import raw_answer, extract_index


@interface
def answer_question(narrative: str, question: str, choices: list) -> int:
    """Read the narrative and determine the best team allocation.
    Return the 0-based index of the correct choice.
    """
    text = raw_answer(narrative, question, choices)
    return extract_index(text, choices)
