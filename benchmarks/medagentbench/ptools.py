"""MedAgentBench ptools: interface for solving medical EHR tasks via FHIR API.

The single top-level interface uses simulate_pydantic with fhir_get/fhir_post
as tools. The FHIR API reference and base URL are passed via the context
argument at runtime (injected during dataset loading in expt.py).
"""

from secretagent.core import interface


@interface
def solve_medical_task(instruction: str, context: str) -> list[str]:
    """You are an expert in using FHIR functions to assist medical professionals.
    You are given a question and a set of possible functions.
    Based on the question, you will need to make one or more function/tool calls
    to achieve the purpose.

    You have access to two tools:
    - fhir_get(url): Send a GET request to the FHIR server. The url should be
      the full FHIR endpoint with query parameters.
    - fhir_post(url, payload): Send a POST request. url is the FHIR endpoint,
      payload is a JSON string with the resource data.

    The context argument contains the FHIR API base URL, available FHIR
    endpoint definitions, and any task-specific context (timestamps, lab codes,
    dosing instructions, etc.).

    Use the tools to interact with the FHIR server and solve the given task.
    Return your final answers as a list of strings.
    """
    ...
