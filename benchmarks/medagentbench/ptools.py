"""MedAgentBench ptools: interfaces for solving medical EHR tasks via FHIR API.

Supports four experiment levels:
  - baseline: simulate_pydantic with fhir_get + fhir_post tools (paper's protocol)
  - simulate: LLM predicts answer from docstring alone (no FHIR access)
  - pot: program_of_thought generates Python code with FHIR tool access
  - pipeline: Python-orchestrated read→act stages with specialist sub-interfaces
"""

from secretagent.core import interface


# ──────────────────────────────────────────────────────────────────────
# Main entry point (used by all experiments)
# ──────────────────────────────────────────────────────────────────────

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

    IMPORTANT: Return ONLY the exact values requested, nothing else.
    - For patient lookups: just the MRN (e.g. ["S6534835"])
    - For numeric values: just the number (e.g. ["28"] or ["2.3"])
    - For dates/times: just the ISO timestamp
    - Do NOT include explanations, descriptions, or units in the answer list.
    - If no value is found, return ["-1"].
    """
    ...


# ──────────────────────────────────────────────────────────────────────
# FHIR tool interfaces for PoT (L3)
#
# These are @interface wrappers around the plain fhir_tools functions.
# PoTFactory only includes tools in the prompt if they are Interface
# objects with implementations. Plain callables are silently omitted
# from the prompt's tool stub listing.
# ──────────────────────────────────────────────────────────────────────

@interface
def fhir_get_iface(url: str) -> str:
    """Send a GET request to the FHIR server.

    The url should be the full FHIR endpoint with query parameters,
    e.g. "http://localhost:8080/fhir/Patient?family=Smith&birthdate=1990-01-01"

    The response is returned as a JSON string. Use json.loads() to parse it.
    Returns an error message string if the request fails.
    """
    ...


@interface
def fhir_post_iface(url: str, payload: str) -> str:
    """Send a POST request to create a FHIR resource.

    The url is the FHIR endpoint, e.g. "http://localhost:8080/fhir/MedicationRequest".
    The payload must be a JSON string with the resource data.

    Returns "POST request accepted and executed successfully..." on success,
    or an error message if the JSON payload is invalid.
    """
    ...


# ──────────────────────────────────────────────────────────────────────
# Sub-interfaces for pipeline experiment (L4)
# ──────────────────────────────────────────────────────────────────────

@interface
def search_fhir_data(instruction: str, context: str) -> str:
    """Read phase: query the FHIR server to gather all information needed
    for the given medical task.

    You have access to fhir_get(url) to send GET requests to the FHIR server.
    The context contains the FHIR API base URL and available endpoint definitions.

    Make one or more GET requests to retrieve patient demographics, lab results,
    vitals, medication orders, or other relevant data. Do NOT make any POST
    requests — this is a read-only phase.

    Return a structured summary including:
    - Exact numeric values with units (e.g. "magnesium: 1.3 mg/dL")
    - Exact timestamps in ISO format
    - Patient FHIR resource references (e.g. "Patient/S6315806")
    - Any relevant status codes or identifiers
    Be precise — downstream processing depends on exact values.
    """
    ...


@interface
def act_on_results(instruction: str, context: str, search_results: str) -> list[str]:
    """Act phase: given the FHIR search results, determine what actions are
    needed and return the final answers.

    You have access to both fhir_get(url) and fhir_post(url, payload) tools.
    Use fhir_get if you need additional data not covered by search_results.
    Use fhir_post to create FHIR resources (medication orders, service
    requests, vital observations, etc.).

    Based on the instruction and search results:
    1. Decide if any POST operations are needed (e.g., ordering medications,
       creating referrals, recording vitals)
    2. If POST operations are needed, construct the correct FHIR resource
       payloads and submit them
    3. Return the final answers as a list of strings

    Only make POST requests if the task explicitly requires creating or
    ordering something. For read-only tasks, just return the answer.

    IMPORTANT: Return ONLY the exact values requested, nothing else.
    - For numeric values: just the number (e.g. ["28"] or ["2.3"])
    - Do NOT include explanations, descriptions, or units.
    - If no value is found, return ["-1"].
    """
    ...


@interface
def simulate_medical_task(instruction: str, context: str) -> list[str]:
    """Fallback: solve a medical EHR task using FHIR tools.

    Same as solve_medical_task — used as a fallback when the pipeline
    fails. Bound to simulate_pydantic with both fhir_get and fhir_post.

    IMPORTANT: Return ONLY the exact values requested, nothing else.
    - For numeric values: just the number (e.g. ["28"] or ["2.3"])
    - Do NOT include explanations, descriptions, or units.
    - If no value is found, return ["-1"].
    """
    ...


# ──────────────────────────────────────────────────────────────────────
# Pipeline workflow (L4)
# ──────────────────────────────────────────────────────────────────────

def pipeline_workflow(instruction: str, context: str) -> list[str]:
    """L4 pipeline: Python-orchestrated read→act stages.

    Stage 1 (Read): search_fhir_data gathers patient data via GET requests
    Stage 2 (Act): act_on_results decides actions and returns answers
    Fallback: simulate_medical_task if pipeline fails
    """
    try:
        # Stage 1: Read — gather all FHIR data
        search_results = search_fhir_data(instruction, context)

        # Stage 2: Act — decide and execute actions, return answers
        answers = act_on_results(instruction, context, search_results)
        return answers
    except Exception as ex:
        print(f'[pipeline] stage failed ({ex}), falling back to simulate')
        return simulate_medical_task(instruction, context)
