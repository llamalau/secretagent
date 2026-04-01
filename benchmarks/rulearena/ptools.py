"""RuleArena benchmark tools and workflows.

Domains: airline (baggage fees), tax (US federal income tax), nba (CBA compliance).
Experiment levels:
  L0  - oracle: ground truth params fed directly to Python calculators
  L0F - CoT: single LLM call per domain with chain-of-thought prompt
  L1  - extract: LLM extracts structured params, Python computes answer
  L3  - ReAct: autonomous agent with extraction and calculator tools

Example CLI commands:
    uv run expt.py run --help
    uv run expt.py run evaluate.expt_name=l1_airline dataset.domain=airline \
        ptools.compute_rulearena_answer.method=direct \
        ptools.compute_rulearena_answer.fn=ptools.l1_extract_workflow
"""

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

from secretagent.core import interface, implement_via

_DATA_DIR = Path(__file__).parent / "data"

# ---------------------------------------------------------------------------
# LLM extraction interfaces
# ---------------------------------------------------------------------------

@interface
def extract_airline_params(query: str) -> dict:
    """Extract structured baggage parameters from an airline fee query.

    Return a JSON object with these exact fields:
    {
        "base_price": <integer ticket price in USD>,
        "customer_class": <one of: "Basic Economy", "Main Cabin", "Main Plus",
                           "Premium Economy", "Business", "First">,
        "routine": <destination region, e.g. "U.S.", "Japan", "Europe">,
        "direction": <0 for departing from US, 1 for arriving to US>,
        "bag_list": [
            {"id": 1, "name": "<item>", "size": [length, width, height], "weight": <lbs>},
            ...
        ]
    }

    Requirements:
    - customer_class must be an exact match from the list above
    - routine must match the fee table region names (e.g. "U.S." with period)
    - direction: 0=from US, 1=to US
    - size: array of 3 integers [length, width, height] in inches
    - weight: integer in pounds

    Examples:
    >>> extract_airline_params("A Main Cabin passenger flies Chicago to Tokyo, $1200 ticket. Carry-on 22x14x9 15lbs, checked suitcase 28x20x12 45lbs.")
    {"base_price": 1200, "customer_class": "Main Cabin", "routine": "Japan", "direction": 0, "bag_list": [{"id": 1, "name": "carry-on", "size": [22, 14, 9], "weight": 15}, {"id": 2, "name": "suitcase", "size": [28, 20, 12], "weight": 45}]}
    """


@interface
def extract_tax_params(query: str) -> str:
    """Extract taxpayer parameters from filled IRS forms.

    The input is a set of IRS forms with dollar values already filled in and
    computed fields marked [__]. Extract the INPUT values that appear on the forms.

    Return a JSON object with these fields:
    {
        // --- Basic info ---
        "name": "<taxpayer name>",
        "age": <int>,
        "spouse_age": <int>,
        "filing_status": "<single|married filing jointly|married filing separately|head of household|qualifying surviving spouse>",
        "blind": <bool>,
        "spouse_blind": <bool>,
        "itemized": <bool>,
        "self_employed": <bool>,
        "has_student_loans_or_education_expenses": <bool>,
        "num_qualifying_children": <int>,
        "num_other_dependents": <int>,

        // --- Form 1040 income lines ---
        "wage_tip_compensation": <float>,
        "household_employee_wage": <float>,
        "unreported_tip": <float>,
        "nontaxable_combat_pay": <float>,
        "tax_exempt_interest": <float>,
        "taxable_interest": <float>,
        "qualified_dividends": <float>,
        "ordinary_dividends": <float>,
        "ira_distributions": <float>,
        "taxable_ira_distributions": <float>,
        "all_pensions": <float>,
        "taxable_pensions": <float>,
        "social_security_benefits": <float>,
        "taxable_social_security_benefits": <float>,
        "qualified_business_income": <float>,  // Line 13. Read 'Deduction from Form 8995 or Form 8995-A' value. Do NOT read [__] as zero.
        "federal_income_tax_withheld": <float>,
        "earned_income_credit": <float>,

        // --- Schedule 1 ---
        "taxable_state_refunds": <float>,
        "alimony_income": <float>,
        "sale_of_business": <float>,  // Line 4 'Other gains or (losses)'
        "rental_real_estate_sch1": <float>,
        "farm_income": <float>,
        "unemployment_compensation": <float>,
        "other_income": <float>,
        "educator_expenses": <float>,
        "hsa_deduction": <float>,
        "ira_deduction": <float>,
        "student_loan_interest_deduction": <float>,
        "other_adjustments": <float>,

        // --- Schedule 2 ---
        "amt_f6251": <float>,
        "credit_repayment": <float>,
        "other_additional_taxes": <float>,

        // --- Schedule 3 ---
        "foreign_tax_credit": <float>,
        "dependent_care": <float>,
        "retirement_savings": <float>,
        "elderly_disabled_credits": <float>,
        "plug_in_motor_vehicle": <float>,
        "alt_motor_vehicle": <float>,

        // --- Schedule A (only when itemized=true) ---
        "medical_dental_expenses": <float>,            // Line 1
        "state_local_income_or_sales_tax": <float>,    // Line 5a
        "state_local_real_estate_tax": <float>,         // Line 5b
        "state_local_personal_property_tax": <float>,   // Line 5c
        "other_taxes_paid": <float>,                    // Line 6
        "home_mortgage_interest_and_points": <float>,   // Line 8a
        "home_mortgage_interest_unreported": <float>,   // Line 8b
        "home_mortgage_points_unreported": <float>,     // Line 8c
        "investment_interest": <float>,                 // Line 9
        "charity_cash": <float>,                        // Line 11
        "charity_non_cash": <float>,                    // Line 12
        "casualty_and_theft_loss": <float>,             // Line 15
        "other_itemized_deductions": <float>,           // Line 16

        // --- Schedule C (only when self_employed=true) ---
        "gross_receipts": <float>,            // Line 1
        "returns_and_allowances": <float>,    // Line 2
        "cost_of_goods_sold": <float>,        // Line 4
        "other_inc_sched_c": <float>,         // Line 6
        "total_expenses": <float>,            // Line 28
        "expenses_of_home": <float>,          // Line 30
        "total_social_security_wages": <float>, // Schedule SE Line 8a (or W-2 boxes 3+7)

        // --- Form 8863 education (only when has_student_loans_or_education_expenses=true) ---
        "student_list": [   // One entry per student in Part III
            {
                "qualified_student_expenses": <int>,    // Part III Line 21
                "f8863_part_iii_23": "<Yes or No>",     // Line 23
                "f8863_part_iii_24": "<Yes or No>",     // Line 24
                "f8863_part_iii_25": "<Yes or No>",     // Line 25
                "f8863_part_iii_26": "<Yes or No>"      // Line 26
            }
        ]
    }

    Extraction rules:
    - Dollar values on forms appear as "$1,234" -- extract as numeric (1234.0).
    - Fields for schedules not present in the forms should be 0, 0.0, or [].
    - self_employed = true if Schedule C is present.
    - has_student_loans_or_education_expenses = true if Form 8863 is present.
    - itemized = true if Schedule A is present.
    - student_list = [] if Form 8863 is not present.

    Examples:
    >>> extract_tax_params("Name: Jane, Age: 35, Filing Status: single, Line 1a W-2: $50,000")
    {"name": "Jane", "age": 35, "filing_status": "single", "wage_tip_compensation": 50000.0, "blind": false, "spouse_blind": false, "itemized": false, "self_employed": false, "has_student_loans_or_education_expenses": false, "num_qualifying_children": 0, "num_other_dependents": 0, "spouse_age": 0, "household_employee_wage": 0.0, "unreported_tip": 0.0, "nontaxable_combat_pay": 0.0, "tax_exempt_interest": 0.0, "taxable_interest": 0.0, "qualified_dividends": 0.0, "ordinary_dividends": 0.0, "ira_distributions": 0.0, "taxable_ira_distributions": 0.0, "all_pensions": 0.0, "taxable_pensions": 0.0, "social_security_benefits": 0.0, "taxable_social_security_benefits": 0.0, "qualified_business_income": 0.0, "federal_income_tax_withheld": 0.0, "earned_income_credit": 0.0, "taxable_state_refunds": 0.0, "alimony_income": 0.0, "sale_of_business": 0.0, "rental_real_estate_sch1": 0.0, "farm_income": 0.0, "unemployment_compensation": 0.0, "other_income": 0.0, "educator_expenses": 0.0, "hsa_deduction": 0.0, "ira_deduction": 0.0, "student_loan_interest_deduction": 0.0, "other_adjustments": 0.0, "amt_f6251": 0.0, "credit_repayment": 0.0, "other_additional_taxes": 0.0, "foreign_tax_credit": 0.0, "dependent_care": 0.0, "retirement_savings": 0.0, "elderly_disabled_credits": 0.0, "plug_in_motor_vehicle": 0.0, "alt_motor_vehicle": 0.0, "student_list": []}
    """


@interface
def extract_nba_params(query: str) -> dict:
    """Determine whether any NBA team operation violates CBA salary cap rules.

    Given reference rules, team situations, player situations, and proposed
    operations, analyze each operation against the CBA rules.

    Return a JSON object:
    {
        "verdict": <true if any operation violates rules, false if all compliant>,
        "illegal_operation": "<letter of the violating operation, or empty string>",
        "problematic_team": "<letter of the violating team, or empty string>",
        "reasoning": "<brief explanation>"
    }

    Examples:
    >>> extract_nba_params("Rules: [cap $140M] Team A salary $130M. Operations: A. Team A signs Player A at $15M/year.")
    {"verdict": false, "illegal_operation": "", "problematic_team": "", "reasoning": "Within cap"}
    """


# ---------------------------------------------------------------------------
# Calculator interfaces (always Python via direct method)
# ---------------------------------------------------------------------------

@interface
def compute_airline_calculator(params: dict) -> int:
    """Compute airline baggage fee and total ticket cost.

    Pass the dict returned by extract_airline_params directly as params.
    Required keys: base_price, customer_class, routine, direction, bag_list.

    Returns:
        Total cost (ticket price + baggage fees) as integer dollars.

    Examples:
    >>> compute_airline_calculator({"base_price": 500, "customer_class": "Main Cabin", "routine": "U.S.", "direction": 0, "bag_list": [{"id": 1, "name": "carry-on", "size": [22, 14, 9], "weight": 10}]})
    500
    """


@interface
def compute_tax_calculator(params: dict) -> float:
    """Compute federal tax amount from extracted TaxPayer fields.

    Pass the dict returned by extract_tax_params directly as params.
    Optional schedule fields are defaulted to 0 if absent.

    Returns:
        Amount owed (positive) or overpaid/refund (negative) as float.

    Examples:
    >>> compute_tax_calculator({"filing_status": "single", "age": 35, "wage_tip_compensation": 50000.0})
    5000.0
    """


# ---------------------------------------------------------------------------
# CoT prompt interfaces (always prompt_llm, bound at definition time)
# ---------------------------------------------------------------------------

@implement_via('prompt_llm', prompt_template_file='prompt_templates/airline_cot.txt')
def _cot_airline(problem_text: str, rules_text: str) -> str:
    ...


@implement_via('prompt_llm', prompt_template_file='prompt_templates/tax_cot.txt')
def _cot_tax(forms_text: str) -> str:
    ...


@implement_via('prompt_llm', prompt_template_file='prompt_templates/nba_cot.txt')
def _cot_nba(problem_text: str, rules_text: str) -> str:
    ...


@implement_via('simulate')
def _parse_numeric_answer(llm_output: str) -> float:
    """Extract the numeric dollar amount from LLM text output.

    The output may contain "The total cost is $1,234." or
    "The total tax owed is $567." or "The total tax overpaid is $89."
    For overpaid amounts, return the value as negative.
    """
    ...


@implement_via('simulate')
def _parse_bool_answer(llm_output: str) -> bool:
    """Extract True or False from LLM text output.

    The output contains either "Answer: True." or "Answer: False."
    """
    ...


# ---------------------------------------------------------------------------
# Python calculator implementations (called from workflows, not interfaces)
# ---------------------------------------------------------------------------

# Region normalization for airline extraction
_VALID_REGIONS = {
    "U.S.", "Puerto Rico", "Canada", "Mexico", "Cuba", "Haiti", "Panama",
    "Colombia", "Ecuador", "Peru", "South America", "Israel", "Qatar",
    "Europe", "India", "China", "Japan", "South Korea", "Hong Kong",
    "Australia", "New Zealand",
}

_REGION_FIXES = {
    "asia": "China",
    "north america": "U.S.",
    "us": "U.S.",
    "usa": "U.S.",
    "united states": "U.S.",
    "domestic": "U.S.",
    "tokyo": "Japan",
    "beijing": "China",
    "shanghai": "China",
    "seoul": "South Korea",
    "sydney": "Australia",
    "london": "Europe",
    "paris": "Europe",
    "berlin": "Europe",
}


def _normalize_region(routine: str) -> str:
    if routine in _VALID_REGIONS:
        return routine
    fixed = _REGION_FIXES.get(routine.lower().strip())
    return fixed if fixed else "U.S."


def _airline_calc_fn(params: dict) -> int:
    from calculators.airline import compute_airline_fee
    p = dict(params)
    p["routine"] = _normalize_region(p.get("routine", "U.S."))
    return compute_airline_fee(p)


_SCHED_C_DEFAULTS = {
    "gross_receipts": 0.0, "returns_and_allowances": 0.0,
    "cost_of_goods_sold": 0.0, "other_inc_sched_c": 0.0,
    "total_expenses": 0.0, "expenses_of_home": 0.0,
    "total_social_security_wages": 0.0,
}
_SCHED_A_DEFAULTS = {
    "medical_dental_expenses": 0.0, "state_local_income_or_sales_tax": 0.0,
    "state_local_real_estate_tax": 0.0, "state_local_personal_property_tax": 0.0,
    "other_taxes_paid": 0.0, "home_mortgage_interest_and_points": 0.0,
    "home_mortgage_interest_unreported": 0.0, "home_mortgage_points_unreported": 0.0,
    "investment_interest": 0.0, "charity_cash": 0.0, "charity_non_cash": 0.0,
    "casualty_and_theft_loss": 0.0, "other_itemized_deductions": 0.0,
}
_EDU_DEFAULTS = {"student_list": []}


def _apply_tax_defaults(params: dict) -> dict:
    result = dict(params)
    for defaults in (_SCHED_C_DEFAULTS, _SCHED_A_DEFAULTS, _EDU_DEFAULTS):
        for k, v in defaults.items():
            result.setdefault(k, v)
    return result


def _tax_calc_fn(params: dict) -> float:
    from calculators.tax import compute_tax_fee
    result = compute_tax_fee({"pydantic": _apply_tax_defaults(params)})
    if result is None:
        raise RuntimeError("Tax calculator returned None")
    return result


# ---------------------------------------------------------------------------
# Tax forms query builder (needed for L0F and L1 tax domain)
# ---------------------------------------------------------------------------

_tax_prompt_module = None


def _get_tax_prompt_module():
    global _tax_prompt_module
    if _tax_prompt_module is not None:
        return _tax_prompt_module
    prompt_path = _DATA_DIR / "tax" / "prompt.py"
    spec = importlib.util.spec_from_file_location("tax_prompt", prompt_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _tax_prompt_module = mod
    return mod


_TAX_PROMPT_TEMPLATE = (
    "\nYou are given several forms used to report US income tax and the instructions "
    "or rules about how to fill the forms. Then you will be given the income and/or "
    "payment information about a tax payer According to the given information. You "
    "should calculate the income tax owed by this payer.\n\nIRS Forms for the tax "
    "payer:\n$forms\nCalculate the tax owed by the payer step-by-step according to "
    "the information provided by the forms. You should calculate all fields marked "
    "with [__]. DO NOT round numbers without explicit instructions. End your response "
    "with <answer>xxx</answer> where xxx is the total tax amount as a number "
    "(negative if overpaid/refunded).\nYour response:\n"
)


def build_tax_forms_query(metadata: dict) -> str:
    """Build the filled IRS forms prompt from taxpayer metadata dict.

    Replicates the logic from external/RuleArena/tax/auto_test.py.
    """
    mod = _get_tax_prompt_module()
    tax_payer = metadata["dict"]

    forms = [mod.basic_forms]
    if tax_payer["itemized"]:
        forms.append(mod.itemized_forms)
    if tax_payer["self_employed"]:
        forms.append(mod.self_employ_forms)
    if tax_payer["has_student_loans_or_education_expenses"]:
        forms.append(mod.edu_forms)
    if tax_payer["child_and_dependent"]:
        forms.append(mod.schedule_8812)
    forms = "".join(forms)

    for k, v in tax_payer["data"].items():
        if isinstance(v, str):
            forms = forms.replace("$" + k, v)
        else:
            forms = forms.replace("$" + k, "$" + f"{v:,}")

    forms = forms.replace("$TBD", "[__]")

    prompt = _TAX_PROMPT_TEMPLATE.replace("$forms", forms)
    prompt = prompt.replace("$name", tax_payer["name"])
    prompt = prompt.replace("$age", str(tax_payer["age"]))
    prompt = prompt.replace("$spouse_age", str(tax_payer["spouse_age"]))
    prompt = prompt.replace("$blind", str(tax_payer["blind"]))
    prompt = prompt.replace("$spouse_blind", str(tax_payer["spouse_blind"]))
    prompt = prompt.replace("$filing_status", tax_payer["filing_status"])
    prompt = prompt.replace("$itemized", str(tax_payer["itemized"]))
    prompt = prompt.replace("$num_qualifying_children", str(tax_payer["num_qualifying_children"]))
    prompt = prompt.replace("$num_other_dependents", str(tax_payer["num_other_dependents"]))
    return prompt


# ---------------------------------------------------------------------------
# NBA query builder
# ---------------------------------------------------------------------------

_NBA_ASSUMPTIONS = (
    "Assume:\n"
    "* the Salary Cap for the prior (2023-24) Salary Cap Year is $136,000,000;\n"
    "* the Average Player Salary for the prior (2023-24) Salary Cap Year is $9,700,000;\n"
    "* the Salary Cap for the current (2024-25) NBA Salary Cap Year is $140,588,000;\n"
    "* the Luxury Tax is $170,814,000;\n"
    "* the First Apron Level is $178,132,000;\n"
    "* the Second Apron Level is $188,931,000;\n"
    "* the Team Salary of each team listed under \"Team Situations:\" do not "
    "include the amount of contracts that expire at the end of 2023-2024 Salary Cap Year.\n"
)


def _build_nba_query(problem_text: str, rules_text: str, metadata: dict) -> str:
    team_info = "Team Situations:\n" + "\n".join(metadata.get("team_situations", []))
    player_info = "Player Situations:\n" + "\n".join(metadata.get("player_situations", []))
    operations = "Operations:\n" + "\n".join(metadata.get("operations", []))
    question = team_info + "\n\n" + player_info + "\n\n" + operations

    return (
        f"Reference Rules in NBA Collective Bargaining Agreement:\n\n"
        f"{rules_text}\n\n"
        f"{_NBA_ASSUMPTIONS}\n"
        f"Decide whether any operation by any team violates the rules:\n\n"
        f"{question}"
    )


# ---------------------------------------------------------------------------
# Workflow functions
# ---------------------------------------------------------------------------

def l0_oracle_workflow(
    problem_text: str, domain: str, rules_text: str,
    metadata_json: str, forms_text: str
) -> float:
    """Oracle workflow: feed ground-truth params directly to Python calculators.

    Zero LLM calls. Establishes the accuracy ceiling for each domain.
    NBA returns 1.0 for violation, 0.0 for compliant.
    """
    metadata = json.loads(metadata_json)
    if domain == "airline":
        return float(_airline_calc_fn(metadata))
    if domain == "tax":
        return _tax_calc_fn(metadata.get("pydantic", metadata))
    if domain == "nba":
        return float(bool(metadata.get("answer", False)))
    raise ValueError(f"Unknown domain: {domain!r}")


def l0f_cot_workflow(
    problem_text: str, domain: str, rules_text: str,
    metadata_json: str, forms_text: str
) -> float:
    """CoT baseline: single LLM call per domain with structured prompt, then parse answer.

    Replicates the original RuleArena paper's CoT evaluation methodology.
    """
    if domain == "airline":
        raw = _cot_airline(problem_text=problem_text, rules_text=rules_text)
        return _parse_numeric_answer(raw)
    if domain == "tax":
        raw = _cot_tax(forms_text=forms_text)
        return _parse_numeric_answer(raw)
    if domain == "nba":
        metadata = json.loads(metadata_json)
        nba_query = _build_nba_query(problem_text, rules_text, metadata)
        raw = _cot_nba(problem_text=nba_query, rules_text="")
        return float(_parse_bool_answer(raw))
    raise ValueError(f"Unknown domain: {domain!r}")


def l1_extract_workflow(
    problem_text: str, domain: str, rules_text: str,
    metadata_json: str, forms_text: str
) -> float:
    """L1 PTool workflow: LLM extracts structured params, Python computes answer.

    Phase 1: call domain-specific extraction interface (LLM via simulate).
    Phase 2: call domain-specific Python calculator.
    """
    metadata = json.loads(metadata_json)
    if domain == "airline":
        query = f"RULES:\n{rules_text}\n\nQUERY:\n{problem_text}"
        params = extract_airline_params(query)
        return float(_airline_calc_fn(params))

    if domain == "tax":
        raw = extract_tax_params(forms_text)
        # Return type is str to avoid ast.literal_eval failures on JSON
        # (true/false, trailing commas, // comments). Parse with json.loads
        # after stripping JS-style comments.
        import re as _re
        cleaned = _re.sub(r'//.*', '', raw)
        params = json.loads(cleaned)
        if not isinstance(params, dict):
            raise TypeError(f"extract_tax_params returned {type(params).__name__}, expected dict")
        return _tax_calc_fn(params)

    if domain == "nba":
        query = _build_nba_query(problem_text, rules_text, metadata)
        result = extract_nba_params(query)
        if isinstance(result, dict):
            # Handle case where result has NBA verdict fields
            if "verdict" in result:
                return float(bool(result["verdict"]))
            # Fallback: regex extraction from string representation
            raw = str(result)
            m = re.search(r'"verdict"\s*:\s*(true|false)', raw, re.IGNORECASE)
            if m:
                return float(m.group(1).lower() == "true")
        return float(bool(result))

    raise ValueError(f"Unknown domain: {domain!r}")


# ---------------------------------------------------------------------------
# Top-level interface
# ---------------------------------------------------------------------------

@interface
def compute_rulearena_answer(
    problem_text: str,
    domain: str,
    rules_text: str,
    metadata_json: str,
    forms_text: str,
) -> float:
    """Compute the answer for a RuleArena benchmark problem.

    Returns the numeric answer: dollar amount for airline and tax domains,
    or 1.0 (violation) / 0.0 (compliant) for the NBA domain.

    Args:
        problem_text: Natural language problem statement.
        domain: One of "airline", "tax", "nba".
        rules_text: Domain rules text provided to the model.
        metadata_json: JSON-encoded ground truth metadata (used by L0 oracle).
        forms_text: Pre-built IRS forms text for tax domain; empty for others.
    """
    ...
