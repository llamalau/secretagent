"""RuleArena benchmark experiment.

Example CLI commands:

    # run L1 on airline domain (30 instances, validation split)
    uv run python expt.py run evaluate.expt_name=l1_airline dataset.domain=airline \
        ptools.compute_rulearena_answer.method=direct \
        ptools.compute_rulearena_answer.fn=ptools.l1_extract_workflow

    # run L0 oracle on tax
    uv run python expt.py run evaluate.expt_name=l0_tax dataset.domain=tax \
        ptools.compute_rulearena_answer.method=direct \
        ptools.compute_rulearena_answer.fn=ptools.l0_oracle_workflow

    # run L3 ReAct on airline
    uv run python expt.py run evaluate.expt_name=l3_airline dataset.domain=airline \
        ptools.compute_rulearena_answer.method=simulate_pydantic \
        "ptools.compute_rulearena_answer.tools=[ptools.extract_airline_params,ptools.compute_airline_calculator]"

    # override model and number of instances
    uv run python expt.py run llm.model=claude-haiku-4-5-20251001 dataset.n=10

    # quick trace of first instance
    uv run python expt.py quick_test dataset.domain=airline \
        ptools.compute_rulearena_answer.method=direct \
        ptools.compute_rulearena_answer.fn=ptools.l1_extract_workflow
"""

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pprint
import typer

from secretagent import config, record
from secretagent.core import implement_via_config
from secretagent import implement_pydantic  # force registration
from secretagent.dataset import Dataset, Case
from secretagent.evaluate import Evaluator

import ptools

# Path to external RuleArena repository
_RULEARENA_PATH = Path(__file__).parent.parent.parent.parent / "external" / "RuleArena"


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def _within_tolerance(predicted: Any, expected: Any, tol: float = 0.01) -> bool:
    try:
        p = float(predicted)
        e = float(expected)
    except (TypeError, ValueError):
        return False
    if abs(e) < 1e-9:
        return abs(p - e) < 0.01
    return abs(p - e) / abs(e) <= tol


class RuleArenaEvaluator(Evaluator):
    def compare_predictions(self, predicted_output, expected_output) -> dict[str, Any]:
        if isinstance(expected_output, bool):
            correct = float(bool(predicted_output) == expected_output)
        else:
            correct = float(_within_tolerance(predicted_output, expected_output))
        return dict(correct=correct)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_rules(domain: str) -> str:
    if domain == "airline":
        rules_file = _RULEARENA_PATH / "airline" / "reference_rules_textual.txt"
    elif domain == "nba":
        rules_file = _RULEARENA_PATH / "nba" / "reference_rules.txt"
    elif domain == "tax":
        return ""
    else:
        raise ValueError(f"Unknown domain: {domain!r}")
    if rules_file.exists():
        return rules_file.read_text(encoding="utf-8")
    return f"Rules for {domain} domain"


def _compute_ground_truth(domain: str, problem_data: dict, metadata: dict) -> Any:
    if domain == "nba":
        return problem_data.get("answer")
    try:
        if domain == "airline":
            from calculators.airline import compute_airline_fee
            return compute_airline_fee(metadata)
        if domain == "tax":
            from calculators.tax import compute_tax_fee
            return compute_tax_fee(metadata)
    except Exception:
        pass
    return None


def _iter_domain(domain: str, complexity: str):
    """Yield (problem_data, metadata, complexity_level) for each instance."""
    levels = [0, 1, 2] if complexity == "all" else [int(complexity)]

    for level in levels:
        if domain == "airline":
            data_dir = _RULEARENA_PATH / "airline" / "synthesized_problems"
            problem_file = data_dir / f"comp_{level}.jsonl"
            if not problem_file.exists():
                continue
            with open(problem_file, encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    problem_data = json.loads(line)
                    metadata = problem_data.get("info", {})
                    yield idx, level, problem_data, metadata

        elif domain == "nba":
            data_dir = _RULEARENA_PATH / "nba" / "annotated_problems"
            problem_file = data_dir / f"comp_{level}.json"
            if not problem_file.exists():
                continue
            with open(problem_file, encoding="utf-8") as f:
                problems = json.load(f)
            for idx, problem_data in enumerate(problems):
                metadata = problem_data
                yield idx, level, problem_data, metadata

        elif domain == "tax":
            data_dir = _RULEARENA_PATH / "tax" / "synthesized_problems"
            problem_file = data_dir / f"comp_{level}.json"
            if not problem_file.exists():
                continue
            with open(problem_file, encoding="utf-8") as f:
                problems = json.load(f)
            for idx, problem_data in enumerate(problems):
                metadata = problem_data
                yield idx, level, problem_data, metadata


def load_dataset(domain: str, complexity: str = "all") -> Dataset:
    rules_text = _load_rules(domain)
    cases = []

    for idx, level, problem_data, metadata in _iter_domain(domain, complexity):
        instance_id = f"{domain}_{level}_{idx}"
        problem_text = problem_data.get("prompt", "")

        # Pre-build tax forms query so the prompt_llm template can use $forms_text
        if domain == "tax":
            try:
                forms_text = ptools.build_tax_forms_query(metadata)
                # Sanitize non-ASCII for portability
                forms_text = forms_text.encode("ascii", errors="replace").decode("ascii")
            except Exception:
                forms_text = problem_text
        else:
            forms_text = ""

        ground_truth = _compute_ground_truth(domain, problem_data, metadata)

        cases.append(Case(
            name=instance_id,
            input_args=(
                problem_text,
                domain,
                rules_text,
                json.dumps(metadata, default=str),
                forms_text,
            ),
            expected_output=ground_truth,
        ))

    return Dataset(name=f"rulearena_{domain}", split=complexity, cases=cases)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer()


@app.callback()
def callback():
    """RuleArena benchmark: rule-guided reasoning across airline, NBA, and tax domains."""


CONFIG_DIR = Path(__file__).parent / "conf"


@app.command(context_settings={"allow_extra_args": True, "allow_interspersed_args": False})
def run(ctx: typer.Context, expt_name: str = typer.Option(None, help="Set evaluate.expt_name")):
    """Run RuleArena evaluation.

    Extra args are parsed as config overrides in dot notation, e.g.:
        uv run python expt.py run llm.model=gpt-4o dataset.domain=nba dataset.n=10
    """
    config_file = Path(__file__).parent / "conf" / "conf.yaml"
    config.configure(yaml_file=config_file, dotlist=ctx.args)
    config.set_root(Path(__file__).parent)

    domain = config.require("dataset.domain")
    complexity = config.get("dataset.complexity") or "all"

    dataset = load_dataset(domain, complexity).configure(
        shuffle_seed=config.get("dataset.shuffle_seed"),
        n=config.get("dataset.n") or None,
    )
    print("dataset is", dataset.summary())

    implement_via_config(ptools, config.require("ptools"))

    evaluator = RuleArenaEvaluator()
    csv_path = evaluator.evaluate(dataset, ptools.compute_rulearena_answer)

    df = pd.read_csv(csv_path)
    print(df)
    print()
    print(df[["correct", "latency", "cost"]].mean())


@app.command(context_settings={"allow_extra_args": True, "allow_interspersed_args": False})
def quick_test(ctx: typer.Context, expt_name: str = typer.Option(None, help="Set evaluate.expt_name")):
    """Run a single instance with full trace output for debugging."""
    config_file = Path(__file__).parent / "conf" / "conf.yaml"
    config.configure(yaml_file=config_file, dotlist=ctx.args)
    config.set_root(Path(__file__).parent)
    pprint.pprint(config.GLOBAL_CONFIG)

    domain = config.require("dataset.domain")
    complexity = config.get("dataset.complexity") or "all"

    dataset = load_dataset(domain, complexity).configure(
        shuffle_seed=config.get("dataset.shuffle_seed"),
        n=config.get("dataset.n") or None,
    )
    print("dataset is", dataset.summary())

    implement_via_config(ptools, config.require("ptools"))

    input_args = dataset.cases[0].input_args
    print("input_args[:2]", input_args[:2])
    with config.configuration(
        cachier={"enable_caching": False},
        echo={
            "service": True,
            "llm_input": True,
            "llm_output": True,
            "code_eval_input": True,
            "code_eval_output": True,
        },
    ):
        with record.recorder() as records:
            predicted_output = ptools.compute_rulearena_answer(*input_args)

    print("predicted output", predicted_output)
    print("expected output ", dataset.cases[0].expected_output)
    pprint.pprint(records)


if __name__ == "__main__":
    app()
