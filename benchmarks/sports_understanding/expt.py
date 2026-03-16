"""Sports understanding benchmark experiment.

Example CLI commands:

    # run with defaults from conf/conf.yaml
    uv run python expt.py run

    # run first 6 examples only
    uv run python expt.py run --n 6

    # override model and experiment name
    uv run python expt.py run --model gpt-4o --expt-name gpt4o_test

    # use a different config file
    uv run python expt.py run --config-file conf/ablation.yaml

    # override dataset split
    uv run python expt.py run --split test
"""

import json
import pandas as pd
from pathlib import Path
import pprint
import re
from typing import Any

import typer

from secretagent import record, config
from secretagent.core import Interface, implement_via_config
from secretagent import implement_pydantic # force registry of simulate_pydantic factory
from secretagent.dataset import Dataset, Case
from secretagent.evaluate import Evaluator
#
# tools are the tools and interfaces
#

import ptools

#
# dataset-specific metrics
#

class SportsUnderstandingEvaluator(Evaluator):
    def compare_predictions(self, predicted_output, expected_output) -> dict[str, Any]:
        return dict(correct=float(predicted_output == expected_output))

#
# how to load the dataset
#

def load_dataset(split: str) -> Dataset:
    def example_as_case(index, split, example):
        return Case(
            name=f'{split}.{index:03d}',
            input_args=(re.search(r'"([^"]*)"', example['input']).group(1),),
            expected_output=(example['target']=="yes")
        )
    json_file = Path(__file__).parent / 'data' / f'{split}.json'
    with open(json_file) as fp:
        data = json.load(fp)
        examples = data['examples']
        return Dataset(
            name='sports_understanding',
            split=split,
            cases=[
                example_as_case(i, split, ex)
                for i, ex in enumerate(examples)
            ],
        )

#
# machinery to support using this file as a CLI
#

app = typer.Typer()

@app.callback()
def callback():
    """Sports understanding benchmark.

    This callback ensures typer treats the app as a multi-command CLI
    rather than collapsing a single subcommand to the top level.
    """

CONF_DIR = Path(__file__).parent / 'conf'

@app.command(context_settings={"allow_extra_args": True, "allow_interspersed_args": False})
def run(ctx: typer.Context, expt_name: str = typer.Option(None, help="Set evaluate.expt_name")):
    """Run sports understanding evaluation.

    Extra args are parsed as config overrides in dot notation, e.g.:
        uv run python expt.py run llm.model=gpt-4o cachier.enable_caching=false
    """

    # configure with conf/conf.yaml plus any command-line args
    config_file =  Path(__file__).parent / 'conf' / 'conf.yaml'
    config.configure(yaml_file=config_file, dotlist=ctx.args)

    # make the cache_dir and etc be relative to this directory
    config.set_root(Path(__file__).parent)

    # load the dataset, following the config
    dataset = load_dataset(config.require('dataset.split')).configure(
        shuffle_seed=config.get('dataset.shuffle_seed'),
        n=config.get('dataset.n') or None  # don't pass in 0
        )
    print('dataset is', dataset.summary())

    # configure the ptools
    implement_via_config(ptools, config.require('ptools'))
    evaluator = SportsUnderstandingEvaluator()
    csv_path = evaluator.evaluate(dataset, ptools.are_sports_in_sentence_consistent)
    df = pd.read_csv(csv_path)
    print(df)

@app.command(context_settings={"allow_extra_args": True, "allow_interspersed_args": False})
def quick_test(ctx: typer.Context, expt_name: str = typer.Option(None, help="Set evaluate.expt_name")):
    config_file =  Path(__file__).parent / 'conf' / 'conf.yaml'
    config.configure(yaml_file=config_file, dotlist=ctx.args)
    config.set_root(Path(__file__).parent)
    pprint.pprint(config.GLOBAL_CONFIG)

    # load the dataset, following the config
    dataset = load_dataset(config.require('dataset.split')).configure(
        shuffle_seed=config.get('dataset.shuffle_seed'),
        n=config.get('dataset.n') or None  # don't pass in 0
        )
    print('dataset is', dataset.summary())

    # configure the ptools
    implement_via_config(ptools, config.require('ptools'))

    input_args = dataset.cases[0].input_args
    print('input_args', input_args)
    with config.configuration(
            cachier={'enable_caching':False},
            echo={
                'service': True, 'llm_input': True, 'llm_output': True, 'code_eval_input': True, 'code_eval_output': True}):
        with record.recorder() as records:
            predicted_output = ptools.are_sports_in_sentence_consistent(*input_args)

    print('predicted output', predicted_output)
    pprint.pprint(records)

if __name__ == '__main__':
    app()
