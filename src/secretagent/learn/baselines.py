"""Baseline learners that learn implementations from recorded interface calls.
"""

import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import typer

from secretagent import config, savefile
from secretagent.dataset import Dataset
from secretagent.learn.utils import collect_interface_data

def _make_hashable(obj):
    """Convert a JSON-decoded object to a hashable form."""
    if isinstance(obj, list):
        return tuple(_make_hashable(x) for x in obj)
    if isinstance(obj, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))
    return obj

class RoteLearner:
    """Learns a function that returns most commonly seen output for
    each input.
    """

    def __init__(self, dataset: Dataset):
        # for each possible input, saves a Counter of possible output
        # values with their frequencies
        self.counts = defaultdict(Counter)
        self._original_output = {}  # hashable_output -> original output
        for case in dataset.cases:
            args_key = _make_hashable(case.input_args or [])
            kw_key = _make_hashable(case.input_kw or {})
            input_key = (args_key, kw_key)
            output_key = _make_hashable(case.expected_output)
            self.counts[input_key][output_key] += 1
            self._original_output[output_key] = case.expected_output

    def fit(self, outdir: str | Path, interface_name: str) -> Path:
        """Write a learned.py file with a function that returns the most common output.

        The generated function accepts *args, **kw and looks up the input
        in a precomputed dict, returning the most common output or None.
        """
        # Build mapping from input_key -> most_common_output in original form
        most_common_output = {}
        for input_key, counter in self.counts.items():
            best_output, _ = counter.most_common(1)[0]
            most_common_output[input_key] = self._original_output[best_output]

        outpath = outdir / 'learned.py'
        outpath.write_text(textwrap.dedent(f"""\
            \"\"\"Auto-generated rote-learned implementation for {interface_name}.\"\"\"

            def _make_hashable(obj):
                if isinstance(obj, list):
                    return tuple(_make_hashable(x) for x in obj)
                if isinstance(obj, dict):
                    return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))
                return obj

            _MOST_COMMON_OUTPUT = {repr(most_common_output)}

            def {interface_name}(*args, **kw):
                args_key = _make_hashable(list(args))
                kw_key = _make_hashable(kw)
                return _MOST_COMMON_OUTPUT.get((args_key, kw_key))
        """))
        return outpath

    def report(self) -> list[dict]:
        """Brief report on likely rote-learning performance.
        """
        total = sum(ctr.total() for ctr in self.counts.values())
        total_non_singleton = sum(ctr.total() for ctr in self.counts.values() if ctr.total()!=1)
        return textwrap.dedent(f"""\
           inputs:             {len(self.counts)}
           estimated coverage: {total_non_singleton/total:.2f}""")


app = typer.Typer()
_EXTRA_ARGS = {"allow_extra_args": True, "allow_interspersed_args": False}
@app.command(context_settings=_EXTRA_ARGS)
def rote(
    ctx: typer.Context,
    interface: str = typer.Option(..., help="Interface name to extract, e.g. 'consistent_sports'"),
    latest: int = typer.Option(1, help='Keep latest k dirs per tag; 0 for all'),
    check: Optional[list[str]] = typer.Option(None, help='Config constraint like key=value'),
    train_dir: str = typer.Option('/tmp/rote_train', help='Directory to store collected data'),
):
    """Learn input/output statistics from recorded interface calls.

    Recording directories are passed as extra positional arguments.
    They are filtered through savefile.filter_paths().
    """
    config.configure(cfg={'train_dir': train_dir})
    dirs = savefile.filter_paths(ctx.args, latest=latest, dotlist=check or [])
    out_dir, dataset = collect_interface_data(dirs, interface, file_under=f'{interface}.rote')
    print(f'collected {len(dataset.cases)} examples in working directory {out_dir}')
    learner = RoteLearner(dataset)
    output_file = learner.fit(out_dir, interface)
    print(learner.report())
    print(f'saved output to {output_file}')

if __name__ == '__main__':
    app()
