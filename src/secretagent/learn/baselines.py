"""Baseline learners that learn implementations from recorded interface calls.
"""

import inspect
import pprint
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import typer

from secretagent.learn.core import Learner


def _make_hashable(obj):
    """Convert a JSON-decoded object to a hashable form."""
    if isinstance(obj, list):
        return tuple(_make_hashable(x) for x in obj)
    if isinstance(obj, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))
    return obj


class RoteLearner(Learner):
    """Learns a function that returns most commonly seen output for
    each input.
    """

    def __init__(self, interface_name, dirs, train_dir, *,
                 latest=1, check=None):
        super().__init__(
            interface_name=interface_name,
            dirs=dirs,
            train_dir=train_dir,
            file_under=f'{interface_name}.rote',
            latest=latest,
            check=check,
        )

    def fit(self):
        """Compute the most common output for each input."""
        # for each possible input, count output frequencies
        counts = defaultdict(Counter)
        original_output = {}  # hashable_output -> original output
        for case in self.dataset.cases:
            args_key = _make_hashable(case.input_args or [])
            kw_key = _make_hashable(case.input_kw or {})
            input_key = (args_key, kw_key)
            output_key = _make_hashable(case.expected_output)
            counts[input_key][output_key] += 1
            original_output[output_key] = case.expected_output
        # pick the most common output for each input
        self._most_common_output = {}
        for input_key, counter in counts.items():
            best_output, _ = counter.most_common(1)[0]
            self._most_common_output[input_key] = original_output[best_output]
        self.counts = counts

    def save_code(self) -> Path:
        """Write a learned.py file with a function that returns the most common output.

        The generated function accepts *args, **kw and looks up the input
        in a precomputed dict, returning the most common output or None.
        """
        hashable_src = inspect.getsource(_make_hashable)
        outpath = self.out_dir / 'learned.py'
        outpath.write_text(
            f'"""Auto-generated rote-learned implementation for {self.interface_name}."""\n\n'
            f'{hashable_src}\n'
            f'_MOST_COMMON_OUTPUT = {pprint.pformat(self._most_common_output)}\n\n'
            f'def {self.interface_name}(*args, **kw):\n'
            f'    args_key = _make_hashable(list(args))\n'
            f'    kw_key = _make_hashable(kw)\n'
            f'    return _MOST_COMMON_OUTPUT.get((args_key, kw_key))\n'
        )
        return outpath

    def report(self) -> str:
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
    learner = RoteLearner(
        interface_name=interface,
        dirs=[Path(p) for p in ctx.args],
        train_dir=train_dir,
        latest=latest,
        check=check,
    )
    print(f'collected {len(learner.dataset.cases)} examples in working directory {learner.out_dir}')
    learner.fit()
    output_file = learner.save_code()
    print(learner.report())
    print(f'saved output to {output_file}')

if __name__ == '__main__':
    app()
