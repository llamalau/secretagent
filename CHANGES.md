# Changes - March 22

There is a very simple and barely-tested example of a learner now, in
the Makefile targets of `benchmarks/sports_understanding`.

To learn a new implementation of some interface `foo` you need to

 * Run an evaluation on the **training** split using some
implementation strategy that includes running some implementation of
`foo`, and the `evaluate.record_details=true` flag set.  This
generates distillation data for `foo` (i.e., input-output pairs).
Those results are filed in a directort specified by
`evaluate.result_dir` (in the Makefile `recordings`).

* Run a cli to do the learning.  That will collect the distillation
data into a working directory specified by --train-dir (in the
Makefile, `training`) and write the result of learning as a Python
function under `training/*foo__rote`.

* Configure `foo` with the learned implementation, by using
`foo.implement_via('learned',learner='rote',backoff=True)` or the
corresponding config options.  This will use rote learning to bypass
the original implementation and back off to the implementation used to
generate distillation data when the rote-learned function returns
None.

# Changes - March 13-Mar 18

## Documentation and examples

In `benchmarks/sports_understanding` is an example of evaluating untrained
models for a simple task, with docs in `benchmarks/HOWTO.md`.


## LLM cost tracking 

New `costs` CLI tool with `extract_cached_stats` for analyzing LLM
spending from the cache
  - For example, running in benchmarks/sports_understanding: uv run python -m secretagent.cli.costs llm_cache
  generates this report:

```
  1091 cached LLM calls

       input_tokens  output_tokens      latency         cost
count   1091.000000    1091.000000  1091.000000  1091.000000
mean    1220.851512     197.437214     2.570375     0.002208
std     2341.485373     147.909843     2.503594     0.003005
min       79.000000      12.000000     0.377901     0.000306
25%      308.000000     125.000000     1.340060     0.000915
50%      343.000000     163.000000     1.677382     0.001105
75%      706.000000     207.000000     2.418103     0.001667
max     8375.000000     750.000000    16.032463     0.012125

Totals:
  input_tokens:  1331949
  output_tokens: 215404
  latency:       2804.3s
  cost:          $2.4090
```

## Analyzing results

Added filtering, `--most-recent` default, dotlist support, and config
comparison.

### Example of getting averages

```
% uv run python -m secretagent.cli.results average --metric correct --metric cost results/*
                                                correct                cost          
                                                   mean       sem      mean       sem
path                                                                                 
results/20260316.183513.workflow               0.973333  0.018728  0.004200  0.000115
results/20260317.131602.pot                    0.960000  0.022780  0.006137  0.000158
results/20260317.141836.zeroshot_structured    0.933333  0.028997  0.001071  0.000018
results/20260317.143224.zeroshot_unstructured  0.933333  0.028997  0.001525  0.000032
results/20260317.145854.react                  0.813333  0.045295  0.013779  0.000475
```

### Finding the differences in configs between experiments

```
% uv run python -m secretagent.cli.results compare-configs results/*
properties of results/20260317.131602.pot:
  evaluate.expt_name=pot
  ptools.are_sports_in_sentence_consistent.method=program_of_thought
  ptools.are_sports_in_sentence_consistent.tools=['ptools.analyze_sentence', 'ptools.sport_for', 'ptools.consistent_sports']
properties of results/20260317.145854.react:
  evaluate.expt_name=react
  ptools.are_sports_in_sentence_consistent.method=simulate_pydantic
  ptools.are_sports_in_sentence_consistent.tools=['ptools.analyze_sentence', 'ptools.sport_for', 'ptools.consistent_sports']
...
```


### Running paired tests

```
% uv run python -m secretagent.cli.results pair --metric correct --metric cost results/*
                                    _comparison   n  correct_t  correct_p     cost_t        cost_p
0                                  pot vs react  75   3.238021   0.001803 -15.994221  1.011472e-25
1                               pot vs workflow  75  -1.000000   0.320569  20.859531  1.019114e-32
2                    pot vs zeroshot_structured  75   0.704730   0.483190  32.051952  3.901321e-45
3                  pot vs zeroshot_unstructured  75   0.704730   0.483190  29.922488  4.408926e-43
4                             react vs workflow  75  -3.754363   0.000344  20.763785  1.363903e-32
5                  react vs zeroshot_structured  75  -2.111829   0.038077  26.699630  9.997003e-40
6                react vs zeroshot_unstructured  75  -2.111829   0.038077  25.270505  3.938396e-38
7               workflow vs zeroshot_structured  75   1.136089   0.259585  26.704393  9.878003e-40
8             workflow vs zeroshot_unstructured  75   1.136089   0.259585  23.529151  4.356200e-36
9  zeroshot_structured vs zeroshot_unstructured  75   0.000000   1.000000 -13.698474  5.518676e-22
```

### Other software changes

- **Program of Thoughts** (PoT) factory
- **Linting/typing** — Integrated ruff and mypy, fixed lint errors and
  type annotation bugs
- **Configuration of tools is consistent** for PoT and Pydantic React agent
- **More configuration can be done just from a config files**
- **New MUSR benchmark** — Added a benchmark harness with 3 splits and zero-shot chain-of-thought baseline
- **More tests** - up to 157 tests, now more nc lines of test code than nc lines of code in the core system
