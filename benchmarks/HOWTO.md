# Setting up a benchmark

Obviously this is still in progress....

## Directory structure

Your directory structure will look something like this:

├── conf
│   └── conf.yaml  # common configuration stuff
├── data
│   ├── ...
│   ├── test.json
│   ├── train.json
│   └── valid.json
├── expt.py
├── llm_cache
├── Makefile
├── prompt_templates
│   └── zeroshot.txt
├── ptools.py
└── results
    ├── 20260316.183513.workflow
    │   ├── config.yaml
    │   ├── results.csv
    │   └── results.jsonl
    ├└── 20260316.183519.pot
    ... ├── config.yaml
        ├── results.csv
        └── results.jsonl

