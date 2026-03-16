test:
	uv run pytest tests/ -v

wc:
	wc src/secretagent/*.py
	echo 
	cloc src/secretagent/*.py


quickstart:
	uv run examples/quickstart.py

examples: quickstart
	uv run examples/sports_understanding.py
	uv run examples/sports_understanding_pydantic.py

expt:
	time uv run benchmarks/sports_understanding/expt.py run

results:
	uv run -m secretagent.cli.results --help

costs:
	uv run -m secretagent.cli.costs benchmarks/sports_understanding/llm_cache

