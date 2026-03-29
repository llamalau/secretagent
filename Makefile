#
# code quality checks
#

test:
	uv run pytest tests/ -v

benchmark_tests:
	uv run pytest benchmarks/tests/ -v	

lint:
	uv run ruff check src 

typehints:
	time uv run mypy src --ignore-missing-imports

wc:
	wc src/secretagent/*.py
	echo 
	cloc src/secretagent/*.py

prechecks: test lint typehints


#
# examples
#

quickstart:
	uv run examples/quickstart.py

examples: quickstart
	uv run examples/sports_understanding.py
	uv run examples/sports_understanding_pydantic.py

