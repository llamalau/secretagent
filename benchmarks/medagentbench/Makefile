docker-start:
	docker run -d -p 8080:8080 jyxsu6/medagentbench:latest

docker-stop:
	docker ps -q --filter ancestor=jyxsu6/medagentbench:latest | xargs -r docker stop

# ── Experiments ──────────────────────────────────────────────────────

baseline:
	uv run python expt.py run --config-file conf/baseline.yaml

simulate:
	uv run python expt.py run --config-file conf/simulate.yaml

pot:
	uv run python expt.py run --config-file conf/pot.yaml

pipeline:
	uv run python expt.py run --config-file conf/pipeline.yaml

# ── Quick tests (single case) ───────────────────────────────────────

quick-test:
	uv run python expt.py quick-test --config-file conf/baseline.yaml dataset.n=1

quick-test-simulate:
	uv run python expt.py quick-test --config-file conf/simulate.yaml dataset.n=1

quick-test-pot:
	uv run python expt.py quick-test --config-file conf/pot.yaml dataset.n=1

quick-test-pipeline:
	uv run python expt.py quick-test --config-file conf/pipeline.yaml dataset.n=1

# ── Results ──────────────────────────────────────────────────────────

list:
	uv run python -m secretagent.cli.results list results/*

average:
	uv run python -m secretagent.cli.results average --metric correct --metric cost results/*

compare:
	uv run python -m secretagent.cli.results compare-configs results/*
