docker-start:
	docker run -d -p 8080:8080 jyxsu6/medagentbench:latest

docker-stop:
	docker ps -q --filter ancestor=jyxsu6/medagentbench:latest | xargs -r docker stop

run:
	uv run python expt.py run

quick-test:
	uv run python expt.py quick-test dataset.n=1

run-task1:
	uv run python expt.py run dataset.n=30 evaluate.expt_name=task1_only
