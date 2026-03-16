workflow:
	uv run expt.py run evaluate.expt_name=workflow \
	  ptools.are_sports_in_sentence_consistent.method=direct \
	  ptools.are_sports_in_sentence_consistent.fn=ptools.sports_understanding_workflow

pot:
	uv run expt.py run evaluate.expt_name=pot \
	  ptools.are_sports_in_sentence_consistent.method=program_of_thought 

qpot:
	uv run expt.py quick-test evaluate.expt_name=workflow ptools.are_sports_in_sentence_consistent.method=program_of_thought

# fails because prompt_llm needs a type coercion
zeroshot:
	uv run expt.py run evaluate.expt_name=zero_shot dataset.n=25 \
	  ptools.are_sports_in_sentence_consistent.method=prompt_llm \
	ptools.are_sports_in_sentence_consistent.prompt_template_file=prompt_templates/zeroshot.txt

list:
	uv run python -m secretagent.cli.results list results/*

avg:
	uv run python -m secretagent.cli.results average --metric correct --metric cost results/*

pair:
	uv run python -m secretagent.cli.results pair --metric correct --metric cost results/*

compare:
	uv run python -m secretagent.cli.results compare-configs results/*
