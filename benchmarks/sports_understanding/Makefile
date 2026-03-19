#
# Run experiments - all on the validation data
#

# the hand-coded workflow
workflow:
	uv run expt.py run evaluate.expt_name=workflow \
	  ptools.are_sports_in_sentence_consistent.method=direct \
	  ptools.are_sports_in_sentence_consistent.fn=ptools.sports_understanding_workflow

# program of thoughts
pot:
	uv run expt.py run evaluate.expt_name=pot \
	  ptools.are_sports_in_sentence_consistent.method=program_of_thought \
	  ptools.are_sports_in_sentence_consistent.tools=[ptools.analyze_sentence,ptools.sport_for,ptools.consistent_sports]

# ReAct using simulate_pydantic (which runs ReAct if there are tools)
react:
	uv run expt.py run evaluate.expt_name=react \
	  ptools.are_sports_in_sentence_consistent.method=simulate_pydantic \
	  ptools.are_sports_in_sentence_consistent.tools=[ptools.analyze_sentence,ptools.sport_for,ptools.consistent_sports]

# zero-shot structured queries - i.e. the old ptool LLM / PTP call
zeroshot_structured: 
	uv run expt.py run evaluate.expt_name=zeroshot_structured \
	  ptools.are_sports_in_sentence_consistent.method=simulate

# zero-shot unstructured queries with a vanilla prompt
# technically the vanilla prompt returns a string, which is then coerced to a bool
# in the second stage of a very simple workflow

zeroshot_unstructured: 
	uv run expt.py run evaluate.expt_name=zeroshot_unstructured \
	  ptools.are_sports_in_sentence_consistent.method=direct \
	  ptools.are_sports_in_sentence_consistent.fn=ptools.zeroshot_unstructured_workflow

#
# View results of past experiments
#

list:
	uv run python -m secretagent.cli.results list results/*

avg:
	uv run python -m secretagent.cli.results average --metric correct --metric cost results/*

pair:
	uv run python -m secretagent.cli.results pair --metric correct --metric cost results/*

compare:
	uv run python -m secretagent.cli.results compare-configs results/*

costs:
	uv run python -m secretagent.cli.costs llm_cache

#
# collecting data for experiments with learning
#

RECORD_OPTS = dataset.split=train evaluate.record_details=true evaluate.result_dir=recordings

record_zeroshot_unstructured: 
	uv run expt.py run evaluate.expt_name=zeroshot_unstructured \
	  ptools.are_sports_in_sentence_consistent.method=direct \
	  ptools.are_sports_in_sentence_consistent.fn=ptools.zeroshot_unstructured_workflow \
	  $(RECORD_OPTS)	

record_workflow:
	uv run expt.py run evaluate.expt_name=workflow \
	  ptools.are_sports_in_sentence_consistent.method=direct \
	  ptools.are_sports_in_sentence_consistent.fn=ptools.sports_understanding_workflow \
	  $(RECORD_OPTS)	

record_pot:
	uv run expt.py run evaluate.expt_name=pot \
	  ptools.are_sports_in_sentence_consistent.method=program_of_thought \
	  ptools.are_sports_in_sentence_consistent.tools=[ptools.analyze_sentence,ptools.sport_for,ptools.consistent_sports] \
	  $(RECORD_OPTS)

record_react:
	uv run expt.py run evaluate.expt_name=react \
	  ptools.are_sports_in_sentence_consistent.method=simulate_pydantic \
	  ptools.are_sports_in_sentence_consistent.tools=[ptools.analyze_sentence,ptools.sport_for,ptools.consistent_sports] \
	  $(RECORD_OPTS)

