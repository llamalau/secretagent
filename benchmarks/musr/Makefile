#
# Run experiments — MUSR benchmark (murder, object, team splits)
#
# Usage: make murder_workflow, make object_react, etc.
# Override model: make murder_workflow MODEL=together_ai/deepseek-ai/DeepSeek-V3
# Override n: make murder_workflow N=75
#

MODEL ?= together_ai/deepseek-ai/DeepSeek-V3
N ?= 75

# --- Murder mysteries ---

murder_zs:
	uv run python expt.py run --config-file conf/murder.yaml \
	  evaluate.expt_name=murder_zs llm.model=$(MODEL) dataset.n=$(N)

murder_cot:
	uv run python expt.py run --config-file conf/murder.yaml \
	  evaluate.expt_name=murder_cot llm.model=$(MODEL) dataset.n=$(N) \
	  ptools.raw_answer.method=prompt_llm \
	  ptools.raw_answer.prompt_template_file=prompt_templates/zero_shot_cot.txt

murder_workflow:
	uv run python expt.py run --config-file conf/murder.yaml \
	  evaluate.expt_name=murder_workflow llm.model=$(MODEL) dataset.n=$(N) \
	  evaluate.entry_point=answer_question_workflow

murder_react:
	uv run python expt.py run --config-file conf/murder.yaml \
	  evaluate.expt_name=murder_react llm.model=$(MODEL) dataset.n=$(N) \
	  ptools.answer_question.method=simulate_pydantic \
	  ptools.answer_question.tools=[ptools_murder.extract_suspects_and_evidence,ptools_murder.verify_alibis,ptools_murder.deduce_murderer]

murder_pot:
	uv run python expt.py run --config-file conf/murder.yaml \
	  evaluate.expt_name=murder_pot llm.model=$(MODEL) dataset.n=$(N) \
	  ptools.answer_question.method=program_of_thought \
	  ptools.answer_question.tools=[ptools_murder.extract_suspects_and_evidence,ptools_murder.verify_alibis,ptools_murder.deduce_murderer]

# --- Object placement ---

object_zs:
	uv run python expt.py run --config-file conf/object.yaml \
	  evaluate.expt_name=object_zs llm.model=$(MODEL) dataset.n=$(N)

object_cot:
	uv run python expt.py run --config-file conf/object.yaml \
	  evaluate.expt_name=object_cot llm.model=$(MODEL) dataset.n=$(N) \
	  ptools.raw_answer.method=prompt_llm \
	  ptools.raw_answer.prompt_template_file=prompt_templates/zero_shot_cot.txt

object_workflow:
	uv run python expt.py run --config-file conf/object.yaml \
	  evaluate.expt_name=object_workflow llm.model=$(MODEL) dataset.n=$(N) \
	  evaluate.entry_point=answer_question_workflow

object_react:
	uv run python expt.py run --config-file conf/object.yaml \
	  evaluate.expt_name=object_react llm.model=$(MODEL) dataset.n=$(N) \
	  ptools.answer_question.method=simulate_pydantic \
	  ptools.answer_question.tools=[ptools_object.extract_movements,ptools_object.infer_belief]

object_pot:
	uv run python expt.py run --config-file conf/object.yaml \
	  evaluate.expt_name=object_pot llm.model=$(MODEL) dataset.n=$(N) \
	  ptools.answer_question.method=program_of_thought \
	  ptools.answer_question.tools=[ptools_object.extract_movements,ptools_object.infer_belief]

# --- Team allocation (single-ptool, simulate+thinking wins at 84%) ---

team:
	uv run python expt.py run --config-file conf/team.yaml \
	  evaluate.expt_name=team llm.model=$(MODEL) dataset.n=$(N)

#
# View results of past experiments
#

list:
	uv run python -m secretagent.cli.results list results/*

avg:
	uv run python -m secretagent.cli.results average --metric correct --metric cost results/*

compare:
	uv run python -m secretagent.cli.results compare-configs results/*

costs:
	uv run python -m secretagent.cli.costs llm_cache
