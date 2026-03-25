# Tasks/Known bugs

## Core issues

 * non-primitive types don't work with Simulate
   * problem is output validation, should at least give warnings
   * when errors are caught by evaluator's, should _record relevant
     information from the stack trace
 * move expt.py into core/cli

## Learning

 * You can't filter cases used for distillation in any way - by
   verifiable rewards is an obvious one.

 * Should set up in-context examples as another learning approach.

## Caching

 * Check if disabling caching from the command-line works
 * Revisit how llm_util does caching - as is using the cache bypasses echos

## Experimentation

 * Run experiments in sports_understanding
     * use smaller models till the task gets "interesting"?
       * even with deepseek unstructured_baseline is 69, pot 72, react 65
	   * react often runs out of retries
       * unstructured_baseline has trouble finding final answer

## Learning methods

Thoughts on distilling react traces.

Add a learn/distill_pot.py
 * takes every pot function for every example
 * uses simulated interfaces to convert to a 'canonical' functional form
   "def workflow(...) -> ..."
 * uses ast to rename all the variables to v01, v02, ...
 * hashes them to get a smaller set of functions
 * computes coverage of each function (correct/incorrect)
 * does some sort of greedy set cover or incrementally calls a
   simulated interface to refactor the workflows into one program

## Code quality/etc

 * Some typehints fail
 * More guidance for claude/devs on defense programming
 * Standardize implement strategies: [un]structured_baseline, pot, workflow, react

## Known minor bugs

 * Running the simulate_pydantic with tools leads to a bunch of
   litellm task warnings, which are meaningless but annoying and
   seemingly hard to fix.
