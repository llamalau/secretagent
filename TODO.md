Next steps

 * prompt_llm needs help for non-string outputs
   * maybe just coerce to required output type?
 * Check if disabling caching from the command-line works
 * Revisit how llm_util does caching - as is using the cache bypasses echos
 * Run experiments in sports_understanding
   * Fix problems with pydantic react factory configuration
   * Make tool use consistent between PoT and pydantic factories
   * write conf - data loaders and implementations
     * zero shot prompt_llm
     * zero shot simulate
     * react with ptools
     * use smaller models till the task gets "interesting"?
