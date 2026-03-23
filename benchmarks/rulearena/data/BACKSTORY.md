This benchmark is based on the RuleArena dataset:

    Zhou, R., Hua, W., Pan, L., Cheng, S., Wu, X., Yu, E., & Wang, W. Y. (2025).
    RULEARENA: A Benchmark for Rule-Guided Reasoning with LLMs in Real-World Scenarios.
    Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics, 550-572.

Source repository: https://github.com/SkyRiver-2000/RuleArena

The benchmark covers three domains:
- airline: 300 problems across 3 complexity levels, American Airlines baggage fee calculation
- nba: 216 problems, NBA CBA salary cap rule compliance checking
- tax: 300 problems, US federal income tax computation

Problems are loaded at runtime from the RuleArena repository (external/RuleArena).
Clone it before running experiments:

    git clone https://github.com/SkyRiver-2000/RuleArena external/RuleArena

where external/ is a sibling of the secretagent/ directory.
