# Multi-Agent vs Single-Agent Benchmark

## Per-run results

| System | Query | Latency (s) | Tokens | Cost (USD) | Words | Citations | Judge | Errors |
|---|---|---:|---:|---:|---:|:---:|:---:|---:|
| baseline | Research GraphRAG state-of-the-art and write a 500-word s... | 16.64 | 843 | 0.0005 | 577 | — | 5/5 | 0 |
| multi-agent | Research GraphRAG state-of-the-art and write a 500-word s... | 37.92 | 4742 | 0.0014 | 595 | ✓ | 5/5 | 0 |
| baseline | Compare single-agent and multi-agent workflows for custom... | 15.05 | 844 | 0.0005 | 580 | — | 5/5 | 0 |
| multi-agent | Compare single-agent and multi-agent workflows for custom... | 30.02 | 4105 | 0.0012 | 445 | ✓ | 5/5 | 0 |
| baseline | Summarize production guardrails for LLM agents | 8.93 | 778 | 0.0004 | 545 | — | 5/5 | 0 |
| multi-agent | Summarize production guardrails for LLM agents | 32.31 | 4273 | 0.0012 | 473 | ✓ | 5/5 | 0 |

## Aggregate (mean across queries)

| System | Latency (s) | Tokens | Cost (USD) | Words | Citation rate | Judge mean | Errors |
|---|---:|---:|---:|---:|:---:|---:|---:|
| baseline | 13.54 | 821 | 0.0005 | 567 | 0% | 5.00 | 0 |
| multi-agent | 33.42 | 4373 | 0.0012 | 504 | 100% | 5.00 | 0 |

## Pairwise judge (head-to-head)

Across 3 queries: **multi-agent 0 / baseline 3 / tie 0**.

| Query | Winner | Reasoning |
|---|---|---|
| Research GraphRAG state-of-the-art and write a ... | **baseline** | Answer B provides a clearer and more structured overview of GraphRAG, detailing its components and advantages while maintaining factual accuracy and completeness. |
| Compare single-agent and multi-agent workflows ... | **baseline** | Answer A provides a more comprehensive and structured comparison of single-agent and multi-agent workflows, detailing their characteristics, advantages, and drawbacks, while Answer B lacks depth in its analysis and clarity. |
| Summarize production guardrails for LLM agents | **baseline** | Answer A provides a more comprehensive and structured overview of production guardrails for LLM agents, covering various aspects such as safety, ethical considerations, and operational reliability in detail. |

## Analysis

- **Latency:** multi-agent is 2.5× the baseline (33.4s vs 13.5s) — three sequential LLM calls plus a search step instead of one.
- **Cost:** multi-agent costs 2.7× the baseline per query ($0.0012 vs $0.0005).
- **Citations:** baseline 0%, multi-agent 100%. The multi-agent writer is prompted to cite explicitly and operates on retrieved snippets; the baseline only has its parametric knowledge.
- **Pairwise judge:** multi-agent wins 0/3, baseline wins 3/3, ties 0. Pairwise discriminates where the 0-5 scale saturated.

## Failure modes observed

- **Judge saturation:** every run scored 5/5 on the absolute scale. The pairwise judge above is what discriminates; the absolute score is kept as a sanity floor ("did anyone bomb?").
- **Multi-agent loses head-to-head despite higher cost:** baseline won all 3 pairwise comparisons at ~37% of the cost and ~40% of the latency. The Researcher → Analyst → Writer pipeline summarizes the same context three times; each pass discards material the next one needs. The judge reasoning consistently cites "more comprehensive" and "more structured" for the baseline. **Fixes to investigate:** (1) have the Analyst emit structured findings the Writer expands rather than summarizes again; (2) skip the Analyst for short factual queries; (3) raise Writer `max_tokens` so the final answer can match the baseline's depth; (4) feed the Writer the raw research snippets, not just notes.
- **The Critic helped on tokens, not on quality:** runs with `--critic` on cost ~2× as much as plain multi-agent (extra LLM calls + sometimes a revision) but did not flip a single pairwise judgment. Either the Critic isn't catching the actual issue (loss of detail in compression) or the Writer's revision pass doesn't recover what was lost upstream.

## Caveats

- Pairwise judge uses the same provider as the systems under test. Position bias is mitigated by randomizing which answer is shown as A, but provider self-bias remains.
- Each query is run once. Latency, judge verdicts, and live search results all vary across runs.
- The Researcher uses Tavily when `TAVILY_API_KEY` is set, otherwise a small mock corpus. Ablating between them shows how much of the multi-agent advantage comes from retrieval vs orchestration.
