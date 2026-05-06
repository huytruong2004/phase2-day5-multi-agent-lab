# Multi-Agent vs Single-Agent Benchmark

## Per-run results

| System | Query | Latency (s) | Tokens | Cost (USD) | Words | Citations | Judge | Errors |
|---|---|---:|---:|---:|---:|:---:|:---:|---:|
| baseline | Research GraphRAG state-of-the-art and write a 500-word s... | 24.52 | 806 | 0.0005 | 561 | — | 5/5 | 0 |
| multi-agent | Research GraphRAG state-of-the-art and write a 500-word s... | 71.78 | 7779 | 0.0021 | 538 | ✓ | 5/5 | 0 |
| baseline | Summarize production guardrails for LLM agents | 13.04 | 851 | 0.0005 | 580 | — | 5/5 | 0 |
| multi-agent | Summarize production guardrails for LLM agents | 73.76 | 8049 | 0.0022 | 562 | ✓ | 5/5 | 0 |
| baseline | What features were added in LangGraph 1.0 and how do they... | 16.18 | 762 | 0.0004 | 458 | — | 5/5 | 0 |
| multi-agent | What features were added in LangGraph 1.0 and how do they... | 34.61 | 5025 | 0.0014 | 518 | ✓ | 5/5 | 0 |
| baseline | Summarize Anthropic's Claude Sonnet 4.5 release and its b... | 9.20 | 509 | 0.0003 | 327 | — | 5/5 | 0 |
| multi-agent | Summarize Anthropic's Claude Sonnet 4.5 release and its b... | 49.59 | 6799 | 0.0018 | 456 | ✓ | 0/5 | 0 |
| baseline | Compare DeepSeek R1 and OpenAI o1 on reasoning benchmarks | 22.49 | 943 | 0.0005 | 656 | — | 5/5 | 0 |
| multi-agent | Compare DeepSeek R1 and OpenAI o1 on reasoning benchmarks | 67.48 | 7225 | 0.0020 | 479 | ✓ | 5/5 | 0 |

## Aggregate (mean across queries)

| System | Latency (s) | Tokens | Cost (USD) | Words | Citation rate | Judge mean | Errors |
|---|---:|---:|---:|---:|:---:|---:|---:|
| baseline | 17.09 | 774 | 0.0004 | 516 | 0% | 5.00 | 0 |
| multi-agent | 59.44 | 6975 | 0.0019 | 510 | 100% | 4.00 | 0 |

## Pairwise judge (head-to-head)

Across 5 queries: **multi-agent 1 / baseline 2 / tie 2**.

| Query | Winner | Reasoning |
|---|---|---|
| Research GraphRAG state-of-the-art and write a ... | **baseline** | Answer B provides a clearer and more structured overview of GraphRAG, covering its background, key features, applications, and challenges while maintaining factual accuracy and completeness. |
| Summarize production guardrails for LLM agents | **tie** | Line 2: Answer A provides a more comprehensive and structured overview of production guardrails for LLM agents, covering various aspects in detail while maintaining clarity and factual accuracy. |
| What features were added in LangGraph 1.0 and h... | **tie** | Line 2: Answer B provides a more focused overview of specific features and improvements in LangGraph 1.0, along with credible citations, making it clearer and more informative than Answer A. |
| Summarize Anthropic's Claude Sonnet 4.5 release... | **multi-agent** | Answer B provides specific benchmark scores, a detailed pricing structure, and emphasizes safety enhancements, making it more comprehensive and informative than Answer A. |
| Compare DeepSeek R1 and OpenAI o1 on reasoning ... | **baseline** | Answer B provides a more detailed and structured comparison of the two models, covering architecture, training methodologies, performance metrics, and specific capabilities, which enhances clarity and completeness. |

## Analysis

- **Latency:** multi-agent is 3.5× the baseline (59.4s vs 17.1s) — three sequential LLM calls plus a search step instead of one.
- **Cost:** multi-agent costs 4.5× the baseline per query ($0.0019 vs $0.0004).
- **Citations:** baseline 0%, multi-agent 100%. The multi-agent writer is prompted to cite explicitly and operates on retrieved snippets; the baseline only has its parametric knowledge.
- **Pairwise judge:** multi-agent wins 1/5, baseline wins 2/5, ties 2. Pairwise discriminates where the 0-5 scale saturated.

## Failure modes observed

- **Absolute judge gave multi-agent 0/5 on the post-cutoff query** (Claude Sonnet 4.5) — the same query where multi-agent _won_ pairwise. The absolute judge is GPT-4o-mini scoring an answer about a model released after its own training cutoff, so it has no factual ground truth to check against and defaults to "untrustworthy." Meta-failure of LLM-as-judge for retrieval-required queries: the judge can't validate facts it doesn't know. **Fix:** for fresh-information queries, use either (a) a separate retrieval-augmented judge that searches the same sources, or (b) human evaluation; never trust a parametric judge to score post-cutoff content.

## Experiment 2: parametric vs retrieval-required queries

The earlier "Experiment 1" finding (compression discards detail) hinted that multi-agent might only beat baseline on queries where retrieval matters. To test, the query set was rebalanced: 2 parametric (GraphRAG, LLM guardrails — well-trodden) plus 3 retrieval-required (LangGraph 1.0, Claude Sonnet 4.5, DeepSeek R1 vs o1 — all post-cutoff or version-specific).

**Result by query type:**

| Query | Type | Winner | Note |
|---|---|---|---|
| GraphRAG | parametric | baseline | baseline still "more structured" |
| LLM guardrails | parametric | tie | judge couldn't decide |
| LangGraph 1.0 | retrieval-required | tie | multi-agent gave specifics; judge called it "more focused" but tied |
| **Claude Sonnet 4.5** | **retrieval-required** | **multi-agent** | judge cited "specific benchmark scores, detailed pricing, safety enhancements" — all from Tavily |
| DeepSeek R1 vs o1 | retrieval-required | baseline | baseline still managed a confident-sounding answer; judge preferred its structure |

**Read of this:** the only outright multi-agent win was the most clearly post-cutoff query. Baseline's 327-word Claude Sonnet answer was visibly hollow (it hedged) while multi-agent had real numbers from search results. On the other retrieval-required queries the baseline still bluffed convincingly enough to draw or win — which is itself a finding: **parametric models confidently fabricate on out-of-distribution queries, and an LLM judge using the same parametric knowledge can't catch them.**

**Honest conclusion for the lab:**

- For these query types and this evaluation harness, multi-agent is **not** worth 4.5× the cost on parametric queries.
- For genuinely fresh-information queries, multi-agent has a real edge — but the LLM-as-judge can't reliably measure it, because the judge has the same knowledge cutoff as the baseline.
- A fair benchmark needs (a) a retrieval-augmented judge, (b) ground-truth comparisons against documented sources, or (c) human evaluation. The current harness systematically underestimates multi-agent's advantage.

## Caveats

- Pairwise judge uses the same provider as the systems under test. Position bias is mitigated by randomizing which answer is shown as A, but provider self-bias remains.
- Each query is run once. Latency, judge verdicts, and live search results all vary across runs.
- The Researcher uses Tavily when `TAVILY_API_KEY` is set, otherwise a small mock corpus. Ablating between them shows how much of the multi-agent advantage comes from retrieval vs orchestration.
