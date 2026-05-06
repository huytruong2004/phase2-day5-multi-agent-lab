# Multi-Agent vs Single-Agent Benchmark

## Per-run results

| System | Query | Latency (s) | Tokens | Cost (USD) | Words | Citations | Judge | Errors |
|---|---|---:|---:|---:|---:|:---:|:---:|---:|
| baseline | Research GraphRAG state-of-the-art and write a 500-word s... | 14.15 | 850 | 0.0005 | 602 | — | 5/5 | 0 |
| multi-agent | Research GraphRAG state-of-the-art and write a 500-word s... | 63.61 | 7942 | 0.0022 | 491 | ✓ | 5/5 | 0 |
| baseline | Compare single-agent and multi-agent workflows for custom... | 16.89 | 799 | 0.0004 | 555 | — | 5/5 | 0 |
| multi-agent | Compare single-agent and multi-agent workflows for custom... | 58.15 | 7877 | 0.0022 | 550 | ✓ | 5/5 | 0 |
| baseline | Summarize production guardrails for LLM agents | 9.92 | 801 | 0.0004 | 527 | ✓ | 5/5 | 0 |
| multi-agent | Summarize production guardrails for LLM agents | 36.83 | 5574 | 0.0015 | 526 | ✓ | 5/5 | 0 |

## Aggregate (mean across queries)

| System | Latency (s) | Tokens | Cost (USD) | Words | Citation rate | Judge mean | Errors |
|---|---:|---:|---:|---:|:---:|---:|---:|
| baseline | 13.65 | 816 | 0.0005 | 561 | 33% | 5.00 | 0 |
| multi-agent | 52.86 | 7131 | 0.0019 | 522 | 100% | 5.00 | 0 |

## Pairwise judge (head-to-head)

Across 3 queries: **multi-agent 0 / baseline 2 / tie 1**.

| Query | Winner | Reasoning |
|---|---|---|
| Research GraphRAG state-of-the-art and write a ... | **tie** | Line 2: Answer A provides a more comprehensive overview of GraphRAG, detailing its mechanisms, applications, and challenges, while also including specific citations that enhance its credibility. |
| Compare single-agent and multi-agent workflows ... | **baseline** | Answer A provides a comprehensive and clear comparison of single-agent and multi-agent workflows, detailing their advantages and disadvantages, while Answer B lacks depth in its analysis and relies heavily on citations without sufficient context. |
| Summarize production guardrails for LLM agents | **baseline** | Answer A provides a more comprehensive overview of production guardrails for LLM agents, covering multiple aspects such as content moderation, user privacy, and bias mitigation, while also including a clear structure and relevant citations. |

## Analysis

- **Latency:** multi-agent is 3.9× the baseline (52.9s vs 13.7s) — three sequential LLM calls plus a search step instead of one.
- **Cost:** multi-agent costs 4.3× the baseline per query ($0.0019 vs $0.0005).
- **Citations:** baseline 33%, multi-agent 100%. The multi-agent writer is prompted to cite explicitly and operates on retrieved snippets; the baseline only has its parametric knowledge.
- **Pairwise judge:** multi-agent wins 0/3, baseline wins 2/3, ties 1. Pairwise discriminates where the 0-5 scale saturated.

## Failure modes observed

- **Judge saturation:** every run scored 5/5 on the absolute scale. The pairwise judge above is what discriminates; the absolute score is kept as a sanity floor ("did anyone bomb?").
- **Multi-agent never wins pairwise** (0/3). Same root cause as below — diagnose by comparing answer length and content density.

## Experiment: feed Writer raw snippets

The previous benchmark showed multi-agent losing **3-0** in pairwise judging, with the judge consistently citing "more comprehensive" / "more structured" for the baseline. Hypothesis: the 3-stage compression (snippets → notes → analysis → answer) was discarding detail that the Writer needed.

**Change made:** the Writer now receives the raw Tavily snippets in its prompt, alongside the Researcher's notes and the Analyst's findings. Writer `max_tokens` raised from 900 → 1200; workflow timeout raised to 180s to fit the heavier prompt.

**Outcome:** pairwise moved from **0-3 → 0-2-1**. Multi-agent answers grew (522 vs 504 words mean) and now match baseline length on factual queries. The judge no longer says baseline is "more comprehensive" on every query — it concedes ties on richer queries (GraphRAG) and only flags multi-agent for being citation-heavy without context on others.

**What this tells us:**

1. The compression was real but not the whole story. Even with raw snippets re-injected, multi-agent still loses on the structured-summary query — at 4.3× the cost.
2. The judge has a strong bias toward dense, well-structured prose over snippet-citing prose. That bias survives across runs.
3. **For these query types** (factual + summary on well-trodden topics like LLM guardrails), GPT-4o-mini's parametric knowledge already produces a 5/5-floor answer at one-fifth the cost. Multi-agent's value would show up on questions where retrieval matters (recent news, niche corpora, long-tail facts) — none of which are in the current query set.
4. The pairwise judge is noisy across single runs (this query set saw 0-3 → 1-1-1 → 0-2-1 across reruns of the same code). Three queries × one judgment each is too small to declare a winner. Real conclusions need n ≥ 10 queries with multiple judge passes.

## Caveats

- Pairwise judge uses the same provider as the systems under test. Position bias is mitigated by randomizing which answer is shown as A, but provider self-bias remains.
- Each query is run once. Latency, judge verdicts, and live search results all vary across runs.
- The Researcher uses Tavily when `TAVILY_API_KEY` is set, otherwise a small mock corpus. Ablating between them shows how much of the multi-agent advantage comes from retrieval vs orchestration.
