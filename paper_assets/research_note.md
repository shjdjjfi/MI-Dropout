# Research Note: Structured Information Bottleneck for Tree/DAG Reasoning in LLMs

## Problem statement
Linear chain-of-thought (CoT) constrains reasoning to sequences, while multi-hop QA often requires branch-and-merge structure. We model reasoning as a DAG and compress it to a **minimal sufficient subgraph**.

## Novelty claim
Compared with concise-CoT/token truncation, our method enforces **structural compression** (nodes + edges + support constraints), enabling faithfulness-aware reduction rather than purely lexical reduction.

## Method draft
1. **Candidate DAG induction** from retrieved evidence and intermediate propositions.
2. **Structured IB optimization** with objective:
   - answer loss
   - graph size
   - edge entropy
   - redundancy penalty
   - unsupported-node penalty
3. **Answering** from (a) full graph, (b) bottleneck graph, (c) linearized bottleneck graph.

## Experimental draft
Datasets: HotpotQA, MuSiQue, 2WikiMultihopQA, StrategyQA, WikiHop, ProofWriter.
Models: Llama-3.1-8B, Qwen3-14B/32B, Mistral 24B, Gemma-3-27B, DeepSeek-R1-Distill-Qwen 14B/32B.

Main metrics:
- EM/F1/Accuracy
- support fact F1 / proof overlap
- graph sufficiency + minimality + compression ratio
- tokens, latency, estimated cost
- robustness stressors (distractors, shuffled evidence, missing support)

Statistical rigor:
- ≥3 seeds
- paired bootstrap / approximate randomization
- significance tables

## Expected claims and evidence
1. **Higher robustness** under distractors from structured bottleneck vs linear concise-CoT.
2. **Improved faithfulness** from explicit support-constrained graph selection.
3. **Better efficiency-accuracy frontier** by removing redundant branches while preserving necessary links.

## Limitations
- Graph induction quality may bottleneck end performance.
- Some datasets lack explicit gold edge annotations, requiring silver supervision.
- Large-model experiments require substantial GPU budget and inference engineering.
