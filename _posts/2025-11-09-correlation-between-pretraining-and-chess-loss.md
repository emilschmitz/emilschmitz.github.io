---
layout: post
title: "Correlation Between Pretraining and Chess Loss"
date: 2025-11-09
excerpt: "Stress-testing Epoch's Direct Approach: does lower text loss track chess move prediction vs Leela, and where do models sit on a human Elo ladder?"
---
Report from the [Apart AI Forecasting Hackathon](https://apartresearch.com/sprints/the-ai-forecasting-hackathon-2025-10-31-to-2025-11-02), updated with a July 2026 Tinker + Leela suite. Code: [emilschmitz/chess-check](https://github.com/emilschmitz/chess-check). Run id `20260710T231834Z`.

**Emil Schmitz** · Independent · with Apart Research

## Abstract

Epoch AI's Direct Approach treats lower average training loss as a proxy for broader intellectual capability. We stress-test that claim on chess: measure assistant-only bits-per-byte (BPB) on post-cutoff prose, then cross-entropy to Leela Chess Zero on a fixed grandmaster game, and bridge chess scores to human Elo bands via played-move NLL under the same Leela policy.

On 11 instruct models (Tinker; excluding gpt-oss), text BPB and chess CE track each other closely (Pearson \(r \approx 0.84\)). Absolute chess CE remains high (~3.9–7.7 nats). Model top-1 Leela NLL sits above Deep Blue's played-move NLL on the same game (~2.62) and above strong Lichess bands (~2.33 for 2600+).

**Keywords**: Direct Method, Chess, LLM, Leela Chess Zero

---

## Introduction

Does lower general text loss imply better performance on a narrow, underrepresented domain? Epoch's Direct Approach argues that sufficiently low training loss implies the model can reproduce human-written work across intellectual tasks. That assumes balanced coverage. Chess notation is sparse relative to web prose: a model can look strong on average text while remaining weak at predicting expert moves.

We measure both sides under one protocol: (1) text loss as assistant-only BPB on recent documents, (2) chess loss as \(H(\pi_{\mathrm{Leela}} \| \pi_{\mathrm{LLM}})\) on Deep Blue–Kasparov 1997 Game 6 (White), (3) a human/engine ladder of \(-\log \pi_{\mathrm{Leela}}(m_{\mathrm{played}})\) by Elo band.

---

## Methods

### Models

Instruct / chat checkpoints only (no base models), scored via [Tinker](https://thinkingmachines.ai/):

Qwen3.5-4B, Qwen3-8B, Qwen3.5-9B, Qwen3.6-27B, Nemotron-3-Nano, Qwen3.6-35B-A3B, Nemotron-3-Super, Qwen3.5-397B, Nemotron-3-Ultra, DeepSeek-V3.1, Kimi-K2.6.

We also ran GPT-OSS-20B and GPT-OSS-120B. Under this chat / Harmony protocol their assistant-only text BPB is pathological (~1.28 and ~2.57), so we treat them as outliers—likely a training or template mismatch rather than a real capability signal—and do not include them in the main table, fits, or plots below.

### Text loss (BPB)

- **Corpus:** 10 July-2026 documents (~58 KB total), diverse genres (news, wiki-style, job ad, etc.). Manifest under `eval_data/documents/`.
- **Protocol:** native chat template; user message is a short “write an article titled …” prompt (**title only — no document leak**); assistant content is the full document; score **assistant tokens only**.
- **Metric:** bits per byte (cross-tokenizer comparable), plus nats/token.

### Chess reference and scoring

- **Reference:** Leela Chess Zero BT3, `nodes=1` (policy head), CUDA on an A100.
- **Game:** Deep Blue vs Kasparov, 1997 Game 6 — **White only**, 19 positions (`eval_data/pgn/deep_blue_kasparov_1997_g6.pgn`).
- **LLM move probs:** full-string SAN with leading space (e.g. `" Nf3"`); product of token conditionals; renormalize over legal moves.
- **Primary metric:** \(H(\pi_{\mathrm{Leela}} \| \pi_{\mathrm{LLM}})\).
- **Bridge metric:** top-1 Leela NLL \(-\log \pi_{\mathrm{Leela}}(\arg\max_m \pi_{\mathrm{LLM}})\).

### Human / engine calibration

Same Leela policy. For humans (and Deep Blue), score the **played** move: \(-\log \pi_{\mathrm{Leela}}(m_{\mathrm{played}})\), aggregated by Elo band from PGN headers (Lichess sample) plus Deep Blue as an engine anchor on G6.

### Cost

Tinker estimate for this suite (including the excluded gpt-oss runs): about \$0.11 text + \$0.65 chess ≈ **\$0.76** (cap \$50).

---

## Results

### Text BPB vs chess CE

| Model | Params (B) | Text BPB | Chess CE | Top-1 NLL |
| ----- | ----------:| --------:| --------:| ---------:|
| qwen3.5-4b | 4 | 0.701 | 5.130 | 3.921 |
| qwen3-8b | 8 | 0.768 | 7.687 | 4.368 |
| qwen3.5-9b | 9 | 0.663 | 4.891 | 4.617 |
| qwen3.6-27b | 27 | 0.616 | 5.752 | 4.493 |
| nemotron-3-nano | 30 | 0.690 | 6.061 | 4.579 |
| qwen3.6-35b-a3b | 35 | 0.637 | 4.727 | 4.491 |
| nemotron-3-super | 120 | 0.630 | 4.473 | 4.217 |
| qwen3.5-397b | 397 | 0.602 | 4.480 | 3.483 |
| nemotron-3-ultra | 550 | 0.602 | 4.352 | 4.575 |
| deepseek-v3.1 | 671 (37 act.) | 0.623 | 3.876 | 4.026 |
| kimi-k2.6 | 1000 (32 act.) | 0.584 | 4.081 | 4.026 |

![Text BPB vs chess CE](/assets/chess-check-20260710/text_bpb_vs_chess_ce.png)

Among these 11 models the pattern is fairly clean: Pearson \(r(\mathrm{BPB}, \mathrm{CE}) \approx 0.84\), Spearman \(\approx 0.74\), with fit

\[
\mathrm{CE} \approx -5.94 + 16.99 \cdot \mathrm{BPB}.
\]

Best chess CE: DeepSeek-V3.1 (3.876). Lowest text BPB: Kimi-K2.6 (0.584). Lowest top-1 Leela NLL: Qwen3.5-397B (3.483).

### Human / Deep Blue ladder

| Band | Mean played NLL | n moves |
| ---- | ---------------:| -------:|
| DeepBlue1997 (G6 White) | 2.625 | 19 |
| 1400–1800 | 2.845 | 56 |
| 1800–2200 | 2.962 | 48 |
| 2200–2600 | 2.412 | 423 |
| 2600+ | 2.332 | 476 |

Mid bands are small-\(N\) and noisy. Strong humans (~2.33) and Deep Blue on this game (~2.62) sit well below every model's top-1 Leela NLL (~3.5–4.6). Empirical CE→top-1 fit:

\[
\mathrm{top1} \approx 3.69 + 0.113 \cdot \mathrm{CE}.
\]

![CE vs top-1 NLL with anchors](/assets/chess-check-20260710/ce_vs_top1_nll.png)

---

## Discussion

Excluding gpt-oss, text BPB and chess CE move together — consistent with “better models are better at both,” not a clean dissociation. Two caveats still cut against a strong Direct Approach reading:

1. **Level, not just slope.** Even the best CE (~3.9) is far from matching Leela; top-1 NLL never reaches Deep Blue or 2600+ played-move NLL on this ladder.
2. **Narrow chess sample.** One game (19 White plies) means ranking noise is real; within-family quirks (e.g. Qwen3.6-27B) still appear.

So: lower text loss tracks better chess *relative* ranking in this sample, but absolute chess competence under Leela remains weak, and the human bridge says models are not yet “playing like” strong humans on the played-move NLL metric.

---

## Limitations

- One famous game for the LLM chess curve; contamination / memorization risk for well-known PGNs.
- Human ladder is a convenience Lichess sample; mid Elo bands underpowered.
- Leela BT3 `nodes=1` is a policy snapshot, not search Elo; castling SAN (`O-O`) occasionally missing from parsed VerboseMoveStats → epsilon fallback.
- Instruct-only curve; no base-model pretrain loss comparison in this run.
- gpt-oss excluded from the main curve due to Harmony / chat-template BPB pathology.

---

## References

1. Epoch AI. (2024). Direct Approach to AI Forecasting. https://epoch.ai/files/direct-approach.pdf
2. Ruoss, A., et al. (2024). Grandmaster-Level Chess Without Search. arXiv:2402.04494
3. Leela Chess Zero Project. https://lczero.org/
4. Hoffmann, J., et al. (2022). Training Compute-Optimal Large Language Models. arXiv:2203.15556
5. Thinking Machines Lab. Tinker. https://thinkingmachines.ai/

---

## Appendix

Artifacts for run `20260710T231834Z`: `results/20260710T231834Z/` (merged CSV, human calibration, plots) and `logs/runs/20260710T231834Z/`. Methodology notes also in-repo at `docs/chess_methodology.md`.

### Future work

- More games / Leela self-play positions; BT4 + search for reference variants.
- Stockfish MultiPV as a second reference.
- Fix O-O SAN mapping; enlarge human Elo samples.
- Optional base-model curve with raw-continuation BPB for Direct Approach closer to pretrain loss.
- Revisit gpt-oss with a Harmony-aware scoring protocol (analysis vs final channels).
