---
layout: post
title: "Correlation Between Pretraining and Chess Loss"
date: 2025-11-09
excerpt: "Stress-testing Epoch's Direct Approach: does lower text loss track chess move prediction vs Leela, and where do models sit on a human Elo ladder?"
---

Report from the [Apart AI Forecasting Hackathon](https://apartresearch.com/sprints/the-ai-forecasting-hackathon-2025-10-31-to-2025-11-02), updated with a July 2026 Tinker + Leela suite. Code: [emilschmitz/chess-check](https://github.com/emilschmitz/chess-check). Run id `20260710T231834Z`.

**Emil Schmitz** · Independent · with Apart Research

## Abstract

Epoch AI's Direct Approach treats lower average training loss as a proxy for broader intellectual capability: if a model is indistinguishable from the data distribution over long horizons (blog → manuscript → book), it should be competent on the tasks implicit in that distribution. We stress-test the transfer story on chess.

We measure assistant-only bits-per-byte (BPB) on post-cutoff prose and cross-entropy to Leela Chess Zero on Deep Blue–Kasparov 1997 Game 6 (White), then bridge to human Elo via played-move NLL under the same Leela policy.

On 11 instruct models (Tinker; gpt-oss excluded), text BPB and chess CE track each other (Pearson $$r \approx 0.84$$). Absolute chess remains weak: every model's top-1 Leela NLL sits above Deep Blue's played-move NLL on this game (~2.62) and above strong Lichess blitz bands (~2.14 for 2600+). Under Direct Approach logic, book-length text indistinguishability would imply human-expert chess; our bridge says today's BPB improvements have not closed that gap.

**Keywords:** Direct Method, Chess, LLM, Leela Chess Zero

## Introduction

Does lower general text loss imply better performance on a narrow, underrepresented domain? Epoch's Direct Approach argues that sufficiently low loss — equivalently, high $$k$$-performance (tokens until a judge can tell model from human) — implies competence on tasks in the distribution. Chess notation is sparse relative to web prose, so it is a natural place to look for a gap.

We measure: (1) assistant-only BPB on recent documents, (2) chess loss $$H(\pi_{\mathrm{Leela}} \| \pi_{\mathrm{LLM}})$$ on Deep Blue–Kasparov 1997 G6 (White), (3) a human ladder of $$-\log \pi_{\mathrm{Leela}}(m_{\mathrm{played}})$$ by Elo.

## Methods

### Models

Instruct / chat checkpoints only, via [Tinker](https://thinkingmachines.ai/):

Qwen3.5-4B, Qwen3-8B, Qwen3.5-9B, Qwen3.6-27B, Nemotron-3-Nano, Qwen3.6-35B-A3B, Nemotron-3-Super, Qwen3.5-397B, Nemotron-3-Ultra, DeepSeek-V3.1, Kimi-K2.6.

We also ran GPT-OSS-20B and GPT-OSS-120B. Under this chat / Harmony protocol their assistant-only text BPB is pathological (~1.28 and ~2.57) — likely a template / training mismatch rather than a clean capability signal — so we drop them from the main table, fits, and plots. The remaining 11 models follow a fairly clean BPB–CE pattern; we do not extrapolate the trend through gpt-oss.

### Text loss (BPB)

- **Corpus:** 10 July-2026 documents (~58 KB). Manifest under `eval_data/documents/`.
- **Protocol:** native chat template; user message is a short title-only ask (no document leak); score assistant tokens only.
- **Metric:** bits per byte.

### Chess

- **Reference:** Leela BT3, `nodes=1`, A100.
- **Game:** Deep Blue vs Kasparov 1997 G6, White only, 19 positions.
- **LLM probs:** full-string SAN with leading space; renormalize over legal moves.
- **Metrics:** $$H(\pi_{\mathrm{Leela}} \| \pi_{\mathrm{LLM}})$$ and top-1 Leela NLL $$-\log \pi_{\mathrm{Leela}}(\arg\max \pi_{\mathrm{LLM}})$$.

### Human ladder

Same Leela policy (BT3, `nodes=1`) on Lichess rated blitz (42 games, 3013 moves) with per-move Elo. Fit played-move NLL on Elo (OLS + bootstrap 95% CI). Deep Blue G6 White is an engine anchor (played NLL ≈ 2.62).

### Cost

Tinker estimate including excluded gpt-oss runs: about 0.11 USD text + 0.65 USD chess ≈ **0.76 USD** (cap 50 USD).

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
<p class="figcap">Figure 1. Lower BPB and lower CE are better (bottom-left). Line is OLS; shaded band is a bootstrap 95% CI. gpt-oss omitted.</p>

Among these 11: Pearson $$r(\mathrm{BPB},\mathrm{CE})\approx 0.84$$, Spearman $$\approx 0.74$$,

$$\mathrm{CE} \approx -5.94 + 16.99\cdot\mathrm{BPB}$$

(with a wide bootstrap CI — $$n=11$$ points, so treat the slope as indicative, not precise).

Best chess CE: DeepSeek-V3.1 (3.876). Lowest text BPB: Kimi-K2.6 (0.584). Lowest top-1 NLL: Qwen3.5-397B (3.483).

### Human / Deep Blue calibration

OLS on 3013 moves:

$$\mathrm{NLL} \approx 3.806 - 0.000637\cdot\mathrm{Elo}$$

(slope 95% CI $$[-0.00092,\ -0.00038]$$; Spearman $$\rho\approx -0.22$$).

| Elo | Fit NLL |
| ---:| -------:|
| 1200 | 3.04 |
| 1600 | 2.79 |
| 2000 | 2.53 |
| 2400 | 2.28 |
| 2800 | 2.02 |
| Deep Blue G6 (NLL 2.62) | ≈ 1986 |

Band means:

| Band | Mean Elo | Mean NLL | SE | n |
| ---- | --------:| --------:| --:| -:|
| &lt;1400 | 1180 | 3.002 | 0.17 | 553 |
| 1400–1800 | 1680 | 2.969 | 0.18 | 569 |
| 1800–2200 | 1932 | 2.472 | 0.15 | 669 |
| 2200–2600 | 2348 | 2.199 | 0.15 | 703 |
| 2600+ | 2761 | 2.135 | 0.17 | 519 |

![Human Elo ladder](/assets/chess-check-20260710/human_elo_ladder.png)
<p class="figcap">Figure 2. Played-move NLL vs Elo; OLS with 95% CI.</p>

![CE vs top-1 NLL](/assets/chess-check-20260710/ce_vs_top1_nll.png)
<p class="figcap">Figure 3. Model CE vs top-1 NLL with Elo reference lines from the human fit.</p>

Model CE→top-1: $$\mathrm{top1}\approx 3.69 + 0.113\cdot\mathrm{CE}$$. Model top-1 NLLs (≈3.5–4.6) sit above the human Elo lines on Figure 3.

### Anchoring Epoch's Direct Approach

Epoch's Direct Approach ([report](https://epoch.ai/files/direct-approach.pdf), [interactive model](https://epoch.ai/publications/direct-approach-interactive-model)) interprets loss via $$k$$-performance: how many tokens until a judge can tell model from human. Figure 1 in the report marks tweet / short blog / scientific manuscript / book-length horizons. Their default TAI $$k$$ is manuscript-anchored (roughly thousands to ~$$2\times 10^5$$ tokens); under default interactive-model assumptions, published summaries often cite on the order of **~10% probability by 2025 and ~50% by ~2033** for transformative AI.

If indistinguishability over a horizon implies competence on tasks in the text distribution, then **book- or manuscript-level Direct Approach success would predict human-expert chess**. Our measurements say that has not happened yet at today's text BPB:

| Epoch horizon ($$k$$) | ~Tokens | Direct Approach chess claim | Our empirical status (2026 suite) |
| --------------------- | -------:| --------------------------- | --------------------------------- |
| Tweet-length | ~$$10^2$$ | human-like on short tasks | Text BPB already “good” on short docs; chess top-1 NLL still ≫ human |
| Short blog post | ~$$10^3$$ | human-like on blog-scale | Same gap |
| Scientific manuscript | ~$$10^4$$ | TAI-relevant default anchor | DA would predict expert chess; we still measure top-1 NLL ~3.5–4.6 vs 2600+ played ~2.14 |
| Book-length | ~$$10^5$$ | full long-horizon indistinguishability | Same: if DA transfer held at book $$k$$, expect ≈ human expert chess; present BPB→chess bridge has not closed the NLL gap |

| Target on Leela NLL axis | Human / DB value | What our BPB→CE→top-1 fits would need |
| ------------------------ | ---------------:| ------------------------------------- |
| Lichess 2600+ played | 2.33 | CE and BPB driven **negative** (fits do not reach human NLL) |
| Deep Blue G6 played | 2.62 | Same — shallow model top-1 bridge never meets human played NLL |

So the interesting tension is not “BPB and chess CE are uncorrelated” (among ordinary instruct models they are correlated). It is that **Direct Approach-style optimism about text-loss → general competence overshoots what we see on chess**: relative ranking moves with BPB, absolute chess competence under Leela does not approach human/DB anchors.

## Discussion

Excluding gpt-oss, better text BPB goes with better chess CE — bottom-left on Figure 1. Caveats:

1. **Level, not slope.** Best CE ~3.9 is still far from Leela; top-1 NLL never reaches Deep Blue or 2600+.
2. **Narrow chess sample.** One famous game (19 White plies).
3. **Human ladder is blitz-only** in the expanded sample; classical/rapid may differ. Still one Leela net (`nodes=1`).
4. **gpt-oss.** Harmony / chat scoring pathology; left out of the main curve on purpose.

## Limitations

- Famous-game contamination risk; BT3 `nodes=1` is policy, not search Elo.
- Castling `O-O` occasionally missing from parsed VerboseMoveStats.
- Instruct-only; not raw pretrain loss.
- Epoch years above are order-of-magnitude citations of their default interactive model, not a re-run of their notebook with our BPB numbers.

## References

1. Barnett, M. & Besiroglu, T. (2023). The Direct Approach. [Epoch AI](https://epoch.ai/publications/the-direct-approach) · [PDF](https://epoch.ai/files/direct-approach.pdf) · [Interactive model](https://epoch.ai/publications/direct-approach-interactive-model)
2. Ruoss, A., et al. (2024). Grandmaster-Level Chess Without Search. arXiv:2402.04494
3. Leela Chess Zero. https://lczero.org/
4. Hoffmann, J., et al. (2022). Training Compute-Optimal Large Language Models. arXiv:2203.15556
5. Thinking Machines Lab. Tinker. https://thinkingmachines.ai/

## Appendix

Artifacts: `results/20260710T231834Z/` (merged CSV, human calibration, `epoch_chess_anchor.csv`, plots) and `logs/runs/20260710T231834Z/`. Methodology: `docs/chess_methodology.md`.

### Future work

- More human games at &lt;2200 Elo; isotonic / hierarchical Elo–NLL model.
- More games / Leela self-play positions; BT4 + search.
- Stockfish MultiPV second reference.
- Harmony-aware gpt-oss scoring (analysis vs final).
- Map our BPB into Epoch's KL / $$k$$-performance units more tightly (needs entropy of the doc distribution).
