---
layout: post
title: "Correlation Between Pretraining and Chess Loss"
date: 2025-11-09
published: false
excerpt: "Stress-testing Epoch's Direct Approach: does lower text loss track chess move prediction vs Leela, and where do models sit on a human Elo ladder?"
---

*Notes (2026-07-12).*

Epoch's [Scaling transformative autoregressive models](https://epoch.ai/files/direct-approach.pdf) (Direct Approach technical report) treats each token as contributing equally to the distinguishability / confidence limit. That assumption is counter to how models actually fail: a small deviation early in a sequence (e.g. introduction prose) is much less significant than a deviation at a point with essentially one correct answer (e.g. a math equation).

Epoch cannot measure that difference because they work from human training data — you observe a single "right" continuation, not a full next-token distribution. Formally, this is where the error surfaces: they approximate the odds ratio at the confidence-limit breach (~90%) under the assumption that the limit is approached continuously via small per-token steps. For sharp single-answer tokens the limit would be breached abruptly, so the values differ.

Open question: maybe chess loss can capture and partly quantify that error — lots of hand-waving and approximations; who knows.

Report from the [Apart AI Forecasting Hackathon](https://apartresearch.com/sprints/the-ai-forecasting-hackathon-2025-10-31-to-2025-11-02), updated with a July 2026 Tinker + Leela suite. Code: [emilschmitz/chess-check](https://github.com/emilschmitz/chess-check). Run id `20260710T231834Z`.

**Emil Schmitz** · Independent · with Apart Research

## Abstract

Epoch AI's Direct Approach treats lower average training loss as a proxy for broader intellectual capability: if a model is indistinguishable from the data distribution over long horizons (blog → manuscript → book), it should be competent on the tasks implicit in that distribution. We stress-test the transfer story on chess.

We measure assistant-only bits-per-byte (BPB) on post-cutoff prose and cross-entropy to Leela Chess Zero on three setups (Deep Blue–Kasparov 1997 G6 White; Carlsen–Ernst 2004 both sides; Byrne–Fischer 1963 both sides), then bridge to human Elo via played-move NLL under the same Leela policy.

On 11 instruct models (Tinker; gpt-oss excluded), text BPB and mean chess CE track each other (Pearson $$r \approx 0.88$$). Absolute chess remains weak versus human/Deep Blue played-move anchors. Pushing BPB through our CE→top-1→Elo fits still lands around ~1700–1800 Elo — not expert. That is evidence against a strong *transfer* reading (low text loss ⇒ human-level chess), not a claim Epoch made.

**Keywords:** Direct Method, Chess, LLM, Leela Chess Zero

## Introduction

Does lower general text loss imply better performance on a narrow, underrepresented domain? Epoch's Direct Approach argues that sufficiently low loss — equivalently, high $$k$$-performance (tokens until a judge can tell model from human) — implies competence on tasks in the distribution. Chess notation is sparse relative to web prose, so it is a natural place to look for a gap.

We measure: (1) assistant-only BPB on recent documents, (2) chess loss $$H(\pi_{\mathrm{Leela}} \| \pi_{\mathrm{LLM}})$$ on three famous-game setups, (3) a human ladder of $$-\log \pi_{\mathrm{Leela}}(m_{\mathrm{played}})$$ by Elo.

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

- **Reference:** Leela BT3, `nodes=1` (policy prior).
- **Games:**
  - Deep Blue–Kasparov 1997 G6 — **White only** for now (19 plies). Kasparov (Black) needs a Leela dump before Tinker scoring.
  - Carlsen–Ernst 2004 — **both sides** (57 plies).
  - Byrne–Fischer 1963 — **both sides** (42 plies; shorter Fischer alternative).
- **LLM probs:** full-string SAN with leading space; renormalize over legal moves.
- **Metrics:** $$H(\pi_{\mathrm{Leela}} \| \pi_{\mathrm{LLM}})$$ and top-1 Leela NLL $$-\log \pi_{\mathrm{Leela}}(\arg\max \pi_{\mathrm{LLM}})$$. Primary aggregate: mean CE / top-1 across the three setups (DB White + two both-side games).

### Human ladder

Same Leela policy (BT3, `nodes=1`) on Lichess rated blitz (42 games, 3013 moves) with per-move Elo. Fit played-move NLL on Elo (OLS + bootstrap 95% CI). Deep Blue G6 White is an engine anchor (played NLL ≈ 2.62).

## Results

### Text BPB vs chess CE

| Model | Params (B) | Text BPB | DB W CE | Carlsen CE | Fischer CE | Mean CE | Mean top-1 |
| ----- | ----------:| --------:| -------:| ----------:| ----------:| -------:| ----------:|
| kimi-k2.6 | 1000 (32 act.) | 0.584 | 4.081 | 3.402 | 3.705 | 3.729 | 3.046 |
| nemotron-3-ultra | 550 | 0.602 | 4.352 | 3.789 | 4.193 | 4.111 | 3.931 |
| qwen3.5-397b | 397 | 0.602 | 4.480 | 3.418 | 3.744 | 3.881 | 3.309 |
| qwen3.6-27b | 27 | 0.616 | 5.752 | 4.911 | 4.922 | 5.195 | 4.099 |
| deepseek-v3.1 | 671 (37 act.) | 0.623 | 3.876 | 3.364 | 3.473 | 3.571 | 3.413 |
| nemotron-3-super | 120 | 0.630 | 4.473 | 4.019 | 4.009 | 4.167 | 3.763 |
| qwen3.6-35b-a3b | 35 | 0.637 | 4.727 | 4.368 | 4.229 | 4.441 | 4.462 |
| qwen3.5-9b | 9 | 0.663 | 4.891 | 4.789 | 4.583 | 4.755 | 4.476 |
| nemotron-3-nano | 30 | 0.690 | 6.061 | 5.547 | 5.434 | 5.680 | 4.232 |
| qwen3.5-4b | 4 | 0.701 | 5.130 | 4.817 | 4.947 | 4.965 | 4.087 |
| qwen3-8b | 8 | 0.768 | 7.687 | 6.695 | 6.270 | 6.884 | 4.301 |

Carlsen / Fischer columns are both-side means. Per-side splits live in `results/20260710T231834Z/chess/aggregate_extra_games.csv`.

![Text BPB vs chess CE](/assets/chess-check-20260710/text_bpb_vs_chess_ce.png)
<p class="figcap">Figure 1. Lower BPB and lower mean chess CE are better (bottom-left). Mean CE averages DB White, Carlsen both-side, and Fischer both-side. Line is OLS; shaded band is a bootstrap 95% CI. gpt-oss omitted.</p>

Among these 11: Pearson $$r(\mathrm{BPB},\overline{\mathrm{CE}})\approx 0.88$$, Spearman $$\approx 0.75$$,

$$\overline{\mathrm{CE}} \approx -5.60 + 15.88\cdot\mathrm{BPB}$$

(with a wide bootstrap CI — $$n=11$$ points).

Best mean chess CE: DeepSeek-V3.1 (3.571). Lowest text BPB: Kimi-K2.6 (0.584). Lowest mean top-1: Kimi-K2.6 (3.046).
### Human / Deep Blue calibration

OLS on 3013 moves (after fixing Lc0 castling UCI→SAN):

$$\mathrm{NLL} \approx 3.132 - 0.000594\cdot\mathrm{Elo}$$

(slope 95% CI $$[-0.00068,\ -0.00051]$$; Spearman $$\rho\approx -0.23$$). Inverse: $$\mathrm{Elo} \approx 2196 - 107\cdot\mathrm{NLL}$$.

| Elo | Fit NLL |
| ---:| -------:|
| 1200 | 2.42 |
| 1600 | 2.18 |
| 2000 | 1.94 |
| 2400 | 1.71 |
| 2800 | 1.47 |
| Deep Blue G6 (NLL 2.62) | ≈ 1914 |

Band means:

| Band | Mean Elo | Mean NLL | SE | n |
| ---- | --------:| --------:| --:| -:|
| &lt;1400 | 1180 | 2.419 | 0.06 | 553 |
| 1400–1800 | 1680 | 2.269 | 0.06 | 569 |
| 1800–2200 | 1932 | 1.940 | 0.05 | 669 |
| 2200–2600 | 2348 | 1.606 | 0.04 | 703 |
| 2600+ | 2761 | 1.586 | 0.04 | 519 |

![Human Elo ladder](/assets/chess-check-20260710/human_elo_ladder.png)
<p class="figcap">Figure 2. Played-move NLL vs Elo; OLS with 95% CI.</p>

![CE vs top-1 NLL](/assets/chess-check-20260710/ce_vs_top1_nll.png)
<p class="figcap">Figure 3. Model CE vs top-1 NLL (Deep Blue played-move NLL as a horizontal reference).</p>

Model CE→top-1: $$\mathrm{top1}\approx 3.69 + 0.113\cdot\mathrm{CE}$$. Model top-1 NLLs (≈3.5–4.6) sit above Deep Blue's played-move NLL (~2.62) on Figure 3.

### Anchoring Epoch's Direct Approach

Epoch's Direct Approach ([PDF](https://epoch.ai/files/direct-approach.pdf), [blog](https://epoch.ai/publications/the-direct-approach), [interactive](https://epoch.ai/publications/direct-approach-interactive-model)) does **not** say “loss $$X$$ ⇒ human-level chess.” It says:

1. Cross-entropy decomposes as $$H(p,q)=H(p)+D_{\mathrm{KL}}(p\|q)$$.
2. Reducible loss $$D_{\mathrm{KL}}$$ controls how many samples an ideal judge needs before telling model from data (sequential probability ratio / $$k$$-performance).
3. Humans are slower than ideal by a constant “slowdown”; Figure 1 plots $$k$$ vs compute for tweet / blog / manuscript / book horizons (~$$10^2$$–$$10^5$$ tokens).
4. **Indistinguishability ⇒ competence** on the *same* task distribution (soft upper bound on compute to automate that task). They forecast text/TAI via $$k$$ in tokens; they are not making chess predictions. (A single illustrative sentence defines $$k$$ with “moves vs a GM” only to show that the sample unit is task-dependent.)

**Teacher forcing vs free-run.** Training and our BPB are next-token loss with the context fixed to real text (teacher forcing). True “write an indistinguishable book” is free-run generation: each token conditions on the model’s own past, so errors can compound. Epoch’s §2.4 writes the likelihood ratio on full sequences $$\Lambda_k=p_1(X_{1:k})/p_0(X_{1:k})$$ and approximates with average per-token KL under a stationary-ergodic assumption — i.e. they *intend* free-run indistinguishability, but the loss they plug in from scaling laws is the usual teacher-forced CE. That gap is real; we do not close it here.

**Instruct vs base.** We score released chat/instruct checkpoints (what people actually sample). That is not pretrain CE. Post-training can raise loss on web-like text and change chess behavior. Tinker’s public catalog for this run was instruct-only; we have **no paired base↔instruct BPB** in this suite.

**What we can compute.** We cannot map our BPB → Epoch $$k$$ without the irreducible entropy $$H(p)$$ of the doc distribution (BPB ≠ reducible KL). What we *can* do is push measured text loss through our empirical bridges and ask whether lower BPB predicts stronger chess under those fits — a **transfer hypothesis we impose**, not something Epoch asserts. Using **mean CE / mean top-1 across the three setups**:

| Text BPB | Mean chess CE | Mean top-1 | Fit Elo (from mean top-1) | Source |
| --------:| -------------:| ----------:| -------------------------:| ------ |
| 0.584 | 3.729 | 3.046 | ≈1869 | kimi-k2.6 |
| 0.623 | 3.571 | 3.413 | ≈1830 | deepseek-v3.1 (best mean CE) |
| 0.602 | 3.881 | 3.309 | ≈1841 | qwen3.5-397b |
| 0.50 | — | — | — | (linear BPB→mean-CE fit still does not reach human played NLL) |

Fit Elo uses $$\mathrm{Elo}\approx 2196-107\cdot\mathrm{NLL}$$ on mean top-1. Values stay ~1800s — above Deep Blue played NLL (2.62 ⇒ ≈1914) is not required for the point: model top-1 means (~3.0–4.5) remain worse than strong human played NLL (~1.6 at 2600+).

Epoch’s named **text** horizons (Figure 1; for context only):

| Horizon | ~Tokens $$k$$ | DA claim |
| ------- | ------------:| -------- |
| Tweet | ~$$10^2$$ | hard to tell on short snippets |
| Short blog | ~$$10^3$$ | blog-scale text indistinguishability |
| Scientific manuscript | ~$$10^4$$ | default TAI-ish text anchor |
| Book | ~$$10^5$$ | long-horizon text indistinguishability |

Our probe: if “low text loss / long $$k$$ ⇒ competence on tasks latent in the data” transferred to chess (sparse in web text), Leela top-1 / played NLL should approach human expert anchors as BPB falls. Here BPB and chess CE still correlate, but absolute Leela numbers (and Elo from top-1) do not approach 2600+ / Deep Blue.

## Discussion

Excluding gpt-oss, better text BPB goes with better mean chess CE — bottom-left on Figure 1. Caveats:

1. **Level, not slope.** Best mean CE ~3.57 is still far from Leela; mean top-1 does not reach Deep Blue / 2600+ played NLL.
2. **Chess sample.** Three famous-game setups; Deep Blue still White-only until Kasparov (Black) Leela exists.
3. **Human ladder is blitz-only** in the expanded sample; classical/rapid may differ. Still one Leela net (`nodes=1`).
4. **gpt-oss.** Harmony / chat scoring pathology; left out of the main curve on purpose.
5. **Shallow Elo bridge.** Linear bridges from BPB do not drive top-1 down to human played NLL.

## Limitations

- Famous-game contamination risk; BT3 `nodes=1` is policy, not search Elo.
- Instruct-only next-token BPB (released chat models); no paired base-model pretrain loss in this run.
- BPB is not reducible KL; we cannot read Epoch $$k$$-performance off our BPB without $$H(p)$$.
- Teacher-forced BPB ≠ free-run book indistinguishability (see Direct Approach §2.4 caveat above).
- Deep Blue G6 Black (Kasparov) not yet scored — no Leela dump for that side.

## References

1. Barnett, M. & Besiroglu, T. (2023). The Direct Approach. [Epoch AI](https://epoch.ai/publications/the-direct-approach) · [PDF](https://epoch.ai/files/direct-approach.pdf) · [Interactive model](https://epoch.ai/publications/direct-approach-interactive-model)
2. Ruoss, A., et al. (2024). Grandmaster-Level Chess Without Search. arXiv:2402.04494
3. Leela Chess Zero. https://lczero.org/
4. Hoffmann, J., et al. (2022). Training Compute-Optimal Large Language Models. arXiv:2203.15556
5. Thinking Machines Lab. Tinker. https://thinkingmachines.ai/

## Appendix

Artifacts under `results/20260710T231834Z/`: merged CSVs, `chess/summary_three_games.csv`, `chess/aggregate_extra_games.csv`, per-game JSONs (`chess/carlsen_ernst_2004/`, `chess/byrne_fischer_1963/`), human calibration, plots; logs mirror under `logs/runs/20260710T231834Z/`. Methodology: `docs/chess_methodology.md`.

### Future work

- **Leela dump + Tinker score for Deep Blue G6 Black (Kasparov)** — completes the third both-side game.
- More human games at &lt;2200 Elo; isotonic / hierarchical Elo–NLL model.
- More games / Leela self-play positions; BT4 + search.
- Stockfish MultiPV second reference.
- Harmony-aware gpt-oss scoring (analysis vs final).
- Paired base vs instruct BPB if base checkpoints become available on Tinker.
- Map our BPB into Epoch's KL / $$k$$-performance units more tightly (needs entropy of the doc distribution).
- Free-run (sampled) chess games vs Maia / Lichess bots for a second Elo estimate.
