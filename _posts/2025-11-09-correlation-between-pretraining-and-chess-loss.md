This is a report of some experiments that I began as part of the [Apart AI Forecasting Hackathon](https://apartresearch.com/sprints/the-ai-forecasting-hackathon-2025-10-31-to-2025-11-02). It is still work in progress. Code is available at [https://github.com/emilschmitz/chess-check](https://github.com/emilschmitz/chess-check).

# AI Forecasting Hackathon

## Report Template

**Emil Schmitz**
Independent

**With**
Apart Research

---

## Abstract

Epoch AI's direct method assumes that lower average loss indicates better general capabilities. We posit that the loss may possibly be indicative only of higher performance on specific content. We attempt to prove this by calculating loss on high-level chess games. To calculate loss, we compare the LLM's prediction to those of open-source chess engine Leela Chess Zero.

At the time of submission, the experiments are not yet finished.

**Keywords**: Direct Method, Chess, LLM

---

## Introduction

This project investigates whether an LLM's general training loss serves as a reliable predictor of its performance on specific, narrow tasks. We test this by evaluating how well language models predict chess moves in algebraic notation, comparing their probability distributions against those from Leela Chess Zero, a superhuman chess engine.

The motivation stems from Epoch AI's "Direct Approach" paper, which argues that training loss correlates with an AI system's ability to replicate human performance across all intellectual tasks. If training loss becomes sufficiently low, the theory suggests the model could reproduce any human-written work and thus handle any intellectual task humans can perform. However, this assumes balanced representation across all task types in the training data. In reality, datasets may contain abundant examples of common content (like blog posts or news articles) but sparse representation of specialized domains like notation of professional-level chess games. A model could achieve low overall loss by perfectly replicating frequent content types while remaining weak on rare, difficult tasks. However, a truly general-purpose AI should be able to predict moves in grandmaster-level games with good accuracy, indicating high chess ability.

We hypothesize that LLMs with lower general training losses will also show lower cross-entropy loss when predicting chess moves compared to Leela Zero's distributions, but that loss on chess will not scale proportionally with general loss. By testing this hypothesis, we examine whether training loss truly generalizes as a capability proxy or whether it masks significant performance gaps in underrepresented domains. Chess serves as an ideal test case: it's well-defined, measurable, has verifiable expert-level performance standards, and existing research shows LLMs struggle at chess without specific training.

---

## Methods

### 2.1 Models Evaluated

#### Misc Models

- GPT-2 (124M parameters) - baseline small model
- GPT-Neo 1.3B - mid-size open model
- Pythia 1.0B, 1.4B, 2.8B - suite with fully documented training trajectories

#### OLMo 2 Models (Final Checkpoints)

- OLMo 2 1B (1.0B parameters) - trained on 4T tokens (stage1) + 50B tokens (stage2)
- OLMo 2 7B (7.0B parameters) - trained on 4T tokens (stage1) + 50B tokens (stage2, model averaged)
- OLMo 2 13B (13.0B parameters) - trained on 5T tokens (stage1) + 400B tokens (stage2, model averaged)
- OLMo 2 32B (32.0B parameters) - trained on 6T tokens (stage1) + 400B tokens (stage2, model averaged)

#### OLMo 2 Models (Intermediate Checkpoints at ~50% Training)

- OLMo 2 1B-mid - checkpoint at 2.0T/4.0T tokens (stage1 only)
- OLMo 2 7B-mid - checkpoint at 2.0T/4.0T tokens (stage1 only)
- OLMo 2 13B-mid - checkpoint at 2.5T/5.0T tokens (stage1 only)
- OLMo 2 32B-mid - checkpoint at 3.0T/6.0T tokens (stage1 only)

Model training losses sourced from original papers, model cards, and WandB training logs (see References section for details).

### 2.2 Chess Ground Truth

- Leela Chess Zero (lc0) as superhuman reference
- Rationale: Open source, superhuman strength (3500+ Elo), provides move probability distributions via policy head

### 2.3 Dataset

- Source: Lichess Elite Database / FICS Games Database
- Number of games: 100
- Game quality: Grandmaster level (Elo > 2500)

### 2.4 Procedure

1. Load chess games in PGN format
2. For each position in each game:
   - Convert position to text format for LLM (full game history in algebraic notation)
   - Get LLM's probability distribution over next move (via logits over legal moves)
   - Convert position to Leela Zero input format
   - Get Leela's probability distribution over next move (via policy network output)
   - Calculate cross-entropy loss H(Leela || LLM) between distributions
3. Average loss per game
4. Average loss across all games for each model
5. Correlate with published training/evaluation losses

### 2.5 Technical Implementation

- Python with PyTorch, Transformers (HuggingFace), python-chess
- Leela Chess Zero via python-lczero or UCI interface

### 2.6 Loss Metrics and Training Details

#### Understanding Loss Metrics

The loss metrics used in this study refer to **next-token prediction loss** (cross-entropy loss), which measures how well a language model predicts the next token in a sequence. Lower loss indicates better prediction accuracy.

**Training Loss vs. Evaluation Loss:**

- **Training loss**: Computed on the training dataset during model training. Can be prone to overfitting.
- **Evaluation loss** (or validation loss): Computed on a held-out validation set not seen during training. More reliable indicator of generalization.

For models trained on massive datasets approaching full coverage of available text (e.g., OLMo 2 trained on 4-6T tokens), the distinction between train and eval loss becomes less meaningful, as the model effectively sees most available data only once. In such cases, training loss approximates evaluation loss.

#### OLMo 2 Training Regime

The OLMo 2 models follow a two-stage training process:

1. **Stage 1 (Pretraining)**: Models are trained on 4-6 trillion tokens from the OLMo-mix-1124 dataset, constituting 90-95% of the total training budget.
2. **Stage 2 (Mid-training/Annealing)**: Additional training on 50-400 billion high-quality tokens from the Dolmino-Mix-1124 dataset. For 7B, 13B, and 32B models, multiple runs with different random seeds are trained and then averaged using model souping to produce the final checkpoint.

Training stability improvements in OLMo 2 include RMSNorm, QK-Norm, rotary positional embeddings, and z-loss regularization. These architectural changes result in higher absolute training loss compared to earlier models due to the regularization terms, but improved training stability and downstream performance.

#### Loss Number Sources

Exact next-token prediction loss values for OLMo 2 models are available in:

- **WandB training logs**:
  - 1B model: N/A (?)
  - 7B model: https://api.wandb.ai/links/ai2-llm/fjn0v0ec
  - 13B model: https://api.wandb.ai/links/ai2-llm/ypmumwpc
  - 32B model: https://www.comet.com/ai2/olmo-2-0325-32b/reports/olmo-2-0325-32b?shareable=WhT37Wy7jqttDoy6ysDBumQzf
- **OLMo 2 paper** (Table 9): arXiv:2501.00656
- **Model cards**: Available on HuggingFace under allenai organization

Due to the difficulty of extracting precise numerical values from these sources during this analysis, loss values for OLMo 2 models are marked as TBD in our experiments, with full documentation of where these values can be obtained for future reference.

---

## Results

### 3.1 Chess Loss by Model

Table 1 shows the chess move prediction loss (cross-entropy against Leela Chess Zero) for all evaluated models. Lower values indicate better alignment with superhuman chess play.

| Model Name     | Parameters | Reference Loss | Chess Avg Loss | Num Games | Num Positions |
| -------------- | ---------- | -------------- | -------------- | --------- | ------------- |
| gpt2           | 124M       | 3.31           | 4.352          | 5         | 405           |
| gpt-neo-1.3B   | 1.3B       | 2.85           | 4.151          | 5         | 405           |
| pythia-1b      | 1.0B       | 2.74           | 4.118          | 5         | 405           |
| pythia-1.4b    | 1.4B       | 2.64           | 4.256          | 5         | 405           |
| olmo-2-1b      | 1.0B       | TODO           | 4.247          | 5         | 405           |
| olmo-2-1b-mid  | 1.0B       | TODO           | 4.130          | 5         | 405           |
| olmo-2-7b      | 7.0B       | TODO           | 4.233          | 5         | 405           |
| olmo-2-7b-mid  | 7.0B       | TODO           | 4.196          | 5         | 405           |
| olmo-2-13b     | 13.0B      | TODO           | 4.188          | 5         | 405           |
| olmo-2-13b-mid | 13.0B      | TODO           | 4.043          | 5         | 405           |
| olmo-2-32b     | 32.0B      | TODO           | 3.913          | 5         | 405           |
| olmo-2-32b-mid | 32.0B      | TODO           | 3.882          | 5         | 405           |

*Reference Loss = next-token prediction loss from original papers/training logs (TODO for OLMo models)*

### 3.2 Key Observations

**Model Size Effects:**

- Among OLMo 2 models, chess loss decreases with model size: the 32B model achieves the lowest loss (3.882 for mid-checkpoint, 3.913 for final), while the 1B model shows the highest loss (4.247 for final, 4.130 for mid-checkpoint).
- The OLMo 2 32B mid-checkpoint achieves the best overall chess performance across all evaluated models at 3.882.

**Checkpoint Comparison:**

- Mid-training checkpoints (~50% through stage 1) generally perform slightly better than or comparable to final checkpoints across all OLMo 2 model sizes. These differences may be due to noise.

**Cross-Model Comparisons:**

- TODO: Pending extraction of reference losses for OLMo models from WandB logs and paper to enable full correlation analysis.
- Smaller models (GPT-2 124M, Pythia 1B) show higher chess losses.

---

## Discussion and Conclusion

*…in progress*

---

## References

1. Epoch AI. (2024). Direct Approach to AI Forecasting. https://epoch.ai/files/direct-approach.pdf
2. Ruoss, A., et al. (2024). Grandmaster-Level Chess Without Search. arXiv:2402.04494
3. Leela Chess Zero Project. https://lczero.org/
4. Hoffmann, J., et al. (2022). Training Compute-Optimal Large Language Models. arXiv:2203.15556
5. Biderman, S., et al. (2023). Pythia: A Suite for Analyzing Large Language Models. arXiv:2304.01373
6. Black, S., et al. (2021). GPT-Neo: Large Scale Autoregressive Language Modeling with Mesh-Tensorflow.
7. Touvron, H., et al. (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models. arXiv:2307.09288
8. OLMo Team. (2025). 2 OLMo 2 Furious. arXiv:2501.00656. https://arxiv.org/abs/2501.00656
9. OLMo 2 Model Collection. HuggingFace. https://huggingface.co/collections/allenai/olmo-2-674117b93ab84e98afc72edc
10. OLMo 2 Training Logs. Weights & Biases.
    - 1B: https://wandb.ai/ai2-llm/OLMo2-1B
    - 7B: https://wandb.ai/ai2-llm/OLMo2-7B
    - 13B: https://wandb.ai/ai2-llm/OLMo2-13B

---

## Appendix

### Potential Limitations

- Sample size of models tested limited by computational resources
- Chess may not generalize to other domains

### Suggestions for Future Work

1. Analyze chess loss specifically on models similar to those from the Chinchilla paper (Hoffman 2022). The weights from that paper were never released though. See also [Epoch AI&#39;s replication](https://epoch.ai/publications/chinchilla-scaling-a-replication-attempt) (2024). For those models we'd have measured next-word loss numbers. We can use OLMO models for example. Another alternative would be Pythia models.
   1. Basically, we need models where we have relevant loss numbers that we can compare. Perhaps, it'd also be nice to have number of train tokens and parameters. Then we could use the Chinchilla formula to also estimate loss. I am unsure if this adds any value. Perhaps it is also useful if the models are all the same except for number of train tokens and number of parameters.
   2. olmo loss number are available on links from the header in [olmo 2 paper](https://arxiv.org/pdf/2501.00656)
2. For the OLMO models: check to what extent it is legitimate to use train loss as an approximation for test loss. To do this, we will have to check if they were trained on more than one epoch. We can also check if there is a holdout validation set we can use to get these numbers.
3. Use self-play games from Leela Zero to test on instead of actual games that might have been part of training data for lc0 (only trained on self-play?) or the LLMs
4. Expand to more models and predict loss with Chinchilla scaling law
5. Implement constrained generation for valid moves
6. Add visualization scripts (loss correlation plots)
7. Wandb or similar integration
8. Anything useful here? https://arxiv.org/html/2410.11840v1

- Test on multiple game engines (Stockfish, Komodo) and average their distributions. No engine is optimal. Leela Chess Zero may be biased towards one specific type of play, while another engine may be biased towards another.
- Expand to other domains with verifiable ground truth (mathematical proofs, code correctness)
- Use constrained generation or tool-use paradigm to ensure valid move outputs
- Investigate whether including chess games in training data improves correlation
