# DFlash: Block Diffusion for Flash Speculative Decoding

[DFlash](https://arxiv.org/abs/2602.06036) | [DFlash 1–2 GitHub](https://github.com/z-lab/dflash) | [DFlare](https://arxiv.org/abs/2606.02091) | [DSpark](https://arxiv.org/abs/2607.05147) | [D-PACE](https://arxiv.org/abs/2605.18810)

## Introduction

The main improvements come from two ideas:

1. Generate the drafter's tokens in a single forward pass.
2. Use information from the target (teacher) model to predict future tokens.

## Architecture

The architecture follows two principles:

1. Reuse information from the target model.
2. Condition the drafter on the target model as strongly as possible.

### How to use the teacher's KV-cache

1. Choose a set of transformer layers $[l_1, \ldots, l_n]$ from the teacher model.
2. Extract the hidden-state history (residual streams) from the selected teacher layers.
3. Introduce one learned projection matrix $W_c$ with shape $(\text{hiddensize},\ n \times \text{hiddensize})$.
4. For each token position, concatenate the hidden states from the selected layers and project them back to the original hidden size using $W_c$.
5. Apply RMSNorm to the projected states.
6. Run the drafter's transformer layers: use drafter hidden states $h_d$ as queries (Q), and the concatenation of projected teacher states $h_t$ and drafter states $h_d$ as keys and values (K/V). Masked draft tokens use full attention within their block (see the diagram below).

![](https://i.ibb.co/Pz5NLssC/image.png)

### Training: loss and other details

The diffusion drafter is trained to *imitate* the teacher. We sample question–answer trajectories, choose random anchor positions (with the corresponding teacher KV-cache), and train the drafter to predict masked tokens starting from those anchors.

Since the goal is to imitate the teacher, we use cross-entropy loss. We reweight it according to the token's distance from its anchor, giving greater weight to earlier tokens:

$$
\mathcal{L}(\theta)
=-\sum_b\sum_{k\in\mathcal{M}_b}
\exp\left(-\frac{k-1}{\gamma}\right)
\log p_\theta(y_{b,k}\mid\cdot).
$$

> [!NOTE]
> All tokens in the draft block are masked except for the first anchor token.

> [!NOTE]
> The teacher layers whose hidden states are used must be selected.

> [!NOTE]
> Randomly sampling anchor positions improves convergence.

> [!NOTE]
> Loss reweighting (loss decay) is empirical, but provides a small improvement.

> [!TIP]
> Use a contiguous block of anchor positions during training. This allows training examples to share a teacher KV-cache prefix and makes batching more efficient.
>
> ![](https://i.ibb.co/YBT3qF4P/image.png)

## Datasets & Benchmarks

### Training

- The model was *possibly* trained for 9 epochs (needs verification).
- Training uses *teacher-generated answers*, rather than the reference answers provided by the datasets.
- The training data is the same as in EAGLE-3.

Training datasets:

1. [UltraChat](https://huggingface.co/datasets/openbmb/UltraChat)
2. [ShareGPT](https://huggingface.co/collections/bunnycore/sharegpt-datasets)

### Benchmarking

| Dataset | Category |
|---|---|
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | Math |
| [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | Math / Reasoning |
| [AIME25](https://huggingface.co/datasets/math-ai/aime25) | Math / Reasoning |
| [AIME24](https://huggingface.co/datasets/HuggingFaceH4/aime_2024) | Math |
| [HumanEval](https://github.com/openai/human-eval) | Code |
| [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) | Code |
| [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) | Code |
| [MT-Bench](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/mt_bench/question.jsonl) | Chat |
| [Alpaca](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) | Chat / Instruction Following |
| [GPQA](https://huggingface.co/datasets/Idavidrein/gpqa) | Reasoning |
| [LongBench](https://huggingface.co/datasets/zai-org/LongBench) | Long Context |
| [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa) | Long Context / Question Answering |
| [Qasper](https://huggingface.co/datasets/allenai/qasper) | Long Context / Question Answering |
| [GovReport](https://huggingface.co/datasets/launch/gov_report) | Long Context / Summarization |

### Long-Context Ablation

Average number of accepted draft tokens per step (acceptance length, $\tau$).

> [!NOTE]
> Acceptance depends strongly on the task.

> [!NOTE]
> **Long** refers to a drafter fine-tuned on long-context data; **Base** is the original drafter.

| Context | HotpotQA (Base) | HotpotQA (Long) | Qasper (Base) | Qasper (Long) | GovReport (Base) | GovReport (Long) |
|---|---:|---:|---:|---:|---:|---:|
| 1K | 4.91 | 4.99 | 5.27 | 5.38 | 4.53 | 4.53 |
| 2K | 4.97 | 5.06 | 5.50 | 5.67 | 4.35 | 4.38 |
| 4K | 4.91 | 5.41 | 5.17 | 5.80 | 3.93 | 4.25 |
| 8K | 4.46 | 5.76 | 4.17 | 5.62 | 3.32 | 4.04 |
| 16K | 3.61 | 6.05 | 3.57 | 6.00 | 2.67 | 3.81 |
| 32K | – | – | – | – | 2.09 | 3.56 |

![](https://i.ibb.co/fGk4tm8t/image.png)

### Concurrency Ablations

The columns 1–32 indicate concurrency levels. The baseline is measured in tokens per second (TPS); other rows report speedup relative to the baseline. $\tau$ is the average acceptance length.

| Dataset | Method | 1 | 4 | 8 | 16 | 32 | Avg. τ |
|---|---|---:|---:|---:|---:|---:|---:|
| **GSM8K** | Baseline (TPS) | 249 | 923 | 1739 | 3245 | 5349 | – |
| | EAGLE-3 (10) | 1.6× | 1.5× | 1.4× | 1.2× | 1.0× | 3.49 |
| | EAGLE-3 (60) | 1.9× | 1.6× | 1.3× | 0.9× | 0.6× | 4.55 |
| | **DFlash (10)** | **2.4×** | **2.2×** | **2.1×** | **1.8×** | **1.6×** | **4.32** |
| **HumanEval** | Baseline (TPS) | 245 | 922 | 1778 | 3336 | 5854 | – |
| | EAGLE-3 (10) | 2.0× | 1.9× | 1.8× | 1.5× | 1.2× | 3.62 |
| | EAGLE-3 (60) | 2.0× | 1.7× | 1.3× | 0.9× | 0.6× | 4.65 |
| | **DFlash (10)** | **2.8×** | **2.6×** | **2.5×** | **2.1×** | **1.8×** | **4.91** |
| **Alpaca** | Baseline (TPS) | 245 | 906 | 1745 | 3237 | 5434 | – |
| | EAGLE-3 (10) | 1.5× | 1.4× | 1.4× | 1.1× | 0.9× | 3.11 |
| | EAGLE-3 (60) | 1.8× | 1.5× | 1.2× | 0.8× | 0.5× | 4.07 |
| | **DFlash (10)** | **2.2×** | **2.0×** | **1.8×** | **1.5×** | **1.4×** | **3.73** |

## Questions

- Why does the drafter use a single shared projection matrix $W_c$?
- Why is RMSNorm applied after $W_c$?
- Why is the loss reweighted using this particular exponential decay?

## Thoughts

- Speculative decoding may be particularly effective at generating low-entropy tokens.
- Develop a new mathematical treatment of diffusion samplers.
- Find a way to eliminate long-context degradation.
- Can we fine-tune the teacher to better align it with the drafter?
- Can we predict $M_p$ from $M_q$ more accurately, without providing additional context? Perhaps identify tokens that are easy to generate and sample them preferentially. This may require training a model to estimate token entropy and form token batches based on that entropy.
- If high-entropy tokens can also be predicted correctly, the problem may be more complex than simply distinguishing low-entropy and high-entropy tokens.

## Appendix

### 1. Mathematics

Some basic results from the original [Speculative Decoding](https://arxiv.org/abs/2211.17192) paper.

**Sampling algorithm**

```text
Sample a candidate token x from q(x).
1. If q(x) <= p(x), accept x.
2. Otherwise, accept x with probability p(x) / q(x).
3. If x is rejected, sample a replacement from the normalized
   residual distribution proportional to max(0, p(x) - q(x)).
```

Define the average acceptance probability as

$$
\alpha = \mathbb{E}_{\text{sampled text}}\left[\mathbf{1}\{\text{token accepted}\}\right].
$$

Assume that acceptance decisions can be modeled as independent, identically distributed Bernoulli trials, with acceptance probability $\alpha$ and rejection probability $1-\alpha$. If the drafter proposes $\gamma$ tokens, at most $\gamma+1$ tokens are produced in one speculative decoding step.

The expected number of output tokens per step is

$$
\mathbb{E}[N] = \frac{1-\alpha^{\gamma+1}}{1-\alpha}.
$$

The acceptance probability can be expressed using total variation distance:

$$
\alpha = \mathbb{E}\left[1-\operatorname{TV}(p,q)\right]
= \mathbb{E}\left[\sum_x\min\bigl(p(x),q(x)\bigr)\right].
$$
