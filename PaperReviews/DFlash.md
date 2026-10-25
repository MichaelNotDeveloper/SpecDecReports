# DFlash: Block Diffusion for Flash Speculative Decoding
[DFlash](https://arxiv.org/abs/2602.06036) | 
[DFlash1-2 github](https://github.com/z-lab/dflash) |
[DFlare](https://arxiv.org/abs/2606.02091) |
[DSpark](https://arxiv.org/abs/2607.05147) |
[D-PACE](https://arxiv.org/abs/2605.18810)


# Introduction
Main improvements come from two primitives :
1. Generate drafter's tokens in one run
2. Use Information from teacher model to generate future tokens.

# Architecture
1. We use information from target model.
2. Try to condition drafter on target model as strong as possible.

### How to use teacher's KV-cache
```
1. Choose set of trasformer layers [l1, ..., ln] of teacher model
2. Take all history of hidden states of desired layers (residual stream of theacher model): ht
3. Create One Learned matrix Wc (hidden_size x (hidden_size * n))
4. Project hiddens via Wc (concatenate hiddens from all layers, and project each hidden to normal size using Wc): ht
5. Apply RMSNorm.
6. Now compute Transfomer layers of drafter, passing hd to Q and concat[ht, hd] to KV, where hd are new masked tokens, for hd tokens use full mask (block alike, search for image lower)
```
![](https://i.ibb.co/Pz5NLssC/image.png)

### How to Train : loss & more
We are training diffusion drafter to *imitate* teacher. So we sample [question -> answer] trajectories, sample random anchor-indexes (wih teacher KV-cache), and force drafter to generate masked tokens, using anchor tokens as start. 

Since we condition to imitate teacher model, we use cross entropy, and reweigt it, by how close token is to its anchor (so earlier tokens would generate more correctly)

$$
\mathcal{L}(\theta)=
-\sum_b\sum_{k\in\mathcal{M}_b}
\exp \left(-\frac{k-1}{\gamma}\right)
\log p_\theta(y_{b,k}\mid\cdot)
$$

> [!NOTE]
> We mask everything except first anchor token.

> [!NOTE]
> We choos layers to sample from 

> [!NOTE]
> Random sampling of anchors improves convergence.

> [!NOTE]
> Loss reweight (loss decay) is empirical, but slightly helps.

> [!TIP]
> Use continiuos block of anchors while traning, so you can batch training for  KV cache prefix.
> ![](https://i.ibb.co/YBT3qF4P/image.png)

# Datasets & Bencmarks
### Training
They  _probably_ did 9 epochs for training\
Training was built on *answers* of trainer, not the answers of datasets.\
Datasets:
1. [UltraChat](https://huggingface.co/datasets/openbmb/UltraChat)
2. [ShareGPT](https://huggingface.co/collections/bunnycore/sharegpt-datasets)

Same training data as EAGLE-3

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


## Long Context Ablation

Drafter average acceptance rate

> [!NOTE]
> Acceptance rate strongly depends on tasks.

> [!NOTE]
> Long drafter is drafter fine-tuned on long context.

| Context | HotpotQA (Base) | HotpotQA (Long) | Qasper (Base) | Qasper (Long) | GovReport (Base) | GovReport (Long) |
|---|---:|---:|---:|---:|---:|---:|
| 1K | 4.91 | 4.99 | 5.27 | 5.38 | 4.53 | 4.53 |
| 2K | 4.97 | 5.06 | 5.50 | 5.67 | 4.35 | 4.38 |
| 4K | 4.91 | 5.41 | 5.17 | 5.80 | 3.93 | 4.25 |
| 8K | 4.46 | 5.76 | 4.17 | 5.62 | 3.32 | 4.04 |
| 16K | 3.61 | 6.05 | 3.57 | 6.00 | 2.67 | 3.81 |
| 32K | – | – | – | – | 2.09 | 3.56 |

![](https://i.ibb.co/fGk4tm8t/image.png)

## Concurrency Ablations

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

# Questions
- Why does drafter use unified matrix Wc?
- Why do we use RMS norm after Wc?
- Why do we reweight loss in this specific style?

# Thoughts
- Specdec might easily generate low-entropy tokens.
- new math for diffusion samplers
- Get rid of long-context degradation
- Can we finetune the teacher to adjust from drafter?
- Find ways to predict $M_p$ with $M_q$ better. (not new context better) (Somehow understand easy-to generated tokens and sample them.) - We have to teach a model to predict entropy to entropy token batches.
- If model generates high-entropy tokens correctly then the porblem might be more complex


# Appendix

## 1. Mathematics
Some basic propositions from original [SpecDec](https://arxiv.org/abs/2211.17192) paper:

Sampling algo :
```
Generate tokens from q(x).
1. Keep tokens if q(x) < p(x)
2. Keep tokens with prob. p(x)/q(x)
3. Generate new, from max(0, p(x) - q(x))
```

Set $\alpha = \mathbb E_\text{sampled text} [I ( \text{token kept} )]$

If we make an assumtion, that new token is kept can be simulated as unfair coin-flips $\gamma$ (max-amount if generated tokens is $\gamma + 1$) times, with success rate of $1 - \alpha$. We get desired speed-up rate.

Expected amount of tokens : $\frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$

Alpha can be calculated, as

$$
\alpha=\mathbb E\left[1 - TV(q(x) - p(x))\right] = \mathbb E \left[\max (q(x), p(x))\right]
$$

