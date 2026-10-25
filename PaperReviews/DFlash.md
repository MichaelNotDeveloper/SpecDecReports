# DFlash: Block Diffusion for Flash Speculative Decoding
[DFlash](https://arxiv.org/abs/2602.06036) | 
[DFlash1-2 github](https://github.com/z-lab/dflash) |
[DFlare](https://arxiv.org/abs/2606.02091) |
[DSpark](https://arxiv.org/abs/2607.05147) |
[D-PACE](https://arxiv.org/abs/2605.18810)


# Introduction
Main improvements come from two primitives :
1. Generate drafter tokens in one run
2. Use Information from teacher model to generate future tokens.

# Mathematics
Main improvements come from anaylsing how speculative decoding improves infernece.

Some basic propositions from original [SpecDec](https://arxiv.org/abs/2211.17192) paper:


Sampling algo :
```
Generate tokens from q(x).
1. Keep tokens if q(x) < p(x)
2. Keep tokens with prob. p(x)/q(x)
3. Generate new, from max(0, p(x) - q(x))
```

Set $\alpha = \mathbb E_\text{sampled text} [I\{\text{token kept}\}$

If we make an assumtion, that new token is kept can be simulated as unfair coin-flips $\gamma$ (max-amount if generated tokens is $\gamma + 1$) times, with success rate of $1 - \alpha$. We get desired speed-up rate.

Expected amount of tokens : $\frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$

Alpha can be calculated, as
$$
\alpha = \mathbb E \left[1 - TV(q(x) - p(x))\right] = \mathbb E \left[ \max (q(x), p(x))\right]
$$

$$

$$

# Architecture
1. We use information from target model.
2. Try to condition drafter on target model as strong as possible.

### How to use teacher's KV-cache
```
1. Choose set of trasformer layers [l1, ..., ln]
2. Take hidden states of desired layers (residual stream) : ht
3. Create One Learned matrix Wc
4. Project hiddens via this matrix : hd
5. Now compute Transfomer layers, passing hd to Q and ht to KV
```

### How to Train : loss & more
We are training diffusion drafter to *imitate* teacher. So we sample [question -> answer] trajectories, sample random anchor-indexes (wih teacher KV-cache), and force drafter to generate masked tokens, using anchor tokens as start.

[!NOTE]



# Thoughts
- Specdec might easily generate low-entropy tokens.
- new math for diffusion samplers
- Get rid of long-context degradation
- Can we finetune the teacher to adjust from drafter?
- Find ways to predict $M_p$ with $M_q$ better. (not new context better) (Somehow understand easy-to generated tokens and sample them.) - We have to teach a model to predict entropy to entropy token batches.
- If model generates high-entropy tokens correctly then the porblem might be more complex