# DFLARE: Scaling Up Draft Capacity for Block Diffusion Speculative Decoding
[DFlash](https://arxiv.org/abs/2602.06036) | 
[DFlash1-2 github](https://github.com/z-lab/dflash) |
[DFlare](https://arxiv.org/abs/2606.02091) |
[DSpark](https://arxiv.org/abs/2607.05147) |
[D-PACE](https://arxiv.org/abs/2605.18810)

> [!NOTE]
> This summary assumes that you are familiar with DFlash.

# Problems of DFlash

Quality doesn't grow by the amount of transformer layers.

| Setting | Math500 Speedup | Math500 τ | HumanEval Speedup | HumanEval τ | MT-Bench Speedup | MT-Bench τ |
|---|---:|---:|---:|---:|---:|---:|
| 3-L | 4.69× | 5.64 | 3.90× | 4.61 | 2.38× | 3.18 |
| 5-L | 4.71× | 5.99 | 3.96× | 4.94 | 2.35× | 3.37 |
| 8-L | 4.64× | 6.33 | 3.96× | 5.29 | 2.23× | 3.50 |

# Improvements of DFlare
Layer-wise fusion allows each draft layer have its own information about hidden layers.

1. DFlash : $W_c [l_1, \ldots, l_n]$
2. DFlare : $[l_1, \ldots, l_n] \text{ softmax}(W_c)$ a.k.a learned linear combination of hiddens, for each drafter layer.

![](https://i.ibb.co/x81nNTTp/image.png)

## Dynamic loss decay
- DFlash : $w_k = \exp\left( -\frac{k - 1}{\gamma} \right)$
- DFlare : $\gamma(s) = \gamma_0 + \frac{s}{S} (\gamma_{\max} - \gamma_0)$, $w_k(s) = \exp\left( -\frac{k - 1}{\gamma(s)} \right)$


# Questions
- Why do we need anchor token?
- What info get distributed between layers? (Hiddens are residuals : x + sum vs x + sum + sum)

