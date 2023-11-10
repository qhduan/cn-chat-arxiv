# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ATMS: Algorithmic Trading-Guided Market Simulation.](http://arxiv.org/abs/2309.01784) | 本文提出了一种算法交易引导的市场模拟方法(ATMS)，通过优化提出的度量标准，该方法能够适应交易活动的序列和动态特性。实验结果表明，在半真实市场上，该方法取得了良好的效果。 |
| [^2] | [Exact solution to a generalised Lillo-Mike-Farmer model with heterogeneous order-splitting strategies.](http://arxiv.org/abs/2306.13378) | 本研究提出了一种广义 LMF 模型，考虑到了交易者的订单拆分行为的异质性，对该模型进行了精确求解，发现订单符号 ACF 中幂律指数稳健，但前置因子对交易策略异质性极为敏感并在同质 LMF 模型中被低估。 |

# 详细

[^1]: ATMS: 算法交易引导的市场模拟

    ATMS: Algorithmic Trading-Guided Market Simulation. (arXiv:2309.01784v1 [cs.LG])

    [http://arxiv.org/abs/2309.01784](http://arxiv.org/abs/2309.01784)

    本文提出了一种算法交易引导的市场模拟方法(ATMS)，通过优化提出的度量标准，该方法能够适应交易活动的序列和动态特性。实验结果表明，在半真实市场上，该方法取得了良好的效果。

    

    有效构建算法交易策略通常依赖于市场模拟器，然而现有方法很难适应交易活动的序列和动态特性。本文通过提出一个衡量市场差异的度量标准来填补这一空白。该度量标准通过算法交易代理和市场之间的交互来评估底层市场的因果效应差异。最重要的是，我们引入了算法交易引导的市场模拟(ATMS)，通过优化我们提出的度量标准。受SeqGAN的启发，ATMS将模拟器形式化为强化学习中的随机策略，以考虑交易的序列特性。此外，ATMS利用策略梯度更新来绕过对提出的度量标准的微分，这涉及到非可微分操作，如从市场中删除订单。通过在半真实市场上进行大量的实验

    The effective construction of an Algorithmic Trading (AT) strategy often relies on market simulators, which remains challenging due to existing methods' inability to adapt to the sequential and dynamic nature of trading activities. This work fills this gap by proposing a metric to quantify market discrepancy. This metric measures the difference between a causal effect from underlying market unique characteristics and it is evaluated through the interaction between the AT agent and the market. Most importantly, we introduce Algorithmic Trading-guided Market Simulation (ATMS) by optimizing our proposed metric. Inspired by SeqGAN, ATMS formulates the simulator as a stochastic policy in reinforcement learning (RL) to account for the sequential nature of trading. Moreover, ATMS utilizes the policy gradient update to bypass differentiating the proposed metric, which involves non-differentiable operations such as order deletion from the market. Through extensive experiments on semi-real marke
    
[^2]: 具有异质性订单拆分策略的一般化 Lillo-Mike-Farmer 模型的精确解

    Exact solution to a generalised Lillo-Mike-Farmer model with heterogeneous order-splitting strategies. (arXiv:2306.13378v1 [q-fin.TR])

    [http://arxiv.org/abs/2306.13378](http://arxiv.org/abs/2306.13378)

    本研究提出了一种广义 LMF 模型，考虑到了交易者的订单拆分行为的异质性，对该模型进行了精确求解，发现订单符号 ACF 中幂律指数稳健，但前置因子对交易策略异质性极为敏感并在同质 LMF 模型中被低估。

    

    Lillo-Mike-Farmer（LMF）模型是一个与经济物理学相关的模型，描述机构投资者在金融市场中的订单拆分行为。LMF 假设交易者的订单拆分策略是同质的，并基于几种启发式推理导出了订单符号自相关函数（ACF）的幂律渐近解。本文提出了一种广义 LMF 模型，通过将交易者的订单拆分行为异质化，对该模型进行了精确求解且无需引入启发式方法。我们发现，订单符号ACF中的幂律指数对于任意异质的强度分布是稳健的。另一方面，ACF中的前置因子对交易策略的异质性非常敏感，并且在原始同质 LMF 模型中被系统地低估。我们的工作强调，在解释ACF前置因子方面需要更加小心谨慎。

    The Lillo-Mike-Farmer (LMF) model is an established econophysics model describing the order-splitting behaviour of institutional investors in financial markets. In the original article (LMF, Physical Review E 71, 066122 (2005)), LMF assumed the homogeneity of the traders' order-splitting strategy and derived a power-law asymptotic solution to the order-sign autocorrelation function (ACF) based on several heuristic reasonings. This report proposes a generalised LMF model by incorporating the heterogeneity of traders' order-splitting behaviour that is exactly solved without heuristics. We find that the power-law exponent in the order-sign ACF is robust for arbitrary heterogeneous intensity distributions. On the other hand, the prefactor in the ACF is very sensitive to heterogeneity in trading strategies and is shown to be systematically underestimated in the original homogeneous LMF model. Our work highlights that the ACF prefactor should be more carefully interpreted than the ACF power-
    

