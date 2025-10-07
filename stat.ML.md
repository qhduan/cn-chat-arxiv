# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fitted Value Iteration Methods for Bicausal Optimal Transport.](http://arxiv.org/abs/2306.12658) | 本文提出了一种适用于双因果最优传输问题的拟合值迭代方法，能够在保证精度的同时具有良好的可扩展性，数值实验结果也证明了该方法的优越性。 |
| [^2] | [Quick Adaptive Ternary Segmentation: An Efficient Decoding Procedure For Hidden Markov Models.](http://arxiv.org/abs/2305.18578) | 提出了一种名为QATS的新方法，用于高效解码隐藏马尔可夫模型序列。它的计算复杂性为多对数和立方，特别适用于具有相对较少状态的大型HMM。 |

# 详细

[^1]: 拟合值迭代方法求解适应结构双因果最优传输问题

    Fitted Value Iteration Methods for Bicausal Optimal Transport. (arXiv:2306.12658v1 [stat.ML])

    [http://arxiv.org/abs/2306.12658](http://arxiv.org/abs/2306.12658)

    本文提出了一种适用于双因果最优传输问题的拟合值迭代方法，能够在保证精度的同时具有良好的可扩展性，数值实验结果也证明了该方法的优越性。

    

    本文提出一种拟合值迭代方法(FVI)用于计算具有适应结构的双因果最优传输(OT)。基于动态规划的形式化表述，FVI采用函数类用于近似双因果OT中的值函数。在可集中条件和近似完备性假设下，我们使用（局部）Rademacher复杂度证明了样本复杂度。此外，我们证明了深度多层神经网络具有适当结构，满足样本复杂度证明所需的关键假设条件。数值实验表明，FVI在时间跨度增加时优于线性规划和适应性Sinkhorn方法，在保持可接受精度的同时具有很好的可扩展性。

    We develop a fitted value iteration (FVI) method to compute bicausal optimal transport (OT) where couplings have an adapted structure. Based on the dynamic programming formulation, FVI adopts a function class to approximate the value functions in bicausal OT. Under the concentrability condition and approximate completeness assumption, we prove the sample complexity using (local) Rademacher complexity. Furthermore, we demonstrate that multilayer neural networks with appropriate structures satisfy the crucial assumptions required in sample complexity proofs. Numerical experiments reveal that FVI outperforms linear programming and adapted Sinkhorn methods in scalability as the time horizon increases, while still maintaining acceptable accuracy.
    
[^2]: 快速自适应三元分割：隐马尔可夫模型的有效解码程序。

    Quick Adaptive Ternary Segmentation: An Efficient Decoding Procedure For Hidden Markov Models. (arXiv:2305.18578v1 [stat.ME])

    [http://arxiv.org/abs/2305.18578](http://arxiv.org/abs/2305.18578)

    提出了一种名为QATS的新方法，用于高效解码隐藏马尔可夫模型序列。它的计算复杂性为多对数和立方，特别适用于具有相对较少状态的大型HMM。

    

    隐马尔可夫模型（HMM）以不可观察的（隐藏的）马尔可夫链和可观测的过程为特征，后者是隐藏链的噪声版本。从嘈杂的观测中解码原始信号（即隐藏链）是几乎所有基于HMM的数据分析的主要目标。现有的解码算法，如维特比算法，在观测序列长度最多线性的情况下具有计算复杂度，并且在马尔可夫链状态空间的大小中具有次二次计算复杂度。我们提出了快速自适应三元分割（QATS），这是一种分而治之的过程，可在序列长度的多对数计算复杂度和马尔可夫链状态空间的三次计算复杂度下解码隐藏的序列，因此特别适用于具有相对较少状态的大规模HMM。该程序还建议一种有效的数据存储方式，即特定的累积总和。实质上，估计的状态序列按顺序最大化局部似然。

    Hidden Markov models (HMMs) are characterized by an unobservable (hidden) Markov chain and an observable process, which is a noisy version of the hidden chain. Decoding the original signal (i.e., hidden chain) from the noisy observations is one of the main goals in nearly all HMM based data analyses. Existing decoding algorithms such as the Viterbi algorithm have computational complexity at best linear in the length of the observed sequence, and sub-quadratic in the size of the state space of the Markov chain. We present Quick Adaptive Ternary Segmentation (QATS), a divide-and-conquer procedure which decodes the hidden sequence in polylogarithmic computational complexity in the length of the sequence, and cubic in the size of the state space, hence particularly suited for large scale HMMs with relatively few states. The procedure also suggests an effective way of data storage as specific cumulative sums. In essence, the estimated sequence of states sequentially maximizes local likeliho
    

