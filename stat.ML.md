# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An Ordering of Divergences for Variational Inference with Factorized Gaussian Approximations](https://arxiv.org/abs/2403.13748) | 不同的散度排序可以通过它们的变分近似误估不确定性的各种度量，并且因子化近似无法同时匹配这些度量中的任意两个 |
| [^2] | [FreDF: Learning to Forecast in Frequency Domain](https://arxiv.org/abs/2402.02399) | FreDF是一种在频域中学习预测的方法，解决了时间序列建模中标签序列的自相关问题，相比现有方法有更好的性能表现，并且与各种预测模型兼容。 |
| [^3] | [Closed-Form Diffusion Models.](http://arxiv.org/abs/2310.12395) | 本研究提出了一种闭式扩散模型，通过显式平滑的闭式得分函数来生成新样本，无需训练，且在消费级CPU上能够实现与神经SGMs相竞争的采样速度。 |

# 详细

[^1]: 变分推断中因子化高斯近似的差异排序

    An Ordering of Divergences for Variational Inference with Factorized Gaussian Approximations

    [https://arxiv.org/abs/2403.13748](https://arxiv.org/abs/2403.13748)

    不同的散度排序可以通过它们的变分近似误估不确定性的各种度量，并且因子化近似无法同时匹配这些度量中的任意两个

    

    在变分推断（VI）中，给定一个难以处理的分布$p$，问题是从一些更易处理的族$\mathcal{Q}$中计算最佳近似$q$。通常情况下，这种近似是通过最小化Kullback-Leibler (KL)散度来找到的。然而，存在其他有效的散度选择，当$\mathcal{Q}$不包含$p$时，每个散度都支持不同的解决方案。我们分析了在高斯的密集协方差矩阵被对角协方差矩阵的高斯近似所影响的VI结果中，散度选择如何影响VI结果。在这种设置中，我们展示了不同的散度可以通过它们的变分近似误估不确定性的各种度量，如方差、精度和熵，进行\textit{排序}。我们还得出一个不可能定理，表明无法通过因子化近似同时匹配这些度量中的任意两个；因此

    arXiv:2403.13748v1 Announce Type: cross  Abstract: Given an intractable distribution $p$, the problem of variational inference (VI) is to compute the best approximation $q$ from some more tractable family $\mathcal{Q}$. Most commonly the approximation is found by minimizing a Kullback-Leibler (KL) divergence. However, there exist other valid choices of divergences, and when $\mathcal{Q}$ does not contain~$p$, each divergence champions a different solution. We analyze how the choice of divergence affects the outcome of VI when a Gaussian with a dense covariance matrix is approximated by a Gaussian with a diagonal covariance matrix. In this setting we show that different divergences can be \textit{ordered} by the amount that their variational approximations misestimate various measures of uncertainty, such as the variance, precision, and entropy. We also derive an impossibility theorem showing that no two of these measures can be simultaneously matched by a factorized approximation; henc
    
[^2]: FreDF: 在频域中学习预测

    FreDF: Learning to Forecast in Frequency Domain

    [https://arxiv.org/abs/2402.02399](https://arxiv.org/abs/2402.02399)

    FreDF是一种在频域中学习预测的方法，解决了时间序列建模中标签序列的自相关问题，相比现有方法有更好的性能表现，并且与各种预测模型兼容。

    

    时间序列建模在历史序列和标签序列中都面临自相关的挑战。当前的研究主要集中在处理历史序列中的自相关问题，但往往忽视了标签序列中的自相关存在。具体来说，新兴的预测模型主要遵循直接预测（DF）范式，在标签序列中假设条件独立性下生成多步预测。这种假设忽视了标签序列中固有的自相关性，从而限制了基于DF的模型的性能。针对这一问题，我们引入了频域增强直接预测（FreDF），通过在频域中学习预测来避免标签自相关的复杂性。我们的实验证明，FreDF在性能上大大超过了包括iTransformer在内的现有最先进方法，并且与各种预测模型兼容。

    Time series modeling is uniquely challenged by the presence of autocorrelation in both historical and label sequences. Current research predominantly focuses on handling autocorrelation within the historical sequence but often neglects its presence in the label sequence. Specifically, emerging forecast models mainly conform to the direct forecast (DF) paradigm, generating multi-step forecasts under the assumption of conditional independence within the label sequence. This assumption disregards the inherent autocorrelation in the label sequence, thereby limiting the performance of DF-based models. In response to this gap, we introduce the Frequency-enhanced Direct Forecast (FreDF), which bypasses the complexity of label autocorrelation by learning to forecast in the frequency domain. Our experiments demonstrate that FreDF substantially outperforms existing state-of-the-art methods including iTransformer and is compatible with a variety of forecast models.
    
[^3]: 闭式扩散模型

    Closed-Form Diffusion Models. (arXiv:2310.12395v1 [cs.LG])

    [http://arxiv.org/abs/2310.12395](http://arxiv.org/abs/2310.12395)

    本研究提出了一种闭式扩散模型，通过显式平滑的闭式得分函数来生成新样本，无需训练，且在消费级CPU上能够实现与神经SGMs相竞争的采样速度。

    

    基于得分的生成模型(SGMs)通过迭代地使用扰动目标函数的得分函数来从目标分布中采样。对于任何有限的训练集，可以闭式地评估这个得分函数，但由此得到的SGMs会记忆其训练数据，不能生成新样本。在实践中，可以通过训练神经网络来近似得分函数，但这种近似的误差有助于推广，然而神经SGMs的训练和采样代价高，而且对于这种误差提供的有效正则化方法在理论上尚不清楚。因此，在这项工作中，我们采用显式平滑的闭式得分来获得一个生成新样本的SGMs，而无需训练。我们分析了我们的模型，并提出了一个基于最近邻的高效得分函数估计器。利用这个估计器，我们的方法在消费级CPU上运行时能够达到与神经SGMs相竞争的采样速度。

    Score-based generative models (SGMs) sample from a target distribution by iteratively transforming noise using the score function of the perturbed target. For any finite training set, this score function can be evaluated in closed form, but the resulting SGM memorizes its training data and does not generate novel samples. In practice, one approximates the score by training a neural network via score-matching. The error in this approximation promotes generalization, but neural SGMs are costly to train and sample, and the effective regularization this error provides is not well-understood theoretically. In this work, we instead explicitly smooth the closed-form score to obtain an SGM that generates novel samples without training. We analyze our model and propose an efficient nearest-neighbor-based estimator of its score function. Using this estimator, our method achieves sampling times competitive with neural SGMs while running on consumer-grade CPUs.
    

