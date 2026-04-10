# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Time series generation for option pricing on quantum computers using tensor network](https://arxiv.org/abs/2402.17148) | 提出了一种使用矩阵乘积态作为时间序列生成的方法，可以有效生成多个时间点处基础资产价格的联合分布的态，并证实了该方法在Heston模型中的可行性。 |

# 详细

[^1]: 使用张量网络在量子计算机上生成期权定价的时间序列

    Time series generation for option pricing on quantum computers using tensor network

    [https://arxiv.org/abs/2402.17148](https://arxiv.org/abs/2402.17148)

    提出了一种使用矩阵乘积态作为时间序列生成的方法，可以有效生成多个时间点处基础资产价格的联合分布的态，并证实了该方法在Heston模型中的可行性。

    

    金融，特别是期权定价，是一个有望从量子计算中受益的行业。尽管已经提出了用于期权定价的量子算法，但人们希望在算法中设计出更高效的实现方式，其中之一是准备编码基础资产价格概率分布的量子态。特别是在定价依赖路径的期权时，我们需要生成一个编码多个时间点处基础资产价格的联合分布的态，这更具挑战性。为解决这些问题，我们提出了一种使用矩阵乘积态（MPS）作为时间序列生成的生成模型的新方法。为了验证我们的方法，以Heston模型为目标，我们进行数值实验以在模型中生成时间序列。我们的研究结果表明MPS模型能够生成Heston模型中的路径，突显了...

    arXiv:2402.17148v1 Announce Type: cross  Abstract: Finance, especially option pricing, is a promising industrial field that might benefit from quantum computing. While quantum algorithms for option pricing have been proposed, it is desired to devise more efficient implementations of costly operations in the algorithms, one of which is preparing a quantum state that encodes a probability distribution of the underlying asset price. In particular, in pricing a path-dependent option, we need to generate a state encoding a joint distribution of the underlying asset price at multiple time points, which is more demanding. To address these issues, we propose a novel approach using Matrix Product State (MPS) as a generative model for time series generation. To validate our approach, taking the Heston model as a target, we conduct numerical experiments to generate time series in the model. Our findings demonstrate the capability of the MPS model to generate paths in the Heston model, highlightin
    

