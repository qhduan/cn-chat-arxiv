# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ConjNorm: Tractable Density Estimation for Out-of-Distribution Detection](https://arxiv.org/abs/2402.17888) | 提出了一种新颖的理论框架，基于Bregman散度，通过引入共轭约束，提出了一种\textsc{ConjNorm}方法，以在给定数据集中搜索最佳规范系数$p$来重新构想密度函数设计。 |
| [^2] | [Moonwalk: Inverse-Forward Differentiation](https://arxiv.org/abs/2402.14212) | Moonwalk引入了一种基于向量-逆-Jacobian乘积的新技术，加速前向梯度计算，显著减少内存占用，并在保持真实梯度准确性的同时，将计算时间降低了几个数量级。 |

# 详细

[^1]: ConjNorm：用于异常分布检测的可处理密度估计

    ConjNorm: Tractable Density Estimation for Out-of-Distribution Detection

    [https://arxiv.org/abs/2402.17888](https://arxiv.org/abs/2402.17888)

    提出了一种新颖的理论框架，基于Bregman散度，通过引入共轭约束，提出了一种\textsc{ConjNorm}方法，以在给定数据集中搜索最佳规范系数$p$来重新构想密度函数设计。

    

    后续异常分布（OOD）检测在可靠机器学习中受到密切关注。许多工作致力于推导基于logits、距离或严格数据分布假设的评分函数，以识别得分低的OOD样本。然而，这些估计得分可能无法准确反映真实数据密度或施加不切实际的约束。为了在基于密度得分设计方面提供一个统一的视角，我们提出了一个以Bregman散度为基础的新颖理论框架，该框架将分布考虑扩展到涵盖一系列指数族分布。利用我们定理中揭示的共轭约束，我们引入了一种\textsc{ConjNorm}方法，将密度函数设计重新构想为针对给定数据集搜索最佳规范系数$p$的过程。鉴于归一化的计算挑战，我们设计了一种无偏和解析可追踪的方法

    arXiv:2402.17888v1 Announce Type: cross  Abstract: Post-hoc out-of-distribution (OOD) detection has garnered intensive attention in reliable machine learning. Many efforts have been dedicated to deriving score functions based on logits, distances, or rigorous data distribution assumptions to identify low-scoring OOD samples. Nevertheless, these estimate scores may fail to accurately reflect the true data density or impose impractical constraints. To provide a unified perspective on density-based score design, we propose a novel theoretical framework grounded in Bregman divergence, which extends distribution considerations to encompass an exponential family of distributions. Leveraging the conjugation constraint revealed in our theorem, we introduce a \textsc{ConjNorm} method, reframing density function design as a search for the optimal norm coefficient $p$ against the given dataset. In light of the computational challenges of normalization, we devise an unbiased and analytically tract
    
[^2]: Moonwalk：逆向-前向微分

    Moonwalk: Inverse-Forward Differentiation

    [https://arxiv.org/abs/2402.14212](https://arxiv.org/abs/2402.14212)

    Moonwalk引入了一种基于向量-逆-Jacobian乘积的新技术，加速前向梯度计算，显著减少内存占用，并在保持真实梯度准确性的同时，将计算时间降低了几个数量级。

    

    反向传播虽然在梯度计算方面有效，但在解决内存消耗和扩展性方面表现不佳。这项工作探索了前向梯度计算作为可逆网络中的一种替代方法，展示了它在减少内存占用的潜力，并不带来重大缺点。我们引入了一种基于向量-逆-Jacobian乘积的新技术，加速了前向梯度的计算，同时保留了减少内存和保持真实梯度准确性的优势。我们的方法Moonwalk在网络深度方面具有线性时间复杂度，与朴素前向的二次时间复杂度相比，在没有分配更多内存的情况下，从实证的角度减少了几个数量级的计算时间。我们进一步通过将Moonwalk与反向模式微分相结合来加速，以实现与反向传播相当的时间复杂度，同时保持更小的内存使用量。

    arXiv:2402.14212v1 Announce Type: cross  Abstract: Backpropagation, while effective for gradient computation, falls short in addressing memory consumption, limiting scalability. This work explores forward-mode gradient computation as an alternative in invertible networks, showing its potential to reduce the memory footprint without substantial drawbacks. We introduce a novel technique based on a vector-inverse-Jacobian product that accelerates the computation of forward gradients while retaining the advantages of memory reduction and preserving the fidelity of true gradients. Our method, Moonwalk, has a time complexity linear in the depth of the network, unlike the quadratic time complexity of na\"ive forward, and empirically reduces computation time by several orders of magnitude without allocating more memory. We further accelerate Moonwalk by combining it with reverse-mode differentiation to achieve time complexity comparable with backpropagation while maintaining a much smaller mem
    

