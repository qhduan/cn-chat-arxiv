# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Online model error correction with neural networks: application to the Integrated Forecasting System](https://arxiv.org/abs/2403.03702) | 使用神经网络为欧洲中程气象中心的集成预测系统开发模型误差校正，以解决机器学习天气预报模型在表示动力平衡和适用于数据同化实验方面的挑战。 |
| [^2] | [Batched Nonparametric Contextual Bandits](https://arxiv.org/abs/2402.17732) | 该论文研究了批处理约束下的非参数上下文臂问题，提出了一种名为BaSEDB的方案，在动态分割协变量空间的同时，实现了最优的后悔。 |
| [^3] | [Understanding the Training Speedup from Sampling with Approximate Losses](https://arxiv.org/abs/2402.07052) | 本文研究利用近似损失进行样本采样的训练加速方法，通过贪婪策略选择具有大约损失的样本，减少选择的开销，并证明其收敛速度优于随机选择。同时开发了使用中间层表示获取近似损失的SIFT方法，并在训练BERT模型上取得了显著的提升。 |
| [^4] | [Learning Two-Layer Neural Networks, One (Giant) Step at a Time.](http://arxiv.org/abs/2305.18270) | 本文研究了浅层神经网络的训练动态及其条件，证明了动态下梯度下降可以通过有限数量的大批量梯度下降步骤来促进特征学习，并找到了多个和单一方向的最佳批量大小，有助于促进特征学习和方向的专业化。 |

# 详细

[^1]: 在线模型误差校正与神经网络: 应用于集成预测系统

    Online model error correction with neural networks: application to the Integrated Forecasting System

    [https://arxiv.org/abs/2403.03702](https://arxiv.org/abs/2403.03702)

    使用神经网络为欧洲中程气象中心的集成预测系统开发模型误差校正，以解决机器学习天气预报模型在表示动力平衡和适用于数据同化实验方面的挑战。

    

    最近几年，在全球数值天气预报模型的完全数据驱动开发方面取得了显著进展。这些机器学习天气预报模型具有其优势，尤其是准确性和较低的计算需求，但也存在其弱点：它们难以表示基本动力平衡，并且远未适用于资料同化实验。混合建模出现为解决这些限制的一种有希望的方法。混合模型将基于物理的核心组件与统计组件（通常是神经网络）集成在一起，以增强预测能力。在本文中，我们提出使用神经网络为欧洲中程气象中心的运行集成预测系统（IFS）开发模型误差校正。神经网络最初会离线进行预训练，使用大量运行分析数据集

    arXiv:2403.03702v1 Announce Type: cross  Abstract: In recent years, there has been significant progress in the development of fully data-driven global numerical weather prediction models. These machine learning weather prediction models have their strength, notably accuracy and low computational requirements, but also their weakness: they struggle to represent fundamental dynamical balances, and they are far from being suitable for data assimilation experiments. Hybrid modelling emerges as a promising approach to address these limitations. Hybrid models integrate a physics-based core component with a statistical component, typically a neural network, to enhance prediction capabilities. In this article, we propose to develop a model error correction for the operational Integrated Forecasting System (IFS) of the European Centre for Medium-Range Weather Forecasts using a neural network. The neural network is initially pre-trained offline using a large dataset of operational analyses and a
    
[^2]: 批处理非参数上下文臂

    Batched Nonparametric Contextual Bandits

    [https://arxiv.org/abs/2402.17732](https://arxiv.org/abs/2402.17732)

    该论文研究了批处理约束下的非参数上下文臂问题，提出了一种名为BaSEDB的方案，在动态分割协变量空间的同时，实现了最优的后悔。

    

    我们研究了在批处理约束下的非参数上下文臂问题，在这种情况下，每个动作的期望奖励被建模为协变量的平滑函数，并且策略更新是在每个Observations批次结束时进行的。我们为这种设置建立了一个最小化后悔的下限，并提出了一种名为Batched Successive Elimination with Dynamic Binning（BaSEDB）的方案，可以实现最优的后悔（达到对数因子）。实质上，BaSEDB动态地将协变量空间分割成更小的箱子，并仔细调整它们的宽度以符合批次大小。我们还展示了在批处理约束下静态分箱的非最优性，突出了动态分箱的必要性。另外，我们的结果表明，在完全在线设置中，几乎恒定数量的策略更新可以达到最佳后悔。

    arXiv:2402.17732v1 Announce Type: cross  Abstract: We study nonparametric contextual bandits under batch constraints, where the expected reward for each action is modeled as a smooth function of covariates, and the policy updates are made at the end of each batch of observations. We establish a minimax regret lower bound for this setting and propose Batched Successive Elimination with Dynamic Binning (BaSEDB) that achieves optimal regret (up to logarithmic factors). In essence, BaSEDB dynamically splits the covariate space into smaller bins, carefully aligning their widths with the batch size. We also show the suboptimality of static binning under batch constraints, highlighting the necessity of dynamic binning. Additionally, our results suggest that a nearly constant number of policy updates can attain optimal regret in the fully online setting.
    
[^3]: 理解通过使用近似损失进行采样的训练加速

    Understanding the Training Speedup from Sampling with Approximate Losses

    [https://arxiv.org/abs/2402.07052](https://arxiv.org/abs/2402.07052)

    本文研究利用近似损失进行样本采样的训练加速方法，通过贪婪策略选择具有大约损失的样本，减少选择的开销，并证明其收敛速度优于随机选择。同时开发了使用中间层表示获取近似损失的SIFT方法，并在训练BERT模型上取得了显著的提升。

    

    众所周知，选择具有较大损失/梯度的样本可以显著减少训练步骤的数量。然而，选择的开销往往过高，无法在总体训练时间方面获得有意义的提升。在本文中，我们专注于选择具有大约损失的样本的贪婪方法，而不是准确损失，以减少选择的开销。对于平滑凸损失，我们证明了这种贪婪策略可以在比随机选择更少的迭代次数内收敛到平均损失的最小值的常数因子。我们还理论上量化了近似水平的影响。然后，我们开发了使用中间层表示获取近似损失以进行样本选择的SIFT。我们评估了SIFT在训练一个具有1.1亿参数的12层BERT基础模型上的任务，并展示了显著的提升（以训练时间和反向传播步骤的数量衡量）。

    It is well known that selecting samples with large losses/gradients can significantly reduce the number of training steps. However, the selection overhead is often too high to yield any meaningful gains in terms of overall training time. In this work, we focus on the greedy approach of selecting samples with large \textit{approximate losses} instead of exact losses in order to reduce the selection overhead. For smooth convex losses, we show that such a greedy strategy can converge to a constant factor of the minimum value of the average loss in fewer iterations than the standard approach of random selection. We also theoretically quantify the effect of the approximation level. We then develop SIFT which uses early exiting to obtain approximate losses with an intermediate layer's representations for sample selection. We evaluate SIFT on the task of training a 110M parameter 12-layer BERT base model and show significant gains (in terms of training hours and number of backpropagation step
    
[^4]: 学习两层神经网络：一次(巨大)的步骤。

    Learning Two-Layer Neural Networks, One (Giant) Step at a Time. (arXiv:2305.18270v1 [stat.ML])

    [http://arxiv.org/abs/2305.18270](http://arxiv.org/abs/2305.18270)

    本文研究了浅层神经网络的训练动态及其条件，证明了动态下梯度下降可以通过有限数量的大批量梯度下降步骤来促进特征学习，并找到了多个和单一方向的最佳批量大小，有助于促进特征学习和方向的专业化。

    

    我们研究了浅层神经网络的训练动态，研究了有限数量的大批量梯度下降步骤有助于在核心范围之外促进特征学习的条件。我们比较了批量大小和多个(但有限的)步骤的影响。我们分析了单步骤过程，发现批量大小为$n=O(d)$可以促进特征学习，但只适合学习单一方向或单索引模型。相比之下，$n=O(d^2)$对于学习多个方向和专业化至关重要。此外，我们证明“硬”方向缺乏前$\ell$个Hermite系数，仍未被发现，并且需要批量大小为$n=O(d^\ell)$才能被梯度下降捕获。经过几次迭代，情况发生变化：批量大小为$n=O(d)$足以学习新的目标方向，这些方向在Hermite基础上线性连接到之前学习的方向所涵盖的子空间。

    We study the training dynamics of shallow neural networks, investigating the conditions under which a limited number of large batch gradient descent steps can facilitate feature learning beyond the kernel regime. We compare the influence of batch size and that of multiple (but finitely many) steps. Our analysis of a single-step process reveals that while a batch size of $n = O(d)$ enables feature learning, it is only adequate for learning a single direction, or a single-index model. In contrast, $n = O(d^2)$ is essential for learning multiple directions and specialization. Moreover, we demonstrate that ``hard'' directions, which lack the first $\ell$ Hermite coefficients, remain unobserved and require a batch size of $n = O(d^\ell)$ for being captured by gradient descent. Upon iterating a few steps, the scenario changes: a batch-size of $n = O(d)$ is enough to learn new target directions spanning the subspace linearly connected in the Hermite basis to the previously learned directions,
    

