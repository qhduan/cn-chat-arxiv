# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Graph Neural Networks for Learning Equivariant Representations of Neural Networks](https://arxiv.org/abs/2403.12143) | 本研究提出了将神经网络表示为参数的计算图的方法，利用图神经网络和变压器来实现置换对称性，使得单个模型能够处理具有多种架构的神经计算图。 |
| [^2] | [Interpretable Machine Learning for TabPFN](https://arxiv.org/abs/2403.10923) | TabPFN模型在低数据情况下实现了良好的分类性能，并能够以秒级速度生成后验预测分布，我们提出了几种专为TabPFN设计的可解释性方法的改进，实现了更高效的计算。 |
| [^3] | [Minimax Optimality of Score-based Diffusion Models: Beyond the Density Lower Bound Assumptions](https://arxiv.org/abs/2402.15602) | 该研究展示了基于分数的扩散模型的采样具有极小均方误差，可以获得扩散模型生成样本的总变差误差的上界，这突破了仅做次高斯假设的限制。 |
| [^4] | [Removing Spurious Concepts from Neural Network Representations via Joint Subspace Estimation.](http://arxiv.org/abs/2310.11991) | 该论文提出了一种通过联合子空间估计从神经网络表示中消除错误概念的迭代算法，并在计算机视觉和自然语言处理的基准数据集上展示了其优越性能。 |
| [^5] | [Batched Predictors Generalize within Distribution.](http://arxiv.org/abs/2307.09379) | 批量预测器提供了指数级更强的泛化保证，可应用于离线测试前化合物质量的预测任务。 |
| [^6] | [Sharp Convergence Rates for Matching Pursuit.](http://arxiv.org/abs/2307.07679) | 本文通过提升现有的下界来匹配最佳上界，对匹配追踪的性能进行了精确描述，并构造了一个最坏情况的字典来证明现有上界的无法改进。 |
| [^7] | [Mixture of segmentation for heterogeneous functional data.](http://arxiv.org/abs/2303.10712) | 本文提出了一种混合分割模型，可以处理异质性功能数据，通过动态规划的EM算法近似最大似然估计器，方法在模拟与真实数据集上得到验证。 |
| [^8] | [First-order ANIL learns linear representations despite misspecified latent dimension.](http://arxiv.org/abs/2303.01335) | 本研究表明，在存在架构误指定的情况下，初阶ANIL可以成功学习到线性的共享表示。这个结果是基于对无限数量任务的极限情况下的推导。 |

# 详细

[^1]: 用于学习神经网络等变表示的图神经网络

    Graph Neural Networks for Learning Equivariant Representations of Neural Networks

    [https://arxiv.org/abs/2403.12143](https://arxiv.org/abs/2403.12143)

    本研究提出了将神经网络表示为参数的计算图的方法，利用图神经网络和变压器来实现置换对称性，使得单个模型能够处理具有多种架构的神经计算图。

    

    处理其他神经网络参数的神经网络在诸如分类隐式神经表示、生成神经网络权重和预测泛化错误等领域中得到应用。然而，现有方法要么忽视神经网络中固有的置换对称性，要么依赖复杂的权重共享模式来实现等变性，同时忽略网络架构本身的影响。在本文中，我们提出将神经网络表示为参数的计算图，这使我们能够利用强大的保留置换对称性的图神经网络和变压器。因此，我们的方法使得单个模型能够对具有多样架构的神经计算图进行编码。我们展示了我们的方法在包括分类和编辑隐式神经表示、预测泛化错误等多种任务中的有效性。

    arXiv:2403.12143v1 Announce Type: cross  Abstract: Neural networks that process the parameters of other neural networks find applications in domains as diverse as classifying implicit neural representations, generating neural network weights, and predicting generalization errors. However, existing approaches either overlook the inherent permutation symmetry in the neural network or rely on intricate weight-sharing patterns to achieve equivariance, while ignoring the impact of the network architecture itself. In this work, we propose to represent neural networks as computational graphs of parameters, which allows us to harness powerful graph neural networks and transformers that preserve permutation symmetry. Consequently, our approach enables a single model to encode neural computational graphs with diverse architectures. We showcase the effectiveness of our method on a wide range of tasks, including classification and editing of implicit neural representations, predicting generalizati
    
[^2]: TabPFN的可解释机器学习

    Interpretable Machine Learning for TabPFN

    [https://arxiv.org/abs/2403.10923](https://arxiv.org/abs/2403.10923)

    TabPFN模型在低数据情况下实现了良好的分类性能，并能够以秒级速度生成后验预测分布，我们提出了几种专为TabPFN设计的可解释性方法的改进，实现了更高效的计算。

    

    最近开发的Prior-Data Fitted Networks（PFNs）已经显示出在低数据情况下具有非常有希望的应用结果。TabPFN模型是PFN的一种特殊情况，适用于表格数据，在不需要学习参数或超参数调整的情况下，能够在短短几秒钟内实现多种分类任务的最先进性能，并且能够生成后验预测分布。TabPFN因此成为了许多领域应用中非常吸引人的选择。然而，该方法的一个主要缺点是缺乏可解释性。因此，我们提出了几种针对TabPFN专门设计的流行解释性方法的改进。通过利用该模型的独特性质，我们的改进允许比现有实现更高效的计算。特别是，我们展示了通过避免...

    arXiv:2403.10923v1 Announce Type: cross  Abstract: The recently developed Prior-Data Fitted Networks (PFNs) have shown very promising results for applications in low-data regimes. The TabPFN model, a special case of PFNs for tabular data, is able to achieve state-of-the-art performance on a variety of classification tasks while producing posterior predictive distributions in mere seconds by in-context learning without the need for learning parameters or hyperparameter tuning. This makes TabPFN a very attractive option for a wide range of domain applications. However, a major drawback of the method is its lack of interpretability. Therefore, we propose several adaptations of popular interpretability methods that we specifically design for TabPFN. By taking advantage of the unique properties of the model, our adaptations allow for more efficient computations than existing implementations. In particular, we show how in-context learning facilitates the estimation of Shapley values by avoid
    
[^3]: 基于分数的扩散模型的极小化最优性：超越密度下界假设

    Minimax Optimality of Score-based Diffusion Models: Beyond the Density Lower Bound Assumptions

    [https://arxiv.org/abs/2402.15602](https://arxiv.org/abs/2402.15602)

    该研究展示了基于分数的扩散模型的采样具有极小均方误差，可以获得扩散模型生成样本的总变差误差的上界，这突破了仅做次高斯假设的限制。

    

    我们从非参数统计的角度研究了在大样本场景下得分扩散模型抽样的渐近误差。我们展示了基于核的得分估计器可以实现对 $p_0*\mathcal{N}(0,t\boldsymbol{I}_d)$ 的得分函数的最优均方误差为 $\widetilde{O}\left(n^{-1} t^{-\frac{d+2}{2}}(t^{\frac{d}{2}} \vee 1)\right)$，其中 $n$ 和 $d$ 分别代表样本大小和维度，$t$ 在上下受到 $n$ 的多项式的限制，并且 $p_0$ 是任意次亚高斯分布。因此，这导致在仅进行次高斯假设时，扩散模型生成的样本分布的总变差误差的上界为 $\widetilde{O}\left(n^{-1/2} t^{-\frac{d}{4}}\right)$。如果此外，$p_0$ 属于 $\beta\le 2$ 的 $\beta$-Sobolev空间的非参数族，通过采用早停策略，我们得到该扩散模型的样本的分布的总变差误差的上界为 $\widetilde{O}\left(n^{-1/2} t^{-\frac{d}{4}}\right)$。

    arXiv:2402.15602v1 Announce Type: cross  Abstract: We study the asymptotic error of score-based diffusion model sampling in large-sample scenarios from a non-parametric statistics perspective. We show that a kernel-based score estimator achieves an optimal mean square error of $\widetilde{O}\left(n^{-1} t^{-\frac{d+2}{2}}(t^{\frac{d}{2}} \vee 1)\right)$ for the score function of $p_0*\mathcal{N}(0,t\boldsymbol{I}_d)$, where $n$ and $d$ represent the sample size and the dimension, $t$ is bounded above and below by polynomials of $n$, and $p_0$ is an arbitrary sub-Gaussian distribution. As a consequence, this yields an $\widetilde{O}\left(n^{-1/2} t^{-\frac{d}{4}}\right)$ upper bound for the total variation error of the distribution of the sample generated by the diffusion model under a mere sub-Gaussian assumption. If in addition, $p_0$ belongs to the nonparametric family of the $\beta$-Sobolev space with $\beta\le 2$, by adopting an early stopping strategy, we obtain that the diffusion
    
[^4]: 通过联合子空间估计消除神经网络表示中的错误概念

    Removing Spurious Concepts from Neural Network Representations via Joint Subspace Estimation. (arXiv:2310.11991v1 [cs.LG])

    [http://arxiv.org/abs/2310.11991](http://arxiv.org/abs/2310.11991)

    该论文提出了一种通过联合子空间估计从神经网络表示中消除错误概念的迭代算法，并在计算机视觉和自然语言处理的基准数据集上展示了其优越性能。

    

    神经网络中的错误相关性经常会影响到模型在样本外的泛化能力。常见的策略是通过从神经网络表示中消除错误概念来缓解这个问题。现有的错误概念消除方法往往过于激进，不经意间会消除与模型主要任务相关的特征，从而影响模型性能。我们提出了一种迭代算法，通过共同识别神经网络表示中的两个低维正交子空间来分离错误和主要任务的概念。我们在计算机视觉（Waterbirds，CelebA）和自然语言处理（MultiNLI）的基准数据集上评估了该算法，并表明它优于现有的概念消除方法。

    Out-of-distribution generalization in neural networks is often hampered by spurious correlations. A common strategy is to mitigate this by removing spurious concepts from the neural network representation of the data. Existing concept-removal methods tend to be overzealous by inadvertently eliminating features associated with the main task of the model, thereby harming model performance. We propose an iterative algorithm that separates spurious from main-task concepts by jointly identifying two low-dimensional orthogonal subspaces in the neural network representation. We evaluate the algorithm on benchmark datasets for computer vision (Waterbirds, CelebA) and natural language processing (MultiNLI), and show that it outperforms existing concept removal methods
    
[^5]: 批量预测器在分布内具有广义性。

    Batched Predictors Generalize within Distribution. (arXiv:2307.09379v1 [stat.ML])

    [http://arxiv.org/abs/2307.09379](http://arxiv.org/abs/2307.09379)

    批量预测器提供了指数级更强的泛化保证，可应用于离线测试前化合物质量的预测任务。

    

    我们研究了批量预测器的广义性质，即任务是预测一小组（或批量）示例的均值标签的模型。批量预测范式对于部署在离线测试前确定一组化合物的质量的模型尤为相关。通过利用适当的Rademacher复杂性的广义化，我们证明批量预测器具有指数级更强的泛化保证，与标准的逐个样本方法相比。令人惊讶的是，该提议的上界独立于过参数化。我们的理论洞察力在各种任务、架构和应用中通过实验证实。

    We study the generalization properties of batched predictors, i.e., models tasked with predicting the mean label of a small set (or batch) of examples. The batched prediction paradigm is particularly relevant for models deployed to determine the quality of a group of compounds in preparation for offline testing. By utilizing a suitable generalization of the Rademacher complexity, we prove that batched predictors come with exponentially stronger generalization guarantees as compared to the standard per-sample approach. Surprisingly, the proposed bound holds independently of overparametrization. Our theoretical insights are validated experimentally for various tasks, architectures, and applications.
    
[^6]: 匹配追踪的快速收敛速度

    Sharp Convergence Rates for Matching Pursuit. (arXiv:2307.07679v1 [stat.ML])

    [http://arxiv.org/abs/2307.07679](http://arxiv.org/abs/2307.07679)

    本文通过提升现有的下界来匹配最佳上界，对匹配追踪的性能进行了精确描述，并构造了一个最坏情况的字典来证明现有上界的无法改进。

    

    本文研究了匹配追踪的基本限制，即通过字典中的元素的稀疏线性组合来近似目标函数的纯贪婪算法。当目标函数包含在对应于字典的变化空间中时，许多令人印象深刻的研究在过去几十年中获得了匹配追踪的收敛速度的上界和下界，但它们并不匹配。本文的主要贡献是填补这一差距，并获得匹配追踪性能的精确描述。我们通过改进现有的下界以匹配最佳上界来实现这一目标。具体来说，我们构造了一个最坏情况的字典，证明了现有的上界不能改进。事实证明，与其他贪婪算法变体不同，收敛速度是次优的，并且由解某个非线性方程的解决方案决定。这使我们得出结论，任意程度的收缩都会改善匹配追踪效果。

    We study the fundamental limits of matching pursuit, or the pure greedy algorithm, for approximating a target function by a sparse linear combination of elements from a dictionary. When the target function is contained in the variation space corresponding to the dictionary, many impressive works over the past few decades have obtained upper and lower bounds on the convergence rate of matching pursuit, but they do not match. The main contribution of this paper is to close this gap and obtain a sharp characterization of the performance of matching pursuit. We accomplish this by improving the existing lower bounds to match the best upper bound. Specifically, we construct a worst case dictionary which proves that the existing upper bound cannot be improved. It turns out that, unlike other greedy algorithm variants, the converge rate is suboptimal and is determined by the solution to a certain non-linear equation. This enables us to conclude that any amount of shrinkage improves matching pu
    
[^7]: 异质性功能数据的混合分割模型

    Mixture of segmentation for heterogeneous functional data. (arXiv:2303.10712v1 [stat.ME])

    [http://arxiv.org/abs/2303.10712](http://arxiv.org/abs/2303.10712)

    本文提出了一种混合分割模型，可以处理异质性功能数据，通过动态规划的EM算法近似最大似然估计器，方法在模拟与真实数据集上得到验证。

    

    本文针对时间和人口异质性的功能数据提出了一种混合分割模型，旨在保持功能结构的同时表示异质性。 讨论了最大似然估计器的可辨识性和一致性，并采用动态规划的EM算法来近似最大似然估计器。 该方法在模拟数据上进行了说明，并在用电量真实数据集上得到了应用。

    In this paper we consider functional data with heterogeneity in time and in population. We propose a mixture model with segmentation of time to represent this heterogeneity while keeping the functional structure. Maximum likelihood estimator is considered, proved to be identifiable and consistent. In practice, an EM algorithm is used, combined with dynamic programming for the maximization step, to approximate the maximum likelihood estimator. The method is illustrated on a simulated dataset, and used on a real dataset of electricity consumption.
    
[^8]: 初阶ANIL在存在误指定的潜在维度情况下学习线性表示

    First-order ANIL learns linear representations despite misspecified latent dimension. (arXiv:2303.01335v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.01335](http://arxiv.org/abs/2303.01335)

    本研究表明，在存在架构误指定的情况下，初阶ANIL可以成功学习到线性的共享表示。这个结果是基于对无限数量任务的极限情况下的推导。

    

    最近，由于在少样本分类和强化学习中的经验成功，元学习引起了极大的关注。元学习方法利用来自以前任务的数据以一种样本高效的方式学习新任务。特别是，模型无关的方法寻找起始点，从该起始点开始梯度下降可以迅速适应任何新任务。尽管经验上建议这样的方法通过在预训练期间学习共享表示表现良好，但对于这种行为的理论证据有限。更重要的是，并没有严格证明这些方法在存在架构误指定的情况下仍会学习到共享结构。在这个方向上，本文在无限数量的任务的极限情况下展示了，使用线性双层网络结构的初阶ANIL成功地学习到了线性的共享表示。即使是在参数化误指定的情况下，这个结果仍然成立，即网络的宽度大于

    Due to its empirical success in few-shot classification and reinforcement learning, meta-learning has recently received significant interest. Meta-learning methods leverage data from previous tasks to learn a new task in a sample-efficient manner. In particular, model-agnostic methods look for initialisation points from which gradient descent quickly adapts to any new task. Although it has been empirically suggested that such methods perform well by learning shared representations during pretraining, there is limited theoretical evidence of such behavior. More importantly, it has not been rigorously shown that these methods still learn a shared structure, despite architectural misspecifications. In this direction, this work shows, in the limit of an infinite number of tasks, that first-order ANIL with a linear two-layer network architecture successfully learns linear shared representations. This result even holds with a misspecified network parameterisation; having a width larger than 
    

