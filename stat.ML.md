# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RNNs are not Transformers (Yet): The Key Bottleneck on In-context Retrieval](https://arxiv.org/abs/2402.18510) | 本文研究了RNNs和Transformer在处理算法问题时的表现能力差距，发现RNNs存在关键瓶颈，即无法完美地从上下文中检索信息，导致无法像Transformer那样轻松解决需要这种能力的任务。 |
| [^2] | [Information-Theoretic Safe Bayesian Optimization](https://arxiv.org/abs/2402.15347) | 提出了一种信息论安全探索准则，结合贝叶斯优化收益函数，形成了一种新颖的安全贝叶斯优化选择准则。 |
| [^3] | [Logistic-beta processes for modeling dependent random probabilities with beta marginals](https://arxiv.org/abs/2402.07048) | 本文提出了一种新颖的logistic-beta过程用于建模具有beta边际分布的相关随机概率。该过程具有灵活的相关结构和计算优势，并通过非参数二分类回归模拟研究进行了验证。 |
| [^4] | [Riemann-Lebesgue Forest for Regression](https://arxiv.org/abs/2402.04550) | 提出了一种新颖的集成方法Riemann-Lebesgue Forest (RLF)用于回归问题，通过划分函数的值域为多个区间来逼近可测函数的思想，开发了一种新的树学习算法Riemann-Lebesgue Tree。通过Hoeffding分解和Stein方法推导了RLF在不同参数设置下的渐近性能，并在仿真数据和真实世界数据集上的实验中证明了RLF与原始随机森林相比具有竞争力的性能。 |
| [^5] | [Neurosymbolic Grounding for Compositional World Models.](http://arxiv.org/abs/2310.12690) | 本论文介绍了一种名为Cosmos的框架，用于对象为中心的世界建模，通过使用神经符号化基础和视觉-语言基础模型，实现了在未见过的输入场景上的高性能组合泛化能力。 |
| [^6] | [Flexible and efficient spatial extremes emulation via variational autoencoders.](http://arxiv.org/abs/2307.08079) | 本文提出了一种新的空间极端值模型，通过集成在变分自动编码器的结构中，可以灵活、高效地模拟具有非平稳相关性的极端事件。实验证明，在时间效率和性能上，相对于传统的贝叶斯推断和许多具有平稳相关性的空间极端值模型，我们的方法具有优势。 |
| [^7] | [Adjusted Wasserstein Distributionally Robust Estimator in Statistical Learning.](http://arxiv.org/abs/2303.15579) | 本文提出了一种统计学习中的调整Wasserstein分布鲁棒估计方法，能够提高估计的统计性能，保持样本外性能保证，特别适用于广义线性模型。 |

# 详细

[^1]: RNNs还不是Transformer：在上下文检索中的关键瓶颈

    RNNs are not Transformers (Yet): The Key Bottleneck on In-context Retrieval

    [https://arxiv.org/abs/2402.18510](https://arxiv.org/abs/2402.18510)

    本文研究了RNNs和Transformer在处理算法问题时的表现能力差距，发现RNNs存在关键瓶颈，即无法完美地从上下文中检索信息，导致无法像Transformer那样轻松解决需要这种能力的任务。

    

    本文探讨循环神经网络（RNNs）和Transformer在解决算法问题时的表示能力差距。我们重点关注RNNs是否能在处理长序列时，通过Chain-of-Thought (CoT)提示，与Transformer的性能相匹配。我们的理论分析显示CoT可以改进RNNs，但无法弥补与Transformer之间的差距。关键瓶颈在于RNNs无法完全从上下文中检索信息，即使经过CoT的增强：对于几个明确或隐式需要这种能力的任务，如联想召回和确定图是否为树，我们证明RNNs表达能力不足以解决这些任务，而Transformer可以轻松解决。相反，我们证明采用增强RNNs上下文检索能力的技术，包括

    arXiv:2402.18510v1 Announce Type: cross  Abstract: This paper investigates the gap in representation powers of Recurrent Neural Networks (RNNs) and Transformers in the context of solving algorithmic problems. We focus on understanding whether RNNs, known for their memory efficiency in handling long sequences, can match the performance of Transformers, particularly when enhanced with Chain-of-Thought (CoT) prompting. Our theoretical analysis reveals that CoT improves RNNs but is insufficient to close the gap with Transformers. A key bottleneck lies in the inability of RNNs to perfectly retrieve information from the context, even with CoT: for several tasks that explicitly or implicitly require this capability, such as associative recall and determining if a graph is a tree, we prove that RNNs are not expressive enough to solve the tasks while Transformers can solve them with ease. Conversely, we prove that adopting techniques to enhance the in-context retrieval capability of RNNs, inclu
    
[^2]: 信息论安全贝叶斯优化

    Information-Theoretic Safe Bayesian Optimization

    [https://arxiv.org/abs/2402.15347](https://arxiv.org/abs/2402.15347)

    提出了一种信息论安全探索准则，结合贝叶斯优化收益函数，形成了一种新颖的安全贝叶斯优化选择准则。

    

    我们考虑了一个顺序决策任务，其目标是在不评估违反先验未知（安全）约束的参数的情况下优化未知函数。一个常见的方法是在未知函数上放置高斯过程先验，并且仅允许在高概率安全区域内进行评估。大多数当前方法依赖于对域的离散化，并且不能直接扩展到连续情况。此外，它们利用约束的规则假设的方式引入了一个额外的关键超参数。在本文中，我们提出了一个信息论安全探索准则，该准则直接利用GP后验来识别最具信息的安全参数进行评估。将这一探索准则与众所周知的贝叶斯优化收益函数结合起来，产生了一种新颖的安全贝叶斯优化选择准则。

    arXiv:2402.15347v1 Announce Type: cross  Abstract: We consider a sequential decision making task, where the goal is to optimize an unknown function without evaluating parameters that violate an a~priori unknown (safety) constraint. A common approach is to place a Gaussian process prior on the unknown functions and allow evaluations only in regions that are safe with high probability. Most current methods rely on a discretization of the domain and cannot be directly extended to the continuous case. Moreover, the way in which they exploit regularity assumptions about the constraint introduces an additional critical hyperparameter. In this paper, we propose an information-theoretic safe exploration criterion that directly exploits the GP posterior to identify the most informative safe parameters to evaluate. The combination of this exploration criterion with a well known Bayesian optimization acquisition function yields a novel safe Bayesian optimization selection criterion. Our approach 
    
[^3]: 用于建模具有beta边际分布的相关随机概率的logistic-beta过程

    Logistic-beta processes for modeling dependent random probabilities with beta marginals

    [https://arxiv.org/abs/2402.07048](https://arxiv.org/abs/2402.07048)

    本文提出了一种新颖的logistic-beta过程用于建模具有beta边际分布的相关随机概率。该过程具有灵活的相关结构和计算优势，并通过非参数二分类回归模拟研究进行了验证。

    

    beta分布被广泛应用于概率建模，并在统计学和机器学习中被广泛使用，尤其在贝叶斯非参数领域。尽管其被广泛使用，但在建模相关随机概率的灵活和计算方便的随机过程扩展方面，相关工作有限。我们提出了一种新颖的随机过程，称为logistic-beta过程，其logistic变换生成具有常见beta边际分布的随机过程。类似于高斯过程，logistic-beta过程可以建模离散和连续域（例如空间或时间）上的相关性，并通过相关核函数具有高度灵活的相关结构。此外，它的正态方差-均值混合表示导致了高效的后验推理算法。通过非参数二分类回归模拟研究，展示了logistic-beta过程的灵活性和计算优势。

    The beta distribution serves as a canonical tool for modeling probabilities and is extensively used in statistics and machine learning, especially in the field of Bayesian nonparametrics. Despite its widespread use, there is limited work on flexible and computationally convenient stochastic process extensions for modeling dependent random probabilities. We propose a novel stochastic process called the logistic-beta process, whose logistic transformation yields a stochastic process with common beta marginals. Similar to the Gaussian process, the logistic-beta process can model dependence on both discrete and continuous domains, such as space or time, and has a highly flexible dependence structure through correlation kernels. Moreover, its normal variance-mean mixture representation leads to highly effective posterior inference algorithms. The flexibility and computational benefits of logistic-beta processes are demonstrated through nonparametric binary regression simulation studies. Fur
    
[^4]: Riemann-Lebesgue Forest回归方法的研究

    Riemann-Lebesgue Forest for Regression

    [https://arxiv.org/abs/2402.04550](https://arxiv.org/abs/2402.04550)

    提出了一种新颖的集成方法Riemann-Lebesgue Forest (RLF)用于回归问题，通过划分函数的值域为多个区间来逼近可测函数的思想，开发了一种新的树学习算法Riemann-Lebesgue Tree。通过Hoeffding分解和Stein方法推导了RLF在不同参数设置下的渐近性能，并在仿真数据和真实世界数据集上的实验中证明了RLF与原始随机森林相比具有竞争力的性能。

    

    我们提出了一种新颖的用于回归问题的集成方法，称为Riemann-Lebesgue Forest (RLF)。RLF的核心思想是通过将函数的值域划分为几个区间来模拟可测函数的逼近方式。基于这个思想，我们开发了一种新的树学习算法，称为Riemann-Lebesgue Tree，它在每个非叶节点上有机会从响应Y或特征空间X中的方向进行切割。我们通过Hoeffding分解和Stein方法来推导不同参数设置下RLF的渐近性能。当底层函数Y=f(X)遵循加法回归模型时，RLF与Scornet等人的论证（2014年）保持一致。通过在仿真数据和真实世界数据集上的实验证明，RLF与原始随机森林相比具有竞争力的性能。

    We propose a novel ensemble method called Riemann-Lebesgue Forest (RLF) for regression. The core idea of RLF is to mimic the way how a measurable function can be approximated by partitioning its range into a few intervals. With this idea in mind, we develop a new tree learner named Riemann-Lebesgue Tree which has a chance to split the node from response $Y$ or a direction in feature space $\mathbf{X}$ at each non-terminal node. We generalize the asymptotic performance of RLF under different parameter settings mainly through Hoeffding decomposition \cite{Vaart} and Stein's method \cite{Chen2010NormalAB}. When the underlying function $Y=f(\mathbf{X})$ follows an additive regression model, RLF is consistent with the argument from \cite{Scornet2014ConsistencyOR}. The competitive performance of RLF against original random forest \cite{Breiman2001RandomF} is demonstrated by experiments in simulation data and real world datasets.
    
[^5]: 神经符号化基础上的组合式世界建模

    Neurosymbolic Grounding for Compositional World Models. (arXiv:2310.12690v1 [cs.LG])

    [http://arxiv.org/abs/2310.12690](http://arxiv.org/abs/2310.12690)

    本论文介绍了一种名为Cosmos的框架，用于对象为中心的世界建模，通过使用神经符号化基础和视觉-语言基础模型，实现了在未见过的输入场景上的高性能组合泛化能力。

    

    我们引入了Cosmos，一个针对组合泛化（CG）设计的以对象为中心的世界建模框架，即在通过已知的视觉“原子”组合获得的未见过的输入场景上具有高性能。Cosmos的核心洞察力是使用一种新颖的神经符号化基础。具体来说，该框架引入了两个新工具：（i）神经符号化场景编码，使用神经编码器计算每个场景中的实体的实向量表示，并使用描述实体属性的可组合符号向量，以及（ii）神经符号化注意机制，将这些实体与学习到的交互规则绑定起来。Cosmos是端到端可微分的；此外，与传统的神经符号化方法需要手动将表示映射为符号不同，它使用视觉-语言基础模型计算实体的符号属性。通过对已建立的blocks场景进行两种不同形式的CG评估，我们验证了Cosmos的有效性。

    We introduce Cosmos, a framework for object-centric world modeling that is designed for compositional generalization (CG), i.e., high performance on unseen input scenes obtained through the composition of known visual "atoms." The central insight behind Cosmos is the use of a novel form of neurosymbolic grounding. Specifically, the framework introduces two new tools: (i) neurosymbolic scene encodings, which represent each entity in a scene using a real vector computed using a neural encoder, as well as a vector of composable symbols describing attributes of the entity, and (ii) a neurosymbolic attention mechanism that binds these entities to learned rules of interaction. Cosmos is end-to-end differentiable; also, unlike traditional neurosymbolic methods that require representations to be manually mapped to symbols, it computes an entity's symbolic attributes using vision-language foundation models. Through an evaluation that considers two different forms of CG on an established blocks-
    
[^6]: 通过变分自动 编码器实现灵活高效的空间极端值模拟

    Flexible and efficient spatial extremes emulation via variational autoencoders. (arXiv:2307.08079v1 [stat.ML])

    [http://arxiv.org/abs/2307.08079](http://arxiv.org/abs/2307.08079)

    本文提出了一种新的空间极端值模型，通过集成在变分自动编码器的结构中，可以灵活、高效地模拟具有非平稳相关性的极端事件。实验证明，在时间效率和性能上，相对于传统的贝叶斯推断和许多具有平稳相关性的空间极端值模型，我们的方法具有优势。

    

    许多现实世界的过程具有复杂的尾依赖结构，这种结构无法使用传统的高斯过程来描述。更灵活的空间极端值模型， 如高斯尺度混合模型和单站点调节模型，具有吸引人的极端依赖性质，但往往难以拟合和模拟。本文中，我们提出了一种新的空间极端值模型，具有灵活和非平稳的相关性属性，并将其集成到变分自动编码器 (extVAE) 的编码-解码结构中。 extVAE 可以作为一个时空模拟器，对潜在的机制模型输出状态的分布进行建模，并产生具有与输入相同属性的输出，尤其是在尾部区域。通过广泛的模拟研究，我们证明我们的extVAE比传统的贝叶斯推断更高效，并且在具有 平稳相关性结构的许多空间极端值模型中表现 更好。

    Many real-world processes have complex tail dependence structures that cannot be characterized using classical Gaussian processes. More flexible spatial extremes models such as Gaussian scale mixtures and single-station conditioning models exhibit appealing extremal dependence properties but are often exceedingly prohibitive to fit and simulate from. In this paper, we develop a new spatial extremes model that has flexible and non-stationary dependence properties, and we integrate it in the encoding-decoding structure of a variational autoencoder (extVAE). The extVAE can be used as a spatio-temporal emulator that characterizes the distribution of potential mechanistic model output states and produces outputs that have the same properties as the inputs, especially in the tail. Through extensive simulation studies, we show that our extVAE is vastly more time-efficient than traditional Bayesian inference while also outperforming many spatial extremes models with a stationary dependence str
    
[^7]: 统计学习中的调整Wasserstein分布鲁棒估计

    Adjusted Wasserstein Distributionally Robust Estimator in Statistical Learning. (arXiv:2303.15579v1 [stat.ML])

    [http://arxiv.org/abs/2303.15579](http://arxiv.org/abs/2303.15579)

    本文提出了一种统计学习中的调整Wasserstein分布鲁棒估计方法，能够提高估计的统计性能，保持样本外性能保证，特别适用于广义线性模型。

    

    我们在统计学习中提出了一种调整的Wasserstein分布鲁棒估计——基于Wasserstein分布鲁棒估计（WDRO）的非线性转换。这种转换将提高WDRO的统计性能，因为调整后的WDRO估计器渐进无偏并且均方误差趋近于零。调整后的WDRO不会削弱WDRO的样本外性能保证。我们提出了调整WDRO估计器的存在的充分条件，并给出了计算调整WDRO估计器的过程。具体而言，我们将展示如何在广义线性模型中开发调整WDRO估计器。数值实验表明，调整后的估计器比经典估计器具有更好的实际性能。

    We propose an adjusted Wasserstein distributionally robust estimator -- based on a nonlinear transformation of the Wasserstein distributionally robust (WDRO) estimator in statistical learning. This transformation will improve the statistical performance of WDRO because the adjusted WDRO estimator is asymptotically unbiased and has an asymptotically smaller mean squared error. The adjusted WDRO will not mitigate the out-of-sample performance guarantee of WDRO. Sufficient conditions for the existence of the adjusted WDRO estimator are presented, and the procedure for the computation of the adjusted WDRO estimator is given. Specifically, we will show how the adjusted WDRO estimator is developed in the generalized linear model. Numerical experiments demonstrate the favorable practical performance of the adjusted estimator over the classic one.
    

