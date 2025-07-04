# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Tree-based Learning for High-Fidelity Prediction of Chaos](https://arxiv.org/abs/2403.13836) | TreeDOX是一种基于树的方法，不需要超参数调整，使用时间延迟过度嵌入和额外树回归器进行特征降维和预测，并在深度预测混沌系统中表现出state-of-the-art的性能。 |
| [^2] | [Temporally-Consistent Koopman Autoencoders for Forecasting Dynamical Systems](https://arxiv.org/abs/2403.12335) | 引入了时间一致Koopman自编码器（tcKAE）来生成准确的长期预测，在有限嘈杂的训练数据下通过一致性正则化项增强了模型的稳健性和泛化能力。 |
| [^3] | [Towards Explaining Deep Neural Network Compression Through a Probabilistic Latent Space](https://arxiv.org/abs/2403.00155) | 通过概率潜在空间提出了一个新的理论框架，解释了深度神经网络压缩的优化网络稀疏度，并探讨了网络层的AP3/AP2属性与性能之间的关系。 |
| [^4] | [Avoiding Catastrophe in Continuous Spaces by Asking for Help](https://arxiv.org/abs/2402.08062) | 在连续空间中，通过寻求帮助来避免灾难。引入了一种上下文多臂赌博问题的变体，目标是最小化灾难发生的概率。提出了一种算法，在连续1D状态空间和相对简单的回报函数下，遗憾和向导师查询率都趋近于0。 |
| [^5] | [Beyond Scale: the Diversity Coefficient as a Data Quality Metric Demonstrates LLMs are Pre-trained on Formally Diverse Data.](http://arxiv.org/abs/2306.13840) | 本论文提出使用多样性系数作为LLM预训练数据质量的指标，研究表明公开可用的LLM数据集的多样性系数很高。 |
| [^6] | [Implicit Counterfactual Data Augmentation for Deep Neural Networks.](http://arxiv.org/abs/2304.13431) | 本研究提出了隐式反事实数据增强（ICDA）方法，通过新的样本增强策略、易于计算的代理损失和具体方案，消除了虚假关联并进行了稳健预测。 |
| [^7] | [Kernel Density Bayesian Inverse Reinforcement Learning.](http://arxiv.org/abs/2303.06827) | KD-BIRL是一种核密度贝叶斯逆强化学习方法，通过直接逼近似然函数来学习代理的奖励函数，克服了学习点估计的缺点，并适用于复杂和无限环境。 |
| [^8] | [The unstable formula theorem revisited via algorithms.](http://arxiv.org/abs/2212.05050) | 本文介绍了模型理论中有关理论稳定性和学习中算法稳定性的交互，通过算法性质取代了无限，重访了Shelah闻名的不稳定公式定理，并引入了可能最终正确学习模型，表征了Littlestone（稳定）类，透过模型论中类型的可定义性形式化了Littlestone类的逼近。 |
| [^9] | [Benign Overfitting without Linearity: Neural Network Classifiers Trained by Gradient Descent for Noisy Linear Data.](http://arxiv.org/abs/2202.05928) | 本文研究了使用梯度下降训练的神经网络在泛化时能够很好应对噪声数据的良性过拟合现象。研究表明，在特定条件下，神经网络能够将训练误差降至零并完美地适应带有噪声标签的数据，并同时达到最优的测试误差。 |

# 详细

[^1]: 基于树的学习用于深度预测混沌现象

    Tree-based Learning for High-Fidelity Prediction of Chaos

    [https://arxiv.org/abs/2403.13836](https://arxiv.org/abs/2403.13836)

    TreeDOX是一种基于树的方法，不需要超参数调整，使用时间延迟过度嵌入和额外树回归器进行特征降维和预测，并在深度预测混沌系统中表现出state-of-the-art的性能。

    

    深度预测混沌系统的时间演变是至关重要但具有挑战性的。现有解决方案需要进行超参数调整，这严重阻碍了它们的广泛应用。在这项工作中，我们引入了一种无需超参数调整的基于树的方法：TreeDOX。它使用时间延迟过度嵌入作为显式短期记忆，以及额外树回归器来执行特征降维和预测。我们使用Henon映射，Lorenz和Kuramoto-Sivashinsky系统以及现实世界的Southern Oscillation Index展示了TreeDOX的最先进性能。

    arXiv:2403.13836v1 Announce Type: new  Abstract: Model-free forecasting of the temporal evolution of chaotic systems is crucial but challenging. Existing solutions require hyperparameter tuning, significantly hindering their wider adoption. In this work, we introduce a tree-based approach not requiring hyperparameter tuning: TreeDOX. It uses time delay overembedding as explicit short-term memory and Extra-Trees Regressors to perform feature reduction and forecasting. We demonstrate the state-of-the-art performance of TreeDOX using the Henon map, Lorenz and Kuramoto-Sivashinsky systems, and the real-world Southern Oscillation Index.
    
[^2]: 一种用于预测动态系统的时间一致的Koopman自编码器

    Temporally-Consistent Koopman Autoencoders for Forecasting Dynamical Systems

    [https://arxiv.org/abs/2403.12335](https://arxiv.org/abs/2403.12335)

    引入了时间一致Koopman自编码器（tcKAE）来生成准确的长期预测，在有限嘈杂的训练数据下通过一致性正则化项增强了模型的稳健性和泛化能力。

    

    缺乏足够高质量的数据经常是高维时空动态系统数据驱动建模中关键挑战。Koopman自编码器（KAEs）利用深度神经网络（DNNs）的表达能力、自编码器的降维能力以及Koopman算子的谱特性，学习具有更简单线性动态的降阶特征空间。然而，KAEs的有效性受限于有限而嘈杂的训练数据集，导致泛化能力较差。为解决这一问题，我们引入了一种称为时间一致Koopman自编码器（tcKAE）的模型，旨在即使在受限且嘈杂的训练数据情况下生成准确的长期预测。这是通过强制在不同时间步上保持预测一致性的一致性正则化项实现的，从而增强了tcKAE相对于现有模型的稳健性和泛化能力。

    arXiv:2403.12335v1 Announce Type: new  Abstract: Absence of sufficiently high-quality data often poses a key challenge in data-driven modeling of high-dimensional spatio-temporal dynamical systems. Koopman Autoencoders (KAEs) harness the expressivity of deep neural networks (DNNs), the dimension reduction capabilities of autoencoders, and the spectral properties of the Koopman operator to learn a reduced-order feature space with simpler, linear dynamics. However, the effectiveness of KAEs is hindered by limited and noisy training datasets, leading to poor generalizability. To address this, we introduce the Temporally-Consistent Koopman Autoencoder (tcKAE), designed to generate accurate long-term predictions even with constrained and noisy training data. This is achieved through a consistency regularization term that enforces prediction coherence across different time steps, thus enhancing the robustness and generalizability of tcKAE over existing models. We provide analytical justifica
    
[^3]: 通过概率潜在空间解释深度神经网络压缩

    Towards Explaining Deep Neural Network Compression Through a Probabilistic Latent Space

    [https://arxiv.org/abs/2403.00155](https://arxiv.org/abs/2403.00155)

    通过概率潜在空间提出了一个新的理论框架，解释了深度神经网络压缩的优化网络稀疏度，并探讨了网络层的AP3/AP2属性与性能之间的关系。

    

    尽管深度神经网络（DNNs）表现出色，但它们的计算复杂性和存储空间消耗导致了网络压缩的概念。尽管已广泛研究了诸如修剪和低秩分解等DNN压缩技术，但对它们的理论解释仍未受到足够关注。本文提出了一个利用DNN权重的概率潜在空间并利用信息理论分歧度量解释最佳网络稀疏性的新理论框架。我们为DNN引入了新的类比投影模式（AP2）和概率中的类比投影模式（AP3）概念，并证明网络中层的AP3/AP2特性与其性能之间存在关系。此外，我们提供了一个理论分析，解释了压缩网络的训练过程。这些理论结果是从实证实验

    arXiv:2403.00155v1 Announce Type: new  Abstract: Despite the impressive performance of deep neural networks (DNNs), their computational complexity and storage space consumption have led to the concept of network compression. While DNN compression techniques such as pruning and low-rank decomposition have been extensively studied, there has been insufficient attention paid to their theoretical explanation. In this paper, we propose a novel theoretical framework that leverages a probabilistic latent space of DNN weights and explains the optimal network sparsity by using the information-theoretic divergence measures. We introduce new analogous projected patterns (AP2) and analogous-in-probability projected patterns (AP3) notions for DNNs and prove that there exists a relationship between AP3/AP2 property of layers in the network and its performance. Further, we provide a theoretical analysis that explains the training process of the compressed network. The theoretical results are empirica
    
[^4]: 避免连续空间中的灾难：通过寻求帮助

    Avoiding Catastrophe in Continuous Spaces by Asking for Help

    [https://arxiv.org/abs/2402.08062](https://arxiv.org/abs/2402.08062)

    在连续空间中，通过寻求帮助来避免灾难。引入了一种上下文多臂赌博问题的变体，目标是最小化灾难发生的概率。提出了一种算法，在连续1D状态空间和相对简单的回报函数下，遗憾和向导师查询率都趋近于0。

    

    大多数具有正式遗憾保证的强化学习算法假设所有错误都是可逆的，并依赖于尝试所有可能的选项。当一些错误是无法修复甚至是灾难性的时，这种方法会导致糟糕的结果。我们提出了一种上下文多臂赌博问题的变体，在这个问题中，目标是最小化发生灾难的概率。具体而言，我们假设每轮的回报代表了在该轮避免灾难的概率，并尝试最大化回报的乘积（总体避免灾难的概率）。为了给 agent 一些成功的机会，我们允许有限次向导师提问，并假设回报函数为 Lipschitz 连续的。我们提出了一种算法，当时间跨度增长时，它的遗憾和向导师查询率都趋近于 0，假设是一个连续的 1D 状态空间和相对"简单"的回报函数。我们还提供了一个匹配的下界：在没有简单性假设的情况下，任何算法要么不断查询异常的行为，要么每次查询完全相同的行为。

    Most reinforcement learning algorithms with formal regret guarantees assume all mistakes are reversible and rely on essentially trying all possible options. This approach leads to poor outcomes when some mistakes are irreparable or even catastrophic. We propose a variant of the contextual bandit problem where the goal is to minimize the chance of catastrophe. Specifically, we assume that the payoff each round represents the chance of avoiding catastrophe that round, and try to maximize the product of payoffs (the overall chance of avoiding catastrophe). To give the agent some chance of success, we allow a limited number of queries to a mentor and assume a Lipschitz continuous payoff function. We present an algorithm whose regret and rate of querying the mentor both approach 0 as the time horizon grows, assuming a continuous 1D state space and a relatively "simple" payoff function. We also provide a matching lower bound: without the simplicity assumption: any algorithm either constantly
    
[^5]: 超越规模：多样性系数作为数据质量指标证明了LLMs是在形式多样的数据上预先训练的

    Beyond Scale: the Diversity Coefficient as a Data Quality Metric Demonstrates LLMs are Pre-trained on Formally Diverse Data. (arXiv:2306.13840v1 [cs.CL])

    [http://arxiv.org/abs/2306.13840](http://arxiv.org/abs/2306.13840)

    本论文提出使用多样性系数作为LLM预训练数据质量的指标，研究表明公开可用的LLM数据集的多样性系数很高。

    

    当前，预先训练强大的大语言模型(LLMs)的趋势主要集中在模型和数据集规模的扩大。然而，预先训练数据的质量对于训练强大的LLMs来说是一个重要因素，但它是一个模糊的概念，尚未完全表征。因此，我们使用最近提出的Task2Vec多样性系数来基于数据质量的形式方面，超越规模本身。具体而言，我们测量公开可用的预先训练数据集的多样性系数，以证明它们的形式多样性高于理论的下限和上限。此外，为了建立对多样性系数的信心，我们进行可解释性实验，并发现该系数与多样性的直观属性相吻合，例如，随着潜在概念数量的增加，它增加。我们得出结论，多样性系数是可靠的，表明公开可用的LLM数据集的多样性系数很高，并推测它可以作为预训练LLMs模型的数据质量指标。

    Current trends to pre-train capable Large Language Models (LLMs) mostly focus on scaling of model and dataset size. However, the quality of pre-training data is an important factor for training powerful LLMs, yet it is a nebulous concept that has not been fully characterized. Therefore, we use the recently proposed Task2Vec diversity coefficient to ground and understand formal aspects of data quality, to go beyond scale alone. Specifically, we measure the diversity coefficient of publicly available pre-training datasets to demonstrate that their formal diversity is high when compared to theoretical lower and upper bounds. In addition, to build confidence in the diversity coefficient, we conduct interpretability experiments and find that the coefficient aligns with intuitive properties of diversity, e.g., it increases as the number of latent concepts increases. We conclude the diversity coefficient is reliable, show it's high for publicly available LLM datasets, and conjecture it can be
    
[^6]: 深度神经网络的隐式反事实数据增强

    Implicit Counterfactual Data Augmentation for Deep Neural Networks. (arXiv:2304.13431v1 [cs.LG])

    [http://arxiv.org/abs/2304.13431](http://arxiv.org/abs/2304.13431)

    本研究提出了隐式反事实数据增强（ICDA）方法，通过新的样本增强策略、易于计算的代理损失和具体方案，消除了虚假关联并进行了稳健预测。

    

    机器学习模型易于捕捉非因果属性和类别之间的虚假相关性，使用反事实数据增强是破除这些虚假的联想的有效方法。然而，明确生成反事实数据很具挑战性，训练效率会降低。因此，本研究提出了一种隐式反事实数据增强（Implicit Counterfactual Data Augmentation，ICDA）方法来消除虚假关联并进行稳健预测。具体而言，首先，开发了一种新的样本增强策略，为每个样本生成在语义和反事实意义上有意义的深度特征，并具有不同的增强强度。其次，当增广样本数变为无穷大时，我们推导出对于增广特征集的易于计算的代理损失。第三，提出了两种具体的方案，包括直接量化和元学习，以确定鲁棒性损失的关键参数。此外，还从实验的角度解释了ICDA的作用。

    Machine-learning models are prone to capturing the spurious correlations between non-causal attributes and classes, with counterfactual data augmentation being a promising direction for breaking these spurious associations. However, explicitly generating counterfactual data is challenging, with the training efficiency declining. Therefore, this study proposes an implicit counterfactual data augmentation (ICDA) method to remove spurious correlations and make stable predictions. Specifically, first, a novel sample-wise augmentation strategy is developed that generates semantically and counterfactually meaningful deep features with distinct augmentation strength for each sample. Second, we derive an easy-to-compute surrogate loss on the augmented feature set when the number of augmented samples becomes infinite. Third, two concrete schemes are proposed, including direct quantification and meta-learning, to derive the key parameters for the robust loss. In addition, ICDA is explained from 
    
[^7]: 核密度贝叶斯逆强化学习

    Kernel Density Bayesian Inverse Reinforcement Learning. (arXiv:2303.06827v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.06827](http://arxiv.org/abs/2303.06827)

    KD-BIRL是一种核密度贝叶斯逆强化学习方法，通过直接逼近似然函数来学习代理的奖励函数，克服了学习点估计的缺点，并适用于复杂和无限环境。

    

    逆强化学习（IRL）是一种通过观察代理行为来推断其奖励函数的强大框架，但学习奖励函数的点估计可能会误导，因为可能有多个函数能够很好地描述代理的行为。贝叶斯逆强化学习采用贝叶斯方法模拟候选奖励函数的分布，克服了学习点估计的缺点。然而，一些贝叶斯逆强化学习算法使用Q值函数代替似然函数。由此得到的后验计算量大，理论保证少，并且Q值函数通常对似然函数的逼近效果较差。我们引入了核密度贝叶斯逆强化学习（KD-BIRL），该方法使用条件核密度估计直接逼近似然函数，提供了一个高效的框架，在经过改进的奖励函数参数化下，适用于具有复杂和无限的环境。

    Inverse reinforcement learning~(IRL) is a powerful framework to infer an agent's reward function by observing its behavior, but IRL algorithms that learn point estimates of the reward function can be misleading because there may be several functions that describe an agent's behavior equally well. A Bayesian approach to IRL models a distribution over candidate reward functions, alleviating the shortcomings of learning a point estimate. However, several Bayesian IRL algorithms use a $Q$-value function in place of the likelihood function. The resulting posterior is computationally intensive to calculate, has few theoretical guarantees, and the $Q$-value function is often a poor approximation for the likelihood. We introduce kernel density Bayesian IRL (KD-BIRL), which uses conditional kernel density estimation to directly approximate the likelihood, providing an efficient framework that, with a modified reward function parameterization, is applicable to environments with complex and infin
    
[^8]: 通过算法重访不稳定公式定理

    The unstable formula theorem revisited via algorithms. (arXiv:2212.05050v2 [math.LO] UPDATED)

    [http://arxiv.org/abs/2212.05050](http://arxiv.org/abs/2212.05050)

    本文介绍了模型理论中有关理论稳定性和学习中算法稳定性的交互，通过算法性质取代了无限，重访了Shelah闻名的不稳定公式定理，并引入了可能最终正确学习模型，表征了Littlestone（稳定）类，透过模型论中类型的可定义性形式化了Littlestone类的逼近。

    

    本文介绍了有关模型理论中关于理论稳定性的基础结果与学习中算法稳定性之间惊人的交互，特别是通过算法性质取代无限，我们开发了Shelah闻名的不稳定公式定理的完整算法类比。这其中涉及了几个新定理以及最近的研究。特别地，我们引入了一个新的“可能最终正确”的学习模型，并通过这个模型表征了Littlestone（稳定）类；并通过模型论中类型的可定义性类比描述了Littlestone类的逼近。

    This paper is about the surprising interaction of a foundational result from model theory about stability of theories, which seems to be inherently about the infinite, with algorithmic stability in learning. Specifically, we develop a complete algorithmic analogue of Shelah's celebrated Unstable Formula Theorem, with algorithmic properties taking the place of the infinite. This draws on several new theorems as well as much recent work. In particular we introduce a new ``Probably Eventually Correct'' learning model, of independent interest, and characterize Littlestone (stable) classes in terms of this model; and we describe Littlestone classes via approximations, by analogy to definability of types in model theory.
    
[^9]: 不需要线性关系的良性过拟合：通过梯度下降训练的神经网络分类器用于噪声线性数据

    Benign Overfitting without Linearity: Neural Network Classifiers Trained by Gradient Descent for Noisy Linear Data. (arXiv:2202.05928v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2202.05928](http://arxiv.org/abs/2202.05928)

    本文研究了使用梯度下降训练的神经网络在泛化时能够很好应对噪声数据的良性过拟合现象。研究表明，在特定条件下，神经网络能够将训练误差降至零并完美地适应带有噪声标签的数据，并同时达到最优的测试误差。

    

    良性过拟合是指插值模型在存在噪声数据的情况下能够很好地泛化的现象，最早出现在使用梯度下降训练的神经网络模型中。为了更好地理解这一实证观察，我们考虑了两层神经网络在随机初始化后通过梯度下降在逻辑损失函数上进行插值训练的泛化误差。我们假设数据来自于明显分离的类条件对数凹分布，并允许训练标签中的一定比例被对手篡改。我们证明在这种情况下，神经网络表现出良性过拟合的特点：它们可以被驱动到零训练误差，完美地拟合任何有噪声的训练标签，并同时达到极小化最大化最优测试误差。与之前关于良性过拟合需要线性或基于核的预测器的工作相比，我们的分析在模型和学习动态都是基本非线性的情况下成立。

    Benign overfitting, the phenomenon where interpolating models generalize well in the presence of noisy data, was first observed in neural network models trained with gradient descent. To better understand this empirical observation, we consider the generalization error of two-layer neural networks trained to interpolation by gradient descent on the logistic loss following random initialization. We assume the data comes from well-separated class-conditional log-concave distributions and allow for a constant fraction of the training labels to be corrupted by an adversary. We show that in this setting, neural networks exhibit benign overfitting: they can be driven to zero training error, perfectly fitting any noisy training labels, and simultaneously achieve minimax optimal test error. In contrast to previous work on benign overfitting that require linear or kernel-based predictors, our analysis holds in a setting where both the model and learning dynamics are fundamentally nonlinear.
    

