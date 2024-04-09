# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Guarantees of confidentiality via Hammersley-Chapman-Robbins bounds](https://arxiv.org/abs/2404.02866) | 通过添加噪声到最后层的激活来保护隐私，使用HCR界限可量化保护机密性的可信度 |
| [^2] | [Statistical Inference of Optimal Allocations I: Regularities and their Implications](https://arxiv.org/abs/2403.18248) | 这项研究提出了一种函数可微方法来解决统计最优分配问题，通过对排序运算符的一般属性进行详细分析，推导出值函数的Hadamard可微性，并展示了如何利用函数偏微分法直接推导出值函数过程的渐近性质。 |
| [^3] | [Mean-field Analysis on Two-layer Neural Networks from a Kernel Perspective](https://arxiv.org/abs/2403.14917) | 本文通过核方法的视角研究了两层神经网络在平均场极限下的特征学习能力，展示了它们比任何核方法更有效地学习多个再现核希尔伯特空间的并集，并且神经网络会获得与目标函数对齐的数据相关核。 |
| [^4] | [Provable Privacy with Non-Private Pre-Processing](https://arxiv.org/abs/2403.13041) | 提出了一个框架，能够评估非私密数据相关预处理算法引起的额外隐私成本，并利用平滑DP和预处理算法的有界敏感性建立整体隐私保证的上限 |
| [^5] | [Causal Bayesian Optimization via Exogenous Distribution Learning](https://arxiv.org/abs/2402.02277) | 本文引入了一种新的方法，通过学习外源变量的分布，提高了结构化因果模型的近似精度，并将因果贝叶斯优化扩展到更一般的因果方案。 |
| [^6] | [Multinomial belief networks](https://arxiv.org/abs/2311.16909) | 提出了一种深度生成模型，用于处理具有多项式计数数据的分析需求，并能够从数据中完全自动提取出生物意义的元签名。 |
| [^7] | [Faithful and Robust Local Interpretability for Textual Predictions.](http://arxiv.org/abs/2311.01605) | 提出了一种名为FRED的新颖方法，用于解释文本预测。FRED可以识别文档中的关键词，并且通过与最先进的方法进行的实证评估证明了其在提供对文本模型的深入见解方面的有效性。 |
| [^8] | [SepVAE: a contrastive VAE to separate pathological patterns from healthy ones.](http://arxiv.org/abs/2307.06206) | SepVAE是一种对比VAE方法，可以从健康数据和患者数据中分离出共同的和特定的变化因素。在多个应用中表现出比现有方法更好的性能。 |
| [^9] | [Statistically Optimal K-means Clustering via Nonnegative Low-rank Semidefinite Programming.](http://arxiv.org/abs/2305.18436) | 本文提出了一种与NMF算法一样简单且可扩展的K均值聚类算法，该算法通过解决非负低秩半定规划问题获得了强大的统计最优性保证，实验证明该算法在合成和实际数据集上表现优异。 |
| [^10] | [Silent Abandonment in Contact Centers: Estimating Customer Patience from Uncertain Data.](http://arxiv.org/abs/2304.11754) | 该研究探究了客户在联系中心无声放弃的现象，通过不确定数据估计客户等待的耐心，揭示了这种现象对代理人时间和能力的浪费。 |
| [^11] | [Online Continuous Hyperparameter Optimization for Contextual Bandits.](http://arxiv.org/abs/2302.09440) | 该论文提出了面向上下文强化学习的在线连续超参数调整框架CDT，能够动态地在搜索空间内学习最优参数配置。 |

# 详细

[^1]: 通过Hammersley-Chapman-Robbins界限保证机密性

    Guarantees of confidentiality via Hammersley-Chapman-Robbins bounds

    [https://arxiv.org/abs/2404.02866](https://arxiv.org/abs/2404.02866)

    通过添加噪声到最后层的激活来保护隐私，使用HCR界限可量化保护机密性的可信度

    

    在深度神经网络推断过程中通过向最后几层的激活添加噪声来保护隐私是可能的。这些层中的激活被称为“特征”（少见的称为“嵌入”或“特征嵌入”）。添加的噪声有助于防止从嘈杂的特征中重建输入。通过对所有可能的无偏估计量的方差进行下限估计，量化了由此添加的噪声产生的机密性。经典不等式Hammersley和Chapman以及Robbins提供便利的、可计算的界限-- HCR界限。数值实验表明，对于包含10个类别的图像分类数据集“MNIST”和“CIFAR-10”，HCR界限在小型神经网络上表现良好。HCR界限似乎单独无法保证

    arXiv:2404.02866v1 Announce Type: new  Abstract: Protecting privacy during inference with deep neural networks is possible by adding noise to the activations in the last layers prior to the final classifiers or other task-specific layers. The activations in such layers are known as "features" (or, less commonly, as "embeddings" or "feature embeddings"). The added noise helps prevent reconstruction of the inputs from the noisy features. Lower bounding the variance of every possible unbiased estimator of the inputs quantifies the confidentiality arising from such added noise. Convenient, computationally tractable bounds are available from classic inequalities of Hammersley and of Chapman and Robbins -- the HCR bounds. Numerical experiments indicate that the HCR bounds are on the precipice of being effectual for small neural nets with the data sets, "MNIST" and "CIFAR-10," which contain 10 classes each for image classification. The HCR bounds appear to be insufficient on their own to guar
    
[^2]: 统计推断中的最优分配I：规律性及其影响

    Statistical Inference of Optimal Allocations I: Regularities and their Implications

    [https://arxiv.org/abs/2403.18248](https://arxiv.org/abs/2403.18248)

    这项研究提出了一种函数可微方法来解决统计最优分配问题，通过对排序运算符的一般属性进行详细分析，推导出值函数的Hadamard可微性，并展示了如何利用函数偏微分法直接推导出值函数过程的渐近性质。

    

    在这篇论文中，我们提出了一种用于解决统计最优分配问题的函数可微方法。通过对排序运算符的一般属性进行详细分析，我们首先推导出了值函数的Hadamard可微性。在我们的框架中，Hausdorff测度的概念以及几何测度论中的面积和共面积积分公式是核心。基于我们的Hadamard可微性结果，我们展示了如何利用函数偏微分法直接推导出二元约束最优分配问题的值函数过程以及两步ROC曲线估计量的渐近性质。此外，利用对凸和局部Lipschitz泛函的深刻见解，我们得到了最优分配问题的值函数的额外一般Frechet可微性结果。这些引人入胜的发现激励了我们

    arXiv:2403.18248v1 Announce Type: new  Abstract: In this paper, we develp a functional differentiability approach for solving statistical optimal allocation problems. We first derive Hadamard differentiability of the value function through a detailed analysis of the general properties of the sorting operator. Central to our framework are the concept of Hausdorff measure and the area and coarea integration formulas from geometric measure theory. Building on our Hadamard differentiability results, we demonstrate how the functional delta method can be used to directly derive the asymptotic properties of the value function process for binary constrained optimal allocation problems, as well as the two-step ROC curve estimator. Moreover, leveraging profound insights from geometric functional analysis on convex and local Lipschitz functionals, we obtain additional generic Fr\'echet differentiability results for the value functions of optimal allocation problems. These compelling findings moti
    
[^3]: 从核方法的角度对两层神经网络进行平均场分析

    Mean-field Analysis on Two-layer Neural Networks from a Kernel Perspective

    [https://arxiv.org/abs/2403.14917](https://arxiv.org/abs/2403.14917)

    本文通过核方法的视角研究了两层神经网络在平均场极限下的特征学习能力，展示了它们比任何核方法更有效地学习多个再现核希尔伯特空间的并集，并且神经网络会获得与目标函数对齐的数据相关核。

    

    在本文中，我们通过核方法的视角研究了两层神经网络在平均场极限下的特征学习能力。为了聚焦于第一层诱导的核的动态，我们利用了两个时间尺度的极限，其中第二层比第一层移动得快得多。在这个极限下，学习问题被简化为在内在核上的最小化问题。然后，我们展示了平均场 Langevin 动力学的全局收敛性，并推导了时间和粒子离散化误差。我们还证明了两层神经网络可以比任何核方法更有效地学习多个再现核希尔伯特空间的并集，并且神经网络会获得与目标函数对齐的数据相关核。此外，我们还开发了一个收敛到全局最优的标签噪声过程，并展示自由度出现作为一种隐式正则化。

    arXiv:2403.14917v1 Announce Type: new  Abstract: In this paper, we study the feature learning ability of two-layer neural networks in the mean-field regime through the lens of kernel methods. To focus on the dynamics of the kernel induced by the first layer, we utilize a two-timescale limit, where the second layer moves much faster than the first layer. In this limit, the learning problem is reduced to the minimization problem over the intrinsic kernel. Then, we show the global convergence of the mean-field Langevin dynamics and derive time and particle discretization error. We also demonstrate that two-layer neural networks can learn a union of multiple reproducing kernel Hilbert spaces more efficiently than any kernel methods, and neural networks acquire data-dependent kernel which aligns with the target function. In addition, we develop a label noise procedure, which converges to the global optimum and show that the degrees of freedom appears as an implicit regularization.
    
[^4]: 具有非私密预处理的可证明隐私

    Provable Privacy with Non-Private Pre-Processing

    [https://arxiv.org/abs/2403.13041](https://arxiv.org/abs/2403.13041)

    提出了一个框架，能够评估非私密数据相关预处理算法引起的额外隐私成本，并利用平滑DP和预处理算法的有界敏感性建立整体隐私保证的上限

    

    当分析差分私密（DP）机器学习管道时，通常会忽略数据相关的预处理的潜在隐私成本。在这项工作中，我们提出了一个通用框架，用于评估由非私密数据相关预处理算法引起的额外隐私成本。我们的框架通过利用两个新的技术概念建立了整体隐私保证的上限：一种称为平滑DP的DP变体以及预处理算法的有界敏感性。

    arXiv:2403.13041v1 Announce Type: cross  Abstract: When analysing Differentially Private (DP) machine learning pipelines, the potential privacy cost of data-dependent pre-processing is frequently overlooked in privacy accounting. In this work, we propose a general framework to evaluate the additional privacy cost incurred by non-private data-dependent pre-processing algorithms. Our framework establishes upper bounds on the overall privacy guarantees by utilising two new technical notions: a variant of DP termed Smooth DP and the bounded sensitivity of the pre-processing algorithms. In addition to the generic framework, we provide explicit overall privacy guarantees for multiple data-dependent pre-processing algorithms, such as data imputation, quantization, deduplication and PCA, when used in combination with several DP algorithms. Notably, this framework is also simple to implement, allowing direct integration into existing DP pipelines.
    
[^5]: 因果贝叶斯优化通过外源分布学习

    Causal Bayesian Optimization via Exogenous Distribution Learning

    [https://arxiv.org/abs/2402.02277](https://arxiv.org/abs/2402.02277)

    本文引入了一种新的方法，通过学习外源变量的分布，提高了结构化因果模型的近似精度，并将因果贝叶斯优化扩展到更一般的因果方案。

    

    在结构化因果模型中，将目标变量最大化作为操作目标是一个重要的问题。现有的因果贝叶斯优化（CBO）方法要么依赖于改变因果结构以最大化奖励的硬干预，要么引入动作节点到内生变量中，以调整数据生成机制以实现目标。本文引入了一种新的方法来学习外源变量的分布，这在现有方法中通常被忽略或通过期望进行边缘化。外源分布学习提高了通常通过有限观测数据训练的代理模型中的结构化因果模型的近似精度。此外，学习到的外源分布将现有的CBO扩展到超出加性噪声模型（ANM）的一般因果方案。恢复外源变量使我们能够为噪声或未观测到的隐藏变量使用更灵活的先验。引入了一种新的CBO方法。

    Maximizing a target variable as an operational objective in a structured causal model is an important problem. Existing Causal Bayesian Optimization (CBO) methods either rely on hard interventions that alter the causal structure to maximize the reward; or introduce action nodes to endogenous variables so that the data generation mechanisms are adjusted to achieve the objective. In this paper, a novel method is introduced to learn the distribution of exogenous variables, which is typically ignored or marginalized through expectation by existing methods.   Exogenous distribution learning improves the approximation accuracy of structured causal models in a surrogate model that is usually trained with limited observational data. Moreover, the learned exogenous distribution extends existing CBO to general causal schemes beyond Additive Noise Models (ANM). The recovery of exogenous variables allows us to use a more flexible prior for noise or unobserved hidden variables. A new CBO method is 
    
[^6]: 多项式信念网络

    Multinomial belief networks

    [https://arxiv.org/abs/2311.16909](https://arxiv.org/abs/2311.16909)

    提出了一种深度生成模型，用于处理具有多项式计数数据的分析需求，并能够从数据中完全自动提取出生物意义的元签名。

    

    机器学习中的贝叶斯方法在需要量化不确定性、处理缺失观测、样本稀缺或数据稀疏时是具有吸引力的。为了满足这些分析需求，我们提出了一种用于多项式计数数据的深层生成模型，其中网络的权重和隐藏单元均服从狄利克雷分布。我们制定了一个利用一系列增广关系的吉布斯抽样过程，类似于周-丛-陈模型。我们将该模型应用于小规模手写数字和一个大型的DNA突变癌症实验数据集，并展示了该模型如何能够完全数据驱动地提取出生物意义的元签名。

    arXiv:2311.16909v2 Announce Type: replace-cross  Abstract: A Bayesian approach to machine learning is attractive when we need to quantify uncertainty, deal with missing observations, when samples are scarce, or when the data is sparse. All of these commonly apply when analysing healthcare data. To address these analytical requirements, we propose a deep generative model for multinomial count data where both the weights and hidden units of the network are Dirichlet distributed. A Gibbs sampling procedure is formulated that takes advantage of a series of augmentation relations, analogous to the Zhou--Cong--Chen model. We apply the model on small handwritten digits, and a large experimental dataset of DNA mutations in cancer, and we show how the model is able to extract biologically meaningful meta-signatures in a fully data-driven way.
    
[^7]: 对于文本预测的忠实和稳健的本地可解释性

    Faithful and Robust Local Interpretability for Textual Predictions. (arXiv:2311.01605v1 [cs.CL])

    [http://arxiv.org/abs/2311.01605](http://arxiv.org/abs/2311.01605)

    提出了一种名为FRED的新颖方法，用于解释文本预测。FRED可以识别文档中的关键词，并且通过与最先进的方法进行的实证评估证明了其在提供对文本模型的深入见解方面的有效性。

    

    可解释性对于机器学习模型在关键领域中得到信任和部署是至关重要的。然而，现有的用于解释文本模型的方法通常复杂，并且缺乏坚实的数学基础，它们的性能也不能保证。在本文中，我们提出了一种新颖的方法FRED（Faithful and Robust Explainer for textual Documents），用于解释文本预测。FRED可以识别文档中的关键词，当这些词被移除时对预测结果产生重大影响。我们通过正式的定义和对可解释分类器的理论分析，确立了FRED的可靠性。此外，我们还通过与最先进的方法进行的实证评估，证明了FRED在提供对文本模型的深入见解方面的有效性。

    Interpretability is essential for machine learning models to be trusted and deployed in critical domains. However, existing methods for interpreting text models are often complex, lack solid mathematical foundations, and their performance is not guaranteed. In this paper, we propose FRED (Faithful and Robust Explainer for textual Documents), a novel method for interpreting predictions over text. FRED identifies key words in a document that significantly impact the prediction when removed. We establish the reliability of FRED through formal definitions and theoretical analyses on interpretable classifiers. Additionally, our empirical evaluation against state-of-the-art methods demonstrates the effectiveness of FRED in providing insights into text models.
    
[^8]: SepVAE: 一种对比VAE用于分离病理模式和健康模式

    SepVAE: a contrastive VAE to separate pathological patterns from healthy ones. (arXiv:2307.06206v1 [cs.CV])

    [http://arxiv.org/abs/2307.06206](http://arxiv.org/abs/2307.06206)

    SepVAE是一种对比VAE方法，可以从健康数据和患者数据中分离出共同的和特定的变化因素。在多个应用中表现出比现有方法更好的性能。

    

    对比分析VAE（CA-VAEs）是一类变分自编码器（VAEs），旨在从背景数据集（BG）（即健康人群）和目标数据集（TG）（即患者）之间分离共同变化因素和仅存在于目标数据集中的因素。为此，这些方法将潜在空间分为一组显著特征（即特定于目标数据集）和一组共同特征（即存在于两个数据集中）。目前，所有模型都未能有效防止潜在空间之间的信息共享，并捕捉所有显著的变化因素。为此，我们引入了两个关键的正则化损失：共同表示和显著表示之间的解缠绕项，以及显著空间中背景和目标样本之间的分类项。我们展示了对三个医学应用和一个自然图像数据集（CelebA）的先前CA-VAEs方法的更好性能。代码和数据集可在GitHub上获取。

    Contrastive Analysis VAE (CA-VAEs) is a family of Variational auto-encoders (VAEs) that aims at separating the common factors of variation between a background dataset (BG) (i.e., healthy subjects) and a target dataset (TG) (i.e., patients) from the ones that only exist in the target dataset. To do so, these methods separate the latent space into a set of salient features (i.e., proper to the target dataset) and a set of common features (i.e., exist in both datasets). Currently, all models fail to prevent the sharing of information between latent spaces effectively and to capture all salient factors of variation. To this end, we introduce two crucial regularization losses: a disentangling term between common and salient representations and a classification term between background and target samples in the salient space. We show a better performance than previous CA-VAEs methods on three medical applications and a natural images dataset (CelebA). Code and datasets are available on GitHu
    
[^9]: 通过非负低秩半定规划实现最优K均值聚类

    Statistically Optimal K-means Clustering via Nonnegative Low-rank Semidefinite Programming. (arXiv:2305.18436v1 [stat.ML])

    [http://arxiv.org/abs/2305.18436](http://arxiv.org/abs/2305.18436)

    本文提出了一种与NMF算法一样简单且可扩展的K均值聚类算法，该算法通过解决非负低秩半定规划问题获得了强大的统计最优性保证，实验证明该算法在合成和实际数据集上表现优异。

    

    K均值聚类是一种广泛应用于大数据集中发现模式的机器学习方法。半定规划（SDP）松弛最近被提出用于解决K均值优化问题，具有很强的统计最优性保证。但实现SDP求解器的巨大成本使得这些保证无法应用于实际数据集。相比之下，非负矩阵分解（NMF）是一种简单的聚类算法，被机器学习从业者广泛使用，但缺乏坚实的统计基础或严格的保证。在本文中，我们描述了一种类似于NMF的算法，通过使用非凸Burer-Monteiro分解方法解决半定规划松弛的K均值公式的非负低秩限制。所得到的算法与最先进的NMF算法一样简单和可扩展，同时也享有与SDP相同的强大的统计最优性保证。在我们的实验中，我们证明了我们的算法优于现有的NMF算法，并在合成和实际数据集上表现与最先进的SDP求解器相当。

    $K$-means clustering is a widely used machine learning method for identifying patterns in large datasets. Semidefinite programming (SDP) relaxations have recently been proposed for solving the $K$-means optimization problem that enjoy strong statistical optimality guarantees, but the prohibitive cost of implementing an SDP solver renders these guarantees inaccessible to practical datasets. By contrast, nonnegative matrix factorization (NMF) is a simple clustering algorithm that is widely used by machine learning practitioners, but without a solid statistical underpinning nor rigorous guarantees. In this paper, we describe an NMF-like algorithm that works by solving a nonnegative low-rank restriction of the SDP relaxed $K$-means formulation using a nonconvex Burer--Monteiro factorization approach. The resulting algorithm is just as simple and scalable as state-of-the-art NMF algorithms, while also enjoying the same strong statistical optimality guarantees as the SDP. In our experiments,
    
[^10]: 无声放弃：如何从不确定数据中估计客户等待的耐心

    Silent Abandonment in Contact Centers: Estimating Customer Patience from Uncertain Data. (arXiv:2304.11754v1 [cs.SI])

    [http://arxiv.org/abs/2304.11754](http://arxiv.org/abs/2304.11754)

    该研究探究了客户在联系中心无声放弃的现象，通过不确定数据估计客户等待的耐心，揭示了这种现象对代理人时间和能力的浪费。

    

    为了提高服务质量，公司为客户提供与代理人进行交互的机会，其中大部分交流是基于文本的。这已成为近年来客户与公司交流的最受欢迎的渠道之一。然而，联系中心面临运营挑战，因为客户体验的常见代理，例如是否知道客户已放弃排队和他们等待服务的意愿（耐心），受到信息不确定性的影响。我们的研究聚焦于主要不确定性来源的影响：客户的无声放弃。这些客户在等待回答他们的查询时离开系统，但没有给出任何指示，例如关闭互动的移动应用程序。因此，系统不知道他们已经离开，并浪费代理人的时间和能力，直到意识到这一事实。本文表明，放弃客户中的30％-67％放弃时会采取无声放弃策略。

    In the quest to improve services, companies offer customers the opportunity to interact with agents through contact centers, where the communication is mainly text-based. This has become one of the favorite channels of communication with companies in recent years. However, contact centers face operational challenges, since the measurement of common proxies for customer experience, such as knowledge of whether customers have abandoned the queue and their willingness to wait for service (patience), are subject to information uncertainty. We focus this research on the impact of a main source of such uncertainty: silent abandonment by customers. These customers leave the system while waiting for a reply to their inquiry, but give no indication of doing so, such as closing the mobile app of the interaction. As a result, the system is unaware that they have left and waste agent time and capacity until this fact is realized. In this paper, we show that 30%-67% of the abandoning customers aban
    
[^11]: 在上下文强化学习领域进行在线连续超参数优化

    Online Continuous Hyperparameter Optimization for Contextual Bandits. (arXiv:2302.09440v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.09440](http://arxiv.org/abs/2302.09440)

    该论文提出了面向上下文强化学习的在线连续超参数调整框架CDT，能够动态地在搜索空间内学习最优参数配置。

    

    在随机上下文强化学习中，代理根据过去的经验从时间相关行动集中依次采取行动，以最小化总后悔。与许多其他机器学习算法一样，强化学习的性能严重依赖于其多个超参数，并且理论推导出的参数值可能导致实际上不令人满意的结果。此外，在强化学习环境下使用离线优化方法（如交叉验证）选择超参数是不可行的，因为决策必须实时进行。因此，我们提出了第一个面向上下文强化学习的在线连续超参数调整框架，以学习飞行中的最佳参数配置。具体而言，我们使用了一个名为CDT（Continuous Dynamic Tuning）的双层强化学习框架，并将超参数优化形式化为非平稳连续武器强化学习，在其中每个武器代表一种超参数组合。

    In stochastic contextual bandits, an agent sequentially makes actions from a time-dependent action set based on past experience to minimize the cumulative regret. Like many other machine learning algorithms, the performance of bandits heavily depends on their multiple hyperparameters, and theoretically derived parameter values may lead to unsatisfactory results in practice. Moreover, it is infeasible to use offline tuning methods like cross-validation to choose hyperparameters under the bandit environment, as the decisions should be made in real time. To address this challenge, we propose the first online continuous hyperparameter tuning framework for contextual bandits to learn the optimal parameter configuration within a search space on the fly. Specifically, we use a double-layer bandit framework named CDT (Continuous Dynamic Tuning) and formulate the hyperparameter optimization as a non-stationary continuum-armed bandit, where each arm represents a combination of hyperparameters, a
    

