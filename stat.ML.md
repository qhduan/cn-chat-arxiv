# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards Understanding the Word Sensitivity of Attention Layers: A Study via Random Features](https://arxiv.org/abs/2402.02969) | 通过研究随机特征，我们发现注意力层具有较高的词敏感性，这对于理解transformers的成功以及自然语言处理任务中的上下文含义非常重要。 |
| [^2] | [Convergence of flow-based generative models via proximal gradient descent in Wasserstein space.](http://arxiv.org/abs/2310.17582) | 本文通过在Wasserstein空间中应用近端梯度下降，证明了基于流的生成模型的收敛性，并提供了生成数据分布的理论保证。 |
| [^3] | [Geometric-Based Pruning Rules For Change Point Detection in Multiple Independent Time Series.](http://arxiv.org/abs/2306.09555) | 该论文提出了一些基于几何形状的扩展函数剪枝规则，用于解决在多个独立时间序列中检测多个变点的问题，在小维度情况下可以比函数剪枝更快地准确检测更好。 |
| [^4] | [Stability, Generalization and Privacy: Precise Analysis for Random and NTK Features.](http://arxiv.org/abs/2305.12100) | 本论文研究了ERM训练模型对抗强大黑盒攻击的安全问题，并通过两个指标量化模型安全性：单个样本的稳定性和查询与原始数据特征的对齐。在研究中，通过研究RF和NTK回归，证明随着泛化能力的提高，隐私保护可以得到加强。 |
| [^5] | [Uniform Pessimistic Risk and Optimal Portfolio.](http://arxiv.org/abs/2303.07158) | 本文提出了一种称为统一悲观风险的综合$\alpha$-风险版本和基于风险获得最优组合的计算算法，该方法可以用于估计韩国股票的悲观最优组合模型。 |

# 详细

[^1]: 关于注意力层的词敏感性的理解：通过随机特征的研究

    Towards Understanding the Word Sensitivity of Attention Layers: A Study via Random Features

    [https://arxiv.org/abs/2402.02969](https://arxiv.org/abs/2402.02969)

    通过研究随机特征，我们发现注意力层具有较高的词敏感性，这对于理解transformers的成功以及自然语言处理任务中的上下文含义非常重要。

    

    揭示transformers异常成功背后原因需要更好地理解为什么注意力层适用于自然语言处理任务。特别是，这些任务要求预测模型捕捉上下文含义，即使句子很长，这往往取决于一个或几个词。我们的工作在随机特征的典型设置中研究了这一关键属性，称为词敏感性（WS）。我们展示了注意力层具有较高的WS，即在嵌入空间中存在一个向量，能够大幅扰动随机注意力特征映射。这个论点关键地利用了注意力层中softmax的作用，突显了它相对于其他激活函数（如ReLU）的优势。相反，标准随机特征的WS是$1/\sqrt{n}$阶的，$n$是文本样本中的单词数，因此它随上下文的长度而衰减。然后，我们将这些关于词敏感性的结果转化为泛化界：由于...

    Unveiling the reasons behind the exceptional success of transformers requires a better understanding of why attention layers are suitable for NLP tasks. In particular, such tasks require predictive models to capture contextual meaning which often depends on one or few words, even if the sentence is long. Our work studies this key property, dubbed word sensitivity (WS), in the prototypical setting of random features. We show that attention layers enjoy high WS, namely, there exists a vector in the space of embeddings that largely perturbs the random attention features map. The argument critically exploits the role of the softmax in the attention layer, highlighting its benefit compared to other activations (e.g., ReLU). In contrast, the WS of standard random features is of order $1/\sqrt{n}$, $n$ being the number of words in the textual sample, and thus it decays with the length of the context. We then translate these results on the word sensitivity into generalization bounds: due to th
    
[^2]: 在Wasserstein空间中通过近端梯度下降实现基于流的生成模型的收敛性

    Convergence of flow-based generative models via proximal gradient descent in Wasserstein space. (arXiv:2310.17582v1 [stat.ML])

    [http://arxiv.org/abs/2310.17582](http://arxiv.org/abs/2310.17582)

    本文通过在Wasserstein空间中应用近端梯度下降，证明了基于流的生成模型的收敛性，并提供了生成数据分布的理论保证。

    

    基于流的生成模型在计算数据生成和似然函数方面具有一定的优势，并且最近在实证表现上显示出竞争力。与相关基于分数扩散模型的积累理论研究相比，对于在正向（数据到噪声）和反向（噪声到数据）方向上都是确定性的流模型的分析还很少。本文通过在归一化流网络中实施Jordan-Kinderleherer-Otto（JKO）方案的所谓JKO流模型，提供了通过渐进流模型生成数据分布的理论保证。利用Wasserstein空间中近端梯度下降（GD）的指数收敛性，我们证明了通过JKO流模型生成数据的Kullback-Leibler（KL）保证为$O(\varepsilon^2)$，其中使用$N \lesssim \log (1/\varepsilon)$个JKO步骤（流中的$N$个残差块），其中$\varepsilon$是每步一阶条件的误差。

    Flow-based generative models enjoy certain advantages in computing the data generation and the likelihood, and have recently shown competitive empirical performance. Compared to the accumulating theoretical studies on related score-based diffusion models, analysis of flow-based models, which are deterministic in both forward (data-to-noise) and reverse (noise-to-data) directions, remain sparse. In this paper, we provide a theoretical guarantee of generating data distribution by a progressive flow model, the so-called JKO flow model, which implements the Jordan-Kinderleherer-Otto (JKO) scheme in a normalizing flow network. Leveraging the exponential convergence of the proximal gradient descent (GD) in Wasserstein space, we prove the Kullback-Leibler (KL) guarantee of data generation by a JKO flow model to be $O(\varepsilon^2)$ when using $N \lesssim \log (1/\varepsilon)$ many JKO steps ($N$ Residual Blocks in the flow) where $\varepsilon $ is the error in the per-step first-order condit
    
[^3]: 基于几何的规则在多个独立时间序列中进行变点检测的剪枝

    Geometric-Based Pruning Rules For Change Point Detection in Multiple Independent Time Series. (arXiv:2306.09555v1 [stat.ME])

    [http://arxiv.org/abs/2306.09555](http://arxiv.org/abs/2306.09555)

    该论文提出了一些基于几何形状的扩展函数剪枝规则，用于解决在多个独立时间序列中检测多个变点的问题，在小维度情况下可以比函数剪枝更快地准确检测更好。

    

    我们考虑检测多个独立时间序列中的多个变点的问题。寻找最佳分割可以表达为在给定成本函数上的最小化问题。我们专注于解决此问题的动态规划算法。当变化次数与数据长度成比例时，PELT算法中编码的基于不等式的剪枝规则会导致线性时间复杂度。另一种称为函数剪枝的剪枝方法，对于分析单变量时间序列而言，无论变化次数如何，其时间复杂度都接近于线性。我们提出了一些基于使用简单几何形状（球体和超矩形）的函数剪枝的扩展，重点关注高斯情况，但我们的一些规则可以轻松扩展到指数族。在模拟研究中，我们比较了不同基于几何的剪枝规则的计算效率。我们表明，在小维度情况下，使用超矩形和球体进行剪枝可以比函数剪枝更快速地准确检测更好。对于较大维度，超矩形变得不那么高效，而球体仅在高信噪比的情况下仍然具有竞争力。

    We consider the problem of detecting multiple changes in multiple independent time series. The search for the best segmentation can be expressed as a minimization problem over a given cost function. We focus on dynamic programming algorithms that solve this problem exactly. When the number of changes is proportional to data length, an inequality-based pruning rule encoded in the PELT algorithm leads to a linear time complexity. Another type of pruning, called functional pruning, gives a close-to-linear time complexity whatever the number of changes, but only for the analysis of univariate time series.  We propose a few extensions of functional pruning for multiple independent time series based on the use of simple geometric shapes (balls and hyperrectangles). We focus on the Gaussian case, but some of our rules can be easily extended to the exponential family. In a simulation study we compare the computational efficiency of different geometric-based pruning rules. We show that for smal
    
[^4]: 稳定性、泛化性和隐私保护：对于随机特征和NTK特征的精确分析

    Stability, Generalization and Privacy: Precise Analysis for Random and NTK Features. (arXiv:2305.12100v1 [stat.ML])

    [http://arxiv.org/abs/2305.12100](http://arxiv.org/abs/2305.12100)

    本论文研究了ERM训练模型对抗强大黑盒攻击的安全问题，并通过两个指标量化模型安全性：单个样本的稳定性和查询与原始数据特征的对齐。在研究中，通过研究RF和NTK回归，证明随着泛化能力的提高，隐私保护可以得到加强。

    

    深度学习模型容易受到恢复攻击，引起用户隐私保护的担忧。针对经验风险最小化（ERM）等常见算法通常不能直接实施安全保障的问题，本文研究了ERM训练模型对抗特定强大黑盒子攻击的安全问题。我们的分析通过两个看似不同但有联系的指标来量化模型安全性：一是相对于单个训练样本的模型稳定性，另一个是攻击查询和原始数据特征的特征对齐。虽然前者在学习理论中已经得到了很好的阐述，并与经典工作中的泛化误差相关，但在我们的研究中，第二种特性是新颖的。我们的关键技术结果为两种原型设置提供了特征对齐的精确刻画：随机特征（RF）和神经切向核（NTK）回归。这证明，随着泛化能力的提高，隐私保护能够得到加强，同时揭示了其他有趣的性质。

    Deep learning models can be vulnerable to recovery attacks, raising privacy concerns to users, and widespread algorithms such as empirical risk minimization (ERM) often do not directly enforce safety guarantees. In this paper, we study the safety of ERM-trained models against a family of powerful black-box attacks. Our analysis quantifies this safety via two separate terms: (i) the model stability with respect to individual training samples, and (ii) the feature alignment between the attacker query and the original data. While the first term is well established in learning theory and it is connected to the generalization error in classical work, the second one is, to the best of our knowledge, novel. Our key technical result provides a precise characterization of the feature alignment for the two prototypical settings of random features (RF) and neural tangent kernel (NTK) regression. This proves that privacy strengthens with an increase in the generalization capability, unveiling also
    
[^5]: 统一悲观风险和最优组合

    Uniform Pessimistic Risk and Optimal Portfolio. (arXiv:2303.07158v1 [q-fin.PM])

    [http://arxiv.org/abs/2303.07158](http://arxiv.org/abs/2303.07158)

    本文提出了一种称为统一悲观风险的综合$\alpha$-风险版本和基于风险获得最优组合的计算算法，该方法可以用于估计韩国股票的悲观最优组合模型。

    This paper proposes a version of integrated $\alpha$-risk called the uniform pessimistic risk and a computational algorithm to obtain an optimal portfolio based on the risk. The proposed method can be used to estimate the pessimistic optimal portfolio models for Korean stocks.

    资产配置的最优性已经在风险度量的理论分析中得到广泛讨论。悲观主义是一种超越传统最优组合模型的最有吸引力的方法之一，$\alpha$-风险在推导出广泛的悲观最优组合中起着关键作用。然而，由悲观风险评估的最优组合的估计仍然具有挑战性，因为缺乏可用的估计模型和计算算法。在本研究中，我们提出了一种称为统一悲观风险的综合$\alpha$-风险版本和基于风险获得最优组合的计算算法。此外，我们从多个分位数回归、适当的评分规则和分布鲁棒优化三个不同的方法来研究所提出的风险的理论性质。同时，统一悲观风险被应用于估计韩国股票的悲观最优组合模型。

    The optimality of allocating assets has been widely discussed with the theoretical analysis of risk measures. Pessimism is one of the most attractive approaches beyond the conventional optimal portfolio model, and the $\alpha$-risk plays a crucial role in deriving a broad class of pessimistic optimal portfolios. However, estimating an optimal portfolio assessed by a pessimistic risk is still challenging due to the absence of an available estimation model and a computational algorithm. In this study, we propose a version of integrated $\alpha$-risk called the uniform pessimistic risk and the computational algorithm to obtain an optimal portfolio based on the risk. Further, we investigate the theoretical properties of the proposed risk in view of three different approaches: multiple quantile regression, the proper scoring rule, and distributionally robust optimization. Also, the uniform pessimistic risk is applied to estimate the pessimistic optimal portfolio models for the Korean stock 
    

