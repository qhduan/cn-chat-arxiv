# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Boosting for Bounding the Worst-class Error.](http://arxiv.org/abs/2310.14890) | 该论文提出了一种基于Boosting的算法，可以保证最差类别训练误差的上界，并降低了最差类别的测试误差率。 |
| [^2] | [Conformal inference for regression on Riemannian Manifolds.](http://arxiv.org/abs/2310.08209) | 本文研究了在黎曼流形上进行回归场景的预测集，并证明了这些区域的经验版本在大样本下的收敛性。 |
| [^3] | [Boolformer: Symbolic Regression of Logic Functions with Transformers.](http://arxiv.org/abs/2309.12207) | Boolformer是第一个经过训练的Transformer架构，用于执行端到端的布尔函数符号回归。它可以预测复杂函数的简洁公式，并在提供不完整和有噪声观测时找到近似表达式。Boolformer在真实二分类数据集上展现出潜力作为可解释性替代方案，并在基因调控网络动力学建模任务中与最先进的遗传算法相比表现出竞争力。 |
| [^4] | [TBDetector:Transformer-Based Detector for Advanced Persistent Threats with Provenance Graph.](http://arxiv.org/abs/2304.02838) | 本论文提出了一种采用来源图和Transformer的高级持久性威胁检测方法，利用Transformer的自注意力编码器-解码器提取系统状态的长期上下文特征，并通过来源分析实现对长期运行系统的概括，以检测缓慢攻击。 |
| [^5] | [Real-world Machine Learning Systems: A survey from a Data-Oriented Architecture Perspective.](http://arxiv.org/abs/2302.04810) | 这项调查研究了现实世界中部署机器学习系统的数据导向架构（DOA）的采用情况，发现尽管没有明确提及DOA，但许多论文中的设计决策默默地遵循了DOA的原则。 |

# 详细

[^1]: Boosting用于界定最差分类误差

    Boosting for Bounding the Worst-class Error. (arXiv:2310.14890v1 [stat.ML])

    [http://arxiv.org/abs/2310.14890](http://arxiv.org/abs/2310.14890)

    该论文提出了一种基于Boosting的算法，可以保证最差类别训练误差的上界，并降低了最差类别的测试误差率。

    

    本文解决了最差类别误差率的问题，而不是针对所有类别的标准误差率的平均。例如，一个三类别分类任务，其中各类别的误差率分别为10％，10％和40％，其最差类别误差率为40％，而在类别平衡条件下的平均误差率为20％。最差类别错误在许多应用中很重要。例如，在医学图像分类任务中，对于恶性肿瘤类别具有40％的错误率而良性和健康类别具有10％的错误率是不能被接受的。我们提出了一种保证最差类别训练误差上界的提升算法，并推导出其泛化界。实验结果表明，该算法降低了最差类别的测试误差率，同时避免了对训练集的过拟合。

    This paper tackles the problem of the worst-class error rate, instead of the standard error rate averaged over all classes. For example, a three-class classification task with class-wise error rates of 10\%, 10\%, and 40\% has a worst-class error rate of 40\%, whereas the average is 20\% under the class-balanced condition. The worst-class error is important in many applications. For example, in a medical image classification task, it would not be acceptable for the malignant tumor class to have a 40\% error rate, while the benign and healthy classes have 10\% error rates.We propose a boosting algorithm that guarantees an upper bound of the worst-class training error and derive its generalization bound. Experimental results show that the algorithm lowers worst-class test error rates while avoiding overfitting to the training set.
    
[^2]: 在黎曼流形上进行回归的一致推断

    Conformal inference for regression on Riemannian Manifolds. (arXiv:2310.08209v1 [stat.ML])

    [http://arxiv.org/abs/2310.08209](http://arxiv.org/abs/2310.08209)

    本文研究了在黎曼流形上进行回归场景的预测集，并证明了这些区域的经验版本在大样本下的收敛性。

    

    在流形上进行回归，以及更广泛地说，对流形上的统计学有了重要的关注，因为这种类型的数据有大量的应用。圆形数据是一个经典示例，但协方差矩阵空间上的数据、主成分分析得到的Grassmann流形上的数据等也是如此。在本文中，我们研究了当响应变量$Y$位于流形上，而协变量$X$位于欧几里德空间时，回归场景的预测集。这扩展了[Lei and Wasserman, 2014]中在这一新领域中概述的概念。与一致推断中的传统原则一致，这些预测集是无分布的，表明对$(X, Y)$的联合分布没有施加特定的假设，而且它们保持非参数性质。我们证明了这些区域的经验版本在几乎必然收敛于无穷大时的收敛性。

    Regression on manifolds, and, more broadly, statistics on manifolds, has garnered significant importance in recent years due to the vast number of applications for this type of data. Circular data is a classic example, but so is data in the space of covariance matrices, data on the Grassmannian manifold obtained as a result of principal component analysis, among many others. In this work we investigate prediction sets for regression scenarios when the response variable, denoted by $Y$, resides in a manifold, and the covariable, denoted by X, lies in Euclidean space. This extends the concepts delineated in [Lei and Wasserman, 2014] to this novel context. Aligning with traditional principles in conformal inference, these prediction sets are distribution-free, indicating that no specific assumptions are imposed on the joint distribution of $(X, Y)$, and they maintain a non-parametric character. We prove the asymptotic almost sure convergence of the empirical version of these regions on th
    
[^3]: Boolformer: 用Transformer进行逻辑函数的符号回归

    Boolformer: Symbolic Regression of Logic Functions with Transformers. (arXiv:2309.12207v1 [cs.LG])

    [http://arxiv.org/abs/2309.12207](http://arxiv.org/abs/2309.12207)

    Boolformer是第一个经过训练的Transformer架构，用于执行端到端的布尔函数符号回归。它可以预测复杂函数的简洁公式，并在提供不完整和有噪声观测时找到近似表达式。Boolformer在真实二分类数据集上展现出潜力作为可解释性替代方案，并在基因调控网络动力学建模任务中与最先进的遗传算法相比表现出竞争力。

    

    在这项工作中，我们介绍了Boolformer，这是第一个经过训练的Transformer架构，用于执行端到端的布尔函数符号回归。首先，我们展示了当提供干净的真值表时，它可以预测复杂函数的简洁公式。然后，我们展示了它在提供不完整和有噪声观测时找到近似表达式的能力。我们在广泛的真实二分类数据集上评估了Boolformer，证明了它作为传统机器学习方法的可解释性替代品的潜力。最后，我们将其应用于建模基因调控网络动力学的常见任务。使用最近的基准测试，我们展示了Boolformer与最先进的遗传算法相比，速度提高了几个数量级。我们的代码和模型公开可用。

    In this work, we introduce Boolformer, the first Transformer architecture trained to perform end-to-end symbolic regression of Boolean functions. First, we show that it can predict compact formulas for complex functions which were not seen during training, when provided a clean truth table. Then, we demonstrate its ability to find approximate expressions when provided incomplete and noisy observations. We evaluate the Boolformer on a broad set of real-world binary classification datasets, demonstrating its potential as an interpretable alternative to classic machine learning methods. Finally, we apply it to the widespread task of modelling the dynamics of gene regulatory networks. Using a recent benchmark, we show that Boolformer is competitive with state-of-the art genetic algorithms with a speedup of several orders of magnitude. Our code and models are available publicly.
    
[^4]: 基于Transformer和来源图的高级持久性威胁检测方法

    TBDetector:Transformer-Based Detector for Advanced Persistent Threats with Provenance Graph. (arXiv:2304.02838v1 [cs.CR])

    [http://arxiv.org/abs/2304.02838](http://arxiv.org/abs/2304.02838)

    本论文提出了一种采用来源图和Transformer的高级持久性威胁检测方法，利用Transformer的自注意力编码器-解码器提取系统状态的长期上下文特征，并通过来源分析实现对长期运行系统的概括，以检测缓慢攻击。

    

    针对高级持久性威胁（APT）攻击的长期潜伏、隐秘多阶段攻击模式，本文提出了一种基于Transformer的APT检测方法，利用来源图提供的历史信息进行APT检测。该方法利用Transformer的自注意力编码器-解码器提取系统状态的长期上下文特征，并通过来源分析实现对长期运行系统的概括，以检测缓慢攻击。此外，作者还引入了异常评分，可评估不同系统状态的异常性。每个状态都有相应的相似度和隔离度分数的异常分数计算。为了评估该方法的有效性

    APT detection is difficult to detect due to the long-term latency, covert and slow multistage attack patterns of Advanced Persistent Threat (APT). To tackle these issues, we propose TBDetector, a transformer-based advanced persistent threat detection method for APT attack detection. Considering that provenance graphs provide rich historical information and have the powerful attacks historic correlation ability to identify anomalous activities, TBDetector employs provenance analysis for APT detection, which summarizes long-running system execution with space efficiency and utilizes transformer with self-attention based encoder-decoder to extract long-term contextual features of system states to detect slow-acting attacks. Furthermore, we further introduce anomaly scores to investigate the anomaly of different system states, where each state is calculated with an anomaly score corresponding to its similarity score and isolation score. To evaluate the effectiveness of the proposed method,
    
[^5]: 现实世界中的机器学习系统：基于数据导向架构的调查

    Real-world Machine Learning Systems: A survey from a Data-Oriented Architecture Perspective. (arXiv:2302.04810v2 [cs.SE] UPDATED)

    [http://arxiv.org/abs/2302.04810](http://arxiv.org/abs/2302.04810)

    这项调查研究了现实世界中部署机器学习系统的数据导向架构（DOA）的采用情况，发现尽管没有明确提及DOA，但许多论文中的设计决策默默地遵循了DOA的原则。

    

    随着对人工智能的兴趣不断增长，机器学习模型正在作为现实世界系统的一部分部署。这些系统的设计、实现和维护受到现实世界环境的挑战，这些环境产生了更多的异构数据，用户需要更快的响应速度和高效的资源消耗。这些要求将普遍存在的软件架构推向了极限，当部署基于机器学习的系统时。数据导向架构（DOA）是一个新兴的概念，它能更好地为集成机器学习模型的系统提供支持。DOA扩展了当前的架构，创建了数据驱动、松耦合、去中心化和开放的系统。尽管部署的机器学习系统的论文中没有提到DOA，但它们的作者在设计上隐含地遵循了DOA。为什么、如何以及在多大程度上采用DOA在这些系统中尚不清楚。隐含的设计决策限制了从业者对于设计基于机器学习的系统时DOA的认识。

    Machine Learning models are being deployed as parts of real-world systems with the upsurge of interest in artificial intelligence. The design, implementation, and maintenance of such systems are challenged by real-world environments that produce larger amounts of heterogeneous data and users requiring increasingly faster responses with efficient resource consumption. These requirements push prevalent software architectures to the limit when deploying ML-based systems. Data-oriented Architecture (DOA) is an emerging concept that equips systems better for integrating ML models. DOA extends current architectures to create data-driven, loosely coupled, decentralised, open systems. Even though papers on deployed ML-based systems do not mention DOA, their authors made design decisions that implicitly follow DOA. The reasons why, how, and the extent to which DOA is adopted in these systems are unclear. Implicit design decisions limit the practitioners' knowledge of DOA to design ML-based syst
    

