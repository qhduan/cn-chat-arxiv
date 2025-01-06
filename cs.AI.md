# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Specification Overfitting in Artificial Intelligence](https://arxiv.org/abs/2403.08425) | 本文定义了规格过度拟合问题，即系统过度关注指定指标而损害了高级要求和任务性能。 |
| [^2] | [Self-Supervised Deconfounding Against Spatio-Temporal Shifts: Theory and Modeling](https://arxiv.org/abs/2311.12472) | 该论文针对时空数据中常见的分布变化问题提出了一种自监督去混淆方法并提出了名为DCA的理论解决方案。 |
| [^3] | [Provably Efficient Learning in Partially Observable Contextual Bandit.](http://arxiv.org/abs/2308.03572) | 本文研究了在部分可观察情境轮盘赌中的转移学习问题，提出了一种通过优化问题识别行为和奖励因果效应的方法，并利用因果约束来改进轮盘赌算法。 |
| [^4] | [Communication-Efficient Split Learning via Adaptive Feature-Wise Compression.](http://arxiv.org/abs/2307.10805) | 该论文提出了一个名为SplitFC的通信高效的分割学习框架，通过两种自适应压缩策略来减少中间特征和梯度向量的通信开销，这些策略分别是自适应特征逐渐掉落和自适应特征逐渐量化。 |

# 详细

[^1]: 人工智能规格过度拟合问题

    Specification Overfitting in Artificial Intelligence

    [https://arxiv.org/abs/2403.08425](https://arxiv.org/abs/2403.08425)

    本文定义了规格过度拟合问题，即系统过度关注指定指标而损害了高级要求和任务性能。

    

    机器学习（ML）和人工智能（AI）方法经常被批评存在固有的偏见，以及缺乏控制、问责和透明度，监管机构因此难以控制这种技术的潜在负面影响。高级要求，如公平性和鲁棒性，需要被形式化为具体的规格度量，而这些度量是捕捉基本要求的独立方面的不完美代理。鉴于不同指标之间可能存在的权衡及其对过度优化的脆弱性，将规格度量整合到系统开发过程中并不是一件简单的事情。本文定义了规格过度拟合，即系统过度侧重于指定的度量，从而损害了高级要求和任务性能。我们进行了大量文献调研，对研究人员如何提出、测量和优化规格进行了分类。

    arXiv:2403.08425v1 Announce Type: new  Abstract: Machine learning (ML) and artificial intelligence (AI) approaches are often criticized for their inherent bias and for their lack of control, accountability, and transparency. Consequently, regulatory bodies struggle with containing this technology's potential negative side effects. High-level requirements such as fairness and robustness need to be formalized into concrete specification metrics, imperfect proxies that capture isolated aspects of the underlying requirements. Given possible trade-offs between different metrics and their vulnerability to over-optimization, integrating specification metrics in system development processes is not trivial. This paper defines specification overfitting, a scenario where systems focus excessively on specified metrics to the detriment of high-level requirements and task performance. We present an extensive literature survey to categorize how researchers propose, measure, and optimize specification
    
[^2]: 针对时空偏移的自监督去混淆：理论与建模

    Self-Supervised Deconfounding Against Spatio-Temporal Shifts: Theory and Modeling

    [https://arxiv.org/abs/2311.12472](https://arxiv.org/abs/2311.12472)

    该论文针对时空数据中常见的分布变化问题提出了一种自监督去混淆方法并提出了名为DCA的理论解决方案。

    

    作为时空（ST）数据的重要应用，ST交通预测在提高城市出行效率和促进可持续发展中起着至关重要的作用。本文首先通过构建过去交通数据、未来交通数据和外部ST上下文的因果图，系统地阐明了过去艺术作品在OOD交通数据上的失败是由于ST上下文充当了混淆因素，即过去数据和未来数据的共同原因。然后，我们从因果角度提出了一种理论解决方案，称为Disentangled Contextual Adjustment（DCA）。

    arXiv:2311.12472v2 Announce Type: replace  Abstract: As an important application of spatio-temporal (ST) data, ST traffic forecasting plays a crucial role in improving urban travel efficiency and promoting sustainable development. In practice, the dynamics of traffic data frequently undergo distributional shifts attributed to external factors such as time evolution and spatial differences. This entails forecasting models to handle the out-of-distribution (OOD) issue where test data is distributed differently from training data. In this work, we first formalize the problem by constructing a causal graph of past traffic data, future traffic data, and external ST contexts. We reveal that the failure of prior arts in OOD traffic data is due to ST contexts acting as a confounder, i.e., the common cause for past data and future ones. Then, we propose a theoretical solution named Disentangled Contextual Adjustment (DCA) from a causal lens. It differentiates invariant causal correlations again
    
[^3]: 在部分可观察情境轮盘赌中的可证效率学习

    Provably Efficient Learning in Partially Observable Contextual Bandit. (arXiv:2308.03572v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2308.03572](http://arxiv.org/abs/2308.03572)

    本文研究了在部分可观察情境轮盘赌中的转移学习问题，提出了一种通过优化问题识别行为和奖励因果效应的方法，并利用因果约束来改进轮盘赌算法。

    

    本文研究了在部分可观察情境轮盘赌中的转移学习问题，其中代理人仅有来自其他代理人的有限知识，并且对隐藏的混淆因素只有部分信息。我们将该问题转化为通过优化问题来识别或部分识别行为和奖励之间的因果效应。为了解决这些优化问题，我们将未知分布的原始功能约束离散化为线性约束，并通过顺序解线性规划来采样兼容的因果模型，以考虑估计误差得到因果约束。我们的采样算法为适当的采样分布提供了理想的收敛结果。然后，我们展示了如何将因果约束应用于改进经典的轮盘赌算法，并以行动集和函数空间规模为参考改变了遗憾值。值得注意的是，在允许我们处理一般情境分布的函数逼近任务中

    In this paper, we investigate transfer learning in partially observable contextual bandits, where agents have limited knowledge from other agents and partial information about hidden confounders. We first convert the problem to identifying or partially identifying causal effects between actions and rewards through optimization problems. To solve these optimization problems, we discretize the original functional constraints of unknown distributions into linear constraints, and sample compatible causal models via sequentially solving linear programmings to obtain causal bounds with the consideration of estimation error. Our sampling algorithms provide desirable convergence results for suitable sampling distributions. We then show how causal bounds can be applied to improving classical bandit algorithms and affect the regrets with respect to the size of action sets and function spaces. Notably, in the task with function approximation which allows us to handle general context distributions
    
[^4]: 通过自适应特征逐渐压缩实现高效的分割学习

    Communication-Efficient Split Learning via Adaptive Feature-Wise Compression. (arXiv:2307.10805v1 [cs.DC])

    [http://arxiv.org/abs/2307.10805](http://arxiv.org/abs/2307.10805)

    该论文提出了一个名为SplitFC的通信高效的分割学习框架，通过两种自适应压缩策略来减少中间特征和梯度向量的通信开销，这些策略分别是自适应特征逐渐掉落和自适应特征逐渐量化。

    

    本文提出了一种名为SplitFC的新颖的通信高效的分割学习（SL）框架，它减少了在SL培训过程中传输中间特征和梯度向量所需的通信开销。SplitFC的关键思想是利用矩阵的列所展示的不同的离散程度。SplitFC整合了两种压缩策略：（i）自适应特征逐渐掉落和（ii）自适应特征逐渐量化。在第一种策略中，中间特征向量根据这些向量的标准偏差确定自适应掉落概率进行掉落。然后，由于链式规则，与被丢弃的特征向量相关联的中间梯度向量也会被丢弃。在第二种策略中，非丢弃的中间特征和梯度向量使用基于向量范围确定的自适应量化级别进行量化。为了尽量减小量化误差，最优量化是。

    This paper proposes a novel communication-efficient split learning (SL) framework, named SplitFC, which reduces the communication overhead required for transmitting intermediate feature and gradient vectors during the SL training process. The key idea of SplitFC is to leverage different dispersion degrees exhibited in the columns of the matrices. SplitFC incorporates two compression strategies: (i) adaptive feature-wise dropout and (ii) adaptive feature-wise quantization. In the first strategy, the intermediate feature vectors are dropped with adaptive dropout probabilities determined based on the standard deviation of these vectors. Then, by the chain rule, the intermediate gradient vectors associated with the dropped feature vectors are also dropped. In the second strategy, the non-dropped intermediate feature and gradient vectors are quantized using adaptive quantization levels determined based on the ranges of the vectors. To minimize the quantization error, the optimal quantizatio
    

