# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GreeDy and CoDy: Counterfactual Explainers for Dynamic Graphs](https://arxiv.org/abs/2403.16846) | 该论文介绍了两种针对动态图的新颖反事实解释方法：GreeDy和CoDy。实验证明，CoDy在寻找重要反事实输入方面表现优异，成功率高达59%。 |
| [^2] | [Optimal Transport for Domain Adaptation through Gaussian Mixture Models](https://arxiv.org/abs/2403.13847) | 通过高斯混合模型进行域自适应的最优输运，可以实现源域和目标域混合成分之间的匹配，从而在失效诊断中取得最先进的性能。 |
| [^3] | [Federated Complex Qeury Answering](https://arxiv.org/abs/2402.14609) | 研究了在多源知识图谱上回答复杂查询的联邦式方法，解决了知识图谱中的隐私保护和答案检索的挑战 |
| [^4] | [Do deep neural networks have an inbuilt Occam's razor?.](http://arxiv.org/abs/2304.06670) | 该研究利用基于函数先验的贝叶斯视角来研究深度神经网络（DNNs）的表现来源，结果表明DNNs之所以成功，是因为它对于具有结构的数据，具备一种内在的奥卡姆剃刀式的归纳偏差，足以抵消函数数量及复杂度的指数级增长。 |

# 详细

[^1]: GreeDy和CoDy：动态图的反事实解释器

    GreeDy and CoDy: Counterfactual Explainers for Dynamic Graphs

    [https://arxiv.org/abs/2403.16846](https://arxiv.org/abs/2403.16846)

    该论文介绍了两种针对动态图的新颖反事实解释方法：GreeDy和CoDy。实验证明，CoDy在寻找重要反事实输入方面表现优异，成功率高达59%。

    

    时间图神经网络（TGNNs）对于建模具有时间变化交互的动态图至关重要，但由于其复杂的模型结构，在可解释性方面面临重大挑战。反事实解释对于理解模型决策至关重要，它研究输入图的变化如何影响结果。本文介绍了两种新颖的 TGNNs 反事实解释方法：GreeDy（动态图的贪心解释器）和 CoDy（动态图的反事实解释器）。它们将解释视为一个搜索问题，寻找改变模型预测的输入图修改。GreeDy 使用简单的贪心方法，而 CoDy 使用复杂的蒙特卡洛树搜索算法。实验证明，两种方法都能有效生成清晰的解释。值得注意的是，CoDy 的性能优于 GreeDy 和现有的事实方法，寻找到重要的反事实输入的成功率提高了高达 59\%。这突出了 CoDy 的优势。

    arXiv:2403.16846v1 Announce Type: cross  Abstract: Temporal Graph Neural Networks (TGNNs), crucial for modeling dynamic graphs with time-varying interactions, face a significant challenge in explainability due to their complex model structure. Counterfactual explanations, crucial for understanding model decisions, examine how input graph changes affect outcomes. This paper introduces two novel counterfactual explanation methods for TGNNs: GreeDy (Greedy Explainer for Dynamic Graphs) and CoDy (Counterfactual Explainer for Dynamic Graphs). They treat explanations as a search problem, seeking input graph alterations that alter model predictions. GreeDy uses a simple, greedy approach, while CoDy employs a sophisticated Monte Carlo Tree Search algorithm. Experiments show both methods effectively generate clear explanations. Notably, CoDy outperforms GreeDy and existing factual methods, with up to 59\% higher success rate in finding significant counterfactual inputs. This highlights CoDy's p
    
[^2]: 通过高斯混合模型进行域自适应的最优输运

    Optimal Transport for Domain Adaptation through Gaussian Mixture Models

    [https://arxiv.org/abs/2403.13847](https://arxiv.org/abs/2403.13847)

    通过高斯混合模型进行域自适应的最优输运，可以实现源域和目标域混合成分之间的匹配，从而在失效诊断中取得最先进的性能。

    

    在这篇论文中，我们探讨了通过最优输运进行域自适应的方法。我们提出了一种新颖的方法，即通过高斯混合模型对数据分布进行建模。这种策略使我们能够通过等价的离散问题解决连续最优输运。最优输运解决方案为我们提供了源域和目标域混合成分之间的匹配。通过这种匹配，我们可以在域之间映射数据点，或者将标签从源域组件转移到目标域。我们在失效诊断的两个域自适应基准测试中进行了实验，结果表明我们的方法具有最先进的性能。

    arXiv:2403.13847v1 Announce Type: cross  Abstract: In this paper we explore domain adaptation through optimal transport. We propose a novel approach, where we model the data distributions through Gaussian mixture models. This strategy allows us to solve continuous optimal transport through an equivalent discrete problem. The optimal transport solution gives us a matching between source and target domain mixture components. From this matching, we can map data points between domains, or transfer the labels from the source domain components towards the target domain. We experiment with 2 domain adaptation benchmarks in fault diagnosis, showing that our methods have state-of-the-art performance.
    
[^3]: 联邦式复杂查询答案方法研究

    Federated Complex Qeury Answering

    [https://arxiv.org/abs/2402.14609](https://arxiv.org/abs/2402.14609)

    研究了在多源知识图谱上回答复杂查询的联邦式方法，解决了知识图谱中的隐私保护和答案检索的挑战

    

    知识图谱中的复杂逻辑查询答案是一个具有挑战性的任务，已经得到广泛研究。执行复杂逻辑推理的能力是必不可少的，并支持各种基于图推理的下游任务，比如搜索引擎。最近提出了一些方法，将知识图谱实体和逻辑查询表示为嵌入向量，并从知识图谱中找到逻辑查询的答案。然而，现有的方法主要集中在查询单个知识图谱上，并不能应用于多个图形。此外，直接共享带有敏感信息的知识图谱可能会带来隐私风险，使得共享和构建一个聚合知识图谱用于推理以检索查询答案是不切实际的。因此，目前仍然不清楚如何在多源知识图谱上回答查询。一个实体可能涉及到多个知识图谱，对多个知识图谱进行推理，并在多源知识图谱上回答复杂查询对于发现知识是重要的。

    arXiv:2402.14609v1 Announce Type: cross  Abstract: Complex logical query answering is a challenging task in knowledge graphs (KGs) that has been widely studied. The ability to perform complex logical reasoning is essential and supports various graph reasoning-based downstream tasks, such as search engines. Recent approaches are proposed to represent KG entities and logical queries into embedding vectors and find answers to logical queries from the KGs. However, existing proposed methods mainly focus on querying a single KG and cannot be applied to multiple graphs. In addition, directly sharing KGs with sensitive information may incur privacy risks, making it impractical to share and construct an aggregated KG for reasoning to retrieve query answers. Thus, it remains unknown how to answer queries on multi-source KGs. An entity can be involved in various knowledge graphs and reasoning on multiple KGs and answering complex queries on multi-source KGs is important in discovering knowledge 
    
[^4]: 深度神经网络是否具备内置的奥卡姆剃刀？

    Do deep neural networks have an inbuilt Occam's razor?. (arXiv:2304.06670v1 [cs.LG])

    [http://arxiv.org/abs/2304.06670](http://arxiv.org/abs/2304.06670)

    该研究利用基于函数先验的贝叶斯视角来研究深度神经网络（DNNs）的表现来源，结果表明DNNs之所以成功，是因为它对于具有结构的数据，具备一种内在的奥卡姆剃刀式的归纳偏差，足以抵消函数数量及复杂度的指数级增长。

    

    超参数化深度神经网络（DNNs）的卓越性能必须源自于网络架构、训练算法和数据结构之间的相互作用。为了区分这三个部分，我们应用了基于DNN所表达的函数的贝叶斯视角来进行监督学习。经过网络确定的函数先验通过利用有序和混沌状态之间的转变而变化。对于布尔函数分类，我们利用函数的误差谱在数据上进行可能性的近似。当与先验相结合时，它可以精确地预测使用随机梯度下降训练的DNN的后验概率。该分析揭示了结构化数据，以及内在的奥卡姆剃刀式归纳偏差，即足以抵消复杂度随函数数量呈指数增长而产生的影响，是DNNs成功的关键。

    The remarkable performance of overparameterized deep neural networks (DNNs) must arise from an interplay between network architecture, training algorithms, and structure in the data. To disentangle these three components, we apply a Bayesian picture, based on the functions expressed by a DNN, to supervised learning. The prior over functions is determined by the network, and is varied by exploiting a transition between ordered and chaotic regimes. For Boolean function classification, we approximate the likelihood using the error spectrum of functions on data. When combined with the prior, this accurately predicts the posterior, measured for DNNs trained with stochastic gradient descent. This analysis reveals that structured data, combined with an intrinsic Occam's razor-like inductive bias towards (Kolmogorov) simple functions that is strong enough to counteract the exponential growth of the number of functions with complexity, is a key to the success of DNNs.
    

