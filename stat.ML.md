# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Causal Discovery from Conditionally Stationary Time Series](https://arxiv.org/abs/2110.06257) | 该论文提出了一种State-Dependent Causal Inference（SDCI）方法，可以处理一类宽泛的非平稳时间序列，成功地回复出潜在的因果依赖关系。 |
| [^2] | [A Stability Principle for Learning under Non-Stationarity.](http://arxiv.org/abs/2310.18304) | 本研究提出了一个适用于非稳态环境的统计学习框架，通过应用稳定性原则选择回溯窗口来最大化历史数据利用，并保持累积偏差在可接受范围内。该方法展示了对未知非稳态的适应性，遗憾界在强凸或满足Lipschitz条件下是极小化的最优解。该研究的创新点是函数相似度度量和非稳态数据序列划分技术。 |
| [^3] | [Rethinking Fairness for Human-AI Collaboration.](http://arxiv.org/abs/2310.03647) | 在人工智能与人类合作中，需要重新思考公平性，因为完全遵守算法决策很少是现实可行的，因此我们需要设计稳健公平的算法推荐来提升公平性。 |
| [^4] | [A Nearly-Linear Time Algorithm for Structured Support Vector Machines.](http://arxiv.org/abs/2307.07735) | 这篇论文提出了针对结构化支持向量机的接近线性时间算法，解决了二次规划输入规模和解决时间的问题。 |
| [^5] | [Convergence of Message Passing Graph Neural Networks with Generic Aggregation On Large Random Graphs.](http://arxiv.org/abs/2304.11140) | 本文研究了消息传递图神经网络在随机图模型上的收敛性，将收敛结论从只适用于度规范化平均聚合函数扩展到所有传统聚合函数，并考虑了聚合函数采用逐个坐标最大值时的情况。 |
| [^6] | [Predicting Cellular Responses with Variational Causal Inference and Refined Relational Information.](http://arxiv.org/abs/2210.00116) | 本研究利用基因调控网络信息设计了一种新的因果推断框架，并通过邻接矩阵更新技术预训练图卷积网络以更好地预测细胞在反事实干扰下的基因表达。同时，我们提出了一个鲁棒的估计器来高效估计边缘干扰效应。研究结果展示了该框架的优越性能。 |

# 详细

[^1]: 从有条件平稳时间序列中进行因果发现

    Causal Discovery from Conditionally Stationary Time Series

    [https://arxiv.org/abs/2110.06257](https://arxiv.org/abs/2110.06257)

    该论文提出了一种State-Dependent Causal Inference（SDCI）方法，可以处理一类宽泛的非平稳时间序列，成功地回复出潜在的因果依赖关系。

    

    因果发现，即从观测数据推断潜在的因果关系，已被证明对AI系统具有极大挑战。在时间序列建模背景下，传统的因果发现方法主要考虑具有完全观测变量和/或来自平稳时间序列的数据的受限场景。我们开发了一种因果发现方法来处理一类宽泛的非平稳时间序列，即在条件上是平稳的条件平稳时间序列，其中非平稳行为被建模为在一组（可能是隐藏的）状态变量上的平稳性。命名为State-Dependent Causal Inference（SDCI），我们的方法能够可证地回复出潜在的因果依赖关系，证明在完全观察到的状态下，并在存在隐藏状态时经验性地实现。后者通过对合成线性系统和非线性粒子相互作用数据的实验进行验证，SDCI实现了优于基线因果发现方法的性能。

    arXiv:2110.06257v2 Announce Type: replace  Abstract: Causal discovery, i.e., inferring underlying causal relationships from observational data, has been shown to be highly challenging for AI systems. In time series modeling context, traditional causal discovery methods mainly consider constrained scenarios with fully observed variables and/or data from stationary time-series. We develop a causal discovery approach to handle a wide class of non-stationary time-series that are conditionally stationary, where the non-stationary behaviour is modeled as stationarity conditioned on a set of (possibly hidden) state variables. Named State-Dependent Causal Inference (SDCI), our approach is able to recover the underlying causal dependencies, provably with fully-observed states and empirically with hidden states. The latter is confirmed by experiments on synthetic linear system and nonlinear particle interaction data, where SDCI achieves superior performance over baseline causal discovery methods
    
[^2]: 学习非稳态条件下的稳定性原则

    A Stability Principle for Learning under Non-Stationarity. (arXiv:2310.18304v1 [cs.LG])

    [http://arxiv.org/abs/2310.18304](http://arxiv.org/abs/2310.18304)

    本研究提出了一个适用于非稳态环境的统计学习框架，通过应用稳定性原则选择回溯窗口来最大化历史数据利用，并保持累积偏差在可接受范围内。该方法展示了对未知非稳态的适应性，遗憾界在强凸或满足Lipschitz条件下是极小化的最优解。该研究的创新点是函数相似度度量和非稳态数据序列划分技术。

    

    我们在非稳定环境中开发了一个灵活的统计学习框架。在每个时间段，我们的方法应用稳定性原则来选择一个回溯窗口，最大限度地利用历史数据，同时将累积偏差保持在与随机误差相对可接受的范围内。我们的理论展示了该方法对未知非稳定性的适应性。当人口损失函数强凸或仅满足Lipschitz条件时，遗憾界是极小化的最优解，仅受对数因子的影响。我们的分析核心是两个新颖的组成部分：函数之间的相似度度量和将非稳态数据序列划分为准稳态片段的分割技术。

    We develop a versatile framework for statistical learning in non-stationary environments. In each time period, our approach applies a stability principle to select a look-back window that maximizes the utilization of historical data while keeping the cumulative bias within an acceptable range relative to the stochastic error. Our theory showcases the adaptability of this approach to unknown non-stationarity. The regret bound is minimax optimal up to logarithmic factors when the population losses are strongly convex, or Lipschitz only. At the heart of our analysis lie two novel components: a measure of similarity between functions and a segmentation technique for dividing the non-stationary data sequence into quasi-stationary pieces.
    
[^3]: 重新思考人工智能与人类合作的公平性

    Rethinking Fairness for Human-AI Collaboration. (arXiv:2310.03647v1 [cs.LG])

    [http://arxiv.org/abs/2310.03647](http://arxiv.org/abs/2310.03647)

    在人工智能与人类合作中，需要重新思考公平性，因为完全遵守算法决策很少是现实可行的，因此我们需要设计稳健公平的算法推荐来提升公平性。

    

    现有的算法公平性方法旨在确保人类决策者完全遵守算法决策时实现公平的结果。然而，在人工智能与人类合作中，完全遵守算法决策很少是现实或理想的结果。然而，最近的研究表明，对公平算法的选择性遵守会相对于人类以前的政策增加歧视。因此，确保公平结果需要基本不同的算法设计原则，以确保对决策者（事先不知道）的遵守模式具有稳健性。我们定义了一种遵守稳健公平的算法推荐，无论人类的遵守模式如何，它们都能确保在决策中改善公平性（弱形意义上）。我们提出了一种简单的优化策略来确定最佳的性能改进遵守稳健公平策略。然而，我们发现设计算法推荐可能是不可行的。

    Existing approaches to algorithmic fairness aim to ensure equitable outcomes if human decision-makers comply perfectly with algorithmic decisions. However, perfect compliance with the algorithm is rarely a reality or even a desirable outcome in human-AI collaboration. Yet, recent studies have shown that selective compliance with fair algorithms can amplify discrimination relative to the prior human policy. As a consequence, ensuring equitable outcomes requires fundamentally different algorithmic design principles that ensure robustness to the decision-maker's (a priori unknown) compliance pattern. We define the notion of compliance-robustly fair algorithmic recommendations that are guaranteed to (weakly) improve fairness in decisions, regardless of the human's compliance pattern. We propose a simple optimization strategy to identify the best performance-improving compliance-robustly fair policy. However, we show that it may be infeasible to design algorithmic recommendations that are s
    
[^4]: 结构化支持向量机的接近线性时间算法

    A Nearly-Linear Time Algorithm for Structured Support Vector Machines. (arXiv:2307.07735v1 [math.OC])

    [http://arxiv.org/abs/2307.07735](http://arxiv.org/abs/2307.07735)

    这篇论文提出了针对结构化支持向量机的接近线性时间算法，解决了二次规划输入规模和解决时间的问题。

    

    二次规划是凸优化领域中的基本问题。许多实际任务可以表示为二次规划，例如支持向量机（SVM）。在深度学习方法盛行之前，线性SVM是过去三十年来最流行的机器学习工具之一。一般来说，一个二次规划的输入规模为Θ(n^2)（其中n是变量的数量），因此解决该问题需要Ω(n^2)的时间。然而，SVM产生的二次规划的输入规模为O(n)，这使得设计接近线性时间算法成为可能。两个重要的SVM类别是具有低秩核因式分解和低树宽规模的程序。低树宽凸优化在过去几年中引起了越来越多的关注（例如线性规划[Dong, Lee and Ye 2021]和半定规划[Gu and Song 2022]）。因此，一个重要的开放问题是是否存在接近线性时间算法。

    Quadratic programming is a fundamental problem in the field of convex optimization. Many practical tasks can be formulated as quadratic programming, for example, the support vector machine (SVM). Linear SVM is one of the most popular tools over the last three decades in machine learning before deep learning method dominating.  In general, a quadratic program has input size $\Theta(n^2)$ (where $n$ is the number of variables), thus takes $\Omega(n^2)$ time to solve. Nevertheless, quadratic programs coming from SVMs has input size $O(n)$, allowing the possibility of designing nearly-linear time algorithms. Two important classes of SVMs are programs admitting low-rank kernel factorizations and low-treewidth programs. Low-treewidth convex optimization has gained increasing interest in the past few years (e.g.~linear programming [Dong, Lee and Ye 2021] and semidefinite programming [Gu and Song 2022]). Therefore, an important open question is whether there exist nearly-linear time algorithms
    
[^5]: 基于消息传递的图神经网络在大规模随机图上的通用聚合收敛性研究

    Convergence of Message Passing Graph Neural Networks with Generic Aggregation On Large Random Graphs. (arXiv:2304.11140v1 [stat.ML])

    [http://arxiv.org/abs/2304.11140](http://arxiv.org/abs/2304.11140)

    本文研究了消息传递图神经网络在随机图模型上的收敛性，将收敛结论从只适用于度规范化平均聚合函数扩展到所有传统聚合函数，并考虑了聚合函数采用逐个坐标最大值时的情况。

    

    本文研究了消息传递图神经网络在随机图模型上的收敛性，当节点数量趋近于无限时，该网络模型能收敛于其连续模型。迄今为止，该收敛性结果只适用于聚合函数采用度规范化平均值形式的网络结构。我们将此结果扩展到包含所有传统消息传递图神经网络的大类聚合函数上，例如基于注意力和最大卷积的网络。在一定假设下，我们给出了高概率的非渐进上限来量化这种收敛性。我们的主要结果基于McDiarmid不等式。有趣的是，我们特别处理了聚合函数采用逐个坐标最大值的情况，因为它需要非常不同的证明技巧，并产生了定性不同的收敛率。

    We study the convergence of message passing graph neural networks on random graph models to their continuous counterpart as the number of nodes tends to infinity. Until now, this convergence was only known for architectures with aggregation functions in the form of degree-normalized means. We extend such results to a very large class of aggregation functions, that encompasses all classically used message passing graph neural networks, such as attention-based mesage passing or max convolutional message passing on top of (degree-normalized) convolutional message passing. Under mild assumptions, we give non asymptotic bounds with high probability to quantify this convergence. Our main result is based on the McDiarmid inequality. Interestingly, we treat the case where the aggregation is a coordinate-wise maximum separately, at it necessitates a very different proof technique and yields a qualitatively different convergence rate.
    
[^6]: 利用变分因果推断和精细关系信息预测细胞响应

    Predicting Cellular Responses with Variational Causal Inference and Refined Relational Information. (arXiv:2210.00116v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.00116](http://arxiv.org/abs/2210.00116)

    本研究利用基因调控网络信息设计了一种新的因果推断框架，并通过邻接矩阵更新技术预训练图卷积网络以更好地预测细胞在反事实干扰下的基因表达。同时，我们提出了一个鲁棒的估计器来高效估计边缘干扰效应。研究结果展示了该框架的优越性能。

    

    预测细胞在干扰下的响应可能为药物研发和个性化治疗带来重要好处。在本研究中，我们提出了一种新的图形变分贝叶斯因果推断框架，预测细胞在反事实干扰下（即细胞未真实接收的干扰）的基因表达，利用代表生物学知识的基因调控网络（GRN）信息来辅助个性化细胞响应预测。我们还针对数据自适应GRN开发了邻接矩阵更新技术用于图卷积网络的预训练，在模型性能上提供了更多的基因关系洞见。

    Predicting the responses of a cell under perturbations may bring important benefits to drug discovery and personalized therapeutics. In this work, we propose a novel graph variational Bayesian causal inference framework to predict a cell's gene expressions under counterfactual perturbations (perturbations that this cell did not factually receive), leveraging information representing biological knowledge in the form of gene regulatory networks (GRNs) to aid individualized cellular response predictions. Aiming at a data-adaptive GRN, we also developed an adjacency matrix updating technique for graph convolutional networks and used it to refine GRNs during pre-training, which generated more insights on gene relations and enhanced model performance. Additionally, we propose a robust estimator within our framework for the asymptotically efficient estimation of marginal perturbation effect, which is yet to be carried out in previous works. With extensive experiments, we exhibited the advanta
    

