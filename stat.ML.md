# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CAS: A General Algorithm for Online Selective Conformal Prediction with FCR Control](https://arxiv.org/abs/2403.07728) | CAS框架允许在在线选择性预测中控制FCR，通过自适应选择和校准集构造输出符合预测区间 |
| [^2] | [Scaling Laws for Imitation Learning in NetHack.](http://arxiv.org/abs/2307.09423) | 本文研究了在NetHack游戏中的模仿学习，发现通过扩大模型和数据规模可以改进模仿学习的效果，并建立了训练计算最优IL代理人的幂律。 |
| [^3] | [Strong consistency and optimality of spectral clustering in symmetric binary non-uniform Hypergraph Stochastic Block Model.](http://arxiv.org/abs/2306.06845) | 论文提出了非均匀超图随机块模型下谱聚类的强一致性信息理论阈值，并且在该阈值以下给出估计标签的期望“不匹配率”上界。并且，单步谱算法可以在超过该阈值时非常高的概率正确地给定每个顶点的标签。 |
| [^4] | [Toward Falsifying Causal Graphs Using a Permutation-Based Test.](http://arxiv.org/abs/2305.09565) | 本文提出了一种通过构建节点置换基线的新型一致性度量方法，用于验证因果图的正确性并指导其在下游任务中的应用。 |
| [^5] | [On the Expressivity of Persistent Homology in Graph Learning.](http://arxiv.org/abs/2302.09826) | 本文通过在图学习任务中的理论讨论和实证分析，证明了持续同调技术在捕捉具有显著拓扑结构的数据集中的长程图性质方面表现出的高表达性。 |
| [^6] | [Holdouts set for predictive model updating.](http://arxiv.org/abs/2202.06374) | 该论文研究了在复杂环境中如何更新预测风险评分来指导干预。作者提出使用留置集的方式进行更新，通过找到留置集的合适大小可以保证更新后的风险评分性能良好，同时减少留置样本数量。研究结果表明，该方法在总成本增长速度方面具有竞争优势。 |

# 详细

[^1]: CAS: 一种具有FCR控制的在线选择性符合预测的通用算法

    CAS: A General Algorithm for Online Selective Conformal Prediction with FCR Control

    [https://arxiv.org/abs/2403.07728](https://arxiv.org/abs/2403.07728)

    CAS框架允许在在线选择性预测中控制FCR，通过自适应选择和校准集构造输出符合预测区间

    

    我们研究了在线方式下后选择预测推断的问题。为了避免将资源耗费在不重要的单位上，在报告其预测区间之前对当前个体进行初步选择在在线预测任务中是常见且有意义的。由于在线选择导致所选预测区间中存在时间多重性，因此控制实时误覆盖陈述率（FCR）来测量平均误覆盖误差是重要的。我们开发了一个名为CAS（适应性选择后校准）的通用框架，可以包裹任何预测模型和在线选择规则，以输出后选择的预测区间。如果选择了当前个体，我们首先对历史数据进行自适应选择来构建校准集，然后为未观察到的标签输出符合预测区间。我们为校准集提供了可行的构造方式

    arXiv:2403.07728v1 Announce Type: cross  Abstract: We study the problem of post-selection predictive inference in an online fashion. To avoid devoting resources to unimportant units, a preliminary selection of the current individual before reporting its prediction interval is common and meaningful in online predictive tasks. Since the online selection causes a temporal multiplicity in the selected prediction intervals, it is important to control the real-time false coverage-statement rate (FCR) to measure the averaged miscoverage error. We develop a general framework named CAS (Calibration after Adaptive Selection) that can wrap around any prediction model and online selection rule to output post-selection prediction intervals. If the current individual is selected, we first perform an adaptive selection on historical data to construct a calibration set, then output a conformal prediction interval for the unobserved label. We provide tractable constructions for the calibration set for 
    
[^2]: 在NetHack中的模仿学习的规模律

    Scaling Laws for Imitation Learning in NetHack. (arXiv:2307.09423v1 [cs.LG])

    [http://arxiv.org/abs/2307.09423](http://arxiv.org/abs/2307.09423)

    本文研究了在NetHack游戏中的模仿学习，发现通过扩大模型和数据规模可以改进模仿学习的效果，并建立了训练计算最优IL代理人的幂律。

    

    模仿学习 (IL) 是机器学习中最常用的方法之一。然而，虽然强大，但许多研究发现它往往不能完全恢复出潜在的专家行为。然而，这些研究没有深入探究模型和数据规模的扩大在其中的作用。受最近在自然语言处理 (NLP) 领域的工作的启发，在那里“扩大规模”已经导致了越来越有能力的领域特定语言模型 (LLMs)，我们研究了仔细扩大模型和数据规模是否可以在模仿学习的设置中带来类似的改进。为了展示我们的发现，我们将重点放在 NetHack 游戏上，这是一个具有程序生成、随机性、长期依赖性和部分可观测性的具有挑战性的环境。我们发现 IL 的损失和平均回报随着计算预算的变化而平滑变化且强相关，从而在模型大小和样本数量方面为训练计算最优的 IL 代理人的计算预算建立了幂律。我们预测并训练了几个具有 IL 的NetHack代理。

    Imitation Learning (IL) is one of the most widely used methods in machine learning. Yet, while powerful, many works find it is often not able to fully recover the underlying expert behavior. However, none of these works deeply investigate the role of scaling up the model and data size. Inspired by recent work in Natural Language Processing (NLP) where "scaling up" has resulted in increasingly more capable LLMs, we investigate whether carefully scaling up model and data size can bring similar improvements in the imitation learning setting. To demonstrate our findings, we focus on the game of NetHack, a challenging environment featuring procedural generation, stochasticity, long-term dependencies, and partial observability. We find IL loss and mean return scale smoothly with the compute budget and are strongly correlated, resulting in power laws for training compute-optimal IL agents with respect to model size and number of samples. We forecast and train several NetHack agents with IL an
    
[^3]: 对称二元非均匀超图随机块模型中谱聚类的强一致性与最优性

    Strong consistency and optimality of spectral clustering in symmetric binary non-uniform Hypergraph Stochastic Block Model. (arXiv:2306.06845v1 [math.ST])

    [http://arxiv.org/abs/2306.06845](http://arxiv.org/abs/2306.06845)

    论文提出了非均匀超图随机块模型下谱聚类的强一致性信息理论阈值，并且在该阈值以下给出估计标签的期望“不匹配率”上界。并且，单步谱算法可以在超过该阈值时非常高的概率正确地给定每个顶点的标签。

    

    本论文考虑了在非均匀超图随机块模型下，两个等大小的社区（n/2）中的随机超图上的无监督分类问题，其中每个边只依赖于其顶点的标签，边以一定概率独立出现。在这篇论文中，建立了强一致性的信息理论阈值，在该阈值以下，任何算法都有很高概率会误分类至少两个顶点，而特征向量估计量的期望“不匹配率”上界为$n$的阈值的负指数。另一方面，当超过该阈值时，尽管张量收缩引起了信息损失，但单步谱算法仅在给定收缩的邻接矩阵时，即使SDP在某些情况下失败，也可以非常高的概率正确地给定每个顶点分配标签。此外，强一致性可以通过对所有次优聚合信息实现。

    Consider the unsupervised classification problem in random hypergraphs under the non-uniform \emph{Hypergraph Stochastic Block Model} (HSBM) with two equal-sized communities ($n/2$), where each edge appears independently with some probability depending only on the labels of its vertices. In this paper, an \emph{information-theoretical} threshold for strong consistency is established. Below the threshold, every algorithm would misclassify at least two vertices with high probability, and the expected \emph{mismatch ratio} of the eigenvector estimator is upper bounded by $n$ to the power of minus the threshold. On the other hand, when above the threshold, despite the information loss induced by tensor contraction, one-stage spectral algorithms assign every vertex correctly with high probability when only given the contracted adjacency matrix, even if \emph{semidefinite programming} (SDP) fails in some scenarios. Moreover, strong consistency is achievable by aggregating information from al
    
[^4]: 基于置换检验的因果图假设验证方法

    Toward Falsifying Causal Graphs Using a Permutation-Based Test. (arXiv:2305.09565v1 [stat.ML])

    [http://arxiv.org/abs/2305.09565](http://arxiv.org/abs/2305.09565)

    本文提出了一种通过构建节点置换基线的新型一致性度量方法，用于验证因果图的正确性并指导其在下游任务中的应用。

    

    理解系统变量之间的因果关系对于解释和控制其行为至关重要。但是，从观察数据中推断因果图需要很多不总是现实的强假设。对于领域专家来说，很难表达因果图。因此，在将因果图用于下游任务之前，定量评估因果图的优劣的度量提供了有用的检查。现有的度量提供了一个绝对数量的因果图与观察数据之间的不一致性，而没有基础线，从业人员需要回答有多少这样的不一致性是可接受或预期的这一难题。在这里，我们提出了一种新的一致性度量方法，通过构建节点置换的替代基线。通过将不一致性的数量与替代基线上的数量进行比较，我们得出了一个可以解释的度量，捕捉有向无环图是否显著适合。

    Understanding the causal relationships among the variables of a system is paramount to explain and control its behaviour. Inferring the causal graph from observational data without interventions, however, requires a lot of strong assumptions that are not always realistic. Even for domain experts it can be challenging to express the causal graph. Therefore, metrics that quantitatively assess the goodness of a causal graph provide helpful checks before using it in downstream tasks. Existing metrics provide an absolute number of inconsistencies between the graph and the observed data, and without a baseline, practitioners are left to answer the hard question of how many such inconsistencies are acceptable or expected. Here, we propose a novel consistency metric by constructing a surrogate baseline through node permutations. By comparing the number of inconsistencies with those on the surrogate baseline, we derive an interpretable metric that captures whether the DAG fits significantly bet
    
[^5]: 持续同调在图学习中的表达性

    On the Expressivity of Persistent Homology in Graph Learning. (arXiv:2302.09826v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.09826](http://arxiv.org/abs/2302.09826)

    本文通过在图学习任务中的理论讨论和实证分析，证明了持续同调技术在捕捉具有显著拓扑结构的数据集中的长程图性质方面表现出的高表达性。

    

    近来，计算拓扑学中的一项技术，持续同调展现出在图分类方面强大的实证性能。它能够通过高阶拓扑特征——如任意长度的环——以及多尺度拓扑描述符捕捉长程图性质，从而提高对具有显著拓扑结构的数据集——如分子——的预测性能。与此同时，持续同调的理论性质在这个背景下尚未得到正式评估。本文旨在通过提供持续同调在图中的简要介绍以及对其在图学习任务中的表达性进行理论讨论和实证分析，弥合计算拓扑学和图机器学习之间的差距。

    Persistent homology, a technique from computational topology, has recently shown strong empirical performance in the context of graph classification. Being able to capture long range graph properties via higher-order topological features, such as cycles of arbitrary length, in combination with multi-scale topological descriptors, has improved predictive performance for data sets with prominent topological structures, such as molecules. At the same time, the theoretical properties of persistent homology have not been formally assessed in this context. This paper intends to bridge the gap between computational topology and graph machine learning by providing a brief introduction to persistent homology in the context of graphs, as well as a theoretical discussion and empirical analysis of its expressivity for graph learning tasks.
    
[^6]: 针对预测模型更新的留置集

    Holdouts set for predictive model updating. (arXiv:2202.06374v4 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2202.06374](http://arxiv.org/abs/2202.06374)

    该论文研究了在复杂环境中如何更新预测风险评分来指导干预。作者提出使用留置集的方式进行更新，通过找到留置集的合适大小可以保证更新后的风险评分性能良好，同时减少留置样本数量。研究结果表明，该方法在总成本增长速度方面具有竞争优势。

    

    在复杂的环境中，如医疗保健领域，预测风险评分在指导干预方面起着越来越重要的作用。然而，直接更新用于指导干预的风险评分可能导致偏差风险估计。为了解决这个问题，我们提出使用“留置集”来进行更新-留置集是一个不接受风险评分指导干预的人群的子集。在留置集的大小上取得平衡是关键，以确保更新后的风险评分性能良好，同时最大限度地减少留置样本的数量。我们证明了这种方法使得总成本可以以$O\left(N^{2/3}\right)$的速度增长，其中$N$是人口规模，并且认为在一般情况下没有竞争性的替代方法。通过定义适当的损失函数，我们描述了一些条件，可以很容易地确定最佳留置集大小（OHS），并引入参数化和半参数化算法来估计OHS，并展示了其在最新风险评分中的应用。

    In complex settings, such as healthcare, predictive risk scores play an increasingly crucial role in guiding interventions. However, directly updating risk scores used to guide intervention can lead to biased risk estimates. To address this, we propose updating using a `holdout set' - a subset of the population that does not receive interventions guided by the risk score. Striking a balance in the size of the holdout set is essential, to ensure good performance of the updated risk score whilst minimising the number of held out samples. We prove that this approach enables total costs to grow at a rate $O\left(N^{2/3}\right)$ for a population of size $N$, and argue that in general circumstances there is no competitive alternative. By defining an appropriate loss function, we describe conditions under which an optimal holdout size (OHS) can be readily identified, and introduce parametric and semi-parametric algorithms for OHS estimation, demonstrating their use on a recent risk score for 
    

