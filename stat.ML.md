# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Network Representations with Disentangled Graph Auto-Encoder](https://rss.arxiv.org/abs/2402.01143) | 本文介绍了解缠离散图自编码器(DGA)和解缠变分图自编码器(DVGA)的方法，利用生成模型来学习解缠表示。 |
| [^2] | [Discrete Latent Graph Generative Modeling with Diffusion Bridges](https://arxiv.org/abs/2403.16883) | GLAD是一个在离散潜在空间上操作的图生成模型，通过适应扩散桥结构学习其离散潜在空间的先验，避免了依赖于原始数据空间的分解，在图生成任务中表现出优越性。 |
| [^3] | [Bipartite Graph Variational Auto-Encoder with Fair Latent Representation to Account for Sampling Bias in Ecological Networks](https://arxiv.org/abs/2403.02011) | 本研究提出了一种公平潜在表示的二分图变分自动编码器方法，以解决生态网络中的抽样偏差问题，通过在损失函数中引入额外的HSIC惩罚项，确保了潜在空间结构与连续变量的独立性。 |
| [^4] | [Rao-Blackwellising Bayesian Causal Inference](https://arxiv.org/abs/2402.14781) | 本文结合顺序化的MCMC结构学习技术和梯度图学习的最新进展，构建了一个有效的贝叶斯因果推断框架，将因果结构推断问题分解为变量拓扑顺序推断和变量父节点集合推断，同时使用高斯过程进行因果机制建模实现精确边缘化，引入了一个Rao-Blackwell化方案。 |
| [^5] | [Interpretable Multi-Source Data Fusion Through Latent Variable Gaussian Process](https://arxiv.org/abs/2402.04146) | 这篇论文提出了一种基于潜变量高斯过程的多源数据融合框架，用于解决多个数据源之间质量和全面性差异给系统优化带来的问题。 |
| [^6] | [Multi-Armed Bandits with Interference](https://arxiv.org/abs/2402.01845) | 这篇论文研究了在在线平台中与干扰进行的实验。在多臂赌博机问题中，学习者分配不同的臂给每个实验单元，根据单元之间的空间距离和对手选择的匹配函数来决定每个单元在每轮的回报。研究发现，转换政策能够实现最佳的预期遗憾，但任何转换政策都会遭受一定的遗憾现象。 |
| [^7] | [Frequentist Guarantees of Distributed (Non)-Bayesian Inference](https://arxiv.org/abs/2311.08214) | 本文针对通过通信网络连接的代理之间的分布式(非)贝叶斯推断问题建立了频率特性，探讨了在适当假设下分布式贝叶斯推断在参数效率和不确定性量化方面的表现，以及通信图设计和大小对后验收缩率的影响。 |
| [^8] | [Gradients Look Alike: Sensitivity is Often Overestimated in DP-SGD.](http://arxiv.org/abs/2307.00310) | 本文开发了一种新的DP-SGD分析方法，可以在训练过程中对许多数据点的隐私泄漏进行更准确的评估。 |
| [^9] | [Designing Decision Support Systems Using Counterfactual Prediction Sets.](http://arxiv.org/abs/2306.03928) | 本文提出了一种基于反事实预测集的决策支持系统设计方法，不同于传统的单一标签预测，它使用符合预测器构建预测集，并引导人类专家从中选择标签值。 |
| [^10] | [When Do Neural Nets Outperform Boosted Trees on Tabular Data?.](http://arxiv.org/abs/2305.02997) | 这项研究通过对176个数据集的比较分析发现，在许多数据集中，GBDT和NN之间的性能差异可以忽略不计，或者GBDT的轻微超参数调整比选择最佳算法更重要。此外，研究人员对965个元特征进行了分析，发现GBDT在高维稀疏数据上表现更好。 |

# 详细

[^1]: 用解缠离散图自编码器学习网络表示

    Learning Network Representations with Disentangled Graph Auto-Encoder

    [https://rss.arxiv.org/abs/2402.01143](https://rss.arxiv.org/abs/2402.01143)

    本文介绍了解缠离散图自编码器(DGA)和解缠变分图自编码器(DVGA)的方法，利用生成模型来学习解缠表示。

    

    (变分)图自编码器广泛用于学习图结构化数据的表示。然而，现实世界图的形成是一个由潜在因素影响的复杂和异质的过程。现有的编码器基本上是整体的，忽视了潜在因素的纠缠。这不仅使得图分析任务不太有效，而且使得理解和解释这些表示变得更加困难。用(变分)图自编码器学习解缠的图表示面临着重要挑战，在现有文献中尚未得到充分探索。在本文中，我们介绍了解缠离散图自编码器(DGA)和解缠变分图自编码器(DVGA)的方法，利用生成模型来学习解缠表示。具体地，我们首先设计了一个解缠的图卷积网络，使用多通道消息传递层作为编码器，聚合与每个节点相关的信息。

    The (variational) graph auto-encoder is extensively employed for learning representations of graph-structured data. However, the formation of real-world graphs is a complex and heterogeneous process influenced by latent factors. Existing encoders are fundamentally holistic, neglecting the entanglement of latent factors. This not only makes graph analysis tasks less effective but also makes it harder to understand and explain the representations. Learning disentangled graph representations with (variational) graph auto-encoder poses significant challenges, and remains largely unexplored in the existing literature. In this article, we introduce the Disentangled Graph Auto-Encoder (DGA) and Disentangled Variational Graph Auto-Encoder (DVGA), approaches that leverage generative models to learn disentangled representations. Specifically, we first design a disentangled graph convolutional network with multi-channel message-passing layers, as the encoder aggregating information related to eac
    
[^2]: 带扩散桥的离散潜在图生成建模

    Discrete Latent Graph Generative Modeling with Diffusion Bridges

    [https://arxiv.org/abs/2403.16883](https://arxiv.org/abs/2403.16883)

    GLAD是一个在离散潜在空间上操作的图生成模型，通过适应扩散桥结构学习其离散潜在空间的先验，避免了依赖于原始数据空间的分解，在图生成任务中表现出优越性。

    

    学习潜在空间中的图生成模型相比于在原始数据空间上操作的模型受到较少关注，迄今表现出的性能乏善可陈。我们提出了GLAD，一个潜在空间图生成模型。与大多数先前的潜在空间图生成模型不同，GLAD在保留图结构的离散性质方面运行，无需进行诸如潜在空间连续性等不自然的假设。我们通过将扩散桥调整到其结构，来学习我们离散潜在空间的先验。通过在适当构建的潜在空间上操作，我们避免依赖于常用于在原始数据空间操作的模型中的分解。我们在一系列图基准数据集上进行实验，明显展示了离散潜在空间的优越性，并取得了最先进的图生成性能，使GLA

    arXiv:2403.16883v1 Announce Type: new  Abstract: Learning graph generative models over latent spaces has received less attention compared to models that operate on the original data space and has so far demonstrated lacklustre performance. We present GLAD a latent space graph generative model. Unlike most previous latent space graph generative models, GLAD operates on a discrete latent space that preserves to a significant extent the discrete nature of the graph structures making no unnatural assumptions such as latent space continuity. We learn the prior of our discrete latent space by adapting diffusion bridges to its structure. By operating over an appropriately constructed latent space we avoid relying on decompositions that are often used in models that operate in the original data space. We present experiments on a series of graph benchmark datasets which clearly show the superiority of the discrete latent space and obtain state of the art graph generative performance, making GLA
    
[^3]: 公平潜在表示的二分图变分自动编码器，以解决生态网络中的抽样偏差问题

    Bipartite Graph Variational Auto-Encoder with Fair Latent Representation to Account for Sampling Bias in Ecological Networks

    [https://arxiv.org/abs/2403.02011](https://arxiv.org/abs/2403.02011)

    本研究提出了一种公平潜在表示的二分图变分自动编码器方法，以解决生态网络中的抽样偏差问题，通过在损失函数中引入额外的HSIC惩罚项，确保了潜在空间结构与连续变量的独立性。

    

    我们提出一种方法，使用图嵌入来表示二分网络，以解决研究生态网络所面临的挑战，比如连接植物和传粉者等网络，需考虑许多协变量，尤其要控制抽样偏差。我们将变分图自动编码器方法调整为二分情况，从而能够在潜在空间中生成嵌入，其中两组节点的位置基于它们的连接概率。我们将在社会学中常考虑的公平性框架转化为生态学中的抽样偏差问题。通过在损失函数中添加Hilbert-Schmidt独立准则（HSIC）作为额外惩罚项，我们确保潜在空间结构与连续变量（与抽样过程相关）无关。最后，我们展示了我们的方法如何改变我们对生态网络的理解。

    arXiv:2403.02011v1 Announce Type: cross  Abstract: We propose a method to represent bipartite networks using graph embeddings tailored to tackle the challenges of studying ecological networks, such as the ones linking plants and pollinators, where many covariates need to be accounted for, in particular to control for sampling bias. We adapt the variational graph auto-encoder approach to the bipartite case, which enables us to generate embeddings in a latent space where the two sets of nodes are positioned based on their probability of connection. We translate the fairness framework commonly considered in sociology in order to address sampling bias in ecology. By incorporating the Hilbert-Schmidt independence criterion (HSIC) as an additional penalty term in the loss we optimize, we ensure that the structure of the latent space is independent of continuous variables, which are related to the sampling process. Finally, we show how our approach can change our understanding of ecological n
    
[^4]: Rao-Blackwellising Bayesian Causal Inference

    Rao-Blackwellising Bayesian Causal Inference

    [https://arxiv.org/abs/2402.14781](https://arxiv.org/abs/2402.14781)

    本文结合顺序化的MCMC结构学习技术和梯度图学习的最新进展，构建了一个有效的贝叶斯因果推断框架，将因果结构推断问题分解为变量拓扑顺序推断和变量父节点集合推断，同时使用高斯过程进行因果机制建模实现精确边缘化，引入了一个Rao-Blackwell化方案。

    

    贝叶斯因果推断，即推断用于下游因果推理任务中的因果模型的后验概率，构成了一个在文献中鲜有探讨的难解的计算推断问题。本文将基于顺序的MCMC结构学习技术与最近梯度图学习的进展相结合，构建了一个有效的贝叶斯因果推断框架。具体而言，我们将推断因果结构的问题分解为(i)推断变量之间的拓扑顺序以及(ii)推断每个变量的父节点集合。当限制每个变量的父节点数量时，我们可以在多项式时间内完全边缘化父节点集合。我们进一步使用高斯过程来建模未知的因果机制，从而允许其精确边缘化。这引入了一个Rao-Blackwell化方案，其中除了因果顺序之外，模型中的所有组件都被消除。

    arXiv:2402.14781v1 Announce Type: cross  Abstract: Bayesian causal inference, i.e., inferring a posterior over causal models for the use in downstream causal reasoning tasks, poses a hard computational inference problem that is little explored in literature. In this work, we combine techniques from order-based MCMC structure learning with recent advances in gradient-based graph learning into an effective Bayesian causal inference framework. Specifically, we decompose the problem of inferring the causal structure into (i) inferring a topological order over variables and (ii) inferring the parent sets for each variable. When limiting the number of parents per variable, we can exactly marginalise over the parent sets in polynomial time. We further use Gaussian processes to model the unknown causal mechanisms, which also allows their exact marginalisation. This introduces a Rao-Blackwellization scheme, where all components are eliminated from the model, except for the causal order, for whi
    
[^5]: 可解释的多源数据融合通过潜变量高斯过程

    Interpretable Multi-Source Data Fusion Through Latent Variable Gaussian Process

    [https://arxiv.org/abs/2402.04146](https://arxiv.org/abs/2402.04146)

    这篇论文提出了一种基于潜变量高斯过程的多源数据融合框架，用于解决多个数据源之间质量和全面性差异给系统优化带来的问题。

    

    随着人工智能（AI）和机器学习（ML）的出现，各个科学和工程领域已经利用数据驱动的替代模型来建模来自大量信息源（数据）的复杂系统。这种增加导致了开发出用于执行特定功能的优越系统所需的成本和时间的显著降低。这样的替代模型往往广泛地融合多个数据来源，可能是发表的论文、专利、开放资源库或其他资源。然而，对于已知和未知的信息来源的基础物理参数的质量和全面性的差异，可能对系统优化过程产生后续影响，却没有得到充分的关注。为了解决这个问题，提出了一种基于潜变量高斯过程（LVGP）的多源数据融合框架。

    With the advent of artificial intelligence (AI) and machine learning (ML), various domains of science and engineering communites has leveraged data-driven surrogates to model complex systems from numerous sources of information (data). The proliferation has led to significant reduction in cost and time involved in development of superior systems designed to perform specific functionalities. A high proposition of such surrogates are built extensively fusing multiple sources of data, may it be published papers, patents, open repositories, or other resources. However, not much attention has been paid to the differences in quality and comprehensiveness of the known and unknown underlying physical parameters of the information sources that could have downstream implications during system optimization. Towards resolving this issue, a multi-source data fusion framework based on Latent Variable Gaussian Process (LVGP) is proposed. The individual data sources are tagged as a characteristic cate
    
[^6]: 具有干扰的多臂赌博机问题

    Multi-Armed Bandits with Interference

    [https://arxiv.org/abs/2402.01845](https://arxiv.org/abs/2402.01845)

    这篇论文研究了在在线平台中与干扰进行的实验。在多臂赌博机问题中，学习者分配不同的臂给每个实验单元，根据单元之间的空间距离和对手选择的匹配函数来决定每个单元在每轮的回报。研究发现，转换政策能够实现最佳的预期遗憾，但任何转换政策都会遭受一定的遗憾现象。

    

    在当代在线平台上，与干扰进行实验是一个重大挑战。以往有关干扰实验的研究集中在政策的最终输出上，而对于累计性能则了解不足。为了填补这一空白，我们引入了“具有干扰的多臂赌博机”（MABI）问题，在时间段为T轮的情况下，学习者为N个实验单元中的每个分配一个臂。每个单元在每一轮的回报取决于“所有”单元的治疗方式，而单元之间的空间距离会导致单元的影响力逐渐衰减。此外，我们使用了一个通用设置，其中回报函数由对手选择，并且在轮次和单元之间可以任意变化。我们首先证明了转换政策能够对最佳固定臂政策实现最优的“预期”遗憾，遗憾值为$O(\sqrt T)$。然而，任何一个转换政策的遗憾（作为一个随机变量）都会遭受一定的遗憾现象。

    Experimentation with interference poses a significant challenge in contemporary online platforms. Prior research on experimentation with interference has concentrated on the final output of a policy. The cumulative performance, while equally crucial, is less well understood. To address this gap, we introduce the problem of {\em Multi-armed Bandits with Interference} (MABI), where the learner assigns an arm to each of $N$ experimental units over a time horizon of $T$ rounds. The reward of each unit in each round depends on the treatments of {\em all} units, where the influence of a unit decays in the spatial distance between units. Furthermore, we employ a general setup wherein the reward functions are chosen by an adversary and may vary arbitrarily across rounds and units. We first show that switchback policies achieve an optimal {\em expected} regret $\tilde O(\sqrt T)$ against the best fixed-arm policy. Nonetheless, the regret (as a random variable) for any switchback policy suffers 
    
[^7]: 分布式(非)贝叶斯推断的频率保证

    Frequentist Guarantees of Distributed (Non)-Bayesian Inference

    [https://arxiv.org/abs/2311.08214](https://arxiv.org/abs/2311.08214)

    本文针对通过通信网络连接的代理之间的分布式(非)贝叶斯推断问题建立了频率特性，探讨了在适当假设下分布式贝叶斯推断在参数效率和不确定性量化方面的表现，以及通信图设计和大小对后验收缩率的影响。

    

    受分析大型分散数据集的需求推动，分布式贝叶斯推断已成为跨多个领域（包括统计学、电气工程和经济学）的关键研究领域。本文针对通过通信网络连接的代理之间的分布式(非)贝叶斯推断问题建立了频率特性，如后验一致性、渐近正态性和后验收缩率。我们的结果表明，在通信图上的适当假设下，分布式贝叶斯推断保留了参数效率，同时在不确定性量化方面增强了鲁棒性。我们还通过研究设计和通信图的大小如何影响后验收缩率来探讨了统计效率和通信效率之间的权衡。此外，我们将我们的分析扩展到时变图，并将结果应用于指数f

    arXiv:2311.08214v2 Announce Type: replace-cross  Abstract: Motivated by the need to analyze large, decentralized datasets, distributed Bayesian inference has become a critical research area across multiple fields, including statistics, electrical engineering, and economics. This paper establishes Frequentist properties, such as posterior consistency, asymptotic normality, and posterior contraction rates, for the distributed (non-)Bayes Inference problem among agents connected via a communication network. Our results show that, under appropriate assumptions on the communication graph, distributed Bayesian inference retains parametric efficiency while enhancing robustness in uncertainty quantification. We also explore the trade-off between statistical efficiency and communication efficiency by examining how the design and size of the communication graph impact the posterior contraction rate. Furthermore, We extend our analysis to time-varying graphs and apply our results to exponential f
    
[^8]: 梯度相似：敏感度经常被过高估计在DP-SGD中

    Gradients Look Alike: Sensitivity is Often Overestimated in DP-SGD. (arXiv:2307.00310v1 [cs.LG])

    [http://arxiv.org/abs/2307.00310](http://arxiv.org/abs/2307.00310)

    本文开发了一种新的DP-SGD分析方法，可以在训练过程中对许多数据点的隐私泄漏进行更准确的评估。

    

    差分隐私随机梯度下降（DP-SGD）是私有深度学习的标准算法。虽然已知其隐私分析在最坏情况下是紧密的，但是一些实证结果表明，在常见的基准数据集上训练时，所得到的模型对许多数据点的隐私泄漏显著减少。在本文中，我们为DP-SGD开发了一种新的分析方法，捕捉到在数据集中具有相似邻居的点享受更好隐私性的直觉。形式上来说，这是通过修改从训练数据集计算得到的模型更新的每步隐私性分析来实现的。我们进一步开发了一个新的组合定理，以有效地利用这个新的每步分析来推理整个训练过程。总而言之，我们的评估结果表明，这种新颖的DP-SGD分析使我们能够正式地显示DP-SGD对许多数据点的隐私泄漏显著减少。

    Differentially private stochastic gradient descent (DP-SGD) is the canonical algorithm for private deep learning. While it is known that its privacy analysis is tight in the worst-case, several empirical results suggest that when training on common benchmark datasets, the models obtained leak significantly less privacy for many datapoints. In this paper, we develop a new analysis for DP-SGD that captures the intuition that points with similar neighbors in the dataset enjoy better privacy than outliers. Formally, this is done by modifying the per-step privacy analysis of DP-SGD to introduce a dependence on the distribution of model updates computed from a training dataset. We further develop a new composition theorem to effectively use this new per-step analysis to reason about an entire training run. Put all together, our evaluation shows that this novel DP-SGD analysis allows us to now formally show that DP-SGD leaks significantly less privacy for many datapoints. In particular, we ob
    
[^9]: 使用反事实预测集设计决策支持系统

    Designing Decision Support Systems Using Counterfactual Prediction Sets. (arXiv:2306.03928v1 [cs.LG])

    [http://arxiv.org/abs/2306.03928](http://arxiv.org/abs/2306.03928)

    本文提出了一种基于反事实预测集的决策支持系统设计方法，不同于传统的单一标签预测，它使用符合预测器构建预测集，并引导人类专家从中选择标签值。

    

    分类任务的决策支持系统通常被设计用于预测地面实况标签的值。然而，由于它们的预测并不完美，这些系统还需要让人类专家了解何时以及如何使用这些预测来更新自己的预测。不幸的是，这被证明是具有挑战性的。最近有人认为，另一种类型的决策支持系统可能会避开这个挑战。这些系统不是提供单个标签预测，而是使用符合预测器构建一组标签预测值，即预测集，并强制要求专家从预测集中预测一个标签值。然而，这些系统的设计和评估迄今仍依赖于样式化的专家模型，这引发了人们对它们的承诺的质疑。本文从在线学习的角度重新审视了这种系统的设计，并开发了一种不需要。

    Decision support systems for classification tasks are predominantly designed to predict the value of the ground truth labels. However, since their predictions are not perfect, these systems also need to make human experts understand when and how to use these predictions to update their own predictions. Unfortunately, this has been proven challenging. In this context, it has been recently argued that an alternative type of decision support systems may circumvent this challenge. Rather than providing a single label prediction, these systems provide a set of label prediction values constructed using a conformal predictor, namely a prediction set, and forcefully ask experts to predict a label value from the prediction set. However, the design and evaluation of these systems have so far relied on stylized expert models, questioning their promise. In this paper, we revisit the design of this type of systems from the perspective of online learning and develop a methodology that does not requi
    
[^10]: 神经网络何时在表格数据上胜过增强树？

    When Do Neural Nets Outperform Boosted Trees on Tabular Data?. (arXiv:2305.02997v1 [cs.LG])

    [http://arxiv.org/abs/2305.02997](http://arxiv.org/abs/2305.02997)

    这项研究通过对176个数据集的比较分析发现，在许多数据集中，GBDT和NN之间的性能差异可以忽略不计，或者GBDT的轻微超参数调整比选择最佳算法更重要。此外，研究人员对965个元特征进行了分析，发现GBDT在高维稀疏数据上表现更好。

    

    表格数据是机器学习中最常用的数据类型之一。尽管神经网络（NN）在表格数据上取得了最近的进展，但人们仍在积极讨论NN是否通常优于梯度提升决策树（GBDT）在表格数据上的表现，一些最近的工作要么认为GBDT在表格数据上一贯优于NN，要么认为NN优于GBDT。在这项工作中，我们退一步问：'这重要吗？'我们通过对176个数据集比较19种算法，进行了迄今为止最大的表格数据分析，并发现'NN vs. GBDT'争论被过分强调：令人惊讶的是，在相当多的数据集中，GBDT和NN之间的性能差异要么可以忽略不计，要么GBDT的轻微超参数调整比选择最佳算法更重要。接下来，我们分析了965个元特征，以确定数据集的哪些特性使NN或GBDT更适合表现良好。例如，我们发现GBDT要比NN在高维稀疏数据上表现更好。

    Tabular data is one of the most commonly used types of data in machine learning. Despite recent advances in neural nets (NNs) for tabular data, there is still an active discussion on whether or not NNs generally outperform gradient-boosted decision trees (GBDTs) on tabular data, with several recent works arguing either that GBDTs consistently outperform NNs on tabular data, or vice versa. In this work, we take a step back and ask, 'does it matter?' We conduct the largest tabular data analysis to date, by comparing 19 algorithms across 176 datasets, and we find that the 'NN vs. GBDT' debate is overemphasized: for a surprisingly high number of datasets, either the performance difference between GBDTs and NNs is negligible, or light hyperparameter tuning on a GBDT is more important than selecting the best algorithm. Next, we analyze 965 metafeatures to determine what properties of a dataset make NNs or GBDTs better-suited to perform well. For example, we find that GBDTs are much better th
    

