# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PARMESAN: Parameter-Free Memory Search and Transduction for Dense Prediction Tasks](https://arxiv.org/abs/2403.11743) | 通过引入转导的概念，提出了PARMESAN，一种用于解决密集预测任务的无参数内存搜索和转导方法，实现了灵活性和无需连续训练的学习。 |
| [^2] | [Hessian-Free Laplace in Bayesian Deep Learning](https://arxiv.org/abs/2403.10671) | 提出了一种无Hessian计算和求逆的Hessian-Free Laplace近似框架，通过对数后验和网络预测的曲率来估计后验的方差。 |
| [^3] | [Towards Spatially-Lucid AI Classification in Non-Euclidean Space: An Application for MxIF Oncology Data](https://arxiv.org/abs/2402.14974) | 发展了一个使用空间集合框架的分类器，可以根据点的排列在非欧几里得空间中区分两个类别，对于肿瘤学等应用具有重要意义 |
| [^4] | [Rao-Blackwellising Bayesian Causal Inference](https://arxiv.org/abs/2402.14781) | 本文结合顺序化的MCMC结构学习技术和梯度图学习的最新进展，构建了一个有效的贝叶斯因果推断框架，将因果结构推断问题分解为变量拓扑顺序推断和变量父节点集合推断，同时使用高斯过程进行因果机制建模实现精确边缘化，引入了一个Rao-Blackwell化方案。 |
| [^5] | [Learning by Doing: An Online Causal Reinforcement Learning Framework with Causal-Aware Policy](https://arxiv.org/abs/2402.04869) | 本文提出了一个在线因果强化学习框架，其中通过明确建模状态的生成过程和使用因果结构进行策略引导，以帮助强化学习代理的决策可解释性。该框架具有理论性能保证，并在探索和开发过程中使用干预进行因果结构学习。 |
| [^6] | [Efficient Neural Network Approaches for Conditional Optimal Transport with Applications in Bayesian Inference.](http://arxiv.org/abs/2310.16975) | 提出了两种神经网络方法来逼近静态和动态条件最优传输问题的解，实现了对条件概率分布的采样和密度估计，适用于贝叶斯推断。算法利用神经网络参数化传输映射以提高可扩展性。 |
| [^7] | [Persistent Homology of the Multiscale Clustering Filtration.](http://arxiv.org/abs/2305.04281) | 该论文引入了一种多尺度聚类过滤方法（MCF），用于描述不同尺度下的数据聚类，其中的持久同调可测量分区序列的层次关系和聚类分配冲突的出现和解决。 |

# 详细

[^1]: PARMESAN: 用于密集预测任务的无参数内存搜索与转导

    PARMESAN: Parameter-Free Memory Search and Transduction for Dense Prediction Tasks

    [https://arxiv.org/abs/2403.11743](https://arxiv.org/abs/2403.11743)

    通过引入转导的概念，提出了PARMESAN，一种用于解决密集预测任务的无参数内存搜索和转导方法，实现了灵活性和无需连续训练的学习。

    

    在这项工作中，我们通过转导推理来解决深度学习中的灵活性问题。我们提出了PARMESAN（无参数内存搜索与转导），这是一种可扩展的转导方法，利用内存模块来解决密集预测任务。在推断过程中，内存中的隐藏表示被搜索以找到相应的示例。与其他方法不同，PARMESAN通过修改内存内容学习，而无需进行任何连续训练或微调可学习参数。我们的方法与常用的神经结构兼容。

    arXiv:2403.11743v1 Announce Type: new  Abstract: In this work we address flexibility in deep learning by means of transductive reasoning. For adaptation to new tasks or new data, existing methods typically involve tuning of learnable parameters or even complete re-training from scratch, rendering such approaches unflexible in practice. We argue that the notion of separating computation from memory by the means of transduction can act as a stepping stone for solving these issues. We therefore propose PARMESAN (parameter-free memory search and transduction), a scalable transduction method which leverages a memory module for solving dense prediction tasks. At inference, hidden representations in memory are being searched to find corresponding examples. In contrast to other methods, PARMESAN learns without the requirement for any continuous training or fine-tuning of learnable parameters simply by modifying the memory content. Our method is compatible with commonly used neural architecture
    
[^2]: Bayesian深度学习中的无Hessian-Laplace

    Hessian-Free Laplace in Bayesian Deep Learning

    [https://arxiv.org/abs/2403.10671](https://arxiv.org/abs/2403.10671)

    提出了一种无Hessian计算和求逆的Hessian-Free Laplace近似框架，通过对数后验和网络预测的曲率来估计后验的方差。

    

    贝叶斯后验的Laplace近似（LA）是以最大后验估计为中心的高斯分布。它在贝叶斯深度学习中的吸引力源于能够在标准网络参数优化之后量化不确定性（即事后），从近似后验中抽样的便利性以及模型证据的解析形式。然而，LA的一个重要计算瓶颈是必须计算和求逆对数后验的Hessian矩阵。Hessian可以以多种方式近似，质量与网络、数据集和推断任务等多个因素有关。在本文中，我们提出了一个绕过Hessian计算和求逆的替代框架。无Hessian-Laplace（HFL）近似使用对数后验和网络预测的曲率来估计其方差。只需要两个点估计：最大后验估计和等价的曲率方向。

    arXiv:2403.10671v1 Announce Type: cross  Abstract: The Laplace approximation (LA) of the Bayesian posterior is a Gaussian distribution centered at the maximum a posteriori estimate. Its appeal in Bayesian deep learning stems from the ability to quantify uncertainty post-hoc (i.e., after standard network parameter optimization), the ease of sampling from the approximate posterior, and the analytic form of model evidence. However, an important computational bottleneck of LA is the necessary step of calculating and inverting the Hessian matrix of the log posterior. The Hessian may be approximated in a variety of ways, with quality varying with a number of factors including the network, dataset, and inference task. In this paper, we propose an alternative framework that sidesteps Hessian calculation and inversion. The Hessian-free Laplace (HFL) approximation uses curvature of both the log posterior and network prediction to estimate its variance. Only two point estimates are needed: the st
    
[^3]: 在非欧几里得空间中实现空间透明的AI分类：MxIF肿瘤数据应用

    Towards Spatially-Lucid AI Classification in Non-Euclidean Space: An Application for MxIF Oncology Data

    [https://arxiv.org/abs/2402.14974](https://arxiv.org/abs/2402.14974)

    发展了一个使用空间集合框架的分类器，可以根据点的排列在非欧几里得空间中区分两个类别，对于肿瘤学等应用具有重要意义

    

    针对来自不同地点类型的多类别点集，我们的目标是开发一个空间透明的分类器，可以根据点的排列区分两个类别。这个问题对于许多应用非常重要，比如肿瘤学，用于分析免疫-肿瘤关系和设计新的免疫治疗方法。这个问题具有挑战性，因为需要考虑空间变化和可解释性需求。以前提出的技术要求密集的训练数据，或者在处理单个地点类型内的显著空间变异性方面能力有限。最重要的是，这些深度神经网络（DNN）方法没有设计用于在非欧几里得空间中工作，特别是点集。现有的非欧几里得DNN方法局限于一刀切的方法。我们探索了一种空间集合框架，明确使用不同的训练策略，包括加权距离学习率和空间域自适应。

    arXiv:2402.14974v1 Announce Type: cross  Abstract: Given multi-category point sets from different place-types, our goal is to develop a spatially-lucid classifier that can distinguish between two classes based on the arrangements of their points. This problem is important for many applications, such as oncology, for analyzing immune-tumor relationships and designing new immunotherapies. It is challenging due to spatial variability and interpretability needs. Previously proposed techniques require dense training data or have limited ability to handle significant spatial variability within a single place-type. Most importantly, these deep neural network (DNN) approaches are not designed to work in non-Euclidean space, particularly point sets. Existing non-Euclidean DNN methods are limited to one-size-fits-all approaches. We explore a spatial ensemble framework that explicitly uses different training strategies, including weighted-distance learning rate and spatial domain adaptation, on v
    
[^4]: Rao-Blackwellising Bayesian Causal Inference

    Rao-Blackwellising Bayesian Causal Inference

    [https://arxiv.org/abs/2402.14781](https://arxiv.org/abs/2402.14781)

    本文结合顺序化的MCMC结构学习技术和梯度图学习的最新进展，构建了一个有效的贝叶斯因果推断框架，将因果结构推断问题分解为变量拓扑顺序推断和变量父节点集合推断，同时使用高斯过程进行因果机制建模实现精确边缘化，引入了一个Rao-Blackwell化方案。

    

    贝叶斯因果推断，即推断用于下游因果推理任务中的因果模型的后验概率，构成了一个在文献中鲜有探讨的难解的计算推断问题。本文将基于顺序的MCMC结构学习技术与最近梯度图学习的进展相结合，构建了一个有效的贝叶斯因果推断框架。具体而言，我们将推断因果结构的问题分解为(i)推断变量之间的拓扑顺序以及(ii)推断每个变量的父节点集合。当限制每个变量的父节点数量时，我们可以在多项式时间内完全边缘化父节点集合。我们进一步使用高斯过程来建模未知的因果机制，从而允许其精确边缘化。这引入了一个Rao-Blackwell化方案，其中除了因果顺序之外，模型中的所有组件都被消除。

    arXiv:2402.14781v1 Announce Type: cross  Abstract: Bayesian causal inference, i.e., inferring a posterior over causal models for the use in downstream causal reasoning tasks, poses a hard computational inference problem that is little explored in literature. In this work, we combine techniques from order-based MCMC structure learning with recent advances in gradient-based graph learning into an effective Bayesian causal inference framework. Specifically, we decompose the problem of inferring the causal structure into (i) inferring a topological order over variables and (ii) inferring the parent sets for each variable. When limiting the number of parents per variable, we can exactly marginalise over the parent sets in polynomial time. We further use Gaussian processes to model the unknown causal mechanisms, which also allows their exact marginalisation. This introduces a Rao-Blackwellization scheme, where all components are eliminated from the model, except for the causal order, for whi
    
[^5]: 学习通过实践：具有因果感知的在线因果强化学习框架

    Learning by Doing: An Online Causal Reinforcement Learning Framework with Causal-Aware Policy

    [https://arxiv.org/abs/2402.04869](https://arxiv.org/abs/2402.04869)

    本文提出了一个在线因果强化学习框架，其中通过明确建模状态的生成过程和使用因果结构进行策略引导，以帮助强化学习代理的决策可解释性。该框架具有理论性能保证，并在探索和开发过程中使用干预进行因果结构学习。

    

    因果知识作为人类智能中直观认知和推理解决方案的关键组成部分，为强化学习（RL）代理的可解释性决策提供了巨大的潜力，通过帮助减少搜索空间。然而，发现和整合因果关系进入RL仍存在相当大的差距，这阻碍了因果关系RL的快速发展。在本文中，我们考虑使用因果图模型明确地建模状态的生成过程，并在此基础上增强策略。我们将因果结构更新形式化为具有主动环境干预学习的RL交互过程。为了优化衍生的目标，我们提出了一个具有理论性能保证的框架，两个步骤交替进行：在探索过程中使用干预进行因果结构学习，在开发过程中使用学习到的因果结构进行策略引导。由于缺少公共基准，用于对所提出的方法进行评估。

    As a key component to intuitive cognition and reasoning solutions in human intelligence, causal knowledge provides great potential for reinforcement learning (RL) agents' interpretability towards decision-making by helping reduce the searching space. However, there is still a considerable gap in discovering and incorporating causality into RL, which hinders the rapid development of causal RL. In this paper, we consider explicitly modeling the generation process of states with the causal graphical model, based on which we augment the policy. We formulate the causal structure updating into the RL interaction process with active intervention learning of the environment. To optimize the derived objective, we propose a framework with theoretical performance guarantees that alternates between two steps: using interventions for causal structure learning during exploration and using the learned causal structure for policy guidance during exploitation. Due to the lack of public benchmarks that 
    
[^6]: 条件最优传输的高效神经网络方法及贝叶斯推断中的应用

    Efficient Neural Network Approaches for Conditional Optimal Transport with Applications in Bayesian Inference. (arXiv:2310.16975v1 [stat.ML])

    [http://arxiv.org/abs/2310.16975](http://arxiv.org/abs/2310.16975)

    提出了两种神经网络方法来逼近静态和动态条件最优传输问题的解，实现了对条件概率分布的采样和密度估计，适用于贝叶斯推断。算法利用神经网络参数化传输映射以提高可扩展性。

    

    我们提出了两种神经网络方法，分别逼近静态和动态条件最优传输问题的解。这两种方法可以对条件概率分布进行采样和密度估计，这是贝叶斯推断中的核心任务。我们的方法将目标条件分布表示为可处理的参考分布的转换，因此属于测度传输的框架。在该框架中，COT映射是一个典型的选择，具有唯一性和单调性等可取的属性。然而，相关的COT问题在中等维度下计算具有挑战性。为了提高可扩展性，我们的数值算法利用神经网络对COT映射进行参数化。我们的方法充分利用了COT问题的静态和动态表达形式的结构。PCP-Map将条件传输映射建模为部分输入凸神经网络（PICNN）的梯度。

    We present two neural network approaches that approximate the solutions of static and dynamic conditional optimal transport (COT) problems, respectively. Both approaches enable sampling and density estimation of conditional probability distributions, which are core tasks in Bayesian inference. Our methods represent the target conditional distributions as transformations of a tractable reference distribution and, therefore, fall into the framework of measure transport. COT maps are a canonical choice within this framework, with desirable properties such as uniqueness and monotonicity. However, the associated COT problems are computationally challenging, even in moderate dimensions. To improve the scalability, our numerical algorithms leverage neural networks to parameterize COT maps. Our methods exploit the structure of the static and dynamic formulations of the COT problem. PCP-Map models conditional transport maps as the gradient of a partially input convex neural network (PICNN) and 
    
[^7]: 多尺度聚类过滤中的持久同调

    Persistent Homology of the Multiscale Clustering Filtration. (arXiv:2305.04281v2 [math.AT] UPDATED)

    [http://arxiv.org/abs/2305.04281](http://arxiv.org/abs/2305.04281)

    该论文引入了一种多尺度聚类过滤方法（MCF），用于描述不同尺度下的数据聚类，其中的持久同调可测量分区序列的层次关系和聚类分配冲突的出现和解决。

    

    在许多数据聚类应用中，不仅希望找到一种单一的分区方式，还希望找到描述不同尺度或粗糙层次下的数据的一系列分区方式。因此，一个自然的问题是分析和比较支撑这种多尺度数据描述的（不一定是层次性的）分区序列。在这里，我们引入了一种抽象单纯复形的过滤，称为多尺度聚类过滤（MCF），它编码了跨尺度的任意模式的聚类分配，并证明了MCF产生稳定的持久图。然后，我们展示了MCF的零维持久同调测量了分区序列中的层次关系程度，而高维持久同调则跟踪了分区序列中聚类分配冲突的出现和解决。为了拓宽MCF的理论基础，我们还提供了一个等价的构造方法。

    In many applications in data clustering, it is desirable to find not just a single partition into clusters but a sequence of partitions describing the data at different scales, or levels of coarseness. A natural problem then is to analyse and compare the (not necessarily hierarchical) sequences of partitions that underpin such multiscale descriptions of data. Here, we introduce a filtration of abstract simplicial complexes, denoted the Multiscale Clustering Filtration (MCF), which encodes arbitrary patterns of cluster assignments across scales, and we prove that the MCF produces stable persistence diagrams. We then show that the zero-dimensional persistent homology of the MCF measures the degree of hierarchy in the sequence of partitions, and that the higher-dimensional persistent homology tracks the emergence and resolution of conflicts between cluster assignments across the sequence of partitions. To broaden the theoretical foundations of the MCF, we also provide an equivalent constr
    

