# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Energy Model-based Accurate Shapley Value Estimation for Interpretable Deep Learning Predictive Modelling](https://arxiv.org/abs/2404.01078) | EmSHAP提出了基于能量模型的Shapley值估计方法，通过引入GRU来消除输入特征顺序的影响，从而可以有效近似任意特征子集下深度学习模型的Shapley值贡献函数。 |
| [^2] | [Reset & Distill: A Recipe for Overcoming Negative Transfer in Continual Reinforcement Learning](https://arxiv.org/abs/2403.05066) | 开发了Reset & Distill（R&D）方法，通过重置代理的网络和提炼知识，有效克服了持续强化学习中负迁移问题。 |
| [^3] | [Towards Understanding the Riemannian SGD and SVRG Flows on Wasserstein Probabilistic Space.](http://arxiv.org/abs/2401.13530) | 本文研究了在Wasserstein概率空间上的Riemannian SGD和SVRG流的优化方法，通过构建随机微分方程来丰富Wasserstein空间中的连续优化方法。 |
| [^4] | [Efficient and Explainable Graph Neural Architecture Search via Monte-Carlo Tree Search.](http://arxiv.org/abs/2308.15734) | 该论文提出了一种高效且可解释的图神经网络架构搜索方法，名为ExGNAS。它包括适应各种图形的简单搜索空间和能解释决策过程的搜索算法。通过蒙特卡洛树搜索高效地搜索最佳GNN架构。 |
| [^5] | [Efficient Learning of Quantum States Prepared With Few Non-Clifford Gates.](http://arxiv.org/abs/2305.13409) | 该研究提出了一种能有效学习几个非克利福德门制备的量子状态的算法，并给出了一个随着稳定维数增大而学习所有状态的算法。 |

# 详细

[^1]: 基于能量模型的准确Shapley值估计用于可解释深度学习预测建模

    Energy Model-based Accurate Shapley Value Estimation for Interpretable Deep Learning Predictive Modelling

    [https://arxiv.org/abs/2404.01078](https://arxiv.org/abs/2404.01078)

    EmSHAP提出了基于能量模型的Shapley值估计方法，通过引入GRU来消除输入特征顺序的影响，从而可以有效近似任意特征子集下深度学习模型的Shapley值贡献函数。

    

    作为可解释人工智能（XAI）的有利工具，Shapley值已被广泛用于解释基于深度学习的预测模型。然而，由于计算负载随着输入特征的增加呈指数级增长，准确且高效地估计Shapley值是一项困难任务。大多数现有的加速Shapley值估计方法必须在估计准确性和效率之间做出妥协。在本文中，我们提出了EmSHAP（基于能量模型的Shapley值估计），它可以有效地近似预期Shapley贡献函数/深度学习模型在任意特征子集下给出其余特征的情况。为了确定能量模型中的提议条件分布，引入了门控循环单元（GRU），通过将输入特征映射到隐藏空间，从而消除了输入特征顺序的影响。此外，还采用了动态掩蔽方案.

    arXiv:2404.01078v1 Announce Type: new  Abstract: As a favorable tool for explainable artificial intelligence (XAI), Shapley value has been widely used to interpret deep learning based predictive models. However, accurate and efficient estimation of Shapley value is a difficult task since the computation load grows exponentially with the increase of input features. Most existing accelerated Shapley value estimation methods have to compromise on estimation accuracy with efficiency. In this article, we present EmSHAP(Energy model-based Shapley value estimation), which can effectively approximate the expectation of Shapley contribution function/deep learning model under arbitrary subset of features given the rest. In order to determine the proposal conditional distribution in the energy model, a gated recurrent unit(GRU) is introduced by mapping the input features onto a hidden space, so that the impact of input feature orderings can be eliminated. In addition, a dynamic masking scheme is 
    
[^2]: 复位和提炼：克服持续强化学习中负迁移的有效方法

    Reset & Distill: A Recipe for Overcoming Negative Transfer in Continual Reinforcement Learning

    [https://arxiv.org/abs/2403.05066](https://arxiv.org/abs/2403.05066)

    开发了Reset & Distill（R&D）方法，通过重置代理的网络和提炼知识，有效克服了持续强化学习中负迁移问题。

    

    我们认为发展有效的持续强化学习（CRL）算法的主要障碍之一是当需要学习新任务时会发生负迁移问题。通过全面的实验证实，我们证明这种问题在CRL中经常存在，并且无法通过最近一些旨在减轻RL代理的可塑性损失的工作来有效解决。为此，我们开发了Reset & Distill（R&D），这是一种简单但高效的方法，用于克服CRL中负迁移问题。R&D结合了一种策略，即重置代理的在线演员和评论网络以学习新任务，以及离线学习步骤，用于提炼在线演员和以前专家动作概率的知识。我们在Meta-World任务的长序列上进行了大量实验，并展示了我们的方法始终优于最近的基线，取得了显着更高的成功率。

    arXiv:2403.05066v1 Announce Type: cross  Abstract: We argue that one of the main obstacles for developing effective Continual Reinforcement Learning (CRL) algorithms is the negative transfer issue occurring when the new task to learn arrives. Through comprehensive experimental validation, we demonstrate that such issue frequently exists in CRL and cannot be effectively addressed by several recent work on mitigating plasticity loss of RL agents. To that end, we develop Reset & Distill (R&D), a simple yet highly effective method, to overcome the negative transfer problem in CRL. R&D combines a strategy of resetting the agent's online actor and critic networks to learn a new task and an offline learning step for distilling the knowledge from the online actor and previous expert's action probabilities. We carried out extensive experiments on long sequence of Meta-World tasks and show that our method consistently outperforms recent baselines, achieving significantly higher success rates acr
    
[^3]: 在Wasserstein概率空间上理解Riemannian SGD和SVRG流的研究

    Towards Understanding the Riemannian SGD and SVRG Flows on Wasserstein Probabilistic Space. (arXiv:2401.13530v1 [cs.LG])

    [http://arxiv.org/abs/2401.13530](http://arxiv.org/abs/2401.13530)

    本文研究了在Wasserstein概率空间上的Riemannian SGD和SVRG流的优化方法，通过构建随机微分方程来丰富Wasserstein空间中的连续优化方法。

    

    最近，对于Riemannian流形上的优化研究为优化领域提供了新的见解。在这方面，概率测度度量空间作为流形，配备第二阶Wasserstein距离，尤其引人关注，因为在其上的优化可以与实际的采样过程相关联。一般来说，Wasserstein空间上的最优化方法是Riemannian梯度流（即，在最小化KL散度时的Langevin动力学）。在本文中，我们旨在通过将梯度流延展到随机梯度下降（SGD）流和随机方差减少梯度（SVRG）流，丰富Wasserstein空间中的连续优化方法。Euclidean空间上的这两种流是标准的随机优化方法，而它们在Riemannian空间中的对应方法尚未被探索。通过利用Wasserstein空间中的结构，我们构建了一个随机微分方程（SDE）来近似离散动态。

    Recently, optimization on the Riemannian manifold has provided new insights to the optimization community. In this regard, the manifold taken as the probability measure metric space equipped with the second-order Wasserstein distance is of particular interest, since optimization on it can be linked to practical sampling processes. In general, the oracle (continuous) optimization method on Wasserstein space is Riemannian gradient flow (i.e., Langevin dynamics when minimizing KL divergence). In this paper, we aim to enrich the continuous optimization methods in the Wasserstein space by extending the gradient flow into the stochastic gradient descent (SGD) flow and stochastic variance reduction gradient (SVRG) flow. The two flows on Euclidean space are standard stochastic optimization methods, while their Riemannian counterparts are not explored yet. By leveraging the structures in Wasserstein space, we construct a stochastic differential equation (SDE) to approximate the discrete dynamic
    
[^4]: 高效且可解释的图神经网络架构搜索通过蒙特卡洛树搜索

    Efficient and Explainable Graph Neural Architecture Search via Monte-Carlo Tree Search. (arXiv:2308.15734v1 [cs.LG])

    [http://arxiv.org/abs/2308.15734](http://arxiv.org/abs/2308.15734)

    该论文提出了一种高效且可解释的图神经网络架构搜索方法，名为ExGNAS。它包括适应各种图形的简单搜索空间和能解释决策过程的搜索算法。通过蒙特卡洛树搜索高效地搜索最佳GNN架构。

    

    图神经网络（GNNs）是在各个领域进行数据科学任务的强大工具。尽管我们在广泛的应用场景中使用GNNs，但对研究人员和实践者来说，在不同的图中设计/选择最佳GNN架构是一项费力的任务。为了节省人力和计算成本，已经使用图神经网络架构搜索（Graph NAS）来搜索结合现有组件的次优GNN架构。然而，目前没有现有的Graph NAS方法能够同时满足可解释性、高效性和适应多样化图形的要求。因此，我们提出了一种高效且可解释的Graph NAS方法，称为ExGNAS，它包括（i）一个可以适应各种图形的简单搜索空间和（ii）一个能够解释决策过程的搜索算法。搜索空间仅包含可以处理同质和异质图的基本函数。搜索算法通过蒙特卡洛树搜索高效地搜索最佳GNN架构。

    Graph neural networks (GNNs) are powerful tools for performing data science tasks in various domains. Although we use GNNs in wide application scenarios, it is a laborious task for researchers and practitioners to design/select optimal GNN rchitectures in diverse graphs. To save human efforts and computational costs, graph neural architecture search (Graph NAS) has been used to search for a sub-optimal GNN architecture that combines existing components. However, there are no existing Graph NAS methods that satisfy explainability, efficiency, and adaptability to various graphs. Therefore, we propose an efficient and explainable Graph NAS method, called ExGNAS, which consists of (i) a simple search space that can adapt to various graphs and (ii) a search algorithm that makes the decision process explainable. The search space includes only fundamental functions that can handle homophilic and heterophilic graphs. The search algorithm efficiently searches for the best GNN architecture via M
    
[^5]: 几个非克利福德门制备的量子状态的有效学习

    Efficient Learning of Quantum States Prepared With Few Non-Clifford Gates. (arXiv:2305.13409v1 [quant-ph])

    [http://arxiv.org/abs/2305.13409](http://arxiv.org/abs/2305.13409)

    该研究提出了一种能有效学习几个非克利福德门制备的量子状态的算法，并给出了一个随着稳定维数增大而学习所有状态的算法。

    

    我们提出了一种算法，可以有效地学习通过克利福德门和$O(\log(n))$个非克利福德门制备的量子状态。具体而言，对于最多使用$t$个非克利福德门制备的$n$量子比特状态$|\psi\rangle$，我们证明可以用$\mathsf{poly}(n,2^t,1/\epsilon)$时间和$|\psi\rangle$的复制来学习$|\psi\rangle$，使其跟真实状态的距离不超过$\epsilon$。该结果是一个稳定维数较大的状态学习算法的特例，当一个量子状态的稳定子维数为$k$，表示它被一个由$2^k$个Pauli算子的Abel群稳定。

    We give an algorithm that efficiently learns a quantum state prepared by Clifford gates and $O(\log(n))$ non-Clifford gates. Specifically, for an $n$-qubit state $\lvert \psi \rangle$ prepared with at most $t$ non-Clifford gates, we show that $\mathsf{poly}(n,2^t,1/\epsilon)$ time and copies of $\lvert \psi \rangle$ suffice to learn $\lvert \psi \rangle$ to trace distance at most $\epsilon$. This result follows as a special case of an algorithm for learning states with large stabilizer dimension, where a quantum state has stabilizer dimension $k$ if it is stabilized by an abelian group of $2^k$ Pauli operators. We also develop an efficient property testing algorithm for stabilizer dimension, which may be of independent interest.
    

