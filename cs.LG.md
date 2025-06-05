# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AdaptSFL: Adaptive Split Federated Learning in Resource-constrained Edge Networks](https://arxiv.org/abs/2403.13101) | 提出了AdaptSFL自适应分割联邦学习框架，以加速资源受限边缘系统中的学习性能。 |
| [^2] | [Online model error correction with neural networks: application to the Integrated Forecasting System](https://arxiv.org/abs/2403.03702) | 使用神经网络为欧洲中程气象中心的集成预测系统开发模型误差校正，以解决机器学习天气预报模型在表示动力平衡和适用于数据同化实验方面的挑战。 |
| [^3] | [Batched Nonparametric Contextual Bandits](https://arxiv.org/abs/2402.17732) | 该论文研究了批处理约束下的非参数上下文臂问题，提出了一种名为BaSEDB的方案，在动态分割协变量空间的同时，实现了最优的后悔。 |
| [^4] | [Understanding the Training Speedup from Sampling with Approximate Losses](https://arxiv.org/abs/2402.07052) | 本文研究利用近似损失进行样本采样的训练加速方法，通过贪婪策略选择具有大约损失的样本，减少选择的开销，并证明其收敛速度优于随机选择。同时开发了使用中间层表示获取近似损失的SIFT方法，并在训练BERT模型上取得了显著的提升。 |
| [^5] | [Pushing Large Language Models to the 6G Edge: Vision, Challenges, and Opportunities.](http://arxiv.org/abs/2309.16739) | 本文探讨了将大型语言模型(LLMs)部署在6G边缘的潜力和挑战。我们介绍了由LLMs支持的关键应用，并从响应时间、带宽成本和数据隐私等方面分析了云端部署面临的问题。我们提出了6G移动边缘计算(MEC)系统可能解决这些问题的方案，并讨论了边缘训练和边缘推理的创新技术。 |
| [^6] | [Easy attention: A simple self-attention mechanism for Transformers.](http://arxiv.org/abs/2308.12874) | 本论文提出了一种名为简易注意力的注意力机制，用于提高Transformer神经网络在混沌系统时间动态预测中的鲁棒性。该方法不依赖于键、查询和softmax，直接将注意力得分作为可学习参数。实验结果表明，该方法在重构和预测混沌系统的时间动态方面比传统的自注意机制和长短期记忆方法更具鲁棒性和简化性。 |
| [^7] | [Transforming to Yoked Neural Networks to Improve ANN Structure.](http://arxiv.org/abs/2306.02157) | 本文提出了一种叫做YNN的方法，将同一级别的ANN节点连接在一起形成神经模块，解决了普通ANN无法共享信息的缺陷，显著提高了信息传输和性能。 |
| [^8] | [EPIC: Graph Augmentation with Edit Path Interpolation via Learnable Cost.](http://arxiv.org/abs/2306.01310) | EPIC提出了一种基于插值的方法来增强图数据集，通过利用图编辑距离生成与原始图相似但有结构变化的新图，从而提高了分类模型的泛化能力。 |
| [^9] | [Learning Two-Layer Neural Networks, One (Giant) Step at a Time.](http://arxiv.org/abs/2305.18270) | 本文研究了浅层神经网络的训练动态及其条件，证明了动态下梯度下降可以通过有限数量的大批量梯度下降步骤来促进特征学习，并找到了多个和单一方向的最佳批量大小，有助于促进特征学习和方向的专业化。 |
| [^10] | [Torch-Choice: A PyTorch Package for Large-Scale Choice Modelling with Python.](http://arxiv.org/abs/2304.01906) | 本文介绍了一款名为 Torch-Choice 的 PyTorch 软件包，用于管理数据库、构建多项式Logit和嵌套Logit模型，并支持GPU加速，具有灵活性和高效性。 |

# 详细

[^1]: AdaptSFL：资源受限边缘网络中的自适应分割联邦学习

    AdaptSFL: Adaptive Split Federated Learning in Resource-constrained Edge Networks

    [https://arxiv.org/abs/2403.13101](https://arxiv.org/abs/2403.13101)

    提出了AdaptSFL自适应分割联邦学习框架，以加速资源受限边缘系统中的学习性能。

    

    深度神经网络的日益复杂使得将其民主化到资源有限的边缘设备面临重要障碍。为了解决这一挑战，通过模型分区将主要训练工作负荷转移到服务器上，并在边缘设备之间实现并行训练的分割联邦学习（SFL）已经成为一种有前途的解决方案。然而，尽管系统优化极大地影响了资源受限系统下SFL的性能，但这个问题仍然很大程度上没有被探索。本文提供了SFL的收敛分析，量化了模型分割（MS）和客户端模型聚合（MA）对学习性能的影响，作为理论基础。然后，我们提出了AdaptSFL，一种新颖的资源自适应SFL框架，以加速资源受限边缘计算系统下的SFL。具体来说，AdaptSFL自适应地控制客户端MA和MS，以平衡通信

    arXiv:2403.13101v1 Announce Type: new  Abstract: The increasing complexity of deep neural networks poses significant barriers to democratizing them to resource-limited edge devices. To address this challenge, split federated learning (SFL) has emerged as a promising solution by of floading the primary training workload to a server via model partitioning while enabling parallel training among edge devices. However, although system optimization substantially influences the performance of SFL under resource-constrained systems, the problem remains largely uncharted. In this paper, we provide a convergence analysis of SFL which quantifies the impact of model splitting (MS) and client-side model aggregation (MA) on the learning performance, serving as a theoretical foundation. Then, we propose AdaptSFL, a novel resource-adaptive SFL framework, to expedite SFL under resource-constrained edge computing systems. Specifically, AdaptSFL adaptively controls client-side MA and MS to balance commun
    
[^2]: 在线模型误差校正与神经网络: 应用于集成预测系统

    Online model error correction with neural networks: application to the Integrated Forecasting System

    [https://arxiv.org/abs/2403.03702](https://arxiv.org/abs/2403.03702)

    使用神经网络为欧洲中程气象中心的集成预测系统开发模型误差校正，以解决机器学习天气预报模型在表示动力平衡和适用于数据同化实验方面的挑战。

    

    最近几年，在全球数值天气预报模型的完全数据驱动开发方面取得了显著进展。这些机器学习天气预报模型具有其优势，尤其是准确性和较低的计算需求，但也存在其弱点：它们难以表示基本动力平衡，并且远未适用于资料同化实验。混合建模出现为解决这些限制的一种有希望的方法。混合模型将基于物理的核心组件与统计组件（通常是神经网络）集成在一起，以增强预测能力。在本文中，我们提出使用神经网络为欧洲中程气象中心的运行集成预测系统（IFS）开发模型误差校正。神经网络最初会离线进行预训练，使用大量运行分析数据集

    arXiv:2403.03702v1 Announce Type: cross  Abstract: In recent years, there has been significant progress in the development of fully data-driven global numerical weather prediction models. These machine learning weather prediction models have their strength, notably accuracy and low computational requirements, but also their weakness: they struggle to represent fundamental dynamical balances, and they are far from being suitable for data assimilation experiments. Hybrid modelling emerges as a promising approach to address these limitations. Hybrid models integrate a physics-based core component with a statistical component, typically a neural network, to enhance prediction capabilities. In this article, we propose to develop a model error correction for the operational Integrated Forecasting System (IFS) of the European Centre for Medium-Range Weather Forecasts using a neural network. The neural network is initially pre-trained offline using a large dataset of operational analyses and a
    
[^3]: 批处理非参数上下文臂

    Batched Nonparametric Contextual Bandits

    [https://arxiv.org/abs/2402.17732](https://arxiv.org/abs/2402.17732)

    该论文研究了批处理约束下的非参数上下文臂问题，提出了一种名为BaSEDB的方案，在动态分割协变量空间的同时，实现了最优的后悔。

    

    我们研究了在批处理约束下的非参数上下文臂问题，在这种情况下，每个动作的期望奖励被建模为协变量的平滑函数，并且策略更新是在每个Observations批次结束时进行的。我们为这种设置建立了一个最小化后悔的下限，并提出了一种名为Batched Successive Elimination with Dynamic Binning（BaSEDB）的方案，可以实现最优的后悔（达到对数因子）。实质上，BaSEDB动态地将协变量空间分割成更小的箱子，并仔细调整它们的宽度以符合批次大小。我们还展示了在批处理约束下静态分箱的非最优性，突出了动态分箱的必要性。另外，我们的结果表明，在完全在线设置中，几乎恒定数量的策略更新可以达到最佳后悔。

    arXiv:2402.17732v1 Announce Type: cross  Abstract: We study nonparametric contextual bandits under batch constraints, where the expected reward for each action is modeled as a smooth function of covariates, and the policy updates are made at the end of each batch of observations. We establish a minimax regret lower bound for this setting and propose Batched Successive Elimination with Dynamic Binning (BaSEDB) that achieves optimal regret (up to logarithmic factors). In essence, BaSEDB dynamically splits the covariate space into smaller bins, carefully aligning their widths with the batch size. We also show the suboptimality of static binning under batch constraints, highlighting the necessity of dynamic binning. Additionally, our results suggest that a nearly constant number of policy updates can attain optimal regret in the fully online setting.
    
[^4]: 理解通过使用近似损失进行采样的训练加速

    Understanding the Training Speedup from Sampling with Approximate Losses

    [https://arxiv.org/abs/2402.07052](https://arxiv.org/abs/2402.07052)

    本文研究利用近似损失进行样本采样的训练加速方法，通过贪婪策略选择具有大约损失的样本，减少选择的开销，并证明其收敛速度优于随机选择。同时开发了使用中间层表示获取近似损失的SIFT方法，并在训练BERT模型上取得了显著的提升。

    

    众所周知，选择具有较大损失/梯度的样本可以显著减少训练步骤的数量。然而，选择的开销往往过高，无法在总体训练时间方面获得有意义的提升。在本文中，我们专注于选择具有大约损失的样本的贪婪方法，而不是准确损失，以减少选择的开销。对于平滑凸损失，我们证明了这种贪婪策略可以在比随机选择更少的迭代次数内收敛到平均损失的最小值的常数因子。我们还理论上量化了近似水平的影响。然后，我们开发了使用中间层表示获取近似损失以进行样本选择的SIFT。我们评估了SIFT在训练一个具有1.1亿参数的12层BERT基础模型上的任务，并展示了显著的提升（以训练时间和反向传播步骤的数量衡量）。

    It is well known that selecting samples with large losses/gradients can significantly reduce the number of training steps. However, the selection overhead is often too high to yield any meaningful gains in terms of overall training time. In this work, we focus on the greedy approach of selecting samples with large \textit{approximate losses} instead of exact losses in order to reduce the selection overhead. For smooth convex losses, we show that such a greedy strategy can converge to a constant factor of the minimum value of the average loss in fewer iterations than the standard approach of random selection. We also theoretically quantify the effect of the approximation level. We then develop SIFT which uses early exiting to obtain approximate losses with an intermediate layer's representations for sample selection. We evaluate SIFT on the task of training a 110M parameter 12-layer BERT base model and show significant gains (in terms of training hours and number of backpropagation step
    
[^5]: 将大型语言模型推至6G边缘：视野、挑战和机遇

    Pushing Large Language Models to the 6G Edge: Vision, Challenges, and Opportunities. (arXiv:2309.16739v1 [cs.LG])

    [http://arxiv.org/abs/2309.16739](http://arxiv.org/abs/2309.16739)

    本文探讨了将大型语言模型(LLMs)部署在6G边缘的潜力和挑战。我们介绍了由LLMs支持的关键应用，并从响应时间、带宽成本和数据隐私等方面分析了云端部署面临的问题。我们提出了6G移动边缘计算(MEC)系统可能解决这些问题的方案，并讨论了边缘训练和边缘推理的创新技术。

    

    大型语言模型(LLMs)展示了显著的能力，正在改变人工智能的发展并有可能塑造我们的未来。然而，由于LLMs的多模态特性，当前的基于云的部署面临着一些关键挑战：1) 响应时间长；2) 高带宽成本；以及3) 违反数据隐私。6G移动边缘计算(MEC)系统可能解决这些迫切问题。本文探讨了在6G边缘部署LLMs的潜力。我们首先介绍了由多模态LLMs提供支持的关键应用，包括机器人技术和医疗保健，以突出在终端用户附近部署LLMs的需求。然后，我们确定了在边缘部署LLMs时面临的关键挑战，并设想了适用于LLMs的6G MEC架构。此外，我们深入探讨了两个设计方面，即LLMs的边缘训练和边缘推理。在这两个方面，考虑到边缘的固有资源限制，我们讨论了各种前沿技术。

    Large language models (LLMs), which have shown remarkable capabilities, are revolutionizing AI development and potentially shaping our future. However, given their multimodality, the status quo cloud-based deployment faces some critical challenges: 1) long response time; 2) high bandwidth costs; and 3) the violation of data privacy. 6G mobile edge computing (MEC) systems may resolve these pressing issues. In this article, we explore the potential of deploying LLMs at the 6G edge. We start by introducing killer applications powered by multimodal LLMs, including robotics and healthcare, to highlight the need for deploying LLMs in the vicinity of end users. Then, we identify the critical challenges for LLM deployment at the edge and envision the 6G MEC architecture for LLMs. Furthermore, we delve into two design aspects, i.e., edge training and edge inference for LLMs. In both aspects, considering the inherent resource limitations at the edge, we discuss various cutting-edge techniques, i
    
[^6]: 简易注意力：一种用于Transformer的简单自注意机制

    Easy attention: A simple self-attention mechanism for Transformers. (arXiv:2308.12874v1 [cs.LG])

    [http://arxiv.org/abs/2308.12874](http://arxiv.org/abs/2308.12874)

    本论文提出了一种名为简易注意力的注意力机制，用于提高Transformer神经网络在混沌系统时间动态预测中的鲁棒性。该方法不依赖于键、查询和softmax，直接将注意力得分作为可学习参数。实验结果表明，该方法在重构和预测混沌系统的时间动态方面比传统的自注意机制和长短期记忆方法更具鲁棒性和简化性。

    

    为了提高用于混沌系统时间动态预测的Transformer神经网络的鲁棒性，我们提出了一种新颖的注意力机制，称为简易注意力。由于自注意机制仅使用查询和键的内积，因此证明了为了获取捕捉时间序列的长期依赖关系所需的注意力得分，并不需要键、查询和softmax。通过在softmax注意力得分上实施奇异值分解（SVD），我们进一步观察到自注意力在注意力得分的张成空间中压缩了来自查询和键的贡献。因此，我们提出的简易注意力方法直接将注意力得分作为可学习参数。这种方法在重构和预测展现更强鲁棒性和更少复杂性的混沌系统的时间动态时取得了出色的结果，比自注意机制或广泛使用的长短期记忆

    To improve the robustness of transformer neural networks used for temporal-dynamics prediction of chaotic systems, we propose a novel attention mechanism called easy attention. Due to the fact that self attention only makes usage of the inner product of queries and keys, it is demonstrated that the keys, queries and softmax are not necessary for obtaining the attention score required to capture long-term dependencies in temporal sequences. Through implementing singular-value decomposition (SVD) on the softmax attention score, we further observe that the self attention compresses contribution from both queries and keys in the spanned space of the attention score. Therefore, our proposed easy-attention method directly treats the attention scores as learnable parameters. This approach produces excellent results when reconstructing and predicting the temporal dynamics of chaotic systems exhibiting more robustness and less complexity than the self attention or the widely-used long short-ter
    
[^7]: 将神经网络转化为Yoked神经网络以改进ANN结构

    Transforming to Yoked Neural Networks to Improve ANN Structure. (arXiv:2306.02157v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.02157](http://arxiv.org/abs/2306.02157)

    本文提出了一种叫做YNN的方法，将同一级别的ANN节点连接在一起形成神经模块，解决了普通ANN无法共享信息的缺陷，显著提高了信息传输和性能。

    

    大部分已经存在的经典人工神经网络（ANN）都被设计成树形结构以模拟神经网络。本文认为，树形结构的连接不足以描述神经网络。同一级别的节点不能连接在一起，即这些神经元不能相互共享信息，这是ANN的一个重大缺陷。我们提出了一种方法，即为同一级别的节点建立双向完全图，将同一级别的节点连接到一起形成神经模块。我们把我们的模型称为YNN。YNN显著促进了信息传输，明显有助于提高该方法的性能。我们的YNN可以更好地模拟神经网络，相比其他ANN方法有着明显的优势。

    Most existing classical artificial neural networks (ANN) are designed as a tree structure to imitate neural networks. In this paper, we argue that the connectivity of a tree is not sufficient to characterize a neural network. The nodes of the same level of a tree cannot be connected with each other, i.e., these neural unit cannot share information with each other, which is a major drawback of ANN. Although ANN has been significantly improved in recent years to more complex structures, such as the directed acyclic graph (DAG), these methods also have unidirectional and acyclic bias for ANN. In this paper, we propose a method to build a bidirectional complete graph for the nodes in the same level of an ANN, which yokes the nodes of the same level to formulate a neural module. We call our model as YNN in short. YNN promotes the information transfer significantly which obviously helps in improving the performance of the method. Our YNN can imitate neural networks much better compared with 
    
[^8]: EPIC: 通过可学习的代价实现的编辑路径插值的图形增强

    EPIC: Graph Augmentation with Edit Path Interpolation via Learnable Cost. (arXiv:2306.01310v1 [cs.LG])

    [http://arxiv.org/abs/2306.01310](http://arxiv.org/abs/2306.01310)

    EPIC提出了一种基于插值的方法来增强图数据集，通过利用图编辑距离生成与原始图相似但有结构变化的新图，从而提高了分类模型的泛化能力。

    

    基于图的模型在各个领域中变得越来越重要，但现有图数据集的有限规模和多样性经常限制它们的性能。为解决这个问题，我们提出了EPIC（通过可学习的代价实现的编辑路径插值），这是一种新颖的基于插值的增强图数据集的方法。我们的方法利用了图编辑距离来生成与原始图相似但结构有所变化的新图。为了实现这一点，我们通过比较带标签的图来学习图编辑距离，并利用这一知识在原始图对之间创建了图编辑路径。通过从图编辑路径中随机抽样的图形，我们丰富了训练集以增强分类模型的泛化能力。我们在几个基准数据集上展示了我们方法的有效性，并表明它在图分类任务中优于现有的增强方法。

    Graph-based models have become increasingly important in various domains, but the limited size and diversity of existing graph datasets often limit their performance. To address this issue, we propose EPIC (Edit Path Interpolation via learnable Cost), a novel interpolation-based method for augmenting graph datasets. Our approach leverages graph edit distance to generate new graphs that are similar to the original ones but exhibit some variation in their structures. To achieve this, we learn the graph edit distance through a comparison of labeled graphs and utilize this knowledge to create graph edit paths between pairs of original graphs. With randomly sampled graphs from a graph edit path, we enrich the training set to enhance the generalization capability of classification models. We demonstrate the effectiveness of our approach on several benchmark datasets and show that it outperforms existing augmentation methods in graph classification tasks.
    
[^9]: 学习两层神经网络：一次(巨大)的步骤。

    Learning Two-Layer Neural Networks, One (Giant) Step at a Time. (arXiv:2305.18270v1 [stat.ML])

    [http://arxiv.org/abs/2305.18270](http://arxiv.org/abs/2305.18270)

    本文研究了浅层神经网络的训练动态及其条件，证明了动态下梯度下降可以通过有限数量的大批量梯度下降步骤来促进特征学习，并找到了多个和单一方向的最佳批量大小，有助于促进特征学习和方向的专业化。

    

    我们研究了浅层神经网络的训练动态，研究了有限数量的大批量梯度下降步骤有助于在核心范围之外促进特征学习的条件。我们比较了批量大小和多个(但有限的)步骤的影响。我们分析了单步骤过程，发现批量大小为$n=O(d)$可以促进特征学习，但只适合学习单一方向或单索引模型。相比之下，$n=O(d^2)$对于学习多个方向和专业化至关重要。此外，我们证明“硬”方向缺乏前$\ell$个Hermite系数，仍未被发现，并且需要批量大小为$n=O(d^\ell)$才能被梯度下降捕获。经过几次迭代，情况发生变化：批量大小为$n=O(d)$足以学习新的目标方向，这些方向在Hermite基础上线性连接到之前学习的方向所涵盖的子空间。

    We study the training dynamics of shallow neural networks, investigating the conditions under which a limited number of large batch gradient descent steps can facilitate feature learning beyond the kernel regime. We compare the influence of batch size and that of multiple (but finitely many) steps. Our analysis of a single-step process reveals that while a batch size of $n = O(d)$ enables feature learning, it is only adequate for learning a single direction, or a single-index model. In contrast, $n = O(d^2)$ is essential for learning multiple directions and specialization. Moreover, we demonstrate that ``hard'' directions, which lack the first $\ell$ Hermite coefficients, remain unobserved and require a batch size of $n = O(d^\ell)$ for being captured by gradient descent. Upon iterating a few steps, the scenario changes: a batch-size of $n = O(d)$ is enough to learn new target directions spanning the subspace linearly connected in the Hermite basis to the previously learned directions,
    
[^10]: Torch-Choice: 用Python实现大规模选择建模的PyTorch包

    Torch-Choice: A PyTorch Package for Large-Scale Choice Modelling with Python. (arXiv:2304.01906v1 [cs.LG])

    [http://arxiv.org/abs/2304.01906](http://arxiv.org/abs/2304.01906)

    本文介绍了一款名为 Torch-Choice 的 PyTorch 软件包，用于管理数据库、构建多项式Logit和嵌套Logit模型，并支持GPU加速，具有灵活性和高效性。

    

    $\texttt{torch-choice}$ 是一款开源软件包，使用Python和PyTorch实现灵活、快速的选择建模。它提供了 $\texttt{ChoiceDataset}$ 数据结构，以便灵活而高效地管理数据库。本文演示了如何从各种格式的数据库中构建 $\texttt{ChoiceDataset}$，并展示了 $\texttt{ChoiceDataset}$ 的各种功能。该软件包实现了两种常用的模型: 多项式Logit和嵌套Logit模型，并支持模型估计期间的正则化。该软件包还支持使用GPU进行估计，使其可以扩展到大规模数据集而且在计算上更高效。模型可以使用R风格的公式字符串或Python字典进行初始化。最后，我们比较了 $\texttt{torch-choice}$ 和 R中的 $\texttt{mlogit}$ 在以下几个方面的计算效率: (1) 观测数增加时，(2) 协变量个数增加时， (3) 测试数升高时。

    The $\texttt{torch-choice}$ is an open-source library for flexible, fast choice modeling with Python and PyTorch. $\texttt{torch-choice}$ provides a $\texttt{ChoiceDataset}$ data structure to manage databases flexibly and memory-efficiently. The paper demonstrates constructing a $\texttt{ChoiceDataset}$ from databases of various formats and functionalities of $\texttt{ChoiceDataset}$. The package implements two widely used models, namely the multinomial logit and nested logit models, and supports regularization during model estimation. The package incorporates the option to take advantage of GPUs for estimation, allowing it to scale to massive datasets while being computationally efficient. Models can be initialized using either R-style formula strings or Python dictionaries. We conclude with a comparison of the computational efficiencies of $\texttt{torch-choice}$ and $\texttt{mlogit}$ in R as (1) the number of observations increases, (2) the number of covariates increases, and (3) th
    

