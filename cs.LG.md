# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MoPE: Parameter-Efficient and Scalable Multimodal Fusion via Mixture of Prompt Experts](https://arxiv.org/abs/2403.10568) | 本文提出了MoPE技术，通过解开提示以自适应捕获数据集级和实例级特征，引入了混合Prompt专家来增强表达能力，并且在多模态融合中表现出更大的表达能力和可扩展性。 |
| [^2] | [Mirror Descent Algorithms with Nearly Dimension-Independent Rates for Differentially-Private Stochastic Saddle-Point Problems](https://arxiv.org/abs/2403.02912) | 提出了具有差分隐私的随机鞍点问题的镜像下降算法，实现了几乎与维度无关的收敛速率，这种速率之前只针对双线性目标已知。 |
| [^3] | [Understanding Deep Representation Learning via Layerwise Feature Compression and Discrimination.](http://arxiv.org/abs/2311.02960) | 本文通过研究中间特征的结构，揭示了深度网络在层级特征学习过程中的演化模式。研究发现线性层在特征学习中起到了与深层非线性网络类似的作用。 |
| [^4] | [Connected Hidden Neurons (CHNNet): An Artificial Neural Network for Rapid Convergence.](http://arxiv.org/abs/2305.10468) | 该论文提出了一个更为强大的人工神经网络模型，该模型中同一隐藏层中的隐藏神经元相互连接，可以学习复杂模式并加速收敛速度。 |
| [^5] | [NervePool: A Simplicial Pooling Layer.](http://arxiv.org/abs/2305.06315) | 单纯复形池化层NervePool在池化图结构数据时，基于顶点分区生成单纯复形的分层表示，可以灵活地建模更高阶的关系，同时缩小高维单纯形，实现降采样，减少计算成本和减少过拟合。 |

# 详细

[^1]: MoPE：通过Prompt专家混合实现参数高效和可扩展的多模态融合

    MoPE: Parameter-Efficient and Scalable Multimodal Fusion via Mixture of Prompt Experts

    [https://arxiv.org/abs/2403.10568](https://arxiv.org/abs/2403.10568)

    本文提出了MoPE技术，通过解开提示以自适应捕获数据集级和实例级特征，引入了混合Prompt专家来增强表达能力，并且在多模态融合中表现出更大的表达能力和可扩展性。

    

    Prompt调整已经证明在融合多模态任务的单模基础模型时具有参数效率性。然而，其有限的适应性和表达能力导致性能不佳与其他调整方法相比。本文通过将简单提示解开以自适应地捕获数据集级和实例级特征来解决这个问题。建立在这种解开的基础上，我们引入了Prompt专家的混合（MoPE）技术来增强表达能力。MoPE利用多模态配对先验在每个实例基础上路由最有效的提示。与简单提示相比，我们基于MoPE的条件提示对多模态融合具有更大的表达能力，在训练数据和可训练参数总数上具有更好的扩展性。我们还研究了一个专家路由的正则化项，导致专家的不断发展专长，不同专家专注于不同的特征。

    arXiv:2403.10568v1 Announce Type: cross  Abstract: Prompt-tuning has demonstrated parameter-efficiency in fusing unimodal foundation models for multimodal tasks. However, its limited adaptivity and expressiveness lead to suboptimal performance when compared with other tuning methods. In this paper, we address this issue by disentangling the vanilla prompts to adaptively capture dataset-level and instance-level features. Building upon this disentanglement, we introduce the mixture of prompt experts (MoPE) technique to enhance expressiveness. MoPE leverages multimodal pairing priors to route the most effective prompt on a per-instance basis. Compared to vanilla prompting, our MoPE-based conditional prompting exhibits greater expressiveness for multimodal fusion, scaling better with the training data and the overall number of trainable parameters. We also study a regularization term for expert routing, leading to emergent expert specialization, where different experts focus on different c
    
[^2]: 具有几乎与维度无关收敛速率的镜像下降算法，用于具有差分隐私的随机鞍点问题

    Mirror Descent Algorithms with Nearly Dimension-Independent Rates for Differentially-Private Stochastic Saddle-Point Problems

    [https://arxiv.org/abs/2403.02912](https://arxiv.org/abs/2403.02912)

    提出了具有差分隐私的随机鞍点问题的镜像下降算法，实现了几乎与维度无关的收敛速率，这种速率之前只针对双线性目标已知。

    

    我们研究了多面体设置中具有差分隐私（DP）的随机（凸凹）鞍点问题。我们提出了基于随机镜像下降的（ϵ，δ）-DP算法，其实现了预期对偶间隙的几乎与维度无关的收敛速率，这种保证在以前只针对双线性目标已知。对于凸凹和一阶平滑随机目标，我们的算法实现了一个率，即sqrt(log(d)/n) + (log(d)^{3/2}/[nϵ])^{1/3}，其中d是问题的维度，n是数据集大小。在额外的二阶平滑性假设下，我们将预期间隙的速率改进为sqrt(log(d)/n) + (log(d)^{3/2}/[nϵ])^{2/5}。在这种额外假设下，我们还通过使用偏差减少的梯度估计器，证明了对偶间隙受常数成功概率的界为log(d)/sqrt(n) + log(d)/[nϵ]^{1/2}。

    arXiv:2403.02912v1 Announce Type: cross  Abstract: We study the problem of differentially-private (DP) stochastic (convex-concave) saddle-points in the polyhedral setting. We propose $(\varepsilon, \delta)$-DP algorithms based on stochastic mirror descent that attain nearly dimension-independent convergence rates for the expected duality gap, a type of guarantee that was known before only for bilinear objectives. For convex-concave and first-order-smooth stochastic objectives, our algorithms attain a rate of $\sqrt{\log(d)/n} + (\log(d)^{3/2}/[n\varepsilon])^{1/3}$, where $d$ is the dimension of the problem and $n$ the dataset size. Under an additional second-order-smoothness assumption, we improve the rate on the expected gap to $\sqrt{\log(d)/n} + (\log(d)^{3/2}/[n\varepsilon])^{2/5}$. Under this additional assumption, we also show, by using bias-reduced gradient estimators, that the duality gap is bounded by $\log(d)/\sqrt{n} + \log(d)/[n\varepsilon]^{1/2}$ with constant success pro
    
[^3]: 通过层间特征压缩和差别性学习理解深度表示学习

    Understanding Deep Representation Learning via Layerwise Feature Compression and Discrimination. (arXiv:2311.02960v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2311.02960](http://arxiv.org/abs/2311.02960)

    本文通过研究中间特征的结构，揭示了深度网络在层级特征学习过程中的演化模式。研究发现线性层在特征学习中起到了与深层非线性网络类似的作用。

    

    在过去的十年中，深度学习已经证明是从原始数据中学习有意义特征的一种高效工具。然而，深度网络如何在不同层级上进行等级特征学习仍然是一个开放问题。在这项工作中，我们试图通过研究中间特征的结构揭示这个谜团。受到我们实证发现的线性层在特征学习中模仿非线性网络中深层的角色的启发，我们研究了深度线性网络如何将输入数据转化为输出，通过研究训练后的每个层的输出（即特征）在多类分类问题的背景下。为了实现这个目标，我们首先定义了衡量中间特征的类内压缩和类间差别性的度量标准。通过对这两个度量标准的理论分析，我们展示了特征从浅层到深层的演变遵循着一种简单而量化的模式，前提是输入数据是

    Over the past decade, deep learning has proven to be a highly effective tool for learning meaningful features from raw data. However, it remains an open question how deep networks perform hierarchical feature learning across layers. In this work, we attempt to unveil this mystery by investigating the structures of intermediate features. Motivated by our empirical findings that linear layers mimic the roles of deep layers in nonlinear networks for feature learning, we explore how deep linear networks transform input data into output by investigating the output (i.e., features) of each layer after training in the context of multi-class classification problems. Toward this goal, we first define metrics to measure within-class compression and between-class discrimination of intermediate features, respectively. Through theoretical analysis of these two metrics, we show that the evolution of features follows a simple and quantitative pattern from shallow to deep layers when the input data is
    
[^4]: 连接隐藏神经元（CHNNet）：一种快速收敛的人工神经网络

    Connected Hidden Neurons (CHNNet): An Artificial Neural Network for Rapid Convergence. (arXiv:2305.10468v1 [cs.NE])

    [http://arxiv.org/abs/2305.10468](http://arxiv.org/abs/2305.10468)

    该论文提出了一个更为强大的人工神经网络模型，该模型中同一隐藏层中的隐藏神经元相互连接，可以学习复杂模式并加速收敛速度。

    

    人工神经网络的核心目的是模仿生物神经网络的功能。然而，与生物神经网络不同，传统的人工神经网络通常是按层次结构化的，这可能会妨碍神经元之间的信息流动，因为同一层中的神经元之间没有连接。因此，我们提出了一种更为强大的人工神经网络模型，其中同一隐藏层中的隐藏神经元是互相连接的，使得神经元能够学习复杂的模式并加速收敛速度。通过在浅层和深层网络中将我们提出的模型作为完全连接的层进行实验研究，我们证明这个模型可以显著提高收敛速率。

    The core purpose of developing artificial neural networks was to mimic the functionalities of biological neural networks. However, unlike biological neural networks, traditional artificial neural networks are often structured hierarchically, which can impede the flow of information between neurons as the neurons in the same layer have no connections between them. Hence, we propose a more robust model of artificial neural networks where the hidden neurons, residing in the same hidden layer, are interconnected, enabling the neurons to learn complex patterns and speeding up the convergence rate. With the experimental study of our proposed model as fully connected layers in shallow and deep networks, we demonstrate that the model results in a significant increase in convergence rate.
    
[^5]: NervePool: 一个单纯复形池化层

    NervePool: A Simplicial Pooling Layer. (arXiv:2305.06315v1 [cs.CG])

    [http://arxiv.org/abs/2305.06315](http://arxiv.org/abs/2305.06315)

    单纯复形池化层NervePool在池化图结构数据时，基于顶点分区生成单纯复形的分层表示，可以灵活地建模更高阶的关系，同时缩小高维单纯形，实现降采样，减少计算成本和减少过拟合。

    

    对于图结构数据的深度学习问题，池化层对于降采样、减少计算成本和减少过拟合都很重要。我们提出了一个池化层，NervePool，适用于单纯复形结构的数据，这种结构是图的推广，包括比顶点和边更高维度的单纯形；这种结构可以更灵活地建模更高阶的关系。所提出的单纯复合缩小方案基于顶点的分区构建，这使得我们可以生成单纯复形的分层表示，以一种学习的方式折叠信息。NervePool建立在学习的顶点群集分配的基础上，并以一种确定性的方式扩展到高维单纯形的缩小。虽然在实践中，池化操作是通过一系列矩阵运算来计算的，但是其拓扑动机是一个基于单纯形星星的并集和神经复合体的集合构造。

    For deep learning problems on graph-structured data, pooling layers are important for down sampling, reducing computational cost, and to minimize overfitting. We define a pooling layer, NervePool, for data structured as simplicial complexes, which are generalizations of graphs that include higher-dimensional simplices beyond vertices and edges; this structure allows for greater flexibility in modeling higher-order relationships. The proposed simplicial coarsening scheme is built upon partitions of vertices, which allow us to generate hierarchical representations of simplicial complexes, collapsing information in a learned fashion. NervePool builds on the learned vertex cluster assignments and extends to coarsening of higher dimensional simplices in a deterministic fashion. While in practice, the pooling operations are computed via a series of matrix operations, the topological motivation is a set-theoretic construction based on unions of stars of simplices and the nerve complex
    

