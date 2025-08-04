# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Developmental Landscape of In-Context Learning](https://arxiv.org/abs/2402.02364) | 在transformers模型中，我们展示了在上下文学习中的离散发展阶段，并引入了两种方法来检测这些阶段的关键里程碑。我们使用行为和结构度量验证了这些方法的有效性。 |
| [^2] | [Size Generalizability of Graph Neural Networks on Biological Data: Insights and Practices from the Spectral Perspective.](http://arxiv.org/abs/2305.15611) | 本文通过谱角度的方法，研究了GNNs的尺寸可泛化性问题，并在真实生物数据集上进行了实验，发现GNNs在度分布和谱分布偏移时均表现敏感，在同一数据集的大图上的性能仍然下降，揭示了 GNNs的尺寸可泛化性问题。 |
| [^3] | [Gradient Leakage Defense with Key-Lock Module for Federated Learning.](http://arxiv.org/abs/2305.04095) | 本研究提出了一种新的联邦学习梯度泄露防御技术，使用私钥锁模块保护任意模型体系结构，并可确保无法从共享的梯度中重建私有训练数据。 |
| [^4] | [Identifying Unique Causal Network from Nonstationary Time Series.](http://arxiv.org/abs/2211.10085) | 本文提出了一种名为UCN的新型因果模型，它考虑了时间延迟的影响，并证明了所得到的网络结构的唯一性，解决了因果解释性和非静态性问题。 |

# 详细

[^1]: 在上下文中学习的发展景观

    The Developmental Landscape of In-Context Learning

    [https://arxiv.org/abs/2402.02364](https://arxiv.org/abs/2402.02364)

    在transformers模型中，我们展示了在上下文学习中的离散发展阶段，并引入了两种方法来检测这些阶段的关键里程碑。我们使用行为和结构度量验证了这些方法的有效性。

    

    我们展示了在transformers中，当它们通过语言建模或线性回归任务进行训练时，上下文学习是如何以离散的发展阶段出现的。我们引入了两种方法来检测分隔这些阶段的关键里程碑，通过探测参数空间和函数空间中种群损失的几何特征。我们使用一系列行为和结构度量研究这些新方法揭示的阶段，以建立它们的有效性。

    We show that in-context learning emerges in transformers in discrete developmental stages, when they are trained on either language modeling or linear regression tasks. We introduce two methods for detecting the milestones that separate these stages, by probing the geometry of the population loss in both parameter space and function space. We study the stages revealed by these new methods using a range of behavioral and structural metrics to establish their validity.
    
[^2]: 基于谱角度剖析生物数据中图神经网络的尺寸可泛化性：观点和实践

    Size Generalizability of Graph Neural Networks on Biological Data: Insights and Practices from the Spectral Perspective. (arXiv:2305.15611v1 [cs.LG])

    [http://arxiv.org/abs/2305.15611](http://arxiv.org/abs/2305.15611)

    本文通过谱角度的方法，研究了GNNs的尺寸可泛化性问题，并在真实生物数据集上进行了实验，发现GNNs在度分布和谱分布偏移时均表现敏感，在同一数据集的大图上的性能仍然下降，揭示了 GNNs的尺寸可泛化性问题。

    

    本文探讨了图神经网络 (GNNs) 是否具有从小图中学习的知识可推广到同一领域的大图中。之前的研究表明，不同大小的图之间的分布偏移，尤其是度分布，可能会导致图分类任务的性能下降。然而，在生物数据集中，度数是有界的，因此度分布的偏移很小。即使度分布偏移很小，我们观察到GNNs在同一数据集的大图上的性能仍然下降，暗示有其他原因。事实上，以往对于真实数据集中各种图尺寸引起的分布偏移类型和属性的探索不足。此外，以前的尺寸可泛化性分析大多集中在空间领域。为填补这些空白，我们采用谱角度去研究GNNs在生物图数据上的尺寸可泛化性。我们首先提出一个新框架来模拟各种类型的度分布偏移，并利用它来测试GNNs 在真实生物数据集上的尺寸可泛化性。我们的实验表明，除了度分布偏移外，GNNs 还对图大小变化引起的谱分布偏移很敏感。我们进一步分析了不同的GNN模型的影响，并表明，一些模型比其他模型更具有尺寸泛化性。本文展示了关于GNNs尺寸可泛化性问题的新观点和实践，并为该领域的未来研究提供了有益的洞察和建议。

    We investigate the question of whether the knowledge learned by graph neural networks (GNNs) from small graphs is generalizable to large graphs in the same domain. Prior works suggest that the distribution shift, particularly in the degree distribution, between graphs of different sizes can lead to performance degradation in the graph classification task. However, this may not be the case for biological datasets where the degrees are bounded and the distribution shift of degrees is small. Even with little degree distribution shift, our observations show that GNNs' performance on larger graphs from the same datasets still degrades, suggesting other causes. In fact, there has been a lack of exploration in real datasets to understand the types and properties of distribution shifts caused by various graph sizes. Furthermore, previous analyses of size generalizability mostly focus on the spatial domain.  To fill these gaps, we take the spectral perspective and study the size generalizabilit
    
[^3]: 基于密钥锁模块的联邦学习梯度泄露防御

    Gradient Leakage Defense with Key-Lock Module for Federated Learning. (arXiv:2305.04095v1 [cs.LG])

    [http://arxiv.org/abs/2305.04095](http://arxiv.org/abs/2305.04095)

    本研究提出了一种新的联邦学习梯度泄露防御技术，使用私钥锁模块保护任意模型体系结构，并可确保无法从共享的梯度中重建私有训练数据。

    

    联邦学习是一种广泛采用的隐私保护机器学习方法，其中私有数据保持本地，允许安全计算和本地模型梯度与第三方参数服务器之间的交换。然而，最近的研究发现，通过共享的梯度可能会危及隐私并恢复敏感信息。本研究提供了详细的分析和对梯度泄漏问题的新视角。这些理论工作导致了一种新的梯度泄露防御技术，使用私钥锁模块保护任意模型体系结构。只有锁定的梯度被传输到参数服务器进行全局模型聚合。我们提出的学习方法对梯度泄露攻击具有抵抗力，并且所设计和训练的密钥锁模块可以确保，没有密钥锁模块的私有信息：a) 无法从共享的梯度中重建私有训练数据。

    Federated Learning (FL) is a widely adopted privacy-preserving machine learning approach where private data remains local, enabling secure computations and the exchange of local model gradients between local clients and third-party parameter servers. However, recent findings reveal that privacy may be compromised and sensitive information potentially recovered from shared gradients. In this study, we offer detailed analysis and a novel perspective on understanding the gradient leakage problem. These theoretical works lead to a new gradient leakage defense technique that secures arbitrary model architectures using a private key-lock module. Only the locked gradient is transmitted to the parameter server for global model aggregation. Our proposed learning method is resistant to gradient leakage attacks, and the key-lock module is designed and trained to ensure that, without the private information of the key-lock module: a) reconstructing private training data from the shared gradient is
    
[^4]: 从非静止时间序列中识别独特的因果网络

    Identifying Unique Causal Network from Nonstationary Time Series. (arXiv:2211.10085v3 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2211.10085](http://arxiv.org/abs/2211.10085)

    本文提出了一种名为UCN的新型因果模型，它考虑了时间延迟的影响，并证明了所得到的网络结构的唯一性，解决了因果解释性和非静态性问题。

    

    在许多数据密集型场景下，识别因果关系是一项具有挑战性的任务。已经提出了许多用于此关键任务的算法。然而，大多数算法仅考虑了贝叶斯网络（BN）的有向无环图（DAG）的学习算法。这些基于BN的模型仅具有有限的因果可解释性，因为存在马尔可夫等价类的问题。此外，它们依赖于静止性假设，而来自复杂系统的许多采样时间序列是非静止的。非静止的时间序列带来了数据集漂移问题，导致这些算法的性能不佳。为了填补这些空白，本文提出了一种名为Unique Causal Network（UCN）的新型因果模型。与以前的基于BN的模型不同，UCN考虑了时间延迟的影响，并证明了所得到的网络结构的唯一性，解决了马尔可夫等价类的问题。此外，基于UCN的可分解性属性，提出了更高的...

    Identifying causality is a challenging task in many data-intensive scenarios. Many algorithms have been proposed for this critical task. However, most of them consider the learning algorithms for directed acyclic graph (DAG) of Bayesian network (BN). These BN-based models only have limited causal explainability because of the issue of Markov equivalence class. Moreover, they are dependent on the assumption of stationarity, whereas many sampling time series from complex system are nonstationary. The nonstationary time series bring dataset shift problem, which leads to the unsatisfactory performances of these algorithms. To fill these gaps, a novel causation model named Unique Causal Network (UCN) is proposed in this paper. Different from the previous BN-based models, UCN considers the influence of time delay, and proves the uniqueness of obtained network structure, which addresses the issue of Markov equivalence class. Furthermore, based on the decomposability property of UCN, a higher-
    

