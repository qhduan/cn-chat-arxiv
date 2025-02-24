# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MIM-Refiner: A Contrastive Learning Boost from Intermediate Pre-Trained Representations](https://arxiv.org/abs/2402.10093) | MIM-Refiner是一种对比学习提升方法，通过利用MIM模型中的中间层表示和多个对比头，能够将MIM模型的特征从次优的状态提升到最先进的状态，并在ImageNet-1K数据集上取得了新的最先进结果。 |
| [^2] | [Scaling Laws for Downstream Task Performance of Large Language Models](https://arxiv.org/abs/2402.04177) | 本研究探讨了在转移学习环境中大型语言模型的尺度行为，发现微调数据集的大小和预训练数据与下游数据的分布一致性对下游性能有显著影响。 |
| [^3] | [Peer-review-in-LLMs: Automatic Evaluation Method for LLMs in Open-environment](https://arxiv.org/abs/2402.01830) | 本文提出了一种新的无监督评估方法，利用同行评审机制在开放环境中衡量LLMs。通过为每个LLM分配可学习的能力参数，以最大化各个LLM的能力和得分的一致性。结果表明，高层次的LLM能够更准确地评估其他模型的答案，并能够获得更高的响应得分。 |
| [^4] | [GraphFM: Graph Factorization Machines for Feature Interaction Modeling](https://arxiv.org/abs/2105.11866) | 提出了一种名为GraphFM的图因子分解机方法，通过图结构自然表示特征，并将FM的交互功能集成到GNN的特征聚合策略中，能够模拟任意阶特征交互。 |
| [^5] | [Invertible Solution of Neural Differential Equations for Analysis of Irregularly-Sampled Time Series.](http://arxiv.org/abs/2401.04979) | 我们提出了一种可逆解决非规则采样时间序列的神经微分方程分析方法，通过引入神经流的概念，我们的方法既保证了可逆性又降低了计算负担，并且在分类和插值任务中表现出了优异的性能。 |
| [^6] | [Graph Neural Diffusion Networks for Semi-supervised Learning.](http://arxiv.org/abs/2201.09698) | 提出了一种名为 GND-Nets 的图神经网络，利用浅层网络和局部、全局邻域信息来解决图半监督学习中的过度平滑和欠平滑问题。 |

# 详细

[^1]: MIM-Refiner：一种从中间预训练表示中获得对比学习提升的方法

    MIM-Refiner: A Contrastive Learning Boost from Intermediate Pre-Trained Representations

    [https://arxiv.org/abs/2402.10093](https://arxiv.org/abs/2402.10093)

    MIM-Refiner是一种对比学习提升方法，通过利用MIM模型中的中间层表示和多个对比头，能够将MIM模型的特征从次优的状态提升到最先进的状态，并在ImageNet-1K数据集上取得了新的最先进结果。

    

    我们引入了MIM-Refiner，这是一种用于预训练MIM模型的对比学习提升方法。MIM-Refiner的动机在于MIM模型中的最佳表示通常位于中间层。因此，MIM-Refiner利用连接到不同中间层的多个对比头。在每个头中，修改后的最近邻目标帮助构建相应的语义聚类。此过程短而有效，在几个epochs内，我们将MIM模型的特征从次优的状态提升到最先进的状态。使用data2vec 2.0在ImageNet-1K上预训练的ViT-H经过改进后，在线性探测和低样本分类方面取得了新的最先进结果（分别为84.7%和64.2%），超过了在ImageNet-1K上预训练的其他模型的表现。

    arXiv:2402.10093v1 Announce Type: cross  Abstract: We introduce MIM (Masked Image Modeling)-Refiner, a contrastive learning boost for pre-trained MIM models. The motivation behind MIM-Refiner is rooted in the insight that optimal representations within MIM models generally reside in intermediate layers. Accordingly, MIM-Refiner leverages multiple contrastive heads that are connected to diverse intermediate layers. In each head, a modified nearest neighbor objective helps to construct respective semantic clusters.   The refinement process is short but effective. Within a few epochs, we refine the features of MIM models from subpar to state-of-the-art, off-the-shelf features. Refining a ViT-H, pre-trained with data2vec 2.0 on ImageNet-1K, achieves new state-of-the-art results in linear probing (84.7%) and low-shot classification among models that are pre-trained on ImageNet-1K. In ImageNet-1K 1-shot classification, MIM-Refiner sets a new state-of-the-art of 64.2%, outperforming larger mo
    
[^2]: 大型语言模型的下游任务性能的尺度律

    Scaling Laws for Downstream Task Performance of Large Language Models

    [https://arxiv.org/abs/2402.04177](https://arxiv.org/abs/2402.04177)

    本研究探讨了在转移学习环境中大型语言模型的尺度行为，发现微调数据集的大小和预训练数据与下游数据的分布一致性对下游性能有显著影响。

    

    尺度律提供了重要的见解，可以指导大型语言模型（LLM）的设计。现有研究主要集中在研究预训练（上游）损失的尺度律。然而，在转移学习环境中，LLM先在无监督数据集上进行预训练，然后在下游任务上进行微调，我们通常也关心下游性能。在这项工作中，我们研究了在转移学习环境中的尺度行为，其中LLM被微调用于机器翻译任务。具体而言，我们研究了预训练数据的选择和大小对下游性能（翻译质量）的影响，使用了两个评价指标：下游交叉熵和BLEU分数。我们的实验证明，微调数据集的大小和预训练数据与下游数据的分布一致性显著影响尺度行为。在充分一致性情况下，下游交叉熵和BLEU分数都会逐渐提升。

    Scaling laws provide important insights that can guide the design of large language models (LLMs). Existing work has primarily focused on studying scaling laws for pretraining (upstream) loss. However, in transfer learning settings, in which LLMs are pretrained on an unsupervised dataset and then finetuned on a downstream task, we often also care about the downstream performance. In this work, we study the scaling behavior in a transfer learning setting, where LLMs are finetuned for machine translation tasks. Specifically, we investigate how the choice of the pretraining data and its size affect downstream performance (translation quality) as judged by two metrics: downstream cross-entropy and BLEU score. Our experiments indicate that the size of the finetuning dataset and the distribution alignment between the pretraining and downstream data significantly influence the scaling behavior. With sufficient alignment, both downstream cross-entropy and BLEU score improve monotonically with 
    
[^3]: LLM中的同行评审方法：开放环境下LLMs的自动评估方法

    Peer-review-in-LLMs: Automatic Evaluation Method for LLMs in Open-environment

    [https://arxiv.org/abs/2402.01830](https://arxiv.org/abs/2402.01830)

    本文提出了一种新的无监督评估方法，利用同行评审机制在开放环境中衡量LLMs。通过为每个LLM分配可学习的能力参数，以最大化各个LLM的能力和得分的一致性。结果表明，高层次的LLM能够更准确地评估其他模型的答案，并能够获得更高的响应得分。

    

    现有的大型语言模型（LLMs）评估方法通常集中于在一些有人工注释的封闭环境和特定领域基准上测试性能。本文探索了一种新颖的无监督评估方法，利用同行评审机制自动衡量LLMs。在这个设置中，开源和闭源的LLMs处于同一环境中，能够回答未标记的问题并互相评估，每个LLM的响应得分由其他匿名的LLMs共同决定。为了获取这些模型之间的能力层次结构，我们为每个LLM分配一个可学习的能力参数来调整最终排序结果。我们将其形式化为一个受约束的优化问题，旨在最大化每个LLM的能力和得分的一致性。背后的关键假设是高层次的LLM能够比低层次的LLM更准确地评估其他模型的答案，而高层次的LLM也可以达到较高的响应得分。

    Existing large language models (LLMs) evaluation methods typically focus on testing the performance on some closed-environment and domain-specific benchmarks with human annotations. In this paper, we explore a novel unsupervised evaluation direction, utilizing peer-review mechanisms to measure LLMs automatically. In this setting, both open-source and closed-source LLMs lie in the same environment, capable of answering unlabeled questions and evaluating each other, where each LLM's response score is jointly determined by other anonymous ones. To obtain the ability hierarchy among these models, we assign each LLM a learnable capability parameter to adjust the final ranking. We formalize it as a constrained optimization problem, intending to maximize the consistency of each LLM's capabilities and scores. The key assumption behind is that high-level LLM can evaluate others' answers more accurately than low-level ones, while higher-level LLM can also achieve higher response scores. Moreover
    
[^4]: GraphFM：图因子分解机用于特征交互建模

    GraphFM: Graph Factorization Machines for Feature Interaction Modeling

    [https://arxiv.org/abs/2105.11866](https://arxiv.org/abs/2105.11866)

    提出了一种名为GraphFM的图因子分解机方法，通过图结构自然表示特征，并将FM的交互功能集成到GNN的特征聚合策略中，能够模拟任意阶特征交互。

    

    因子分解机（FM）是处理高维稀疏数据时建模成对（二阶）特征交互的一种常见方法。然而，一方面，FM未能捕捉到高阶特征交互，受到组合扩展的影响。另一方面，考虑每对特征之间的交互可能会引入噪声并降低预测准确性。为了解决这些问题，我们提出了一种新方法，称为Graph Factorization Machine（GraphFM），通过将特征自然表示成图结构。具体而言，我们设计了一种机制来选择有益的特征交互，并将其形式化为特征之间的边。然后，所提出的模型将FM的交互功能整合到图神经网络（GNN）的特征聚合策略中，通过堆叠层来模拟图结构特征上的任意阶特征交互。

    arXiv:2105.11866v4 Announce Type: replace-cross  Abstract: Factorization machine (FM) is a prevalent approach to modeling pairwise (second-order) feature interactions when dealing with high-dimensional sparse data. However, on the one hand, FM fails to capture higher-order feature interactions suffering from combinatorial expansion. On the other hand, taking into account interactions between every pair of features may introduce noise and degrade prediction accuracy. To solve the problems, we propose a novel approach, Graph Factorization Machine (GraphFM), by naturally representing features in the graph structure. In particular, we design a mechanism to select the beneficial feature interactions and formulate them as edges between features. Then the proposed model, which integrates the interaction function of FM into the feature aggregation strategy of Graph Neural Network (GNN), can model arbitrary-order feature interactions on the graph-structured features by stacking layers. Experime
    
[^5]: 可逆解决非规则采样时间序列的神经微分方程分析方法

    Invertible Solution of Neural Differential Equations for Analysis of Irregularly-Sampled Time Series. (arXiv:2401.04979v1 [cs.LG])

    [http://arxiv.org/abs/2401.04979](http://arxiv.org/abs/2401.04979)

    我们提出了一种可逆解决非规则采样时间序列的神经微分方程分析方法，通过引入神经流的概念，我们的方法既保证了可逆性又降低了计算负担，并且在分类和插值任务中表现出了优异的性能。

    

    为了处理非规则和不完整的时间序列数据的复杂性，我们提出了一种基于神经微分方程（NDE）的可逆解决方案。虽然基于NDE的方法是分析非规则采样时间序列的一种强大方法，但它们通常不能保证在其标准形式下进行可逆变换。我们的方法建议使用具有神经流的神经控制微分方程（Neural CDEs）的变种，该方法在保持较低的计算负担的同时确保了可逆性。此外，它还可以训练双重潜在空间，增强了对动态时间动力学的建模能力。我们的研究提出了一个先进的框架，在分类和插值任务中都表现出色。我们方法的核心是一个经过精心设计的增强型双重潜在状态架构，用于在各种时间序列任务中提高精度。实证分析表明，我们的方法明显优于现有模型。

    To handle the complexities of irregular and incomplete time series data, we propose an invertible solution of Neural Differential Equations (NDE)-based method. While NDE-based methods are a powerful method for analyzing irregularly-sampled time series, they typically do not guarantee reversible transformations in their standard form. Our method suggests the variation of Neural Controlled Differential Equations (Neural CDEs) with Neural Flow, which ensures invertibility while maintaining a lower computational burden. Additionally, it enables the training of a dual latent space, enhancing the modeling of dynamic temporal dynamics. Our research presents an advanced framework that excels in both classification and interpolation tasks. At the core of our approach is an enhanced dual latent states architecture, carefully designed for high precision across various time series tasks. Empirical analysis demonstrates that our method significantly outperforms existing models. This work significan
    
[^6]: 图神经扩散网络用于半监督学习

    Graph Neural Diffusion Networks for Semi-supervised Learning. (arXiv:2201.09698v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2201.09698](http://arxiv.org/abs/2201.09698)

    提出了一种名为 GND-Nets 的图神经网络，利用浅层网络和局部、全局邻域信息来解决图半监督学习中的过度平滑和欠平滑问题。

    

    图卷积网络 (GCN) 是用于基于图的半监督学习的先驱模型。然而，GCN 在标记稀疏的图上表现不佳。其两层版本不能有效地将标签信息传播到整个图结构（即欠平滑问题），而其深层版本则过度平滑且难以训练（即过度平滑问题）。为了解决这两个问题，我们提出了一种新的图神经网络，称为 GND-Nets（图神经扩散网络），它在单层中利用了顶点的局部和全局邻域信息。利用浅层网络可以缓解过度平滑问题，而利用局部和全局邻域信息可以缓解欠平滑问题。顶点的局部和全局邻域信息的利用是通过一种称为神经扩散的新图扩散方法实现的，该方法将神经网络融入传统的线性和非线性图扩散中。

    Graph Convolutional Networks (GCN) is a pioneering model for graph-based semi-supervised learning. However, GCN does not perform well on sparsely-labeled graphs. Its two-layer version cannot effectively propagate the label information to the whole graph structure (i.e., the under-smoothing problem) while its deep version over-smoothens and is hard to train (i.e., the over-smoothing problem). To solve these two issues, we propose a new graph neural network called GND-Nets (for Graph Neural Diffusion Networks) that exploits the local and global neighborhood information of a vertex in a single layer. Exploiting the shallow network mitigates the over-smoothing problem while exploiting the local and global neighborhood information mitigates the under-smoothing problem. The utilization of the local and global neighborhood information of a vertex is achieved by a new graph diffusion method called neural diffusions, which integrate neural networks into the conventional linear and nonlinear gra
    

