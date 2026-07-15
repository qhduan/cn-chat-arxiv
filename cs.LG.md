# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Beyond the Hard Budget: Sparsity Regularizers for More Interpretable Top-k Sparse Autoencoders](https://arxiv.org/abs/2606.27321) | 本文针对Top-k稀疏自编码器固定预算和过拟合缺陷，提出了两种新的稀疏正则化方法，通过作用于Top-k选择前的激活值来提升模型可解释性。 |
| [^2] | [Explaining Temporal Graph Neural Networks via Feature-induced Information Flow](https://arxiv.org/abs/2606.27201) | 提出了一种基于正则化相关性度量框架的新归因方法，通过分析所有事件相关变量中的完整信息流，克服了现有方法忽略事件诱导变量路径的局限，从而更全面地解释基于事件的时序图神经网络。 |
| [^3] | [TopCoW: Benchmarking Topology-Aware Anatomical Segmentation of the Circle of Willis (CoW) for CTA and MRA.](http://arxiv.org/abs/2312.17670) | 这项研究提出了TopCoW挑战，通过发布具有13种血管组分注释的Willis循环（CoW）数据集，并使用虚拟现实（VR）技术进行拓扑感知解剖分割，解决了手动和耗时的CoW表征问题。 |

# 详细

[^1]: 超越硬预算：面向更可解释的Top-k稀疏自编码器的稀疏正则化方法

    Beyond the Hard Budget: Sparsity Regularizers for More Interpretable Top-k Sparse Autoencoders

    [https://arxiv.org/abs/2606.27321](https://arxiv.org/abs/2606.27321)

    本文针对Top-k稀疏自编码器固定预算和过拟合缺陷，提出了两种新的稀疏正则化方法，通过作用于Top-k选择前的激活值来提升模型可解释性。

    

    arXiv:2606.27321v1 公告类型：交叉 摘要：稀疏自编码器已成为解释视觉基础模型表示的主要工具，将其多语义激活分解为更大规模的稀疏、更单语义特征集合。Top-k稀疏自编码器作为当前标准变体，通过其激活函数从架构层面强制实现稀疏性，每个输入仅保留最活跃的k个潜在变量。由于该设计旨在规避早期稀疏自编码器使用的ℓ1惩罚及其已知缺陷，因此尽管其本身存在局限性——如预算k固定不变（不随输入复杂度调整）以及倾向于对训练时设定的k值过拟合——但至今未与显式稀疏正则化方法结合。我们提出了两种与Top-k架构兼容的稀疏正则化方法，两者均在Top-k选择之前作用于激活值：对未选中单元施加ℓ1惩罚，以及一种尺度不变的ℓ...（原文截断）

    arXiv:2606.27321v1 Announce Type: cross  Abstract: Sparse autoencoders (SAEs) have become a leading tool for interpreting the representations of vision foundation models, decomposing their polysemantic activations into a larger set of sparse, more monosemantic features. The Top-$k$ SAE, a now-standard variant, enforces sparsity architecturally through its activation function, retaining only the $k$ most active latents per input. Because it was designed precisely to avoid the $\ell_1$ penalty used by earlier SAEs and its known drawbacks, it has not been combined with an explicit sparsity regularizer, despite retaining limitations of its own, such as a budget $k$ that is fixed regardless of input complexity and a tendency to overfit to the training value of $k$. We introduce two sparsity regularizers compatible with the Top-$k$ architecture, both acting on the activations before the Top-$k$ selection: an $\ell_1$ penalty on the unselected (off-support) units, and a scale-invariant $\ell_
    
[^2]: 通过特征诱导的信息流解释时序图神经网络

    Explaining Temporal Graph Neural Networks via Feature-induced Information Flow

    [https://arxiv.org/abs/2606.27201](https://arxiv.org/abs/2606.27201)

    提出了一种基于正则化相关性度量框架的新归因方法，通过分析所有事件相关变量中的完整信息流，克服了现有方法忽略事件诱导变量路径的局限，从而更全面地解释基于事件的时序图神经网络。

    

    arXiv:2606.27201v1 公告类型：新 摘要：基于事件的时序图神经网络（ETGNNs）在社交网络分析、疫情追踪、推荐系统和政治事件预测等多种应用中表现出强大的性能。然而，其日益增长的复杂性对可解释性提出了重大挑战。现有的解释方法仅关注ETGNNs中信息流的一部分，通常追踪从事件相关嵌入到输出的贡献。因此，它们忽略了通过事件诱导变量的重要路径，这些变量介导了节点之间的交互，从而在捕捉长期时序依赖中发挥核心作用。为克服这一局限，我们提出了一种新的归因方法，分析所有事件相关变量中的完整信息流。我们的方法建立在最近的正则化相关性度量（NRM）框架之上，从而实现了可解释性。

    arXiv:2606.27201v1 Announce Type: new  Abstract: Event-based Temporal Graph Neural Networks (ETGNNs) have demonstrated strong performance across a wide range of applications, including social network analysis, epidemic tracing, recommender systems, and political event forecasting. However, their increasing complexity poses significant challenges for explainability. Existing explanation methods focus only on a subset of the information flow within ETGNNs, typically tracing contributions from the event-related embeddings to the output. Consequently, they overlook the important pathways through event-induced variables, which mediate interactions between nodes and thereby play a central role in capturing long-range temporal dependencies. To overcome this limitation, we propose a novel attribution method that analyzes the \emph{entire} information flow through all event-associated variables. Our method is built upon the recent Normalized Relevance Measure (NRM) framework, which enables expl
    
[^3]: TopCoW：基于拓扑感知解剖分割的Willis循环（CoW）在CTA和MRA中的基准测试

    TopCoW: Benchmarking Topology-Aware Anatomical Segmentation of the Circle of Willis (CoW) for CTA and MRA. (arXiv:2312.17670v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2312.17670](http://arxiv.org/abs/2312.17670)

    这项研究提出了TopCoW挑战，通过发布具有13种血管组分注释的Willis循环（CoW）数据集，并使用虚拟现实（VR）技术进行拓扑感知解剖分割，解决了手动和耗时的CoW表征问题。

    

    Willis循环（CoW）是连接大脑主要循环的重要动脉网络。其血管结构被认为影响着严重神经血管疾病的风险、严重程度和临床结果。然而，对高度变化的CoW解剖进行表征仍然是一项需要手动和耗时的专家任务。CoW通常通过两种血管造影成像模式进行成像，即磁共振血管成像（MRA）和计算机断层血管造影（CTA），但是关于CTA的CoW解剖的公共数据集及其注释非常有限。因此，我们在2023年组织了TopCoW挑战赛，并发布了一个带有注释的CoW数据集。TopCoW数据集是第一个具有13种可能的CoW血管组分的体素级注释的公共数据集，通过虚拟现实（VR）技术实现。它也是第一个带有来自同一患者的成对MRA和CTA的大型数据集。TopCoW挑战将CoW表征问题形式化为多类问题。

    The Circle of Willis (CoW) is an important network of arteries connecting major circulations of the brain. Its vascular architecture is believed to affect the risk, severity, and clinical outcome of serious neuro-vascular diseases. However, characterizing the highly variable CoW anatomy is still a manual and time-consuming expert task. The CoW is usually imaged by two angiographic imaging modalities, magnetic resonance angiography (MRA) and computed tomography angiography (CTA), but there exist limited public datasets with annotations on CoW anatomy, especially for CTA. Therefore we organized the TopCoW Challenge in 2023 with the release of an annotated CoW dataset. The TopCoW dataset was the first public dataset with voxel-level annotations for thirteen possible CoW vessel components, enabled by virtual-reality (VR) technology. It was also the first large dataset with paired MRA and CTA from the same patients. TopCoW challenge formalized the CoW characterization problem as a multiclas
    

