# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Knowledge Navigation: Inferring the Interlocking Map of Knowledge from Research Trajectories.](http://arxiv.org/abs/2401.11742) | 本研究利用自然语言处理技术引入了一种创新的嵌入方案，推断出了知识交错地图，揭示了知识之间错综复杂的联系，并展示了多个应用场景。 |
| [^2] | [Towards Automatic Sampling of User Behaviors for Sequential Recommender Systems.](http://arxiv.org/abs/2311.00388) | 本论文提出了一个名为AutoSAM的自动采样框架，用于对连续推荐系统中的用户行为进行非均匀处理。该框架通过自适应地学习历史行为的偏斜分布，并采样出信息丰富的子集，以构建更具可泛化性的连续推荐系统。 |
| [^3] | [CompoDiff: Versatile Composed Image Retrieval With Latent Diffusion.](http://arxiv.org/abs/2303.11916) | CompoDiff 是一种多功能的组合图像检索模型，通过接受各种条件，具有潜在扩散的能力，并在 FashionIQ 上实现了新的零样本最新技术水平。其特征位于完整的 CLIP 嵌入空间中，可以直接用于所有利用 CLIP 空间的模型。 |

# 详细

[^1]: 知识导航：从研究轨迹中推断知识的交错地图

    Knowledge Navigation: Inferring the Interlocking Map of Knowledge from Research Trajectories. (arXiv:2401.11742v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2401.11742](http://arxiv.org/abs/2401.11742)

    本研究利用自然语言处理技术引入了一种创新的嵌入方案，推断出了知识交错地图，揭示了知识之间错综复杂的联系，并展示了多个应用场景。

    

    "如果我看得更远，那是因为我站在巨人的肩膀上。"艾萨克·牛顿的著名声明暗示了新知识建立在现有基础之上的事实，这意味着知识之间存在着相互依赖的关系，而这种关系在科学体系的历史发展中一直未被揭示。通过利用自然语言处理技术，本研究引入了一种创新的嵌入方案，旨在推断“知识交错地图”。这个地图是从数百万学者的研究轨迹中推导出来的，揭示了知识之间错综复杂的联系。我们验证了推断出的地图有效地勾画了学科边界，并捕捉到了不同概念之间复杂的关系。交错地图的实用性通过多个应用展示出来。首先，我们展示了在知识空间中的多步类比推理和概念之间的功能连接。

    "If I have seen further, it is by standing on the shoulders of giants," Isaac Newton's renowned statement hints that new knowledge builds upon existing foundations, which means there exists an interdependent relationship between knowledge, which, yet uncovered, is implied in the historical development of scientific systems for hundreds of years. By leveraging natural language processing techniques, this study introduces an innovative embedding scheme designed to infer the "knowledge interlocking map." This map, derived from the research trajectories of millions of scholars, reveals the intricate connections among knowledge. We validate that the inferred map effectively delineates disciplinary boundaries and captures the intricate relationships between diverse concepts. The utility of the interlocking map is showcased through multiple applications. Firstly, we demonstrated the multi-step analogy inferences within the knowledge space and the functional connectivity between concepts in di
    
[^2]: 实现自动采样对于连续推荐系统中用户行为的研究

    Towards Automatic Sampling of User Behaviors for Sequential Recommender Systems. (arXiv:2311.00388v1 [cs.IR])

    [http://arxiv.org/abs/2311.00388](http://arxiv.org/abs/2311.00388)

    本论文提出了一个名为AutoSAM的自动采样框架，用于对连续推荐系统中的用户行为进行非均匀处理。该框架通过自适应地学习历史行为的偏斜分布，并采样出信息丰富的子集，以构建更具可泛化性的连续推荐系统。

    

    由于连续推荐系统能够有效捕捉动态用户偏好，因此它们在推荐领域中广受欢迎。当前连续推荐系统的一个默认设置是将每个历史行为均匀地视为正向交互。然而，实际上，这种设置有可能导致性能不佳，因为每个商品对用户的兴趣有不同的贡献。例如，购买的商品应该比点击的商品更重要。因此，我们提出了一个通用的自动采样框架，名为AutoSAM，用于非均匀地处理历史行为。具体而言，AutoSAM通过在标准的连续推荐架构中增加一个采样器层，自适应地学习原始输入的偏斜分布，并采样出信息丰富的子集，以构建更具可泛化性的连续推荐系统。为了克服非可微分采样操作的挑战，同时引入多个决策因素进行采样，我们还提出了进一步的方法。

    Sequential recommender systems (SRS) have gained widespread popularity in recommendation due to their ability to effectively capture dynamic user preferences. One default setting in the current SRS is to uniformly consider each historical behavior as a positive interaction. Actually, this setting has the potential to yield sub-optimal performance, as each item makes a distinct contribution to the user's interest. For example, purchased items should be given more importance than clicked ones. Hence, we propose a general automatic sampling framework, named AutoSAM, to non-uniformly treat historical behaviors. Specifically, AutoSAM augments the standard sequential recommendation architecture with an additional sampler layer to adaptively learn the skew distribution of the raw input, and then sample informative sub-sets to build more generalizable SRS. To overcome the challenges of non-differentiable sampling actions and also introduce multiple decision factors for sampling, we further int
    
[^3]: CompoDiff: 基于潜在扩散的多功能组合图像检索

    CompoDiff: Versatile Composed Image Retrieval With Latent Diffusion. (arXiv:2303.11916v1 [cs.CV])

    [http://arxiv.org/abs/2303.11916](http://arxiv.org/abs/2303.11916)

    CompoDiff 是一种多功能的组合图像检索模型，通过接受各种条件，具有潜在扩散的能力，并在 FashionIQ 上实现了新的零样本最新技术水平。其特征位于完整的 CLIP 嵌入空间中，可以直接用于所有利用 CLIP 空间的模型。

    

    本文提出了一种新颖的基于扩散的模型 CompoDiff，用于解决具有潜在扩散的组合图像检索（CIR）问题，并提供了一个由 1800 万个参考图像、条件和相应的目标图像三元组组成的新数据集，用于训练模型。CompoDiff 不仅在像 FashionIQ 这样的 CIR 基准测试上实现了新的零样本最新技术水平，而且还通过接收各种条件（如负文本和图像遮罩条件），使得 CIR 更加多功能，这是现有 CIR 方法所不具备的。此外，CompoDiff 特征位于完整的 CLIP 嵌入空间中，因此它们可以直接用于利用 CLIP 空间的所有现有模型。训练所使用的代码和数据集，以及预训练权重可在 https://github.com/navervision/CompoDiff 上获得。

    This paper proposes a novel diffusion-based model, CompoDiff, for solving Composed Image Retrieval (CIR) with latent diffusion and presents a newly created dataset of 18 million reference images, conditions, and corresponding target image triplets to train the model. CompoDiff not only achieves a new zero-shot state-of-the-art on a CIR benchmark such as FashionIQ but also enables a more versatile CIR by accepting various conditions, such as negative text and image mask conditions, which are unavailable with existing CIR methods. In addition, the CompoDiff features are on the intact CLIP embedding space so that they can be directly used for all existing models exploiting the CLIP space. The code and dataset used for the training, and the pre-trained weights are available at https://github.com/navervision/CompoDiff
    

