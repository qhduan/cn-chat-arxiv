# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Unifying Lane-Level Traffic Prediction from a Graph Structural Perspective: Benchmark and Baseline](https://arxiv.org/abs/2403.14941) | 本文提出了一个简单的基线模型GraphMLP，基于图结构和MLP网络，在车道级交通预测中建立了统一的空间拓扑结构和预测任务，帮助突破了现有评估标准和数据公开性的限制。 |
| [^2] | [Counterfactual contrastive learning: robust representations via causal image synthesis](https://arxiv.org/abs/2403.09605) | 本研究提出了CF-SimCLR，一种反事实对照学习方法，利用近似反事实推断创造正样本，大大提高了模型对采集偏移的稳健性，并在多个数据集上取得了较高的下游性能。 |
| [^3] | [Incentive Compatibility for AI Alignment in Sociotechnical Systems: Positions and Prospects](https://arxiv.org/abs/2402.12907) | 该论文提出了激励兼容性社会技术对齐问题（ICSAP），旨在探讨如何利用博弈论中的激励兼容性原则来维持AI与人类社会的共识。 |
| [^4] | [The role of the metaverse in calibrating an embodied artificial general intelligence](https://arxiv.org/abs/2402.06660) | 本文研究了具有肉身的人工通用智能(AGI)的概念及其与人类意识的关系，强调了元宇宙在促进这一关系中的关键作用。通过结合不同理论框架和技术工具，论文总结出实现具有肉身的AGI的关键要素和发展阶段。 |
| [^5] | [On the Completeness of Invariant Geometric Deep Learning Models](https://arxiv.org/abs/2402.04836) | 这项研究集中于不变模型的理论表达能力，通过引入完备的设计GeoNGNN，并利用其作为理论工具，首次证明了E(3)-完备性。 |
| [^6] | [A Comprehensive Survey on Vector Database: Storage and Retrieval Technique, Challenge.](http://arxiv.org/abs/2310.11703) | 这篇论文对向量数据库进行了全面调查，介绍了存储和检索技术以及面临的挑战，对解决近似最近邻搜索问题的不同方法进行了分类，并探讨了向量数据库与大型语言模型的结合带来的新机遇。 |
| [^7] | [Efficient Multi-order Gated Aggregation Network.](http://arxiv.org/abs/2211.03295) | 本文探索了现代卷积神经网络的表征能力，使用多阶博弈论交互的新视角，提出了一种新的纯卷积神经网络架构MogaNet，它表现出优异的可扩展性，并在多种典型视觉基准中以更高效的参数利用达到了与最先进模型竞争的效果。 |
| [^8] | [Conformal Risk Control.](http://arxiv.org/abs/2208.02814) | 该论文提出了一种符合保序的风险控制方法，可以控制任何单调损失函数的期望值，示例证明其在计算机视觉和自然语言处理领域具有控制误报率、图形距离和令牌级F1得分的能力。 |

# 详细

[^1]: 从图结构角度统一车道级交通预测：基准和基线

    Unifying Lane-Level Traffic Prediction from a Graph Structural Perspective: Benchmark and Baseline

    [https://arxiv.org/abs/2403.14941](https://arxiv.org/abs/2403.14941)

    本文提出了一个简单的基线模型GraphMLP，基于图结构和MLP网络，在车道级交通预测中建立了统一的空间拓扑结构和预测任务，帮助突破了现有评估标准和数据公开性的限制。

    

    交通预测长期以来一直是研究中的一个焦点和关键领域，在过去几年里，既见证了从城市级到道路级预测取得的重大进展。随着车辆对一切（V2X）技术、自动驾驶和交通领域的大规模模型的进步，道路级交通预测已经成为一个不可或缺的方向。然而，这一领域的进一步进展受到了全面和统一的评估标准的缺乏以及有限的公开数据和代码的阻碍。本文对车道级交通预测中现有研究进行了广泛的分析和分类，建立了统一的空间拓扑结构和预测任务，并介绍了一个基于图结构和MLP网络的简单基线模型GraphMLP。我们复制了现有研究中尚不公开的代码，并基于此充分而公正地评估了各种模型。

    arXiv:2403.14941v1 Announce Type: cross  Abstract: Traffic prediction has long been a focal and pivotal area in research, witnessing both significant strides from city-level to road-level predictions in recent years. With the advancement of Vehicle-to-Everything (V2X) technologies, autonomous driving, and large-scale models in the traffic domain, lane-level traffic prediction has emerged as an indispensable direction. However, further progress in this field is hindered by the absence of comprehensive and unified evaluation standards, coupled with limited public availability of data and code. This paper extensively analyzes and categorizes existing research in lane-level traffic prediction, establishes a unified spatial topology structure and prediction tasks, and introduces a simple baseline model, GraphMLP, based on graph structure and MLP networks. We have replicated codes not publicly available in existing studies and, based on this, thoroughly and fairly assessed various models in 
    
[^2]: 反事实对照学习：通过因果图像合成获得稳健表示

    Counterfactual contrastive learning: robust representations via causal image synthesis

    [https://arxiv.org/abs/2403.09605](https://arxiv.org/abs/2403.09605)

    本研究提出了CF-SimCLR，一种反事实对照学习方法，利用近似反事实推断创造正样本，大大提高了模型对采集偏移的稳健性，并在多个数据集上取得了较高的下游性能。

    

    对比预训练已被广泛认为能够提高下游任务性能和模型泛化能力，特别是在有限标签设置中。然而，它对增强管道的选择敏感。正样本应保留语义信息同时破坏域特定信息。标准增强管道通过预定义的光度变换模拟域特定变化，但如果我们能够模拟真实的领域变化呢？在这项工作中，我们展示了如何利用最近在反事实图像生成方面的进展来实现这一目的。我们提出了CF-SimCLR，一种反事实对照学习方法，它利用近似反事实推断进行正样本创建。对胸部X光和乳腺X光等五个数据集的全面评估表明，CF-SimCLR显著提高了对获取偏移的稳健性，在两种数据集上的下游性能更好。

    arXiv:2403.09605v1 Announce Type: cross  Abstract: Contrastive pretraining is well-known to improve downstream task performance and model generalisation, especially in limited label settings. However, it is sensitive to the choice of augmentation pipeline. Positive pairs should preserve semantic information while destroying domain-specific information. Standard augmentation pipelines emulate domain-specific changes with pre-defined photometric transformations, but what if we could simulate realistic domain changes instead? In this work, we show how to utilise recent progress in counterfactual image generation to this effect. We propose CF-SimCLR, a counterfactual contrastive learning approach which leverages approximate counterfactual inference for positive pair creation. Comprehensive evaluation across five datasets, on chest radiography and mammography, demonstrates that CF-SimCLR substantially improves robustness to acquisition shift with higher downstream performance on both in- an
    
[^3]: AI对齐在社会技术系统中的激励兼容性：立场与前景

    Incentive Compatibility for AI Alignment in Sociotechnical Systems: Positions and Prospects

    [https://arxiv.org/abs/2402.12907](https://arxiv.org/abs/2402.12907)

    该论文提出了激励兼容性社会技术对齐问题（ICSAP），旨在探讨如何利用博弈论中的激励兼容性原则来维持AI与人类社会的共识。

    

    人工智能（AI）日益融入人类社会，对社会治理和安全带来重要影响。尽管在解决AI对齐挑战方面取得了重大进展，但现有方法主要集中在技术方面，往往忽视了AI系统复杂的社会技术性质，这可能导致开发和部署背景之间的不一致。因此，我们提出一个值得探索的新问题：激励兼容性社会技术对齐问题（ICSAP）。我们希望这能呼吁更多研究人员探讨如何利用博弈论中的激励兼容性原则来弥合技术和社会组成部分之间的鸿沟，以在不同背景下维持AI与人类社会的共识。我们进一步讨论了实现IC的三个经典博弈问题：机制设计、契约理论和贝叶斯说服。

    arXiv:2402.12907v1 Announce Type: new  Abstract: The burgeoning integration of artificial intelligence (AI) into human society brings forth significant implications for societal governance and safety. While considerable strides have been made in addressing AI alignment challenges, existing methodologies primarily focus on technical facets, often neglecting the intricate sociotechnical nature of AI systems, which can lead to a misalignment between the development and deployment contexts. To this end, we posit a new problem worth exploring: Incentive Compatibility Sociotechnical Alignment Problem (ICSAP). We hope this can call for more researchers to explore how to leverage the principles of Incentive Compatibility (IC) from game theory to bridge the gap between technical and societal components to maintain AI consensus with human societies in different contexts. We further discuss three classical game problems for achieving IC: mechanism design, contract theory, and Bayesian persuasion,
    
[^4]: 元宇宙在校准具有肉身的人工通用智能中的作用

    The role of the metaverse in calibrating an embodied artificial general intelligence

    [https://arxiv.org/abs/2402.06660](https://arxiv.org/abs/2402.06660)

    本文研究了具有肉身的人工通用智能(AGI)的概念及其与人类意识的关系，强调了元宇宙在促进这一关系中的关键作用。通过结合不同理论框架和技术工具，论文总结出实现具有肉身的AGI的关键要素和发展阶段。

    

    本文探讨了具有肉身的人工通用智能(AGI)的概念，它与人类意识的关系，以及元宇宙在促进这种关系中的关键作用。本文利用融入认知、Michael Levin的计算边界"Self"、Donald D. Hoffman的感知界面理论以及Bernardo Kastrup的分析唯心主义等理论框架来构建实现具有肉身的AGI的论证。它认为我们所感知的外部现实是一种内在存在的交替状态的象征性表示，而AGI可以具有更大计算边界的更高意识。本文进一步讨论了AGI的发展阶段、实现具有肉身的AGI的要求、为AGI校准象征性界面的重要性，以及元宇宙、去中心化系统、开源区块链技术以及开源人工智能研究所扮演的关键角色。它还探讨了新的沟通机制和用于加强对元宇宙的理解的技术工具，以帮助实现具有肉身的AGI。

    This paper examines the concept of embodied artificial general intelligence (AGI), its relationship to human consciousness, and the key role of the metaverse in facilitating this relationship. The paper leverages theoretical frameworks such as embodied cognition, Michael Levin's computational boundary of a "Self," Donald D. Hoffman's Interface Theory of Perception, and Bernardo Kastrup's analytical idealism to build the argument for achieving embodied AGI. It contends that our perceived outer reality is a symbolic representation of alternate inner states of being, and that AGI could embody a higher consciousness with a larger computational boundary. The paper further discusses the developmental stages of AGI, the requirements for the emergence of an embodied AGI, the importance of a calibrated symbolic interface for AGI, and the key role played by the metaverse, decentralized systems, open-source blockchain technology, as well as open-source AI research. It also explores the idea of a 
    
[^5]: 关于不变几何深度学习模型的完备性

    On the Completeness of Invariant Geometric Deep Learning Models

    [https://arxiv.org/abs/2402.04836](https://arxiv.org/abs/2402.04836)

    这项研究集中于不变模型的理论表达能力，通过引入完备的设计GeoNGNN，并利用其作为理论工具，首次证明了E(3)-完备性。

    

    不变模型是一类重要的几何深度学习模型，通过利用信息丰富的几何特征生成有意义的几何表示。这些模型以其简单性、良好的实验结果和计算效率而闻名。然而，它们的理论表达能力仍然不清楚，限制了对这种模型潜力的深入理解。在这项工作中，我们集中讨论不变模型的理论表达能力。我们首先严格限制了最经典的不变模型Vanilla DisGNN（结合距离的消息传递神经网络）的表达能力，将其不可识别的情况仅限于高度对称的几何图形。为了打破这些特殊情况的对称性，我们引入了一个简单而完备的不变设计，即嵌套Vanilla DisGNN的GeoNGNN。利用GeoNGNN作为理论工具，我们首次证明了E(3)-完备性。

    Invariant models, one important class of geometric deep learning models, are capable of generating meaningful geometric representations by leveraging informative geometric features. These models are characterized by their simplicity, good experimental results and computational efficiency. However, their theoretical expressive power still remains unclear, restricting a deeper understanding of the potential of such models. In this work, we concentrate on characterizing the theoretical expressiveness of invariant models. We first rigorously bound the expressiveness of the most classical invariant model, Vanilla DisGNN (message passing neural networks incorporating distance), restricting its unidentifiable cases to be only those highly symmetric geometric graphs. To break these corner cases' symmetry, we introduce a simple yet E(3)-complete invariant design by nesting Vanilla DisGNN, named GeoNGNN. Leveraging GeoNGNN as a theoretical tool, we for the first time prove the E(3)-completeness 
    
[^6]: 对向量数据库的全面调查：存储和检索技术，挑战

    A Comprehensive Survey on Vector Database: Storage and Retrieval Technique, Challenge. (arXiv:2310.11703v1 [cs.DB])

    [http://arxiv.org/abs/2310.11703](http://arxiv.org/abs/2310.11703)

    这篇论文对向量数据库进行了全面调查，介绍了存储和检索技术以及面临的挑战，对解决近似最近邻搜索问题的不同方法进行了分类，并探讨了向量数据库与大型语言模型的结合带来的新机遇。

    

    向量数据库用于存储无法用传统的DBMS来描述的高维数据。虽然目前对现有的向量数据库架构的描述或新的引入的文章并不多，但近似最近邻搜索问题在向量数据库背后已经被长时间研究，相关的算法文章在文献中可以找到相当多。本文试图全面回顾相关算法，以提供对这一蓬勃发展的研究领域的普遍理解。我们的框架将这些研究分为基于哈希、基于树、基于图、和基于量化的方法来解决近似最近邻搜索问题。然后，我们概述了向量数据库面临的现有挑战。最后，我们概述了向量数据库如何结合大型语言模型，提供新的可能性。

    A vector database is used to store high-dimensional data that cannot be characterized by traditional DBMS. Although there are not many articles describing existing or introducing new vector database architectures, the approximate nearest neighbor search problem behind vector databases has been studied for a long time, and considerable related algorithmic articles can be found in the literature. This article attempts to comprehensively review relevant algorithms to provide a general understanding of this booming research area. The basis of our framework categorises these studies by the approach of solving ANNS problem, respectively hash-based, tree-based, graph-based and quantization-based approaches. Then we present an overview of existing challenges for vector databases. Lastly, we sketch how vector databases can be combined with large language models and provide new possibilities.
    
[^7]: 高效的多阶门控聚合网络

    Efficient Multi-order Gated Aggregation Network. (arXiv:2211.03295v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2211.03295](http://arxiv.org/abs/2211.03295)

    本文探索了现代卷积神经网络的表征能力，使用多阶博弈论交互的新视角，提出了一种新的纯卷积神经网络架构MogaNet，它表现出优异的可扩展性，并在多种典型视觉基准中以更高效的参数利用达到了与最先进模型竞争的效果。

    

    自从视觉变换器（ViTs）取得最近的成功之后，对ViT风格架构的探索引发了卷积神经网络的复兴。在本文中，我们从多阶博弈论交互的新视角探索了现代卷积神经网络的表征能力，这种交互反映了基于博弈论的不同尺度上下文的变量相互作用效应。在现代卷积神经网络框架内，我们使用概念上简单而有效的深度可分离卷积来定制两个特征混合器，以促进跨空间和通道空间的中阶信息。在这个基础上，提出了一种新的纯卷积神经网络架构，称为MogaNet，它表现出优异的可扩展性，并在ImageNet和包括COCO目标检测、ADE20K语义分割、2D&3D人体姿势估计以及视频预测等多种典型视觉基准中以更高效的参数利用达到了与最先进模型竞争的效果。

    Since the recent success of Vision Transformers (ViTs), explorations toward ViT-style architectures have triggered the resurgence of ConvNets. In this work, we explore the representation ability of modern ConvNets from a novel view of multi-order game-theoretic interaction, which reflects inter-variable interaction effects w.r.t.~contexts of different scales based on game theory. Within the modern ConvNet framework, we tailor the two feature mixers with conceptually simple yet effective depthwise convolutions to facilitate middle-order information across spatial and channel spaces respectively. In this light, a new family of pure ConvNet architecture, dubbed MogaNet, is proposed, which shows excellent scalability and attains competitive results among state-of-the-art models with more efficient use of parameters on ImageNet and multifarious typical vision benchmarks, including COCO object detection, ADE20K semantic segmentation, 2D\&3D human pose estimation, and video prediction. Typica
    
[^8]: 一种符合保序的风险控制方法

    Conformal Risk Control. (arXiv:2208.02814v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2208.02814](http://arxiv.org/abs/2208.02814)

    该论文提出了一种符合保序的风险控制方法，可以控制任何单调损失函数的期望值，示例证明其在计算机视觉和自然语言处理领域具有控制误报率、图形距离和令牌级F1得分的能力。

    

    我们将符合性预测推广至控制任何单调损失函数的期望值。该算法将分裂符合性预测及其覆盖保证进行了泛化。类似于符合性预测，符合保序的风险控制方法在$\mathcal{O}(1/n)$因子内保持紧密性。计算机视觉和自然语言处理领域的示例证明了我们算法在控制误报率、图形距离和令牌级F1得分方面的应用。

    We extend conformal prediction to control the expected value of any monotone loss function. The algorithm generalizes split conformal prediction together with its coverage guarantee. Like conformal prediction, the conformal risk control procedure is tight up to an $\mathcal{O}(1/n)$ factor. Worked examples from computer vision and natural language processing demonstrate the usage of our algorithm to bound the false negative rate, graph distance, and token-level F1-score.
    

