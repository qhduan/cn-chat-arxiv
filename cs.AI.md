# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multimodal Sentiment Analysis with Missing Modality: A Knowledge-Transfer Approach.](http://arxiv.org/abs/2401.10747) | 本文提出了一种知识迁移方法，用于在缺失模态下进行多模态情感分析。通过翻译不同模态之间的内容以重构缺失的音频模态，并利用跨模态注意机制进行情感预测，实验证明了该方法在多个数据集上表现出显著的改进和与完整多模态监督方法相媲美的效果。 |
| [^2] | [Neural Operator: Is data all you need to model the world? An insight into the impact of Physics Informed Machine Learning.](http://arxiv.org/abs/2301.13331) | 本文探讨了如何将数据驱动方法与传统技术相结合，以解决工程和物理问题，并指出了机器学习方法的一些主要问题。 |

# 详细

[^1]: 缺失模态下的多模态情感分析:一种知识迁移方法

    Multimodal Sentiment Analysis with Missing Modality: A Knowledge-Transfer Approach. (arXiv:2401.10747v1 [cs.SD])

    [http://arxiv.org/abs/2401.10747](http://arxiv.org/abs/2401.10747)

    本文提出了一种知识迁移方法，用于在缺失模态下进行多模态情感分析。通过翻译不同模态之间的内容以重构缺失的音频模态，并利用跨模态注意机制进行情感预测，实验证明了该方法在多个数据集上表现出显著的改进和与完整多模态监督方法相媲美的效果。

    

    多模态情感分析旨在通过视觉、语言和声音线索来识别个体表达的情绪。然而，现有研究大多假设在训练和测试过程中所有模态都是可用的，这使得它们的算法容易受到缺失模态的影响。在本文中，我们提出了一种新颖的知识迁移网络，用于在不同模态之间进行翻译，以重构缺失的音频模态。此外，我们还开发了一种跨模态注意机制，以保留重构和观察到的模态的最大信息，用于情感预测。在三个公开数据集上进行的大量实验证明了相对于基线算法的显著改进，并实现了与具有完整多模态监督的先前方法相媲美的结果。

    Multimodal sentiment analysis aims to identify the emotions expressed by individuals through visual, language, and acoustic cues. However, most of the existing research efforts assume that all modalities are available during both training and testing, making their algorithms susceptible to the missing modality scenario. In this paper, we propose a novel knowledge-transfer network to translate between different modalities to reconstruct the missing audio modalities. Moreover, we develop a cross-modality attention mechanism to retain the maximal information of the reconstructed and observed modalities for sentiment prediction. Extensive experiments on three publicly available datasets demonstrate significant improvements over baselines and achieve comparable results to the previous methods with complete multi-modality supervision.
    
[^2]: 神经操作员：数据是否足以模拟世界？对物理启示机器学习影响的洞察

    Neural Operator: Is data all you need to model the world? An insight into the impact of Physics Informed Machine Learning. (arXiv:2301.13331v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2301.13331](http://arxiv.org/abs/2301.13331)

    本文探讨了如何将数据驱动方法与传统技术相结合，以解决工程和物理问题，并指出了机器学习方法的一些主要问题。

    

    常常使用偏微分方程（PDE）的数值近似来构建解决物理、工程和数学问题的方案，这些问题涉及到多个变量的函数，比如热传导或声音传播、流体流动、弹性、静电学、电动力学等。虽然这在解决许多复杂现象方面发挥了作用，但存在一些限制。常规方法如有限元法（FEM）和有限差分法（FDM）需要大量时间且计算成本高。相比之下，数据驱动的基于神经网络的方法提供了一种更快速、相对准确的替代方案，并具有离散不变性和分辨率不变性等优势。本文旨在深入了解数据驱动方法如何与传统技术相辅相成，解决工程和物理问题，同时指出机器学习方法的一些主要问题。

    Numerical approximations of partial differential equations (PDEs) are routinely employed to formulate the solution of physics, engineering and mathematical problems involving functions of several variables, such as the propagation of heat or sound, fluid flow, elasticity, electrostatics, electrodynamics, and more. While this has led to solving many complex phenomena, there are some limitations. Conventional approaches such as Finite Element Methods (FEMs) and Finite Differential Methods (FDMs) require considerable time and are computationally expensive. In contrast, data driven machine learning-based methods such as neural networks provide a faster, fairly accurate alternative, and have certain advantages such as discretization invariance and resolution invariance. This article aims to provide a comprehensive insight into how data-driven approaches can complement conventional techniques to solve engineering and physics problems, while also noting some of the major pitfalls of machine l
    

