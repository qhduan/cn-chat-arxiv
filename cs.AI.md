# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Twin Auto-Encoder Model for Learning Separable Representation in Cyberattack Detection](https://arxiv.org/abs/2403.15509) | 提出一种新型双自动编码器模型(TAE)，通过将潜在表示转换为可分离表示来解决网络攻击检测中混合表示的问题 |
| [^2] | [The Era of Semantic Decoding](https://arxiv.org/abs/2403.14562) | 提出了一种名为语义解码的新观点，将LLM、人类输入和各种工具之间的协作过程构建为语义空间中的优化过程，促进了高效输出的构建。 |
| [^3] | [3DCoMPaT$^{++}$: An improved Large-scale 3D Vision Dataset for Compositional Recognition](https://arxiv.org/abs/2310.18511) | 3DCoMPaT$^{++}$提出了一个大规模的多模态2D/3D数据集，包含1.6亿个渲染视图的风格化三维形状，带有详细的部件实例级别标注，用于组合识别。 |
| [^4] | [Intelligent Condition Monitoring of Industrial Plants: An Overview of Methodologies and Uncertainty Management Strategies.](http://arxiv.org/abs/2401.10266) | 本论文综述了工业厂房智能状态监测和故障检测和诊断方法，重点关注了Tennessee Eastman Process。调研总结了最流行和最先进的深度学习和机器学习算法，并探讨了算法的优劣势。还讨论了不平衡数据和无标记样本等挑战，以及深度学习模型如何应对。比较了不同算法在Tennessee Eastman Process上的准确性和规格。 |

# 详细

[^1]: 双自动编码器模型用于学习网络攻击检测中的可分离表示

    Twin Auto-Encoder Model for Learning Separable Representation in Cyberattack Detection

    [https://arxiv.org/abs/2403.15509](https://arxiv.org/abs/2403.15509)

    提出一种新型双自动编码器模型(TAE)，通过将潜在表示转换为可分离表示来解决网络攻击检测中混合表示的问题

    

    表征学习在网络攻击检测等许多问题的成功中起着关键作用。大多数网络攻击检测的表征学习方法基于自动编码器（AE）模型的潜在向量。为了解决AEs表示中混合的问题，我们提出了一种称为双自动编码器（TAE）的新型模型。TAE将潜在表示确定地转换为更易区分的表示，即\textit{可分离表示}，并在输出端重建可分离表示。

    arXiv:2403.15509v1 Announce Type: cross  Abstract: Representation Learning (RL) plays a pivotal role in the success of many problems including cyberattack detection. Most of the RL methods for cyberattack detection are based on the latent vector of Auto-Encoder (AE) models. An AE transforms raw data into a new latent representation that better exposes the underlying characteristics of the input data. Thus, it is very useful for identifying cyberattacks. However, due to the heterogeneity and sophistication of cyberattacks, the representation of AEs is often entangled/mixed resulting in the difficulty for downstream attack detection models. To tackle this problem, we propose a novel mod called Twin Auto-Encoder (TAE). TAE deterministically transforms the latent representation into a more distinguishable representation namely the \textit{separable representation} and the reconstructsuct the separable representation at the output. The output of TAE called the \textit{reconstruction represe
    
[^2]: 语义解码时代

    The Era of Semantic Decoding

    [https://arxiv.org/abs/2403.14562](https://arxiv.org/abs/2403.14562)

    提出了一种名为语义解码的新观点，将LLM、人类输入和各种工具之间的协作过程构建为语义空间中的优化过程，促进了高效输出的构建。

    

    最近的研究展现了在LLM（大型语言模型）、人类输入和各种工具之间编排协作以解决LLM固有局限性的想法具有巨大潜力。我们提出了一个名为语义解码的新观点，将这些协作过程构建为语义空间中的优化过程。具体来说，我们将LLM概念化为操纵我们称之为语义标记（已知思想）的有意义信息片段的语义处理器。LLM是众多其他语义处理器之一，包括人类和工具，比如搜索引擎或代码执行器。语义处理器集体参与语义标记的动态交流，逐步构建高效输出。我们称这些在语义空间中进行优化和搜索的协同作用，为语义解码算法。这个概念与已广为研究的语义解码问题直接平行。

    arXiv:2403.14562v1 Announce Type: cross  Abstract: Recent work demonstrated great promise in the idea of orchestrating collaborations between LLMs, human input, and various tools to address the inherent limitations of LLMs. We propose a novel perspective called semantic decoding, which frames these collaborative processes as optimization procedures in semantic space. Specifically, we conceptualize LLMs as semantic processors that manipulate meaningful pieces of information that we call semantic tokens (known thoughts). LLMs are among a large pool of other semantic processors, including humans and tools, such as search engines or code executors. Collectively, semantic processors engage in dynamic exchanges of semantic tokens to progressively construct high-utility outputs. We refer to these orchestrated interactions among semantic processors, optimizing and searching in semantic space, as semantic decoding algorithms. This concept draws a direct parallel to the well-studied problem of s
    
[^3]: 3DCoMPaT$^{++}$：一个用于组合识别的改进型大规模三维视觉数据集

    3DCoMPaT$^{++}$: An improved Large-scale 3D Vision Dataset for Compositional Recognition

    [https://arxiv.org/abs/2310.18511](https://arxiv.org/abs/2310.18511)

    3DCoMPaT$^{++}$提出了一个大规模的多模态2D/3D数据集，包含1.6亿个渲染视图的风格化三维形状，带有详细的部件实例级别标注，用于组合识别。

    

    在这项工作中，我们提出了3DCoMPaT$^{++}$，这是一个包含1.6亿个以上10百万个风格化三维形状的渲染视图的多模态2D/3D数据集，这些形状在部件实例级别上进行了精心注释，并配有匹配的RGB点云、3D纹理网格、深度图和分割蒙版。3DCoMPaT$^{++}$涵盖了41个形状类别、275个细粒度部分类别和293个细粒度材料类别，这些类别可以组合应用于三维物体的各部分。我们从四个等间距视图和四个随机视图中渲染了一百万个风格化形状的子集，共计1.6亿个渲染。部件在实例级别、粗粒度和细粒度语义级别上进行了分割。我们引入了一个名为Grounded CoMPaT Recognition (GCR)的新任务，旨在共同识别和基于物体部分的材料组合。另外，我们还报告了一个数据挑战活动的结果。

    arXiv:2310.18511v2 Announce Type: replace-cross  Abstract: In this work, we present 3DCoMPaT$^{++}$, a multimodal 2D/3D dataset with 160 million rendered views of more than 10 million stylized 3D shapes carefully annotated at the part-instance level, alongside matching RGB point clouds, 3D textured meshes, depth maps, and segmentation masks. 3DCoMPaT$^{++}$ covers 41 shape categories, 275 fine-grained part categories, and 293 fine-grained material classes that can be compositionally applied to parts of 3D objects. We render a subset of one million stylized shapes from four equally spaced views as well as four randomized views, leading to a total of 160 million renderings. Parts are segmented at the instance level, with coarse-grained and fine-grained semantic levels. We introduce a new task, called Grounded CoMPaT Recognition (GCR), to collectively recognize and ground compositions of materials on parts of 3D objects. Additionally, we report the outcomes of a data challenge organized a
    
[^4]: 工业厂房智能状态监测: 方法论和不确定性管理策略综述

    Intelligent Condition Monitoring of Industrial Plants: An Overview of Methodologies and Uncertainty Management Strategies. (arXiv:2401.10266v1 [cs.LG])

    [http://arxiv.org/abs/2401.10266](http://arxiv.org/abs/2401.10266)

    本论文综述了工业厂房智能状态监测和故障检测和诊断方法，重点关注了Tennessee Eastman Process。调研总结了最流行和最先进的深度学习和机器学习算法，并探讨了算法的优劣势。还讨论了不平衡数据和无标记样本等挑战，以及深度学习模型如何应对。比较了不同算法在Tennessee Eastman Process上的准确性和规格。

    

    状态监测在现代工业系统的安全性和可靠性中起着重要作用。人工智能（AI）方法作为一种在工业应用中日益受到学术界和行业关注的增长主题和一种强大的故障识别方式。本文概述了工业厂房智能状态监测和故障检测和诊断方法，重点关注开源基准Tennessee Eastman Process（TEP）。在这项调查中，总结了用于工业厂房状态监测、故障检测和诊断的最流行和最先进的深度学习（DL）和机器学习（ML）算法，并研究了每种算法的优点和缺点。还涵盖了不平衡数据、无标记样本以及深度学习模型如何处理这些挑战。最后，比较了利用Tennessee Eastman Process的不同算法的准确性和规格。

    Condition monitoring plays a significant role in the safety and reliability of modern industrial systems. Artificial intelligence (AI) approaches are gaining attention from academia and industry as a growing subject in industrial applications and as a powerful way of identifying faults. This paper provides an overview of intelligent condition monitoring and fault detection and diagnosis methods for industrial plants with a focus on the open-source benchmark Tennessee Eastman Process (TEP). In this survey, the most popular and state-of-the-art deep learning (DL) and machine learning (ML) algorithms for industrial plant condition monitoring, fault detection, and diagnosis are summarized and the advantages and disadvantages of each algorithm are studied. Challenges like imbalanced data, unlabelled samples and how deep learning models can handle them are also covered. Finally, a comparison of the accuracies and specifications of different algorithms utilizing the Tennessee Eastman Process 
    

