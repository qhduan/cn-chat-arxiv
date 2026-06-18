# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A DeepLearning Framework for Dynamic Estimation of Origin-Destination Sequence.](http://arxiv.org/abs/2307.05623) | 本文提出了一个综合方法，使用深度学习方法推断OD序列的结构，并使用结构约束指导传统的数值优化，解决了交通领域中静态和动态OD矩阵估计中的欠定和滞后挑战。 |
| [^2] | [Simple Domain Generalization Methods are Strong Baselines for Open Domain Generalization.](http://arxiv.org/abs/2303.18031) | 该论文评估了基于领域泛化的方法在开放领域泛化中的表现，证明了CORAL和MMD等简单DG方法在某些情况下的竞争力，提出了这些方法的简单扩展。 |
| [^3] | [Enhancing Core Image Classification Using Generative Adversarial Networks (GANs).](http://arxiv.org/abs/2204.14224) | 本研究提出了一种使用生成对抗网络(GANs)增强核心图像分类的创新方法，通过应用先进的模型来检测和分割岩心图像中的核心和洞，并利用强大的GANs技术填补岩心图像中的洞。这项研究将为油气勘探行业带来重大转变。 |

# 详细

[^1]: 用于动态估计出发地-目的地序列的深度学习框架

    A DeepLearning Framework for Dynamic Estimation of Origin-Destination Sequence. (arXiv:2307.05623v1 [cs.LG])

    [http://arxiv.org/abs/2307.05623](http://arxiv.org/abs/2307.05623)

    本文提出了一个综合方法，使用深度学习方法推断OD序列的结构，并使用结构约束指导传统的数值优化，解决了交通领域中静态和动态OD矩阵估计中的欠定和滞后挑战。

    

    OD矩阵估计是交通领域的一个关键问题。主要方法使用交通传感器测量信息（如交通流量）来估计由OD矩阵表示的交通需求。该问题分为静态OD矩阵估计和动态OD矩阵序列（简称OD序列）估计两类。上述两种方法面临由于大量估计参数和不足的约束信息造成的欠定问题。此外，OD序列估计还面临滞后挑战：由于拥堵等不同交通条件，相同的车辆在同一观测时段内会出现在不同的路段上，导致相同的OD需求对应不同的行程。为此，本文提出了一种综合方法，它使用深度学习方法推断OD序列的结构，并使用结构约束指导传统的数值优化。我们的实验显示...

    OD matrix estimation is a critical problem in the transportation domain. The principle method uses the traffic sensor measured information such as traffic counts to estimate the traffic demand represented by the OD matrix. The problem is divided into two categories: static OD matrix estimation and dynamic OD matrices sequence(OD sequence for short) estimation. The above two face the underdetermination problem caused by abundant estimated parameters and insufficient constraint information. In addition, OD sequence estimation also faces the lag challenge: due to different traffic conditions such as congestion, identical vehicle will appear on different road sections during the same observation period, resulting in identical OD demands correspond to different trips. To this end, this paper proposes an integrated method, which uses deep learning methods to infer the structure of OD sequence and uses structural constraints to guide traditional numerical optimization. Our experiments show th
    
[^2]: 简单的领域泛化方法是开放领域泛化的强大基准方法

    Simple Domain Generalization Methods are Strong Baselines for Open Domain Generalization. (arXiv:2303.18031v1 [cs.CV])

    [http://arxiv.org/abs/2303.18031](http://arxiv.org/abs/2303.18031)

    该论文评估了基于领域泛化的方法在开放领域泛化中的表现，证明了CORAL和MMD等简单DG方法在某些情况下的竞争力，提出了这些方法的简单扩展。

    

    在现实世界的应用中，机器学习模型需要处理开放集识别（OSR），即在推理过程中出现未知类别，以及领域漂移（domain shift），即训练和推理阶段之间数据分布不同的情况。领域泛化（DG）旨在处理推理阶段的目标领域在模型训练期间不可访问的情况下的领域漂移情况。开放领域泛化（ODG）同时考虑了DG和OSR。领域增强元学习（DAML）是一个面向ODG的方法，但其学习过程较为复杂。另一方面，尽管提出了各种DG方法，但它们尚未在ODG情况下进行评估。本文全面评估现有的DG方法在ODG中的表现，并展示了两种简单的DG方法，即CORrelation ALignment（CORAL）和Maximum Mean Discrepancy（MMD）在若干情况下与DAML具有竞争力。此外，我们通过引入一个小调整，提出了CORAL和MMD的简单扩展。

    In real-world applications, a machine learning model is required to handle an open-set recognition (OSR), where unknown classes appear during the inference, in addition to a domain shift, where the distribution of data differs between the training and inference phases. Domain generalization (DG) aims to handle the domain shift situation where the target domain of the inference phase is inaccessible during model training. Open domain generalization (ODG) takes into account both DG and OSR. Domain-Augmented Meta-Learning (DAML) is a method targeting ODG but has a complicated learning process. On the other hand, although various DG methods have been proposed, they have not been evaluated in ODG situations. This work comprehensively evaluates existing DG methods in ODG and shows that two simple DG methods, CORrelation ALignment (CORAL) and Maximum Mean Discrepancy (MMD), are competitive with DAML in several cases. In addition, we propose simple extensions of CORAL and MMD by introducing th
    
[^3]: 通过生成对抗网络(GANs)增强核心图像分类

    Enhancing Core Image Classification Using Generative Adversarial Networks (GANs). (arXiv:2204.14224v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2204.14224](http://arxiv.org/abs/2204.14224)

    本研究提出了一种使用生成对抗网络(GANs)增强核心图像分类的创新方法，通过应用先进的模型来检测和分割岩心图像中的核心和洞，并利用强大的GANs技术填补岩心图像中的洞。这项研究将为油气勘探行业带来重大转变。

    

    在兴奋人心的油气勘探世界中，岩心样品是解锁地质信息以寻找有利可图的油气矿床的关键。尽管这些样品的重要性，传统的岩心记录技术被认为是耗时且主观的。幸运的是，该行业已经采用了一种创新的解决方案-岩心成像，它可以对大量岩心进行无损和非侵入性的快速表征。我们杰出的研究论文旨在解决岩心检测和分类的紧迫问题。使用最先进的技术，我们提出了一个突破性的解决方案，将改变该行业。我们首先面临的挑战是检测图像中的岩心并分割出孔洞，我们将分别使用Faster RCNN和Mask RCNN模型来实现。然后，我们将利用强大的生成对抗网络(GANs)和Contextual Residual来解决填补岩心图像中的洞的问题。

    In the thrilling world of oil exploration, drill core samples are key to unlocking geological information critical to finding lucrative oil deposits. Despite the importance of these samples, traditional core logging techniques are known to be laborious and, worse still, subjective. Thankfully, the industry has embraced an innovative solution core imaging that allows for nondestructive and noninvasive rapid characterization of large quantities of drill cores. Our preeminent research paper aims to tackle the pressing problem of core detection and classification. Using state-of-the-art techniques, we present a groundbreaking solution that will transform the industry. Our first challenge is detecting the cores and segmenting the holes in images, which we will achieve using the Faster RCNN and Mask RCNN models, respectively. Then, we will address the problem of filling the hole in the core image, utilizing the powerful Generative Adversarial Networks (GANs) and employing Contextual Residual
    

