# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An In-depth Evaluation of GPT-4 in Sentence Simplification with Error-based Human Assessment](https://arxiv.org/abs/2403.04963) | 本文深入评估了GPT-4在句子简化中的表现，指出现有自动评估指标和人类评估方法对于大型语言模型的适用性仍有待进一步研究。 |
| [^2] | [SoK: Challenges and Opportunities in Federated Unlearning](https://arxiv.org/abs/2403.02437) | 联邦学习引入了新的隐私要求，促使研究开始关注适用于联邦学习环境的反学习机制。 |
| [^3] | [Ising on the Graph: Task-specific Graph Subsampling via the Ising Model](https://arxiv.org/abs/2402.10206) | 该论文提出了一种基于伊辛模型的图子抽样方法，可以针对特定任务在图结构上进行减小，并通过学习伊辛模型的外部磁场来实现。该方法的多功能性在图像分割、三维形状稀疏化和稀疏逼近矩阵求逆等应用中得到展示。 |
| [^4] | [Relating Implicit Bias and Adversarial Attacks through Intrinsic Dimension.](http://arxiv.org/abs/2305.15203) | 本文通过研究神经网络的隐性偏差，着眼于其中涉及的傅里叶频率与图像分类和对抗性攻击之间的关系。研究提出了一种新方法，可以发现这些频率之间的非线性相关性。 |

# 详细

[^1]: 在基于错误的人类评估中深入评估GPT-4在句子简化中的表现

    An In-depth Evaluation of GPT-4 in Sentence Simplification with Error-based Human Assessment

    [https://arxiv.org/abs/2403.04963](https://arxiv.org/abs/2403.04963)

    本文深入评估了GPT-4在句子简化中的表现，指出现有自动评估指标和人类评估方法对于大型语言模型的适用性仍有待进一步研究。

    

    句子简化是一种重写句子以便更易阅读和理解的方法，对于帮助有各种阅读难题的人来说是一种有前途的技术。随着先进大型语言模型（LLMs）的兴起，评估它们在句子简化中的表现变得迫在眉睫。最近的研究利用自动评估指标和人类评估来评估LLMs的简化能力。然而，现有评估方法对LLMs在简化评估中的适用性仍然存在疑问。首先，现有自动指标在LLMs的简化评估中的适用性仍不确定。其次，当前在句子简化中的人类评估方法通常陷入两个极端：要么过于肤浅，无法清晰理解模型的表现，要么过于详细，使注释过程复杂且容易出现不一致性，从而影响评估的可靠性。

    arXiv:2403.04963v1 Announce Type: cross  Abstract: Sentence simplification, which rewrites a sentence to be easier to read and understand, is a promising technique to help people with various reading difficulties. With the rise of advanced large language models (LLMs), evaluating their performance in sentence simplification has become imperative. Recent studies have used both automatic metrics and human evaluations to assess the simplification abilities of LLMs. However, the suitability of existing evaluation methodologies for LLMs remains in question. First, the suitability of current automatic metrics on LLMs' simplification evaluation is still uncertain. Second, current human evaluation approaches in sentence simplification often fall into two extremes: they are either too superficial, failing to offer a clear understanding of the models' performance, or overly detailed, making the annotation process complex and prone to inconsistency, which in turn affects the evaluation's reliabil
    
[^2]: SoK: 联邦反学习中的挑战与机遇

    SoK: Challenges and Opportunities in Federated Unlearning

    [https://arxiv.org/abs/2403.02437](https://arxiv.org/abs/2403.02437)

    联邦学习引入了新的隐私要求，促使研究开始关注适用于联邦学习环境的反学习机制。

    

    引入于2017年的联邦学习（FL）促进了不信任方之间的合作学习，无需各方明确共享其数据。这允许在尊重GDPR和CPRA等隐私规定的同时，在用户数据上训练模型。然而，新兴的隐私要求可能要求模型所有者能够“遗忘”一些已学习的数据，例如当数据所有者或执法机构要求时。这催生了一个名为“机器反学习”的活跃研究领域。在FL的背景下，许多为集中式环境开发的反学习技术并不容易应用！这是由于FL中集中式和分布式学习之间的独特差异，特别是互动性、随机性、异构性和有限可访问性。为应对这一挑战，最近的一系列研究工作聚焦于开发适用于FL的反学习机制。

    arXiv:2403.02437v1 Announce Type: cross  Abstract: Federated learning (FL), introduced in 2017, facilitates collaborative learning between non-trusting parties with no need for the parties to explicitly share their data among themselves. This allows training models on user data while respecting privacy regulations such as GDPR and CPRA. However, emerging privacy requirements may mandate model owners to be able to \emph{forget} some learned data, e.g., when requested by data owners or law enforcement. This has given birth to an active field of research called \emph{machine unlearning}. In the context of FL, many techniques developed for unlearning in centralized settings are not trivially applicable! This is due to the unique differences between centralized and distributed learning, in particular, interactivity, stochasticity, heterogeneity, and limited accessibility in FL. In response, a recent line of work has focused on developing unlearning mechanisms tailored to FL.   This SoK pape
    
[^3]: 异构图上基于伊辛模型的特定任务图子抽样

    Ising on the Graph: Task-specific Graph Subsampling via the Ising Model

    [https://arxiv.org/abs/2402.10206](https://arxiv.org/abs/2402.10206)

    该论文提出了一种基于伊辛模型的图子抽样方法，可以针对特定任务在图结构上进行减小，并通过学习伊辛模型的外部磁场来实现。该方法的多功能性在图像分割、三维形状稀疏化和稀疏逼近矩阵求逆等应用中得到展示。

    

    减少图的大小同时保持其整体结构是一个具有许多应用的重要问题。通常，减小图的方法要么删除边缘（稀疏化），要么合并节点（粗化），而没有特定的下游任务。在本文中，我们提出了一种使用在节点或边上定义的伊辛模型对图结构进行子抽样的方法，并使用图神经网络学习伊辛模型的外部磁场。我们的方法是任务特定的，因为它可以端到端地学习如何为特定的下游任务减小图的大小。所使用的任务损失函数甚至不需要可微分性。我们在三个不同的应用上展示了我们方法的多功能性：图像分割、三维形状稀疏化和稀疏逼近矩阵求逆。

    arXiv:2402.10206v1 Announce Type: cross  Abstract: Reducing a graph while preserving its overall structure is an important problem with many applications. Typically, the reduction approaches either remove edges (sparsification) or merge nodes (coarsening) in an unsupervised way with no specific downstream task in mind. In this paper, we present an approach for subsampling graph structures using an Ising model defined on either the nodes or edges and learning the external magnetic field of the Ising model using a graph neural network. Our approach is task-specific as it can learn how to reduce a graph for a specific downstream task in an end-to-end fashion. The utilized loss function of the task does not even have to be differentiable. We showcase the versatility of our approach on three distinct applications: image segmentation, 3D shape sparsification, and sparse approximate matrix inverse determination.
    
[^4]: 通过内在维度将隐性偏见和对抗性攻击相关联

    Relating Implicit Bias and Adversarial Attacks through Intrinsic Dimension. (arXiv:2305.15203v1 [cs.LG])

    [http://arxiv.org/abs/2305.15203](http://arxiv.org/abs/2305.15203)

    本文通过研究神经网络的隐性偏差，着眼于其中涉及的傅里叶频率与图像分类和对抗性攻击之间的关系。研究提出了一种新方法，可以发现这些频率之间的非线性相关性。

    

    尽管神经网络在分类方面表现出色，但众所周知它们易受对抗性攻击的影响。这些攻击是针对模型的输入数据进行的小干扰，旨在欺骗模型。自然而然的问题是，模型的结构、设置或属性与攻击的性质之间可能存在潜在联系。在本文中，我们旨在通过关注神经网络的隐性偏差来解决这个问题，这指的是其固有倾向于支持特定模式或结果。具体而言，我们研究了隐性偏差的一个方面，其中包括进行准确图像分类所需的基本傅里叶频率。我们进行测试以评估这些频率与成功攻击所需的频率之间的统计关系。为了深入探讨这种关系，我们提出了一种新的方法，可以揭示坐标集之间的非线性相关性，在我们的情况下，这些坐标集就是前述的傅里叶频率。

    Despite their impressive performance in classification, neural networks are known to be vulnerable to adversarial attacks. These attacks are small perturbations of the input data designed to fool the model. Naturally, a question arises regarding the potential connection between the architecture, settings, or properties of the model and the nature of the attack. In this work, we aim to shed light on this problem by focusing on the implicit bias of the neural network, which refers to its inherent inclination to favor specific patterns or outcomes. Specifically, we investigate one aspect of the implicit bias, which involves the essential Fourier frequencies required for accurate image classification. We conduct tests to assess the statistical relationship between these frequencies and those necessary for a successful attack. To delve into this relationship, we propose a new method that can uncover non-linear correlations between sets of coordinates, which, in our case, are the aforementio
    

