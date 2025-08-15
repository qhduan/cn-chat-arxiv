# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Hypothesis Spaces for Deep Learning](https://arxiv.org/abs/2403.03353) | 本文介绍了一种应用深度神经网络的深度学习假设空间，并构建了一个再生核Banach空间，研究了正则化学习和最小插值问题，证明了学习模型的解可以表示为线性组合。 |
| [^2] | [An Explainable Transformer-based Model for Phishing Email Detection: A Large Language Model Approach](https://arxiv.org/abs/2402.13871) | 优化的Transformer模型DistilBERT用于检测钓鱼邮件，通过预处理技术解决了类别不平衡问题 |
| [^3] | [Continuous Tensor Relaxation for Finding Diverse Solutions in Combinatorial Optimization Problems](https://arxiv.org/abs/2402.02190) | 本研究提出了连续张量放松方法(CTRA)，用于在组合优化问题中寻找多样化的解决方案。CTRA通过对离散决策变量进行连续放松，解决了寻找多样化解决方案的挑战。 |
| [^4] | [A Bayesian Unification of Self-Supervised Clustering and Energy-Based Models.](http://arxiv.org/abs/2401.00873) | 该论文研究了用贝叶斯方法统一自监督聚类和能量模型，提出了一种标准化的推导方法，并设计了一个新的可靠地惩罚失败模式的下界。这个下界使得能够训练一个标准的骨架架构，而无需使用非对称元素。 |

# 详细

[^1]: 深度学习的假设空间

    Hypothesis Spaces for Deep Learning

    [https://arxiv.org/abs/2403.03353](https://arxiv.org/abs/2403.03353)

    本文介绍了一种应用深度神经网络的深度学习假设空间，并构建了一个再生核Banach空间，研究了正则化学习和最小插值问题，证明了学习模型的解可以表示为线性组合。

    

    本文介绍了一种应用深度神经网络（DNNs）的深度学习假设空间。通过将DNN视为两个变量的函数，即物理变量和参数变量，我们考虑了DNNs的原始集合，参数变量位于由DNNs的权重矩阵和偏置决定的一组深度和宽度中。然后在弱*拓扑中完成原始DNN集合的线性跨度，以构建一个物理变量函数的Banach空间。我们证明所构造的Banach空间是一个再生核Banach空间（RKBS），并构造其再生核。通过为学习模型的解建立表达定理，我们研究了两个学习模型，正则化学习和最小插值问题在结果RKBS中。表达定理揭示了这些学习模型的解可以表示为线性组合

    arXiv:2403.03353v1 Announce Type: cross  Abstract: This paper introduces a hypothesis space for deep learning that employs deep neural networks (DNNs). By treating a DNN as a function of two variables, the physical variable and parameter variable, we consider the primitive set of the DNNs for the parameter variable located in a set of the weight matrices and biases determined by a prescribed depth and widths of the DNNs. We then complete the linear span of the primitive DNN set in a weak* topology to construct a Banach space of functions of the physical variable. We prove that the Banach space so constructed is a reproducing kernel Banach space (RKBS) and construct its reproducing kernel. We investigate two learning models, regularized learning and minimum interpolation problem in the resulting RKBS, by establishing representer theorems for solutions of the learning models. The representer theorems unfold that solutions of these learning models can be expressed as linear combination of
    
[^2]: 一种可解释的基于Transformer的钓鱼邮件检测模型：基于大型语言模型的方法

    An Explainable Transformer-based Model for Phishing Email Detection: A Large Language Model Approach

    [https://arxiv.org/abs/2402.13871](https://arxiv.org/abs/2402.13871)

    优化的Transformer模型DistilBERT用于检测钓鱼邮件，通过预处理技术解决了类别不平衡问题

    

    钓鱼邮件是一种严重的网络威胁，试图通过发送虚假邮件来欺骗用户，意图是窃取机密信息或造成财务损失。攻击者常常冒充可信实体，利用技术进步和复杂性使得钓鱼的检测和预防更具挑战性。尽管进行了大量学术研究，但钓鱼邮件检测在网络安全领域仍然是一个持续且严峻的挑战。大型语言模型（LLMs）和掩盖语言模型（MLMs）拥有巨大潜力，能够提供创新解决方案来解决长期存在的挑战。在本研究论文中，我们提出了一个经过优化的、经过微调的基于Transformer的DistilBERT模型，用于检测钓鱼邮件。在检测过程中，我们使用了一组钓鱼邮件数据集，并利用预处理技术来清理和解决类别不平衡问题。通过我们的实验，

    arXiv:2402.13871v1 Announce Type: cross  Abstract: Phishing email is a serious cyber threat that tries to deceive users by sending false emails with the intention of stealing confidential information or causing financial harm. Attackers, often posing as trustworthy entities, exploit technological advancements and sophistication to make detection and prevention of phishing more challenging. Despite extensive academic research, phishing detection remains an ongoing and formidable challenge in the cybersecurity landscape. Large Language Models (LLMs) and Masked Language Models (MLMs) possess immense potential to offer innovative solutions to address long-standing challenges. In this research paper, we present an optimized, fine-tuned transformer-based DistilBERT model designed for the detection of phishing emails. In the detection process, we work with a phishing email dataset and utilize the preprocessing techniques to clean and solve the imbalance class issues. Through our experiments, 
    
[^3]: 在组合优化问题中寻找多样化解决方案的连续张量放松方法

    Continuous Tensor Relaxation for Finding Diverse Solutions in Combinatorial Optimization Problems

    [https://arxiv.org/abs/2402.02190](https://arxiv.org/abs/2402.02190)

    本研究提出了连续张量放松方法(CTRA)，用于在组合优化问题中寻找多样化的解决方案。CTRA通过对离散决策变量进行连续放松，解决了寻找多样化解决方案的挑战。

    

    在组合优化问题中，寻找最佳解是最常见的目标。然而，在实际场景中，单一解决方案可能不适用，因为目标函数和约束条件只是原始现实世界情况的近似值。为了解决这个问题，寻找具有不同特征的多样化解决方案和约束严重性的变化成为自然的方向。这种策略提供了在后处理过程中选择合适解决方案的灵活性。然而，发现这些多样化解决方案比确定单一解决方案更具挑战性。为了克服这一挑战，本研究引入了连续张量松弛退火 (CTRA) 方法，用于基于无监督学习的组合优化求解器。CTRA通过扩展连续松弛方法，将离散决策变量转换为连续张量，同时解决了多个问题。该方法找到了不同特征的多样化解决方案和约束严重性的变化。

    Finding the best solution is the most common objective in combinatorial optimization (CO) problems. However, a single solution may not be suitable in practical scenarios, as the objective functions and constraints are only approximations of original real-world situations. To tackle this, finding (i) "heterogeneous solutions", diverse solutions with distinct characteristics, and (ii) "penalty-diversified solutions", variations in constraint severity, are natural directions. This strategy provides the flexibility to select a suitable solution during post-processing. However, discovering these diverse solutions is more challenging than identifying a single solution. To overcome this challenge, this study introduces Continual Tensor Relaxation Annealing (CTRA) for unsupervised-learning-based CO solvers. CTRA addresses various problems simultaneously by extending the continual relaxation approach, which transforms discrete decision variables into continual tensors. This method finds heterog
    
[^4]: 用贝叶斯方法统一自监督聚类和能量模型

    A Bayesian Unification of Self-Supervised Clustering and Energy-Based Models. (arXiv:2401.00873v1 [cs.LG])

    [http://arxiv.org/abs/2401.00873](http://arxiv.org/abs/2401.00873)

    该论文研究了用贝叶斯方法统一自监督聚类和能量模型，提出了一种标准化的推导方法，并设计了一个新的可靠地惩罚失败模式的下界。这个下界使得能够训练一个标准的骨架架构，而无需使用非对称元素。

    

    自监督学习是一种利用大量无标签数据的流行且强大的方法，文献中提出了各种训练目标。本研究对最先进的自监督学习目标进行贝叶斯分析，阐明了每个类别中潜在的概率图模型，并提出了一种从基本原理出发推导这些模型的标准方法。分析还表明了将自监督学习与基于似然的生成模型自然整合的方法。我们在基于聚类的自监督学习和能量模型领域中实现了这个概念，引入了一个新的下界，经证明能可靠地惩罚最重要的失败模式。此外，这个新提出的下界使得能够训练一个标准的骨干架构，而无需使用诸如停止梯度、动量编码器或专门的聚类等非对称元素。

    Self-supervised learning is a popular and powerful method for utilizing large amounts of unlabeled data, for which a wide variety of training objectives have been proposed in the literature. In this study, we perform a Bayesian analysis of state-of-the-art self-supervised learning objectives, elucidating the underlying probabilistic graphical models in each class and presenting a standardized methodology for their derivation from first principles. The analysis also indicates a natural means of integrating self-supervised learning with likelihood-based generative models. We instantiate this concept within the realm of cluster-based self-supervised learning and energy models, introducing a novel lower bound which is proven to reliably penalize the most important failure modes. Furthermore, this newly proposed lower bound enables the training of a standard backbone architecture without the necessity for asymmetric elements such as stop gradients, momentum encoders, or specialized clusteri
    

