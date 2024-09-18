# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Unsupervised End-to-End Training with a Self-Defined Bio-Inspired Target](https://arxiv.org/abs/2403.12116) | 本研究提出了一种使用Winner-Take-All（WTA）选择性和生物启发的稳态机制相结合的“自定义目标”方法，旨在解决无监督学习方法在边缘AI硬件上的计算资源稀缺性问题。 |
| [^2] | [Estimating the history of a random recursive tree](https://arxiv.org/abs/2403.09755) | 本文研究了估计随机递归树中顶点到达顺序的问题，提出了基于Jordan中心性度量的顺序估计器，并证明其几乎是最优的。 |
| [^3] | [Data Augmentation is Dead, Long Live Data Augmentation](https://arxiv.org/abs/2402.14895) | 数据增强不过是更好地微调模型，零唁态和少样本数据生成可提高性能 |
| [^4] | [High-arity PAC learning via exchangeability](https://arxiv.org/abs/2402.14294) | 提出高参数PAC学习理论，利用结构化相关性和交换分布取代i.i.d.抽样，证明了统计学习基本定理的高维版本。 |
| [^5] | [Bayesian Parameter-Efficient Fine-Tuning for Overcoming Catastrophic Forgetting](https://arxiv.org/abs/2402.12220) | 这项研究展示了如何利用贝叶斯学习技术应用于参数高效微调，以防止灾难性遗忘，实现了预训练知识的保留，并在语言建模和语音合成任务中取得成功。 |
| [^6] | [Optimal Potential Shaping on SE(3) via Neural ODEs on Lie Groups.](http://arxiv.org/abs/2401.15107) | 本文提出了一种在有限维李群上优化动态系统的新方法，通过将动态系统表示为神经常微分方程，并在李群上制定优化问题。提出了一种可扩展的梯度下降算法来解决优化问题，并通过在李代数级别表示系统来降低计算成本。在一个例子中，处理了刚体控制的最优势能塑形，并通过迭代优化控制器来验证最终结果。 |
| [^7] | [Robust Multimodal Learning with Missing Modalities via Parameter-Efficient Adaptation.](http://arxiv.org/abs/2310.03986) | 通过低秩适应和中间特征的调制，我们提出了针对预训练多模态网络的参数高效适应程序，以实现对缺失模态的鲁棒性，并在某些情况下胜过独立的专门网络。 |
| [^8] | [Learning by Self-Explaining.](http://arxiv.org/abs/2309.08395) | 学习通过自我解释（LSX）是一种新的学习范式，通过给予解释和批评者的反馈来改进学习者的性能。这种方法适用于图像分类等基本任务，并有潜力在人工智能研究中发挥作用。 |
| [^9] | [Bengali Document Layout Analysis -- A YOLOV8 Based Ensembling Approach.](http://arxiv.org/abs/2309.00848) | 本文提出了一种基于YOLOv8模型和创新的后处理技术的孟加拉文档布局分析方法，通过数据增强和两阶段预测策略实现了准确的元素分割。该方法优于单个基础架构，并解决了BaDLAD数据集中的问题，有助于提高OCR和文档理解能力。 |
| [^10] | [Uncertainty Estimation of Transformers' Predictions via Topological Analysis of the Attention Matrices.](http://arxiv.org/abs/2308.11295) | 本论文通过拓扑数据分析方法，提出一种基于注意力机制的拓扑性质的不确定性估计方法，用于Transformer模型的预测，超越传统方法，开辟了注意力机制的新应用领域。 |
| [^11] | [Overcoming the Stability Gap in Continual Learning.](http://arxiv.org/abs/2306.01904) | 本论文研究了如何克服连续学习中的稳定性差距，并通过发现一种显著减少这种差距的方法，在大规模类别增量学习实验中大幅减少了网络更新的次数。 |
| [^12] | [Concentration of Contractive Stochastic Approximation: Additive and Multiplicative Noise.](http://arxiv.org/abs/2303.15740) | 本文研究了具有合同算子的随机逼近算法在有界乘法噪声和加性次高斯噪声设置下的浓度行为，提供了关于收敛误差的极大浓度不等式，并表明这些误差具有亚高斯尾巴或者超多项式尾巴。同时还发现，乘法噪声情况下一般不可能实现亚指数尾巴。 |
| [^13] | [Backdoor Attacks in Peer-to-Peer Federated Learning.](http://arxiv.org/abs/2301.09732) | 本文提出了一种基于点对点联邦学习（P2PFL）的新型后门攻击，利用结构图属性选择恶意节点，实现高攻击成功率，同时保持隐蔽性。同时还评估了这些攻击在多种现实条件下的鲁棒性，并设计了新的防御措施。 |
| [^14] | [A Dynamical System View of Langevin-Based Non-Convex Sampling.](http://arxiv.org/abs/2210.13867) | 本文提出了一种新的框架，通过利用动力系统理论中的几个工具来解决非凸采样中的重要挑战。对于一大类最先进的采样方案，它们在Wasserstein距离下的最后迭代收敛可以归结为对它们的连续时间对应物的研究，这是更好理解的。 |
| [^15] | [ES-GNN: Generalizing Graph Neural Networks Beyond Homophily with Edge Splitting.](http://arxiv.org/abs/2205.13700) | ES-GNN是一种创新的图神经网络框架，通过边分割将图分割为两个子图，以自适应地区分对学习任务相关或不相关的图边。这种方法能够提高GNN在异质图上的普适性和鲁棒性。 |

# 详细

[^1]: 基于自定义生物启发目标的无监督端到端训练

    Unsupervised End-to-End Training with a Self-Defined Bio-Inspired Target

    [https://arxiv.org/abs/2403.12116](https://arxiv.org/abs/2403.12116)

    本研究提出了一种使用Winner-Take-All（WTA）选择性和生物启发的稳态机制相结合的“自定义目标”方法，旨在解决无监督学习方法在边缘AI硬件上的计算资源稀缺性问题。

    

    当前的无监督学习方法依赖于通过深度学习技术（如自监督学习）进行端到端训练，具有较高的计算需求，或者采用通过类似Hebbian学习的生物启发方法逐层训练，使用与监督学习不兼容的局部学习规则。为了解决这一挑战，在这项工作中，我们引入了一种使用网络最终层的胜者通吃（WTA）选择性的“自定义目标”，并通过生物启发的稳态机制进行正则化。

    arXiv:2403.12116v1 Announce Type: cross  Abstract: Current unsupervised learning methods depend on end-to-end training via deep learning techniques such as self-supervised learning, with high computational requirements, or employ layer-by-layer training using bio-inspired approaches like Hebbian learning, using local learning rules incompatible with supervised learning. Both approaches are problematic for edge AI hardware that relies on sparse computational resources and would strongly benefit from alternating between unsupervised and supervised learning phases - thus leveraging widely available unlabeled data from the environment as well as labeled training datasets. To solve this challenge, in this work, we introduce a 'self-defined target' that uses Winner-Take-All (WTA) selectivity at the network's final layer, complemented by regularization through biologically inspired homeostasis mechanism. This approach, framework-agnostic and compatible with both global (Backpropagation) and l
    
[^2]: 估计随机递归树的历史

    Estimating the history of a random recursive tree

    [https://arxiv.org/abs/2403.09755](https://arxiv.org/abs/2403.09755)

    本文研究了估计随机递归树中顶点到达顺序的问题，提出了基于Jordan中心性度量的顺序估计器，并证明其几乎是最优的。

    

    本文研究了估计随机递归树中顶点到达顺序的问题。具体来说，我们研究了两个基本模型：均匀连接模型和线性优先连接模型。我们提出了一种基于Jordan中心性度量的顺序估计器，并定义了一族风险度量来量化排序过程的质量。此外，我们为这个问题建立了极小-最大下界，并证明所提出的估计器几乎是最优的。最后，我们通过数值实验表明所提出的估计器优于基于度数和谱排序程序。

    arXiv:2403.09755v1 Announce Type: cross  Abstract: This paper studies the problem of estimating the order of arrival of the vertices in a random recursive tree. Specifically, we study two fundamental models: the uniform attachment model and the linear preferential attachment model. We propose an order estimator based on the Jordan centrality measure and define a family of risk measures to quantify the quality of the ordering procedure. Moreover, we establish a minimax lower bound for this problem, and prove that the proposed estimator is nearly optimal. Finally, we numerically demonstrate that the proposed estimator outperforms degree-based and spectral ordering procedures.
    
[^3]: 数据增强已死，数据增强万岁

    Data Augmentation is Dead, Long Live Data Augmentation

    [https://arxiv.org/abs/2402.14895](https://arxiv.org/abs/2402.14895)

    数据增强不过是更好地微调模型，零唁态和少样本数据生成可提高性能

    

    文本数据增强（DA）是一个繁荣的研究领域，不断提出新颖的技术来创建人工数据，已经在小数据环境中表现出很高的效率，至少对于文本分类任务而言。在本文中，我们质疑这些结果，表明经典的数据增强只是一种更好地进行微调的方式，并且在应用数据增强之前花更多时间进行微调会抵消其效果。这是一个重要的贡献，因为它回答了最近几年留下的几个问题，即：哪种DA技术表现最佳（只要它们生成的数据与训练集足够接近，不会损害训练），为什么DA表现出积极的结果（简化网络训练）。此外，我们还展示了通过对话代理（如ChatGPT或LLama2）零唁态和少样本数据生成可以提高性能，从而得出了结论，此法可以提高模型性能。

    arXiv:2402.14895v1 Announce Type: cross  Abstract: Textual data augmentation (DA) is a prolific field of study where novel techniques to create artificial data are regularly proposed, and that has demonstrated great efficiency on small data settings, at least for text classification tasks. In this paper, we challenge those results, showing that classical data augmentation is simply a way of performing better fine-tuning, and that spending more time fine-tuning before applying data augmentation negates its effect. This is a significant contribution as it answers several questions that were left open in recent years, namely~: which DA technique performs best (all of them as long as they generate data close enough to the training set as to not impair training) and why did DA show positive results (facilitates training of network). We furthermore show that zero and few-shot data generation via conversational agents such as ChatGPT or LLama2 can increase performances, concluding that this f
    
[^4]: 通过可互换性实现高参数PAC学习

    High-arity PAC learning via exchangeability

    [https://arxiv.org/abs/2402.14294](https://arxiv.org/abs/2402.14294)

    提出高参数PAC学习理论，利用结构化相关性和交换分布取代i.i.d.抽样，证明了统计学习基本定理的高维版本。

    

    我们开发了一种高维PAC学习理论，即在“结构化相关性”存在的统计学习中。 在这个理论中，假设可以是图形、超图，或者更一般地说，是有限关系语言中的结构，并且i.i.d.抽样被抽样产生可互换分布的诱导子结构取代。我们证明了统计学习基本定理的高维版本，通过表征高维（agnostic）PAC可学性，以纯组合维度的有限性及适当版本的均匀收敛。

    arXiv:2402.14294v1 Announce Type: new  Abstract: We develop a theory of high-arity PAC learning, which is statistical learning in the presence of "structured correlation". In this theory, hypotheses are either graphs, hypergraphs or, more generally, structures in finite relational languages, and i.i.d. sampling is replaced by sampling an induced substructure, producing an exchangeable distribution. We prove a high-arity version of the fundamental theorem of statistical learning by characterizing high-arity (agnostic) PAC learnability in terms of finiteness of a purely combinatorial dimension and in terms of an appropriate version of uniform convergence.
    
[^5]: 贝叶斯参数高效微调以克服灾难性遗忘

    Bayesian Parameter-Efficient Fine-Tuning for Overcoming Catastrophic Forgetting

    [https://arxiv.org/abs/2402.12220](https://arxiv.org/abs/2402.12220)

    这项研究展示了如何利用贝叶斯学习技术应用于参数高效微调，以防止灾难性遗忘，实现了预训练知识的保留，并在语言建模和语音合成任务中取得成功。

    

    虽然最初是被文本转语音合成模型的自适应所激发，但我们认为更通用的参数高效微调（PEFT）是进行这种自适应的适当框架。然而，灾难性遗忘仍然是PEFT面临的问题，它损害了预训练模型固有的能力。我们证明现有的贝叶斯学习技术可以应用于PEFT，以防止灾难性遗忘，只要能够可微地计算微调层的参数转换。在一系列关于语言建模和语音合成任务的基础性实验中，我们利用建立的拉普拉斯近似，包括对角线和Kronecker分解方法，来正则化PEFT与低秩适应（LoRA）并比较它们在保留预训练知识方面的性能。我们的结果表明，我们的方法可以克服灾难性遗忘，而不会降低微调性能。

    arXiv:2402.12220v1 Announce Type: cross  Abstract: Although motivated by the adaptation of text-to-speech synthesis models, we argue that more generic parameter-efficient fine-tuning (PEFT) is an appropriate framework to do such adaptation. However, catastrophic forgetting remains an issue with PEFT, damaging the pre-trained model's inherent capabilities. We demonstrate that existing Bayesian learning techniques can be applied to PEFT to prevent catastrophic forgetting as long as the parameter shift of the fine-tuned layers can be calculated differentiably. In a principled series of experiments on language modeling and speech synthesis tasks, we utilize established Laplace approximations, including diagonal and Kronecker factored approaches, to regularize PEFT with the low-rank adaptation (LoRA) and compare their performance in pre-training knowledge preservation. Our results demonstrate that catastrophic forgetting can be overcome by our methods without degrading the fine-tuning perfo
    
[^6]: 在李群上的神经常微分方程对SE(3)的优化潜力塑造

    Optimal Potential Shaping on SE(3) via Neural ODEs on Lie Groups. (arXiv:2401.15107v1 [math.OC])

    [http://arxiv.org/abs/2401.15107](http://arxiv.org/abs/2401.15107)

    本文提出了一种在有限维李群上优化动态系统的新方法，通过将动态系统表示为神经常微分方程，并在李群上制定优化问题。提出了一种可扩展的梯度下降算法来解决优化问题，并通过在李代数级别表示系统来降低计算成本。在一个例子中，处理了刚体控制的最优势能塑形，并通过迭代优化控制器来验证最终结果。

    

    本工作提出了一种新颖的方法，用于优化有限维李群上的动态系统。我们将动态系统重新表述为所谓的神经常微分方程(neural ODEs)，并在李群上制定优化问题。提出了一种梯度下降优化算法来解决数值优化问题。我们的算法可扩展，并适用于任何有限维李群，包括矩阵李群。通过在李代数级别表示系统，减少了梯度计算的计算成本。在一个广泛的例子中，处理了对刚体控制的最优势能塑形。将最优控制问题表述为对李群SE(3)上的神经常微分方程(ODE)的优化，并对控制器进行迭代优化。最后，在状态调节任务上验证了最终的控制器。

    This work presents a novel approach for the optimization of dynamic systems on finite-dimensional Lie groups. We rephrase dynamic systems as so-called neural ordinary differential equations (neural ODEs), and formulate the optimization problem on Lie groups. A gradient descent optimization algorithm is presented to tackle the optimization numerically. Our algorithm is scalable, and applicable to any finite dimensional Lie group, including matrix Lie groups. By representing the system at the Lie algebra level, we reduce the computational cost of the gradient computation. In an extensive example, optimal potential energy shaping for control of a rigid body is treated. The optimal control problem is phrased as an optimization of a neural ODE on the Lie group SE(3), and the controller is iteratively optimized. The final controller is validated on a state-regulation task.
    
[^7]: 通过参数高效适应，实现对缺失模态的鲁棒多模态学习

    Robust Multimodal Learning with Missing Modalities via Parameter-Efficient Adaptation. (arXiv:2310.03986v1 [cs.CV])

    [http://arxiv.org/abs/2310.03986](http://arxiv.org/abs/2310.03986)

    通过低秩适应和中间特征的调制，我们提出了针对预训练多模态网络的参数高效适应程序，以实现对缺失模态的鲁棒性，并在某些情况下胜过独立的专门网络。

    

    多模态学习旨在利用多个数据源来提高下游任务的整体性能。在一些相关的模态中观察到，如果在测试时间缺少一个或多个模态，现有的多模态网络的性能会显著下降。为了实现对缺失模态的鲁棒性，我们提出了预训练的多模态网络的简单和参数高效的适应程序。特别地，我们利用低秩适应和中间特征的调制来补偿缺失的模态。我们证明，这种适应可以部分弥补由于缺失模态而导致的性能下降，并在某些情况下胜过针对可用模态组合进行训练的独立的、专门的网络。所提出的适应所需的参数非常少（例如，少于）

    Multimodal learning seeks to utilize data from multiple sources to improve the overall performance of downstream tasks. It is desirable for redundancies in the data to make multimodal systems robust to missing or corrupted observations in some correlated modalities. However, we observe that the performance of several existing multimodal networks significantly deteriorates if one or multiple modalities are absent at test time. To enable robustness to missing modalities, we propose simple and parameter-efficient adaptation procedures for pretrained multimodal networks. In particular, we exploit low-rank adaptation and modulation of intermediate features to compensate for the missing modalities. We demonstrate that such adaptation can partially bridge performance drop due to missing modalities and outperform independent, dedicated networks trained for the available modality combinations in some cases. The proposed adaptation requires extremely small number of parameters (e.g., fewer than 
    
[^8]: 学习通过自我解释

    Learning by Self-Explaining. (arXiv:2309.08395v1 [cs.AI])

    [http://arxiv.org/abs/2309.08395](http://arxiv.org/abs/2309.08395)

    学习通过自我解释（LSX）是一种新的学习范式，通过给予解释和批评者的反馈来改进学习者的性能。这种方法适用于图像分类等基本任务，并有潜力在人工智能研究中发挥作用。

    

    人工智能研究长期以来一直从生物学中寻找灵感，特别是人类智能。与目前主要将解释视为模型检查手段的人工智能研究相比，从心理学中发现自我解释在代理学习过程中的好处有些被忽视了。受到这个启发，我们引入了一种新的学习范式，称为学习通过自我解释 (LSX)。其中的基本思想是，一个学习模块 (学习者) 执行一个基本任务，比如图像分类，并对其决策进行解释。随后，一个内部批评者模块基于原始任务评估这些解释的质量。最后，学习者通过批评者的反馈得到改进，并根据需要重复这个循环。背后的直觉是，如果批评者能够根据相应的解释执行相同的任务，则该解释被认为是“好”的。尽管有许多实现可能性，但本文旨在提供关于实施学习通过自我解释的一般指导原则。有待进一步的研究和实践来探索这一学习范式的潜力。

    Artificial intelligence (AI) research has a long track record of drawing inspirations from findings from biology, in particular human intelligence. In contrast to current AI research that mainly treats explanations as a means for model inspection, a somewhat neglected finding from human psychology is the benefit of self-explaining in an agents' learning process. Motivated by this, we introduce a novel learning paradigm, termed Learning by Self-Explaining (LSX). The underlying idea is that a learning module (learner) performs a base task, e.g. image classification, and provides explanations to its decisions. An internal critic module next evaluates the quality of these explanations given the original task. Finally, the learner is refined with the critic's feedback and the loop is repeated as required. The intuition behind this is that an explanation is considered "good" if the critic can perform the same task given the respective explanation. Despite many implementation possibilities th
    
[^9]: 孟加拉文档布局分析-一种基于YOLOv8的集成方法

    Bengali Document Layout Analysis -- A YOLOV8 Based Ensembling Approach. (arXiv:2309.00848v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2309.00848](http://arxiv.org/abs/2309.00848)

    本文提出了一种基于YOLOv8模型和创新的后处理技术的孟加拉文档布局分析方法，通过数据增强和两阶段预测策略实现了准确的元素分割。该方法优于单个基础架构，并解决了BaDLAD数据集中的问题，有助于提高OCR和文档理解能力。

    

    本文侧重于利用YOLOv8模型和创新的后处理技术提升孟加拉文档布局分析（DLA）。我们通过数据增强以应对孟加拉复杂文字独特的挑战，经过严格的验证集评估，对完整数据集进行微调，实现准确的元素分割的两阶段预测策略。我们的集成模型结合后处理性能优于单个基础架构，解决了BaDLAD数据集中的问题。通过利用这种方法，我们旨在推动孟加拉文档分析的发展，提高OCR和文档理解能力，同时BaDLAD作为基础资源有助于未来的研究。此外，我们的实验为将新策略纳入现有解决方案提供了关键见解。

    This paper focuses on enhancing Bengali Document Layout Analysis (DLA) using the YOLOv8 model and innovative post-processing techniques. We tackle challenges unique to the complex Bengali script by employing data augmentation for model robustness. After meticulous validation set evaluation, we fine-tune our approach on the complete dataset, leading to a two-stage prediction strategy for accurate element segmentation. Our ensemble model, combined with post-processing, outperforms individual base architectures, addressing issues identified in the BaDLAD dataset. By leveraging this approach, we aim to advance Bengali document analysis, contributing to improved OCR and document comprehension and BaDLAD serves as a foundational resource for this endeavor, aiding future research in the field. Furthermore, our experiments provided key insights to incorporate new strategies into the established solution.
    
[^10]: 通过注意力矩阵的拓扑分析来估算Transformer模型预测的不确定性

    Uncertainty Estimation of Transformers' Predictions via Topological Analysis of the Attention Matrices. (arXiv:2308.11295v1 [cs.LG])

    [http://arxiv.org/abs/2308.11295](http://arxiv.org/abs/2308.11295)

    本论文通过拓扑数据分析方法，提出一种基于注意力机制的拓扑性质的不确定性估计方法，用于Transformer模型的预测，超越传统方法，开辟了注意力机制的新应用领域。

    

    在自然语言处理领域，确定深度学习模型预测的置信度是一个开放的问题。传统的不确定性估计方法对于文本分类模型并不有效。我们提出了一种基于Transformer架构的神经网络的不确定性估计任务。这种模型的一个关键特点是注意力机制，它支持神经网络中的令牌之间的信息流。我们利用拓扑数据分析方法探索内部表示之间的关系，并利用它们来预测模型的置信度。本文提出了一种基于注意力机制的拓扑性质的不确定性估计方法，并与传统方法进行了比较。结果表明，该算法在质量上超过了现有的方法，并开辟了注意力机制的新应用领域，但需要...

    Determining the degree of confidence of deep learning model in its prediction is an open problem in the field of natural language processing. Most of the classical methods for uncertainty estimation are quite weak for text classification models. We set the task of obtaining an uncertainty estimate for neural networks based on the Transformer architecture. A key feature of such mo-dels is the attention mechanism, which supports the information flow between the hidden representations of tokens in the neural network. We explore the formed relationships between internal representations using Topological Data Analysis methods and utilize them to predict model's confidence. In this paper, we propose a method for uncertainty estimation based on the topological properties of the attention mechanism and compare it with classical methods. As a result, the proposed algorithm surpasses the existing methods in quality and opens up a new area of application of the attention mechanism, but requires t
    
[^11]: 克服连续学习中的稳定性差距

    Overcoming the Stability Gap in Continual Learning. (arXiv:2306.01904v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2306.01904](http://arxiv.org/abs/2306.01904)

    本论文研究了如何克服连续学习中的稳定性差距，并通过发现一种显著减少这种差距的方法，在大规模类别增量学习实验中大幅减少了网络更新的次数。

    

    在许多实际应用中，随着数据集大小的增长，深度神经网络往往需要从头开始重新训练。考虑到重新训练的计算开销，人们认为连续学习可以使网络更新更加高效。实现这一目标的障碍是稳定性差距，即在更新新数据时，先前学习的数据性能会下降，然后才得以恢复。解决这个问题可以减少网络更新的次数，提高计算效率。我们研究了如何缓解稳定性差距，并测试了多种假设以了解其产生原因。这使我们发现了一种显著减少稳定性差距的方法。在大规模的增量类别学习实验中，我们能够显著减少连续学习所需的网络更新次数。我们的工作有可能推动连续学习在实际应用中的最新进展。

    In many real-world applications, deep neural networks are retrained from scratch as a dataset grows in size. Given the computational expense for retraining networks, it has been argued that continual learning could make updating networks more efficient. An obstacle to achieving this goal is the stability gap, which refers to an observation that when updating on new data, performance on previously learned data degrades before recovering. Addressing this problem would enable learning new data with fewer network updates, resulting in increased computational efficiency. We study how to mitigate the stability gap. We test a variety of hypotheses to understand why the stability gap occurs. This leads us to discover a method that vastly reduces this gap. In large-scale class incremental learning experiments, we are able to significantly reduce the number of network updates needed for continual learning. Our work has the potential to advance the state-of-the-art in continual learning for real-
    
[^12]: 合同扩张随机近似的浓度：加法和乘法噪声

    Concentration of Contractive Stochastic Approximation: Additive and Multiplicative Noise. (arXiv:2303.15740v1 [cs.LG])

    [http://arxiv.org/abs/2303.15740](http://arxiv.org/abs/2303.15740)

    本文研究了具有合同算子的随机逼近算法在有界乘法噪声和加性次高斯噪声设置下的浓度行为，提供了关于收敛误差的极大浓度不等式，并表明这些误差具有亚高斯尾巴或者超多项式尾巴。同时还发现，乘法噪声情况下一般不可能实现亚指数尾巴。

    

    本文研究了在任何范数下，具有合同算子的随机逼近(SA)算法的浓度行为。 我们考虑两种情况，其中迭代可能无界：（1）有界乘法噪声，（2）加性次高斯噪声。 我们得到了关于收敛误差的极大浓度不等式，并表明这些误差在加性噪声设置下具有亚高斯尾巴，在乘法噪声设置下具有超多项式尾巴（快于多项式衰减）。 此外，我们提供了一个不可能结果，显示通常无法通过乘法噪声的SA实现亚指数尾巴。 为了确立这些结果，我们开发了一种新的自举论证，其中涉及边界误差的广义Moreau包络的矩生成函数和指数超马尔可夫构造，以启用使用Ville的极大不等式。

    In this work, we study the concentration behavior of a stochastic approximation (SA) algorithm under a contractive operator with respect to an arbitrary norm. We consider two settings where the iterates are potentially unbounded: (1) bounded multiplicative noise, and (2) additive sub-Gaussian noise. We obtain maximal concentration inequalities on the convergence errors, and show that these errors have sub-Gaussian tails in the additive noise setting, and super-polynomial tails (faster than polynomial decay) in the multiplicative noise setting. In addition, we provide an impossibility result showing that it is in general not possible to achieve sub-exponential tails for SA with multiplicative noise. To establish these results, we develop a novel bootstrapping argument that involves bounding the moment generating function of the generalized Moreau envelope of the error and the construction of an exponential supermartingale to enable using Ville's maximal inequality.  To demonstrate the a
    
[^13]: 点对点联邦学习中的后门攻击

    Backdoor Attacks in Peer-to-Peer Federated Learning. (arXiv:2301.09732v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.09732](http://arxiv.org/abs/2301.09732)

    本文提出了一种基于点对点联邦学习（P2PFL）的新型后门攻击，利用结构图属性选择恶意节点，实现高攻击成功率，同时保持隐蔽性。同时还评估了这些攻击在多种现实条件下的鲁棒性，并设计了新的防御措施。

    

    大多数机器学习应用程序依赖于集中式学习过程，这开放了曝光其训练数据集的风险。尽管联邦学习（FL）在某种程度上缓解了这些隐私风险，但它仍依赖于可信的聚合服务器来训练共享全局模型。最近，基于点对点联邦学习（P2PFL）的新分布式学习架构在隐私和可靠性方面都提供了优势。然而，在训练期间对毒化攻击的鲁棒性尚未得到研究。在本文中，我们提出了一种新的P2PFL后门攻击，利用结构图属性选择恶意节点，实现高攻击成功率，同时保持隐蔽性。我们在各种实际条件下评估我们的攻击，包括多个图形拓扑、网络中有限的敌对能见度以及具有非独立同分布数据的客户端。最后，我们展示了从FL中适应的现有防御措施的局限性，并设计了一种新的防御措施。

    Most machine learning applications rely on centralized learning processes, opening up the risk of exposure of their training datasets. While federated learning (FL) mitigates to some extent these privacy risks, it relies on a trusted aggregation server for training a shared global model. Recently, new distributed learning architectures based on Peer-to-Peer Federated Learning (P2PFL) offer advantages in terms of both privacy and reliability. Still, their resilience to poisoning attacks during training has not been investigated. In this paper, we propose new backdoor attacks for P2PFL that leverage structural graph properties to select the malicious nodes, and achieve high attack success, while remaining stealthy. We evaluate our attacks under various realistic conditions, including multiple graph topologies, limited adversarial visibility of the network, and clients with non-IID data. Finally, we show the limitations of existing defenses adapted from FL and design a new defense that su
    
[^14]: Langevin-Based Non-Convex Sampling的动力学系统视角

    A Dynamical System View of Langevin-Based Non-Convex Sampling. (arXiv:2210.13867v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.13867](http://arxiv.org/abs/2210.13867)

    本文提出了一种新的框架，通过利用动力系统理论中的几个工具来解决非凸采样中的重要挑战。对于一大类最先进的采样方案，它们在Wasserstein距离下的最后迭代收敛可以归结为对它们的连续时间对应物的研究，这是更好理解的。

    This paper proposes a new framework that uses tools from the theory of dynamical systems to address important challenges in non-convex sampling. For a large class of state-of-the-art sampling schemes, their last-iterate convergence in Wasserstein distances can be reduced to the study of their continuous-time counterparts, which is much better understood.

    非凸采样是机器学习中的一个关键挑战，对于深度学习中的非凸优化以及近似概率推断都至关重要。尽管其重要性，理论上仍存在许多重要挑战：现有的保证通常仅适用于平均迭代而不是更理想的最后迭代，缺乏捕捉变量尺度（如Wasserstein距离）的收敛度量，主要适用于随机梯度Langevin动力学等基本方案。在本文中，我们开发了一个新的框架，通过利用动力系统理论中的几个工具来解决上述问题。我们的关键结果是，对于一大类最先进的采样方案，它们在Wasserstein距离下的最后迭代收敛可以归结为对它们的连续时间对应物的研究，这是更好理解的。结合MCMC采样的标准假设，我们的理论立即产生了

    Non-convex sampling is a key challenge in machine learning, central to non-convex optimization in deep learning as well as to approximate probabilistic inference. Despite its significance, theoretically there remain many important challenges: Existing guarantees (1) typically only hold for the averaged iterates rather than the more desirable last iterates, (2) lack convergence metrics that capture the scales of the variables such as Wasserstein distances, and (3) mainly apply to elementary schemes such as stochastic gradient Langevin dynamics. In this paper, we develop a new framework that lifts the above issues by harnessing several tools from the theory of dynamical systems. Our key result is that, for a large class of state-of-the-art sampling schemes, their last-iterate convergence in Wasserstein distances can be reduced to the study of their continuous-time counterparts, which is much better understood. Coupled with standard assumptions of MCMC sampling, our theory immediately yie
    
[^15]: ES-GNN: 通过边分割将图神经网络推广到异质图

    ES-GNN: Generalizing Graph Neural Networks Beyond Homophily with Edge Splitting. (arXiv:2205.13700v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2205.13700](http://arxiv.org/abs/2205.13700)

    ES-GNN是一种创新的图神经网络框架，通过边分割将图分割为两个子图，以自适应地区分对学习任务相关或不相关的图边。这种方法能够提高GNN在异质图上的普适性和鲁棒性。

    

    尽管图神经网络在多个图分析任务中取得了巨大成功，但现代变体主要依赖于同质性的强归纳偏差。然而，现实世界的网络通常同时显示同质性和异质性的链接模式，其中相邻节点可能具有不同的属性和不同的标签。因此，GNN在整体上平滑节点接近性可能会聚合任务相关和不相关（甚至有害）的信息，限制了它们推广到异质图的能力，并可能导致非鲁棒性。在这项工作中，我们提出了一种创新的边分割GNN（ES-GNN）框架，以自适应地区分对学习任务相关或不相关的图边。这将原始图转化为两个具有相同节点集但具有独占边集的子图。在这两个子图上分别进行信息传播和边分割，从而使信息传播和边分割交替进行，实现了解耦。

    While Graph Neural Networks (GNNs) have achieved enormous success in multiple graph analytical tasks, modern variants mostly rely on the strong inductive bias of homophily. However, real-world networks typically exhibit both homophilic and heterophilic linking patterns, wherein adjacent nodes may share dissimilar attributes and distinct labels. Therefore, GNNs smoothing node proximity holistically may aggregate both task-relevant and irrelevant (even harmful) information, limiting their ability to generalize to heterophilic graphs and potentially causing non-robustness. In this work, we propose a novel edge splitting GNN (ES-GNN) framework to adaptively distinguish between graph edges either relevant or irrelevant to learning tasks. This essentially transfers the original graph into two subgraphs with the same node set but exclusive edge sets dynamically. Given that, information propagation separately on these subgraphs and edge splitting are alternatively conducted, thus disentangling
    

