# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Equipping Sketch Patches with Context-Aware Positional Encoding for Graphic Sketch Representation](https://arxiv.org/abs/2403.17525) | 提出了一种通过为素描补丁配备上下文感知的位置编码来保护不同绘图版本的方法，将绘图顺序信息嵌入图节点中，以更好地学习图形素描表示。 |
| [^2] | [Do You Trust Your Model? Emerging Malware Threats in the Deep Learning Ecosystem](https://arxiv.org/abs/2403.03593) | 介绍了MaleficNet 2.0，一种在神经网络中嵌入恶意软件的新技术，其注入技术具有隐蔽性，不会降低模型性能，并且对神经网络参数中的恶意有效负载进行注入 |
| [^3] | [LLM Multi-Agent Systems: Challenges and Open Problems](https://arxiv.org/abs/2402.03578) | 本文讨论了多智能体系统的挑战与开放问题，包括任务分配优化、增强推理能力、管理上下文信息和改善内存管理，同时探讨了多智能体系统在区块链系统中的潜力和未来发展。 |
| [^4] | [Unravelling Responsibility for AI.](http://arxiv.org/abs/2308.02608) | 本文旨在解构人工智能责任的概念，提出了一种包含四种责任意义的有效组合，以支持对人工智能责任的实践推理。 |
| [^5] | [A stochastic optimization approach to train non-linear neural networks with regularization of higher-order total variation.](http://arxiv.org/abs/2308.02293) | 通过引入高阶总变差正则化的随机优化算法，可以高效地训练非线性神经网络，避免过拟合问题。 |

# 详细

[^1]: 为图形素描表示装备具有上下文感知的位置编码的素描补丁

    Equipping Sketch Patches with Context-Aware Positional Encoding for Graphic Sketch Representation

    [https://arxiv.org/abs/2403.17525](https://arxiv.org/abs/2403.17525)

    提出了一种通过为素描补丁配备上下文感知的位置编码来保护不同绘图版本的方法，将绘图顺序信息嵌入图节点中，以更好地学习图形素描表示。

    

    一幅素描的绘制顺序记录了它是如何逐笔由人类创建的。对于图形素描表示学习，最近的研究通过根据基于时间的最近邻策略将每个补丁与另一个相连，将素描绘图顺序注入到图边构建中。然而，这样构建的图边可能不可靠，因为素描可能有不同版本的绘图。在本文中，我们提出了一种经过变体绘制保护的方法，通过为素描补丁配备具有上下文感知的位置编码(PE)，以更好地利用绘图顺序来学习图形素描表示。我们没有将素描绘制注入到图边中，而是仅将这些顺序信息嵌入到图节点中。具体来说，每个补丁嵌入都配备有正弦绝对PE，以突出绘图顺序中的顺序位置。它的相邻补丁按self-att的价值排序

    arXiv:2403.17525v1 Announce Type: cross  Abstract: The drawing order of a sketch records how it is created stroke-by-stroke by a human being. For graphic sketch representation learning, recent studies have injected sketch drawing orders into graph edge construction by linking each patch to another in accordance to a temporal-based nearest neighboring strategy. However, such constructed graph edges may be unreliable, since a sketch could have variants of drawings. In this paper, we propose a variant-drawing-protected method by equipping sketch patches with context-aware positional encoding (PE) to make better use of drawing orders for learning graphic sketch representation. Instead of injecting sketch drawings into graph edges, we embed these sequential information into graph nodes only. More specifically, each patch embedding is equipped with a sinusoidal absolute PE to highlight the sequential position in the drawing order. And its neighboring patches, ranked by the values of self-att
    
[^2]: 您信任您的模型吗？深度学习生态系统中新兴的恶意软件威胁

    Do You Trust Your Model? Emerging Malware Threats in the Deep Learning Ecosystem

    [https://arxiv.org/abs/2403.03593](https://arxiv.org/abs/2403.03593)

    介绍了MaleficNet 2.0，一种在神经网络中嵌入恶意软件的新技术，其注入技术具有隐蔽性，不会降低模型性能，并且对神经网络参数中的恶意有效负载进行注入

    

    训练高质量的深度学习模型是一项具有挑战性的任务，这是因为需要计算和技术要求。越来越多的个人、机构和公司越来越多地依赖于在公共代码库中提供的预训练的第三方模型。这些模型通常直接使用或集成到产品管道中而没有特殊的预防措施，因为它们实际上只是以张量形式的数据，被认为是安全的。在本文中，我们提出了一种针对神经网络的新的机器学习供应链威胁。我们介绍了MaleficNet 2.0，一种在神经网络中嵌入自解压自执行恶意软件的新技术。MaleficNet 2.0使用扩频信道编码结合纠错技术在深度神经网络的参数中注入恶意有效载荷。MaleficNet 2.0注入技术具有隐蔽性，不会降低模型的性能，并且对...

    arXiv:2403.03593v1 Announce Type: cross  Abstract: Training high-quality deep learning models is a challenging task due to computational and technical requirements. A growing number of individuals, institutions, and companies increasingly rely on pre-trained, third-party models made available in public repositories. These models are often used directly or integrated in product pipelines with no particular precautions, since they are effectively just data in tensor form and considered safe. In this paper, we raise awareness of a new machine learning supply chain threat targeting neural networks. We introduce MaleficNet 2.0, a novel technique to embed self-extracting, self-executing malware in neural networks. MaleficNet 2.0 uses spread-spectrum channel coding combined with error correction techniques to inject malicious payloads in the parameters of deep neural networks. MaleficNet 2.0 injection technique is stealthy, does not degrade the performance of the model, and is robust against 
    
[^3]: LLM多智能体系统：挑战与开放问题

    LLM Multi-Agent Systems: Challenges and Open Problems

    [https://arxiv.org/abs/2402.03578](https://arxiv.org/abs/2402.03578)

    本文讨论了多智能体系统的挑战与开放问题，包括任务分配优化、增强推理能力、管理上下文信息和改善内存管理，同时探讨了多智能体系统在区块链系统中的潜力和未来发展。

    

    本文探讨了现有多智能体系统的研究工作，并识别出尚未充分解决的挑战。通过利用多智能体系统内个体智能体的多样能力和角色，这些系统可以通过协作来处理复杂任务。我们讨论了优化任务分配、通过迭代辩论促进强大推理、管理复杂和分层的上下文信息以及增强内存管理以支持多智能体系统内的复杂交互。我们还探讨了在区块链系统中应用多智能体系统的潜力，以启示其在真实分布式系统中的未来发展和应用。

    This paper explores existing works of multi-agent systems and identifies challenges that remain inadequately addressed. By leveraging the diverse capabilities and roles of individual agents within a multi-agent system, these systems can tackle complex tasks through collaboration. We discuss optimizing task allocation, fostering robust reasoning through iterative debates, managing complex and layered context information, and enhancing memory management to support the intricate interactions within multi-agent systems. We also explore the potential application of multi-agent systems in blockchain systems to shed light on their future development and application in real-world distributed systems.
    
[^4]: 解构人工智能责任

    Unravelling Responsibility for AI. (arXiv:2308.02608v1 [cs.AI])

    [http://arxiv.org/abs/2308.02608](http://arxiv.org/abs/2308.02608)

    本文旨在解构人工智能责任的概念，提出了一种包含四种责任意义的有效组合，以支持对人工智能责任的实践推理。

    

    为了在涉及人工智能系统的复杂情况下合理思考责任应该放在何处，我们首先需要一个足够清晰和详细的跨学科词汇来谈论责任。责任是一种三元关系，涉及到一个行为者、一个事件和一种责任方式。作为一种有意识的为了支持对人工智能责任进行实践推理的“解构”责任概念的努力，本文采取了“行为者A对事件O负责”的三部分表述，并确定了A、负责、O的子类别的有效组合。这些有效组合我们称之为“责任串”，分为四种责任意义：角色责任、因果责任、法律责任和道德责任。我们通过两个运行示例进行了说明，一个涉及医疗AI系统，另一个涉及AV与行人的致命碰撞。

    To reason about where responsibility does and should lie in complex situations involving AI-enabled systems, we first need a sufficiently clear and detailed cross-disciplinary vocabulary for talking about responsibility. Responsibility is a triadic relation involving an actor, an occurrence, and a way of being responsible. As part of a conscious effort towards 'unravelling' the concept of responsibility to support practical reasoning about responsibility for AI, this paper takes the three-part formulation, 'Actor A is responsible for Occurrence O' and identifies valid combinations of subcategories of A, is responsible for, and O. These valid combinations - which we term "responsibility strings" - are grouped into four senses of responsibility: role-responsibility; causal responsibility; legal liability-responsibility; and moral responsibility. They are illustrated with two running examples, one involving a healthcare AI-based system and another the fatal collision of an AV with a pedes
    
[^5]: 用正则化高阶总变差的随机优化方法训练非线性神经网络

    A stochastic optimization approach to train non-linear neural networks with regularization of higher-order total variation. (arXiv:2308.02293v1 [stat.ME])

    [http://arxiv.org/abs/2308.02293](http://arxiv.org/abs/2308.02293)

    通过引入高阶总变差正则化的随机优化算法，可以高效地训练非线性神经网络，避免过拟合问题。

    

    尽管包括深度神经网络在内的高度表达的参数模型可以更好地建模复杂概念，但训练这种高度非线性模型已知会导致严重的过拟合风险。针对这个问题，本研究考虑了一种k阶总变差（k-TV）正则化，它被定义为要训练的参数模型的k阶导数的平方积分，通过惩罚k-TV来产生一个更平滑的函数，从而避免过拟合。尽管将k-TV项应用于一般的参数模型由于积分而导致计算复杂，本研究提供了一种随机优化算法，可以高效地训练带有k-TV正则化的一般模型，而无需进行显式的数值积分。这种方法可以应用于结构任意的深度神经网络的训练，因为它只需要进行简单的随机梯度优化即可实现。

    While highly expressive parametric models including deep neural networks have an advantage to model complicated concepts, training such highly non-linear models is known to yield a high risk of notorious overfitting. To address this issue, this study considers a $k$th order total variation ($k$-TV) regularization, which is defined as the squared integral of the $k$th order derivative of the parametric models to be trained; penalizing the $k$-TV is expected to yield a smoother function, which is expected to avoid overfitting. While the $k$-TV terms applied to general parametric models are computationally intractable due to the integration, this study provides a stochastic optimization algorithm, that can efficiently train general models with the $k$-TV regularization without conducting explicit numerical integration. The proposed approach can be applied to the training of even deep neural networks whose structure is arbitrary, as it can be implemented by only a simple stochastic gradien
    

