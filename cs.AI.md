# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ALTO: An Efficient Network Orchestrator for Compound AI Systems](https://arxiv.org/abs/2403.04311) | ALTO是一个网络编排器，针对生成语言模型的优化机会，实现了高吞吐量和低延迟，同时解决了流式中间输出的两个新挑战：正确性和负载平衡。 |
| [^2] | [Practical Insights into Knowledge Distillation for Pre-Trained Models](https://arxiv.org/abs/2402.14922) | 研究对知识蒸馏在预训练模型中的应用进行了深入比较，包括优化的温度和权重参数的调整，以及数据分区KD，揭示了最有效的知识蒸馏策略。 |
| [^3] | [Prioritizing Safeguarding Over Autonomy: Risks of LLM Agents for Science](https://arxiv.org/abs/2402.04247) | 本文探讨了科学领域中基于LLM的智能机器人的漏洞与风险，并强调了对安全措施的重要性。 |
| [^4] | [Causal Perception.](http://arxiv.org/abs/2401.13408) | 这项研究提出了因果感知的概念，并将其应用于自动决策系统中。感知对决策的公平性有重要影响，因为公平性是与背景相关的，并且其解释取决于评判人是谁。 |
| [^5] | [Sample-Driven Federated Learning for Energy-Efficient and Real-Time IoT Sensing.](http://arxiv.org/abs/2310.07497) | 本文提出了一种针对具有实时感知能力的IoT网络设计的基于样本驱动的联邦学习方法，通过控制采样过程来减轻过拟合问题，提高整体准确性，并解决能效问题。 |

# 详细

[^1]: ALTO：一种用于复合AI系统的高效网络编排器

    ALTO: An Efficient Network Orchestrator for Compound AI Systems

    [https://arxiv.org/abs/2403.04311](https://arxiv.org/abs/2403.04311)

    ALTO是一个网络编排器，针对生成语言模型的优化机会，实现了高吞吐量和低延迟，同时解决了流式中间输出的两个新挑战：正确性和负载平衡。

    

    我们提出了ALTO，一种用于有效为诸如语言模型管道之类的复合AI系统提供服务的网络编排器。ALTO通过利用生成语言模型特有的优化机会：流式中间输出，实现了高吞吐量和低延迟。由于语言模型逐个生成token的输出，ALTO在可能时暴露了在阶段之间流式传输中间输出的机会。我们强调了在跨分布式管道阶段实例之间流式传输中间数据时出现的两个新挑战：正确性和负载平衡。我们还提出了聚合感知路由接口和分布式提示感知调度以应对这些挑战的需求。我们在一个复杂的聊天机器人验证管道上展示了ALTO部分输出流式传输的影响，将吞吐量提高了最多3倍，同时将固定延迟目标设置为4秒/请求，还减少了尾延迟。

    arXiv:2403.04311v1 Announce Type: new  Abstract: We present ALTO, a network orchestrator for efficiently serving compound AI systems such as pipelines of language models. ALTO achieves high throughput and low latency by taking advantage of an optimization opportunity specific to generative language models: streaming intermediate outputs. As language models produce outputs token by token, ALTO exposes opportunities to stream intermediate outputs between stages when possible. We highlight two new challenges of correctness and load balancing which emerge when streaming intermediate data across distributed pipeline stage instances. We also motivate the need for an aggregation-aware routing interface and distributed prompt-aware scheduling to address these challenges. We demonstrate the impact of ALTO's partial output streaming on a complex chatbot verification pipeline, increasing throughput by up to 3x for a fixed latency target of 4 seconds / request while also reducing tail latency by 1
    
[^2]: 针对预训练模型的知识蒸馏的实践见解

    Practical Insights into Knowledge Distillation for Pre-Trained Models

    [https://arxiv.org/abs/2402.14922](https://arxiv.org/abs/2402.14922)

    研究对知识蒸馏在预训练模型中的应用进行了深入比较，包括优化的温度和权重参数的调整，以及数据分区KD，揭示了最有效的知识蒸馏策略。

    

    这项研究探讨了在预训练模型中对知识蒸馏（KD）过程的增强，这是知识传输中一个新兴领域，并对分布式训练和联邦学习环境产生重要影响。尽管采用了许多知识蒸馏方法来在预训练模型之间传递知识，但在这些场景中了解知识蒸馏的应用仍然缺乏全面的理解。我们的研究对多种知识蒸馏技术进行了广泛比较，包括标准KD、经过优化温度和权重参数调整的KD、深度相互学习以及数据分区KD。我们评估这些方法在不同数据分布策略下的表现，以确定每种方法最有效的情境。通过详细研究超参数调整，结合广泛的网格搜索评估来获取信息

    arXiv:2402.14922v1 Announce Type: cross  Abstract: This research investigates the enhancement of knowledge distillation (KD) processes in pre-trained models, an emerging field in knowledge transfer with significant implications for distributed training and federated learning environments. These environments benefit from reduced communication demands and accommodate various model architectures. Despite the adoption of numerous KD approaches for transferring knowledge among pre-trained models, a comprehensive understanding of KD's application in these scenarios is lacking. Our study conducts an extensive comparison of multiple KD techniques, including standard KD, tuned KD (via optimized temperature and weight parameters), deep mutual learning, and data partitioning KD. We assess these methods across various data distribution strategies to identify the most effective contexts for each. Through detailed examination of hyperparameter tuning, informed by extensive grid search evaluations, w
    
[^3]: 优先安全保障而非自治：科学中LLM智能机器人的风险

    Prioritizing Safeguarding Over Autonomy: Risks of LLM Agents for Science

    [https://arxiv.org/abs/2402.04247](https://arxiv.org/abs/2402.04247)

    本文探讨了科学领域中基于LLM的智能机器人的漏洞与风险，并强调了对安全措施的重要性。

    

    由大型语言模型（LLMs）驱动的智能机器人在各个学科中自主进行实验和促进科学发现方面展示了巨大的前景。尽管它们的能力非常有前途，但也引入了一些新的漏洞，需要仔细考虑安全性。然而，文献中存在显著的空白，尚未对这些漏洞进行全面探讨。本文通过对科学领域中基于LLM的机器人的漏洞进行深入研究，揭示了它们误用可能带来的潜在风险，并强调了对安全措施的需求，填补了这一空白。我们首先全面概述了科学LLM机器人固有的潜在风险，考虑了用户意图、特定的科学领域以及它们对外部环境可能造成的影响。然后，我们深入探讨了这些漏洞的起源和提供的解决方案。

    Intelligent agents powered by large language models (LLMs) have demonstrated substantial promise in autonomously conducting experiments and facilitating scientific discoveries across various disciplines. While their capabilities are promising, they also introduce novel vulnerabilities that demand careful consideration for safety. However, there exists a notable gap in the literature, as there has been no comprehensive exploration of these vulnerabilities. This position paper fills this gap by conducting a thorough examination of vulnerabilities in LLM-based agents within scientific domains, shedding light on potential risks associated with their misuse and emphasizing the need for safety measures. We begin by providing a comprehensive overview of the potential risks inherent to scientific LLM agents, taking into account user intent, the specific scientific domain, and their potential impact on the external environment. Then, we delve into the origins of these vulnerabilities and provid
    
[^4]: 因果感知

    Causal Perception. (arXiv:2401.13408v1 [cs.AI])

    [http://arxiv.org/abs/2401.13408](http://arxiv.org/abs/2401.13408)

    这项研究提出了因果感知的概念，并将其应用于自动决策系统中。感知对决策的公平性有重要影响，因为公平性是与背景相关的，并且其解释取决于评判人是谁。

    

    当两个个体对相同的信息进行不同解读时，感知会发生。尽管这是一个已知现象，对决策中偏见有影响，但是感知在自动决策系统中仍然被忽视。感知对于ADM系统的公平性或公平使用具有重要影响，因为公平本身是与背景相关的，其解释取决于评判人是谁。本文将感知在因果推理中形式化，以捕捉个体的解释行为。我们还将个体经验形式化为额外的因果知识，个体会使用这些知识。此外，我们定义和讨论了易引发感知的属性，即易引发感知的属性。敏感属性，如性别和种族，就是易引发感知的明确示例。我们根据因果原则定义了两种感知，即不忠实感知和不一致感知。

    Perception occurs when two individuals interpret the same information differently. Despite being a known phenomenon with implications for bias in decision-making, as individuals' experience determines interpretation, perception remains largely overlooked in automated decision-making (ADM) systems. In particular, it can have considerable effects on the fairness or fair usage of an ADM system, as fairness itself is context-specific and its interpretation dependent on who is judging. In this work, we formalize perception under causal reasoning to capture the act of interpretation by an individual. We also formalize individual experience as additional causal knowledge that comes with and is used by an individual. Further, we define and discuss loaded attributes, which are attributes prone to evoke perception. Sensitive attributes, such as gender and race, are clear examples of loaded attributes. We define two kinds of causal perception, unfaithful and inconsistent, based on the causal prop
    
[^5]: 基于样本驱动的联邦学习用于能效和实时IoT感知

    Sample-Driven Federated Learning for Energy-Efficient and Real-Time IoT Sensing. (arXiv:2310.07497v1 [cs.LG])

    [http://arxiv.org/abs/2310.07497](http://arxiv.org/abs/2310.07497)

    本文提出了一种针对具有实时感知能力的IoT网络设计的基于样本驱动的联邦学习方法，通过控制采样过程来减轻过拟合问题，提高整体准确性，并解决能效问题。

    

    在联邦学习系统领域，最近的前沿方法在收敛分析中严重依赖于理想条件。特别地，这些方法假设IoT设备上的训练数据具有与全局数据分布相似的属性。然而，在实时感知联邦学习系统中，这种方法无法捕捉到数据特征的全面范围。为了克服这个限制，我们提出了一种针对具有实时感知能力的IoT网络设计的新方法。我们的方法考虑了由用户数据采样过程引起的泛化差距。通过有效地控制这个采样过程，我们可以减轻过拟合问题，并提高整体准确性。特别地，我们首先制定了一个优化问题，利用采样过程同时减少过拟合和最大化准确性。为了达到这个目标，我们的替代优化问题擅长处理能效问题。

    In the domain of Federated Learning (FL) systems, recent cutting-edge methods heavily rely on ideal conditions convergence analysis. Specifically, these approaches assume that the training datasets on IoT devices possess similar attributes to the global data distribution. However, this approach fails to capture the full spectrum of data characteristics in real-time sensing FL systems. In order to overcome this limitation, we suggest a new approach system specifically designed for IoT networks with real-time sensing capabilities. Our approach takes into account the generalization gap due to the user's data sampling process. By effectively controlling this sampling process, we can mitigate the overfitting issue and improve overall accuracy. In particular, We first formulate an optimization problem that harnesses the sampling process to concurrently reduce overfitting while maximizing accuracy. In pursuit of this objective, our surrogate optimization problem is adept at handling energy ef
    

