# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Algorithmic Collusion by Large Language Models](https://arxiv.org/abs/2404.00806) | 大型语言模型的算法定价代理在寡头市场环境中自主勾结，对消费者利益有害，其说明书中的短语变化可能增加勾结。 |
| [^2] | [Learning to Project for Cross-Task Knowledge Distillation](https://arxiv.org/abs/2403.14494) | 提出了一种通过学习投影，以有效地将传统知识蒸馏方法应用于跨任务设置的方法，取得了显著的性能提升 |
| [^3] | [Scalable Spatiotemporal Prediction with Bayesian Neural Fields](https://arxiv.org/abs/2403.07657) | 该论文提出了贝叶斯神经场（BayesNF），结合了深度神经网络和分层贝叶斯推断，用于处理大规模时空预测问题。 |
| [^4] | [CaT-GNN: Enhancing Credit Card Fraud Detection via Causal Temporal Graph Neural Networks](https://arxiv.org/abs/2402.14708) | 该论文提出了一种名为CaT-GNN的新型信用卡欺诈检测方法，通过因果不变性学习揭示交易数据中的固有相关性，并引入因果混合策略来增强模型的鲁棒性和可解释性。 |
| [^5] | [MENTOR: Guiding Hierarchical Reinforcement Learning with Human Feedback and Dynamic Distance Constraint](https://arxiv.org/abs/2402.14244) | 使用人类反馈和动态距离约束对层次化强化学习进行引导，解决了找到适当子目标的问题，并设计了双策略以稳定训练。 |
| [^6] | [EnrichEvent: Enriching Social Data with Contextual Information for Emerging Event Extraction.](http://arxiv.org/abs/2307.16082) | 本文提出了一个利用词汇、语义和上下文表示的框架，旨在解决现有事件检测方法在识别新兴社交事件方面的局限性，并提供了对社交数据进行丰富的上下文化处理的方法。 |
| [^7] | [Referential communication in heterogeneous communities of pre-trained visual deep networks.](http://arxiv.org/abs/2302.08913) | 异构视觉深度网络社区中的预训练网络可以自我监督地开发出共享协议，以指代一组目标中的目标对象，并可用于沟通不同粒度的未知对象类别。 |
| [^8] | [A Semantic Framework for Neural-Symbolic Computing.](http://arxiv.org/abs/2212.12050) | 该论文提出了一个神经符号计算的语义框架，用于将神经网络和符号AI相结合成为综合系统，并通过将符号知识编码到神经网络中来解决通用推理能力的问题。 |

# 详细

[^1]: 大型语言模型的算法勾结

    Algorithmic Collusion by Large Language Models

    [https://arxiv.org/abs/2404.00806](https://arxiv.org/abs/2404.00806)

    大型语言模型的算法定价代理在寡头市场环境中自主勾结，对消费者利益有害，其说明书中的短语变化可能增加勾结。

    

    arXiv:2404.00806v1 公告类型:交叉摘要:算法定价的兴起引起了对算法勾结的担忧。我们对基于大型语言模型（LLMs）特别是GPT-4的算法定价代理进行实验。我们发现：（1）基于LLM的代理在定价任务上表现出色，（2）基于LLM的定价代理在寡头市场环境中自主勾结，损害消费者利益，（3）LLM说明书中看似无害短语("提示")的变化可能会增加勾结。这些结果也适用于拍卖设置。我们的发现强调了有关算法定价的反垄断监管的必要性，并发现了基于LLM的定价代理所面临的监管挑战。

    arXiv:2404.00806v1 Announce Type: cross  Abstract: The rise of algorithmic pricing raises concerns of algorithmic collusion. We conduct experiments with algorithmic pricing agents based on Large Language Models (LLMs), and specifically GPT-4. We find that (1) LLM-based agents are adept at pricing tasks, (2) LLM-based pricing agents autonomously collude in oligopoly settings to the detriment of consumers, and (3) variation in seemingly innocuous phrases in LLM instructions ("prompts") may increase collusion. These results extend to auction settings. Our findings underscore the need for antitrust regulation regarding algorithmic pricing, and uncover regulatory challenges unique to LLM-based pricing agents.
    
[^2]: 学习投影以进行跨任务知识蒸馏

    Learning to Project for Cross-Task Knowledge Distillation

    [https://arxiv.org/abs/2403.14494](https://arxiv.org/abs/2403.14494)

    提出了一种通过学习投影，以有效地将传统知识蒸馏方法应用于跨任务设置的方法，取得了显著的性能提升

    

    传统知识蒸馏(KD)依赖于在目标任务上训练过的熟练教师，而这并不总是可用的。在这种情况下，可以使用跨任务蒸馏，使得可以利用在不同任务上训练过的任何教师模型。然而，许多知识蒸馏方法在应用于这种跨任务设置时被证明是无效的。为了解决这一限制，我们提出了一个简单的修改：使用反向投影。我们展示了这种对标准投影的插入式替代是有效的，通过学习排除可能降低学生表现的任何任务特定特征。我们发现，这个简单的修改足以将许多知识蒸馏方法扩展到跨任务设置，其中教师和学生任务可能非常不同。这样一来，在跨任务设置中，我们相比于传统投影，可获得最高1.9%的改进，而无需额外成本。我们的方法可以获得显著的性能提升

    arXiv:2403.14494v1 Announce Type: cross  Abstract: Traditional knowledge distillation (KD) relies on a proficient teacher trained on the target task, which is not always available. In this setting, cross-task distillation can be used, enabling the use of any teacher model trained on a different task. However, many KD methods prove ineffective when applied to this cross-task setting. To address this limitation, we propose a simple modification: the use of an inverted projection. We show that this drop-in replacement for a standard projector is effective by learning to disregard any task-specific features which might degrade the student's performance. We find that this simple modification is sufficient for extending many KD methods to the cross-task setting, where the teacher and student tasks can be very different. In doing so, we obtain up to a 1.9% improvement in the cross-task setting compared to the traditional projection, at no additional cost. Our method can obtain significant per
    
[^3]: 使用贝叶斯神经场进行可扩展的时空预测

    Scalable Spatiotemporal Prediction with Bayesian Neural Fields

    [https://arxiv.org/abs/2403.07657](https://arxiv.org/abs/2403.07657)

    该论文提出了贝叶斯神经场（BayesNF），结合了深度神经网络和分层贝叶斯推断，用于处理大规模时空预测问题。

    

    时空数据集由空间参考的时间序列表示，广泛应用于许多科学和商业智能领域，例如空气污染监测，疾病跟踪和云需求预测。随着现代数据集规模和复杂性的不断增加，需要新的统计方法来捕捉复杂的时空动态并处理大规模预测问题。本研究介绍了Bayesian Neural Field (BayesNF)，这是一个用于推断时空域上丰富概率分布的通用领域统计模型，可用于包括预测、插值和变异分析在内的数据分析任务。BayesNF将用于高容量函数估计的新型深度神经网络架构与用于鲁棒不确定性量化的分层贝叶斯推断相结合。通过在定义先验分布方面进行序列化

    arXiv:2403.07657v1 Announce Type: cross  Abstract: Spatiotemporal datasets, which consist of spatially-referenced time series, are ubiquitous in many scientific and business-intelligence applications, such as air pollution monitoring, disease tracking, and cloud-demand forecasting. As modern datasets continue to increase in size and complexity, there is a growing need for new statistical methods that are flexible enough to capture complex spatiotemporal dynamics and scalable enough to handle large prediction problems. This work presents the Bayesian Neural Field (BayesNF), a domain-general statistical model for inferring rich probability distributions over a spatiotemporal domain, which can be used for data-analysis tasks including forecasting, interpolation, and variography. BayesNF integrates a novel deep neural network architecture for high-capacity function estimation with hierarchical Bayesian inference for robust uncertainty quantification. By defining the prior through a sequenc
    
[^4]: 通过因果时间图神经网络增强信用卡欺诈检测

    CaT-GNN: Enhancing Credit Card Fraud Detection via Causal Temporal Graph Neural Networks

    [https://arxiv.org/abs/2402.14708](https://arxiv.org/abs/2402.14708)

    该论文提出了一种名为CaT-GNN的新型信用卡欺诈检测方法，通过因果不变性学习揭示交易数据中的固有相关性，并引入因果混合策略来增强模型的鲁棒性和可解释性。

    

    信用卡欺诈对经济构成重大威胁。尽管基于图神经网络（GNN）的欺诈检测方法表现良好，但它们经常忽视节点的本地结构对预测的因果效应。本文引入了一种新颖的信用卡欺诈检测方法——CaT-GNN（Causal Temporal Graph Neural Networks），利用因果不变性学习来揭示交易数据中的固有相关性。通过将问题分解为发现和干预阶段，CaT-GNN确定交易图中的因果节点，并应用因果混合策略来增强模型的鲁棒性和可解释性。CaT-GNN由两个关键组件组成：Causal-Inspector和Causal-Intervener。Causal-Inspector利用时间注意力机制中的注意力权重来识别因果和环境

    arXiv:2402.14708v1 Announce Type: cross  Abstract: Credit card fraud poses a significant threat to the economy. While Graph Neural Network (GNN)-based fraud detection methods perform well, they often overlook the causal effect of a node's local structure on predictions. This paper introduces a novel method for credit card fraud detection, the \textbf{\underline{Ca}}usal \textbf{\underline{T}}emporal \textbf{\underline{G}}raph \textbf{\underline{N}}eural \textbf{N}etwork (CaT-GNN), which leverages causal invariant learning to reveal inherent correlations within transaction data. By decomposing the problem into discovery and intervention phases, CaT-GNN identifies causal nodes within the transaction graph and applies a causal mixup strategy to enhance the model's robustness and interpretability. CaT-GNN consists of two key components: Causal-Inspector and Causal-Intervener. The Causal-Inspector utilizes attention weights in the temporal attention mechanism to identify causal and environm
    
[^5]: MENTOR：在层次化强化学习中引导人类反馈和动态距离约束

    MENTOR: Guiding Hierarchical Reinforcement Learning with Human Feedback and Dynamic Distance Constraint

    [https://arxiv.org/abs/2402.14244](https://arxiv.org/abs/2402.14244)

    使用人类反馈和动态距离约束对层次化强化学习进行引导，解决了找到适当子目标的问题，并设计了双策略以稳定训练。

    

    层次化强化学习（HRL）为智能体的复杂任务提供了一种有前途的解决方案，其中使用了将任务分解为子目标并依次完成的层次框架。然而，当前的方法难以找到适当的子目标来确保稳定的学习过程。为了解决这个问题，我们提出了一个通用的层次强化学习框架，将人类反馈和动态距离约束整合到其中（MENTOR）。MENTOR充当“导师”，将人类反馈纳入高层策略学习中，以找到更好的子目标。至于低层策略，MENTOR设计了一个双策略以分别进行探索-开发解耦，以稳定训练。此外，尽管人类可以简单地将任务拆分成...

    arXiv:2402.14244v1 Announce Type: new  Abstract: Hierarchical reinforcement learning (HRL) provides a promising solution for complex tasks with sparse rewards of intelligent agents, which uses a hierarchical framework that divides tasks into subgoals and completes them sequentially. However, current methods struggle to find suitable subgoals for ensuring a stable learning process. Without additional guidance, it is impractical to rely solely on exploration or heuristics methods to determine subgoals in a large goal space. To address the issue, We propose a general hierarchical reinforcement learning framework incorporating human feedback and dynamic distance constraints (MENTOR). MENTOR acts as a "mentor", incorporating human feedback into high-level policy learning, to find better subgoals. As for low-level policy, MENTOR designs a dual policy for exploration-exploitation decoupling respectively to stabilize the training. Furthermore, although humans can simply break down tasks into s
    
[^6]: EnrichEvent: 使用上下文信息为新出现的事件提供丰富的社交数据

    EnrichEvent: Enriching Social Data with Contextual Information for Emerging Event Extraction. (arXiv:2307.16082v1 [cs.CL])

    [http://arxiv.org/abs/2307.16082](http://arxiv.org/abs/2307.16082)

    本文提出了一个利用词汇、语义和上下文表示的框架，旨在解决现有事件检测方法在识别新兴社交事件方面的局限性，并提供了对社交数据进行丰富的上下文化处理的方法。

    

    社交平台已成为传播和讨论真实事件信息的关键平台，为及早发现有新闻价值的事件提供了良好的机会。然而，现有的大多数事件检测方法仅利用关键词突发性或网络结构来检测热点事件。因此，对于事件和社交数据的复杂性而言，它们往往无法在达到趋势状态之前识别出新出现的社交事件。社交数据，例如推文，具有拼写错误、不完整性、歧义性和语言不规范性，以及意见方面的变化。此外，利用有限的上下文知识来学习事件的演变特征对于机器学习模型几乎是不可行的。为了解决这些问题，本文提出了一个利用流式社交数据的词汇、语义和上下文表示的框架。

    Social platforms have emerged as a crucial platform for disseminating and discussing information about real-life events, which offers an excellent opportunity for early detection of newsworthy events. However, most existing approaches for event detection solely exploit keyword burstiness or network structures to detect hot events. Thus, they often fail to identify emerging social events before reaching a trending state regarding the challenging nature of events and social data. Social data, e.g., tweets, is characterized by misspellings, incompleteness, ambiguity, and irregular language, as well as variation in aspects of opinions. Moreover, learning the evolving characteristics of the events utilizing limited contextual knowledge is almost infeasible for machine learning models. To address these problems, in this paper, we propose a framework that exploits the lexical, semantic, and contextual representations of streaming social data. In particular, we leverage contextual knowledge to
    
[^7]: 异构视觉深度网络社区中的指代性沟通

    Referential communication in heterogeneous communities of pre-trained visual deep networks. (arXiv:2302.08913v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2302.08913](http://arxiv.org/abs/2302.08913)

    异构视觉深度网络社区中的预训练网络可以自我监督地开发出共享协议，以指代一组目标中的目标对象，并可用于沟通不同粒度的未知对象类别。

    

    随着大型预训练图像处理神经网络被嵌入自动驾驶汽车或机器人等自主代理中，一个问题出现了：在它们具有不同架构和训练方式的情况下，这些系统如何相互之间进行沟通以了解周围的世界。作为朝着这个方向的第一步，我们系统地探索了在一组异构最先进的预训练视觉网络社区中进行"指代性沟通"的任务，结果表明它们可以自我监督地发展一种共享协议来指代一组候选目标中的目标对象。在某种程度上，这种共享协议也可以用来沟通不同粒度的先前未见过的对象类别。此外，一个最初不属于现有社区的视觉网络可以轻松地学习到社区的协议。最后，我们定性和定量地研究了这种新产生的协议的属性，提供了一些证据。

    As large pre-trained image-processing neural networks are being embedded in autonomous agents such as self-driving cars or robots, the question arises of how such systems can communicate with each other about the surrounding world, despite their different architectures and training regimes. As a first step in this direction, we systematically explore the task of \textit{referential communication} in a community of heterogeneous state-of-the-art pre-trained visual networks, showing that they can develop, in a self-supervised way, a shared protocol to refer to a target object among a set of candidates. This shared protocol can also be used, to some extent, to communicate about previously unseen object categories of different granularity. Moreover, a visual network that was not initially part of an existing community can learn the community's protocol with remarkable ease. Finally, we study, both qualitatively and quantitatively, the properties of the emergent protocol, providing some evi
    
[^8]: 神经符号计算的语义框架

    A Semantic Framework for Neural-Symbolic Computing. (arXiv:2212.12050v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2212.12050](http://arxiv.org/abs/2212.12050)

    该论文提出了一个神经符号计算的语义框架，用于将神经网络和符号AI相结合成为综合系统，并通过将符号知识编码到神经网络中来解决通用推理能力的问题。

    

    人工智能的两种方法，神经网络和符号系统，对于一系列AI问题已经被证明非常成功。然而，两者均未能达到人类智能所需的通用推理能力。人们认为这是每种方法内在弱点所致。幸运的是，这些弱点似乎是互补的，符号系统擅长神经网络难以处理的事物，反之亦然。神经符号AI领域试图利用这种不对称性通过将神经网络和符号AI相结合成为综合系统。通常这是通过将符号知识编码到神经网络中实现的。不幸的是，虽然提出了许多不同的方法来实现这一点，但没有公共的编码定义可供比较。我们通过引入神经符号AI的语义框架来解决这个问题，然后证明它足以解释大量神经符号系统。我们的框架是基于符号系统植根于其领域的神经表征的概念。我们展示了我们的框架可以解释各种符号系统在神经表征中的实现方式，包括使用学习的神经表征和使用固定神经表征的系统。

    Two approaches to AI, neural networks and symbolic systems, have been proven very successful for an array of AI problems. However, neither has been able to achieve the general reasoning ability required for human-like intelligence. It has been argued that this is due to inherent weaknesses in each approach. Luckily, these weaknesses appear to be complementary, with symbolic systems being adept at the kinds of things neural networks have trouble with and vice-versa. The field of neural-symbolic AI attempts to exploit this asymmetry by combining neural networks and symbolic AI into integrated systems. Often this has been done by encoding symbolic knowledge into neural networks. Unfortunately, although many different methods for this have been proposed, there is no common definition of an encoding to compare them. We seek to rectify this problem by introducing a semantic framework for neural-symbolic AI, which is then shown to be general enough to account for a large family of neural-symb
    

