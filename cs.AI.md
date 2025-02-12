# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Safe Interval RRT* for Scalable Multi-Robot Path Planning in Continuous Space](https://arxiv.org/abs/2404.01752) | 提出了安全间隔RRT*（SI-RRT*）两级方法，低级采用采样规划器找到单个机器人的无碰撞轨迹，高级通过优先规划或基于冲突的搜索解决机器人间冲突，实验结果表明SI-RRT*能够快速找到高质量解决方案 |
| [^2] | [Hallucination Detection in Foundation Models for Decision-Making: A Flexible Definition and Review of the State of the Art](https://arxiv.org/abs/2403.16527) | 基于基础模型的幻觉检测旨在填补现有规划者缺少的常识推理，以适用于超出分布任务的场景。 |
| [^3] | [NeuPAN: Direct Point Robot Navigation with End-to-End Model-based Learning](https://arxiv.org/abs/2403.06828) | NeuPAN 是一种实时、高度准确、无地图、适用于各种机器人且对环境不变的机器人导航解决方案，最大的创新在于将原始点直接映射到学习到的多帧距离空间，并具有端到端模型学习的可解释性，从而实现了可证明的收敛。 |
| [^4] | [Fraud Detection with Binding Global and Local Relational Interaction](https://arxiv.org/abs/2402.17472) | 这项工作提出了一个名为RAGFormer的新框架，同时将局部和全局特征嵌入目标节点，以改进欺诈检测性能。 |
| [^5] | [Value Preferences Estimation and Disambiguation in Hybrid Participatory Systems](https://arxiv.org/abs/2402.16751) | 本研究针对混合参与式系统中的价值偏好估计提出了新方法，通过与参与者互动解决了选择与动机之间的冲突，并重点比较了从动机中估计的价值与仅从选择中估计的价值。 |
| [^6] | [Reinforcement Learning from Human Feedback with Active Queries](https://arxiv.org/abs/2402.09401) | 本文提出了一种基于主动查询的强化学习方法，用于解决与人类反馈的对齐问题。通过在强化学习过程中减少人工标注偏好数据的需求，该方法具有较低的代价，并在实验中表现出较好的性能。 |
| [^7] | [Introspective Planning: Guiding Language-Enabled Agents to Refine Their Own Uncertainty](https://arxiv.org/abs/2402.06529) | 本文研究了内省规划的概念，作为一种引导语言驱动的代理机器人改进自身不确定性的系统方法。通过识别任务不确定性并主动寻求澄清，内省显著提高了机器人任务规划的成功率和安全性。 |
| [^8] | [(Ir)rationality in AI: State of the Art, Research Challenges and Open Questions](https://arxiv.org/abs/2311.17165) | 这篇论文调查了人工智能中理性与非理性的概念，提出了未解问题。重点讨论了行为在某些情况下的非理性行为可能是最优的情况。已经提出了一些方法来处理非理性代理，但仍存在挑战和问题。 |
| [^9] | [A Simple Way to Incorporate Novelty Detection in World Models.](http://arxiv.org/abs/2310.08731) | 本文提出了一个简单的方法，通过利用世界模型幻觉状态和真实观察状态的不匹配性作为异常分数，将新颖性检测纳入世界模型强化学习代理中。在新环境中与传统方法相比，我们的工作具有优势。 |
| [^10] | [A Transfer-Learning-Based Prognosis Prediction Paradigm that Bridges Data Distribution Shift across EMR Datasets.](http://arxiv.org/abs/2310.07799) | 本研究提出了一种基于迁移学习的EMR数据集之间数据分布变化的预后预测模型，通过构建过渡模型解决了不同数据集的特征不匹配问题，提高了深度学习模型的效率。 |

# 详细

[^1]: 安全间隔RRT*用于连续空间中可扩展的多机器人路径规划

    Safe Interval RRT* for Scalable Multi-Robot Path Planning in Continuous Space

    [https://arxiv.org/abs/2404.01752](https://arxiv.org/abs/2404.01752)

    提出了安全间隔RRT*（SI-RRT*）两级方法，低级采用采样规划器找到单个机器人的无碰撞轨迹，高级通过优先规划或基于冲突的搜索解决机器人间冲突，实验结果表明SI-RRT*能够快速找到高质量解决方案

    

    在本文中，我们考虑了在连续空间中解决多机器人路径规划（MRPP）问题以找到无冲突路径的问题。问题的困难主要来自两个因素。首先，涉及多个机器人会导致组合决策，使搜索空间呈指数级增长。其次，连续空间呈现出潜在无限的状态和动作。针对这个问题，我们提出了一个两级方法，低级是基于采样的规划器安全间隔RRT*（SI-RRT*），用于找到单个机器人的无碰撞轨迹。高级可以使用能够解决机器人间冲突的任何方法，我们采用了两种代表性方法，即优先规划（SI-CPP）和基于冲突的搜索（SI-CCBS）。实验结果表明，SI-RRT*能够快速找到高质量解决方案，并且所需的样本数量较少。SI-CPP表现出更好的可扩展性，而SI-CCBS产生

    arXiv:2404.01752v1 Announce Type: cross  Abstract: In this paper, we consider the problem of Multi-Robot Path Planning (MRPP) in continuous space to find conflict-free paths. The difficulty of the problem arises from two primary factors. First, the involvement of multiple robots leads to combinatorial decision-making, which escalates the search space exponentially. Second, the continuous space presents potentially infinite states and actions. For this problem, we propose a two-level approach where the low level is a sampling-based planner Safe Interval RRT* (SI-RRT*) that finds a collision-free trajectory for individual robots. The high level can use any method that can resolve inter-robot conflicts where we employ two representative methods that are Prioritized Planning (SI-CPP) and Conflict Based Search (SI-CCBS). Experimental results show that SI-RRT* can find a high-quality solution quickly with a small number of samples. SI-CPP exhibits improved scalability while SI-CCBS produces 
    
[^2]: 基于基础模型的幻觉检测：灵活定义与现有技术综述

    Hallucination Detection in Foundation Models for Decision-Making: A Flexible Definition and Review of the State of the Art

    [https://arxiv.org/abs/2403.16527](https://arxiv.org/abs/2403.16527)

    基于基础模型的幻觉检测旨在填补现有规划者缺少的常识推理，以适用于超出分布任务的场景。

    

    自主系统即将无处不在，从制造业的自主性到农业领域的机器人，从医疗助理到娱乐产业。大多数系统是通过模块化的子组件开发的，用于决策、规划和控制，这些组件可能是手工设计的，也可能是基于学习的。虽然现有方法在它们专门设计的情境下表现良好，但在测试时不可避免地会在罕见的、超出分布范围的场景下表现特别差。基于多个任务训练的基础模型的兴起，以及从各个领域采集的令人印象深刻的大型数据集，使研究人员相信这些模型可能提供现有规划者所缺乏的常识推理。研究人员认为，这种常识推理将弥合算法开发和部署之间的差距，适用于超出分布任务的情况。

    arXiv:2403.16527v1 Announce Type: new  Abstract: Autonomous systems are soon to be ubiquitous, from manufacturing autonomy to agricultural field robots, and from health care assistants to the entertainment industry. The majority of these systems are developed with modular sub-components for decision-making, planning, and control that may be hand-engineered or learning-based. While these existing approaches have been shown to perform well under the situations they were specifically designed for, they can perform especially poorly in rare, out-of-distribution scenarios that will undoubtedly arise at test-time. The rise of foundation models trained on multiple tasks with impressively large datasets from a variety of fields has led researchers to believe that these models may provide common sense reasoning that existing planners are missing. Researchers posit that this common sense reasoning will bridge the gap between algorithm development and deployment to out-of-distribution tasks, like
    
[^3]: NeuPAN:直接点机器人导航的端到端基于模型学习

    NeuPAN: Direct Point Robot Navigation with End-to-End Model-based Learning

    [https://arxiv.org/abs/2403.06828](https://arxiv.org/abs/2403.06828)

    NeuPAN 是一种实时、高度准确、无地图、适用于各种机器人且对环境不变的机器人导航解决方案，最大的创新在于将原始点直接映射到学习到的多帧距离空间，并具有端到端模型学习的可解释性，从而实现了可证明的收敛。

    

    在拥挤环境中对非全向机器人进行导航需要极其精确的感知和运动以避免碰撞。本文提出NeuPAN：一种实时、高度准确、无地图、适用于各种机器人，且对环境不变的机器人导航解决方案。NeuPAN采用紧耦合的感知-运动框架，与现有方法相比有两个关键创新：1）它直接将原始点映射到学习到的多帧距离空间，避免了从感知到控制的误差传播；2）从端到端基于模型学习的角度进行解释，实现了可证明的收敛。NeuPAN的关键在于利用插拔式（PnP）交替最小化传感器（PAN）网络解高维端到端数学模型，其中包含各种点级约束，使NeuPAN能够直接生成实时、端到端、物理可解释的运动。

    arXiv:2403.06828v1 Announce Type: cross  Abstract: Navigating a nonholonomic robot in a cluttered environment requires extremely accurate perception and locomotion for collision avoidance. This paper presents NeuPAN: a real-time, highly-accurate, map-free, robot-agnostic, and environment-invariant robot navigation solution. Leveraging a tightly-coupled perception-locomotion framework, NeuPAN has two key innovations compared to existing approaches: 1) it directly maps raw points to a learned multi-frame distance space, avoiding error propagation from perception to control; 2) it is interpretable from an end-to-end model-based learning perspective, enabling provable convergence. The crux of NeuPAN is to solve a high-dimensional end-to-end mathematical model with various point-level constraints using the plug-and-play (PnP) proximal alternating-minimization network (PAN) with neurons in the loop. This allows NeuPAN to generate real-time, end-to-end, physically-interpretable motions direct
    
[^4]: 通过融合全局和局部关系交互进行欺诈检测

    Fraud Detection with Binding Global and Local Relational Interaction

    [https://arxiv.org/abs/2402.17472](https://arxiv.org/abs/2402.17472)

    这项工作提出了一个名为RAGFormer的新框架，同时将局部和全局特征嵌入目标节点，以改进欺诈检测性能。

    

    图神经网络已被证明对于欺诈检测具有有效性，因为它能够在整体视角中编码节点交互和聚合特征。最近，具有出色序列编码能力的Transformer网络在文献中也表现出优于其他基于GNN的方法。然而，基于GNN和基于Transformer的网络只编码整个图的一个视角，而GNN编码全局特征，Transformer网络编码局部特征。此外，先前的工作忽视了使用单独网络编码异构图的全局交互特征，导致性能不佳。在这项工作中，我们提出了一个称为Relation-Aware GNN with transFormer（RAGFormer）的新颖框架，将局部和全局特征同时嵌入目标节点中。这个简单而有效的网络应用了一个修改后的GAGA模块，其中每个Transformer层后面都跟着一个跨关系聚合层。

    arXiv:2402.17472v1 Announce Type: cross  Abstract: Graph Neural Network has been proved to be effective for fraud detection for its capability to encode node interaction and aggregate features in a holistic view. Recently, Transformer network with great sequence encoding ability, has also outperformed other GNN-based methods in literatures. However, both GNN-based and Transformer-based networks only encode one perspective of the whole graph, while GNN encodes global features and Transformer network encodes local ones. Furthermore, previous works ignored encoding global interaction features of the heterogeneous graph with separate networks, thus leading to suboptimal performance. In this work, we present a novel framework called Relation-Aware GNN with transFormer (RAGFormer) which simultaneously embeds local and global features into a target node. The simple yet effective network applies a modified GAGA module where each transformer layer is followed by a cross-relation aggregation lay
    
[^5]: 混合参与式系统中的价值偏好估计和消歧

    Value Preferences Estimation and Disambiguation in Hybrid Participatory Systems

    [https://arxiv.org/abs/2402.16751](https://arxiv.org/abs/2402.16751)

    本研究针对混合参与式系统中的价值偏好估计提出了新方法，通过与参与者互动解决了选择与动机之间的冲突，并重点比较了从动机中估计的价值与仅从选择中估计的价值。

    

    在混合参与式系统中理解公民的价值观对于以公民为中心的政策制定至关重要。我们设想了一个混合参与式系统，在这个系统中，参与者做出选择并提供选择的动机，人工智能代理通过与他们互动来估计他们的价值偏好。我们专注于在参与者的选择和动机之间检测到冲突的情况，并提出了估计价值偏好的方法，同时通过与参与者互动来解决检测到的不一致性。我们将“珍视是经过深思熟虑的有意义行为”这一哲学立场操作化。也就是如果参与者的选择是基于对价值偏好的深思熟虑，那么可以在参与者为选择提供的动机中观察到价值偏好。因此，我们提出并比较了优先考虑从动机中估计的价值而不是仅从选择中估计的价值的价值估计方法。

    arXiv:2402.16751v1 Announce Type: cross  Abstract: Understanding citizens' values in participatory systems is crucial for citizen-centric policy-making. We envision a hybrid participatory system where participants make choices and provide motivations for those choices, and AI agents estimate their value preferences by interacting with them. We focus on situations where a conflict is detected between participants' choices and motivations, and propose methods for estimating value preferences while addressing detected inconsistencies by interacting with the participants. We operationalize the philosophical stance that "valuing is deliberatively consequential." That is, if a participant's choice is based on a deliberation of value preferences, the value preferences can be observed in the motivation the participant provides for the choice. Thus, we propose and compare value estimation methods that prioritize the values estimated from motivations over the values estimated from choices alone.
    
[^6]: 使用主动查询的人类反馈强化学习

    Reinforcement Learning from Human Feedback with Active Queries

    [https://arxiv.org/abs/2402.09401](https://arxiv.org/abs/2402.09401)

    本文提出了一种基于主动查询的强化学习方法，用于解决与人类反馈的对齐问题。通过在强化学习过程中减少人工标注偏好数据的需求，该方法具有较低的代价，并在实验中表现出较好的性能。

    

    将大型语言模型（LLM）与人类偏好进行对齐，在构建现代生成模型中发挥重要作用，这可以通过从人类反馈中进行强化学习来实现。然而，尽管当前的强化学习方法表现出优越性能，但往往需要大量的人工标注偏好数据，而这种数据收集费时费力。本文受到主动学习的成功启发，通过提出查询效率高的强化学习方法来解决这个问题。我们首先将对齐问题形式化为上下文竞争二臂强盗问题，并设计了基于主动查询的近端策略优化（APPO）算法，具有$\tilde{O}(d^2/\Delta)$的遗憾界和$\tilde{O}(d^2/\Delta^2)$的查询复杂度，其中$d$是特征空间的维度，$\Delta$是所有上下文中的次优差距。然后，我们提出了ADPO，这是我们算法的实际版本，基于直接偏好优化（DPO）并将其应用于...

    arXiv:2402.09401v1 Announce Type: cross Abstract: Aligning large language models (LLM) with human preference plays a key role in building modern generative models and can be achieved by reinforcement learning from human feedback (RLHF). Despite their superior performance, current RLHF approaches often require a large amount of human-labelled preference data, which is expensive to collect. In this paper, inspired by the success of active learning, we address this problem by proposing query-efficient RLHF methods. We first formalize the alignment problem as a contextual dueling bandit problem and design an active-query-based proximal policy optimization (APPO) algorithm with an $\tilde{O}(d^2/\Delta)$ regret bound and an $\tilde{O}(d^2/\Delta^2)$ query complexity, where $d$ is the dimension of feature space and $\Delta$ is the sub-optimality gap over all the contexts. We then propose ADPO, a practical version of our algorithm based on direct preference optimization (DPO) and apply it to 
    
[^7]: 内省规划：引导语言驱动的代理机器人改进自身的不确定性

    Introspective Planning: Guiding Language-Enabled Agents to Refine Their Own Uncertainty

    [https://arxiv.org/abs/2402.06529](https://arxiv.org/abs/2402.06529)

    本文研究了内省规划的概念，作为一种引导语言驱动的代理机器人改进自身不确定性的系统方法。通过识别任务不确定性并主动寻求澄清，内省显著提高了机器人任务规划的成功率和安全性。

    

    大型语言模型（LLM）展示了先进的推理能力，使得机器人能够理解自然语言指令，并通过适当的基础塑造来策略性地进行高级行动规划。然而，LLM产生的幻觉可能导致机器人自信地执行与用户目标不符或在极端情况下不安全的计划。此外，自然语言指令中的固有歧义可能引发任务的不确定性，尤其是在存在多个有效选项的情况下。为了解决这个问题，LLMs必须识别此类不确定性并主动寻求澄清。本文探索了内省规划的概念，作为一种系统方法，引导LLMs在无需微调的情况下形成意识到不确定性的机器人任务执行计划。我们研究了任务级机器人规划中的不确定性量化，并证明与最先进的基于LLM的规划方法相比，内省显著提高了成功率和安全性。

    Large language models (LLMs) exhibit advanced reasoning skills, enabling robots to comprehend natural language instructions and strategically plan high-level actions through proper grounding. However, LLM hallucination may result in robots confidently executing plans that are misaligned with user goals or, in extreme cases, unsafe. Additionally, inherent ambiguity in natural language instructions can induce task uncertainty, particularly in situations where multiple valid options exist. To address this issue, LLMs must identify such uncertainty and proactively seek clarification. This paper explores the concept of introspective planning as a systematic method for guiding LLMs in forming uncertainty--aware plans for robotic task execution without the need for fine-tuning. We investigate uncertainty quantification in task-level robot planning and demonstrate that introspection significantly improves both success rates and safety compared to state-of-the-art LLM-based planning approaches.
    
[^8]: (非)理性在人工智能中的应用：现状、研究挑战和未解之问

    (Ir)rationality in AI: State of the Art, Research Challenges and Open Questions

    [https://arxiv.org/abs/2311.17165](https://arxiv.org/abs/2311.17165)

    这篇论文调查了人工智能中理性与非理性的概念，提出了未解问题。重点讨论了行为在某些情况下的非理性行为可能是最优的情况。已经提出了一些方法来处理非理性代理，但仍存在挑战和问题。

    

    理性概念在人工智能领域中占据着重要地位。无论是模拟人类推理还是追求有限最优性，我们通常希望使人工智能代理尽可能理性。尽管这个概念在人工智能中非常核心，但对于什么构成理性代理并没有统一的定义。本文调查了人工智能中的理性与非理性，并提出了这个领域的未解问题。在其他领域对理性的理解对其在人工智能中的概念产生了影响，特别是经济学、哲学和心理学方面的研究。着重考虑人工智能代理的行为，我们探讨了在某些情境中非理性行为可能是最优的情况。关于处理非理性代理的方法已经得到了一些发展，包括识别和交互等方面的研究，然而，在这个领域的工作仍然存在一些挑战和问题。

    arXiv:2311.17165v2 Announce Type: replace Abstract: The concept of rationality is central to the field of artificial intelligence. Whether we are seeking to simulate human reasoning, or the goal is to achieve bounded optimality, we generally seek to make artificial agents as rational as possible. Despite the centrality of the concept within AI, there is no unified definition of what constitutes a rational agent. This article provides a survey of rationality and irrationality in artificial intelligence, and sets out the open questions in this area. The understanding of rationality in other fields has influenced its conception within artificial intelligence, in particular work in economics, philosophy and psychology. Focusing on the behaviour of artificial agents, we consider irrational behaviours that can prove to be optimal in certain scenarios. Some methods have been developed to deal with irrational agents, both in terms of identification and interaction, however work in this area re
    
[^9]: 一个简单的方法在世界模型中实现新颖性检测

    A Simple Way to Incorporate Novelty Detection in World Models. (arXiv:2310.08731v1 [cs.AI])

    [http://arxiv.org/abs/2310.08731](http://arxiv.org/abs/2310.08731)

    本文提出了一个简单的方法，通过利用世界模型幻觉状态和真实观察状态的不匹配性作为异常分数，将新颖性检测纳入世界模型强化学习代理中。在新环境中与传统方法相比，我们的工作具有优势。

    

    使用世界模型进行强化学习已经取得了显著的成功。然而，当世界机制或属性发生突然变化时，代理的性能和可靠性可能会显著下降。我们将视觉属性或状态转换的突变称为“新颖性”。在生成的世界模型框架中实施新颖性检测是保护部署时代理的关键任务。在本文中，我们提出了一种简单的边界方法，用于将新颖性检测纳入世界模型强化学习代理中，通过利用世界模型幻觉状态和真实观察状态的不匹配性作为异常分数。首先，我们提供了与序列决策相关的新颖性检测本体论，然后我们提供了在代理在世界模型中学习的转换分布中检测新颖性的有效方法。最后，我们展示了我们的工作在新环境中与传统方法相比的优势。

    Reinforcement learning (RL) using world models has found significant recent successes. However, when a sudden change to world mechanics or properties occurs then agent performance and reliability can dramatically decline. We refer to the sudden change in visual properties or state transitions as {\em novelties}. Implementing novelty detection within generated world model frameworks is a crucial task for protecting the agent when deployed. In this paper, we propose straightforward bounding approaches to incorporate novelty detection into world model RL agents, by utilizing the misalignment of the world model's hallucinated states and the true observed states as an anomaly score. We first provide an ontology of novelty detection relevant to sequential decision making, then we provide effective approaches to detecting novelties in a distribution of transitions learned by an agent in a world model. Finally, we show the advantage of our work in a novel environment compared to traditional ma
    
[^10]: 基于迁移学习的EMR数据集之间数据分布变化的预后预测模型

    A Transfer-Learning-Based Prognosis Prediction Paradigm that Bridges Data Distribution Shift across EMR Datasets. (arXiv:2310.07799v1 [cs.LG])

    [http://arxiv.org/abs/2310.07799](http://arxiv.org/abs/2310.07799)

    本研究提出了一种基于迁移学习的EMR数据集之间数据分布变化的预后预测模型，通过构建过渡模型解决了不同数据集的特征不匹配问题，提高了深度学习模型的效率。

    

    由于对新兴疾病的信息有限，症状很难被察觉和认识到，因此可能忽视临床干预的窗口。期望能够建立一个有效的预后模型，辅助医生进行正确诊断和制定个性化治疗方案，从而及时预防不利结果。然而，在疾病早期阶段，由于数据收集和临床经验有限，再加上对隐私和伦理的考虑，导致可供参考的数据受限，甚至难以正确标记数据标签。此外，不同疾病或同一疾病不同来源的电子医疗记录（EMR）数据可能存在严重的跨数据集特征不匹配问题，严重影响深度学习模型的效率。本文介绍了一种迁移学习方法，建立一个从源数据集到目标数据集的过渡模型，通过对特征进行约束，来解决数据分布变化的问题。

    Due to the limited information about emerging diseases, symptoms are hard to be noticed and recognized, so that the window for clinical intervention could be ignored. An effective prognostic model is expected to assist doctors in making right diagnosis and designing personalized treatment plan, so to promptly prevent unfavorable outcomes. However, in the early stage of a disease, limited data collection and clinical experiences, plus the concern out of privacy and ethics, may result in restricted data availability for reference, to the extent that even data labels are difficult to mark correctly. In addition, Electronic Medical Record (EMR) data of different diseases or of different sources of the same disease can prove to be having serious cross-dataset feature misalignment problems, greatly mutilating the efficiency of deep learning models. This article introduces a transfer learning method to build a transition model from source dataset to target dataset. By way of constraining the 
    

