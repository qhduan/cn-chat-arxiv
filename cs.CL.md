# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Hallucination Detection in Foundation Models for Decision-Making: A Flexible Definition and Review of the State of the Art](https://arxiv.org/abs/2403.16527) | 基于基础模型的幻觉检测旨在填补现有规划者缺少的常识推理，以适用于超出分布任务的场景。 |
| [^2] | [Value Preferences Estimation and Disambiguation in Hybrid Participatory Systems](https://arxiv.org/abs/2402.16751) | 本研究针对混合参与式系统中的价值偏好估计提出了新方法，通过与参与者互动解决了选择与动机之间的冲突，并重点比较了从动机中估计的价值与仅从选择中估计的价值。 |
| [^3] | [What Changed? Converting Representational Interventions to Natural Language](https://arxiv.org/abs/2402.11355) | 将表征空间的反事实转化为自然语言，以分析和解释模型干预所引起的语言变化，并减轻分类中的偏见。 |
| [^4] | [Reinforcement Learning from Human Feedback with Active Queries](https://arxiv.org/abs/2402.09401) | 本文提出了一种基于主动查询的强化学习方法，用于解决与人类反馈的对齐问题。通过在强化学习过程中减少人工标注偏好数据的需求，该方法具有较低的代价，并在实验中表现出较好的性能。 |
| [^5] | [Introspective Planning: Guiding Language-Enabled Agents to Refine Their Own Uncertainty](https://arxiv.org/abs/2402.06529) | 本文研究了内省规划的概念，作为一种引导语言驱动的代理机器人改进自身不确定性的系统方法。通过识别任务不确定性并主动寻求澄清，内省显著提高了机器人任务规划的成功率和安全性。 |

# 详细

[^1]: 基于基础模型的幻觉检测：灵活定义与现有技术综述

    Hallucination Detection in Foundation Models for Decision-Making: A Flexible Definition and Review of the State of the Art

    [https://arxiv.org/abs/2403.16527](https://arxiv.org/abs/2403.16527)

    基于基础模型的幻觉检测旨在填补现有规划者缺少的常识推理，以适用于超出分布任务的场景。

    

    自主系统即将无处不在，从制造业的自主性到农业领域的机器人，从医疗助理到娱乐产业。大多数系统是通过模块化的子组件开发的，用于决策、规划和控制，这些组件可能是手工设计的，也可能是基于学习的。虽然现有方法在它们专门设计的情境下表现良好，但在测试时不可避免地会在罕见的、超出分布范围的场景下表现特别差。基于多个任务训练的基础模型的兴起，以及从各个领域采集的令人印象深刻的大型数据集，使研究人员相信这些模型可能提供现有规划者所缺乏的常识推理。研究人员认为，这种常识推理将弥合算法开发和部署之间的差距，适用于超出分布任务的情况。

    arXiv:2403.16527v1 Announce Type: new  Abstract: Autonomous systems are soon to be ubiquitous, from manufacturing autonomy to agricultural field robots, and from health care assistants to the entertainment industry. The majority of these systems are developed with modular sub-components for decision-making, planning, and control that may be hand-engineered or learning-based. While these existing approaches have been shown to perform well under the situations they were specifically designed for, they can perform especially poorly in rare, out-of-distribution scenarios that will undoubtedly arise at test-time. The rise of foundation models trained on multiple tasks with impressively large datasets from a variety of fields has led researchers to believe that these models may provide common sense reasoning that existing planners are missing. Researchers posit that this common sense reasoning will bridge the gap between algorithm development and deployment to out-of-distribution tasks, like
    
[^2]: 混合参与式系统中的价值偏好估计和消歧

    Value Preferences Estimation and Disambiguation in Hybrid Participatory Systems

    [https://arxiv.org/abs/2402.16751](https://arxiv.org/abs/2402.16751)

    本研究针对混合参与式系统中的价值偏好估计提出了新方法，通过与参与者互动解决了选择与动机之间的冲突，并重点比较了从动机中估计的价值与仅从选择中估计的价值。

    

    在混合参与式系统中理解公民的价值观对于以公民为中心的政策制定至关重要。我们设想了一个混合参与式系统，在这个系统中，参与者做出选择并提供选择的动机，人工智能代理通过与他们互动来估计他们的价值偏好。我们专注于在参与者的选择和动机之间检测到冲突的情况，并提出了估计价值偏好的方法，同时通过与参与者互动来解决检测到的不一致性。我们将“珍视是经过深思熟虑的有意义行为”这一哲学立场操作化。也就是如果参与者的选择是基于对价值偏好的深思熟虑，那么可以在参与者为选择提供的动机中观察到价值偏好。因此，我们提出并比较了优先考虑从动机中估计的价值而不是仅从选择中估计的价值的价值估计方法。

    arXiv:2402.16751v1 Announce Type: cross  Abstract: Understanding citizens' values in participatory systems is crucial for citizen-centric policy-making. We envision a hybrid participatory system where participants make choices and provide motivations for those choices, and AI agents estimate their value preferences by interacting with them. We focus on situations where a conflict is detected between participants' choices and motivations, and propose methods for estimating value preferences while addressing detected inconsistencies by interacting with the participants. We operationalize the philosophical stance that "valuing is deliberatively consequential." That is, if a participant's choice is based on a deliberation of value preferences, the value preferences can be observed in the motivation the participant provides for the choice. Thus, we propose and compare value estimation methods that prioritize the values estimated from motivations over the values estimated from choices alone.
    
[^3]: 改变了什么？将表征干预转化为自然语言

    What Changed? Converting Representational Interventions to Natural Language

    [https://arxiv.org/abs/2402.11355](https://arxiv.org/abs/2402.11355)

    将表征空间的反事实转化为自然语言，以分析和解释模型干预所引起的语言变化，并减轻分类中的偏见。

    

    针对语言模型（LMs）表征空间的干预方法已经被证明是影响模型行为的有效手段。这些方法被用来消除或改变模型表示中的人口统计信息（如性别）的编码，创建一个反事实的表示。然而，由于干预操作在表示空间内，准确理解它修改了哪些特征是一个挑战。我们展示了表征空间的反事实可以转化为自然语言的反事实。我们证明了这种方法使我们能够分析对应于给定表示空间干预的语言变化，并解释用于编码特定概念的特征。此外，由此产生的反事实可以用于减轻分类中的偏见。

    arXiv:2402.11355v1 Announce Type: new  Abstract: Interventions targeting the representation space of language models (LMs) have emerged as effective means to influence model behavior. These methods are employed, for example, to eliminate or alter the encoding of demographic information such as gender within the model's representations, creating a counterfactual representation. However, since the intervention operates within the representation space, understanding precisely which features it modifies poses a challenge. We show that representation-space counterfactuals can be converted into natural language counterfactuals. We demonstrate that this approach enables us to analyze the linguistic alterations corresponding to a given representation-space intervention and to interpret the features utilized for encoding a specific concept. Moreover, the resulting counterfactuals can be used to mitigate bias in classification.
    
[^4]: 使用主动查询的人类反馈强化学习

    Reinforcement Learning from Human Feedback with Active Queries

    [https://arxiv.org/abs/2402.09401](https://arxiv.org/abs/2402.09401)

    本文提出了一种基于主动查询的强化学习方法，用于解决与人类反馈的对齐问题。通过在强化学习过程中减少人工标注偏好数据的需求，该方法具有较低的代价，并在实验中表现出较好的性能。

    

    将大型语言模型（LLM）与人类偏好进行对齐，在构建现代生成模型中发挥重要作用，这可以通过从人类反馈中进行强化学习来实现。然而，尽管当前的强化学习方法表现出优越性能，但往往需要大量的人工标注偏好数据，而这种数据收集费时费力。本文受到主动学习的成功启发，通过提出查询效率高的强化学习方法来解决这个问题。我们首先将对齐问题形式化为上下文竞争二臂强盗问题，并设计了基于主动查询的近端策略优化（APPO）算法，具有$\tilde{O}(d^2/\Delta)$的遗憾界和$\tilde{O}(d^2/\Delta^2)$的查询复杂度，其中$d$是特征空间的维度，$\Delta$是所有上下文中的次优差距。然后，我们提出了ADPO，这是我们算法的实际版本，基于直接偏好优化（DPO）并将其应用于...

    arXiv:2402.09401v1 Announce Type: cross Abstract: Aligning large language models (LLM) with human preference plays a key role in building modern generative models and can be achieved by reinforcement learning from human feedback (RLHF). Despite their superior performance, current RLHF approaches often require a large amount of human-labelled preference data, which is expensive to collect. In this paper, inspired by the success of active learning, we address this problem by proposing query-efficient RLHF methods. We first formalize the alignment problem as a contextual dueling bandit problem and design an active-query-based proximal policy optimization (APPO) algorithm with an $\tilde{O}(d^2/\Delta)$ regret bound and an $\tilde{O}(d^2/\Delta^2)$ query complexity, where $d$ is the dimension of feature space and $\Delta$ is the sub-optimality gap over all the contexts. We then propose ADPO, a practical version of our algorithm based on direct preference optimization (DPO) and apply it to 
    
[^5]: 内省规划：引导语言驱动的代理机器人改进自身的不确定性

    Introspective Planning: Guiding Language-Enabled Agents to Refine Their Own Uncertainty

    [https://arxiv.org/abs/2402.06529](https://arxiv.org/abs/2402.06529)

    本文研究了内省规划的概念，作为一种引导语言驱动的代理机器人改进自身不确定性的系统方法。通过识别任务不确定性并主动寻求澄清，内省显著提高了机器人任务规划的成功率和安全性。

    

    大型语言模型（LLM）展示了先进的推理能力，使得机器人能够理解自然语言指令，并通过适当的基础塑造来策略性地进行高级行动规划。然而，LLM产生的幻觉可能导致机器人自信地执行与用户目标不符或在极端情况下不安全的计划。此外，自然语言指令中的固有歧义可能引发任务的不确定性，尤其是在存在多个有效选项的情况下。为了解决这个问题，LLMs必须识别此类不确定性并主动寻求澄清。本文探索了内省规划的概念，作为一种系统方法，引导LLMs在无需微调的情况下形成意识到不确定性的机器人任务执行计划。我们研究了任务级机器人规划中的不确定性量化，并证明与最先进的基于LLM的规划方法相比，内省显著提高了成功率和安全性。

    Large language models (LLMs) exhibit advanced reasoning skills, enabling robots to comprehend natural language instructions and strategically plan high-level actions through proper grounding. However, LLM hallucination may result in robots confidently executing plans that are misaligned with user goals or, in extreme cases, unsafe. Additionally, inherent ambiguity in natural language instructions can induce task uncertainty, particularly in situations where multiple valid options exist. To address this issue, LLMs must identify such uncertainty and proactively seek clarification. This paper explores the concept of introspective planning as a systematic method for guiding LLMs in forming uncertainty--aware plans for robotic task execution without the need for fine-tuning. We investigate uncertainty quantification in task-level robot planning and demonstrate that introspection significantly improves both success rates and safety compared to state-of-the-art LLM-based planning approaches.
    

