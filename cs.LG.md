# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fine-Tuning, Prompting, In-Context Learning and Instruction-Tuning: How Many Labelled Samples Do We Need?](https://arxiv.org/abs/2402.12819) | 专门模型通常只需少量标记样本（100-1000个）就能与通用模型持平甚至更好，取决于任务的复杂性和结果的变化。 |
| [^2] | [Entire Chain Uplift Modeling with Context-Enhanced Learning for Intelligent Marketing](https://arxiv.org/abs/2402.03379) | 全链路上升建模方法ECUP旨在解决链路偏差和处理不适应问题，在线营销中有重要的应用价值。 |
| [^3] | [Task Aware Dreamer for Task Generalization in Reinforcement Learning.](http://arxiv.org/abs/2303.05092) | 本文提出了一种名为Task Aware Dreamer（TAD）的方法用于强化学习中的任务泛化。通过量化任务分布的相关性，TAD能够将历史信息编码到策略中，以便区分不同任务，并在泛化到未见任务时具有较好的性能。 |
| [^4] | [Vertical Semi-Federated Learning for Efficient Online Advertising.](http://arxiv.org/abs/2209.15635) | 垂直半联合学习为在线广告领域提供了高效的解决方案，通过学习一个联合感知的局部模型以应对传统垂直联合学习的限制。 |

# 详细

[^1]: 微调、提示、上下文学习和指导微调：我们需要多少标记样本？

    Fine-Tuning, Prompting, In-Context Learning and Instruction-Tuning: How Many Labelled Samples Do We Need?

    [https://arxiv.org/abs/2402.12819](https://arxiv.org/abs/2402.12819)

    专门模型通常只需少量标记样本（100-1000个）就能与通用模型持平甚至更好，取决于任务的复杂性和结果的变化。

    

    当解决具有有限标记数据的任务时，研究人员可以选择使用通用的大型语言模型而不进行进一步更新，或者使用少量示例来调整专门的较小模型。 当有足够的标记可用时，专门的模型在许多自然语言处理任务上表现优于通用模型。 在这项工作中，我们旨在调查专门模型需要多少标记样本才能实现这种出色的性能，同时考虑结果的变化。观察提示、上下文学习、微调和指导微调的行为，识别它们在增加不同复杂性任务的标记训练样本数量时的收支平衡点，我们发现专门模型通常只需少量样本（100-1000个）就能与通用模型持平甚至更好。 同时，所需的标记数据量强烈依赖于任务的复杂性和结果的变化。

    arXiv:2402.12819v1 Announce Type: cross  Abstract: When solving a task with limited labelled data, researchers can either use a general large language model without further update, or use the few examples to tune a specialised smaller model. When enough labels are available, the specialised models outperform the general ones on many NLP tasks. In this work, we aim to investigate how many labelled samples are required for the specialised models to achieve this superior performance, while taking the results variance into consideration. Observing the behaviour of prompting, in-context learning, fine-tuning and instruction-tuning, identifying their break-even points when increasing number of labelled training samples across three tasks of varying complexity, we find that the specialised models often need only few samples ($100-1000$) to be on par or better than the general ones. At the same time, the amount of required labelled data strongly depends on the task complexity and results varia
    
[^2]: 全链路上升建模与上下文增强学习用于智能营销

    Entire Chain Uplift Modeling with Context-Enhanced Learning for Intelligent Marketing

    [https://arxiv.org/abs/2402.03379](https://arxiv.org/abs/2402.03379)

    全链路上升建模方法ECUP旨在解决链路偏差和处理不适应问题，在线营销中有重要的应用价值。

    

    上升建模在在线营销中非常重要，它旨在通过预测个体处理效果（ITE）来准确衡量不同策略（如优惠券或折扣）对不同用户的影响。在电子商务环境中，用户行为遵循确定的顺序链路，包括展示、点击和转化。营销策略在这个链路中的每个阶段都会产生不同的上升效应，影响着点击率和转化率等指标。尽管其实用性，现有研究忽视了特定处理中所有阶段的相互影响，并未充分利用处理信息，可能给后续的营销决策引入了重大偏差。本文将这两个问题称为链路偏差问题和处理不适应问题。本文介绍了一种用于解决这些问题的具有上下文增强学习的全链路上升方法（ECUP）。ECUP包括两个主要组成部分：

    Uplift modeling, vital in online marketing, seeks to accurately measure the impact of various strategies, such as coupons or discounts, on different users by predicting the Individual Treatment Effect (ITE). In an e-commerce setting, user behavior follows a defined sequential chain, including impression, click, and conversion. Marketing strategies exert varied uplift effects at each stage within this chain, impacting metrics like click-through and conversion rate. Despite its utility, existing research has neglected to consider the inter-task across all stages impacts within a specific treatment and has insufficiently utilized the treatment information, potentially introducing substantial bias into subsequent marketing decisions. We identify these two issues as the chain-bias problem and the treatment-unadaptive problem. This paper introduces the Entire Chain UPlift method with context-enhanced learning (ECUP), devised to tackle these issues. ECUP consists of two primary components: 1)
    
[^3]: Task Aware Dreamer用于强化学习中的任务泛化

    Task Aware Dreamer for Task Generalization in Reinforcement Learning. (arXiv:2303.05092v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.05092](http://arxiv.org/abs/2303.05092)

    本文提出了一种名为Task Aware Dreamer（TAD）的方法用于强化学习中的任务泛化。通过量化任务分布的相关性，TAD能够将历史信息编码到策略中，以便区分不同任务，并在泛化到未见任务时具有较好的性能。

    

    强化学习的一个长期目标是获得能够在训练任务上学习并且在不同奖励函数下可以很好地泛化到未见任务的代理。一个通用的挑战是定量地衡量这些不同任务之间的相似性，这对于分析任务分布并进一步设计具有更强泛化能力的算法至关重要。为了解决这个问题，我们提出了一种新的度量方法，名为任务分布相关性（TDR），通过不同任务的最优Q函数来量化任务分布的相关性。在具有高TDR的任务情况下，即任务之间显著不同，我们发现马尔可夫策略无法区分它们，导致性能较差。基于这一观察，我们将所有历史信息编码到策略中以区分不同任务，并提出了Task Aware Dreamer（TAD），它将世界模型扩展为我们的奖励感知世界模型以捕捉任务的相关性。

    A long-standing goal of reinforcement learning is to acquire agents that can learn on training tasks and generalize well on unseen tasks that may share a similar dynamic but with different reward functions. A general challenge is to quantitatively measure the similarities between these different tasks, which is vital for analyzing the task distribution and further designing algorithms with stronger generalization. To address this, we present a novel metric named Task Distribution Relevance (TDR) via optimal Q functions of different tasks to capture the relevance of the task distribution quantitatively. In the case of tasks with a high TDR, i.e., the tasks differ significantly, we show that the Markovian policies cannot differentiate them, leading to poor performance. Based on this insight, we encode all historical information into policies for distinguishing different tasks and propose Task Aware Dreamer (TAD), which extends world models into our reward-informed world models to capture
    
[^4]: 垂直半联合学习用于高效在线广告

    Vertical Semi-Federated Learning for Efficient Online Advertising. (arXiv:2209.15635v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2209.15635](http://arxiv.org/abs/2209.15635)

    垂直半联合学习为在线广告领域提供了高效的解决方案，通过学习一个联合感知的局部模型以应对传统垂直联合学习的限制。

    

    传统的垂直联合学习架构存在两个主要问题：1）适用范围受限于重叠样本；2）实时联合服务的系统挑战较高，这限制了其在广告系统中的应用。为解决这些问题，我们提出了一种新的学习设置——半垂直联合学习(Semi-VFL)，以应对这些挑战。半垂直联合学习旨在实现垂直联合学习的实际工业应用方式，通过学习一个联合感知的局部模型，该模型表现优于单方模型，同时保持了局部服务的便利性。为此，我们提出了精心设计的联合特权学习框架(JPL)，来解决被动方特征缺失和适应整个样本空间这两个问题。具体而言，我们构建了一个推理高效的适用于整个样本空间的单方学生模型，同时保持了联合特征扩展的优势。新的表示蒸馏

    The traditional vertical federated learning schema suffers from two main issues: 1) restricted applicable scope to overlapped samples and 2) high system challenge of real-time federated serving, which limits its application to advertising systems. To this end, we advocate a new learning setting Semi-VFL (Vertical Semi-Federated Learning) to tackle these challenge. Semi-VFL is proposed to achieve a practical industry application fashion for VFL, by learning a federation-aware local model which performs better than single-party models and meanwhile maintain the convenience of local-serving. For this purpose, we propose the carefully designed Joint Privileged Learning framework (JPL) to i) alleviate the absence of the passive party's feature and ii) adapt to the whole sample space. Specifically, we build an inference-efficient single-party student model applicable to the whole sample space and meanwhile maintain the advantage of the federated feature extension. New representation distilla
    

