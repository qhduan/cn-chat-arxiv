# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning with Noisy Foundation Models](https://arxiv.org/abs/2403.06869) | 本文首次全面了解和分析了预训练数据集中的噪声性质，有效减轻其对下游任务影响。 |
| [^2] | [On the Byzantine-Resilience of Distillation-Based Federated Learning](https://arxiv.org/abs/2402.12265) | 基于蒸馏的联邦学习在拜占庭环境下表现出极强的弹性，介绍了两种新的拜占庭攻击，并提出了一种增强拜占庭弹性的新方法。 |
| [^3] | [Jailbreaking Proprietary Large Language Models using Word Substitution Cipher](https://arxiv.org/abs/2402.10601) | 本文使用密码技术编码了越狱提示，成功地绕过了大型语言模型对有害问题的检测，实验结果显示攻击成功率高达59.42%。 |
| [^4] | [A Systematic Survey of Prompt Engineering in Large Language Models: Techniques and Applications](https://arxiv.org/abs/2402.07927) | 这篇调查论文系统概述了大型语言模型中提示工程的最新进展，探讨了提示工程的方法和技术，并说明了其在各种应用中的重要作用。 |
| [^5] | [Learning from Time Series under Temporal Label Noise](https://arxiv.org/abs/2402.04398) | 该论文研究了在时间序列下处理时间标签噪声的问题，提出了一种可以从数据中直接估计时间标签噪声函数并训练出噪声容忍分类器的方法，并在实验中展示了该方法在各种时间标签噪声函数下都取得了最先进的性能。 |
| [^6] | [SWBT: Similarity Weighted Behavior Transformer with the Imperfect Demonstration for Robotic Manipulation.](http://arxiv.org/abs/2401.08957) | 本论文提出了一种新型框架SWBT，能够在机器人操作任务中有效地从专家演示和不完美演示中学习，而无需与环境进行交互。这是第一个将不完美演示整合到离线模仿学习设置中的机器人操作任务的研究。 |
| [^7] | [Leveraging Large Language Models for Collective Decision-Making.](http://arxiv.org/abs/2311.04928) | 本论文提出了一种利用大型语言模型（LLM）促进集体决策的系统，通过管理对话和平衡个人偏好来提供满足成员需求的选项，实现高效协调并不断优化系统性能。 |
| [^8] | [Automated Verification of Equivalence Properties in Advanced Logic Programs -- Bachelor Thesis.](http://arxiv.org/abs/2310.19806) | 这篇论文介绍了一种自动验证工具，用于验证优化的逻辑子程序是否可以替代原始子程序，在工业应用中具有重要意义。 |
| [^9] | [Life-inspired Interoceptive Artificial Intelligence for Autonomous and Adaptive Agents.](http://arxiv.org/abs/2309.05999) | 该论文提出了一种基于生命理论和控制论的新视角，通过将控制论、强化学习和神经科学的最新进展与生命理论相结合，将内感知应用于构建具有自主和适应性能力的人工智能代理。 |
| [^10] | [Job Shop Scheduling Benchmark: Environments and Instances for Learning and Non-learning Methods.](http://arxiv.org/abs/2308.12794) | 这个开源的GitHub仓库为机器调度问题提供了综合基准，包括多种环境和实例，为研究人员和从业者提供了一个集中的中心。 |
| [^11] | [Valley: Video Assistant with Large Language model Enhanced abilitY.](http://arxiv.org/abs/2306.07207) | 本文介绍了一个名为Valley的视频助手，它是一个以大型语言模型增强的多模态基础模型，能够在一个通用框架内理解视频、图像和语言。 |
| [^12] | [A Comprehensive Overview and Comparative Analysis on Deep Learning Models: CNN, RNN, LSTM, GRU.](http://arxiv.org/abs/2305.17473) | 本文全面概括了深度学习模型的类型和应用，比较分析了各个模型的结构、优点和局限性，有助于选择和设计深度学习模型。 |
| [^13] | [Model Stealing Attack against Multi-Exit Networks.](http://arxiv.org/abs/2305.13584) | 该论文介绍了第一个能同时窃取多出口网络模型函数和输出策略的攻击方法，并使用贝叶斯变点检测和性能损失、策略损失指导替代模型的训练。开发了一种新的输出策略搜索方法。 |
| [^14] | [A Survey on Deep Learning based Time Series Analysis with Frequency Transformation.](http://arxiv.org/abs/2302.02173) | 近期，频率变换（FT）在深度学习时间序列分析中得到广泛应用，显著提高了准确性和效率。本文系统回顾和总结了基于FT的深度学习时间序列模型的研究进展，并探讨了其优势、限制以及主要方法。 |

# 详细

[^1]: 在有噪声基础模型中学习

    Learning with Noisy Foundation Models

    [https://arxiv.org/abs/2403.06869](https://arxiv.org/abs/2403.06869)

    本文首次全面了解和分析了预训练数据集中的噪声性质，有效减轻其对下游任务影响。

    

    基础模型通常是在大规模数据集上进行预训练，然后通过调整来适应下游任务。然而，大规模预训练数据集往往无法获取或成本过高，可能包含标签噪声，这可能会对模型的泛化能力造成不利影响，并带来意想不到的风险。本文是首个全面了解和分析预训练数据集中噪声性质，并有效减轻其对下游任务影响的工作。具体而言，通过在合成有噪声的ImageNet-1K、YFCC15M和CC12M数据集上进行完全监督和图像-文本对比预训练的广泛实验，我们证明了，尽管预训练中的轻微噪声可以使同领域（ID）性能受益，即训练和测试数据共享类似分布，但它总是会破坏跨领域（OOD）性能，在那里训练和测试分布明显不同。

    arXiv:2403.06869v1 Announce Type: cross  Abstract: Foundation models are usually pre-trained on large-scale datasets and then adapted to downstream tasks through tuning. However, the large-scale pre-training datasets, often inaccessible or too expensive to handle, can contain label noise that may adversely affect the generalization of the model and pose unexpected risks. This paper stands out as the first work to comprehensively understand and analyze the nature of noise in pre-training datasets and then effectively mitigate its impacts on downstream tasks. Specifically, through extensive experiments of fully-supervised and image-text contrastive pre-training on synthetic noisy ImageNet-1K, YFCC15M, and CC12M datasets, we demonstrate that, while slight noise in pre-training can benefit in-domain (ID) performance, where the training and testing data share a similar distribution, it always deteriorates out-of-domain (OOD) performance, where training and testing distributions are signific
    
[^2]: 论基于蒸馏的联邦学习在拜占庭环境下的弹性

    On the Byzantine-Resilience of Distillation-Based Federated Learning

    [https://arxiv.org/abs/2402.12265](https://arxiv.org/abs/2402.12265)

    基于蒸馏的联邦学习在拜占庭环境下表现出极强的弹性，介绍了两种新的拜占庭攻击，并提出了一种增强拜占庭弹性的新方法。

    

    由于在隐私、非独立同分布数据和通信成本方面的优势，使用知识蒸馏（KD）的联邦学习（FL）算法受到越来越多的关注。本文研究了这些方法在拜占庭环境中的性能，展示了基于KD的FL算法相当具有弹性，并分析了拜占庭客户端如何影响学习过程相对于联邦平均算法。根据这些见解，我们介绍了两种新的拜占庭攻击，并证明它们对先前的拜占庭弹性方法是有效的。此外，我们提出了FilterExp，一种旨在增强拜占庭弹性的新方法。

    arXiv:2402.12265v1 Announce Type: cross  Abstract: Federated Learning (FL) algorithms using Knowledge Distillation (KD) have received increasing attention due to their favorable properties with respect to privacy, non-i.i.d. data and communication cost. These methods depart from transmitting model parameters and, instead, communicate information about a learning task by sharing predictions on a public dataset. In this work, we study the performance of such approaches in the byzantine setting, where a subset of the clients act in an adversarial manner aiming to disrupt the learning process. We show that KD-based FL algorithms are remarkably resilient and analyze how byzantine clients can influence the learning process compared to Federated Averaging. Based on these insights, we introduce two new byzantine attacks and demonstrate that they are effective against prior byzantine-resilient methods. Additionally, we propose FilterExp, a novel method designed to enhance the byzantine resilien
    
[^3]: 使用单词替换密码来越狱专有的大型语言模型

    Jailbreaking Proprietary Large Language Models using Word Substitution Cipher

    [https://arxiv.org/abs/2402.10601](https://arxiv.org/abs/2402.10601)

    本文使用密码技术编码了越狱提示，成功地绕过了大型语言模型对有害问题的检测，实验结果显示攻击成功率高达59.42%。

    

    大型语言模型（LLMs）遵循道德和伦理准则，但仍然容易受到名为Jailbreak的创意提示的影响，这些提示可以绕过对齐过程。然而，大多数越狱提示包含自然语言（主要是英语）中的有害问题，可以被LLMs自身检测到。本文提出了使用密码技术编码的越狱提示。我们首先在最先进的LLM，GPT-4上进行了一个试点研究，解码了使用各种密码技术加密的几个安全句子，发现简单的单词替换密码可以被最有效地解码。受此结果启发，我们使用这种编码技术来编写越狱提示。我们提供了将不安全单词映射到安全单词，并使用这些映射的单词提出不安全问题的映射。实验结果显示，我们提出的越狱攻击成功率（高达59.42%）。

    arXiv:2402.10601v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are aligned to moral and ethical guidelines but remain susceptible to creative prompts called Jailbreak that can bypass the alignment process. However, most jailbreaking prompts contain harmful questions in the natural language (mainly English), which can be detected by the LLM themselves. In this paper, we present jailbreaking prompts encoded using cryptographic techniques. We first present a pilot study on the state-of-the-art LLM, GPT-4, in decoding several safe sentences that have been encrypted using various cryptographic techniques and find that a straightforward word substitution cipher can be decoded most effectively. Motivated by this result, we use this encoding technique for writing jailbreaking prompts. We present a mapping of unsafe words with safe words and ask the unsafe question using these mapped words. Experimental results show an attack success rate (up to 59.42%) of our proposed jailbrea
    
[^4]: 大型语言模型中提示工程的系统调查：技术和应用

    A Systematic Survey of Prompt Engineering in Large Language Models: Techniques and Applications

    [https://arxiv.org/abs/2402.07927](https://arxiv.org/abs/2402.07927)

    这篇调查论文系统概述了大型语言模型中提示工程的最新进展，探讨了提示工程的方法和技术，并说明了其在各种应用中的重要作用。

    

    提示工程已成为扩展大型语言模型（LLM）和视觉语言模型（VLM）能力的不可或缺的技术。该方法利用任务特定的指令（称为提示）在不修改核心模型参数的情况下增强模型的效果。提示允许将预训练模型无缝集成到下游任务中，仅根据给定的提示引发所需的模型行为，而不是更新模型参数。提示可以是提供上下文以指导模型的自然语言指令，也可以是调用相关知识的学习向量表示。这个新兴领域在各种应用中取得了成功，从问答到常识推理都有涉及。然而，对于多样的提示工程方法和技术缺乏系统的组织和理解。本调查论文通过提供对最近进展的结构化概述来填补这一空白。

    Prompt engineering has emerged as an indispensable technique for extending the capabilities of large language models (LLMs) and vision-language models (VLMs). This approach leverages task-specific instructions, known as prompts, to enhance model efficacy without modifying the core model parameters. Rather than updating the model parameters, prompts allow seamless integration of pre-trained models into downstream tasks by eliciting desired model behaviors solely based on the given prompt. Prompts can be natural language instructions that provide context to guide the model or learned vector representations that activate relevant knowledge. This burgeoning field has enabled success across various applications, from question-answering to commonsense reasoning. However, there remains a lack of systematic organization and understanding of the diverse prompt engineering methods and techniques. This survey paper addresses the gap by providing a structured overview of recent advancements in pro
    
[^5]: 学习在时间序列下处理时间标签噪声

    Learning from Time Series under Temporal Label Noise

    [https://arxiv.org/abs/2402.04398](https://arxiv.org/abs/2402.04398)

    该论文研究了在时间序列下处理时间标签噪声的问题，提出了一种可以从数据中直接估计时间标签噪声函数并训练出噪声容忍分类器的方法，并在实验中展示了该方法在各种时间标签噪声函数下都取得了最先进的性能。

    

    许多顺序分类任务受到随时间变化的标签噪声的影响。这种噪声可能会导致标签质量随时间改善、恶化或周期性变化。我们首先提出和系统化了时间标签噪声的概念，这是关于时间序列顺序分类的一个未经研究的问题。在这种设置下，多个标签连续记录，同时受到一个与时间相关的噪声函数的干扰。我们首先展示了建模时间标签噪声函数的重要性，以及现有方法的持续低效。然后，我们提出了一种直接从数据中估计时间标签噪声函数的方法，可以训练出对噪声具有容忍性的分类器。我们展示了我们的方法在各种各样的时间标签噪声函数下，使用真实和合成数据在性能上达到了最先进水平。

    Many sequential classification tasks are affected by label noise that varies over time. Such noise can cause label quality to improve, worsen, or periodically change over time. We first propose and formalize temporal label noise, an unstudied problem for sequential classification of time series. In this setting, multiple labels are recorded in sequence while being corrupted by a time-dependent noise function. We first demonstrate the importance of modelling the temporal nature of the label noise function and how existing methods will consistently underperform. We then propose methods that can train noise-tolerant classifiers by estimating the temporal label noise function directly from data. We show that our methods lead to state-of-the-art performance in the presence of diverse temporal label noise functions using real and synthetic data.
    
[^6]: SWBT：具有不完美演示的相似性加权行为转换器用于机器人操作

    SWBT: Similarity Weighted Behavior Transformer with the Imperfect Demonstration for Robotic Manipulation. (arXiv:2401.08957v1 [cs.RO])

    [http://arxiv.org/abs/2401.08957](http://arxiv.org/abs/2401.08957)

    本论文提出了一种新型框架SWBT，能够在机器人操作任务中有效地从专家演示和不完美演示中学习，而无需与环境进行交互。这是第一个将不完美演示整合到离线模仿学习设置中的机器人操作任务的研究。

    

    模仿学习旨在从专家演示中学习最佳控制策略，已成为机器人操作任务的有效方法。然而，先前的模仿学习方法要么仅使用昂贵的专家演示并忽略不完美的演示，要么依赖于与环境的交互和从在线经验中学习。在机器人操作的背景下，我们旨在克服上述两个挑战，并提出了一种名为Similarity Weighted Behavior Transformer（SWBT）的新型框架。SWBT能够有效地从专家演示和不完美演示中学习，而无需与环境进行交互。我们揭示了易获取的不完美演示，如正向和反向动力学，通过学习有益信息显著增强了网络。据我们所知，我们是第一个尝试将不完美演示整合到离线模仿学习设置中的机器人操作任务中的研究。在ManiSkill2 bench上进行了大量实验。

    Imitation learning (IL), aiming to learn optimal control policies from expert demonstrations, has been an effective method for robot manipulation tasks. However, previous IL methods either only use expensive expert demonstrations and omit imperfect demonstrations or rely on interacting with the environment and learning from online experiences. In the context of robotic manipulation, we aim to conquer the above two challenges and propose a novel framework named Similarity Weighted Behavior Transformer (SWBT). SWBT effectively learn from both expert and imperfect demonstrations without interaction with environments. We reveal that the easy-to-get imperfect demonstrations, such as forward and inverse dynamics, significantly enhance the network by learning fruitful information. To the best of our knowledge, we are the first to attempt to integrate imperfect demonstrations into the offline imitation learning setting for robot manipulation tasks. Extensive experiments on the ManiSkill2 bench
    
[^7]: 利用大型语言模型进行集体决策

    Leveraging Large Language Models for Collective Decision-Making. (arXiv:2311.04928v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2311.04928](http://arxiv.org/abs/2311.04928)

    本论文提出了一种利用大型语言模型（LLM）促进集体决策的系统，通过管理对话和平衡个人偏好来提供满足成员需求的选项，实现高效协调并不断优化系统性能。

    

    在各种工作环境中，如会议安排、合作和项目规划中，集体决策是必不可少的，但由于个体偏好多样性、工作焦点不同和成员之间的权力动态等因素，常常具有挑战性。为了解决这个问题，我们提出了一种利用大型语言模型（LLM）来促进群体决策的系统，通过管理对话和平衡个人偏好来实现。我们的系统旨在从对话中提取个体偏好，并提出满足成员偏好的选项。我们特别将此系统应用于企业会议安排。我们利用LLM创建了合成员工配置文件，并模拟了大规模的对话，通过利用LLM评估系统表现来作为开展用户研究的新方法。我们的结果表明，系统能实现成员与LLM系统之间的高效协调，并随着时间的推移对其提出的选项进行改进和完善，确保优化系统性能。

    In various work contexts, such as meeting scheduling, collaborating, and project planning, collective decision-making is essential but often challenging due to diverse individual preferences, varying work focuses, and power dynamics among members. To address this, we propose a system leveraging Large Language Models (LLMs) to facilitate group decision-making by managing conversations and balancing preferences among individuals. Our system aims to extract individual preferences from conversations and suggest options that satisfy the preferences of the members. We specifically apply this system to corporate meeting scheduling. We create synthetic employee profiles and simulate conversations at scale, leveraging LLMs to evaluate the system performance as a novel approach to conducting a user study. Our results indicate efficient coordination with reduced interactions between the members and the LLM-based system. The system refines and improves its proposed options over time, ensuring that
    
[^8]: 高级逻辑程序等价性属性的自动验证-学士论文

    Automated Verification of Equivalence Properties in Advanced Logic Programs -- Bachelor Thesis. (arXiv:2310.19806v1 [cs.LO])

    [http://arxiv.org/abs/2310.19806](http://arxiv.org/abs/2310.19806)

    这篇论文介绍了一种自动验证工具，用于验证优化的逻辑子程序是否可以替代原始子程序，在工业应用中具有重要意义。

    

    随着使用答案集编程的工业应用增加，对形式验证工具，特别是对关键应用的需求也增加了。在程序优化过程中，希望有一种工具可以自动验证优化的子程序是否可以替代原始子程序。从形式上讲，这对应于验证两个程序的强等价性的问题。为了做到这一点，开发了翻译工具anthem。它可以与用于经典逻辑的自动定理证明器一起使用，以验证两个程序是否强等价。在当前版本的anthem中，只能验证具有受限输入语言的正程序的强等价性。这是anthem中实现的翻译τ*的结果，它生成了here-and-there逻辑中的公式，该逻辑只对正程序与经典逻辑相一致。这篇论文扩展了anthem，以便可以验证更广泛的高级逻辑程序的强等价性。

    With the increase in industrial applications using Answer Set Programming, the need for formal verification tools, particularly for critical applications, has also increased. During the program optimisation process, it would be desirable to have a tool which can automatically verify whether an optimised subprogram can replace the original subprogram. Formally this corresponds to the problem of verifying the strong equivalence of two programs. In order to do so, the translation tool anthem was developed. It can be used in conjunction with an automated theorem prover for classical logic to verify that two programs are strongly equivalent. With the current version of anthem, only the strong equivalence of positive programs with a restricted input language can be verified. This is a result of the translation $\tau^*$ implemented in anthem that produces formulas in the logic of here-and-there, which coincides with classical logic only for positive programs. This thesis extends anthem in ord
    
[^9]: 生命启发的自主和适应智能为自主和适应性代理构建具有自主能力和自适应能力的代理一直是人工智能（AI）的终极目标。生物体是这样一个代理的最好例证，它为自适应自主性提供了重要的经验教训。在这里，我们关注内感知，这是一个监控自身内部环境来保持在一定范围内的过程，它为生物体的生存提供了基础。为了开发具有内感知的AI，我们需要将表示内部环境的状态变量与外部环境相分离，并采用生命启发的内部环境状态的数学特性。本文提出了一个新的视角，即通过将控制论、强化学习和神经科学的最新进展与生命理论相结合，内感知如何帮助构建自主和适应性代理。

    Life-inspired Interoceptive Artificial Intelligence for Autonomous and Adaptive Agents. (arXiv:2309.05999v1 [cs.AI])

    [http://arxiv.org/abs/2309.05999](http://arxiv.org/abs/2309.05999)

    该论文提出了一种基于生命理论和控制论的新视角，通过将控制论、强化学习和神经科学的最新进展与生命理论相结合，将内感知应用于构建具有自主和适应性能力的人工智能代理。

    

    构建具有自主能力和自适应能力的代理一直是人工智能（AI）的终极目标。生物体是这样一个代理的最好例证，它为自适应自主性提供了重要的经验教训。本文关注内感知，这是一个监控自身内部环境来保持在一定范围内的过程，它为生物体的生存提供了基础。为了开发具有内感知的AI，我们需要将表示内部环境的状态变量与外部环境相分离，并采用生命启发的内部环境状态的数学特性。本文提出了一个新的视角，即通过将控制论、强化学习和神经科学的最新进展与生命理论相结合，内感知如何帮助构建自主和适应性代理。

    Building autonomous --- i.e., choosing goals based on one's needs -- and adaptive -- i.e., surviving in ever-changing environments -- agents has been a holy grail of artificial intelligence (AI). A living organism is a prime example of such an agent, offering important lessons about adaptive autonomy. Here, we focus on interoception, a process of monitoring one's internal environment to keep it within certain bounds, which underwrites the survival of an organism. To develop AI with interoception, we need to factorize the state variables representing internal environments from external environments and adopt life-inspired mathematical properties of internal environment states. This paper offers a new perspective on how interoception can help build autonomous and adaptive agents by integrating the legacy of cybernetics with recent advances in theories of life, reinforcement learning, and neuroscience.
    
[^10]: 工作车间调度基准：用于学习和非学习方法的环境和实例

    Job Shop Scheduling Benchmark: Environments and Instances for Learning and Non-learning Methods. (arXiv:2308.12794v1 [cs.AI])

    [http://arxiv.org/abs/2308.12794](http://arxiv.org/abs/2308.12794)

    这个开源的GitHub仓库为机器调度问题提供了综合基准，包括多种环境和实例，为研究人员和从业者提供了一个集中的中心。

    

    我们介绍了一个开源的GitHub仓库，其中包含了广泛的机器调度问题的综合基准，包括工作车间调度（JSP），流水车间调度（FSP），灵活工作车间调度（FJSP），具有装配约束的FJSP（FAJSP），具有序列依赖设置时间的FJSP（FJSP-SDST）和在线FJSP（在线作业到达）。我们的主要目标是为对机器调度挑战感兴趣的研究人员，从业者和爱好者提供一个集中的中心。

    We introduce an open-source GitHub repository containing comprehensive benchmarks for a wide range of machine scheduling problems, including Job Shop Scheduling (JSP), Flow Shop Scheduling (FSP), Flexible Job Shop Scheduling (FJSP), FJSP with Assembly constraints (FAJSP), FJSP with Sequence-Dependent Setup Times (FJSP-SDST), and the online FJSP (with online job arrivals). Our primary goal is to provide a centralized hub for researchers, practitioners, and enthusiasts interested in tackling machine scheduling challenges.
    
[^11]: Valley: 大型语言模型增强视频助手

    Valley: Video Assistant with Large Language model Enhanced abilitY. (arXiv:2306.07207v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2306.07207](http://arxiv.org/abs/2306.07207)

    本文介绍了一个名为Valley的视频助手，它是一个以大型语言模型增强的多模态基础模型，能够在一个通用框架内理解视频、图像和语言。

    

    大型语言模型(LLMs)以其卓越的会话能力，在各种应用中表现出色，并成为强大的AI助手。鉴于此，一个直观的问题是：我们能否利用LLMs的能力构建多模态的视觉应用AI助手？最近，已经开发了几个多模态模型来实现这个目的。它们通常预先训练一个适应模块来对齐视觉编码器和语言模型的语义，然后在指令跟随数据上进行微调。然而，尽管这个流程在图像和语言理解方面取得了成功，在视频和语言理解方面的有效性还没有得到广泛探索。在本文中，我们旨在开发一个能够在一个通用框架内理解视频、图像和语言的新型多模态基础模型。为了实现这一目标，我们引入了Valley，一个以大型语言模型增强的视频助手。

    Large language models (LLMs), with their remarkable conversational capabilities, have demonstrated impressive performance across various applications and have emerged as formidable AI assistants. In view of this, it raises an intuitive question: Can we harness the power of LLMs to build multimodal AI assistants for visual applications? Recently, several multi-modal models have been developed for this purpose. They typically pre-train an adaptation module to align the semantics of the vision encoder and language model, followed by fine-tuning on instruction-following data. However, despite the success of this pipeline in image and language understanding, its effectiveness in joint video and language understanding has not been widely explored. In this paper, we aim to develop a novel multi-modal foundation model capable of comprehending video, image, and language within a general framework. To achieve this goal, we introduce Valley, a Video Assistant with Large Language model Enhanced ab
    
[^12]: 深度学习模型概述与比较分析：CNN、RNN、LSTM、GRU。

    A Comprehensive Overview and Comparative Analysis on Deep Learning Models: CNN, RNN, LSTM, GRU. (arXiv:2305.17473v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2305.17473](http://arxiv.org/abs/2305.17473)

    本文全面概括了深度学习模型的类型和应用，比较分析了各个模型的结构、优点和局限性，有助于选择和设计深度学习模型。

    

    深度学习（DL）是机器学习（ML）和人工智能（AI）的强大子集，特别在处理非结构化和大型数据集方面优于传统的ML方法。其影响跨越各个领域，包括语音识别、医疗保健、自动驾驶汽车、网络安全、预测分析等。然而，实际问题的复杂性和动态性给设计有效的深度学习模型带来了挑战。因此，人们开发出了几种不同的深度学习模型来解决不同的问题和应用。在本文中，我们对各种深度学习模型进行了全面调查，包括卷积神经网络（CNN）、循环神经网络（RNN）、生成模型、深度强化学习（DRL）和深度迁移学习。我们考察了每个模型的结构、应用、好处和局限性。此外，我们使用了三个公开可用的数据集进行了分析。

    Deep learning (DL) has emerged as a powerful subset of machine learning (ML) and artificial intelligence (AI), outperforming traditional ML methods, especially in handling unstructured and large datasets. Its impact spans across various domains, including speech recognition, healthcare, autonomous vehicles, cybersecurity, predictive analytics, and more. However, the complexity and dynamic nature of real-world problems present challenges in designing effective deep learning models. Consequently, several deep learning models have been developed to address different problems and applications. In this article, we conduct a comprehensive survey of various deep learning models, including Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), Generative Models, Deep Reinforcement Learning (DRL), and Deep Transfer Learning. We examine the structure, applications, benefits, and limitations of each model. Furthermore, we perform an analysis using three publicly available dataset
    
[^13]: 针对多出口网络的模型窃取攻击

    Model Stealing Attack against Multi-Exit Networks. (arXiv:2305.13584v1 [cs.CR])

    [http://arxiv.org/abs/2305.13584](http://arxiv.org/abs/2305.13584)

    该论文介绍了第一个能同时窃取多出口网络模型函数和输出策略的攻击方法，并使用贝叶斯变点检测和性能损失、策略损失指导替代模型的训练。开发了一种新的输出策略搜索方法。

    

    与具有单个出口的传统神经网络相比，多出口网络具有多个出口，这些出口允许从模型的中间层早期输出，从而在保持类似识别精度的情况下提高计算效率。当使用传统的模型窃取攻击方法尝试窃取这些有价值的模型时，我们发现传统方法只能窃取模型的分类函数，而不能捕捉其输出策略。这导致窃取的替代模型的计算效率显著降低，失去多出口网络的优点。在本文中，我们提出了第一个窃取模型攻击，可以提取模型函数和输出策略。我们采用贝叶斯变点检测来分析目标模型的输出策略，并使用性能损失和策略损失来指导替代模型的训练。此外，我们设计了一种新颖的输出策略搜索方法，以使替代模型还原窃取目标模型的输出策略。

    Compared to traditional neural networks with a single exit, a multi-exit network has multiple exits that allow for early output from intermediate layers of the model, thus bringing significant improvement in computational efficiency while maintaining similar recognition accuracy. When attempting to steal such valuable models using traditional model stealing attacks, we found that conventional methods can only steal the model's classification function while failing to capture its output strategy. This results in a significant decrease in computational efficiency for the stolen substitute model, thereby losing the advantages of multi-exit networks.In this paper, we propose the first model stealing attack to extract both the model function and output strategy. We employ bayesian changepoint detection to analyze the target model's output strategy and use performance loss and strategy loss to guide the training of the substitute model. Furthermore, we designed a novel output strategy search
    
[^14]: 基于频率变换的深度学习时间序列分析综述

    A Survey on Deep Learning based Time Series Analysis with Frequency Transformation. (arXiv:2302.02173v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.02173](http://arxiv.org/abs/2302.02173)

    近期，频率变换（FT）在深度学习时间序列分析中得到广泛应用，显著提高了准确性和效率。本文系统回顾和总结了基于FT的深度学习时间序列模型的研究进展，并探讨了其优势、限制以及主要方法。

    

    最近，频率变换（FT）越来越多地被纳入深度学习模型中，可以显著提高时间序列分析的最新准确性和效率。频率变换的优势，如高效性和全局视角，在各种时间序列任务和应用中被迅速探索和利用，展示了频率变换作为一种新的深度学习范式在时间序列分析领域的潜力。尽管这个新兴领域受到了越来越多的关注和研究，但目前还缺乏对基于频率变换的深度学习时间序列模型的系统回顾和深入分析。目前还不清楚为什么频率变换可以提升时间序列分析的效果，以及它在该领域的限制是什么。为了填补这些空白，我们提供了一份全面的综述，系统调查和总结了基于频率变换的深度学习时间序列分析的最新研究进展。具体而言，我们探讨了主要的方法。

    Recently, frequency transformation (FT) has been increasingly incorporated into deep learning models to significantly enhance state-of-the-art accuracy and efficiency in time series analysis. The advantages of FT, such as high efficiency and a global view, have been rapidly explored and exploited in various time series tasks and applications, demonstrating the promising potential of FT as a new deep learning paradigm for time series analysis. Despite the growing attention and the proliferation of research in this emerging field, there is currently a lack of a systematic review and in-depth analysis of deep learning-based time series models with FT. It is also unclear why FT can enhance time series analysis and what its limitations in the field are. To address these gaps, we present a comprehensive review that systematically investigates and summarizes the recent research advancements in deep learning-based time series analysis with FT. Specifically, we explore the primary approaches us
    

