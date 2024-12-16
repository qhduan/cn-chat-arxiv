# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Using Quantum Computing to Infer Dynamic Behaviors of Biological and Artificial Neural Networks](https://arxiv.org/abs/2403.18963) | 本研究展示了如何利用Grover和Deutsch-Josza等基础量子算法，通过一组精心构建的条件，推断生物和人工神经网络在一段时间内是否具有继续维持动态活动的潜力。 |
| [^2] | [Shifting the Lens: Detecting Malware in npm Ecosystem with Large Language Models](https://arxiv.org/abs/2403.12196) | 通过大型语言模型在npm生态系统中进行实证研究，以协助安全分析师识别恶意软件包 |
| [^3] | [Towards Multimodal Human Intention Understanding Debiasing via Subject-Deconfounding](https://arxiv.org/abs/2403.05025) | 通过引入概括性因果图和分析主题混淆效应，本文提出了SuCI，实现了多模态人类意图理解的去偏见，解决了MIU模型受主体变异问题困扰的挑战。 |
| [^4] | [TreeEval: Benchmark-Free Evaluation of Large Language Models through Tree Planning](https://arxiv.org/abs/2402.13125) | TreeEval提出了一种无基准评估方法，通过树规划策略提升了大型语言模型的评估效率和完整性 |
| [^5] | [Solid Waste Detection in Remote Sensing Images: A Survey](https://arxiv.org/abs/2402.09066) | 本文调查了固体废物在遥感图像中的检测方法。研究者利用地球观测卫星提供的高分辨率数据，通过遥感图像实现了固体废物处置场地的识别、监测和评估。 |
| [^6] | [Inverse Reinforcement Learning by Estimating Expertise of Demonstrators](https://arxiv.org/abs/2402.01886) | 本文介绍了一个新颖的框架，IRLEED，它通过估计演示者的专业知识来解决模仿学习中的次优和异质演示的问题。IRLEED通过结合演示者次优性的普适模型和最大熵IRL框架，有效地从多样的次优演示中得出最佳策略。 |
| [^7] | [Generative Ghosts: Anticipating Benefits and Risks of AI Afterlives](https://arxiv.org/abs/2402.01662) | 本文讨论了生成幽灵的潜在实施设计空间和其对个人和社会的实际和伦理影响，提出了研究议程以便使人们能够安全而有益地创建和与人工智能来世进行互动。 |
| [^8] | [AgentMixer: Multi-Agent Correlated Policy Factorization.](http://arxiv.org/abs/2401.08728) | AgentMixer提出了一种新颖的框架，允许智能体通过策略修改来实现协同决策。通过构造联合策略为各个部分策略的非线性组合，可实现部分可观测智能体的稳定训练和分散执行。 |
| [^9] | [IGNITE: Individualized GeNeration of Imputations in Time-series Electronic health records.](http://arxiv.org/abs/2401.04402) | 个体化时间序列电子健康记录的生成模型IGNITE通过学习个体的动态特征，结合人口特征和治疗信息，生成个性化的真实值，为个体化医疗提供了有价值的方式。 |

# 详细

[^1]: 使用量子计算推断生物和人工神经网络的动态行为

    Using Quantum Computing to Infer Dynamic Behaviors of Biological and Artificial Neural Networks

    [https://arxiv.org/abs/2403.18963](https://arxiv.org/abs/2403.18963)

    本研究展示了如何利用Grover和Deutsch-Josza等基础量子算法，通过一组精心构建的条件，推断生物和人工神经网络在一段时间内是否具有继续维持动态活动的潜力。

    

    新问题类别的探索是量子计算研究的一个活跃领域。一个基本上完全未被探讨的主题是使用量子算法和计算来探索和询问神经网络的功能动态。这是将量子计算应用于生物和人工神经网络建模和仿真的尚未成熟的主题的一个组成部分。在本研究中，我们展示了如何通过精心构建的一组条件来使用两个基础量子算法，Grover和Deutsch-Josza，以使输出测量具有一种解释，保证我们能够推断一个简单的神经网络表示（适用于生物和人工网络）在一段时间后是否有可能继续维持动态活动。或者这些动态保证会停止，要么是通过'癫痫'动态，要么是静止状态。

    arXiv:2403.18963v1 Announce Type: cross  Abstract: The exploration of new problem classes for quantum computation is an active area of research. An essentially completely unexplored topic is the use of quantum algorithms and computing to explore and ask questions \textit{about} the functional dynamics of neural networks. This is a component of the still-nascent topic of applying quantum computing to the modeling and simulations of biological and artificial neural networks. In this work, we show how a carefully constructed set of conditions can use two foundational quantum algorithms, Grover and Deutsch-Josza, in such a way that the output measurements admit an interpretation that guarantees we can infer if a simple representation of a neural network (which applies to both biological and artificial networks) after some period of time has the potential to continue sustaining dynamic activity. Or whether the dynamics are guaranteed to stop either through 'epileptic' dynamics or quiescence
    
[^2]: 用大型语言模型在npm生态系统中检测恶意软件

    Shifting the Lens: Detecting Malware in npm Ecosystem with Large Language Models

    [https://arxiv.org/abs/2403.12196](https://arxiv.org/abs/2403.12196)

    通过大型语言模型在npm生态系统中进行实证研究，以协助安全分析师识别恶意软件包

    

    Gartner 2022年的报告预测，到2025年，全球45%的组织将遭遇软件供应链攻击，凸显了改善软件供应链安全对社区和国家利益的迫切性。当前的恶意软件检测技术通过过滤良性和恶意软件包来辅助手动审核过程，然而这种技术存在较高的误报率和有限的自动化支持。因此，恶意软件检测技术可以受益于先进、更自动化的方法，得到准确且误报较少的结果。该研究的目标是通过对大型语言模型（LLMs）进行实证研究，帮助安全分析师识别npm生态系统中的恶意软件。

    arXiv:2403.12196v1 Announce Type: cross  Abstract: The Gartner 2022 report predicts that 45% of organizations worldwide will encounter software supply chain attacks by 2025, highlighting the urgency to improve software supply chain security for community and national interests. Current malware detection techniques aid in the manual review process by filtering benign and malware packages, yet such techniques have high false-positive rates and limited automation support. Therefore, malware detection techniques could benefit from advanced, more automated approaches for accurate and minimally false-positive results. The goal of this study is to assist security analysts in identifying malicious packages through the empirical study of large language models (LLMs) to detect potential malware in the npm ecosystem.   We present SocketAI Scanner, a multi-stage decision-maker malware detection workflow using iterative self-refinement and zero-shot-role-play-Chain of Thought (CoT) prompting techni
    
[^3]: 通过主题去相关实现多模态人类意图理解去偏见

    Towards Multimodal Human Intention Understanding Debiasing via Subject-Deconfounding

    [https://arxiv.org/abs/2403.05025](https://arxiv.org/abs/2403.05025)

    通过引入概括性因果图和分析主题混淆效应，本文提出了SuCI，实现了多模态人类意图理解的去偏见，解决了MIU模型受主体变异问题困扰的挑战。

    

    arXiv:2403.05025v1 公告类型: 新摘要: 多模态意图理解(MIU)是人类表达分析(例如情感或幽默)不可或缺的组成部分，涉及视觉姿势、语言内容和声学行为等异构模态。现有工作始终专注于设计复杂的结构或融合策略，取得显著进展。然而，它们都受到主题变异问题的困扰，因为不同主题之间的数据分布差异导致。具体而言，由于训练数据中具有不同表达习惯和特征的不同主题，MIU模型很容易被误导，以学习特定于主题的伪相关性，从而显着限制了跨未接触主题的性能和泛化能力。受这一观察启发，我们引入了一个概括性因果图来制定MIU过程，并分析主题的混淆效应。然后，我们提出了SuCI，一个简单而有效的因果

    arXiv:2403.05025v1 Announce Type: new  Abstract: Multimodal intention understanding (MIU) is an indispensable component of human expression analysis (e.g., sentiment or humor) from heterogeneous modalities, including visual postures, linguistic contents, and acoustic behaviors. Existing works invariably focus on designing sophisticated structures or fusion strategies to achieve impressive improvements. Unfortunately, they all suffer from the subject variation problem due to data distribution discrepancies among subjects. Concretely, MIU models are easily misled by distinct subjects with different expression customs and characteristics in the training data to learn subject-specific spurious correlations, significantly limiting performance and generalizability across uninitiated subjects.Motivated by this observation, we introduce a recapitulative causal graph to formulate the MIU procedure and analyze the confounding effect of subjects. Then, we propose SuCI, a simple yet effective caus
    
[^4]: TreeEval：通过树规划实现对大型语言模型的无基准评估

    TreeEval: Benchmark-Free Evaluation of Large Language Models through Tree Planning

    [https://arxiv.org/abs/2402.13125](https://arxiv.org/abs/2402.13125)

    TreeEval提出了一种无基准评估方法，通过树规划策略提升了大型语言模型的评估效率和完整性

    

    最近，建立了许多新的基准来评估大型语言模型（LLMs）的性能，通过计算整体得分或使用另一个LLM作为评判者。然而，这些方法由于基准的公开访问和评估过程的不灵活而遭受数据泄漏的困扰。为了解决这个问题，我们引入了TreeEval，这是一种无基准评估方法，让一个高性能的LLM主持一个不可重现的评估会话，从根本上避免了数据泄漏。此外，这个LLM充当一个考官，提出一系列关于一个主题的问题，并采用树规划策略，考虑当前的评估状态来决定下一个问题的生成，确保评估过程的完整性和效率。我们评估了不同参数大小的6个模型，包括7B、13B和33B，最终实现了最高的相关系数。

    arXiv:2402.13125v1 Announce Type: cross  Abstract: Recently, numerous new benchmarks have been established to evaluate the performance of large language models (LLMs) via either computing a holistic score or employing another LLM as a judge. However, these approaches suffer from data leakage due to the open access of the benchmark and inflexible evaluation process. To address this issue, we introduce $\textbf{TreeEval}$, a benchmark-free evaluation method for LLMs that let a high-performance LLM host an irreproducible evaluation session and essentially avoids the data leakage. Moreover, this LLM performs as an examiner to raise up a series of questions under a topic with a tree planing strategy, which considers the current evaluation status to decide the next question generation and ensures the completeness and efficiency of the evaluation process. We evaluate $6$ models of different parameter sizes, including $7$B, $13$B, and $33$B, and ultimately achieved the highest correlation coef
    
[^5]: 遥感图像中的固体废物检测：一项调查

    Solid Waste Detection in Remote Sensing Images: A Survey

    [https://arxiv.org/abs/2402.09066](https://arxiv.org/abs/2402.09066)

    本文调查了固体废物在遥感图像中的检测方法。研究者利用地球观测卫星提供的高分辨率数据，通过遥感图像实现了固体废物处置场地的识别、监测和评估。

    

    识别和表征非法固体废物处置场地对环境保护至关重要，特别是应对污染和健康危害。不当管理的垃圾填埋场通过雨水渗透污染土壤和地下水，对动物和人类构成威胁。传统的填埋场辨识方法，如现场检查，耗时且昂贵。遥感技术是用于识别和监测固体废物处置场地的一种经济有效的解决方案，可以实现广泛覆盖和多次获取。地球观测（EO）卫星配备了一系列传感器和成像能力，几十年来一直提供高分辨率的数据。研究人员提出了专门的技术，利用遥感图像执行一系列任务，如废物场地检测、倾倒场监测和适宜位置评估。

    arXiv:2402.09066v1 Announce Type: cross Abstract: The detection and characterization of illegal solid waste disposal sites are essential for environmental protection, particularly for mitigating pollution and health hazards. Improperly managed landfills contaminate soil and groundwater via rainwater infiltration, posing threats to both animals and humans. Traditional landfill identification approaches, such as on-site inspections, are time-consuming and expensive. Remote sensing is a cost-effective solution for the identification and monitoring of solid waste disposal sites that enables broad coverage and repeated acquisitions over time. Earth Observation (EO) satellites, equipped with an array of sensors and imaging capabilities, have been providing high-resolution data for several decades. Researchers proposed specialized techniques that leverage remote sensing imagery to perform a range of tasks such as waste site detection, dumping site monitoring, and assessment of suitable locati
    
[^6]: 通过估计演示者的专业知识的逆向强化学习

    Inverse Reinforcement Learning by Estimating Expertise of Demonstrators

    [https://arxiv.org/abs/2402.01886](https://arxiv.org/abs/2402.01886)

    本文介绍了一个新颖的框架，IRLEED，它通过估计演示者的专业知识来解决模仿学习中的次优和异质演示的问题。IRLEED通过结合演示者次优性的普适模型和最大熵IRL框架，有效地从多样的次优演示中得出最佳策略。

    

    在模仿学习中，利用次优和异质的演示提出了一个重大挑战，因为现实世界数据的性质各不相同。然而，标准的模仿学习算法将这些数据集视为同质的，从而继承了次优演示的缺陷。先前处理这个问题的方法通常依赖于不切实际的假设，如高质量的数据子集、置信度排名或明确的环境知识。本文介绍了IRLEED（通过估计演示者的专业知识的逆向强化学习），这是一个新颖的框架，能够克服这些障碍，而不需要先前对演示者专业知识进行了解。IRLEED通过将演示者次优性的普适模型与最大熵IRL框架相结合，来处理奖励偏差和行动方差，从而有效地从多样的次优演示中得出最优策略。在在线和离线实验中进行了验证。

    In Imitation Learning (IL), utilizing suboptimal and heterogeneous demonstrations presents a substantial challenge due to the varied nature of real-world data. However, standard IL algorithms consider these datasets as homogeneous, thereby inheriting the deficiencies of suboptimal demonstrators. Previous approaches to this issue typically rely on impractical assumptions like high-quality data subsets, confidence rankings, or explicit environmental knowledge. This paper introduces IRLEED, Inverse Reinforcement Learning by Estimating Expertise of Demonstrators, a novel framework that overcomes these hurdles without prior knowledge of demonstrator expertise. IRLEED enhances existing Inverse Reinforcement Learning (IRL) algorithms by combining a general model for demonstrator suboptimality to address reward bias and action variance, with a Maximum Entropy IRL framework to efficiently derive the optimal policy from diverse, suboptimal demonstrations. Experiments in both online and offline I
    
[^7]: 生成幽灵：预测人工智能来世的益处和风险

    Generative Ghosts: Anticipating Benefits and Risks of AI Afterlives

    [https://arxiv.org/abs/2402.01662](https://arxiv.org/abs/2402.01662)

    本文讨论了生成幽灵的潜在实施设计空间和其对个人和社会的实际和伦理影响，提出了研究议程以便使人们能够安全而有益地创建和与人工智能来世进行互动。

    

    随着人工智能系统在性能的广度和深度上迅速提升，它们越来越适合创建功能强大、逼真的代理人，包括基于特定人物建模的代理人的可能性。我们预计，在我们有生之年，人们可能会普遍使用定制的人工智能代理人与爱的人和/或更广大的世界进行互动。我们称之为生成幽灵，因为这些代理人将能够生成新颖的内容，而不只是复述其创作者在生前的内容。在本文中，我们首先讨论了生成幽灵潜在实施的设计空间。然后，我们讨论了生成幽灵的实际和伦理影响，包括对个人和社会的潜在积极和消极影响。基于这些考虑，我们制定了一个研究议程，旨在使人们能够安全而有益地创建和与人工智能来世进行互动。

    As AI systems quickly improve in both breadth and depth of performance, they lend themselves to creating increasingly powerful and realistic agents, including the possibility of agents modeled on specific people. We anticipate that within our lifetimes it may become common practice for people to create a custom AI agent to interact with loved ones and/or the broader world after death. We call these generative ghosts, since such agents will be capable of generating novel content rather than merely parroting content produced by their creator while living. In this paper, we first discuss the design space of potential implementations of generative ghosts. We then discuss the practical and ethical implications of generative ghosts, including potential positive and negative impacts on individuals and society. Based on these considerations, we lay out a research agenda for the AI and HCI research communities to empower people to create and interact with AI afterlives in a safe and beneficial 
    
[^8]: AgentMixer: 多智能体相关策略因子分解

    AgentMixer: Multi-Agent Correlated Policy Factorization. (arXiv:2401.08728v1 [cs.MA])

    [http://arxiv.org/abs/2401.08728](http://arxiv.org/abs/2401.08728)

    AgentMixer提出了一种新颖的框架，允许智能体通过策略修改来实现协同决策。通过构造联合策略为各个部分策略的非线性组合，可实现部分可观测智能体的稳定训练和分散执行。

    

    集中式训练与分散式执行（CTDE）广泛应用于通过在训练过程中利用集中式值函数来稳定部分可观察的多智能体强化学习（MARL）。然而，现有方法通常假设智能体基于本地观测独立地做决策，这可能不会导致具有足够协调性的相关联的联合策略。受相关均衡概念的启发，我们提出引入"策略修改"来为智能体提供协调策略的机制。具体地，我们提出了一个新颖的框架AgentMixer，将联合完全可观测策略构造为各个部分可观测策略的非线性组合。为了实现分散式执行，可以通过模仿联合策略来得到各个部分策略。不幸的是，这种模仿学习可能会导致由于联合策略和个体策略之间的不匹配而导致的非对称学习失败。

    Centralized training with decentralized execution (CTDE) is widely employed to stabilize partially observable multi-agent reinforcement learning (MARL) by utilizing a centralized value function during training. However, existing methods typically assume that agents make decisions based on their local observations independently, which may not lead to a correlated joint policy with sufficient coordination. Inspired by the concept of correlated equilibrium, we propose to introduce a \textit{strategy modification} to provide a mechanism for agents to correlate their policies. Specifically, we present a novel framework, AgentMixer, which constructs the joint fully observable policy as a non-linear combination of individual partially observable policies. To enable decentralized execution, one can derive individual policies by imitating the joint policy. Unfortunately, such imitation learning can lead to \textit{asymmetric learning failure} caused by the mismatch between joint policy and indi
    
[^9]: IGNITE: 个体化时间序列电子健康记录的生成模型

    IGNITE: Individualized GeNeration of Imputations in Time-series Electronic health records. (arXiv:2401.04402v1 [cs.LG])

    [http://arxiv.org/abs/2401.04402](http://arxiv.org/abs/2401.04402)

    个体化时间序列电子健康记录的生成模型IGNITE通过学习个体的动态特征，结合人口特征和治疗信息，生成个性化的真实值，为个体化医疗提供了有价值的方式。

    

    电子健康记录为推动个体化医疗提供了有价值的方式，可以根据个体差异量身定制治疗方案。为了实现这一目标，许多数据驱动的机器学习和统计模型借助丰富的纵向电子健康记录来研究患者的生理和治疗效果。然而，纵向电子健康记录往往稀疏且存在大量缺失，其中缺失的信息也可能反映患者的健康状况。因此，数据驱动模型在个体化医疗中的成功严重依赖于如何从生理数据、治疗以及数据中的缺失值来表示电子健康记录。为此，我们提出了一种新颖的深度学习模型，该模型可以在个体的人口特征和治疗的条件下，学习多变量数据的患者动态，并生成个性化的真实值。

    Electronic Health Records present a valuable modality for driving personalized medicine, where treatment is tailored to fit individual-level differences. For this purpose, many data-driven machine learning and statistical models rely on the wealth of longitudinal EHRs to study patients' physiological and treatment effects. However, longitudinal EHRs tend to be sparse and highly missing, where missingness could also be informative and reflect the underlying patient's health status. Therefore, the success of data-driven models for personalized medicine highly depends on how the EHR data is represented from physiological data, treatments, and the missing values in the data. To this end, we propose a novel deep-learning model that learns the underlying patient dynamics over time across multivariate data to generate personalized realistic values conditioning on an individual's demographic characteristics and treatments. Our proposed model, IGNITE (Individualized GeNeration of Imputations in
    

