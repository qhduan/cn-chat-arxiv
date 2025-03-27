# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TwoStep: Multi-agent Task Planning using Classical Planners and Large Language Models](https://arxiv.org/abs/2403.17246) | 该论文将经典规划和大型语言模型相结合，通过近似人类直觉，以实现多智能体任务规划。 |
| [^2] | [Rethinking Low-quality Optical Flow in Unsupervised Surgical Instrument Segmentation](https://arxiv.org/abs/2403.10039) | 本研究在无监督手术器械分割中解决了由于低质量光流而引起的挑战，提出了一种三重策略：直接从光流中提取边界、选择性丢弃质量较差的帧、以及利用可变帧率进行微调。在数据集上进行了充分评估，展示出有前景的结果。 |
| [^3] | [Lemur: Log Parsing with Entropy Sampling and Chain-of-Thought Merging](https://arxiv.org/abs/2402.18205) | Lemur提出了一种先进的日志解析框架，采用熵抽样和思维链合并，解决了日志解析中存在的人工规则依赖和语义信息忽略等问题。 |
| [^4] | [Graph-Informed Neural Networks for Sparse Grid-Based Discontinuity Detectors.](http://arxiv.org/abs/2401.13652) | 本文提出了一种利用图信息神经网络和稀疏网格来检测不连续函数不连续界面的新方法，该方法在维度大于3的情况下表现出高效且准确的不连续性检测能力，在维度n = 2和n = 4的函数上进行的实验验证了其高效性和泛化能力，并具有可移植性和多功能性。 |
| [^5] | [Truck Parking Usage Prediction with Decomposed Graph Neural Networks.](http://arxiv.org/abs/2401.12920) | 提出了Regional Temporal Graph Neural Network (RegT-GCN)作为一个预测框架，用于评估整个州的卡车停车使用情况，以提供更准确的停车信息并缓解未经授权的停车问题。 |
| [^6] | [Semiring Provenance for Lightweight Description Logics.](http://arxiv.org/abs/2310.16472) | 这篇论文研究了在描述逻辑中使用半环溯源的框架，并定义了一种适用于轻量级描述逻辑的溯源语义。论文证明了在半环施加限制的情况下，语义满足一些重要的特性，并对why溯源方法进行了研究。 |
| [^7] | [Certified Robustness via Dynamic Margin Maximization and Improved Lipschitz Regularization.](http://arxiv.org/abs/2310.00116) | 本文提出了一种基于动态边界最大化和改进的Lipschitz正则化的认证鲁棒性训练算法，通过增加输出空间中的边界和正则化模型的Lipschitz常数来提高深度分类器对抗性扰动的鲁棒性。 |
| [^8] | [Framework for developing quantitative agent based models based on qualitative expert knowledge: an organised crime use-case.](http://arxiv.org/abs/2308.00505) | 提出了一个基于定性专家知识的量化代理模型开发框架，该框架通过将定性数据翻译成定量规则，为模型构建者和领域专家提供了一个系统和透明的建模过程。以一个有组织犯罪的应用案例为例，演示了该框架的方法。 |
| [^9] | [Synergies Between Federated Learning and O-RAN: Towards an Elastic Virtualized Architecture for Multiple Distributed Machine Learning Services.](http://arxiv.org/abs/2305.02109) | 本文研究了联邦学习在现代无线网络下的挑战，提出了一种方法称为动态多服务联邦学习（DMS-FL）来解决这个问题。同时，还提出了一种名为弹性虚拟化联邦学习（EV-FL）的分布式机器学习架构，来支持DMS-FL中的设计要求。 |
| [^10] | [Making AI Less "Thirsty": Uncovering and Addressing the Secret Water Footprint of AI Models.](http://arxiv.org/abs/2304.03271) | 本论文揭示以及提出了解决人工智能模型巨大水足迹的方法，因为其淡水消耗已经引起国际社会的重视，并且AI模型应该承担社会责任，做出面对水危机的表率。 |

# 详细

[^1]: TwoStep: 使用经典规划器和大型语言模型进行多智能体任务规划

    TwoStep: Multi-agent Task Planning using Classical Planners and Large Language Models

    [https://arxiv.org/abs/2403.17246](https://arxiv.org/abs/2403.17246)

    该论文将经典规划和大型语言模型相结合，通过近似人类直觉，以实现多智能体任务规划。

    

    类似规划领域定义语言（PDDL）之类的经典规划公式允许确定可实现目标状态的动作序列，只要存在任何可能的初始状态。然而，PDDL中定义的推理问题并未捕获行动进行的时间方面，例如领域中的两个智能体如果彼此的后况不干扰前提条件，则可以同时执行一个动作。人类专家可以将目标分解为大部分独立的组成部分，并将每个智能体分配给其中一个子目标，以利用同时进行动作来加快计划步骤的执行，每个部分仅使用单个智能体规划。相比之下，直接推断计划步骤的大型语言模型（LLMs）并不保证执行成功，但利用常识推理来组装动作序列。我们通过近似人类直觉，结合了经典规划和LLMs的优势

    arXiv:2403.17246v1 Announce Type: new  Abstract: Classical planning formulations like the Planning Domain Definition Language (PDDL) admit action sequences guaranteed to achieve a goal state given an initial state if any are possible. However, reasoning problems defined in PDDL do not capture temporal aspects of action taking, for example that two agents in the domain can execute an action simultaneously if postconditions of each do not interfere with preconditions of the other. A human expert can decompose a goal into largely independent constituent parts and assign each agent to one of these subgoals to take advantage of simultaneous actions for faster execution of plan steps, each using only single agent planning. By contrast, large language models (LLMs) used for directly inferring plan steps do not guarantee execution success, but do leverage commonsense reasoning to assemble action sequences. We combine the strengths of classical planning and LLMs by approximating human intuition
    
[^2]: 重新思考无监督手术器械分割中低质量光流问题

    Rethinking Low-quality Optical Flow in Unsupervised Surgical Instrument Segmentation

    [https://arxiv.org/abs/2403.10039](https://arxiv.org/abs/2403.10039)

    本研究在无监督手术器械分割中解决了由于低质量光流而引起的挑战，提出了一种三重策略：直接从光流中提取边界、选择性丢弃质量较差的帧、以及利用可变帧率进行微调。在数据集上进行了充分评估，展示出有前景的结果。

    

    视频的手术器械分割在机器人辅助手术中扮演着重要角色。与监督设置不同，无监督分割主要依赖于运动线索，然而由于手术镜头中光流通常比自然场景中的要低质量，这些运动线索很难识别。本研究致力于解决即使面对低质量光流固有限制，提高模型性能的挑战。我们的方法从三个方面入手：直接从光流中提取边界、有选择地丢弃质量较差的帧、以及利用可变帧率的微调过程。我们在EndoVis2017 VOS数据集和Endovis2017挑战数据集上对我们的策略进行了彻底评估，模型展现出有前景的结果，实现了均值交叉。

    arXiv:2403.10039v1 Announce Type: cross  Abstract: Video-based surgical instrument segmentation plays an important role in robot-assisted surgeries. Unlike supervised settings, unsupervised segmentation relies heavily on motion cues, which are challenging to discern due to the typically lower quality of optical flow in surgical footage compared to natural scenes. This presents a considerable burden for the advancement of unsupervised segmentation techniques. In our work, we address the challenge of enhancing model performance despite the inherent limitations of low-quality optical flow. Our methodology employs a three-pronged approach: extracting boundaries directly from the optical flow, selectively discarding frames with inferior flow quality, and employing a fine-tuning process with variable frame rates. We thoroughly evaluate our strategy on the EndoVis2017 VOS dataset and Endovis2017 Challenge dataset, where our model demonstrates promising results, achieving a mean Intersection-o
    
[^3]: Lemur: 使用熵抽样和思维链合并进行日志解析

    Lemur: Log Parsing with Entropy Sampling and Chain-of-Thought Merging

    [https://arxiv.org/abs/2402.18205](https://arxiv.org/abs/2402.18205)

    Lemur提出了一种先进的日志解析框架，采用熵抽样和思维链合并，解决了日志解析中存在的人工规则依赖和语义信息忽略等问题。

    

    大型软件系统产生的日志对监视系统行为至关重要。先进的日志分析有助于检测、报警和诊断系统故障。日志解析是日志分析自动化的关键阶段，它涉及将原始日志消息转换为结构化模板。现有的日志解析器由于依赖于人工制定的规则而无法识别正确的模板。此外，这些方法侧重于统计特征，而忽略了日志消息中的语义信息。为了解决这些挑战，我们提出了一种先进的日志解析框架，采用熵抽样和思维链合并（Lemur）。具体而言，为了摆脱繁琐的手动规则，我们提出了一种受信息熵启发的新型抽样方法，能够有效地对典型日志进行聚类。此外，为了增强日志模板的合并，我们设计了一种思维链方法。

    arXiv:2402.18205v1 Announce Type: cross  Abstract: Logs produced by extensive software systems are integral to monitoring system behaviors. Advanced log analysis facilitates the detection, alerting, and diagnosis of system faults. Log parsing, which entails transforming raw log messages into structured templates, constitutes a critical phase in the automation of log analytics. Existing log parsers fail to identify the correct templates due to reliance on human-made rules. Besides, These methods focus on statistical features while ignoring semantic information in log messages. To address these challenges, we introduce a cutting-edge \textbf{L}og parsing framework with \textbf{E}ntropy sampling and Chain-of-Thought \textbf{M}erging (Lemur). Specifically, to discard the tedious manual rules. We propose a novel sampling method inspired by information entropy, which efficiently clusters typical logs. Furthermore, to enhance the merging of log templates, we design a chain-of-thought method f
    
[^4]: 基于稀疏网格的不连续性检测的图信息神经网络

    Graph-Informed Neural Networks for Sparse Grid-Based Discontinuity Detectors. (arXiv:2401.13652v1 [cs.LG])

    [http://arxiv.org/abs/2401.13652](http://arxiv.org/abs/2401.13652)

    本文提出了一种利用图信息神经网络和稀疏网格来检测不连续函数不连续界面的新方法，该方法在维度大于3的情况下表现出高效且准确的不连续性检测能力，在维度n = 2和n = 4的函数上进行的实验验证了其高效性和泛化能力，并具有可移植性和多功能性。

    

    本文提出了一种新颖的方法来检测不连续函数的不连续界面。该方法利用了基于图的神经网络（GINNs）和稀疏网格来解决维度大于3的情况下的不连续性检测。训练过的GINNs在稀疏网格上识别有问题的点，并利用构建在网格上的图结构实现高效准确的不连续性检测性能。我们还引入了一种递归算法用于一般的基于稀疏网格的检测器，具有收敛性和易于应用性。在维度n=2和n=4的函数上进行的数值实验证明了GINNs在检测不连续界面方面的高效性和鲁棒泛化能力。值得注意的是，经过训练的GINNs具有可移植性和多功能性，可以集成到各种算法中并共享给用户。

    In this paper, we present a novel approach for detecting the discontinuity interfaces of a discontinuous function. This approach leverages Graph-Informed Neural Networks (GINNs) and sparse grids to address discontinuity detection also in domains of dimension larger than 3. GINNs, trained to identify troubled points on sparse grids, exploit graph structures built on the grids to achieve efficient and accurate discontinuity detection performances. We also introduce a recursive algorithm for general sparse grid-based detectors, characterized by convergence properties and easy applicability. Numerical experiments on functions with dimensions n = 2 and n = 4 demonstrate the efficiency and robust generalization of GINNs in detecting discontinuity interfaces. Notably, the trained GINNs offer portability and versatility, allowing integration into various algorithms and sharing among users.
    
[^5]: 用分解的图神经网络预测卡车停车使用情况

    Truck Parking Usage Prediction with Decomposed Graph Neural Networks. (arXiv:2401.12920v1 [cs.AI])

    [http://arxiv.org/abs/2401.12920](http://arxiv.org/abs/2401.12920)

    提出了Regional Temporal Graph Neural Network (RegT-GCN)作为一个预测框架，用于评估整个州的卡车停车使用情况，以提供更准确的停车信息并缓解未经授权的停车问题。

    

    货运走廊上的卡车停车面临诸多挑战，如停车位不足和遵守工时规定。这些限制往往导致未经授权的停车行为，引发安全问题。为了提高货运作业的安全性，提供准确的停车使用预测被证明是一种经济高效的解决方案。尽管已有研究表明对于单个卡车停车场使用情况的预测准确度较高，但对多个卡车停车场的空间依赖关系进行使用预测的方法很少。我们提出了区域时空图神经网络（RegT-GCN）作为一个预测框架，用于评估整个州的停车使用情况，以提供更好的卡车停车信息和缓解未经授权的停车问题。该框架利用卡车停车场分布的拓扑结构和历史停车数据来预测整个州的占用率。

    Truck parking on freight corridors faces various challenges, such as insufficient parking spaces and compliance with Hour-of-Service (HOS) regulations. These constraints often result in unauthorized parking practices, causing safety concerns. To enhance the safety of freight operations, providing accurate parking usage prediction proves to be a cost-effective solution. Despite the existing research demonstrating satisfactory accuracy for predicting individual truck parking site usage, few approaches have been proposed for predicting usage with spatial dependencies of multiple truck parking sites. We present the Regional Temporal Graph Neural Network (RegT-GCN) as a predictive framework for assessing parking usage across the entire state to provide better truck parking information and mitigate unauthorized parking. The framework leverages the topological structures of truck parking site distributions and historical parking data to predict occupancy rates across a state. To achieve this,
    
[^6]: 适用于轻量级描述逻辑的半环溯源

    Semiring Provenance for Lightweight Description Logics. (arXiv:2310.16472v1 [cs.LO])

    [http://arxiv.org/abs/2310.16472](http://arxiv.org/abs/2310.16472)

    这篇论文研究了在描述逻辑中使用半环溯源的框架，并定义了一种适用于轻量级描述逻辑的溯源语义。论文证明了在半环施加限制的情况下，语义满足一些重要的特性，并对why溯源方法进行了研究。

    

    我们研究了半环溯源——一种最初在关系数据库环境中定义的成功框架，用于描述逻辑。在此上下文中，本体公理被用交换半环的元素进行注释，并且这些注释根据它们的推导方式传播到本体的结果中。我们定义了一种溯源语义，适用于包括几种轻量级描述逻辑的语言，并展示了它与为带有特定类型注释（如模糊度）的本体定义的其他语义之间的关系。我们证明了在一些对半环施加限制的情况下，语义满足一些期望的特性（如扩展了数据库中定义的半环溯源）。然后我们专注于著名的why溯源方法，它允许计算每个加法幂等和乘法幂等的交换半环的半环溯源，并研究了与这种溯源方法相关的问题的复杂性。

    We investigate semiring provenance--a successful framework originally defined in the relational database setting--for description logics. In this context, the ontology axioms are annotated with elements of a commutative semiring and these annotations are propagated to the ontology consequences in a way that reflects how they are derived. We define a provenance semantics for a language that encompasses several lightweight description logics and show its relationships with semantics that have been defined for ontologies annotated with a specific kind of annotation (such as fuzzy degrees). We show that under some restrictions on the semiring, the semantics satisfies desirable properties (such as extending the semiring provenance defined for databases). We then focus on the well-known why-provenance, which allows to compute the semiring provenance for every additively and multiplicatively idempotent commutative semiring, and for which we study the complexity of problems related to the prov
    
[^7]: 动态边界最大化和改进的Lipschitz正则化的认证鲁棒性

    Certified Robustness via Dynamic Margin Maximization and Improved Lipschitz Regularization. (arXiv:2310.00116v1 [cs.LG])

    [http://arxiv.org/abs/2310.00116](http://arxiv.org/abs/2310.00116)

    本文提出了一种基于动态边界最大化和改进的Lipschitz正则化的认证鲁棒性训练算法，通过增加输出空间中的边界和正则化模型的Lipschitz常数来提高深度分类器对抗性扰动的鲁棒性。

    

    为了提高深度分类器对抗性扰动的鲁棒性，已经提出了许多方法，例如设计具有更好鲁棒性性质的新架构（例如，Lipschitz-capped网络）或修改训练过程本身（例如，最小-最大优化，约束学习或正则化）。然而，这些方法对于增加输入（特征）空间中的边界可能并不有效。因此，越来越多的人开始对开发能够直接操纵输入空间中的决策边界的训练过程感兴趣。在本文中，我们在该类别的最新发展基础上，开发了一种鲁棒训练算法，其目标是在输出（logit）空间中增加边界，并沿着脆弱方向正则化模型的Lipschitz常数。我们证明这两个目标可以直接促进输入空间中更大的边界。为此，我们开发了一种可扩展的方法来计算...

    To improve the robustness of deep classifiers against adversarial perturbations, many approaches have been proposed, such as designing new architectures with better robustness properties (e.g., Lipschitz-capped networks), or modifying the training process itself (e.g., min-max optimization, constrained learning, or regularization). These approaches, however, might not be effective at increasing the margin in the input (feature) space. As a result, there has been an increasing interest in developing training procedures that can directly manipulate the decision boundary in the input space. In this paper, we build upon recent developments in this category by developing a robust training algorithm whose objective is to increase the margin in the output (logit) space while regularizing the Lipschitz constant of the model along vulnerable directions. We show that these two objectives can directly promote larger margins in the input space. To this end, we develop a scalable method for calcula
    
[^8]: 基于定性专家知识的量化代理模型开发框架：一个有组织犯罪的应用案例

    Framework for developing quantitative agent based models based on qualitative expert knowledge: an organised crime use-case. (arXiv:2308.00505v1 [cs.AI])

    [http://arxiv.org/abs/2308.00505](http://arxiv.org/abs/2308.00505)

    提出了一个基于定性专家知识的量化代理模型开发框架，该框架通过将定性数据翻译成定量规则，为模型构建者和领域专家提供了一个系统和透明的建模过程。以一个有组织犯罪的应用案例为例，演示了该框架的方法。

    

    为了对执法目的建模犯罪网络，需要将有限的数据转化为经过验证的基于代理的模型。当前刑事学建模中缺少一个为模型构建者和领域专家提供系统和透明框架的方法，该方法建立了计算犯罪建模的建模过程，包括将定性数据转化为定量规则。因此，我们提出了FREIDA（基于专家知识驱动的数据驱动代理模型框架）。在本文中，犯罪可卡因替代模型（CCRM）将作为示例案例，以演示FREIDA方法。对于CCRM，正在建模荷兰的一个有组织可卡因网络，试图通过移除首脑节点，使剩余代理重新组织，并将网络恢复到稳定状态。定性数据源，例如案件文件，文献和采访，被转化为经验法则。

    In order to model criminal networks for law enforcement purposes, a limited supply of data needs to be translated into validated agent-based models. What is missing in current criminological modelling is a systematic and transparent framework for modelers and domain experts that establishes a modelling procedure for computational criminal modelling that includes translating qualitative data into quantitative rules. For this, we propose FREIDA (Framework for Expert-Informed Data-driven Agent-based models). Throughout the paper, the criminal cocaine replacement model (CCRM) will be used as an example case to demonstrate the FREIDA methodology. For the CCRM, a criminal cocaine network in the Netherlands is being modelled where the kingpin node is being removed, the goal being for the remaining agents to reorganize after the disruption and return the network into a stable state. Qualitative data sources such as case files, literature and interviews are translated into empirical laws, and c
    
[^9]: 联邦学习与O-RAN的协同：面向多个分布式机器学习服务的弹性虚拟化架构

    Synergies Between Federated Learning and O-RAN: Towards an Elastic Virtualized Architecture for Multiple Distributed Machine Learning Services. (arXiv:2305.02109v1 [cs.NI])

    [http://arxiv.org/abs/2305.02109](http://arxiv.org/abs/2305.02109)

    本文研究了联邦学习在现代无线网络下的挑战，提出了一种方法称为动态多服务联邦学习（DMS-FL）来解决这个问题。同时，还提出了一种名为弹性虚拟化联邦学习（EV-FL）的分布式机器学习架构，来支持DMS-FL中的设计要求。

    

    联邦学习是最流行的分布式机器学习技术，但是在现代无线网络中实现联邦学习面临着许多挑战，主要包括网络条件的动态性、系统中多个联邦学习服务/任务的并存以及联邦学习服务与其他网络服务的并行执行等。针对这些挑战，本文提出了一种名为动态多服务联邦学习（DMS-FL）的联邦学习泛型架构，并通过提出一种新的分布式机器学习架构——弹性虚拟化联邦学习（EV-FL）来解决DMS-FL中的三个未探索的设计问题。

    Federated learning (FL) is the most popular distributed machine learning technique. However, implementation of FL over modern wireless networks faces key challenges caused by (i) dynamics of the network conditions, (ii) coexistence of multiple FL services/tasks in the system, and (iii) concurrent execution of FL services with other network services, which are not jointly considered in prior works. Motivated by these challenges, we introduce a generic FL paradigm over next-generation (NextG) networks, called dynamic multi-service FL (DMS-FL). We identify three unexplored design considerations in DMS-FL: (i) FL service operator accumulation, (ii) wireless resource fragmentation, and (iii) signal strength fluctuations. We take the first steps towards addressing these design considerations through proposing a novel distributed ML architecture called elastic virtualized FL (EV-FL). EV-FL unleashes the full potential of Open RAN (O-RAN) systems and introduces an elastic resource provisioning
    
[^10]: 使AI“口渴”减少的方法：揭示和解决AI模型的秘密水消耗

    Making AI Less "Thirsty": Uncovering and Addressing the Secret Water Footprint of AI Models. (arXiv:2304.03271v1 [cs.LG])

    [http://arxiv.org/abs/2304.03271](http://arxiv.org/abs/2304.03271)

    本论文揭示以及提出了解决人工智能模型巨大水足迹的方法，因为其淡水消耗已经引起国际社会的重视，并且AI模型应该承担社会责任，做出面对水危机的表率。

    

    人工智能（AI）模型的碳足迹不断增长，特别是像GPT-3和GPT-4这样的大型模型，已经受到公众的关注。然而，同等重要且巨大的AI模型水印尚未引起人们的注意。例如，在微软最先进的美国数据中心中训练GPT-3可以直接消耗70万升清洁淡水（相当于生产370辆宝马汽车或320辆特斯拉电动汽车），如果在微软的亚洲数据中心进行训练，这个水消耗量将增加三倍，但这样的信息一直被保密。这极其令人担忧，因为淡水短缺已成为在人口迅速增长、水资源减少和老化的水基础设施的背景下，我们所有人面临的最紧迫的挑战之一。为了应对全球水资源的挑战，人工智能模型可以，而且应该，承担社会责任，以身作则解决自己的问题。

    The growing carbon footprint of artificial intelligence (AI) models, especially large ones such as GPT-3 and GPT-4, has been undergoing public scrutiny. Unfortunately, however, the equally important and enormous water footprint of AI models has remained under the radar. For example, training GPT-3 in Microsoft's state-of-the-art U.S. data centers can directly consume 700,000 liters of clean freshwater (enough for producing 370 BMW cars or 320 Tesla electric vehicles) and the water consumption would have been tripled if training were done in Microsoft's Asian data centers, but such information has been kept as a secret. This is extremely concerning, as freshwater scarcity has become one of the most pressing challenges shared by all of us in the wake of the rapidly growing population, depleting water resources, and aging water infrastructures. To respond to the global water challenges, AI models can, and also should, take social responsibility and lead by example by addressing their own 
    

