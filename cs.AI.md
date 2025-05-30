# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [How many views does your deep neural network use for prediction?](https://rss.arxiv.org/abs/2402.01095) | 本文提出了最小有效视图（MSVs）的概念，该概念类似于多视图，但适用于实际图像，并且通过实证研究表明，MSV的数量与模型的预测准确性之间存在关系。 |
| [^2] | [Leveraging Intelligent Recommender system as a first step resilience measure -- A data-driven supply chain disruption response framework](https://arxiv.org/abs/2404.00306) | 提出了一种基于智能推荐系统技术的数据驱动供应链紧急响应框架，能够作为供应链中断的有效措施，并帮助参与者在危机发生后获得更好的反应表现。 |
| [^3] | [Leveraging Large Language Models for Relevance Judgments in Legal Case Retrieval](https://arxiv.org/abs/2403.18405) | 设计一种新颖的几轮工作流程，专门用于法律案例的相关判断，能够通过模仿人类注释者的过程并整合专家推理，提高相关性判断的准确性。 |
| [^4] | [Safety Implications of Explainable Artificial Intelligence in End-to-End Autonomous Driving](https://arxiv.org/abs/2403.12176) | 自动驾驶中可解释人工智能的安全影响对于确保车辆自动化安全至关重要，但当前研究中安全性和可解释性方面往往被分开砠。 |
| [^5] | [Accelerated Inference and Reduced Forgetting: The Dual Benefits of Early-Exit Networks in Continual Learning](https://arxiv.org/abs/2403.07404) | 早期退出网络在持续学习中展现出降低遗忘和在资源利用上表现优异的特点 |
| [^6] | [Building Flexible Machine Learning Models for Scientific Computing at Scale](https://arxiv.org/abs/2402.16014) | OmniArch通过多物理学时空数据处理、可扩展的自回归任务和物理信息增强学习技术，在科学计算领域构建灵活的基础模型，并在性能、适应性和逆问题求解方面取得突破，展现了AI对科学计算的潜力。 |
| [^7] | [Language Agents with Reinforcement Learning for Strategic Play in the Werewolf Game](https://arxiv.org/abs/2310.18940) | 使用强化学习为基础，提出了一种框架，用于在狼人杀游戏中开发具有灵活语言行为和强大决策能力的战略语言代理 |

# 详细

[^1]: 深度神经网络预测使用多少个视图？

    How many views does your deep neural network use for prediction?

    [https://rss.arxiv.org/abs/2402.01095](https://rss.arxiv.org/abs/2402.01095)

    本文提出了最小有效视图（MSVs）的概念，该概念类似于多视图，但适用于实际图像，并且通过实证研究表明，MSV的数量与模型的预测准确性之间存在关系。

    

    尽管进行了许多理论和实证分析，但深度神经网络（DNN）的泛化能力仍未完全理解。最近，Allen-Zhu和Li（2023）引入了多视图的概念来解释DNN的泛化能力，但他们的主要目标是集成或蒸馏模型，并未讨论用于特定输入预测的多视图估计方法。在本文中，我们提出了最小有效视图（MSVs），它类似于多视图，但可以高效地计算真实图像。MSVs是输入中的一组最小且不同的特征，每个特征保留了模型对该输入的预测。我们通过实证研究表明，不同模型（包括卷积和转换模型）的MSV数量与预测准确性之间存在明确的关系，这表明多视图的角度对于理解（非集成或非蒸馏）DNN的泛化能力也很重要。

    The generalization ability of Deep Neural Networks (DNNs) is still not fully understood, despite numerous theoretical and empirical analyses. Recently, Allen-Zhu & Li (2023) introduced the concept of multi-views to explain the generalization ability of DNNs, but their main target is ensemble or distilled models, and no method for estimating multi-views used in a prediction of a specific input is discussed. In this paper, we propose Minimal Sufficient Views (MSVs), which is similar to multi-views but can be efficiently computed for real images. MSVs is a set of minimal and distinct features in an input, each of which preserves a model's prediction for the input. We empirically show that there is a clear relationship between the number of MSVs and prediction accuracy across models, including convolutional and transformer models, suggesting that a multi-view like perspective is also important for understanding the generalization ability of (non-ensemble or non-distilled) DNNs.
    
[^2]: 利用智能推荐系统作为第一步弹性措施 —— 一种基于数据驱动的供应链紧急响应框架

    Leveraging Intelligent Recommender system as a first step resilience measure -- A data-driven supply chain disruption response framework

    [https://arxiv.org/abs/2404.00306](https://arxiv.org/abs/2404.00306)

    提出了一种基于智能推荐系统技术的数据驱动供应链紧急响应框架，能够作为供应链中断的有效措施，并帮助参与者在危机发生后获得更好的反应表现。

    

    数字技术在提高供应链弹性方面的潜在用途越来越受到关注，尤其是在工业4.0和全球大流行病背景下。尽管推荐系统 (RS) 作为一种能够提升供应链弹性的工具被忽视，但从应变的角度来看，RS 是一种有效的工具。为了解决这一问题，本研究提出了一种基于智能推荐系统技术的全新数据驱动供应链紧急响应框架，并通过一个实际案例验证了概念模型。结果表明，我们的框架可以作为一种有效的供应链紧急响应措施在第一阶段得到实施，并帮助供应链参与者在供应链中断之后获得更好的反应表现。

    arXiv:2404.00306v1 Announce Type: cross  Abstract: Interests in the value of digital technologies for its potential uses to increase supply chain resilience (SCRes) are increasing in light to the industry 4.0 and the global pandemic. Utilization of Recommender systems (RS) as a supply chain (SC) resilience measure is neglected although RS is a capable tool to enhance SC resilience from a reactive aspect. To address this problem, this research proposed a novel data-driven supply chain disruption response framework based on the intelligent recommender system techniques and validated the conceptual model through a practical use case. Results show that our framework can be implemented as an effective SC disruption mitigation measure in the very first response phrase and help SC participants get better reaction performance after the SC disruption.
    
[^3]: 利用大型语言模型进行法律案例检索中的相关性判断

    Leveraging Large Language Models for Relevance Judgments in Legal Case Retrieval

    [https://arxiv.org/abs/2403.18405](https://arxiv.org/abs/2403.18405)

    设计一种新颖的几轮工作流程，专门用于法律案例的相关判断，能够通过模仿人类注释者的过程并整合专家推理，提高相关性判断的准确性。

    

    收集法律案例检索的相关判决是一项具有挑战性且耗时的任务。准确判断两个法律案例之间的相关性需要阅读冗长的文本并具备高水平的领域专业知识以提取法律事实并作出司法判断。随着先进的大型语言模型的出现，一些最近的研究表明使用LLM（Large Language Models）进行相关性判断是有前途的。然而，将一般性大型语言模型应用于法律案例检索中可靠的相关性判断的方法尚未得到充分探讨。为了填补这一研究空白，我们设计了一种新颖的几轮工作流程，专门用于法律案例的相关判断。所提出的工作流程将注释过程分解为一系列阶段，模仿人类注释者所使用的过程，并使专家推理能够灵活地整合以增强相关性判断的准确性。

    arXiv:2403.18405v1 Announce Type: new  Abstract: Collecting relevant judgments for legal case retrieval is a challenging and time-consuming task. Accurately judging the relevance between two legal cases requires a considerable effort to read the lengthy text and a high level of domain expertise to extract Legal Facts and make juridical judgments. With the advent of advanced large language models, some recent studies have suggested that it is promising to use LLMs for relevance judgment. Nonetheless, the method of employing a general large language model for reliable relevance judgments in legal case retrieval is yet to be thoroughly explored. To fill this research gap, we devise a novel few-shot workflow tailored to the relevant judgment of legal cases. The proposed workflow breaks down the annotation process into a series of stages, imitating the process employed by human annotators and enabling a flexible integration of expert reasoning to enhance the accuracy of relevance judgments.
    
[^4]: 自动驾驶中可解释人工智能的安全影响

    Safety Implications of Explainable Artificial Intelligence in End-to-End Autonomous Driving

    [https://arxiv.org/abs/2403.12176](https://arxiv.org/abs/2403.12176)

    自动驾驶中可解释人工智能的安全影响对于确保车辆自动化安全至关重要，但当前研究中安全性和可解释性方面往往被分开砠。

    

    末端到末端学习管道正在逐渐改变高度自主车辆的持续发展，这主要归功于深度学习的进步、大规模训练数据集的可用性以及综合传感器设备的改进。然而，当代学习方法在实时决策中缺乏可解释性，妨碍了用户的信任，并减弱了这类车辆的广泛部署和商业化。此外，当这些汽车参与或导致交通事故时，问题会变得更加严重。这种缺点从社会和法律的角度引起了严重的安全担忧。因此，在末端到末端自动驾驶中解释性是促进车辆自动化安全的关键。然而，当今最先进技术中研究人员通常将自动驾驶的安全性和可解释性方面分开研究。在本文中，我们旨在弥合这一差距

    arXiv:2403.12176v1 Announce Type: cross  Abstract: The end-to-end learning pipeline is gradually creating a paradigm shift in the ongoing development of highly autonomous vehicles, largely due to advances in deep learning, the availability of large-scale training datasets, and improvements in integrated sensor devices. However, a lack of interpretability in real-time decisions with contemporary learning methods impedes user trust and attenuates the widespread deployment and commercialization of such vehicles. Moreover, the issue is exacerbated when these cars are involved in or cause traffic accidents. Such drawback raises serious safety concerns from societal and legal perspectives. Consequently, explainability in end-to-end autonomous driving is essential to enable the safety of vehicular automation. However, the safety and explainability aspects of autonomous driving have generally been investigated disjointly by researchers in today's state of the art. In this paper, we aim to brid
    
[^5]: 提高推理速度和减少遗忘：早期退出网络在持续学习中的双重好处

    Accelerated Inference and Reduced Forgetting: The Dual Benefits of Early-Exit Networks in Continual Learning

    [https://arxiv.org/abs/2403.07404](https://arxiv.org/abs/2403.07404)

    早期退出网络在持续学习中展现出降低遗忘和在资源利用上表现优异的特点

    

    arXiv:2403.07404v1 公告类型: 跨界 摘要: 受深度神经网络能源高效利用需求驱动，早期退出方法备受关注。这些策略通过在网络早期做出决定，实现快速预测，从而节省计算时间和资源。然而，迄今为止，早期退出网络仅针对静态数据分布进行了开发，限制了它们在具有持续非静态数据的实际场景中的应用。本研究旨在探讨早期退出网络的持续学习。我们改编现有的持续学习方法以适应早期退出架构，并研究它们在持续设置中的行为。我们注意到，早期网络层表现出减少遗忘，即使使用的资源显著更少，也能胜过标准网络。此外，我们分析任务最近性偏差对早期退出推理的影响，并提出任务...

    arXiv:2403.07404v1 Announce Type: cross  Abstract: Driven by the demand for energy-efficient employment of deep neural networks, early-exit methods have experienced a notable increase in research attention. These strategies allow for swift predictions by making decisions early in the network, thereby conserving computation time and resources. However, so far the early-exit networks have only been developed for stationary data distributions, which restricts their application in real-world scenarios with continuous non-stationary data. This study aims to explore the continual learning of the early-exit networks. We adapt existing continual learning methods to fit with early-exit architectures and investigate their behavior in the continual setting. We notice that early network layers exhibit reduced forgetting and can outperform standard networks even when using significantly fewer resources. Furthermore, we analyze the impact of task-recency bias on early-exit inference and propose Task
    
[^6]: 在科学计算规模上构建灵活的机器学习模型

    Building Flexible Machine Learning Models for Scientific Computing at Scale

    [https://arxiv.org/abs/2402.16014](https://arxiv.org/abs/2402.16014)

    OmniArch通过多物理学时空数据处理、可扩展的自回归任务和物理信息增强学习技术，在科学计算领域构建灵活的基础模型，并在性能、适应性和逆问题求解方面取得突破，展现了AI对科学计算的潜力。

    

    arXiv:2402.16014v1

    arXiv:2402.16014v1 Announce Type: cross  Abstract: Foundation models have revolutionized knowledge acquisition across domains, and our study introduces OmniArch, a paradigm-shifting approach designed for building foundation models in multi-physics scientific computing. OmniArch's pre-training involves a versatile pipeline that processes multi-physics spatio-temporal data, casting forward problem learning into scalable auto-regressive tasks, while our novel Physics-Informed Reinforcement Learning (PIRL) technique during fine-tuning ensures alignment with physical laws. Pre-trained on the comprehensive PDEBench dataset, OmniArch not only sets new performance benchmarks for 1D, 2D and 3D PDEs but also demonstrates exceptional adaptability to new physics via few-shot and zero-shot learning approaches. The model's representations further extend to inverse problem-solving, highlighting the transformative potential of AI-enabled Scientific Computing(AI4SC) foundation models for engineering ap
    
[^7]: 使用强化学习的语言代理在狼人杀游戏中进行战略对战

    Language Agents with Reinforcement Learning for Strategic Play in the Werewolf Game

    [https://arxiv.org/abs/2310.18940](https://arxiv.org/abs/2310.18940)

    使用强化学习为基础，提出了一种框架，用于在狼人杀游戏中开发具有灵活语言行为和强大决策能力的战略语言代理

    

    基于大型语言模型（LLMs）构建的代理在各领域展现出巨大潜力。然而，在复杂决策任务中，纯LLM代理往往表现出固有偏见，这些偏见来源于模型的训练数据，导致性能不佳。为了开发具有灵活语言行为和强大决策能力的战略语言代理，我们提出了一种新颖的框架，通过强化学习（RL）提升LLM代理的能力。我们选择狼人杀作为具有多样沟通和战略游戏玩法的挑战测试平台。为了减轻语言行为中的固有偏见，我们的代理使用LLM进行演绎推理并生成多样行为候选集。然后，经过训练以优化决策能力的RL策略从候选集中选择行为。

    arXiv:2310.18940v3 Announce Type: replace  Abstract: Agents built with large language models (LLMs) have shown great potential across a wide range of domains. However, in complex decision-making tasks, pure LLM-based agents tend to exhibit intrinsic bias in their choice of actions, which is inherited from the model's training data and results in suboptimal performance. To develop strategic language agents, i.e., agents that generate flexible language actions and possess strong decision-making abilities, we propose a novel framework that powers LLM-based agents with reinforcement learning (RL). We consider Werewolf, a popular social deduction game, as a challenging testbed that emphasizes versatile communication and strategic gameplay. To mitigate the intrinsic bias in language actions, our agents use an LLM to perform deductive reasoning and generate a diverse set of action candidates. Then an RL policy trained to optimize the decision-making ability chooses an action from the candidat
    

