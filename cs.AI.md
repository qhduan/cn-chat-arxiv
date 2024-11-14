# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Training Survival Models using Scoring Rules](https://arxiv.org/abs/2403.13150) | 提出了一种使用评分规则训练生存模型的通用方法，将其应用于各种模型类别中并与神经网络结合，实现了高效可扩展的优化例程，并展示了优于基于似然性方法的预测性能。 |
| [^2] | [Morphological Symmetries in Robotics](https://arxiv.org/abs/2402.15552) | 形态对称性是机器人系统中的固有性质，通过对运动结构和质量的对称分布，延伸至机器人状态空间和传感器测量，进而影响机器人的运动方程和最优控制策略，并在机器人学建模、控制和设计中具有重要意义。 |
| [^3] | [Controlling Large Electric Vehicle Charging Stations via User Behavior Modeling and Stochastic Programming](https://arxiv.org/abs/2402.13224) | 本文介绍了一个新的电动汽车充电站模型，通过用户行为建模和随机规划，解决了充电会话不确定性问题，并提出了两种方法来优化成本并提高用户满意度。 |
| [^4] | [Evaluating Program Repair with Semantic-Preserving Transformations: A Naturalness Assessment](https://arxiv.org/abs/2402.11892) | 本文研究了保留语义的转换的自然性及其对NPR评估的影响，发现了NPR系统在面对不自然的代码转换时会产生较高的误报率，且在使用自然转换进行评估时性能明显下降。 |
| [^5] | [AutoSAT: Automatically Optimize SAT Solvers via Large Language Models](https://arxiv.org/abs/2402.10705) | AutoSAT通过大型语言模型自动优化SAT求解器中的启发式，减少人为干预，提升求解器能力，实现了即插即用操作，保证了容错性，在广泛实验中表现出优越性能。 |
| [^6] | [HEAM : Hashed Embedding Acceleration using Processing-In-Memory](https://arxiv.org/abs/2402.04032) | HEAM是一种采用异构内存架构的方法，将3D堆叠DRAM与DIMM集成，用于加速处理大规模个性化推荐系统中的嵌入操作。 |
| [^7] | [Uncertainty of Thoughts: Uncertainty-Aware Planning Enhances Information Seeking in Large Language Models](https://arxiv.org/abs/2402.03271) | 通过引入不确定性感知规划（UoT）算法，我们实现了增强大型语言模型的主动寻求信息的能力，通过模拟未来场景、基于不确定性的奖励机制和奖励传播方案，优化问题提问方式。 |
| [^8] | [Are Large Language Models Table-based Fact-Checkers?](https://arxiv.org/abs/2402.02549) | 本研究初步探讨了大型语言模型在基于表格的事实检查方面的潜力。实验结果表明，通过提示工程，大型语言模型在零样本和少样本的情况下可以实现可接受的表现。 |
| [^9] | [Toxicity Detection is NOT all you Need: Measuring the Gaps to Supporting Volunteer Content Moderators](https://arxiv.org/abs/2311.07879) | 本研究揭示了人工智能模型在识别有毒、冒犯和令人讨厌的内容方面的进展，并探讨了这些改进是否真正满足了志愿内容管理员在工作中的需求。 |
| [^10] | [ShaRP: Explaining Rankings with Shapley Values.](http://arxiv.org/abs/2401.16744) | ShaRP是一个基于Shapley值的框架，用于解释排名结果中各个特征的贡献。即使使用线性评分函数，特征的权重也不一定对应其Shapley值的贡献，而是取决于特征分布和评分特征之间的局部相互作用。 |
| [^11] | [Reinforcement Learning (RL) Augmented Cold Start Frequency Reduction in Serverless Computing.](http://arxiv.org/abs/2308.07541) | 本文提出了一种基于强化学习的方法来降低无服务器计算中的冷启动频率。通过使用Q学习和考虑多种指标，我们可以在预期需求的基础上提前初始化函数，从而减少冷启动次数。 |
| [^12] | [On the Need for a Language Describing Distribution Shifts: Illustrations on Tabular Datasets.](http://arxiv.org/abs/2307.05284) | 该论文通过对表格数据集中的自然偏移进行研究，发现$Y|X$-偏移最为普遍。为了推动研究人员开发描述数据分布偏移的精细语言，作者构建了WhyShift实验平台，并讨论了$Y|X$-偏移对算法的影响。 |
| [^13] | [V-LoL: A Diagnostic Dataset for Visual Logical Learning.](http://arxiv.org/abs/2306.07743) | V-LoL是一个结合视觉和逻辑挑战的诊断数据集，其中包括了V-LoL-Trains，该数据集首次将复杂的视觉场景和灵活的逻辑推理任务结合起来，为研究广泛的视觉逻辑学习挑战提供了平台。 |
| [^14] | [China and the U.S. produce more impactful AI research when collaborating together.](http://arxiv.org/abs/2304.11123) | 中国和美国在人工智能领域合作能产生更大影响力，最近数据显示两国自2000年来一直处于领导地位，而大多数人才流失在两国之间。 |

# 详细

[^1]: 使用评分规则训练生存模型

    Training Survival Models using Scoring Rules

    [https://arxiv.org/abs/2403.13150](https://arxiv.org/abs/2403.13150)

    提出了一种使用评分规则训练生存模型的通用方法，将其应用于各种模型类别中并与神经网络结合，实现了高效可扩展的优化例程，并展示了优于基于似然性方法的预测性能。

    

    生存分析为各个领域中部分不完整的事件发生时间数据提供了关键见解。它也是概率机器学习的一个重要示例。我们的提案以一种通用的方式利用了预测的概率性质，通过在模型拟合过程中使用（合适的）评分规则而非基于似然性的优化。我们建立了不同的参数化和非参数化子框架，允许不同程度的灵活性。将其混入神经网络中，导致了一个计算有效且可扩展的优化例程，产生了最先进的预测性能。最后，我们展示了使用我们的框架，可以恢复各种参数化模型，并证明在与基于似然性方法的比较中，优化效果同样出色。

    arXiv:2403.13150v1 Announce Type: new  Abstract: Survival Analysis provides critical insights for partially incomplete time-to-event data in various domains. It is also an important example of probabilistic machine learning. The probabilistic nature of the predictions can be exploited by using (proper) scoring rules in the model fitting process instead of likelihood-based optimization. Our proposal does so in a generic manner and can be used for a variety of model classes. We establish different parametric and non-parametric sub-frameworks that allow different degrees of flexibility. Incorporated into neural networks, it leads to a computationally efficient and scalable optimization routine, yielding state-of-the-art predictive performance. Finally, we show that using our framework, we can recover various parametric models and demonstrate that optimization works equally well when compared to likelihood-based methods.
    
[^2]: 机器人学中的形态对称性

    Morphological Symmetries in Robotics

    [https://arxiv.org/abs/2402.15552](https://arxiv.org/abs/2402.15552)

    形态对称性是机器人系统中的固有性质，通过对运动结构和质量的对称分布，延伸至机器人状态空间和传感器测量，进而影响机器人的运动方程和最优控制策略，并在机器人学建模、控制和设计中具有重要意义。

    

    我们提出了一个全面的框架来研究和利用机器人系统中的形态对称性。这些是机器人形态的固有特性，经常在动物生物学和机器人学中观察到，源于运动结构的复制和质量的对称分布。我们说明了这些对称性如何延伸到机器人的状态空间以及本体感知和外部感知传感器测量，导致机器人的运动方程和最优控制策略的等不变性。因此，我们认识到形态对称性作为一个相关且以前未被探索的受物理启示的几何先验，对机器人建模、控制、估计和设计中使用的数据驱动和分析方法都具有重要影响。对于数据驱动方法，我们演示了形态对称性如何提高机器学习模型的样本效率和泛化能力

    arXiv:2402.15552v1 Announce Type: cross  Abstract: We present a comprehensive framework for studying and leveraging morphological symmetries in robotic systems. These are intrinsic properties of the robot's morphology, frequently observed in animal biology and robotics, which stem from the replication of kinematic structures and the symmetrical distribution of mass. We illustrate how these symmetries extend to the robot's state space and both proprioceptive and exteroceptive sensor measurements, resulting in the equivariance of the robot's equations of motion and optimal control policies. Thus, we recognize morphological symmetries as a relevant and previously unexplored physics-informed geometric prior, with significant implications for both data-driven and analytical methods used in modeling, control, estimation and design in robotics. For data-driven methods, we demonstrate that morphological symmetries can enhance the sample efficiency and generalization of machine learning models 
    
[^3]: 通过用户行为建模和随机规划控制大型电动汽车充电站

    Controlling Large Electric Vehicle Charging Stations via User Behavior Modeling and Stochastic Programming

    [https://arxiv.org/abs/2402.13224](https://arxiv.org/abs/2402.13224)

    本文介绍了一个新的电动汽车充电站模型，通过用户行为建模和随机规划，解决了充电会话不确定性问题，并提出了两种方法来优化成本并提高用户满意度。

    

    本文介绍了一个电动汽车充电站（EVCS）模型，该模型融合了真实世界的约束条件，如插槽功率限制、合同阈值超限惩罚以及电动汽车（EVs）的早期断开。我们提出了一个在不确定性下控制EVCS的问题形式，并实施了两种多阶段随机规划方法，利用用户提供的信息，即模型预测控制和二阶段随机规划。该模型解决了充电会话开始和结束时间以及能量需求的不确定性。基于驻留时间依赖随机过程的用户行为模型增强了成本降低的同时保持客户满意度。通过使用真实世界数据集进行的22天模拟展示了两种提出方法相对于两个基线的优势。两阶段方法证明了针对早期断开的鲁棒性，考虑了更多

    arXiv:2402.13224v1 Announce Type: cross  Abstract: This paper introduces an Electric Vehicle Charging Station (EVCS) model that incorporates real-world constraints, such as slot power limitations, contract threshold overruns penalties, or early disconnections of electric vehicles (EVs). We propose a formulation of the problem of EVCS control under uncertainty, and implement two Multi-Stage Stochastic Programming approaches that leverage user-provided information, namely, Model Predictive Control and Two-Stage Stochastic Programming. The model addresses uncertainties in charging session start and end times, as well as in energy demand. A user's behavior model based on a sojourn-time-dependent stochastic process enhances cost reduction while maintaining customer satisfaction. The benefits of the two proposed methods are showcased against two baselines over a 22-day simulation using a real-world dataset. The two-stage approach proves robust against early disconnections, considering a more
    
[^4]: 用保留语义的转换评估程序修复：自然性评估

    Evaluating Program Repair with Semantic-Preserving Transformations: A Naturalness Assessment

    [https://arxiv.org/abs/2402.11892](https://arxiv.org/abs/2402.11892)

    本文研究了保留语义的转换的自然性及其对NPR评估的影响，发现了NPR系统在面对不自然的代码转换时会产生较高的误报率，且在使用自然转换进行评估时性能明显下降。

    

    在本文中，我们研究了保留语义的转换的自然性及其对NPR评估的影响。为了达到这个目的，我们进行了一个两阶段的人类研究，包括(1)与资深软件开发人员的访谈，以建立评估代码转换自然性的第一个具体标准；(2)进行了一项涉及10名开发人员的调查，评估了应用于225个真实世界bug的1178个转换（即原始和转换程序成对的情况）的自然性。我们的研究结果显示，其中接近60%的转换被认为是自然的，20%的转换被认为是不自然的，并且在人类标注者之间有相当高的一致性。此外，不自然的代码转换引入了五个知名NPR系统的稳健性的25.2%误报率。此外，当使用自然转换进行评估时，NPR系统的性能显着下降，即性能下降高达22.9%和23.6%。

    arXiv:2402.11892v1 Announce Type: cross  Abstract: In this paper, we investigate the naturalness of semantic-preserving transformations and their impacts on the evaluation of NPR. To achieve this, we conduct a two-stage human study, including (1) interviews with senior software developers to establish the first concrete criteria for assessing the naturalness of code transformations and (2) a survey involving 10 developers to assess the naturalness of 1178 transformations, i.e., pairs of original and transformed programs, applied to 225 real-world bugs. Our findings reveal that nearly 60% and 20% of these transformations are considered natural and unnatural with substantially high agreement among human annotators. Furthermore, the unnatural code transformations introduce a 25.2% false alarm rate on robustness of five well-known NPR systems. Additionally, the performance of the NPR systems drops notably when evaluated using natural transformations, i.e., a drop of up to 22.9% and 23.6% i
    
[^5]: AutoSAT:通过大型语言模型自动优化SAT求解器

    AutoSAT: Automatically Optimize SAT Solvers via Large Language Models

    [https://arxiv.org/abs/2402.10705](https://arxiv.org/abs/2402.10705)

    AutoSAT通过大型语言模型自动优化SAT求解器中的启发式，减少人为干预，提升求解器能力，实现了即插即用操作，保证了容错性，在广泛实验中表现出优越性能。

    

    启发式在SAT求解器中至关重要，然而，并没有适用于所有问题实例的启发式规则。因此，通常需要为特定问题实例优化特定求解器。在这种情况下，我们提出了AutoSAT，这是一个新颖的框架，用于自动优化SAT求解器中的启发式。AutoSAT基于大型语言模型（LLMs），能够自动生成代码，进行评估，然后利用反馈进一步优化启发式，从而减少人为干预，增强求解器能力。AutoSAT基于即插即用的方式运行，消除了对广泛的初步设置和模型训练的需求，并促进了一种带有容错能力的思维链协作过程，确保启发式优化的稳健性。对使用冲突驱动子句学习（CDCL）求解器的广泛实验表明AutoSAT的整体性能优越，特别在解决某些特定的SAT问题时。

    arXiv:2402.10705v1 Announce Type: new  Abstract: Heuristics are crucial in SAT solvers, while no heuristic rules are suitable for all problem instances. Therefore, it typically requires to refine specific solvers for specific problem instances. In this context, we present AutoSAT, a novel framework for automatically optimizing heuristics in SAT solvers. AutoSAT is based on Large Large Models (LLMs) which is able to autonomously generate code, conduct evaluation, then utilize the feedback to further optimize heuristics, thereby reducing human intervention and enhancing solver capabilities. AutoSAT operates on a plug-and-play basis, eliminating the need for extensive preliminary setup and model training, and fosters a Chain of Thought collaborative process with fault-tolerance, ensuring robust heuristic optimization. Extensive experiments on a Conflict-Driven Clause Learning (CDCL) solver demonstrates the overall superior performance of AutoSAT, especially in solving some specific SAT pr
    
[^6]: HEAM: 使用处理-内存进行散列嵌入加速的方法

    HEAM : Hashed Embedding Acceleration using Processing-In-Memory

    [https://arxiv.org/abs/2402.04032](https://arxiv.org/abs/2402.04032)

    HEAM是一种采用异构内存架构的方法，将3D堆叠DRAM与DIMM集成，用于加速处理大规模个性化推荐系统中的嵌入操作。

    

    在当今的数据中心中，个性化推荐系统面临着诸多挑战，特别是在执行嵌入操作时需要大容量的内存和高带宽。之前的方法依赖于DIMM-based近内存处理技术或引入3D堆叠DRAM来解决内存限制和扩展内存带宽的问题。然而，这些解决方案在处理日益扩大的个性化推荐系统大小时存在不足之处。推荐模型已经增长到超过数十TB的大小，导致在传统单节点推断服务器上高效运行变得困难。尽管已经提出了各种算法方法来减小嵌入表容量，但通常会导致内存访问增加或内存资源利用低效的问题。本文引入了HEAM，一种异构内存架构，将3D堆叠DRAM与DIMM集成在一起，以加速组合嵌入的推荐系统。

    In today's data centers, personalized recommendation systems face challenges such as the need for large memory capacity and high bandwidth, especially when performing embedding operations. Previous approaches have relied on DIMM-based near-memory processing techniques or introduced 3D-stacked DRAM to address memory-bound issues and expand memory bandwidth. However, these solutions fall short when dealing with the expanding size of personalized recommendation systems. Recommendation models have grown to sizes exceeding tens of terabytes, making them challenging to run efficiently on traditional single-node inference servers. Although various algorithmic methods have been proposed to reduce embedding table capacity, they often result in increased memory access or inefficient utilization of memory resources. This paper introduces HEAM, a heterogeneous memory architecture that integrates 3D-stacked DRAM with DIMM to accelerate recommendation systems in which compositional embedding is util
    
[^7]: 想法的不确定性：不确定性感知规划增强大型语言模型的信息搜索能力

    Uncertainty of Thoughts: Uncertainty-Aware Planning Enhances Information Seeking in Large Language Models

    [https://arxiv.org/abs/2402.03271](https://arxiv.org/abs/2402.03271)

    通过引入不确定性感知规划（UoT）算法，我们实现了增强大型语言模型的主动寻求信息的能力，通过模拟未来场景、基于不确定性的奖励机制和奖励传播方案，优化问题提问方式。

    

    在面对不确定性时，寻求信息的能力至关重要。在许多实际应用中，比如医学诊断和故障排除，解决任务所需的信息不是初始给定的，而需要通过询问后续问题来主动寻求（例如，医生向患者询问症状的更多细节）。在这项工作中，我们引入了思想的不确定性（UoT），一种算法将大型语言模型的能力与主动提问信息的能力相结合。UoT结合了1）不确定性感知仿真方法，使模型能够模拟可能的未来场景，并估计其发生的可能性；2）基于不确定性的奖励机制，激励模型寻求信息；3）奖励传播方案，以最大化预期奖励的方式选择最佳的问题提问方式。在医学诊断、故障排除和'20的实验中。

    In the face of uncertainty, the ability to seek information is of fundamental importance. In many practical applications, such as medical diagnosis and troubleshooting, the information needed to solve the task is not initially given, and has to be actively sought by asking follow-up questions (for example, a doctor asking a patient for more details about their symptoms). In this work, we introduce Uncertainty of Thoughts (UoT), an algorithm to augment large language models with the ability to actively seek information by asking effective questions. UoT combines 1) an uncertainty-aware simulation approach which enables the model to simulate possible future scenarios and how likely they are to occur, 2) uncertainty-based rewards motivated by information gain which incentivizes the model to seek information, and 3) a reward propagation scheme to select the optimal question to ask in a way that maximizes the expected reward. In experiments on medical diagnosis, troubleshooting and the '20 
    
[^8]: 大型语言模型是否适合基于表格的事实检查？

    Are Large Language Models Table-based Fact-Checkers?

    [https://arxiv.org/abs/2402.02549](https://arxiv.org/abs/2402.02549)

    本研究初步探讨了大型语言模型在基于表格的事实检查方面的潜力。实验结果表明，通过提示工程，大型语言模型在零样本和少样本的情况下可以实现可接受的表现。

    

    基于表格的事实验证（TFV）旨在提取语句和结构化表格之间的蕴涵关系。现有基于小规模模型的TFV方法在标注数据不足和零样本能力薄弱方面存在问题。近年来，大型语言模型（LLMs）在研究领域引起了广泛关注。它们在几个自然语言处理任务上展示了强大的零样本和上下文学习能力，但它们在TFV领域的潜力还不清楚。在本文中，我们进行了关于LLMs是否适合作为基于表格的事实检查器的初步研究。具体来说，我们设计了多样化的提示语来探索上下文学习如何帮助LLMs在TFV方面，即零样本和少样本TFV能力。此外，我们精心设计和构建了TFV指导以研究LLMs的指导调整带来的性能改进。实验结果表明，通过提示工程，LLMs在零样本和少样本TFV方面可以达到可接受的结果，而指导调整则进一步提升了性能。

    Table-based Fact Verification (TFV) aims to extract the entailment relation between statements and structured tables. Existing TFV methods based on small-scaled models suffer from insufficient labeled data and weak zero-shot ability. Recently, the appearance of Large Language Models (LLMs) has gained lots of attraction in research fields. They have shown powerful zero-shot and in-context learning abilities on several NLP tasks, but their potential on TFV is still unknown. In this work, we implement a preliminary study about whether LLMs are table-based fact-checkers. In detail, we design diverse prompts to explore how the in-context learning can help LLMs in TFV, i.e., zero-shot and few-shot TFV capability. Besides, we carefully design and construct TFV instructions to study the performance gain brought by the instruction tuning of LLMs. Experimental results demonstrate that LLMs can achieve acceptable results on zero-shot and few-shot TFV with prompt engineering, while instruction-tun
    
[^9]: 毒性检测并不是你所需要的全部：弥合支持志愿内容管理员的差距

    Toxicity Detection is NOT all you Need: Measuring the Gaps to Supporting Volunteer Content Moderators

    [https://arxiv.org/abs/2311.07879](https://arxiv.org/abs/2311.07879)

    本研究揭示了人工智能模型在识别有毒、冒犯和令人讨厌的内容方面的进展，并探讨了这些改进是否真正满足了志愿内容管理员在工作中的需求。

    

    人工智能模型在识别有毒、冒犯和令人讨厌的内容方面取得了长足的进展，旨在减轻管理员的工作负担。然而，目前尚不清楚这些任务的改进是否真正满足了管理员在工作中的需求。本文揭示了过去研究努力致力于为内容管理的各个方面提供自动化支持与志愿内容管理员的需求之间存在的差距，尤其是在识别违反各种管理规则方面。为此，我们在Hugging Face上对模型进行了调查，以揭示涵盖三个示范论坛的各种管理规则和指南的模型的可用性。我们进一步对最先进的LLM进行了测试，评估这些模型在标记某个特定论坛的平台规则违规方面的表现。最后，我们进行了用户调查研究。

    arXiv:2311.07879v2 Announce Type: replace-cross  Abstract: Extensive efforts in automated approaches for content moderation have been focused on developing models to identify toxic, offensive, and hateful content with the aim of lightening the load for moderators. Yet, it remains uncertain whether improvements on those tasks have truly addressed moderators' needs in accomplishing their work. In this paper, we surface gaps between past research efforts that have aimed to provide automation for aspects of content moderation and the needs of volunteer content moderators, regarding identifying violations of various moderation rules. To do so, we conduct a model review on Hugging Face to reveal the availability of models to cover various moderation rules and guidelines from three exemplar forums. We further put state-of-the-art LLMs to the test, evaluating how well these models perform in flagging violations of platform rules from one particular forum. Finally, we conduct a user survey stud
    
[^10]: ShaRP：用Shapley值解释排名

    ShaRP: Explaining Rankings with Shapley Values. (arXiv:2401.16744v1 [cs.AI])

    [http://arxiv.org/abs/2401.16744](http://arxiv.org/abs/2401.16744)

    ShaRP是一个基于Shapley值的框架，用于解释排名结果中各个特征的贡献。即使使用线性评分函数，特征的权重也不一定对应其Shapley值的贡献，而是取决于特征分布和评分特征之间的局部相互作用。

    

    在招聘、大学招生和贷款等重要领域的算法决策常常是基于排名的。由于这些决策对个人、组织和人群的影响，有必要了解它们：了解决策是否遵守法律，帮助个人提高他们的排名，并设计更好的排名程序。本文提出了ShaRP（Shapley for Rankings and Preferences），这是一个基于Shapley值的框架，用于解释特征对排名结果不同方面的贡献。使用ShaRP，我们展示了即使算法排名器使用的评分函数是已知的且是线性的，每个特征的权重也不一定对应其Shapley值的贡献。贡献取决于特征的分布以及评分特征之间微妙的局部相互作用。ShaRP基于量化输入影响框架，并可以计算贡献。

    Algorithmic decisions in critical domains such as hiring, college admissions, and lending are often based on rankings. Because of the impact these decisions have on individuals, organizations, and population groups, there is a need to understand them: to know whether the decisions are abiding by the law, to help individuals improve their rankings, and to design better ranking procedures.  In this paper, we present ShaRP (Shapley for Rankings and Preferences), a framework that explains the contributions of features to different aspects of a ranked outcome, and is based on Shapley values. Using ShaRP, we show that even when the scoring function used by an algorithmic ranker is known and linear, the weight of each feature does not correspond to its Shapley value contribution. The contributions instead depend on the feature distributions, and on the subtle local interactions between the scoring features. ShaRP builds on the Quantitative Input Influence framework, and can compute the contri
    
[^11]: 基于强化学习的无服务器计算中冷启动频率降低方法

    Reinforcement Learning (RL) Augmented Cold Start Frequency Reduction in Serverless Computing. (arXiv:2308.07541v1 [cs.DC])

    [http://arxiv.org/abs/2308.07541](http://arxiv.org/abs/2308.07541)

    本文提出了一种基于强化学习的方法来降低无服务器计算中的冷启动频率。通过使用Q学习和考虑多种指标，我们可以在预期需求的基础上提前初始化函数，从而减少冷启动次数。

    

    函数即服务是一种云计算范例，为应用程序提供了事件驱动执行模型。它通过从开发者那里消除资源管理责任，提供透明和按需可扩展性来实现无服务器特性。典型的无服务器应用程序对响应时间和可扩展性有严格要求，因此依赖于部署的服务为客户提供快速和容错的反馈。然而，函数即服务范例在需要按需初始化函数时存在非常可观的延迟，即冷启动问题。本研究旨在通过使用强化学习来减少平台上的冷启动频率。我们的方法使用Q学习，并考虑函数的CPU利用率、已有函数实例和响应失败率等指标，根据预期需求提前主动初始化函数。我们提出的解决方案在Kubeless上实现并进行评估。

    Function-as-a-Service is a cloud computing paradigm offering an event-driven execution model to applications. It features serverless attributes by eliminating resource management responsibilities from developers and offers transparent and on-demand scalability of applications. Typical serverless applications have stringent response time and scalability requirements and therefore rely on deployed services to provide quick and fault-tolerant feedback to clients. However, the FaaS paradigm suffers from cold starts as there is a non-negligible delay associated with on-demand function initialization. This work focuses on reducing the frequency of cold starts on the platform by using Reinforcement Learning. Our approach uses Q-learning and considers metrics such as function CPU utilization, existing function instances, and response failure rate to proactively initialize functions in advance based on the expected demand. The proposed solution was implemented on Kubeless and was evaluated usin
    
[^12]: 关于需要描述分布偏移的语言：基于表格数据集的案例分析

    On the Need for a Language Describing Distribution Shifts: Illustrations on Tabular Datasets. (arXiv:2307.05284v1 [cs.LG])

    [http://arxiv.org/abs/2307.05284](http://arxiv.org/abs/2307.05284)

    该论文通过对表格数据集中的自然偏移进行研究，发现$Y|X$-偏移最为普遍。为了推动研究人员开发描述数据分布偏移的精细语言，作者构建了WhyShift实验平台，并讨论了$Y|X$-偏移对算法的影响。

    

    不同的分布偏移需要不同的算法和操作干预。方法研究必须以其所涉及的具体偏移为基础。尽管新兴的基准数据为实证研究提供了有希望的基础，但它们隐含地关注协变量偏移，并且实证发现的有效性取决于偏移类型，例如，当$Y|X$分布发生变化时，之前关于算法性能的观察可能无效。我们对5个表格数据集中的自然偏移进行了深入研究，通过对86,000个模型配置进行实验，发现$Y|X$-偏移最为普遍。为了鼓励研究人员开发一种精细的描述数据分布偏移的语言，我们构建了WhyShift，一个由策划的真实世界偏移测试平台，在其中我们对我们基准性能的偏移类型进行了表征。由于$Y|X$-偏移在表格设置中很常见，我们确定了受到最大$Y|X$-偏移影响的协变量区域，并讨论了对算法的影响。

    Different distribution shifts require different algorithmic and operational interventions. Methodological research must be grounded by the specific shifts they address. Although nascent benchmarks provide a promising empirical foundation, they implicitly focus on covariate shifts, and the validity of empirical findings depends on the type of shift, e.g., previous observations on algorithmic performance can fail to be valid when the $Y|X$ distribution changes. We conduct a thorough investigation of natural shifts in 5 tabular datasets over 86,000 model configurations, and find that $Y|X$-shifts are most prevalent. To encourage researchers to develop a refined language for distribution shifts, we build WhyShift, an empirical testbed of curated real-world shifts where we characterize the type of shift we benchmark performance over. Since $Y|X$-shifts are prevalent in tabular settings, we identify covariate regions that suffer the biggest $Y|X$-shifts and discuss implications for algorithm
    
[^13]: V-LoL: 一种用于视觉逻辑学习的诊断数据集

    V-LoL: A Diagnostic Dataset for Visual Logical Learning. (arXiv:2306.07743v1 [cs.AI])

    [http://arxiv.org/abs/2306.07743](http://arxiv.org/abs/2306.07743)

    V-LoL是一个结合视觉和逻辑挑战的诊断数据集，其中包括了V-LoL-Trains，该数据集首次将复杂的视觉场景和灵活的逻辑推理任务结合起来，为研究广泛的视觉逻辑学习挑战提供了平台。

    

    尽管近期在视觉AI领域有了许多成功的进展，但仍存在不同的缺点；包括缺少精确的逻辑推理、抽象的概括能力以及理解复杂和嘈杂的场景等。不幸的是，现有的基准测试数据集并不能捕捉到这些方面中的多数。深度学习数据集关注视觉复杂数据但只有简单的视觉推理任务，归纳逻辑数据集包括复杂的逻辑学习任务，但是缺乏视觉的组成部分。为了解决这个问题，我们提出了视觉逻辑学习数据集V-LoL，它无缝地结合了视觉和逻辑的挑战。值得注意的是，我们首次推出了V-LoL的第一个实例，名为V-LoL-Trains，它是符号AI中一个经典基准测试的视觉呈现，即Michalski火车问题。通过在一个通用框架内结合复杂的视觉场景和灵活的逻辑推理任务，V-LoL-Trains为研究广泛的视觉逻辑学习挑战提供了平台。

    Despite the successes of recent developments in visual AI, different shortcomings still exist; from missing exact logical reasoning, to abstract generalization abilities, to understanding complex and noisy scenes. Unfortunately, existing benchmarks, were not designed to capture more than a few of these aspects. Whereas deep learning datasets focus on visually complex data but simple visual reasoning tasks, inductive logic datasets involve complex logical learning tasks, however, lack the visual component. To address this, we propose the visual logical learning dataset, V-LoL, that seamlessly combines visual and logical challenges. Notably, we introduce the first instantiation of V-LoL, V-LoL-Trains, -- a visual rendition of a classic benchmark in symbolic AI, the Michalski train problem. By incorporating intricate visual scenes and flexible logical reasoning tasks within a versatile framework, V-LoL-Trains provides a platform for investigating a wide range of visual logical learning ch
    
[^14]: 中美合作时，中国和美国在人工智能领域能够产生更大的影响力

    China and the U.S. produce more impactful AI research when collaborating together. (arXiv:2304.11123v1 [cs.CY])

    [http://arxiv.org/abs/2304.11123](http://arxiv.org/abs/2304.11123)

    中国和美国在人工智能领域合作能产生更大影响力，最近数据显示两国自2000年来一直处于领导地位，而大多数人才流失在两国之间。

    

    人工智能已经成为颠覆性技术，有望为掌握其力量的国家带来显著的经济和战略优势。最近，中国推动人工智能技术的采用，正在挑战美国在这一领域的全球领导地位。考虑到人工智能的巨大潜力，以及两国之间激烈的地缘政治紧张局势，已经制定了一些政策，以防止人工智能科学家移民到对方国家或与之合作。然而，这种人才流失和跨境合作的程度还没有被完全了解。在此，我们分析了超过350,000名人工智能科学家和5,000,000篇人工智能文献的数据集。我们发现自2000年以来，中国和美国在影响力、创新性、生产力和劳动力方面一直处于领先地位。大多数移民到中国的人工智能科学家来自美国，而移民到美国的人工智能科学家来自中国，凸显出明显的人才流失现象。

    Artificial Intelligence (AI) has become a disruptive technology, promising to grant a significant economic and strategic advantage to the nations that harness its power. China, with its recent push towards AI adoption, is challenging the U.S.'s position as the global leader in this field. Given AI's massive potential, as well as the fierce geopolitical tensions between the two nations, a number of policies have been put in place that discourage AI scientists from migrating to, or collaborating with, the other country. However, the extents of such brain drain and cross-border collaboration are not fully understood. Here, we analyze a dataset of over 350,000 AI scientists and 5,000,000 AI papers. We find that, since the year 2000, China and the U.S. have been leading the field in terms of impact, novelty, productivity, and workforce. Most AI scientists who migrate to China come from the U.S., and most who migrate to the U.S. come from China, highlighting a notable brain drain in both dir
    

