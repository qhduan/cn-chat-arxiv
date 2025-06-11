# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [New Perspectives in Online Contract Design: Heterogeneous, Homogeneous, Non-myopic Agents and Team Production](https://arxiv.org/abs/2403.07143) | 本研究从在线学习的视角研究了重复的委托-代理问题，针对不同情形提出了设计学习算法的不同方法和技术，包括异质代理、同质代理和非单纯视角代理。 |
| [^2] | [Spectral invariance and maximality properties of the frequency spectrum of quantum neural networks](https://arxiv.org/abs/2402.14515) | 量子神经网络研究了频谱的极大性质，证明了在一类模型中存在极大结果，以及在一些条件下存在保持频谱的光谱不变性，解释了文献中观察到的结果对称性。 |
| [^3] | [A Data Centric Approach for Unsupervised Domain Generalization via Retrieval from Web Scale Multimodal Data](https://arxiv.org/abs/2402.04416) | 该论文基于大规模多模态数据检索，提出了一个无监督领域泛化的数据中心方法。在多模态无监督领域泛化问题中，通过构建一个小型的源数据子集，而不是依赖丰富的源数据，来解决目标标签空间数据获取困难的问题。 |
| [^4] | [Data-Driven Discovery of PDEs via the Adjoint Method](https://arxiv.org/abs/2401.17177) | 本文提出了一种通过伴随方法从数据中发现潜在偏微分方程的方法，展示了在给定光滑数据集的情况下，该方法可以恢复真实的PDE，但在存在噪声的情况下，准确性与PDE-FIND方法相当。 |
| [^5] | [Quantum Architecture Search with Unsupervised Representation Learning](https://arxiv.org/abs/2401.11576) | 通过利用无监督表示学习，量子架构搜索（QAS）的性能可以得以提升，而不需要耗费大量时间进行标记。 |
| [^6] | [High-Dimensional Independence Testing via Maximum and Average Distance Correlations](https://arxiv.org/abs/2001.01095) | 本文介绍并研究了利用最大和平均距离相关性进行高维度独立性检测的方法，并提出了一种快速卡方检验的程序。该方法适用于欧氏距离和高斯核，具有较好的实证表现和广泛的应用场景。 |
| [^7] | [Innate-Values-driven Reinforcement Learning for Cooperative Multi-Agent Systems.](http://arxiv.org/abs/2401.05572) | 本文提出了一个先天价值驱动增强学习（IVRL）模型，用于描述多智能体在合作中的复杂行为。该模型通过建立智能体对群体效用和系统成本的认知，满足其合作伙伴的需求，支持其社区并融入人类社会。 |
| [^8] | [Provably Robust Cost-Sensitive Learning via Randomized Smoothing.](http://arxiv.org/abs/2310.08732) | 本研究通过随机平滑认证框架，为成本敏感的稳健分类器提供了严格的稳健性保证，并通过优化方案针对不同数据子组设计了细粒度认证半径，取得了优越的性能。 |
| [^9] | [Stabilizing Contrastive RL: Techniques for Offline Goal Reaching.](http://arxiv.org/abs/2306.03346) | 本文提出了一种稳定的对比强化学习方法，通过浅而宽的结构，结合谨慎的权重初始化和数据增强等实验方法，在具有挑战性的仿真基准测试中显著提高了性能，并演示了对比方法可以解决现实世界的机器人任务。 |
| [^10] | [The Face of Populism: Examining Differences in Facial Emotional Expressions of Political Leaders Using Machine Learning.](http://arxiv.org/abs/2304.09914) | 本文使用机器学习的算法分析了来自15个不同国家的220个政治领袖的YouTube视频，总结了政治领袖面部情感表达的差异。 |
| [^11] | [General Loss Functions Lead to (Approximate) Interpolation in High Dimensions.](http://arxiv.org/abs/2303.07475) | 本研究提供了一般凸损失函数和超参数化阶段下梯度下降的隐含偏差的近似表征。具体而言是近似最小范数插值。 |
| [^12] | [Refiner: Data Refining against Gradient Leakage Attacks in Federated Learning.](http://arxiv.org/abs/2212.02042) | Refiner提出了一种创新的防御范式，通过构建与原始数据具有低语义相似性的健壮数据，有效地混淆梯度泄漏攻击者，从而提高联邦学习系统的隐私保护能力。 |
| [^13] | [Precise High-Dimensional Asymptotics for Quantifying Heterogeneous Transfers.](http://arxiv.org/abs/2010.11750) | 本文利用随机矩阵理论在线性回归设置中，对于具有两个任务的高维情况下的常用估计量的超额风险进行了精确渐近分析。 |

# 详细

[^1]: 在线合同设计的新视角：异质、同质、非单纯视角代理和团队生产

    New Perspectives in Online Contract Design: Heterogeneous, Homogeneous, Non-myopic Agents and Team Production

    [https://arxiv.org/abs/2403.07143](https://arxiv.org/abs/2403.07143)

    本研究从在线学习的视角研究了重复的委托-代理问题，针对不同情形提出了设计学习算法的不同方法和技术，包括异质代理、同质代理和非单纯视角代理。

    

    这项工作从在线学习的视角研究了重复的委托-代理问题。 委托方的目标是通过重复互动学习最大化其效用的最佳合同，而没有关于代理方类型（即代理方的成本和生产函数）的先验知识。 我研究了三种不同的情境，委托方在每一轮与$\textit{单个}$代理方签订合同时：1. 代理方是异质的；2. 代理方是同质的；3. 委托方与相同的代理方互动且该代理方是非单纯的。 我提出不同的方法和技术来设计每种情况下的学习算法。 对于异质代理类型，我确定了一个条件，允许将问题直接简化为Lipschitz老虎机问题。 对于相同代理方，我提出了一个基于逆博弈论的多项式样本复杂度方案来学习最佳合同。 对于战略性非单纯代理，我设计了一个低战略性

    arXiv:2403.07143v1 Announce Type: cross  Abstract: This work studies the repeated principal-agent problem from an online learning perspective. The principal's goal is to learn the optimal contract that maximizes her utility through repeated interactions, without prior knowledge of the agent's type (i.e., the agent's cost and production functions).   I study three different settings when the principal contracts with a $\textit{single}$ agent each round: 1. The agents are heterogeneous; 2. the agents are homogenous; 3. the principal interacts with the same agent and the agent is non-myopic. I present different approaches and techniques for designing learning algorithms in each setting. For heterogeneous agent types, I identify a condition that allows the problem to be reduced to Lipschitz bandits directly. For identical agents, I give a polynomial sample complexity scheme to learn the optimal contract based on inverse game theory. For strategic non-myopic agents, I design a low strategic
    
[^2]: 量子神经网络频谱的光谱不变性和极大性质

    Spectral invariance and maximality properties of the frequency spectrum of quantum neural networks

    [https://arxiv.org/abs/2402.14515](https://arxiv.org/abs/2402.14515)

    量子神经网络研究了频谱的极大性质，证明了在一类模型中存在极大结果，以及在一些条件下存在保持频谱的光谱不变性，解释了文献中观察到的结果对称性。

    

    量子神经网络（QNNs）是量子机器学习领域的热门方法，由于其与变分量子电路的密切联系，使其成为在噪声中间尺度量子（NISQ）设备上进行实际应用的有前途的候选方法。QNN可以表示为有限傅里叶级数，其中频率集被称为频谱。我们分析了这个频谱并证明，对于一大类模型，存在各种极大性结果。此外，我们证明在一些温和条件下，存在一个保持频谱的具有相同面积$A = RL$的模型类之间的双射，其中$R$表示量子比特数量，$L$表示层数，我们因此称之为面积保持变换下的光谱不变性。通过这个，我们解释了文献中经常观察到的在结果中$R$和$L$的对称性，并展示了最大频谱的依赖性

    arXiv:2402.14515v1 Announce Type: cross  Abstract: Quantum Neural Networks (QNNs) are a popular approach in Quantum Machine Learning due to their close connection to Variational Quantum Circuits, making them a promising candidate for practical applications on Noisy Intermediate-Scale Quantum (NISQ) devices. A QNN can be expressed as a finite Fourier series, where the set of frequencies is called the frequency spectrum. We analyse this frequency spectrum and prove, for a large class of models, various maximality results. Furthermore, we prove that under some mild conditions there exists a bijection between classes of models with the same area $A = RL$ that preserves the frequency spectrum, where $R$ denotes the number of qubits and $L$ the number of layers, which we consequently call spectral invariance under area-preserving transformations. With this we explain the symmetry in $R$ and $L$ in the results often observed in the literature and show that the maximal frequency spectrum depen
    
[^3]: 基于大规模多模态数据检索的无监督领域泛化的数据中心方法

    A Data Centric Approach for Unsupervised Domain Generalization via Retrieval from Web Scale Multimodal Data

    [https://arxiv.org/abs/2402.04416](https://arxiv.org/abs/2402.04416)

    该论文基于大规模多模态数据检索，提出了一个无监督领域泛化的数据中心方法。在多模态无监督领域泛化问题中，通过构建一个小型的源数据子集，而不是依赖丰富的源数据，来解决目标标签空间数据获取困难的问题。

    

    领域泛化(DG)是一个重要的问题，它通过利用一个或多个源领域在共享标签空间的假设下学习一个能够推广到未见测试领域的模型。然而，大多数DG方法假设可以访问丰富的目标标签空间中的源数据，这个要求在许多现实应用中太过严格，因为获取与目标任务相同的标签空间费用高昂。为了解决这个问题，我们处理了无监督领域泛化(UDG)问题的多模态版本，该问题使用一个大型的任务无关的未标记的源数据集，例如LAION-2B在微调期间。我们的框架不显式地假设源数据集与目标任务之间存在任何关系。相反，它只依赖于源数据集可以在联合视觉-语言空间中高效搜索的前提。针对这种多模态UDG设置，我们提出了一种新的方法来构建一个小型（小于100K）的源数据子集。

    Domain generalization (DG) is an important problem that learns a model that can generalize to unseen test domains leveraging one or more source domains, under the assumption of shared label spaces. However, most DG methods assume access to abundant source data in the target label space, a requirement that proves overly stringent for numerous real-world applications, where acquiring the same label space as the target task is prohibitively expensive. For this setting, we tackle the multimodal version of the unsupervised domain generalization (UDG) problem, which uses a large task-agnostic unlabeled source dataset, such as LAION-2B during finetuning. Our framework does not explicitly assume any relationship between the source dataset and target task. Instead, it relies only on the premise that the source dataset can be efficiently searched in a joint vision-language space. For this multimodal UDG setting, we propose a novel method to build a small ($<$100K) subset of the source data in th
    
[^4]: 数据驱动的通过伴随方法发现偏微分方程

    Data-Driven Discovery of PDEs via the Adjoint Method

    [https://arxiv.org/abs/2401.17177](https://arxiv.org/abs/2401.17177)

    本文提出了一种通过伴随方法从数据中发现潜在偏微分方程的方法，展示了在给定光滑数据集的情况下，该方法可以恢复真实的PDE，但在存在噪声的情况下，准确性与PDE-FIND方法相当。

    

    在这项工作中，我们提出了一种通过伴随方法来发现给定数据的潜在偏微分方程（PDEs）的方法。我们的思路是以一般形式考虑参数化的PDE，并制定最小化PDE解与数据误差的优化问题。利用变分计算，我们得到了拉格朗日乘子（伴随方程）的演化方程，使我们能够直接计算出与PDE参数相关的目标函数的梯度。特别是对于一族参数化和非线性PDEs，我们展示了如何推导出相应的伴随方程。我们在这里展示了，在给定光滑数据集的情况下，所提出的伴随方法可以以机器精度恢复真实的PDE。然而，在存在噪声的情况下，伴随方法的准确性与著名的PDE-FIND（Rudy et al., 2017）方法相当。

    In this work, we present an adjoint-based method for discovering the underlying governing partial differential equations (PDEs) given data. The idea is to consider a parameterized PDE in a general form, and formulate the optimization problem that minimizes the error of PDE solution from data. Using variational calculus, we obtain an evolution equation for the Lagrange multipliers (adjoint equations) allowing us to compute the gradient of the objective function with respect to the parameters of PDEs given data in a straightforward manner. In particular, for a family of parameterized and nonlinear PDEs, we show how the corresponding adjoint equations can be derived. Here, we show that given smooth data set, the proposed adjoint method can recover the true PDE up to machine accuracy. However, in the presence of noise, the accuracy of the adjoint method becomes comparable to the famous PDE Functional Identification of Nonlinear Dynamics method known as PDE-FIND (Rudy et al., 2017). Even th
    
[^5]: 利用无监督表示学习进行量子架构搜索

    Quantum Architecture Search with Unsupervised Representation Learning

    [https://arxiv.org/abs/2401.11576](https://arxiv.org/abs/2401.11576)

    通过利用无监督表示学习，量子架构搜索（QAS）的性能可以得以提升，而不需要耗费大量时间进行标记。

    

    使用无监督表示学习进行量子架构搜索（QAS）代表了一种前沿方法，有望在嘈杂的中间规模量子（NISQ）设备上实现潜在的量子优势。大多数QAS算法将它们的搜索空间和搜索算法结合在一起，因此通常需要在搜索过程中评估大量的量子电路。基于预测的QAS算法可以通过直接根据电路结构估计电路的性能来缓解这个问题。然而，高性能的预测器通常需要耗费大量时间进行标记，以获得大量带标签的量子电路。最近，一个经典的神经架构搜索算法Arch2vec启发我们，表明架构搜索可以从将无监督表示学习与搜索过程分离中获益。无监督表示学习是否能帮助QAS

    arXiv:2401.11576v2 Announce Type: replace-cross  Abstract: Utilizing unsupervised representation learning for quantum architecture search (QAS) represents a cutting-edge approach poised to realize potential quantum advantage on Noisy Intermediate-Scale Quantum (NISQ) devices. Most QAS algorithms combine their search space and search algorithms together and thus generally require evaluating a large number of quantum circuits during the search process. Predictor-based QAS algorithms can alleviate this problem by directly estimating the performance of circuits according to their structures. However, a high-performance predictor generally requires very time-consuming labeling to obtain a large number of labeled quantum circuits. Recently, a classical neural architecture search algorithm Arch2vec inspires us by showing that architecture search can benefit from decoupling unsupervised representation learning from the search process. Whether unsupervised representation learning can help QAS w
    
[^6]: 高维度独立性检测: 通过最大和平均距离相关性

    High-Dimensional Independence Testing via Maximum and Average Distance Correlations

    [https://arxiv.org/abs/2001.01095](https://arxiv.org/abs/2001.01095)

    本文介绍并研究了利用最大和平均距离相关性进行高维度独立性检测的方法，并提出了一种快速卡方检验的程序。该方法适用于欧氏距离和高斯核，具有较好的实证表现和广泛的应用场景。

    

    本文介绍并研究了利用最大和平均距离相关性进行多元独立性检测的方法。我们在高维环境中表征了它们相对于边际相关维度数量的一致性特性，评估了每个检验统计量的优势，检查了它们各自的零分布，并提出了一种基于快速卡方检验的检测程序。得出的检验是非参数的，并适用于欧氏距离和高斯核作为底层度量。为了更好地理解所提出的测试的实际使用情况，我们在各种多元相关场景中评估了最大距离相关性、平均距离相关性和原始距离相关性的实证表现，同时进行了一个真实数据实验，以检测人类血浆中不同癌症类型和肽水平的存在。

    This paper introduces and investigates the utilization of maximum and average distance correlations for multivariate independence testing. We characterize their consistency properties in high-dimensional settings with respect to the number of marginally dependent dimensions, assess the advantages of each test statistic, examine their respective null distributions, and present a fast chi-square-based testing procedure. The resulting tests are non-parametric and applicable to both Euclidean distance and the Gaussian kernel as the underlying metric. To better understand the practical use cases of the proposed tests, we evaluate the empirical performance of the maximum distance correlation, average distance correlation, and the original distance correlation across various multivariate dependence scenarios, as well as conduct a real data experiment to test the presence of various cancer types and peptide levels in human plasma.
    
[^7]: 用于合作多智能体系统的先天价值驱动增强学习

    Innate-Values-driven Reinforcement Learning for Cooperative Multi-Agent Systems. (arXiv:2401.05572v1 [cs.LG])

    [http://arxiv.org/abs/2401.05572](http://arxiv.org/abs/2401.05572)

    本文提出了一个先天价值驱动增强学习（IVRL）模型，用于描述多智能体在合作中的复杂行为。该模型通过建立智能体对群体效用和系统成本的认知，满足其合作伙伴的需求，支持其社区并融入人类社会。

    

    先天价值描述了智能体的内在动机，反映了他们追求目标和发展多样技能以满足各种需求的固有兴趣和偏好。强化学习的本质是基于奖励驱动（如效用）的行为互动学习，类似于自然智能体。特别是在多智能体系统中，建立智能体对平衡群体效用和系统成本的认知，满足群体成员在合作中的需求，是个体为支持其社区和融入人类社会而学习的一个关键问题。本文提出了一种分层复合内在价值增强学习模型 - 先天价值驱动增强学习，用于描述多智能体合作中复杂的互动行为。

    Innate values describe agents' intrinsic motivations, which reflect their inherent interests and preferences to pursue goals and drive them to develop diverse skills satisfying their various needs. The essence of reinforcement learning (RL) is learning from interaction based on reward-driven (such as utilities) behaviors, much like natural agents. It is an excellent model to describe the innate-values-driven (IV) behaviors of AI agents. Especially in multi-agent systems (MAS), building the awareness of AI agents to balance the group utilities and system costs and satisfy group members' needs in their cooperation is a crucial problem for individuals learning to support their community and integrate human society in the long term. This paper proposes a hierarchical compound intrinsic value reinforcement learning model -innate-values-driven reinforcement learning termed IVRL to describe the complex behaviors of multi-agent interaction in their cooperation. We implement the IVRL architec
    
[^8]: 可证保健康商务字体尹包通过随机平滑(译注)水。

    Provably Robust Cost-Sensitive Learning via Randomized Smoothing. (arXiv:2310.08732v1 [cs.LG])

    [http://arxiv.org/abs/2310.08732](http://arxiv.org/abs/2310.08732)

    本研究通过随机平滑认证框架，为成本敏感的稳健分类器提供了严格的稳健性保证，并通过优化方案针对不同数据子组设计了细粒度认证半径，取得了优越的性能。

    

    我们关注于在成本敏感的情景下学习对抗性稳健分类器，在这种情况下，不同类别的对抗性变换的潜在危害被编码在一个二进制成本矩阵中。现有的方法要么是经验性的，无法证明稳健性，要么存在固有的可扩展性问题。在这项工作中，我们研究了随机平滑，一种更可扩展的稳健性认证框架，是否可以用于证明成本敏感的稳健性。建立在一种成本敏感认证半径的概念之上，我们展示了如何调整标准的随机平滑认证流程，为任何成本矩阵产生严格的稳健性保证。此外，通过针对不同数据子组设计的细粒度认证半径优化方案，我们提出了一种算法，用于训练针对成本敏感稳健性优化的平滑分类器。在图像基准测试和真实的医学数据集上进行了大量实验，证明了我们方法的优越性。

    We focus on learning adversarially robust classifiers under a cost-sensitive scenario, where the potential harm of different classwise adversarial transformations is encoded in a binary cost matrix. Existing methods are either empirical that cannot certify robustness or suffer from inherent scalability issues. In this work, we study whether randomized smoothing, a more scalable robustness certification framework, can be leveraged to certify cost-sensitive robustness. Built upon a notion of cost-sensitive certified radius, we show how to adapt the standard randomized smoothing certification pipeline to produce tight robustness guarantees for any cost matrix. In addition, with fine-grained certified radius optimization schemes specifically designed for different data subgroups, we propose an algorithm to train smoothed classifiers that are optimized for cost-sensitive robustness. Extensive experiments on image benchmarks and a real-world medical dataset demonstrate the superiority of our
    
[^9]: 稳定对比强化学习: 离线目标达成的技术

    Stabilizing Contrastive RL: Techniques for Offline Goal Reaching. (arXiv:2306.03346v1 [cs.LG])

    [http://arxiv.org/abs/2306.03346](http://arxiv.org/abs/2306.03346)

    本文提出了一种稳定的对比强化学习方法，通过浅而宽的结构，结合谨慎的权重初始化和数据增强等实验方法，在具有挑战性的仿真基准测试中显著提高了性能，并演示了对比方法可以解决现实世界的机器人任务。

    

    计算机视觉和自然语言处理领域已经开发了自监督方法，强化学习也可以被视为自监督问题：学习达到任何目标，而不需要人类指定的奖励或标签。然而，为强化学习建立自监督基础实际上面临着一些重要的挑战。基于此前对比学习方法，我们进行了细致的剖析实验，并发现一个浅而宽的结构，结合谨慎的权重初始化和数据增强，可以显着提高与对比强化学习方法的性能，特别是在具有挑战性的仿真基准测试中。此外，我们还演示了通过这些设计决策，对比方法可以解决现实世界的机器人操作任务，其中任务由训练后提供的单个目标图像指定。

    In the same way that the computer vision (CV) and natural language processing (NLP) communities have developed self-supervised methods, reinforcement learning (RL) can be cast as a self-supervised problem: learning to reach any goal, without requiring human-specified rewards or labels. However, actually building a self-supervised foundation for RL faces some important challenges. Building on prior contrastive approaches to this RL problem, we conduct careful ablation experiments and discover that a shallow and wide architecture, combined with careful weight initialization and data augmentation, can significantly boost the performance of these contrastive RL approaches on challenging simulated benchmarks. Additionally, we demonstrate that, with these design decisions, contrastive approaches can solve real-world robotic manipulation tasks, with tasks being specified by a single goal image provided after training.
    
[^10]: 柿子政治的面孔：使用机器学习比较政治领袖面部情感表达的差异

    The Face of Populism: Examining Differences in Facial Emotional Expressions of Political Leaders Using Machine Learning. (arXiv:2304.09914v1 [cs.CY])

    [http://arxiv.org/abs/2304.09914](http://arxiv.org/abs/2304.09914)

    本文使用机器学习的算法分析了来自15个不同国家的220个政治领袖的YouTube视频，总结了政治领袖面部情感表达的差异。

    

    网络媒体已经彻底改变了政治信息在全球范围内的传播和消费方式，这种转变促使政治人物采取新的策略来捕捉和保持选民的注意力。这些策略往往依赖于情感说服和吸引。随着虚拟空间中视觉内容越来越普遍，很多政治沟通也被标志着唤起情感的视频内容和图像。本文提供了一种新的分析方法。我们将基于现有训练好的卷积神经网络架构提供的Python库fer，应用一种基于深度学习的计算机视觉算法，对描绘来自15个不同国家的政治领袖的220个YouTube视频样本进行分析。该算法返回情绪分数，每一帧都代表6种情绪状态（愤怒，厌恶，恐惧，快乐，悲伤和惊讶）和一个中性表情。

    Online media has revolutionized the way political information is disseminated and consumed on a global scale, and this shift has compelled political figures to adopt new strategies of capturing and retaining voter attention. These strategies often rely on emotional persuasion and appeal, and as visual content becomes increasingly prevalent in virtual space, much of political communication too has come to be marked by evocative video content and imagery. The present paper offers a novel approach to analyzing material of this kind. We apply a deep-learning-based computer-vision algorithm to a sample of 220 YouTube videos depicting political leaders from 15 different countries, which is based on an existing trained convolutional neural network architecture provided by the Python library fer. The algorithm returns emotion scores representing the relative presence of 6 emotional states (anger, disgust, fear, happiness, sadness, and surprise) and a neutral expression for each frame of the pr
    
[^11]: 一般损失函数在高维情况下导致（近似）插值

    General Loss Functions Lead to (Approximate) Interpolation in High Dimensions. (arXiv:2303.07475v1 [stat.ML])

    [http://arxiv.org/abs/2303.07475](http://arxiv.org/abs/2303.07475)

    本研究提供了一般凸损失函数和超参数化阶段下梯度下降的隐含偏差的近似表征。具体而言是近似最小范数插值。

    

    我们提供了一个统一的框架，适用于一般凸损失函数和超参数化阶段的二元和多元分类设置，以近似地表征梯度下降的隐含偏差。具体而言，我们展示了在高维情况下梯度下降的隐含偏差近似于最小范数插值，最小范数插值来自于对平方损失的训练。与之前专门针对指数尾损失并使用中间支持向量机公式的研究不同，我们的框架直接基于Ji和Telgarsky（2021）的原始-对偶分析方法，通过新颖的敏感度分析提供了一般凸损失的新近似等效性结果。我们的框架还恢复了二元和多元分类设置下指数尾损失的现有精确等效结果。最后，我们提供了我们技术的紧密性的证据，我们使用这些证据来演示我们的方法的有效性。

    We provide a unified framework, applicable to a general family of convex losses and across binary and multiclass settings in the overparameterized regime, to approximately characterize the implicit bias of gradient descent in closed form. Specifically, we show that the implicit bias is approximated (but not exactly equal to) the minimum-norm interpolation in high dimensions, which arises from training on the squared loss. In contrast to prior work which was tailored to exponentially-tailed losses and used the intermediate support-vector-machine formulation, our framework directly builds on the primal-dual analysis of Ji and Telgarsky (2021), allowing us to provide new approximate equivalences for general convex losses through a novel sensitivity analysis. Our framework also recovers existing exact equivalence results for exponentially-tailed losses across binary and multiclass settings. Finally, we provide evidence for the tightness of our techniques, which we use to demonstrate the ef
    
[^12]: Refiner: 针对联邦学习中的梯度泄漏攻击的数据精炼方法

    Refiner: Data Refining against Gradient Leakage Attacks in Federated Learning. (arXiv:2212.02042v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2212.02042](http://arxiv.org/abs/2212.02042)

    Refiner提出了一种创新的防御范式，通过构建与原始数据具有低语义相似性的健壮数据，有效地混淆梯度泄漏攻击者，从而提高联邦学习系统的隐私保护能力。

    

    最近的研究引起了对联邦学习系统易受梯度泄漏攻击的关注。这类攻击利用客户端上传的梯度来重构其敏感数据，从而破坏了联邦学习的隐私保护能力。为了应对这一威胁，已经提出了各种防御机制来减轻攻击的影响，这些机制通过操纵上传的梯度来防止攻击。然而，实证评估表明这些防御措施在面对复杂攻击时具有有限的弹性，这表明迫切需要更有效的防御方法。本文提出了一种新的防御范式，不同于传统的梯度扰动方法，而是专注于构建健壮数据。直观地说，如果健壮数据与客户端原始数据具有很低的语义相似性，与健壮数据相关的梯度可以有效地混淆攻击者。为此，我们设计了Refiner，它同时优化了两个指标，用于隐私保护和...

    Recent works have brought attention to the vulnerability of Federated Learning (FL) systems to gradient leakage attacks. Such attacks exploit clients' uploaded gradients to reconstruct their sensitive data, thereby compromising the privacy protection capability of FL. In response, various defense mechanisms have been proposed to mitigate this threat by manipulating the uploaded gradients. Unfortunately, empirical evaluations have demonstrated limited resilience of these defenses against sophisticated attacks, indicating an urgent need for more effective defenses. In this paper, we explore a novel defensive paradigm that departs from conventional gradient perturbation approaches and instead focuses on the construction of robust data. Intuitively, if robust data exhibits low semantic similarity with clients' raw data, the gradients associated with robust data can effectively obfuscate attackers. To this end, we design Refiner that jointly optimizes two metrics for privacy protection and 
    
[^13]: 量化异构转移的精确高维渐近分析

    Precise High-Dimensional Asymptotics for Quantifying Heterogeneous Transfers. (arXiv:2010.11750v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2010.11750](http://arxiv.org/abs/2010.11750)

    本文利用随机矩阵理论在线性回归设置中，对于具有两个任务的高维情况下的常用估计量的超额风险进行了精确渐近分析。

    

    最近，学习一个任务时使用来自另一个任务的样本的问题引起了广泛关注。本文提出了一个基本问题：什么时候将来自两个任务的数据合并比单独学习一个任务更好？直观上，从一个任务到另一个任务的转移效应取决于数据集的转移，如样本大小和协方差矩阵。然而，量化这种转移效应是具有挑战性的，因为我们需要比较联合学习和单任务学习之间的风险，并且一个任务是否比另一个任务具有比较优势取决于两个任务之间确切的数据集转移类型。本文利用随机矩阵理论在具有两个任务的线性回归设置中解决了这一挑战。我们给出了在高维情况下一些常用估计量的超额风险的精确渐近分析，当样本大小与特征维度成比例增加时，固定比例。精确渐近分析以样本大小的函数形式给出。

    The problem of learning one task with samples from another task has received much interest recently. In this paper, we ask a fundamental question: when is combining data from two tasks better than learning one task alone? Intuitively, the transfer effect from one task to another task depends on dataset shifts such as sample sizes and covariance matrices. However, quantifying such a transfer effect is challenging since we need to compare the risks between joint learning and single-task learning, and the comparative advantage of one over the other depends on the exact kind of dataset shift between both tasks. This paper uses random matrix theory to tackle this challenge in a linear regression setting with two tasks. We give precise asymptotics about the excess risks of some commonly used estimators in the high-dimensional regime, when the sample sizes increase proportionally with the feature dimension at fixed ratios. The precise asymptotics is provided as a function of the sample sizes 
    

