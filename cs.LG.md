# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SARDet-100K: Towards Open-Source Benchmark and ToolKit for Large-Scale SAR Object Detection](https://arxiv.org/abs/2403.06534) | SARDet-100K是第一个COCO级别的大规模多类别SAR物体检测数据集，为研究提供了大规模且多样化的数据集，揭示了SAR物体检测中预训练模型显著差异的关键挑战。 |
| [^2] | [Monte Carlo with kernel-based Gibbs measures: Guarantees for probabilistic herding](https://arxiv.org/abs/2402.11736) | 该论文研究了一种联合概率分布，其支持趋于最小化最坏情况误差，证明了它在最坏情况积分误差集中不等式上优于i.i.d.蒙特卡罗。 |
| [^3] | [Learn to Teach: Improve Sample Efficiency in Teacher-student Learning for Sim-to-Real Transfer](https://arxiv.org/abs/2402.06783) | 本文提出了一种样本效率学习框架，名为学习教学（L2T），通过回收教师智能体收集的经验，解决了教师-学生学习中的样本效率问题。 |
| [^4] | [Optimal Multi-Distribution Learning.](http://arxiv.org/abs/2312.05134) | 本论文提出了一种最优化多分布学习的方法，通过自适应采样来实现数据高效的学习。针对Vapnik-Chervonenkis (VC)维数为d的假设类，算法可以生成一个ε-最优随机假设，并且样本复杂度与最佳下界保持一致。同时，该算法的思想和理论还被进一步扩展以适应Rademacher类。最终提出的算法是奥拉克尔高效的，仅访问假设类。 |
| [^5] | [Empathy Detection Using Machine Learning on Text, Audiovisual, Audio or Physiological Signals.](http://arxiv.org/abs/2311.00721) | 本论文对共情检测领域的机器学习研究进行了综述和分析，包括文本、视听、音频和生理信号四种输入模态的处理和网络设计，以及评估协议和数据集的描述。 |
| [^6] | [Blending Imitation and Reinforcement Learning for Robust Policy Improvement.](http://arxiv.org/abs/2310.01737) | 本文提出了一种融合模仿学习和强化学习的方法，根据在线评估结果交替使用二者，以提高样本效率和学习效果。 |
| [^7] | [Thompson Exploration with Best Challenger Rule in Best Arm Identification.](http://arxiv.org/abs/2310.00539) | 本文提出了一种新的策略，将Thompson采样与最佳候选规则相结合，用于解决最佳臂识别问题。该策略在渐近情况下是最优的，并在一般的多臂赌博机问题中达到接近最优的性能。 |
| [^8] | [Active Policy Improvement from Multiple Black-box Oracles.](http://arxiv.org/abs/2306.10259) | 本研究提出了MAPS和MAPS-SE两个算法，可在多黑盒预言情况下，采用模仿学习并主动选择和改进最优预言，显著提升了性能。 |
| [^9] | [Efficient Utility Function Learning for Multi-Objective Parameter Optimization with Prior Knowledge.](http://arxiv.org/abs/2208.10300) | 该论文提出了一种利用偏好学习离线学习效用函数的方法，以应对真实世界问题中用专家知识定义效用函数困难且与专家反复互动昂贵的问题。使用效用函数空间的粗略信息，能够在使用很少结果时提高效用函数估计，并通过整个优化链中传递效用函数学习任务中出现的不确定性。 |

# 详细

[^1]: SARDet-100K: 面向大规模合成孔径雷达 SAR 物体检测的开源基准和工具包

    SARDet-100K: Towards Open-Source Benchmark and ToolKit for Large-Scale SAR Object Detection

    [https://arxiv.org/abs/2403.06534](https://arxiv.org/abs/2403.06534)

    SARDet-100K是第一个COCO级别的大规模多类别SAR物体检测数据集，为研究提供了大规模且多样化的数据集，揭示了SAR物体检测中预训练模型显著差异的关键挑战。

    

    面向合成孔径雷达（SAR）物体检测近来备受关注，因其不可替代的全天候成像能力。然而，这一研究领域面临着有限的公共数据集（主要包含 <2K 张图像，且仅包含单类别物体）和源代码不可访问的挑战。为解决这些问题，我们建立了一个新的基准数据集和一个针对大规模 SAR 物体检测的开源方法。我们的数据集 SARDet-100K 结果是对 10 个现有 SAR 检测数据集进行深入调研、收集和标准化的产物，为研究提供了一个大规模且多样化的数据集。据我们所知，SARDet-100K 是有史以来第一个达到 COCO 水平的大规模多类别 SAR 物体检测数据集。凭借这一高质量数据集，我们进行了全面实验，并揭示了 SAR 物体检测中一个关键挑战：预训练模型的显著差异。

    arXiv:2403.06534v1 Announce Type: cross  Abstract: Synthetic Aperture Radar (SAR) object detection has gained significant attention recently due to its irreplaceable all-weather imaging capabilities. However, this research field suffers from both limited public datasets (mostly comprising <2K images with only mono-category objects) and inaccessible source code. To tackle these challenges, we establish a new benchmark dataset and an open-source method for large-scale SAR object detection. Our dataset, SARDet-100K, is a result of intense surveying, collecting, and standardizing 10 existing SAR detection datasets, providing a large-scale and diverse dataset for research purposes. To the best of our knowledge, SARDet-100K is the first COCO-level large-scale multi-class SAR object detection dataset ever created. With this high-quality dataset, we conducted comprehensive experiments and uncovered a crucial challenge in SAR object detection: the substantial disparities between the pretraining
    
[^2]: 基于核Gibbs测度的蒙特卡罗：概率随机放牧的保证

    Monte Carlo with kernel-based Gibbs measures: Guarantees for probabilistic herding

    [https://arxiv.org/abs/2402.11736](https://arxiv.org/abs/2402.11736)

    该论文研究了一种联合概率分布，其支持趋于最小化最坏情况误差，证明了它在最坏情况积分误差集中不等式上优于i.i.d.蒙特卡罗。

    

    Kernel herding属于一类确定性的四位数法，旨在通过再生核希尔伯特空间（RKHS）上的最坏情况积分误差。尽管有很强的实验支持，但在通常情况下，即RKHS是无限维时，证明这种最坏情况误差以比标准积分节点数量的平方根更快的速率减少是困难的。在这篇理论论文中，我们研究了一个关于积分节点的联合概率分布，其支持趋于最小化与核放牧相同的最坏情况误差。我们证明它优于i.i.d.蒙特卡罗，意味着在最坏情况积分误差上具有更紧的集中不等式。尽管尚未提高速率，但这表明了研究Gibbs测度的数学工具可以帮助理解核放牧及其变体在计算上的改进程度

    arXiv:2402.11736v1 Announce Type: new  Abstract: Kernel herding belongs to a family of deterministic quadratures that seek to minimize the worst-case integration error over a reproducing kernel Hilbert space (RKHS). In spite of strong experimental support, it has revealed difficult to prove that this worst-case error decreases at a faster rate than the standard square root of the number of quadrature nodes, at least in the usual case where the RKHS is infinite-dimensional. In this theoretical paper, we study a joint probability distribution over quadrature nodes, whose support tends to minimize the same worst-case error as kernel herding. We prove that it does outperform i.i.d. Monte Carlo, in the sense of coming with a tighter concentration inequality on the worst-case integration error. While not improving the rate yet, this demonstrates that the mathematical tools of the study of Gibbs measures can help understand to what extent kernel herding and its variants improve on computation
    
[^3]: 学习教学：改善教师-学生学习中的样本效率，实现从模拟到现实的迁移

    Learn to Teach: Improve Sample Efficiency in Teacher-student Learning for Sim-to-Real Transfer

    [https://arxiv.org/abs/2402.06783](https://arxiv.org/abs/2402.06783)

    本文提出了一种样本效率学习框架，名为学习教学（L2T），通过回收教师智能体收集的经验，解决了教师-学生学习中的样本效率问题。

    

    模拟到现实（sim-to-real）的迁移是机器人学习中的一个基本问题。域随机化是一种在训练过程中添加随机性的强大技术，可以有效解决模拟与现实之间的差距。然而，观测中的噪声使得学习变得更加困难。最近的研究表明，采用教师-学生学习范式可以加速随机化环境中的训练。通过使用特权信息进行学习，教师智能体可以指导学生智能体在噪声环境中操作。然而，这种方法通常不是样本效率的，因为在训练学生智能体时完全舍弃了教师智能体收集的经验，浪费了环境所透露的信息。在这项工作中，我们通过提出一个名为学习教学（L2T）的样本效率学习框架来扩展教师-学生学习范式，该框架可以回收教师智能体收集的经验。我们观察到，对于一对教师-学生智能体，环境的动态特性对两者都有重要影响。

    Simulation-to-reality (sim-to-real) transfer is a fundamental problem for robot learning. Domain Randomization, which adds randomization during training, is a powerful technique that effectively addresses the sim-to-real gap. However, the noise in observations makes learning significantly harder. Recently, studies have shown that employing a teacher-student learning paradigm can accelerate training in randomized environments. Learned with privileged information, a teacher agent can instruct the student agent to operate in noisy environments. However, this approach is often not sample efficient as the experience collected by the teacher is discarded completely when training the student, wasting information revealed by the environment. In this work, we extend the teacher-student learning paradigm by proposing a sample efficient learning framework termed Learn to Teach (L2T) that recycles experience collected by the teacher agent. We observe that the dynamics of the environments for both 
    
[^4]: 最优化多分布学习

    Optimal Multi-Distribution Learning. (arXiv:2312.05134v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2312.05134](http://arxiv.org/abs/2312.05134)

    本论文提出了一种最优化多分布学习的方法，通过自适应采样来实现数据高效的学习。针对Vapnik-Chervonenkis (VC)维数为d的假设类，算法可以生成一个ε-最优随机假设，并且样本复杂度与最佳下界保持一致。同时，该算法的思想和理论还被进一步扩展以适应Rademacher类。最终提出的算法是奥拉克尔高效的，仅访问假设类。

    

    多分布学习（MDL）旨在学习一个共享模型，使得在k个不同的数据分布下，最小化最坏情况风险，已成为适应健壮性、公平性、多组合作等需求的统一框架。实现数据高效的MDL需要在学习过程中进行自适应采样，也称为按需采样。然而，最优样本复杂度的上下界之间存在较大差距。针对Vapnik-Chervonenkis（VC）维数为d的假设类，我们提出了一种新颖的算法，可生成一个ε-最优随机假设，其样本复杂度接近于（d+k）/ε^2（在某些对数因子中），与已知的最佳下界匹配。我们的算法思想和理论被进一步扩展，以适应Rademacher类。提出的算法是奥拉克尔高效的，仅仅访问假设类

    Multi-distribution learning (MDL), which seeks to learn a shared model that minimizes the worst-case risk across $k$ distinct data distributions, has emerged as a unified framework in response to the evolving demand for robustness, fairness, multi-group collaboration, etc. Achieving data-efficient MDL necessitates adaptive sampling, also called on-demand sampling, throughout the learning process. However, there exist substantial gaps between the state-of-the-art upper and lower bounds on the optimal sample complexity. Focusing on a hypothesis class of Vapnik-Chervonenkis (VC) dimension $d$, we propose a novel algorithm that yields an $varepsilon$-optimal randomized hypothesis with a sample complexity on the order of $(d+k)/\varepsilon^2$ (modulo some logarithmic factor), matching the best-known lower bound. Our algorithmic ideas and theory have been further extended to accommodate Rademacher classes. The proposed algorithms are oracle-efficient, which access the hypothesis class solely
    
[^5]: 使用机器学习在文本、视听、音频或生理信号上进行共情检测

    Empathy Detection Using Machine Learning on Text, Audiovisual, Audio or Physiological Signals. (arXiv:2311.00721v1 [cs.HC])

    [http://arxiv.org/abs/2311.00721](http://arxiv.org/abs/2311.00721)

    本论文对共情检测领域的机器学习研究进行了综述和分析，包括文本、视听、音频和生理信号四种输入模态的处理和网络设计，以及评估协议和数据集的描述。

    

    共情是一个社交技能，表明一个个体理解他人的能力。近年来，共情引起了包括情感计算、认知科学和心理学在内的各个学科的关注。共情是一个依赖于上下文的术语，因此检测或识别共情在社会、医疗和教育等领域具有潜在的应用。尽管共情检测领域涉及范围广泛且有重叠，但从整体文献角度来看，利用机器学习的共情检测研究仍然相对较少。为此，我们系统收集和筛选了来自10个知名数据库的801篇论文，并分析了选定的54篇论文。我们根据共情检测系统的输入模态，即文本、视听、音频和生理信号，对论文进行分组。我们分别研究了特定模态的预处理和网络架构设计协议、常见数据集的描述和可用性详情，以及评估协议。

    Empathy is a social skill that indicates an individual's ability to understand others. Over the past few years, empathy has drawn attention from various disciplines, including but not limited to Affective Computing, Cognitive Science and Psychology. Empathy is a context-dependent term; thus, detecting or recognising empathy has potential applications in society, healthcare and education. Despite being a broad and overlapping topic, the avenue of empathy detection studies leveraging Machine Learning remains underexplored from a holistic literature perspective. To this end, we systematically collect and screen 801 papers from 10 well-known databases and analyse the selected 54 papers. We group the papers based on input modalities of empathy detection systems, i.e., text, audiovisual, audio and physiological signals. We examine modality-specific pre-processing and network architecture design protocols, popular dataset descriptions and availability details, and evaluation protocols. We fur
    
[^6]: 融合模仿学习和强化学习以实现鲁棒策略改进

    Blending Imitation and Reinforcement Learning for Robust Policy Improvement. (arXiv:2310.01737v1 [cs.LG])

    [http://arxiv.org/abs/2310.01737](http://arxiv.org/abs/2310.01737)

    本文提出了一种融合模仿学习和强化学习的方法，根据在线评估结果交替使用二者，以提高样本效率和学习效果。

    

    虽然强化学习在性能上表现出色，但其样本复杂度仍然是一个重大障碍，限制了其在各个领域的广泛应用。模仿学习利用神经网络优化样本效率，但通常受到所使用的专家示范的质量限制。本文介绍了一种融合模仿学习和强化学习的方法，该方法根据在线评估结果交替使用二者，有效地提高了学习效率。这种算法能够从多种黑盒专家示范中学习和改进。

    While reinforcement learning (RL) has shown promising performance, its sample complexity continues to be a substantial hurdle, restricting its broader application across a variety of domains. Imitation learning (IL) utilizes oracles to improve sample efficiency, yet it is often constrained by the quality of the oracles deployed. which actively interleaves between IL and RL based on an online estimate of their performance. RPI draws on the strengths of IL, using oracle queries to facilitate exploration, an aspect that is notably challenging in sparse-reward RL, particularly during the early stages of learning. As learning unfolds, RPI gradually transitions to RL, effectively treating the learned policy as an improved oracle. This algorithm is capable of learning from and improving upon a diverse set of black-box oracles. Integral to RPI are Robust Active Policy Selection (RAPS) and Robust Policy Gradient (RPG), both of which reason over whether to perform state-wise imitation from the o
    
[^7]: 最佳候选规则下的Thompson探索在最佳臂识别中的应用

    Thompson Exploration with Best Challenger Rule in Best Arm Identification. (arXiv:2310.00539v1 [stat.ML])

    [http://arxiv.org/abs/2310.00539](http://arxiv.org/abs/2310.00539)

    本文提出了一种新的策略，将Thompson采样与最佳候选规则相结合，用于解决最佳臂识别问题。该策略在渐近情况下是最优的，并在一般的多臂赌博机问题中达到接近最优的性能。

    

    本文研究了在经典单参数指数模型下，固定置信度下的最佳臂识别（BAI）问题。针对这个问题，目前已有很多策略被提出，但大多数需要在每一轮解决一个最优化问题和/或者需要探索一个臂至少一定次数，除非是针对高斯模型的限制。为了解决这些限制，我们提出了一种新的策略，将Thompson采样与一个计算效率高的方法——最佳候选规则相结合。虽然Thompson采样最初被考虑用于最大化累积奖励，但我们证明它也可以自然地用于在BAI中探索臂而不强迫最大化奖励。我们证明了我们的策略在任意两臂赌博机问题上是渐近最优的，并且在一般的$K$臂赌博机问题上（$K\geq 3$）达到接近最优的性能。然而，在数值实验中，我们的策略与现有方法相比表现出了竞争性的性能。

    This paper studies the fixed-confidence best arm identification (BAI) problem in the bandit framework in the canonical single-parameter exponential models. For this problem, many policies have been proposed, but most of them require solving an optimization problem at every round and/or are forced to explore an arm at least a certain number of times except those restricted to the Gaussian model. To address these limitations, we propose a novel policy that combines Thompson sampling with a computationally efficient approach known as the best challenger rule. While Thompson sampling was originally considered for maximizing the cumulative reward, we demonstrate that it can be used to naturally explore arms in BAI without forcing it. We show that our policy is asymptotically optimal for any two-armed bandit problems and achieves near optimality for general $K$-armed bandit problems for $K\geq 3$. Nevertheless, in numerical experiments, our policy shows competitive performance compared to as
    
[^8]: 多黑盒预言下的主动策略改进

    Active Policy Improvement from Multiple Black-box Oracles. (arXiv:2306.10259v1 [cs.LG])

    [http://arxiv.org/abs/2306.10259](http://arxiv.org/abs/2306.10259)

    本研究提出了MAPS和MAPS-SE两个算法，可在多黑盒预言情况下，采用模仿学习并主动选择和改进最优预言，显著提升了性能。

    

    强化学习在各种复杂领域中取得了重大进展，但是通过强化学习确定有效策略往往需要进行广泛的探索，而模仿学习旨在通过使用专家演示来指导探索，缓解这个问题。在真实世界情境下，人们通常只能接触到多个次优的黑盒预言，而不是单个最优的预言，这些预言不能在所有状态下普遍优于彼此，这给主动决定在哪种状态下使用哪种预言以及如何改进各自估计值函数提出了挑战。本文介绍了一个可行的解决方案，即MAPS和MAPS-SE算法。

    Reinforcement learning (RL) has made significant strides in various complex domains. However, identifying an effective policy via RL often necessitates extensive exploration. Imitation learning aims to mitigate this issue by using expert demonstrations to guide exploration. In real-world scenarios, one often has access to multiple suboptimal black-box experts, rather than a single optimal oracle. These experts do not universally outperform each other across all states, presenting a challenge in actively deciding which oracle to use and in which state. We introduce MAPS and MAPS-SE, a class of policy improvement algorithms that perform imitation learning from multiple suboptimal oracles. In particular, MAPS actively selects which of the oracles to imitate and improve their value function estimates, and MAPS-SE additionally leverages an active state exploration criterion to determine which states one should explore. We provide a comprehensive theoretical analysis and demonstrate that MAP
    
[^9]: 多目标参数优化中的有效效用函数学习与先验知识

    Efficient Utility Function Learning for Multi-Objective Parameter Optimization with Prior Knowledge. (arXiv:2208.10300v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2208.10300](http://arxiv.org/abs/2208.10300)

    该论文提出了一种利用偏好学习离线学习效用函数的方法，以应对真实世界问题中用专家知识定义效用函数困难且与专家反复互动昂贵的问题。使用效用函数空间的粗略信息，能够在使用很少结果时提高效用函数估计，并通过整个优化链中传递效用函数学习任务中出现的不确定性。

    

    目前的多目标优化技术通常假定已有效用函数、通过互动学习效用函数或尝试确定完整的Pareto前沿来进行。然而，在真实世界的问题中，结果往往基于隐含和显性的专家知识，难以定义一个效用函数，而互动学习或后续启发式需要反复并且昂贵地专家参与。为了缓解这种情况，我们使用偏好学习离线学习效用函数，利用专家知识。与其他工作不同的是，我们不仅使用（成对的）结果偏好，而且使用效用函数空间的粗略信息。这使我们能够提高效用函数估计，特别是在使用很少的结果时。此外，我们对效用函数学习任务中出现的不确定性进行建模，并将其传递到整个优化链中。

    The current state-of-the-art in multi-objective optimization assumes either a given utility function, learns a utility function interactively or tries to determine the complete Pareto front, requiring a post elicitation of the preferred result. However, result elicitation in real world problems is often based on implicit and explicit expert knowledge, making it difficult to define a utility function, whereas interactive learning or post elicitation requires repeated and expensive expert involvement. To mitigate this, we learn a utility function offline, using expert knowledge by means of preference learning. In contrast to other works, we do not only use (pairwise) result preferences, but also coarse information about the utility function space. This enables us to improve the utility function estimate, especially when using very few results. Additionally, we model the occurring uncertainties in the utility function learning task and propagate them through the whole optimization chain. 
    

