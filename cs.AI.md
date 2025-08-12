# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Griffon v2: Advancing Multimodal Perception with High-Resolution Scaling and Visual-Language Co-Referring](https://arxiv.org/abs/2403.09333) | Griffon v2通过引入高分辨率缩放和视觉-语言共指，提升了多模态感知能力，尤其是对于小对象的感知能力。 |
| [^2] | [SARDet-100K: Towards Open-Source Benchmark and ToolKit for Large-Scale SAR Object Detection](https://arxiv.org/abs/2403.06534) | SARDet-100K是第一个COCO级别的大规模多类别SAR物体检测数据集，为研究提供了大规模且多样化的数据集，揭示了SAR物体检测中预训练模型显著差异的关键挑战。 |
| [^3] | [Blending Imitation and Reinforcement Learning for Robust Policy Improvement.](http://arxiv.org/abs/2310.01737) | 本文提出了一种融合模仿学习和强化学习的方法，根据在线评估结果交替使用二者，以提高样本效率和学习效果。 |
| [^4] | [Active Policy Improvement from Multiple Black-box Oracles.](http://arxiv.org/abs/2306.10259) | 本研究提出了MAPS和MAPS-SE两个算法，可在多黑盒预言情况下，采用模仿学习并主动选择和改进最优预言，显著提升了性能。 |
| [^5] | [Efficient Utility Function Learning for Multi-Objective Parameter Optimization with Prior Knowledge.](http://arxiv.org/abs/2208.10300) | 该论文提出了一种利用偏好学习离线学习效用函数的方法，以应对真实世界问题中用专家知识定义效用函数困难且与专家反复互动昂贵的问题。使用效用函数空间的粗略信息，能够在使用很少结果时提高效用函数估计，并通过整个优化链中传递效用函数学习任务中出现的不确定性。 |

# 详细

[^1]: Griffon v2：通过高分辨率缩放和视觉-语言共指推进多模态感知

    Griffon v2: Advancing Multimodal Perception with High-Resolution Scaling and Visual-Language Co-Referring

    [https://arxiv.org/abs/2403.09333](https://arxiv.org/abs/2403.09333)

    Griffon v2通过引入高分辨率缩放和视觉-语言共指，提升了多模态感知能力，尤其是对于小对象的感知能力。

    

    大型视觉语言模型已经实现了细粒度对象感知，但图像分辨率的限制仍然是超越复杂和密集场景中特定任务专家表现的重要障碍。为了解决这一问题，我们引入了一个统一的高分辨率通用模型，Griffon v2，实现了具有视觉和文本提示的灵活对象引用。为了有效提高图像分辨率，我们设计了一个简单轻量级的下采样投影仪，以克服大型语言模型中输入标记的约束。这种设计固有地保留了完整的上下文和细节，并显著提高了多模态感知能力，特别是对于小对象。在此基础上，我们进一步为模型配置了视觉-语言共指。

    arXiv:2403.09333v1 Announce Type: cross  Abstract: Large Vision Language Models have achieved fine-grained object perception, but the limitation of image resolution remains a significant obstacle to surpass the performance of task-specific experts in complex and dense scenarios. Such limitation further restricts the model's potential to achieve nuanced visual and language referring in domains such as GUI Agents, Counting and \etc. To address this issue, we introduce a unified high-resolution generalist model, Griffon v2, enabling flexible object referring with visual and textual prompts. To efficiently scaling up image resolution, we design a simple and lightweight down-sampling projector to overcome the input tokens constraint in Large Language Models. This design inherently preserves the complete contexts and fine details, and significantly improves multimodal perception ability especially for small objects. Building upon this, we further equip the model with visual-language co-refer
    
[^2]: SARDet-100K: 面向大规模合成孔径雷达 SAR 物体检测的开源基准和工具包

    SARDet-100K: Towards Open-Source Benchmark and ToolKit for Large-Scale SAR Object Detection

    [https://arxiv.org/abs/2403.06534](https://arxiv.org/abs/2403.06534)

    SARDet-100K是第一个COCO级别的大规模多类别SAR物体检测数据集，为研究提供了大规模且多样化的数据集，揭示了SAR物体检测中预训练模型显著差异的关键挑战。

    

    面向合成孔径雷达（SAR）物体检测近来备受关注，因其不可替代的全天候成像能力。然而，这一研究领域面临着有限的公共数据集（主要包含 <2K 张图像，且仅包含单类别物体）和源代码不可访问的挑战。为解决这些问题，我们建立了一个新的基准数据集和一个针对大规模 SAR 物体检测的开源方法。我们的数据集 SARDet-100K 结果是对 10 个现有 SAR 检测数据集进行深入调研、收集和标准化的产物，为研究提供了一个大规模且多样化的数据集。据我们所知，SARDet-100K 是有史以来第一个达到 COCO 水平的大规模多类别 SAR 物体检测数据集。凭借这一高质量数据集，我们进行了全面实验，并揭示了 SAR 物体检测中一个关键挑战：预训练模型的显著差异。

    arXiv:2403.06534v1 Announce Type: cross  Abstract: Synthetic Aperture Radar (SAR) object detection has gained significant attention recently due to its irreplaceable all-weather imaging capabilities. However, this research field suffers from both limited public datasets (mostly comprising <2K images with only mono-category objects) and inaccessible source code. To tackle these challenges, we establish a new benchmark dataset and an open-source method for large-scale SAR object detection. Our dataset, SARDet-100K, is a result of intense surveying, collecting, and standardizing 10 existing SAR detection datasets, providing a large-scale and diverse dataset for research purposes. To the best of our knowledge, SARDet-100K is the first COCO-level large-scale multi-class SAR object detection dataset ever created. With this high-quality dataset, we conducted comprehensive experiments and uncovered a crucial challenge in SAR object detection: the substantial disparities between the pretraining
    
[^3]: 融合模仿学习和强化学习以实现鲁棒策略改进

    Blending Imitation and Reinforcement Learning for Robust Policy Improvement. (arXiv:2310.01737v1 [cs.LG])

    [http://arxiv.org/abs/2310.01737](http://arxiv.org/abs/2310.01737)

    本文提出了一种融合模仿学习和强化学习的方法，根据在线评估结果交替使用二者，以提高样本效率和学习效果。

    

    虽然强化学习在性能上表现出色，但其样本复杂度仍然是一个重大障碍，限制了其在各个领域的广泛应用。模仿学习利用神经网络优化样本效率，但通常受到所使用的专家示范的质量限制。本文介绍了一种融合模仿学习和强化学习的方法，该方法根据在线评估结果交替使用二者，有效地提高了学习效率。这种算法能够从多种黑盒专家示范中学习和改进。

    While reinforcement learning (RL) has shown promising performance, its sample complexity continues to be a substantial hurdle, restricting its broader application across a variety of domains. Imitation learning (IL) utilizes oracles to improve sample efficiency, yet it is often constrained by the quality of the oracles deployed. which actively interleaves between IL and RL based on an online estimate of their performance. RPI draws on the strengths of IL, using oracle queries to facilitate exploration, an aspect that is notably challenging in sparse-reward RL, particularly during the early stages of learning. As learning unfolds, RPI gradually transitions to RL, effectively treating the learned policy as an improved oracle. This algorithm is capable of learning from and improving upon a diverse set of black-box oracles. Integral to RPI are Robust Active Policy Selection (RAPS) and Robust Policy Gradient (RPG), both of which reason over whether to perform state-wise imitation from the o
    
[^4]: 多黑盒预言下的主动策略改进

    Active Policy Improvement from Multiple Black-box Oracles. (arXiv:2306.10259v1 [cs.LG])

    [http://arxiv.org/abs/2306.10259](http://arxiv.org/abs/2306.10259)

    本研究提出了MAPS和MAPS-SE两个算法，可在多黑盒预言情况下，采用模仿学习并主动选择和改进最优预言，显著提升了性能。

    

    强化学习在各种复杂领域中取得了重大进展，但是通过强化学习确定有效策略往往需要进行广泛的探索，而模仿学习旨在通过使用专家演示来指导探索，缓解这个问题。在真实世界情境下，人们通常只能接触到多个次优的黑盒预言，而不是单个最优的预言，这些预言不能在所有状态下普遍优于彼此，这给主动决定在哪种状态下使用哪种预言以及如何改进各自估计值函数提出了挑战。本文介绍了一个可行的解决方案，即MAPS和MAPS-SE算法。

    Reinforcement learning (RL) has made significant strides in various complex domains. However, identifying an effective policy via RL often necessitates extensive exploration. Imitation learning aims to mitigate this issue by using expert demonstrations to guide exploration. In real-world scenarios, one often has access to multiple suboptimal black-box experts, rather than a single optimal oracle. These experts do not universally outperform each other across all states, presenting a challenge in actively deciding which oracle to use and in which state. We introduce MAPS and MAPS-SE, a class of policy improvement algorithms that perform imitation learning from multiple suboptimal oracles. In particular, MAPS actively selects which of the oracles to imitate and improve their value function estimates, and MAPS-SE additionally leverages an active state exploration criterion to determine which states one should explore. We provide a comprehensive theoretical analysis and demonstrate that MAP
    
[^5]: 多目标参数优化中的有效效用函数学习与先验知识

    Efficient Utility Function Learning for Multi-Objective Parameter Optimization with Prior Knowledge. (arXiv:2208.10300v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2208.10300](http://arxiv.org/abs/2208.10300)

    该论文提出了一种利用偏好学习离线学习效用函数的方法，以应对真实世界问题中用专家知识定义效用函数困难且与专家反复互动昂贵的问题。使用效用函数空间的粗略信息，能够在使用很少结果时提高效用函数估计，并通过整个优化链中传递效用函数学习任务中出现的不确定性。

    

    目前的多目标优化技术通常假定已有效用函数、通过互动学习效用函数或尝试确定完整的Pareto前沿来进行。然而，在真实世界的问题中，结果往往基于隐含和显性的专家知识，难以定义一个效用函数，而互动学习或后续启发式需要反复并且昂贵地专家参与。为了缓解这种情况，我们使用偏好学习离线学习效用函数，利用专家知识。与其他工作不同的是，我们不仅使用（成对的）结果偏好，而且使用效用函数空间的粗略信息。这使我们能够提高效用函数估计，特别是在使用很少的结果时。此外，我们对效用函数学习任务中出现的不确定性进行建模，并将其传递到整个优化链中。

    The current state-of-the-art in multi-objective optimization assumes either a given utility function, learns a utility function interactively or tries to determine the complete Pareto front, requiring a post elicitation of the preferred result. However, result elicitation in real world problems is often based on implicit and explicit expert knowledge, making it difficult to define a utility function, whereas interactive learning or post elicitation requires repeated and expensive expert involvement. To mitigate this, we learn a utility function offline, using expert knowledge by means of preference learning. In contrast to other works, we do not only use (pairwise) result preferences, but also coarse information about the utility function space. This enables us to improve the utility function estimate, especially when using very few results. Additionally, we model the occurring uncertainties in the utility function learning task and propagate them through the whole optimization chain. 
    

