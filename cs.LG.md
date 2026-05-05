# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A tutorial on learning from preferences and choices with Gaussian Processes](https://arxiv.org/abs/2403.11782) | 提供了一个使用高斯过程进行偏好学习的框架，能够将理性原则融入学习过程，涵盖了多种偏好学习模型。 |
| [^2] | [Analyzing Adversarial Inputs in Deep Reinforcement Learning](https://arxiv.org/abs/2402.05284) | 这篇论文通过形式验证的视角，分析了深度强化学习中对抗输入的特征，并提出了一个新的度量标准——对抗率，以及计算该度量标准的一套工具和算法。 |
| [^3] | [CCNETS: A Novel Brain-Inspired Approach for Enhanced Pattern Recognition in Imbalanced Datasets.](http://arxiv.org/abs/2401.04139) | CCNETS是一种新颖的脑启发方法，通过模拟大脑的信息处理，通过生成高质量的数据集来增强不平衡数据集中的模式识别，特别关注处理机器学习中的不平衡数据集的挑战。 |
| [^4] | [Optimal Fair Multi-Agent Bandits.](http://arxiv.org/abs/2306.04498) | 本文针对多智能体之间公平多臂赌博机学习问题提出了一种算法，通过分布式拍卖算法学习样本最优匹配，使用一种新的利用阶段和一种基于顺序统计的遗憾分析实现，相较于先前的结果遗憾阶数从$O(\log T \log\log T)$到了$O\left(N^3 \log N \log T \right)$，能够更好地处理多个智能体之间的依赖关系。 |

# 详细

[^1]: 使用高斯过程从偏好和选择中学习的教程

    A tutorial on learning from preferences and choices with Gaussian Processes

    [https://arxiv.org/abs/2403.11782](https://arxiv.org/abs/2403.11782)

    提供了一个使用高斯过程进行偏好学习的框架，能够将理性原则融入学习过程，涵盖了多种偏好学习模型。

    

    偏好建模位于经济学、决策理论、机器学习和统计学的交叉点。通过理解个体的偏好及其选择方式，我们可以构建更接近他们期望的产品，为跨领域的更高效、个性化应用铺平道路。此教程的目标是提供一个连贯、全面的偏好学习框架，使用高斯过程演示如何将理性原则（来自经济学和决策理论）无缝地纳入学习过程中。通过合适地定制似然函数，这一框架使得能够构建涵盖随机效用模型、辨识限制和对象和标签偏好的多重冲突效用情景的偏好学习模型。

    arXiv:2403.11782v1 Announce Type: new  Abstract: Preference modelling lies at the intersection of economics, decision theory, machine learning and statistics. By understanding individuals' preferences and how they make choices, we can build products that closely match their expectations, paving the way for more efficient and personalised applications across a wide range of domains. The objective of this tutorial is to present a cohesive and comprehensive framework for preference learning with Gaussian Processes (GPs), demonstrating how to seamlessly incorporate rationality principles (from economics and decision theory) into the learning process. By suitably tailoring the likelihood function, this framework enables the construction of preference learning models that encompass random utility models, limits of discernment, and scenarios with multiple conflicting utilities for both object- and label-preference. This tutorial builds upon established research while simultaneously introducin
    
[^2]: 分析深度强化学习中的对抗输入

    Analyzing Adversarial Inputs in Deep Reinforcement Learning

    [https://arxiv.org/abs/2402.05284](https://arxiv.org/abs/2402.05284)

    这篇论文通过形式验证的视角，分析了深度强化学习中对抗输入的特征，并提出了一个新的度量标准——对抗率，以及计算该度量标准的一套工具和算法。

    

    近年来，深度强化学习（DRL）由于在实际和复杂系统中取得的成功应用而成为机器学习中受欢迎的范例。然而，即使最先进的DRL模型也被证明存在可靠性问题，例如对抗输入的敏感性，即小型且大量的输入扰动会导致模型做出不可预测且潜在危险的决策。这个缺点限制了DRL系统在安全关键环境中的部署，即使是小的错误都是不可容忍的。在这项工作中，我们通过形式验证的视角提出了对对抗输入进行分类的新度量标准——对抗率，并提出了一套用于计算对抗率的工具和算法。我们的分析通过实验证明了对抗输入对DRL模型的影响。

    In recent years, Deep Reinforcement Learning (DRL) has become a popular paradigm in machine learning due to its successful applications to real-world and complex systems. However, even the state-of-the-art DRL models have been shown to suffer from reliability concerns -- for example, their susceptibility to adversarial inputs, i.e., small and abundant input perturbations that can fool the models into making unpredictable and potentially dangerous decisions. This drawback limits the deployment of DRL systems in safety-critical contexts, where even a small error cannot be tolerated. In this work, we present a comprehensive analysis of the characterization of adversarial inputs, through the lens of formal verification. Specifically, we introduce a novel metric, the Adversarial Rate, to classify models based on their susceptibility to such perturbations, and present a set of tools and algorithms for its computation. Our analysis empirically demonstrates how adversarial inputs can affect th
    
[^3]: CCNETS:一种新颖的脑启发方法用于增强不平衡数据集中的模式识别

    CCNETS: A Novel Brain-Inspired Approach for Enhanced Pattern Recognition in Imbalanced Datasets. (arXiv:2401.04139v1 [cs.LG])

    [http://arxiv.org/abs/2401.04139](http://arxiv.org/abs/2401.04139)

    CCNETS是一种新颖的脑启发方法，通过模拟大脑的信息处理，通过生成高质量的数据集来增强不平衡数据集中的模式识别，特别关注处理机器学习中的不平衡数据集的挑战。

    

    本研究介绍了CCNETS（具有因果合作网络的因果学习），这是一种新颖的基于生成模型的分类器，旨在解决模式识别中不平衡数据集生成的挑战。CCNETS独特地设计成模拟类似于大脑的信息处理，并包括三个主要组件：解释器、生成器和推理器。每个组件都被设计成模仿特定的大脑功能，有助于生成高质量的数据集并增强分类性能。该模型特别关注在机器学习中处理不平衡数据集的常见和重要挑战。通过将CCNETS应用于一个“欺诈数据集”，其中正常交易明显多于欺诈交易（99.83％ vs. 0.17％），证明了CCNETS的有效性。传统方法往往在处理这种不平衡时遇到困难，导致性能指标不均衡。然而，CCNETS展现出优越的分类能力，通过其性能指标的改善来体现。

    This study introduces CCNETS (Causal Learning with Causal Cooperative Nets), a novel generative model-based classifier designed to tackle the challenge of generating data for imbalanced datasets in pattern recognition. CCNETS is uniquely crafted to emulate brain-like information processing and comprises three main components: Explainer, Producer, and Reasoner. Each component is designed to mimic specific brain functions, which aids in generating high-quality datasets and enhancing classification performance.  The model is particularly focused on addressing the common and significant challenge of handling imbalanced datasets in machine learning. CCNETS's effectiveness is demonstrated through its application to a "fraud dataset," where normal transactions significantly outnumber fraudulent ones (99.83% vs. 0.17%). Traditional methods often struggle with such imbalances, leading to skewed performance metrics. However, CCNETS exhibits superior classification ability, as evidenced by its pe
    
[^4]: 公平多智能体赌博机的最优算法研究

    Optimal Fair Multi-Agent Bandits. (arXiv:2306.04498v1 [cs.LG])

    [http://arxiv.org/abs/2306.04498](http://arxiv.org/abs/2306.04498)

    本文针对多智能体之间公平多臂赌博机学习问题提出了一种算法，通过分布式拍卖算法学习样本最优匹配，使用一种新的利用阶段和一种基于顺序统计的遗憾分析实现，相较于先前的结果遗憾阶数从$O(\log T \log\log T)$到了$O\left(N^3 \log N \log T \right)$，能够更好地处理多个智能体之间的依赖关系。

    

    本文研究了在多个不相互通信的智能体之间进行公平的多臂赌博机学习的问题，这些智能体只有在同时访问同一个臂时才提供碰撞信息。我们提出了一种算法，其遗憾为$O\left(N^3 \log N \log T \right)$（假设奖励有界，但未知上界）。这大大改进了之前结果，其遗憾阶数为$O(\log T \log\log T)$，并且对智能体数量具有指数依赖性。结果是通过使用分布式拍卖算法来学习样本最优匹配，一种新的利用阶段，其长度来自于观察到的样本，以及一种基于顺序统计的遗憾分析实现的。仿真结果显示了遗憾对$\log T$的依存关系。

    In this paper, we study the problem of fair multi-agent multi-arm bandit learning when agents do not communicate with each other, except collision information, provided to agents accessing the same arm simultaneously. We provide an algorithm with regret $O\left(N^3 \log N \log T \right)$ (assuming bounded rewards, with unknown bound). This significantly improves previous results which had regret of order $O(\log T \log\log T)$ and exponential dependence on the number of agents. The result is attained by using a distributed auction algorithm to learn the sample-optimal matching, a new type of exploitation phase whose length is derived from the observed samples, and a novel order-statistics-based regret analysis. Simulation results present the dependence of the regret on $\log T$.
    

