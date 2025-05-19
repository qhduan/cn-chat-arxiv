# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FreeA: Human-object Interaction Detection using Free Annotation Labels](https://arxiv.org/abs/2403.01840) | 提出了一种新颖的自适应语言驱动的HOI检测方法FreeA，无需标记，利用了CLIP来生成潜在的HOI标签，并在两个基准数据集上取得了最先进的性能。 |
| [^2] | [COBIAS: Contextual Reliability in Bias Assessment](https://arxiv.org/abs/2402.14889) | 我们提出了COBIAS，旨在通过考虑多样情境的用户输入内容，衡量语句的情境可靠性，从而培养偏见意识。 |
| [^3] | [Wavelet Analysis of Noninvasive EEG Signals Discriminates Complex and Natural Grasp Types](https://arxiv.org/abs/2402.09447) | 该研究使用小波分析技术对非侵入性脑电图信号进行解码，成功区分复杂和自然的抓握类型，并且证明了小波特征在基于脑电图的抓握区分中的有效性。 |
| [^4] | [A Stability Principle for Learning under Non-Stationarity.](http://arxiv.org/abs/2310.18304) | 本研究提出了一个适用于非稳态环境的统计学习框架，通过应用稳定性原则选择回溯窗口来最大化历史数据利用，并保持累积偏差在可接受范围内。该方法展示了对未知非稳态的适应性，遗憾界在强凸或满足Lipschitz条件下是极小化的最优解。该研究的创新点是函数相似度度量和非稳态数据序列划分技术。 |
| [^5] | [Auditing Fairness by Betting.](http://arxiv.org/abs/2305.17570) | 本文提供了一种通过赌博的方式进行公平性审计的方法，相比之前的方法，这种方法具有更高的实用性和效率，能够对不断产生的数据进行连续的监控，并处理因分布漂移导致的公平性问题。 |

# 详细

[^1]: 使用自由注释标签进行人-物互动检测的FreeA方法

    FreeA: Human-object Interaction Detection using Free Annotation Labels

    [https://arxiv.org/abs/2403.01840](https://arxiv.org/abs/2403.01840)

    提出了一种新颖的自适应语言驱动的HOI检测方法FreeA，无需标记，利用了CLIP来生成潜在的HOI标签，并在两个基准数据集上取得了最先进的性能。

    

    最近的人-物互动（HOI）检测方法依赖于劳动力成本高昂，并需要全面注释的图像数据集。本文提出了一种新颖的自适应语言驱动的HOI检测方法FreeA，这种方法利用了CLIP的适应性来生成潜在的HOI标签，无需标记。具体而言，FreeA将人-物对的图像特征与HOI文本模板进行匹配，并开发了基于先验知识的掩模方法来抑制不太可能的交互作用。此外，FreeA利用了提出的交互相关性匹配方法来增强与指定动作相关的动作的可能性，进一步完善生成的HOI标签。在两个基准数据集上的实验证明，FreeA在弱监督HOI模型中实现了最先进的性能。我们的方法在HICO-DET上的平均精度（mAP）提高了+8.58，在V-COCO上提高了+1.23。

    arXiv:2403.01840v1 Announce Type: cross  Abstract: Recent human-object interaction (HOI) detection approaches rely on high cost of manpower and require comprehensive annotated image datasets. In this paper, we propose a novel self-adaption language-driven HOI detection method, termed as FreeA, without labeling by leveraging the adaptability of CLIP to generate latent HOI labels. To be specific, FreeA matches image features of human-object pairs with HOI text templates, and a priori knowledge-based mask method is developed to suppress improbable interactions. In addition, FreeA utilizes the proposed interaction correlation matching method to enhance the likelihood of actions related to a specified action, further refine the generated HOI labels. Experiments on two benchmark datasets show that FreeA achieves state-of-the-art performance among weakly supervised HOI models. Our approach is +8.58 mean Average Precision (mAP) on HICO-DET and +1.23 mAP on V-COCO more accurate in localizing an
    
[^2]: COBIAS：偏见评估中的情境可靠性

    COBIAS: Contextual Reliability in Bias Assessment

    [https://arxiv.org/abs/2402.14889](https://arxiv.org/abs/2402.14889)

    我们提出了COBIAS，旨在通过考虑多样情境的用户输入内容，衡量语句的情境可靠性，从而培养偏见意识。

    

    大型语言模型（LLMs）是基于固有偏见数据训练的。以往的去偏见模型研究依赖基准数据集来衡量模型性能。然而，这些数据集由于对偏见的极其主观理解而存在多个缺陷，凸显出对情境探索的迫切需求。我们提出考虑输入用户内容的情境，考虑到输入语句可能存在的多种情况。这种方法将允许培养偏见意识的框架，而不是伤害用户参与的防护设施。我们的贡献有两个方面：(i) 我们创建了一个包含2287个陈词滥调语句以及添加情境要点的数据集；(ii) 我们开发了面向情境的偏见指标和评估分数（COBIAS）来评估语句在衡量偏见方面的情境可靠性。我们的度量是衡量偏见基准数据集情境可靠性的重要预测因子。

    arXiv:2402.14889v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are trained on inherently biased data. Previous works on debiasing models rely on benchmark datasets to measure model performance. However, these datasets suffer from several pitfalls due to the extremely subjective understanding of bias, highlighting a critical need for contextual exploration. We propose understanding the context of user inputs with consideration of the diverse situations in which input statements are possible. This approach would allow for frameworks that foster bias awareness rather than guardrails that hurt user engagement. Our contribution is twofold: (i) we create a dataset of 2287 stereotyped statements augmented with points for adding context; (ii) we develop the Context-Oriented Bias Indicator and Assessment Score (COBIAS) to assess statements' contextual reliability in measuring bias. Our metric is a significant predictor of the contextual reliability of bias-benchmark datasets ($
    
[^3]: 非侵入性脑电图信号的小波分析区分复杂和自然的抓握类型

    Wavelet Analysis of Noninvasive EEG Signals Discriminates Complex and Natural Grasp Types

    [https://arxiv.org/abs/2402.09447](https://arxiv.org/abs/2402.09447)

    该研究使用小波分析技术对非侵入性脑电图信号进行解码，成功区分复杂和自然的抓握类型，并且证明了小波特征在基于脑电图的抓握区分中的有效性。

    

    该研究旨在通过对脑电图（EEG）信号进行解码，为灵巧的神经假肢开发和脑机接口（BCI）应用来区分手部抓握，特别是针对运动障碍患者。具体而言，它专注于使用一种新的基于脑电图的BCI平台和小波信号处理，区分两种复杂的自然力量和精确抓握类型以及一种中立条件作为无运动条件。小波分析涉及从小波能量系数生成时间频率和拓扑图。然后，通过使用机器学习技术和新型小波特征，我们实现了高平均准确率：多类别为85.16%，无运动 vs 力量为95.37%，无运动 vs 精确为95.40%，力量 vs 精确为88.07%，证明了这些特征在基于脑电图的抓握区分中的有效性。与先前的研究不同，我们研究的关键部分是排列特征重要性的部分。

    arXiv:2402.09447v1 Announce Type: cross  Abstract: This research aims to decode hand grasps from Electroencephalograms (EEGs) for dexterous neuroprosthetic development and Brain-Computer Interface (BCI) applications, especially for patients with motor disorders. Particularly, it focuses on distinguishing two complex natural power and precision grasps in addition to a neutral condition as a no-movement condition using a new EEG-based BCI platform and wavelet signal processing. Wavelet analysis involved generating time-frequency and topographic maps from wavelet power coefficients. Then, by using machine learning techniques with novel wavelet features, we achieved high average accuracies: 85.16% for multiclass, 95.37% for No-Movement vs Power, 95.40% for No-Movement vs Precision, and 88.07% for Power vs Precision, demonstrating the effectiveness of these features in EEG-based grasp differentiation. In contrast to previous studies, a critical part of our study was permutation feature impo
    
[^4]: 学习非稳态条件下的稳定性原则

    A Stability Principle for Learning under Non-Stationarity. (arXiv:2310.18304v1 [cs.LG])

    [http://arxiv.org/abs/2310.18304](http://arxiv.org/abs/2310.18304)

    本研究提出了一个适用于非稳态环境的统计学习框架，通过应用稳定性原则选择回溯窗口来最大化历史数据利用，并保持累积偏差在可接受范围内。该方法展示了对未知非稳态的适应性，遗憾界在强凸或满足Lipschitz条件下是极小化的最优解。该研究的创新点是函数相似度度量和非稳态数据序列划分技术。

    

    我们在非稳定环境中开发了一个灵活的统计学习框架。在每个时间段，我们的方法应用稳定性原则来选择一个回溯窗口，最大限度地利用历史数据，同时将累积偏差保持在与随机误差相对可接受的范围内。我们的理论展示了该方法对未知非稳定性的适应性。当人口损失函数强凸或仅满足Lipschitz条件时，遗憾界是极小化的最优解，仅受对数因子的影响。我们的分析核心是两个新颖的组成部分：函数之间的相似度度量和将非稳态数据序列划分为准稳态片段的分割技术。

    We develop a versatile framework for statistical learning in non-stationary environments. In each time period, our approach applies a stability principle to select a look-back window that maximizes the utilization of historical data while keeping the cumulative bias within an acceptable range relative to the stochastic error. Our theory showcases the adaptability of this approach to unknown non-stationarity. The regret bound is minimax optimal up to logarithmic factors when the population losses are strongly convex, or Lipschitz only. At the heart of our analysis lie two novel components: a measure of similarity between functions and a segmentation technique for dividing the non-stationary data sequence into quasi-stationary pieces.
    
[^5]: 通过赌博进行公平性审计

    Auditing Fairness by Betting. (arXiv:2305.17570v1 [stat.ML])

    [http://arxiv.org/abs/2305.17570](http://arxiv.org/abs/2305.17570)

    本文提供了一种通过赌博的方式进行公平性审计的方法，相比之前的方法，这种方法具有更高的实用性和效率，能够对不断产生的数据进行连续的监控，并处理因分布漂移导致的公平性问题。

    

    我们提供了实用、高效、非参数方法，用于审计已部署的分类和回归模型的公平性。相比之前依赖于固定样本量的方法，我们的方法是序贯的，并允许对不断产生的数据进行连续的监控，因此非常适用于跟踪现实世界系统的公平性。我们也允许数据通过概率策略进行收集，而不是从人口中均匀采样。这使得审计可以在为其他目的收集的数据上进行。此外，该策略可以随时间改变，并且不同的子人群可以使用不同的策略。最后，我们的方法可以处理因模型变更或基础人群变更导致的分布漂移。我们的方法基于最近关于 anytime-valid 推断和博弈统计学的进展，尤其是"通过赌博进行测试"框架。这些联系确保了我们的方法具有可解释性、快速和提供统计保证。

    We provide practical, efficient, and nonparametric methods for auditing the fairness of deployed classification and regression models. Whereas previous work relies on a fixed-sample size, our methods are sequential and allow for the continuous monitoring of incoming data, making them highly amenable to tracking the fairness of real-world systems. We also allow the data to be collected by a probabilistic policy as opposed to sampled uniformly from the population. This enables auditing to be conducted on data gathered for another purpose. Moreover, this policy may change over time and different policies may be used on different subpopulations. Finally, our methods can handle distribution shift resulting from either changes to the model or changes in the underlying population. Our approach is based on recent progress in anytime-valid inference and game-theoretic statistics-the "testing by betting" framework in particular. These connections ensure that our methods are interpretable, fast, 
    

