# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [InterDreamer: Zero-Shot Text to 3D Dynamic Human-Object Interaction](https://arxiv.org/abs/2403.19652) | 通过解耦交互语义和动态，本文展示了在没有直接训练文本-交互对数据的情况下生成人物-物体交互的潜力。 |
| [^2] | [Deep Learning for Trajectory Data Management and Mining: A Survey and Beyond](https://arxiv.org/abs/2403.14151) | 本文综述了深度学习在轨迹数据管理与挖掘中的发展和最新进展，探讨了其在预处理、存储、分析、预测、推荐、分类、估计和检测等方面的应用。 |
| [^3] | [Generating Synthetic Health Sensor Data for Privacy-Preserving Wearable Stress Detection.](http://arxiv.org/abs/2401.13327) | 本论文介绍了一种隐私保护的合成健康传感器数据的方法，通过生成对抗网络（GANs）和差分隐私（DP）防护，生成与压力时刻相关的合成序列数据，确保患者信息的保护，并对合成数据进行质量评估。在压力检测数据集上验证了该方法的有效性。 |
| [^4] | [T-COL: Generating Counterfactual Explanations for General User Preferences on Variable Machine Learning Systems.](http://arxiv.org/abs/2309.16146) | 该论文提出了一个名为T-COL的方法，针对可变的机器学习系统和一般用户偏好生成反事实解释。这些解释不仅能够解释预测结果的原因，还提供了可操作的建议给用户。通过将一般用户偏好映射到CEs的属性上，以及采用定制化的方式来适应可变的机器学习模型，T-COL能够克服现有挑战并保持健壮性。 |
| [^5] | [Privacy in Practice: Private COVID-19 Detection in X-Ray Images (Extended Version).](http://arxiv.org/abs/2211.11434) | 该研究提出了通过差分隐私保护COVID-19检测模型，解决数据分析和患者隐私保护的问题。通过黑盒成员推理攻击，实现了对实际隐私的评估，结论表明所需的隐私等级可能因受到实际威胁的任务而异。 |

# 详细

[^1]: InterDreamer：零样本文本到三维动态人物-物体交互

    InterDreamer: Zero-Shot Text to 3D Dynamic Human-Object Interaction

    [https://arxiv.org/abs/2403.19652](https://arxiv.org/abs/2403.19652)

    通过解耦交互语义和动态，本文展示了在没有直接训练文本-交互对数据的情况下生成人物-物体交互的潜力。

    

    arXiv:2403.19652v1 宣布类型：跨领域 摘要：在广泛的动作捕捉数据和相应的文本注释上训练的扩散模型已经显著推动了文本条件的人体运动生成。然而，将这种成功延伸到三维动态人物-物体交互（HOI）生成面临着显著挑战，主要是由于缺乏大规模交互数据和与这些交互一致的全面描述。本文采取了行动，并展示了在没有直接训练文本-交互对数据的情况下生成人物-物体交互的潜力。我们在实现这一点的关键见解是交互语义和动态可以解耦。无法通过监督训练学习交互语义，我们转而利用预训练的大型模型，将来自大型语言模型和文本到运动模型的知识相辅相成。尽管这样的知识提供了对交互语义的高级控制，但不能提供到不成对交互文本的直接学习。

    arXiv:2403.19652v1 Announce Type: cross  Abstract: Text-conditioned human motion generation has experienced significant advancements with diffusion models trained on extensive motion capture data and corresponding textual annotations. However, extending such success to 3D dynamic human-object interaction (HOI) generation faces notable challenges, primarily due to the lack of large-scale interaction data and comprehensive descriptions that align with these interactions. This paper takes the initiative and showcases the potential of generating human-object interactions without direct training on text-interaction pair data. Our key insight in achieving this is that interaction semantics and dynamics can be decoupled. Being unable to learn interaction semantics through supervised training, we instead leverage pre-trained large models, synergizing knowledge from a large language model and a text-to-motion model. While such knowledge offers high-level control over interaction semantics, it c
    
[^2]: 轨迹数据管理与挖掘的深度学习：调查与展望

    Deep Learning for Trajectory Data Management and Mining: A Survey and Beyond

    [https://arxiv.org/abs/2403.14151](https://arxiv.org/abs/2403.14151)

    本文综述了深度学习在轨迹数据管理与挖掘中的发展和最新进展，探讨了其在预处理、存储、分析、预测、推荐、分类、估计和检测等方面的应用。

    

    arXiv:2403.14151v1 公告类型：跨越 抽象：轨迹计算是一个重要的领域，涵盖轨迹数据管理和挖掘，因其在诸如位置服务、城市交通和公共安全等各种实际应用中的关键作用而受到广泛关注。传统方法侧重于简单的时空特征，面临复杂计算、有限的可扩展性和不足以适应现实复杂性的挑战。在本文中，我们对轨迹计算中深度学习的发展和最新进展进行了全面的回顾（DL4Traj）。我们首先定义轨迹数据，并简要介绍了广泛使用的深度学习模型。系统地探讨了深度学习在轨迹管理（预处理、存储、分析和可视化）和挖掘（与轨迹相关的预测、轨迹相关的推荐、轨迹分类、旅行时间估计、异常检测）

    arXiv:2403.14151v1 Announce Type: cross  Abstract: Trajectory computing is a pivotal domain encompassing trajectory data management and mining, garnering widespread attention due to its crucial role in various practical applications such as location services, urban traffic, and public safety. Traditional methods, focusing on simplistic spatio-temporal features, face challenges of complex calculations, limited scalability, and inadequate adaptability to real-world complexities. In this paper, we present a comprehensive review of the development and recent advances in deep learning for trajectory computing (DL4Traj). We first define trajectory data and provide a brief overview of widely-used deep learning models. Systematically, we explore deep learning applications in trajectory management (pre-processing, storage, analysis, and visualization) and mining (trajectory-related forecasting, trajectory-related recommendation, trajectory classification, travel time estimation, anomaly detecti
    
[^3]: 为隐私保护可穿戴压力检测生成合成健康传感器数据

    Generating Synthetic Health Sensor Data for Privacy-Preserving Wearable Stress Detection. (arXiv:2401.13327v1 [cs.LG])

    [http://arxiv.org/abs/2401.13327](http://arxiv.org/abs/2401.13327)

    本论文介绍了一种隐私保护的合成健康传感器数据的方法，通过生成对抗网络（GANs）和差分隐私（DP）防护，生成与压力时刻相关的合成序列数据，确保患者信息的保护，并对合成数据进行质量评估。在压力检测数据集上验证了该方法的有效性。

    

    智能手表的健康传感器数据在智能健康应用和患者监测中越来越被使用，包括压力检测。然而，这类医疗数据往往包含敏感的个人信息，并且获取这些数据以进行研究是资源密集型的。为了应对这一挑战，我们介绍了一种关注隐私的合成多传感器智能手表健康读数与压力时刻相关的方法。我们的方法包括通过生成对抗网络（GANs）生成合成序列数据，并在模型训练过程中实施差分隐私（DP）防护以保护患者信息。为了确保合成数据的完整性，我们采用一系列质量评估，并监测合成数据与原始数据之间的合理性。为了测试其有用性，我们在一个常用但规模较小的压力检测数据集上创建了私有机器学习模型，并探索了增强现有数据基础的策略。

    Smartwatch health sensor data is increasingly utilized in smart health applications and patient monitoring, including stress detection. However, such medical data often comprises sensitive personal information and is resource-intensive to acquire for research purposes. In response to this challenge, we introduce the privacy-aware synthetization of multi-sensor smartwatch health readings related to moments of stress. Our method involves the generation of synthetic sequence data through Generative Adversarial Networks (GANs), coupled with the implementation of Differential Privacy (DP) safeguards for protecting patient information during model training. To ensure the integrity of our synthetic data, we employ a range of quality assessments and monitor the plausibility between synthetic and original data. To test the usefulness, we create private machine learning models on a commonly used, albeit small, stress detection dataset, exploring strategies for enhancing the existing data foundat
    
[^4]: T-COL: 为可变机器学习系统生成一般用户偏好的反事实解释

    T-COL: Generating Counterfactual Explanations for General User Preferences on Variable Machine Learning Systems. (arXiv:2309.16146v1 [cs.AI])

    [http://arxiv.org/abs/2309.16146](http://arxiv.org/abs/2309.16146)

    该论文提出了一个名为T-COL的方法，针对可变的机器学习系统和一般用户偏好生成反事实解释。这些解释不仅能够解释预测结果的原因，还提供了可操作的建议给用户。通过将一般用户偏好映射到CEs的属性上，以及采用定制化的方式来适应可变的机器学习模型，T-COL能够克服现有挑战并保持健壮性。

    

    基于机器学习的系统缺乏可解释性。为了解决这个问题，提出了反事实解释（CEs）。CEs独特之处在于它们不仅解释为什么会预测某个特定结果，还提供可操作的建议给用户。然而，CEs的应用受到了两个主要挑战的限制，即一般用户偏好和可变的机器学习系统。特别是，用户偏好往往是一般性的而不是特定的特征值。此外，CEs需要根据机器学习模型的可变性进行定制，并且在这些验证模型发生变化时仍然保持健壮性。为了克服这些挑战，我们提出了几个可能验证的一般用户偏好，并将它们映射到CEs的属性上。我们还引入了一种名为T-COL的新方法，它具有两种可选结构和几组协同操作。

    Machine learning (ML) based systems have been suffering a lack of interpretability. To address this problem, counterfactual explanations (CEs) have been proposed. CEs are unique as they provide workable suggestions to users, in addition to explaining why a certain outcome was predicted. However, the application of CEs has been hindered by two main challenges, namely general user preferences and variable ML systems. User preferences, in particular, tend to be general rather than specific feature values. Additionally, CEs need to be customized to suit the variability of ML models, while also maintaining robustness even when these validation models change. To overcome these challenges, we propose several possible general user preferences that have been validated by user research and map them to the properties of CEs. We also introduce a new method called \uline{T}ree-based \uline{C}onditions \uline{O}ptional \uline{L}inks (T-COL), which has two optional structures and several groups of co
    
[^5]: 实践中的隐私：X射线图像中的COVID-19检测的私有化（扩展版）

    Privacy in Practice: Private COVID-19 Detection in X-Ray Images (Extended Version). (arXiv:2211.11434v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.11434](http://arxiv.org/abs/2211.11434)

    该研究提出了通过差分隐私保护COVID-19检测模型，解决数据分析和患者隐私保护的问题。通过黑盒成员推理攻击，实现了对实际隐私的评估，结论表明所需的隐私等级可能因受到实际威胁的任务而异。

    

    机器学习（ML）可以通过使大量图像快速筛选来帮助抗击COVID-19等全球大流行病。为了在保护患者隐私的同时进行数据分析，我们创建了满足差分隐私（DP）要求的ML模型。以往探索私有COVID-19模型的研究在一定程度上基于小型数据集，提供较弱或不明确的隐私保证，并且没有研究实际隐私。我们提出改进措施以解决这些空缺。我们考虑天生的类别不平衡，并更广泛地评估效用-隐私权衡以及更严格的隐私预算。我们的评估得到黑盒成员推理攻击（MIAs）的实际隐私估计支持。引入的DP应有助于限制MIAs带来的泄漏威胁，而我们的实际分析是第一个在COVID-19分类任务上测试这个假设的。

    Machine learning (ML) can help fight pandemics like COVID-19 by enabling rapid screening of large volumes of images. To perform data analysis while maintaining patient privacy, we create ML models that satisfy Differential Privacy (DP). Previous works exploring private COVID-19 models are in part based on small datasets, provide weaker or unclear privacy guarantees, and do not investigate practical privacy. We suggest improvements to address these open gaps. We account for inherent class imbalances and evaluate the utility-privacy trade-off more extensively and over stricter privacy budgets. Our evaluation is supported by empirically estimating practical privacy through black-box Membership Inference Attacks (MIAs). The introduced DP should help limit leakage threats posed by MIAs, and our practical analysis is the first to test this hypothesis on the COVID-19 classification task. Our results indicate that needed privacy levels might differ based on the task-dependent practical threat 
    

