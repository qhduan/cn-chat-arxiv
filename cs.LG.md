# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance](https://arxiv.org/abs/2403.16952) | 该研究发现了数据混合规律，可以量化地预测模型性能与数据混合比例之间的关系，并提出了一种方法来通过拟合函数形式来引导理想的数据混合选择，从而优化大型语言模型的训练混合。 |
| [^2] | [A Scalable and Parallelizable Digital Twin Framework for Sustainable Sim2Real Transition of Multi-Agent Reinforcement Learning Systems](https://arxiv.org/abs/2403.10996) | 提出了一个可持续的多智能体深度强化学习框架，利用分散的学习架构，来解决交通路口穿越和自主赛车等问题 |
| [^3] | [Intelligent Agricultural Greenhouse Control System Based on Internet of Things and Machine Learning](https://arxiv.org/abs/2402.09488) | 这项研究提出了一种基于物联网和机器学习的智能农业温室控制系统，通过监测和调控温室内环境条件，提高作物生长效率和产量，减少资源浪费。 |
| [^4] | [GenEFT: Understanding Statics and Dynamics of Model Generalization via Effective Theory](https://arxiv.org/abs/2402.05916) | GenEFT是一个有效的理论框架，通过研究泛化相变和表示学习动态，揭示了神经网络泛化的静态和动态特性，这弥合了机器学习理论预测与实践之间的差距。 |
| [^5] | [Crowd-PrefRL: Preference-Based Reward Learning from Crowds.](http://arxiv.org/abs/2401.10941) | Crowd-PrefRL是一种基于众包的偏好反馈学习框架，能够从来自群体的反馈中学习奖励函数，并且能够强大地聚合群体偏好反馈并估计用户的可靠性。 |
| [^6] | [CLAN: A Contrastive Learning based Novelty Detection Framework for Human Activity Recognition.](http://arxiv.org/abs/2401.10288) | CLAN是一种基于对比学习的新颖性检测框架，用于处理人体活动识别中的挑战，并构建对挑战具有不变性的已知活动的表示方法。 |
| [^7] | [Interpreting the Curse of Dimensionality from Distance Concentration and Manifold Effect.](http://arxiv.org/abs/2401.00422) | 这篇论文从理论和实证分析的角度深入研究了维度诅咒的两个主要原因——距离集中和流形效应，并通过实验证明了使用Minkowski距离进行最近邻搜索（NNS）在高维数据中取得了最佳性能。 |
| [^8] | [Graphical Object-Centric Actor-Critic.](http://arxiv.org/abs/2310.17178) | 这项研究提出了一种新颖的以对象为中心的强化学习算法，将演员-评论家和基于模型的方法结合起来，利用解耦的对象表示有效地学习策略。该方法填补了以对象为中心的强化学习环境中高效且适用于离散或连续动作空间的世界模型的研究空白。 |
| [^9] | [Distributionally Robust Machine Learning with Multi-source Data.](http://arxiv.org/abs/2309.02211) | 本文提出了一种基于多源数据的分布鲁棒机器学习方法，通过引入组分布鲁棒预测模型来提高具有分布偏移的目标人群的预测准确性。 |
| [^10] | [BELLA: Black box model Explanations by Local Linear Approximations.](http://arxiv.org/abs/2305.11311) | 本文提出了一种确定性的、与模型无关的事后方法BELLA，用于解释回归黑盒模型的个别预测。该方法通过特征空间中训练的线性模型提供解释，使得该模型的系数可以直接用于计算特征值的预测值。此外，BELLA最大化了线性模型适用的领域范围。 |
| [^11] | [Human Choice Prediction in Non-Cooperative Games: Simulation-based Off-Policy Evaluation.](http://arxiv.org/abs/2305.10361) | 本文研究了语言游戏中的离线策略评估，并提出了一种结合真实和模拟数据的新方法。 |
| [^12] | [A Graph Neural Network Approach to Nanosatellite Task Scheduling: Insights into Learning Mixed-Integer Models.](http://arxiv.org/abs/2303.13773) | 本研究提出基于GNN的纳米卫星任务调度方法，以更好地优化服务质量，解决ONTS问题的复杂性。 |
| [^13] | [Karyotype AI for Precision Oncology.](http://arxiv.org/abs/2211.14312) | 本研究针对精准肿瘤学中的染色体分析问题，通过使用Fred Hutchinson癌症研究中心的大量数据，利用深度学习模型和拓扑视觉转换器(TopViTs)，成功开发出了一种自动识别染色体异常的方法。 |
| [^14] | [A policy gradient approach for Finite Horizon Constrained Markov Decision Processes.](http://arxiv.org/abs/2210.04527) | 本文提出了一种针对有限时域受限马尔可夫决策过程的策略梯度方法，该方法能够在固定时间后终止，通过函数逼近和策略梯度方法找到最优策略。 |
| [^15] | [Stochastic coordinate transformations with applications to robust machine learning.](http://arxiv.org/abs/2110.01729) | 本文提出了一种利用随机坐标变换进行异常检测的新方法，该方法通过层级张量积展开来逼近随机过程，并通过训练机器学习分类器对投影系数进行检测。在基准数据集上的实验表明，该方法胜过现有的最先进方法。 |

# 详细

[^1]: 数据混合规律：通过预测语言建模性能来优化数据混合

    Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance

    [https://arxiv.org/abs/2403.16952](https://arxiv.org/abs/2403.16952)

    该研究发现了数据混合规律，可以量化地预测模型性能与数据混合比例之间的关系，并提出了一种方法来通过拟合函数形式来引导理想的数据混合选择，从而优化大型语言模型的训练混合。

    

    大型语言模型的预训练数据包括多个领域（例如网络文本、学术论文、代码），其混合比例对结果模型的能力至关重要。现有的工作通常依赖于启发式方法或定性策略来调整比例，我们发现了模型性能与混合比例之间的函数形式的定量可预测性，我们称之为数据混合规律。在样本混合上拟合这种函数揭示了未见混合的模型性能，从而引导选择理想的数据混合。此外，我们提出了训练步骤、模型大小和我们的数据混合规律的缩放规律的嵌套使用，以使得仅通过小规模训练就能够预测在各种混合数据下训练的大模型的性能。此外，实验结果验证了我们的方法有效地优化了训练混合。

    arXiv:2403.16952v1 Announce Type: cross  Abstract: Pretraining data of large language models composes multiple domains (e.g., web texts, academic papers, codes), whose mixture proportions crucially impact the competence of outcome models. While existing endeavors rely on heuristics or qualitative strategies to tune the proportions, we discover the quantitative predictability of model performance regarding the mixture proportions in function forms, which we refer to as the data mixing laws. Fitting such functions on sample mixtures unveils model performance on unseen mixtures before actual runs, thus guiding the selection of an ideal data mixture. Furthermore, we propose nested use of the scaling laws of training steps, model sizes, and our data mixing law to enable predicting the performance of large models trained on massive data under various mixtures with only small-scale training. Moreover, experimental results verify that our method effectively optimizes the training mixture of a 
    
[^2]: 一个可扩展且可并行化的数字孪生框架，用于多智能体强化学习系统可持续Sim2Real转换

    A Scalable and Parallelizable Digital Twin Framework for Sustainable Sim2Real Transition of Multi-Agent Reinforcement Learning Systems

    [https://arxiv.org/abs/2403.10996](https://arxiv.org/abs/2403.10996)

    提出了一个可持续的多智能体深度强化学习框架，利用分散的学习架构，来解决交通路口穿越和自主赛车等问题

    

    本工作提出了一个可持续的多智能体深度强化学习框架，能够选择性地按需扩展并行化训练工作负载，并利用最少的硬件资源将训练好的策略从模拟环境转移到现实世界。我们引入了AutoDRIVE生态系统作为一个启动数字孪生框架，用于训练、部署和转移合作和竞争的多智能体强化学习策略从模拟环境到现实世界。具体来说，我们首先探究了4台合作车辆(Nigel)在单智能体和多智能体学习环境中共享有限状态信息的交叉遍历问题，采用了一种通用策略方法。然后，我们使用个体策略方法研究了2辆车(F1TENTH)的对抗性自主赛车问题。在任何一组实验中，我们采用了去中心化学习架构，这允许对策略进行有力的训练和测试。

    arXiv:2403.10996v1 Announce Type: cross  Abstract: This work presents a sustainable multi-agent deep reinforcement learning framework capable of selectively scaling parallelized training workloads on-demand, and transferring the trained policies from simulation to reality using minimal hardware resources. We introduce AutoDRIVE Ecosystem as an enabling digital twin framework to train, deploy, and transfer cooperative as well as competitive multi-agent reinforcement learning policies from simulation to reality. Particularly, we first investigate an intersection traversal problem of 4 cooperative vehicles (Nigel) that share limited state information in single as well as multi-agent learning settings using a common policy approach. We then investigate an adversarial autonomous racing problem of 2 vehicles (F1TENTH) using an individual policy approach. In either set of experiments, a decentralized learning architecture was adopted, which allowed robust training and testing of the policies 
    
[^3]: 基于物联网和机器学习的智能农业温室控制系统

    Intelligent Agricultural Greenhouse Control System Based on Internet of Things and Machine Learning

    [https://arxiv.org/abs/2402.09488](https://arxiv.org/abs/2402.09488)

    这项研究提出了一种基于物联网和机器学习的智能农业温室控制系统，通过监测和调控温室内环境条件，提高作物生长效率和产量，减少资源浪费。

    

    本研究试图将物联网和机器学习相结合，构建一个先进的农业温室控制系统。通过对温室内固有环境参数的细致监测和机器学习算法的整合，能够适当调控温室内的条件。预期的结果是增加作物生长效率和产量，同时减少资源浪费。在全球人口持续增长和气候变化不断加剧的背景下，农业面临前所未有的挑战。传统农业范式已经被证明无法满足食品安全和生产效率的要求。在这种背景下，温室农业成为一种可行的解决方案，为作物种植提供了一个受控的环境来增加产量，改善品质。

    arXiv:2402.09488v1 Announce Type: cross  Abstract: This study endeavors to conceptualize and execute a sophisticated agricultural greenhouse control system grounded in the amalgamation of the Internet of Things (IoT) and machine learning. Through meticulous monitoring of intrinsic environmental parameters within the greenhouse and the integration of machine learning algorithms, the conditions within the greenhouse are aptly modulated. The envisaged outcome is an enhancement in crop growth efficiency and yield, accompanied by a reduction in resource wastage. In the backdrop of escalating global population figures and the escalating exigencies of climate change, agriculture confronts unprecedented challenges. Conventional agricultural paradigms have proven inadequate in addressing the imperatives of food safety and production efficiency. Against this backdrop, greenhouse agriculture emerges as a viable solution, proffering a controlled milieu for crop cultivation to augment yields, refin
    
[^4]: GenEFT: 通过有效理论理解模型泛化的静态和动态

    GenEFT: Understanding Statics and Dynamics of Model Generalization via Effective Theory

    [https://arxiv.org/abs/2402.05916](https://arxiv.org/abs/2402.05916)

    GenEFT是一个有效的理论框架，通过研究泛化相变和表示学习动态，揭示了神经网络泛化的静态和动态特性，这弥合了机器学习理论预测与实践之间的差距。

    

    我们提出了GenEFT：一个有效的理论框架，用于揭示神经网络泛化的静态和动态，以图学习为例进行了说明。首先，我们研究了数据规模增加时的泛化相变，将实验结果与基于信息理论的近似进行比较。我们发现，在解码器既不太弱也不太强的“小熊宝贝区域”中存在着泛化。然后，我们介绍了一种表示学习动态的有效理论，将潜在空间表示建模为相互作用粒子（repons），发现它解释了我们在编码器和解码器学习速率扫描时观察到的泛化和过拟合之间的相变。这突出了受物理启发的有效理论在弥合机器学习中理论预测与实践之间的差距方面的力量。

    We present GenEFT: an effective theory framework for shedding light on the statics and dynamics of neural network generalization, and illustrate it with graph learning examples. We first investigate the generalization phase transition as data size increases, comparing experimental results with information-theory-based approximations. We find generalization in a Goldilocks zone where the decoder is neither too weak nor too powerful. We then introduce an effective theory for the dynamics of representation learning, where latent-space representations are modeled as interacting particles (repons), and find that it explains our experimentally observed phase transition between generalization and overfitting as encoder and decoder learning rates are scanned. This highlights the power of physics-inspired effective theories for bridging the gap between theoretical predictions and practice in machine learning.
    
[^5]: Crowd-PrefRL: 基于众包的偏好反馈学习

    Crowd-PrefRL: Preference-Based Reward Learning from Crowds. (arXiv:2401.10941v1 [cs.HC])

    [http://arxiv.org/abs/2401.10941](http://arxiv.org/abs/2401.10941)

    Crowd-PrefRL是一种基于众包的偏好反馈学习框架，能够从来自群体的反馈中学习奖励函数，并且能够强大地聚合群体偏好反馈并估计用户的可靠性。

    

    基于偏好的强化学习提供了一个框架，通过对行为对的偏好进行人类反馈来训练智能体，使其能够在难以指定数值奖励函数的情况下学习期望的行为。尽管这个范式利用了人类的反馈，但目前将反馈视为单个人类用户所给出的。与此同时，以强大的方式合并来自群体（即用户集合）的偏好反馈仍然是一个挑战，而使用来自多个用户的反馈来训练强化学习智能体的问题仍然被研究不足。在这项工作中，我们引入了Crowd-PrefRL，一个利用来自群体的反馈进行基于偏好的强化学习的框架。这项工作展示了利用未知专业水平和可靠性的群体偏好反馈来学习奖励函数的可行性。Crowd-PrefRL不仅能够强大地聚合群体偏好反馈，还能够估计每个用户的可靠性。

    Preference-based reinforcement learning (RL) provides a framework to train agents using human feedback through pairwise preferences over pairs of behaviors, enabling agents to learn desired behaviors when it is difficult to specify a numerical reward function. While this paradigm leverages human feedback, it currently treats the feedback as given by a single human user. Meanwhile, incorporating preference feedback from crowds (i.e. ensembles of users) in a robust manner remains a challenge, and the problem of training RL agents using feedback from multiple human users remains understudied. In this work, we introduce Crowd-PrefRL, a framework for performing preference-based RL leveraging feedback from crowds. This work demonstrates the viability of learning reward functions from preference feedback provided by crowds of unknown expertise and reliability. Crowd-PrefRL not only robustly aggregates the crowd preference feedback, but also estimates the reliability of each user within the cr
    
[^6]: CLAN:基于对比学习的用于人体活动识别的新颖性检测框架

    CLAN: A Contrastive Learning based Novelty Detection Framework for Human Activity Recognition. (arXiv:2401.10288v1 [cs.LG])

    [http://arxiv.org/abs/2401.10288](http://arxiv.org/abs/2401.10288)

    CLAN是一种基于对比学习的新颖性检测框架，用于处理人体活动识别中的挑战，并构建对挑战具有不变性的已知活动的表示方法。

    

    在环境辅助生活中，从时间序列传感器数据进行人体活动识别主要集中于预定义的活动，往往忽略了新的活动模式。我们提出了CLAN，一种基于对比学习的新颖性检测框架，其中包含了不同类型的负样本对于人体活动识别。该框架针对人体活动特征的挑战进行了优化，包括时间和频率特征的重要性、复杂的活动动态、活动之间共享的特征，以及传感器模态的变化。该框架旨在构建对挑战具有不变性的已知活动的表示方法。为了生成合适的负样本对，它根据每个数据集的时间和频率特征选择数据增强方法。它通过对比和分类损失的表示学习以及基于评分函数的新颖性检测，从中导出针对无意义动态的关键表示。

    In ambient assisted living, human activity recognition from time series sensor data mainly focuses on predefined activities, often overlooking new activity patterns. We propose CLAN, a two-tower contrastive learning-based novelty detection framework with diverse types of negative pairs for human activity recognition. It is tailored to challenges with human activity characteristics, including the significance of temporal and frequency features, complex activity dynamics, shared features across activities, and sensor modality variations. The framework aims to construct invariant representations of known activity robust to the challenges. To generate suitable negative pairs, it selects data augmentation methods according to the temporal and frequency characteristics of each dataset. It derives the key representations against meaningless dynamics by contrastive and classification losses-based representation learning and score function-based novelty detection that accommodate dynamic number
    
[^7]: 从距离集中和流形效应解读维度诅咒

    Interpreting the Curse of Dimensionality from Distance Concentration and Manifold Effect. (arXiv:2401.00422v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2401.00422](http://arxiv.org/abs/2401.00422)

    这篇论文从理论和实证分析的角度深入研究了维度诅咒的两个主要原因——距离集中和流形效应，并通过实验证明了使用Minkowski距离进行最近邻搜索（NNS）在高维数据中取得了最佳性能。

    

    随着维度的增加，数据的特征如分布和异质性变得越来越复杂和违反直觉。这种现象被称为维度诅咒，低维空间中成立的常见模式和关系（例如内部和边界模式）在高维空间中可能无效。这导致回归、分类或聚类模型或算法的性能降低。维度诅咒可以归因于许多原因。本文首先总结了与处理高维数据相关的五个挑战，并解释了回归、分类或聚类任务失败的潜在原因。随后，我们通过理论和实证分析深入研究了维度诅咒的两个主要原因，即距离集中和流形效应。结果表明，使用三种典型的距离测量进行最近邻搜索（NNS）时，Minkowski距离的性能最佳。

    The characteristics of data like distribution and heterogeneity, become more complex and counterintuitive as the dimensionality increases. This phenomenon is known as curse of dimensionality, where common patterns and relationships (e.g., internal and boundary pattern) that hold in low-dimensional space may be invalid in higher-dimensional space. It leads to a decreasing performance for the regression, classification or clustering models or algorithms. Curse of dimensionality can be attributed to many causes. In this paper, we first summarize five challenges associated with manipulating high-dimensional data, and explains the potential causes for the failure of regression, classification or clustering tasks. Subsequently, we delve into two major causes of the curse of dimensionality, distance concentration and manifold effect, by performing theoretical and empirical analyses. The results demonstrate that nearest neighbor search (NNS) using three typical distance measurements, Minkowski
    
[^8]: 图形化的以对象为中心的Actor-Critic算法

    Graphical Object-Centric Actor-Critic. (arXiv:2310.17178v1 [cs.AI])

    [http://arxiv.org/abs/2310.17178](http://arxiv.org/abs/2310.17178)

    这项研究提出了一种新颖的以对象为中心的强化学习算法，将演员-评论家和基于模型的方法结合起来，利用解耦的对象表示有效地学习策略。该方法填补了以对象为中心的强化学习环境中高效且适用于离散或连续动作空间的世界模型的研究空白。

    

    最近在无监督的以对象为中心的表示学习及其在下游任务中的应用方面取得了重要进展。最新的研究支持这样一个观点，即在基于图像的以对象为中心的强化学习任务中采用解耦的对象表示能够促进策略学习。我们提出了一种新颖的以对象为中心的强化学习算法，将演员-评论家算法和基于模型的方法结合起来，以有效利用这些表示。在我们的方法中，我们使用一个变换器编码器来提取对象表示，并使用图神经网络来近似环境的动力学。所提出的方法填补了开发强化学习环境中可以用于离散或连续动作空间的高效以对象为中心的世界模型的研究空白。我们的算法在一个具有复杂视觉3D机器人环境和一个具有组合结构的2D环境中表现更好。

    There have recently been significant advances in the problem of unsupervised object-centric representation learning and its application to downstream tasks. The latest works support the argument that employing disentangled object representations in image-based object-centric reinforcement learning tasks facilitates policy learning. We propose a novel object-centric reinforcement learning algorithm combining actor-critic and model-based approaches to utilize these representations effectively. In our approach, we use a transformer encoder to extract object representations and graph neural networks to approximate the dynamics of an environment. The proposed method fills a research gap in developing efficient object-centric world models for reinforcement learning settings that can be used for environments with discrete or continuous action spaces. Our algorithm performs better in a visually complex 3D robotic environment and a 2D environment with compositional structure than the state-of-t
    
[^9]: 基于多源数据的分布鲁棒机器学习

    Distributionally Robust Machine Learning with Multi-source Data. (arXiv:2309.02211v1 [stat.ML])

    [http://arxiv.org/abs/2309.02211](http://arxiv.org/abs/2309.02211)

    本文提出了一种基于多源数据的分布鲁棒机器学习方法，通过引入组分布鲁棒预测模型来提高具有分布偏移的目标人群的预测准确性。

    

    当目标分布与源数据集不同时，传统的机器学习方法可能导致较差的预测性能。本文利用多个数据源，并引入了一种基于组分布鲁棒预测模型来优化关于目标分布类的可解释方差的对抗性奖励。与传统的经验风险最小化相比，所提出的鲁棒预测模型改善了具有分布偏移的目标人群的预测准确性。我们证明了组分布鲁棒预测模型是源数据集条件结果模型的加权平均。我们利用这一关键鉴别结果来提高任意机器学习算法的鲁棒性，包括随机森林和神经网络等。我们设计了一种新的偏差校正估计器来估计通用机器学习算法的最优聚合权重，并展示了其在c方面的改进。

    Classical machine learning methods may lead to poor prediction performance when the target distribution differs from the source populations. This paper utilizes data from multiple sources and introduces a group distributionally robust prediction model defined to optimize an adversarial reward about explained variance with respect to a class of target distributions. Compared to classical empirical risk minimization, the proposed robust prediction model improves the prediction accuracy for target populations with distribution shifts. We show that our group distributionally robust prediction model is a weighted average of the source populations' conditional outcome models. We leverage this key identification result to robustify arbitrary machine learning algorithms, including, for example, random forests and neural networks. We devise a novel bias-corrected estimator to estimate the optimal aggregation weight for general machine-learning algorithms and demonstrate its improvement in the c
    
[^10]: BELLA: 通过本地线性逼近进行黑盒模型解释

    BELLA: Black box model Explanations by Local Linear Approximations. (arXiv:2305.11311v1 [cs.LG])

    [http://arxiv.org/abs/2305.11311](http://arxiv.org/abs/2305.11311)

    本文提出了一种确定性的、与模型无关的事后方法BELLA，用于解释回归黑盒模型的个别预测。该方法通过特征空间中训练的线性模型提供解释，使得该模型的系数可以直接用于计算特征值的预测值。此外，BELLA最大化了线性模型适用的领域范围。

    

    近年来，理解黑盒模型的决策过程不仅成为法律要求，也成为评估其性能的另一种方式。然而，现有的事后解释方法依赖于合成数据生成，这引入了不确定性并可能损害解释的可靠性，并且它们 tend to produce explanations that apply to only very few data points. This makes the explanations brittle and limited in scope. Finally, they provide scores that have no direct verifiable meaning. In this paper, we present BELLA, a deterministic model-agnostic post-hoc approach for explaining the individual predictions of regression black-box models. BELLA provides explanations in the form of a linear model trained in the feature space. Thus, its coefficients can be used directly to compute the predicted value from the feature values. Furthermore, BELLA maximizes the size of the neighborhood to which the linear model a

    In recent years, understanding the decision-making process of black-box models has become not only a legal requirement but also an additional way to assess their performance. However, the state of the art post-hoc interpretation approaches rely on synthetic data generation. This introduces uncertainty and can hurt the reliability of the interpretations. Furthermore, they tend to produce explanations that apply to only very few data points. This makes the explanations brittle and limited in scope. Finally, they provide scores that have no direct verifiable meaning. In this paper, we present BELLA, a deterministic model-agnostic post-hoc approach for explaining the individual predictions of regression black-box models. BELLA provides explanations in the form of a linear model trained in the feature space. Thus, its coefficients can be used directly to compute the predicted value from the feature values. Furthermore, BELLA maximizes the size of the neighborhood to which the linear model a
    
[^11]: 非合作博弈中的人类选择预测：基于模拟的离线策略评估

    Human Choice Prediction in Non-Cooperative Games: Simulation-based Off-Policy Evaluation. (arXiv:2305.10361v1 [cs.LG])

    [http://arxiv.org/abs/2305.10361](http://arxiv.org/abs/2305.10361)

    本文研究了语言游戏中的离线策略评估，并提出了一种结合真实和模拟数据的新方法。

    

    说服游戏在经济和人工智能研究中具有重要意义并具有重要的实际应用。本文探讨了在基于语言的说服游戏中离线策略评估（OPE）的挑战性问题，提出了一种结合真实和模拟人类 - 机器人交互数据的新方法，并给出了一种深度学习训练算法，该算法有效地整合了真实交互和模拟数据。

    Persuasion games have been fundamental in economics and AI research, and have significant practical applications. Recent works in this area have started to incorporate natural language, moving beyond the traditional stylized message setting. However, previous research has focused on on-policy prediction, where the train and test data have the same distribution, which is not representative of real-life scenarios. In this paper, we tackle the challenging problem of off-policy evaluation (OPE) in language-based persuasion games. To address the inherent difficulty of human data collection in this setup, we propose a novel approach which combines real and simulated human-bot interaction data. Our simulated data is created by an exogenous model assuming decision makers (DMs) start with a mixture of random and decision-theoretic based behaviors and improve over time. We present a deep learning training algorithm that effectively integrates real interaction and simulated data, substantially im
    
[^12]: 基于图神经网络的纳米卫星任务调度方法：学习混合整数模型的洞见

    A Graph Neural Network Approach to Nanosatellite Task Scheduling: Insights into Learning Mixed-Integer Models. (arXiv:2303.13773v1 [cs.LG])

    [http://arxiv.org/abs/2303.13773](http://arxiv.org/abs/2303.13773)

    本研究提出基于GNN的纳米卫星任务调度方法，以更好地优化服务质量，解决ONTS问题的复杂性。

    

    本研究探讨如何利用图神经网络（GNN）更有效地调度纳米卫星任务。在离线纳米卫星任务调度（ONTS）问题中，目标是找到在轨道上执行任务的最佳安排，同时考虑服务质量（QoS）方面的考虑因素，如优先级，最小和最大激活事件，执行时间框架，周期和执行窗口，以及卫星电力资源和能量收集和管理的复杂性的约束。ONTS问题已经使用传统的数学公式和精确方法进行了处理，但是它们在问题的挑战性案例中的适用性有限。本研究考察了在这种情况下使用GNN的方法，该方法已经成功应用于许多优化问题，包括旅行商问题，调度问题和设施放置问题。在本文中，我们将ONTS问题的MILP实例完全表示成二分图网络结构来应用GNN。

    This study investigates how to schedule nanosatellite tasks more efficiently using Graph Neural Networks (GNN). In the Offline Nanosatellite Task Scheduling (ONTS) problem, the goal is to find the optimal schedule for tasks to be carried out in orbit while taking into account Quality-of-Service (QoS) considerations such as priority, minimum and maximum activation events, execution time-frames, periods, and execution windows, as well as constraints on the satellite's power resources and the complexity of energy harvesting and management. The ONTS problem has been approached using conventional mathematical formulations and precise methods, but their applicability to challenging cases of the problem is limited. This study examines the use of GNNs in this context, which has been effectively applied to many optimization problems, including traveling salesman problems, scheduling problems, and facility placement problems. Here, we fully represent MILP instances of the ONTS problem in biparti
    
[^13]: 精准肿瘤学的染色体AI

    Karyotype AI for Precision Oncology. (arXiv:2211.14312v3 [q-bio.QM] UPDATED)

    [http://arxiv.org/abs/2211.14312](http://arxiv.org/abs/2211.14312)

    本研究针对精准肿瘤学中的染色体分析问题，通过使用Fred Hutchinson癌症研究中心的大量数据，利用深度学习模型和拓扑视觉转换器(TopViTs)，成功开发出了一种自动识别染色体异常的方法。

    

    染色体分析对于诊断遗传疾病至关重要。对于血液系统恶性肿瘤，通过染色体组型分析来发现体细胞突变是标准的护理方法。然而，染色体组型分析因为大部分是手动操作，且需要专业知识来识别和注释突变，所以昂贵且耗时。以Fred Hutchinson癌症研究中心过去五年的约10,000个患者标本和约50,000个染色体组型图片作为训练集，我们创建了一组代表单个染色体的标记图片。这些单个染色体用于训练和评估深度学习模型，以分类人类的24条染色体和识别染色体异常。具有最高准确性的模型使用了最近引入的拓扑视觉转换器(TopViTs)和二级块-托普利茨蒙版，以融入结构性归纳偏置。TopViT的性能优于CNN(Inc)

    Chromosome analysis is essential for diagnosing genetic disorders. For hematologic malignancies, identification of somatic clonal aberrations by karyotype analysis remains the standard of care. However, karyotyping is costly and time-consuming because of the largely manual process and the expertise required in identifying and annotating aberrations. Efforts to automate karyotype analysis to date fell short in aberration detection. Using a training set of ~10k patient specimens and ~50k karyograms from over 5 years from the Fred Hutchinson Cancer Center, we created a labeled set of images representing individual chromosomes. These individual chromosomes were used to train and assess deep learning models for classifying the 24 human chromosomes and identifying chromosomal aberrations. The top-accuracy models utilized the recently introduced Topological Vision Transformers (TopViTs) with 2-level-block-Toeplitz masking, to incorporate structural inductive bias. TopViT outperformed CNN (Inc
    
[^14]: 有限时域受限马尔可夫决策过程的策略梯度方法

    A policy gradient approach for Finite Horizon Constrained Markov Decision Processes. (arXiv:2210.04527v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.04527](http://arxiv.org/abs/2210.04527)

    本文提出了一种针对有限时域受限马尔可夫决策过程的策略梯度方法，该方法能够在固定时间后终止，通过函数逼近和策略梯度方法找到最优策略。

    

    无限时域设置通常用于强化学习问题，导致产生最优的固定策略。然而，在许多情况下，有限时域控制问题更具有实际意义，并且在这种情况下，最优策略通常随时间变化。最近，约束强化学习的设置也越来越受到关注，其中代理同时在最大化奖励的同时满足某些给定的约束条件。然而，这个设置仅在无限时域马尔可夫决策过程的背景下得到了研究，其中固定策略是最优的。本文提出了一种在有限时域设置下进行约束强化学习的算法，其中在一个固定的时间后终止。我们在算法中使用函数逼近，这在状态和动作空间较大或连续的情况下是必不可少的，并使用策略梯度方法来找到最优策略。我们得到的最优策略取决于时间段。

    The infinite horizon setting is widely adopted for problems of reinforcement learning (RL). These invariably result in stationary policies that are optimal. In many situations, finite horizon control problems are of interest and for such problems, the optimal policies are time-varying in general. Another setting that has become popular in recent times is of Constrained Reinforcement Learning, where the agent maximizes its rewards while it also aims to satisfy some given constraint criteria. However, this setting has only been studied in the context of infinite horizon MDPs where stationary policies are optimal. We present an algorithm for constrained RL in the Finite Horizon Setting where the horizon terminates after a fixed (finite) time. We use function approximation in our algorithm which is essential when the state and action spaces are large or continuous and use the policy gradient method to find the optimal policy. The optimal policy that we obtain depends on the stage and so is
    
[^15]: 随机坐标变换及其在鲁棒机器学习中的应用

    Stochastic coordinate transformations with applications to robust machine learning. (arXiv:2110.01729v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2110.01729](http://arxiv.org/abs/2110.01729)

    本文提出了一种利用随机坐标变换进行异常检测的新方法，该方法通过层级张量积展开来逼近随机过程，并通过训练机器学习分类器对投影系数进行检测。在基准数据集上的实验表明，该方法胜过现有的最先进方法。

    

    本文介绍了一组新的特征，利用Karhunen-Loeve展开法来识别输入数据的潜在随机行为。这些新特征是通过基于最近的函数数据分析理论进行的坐标变换构建的，用于异常检测。相关的信号分解是用已知优化属性的层级张量积展开来逼近具有有限功能空间的随机过程（随机场）。原则上，这些低维空间可以捕捉给定名义类别的'底层信号'的大部分随机变化，并且可以将来自其它类别的信号拒绝为随机异常。通过名义类别的层级有限维展开，构建了一系列用于检测异常信号组件的正交嵌套子空间。然后使用这些子空间中的投影系数来训练用于异常检测的机器学习（ML）分类器。我们在几个基准数据集上评估所提出的方法，结果表明其胜过现有的最先进方法。

    In this paper we introduce a set of novel features for identifying underlying stochastic behavior of input data using the Karhunen-Loeve expansion. These novel features are constructed by applying a coordinate transformation based on the recent Functional Data Analysis theory for anomaly detection. The associated signal decomposition is an exact hierarchical tensor product expansion with known optimality properties for approximating stochastic processes (random fields) with finite dimensional function spaces. In principle these low dimensional spaces can capture most of the stochastic behavior of `underlying signals' in a given nominal class, and can reject signals in alternative classes as stochastic anomalies. Using a hierarchical finite dimensional expansion of the nominal class, a series of orthogonal nested subspaces is constructed for detecting anomalous signal components. Projection coefficients of input data in these subspaces are then used to train a Machine Learning (ML) clas
    

