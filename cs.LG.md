# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Chebyshev Policies and the Mountain Car Problem: Reinforcement Learning for Low-Dimensional Control Tasks](https://arxiv.org/abs/2605.22305) | 本文通过解析求解山地车问题的最优控制，提出切比雪夫策略作为神经网络的轻量级替代，在低维控制任务中大幅降低参数量和遗憾值，并提升性能。 |
| [^2] | [A Distributionally Robust Optimisation Approach to Fair Credit Scoring](https://arxiv.org/abs/2402.01811) | 本文研究了如何在信用评分中应用分布鲁棒优化方法，并对现有技术的鲁棒性效果进行了实证评估。 |
| [^3] | [In-context learning for model-free system identification.](http://arxiv.org/abs/2308.13380) | 本文提出了一种基于上下文学习的无模型系统辨识方法，通过观察同一类别中其他系统的行为来理解动态系统的复杂性。 |
| [^4] | [Reinforcement Federated Learning Method Based on Adaptive OPTICS Clustering.](http://arxiv.org/abs/2306.12859) | 本文提出了一种基于自适应OPTICS聚类的强化联邦学习方法，旨在缓解不同用户终端上的数据分布不同所带来的负面影响，并有效提高了联邦学习方法的性能。 |
| [^5] | [EZClone: Improving DNN Model Extraction Attack via Shape Distillation from GPU Execution Profiles.](http://arxiv.org/abs/2304.03388) | 本论文介绍了两种不同威胁模型下的DNN结构提取技术，其中EZClone利用聚合GPU文件作为侧信道来预测DNN结构，并且通过实验验证了其有效性。 |

# 详细

[^1]: 切比雪夫策略与山地车问题：面向低维控制任务的强化学习

    Chebyshev Policies and the Mountain Car Problem: Reinforcement Learning for Low-Dimensional Control Tasks

    [https://arxiv.org/abs/2605.22305](https://arxiv.org/abs/2605.22305)

    本文通过解析求解山地车问题的最优控制，提出切比雪夫策略作为神经网络的轻量级替代，在低维控制任务中大幅降低参数量和遗憾值，并提升性能。

    

    我们解析求解了强化学习中的经典基准问题——山地车问题，并推导出最优控制解，填补了36年来的理论空白。这揭示了两个令人惊讶的发现：最优控制实际上非常简单，但现代强化学习智能体与最优解之间存在巨大差距。受最优控制分析的启发，我们从基本原理出发，提出切比雪夫策略作为一种通用（即密集）的强化学习策略类别。它们可以作为神经网络的直接替代品进行训练，将遗憾值降低4.18倍，同时所需参数减少277倍，从而提升样本效率、可解释性和实时处理能力。切比雪夫策略在更多强化学习任务上进行了评估，包括一个真实世界的非线性运动控制测试平台。在PPO、ARS和REINFORCE算法中，它们始终优于神经网络。我们的结果表明，切比雪夫策略提供了一种极具吸引力且轻量级的替代方案。

    arXiv:2605.22305v3 Announce Type: replace  Abstract: We analytically solve the Mountain Car problem, a canonical benchmark in RL, and derive an optimal control solution, closing a gap after 36 years. This enables us to reveal two surprising insights: The optimal control is quite simple, yet modern RL agents display a large gap to optimality. Motivated by the analysis of the optimal control, we introduce Chebyshev policies as a universal (i.e. dense) class of RL policies from first principles. They can be trained as drop-in replacements of neural nets, reducing the regret by a factor of 4.18, while requiring 277 times fewer parameters, fostering sample efficiency, explainability and realtime capability. Chebyshev policies are evaluated on further RL tasks, including a real-world nonlinear motion control testbed. They consistently improve performance over neural nets with PPO, ARS and REINFORCE. Our results demonstrate how Chebyshev policies offer a compelling and lightweight alternative
    
[^2]: 分布鲁棒优化方法在公平信用评分中的应用

    A Distributionally Robust Optimisation Approach to Fair Credit Scoring

    [https://arxiv.org/abs/2402.01811](https://arxiv.org/abs/2402.01811)

    本文研究了如何在信用评分中应用分布鲁棒优化方法，并对现有技术的鲁棒性效果进行了实证评估。

    

    信用评分被欧洲委员会和美国总统办公室归为高风险分类任务，关键问题是基于可能偏向某些群体的模型进行贷款批准决策可能造成的潜在风险。为解决这一问题，近期的信用评分研究考虑了机器学习领域提出的一系列增强公平性的技术来减少分类系统中的偏见和不公平对待。然而，尽管公平性的定义或实施方法各有不同，这些技术大多忽视了结果的鲁棒性。这可能导致在训练集中有效纠正不公平对待，但在生成样本外的分类时会再次产生不公平对待。因此，在本文中，我们将研究如何将分布鲁棒优化(DRO)方法应用于信用评分，并为此对现有技术的鲁棒性效果进行实证评估。

    Credit scoring has been catalogued by the European Commission and the Executive Office of the US President as a high-risk classification task, a key concern being the potential harms of making loan approval decisions based on models that would be biased against certain groups. To address this concern, recent credit scoring research has considered a range of fairness-enhancing techniques put forward by the machine learning community to reduce bias and unfair treatment in classification systems. While the definition of fairness or the approach they follow to impose it may vary, most of these techniques, however, disregard the robustness of the results. This can create situations where unfair treatment is effectively corrected in the training set, but when producing out-of-sample classifications, unfair treatment is incurred again. Instead, in this paper, we will investigate how to apply Distributionally Robust Optimisation (DRO) methods to credit scoring, thereby empirically evaluating h
    
[^3]: 基于上下文学习的无模型系统辨识

    In-context learning for model-free system identification. (arXiv:2308.13380v1 [eess.SY])

    [http://arxiv.org/abs/2308.13380](http://arxiv.org/abs/2308.13380)

    本文提出了一种基于上下文学习的无模型系统辨识方法，通过观察同一类别中其他系统的行为来理解动态系统的复杂性。

    

    在传统的系统辨识中，我们通过给定的输入/输出序列和可用的物理知识来估计未知动态系统的模型。然而，是否还可以通过观察同一类别中其他系统的行为，而不仅仅是从它们的输入/输出模式中理解动态系统的复杂性呢？这个核心问题驱动着本文的研究。作为对这个问题的回应，我们引入了一种新的系统辨识范式，解决了两个主要任务：一步预测和多步模拟。与传统方法不同的是，我们不直接对特定系统进行模型估计，而是预先训练一个代表动态系统类别的元模型。该元模型是通过从某个分布中随机抽取的系统生成的潜在无限流的合成数据进行训练的。在其核心，元模型作为对主要特征的隐式表示，

    In traditional system identification, we estimate a model of an unknown dynamical system based on given input/output sequences and available physical knowledge. Yet, is it also possible to understand the intricacies of dynamical systems not solely from their input/output patterns, but by observing the behavior of other systems within the same class? This central question drives the study presented in this paper.  In response to this query, we introduce a novel paradigm for system identification, addressing two primary tasks: one-step-ahead prediction and multi-step simulation. Unlike conventional methods, we do not directly estimate a model for the specific system. Instead, we pretrain a meta model that represents a class of dynamical systems. This meta model is trained from a potentially infinite stream of synthetic data, generated by systems randomly extracted from a certain distribution. At its core, the meta model serves as an implicit representation of the main characteristics of 
    
[^4]: 基于自适应OPTICS聚类的强化联邦学习方法

    Reinforcement Federated Learning Method Based on Adaptive OPTICS Clustering. (arXiv:2306.12859v1 [cs.LG])

    [http://arxiv.org/abs/2306.12859](http://arxiv.org/abs/2306.12859)

    本文提出了一种基于自适应OPTICS聚类的强化联邦学习方法，旨在缓解不同用户终端上的数据分布不同所带来的负面影响，并有效提高了联邦学习方法的性能。

    

    联邦学习是一种分布式机器学习技术，它实现了数据隐私保护和数据共享计算之间的平衡。为了保护数据隐私，联邦学习通过在参与设备上本地执行分布式训练并将本地模型聚合成全局模型来学习共享模型。联邦学习存在的问题是，由于数据在不同用户终端上的非独立和相同分布所导致的负面影响。为了缓解这个问题，本文提出了一种基于自适应OPTICS聚类的增强型联邦聚合方法。具体来说，该方法将聚类环境视为马尔科夫决策过程，并对参数搜索方向的调整过程进行建模，以找到最佳聚类参数以达到最佳联邦聚合方法。本文的核心贡献是提出了一种适用于联邦学习的自适应OPTICS聚类算法，可有效提高联邦学习方法的性能。

    Federated learning is a distributed machine learning technology, which realizes the balance between data privacy protection and data sharing computing. To protect data privacy, feder-ated learning learns shared models by locally executing distributed training on participating devices and aggregating local models into global models. There is a problem in federated learning, that is, the negative impact caused by the non-independent and identical distribu-tion of data across different user terminals. In order to alleviate this problem, this paper pro-poses a strengthened federation aggregation method based on adaptive OPTICS clustering. Specifically, this method perceives the clustering environment as a Markov decision process, and models the adjustment process of parameter search direction, so as to find the best clus-tering parameters to achieve the best federated aggregation method. The core contribution of this paper is to propose an adaptive OPTICS clustering algorithm for federated
    
[^5]: EZClone：通过GPU执行文件的形状精炼提高DNN模型提取攻击

    EZClone: Improving DNN Model Extraction Attack via Shape Distillation from GPU Execution Profiles. (arXiv:2304.03388v1 [cs.LG])

    [http://arxiv.org/abs/2304.03388](http://arxiv.org/abs/2304.03388)

    本论文介绍了两种不同威胁模型下的DNN结构提取技术，其中EZClone利用聚合GPU文件作为侧信道来预测DNN结构，并且通过实验验证了其有效性。

    

    由于在预测和分类问题上表现出色，深度神经网络（DNN）已经变得无处不在。然而，随着它们的使用扩展，它们面临各种威胁。模型提取攻击窃取DNN会危及知识产权、数据隐私和安全。先前的研究表明，系统级侧信道可用于通过暴露受害者DNN的体系结构来泄露模型的细节，从而加剧这些风险。我们提出了两种针对不同威胁模型的DNN结构提取技术。第一种技术使用恶意的、动态链接的PyTorch版本，在通过PyTorch分析器暴露受害者DNN结构。第二种技术称为EZClone，利用聚合（而不是时间序列）GPU文件作为侧信道来预测DNN结构，使用简单的方法，假设攻击者的能力比先前的研究低。我们在最小化攻击复杂性的情况下调查了EZClone的有效性，并在多种模型和数据集上进行了实验。

    Deep Neural Networks (DNNs) have become ubiquitous due to their performance on prediction and classification problems. However, they face a variety of threats as their usage spreads. Model extraction attacks, which steal DNNs, endanger intellectual property, data privacy, and security. Previous research has shown that system-level side-channels can be used to leak the architecture of a victim DNN, exacerbating these risks. We propose two DNN architecture extraction techniques catering to various threat models. The first technique uses a malicious, dynamically linked version of PyTorch to expose a victim DNN architecture through the PyTorch profiler. The second, called EZClone, exploits aggregate (rather than time-series) GPU profiles as a side-channel to predict DNN architecture, employing a simple approach and assuming little adversary capability as compared to previous work. We investigate the effectiveness of EZClone when minimizing the complexity of the attack, when applied to prun
    

