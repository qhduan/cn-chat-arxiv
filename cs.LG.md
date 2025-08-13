# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Whispers in the Machine: Confidentiality in LLM-integrated Systems](https://arxiv.org/abs/2402.06922) | 本研究提供了一种评估LLM集成系统保密性的系统化方法，通过形式化一个"秘密密钥"游戏来捕捉模型隐藏私人信息的能力。评估了八种攻击和四种防御方法，发现当前的防御方法缺乏泛化性能。 |
| [^2] | [Combat Urban Congestion via Collaboration: Heterogeneous GNN-based MARL for Coordinated Platooning and Traffic Signal Control.](http://arxiv.org/abs/2310.10948) | 本文提出了一种基于异构图多智能体强化学习和交通理论的创新解决方案，通过将车辆编队和交通信号控制作为不同的强化学习智能体，并结合图神经网络实现协调，以优化交通流量和缓解城市拥堵。 |
| [^3] | [Learning Generative Models for Climbing Aircraft from Radar Data.](http://arxiv.org/abs/2309.14941) | 本文提出了一种利用雷达数据学习的生成模型，能够准确预测攀升飞机的轨迹，并通过学习修正推力的函数来提高预测准确性。该方法的优势包括：与标准模型相比，到达时间的预测误差减少了66.3%；生成的轨迹与测试数据相比更加真实；并且能够以最小的计算成本计算置信区间。 |
| [^4] | [Neural Operator Variational Inference based on Regularized Stein Discrepancy for Deep Gaussian Processes.](http://arxiv.org/abs/2309.12658) | 基于正则化Stein差异的神经算子变分推断用于深度高斯过程，通过使用神经生成器获得取样器以及使用蒙特卡罗估计和子采样随机优化技术解决极小极大问题，提高了深度高斯过程模型的表达能力和推断效果。 |
| [^5] | [BELLA: Black box model Explanations by Local Linear Approximations.](http://arxiv.org/abs/2305.11311) | 本文提出了一种确定性的、与模型无关的事后方法BELLA，用于解释回归黑盒模型的个别预测。该方法通过特征空间中训练的线性模型提供解释，使得该模型的系数可以直接用于计算特征值的预测值。此外，BELLA最大化了线性模型适用的领域范围。 |

# 详细

[^1]: 机器中的私语：LLM集成系统中的保密性

    Whispers in the Machine: Confidentiality in LLM-integrated Systems

    [https://arxiv.org/abs/2402.06922](https://arxiv.org/abs/2402.06922)

    本研究提供了一种评估LLM集成系统保密性的系统化方法，通过形式化一个"秘密密钥"游戏来捕捉模型隐藏私人信息的能力。评估了八种攻击和四种防御方法，发现当前的防御方法缺乏泛化性能。

    

    大规模语言模型（LLM）越来越多地与外部工具集成。尽管这些集成可以显著提高LLM的功能，但它们也在不同组件之间创建了一个新的攻击面，可能泄露机密数据。具体而言，恶意工具可以利用LLM本身的漏洞来操纵模型并损害其他服务的数据，这引发了在LLM集成环境中如何保护私密数据的问题。在这项工作中，我们提供了一种系统评估LLM集成系统保密性的方法。为此，我们形式化了一个"秘密密钥"游戏，可以捕捉模型隐藏私人信息的能力。这使我们能够比较模型对保密性攻击的脆弱性以及不同防御策略的有效性。在这个框架中，我们评估了八种先前发表的攻击和四种防御方法。我们发现当前的防御方法缺乏泛化性能。

    Large Language Models (LLMs) are increasingly integrated with external tools. While these integrations can significantly improve the functionality of LLMs, they also create a new attack surface where confidential data may be disclosed between different components. Specifically, malicious tools can exploit vulnerabilities in the LLM itself to manipulate the model and compromise the data of other services, raising the question of how private data can be protected in the context of LLM integrations.   In this work, we provide a systematic way of evaluating confidentiality in LLM-integrated systems. For this, we formalize a "secret key" game that can capture the ability of a model to conceal private information. This enables us to compare the vulnerability of a model against confidentiality attacks and also the effectiveness of different defense strategies. In this framework, we evaluate eight previously published attacks and four defenses. We find that current defenses lack generalization
    
[^2]: 通过协作解决城市拥堵：基于异构GNN的协调编队和交通信号控制的多智能体强化学习方法

    Combat Urban Congestion via Collaboration: Heterogeneous GNN-based MARL for Coordinated Platooning and Traffic Signal Control. (arXiv:2310.10948v1 [cs.LG])

    [http://arxiv.org/abs/2310.10948](http://arxiv.org/abs/2310.10948)

    本文提出了一种基于异构图多智能体强化学习和交通理论的创新解决方案，通过将车辆编队和交通信号控制作为不同的强化学习智能体，并结合图神经网络实现协调，以优化交通流量和缓解城市拥堵。

    

    多年来，强化学习已经成为一种流行的方法，用于独立或分层方式开发信号控制和车辆编队策略。然而，在实时中联合控制这两者以减轻交通拥堵带来了新的挑战，如信号控制和编队之间固有的物理和行为异质性，以及它们之间的协调。本文提出了一种创新的解决方案来应对这些挑战，基于异构图多智能体强化学习和交通理论。我们的方法包括：1）将编队和信号控制设计为不同的强化学习智能体，具有自己的观测、动作和奖励函数，以优化交通流量；2）通过在多智能体强化学习中引入图神经网络来设计协调，以促进区域范围内智能体之间的无缝信息交换。我们通过SUMO模拟环境评估了我们的方法。

    Over the years, reinforcement learning has emerged as a popular approach to develop signal control and vehicle platooning strategies either independently or in a hierarchical way. However, jointly controlling both in real-time to alleviate traffic congestion presents new challenges, such as the inherent physical and behavioral heterogeneity between signal control and platooning, as well as coordination between them. This paper proposes an innovative solution to tackle these challenges based on heterogeneous graph multi-agent reinforcement learning and traffic theories. Our approach involves: 1) designing platoon and signal control as distinct reinforcement learning agents with their own set of observations, actions, and reward functions to optimize traffic flow; 2) designing coordination by incorporating graph neural networks within multi-agent reinforcement learning to facilitate seamless information exchange among agents on a regional scale. We evaluate our approach through SUMO simu
    
[^3]: 从雷达数据学习攀升飞机的生成模型

    Learning Generative Models for Climbing Aircraft from Radar Data. (arXiv:2309.14941v1 [eess.SY])

    [http://arxiv.org/abs/2309.14941](http://arxiv.org/abs/2309.14941)

    本文提出了一种利用雷达数据学习的生成模型，能够准确预测攀升飞机的轨迹，并通过学习修正推力的函数来提高预测准确性。该方法的优势包括：与标准模型相比，到达时间的预测误差减少了66.3%；生成的轨迹与测试数据相比更加真实；并且能够以最小的计算成本计算置信区间。

    

    攀升飞机的准确轨迹预测受到机载设备操作的不确定性的影响，可能导致预测轨迹与观测轨迹之间存在显著的差异。本文提出了一种生成模型，通过从数据中学习修正推力的函数来丰富标准的飞机基础数据（BADA）模型。该方法具有三个特点：与BADA相比，到达时间的预测误差减少了66.3%；生成的轨迹与测试数据相比更加真实；并且能够以最小的计算成本计算置信区间。

    Accurate trajectory prediction (TP) for climbing aircraft is hampered by the presence of epistemic uncertainties concerning aircraft operation, which can lead to significant misspecification between predicted and observed trajectories. This paper proposes a generative model for climbing aircraft in which the standard Base of Aircraft Data (BADA) model is enriched by a functional correction to the thrust that is learned from data. The method offers three features: predictions of the arrival time with 66.3% less error when compared to BADA; generated trajectories that are realistic when compared to test data; and a means of computing confidence bounds for minimal computational cost.
    
[^4]: 基于正则化Stein差异的神经算子变分推断用于深度高斯过程

    Neural Operator Variational Inference based on Regularized Stein Discrepancy for Deep Gaussian Processes. (arXiv:2309.12658v1 [cs.LG])

    [http://arxiv.org/abs/2309.12658](http://arxiv.org/abs/2309.12658)

    基于正则化Stein差异的神经算子变分推断用于深度高斯过程，通过使用神经生成器获得取样器以及使用蒙特卡罗估计和子采样随机优化技术解决极小极大问题，提高了深度高斯过程模型的表达能力和推断效果。

    

    深度高斯过程（DGP）模型提供了一种强大的非参数贝叶斯推断方法，但精确推断通常是难以求解的，这促使我们使用各种近似方法。然而，现有的方法，如均值场高斯假设，限制了DGP模型的表达能力和效果，而随机逼近可能计算代价高昂。为解决这些挑战，我们引入了基于神经算子的变分推断（NOVI）用于深度高斯过程。NOVI使用神经生成器获得取样器，并在L2空间中最小化生成分布和真实后验之间的正则化Stein差异。我们使用蒙特卡罗估计和子采样随机优化技术解决了极小极大问题。我们证明了通过将Fisher散度与常数相乘来控制方法引入的偏差，从而实现了鲁棒的误差控制，确保了算法的稳定性和精确性。

    Deep Gaussian Process (DGP) models offer a powerful nonparametric approach for Bayesian inference, but exact inference is typically intractable, motivating the use of various approximations. However, existing approaches, such as mean-field Gaussian assumptions, limit the expressiveness and efficacy of DGP models, while stochastic approximation can be computationally expensive. To tackle these challenges, we introduce Neural Operator Variational Inference (NOVI) for Deep Gaussian Processes. NOVI uses a neural generator to obtain a sampler and minimizes the Regularized Stein Discrepancy in L2 space between the generated distribution and true posterior. We solve the minimax problem using Monte Carlo estimation and subsampling stochastic optimization techniques. We demonstrate that the bias introduced by our method can be controlled by multiplying the Fisher divergence with a constant, which leads to robust error control and ensures the stability and precision of the algorithm. Our experim
    
[^5]: BELLA: 通过本地线性逼近进行黑盒模型解释

    BELLA: Black box model Explanations by Local Linear Approximations. (arXiv:2305.11311v1 [cs.LG])

    [http://arxiv.org/abs/2305.11311](http://arxiv.org/abs/2305.11311)

    本文提出了一种确定性的、与模型无关的事后方法BELLA，用于解释回归黑盒模型的个别预测。该方法通过特征空间中训练的线性模型提供解释，使得该模型的系数可以直接用于计算特征值的预测值。此外，BELLA最大化了线性模型适用的领域范围。

    

    近年来，理解黑盒模型的决策过程不仅成为法律要求，也成为评估其性能的另一种方式。然而，现有的事后解释方法依赖于合成数据生成，这引入了不确定性并可能损害解释的可靠性，并且它们 tend to produce explanations that apply to only very few data points. This makes the explanations brittle and limited in scope. Finally, they provide scores that have no direct verifiable meaning. In this paper, we present BELLA, a deterministic model-agnostic post-hoc approach for explaining the individual predictions of regression black-box models. BELLA provides explanations in the form of a linear model trained in the feature space. Thus, its coefficients can be used directly to compute the predicted value from the feature values. Furthermore, BELLA maximizes the size of the neighborhood to which the linear model a

    In recent years, understanding the decision-making process of black-box models has become not only a legal requirement but also an additional way to assess their performance. However, the state of the art post-hoc interpretation approaches rely on synthetic data generation. This introduces uncertainty and can hurt the reliability of the interpretations. Furthermore, they tend to produce explanations that apply to only very few data points. This makes the explanations brittle and limited in scope. Finally, they provide scores that have no direct verifiable meaning. In this paper, we present BELLA, a deterministic model-agnostic post-hoc approach for explaining the individual predictions of regression black-box models. BELLA provides explanations in the form of a linear model trained in the feature space. Thus, its coefficients can be used directly to compute the predicted value from the feature values. Furthermore, BELLA maximizes the size of the neighborhood to which the linear model a
    

