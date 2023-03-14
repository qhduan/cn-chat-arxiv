# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Digital Twin-Assisted Knowledge Distillation Framework for Heterogeneous Federated Learning.](http://arxiv.org/abs/2303.06155) | 本文提出了一种数字孪生辅助的知识蒸馏框架，用于解决联邦学习系统中的异构性问题，用户可以选择自己的神经网络模型并从大型教师模型中蒸馏知识，同时利用数字孪生在服务器上训练大型教师模型，最终通过混合整数规划和Q-learning算法实现模型选择和资源分配。 |
| [^2] | [Real-time scheduling of renewable power systems through planning-based reinforcement learning.](http://arxiv.org/abs/2303.05205) | 本文提出了一种基于规划强化学习算法和真实电力网环境的系统解决方案，可以实现发电机的规划和更细的时间分辨率调整，从而增加了电网的能力。 |
| [^3] | [On the Stability Analysis of Open Federated Learning Systems.](http://arxiv.org/abs/2209.12307) | 本文研究了开放式联邦学习系统的稳定性问题，提出了一种新的性能度量，即开放式FL系统的稳定性，并在假设本地客户端函数是强凸和平滑的情况下，理论上量化了两种FL算法的稳定半径。 |
| [^4] | [DynLight: Realize dynamic phase duration with multi-level traffic signal control.](http://arxiv.org/abs/2204.03471) | 本文已被撤回，原因是语言和理论描述不够令人满意，作者已经进行了修订和更新。 |
| [^5] | [Learning Torque Control for Quadrupedal Locomotion.](http://arxiv.org/abs/2203.05194) | 本文提出了一种基于扭矩的强化学习框架，直接预测关节扭矩，避免使用PD控制器，通过广泛的实验验证，四足动物能够穿越各种地形并抵抗外部干扰，同时保持运动。 |
| [^6] | [FLSys: Toward an Open Ecosystem for Federated Learning Mobile Apps.](http://arxiv.org/abs/2111.09445) | 本文介绍了FLSys，一个移动-云联邦学习（FL）系统，可以成为FL模型和应用程序开放生态系统的关键组成部分。FLSys旨在在智能手机上使用移动感测数据。它平衡了模型性能和资源消耗，容忍通信故障，并实现了可扩展性。FLSys提供了先进的隐私保护机制和一个通用的API，供第三方应用程序开发人员访问FL模型。 |
| [^7] | [Data-Driven Reachability Analysis from Noisy Data.](http://arxiv.org/abs/2105.07229) | 本文提出了一种从嘈杂的数据中计算可达集的算法，适用于不同类型的系统，包括线性、多项式和非线性系统。算法基于矩阵zonotope，可以提供较少保守的可达集，并且可以将关于未知系统模型的先前知识纳入计算。算法具有理论保证，并在多个数值示例和实际实验中得到了验证。 |

# 详细

[^1]: 数字孪生辅助异构联邦学习的知识蒸馏框架

    Digital Twin-Assisted Knowledge Distillation Framework for Heterogeneous Federated Learning. (arXiv:2303.06155v1 [cs.LG])

    [http://arxiv.org/abs/2303.06155](http://arxiv.org/abs/2303.06155)

    本文提出了一种数字孪生辅助的知识蒸馏框架，用于解决联邦学习系统中的异构性问题，用户可以选择自己的神经网络模型并从大型教师模型中蒸馏知识，同时利用数字孪生在服务器上训练大型教师模型，最终通过混合整数规划和Q-learning算法实现模型选择和资源分配。

    This paper proposes a digital twin-assisted knowledge distillation framework for heterogeneous federated learning, where users can select their own neural network models and distill knowledge from a big teacher model, and the teacher model can be trained on a digital twin located in the server. The joint problem of model selection and training offloading and resource allocation for users is formulated as a mixed integer programming problem and solved using Q-learning and optimization algorithms.

    本文提出了一种知识蒸馏驱动的联邦学习框架，以应对联邦学习系统中的异构性，其中每个用户可以根据需要选择其神经网络模型，并使用自己的私有数据集从大型教师模型中蒸馏知识。为了克服在资源有限的用户设备上训练大型教师模型的挑战，利用数字孪生的方式，教师模型可以在具有足够计算资源的服务器上的数字孪生中进行训练。然后，在模型蒸馏期间，每个用户可以在物理实体或数字代理处更新其模型的参数。为用户选择模型和训练卸载和资源分配制定了混合整数规划（MIP）问题。为了解决这个问题，联合使用Q-learning和优化，其中Q-learning为用户选择模型并确定是在本地还是在服务器上进行训练，而优化则用于资源分配。

    In this paper, to deal with the heterogeneity in federated learning (FL) systems, a knowledge distillation (KD) driven training framework for FL is proposed, where each user can select its neural network model on demand and distill knowledge from a big teacher model using its own private dataset. To overcome the challenge of train the big teacher model in resource limited user devices, the digital twin (DT) is exploit in the way that the teacher model can be trained at DT located in the server with enough computing resources. Then, during model distillation, each user can update the parameters of its model at either the physical entity or the digital agent. The joint problem of model selection and training offloading and resource allocation for users is formulated as a mixed integer programming (MIP) problem. To solve the problem, Q-learning and optimization are jointly used, where Q-learning selects models for users and determines whether to train locally or on the server, and optimiz
    
[^2]: 基于规划强化学习的可再生能源电力系统实时调度

    Real-time scheduling of renewable power systems through planning-based reinforcement learning. (arXiv:2303.05205v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2303.05205](http://arxiv.org/abs/2303.05205)

    本文提出了一种基于规划强化学习算法和真实电力网环境的系统解决方案，可以实现发电机的规划和更细的时间分辨率调整，从而增加了电网的能力。

    This paper proposes a systematic solution based on the state-of-the-art reinforcement learning algorithm and the real power grid environment, which enables planning and finer time resolution adjustments of power generators, including unit commitment and economic dispatch, thus increasing the grid's ability.

    不断增长的可再生能源来源对传统电力调度提出了重大挑战。运营商难以获得准确的可再生能源发电日前预测，因此需要未来调度系统根据超短期预测进行实时调度决策。受计算速度限制，传统的基于优化的方法无法解决这个问题。最近强化学习（RL）的发展已经展示了解决这个挑战的潜力。然而，现有的RL方法在约束复杂性、算法性能和环境保真度方面不足。我们是第一个提出基于最先进的强化学习算法和真实电力网环境的系统解决方案。所提出的方法使发电机的规划和更细的时间分辨率调整成为可能，包括机组组合和经济调度，从而增加了电网的能力。

    The growing renewable energy sources have posed significant challenges to traditional power scheduling. It is difficult for operators to obtain accurate day-ahead forecasts of renewable generation, thereby requiring the future scheduling system to make real-time scheduling decisions aligning with ultra-short-term forecasts. Restricted by the computation speed, traditional optimization-based methods can not solve this problem. Recent developments in reinforcement learning (RL) have demonstrated the potential to solve this challenge. However, the existing RL methods are inadequate in terms of constraint complexity, algorithm performance, and environment fidelity. We are the first to propose a systematic solution based on the state-of-the-art reinforcement learning algorithm and the real power grid environment. The proposed approach enables planning and finer time resolution adjustments of power generators, including unit commitment and economic dispatch, thus increasing the grid's abilit
    
[^3]: 开放式联邦学习系统的稳定性分析

    On the Stability Analysis of Open Federated Learning Systems. (arXiv:2209.12307v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2209.12307](http://arxiv.org/abs/2209.12307)

    本文研究了开放式联邦学习系统的稳定性问题，提出了一种新的性能度量，即开放式FL系统的稳定性，并在假设本地客户端函数是强凸和平滑的情况下，理论上量化了两种FL算法的稳定半径。

    This paper studies the stability issue of open federated learning systems, proposes a new performance metric, namely the stability of open FL systems, and theoretically quantifies the stability radius of two FL algorithms under the assumption that local clients' functions are strongly convex and smooth.

    我们考虑开放式联邦学习系统，其中客户端可能在联邦学习过程中加入和/或离开系统。由于存在客户端数量的变化，无法保证在开放系统中收敛到固定模型。因此，我们采用一种新的性能度量，称为开放式FL系统的稳定性，它量化了在开放系统中学习模型的大小。在假设本地客户端函数是强凸和平滑的情况下，我们理论上量化了两种FL算法（即本地SGD和本地Adam）的稳定半径。我们观察到，这个半径依赖于几个关键参数，包括函数条件数以及随机梯度的方差。我们的理论结果在合成和真实世界基准数据集上通过数值模拟进一步验证。

    We consider the open federated learning (FL) systems, where clients may join and/or leave the system during the FL process. Given the variability of the number of present clients, convergence to a fixed model cannot be guaranteed in open systems. Instead, we resort to a new performance metric that we term the stability of open FL systems, which quantifies the magnitude of the learned model in open systems. Under the assumption that local clients' functions are strongly convex and smooth, we theoretically quantify the radius of stability for two FL algorithms, namely local SGD and local Adam. We observe that this radius relies on several key parameters, including the function condition number as well as the variance of the stochastic gradient. Our theoretical results are further verified by numerical simulations on both synthetic and real-world benchmark data-sets.
    
[^4]: DynLight: 多级交通信号控制实现动态相位时长

    DynLight: Realize dynamic phase duration with multi-level traffic signal control. (arXiv:2204.03471v4 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2204.03471](http://arxiv.org/abs/2204.03471)

    本文已被撤回，原因是语言和理论描述不够令人满意，作者已经进行了修订和更新。

    The article has been withdrawn due to unsatisfactory language and theoretical description, and the authors have revised and updated it.

    我们因以下原因撤回本文：1.本文的语言和理论描述不够令人满意；2.我们在其他作者的帮助下丰富和修订了本文；3.我们必须更新作者贡献信息。

    We would like to withdraw this article for the following reasons: 1 this article is not satisfactory for limited language and theoretical description; 2 we have enriched and revised this article with the help of other authors; 3 we must update the author contribution information.
    
[^5]: 学习四足动物运动的扭矩控制

    Learning Torque Control for Quadrupedal Locomotion. (arXiv:2203.05194v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2203.05194](http://arxiv.org/abs/2203.05194)

    本文提出了一种基于扭矩的强化学习框架，直接预测关节扭矩，避免使用PD控制器，通过广泛的实验验证，四足动物能够穿越各种地形并抵抗外部干扰，同时保持运动。

    This paper proposes a torque-based reinforcement learning framework that directly predicts joint torques, avoiding the use of a PD controller. The framework is validated through extensive experiments, where a quadruped is capable of traversing various terrain and resisting external disturbances while maintaining locomotion.

    强化学习已成为开发四足机器人控制器的一种有前途的方法。传统上，用于运动的RL设计遵循基于位置的范例，其中RL策略以低频率输出目标关节位置，然后由高频比例-导数（PD）控制器跟踪以产生关节扭矩。相比之下，对于四足动物运动的基于模型的控制，已经从基于位置的控制范例转向基于扭矩的控制。鉴于基于模型的控制的最新进展，我们通过引入基于扭矩的RL框架，探索了一种替代基于位置的RL范例的方法，其中RL策略直接在高频率下预测关节扭矩，从而避免使用PD控制器。所提出的学习扭矩控制框架通过广泛的实验进行了验证，在这些实验中，四足动物能够穿越各种地形并抵抗外部干扰，同时保持运动。

    Reinforcement learning (RL) has become a promising approach to developing controllers for quadrupedal robots. Conventionally, an RL design for locomotion follows a position-based paradigm, wherein an RL policy outputs target joint positions at a low frequency that are then tracked by a high-frequency proportional-derivative (PD) controller to produce joint torques. In contrast, for the model-based control of quadrupedal locomotion, there has been a paradigm shift from position-based control to torque-based control. In light of the recent advances in model-based control, we explore an alternative to the position-based RL paradigm, by introducing a torque-based RL framework, where an RL policy directly predicts joint torques at a high frequency, thus circumventing the use of a PD controller. The proposed learning torque control framework is validated with extensive experiments, in which a quadruped is capable of traversing various terrain and resisting external disturbances while followi
    
[^6]: FLSys：面向联邦学习移动应用的开放生态系统

    FLSys: Toward an Open Ecosystem for Federated Learning Mobile Apps. (arXiv:2111.09445v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2111.09445](http://arxiv.org/abs/2111.09445)

    本文介绍了FLSys，一个移动-云联邦学习（FL）系统，可以成为FL模型和应用程序开放生态系统的关键组成部分。FLSys旨在在智能手机上使用移动感测数据。它平衡了模型性能和资源消耗，容忍通信故障，并实现了可扩展性。FLSys提供了先进的隐私保护机制和一个通用的API，供第三方应用程序开发人员访问FL模型。

    This article introduces FLSys, a mobile-cloud federated learning (FL) system that can be a key component for an open ecosystem of FL models and apps. FLSys is designed to work on smart phones with mobile sensing data. It balances model performance with resource consumption, tolerates communication failures, and achieves scalability. FLSys provides advanced privacy preserving mechanisms and a common API for third-party app developers to access FL models.

    本文介绍了FLSys的设计、实现和评估，这是一个移动-云联邦学习（FL）系统，可以成为FL模型和应用程序开放生态系统的关键组成部分。FLSys旨在在智能手机上使用移动感测数据。它平衡了模型性能和资源消耗，容忍通信故障，并实现了可扩展性。在FLSys中，不同的DL模型和不同的FL聚合方法可以同时被不同的应用程序训练和访问。此外，FLSys提供了先进的隐私保护机制和一个通用的API，供第三方应用程序开发人员访问FL模型。FLSys采用模块化设计，实现在Android和AWS云中。我们与人类活动识别（HAR）模型共同设计了FLSys。在4个月的时间里，从100多名大学生中收集了HAR感测数据。我们实现了HAR-Wild，这是一个针对移动设备量身定制的CNN模型，具有数据增强机制以减轻p

    This article presents the design, implementation, and evaluation of FLSys, a mobile-cloud federated learning (FL) system, which can be a key component for an open ecosystem of FL models and apps. FLSys is designed to work on smart phones with mobile sensing data. It balances model performance with resource consumption, tolerates communication failures, and achieves scalability. In FLSys, different DL models with different FL aggregation methods can be trained and accessed concurrently by different apps. Furthermore, FLSys provides advanced privacy preserving mechanisms and a common API for third-party app developers to access FL models. FLSys adopts a modular design and is implemented in Android and AWS cloud. We co-designed FLSys with a human activity recognition (HAR) model. HAR sensing data was collected in the wild from 100+ college students during a 4-month period. We implemented HAR-Wild, a CNN model tailored to mobile devices, with a data augmentation mechanism to mitigate the p
    
[^7]: 从嘈杂的数据中进行数据驱动的可达性分析

    Data-Driven Reachability Analysis from Noisy Data. (arXiv:2105.07229v3 [eess.SY] UPDATED)

    [http://arxiv.org/abs/2105.07229](http://arxiv.org/abs/2105.07229)

    本文提出了一种从嘈杂的数据中计算可达集的算法，适用于不同类型的系统，包括线性、多项式和非线性系统。算法基于矩阵zonotope，可以提供较少保守的可达集，并且可以将关于未知系统模型的先前知识纳入计算。算法具有理论保证，并在多个数值示例和实际实验中得到了验证。

    This paper proposes an algorithm for computing reachable sets directly from noisy data without a given system model, which is applicable to different types of systems including linear, polynomial, and nonlinear systems. The algorithm is based on matrix zonotopes and can provide less conservative reachable sets while incorporating prior knowledge about the unknown system model. Theoretical guarantees are given and the applicability of the algorithm is demonstrated through numerical examples and real experiments.

    我们考虑在没有给定系统模型的情况下直接从嘈杂的数据中计算可达集的问题。我们提出了几种适用于生成数据的不同类型系统的可达性算法。首先，我们提出了一种基于矩阵zonotope的算法，用于计算线性系统的过估计可达集。引入了约束矩阵zonotope以提供较少保守的可达集，但代价是增加计算开销，并用于将关于未知系统模型的先前知识纳入计算。然后，我们将这种方法扩展到多项式系统，并在Lipschitz连续性的假设下扩展到非线性系统。这些算法的理论保证是它们给出一个包含真实可达集的适当过估计可达集。多个数值示例和实际实验显示了引入算法的适用性，并进行了算法之间的比较。

    We consider the problem of computing reachable sets directly from noisy data without a given system model. Several reachability algorithms are presented for different types of systems generating the data. First, an algorithm for computing over-approximated reachable sets based on matrix zonotopes is proposed for linear systems. Constrained matrix zonotopes are introduced to provide less conservative reachable sets at the cost of increased computational expenses and utilized to incorporate prior knowledge about the unknown system model. Then we extend the approach to polynomial systems and, under the assumption of Lipschitz continuity, to nonlinear systems. Theoretical guarantees are given for these algorithms in that they give a proper over-approximate reachable set containing the true reachable set. Multiple numerical examples and real experiments show the applicability of the introduced algorithms, and comparisons are made between algorithms.
    

