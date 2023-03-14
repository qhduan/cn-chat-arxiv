# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards Practical Multi-Robot Hybrid Tasks Allocation for Autonomous Cleaning.](http://arxiv.org/abs/2303.06531) | 本文提出了一个新的鲁棒混合整数线性规划模型，用于解决多机器人自主清洁系统中的混合任务分配问题，并建立了一个包括100个实例的数据集。 |
| [^2] | [Understanding the Synergies between Quality-Diversity and Deep Reinforcement Learning.](http://arxiv.org/abs/2303.06164) | 本文提出了广义演员-评论家QD-RL框架，用于QD-RL设置中的演员-评论家深度RL方法。该框架引入了两种新算法，PGA-ME（SAC）和PGA-ME（DroQ），将深度RL的最新进展应用于QD-RL设置，并解决了现有QD-RL算法无法解决的人形环境问题。 |
| [^3] | [Probabilistic Point Cloud Modeling via Self-Organizing Gaussian Mixture Models.](http://arxiv.org/abs/2302.00047) | 本文提出了一种基于自组织高斯混合模型的概率点云建模方法，可以根据场景复杂度自动调整模型复杂度，相比现有技术具有更好的泛化性能。 |
| [^4] | [Simulating the Integration of Urban Air Mobility into Existing Transportation Systems: A Survey.](http://arxiv.org/abs/2301.12901) | 本文调查了城市空中出行（UAM）在大都市交通中的研究现状，确定了将UAM融入城市交通系统的关键挑战和机遇，包括对现有交通模式和拥堵的影响；安全分析和风险评估；潜在的经济和环境效益；以及为UAM和地面交通开发共享基础设施和路线。同时，我们讨论了UAM的潜在好处，如缩短旅行时间和改善服务不足地区的可达性。 |
| [^5] | [Learning Neuro-symbolic Programs for Language Guided Robot Manipulation.](http://arxiv.org/abs/2211.06652) | 该论文提出了一种学习神经符号程序以进行语言引导的机器人操作的方法，可以处理语言和感知变化，端到端可训练，不需要中间监督。该方法使用符号推理构造，在潜在的神经物体为中心的表示上操作，允许对输入场景进行更深入的推理。 |
| [^6] | [D-Shape: Demonstration-Shaped Reinforcement Learning via Goal Conditioning.](http://arxiv.org/abs/2210.14428) | D-Shape是一种新的结合IL和RL的方法，它使用奖励塑形和目标条件化RL的思想来解决次优演示与回报最大化目标之间的冲突，能够在稀疏奖励网格世界领域中提高样本效率并一致地收敛到最优策略。 |
| [^7] | [Policy-Guided Lazy Search with Feedback for Task and Motion Planning.](http://arxiv.org/abs/2210.14055) | 本文提出了一种用于PDDLStream问题的求解器LAZY，它在动作骨架上维护单个集成搜索，随着在运动规划期间懒惰地绘制可能运动的样本，逐渐变得更具几何信息。同时，学习模型的目标导向策略和当前运动采样数据合并到LAZY中，以自适应地引导任务规划器，这导致了在不同数量的对象、目标和初始条件的未见测试环境中评估可行解搜索的显着加速。 |
| [^8] | [Differentiable Physics Simulation of Dynamics-Augmented Neural Objects.](http://arxiv.org/abs/2210.09420) | 本文提出了一种可微分的流程，用于模拟将物体的几何形状表示为连续密度场的对象的运动，从密度场中估计物体的动力学特性，并引入了一种基于密度场的可微接触模型，使机器人能够自主构建物体模型并优化神经对象的抓取和操作轨迹。 |
| [^9] | [Self-Supervised Geometric Correspondence for Category-Level 6D Object Pose Estimation in the Wild.](http://arxiv.org/abs/2210.07199) | 本文提出了一种自监督学习方法，直接在大规模真实世界物体视频上进行类别级6D姿态估计。通过表面嵌入学习了输入图像和规范形状之间的密集对应关系，并提出了新颖的几何循环一致性损失。学习到的对应关系可以应用于6D姿态估计和其他任务。 |
| [^10] | [Differentiable Parsing and Visual Grounding of Natural Language Instructions for Object Placement.](http://arxiv.org/abs/2210.00215) | ParaGon是一种可微分的自然语言指令解析和视觉定位方法，通过将语言指令解析为以对象为中心的图形表示，以单独定位对象，并使用一种新颖的基于粒子的图神经网络来推理关于带有不确定性的物体放置。 |
| [^11] | [DM$^2$: Decentralized Multi-Agent Reinforcement Learning for Distribution Matching.](http://arxiv.org/abs/2206.00233) | 本文提出了一种基于分布匹配的去中心化多智能体强化学习方法，每个智能体独立地最小化与目标访问分布的相应分量的分布不匹配，可以实现收敛到生成目标分布的联合策略。 |
| [^12] | [Visuomotor Control in Multi-Object Scenes Using Object-Aware Representations.](http://arxiv.org/abs/2205.06333) | 本文探讨了使用物体感知表示学习技术进行机器人任务的有效性，以解决当前方法学习任务特定表示不能很好地转移到其他任务的问题，以及由监督方法学习的表示需要大量标记数据集的问题。 |
| [^13] | [Learning Torque Control for Quadrupedal Locomotion.](http://arxiv.org/abs/2203.05194) | 本文提出了一种基于扭矩的强化学习框架，直接预测关节扭矩，避免使用PD控制器，通过广泛的实验验证，四足动物能够穿越各种地形并抵抗外部干扰，同时保持运动。 |
| [^14] | [Learning to Search in Task and Motion Planning with Streams.](http://arxiv.org/abs/2111.13144) | 本文提出了一种几何信息的符号规划器，使用图神经网络优先级排序，以最佳优先方式扩展对象和事实集合，从而改善了在任务和动作规划中的长期推理能力。在7自由度机械臂的堆叠操纵任务中得到了应用。 |
| [^15] | [A trained humanoid robot can perform human-like crossmodal social attention and conflict resolution.](http://arxiv.org/abs/2111.01906) | 本研究采用跨模态冲突解决的神经机器人范例，使机器人表现出类人的社交关注，为增强人机社交互动提供了新思路。 |

# 详细

[^1]: 面向自主清洁的多机器人混合任务分配的实践研究

    Towards Practical Multi-Robot Hybrid Tasks Allocation for Autonomous Cleaning. (arXiv:2303.06531v1 [cs.RO])

    [http://arxiv.org/abs/2303.06531](http://arxiv.org/abs/2303.06531)

    本文提出了一个新的鲁棒混合整数线性规划模型，用于解决多机器人自主清洁系统中的混合任务分配问题，并建立了一个包括100个实例的数据集。

    This paper proposes a novel robust mixed-integer linear programming model for multi-robot hybrid-task allocation in uncertain autonomous cleaning systems, and establishes a dataset of 100 instances.

    任务分配在多机器人自主清洁系统中起着至关重要的作用，多个机器人一起工作以清洁大面积区域。然而，迄今为止相关研究存在几个问题。大多数当前研究主要关注于确定性的单任务分配，而不考虑不确定工作环境中的混合任务。此外，缺乏相关研究的数据集和基准。在本文中，我们通过解决这些问题，为不确定的自主清洁系统的多机器人混合任务分配做出了贡献。首先，我们通过鲁棒优化模型来建模清洁环境中的不确定性，并提出了一个新的鲁棒混合整数线性规划模型，其中包括混合清洁任务顺序和机器人的能力等实际约束。其次，我们建立了一个数据集，包括100个实例，每个实例都有2D手动标记的图像和3D模型。第三，我们提供了关于所提出模型的全面结果。

    Task allocation plays a vital role in multi-robot autonomous cleaning systems, where multiple robots work together to clean a large area. However, there are several problems in relevant research to date. Most current studies mainly focus on deterministic, single-task allocation for cleaning robots, without considering hybrid tasks in uncertain working environments. Moreover, there is a lack of datasets and benchmarks for relevant research. In this paper, we contribute to multi-robot hybrid-task allocation for uncertain autonomous cleaning systems by addressing these problems. First, we model the uncertainties in the cleaning environment via robust optimization and propose a novel robust mixed-integer linear programming model with practical constraints including hybrid cleaning task order and robot's ability. Second, we establish a dataset of 100 instances made from floor plans, each of which has 2D manually-labeled images and a 3D model. Third, we provide comprehensive results on the c
    
[^2]: 理解质量多样性和深度强化学习之间的协同作用

    Understanding the Synergies between Quality-Diversity and Deep Reinforcement Learning. (arXiv:2303.06164v1 [cs.LG])

    [http://arxiv.org/abs/2303.06164](http://arxiv.org/abs/2303.06164)

    本文提出了广义演员-评论家QD-RL框架，用于QD-RL设置中的演员-评论家深度RL方法。该框架引入了两种新算法，PGA-ME（SAC）和PGA-ME（DroQ），将深度RL的最新进展应用于QD-RL设置，并解决了现有QD-RL算法无法解决的人形环境问题。

    This paper proposes a Generalized Actor-Critic QD-RL framework for actor-critic deep RL methods in the QD-RL setting. The framework introduces two new algorithms, PGA-ME (SAC) and PGA-ME (DroQ), which apply recent advancements in Deep RL to the QD-RL setting and solve the humanoid environment problem that existing QD-RL algorithms cannot solve.

    质量多样性（QD）和深度强化学习（RL）之间的协同作用已经导致了强大的混合QD-RL算法，展示了巨大的潜力，并带来了两个领域的最佳实践。然而，尽管其他RL算法取得了显著进展，但在先前的混合方法中仅使用了单个深度RL算法（TD3）。此外，QD和RL之间的优化过程存在根本差异，需要更加原则性的方法。我们提出了广义演员-评论家QD-RL，这是一个统一的模块化框架，用于QD-RL设置中的演员-评论家深度RL方法。该框架提供了一条研究深度RL在QD-RL设置中的见解的路径，这是在QD-RL中取得进展的重要且有效的方法。我们引入了两种新算法，PGA-ME（SAC）和PGA-ME（DroQ），将深度RL的最新进展应用于QD-RL设置，并解决了现有QD-RL算法无法解决的人形环境问题。

    The synergies between Quality-Diversity (QD) and Deep Reinforcement Learning (RL) have led to powerful hybrid QD-RL algorithms that have shown tremendous potential, and brings the best of both fields. However, only a single deep RL algorithm (TD3) has been used in prior hybrid methods despite notable progress made by other RL algorithms. Additionally, there are fundamental differences in the optimization procedures between QD and RL which would benefit from a more principled approach. We propose Generalized Actor-Critic QD-RL, a unified modular framework for actor-critic deep RL methods in the QD-RL setting. This framework provides a path to study insights from Deep RL in the QD-RL setting, which is an important and efficient way to make progress in QD-RL. We introduce two new algorithms, PGA-ME (SAC) and PGA-ME (DroQ) which apply recent advancements in Deep RL to the QD-RL setting, and solves the humanoid environment which was not possible using existing QD-RL algorithms. However, we 
    
[^3]: 基于自组织高斯混合模型的概率点云建模

    Probabilistic Point Cloud Modeling via Self-Organizing Gaussian Mixture Models. (arXiv:2302.00047v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.00047](http://arxiv.org/abs/2302.00047)

    本文提出了一种基于自组织高斯混合模型的概率点云建模方法，可以根据场景复杂度自动调整模型复杂度，相比现有技术具有更好的泛化性能。

    This paper proposes a probabilistic point cloud modeling method based on self-organizing Gaussian mixture models, which can automatically adjust the model complexity according to the scene complexity, and has better generalization performance compared to existing techniques.

    本文提出了一种连续的概率建模方法，用于使用有限高斯混合模型（GMM）对空间点云数据进行建模，其中组件的数量基于场景复杂性进行调整。我们利用信息论学习中的自组织原理，根据传感器数据中的相关信息自动调整GMM模型的复杂度。该方法在具有不同场景复杂度的实际数据上与现有的点云建模技术进行了评估。

    This letter presents a continuous probabilistic modeling methodology for spatial point cloud data using finite Gaussian Mixture Models (GMMs) where the number of components are adapted based on the scene complexity. Few hierarchical and adaptive methods have been proposed to address the challenge of balancing model fidelity with size. Instead, state-of-the-art mapping approaches require tuning parameters for specific use cases, but do not generalize across diverse environments. To address this gap, we utilize a self-organizing principle from information-theoretic learning to automatically adapt the complexity of the GMM model based on the relevant information in the sensor data. The approach is evaluated against existing point cloud modeling techniques on real-world data with varying degrees of scene complexity.
    
[^4]: 模拟城市空中出行融入现有交通系统：一项调查

    Simulating the Integration of Urban Air Mobility into Existing Transportation Systems: A Survey. (arXiv:2301.12901v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2301.12901](http://arxiv.org/abs/2301.12901)

    本文调查了城市空中出行（UAM）在大都市交通中的研究现状，确定了将UAM融入城市交通系统的关键挑战和机遇，包括对现有交通模式和拥堵的影响；安全分析和风险评估；潜在的经济和环境效益；以及为UAM和地面交通开发共享基础设施和路线。同时，我们讨论了UAM的潜在好处，如缩短旅行时间和改善服务不足地区的可达性。

    This paper surveys the current state of research on urban air mobility (UAM) in metropolitan-scale traffic using simulation techniques, identifying key challenges and opportunities for integrating UAM into urban transportation systems, including impacts on existing traffic patterns and congestion, safety analysis and risk assessment, potential economic and environmental benefits, and the development of shared infrastructure and routes for UAM and ground-based transportation. The potential benefits of UAM, such as reduced travel times and improved accessibility for underserved areas, are also discussed.

    城市空中出行（UAM）有可能彻底改变大都市地区的交通方式，提供一种新的交通方式，缓解拥堵，提高可达性。然而，将UAM融入现有交通系统是一项复杂的任务，需要深入了解其对交通流量和容量的影响。在本文中，我们进行了一项调查，使用模拟技术调查了UAM在大都市交通中的研究现状。我们确定了将UAM融入城市交通系统的关键挑战和机遇，包括对现有交通模式和拥堵的影响；安全分析和风险评估；潜在的经济和环境效益；以及为UAM和地面交通开发共享基础设施和路线。我们还讨论了UAM的潜在好处，如缩短旅行时间和改善服务不足地区的可达性。我们的调查

    Urban air mobility (UAM) has the potential to revolutionize transportation in metropolitan areas, providing a new mode of transportation that could alleviate congestion and improve accessibility. However, the integration of UAM into existing transportation systems is a complex task that requires a thorough understanding of its impact on traffic flow and capacity. In this paper, we conduct a survey to investigate the current state of research on UAM in metropolitan-scale traffic using simulation techniques. We identify key challenges and opportunities for the integration of UAM into urban transportation systems, including impacts on existing traffic patterns and congestion; safety analysis and risk assessment; potential economic and environmental benefits; and the development of shared infrastructure and routes for UAM and ground-based transportation. We also discuss the potential benefits of UAM, such as reduced travel times and improved accessibility for underserved areas. Our survey 
    
[^5]: 学习神经符号程序以进行语言引导的机器人操作

    Learning Neuro-symbolic Programs for Language Guided Robot Manipulation. (arXiv:2211.06652v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2211.06652](http://arxiv.org/abs/2211.06652)

    该论文提出了一种学习神经符号程序以进行语言引导的机器人操作的方法，可以处理语言和感知变化，端到端可训练，不需要中间监督。该方法使用符号推理构造，在潜在的神经物体为中心的表示上操作，允许对输入场景进行更深入的推理。

    This paper proposes a method for learning neuro-symbolic programs for language guided robot manipulation, which can handle linguistic and perceptual variations, is end-to-end trainable, and requires no intermediate supervision. The method uses symbolic reasoning constructs that operate on a latent neural object-centric representation, allowing for deeper reasoning over the input scene.

    给定自然语言指令和输入场景，我们的目标是训练一个模型，输出一个可以由机器人执行的操作程序。先前的方法存在以下限制之一：（i）依赖手工编码的概念符号，限制了超出训练期间所见的一般化能力[1]（ii）从指令中推断出动作序列，但需要密集的子目标监督[2]或（iii）缺乏解释复杂指令所需的语义，这种语义需要更深入的以物体为中心的推理[3]。相比之下，我们的方法可以处理语言和感知变化，端到端可训练，不需要中间监督。所提出的模型使用符号推理构造，这些构造在潜在的神经物体为中心的表示上操作，允许对输入场景进行更深入的推理。我们方法的核心是一个模块化结构，包括分层指令解析器和动作模拟器，以学习解耦的行动序列。

    Given a natural language instruction and an input scene, our goal is to train a model to output a manipulation program that can be executed by the robot. Prior approaches for this task possess one of the following limitations: (i) rely on hand-coded symbols for concepts limiting generalization beyond those seen during training [1] (ii) infer action sequences from instructions but require dense sub-goal supervision [2] or (iii) lack semantics required for deeper object-centric reasoning inherent in interpreting complex instructions [3]. In contrast, our approach can handle linguistic as well as perceptual variations, end-to-end trainable and requires no intermediate supervision. The proposed model uses symbolic reasoning constructs that operate on a latent neural object-centric representation, allowing for deeper reasoning over the input scene. Central to our approach is a modular structure consisting of a hierarchical instruction parser and an action simulator to learn disentangled act
    
[^6]: D-Shape: 通过目标条件化实现演示形状的强化学习

    D-Shape: Demonstration-Shaped Reinforcement Learning via Goal Conditioning. (arXiv:2210.14428v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.14428](http://arxiv.org/abs/2210.14428)

    D-Shape是一种新的结合IL和RL的方法，它使用奖励塑形和目标条件化RL的思想来解决次优演示与回报最大化目标之间的冲突，能够在稀疏奖励网格世界领域中提高样本效率并一致地收敛到最优策略。

    D-Shape is a new method that combines imitation learning (IL) and reinforcement learning (RL) using reward shaping and goal-conditioned RL to resolve the conflict between suboptimal demonstrations and return-maximization objective of RL. It improves sample efficiency and consistently converges to the optimal policy in sparse-reward gridworld domains.

    将模仿学习（IL）和强化学习（RL）相结合是解决自主行为获取中样本效率低下的一种有前途的方法，但这样做的方法通常假定所需的行为演示由专家提供，该专家相对于任务奖励表现最佳。然而，如果提供的演示是次优的，则面临一个基本挑战，即IL的演示匹配目标与RL的回报最大化目标冲突。本文介绍了D-Shape，一种新的结合IL和RL的方法，它使用奖励塑形和目标条件化RL的思想来解决上述冲突。D-Shape允许从次优演示中学习，同时保留了找到相对于任务奖励的最优策略的能力。我们在稀疏奖励网格世界领域实验验证了D-Shape，结果表明它在样本效率方面优于RL，并且能够一致地收敛到最优策略。

    While combining imitation learning (IL) and reinforcement learning (RL) is a promising way to address poor sample efficiency in autonomous behavior acquisition, methods that do so typically assume that the requisite behavior demonstrations are provided by an expert that behaves optimally with respect to a task reward. If, however, suboptimal demonstrations are provided, a fundamental challenge appears in that the demonstration-matching objective of IL conflicts with the return-maximization objective of RL. This paper introduces D-Shape, a new method for combining IL and RL that uses ideas from reward shaping and goal-conditioned RL to resolve the above conflict. D-Shape allows learning from suboptimal demonstrations while retaining the ability to find the optimal policy with respect to the task reward. We experimentally validate D-Shape in sparse-reward gridworld domains, showing that it both improves over RL in terms of sample efficiency and converges consistently to the optimal polic
    
[^7]: 带反馈的策略引导懒惰搜索用于任务和动作规划

    Policy-Guided Lazy Search with Feedback for Task and Motion Planning. (arXiv:2210.14055v3 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2210.14055](http://arxiv.org/abs/2210.14055)

    本文提出了一种用于PDDLStream问题的求解器LAZY，它在动作骨架上维护单个集成搜索，随着在运动规划期间懒惰地绘制可能运动的样本，逐渐变得更具几何信息。同时，学习模型的目标导向策略和当前运动采样数据合并到LAZY中，以自适应地引导任务规划器，这导致了在不同数量的对象、目标和初始条件的未见测试环境中评估可行解搜索的显着加速。

    This paper proposes a solver LAZY for PDDLStream problems, which maintains a single integrated search over action skeletons, gradually becoming more geometrically informed as samples of possible motions are lazily drawn during motion planning. Meanwhile, learned models of goal-directed policies and current motion sampling data are incorporated in LAZY to adaptively guide the task planner, leading to significant speed-ups in the search for a feasible solution evaluated over unseen test environments of varying numbers of objects, goals, and initial conditions.

    PDDLStream求解器最近已经成为任务和动作规划（TAMP）问题的可行解决方案，将PDDL扩展到具有连续动作空间的问题。先前的工作已经展示了如何将PDDLStream问题简化为一系列PDDL规划问题，然后使用现成的规划器解决。然而，这种方法可能会导致长时间运行。在本文中，我们提出了LAZY，一种用于PDDLStream问题的求解器，它在动作骨架上维护单个集成搜索，随着在运动规划期间懒惰地绘制可能运动的样本，逐渐变得更具几何信息。我们探讨了如何将目标导向策略的学习模型和当前运动采样数据合并到LAZY中，以自适应地引导任务规划器。我们展示了这导致了在不同数量的对象、目标和初始条件的未见测试环境中评估可行解搜索的显着加速。我们评估了我们的TAMP方法

    PDDLStream solvers have recently emerged as viable solutions for Task and Motion Planning (TAMP) problems, extending PDDL to problems with continuous action spaces. Prior work has shown how PDDLStream problems can be reduced to a sequence of PDDL planning problems, which can then be solved using off-the-shelf planners. However, this approach can suffer from long runtimes. In this paper we propose LAZY, a solver for PDDLStream problems that maintains a single integrated search over action skeletons, which gets progressively more geometrically informed, as samples of possible motions are lazily drawn during motion planning. We explore how learned models of goal-directed policies and current motion sampling data can be incorporated in LAZY to adaptively guide the task planner. We show that this leads to significant speed-ups in the search for a feasible solution evaluated over unseen test environments of varying numbers of objects, goals, and initial conditions. We evaluate our TAMP appro
    
[^8]: 可微分的动力学增强神经对象的物理模拟

    Differentiable Physics Simulation of Dynamics-Augmented Neural Objects. (arXiv:2210.09420v3 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2210.09420](http://arxiv.org/abs/2210.09420)

    本文提出了一种可微分的流程，用于模拟将物体的几何形状表示为连续密度场的对象的运动，从密度场中估计物体的动力学特性，并引入了一种基于密度场的可微接触模型，使机器人能够自主构建物体模型并优化神经对象的抓取和操作轨迹。

    This paper proposes a differentiable pipeline for simulating the motion of objects that represent their geometry as a continuous density field parameterized as a deep network, estimates the dynamical properties of the object from the density field, introduces a differentiable contact model based on the density field for computing normal and friction forces resulting from collisions, and allows a robot to autonomously build object models and optimize grasps and manipulation trajectories of neural objects.

    我们提出了一种可微分的流程，用于模拟将物体的几何形状表示为连续密度场的对象的运动，该密度场被参数化为深度网络。这包括神经辐射场（NeRF）和其他相关模型。从密度场中，我们估计物体的动力学特性，包括其质量、质心和惯性矩阵。然后，我们引入了一种基于密度场的可微接触模型，用于计算由碰撞产生的法向和摩擦力。这使得机器人能够自主构建物体模型，这些模型从静止图像和运动中的物体视频中视觉上和动态上都是准确的。由此产生的动力学增强神经对象（DANOs）使用现有的可微分模拟引擎Dojo进行模拟，与其他标准模拟对象（如球体、平面和以URDF指定的机器人）进行交互。机器人可以使用这个模拟来优化神经对象的抓取和操作轨迹。

    We present a differentiable pipeline for simulating the motion of objects that represent their geometry as a continuous density field parameterized as a deep network. This includes Neural Radiance Fields (NeRFs), and other related models. From the density field, we estimate the dynamical properties of the object, including its mass, center of mass, and inertia matrix. We then introduce a differentiable contact model based on the density field for computing normal and friction forces resulting from collisions. This allows a robot to autonomously build object models that are visually and \emph{dynamically} accurate from still images and videos of objects in motion. The resulting Dynamics-Augmented Neural Objects (DANOs) are simulated with an existing differentiable simulation engine, Dojo, interacting with other standard simulation objects, such as spheres, planes, and robots specified as URDFs. A robot can use this simulation to optimize grasps and manipulation trajectories of neural ob
    
[^9]: 自监督几何对应用于野外类别级6D物体姿态估计

    Self-Supervised Geometric Correspondence for Category-Level 6D Object Pose Estimation in the Wild. (arXiv:2210.07199v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2210.07199](http://arxiv.org/abs/2210.07199)

    本文提出了一种自监督学习方法，直接在大规模真实世界物体视频上进行类别级6D姿态估计。通过表面嵌入学习了输入图像和规范形状之间的密集对应关系，并提出了新颖的几何循环一致性损失。学习到的对应关系可以应用于6D姿态估计和其他任务。

    This paper proposes a self-supervised learning approach for category-level 6D object pose estimation in the wild, which reconstructs the canonical 3D shape of an object category and learns dense correspondences between input images and the canonical shape via surface embedding. The proposed novel geometrical cycle-consistency losses construct cycles across 2D-3D spaces, across different instances and different time steps. The learned correspondence can be applied for 6D pose estimation and other tasks.

    尽管6D物体姿态估计在计算机视觉和机器人领域有广泛的应用，但由于缺乏注释，它仍然远未解决。当转向类别级6D姿态时，问题变得更加具有挑战性，因为需要对未见实例进行泛化。目前的方法受到从模拟或从人类收集的注释的限制。在本文中，我们通过引入一种自监督学习方法，直接在大规模真实世界物体视频上进行类别级6D姿态估计，克服了这一障碍。我们的框架重构了物体类别的规范3D形状，并通过表面嵌入学习了输入图像和规范形状之间的密集对应关系。对于训练，我们提出了新颖的几何循环一致性损失，它们在2D-3D空间、不同实例和不同时间步之间构建循环。学习到的对应关系可以应用于6D姿态估计和其他任务。

    While 6D object pose estimation has wide applications across computer vision and robotics, it remains far from being solved due to the lack of annotations. The problem becomes even more challenging when moving to category-level 6D pose, which requires generalization to unseen instances. Current approaches are restricted by leveraging annotations from simulation or collected from humans. In this paper, we overcome this barrier by introducing a self-supervised learning approach trained directly on large-scale real-world object videos for category-level 6D pose estimation in the wild. Our framework reconstructs the canonical 3D shape of an object category and learns dense correspondences between input images and the canonical shape via surface embedding. For training, we propose novel geometrical cycle-consistency losses which construct cycles across 2D-3D spaces, across different instances and different time steps. The learned correspondence can be applied for 6D pose estimation and othe
    
[^10]: 可微分的自然语言指令解析和视觉定位在物体放置任务中的应用

    Differentiable Parsing and Visual Grounding of Natural Language Instructions for Object Placement. (arXiv:2210.00215v4 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2210.00215](http://arxiv.org/abs/2210.00215)

    ParaGon是一种可微分的自然语言指令解析和视觉定位方法，通过将语言指令解析为以对象为中心的图形表示，以单独定位对象，并使用一种新颖的基于粒子的图神经网络来推理关于带有不确定性的物体放置。

    ParaGon is a differentiable method for natural language instruction parsing and visual grounding in object placement tasks. It parses language instructions into an object-centric graph representation to ground objects individually and uses a novel particle-based graph neural network to reason about object placements with uncertainty.

    我们提出了一种新的方法，PARsing And visual GrOuNding (ParaGon)，用于在物体放置任务中对自然语言进行定位。自然语言通常用组合性和歧义性描述对象和空间关系，这是有效语言定位的两个主要障碍。对于组合性，ParaGon将语言指令解析为以对象为中心的图形表示，以单独定位对象。对于歧义性，ParaGon使用一种新颖的基于粒子的图神经网络来推理关于带有不确定性的物体放置。本质上，ParaGon将解析算法集成到概率的数据驱动学习框架中。它是完全可微分的，并从数据中端到端地训练，以对抗复杂的，模糊的语言输入。

    We present a new method, PARsing And visual GrOuNding (ParaGon), for grounding natural language in object placement tasks. Natural language generally describes objects and spatial relations with compositionality and ambiguity, two major obstacles to effective language grounding. For compositionality, ParaGon parses a language instruction into an object-centric graph representation to ground objects individually. For ambiguity, ParaGon uses a novel particle-based graph neural network to reason about object placements with uncertainty. Essentially, ParaGon integrates a parsing algorithm into a probabilistic, data-driven learning framework. It is fully differentiable and trained end-to-end from data for robustness against complex, ambiguous language input.
    
[^11]: DM$^2$: 基于分布匹配的去中心化多智能体强化学习

    DM$^2$: Decentralized Multi-Agent Reinforcement Learning for Distribution Matching. (arXiv:2206.00233v3 [cs.MA] UPDATED)

    [http://arxiv.org/abs/2206.00233](http://arxiv.org/abs/2206.00233)

    本文提出了一种基于分布匹配的去中心化多智能体强化学习方法，每个智能体独立地最小化与目标访问分布的相应分量的分布不匹配，可以实现收敛到生成目标分布的联合策略。

    This paper proposes a decentralized multi-agent reinforcement learning method based on distribution matching, where each agent independently minimizes the distribution mismatch to the corresponding component of a target visitation distribution, achieving convergence to the joint policy that generated the target distribution.

    当前的多智能体协作方法往往依赖于集中式机制或显式通信协议以确保收敛。本文研究了分布匹配在不依赖于集中式组件或显式通信的分布式多智能体学习中的应用。在所提出的方案中，每个智能体独立地最小化与目标访问分布的相应分量的分布不匹配。理论分析表明，在某些条件下，每个智能体最小化其个体分布不匹配可以实现收敛到生成目标分布的联合策略。此外，如果目标分布来自优化合作任务的联合策略，则该任务奖励和分布匹配奖励的组合的最优策略是相同的联合策略。这一见解被用来制定一个实用的算法。

    Current approaches to multi-agent cooperation rely heavily on centralized mechanisms or explicit communication protocols to ensure convergence. This paper studies the problem of distributed multi-agent learning without resorting to centralized components or explicit communication. It examines the use of distribution matching to facilitate the coordination of independent agents. In the proposed scheme, each agent independently minimizes the distribution mismatch to the corresponding component of a target visitation distribution. The theoretical analysis shows that under certain conditions, each agent minimizing its individual distribution mismatch allows the convergence to the joint policy that generated the target distribution. Further, if the target distribution is from a joint policy that optimizes a cooperative task, the optimal policy for a combination of this task reward and the distribution matching reward is the same joint policy. This insight is used to formulate a practical al
    
[^12]: 使用物体感知表示在多物体场景中进行视觉运动控制

    Visuomotor Control in Multi-Object Scenes Using Object-Aware Representations. (arXiv:2205.06333v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2205.06333](http://arxiv.org/abs/2205.06333)

    本文探讨了使用物体感知表示学习技术进行机器人任务的有效性，以解决当前方法学习任务特定表示不能很好地转移到其他任务的问题，以及由监督方法学习的表示需要大量标记数据集的问题。

    This paper explores the effectiveness of using object-aware representation learning techniques for robotic tasks, to address the problem that current methodologies learn task specific representations that do not necessarily transfer well to other tasks, and that representations learned by supervised methods require large labeled datasets for each task that are expensive to collect in the real world.

    场景的感知理解以及其不同组件之间的关系对于成功完成机器人任务至关重要。表示学习已被证明是一种强大的技术，但大多数当前的方法学习任务特定的表示，不一定能够很好地转移到其他任务。此外，由监督方法学习的表示需要大量标记数据集，这在现实世界中收集起来很昂贵。使用自监督学习从未标记的数据中获取表示可以缓解这个问题。然而，当前的自监督表示学习方法大多是物体无关的，我们证明了由此得到的表示对于具有许多组件的场景的通用机器人任务是不足够的。在本文中，我们探讨了使用物体感知表示学习技术进行机器人任务的有效性。

    Perceptual understanding of the scene and the relationship between its different components is important for successful completion of robotic tasks. Representation learning has been shown to be a powerful technique for this, but most of the current methodologies learn task specific representations that do not necessarily transfer well to other tasks. Furthermore, representations learned by supervised methods require large labeled datasets for each task that are expensive to collect in the real world. Using self-supervised learning to obtain representations from unlabeled data can mitigate this problem. However, current self-supervised representation learning methods are mostly object agnostic, and we demonstrate that the resulting representations are insufficient for general purpose robotics tasks as they fail to capture the complexity of scenes with many components. In this paper, we explore the effectiveness of using object-aware representation learning techniques for robotic tasks. 
    
[^13]: 学习四足动物运动的扭矩控制

    Learning Torque Control for Quadrupedal Locomotion. (arXiv:2203.05194v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2203.05194](http://arxiv.org/abs/2203.05194)

    本文提出了一种基于扭矩的强化学习框架，直接预测关节扭矩，避免使用PD控制器，通过广泛的实验验证，四足动物能够穿越各种地形并抵抗外部干扰，同时保持运动。

    This paper proposes a torque-based reinforcement learning framework that directly predicts joint torques, avoiding the use of a PD controller. The framework is validated through extensive experiments, where a quadruped is capable of traversing various terrain and resisting external disturbances while maintaining locomotion.

    强化学习已成为开发四足机器人控制器的一种有前途的方法。传统上，用于运动的RL设计遵循基于位置的范例，其中RL策略以低频率输出目标关节位置，然后由高频比例-导数（PD）控制器跟踪以产生关节扭矩。相比之下，对于四足动物运动的基于模型的控制，已经从基于位置的控制范例转向基于扭矩的控制。鉴于基于模型的控制的最新进展，我们通过引入基于扭矩的RL框架，探索了一种替代基于位置的RL范例的方法，其中RL策略直接在高频率下预测关节扭矩，从而避免使用PD控制器。所提出的学习扭矩控制框架通过广泛的实验进行了验证，在这些实验中，四足动物能够穿越各种地形并抵抗外部干扰，同时保持运动。

    Reinforcement learning (RL) has become a promising approach to developing controllers for quadrupedal robots. Conventionally, an RL design for locomotion follows a position-based paradigm, wherein an RL policy outputs target joint positions at a low frequency that are then tracked by a high-frequency proportional-derivative (PD) controller to produce joint torques. In contrast, for the model-based control of quadrupedal locomotion, there has been a paradigm shift from position-based control to torque-based control. In light of the recent advances in model-based control, we explore an alternative to the position-based RL paradigm, by introducing a torque-based RL framework, where an RL policy directly predicts joint torques at a high frequency, thus circumventing the use of a PD controller. The proposed learning torque control framework is validated with extensive experiments, in which a quadruped is capable of traversing various terrain and resisting external disturbances while followi
    
[^14]: 学习在任务和动作规划中使用流进行搜索

    Learning to Search in Task and Motion Planning with Streams. (arXiv:2111.13144v5 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2111.13144](http://arxiv.org/abs/2111.13144)

    本文提出了一种几何信息的符号规划器，使用图神经网络优先级排序，以最佳优先方式扩展对象和事实集合，从而改善了在任务和动作规划中的长期推理能力。在7自由度机械臂的堆叠操纵任务中得到了应用。

    This paper proposes a geometrically informed symbolic planner that expands the set of objects and facts in a best-first manner, prioritized by a Graph Neural Network that is learned from prior search computations, improving the long-term reasoning ability in task and motion planning. The algorithm is applied to a 7DOF robotic arm in block-stacking manipulation tasks.

    机器人中的任务和动作规划问题将离散任务变量上的符号规划与连续状态和动作变量上的运动优化相结合。最近的作品，如PDDLStream，专注于乐观规划，使用逐步增长的对象集，直到找到可行的轨迹。然而，这个集合是以广度优先的方式穷举扩展的，而不考虑手头问题的逻辑和几何结构，这使得具有大量对象的长期推理变得耗时。为了解决这个问题，我们提出了一个几何信息的符号规划器，以最佳优先方式扩展对象和事实集合，由先前的搜索计算学习的图神经网络优先级排序。我们在各种问题上评估了我们的方法，并展示了在困难情况下规划的能力得到了改善。我们还将我们的算法应用于7自由度机械臂在堆叠操纵任务中。

    Task and motion planning problems in robotics combine symbolic planning over discrete task variables with motion optimization over continuous state and action variables. Recent works such as PDDLStream have focused on optimistic planning with an incrementally growing set of objects until a feasible trajectory is found. However, this set is exhaustively expanded in a breadth-first manner, regardless of the logical and geometric structure of the problem at hand, which makes long-horizon reasoning with large numbers of objects prohibitively time-consuming. To address this issue, we propose a geometrically informed symbolic planner that expands the set of objects and facts in a best-first manner, prioritized by a Graph Neural Network that is learned from prior search computations. We evaluate our approach on a diverse set of problems and demonstrate an improved ability to plan in difficult scenarios. We also apply our algorithm on a 7DOF robotic arm in block-stacking manipulation tasks.
    
[^15]: 训练过的人形机器人可以执行类人的跨模态社交关注和冲突解决

    A trained humanoid robot can perform human-like crossmodal social attention and conflict resolution. (arXiv:2111.01906v5 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2111.01906](http://arxiv.org/abs/2111.01906)

    本研究采用跨模态冲突解决的神经机器人范例，使机器人表现出类人的社交关注，为增强人机社交互动提供了新思路。

    This study adopts the neurorobotic paradigm of crossmodal conflict resolution to make a robot express human-like social attention, providing a new approach to enhance human-robot social interaction.

    为了增强人机社交互动，机器人在复杂的现实环境中处理多个社交线索至关重要。然而，跨模态输入信息的不一致性是不可避免的，这可能对机器人的处理造成挑战。为了解决这个问题，我们的研究采用了跨模态冲突解决的神经机器人范例，使机器人表现出类人的社交关注。我们对37名参与者进行了一项行为实验。我们设计了一个圆桌会议场景，有三个动画化的头像，以提高生态效度。每个头像都戴着医用口罩，遮盖了鼻子、嘴巴和下巴的面部线索。中央头像移动其眼睛注视，而外围头像则发出声音。凝视方向和声音位置要么是空间上一致的，要么是不一致的。我们观察到，中央头像的动态凝视可以触发跨模态社交关注反应。特别是，人类表现

    To enhance human-robot social interaction, it is essential for robots to process multiple social cues in a complex real-world environment. However, incongruency of input information across modalities is inevitable and could be challenging for robots to process. To tackle this challenge, our study adopted the neurorobotic paradigm of crossmodal conflict resolution to make a robot express human-like social attention. A behavioural experiment was conducted on 37 participants for the human study. We designed a round-table meeting scenario with three animated avatars to improve ecological validity. Each avatar wore a medical mask to obscure the facial cues of the nose, mouth, and jaw. The central avatar shifted its eye gaze while the peripheral avatars generated sound. Gaze direction and sound locations were either spatially congruent or incongruent. We observed that the central avatar's dynamic gaze could trigger crossmodal social attention responses. In particular, human performances are 
    

