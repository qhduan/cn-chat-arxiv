# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Physics-Based Causal Reasoning for Safe & Robust Next-Best Action Selection in Robot Manipulation Tasks](https://arxiv.org/abs/2403.14488) | 该论文提出了一个基于物理因果推理的框架，用于机器人在部分可观察的环境中进行概率推理，成功预测积木塔稳定性并选择下一最佳动作。 |
| [^2] | [Large Intestine 3D Shape Refinement Using Point Diffusion Models for Digital Phantom Generation.](http://arxiv.org/abs/2309.08289) | 本研究利用几何深度学习和去噪扩散概率模型优化大肠的分割结果，并结合先进的表面重构模型，实现对大肠3D形状的精化恢复。 |
| [^3] | [Policy Expansion for Bridging Offline-to-Online Reinforcement Learning.](http://arxiv.org/abs/2302.00935) | 本文提出了一种政策扩展方案，用于离线到在线强化学习，以保留离线学习的策略并在在线学习中进行自适应扩展。 |

# 详细

[^1]: 基于物理学因果推理的机器人操作任务中安全稳健的下一最佳动作选择

    Physics-Based Causal Reasoning for Safe & Robust Next-Best Action Selection in Robot Manipulation Tasks

    [https://arxiv.org/abs/2403.14488](https://arxiv.org/abs/2403.14488)

    该论文提出了一个基于物理因果推理的框架，用于机器人在部分可观察的环境中进行概率推理，成功预测积木塔稳定性并选择下一最佳动作。

    

    安全高效的物体操作是许多真实世界机器人应用的关键推手。然而，这种挑战在于机器人操作必须对一系列传感器和执行器的不确定性具有稳健性。本文提出了一个基于物理知识和因果推理的框架，用于让机器人在部分可观察的环境中对候选动作进行概率推理，以完成一个积木堆叠任务。我们将刚体系统动力学的基于物理学的仿真与因果贝叶斯网络（CBN）结合起来，定义了机器人决策过程的因果生成概率模型。通过基于仿真的蒙特卡洛实验，我们展示了我们的框架成功地能够：(1) 高准确度地预测积木塔的稳定性（预测准确率：88.6%）；和，(2) 为积木堆叠任务选择一个近似的下一最佳动作，供整合的机器人系统执行，实现94.2%的任务成功率。

    arXiv:2403.14488v1 Announce Type: cross  Abstract: Safe and efficient object manipulation is a key enabler of many real-world robot applications. However, this is challenging because robot operation must be robust to a range of sensor and actuator uncertainties. In this paper, we present a physics-informed causal-inference-based framework for a robot to probabilistically reason about candidate actions in a block stacking task in a partially observable setting. We integrate a physics-based simulation of the rigid-body system dynamics with a causal Bayesian network (CBN) formulation to define a causal generative probabilistic model of the robot decision-making process. Using simulation-based Monte Carlo experiments, we demonstrate our framework's ability to successfully: (1) predict block tower stability with high accuracy (Pred Acc: 88.6%); and, (2) select an approximate next-best action for the block stacking task, for execution by an integrated robot system, achieving 94.2% task succe
    
[^2]: 利用点扩散模型对大肠的3D形状进行精化以生成数字幻影

    Large Intestine 3D Shape Refinement Using Point Diffusion Models for Digital Phantom Generation. (arXiv:2309.08289v1 [cs.CV])

    [http://arxiv.org/abs/2309.08289](http://arxiv.org/abs/2309.08289)

    本研究利用几何深度学习和去噪扩散概率模型优化大肠的分割结果，并结合先进的表面重构模型，实现对大肠3D形状的精化恢复。

    

    准确建模人体器官在构建虚拟成像试验的计算仿真中起着至关重要的作用。然而，从计算机断层扫描中生成解剖学上可信的器官表面重建仍然对人体结构中的许多器官来说是个挑战。在处理大肠时，这个挑战尤为明显。在这项研究中，我们利用几何深度学习和去噪扩散概率模型的最新进展来优化大肠分割结果。首先，我们将器官表示为从3D分割掩模表面采样得到的点云。随后，我们使用分层变分自编码器获得器官形状的全局和局部潜在表示。我们在分层潜在空间中训练两个条件去噪扩散模型来进行形状精化。为了进一步提高我们的方法，我们还结合了一种先进的表面重构模型，从而实现形状的更好恢复。

    Accurate 3D modeling of human organs plays a crucial role in building computational phantoms for virtual imaging trials. However, generating anatomically plausible reconstructions of organ surfaces from computed tomography scans remains challenging for many structures in the human body. This challenge is particularly evident when dealing with the large intestine. In this study, we leverage recent advancements in geometric deep learning and denoising diffusion probabilistic models to refine the segmentation results of the large intestine. We begin by representing the organ as point clouds sampled from the surface of the 3D segmentation mask. Subsequently, we employ a hierarchical variational autoencoder to obtain global and local latent representations of the organ's shape. We train two conditional denoising diffusion models in the hierarchical latent space to perform shape refinement. To further enhance our method, we incorporate a state-of-the-art surface reconstruction model, allowin
    
[^3]: 离线到在线强化学习的政策扩展

    Policy Expansion for Bridging Offline-to-Online Reinforcement Learning. (arXiv:2302.00935v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2302.00935](http://arxiv.org/abs/2302.00935)

    本文提出了一种政策扩展方案，用于离线到在线强化学习，以保留离线学习的策略并在在线学习中进行自适应扩展。

    

    利用离线数据进行预训练，并使用在线强化学习进行微调，是一种学习控制策略的有希望的策略，能够在样本效率和性能方面充分利用两者的优点。一种自然的方法是使用离线训练的策略初始化在线学习的策略。本文提出了一种用于此任务的政策扩展方案。在学习离线策略后，我们将其用作策略集中的一个候选策略。然后，我们通过另一个策略来扩展策略集，该策略将负责进一步的学习。两个策略将以自适应的方式组合起来与环境进行交互。通过这种方法，先前离线学习的策略完全在在线学习过程中得以保留，因此减轻了潜在问题，例如在在线学习的初始阶段破坏离线策略的有用行为，同时允许离线策略在自适应方式下自然参与

    Pre-training with offline data and online fine-tuning using reinforcement learning is a promising strategy for learning control policies by leveraging the best of both worlds in terms of sample efficiency and performance. One natural approach is to initialize the policy for online learning with the one trained offline. In this work, we introduce a policy expansion scheme for this task. After learning the offline policy, we use it as one candidate policy in a policy set. We then expand the policy set with another policy which will be responsible for further learning. The two policies will be composed in an adaptive manner for interacting with the environment. With this approach, the policy previously learned offline is fully retained during online learning, thus mitigating the potential issues such as destroying the useful behaviors of the offline policy in the initial stage of online learning while allowing the offline policy participate in the exploration naturally in an adaptive mann
    

