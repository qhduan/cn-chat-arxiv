# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning to Fold: prizewinning solution at LeHome Challenge 2026 (1st place online, 2nd offline)](https://arxiv.org/abs/2606.27163) | 本文提出了一种通过强化学习改进的视觉-语言-动作策略，实现了在线仿真和真实世界双臂衣物折叠的高性能，并整合了多种优化技术。 |
| [^2] | [Data-driven Machine Learning Cannot Reach Symbolic-level Logical Reasoning -- The Limit of the Scaling Law](https://arxiv.org/abs/2606.26454) | 本文证明了数据驱动的机器学习系统由于训练数据无法覆盖所有推理类型和端到端映射引入的矛盾目标，无法达到符号级逻辑推理，揭示了规模定律在逻辑推理领域的极限。 |
| [^3] | [CarbonNet: How Computer Vision Plays a Role in Climate Change? Application: Learning Geomechanics from Subsurface Geometry of CCS to Mitigate Global Warming](https://arxiv.org/abs/2403.06025) | 这项研究介绍了一种利用计算机视觉从地下储存空间几何图像中预测陆地表面位移的新方法，为碳捕集和封存项目中的决策提供支持。 |

# 详细

[^1]: 学习折叠：LeHome 2026挑战赛获奖方案（在线第一名，线下第二名）

    Learning to Fold: prizewinning solution at LeHome Challenge 2026 (1st place online, 2nd offline)

    [https://arxiv.org/abs/2606.27163](https://arxiv.org/abs/2606.27163)

    本文提出了一种通过强化学习改进的视觉-语言-动作策略，实现了在线仿真和真实世界双臂衣物折叠的高性能，并整合了多种优化技术。

    

    本文描述了我在LeHome 2026挑战赛（ICRA 2026双臂衣物折叠竞赛）中的解决方案。该系统在在线（仿真）环节中于62支队伍中排名第一，并在真实世界决赛中排名第二。它通过强化学习循环改进了视觉-语言-动作（VLA）策略。该策略本身即为价值函数：同一网络不仅预测动作，还预测成功、进度以及若干任务相关的未来量，这些预测用于优势估计、实时故障检测和候选选择。本工作主要将现有强化学习思想与工程和优化贡献相结合，这些贡献可作为整体配方或单独使用：结合AWR和RECAP用于流匹配VLA；通过HuggingFace Hub实现异步分布式训练/部署流水线；通过汤普森采样进行推理时超参数优化；以及包含相机对齐工具和强数据增强的仿真到现实迁移方案。

    arXiv:2606.27163v1 Announce Type: cross  Abstract: I describe my solution to the LeHome Challenge 2026, an ICRA 2026 competition on bimanual garment folding. The system placed 1st of 62 teams in the online (simulation) round and 2nd in the real-world final. It improves a vision-language-action (VLA) policy with a reinforcement-learning loop. The policy is its own value function: the same network that predicts actions also predicts success, progress, and a few task-relevant future quantities, and those predictions drive advantage estimation, live failure detection, and candidate selection. The work mostly recombines existing RL ideas with engineering and optimization contributions that can be used together as one recipe or individually: AWR + RECAP combined for flow-matching VLA; an asynchronous distributed training / rollout pipeline through HuggingFace Hub; inference-time hyperparameters optimization via Thompson sampling; a sim-to-real recipe with camera-alignment tooling, heavy augm
    
[^2]: 数据驱动的机器学习无法达到符号级逻辑推理——规模定律的极限

    Data-driven Machine Learning Cannot Reach Symbolic-level Logical Reasoning -- The Limit of the Scaling Law

    [https://arxiv.org/abs/2606.26454](https://arxiv.org/abs/2606.26454)

    本文证明了数据驱动的机器学习系统由于训练数据无法覆盖所有推理类型和端到端映射引入的矛盾目标，无法达到符号级逻辑推理，揭示了规模定律在逻辑推理领域的极限。

    

    arXiv:2606.26454v1 公告类型：新 摘要：球面神经网络在没有训练数据的情况下实现了符号级三段论推理，这引发了一个问题：逻辑推理的规模定律极限在哪里？即数据驱动的机器学习系统能否通过增加训练数据和训练时间达到同样的水平。我们展示了两个方法论上的局限性，阻碍了监督深度学习达到符号级三段论推理：（1）训练数据无法区分所有24种有效的三段论推理类型；（2）从前提到结论的端到端映射在模式识别和逻辑推理的神经组件之间引入了矛盾的训练目标。除了理论分析，我们还通过实验说明欧拉网络无法实现严谨的三段论推理。我们进一步挑战了最新的ChatGPTs（GPT-5-nano和GPT-5），以确定在四种表面形式下三段论语句的可满足性。

    arXiv:2606.26454v1 Announce Type: new  Abstract: Sphere neural networks have achieved symbolic level syllogistic reasoning without training data, raising the question of where the limit of the scaling law for logical reasoning lies, i.e., whether data-driven machine learning systems can achieve the same level by increasing training data and training time. We show two methodological limitations that prevent supervised deep learning from reaching the symbolic-level syllogistic reasoning: (1) training data can not distinguish all 24 types of valid syllogistic reasoning; (2) end-to-end mapping from premises to conclusion introduces contradictory training targets between neural components for pattern recognition and logical reasoning. Beside theoretical analysis, we experimentally illustrate that Euler Net cannot achieve rigorous syllogistic reasoning. We further challenge the most recent ChatGPTs (GPT-5-nano and GPT-5) to determine the satisfiability of syllogistic statements in four surfa
    
[^3]: CarbonNet: 计算机视觉在气候变化中的作用是什么？ 应用：学习从地下储存空间几何形状中减缓全球变暖的地质力学

    CarbonNet: How Computer Vision Plays a Role in Climate Change? Application: Learning Geomechanics from Subsurface Geometry of CCS to Mitigate Global Warming

    [https://arxiv.org/abs/2403.06025](https://arxiv.org/abs/2403.06025)

    这项研究介绍了一种利用计算机视觉从地下储存空间几何图像中预测陆地表面位移的新方法，为碳捕集和封存项目中的决策提供支持。

    

    我们介绍了一种新方法，使用计算机视觉从地下储存空间几何图像中预测陆地表面位移，以应用于碳捕集和封存（CCS）。CCS已被证明是碳中和社会的关键组成部分。然而，科学家发现存在挑战，包括由于大模型尺度而导致的高计算成本，以及难以泛化具有复杂物理学的预训练模型的限制。我们通过直接从地下储存空间几何图像训练模型来应对这些挑战。我们的目标是理解由碳注入导致的陆地表面位移响应，并利用我们训练的模型来为CCS项目的决策提供信息。

    arXiv:2403.06025v1 Announce Type: cross  Abstract: We introduce a new approach using computer vision to predict the land surface displacement from subsurface geometry images for Carbon Capture and Sequestration (CCS). CCS has been proved to be a key component for a carbon neutral society. However, scientists see there are challenges along the way including the high computational cost due to the large model scale and limitations to generalize a pre-trained model with complex physics. We tackle those challenges by training models directly from the subsurface geometry images. The goal is to understand the respons of land surface displacement due to carbon injection and utilize our trained models to inform decision making in CCS projects.   We implement multiple models (CNN, ResNet, and ResNetUNet) for static mechanics problem, which is a image prediction problem. Next, we use the LSTM and transformer for transient mechanics scenario, which is a video prediction problem. It shows ResNetUNe
    

