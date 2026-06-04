# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Transformer-based models and hardware acceleration analysis in autonomous driving: A survey.](http://arxiv.org/abs/2304.10891) | 本文综述了基于Transformer的模型在自动驾驶中的应用，探讨了不同体系结构和运算符的优缺点，重点讨论了针对便携计算平台的硬件加速方案，并对卷积神经网络和Transformer的层进行了对比。 |
| [^2] | [Blessing from Human-AI Interaction: Super Reinforcement Learning in Confounded Environments.](http://arxiv.org/abs/2209.15448) | 本文介绍了一种新的强化学习范式——超级强化学习，它通过人工智能与人类的互动来实现数据驱动的顺序决策。在决策过程中，利用过去代理的行为可以提供有关未披露信息的洞见。通过以合法的方式将这些信息纳入策略搜索中，超级强化学习将得到一个在性能上优于标准最优策略和行为策略的超级策略。我们将这个更强大的神谕称为人工智能与人类互动的福音。 |

# 详细

[^1]: 自动驾驶中基于Transformer的模型及其硬件加速分析：综述 (arXiv:2304.10891v1 [cs.LG])

    Transformer-based models and hardware acceleration analysis in autonomous driving: A survey. (arXiv:2304.10891v1 [cs.LG])

    [http://arxiv.org/abs/2304.10891](http://arxiv.org/abs/2304.10891)

    本文综述了基于Transformer的模型在自动驾驶中的应用，探讨了不同体系结构和运算符的优缺点，重点讨论了针对便携计算平台的硬件加速方案，并对卷积神经网络和Transformer的层进行了对比。

    

    近年来，Transformer架构在各种自动驾驶应用中表现出了很好的性能。另一方面，将其专门用于便携式计算平台的硬件加速已成为实际部署在真实自动汽车中的下一步关键步骤。本综述论文提供了针对自动驾驶任务的基于Transformer的模型的全面概述、基准和分析，例如车道检测、分割、跟踪、规划和决策制定。我们审查了不同的体系结构，用于组织Transformer的输入和输出，例如编码器-解码器和仅编码器结构，并探讨了它们各自的优缺点。此外，我们深入讨论了Transformer相关的运算符及其硬件加速方案，考虑到关键因素，如量化和运行时。我们特别在移动和桌面平台上对卷积神经网络的层与基于Transformer的模型的运算符进行了对比。总的来说，本综述论文为研究人员和从业者提供了系统的指南，以了解基于Transformer的模型及其在自动驾驶中的硬件加速的当前进展和挑战。

    Transformer architectures have exhibited promising performance in various autonomous driving applications in recent years. On the other hand, its dedicated hardware acceleration on portable computational platforms has become the next critical step for practical deployment in real autonomous vehicles. This survey paper provides a comprehensive overview, benchmark, and analysis of Transformer-based models specifically tailored for autonomous driving tasks such as lane detection, segmentation, tracking, planning, and decision-making. We review different architectures for organizing Transformer inputs and outputs, such as encoder-decoder and encoder-only structures, and explore their respective advantages and disadvantages. Furthermore, we discuss Transformer-related operators and their hardware acceleration schemes in depth, taking into account key factors such as quantization and runtime. We specifically illustrate the operator level comparison between layers from convolutional neural ne
    
[^2]: 人工智能与人类互动的福音：在混杂环境中的超级强化学习

    Blessing from Human-AI Interaction: Super Reinforcement Learning in Confounded Environments. (arXiv:2209.15448v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2209.15448](http://arxiv.org/abs/2209.15448)

    本文介绍了一种新的强化学习范式——超级强化学习，它通过人工智能与人类的互动来实现数据驱动的顺序决策。在决策过程中，利用过去代理的行为可以提供有关未披露信息的洞见。通过以合法的方式将这些信息纳入策略搜索中，超级强化学习将得到一个在性能上优于标准最优策略和行为策略的超级策略。我们将这个更强大的神谕称为人工智能与人类互动的福音。

    

    随着人工智能在社会中的普及，有效地整合人类和人工智能系统，发挥各自的优势并减少风险已成为一个重要的优先事项。在本文中，我们介绍了利用人工智能与人类互动的超级强化学习范式，用于数据驱动的顺序决策。该方法利用观察到的行为（来自人工智能或人类）作为决策者（人类或人工智能）策略学习的更强大的神谕输入。在存在未测量混杂的决策过程中，过去代理的行为可以提供有关未披露信息的宝贵见解。通过以一种新颖和合法的方式将这些信息包括在策略搜索中，所提出的超级强化学习将产生一个管保能在标准最优策略和行为策略（例如过去代理的行为）之上表现更好的超级策略。我们将这个更强大的神谕称为来自人工智能与人类互动的福音。

    As AI becomes more prevalent throughout society, effective methods of integrating humans and AI systems that leverage their respective strengths and mitigate risk have become an important priority. In this paper, we introduce the paradigm of super reinforcement learning that takes advantage of Human-AI interaction for data driven sequential decision making. This approach utilizes the observed action, either from AI or humans, as input for achieving a stronger oracle in policy learning for the decision maker (humans or AI). In the decision process with unmeasured confounding, the actions taken by past agents can offer valuable insights into undisclosed information. By including this information for the policy search in a novel and legitimate manner, the proposed super reinforcement learning will yield a super-policy that is guaranteed to outperform both the standard optimal policy and the behavior one (e.g., past agents' actions). We call this stronger oracle a blessing from human-AI in
    

