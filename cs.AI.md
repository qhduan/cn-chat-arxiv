# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [3D Human Pose Analysis via Diffusion Synthesis.](http://arxiv.org/abs/2401.08930) | 本文提出了一种名为PADS的框架，通过扩散合成过程学习姿势先验，解决3D人体姿势分析中的各种挑战，将多个姿势分析任务统一为逆问题的实例，验证了其性能和适应性。 |
| [^2] | [Enhancing Interpretability and Interactivity in Robot Manipulation: A Neurosymbolic Approach.](http://arxiv.org/abs/2210.00858) | 本文介绍了一种机器人操作的神经符号架构，可以通过自然语言指引机器人完成各种任务，利用共享的原始技能库以任务无关的方式解决所有情况。这将离散符号方法的可解释性和系统化泛化优势与可扩展性和代表性权力相结合。 |

# 详细

[^1]: 通过扩散合成进行3D人体姿势分析

    3D Human Pose Analysis via Diffusion Synthesis. (arXiv:2401.08930v1 [cs.CV])

    [http://arxiv.org/abs/2401.08930](http://arxiv.org/abs/2401.08930)

    本文提出了一种名为PADS的框架，通过扩散合成过程学习姿势先验，解决3D人体姿势分析中的各种挑战，将多个姿势分析任务统一为逆问题的实例，验证了其性能和适应性。

    

    扩散模型在生成建模方面取得了显著的成功。本文提出了一种名为PADS（通过扩散合成进行姿势分析）的新框架，旨在通过一个统一的流程解决3D人体姿势分析中的各种挑战。PADS的核心是两个独特的策略：i）使用扩散合成过程学习一个任务无关的姿势先验，从而有效地捕捉人体姿势数据中的运动约束；ii）将估计、补全、去噪等多个姿势分析任务统一为逆问题的实例。学习到的姿势先验将被视为对任务特定约束的正则化，通过一系列条件去噪步骤引导优化过程。PADS代表了首个基于扩散的框架，用于解决逆问题框架内的通用3D人体姿势分析。其性能已在不同基准测试上得到了验证，显示出其适应性和鲁棒性。

    Diffusion models have demonstrated remarkable success in generative modeling. In this paper, we propose PADS (Pose Analysis by Diffusion Synthesis), a novel framework designed to address various challenges in 3D human pose analysis through a unified pipeline. Central to PADS are two distinctive strategies: i) learning a task-agnostic pose prior using a diffusion synthesis process to effectively capture the kinematic constraints in human pose data, and ii) unifying multiple pose analysis tasks like estimation, completion, denoising, etc, as instances of inverse problems. The learned pose prior will be treated as a regularization imposing on task-specific constraints, guiding the optimization process through a series of conditional denoising steps. PADS represents the first diffusion-based framework for tackling general 3D human pose analysis within the inverse problem framework. Its performance has been validated on different benchmarks, signaling the adaptability and robustness of this
    
[^2]: 增强机器人操作的可解释性和互动性：一种神经符号方法

    Enhancing Interpretability and Interactivity in Robot Manipulation: A Neurosymbolic Approach. (arXiv:2210.00858v3 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2210.00858](http://arxiv.org/abs/2210.00858)

    本文介绍了一种机器人操作的神经符号架构，可以通过自然语言指引机器人完成各种任务，利用共享的原始技能库以任务无关的方式解决所有情况。这将离散符号方法的可解释性和系统化泛化优势与可扩展性和代表性权力相结合。

    

    本文提出了一种神经符号架构，用于将语言引导的视觉推理与机器人操作相结合。非专业人士可以使用自然语言引导机器人，提供指代表达式（REF）、问题（VQA）或抓握动作指令。该系统通过利用共享的原始技能库以任务无关的方式解决所有情况。每个原始技能都处理一个独立的子任务，例如推理视觉属性、空间关系理解、逻辑和枚举以及手臂控制。语言解析器将输入查询映射到由这些原语组成的可执行程序上，具体取决于上下文。尽管有些原语是纯符号操作（例如计数），但另一些是可训练的神经函数（例如视觉接地），因此融合了离散符号方法的可解释性和系统化泛化优势与可扩展性和再现性的代表性权力。

    In this paper we present a neurosymbolic architecture for coupling language-guided visual reasoning with robot manipulation. A non-expert human user can prompt the robot using unconstrained natural language, providing a referring expression (REF), a question (VQA), or a grasp action instruction. The system tackles all cases in a task-agnostic fashion through the utilization of a shared library of primitive skills. Each primitive handles an independent sub-task, such as reasoning about visual attributes, spatial relation comprehension, logic and enumeration, as well as arm control. A language parser maps the input query to an executable program composed of such primitives, depending on the context. While some primitives are purely symbolic operations (e.g. counting), others are trainable neural functions (e.g. visual grounding), therefore marrying the interpretability and systematic generalization benefits of discrete symbolic approaches with the scalability and representational power o
    

