# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bandits with Abstention under Expert Advice](https://arxiv.org/abs/2402.14585) | 我们提出了CBA算法，其利用放弃参与游戏的假设获得了可以显著改进经典Exp4算法的奖励界限，成为首个对一般置信评级预测器的预期累积奖励实现界限的研究者，并在专家案例中实现了一种新颖的奖励界限。 |
| [^2] | [A Theoretical Analysis of Nash Learning from Human Feedback under General KL-Regularized Preference](https://arxiv.org/abs/2402.07314) | 本论文从理论层面分析了一种关于一般偏好下纳什学习从人类反馈中的方法，通过对两个竞争的LLM进行博弈来找到一种一致生成响应的策略。 |
| [^3] | [Signal reconstruction using determinantal sampling.](http://arxiv.org/abs/2310.09437) | 本研究提出了使用行列式抽样进行信号重建的方法，在有限数量的随机节点评估中近似表示方可积函数，实现了快速收敛和更高的适应性正则性。 |
| [^4] | [A Bayesian Framework for Causal Analysis of Recurrent Events in Presence of Immortal Risk.](http://arxiv.org/abs/2304.03247) | 论文提出了一种贝叶斯框架，针对错位处理问题，将其视为治疗切换问题，并通过概率模型解决了复增和末事件偏差的问题。 |

# 详细

[^1]: 具有弃权选项的专家建议下的赌徒问题

    Bandits with Abstention under Expert Advice

    [https://arxiv.org/abs/2402.14585](https://arxiv.org/abs/2402.14585)

    我们提出了CBA算法，其利用放弃参与游戏的假设获得了可以显著改进经典Exp4算法的奖励界限，成为首个对一般置信评级预测器的预期累积奖励实现界限的研究者，并在专家案例中实现了一种新颖的奖励界限。

    

    我们研究了在赌徒反馈下利用专家建议进行预测的经典问题。我们的模型假设一种行动，即学习者放弃参与游戏，在每次试验中都没有奖励或损失。我们提出了CBA算法，利用这一假设获得了可以显著改进经典Exp4算法的奖励界限。我们可以将我们的问题视为在学习者有放弃参与游戏选项时对置信评级预测器进行聚合。重要的是，我们是第一个对一般置信评级预测器的预期累积奖励实现界限的研究者。在专家案例中，我们实现了一种新颖的奖励界限，显著改进了之前在专家Exp（将弃权视为另一种行动）的边界。作为一个示例应用，我们讨论了在有限度量空间中学习球的并集。在这个上下文设置中，我们设计了CBA的有效实现，re

    arXiv:2402.14585v1 Announce Type: new  Abstract: We study the classic problem of prediction with expert advice under bandit feedback. Our model assumes that one action, corresponding to the learner's abstention from play, has no reward or loss on every trial. We propose the CBA algorithm, which exploits this assumption to obtain reward bounds that can significantly improve those of the classical Exp4 algorithm. We can view our problem as the aggregation of confidence-rated predictors when the learner has the option of abstention from play. Importantly, we are the first to achieve bounds on the expected cumulative reward for general confidence-rated predictors. In the special case of specialists we achieve a novel reward bound, significantly improving previous bounds of SpecialistExp (treating abstention as another action). As an example application, we discuss learning unions of balls in a finite metric space. In this contextual setting, we devise an efficient implementation of CBA, re
    
[^2]: 一种关于一般KL正则化偏好下纳什学习从人类反馈中的理论分析

    A Theoretical Analysis of Nash Learning from Human Feedback under General KL-Regularized Preference

    [https://arxiv.org/abs/2402.07314](https://arxiv.org/abs/2402.07314)

    本论文从理论层面分析了一种关于一般偏好下纳什学习从人类反馈中的方法，通过对两个竞争的LLM进行博弈来找到一种一致生成响应的策略。

    

    来自人类反馈的强化学习（RLHF）从一个概率偏好模型提供的偏好信号中学习，该模型以一个提示和两个响应作为输入，并产生一个分数，表示对一个响应相对于另一个响应的偏好程度。迄今为止，最流行的RLHF范式是基于奖励的，它从奖励建模的初始步骤开始，然后使用构建的奖励为后续的奖励优化阶段提供奖励信号。然而，奖励函数的存在是一个强假设，基于奖励的RLHF在表达能力上有局限性，不能捕捉到真实世界中复杂的人类偏好。在这项工作中，我们为最近提出的学习范式Nash学习从人类反馈（NLHF）提供了理论洞察力，该学习范式考虑了一个一般的偏好模型，并将对齐过程定义为两个竞争的LLM之间的博弈。学习目标是找到一个一致生成响应的策略。

    Reinforcement Learning from Human Feedback (RLHF) learns from the preference signal provided by a probabilistic preference model, which takes a prompt and two responses as input, and produces a score indicating the preference of one response against another. So far, the most popular RLHF paradigm is reward-based, which starts with an initial step of reward modeling, and the constructed reward is then used to provide a reward signal for the subsequent reward optimization stage. However, the existence of a reward function is a strong assumption and the reward-based RLHF is limited in expressivity and cannot capture the real-world complicated human preference.   In this work, we provide theoretical insights for a recently proposed learning paradigm, Nash learning from human feedback (NLHF), which considered a general preference model and formulated the alignment process as a game between two competitive LLMs. The learning objective is to find a policy that consistently generates responses
    
[^3]: 使用行列式抽样进行信号重建

    Signal reconstruction using determinantal sampling. (arXiv:2310.09437v1 [stat.ML])

    [http://arxiv.org/abs/2310.09437](http://arxiv.org/abs/2310.09437)

    本研究提出了使用行列式抽样进行信号重建的方法，在有限数量的随机节点评估中近似表示方可积函数，实现了快速收敛和更高的适应性正则性。

    

    我们研究了从随机节点的有限数量评估中近似表示一个方可积函数的问题，其中随机节点的选择依据是一个精心选择的分布。当函数被假设属于再生核希尔伯特空间（RKHS）时，这尤为相关。本研究提出了将基于两种可能的节点概率分布的几个自然有限维逼近方法相结合。这些概率分布与行列式点过程相关，并利用RKHS的核函数来优化在随机设计中的RKHS适应性正则性。虽然先前的行列式抽样工作依赖于RKHS范数，我们证明了在$L^2$范数下的均方保证。我们表明，行列式点过程及其混合体可以产生快速收敛速度。我们的结果还揭示了当假设更多的平滑性时收敛速度如何变化，这种现象被称为超收敛。此外，行列式抽样推广了从Christoffel函数进行i.i.d.抽样的方法。

    We study the approximation of a square-integrable function from a finite number of evaluations on a random set of nodes according to a well-chosen distribution. This is particularly relevant when the function is assumed to belong to a reproducing kernel Hilbert space (RKHS). This work proposes to combine several natural finite-dimensional approximations based two possible probability distributions of nodes. These distributions are related to determinantal point processes, and use the kernel of the RKHS to favor RKHS-adapted regularity in the random design. While previous work on determinantal sampling relied on the RKHS norm, we prove mean-square guarantees in $L^2$ norm. We show that determinantal point processes and mixtures thereof can yield fast convergence rates. Our results also shed light on how the rate changes as more smoothness is assumed, a phenomenon known as superconvergence. Besides, determinantal sampling generalizes i.i.d. sampling from the Christoffel function which is
    
[^4]: 一种在不可避免风险存在下进行复发事件因果分析的贝叶斯框架

    A Bayesian Framework for Causal Analysis of Recurrent Events in Presence of Immortal Risk. (arXiv:2304.03247v1 [stat.ME])

    [http://arxiv.org/abs/2304.03247](http://arxiv.org/abs/2304.03247)

    论文提出了一种贝叶斯框架，针对错位处理问题，将其视为治疗切换问题，并通过概率模型解决了复增和末事件偏差的问题。

    

    生物医学统计学中对复发事件率的观测研究很常见。通常的目标是在规定的随访时间窗口内，估计在一个明确定义的目标人群中两种治疗方法的事件率差异。使用观测性索赔数据进行估计是具有挑战性的，因为在目标人群的成员资格方面定义时，很少在资格确认时准确分配治疗方式。目前的解决方案通常是错位处理，比如基于后续分配，在资格确认时分配治疗方式，这会将先前的事件率错误地归因于治疗-从而产生不可避免的风险偏差。即使资格和治疗已经对齐，终止事件过程（例如死亡）也经常停止感兴趣的复发事件过程。同样，这两个过程也受到审查的影响，因此在整个随访时间窗口内不能观察到事件。我们的方法将错位处理转化为治疗切换问题：一些患者在整个随访时间窗口内坚持一个特定的治疗策略，另一些患者在这个时间窗口内经历治疗策略的切换。我们提出了一个概率模型，其中包括两个基本元素：通过一个合理的时刻切换模型，正确地建模治疗之间的切换和不可避免风险，通过将非观察事件模型化为复发事件模型，解决了复增和末事件偏差的问题。

    Observational studies of recurrent event rates are common in biomedical statistics. Broadly, the goal is to estimate differences in event rates under two treatments within a defined target population over a specified followup window. Estimation with observational claims data is challenging because while membership in the target population is defined in terms of eligibility criteria, treatment is rarely assigned exactly at the time of eligibility. Ad-hoc solutions to this timing misalignment, such as assigning treatment at eligibility based on subsequent assignment, incorrectly attribute prior event rates to treatment - resulting in immortal risk bias. Even if eligibility and treatment are aligned, a terminal event process (e.g. death) often stops the recurrent event process of interest. Both processes are also censored so that events are not observed over the entire followup window. Our approach addresses misalignment by casting it as a treatment switching problem: some patients are on
    

