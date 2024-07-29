# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reinforcement Learning with Hidden Markov Models for Discovering Decision-Making Dynamics.](http://arxiv.org/abs/2401.13929) | 本论文针对重性抑郁障碍(MDD)中的奖励处理异常，使用强化学习模型和隐马尔可夫模型结合，探索决策制定过程中的学习策略动态对个体奖励学习能力的影响。 |
| [^2] | [Anticipatory Music Transformer.](http://arxiv.org/abs/2306.08620) | 该论文提出一种预测音乐转换器，它能够实现在符号音乐生成的过程中进行控制，包括补全控制任务和伴奏，并且在大型且多样的数据集上表现出色。 |

# 详细

[^1]: 使用隐马尔可夫模型的强化学习来发现决策动态

    Reinforcement Learning with Hidden Markov Models for Discovering Decision-Making Dynamics. (arXiv:2401.13929v1 [cs.LG])

    [http://arxiv.org/abs/2401.13929](http://arxiv.org/abs/2401.13929)

    本论文针对重性抑郁障碍(MDD)中的奖励处理异常，使用强化学习模型和隐马尔可夫模型结合，探索决策制定过程中的学习策略动态对个体奖励学习能力的影响。

    

    由于其复杂和异质性，重性抑郁障碍(MDD)在诊断和治疗方面存在挑战。新的证据表明奖励处理异常可能作为MDD的行为标记。为了衡量奖励处理，患者执行涉及做出选择或对与不同结果相关联的刺激作出反应的基于计算机的行为任务。强化学习(RL)模型被拟合以提取衡量奖励处理各个方面的参数，以表征患者在行为任务中的决策方式。最近的研究发现，仅基于单个RL模型的奖励学习表征不足; 相反，决策过程中可能存在多种策略之间的切换。一个重要的科学问题是决策制定中学习策略的动态如何影响MDD患者的奖励学习能力。由概率奖励任务(PRT)所启发

    Major depressive disorder (MDD) presents challenges in diagnosis and treatment due to its complex and heterogeneous nature. Emerging evidence indicates that reward processing abnormalities may serve as a behavioral marker for MDD. To measure reward processing, patients perform computer-based behavioral tasks that involve making choices or responding to stimulants that are associated with different outcomes. Reinforcement learning (RL) models are fitted to extract parameters that measure various aspects of reward processing to characterize how patients make decisions in behavioral tasks. Recent findings suggest the inadequacy of characterizing reward learning solely based on a single RL model; instead, there may be a switching of decision-making processes between multiple strategies. An important scientific question is how the dynamics of learning strategies in decision-making affect the reward learning ability of individuals with MDD. Motivated by the probabilistic reward task (PRT) wi
    
[^2]: 预测音乐转换器

    Anticipatory Music Transformer. (arXiv:2306.08620v1 [cs.SD])

    [http://arxiv.org/abs/2306.08620](http://arxiv.org/abs/2306.08620)

    该论文提出一种预测音乐转换器，它能够实现在符号音乐生成的过程中进行控制，包括补全控制任务和伴奏，并且在大型且多样的数据集上表现出色。

    

    我们引入了anticipation（预测）：一种构建生成模型的方法，该模型基于事件过程（时间点过程）的实现，以异步地控制与第二个相关过程（控制过程）的相关性。我们通过交错事件和控件序列来实现这一目标，使控件出现在事件序列的停止时间之后。这项工作的动机来自符号音乐生成控制中出现的问题。我们专注于infiling（补全）控制任务，其中控制事件是事件本身的子集，并且条件生成完成给定固定控制事件的事件序列。我们使用大型多样的Lakh MIDI音乐数据集训练预测infiling模型。这些模型与提示音乐生成的自回归模型性能相当，并具有执行infilling控制任务的附加能力，包括伴奏。人工评估员报告说，预测模型产生的伴奏具有高可辨性和优美性。

    We introduce anticipation: a method for constructing a controllable generative model of a temporal point process (the event process) conditioned asynchronously on realizations of a second, correlated process (the control process). We achieve this by interleaving sequences of events and controls, such that controls appear following stopping times in the event sequence. This work is motivated by problems arising in the control of symbolic music generation. We focus on infilling control tasks, whereby the controls are a subset of the events themselves, and conditional generation completes a sequence of events given the fixed control events. We train anticipatory infilling models using the large and diverse Lakh MIDI music dataset. These models match the performance of autoregressive models for prompted music generation, with the additional capability to perform infilling control tasks, including accompaniment. Human evaluators report that an anticipatory model produces accompaniments with
    

