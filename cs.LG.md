# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Log Neural Controlled Differential Equations: The Lie Brackets Make a Difference](https://arxiv.org/abs/2402.18512) | Log-NCDEs是一种新颖而有效的训练NCDEs的方法，通过引入Log-ODE方法从粗糙路径研究中近似CDE的解，并在多变量时间序列分类基准上表现出比其他模型更高的准确率。 |
| [^2] | [Multi Task Inverse Reinforcement Learning for Common Sense Reward](https://arxiv.org/abs/2402.11367) | 将奖励分解为任务特定奖励和常识奖励，探索如何从专家演示中学习常识奖励；研究发现，逆强化学习成功训练代理后，并不会学到有用的奖励函数。 |
| [^3] | [Optimal signal propagation in ResNets through residual scaling.](http://arxiv.org/abs/2305.07715) | 本文为ResNets导出系统的有限尺寸理论，指出对于深层网络架构，缩放参数是优化信号传播和确保有效利用网络深度方面的关键。 |

# 详细

[^1]: Log神经控制微分方程：李括号的差异

    Log Neural Controlled Differential Equations: The Lie Brackets Make a Difference

    [https://arxiv.org/abs/2402.18512](https://arxiv.org/abs/2402.18512)

    Log-NCDEs是一种新颖而有效的训练NCDEs的方法，通过引入Log-ODE方法从粗糙路径研究中近似CDE的解，并在多变量时间序列分类基准上表现出比其他模型更高的准确率。

    

    受控微分方程（CDE）的矢量场描述了控制路径与解路径演化之间的关系。神经CDE（NCDE）将时间序列数据视为对控制路径的观测，使用神经网络对CDE的矢量场进行参数化，并将解路径作为持续演化的隐藏状态。由于其构造使其能够抵抗不规则采样率，NCDE是建模现实世界数据的强大方法。在神经粗糙微分方程（NRDE）的基础上，我们引入了Log-NCDE，这是一种训练NCDE的新颖且有效的方法。Log-NCDE的核心组件是Log-ODE方法，这是从粗糙路径研究中的一种用于近似CDE解的工具。在一系列多变量时间序列分类基准上，展示了Log-NCDE比NCDE，NRDE和两种最先进模型S5和线性递归模型具有更高的平均测试集准确率。

    arXiv:2402.18512v1 Announce Type: new  Abstract: The vector field of a controlled differential equation (CDE) describes the relationship between a control path and the evolution of a solution path. Neural CDEs (NCDEs) treat time series data as observations from a control path, parameterise a CDE's vector field using a neural network, and use the solution path as a continuously evolving hidden state. As their formulation makes them robust to irregular sampling rates, NCDEs are a powerful approach for modelling real-world data. Building on neural rough differential equations (NRDEs), we introduce Log-NCDEs, a novel and effective method for training NCDEs. The core component of Log-NCDEs is the Log-ODE method, a tool from the study of rough paths for approximating a CDE's solution. On a range of multivariate time series classification benchmarks, Log-NCDEs are shown to achieve a higher average test set accuracy than NCDEs, NRDEs, and two state-of-the-art models, S5 and the linear recurren
    
[^2]: 多任务逆强化学习用于常识奖励

    Multi Task Inverse Reinforcement Learning for Common Sense Reward

    [https://arxiv.org/abs/2402.11367](https://arxiv.org/abs/2402.11367)

    将奖励分解为任务特定奖励和常识奖励，探索如何从专家演示中学习常识奖励；研究发现，逆强化学习成功训练代理后，并不会学到有用的奖励函数。

    

    在将强化学习应用于复杂的现实环境中，一个挑战在于为agent提供足够详细的奖励函数。奖励与期望行为之间的不一致可能导致意外结果，如“奖励篡改”，agent通过意外行为最大化奖励。本文提出将奖励分解为两个明确部分：一个简单的任务特定奖励，概述了当前任务的细节；以及一个未知的常识奖励，指示agent在环境中的预期行为。我们探讨了如何从专家演示中学习这种常识奖励。我们首先展示，即使逆强化学习成功训练了一个代理，也不会学到一个有用的奖励函数。也就是说，使用学到的奖励训练新代理不会影响期望行为。

    arXiv:2402.11367v1 Announce Type: new  Abstract: One of the challenges in applying reinforcement learning in a complex real-world environment lies in providing the agent with a sufficiently detailed reward function. Any misalignment between the reward and the desired behavior can result in unwanted outcomes. This may lead to issues like "reward hacking" where the agent maximizes rewards by unintended behavior. In this work, we propose to disentangle the reward into two distinct parts. A simple task-specific reward, outlining the particulars of the task at hand, and an unknown common-sense reward, indicating the expected behavior of the agent within the environment. We then explore how this common-sense reward can be learned from expert demonstrations. We first show that inverse reinforcement learning, even when it succeeds in training an agent, does not learn a useful reward function. That is, training a new agent with the learned reward does not impair the desired behaviors. We then d
    
[^3]: 通过残差缩放实现ResNets的信号最优传递

    Optimal signal propagation in ResNets through residual scaling. (arXiv:2305.07715v1 [cond-mat.dis-nn])

    [http://arxiv.org/abs/2305.07715](http://arxiv.org/abs/2305.07715)

    本文为ResNets导出系统的有限尺寸理论，指出对于深层网络架构，缩放参数是优化信号传播和确保有效利用网络深度方面的关键。

    

    Residual网络（ResNets）在大深度上比前馈神经网络具有更好的训练能力和性能。引入跳过连接可以促进信号向更深层的传递。此外，先前的研究发现为残差分支添加缩放参数可以进一步提高泛化性能。尽管他们经验性地确定了这种缩放参数特别有利的取值范围，但其相关的性能提升及其在网络超参数上的普适性仍需要进一步理解。对于前馈神经网络（FFNets），有限尺寸理论在信号传播和超参数调节方面获得了重要洞见。我们在这里为ResNets导出了一个系统的有限尺寸理论，以研究信号传播及其对残差分支缩放的依赖性。我们导出响应函数的分析表达式，这是衡量网络对输入敏感性的一种指标，并表明对于深层网络架构，缩放参数在优化信号传播和确保有效利用网络深度方面发挥着至关重要的作用。

    Residual networks (ResNets) have significantly better trainability and thus performance than feed-forward networks at large depth. Introducing skip connections facilitates signal propagation to deeper layers. In addition, previous works found that adding a scaling parameter for the residual branch further improves generalization performance. While they empirically identified a particularly beneficial range of values for this scaling parameter, the associated performance improvement and its universality across network hyperparameters yet need to be understood. For feed-forward networks (FFNets), finite-size theories have led to important insights with regard to signal propagation and hyperparameter tuning. We here derive a systematic finite-size theory for ResNets to study signal propagation and its dependence on the scaling for the residual branch. We derive analytical expressions for the response function, a measure for the network's sensitivity to inputs, and show that for deep netwo
    

