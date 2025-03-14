# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CMP: Cooperative Motion Prediction with Multi-Agent Communication](https://arxiv.org/abs/2403.17916) | 该论文提出了一种名为CMP的方法，利用LiDAR信号作为输入，通过合作感知和运动预测模块共享信息，解决了合作运动预测的问题。 |
| [^2] | [Adaptive Split Learning over Energy-Constrained Wireless Edge Networks](https://arxiv.org/abs/2403.05158) | 设计了一种在无线边缘网络中为设备动态选择分裂点并为服务器分配计算资源的自适应分裂学习方案，以最小化平均训练延迟为目标，并提出了一种名为OPEN的在线算法解决此问题。 |
| [^3] | [Preventing Reward Hacking with Occupancy Measure Regularization](https://arxiv.org/abs/2403.03185) | 用占用度测量正则化方法可以有效防止奖励欺骗，通过考虑代理与真实奖励之间大的状态占用度偏差来避免潜在的灾难后果。 |
| [^4] | [Non-autoregressive Sequence-to-Sequence Vision-Language Models](https://arxiv.org/abs/2403.02249) | 提出了一种非自回归序列到序列视觉语言模型，通过在解码器中边际化多个推理路径的方式，实现了对标记的联合分布建模，从而在保持性能的同时加快了推理速度。 |
| [^5] | [Risk-reducing design and operations toolkit: 90 strategies for managing risk and uncertainty in decision problems.](http://arxiv.org/abs/2309.03133) | 该论文研究了风险降低设计和运营工具包（RDOT）中的90种策略，这些策略可在高度不确定的决策问题中提供有效的响应。这些策略包括将稳健性纳入设计、事后预防措施等，能够帮助工程师、公共规划者和其他决策者应对挑战。 |
| [^6] | [Networked Communication for Decentralised Agents in Mean-Field Games.](http://arxiv.org/abs/2306.02766) | 本研究在均场博弈中引入网络通信，提出了一种提高分布式智能体学习效率的方案，并进行了实际实验验证。 |

# 详细

[^1]: CMP：具有多智能体通信的合作运动预测

    CMP: Cooperative Motion Prediction with Multi-Agent Communication

    [https://arxiv.org/abs/2403.17916](https://arxiv.org/abs/2403.17916)

    该论文提出了一种名为CMP的方法，利用LiDAR信号作为输入，通过合作感知和运动预测模块共享信息，解决了合作运动预测的问题。

    

    随着自动驾驶车辆（AVs）的发展和车联网（V2X）通信的成熟，合作连接的自动化车辆（CAVs）的功能变得可能。本文基于合作感知，探讨了合作运动预测的可行性和有效性。我们的方法CMP以LiDAR信号作为输入，以增强跟踪和预测能力。与过去专注于合作感知或运动预测的工作不同，我们的框架是我们所知的第一个解决CAVs在感知和预测模块中共享信息的统一问题。我们的设计中还融入了能够容忍现实V2X带宽限制和传输延迟的独特能力，同时处理庞大的感知表示。我们还提出了预测聚合模块，统一了预测

    arXiv:2403.17916v1 Announce Type: cross  Abstract: The confluence of the advancement of Autonomous Vehicles (AVs) and the maturity of Vehicle-to-Everything (V2X) communication has enabled the capability of cooperative connected and automated vehicles (CAVs). Building on top of cooperative perception, this paper explores the feasibility and effectiveness of cooperative motion prediction. Our method, CMP, takes LiDAR signals as input to enhance tracking and prediction capabilities. Unlike previous work that focuses separately on either cooperative perception or motion prediction, our framework, to the best of our knowledge, is the first to address the unified problem where CAVs share information in both perception and prediction modules. Incorporated into our design is the unique capability to tolerate realistic V2X bandwidth limitations and transmission delays, while dealing with bulky perception representations. We also propose a prediction aggregation module, which unifies the predict
    
[^2]: 能量受限的无线边缘网络中的自适应分裂学习

    Adaptive Split Learning over Energy-Constrained Wireless Edge Networks

    [https://arxiv.org/abs/2403.05158](https://arxiv.org/abs/2403.05158)

    设计了一种在无线边缘网络中为设备动态选择分裂点并为服务器分配计算资源的自适应分裂学习方案，以最小化平均训练延迟为目标，并提出了一种名为OPEN的在线算法解决此问题。

    

    分裂学习（SL）是一种有希望的用于训练人工智能（AI）模型的方法，其中设备与服务器合作以分布式方式训练AI模型，基于相同的固定分裂点。然而，由于设备的异构性和信道条件的变化，这种方式在训练延迟和能量消耗方面并不是最优的。在本文中，我们设计了一种自适应分裂学习（ASL）方案，可以在无线边缘网络中为设备动态选择分裂点，并为服务器分配计算资源。我们制定了一个优化问题，旨在在满足长期能量消耗约束的情况下最小化平均训练延迟。解决这个问题的困难在于缺乏未来信息和混合整数规划（MIP）。为了解决这个问题，我们提出了一种利用Lyapunov理论的在线算法，名为OPEN，它将其分解为一个具有当前的新MIP问题。

    arXiv:2403.05158v1 Announce Type: cross  Abstract: Split learning (SL) is a promising approach for training artificial intelligence (AI) models, in which devices collaborate with a server to train an AI model in a distributed manner, based on a same fixed split point. However, due to the device heterogeneity and variation of channel conditions, this way is not optimal in training delay and energy consumption. In this paper, we design an adaptive split learning (ASL) scheme which can dynamically select split points for devices and allocate computing resource for the server in wireless edge networks. We formulate an optimization problem to minimize the average training latency subject to long-term energy consumption constraint. The difficulties in solving this problem are the lack of future information and mixed integer programming (MIP). To solve it, we propose an online algorithm leveraging the Lyapunov theory, named OPEN, which decomposes it into a new MIP problem only with the curren
    
[^3]: 用占用度测量正则化防止奖励欺骗

    Preventing Reward Hacking with Occupancy Measure Regularization

    [https://arxiv.org/abs/2403.03185](https://arxiv.org/abs/2403.03185)

    用占用度测量正则化方法可以有效防止奖励欺骗，通过考虑代理与真实奖励之间大的状态占用度偏差来避免潜在的灾难后果。

    

    当代理根据一个“代理”奖励函数（可能是手动指定或学习的）表现出色，但相对于未知的真实奖励却表现糟糕时，就会发生奖励欺骗。由于确保代理和真实奖励之间良好对齐极为困难，预防奖励欺骗的一种方法是保守地优化代理。以往的研究特别关注于通过惩罚他们的行为分布之间的KL散度来强制让学习到的策略表现类似于“安全”策略。然而，行为分布的正则化并不总是有效，因为在单个状态下行为分布的微小变化可能导致潜在的灾难性后果，而较大的变化可能并不代表任何危险活动。我们的见解是，当奖励欺骗时，代理访问的状态与安全策略达到的状态截然不同，导致状态占用度的巨大偏差。

    arXiv:2403.03185v1 Announce Type: cross  Abstract: Reward hacking occurs when an agent performs very well with respect to a "proxy" reward function (which may be hand-specified or learned), but poorly with respect to the unknown true reward. Since ensuring good alignment between the proxy and true reward is extremely difficult, one approach to prevent reward hacking is optimizing the proxy conservatively. Prior work has particularly focused on enforcing the learned policy to behave similarly to a "safe" policy by penalizing the KL divergence between their action distributions (AD). However, AD regularization doesn't always work well since a small change in action distribution at a single state can lead to potentially calamitous outcomes, while large changes might not be indicative of any dangerous activity. Our insight is that when reward hacking, the agent visits drastically different states from those reached by the safe policy, causing large deviations in state occupancy measure (OM
    
[^4]: 非自回归序列到序列视觉语言模型

    Non-autoregressive Sequence-to-Sequence Vision-Language Models

    [https://arxiv.org/abs/2403.02249](https://arxiv.org/abs/2403.02249)

    提出了一种非自回归序列到序列视觉语言模型，通过在解码器中边际化多个推理路径的方式，实现了对标记的联合分布建模，从而在保持性能的同时加快了推理速度。

    

    序列到序列的视觉语言模型表现出了潜力，但由于它们生成预测的自回归方式，它们的推理延迟限制了它们的适用性。我们提出了一个并行解码的序列到序列视觉语言模型，使用Query-CTC损失进行训练，在解码器中边际化多个推理路径。这使我们能够对标记的联合分布进行建模，而不像自回归模型那样限制在条件分布上。结果模型NARVL在推理时间上达到了与最新自回归对应物相当的性能，但更快，从与顺序生成标记相关的线性复杂度减少到常量时间联合推理的范式。

    arXiv:2403.02249v1 Announce Type: cross  Abstract: Sequence-to-sequence vision-language models are showing promise, but their applicability is limited by their inference latency due to their autoregressive way of generating predictions. We propose a parallel decoding sequence-to-sequence vision-language model, trained with a Query-CTC loss, that marginalizes over multiple inference paths in the decoder. This allows us to model the joint distribution of tokens, rather than restricting to conditional distribution as in an autoregressive model. The resulting model, NARVL, achieves performance on-par with its state-of-the-art autoregressive counterpart, but is faster at inference time, reducing from the linear complexity associated with the sequential generation of tokens to a paradigm of constant time joint inference.
    
[^5]: 风险降低设计和运营工具包: 管理决策问题中的风险和不确定性的90种策略

    Risk-reducing design and operations toolkit: 90 strategies for managing risk and uncertainty in decision problems. (arXiv:2309.03133v1 [q-fin.RM])

    [http://arxiv.org/abs/2309.03133](http://arxiv.org/abs/2309.03133)

    该论文研究了风险降低设计和运营工具包（RDOT）中的90种策略，这些策略可在高度不确定的决策问题中提供有效的响应。这些策略包括将稳健性纳入设计、事后预防措施等，能够帮助工程师、公共规划者和其他决策者应对挑战。

    

    不确定性是决策分析中普遍存在的挑战，决策理论承认两类解决方案: 概率模型和认知启发式。然而，工程师、公共规划者和其他决策者使用的是另一类被称为RDOT（风险降低设计和运营工具包）的策略。这些策略包括将稳健性纳入设计、事后预防措施等，并不属于概率模型或认知启发式的类别。此外，相同的策略出现在多个领域和学科中，指向了一个重要的共享工具包。本文的重点是开发这些策略的目录并为其开发一个框架。本文发现了超过90个属于六个广泛类别的这样的策略，并认为它们对于由于高度不确定性而似乎棘手的决策问题提供了高效的响应。然后，本文提出了一个将它们纳入决策模型的框架。

    Uncertainty is a pervasive challenge in decision analysis, and decision theory recognizes two classes of solutions: probabilistic models and cognitive heuristics. However, engineers, public planners and other decision-makers instead use a third class of strategies that could be called RDOT (Risk-reducing Design and Operations Toolkit). These include incorporating robustness into designs, contingency planning, and others that do not fall into the categories of probabilistic models or cognitive heuristics. Moreover, identical strategies appear in several domains and disciplines, pointing to an important shared toolkit.  The focus of this paper is to develop a catalog of such strategies and develop a framework for them. The paper finds more than 90 examples of such strategies falling into six broad categories and argues that they provide an efficient response to decision problems that are seemingly intractable due to high uncertainty. It then proposes a framework to incorporate them into 
    
[^6]: 分布式智能体在均场博弈中的网络通信

    Networked Communication for Decentralised Agents in Mean-Field Games. (arXiv:2306.02766v2 [cs.MA] UPDATED)

    [http://arxiv.org/abs/2306.02766](http://arxiv.org/abs/2306.02766)

    本研究在均场博弈中引入网络通信，提出了一种提高分布式智能体学习效率的方案，并进行了实际实验验证。

    

    我们将网络通信引入均场博弈框架，特别是在无oracle的情况下，N个分布式智能体沿着经过的经验系统的单一非周期演化路径学习。我们证明，我们的架构在只有一些关于网络结构的合理假设的情况下，具有样本保证，在集中学习和独立学习情况之间有界。我们讨论了三个理论算法的样本保证实际上并不会导致实际收敛。因此，我们展示了在实际设置中，当理论参数未被观察到（导致Q函数的估计不准确）时，我们的通信方案显著加速了收敛速度，而无需依赖于一个不可取的集中式控制器的假设。我们对三个理论算法进行了几种实际的改进，使我们能够展示它们的第一个实证表现。

    We introduce networked communication to the mean-field game framework, in particular to oracle-free settings where $N$ decentralised agents learn along a single, non-episodic evolution path of the empirical system. We prove that our architecture, with only a few reasonable assumptions about network structure, has sample guarantees bounded between those of the centralised- and independent-learning cases. We discuss how the sample guarantees of the three theoretical algorithms do not actually result in practical convergence. Accordingly, we show that in practical settings where the theoretical parameters are not observed (leading to poor estimation of the Q-function), our communication scheme significantly accelerates convergence over the independent case, without relying on the undesirable assumption of a centralised controller. We contribute several further practical enhancements to all three theoretical algorithms, allowing us to showcase their first empirical demonstrations. Our expe
    

