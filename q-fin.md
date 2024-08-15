# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LLM Voting: Human Choices and AI Collective Decision Making](https://arxiv.org/abs/2402.01766) | 本文研究了大型语言模型（LLMs），特别是OpenAI的GPT4和LLaMA2的投票行为，并揭示了LLMs与人类在决策和偏见方面的差异。研究发现，在投票辅助中使用LLMs可能会导致更同质化的集体结果，强调了谨慎将LLMs整合到民主过程中的必要性。 |
| [^2] | [Limit Order Book Dynamics and Order Size Modelling Using Compound Hawkes Process](https://arxiv.org/abs/2312.08927) | 本文提出了一种使用复合霍克斯进程建模限价单簿动态和订单尺寸的新方法，以校准分布抽取每个事件的订单尺寸，并在模型中保持正的价差。进一步地，我们根据时间条件模型参数支持经验观察，并使用改进的非参数方法校准霍克斯核函数和抑制性交叉激发核函数。 |
| [^3] | [Many learning agents interacting with an agent-based market model.](http://arxiv.org/abs/2303.07393) | 本论文介绍了多个强化学习最优执行交易智能体与反应式基于智能体的金融市场模型的交互。通过平衡执行差价和未能及时执行订单的惩罚，说明了奖励函数的作用。研究表明，学习智能体的数量、初始订单大小和状态空间的变化，会对最小智能市场模拟造成不同的影响。 |

# 详细

[^1]: LLM投票：人类选择和AI集体决策

    LLM Voting: Human Choices and AI Collective Decision Making

    [https://arxiv.org/abs/2402.01766](https://arxiv.org/abs/2402.01766)

    本文研究了大型语言模型（LLMs），特别是OpenAI的GPT4和LLaMA2的投票行为，并揭示了LLMs与人类在决策和偏见方面的差异。研究发现，在投票辅助中使用LLMs可能会导致更同质化的集体结果，强调了谨慎将LLMs整合到民主过程中的必要性。

    

    本文研究了大型语言模型（LLMs），特别是OpenAI的GPT4和LLaMA2的投票行为，并与人类投票模式进行了对比。我们的方法包括进行人类投票实验以建立人类偏好的基准，并与LLM代理进行平行实验。研究聚焦于集体结果和个体偏好，揭示了人类和LLMs之间在决策和固有偏见方面的差异。我们观察到LLMs在偏好多样性和一致性之间存在权衡，相比人类选民的多样偏好，LLMs有更趋向于一致选择的倾向。这一发现表明，在投票辅助中使用LLMs可能会导致更同质化的集体结果，强调了谨慎将LLMs整合到民主过程中的必要性。

    This paper investigates the voting behaviors of Large Language Models (LLMs), particularly OpenAI's GPT4 and LLaMA2, and their alignment with human voting patterns. Our approach included a human voting experiment to establish a baseline for human preferences and a parallel experiment with LLM agents. The study focused on both collective outcomes and individual preferences, revealing differences in decision-making and inherent biases between humans and LLMs. We observed a trade-off between preference diversity and alignment in LLMs, with a tendency towards more uniform choices as compared to the diverse preferences of human voters. This finding indicates that LLMs could lead to more homogenized collective outcomes when used in voting assistance, underscoring the need for cautious integration of LLMs into democratic processes.
    
[^2]: 限价单簿动态与订单尺寸建模：复合霍克斯进程

    Limit Order Book Dynamics and Order Size Modelling Using Compound Hawkes Process

    [https://arxiv.org/abs/2312.08927](https://arxiv.org/abs/2312.08927)

    本文提出了一种使用复合霍克斯进程建模限价单簿动态和订单尺寸的新方法，以校准分布抽取每个事件的订单尺寸，并在模型中保持正的价差。进一步地，我们根据时间条件模型参数支持经验观察，并使用改进的非参数方法校准霍克斯核函数和抑制性交叉激发核函数。

    

    霍克斯进程已在文献中多种方式被用于模拟限价单簿动态，但往往仅关注事件间隔，而订单尺寸通常被假设为常数。我们提出了一种新颖的方法，使用复合霍克斯进程来模拟限价单簿，其中每个事件的订单尺寸来自校准分布。该方法以一种新颖的方式构建，使进程的价差始终保持正值。此外，我们根据时间条件模型参数以支持经验观察。我们使用改进的非参数方法来校准霍克斯核函数，并允许抑制性交叉激发核函数。我们展示了在纳斯达克交易所中一只股票的限价单簿上的结果和适度程度。

    Hawkes Process has been used to model Limit Order Book (LOB) dynamics in several ways in the literature however the focus has been limited to capturing the inter-event times while the order size is usually assumed to be constant. We propose a novel methodology of using Compound Hawkes Process for the LOB where each event has an order size sampled from a calibrated distribution. The process is formulated in a novel way such that the spread of the process always remains positive. Further, we condition the model parameters on time of day to support empirical observations. We make use of an enhanced non-parametric method to calibrate the Hawkes kernels and allow for inhibitory cross-excitation kernels. We showcase the results and quality of fits for an equity stock's LOB in the NASDAQ exchange.
    
[^3]: 多个学习智能体与基于智能体的市场模型的交互

    Many learning agents interacting with an agent-based market model. (arXiv:2303.07393v1 [q-fin.TR])

    [http://arxiv.org/abs/2303.07393](http://arxiv.org/abs/2303.07393)

    本论文介绍了多个强化学习最优执行交易智能体与反应式基于智能体的金融市场模型的交互。通过平衡执行差价和未能及时执行订单的惩罚，说明了奖励函数的作用。研究表明，学习智能体的数量、初始订单大小和状态空间的变化，会对最小智能市场模拟造成不同的影响。

    

    本文考虑了多个强化学习最优执行交易智能体与在事件时间下的反应式基于智能体的金融市场模型的动态和相互作用。模型代表了一个市场生态系统，由三个营养级别代表：最优执行学习智能体，最小智能的流动性需要者和快速的电子流动性提供者。最优执行代理类别包括买入和卖出代理，可以使用限价单和市价单的组合，或者仅使用市价单进行交易。奖励函数明确平衡了交易执行差价与未能及时执行订单的惩罚之间的关系。本文展示了多个竞争学习智能体如何随着智能体数量、初始订单的大小和用于学习的状态空间的函数影响最小智能市场模拟。我们使用相空间图来研究ABM的动态，当特定规范被应用

    We consider the dynamics and the interactions of multiple reinforcement learning optimal execution trading agents interacting with a reactive Agent-Based Model (ABM) of a financial market in event time. The model represents a market ecology with 3-trophic levels represented by: optimal execution learning agents, minimally intelligent liquidity takers, and fast electronic liquidity providers. The optimal execution agent classes include buying and selling agents that can either use a combination of limit orders and market orders, or only trade using market orders. The reward function explicitly balances trade execution slippage against the penalty of not executing the order timeously. This work demonstrates how multiple competing learning agents impact a minimally intelligent market simulation as functions of the number of agents, the size of agents' initial orders, and the state spaces used for learning. We use phase space plots to examine the dynamics of the ABM, when various specifica
    

