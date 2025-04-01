# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Survey on Large Language Model-Based Game Agents](https://arxiv.org/abs/2404.02039) | LLM和MLLM的进步为游戏智能体提供了强大的人类决策能力，本文全面综述了基于LLM的游戏智能体的概念架构、方法论和未来研究方向 |
| [^2] | [Learning Algorithms for Verification of Markov Decision Processes](https://arxiv.org/abs/2403.09184) | 该研究提出了一个通用框架，将学习算法和启发式引导应用于马尔可夫决策过程（MDP）的验证，旨在提高性能，避免对状态空间进行穷尽探索。 |
| [^3] | [Verifiably Following Complex Robot Instructions with Foundation Models](https://arxiv.org/abs/2402.11498) | 提出了一种名为语言指令地面化运动规划（LIMP）系统，利用基础模型和时间逻辑生成指令条件的语义地图，使机器人能够可验证地遵循富有表现力和长期的指令，包括开放词汇参照和复杂的时空约束。 |
| [^4] | [Rethinking Optimization and Architecture for Tiny Language Models](https://arxiv.org/abs/2402.02791) | 本研究重新思考了微型语言模型的优化和架构，通过经验研究发现了在微型语言模型中特别有效的设计公式，并在多语种数据集上训练了高性能的微型语言模型。 |
| [^5] | [Online Reinforcement Learning in Non-Stationary Context-Driven Environments](https://arxiv.org/abs/2302.02182) | 提出了一种名为LCPO的在线强化学习方法，通过在优化当前经验回报的同时将策略对旧经验进行锚定来解决强化学习中的灾难性遗忘问题。 |
| [^6] | [Scaling Session-Based Transformer Recommendations using Optimized Negative Sampling and Loss Functions.](http://arxiv.org/abs/2307.14906) | 本文介绍了一种使用优化的负采样和损失函数扩展基于会话的Transformer推荐系统，该系统在大规模电商数据集上通过集成负采样和列表损失函数实现了较高的推荐准确性，并在实践中表现出潜力。 |
| [^7] | [Can Large Language Models Play Text Games Well? Current State-of-the-Art and Open Questions.](http://arxiv.org/abs/2304.02868) | 本文探究大型语言模型在玩文字游戏的能力，并发现其表现有竞争力，但仍然缺乏智能，有待提升。 |
| [^8] | [Quantum Circuit Fidelity Improvement with Long Short-Term Memory Networks.](http://arxiv.org/abs/2303.17523) | 本文提出使用长短期记忆网络解决量子计算中的保真度问题，利用时间序列预测方法预测量子电路的保真度。 |

# 详细

[^1]: 基于大型语言模型的游戏智能体综述

    A Survey on Large Language Model-Based Game Agents

    [https://arxiv.org/abs/2404.02039](https://arxiv.org/abs/2404.02039)

    LLM和MLLM的进步为游戏智能体提供了强大的人类决策能力，本文全面综述了基于LLM的游戏智能体的概念架构、方法论和未来研究方向

    

    游戏智能体的发展在推动人工通用智能（AGI）方面扮演着关键角色。LLM及其多模态对应物（MLLM）的进展为游戏智能体在复杂的电脑游戏环境中具备类似人类决策能力提供了前所未有的机会。本文从整体视角全面概述了基于LLM的游戏智能体。首先，我们介绍了以感知、记忆、思维、角色扮演、行动和学习为中心的LLM游戏智能体的概念架构。其次，我们调查了文献中已有的代表性LLM游戏智能体，涉及到六类游戏中的方法论和适应能力，包括冒险、沟通、竞争、合作、模拟以及创造与探索游戏。最后，我们展望了未来研究的方向。

    arXiv:2404.02039v1 Announce Type: new  Abstract: The development of game agents holds a critical role in advancing towards Artificial General Intelligence (AGI). The progress of LLMs and their multimodal counterparts (MLLMs) offers an unprecedented opportunity to evolve and empower game agents with human-like decision-making capabilities in complex computer game environments. This paper provides a comprehensive overview of LLM-based game agents from a holistic viewpoint. First, we introduce the conceptual architecture of LLM-based game agents, centered around six essential functional components: perception, memory, thinking, role-playing, action, and learning. Second, we survey existing representative LLM-based game agents documented in the literature with respect to methodologies and adaptation agility across six genres of games, including adventure, communication, competition, cooperation, simulation, and crafting & exploration games. Finally, we present an outlook of future research
    
[^2]: 学习算法用于验证马尔可夫决策过程

    Learning Algorithms for Verification of Markov Decision Processes

    [https://arxiv.org/abs/2403.09184](https://arxiv.org/abs/2403.09184)

    该研究提出了一个通用框架，将学习算法和启发式引导应用于马尔可夫决策过程（MDP）的验证，旨在提高性能，避免对状态空间进行穷尽探索。

    

    我们提出了一个通用框架，将学习算法和启发式引导应用于马尔可夫决策过程（MDP）的验证，基于Br\'azdil, T.等人（2014）的想法。该框架的主要目标是通过避免对状态空间进行穷尽探索来提高性能，而是依靠启发式。本研究在很大程度上扩展了这种方法。对基础理论的几个细节进行了改进和错误修正。第1.3节提供了所有差异的概述。该框架专注于概率可达性，这是验证中的一个核心问题，并具体化为两种不同的场景。第一个假设完全了解MDP，尤其是精确的转移概率。它执行基于启发式的模型部分探索，产生精准的结果。

    arXiv:2403.09184v1 Announce Type: cross  Abstract: We present a general framework for applying learning algorithms and heuristical guidance to the verification of Markov decision processes (MDPs), based on the ideas of Br\'azdil, T. et al. (2014). Verification of Markov Decision Processes Using Learning Algorithms. The primary goal of the techniques presented in that work is to improve performance by avoiding an exhaustive exploration of the state space, guided by heuristics. This approach is significantly extended in this work. Several details of the base theory are refined and errors are fixed. Section 1.3 provides an overview of all differences.   The presented framework focuses on probabilistic reachability, which is a core problem in verification, and is instantiated in two distinct scenarios. The first assumes that full knowledge of the MDP is available, in particular precise transition probabilities. It performs a heuristic-driven partial exploration of the model, yielding preci
    
[^3]: 使用基础模型可验证地遵循复杂机器人指令

    Verifiably Following Complex Robot Instructions with Foundation Models

    [https://arxiv.org/abs/2402.11498](https://arxiv.org/abs/2402.11498)

    提出了一种名为语言指令地面化运动规划（LIMP）系统，利用基础模型和时间逻辑生成指令条件的语义地图，使机器人能够可验证地遵循富有表现力和长期的指令，包括开放词汇参照和复杂的时空约束。

    

    让机器人能够遵循复杂的自然语言指令是一个重要但具有挑战性的问题。人们希望在指导机器人时能够灵活表达约束，指向任意地标并验证行为。相反，机器人必须将人类指令消除歧义，将指令参照物联系到真实世界中。我们提出了一种名为语言指令地面化运动规划（LIMP）的系统，该系统利用基础模型和时间逻辑生成指令条件的语义地图，使机器人能够可验证地遵循富有表现力和长期的指令，涵盖了开放词汇参照和复杂的时空约束。与先前在机器人任务执行中使用基础模型的方法相比，LIMP构建了一个可解释的指令表示，揭示了机器人与指导者预期动机的一致性，并实现了机器人行为的综合。

    arXiv:2402.11498v1 Announce Type: cross  Abstract: Enabling robots to follow complex natural language instructions is an important yet challenging problem. People want to flexibly express constraints, refer to arbitrary landmarks and verify behavior when instructing robots. Conversely, robots must disambiguate human instructions into specifications and ground instruction referents in the real world. We propose Language Instruction grounding for Motion Planning (LIMP), a system that leverages foundation models and temporal logics to generate instruction-conditioned semantic maps that enable robots to verifiably follow expressive and long-horizon instructions with open vocabulary referents and complex spatiotemporal constraints. In contrast to prior methods for using foundation models in robot task execution, LIMP constructs an explainable instruction representation that reveals the robot's alignment with an instructor's intended motives and affords the synthesis of robot behaviors that 
    
[^4]: 重新思考微型语言模型的优化和架构

    Rethinking Optimization and Architecture for Tiny Language Models

    [https://arxiv.org/abs/2402.02791](https://arxiv.org/abs/2402.02791)

    本研究重新思考了微型语言模型的优化和架构，通过经验研究发现了在微型语言模型中特别有效的设计公式，并在多语种数据集上训练了高性能的微型语言模型。

    

    大型语言模型（LLMs）的威力通过大量的数据和计算资源得到了证明。然而，在移动设备上应用语言模型面临着计算和内存成本的巨大挑战，迫切需要高性能的微型语言模型。受复杂训练过程的限制，优化语言模型的许多细节很少得到仔细研究。在本研究中，基于一个具有10亿参数的微型语言模型，我们仔细设计了一系列经验研究来分析每个组件的影响。主要讨论了三个方面，即神经架构、参数初始化和优化策略。多个设计公式在微型语言模型中经验性地被证明特别有效，包括分词器压缩、架构调整、参数继承和多轮训练。然后，我们在1.6T多语种数据集上训练了PanGu-$\pi$-1B Pro和PanGu-$\pi$-1.5B Pro。

    The power of large language models (LLMs) has been demonstrated through numerous data and computing resources. However, the application of language models on mobile devices is facing huge challenge on the computation and memory costs, that is, tiny language models with high performance are urgently required. Limited by the highly complex training process, there are many details for optimizing language models that are seldom studied carefully. In this study, based on a tiny language model with 1B parameters, we carefully design a series of empirical study to analyze the effect of each component. Three perspectives are mainly discussed, i.e., neural architecture, parameter initialization, and optimization strategy. Several design formulas are empirically proved especially effective for tiny language models, including tokenizer compression, architecture tweaking, parameter inheritance and multiple-round training. Then we train PanGu-$\pi$-1B Pro and PanGu-$\pi$-1.5B Pro on 1.6T multilingu
    
[^5]: 在非静态上下文驱动环境中的在线强化学习

    Online Reinforcement Learning in Non-Stationary Context-Driven Environments

    [https://arxiv.org/abs/2302.02182](https://arxiv.org/abs/2302.02182)

    提出了一种名为LCPO的在线强化学习方法，通过在优化当前经验回报的同时将策略对旧经验进行锚定来解决强化学习中的灾难性遗忘问题。

    

    我们研究了在非静态环境中的在线强化学习，其中一个随时间变化的外生上下文过程影响着环境动态。在线强化学习在这样的环境中具有挑战性，因为存在“灾难性遗忘”现象。随着训练过程中的新经验增加，代理 tend to forget 先前的知识。以往的方法通常假设任务标签（这在实践中往往是不存在的）或者使用脱机策略学习方法，但这些方法存在不稳定性和性能差的问题。我们提出了一种名为 Locally Constrained Policy Optimization (LCPO) 的在线强化学习方法，通过在优化当前经验回报的同时将策略对旧的经验进行锚定来解决灾难性遗忘问题。为了实现这种锚定，LCPO使用来自当前上下文分布之外的经验样本来局部约束策略优化。我们在Mujoco、经典控制和计算机系统环境中使用多种合成和真实上下文跟踪，评估了LCPO的性能，并发现它能够取得令人满意的结果。

    We study online reinforcement learning (RL) in non-stationary environments, where a time-varying exogenous context process affects the environment dynamics. Online RL is challenging in such environments due to "catastrophic forgetting" (CF). The agent tends to forget prior knowledge as it trains on new experiences. Prior approaches to mitigate this issue assume task labels (which are often not available in practice) or use off-policy methods that suffer from instability and poor performance.   We present Locally Constrained Policy Optimization (LCPO), an online RL approach that combats CF by anchoring policy outputs on old experiences while optimizing the return on current experiences. To perform this anchoring, LCPO locally constrains policy optimization using samples from experiences that lie outside of the current context distribution. We evaluate LCPO in Mujoco, classic control and computer systems environments with a variety of synthetic and real context traces, and find that it o
    
[^6]: 使用优化的负采样和损失函数扩展基于会话的Transformer推荐系统

    Scaling Session-Based Transformer Recommendations using Optimized Negative Sampling and Loss Functions. (arXiv:2307.14906v1 [cs.IR])

    [http://arxiv.org/abs/2307.14906](http://arxiv.org/abs/2307.14906)

    本文介绍了一种使用优化的负采样和损失函数扩展基于会话的Transformer推荐系统，该系统在大规模电商数据集上通过集成负采样和列表损失函数实现了较高的推荐准确性，并在实践中表现出潜力。

    

    本文介绍了TRON，一种使用优化的负采样的可扩展的基于会话的Transformer推荐系统。受到SASRec和GRU4Rec+等现有模型在可扩展性和性能方面的限制，TRON集成了top-k负采样和列表损失函数，以提高其推荐准确性。在相关的大规模电子商务数据集上的评估结果表明，TRON在保持与SASRec类似的训练速度的同时，改进了当前方法的推荐质量。一项实时的A/B测试显示，相对于SASRec，TRON的点击率增加了18.14%，突显了其在实际环境中的潜力。

    This work introduces TRON, a scalable session-based Transformer Recommender using Optimized Negative-sampling. Motivated by the scalability and performance limitations of prevailing models such as SASRec and GRU4Rec+, TRON integrates top-k negative sampling and listwise loss functions to enhance its recommendation accuracy. Evaluations on relevant large-scale e-commerce datasets show that TRON improves upon the recommendation quality of current methods while maintaining training speeds similar to SASRec. A live A/B test yielded an 18.14% increase in click-through rate over SASRec, highlighting the potential of TRON in practical settings. For further research, we provide access to our source code at https://github.com/otto-de/TRON and an anonymized dataset at https://github.com/otto-de/recsys-dataset.
    
[^7]: 大型语言模型能否能够很好地玩文字游戏？现状和未来问题研究

    Can Large Language Models Play Text Games Well? Current State-of-the-Art and Open Questions. (arXiv:2304.02868v1 [cs.CL])

    [http://arxiv.org/abs/2304.02868](http://arxiv.org/abs/2304.02868)

    本文探究大型语言模型在玩文字游戏的能力，并发现其表现有竞争力，但仍然缺乏智能，有待提升。

    

    最近，诸如ChatGPT和GPT-4之类的大型语言模型展示了它们与人类用户通信的卓越能力。本技术报告旨在调查它们在玩文字游戏方面的能力，这要求玩家通过与游戏世界的对话来理解环境并对情况做出反应。我们的实验表明，与所有现有系统相比，ChatGPT表现出有竞争力，但仍然表现出较低的智能水平。确切地说，ChatGPT无法通过玩游戏或阅读游戏手册来构建世界模型；它可能无法利用它已经拥有的世界知识；它无法推断出随着游戏进展的每一步的目标。我们的结果在人工智能、机器学习和自然语言处理交叉领域开启了新的研究问题。

    Large language models (LLMs) such as ChatGPT and GPT-4 have recently demonstrated their remarkable abilities of communicating with human users. In this technical report, we take an initiative to investigate their capacities of playing text games, in which a player has to understand the environment and respond to situations by having dialogues with the game world. Our experiments show that ChatGPT performs competitively compared to all the existing systems but still exhibits a low level of intelligence. Precisely, ChatGPT can not construct the world model by playing the game or even reading the game manual; it may fail to leverage the world knowledge that it already has; it cannot infer the goal of each step as the game progresses. Our results open up new research questions at the intersection of artificial intelligence, machine learning, and natural language processing.
    
[^8]: 利用长短期记忆网络提高量子电路保真度

    Quantum Circuit Fidelity Improvement with Long Short-Term Memory Networks. (arXiv:2303.17523v1 [quant-ph])

    [http://arxiv.org/abs/2303.17523](http://arxiv.org/abs/2303.17523)

    本文提出使用长短期记忆网络解决量子计算中的保真度问题，利用时间序列预测方法预测量子电路的保真度。

    

    量子计算已进入噪声中间规模量子（NISQ）时代，目前我们拥有的量子处理器对辐射和温度等环境变量敏感，因此会产生嘈杂的输出。虽然已经有许多算法和应用程序用于NISQ处理器，但我们仍面临着解释其嘈杂结果的不确定性。具体来说，我们对所选择的量子态有多少信心？这种信心很重要，因为NISQ计算机将输出其量子位测量的概率分布，有时很难区分分布是否表示有意义的计算或只是随机噪声。本文提出了一种新方法来解决这个问题，将量子电路保真度预测框架为时间序列预测问题，因此可以利用长短期记忆（LSTM）神经网络的强大能力。一个完整的工作流程来构建训练电路

    Quantum computing has entered the Noisy Intermediate-Scale Quantum (NISQ) era. Currently, the quantum processors we have are sensitive to environmental variables like radiation and temperature, thus producing noisy outputs. Although many proposed algorithms and applications exist for NISQ processors, we still face uncertainties when interpreting their noisy results. Specifically, how much confidence do we have in the quantum states we are picking as the output? This confidence is important since a NISQ computer will output a probability distribution of its qubit measurements, and it is sometimes hard to distinguish whether the distribution represents meaningful computation or just random noise. This paper presents a novel approach to attack this problem by framing quantum circuit fidelity prediction as a Time Series Forecasting problem, therefore making it possible to utilize the power of Long Short-Term Memory (LSTM) neural networks. A complete workflow to build the training circuit d
    

