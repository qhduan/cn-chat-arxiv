# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Offline Fictitious Self-Play for Competitive Games](https://arxiv.org/abs/2403.00841) | 本文介绍了Off-FSP，这是竞争游戏的第一个实用的无模型离线RL算法，通过调整固定数据集的权重，使用重要性抽样，模拟与各种对手的互动。 |
| [^2] | [Jailbreaking Proprietary Large Language Models using Word Substitution Cipher](https://arxiv.org/abs/2402.10601) | 本文使用密码技术编码了越狱提示，成功地绕过了大型语言模型对有害问题的检测，实验结果显示攻击成功率高达59.42%。 |

# 详细

[^1]: 竞争游戏的离线虚构自我对弈

    Offline Fictitious Self-Play for Competitive Games

    [https://arxiv.org/abs/2403.00841](https://arxiv.org/abs/2403.00841)

    本文介绍了Off-FSP，这是竞争游戏的第一个实用的无模型离线RL算法，通过调整固定数据集的权重，使用重要性抽样，模拟与各种对手的互动。

    

    离线强化学习（RL）因其在以前收集的数据集中改进策略而不需要在线交互的能力而受到重视。尽管在单一智能体设置中取得成功，但离线多智能体RL仍然是一个挑战，特别是在竞争游戏中。为了解决这些问题，本文介绍了Off-FSP，这是竞争游戏的第一个实用的无模型离线RL算法。我们首先通过调整固定数据集的权重，使用重要性抽样模拟与各种对手的互动。

    arXiv:2403.00841v1 Announce Type: cross  Abstract: Offline Reinforcement Learning (RL) has received significant interest due to its ability to improve policies in previously collected datasets without online interactions. Despite its success in the single-agent setting, offline multi-agent RL remains a challenge, especially in competitive games. Firstly, unaware of the game structure, it is impossible to interact with the opponents and conduct a major learning paradigm, self-play, for competitive games. Secondly, real-world datasets cannot cover all the state and action space in the game, resulting in barriers to identifying Nash equilibrium (NE). To address these issues, this paper introduces Off-FSP, the first practical model-free offline RL algorithm for competitive games. We start by simulating interactions with various opponents by adjusting the weights of the fixed dataset with importance sampling. This technique allows us to learn best responses to different opponents and employ
    
[^2]: 使用单词替换密码来越狱专有的大型语言模型

    Jailbreaking Proprietary Large Language Models using Word Substitution Cipher

    [https://arxiv.org/abs/2402.10601](https://arxiv.org/abs/2402.10601)

    本文使用密码技术编码了越狱提示，成功地绕过了大型语言模型对有害问题的检测，实验结果显示攻击成功率高达59.42%。

    

    大型语言模型（LLMs）遵循道德和伦理准则，但仍然容易受到名为Jailbreak的创意提示的影响，这些提示可以绕过对齐过程。然而，大多数越狱提示包含自然语言（主要是英语）中的有害问题，可以被LLMs自身检测到。本文提出了使用密码技术编码的越狱提示。我们首先在最先进的LLM，GPT-4上进行了一个试点研究，解码了使用各种密码技术加密的几个安全句子，发现简单的单词替换密码可以被最有效地解码。受此结果启发，我们使用这种编码技术来编写越狱提示。我们提供了将不安全单词映射到安全单词，并使用这些映射的单词提出不安全问题的映射。实验结果显示，我们提出的越狱攻击成功率（高达59.42%）。

    arXiv:2402.10601v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are aligned to moral and ethical guidelines but remain susceptible to creative prompts called Jailbreak that can bypass the alignment process. However, most jailbreaking prompts contain harmful questions in the natural language (mainly English), which can be detected by the LLM themselves. In this paper, we present jailbreaking prompts encoded using cryptographic techniques. We first present a pilot study on the state-of-the-art LLM, GPT-4, in decoding several safe sentences that have been encrypted using various cryptographic techniques and find that a straightforward word substitution cipher can be decoded most effectively. Motivated by this result, we use this encoding technique for writing jailbreaking prompts. We present a mapping of unsafe words with safe words and ask the unsafe question using these mapped words. Experimental results show an attack success rate (up to 59.42%) of our proposed jailbrea
    

