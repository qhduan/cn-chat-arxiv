# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Paramanu: A Family of Novel Efficient Indic Generative Foundation Language Models](https://arxiv.org/abs/2401.18034) | Paramanu是一种高效的印度生成式基础语言模型系列，包含多种印度语言模型，并且在单个GPU上进行了从头预训练。它还包括一个先进的印度分词器以及避免多语言诅咒的预训练方法。这些模型在人工评估中展现出良好的语法、连贯性、创造性和事实准确性。 |
| [^2] | [Representation-Driven Reinforcement Learning.](http://arxiv.org/abs/2305.19922) | 该论文提出了一个表示驱动的强化学习框架，通过在线性特征空间中嵌入策略网络，重新框定探索-利用问题为表示-利用问题，以实现最佳的探索。该框架通过应用进化和策略梯度法取得了显著的性能提升。 |

# 详细

[^1]: Paramanu: 一种高效的印度生成式基础语言模型系列

    Paramanu: A Family of Novel Efficient Indic Generative Foundation Language Models

    [https://arxiv.org/abs/2401.18034](https://arxiv.org/abs/2401.18034)

    Paramanu是一种高效的印度生成式基础语言模型系列，包含多种印度语言模型，并且在单个GPU上进行了从头预训练。它还包括一个先进的印度分词器以及避免多语言诅咒的预训练方法。这些模型在人工评估中展现出良好的语法、连贯性、创造性和事实准确性。

    

    我们介绍了Gyan AI Paramanu（“原子”），一种适用于印度语言的新型语言模型系列。它是一个在单个GPU上从头开始预训练的包含单语、双语和多语印度语言模型的集合，涵盖了10种印度语言（阿萨姆语、孟加拉语、印地语、康坎尼语、迈蒂利语、马拉地语、奥迪亚语、梵语、泰米尔语和泰卢固语）以及5种不同大小的字母表（孟加拉语、天城体、奥迪亚语、泰米尔语和泰卢固语）。这些模型以1024的上下文大小在单个GPU上预训练，非常高效、小巧、快速且强大。我们还开发了一种高效的先进的印度语分词器，甚至可以标记未知语言。为了避免我们的多语言mParamanu模型中的“多语言诅咒”，我们使用相同的字母表按语言类型进行了可比较语料库的预训练。我们对我们预训练模型进行了人工评估，评估指标包括语法、连贯性、创造性和事实准确性。

    We present Gyan AI Paramanu ("atom"), a family of novel language models for Indian languages. It is a collection of auto-regressive monolingual, bilingual, and multilingual Indic language models pretrained from scratch on a single GPU for 10 Indian languages (Assamese, Bangla, Hindi, Konkani, Maithili, Marathi, Odia, Sanskrit, Tamil, Telugu) across 5 scripts (Bangla, Devanagari, Odia, Tamil, Telugu) of varying sizes ranging from 13.29M to 367.5M.The models are pretrained with a context size of 1024 on a single GPU. The models are very efficient, small, fast, and powerful. We have also developed an efficient most advanced Indic tokenizer that can even tokenize unseen languages. In order to avoid the "curse of multi-linguality" in our multilingual mParamanu model, we pretrained on comparable corpora by typological grouping using the same script. We performed human evaluation of our pretrained models for open end text generation on grammar, coherence, creativity, and factuality metrics fo
    
[^2]: 表示驱动的强化学习框架

    Representation-Driven Reinforcement Learning. (arXiv:2305.19922v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2305.19922](http://arxiv.org/abs/2305.19922)

    该论文提出了一个表示驱动的强化学习框架，通过在线性特征空间中嵌入策略网络，重新框定探索-利用问题为表示-利用问题，以实现最佳的探索。该框架通过应用进化和策略梯度法取得了显著的性能提升。

    

    我们提出了一个表示驱动的强化学习框架。通过将策略表示为其期望值的估计，我们利用来自情境推断的方法来指导探索和利用。特别地，将策略网络嵌入到线性特征空间中，使我们能够将探索-利用问题重新框定为表示-利用问题，其中良好的策略表示能够实现最佳的探索。我们通过应用进化和策略梯度法来展示该框架的有效性，相比于传统方法，这些方法带来了显著的性能提升。我们的框架提供了一种强化学习的新视角，强调了策略表示在决定最佳探索-利用策略方面的重要性。

    We present a representation-driven framework for reinforcement learning. By representing policies as estimates of their expected values, we leverage techniques from contextual bandits to guide exploration and exploitation. Particularly, embedding a policy network into a linear feature space allows us to reframe the exploration-exploitation problem as a representation-exploitation problem, where good policy representations enable optimal exploration. We demonstrate the effectiveness of this framework through its application to evolutionary and policy gradient-based approaches, leading to significantly improved performance compared to traditional methods. Our framework provides a new perspective on reinforcement learning, highlighting the importance of policy representation in determining optimal exploration-exploitation strategies.
    

