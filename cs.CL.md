# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Semi-Offline Reinforcement Learning for Optimized Text Generation.](http://arxiv.org/abs/2306.09712) | 该研究提出了一种半离线强化学习范式，该范式平衡了探索能力和培训成本，提供了一个理论基础来比较不同的强化学习设置，并在优化成本、渐近误差和过度拟合误差界方面实现了最优的RL设置。实验结果表明，该方法高效且性能优异。 |
| [^2] | [Ousiometrics and Telegnomics: The essence of meaning conforms to a two-dimensional powerful-weak and dangerous-safe framework with diverse corpora presenting a safety bias.](http://arxiv.org/abs/2110.06847) | 本文提出了ousiometrics和Telegnomics，发现单词传达的基本含义最好用指南针般的强度-危险（PD）框架来描述，而自然语言对安全低危险的词汇有系统偏见。 |

# 详细

[^1]: 半离线强化学习用于优化文本生成

    Semi-Offline Reinforcement Learning for Optimized Text Generation. (arXiv:2306.09712v1 [cs.LG])

    [http://arxiv.org/abs/2306.09712](http://arxiv.org/abs/2306.09712)

    该研究提出了一种半离线强化学习范式，该范式平衡了探索能力和培训成本，提供了一个理论基础来比较不同的强化学习设置，并在优化成本、渐近误差和过度拟合误差界方面实现了最优的RL设置。实验结果表明，该方法高效且性能优异。

    

    在强化学习中，与环境交互有两种主要方式：在线和离线。在线方法探索环境所需时间较长，而离线方法通过牺牲探索能力有效地获得奖励信号。我们提出了半离线RL，一种新的范式，可以平滑地从离线转换到在线设置，平衡探索能力和培训成本，并为比较不同RL设置提供理论基础。基于半离线公式，我们提出了在优化成本、渐近误差和过度拟合误差界方面最优的RL设置。广泛的实验表明，我们的半离线方法效率高，与最先进的方法相比具有可比性或更好的性能。

    In reinforcement learning (RL), there are two major settings for interacting with the environment: online and offline. Online methods explore the environment at significant time cost, and offline methods efficiently obtain reward signals by sacrificing exploration capability. We propose semi-offline RL, a novel paradigm that smoothly transits from offline to online settings, balances exploration capability and training cost, and provides a theoretical foundation for comparing different RL settings. Based on the semi-offline formulation, we present the RL setting that is optimal in terms of optimization cost, asymptotic error, and overfitting error bound. Extensive experiments show that our semi-offline approach is efficient and yields comparable or often better performance compared with state-of-the-art methods.
    
[^2]: Ousiometrics和Telegnomics：基于强度-弱度和危险-安全两个维度的意义本质框架，具有安全偏见的多样语料库。

    Ousiometrics and Telegnomics: The essence of meaning conforms to a two-dimensional powerful-weak and dangerous-safe framework with diverse corpora presenting a safety bias. (arXiv:2110.06847v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2110.06847](http://arxiv.org/abs/2110.06847)

    本文提出了ousiometrics和Telegnomics，发现单词传达的基本含义最好用指南针般的强度-危险（PD）框架来描述，而自然语言对安全低危险的词汇有系统偏见。

    

    我们定义“ousiometrics”为在任何有意义信号传递的上下文中研究基本意义的学科，“telegnomics”为远程感知知识的研究。通过中期出现的工作，基本意义已被普遍接受为由评估（evaluation）、效能（potency）和激活（activation）三个正交维度很好地捕捉。通过重新检查英语语言的类型和标记，以及通过使用自动注释的直方图——“ousiograms”，我们发现：1. 用单词传达的基本含义最好用指南针般的强度-危险（PD）框架来描述。2. 对大规模英语语言语料库（文学、新闻、维基百科、脱口秀和社交媒体）的分散集合进行分析显示，自然语言对安全、低危险的词汇存在系统偏见，这是对Pollyanna原则的书面表达积极偏差的重新解释。

    We define `ousiometrics' to be the study of essential meaning in whatever context that meaningful signals are communicated, and `telegnomics' as the study of remotely sensed knowledge. From work emerging through the middle of the 20th century, the essence of meaning has become generally accepted as being well captured by the three orthogonal dimensions of evaluation, potency, and activation (EPA). By re-examining first types and then tokens for the English language, and through the use of automatically annotated histograms -`ousiograms' -- we find here that: 1. The essence of meaning conveyed by words is instead best described by a compass-like power-danger (PD) framework, and 2. Analysis of a disparate collection of large-scale English language corpora -literature, news, Wikipedia, talk radio, and social media -- shows that natural language exhibits a systematic bias toward safe, low danger words -- a reinterpretation of the Pollyanna principle's positivity bias for written expres
    

