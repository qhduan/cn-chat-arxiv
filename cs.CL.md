# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Scaling Data-Constrained Language Models.](http://arxiv.org/abs/2305.16264) | 研究人员研究了在数据受限制的情况下缩放语言模型，并提出了一个计算最优性的缩放定律，考虑到重复令牌和过量参数的价值递减。 |

# 详细

[^1]: 缩放数据受限的语言模型

    Scaling Data-Constrained Language Models. (arXiv:2305.16264v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.16264](http://arxiv.org/abs/2305.16264)

    研究人员研究了在数据受限制的情况下缩放语言模型，并提出了一个计算最优性的缩放定律，考虑到重复令牌和过量参数的价值递减。

    

    现在扩展语言模型的趋势涉及增加参数计数和训练数据集大小。推断这个趋势表明，训练数据集大小可能很快就会受到互联网上可用文本数据的限制。出于此限制的动机，我们研究在数据受限制的情况下缩放语言模型。具体而言，我们运行了大量的实验，变化数据重复程度和计算预算，范围达到了9000亿个训练令牌和9亿参数模型。我们发现，在有限的数据的情况下，使用高达4次重复数据的训练与使用唯一数据相比对损失的贡献微不足道。然而，使用更多的重复数据，添加计算的价值最终会衰减为零。我们提出并经验证了一个计算最优性的缩放定律，考虑到重复令牌和过量参数的价值递减。最后，我们尝试了缓解数据稀缺的方法。

    The current trend of scaling language models involves increasing both parameter count and training dataset size. Extrapolating this trend suggests that training dataset size may soon be limited by the amount of text data available on the internet. Motivated by this limit, we investigate scaling language models in data-constrained regimes. Specifically, we run a large set of experiments varying the extent of data repetition and compute budget, ranging up to 900 billion training tokens and 9 billion parameter models. We find that with constrained data for a fixed compute budget, training with up to 4 epochs of repeated data yields negligible changes to loss compared to having unique data. However, with more repetition, the value of adding compute eventually decays to zero. We propose and empirically validate a scaling law for compute optimality that accounts for the decreasing value of repeated tokens and excess parameters. Finally, we experiment with approaches mitigating data scarcity,
    

