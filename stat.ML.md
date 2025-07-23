# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Randomization Can Reduce Both Bias and Variance: A Case Study in Random Forests](https://arxiv.org/abs/2402.12668) | 随机森林相对于装袋法具有减少偏差的能力，在揭示数据模式和高信噪比情况下表现更好的特点，为随机森林在不同信噪比环境下的成功提供了解释和实用见解。 |

# 详细

[^1]: 随机化既可以减少偏差又可以减少方差：随机森林的案例研究

    Randomization Can Reduce Both Bias and Variance: A Case Study in Random Forests

    [https://arxiv.org/abs/2402.12668](https://arxiv.org/abs/2402.12668)

    随机森林相对于装袋法具有减少偏差的能力，在揭示数据模式和高信噪比情况下表现更好的特点，为随机森林在不同信噪比环境下的成功提供了解释和实用见解。

    

    我们研究了往往被忽视的现象，首次在\cite{breiman2001random}中指出，即随机森林似乎比装袋法减少了偏差。受\cite{mentch2020randomization}一篇有趣的论文的启发，其中作者认为随机森林减少了有效自由度，并且只有在低信噪比（SNR）环境下才能胜过装袋集成，我们探讨了随机森林如何能够揭示被装袋法忽视的数据模式。我们在实证中证明，在存在这种模式的情况下，随机森林不仅可以减小偏差还能减小方差，并且当信噪比高时随机森林的表现愈发好于装袋集成。我们的观察为解释随机森林在各种信噪比情况下的真实世界成功提供了见解，并增进了我们对随机森林与装袋集成在每次分割注入的随机化方面的差异的理解。我们的调查结果还提供了实用见解。

    arXiv:2402.12668v1 Announce Type: cross  Abstract: We study the often overlooked phenomenon, first noted in \cite{breiman2001random}, that random forests appear to reduce bias compared to bagging. Motivated by an interesting paper by \cite{mentch2020randomization}, where the authors argue that random forests reduce effective degrees of freedom and only outperform bagging ensembles in low signal-to-noise ratio (SNR) settings, we explore how random forests can uncover patterns in the data missed by bagging. We empirically demonstrate that in the presence of such patterns, random forests reduce bias along with variance and increasingly outperform bagging ensembles when SNR is high. Our observations offer insights into the real-world success of random forests across a range of SNRs and enhance our understanding of the difference between random forests and bagging ensembles with respect to the randomization injected into each split. Our investigations also yield practical insights into the 
    

