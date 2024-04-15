# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Non-robustness of diffusion estimates on networks with measurement error](https://arxiv.org/abs/2403.05704) | 测量误差对网络扩散估计产生了严重的非鲁棒性，即使误差很小，也会导致扩散程度被低估，并且初始种子的微小误差会大大改变预期扩散路径。 |
| [^2] | [Inference for Synthetic Controls via Refined Placebo Tests.](http://arxiv.org/abs/2401.07152) | 本文提出了一种通过精细安慰剂测试进行合成控制的推断方法，用于解决只有一个处理单元和少数对照单元的问题，并解决了样本量小和实际估计过程简化的问题。 |
| [^3] | [Quantifying the Influence of Climate on Human Mind and Culture: Evidence from Visual Art.](http://arxiv.org/abs/2307.15540) | 本研究通过分析大量绘画作品和艺术家的生物数据，发现气候变化对人类思维和文化产生了显著和持久的影响，而且这种影响在更依赖艺术家想象力的艺术流派中更加明显。 |

# 详细

[^1]: 测量误差对网络扩散估计的非鲁棒性

    Non-robustness of diffusion estimates on networks with measurement error

    [https://arxiv.org/abs/2403.05704](https://arxiv.org/abs/2403.05704)

    测量误差对网络扩散估计产生了严重的非鲁棒性，即使误差很小，也会导致扩散程度被低估，并且初始种子的微小误差会大大改变预期扩散路径。

    

    网络扩散模型被用于研究疾病传播、信息传播和技术采用等问题。然而，构建这些模型的网络中极有可能存在少量测量误差。我们展示了扩散估计对这种测量误差的非常非鲁棒性。首先，我们展示即使测量误差几乎可以忽略不计，错过链接的比例接近于零，关于扩散程度的预测也会大大低估真实情况。其次，初始种子身份的微小测量误差会导致预期扩散路径的位置发生很大偏移。我们展示了即使微小的测量误差只是局部性质，这两项结果仍然成立。即使在基础传播数可持续估计的条件下，预测中存在这种非鲁棒性。可能的解决方案，比如估计测量误差的方法。

    arXiv:2403.05704v1 Announce Type: new  Abstract: Network diffusion models are used to study things like disease transmission, information spread, and technology adoption. However, small amounts of mismeasurement are extremely likely in the networks constructed to operationalize these models. We show that estimates of diffusions are highly non-robust to this measurement error. First, we show that even when measurement error is vanishingly small, such that the share of missed links is close to zero, forecasts about the extent of diffusion will greatly underestimate the truth. Second, a small mismeasurement in the identity of the initial seed generates a large shift in the locations of expected diffusion path. We show that both of these results still hold when the vanishing measurement error is only local in nature. Such non-robustness in forecasting exists even under conditions where the basic reproductive number is consistently estimable. Possible solutions, such as estimating the measu
    
[^2]: 通过精细安慰剂测试进行合成控制的推断

    Inference for Synthetic Controls via Refined Placebo Tests. (arXiv:2401.07152v1 [stat.ME])

    [http://arxiv.org/abs/2401.07152](http://arxiv.org/abs/2401.07152)

    本文提出了一种通过精细安慰剂测试进行合成控制的推断方法，用于解决只有一个处理单元和少数对照单元的问题，并解决了样本量小和实际估计过程简化的问题。

    

    合成控制方法通常用于只有一个处理单元和少数对照单元的问题。在这种情况下，一种常见的推断任务是测试关于对待处理单元的平均处理效应的零假设。由于（1）样本量较小导致大样本近似不稳定和（2）在实践中实施的估计过程的简化，因此通常无法满足渐近合理性的推断程序常常不令人满意。一种替代方法是置换推断，它与常见的称为安慰剂测试的诊断相关。当治疗均匀分配时，它在有限样本中具有可证明的 Type-I 错误保证，而无需简化方法。尽管具有这种健壮性，安慰剂测试由于只从 $N$ 个参考估计构造零分布，其中 $N$ 是样本量，因此分辨率较低。这在常见的水平 $\alpha = 0.05$ 的统计推断中形成了一个障碍，特别是在小样本问题中。

    The synthetic control method is often applied to problems with one treated unit and a small number of control units. A common inferential task in this setting is to test null hypotheses regarding the average treatment effect on the treated. Inference procedures that are justified asymptotically are often unsatisfactory due to (1) small sample sizes that render large-sample approximation fragile and (2) simplification of the estimation procedure that is implemented in practice. An alternative is permutation inference, which is related to a common diagnostic called the placebo test. It has provable Type-I error guarantees in finite samples without simplification of the method, when the treatment is uniformly assigned. Despite this robustness, the placebo test suffers from low resolution since the null distribution is constructed from only $N$ reference estimates, where $N$ is the sample size. This creates a barrier for statistical inference at a common level like $\alpha = 0.05$, especia
    
[^3]: 量化气候对人类思维和文化的影响: 来自视觉艺术的证据

    Quantifying the Influence of Climate on Human Mind and Culture: Evidence from Visual Art. (arXiv:2307.15540v1 [q-bio.PE])

    [http://arxiv.org/abs/2307.15540](http://arxiv.org/abs/2307.15540)

    本研究通过分析大量绘画作品和艺术家的生物数据，发现气候变化对人类思维和文化产生了显著和持久的影响，而且这种影响在更依赖艺术家想象力的艺术流派中更加明显。

    

    本文研究了气候变化对人类思维和文化的影响，时间跨度从13世纪到21世纪。通过对10万幅绘画作品和2000多位艺术家的生物数据进行定量分析，发现了绘画明亮度的一种有趣的U型模式，与全球温度趋势相关。事件研究分析发现，当艺术家遭受高温冲击时，他们的作品在后期变得更加明亮。此外，这种影响在更多依赖艺术家想象力而非现实事物的艺术流派中更加明显，表明了艺术家思维的影响。总体而言，本研究证明了气候对人类思维和文化的显著和持久影响。

    This paper examines the influence of climate change on the human mind and culture from the 13th century to the 21st century. By quantitatively analyzing 100,000 paintings and the biological data of over 2,000 artists, an interesting U-shaped pattern in the lightness of paintings was found, which correlated with trends in global temperature. Event study analysis revealed that when an artist is subjected to a high-temperature shock, their paintings become brighter in later periods. Moreover, the effects are more pronounced in art genres that rely less on real things and more on the artist's imagination, indicating the influence of artists' minds. Overall, this study demonstrates the significant and enduring influence of climate on the human mind and culture over centuries.
    

