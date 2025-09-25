# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Differentially Private Bootstrap: New Privacy Analysis and Inference Strategies.](http://arxiv.org/abs/2210.06140) | 本文研究了一种差分隐私引导采样方法，提供了隐私成本的新结果，可用于推断样本分布并构建置信区间，同时指出了现有文献中的误用。随着采样次数趋近无限大，此方法逐渐满足更严格的差分隐私要求。 |

# 详细

[^1]: 差分隐私引导采样：新的隐私分析与推断策略

    Differentially Private Bootstrap: New Privacy Analysis and Inference Strategies. (arXiv:2210.06140v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2210.06140](http://arxiv.org/abs/2210.06140)

    本文研究了一种差分隐私引导采样方法，提供了隐私成本的新结果，可用于推断样本分布并构建置信区间，同时指出了现有文献中的误用。随着采样次数趋近无限大，此方法逐渐满足更严格的差分隐私要求。

    

    差分隐私机制通过引入随机性来保护个人信息，但在应用中，统计推断仍然缺乏通用技术。本文研究了一个差分隐私引导采样方法，通过发布多个私有引导采样估计来推断样本分布并构建置信区间。我们的隐私分析提供了单个差分隐私引导采样估计的隐私成本新结果，适用于任何差分隐私机制，并指出了现有文献中引导采样的一些误用。使用Gaussian-DP（GDP）框架，我们证明从满足 $(\mu/\sqrt{(2-2/\mathrm{e})B})$-GDP 的机制中释放 $B$ 个差分隐私引导采样估计，在 $B$ 趋近无限大时渐近地满足 $\mu$-GDP。此外，我们使用差分隐私引导采样估计的反卷积对样本分布进行准确推断。

    Differentially private (DP) mechanisms protect individual-level information by introducing randomness into the statistical analysis procedure. Despite the availability of numerous DP tools, there remains a lack of general techniques for conducting statistical inference under DP. We examine a DP bootstrap procedure that releases multiple private bootstrap estimates to infer the sampling distribution and construct confidence intervals (CIs). Our privacy analysis presents new results on the privacy cost of a single DP bootstrap estimate, applicable to any DP mechanisms, and identifies some misapplications of the bootstrap in the existing literature. Using the Gaussian-DP (GDP) framework (Dong et al.,2022), we show that the release of $B$ DP bootstrap estimates from mechanisms satisfying $(\mu/\sqrt{(2-2/\mathrm{e})B})$-GDP asymptotically satisfies $\mu$-GDP as $B$ goes to infinity. Moreover, we use deconvolution with the DP bootstrap estimates to accurately infer the sampling distribution
    

