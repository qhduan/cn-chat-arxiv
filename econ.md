# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Comprehensive OOS Evaluation of Predictive Algorithms with Statistical Decision Theory](https://arxiv.org/abs/2403.11016) | 使用统计决策理论（SDT）进行全面的样本外（OOS）评估，取代当前机器学习（ML）研究中常见的K-fold和公共任务框架验证方法。 |
| [^2] | [Was Javert right to be suspicious? Unpacking treatment effect heterogeneity of alternative sentences on time-to-recidivism in Brazil](https://arxiv.org/abs/2311.13969) | 本文提出了新的计量经济学工具来解析轻罪犯的刑罚对再犯时间的异质效应。研究结果发现，对于绝大多数法官会惩罚的人而言，刑罚会延长再犯时间，而只有严厉的法官才会对再犯率产生影响。 |
| [^3] | [A Bias-Corrected CD Test for Error Cross-Sectional Dependence in Panel Data Models with Latent Factors.](http://arxiv.org/abs/2109.00408) | 本文提出了一种修正CD检验的新方法CD*，适用于纯潜在因素模型以及面板回归，具有渐近正态性，相比于现有方法具有更好的性能。 |

# 详细

[^1]: 预测算法的全面OOS评估与统计决策理论

    Comprehensive OOS Evaluation of Predictive Algorithms with Statistical Decision Theory

    [https://arxiv.org/abs/2403.11016](https://arxiv.org/abs/2403.11016)

    使用统计决策理论（SDT）进行全面的样本外（OOS）评估，取代当前机器学习（ML）研究中常见的K-fold和公共任务框架验证方法。

    

    我们认为，应当使用统计决策理论（SDT）进行全面的样本外（OOS）评估，取代机器学习（ML）研究中当前的K-fold和公共任务框架验证的做法。SDT为在所有可能的（1）训练样本、（2）可能生成训练数据的人群和（3）预测利益人群之间进行全面的OOS评估提供了一个形式化框架。关于特征（3），我们强调SDT要求从业者直接面对未来可能不同于过去的可能性，并在构建预测算法时考虑从一个人群推断到另一个人群的可能需要。SDT在抽象上是简单的，但在实施时往往需要大量计算。我们讨论了当预测准确性通过均方误差或误分类率衡量时，SDT的可行实施进展。

    arXiv:2403.11016v1 Announce Type: new  Abstract: We argue that comprehensive out-of-sample (OOS) evaluation using statistical decision theory (SDT) should replace the current practice of K-fold and Common Task Framework validation in machine learning (ML) research. SDT provides a formal framework for performing comprehensive OOS evaluation across all possible (1) training samples, (2) populations that may generate training data, and (3) populations of prediction interest. Regarding feature (3), we emphasize that SDT requires the practitioner to directly confront the possibility that the future may not look like the past and to account for a possible need to extrapolate from one population to another when building a predictive algorithm. SDT is simple in abstraction, but it is often computationally demanding to implement. We discuss progress in tractable implementation of SDT when prediction accuracy is measured by mean square error or by misclassification rate. We summarize research st
    
[^2]: 对巴西再犯时间的替代刑罚效果异质性的厘清，雅弗尔特的怀疑是否合理？

    Was Javert right to be suspicious? Unpacking treatment effect heterogeneity of alternative sentences on time-to-recidivism in Brazil

    [https://arxiv.org/abs/2311.13969](https://arxiv.org/abs/2311.13969)

    本文提出了新的计量经济学工具来解析轻罪犯的刑罚对再犯时间的异质效应。研究结果发现，对于绝大多数法官会惩罚的人而言，刑罚会延长再犯时间，而只有严厉的法官才会对再犯率产生影响。

    

    本文提出了新的计量经济学工具，以解析惩罚轻罪犯对再犯时间的异质治疗效应。我们展示了如何在治疗选择是内生的情况下，对经常是在右侧截断的持续变量等感兴趣的结果进行分布、分位数和平均边际治疗效应的识别、估计和推断。利用在巴西圣保罗州2010年至2019年法官对案件的仿拟随机分配，我们探索了我们提出的计量经济学方法来评估罚款和社区服务句作为一种惩罚形式对再犯时间的影响。我们的结果突显了其他工具无法捕捉到的显著治疗效应异质性。例如，我们发现大多数法官会惩罚的人由于处罚而再犯的时间更长，而只有严厉的法官会惩罚的人再犯率较高。

    This paper presents new econometric tools to unpack the treatment effect heterogeneity of punishing misdemeanor offenses on time-to-recidivism. We show how one can identify, estimate, and make inferences on the distributional, quantile, and average marginal treatment effects in setups where the treatment selection is endogenous and the outcome of interest, usually a duration variable, is potentially right-censored. We explore our proposed econometric methodology to evaluate the effect of fines and community service sentences as a form of punishment on time-to-recidivism in the State of S\~ao Paulo, Brazil, between 2010 and 2019, leveraging the as-if random assignment of judges to cases. Our results highlight substantial treatment effect heterogeneity that other tools are not meant to capture. For instance, we find that people whom most judges would punish take longer to recidivate as a consequence of the punishment, while people who would be punished only by strict judges recidivate at
    
[^3]: 修正潜在因素面板数据模型中误差交叉相关的CD检验

    A Bias-Corrected CD Test for Error Cross-Sectional Dependence in Panel Data Models with Latent Factors. (arXiv:2109.00408v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2109.00408](http://arxiv.org/abs/2109.00408)

    本文提出了一种修正CD检验的新方法CD*，适用于纯潜在因素模型以及面板回归，具有渐近正态性，相比于现有方法具有更好的性能。

    

    本文针对潜在因素面板数据模型中的误差交叉相关问题，提出了一种简单的CD*检验方法，该方法是渐近正态的，适用于纯潜在因素模型以及面板回归。通过Monte Carlo实验研究了CD*检验的小样本性质，结果表明该检验具有正确的大小和令人满意的功率。

    In a recent paper Juodis and Reese (2021) (JR) show that the application of the CD test proposed by Pesaran (2004) to residuals from panels with latent factors results in over-rejection and propose a randomized test statistic to correct for over-rejection, and add a screening component to achieve power. This paper considers the same problem but from a different perspective and shows that the standard CD test remains valid if the latent factors are weak, and proposes a simple bias-corrected CD test, labelled CD*, which is shown to be asymptotically normal, irrespective of whether the latent factors are weak or strong. This result is shown to hold for pure latent factor models as well as for panel regressions with latent factors. Small sample properties of the CD* test are investigated by Monte Carlo experiments and are shown to have the correct size and satisfactory power for both Gaussian and non-Gaussian errors. In contrast, it is found that JR's test tends to over-reject in the case 
    

