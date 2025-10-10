# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robust Price Discrimination.](http://arxiv.org/abs/2401.16942) | 该论文研究了鲁棒的第三度价格歧视模型，在买方估值未知的情况下，通过向卖方透露信息来最大化买方剩余，并得到了关于遗憾度的界限。 |
| [^2] | [Bloated Disclosures: Can ChatGPT Help Investors Process Financial Information?.](http://arxiv.org/abs/2306.10224) | 研究发现生成式 AI 工具 ChatGPT 可以更有效地展示股票市场相关信息，提出了信息膨胀指标并证明其与负面的资本市场后果相关，同时展示其在构建针对性总结方面的效果。 |
| [^3] | [Identification- and many instrument-robust inference via invariant moment conditions.](http://arxiv.org/abs/2303.07822) | 本文介绍了一种使用不变矩条件进行多工具鲁棒性推断及识别的方法，并确定了许多工具序列下传统测试统计量的渐进正常性。 |
| [^4] | [Strategyproofness-Exposing Mechanism Descriptions.](http://arxiv.org/abs/2209.13148) | 本文研究了菜单描述在展示机制策略无懈可击性方面的作用，提出了一种新的简单菜单描述的延迟接受机制，并通过实验验证了菜单描述的优势和挑战。 |

# 详细

[^1]: 鲁棒的价格歧视

    Robust Price Discrimination. (arXiv:2401.16942v1 [econ.TH])

    [http://arxiv.org/abs/2401.16942](http://arxiv.org/abs/2401.16942)

    该论文研究了鲁棒的第三度价格歧视模型，在买方估值未知的情况下，通过向卖方透露信息来最大化买方剩余，并得到了关于遗憾度的界限。

    

    我们考虑了一个第三度价格歧视的模型，其中卖方对产品的估值未知于市场设计者，后者通过向卖方透露买方估值的信息来最大化买方剩余。我们的主要结果表明，遗憾度限制为$U^*(0)/e$，其中 $U^*(0)$是在卖方对产品没有估值的情况下的最优买方剩余。该限制通过随机抽取卖方估值并应用Bergemann等人（2015）对于抽取估值的分割方法得到。我们还证明了在二分类买方估值的情况下，$U^*(0)/e$限制是紧密的。

    We consider a model of third-degree price discrimination, in which the seller has a valuation for the product which is unknown to the market designer, who aims to maximize the buyers' surplus by revealing information regarding the buyer's valuation to the seller. Our main result shows that the regret is bounded by $U^*(0)/e$, where $U^*(0)$ is the optimal buyer surplus in the case where the seller has zero valuation for the product. This bound is attained by randomly drawing a seller valuation and applying the segmentation of Bergemann et al. (2015) with respect to the drawn valuation. We show that the $U^*(0)/e$ bound is tight in the case of binary buyer valuation.
    
[^2]: 膨胀的披露：ChatGPT是否能帮助投资者处理财务信息？

    Bloated Disclosures: Can ChatGPT Help Investors Process Financial Information?. (arXiv:2306.10224v1 [econ.GN])

    [http://arxiv.org/abs/2306.10224](http://arxiv.org/abs/2306.10224)

    研究发现生成式 AI 工具 ChatGPT 可以更有效地展示股票市场相关信息，提出了信息膨胀指标并证明其与负面的资本市场后果相关，同时展示其在构建针对性总结方面的效果。

    

    生成式 AI 工具（如 ChatGPT）可以从根本上改变投资者处理信息的方式。我们使用股票市场作为实验室，探究这些工具在总结复杂的公司披露信息时的经济效用。总结摘要明显更短，通常比原始文本缩短超过 70%，而信息内容得到增强。当一份文件具有积极（消极）情感时，其总结变得更积极（消极）。更重要的是，总结对解释股市对披露信息的反应更有效。基于这些发现，我们提出了信息“膨胀”指标。我们显示，膨胀的披露与负面的资本市场后果相关，例如更低的价格有效性和更高的信息不对称性。最后，我们展示了这个模型在构建针对性总结方面的有效性，以确定公司的（非）财务表现和风险。总之，我们的研究结果表明，像 ChatGPT 这样的生成式 AI 工具可以有效地帮助投资者更高效地处理财务信息。

    Generative AI tools such as ChatGPT can fundamentally change the way investors process information. We probe the economic usefulness of these tools in summarizing complex corporate disclosures using the stock market as a laboratory. The unconstrained summaries are dramatically shorter, often by more than 70% compared to the originals, whereas their information content is amplified. When a document has a positive (negative) sentiment, its summary becomes more positive (negative). More importantly, the summaries are more effective at explaining stock market reactions to the disclosed information. Motivated by these findings, we propose a measure of information "bloat." We show that bloated disclosure is associated with adverse capital markets consequences, such as lower price efficiency and higher information asymmetry. Finally, we show that the model is effective at constructing targeted summaries that identify firms' (non-)financial performance and risks. Collectively, our results indi
    
[^3]: 通过不变矩条件进行多工具鲁棒性推断及识别

    Identification- and many instrument-robust inference via invariant moment conditions. (arXiv:2303.07822v1 [econ.EM])

    [http://arxiv.org/abs/2303.07822](http://arxiv.org/abs/2303.07822)

    本文介绍了一种使用不变矩条件进行多工具鲁棒性推断及识别的方法，并确定了许多工具序列下传统测试统计量的渐进正常性。

    

    识别鲁棒性假设检验通常基于连续更新目标函数或其分数。当矩条件的数量与样本量成比例增长时，大维的加权矩阵阻止了使用传统的渐近逼近方法，这些测试的行为仍然未知。我们表明，当在零假设下，矩条件的分布是反射不变的时，加权矩阵的结构开辟了另一条渐近结果的路线。在异方差线性工具变量模型中，我们确定了许多工具序列下传统测试统计量的渐进正常性。一个关键结果是方差中出现的附加项为负。重新审视了关于移民和本地工人之间替代弹性的一项研究，工具的数量超过了样本大小的四分之一，许多工具鲁棒的近似值。

    Identification-robust hypothesis tests are commonly based on the continuous updating objective function or its score. When the number of moment conditions grows proportionally with the sample size, the large-dimensional weighting matrix prohibits the use of conventional asymptotic approximations and the behavior of these tests remains unknown. We show that the structure of the weighting matrix opens up an alternative route to asymptotic results when, under the null hypothesis, the distribution of the moment conditions is reflection invariant. In a heteroskedastic linear instrumental variables model, we then establish asymptotic normality of conventional tests statistics under many instrument sequences. A key result is that the additional terms that appear in the variance are negative. Revisiting a study on the elasticity of substitution between immigrant and native workers where the number of instruments is over a quarter of the sample size, the many instrument-robust approximation ind
    
[^4]: 具有策略无懈可击性的暴露机制描述

    Strategyproofness-Exposing Mechanism Descriptions. (arXiv:2209.13148v2 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2209.13148](http://arxiv.org/abs/2209.13148)

    本文研究了菜单描述在展示机制策略无懈可击性方面的作用，提出了一种新的简单菜单描述的延迟接受机制，并通过实验验证了菜单描述的优势和挑战。

    

    菜单描述以两个步骤向玩家i展示机制。第一步使用其他玩家的报告描述i的菜单：即i的潜在结果集合。第二步使用i的报告从她的菜单中选择i最喜欢的结果。菜单描述能更好地暴露策略无懈可击性吗，而不会牺牲简单性？我们提出了一个新的简单菜单描述的延迟接受机制。我们证明了，与其他常见的匹配机制相比，这种菜单描述必须与相应的传统描述有着实质性的不同。我们通过对两种基本机制的实验室实验证明了菜单描述的优势和挑战。

    A menu description presents a mechanism to player $i$ in two steps. Step (1) uses the reports of other players to describe $i$'s menu: the set of $i$'s potential outcomes. Step (2) uses $i$'s report to select $i$'s favorite outcome from her menu. Can menu descriptions better expose strategyproofness, without sacrificing simplicity? We propose a new, simple menu description of Deferred Acceptance. We prove that -- in contrast with other common matching mechanisms -- this menu description must differ substantially from the corresponding traditional description. We demonstrate, with a lab experiment on two elementary mechanisms, the promise and challenges of menu descriptions.
    

