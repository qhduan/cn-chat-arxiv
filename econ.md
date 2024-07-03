# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Shall We Talk: Exploring Spontaneous Collaborations of Competing LLM Agents](https://arxiv.org/abs/2402.12327) | 该研究揭示了LLM代理甚至在竞争环境中也能自发形成合作关系的能力，验证了计算社会科学的愿景，表明LLM代理可以用于模拟人类社会互动，包括自发合作的互动，为社会现象提供洞察。 |
| [^2] | [Inference for Rank-Rank Regressions.](http://arxiv.org/abs/2310.15512) | 本文研究了等级回归中常用的方差估计器在估计OLS估计器的渐进方差时的不一致性问题，并提出了一种一致估计器。应用新的推论方法在三个经验研究中发现，基于正确方差的估计器的置信区间可能欠精确。 |
| [^3] | [Mechanism Design for Large Language Models.](http://arxiv.org/abs/2310.10826) | 本研究主要研究了支持AI生成内容的拍卖机制设计，通过提出令牌拍卖模型，实现了以激励兼容的方式聚合多个大型语言模型，并用于结合不同广告商的输入。这个机制设计有独特的特点，并通过制定自然的激励特性得到了验证。 |
| [^4] | [Efficient Variational Inference for Large Skew-t Copulas with Application to Intraday Equity Returns.](http://arxiv.org/abs/2308.05564) | 本研究提出一种快速而准确的贝叶斯变分推理方法，用于估计大规模偏t乌鸦因子勾结模型。该方法能够捕捉到金融数据中的不对称和极端尾部相关性，以及股票对之间的异质性非对称依赖。 |

# 详细

[^1]: 我们应该交流吗：探索竞争LLM代理之间的自发合作

    Shall We Talk: Exploring Spontaneous Collaborations of Competing LLM Agents

    [https://arxiv.org/abs/2402.12327](https://arxiv.org/abs/2402.12327)

    该研究揭示了LLM代理甚至在竞争环境中也能自发形成合作关系的能力，验证了计算社会科学的愿景，表明LLM代理可以用于模拟人类社会互动，包括自发合作的互动，为社会现象提供洞察。

    

    最近的进展表明，由大型语言模型（LLMs）驱动的代理具有模拟人类行为和社会动态的能力。然而，尚未研究LLM代理在没有明确指令的情况下自发建立合作关系的潜力。为了弥补这一空白，我们进行了三项案例研究，揭示了LLM代理甚至在竞争环境中也能自发形成合作关系的能力。这一发现不仅展示了LLM代理模拟人类社会中竞争与合作的能力，也验证了计算社会科学的一个有前途的愿景。具体来说，这表明LLM代理可以用于建模人类社会互动，包括那些自发合作的互动，从而提供对社会现象的洞察。这项研究的源代码可在https://github.com/wuzengqing001225/SABM_ShallWe 找到。

    arXiv:2402.12327v1 Announce Type: new  Abstract: Recent advancements have shown that agents powered by large language models (LLMs) possess capabilities to simulate human behaviors and societal dynamics. However, the potential for LLM agents to spontaneously establish collaborative relationships in the absence of explicit instructions has not been studied. To address this gap, we conduct three case studies, revealing that LLM agents are capable of spontaneously forming collaborations even within competitive settings. This finding not only demonstrates the capacity of LLM agents to mimic competition and cooperation in human societies but also validates a promising vision of computational social science. Specifically, it suggests that LLM agents could be utilized to model human social interactions, including those with spontaneous collaborations, thus offering insights into social phenomena. The source codes for this study are available at https://github.com/wuzengqing001225/SABM_ShallWe
    
[^2]: 推论用于等级回归

    Inference for Rank-Rank Regressions. (arXiv:2310.15512v1 [econ.EM])

    [http://arxiv.org/abs/2310.15512](http://arxiv.org/abs/2310.15512)

    本文研究了等级回归中常用的方差估计器在估计OLS估计器的渐进方差时的不一致性问题，并提出了一种一致估计器。应用新的推论方法在三个经验研究中发现，基于正确方差的估计器的置信区间可能欠精确。

    

    在等级回归中，斜率系数是衡量代际流动性的常用指标，例如在子女收入等级与父母收入等级回归中。本文首先指出，常用的方差估计器如同方差估计器或鲁棒方差估计器未能一致估计OLS估计器在等级回归中的渐进方差。我们表明，这些估计器的概率极限可能过大或过小，取决于子女收入和父母收入的联合分布函数的形状。其次，我们导出了等级回归的一般渐进理论，并提供了OLS估计器渐进方差的一致估计器。然后，我们将渐进理论扩展到其他经验工作中涉及等级的回归。最后，我们将新的推论方法应用于三个经验研究。我们发现，基于正确方差的估计器的置信区间有时可能欠精确。

    Slope coefficients in rank-rank regressions are popular measures of intergenerational mobility, for instance in regressions of a child's income rank on their parent's income rank. In this paper, we first point out that commonly used variance estimators such as the homoskedastic or robust variance estimators do not consistently estimate the asymptotic variance of the OLS estimator in a rank-rank regression. We show that the probability limits of these estimators may be too large or too small depending on the shape of the copula of child and parent incomes. Second, we derive a general asymptotic theory for rank-rank regressions and provide a consistent estimator of the OLS estimator's asymptotic variance. We then extend the asymptotic theory to other regressions involving ranks that have been used in empirical work. Finally, we apply our new inference methods to three empirical studies. We find that the confidence intervals based on estimators of the correct variance may sometimes be sub
    
[^3]: 大型语言模型的机制设计

    Mechanism Design for Large Language Models. (arXiv:2310.10826v1 [cs.GT])

    [http://arxiv.org/abs/2310.10826](http://arxiv.org/abs/2310.10826)

    本研究主要研究了支持AI生成内容的拍卖机制设计，通过提出令牌拍卖模型，实现了以激励兼容的方式聚合多个大型语言模型，并用于结合不同广告商的输入。这个机制设计有独特的特点，并通过制定自然的激励特性得到了验证。

    

    我们研究拍卖机制以支持新兴的AI生成内容的格式。我们特别研究如何以激励兼容的方式聚合多个LLM。在这个问题中，每个代理对随机生成的内容的偏好被描述/编码为一个LLM。设计一个用于AI生成的广告创意的拍卖格式来结合来自不同广告商的输入是一个关键动机。我们认为，尽管这个问题通常属于机制设计的范畴，但它具有一些独特的特点。我们提出了一个通用的形式化方法——令牌拍卖模型——来研究这个问题。这个模型的一个关键特点是它以令牌为单位进行操作，并允许LLM代理通过一维出价影响生成的内容。我们首先探讨了一个强大的拍卖设计方法，其中我们假设的是代理人的偏好涉及到结果分布的偏序。我们制定了两个自然的激励特性，并证明了这些特性的重要性。

    We investigate auction mechanisms to support the emerging format of AI-generated content. We in particular study how to aggregate several LLMs in an incentive compatible manner. In this problem, the preferences of each agent over stochastically generated contents are described/encoded as an LLM. A key motivation is to design an auction format for AI-generated ad creatives to combine inputs from different advertisers. We argue that this problem, while generally falling under the umbrella of mechanism design, has several unique features. We propose a general formalism -- the token auction model -- for studying this problem. A key feature of this model is that it acts on a token-by-token basis and lets LLM agents influence generated contents through single dimensional bids.  We first explore a robust auction design approach, in which all we assume is that agent preferences entail partial orders over outcome distributions. We formulate two natural incentive properties, and show that these 
    
[^4]: 大规模偏t乌鸦勾结的高效变分推理及其在股票收益率中的应用

    Efficient Variational Inference for Large Skew-t Copulas with Application to Intraday Equity Returns. (arXiv:2308.05564v1 [econ.EM])

    [http://arxiv.org/abs/2308.05564](http://arxiv.org/abs/2308.05564)

    本研究提出一种快速而准确的贝叶斯变分推理方法，用于估计大规模偏t乌鸦因子勾结模型。该方法能够捕捉到金融数据中的不对称和极端尾部相关性，以及股票对之间的异质性非对称依赖。

    

    大规模偏t乌鸦因子勾结模型对金融数据建模具有吸引力，因为它们允许不对称和极端的尾部相关性。我们展示了Azzalini和Capitanio（2003）所隐含的乌鸦勾结在成对非对称依赖性方面比两种流行的乌鸦勾结更高。在高维情况下，对该乌鸦勾结的估计具有挑战性，我们提出了一种快速而准确的贝叶斯变分推理方法来解决这个问题。该方法使用条件高斯生成表示法定义了一个可以准确近似的附加后验。使用快速随机梯度上升算法来解决变分优化。这种新的方法被用来估计2017年至2021年间93个美国股票的股票收益率的勾结模型。除了成对相关性的变化外，该勾结还捕捉到了股票对之间的非对称依赖的大量异质性。

    Large skew-t factor copula models are attractive for the modeling of financial data because they allow for asymmetric and extreme tail dependence. We show that the copula implicit in the skew-t distribution of Azzalini and Capitanio (2003) allows for a higher level of pairwise asymmetric dependence than two popular alternative skew-t copulas. Estimation of this copula in high dimensions is challenging, and we propose a fast and accurate Bayesian variational inference (VI) approach to do so. The method uses a conditionally Gaussian generative representation of the skew-t distribution to define an augmented posterior that can be approximated accurately. A fast stochastic gradient ascent algorithm is used to solve the variational optimization. The new methodology is used to estimate copula models for intraday returns from 2017 to 2021 on 93 U.S. equities. The copula captures substantial heterogeneity in asymmetric dependence over equity pairs, in addition to the variability in pairwise co
    

