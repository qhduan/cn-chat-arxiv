# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Inference for Regression with Variables Generated from Unstructured Data](https://arxiv.org/abs/2402.15585) | 提出了一种使用联合上游和下游模型进行有效推断的一步策略，显著减少了偏误，在CEO时间利用数据的应用中产生了重要效果，适合应用研究人员。 |
| [^2] | [Estimating the Effects of Fiscal Policy using a Novel Proxy Shrinkage Prior.](http://arxiv.org/abs/2302.13066) | 本文提出了一种新型代理收缩先验方法，能够在不依赖于代理变量有效性的强烈假设的情况下，估计财政政策冲击的效应。研究发现，增加政府支出比减税更有效地刺激经济，并构建了新的外生代理变量，可以用于传统的代理VAR方法中，与代理收缩模型的结果相似。 |
| [^3] | [On the Non-Identification of Revenue Production Functions.](http://arxiv.org/abs/2212.04620) | 当将收入作为产出的代理变量时，不能识别生产函数和Hicks-中性生产力。只有对基础需求系统进行假设的方法可能识别生产函数。 |

# 详细

[^1]: 使用来自非结构化数据生成的变量进行回归的推断

    Inference for Regression with Variables Generated from Unstructured Data

    [https://arxiv.org/abs/2402.15585](https://arxiv.org/abs/2402.15585)

    提出了一种使用联合上游和下游模型进行有效推断的一步策略，显著减少了偏误，在CEO时间利用数据的应用中产生了重要效果，适合应用研究人员。

    

    分析非结构化数据的主要策略包括两个步骤。首先，使用上游信息检索模型估计感兴趣的潜在经济变量。其次，将估计值视为下游计量经济模型中的“数据”。我们建立了理论论点，解释为什么在实证合理的设置中，这种两步策略会导致偏误的推断。更具建设性的是，我们提出了一个有效推断的一步策略，该策略同时使用上游和下游模型。在模拟中，这一步策略(i) 显著减少了偏误；(ii) 在使用CEO时间利用数据的主要应用中产生了定量重要的效果；(iii) 可以很容易地被应用研究人员采用。

    arXiv:2402.15585v1 Announce Type: new  Abstract: The leading strategy for analyzing unstructured data uses two steps. First, latent variables of economic interest are estimated with an upstream information retrieval model. Second, the estimates are treated as "data" in a downstream econometric model. We establish theoretical arguments for why this two-step strategy leads to biased inference in empirically plausible settings. More constructively, we propose a one-step strategy for valid inference that uses the upstream and downstream models jointly. The one-step strategy (i) substantially reduces bias in simulations; (ii) has quantitatively important effects in a leading application using CEO time-use data; and (iii) can be readily adapted by applied researchers.
    
[^2]: 使用新型代理收缩先验估计财政政策效应

    Estimating the Effects of Fiscal Policy using a Novel Proxy Shrinkage Prior. (arXiv:2302.13066v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2302.13066](http://arxiv.org/abs/2302.13066)

    本文提出了一种新型代理收缩先验方法，能够在不依赖于代理变量有效性的强烈假设的情况下，估计财政政策冲击的效应。研究发现，增加政府支出比减税更有效地刺激经济，并构建了新的外生代理变量，可以用于传统的代理VAR方法中，与代理收缩模型的结果相似。

    

    在财政政策结构向量自回归中，不同的代理变量会得出冲突的结论，意味着某些外生性假设可能无法满足。我们将数据驱动识别与新型代理收缩先验相结合，使我们能够在不依赖于代理变量有效性的强烈假设的情况下估计财政政策冲击的效应。我们的研究结果表明，增加政府支出是刺激经济的更有效工具比减税。此外，我们提供证据表明，文献中常用的代理变量与结构性冲击具有内生相关性，这导致估计结果存在偏差。我们构建了新的外生代理变量，可以用于传统的代理 VAR 方法中，结果与我们的代理收缩模型相似。

    Different proxy variables commonly used in fiscal policy SVARs lead to contradicting conclusions implying that some of the exogeneity assumptions may not be fulfilled. We combine data-driven identification with a novel proxy shrinkage prior which enables us to estimate the effects of fiscal policy shocks without relying on strong assumptions about the validity of the proxy variables. Our results suggest that increasing government spending is a more effective tool to stimulate the economy than reducing taxes. Additionally, we provide evidence that the commonly used proxies in the literature are endogenously related to the structural shocks which leads to biased estimates. We construct new exogenous proxies that can be used in the traditional proxy VAR approach resulting in similar estimates compared to our proxy shrinkage model.
    
[^3]: 关于收入生产函数的非识别问题

    On the Non-Identification of Revenue Production Functions. (arXiv:2212.04620v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2212.04620](http://arxiv.org/abs/2212.04620)

    当将收入作为产出的代理变量时，不能识别生产函数和Hicks-中性生产力。只有对基础需求系统进行假设的方法可能识别生产函数。

    

    当将收入作为产出的代理变量时，生产函数可能会出现误设。我通过将这个常识形式化和加强，展示出既不能用生产函数，也不能用Hicks-中性生产力来识别这种收入代理。这个结果适用于文献中常用的大类生产函数，包括所有常用的参数形式。在解决这个问题的各种方法中，只有对基础需求系统进行假设的方法，才能可能识别生产函数。

    Production functions are potentially misspecified when revenue is used as a proxy for output. I formalize and strengthen this common knowledge by showing that neither the production function nor Hicks-neutral productivity can be identified with such a revenue proxy. This result holds under the standard assumptions used in the literature for a large class of production functions, including all commonly used parametric forms. Among the prevalent approaches to address this issue, only those that impose assumptions on the underlying demand system can possibly identify the production function.
    

