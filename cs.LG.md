# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Proximal Oracles for Optimization and Sampling](https://arxiv.org/abs/2404.02239) | 论文研究了具有非光滑函数和对数凹抽样的凸优化问题，提出了在优化和抽样中应用近端框架的方法，并建立了近端映射的迭代复杂度。 |
| [^2] | [Bandit Convex Optimisation](https://arxiv.org/abs/2402.06535) | 这篇论文介绍了强盗凸优化的基本框架和用于解决这一问题的多种工具。虽然没有太多创新，但通过以新颖的方式应用现有工具，获得了新的算法和改进了一些界限。 |
| [^3] | [Knowledge Enhanced Conditional Imputation for Healthcare Time-series.](http://arxiv.org/abs/2312.16713) | 本研究提出了一种知识增强的条件插补方法，针对医疗时间序列数据中的缺失数据问题。通过整合先进的知识嵌入和非均匀掩蔽策略，该方法能够灵活适应不同模式的电子健康记录中的缺失数据分布不平衡问题。 |
| [^4] | [Prompt Injection Attacks and Defenses in LLM-Integrated Applications.](http://arxiv.org/abs/2310.12815) | 本文提出了一个通用框架来形式化提示注入攻击，并系统化防御这种类型的攻击。 |
| [^5] | [Online Ensemble of Models for Optimal Predictive Performance with Applications to Sector Rotation Strategy.](http://arxiv.org/abs/2304.09947) | 通过机器学习模型和资产特定因素在预测行业回报和测量行业特定风险溢价方面获得更大经济收益，开发了一种新型在线集成算法来学习优化预测性能，特别适用于时间序列问题和可能的黑盒模型系统。 |

# 详细

[^1]: 优化和抽样的近端预言

    Proximal Oracles for Optimization and Sampling

    [https://arxiv.org/abs/2404.02239](https://arxiv.org/abs/2404.02239)

    论文研究了具有非光滑函数和对数凹抽样的凸优化问题，提出了在优化和抽样中应用近端框架的方法，并建立了近端映射的迭代复杂度。

    

    我们考虑具有非光滑目标函数和对数凹抽样（带非光滑潜势，即负对数密度）的凸优化。特别地，我们研究了两种具体设置，其中凸目标/潜势函数要么是半光滑的，要么是复合形式，作为半光滑分量的有限和。为了克服由于非光滑性而带来的挑战，我们的算法在优化和抽样中采用了两种强大的近端框架：优化中的近端点框架和替代抽样框架（ASF），该框架在增广分布上使用Gibbs抽样。优化和抽样算法的一个关键组件是通过正则化切平面方法高效实现近端映射。我们在半光滑和复合设置中建立了近端映射的迭代复杂度。我们进一步提出了一种用于非光滑优化的自适应近端捆绑方法。

    arXiv:2404.02239v1 Announce Type: cross  Abstract: We consider convex optimization with non-smooth objective function and log-concave sampling with non-smooth potential (negative log density). In particular, we study two specific settings where the convex objective/potential function is either semi-smooth or in composite form as the finite sum of semi-smooth components. To overcome the challenges caused by non-smoothness, our algorithms employ two powerful proximal frameworks in optimization and sampling: the proximal point framework for optimization and the alternating sampling framework (ASF) that uses Gibbs sampling on an augmented distribution. A key component of both optimization and sampling algorithms is the efficient implementation of the proximal map by the regularized cutting-plane method. We establish the iteration-complexity of the proximal map in both semi-smooth and composite settings. We further propose an adaptive proximal bundle method for non-smooth optimization. The 
    
[^2]: Bandit Convex Optimisation（强盗凸优化）

    Bandit Convex Optimisation

    [https://arxiv.org/abs/2402.06535](https://arxiv.org/abs/2402.06535)

    这篇论文介绍了强盗凸优化的基本框架和用于解决这一问题的多种工具。虽然没有太多创新，但通过以新颖的方式应用现有工具，获得了新的算法和改进了一些界限。

    

    强盗凸优化是研究零阶凸优化的基本框架。本文介绍了用于解决该问题的许多工具，包括切平面方法、内点方法、连续指数权重、梯度下降和在线牛顿步骤。解释了许多假设和设置之间的细微差别。尽管在这里没有太多真正新的东西，但一些现有工具以新颖的方式应用于获得新算法。一些界限稍微改进了一些。

    Bandit convex optimisation is a fundamental framework for studying zeroth-order convex optimisation. These notes cover the many tools used for this problem, including cutting plane methods, interior point methods, continuous exponential weights, gradient descent and online Newton step. The nuances between the many assumptions and setups are explained. Although there is not much truly new here, some existing tools are applied in novel ways to obtain new algorithms. A few bounds are improved in minor ways.
    
[^3]: 知识增强的医疗时间序列条件插补方法

    Knowledge Enhanced Conditional Imputation for Healthcare Time-series. (arXiv:2312.16713v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2312.16713](http://arxiv.org/abs/2312.16713)

    本研究提出了一种知识增强的条件插补方法，针对医疗时间序列数据中的缺失数据问题。通过整合先进的知识嵌入和非均匀掩蔽策略，该方法能够灵活适应不同模式的电子健康记录中的缺失数据分布不平衡问题。

    

    本研究提出了一种新颖的方法来解决多变量时间序列中的缺失数据问题，特别关注医疗数据的复杂性。我们的条件自注意力插补（CSAI）模型以基于Transformer的框架为基础，引入了一种针对医疗时间序列数据细节的条件隐藏状态初始化方式。该方法与传统的插补技术不同，它特别针对医疗数据集中缺失数据分布的不平衡问题，这一关键问题常常被忽视。通过整合先进的知识嵌入和非均匀掩蔽策略，CSAI能够灵活适应电子健康记录（EHR）中缺失数据的不同模式。

    This study presents a novel approach to addressing the challenge of missing data in multivariate time series, with a particular focus on the complexities of healthcare data. Our Conditional Self-Attention Imputation (CSAI) model, grounded in a transformer-based framework, introduces a conditional hidden state initialization tailored to the intricacies of medical time series data. This methodology diverges from traditional imputation techniques by specifically targeting the imbalance in missing data distribution, a crucial aspect often overlooked in healthcare datasets. By integrating advanced knowledge embedding and a non-uniform masking strategy, CSAI adeptly adjusts to the distinct patterns of missing data in Electronic Health Records (EHRs).
    
[^4]: LLM-集成应用中的提示注入攻击和防御

    Prompt Injection Attacks and Defenses in LLM-Integrated Applications. (arXiv:2310.12815v1 [cs.CR])

    [http://arxiv.org/abs/2310.12815](http://arxiv.org/abs/2310.12815)

    本文提出了一个通用框架来形式化提示注入攻击，并系统化防御这种类型的攻击。

    

    大型语言模型（LLMs）越来越多地用作各种称为LLM-集成应用的实际应用程序的后端。最近的多项研究表明，LLM-集成应用容易受到提示注入攻击的威胁，攻击者可以将恶意指令/数据注入这些应用程序的输入中，以达到攻击者的预期结果。然而，现有的研究仅限于案例研究，缺乏对提示注入攻击及其防御的系统理解。本论文旨在填补这一空白。我们提出了一个通用框架来形式化提示注入攻击，并将研究论文和博客文章中讨论的现有攻击视为我们框架的特例。我们的框架使我们能够通过组合现有攻击设计新的攻击方式。此外，我们还提出了一个系统化提示注入攻击防御的框架。利用我们的框架，我们可以预防和缓解这种类型的攻击。

    Large Language Models (LLMs) are increasingly deployed as the backend for a variety of real-world applications called LLM-Integrated Applications. Multiple recent works showed that LLM-Integrated Applications are vulnerable to prompt injection attacks, in which an attacker injects malicious instruction/data into the input of those applications such that they produce results as the attacker desires. However, existing works are limited to case studies. As a result, the literature lacks a systematic understanding of prompt injection attacks and their defenses. We aim to bridge the gap in this work. In particular, we propose a general framework to formalize prompt injection attacks. Existing attacks, which are discussed in research papers and blog posts, are special cases in our framework. Our framework enables us to design a new attack by combining existing attacks. Moreover, we also propose a framework to systematize defenses against prompt injection attacks. Using our frameworks, we con
    
[^5]: 在线模型集成对最优预测性能的应用和行业轮换策略

    Online Ensemble of Models for Optimal Predictive Performance with Applications to Sector Rotation Strategy. (arXiv:2304.09947v1 [q-fin.ST])

    [http://arxiv.org/abs/2304.09947](http://arxiv.org/abs/2304.09947)

    通过机器学习模型和资产特定因素在预测行业回报和测量行业特定风险溢价方面获得更大经济收益，开发了一种新型在线集成算法来学习优化预测性能，特别适用于时间序列问题和可能的黑盒模型系统。

    

    资产特定因素通常用于预测金融回报并量化资产特定风险溢价。我们使用各种机器学习模型证明，这些因素包含的信息可以在预测行业回报和测量行业特定风险溢价方面带来更大的经济收益。为了利用不同行业表现的单个模型的强预测结果，我们开发了一种新型在线集成算法，该算法学习优化预测性能。该算法随着时间的推移不断适应，通过分析它们最近的预测性能来确定个体模型的最佳组合。这使它特别适用于时间序列问题，滚动窗口回测程序和可能的黑盒模型系统。我们推导出最优增益函数，用样本外R平方度量表达相应的遗憾界，并推导出最优解。

    Asset-specific factors are commonly used to forecast financial returns and quantify asset-specific risk premia. Using various machine learning models, we demonstrate that the information contained in these factors leads to even larger economic gains in terms of forecasts of sector returns and the measurement of sector-specific risk premia. To capitalize on the strong predictive results of individual models for the performance of different sectors, we develop a novel online ensemble algorithm that learns to optimize predictive performance. The algorithm continuously adapts over time to determine the optimal combination of individual models by solely analyzing their most recent prediction performance. This makes it particularly suited for time series problems, rolling window backtesting procedures, and systems of potentially black-box models. We derive the optimal gain function, express the corresponding regret bounds in terms of the out-of-sample R-squared measure, and derive optimal le
    

