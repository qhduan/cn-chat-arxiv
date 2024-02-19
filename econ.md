# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Manipulation Test for Multidimensional RDD](https://arxiv.org/abs/2402.10836) | 本文提出了多维RDD的操纵测试方法，通过理论模型推导出关于条件边际密度的可测试暗示，并基于统计量矢量的二次形式构建测试。 |
| [^2] | [Optimizing Adaptive Experiments: A Unified Approach to Regret Minimization and Best-Arm Identification](https://arxiv.org/abs/2402.10592) | 提出了一种统一模型，同时考虑了实验内部性能和实验后结果，在优化大规模人群中的表现方面提供了尖锐理论，揭示了新颖的见解 |
| [^3] | [Nowcasting with mixed frequency data using Gaussian processes](https://arxiv.org/abs/2402.10574) | 使用高斯过程和贝叶斯添加回归树作为线性惩罚估计的灵活扩展，解决了混合频率数据中的频率不匹配问题，提高了现场预测的准确性。 |
| [^4] | [The Famous American Economist H. Markowitz and Mathematical Overview of his Portfolio Selection Theory](https://arxiv.org/abs/2402.10253) | 该论文对著名美国经济学家H. Markowitz的生平及其投资组合选择理论的数学综述进行了研究。 |
| [^5] | [Robust Bayesian Method for Refutable Models.](http://arxiv.org/abs/2401.04512) | 鲁棒的贝叶斯方法适用于可被拒绝的经济模型，经济计量学家可以根据违反程度选择最不确定的假设。 |
| [^6] | [Teacher bias or measurement error?.](http://arxiv.org/abs/2401.04200) | 本研究发现，在分配学生到中学阶段时，教师的学术建议对低社会经济地位家庭的学生有偏见。但是，这个偏见可能是由测试成绩的测量误差导致的，测量误差解释了35%到43%的条件SES差距。 |
| [^7] | [The Local Projection Residual Bootstrap for AR(1) Models.](http://arxiv.org/abs/2309.01889) | 本文提出了一种局部投影残差自助法来构建AR(1)模型的脉冲响应系数的置信区间，并证明了其一致性和渐近改进性。 |
| [^8] | [Predict-AI-bility of how humans balance self-interest with the interest of others.](http://arxiv.org/abs/2307.12776) | 生成式AI能够准确预测人类在决策中平衡自身利益与他人利益的行为模式，但存在高估他人关注行为的倾向，这对AI的开发者和用户具有重要意义。 |
| [^9] | [Correlation between upstreamness and downstreamness in random global value chains.](http://arxiv.org/abs/2303.06603) | 本文研究了全球价值链中产业和国家的上游和下游，发现同一产业部门的上游和下游之间存在正相关性。 |

# 详细

[^1]: 多维RDD的操纵测试

    Manipulation Test for Multidimensional RDD

    [https://arxiv.org/abs/2402.10836](https://arxiv.org/abs/2402.10836)

    本文提出了多维RDD的操纵测试方法，通过理论模型推导出关于条件边际密度的可测试暗示，并基于统计量矢量的二次形式构建测试。

    

    Lee (2008)提出的因果推断模型适用于回归断点设计(RDD)，依赖于暗示分配（运行）变量的密度连续性的假设。这种假设的测试通常被称为操纵测试，并在应用研究中经常报告，以加强设计的有效性。多维RDD（MRDD）将RDD扩展到治疗分配取决于多个运行变量的情境。本文引入了MRDD的操纵测试。首先，它为MRDD进行因果推断的理论模型，用于推导关于运行变量的条件边际密度的可测试暗示。然后，它基于为每个边际密度单独计算的统计量矢量的二次形式构建了该假设的测试。最后，提出的测试与常用的备选程序进行了比较。

    arXiv:2402.10836v1 Announce Type: new  Abstract: The causal inference model proposed by Lee (2008) for the regression discontinuity design (RDD) relies on assumptions that imply the continuity of the density of the assignment (running) variable. The test for this implication is commonly referred to as the manipulation test and is regularly reported in applied research to strengthen the design's validity. The multidimensional RDD (MRDD) extends the RDD to contexts where treatment assignment depends on several running variables. This paper introduces a manipulation test for the MRDD. First, it develops a theoretical model for causal inference with the MRDD, used to derive a testable implication on the conditional marginal densities of the running variables. Then, it constructs the test for the implication based on a quadratic form of a vector of statistics separately computed for each marginal density. Finally, the proposed test is compared with alternative procedures commonly employed i
    
[^2]: 优化自适应实验：最小化后悔和最佳臂识别的统一方法

    Optimizing Adaptive Experiments: A Unified Approach to Regret Minimization and Best-Arm Identification

    [https://arxiv.org/abs/2402.10592](https://arxiv.org/abs/2402.10592)

    提出了一种统一模型，同时考虑了实验内部性能和实验后结果，在优化大规模人群中的表现方面提供了尖锐理论，揭示了新颖的见解

    

    进行自适应实验的从业者通常面临两个竞争性优先级：通过在实验过程中有效地分配治疗来降低实验成本，以及迅速收集信息以结束实验并在整个人群中实施治疗。当前，文献意见分歧，有关最小化后悔的研究独立地处理前者的优先级，而有关最佳臂识别的研究则专注于后者。本文提出了一种统一模型，考虑到实验内部性能和实验后结果。我们随后提供了一个针对大规模人群的最佳性能的尖锐理论，将文献中的经典结果统一起来。这种统一还揭示了新的见解。例如，理论揭示了类似最近提出的顶部两个Thompson抽样算法等熟悉算法可被调整以优化广泛类别的目标。

    arXiv:2402.10592v1 Announce Type: new  Abstract: Practitioners conducting adaptive experiments often encounter two competing priorities: reducing the cost of experimentation by effectively assigning treatments during the experiment itself, and gathering information swiftly to conclude the experiment and implement a treatment across the population. Currently, the literature is divided, with studies on regret minimization addressing the former priority in isolation, and research on best-arm identification focusing solely on the latter. This paper proposes a unified model that accounts for both within-experiment performance and post-experiment outcomes. We then provide a sharp theory of optimal performance in large populations that unifies canonical results in the literature. This unification also uncovers novel insights. For example, the theory reveals that familiar algorithms, like the recently proposed top-two Thompson sampling algorithm, can be adapted to optimize a broad class of obj
    
[^3]: 使用高斯过程进行混合频率数据的现场预测

    Nowcasting with mixed frequency data using Gaussian processes

    [https://arxiv.org/abs/2402.10574](https://arxiv.org/abs/2402.10574)

    使用高斯过程和贝叶斯添加回归树作为线性惩罚估计的灵活扩展，解决了混合频率数据中的频率不匹配问题，提高了现场预测的准确性。

    

    我们提出并讨论了用于混合数据采样（MIDAS）回归的贝叶斯机器学习方法。这涉及使用受限和非受限的MIDAS变体处理频率不匹配，并指定许多预测变量与因变量之间的函数关系。我们使用高斯过程（GP）和贝叶斯添加回归树（BART）作为线性惩罚估计的灵活扩展。在现场预测和预测练习中，我们专注于季度美国产出增长和GDP价格指数的通货膨胀。这些新模型以计算效率的方式利用宏观经济大数据，并在多个维度上提供了预测准确度的增益。

    arXiv:2402.10574v1 Announce Type: new  Abstract: We propose and discuss Bayesian machine learning methods for mixed data sampling (MIDAS) regressions. This involves handling frequency mismatches with restricted and unrestricted MIDAS variants and specifying functional relationships between many predictors and the dependent variable. We use Gaussian processes (GP) and Bayesian additive regression trees (BART) as flexible extensions to linear penalized estimation. In a nowcasting and forecasting exercise we focus on quarterly US output growth and inflation in the GDP deflator. The new models leverage macroeconomic Big Data in a computationally efficient way and offer gains in predictive accuracy along several dimensions.
    
[^4]: 著名美国经济学家H. Markowitz及其投资组合选择理论的数学综述

    The Famous American Economist H. Markowitz and Mathematical Overview of his Portfolio Selection Theory

    [https://arxiv.org/abs/2402.10253](https://arxiv.org/abs/2402.10253)

    该论文对著名美国经济学家H. Markowitz的生平及其投资组合选择理论的数学综述进行了研究。

    

    这篇调查文章致力于介绍著名美国经济学家H. Markowitz（1927-2023）的生平。我们回顾了投资组合选择理论的主要观点，以数学完整性的角度包括所有必要的辅助细节。

    arXiv:2402.10253v1 Announce Type: new  Abstract: This survey article is dedicated to the life of the famous American economist H. Markowitz (1927--2023). We do revisit the main statements of the portfolio selection theory in terms of mathematical completeness including all the necessary auxiliary details.
    
[^5]: 鲁棒的贝叶斯方法用于可证伪模型

    Robust Bayesian Method for Refutable Models. (arXiv:2401.04512v1 [econ.EM])

    [http://arxiv.org/abs/2401.04512](http://arxiv.org/abs/2401.04512)

    鲁棒的贝叶斯方法适用于可被拒绝的经济模型，经济计量学家可以根据违反程度选择最不确定的假设。

    

    我们提出了一种鲁棒的贝叶斯方法，适用于在某些数据分布下可被拒绝的经济模型。计量经济学家首先基于某种结构性假设，该假设可以被写成几个假设的交集，这个联合假设是可证伪的。为了避免模型被拒绝，经济计量学家首先选择可能被违反的假设$j$，并考虑违反这个假设$j$的程度的度量。然后她考虑对违反程度$(\pi_{m_j})$的（边际）先验偏好：她考虑了一类先验分布$\pi_s$，这些分布对所有经济结构都有相同的边际分布$\pi_m$。与将单一先验放在所有经济结构上的标准非参数贝叶斯方法相比，鲁棒的贝叶斯方法对违反程度施加了单一的边际先验分布。因此，鲁棒的贝叶斯方法允许经济计量学家仅对最不确定的假设采取立场。

    We propose a robust Bayesian method for economic models that can be rejected under some data distributions. The econometrician starts with a structural assumption which can be written as the intersection of several assumptions, and the joint assumption is refutable. To avoid the model rejection, the econometrician first takes a stance on which assumption $j$ is likely to be violated and considers a measurement of the degree of violation of this assumption $j$. She then considers a (marginal) prior belief on the degree of violation $(\pi_{m_j})$: She considers a class of prior distributions $\pi_s$ on all economic structures such that all $\pi_s$ have the same marginal distribution $\pi_m$. Compared to the standard nonparametric Bayesian method that puts a single prior on all economic structures, the robust Bayesian method imposes a single marginal prior distribution on the degree of violation. As a result, the robust Bayesian method allows the econometrician to take a stance only on th
    
[^6]: 教师偏见还是测量误差？

    Teacher bias or measurement error?. (arXiv:2401.04200v1 [econ.EM])

    [http://arxiv.org/abs/2401.04200](http://arxiv.org/abs/2401.04200)

    本研究发现，在分配学生到中学阶段时，教师的学术建议对低社会经济地位家庭的学生有偏见。但是，这个偏见可能是由测试成绩的测量误差导致的，测量误差解释了35%到43%的条件SES差距。

    

    在许多国家，教师的学术建议用于将学生分配到不同的中学阶段。先前的研究表明，低社会经济地位（SES）家庭的学生在标准化考试成绩相同的情况下，与高SES家庭的同龄人相比，他们得到的学术建议较低。通常认为这可能是教师的偏见。然而，如果存在测试成绩的测量误差，这个论断是无效的。本文讨论了测试成绩的测量误差如何导致条件SES差距的偏误，并考虑了三种实证策略来解决这种偏误。使用荷兰的行政数据，我们发现测量误差解释了学术建议中条件SES差距的35%到43%。

    In many countries, teachers' track recommendations are used to allocate students to secondary school tracks. Previous studies have shown that students from families with low socioeconomic status (SES) receive lower track recommendations than their peers from high SES families, conditional on standardized test scores. It is often argued this indicates teacher bias. However, this claim is invalid in the presence of measurement error in test scores. We discuss how measurement error in test scores generates a biased coefficient of the conditional SES gap, and consider three empirical strategies to address this bias. Using administrative data from the Netherlands, we find that measurement error explains 35 to 43% of the conditional SES gap in track recommendations.
    
[^7]: 用于AR(1)模型的局部投影残差自助法

    The Local Projection Residual Bootstrap for AR(1) Models. (arXiv:2309.01889v1 [econ.EM])

    [http://arxiv.org/abs/2309.01889](http://arxiv.org/abs/2309.01889)

    本文提出了一种局部投影残差自助法来构建AR(1)模型的脉冲响应系数的置信区间，并证明了其一致性和渐近改进性。

    

    本文在基于局部投影（LP）方法的脉冲响应系数置信区间构建领域做出了贡献。我们提出了一种LP残差自助法来构建AR（1）模型的脉冲响应系数的置信区间。该方法使用LP方法和残差自助程序来计算临界值。我们提出了两个理论结果。首先，我们证明了在一般条件下LP残差自助法的一致性，这意味着所提出的置信区间是一致渐近有效的。其次，我们证明了在某些条件下LP残差自助法可以提供置信区间的渐近改进。我们通过模拟研究来说明我们的结果。

    This paper contributes to a growing literature on confidence interval construction for impulse response coefficients based on the local projection (LP) approach. We propose an LP-residual bootstrap method to construct confidence intervals for the impulse response coefficients of AR(1) models. The method uses the LP approach and a residual bootstrap procedure to compute critical values. We present two theoretical results. First, we prove the uniform consistency of the LP-residual bootstrap under general conditions, which implies that the proposed confidence intervals are uniformly asymptotically valid. Second, we show that the LP-residual bootstrap can provide asymptotic refinements to the confidence intervals under certain conditions. We illustrate our results with a simulation study.
    
[^8]: 预测人类如何在自身利益与他人利益之间平衡的可预测性

    Predict-AI-bility of how humans balance self-interest with the interest of others. (arXiv:2307.12776v1 [econ.GN])

    [http://arxiv.org/abs/2307.12776](http://arxiv.org/abs/2307.12776)

    生成式AI能够准确预测人类在决策中平衡自身利益与他人利益的行为模式，但存在高估他人关注行为的倾向，这对AI的开发者和用户具有重要意义。

    

    生成式人工智能具有革命性的潜力，可以改变从日常生活到高风险场景的决策过程。然而，由于许多决策具有社会影响，为了使AI能够成为可靠的决策助手，它必须能够捕捉自身利益与他人利益之间的平衡。我们对三种最先进的聊天机器人对来自12个国家的78个实验的独裁者游戏决策进行了研究。我们发现，只有GPT-4（而不是Bard或Bing）能够正确捕捉到行为模式的定性特征，识别出三种主要的行为类别：自私的、不公平厌恶的和完全无私的。然而，GPT-4一直高估了他人关注行为，夸大了不公平厌恶和完全无私参与者的比例。这种偏见对于AI开发人员和用户具有重要意义。

    Generative artificial intelligence holds enormous potential to revolutionize decision-making processes, from everyday to high-stake scenarios. However, as many decisions carry social implications, for AI to be a reliable assistant for decision-making it is crucial that it is able to capture the balance between self-interest and the interest of others. We investigate the ability of three of the most advanced chatbots to predict dictator game decisions across 78 experiments with human participants from 12 countries. We find that only GPT-4 (not Bard nor Bing) correctly captures qualitative behavioral patterns, identifying three major classes of behavior: self-interested, inequity-averse, and fully altruistic. Nonetheless, GPT-4 consistently overestimates other-regarding behavior, inflating the proportion of inequity-averse and fully altruistic participants. This bias has significant implications for AI developers and users.
    
[^9]: 随机全球价值链中上游和下游之间的相关性

    Correlation between upstreamness and downstreamness in random global value chains. (arXiv:2303.06603v1 [stat.AP])

    [http://arxiv.org/abs/2303.06603](http://arxiv.org/abs/2303.06603)

    本文研究了全球价值链中产业和国家的上游和下游，发现同一产业部门的上游和下游之间存在正相关性。

    This paper studies the upstreamness and downstreamness of industries and countries in global value chains, and finds a positive correlation between upstreamness and downstreamness of the same industrial sector.

    本文关注全球价值链中产业和国家的上游和下游。上游和下游分别衡量产业部门与最终消费和初级输入之间的平均距离，并基于最常用的全球投入产出表数据库（例如世界投入产出数据库（WIOD））进行计算。最近，Antr\`as和Chor在1995-2011年的数据中报告了一个令人困惑和反直觉的发现，即（在国家层面上）上游似乎与下游呈正相关，相关斜率接近+1。这种效应随时间和跨国家稳定存在，并已得到后续分析的确认和验证。我们分析了一个简单的随机投入产出表模型，并展示了在最小和现实的结构假设下，同一产业部门的上游和下游之间存在正相关性，具有相关性。

    This paper is concerned with upstreamness and downstreamness of industries and countries in global value chains. Upstreamness and downstreamness measure respectively the average distance of an industrial sector from final consumption and from primary inputs, and they are computed from based on the most used global Input-Output tables databases, e.g., the World Input-Output Database (WIOD). Recently, Antr\`as and Chor reported a puzzling and counter-intuitive finding in data from the period 1995-2011, namely that (at country level) upstreamness appears to be positively correlated with downstreamness, with a correlation slope close to $+1$. This effect is stable over time and across countries, and it has been confirmed and validated by later analyses. We analyze a simple model of random Input/Output tables, and we show that, under minimal and realistic structural assumptions, there is a positive correlation between upstreamness and downstreamness of the same industrial sector, with corre
    

