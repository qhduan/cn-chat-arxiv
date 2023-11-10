# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Cheap Talking Algorithms.](http://arxiv.org/abs/2310.07867) | 该论文研究了在战略信息传递游戏中，利用独立强化学习算法进行训练的发送者和接收者可以收敛到接近最优均衡策略，并且在代理之间的利益冲突下实现了最大化的通信。这一结论稳健，并对信息传递游戏中的均衡选择理论、计算机科学中的算法间通信和人工智能代理市场中的经济学产生了影响。 |
| [^2] | [Unveiling the Interplay between Central Bank Digital Currency and Bank Deposits.](http://arxiv.org/abs/2308.10359) | 本研究分析了中央银行数字货币（CBDC）引入后对金融稳定性的风险。当CBDC和存款完全可替代时，中央银行可以通过提供贷款来中性化CBDC的影响，贷款利率取决于抵押约束的严格程度。然而，当CBDC和存款不完全可替代时，中央银行无法使银行对来自CBDC的竞争无所适从。 |
| [^3] | [Identification-robust inference for the LATE with high-dimensional covariates.](http://arxiv.org/abs/2302.09756) | 本文提出了一种适用于高维协变量下的局部平均处理效应的检验统计量，证明其具有统一正确的大小，并通过双重/无偏机器学习方法实现了推断和置信区间计算。模拟结果表明，该检验具有鲁棒性，可以有效处理识别力较弱和高维设置下的数据。应用于实证研究中，该方法在铁路通达对城市人口增长的影响研究中表现出更短的置信区间和更小的点估计。 |
| [^4] | [A Vector Monotonicity Assumption for Multiple Instruments.](http://arxiv.org/abs/2009.00553) | 本文提出了向量单调性假设的观点，该假设对于多个工具变量组合的研究具有重要意义。通过假设所有工具变量的治疗接受程度是单调的，我们可以得到一些因果参数的点识别，并提供了相应的估计方法。 |

# 详细

[^1]: 廉价对话算法

    Cheap Talking Algorithms. (arXiv:2310.07867v1 [econ.TH])

    [http://arxiv.org/abs/2310.07867](http://arxiv.org/abs/2310.07867)

    该论文研究了在战略信息传递游戏中，利用独立强化学习算法进行训练的发送者和接收者可以收敛到接近最优均衡策略，并且在代理之间的利益冲突下实现了最大化的通信。这一结论稳健，并对信息传递游戏中的均衡选择理论、计算机科学中的算法间通信和人工智能代理市场中的经济学产生了影响。

    

    我们模拟独立的强化学习算法在克劳福德和索贝尔（1982）的战略信息传递游戏中的行为。我们表明，一个发送者和一个接收者一起进行训练，收敛到接近游戏先验最优均衡的策略。因此，通信在与代理之间的利益冲突程度给出的纳什均衡下，按照最大程度进行。这一结论对超参数和游戏的备选规范稳健。我们讨论了信息传递游戏中均衡选择理论、计算机科学中算法间新兴通信工作以及由人工智能代理组成的市场中的宫斗经济学的影响。

    We simulate behaviour of independent reinforcement learning algorithms playing the Crawford and Sobel (1982) game of strategic information transmission. We show that a sender and a receiver training together converge to strategies close to the exante optimal equilibrium of the game. Hence, communication takes place to the largest extent predicted by Nash equilibrium given the degree of conflict of interest between agents. The conclusion is shown to be robust to alternative specifications of the hyperparameters and of the game. We discuss implications for theories of equilibrium selection in information transmission games, for work on emerging communication among algorithms in computer science and for the economics of collusions in markets populated by artificially intelligent agents.
    
[^2]: 揭示中央银行数字货币与银行存款之间的相互作用

    Unveiling the Interplay between Central Bank Digital Currency and Bank Deposits. (arXiv:2308.10359v1 [econ.TH])

    [http://arxiv.org/abs/2308.10359](http://arxiv.org/abs/2308.10359)

    本研究分析了中央银行数字货币（CBDC）引入后对金融稳定性的风险。当CBDC和存款完全可替代时，中央银行可以通过提供贷款来中性化CBDC的影响，贷款利率取决于抵押约束的严格程度。然而，当CBDC和存款不完全可替代时，中央银行无法使银行对来自CBDC的竞争无所适从。

    

    我们扩展了Niepelt（2022年）的实际商业周期模型，分析引入中央银行数字货币（CBDC）后面临的金融稳定风险。CBDC与商业银行存款竞争，成为家庭的流动性来源。我们考虑了支付工具之间的不同可替代性程度，并通过引入银行从中央银行借款时必须遵守的抵押约束来审查Niepelt（2022年）中的等价结果。当CBDC和存款完全可替代时，中央银行可以向银行提供贷款，使引入CBDC对实体经济中性。我们表明，中央银行的贷款利率的最优水平取决于抵押约束的限制性程度：抵押约束越严格，中央银行需要发布的贷款利率越低。然而，当CBDC和存款不完全可替代时，中央银行无法使银行对来自CBDC的竞争无所适从。因此，引入CBDC后将出现负面影响。

    We extend the Real Business Cycle model in Niepelt (2022) to analyze the risk to financial stability following the introduction of a central bank digital currency (CBDC). CBDC competes with commercial bank deposits as households' source of liquidity. We consider different degrees of substitutability between payment instruments and review the equivalence result in Niepelt (2022) by introducing a collateral constraint banks must respect when borrowing from the central bank. When CBDC and deposits are perfect substitutes, the central bank can offer loans to banks that render the introduction of CBDC neutral to the real economy. We show that the optimal level of the central bank's lending rate depends on the restrictiveness of the collateral constraint: the tighter it is, the lower the loan rate the central bank needs to post. However, when CBDC and deposits are imperfect substitutes, the central bank cannot make banks indifferent to the competition from CBDC. It follows that the introduct
    
[^3]: 高维协变量下的局部平均处理效应鲁棒性推断

    Identification-robust inference for the LATE with high-dimensional covariates. (arXiv:2302.09756v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2302.09756](http://arxiv.org/abs/2302.09756)

    本文提出了一种适用于高维协变量下的局部平均处理效应的检验统计量，证明其具有统一正确的大小，并通过双重/无偏机器学习方法实现了推断和置信区间计算。模拟结果表明，该检验具有鲁棒性，可以有效处理识别力较弱和高维设置下的数据。应用于实证研究中，该方法在铁路通达对城市人口增长的影响研究中表现出更短的置信区间和更小的点估计。

    

    本文研究了高维协变量下的局部平均处理效应(LATE)，不论识别力如何。我们提出了一种新的高维LATE检验统计量，并证明了我们的检验在渐进情况下具有统一正确的大小。通过采用双重/无偏机器学习(DML)方法来估计干扰参数，我们开发了简单易实施的算法来推断和计算高维LATE的置信区间。模拟结果表明，我们的检验对于识别力较弱和高维设置下的大小控制和功效表现具有鲁棒性，优于其他传统检验方法。将所提出的检验应用于铁路和人口数据，研究铁路通达对城市人口增长的影响，我们观察到与传统检验相比，铁路通达系数的置信区间长度更短，点估计更小。

    This paper investigates the local average treatment effect (LATE) with high-dimensional covariates, irrespective of the strength of identification. We propose a novel test statistic for the high-dimensional LATE, demonstrating that our test has uniformly correct asymptotic size. By employing the double/debiased machine learning (DML) method to estimate nuisance parameters, we develop easy-to-implement algorithms for inference and confidence interval calculation of the high-dimensional LATE. Simulations indicate that our test is robust against both weak identification and high-dimensional setting concerning size control and power performance, outperforming other conventional tests. Applying the proposed test to railroad and population data to study the effect of railroad access on urban population growth, we observe the shorter length of confidence intervals and smaller point estimates for the railroad access coefficients compared to the conventional tests.
    
[^4]: 多个工具变量的向量单调性假设

    A Vector Monotonicity Assumption for Multiple Instruments. (arXiv:2009.00553v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2009.00553](http://arxiv.org/abs/2009.00553)

    本文提出了向量单调性假设的观点，该假设对于多个工具变量组合的研究具有重要意义。通过假设所有工具变量的治疗接受程度是单调的，我们可以得到一些因果参数的点识别，并提供了相应的估计方法。

    

    当研究人员将多个工具变量组合用于单一二元治疗时，本地平均治疗效应（LATE）框架的单调性假设可能变得过于限制性：它要求所有单位在分别改变工具变量方向时具有相同的反应方向。相比之下，我所称的向量单调性仅仅假设所有工具变量的治疗接受程度是单调的，它是Mogstad等人引入的部分单调性假设的特殊情况。在工具变量为二元变量时，我刻画了在向量单调性下被点识别的因果参数类。该类包括对任何一种方式对工具变量集合做出响应的单位的平均治疗效应，或对给定子集做出响应的单位的平均治疗效应。识别结果是建设性的，并提供了对已识别的治疗效应参数的简单估计器。

    When a researcher combines multiple instrumental variables for a single binary treatment, the monotonicity assumption of the local average treatment effects (LATE) framework can become restrictive: it requires that all units share a common direction of response even when separate instruments are shifted in opposing directions. What I call vector monotonicity, by contrast, simply assumes treatment uptake to be monotonic in all instruments, representing a special case of the partial monotonicity assumption introduced by Mogstad et al. (2021). I characterize the class of causal parameters that are point identified under vector monotonicity, when the instruments are binary. This class includes, for example, the average treatment effect among units that are in any way responsive to the collection of instruments, or those that are responsive to a given subset of them. The identification results are constructive and yield a simple estimator for the identified treatment effect parameters. An e
    

