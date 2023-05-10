# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Description Complexity of Regular Distributions.](http://arxiv.org/abs/2305.05590) | 本文研究如何在统计距离小的情况下描述正则分布的复杂度。结果表明，支持[0,1]内的正则分布需要和充分的$\tilde{\Theta}{(\epsilon^{-0.5})}$比特来描述。此外，学习混合两个正则分布物需要$\tilde{\Theta}(\epsilon^{-3})$个定价查询。 |
| [^2] | [GPT Agents in Game Theory Experiments.](http://arxiv.org/abs/2305.05516) | 本文探讨了使用GPT代理作为战略游戏实验参与者的潜力。GPT代理可以生成逼真的结果，并在某些方面表现出与人类行为一致的行为。处理效果在两个游戏中都很明显。这项初步探索表明，基于GPT的代理在博弈论实验中具有潜在的应用。 |
| [^3] | [To AI or not to AI, to Buy Local or not to Buy Local: A Mathematical Theory of Real Price.](http://arxiv.org/abs/2305.05134) | 本文建立了一个数学理论，用于确定代理人的最优全球与本地支出，从而实现代理人在支出和获得效用之间的最优平衡。 |
| [^4] | [Games Under Network Uncertainty.](http://arxiv.org/abs/2305.03124) | 本文研究了一种信息不完全的网络博弈，证明了Bayesian-Nash均衡在纯策略下的唯一性和存在性，并揭示了基于网络位置和代理人身份的不对称信息如何影响这种网络博弈中的战略行为。 |
| [^5] | [Implicit Nickell Bias in Panel Local Projection: Financial Crises Are Worse Than You Think.](http://arxiv.org/abs/2302.13455) | 本文发现面板局部投影中FE估计器存在隐式的尼克尔偏误，使得基于$t$-统计量的标准假设检验无效，我们提出使用半面板交叉检验估计器消除偏误，并通过三个关于金融危机和经济萎缩的研究发现，FE估计器严重低估了金融危机后的经济损失。 |
| [^6] | [Score-based calibration testing for multivariate forecast distributions.](http://arxiv.org/abs/2211.16362) | 该论文提出了针对多元预测分布的基于评分的校准测试方法，包括评分的概率积分变换和预期性能实际性能比较两种方法，并通过模拟测试证明其有效性。 |
| [^7] | [Binary response model with many weak instruments.](http://arxiv.org/abs/2201.04811) | 本文提供了一种使用控制函数方法和正则化方案来获得更好内生二进制响应模型估计结果的方法，并应用于研究家庭收入对大学完成的影响。 |

# 详细

[^1]: 正则分布的描述复杂度

    Description Complexity of Regular Distributions. (arXiv:2305.05590v1 [cs.GT])

    [http://arxiv.org/abs/2305.05590](http://arxiv.org/abs/2305.05590)

    本文研究如何在统计距离小的情况下描述正则分布的复杂度。结果表明，支持[0,1]内的正则分布需要和充分的$\tilde{\Theta}{(\epsilon^{-0.5})}$比特来描述。此外，学习混合两个正则分布物需要$\tilde{\Theta}(\epsilon^{-3})$个定价查询。

    

    Myerson的分布正则性是经济学中的一个标准假设。本文研究了在统计距离小的情况下描述正则分布的复杂度。我们的主要结果是，在$\epsilon$Levy距离内，需要和充分的$\tilde{\Theta}{(\epsilon^{-0.5})}$比特来描述一个支持$[0,1]$的正则分布。我们通过展示我们可以通过针对累积密度函数的$\tilde{O}(\epsilon^{-0.5})$个查询来近似地学习正则分布来证明这一点。作为推论，我们展示了在$\epsilon$Levy距离内学习支持在$[0,1]$区间的正则分布类的定价查询复杂度为$\tilde{\Theta}{(\epsilon^{-2.5})}$。学习两个正则分布混合物需要 $\tilde{\Theta}(\epsilon^{-3})$ 个定价查询。

    Myerson's regularity condition of a distribution is a standard assumption in economics. In this paper, we study the complexity of describing a regular distribution within a small statistical distance. Our main result is that $\tilde{\Theta}{(\epsilon^{-0.5})}$ bits are necessary and sufficient to describe a regular distribution with support $[0,1]$ within $\epsilon$ Levy distance. We prove this by showing that we can learn the regular distribution approximately with $\tilde{O}(\epsilon^{-0.5})$ queries to the cumulative density function. As a corollary, we show that the pricing query complexity to learn the class of regular distribution with support $[0,1]$ within $\epsilon$ Levy distance is $\tilde{\Theta}{(\epsilon^{-2.5})}$. To learn the mixture of two regular distributions, $\tilde{\Theta}(\epsilon^{-3})$ pricing queries are required.
    
[^2]: GPT Agent在博弈论实验中的应用

    GPT Agents in Game Theory Experiments. (arXiv:2305.05516v1 [econ.GN])

    [http://arxiv.org/abs/2305.05516](http://arxiv.org/abs/2305.05516)

    本文探讨了使用GPT代理作为战略游戏实验参与者的潜力。GPT代理可以生成逼真的结果，并在某些方面表现出与人类行为一致的行为。处理效果在两个游戏中都很明显。这项初步探索表明，基于GPT的代理在博弈论实验中具有潜在的应用。

    

    本文探讨了使用基于生成预训练转换器（GPT）的代理作为战略游戏实验参与者的潜力。具体而言，作者关注了在经济学中广受研究的有限重复严肃和囚徒困境两个游戏。作者设计了提示，使GPT代理能够理解游戏规则并参与其中。结果表明，在经过精心设计的提示后，GPT可以生成逼真的结果，并在某些方面表现出与人类行为一致的行为，例如在严肃游戏中，接受率与提供金额之间的正相关关系以及在囚徒困境游戏中的合作率。在一些方面，例如在多轮选择的演化方面，GPT行为与人类会有所不同。作者还研究了两种处理方式，在这两种处理方式中通过提示，GPT代理可以具有或没有社会偏好。处理效果在两个游戏中都很明显。这项初步探索表明，基于GPT的代理在博弈论实验中具有潜在的应用，为研究者研究战略行为提供了一个新的工具。

    This paper explores the potential of using Generative Pre-trained Transformer (GPT)-based agents as participants in strategic game experiments. Specifically, I focus on the finitely repeated ultimatum and prisoner's dilemma games, two well-studied games in economics. I develop prompts to enable GPT agents to understand the game rules and play the games. The results indicate that, given well-crafted prompts, GPT can generate realistic outcomes and exhibit behavior consistent with human behavior in certain important aspects, such as positive relationship between acceptance rates and offered amounts in the ultimatum game and positive cooperation rates in the prisoner's dilemma game. Some differences between the behavior of GPT and humans are observed in aspects like the evolution of choices over rounds. I also study two treatments in which the GPT agents are prompted to either have social preferences or not. The treatment effects are evident in both games. This preliminary exploration ind
    
[^3]: AI或不AI，本地购还是不本地购：真实价格的数学理论

    To AI or not to AI, to Buy Local or not to Buy Local: A Mathematical Theory of Real Price. (arXiv:2305.05134v1 [econ.TH])

    [http://arxiv.org/abs/2305.05134](http://arxiv.org/abs/2305.05134)

    本文建立了一个数学理论，用于确定代理人的最优全球与本地支出，从而实现代理人在支出和获得效用之间的最优平衡。

    

    过去几十年里，全球经济变得越来越全球化。另一方面，也有提倡“本地购”的理念，即人们购买本地生产的商品和服务而不是远离本地的那些。本文建立了一个数学理论，用于确定代理人的最优全球与本地支出，从而实现代理人在支出和获得效用之间的最优平衡。我们的真实价格理论依赖于与生产者和消费者网络相关的马尔可夫链转移概率矩阵的渐近分析。我们表明，产品或服务的真实价格可以从涉及的马尔可夫链矩阵中确定，并且可能与产品的标签价格显著不同。特别地，我们表明，产品和服务的标签价格通常不是“真实的”或直接“有用的”：如果提供相同的近视效用，价格更低的那个产品会更好。

    In the past several decades, the world's economy has become increasingly globalized. On the other hand, there are also ideas advocating the practice of ``buy local'', by which people buy locally produced goods and services rather than those produced farther away. In this paper, we establish a mathematical theory of real price that determines the optimal global versus local spending of an agent which achieves the agent's optimal tradeoff between spending and obtained utility. Our theory of real price depends on the asymptotic analysis of a Markov chain transition probability matrix related to the network of producers and consumers. We show that the real price of a product or service can be determined from the involved Markov chain matrix, and can be dramatically different from the product's label price. In particular, we show that the label prices of products and services are often not ``real'' or directly ``useful'': given two products offering the same myopic utility, the one with low
    
[^4]: 网络不确定性下的博弈论研究

    Games Under Network Uncertainty. (arXiv:2305.03124v1 [econ.TH])

    [http://arxiv.org/abs/2305.03124](http://arxiv.org/abs/2305.03124)

    本文研究了一种信息不完全的网络博弈，证明了Bayesian-Nash均衡在纯策略下的唯一性和存在性，并揭示了基于网络位置和代理人身份的不对称信息如何影响这种网络博弈中的战略行为。

    

    本文研究了一种信息不完全的网络博弈，其中代理人的信息仅限于其直接邻居的身份。代理人对其他人的邻接模式形成信念，并进行线性二次努力博弈以最大化插值回报。我们证明了纯策略下Bayesian-Nash均衡的存在性和唯一性。在这个均衡下，代理人使用本地信息，即他们的直接连接的知识来推断他们的行动与其他代理人的互补强度，这是通过他们对网络中步长数量的更新信仰所给出的。我们的模型清晰展示了基于网络位置和代理人身份的不对称信息如何影响这种网络博弈中的战略行为。我们还表征了不同形式的先验信仰下的均衡代理行为，例如对所有网络的均匀先验，Erdos-Reyni网络生成以及对网络的所有子图的同质和异质先验。

    We consider an incomplete information network game in which agents' information is restricted only to the identity of their immediate neighbors. Agents form beliefs about the adjacency pattern of others and play a linear-quadratic effort game to maximize interim payoffs. We establish the existence and uniqueness of Bayesian-Nash equilibria in pure strategies. In this equilibrium agents use local information, i.e., knowledge of their direct connections to make inferences about the complementarity strength of their actions with those of other agents which is given by their updated beliefs regarding the number of walks they have in the network. Our model clearly demonstrates how asymmetric information based on network position and the identity of agents affect strategic behavior in such network games. We also characterize agent behavior in equilibria under different forms of ex-ante prior beliefs such as uniform priors over the set of all networks, Erdos-Reyni network generation, and homo
    
[^5]: 面板局部投影中的隐式尼克尔偏误：金融危机比你想象的更糟。

    Implicit Nickell Bias in Panel Local Projection: Financial Crises Are Worse Than You Think. (arXiv:2302.13455v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2302.13455](http://arxiv.org/abs/2302.13455)

    本文发现面板局部投影中FE估计器存在隐式的尼克尔偏误，使得基于$t$-统计量的标准假设检验无效，我们提出使用半面板交叉检验估计器消除偏误，并通过三个关于金融危机和经济萎缩的研究发现，FE估计器严重低估了金融危机后的经济损失。

    

    局部投影（LP）是经济计量学中估计冲击响应的常用方法，而固定效应（FE）估计器是当LP扩展到面板数据时的默认估计方法。本文发现了由于其固有的动态结构，面板LP中FE估计器存在隐式的尼克尔偏误，使基于$t$-统计量的标准假设检验无效。我们提出使用半面板交叉检验估计器消除偏误，并展示理论结果得到蒙特卡罗模拟的支持。通过重新审视三个关于金融危机和经济萎缩之间联系的经济金融研究，我们发现FE估计器严重低估了金融危机后的经济损失。

    Local projection (LP) is a popular approach in empirical macroeconomics to estimate the impulse responses, and the conventional fixed effect (FE) estimator is the default estimation method when LP is carried over into panel data. This paper discovers an implicit Nickell bias for the FE estimator in the panel LP due to its inherent dynamic structure, invalidating the standard hypothesis testing based on the $t$-statistic. We propose using the half-panel jackknife estimator to eliminate the bias and restore the standard statistical inference, and show that the theoretical results are supported by Monte Carlo simulations. By revisiting three seminal macro-finance studies on the linkage between financial crises and economic contraction, we find that the FE estimator substantially underestimates the economic losses following financial crises.
    
[^6]: 多元预测分布的基于评分的校准测试

    Score-based calibration testing for multivariate forecast distributions. (arXiv:2211.16362v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2211.16362](http://arxiv.org/abs/2211.16362)

    该论文提出了针对多元预测分布的基于评分的校准测试方法，包括评分的概率积分变换和预期性能实际性能比较两种方法，并通过模拟测试证明其有效性。

    

    多元分布预测近年来已经广泛应用。为了评估此类预测的质量，需要合适的评估方法。在单变量情况下，基于概率积分变换的校准测试已成为常规方法。然而，基于PIT的多变量校准测试面临着各种挑战。因此，我们引入了一个适用于多元情况的普适性校准测试框架，并提出了两个由此产生的新测试。两种方法都使用适当的评分规则，即使在大维数情况下也很容易实现。第一种方法使用评分的概率积分变换。第二种方法基于将预测分布的预期性能（即预期得分）与基于实际观测数据（即实现得分）的实际性能进行比较。这些测试在模拟中具有良好的尺寸和功率特性，解决了现有测试的各种问题。我们将新测试应用于宏观经济预测分布。

    Multivariate distributional forecasts have become widespread in recent years. To assess the quality of such forecasts, suitable evaluation methods are needed. In the univariate case, calibration tests based on the probability integral transform (PIT) are routinely used. However, multivariate extensions of PIT-based calibration tests face various challenges. We therefore introduce a general framework for calibration testing in the multivariate case and propose two new tests that arise from it. Both approaches use proper scoring rules and are simple to implement even in large dimensions. The first employs the PIT of the score. The second is based on comparing the expected performance of the forecast distribution (i.e., the expected score) to its actual performance based on realized observations (i.e., the realized score). The tests have good size and power properties in simulations and solve various problems of existing tests. We apply the new tests to forecast distributions for macroeco
    
[^7]: 多个弱工具的二进制响应模型

    Binary response model with many weak instruments. (arXiv:2201.04811v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2201.04811](http://arxiv.org/abs/2201.04811)

    本文提供了一种使用控制函数方法和正则化方案来获得更好内生二进制响应模型估计结果的方法，并应用于研究家庭收入对大学完成的影响。

    

    本文考虑了具有许多弱工具的内生二进制响应模型。我们采用控制函数方法和正则化方案，在存在许多弱工具的情况下获得更好的内生二进制响应模型估计结果。提供了两个一致且渐近正态分布的估计器，分别称为正则化条件最大似然估计器（RCMLE）和正则化非线性最小二乘估计器（RNLSE）。Monte Carlo模拟表明，所提出的估计量在存在许多弱工具时优于现有的估计量。我们应用估计方法研究家庭收入对大学完成的影响。

    This paper considers an endogenous binary response model with many weak instruments. We in the current paper employ a control function approach and a regularization scheme to obtain better estimation results for the endogenous binary response model in the presence of many weak instruments. Two consistent and asymptotically normally distributed estimators are provided, each of which is called a regularized conditional maximum likelihood estimator (RCMLE) and a regularized nonlinear least square estimator (RNLSE) respectively. Monte Carlo simulations show that the proposed estimators outperform the existing estimators when many weak instruments are present. We apply our estimation method to study the effect of family income on college completion.
    

