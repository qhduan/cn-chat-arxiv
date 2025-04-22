# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fused LASSO as Non-Crossing Quantile Regression](https://arxiv.org/abs/2403.14036) | 通过扩展非交叉约束，通过变化单一超参数（$\alpha$），可以获得常用的分位数估计量，显示了非交叉约束只是融合收缩的一个特殊类型。 |
| [^2] | [Inference for Synthetic Controls via Refined Placebo Tests.](http://arxiv.org/abs/2401.07152) | 本文提出了一种通过精细安慰剂测试进行合成控制的推断方法，用于解决只有一个处理单元和少数对照单元的问题，并解决了样本量小和实际估计过程简化的问题。 |
| [^3] | [Nonparametric estimation of k-modal taste heterogeneity for group level agent-based mixed logit.](http://arxiv.org/abs/2309.13159) | 这项研究提出了一种群体级别的代理人混合logit方法，通过使用逆优化和群体级别的市场份额进行估计，该方法克服了现有方法的局限性，并在一个纽约州的案例研究中进行了验证。 |
| [^4] | [Auctions with Tokens: Monetary Policy as a Mechanism Design Choice.](http://arxiv.org/abs/2301.13794) | 本文研究了基于区块链代币的机制设计，通过研究重复的私有价值拍卖发现，拍卖与代币相比于美元和金融证券拍卖，具有提前积累收入、变动较小和更适用于严重契约摩擦的优点。 |
| [^5] | [Policy Learning with New Treatments.](http://arxiv.org/abs/2210.04703) | 本文研究了使用新的处理方式进行政策学习的问题，通过对处理反应的形状限制进行部分识别，提出了一种基于最小后悔准则比较不同政策的方法，并应用于肯尼亚农村电网连接的有针对性的补贴设计。 |
| [^6] | [Wildfire Modeling: Designing a Market to Restore Assets.](http://arxiv.org/abs/2205.13773) | 该论文研究了如何设计一个市场来恢复由森林火灾造成的资产损失。研究通过分析电力公司引发火灾的原因，并将火灾风险纳入经济模型中，提出了收取森林火灾基金和公平收费的方案，以最大化社会影响和总盈余。 |

# 详细

[^1]: Fused LASSO作为非交叉分位数回归

    Fused LASSO as Non-Crossing Quantile Regression

    [https://arxiv.org/abs/2403.14036](https://arxiv.org/abs/2403.14036)

    通过扩展非交叉约束，通过变化单一超参数（$\alpha$），可以获得常用的分位数估计量，显示了非交叉约束只是融合收缩的一个特殊类型。

    

    分位数交叉一直是分位数回归中一个长久存在的问题，推动了对获得遵守分位数单调性性质的密度和系数的研究。本文扩展了非交叉约束，展示通过变化单一超参数（$\alpha$）可以获得常用的分位数估计量。具体来说，当 $\alpha=0$ 时，我们获得了Koenker和Bassett（1978）的分位数回归估计量，当 $\alpha=1$ 时，获得了Bondell等人（2010）的非交叉分位数回归估计量，当 $\alpha\rightarrow\infty$ 时，获得了Koenker（1984）和Zou以及Yuan（2008）的复合分位数回归估计量。因此，我们展示了非交叉约束只是融合收缩的一个特殊类型。

    arXiv:2403.14036v1 Announce Type: new  Abstract: Quantile crossing has been an ever-present thorn in the side of quantile regression. This has spurred research into obtaining densities and coefficients that obey the quantile monotonicity property. While important contributions, these papers do not provide insight into how exactly these constraints influence the estimated coefficients. This paper extends non-crossing constraints and shows that by varying a single hyperparameter ($\alpha$) one can obtain commonly used quantile estimators. Namely, we obtain the quantile regression estimator of Koenker and Bassett (1978) when $\alpha=0$, the non crossing quantile regression estimator of Bondell et al. (2010) when $\alpha=1$, and the composite quantile regression estimator of Koenker (1984) and Zou and Yuan (2008) when $\alpha\rightarrow\infty$. As such, we show that non-crossing constraints are simply a special type of fused-shrinkage.
    
[^2]: 通过精细安慰剂测试进行合成控制的推断

    Inference for Synthetic Controls via Refined Placebo Tests. (arXiv:2401.07152v1 [stat.ME])

    [http://arxiv.org/abs/2401.07152](http://arxiv.org/abs/2401.07152)

    本文提出了一种通过精细安慰剂测试进行合成控制的推断方法，用于解决只有一个处理单元和少数对照单元的问题，并解决了样本量小和实际估计过程简化的问题。

    

    合成控制方法通常用于只有一个处理单元和少数对照单元的问题。在这种情况下，一种常见的推断任务是测试关于对待处理单元的平均处理效应的零假设。由于（1）样本量较小导致大样本近似不稳定和（2）在实践中实施的估计过程的简化，因此通常无法满足渐近合理性的推断程序常常不令人满意。一种替代方法是置换推断，它与常见的称为安慰剂测试的诊断相关。当治疗均匀分配时，它在有限样本中具有可证明的 Type-I 错误保证，而无需简化方法。尽管具有这种健壮性，安慰剂测试由于只从 $N$ 个参考估计构造零分布，其中 $N$ 是样本量，因此分辨率较低。这在常见的水平 $\alpha = 0.05$ 的统计推断中形成了一个障碍，特别是在小样本问题中。

    The synthetic control method is often applied to problems with one treated unit and a small number of control units. A common inferential task in this setting is to test null hypotheses regarding the average treatment effect on the treated. Inference procedures that are justified asymptotically are often unsatisfactory due to (1) small sample sizes that render large-sample approximation fragile and (2) simplification of the estimation procedure that is implemented in practice. An alternative is permutation inference, which is related to a common diagnostic called the placebo test. It has provable Type-I error guarantees in finite samples without simplification of the method, when the treatment is uniformly assigned. Despite this robustness, the placebo test suffers from low resolution since the null distribution is constructed from only $N$ reference estimates, where $N$ is the sample size. This creates a barrier for statistical inference at a common level like $\alpha = 0.05$, especia
    
[^3]: 非参数性估计k模式的群体级别基于混合logit的口味异质性

    Nonparametric estimation of k-modal taste heterogeneity for group level agent-based mixed logit. (arXiv:2309.13159v1 [econ.EM])

    [http://arxiv.org/abs/2309.13159](http://arxiv.org/abs/2309.13159)

    这项研究提出了一种群体级别的代理人混合logit方法，通过使用逆优化和群体级别的市场份额进行估计，该方法克服了现有方法的局限性，并在一个纽约州的案例研究中进行了验证。

    

    使用大规模信息和通信技术（ICT）数据集估计特定代理人的口味异质性需要模型的灵活性和计算效率。我们提出了一种基于群体级别的代理人混合（GLAM）logit方法，该方法使用逆优化（IO）和群体级别的市场份额进行估计。该模型在理论上与RUM模型框架一致，而估计方法是一种非参数方法，适用于市场级别的数据集，克服了现有方法的局限性。我们使用 Replica Inc. 提供的合成人口数据集进行了纽约州出行方式选择的案例研究，其中包括2019年秋季和2021年秋季两个典型工作日1953万名居民的出行方式选择。将个体出行方式选择按照人口普查区组OD对和四个人口段分组，得到120,740个群体级别的代理人。我们使用GLAM logit模型校准了该模型。

    Estimating agent-specific taste heterogeneity with a large information and communication technology (ICT) dataset requires both model flexibility and computational efficiency. We propose a group-level agent-based mixed (GLAM) logit approach that is estimated with inverse optimization (IO) and group-level market share. The model is theoretically consistent with the RUM model framework, while the estimation method is a nonparametric approach that fits to market-level datasets, which overcomes the limitations of existing approaches. A case study of New York statewide travel mode choice is conducted with a synthetic population dataset provided by Replica Inc., which contains mode choices of 19.53 million residents on two typical weekdays, one in Fall 2019 and another in Fall 2021. Individual mode choices are grouped into market-level market shares per census block-group OD pair and four population segments, resulting in 120,740 group-level agents. We calibrate the GLAM logit model with the
    
[^4]: 拍卖与代币：货币政策作为机制设计选择

    Auctions with Tokens: Monetary Policy as a Mechanism Design Choice. (arXiv:2301.13794v2 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2301.13794](http://arxiv.org/abs/2301.13794)

    本文研究了基于区块链代币的机制设计，通过研究重复的私有价值拍卖发现，拍卖与代币相比于美元和金融证券拍卖，具有提前积累收入、变动较小和更适用于严重契约摩擦的优点。

    

    本文研究了基于区块链代币的机制设计，即在机制内可以使用的代币同时也可以在机制外保存和交易。通过研究一个重复的私有价值拍卖，拍卖者接受自己创建和拥有的基于区块链的代币作为支付。研究发现，预期收入的折现价值与标准货币拍卖相同，但这些收入的积累较早且变动较小。最优的货币政策涉及到燃烧用于拍卖的代币，这是许多基于区块链的拍卖的共同特征。进一步引入不可契约的努力和收入侵占的可能性。通过将代币拍卖与可以发行金融证券的美元拍卖进行比较，发现在存在严重的契约摩擦时，代币拍卖更受青睐，而当契约摩擦较低时，则相反。

    I study mechanism design with blockchain-based tokens, that is, tokens that can be used within a mechanism but can also be saved and traded outside of the mechanism. I do so by considering a repeated, private-value auction, in which the auctioneer accepts payments in a blockchain-based token he creates and initially owns. I show that the present-discounted value of the expected revenues is the same as in a standard auction with dollars, but these revenues accrue earlier and are less variable. The optimal monetary policy involves the burning of tokens used in the auction, a common feature of many blockchain-based auctions. I then introduce non-contractible effort and the possibility of misappropriating revenues. I compare the auction with tokens to an auction with dollars in which the auctioneer can also issue financial securities. An auction with tokens is preferred when there are sufficiently severe contracting frictions, while the opposite is true when contracting frictions are low.
    
[^5]: 使用新的处理方式进行政策学习

    Policy Learning with New Treatments. (arXiv:2210.04703v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2210.04703](http://arxiv.org/abs/2210.04703)

    本文研究了使用新的处理方式进行政策学习的问题，通过对处理反应的形状限制进行部分识别，提出了一种基于最小后悔准则比较不同政策的方法，并应用于肯尼亚农村电网连接的有针对性的补贴设计。

    

    我研究了决策者根据实验数据选择政策以在异质人群中分配处理方式的问题，该实验数据仅包括可能处理方式的子集。新的处理方式的效果是通过对处理反应的形状限制进行部分识别的。根据最小后悔准则比较不同的政策，并且我证明了人口决策问题的经验类比具有可行的线性和整数规划形式。我证明了估计政策的最大后悔收敛于最低的最大后悔，其速度是N^-1/2和实验数据中条件平均处理效应估计的速度的最大值。我应用我的结果设计了肯尼亚农村电网连接的有针对性的补贴，估计有97%的人口应该得到未在实验中实施的处理方式。

    I study the problem of a decision maker choosing a policy which allocates treatment to a heterogeneous population on the basis of experimental data that includes only a subset of possible treatment values. The effects of new treatments are partially identified by shape restrictions on treatment response. Policies are compared according to the minimax regret criterion, and I show that the empirical analog of the population decision problem has a tractable linear- and integer-programming formulation. I prove the maximum regret of the estimated policy converges to the lowest possible maximum regret at a rate which is the maximum of N^-1/2 and the rate at which conditional average treatment effects are estimated in the experimental data. I apply my results to design targeted subsidies for electrical grid connections in rural Kenya, and estimate that 97% of the population should be given a treatment not implemented in the experiment.
    
[^6]: 森林火灾模型：设计市场以恢复资产

    Wildfire Modeling: Designing a Market to Restore Assets. (arXiv:2205.13773v3 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2205.13773](http://arxiv.org/abs/2205.13773)

    该论文研究了如何设计一个市场来恢复由森林火灾造成的资产损失。研究通过分析电力公司引发火灾的原因，并将火灾风险纳入经济模型中，提出了收取森林火灾基金和公平收费的方案，以最大化社会影响和总盈余。

    

    在过去的十年里，夏季森林火灾已经成为加利福尼亚和美国的常态。这些火灾的原因多种多样。州政府会收取森林火灾基金来帮助受灾人员。然而，基金只在特定条件下发放，并且在整个加利福尼亚州均匀收取。因此，该项目的整体思路是寻找关于电力公司如何引发火灾以及如何帮助收取森林火灾基金或者公平收费以最大限度地实现社会影响的数量结果。该研究项目旨在提出与植被、输电线路相关的森林火灾风险，并将其与金钱挂钩。因此，该项目有助于解决与每个地点相关的森林火灾基金收取问题，并结合能源价格根据地点的森林火灾风险向客户收费，以实现社会的总盈余最大化。

    In the past decade, summer wildfires have become the norm in California, and the United States of America. These wildfires are caused due to variety of reasons. The state collects wildfire funds to help the impacted customers. However, the funds are eligible only under certain conditions and are collected uniformly throughout California. Therefore, the overall idea of this project is to look for quantitative results on how electrical corporations cause wildfires and how they can help to collect the wildfire funds or charge fairly to the customers to maximize the social impact. The research project aims to propose the implication of wildfire risk associated with vegetation, and due to power lines and incorporate that in dollars. Therefore, the project helps to solve the problem of collecting wildfire funds associated with each location and incorporate energy prices to charge their customers according to their wildfire risk related to the location to maximize the social surplus for the s
    

