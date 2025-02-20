# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Power of Linear Programming in Sponsored Listings Ranking: Evidence from Field Experiments](https://arxiv.org/abs/2403.14862) | 线性规划在赞助列表排名中的应用为解决在线市场中的排名挑战提供了有力工具 |
| [^2] | [On the implied volatility of European and Asian call options under the stochastic volatility Bachelier model.](http://arxiv.org/abs/2308.15341) | 本文研究了随机波动率Bachelier模型下欧式和亚式看涨期权的隐含波动率行为，并找到了与波动率模型不平滑程度有关的短期到期公式来计算隐含波动率的偏斜度。 |
| [^3] | [Sequential Elimination Contests with All-Pay Auctions.](http://arxiv.org/abs/2205.08104) | 本文研究了带全付拍卖的顺序淘汰比赛，在设计者无法得知所有选手类型情况下，对于有限的选手数量，我们明确定义了均衡策略和期望的最优机制，并为设计者提供了一些众包比赛的设计启示。 |

# 详细

[^1]: 线性规划在赞助列表排名中的威力：来自现场实验的证据

    The Power of Linear Programming in Sponsored Listings Ranking: Evidence from Field Experiments

    [https://arxiv.org/abs/2403.14862](https://arxiv.org/abs/2403.14862)

    线性规划在赞助列表排名中的应用为解决在线市场中的排名挑战提供了有力工具

    

    赞助列表是许多知名在线市场的主要收入来源之一，如亚马逊、沃尔玛和阿里巴巴。在线市场可能在展示特定商品的网页上除了该商品外，还会展示来自各种第三方卖家的赞助商品的排名列表。针对每次访问确定如何对这些赞助商品进行排名是在线市场的一个关键挑战，这个问题被称为赞助列表排名（SLR）。SLR的主要困难在于平衡最大化整体收入和推荐高质量和相关排名列表之间的权衡。虽然更相关的排名可能导致更多的购买和消费者参与，但市场在制定排名决策时还需要考虑潜在收入。

    arXiv:2403.14862v1 Announce Type: new  Abstract: Sponsored listing is one of the major revenue sources for many prominent online marketplaces, such as Amazon, Walmart, and Alibaba. When consumers visit a marketplace's webpage for a specific item, in addition to that item, the marketplace might also display a ranked listing of sponsored items from various third-party sellers. These sellers are charged an advertisement fee if a user purchases any of the sponsored items from this listing. Determining how to rank these sponsored items for each incoming visit is a crucial challenge for online marketplaces, a problem known as sponsored listings ranking (SLR). The major difficulty of SLR lies in balancing the trade-off between maximizing the overall revenue and recommending high-quality and relevant ranked listings. While a more relevant ranking may result in more purchases and consumer engagement, the marketplace also needs to take account of the potential revenue when making ranking decisio
    
[^2]: 关于随机波动率Bachelier模型下欧式和亚式看涨期权的隐含波动率研究

    On the implied volatility of European and Asian call options under the stochastic volatility Bachelier model. (arXiv:2308.15341v1 [q-fin.MF])

    [http://arxiv.org/abs/2308.15341](http://arxiv.org/abs/2308.15341)

    本文研究了随机波动率Bachelier模型下欧式和亚式看涨期权的隐含波动率行为，并找到了与波动率模型不平滑程度有关的短期到期公式来计算隐含波动率的偏斜度。

    

    本文研究了固定行权价的欧式和等差亚式看涨期权的平值隐含波动率在短期内的行为。资产价格假设遵循具有一般随机波动率过程的Bachelier模型。使用Malliavin微积分的技术，比如预测性伊藤公式，我们首先计算了当到期时间趋于零时的隐含波动率水平。然后，我们找到了一个与波动率模型的不平滑程度有关的短期到期公式，用于计算隐含波动率的偏斜度。我们将我们的普遍结果应用于SABR和分数Bergomi模型，并提供了一些数值模拟来确认偏斜度的渐近公式的准确性。

    In this paper we study the short-time behavior of the at-the-money implied volatility for European and arithmetic Asian call options with fixed strike price. The asset price is assumed to follow the Bachelier model with a general stochastic volatility process. Using techniques of the Malliavin calculus such as the anticipating It\^o's formula we first compute the level of the implied volatility when the maturity converges to zero. Then, we find a short maturity asymptotic formula for the skew of the implied volatility that depends on the roughness of the volatility model. We apply our general results to the SABR and fractional Bergomi models, and provide some numerical simulations that confirm the accurateness of the asymptotic formula for the skew.
    
[^3]: 带全付拍卖的顺序淘汰比赛研究

    Sequential Elimination Contests with All-Pay Auctions. (arXiv:2205.08104v2 [cs.GT] UPDATED)

    [http://arxiv.org/abs/2205.08104](http://arxiv.org/abs/2205.08104)

    本文研究了带全付拍卖的顺序淘汰比赛，在设计者无法得知所有选手类型情况下，对于有限的选手数量，我们明确定义了均衡策略和期望的最优机制，并为设计者提供了一些众包比赛的设计启示。

    

    本文研究了一种顺序淘汰比赛，比赛前需要过滤选手。这一模型的构建是基于许多众包比赛只有有限的评审资源，需要提高提交物品的整体质量这一实践的考虑。在第一种情况下，我们考虑选手的能力排名的知情设计者，允许 $2\leq n_2 \leq n_1$ 的前 $n_2$ 名选手参加比赛。进入比赛的选手会根据自己的能力排名可以进入前 $n_2$ 名的信号来更新对对手能力的信念。我们发现，即使具有IID先验，他们的后验信念仍然存在相关性，并且取决于选手的个人能力。我们明确地刻画了对称的和唯一的贝叶斯均衡策略。我们发现，每个被录取选手的均衡付出随 $n_2$ 的增加而增加，当 $n_2 \in [\lfloor{(n_1+1)/2}\rfloor+1,n_1]$ 时，但在 $n_2 \in [2, \lfloor{(n_1+1)/2}\rfloor]$ 时不一定单调。然后，我们考虑了更加现实的情况，设计者只具有有关选手类型的部分信息，选手类型在 $[0,1]$ 上独立均匀分布。我们得到了关于接受选手的期望最优机制性能的尖锐界限，以及每个进入比赛的选手均衡付出的界限。我们的结果对于众包比赛的设计具有重要意义。

    We study a sequential elimination contest where players are filtered prior to the round of competing for prizes. This is motivated by the practice that many crowdsourcing contests have very limited resources of reviewers and want to improve the overall quality of the submissions. We first consider a setting where the designer knows the ranking of the abilities (types) of all $n_1$ registered players, and admit the top $n_2$ players with $2\leq n_2 \leq n_1$ into the contest. The players admitted into the contest update their beliefs about their opponents based on the signal that their abilities are among the top $n_2$. We find that their posterior beliefs, even with IID priors, are correlated and depend on players' private abilities.  We explicitly characterize the symmetric and unique Bayesian equilibrium strategy. We find that each admitted player's equilibrium effort is increasing in $n_2$ when $n_2 \in [\lfloor{(n_1+1)/2}\rfloor+1,n_1]$, but not monotone in general when $n_2 \in [2
    

