# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A path-dependent PDE solver based on signature kernels](https://arxiv.org/abs/2403.11738) | 该论文开发了一种基于特征核的路径依赖PDE求解器，证明了其一致性和收敛性，并展示了在期权定价领域的数值示例。 |
| [^2] | [A Unifying Approach for the Pricing of Debt Securities](https://arxiv.org/abs/2403.06303) | 提出了一种统一的框架，用于在一般的时间非齐次短期利率扩散过程下定价债券，包括债券、债券期权、可赎回/可买入债券和可转换债券；通过CTMC近似获得了闭式矩阵表达式来近似计算债券和债券期权价格；开发了用于定价可赎回/可买入债务的简单高效算法；可以将近似模型完美拟合到当前市场利率期限结构。 |
| [^3] | [Shall We Talk: Exploring Spontaneous Collaborations of Competing LLM Agents](https://arxiv.org/abs/2402.12327) | 该研究揭示了LLM代理甚至在竞争环境中也能自发形成合作关系的能力，验证了计算社会科学的愿景，表明LLM代理可以用于模拟人类社会互动，包括自发合作的互动，为社会现象提供洞察。 |
| [^4] | [Tournaments, Contestant Heterogeneity and Performance.](http://arxiv.org/abs/2401.05210) | 本文使用实地数据研究发现，选手之间的技能差异对绩效产生不对称影响，对低能选手有负面影响，对高能选手有正面影响。同时讨论了行为方法的解释和竞赛设计的最优影响。 |
| [^5] | [Pricing and hedging for a sticky diffusion.](http://arxiv.org/abs/2311.17011) | 该论文研究了粘性扩散模型的定价与对冲问题。研究发现当利率为零时模型是无套利的，并找到了唯一的无风险复制策略和定价方程。研究还发现模型中存在一类可复制的偿付，与传统的Black-Scholes模型相一致。最后，通过数值评估研究了离散时间对冲和由于误判价格粘性而产生的对冲误差。 |
| [^6] | [Causal Feature Engineering of Price Directions of Cryptocurrencies using Dynamic Bayesian Networks.](http://arxiv.org/abs/2306.08157) | 本文提出了一种基于动态贝叶斯网络的方法，来预测加密货币价格方向，以帮助投资者做出明智的投资决策。 |
| [^7] | [Econotaxis in modeling urbanization by labor force migration.](http://arxiv.org/abs/2303.09720) | 本研究提出了一个劳动力迁移模型，通过模拟发现模型可以产生聚集行为，并展现了两个经验规律。更进一步，研究证明了经济性趋向性，这是一种新型人类行为中心的趋向性，可以解释在现实世界中的劳动力聚集现象，这一结论突显了城市化与所导出的PDE系统中的吹起现象的相关性。 |

# 详细

[^1]: 基于特征核的路径依赖PDE求解器

    A path-dependent PDE solver based on signature kernels

    [https://arxiv.org/abs/2403.11738](https://arxiv.org/abs/2403.11738)

    该论文开发了一种基于特征核的路径依赖PDE求解器，证明了其一致性和收敛性，并展示了在期权定价领域的数值示例。

    

    我们开发了一种基于特征核的路径依赖PDE（PPDE）的收敛证明求解器。我们的数值方案利用了特征核，这是最近在路径空间上引入的一类核。具体来说，我们通过在符号再生核希尔伯特空间（RKHS）中近似PPDE的解来解决一个最优恢复问题，该空间受到在有限集合的远程路径上满足PPDE约束的元素的约束。在线性情况下，我们证明了优化具有唯一的闭式解，其以远程路径处的特征核评估的形式表示。我们证明了所提出方案的一致性，保证在远程点数增加时收敛到PPDE解。最后，我们提供了几个数值例子，尤其是在粗糙波动率下的期权定价背景下。我们的数值方案构成了一种替代性的蒙特卡洛方法的有效替代方案。

    arXiv:2403.11738v1 Announce Type: cross  Abstract: We develop a provably convergent kernel-based solver for path-dependent PDEs (PPDEs). Our numerical scheme leverages signature kernels, a recently introduced class of kernels on path-space. Specifically, we solve an optimal recovery problem by approximating the solution of a PPDE with an element of minimal norm in the signature reproducing kernel Hilbert space (RKHS) constrained to satisfy the PPDE at a finite collection of collocation paths. In the linear case, we show that the optimisation has a unique closed-form solution expressed in terms of signature kernel evaluations at the collocation paths. We prove consistency of the proposed scheme, guaranteeing convergence to the PPDE solution as the number of collocation points increases. Finally, several numerical examples are presented, in particular in the context of option pricing under rough volatility. Our numerical scheme constitutes a valid alternative to the ubiquitous Monte Carl
    
[^2]: 一种关于债券定价的统一方法

    A Unifying Approach for the Pricing of Debt Securities

    [https://arxiv.org/abs/2403.06303](https://arxiv.org/abs/2403.06303)

    提出了一种统一的框架，用于在一般的时间非齐次短期利率扩散过程下定价债券，包括债券、债券期权、可赎回/可买入债券和可转换债券；通过CTMC近似获得了闭式矩阵表达式来近似计算债券和债券期权价格；开发了用于定价可赎回/可买入债务的简单高效算法；可以将近似模型完美拟合到当前市场利率期限结构。

    

    我们提出了一个统一的框架，用于在一般的时间非齐次短期利率扩散过程下定价债券。涵盖了债券、债券期权、可赎回/可买入债券和可转换债券(CBs)的定价。通过连续时间马尔可夫链 (CTMC) 近似，我们获得了用于在一般一维短期利率过程下近似计算债券和债券期权价格的闭式矩阵表达式。还开发了一种简单且高效的算法来定价可赎回/可买入债务。零息债券价格的闭式表达式的可用性允许将近似模型完美拟合到当前市场利率期限结构，无论所选的基础扩散过程的复杂性如何。我们进一步考虑了在一般的双向时间非齐次扩散过程下对可转换债券（CBs）的定价，以建模股票和短期利率动力学。也考虑了信用风险。

    arXiv:2403.06303v1 Announce Type: new  Abstract: We propose a unifying framework for the pricing of debt securities under general time-inhomogeneous short-rate diffusion processes. The pricing of bonds, bond options, callable/putable bonds, and convertible bonds (CBs) are covered. Using continuous-time Markov chain (CTMC) approximation, we obtain closed-form matrix expressions to approximate the price of bonds and bond options under general one-dimensional short-rate processes. A simple and efficient algorithm is also developed to price callable/putable debts. The availability of a closed-form expression for the price of zero-coupon bonds allows for the perfect fit of the approximated model to the current market term structure of interest rates, regardless of the complexity of the underlying diffusion process selected. We further consider the pricing of CBs under general bi-dimensional time-inhomogeneous diffusion processes to model equity and short-rate dynamics. Credit risk is also i
    
[^3]: 我们应该交流吗：探索竞争LLM代理之间的自发合作

    Shall We Talk: Exploring Spontaneous Collaborations of Competing LLM Agents

    [https://arxiv.org/abs/2402.12327](https://arxiv.org/abs/2402.12327)

    该研究揭示了LLM代理甚至在竞争环境中也能自发形成合作关系的能力，验证了计算社会科学的愿景，表明LLM代理可以用于模拟人类社会互动，包括自发合作的互动，为社会现象提供洞察。

    

    最近的进展表明，由大型语言模型（LLMs）驱动的代理具有模拟人类行为和社会动态的能力。然而，尚未研究LLM代理在没有明确指令的情况下自发建立合作关系的潜力。为了弥补这一空白，我们进行了三项案例研究，揭示了LLM代理甚至在竞争环境中也能自发形成合作关系的能力。这一发现不仅展示了LLM代理模拟人类社会中竞争与合作的能力，也验证了计算社会科学的一个有前途的愿景。具体来说，这表明LLM代理可以用于建模人类社会互动，包括那些自发合作的互动，从而提供对社会现象的洞察。这项研究的源代码可在https://github.com/wuzengqing001225/SABM_ShallWe 找到。

    arXiv:2402.12327v1 Announce Type: new  Abstract: Recent advancements have shown that agents powered by large language models (LLMs) possess capabilities to simulate human behaviors and societal dynamics. However, the potential for LLM agents to spontaneously establish collaborative relationships in the absence of explicit instructions has not been studied. To address this gap, we conduct three case studies, revealing that LLM agents are capable of spontaneously forming collaborations even within competitive settings. This finding not only demonstrates the capacity of LLM agents to mimic competition and cooperation in human societies but also validates a promising vision of computational social science. Specifically, it suggests that LLM agents could be utilized to model human social interactions, including those with spontaneous collaborations, thus offering insights into social phenomena. The source codes for this study are available at https://github.com/wuzengqing001225/SABM_ShallWe
    
[^4]: 锦标赛、选手异质性和绩效

    Tournaments, Contestant Heterogeneity and Performance. (arXiv:2401.05210v1 [econ.GN])

    [http://arxiv.org/abs/2401.05210](http://arxiv.org/abs/2401.05210)

    本文使用实地数据研究发现，选手之间的技能差异对绩效产生不对称影响，对低能选手有负面影响，对高能选手有正面影响。同时讨论了行为方法的解释和竞赛设计的最优影响。

    

    锦标赛经常被用作激励机制来提高绩效。本文使用实地数据，并展示了选手之间的技能差异对选手绩效的不对称影响。技能差异对低能选手的绩效有负面影响，但对高能选手的绩效有正面影响。我们讨论了不同行为方法来解释我们的研究结果，并讨论了结果对竞赛的最优设计的影响。此外，我们的研究揭示了两个重要的实证结果：(a) 象争取平权政策可能有助于减轻低能选手的不利影响，(b) 后续比赛阶段潜在未来选手的技能水平可能对高能选手的绩效产生不利影响，但不会影响低能选手。

    Tournaments are frequently used incentive mechanisms to enhance performance. In this paper, we use field data and show that skill disparities among contestants asymmetrically affect the performance of contestants. Skill disparities have detrimental effects on the performance of the lower-ability contestant but positive effects on the performance of the higher-ability contestant. We discuss the potential of different behavioral approaches to explain our findings and discuss the implications of our results for the optimal design of contests. Beyond that, our study reveals two important empirical results: (a) affirmative action-type policies may help to mitigate the adverse effects on lower-ability contestants, and (b) the skill level of potential future contestants in subsequent tournament stages can detrimentally influence the performance of higher-ability contestants but does not affect the lower-ability contestant.
    
[^5]: 一个关于粘性扩散的定价与对冲研究

    Pricing and hedging for a sticky diffusion. (arXiv:2311.17011v3 [q-fin.MF] UPDATED)

    [http://arxiv.org/abs/2311.17011](http://arxiv.org/abs/2311.17011)

    该论文研究了粘性扩散模型的定价与对冲问题。研究发现当利率为零时模型是无套利的，并找到了唯一的无风险复制策略和定价方程。研究还发现模型中存在一类可复制的偿付，与传统的Black-Scholes模型相一致。最后，通过数值评估研究了离散时间对冲和由于误判价格粘性而产生的对冲误差。

    

    我们考虑了一个金融市场模型，其特点是具有粘性几何布朗运动价格动态的风险资产和一个常数利率$r \in \mathbb R$。我们证明了该模型只有当$r = 0$时才是无套利的。在这种情况下，我们找到了唯一的无风险复制策略，并推导出相关的定价方程。我们还确定了一类可复制的偿付，与标准的Black-Scholes模型中的可复制的偿付相一致。最后，我们对离散时间对冲和由于误判价格粘性而产生的对冲误差进行了数值评估。

    We consider a financial market model featuring a risky asset with a sticky geometric Brownian motion price dynamic and a constant interest rate $r \in \mathbb R$. We prove that the model is arbitrage-free if and only if $r =0 $. In this case, we find the unique riskless replication strategy, derive the associated pricing equation. We also identify a class of replicable payoffs that coincides with the replicable payoffs in the standard Black-Scholes model. Last, we numerically evaluate discrete-time hedging and the hedging error incurred from misrepresenting price stickiness.
    
[^6]: 使用动态贝叶斯网络进行加密货币价格方向因果特征工程

    Causal Feature Engineering of Price Directions of Cryptocurrencies using Dynamic Bayesian Networks. (arXiv:2306.08157v1 [cs.LG])

    [http://arxiv.org/abs/2306.08157](http://arxiv.org/abs/2306.08157)

    本文提出了一种基于动态贝叶斯网络的方法，来预测加密货币价格方向，以帮助投资者做出明智的投资决策。

    

    加密货币在各个领域，特别是金融和投资领域中越来越受到关注。其独特的区块链相关特性，如隐私、去中心化和不可追踪性，部分原因是其受欢迎的原因。然而，由于加密货币价格的波动性和不确定性，加密货币仍然是一种高风险投资。本文提出了一个动态贝叶斯网络（DBN）方法，可以在多元设置下模拟复杂系统，以预测五种流行加密货币的价格运动方向，以解决这个问题。

    Cryptocurrencies have gained popularity across various sectors, especially in finance and investment. The popularity is partly due to their unique specifications originating from blockchain-related characteristics such as privacy, decentralisation, and untraceability. Despite their growing popularity, cryptocurrencies remain a high-risk investment due to their price volatility and uncertainty. The inherent volatility in cryptocurrency prices, coupled with internal cryptocurrency-related factors and external influential global economic factors makes predicting their prices and price movement directions challenging. Nevertheless, the knowledge obtained from predicting the direction of cryptocurrency prices can provide valuable guidance for investors in making informed investment decisions. To address this issue, this paper proposes a dynamic Bayesian network (DBN) approach, which can model complex systems in multivariate settings, to predict the price movement direction of five popular a
    
[^7]: 劳动力迁移模拟中的经济性趋向性

    Econotaxis in modeling urbanization by labor force migration. (arXiv:2303.09720v1 [nlin.AO])

    [http://arxiv.org/abs/2303.09720](http://arxiv.org/abs/2303.09720)

    本研究提出了一个劳动力迁移模型，通过模拟发现模型可以产生聚集行为，并展现了两个经验规律。更进一步，研究证明了经济性趋向性，这是一种新型人类行为中心的趋向性，可以解释在现实世界中的劳动力聚集现象，这一结论突显了城市化与所导出的PDE系统中的吹起现象的相关性。

    

    本研究采用主动布朗粒子框架，提出了一个简单的劳动力迁移微观模型。通过基于代理的模拟，我们发现我们的模型产生了从随机初始分布中聚集到一起的一群代理。此外，在我们的模型中观察到了Zipf和Okun定律这两个经验规律。为了揭示产生的聚集现象背后的机制，我们从我们的微观模型中导出了一个扩展的Keller-Segel系统。得到的宏观系统表明人力资源在现实世界中的聚集可以通过一种新型人类行为中心的趋向性来解释，这突显了城市化与所导出的PDE系统中的吹起现象的相关性。我们将其称为“经济性趋向性”。

    Individual participants in human society collectively exhibit aggregation behavior. In this study, we present a simple microscopic model of labor force migration by employing the active Brownian particles framework. Through agent-based simulations, we find that our model produces clusters of agents from a random initial distribution. Furthermore, two empirical regularities called Zipf's and Okun's laws were observed in our model. To reveal the mechanism underlying the reproduced agglomeration phenomena, we derived an extended Keller-Segel system, a classic model that describes the aggregation behavior of biological organisms called "taxis," from our microscopic model. The obtained macroscopic system indicates that the agglomeration of the workforce in real world can be accounted for through a new type of taxis central to human behavior, which highlights the relevance of urbanization to blow-up phenomena in the derived PDE system. We term it "econotaxis."
    

