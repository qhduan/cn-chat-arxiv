# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [am-AMM: An Auction-Managed Automated Market Maker](https://arxiv.org/abs/2403.03367) | 本文提出了一个针对AMMs两个重要未解决问题的单一机制：减少对有信息的订单流的损失，最大化对无信息的订单流的收益。 |
| [^2] | [Meritocracy and Its Discontents: Long-run Effects of Repeated School Admission Reforms](https://arxiv.org/abs/2402.04429) | 该论文通过分析世界上第一次引入全国中央化的优质教育招生制度改革，发现了持久的优质教育与公平之间的权衡。中央化系统相较于分散化系统录取了更多优秀的申请者，长期来看产生了更多顶尖精英官僚，但这也导致了地区接触优质高等教育和职业晋升的不平等。几十年后，优质教育的集中化增加了城市出生的职业精英数量相对于农村出生者。 |
| [^3] | [Convergence of the deep BSDE method for stochastic control problems formulated through the stochastic maximum principle](https://arxiv.org/abs/2401.17472) | 本文研究了基于深度BSDE方法和随机最大原则的随机控制问题，提供了该方法的收敛结果，并展示了在高维问题中相比其他方法具有卓越性能。 |
| [^4] | [Leverage Staking with Liquid Staking Derivatives (LSDs): Opportunities and Risks.](http://arxiv.org/abs/2401.08610) | 这项研究系统地研究了Liquid Staking Derivatives (LSDs)的杠杆质押机会与风险。他们发现杠杆质押在Lido-Aave生态系统中能够实现较高的回报，并有潜力通过优化策略获得更多收益。 |
| [^5] | [Efficient Variational Inference for Large Skew-t Copulas with Application to Intraday Equity Returns.](http://arxiv.org/abs/2308.05564) | 本研究提出一种快速而准确的贝叶斯变分推理方法，用于估计大规模偏t乌鸦因子勾结模型。该方法能够捕捉到金融数据中的不对称和极端尾部相关性，以及股票对之间的异质性非对称依赖。 |

# 详细

[^1]: am-AMM: 一个拍卖管理的自动做市商

    am-AMM: An Auction-Managed Automated Market Maker

    [https://arxiv.org/abs/2403.03367](https://arxiv.org/abs/2403.03367)

    本文提出了一个针对AMMs两个重要未解决问题的单一机制：减少对有信息的订单流的损失，最大化对无信息的订单流的收益。

    

    自动做市商（AMMs）已成为区块链上去中心化交易所的主要市场机制。本文提出了一个针对AMMs两个重要未解决问题的单一机制：减少对有信息的订单流的损失，最大化对无信息的订单流的收益。这个“拍卖管理的AMM”通过在链上进行一次不受审查的拍卖，以临时行使“流动性池管理者”职能的权利。流动性池管理者设置池的交换费率，并从交换中获得的费用。流动性池管理者可以通过针对小价格波动而对冲流动性池来独占一些套利，并且可以设置交换费用，包括零售订单流的价格敏感性，并适应不断变化的市场条件，最终将两者的收益归结为流动性提供方。流动性提供方可以进入和退出池…

    arXiv:2403.03367v1 Announce Type: new  Abstract: Automated market makers (AMMs) have emerged as the dominant market mechanism for trading on decentralized exchanges implemented on blockchains. This paper presents a single mechanism that targets two important unsolved problems for AMMs: reducing losses to informed orderflow, and maximizing revenue from uninformed orderflow. The "auction-managed AMM" works by running a censorship-resistant onchain auction for the right to temporarily act as "pool manager" for a constant-product AMM. The pool manager sets the swap fee rate on the pool, and also receives the accrued fees from swaps. The pool manager can exclusively capture some arbitrage by trading against the pool in response to small price movements, and also can set swap fees incorporating price sensitivity of retail orderflow and adapting to changing market conditions, with the benefits from both ultimately accruing to liquidity providers. Liquidity providers can enter and exit the poo
    
[^2]: 优质教育方法与其不满：重复学校招生改革的长期影响

    Meritocracy and Its Discontents: Long-run Effects of Repeated School Admission Reforms

    [https://arxiv.org/abs/2402.04429](https://arxiv.org/abs/2402.04429)

    该论文通过分析世界上第一次引入全国中央化的优质教育招生制度改革，发现了持久的优质教育与公平之间的权衡。中央化系统相较于分散化系统录取了更多优秀的申请者，长期来看产生了更多顶尖精英官僚，但这也导致了地区接触优质高等教育和职业晋升的不平等。几十年后，优质教育的集中化增加了城市出生的职业精英数量相对于农村出生者。

    

    如果精英学院改变他们的入学政策会发生什么？通过分析20世纪初全球首次引入全国中央化的优质招生制度，我们回答了这个问题。我们发现存在持久的优质教育与公平之间的权衡。相较于分散化系统，中央化系统录取了更多优秀的申请者，在长期内产生了更多顶尖精英官僚。然而，这种影响以公平地区接触精英高等教育和职业晋升的代价为代表。几十年后，优质教育的集中化增加了城市出生的职业精英（例如，高收入者）相对于农村出生的数量。

    What happens if selective colleges change their admission policies? We answer this question by analyzing the world's first introduction of nationally centralized meritocratic admissions in the early twentieth century. We find a persistent meritocracy-equity tradeoff. Compared to the decentralized system, the centralized system admitted more high-achieving applicants, producing a greater number of top elite bureaucrats in the long run. However, this impact came at the distributional cost of equal regional access to elite higher education and career advancement. Several decades later, the meritocratic centralization increased the number of urban-born career elites (e.g., top income earners) relative to rural-born ones.
    
[^3]: 通过随机最大原则，基于深度BSDE方法的随机控制问题的收敛性研究

    Convergence of the deep BSDE method for stochastic control problems formulated through the stochastic maximum principle

    [https://arxiv.org/abs/2401.17472](https://arxiv.org/abs/2401.17472)

    本文研究了基于深度BSDE方法和随机最大原则的随机控制问题，提供了该方法的收敛结果，并展示了在高维问题中相比其他方法具有卓越性能。

    

    众所周知，随机控制的决策问题可以通过前向后向随机微分方程（FBSDE）来表述。最近，Ji等人（2022）提出了一种基于随机最大原则（SMP）的高效深度学习算法。本文提供了该深度SMP-BSDE算法的收敛结果，并将其性能与其他现有方法进行比较。通过采用类似于Han和Long（2020）的策略，我们推导出后验误差估计，并展示了总近似误差可以由损失函数值和离散化误差的值来限制。我们在高维随机控制问题的数值例子中展示了该算法在漂移控制和扩散控制的情况下，相比现有算法表现出的卓越性能。

    It is well-known that decision-making problems from stochastic control can be formulated by means of forward-backward stochastic differential equation (FBSDE). Recently, the authors of Ji et al. 2022 proposed an efficient deep learning-based algorithm which was based on the stochastic maximum principle (SMP). In this paper, we provide a convergence result for this deep SMP-BSDE algorithm and compare its performance with other existing methods. In particular, by adopting a similar strategy as in Han and Long 2020, we derive a posteriori error estimate, and show that the total approximation error can be bounded by the value of the loss functional and the discretization error. We present numerical examples for high-dimensional stochastic control problems, both in case of drift- and diffusion control, which showcase superior performance compared to existing algorithms.
    
[^4]: 使用Liquid Staking Derivatives (LSDs)进行杠杆质押: 机会与风险

    Leverage Staking with Liquid Staking Derivatives (LSDs): Opportunities and Risks. (arXiv:2401.08610v1 [q-fin.GN])

    [http://arxiv.org/abs/2401.08610](http://arxiv.org/abs/2401.08610)

    这项研究系统地研究了Liquid Staking Derivatives (LSDs)的杠杆质押机会与风险。他们发现杠杆质押在Lido-Aave生态系统中能够实现较高的回报，并有潜力通过优化策略获得更多收益。

    

    Lido是以太坊上最主要的Liquid Staking Derivative (LSD)提供商，允许用户抵押任意数量的ETH来获得stETH，这可以与DeFi协议如Aave进行整合。Lido与Aave之间的互通性使得一种新型策略“杠杆质押”得以实现，用户在Lido上质押ETH获取stETH，将stETH作为Aave上的抵押品借入ETH，然后将借入的ETH重新投入Lido。用户可以迭代执行此过程，根据自己的风险偏好来优化潜在回报。本文系统地研究了杠杆质押所涉及的机会和风险。我们是第一个在Lido-Aave生态系统中对杠杆质押策略进行形式化的研究。我们的经验研究发现，在以太坊上有262个杠杆质押头寸，总质押金额为295,243 ETH（482M USD）。我们发现，90.13%的杠杆质押头寸实现了比传统质押更高的回报。

    Lido, the leading Liquid Staking Derivative (LSD) provider on Ethereum, allows users to stake an arbitrary amount of ETH to receive stETH, which can be integrated with Decentralized Finance (DeFi) protocols such as Aave. The composability between Lido and Aave enables a novel strategy called "leverage staking", where users stake ETH on Lido to acquire stETH, utilize stETH as collateral on Aave to borrow ETH, and then restake the borrowed ETH on Lido. Users can iteratively execute this process to optimize potential returns based on their risk profile.  This paper systematically studies the opportunities and risks associated with leverage staking. We are the first to formalize the leverage staking strategy within the Lido-Aave ecosystem. Our empirical study identifies 262 leverage staking positions on Ethereum, with an aggregated staking amount of 295,243 ETH (482M USD). We discover that 90.13% of leverage staking positions have achieved higher returns than conventional staking. Furtherm
    
[^5]: 大规模偏t乌鸦勾结的高效变分推理及其在股票收益率中的应用

    Efficient Variational Inference for Large Skew-t Copulas with Application to Intraday Equity Returns. (arXiv:2308.05564v1 [econ.EM])

    [http://arxiv.org/abs/2308.05564](http://arxiv.org/abs/2308.05564)

    本研究提出一种快速而准确的贝叶斯变分推理方法，用于估计大规模偏t乌鸦因子勾结模型。该方法能够捕捉到金融数据中的不对称和极端尾部相关性，以及股票对之间的异质性非对称依赖。

    

    大规模偏t乌鸦因子勾结模型对金融数据建模具有吸引力，因为它们允许不对称和极端的尾部相关性。我们展示了Azzalini和Capitanio（2003）所隐含的乌鸦勾结在成对非对称依赖性方面比两种流行的乌鸦勾结更高。在高维情况下，对该乌鸦勾结的估计具有挑战性，我们提出了一种快速而准确的贝叶斯变分推理方法来解决这个问题。该方法使用条件高斯生成表示法定义了一个可以准确近似的附加后验。使用快速随机梯度上升算法来解决变分优化。这种新的方法被用来估计2017年至2021年间93个美国股票的股票收益率的勾结模型。除了成对相关性的变化外，该勾结还捕捉到了股票对之间的非对称依赖的大量异质性。

    Large skew-t factor copula models are attractive for the modeling of financial data because they allow for asymmetric and extreme tail dependence. We show that the copula implicit in the skew-t distribution of Azzalini and Capitanio (2003) allows for a higher level of pairwise asymmetric dependence than two popular alternative skew-t copulas. Estimation of this copula in high dimensions is challenging, and we propose a fast and accurate Bayesian variational inference (VI) approach to do so. The method uses a conditionally Gaussian generative representation of the skew-t distribution to define an augmented posterior that can be approximated accurately. A fast stochastic gradient ascent algorithm is used to solve the variational optimization. The new methodology is used to estimate copula models for intraday returns from 2017 to 2021 on 93 U.S. equities. The copula captures substantial heterogeneity in asymmetric dependence over equity pairs, in addition to the variability in pairwise co
    

