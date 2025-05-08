# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Calibration of Local Volatility Models with Stochastic Interest Rates using Optimal Transport.](http://arxiv.org/abs/2305.00200) | 本文提出了一种使用最优传输法实现校准的随机利率下的局部波动率模型的非参数方法，适用于校准股票价格的局部波动率模型。 |
| [^2] | [Auto.gov: Learning-based On-chain Governance for Decentralized Finance (DeFi).](http://arxiv.org/abs/2302.09551) | 这项研究提出了一个“Auto.gov”框架，可增强去中心化金融（DeFi）的安全性和降低受攻击的风险。该框架利用深度Q-网络（DQN）强化学习方法，提出了半自动的、直观的治理提案，并量化了其理由，使系统能够有效地应对恶意行为和意外的市场情况。 |

# 详细

[^1]: 使用最优传输法校准随机利率下的局部波动率模型。（arXiv：2305.00200v1 [q-fin.MF]）

    Calibration of Local Volatility Models with Stochastic Interest Rates using Optimal Transport. (arXiv:2305.00200v1 [q-fin.MF])

    [http://arxiv.org/abs/2305.00200](http://arxiv.org/abs/2305.00200)

    本文提出了一种使用最优传输法实现校准的随机利率下的局部波动率模型的非参数方法，适用于校准股票价格的局部波动率模型。

    

    我们发展了一种非参数、最优传输驱动的随机利率下局部波动率模型的校准方法。该方法找到一个完全校准的模型，该模型与给定的参考模型最接近。我们建立了一个通用的对偶性结果，可以通过优化非线性HJB方程的解来解决问题。然后，我们将该方法应用于顺序校准设置：我们假设给定了一个利率模型，并对观察到的市场期限结构进行了校准。然后，我们寻求校准股票价格局部波动率模型，波动率系数取决于时间、基础资产和短期利率过程，并由可以与利率过程的随机性相关的布朗运动驱动。局部波动率模型通过从半鞅最优传输的PDE表述中导出的凸优化问题对有限数量的欧式期权价格进行校准。我们的方法是一个

    We develop a non-parametric, optimal transport driven, calibration methodology for local volatility models with stochastic interest rate. The method finds a fully calibrated model which is the closest to a given reference model. We establish a general duality result which allows to solve the problem via optimising over solutions to a non-linear HJB equation. We then apply the method to a sequential calibration setup: we assume that an interest rate model is given and is calibrated to the observed term structure in the market. We then seek to calibrate a stock price local volatility model with volatility coefficient depending on time, the underlying and the short rate process, and driven by a Brownian motion which can be correlated with the randomness driving the rates process. The local volatility model is calibrated to a finite number of European options prices via a convex optimisation problem derived from the PDE formulation of semimartingale optimal transport. Our methodology is an
    
[^2]: Auto.gov：面向DeFi的基于学习的链上治理

    Auto.gov: Learning-based On-chain Governance for Decentralized Finance (DeFi). (arXiv:2302.09551v2 [q-fin.RM] UPDATED)

    [http://arxiv.org/abs/2302.09551](http://arxiv.org/abs/2302.09551)

    这项研究提出了一个“Auto.gov”框架，可增强去中心化金融（DeFi）的安全性和降低受攻击的风险。该框架利用深度Q-网络（DQN）强化学习方法，提出了半自动的、直观的治理提案，并量化了其理由，使系统能够有效地应对恶意行为和意外的市场情况。

    

    近年来，去中心化金融（DeFi）经历了显著增长，涌现出了各种协议，例如借贷协议和自动化做市商（AMM）。传统上，这些协议采用链下治理，其中代币持有者投票修改参数。然而，由协议核心团队进行的手动参数调整容易遭受勾结攻击，危及系统的完整性和安全性。此外，纯粹的确定性算法方法可能会使协议受到新的利用和攻击的威胁。本文提出了“Auto.gov”，这是一个面向DeFi的基于学习的链上治理框架，可增强安全性并降低受攻击的风险。我们的模型利用了深度Q-网络（DQN）强化学习方法，提出了半自动化的、直观的治理提案与量化的理由。这种方法使系统能够有效地适应和缓解恶意行为和意外的市场情况的负面影响。

    In recent years, decentralized finance (DeFi) has experienced remarkable growth, with various protocols such as lending protocols and automated market makers (AMMs) emerging. Traditionally, these protocols employ off-chain governance, where token holders vote to modify parameters. However, manual parameter adjustment, often conducted by the protocol's core team, is vulnerable to collusion, compromising the integrity and security of the system. Furthermore, purely deterministic, algorithm-based approaches may expose the protocol to novel exploits and attacks.  In this paper, we present "Auto.gov", a learning-based on-chain governance framework for DeFi that enhances security and reduces susceptibility to attacks. Our model leverages a deep Q- network (DQN) reinforcement learning approach to propose semi-automated, intuitive governance proposals with quantitative justifications. This methodology enables the system to efficiently adapt to and mitigate the negative impact of malicious beha
    

