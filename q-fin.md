# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Quasi-Monte Carlo for Efficient Fourier Pricing of Multi-Asset Options](https://arxiv.org/abs/2403.02832) | 本研究提出使用随机化拟蒙特卡洛积分来提高傅里叶方法在高维情境下的可扩展性，以解决高效定价多资产期权的挑战。 |

# 详细

[^1]: 高效傅里叶定价多资产期权的拟蒙特卡洛方法

    Quasi-Monte Carlo for Efficient Fourier Pricing of Multi-Asset Options

    [https://arxiv.org/abs/2403.02832](https://arxiv.org/abs/2403.02832)

    本研究提出使用随机化拟蒙特卡洛积分来提高傅里叶方法在高维情境下的可扩展性，以解决高效定价多资产期权的挑战。

    

    在定量金融中，高效定价多资产期权是一个重要挑战。蒙特卡洛（MC）方法仍然是定价引擎的主要选择；然而，其收敛速度慢阻碍了其实际应用。傅里叶方法利用特征函数的知识，准确快速地估值多达两个资产的期权。然而，在高维设置中，由于常用的积分技术具有张量积（TP）结构，它们面临障碍。本文主张使用随机化拟蒙特卡洛（RQMC）积分来改善高维傅里叶方法的可扩展性。RQMC技术受益于被积函数的光滑性，缓解了维度灾难，同时提供了实用的误差估计。然而，RQMC在无界域$\mathbb{R}^d$上的适用性需要将域转换为$[0,1]^d$，这可能...

    arXiv:2403.02832v1 Announce Type: new  Abstract: Efficiently pricing multi-asset options poses a significant challenge in quantitative finance. The Monte Carlo (MC) method remains the prevalent choice for pricing engines; however, its slow convergence rate impedes its practical application. Fourier methods leverage the knowledge of the characteristic function to accurately and rapidly value options with up to two assets. Nevertheless, they face hurdles in the high-dimensional settings due to the tensor product (TP) structure of commonly employed quadrature techniques. This work advocates using the randomized quasi-MC (RQMC) quadrature to improve the scalability of Fourier methods with high dimensions. The RQMC technique benefits from the smoothness of the integrand and alleviates the curse of dimensionality while providing practical error estimates. Nonetheless, the applicability of RQMC on the unbounded domain, $\mathbb{R}^d$, requires a domain transformation to $[0,1]^d$, which may r
    

