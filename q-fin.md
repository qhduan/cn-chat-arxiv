# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Quasi-Monte Carlo for Efficient Fourier Pricing of Multi-Asset Options](https://arxiv.org/abs/2403.02832) | 本研究提出使用随机化拟蒙特卡洛积分来提高傅里叶方法在高维情境下的可扩展性，以解决高效定价多资产期权的挑战。 |
| [^2] | [Large Banks and Systemic Risk: Insights from a Mean-Field Game Model.](http://arxiv.org/abs/2305.17830) | 本文使用均场博弈模型研究大银行对金融系统稳定性的影响，结果发现大银行在规模不太大的情况下可以对稳定性产生积极的贡献，但在违约事件中也有可能产生负面溢出效应，导致系统性风险增加。随着银行对跨行业市场的依赖程度越来越高，整个系统变得更加稳定，但罕见系统性崩溃的概率也会增加。而大银行的存在进一步放大了这种风险。 |
| [^3] | [Wildfire Modeling: Designing a Market to Restore Assets.](http://arxiv.org/abs/2205.13773) | 该论文研究了如何设计一个市场来恢复由森林火灾造成的资产损失。研究通过分析电力公司引发火灾的原因，并将火灾风险纳入经济模型中，提出了收取森林火灾基金和公平收费的方案，以最大化社会影响和总盈余。 |

# 详细

[^1]: 高效傅里叶定价多资产期权的拟蒙特卡洛方法

    Quasi-Monte Carlo for Efficient Fourier Pricing of Multi-Asset Options

    [https://arxiv.org/abs/2403.02832](https://arxiv.org/abs/2403.02832)

    本研究提出使用随机化拟蒙特卡洛积分来提高傅里叶方法在高维情境下的可扩展性，以解决高效定价多资产期权的挑战。

    

    在定量金融中，高效定价多资产期权是一个重要挑战。蒙特卡洛（MC）方法仍然是定价引擎的主要选择；然而，其收敛速度慢阻碍了其实际应用。傅里叶方法利用特征函数的知识，准确快速地估值多达两个资产的期权。然而，在高维设置中，由于常用的积分技术具有张量积（TP）结构，它们面临障碍。本文主张使用随机化拟蒙特卡洛（RQMC）积分来改善高维傅里叶方法的可扩展性。RQMC技术受益于被积函数的光滑性，缓解了维度灾难，同时提供了实用的误差估计。然而，RQMC在无界域$\mathbb{R}^d$上的适用性需要将域转换为$[0,1]^d$，这可能...

    arXiv:2403.02832v1 Announce Type: new  Abstract: Efficiently pricing multi-asset options poses a significant challenge in quantitative finance. The Monte Carlo (MC) method remains the prevalent choice for pricing engines; however, its slow convergence rate impedes its practical application. Fourier methods leverage the knowledge of the characteristic function to accurately and rapidly value options with up to two assets. Nevertheless, they face hurdles in the high-dimensional settings due to the tensor product (TP) structure of commonly employed quadrature techniques. This work advocates using the randomized quasi-MC (RQMC) quadrature to improve the scalability of Fourier methods with high dimensions. The RQMC technique benefits from the smoothness of the integrand and alleviates the curse of dimensionality while providing practical error estimates. Nonetheless, the applicability of RQMC on the unbounded domain, $\mathbb{R}^d$, requires a domain transformation to $[0,1]^d$, which may r
    
[^2]: 大银行与系统风险：基于均场博弈模型的研究

    Large Banks and Systemic Risk: Insights from a Mean-Field Game Model. (arXiv:2305.17830v1 [q-fin.MF])

    [http://arxiv.org/abs/2305.17830](http://arxiv.org/abs/2305.17830)

    本文使用均场博弈模型研究大银行对金融系统稳定性的影响，结果发现大银行在规模不太大的情况下可以对稳定性产生积极的贡献，但在违约事件中也有可能产生负面溢出效应，导致系统性风险增加。随着银行对跨行业市场的依赖程度越来越高，整个系统变得更加稳定，但罕见系统性崩溃的概率也会增加。而大银行的存在进一步放大了这种风险。

    

    本文旨在研究大银行对金融系统稳定性的影响。为此，我们采用一个线性二次高斯（LQG）均场博弈（MFG）模型来研究跨行业市场，其中包括一个大银行和多个小银行。我们采用MFG方法推导每个银行的最优交易策略，并得出市场的均衡状态。随后，我们进行Monte Carlo模拟，探讨大银行在各种情况下对系统性风险扮演的角色。我们的研究发现，虽然大银行如果规模不太大可以对稳定性产生积极的贡献，但在违约事件中也有可能产生负面溢出效应，导致系统性风险增加。我们还发现，随着银行对跨行业市场的依赖程度越来越高，整个系统变得更加稳定，但罕见系统性崩溃的概率也会增加。而大银行的存在进一步放大了这种风险。

    This paper aims to investigate the impact of large banks on the financial system stability. To achieve this, we employ a linear-quadratic-Gaussian (LQG) mean-field game (MFG) model of an interbank market, which involves one large bank and multiple small banks. Our approach involves utilizing the MFG methodology to derive the optimal trading strategies for each bank, resulting in an equilibrium for the market. Subsequently, we conduct Monte Carlo simulations to explore the role played by the large bank in systemic risk under various scenarios. Our findings indicate that while the major bank, if its size is not too large, can contribute positively to stability, it also has the potential to generate negative spillover effects in the event of default, leading to increased systemic risk. We also discover that as banks become more reliant on the interbank market, the overall system becomes more stable but the probability of a rare systemic failure increases. This risk is further amplified by
    
[^3]: 森林火灾模型：设计市场以恢复资产

    Wildfire Modeling: Designing a Market to Restore Assets. (arXiv:2205.13773v3 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2205.13773](http://arxiv.org/abs/2205.13773)

    该论文研究了如何设计一个市场来恢复由森林火灾造成的资产损失。研究通过分析电力公司引发火灾的原因，并将火灾风险纳入经济模型中，提出了收取森林火灾基金和公平收费的方案，以最大化社会影响和总盈余。

    

    在过去的十年里，夏季森林火灾已经成为加利福尼亚和美国的常态。这些火灾的原因多种多样。州政府会收取森林火灾基金来帮助受灾人员。然而，基金只在特定条件下发放，并且在整个加利福尼亚州均匀收取。因此，该项目的整体思路是寻找关于电力公司如何引发火灾以及如何帮助收取森林火灾基金或者公平收费以最大限度地实现社会影响的数量结果。该研究项目旨在提出与植被、输电线路相关的森林火灾风险，并将其与金钱挂钩。因此，该项目有助于解决与每个地点相关的森林火灾基金收取问题，并结合能源价格根据地点的森林火灾风险向客户收费，以实现社会的总盈余最大化。

    In the past decade, summer wildfires have become the norm in California, and the United States of America. These wildfires are caused due to variety of reasons. The state collects wildfire funds to help the impacted customers. However, the funds are eligible only under certain conditions and are collected uniformly throughout California. Therefore, the overall idea of this project is to look for quantitative results on how electrical corporations cause wildfires and how they can help to collect the wildfire funds or charge fairly to the customers to maximize the social impact. The research project aims to propose the implication of wildfire risk associated with vegetation, and due to power lines and incorporate that in dollars. Therefore, the project helps to solve the problem of collecting wildfire funds associated with each location and incorporate energy prices to charge their customers according to their wildfire risk related to the location to maximize the social surplus for the s
    

