# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Leverage Staking with Liquid Staking Derivatives (LSDs): Opportunities and Risks.](http://arxiv.org/abs/2401.08610) | 这项研究系统地研究了Liquid Staking Derivatives (LSDs)的杠杆质押机会与风险。他们发现杠杆质押在Lido-Aave生态系统中能够实现较高的回报，并有潜力通过优化策略获得更多收益。 |
| [^2] | [Navigating Uncertainty in ESG Investing.](http://arxiv.org/abs/2310.02163) | 该论文提出了一种方法来解决ESG投资中的不确定性问题。通过开发ESG集合策略和将ESG评分整合到强化学习模型中，研究提供了量身定制的投资策略，并引入了双效均值方差模型来分类投资者。此外，引入ESG调整的CAPM模型评估优化投资组合的表现。最终，该方法为投资者提供了导航ESG评级固有模糊性的工具，帮助做出更明智的投资决策。 |
| [^3] | [Mean-field equilibrium price formation with exponential utility.](http://arxiv.org/abs/2304.07108) | 本文研究了多个投资者在初始财富、风险规避参数以及终止时间的随机负债方面存在差异时的均衡价格形成问题，通过一个新的均场反向随机微分方程（BSDE）的解来表征风险股票的均衡风险溢价过程，并证明其清除市场。 |

# 详细

[^1]: 使用Liquid Staking Derivatives (LSDs)进行杠杆质押: 机会与风险

    Leverage Staking with Liquid Staking Derivatives (LSDs): Opportunities and Risks. (arXiv:2401.08610v1 [q-fin.GN])

    [http://arxiv.org/abs/2401.08610](http://arxiv.org/abs/2401.08610)

    这项研究系统地研究了Liquid Staking Derivatives (LSDs)的杠杆质押机会与风险。他们发现杠杆质押在Lido-Aave生态系统中能够实现较高的回报，并有潜力通过优化策略获得更多收益。

    

    Lido是以太坊上最主要的Liquid Staking Derivative (LSD)提供商，允许用户抵押任意数量的ETH来获得stETH，这可以与DeFi协议如Aave进行整合。Lido与Aave之间的互通性使得一种新型策略“杠杆质押”得以实现，用户在Lido上质押ETH获取stETH，将stETH作为Aave上的抵押品借入ETH，然后将借入的ETH重新投入Lido。用户可以迭代执行此过程，根据自己的风险偏好来优化潜在回报。本文系统地研究了杠杆质押所涉及的机会和风险。我们是第一个在Lido-Aave生态系统中对杠杆质押策略进行形式化的研究。我们的经验研究发现，在以太坊上有262个杠杆质押头寸，总质押金额为295,243 ETH（482M USD）。我们发现，90.13%的杠杆质押头寸实现了比传统质押更高的回报。

    Lido, the leading Liquid Staking Derivative (LSD) provider on Ethereum, allows users to stake an arbitrary amount of ETH to receive stETH, which can be integrated with Decentralized Finance (DeFi) protocols such as Aave. The composability between Lido and Aave enables a novel strategy called "leverage staking", where users stake ETH on Lido to acquire stETH, utilize stETH as collateral on Aave to borrow ETH, and then restake the borrowed ETH on Lido. Users can iteratively execute this process to optimize potential returns based on their risk profile.  This paper systematically studies the opportunities and risks associated with leverage staking. We are the first to formalize the leverage staking strategy within the Lido-Aave ecosystem. Our empirical study identifies 262 leverage staking positions on Ethereum, with an aggregated staking amount of 295,243 ETH (482M USD). We discover that 90.13% of leverage staking positions have achieved higher returns than conventional staking. Furtherm
    
[^2]: 在ESG投资中导航不确定性

    Navigating Uncertainty in ESG Investing. (arXiv:2310.02163v1 [q-fin.PM])

    [http://arxiv.org/abs/2310.02163](http://arxiv.org/abs/2310.02163)

    该论文提出了一种方法来解决ESG投资中的不确定性问题。通过开发ESG集合策略和将ESG评分整合到强化学习模型中，研究提供了量身定制的投资策略，并引入了双效均值方差模型来分类投资者。此外，引入ESG调整的CAPM模型评估优化投资组合的表现。最终，该方法为投资者提供了导航ESG评级固有模糊性的工具，帮助做出更明智的投资决策。

    

    投资者对评级机构对环境、社会和公司治理(ESG)所分配的排名普遍存在困惑，凸显出可持续投资中的一个关键问题。为了解决这种不确定性，我们的研究提出了一种方法，不仅可识别出这种模糊性，而且为不同投资者提供量身定制的投资策略。通过开发ESG集合策略，并将ESG评分整合到强化学习模型中，我们旨在优化既能获得金融回报又能关注ESG目标的投资组合。此外，通过提出双效均值方差模型，我们基于风险偏好对投资者进行分类。我们还引入了ESG调整的资本资产定价模型(CAPM)来评估这些优化投资组合的表现。最终，我们综合的方法为投资者提供了导航ESG评级固有模糊性的工具，促进更明智的投资决策。

    The widespread confusion among investors regarding Environmental, Social, and Governance (ESG) rankings assigned by rating agencies has underscored a critical issue in sustainable investing. To address this uncertainty, our research has devised methods that not only recognize this ambiguity but also offer tailored investment strategies for different investor profiles. By developing ESG ensemble strategies and integrating ESG scores into a Reinforcement Learning (RL) model, we aim to optimize portfolios that cater to both financial returns and ESG-focused outcomes. Additionally, by proposing the Double-Mean-Variance model, we classify three types of investors based on their risk preferences. We also introduce ESG-adjusted Capital Asset Pricing Models (CAPMs) to assess the performance of these optimized portfolios. Ultimately, our comprehensive approach provides investors with tools to navigate the inherent ambiguities of ESG ratings, facilitating more informed investment decisions.
    
[^3]: 带指数效用函数的均值场均衡价格形成

    Mean-field equilibrium price formation with exponential utility. (arXiv:2304.07108v1 [q-fin.MF])

    [http://arxiv.org/abs/2304.07108](http://arxiv.org/abs/2304.07108)

    本文研究了多个投资者在初始财富、风险规避参数以及终止时间的随机负债方面存在差异时的均衡价格形成问题，通过一个新的均场反向随机微分方程（BSDE）的解来表征风险股票的均衡风险溢价过程，并证明其清除市场。

    

    本文研究了多位投资者在初始财富、风险规避参数以及终止时间的随机负债方面存在差异时的均衡价格形成问题。我们通过一个新的均场反向随机微分方程（BSDE）的解来表征风险股票的均衡风险溢价过程，其特征是驱动程序在随机积分和条件期望上都具有二次增长。我们证明了在多个条件下均场BSDE存在解，并且表明随着人口规模的增大，结果风险溢价进程实际上会清除市场。

    In this paper, we study a problem of equilibrium price formation among many investors with exponential utility. The investors are heterogeneous in their initial wealth, risk-averseness parameter, as well as stochastic liability at the terminal time. We characterize the equilibrium risk-premium process of the risky stocks in terms of the solution to a novel mean-field backward stochastic differential equation (BSDE), whose driver has quadratic growth both in the stochastic integrands and in their conditional expectations. We prove the existence of a solution to the mean-field BSDE under several conditions and show that the resultant risk-premium process actually clears the market in the large population limit.
    

