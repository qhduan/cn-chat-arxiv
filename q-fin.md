# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Gaussian Process Based Method with Deep Kernel Learning for Pricing High-dimensional American Options](https://arxiv.org/abs/2311.07211) | 本论文利用深度核学习和变分推断方法改进了高斯过程回归在高维美式期权定价中的应用，实验证明该方法相对于传统方法在高维情况下有更好的性能表现，特别是在跳跃扩散模型中。 |
| [^2] | [Assessing the Solvency of Virtual Asset Service Providers: Are Current Standards Sufficient?.](http://arxiv.org/abs/2309.16408) | 本文提出了一种通过交叉参考不同数据源来评估虚拟资产服务提供商（VASP）偿付能力的方法，并调查了24个注册在奥地利金融市场管理局的VASP的数据。研究发现，VASP的年收支交易额约为20亿欧元，涉及约180万用户。 |
| [^3] | [The Economic Value of User Tracking for Publishers.](http://arxiv.org/abs/2303.10906) | 发布者在追踪用户时为广告带来的收益将大幅下降，对大多数发布者来说，用户追踪限制会使价格降低，提供广泛内容的发布者受影响最大。 |

# 详细

[^1]: 基于深度核学习的高维美式期权定价的高斯过程方法

    A Gaussian Process Based Method with Deep Kernel Learning for Pricing High-dimensional American Options

    [https://arxiv.org/abs/2311.07211](https://arxiv.org/abs/2311.07211)

    本论文利用深度核学习和变分推断方法改进了高斯过程回归在高维美式期权定价中的应用，实验证明该方法相对于传统方法在高维情况下有更好的性能表现，特别是在跳跃扩散模型中。

    

    高斯过程回归（GPR）被认为是在基于回归的蒙特卡洛方法中用于估计美式期权继续价值的潜在方法。然而，它存在一些缺点，比如在高维情况下不可靠，并且当模拟路径的数量很大时计算成本高。在本研究中，我们应用深度核学习和变分推断到GPR中，以克服这些缺点，并在几何布朗运动和Merton跳跃扩散模型下测试其性能。实验结果表明，所提出的方法在高维情况下优于最小二乘蒙特卡洛方法，特别是在跳跃扩散模型中。

    Gaussian process regression (GPR) is considered as a potential method for estimating the continuation value of an American option in the regression-based Monte Carlo method. However, it has some drawbacks, such as the unreliability in high-dimensional cases and the high computational cost when the number of simulated paths is large. In this work, we apply the deep kernel learning and variational inference to GPR in order to overcome these drawbacks, and test its performance under geometric Brownian motion and Merton's jump diffusion models. The experiments show that the proposed method outperforms the Least Square Monte Carlo method in high-dimensional cases, especially with jump diffusion models.
    
[^2]: 评估虚拟资产服务提供商的偿付能力：当前的标准足够吗？

    Assessing the Solvency of Virtual Asset Service Providers: Are Current Standards Sufficient?. (arXiv:2309.16408v1 [q-fin.GN])

    [http://arxiv.org/abs/2309.16408](http://arxiv.org/abs/2309.16408)

    本文提出了一种通过交叉参考不同数据源来评估虚拟资产服务提供商（VASP）偿付能力的方法，并调查了24个注册在奥地利金融市场管理局的VASP的数据。研究发现，VASP的年收支交易额约为20亿欧元，涉及约180万用户。

    

    像集中式加密货币交易所这样的实体属于虚拟资产服务提供商（VASP）的业务范畴。与其他企业一样，它们也可能破产。VASP促使在分布式账本技术（DLT）中以钱包形式组织的加密资产的交换、保管和转移。尽管DLT交易公开可见，但VASP的加密资产持有情况尚未受到系统化的审计程序的监管。本文提出了一种评估VASP偿付能力的方法，即通过交叉参考来自三个不同来源的数据：加密资产钱包、商业注册的资产负债表和监管机构的数据。我们调查了奥地利金融市场管理局注册的24个VASP，并提供了关于客户身份和来源的监管数据见解。他们每年的收入和支出交易量约为20亿欧元，涉及约180万用户。我们描述了金融服务提供者如何使用这些资产以及他们面临的挑战。

    Entities like centralized cryptocurrency exchanges fall under the business category of virtual asset service providers (VASPs). As any other enterprise, they can become insolvent. VASPs enable the exchange, custody, and transfer of cryptoassets organized in wallets across distributed ledger technologies (DLTs). Despite the public availability of DLT transactions, the cryptoasset holdings of VASPs are not yet subject to systematic auditing procedures. In this paper, we propose an approach to assess the solvency of a VASP by cross-referencing data from three distinct sources: cryptoasset wallets, balance sheets from the commercial register, and data from supervisory entities. We investigate 24 VASPs registered with the Financial Market Authority in Austria and provide regulatory data insights such as who are the customers and where do they come from. Their yearly incoming and outgoing transaction volume amount to 2 billion EUR for around 1.8 million users. We describe what financial serv
    
[^3]: 发布者用户追踪的经济价值

    The Economic Value of User Tracking for Publishers. (arXiv:2303.10906v1 [econ.GN])

    [http://arxiv.org/abs/2303.10906](http://arxiv.org/abs/2303.10906)

    发布者在追踪用户时为广告带来的收益将大幅下降，对大多数发布者来说，用户追踪限制会使价格降低，提供广泛内容的发布者受影响最大。

    

    为了保护用户的在线隐私，监管机构和浏览器越来越限制用户追踪。这种限制对卖广告空间以资助业务、包括内容的发布者有经济影响。根据对111家发行商涉及4200万次广告展示的分析，当用户追踪不可用时，发布者为广告展示所收到的原始价格下降了约60％。在控制用户、广告商和发布者的差异后，这种下降仍然很大，为18％。超过90％的发布者感到当他们无法进行用户追踪时，价格变低。提供广泛内容的发布者（如新闻网站）比提供主题内容的发布者更受用户追踪限制的影响。收集用户浏览历史记录被认为普遍具有侵入性，对发布者的价值微不足道。这些结果证实了保护用户在线隐私将对发布者的经济利益产生重大影响的预测。

    Regulators and browsers increasingly restrict user tracking to protect users privacy online. Such restrictions also have economic implications for publishers that rely on selling advertising space to finance their business, including their content. According to an analysis of 42 million ad impressions related to 111 publishers, when user tracking is unavailable, the raw price paid to publishers for ad impressions decreases by about -60%. After controlling for differences in users, advertisers, and publishers, this decrease remains substantial, at -18%. More than 90% of the publishers realize lower prices when prevented from engaging in user tracking. Publishers offering broad content, such as news websites, suffer more from user tracking restrictions than publishers with thematically focused content. Collecting a users browsing history, perceived as generally intrusive to most users, generates negligible value for publishers. These results affirm the prediction that ensuring user priva
    

