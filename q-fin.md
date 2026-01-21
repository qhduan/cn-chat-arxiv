# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Composite likelihood estimation of stationary Gaussian processes with a view toward stochastic volatility](https://arxiv.org/abs/2403.12653) | 开发了复合似然推断框架，用于稳态高斯过程的参数估计，成功改善了矩法估计方法，实证结果支持随机波动性在不同时间尺度上的特性 |
| [^2] | [Database for the meta-analysis of the social cost of carbon (v2024.0)](https://arxiv.org/abs/2402.09125) | 该论文介绍了社会碳成本估计元分析数据库的新版本，新增了关于气候变化影响和福利函数形状的字段，并扩展了合作者和引用网络。 |
| [^3] | [Market-Based Asset Price Probability](https://arxiv.org/abs/2205.07256) | 这篇论文探讨了市场交易价值和交易量的随机性如何影响资产价格的随机性，并通过市场基于价格的统计矩来近似价格概率。研究发现使用交易量加权平均价格可以消除价格和交易量之间的相关性，并推导出了其他价格和交易量相关性。研究结果对资产定价模型和风险价值有重要影响。 |
| [^4] | [Effective and scalable programs to facilitate labor market transitions for women in technology.](http://arxiv.org/abs/2211.09968) | 这个论文描述了一个低成本和可扩展的计划，旨在帮助波兰的女性转向科技行业的工作。研究表明，参与指导计划或挑战计划可以显著增加在科技行业找到工作的概率。 |

# 详细

[^1]: 复合似然估计在稳态高斯过程中的应用并针对随机波动性进行研究

    Composite likelihood estimation of stationary Gaussian processes with a view toward stochastic volatility

    [https://arxiv.org/abs/2403.12653](https://arxiv.org/abs/2403.12653)

    开发了复合似然推断框架，用于稳态高斯过程的参数估计，成功改善了矩法估计方法，实证结果支持随机波动性在不同时间尺度上的特性

    

    我们开发了一个框架，用于对参数化连续时间稳态高斯过程进行复合似然推断。我们推导了相关最大复合似然估计器的渐近理论。我们将我们的方法应用于一对旨在描述金融资产回报的随机对数现货波动性的模型。模拟研究表明，在这些情景下，它表现良好，并改善了矩法估计方法。在应用中，我们研究了利用加密货币市场高频数据计算的一种日内度量的动态特性。经验证据支持一种机制，在这种机制中，随机波动性的短期和长期相关结构是分开的，以捕捉其在不同时间尺度上的特性。

    arXiv:2403.12653v1 Announce Type: new  Abstract: We develop a framework for composite likelihood inference of parametric continuous-time stationary Gaussian processes. We derive the asymptotic theory of the associated maximum composite likelihood estimator. We implement our approach on a pair of models that has been proposed to describe the random log-spot variance of financial asset returns. A simulation study shows that it delivers good performance in these settings and improves upon a method-of-moments estimation. In an application, we inspect the dynamic of an intraday measure of spot variance computed with high-frequency data from the cryptocurrency market. The empirical evidence supports a mechanism, where the short- and long-term correlation structure of stochastic volatility are decoupled in order to capture its properties at different time scales.
    
[^2]: 社会碳成本的元分析数据库 (v2024.0)

    Database for the meta-analysis of the social cost of carbon (v2024.0)

    [https://arxiv.org/abs/2402.09125](https://arxiv.org/abs/2402.09125)

    该论文介绍了社会碳成本估计元分析数据库的新版本，新增了关于气候变化影响和福利函数形状的字段，并扩展了合作者和引用网络。

    

    本文介绍了社会碳成本估计元分析数据库的新版本。新增了记录，并添加了关于气候变化影响和福利函数形状的新字段。该数据库还扩展了合作者和引用网络。

    arXiv:2402.09125v1 Announce Type: new Abstract: A new version of the database for the meta-analysis of estimates of the social cost of carbon is presented. New records were added, and new fields on the impact of climate change and the shape of the welfare function. The database was extended to co-author and citation networks.
    
[^3]: 基于市场的资产价格概率

    Market-Based Asset Price Probability

    [https://arxiv.org/abs/2205.07256](https://arxiv.org/abs/2205.07256)

    这篇论文探讨了市场交易价值和交易量的随机性如何影响资产价格的随机性，并通过市场基于价格的统计矩来近似价格概率。研究发现使用交易量加权平均价格可以消除价格和交易量之间的相关性，并推导出了其他价格和交易量相关性。研究结果对资产定价模型和风险价值有重要影响。

    

    我们将市场交易价值和交易量的随机性视为资产价格随机性的起源。我们定义了依赖于市场交易价值和交易量的统计矩的前四个市场基于价格的统计矩。如果在时间平均间隔内所有交易量都保持恒定，那么市场基于价格的统计矩与传统基于频率的统计矩相一致。我们通过有限数量的价格统计矩来近似市场基于价格的概率。我们考虑基于市场价格统计矩在资产定价模型和风险价值方面的影响。我们证明了使用交易量加权平均价格会导致价格和交易量的相关性为零。我们推导了基于市场的价格和交易量平方之间的相关性以及价格平方和交易量之间的相关性。要预测期限为T的基于市场的价格波动性，需要预测市场交易价值和交易量的前两个统计矩。

    We consider the randomness of market trade values and volumes as the origin of asset price stochasticity. We define the first four market-based price statistical moments that depend on statistical moments and correlations of market trade values and volumes. Market-based price statistical moments coincide with conventional frequency-based ones if all trade volumes are constant during the time averaging interval. We present approximations of market-based price probability by a finite number of price statistical moments. We consider the consequences of the use of market-based price statistical moments for asset-pricing models and Value-at-Risk. We show that the use of volume weighted average price results in zero price-volume correlations. We derive market-based correlations between price and squares of volume and between squares of price and volume. To forecast market-based price volatility at horizon T one should predict the first two statistical moments of market trade values and volum
    
[^4]: 为促进科技行业女性就业转型的有效可扩展方案

    Effective and scalable programs to facilitate labor market transitions for women in technology. (arXiv:2211.09968v3 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2211.09968](http://arxiv.org/abs/2211.09968)

    这个论文描述了一个低成本和可扩展的计划，旨在帮助波兰的女性转向科技行业的工作。研究表明，参与指导计划或挑战计划可以显著增加在科技行业找到工作的概率。

    

    我们描述了一个低成本（每人约15美元）和可扩展的计划，称为“挑战”，旨在帮助波兰的女性转向科技行业的工作。该计划帮助参与者开发展示与工作相关的能力的作品集。我们进行了两个独立评估，一个是对挑战计划的评估，另一个是对传统的指导计划“指导”进行的评估，其中经验丰富的技术专业人员与被指导者个别合作，支持他们的就业搜索。利用两个计划都超额订阅的事实，我们随机录取了参与者，并测量了它们对找到科技行业工作的概率的影响。我们估计，指导计划将四个月内找到科技工作的概率从29%增加到42%，挑战计划将该概率从20%增加到29%，并且治疗效果在12个月内不衰减。由于这两个计划在实践中有容量限制（只有申请者的28%可以获得录取），

    We describe the design, implementation, and evaluation of a low-cost (approximately $15 per person) and scalable program, called Challenges, aimed at aiding women in Poland transition to technology-sector jobs. This program helps participants develop portfolios demonstrating job-relevant competencies. We conduct two independent evaluations, one of the Challenges program and the other of a traditional mentoring program -- Mentoring -- where experienced tech professionals work individually with mentees to support them in their job search. Exploiting the fact that both programs were oversubscribed, we randomized admissions and measured their impact on the probability of finding a job in the technology sector. We estimate that Mentoring increases the probability of finding a technology job within four months from 29% to 42% and Challenges from 20% to 29%, and the treatment effects do not attenuate over 12 months. Since both programs are capacity constrained in practice (only 28% of applica
    

