# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimizing Investment Strategies with Lazy Factor and Probability Weighting: A Price Portfolio Forecasting and Mean-Variance Model with Transaction Costs Approach.](http://arxiv.org/abs/2306.07928) | 本论文提出了一种使用“懒惰因子”和概率加权来优化投资策略的模型。通过价格组合预测和均值方差模型方法，实现投资风险与回报之间的最优平衡。 |
| [^2] | [Modeling Large Spot Price Deviations in Electricity Markets.](http://arxiv.org/abs/2306.07731) | 本文研究了电力市场中的电价波动模型，并发现在非危机时期四因子模型比三因子模型更为适用。 |
| [^3] | [Making forecasting self-learning and adaptive -- Pilot forecasting rack.](http://arxiv.org/abs/2306.07305) | 本文提出了通过使用试点预测架构中的算法干预来提高非AI模型下针织品类别的预测准确性。在决策模型中动态地选择最佳算法可提高预测准确性，并通过AI / ML预测模型使用先进的特征工程来实现。 |
| [^4] | [Centrality in Production Networks and International Technology Diffusion.](http://arxiv.org/abs/2306.06680) | 本文研究了全球价值链（GVCs）的中心性对研发（R&D）国际溢出的影响，结果表明，具有高中心性的出口国的R&D溢出最大，中心国在收集和扩散知识方面发挥了重要作用。 |
| [^5] | [Conditional Generative Models for Learning Stochastic Processes.](http://arxiv.org/abs/2304.10382) | 提出了一种称为 C-qGAN 的框架，利用量子电路结构实现了有效的状态准备过程，可以利用该方法加速蒙特卡罗分析等算法，并将其应用于亚式期权衍生品定价的任务中。 |
| [^6] | [Collective dynamics, diversification and optimal portfolio construction for cryptocurrencies.](http://arxiv.org/abs/2304.08902) | 本研究旨在探究加密货币市场的集体动力学和投资组合多样化，以及尝试确认先前在股票市场中建立的结果是否适用于加密货币市场。 |
| [^7] | [Pricing cyber-insurance for systems via maturity models.](http://arxiv.org/abs/2302.04734) | 本篇论文提出了一种使用安全成熟度模型的方法，以评估组织的安全水平并确定网络保险的适当保费。 |

# 详细

[^1]: 带有概率加权的“懒惰因子”的优化投资策略：结合交易成本的价格组合预测和均值方差模型方法

    Optimizing Investment Strategies with Lazy Factor and Probability Weighting: A Price Portfolio Forecasting and Mean-Variance Model with Transaction Costs Approach. (arXiv:2306.07928v1 [q-fin.PM])

    [http://arxiv.org/abs/2306.07928](http://arxiv.org/abs/2306.07928)

    本论文提出了一种使用“懒惰因子”和概率加权来优化投资策略的模型。通过价格组合预测和均值方差模型方法，实现投资风险与回报之间的最优平衡。

    

    市场交易员常常通过频繁交易波动性资产来优化他们的总回报。本研究提出了一种新的投资策略模型，基于“懒惰因子”。我们的方法分为价格组合预测模型和结合交易成本的均值方差模型。通过概率加权作为懒惰因子的系数，实现了在风险和回报之间达到最优平衡。

    Market traders often engage in the frequent transaction of volatile assets to optimize their total return. In this study, we introduce a novel investment strategy model, anchored on the 'lazy factor.' Our approach bifurcates into a Price Portfolio Forecasting Model and a Mean-Variance Model with Transaction Costs, utilizing probability weights as the coefficients of laziness factors. The Price Portfolio Forecasting Model, leveraging the EXPMA Mean Method, plots the long-term price trend line and forecasts future price movements, incorporating the tangent slope and rate of change. For short-term investments, we apply the ARIMA Model to predict ensuing prices. The Mean-Variance Model with Transaction Costs employs the Monte Carlo Method to formulate the feasible region. To strike an optimal balance between risk and return, equal probability weights are incorporated as coefficients of the laziness factor. To assess the efficacy of this combined strategy, we executed extensive experiments 
    
[^2]: 电力市场中的大型电价波动建模

    Modeling Large Spot Price Deviations in Electricity Markets. (arXiv:2306.07731v1 [q-fin.MF])

    [http://arxiv.org/abs/2306.07731](http://arxiv.org/abs/2306.07731)

    本文研究了电力市场中的电价波动模型，并发现在非危机时期四因子模型比三因子模型更为适用。

    

    过去两年里，能源市场的不确定性增加，导致电力即期价格出现了大幅波动。本文研究了经典三因子模型与高斯基信号、一个正跳跃信号和一个负跳跃信号的拟合，以及在这个新的市场环境下添加第二个高斯基信号的影响。我们使用基于所谓的Gibbs采样的马尔可夫链蒙特卡罗算法来校准模型。将得到的四因子模型与特定时期的三因子模型进行比较，并使用后验预测检验进行评估。此外，我们还推导出了基于四因子即期价格模型的期货合约价格的闭式解。我们发现，四因子模型在非危机时期的表现优于三因子模型。在危机时期，第二个高斯基信号并没有导致更好的拟合效果。

    Increased insecurities on the energy markets have caused massive fluctuations of the electricity spot price within the past two years. In this work, we investigate the fit of a classical 3-factor model with a Gaussian base signal as well as one positive and one negative jump signal in this new market environment. We also study the influence of adding a second Gaussian base signal to the model. For the calibration of our model we use a Markov Chain Monte Carlo algorithm based on the so-called Gibbs sampling. The resulting 4-factor model is than compared to the 3-factor model in different time periods of particular interest and evaluated using posterior predictive checking. Additionally, we derive closed-form solutions for the price of futures contracts in our 4-factor spot price model. We find that the 4-factor model outperforms the 3-factor model in times of non-crises. In times of crises, the second Gaussian base signal does not lead to a better the fit of the model. To the best of ou
    
[^3]: 让预测变得自学习和自适应--试点预测架构。

    Making forecasting self-learning and adaptive -- Pilot forecasting rack. (arXiv:2306.07305v1 [cs.LG])

    [http://arxiv.org/abs/2306.07305](http://arxiv.org/abs/2306.07305)

    本文提出了通过使用试点预测架构中的算法干预来提高非AI模型下针织品类别的预测准确性。在决策模型中动态地选择最佳算法可提高预测准确性，并通过AI / ML预测模型使用先进的特征工程来实现。

    

    零售销售和价格预测通常基于时间序列预测。对于某些产品类别，预测需求的准确性较低，会对库存、运输和补货计划造成负面影响。本文介绍了我们基于积极探索的试点演练的发现，以探索帮助零售商提高此类产品类别的预测准确性的方法。我们评估了通过一个样本产品类别“针织品”提高预测准确性的算法干预机会。目前，针织品产品类别的预测准确度在非AI模型中的范围为60%。我们探索了如何使用架构方法提高预测准确性。为了生成预测结果，我们的决策模型根据给定状态和上下文动态地从算法架中选择最佳算法。使用先进的特征工程构建的AI / ML预测模型的结果显示，需求预测的准确性有所提高。

    Retail sales and price projections are typically based on time series forecasting. For some product categories, the accuracy of demand forecasts achieved is low, negatively impacting inventory, transport, and replenishment planning. This paper presents our findings based on a proactive pilot exercise to explore ways to help retailers to improve forecast accuracy for such product categories.  We evaluated opportunities for algorithmic interventions to improve forecast accuracy based on a sample product category, Knitwear. The Knitwear product category has a current demand forecast accuracy from non-AI models in the range of 60%. We explored how to improve the forecast accuracy using a rack approach. To generate forecasts, our decision model dynamically selects the best algorithm from an algorithm rack based on performance for a given state and context. Outcomes from our AI/ML forecasting model built using advanced feature engineering show an increase in the accuracy of demand forecast f
    
[^4]: 《生产网络中的中心性与国际技术扩散》

    Centrality in Production Networks and International Technology Diffusion. (arXiv:2306.06680v1 [econ.GN])

    [http://arxiv.org/abs/2306.06680](http://arxiv.org/abs/2306.06680)

    本文研究了全球价值链（GVCs）的中心性对研发（R&D）国际溢出的影响，结果表明，具有高中心性的出口国的R&D溢出最大，中心国在收集和扩散知识方面发挥了重要作用。

    

    本研究探讨全球价值链（GVCs）的结构是否影响研发（R&D）的国际溢出。通过对1995-2007年21个国家和14个制造业的样本进行分类，研究了“中心性”度量的角色。结果表明，具有高中心性的出口国的R&D溢出最大，这表明中心国在收集和扩散知识方面发挥了重要作用。同时发现，具有中等中心性的国家在知识扩散方面越来越重要，最后，只有在G5国家中才能观察到自身的正向溢出效应。

    This study examines whether the structure of global value chains (GVCs) affects international spillovers of research and development (R&D). Although the presence of ``hub'' countries in GVCs has been confirmed by previous studies, the role of these hub countries in the diffusion of the technology has not been analyzed. Using a sample of 21 countries and 14 manufacturing industries during the period 1995-2007, I explore the role of hubs as the mediator of knowledge by classifying countries and industries based on a ``centrality'' measure. I find that R&D spillovers from exporters with High centrality are the largest, suggesting that hub countries play an important role in both gathering and diffusing knowledge. I also find that countries with Middle centrality are getting important in the diffusion of knowledge. Finally, positive spillover effects from own are observed only in the G5 countries.
    
[^5]: 学习随机过程的有条件生成模型

    Conditional Generative Models for Learning Stochastic Processes. (arXiv:2304.10382v1 [quant-ph])

    [http://arxiv.org/abs/2304.10382](http://arxiv.org/abs/2304.10382)

    提出了一种称为 C-qGAN 的框架，利用量子电路结构实现了有效的状态准备过程，可以利用该方法加速蒙特卡罗分析等算法，并将其应用于亚式期权衍生品定价的任务中。

    

    提出了一种学习多模态分布的框架，称为条件量子生成对抗网络（C-qGAN）。神经网络结构严格采用量子电路，因此被证明能够比当前的方法更有效地表示状态准备过程。这种方法有潜力加速蒙特卡罗分析等算法。特别地，在展示了网络在学习任务中的有效性后，将该技术应用于定价亚式期权衍生品，为未来研究其他路径相关期权打下基础。

    A framework to learn a multi-modal distribution is proposed, denoted as the Conditional Quantum Generative Adversarial Network (C-qGAN). The neural network structure is strictly within a quantum circuit and, as a consequence, is shown to represents a more efficient state preparation procedure than current methods. This methodology has the potential to speed-up algorithms, such as Monte Carlo analysis. In particular, after demonstrating the effectiveness of the network in the learning task, the technique is applied to price Asian option derivatives, providing the foundation for further research on other path-dependent options.
    
[^6]: 加密货币的集体动力学、多样化和最优投资组合构建

    Collective dynamics, diversification and optimal portfolio construction for cryptocurrencies. (arXiv:2304.08902v1 [q-fin.ST])

    [http://arxiv.org/abs/2304.08902](http://arxiv.org/abs/2304.08902)

    本研究旨在探究加密货币市场的集体动力学和投资组合多样化，以及尝试确认先前在股票市场中建立的结果是否适用于加密货币市场。

    

    自发明以来，加密货币市场经常被描述为一个不成熟的市场，其波动率显著波动，并且偶尔被描述为缺乏规律。人们一直在猜测它在多样化投资组合中扮演着什么角色。例如，加密货币是否是通货膨胀对冲的工具，还是一种遵循市场情绪的投机性投资，其Beta值放大了呢？本文旨在调查加密货币市场是否展现出与更成熟的股票市场类似的复杂数学特性。我们关注加密货币市场的集体动力学和投资组合多样化，并研究在加密货币市场中先前建立的股票市场结果是否成立，以及程度如何。

    Since its conception, the cryptocurrency market has been frequently described as an immature market, characterized by significant swings in volatility and occasionally described as lacking rhyme or reason. There has been great speculation as to what role it plays in a diversified portfolio. For instance, is cryptocurrency exposure an inflationary hedge or a speculative investment that follows broad market sentiment with amplified beta? This paper aims to investigate whether the cryptocurrency market has recently exhibited similarly nuanced mathematical properties as the much more mature equity market. Our focus is on collective dynamics and portfolio diversification in the cryptocurrency market, and examining whether previously established results in the equity market hold in the cryptocurrency market, and to what extent.
    
[^7]: 基于成熟度模型的信息系统网络保险定价

    Pricing cyber-insurance for systems via maturity models. (arXiv:2302.04734v2 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2302.04734](http://arxiv.org/abs/2302.04734)

    本篇论文提出了一种使用安全成熟度模型的方法，以评估组织的安全水平并确定网络保险的适当保费。

    

    对于与信息技术系统相关的风险进行保险定价提出了一个综合的建议，结合运营管理、安全和经济学，提出了一个社会经济模型。该模型包括实体关系图、安全成熟度模型和经济模型，解决了一个长期以来的研究难题，即如何在设计和定价网络保险政策时捕捉组织结构。文中提出了一个新的挑战，即网络保险的数据历史有限，不能直接应用于其它险种，因此提出一个安全成熟度模型，以评估组织的安全水平并确定相应的保险费用。

    Pricing insurance for risks associated with information technology systems presents a complex modelling challenge, combining the disciplines of operations management, security, and economics. This work proposes a socioeconomic model for cyber-insurance decisions compromised of entity relationship diagrams, security maturity models, and economic models, addressing a long-standing research challenge of capturing organizational structure in the design and pricing of cyber-insurance policies. Insurance pricing is usually informed by the long experience insurance companies have of the magnitude and frequency of losses that arise in organizations based on their size, industry sector, and location. Consequently, their calculations of premia will start from a baseline determined by these considerations. A unique challenge of cyber-insurance is that data history is limited and not necessarily informative of future loss risk meaning that established actuarial methodology for other lines of insur
    

