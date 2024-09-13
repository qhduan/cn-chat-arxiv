# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Hierarchical Structure of Uncertainty.](http://arxiv.org/abs/2311.14219) | 本论文介绍了一种称为不确定性空间的新概念，用于表达不同层级的不确定性。通过引入不确定性空间序列，可以对金融和经济领域的不确定性进行更准确和细致的建模。 |
| [^2] | [Greeks' pitfalls for the COS method in the Laplace model.](http://arxiv.org/abs/2306.08421) | 本文在拉普拉斯模型中研究了欧式期权中希腊字母Speed的解析表达式，并提供了COS方法近似Speed所需满足的充分条件，实验证明未满足条件时结果可能不准确。 |
| [^3] | [Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models.](http://arxiv.org/abs/2304.07619) | 本研究探究了使用ChatGPT及其他大型语言模型预测股市回报的潜力，发现ChatGPT的预测表现优于传统情感分析方法，而基础模型无法准确预测股票价格变化，表明复杂模型可预测能力的崛起。这表明在投资决策过程中引入先进的语言模型可以提高预测准确性并增强定量交易策略的表现。 |
| [^4] | [Reconciling rough volatility with jumps.](http://arxiv.org/abs/2303.07222) | 本文提出了一种将粗糙波动和跳跃模型相结合的回归Heston模型，能够生成类似于粗糙、超粗糙和跳跃模型的平价偏斜。 |
| [^5] | [Synchronization of endogenous business cycles.](http://arxiv.org/abs/2002.06555) | 本文研究了内生经济周期的同步化，发现需求驱动的模型更能产生商业周期的同步振荡，并通过将非线性动力学、冲击传播和网络结构相互作用的特征值分解方法来理解同步机制。 |

# 详细

[^1]: 不确定性的层级结构

    Hierarchical Structure of Uncertainty. (arXiv:2311.14219v3 [q-fin.MF] UPDATED)

    [http://arxiv.org/abs/2311.14219](http://arxiv.org/abs/2311.14219)

    本论文介绍了一种称为不确定性空间的新概念，用于表达不同层级的不确定性。通过引入不确定性空间序列，可以对金融和经济领域的不确定性进行更准确和细致的建模。

    

    金融危机和传染病危机等未知事件的经验揭示了在固定的概率测度下测量风险的局限性。为了解决这个问题，在金融界长期以来一直认识到所谓的不确定性的重要性，它允许概率测度本身发生变化。另一方面，在经济学领域对主观概率测度进行了许多研究。但即使在这些情况下，研究也是基于两个层次的不确定性：在已知传统概率测度（概率分布）的情况下的风险，以及由于主观概率测度可以在某个空间中任意取值所带来的不确定性。在这项研究中，我们通过引入一个称为不确定性空间（uncertainty spaces）的新概念来表达所谓的n层不确定性，这是概率空间的扩展概念，并使用不确定性空间序列（U-sequence）来进行研究。

    The experience of unknown events such as financial crises and infectious disease crises has revealed the limitations of measuring risk under a fixed probability measure. In order to solve this problem, the importance of so-called ambiguity, which allows the probability measure itself to change, has long been recognized in the financial world. On the other hand, there have been many studies o n subjective probability measures in the field of economics.But even in those cases, the studies are based on the two levels of uncertainty: risk when a conventional probability measure (probability distribution) is known, and ambiguity due to the fact that the subjective probability measure can be taken arbitrarily in a certain space. In this study, we express n-layer uncertainty, which we call hierarchical uncertainty by introducing a new concepts called uncertainty spaces which is an extended concept of probability spaces and U-sequence that are sequences of uncertainty spaces. We use U-sequence
    
[^2]: 拉普拉斯模型中COS方法中希腊字母的陷阱

    Greeks' pitfalls for the COS method in the Laplace model. (arXiv:2306.08421v1 [q-fin.CP])

    [http://arxiv.org/abs/2306.08421](http://arxiv.org/abs/2306.08421)

    本文在拉普拉斯模型中研究了欧式期权中希腊字母Speed的解析表达式，并提供了COS方法近似Speed所需满足的充分条件，实验证明未满足条件时结果可能不准确。

    

    希腊字母Delta，Gamma和Speed是欧式期权相对于标的资产当前价格的一阶、二阶和三阶导数。傅里叶余弦级数展开法（COS 方法）是一种数值方法，用于近似欧式期权的价格和希腊字母。我们开发了拉普拉斯模型中各种欧式期权的Speed的闭合形式表达式，并提供了COS方法近似Speed所需满足的充分条件。我们实证表明，如果这些充分条件不满足，COS方法可能会产生数字上没有意义的结果。

    The Greeks Delta, Gamma and Speed are the first, second and third derivative of a European option with respect to the current price of the underlying. The Fourier cosine series expansion method (COS method) is a numerical method for approximating the price and the Greeks of European options. We develop a closed-form expression of Speed of various European options in the Laplace model and we provide sufficient conditions for the COS method to approximate Speed. We show empirically that the COS method may produce numerically nonsensical results if theses sufficient conditions are not met.
    
[^3]: ChatGPT是否能够预测股票价格波动？回报可预测性与大语言模型。

    Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models. (arXiv:2304.07619v1 [q-fin.ST])

    [http://arxiv.org/abs/2304.07619](http://arxiv.org/abs/2304.07619)

    本研究探究了使用ChatGPT及其他大型语言模型预测股市回报的潜力，发现ChatGPT的预测表现优于传统情感分析方法，而基础模型无法准确预测股票价格变化，表明复杂模型可预测能力的崛起。这表明在投资决策过程中引入先进的语言模型可以提高预测准确性并增强定量交易策略的表现。

    

    本文研究了使用情感分析预测股市回报的潜力，探讨了使用ChatGPT以及其他大语言模型在预测股市回报方面的表现。我们使用ChatGPT判断新闻标题对公司股票价格是好消息、坏消息或无关消息。通过计算数字分数，我们发现这些"ChatGPT分数"和随后的日常股票市场回报之间存在正相关性。而且，ChatGPT的表现优于传统的情感分析方法。同时，我们发现GPT-1、GPT-2和BERT等基础模型无法准确预测回报，这表明回报可预测性是复杂模型的一种新兴能力。我们的研究结果表明，将先进的语言模型纳入投资决策过程可以产生更准确的预测，并提高定量交易策略的表现。

    We examine the potential of ChatGPT, and other large language models, in predicting stock market returns using sentiment analysis of news headlines. We use ChatGPT to indicate whether a given headline is good, bad, or irrelevant news for firms' stock prices. We then compute a numerical score and document a positive correlation between these ``ChatGPT scores'' and subsequent daily stock market returns. Further, ChatGPT outperforms traditional sentiment analysis methods. We find that more basic models such as GPT-1, GPT-2, and BERT cannot accurately forecast returns, indicating return predictability is an emerging capacity of complex models. Our results suggest that incorporating advanced language models into the investment decision-making process can yield more accurate predictions and enhance the performance of quantitative trading strategies.
    
[^4]: 将粗糙波动与跳跃相结合

    Reconciling rough volatility with jumps. (arXiv:2303.07222v1 [q-fin.MF])

    [http://arxiv.org/abs/2303.07222](http://arxiv.org/abs/2303.07222)

    本文提出了一种将粗糙波动和跳跃模型相结合的回归Heston模型，能够生成类似于粗糙、超粗糙和跳跃模型的平价偏斜。

    This paper proposes a reversionary Heston model that reconciles rough volatility models and jump models, and is capable of generating at-the-money skews similar to those generated by rough, hyper-rough, and jump models.

    我们使用一类具有快速均值回归和大波动率的回归Heston模型来调和粗糙波动模型和跳跃模型。从具有Hurst指数$H \in (-1/2,1/2)$的超粗糙Heston模型开始，我们推导出一维回归Heston类型模型的马尔可夫逼近类。这些代理编码了一个在爆炸性波动率和由回归时间尺度$\epsilon>0$和无约束参数$H \in \mathbb R$控制的快速均值回归速度之间的权衡。将$\epsilon$发送到0会导致回归Heston模型收敛于基于参数H的不同显式渐近极限。特别地，对于$H \leq -1/2$，回归Heston模型收敛于正态逆高斯类型的Lévy跳跃过程类。数值实例表明，回归Heston模型能够生成类似于粗糙、超粗糙和跳跃模型生成的平价偏斜。

    We reconcile rough volatility models and jump models using a class of reversionary Heston models with fast mean reversions and large vol-of-vols. Starting from hyper-rough Heston models with a Hurst index $H \in (-1/2,1/2)$, we derive a Markovian approximating class of one dimensional reversionary Heston-type models. Such proxies encode a trade-off between an exploding vol-of-vol and a fast mean-reversion speed controlled by a reversionary time-scale $\epsilon>0$ and an unconstrained parameter $H \in \mathbb R$. Sending $\epsilon$ to 0 yields convergence of the reversionary Heston model towards different explicit asymptotic regimes based on the value of the parameter H. In particular, for $H \leq -1/2$, the reversionary Heston model converges to a class of L\'evy jump processes of Normal Inverse Gaussian type. Numerical illustrations show that the reversionary Heston model is capable of generating at-the-money skews similar to the ones generated by rough, hyper-rough and jump models.
    
[^5]: 内生经济周期的同步化研究

    Synchronization of endogenous business cycles. (arXiv:2002.06555v3 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2002.06555](http://arxiv.org/abs/2002.06555)

    本文研究了内生经济周期的同步化，发现需求驱动的模型更能产生商业周期的同步振荡，并通过将非线性动力学、冲击传播和网络结构相互作用的特征值分解方法来理解同步机制。

    

    商业周期在不同国家之间呈现正相关性（“共振”）。然而，那些把共振归因于外部冲击传导的标准模型很难产生与数据相同程度的共振。本文研究通过某种非线性动力学——极限环或混沌，来内生地产生商业周期的模型。这些模型产生更强的共振，因为它们将冲击传导与内生动态的同步化相结合。特别地，我们研究了一种需求驱动的模型，其中商业周期源于国内的战略互补性，并通过国际贸易联系同步振荡。我们开发了一种特征值分解方法来探讨非线性动力学、冲击传播和网络结构之间的相互作用，并使用这种理论来理解同步机制。接下来，我们将模型校准到24个国家的数据上，并展示了实证共振程度。

    Business cycles are positively correlated (``comove'') across countries. However, standard models that attribute comovement to propagation of exogenous shocks struggle to generate a level of comovement that is as high as in the data. In this paper, we consider models that produce business cycles endogenously, through some form of non-linear dynamics -- limit cycles or chaos. These models generate stronger comovement, because they combine shock propagation with synchronization of endogenous dynamics. In particular, we study a demand-driven model in which business cycles emerge from strategic complementarities within countries, synchronizing their oscillations through international trade linkages. We develop an eigendecomposition that explores the interplay between non-linear dynamics, shock propagation and network structure, and use this theory to understand the mechanisms of synchronization. Next, we calibrate the model to data on 24 countries and show that the empirical level of comov
    

