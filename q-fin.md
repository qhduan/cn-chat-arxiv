# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An innovative Deep Learning Based Approach for Accurate Agricultural Crop Price Prediction.](http://arxiv.org/abs/2304.09761) | 本文提出了一种基于深度学习的方法，利用历史价格信息、气候条件、土壤类型、地理位置和其他关键决策因素，精确预测农产品价格。该方法使用图神经网络和卷积神经网络相结合，适用于有噪声的历史数据，并且表现至少比现有文献中的结果好20%。 |
| [^2] | [Application of Tensor Neural Networks to Pricing Bermudan Swaptions.](http://arxiv.org/abs/2304.09750) | 本论文用张量神经网络(TNN)对百慕大掉期进行定价，相比于传统方法，TNN具有更快的收敛速度和减少参数敏感度的优点。 |
| [^3] | [An extension of martingale transport and stability in robust finance.](http://arxiv.org/abs/2304.09551) | 本文介绍了在鲁棒金融中考虑额外信息的弱鞅最优输运问题，证明了所得问题的风险中性边际分布的稳定性，并提供了一个关键步骤，它将任何鞅耦合逼近为具有指定边际分布的鞅耦合。 |
| [^4] | [The Unintended Consequences of Censoring Digital Technology -- Evidence from Italy's ChatGPT Ban.](http://arxiv.org/abs/2304.09339) | 意大利禁止ChatGPT产生了短期的生产力干扰，但也导致了审查绕过工具的显著增加。 |
| [^5] | [How to handle the COS method for option pricing.](http://arxiv.org/abs/2303.16012) | 介绍了用于欧式期权定价的 Fourier余弦展开 (COS) 方法，通过指定截断范围和项数N进行逼近，文章提出明确的N的上界，对密度平滑并指数衰减的情况，COS方法的收敛阶数至少是指数收敛阶数。 |
| [^6] | [Understanding Model Complexity for temporal tabular and multi-variate time series, case study with Numerai data science tournament.](http://arxiv.org/abs/2303.07925) | 本文采用 Numerai 数据科学竞赛的数据，探究了多变量时间序列建模中不同特征工程和降维方法的应用；提出了一种新的集成方法，用于高维时间序列建模，该方法在通用性、鲁棒性和效率上优于一些深度学习模型。 |
| [^7] | [Sequential Elimination Contests with All-Pay Auctions.](http://arxiv.org/abs/2205.08104) | 本文研究了带全付拍卖的顺序淘汰比赛，在设计者无法得知所有选手类型情况下，对于有限的选手数量，我们明确定义了均衡策略和期望的最优机制，并为设计者提供了一些众包比赛的设计启示。 |
| [^8] | [The Rising Entropy of English in the Attention Economy.](http://arxiv.org/abs/2107.12848) | 美式英语的熵值自1900年左右持续上升并且不同媒体之间存在着熵值上的差异，作者认为这是由于注意力经济的生态系统导致的。 |
| [^9] | [Peace through bribing.](http://arxiv.org/abs/2107.11575) | 本文研究了一种贿赂带来和平的模型。作者发现在该模型中和平是不可能的，并表征了实现和平的必要和充分条件。作者还考虑了一个请求模型，在该模型中可以实现和平安全。 |

# 详细

[^1]: 一种基于深度学习的创新方法用于准确的农产品价格预测

    An innovative Deep Learning Based Approach for Accurate Agricultural Crop Price Prediction. (arXiv:2304.09761v1 [cs.LG])

    [http://arxiv.org/abs/2304.09761](http://arxiv.org/abs/2304.09761)

    本文提出了一种基于深度学习的方法，利用历史价格信息、气候条件、土壤类型、地理位置和其他关键决策因素，精确预测农产品价格。该方法使用图神经网络和卷积神经网络相结合，适用于有噪声的历史数据，并且表现至少比现有文献中的结果好20%。

    

    农产品价格的准确预测对于农业各利益相关者（农民、消费者、零售商、批发商和政府）的决策至关重要，特别是对农民的经济福祉产生重要影响。本文的目标是利用历史价格信息、气候条件、土壤类型、地理位置和其他关键决策因素，精确预测农产品价格。这是一个技术挑战，以前也曾被尝试过。本文提出了一种创新的基于深度学习的方法，以实现在价格预测中的提高准确性。所提出的方法使用图神经网络（GNNs）与标准卷积神经网络（CNN）模型相结合，利用价格的地理空间相关性。我们的方法适用于有噪声的历史数据，并且表现至少比现有文献中的结果好20%。

    Accurate prediction of agricultural crop prices is a crucial input for decision-making by various stakeholders in agriculture: farmers, consumers, retailers, wholesalers, and the Government. These decisions have significant implications including, most importantly, the economic well-being of the farmers. In this paper, our objective is to accurately predict crop prices using historical price information, climate conditions, soil type, location, and other key determinants of crop prices. This is a technically challenging problem, which has been attempted before. In this paper, we propose an innovative deep learning based approach to achieve increased accuracy in price prediction. The proposed approach uses graph neural networks (GNNs) in conjunction with a standard convolutional neural network (CNN) model to exploit geospatial dependencies in prices. Our approach works well with noisy legacy data and produces a performance that is at least 20% better than the results available in the li
    
[^2]: 张量神经网络在百慕大掉期定价中的应用

    Application of Tensor Neural Networks to Pricing Bermudan Swaptions. (arXiv:2304.09750v1 [q-fin.CP])

    [http://arxiv.org/abs/2304.09750](http://arxiv.org/abs/2304.09750)

    本论文用张量神经网络(TNN)对百慕大掉期进行定价，相比于传统方法，TNN具有更快的收敛速度和减少参数敏感度的优点。

    

    Cheyette模型是一种准高斯波动率利率模型，广泛用于定价利率衍生品，例如欧式掉期和百慕大掉期，而蒙特卡罗模拟已成为行业标准。在低维度下，这些方法为欧式掉期提供了准确而稳健的价格，但即使在这种计算简单的情况下，当使用状态变量作为回归器时，它们也会低估百慕大掉期的价值。这主要是由于所用回归器中预先确定的基函数数量有限。此外，在高维环境中，这些方法也面临着维度灾难的问题。为了解决这些问题，研究者提出利用张量神经网络(TNN)来进行百慕大掉期的定价。研究结果表明，与传统方法相比，TNN具有更快的收敛速度，对于回归器中所用基函数的数量等参数，减少了敏感度。数值实验证实TNN能够在高维度情况下准确地定价欧式掉期和百慕大掉期。

    The Cheyette model is a quasi-Gaussian volatility interest rate model widely used to price interest rate derivatives such as European and Bermudan Swaptions for which Monte Carlo simulation has become the industry standard. In low dimensions, these approaches provide accurate and robust prices for European Swaptions but, even in this computationally simple setting, they are known to underestimate the value of Bermudan Swaptions when using the state variables as regressors. This is mainly due to the use of a finite number of predetermined basis functions in the regression. Moreover, in high-dimensional settings, these approaches succumb to the Curse of Dimensionality. To address these issues, Deep-learning techniques have been used to solve the backward Stochastic Differential Equation associated with the value process for European and Bermudan Swaptions; however, these methods are constrained by training time and memory. To overcome these limitations, we propose leveraging Tensor Neura
    
[^3]: 鲁棒金融中的鞅输运扩展与稳定性

    An extension of martingale transport and stability in robust finance. (arXiv:2304.09551v1 [math.PR])

    [http://arxiv.org/abs/2304.09551](http://arxiv.org/abs/2304.09551)

    本文介绍了在鲁棒金融中考虑额外信息的弱鞅最优输运问题，证明了所得问题的风险中性边际分布的稳定性，并提供了一个关键步骤，它将任何鞅耦合逼近为具有指定边际分布的鞅耦合。

    

    虽然鲁棒金融中的许多问题可以在鞅最优输运框架或其弱扩展中提出，例如 VIX 期货的次复制价格、美式期权的鲁棒定价或影子耦合的构造，但其他问题则需要将额外的信息纳入优化问题中。在本篇论文中，我们通过在弱鞅最优输运问题中引入一个额外参数来考虑这个额外信息。我们证明了所得问题相对于基础资产的风险中性边际分布的稳定性，从而扩展了 \cite{BeJoMaPa21b} 中的结果。其中一个关键步骤是将 \cite{BJMP22} 中的主要结果推广到该设置中，该结果确立了任何鞅耦合都可以通过一系列具有指定边际分布的鞅耦合逼近，前提是这些子序列的边际分布在适当的意义下收敛。

    While many questions in robust finance can be posed in the martingale optimal transport framework or its weak extension, others like the subreplication price of VIX futures, the robust pricing of American options or the construction of shadow couplings necessitate additional information to be incorporated into the optimization problem beyond that of the underlying asset. In the present paper, we take into account this extra information by introducing an additional parameter to the weak martingale optimal transport problem. We prove the stability of the resulting problem with respect to the risk neutral marginal distributions of the underlying asset, thus extending the results in \cite{BeJoMaPa21b}. A key step is the generalization of the main result in \cite{BJMP22} to include the extra parameter into the setting. This result establishes that any martingale coupling can be approximated by a sequence of martingale couplings with specified marginals, provided that the marginals of this s
    
[^4]: 禁止数字技术的意外后果——以意大利ChatGPT禁令为例的证据

    The Unintended Consequences of Censoring Digital Technology -- Evidence from Italy's ChatGPT Ban. (arXiv:2304.09339v1 [econ.GN])

    [http://arxiv.org/abs/2304.09339](http://arxiv.org/abs/2304.09339)

    意大利禁止ChatGPT产生了短期的生产力干扰，但也导致了审查绕过工具的显著增加。

    

    本文分析了ChatGPT，一种生成式预训练变压器聊天机器人被禁止对个人生产率的影响。我们首先收集了超过8,000名专业GitHub用户在意大利和其他欧洲国家的每小时编码产出数据，以分析禁令对个人生产力的影响。将高频率数据与禁令的突然宣布结合在一起，运用差异法，我们发现在禁令后的前两个工作日，意大利开发者的产出减少了约50％，之后逐渐恢复。运用合成控制方法来分析每日Google搜索和Tor使用数据，结果显示，该禁令导致了绕过审查的工具的使用量显著增加。我们的研究结果表明，用户很快采取绕过互联网限制的策略，但这种适应活动会造成短期的干扰和影响生产力。

    We analyse the effects of the ban of ChatGPT, a generative pre-trained transformer chatbot, on individual productivity. We first compile data on the hourly coding output of over 8,000 professional GitHub users in Italy and other European countries to analyse the impact of the ban on individual productivity. Combining the high-frequency data with the sudden announcement of the ban in a difference-in-differences framework, we find that the output of Italian developers decreased by around 50% in the first two business days after the ban and recovered after that. Applying a synthetic control approach to daily Google search and Tor usage data shows that the ban led to a significant increase in the use of censorship bypassing tools. Our findings show that users swiftly implement strategies to bypass Internet restrictions but this adaptation activity creates short-term disruptions and hampers productivity.
    
[^5]: 如何处理用于期权定价的 COS 方法

    How to handle the COS method for option pricing. (arXiv:2303.16012v1 [q-fin.CP])

    [http://arxiv.org/abs/2303.16012](http://arxiv.org/abs/2303.16012)

    介绍了用于欧式期权定价的 Fourier余弦展开 (COS) 方法，通过指定截断范围和项数N进行逼近，文章提出明确的N的上界，对密度平滑并指数衰减的情况，COS方法的收敛阶数至少是指数收敛阶数。

    

    Fourier余弦展开（COS）方法用于高效地计算欧式期权价格。要应用COS方法，必须指定两个参数：对数收益率密度的截断范围和用余弦级数逼近截断密度的项数N。如何选择截断范围已经为人所知。在这里，我们还能找到一个明确的并且有用的项数N的界限。我们还进一步表明，如果密度是平滑的并且呈指数衰减，则COS方法至少具有指数收敛阶数。但是，如果密度平滑但有重尾巴，就像在有限矩阵log稳定模型中一样，则COS方法没有指数收敛阶数。数值实验确认了理论发现。

    The Fourier cosine expansion (COS) method is used for pricing European options numerically very efficiently. To apply the COS method, one has to specify two parameters: a truncation range for the density of the log-returns and a number of terms N to approximate the truncated density by a cosine series. How to choose the truncation range is already known. Here, we are able to find an explicit and useful bound for N as well. We further show that the COS method has at least an exponential order of convergence if the density is smooth and decays exponentially. But, if the density is smooth and has heavy tails like in the Finite Moment Log Stable model, the COS method has not an exponential order of convergence. Numerical experiments confirm the theoretical findings.
    
[^6]: 通过 Numerai 数据科学竞赛案例，理解时间表格和多变量时间序列的模型复杂度

    Understanding Model Complexity for temporal tabular and multi-variate time series, case study with Numerai data science tournament. (arXiv:2303.07925v1 [cs.LG])

    [http://arxiv.org/abs/2303.07925](http://arxiv.org/abs/2303.07925)

    本文采用 Numerai 数据科学竞赛的数据，探究了多变量时间序列建模中不同特征工程和降维方法的应用；提出了一种新的集成方法，用于高维时间序列建模，该方法在通用性、鲁棒性和效率上优于一些深度学习模型。

    

    本文探究了在多变量时间序列建模中使用不同特征工程和降维方法的应用。利用从 Numerai 数据竞赛创建的特征目标交叉相关时间序列数据集，我们证明在过度参数化的情况下，不同特征工程方法的性能与预测会收敛到可由再生核希尔伯特空间刻画的相同平衡态。我们提出了一种新的集成方法，该方法结合了不同的随机非线性变换，随后采用岭回归模型进行高维时间序列建模。与一些常用的用于序列建模的深度学习模型（如 LSTM 和 transformer）相比，我们的方法更加鲁棒（在不同的随机种子下具有较低的模型方差，且对架构的选择不太敏感），并且更有效率。我们方法的另一个优势在于模型的简单性，因为没有必要使用复杂的深度学习框架。

    In this paper, we explore the use of different feature engineering and dimensionality reduction methods in multi-variate time-series modelling. Using a feature-target cross correlation time series dataset created from Numerai tournament, we demonstrate under over-parameterised regime, both the performance and predictions from different feature engineering methods converge to the same equilibrium, which can be characterised by the reproducing kernel Hilbert space. We suggest a new Ensemble method, which combines different random non-linear transforms followed by ridge regression for modelling high dimensional time-series. Compared to some commonly used deep learning models for sequence modelling, such as LSTM and transformers, our method is more robust (lower model variance over different random seeds and less sensitive to the choice of architecture) and more efficient. An additional advantage of our method is model simplicity as there is no need to use sophisticated deep learning frame
    
[^7]: 带全付拍卖的顺序淘汰比赛研究

    Sequential Elimination Contests with All-Pay Auctions. (arXiv:2205.08104v2 [cs.GT] UPDATED)

    [http://arxiv.org/abs/2205.08104](http://arxiv.org/abs/2205.08104)

    本文研究了带全付拍卖的顺序淘汰比赛，在设计者无法得知所有选手类型情况下，对于有限的选手数量，我们明确定义了均衡策略和期望的最优机制，并为设计者提供了一些众包比赛的设计启示。

    

    本文研究了一种顺序淘汰比赛，比赛前需要过滤选手。这一模型的构建是基于许多众包比赛只有有限的评审资源，需要提高提交物品的整体质量这一实践的考虑。在第一种情况下，我们考虑选手的能力排名的知情设计者，允许 $2\leq n_2 \leq n_1$ 的前 $n_2$ 名选手参加比赛。进入比赛的选手会根据自己的能力排名可以进入前 $n_2$ 名的信号来更新对对手能力的信念。我们发现，即使具有IID先验，他们的后验信念仍然存在相关性，并且取决于选手的个人能力。我们明确地刻画了对称的和唯一的贝叶斯均衡策略。我们发现，每个被录取选手的均衡付出随 $n_2$ 的增加而增加，当 $n_2 \in [\lfloor{(n_1+1)/2}\rfloor+1,n_1]$ 时，但在 $n_2 \in [2, \lfloor{(n_1+1)/2}\rfloor]$ 时不一定单调。然后，我们考虑了更加现实的情况，设计者只具有有关选手类型的部分信息，选手类型在 $[0,1]$ 上独立均匀分布。我们得到了关于接受选手的期望最优机制性能的尖锐界限，以及每个进入比赛的选手均衡付出的界限。我们的结果对于众包比赛的设计具有重要意义。

    We study a sequential elimination contest where players are filtered prior to the round of competing for prizes. This is motivated by the practice that many crowdsourcing contests have very limited resources of reviewers and want to improve the overall quality of the submissions. We first consider a setting where the designer knows the ranking of the abilities (types) of all $n_1$ registered players, and admit the top $n_2$ players with $2\leq n_2 \leq n_1$ into the contest. The players admitted into the contest update their beliefs about their opponents based on the signal that their abilities are among the top $n_2$. We find that their posterior beliefs, even with IID priors, are correlated and depend on players' private abilities.  We explicitly characterize the symmetric and unique Bayesian equilibrium strategy. We find that each admitted player's equilibrium effort is increasing in $n_2$ when $n_2 \in [\lfloor{(n_1+1)/2}\rfloor+1,n_1]$, but not monotone in general when $n_2 \in [2
    
[^8]: 注意力经济中英语熵值的不断上升

    The Rising Entropy of English in the Attention Economy. (arXiv:2107.12848v4 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2107.12848](http://arxiv.org/abs/2107.12848)

    美式英语的熵值自1900年左右持续上升并且不同媒体之间存在着熵值上的差异，作者认为这是由于注意力经济的生态系统导致的。

    

    我们提供了证据表明自1900年左右起，美式英语的词汇熵值一直在稳步上升，这与现有社会语言学理论的预测相反。我们还发现，不同媒体之间存在着词汇熵值上的差异，短格式媒体如新闻和杂志的熵值比长格式媒体更高，而社交媒体的熵值则更高。为了解释这些结果，我们提出了一种注意力经济生态模型，结合了Zipf定律和信息搜寻的思想。在该模型中，媒体消费者在考虑信息检索成本的同时，最大化信息效用率，而媒体生产者则适应减少检索成本的技术，推动他们生成越来越短的更高熵值的内容。

    We present evidence that the word entropy of American English has been rising steadily since around 1900, contrary to predictions from existing sociolinguistic theories. We also find differences in word entropy between media categories, with short-form media such as news and magazines having higher entropy than long-form media, and social media feeds having higher entropy still. To explain these results we develop an ecological model of the attention economy that combines ideas from Zipf's law and information foraging. In this model, media consumers maximize information utility rate taking into account the costs of information search, while media producers adapt to technologies that reduce search costs, driving them to generate higher entropy content in increasingly shorter formats.
    
[^9]: 贿赂带来的和平

    Peace through bribing. (arXiv:2107.11575v3 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2107.11575](http://arxiv.org/abs/2107.11575)

    本文研究了一种贿赂带来和平的模型。作者发现在该模型中和平是不可能的，并表征了实现和平的必要和充分条件。作者还考虑了一个请求模型，在该模型中可以实现和平安全。

    

    本文研究了一个模型，在这个模型中，冲突双方在升级为“全付出拍卖”之前，一方可以向另一方提供一个“接受或离开”的贿赂，以达成和平解决。与现有文献不同的是，我们发现在我们的模型中和平是不可能的。我们表征了和平实施的必要和充分条件。此外，我们发现分离均衡不存在，在任何非和平均衡中的(路径上的)贿赂数量最多为两个。我们还考虑了一个请求模型，并表征了稳健和平均衡的必要和充分条件，所有这些均由相同的(路径上的)请求维持。与贿赂模型相反，在请求模型中可以实现和平安全。

    We study a model in which before a conflict between two parties escalates into a war (in the form of an all-pay auction), a party can offer a take-it-or-leave-it bribe to the other for a peaceful settlement. In contrast to the received literature, we find that peace security is impossible in our model. We characterize the necessary and sufficient conditions for peace implementability. Furthermore, we find that separating equilibria do not exist and the number of (on-path) bribes in any non-peaceful equilibria is at most two. We also consider a requesting model and characterize the necessary and sufficient conditions for the existence of robust peaceful equilibria, all of which are sustained by the identical (on-path) request. Contrary to the bribing model, peace security is possible in the requesting model.
    

