# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Permutation invariant Gaussian matrix models for financial correlation matrices.](http://arxiv.org/abs/2306.04569) | 使用高频外汇市场数据构建了一个包含446天每天一个矩阵的相关矩阵集合，构建了一个可预测三次和四次多项式期望值的置换不变高斯矩阵模型，并表明该模型预测结果与传统金融相关矩阵模型一致。 |
| [^2] | [International Spillovers of ECB Interest Rates: Monetary Policy & Information Effects.](http://arxiv.org/abs/2306.04562) | 本文研究了欧洲央行利率的国际溢出效应，并证明忽略信息效应会导致其国际溢出失真。作者通过将纯货币政策冲击与信息效应区分开来，发现其国际产量、汇率和股票指数的溢出效应比标准策略高2到3倍。 |
| [^3] | [Calibrating Chevron for Preemption.](http://arxiv.org/abs/2306.04463) | 本文探讨了最高法院在Chevron决定下对机构预防措施的解释应该遵从哪些规则，对传统州权领域内的国会意图采取高度重视。 |
| [^4] | [An Empirical Study of Obstacle Preemption in the Supreme Court.](http://arxiv.org/abs/2306.04462) | 本文实证分析了最高法院障碍物抢先权的决定，并揭示了令人惊讶的反障碍物抢先权联盟的形成，该联盟由托马斯大法官逐渐与法院自由党派组成的五个法官投票集团。 |
| [^5] | [Dynamic Programming on a Quantum Annealer: Solving the RBC Model.](http://arxiv.org/abs/2306.04285) | 这篇论文介绍了在量子退火器上使用动态规划解决经济模型的方法，并取得了数量级的加速，这对于解决更具挑战性的经济问题具有潜在价值。 |
| [^6] | [Bachelier's Market Model for ESG Asset Pricing.](http://arxiv.org/abs/2306.04158) | 本论文的创新点在于提出了一种基于巴舍利尔市场模型的ESG资产定价模型，并且将其应用于期权定价估值中。 |
| [^7] | [Sustainability criterion implied externality pricing for resource extraction.](http://arxiv.org/abs/2306.04065) | 该文构建了动态模型，应用可持续性标准隐含外部性定价，得出最优采伐进度，并揭示外部性定价对采伐进度的影响。 |
| [^8] | [ChatGPT Informed Graph Neural Network for Stock Movement Prediction.](http://arxiv.org/abs/2306.03763) | 该研究介绍了一种新的框架，利用ChatGPT技术增强图神经网络，能够从财经新闻中提取出不断变化的网络结构，并用于股票价格预测，获得了超过基于深度学习的最新基准的表现，提示了ChatGPT在文本推断和金融预测方面的潜力。 |
| [^9] | [Building Floorspace in China: A Dataset and Learning Pipeline.](http://arxiv.org/abs/2303.02230) | 本文构建了中国40个主要城市楼宇建筑面积数据集并利用多任务对象分割器方法学习了建筑物占地面积和高度，为城市规划提供数据支持。 |
| [^10] | [Variance of entropy for testing time-varying regimes with an application to meme stocks.](http://arxiv.org/abs/2211.05415) | 本文提出了一种假设检验程序，以检验时间序列Shannon熵的恒定零假设，对立的是两个相邻时间段之间熵的显著变化的备择假设，并找到了Shannon熵估计量的方差的无偏估计。 |
| [^11] | [The roughness exponent and its model-free estimation.](http://arxiv.org/abs/2111.10301) | 本文提出了对于函数$x$，基于Faber-Schauder系数的温和条件，无需对其动态作出任何假设即可一致地估计其粗糙指数，并给出了推广方法。 |

# 详细

[^1]: 金融相关矩阵的置换不变高斯矩阵模型

    Permutation invariant Gaussian matrix models for financial correlation matrices. (arXiv:2306.04569v1 [q-fin.ST])

    [http://arxiv.org/abs/2306.04569](http://arxiv.org/abs/2306.04569)

    使用高频外汇市场数据构建了一个包含446天每天一个矩阵的相关矩阵集合，构建了一个可预测三次和四次多项式期望值的置换不变高斯矩阵模型，并表明该模型预测结果与传统金融相关矩阵模型一致。

    

    我们使用高频外汇市场数据构建了一个包含446天每天一个矩阵的相关矩阵集合。这些矩阵是对称的，减去单位矩阵后对角元素为零。针对该情况，我们构建了一般的置换不变高斯矩阵模型，其中的4个参数是由对称群的表示理论所刻画的。对称的、对角线为零的矩阵的置换不变多项式函数有一个基，由无向无环图标记。使用数据集中矩阵的一般线性和二次置换不变函数的期望值，我们确定了矩阵模型的4个参数。然后，该模型可以预测三次和四次多项式的期望值。通过将这些预测与数据进行比较，我们可以得到强有力的证据，证明置换不变高斯矩阵模型具有良好的整体拟合效果。线性、二次和三次的预测结果都与Chen模型一致，Chen模型是一个标准的金融相关矩阵模型。

    We construct an ensemble of correlation matrices from high-frequency foreign exchange market data, with one matrix for every day for 446 days. The matrices are symmetric and have vanishing diagonal elements after subtracting the identity matrix. For this case, we construct the general permutation invariant Gaussian matrix model, which has 4 parameters characterised using the representation theory of symmetric groups. The permutation invariant polynomial functions of the symmetric, diagonally vanishing matrices have a basis labelled by undirected loop-less graphs. Using the expectation values of the general linear and quadratic permutation invariant functions of the matrices in the dataset, the 4 parameters of the matrix model are determined. The model then predicts the expectation values of the cubic and quartic polynomials. These predictions are compared to the data to give strong evidence for a good overall fit of the permutation invariant Gaussian matrix model. The linear, quadratic
    
[^2]: 欧洲央行利率的国际溢出效应：货币政策和信息效应

    International Spillovers of ECB Interest Rates: Monetary Policy & Information Effects. (arXiv:2306.04562v1 [econ.GN])

    [http://arxiv.org/abs/2306.04562](http://arxiv.org/abs/2306.04562)

    本文研究了欧洲央行利率的国际溢出效应，并证明忽略信息效应会导致其国际溢出失真。作者通过将纯货币政策冲击与信息效应区分开来，发现其国际产量、汇率和股票指数的溢出效应比标准策略高2到3倍。

    

    本文展示了无视欧洲央行货币政策决策公告周围的信息效应会导致其国际溢出失真。使用23个经济体的数据，包括新兴市场和发达市场，作者展示了遵循将纯货币政策冲击与信息效应区分开来的识别策略，将引起的国际产量、汇率和股票指数的溢出效应，其数量级比遵循标准高频率识别策略引起的溢出效应高2到3倍。这种偏差是由于纯货币政策和信息效应在国际溢出方面有直观相反的作用。对于一系列稳健性检验，如子样本“接近”和“远离”国家、新兴市场和发达市场、使用本地投影技术和控制“信息效应”的替代方法都得到了结果。作者认为这种偏差可能导致了...

    This paper shows that disregarding the information effects around the European Central Bank monetary policy decision announcements biases its international spillovers. Using data from 23 economies, both Emerging and Advanced, I show that following an identification strategy that disentangles pure monetary policy shocks from information effects lead to international spillovers on industrial production, exchange rates and equity indexes which are between 2 to 3 times larger in magnitude than those arising from following the standard high frequency identification strategy. This bias is driven by pure monetary policy and information effects having intuitively opposite international spillovers. Results are present for a battery of robustness checks: for a sub-sample of ``close'' and ``further away'' countries, for both Emerging and Advanced economies, using local projection techniques and for alternative methods that control for ``information effects''. I argue that this biases may have led
    
[^3]: 预先阻止授权的Chevron校准

    Calibrating Chevron for Preemption. (arXiv:2306.04463v1 [econ.GN])

    [http://arxiv.org/abs/2306.04463](http://arxiv.org/abs/2306.04463)

    本文探讨了最高法院在Chevron决定下对机构预防措施的解释应该遵从哪些规则，对传统州权领域内的国会意图采取高度重视。

    

    自划时代的Chevron决定以来，几乎已经过去了将近三十年，最高法院尚未阐明该案的机构法解释原则与我们这个时代最具争议的联邦主义问题之一——州法规管辖权——之间的关系。法院应该在Chevron下遵从阻止机构的解释，还是预先防止的联邦主义影响要求提出一种不太顺从的解决方案？评论员提供了无数可能的解决方案，但迄今为止，法院已经抵制了所有这些解决方案。本文对这场辩论作出了两个贡献。首先，通过对法院最近的机构阻止决定进行详细分析，追踪其对各种提出的规则犹豫不决的态度，并体现出对传统州权领域的国会意图的高度重视。法院认识到，授权阻止的国会意图在每个案件中都有所不同，因此一直犹豫采用全面的规则。

    Now almost three decades since its seminal Chevron decision, the Supreme Court has yet to articulate how that case's doctrine of deference to agency statutory interpretations relates to one of the most compelling federalism issues of our time: regulatory preemption of state law. Should courts defer to preemptive agency interpretations under Chevron, or do preemption's federalism implications demand a less deferential approach? Commentators have provided no shortage of possible solutions, but thus far the Court has resisted all of them.  This Article makes two contributions to the debate. First, through a detailed analysis of the Court's recent agency-preemption decisions, I trace its hesitancy to adopt any of the various proposed rules to its high regard for congressional intent where areas of traditional state sovereignty are at risk. Recognizing that congressional intent to delegate preemptive authority varies from case to case, the Court has hesitated to adopt an across-the-board ru
    
[^4]: 最高法院障碍物抢先权的实证研究

    An Empirical Study of Obstacle Preemption in the Supreme Court. (arXiv:2306.04462v1 [econ.GN])

    [http://arxiv.org/abs/2306.04462](http://arxiv.org/abs/2306.04462)

    本文实证分析了最高法院障碍物抢先权的决定，并揭示了令人惊讶的反障碍物抢先权联盟的形成，该联盟由托马斯大法官逐渐与法院自由党派组成的五个法官投票集团。

    

    最高法院联邦抢先权决定是出了名的不可预测。传统的左右阵营投票在竞争性意识形态的拉动下瓦解了。可预测投票集团的瓦解使得最受联邦抢先权影响的商业利益方无法确定潜在第三方受害责任的范围，甚至不确定未来的索赔会适用州法还是联邦法。本文对最近十五年法院判决的实证分析揭示了法院在障碍物抢先权案件中独特的投票联盟。一个令人惊讶的反障碍物抢先权联盟正在形成，随着托马斯大法官逐渐与法院自由党派站在一起，形成一个反对障碍物抢先权的五个法官投票集团。

    The Supreme Court's federal preemption decisions are notoriously unpredictable. Traditional left-right voting alignments break down in the face of competing ideological pulls. The breakdown of predictable voting blocs leaves the business interests most affected by federal preemption uncertain of the scope of potential liability to injured third parties and unsure even of whether state or federal law will be applied to future claims.  This empirical analysis of the Court's decisions over the last fifteen years sheds light on the Court's unique voting alignments in obstacle preemption cases. A surprising anti-obstacle preemption coalition is forming as Justice Thomas gradually positions himself alongside the Court's liberals to form a five-justice voting bloc opposing obstacle preemption.
    
[^5]: 量子退火中的动态规划：解决RBC模型

    Dynamic Programming on a Quantum Annealer: Solving the RBC Model. (arXiv:2306.04285v1 [econ.GN])

    [http://arxiv.org/abs/2306.04285](http://arxiv.org/abs/2306.04285)

    这篇论文介绍了在量子退火器上使用动态规划解决经济模型的方法，并取得了数量级的加速，这对于解决更具挑战性的经济问题具有潜在价值。

    

    我们介绍了一种新方法，使用量子退火器解决动态规划问题，例如许多经济模型中的问题。量子退火器是一种专门用于组合优化的设备，它尝试通过在所有状态的量子叠加态中开始，以毫秒为单位生成候选的全局解，而不考虑问题规模。使用现有的量子硬件，我们在解决实际商业周期模型方面取得了数量级的加速，优于文献中的基准。我们还详细介绍了量子退火，并讨论了它在解决更具挑战性的经济问题方面的潜在用途。

    We introduce a novel approach to solving dynamic programming problems, such as those in many economic models, on a quantum annealer, a specialized device that performs combinatorial optimization. Quantum annealers attempt to solve an NP-hard problem by starting in a quantum superposition of all states and generating candidate global solutions in milliseconds, irrespective of problem size. Using existing quantum hardware, we achieve an order-of-magnitude speed-up in solving the real business cycle model over benchmarks in the literature. We also provide a detailed introduction to quantum annealing and discuss its potential use for more challenging economic problems.
    
[^6]: 巴舍利尔ESG资产定价市场模型

    Bachelier's Market Model for ESG Asset Pricing. (arXiv:2306.04158v1 [q-fin.MF])

    [http://arxiv.org/abs/2306.04158](http://arxiv.org/abs/2306.04158)

    本论文的创新点在于提出了一种基于巴舍利尔市场模型的ESG资产定价模型，并且将其应用于期权定价估值中。

    

    环境、社会和治理（ESG）金融是现代金融和投资的基石，通过将投资表现的另一维度——投资的ESG得分纳入经典的回报风险投资观念中，改变了投资的观念。我们定义了ESG价格过程，并将其集成到巴舍利尔市场模型的离散和连续时间的扩展中，实现了期权定价的估值。

    Environmental, Social, and Governance (ESG) finance is a cornerstone of modern finance and investment, as it changes the classical return-risk view of investment by incorporating an additional dimension of investment performance: the ESG score of the investment. We define the ESG price process and integrate it into an extension of Bachelier's market model in both discrete and continuous time, enabling option pricing valuation.
    
[^7]: 资源采集的可持续性标准隐含外部性定价

    Sustainability criterion implied externality pricing for resource extraction. (arXiv:2306.04065v1 [econ.GN])

    [http://arxiv.org/abs/2306.04065](http://arxiv.org/abs/2306.04065)

    该文构建了动态模型，应用可持续性标准隐含外部性定价，得出最优采伐进度，并揭示外部性定价对采伐进度的影响。

    

    该文构建了一个动态模型，将外部性引入到Hartwick和Van Long（2020）的内生贴现设置中，并询问这对具有恒定消费的自然资源最优采取的影响。结果表明，Hotelling和Hartwick规则的修改形式仍然适用，其中价格的外部性部分是瞬时用户成本和交叉价格弹性的特定函数。它证明了剩余自然储备的外部性调整边际用户成本等于投资于人造再生资本的已开采资源的边际用户成本。这导致了具有直观经济解释的离散形式，解释了外部性定价对最优采伐进度的逐步影响。

    A dynamic model is constructed that generalises the Hartwick and Van Long (2020) endogenous discounting setup by introducing externalities and asks what implications this has for optimal natural resource extraction with constant consumption. It is shown that a modified form of the Hotelling and Hartwick rule holds in which the externality component of price is a specific function of the instantaneous user costs and cross price elasticities. It is demonstrated that the externality adjusted marginal user cost of remaining natural reserves is equal to the marginal user cost of extracted resources invested in human-made reproducible capital. This lends itself to a discrete form with a readily intuitive economic interpretation that illuminates the stepwise impact of externality pricing on optimal extraction schedules.
    
[^8]: ChatGPT信息的图神经网络用于股票价格预测

    ChatGPT Informed Graph Neural Network for Stock Movement Prediction. (arXiv:2306.03763v1 [q-fin.ST])

    [http://arxiv.org/abs/2306.03763](http://arxiv.org/abs/2306.03763)

    该研究介绍了一种新的框架，利用ChatGPT技术增强图神经网络，能够从财经新闻中提取出不断变化的网络结构，并用于股票价格预测，获得了超过基于深度学习的最新基准的表现，提示了ChatGPT在文本推断和金融预测方面的潜力。

    

    ChatGPT已在各种自然语言处理（NLP）任务中展示了出色的能力。然而，它从时间文本数据（尤其是财经新闻）推断动态网络结构的潜力仍是一个未开发的领域。在这项研究中，我们介绍了一个新的框架，利用ChatGPT的图推断能力来增强图神经网络（GNN）。我们的框架巧妙地从文本数据中提取出不断变化的网络结构，并将这些网络结构融合到图神经网络中，进行后续的预测任务。股票价格预测的实验结果表明，我们的模型始终优于基于深度学习的最新基准。此外，基于我们模型的产出构建的组合展示出更高的年化累计回报、更低的波动性和最大回撤。这种卓越表现突显了ChatGPT用于基于文本的网络推断和金融预测应用的潜力。

    ChatGPT has demonstrated remarkable capabilities across various natural language processing (NLP) tasks. However, its potential for inferring dynamic network structures from temporal textual data, specifically financial news, remains an unexplored frontier. In this research, we introduce a novel framework that leverages ChatGPT's graph inference capabilities to enhance Graph Neural Networks (GNN). Our framework adeptly extracts evolving network structures from textual data, and incorporates these networks into graph neural networks for subsequent predictive tasks. The experimental results from stock movement forecasting indicate our model has consistently outperformed the state-of-the-art Deep Learning-based benchmarks. Furthermore, the portfolios constructed based on our model's outputs demonstrate higher annualized cumulative returns, alongside reduced volatility and maximum drawdown. This superior performance highlights the potential of ChatGPT for text-based network inferences and 
    
[^9]: 中国楼宇建筑面积数据集与学习流程的构建

    Building Floorspace in China: A Dataset and Learning Pipeline. (arXiv:2303.02230v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2303.02230](http://arxiv.org/abs/2303.02230)

    本文构建了中国40个主要城市楼宇建筑面积数据集并利用多任务对象分割器方法学习了建筑物占地面积和高度，为城市规划提供数据支持。

    

    本文提供了中国40个主要城市楼宇建筑面积（建筑物占地面积和高度）测量的第一个里程碑。我们利用多任务对象分割器方法在同一框架中学习建筑物占地面积和高度，利用Sentinel-1和-2卫星图像作为主要数据源。本文提供了详细的数据集构建和学习流程说明。

    This paper provides a first milestone in measuring the floorspace of buildings (that is, building footprint and height) for 40 major Chinese cities. The intent is to maximize city coverage and, eventually provide longitudinal data. Doing so requires building on imagery that is of a medium-fine-grained granularity, as larger cross sections of cities and longer time series for them are only available in such format. We use a multi-task object segmenter approach to learn the building footprint and height in the same framework in parallel: (1) we determine the surface area is covered by any buildings (the square footage of occupied land); (2) we determine floorspace from multi-image representations of buildings from various angles to determine the height of buildings. We use Sentinel-1 and -2 satellite images as our main data source. The benefits of these data are their large cross-sectional and longitudinal scope plus their unrestricted accessibility. We provide a detailed description of 
    
[^10]: 应用于模因股票的时变制度检验中熵的方差

    Variance of entropy for testing time-varying regimes with an application to meme stocks. (arXiv:2211.05415v3 [q-fin.ST] UPDATED)

    [http://arxiv.org/abs/2211.05415](http://arxiv.org/abs/2211.05415)

    本文提出了一种假设检验程序，以检验时间序列Shannon熵的恒定零假设，对立的是两个相邻时间段之间熵的显著变化的备择假设，并找到了Shannon熵估计量的方差的无偏估计。

    

    Shannon熵是用于衡量许多领域（从物理学和金融到医学和生物学）时间序列随机程度的最常用度量。现实世界中的系统普遍非平稳，熵值在时间上不是恒定的。本文旨在提出一种假设检验程序，以检验时间序列Shannon熵的恒定零假设，对立的是两个相邻时间段之间熵的显著变化的备择假设。为此，我们找到了Shannon熵估计量的方差的无偏估计，直到n为样本大小的高阶项O(n−4)。为了表征估计量的方差，我们首先得到二项分布和多项分布的中心矩的显式公式，它们描述了Shannon熵的分布。其次，我们找到了用于估计时变Shannon熵的滚动窗口的最佳长度。

    Shannon entropy is the most common metric to measure the degree of randomness of time series in many fields, ranging from physics and finance to medicine and biology. Real-world systems may be in general non stationary, with an entropy value that is not constant in time. The goal of this paper is to propose a hypothesis testing procedure to test the null hypothesis of constant Shannon entropy for time series, against the alternative of a significant variation of the entropy between two subsequent periods. To this end, we find an unbiased approximation of the variance of the Shannon entropy's estimator, up to the order O(n^(-4)) with n the sample size. In order to characterize the variance of the estimator, we first obtain the explicit formulas of the central moments for both the binomial and the multinomial distributions, which describe the distribution of the Shannon entropy. Second, we find the optimal length of the rolling window used for estimating the time-varying Shannon entropy 
    
[^11]: 粗糙指数及其无模型估计

    The roughness exponent and its model-free estimation. (arXiv:2111.10301v4 [math.ST] UPDATED)

    [http://arxiv.org/abs/2111.10301](http://arxiv.org/abs/2111.10301)

    本文提出了对于函数$x$，基于Faber-Schauder系数的温和条件，无需对其动态作出任何假设即可一致地估计其粗糙指数，并给出了推广方法。

    

    本文基于路径随机微积分，定义了一个连续的实函数$x$具有粗糙指数$R$，如果$x$的$p$阶变差在$p>1/R$时趋向于零，而在$p<1/R$时趋向于无穷大。对于许多随机过程的样本路径，例如分数布朗运动，其粗糙指数存在并等于标准赫斯特参数。在我们的主要结果中，我们提供了在Faber-Schauder系数上的一个温和条件，使得粗糙指数存在，并且给出为经典Gladyshev估计$\widehat{R_n}(x)$的极限。这个结果可以被视为Gladyshev估计器在完全无模型设置中的强一致性结果，因为在函数$x$的可能动态方面没有任何假设。尽管如此，我们的证明具有概率性，依赖于隐藏在$x$的Faber-Schauder展开中的一个鞅。由于Gladyshev估计器不是尺度不变的，我们构造了若干量尺度不变的有用推广。

    Motivated by pathwise stochastic calculus, we say that a continuous real-valued function $x$ admits the roughness exponent $R$ if the $p^{\text{th}}$ variation of $x$ converges to zero if $p>1/R$ and to infinity if $p<1/R$. For the sample paths of many stochastic processes, such as fractional Brownian motion, the roughness exponent exists and equals the standard Hurst parameter. In our main result, we provide a mild condition on the Faber--Schauder coefficients of $x$ under which the roughness exponent exists and is given as the limit of the classical Gladyshev estimates $\widehat R_n(x)$. This result can be viewed as a strong consistency result for the Gladyshev estimators in an entirely model-free setting, because no assumption whatsoever is made on the possible dynamics of the function $x$. Nonetheless, our proof is probabilistic and relies on a martingale that is hidden in the Faber--Schauder expansion of $x$. Since the Gladyshev estimators are not scale-invariant, we construct sev
    

