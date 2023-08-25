# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Uncertainty Propagation and Dynamic Robust Risk Measures.](http://arxiv.org/abs/2308.12856) | 这篇论文提出了一种用于量化动态环境中不确定性传播的框架。它定义了专门针对有限时间段内的离散随机过程而设计的动态不确定性集合，以捕捉围绕随机过程和模型的不确定性。论文还探讨了导致动态稳健风险度量的一些已知性质的不确定性集合的条件，并发现由$f$-divergences引起的不确定性集合导致较强的时间一致性。 |
| [^2] | [Financial Inclusion and Monetary Policy: A Study on the Relationship between Financial Inclusion and Effectiveness of Monetary Policy in Developing Countries.](http://arxiv.org/abs/2308.12542) | 本研究通过分析发展中国家的实证数据，发现金融包容性对货币政策效果有着复杂的影响，具体而言，ATM数量的增加对货币政策产生负面影响，而银行账户数量的增加对货币政策的影响不显著。此外，贷款利率对通胀有显著影响，而外商直接投资和汇率的影响对通胀不显著。因此，发展中国家的政府应采取措施提高金融包容性水平，以稳定经济中的价格水平。 |
| [^3] | [Procurement in welfare programs: Evidence and implications from WIC infant formula contracts.](http://arxiv.org/abs/2308.12479) | 本研究探讨了政府采购在社会福利计划中对消费者、制造商和政府的影响。我们研究了美国婴儿配方奶粉市场，并发现了赢得竞标的三个重要优势：对非-WIC需求的溢出效应、边际成本降低以及更高的零售价格。 |
| [^4] | [American Stories: A Large-Scale Structured Text Dataset of Historical U.S. Newspapers.](http://arxiv.org/abs/2308.12477) | 这项研究开发了一种新颖的深度学习流水线，用于从历史美国报纸图像中提取完整的文章文本，以解决现有数据集中布局识别和OCR质量的问题。通过构建高效的架构，实现了高扩展性，并创建了高质量的数据集，可用于预训练大型语言模型，并提升对历史英语和历史世界知识的理解。 |
| [^5] | [A Massive Scale Semantic Similarity Dataset of Historical English.](http://arxiv.org/abs/2306.17810) | 本研究利用重新数字化的无版权美国本地报纸文章，构建了一个大规模的跨越了70年的语义相似性数据集，并包含近4亿个正向语义相似性对。 |
| [^6] | [A stochastic control problem arising from relaxed wealth tracking with a monotone benchmark process.](http://arxiv.org/abs/2302.08302) | 本文研究了一种非标准的随机控制问题，旨在最大化消费效用，同时满足底部约束和基准过程。通过引入两个带反射的辅助状态过程，建立了等效的辅助控制问题，得到了关键的对偶价值函数性质，并推导出了一些有趣的经济影响。 |
| [^7] | [Constrained monotone mean-variance problem with random coefficients.](http://arxiv.org/abs/2212.14188) | 本文研究了具有随机系数的受约束单调均值-方差问题和经典均值-方差问题，并通过逆向随机微分方程提供了最优策略和最优值。结果表明这两个问题共享相同的最优投资组合和最优值。 |
| [^8] | [The Interpretability of LSTM Models for Predicting Oil Company Stocks: impacts of correlated features.](http://arxiv.org/abs/2201.00350) | 研究探究了相关特征对用于预测石油公司股票的LSTM模型的可解释性的影响。结果表明，添加与石油股票相关的特征并不会提高LSTM模型的可解释性，因此应谨慎依靠LSTM模型进行股票市场决策。 |
| [^9] | [Optimal Trading with Signals and Stochastic Price Impact.](http://arxiv.org/abs/2101.10053) | 本研究探讨了在具有随机价格冲击的市场中的最优交易策略，并使用奇异摄动方法进行了有效的近似。数值实验进一步展示了随机交易摩擦对最优交易的影响。 |

# 详细

[^1]: 不确定性传播和动态稳健风险度量

    Uncertainty Propagation and Dynamic Robust Risk Measures. (arXiv:2308.12856v1 [q-fin.RM])

    [http://arxiv.org/abs/2308.12856](http://arxiv.org/abs/2308.12856)

    这篇论文提出了一种用于量化动态环境中不确定性传播的框架。它定义了专门针对有限时间段内的离散随机过程而设计的动态不确定性集合，以捕捉围绕随机过程和模型的不确定性。论文还探讨了导致动态稳健风险度量的一些已知性质的不确定性集合的条件，并发现由$f$-divergences引起的不确定性集合导致较强的时间一致性。

    

    我们引入了一个用于量化动态环境中不确定性传播的框架。具体地，我们定义了专门针对有限时间段内的离散随机过程而设计的动态不确定性集合。这些动态不确定性集合捕捉了围绕随机过程和模型的不确定性，包括分布模糊等因素。不确定性集合的例子包括由Wasserstein距离和$f$-divergences引起的不确定性集合。我们进一步将动态稳健风险度量定义为不确定性集合内所有候选风险的上确界。我们以公理化的方式讨论了导致动态稳健风险度量的一些已知性质（如凸性和一致性）的不确定性集合的条件。此外，我们还讨论了导致稳健动态风险度量时间一致性的动态不确定性集合的必要和充分特性。我们发现，由$f$-divergences引起的不确定性集合导致较强的时间一致性。

    We introduce a framework for quantifying propagation of uncertainty arising in a dynamic setting. Specifically, we define dynamic uncertainty sets designed explicitly for discrete stochastic processes over a finite time horizon. These dynamic uncertainty sets capture the uncertainty surrounding stochastic processes and models, accounting for factors such as distributional ambiguity. Examples of uncertainty sets include those induced by the Wasserstein distance and $f$-divergences.  We further define dynamic robust risk measures as the supremum of all candidates' risks within the uncertainty set. In an axiomatic way, we discuss conditions on the uncertainty sets that lead to well-known properties of dynamic robust risk measures, such as convexity and coherence. Furthermore, we discuss the necessary and sufficient properties of dynamic uncertainty sets that lead to time-consistencies of robust dynamic risk measures. We find that uncertainty sets stemming from $f$-divergences lead to stro
    
[^2]: 金融包容性和货币政策：关于金融包容性与发展中国家货币政策效果关系的研究

    Financial Inclusion and Monetary Policy: A Study on the Relationship between Financial Inclusion and Effectiveness of Monetary Policy in Developing Countries. (arXiv:2308.12542v1 [econ.GN])

    [http://arxiv.org/abs/2308.12542](http://arxiv.org/abs/2308.12542)

    本研究通过分析发展中国家的实证数据，发现金融包容性对货币政策效果有着复杂的影响，具体而言，ATM数量的增加对货币政策产生负面影响，而银行账户数量的增加对货币政策的影响不显著。此外，贷款利率对通胀有显著影响，而外商直接投资和汇率的影响对通胀不显著。因此，发展中国家的政府应采取措施提高金融包容性水平，以稳定经济中的价格水平。

    

    本研究分析了金融包容性对发展中国家货币政策效果的影响。通过使用2004年至2020年10个发展中国家的面板数据，研究发现衡量金融包容性的指标——每10万成年人的ATM数量对货币政策有显著负面影响，而另一个金融包容性指标——每10万成年人的银行账户数量对货币政策有积极影响，但统计上不显著。研究还发现外商直接投资、贷款利率和汇率对通胀有积极影响，但只有贷款利率的影响是统计上显著的。因此，这些国家的政府应采取必要措施提高金融包容性水平，以通过降低通胀来稳定价格水平。

    The study analyzed the impact of financial inclusion on the effectiveness of monetary policy in developing countries. By using a panel data set of 10 developing countries during 2004-2020, the study revealed that the financial inclusion measured by the number of ATM per 100,000 adults had a significant negative effect on monetary policy, whereas the other measure of financial inclusion i.e. the number of bank accounts per 100,000 adults had a positive impact on monetary policy, which is not statistically significant. The study also revealed that foreign direct investment (FDI), lending rate and exchange rate had a positive impact on inflation, but only the effect of lending rate is statistically significant. Therefore, the governments of these countries should make necessary drives to increase the level of financial inclusion as it stabilizes the price level by reducing the inflation in the economy.
    
[^3]: 福利计划中的采购: WIC婴儿配方奶粉合同的证据和影响

    Procurement in welfare programs: Evidence and implications from WIC infant formula contracts. (arXiv:2308.12479v1 [econ.GN])

    [http://arxiv.org/abs/2308.12479](http://arxiv.org/abs/2308.12479)

    本研究探讨了政府采购在社会福利计划中对消费者、制造商和政府的影响。我们研究了美国婴儿配方奶粉市场，并发现了赢得竞标的三个重要优势：对非-WIC需求的溢出效应、边际成本降低以及更高的零售价格。

    

    本文研究了政府采购对消费者、制造商和政府在社会福利计划中的影响。我们分析了美国婴儿配方奶粉市场，其中超过一半的销售额由妇女、婴儿和儿童（WIC）计划购买。WIC计划利用一级价格竞标从三家主要奶粉制造商那里获取回扣，赢家将独家为获胜州内的所有WIC消费者提供服务。制造商在提供回扣方面竞争激烈，回扣占批发价格的约85%。为了解释和解开导致这一现象的因素，我们通过引入两个独特的特征，即价格不弹性的WIC消费者和WIC品牌价格的政府监管，模拟了制造商的零售定价竞争。我们的研究结果证实了赢得竞标的三个巨大优势: 对非-WIC需求的显着溢出效应、显著的边际成本降低和更高的零售价格。

    This paper examines the impact of government procurement in social welfare programs on consumers, manufacturers, and the government. We analyze the U.S. infant formula market, where over half of the total sales are purchased by the Women, Infants, and Children (WIC) program. The WIC program utilizes first-price auctions to solicit rebates from the three main formula manufacturers, with the winner exclusively serving all WIC consumers in the winning state. The manufacturers compete aggressively in providing rebates which account for around 85% of the wholesale price. To rationalize and disentangle the factors contributing to this phenomenon, we model manufacturers' retail pricing competition by incorporating two unique features: price inelastic WIC consumers and government regulation on WIC brand prices. Our findings confirm three sizable benefits from winning the auction: a notable spill-over effect on non-WIC demand, a significant marginal cost reduction, and a higher retail price for
    
[^4]: 美国故事：一种基于历史美国报纸的大规模结构化文本数据集

    American Stories: A Large-Scale Structured Text Dataset of Historical U.S. Newspapers. (arXiv:2308.12477v1 [cs.CL])

    [http://arxiv.org/abs/2308.12477](http://arxiv.org/abs/2308.12477)

    这项研究开发了一种新颖的深度学习流水线，用于从历史美国报纸图像中提取完整的文章文本，以解决现有数据集中布局识别和OCR质量的问题。通过构建高效的架构，实现了高扩展性，并创建了高质量的数据集，可用于预训练大型语言模型，并提升对历史英语和历史世界知识的理解。

    

    现有的美国公共领域报纸全文数据集没有识别报纸扫描的复杂布局，结果导致数字化内容对文章、标题、字幕、广告等布局区域的文本进行了混合。光学字符识别（OCR）的质量也可能较低。本研究开发了一种新颖的深度学习流水线，用于从报纸图像中提取完整的文章文本，并将其应用于美国国会图书馆公共领域《慢性美国》集合中的近2000万份扫描。该流水线包括布局检测、可读性分类、自定义OCR和跨多个边界框关联文章文本等步骤。为了实现高扩展性，它采用了专为移动电话设计的高效架构。结果产生的美国故事数据集提供了高质量的数据，可以用于对大型语言模型进行预训练，以实现对历史英语和历史世界知识的更好理解。该数据集还可以添加到...

    Existing full text datasets of U.S. public domain newspapers do not recognize the often complex layouts of newspaper scans, and as a result the digitized content scrambles texts from articles, headlines, captions, advertisements, and other layout regions. OCR quality can also be low. This study develops a novel, deep learning pipeline for extracting full article texts from newspaper images and applies it to the nearly 20 million scans in Library of Congress's public domain Chronicling America collection. The pipeline includes layout detection, legibility classification, custom OCR, and association of article texts spanning multiple bounding boxes. To achieve high scalability, it is built with efficient architectures designed for mobile phones. The resulting American Stories dataset provides high quality data that could be used for pre-training a large language model to achieve better understanding of historical English and historical world knowledge. The dataset could also be added to 
    
[^5]: 一个历史英语的大规模语义相似性数据集

    A Massive Scale Semantic Similarity Dataset of Historical English. (arXiv:2306.17810v1 [cs.CL])

    [http://arxiv.org/abs/2306.17810](http://arxiv.org/abs/2306.17810)

    本研究利用重新数字化的无版权美国本地报纸文章，构建了一个大规模的跨越了70年的语义相似性数据集，并包含近4亿个正向语义相似性对。

    

    各种任务使用在语义相似性数据上训练的语言模型。虽然有多种数据集可捕捉语义相似性，但它们要么是从现代网络数据构建的，要么是由人工标注员在过去十年中创建的相对较小的数据集。本研究利用一种新颖的来源，即重新数字化的无版权美国本地报纸文章，构建了一个大规模的语义相似性数据集，跨越了1920年到1989年的70年，并包含近4亿个正向语义相似性对。在美国本地报纸中，大约一半的文章来自新闻机构的新闻稿，而本地报纸复制了新闻稿的文章，并撰写了自己的标题，这些标题形成了与文章相关的提取性摘要。我们通过利用文档布局和语言理解将文章和标题关联起来。然后，我们使用深度神经方法来检测哪些文章来自相同的基础来源。

    A diversity of tasks use language models trained on semantic similarity data. While there are a variety of datasets that capture semantic similarity, they are either constructed from modern web data or are relatively small datasets created in the past decade by human annotators. This study utilizes a novel source, newly digitized articles from off-copyright, local U.S. newspapers, to assemble a massive-scale semantic similarity dataset spanning 70 years from 1920 to 1989 and containing nearly 400M positive semantic similarity pairs. Historically, around half of articles in U.S. local newspapers came from newswires like the Associated Press. While local papers reproduced articles from the newswire, they wrote their own headlines, which form abstractive summaries of the associated articles. We associate articles and their headlines by exploiting document layouts and language understanding. We then use deep neural methods to detect which articles are from the same underlying source, in th
    
[^6]: 起源于带有单调基准过程的松弛财富跟踪的随机控制问题

    A stochastic control problem arising from relaxed wealth tracking with a monotone benchmark process. (arXiv:2302.08302v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2302.08302](http://arxiv.org/abs/2302.08302)

    本文研究了一种非标准的随机控制问题，旨在最大化消费效用，同时满足底部约束和基准过程。通过引入两个带反射的辅助状态过程，建立了等效的辅助控制问题，得到了关键的对偶价值函数性质，并推导出了一些有趣的经济影响。

    

    本文研究了一种非标准的随机控制问题，其灵感来源于带有非递减基准过程的最优消费与财富跟踪。具体而言，单调基准是通过漂移布朗运动的运行最大值来建模的。我们考虑使用资本注入的松弛跟踪公式，使得注入的资本所补偿的财富在任何时候都优于基准过程。随机控制问题是在动态底部约束下，最大化削减资本注入成本后的消费的期望效用。通过引入两个带反射的辅助状态过程，我们制定了一个等效的辅助控制问题，同时研究了杠杆资本注入和底部约束控制，使其隐匿起来。为了解决带有两个Neumann边界条件的HJB方程，我们使用一些新颖的概率技巧建立了对偶PDE的唯一经典解分离形式的存在性。我们的主要贡献是确定了对偶价值函数的一些关键性质，从而使我们能够建立松弛财富跟踪问题的最优解的存在性和唯一性，并得出了一些有趣的经济影响。

    This paper studies a nonstandard stochastic control problem motivated by the optimal consumption with wealth tracking of a non-decreasing benchmark process. In particular, the monotone benchmark is modelled by the running maximum of a drifted Brownian motion. We consider a relaxed tracking formulation using capital injection such that the wealth compensated by the injected capital dominates the benchmark process at all times. The stochastic control problem is to maximize the expected utility on consumption deducted by the cost of the capital injection under the dynamic floor constraint. By introducing two auxiliary state processes with reflections, an equivalent auxiliary control problem is formulated and studied such that the singular control of capital injection and the floor constraint can be hidden. To tackle the HJB equation with two Neumann boundary conditions, we establish the existence of a unique classical solution to the dual PDE in a separation form using some novel probabil
    
[^7]: 具有随机系数的受约束单调均值-方差问题

    Constrained monotone mean-variance problem with random coefficients. (arXiv:2212.14188v2 [q-fin.MF] UPDATED)

    [http://arxiv.org/abs/2212.14188](http://arxiv.org/abs/2212.14188)

    本文研究了具有随机系数的受约束单调均值-方差问题和经典均值-方差问题，并通过逆向随机微分方程提供了最优策略和最优值。结果表明这两个问题共享相同的最优投资组合和最优值。

    

    本文研究了具有随机系数的受约束单调均值-方差（MMV）问题和具有凸锥交易约束的经典均值-方差（MV）问题。我们通过一定的逆向随机微分方程（BSDEs）提供了两个问题的半封闭最优策略和最优值。在注意到这些BSDEs之间的联系后，我们发现这两个问题共享相同的最优投资组合和最优值。这将Shen和Zou的结果（SIAM J. Financial Math.，13（2022），pp. SC99-SC112）从确定性系数推广到随机系数。

    This paper studies the monotone mean-variance (MMV) problem and the classical mean-variance (MV) problem with convex cone trading constraints in a market with random coefficients. We provide semiclosed optimal strategies and optimal values for both problems via certain backward stochastic differential equations (BSDEs). After noting the links between these BSDEs, we find that the two problems share the same optimal portfolio and optimal value. This generalizes the result of Shen and Zou $[$ SIAM J. Financial Math., 13 (2022), pp. SC99-SC112$]$ from deterministic coefficients to random ones.
    
[^8]: LSTM模型用于预测石油公司股票的可解释性：相关特征的影响

    The Interpretability of LSTM Models for Predicting Oil Company Stocks: impacts of correlated features. (arXiv:2201.00350v3 [q-fin.ST] UPDATED)

    [http://arxiv.org/abs/2201.00350](http://arxiv.org/abs/2201.00350)

    研究探究了相关特征对用于预测石油公司股票的LSTM模型的可解释性的影响。结果表明，添加与石油股票相关的特征并不会提高LSTM模型的可解释性，因此应谨慎依靠LSTM模型进行股票市场决策。

    

    石油公司是全球最大的公司之一，由于与黄金、原油和美元相关，其经济指标对全球经济和市场有着巨大的影响。本研究调查了相关特征对用于预测石油公司股票的长短期记忆(LSTM)模型的可解释性的影响。为了实现这一目标，我们设计了标准的LSTM网络，并使用各种相关数据集进行了训练。我们的方法旨在通过考虑影响市场的多个因素，如原油价格、黄金价格和美元，来提高股票价格预测的准确性。结果表明，添加与石油股票相关的特征并不会提高LSTM模型的可解释性。这些发现表明，虽然LSTM模型在预测股票价格方面可能是有效的，但其可解释性可能有限。在仅依靠LSTM模型进行股票市场决策时应格外谨慎。

    Oil companies are among the largest companies in the world whose economic indicators in the global stock market have a great impact on the world economy and market due to their relation to gold, crude oil, and the dollar. This study investigates the impact of correlated features on the interpretability of Long Short-Term Memory (LSTM) models for predicting oil company stocks. To achieve this, we designed a Standard Long Short-Term Memory (LSTM) network and trained it using various correlated datasets. Our approach aims to improve the accuracy of stock price prediction by considering the multiple factors affecting the market, such as crude oil prices, gold prices, and the US dollar. The results demonstrate that adding a feature correlated with oil stocks does not improve the interpretability of LSTM models. These findings suggest that while LSTM models may be effective in predicting stock prices, their interpretability may be limited. Caution should be exercised when relying solely on L
    
[^9]: 信号和随机价格冲击下的最优交易

    Optimal Trading with Signals and Stochastic Price Impact. (arXiv:2101.10053v3 [q-fin.MF] UPDATED)

    [http://arxiv.org/abs/2101.10053](http://arxiv.org/abs/2101.10053)

    本研究探讨了在具有随机价格冲击的市场中的最优交易策略，并使用奇异摄动方法进行了有效的近似。数值实验进一步展示了随机交易摩擦对最优交易的影响。

    

    交易摩擦是随机的，在许多情况下是快速均值回归的。在这里，我们研究了在具有随机价格冲击的市场中如何进行最优交易，并使用奇异摄动方法研究了对应的最优控制问题的近似方法。通过构建次解和超解，我们证明了这些近似是准确的到指定的阶数。最后，我们进行了一些数值实验，以说明随机交易摩擦对最优交易的影响。

    Trading frictions are stochastic. They are, moreover, in many instances fast-mean reverting. Here, we study how to optimally trade in a market with stochastic price impact and study approximations to the resulting optimal control problem using singular perturbation methods. We prove, by constructing sub- and super-solutions, that the approximations are accurate to the specified order. Finally, we perform some numerical experiments to illustrate the effect that stochastic trading frictions have on optimal trading.
    

