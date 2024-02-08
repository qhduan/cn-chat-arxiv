# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Inference for Two-Stage Extremum Estimators](https://arxiv.org/abs/2402.05030) | 本文介绍了一种针对两阶段估计量的模拟逼近方法，能够计算极值估计器的渐近方差和渐近分布函数。该方法适用于各类估计器，并且能解决渐近分布非正态和第二阶段估计器偏差的问题。 |
| [^2] | [Does the Use of Unusual Combinations of Datasets Contribute to Greater Scientific Impact?](https://arxiv.org/abs/2402.05024) | 使用不寻常的数据集组合可以显著贡献于科学和更广泛的影响，并促进科学的进步。 |
| [^3] | [What drives the European carbon market? Macroeconomic factors and forecasts](https://arxiv.org/abs/2402.04828) | 本论文通过使用简单的贝叶斯向量自回归模型以及相关因子来预测欧盟碳市场中的碳价格，提高了预测准确性，并通过扩展研究进一步改进了预测性能。 |
| [^4] | [Hyperparameter Tuning for Causal Inference with Double Machine Learning: A Simulation Study](https://arxiv.org/abs/2402.04674) | 这项研究通过模拟研究评估了在因果推断中使用双机器学习方法的预测性能与因果估计之间关系，并提供了超参数调整和其他实际决策对于DML的因果估计的实证见解。 |
| [^5] | [The Role of Child Gender in the Formation of Parents' Social Networks](https://arxiv.org/abs/2402.04474) | 孩子的性别对父母社交网络的形成具有重要影响，孩子们倾向于与同性朋友交友，这导致了父母之间基于孩子性别的互动。研究发现，如果所有孩子都是同性别，家庭之间的联系将增加约15％，而对于有女孩的家庭影响更为明显。 |
| [^6] | [Healthcare Quality by Specialists under a Mixed Compensation System: an Empirical Analysis](https://arxiv.org/abs/2402.04472) | 这项研究分析了混合补偿制度对专家医疗服务质量的影响，通过实证分析发现，改革导致了MC专家服务的质量下降，表现为出院后再住院风险和死亡率的增加。 |
| [^7] | [Fast Online Changepoint Detection](https://arxiv.org/abs/2402.04433) | 该论文研究了在线变点检测在线性回归模型中的应用，并提出了一种基于CUSUM过程的高加权统计量，以确保及时检测到早期发生的变点。通过使用不同的加权方案构造复合统计量，并使用最大统计量作为标记变点的决策规则，实现了快速检测无论变点位置在哪里。此方法适用于几乎所有经济学、医学和其他应用科学中的时间序列。在蒙特卡罗模拟中验证了方法的有效性。 |
| [^8] | [Meritocracy and Its Discontents: Long-run Effects of Repeated School Admission Reforms](https://arxiv.org/abs/2402.04429) | 该论文通过分析世界上第一次引入全国中央化的优质教育招生制度改革，发现了持久的优质教育与公平之间的权衡。中央化系统相较于分散化系统录取了更多优秀的申请者，长期来看产生了更多顶尖精英官僚，但这也导致了地区接触优质高等教育和职业晋升的不平等。几十年后，优质教育的集中化增加了城市出生的职业精英数量相对于农村出生者。 |
| [^9] | [The Effect of External Debt on Greenhouse Gas Emissions](https://arxiv.org/abs/2206.01840) | 外债对温室气体排放有正面且显著的影响，增加的外债会导致温室气体排放的增加。造成这种影响的可能机制之一是，政府由于增加的债务服务压力，无法有效执行环境法规，或被私营部门所控制。 |
| [^10] | [Wardrop Equilibrium Can Be Boundedly Rational: A New Behavioral Theory of Route Choice.](http://arxiv.org/abs/2304.02500) | 该论文提出了一种新的行为理论来加强Wardrop均衡基础，该理论可以通过有限理性的旅行者玩的路径游戏达到Wardrop均衡和全局稳定。 |
| [^11] | [Multivariate Probabilistic CRPS Learning with an Application to Day-Ahead Electricity Prices.](http://arxiv.org/abs/2303.10019) | 本文提出一种新的多元概率CRPS学习方法，应用于日前电价预测中，相比于统一组合在CRPS方面取得了显著改进。 |
| [^12] | [Semiparametric Estimation of Dynamic Binary Choice Panel Data Models.](http://arxiv.org/abs/2202.12062) | 该论文提出了一种新的方法来分析固定效应和动态二元选择模型的面板数据，通过约束条件和指标函数匹配的方式实现了模型的识别，并提出了具有独立于模型维度的收敛速度的估计器和基于自助法的推断方法。 |

# 详细

[^1]: 两阶段极值估计的推理

    Inference for Two-Stage Extremum Estimators

    [https://arxiv.org/abs/2402.05030](https://arxiv.org/abs/2402.05030)

    本文介绍了一种针对两阶段估计量的模拟逼近方法，能够计算极值估计器的渐近方差和渐近分布函数。该方法适用于各类估计器，并且能解决渐近分布非正态和第二阶段估计器偏差的问题。

    

    我们提出了一种基于模拟的方法，用于近似计算两阶段估计量的渐近方差和渐近分布函数。我们着眼于第二阶段的极值估计器，并考虑了第一阶段的大类估计器，包括极值估计器、高维估计器和其他类型的估计器（例如贝叶斯估计器）。我们适应了第一和第二阶段估计器的渐近分布均为非正态分布的情况。我们还允许第二阶段估计器因第一阶段抽样误差而存在显著偏差。我们引入了一种无偏的插值估计器，并建立了其极限分布。我们的方法适用于复杂模型。与重采样方法不同，我们消除了多次计算插值估计器的需要。蒙特卡洛模拟验证了我们方法在有限样本中的有效性。我们还展示了一个关于同侪效应的实证应用。

    We present a simulation-based approach to approximate the asymptotic variance and asymptotic distribution function of two-stage estimators. We focus on extremum estimators in the second stage and consider a large class of estimators in the first stage. This class includes extremum estimators, high-dimensional estimators, and other types of estimators (e.g., Bayesian estimators). We accommodate scenarios where the asymptotic distributions of both the first- and second-stage estimators are non-normal. We also allow for the second-stage estimator to exhibit a significant bias due to the first-stage sampling error. We introduce a debiased plug-in estimator and establish its limiting distribution. Our method is readily implementable with complex models. Unlike resampling methods, we eliminate the need for multiple computations of the plug-in estimator. Monte Carlo simulations confirm the effectiveness of our approach in finite samples. We present an empirical application with peer effects o
    
[^2]: 使用不寻常的数据集组合是否有助于更大的科学影响力？

    Does the Use of Unusual Combinations of Datasets Contribute to Greater Scientific Impact?

    [https://arxiv.org/abs/2402.05024](https://arxiv.org/abs/2402.05024)

    使用不寻常的数据集组合可以显著贡献于科学和更广泛的影响，并促进科学的进步。

    

    科学数据集在当代数据驱动研究中起着至关重要的作用，它们通过促进新模式和现象的发现，推动科学的进步。这种对实证研究的不断需求引发了关于如何在研究项目中战略性地利用数据以促进科学进步的重要问题。本研究基于重组理论提出了一个假设，即创新性地组合现有知识，包括使用不寻常的数据集组合，可以导致高影响的发现。我们调查了超过30,000篇文章中使用了超过6,000个数据集的社会科学数据库ICPSR中这种非典型数据组合的科学成果。本研究提供了四个重要的见解。首先，组合数据集，特别是那些不常见的组合，对科学和更广泛的影响（例如向一般公众传播）都有显著贡献。

    Scientific datasets play a crucial role in contemporary data-driven research, as they allow for the progress of science by facilitating the discovery of new patterns and phenomena. This mounting demand for empirical research raises important questions on how strategic data utilization in research projects can stimulate scientific advancement. In this study, we examine the hypothesis inspired by the recombination theory, which suggests that innovative combinations of existing knowledge, including the use of unusual combinations of datasets, can lead to high-impact discoveries. We investigate the scientific outcomes of such atypical data combinations in more than 30,000 publications that leverage over 6,000 datasets curated within one of the largest social science databases, ICPSR. This study offers four important insights. First, combining datasets, particularly those infrequently paired, significantly contributes to both scientific and broader impacts (e.g., dissemination to the genera
    
[^3]: 欧洲碳市场的驱动因素是什么？宏观经济因素和预测

    What drives the European carbon market? Macroeconomic factors and forecasts

    [https://arxiv.org/abs/2402.04828](https://arxiv.org/abs/2402.04828)

    本论文通过使用简单的贝叶斯向量自回归模型以及相关因子来预测欧盟碳市场中的碳价格，提高了预测准确性，并通过扩展研究进一步改进了预测性能。

    

    在实现到2050年零排放目标的过程中，采用碳税或发展碳市场等放置碳定价的政策措施被广泛使用。本文探讨了如何在欧盟排放交易体系（EU ETS）内对每月真实碳价格进行点、变动方向和密度预测的问题。我们旨在揭示供给和需求两方面的力量，以提高模型在短期和中期的预测准确性。我们展示了一个简单的贝叶斯向量自回归（BVAR）模型，加上一个或两个能够影响碳价格的预测因子，可以在多个基准预测中提供显著的准确性提升，包括调查预期和数据提供者提供的预测。我们还扩展了对核实排放的研究，并证明在这种情况下，添加随机波动性可以进一步提高单因素BVAR模型的预测性能。

    Putting a price on carbon -- with taxes or developing carbon markets -- is a widely used policy measure to achieve the target of net-zero emissions by 2050. This paper tackles the issue of producing point, direction-of-change, and density forecasts for the monthly real price of carbon within the EU Emissions Trading Scheme (EU ETS). We aim to uncover supply- and demand-side forces that can contribute to improving the prediction accuracy of models at short- and medium-term horizons. We show that a simple Bayesian Vector Autoregressive (BVAR) model, augmented with either one or two factors capturing a set of predictors affecting the price of carbon, provides substantial accuracy gains over a wide set of benchmark forecasts, including survey expectations and forecasts made available by data providers. We extend the study to verified emissions and demonstrate that, in this case, adding stochastic volatility can further improve the forecasting performance of a single-factor BVAR model. We r
    
[^4]: 用于双机器学习的因果推断的超参数调整：一项模拟研究

    Hyperparameter Tuning for Causal Inference with Double Machine Learning: A Simulation Study

    [https://arxiv.org/abs/2402.04674](https://arxiv.org/abs/2402.04674)

    这项研究通过模拟研究评估了在因果推断中使用双机器学习方法的预测性能与因果估计之间关系，并提供了超参数调整和其他实际决策对于DML的因果估计的实证见解。

    

    在预测任务中，适当的超参数调整对于现代机器学习方法的最佳性能至关重要。尽管有大量关于为预测调整机器学习算法的文献，但关于对因果机器学习算法进行调整和在不同算法之间进行选择方面的指导非常有限。本文通过Chernozhukov等人（2018）的双机器学习（DML）方法，从实证角度评估了机器学习方法的预测性能与其产生的因果估计之间的关系。DML依赖于通过将其视为监督学习问题并将其用作插件估计来估计所谓的干扰参数，并利用它们来解决（因果）参数。我们使用2019年大西洋因果推断会议数据挑战的数据进行了广泛的模拟研究。我们提供了关于超参数调整和其他实际决策对于DML的因果估计的作用的实证见解。

    Proper hyperparameter tuning is essential for achieving optimal performance of modern machine learning (ML) methods in predictive tasks. While there is an extensive literature on tuning ML learners for prediction, there is only little guidance available on tuning ML learners for causal machine learning and how to select among different ML learners. In this paper, we empirically assess the relationship between the predictive performance of ML methods and the resulting causal estimation based on the Double Machine Learning (DML) approach by Chernozhukov et al. (2018). DML relies on estimating so-called nuisance parameters by treating them as supervised learning problems and using them as plug-in estimates to solve for the (causal) parameter. We conduct an extensive simulation study using data from the 2019 Atlantic Causal Inference Conference Data Challenge. We provide empirical insights on the role of hyperparameter tuning and other practical decisions for causal estimation with DML. Fi
    
[^5]: 孩子的性别在父母社交网络形成中的作用

    The Role of Child Gender in the Formation of Parents' Social Networks

    [https://arxiv.org/abs/2402.04474](https://arxiv.org/abs/2402.04474)

    孩子的性别对父母社交网络的形成具有重要影响，孩子们倾向于与同性朋友交友，这导致了父母之间基于孩子性别的互动。研究发现，如果所有孩子都是同性别，家庭之间的联系将增加约15％，而对于有女孩的家庭影响更为明显。

    

    社交网络在生活的各个方面起着重要的作用。尽管广泛的研究探索了性别、种族和教育等因素在网络形成中的作用，但一个得到较少关注的维度是孩子的性别。孩子倾向于与同性朋友形成友谊，这可能导致父母之间的互动基于他们孩子的性别。本研究以3-5岁孩子为对象，利用孟加拉农村的丰富数据来研究孩子的性别对父母网络形成的作用。我们估计了一个考虑了孩子性别和其他社会经济因素的网络形成均衡模型。对照分析结果表明，孩子的性别在父母网络结构中起到了重要的作用。具体来说，如果所有孩子都是同性别，家庭之间的联系将增加约15％，对于有女孩的家庭影响更为显著。重要的是，孩子的性别对网络结构的影响十分明显。

    Social networks play an important role in various aspects of life. While extensive research has explored factors such as gender, race, and education in network formation, one dimension that has received less attention is the gender of one's child. Children tend to form friendships with same-gender peers, potentially leading their parents to interact based on their child's gender. Focusing on households with children aged 3-5, we leverage a rich dataset from rural Bangladesh to investigate the role of children's gender in parental network formation. We estimate an equilibrium model of network formation that considers a child's gender alongside other socioeconomic factors. Counterfactual analyses reveal that children's gender significantly shapes parents' network structure. Specifically, if all children share the same gender, households would have approximately 15% more links, with a stronger effect for families having girls. Importantly, the impact of children's gender on network struct
    
[^6]: 专家在混合补偿制度下的医疗质量：一个实证分析

    Healthcare Quality by Specialists under a Mixed Compensation System: an Empirical Analysis

    [https://arxiv.org/abs/2402.04472](https://arxiv.org/abs/2402.04472)

    这项研究分析了混合补偿制度对专家医疗服务质量的影响，通过实证分析发现，改革导致了MC专家服务的质量下降，表现为出院后再住院风险和死亡率的增加。

    

    我们分析了混合补偿（MC）制度对专家医疗服务质量的影响。我们利用加拿大魁北克省在1999年实施的一项改革。政府引入了一种每日津贴与每次临床服务收费降低的支付机制。利用大型的患者/医生数据集，我们估计了一个类似区别于区别方法的多状态多次发生危险模型。我们从模型中计算出质量指标。我们的结果表明，这项改革降低了MC专家服务的质量，表现为出院后再住院风险和死亡率的增加。这些效应在不同的专业领域之间有所差异。

    We analyze the effects of a mixed compensation (MC) scheme for specialists on the quality of their healthcare services. We exploit a reform implemented in Quebec (Canada) in 1999. The government introduced a payment mechanism combining a per diem with a reduced fee per clinical service. Using a large patient/physician panel dataset, we estimate a multi-state multi-spell hazard model analogous to a difference-in-differences approach. We compute quality indicators from our model. Our results suggest that the reform reduced the quality of MC specialist services measured by the risk of re-hospitalization and mortality after discharge. These effects vary across specialties.
    
[^7]: 快速在线变点检测

    Fast Online Changepoint Detection

    [https://arxiv.org/abs/2402.04433](https://arxiv.org/abs/2402.04433)

    该论文研究了在线变点检测在线性回归模型中的应用，并提出了一种基于CUSUM过程的高加权统计量，以确保及时检测到早期发生的变点。通过使用不同的加权方案构造复合统计量，并使用最大统计量作为标记变点的决策规则，实现了快速检测无论变点位置在哪里。此方法适用于几乎所有经济学、医学和其他应用科学中的时间序列。在蒙特卡罗模拟中验证了方法的有效性。

    

    我们研究在线变点检测在线性回归模型的背景下。我们提出了一类基于回归残差的累积和（CUSUM）过程的高加权统计量，这些统计量专门设计用于确保在监测时间段的早期及时检测到变点。我们随后提出了一类复合统计量，使用不同的加权方案构造；标记变点的决策规则基于各个权重中最大的统计量，从而有效地像否决制投票机制一样工作，确保无论变点位置在哪里，都能快速检测到。我们的理论推导基于一种非常普遍的弱相关性形式，因此能够将我们的测试应用于经济学、医学和其他应用科学中遇到的几乎所有时间序列。蒙特卡罗模拟表明，我们的方法能够控制程序级别的I型错误，并且在存在变点时具有较短的检测延迟。

    We study online changepoint detection in the context of a linear regression model. We propose a class of heavily weighted statistics based on the CUSUM process of the regression residuals, which are specifically designed to ensure timely detection of breaks occurring early on during the monitoring horizon. We subsequently propose a class of composite statistics, constructed using different weighing schemes; the decision rule to mark a changepoint is based on the largest statistic across the various weights, thus effectively working like a veto-based voting mechanism, which ensures fast detection irrespective of the location of the changepoint. Our theory is derived under a very general form of weak dependence, thus being able to apply our tests to virtually all time series encountered in economics, medicine, and other applied sciences. Monte Carlo simulations show that our methodologies are able to control the procedure-wise Type I Error, and have short detection delays in the presence
    
[^8]: 优质教育方法与其不满：重复学校招生改革的长期影响

    Meritocracy and Its Discontents: Long-run Effects of Repeated School Admission Reforms

    [https://arxiv.org/abs/2402.04429](https://arxiv.org/abs/2402.04429)

    该论文通过分析世界上第一次引入全国中央化的优质教育招生制度改革，发现了持久的优质教育与公平之间的权衡。中央化系统相较于分散化系统录取了更多优秀的申请者，长期来看产生了更多顶尖精英官僚，但这也导致了地区接触优质高等教育和职业晋升的不平等。几十年后，优质教育的集中化增加了城市出生的职业精英数量相对于农村出生者。

    

    如果精英学院改变他们的入学政策会发生什么？通过分析20世纪初全球首次引入全国中央化的优质招生制度，我们回答了这个问题。我们发现存在持久的优质教育与公平之间的权衡。相较于分散化系统，中央化系统录取了更多优秀的申请者，在长期内产生了更多顶尖精英官僚。然而，这种影响以公平地区接触精英高等教育和职业晋升的代价为代表。几十年后，优质教育的集中化增加了城市出生的职业精英（例如，高收入者）相对于农村出生的数量。

    What happens if selective colleges change their admission policies? We answer this question by analyzing the world's first introduction of nationally centralized meritocratic admissions in the early twentieth century. We find a persistent meritocracy-equity tradeoff. Compared to the decentralized system, the centralized system admitted more high-achieving applicants, producing a greater number of top elite bureaucrats in the long run. However, this impact came at the distributional cost of equal regional access to elite higher education and career advancement. Several decades later, the meritocratic centralization increased the number of urban-born career elites (e.g., top income earners) relative to rural-born ones.
    
[^9]: 外债对温室气体排放的影响

    The Effect of External Debt on Greenhouse Gas Emissions

    [https://arxiv.org/abs/2206.01840](https://arxiv.org/abs/2206.01840)

    外债对温室气体排放有正面且显著的影响，增加的外债会导致温室气体排放的增加。造成这种影响的可能机制之一是，政府由于增加的债务服务压力，无法有效执行环境法规，或被私营部门所控制。

    

    我们在78个新兴市场和发展中国家的面板数据中估计了外债对温室气体排放的因果效应，时间跨度为1990年至2015年。与以往文献不同，我们使用外部工具变量来解决外债与温室气体排放之间的潜在内生性问题。具体而言，我们使用国际流动性冲击作为外债的工具变量。我们发现，处理潜在内生性问题带来了外债对温室气体排放的正面且显著影响：外债增加1个百分点（pp.），平均会导致温室气体排放增加0.4%。一个可能的作用机制是，随着外债的增加，政府难以执行环境法规，因为他们的主要优先事项是扩大税基以支付不断增加的债务服务，或者因为他们被持有债务的私营部门所控制。

    We estimate the causal effect of external debt on greenhouse gas emissions in a panel of 78 emerging market and developing economies over the 1990-2015 period. Unlike previous literature, we use external instruments to address the potential endogeneity in the relationship between external debt and greenhouse gas emissions. Specifically, we use international liquidity shocks as instrumental variables for external debt. We find that dealing with the potential endogeneity problem brings about a positive and statistically significant effect of external debt on greenhouse gas emissions: a 1 percentage point (pp.) rise in external debt causes, on average, a 0.4% increase in greenhouse gas emissions. One possible mechanism of action could be that, as external debt increases, governments are less able to enforce environmental regulations because their main priority is to increase the tax base to pay increasing debt services or because they are captured by the private sector who owns that debt 
    
[^10]: Wardrop均衡可以被有限理性进行边界限制：路径选择的新行为理论。

    Wardrop Equilibrium Can Be Boundedly Rational: A New Behavioral Theory of Route Choice. (arXiv:2304.02500v1 [econ.TH])

    [http://arxiv.org/abs/2304.02500](http://arxiv.org/abs/2304.02500)

    该论文提出了一种新的行为理论来加强Wardrop均衡基础，该理论可以通过有限理性的旅行者玩的路径游戏达到Wardrop均衡和全局稳定。

    

    作为交通科学中最基本的概念之一, Wardrop 均衡一直缺乏相对较强的行为学支撑。为了加强这一基础，必须考虑人类决策过程中的有限理性，如缺乏准确信息、有限的计算能力和次优选择。然而，文献中的这种对行为完美主义的放弃通常伴随着WE的概念修改。在这里，我们展示了放弃完美理性不必导致WE的离开。相反，一种称为累积logit（CULO）的动态模型可以通过有限理性旅行者玩的路径游戏达到WE和全局稳定。我们通过开发一种称为每日（DTD）的动态模型达成了这个结果，该模型模拟旅客根据过去的经验逐渐调整其路径价值，从而得到了选择概率。我们的模型类似于经典的DTD模型，但发生了重大变化：经典模型中采取实数，而CULO模型使用积极的整数。

    As one of the most fundamental concepts in transportation science, Wardrop equilibrium (WE) has always had a relatively weak behavioral underpinning. To strengthen this foundation, one must reckon with bounded rationality in human decision-making processes, such as the lack of accurate information, limited computing power, and sub-optimal choices. This retreat from behavioral perfectionism in the literature, however, was typically accompanied by a conceptual modification of WE. Here we show that giving up perfect rationality need not force a departure from WE. On the contrary, WE can be reached with global stability in a routing game played by boundedly rational travelers. We achieve this result by developing a day-to-day (DTD) dynamical model that mimics how travelers gradually adjust their route valuations, hence choice probabilities, based on past experiences. Our model, called cumulative logit (CULO), resembles the classical DTD models but makes a crucial change: whereas the classi
    
[^11]: 多元概率CRPS学习及其在日前电价预测中的应用

    Multivariate Probabilistic CRPS Learning with an Application to Day-Ahead Electricity Prices. (arXiv:2303.10019v1 [stat.ML])

    [http://arxiv.org/abs/2303.10019](http://arxiv.org/abs/2303.10019)

    本文提出一种新的多元概率CRPS学习方法，应用于日前电价预测中，相比于统一组合在CRPS方面取得了显著改进。

    

    本文提出了一种考虑分位数和协变量依赖关系的多元概率预测的结合方法，并通过平滑过程允许在线学习。通过维数降低和罚函数平滑等两种平滑方法来将标准CRPS学习框架推广到多元维度中。将该方法应用于预测日前电价，相比于统一组合，在CRPS方面取得了显著改进。

    This paper presents a new method for combining (or aggregating or ensembling) multivariate probabilistic forecasts, taking into account dependencies between quantiles and covariates through a smoothing procedure that allows for online learning. Two smoothing methods are discussed: dimensionality reduction using Basis matrices and penalized smoothing. The new online learning algorithm generalizes the standard CRPS learning framework into multivariate dimensions. It is based on Bernstein Online Aggregation (BOA) and yields optimal asymptotic learning properties. We provide an in-depth discussion on possible extensions of the algorithm and several nested cases related to the existing literature on online forecast combination. The methodology is applied to forecasting day-ahead electricity prices, which are 24-dimensional distributional forecasts. The proposed method yields significant improvements over uniform combination in terms of continuous ranked probability score (CRPS). We discuss 
    
[^12]: 动态二元选择面板数据模型的半参数估计

    Semiparametric Estimation of Dynamic Binary Choice Panel Data Models. (arXiv:2202.12062v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2202.12062](http://arxiv.org/abs/2202.12062)

    该论文提出了一种新的方法来分析固定效应和动态二元选择模型的面板数据，通过约束条件和指标函数匹配的方式实现了模型的识别，并提出了具有独立于模型维度的收敛速度的估计器和基于自助法的推断方法。

    

    我们提出了一种用于固定效应和动态（滞后依赖变量）面板数据二元选择模型的半参数分析的新方法。我们考虑的模型与Honore和Kyriazidou（2000）中的随机效用框架相同。我们证明，在对确定性效用过程和误差分布的尾部约束条件下，模型的（点）识别可以在两个步骤中进行，只需要在时间上匹配解释变量的指标函数的值，而不是每个解释变量的值。我们的识别方法提出了一种易于实施的两步最大得分（2SMS）程序--产生估计器，其收敛率与模型维数无关，与Honore和Kyriazidou（2000）的方法相比。然后，我们推导了2SMS程序的渐近性质，并提出了基于自助法的分布逼近方法进行推断。

    We propose a new approach to the semiparametric analysis of panel data binary choice models with fixed effects and dynamics (lagged dependent variables). The model we consider has the same random utility framework as in Honore and Kyriazidou (2000). We demonstrate that, with additional serial dependence conditions on the process of deterministic utility and tail restrictions on the error distribution, the (point) identification of the model can proceed in two steps, and only requires matching the value of an index function of explanatory variables over time, as opposed to that of each explanatory variable. Our identification approach motivates an easily implementable, two-step maximum score (2SMS) procedure -- producing estimators whose rates of convergence, in contrast to Honore and Kyriazidou's (2000) methods, are independent of the model dimension. We then derive the asymptotic properties of the 2SMS procedure and propose bootstrap-based distributional approximations for inference. 
    

