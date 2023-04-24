# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Can Perturbations Help Reduce Investment Risks? Risk-Aware Stock Recommendation via Split Variational Adversarial Training.](http://arxiv.org/abs/2304.11043) | 本文提出了一种基于分离变分对抗训练的风险感知型股票推荐方法，通过对抗性扰动提高模型对于风险的感知能力，通过变分扰动生成器模拟不同的风险因素并生成代表性的风险指标对抗样本。在真实股票数据上进行的实验表明该方法有效降低了投资风险同时保持高预期收益。 |
| [^2] | [Invariance properties of maximal extractable value.](http://arxiv.org/abs/2304.11010) | 本文研究了在基于区块链的去中心化交易中的最大可提取价值（MEV），并证明了当区块排序机制和区块时间分布发生变化时，该MEV是不变的，这能为设计区块链协议提供指导。 |
| [^3] | [An extended Merton problem with relaxed benchmark tracking.](http://arxiv.org/abs/2304.10802) | 本文在Merton问题中增加基准跟踪，提出了一种放松的跟踪公式，并采用反射辅助状态过程，通过双重转换和概率表示，得到了等效的随机控制问题，并且可以明确地解决。通过这种方法，我们可以清晰地了解资产的组成和绩效。 |
| [^4] | [How 'one-size-fits-all' public works contract does it better? An assessment of infrastructure provision in Italy.](http://arxiv.org/abs/2304.10776) | “一刀切”的公共工程承包策略并不能取得很好的效果，而选择“设计和建造合同”则能够显著提高公共工程执行的绩效并降低交易成本，这对政策有很大的启示。 |
| [^5] | [Multi-Modal Deep Learning for Credit Rating Prediction Using Text and Numerical Data Streams.](http://arxiv.org/abs/2304.10740) | 本文研究了基于多模态的深度学习融合技术在信用评级预测中的应用，通过比较不同融合策略和深度学习模型的组合，证明了一个基于CNN的多模态模型通过两种融合策略优于其他多模态技术，同时在比较简单和复杂的模型中发现，更复杂的模型并不一定表现更好。 |
| [^6] | [The quality of school track assignment decisions by teachers.](http://arxiv.org/abs/2304.10636) | 本文研究了荷兰中学阶段分班决策的质量，发现教师的初始分班决策对于大多数学生来说太低了。 |
| [^7] | [Conditional Generative Models for Learning Stochastic Processes.](http://arxiv.org/abs/2304.10382) | 提出了一种称为 C-qGAN 的框架，利用量子电路结构实现了有效的状态准备过程，可以利用该方法加速蒙特卡罗分析等算法，并将其应用于亚式期权衍生品定价的任务中。 |
| [^8] | [The Economic Effect of Gaining a New Qualification Later in Life.](http://arxiv.org/abs/2304.01490) | 本研究通过机器学习方法分析晚年完成学位与经济回报之间的因果效应，发现获得新资格将带来每年超过3000澳元的经济回报。 |
| [^9] | [Efficient and Accurate Calibration to FX Market Skew with Fully Parameterized Local Volatility Model.](http://arxiv.org/abs/2211.14431) | 研究了一种全参数化的本地波动率模型，可高效准确地校准到外汇市场的偏斜波动率，提供可靠的异型期权价格。 |
| [^10] | [Multidimensional Economic Complexity and Inclusive Green Growth.](http://arxiv.org/abs/2209.08382) | 论文研究结合贸易数据、专利申请和研究出版物的数据，建立了能够解释国际包容性绿色增长差异的经济复杂度模型，并发现高综合评分国家更容易实现低排放强度的绿色增长。 |

# 详细

[^1]: 扰动有助于降低投资风险吗？ 基于分离变分对抗训练的风险感知型股票推荐方法

    Can Perturbations Help Reduce Investment Risks? Risk-Aware Stock Recommendation via Split Variational Adversarial Training. (arXiv:2304.11043v1 [q-fin.RM])

    [http://arxiv.org/abs/2304.11043](http://arxiv.org/abs/2304.11043)

    本文提出了一种基于分离变分对抗训练的风险感知型股票推荐方法，通过对抗性扰动提高模型对于风险的感知能力，通过变分扰动生成器模拟不同的风险因素并生成代表性的风险指标对抗样本。在真实股票数据上进行的实验表明该方法有效降低了投资风险同时保持高预期收益。

    

    在股票市场，成功的投资需要在利润和风险之间取得良好的平衡。最近，在量化投资中广泛研究了股票推荐，以为投资者选择具有更高收益率的股票。尽管在获利方面取得了成功，但大多数现有的推荐方法仍然在风险控制方面较弱，这可能导致实际股票投资中难以承受的亏损。为了有效降低风险，我们从对抗性扰动中获得启示，并提出了一种新的基于分离变分对抗训练（SVAT）框架的风险感知型股票推荐方法。本质上，SVAT鼓励模型对风险股票样本的对抗性扰动敏感，并通过学习扰动来增强模型的风险意识。为了生成代表性的风险指标对抗样本，我们设计了一个变分扰动生成器来模拟不同的风险因素。特别地，变分结构使我们的方法能够捕捉难以明确量化和建模的各种风险因素。在真实股票数据上的综合实验表明，SVAT在降低投资风险的同时保持高预期收益上非常有效。

    In the stock market, a successful investment requires a good balance between profits and risks. Recently, stock recommendation has been widely studied in quantitative investment to select stocks with higher return ratios for investors. Despite the success in making profits, most existing recommendation approaches are still weak in risk control, which may lead to intolerable paper losses in practical stock investing. To effectively reduce risks, we draw inspiration from adversarial perturbations and propose a novel Split Variational Adversarial Training (SVAT) framework for risk-aware stock recommendation. Essentially, SVAT encourages the model to be sensitive to adversarial perturbations of risky stock examples and enhances the model's risk awareness by learning from perturbations. To generate representative adversarial examples as risk indicators, we devise a variational perturbation generator to model diverse risk factors. Particularly, the variational architecture enables our method
    
[^2]: 最大可提取价值的不变性质

    Invariance properties of maximal extractable value. (arXiv:2304.11010v1 [q-fin.MF])

    [http://arxiv.org/abs/2304.11010](http://arxiv.org/abs/2304.11010)

    本文研究了在基于区块链的去中心化交易中的最大可提取价值（MEV），并证明了当区块排序机制和区块时间分布发生变化时，该MEV是不变的，这能为设计区块链协议提供指导。

    

    我们提出了一种形式化推理去研究基于区块链的去中心化交易，并给出了一种特定形式的最大可提取价值（MEV）的表达式，该表达式代表从链上流动性中可提取的总套利机会。我们使用这种形式化推理来证明，在具有确定性区块时间且其流动性池满足实践中满足的一些自然特性的区块链中，这种MEV在区块排序机制和区块时间分布发生变化时是不变的。我们通过将MEV表征为一种特别简单的套利策略获得的利润，证明了这一结果当无人竞争时的结果可以为设计区块链协议提供指导，防止设计会改变排序机制或缩短区块时间以增加交易机会。

    We develop a formalism for reasoning about trading on decentralized exchanges on blockchains and a formulation of a particular form of maximal extractable value (MEV) that represents the total arbitrage opportunity extractable from on-chain liquidity. We use this formalism to prove that for blockchains with deterministic block times whose liquidity pools satisfy some natural properties that are satisfied by pools in practice, this form of MEV is invariant under changes to the ordering mechanism of the blockchain and distribution of block times. We do this by characterizing the MEV as the profit of a particularly simple arbitrage strategy when left uncontested. These results can inform design of blockchain protocols by ruling out designs aiming to increase trading opportunity by changing the ordering mechanism or shortening block times.
    
[^3]: 一种拓展的Merton问题解决了放宽基准跟踪的难题

    An extended Merton problem with relaxed benchmark tracking. (arXiv:2304.10802v1 [math.OC])

    [http://arxiv.org/abs/2304.10802](http://arxiv.org/abs/2304.10802)

    本文在Merton问题中增加基准跟踪，提出了一种放松的跟踪公式，并采用反射辅助状态过程，通过双重转换和概率表示，得到了等效的随机控制问题，并且可以明确地解决。通过这种方法，我们可以清晰地了解资产的组成和绩效。

    

    本文研究了Merton的最优投资组合和消费问题，其扩展形式包括跟踪由几何布朗运动描述的基准过程。我们考虑一种放松的跟踪公式，即资产过程通过虚拟资本注入表现优于外部基准。基金经理旨在最大化消费的预期效用，减去资本注入成本，后者也可以视为相对于基准的预期最大缺口。通过引入一个具有反射的辅助状态过程，我们通过双重转换和概率表示制定和解决了等效的随机控制问题，其中对偶PDE可以明确地解决。凭借闭式结果的力量，我们可以导出并验证原始控制问题的半解析形式的反馈最优控制，从而使我们能够清晰地了解资产的组成和绩效。

    This paper studies a Merton's optimal portfolio and consumption problem in an extended formulation incorporating the tracking of a benchmark process described by a geometric Brownian motion. We consider a relaxed tracking formulation such that that the wealth process compensated by a fictitious capital injection outperforms the external benchmark at all times. The fund manager aims to maximize the expected utility of consumption deducted by the cost of the capital injection, where the latter term can also be regarded as the expected largest shortfall with reference to the benchmark. By introducing an auxiliary state process with reflection, we formulate and tackle an equivalent stochastic control problem by means of the dual transform and probabilistic representation, where the dual PDE can be solved explicitly. On the strength of the closed-form results, we can derive and verify the feedback optimal control in the semi-analytical form for the primal control problem, allowing us to obs
    
[^4]: “一刀切”的公共工程承包是如何做到更好的？意大利基础设施供应的评估。

    How 'one-size-fits-all' public works contract does it better? An assessment of infrastructure provision in Italy. (arXiv:2304.10776v1 [econ.GN])

    [http://arxiv.org/abs/2304.10776](http://arxiv.org/abs/2304.10776)

    “一刀切”的公共工程承包策略并不能取得很好的效果，而选择“设计和建造合同”则能够显著提高公共工程执行的绩效并降低交易成本，这对政策有很大的启示。

    

    公共基础设施采购是公共和私人投资以及经济和社会资本增长的先决条件。然而，执行低效严重阻碍了基础设施提供和利益交付。公共基础设施采购最敏感的阶段之一是设计，因为它潜在地在执行阶段创建采购人和承包商之间的战略关系，影响合同的成本和持续时间。在本文中，利用非参数前沿和倾向性得分匹配的最新发展，我们评估了意大利公共工程执行的表现。分析提供了有力的证据，表明采购人选择设计和建造合同可以显著提高执行绩效，从而降低交易成本，使承包商可以更好地适应项目的执行。我们的发现具有相当大的政策含义。

    Public infrastructure procurement is crucial as a prerequisite for public and private investments and for economic and social capital growth. However, low performance in execution severely hinders infrastructure provision and benefits delivery. One of the most sensitive phases in public infrastructure procurement is the design because of the strategic relationship that it potentially creates between procurers and contractors in the execution stage, affecting the costs and the duration of the contract. In this paper, using recent developments in non-parametric frontiers and propensity score matching, we evaluate the performance in the execution of public works in Italy. The analysis provides robust evidence of significant improvement of performance where procurers opt for a design and build contracts, which lead to lower transaction costs, allowing contractors to better accommodate the project in the execution. Our findings bear considerable policy implications.
    
[^5]: 基于多模态深度学习的信用评级预测方法研究——以文本和数字数据流为例

    Multi-Modal Deep Learning for Credit Rating Prediction Using Text and Numerical Data Streams. (arXiv:2304.10740v1 [q-fin.GN])

    [http://arxiv.org/abs/2304.10740](http://arxiv.org/abs/2304.10740)

    本文研究了基于多模态的深度学习融合技术在信用评级预测中的应用，通过比较不同融合策略和深度学习模型的组合，证明了一个基于CNN的多模态模型通过两种融合策略优于其他多模态技术，同时在比较简单和复杂的模型中发现，更复杂的模型并不一定表现更好。

    

    了解信用评级分配中哪些因素是重要的可以帮助做出更好的决策。然而，目前文献的重点大多集中在结构化数据上，较少研究非结构化或多模态数据集。本文提出了一种分析结构化和非结构化不同类型数据集的深度学习模型融合的有效架构，以预测公司信用评级标准。在模型中，我们测试了不同的深度学习模型及融合策略的组合，包括CNN，LSTM，GRU和BERT。我们研究了数据融合策略（包括早期和中间融合）以及技术（包括串联和交叉注意）等方面。结果表明，一个基于CNN的多模态模型通过两种融合策略优于其他多模态技术。此外，通过比较简单的架构与更复杂的架构，我们发现，更复杂的模型并不一定能在信用评级预测中发挥更好的性能。

    Knowing which factors are significant in credit rating assignment leads to better decision-making. However, the focus of the literature thus far has been mostly on structured data, and fewer studies have addressed unstructured or multi-modal datasets. In this paper, we present an analysis of the most effective architectures for the fusion of deep learning models for the prediction of company credit rating classes, by using structured and unstructured datasets of different types. In these models, we tested different combinations of fusion strategies with different deep learning models, including CNN, LSTM, GRU, and BERT. We studied data fusion strategies in terms of level (including early and intermediate fusion) and techniques (including concatenation and cross-attention). Our results show that a CNN-based multi-modal model with two fusion strategies outperformed other multi-modal techniques. In addition, by comparing simple architectures with more complex ones, we found that more soph
    
[^6]: 教师作出学校分班决策的质量研究——以荷兰为例

    The quality of school track assignment decisions by teachers. (arXiv:2304.10636v1 [econ.GN])

    [http://arxiv.org/abs/2304.10636](http://arxiv.org/abs/2304.10636)

    本文研究了荷兰中学阶段分班决策的质量，发现教师的初始分班决策对于大多数学生来说太低了。

    

    本文使用回归不连续设计研究了荷兰中学阶段分班决策的质量。在小学六年级，小学教师将每个学生分配到中学阶段的一个学习轨迹上。如果学生在小学教育的标准化结束测试中的得分高于特定的轨迹分数线，教师可以向上修改这个分配决策。通过比较这些分数线两侧的学生，发现在四年后，有50-90％的学生“被困在轨迹中”，意味着他们只有在第一年就开始在高轨迹上的情况下，才能在四年后在高轨迹上。其他（少数）学生则“始终处于低水平”，无论他们最初位于哪个水平，四年后都会一直在低水平上。这些比例适用于在第一年通过得分超过分数线而从低轨迹转到高轨迹的接近分数线的学生。因此，对于大多数这些学生来说，最初（未修改的）分班决策太低了。研究结果表明，教师作出的分班决策质量值得关注。

    We study the quality of secondary school track assignment decisions in the Netherlands, using a regression discontinuity design. In 6th grade, primary school teachers assign each student to a secondary school track. If a student scores above a track-specific cutoff on the standardized end-of-primary education test, the teacher can upwardly revise this assignment. By comparing students just left and right of these cutoffs, we find that between 50-90% of the students are "trapped in track": these students are on the high track after four years, only if they started on the high track in first year. The remaining (minority of) students are "always low": they are always on the low track after four years, independently of where they started. These proportions hold for students near the cutoffs that shift from the low to the high track in first year by scoring above the cutoff. Hence, for a majority of these students the initial (unrevised) track assignment decision is too low. The results re
    
[^7]: 学习随机过程的有条件生成模型

    Conditional Generative Models for Learning Stochastic Processes. (arXiv:2304.10382v1 [quant-ph])

    [http://arxiv.org/abs/2304.10382](http://arxiv.org/abs/2304.10382)

    提出了一种称为 C-qGAN 的框架，利用量子电路结构实现了有效的状态准备过程，可以利用该方法加速蒙特卡罗分析等算法，并将其应用于亚式期权衍生品定价的任务中。

    

    提出了一种学习多模态分布的框架，称为条件量子生成对抗网络（C-qGAN）。神经网络结构严格采用量子电路，因此被证明能够比当前的方法更有效地表示状态准备过程。这种方法有潜力加速蒙特卡罗分析等算法。特别地，在展示了网络在学习任务中的有效性后，将该技术应用于定价亚式期权衍生品，为未来研究其他路径相关期权打下基础。

    A framework to learn a multi-modal distribution is proposed, denoted as the Conditional Quantum Generative Adversarial Network (C-qGAN). The neural network structure is strictly within a quantum circuit and, as a consequence, is shown to represents a more efficient state preparation procedure than current methods. This methodology has the potential to speed-up algorithms, such as Monte Carlo analysis. In particular, after demonstrating the effectiveness of the network in the learning task, the technique is applied to price Asian option derivatives, providing the foundation for further research on other path-dependent options.
    
[^8]: 后期获得新资格的经济效应

    The Economic Effect of Gaining a New Qualification Later in Life. (arXiv:2304.01490v1 [econ.GN])

    [http://arxiv.org/abs/2304.01490](http://arxiv.org/abs/2304.01490)

    本研究通过机器学习方法分析晚年完成学位与经济回报之间的因果效应，发现获得新资格将带来每年超过3000澳元的经济回报。

    

    在OECD国家中，追求晚年教育资格是一个越来越普遍的现象，因为技术变革和自动化继续推动许多职业所需的技能的演变。本文着重考虑晚年完成学位对经济回报的因果影响，其中获取额外教育的动机和能力可能与早年教育不同。我们发现，与那些没有完成额外学习的人相比，完成额外学位将带来每年超过3000澳元（2019年）的经济回报。对于结果，我们使用《澳大利亚家庭收入和劳动力动态调查》的极其丰富且具有代表性的纵向数据。为了充分利用这些数据的复杂性和丰富性，我们使用基于机器学习（ML）的方法来估算因果效应。我们也能够使用ML来发现晚年获得新资格对经济回报的影响来源的异质性。

    Pursuing educational qualifications later in life is an increasingly common phenomenon within OECD countries since technological change and automation continues to drive the evolution of skills needed in many professions. We focus on the causal impacts to economic returns of degrees completed later in life, where motivations and capabilities to acquire additional education may be distinct from education in early years. We find that completing and additional degree leads to more than \$3000 (AUD, 2019) per year compared to those who do not complete additional study. For outcomes, treatment and controls we use the extremely rich and nationally representative longitudinal data from the Household Income and Labour Dynamics Australia survey is used for this work. To take full advantage of the complexity and richness of this data we use a Machine Learning (ML) based methodology to estimate the causal effect. We are also able to use ML to discover sources of heterogeneity in the effects of ga
    
[^9]: 高效准确的全参数本地波动率模型对外汇市场偏斜率的校准

    Efficient and Accurate Calibration to FX Market Skew with Fully Parameterized Local Volatility Model. (arXiv:2211.14431v2 [q-fin.PR] UPDATED)

    [http://arxiv.org/abs/2211.14431](http://arxiv.org/abs/2211.14431)

    研究了一种全参数化的本地波动率模型，可高效准确地校准到外汇市场的偏斜波动率，提供可靠的异型期权价格。

    

    在外汇衍生品市场上交易美式和亚式期权时，银行必须使用复杂的数学模型来计算价格。常常观察到不同的模型对于同一个异型期权会产生不同的价格，这违反了衍生品风险管理的无套利要求。为了解决这个问题，我们研究了一种全参数化本地波动率模型，用于定价美式/亚式期权。当采用网格或蒙特卡罗数值方法实现该模型的时候，可以高效且准确地校准到外汇市场的偏斜波动率。因此，在日常交易活动中，该模型可以提供可靠的异型期权价格。

    When trading American and Asian options in the FX derivatives market, banks must calculate prices using a complex mathematical model. It is often observed that different models produce varying prices for the same exotic option, which violates the non-arbitrage requirement of derivative risk management. To address this issue, we have studied a fully parameterized local volatility model for pricing American/Asian options. This model, when implemented using a grid or Monte-Carlo numerical method, can be efficiently and accurately calibrated to FX market skew volatilities. As a result, the model can provide reliable prices for exotic options during daily trading activities.
    
[^10]: 多维经济复杂度与包容性绿色增长

    Multidimensional Economic Complexity and Inclusive Green Growth. (arXiv:2209.08382v2 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2209.08382](http://arxiv.org/abs/2209.08382)

    论文研究结合贸易数据、专利申请和研究出版物的数据，建立了能够解释国际包容性绿色增长差异的经济复杂度模型，并发现高综合评分国家更容易实现低排放强度的绿色增长。

    

    要实现包容性的绿色增长，国家需要考虑多种经济、社会和环境因素。这些因素通常由从贸易地理中导出的经济复杂度的度量所捕捉，因此缺少创新活动的关键信息。为填补这一差距，我们将贸易数据与专利申请和研究出版物的数据相结合，建立模型，显著且稳健地改善了经济复杂度度量解释国际包容性绿色增长差异的能力。我们发现，基于贸易和专利数据建立的复杂度度量结合在一起能够解释未来的经济增长和收入不平等性，并且在所有三个指标得分高的国家往往表现出更低的排放强度。这些发现说明了贸易、技术和研究的地理位置如何结合起来解释包容性绿色增长。

    To achieve inclusive green growth, countries need to consider a multiplicity of economic, social, and environmental factors. These are often captured by metrics of economic complexity derived from the geography of trade, thus missing key information on innovative activities. To bridge this gap, we combine trade data with data on patent applications and research publications to build models that significantly and robustly improve the ability of economic complexity metrics to explain international variations in inclusive green growth. We show that measures of complexity built on trade and patent data combine to explain future economic growth and income inequality and that countries that score high in all three metrics tend to exhibit lower emission intensities. These findings illustrate how the geography of trade, technology, and research combine to explain inclusive green growth.
    

