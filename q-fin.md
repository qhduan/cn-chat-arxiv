# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Uncovering Market Disorder and Liquidity Trends Detection.](http://arxiv.org/abs/2310.09273) | 这篇论文提出了一种新的方法来检测订单驱动市场中流动性的显著变化。通过构建市场流动性模型和采用标记的Hawkes过程，能够精确识别流动性的强度变化，并用于优化执行算法、降低市场影响和成本以及作为市场流动性指标。 |
| [^2] | [A generalization of the rational rough Heston approximation.](http://arxiv.org/abs/2310.09181) | 这个论文将理性逼近从粗Heston分数ODE扩展到Mittag-Leffler核情况，并提供了数值证据表明解的收敛性。 |
| [^3] | [Growth, Poverty Trap and Escape.](http://arxiv.org/abs/2310.09098) | 这个论文提出了一个基于随机Solow增长模型的假设，该模型将储蓄比例设定为人均资本的S形函数，并推导出了稳态概率分布。 |
| [^4] | [Mean-field Libor market model and valuation of long term guarantees.](http://arxiv.org/abs/2310.09022) | 该论文展示了多维均场Libor市场模型的解的存在和唯一性，并基于此模型构建了一个能够根据Solvency~II规定计算未来福利的数值资产负债管理模型。通过综合寿险数据进行数值研究，论文提供了对未来福利下限和上限的估计量的启发性假设的数值证据。这些估计量被应用到公开可得到的寿险数据中。 |
| [^5] | [How Does Artificial Intelligence Improve Human Decision-Making? Evidence from the AI-Powered Go Program.](http://arxiv.org/abs/2310.08704) | 本研究通过一个AI-Powered围棋程序（APG）的引入，发现人类从人工智能中学习可以显著提高着法质量，并减少错误数量和大小。年轻选手和接触过人工智能的国家的选手获得了更大的改进，技术较差的选手获得了更高的边际效益。 |
| [^6] | [Can GPT models be Financial Analysts? An Evaluation of ChatGPT and GPT-4 on mock CFA Exams.](http://arxiv.org/abs/2310.08678) | 本研究评估了ChatGPT和GPT-4在金融分析上的能力，发现它们在模拟CFA考试中具有一定的表现，为将来进一步提升大型语言模型在金融推理方面的能力提供了启示。 |
| [^7] | [The Specter (and Spectra) of Miner Extractable Value.](http://arxiv.org/abs/2310.07865) | 本研究提出了矿工可提取价值的MEV成本的理论定义，在多个示例中证明了该定义的实用性，并与对称群的函数“平滑性”相关联。 |
| [^8] | [Human-AI Interactions and Societal Pitfalls.](http://arxiv.org/abs/2309.10448) | 本研究研究了人工智能与人类互动中面临的同质化和偏见问题，提出了改善人工智能与人类互动的解决办法，实现个性化输出而不牺牲生产力。 |
| [^9] | [Startup Acquisitions: Acquihires and Talent Hoarding.](http://arxiv.org/abs/2308.10046) | 该论文提出了一个创业公司收购模型，指出收购会导致低效的 "人才囤积"。研究发现，即使没有竞争效应，收购也可能是垄断行为的结果，导致人才分配低效，并对消费者剩余和被收购员工的工作稳定性产生负面影响。 |
| [^10] | [A Theory of Interactively Coherent Entanglement for Intelligence-Like Particles.](http://arxiv.org/abs/2306.15554) | 本论文从交易量-价格概率波动方程中提取了一个普遍规律，将其应用于复杂量子系统，并提出了互动一致性纠缠理论，解释智能样粒子的行为。 |

# 详细

[^1]: 揭示市场混乱与流动性趋势检测

    Uncovering Market Disorder and Liquidity Trends Detection. (arXiv:2310.09273v1 [q-fin.MF])

    [http://arxiv.org/abs/2310.09273](http://arxiv.org/abs/2310.09273)

    这篇论文提出了一种新的方法来检测订单驱动市场中流动性的显著变化。通过构建市场流动性模型和采用标记的Hawkes过程，能够精确识别流动性的强度变化，并用于优化执行算法、降低市场影响和成本以及作为市场流动性指标。

    

    本文的主要目标是构思和开发一种新的方法，用于检测订单驱动市场中流动性的显著变化。我们研究了一种市场流动性模型，通过使用其限价订单簿数据来动态量化交易资产的流动性水平。所提出的指标有望提升最优执行算法的侵略性，降低市场影响和交易成本，并作为市场做市商的可靠流动性指标。作为我们方法的一部分，我们采用标记的Hawkes过程来建模流动性代理，即通过交易产生的事件。随后，我们的重点在于准确地识别强度发生显著增加或减少的时刻。我们考虑了一个不可观测的双随机泊松过程的最小最大最快检测问题。目标是开发一种停止准则，以最小化鲁森准则的健壮性。

    The primary objective of this paper is to conceive and develop a new methodology to detect notable changes in liquidity within an order-driven market. We study a market liquidity model which allows us to dynamically quantify the level of liquidity of a traded asset using its limit order book data. The proposed metric holds potential for enhancing the aggressiveness of optimal execution algorithms, minimizing market impact and transaction costs, and serving as a reliable indicator of market liquidity for market makers. As part of our approach, we employ Marked Hawkes processes to model trades-through which constitute our liquidity proxy. Subsequently, our focus lies in accurately identifying the moment when a significant increase or decrease in its intensity takes place. We consider the minimax quickest detection problem of unobservable changes in the intensity of a doubly-stochastic Poisson process. The goal is to develop a stopping rule that minimizes the robust Lorden criterion, meas
    
[^2]: 理性粗Heston逼近的一般化

    A generalization of the rational rough Heston approximation. (arXiv:2310.09181v1 [q-fin.CP])

    [http://arxiv.org/abs/2310.09181](http://arxiv.org/abs/2310.09181)

    这个论文将理性逼近从粗Heston分数ODE扩展到Mittag-Leffler核情况，并提供了数值证据表明解的收敛性。

    

    我们将理性逼近从[GR19]中的粗Heston分数ODE扩展到Mittag-Leffler核情况。我们提供了数值证据表明解的收敛性。

    We extend the rational approximation of the solution of the rough Heston fractional ODE in [GR19] to the case of the Mittag-Leffler kernel. We provide numerical evidence of the convergence of the solution.
    
[^3]: 增长，贫困陷阱和逃脱

    Growth, Poverty Trap and Escape. (arXiv:2310.09098v1 [physics.soc-ph])

    [http://arxiv.org/abs/2310.09098](http://arxiv.org/abs/2310.09098)

    这个论文提出了一个基于随机Solow增长模型的假设，该模型将储蓄比例设定为人均资本的S形函数，并推导出了稳态概率分布。

    

    著名的Solow增长模型是经济增长理论中的基本模型，研究资本积累与时间的关系，其中资本存量、劳动力和技术效率是其基本要素。资本被假设为制造设备和材料的形式。模型的两个重要参数是：生产函数产出的储蓄比例$s$和出现在生产函数中的技术效率参数$A$。储蓄比例的产出全部用于生成新的资本，剩下的则被消费。资本存量还会随时间逐渐减少，原因是旧资本的磨损以及劳动力人口的增加。我们提出了一个假设储蓄比例是人均资本$k_p$的S形函数的随机Solow增长模型。我们从解析上推导出稳态概率分布$P(k_p)$并展示了…

    The well-known Solow growth model is the workhorse model of the theory of economic growth, which studies capital accumulation in a model economy as a function of time with capital stock, labour and technology efiiciency as the basic ingredients. The capital is assumed to be in the form of manufacturing equipments and materials. Two important parameters of the model are: the saving fraction $s$ of the output of a production function and the technology efficiency parameter $A$, appearing in the production function. The saved fraction of the output is fully invested in the generation of new capital and the rest is consumed. The capital stock also depreciates as a function of time due to the wearing out of old capital and the increase in the size of the labour population. We propose a stochastic Solow growth model assuming the saving fraction to be a sigmoidal function of the per capita capital $k_p$. We derive analytically the steady state probability distribution $P(k_p)$ and demonstrate
    
[^4]: 均场Libor市场模型和长期保证的估值

    Mean-field Libor market model and valuation of long term guarantees. (arXiv:2310.09022v1 [q-fin.RM])

    [http://arxiv.org/abs/2310.09022](http://arxiv.org/abs/2310.09022)

    该论文展示了多维均场Libor市场模型的解的存在和唯一性，并基于此模型构建了一个能够根据Solvency~II规定计算未来福利的数值资产负债管理模型。通过综合寿险数据进行数值研究，论文提供了对未来福利下限和上限的估计量的启发性假设的数值证据。这些估计量被应用到公开可得到的寿险数据中。

    

    本研究证明了多维均场Libor市场模型（由[7]引入）的解的存在和唯一性。这一模型被用作能够根据Solvency~II规定计算未来福利的数值资产负债管理（ALM）模型的基础。此ALM模型结合了综合寿险数据进行实际的数值研究，从而提供了对启发性假设的数值证据，这些假设可以用来推导未来福利的下限和上限的估计量。这些估计量被应用到公开可得到的寿险数据中。

    Existence and uniqueness of solutions to the multi-dimensional mean-field Libor market model (introduced by [7]) is shown. This is used as the basis for a numerical asset-liability management (ALM) model capable of calculating future discretionary benefits in accordance with Solvency~II regulation. This ALM model is complimented with aggregated life insurance data to perform a realistic numerical study. This yields numerical evidence for heuristic assumptions which allow to derive estimators of lower and upper bounds for future discretionary benefits. These estimators are applied to publicly available life insurance data.
    
[^5]: 人工智能如何提升人类决策能力？来自AI-Powered围棋程序的证据。

    How Does Artificial Intelligence Improve Human Decision-Making? Evidence from the AI-Powered Go Program. (arXiv:2310.08704v1 [econ.GN])

    [http://arxiv.org/abs/2310.08704](http://arxiv.org/abs/2310.08704)

    本研究通过一个AI-Powered围棋程序（APG）的引入，发现人类从人工智能中学习可以显著提高着法质量，并减少错误数量和大小。年轻选手和接触过人工智能的国家的选手获得了更大的改进，技术较差的选手获得了更高的边际效益。

    

    我们研究了人类如何从人工智能中学习，利用了一个突然击败最好的职业选手的AI-Powered围棋程序（APG）的引入。我们比较了职业选手的着法质量和APG在其公开发布前后的卓越解决方案的着法质量。我们分析了749,190个着法，发现了玩家着法质量的显著提升，同时错误数量和大小的减少。这种效果在比赛的初期阶段尤为显著，因为此时不确定性最大。此外，年轻选手和那些处于接触过人工智能的国家的选手获得了更大的改进，这表明了从人工智能中学习可能存在的不平等问题。此外，虽然各个水平的选手都能学习，但技术较差的选手获得了更高的边际效益。这些发现对于寻求在组织中有效采用和利用人工智能的管理者具有重要影响。

    We study how humans learn from AI, exploiting an introduction of an AI-powered Go program (APG) that unexpectedly outperformed the best professional player. We compare the move quality of professional players to that of APG's superior solutions around its public release. Our analysis of 749,190 moves demonstrates significant improvements in players' move quality, accompanied by decreased number and magnitude of errors. The effect is pronounced in the early stages of the game where uncertainty is highest. In addition, younger players and those in AI-exposed countries experience greater improvement, suggesting potential inequality in learning from AI. Further, while players of all levels learn, less skilled players derive higher marginal benefits. These findings have implications for managers seeking to adopt and utilize AI effectively within their organizations.
    
[^6]: GPT模型能成为金融分析师吗？对模拟CFA考试中的ChatGPT和GPT-4进行评估

    Can GPT models be Financial Analysts? An Evaluation of ChatGPT and GPT-4 on mock CFA Exams. (arXiv:2310.08678v1 [cs.CL])

    [http://arxiv.org/abs/2310.08678](http://arxiv.org/abs/2310.08678)

    本研究评估了ChatGPT和GPT-4在金融分析上的能力，发现它们在模拟CFA考试中具有一定的表现，为将来进一步提升大型语言模型在金融推理方面的能力提供了启示。

    

    大型语言模型（LLM）在各种自然语言处理（NLP）任务中展现了出色的性能，通常能与甚至超越最先进的任务特定模型。本研究旨在评估LLM在金融推理能力方面的表现。我们利用特许金融分析师（CFA）考试的模拟题目对ChatGPT和GPT-4在金融分析中进行全面评估，考虑了零样本（ZS）、思路链（CoT）和少样本（FS）场景。我们对模型的性能和局限性进行了深入分析，并评估它们通过CFA考试的可能性。最后，我们提出了提高LLM在金融领域应用性的潜在策略和改进。在这个视角下，我们希望该研究为未来的研究继续通过严格评估来提升LLM在金融推理方面的能力铺平道路。

    Large Language Models (LLMs) have demonstrated remarkable performance on a wide range of Natural Language Processing (NLP) tasks, often matching or even beating state-of-the-art task-specific models. This study aims at assessing the financial reasoning capabilities of LLMs. We leverage mock exam questions of the Chartered Financial Analyst (CFA) Program to conduct a comprehensive evaluation of ChatGPT and GPT-4 in financial analysis, considering Zero-Shot (ZS), Chain-of-Thought (CoT), and Few-Shot (FS) scenarios. We present an in-depth analysis of the models' performance and limitations, and estimate whether they would have a chance at passing the CFA exams. Finally, we outline insights into potential strategies and improvements to enhance the applicability of LLMs in finance. In this perspective, we hope this work paves the way for future studies to continue enhancing LLMs for financial reasoning through rigorous evaluation.
    
[^7]: 矿工可提取价值的威胁（和谱）。

    The Specter (and Spectra) of Miner Extractable Value. (arXiv:2310.07865v1 [math.OC])

    [http://arxiv.org/abs/2310.07865](http://arxiv.org/abs/2310.07865)

    本研究提出了矿工可提取价值的MEV成本的理论定义，在多个示例中证明了该定义的实用性，并与对称群的函数“平滑性”相关联。

    

    矿工可提取价值（MEV）是指交易验证器通过操纵交易顺序可以实现的任何超额价值。在本研究中，我们引入了一个简单的MEV成本的理论定义，证明了一些基本性质，并通过多个示例展示了该定义的实用性。在多种情况下，该定义与函数在对称群上的“平滑性”相关。通过这个定义和一些基本观察，我们回溯了文献中的一些结果。

    Miner extractable value (MEV) refers to any excess value that a transaction validator can realize by manipulating the ordering of transactions. In this work, we introduce a simple theoretical definition of the 'cost of MEV', prove some basic properties, and show that the definition is useful via a number of examples. In a variety of settings, this definition is related to the 'smoothness' of a function over the symmetric group. From this definition and some basic observations, we recover a number of results from the literature.
    
[^8]: 人工智能与人类互动以及社会陷阱

    Human-AI Interactions and Societal Pitfalls. (arXiv:2309.10448v1 [cs.AI])

    [http://arxiv.org/abs/2309.10448](http://arxiv.org/abs/2309.10448)

    本研究研究了人工智能与人类互动中面临的同质化和偏见问题，提出了改善人工智能与人类互动的解决办法，实现个性化输出而不牺牲生产力。

    

    当与生成式人工智能（AI）合作时，用户可能会看到生产力的提升，但AI生成的内容可能不完全符合他们的偏好。为了研究这种影响，我们引入了一个贝叶斯框架，其中异质用户选择与AI共享多少信息，面临输出保真度和通信成本之间的权衡。我们展示了这些个体决策与AI训练之间的相互作用可能导致社会挑战。输出可能变得更加同质化，特别是当AI在AI生成的内容上进行训练时。而任何AI的偏见可能成为社会偏见。解决同质化和偏见问题的办法是改进人工智能与人类的互动，实现个性化输出而不牺牲生产力。

    When working with generative artificial intelligence (AI), users may see productivity gains, but the AI-generated content may not match their preferences exactly. To study this effect, we introduce a Bayesian framework in which heterogeneous users choose how much information to share with the AI, facing a trade-off between output fidelity and communication cost. We show that the interplay between these individual-level decisions and AI training may lead to societal challenges. Outputs may become more homogenized, especially when the AI is trained on AI-generated content. And any AI bias may become societal bias. A solution to the homogenization and bias issues is to improve human-AI interactions, enabling personalized outputs without sacrificing productivity.
    
[^9]: 创业公司收购：人才抢购和人才囤积

    Startup Acquisitions: Acquihires and Talent Hoarding. (arXiv:2308.10046v1 [econ.GN])

    [http://arxiv.org/abs/2308.10046](http://arxiv.org/abs/2308.10046)

    该论文提出了一个创业公司收购模型，指出收购会导致低效的 "人才囤积"。研究发现，即使没有竞争效应，收购也可能是垄断行为的结果，导致人才分配低效，并对消费者剩余和被收购员工的工作稳定性产生负面影响。

    

    我们提出了一个创业公司收购模型，可能导致低效的 "人才囤积"。我们开发了一个有两个竞争公司的模型，这些公司可以收购和整合一个在不同领域运营的创业公司，这种收购改善了收购公司的竞争力。我们表明，即使没有经典的竞争效应，这种收购也可能不是良性的，而是垄断行为的结果，导致人才分配低效。此外，我们还表明，这种人才囤积可能会降低消费者剩余，并导致被收购员工的工作不稳定性增加。

    We present a model of startup acquisitions, which may give rise to inefficient "talent hoarding." We develop a model with two competing firms that can acquire and integrate (or "acquihire") a startup operating in an orthogonal market. Such an acquihire improves the competitiveness of the acquiring firm. We show that even absent the classical competition effects, acquihires need not be benign but can be the result of oligopolistic behavior, leading to an inefficient allocation of talent. Further, we show that such talent hoarding may reduce consumer surplus and lead to more job volatility for acquihired employees.
    
[^10]: 互动一致性纠缠理论

    A Theory of Interactively Coherent Entanglement for Intelligence-Like Particles. (arXiv:2306.15554v1 [q-fin.GN])

    [http://arxiv.org/abs/2306.15554](http://arxiv.org/abs/2306.15554)

    本论文从交易量-价格概率波动方程中提取了一个普遍规律，将其应用于复杂量子系统，并提出了互动一致性纠缠理论，解释智能样粒子的行为。

    

    复杂适应性学习是智能的，并在生命和非生命复杂系统中发挥作用。一个复杂系统由许多相互作用的个体或单元组成，它们在相互作用中显示出隐藏的模式，并广泛出现在几乎所有学科中，从自然科学到社会科学。这激发了科学家们探索复杂系统形成机制的兴趣。然而，这是非常具有挑战性的。本文从交易量-价格概率波动方程中提取了复杂系统中的互动一致性的普遍规律或法则，并将其应用于复杂量子系统。它假设粒子可以在加强坐标中具有复杂适应性学习或智能样性质，并将金融市场交易者的复杂适应性学习延伸到量子物理中的非生命粒子。在这些假设的基础上，作者提出了一种互动一致性纠缠理论，用于解释智能样粒子的行为。

    Complex adaptive learning is intelligent and plays roles in living and non-living complex systems. A complex system comprises many interacting individuals or units, shows hidden patterns as they interact, and widely occurs in almost every discipline, from natural to social sciences. It stimulates scientists to explore the mechanism of complex systems formulation. However, it is very challenging. Here the authors extract a universal rule or a law for interactive coherence in complex systems from a trading volume-price probability wave equation and apply it to complex quantum systems as its application. It assumes that particles can have a complex adaptive learning- or intelligence-like property in a reinforced coordinate and extend complex adaptive learning of traders in the financial markets to that of non-living particles in quantum physics. With these assumptions, the authors propose a theory of interactively coherent entanglement for intelligence-like particles, attempting to explai
    

