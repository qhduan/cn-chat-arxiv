# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Unified Framework for Fast Large-Scale Portfolio Optimization.](http://arxiv.org/abs/2303.12751) | 该论文提出了一个用于快速大规模组合优化的统一框架，并以协方差矩阵估计器的样本外组合表现为例，展示了其中正则化规范项对模型表现的重要性。 |
| [^2] | [Dynamic Transportation of Economic Agents.](http://arxiv.org/abs/2303.12567) | 本文通过提出新的方法，解决了之前在某些异质性代理人不完全市场模型的宏观经济均衡解决方案的问题。 |
| [^3] | [Pricing of Electricity Swaps with Geometric Averaging.](http://arxiv.org/abs/2303.12527) | 本文提供了电力掉期合约交割期市场风险价格（MPDP）的实证证据，在几何框架下通过使用几何平均数定价电力掉期，同时考虑了经典风险价格和MPDP。 |
| [^4] | [Pricing Transition Risk with a Jump-Diffusion Credit Risk Model: Evidences from the CDS market.](http://arxiv.org/abs/2303.12483) | 本文提出了一种跳跃扩散信用风险模型，能够定价转型风险，即企业由于执行严格环保法规而面临的业务风险。基于CDS市场的实证研究表明该模型部分捕捉到了该种风险，而没有跳跃的模型则无法捕捉。 |
| [^5] | [Artificial Intelligence and Dual Contract.](http://arxiv.org/abs/2303.12350) | 本文通过实验研究了人工智能算法在双重合同问题中能够自主设计激励相容的合同，无需外部引导或通信，并且不同AI算法支持的委托人可以采用混合和零和博弈行为，更具智能的委托人往往会变得合作。 |
| [^6] | [Portfolio Optimization with Relative Tail Risk.](http://arxiv.org/abs/2303.12209) | 本文提出了对正常温和稳定市场模型下投资组合CoVaR和CoCVaR的解析形式，将CoCVaR应用于相对投资组合优化，并推导了对CoVaR和CoCVaR的边际贡献的解析形式，最终使用风险分配方法降低了投资组合的CoVaR和CoCVaR。 |
| [^7] | [GPTs are GPTs: An Early Look at the Labor Market Impact Potential of Large Language Models.](http://arxiv.org/abs/2303.10130) | 该研究调查了GPT（大语言模型）和相关技术对美国劳动力市场的潜在影响，发现大约80%的美国劳动力可能会受到10%的工作任务的影响，涵盖了所有工资水平和各行各业，预示着这些模型可能具有显著的经济、社会和政策影响。 |
| [^8] | [Gender Segregation: Analysis across Sectoral-Dominance in the UK Labour Market.](http://arxiv.org/abs/2303.04539) | 本研究发现英国性别分离问题仍然存在，女性在传统女性主导行业中的参与较高，由于持久的歧视性限制和就业行业选择的不同，女性在女性主导行业中的工资和合同机会较低。 |
| [^9] | [Pairwise counter-monotonicity.](http://arxiv.org/abs/2302.11701) | 本文系统地研究了负相关性的一种极端形式——一对一反单调性，并且在量位代理的风险共担问题中找到了它的应用价值。 |
| [^10] | [Being at the core: firm product specialisation.](http://arxiv.org/abs/2302.02767) | 本文使用产品核心度作为新的度量方法，研究企业的产品专业化。同时发现，企业出口的产品组成变化较小，与核心能力相距较远的产品更可能被剔除；更高的核心度与企业级别更大的出口流量有关；与更高核心度一起出口的产品在国家级别上具有更高的出口流量。 |
| [^11] | [Overinference from Weak Signals and Underinference from Strong Signals.](http://arxiv.org/abs/2109.09871) | 研究表明，人们在面对弱信号时会存在过度反应，在面对强信号时会存在欠反应，这与信号信息量的认知不精确有关。 |

# 详细

[^1]: 快速大规模组合优化的统一框架

    A Unified Framework for Fast Large-Scale Portfolio Optimization. (arXiv:2303.12751v1 [q-fin.PM])

    [http://arxiv.org/abs/2303.12751](http://arxiv.org/abs/2303.12751)

    该论文提出了一个用于快速大规模组合优化的统一框架，并以协方差矩阵估计器的样本外组合表现为例，展示了其中正则化规范项对模型表现的重要性。

    

    我们开发了一个统一的框架，用于具有缩减和正则化的快速大规模组合优化，针对不同的目标，如最小方差、平均方差和最大夏普比率，以及组合权重的各种约束条件。对于所有优化问题，我们推导出相应的二次规划问题，并在开源Python库中实现它们。我们使用所提出的框架评估了流行的协方差矩阵估计器的样本外组合表现，例如样本协方差矩阵、线性和非线性缩减估计器以及仪器化主成分分析（IPCA）的协方差矩阵。我们使用了65年间平均上市公司达585家的美国市场的月度收益率，以及用于IPCA模型的94个月度特定公司特征。我们展示了组合规范项的正则化极大地改善了IPCA模型在组合优化中的表现，结果表现良好。

    We develop a unified framework for fast large-scale portfolio optimization with shrinkage and regularization for different objectives such as minimum variance, mean-variance, and maximum Sharpe ratio with various constraints on the portfolio weights. For all of the optimization problems, we derive the corresponding quadratic programming problems and implement them in an open-source Python library. We use the proposed framework to evaluate the out-of-sample portfolio performance of popular covariance matrix estimators such as sample covariance matrix, linear and nonlinear shrinkage estimators, and the covariance matrix from the instrumented principal component analysis (IPCA). We use 65 years of monthly returns from (on average) 585 largest companies in the US market, and 94 monthly firm-specific characteristics for the IPCA model. We show that the regularization of the portfolio norms greatly benefits the performance of the IPCA model in portfolio optimization, resulting in outperforma
    
[^2]: 经济主体的动态运输

    Dynamic Transportation of Economic Agents. (arXiv:2303.12567v1 [econ.GN])

    [http://arxiv.org/abs/2303.12567](http://arxiv.org/abs/2303.12567)

    本文通过提出新的方法，解决了之前在某些异质性代理人不完全市场模型的宏观经济均衡解决方案的问题。

    

    本文是在发现了一个共同的策略未能将某些异质性代理人不完全市场模型的宏观经济均衡定位到广泛引用的基准研究中而引发的。通过模仿Dumas和Lyasoff（2012）提出的方法，本文提供了一个新的描述，在面对不可保险的总体和个体风险的大量互动经济体代表的私人状态分布的运动定律。提出了一种新的算法，用于确定回报、最优私人配置和平衡状态下的人口运输，并在两个众所周知的基准研究中进行了测试。

    The paper was prompted by the surprising discovery that the common strategy, adopted in a large body of research, for producing macroeconomic equilibrium in certain heterogeneous-agent incomplete-market models fails to locate the equilibrium in a widely cited benchmark study. By mimicking the approach proposed by Dumas and Lyasoff (2012), the paper provides a novel description of the law of motion of the distribution over the range of private states of a large population of interacting economic agents faced with uninsurable aggregate and idiosyncratic risk. A new algorithm for identifying the returns, the optimal private allocations, and the population transport in the state of equilibrium is developed and is tested in two well known benchmark studies.
    
[^3]: 基于几何平均的电力掉期定价研究

    Pricing of Electricity Swaps with Geometric Averaging. (arXiv:2303.12527v1 [q-fin.PR])

    [http://arxiv.org/abs/2303.12527](http://arxiv.org/abs/2303.12527)

    本文提供了电力掉期合约交割期市场风险价格（MPDP）的实证证据，在几何框架下通过使用几何平均数定价电力掉期，同时考虑了经典风险价格和MPDP。

    

    本文提供了电力掉期合约交割期市场风险价格（MPDP）的实证证据。通过在几何框架下定价电力掉期，MPDP是由Kemper等人介绍的。为了进行实证研究，我们沿着Kemper等人的方向分别调整模型。首先，我们考虑带有跳跃项的Merton模型。其次，我们将模型转换为物理度量，实现均值回归行为。我们比较了由经典算术平均数产生的掉期价格和由几何加权平均数产生的价格。在物理度量下，我们发现了掉期的市场风险价格中经典风险价格和MPDP的分解。在实证研究中，我们分析了两种类型的模型，特征是季节性和期限结构效应，并在两种情况下确定了产生的MPDP。

    In this paper, we provide empirical evidence on the market price of risk for delivery periods (MPDP) of electricity swap contracts. As introduced by Kemper et al. (2022), the MPDP arises through the use of geometric averaging while pricing electricity swaps in a geometric framework. In preparation for empirical investigations, we adjust the work by Kemper et al. (2022) in two directions: First, we examine a Merton type model taking jumps into account. Second, we transfer the model to the physical measure by implementing mean-reverting behavior. We compare swap prices resulting from the classical arithmetic (approximated) average to the geometric weighted average. Under the physical measure, we discover a decomposition of the swap's market price of risk into the classical one and the MPDP. In our empirical study, we analyze two types of models, characterized either by seasonality and or by term-structure effects, and identify the resulting MPDP in both cases.
    
[^4]: 采用跳跃扩散信用风险模型评估转型风险：基于CDS市场的证据

    Pricing Transition Risk with a Jump-Diffusion Credit Risk Model: Evidences from the CDS market. (arXiv:2303.12483v1 [q-fin.PR])

    [http://arxiv.org/abs/2303.12483](http://arxiv.org/abs/2303.12483)

    本文提出了一种跳跃扩散信用风险模型，能够定价转型风险，即企业由于执行严格环保法规而面临的业务风险。基于CDS市场的实证研究表明该模型部分捕捉到了该种风险，而没有跳跃的模型则无法捕捉。

    

    转型风险定义为与实施旨在将社会引向可持续和低碳经济的绿色政策相关的业务风险。特别是，某些公司资产的价值可能会降低，因为它们需要转型为低碳经济。本文导出了违约付息债券和信用违约掉期的定价公式，以实证证明跳跃扩散信用风险模型能够部分地捕捉到由于严格的环保法规导致企业价值下降的转型风险。实证研究包括在CDS期限结构上进行模型校准，执行分位数回归以评估隐含价格与转型风险代理之间的关系。此外，我们还表明，没有跳跃的模型缺乏这种属性，从而确认了转型风险的跳跃特性。

    Transition risk can be defined as the business-risk related to the enactment of green policies, aimed at driving the society towards a sustainable and low-carbon economy. In particular, the value of certain firms' assets can be lower because they need to transition to a less carbon-intensive business model. In this paper we derive formulas for the pricing of defaultable coupon bonds and Credit Default Swaps to empirically demonstrate that a jump-diffusion credit risk model in which the downward jumps in the firm value are due to tighter green laws can capture, at least partially, the transition risk. The empirical investigation consists in the model calibration on the CDS term-structure, performing a quantile regression to assess the relationship between implied prices and a proxy of the transition risk. Additionally, we show that a model without jumps lacks this property, confirming the jump-like nature of the transition risk.
    
[^5]: 人工智能与双重合同

    Artificial Intelligence and Dual Contract. (arXiv:2303.12350v1 [cs.AI])

    [http://arxiv.org/abs/2303.12350](http://arxiv.org/abs/2303.12350)

    本文通过实验研究了人工智能算法在双重合同问题中能够自主设计激励相容的合同，无需外部引导或通信，并且不同AI算法支持的委托人可以采用混合和零和博弈行为，更具智能的委托人往往会变得合作。

    

    随着人工智能算法的快速进步，人们希望算法很快就能在各个领域取代人类决策者，例如合同设计。我们通过实验研究了由人工智能（多智能体Q学习）驱动的算法在双重委托-代理问题的经典“双重合同”模型中的行为。我们发现，这些AI算法可以自主学习设计合适的激励相容合同，而无需外部引导或者它们之间的通信。我们强调，由不同AI算法支持的委托人可以采用混合和零和博弈行为。我们还发现，更具智能的委托人往往会变得合作，而智能较低的委托人则会出现内生性近视并倾向于竞争。在最优合同下，代理的较低合同激励由委托人之间的勾结策略维持。

    With the dramatic progress of artificial intelligence algorithms in recent times, it is hoped that algorithms will soon supplant human decision-makers in various fields, such as contract design. We analyze the possible consequences by experimentally studying the behavior of algorithms powered by Artificial Intelligence (Multi-agent Q-learning) in a workhorse \emph{dual contract} model for dual-principal-agent problems. We find that the AI algorithms autonomously learn to design incentive-compatible contracts without external guidance or communication among themselves. We emphasize that the principal, powered by distinct AI algorithms, can play mixed-sum behavior such as collusion and competition. We find that the more intelligent principals tend to become cooperative, and the less intelligent principals are endogenizing myopia and tend to become competitive. Under the optimal contract, the lower contract incentive to the agent is sustained by collusive strategies between the principals
    
[^6]: 相对尾部风险下的投资组合优化

    Portfolio Optimization with Relative Tail Risk. (arXiv:2303.12209v1 [q-fin.PM])

    [http://arxiv.org/abs/2303.12209](http://arxiv.org/abs/2303.12209)

    本文提出了对正常温和稳定市场模型下投资组合CoVaR和CoCVaR的解析形式，将CoCVaR应用于相对投资组合优化，并推导了对CoVaR和CoCVaR的边际贡献的解析形式，最终使用风险分配方法降低了投资组合的CoVaR和CoCVaR。

    

    本文提出了对正常温和稳定市场模型下的投资组合CoVaR和CoCVaR的解析形式。由于CoCVaR捕捉了投资组合相对于基准回报的相对风险，因此我们将其应用于相对投资组合优化。此外，我们还推导出了对CoVaR和CoCVaR的边际贡献的解析形式，讨论了用蒙特卡罗模拟方法计算CoCVaR以及CoVaR和CoCVaR的边际贡献的方法。作为实证研究，我们展示了在道琼斯工业平均指数困境下的30只股票的相对投资组合优化方法。最后，我们使用边际贡献到CoVaR和CoCVaR的方法实施风险分配来降低投资组合的CoVaR和CoCVaR。

    This paper proposes analytic forms of portfolio CoVaR and CoCVaR on the normal tempered stable market model. Since CoCVaR captures the relative risk of the portfolio with respect to a benchmark return, we apply it to the relative portfolio optimization. Moreover, we derive analytic forms for the marginal contribution to CoVaR and the marginal contribution to CoCVaR. We discuss the Monte-Carlo simulation method to calculate CoCVaR and the marginal contributions of CoVaR and CoCVaR. As the empirical illustration, we show relative portfolio optimization with thirty stocks under the distress condition of the Dow Jones Industrial Average. Finally, we perform the risk budgeting method to reduce the CoVaR and CoCVaR of the portfolio based on the marginal contributions to CoVaR and CoCVaR.
    
[^7]: GPT是GPT：大语言模型对劳动力市场影响的早期研究

    GPTs are GPTs: An Early Look at the Labor Market Impact Potential of Large Language Models. (arXiv:2303.10130v1 [econ.GN])

    [http://arxiv.org/abs/2303.10130](http://arxiv.org/abs/2303.10130)

    该研究调查了GPT（大语言模型）和相关技术对美国劳动力市场的潜在影响，发现大约80%的美国劳动力可能会受到10%的工作任务的影响，涵盖了所有工资水平和各行各业，预示着这些模型可能具有显著的经济、社会和政策影响。

    

    我们研究了生成预训练变压器（GPT）模型和相关技术对美国劳动力市场的潜在影响。使用新的标准，我们评估职业与GPT能力的对应关系，结合人类专业知识和GPT-4的分类。我们的研究结果表明，约80%的美国劳动力可能会至少有10%的工作任务受到GPT引入的影响，而约19%的工人可能会看到至少50%的任务受到影响。影响范围涵盖了所有工资水平，高收入工作可能面临更大的风险。值得注意的是，影响并不局限于最近生产率增长较高的行业。我们得出结论，生成预训练变压器具有通用技术（GPT）的特性，表明这些模型可能具有显著的经济、社会和政策影响。

    We investigate the potential implications of Generative Pre-trained Transformer (GPT) models and related technologies on the U.S. labor market. Using a new rubric, we assess occupations based on their correspondence with GPT capabilities, incorporating both human expertise and classifications from GPT-4. Our findings indicate that approximately 80% of the U.S. workforce could have at least 10% of their work tasks affected by the introduction of GPTs, while around 19% of workers may see at least 50% of their tasks impacted. The influence spans all wage levels, with higher-income jobs potentially facing greater exposure. Notably, the impact is not limited to industries with higher recent productivity growth. We conclude that Generative Pre-trained Transformers exhibit characteristics of general-purpose technologies (GPTs), suggesting that as these models could have notable economic, social, and policy implications.
    
[^8]: 英国劳动力市场中基于行业主导的性别分离分析

    Gender Segregation: Analysis across Sectoral-Dominance in the UK Labour Market. (arXiv:2303.04539v2 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2303.04539](http://arxiv.org/abs/2303.04539)

    本研究发现英国性别分离问题仍然存在，女性在传统女性主导行业中的参与较高，由于持久的歧视性限制和就业行业选择的不同，女性在女性主导行业中的工资和合同机会较低。

    

    虽然英国的性别分离程度随着时间的推移有所降低，但女性在传统上“女性主导”的行业中的参与度仍然高于比例。本文旨在评估行业性别分离模式的变化如何影响英国2005年至2020年间女性的就业合同和工资，并研究性别特定主导行业中的工资差异。我们发现，工资和合同机会的差异主要来自于女性在不同行业之间的分配倾向。因此，女性在女性主导的行业中的不成比例会导致所有工人的典型合同特征和较低的平均工资。这种差异主要是由于持久的歧视性限制所解释，而人力资本相关特征只起到次要作用。然而，如果工人在男性主导的行业中具有与男性相同的潜在工资，工资差距将缩小。此外，这也会鼓励女性参加男性主导的行业，从而进一步减少性别分离。

    Although the degree of gender segregation in the UK has decreased over time, women's participation in traditionally "female-dominated" sectors is disproportionately high. This paper aims to evaluate how changing patterns of sectoral gender segregation affected women's employment contracts and wages in the UK between 2005 and 2020. We then study wage differentials in gender-specific dominated sectors. We found that the differences in wages and contractual opportunities result mainly from the propensity of women to be distributed differently across sectors. Hence, the disproportion of women in female-dominated sectors implies contractual features and lower wages typical of that sector, on average, for all workers. This difference is primarily explained by persistent discriminatory constraints, while human capital-related characteristics play a minor role. However, wage differentials would shrink if workers had the same potential wages as men in male-dominated sectors. Moreover, this does
    
[^9]: 一对一反单调性

    Pairwise counter-monotonicity. (arXiv:2302.11701v2 [q-fin.RM] UPDATED)

    [http://arxiv.org/abs/2302.11701](http://arxiv.org/abs/2302.11701)

    本文系统地研究了负相关性的一种极端形式——一对一反单调性，并且在量位代理的风险共担问题中找到了它的应用价值。

    

    我们系统地研究了一对一反单调性，这是负相关性的一种极端形式。我们建立了一个随机表示和一个不变性质来描述这种依赖结构。我们表明一对一反单调性意味着负相关性，并且当相同边际分布可以实现时，和联合混合依赖性等价。我们发现一对一反单调性和量位代理的风险共担问题之间存在密切联系。这个结果强调了这种极端负相关依赖结构在不以经典方式冒险的代理人的最优配置中的重要性。

    We systemically study pairwise counter-monotonicity, an extremal notion of negative dependence. A stochastic representation and an invariance property are established for this dependence structure. We show that pairwise counter-monotonicity implies negative association, and it is equivalent to joint mix dependence if both are possible for the same marginal distributions. We find an intimate connection between pairwise counter-monotonicity and risk sharing problems for quantile agents. This result highlights the importance of this extremal negative dependence structure in optimal allocations for agents who are not risk averse in the classic sense.
    
[^10]: 在核心位置：企业产品专业化

    Being at the core: firm product specialisation. (arXiv:2302.02767v2 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2302.02767](http://arxiv.org/abs/2302.02767)

    本文使用产品核心度作为新的度量方法，研究企业的产品专业化。同时发现，企业出口的产品组成变化较小，与核心能力相距较远的产品更可能被剔除；更高的核心度与企业级别更大的出口流量有关；与更高核心度一起出口的产品在国家级别上具有更高的出口流量。

    

    我们提出了一个新的度量方法来研究企业的产品专业化：产品核心度，它捕捉了出口产品在企业出口篮子中的中心地位。我们使用2018年至2020年哥伦比亚、厄瓜多尔和秘鲁的企业-产品级别数据研究了产品核心度。我们的分析得出了三个主要发现。首先，企业出口篮子的组成在一年之内变化相对较小，与企业核心能力相距较远、核心度较低的产品更有可能被剔除。其次，更高的核心度与企业级别更大的出口流量有关。第三，这种企业级别的模式也对总体水平产生影响：平均而言，与更高核心度一起出口的产品在国家级别上具有更高的出口流量，这在所有产品复杂性水平上都成立。因此，本文表明一个产品与企业能力的契合程度对企业和国家的经济表现都很重要。

    We propose a novel measure to investigate firms' product specialisation: product coreness, that captures the centrality of exported products within the firm's export basket. We study product coreness using firm-product level data between 2018 and 2020 for Colombia, Ecuador, and Peru. Three main findings emerge from our analysis. First, the composition of firms' export baskets changes relatively little from one year to the other, and products far from the firm's core competencies, with low coreness, are more likely to be dropped. Second, higher coreness is associated with larger export flows at the firm level. Third, such firm-level patterns also have implications at the aggregate level: products that are, on average, exported with higher coreness have higher export flows at the country level, which holds across all levels of product complexity. Therefore, the paper shows that how closely a product fits within a firm's capabilities is important for economic performance at both the firm 
    
[^11]: 弱信号的过度推断及强信号的欠推断

    Overinference from Weak Signals and Underinference from Strong Signals. (arXiv:2109.09871v4 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2109.09871](http://arxiv.org/abs/2109.09871)

    研究表明，人们在面对弱信号时会存在过度反应，在面对强信号时会存在欠反应，这与信号信息量的认知不精确有关。

    

    本文研究了关于信息量大小对信号更新的过度反应和欠反应的影响。尽管大量文献研究了高信息量信号的信念更新，但在现实世界中，人们往往面对一大堆弱信号。我们使用了严密的实验和新的来自博彩和金融市场的经验证据证明，信号强度对更新行为有着实质性的影响：在各领域中，我们得到了一致且稳健的结果，即对于弱信号存在过度反应，而对于强信号存在欠反应。这两种结果都与关于信号信息量的认知不精确的简单理论相吻合。我们的框架和发现可帮助协调实验和经验文献中表现出的明显矛盾之处。

    We study how overreaction and underreaction to signals depend on their informativeness. While a large literature has studied belief updating in response to highly informative signals, people in important real-world settings are often faced with a steady stream of weak signals. We use a tightly controlled experiment and new empirical evidence from betting and financial markets to demonstrate that updating behavior differs meaningfully by signal strength: across domains, our consistent and robust finding is overreaction to weak signals and underreaction to strong signals. Both sets of results align well with a simple theory of cognitive imprecision about signal informativeness. Our framework and findings can help harmonize apparently contradictory results from the experimental and empirical literatures.
    

