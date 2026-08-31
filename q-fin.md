# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RetailAgent: Structured Adverse Timing in Self-Conditioned Multimodal LLM Trading Agents](https://arxiv.org/abs/2608.28399) | 该论文提出RetailAgent实验框架，发现大语言模型交易智能体在跨模态、时间跨度、状态和模型系列下均表现出持续的结构化负向择时——其做多决策与随后较差的收益系统性对齐，且打乱动作序列可显著减弱该效应，证明这种不利择时源于动作与后续收益之间的对齐。 |
| [^2] | [Market-Informed Valuation of GMMB Riders with Surrender Options under a Heston Stochastic-Local Volatility Model](https://arxiv.org/abs/2608.28397) | 本文提出在Heston随机-局部波动率模型下对含理性退保权利的GMMB附加条款进行市场化估值的框架，通过前向马尔可夫投影使模型与局部波动率模型保持相同的一维边缘分布，从而在期权校准一致的前提下分离出随机波动率对延续价值与退保决策的影响。 |
| [^3] | [The Race for Elite Destinations: Education Competition and Low Fertility in Korea](https://arxiv.org/abs/2608.27980) | 韩国低生育率的根源是教育竞争产生的“分配外部性”——用抽签取代分数录取可使每对夫妇的生育率提高0.24个孩子，远超现金补贴的效果，而教育税几乎无效。 |
| [^4] | [What survives honest evaluation? Leakage-safe, search-aware assessment of LLM-driven trading strategy discovery](https://arxiv.org/abs/2608.27734) | 该论文提出了一个LLM驱动的交易策略发现系统，通过结构性机制（构造上排除前视偏差的注册表验证工具，以及基于搜索试验次数的性能紧缩修正）应对前视偏差与搜索强度问题，并证明一个夏普比率高达35的泄漏预言机仍能通过标准统计检验，说明结构性防护不可被统计修正替代。 |
| [^5] | [Do Customer Disclosures Affect Suppliers' Internal Capital Allocation Decisions?](https://arxiv.org/abs/2608.27598) | 该研究发现，客户披露的增加会加剧产品市场竞争，促使供应商偏离增长信号重新配置内部资本，以捍卫客户关系并扩大客户基础。 |
| [^6] | [Pricing and Calibration of Bitcoin Inverse Options via the Rough Bergomi Model](https://arxiv.org/abs/2608.27575) | 本文将粗糙Bergomi模型适配于比特币反向期权的非线性反向收益结构，构建并实证验证了包含三种蒙特卡洛计算流程的定价与校准框架，并成功将其校准至Deribit交易所的真实隐含波动率曲面数据。 |
| [^7] | [Disaffection at Work: Employee Responses to Job-Related Information](https://arxiv.org/abs/2608.27538) | 本研究通过在意大利和法国开展的随机调查实验发现，强调社会正义的职场道德叙事会增加员工的工作疏离感但减少被动怠工，而强调工作组织的叙事则无系统性影响，揭示了“安静辞职”这类员工不满主要通过劳动供给的集约边际而非离职来体现。 |
| [^8] | [The Dynamic Trade-Off of Dual-Class Shares](https://arxiv.org/abs/2608.25972) | 本文通过52年数据动态分析双重股权结构，发现其短期提升估值但长期下降，而创新产出持续增长，且效应集中于公司特定投资行业，并据此提出新政策启示。 |
| [^9] | [Does Rank Still Matter? Position Bias When AI Agents Shop on Our Behalf](https://arxiv.org/abs/2608.22697) | 本论文发现AI代理搜索时，结果页面的位置影响微弱且非单调，属性内容比排名位置更重要。 |
| [^10] | [Digital Engagement, Income Disparities, and Job Seeking in the United States since 2010](https://arxiv.org/abs/2511.05294) | 本研究利用美国NLSY97纵向队列数据发现，2010年以来互联网使用频率与劳动力市场结果密切相关：低频使用者收入约低11%–20%，完全不使用者收入约低18%–21%且全年就业可能性低13–23个百分点，凸显了数字参与程度是收入差距与就业稳定性的重要标志。 |

# 详细

[^1]: RetailAgent：自条件多模态大语言模型交易智能体中的结构化不利择时

    RetailAgent: Structured Adverse Timing in Self-Conditioned Multimodal LLM Trading Agents

    [https://arxiv.org/abs/2608.28399](https://arxiv.org/abs/2608.28399)

    该论文提出RetailAgent实验框架，发现大语言模型交易智能体在跨模态、时间跨度、状态和模型系列下均表现出持续的结构化负向择时——其做多决策与随后较差的收益系统性对齐，且打乱动作序列可显著减弱该效应，证明这种不利择时源于动作与后续收益之间的对齐。

    

    在金融市场中，对价格变动做出系统性反应的顺序策略可能会被其他市场参与者所预测。本文研究大语言模型（LLM）智能体是否表现出这种方向性结构，提出了RetailAgent——一个实验框架，其中LLM观察匿名化的日内股票价格历史及允许的状态，然后在后续区间收益揭晓之前反复选择做多（持有股票）或空仓（不参与）。在剔除做多决策的总体比例后，我们比较同一股票日内路径上做多区间与空仓区间的收益。这种敞口匹配的度量揭示了跨模态、时间跨度、状态和模型系列的持续性负向择时。打乱已保存的动作序列会显著减弱该效应，表明动作与后续收益之间的对齐是负向得分的驱动因素。将自生成的记忆馈入决策……（原文摘要至此截断）

    arXiv:2608.28399v1 Announce Type: cross  Abstract: In financial markets, a sequential policy that reacts systematically to price movements may become predictable to other market participants. This paper studies whether large language model (LLM) agents exhibit such directional structure through RetailAgent, an experimental framework in which an LLM observes anonymized intraday equity price histories and permitted state, then repeatedly chooses long (hold the stock) or flat (stay out) before the subsequent interval return is revealed. We compare returns during long and flat intervals along the same stock's intraday path after removing the overall fraction of long decisions. This exposure-matched measure reveals persistent negative timing across modality, horizon, state, and model family. Shuffling saved action sequences substantially attenuates the effect, showing that alignment between actions and subsequent returns drives the negative score. Feeding self-authored memories into decisio
    
[^2]: Heston随机-局部波动率模型下含退保期权的最低到期给付保证（GMMB）附加条款的市场化估值

    Market-Informed Valuation of GMMB Riders with Surrender Options under a Heston Stochastic-Local Volatility Model

    [https://arxiv.org/abs/2608.28397](https://arxiv.org/abs/2608.28397)

    本文提出在Heston随机-局部波动率模型下对含理性退保权利的GMMB附加条款进行市场化估值的框架，通过前向马尔可夫投影使模型与局部波动率模型保持相同的一维边缘分布，从而在期权校准一致的前提下分离出随机波动率对延续价值与退保决策的影响。

    

    我们提出了一个在Heston随机-局部波动率（SLV）模型下、针对具有理性退保行为的最低到期给付保证（GMMB）附加条款的市场化估值框架。该保证以扣除费用后的账户价值为标的，我们同时考虑了仅在到期日给付的形式以及包含提前退保权利的情形。Heston SLV模型将随机波动率与一个根据给定局部波动率曲面校准的杠杆函数相结合。杠杆曲面通过前向马尔可夫投影方程求得，从而在模型层面使SLV动态被约束为与相应的局部波动率（LV）模型具有相同的一维边缘分布。后者仅作为单因子基准使用，使我们能够在保持相同的期权校准局部波动率目标的同时，分离出随机波动率对延续价值与退保决策的影响。

    arXiv:2608.28397v1 Announce Type: new  Abstract: We develop a market-informed valuation framework for   guaranteed minimum maturity benefit (GMMB) riders with   rational surrender under the Heston stochastic-local   volatility (SLV) model. The guarantee is written on the   fee-deducted account value and is considered both in its   terminal-only form and in the presence of early surrender   rights. The Heston SLV specification combines stochastic volatility with a leverage function calibrated to a prescribed local-volatility surface. The leverage surface is obtained through a forward Markovian-projection equation so that, at the model level, the SLV dynamics are constrained to the same one-dimensional marginals as the corresponding local-volatility (LV) model. The latter is used only as a one-factor benchmark, allowing us to isolate the effect of stochastic volatility on continuation values and surrender decisions while preserving the same option-calibrated local-volatility target. We d
    
[^3]: 精英之赛：韩国的教育竞争与低生育率

    The Race for Elite Destinations: Education Competition and Low Fertility in Korea

    [https://arxiv.org/abs/2608.27980](https://arxiv.org/abs/2608.27980)

    韩国低生育率的根源是教育竞争产生的“分配外部性”——用抽签取代分数录取可使每对夫妇的生育率提高0.24个孩子，远超现金补贴的效果，而教育税几乎无效。

    

    韩国拥有世界上最低的生育率和激烈的教育竞赛。家庭将终身收入的百分之九用于教育，主要用于几乎测不出回报的私人补习。我展示了争夺令人向往的职业会产生一种“分配外部性”：当所有家庭都增加支出时，录取门槛随之上升，孩子变得成本高昂。在一个根据该机制校准的定量模型中，用保持招生名额不变的抽签制度取代基于分数的录取分配，可使每对夫妇的完成生育率提高0.24个孩子。鼓励生育的转移支付只能实现该增量的八分之一，教育税则几乎无效。偏好变化，而非更激烈的竞赛，解释了不同世代间生育率的下降。

    arXiv:2608.27980v1 Announce Type: new  Abstract: South Korea has the world's lowest fertility and an intense education race. Families devote nine percent of lifetime income to education, mostly to private tutoring with near-zero measured returns. I show that competition for coveted careers generates an assignment externality: when all families spend more, the admission bar rises and children become costly. In a quantitative model calibrated to this mechanism, replacing score-based assignment with a capacity-preserving lottery raises completed fertility by 0.24 children per couple. Pronatal transfers deliver one eighth of that increase. Education taxes deliver almost none. Preference shifts, not a fiercer race, explain the decline across cohorts.
    
[^4]: 什么能在诚实的评估中幸存？对LLM驱动的交易策略发现的防泄漏、搜索感知评估

    What survives honest evaluation? Leakage-safe, search-aware assessment of LLM-driven trading strategy discovery

    [https://arxiv.org/abs/2608.27734](https://arxiv.org/abs/2608.27734)

    该论文提出了一个LLM驱动的交易策略发现系统，通过结构性机制（构造上排除前视偏差的注册表验证工具，以及基于搜索试验次数的性能紧缩修正）应对前视偏差与搜索强度问题，并证明一个夏普比率高达35的泄漏预言机仍能通过标准统计检验，说明结构性防护不可被统计修正替代。

    

    大语言模型（LLM）越来越多地被用于发现交易策略，而由此产生的许多文献都存在一个共同的方法论缺陷：研究者生成大量候选策略，只报告其中表现最好的一个，既不纠正前视偏差，也不对报告结果背后的搜索强度进行修正。我们提出了一个策略发现系统，将这两项修正从程序性措施转变为结构性保障。首先，智能体只能通过经过注册表验证的工具进行操作，这些工具的特征空间在构造上就排除了前视偏差；我们证明这种防护机制与统计修正并不冗余：一个故意设计为存在数据泄漏的预言机，其夏普比率高达35，却能完全通过紧缩夏普比率和回测过拟合概率检验。其次，系统记录其搜索所执行的每一次策略评估，并根据试验次数对所有报告的性能进行紧缩修正，追踪最优样本内夏普比率如何随每次试验而攀升。

    arXiv:2608.27734v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used to discover trading strategies, and much of the resulting literature shares a methodological weakness: many candidate strategies are generated, the best is reported, and neither look-ahead bias nor the intensity of the search behind the reported result is corrected for. We present a strategy-discovery system that makes both corrections structural rather than procedural. First, the agent can only act through registry-validated tools whose feature space excludes look-ahead by construction; we show that this guardrail is not redundant with statistical correction: a deliberately leaky oracle posting a Sharpe ratio of 35 survives Deflated Sharpe and probability-of-backtest-overfitting testing completely. Second, the system records every strategy evaluation its search performs and deflates all reported performance by that trial count, tracing how the best in-sample Sharpe ratio climbs with eac
    
[^5]: 客户披露会影响供应商的内部资本配置决策吗？

    Do Customer Disclosures Affect Suppliers' Internal Capital Allocation Decisions?

    [https://arxiv.org/abs/2608.27598](https://arxiv.org/abs/2608.27598)

    该研究发现，客户披露的增加会加剧产品市场竞争，促使供应商偏离增长信号重新配置内部资本，以捍卫客户关系并扩大客户基础。

    

    本研究考察客户披露是否会影响供应商企业跨业务部门的资本配置决策。客户披露可以通过两个相互竞争的渠道影响供应商的投资决策：一是改善供应商对下游需求的信息，帮助供应商将资本与增长机会相匹配（“信息渠道”）；二是削弱现有供应商的私人信息优势，诱使其进行高成本投资以捍卫客户关系（“竞争威胁渠道”）。本文采用SFAS 131准则的采用作为客户层面的披露冲击。研究发现，暴露于扩大客户披露的供应商面临更激烈的产品市场竞争，并将资本重新配置到增长机会信号相对较弱的部门。偏离增长信号所预测的配置方案的供应商，在随后几年中更有可能保持市场份额并扩大其客户基础。

    arXiv:2608.27598v1 Announce Type: new  Abstract: This study examines whether customer disclosures affect how supplier firms allocate capital across business segments. Customer disclosures can shape supplier investment decisions through two competing channels. They can improve suppliers' information about downstream demand, helping suppliers align capital with growth opportunities ("information channel"), or erode incumbent suppliers' private information advantage, inducing costly investments to defend customer relationships ("competitive-threat channel"). I use the adoption of SFAS 131 as a customer-level disclosure shock. Suppliers exposed to expanded customer disclosures experience increased product-market competition and reallocate capital toward segments with relatively weak growth-opportunity signals. Suppliers that deviate from allocations predicted by growth signals are more likely to preserve market share and expand their customer base in subsequent years. Using a novel approac
    
[^6]: 基于粗糙Bergomi模型的比特币反向期权定价与校准

    Pricing and Calibration of Bitcoin Inverse Options via the Rough Bergomi Model

    [https://arxiv.org/abs/2608.27575](https://arxiv.org/abs/2608.27575)

    本文将粗糙Bergomi模型适配于比特币反向期权的非线性反向收益结构，构建并实证验证了包含三种蒙特卡洛计算流程的定价与校准框架，并成功将其校准至Deribit交易所的真实隐含波动率曲面数据。

    

    比特币反向期权在Deribit交易所交易，并以标的加密货币而非法定货币进行结算，它将极端且真正粗糙的波动率动态与非线性的、依赖货币的收益结构结合在一起。本文基于Bayer、Friz和Gatheral（2016）提出的粗糙Bergomi（rBergomi）模型，为这类金融工具开发并实证验证了一个定价与校准框架。我们将rBergomi动态调整以适应反向收益结构 max(S_T - K, 0)/S_T，并实现和比较了三种计算流程，这些流程在驱动分数布朗运动的模拟方案（粗网格Cholesky方法 vs. Bennedsen等人2017年提出的混合方案）以及蒙特卡洛定价估计器（普通对数Euler方法 vs. McCrickerd和Pakkanen 2018年提出的混合估计器）方面有所不同。该模型被校准至从2022年5月至2023年3月Deribit交易数据中提取的三十个隐含波动率曲面。

    arXiv:2608.27575v1 Announce Type: new  Abstract: Bitcoin inverse options, traded on the Deribit exchange and settled in the underlying cryptocurrency rather than in fiat currency, combine extreme and genuinely rough volatility dynamics with a non-linear, currency-dependent payoff structure. This paper develops and empirically validates a pricing and calibration framework for these instruments based on the rough Bergomi (rBergomi) model of Bayer, Friz and Gatheral (2016). We adapt the rBergomi dynamics to the inverse payoff max(S_T - K, 0)/S_T, and implement and compare three computational pipelines that differ in the simulation scheme for the driving fractional Brownian motion (coarse-grid Cholesky vs. the Hybrid Scheme of Bennedsen et al., 2017) and in the Monte Carlo pricing estimator (plain log-Euler vs. the Mixed Estimator of McCrickerd and Pakkanen, 2018). The model is calibrated to thirty implied volatility surfaces extracted from Deribit trade data between May 2022 and March 202
    
[^7]: 工作中的不满情绪：员工对工作相关信息的回应

    Disaffection at Work: Employee Responses to Job-Related Information

    [https://arxiv.org/abs/2608.27538](https://arxiv.org/abs/2608.27538)

    本研究通过在意大利和法国开展的随机调查实验发现，强调社会正义的职场道德叙事会增加员工的工作疏离感但减少被动怠工，而强调工作组织的叙事则无系统性影响，揭示了“安静辞职”这类员工不满主要通过劳动供给的集约边际而非离职来体现。

    

    “安静辞职”反映了一种工人不满情绪的表现形式，它通过劳动供给的集约边际而非离职来运作。我们利用一项针对意大利和法国员工代表性样本的随机调查实验，研究不同的职场叙事如何塑造员工的行为反应。受访者接触到基于实证的关于工作的道德框架，这些框架要么强调社会正义与集体权利，要么强调工作组织与雇佣实践。我们发现，道德框架会在不同边际之间重新分配行为：以正义为导向的叙事会增加员工的疏离感，同时减少被动消极怠工，而以组织为中心的框架则没有产生系统性影响。

    arXiv:2608.27538v1 Announce Type: new  Abstract: Quiet quitting reflects a form of worker disaffection that operates along the intensive margin of labor supply rather than through job exit. We study how alternative workplace narratives shape workers' behavioral responses using a randomized survey experiment on a representative sample of employees in Italy and France. Respondents are exposed to empirically grounded moral framings of work emphasizing either social justice and collective rights or work organization and employment practices. We find that moral framings reallocate behavior across margins: justice-oriented narratives increase detachment while reducing passive disengagement, whereas organization-centered framing generates no systematic effects.
    
[^8]: 双重股权结构的动态权衡

    The Dynamic Trade-Off of Dual-Class Shares

    [https://arxiv.org/abs/2608.25972](https://arxiv.org/abs/2608.25972)

    本文通过52年数据动态分析双重股权结构，发现其短期提升估值但长期下降，而创新产出持续增长，且效应集中于公司特定投资行业，并据此提出新政策启示。

    

    双重股权结构将控制权分配给创始人，其公司特定投资驱动了企业价值，但控制权与所有权分离，增加了代理成本。我们动态地分析了这一权衡。利用覆盖52年的美国双重股权公司新数据和差分设计，我们表明在双重股权重组后估值上升，但随时间下降，而创新产出持续增加。这些效应集中在公司特定投资更多的行业中。我们在股票统一中也发现了相应结果。成熟双重股权公司的投资对机会的敏感度较低，投票溢价随成熟度增加。我们的结果支持动态处理效应，并产生了新的政策含义。

    arXiv:2608.25972v1 Announce Type: new  Abstract: Dual-class shares allocate control to founders whose firm-specific investments drive firm value but separate control from ownership, raising agency costs. We analyze this trade-off dynamically. Using new data on US dual-class firms spanning 52 years and difference-in-differences designs, we show that valuations rise following dual-class recapitalizations but decline over time, whereas innovative output increases persistently. These effects are concentrated in industries with greater firm-specific investments. We find corresponding results for stock unifications. Investment by mature dual-class firms is less sensitive to opportunities and voting premia increase with maturity. Our results support dynamic treatment effects and yield new policy implications.
    
[^9]: 排名仍然重要吗？当AI代理替我们购物时的位置偏差

    Does Rank Still Matter? Position Bias When AI Agents Shop on Our Behalf

    [https://arxiv.org/abs/2608.22697](https://arxiv.org/abs/2608.22697)

    本论文发现AI代理搜索时，结果页面的位置影响微弱且非单调，属性内容比排名位置更重要。

    

    arXiv:2608.22697v1 公告类型：新 摘要：搜索排名之所以有价值，是因为人类的注意力稀缺且具有顺序性。排名靠前的选项更容易被找到，因此被查看和购买的可能性更高。如今，消费者正将搜索委托给能够一次性处理整个结果页的AI代理。通过将一百个酒店列表的顺序随机化，并在5,000次AI代理会话中进行实验，我们将四种大型语言模型与人类实地数据进行比较。AI代理比人类搜索得更深入，且从不拒绝购买。位置仍然预测哪些列表被查看，但影响微弱且非单调：结果页中间部分的查看概率最低，而非底部。对于某些模型，位置会影响选择阶段，而对其他模型则不然，这种异质性既不与提供商相关，也不与能力相关。尽管如此，所有模型最终都收敛于同一个未被支配的列表。对于代理搜索而言，结果页上显示的属性比其在页面中的位置更为重要。

    arXiv:2608.22697v1 Announce Type: new  Abstract: Search rankings are valuable because human attention is scarce and sequential. Higher-placed alternatives are easier to find, so they are examined and bought more often. Consumers are now delegating search to AI agents that can ingest an entire results page at once. Randomizing the order of one hundred hotel listings across 5,000 AI agent sessions, we compare four large language models against human field data. AI agents search more deeply than humans and never decline to buy. Position still predicts which listings are inspected, but weakly and non-monotonically: the middle of a results page has the lowest probability of inspection, not the bottom. Position reaches the choice stage for some models and not others, a heterogeneity that tracks neither provider nor capability. All models nonetheless converge on the same undominated listing. For agentic search, the attributes displayed on a results page matter more than placement within it.
    
[^10]: 2010年以来美国的数字参与、收入差距与求职

    Digital Engagement, Income Disparities, and Job Seeking in the United States since 2010

    [https://arxiv.org/abs/2511.05294](https://arxiv.org/abs/2511.05294)

    本研究利用美国NLSY97纵向队列数据发现，2010年以来互联网使用频率与劳动力市场结果密切相关：低频使用者收入约低11%–20%，完全不使用者收入约低18%–21%且全年就业可能性低13–23个百分点，凸显了数字参与程度是收入差距与就业稳定性的重要标志。

    

    调查通常只记录人们使用互联网的频率，却未能测量使数字参与成为可能的基础设施、技能和支持系统。本研究利用美国1997年全国青年纵向调查（NLSY97）队列数据，研究2010年以后互联网使用频率与劳动收入、就业稳定性及求职之间的关系。主要的数字参与分析采用可比的2011年、2013年和2015年调查轮次，并将2017年数据保留作为后期劳动力市场背景。在重复的横截面分析中，每日使用互联网始终与更高的收入和更强的就业稳定性相关联。与每日使用相比，低于每日频率的使用者收入约低11%至20%；而在2011年和2013年，完全不使用互联网者的收入约低18%至21%。报告不使用互联网的受访者报告全年工作的可能性也低13至23个百分点。求职方面的估计显示出一种独特的模式……（原文摘要不完整）

    arXiv:2511.05294v3 Announce Type: replace-cross  Abstract: Surveys often record how frequently people use the internet without measuring the infrastructures, skills, and support systems that make digital participation possible. Using the U.S. National Longitudinal Survey of Youth 1997 cohort, we study how internet-use frequency relates to labor income, employment attachment, and job seeking after 2010. The main digital-engagement analysis uses the comparable 2011, 2013, and 2015 waves, with 2017 retained as later labor-market context. Across repeated cross sections, daily internet use consistently marks higher income and stronger employment attachment. Relative to daily use, less-than-daily use is associated with roughly 11 to 20 percent lower income, while nonuse is associated with about 18 to 21 percent lower income in 2011 and 2013. Respondents reporting no internet use are also 13 to 23 percentage points less likely to report full-year work. Job-search estimates reveal a distinct m
    

