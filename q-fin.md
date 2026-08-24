# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Recommendation Quality and the Concentration of Consumption: Experimental Evidence from Netflix](https://arxiv.org/abs/2608.21274) | 该研究表明，改进推荐算法可提升总消费并分散消费从热门到中等热门产品，而非极化消费，从而支持对中尾产品的投资。 |
| [^2] | [A Synthetic Benchmark Dataset with Endogenous Marketing Spend for Validating Marketing Mix Models](https://arxiv.org/abs/2608.21130) | 本文提出一种参数化合成数据生成器，通过内生营销支出机制生成周度零售数据集，为营销组合模型提供可验证的基准。 |
| [^3] | [Structural Estimation of Marketing Mix Model Parameters from Geo-Experiments](https://arxiv.org/abs/2608.21128) | 本文提出一种从地理实验数据中直接估计市场营销组合模型全部参数的新结构性方法，通过差分处理消除混杂因素，从而恢复真实因果效应。 |
| [^4] | [Rethinking Synthetic Scenario Realism: Compatibility, Not Fidelity, Drives Hedging Performance](https://arxiv.org/abs/2608.20842) | 本文提出兼容性概念，证明并实证显示合成数据生成器的兼容性而非单纯真实性，才是决定深度对冲策略在真实市场中表现的关键因素。 |
| [^5] | [A Multiscale Ball Test for Conditional Mean Independence](https://arxiv.org/abs/2608.20727) | 本文提出了一种多尺度球条件均值独立性检验，通过聚合局部均值对比来增强对未知空间尺度下局部和径向信号的功效，并建立了理论性质及序列数据的自助法有效性。 |
| [^6] | [Priority Transparency, Admission Chances, and Information Acquisition in School Choice](https://arxiv.org/abs/2608.20698) | 本文通过理论和实验表明，在择校中，完全公开优先级虽在理论上可能降低福利，但在实验情境下却能最大化福利，而部分公开介于两者之间，揭示了信息透明度对偏好学习动机的复杂影响。 |
| [^7] | [Calibrating Inelastic Markets to Options: The Lean Marketron and the Generalized Langevin Equation](https://arxiv.org/abs/2608.20589) | 本文通过消除对称性和绝热消去快速信号，将不可识别的十八参数市场子模型精简为稳健的九参数模型，并利用扩散相关性实现期权曲面单参数集校准。 |
| [^8] | [If It Walks Like an Arbitrage: Protocol-Agnostic Detection with Decidable Structural Equivalence](https://arxiv.org/abs/2608.20377) | 本文提出一种协议无关的套利检测方法，通过将交易轨迹归约为可判定的规范形式，无需协议特定模式即可识别套利循环，并在Rocq中机械化验证了其所有关键性质。 |
| [^9] | [Marginally Useful: An Information-Gap Identity in Conformal Prediction](https://arxiv.org/abs/2608.07479) | 本文首次揭示共形预测中残差池化技术存在一个永久的对数分数遗憾损失，其大小可量化，且无法通过数据或调整消除，并提供了金融视角的解读。 |
| [^10] | [Reaction-boundary variance and adjoint-consistent local-volatility projection](https://arxiv.org/abs/2607.05011) | 本文提出了一种基于潜在订单簿反应边界的新方法，通过操作时间方差核分离波动率模型中的结构、时钟和测度选择，并推导了长记忆情况下的闭合渐近形式。 |

# 详细

[^1]: 推荐质量与消费集中度：来自Netflix的实验证据

    Recommendation Quality and the Concentration of Consumption: Experimental Evidence from Netflix

    [https://arxiv.org/abs/2608.21274](https://arxiv.org/abs/2608.21274)

    该研究表明，改进推荐算法可提升总消费并分散消费从热门到中等热门产品，而非极化消费，从而支持对中尾产品的投资。

    

    摘要：arXiv:2608.21274v1 公告类型：交叉 摘要：我们针对Netflix推荐系统进行了一项涉及850万用户的实验，以衡量推荐技术的改进如何影响被消费的产品集合。改进提高了总消费量和用户对推荐的依赖，同时使推荐和消费从最受欢迎的标题（“超级明星”）扩散到数量更多的中等热门标题（“中尾”），对最冷门的标题（“长尾”）影响甚微。我们的结果挑战了推荐系统会使消费极化——即提高头部和尾部消费份额而牺牲中间部分——的观点，并表明随着算法改进和平台规模化，投资于中尾产品的回报会增长。

    arXiv:2608.21274v1 Announce Type: cross  Abstract: We study an experiment with 8.5 million users on Netflix's recommender system to measure how improvements in recommendation technology affect the set of products that get consumed. Improvements increase total consumption and users' reliance on recommendations while diffusing recommendations and consumption away from the most popular titles (``superstars") toward a larger number of moderately popular titles (``middle-tail"), with minimal effects on the most niche titles (``long-tail"). Our results challenge the notion that recommender systems polarize consumption -- raising the consumption shares of the head and tail at the expense of the middle -- and suggest that the returns to investing in middle-tail products grow as algorithms improve and platforms scale.
    
[^2]: 一个带有内生营销支出的合成基准数据集，用于验证营销组合模型

    A Synthetic Benchmark Dataset with Endogenous Marketing Spend for Validating Marketing Mix Models

    [https://arxiv.org/abs/2608.21130](https://arxiv.org/abs/2608.21130)

    本文提出一种参数化合成数据生成器，通过内生营销支出机制生成周度零售数据集，为营销组合模型提供可验证的基准。

    

    营销组合模型（MMMs）通过观察性时间序列估计广告的增量销售效应，但很少根据真实情况进行验证，因为在真实数据中真实情况不可观测。合成数据在原则上弥补了这一差距，但现有的生成器通常外生地生成营销支出——忽略了估计问题的核心难点，因为实际预算基于促销日历、季节和近期表现进行规划。本文提出一个参数化生成器及一个固定参考实例，用于生成一个合成周度零售数据集（156周，三个媒体渠道），其中支出源于四种文档化的协调机制——季度预算反馈、促销日历前的预期支出、计划性电视广告爆发以及算法性能追逐——并基于包含季节性、质量、价格和未观测情绪成分的需求基线。支出转化为销售效果，从而提供了一个可验证的基准。

    arXiv:2608.21130v1 Announce Type: cross  Abstract: Marketing Mix Models (MMMs) estimate the incremental sales effect of advertising from observational time series, yet they are rarely validated against ground truth, because ground truth is unobservable in real data. Synthetic data closes that gap in principle, but existing generators produce marketing spend exogenously - omitting the central difficulty of the estimation problem, since real budgets are planned around promotional calendars, seasons, and recent performance. This paper presents a parameterized generator, and a fixed reference instance, of a synthetic weekly retail dataset (156 weeks, three media channels) in which spend arises from four documented coordination mechanisms - quarterly budget feedback, anticipatory spending ahead of a promotional calendar, scheduled TV bursts, and algorithmic performance chasing - on a demand baseline with seasonal, quality, price, and unobserved sentiment components. Spend translates into in
    
[^3]: 基于地理实验的市场营销组合模型参数的结构性估计

    Structural Estimation of Marketing Mix Model Parameters from Geo-Experiments

    [https://arxiv.org/abs/2608.21128](https://arxiv.org/abs/2608.21128)

    本文提出一种从地理实验数据中直接估计市场营销组合模型全部参数的新结构性方法，通过差分处理消除混杂因素，从而恢复真实因果效应。

    

    arXiv:2608.21128v1 公告类型：交叉 摘要：市场营销组合模型（MMMs）广泛应用于营销效果测量和预算分配，但面临根本性的识别挑战：由于营销支出的内生性决策，基于观测时间序列数据的MMM估计无法恢复营销的真实因果效应。另一方面，地理实验通过随机化提供因果识别，但如何高效利用它们来校准市场营销组合模型尚不明确。我们提出了一种新颖的结构性估计方法，直接从地理实验时间序列中恢复完整的MMM参数——包括广告衰减（$\alpha$）、饱和（$\lambda$）和有效性（$\beta$）。通过差分处理组和对照组之间的结果，我们的方法消除了观测和非观测的混杂因素，同时保留了识别每个参数所需的时间变化。我们在合成数据上证明，该方法能恢复这些参数。

    arXiv:2608.21128v1 Announce Type: cross  Abstract: Marketing Mix Models (MMMs) are widely used for marketing measurement and budget allocation, but face fundamental identification challenges: due to endogenous marketing spend decisions, MMM estimation on observational time-series data cannot recover the true causal effects of marketing. On the other hand, geo-experiments provide causal identification through randomization, but it is not clear how to use them efficiently to calibrate marketing mix models. We propose a novel structural estimation approach that recovers the complete set of MMM parameters - adstock decay ($\alpha$), saturation ($\lambda$), and effectiveness ($\beta$) - directly from geo-experimental time-series. By differencing outcomes between treatment and control regions, our method eliminates observed and unobserved confounding factors while preserving the temporal variation that identifies each parameter. We demonstrate on synthetic data that this approach recovers th
    
[^4]: 重新思考合成场景的真实性：兼容性而非保真度驱动对冲表现

    Rethinking Synthetic Scenario Realism: Compatibility, Not Fidelity, Drives Hedging Performance

    [https://arxiv.org/abs/2608.20842](https://arxiv.org/abs/2608.20842)

    本文提出兼容性概念，证明并实证显示合成数据生成器的兼容性而非单纯真实性，才是决定深度对冲策略在真实市场中表现的关键因素。

    

    arXiv:2608.20842v1 公告类型：新 摘要：深度对冲是一种基于数据驱动的学习方法，用于学习对冲策略。它依赖于合成价格路径生成器，因为真实市场数据在训练中往往有限。现有方法主要基于真实性（即它们捕捉真实市场统计特性的程度）来评估此类生成器，但真实性与对冲表现之间的关系仍不明确。在本工作中，我们从决策中心视角引入合成数据在深度对冲中的概念，基于兼容性。兼容性衡量在合成场景上训练的策略在真实市场中的有效程度。我们从理论上证明：1) 对冲表现可分解为学习误差和兼容性差距；2) 真实性和兼容性可能发散。在实证上，我们发现对冲表现并非仅由真实性决定，而是由生成器与对冲者之间的对齐以及任务结构共同主导。

    arXiv:2608.20842v1 Announce Type: new  Abstract: Deep hedging is a data-driven approach to learn hedging strategies. It relies on synthetic price paths generator, as real market data is often limited for training. Existing approaches primarily evaluate such generators based on realism, i.e., how well they capture statistical properties of real markets, but the relationship between realism and hedging performance remains unclear. In this work, we introduce a decision-centric perspective on synthetic data for deep hedging based on the notion of compatibility. Compatibility measures the extent to which strategies trained on synthetic scenarios remain effective in the true market. We theoretically show that 1) hedging performance decomposes into learning error and a compatibility gap, and 2) realism and compatibility can diverge. Empirically, we find that hedging performance is governed not by realism alone, but by the alignment between the generator and the hedger, together with task stru
    
[^5]: 多尺度球检验用于条件均值独立性

    A Multiscale Ball Test for Conditional Mean Independence

    [https://arxiv.org/abs/2608.20727](https://arxiv.org/abs/2608.20727)

    本文提出了一种多尺度球条件均值独立性检验，通过聚合局部均值对比来增强对未知空间尺度下局部和径向信号的功效，并建立了理论性质及序列数据的自助法有效性。

    

    arXiv:2608.20727v1 公告类型：交叉 摘要：当偏离仅局限于多元预测变量空间的有界部分且相关空间尺度未知时，条件均值独立性检验可能失去功效。我们提出了一种多尺度球条件均值独立性（MBCMI）检验，该检验在预测变量集中每个数据点为中心的球上，聚合结果变量中支持加权的局部均值对比。固定网格理论识别了总体目标，建立了对网格可见替代方案的相合性，并推导了由球平滑均值偏离主导的Pitman局部功效极限。对于序列数据，我们建立了在具有条件符号对称创新的稳定有限阶自回归中，可行的递归符号自助法的有效性。与应用对齐的序列零假设实验拒绝率为4.25%。MBCMI被证明对局部和径向信号最强。预测变量定律实验表明这些结论并非人为产物。

    arXiv:2608.20727v1 Announce Type: cross  Abstract: Tests of conditional mean independence can lose power when departures are confined to a bounded part of a multivariate predictor space and the relevant spatial scale is unknown. We propose a Multiscale Ball Conditional Mean Independence (MBCMI) test that aggregates support-weighted local mean contrasts in an outcome variable across balls centered on each data point in a predictor set. Fixed-grid theory identifies the population target, establishes consistency for grid-visible alternatives, and derives a Pitman local-power limit governed by the ball-smoothed mean departure. For serial data, feasible recursive-sign-bootstrap validity for stable finite-order autoregressions with conditionally sign-symmetric innovations is established. Application-aligned serial null experiments reject 4.25% of the time. MBCI is demonstrated to be strongest for local and radial signals. Predictor-law experiments show that these conclusions are not an artef
    
[^6]: 择校中的优先级透明度、录取机会与信息获取

    Priority Transparency, Admission Chances, and Information Acquisition in School Choice

    [https://arxiv.org/abs/2608.20698](https://arxiv.org/abs/2608.20698)

    本文通过理论和实验表明，在择校中，完全公开优先级虽在理论上可能降低福利，但在实验情境下却能最大化福利，而部分公开介于两者之间，揭示了信息透明度对偏好学习动机的复杂影响。

    

    我们从理论和实验两方面研究了学生优先级和录取机会的透明度如何影响他们在择校和大学录取中获取自身偏好信息的动机。在模型中，未获取信息的学生基于共同先验选择学校。当他们了解自身偏好后，其选择变得更加多样化，这释放了热门学校的名额。知道自己优先级高的学生有更强的学习动机，因为他们能更有效地利用所学信息，而知道自己优先级低的学生则会受到抑制。完全公开优先级会将学习行为集中在高优先级学生中。通过合并优先级，部分公开将学习动机扩展到被合并的学生群体，并带来更高的福利。然而，在实验室中，完全公开反而产生最高福利，其次是部分公开，然后是不公开，因为——

    arXiv:2608.20698v1 Announce Type: new  Abstract: We study, theoretically and experimentally, how transparency about students' priorities and admission chances shapes their incentives to acquire information about their own preferences in school choice and college admissions. In the model, uninformed students choose schools based on a common prior. When they learn their own preferences, their choices become more heterogeneous, which frees up seats at popular schools. Students who know they have high priority have stronger incentives to learn because they can more readily act on what they learn, whereas students who know they have low priority are discouraged. Full priority disclosure concentrates learning among high-priority students. By pooling priorities, partial disclosure spreads learning incentives to pooled students and yields higher welfare. In the laboratory, however, full disclosure yields the highest welfare instead, followed by partial disclosure, and then no disclosure, becau
    
[^7]: 校准非弹性市场至期权：精益市场子与广义朗之万方程

    Calibrating Inelastic Markets to Options: The Lean Marketron and the Generalized Langevin Equation

    [https://arxiv.org/abs/2608.20589](https://arxiv.org/abs/2608.20589)

    本文通过消除对称性和绝热消去快速信号，将不可识别的十八参数市场子模型精简为稳健的九参数模型，并利用扩散相关性实现期权曲面单参数集校准。

    

    arXiv:2608.20589v1 公告类型：新 摘要：\cite{HalperinItkin2025Mark}中的市场子模型及其在\cite{HalperinItkinMarketron2}中的期权定价扩展存在结构性的不可识别性问题：一个十八参数空间使求解器陷入次优局部最小值，并使经济量无法测量。通过去除精确缩放规范与符号对称性，基于明确标准冻结非金融参数，并绝热消除快速隐藏信号，我们推导出一个稳健的九参数简化模型。具有零空间为空的高斯-牛顿海森矩阵和流形边界分析确认，简化核心不携带精确对称性，且不允许进一步简化。流动与收益创新之间的扩散相关性捕捉了短期限偏斜。从物理测度到风险中性测度的分阶段校准，在SPX期权上展示，用单一参数集拟合整个曲面。同样的简化将楔形差转化为...

    arXiv:2608.20589v1 Announce Type: new  Abstract: The Marketron model of \cite{HalperinItkin2025Mark} and its option pricing extension in \cite{HalperinItkinMarketron2} suffer from structural non-identifiability: an eighteen-parameter space traps solvers in suboptimal local minima and renders economic quantities unmeasurable. By removing exact scaling gauges and sign symmetries, freezing non-financial parameters by explicit criteria, and adiabatically eliminating the fast hidden signal, we derive a robust nine-parameter reduced model. A Gauss-Newton Hessian with empty null space and a manifold-boundary analysis confirm that the reduced core carries no exact symmetry and admits no further reduction. A diffusive correlation between flow and return innovations captures the short-maturity skew. A staged calibration from the physical measure to the risk-neutral measure, illustrated on SPX options, fits the whole surface with a single parameter set. The same reduction turns the wedge between 
    
[^8]: 若它行走如套利：基于可判定结构等价性的协议无关检测

    If It Walks Like an Arbitrage: Protocol-Agnostic Detection with Decidable Structural Equivalence

    [https://arxiv.org/abs/2608.20377](https://arxiv.org/abs/2608.20377)

    本文提出一种协议无关的套利检测方法，通过将交易轨迹归约为可判定的规范形式，无需协议特定模式即可识别套利循环，并在Rocq中机械化验证了其所有关键性质。

    

    以太坊交易具有规范的结构形式。每个执行轨迹被构建为按调用帧嵌套分组的代币转移抽象语法树，并通过包含15条规则的收敛重写系统归约为唯一规范形式。该系统具有终止性、可靠性和合流性，且对资金流诱导的结构等价性是可判定的。所有五个性质均在Rocq中机械化验证，零待证义务。规范形式使资金流的结构性问题可判定，为策略家族分类、机器人指纹识别和基于等价性的归因开辟了道路。本文展示了规范形式在套利检测中的应用：循环在不动点处出现并从规范形式中读取，无需协议特定模式。该流程仅依赖标准ERC代币和WETH ABI，不依赖协议特定事件，因此同一二进制代码可在Arbitrum和B（原文截断）上无修改运行。

    arXiv:2608.20377v1 Announce Type: cross  Abstract: Ethereum transactions admit a canonical structural form. Each execution trace is built into an abstract syntax tree of token transfers grouped by call-frame nesting and reduced by a convergent term rewriting system of 15 rules to a unique canonical form. The system is terminating, sound, and confluent, and the induced structural equivalence on fund flows is decidable. All five properties are mechanized in Rocq with zero admitted obligations. The canonical form makes structural questions about fund flows decidable, opening the way to strategy-family classification, bot fingerprinting, and equivalence-based attribution. In this paper, we demonstrate the canonical form on arbitrage detection: cycles emerge at fixpoint and are read off the canonical form, with no protocol-specific patterns. The pipeline depends only on the standard ERC token and WETH ABIs and no protocol-specific events, so the same binary runs unmodified on Arbitrum and B
    
[^9]: 边际有用：共形预测中的信息缺口恒等式

    Marginally Useful: An Information-Gap Identity in Conformal Prediction

    [https://arxiv.org/abs/2608.07479](https://arxiv.org/abs/2608.07479)

    本文首次揭示共形预测中残差池化技术存在一个永久的对数分数遗憾损失，其大小可量化，且无法通过数据或调整消除，并提供了金融视角的解读。

    

    arXiv:2608.07479v2 公告类型：替换 摘要：共形预测被誉为一种更正式、更严谨的为预测添加不确定性的方法。本注释的唯一目的是指出，在残差池化（共形预测应用中最常用的技术）情况下，严谨性是一把双刃剑。无条件覆盖保证的存在性毋庸置疑，但我们明确（我们认为这是首次）指出，还存在一个相反的保证：对数分数遗憾的永久性损失，任何数据量或调整都无法后续减少。我们给出了这一牺牲的确切大小，并将其金融解读为一种“神谕对手”的增长率，该对手针对使用残差池化的人设定的赔率进行下注。

    arXiv:2608.07479v2 Announce Type: replace  Abstract: Conformal prediction has been touted as a more formal, rigorous approach to adding uncertainty to a forecast. The sole objective of this note is to point out that rigor cuts both ways in the case of residual pooling, the technique used in the vast majority of conformal prediction applications. The fact that unconditional guarantee of coverage is provided is not in question, but we make clear, we believe for the first time, that there is an opposing guarantee too: a permanent gambit of logarithmic-score regret which no amount of data or tuning can subsequently reduce. We give the exact size of the sacrifice, and a financial reading of it as the growth rate of an oracle adversary betting against odds set by someone using residual pooling.
    
[^10]: 反应边界方差与伴随一致的局部波动率投影

    Reaction-boundary variance and adjoint-consistent local-volatility projection

    [https://arxiv.org/abs/2607.05011](https://arxiv.org/abs/2607.05011)

    本文提出了一种基于潜在订单簿反应边界的新方法，通过操作时间方差核分离波动率模型中的结构、时钟和测度选择，并推导了长记忆情况下的闭合渐近形式。

    

    我们为潜在订单簿反应边界推导了一个操作时间方差核，并利用它来分离通常在日历时间波动率模型中被合并的三个对象：结构边界累积量、时钟投影和定价测度选择。反应边界是买卖失衡场的一个零点。对于局部线性订单簿，有符号订单流扰动通过阻尼阿贝尔响应核移动该零点，因此边界增量的方差作为有限尺度格林函数累积量获得，而非作为原始扩散系数引入。对于指数$0<\gamma<1$的长记忆强迫，操作方差具有一个闭合渐近形式，涉及有效有符号强迫强度、流动性斜率、弹性、记忆和操作粗粒化尺度。确定性活动时钟给出了基准局部波动率投影。更一般、非唯一的时钟生成...

    arXiv:2607.05011v3 Announce Type: replace  Abstract: We derive an operational-time variance kernel for a latent-order-book reaction boundary and use it to separate three objects usually collapsed in calendar-time volatility models: a structural boundary cumulant, a clock projection, and a pricing-measure choice. The reaction boundary is the zero of a bid--ask imbalance field. For a locally linear book, signed order-flow perturbations displace this zero through a damped Abel response kernel, so the variance of boundary increments is obtained as a finite-scale Green-function cumulant rather than introduced as a primitive diffusion coefficient. For long-memory forcing with exponent $0<\gamma<1$, the operational variance has a closed asymptotic form involving effective signed-forcing intensity, liquidity slope, resilience, memory, and operational coarse-graining scale. A deterministic activity clock gives the benchmark local-volatility projection. More general, non-unique clocks generate c
    

