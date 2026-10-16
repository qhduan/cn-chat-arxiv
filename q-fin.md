# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Extending the Oracle Approximating Shrinkage to General Distributions](https://arxiv.org/abs/2610.22144) | 该论文通过一个由四阶联合累积量构建、在高斯情形下自动消失的修正项，将原本仅适用于高斯数据的OAS协方差收缩估计器扩展到任意具有有限四阶矩的重尾分布，保证收缩强度严格落在[0,1]区间，并在非高斯数据下达到与Ledoit-Wolf估计器相当的性能。 |
| [^2] | [Momentum, Commentary Tone, and Earnings Surprise in LLM Financial Forecasts](https://arxiv.org/abs/2610.22043) | 本研究通过独立操纵动量、评论语气和EPS意外的合成财报实验发现，在大语言模型的金融预测中，非基本面的评论语气和价格动量因素对模型输出的影响往往超过实际的每股收益意外本身，揭示了LLM金融预测易受非信息性输入干扰的问题。 |
| [^3] | [Algorithmic Collusion Without Deterrence: Auditing the Punishment Mechanism in Q-Learning Pricing Agents](https://arxiv.org/abs/2610.21867) | 该研究审计了Q学习定价智能体的“惩罚机制”，发现在关闭学习后脉冲响应模式无法证明威慑机制的存在——超竞争价格可能源于覆盖整个状态空间的单一吸引子（过度决定），而非选择性的报复性惩罚。 |
| [^4] | [Where Backtests and Live Trading Part Ways: Six Defects from One Intraday System](https://arxiv.org/abs/2610.21846) | 本文通过将一个美股日内交易系统的实盘运行与其回测重放对比，发现了六个导致两者偏差的缺陷——三个源于数据、两个源于代码、一个源于配置，并为每个缺陷提供了症状、错误规模、成因及防范措施。 |
| [^5] | [Feeling the generalised rough SABR boundary](https://arxiv.org/abs/2610.21841) | 本文深入分析了金融业广泛使用的广义粗糙SABR波动率模型在原点处的边界行为，重点研究了该模型能否到达原点以及到达原点的速度。 |
| [^6] | [When Does "Post-Call" Begin? An Availability Contract for Earnings-Call NLP Benchmarks](https://arxiv.org/abs/2610.21659) | 论文提出“可用性契约”框架，用于诊断和防范财报电话会议NLP基准测试中的事件内时间错配问题，确保预测目标不会早于模型输入信息可用的时间点。 |
| [^7] | [Pricing the Unprovable: Ambiguity, Turnstile Runs, and the Verifiability Discount in Crypto Protocols](https://arxiv.org/abs/2610.21540) | 该论文提出“可验证性折价”概念及VDISC可复现分析框架，通过旋转门模型与Zcash/Orchard的链上实证研究表明：当加密货币隐私协议无法证明漏洞是否曾被利用时，市场会对其价值进行折价定价。 |
| [^8] | [How Much Screening Buys a Second? Speed, Selection, and the Maker's Spread](https://arxiv.org/abs/2610.21510) | 该论文提出一个盈亏平衡条件，精确量化了较慢的做市商需要多高的知情流筛选率才能匹敌更快的未筛选交易场所，该结论仅需价格变动可积即可在跳跃、原子与无限方差下成立，并以USDC-WETH资金池数据实证表明将报价频率从十二秒缩短到一秒可削减84.5%的预期冲击。 |
| [^9] | [NEST: Stale-Robust Tokenized-Stock Measurement for a Crypto-Treasury LETF Stack](https://arxiv.org/abs/2610.21491) | 论文提出NEST协议，利用共享历史恒等式、方差时钟界限、部分决定指数（PDI）及多重审计，将代币化股票标记对次日开盘缺口的机械性拟合与真实条件度量价值分离，从而稳健评估其对加密财库LETF链路盘外度量的增量价值。 |
| [^10] | [Long-Horizon Investment with Delayed Information in a Mean-Reverting Market](https://arxiv.org/abs/2610.21447) | 本文在Ornstein-Uhlenbeck模型中显式求解了带信息延迟的长期投资问题，发现信息延迟会将确定性等价的增长从二次降为线性，且将完全信息策略直接投影到延迟信息上会导致无穷大的指数损失。 |
| [^11] | [Fundamental Theorem of Discrete Asset Pricing](https://arxiv.org/abs/2610.21442) | 本文在连续时间离散价格框架下建立了资产定价基本定理，证明了无套利、无消失风险自由午餐、无自由午餐与等价局部鞅测度存在性的等价性，且证明完全避开Itô积分，仅使用有限Stieltjes和。 |
| [^12] | [The end-of-month option in futures contracts and the Gittins index of holding the underlying](https://arxiv.org/abs/2610.21196) | 本文利用Bank–El Karoui表示定理将期货合约的月末交割期权刻画为持有标的资产并收取便利收益的基特金斯指数问题，期货价格即为使该项目价值等于现货价格的退休回报，并通过Dinkelbach迭代将定价计算化归为一系列标准的最优停时问题。 |
| [^13] | [Pricing American Commodity Options Under the Schwartz Two-Factor Model Using the Alternating Direction Explicit (ADE) Method](https://arxiv.org/abs/2610.21145) | 本文提出用双扫掠交替方向显式（ADE）有限差分方法在Schwartz双因子模型下定价美式商品期权，该方法兼具显式格式的高效率与隐式格式的无条件稳定性，并通过与闭式解、蒙特卡洛及Longstaff-Schwartz方法的对比验证了其数值精度。 |
| [^14] | [StochLOB: Heavy-Tailed Score Matching for Robust Limit Order Book Trend Prediction](https://arxiv.org/abs/2610.21081) | 该论文提出StochLOB训练框架，用跳跃Lévy噪声和截断α稳定噪声等重尾非高斯扩散过程替代传统的布朗高斯噪声来加噪训练数据，以更好地捕捉限价订单簿订单流的重尾与跳跃特性，从而提升LOB趋势预测的稳健性，同时推理阶段保持确定性预测器。 |
| [^15] | [Loss-versus-rebalancing with external proportional transaction costs](https://arxiv.org/abs/2610.21057) | 该论文证明，当外部市场存在比例交易成本时，以按外部买卖价交易的自融资策略为基准，自动做市商资金池的损失与再平衡之差（LVR）为零，并通过保留费用的效用无差异资金池与几何平均资金池实现特定的递减库存计划和恒定不交易边界。 |
| [^16] | [$q_x$-Bass Martingales: Optimality, Existence and Convergence](https://arxiv.org/abs/2610.20920) | 本文引入 $q_x$-Bass 鞅，证明其为弱最优传输问题的最优解并建立强对偶性，据此提出计算算法并在紧性假设下证明其收敛性，从而确立了此类鞅的存在性。 |
| [^17] | [Neural Estimation of Irreversibility in Real and Simulated Limit Order Book Paths](https://arxiv.org/abs/2610.16151) | 该论文首次将时间不可逆性的度量从标量金融序列拓展到多变量限价订单簿路径，通过神经网络估计发现订单排序本身贡献了显著且在所有十二只股票上稳定为正的不可逆性增量。 |
| [^18] | [Allocative Efficiency and the Returns to Innovation](https://arxiv.org/abs/2610.14969) | 本文揭示在不完全竞争市场中，创新所引发的配置效率变化无法被创新者自身捕获，导致美国上市企业研发的社会回报高达其私人回报的6.24倍。 |
| [^19] | [AlphaRLM: Aligning Language Models for Literature-Grounded Symbolic Factor Discovery](https://arxiv.org/abs/2610.14959) | AlphaRLM是一个强化学习框架，通过文献引导算子组装（LEGO）提供语义先验，并借助可执行有效性与分组收益分离等奖励将生成策略与市场反馈对齐，从而训练大语言模型直接从金融文献中构造可执行的符号化Alpha因子。 |
| [^20] | [RICE-Alpha: Reliability-Informed Correction with Event Graphs for LLM-Agent Stock Forecasting](https://arxiv.org/abs/2609.34004) | 该论文提出RICE-Alpha框架，通过多层记忆层和类型化事件智能体构建事件图，将历史感知的多视图基础Alpha与基于历史事件延续的可靠性校准残差校正相结合，从而在时点约束下提升LLM智能体的股票预测能力。 |
| [^21] | [On the asymptotic shape of quantile surfaces](https://arxiv.org/abs/2609.31345) | 本文证明了对数正态随机变量线性组合生成的分位数曲面在左尾全局凹、右尾全局凸，并具有尺度与形状渐近分离的特性。 |
| [^22] | [Principal component error in high-dimensional factor models](https://arxiv.org/abs/2609.20550) | 本文将高维因子模型中主成分估计的误差分解为可从数据估计的“子空间外误差”与无法从数据估计的“子空间内误差”两个可解释项，并证明二者在变量数增长而样本量有限时具有几乎必然的渐近极限。 |
| [^23] | [Global Multi-Maturity SPX-VIX Calibration Beyond Markovian Stitching](https://arxiv.org/abs/2609.04087) | 本文建立了跨多期限SPX-VIX联合校准的全局框架，证明局部与全局可行性等价、但拼接测度仅为全局可行测度的真子集，并指出相对熵在马尔可夫参考下只能选出拼接的最小信息补全，识别非马尔可夫依赖需要额外的跨期信息。 |
| [^24] | [DYSANOS Generative Dynamic Smooth Arbitrage-free Non-parametric Option Surfaces](https://arxiv.org/abs/2608.12587) | 本文提出首个生成式无套利期权曲面模型DYSANOS，能够生成未来多年每日期权价格路径，并验证其优于传统隐含波动率PCA模型。 |
| [^25] | [Real-time identification of the onset of financial rogue waves](https://arxiv.org/abs/2606.31475) | 该论文将光学与流体中的“怪波”类比引入金融波动率分析，利用带克尔非线性势的薛定谔方程研究VIX等波动率指数，发现极端事件临近时出现安德森局域化迹象，且系统最小特征值的数值梯度在极端事件开始时可靠激增，从而实现对金融极端事件起点的实时识别。 |
| [^26] | [Choice of Collateral Currency in Differential Swaps](https://arxiv.org/abs/2603.07863) | 本文研究了一类参考隔夜利率后视平均值的差额互换的估值与对冲，将抵押账户扩展为可用不同于合约现金流的货币计价，并给出了显式的定价与对冲策略。 |

# 详细

[^1]: 将Oracle近似收缩估计（OAS）扩展至一般分布

    Extending the Oracle Approximating Shrinkage to General Distributions

    [https://arxiv.org/abs/2610.22144](https://arxiv.org/abs/2610.22144)

    该论文通过一个由四阶联合累积量构建、在高斯情形下自动消失的修正项，将原本仅适用于高斯数据的OAS协方差收缩估计器扩展到任意具有有限四阶矩的重尾分布，保证收缩强度严格落在[0,1]区间，并在非高斯数据下达到与Ledoit-Wolf估计器相当的性能。

    

    arXiv:2610.22144v1 公告类型：cross 摘要：Oracle近似收缩（OAS）估计器为将样本协方差矩阵向缩放单位阵目标进行正则化提供了一种闭式、数据驱动的收缩强度，但其推导假设数据服从高斯分布。在实际应用中，金融收益率、神经信号以及许多其他高维数据集往往呈现重尾和非零峰度特征，违背了这一假设。我们通过引入一个由数据四阶联合累积量构建的修正项 α，将OAS扩展到任何具有有限四阶矩的分布。该修正在高斯情形下会消失，从而将原OAS公式作为特例恢复；它拥有一个一致的 O(np) 复杂度样本估计量，并且无论是总体累积量还是其样本对应量，均严格满足收缩强度 ρ ∈ [0, 1]。仿真实验和股票收益率实证研究表明，该修正项在非高斯数据下缩小了OAS与Ledoit-Wolf估计器之间的性能差距，OAS扩展版本（OAS-Extended）……

    arXiv:2610.22144v1 Announce Type: cross  Abstract: The Oracle Approximating Shrinkage (OAS) estimator provides a closed-form, data-driven shrinkage intensity for regularizing sample covariance matrices toward a scaled identity target, but its derivation assumes Gaussian data. In practice, financial returns, neural signals, and many other high-dimensional datasets exhibit heavy tails and non-zero kurtosis, violating this assumption. We extend OAS to any distribution with finite fourth moments by incorporating a correction term $\alpha$ built from the fourth-order joint cumulants of the data. The correction vanishes under Gaussianity, recovering the OAS formula as a special case; it admits a consistent $O(np)$ sample estimator and satisfies $\rho \in [0, 1]$ exactly for both the population cumulant and its sample analogue. Simulations and an equity-returns study confirm that the correction closes the gap between OAS and the Ledoit-Wolf estimator under non-Gaussian data, with OAS-Extended
    
[^2]: 大语言模型金融预测中的动量、评论语气与盈利意外

    Momentum, Commentary Tone, and Earnings Surprise in LLM Financial Forecasts

    [https://arxiv.org/abs/2610.22043](https://arxiv.org/abs/2610.22043)

    本研究通过独立操纵动量、评论语气和EPS意外的合成财报实验发现，在大语言模型的金融预测中，非基本面的评论语气和价格动量因素对模型输出的影响往往超过实际的每股收益意外本身，揭示了LLM金融预测易受非信息性输入干扰的问题。

    

    是什么输入驱动语言模型的金融预测？我们构建了合成的财报公告，独立地反转20日动量、非信息性的评论语气，以及固定的实际每股收益（EPS）是否超越被操纵的共识预期。来自四个模型家族（基础参数量为4.9亿至76.1亿）的六个指令/对话检查点转换版本，在直接回答的Q8_0协议下各接收2,016条提示。实验设计包含24个数字区块、六组评论配对、盈利与亏损情形、上升与下降的年度EPS，以及两种意外幅度。A/B答案分配与U/D（上涨/下跌）覆盖每个转换版本的全部384个主要情景。我们估计的是条件答案前缀分数的匹配变化，而非校准后的市场概率。在这些检查点中，主要EPS对比的范围为-0.19至86.80个百分点。在六个模型中的五个里，主要符号语气对比的幅度超过了EPS对比，动量对比在其中五个模型中也超过了EPS对比。

    arXiv:2610.22043v1 Announce Type: cross  Abstract: Which inputs drive a language model's financial forecast? We construct synthetic earnings announcements that independently reverse 20-day momentum, non-informational commentary tone, and whether fixed actual EPS beats a manipulated consensus. Six instruction/chat checkpoint conversions from four families (0.49-7.61B base parameters) receive 2,016 prompts each under a direct-answer Q8_0 protocol. The design includes 24 numerical blocks, six commentary pairs, profits and losses, rising and falling annual EPS, and two surprise sizes. Both A/B answer assignments and U/D cover all 384 primary scenarios per conversion. We estimate matched changes in conditional answer-prefix scores, not calibrated market probabilities. Across these checkpoints, the primary EPS contrast ranges from -0.19 to 86.80 percentage points. The magnitude of the primary signed tone contrast exceeds that of EPS in 5 of six models, and momentum exceeds it in 5. Natural-w
    
[^3]: 无威慑的算法合谋：审计Q学习定价智能体中的惩罚机制

    Algorithmic Collusion Without Deterrence: Auditing the Punishment Mechanism in Q-Learning Pricing Agents

    [https://arxiv.org/abs/2610.21867](https://arxiv.org/abs/2610.21867)

    该研究审计了Q学习定价智能体的“惩罚机制”，发现在关闭学习后脉冲响应模式无法证明威慑机制的存在——超竞争价格可能源于覆盖整个状态空间的单一吸引子（过度决定），而非选择性的报复性惩罚。

    

    独立的表格型Q学习智能体在模拟的伯川德双寡头市场中重新定价时，会收敛到超竞争价格，而开创性的研究证据将此归因于一种学到的“奖励-惩罚”机制：若你削价对抗对手，对手就会报复，因此偏离定价永远不划算。正是这种机制性解读，将高价格转变为卡特尔行为，并使其成为监管目标。我们复现了Calvano等人（2020）的环境，并确认了他们在共同训练智能体对中的脉冲响应模式。随后我们在冻结策略上审计这一模式——即关闭学习过程——发现该模式本身并不能识别出这一机制：回报是过度决定的，而非选择性的。在十个收敛的智能体对中的九个（1024次中的914次）里，单一吸引子的吸引盆覆盖了整个状态空间，因此任何扰动都会返回，无论其起因是什么，而且强制提价会触发与削价完全相同的报复模式。共同训练真正产生的，是一个狭窄的、仅针对特定智能体对的吸引盆……

    arXiv:2610.21867v1 Announce Type: new  Abstract: Independent tabular Q-learning agents repricing in a simulated Bertrand duopoly converge to supracompetitive prices, and the seminal evidence attributes this to a learned reward-punishment scheme: undercut your rival and it retaliates, so deviation never pays. This mechanistic reading is what turns high prices into a cartel, and into a regulatory target. We replicate the environment of Calvano et al. (2020) and confirm their impulse-response pattern within co-trained pairs. We then audit the pattern on frozen policies, with learning off, and find that on its own it does not identify a mechanism: the return is over-determined rather than selective. In nine of ten converged pairs (914 of 1,024) a single attractor's basin is the entire state space, so every perturbation returns whatever provoked it, and a forced price increase triggers the same retaliatory pattern as an undercut. What co-training produces is a narrow, dyad-specific basin of
    
[^4]: 回测与实盘交易的分道扬镳之处：来自一个日内交易系统的六个缺陷

    Where Backtests and Live Trading Part Ways: Six Defects from One Intraday System

    [https://arxiv.org/abs/2610.21846](https://arxiv.org/abs/2610.21846)

    本文通过将一个美股日内交易系统的实盘运行与其回测重放对比，发现了六个导致两者偏差的缺陷——三个源于数据、两个源于代码、一个源于配置，并为每个缺陷提供了症状、错误规模、成因及防范措施。

    

    日内策略的回测结果可能会偏离实盘系统的实际表现；在我们的案例中，这种偏差从未在汇总统计数据中显现。我们报告了在实盘运行一个美股日内交易系统、并将实盘交易日与其回测重放进行对比的过程中发现的六个缺陷。其中三个来自数据：默认数据源仅携带真实开盘成交量的一小部分、实盘订单从未使用过经股息调整的历史价格，以及实盘系统下载了但回测从未使用过的数据。两个来自代码：一个仓位计算步骤超出了实盘账户强制执行的限额，以及UTC时间戳被误读为纽约时间。一个来自配置：已开启但从未传达至下单代码的设置项。针对每个缺陷，我们都给出了症状、错误规模、成因，以及当前用于防范该问题的测试或控制措施。这六个缺陷全都出现在本应一致的两条执行路径产生分歧的地方。最后，我们给出了一个包含十项内容的检查清单。

    arXiv:2610.21846v1 Announce Type: new  Abstract: Backtests of intraday strategies can drift away from what the live system actually does; in our case, the drift never showed in summary statistics. We report six defects found while running one intraday US equity system live and comparing its live days with their backtest replay. Three came from data: a default feed that carried a small fraction of real opening volume, historical prices adjusted for dividends that live orders never see, and data the live system downloaded but the backtest never did. Two came from code: a sizing step that exceeded a limit the live account enforces, and UTC timestamps read as New York time. One came from configuration: settings that were switched on but never reached the code that places orders. For each we give the symptom, the size of the error, the cause and the test or control that now guards against it. All six sat where two paths that should agree had drifted apart. We close with a ten-point checklis
    
[^5]: 感知广义粗糙SABR模型的边界

    Feeling the generalised rough SABR boundary

    [https://arxiv.org/abs/2610.21841](https://arxiv.org/abs/2610.21841)

    本文深入分析了金融业广泛使用的广义粗糙SABR波动率模型在原点处的边界行为，重点研究了该模型能否到达原点以及到达原点的速度。

    

    我们对金融行业广泛使用的SABR波动率模型的广义粗糙版本进行了深入分析。我们特别关注该模型在原点处的行为，即其是否能够到达原点，以及可能以多快的速度到达原点。

    arXiv:2610.21841v1 Announce Type: new  Abstract: We provide an in-depth analysis of a generalised rough version of the SABR volatility model, widely used in the financial industry. We specifically focus on its behaviour at the origin, whether it can reach it and how fast it may do so.
    
[^6]: “会后”从何时开始？财报电话会议NLP基准的可用性契约

    When Does "Post-Call" Begin? An Availability Contract for Earnings-Call NLP Benchmarks

    [https://arxiv.org/abs/2610.21659](https://arxiv.org/abs/2610.21659)

    论文提出“可用性契约”框架，用于诊断和防范财报电话会议NLP基准测试中的事件内时间错配问题，确保预测目标不会早于模型输入信息可用的时间点。

    

    企业财务信息披露通常随时间逐步展开，公告、准备好的陈述与问答讨论环节会跨越多个高频收益区间。因此，当预测目标在模型的文本或音频输入尚不可能完整之前就已开始时，金融NLP基准测试便会出现事件内的时间错配问题。按时间顺序划分训练集与测试集并不能避免这种错配，因为它发生在单个事件内部，而非样本之间。我们提出了“可用性契约”这一概念，要求基准测试明确说明输入边界、输入可用性的证据、结果的起始时间，以及该证据所能支持的结论。我们将该框架应用于来自24家公司的552个财报电话会议。对于每一次电话会议，我们固定收益终点，并比较三种时间点：前一交易日收盘价、基于文件记录的会议开始时间代理，以及基于重建音频结束时间加规定处理时间宽限的完成时间代理。在此基……（原文摘要在此处截断）

    arXiv:2610.21659v1 Announce Type: cross  Abstract: Corporate financial disclosures often unfold over time, with announcements, prepared remarks, and discussion spanning multiple high-frequency return intervals. Financial NLP benchmarks can therefore suffer from a within-event timing mismatch when the prediction target begins before the textual or audio input used by a model could be complete. Chronological train--test splits do not prevent this mismatch, because it occurs inside individual events rather than across samples. We introduce an Availability Contract, which requires a benchmark to state the input boundary, evidence for input availability, the outcome start, and the claims supported by that evidence. We apply this framework to 552 earnings calls from 24 firms. For each call, we hold the return endpoint fixed and compare the previous close, a documentary call-start proxy, and a completion proxy based on reconstructed audio end plus a stipulated processing allowance. At this ba
    
[^7]: 为不可证明之物定价：加密协议中的模糊性、旋转门式挤兑与可验证性折价

    Pricing the Unprovable: Ambiguity, Turnstile Runs, and the Verifiability Discount in Crypto Protocols

    [https://arxiv.org/abs/2610.21540](https://arxiv.org/abs/2610.21540)

    该论文提出“可验证性折价”概念及VDISC可复现分析框架，通过旋转门模型与Zcash/Orchard的链上实证研究表明：当加密货币隐私协议无法证明漏洞是否曾被利用时，市场会对其价值进行折价定价。

    

    隐私保护协议可以在未事先确认漏洞是否已被利用的情况下完成修复，即使没有任何盗窃报告且总供应量得到保护。我们将这种不确定性差距所导致的潜在价值下降称为“可验证性折价”。一个旋转门模型展示了早期提款如何将可能出现的资金缺口转移给剩余持有者，以及为什么随着风险敞口的扩大，可信证据会变得愈发有价值。我们提出了VDISC，一个可复现的框架，分别记录报告损失、风险敞口价值、公开证据、市场回报和资金池流动。该框架将Orchard的分阶段披露与重构的链上资金流、更早的Zcash案例以及包含40个事件的安全性面板相结合。Orchard的实证结果符合预测方向：在完成修复和供应量保障之后，市场反应较为温和，而最大的负回报与屏蔽资金池的整体净流出，恰好对应于官方承认“无法排除过往滥用可能性”的时点。

    arXiv:2610.21540v1 Announce Type: new  Abstract: Privacy-preserving protocols can repair a vulnerability without establishing whether it was exploited beforehand, even when no theft is reported and aggregate supply is protected. We call the potential value reduction from this gap the verifiability discount. A turnstile model shows how early withdrawal can shift a possible shortfall to remaining holders and why credible evidence can become more valuable as exposure grows. We introduce VDISC, a reproducible framework that records reported loss, exposed value, public evidence, returns, and pool flows separately. It combines Orchard's staged disclosures with reconstructed on-chain flows, earlier Zcash cases, and a 40-event security panel. Orchard follows the predicted direction: market responses were muted after repair and supply assurance, while the largest negative return and aggregate shielded-pool net outflow aligned with the acknowledgment that past abuse could not be ruled out. These
    
[^8]: 多少筛选才能换回一秒？速度、选择与做市商价差

    How Much Screening Buys a Second? Speed, Selection, and the Maker's Spread

    [https://arxiv.org/abs/2610.21510](https://arxiv.org/abs/2610.21510)

    该论文提出一个盈亏平衡条件，精确量化了较慢的做市商需要多高的知情流筛选率才能匹敌更快的未筛选交易场所，该结论仅需价格变动可积即可在跳跃、原子与无限方差下成立，并以USDC-WETH资金池数据实证表明将报价频率从十二秒缩短到一秒可削减84.5%的预期冲击。

    

    我们研究流动性提供者为了在能够利用过时报价的对手方面前实现盈亏平衡所需的价差，并将两种防御手段放在同一标尺上衡量：更快地重新定价和筛选知情交易流。一个盈亏平衡条件精确确定了较慢的做市商要与更快的、未经筛选的交易场所相匹配所需的筛选率：即用来换回额外报价陈旧性的选择性。该结果只需要价格变动过程可积，因此在跳跃、原子分布和无限方差下均成立。可预测的漂移是一个中心化问题：它会使以现货价为中心的报价变宽，但一旦报价以预期参考价格重新居中，其成本就会消失。在2026年6月至8月期间，以币安ETH-USDT为基准的四个USDC-WETH资金池中，一秒粒度的基准价格变动呈现显著的非高斯特征：73%的秒内价格没有变化。将报价节奏从十二秒缩短到一秒可削减84.5%的预期冲击，因此十二秒节奏的做市商必须至少以……（原文摘要在此处截断）

    arXiv:2610.21510v1 Announce Type: new  Abstract: We study the spread a liquidity provider needs to break even against counterparties that can exploit a stale quote, and we place two defenses on a single scale: repricing sooner and screening informed flow. A break-even condition pins down the exact screening rate at which a slower maker matches a faster unscreened venue: the selection that buys back the extra staleness. The result requires only an integrable price innovation and therefore holds under jumps, atoms, and infinite variance. Predictable drift is a centering problem: it widens a spot-centered quote, but its cost vanishes once the quote is recentered on the expected reference. In four USDC-WETH pools benchmarked against Binance ETH-USDT from June to August 2026, the one-second benchmark innovation is sharply non-Gaussian: 73% of seconds show no price change. Cutting cadence from twelve seconds to one sheds 84.5% of the expected shock, so a twelve-second maker must screen at le
    
[^9]: NEST：面向加密财库LETF堆栈的抗陈旧代币化股票度量协议

    NEST: Stale-Robust Tokenized-Stock Measurement for a Crypto-Treasury LETF Stack

    [https://arxiv.org/abs/2610.21491](https://arxiv.org/abs/2610.21491)

    论文提出NEST协议，利用共享历史恒等式、方差时钟界限、部分决定指数（PDI）及多重审计，将代币化股票标记对次日开盘缺口的机械性拟合与真实条件度量价值分离，从而稳健评估其对加密财库LETF链路盘外度量的增量价值。

    

    代币化股票在证券交易所闭市期间仍可交易，因此其标记价格可以改善连接比特币、加密财库股票和杠杆ETF的多层敞口在盘外时段的度量。然而，代币收益率与次日的开盘缺口共享从前收盘到代币标记的价格变动，因此二者表面上的拟合有一部分是机械性的。我们开发了NEST，一个抗陈旧协议，用于将这种共享历史的拟合与条件度量价值区分开来。一个共享历史恒等式和一个方差时钟界限显示了仅凭时间安排本身能产生多少拟合。部分决定指数（PDI）——即在给定公开控制变量条件下代币与缺口之间的偏相关平方——度量了剩余的真实价值，而一个有限样本边界以PDI为单位表达了偏F检验的决策规则。产品、来源、时间戳和执行审计决定一个有信息量的标记是否可被采纳。对于MSTRx——基于MicroStrategy（MSTR）的追踪凭证，一个代币……（摘要截断）

    arXiv:2610.21491v1 Announce Type: new  Abstract: Tokenized equities trade while listed markets are closed, so their marks could improve off-hours measurement along layered exposures that link Bitcoin, crypto-treasury stocks, and leveraged ETFs. A token return and the next opening gap, however, share the price move from the previous close to the token mark, so part of their apparent fit is mechanical. We develop NEST, a stale-robust protocol that separates this shared-history fit from conditional measurement value. A shared-history identity and a variance-clock bound show how much fit timing alone can generate. The partial determination index (PDI), the squared partial correlation between token and gap given public controls, measures what remains, and a finite-sample frontier expresses the partial-F decision in PDI units. Product, provenance, timestamp, and execution audits decide whether an informative mark is admissible. For MSTRx, a tracker certificate on MicroStrategy (MSTR), a toke
    
[^10]: 均值回归市场中具有延迟信息的长期投资

    Long-Horizon Investment with Delayed Information in a Mean-Reverting Market

    [https://arxiv.org/abs/2610.21447](https://arxiv.org/abs/2610.21447)

    本文在Ornstein-Uhlenbeck模型中显式求解了带信息延迟的长期投资问题，发现信息延迟会将确定性等价的增长从二次降为线性，且将完全信息策略直接投影到延迟信息上会导致无穷大的指数损失。

    

    我们研究了在具有指数效用的Ornstein-Uhlenbeck扩散模型中的长期投资问题。我们显式地求解了有限期限问题，并证明信息延迟会定性地改变长期投资行为。在完全信息下，由均值回归产生的确定性等价随投资期限呈二次增长；而在任何固定正延迟下，它仅呈线性增长，其渐近速率我们已显式计算得出。我们推导出了最优延迟投资组合的显式因果表示，并证明将完全信息策略投影到延迟信息上可能会剧烈失效：对于足够长的投资期限，由此产生的指数损失将变为无穷大。数值例子说明了信息延迟的经济影响。

    arXiv:2610.21447v1 Announce Type: new  Abstract: We study long-term investment in an Ornstein-Uhlenbeck diffusion model with exponential utility. We solve the finite-horizon problem explicitly and show that information delay qualitatively alters long-horizon behavior. Under full information, the certainty equivalent generated by mean reversion grows quadratically with the investment horizon. With any fixed positive delay, it grows only linearly, at an asymptotic rate that we compute explicitly. We derive an explicit causal representation of the optimal delayed portfolio and show that projecting the full-information strategy onto delayed information can fail dramatically: the resulting exponential loss becomes infinite for sufficiently long horizons. Numerical examples illustrate the economic impact of information delay.
    
[^11]: 离散资产定价基本定理

    Fundamental Theorem of Discrete Asset Pricing

    [https://arxiv.org/abs/2610.21442](https://arxiv.org/abs/2610.21442)

    本文在连续时间离散价格框架下建立了资产定价基本定理，证明了无套利、无消失风险自由午餐、无自由午餐与等价局部鞅测度存在性的等价性，且证明完全避开Itô积分，仅使用有限Stieltjes和。

    

    受连续时间中离散价格报价的启发，我们寻求一个基于无套利条件而非限制性交易机会条件的基本定理。我们考虑在连续时间内取值于 $\mathbb{R}^d$ 中局部有限子集的贴现价格向量。对于局部有界的càdlàg价格，我们证明了无套利、无消失风险自由午餐、无自由午餐以及等价局部鞅测度的存在性是相互等价的。通过对投资组合进行初等的有限维约化，得到了可容许财富的一个统一的逐路径界。该界将闭性问题约化为加权希尔伯特空间中的凸平均问题，随后应用Kreps-Yan定理即可完成证明。交易收益表示为有限的Stieltjes和，整个证明过程不使用Itô积分。

    arXiv:2610.21442v1 Announce Type: new  Abstract: Motivated by discrete price quotes in continuous time, we seek a fundamental theorem based on no arbitrage rather than a condition on limiting trading opportunities. We consider a vector of discounted prices taking values in a locally finite subset of $R^d$ in continuous time. For a locally bounded c\`adl\`ag price, we prove that no arbitrage, no free lunch with vanishing risk, no free lunch, and the existence of an equivalent local martingale measure are equivalent. An elementary finite-dimensional reduction of portfolios yields a common pathwise bound on admissible wealth. This bound reduces closedness to convex averaging in a weighted Hilbert space, after which the Kreps-Yan theorem applies. Trading gains are finite Stieltjes sums, and the proof uses no It\^o integration.
    
[^12]: 期货合约中的月末交割期权与持有标的资产的基特金斯指数

    The end-of-month option in futures contracts and the Gittins index of holding the underlying

    [https://arxiv.org/abs/2610.21196](https://arxiv.org/abs/2610.21196)

    本文利用Bank–El Karoui表示定理将期货合约的月末交割期权刻画为持有标的资产并收取便利收益的基特金斯指数问题，期货价格即为使该项目价值等于现货价格的退休回报，并通过Dinkelbach迭代将定价计算化归为一系列标准的最优停时问题。

    

    月末交割期权允许期货合约的空头方在交割价格固定的窗口期内选择何时进行交割。我们通过由Bank–El Karoui表示定理得到的通用指数来刻画最优交割策略。在半鞅框架下，该指数即为持有标的资产并收取其便利收益的基特金斯指数，而期货价格则是使该项目的价值等于现货价格的退休回报。这一刻画给出了价格边界以及立即交割的判据。Dinkelbach迭代将价格计算归结为一系列标准的最优停时问题，并给出了显式的压缩界。我们将该理论应用于几何布朗运动以及具有随机便利收益的Gibson–Schwartz模型。对于几何布朗运动，我们刻画了期货价格何时等于现货价格，并推导出平方根形式的边界渐近性质。

    arXiv:2610.21196v1 Announce Type: new  Abstract: The end-of-month option allows the short in a futures contract to choose when to deliver during a window in which the delivery price is fixed. We characterize optimal delivery through a universal index obtained from the Bank--El Karoui representation theorem. In the semimartingale setting, this is the Gittins index of holding the underlying and collecting its convenience yield, and the futures price is the retirement reward for which the value of this project equals the spot price. This characterization yields price bounds and a criterion for immediate delivery. Dinkelbach's iteration reduces the price computation to a sequence of standard optimal stopping problems, with an explicit contraction bound. We apply the theory to geometric Brownian motion and the Gibson--Schwartz model with stochastic convenience yield. For geometric Brownian motion, we characterize when the futures price equals spot and derive a square-root boundary asymptoti
    
[^13]: 使用交替方向显式（ADE）方法在Schwartz双因子模型下定价美式商品期权

    Pricing American Commodity Options Under the Schwartz Two-Factor Model Using the Alternating Direction Explicit (ADE) Method

    [https://arxiv.org/abs/2610.21145](https://arxiv.org/abs/2610.21145)

    本文提出用双扫掠交替方向显式（ADE）有限差分方法在Schwartz双因子模型下定价美式商品期权，该方法兼具显式格式的高效率与隐式格式的无条件稳定性，并通过与闭式解、蒙特卡洛及Longstaff-Schwartz方法的对比验证了其数值精度。

    

    本项目研究使用Schwartz双因子模型对美式商品期权进行估值。我们采用双扫掠交替方向显式（ADE）方法，这是一种在扩散项上无条件稳定的有限差分格式，兼具显式推进的效率和通常只有隐式方法才具备的稳定性，并将其与Saul'yev的单扫掠交替变体进行比较。我们推导了控制偏微分方程（PDE），并将半无限域转换为矩形计算域。此外，我们应用Fichera理论来确定现货价格极限处的边界条件。最后，我们用C++实现了求解器，首先将结果与PDE的闭式解以及作为欧式期权的蒙特卡洛方法进行对比验证，然后施加美式提前行权约束，并与Longstaff-Schwartz蒙特卡洛方法进行基准比较，以分析数值精度。

    arXiv:2610.21145v1 Announce Type: new  Abstract: This project investigates the valuation of American commodity options using the Schwartz two-factor model. We employ the double-sweep Alternating Direction Explicit (ADE) method, an unconditionally stable (for diffusion) finite difference scheme that provides the efficiency of explicit marching with the stability typically reserved for implicit methods, and compare it with the single-sweep alternating variant of Saul'yev. We derive the governing partial differential equation (PDE) and transform the semi-infinite domain into a rectangular computational domain. Furthermore, we apply Fichera theory to determine boundary conditions at the limits of the spot price. Finally, we implement the solver in C++, validating the results first against the closed-form solution of the PDE and Monte Carlo as a European option, then apply the American early exercise constraint and benchmark against Longstaff-Schwartz Monte Carlo to analyze numerical accura
    
[^14]: StochLOB：用于稳健限价订单簿趋势预测的重尾分数匹配

    StochLOB: Heavy-Tailed Score Matching for Robust Limit Order Book Trend Prediction

    [https://arxiv.org/abs/2610.21081](https://arxiv.org/abs/2610.21081)

    该论文提出StochLOB训练框架，用跳跃Lévy噪声和截断α稳定噪声等重尾非高斯扩散过程替代传统的布朗高斯噪声来加噪训练数据，以更好地捕捉限价订单簿订单流的重尾与跳跃特性，从而提升LOB趋势预测的稳健性，同时推理阶段保持确定性预测器。

    

    基于分数的扩散方法通常使用布朗高斯噪声对数据进行扰动，这对应于轻尾的前向随机微分方程（SDE）。对于市场时间序列而言，这种SDE可能是一种糟糕的抽象：限价订单簿（LOB）的订单流具有重尾性、突发性，且偶尔呈现跳跃状。我们研究了STOCHLOB，这是一个仅用于训练阶段的分数匹配框架，它改变了前向扩散的损坏规律，同时保持部署的预测器为确定性模型。该加噪过程不再仅使用布朗高斯损坏，而是采用两种受非高斯SDE启发的替代方案：通过复合泊松方差混合实现的跳跃Lévy噪声，以及通过正稳定高斯尺度混合实现的截断α稳定噪声。这些损坏机制在训练中注入了有限跳跃和重尾残差，而无需在推理阶段进行反向时间扩散采样。所得的分数目标具有一个有用的解释：其径向因子h（摘要在此处截断）

    arXiv:2610.21081v1 Announce Type: new  Abstract: Score-based diffusion methods usually perturb data with Brownian Gaussian noise, corresponding to a light-tailed forward stochastic differential equation (SDE). For market time series this SDE can be a poor abstraction: limit order book (LOB) order flow is heavy-tailed, bursty, and occasionally jump-like. We study STOCHLOB, a training-only score-matching framework that changes the forward diffusion corruption law while leaving the deployed predictor deterministic. Instead of Brownian-only Gaussian corruption, the noising process uses two non-Gaussian SDE-inspired alternatives: jump-L\'evy noise through a compound-Poisson variance mixture, and clipped $\alpha$-stable noise through a positive-stable Gaussian scale mixture. These corruptions inject finite jumps and heavy-tailed residuals into training without requiring reverse-time diffusion sampling at inference. The resulting score target has a useful interpretation: its radial factor $h_
    
[^15]: 外部比例交易成本下的损失与再平衡之差（LVR）

    Loss-versus-rebalancing with external proportional transaction costs

    [https://arxiv.org/abs/2610.21057](https://arxiv.org/abs/2610.21057)

    该论文证明，当外部市场存在比例交易成本时，以按外部买卖价交易的自融资策略为基准，自动做市商资金池的损失与再平衡之差（LVR）为零，并通过保留费用的效用无差异资金池与几何平均资金池实现特定的递减库存计划和恒定不交易边界。

    

    我们研究当在外部市场进行交易会产生比例交易成本时，自动做市商中的流动性提供问题。我们分别考虑了单独收取的费用以及保留在储备中的费用。我们的基准是一种自融资策略，它持有相同总量的风险库存，并按外部买价和卖价进行交易。在连续外部价格与纯套利的条件下，只要资金池收费或外部买卖价差为正，资金池与基准的清算财富便完全相同。因此，相对于这一考虑交易成本的基准，损失与再平衡之差为零——即使资金池不收费，只要外部价差为正亦是如此。随后，我们构造了保留费用的效用无差异资金池，用以实现容许的递减库存计划；并构造了几何平均资金池，用以实现投资组合权重的可行恒定不交易边界。这些结果为流动性提供提供了一种投资视角的解释：通过选择资金池的（原文摘要在此处截断）

    arXiv:2610.21057v1 Announce Type: new  Abstract: We study liquidity provision in automated market makers when trading in the external market incurs proportional transaction costs. We consider both separately collected fees and fees retained in reserves. The benchmark is a self-financing strategy holding the same total risky inventory and trading at the external bid and ask. Under continuous external prices and pure arbitrage, the pool and benchmark have identical liquidation wealth whenever the pool fee or the external spread is positive. Thus loss-versus-rebalancing relative to this transaction-cost benchmark is zero, even without a pool fee if the external spread is positive. We then construct retained-fee utility-indifference pools that implement admissible decreasing inventory schedules, and geometric mean pools that implement feasible constant no-trade boundaries for portfolio weights. These results offer an investment interpretation of liquidity provision: by choosing the pool de
    
[^16]: $q_x$-Bass 鞅：最优性、存在性与收敛性

    $q_x$-Bass Martingales: Optimality, Existence and Convergence

    [https://arxiv.org/abs/2610.20920](https://arxiv.org/abs/2610.20920)

    本文引入 $q_x$-Bass 鞅，证明其为弱最优传输问题的最优解并建立强对偶性，据此提出计算算法并在紧性假设下证明其收敛性，从而确立了此类鞅的存在性。

    

    我们引入了 $q_x$-Bass 鞅这一类新的鞅，扩展了 $q$-Bass 鞅的概念。这类鞅是具有给定边缘分布的鞅耦合，其转移核在某种合适的弱最优传输问题的意义下最接近于给定的参考核族 $(q_x)_{x \in \mathbb R^d}$。我们证明了 $q_x$-Bass 鞅是该问题的最优解，并建立了一个强对偶结果。基于该对偶性，我们提出了一种计算 $q_x$-Bass 鞅的算法，并在给定边缘分布满足紧性假设的条件下证明了算法的收敛性，从而确立了此类鞅的存在性。随后，我们将结果特化到一维边缘分布的情形，并证明该算法在此情形下归结为一个不动点迭代。

    arXiv:2610.20920v1 Announce Type: cross  Abstract: We introduce the class of $q_x$-Bass martingales, extending the notion of $q$-Bass martingales. These are martingale couplings with prescribed marginals whose transition kernels are closest to a given family of reference kernels $(q_x)_{x \in \mathbb R^d}$ in the sense of a suitable weak optimal transport problem. We show that $q_x$-Bass martingales are optimizers of this problem and establish a strong duality result. Building on this duality, we introduce an algorithm for their computation and, under compactness assumptions on the prescribed marginals, prove its convergence, thereby establishing existence. We then specialize to one-dimensional marginals and show that the algorithm reduces to a fixed-point iteration.
    
[^17]: 真实与模拟限价订单簿路径中不可逆性的神经估计

    Neural Estimation of Irreversibility in Real and Simulated Limit Order Book Paths

    [https://arxiv.org/abs/2610.16151](https://arxiv.org/abs/2610.16151)

    该论文首次将时间不可逆性的度量从标量金融序列拓展到多变量限价订单簿路径，通过神经网络估计发现订单排序本身贡献了显著且在所有十二只股票上稳定为正的不可逆性增量。

    

    一条与其自身逆序不可区分的路径不携带时间之箭，因此时间不可逆性是仅凭轨迹即可读取有向机制的必要条件。在金融领域，该性质此前仅在收益率和波动率等标量序列上被测量过。我们将粗粒化限价订单簿路径的不可逆率定义为每个事件的前向-逆向对数密度比，并利用接收相同数据块的顺序不变多层感知机（MLP）和时间卷积网络，从三周的纳斯达克逐笔委托（market-by-order）数据中对其进行估计。对于NVDA，顺序不变与顺序敏感的估计值分别为每事件0.036和0.054 nats，其差值（即排序增量）为0.018。一个保留粗粒化单步转移表的替代数据所恢复的不可逆性比观测路径低30%。该增量在全部十二只标的的每个留出白天日期上均为正，且对于NVDA（原文摘要在此处截断）

    arXiv:2610.16151v2 Announce Type: replace  Abstract: A path indistinguishable from its own reverse carries no arrow of time, so time irreversibility is a necessary condition for a directed mechanism to be readable from trajectories alone. In finance it has been measured only on scalar series such as returns and volatility. We define the irreversibility rate of a coarse-grained limit order book path as the forward-to-reverse log-density ratio per event, and estimate it from three weeks of Nasdaq market-by-order data with an order-invariant MLP and a temporal convolutional network that receive identical blocks. For NVDA, the order-invariant and order-sensitive estimates are 0.036 and 0.054 nats per event, respectively. Their difference, the ordering increment, is 0.018. A surrogate preserving the coarse one-step transition table recovers 30 percent less irreversibility than the observed paths. The increment is positive for all twelve symbols on every held-out daytime date, and for NVDA i
    
[^18]: 配置效率与创新的回报

    Allocative Efficiency and the Returns to Innovation

    [https://arxiv.org/abs/2610.14969](https://arxiv.org/abs/2610.14969)

    本文揭示在不完全竞争市场中，创新所引发的配置效率变化无法被创新者自身捕获，导致美国上市企业研发的社会回报高达其私人回报的6.24倍。

    

    创新通过其增加的技术以及其在不同加成的生产者之间引发的重新配置来改变福利。当生产者设定加成时，创新者自身的产出和价格调整在边际上没有私人价值，因此重新配置只能通过其他企业的反应才能转化为利润。对于在其市场中规模较小的创新者，在数量竞争或价格竞争下，配置效率的变化（其量级与创新的成本节约相同）不会为其利润带来任何收益。一个工艺创新者所获得的回报仅是其成本节约，同时放弃了自身及下游企业对新增产出的加成。就2015年的美国上市公司而言，这些被放弃的加成平均占成本节约的14%至26%，而对于拥有56%销售额的创新者，配置效率的上升或下降幅度超过私人回报的十分之一。结合估计出的溢出效应，申请专利企业的研发的社会回报是其私人回报的6.24倍。

    arXiv:2610.14969v2 Announce Type: replace  Abstract: Innovation changes welfare through the technology it adds and the reallocation it causes between producers with different markups. When producers set markups, an innovator's own output and price adjustments have no private value at the margin, so reallocation reaches its profit only through others' responses. For an innovator small in its markets, none of the change in allocative efficiency, which is of the same order as its cost saving, reaches its profit under quantity or price competition. A process innovator is paid its cost saving, forgoing its own and downstream markups on added output. For U.S. public firms in 2015, these average 14 to 26 percent of that saving, and allocative efficiency rises or falls by more than a tenth of the private return for innovators holding 56 percent of sales. With estimated spillovers, patenting firms' R&D has a social return 6.24 times its private return.
    
[^19]: AlphaRLM：面向基于文献的符号化因子发现的语言模型对齐

    AlphaRLM: Aligning Language Models for Literature-Grounded Symbolic Factor Discovery

    [https://arxiv.org/abs/2610.14959](https://arxiv.org/abs/2610.14959)

    AlphaRLM是一个强化学习框架，通过文献引导算子组装（LEGO）提供语义先验，并借助可执行有效性与分组收益分离等奖励将生成策略与市场反馈对齐，从而训练大语言模型直接从金融文献中构造可执行的符号化Alpha因子。

    

    公式化Alpha挖掘旨在从庞大且充满噪声的搜索空间中发现能够捕捉预测性市场信号的可执行表达式。近期基于大语言模型（LLM）的方法主要依赖提示工程、迭代搜索或种子因子进化，而通过策略学习直接构造因子的路径较少被探索。为解决这一问题，我们提出了AlphaRLM，一个强化学习框架，用于训练大语言模型在无需输入因子表达式的情况下，直接从金融文献中构造可执行的符号化因子。AlphaRLM引入了文献引导算子组装机制，将因子构造建立在文献衍生的思想与可执行算子之上，为探索提供语义先验。它还通过可执行有效性奖励和分组收益分离奖励（后者能够捕捉因子各分位数间的非单调结构）将生成策略与市场反馈对齐。在沪深300、中证1000和标普500上的大量实验表明，AlphaRLM……（摘要至此截断）

    arXiv:2610.14959v2 Announce Type: new  Abstract: Formulaic alpha mining seeks to discover executable expressions that capture predictive market signals in a vast, noisy search space. Recent LLM-based approaches use prompting, iterative search, or seed-factor evolution, leaving direct factor construction through policy learning less explored. To address this, we introduce AlphaRLM, a reinforcement-learning framework that trains LLMs to construct executable symbolic factors from financial literature without input factor expressions. AlphaRLM incorporates Literature-Guided Operator Assembly (LEGO), which conditions factor construction on literature-derived ideas and executable operators, providing semantic priors for exploration. It further aligns the generation policy with market feedback through executable-validity and grouped-return-separation rewards that capture non-monotonic structures across factor quantiles. Extensive experiments on CSI 300, CSI 1000, and S&P 500 show that AlphaRL
    
[^20]: RICE-Alpha：基于事件图的可靠性信息校正方法，用于LLM智能体股票预测

    RICE-Alpha: Reliability-Informed Correction with Event Graphs for LLM-Agent Stock Forecasting

    [https://arxiv.org/abs/2609.34004](https://arxiv.org/abs/2609.34004)

    该论文提出RICE-Alpha框架，通过多层记忆层和类型化事件智能体构建事件图，将历史感知的多视图基础Alpha与基于历史事件延续的可靠性校准残差校正相结合，从而在时点约束下提升LLM智能体的股票预测能力。

    

    与股票相关的新闻通过具有时间依赖性的企业事件不断演变，因此只有当事件连续性、信息可用性和转移可靠性都被建模时，历史信息才有价值。现有的基于LLM的金融智能体虽然引入了历史证据，但在时点约束下保留发行方特定的时间脉络，以及识别历史转移何时能为当前预测提供超出当前信息范围的额外信息方面，其支持能力有限。我们提出了RICE-Alpha（基于事件图的可靠性信息校正），这是一个时点股票评分框架，它将具备历史感知能力的多视图基础Alpha与从历史事件延续中推导出的可靠性校准残差校正相分离。一个多层记忆层将新闻解读锚定于时间上合规的发行方特定历史，同时一个类型化事件智能体构建事件状态，其后续关系在发行方（内形成——原文摘要在此处截断）。

    arXiv:2609.34004v2 Announce Type: replace  Abstract: Equity-relevant news evolves through temporally dependent corporate events, making historical information useful only when event continuity, information availability, and transition reliability are modeled. Existing LLM-based financial agents incorporate historical evidence, yet they provide limited support for preserving issuer-specific chronology under point-in-time constraints and for identifying when historical transitions contribute information beyond the current forecast. We present RICE-Alpha (Reliability-Informed Correction with Event Graphs), a point-in-time stock-scoring framework that separates a history-aware multi-view Base Alpha from a reliability-calibrated residual correction derived from historical event continuation. A Multi-Tier Memory Layer grounds news interpretation in temporally eligible issuer-specific history, while a Typed Event Agent constructs event states whose successor relations are formed within issuer
    
[^21]: 论分位数曲面的渐近形状

    On the asymptotic shape of quantile surfaces

    [https://arxiv.org/abs/2609.31345](https://arxiv.org/abs/2609.31345)

    本文证明了对数正态随机变量线性组合生成的分位数曲面在左尾全局凹、右尾全局凸，并具有尺度与形状渐近分离的特性。

    

    本文研究分位数曲面的渐近形状，其中分位数曲面定义为由受控一维分布在给定水平 $\alpha$ 下生成的分位数集合。具体而言，当该分布由对数正态随机变量的线性组合产生，且控制变量为正系数向量时，我们证明了分位数曲面在左尾（$\alpha\to0$）是全局凹的，在右尾（$\alpha\to1$）是全局凸的。此外，这些曲面还表现出尺度与形状的渐近分离特性。

    arXiv:2609.31345v1 Announce Type: cross  Abstract: This article is concerned with the asymptotic shape of quantile surfaces, defined as the set of quantiles at a given level $\alpha$ generated by a controlled one-dimensional distribution. Specifically, when the distribution arises as a linear combination of log-normal random variables and the control is a vector of positive coefficients, we prove that quantile surfaces are globally concave in the left tail ($\alpha\to0$) and globally convex in the right tail ($\alpha\to1$). Moreover, these surfaces exhibit asymptotic separation of scale and shape.
    
[^22]: 高维因子模型中的主成分误差

    Principal component error in high-dimensional factor models

    [https://arxiv.org/abs/2609.20550](https://arxiv.org/abs/2609.20550)

    本文将高维因子模型中主成分估计的误差分解为可从数据估计的“子空间外误差”与无法从数据估计的“子空间内误差”两个可解释项，并证明二者在变量数增长而样本量有限时具有几乎必然的渐近极限。

    

    在统计因子模型中，样本协方差矩阵的主成分（或特征向量）被用作“主方向”的估计，即一组观测变量共同变动的真实驱动因素。我们将这些估计中通常相当大的误差写成两个可解释项之和，并证明当变量数量增长而样本量保持有界时，这两项具有几乎必然的渐近极限。这种情形在金融经济学、基因组学、机器学习和信号处理中十分常见。“子空间外误差”衡量估计值与由总体因子暴露张成的子空间之间的距离，它可以用数据表示，从而提供一个可估计的误差下界。“子空间内误差”则源于潜在因子收益的固定样本量，无法仅凭数据估计。我们通过美国公开股票市场的三因子模拟来展示这一误差分析。

    arXiv:2609.20550v1 Announce Type: cross  Abstract: In a statistical factor model, principal components (or eigenvectors) of a sample covariance matrix serve as estimates of {\it principal directions}, the true drivers of co-movement of a collection of observed variables. We write the often substantial error in these estimates as a sum of two interpretable terms, which we show have almost sure asymptotic limits as the number of variables grows with sample size bounded. This scenario is commonplace in financial economics, genomics, machine learning and signal processing. {\it Out-of-subspace error} measures the distance from an estimate to the subspace spanned by population factor exposures. It can be expressed in terms of data, providing an estimable floor for error. {\it In-subspace error} arises from the fixed sample size of the latent factor returns and cannot be estimated from data alone. We illustrate our error analysis with a three-factor simulation of the US public equity market,
    
[^23]: 超越马尔可夫拼接的全局多期限SPX-VIX校准

    Global Multi-Maturity SPX-VIX Calibration Beyond Markovian Stitching

    [https://arxiv.org/abs/2609.04087](https://arxiv.org/abs/2609.04087)

    本文建立了跨多期限SPX-VIX联合校准的全局框架，证明局部与全局可行性等价、但拼接测度仅为全局可行测度的真子集，并指出相对熵在马尔可夫参考下只能选出拼接的最小信息补全，识别非马尔可夫依赖需要额外的跨期信息。

    

    我们开发了一个全局框架，用于在多个期限上对标普500（SPX）与VIX微笑曲线进行联合校准，且不受马尔可夫拼接所带来的条件独立性限制。精确的局部可行性与全局可行性是等价的：每一个全局可行的测度都拥有一个保持区块的SPX-马尔可夫化，使每个月度的 $(S_i,V_i,S_{i+1})$ 分布保持不变。然而，拼接得到的测度可能构成全局可行路径测度的一个真子集，因为马尔可夫化会丢弃当前SPX水平之外对更早历史的依赖。因此，相邻期限的微笑曲线无法识别这种依赖性，而具有相同月度校准结果的测度可能会对多期未定权益给出不同的定价。在标准的马尔可夫参考下，相对熵会选出拼接的最小信息补全；而非马尔可夫依赖性则需要跨期信息、适当的目标函数或依赖历史的先验。对于有限离散化情形，我们引入了一个……

    arXiv:2609.04087v1 Announce Type: new  Abstract: We develop a global framework for joint S&P 500 (SPX)-VIX smile calibration across multiple maturities without the conditional-independence restriction induced by Markovian stitching. Exact local and global feasibility are equivalent: every globally feasible law has a block-preserving SPX-Markovization that leaves each monthly $(S_i,V_i,S_{i+1})$ law unchanged. Nevertheless, stitched laws can form a strict subset of globally feasible path laws because Markovization discards dependence on earlier history beyond the current SPX level. Adjacent smiles therefore cannot identify this dependence, and laws with identical monthly calibrations can price multi-period claims differently. Under the standard Markov reference, relative entropy selects the stitched minimum-information completion; non-Markov dependence requires cross-period information, an appropriate objective, or a history-dependent prior. For finite discretizations, we introduce an a
    
[^24]: DYSANOS：生成式动态平滑无套利非参数期权曲面

    DYSANOS Generative Dynamic Smooth Arbitrage-free Non-parametric Option Surfaces

    [https://arxiv.org/abs/2608.12587](https://arxiv.org/abs/2608.12587)

    本文提出首个生成式无套利期权曲面模型DYSANOS，能够生成未来多年每日期权价格路径，并验证其优于传统隐含波动率PCA模型。

    

    本文介绍了DYSANOS，这是首个生成式市场模型，用于生成所有行权价和到期日的平滑且无静态套利的SANOS期权曲面。该模型旨在生成未来数年的每日现货和期权价格完整路径。我们提出了一种稳健且实用但相对简单的基线隐状态生成模型，形式为AR(1)模型。我们讨论了模型设置、数据管道和训练过程，并研究了动态套利的数值存在性。我们使用Option Metrics的IvyDB S&P指数数据（2020年至2025年）展示了模型性能，并将其与纯隐含波动率主成分分析模型进行了比较。

    arXiv:2608.12587v1 Announce Type: cross  Abstract: This article presents with DYSANOS the first generative market model for smooth SANOS option surfaces for all strikes and expiries which are free of static arbitrage. Our model is designed to generate entire paths of daily spot and option prices for years in the future.   We present a robust and useful if somewhat simplistic baseline hidden state generative model in the form of an AR(1) model. We discuss model setup, data pipeline, and training and investigate numerical resence of dynamic arbitrage. We illustrate model performance on Option Metrics' IvyDB S\&P Index data from 2020 to~2025 and compare it to a pure implied-vol PCA model.
    
[^25]: 金融异常波发生的实时识别

    Real-time identification of the onset of financial rogue waves

    [https://arxiv.org/abs/2606.31475](https://arxiv.org/abs/2606.31475)

    该论文将光学与流体中的“怪波”类比引入金融波动率分析，利用带克尔非线性势的薛定谔方程研究VIX等波动率指数，发现极端事件临近时出现安德森局域化迹象，且系统最小特征值的数值梯度在极端事件开始时可靠激增，从而实现对金融极端事件起点的实时识别。

    

    金融系统中的极端事件（通常通过波动率等指标来刻画）在其临近发生时仍然难以识别。波动率与其他经历极端事件的自然复杂系统共享许多统计特性，这正是本文所探讨的内容。我们通过识别具有相似统计特性的类怪波峰值，将光学与流体动力系统中怪波之间的类比拓展到了金融波动率领域。我们使用一个势函数遵循克尔非线性形状的薛定谔方程，在滑动时间窗口内考察金融波动率指数的性质。我们观察到，在VIX指数中当怪波峰值临近时会出现安德森局域化的证据，并证明系统最小特征值的数值梯度会在极端事件发生的起点可靠地出现激增。我们将该方法调整为模拟数据的实时到达，并表明除一个例外之外，（摘要原文在此处截断）

    arXiv:2606.31475v4 Announce Type: replace  Abstract: Extreme events in financial systems, often captured by indicators such as volatility, remain difficult to identify close to their onset. Volatility shares many statistical properties with other natural, complex systems which experience extreme events, which we explore in this manuscript. We extend the analogy between rogue waves in optical and hydrodynamical systems to financial volatility by identifying rogue-wave-like peaks with similar statistical properties. We use a Schr\"odinger equation where the potential follows the shape of a Kerr nonlinearity to examine the properties of financial volatility indices within a moving time window. We see evidence of Anderson localisation as a rogue peak approaches in the VIX, and show that the numerical gradient of the system's minimum eigenvalue reliably spikes at the onset of an extreme event. We adapt our methodology to simulate the real-time arrival of data, and show that all but one of t
    
[^26]: 差额互换中抵押货币的选择

    Choice of Collateral Currency in Differential Swaps

    [https://arxiv.org/abs/2603.07863](https://arxiv.org/abs/2603.07863)

    本文研究了一类参考隔夜利率后视平均值的差额互换的估值与对冲，将抵押账户扩展为可用不同于合约现金流的货币计价，并给出了显式的定价与对冲策略。

    

    抵押品在衍生品定价中的作用已超越了信用风险缓释的范畴，特别是在全球金融危机之后，融资成本和基差利差成为估值实践的核心。这一发展与伦敦银行间同业拆借利率（LIBOR）向无风险利率（RFR）的过渡以及抵押化交易的日益标准化同时发生。我们研究了一类参考隔夜利率后视平均值的差额互换（differential swaps）的估值与对冲问题，其中以担保隔夜融资利率（SOFR）互换作为特例。研究重点是抵押货币的影响。通过扩展Bickersteth等人[Math. Finance 36 (2026), pp. 180--202]的早期结果，我们允许抵押账户以与合约现金流不同的货币计价，并利用基于期货的方法推导出显式的定价与对冲策略。

    arXiv:2603.07863v2 Announce Type: replace  Abstract: The role of collateral in derivative pricing has evolved beyond credit risk mitigation, particularly following the global financial crisis, when funding costs and basis spreads became central to valuation practices. This development coincided with the transition from the London Interbank Offered Rate (LIBOR) to risk-free rates (RFRs) and the increasing standardization of collateralised trading. We study the valuation and hedging of a class of differential swaps referencing backward-looking averages of overnight rates, with Secured Overnight Financing Rate (SOFR) swaps appearing as a particular instance. The focus is on the impact of the collateral currency. Extending the earlier results of Bickersteth et al. [Math. Finance 36 (2026), pp.~180--202], we allow the collateral account to be denominated in a currency different from that of the contractual cash flows and derive explicit pricing and hedging strategies using a futures-based r
    

