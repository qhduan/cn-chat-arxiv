# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [When a Correct Reward Is Not Enough: Diagnosing and Guiding PPO in an Analytically Solved Broker-Trader Game](https://arxiv.org/abs/2610.03598) | 本文将PPO智能体置于解析可解的连续时间经纪商-交易员博弈中，利用已知的解析解来诊断并引导强化学习，发现即使奖励设计正确，PPO在存在随机非知情订单流时依然难以学到准确策略。 |
| [^2] | [PreFER: Interactive Robo-Advisor with Scoring Mechanism](https://arxiv.org/abs/2610.03406) | 该论文提出了一个名为PreFER的交互式智能投顾框架，通过客户对投资建议的评分学习其个性化风险偏好，将偏好学习与逆强化学习联系起来，并据此推导出探索性投资策略。 |
| [^3] | [Mixture-of-Experts for Cryptocurrency Order Execution: Training Stability, Tail Risk, and Failure Modes](https://arxiv.org/abs/2610.03369) | 该论文在BTC/USDT订单簿数据上系统评估了专家混合（MoE）强化学习用于订单执行的效果，发现K≥4的MoE架构能显著提升训练稳定性并消除原始DDQL中拖延至被迫清算的失效模式，但没有任何学习型配置能在平均执行缺口上超越TWAP和立即清算等简单基准。 |
| [^4] | [No Women No Innovation? The Effect of Women on Boards on Hard and Soft Innovation in SMEs](https://arxiv.org/abs/2610.03250) | 本研究利用2015-2024年2762家意大利创新型中小企业数据和2SLS识别策略，首次发现董事会女性对中小企业硬创新和软创新均具有稳健的正向因果影响，纠正了以往研究因严重负向选择偏差而产生的误导性负向结论。 |
| [^5] | [FinNextAssist: Towards Professional Financial Deep Research Assistant](https://arxiv.org/abs/2610.03174) | 提出 FinNextAssist，一个面向专业金融分析的端到端深度研究框架，通过整合异构权威金融数据源、专用分析工具和领域子智能体，并将研究流程分解为任务规划、证据编译、推理和报告四个阶段来完成专业金融分析。 |
| [^6] | [Landscape-Dependent Performance of Photonic Quantum Solvers in QUBO Feature Selection for Financial Risk Detection](https://arxiv.org/abs/2610.03161) | 该论文在信用卡欺诈与消费者违约两个金融数据集上，对经典分支定界（Gurobi）、光子熵计算（QCI Dirac-3）与模拟光子玻色采样（Piquasso）三种计算范式的十三种QUBO特征选择方法进行了系统基准测试，揭示了求解器性能高度依赖于数据集地形：Dirac-3在ULB欺诈数据集上仅用13/30个特征即可匹敌全特征模型（平均F1约0.873），而在159特征的AmEx违约数据集上所有范式均需接近全部特征才能达到F1≈0.80。 |
| [^7] | [MintEval: Do LLMs Implement the Trading Strategy You Asked For? A Behavioural-Equivalence Benchmark for Natural-Language-to-Strategy Code](https://arxiv.org/abs/2610.03080) | 该论文提出MintEval基准，通过程序化生成参考交易策略并回译为自然语言指令让大语言模型重新实现，再在相同市场数据上逐K线比较生成策略与参考策略的实际交易行为（而非代码相似度或利润），以检验大语言模型编写的策略代码是否真正做到了行为等价于交易者的原始意图。 |
| [^8] | [Shapley-based Structural Analysis of Neural Calibration for Stochastic Volatility Models](https://arxiv.org/abs/2610.03076) | 本文首次运用SHAP与νSHAP等基于Shapley值的可解释AI方法，系统分析了Heston与粗糙Heston模型神经校准映射的内部结构，发现短期限与微笑翼部始终主导参数推断，且该归因结构在不同网络架构间保持定性稳定。 |
| [^9] | [Event History Over Scale: Compact Transformers for Low-Latency Limit Order Book Forecasting](https://arxiv.org/abs/2610.02917) | 提出仅约7千至1.4万参数的紧凑因果Transformer（MBOFormer与MBOFusion），直接建模逐笔订单（Level-3）事件历史，在三个市场、四个预测跨度中十一个设置下取得最优宏F1，并实现亚毫秒级推理延迟，显著优于基于Level-2快照的基线模型。 |
| [^10] | [Multi-Agent AI as a Nested Principal-Agent Problem in Private Wealth Management: Mandate Representation and Evidence Control in Switzerland, Germany and Austria](https://arxiv.org/abs/2610.02863) | 本文将财富管理中管理者向AI授权的问题建模为嵌套的委托—代理框架，在瑞士、德国和奥地利的法律约束下结合受限联合最大化，并通过仿真揭示了客户负债或管理者条款信息遗漏会导致流动性与能力违规风险。 |
| [^11] | [Axient: Manifest-Bound Evidence for On-Chain Financial Protocols: Seven-Layer Derivation, Correlation, Tamper Rejection, and Reproducible Claim Promotion](https://arxiv.org/abs/2610.02838) | 本文提出Axient架构，通过封存的执行清单绑定系统身份与发布标识，将七个证据层以共同关联标识链接，并采用要求来源可溯、阴性对照、可复现打包、历史回放与独立审查的合取式声明提升规则，从而解决链上金融协议评估中哈希复制无法证明独立推导的证据可信性问题。 |
| [^12] | [Axient: Canonical Protocol-Graph Composition for Leveraged Event Markets: Single State Authority, Atomic Composition, Durable Sagas, and Exactly-Once Recovery](https://arxiv.org/abs/2610.02834) | 本文提出 Axient 规范协议图架构，通过为每个金融领域设立单一状态权威，并以原子组合、持久 Saga 和恰好一次恢复机制统一协调跨域状态转换，解决了模块化杠杆事件市场协议中各组件各自正确却缺乏统一权威执行路径的问题。 |
| [^13] | [Right Order, Wrong Scale: Auditing LLM Judges for Occupational AI Measurement](https://arxiv.org/abs/2610.02492) | 该研究提出O*NET-BENCH审计套件，发现LLM评判器虽在回答排序上与人类工人基本一致，但在估计AI输出可接受率时产生3.0%-97.9%的巨大偏差，表明排序准确并不等于可靠的量化测量。 |
| [^14] | [Expected Utility Regret Rule: Minimax and Bayes Optimal Portfolio Choice](https://arxiv.org/abs/2610.02290) | 提出期望效用遗憾（EUR）规则，该规则无需先验分布即可同时达到极小极大与贝叶斯最优下界，并将均值-方差组合和风险平价组合统一为该框架的特例。 |
| [^15] | [Social welfare and price discovery in double auction markets](https://arxiv.org/abs/2610.01562) | 本文首次为双向拍卖的价格发现能力提供了理论解释，证明在纯交换经济中代理人按无差异价格出价时，瓦尔拉斯均衡恰为拍卖的不动点，且重复双向拍卖产生的有界配置与价格序列的聚点收敛于带转移支付的瓦尔拉斯均衡。 |
| [^16] | [A continuous-time dynamic contracting problem with limited liability and finite horizon](https://arxiv.org/abs/2609.18287) | 该论文在加入有限责任约束的连续时间Holmström-Milgrom动态契约模型中，通过概率方法证明了委托人价值函数是完全非线性退化偏微分方程的唯一有界古典解，并借助这一强正则性结果确保了最优控制的强形式存在性及其精细刻画。 |
| [^17] | [The Physical Crash Frontier: What Finite Option Quotes Can and Cannot Reveal](https://arxiv.org/abs/2608.23274) | 本文通过有限期权报价界定物理崩溃风险的可达范围，提出物理崩溃前沿，并展示其高效计算方法及实际市场中的显著影响。 |
| [^18] | [How Likely and How Deep? Sharp Joint Bounds on Risk-Neutral Crash Probability and Conditional Depth from Option Bid-Ask Quotes](https://arxiv.org/abs/2607.25353) | 本文提出了“崩溃前沿”概念，通过精确的有限线性系统投影，在不离散化状态空间的情况下，为风险中性崩溃概率和条件深度提供尖锐的联合界，并附带对偶证书。 |
| [^19] | [Daycare Matching with Siblings: Social Implementation and Welfare Evaluation](https://arxiv.org/abs/2604.13597) | 本研究开发了一个纳入兄弟姐妹联合分配偏好的实证框架并应用于日本托儿所分配，发现分开分配的固定负效用相当于4.61个通勤公里，且忽略兄弟姐妹互补性会将多子女家庭的福利收益低估约27%。 |
| [^20] | [Optimal Quantum Speedups for Repeatedly Nested Expectation Estimation](https://arxiv.org/abs/2602.08120) | 该论文提出了一种估计重复嵌套期望的最优量子算法，通过设计经典随机多层蒙特卡洛算法的新型去随机化变体克服了可变时间问题，实现了 $\tilde O(\varepsilon^{-1})$ 的本质上最优复杂度，相比最佳经典算法获得近似二次加速，并将应用范围扩展至最优停止等更广泛的问题。 |
| [^21] | [Error Propagation in Dynamic Programming: From Stochastic Control to American Option Pricing](https://arxiv.org/abs/2509.20239) | 本文为离散时间随机最优控制建立了结合再生核希尔伯特空间回归与蒙特卡洛抽样的动态规划近似框架，提出自然的误差分解并严格分析了误差从到期日向初始时刻反向传播的规律，可应用于美式期权定价。 |
| [^22] | [Optimal Investment and Consumption in a Stochastic Factor Model](https://arxiv.org/abs/2509.09452) | 本文通过建立无边界值开域上二阶常微分方程的下解与上解一般理论，首次为包括Heston模型在内的随机因子模型中的无穷期限最优投资与消费问题证明了HJB方程解的存在性并给出了严格验证，同时在有限状态情形给出了完整适定性刻画与高效数值算法。 |
| [^23] | [Optimal Fees for Liquidity Provision in Automated Market Makers](https://arxiv.org/abs/2508.08152) | 该论文在一个AMM与中心化交易所并行、交易者最优选择交易场所的动态模型中，刻画了使LP利润最大化的固定流动性下最优费用以及竞争性进入下使均衡总锁仓价值（TVL）最大化的最优费用，并通过大规模模拟和市场数据校准验证了模型的有效性。 |
| [^24] | [When defaults cannot be hedged: xVA calculations via local risk-minimization](https://arxiv.org/abs/2502.12774) | 该论文针对银行与交易对手违约均无法对冲的现实情形，提出利用局部风险最小化方法并通过BSDE求解，实现交易对手信用风险与融资成本（xVA）的定价与对冲。 |
| [^25] | [Constrained portfolio optimization in a life-cycle model: A deep pricing kernel approach](https://arxiv.org/abs/2410.20060) | 本文提出一种深度定价核方法，通过构建人工市场和对偶变换，有效求解生命周期模型中带凸集交易约束的投资组合优化问题，并给出紧致上下界。 |

# 详细

[^1]: 当正确的奖励还不够时：在解析求解的经纪商-交易员博弈中诊断与引导PPO

    When a Correct Reward Is Not Enough: Diagnosing and Guiding PPO in an Analytically Solved Broker-Trader Game

    [https://arxiv.org/abs/2610.03598](https://arxiv.org/abs/2610.03598)

    本文将PPO智能体置于解析可解的连续时间经纪商-交易员博弈中，利用已知的解析解来诊断并引导强化学习，发现即使奖励设计正确，PPO在存在随机非知情订单流时依然难以学到准确策略。

    

    当复杂动态使解析策略难以获得时，强化学习（RL）越来越多地被用于金融最优控制问题。金融数学文献提供了许多已求解的模型，其方程与最优控制可以用来评估和引导学习；我们探究强化学习能否利用这些已有成果。我们将一个近端策略优化（PPO）智能体置于一个解析可解的连续时间经纪商-交易员博弈中。PPO取代经纪商并选择其交易速度，同时与知情交易者以及随机的非知情订单流进行交互。我们从经纪商的连续时间收益中推导出有限步奖励，并通过网格细化与精确的单步恒等式验证其离散实现。在无非知情订单流的情况下，经验证集选择的PPO-前馈神经网络（FFNN）能够接近参考最优动作。在存在随机非知情订单流的情况下，所测试的PPO-FFNN与PPO-LSTM仍然不准确，尽管……（摘要在此处被截断）

    arXiv:2610.03598v1 Announce Type: cross  Abstract: Reinforcement learning (RL) is increasingly used for financial optimal-control problems when complex dynamics make analytical strategies difficult to obtain. There are financial mathematics literactures which provides many solved models whose equations and controls could evaluate and guide learning; we ask whether RL can exploit these results.   We place a proximal policy optimisation (PPO) agent in an analytically solved continuous-time broker--trader game. PPO replaces the broker and chooses its trading speed while interacting with an informed trader and stochastic uninformed order flow. We derive a finite-step reward from the broker's continuous-time payoff and verify its discrete implementation through grid refinement and an exact one-step identity. With zero uninformed flow, a validation-selected PPO--FFNN approaches the reference action. With stochastic uninformed flow, the tested PPO--FFNN and PPO--LSTM remain inaccurate, althou
    
[^2]: PreFER：具有评分机制的交互式智能投顾

    PreFER: Interactive Robo-Advisor with Scoring Mechanism

    [https://arxiv.org/abs/2610.03406](https://arxiv.org/abs/2610.03406)

    该论文提出了一个名为PreFER的交互式智能投顾框架，通过客户对投资建议的评分学习其个性化风险偏好，将偏好学习与逆强化学习联系起来，并据此推导出探索性投资策略。

    

    我们提出了一种交互式智能投顾框架，该框架能够从客户提供的评分中学习个性化的风险偏好。由此产生的偏好学习问题与逆强化学习（IRL）密切相关，因为智能投顾需要从反馈中推断客户的潜在奖励设定。智能投顾与客户的迭代交互过程如下：在每个交互时刻，投顾基于由推断出的个性化风险偏好所导出的最优策略分布生成投资建议；客户对该建议进行评分；投顾根据反馈更新其对客户风险偏好的评估。这一学习过程促使我们研究离散时间的可预测前瞻探索性奖励过程，并推导出一种探索性投资策略。通过将评分解释为某条建议被接受的概率，我们的逆学习过程能够学习客户的……

    arXiv:2610.03406v1 Announce Type: new  Abstract: We propose an interactive robo-advising framework that learns personalized risk preferences from scores provided by clients. The resulting preference-learning problem is closely related to inverse reinforcement learning (IRL), as the robo-advisor infers the client's latent reward specification from feedback. The robo-advisor interacts with clients iteratively as follows. At each interaction time, the advisor generates investment advice based on the optimal policy distribution derived from an inferred personalized risk preference. The client scores the advice. The advisor updates its assessment of the client's risk preference based on the feedback. This learning procedure motivates us to investigate discrete-time Predictable Forward Exploratory Reward (PreFER) processes and derive an exploratory investment strategy. By interpreting the score as the acceptance probability of a piece of advice, our inverse learning procedure learns the clie
    
[^3]: 面向加密货币订单执行的专家混合模型：训练稳定性、尾部风险与失效模式

    Mixture-of-Experts for Cryptocurrency Order Execution: Training Stability, Tail Risk, and Failure Modes

    [https://arxiv.org/abs/2610.03369](https://arxiv.org/abs/2610.03369)

    该论文在BTC/USDT订单簿数据上系统评估了专家混合（MoE）强化学习用于订单执行的效果，发现K≥4的MoE架构能显著提升训练稳定性并消除原始DDQL中拖延至被迫清算的失效模式，但没有任何学习型配置能在平均执行缺口上超越TWAP和立即清算等简单基准。

    

    用于订单执行的深度强化学习策略在不同训练随机种子之间可能存在显著差异，因此表面上由架构带来的收益，可能反映的只是有利的训练实现，而非该架构可复现的固有特性。我们在来自币安的5分钟均值聚合BTC/USDT限价订单簿数据上，评估了原始双重深度Q学习（DDQL）、采用K均值划分的DDQL专家混合模型（K∈{2,4,8}），以及与K=4和K=8专家预算参数量相匹配的稠密网络。结果显示，没有任何学习型配置在平均执行缺口上显著优于DDQL。在所报告的设定下，在一个无摩擦回放与终端紧迫性惩罚使得提前清算几乎零成本的环境中，所有方法的平均执行缺口均高于TWAP（0.39个基点）和立即清算（0.21个基点）；原始DDQL的100次运行中有11次收敛到一种等待直至被迫清算的策略，而在两个K≥4的MoE组中均未出现此现象……（摘要至此截断）

    arXiv:2610.03369v1 Announce Type: cross  Abstract: Deep reinforcement-learning policies for order execution can vary substantially across training seeds, so apparent architectural gains may reflect favourable training realisations rather than reproducible properties of the architecture. We evaluate vanilla Double Deep Q-Learning (DDQL), K-means-partitioned mixtures of DDQL experts at $K \in \{2, 4, 8\}$, and dense networks parameter-matched to the $K{=}4$ and $K{=}8$ expert budgets on 5-minute mean-aggregated BTC/USDT limit order book data from Binance. No learned configuration significantly improves mean implementation shortfall over DDQL. Under the reported specification, all have higher mean shortfall than TWAP (0.39 bps) and immediate liquidation (0.21 bps) in an environment whose frictionless replay and terminal-urgency penalty make early liquidation nearly costless; 11/100 vanilla-DDQL runs, versus none in either MoE $K{\geq}4$ arm, converge to a policy that waits until forced li
    
[^4]: 没有女性就没有创新？董事会中的女性对中小企业硬创新与软创新的影响

    No Women No Innovation? The Effect of Women on Boards on Hard and Soft Innovation in SMEs

    [https://arxiv.org/abs/2610.03250](https://arxiv.org/abs/2610.03250)

    本研究利用2015-2024年2762家意大利创新型中小企业数据和2SLS识别策略，首次发现董事会女性对中小企业硬创新和软创新均具有稳健的正向因果影响，纠正了以往研究因严重负向选择偏差而产生的误导性负向结论。

    

    女性在公司董事会中的作用及其对创新的影响仍存在激烈争论。尽管现有文献对女性董事参与究竟是促进还是阻碍创新提供了相互矛盾的观点，但大多数研究仅关注大型企业。本研究通过考察董事会女性对中小企业“硬”（技术性）创新和“软”（非技术性）创新的因果影响来填补这一研究空白。基于2015年至2024年2762家意大利创新型中小企业的纵向数据集和2SLS识别策略，我们发现了高度细致的规律。对于硬创新，工具变量模型揭示了稳健的正向因果影响，扭转了负的基准系数，并暴露出严重的负向选择偏差。对于软创新，稳定正向影响得到证实。我们的发现为中小企业的公司治理提供了关键的因果证据，同时为政策制定者和从业者提供了有针对性的见解。

    arXiv:2610.03250v1 Announce Type: new  Abstract: The role of women on corporate boards and its impact on innovation remains heavily debated. While existing literature offers conflicting perspectives on whether female board representation spurs or hinders innovaiton, most research has focused exlusively on large firms. This studt adresses this gap by investigating the causal impact of women on boards on both "hard" (technological) and "soft" (non-technological) innovation within SMEs. Using a longitudinal dataset f 2762 Italian innovative SMEs from 2015 to 2024 and a 2SLS identification strategy, we find highly nuanced patterns. For hard innovation, the instrumented model reveals a robust positive causal impacr, reversing a negative baseline coefficient and exposing a severe negative selection bias. For soft innovation, a stable positive impact is confirmed. Our findings provide crucial causal evidence for SMEs' governance, while offering tailored insights for policymakers and practitio
    
[^5]: FinNextAssist：迈向专业金融深度研究助手

    FinNextAssist: Towards Professional Financial Deep Research Assistant

    [https://arxiv.org/abs/2610.03174](https://arxiv.org/abs/2610.03174)

    提出 FinNextAssist，一个面向专业金融分析的端到端深度研究框架，通过整合异构权威金融数据源、专用分析工具和领域子智能体，并将研究流程分解为任务规划、证据编译、推理和报告四个阶段来完成专业金融分析。

    

    深度研究（Deep Research, DR）智能体通过自主规划、迭代检索、多步推理和结构化报告，已在复杂的研究型任务中展现出强大的能力。然而，将深度研究智能体应用于金融领域带来了独特的挑战：金融分析需要联合完成跨越多种数据类型、工具和分析工作流的异构子任务。我们识别出专业金融深度研究智能体的三个关键需求：整合权威且异构的金融数据源；具备专业的分析工具与技能；以及设置针对领域特定子任务的专门子智能体。基于这些原则，我们提出了 FinNextAssist，一个面向专业金融分析的端到端深度研究框架。FinNextAssist 将研究过程分解为四个阶段：任务规划器、证据编译器、推理引擎和报告组装器，并引入了两个新颖的轻量级……

    arXiv:2610.03174v1 Announce Type: cross  Abstract: Deep Research (DR) agents have demonstrated strong capabilities in complex, research-oriented tasks through autonomous planning, iterative retrieval, multi-step reasoning, and structured reporting. However, adapting DR agents to finance introduces unique challenges: financial analysis demands the joint completion of heterogeneous sub-tasks spanning diverse data types, tools, and analytical workflows. We identify three key requirements for a professional financial DR agent: integration of authoritative, heterogeneous financial data sources; specialized analytical tools and skills; and dedicated sub-agents for domain-specific sub-tasks. Building on these principles, we propose FinNextAssist, an end-to-end deep research framework designed for professional financial analysis. FinNextAssist decomposes the research process into four stages: Task Planner, Evidence Compiler, Reasoning Engine, and Report Assembler, and introduces two novel ligh
    
[^6]: 光子量子求解器在金融风险检测QUBO特征选择中依赖问题地形的性能表现

    Landscape-Dependent Performance of Photonic Quantum Solvers in QUBO Feature Selection for Financial Risk Detection

    [https://arxiv.org/abs/2610.03161](https://arxiv.org/abs/2610.03161)

    该论文在信用卡欺诈与消费者违约两个金融数据集上，对经典分支定界（Gurobi）、光子熵计算（QCI Dirac-3）与模拟光子玻色采样（Piquasso）三种计算范式的十三种QUBO特征选择方法进行了系统基准测试，揭示了求解器性能高度依赖于数据集地形：Dirac-3在ULB欺诈数据集上仅用13/30个特征即可匹敌全特征模型（平均F1约0.873），而在159特征的AmEx违约数据集上所有范式均需接近全部特征才能达到F1≈0.80。

    

    针对信用卡欺诈和消费者违约检测等不平衡分类任务的特征选择，需要在预测相关性、特征间冗余与计算可行性之间取得平衡。我们在两个数据集上——ULB信用卡欺诈数据集（30个特征）和美国运通（AmEx）消费者违约数据集（159个特征）——对三种计算范式进行了基准测试：经典分支定界优化（Gurobi）、光子熵计算（QCI Dirac-3）以及模拟光子玻色采样，涵盖十三种特征选择方法，每种方法均被路由到与其数学结构相匹配的求解器。在ULB数据集上，Dirac-3的MI-Spearman方法仅使用30个特征中的13个即可达到与全特征模型相当的性能（五次运行的平均F1为0.873 ± 0.023，最佳运行为0.896），而Piquasso在k=5时是表现最佳的方法。在AmEx数据集上，性能随特征预算的增加而稳步提升，且所有范式只有在接近全部特征集时才能达到约F1 = 0.80的水平。Gurobi与其他求解器之间的大多数差异……（原文摘要在此处截断）

    arXiv:2610.03161v1 Announce Type: cross  Abstract: Feature selection for imbalanced classification tasks such as credit card fraud and consumer default detection requires balancing predictive relevance, inter-feature redundancy, and computational feasibility. We benchmark three computing paradigms, classical branch-and-bound optimization (Gurobi), photonic entropy computing (QCI Dirac-3), and simulated photonic boson sampling (Piquasso), across thirteen feature-selection methods on two datasets: ULB Credit Card Fraud (30 features) and AmEx consumer default (159 features). Each method is routed to the solver matched to its mathematical structure. On ULB, Dirac-3 MI-Spearman matches the all-features model using 13 of 30 features (mean F1 0.873 +/- 0.023 over five runs, best run 0.896), and Piquasso is the best method at k=5. On AmEx, performance rises steadily with the feature budget and every paradigm approaches F1 = 0.80 only near the full feature set. Most differences between Gurobi a
    
[^7]: MintEval：大语言模型是否实现了你所要求的交易策略？一个面向自然语言转策略代码的行为等价性基准

    MintEval: Do LLMs Implement the Trading Strategy You Asked For? A Behavioural-Equivalence Benchmark for Natural-Language-to-Strategy Code

    [https://arxiv.org/abs/2610.03080](https://arxiv.org/abs/2610.03080)

    该论文提出MintEval基准，通过程序化生成参考交易策略并回译为自然语言指令让大语言模型重新实现，再在相同市场数据上逐K线比较生成策略与参考策略的实际交易行为（而非代码相似度或利润），以检验大语言模型编写的策略代码是否真正做到了行为等价于交易者的原始意图。

    

    大语言模型正在从生成交易信号转向编写执行这些信号的代码。第二种角色的失败模式是静默的：生成的代码可以运行，回测可以画出图表，但交易者所描述的风险逻辑却并非实际执行的逻辑。现有代码基准通过单元测试检验功能正确性，金融基准则检验预测能力，二者都无法衡量一个实现的行为是否与所要求的策略一致。我们提出MintEval，该基准从可组合的构建模块库中以程序化方式生成参考策略，将其回译为口语化的交易者指令，再由被测模型重新实现。生成的程序与参考程序在完全相同的市场数据和摩擦成本上逐根K线执行，并基于其交易行为而非代码相似度或利润进行比较：超额收益（alpha）被差分消除。MintEval v0包含800个基于BTCUSDT 15分钟的……（摘要不完整）

    arXiv:2610.03080v1 Announce Type: cross  Abstract: Large language models are moving from producing trading signals to writing the code that executes them. The failure mode of the second role is silent: generated code runs, a backtest plots, yet the risk logic that the trader described is not the logic being executed. Existing code benchmarks test functional correctness on unit tests and finance benchmarks test forecasting; neither measures whether an implementation behaves like the strategy that was asked for. We introduce MintEval, a benchmark in which reference strategies are generated programmatically from a library of composable building blocks, back-translated into colloquial trader instructions, and re-implemented by the model under test. Generated and reference programs are executed bar by bar on identical market data and frictions, and compared on their actions rather than on code similarity or profit: alpha is differenced away. MintEval v0 contains 800 tasks on BTCUSDT 15-minu
    
[^8]: 基于Shapley值的随机波动率模型神经校准的结构分析

    Shapley-based Structural Analysis of Neural Calibration for Stochastic Volatility Models

    [https://arxiv.org/abs/2610.03076](https://arxiv.org/abs/2610.03076)

    本文首次运用SHAP与νSHAP等基于Shapley值的可解释AI方法，系统分析了Heston与粗糙Heston模型神经校准映射的内部结构，发现短期限与微笑翼部始终主导参数推断，且该归因结构在不同网络架构间保持定性稳定。

    

    基于神经网络的方法已成为随机波动率模型校准中传统基于优化的流程的高效替代方案。然而，现有工作主要集中于预测精度，相对较少关注对所学到的逆校准映射结构的理解。在这项工作中，我们使用来自可解释人工智能领域的互补的基于Shapley值的方法，分析了针对Heston模型和粗糙Heston模型的神经校准映射，涵盖多层感知机、高速公路网络和softmax参数化高速公路等多种架构。具体而言，我们考虑SHAP和νSHAP解释方法，它们捕捉了特征相关性的两种不同且互补的概念，分别对应于特征子集的敏感性和充分性。研究发现，短期限和微笑曲线的翼部在参数推断中始终占主导地位，且主导性的归因结构在定性上保持稳定。

    arXiv:2610.03076v1 Announce Type: new  Abstract: Neural network-based approaches have emerged as efficient alternatives to traditional optimization-based procedures for the calibration of stochastic volatility models. However, existing work has focused primarily on predictive accuracy, with comparatively little attention devoted to understanding the structure of the learned inverse calibration mappings. In this work, we analyze neural calibration mappings for the Heston and rough Heston models across multilayer perceptron, highway, and softmax-parametrized highway architectures, using complementary Shapley-based methods from explainable AI. Specifically, we consider SHAP and $\nu$SHAP explanations, which capture distinct, complementary notions of feature relevance, corresponding to sensitivity and sufficiency of feature subsets, respectively. Short maturities and smile wings consistently dominate parameter inference, and the dominant attribution structure remains qualitatively stable a
    
[^9]: 事件历史胜过规模：用于低延迟限价订单簿预测的紧凑型Transformer

    Event History Over Scale: Compact Transformers for Low-Latency Limit Order Book Forecasting

    [https://arxiv.org/abs/2610.02917](https://arxiv.org/abs/2610.02917)

    提出仅约7千至1.4万参数的紧凑因果Transformer（MBOFormer与MBOFusion），直接建模逐笔订单（Level-3）事件历史，在三个市场、四个预测跨度中十一个设置下取得最优宏F1，并实现亚毫秒级推理延迟，显著优于基于Level-2快照的基线模型。

    

    在股票市场和日内电力市场中，基于限价订单簿进行短期价格趋势预测，需要模型兼具预测质量、低单样本延迟以及较小的序列化模型体积，以跟上快速且连续的市场更新。我们提出了MBOFormer——一个拥有7,203个参数的因果Transformer，以及MBOFusion——一个拥有14,371个参数、带有慢速时间上下文分支的扩展版本。这两个模型均处理按订单逐笔记录（Level-3）的市场历史，包括单个订单的提交、撤销和成交。我们将其与基于Level-2数据的基线模型进行对比，后者处理的是采样的订单簿快照和简单统计量，而非底层的订单消息。在三个市场和四个预测时间跨度的实验中，与各种规模的基准模型相比，我们的模型在十二个设置中的十一个中取得了最高的平均宏F1分数。两个模型在单个Apple M系列芯片上均实现了亚毫秒级的中位推理延迟。

    arXiv:2610.02917v1 Announce Type: new  Abstract: Short-horizon price-trend prediction from limit order books in equity and intraday electricity markets requires models that combine predictive quality with low single-sample latency and a small serialized model size to keep pace with rapid and continuous market updates. We introduce MBOFormer, a 7,203-parameter causal transformer, and MBOFusion, a 14,371-parameter extension with a slow temporal-context branch. Both models process market-by-order (level-3) histories of individual order submissions, cancellations, and executions. We compare them with level-2-based baselines that process sampled order-book snapshots and simple statistics instead of their underlying messages. Across three markets and four prediction horizons, our models achieve the highest mean macro-F1 in eleven of the twelve settings in a comparison against other benchmark models of varying sizes. Both our models achieve sub-millisecond median inference on a single Apple M
    
[^10]: 多智能体AI作为私人财富管理中的嵌套委托—代理问题：瑞士、德国与奥地利的授权表示与证据控制

    Multi-Agent AI as a Nested Principal-Agent Problem in Private Wealth Management: Mandate Representation and Evidence Control in Switzerland, Germany and Austria

    [https://arxiv.org/abs/2610.02863](https://arxiv.org/abs/2610.02863)

    本文将财富管理中管理者向AI授权的问题建模为嵌套的委托—代理框架，在瑞士、德国和奥地利的法律约束下结合受限联合最大化，并通过仿真揭示了客户负债或管理者条款信息遗漏会导致流动性与能力违规风险。

    

    在私人财富管理中，将任务委托给人工智能（AI）的管理者既是客户的代理人，又是AI系统的委托人。我们提出了一种与模型无关的表述方法，将嵌套的委托—代理授权与受限联合最大化相结合，作为分配给AI系统的任务。该目标函数在“投资组合—工作流”配对上分别刻画客户与管理者的结果。法律义务、授权要求与证据充分性共同决定可采性，并以瑞士、德国和奥地利的法律环境作为背景。权重与参考服务底线使权衡关系显性化；让步核算则将其对客户的影响分离开来。解析构造与基于公开市场观测数据的仿真展示了该方法。在来自四个构造性授权的八个决策状态中，遗漏客户负债导致了两次流动性违规，遗漏管理者条款导致了两次能力违规……

    arXiv:2610.02863v1 Announce Type: new  Abstract: In private wealth management, a manager delegating to artificial intelligence (AI) acts as the client's agent and the system's principal. We introduce a model-independent formulation that combines nested principal--agent delegation with constrained joint maximisation as the task assigned to the AI system. The objective represents client and manager outcomes separately over portfolio--workflow pairs. Legal duties, mandate requirements and evidence sufficiency determine admissibility, with Switzerland, Germany and Austria supplying the legal context. Weights and reference-service floors make the trade-off explicit; concession accounting separates their effects on the client. Analytical constructions and a simulation using public-market observations illustrate the approach. Across eight decision states from four constructed mandates, omitted client liabilities caused two liquidity violations, omitted manager terms caused two capacity violat
    
[^11]: Axient：面向链上金融协议的清单绑定证据体系：七层推导、关联、篡改拒绝与可复现的声明提升

    Axient: Manifest-Bound Evidence for On-Chain Financial Protocols: Seven-Layer Derivation, Correlation, Tamper Rejection, and Reproducible Claim Promotion

    [https://arxiv.org/abs/2610.02838](https://arxiv.org/abs/2610.02838)

    本文提出Axient架构，通过封存的执行清单绑定系统身份与发布标识，将七个证据层以共同关联标识链接，并采用要求来源可溯、阴性对照、可复现打包、历史回放与独立审查的合取式声明提升规则，从而解决链上金融协议评估中哈希复制无法证明独立推导的证据可信性问题。

    

    混合链上金融协议在评估时常常依赖单独有用但整体不足的证据：单元测试、交易回执、截图、服务响应或若干匹配的哈希值可能被当作完整工作流的证明，即使这些证据层共享同一个生成来源，或遗漏了真正关键的金融权威环节。本文为这类系统构建了一种清单绑定的证据架构。一个封存的执行清单绑定所声明的系统身份、角色、模式与发布标识。七个证据层连同已注册的推导关系被保留，并通过一个共同的关联标识相互链接。一条合取式的声明提升规则在哈希相等之外，还要求来源可溯、针对特定谓词的根条件、阴性对照、可复现打包、历史回放以及独立审查。我们形式化了证据图，并证明了被复制的哈希无法建立独立的推导关系。

    arXiv:2610.02838v1 Announce Type: new  Abstract: Hybrid on-chain financial protocols are frequently evaluated with evidence that is individually useful but collectively insufficient: a unit test, transaction receipt, screenshot, service response, or several matching hashes may be presented as proof of a complete workflow even when the layers share one generated source or omit the financial authority that matters. This paper develops a manifest-bound evidence architecture for such systems. A sealed execution manifest binds declared system identities, roles, schemas, and release identity. Seven evidence layers are retained with registered derivations and linked by a common correlation identity. A conjunctive claim-promotion rule requires provenance, predicate-specific root conditions, negative controls, reproducible packaging, historical replay, and independent review in addition to hash equality. We formalize the evidence graph, prove that copied hashes cannot establish independent deri
    
[^12]: Axient：面向杠杆事件市场的规范协议图组合：单一状态权威、原子组合、持久 Saga 与恰好一次恢复

    Axient: Canonical Protocol-Graph Composition for Leveraged Event Markets: Single State Authority, Atomic Composition, Durable Sagas, and Exactly-Once Recovery

    [https://arxiv.org/abs/2610.02834](https://arxiv.org/abs/2610.02834)

    本文提出 Axient 规范协议图架构，通过为每个金融领域设立单一状态权威，并以原子组合、持久 Saga 和恰好一次恢复机制统一协调跨域状态转换，解决了模块化杠杆事件市场协议中各组件各自正确却缺乏统一权威执行路径的问题。

    

    一个模块化的杠杆事件市场协议，即使其风险审批、头寸、债务、结算证据、信用池、劣后保障、储备金、清算和治理等各模块合约单独来看都是正确的，仍可能缺乏一条权威的金融执行路径。本文提出了一个规范协议图架构，其中每个金融领域都拥有唯一的存储权威，而组合组件仅负责协调状态转换，不会重新创建余额、债务、头寸、证据回执、队列、储备金、暂停状态或终局情景状态。该架构形式化了以下机制：原子头寸发起、结算确认的债务 Saga、交易者残值释放、有序的“储备金—劣后保障—高级层”损失瀑布、参与损失分担的提款队列、基于能力的条件执行、协议级暂停与时间锁恢复、确定性部署清单、规范状态投影，以及在链[遭……]之后的恰好一次重建（原文摘要在此处被截断）。

    arXiv:2610.02834v1 Announce Type: new  Abstract: A modular leveraged event-market protocol can contain individually correct contracts for risk approval, positions, debt, settlement evidence, credit pools, junior backstops, reserves, liquidation, and governance while still lacking one authoritative financial execution path. This paper develops a canonical protocol graph in which every financial domain has one storage authority and composition components coordinate transitions without recreating balances, debt, positions, evidence receipts, queues, reserves, pause state, or terminal scenario state. The architecture formalizes atomic position origination, a settlement-confirmed debt saga, trader residual release, an ordered reserve-backstop-Senior loss waterfall, loss-participating withdrawal queues, capability-conditioned execution, protocol-wide pause and timelocked recovery, deterministic deployment manifests, canonical state projection, and exactly-once reconstruction after a chain-co
    
[^13]: 排序正确，尺度有误：审计用于职业AI测量的LLM评判器

    Right Order, Wrong Scale: Auditing LLM Judges for Occupational AI Measurement

    [https://arxiv.org/abs/2610.02492](https://arxiv.org/abs/2610.02492)

    该研究提出O*NET-BENCH审计套件，发现LLM评判器虽在回答排序上与人类工人基本一致，但在估计AI输出可接受率时产生3.0%-97.9%的巨大偏差，表明排序准确并不等于可靠的量化测量。

    

    LLM评判器正被越来越多地用于评估AI输出是否满足职场要求，但对回答排序的一致性并不能确立在接受率或职业总体层面的一致性。我们提出了O*NET-BENCH，这是一个基于包含45,796名工人评分的现有调查构建的审计套件，并在4,501个测试评分上评估了来自六个模型系列的33种现有评判器配置。其中25种配置实现了至少0.60的平局感知成对排序准确率，尽管一个经训练拟合的仅基于回答文本的TF-IDF基线几乎与最强评判器表现相当。尽管存在这种排序上的一致性，评判器对回答可接受比例的估计介于3.0%至97.9%之间，而职业匹配的人类工人给出的估计为61.1%。在一个微调模型谱系中，从逐点评分切换到捆绑的少样本/列表式评估协议虽然改善了回答排序，却降低了在任务和职业层面与工人平均评分的一致性；这一反转现象在一个任务子集上得到了复现。

    arXiv:2610.02492v1 Announce Type: new  Abstract: LLM judges are increasingly used to assess whether AI outputs meet workplace requirements, but agreement on response rankings does not establish agreement on acceptance rates or occupational aggregates. We introduce O*NET-BENCH, an audit suite derived from an existing survey of 45,796 worker ratings, and evaluate 33 pre-existing judge configurations across six model families on 4,501 test ratings. Twenty-five configurations achieve tie-aware pair accuracy of at least 0.60, although a train-fitted response-only TF-IDF baseline nearly matches the strongest judge. Despite this ordering agreement, judges estimate that 3.0%-97.9% of responses are acceptable, compared with 61.1% for occupation-matched workers. In one fine-tuned lineage, changing from pointwise scoring to a bundled few-shot/listwise protocol improves response ordering while reducing agreement with worker means at the task and occupation levels; this reversal replicates on a tas
    
[^14]: 期望效用遗憾规则：极小极大与贝叶斯最优投资组合选择

    Expected Utility Regret Rule: Minimax and Bayes Optimal Portfolio Choice

    [https://arxiv.org/abs/2610.02290](https://arxiv.org/abs/2610.02290)

    提出期望效用遗憾（EUR）规则，该规则无需先验分布即可同时达到极小极大与贝叶斯最优下界，并将均值-方差组合和风险平价组合统一为该框架的特例。

    

    本研究考虑投资组合选择问题，即为投资者推荐一个投资组合，以最大化其财富的期望效用。我们的目标是构建一个在期望效用遗憾（即“先知”投资者的期望效用与从数据中选择的投资组合所实现的期望效用之差）意义上渐近最优的投资组合选择规则。我们提出了期望效用遗憾（EUR）规则，该规则联合选择投资组合类别并估计其权重。在正则参数化收益模型中，单一的EUR规则在不使用定义贝叶斯准则的先验分布的情况下，同时达到了极小极大下界和贝叶斯下界，包括它们的首项常数。随后，我们将均值-方差组合和风险平价组合推导为该框架的特例。在光滑、递增且凹的效用函数下，EUR规则与样本均值-方差组合在……（摘要在此处截断）

    arXiv:2610.02290v1 Announce Type: cross  Abstract: This study considers the problem of portfolio choice, where we recommend a portfolio to an investor to maximize the expected utility of their wealth. Our goal is to construct an asymptotically optimal portfolio choice rule in terms of expected utility regret, the difference between the expected utility of an oracle investor and that achieved by a portfolio chosen from data. We propose the Expected Utility Regret (EUR) rule, which jointly selects a portfolio class and estimates its weights. In a regular parametric return model, a single EUR rule attains both the minimax and the Bayes lower bounds, including their leading constants, without using the prior distribution that defines the Bayes criterion. We then derive the mean--variance and risk-parity portfolios as special cases of this framework. Under smooth increasing and concave utility, the EUR rule and the sample mean--variance portfolio attain the same leading expected regret when
    
[^15]: 双向拍卖市场中的社会福利与价格发现

    Social welfare and price discovery in double auction markets

    [https://arxiv.org/abs/2610.01562](https://arxiv.org/abs/2610.01562)

    本文首次为双向拍卖的价格发现能力提供了理论解释，证明在纯交换经济中代理人按无差异价格出价时，瓦尔拉斯均衡恰为拍卖的不动点，且重复双向拍卖产生的有界配置与价格序列的聚点收敛于带转移支付的瓦尔拉斯均衡。

    

    双向拍卖机制将价格推向竞争性均衡的倾向已在实验室实验中得到充分记录，但这一现象一直缺乏理论上的解释。本文研究了纯交换经济中的动态双向拍卖，其中代理人根据其当前持有和偏好所隐含的无差异价格进行出价。我们证明瓦尔拉斯均衡恰好与双向拍卖的不动点重合，并且重复进行的双向拍卖会产生有界的配置与价格序列，其聚点即为带转移支付的瓦尔拉斯均衡。

    arXiv:2610.01562v1 Announce Type: new  Abstract: The tendency of the double auction mechanism to drive prices to competitive equilibrium has been well documented in laboratory experiments, but the phenomenon has lacked a theoretical explanation. This paper studies dynamic double auctions in a pure exchange economy where agents bid their indifference prices implied by their current holdings and preferences. We show that Walras equilibria coincide with the fixed points of the double auction and that repeated double auctions generate bounded sequences of allocations and prices whose cluster points are Walras equilibria with transfers.
    
[^16]: 具有有限责任和有限期限的连续时间动态契约问题

    A continuous-time dynamic contracting problem with limited liability and finite horizon

    [https://arxiv.org/abs/2609.18287](https://arxiv.org/abs/2609.18287)

    该论文在加入有限责任约束的连续时间Holmström-Milgrom动态契约模型中，通过概率方法证明了委托人价值函数是完全非线性退化偏微分方程的唯一有界古典解，并借助这一强正则性结果确保了最优控制的强形式存在性及其精细刻画。

    

    我们对著名的Holmström-Milgrom模型（Econometrica 55 (2), 1987）的连续时间版本中的委托-代理问题进行了详细研究，并在该模型中为代理人加入了有限责任约束。我们开发了一种概率方法，证明了委托人的价值函数是一个在[0,T]×[0,∞)上带有柯西-狄利克雷（Cauchy-Dirichlet）边界条件的完全非线性、完全退化偏微分方程（PDE）的唯一有界古典解。事实上，我们还证明了该解在区域内部具有无穷阶连续可微性。我们的正则性结果足够强，使得我们能够确保最优控制以强形式存在——这在动态契约问题中是罕见的情况——并且我们得到了最优控制映射的精细性质，包括通过另一个非线性退化偏微分方程对其进行的刻画。

    arXiv:2609.18287v1 Announce Type: cross  Abstract: We perform a detailed study of a principal--agent problem in a continuous time version of the celebrated Holmstr\"om--Milgrom model (Econometrica 55 (2), 1987) where we add limited liability for the Agent. We develop a probabilistic methodology to prove that the Principal's value function is the unique bounded classical solution to a fully nonlinear and fully degenerate partial differential equation (PDE) with Cauchy-Dirichlet boundary conditions on $[0,T]\times[0,\infty)$. Indeed, we also prove infinite continuous differentiability of the solution in the interior of the domain. The strength of our regularity result is such that we can ensure existence of optimal controls in strong form---a rare occurrence in dynamic contracting---and we obtain fine properties of the optimal control map, including a characterisation via a further nonlinear degenerate PDE.
    
[^17]: 物理崩溃边界：有限期权报价能揭示和不能揭示的内容

    The Physical Crash Frontier: What Finite Option Quotes Can and Cannot Reveal

    [https://arxiv.org/abs/2608.23274](https://arxiv.org/abs/2608.23274)

    本文通过有限期权报价界定物理崩溃风险的可达范围，提出物理崩溃前沿，并展示其高效计算方法及实际市场中的显著影响。

    

    期权价格是保险的价格，因此它们隐含的风险中性概率高估了物理崩溃风险。幂效用定价核消除了溢价。但只有有限数量的合约交易，每个合约有买价和卖价，许多分布都适合这些价差。每个分布都暗示其自身的崩溃概率和低于崩溃阈值的预期损失。本文精确刻画了可达对。在有界支持下，它们形成一个紧凸集。我们称其边界为物理崩溃前沿。它将报价所允许的内容与所排除的内容分开。两个坐标都是矩的比率，但有限二阶锥程序可以追踪前沿。在每周标普500指数的1000个横截面中，超出最接近阈值的两个看跌期权的报价将可接受概率范围缩小了约80%的中位数。去掉边界，右尾深处的消失质量会使这些比率的分子膨胀，因此...

    arXiv:2608.23274v1 Announce Type: new  Abstract: Option prices are prices of insurance, so the risk-neutral probabilities they imply overstate physical crash risk. A power utility pricing kernel undoes the premium. But finitely many contracts trade, each at a bid and an ask, and many distributions fit inside the spreads. Each implies its own crash probability and expected loss below a crash threshold. This paper characterizes the attainable pairs exactly. With bounded support, they form a compact convex set. We call its boundary the physical crash frontier. It separates what the quotes admit from what they rule out. Both coordinates are ratios of moments, yet finite second-order cone programs trace the frontier. In a thousand weekly S&P 500 cross sections, quotes beyond the two puts nearest the threshold shrink the admissible probability range by a median of about 80 percent. Remove the bound, and a vanishing mass deep in the right tail inflates the denominator of those ratios, so the 
    
[^18]: 可能性多大，深度多深？来自期权买卖报价的风险中性崩溃概率与条件深度的尖锐联合界

    How Likely and How Deep? Sharp Joint Bounds on Risk-Neutral Crash Probability and Conditional Depth from Option Bid-Ask Quotes

    [https://arxiv.org/abs/2607.25353](https://arxiv.org/abs/2607.25353)

    本文提出了“崩溃前沿”概念，通过精确的有限线性系统投影，在不离散化状态空间的情况下，为风险中性崩溃概率和条件深度提供尖锐的联合界，并附带对偶证书。

    

    具有买卖价差的期权报价无法点识别低于给定阈值的风险中性崩溃概率，也无法识别阈值被突破后崩溃的预期深度。分别针对这两个量计算的界可能产生误导，因为它们的端点可能由不同的风险中性分布实现。我们刻画了联合可实现的概率与损失集合的闭包，并将其边界称为崩溃前沿。在观察到的行权价和阈值处划分状态空间，并增加一个坐标用于阈值处的质量，使得该集合成为有限线性系统的投影。该投影是精确的，不需要对状态空间进行离散化。支持程序追踪前沿，并为数字期权和看跌期权的任何组合给出尖锐的上界和下界。每个界都附带一个对偶证书，即现金、远期和保留报价的静态组合。

    arXiv:2607.25353v3 Announce Type: replace  Abstract: Option quotes with bid-ask spreads do not point-identify the risk-neutral probability of a crash below a given threshold, nor the expected depth of the crash once the threshold is breached. Bounds computed separately for the two quantities can mislead, because their endpoints may be attained by different risk-neutral distributions. We characterize the closure of the set of jointly attainable probability and loss pairs and call its boundary the crash frontier. Partitioning the state space at observed strikes and at the threshold, with one added coordinate for mass at the threshold itself, makes this set the projection of a finite linear system. The projection is exact, and no discretization of the state space is required. Support programs trace the frontier and give sharp upper and lower values for any portfolio of digital and put payoffs. Each bound comes with a dual certificate, a static portfolio of cash, forward, and retained quot
    
[^19]: 基于兄弟姐妹的托儿所匹配：社会实施与福利评估

    Daycare Matching with Siblings: Social Implementation and Welfare Evaluation

    [https://arxiv.org/abs/2604.13597](https://arxiv.org/abs/2604.13597)

    本研究开发了一个纳入兄弟姐妹联合分配偏好的实证框架并应用于日本托儿所分配，发现分开分配的固定负效用相当于4.61个通勤公里，且忽略兄弟姐妹互补性会将多子女家庭的福利收益低估约27%。

    

    在集中式匹配市场中，参与者可能重视联合分配，例如兄弟姐妹或夫妻的情况。标准的偏好估计方法忽略了这种互补性，使得针对配对分配优先规则的福利分析变得复杂。我们开发了一个纳入这些偏好的实证框架，并将其应用于日本的托儿所分配问题。家庭面临额外的通勤距离以及因子女被分开分配而产生的固定负效用。我们估计后者的数值相当于4.61个通勤公里。我们的固定报告反事实分析估计，该改革使平均福利提高了0.032个公里当量单位。忽略兄弟姐妹互补性会使同时为多名子女申请的家庭的福利收益被低估约27%。

    arXiv:2604.13597v3 Announce Type: replace  Abstract: In centralized matching markets, agents may value joint assignment, as with siblings or couples. Standard preference estimation ignores such complementarities, complicating welfare analysis of priority rules for paired assignment. We develop an empirical framework incorporating these preferences and apply it to Japanese daycare assignment. Families face both additional commuting distance and a fixed disutility from split assignment. We estimate the latter at 4.61 commuting-kilometer equivalents. Our fixed-report counterfactual estimates that the reform increased mean welfare by 0.032 kilometer-equivalent units. Ignoring sibling complementarity understates welfare gains for households applying simultaneously for multiple children by about 27%.
    
[^20]: 重复嵌套期望估计的最优量子加速

    Optimal Quantum Speedups for Repeatedly Nested Expectation Estimation

    [https://arxiv.org/abs/2602.08120](https://arxiv.org/abs/2602.08120)

    该论文提出了一种估计重复嵌套期望的最优量子算法，通过设计经典随机多层蒙特卡洛算法的新型去随机化变体克服了可变时间问题，实现了 $\tilde O(\varepsilon^{-1})$ 的本质上最优复杂度，相比最佳经典算法获得近似二次加速，并将应用范围扩展至最优停止等更广泛的问题。

    

    我们研究利用量子计算来估计具有常数视界（即嵌套层数）的重复嵌套期望（RNEs）。我们提出了一种量子算法，该算法在至多相差对数因子的情况下，以 $\tilde O(\varepsilon^{-1})$ 的代价实现 $\varepsilon$ 误差。标准的下界表明这一复杂度标度在本质上是 optimal 的，相比最佳经典算法可实现近似二次加速。我们的结果将先前针对单层嵌套期望的量子加速扩展到了重复嵌套的情形，从而涵盖了更广泛的应用，包括最优停止问题。这一扩展需要对经典随机多层蒙特卡洛算法进行一种新的去随机化变体设计。精心的去随机化处理是克服可变时间问题的关键，而该问题通常会增加经典随机算法量子化版本的计算代价。

    arXiv:2602.08120v2 Announce Type: replace-cross  Abstract: We study the estimation of repeatedly nested expectations (RNEs) with a constant horizon (number of nestings) using quantum computing. We propose a quantum algorithm that achieves $\varepsilon$-error with cost $\tilde O(\varepsilon^{-1})$, up to logarithmic factors. Standard lower bounds show this scaling is essentially optimal, yielding an almost quadratic speedup over the best classical algorithm. Our results extend prior quantum speedups for single nested expectations to repeated nesting, and therefore cover a broader range of applications, including optimal stopping. This extension requires a new derandomized variant of the classical randomized Multilevel Monte Carlo (rMLMC) algorithm. Careful de-randomization is key to overcoming a variable-time issue that typically increases quantized versions of classical randomized algorithms.
    
[^21]: 动态规划中的误差传播：从随机控制到美式期权定价

    Error Propagation in Dynamic Programming: From Stochastic Control to American Option Pricing

    [https://arxiv.org/abs/2509.20239](https://arxiv.org/abs/2509.20239)

    本文为离散时间随机最优控制建立了结合再生核希尔伯特空间回归与蒙特卡洛抽样的动态规划近似框架，提出自然的误差分解并严格分析了误差从到期日向初始时刻反向传播的规律，可应用于美式期权定价。

    

    本文研究离散时间随机最优控制（SOC）的理论与方法基础。我们首先在一个一般的动态规划框架下表述控制问题，并引入进行详细收敛性分析所需的数学结构。相关的价值函数通过结合非参数回归方法与蒙特卡洛子抽样的序列近似来估计。回归步骤在再生核希尔伯特空间（RKHS）中进行，利用经典的核岭回归（KRR）算法，同时引入蒙特卡洛抽样方法来估计续值（continuation value）。为评估价值函数估计器的精度，我们提出了一种自然的误差分解方法，并严格控制在每个时间步产生的误差项。随后我们分析了该误差如何随时间反向传播——从到期日到初始时刻——这是一个相对较少被探索的方面。

    arXiv:2509.20239v2 Announce Type: replace-cross  Abstract: This paper investigates theoretical and methodological foundations for stochastic optimal control (SOC) in discrete time. We start formulating the control problem in a general dynamic programming framework, introducing the mathematical structure needed for a detailed convergence analysis. The associate value function is estimated through a sequence of approximations combining nonparametric regression methods and Monte Carlo subsampling. The regression step is performed within reproducing kernel Hilbert spaces (RKHSs), exploiting the classical KRR algorithm, while Monte Carlo sampling methods are introduced to estimate the continuation value. To assess the accuracy of our value function estimator, we propose a natural error decomposition and rigorously control the resulting error terms at each time step. We then analyze how this error propagates backward in time-from maturity to the initial stage-a relatively underexplored aspec
    
[^22]: 随机因子模型中的最优投资与消费

    Optimal Investment and Consumption in a Stochastic Factor Model

    [https://arxiv.org/abs/2509.09452](https://arxiv.org/abs/2509.09452)

    本文通过建立无边界值开域上二阶常微分方程的下解与上解一般理论，首次为包括Heston模型在内的随机因子模型中的无穷期限最优投资与消费问题证明了HJB方程解的存在性并给出了严格验证，同时在有限状态情形给出了完整适定性刻画与高效数值算法。

    

    本文研究了在无穷时间期限下，具有幂效用投资者的不完备随机因子模型中的最优投资与消费问题。当随机因子的状态空间为有限集时，我们给出了该问题适定性的完整刻画，并提供了一种计算价值函数的高效数值算法。当状态空间是一个（可能无穷的）开区间且随机因子由伊藤扩散表示时，我们发展了一套关于无边界值的开域上二阶常微分方程的下解与上解的一般理论，用以证明哈密顿-雅可比-贝尔曼（HJB）方程解的存在性，并给出解的显式界。通过刻画解的渐近行为，我们还能为多种模型提供严格的验证论证，其中包括——首次——Heston模型。

    arXiv:2509.09452v2 Announce Type: replace  Abstract: In this article, we study optimal investment and consumption in an incomplete stochastic factor model for a power utility investor on the infinite horizon. When the state space of the stochastic factor is finite, we give a complete characterisation of the well-posedness of the problem, and provide an efficient numerical algorithm for computing the value function. When the state space is a (possibly infinite) open interval and the stochastic factor is represented by an It\^o diffusion, we develop a general theory of sub- and supersolutions for second-order ordinary differential equations on open domains without boundary values to prove existence of the solution to the Hamilton-Jacobi-Bellman (HJB) equation along with explicit bounds for the solution. By characterising the asymptotic behaviour of the solution, we are also able to provide rigorous verification arguments for various models, including -- for the first time -- the Heston m
    
[^23]: 自动做市商中流动性提供的最优费用

    Optimal Fees for Liquidity Provision in Automated Market Makers

    [https://arxiv.org/abs/2508.08152](https://arxiv.org/abs/2508.08152)

    该论文在一个AMM与中心化交易所并行、交易者最优选择交易场所的动态模型中，刻画了使LP利润最大化的固定流动性下最优费用以及竞争性进入下使均衡总锁仓价值（TVL）最大化的最优费用，并通过大规模模拟和市场数据校准验证了模型的有效性。

    

    自动做市商（AMM）中的被动流动性提供者（LP）由于逆向选择而面临损失（LVR），而静态交易费用在实践中往往无法弥补这些损失。我们在一个动态简化形式模型中研究LP盈利能力的关键决定因素，在该模型中，AMM与中心化交易所（CEX）并行运行，交易者将订单最优地路由到提供更优价格的交易场所，套利者则利用价格差异进行套利。通过大规模模拟，我们分析了LP利润如何随波动率和交易量等市场条件而变化，并刻画了固定流动性下利润最大化的AMM费用。随后，我们通过竞争性LP进入将流动性内生化，并刻画了使均衡总锁仓价值（TVL）最大化的费用。我们通过大量的比较静态分析揭示了驱动这些关系的机制，并通过市场数据校准确认了模型的相关性。

    arXiv:2508.08152v2 Announce Type: replace-cross  Abstract: Passive liquidity providers (LPs) in automated market makers (AMMs) face losses due to adverse selection (LVR), which static trading fees often fail to offset in practice. We study the key determinants of LP profitability in a dynamic reduced-form model where an AMM operates in parallel with a centralized exchange (CEX), traders route their orders optimally to the venue offering the better price, and arbitrageurs exploit price discrepancies. Using large-scale simulations, we analyze how LP profits vary with market conditions such as volatility and trading volume, and characterize the profit-maximizing AMM fee at fixed liquidity. We then endogenize liquidity through competitive LP entry and characterize the fee that maximizes equilibrium total value locked (TVL). We highlight the mechanisms driving these relationships through extensive comparative statics, and confirm the model's relevance through market data calibration. A key 
    
[^24]: 当违约无法对冲时：通过局部风险最小化计算xVA

    When defaults cannot be hedged: xVA calculations via local risk-minimization

    [https://arxiv.org/abs/2502.12774](https://arxiv.org/abs/2502.12774)

    该论文针对银行与交易对手违约均无法对冲的现实情形，提出利用局部风险最小化方法并通过BSDE求解，实现交易对手信用风险与融资成本（xVA）的定价与对冲。

    

    我们研究了在无法对冲银行自身或交易对手的违约跳跃的情况下，交易对手信用风险与融资成本的定价与对冲问题。这种情况在实践中最为常见，因为市场上往往缺乏以交易对手为标的的公开报价公司债券或CDS合约，同时银行也难以对自身违约进行买卖保护。我们应用局部风险最小化方法来寻找最优策略，并通过倒向随机微分方程（BSDE）对其进行计算。

    arXiv:2502.12774v3 Announce Type: replace  Abstract: We consider the pricing and hedging of counterparty credit risk and funding when there is no possibility to hedge the jump to default of either the bank or the counterparty. This represents the situation which is most often encountered in practice, due to the absence of quoted corporate bonds or CDS contracts written on the counterparty and the difficulty for the bank to buy/sell protection on her own default. We apply local risk-minimization to find the optimal strategy and compute it via a BSDE.
    
[^25]: 生命周期模型中的约束投资组合优化：一种深度定价核方法

    Constrained portfolio optimization in a life-cycle model: A deep pricing kernel approach

    [https://arxiv.org/abs/2410.20060](https://arxiv.org/abs/2410.20060)

    本文提出一种深度定价核方法，通过构建人工市场和对偶变换，有效求解生命周期模型中带凸集交易约束的投资组合优化问题，并给出紧致上下界。

    

    arXiv:2410.20060v4 公告类型：替换 摘要：本文研究广义生命周期模型中的约束投资组合优化问题。个体具有随机收入，管理由股票、债券和人寿保险组成的投资组合，以最大化其消费水平、死亡福利和终端财富。同时，个体面临凸集交易约束，其中不可交易资产约束、禁止卖空约束和禁止借款约束是特殊情况。我们通过操纵基础资产的补偿漂移项来构建人工市场，以满足交易约束。通过对偶变换，我们提出了一种深度定价核方法，用于计算原始问题的紧致下界和上界，该方法可在定价核的条件期望导致价值函数缺乏显式解时使用。最后，我们得出结论，在考虑交易约束时，个体将...

    arXiv:2410.20060v4 Announce Type: replace  Abstract: This paper considers the constrained portfolio optimization in a generalized life-cycle model. The individual with a stochastic income manages a portfolio consisting of stocks, a bond, and life insurance to maximize their consumption level, death benefit, and terminal wealth. Meanwhile, the individual faces a convex-set trading constraint, with the non-tradeable asset constraint, no short-selling constraint, and no borrowing constraint as special cases. We build the artificial markets to solve this problem by manipulating the compensated drift terms of the underlying assets to meet the trading constraints. By dual transform, we propose a deep pricing kernel approach to compute tight lower and upper bounds for the primal problem, which can be used when the value function lacks an explicit solution due to the pricing kernel's conditional expectation. Finally, we conclude that when considering the trading constraints, the individual wil
    

