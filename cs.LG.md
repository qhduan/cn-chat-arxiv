# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LongZero: Enhancing Long-Context Reasoning in LLMs from Zero Data](https://arxiv.org/abs/2610.30132) | LongZero提出了一种无需任何外部数据的自举式强化学习方法，让大语言模型通过自行合成结构化的问题-答案-证据任务来生成训练数据，从而有效提升自身的长上下文推理能力。 |
| [^2] | [Think Harder Where It Matters: Item-Level Adaptive Recursion for Sequential Recommendation](https://arxiv.org/abs/2610.30128) | 提出LoopRec，一种针对序列推荐的物品级自适应递归方法，根据每个物品的实际计算需求动态分配递归深度，取代固定计算预算，从而既节省计算资源又提升整体推荐质量。 |
| [^3] | [A Quasipolynomial Algorithm for $\ell_\infty$ Contractions](https://arxiv.org/abs/2610.30122) | 本文提出了一种在拟多项式时间内计算 $\ell_\infty$ 范数下非扩张映射的近似不动点的算法，由此推出简单随机博弈可以在拟多项式时间内求解。 |
| [^4] | [Self-normalized Bernstein Inequalities for Regenerative Processes and Applications to Learning Without Independence](https://arxiv.org/abs/2610.30119) | 该论文为具有独立同分布标记周期的再生过程证明了自归一化Bernstein不等式，其中与再生时间尾部相关的项在多数情形下仅为二阶小量，并据此为非独立同分布的再生数据分类建立了泛化误差界，进而提出满足预言机不等式的不可知分类器。 |
| [^5] | [The role of locality and weight sharing in convolutional neural networks](https://arxiv.org/abs/2610.30118) | 本研究通过师生模型的精确动力学分析揭示：卷积网络中权重共享带来的学习加速与输入位置数 $P$ 成正比，而局部性则决定样本复杂度，二者的影响均依赖于输入位置间的空间相关性。 |
| [^6] | [Minimax Optimality of the NPMLE for Gaussian Mixtures in Hellinger Distance](https://arxiv.org/abs/2610.30109) | 本文证明了基于n个样本的NPMLE在真实混合测度为次高斯时达到平方Hellinger距离下的极小极大最优速率log(n)/n，并在紧支撑情形下达到更快的log(n)/(n log log n)速率，其证明依赖于对有限高斯混合类局部Hellinger括号熵的精确刻画。 |
| [^7] | [Velocity Ambiguity Profiles: Time-Resolved Bayes-Risk Diagnostics for Flow Matching](https://arxiv.org/abs/2610.30107) | 该论文提出速度模糊剖面（VAP）这一新诊断工具，将流匹配的时间分辨损失分解为不可约的贝叶斯风险下界与可通过数据或改进估计器降低的超额项，并在两个结构化基准中证明了其统计收敛性质，从而揭示了流匹配在哪些时间区域受内在模糊性限制、哪些区域仍可由数据改进。 |
| [^8] | [Asymmetric On-Policy Distillation Learns In-Context Exploration](https://arxiv.org/abs/2610.30105) | 提出ASTEROID学生-教师蒸馏框架，让拥有特权信息的教师基于学生在部分可观测环境中自主收集的在策略历史上下文提供动作标签，从而使学生学会测试时的情境化探索行为。 |
| [^9] | [BatchBench: Benchmarking AI Robustness to Batch Effects and Domain Shift in Biomedical Images - A JUMP Cell Painting Leaderboard](https://arxiv.org/abs/2610.30104) | 该论文推出了BatchBench排行榜，基于大型公开生物医学成像数据集JUMP-CP，评估和比较AI系统在批处理效应和领域偏移下的鲁棒性。 |
| [^10] | [Differentially Private Sparse Reward Estimation with Preference Feedback](https://arxiv.org/abs/2610.30099) | 该论文首次在Bradley-Terry-Luce模型下为基于成对偏好反馈的差分隐私稀疏奖励估计建立了全面的理论框架，突破了现有隐私对齐方法误差界随模型维度多项式增长的瓶颈，使隐私保护的RLHF在过参数化大模型时代变得可行。 |
| [^11] | [VX-One: Typed, Calibrated Decision Heads for Real-Time Perimeter Triage - An empirical study of unknown-attack honesty, cascade gray zones, and calibration under drift](https://arxiv.org/abs/2610.30090) | 本文提出VX-One，一个将商用有类型决策原语（校准门控、含显式未知选项的闭集选择、序数分数）作为输出的小型多头网络，在CPU上以毫秒级延迟实现实时边界分流，并实证评估了未知攻击诚实性、级联灰色地带与漂移下的校准性能。 |
| [^12] | [Orbitol: Crystal Generation on Wyckoff Orbits](https://arxiv.org/abs/2610.30086) | Orbitol通过用Wyckoff轨道标记替代原子标记，实现了对空间群对称性的直接控制，能生成保持所要求空间群的稳定新颖晶体结构，且生成速度比现有对称性感知基线快两到三个数量级。 |
| [^13] | [Circuits of LLM Classifiers](https://arxiv.org/abs/2610.30076) | 该论文将LLM分类器视为对目标标签的含噪测量仪器，提出将其组织成自适应的查询与决策“电路”，刻画了最优电路结构及准确率的理论极限，并在七个分类任务上改进了成本-准确率前沿。 |
| [^14] | [From Classifier Heads to Encoder Prototypes: Personalized Federated Learning with Head-Derived Clustering](https://arxiv.org/abs/2610.30075) | HDC-FL利用联邦学习训练中已有的客户端分类器头进行聚类分组，为每组构建编码器原型并以软混合形式返回给客户端，无需平均分类器头、额外优化或微调即可实现个性化表示共享。 |
| [^15] | [Loopholing All the Way: Language Modeling with pure Self-Conditioning](https://arxiv.org/abs/2610.30068) | 提出LAW方法，将自条件化从辅助技巧提升为连续生成模型的核心机制，通过在去噪步骤间传播期望嵌入并在最后一次性解码，在数独、迷宫和GSM8K等任务上超越了现有的离散与连续扩散语言模型。 |
| [^16] | [Looped Uniform Diffusion Language Models](https://arxiv.org/abs/2610.30066) | 该论文首次将循环机制（在每步去噪中递归复用早期Transformer层）引入均匀扩散语言模型，在不改变训练目标和采样器的情况下显著提升训练效率，使循环UDLM以更少计算量即可达到非循环模型训练约0.5万亿token后的验证NLL水平。 |
| [^17] | [HGM-Dual: Lineage-Aware Guided Agent Harness Optimization for Long-Horizon Planning](https://arxiv.org/abs/2610.30042) | 提出了HGM-Dual框架，通过谱系感知上下文和诊断引导，实现面向规划智能体的预算高效框架演化，解决了开放式自我改进在长程规划任务中反复探索无效改动、消耗大量评估预算的问题。 |
| [^18] | [ExGrad: Beyond Suppression - Explicit Clean-Gradient Subtraction in Adversarial Training](https://arxiv.org/abs/2610.30018) | ExGrad 通过显式构建干净、对抗与 AWP 三类梯度并用全局可解的三变量优化进行重组，证明在对抗训练中对干净梯度进行适度减除（而非单纯抑制或置零）能够显著提升 PGD-20 鲁棒性。 |
| [^19] | [Code Comprehensor: Hierarchical Documentation and Cross-Modal Retrieval for Large Repositories](https://arxiv.org/abs/2610.30016) | 该论文提出Code Comprehensor混合检索-推理框架，通过代码依赖调用图、层次化文档生成（CodeScribe）和CLIP式跨模态检索模型（CodeContrast），实现对大型代码库相关代码上下文的快速语义精准检索，并为基于MCP的LLM问答提供可靠依据。 |
| [^20] | [Agent Behavior Maps: Locating Agents by What They Would Do, Without Running Them](https://arxiv.org/abs/2610.30014) | 提出“智能体行为地图”方法，利用共享参考轨迹上的中心化动作对数似然为LLM智能体定位，无需实际运行目标智能体即可刻画其行为特征、追踪轨迹KL散度并支持性能预测与行为相似性分析。 |
| [^21] | [Mitigating Model Collapse through Dreaming Learning](https://arxiv.org/abs/2610.30009) | 该论文从谱分析角度揭示低温采样（Ts < 1）会使转移算子锐化、谱隙收窄从而导致模型坍塌，并提出“梦境学习”正则化方法——在训练中对模型以高温生成的自身样本施加额外损失项——来缓解这一退化。 |
| [^22] | [EPIC: Rethinking Trust Regions for Autoregressive LLM Policies via Exact-KL Prefix Importance-Ratio Clipping](https://arxiv.org/abs/2610.29999) | 该论文提出EPIC方法，将自回归生成建模为有限时域MDP并推导性能差恒等式，证明生成前缀的累积似然比才是逐步策略改进的精确重要性权重，并据此用精确反向KL裁剪构建前缀级信任域，从而修正标准PPO token级裁剪忽略前缀分布偏移所导致的偏差。 |
| [^23] | [SCDM: Scalable Causal Discovery in Nonlinear Temporal Systems with Meta-Learning](https://arxiv.org/abs/2610.29992) | 该论文提出SCDM框架，通过元学习将每个目标变量作为任务并共享时序预测器与因果强度矩阵，避免了穷举式逐目标优化，从而实现高维非线性时序系统的可扩展因果发现。 |
| [^24] | [TIDE: Tree-Induced Fine-Grained Diagnostic Evaluation from Model Behaviors](https://arxiv.org/abs/2610.29985) | TIDE 将诊断评估转化为行为驱动的决策树归纳，通过从基准样本和模型分数中自动构建分层能力树，实现了对大语言模型能力的细粒度诊断，揭示出聚合分数无法反映的能力差异。 |
| [^25] | [The Price of Current-Effect Inference in Nonstationary Multi-Armed Bandits](https://arxiv.org/abs/2610.29982) | 该论文揭示了非平稳多臂老虎机中收益最大化与现时效应推断之间的内在权衡，证明了以有限宽度置信区间同时覆盖所有现时效应所需付出的动态遗憾极小极大代价，并提出 TRACE 算法来同时兼顾这两个目标。 |
| [^26] | [CORD: Conflict-Aware Optimization for Joint RL and Distillation in Agents](https://arxiv.org/abs/2610.29965) | 提出CORD冲突感知优化框架，通过最近点投影以闭式更新有选择地消除强化学习与蒸馏目标间相互抵消的梯度分量，从而在多种智能体联合训练方案中提升训练效果。 |
| [^27] | [Mastering Model Fit: Applications of the InterModel Vigorish for Interpretable Machine Learning](https://arxiv.org/abs/2610.29960) | 本文将“模型间赌利（IMV）”引入机器学习领域，通过 SHAP-IMV、Multi-IMV 和 Ablate-IMV 等可解释工具，将模型拟合差异量化为相对于参考模型的期望投注优势，从而揭示哪些特征、模型组件和训练阶段真正改善了概率拟合。 |
| [^28] | [FrontierCS 2: A Benchmark for Computer Science R&D](https://arxiv.org/abs/2610.29932) | FrontierCS 2 通过删除论文贡献并将智能体置于真实研究环境中迭代提交补丁的方式，构建了一个包含 48 个开放式任务的基准，用于评估大语言模型智能体在前沿计算机科学研究与开发中的实际贡献能力。 |
| [^29] | [Your Benchmark Is an Empirical Measure Over Difficulty](https://arxiv.org/abs/2610.29927) | 本文提出将基准测试视为难度上的经验测度这一新视角，识别并形式化了“测度失配”与“覆盖缺口”两种失效模式，提出两个具有严格理论保证的诊断指标，并对7个领域25个流行基准的分析表明许多基准的难度权重隐式失衡，其排行榜结论需谨慎解读。 |
| [^30] | [Privacy in Context Is Being Overlooked in the Age of General-Purpose LLMs](https://arxiv.org/abs/2610.29925) | 本文指出隐私本质上是情境相关的，当前为LLM提示隐私构建统一框架的研究方向具有误导性并可能带来虚假安全感，应转向面向特定应用的部署与面向特定情境的评估。 |
| [^31] | [Unintended Side Effects of Value Training in LLMs](https://arxiv.org/abs/2610.29922) | 本文通过在霍夫斯泰德各价值维度上微调四个大语言模型，发现针对特定价值的训练会对未训练的价值观、风险偏好、社会认知和安全特性产生意外副作用，因此建议价值对齐评估应纳入更广泛的行为测试。 |
| [^32] | [Preventing Model Extraction in Contextual Linear Optimization](https://arxiv.org/abs/2610.29918) | 该论文针对预测后优化（PTO）系统研究了模型提取攻击与防御，刻画了线性模型可被攻击者通过决策观察提取的条件，并提出了一种仅在预测成本向量接近决策边界时才扰动决策的防御机制，以在抵御提取攻击的同时控制决策遗憾。 |
| [^33] | [Tractable Inference with Kolmogorov-Arnold Networks](https://arxiv.org/abs/2610.29909) | 该论文提出TOUCKAN，通过为KAN网络引入有限次多项式外函数形成可分解的和-积表示，使期望、高阶矩等统计量能够通过确定性递归精确计算，从而避免了高维场景下昂贵的蒙特卡洛采样。 |
| [^34] | [A Sharp Dynamic Haussler Lemma](https://arxiv.org/abs/2610.29898) | 本文将 Haussler 覆盖引理推广到序贯设定：对归一化 ℓ_p（1≤p<∞）中的有界实值类，存在与时间范围无关、规模为 (C/ε^p)^d 的全局动态覆盖（d 为序列 fat 维数），并给出匹配的下界，从而在 p=2 时解决了 Rakhlin 等人提出的序贯 ℓ₂ 覆盖问题。 |
| [^35] | [Silent Alignment in Two-Layer Linear Networks: A High-Dimensional Perspective](https://arxiv.org/abs/2610.29864) | 该论文在高维比例极限下借助随机矩阵理论精确刻画了双层线性网络梯度流的谱演化，并发现小初始化时存在“静默对齐”现象——网络奇异方向在一阶时间尺度上先完成对齐而损失保持不变，可见的学习随后才在对数时间尺度上以鞍点到鞍点的方式发生。 |
| [^36] | [RangeNav: Task-Aligned LiDAR Range-Image Pretraining for Robot Navigation](https://arxiv.org/abs/2610.29845) | 提出RangeNav，一个仅18.3万参数、完全基于合成数据预训练的轻量级激光雷达距离图像编码器，通过结合距离重建、语义分割和鸟瞰图预测的导航对齐目标，为大规模并行强化学习的机器人导航提供高效的360°感知表征。 |
| [^37] | [NeuroCodec: EEG-Guided Latent-Domain Generative Model for High-Fidelity Target Speaker Extraction](https://arxiv.org/abs/2610.29815) | 提出NeuroCodec，首个在神经音频编解码器潜在空间中运行的生成式EEG引导目标说话人提取框架，通过图卷积建模EEG空间依赖并结合注意力机制与Mamba时序建模进行跨模态融合，突破了传统时频掩蔽方法保真度低和仅限窄带音频的局限。 |
| [^38] | [Real PMIDs, Wrong Papers: Harness Design for Biomedical Claim-Verification Agents](https://arxiv.org/abs/2610.29812) | 该论文通过受控消融实验发现，在基于实时PubMed API的生物医学论断验证任务中，让语言智能体自行构造查询的框架设计能将标准论文召回率提升三倍并取得最佳准确率（62%），而简单的单次检索仅能为6%的论断找到标准论文且无法提升准确率。 |
| [^39] | [Asymmetric Likelihood-Free Hypothesis Testing](https://arxiv.org/abs/2610.29798) | 本文刻画了非对称无似然假设检验中可行的样本预算三元组（至多相差对数因子），证明对参考分布 P 和 Q 分配不等的样本预算可带来多项式级别的节省——例如将来自 D 的所需样本量从 Ω(√n) 降至 Õ(n^(1/4))——并将其应用于非自适应分布 2-聚类问题。 |
| [^40] | [Where Do LLMs Learn to Recommend AI?](https://arxiv.org/abs/2610.29785) | 研究发现LLM过度推荐AI的倾向主要源于指令微调数据中约7%的AI相关样本——剔除这些数据可将推荐率降至预训练模型水平以下，而追求“有帮助”的数据和“最佳”“立刻”式的提问方式则会进一步推高该倾向。 |
| [^41] | [Monitorability Is Trainable: Post-Training for Chain-of-Thought Monitorability](https://arxiv.org/abs/2610.29734) | 本文提出了一种通过最大化思维链与输出之间互信息的无监督后训练目标，能够恢复退化模型损失的思维链可监控性，并进一步提升未退化模型的监控效果与不安全行为预测能力。 |
| [^42] | [Post-hoc Epistemic Uncertainty for Vision-Language-Action Policies](https://arxiv.org/abs/2610.29730) | 提出了一种后验核方差头方法，无需训练多个策略或修改原策略，即可从单一VLA策略的嵌入中估计认知不确定性，其表现与双策略集成方法相当。 |
| [^43] | [Structure-Aware Message Passing on Petri Nets through Directed Walks and Incidence Diffusion](https://arxiv.org/abs/2610.29725) | 该论文提出PNRWPE位置编码和DA-PNGNN两种结构感知机制，使图神经网络能够显式捕捉佩特网中的有向循环（过程回路）和关联矩阵不变量，将预测模型生成因子的宏平衡准确率从0.751提升至0.796，其中过程回路上的准确率几乎翻倍。 |
| [^44] | [Rethinking Post-hoc Debiasing of Large Pretrained Models](https://arxiv.org/abs/2610.29688) | 本文揭示了现有事后去偏方法依赖文本表达偏差特征的局限，证明在多偏差属性场景下其表现甚至不及普通分类器，并指出这些方法所寻求的偏差模式其实可以直接从冻结特征的几何结构中观察到。 |
| [^45] | [TraceForge: Distilling with Your Own Thinking](https://arxiv.org/abs/2610.29680) | 提出TraceForge，以教师模型的答案为输入、按学生模型自身家族的风格生成推理轨迹，并通过“监督+对抗”两阶段训练，在不违反服务条款的前提下实现对闭源前沿模型的推理蒸馏。 |
| [^46] | [Lifting PCA in the Wasserstein Space](https://arxiv.org/abs/2610.29679) | 本文提出了“提升PCA”，通过凸延拓提升泛函的框架，首次将欧氏主成分分析规范化地扩展到Wasserstein空间，并将其归结为两个等价的、具有广义代价的Wasserstein重心最小化问题。 |
| [^47] | [DriveMLP: When Training Strategy Outperforms Model Complexity in Autonomous Driving](https://arxiv.org/abs/2610.29676) | 该论文提出专家引导强化学习（EGRL）训练策略，并训练出轻量级的DriveMLP策略，证明在自动驾驶闭环运动规划中，训练方法而非模型复杂度才是性能提升的关键因素。 |
| [^48] | [Characterizing Oblivious Common-Reward Multi-Player Bandits: Universal Stationarity, Coordination Traps, and Certified Global Rates](https://arxiv.org/abs/2610.29673) | 本文证明在无感知共同奖励多玩家多臂老虎机中，玩家独立的梯度更新共同构成隐藏乘积策略目标的无偏随机梯度估计，从而获得适用于所有实例的通用有限时间一阶平稳性保证，同时通过乘积策略陷阱障碍表明平稳性一般无法提升为全局最优性。 |
| [^49] | [Leveraging Time-Series Foundation Models for Financial Crime Detection](https://arxiv.org/abs/2610.29667) | 该论文提出TCEE框架，利用冻结的时间序列基础模型总结账户交易历史的时间上下文来增强交易边的表示，在六个AMLWorld数据集上将PR-AUC相比最强基线平均相对提升10.8%，同时训练速度提升6倍。 |
| [^50] | [Lineage-Aware Search Control for LLM-Based Evolutionary Factor Mining](https://arxiv.org/abs/2610.29663) | 提出LineageEvo框架，通过谱系条件化的LLM变异将谱系范围搜索经验提炼为算子先验并与全局先验融合，为LLM进化因子挖掘引入谱系层面的搜索控制，克服了传统方法搜索方向性弱、因子结构同质化的局限。 |
| [^51] | [SPARE: Sparsity Allocation via Routing Entropy for Efficient MoEs](https://arxiv.org/abs/2610.29662) | SPARE是一种无需训练的MoE专家剪枝方法，利用路由熵自适应地为各层分配不同的剪枝稀疏度，在相同预算下比均匀剪枝获得更高的模型准确率。 |
| [^52] | [Probabilistic Machine Learning for Short-Term Atmospheric Seeing Forecasting](https://arxiv.org/abs/2610.29654) | 该论文将高斯过程和基于归一化流的概率模型与确定性RNN/LSTM基线进行对比，实现了莫纳克亚山顶大气视宁度提前至两小时的概率预报及不确定性量化。 |
| [^53] | [Recursive Code Evolution for Task-Oriented Semantic Communications](https://arxiv.org/abs/2610.29641) | 该论文提出ComRSI框架，通过大语言模型辅助的递归代码自我改进来自动演进语义通信程序，最终仅用每张图像32个复符号就实现了74.3%的测试精度，优于使用8倍符号资源的初始设计。 |
| [^54] | [Cross-Area Transfer Learning for VLC-based Indoor Localization with a Transfer Efficiency Score](https://arxiv.org/abs/2610.29638) | 本文提出了一种面向VLC室内定位的跨区域迁移学习框架，仅用40%的目标区域数据微调预训练模型即可恢复97%的完整数据性能，并引入迁移效率评分（TES）指标，无需穷举评估即可识别最佳迁移源模型。 |
| [^55] | [Space weather dependence of decameter-scale ionospheric irregularities spatial distribution](https://arxiv.org/abs/2610.29630) | 本文提出了一种基于SuperDARN和SECIRA雷达数据与空间天气参数、采用受图神经网络启发的原创神经网络架构的数据驱动多尺度方法，用于高纬度十米尺度电离层不均匀体空间分布的即时预报和预测。 |
| [^56] | [Reliability-Aware Probabilistic Fusion for Multimodal Learning with Incomplete Evidence](https://arxiv.org/abs/2610.29621) | 该论文提出一种可靠性感知的高斯乘积专家方法，利用输入相关的可靠性权重调节各模态精度并自动忽略缺失专家，从而在模态缺失的临床多模态任务（如HAIM-MIMIC-MM）中取得明显的AUROC性能优势。 |
| [^57] | [Watch Your Step: Anticipated Perception for Humanoid Locomotion](https://arxiv.org/abs/2610.29612) | 提出一种轻量级落足点预测模型，仅凭本体感知和速度指令预测未来落足点，并只在高程图中采样这些落足点周围的小区域作为策略输入，从而在感知进入策略之前剔除冗余信息，加速训练并提升对新地形的泛化能力。 |
| [^58] | [No Frames, Just Frequencies: Bandwidth-Efficient Deployment-Time Adaptation for Smart Glasses](https://arxiv.org/abs/2610.29607) | 提出了一种带宽高效的自适应框架，智能眼镜只需在设备端计算并发送紧凑的频率描述符而非视频帧，由主机设备据此合成分布外数据并自适应模型副本，从而在保障隐私的同时实现智能眼镜的部署时模型自适应。 |
| [^59] | [Improved Training of Strong Stochastic Flow Maps](https://arxiv.org/abs/2610.29605) | 本文通过使用移位勒让德多项式展开来表示布朗路径，并结合 Wong-Zakai 风格的目标函数，改进了强随机流映射的训练，实现了基于导数的损失函数。 |
| [^60] | [Local and Global Recursive Ridge Regression for Online Residual Correction of Frozen Multivariate Forecasters under Label Delay](https://arxiv.org/abs/2610.29604) | 提出DRiC方法，利用基于主DCT系数的递归岭回归和固定规模充分统计量，在标签延迟与小状态约束下对冻结多元预测器进行在线残差校正，提供按通道建模的局部变体和跨通道池化的全局变体，并通过残差尺度保护机制抵御异常数据块。 |
| [^61] | [A Diagnostic Study of Faithfulness in Diffusion Language Models for Table-to-Text Generation](https://arxiv.org/abs/2610.29602) | 本研究诊断发现，扩散语言模型在表格到文本生成中随示例增加或微调后流畅度和BLEU提升，但对源表格的事实忠实性反而下降，揭示了表面流畅性与事实依据性的分离。 |
| [^62] | [Near Minimax Rate for Quantile Distributional Reinforcement Learning](https://arxiv.org/abs/2610.29595) | 本文在生成模型假设下，为分位数Bellman不动点估计建立了在所有分辨率和所有Wasserstein阶数上一致成立的 $\sqrt{n}$ 高概率误差界，达到了分位数分布策略评估的近极小极大最优速率。 |
| [^63] | [A Transfer Learning-Driven Methodology for Efficient and Sustainable Indoor Localization Using VLC](https://arxiv.org/abs/2610.29594) | 本文提出一种基于迁移学习的可见光通信室内定位方法，能够有效应对光照波动和物理障碍等环境变化，同时显著降低计算开销与能耗。 |
| [^64] | [Fragment-Level Graph Encoding for Interpretable Molecular Property Prediction](https://arxiv.org/abs/2610.29589) | 该论文提出用BRICS分解将分子编码为片段级图取代原子级图，在不改变模型结构的前提下使GNN能够给出功能团级别的可解释预测，并在约五分之一大小的图上达到与原子级编码相当的预测精度。 |
| [^65] | [Unlocking Spatial Grounding in Flow-Matching Models via SNR Tuning](https://arxiv.org/abs/2610.29581) | 该论文发现信噪比（SNR）分布是流匹配模型空间接地能力的关键决定因素，指出高噪声区间是空间学习的“关键学习阶段”，通过调节SNR训练分布即可显著提升模型对空间布局的精细控制，而无需额外的结构适配器或复杂的推理时引导。 |
| [^66] | [Fast Bayesian Optimization Through Linear Models](https://arxiv.org/abs/2610.29580) | 本文提出了利用现有贝叶斯优化算法线性结构的新型快速BO方法，在保持与最先进算法相当性能的同时将速度提升100至1000倍以上，从而将贝叶斯优化的适用范围拓展到成本敏感的全新问题类别。 |
| [^67] | [What Prediction-Market Contracts Reveal at Launch: Trading Volume and Dispute Risk](https://arxiv.org/abs/2610.29578) | 仅凭预测市场合约上线时的可解释信息（如市场主题、措辞和结算规则）即可高度准确地预测其未来交易量与结算争议风险，但这种预测能力会随平台环境和上线格式的变化而显著衰减。 |
| [^68] | [The Reflexivity Threshold: A Phase Transition for Multi-Agent Performative Prediction](https://arxiv.org/abs/2610.29577) | 该论文提出用“反身性矩阵”刻画多智能体操演性预测中跨智能体反馈的网络结构，并证明当该矩阵的谱半径越过 1 这一阈值时，重复再训练会发生从稳定收敛到最坏情况学习受阻的相变。 |
| [^69] | [Private Online Prediction from Experts with Small Losses](https://arxiv.org/abs/2610.29575) | 该论文提出了首个遗憾度达到 $\tilde{O}(\sqrt{L^\star})$ 的差分隐私专家在线预测算法，在低损失情形下显著优于现有 $\tilde{O}(\sqrt{T})$ 界限，并证明在忘却输入下隐私代价很小。 |
| [^70] | [Implicit Feature Reweighting for Label-Free Graph Generalization](https://arxiv.org/abs/2610.29571) | 提出了无标签表示学习框架SCARCE，通过变分图自编码器与对抗混淆机制，仅利用训练图的环境划分即可学习到对分布偏移鲁棒的图表示，无需类别标签即实现图级别的分布外泛化。 |
| [^71] | [Certified Neural Approximation of Robust Nonlinear Model Predictive Control](https://arxiv.org/abs/2610.29559) | 该论文提出一个框架，通过将近似误差作为有界输入扰动纳入鲁棒MPC设计，提供先验的构造性设计条件，使神经网络控制器在部署前即可继承模型预测控制的递归可行性与稳定性保证。 |
| [^72] | [SOAR: Social and Opinion-Aware Recommendations](https://arxiv.org/abs/2610.29556) | 提出了SOAR推荐模型，通过结合用户社交好友网络与文本评论的情感信息，实现更精准的个性化兴趣点推荐。 |
| [^73] | [WRDO: Weighted Refusal Direction Optimization in Aligned Language Models](https://arxiv.org/abs/2610.29555) | 提出加权拒绝方向优化方法，通过学习聚合权重从针对每个提示的方向库中构建复合拒绝方向，以强化对齐语言模型的拒绝干预效果。 |
| [^74] | [Hyperspherical Simplex Representations for Backward-Compatible Retrieval](https://arxiv.org/abs/2610.29550) | 提出由softmax输出和logits导出的超球面单纯形表示，利用softmax交叉熵训练所诱导的单纯形结构作为跨模型更新的共同几何参考，无需专门兼容训练、映射函数或依赖旧模型即可实现独立训练模型间的后向兼容检索。 |
| [^75] | [Stochastic Pruning of the DPO Computation Graph: 1.4x Faster Training and Lower Peak Memory](https://arxiv.org/abs/2610.29538) | 提出DPO计算图随机剪枝的统一框架，通过独立采样打分位置、执行前缀与反向传播分支，实现1.4倍训练加速并显著降低峰值内存。 |
| [^76] | [Block Diffusion for Longitudinal EHR: Accuracy, Efficiency, and Representation Trade-offs](https://arxiv.org/abs/2610.29533) | 基于块的掩码扩散模型可作为自回归模型在纵向电子健康记录预测中的高效替代方案，小方块配置在保持接近AR预测性能的同时实现约1.43倍的推理加速，且学习到的临床表示信息量与AR相当。 |
| [^77] | [Phase Transitions in Learning Driven by Extremal Data](https://arxiv.org/abs/2610.29531) | 该论文通过研究多示例学习中的极值回归问题，推导出梯度流的低维投影，揭示了当softmax近似最大池化的锐度参数 $\lambda$ 增大使动力学进入梯度消失区域时，学习会发生相变，从而刻画了梯度下降成功与失效的条件。 |
| [^78] | [Residual-Based Causal Discovery via Pretrained In-Context Quantile Regression](https://arxiv.org/abs/2610.29530) | 该论文提出Loscar方法，将RESIT框架从加性噪声模型扩展至更广泛的位置-尺度噪声模型，通过分位数回归非参数化地估计条件均值和尺度，并支持将预训练的表格基础模型作为即插即用估计器，从而避免了现有方法对参数化假设的依赖。 |
| [^79] | [Are Joint-Embedding Predictive Objectives Effective for Longitudinal EHRs?](https://arxiv.org/abs/2610.29528) | 该研究表明，尽管JEPA方法在纵向电子健康记录的表示学习中能够稳定训练且不发生表示坍塌，但其性能始终不如标准的掩码语言建模，且减小预测目标规模也无法改善下游任务表现。 |
| [^80] | [Geometric knowledge spaces, Attention pairing and Architectures](https://arxiv.org/abs/2610.29526) | 本文提出一个几何化框架，通过形式化“知识世界”与“注意力配对”的概念，统一推导出标准注意力、因果注意力、线性注意力、张量注意力和循环神经网络等多种架构，并引入新的几何空间以启发未来神经网络架构的设计。 |
| [^81] | [pplx-embed-v2-late Technical Report](https://arxiv.org/abs/2610.29523) | Perplexity 推出的 pplx-embed-v2-late 包含 0.6B 和 9B 两个多语言多模态后期交互模型，通过 18B 教师模型的表示蒸馏训练，在文本、图像和视觉文档检索上达到同规模最先进水平，并借助共享嵌入空间支持“9B 建索引、0.6B 低延迟查询”的非对称部署。 |
| [^82] | [Supervision-Adaptive Curriculum and Guarded Anti-Curriculum Learning for Dense Prediction](https://arxiv.org/abs/2610.29519) | 提出 PaceShift 方法，采用“先稳定、后挑战”的范式，将监督自适应课程选择与受保护的谱后训练相结合，通过区分监督可靠性与时间更新价值来提升密集预测的训练效果。 |
| [^83] | [Mitigating Query Shortcuts in Chain-of-Thought with Context-Aware Decoding](https://arxiv.org/abs/2610.29518) | 提出推理依据对比解码（RCD）方法，通过对比查询在有无推理依据时的词元分布来放大推理依据的作用，从而缓解LLM在思维链推理中依赖查询捷径的行为并提升答案准确率。 |
| [^84] | [BRIDGE: An Acceptance Gate to Block False Improvements in Model Updates, with a Retrospective Clinical Evaluation](https://arxiv.org/abs/2610.29515) | BRIDGE 提出了一种模型无关的接受门控机制，通过交叉拟合与 bootstrap 置信下界检验来拦截模型更新中的虚假改进，将虚假改进率从约 20% 降至接近零，并经 12 项回顾性临床任务验证有效。 |
| [^85] | [A unified approach for estimation of unnormalized models via Stein operators](https://arxiv.org/abs/2610.29510) | 本文建立了一个连接适当评分规则与Stein方法的统一框架，证明分数匹配、比率匹配、伪似然和噪声对比估计等评分规则可自然诱导Stein算子，并借助由此产生的几何视角统一刻画了相应估计量的渐近效率。 |
| [^86] | [Shortcut-Resistant CAM Distillation for Long-Tailed Recognition](https://arxiv.org/abs/2610.29508) | 提出抗捷径CAM蒸馏框架SRCD，通过将头部类别以物体为中心的类激活图解释蒸馏到尾部类别，有效缓解长尾识别中尾部类别因上下文相似而容易学习捷径的问题。 |
| [^87] | [Asynchronous, Generation-Free ABC: Streaming AMIS Reweighting for Heterogeneous Simulator Workloads on HPC](https://arxiv.org/abs/2610.29506) | 该论文提出了一种无代际的异步ABC采样器，通过在模拟结果到达时即时更新提议分布，并利用流式AMIS重加权复用全部历史粒子，消除了并行计算中的同步屏障，显著提升了异构模拟器工作负载在高性能计算环境下的吞吐量。 |
| [^88] | [Rethinking Agentic Model Orchestration for Time Series Forecasting](https://arxiv.org/abs/2610.29504) | 该论文提出FairAMO受控评估协议，揭示智能体模型编排（AMO）在时间序列预测中的收益本质上受限于候选基础模型本身，其关键在于模型互补性而非数量，且相对最佳单一模型的MASE改进不足6%。 |
| [^89] | [Generative Modeling via Nonlinear Schr\"odinger Bridges and Branching Diffusions](https://arxiv.org/abs/2610.29476) | 该论文提出了基于前向-后向偏微分方程的非线性薛定谔桥生成建模框架SBNL，并利用分支扩散设计出具有高度并行性和高维可扩展性的高效数值算法。 |
| [^90] | [Density-Level Backward Error for Low-NFE Generative Flows](https://arxiv.org/abs/2610.29475) | 本文提出密度层面的后向误差框架，分离出真正改变边缘分布的求解器误差分量，从而为低NFE生成流导出感知求解器阶数的计算分配规则以及基于规范自由度的速度场选择目标。 |
| [^91] | [Asymmetry Tunneling Beyond Adaptive Equivariance Relaxation](https://arxiv.org/abs/2610.29473) | 该研究发现，即使刻意限制严格等变模型与残差路径之间的接口，打破对称性的任务相关结构仍能通过残差路径“隧穿”进入模型（即“非对称隧穿”现象），表明等变性松弛未必需要自适应控制。 |
| [^92] | [OmniSimulator: Aligning Small Language Models for Authentic Heterogeneous Behavior Modeling](https://arxiv.org/abs/2610.29472) | 本文提出 OmniSimulator 训练框架，通过五维结构化行为推理模式、精选真实用户行为数据以及教师模型生成的高质量推理轨迹，对齐小型语言模型，从而显著提升其在短视频平台上进行真实异构用户行为模拟的预测能力。 |
| [^93] | [Harness Scaling for Generalizable Agentic Reinforcement Learning](https://arxiv.org/abs/2610.29458) | 提出Harness Scaling这一智能体强化学习的新扩展维度，将训练从单一固定框架扩展到多样化执行配置，从而提升模型在不同框架配置下的泛化与迁移能力。 |
| [^94] | [How Far Can Foundation Models Be Distilled for Efficient Jet Tagging?](https://arxiv.org/abs/2610.29439) | 本研究系统地将六个教师模型（包括两个基础模型）蒸馏到小型学生模型中用于喷注标记，发现每位教师都能提升每位学生且增益主要取决于教师强弱，而仅两万参数的蒸馏版L-GATr-slim即可超越所有已发表的从头训练的喷注标记器。 |
| [^95] | [One Act, Many Masks: Effect-Marginal Policy Optimization for Long-Horizon Agents](https://arxiv.org/abs/2610.29427) | 该论文发现了多轮强化学习中信用信号与具体文本响应绑定所导致的“信用更新错位”问题，并提出效应边缘策略优化（EMPO），通过效应级别的代理目标将策略更新与执行层面的等价性对齐，从而改进长程LLM智能体的策略优化。 |
| [^96] | [BIRD-RL: Scaling Agentic Reinforcement Learning over Stateful Data-Centric Environments](https://arxiv.org/abs/2610.29426) | BIRD-RL提出了一种四阶段智能体强化学习课程方法，通过交替增强基础推理与轨迹自适应在线探索，解决了SQL调试等有状态数据中心环境中环境隔离成本高、状态污染和跨轨迹干扰导致的训练不稳定问题。 |
| [^97] | [Generalization Bounds via Vector-Valued $L_p$ Correlation $V_\alpha$](https://arxiv.org/abs/2610.29425) | 该论文提出了一族单参数依赖性度量 $V_\alpha$ 并据此推导出新的泛化界，这些界在 $\alpha=1$ 时退化为全变差界、在 $\alpha=2$ 时与方差和 $\chi^2$ 行为相关，且在 $\alpha>2$ 时即使损失方差无穷大也能保持有限。 |
| [^98] | [When Thoughts Become Memory: Unifying In-Context Learning and Fine-Tuning](https://arxiv.org/abs/2610.29419) | 该论文首次提出基于信息价值精确定律的认证规则，统一判定未见数据应放入预训练模型的上下文还是其权重，并通过随时有效的检验在多种模型与任务上验证了其正确性。 |
| [^99] | [Alternative Strategies for Active Learning with Bayesian Bradley Terry Model](https://arxiv.org/abs/2610.29416) | 该论文开发了基于Polya-Gamma数据增强的贝叶斯Bradley-Terry模型，提出三种基于知识梯度的主动学习策略，并引入无需参考排名即可评估排名可靠性的后验分离可靠性（PSR）指标。 |
| [^100] | [The Predictive--Optimization Gap in GuanDan Self-Play](https://arxiv.org/abs/2610.29411) | 在掼蛋自博弈中，对局进度虽被证明具有真实预测价值，但以冻结胜负预测器的变化 ΔΦ 作回合级监督干预，尽管在个别训练谱系中奏效，五个预注册实验在方法层面均未复现增益，揭示了预测有效性与优化效用之间的差距。 |
| [^101] | [Region-Aware Multi-View Inspection Using Deep Learning and Domain-Informed Decision Fusion](https://arxiv.org/abs/2610.29398) | 该论文提出一种区域感知的双模型检测框架，将座圈区域缺陷的实例分割与轴区域异常的图像分类相结合，并通过形式化工业标准作业程序的确定性决策融合层，对360度旋转采集的多视角图像检测结果进行融合，实现精密部件的部件级自动判定。 |
| [^102] | [Progressive Permutation Alignment with Temperature-Controlled Spectral Sharpening](https://arxiv.org/abs/2610.29396) | 该论文提出一种渐进式置换对齐方法，通过从输入到输出的分层对齐策略，结合匈牙利算法的离散锚点与温度控制的谱退火锐化技术，解决隐式神经表示中因置换对称性导致的权重空间模型对齐难题。 |
| [^103] | [Nominate-then-Adjudicate: LLM-Assisted One-Pass and Per-Instance MIP Solver Tuning](https://arxiv.org/abs/2610.29391) | 该论文提出“先提名后裁决”流水线：先一次性采样并复用固定配置组合作为候选名单，再由无任务微调的大语言模型在推理时以一次模型调用、零求解器调用完成逐实例选择，从而无需昂贵的逐实例配置搜索即可实现MIP求解器的逐实例调优。 |
| [^104] | [Parallelizable Probabilistic Latent Reasoning for Diverse Reasoning Trajectories](https://arxiv.org/abs/2610.29385) | 该论文提出了一种可并行的概率潜在推理方法，通过在紧凑连续的潜在空间中对多个有效推理轨迹的分布进行建模，既实现了多样化推理轨迹的探索，又支持高效的并行化训练。 |
| [^105] | [Cascading Bandits with Noisy Observations: Algorithms and Regret Bounds](https://arxiv.org/abs/2610.29381) | 该论文提出了一种探测成本与位置相关且反馈含噪声的随机级联赌博机模型，刻画了最优短列表可通过按每臂分数排序求得的条件，并给出了实例相关遗憾匹配理论下界的UCB算法。 |
| [^106] | [Right On Q: Recovering Reasoning under Quantization via Self-Generated Trajectories](https://arxiv.org/abs/2610.29378) | 提出OnQ流水线，通过在策略蒸馏利用学生模型自生成的轨迹进行训练，解决了固定响应监督导致的训练与生成前缀不匹配及高熵处分布失配问题，从而恢复低比特量化大模型的推理能力。 |
| [^107] | [Compositional neural networks for multimodal X-ray and neutron scattering inversion](https://arxiv.org/abs/2610.29362) | 该论文提出了一种基于先验摊销流匹配的模块化框架，通过在推理阶段用组合采样器融合分别训练的单观测后验估计器，实现多模态X射线和中子散射数据的概率性联合反演，无需联合再训练即可复用预训练模型。 |
| [^108] | [GaP-MoE : Explicit Expert Specialization in Mixture-of-Experts via Orthogonal Gradient Accumulation and Projection](https://arxiv.org/abs/2610.29355) | 提出GaP-MoE方法，通过正交梯度累积与投影直接操纵专家输出梯度而非修改损失函数，从而在稀疏专家混合模型中诱导显式的专家专业化，解决表示坍缩问题。 |
| [^109] | [Closed-Loop Dual-Reference Guidance for Accent Control in Cross-Lingual Voice Cloning](https://arxiv.org/abs/2610.29351) | 提出闭环双参考引导方法，通过将原始参考与语音转换构建的母语锚点配对并动态更新语言模型状态，有效消除跨语言语音克隆中的源语言口音，在仅损失极小说话人相似度的前提下显著提升目标语言的发音质量。 |
| [^110] | [Data Schedules and Learning Rate Schedules Should Be Designed Together](https://arxiv.org/abs/2610.29348) | 研究表明数据调度与学习率调度必须协同设计：中期训练中最优的数据过渡策略依赖于学习率调度，当学习率持续衰减时，更早引入中期训练数据并与预训练数据混合能降低最终损失。 |
| [^111] | [Benchmarking and Error Analysis of Deep Search and Deep Research LLM Agents](https://arxiv.org/abs/2610.29346) | 本文针对深度搜索与深度研究LLM智能体提出了一套诊断中间执行过程错误的方法与框架，建立了系统级和任务级两大类错误的结构化分类体系，并揭示了这些错误对整体性能的影响。 |
| [^112] | [Decodability Is Not Controllability: Local and Anisotropic Speaker Control in Continuous-Latent TTS](https://arxiv.org/abs/2610.29345) | 该研究揭示连续潜变量TTS中说话人特征的高可解码性并不等于因果可控性，有效的话者音色控制依赖于在参考音频早期位置（L0层）进行干预，而非探针准确率最高的深层。 |
| [^113] | [Reader--Writer Coupling in Transformer MLPs](https://arxiv.org/abs/2610.29340) | 本文提出将Transformer MLP中隐藏通道的读取器与写入器视为耦合对进行联合测量，利用MLP更新雅可比矩阵与自身转置的一致性导出带符号的配对坐标，并据此发现每个模块仅由一小部分强耦合通道构成核心。 |
| [^114] | [BrainCodes: A Lightweight EEG Foundation Model over Discrete Neural Codes](https://arxiv.org/abs/2610.29320) | BrainCodes是一个基于离散神经编码的轻量级EEG基础模型，通过量化神经分词器与时空编码器的自监督预训练，以紧凑架构在睡眠分期等多种神经生理监测任务上达到或超越现有最先进水平。 |
| [^115] | [DISRL: Dual-Scale Incremental Semi-Supervised Reinforcement Learning for Dynamic Community Tracking](https://arxiv.org/abs/2610.29315) | 提出了双尺度增量式半监督强化学习框架DISRL，通过动态生成种子特定指导并将短期适应与长期参数巩固解耦，解决了动态社区追踪中监督不确定性和误差级联累积两大难题。 |
| [^116] | [YingLan: A Multimodal Ocean Foundation Model for Unified Representation Learning from Heterogeneous Observations](https://arxiv.org/abs/2610.29313) | 提出了多模态海洋基础模型 YingLan，通过物理属性条件化动态编码、时空增量耦合（STIC）与海洋状态演化感知预训练，实现了对卫星影像、网格化海洋场和浮标观测等异构数据的统一可迁移表示学习。 |
| [^117] | [Interpretable Cutting Plane Selection for Mixed-Integer Linear Programming Using Genetic Programming](https://arxiv.org/abs/2610.29311) | 本文提出首个基于遗传规划的完全可解释割平面选择框架GPSCS，通过演化符号化评分函数将手工启发式策略显式泛化为可解释的自适应割平面选择策略，弥补了神经网络方法可解释性差且缺乏收敛保证的不足。 |
| [^118] | [Data-driven and physics-guided hindcasting of measured wave spectra](https://arxiv.org/abs/2610.29306) | 该研究提出一种数据驱动并结合物理约束的循环模型，利用浮标谱初始化并输入风、风区和深度信息，实现了对波罗的海实测波浪谱长达12小时的准确后报，其中物理松弛约束将波高RMSE降低5.01%至约2厘米。 |
| [^119] | [ExpEvolve: Structuring and Reusing Experience for Open-Ended Discovery](https://arxiv.org/abs/2610.29303) | 提出ExpEvolve框架，将思想层级的提取与控制注入进化式编码代理，通过发现能够解释程序间评分差异的高层级思想因素并复用经验，提升开放式科学发现的样本效率。 |
| [^120] | [Compressing Onboard Language Models for Space Robots under a Behavioral-Fidelity Requirement](https://arxiv.org/abs/2610.29302) | 本文提出QSelect方法，通过为每个线性模块从AWQ、GPTQ和AutoRound量化器中选择最优方案并以KL散度作为保真度验收标准，为太空机器人构建满足飞行硬件约束且行为接近全精度模型的静态低比特星载语言模型。 |
| [^121] | [Approximate Solomonoff Induction: On the Robustness of Learned Universal Predictors](https://arxiv.org/abs/2610.29289) | 本文研究了实际机器学习算法产生的近似最优预测器的鲁棒性问题，给出了近似对预测误差最坏情况影响的紧致界，从而论证了通过对数损失最小化学习到的通用预测器在近似条件下依然保持稳健。 |
| [^122] | [Flat-RLVR: Sharpness-Aware Reinforcement Learning with Verifiable Rewards for Reasoning Capacity in LLMs](https://arxiv.org/abs/2610.29281) | 提出Flat-RLVR方法，通过锐度感知最小化在RLVR中寻找平坦奖励区域，缓解分布锐化问题，在保持准确率的同时提升大语言模型的样本多样性和推理能力。 |
| [^123] | [Design and Tuning of Multi-Agent Systems Using Bayesian Evolutionary Optimization](https://arxiv.org/abs/2610.29276) | 提出了一种结合进化算子与图感知贝叶斯优化的框架，可自动发现并调优面向特定任务的多智能体系统拓扑结构，在代码生成和表格数据预处理基准上取得了最优性能。 |
| [^124] | [Estimating Performance for Safe Deployment of Machine Learning Models](https://arxiv.org/abs/2610.29262) | 提出了MOVAL模型评估工具，通过类别级置信度校准与多校准分数集成方法，利用无标签测试数据估计模型性能，从而实现机器学习模型的安全部署。 |
| [^125] | [Group Distributionally Robust Optimization over Decentralized Networks](https://arxiv.org/abs/2610.29261) | 该论文提出直接优化去中心化网络上的原始群组分布式鲁棒优化目标，并通过新颖的误差分解方法将其转化为去中心化在线凸优化问题，首次在半去中心化和完全去中心化设置下为工作节点的局部模型建立了理论保证。 |
| [^126] | [Extending FAIR Principles with Observability for Trustworthy Federated Learning](https://arxiv.org/abs/2610.29252) | 提出FAIR-O框架，将FAIR原则扩展出可观测性维度，使每个参与者在每轮联邦学习中生成最小化的签名信任证据（可由本地TinyLLM辅助解释），在不迁移原始数据的前提下建立三级信任模型以实现可信联邦学习。 |
| [^127] | [MA-UITD: Offline Multi-Agent Reward Inference for Characterizing Continuous Social Behavior](https://arxiv.org/abs/2610.29237) | 提出MA-UITD，一种离线多智能体奖励推断方法，无需额外环境交互即可从固定行为数据集的联合演示中推断显式的智能体特定奖励，用于刻画连续社会行为。 |
| [^128] | [Cache Locally, Recompute Selectively for Streaming Video Understanding](https://arxiv.org/abs/2610.29229) | 提出无需训练的即插即用方法FrameBlend，通过在查询时利用查询相关上下文对选定视频标记的KV缓存进行选择性重算，解决了流式视频理解中的上下文不匹配问题。 |
| [^129] | [Schr\"odinger-Bass Bridge Barycenter Models for Measure-Valued Data](https://arxiv.org/abs/2610.29227) | 本文利用薛定谔-巴斯桥的方向不对称性与对偶漂移-扩散控制，提出四种新型测度值数据重心模型，并从理论上证明了这些重心的存在性，且证明高斯输入下解码器形式存在高斯最优解，可将重心计算化为有限维优化。 |
| [^130] | [Delay-Guided Attention for Quality Prediction in Multistage Manufacturing](https://arxiv.org/abs/2610.29220) | 该论文提出了DAMF方法，通过显式估计阶段间延迟分布并利用其引导跨阶段信息传递，解决了多阶段制造中延迟随时间变化导致的质量预测难题。 |
| [^131] | [PerSPL: Personalized Semi-supervised Preference Learning](https://arxiv.org/abs/2610.29219) | 提出PerSPL框架，通过成对均值分数中心化来稳定跨提示的分数坐标系，从而迭代地将大量无标签响应转化为用户特定的训练信号，以低成本方式提升个性化奖励模型的学习效果。 |
| [^132] | [Preference Is Not Permission: Personalizing the Auto-Purchase Line in Agentic Commerce](https://arxiv.org/abs/2610.29216) | 该论文提出将学习个人化的“自动购买边界”作为代理式商务中的新任务，并发现十六个指令微调模型普遍“越权”——即使商品匹配度低、购买依据缺失也会自动下单，与人类委托者的授权意愿存在系统性错位，且这一边界因人而异、需要个性化建模。 |
| [^133] | [On the gradient conjecture at infinity: Finite spherical length for gradient trajectories of definable potentials](https://arxiv.org/abs/2610.29211) | 该论文证明了任意 o-极小结构中可定义 $C^1$-势函数的每条梯度轨迹（无论收敛还是发散到无穷远）的球面投影长度有限（即总角运动有限且方向收敛），从而对无穷远处梯度猜想和可定义 Thom 梯度猜想给出了肯定的回答。 |
| [^134] | [Reconstruction differences rival individual differences in the fly compass connectome and break trained circuit models](https://arxiv.org/abs/2610.29205) | 该研究发现果蝇头朝向（罗盘）连接组中由重建流程差异（如突触检测方法）造成的回路变化可与不同果蝇个体之间的生物学差异相媲美，说明基于连接组训练的回路模型依赖于其构建所用的特定重建版本。 |
| [^135] | [A Unified Evaluation Framework of Physical Adversarial Robustness in the Edge VLM Era](https://arxiv.org/abs/2610.29196) | 本文提出了边缘VLM时代物理对抗鲁棒性的统一评估框架，引入条件攻击成功率指标，在对抗补丁攻击下公平比较了视觉语言模型与传统目标检测器的鲁棒性差异。 |
| [^136] | [Truncated Out-of-Bag Probability Gain: A Noise-Robust Criterion for Random-Forest Wrapper Feature Selection](https://arxiv.org/abs/2610.29195) | 本文提出TOP-Gain准则，通过将真实类别袋外概率的下降截断为零来抑制袋外估计噪声，使随机森林封装式特征选择的特征排序更具可重复性，并在26个数据集的预注册基准测试中取得最佳综合表现，且计算效率比VSURF高21倍。 |
| [^137] | [Curvature-Scaled Sharpness-Aware Minimization: Convergence Beyond Smoothness](https://arxiv.org/abs/2610.29192) | 该论文利用相对梯度失真揭示了SAM在广义平滑条件下失真可能无界、SAM会循环而USAM会发散的现象，建立了梯度扰动统一失真控制的必要条件，并由此提出曲率缩放的锐度感知最小化方法，实现了超越平滑性假设的收敛性保证。 |
| [^138] | [A Decade of Learning-Based Control: From Theoretic Foundations to Practical Deployment](https://arxiv.org/abs/2610.29187) | 本文作为教程论文，受IEEE CDC十年“基于学习的控制”专题启发，系统回顾了机器学习与控制交叉领域的关键进展，涵盖稳定性与安全性保证理论、动力学模型学习方法，以及强化学习、模仿学习和贝叶斯优化等数据驱动策略合成方法及其应用与挑战。 |
| [^139] | [ROCOCO: Robust Outline Completion of Occluded Curvilinear Objects](https://arxiv.org/abs/2610.29177) | 该论文提出ROCOCO，一种基于流匹配的生成模型，结合UNet遮挡掩码预测，无需先验遮挡信息即可鲁棒地补全被环境障碍物遮挡的曲线目标轮廓，性能显著超越现成修复基线。 |
| [^140] | [Conditional Flow Models Convert Epistemic Uncertainty into Apparent Aleatoric Variance](https://arxiv.org/abs/2610.29172) | 该论文发现条件流模型生成的样本方差会系统性地大于真实条件分布的方差，这种额外的方差本质上是模型将拟合真实条件分布时的认知不确定性转化为生成样本中表观的偶然方差。 |
| [^141] | [TeachBack: Learning to Teach via Bidirectional Preference Optimization for On-Policy Self-Distillation](https://arxiv.org/abs/2610.29167) | TeachBack提出了一种双向偏好学习框架，使模型不仅从掌握特权信息的教师处学习，还反过来优化教师的教学方式，从而生成更适合学生学习的推理轨迹。 |
| [^142] | [From Augmentation to Synthetic Evidence: Evaluating Generated Chest Radiographs at Dataset Intake](https://arxiv.org/abs/2610.29152) | 该研究提出在数据集入口处对生成的胸部X光片进行质量检查、来源分类和记录验证的三重评估，发现基于质量的选择虽能显著提高质量通过率，但无法使生成图像同时通过质量与来源双重检查。 |
| [^143] | [SAFTER: Safety-Aware Filtering Through Explicit Recoverability for Robust Reinforcement Learning under Distribution Shift](https://arxiv.org/abs/2610.29149) | SAFTER通过将可微分的闭式备份CBF滤波器嵌入SAC训练，把当前安全性、备份安全性和可恢复性统一为单一可微分证书，在分布偏移下的四足导航任务中显著提升了成功率并大幅降低样本消耗。 |
| [^144] | [Mitigating Reward Hacking in Realistic Scenarios Through Representation Engineering](https://arxiv.org/abs/2610.29129) | 本研究提出一种基于表征工程的动态干预方法，通过多种引导向量以低成本显著减少现实场景中的奖励黑客行为，且不损害模型能力，并能跨领域泛化。 |
| [^145] | [Recursive Self-Improvement on a Budget: Scoring Code Edits Before They Run](https://arxiv.org/abs/2610.29121) | 该论文提出一种预算受限的递归自我改进方法：利用由少量实例评估拟合并冻结的线性读出器，在代码编辑实际运行之前对其进行评分，使自改写的编码智能体能够在有限评估预算下高效选择最有前景的编辑。 |
| [^146] | [MHeatmap: Two-Walk Laplacian Reordering and Mosaic Layout for Rectangular Matrix Visualization](https://arxiv.org/abs/2610.29114) | 该论文提出MHeatmap方法，通过在加权二部图中引入长度为2的游走相似性连接并基于Fiedler向量排序实现行列协调重排，同时采用行列尺寸与对应总和成比例的马赛克布局，解决了矩形矩阵热力图中相关对象分散、单元格大小掩盖数据总量差异的可读性问题。 |
| [^147] | [SU(2)-Equivariant Spinor Message Passing for Molecular Magnetism](https://arxiv.org/abs/2610.29111) | 本文提出eSpinor，一个通过传播旋量特征并结合规范协变消息传递与规范不变读出的SU(2)-等变图神经网络，显著提升了分子磁性性质（如零场分裂和电子自旋关联）的预测精度。 |
| [^148] | [DriftNeRF: Cross-Ray Density Drifting for Sparse-View Neural Radiance Fields](https://arxiv.org/abs/2610.29106) | 该论文提出DriftNeRF，一种即插即用的正则化器，通过带掩码的指数核从其他射线聚合邻近点密度，解决了稀疏视角下NeRF目标函数因缺乏跨射线密度一致性约束而导致的密度矛盾问题。 |
| [^149] | [Dual-Channel Contrastive Encoding of Sperm Whale Coda](https://arxiv.org/abs/2610.29101) | 本文提出双通道对比编码器（DCCE），通过分别编码并融合抹香鲸尾音的节奏与频谱信息，在社会单元、尾音类型和个体识别三个任务上取得了优异性能。 |
| [^150] | [Causal world models under physical parameter shifts](https://arxiv.org/abs/2610.29088) | 该论文将因果建模的两个关键组件——由成对干预导出的依赖掩码和显式的物理参数化机制——融入世界模型，使机器人在连杆质量等物理参数超出训练分布时仍能保持接近预言水平的准确预测。 |
| [^151] | [Internal Guidance as a Sampling Method for Tilted Diffusion Models](https://arxiv.org/abs/2610.29086) | 本文提出“内部引导”方法，通过改变基础去噪器的求值输入点而非修改其输出来采样倾斜密度，并利用随机对角重缩放改进训练算法，解决了所需求值点超出训练分布的问题。 |
| [^152] | [Value-Rectified Distillation for Flow-based Offline Reinforcement Learning](https://arxiv.org/abs/2610.29075) | 提出价值校正蒸馏（VRD）框架，将离线策略蒸馏重新表述为价值引导的生成式轨迹对齐问题，通过动态耦合机制校正流动力学以解耦多模态行为并分离高价值动作，从而解决离线强化学习中的分布外模态平均问题。 |
| [^153] | [The Price of Choice in Selective Prediction](https://arxiv.org/abs/2610.29067) | 本文刻画了选择性预测中审计“存在满足风险-覆盖契约的规则”这一断言时的代价：交并测试（IUT）所需的极小极大样本量随允许规则数K线性增长，且该速率在忽略对数因子的意义下是最优的。 |
| [^154] | [Distilling Multi-Teacher Scoring Principles for Unsupervised Time Series Anomaly Detection](https://arxiv.org/abs/2610.29063) | AnoMix提出了一种无监督多教师蒸馏框架，利用CoRank多教师排序蒸馏目标，将多个异构异常检测器的互补评分原则蒸馏进单个轻量级学生检测器，无需任何异常标签即在七个基准数据集上实现有效整合。 |
| [^155] | [OccJEPA: Occlusion as Masking for Amodal Video Segmentation](https://arxiv.org/abs/2610.29061) | OccJEPA 将视频中的遮挡视为掩码任务，利用预训练的 V-JEPA 2.1 预测器填补被遮挡区域并解码出完整掩码，实现了确定性的完整视频分割，比基于扩散模型的最先进方法快 33.5 倍且精度更高。 |
| [^156] | [Dynamics of Hidden Learning Structure](https://arxiv.org/abs/2610.29054) | 本文提出纤维调节理论与纤维获取理论，刻画了隐藏学习结构中“保护性修复”与“可训练结构获取”这两种操作及其相互约束关系，并通过有限监督学习构造证明二者联合可产生任一操作单独无法实现的有利学习响应。 |
| [^157] | [Our Own Jev: Typed Decisions in One Forward Pass, from an In-House Judge to Open LoRA Models](https://arxiv.org/abs/2610.29052) | 该论文提出用单次前向传播的“类型化决策”接口（直接读取logits或轻量指针头）替代生成式LLM裁判，其自研System 1裁判仅用一块GPU约3分钟、约0.06美元即可判定1,000个智能体编码会话，以约1%的成本取代每轮约1,000美元且标签一致性仅26.9%的前沿LLM裁判。 |
| [^158] | [ART-MPG: Autoregressive R/T Manifold Path Guiding](https://arxiv.org/abs/2610.29049) | 提出ART-MPG，一种通过自回归采样镜面链的长度与反射/透射拓扑字符串、并以拓扑条件化vMF混合分布采样方向，同时将重试采样同时条件化于链长度和R/T字符串的紧凑神经模型，在保持无偏性的前提下降低了流形路径引导的预期重试次数和估计器方差。 |
| [^159] | [Gate Initialisation versus Recurrent Dropout and Zoneout in Learning Long Dependencies: A Controlled Study of LSTM and GRU Networks on Synthetic Tasks](https://arxiv.org/abs/2610.29034) | 该研究通过720次运行的受控析因实验，首次系统交叉比较了门控初始化方式与循环Dropout/Zoneout放置位置对LSTM和GRU学习长程依赖的影响，发现chrono初始化在困难序列长度下使全部360次运行均获得成功。 |
| [^160] | [Better Policies, Worse Verifier Audits](https://arxiv.org/abs/2610.29029) | 本文揭示了在策略改进闭环中，随着策略掌握部分子目标，基于聚合分数的组件验证器审计会出现“同策略审计崩塌”现象，导致审计反而偏好更差的验证器，并给出了精确的识别集、完全擦除阈值与崩塌临界点的理论刻画。 |
| [^161] | [Dissimilarity-Based Markov Decision Processes for Stabilizing State Distributions](https://arxiv.org/abs/2610.29024) | 本文提出基于相异度的马尔可夫决策过程（DMDP）理论框架，其中策略作用于状态分布并通过最小化与目标分布的相异度进行优化，建立了最优策略使目标分布渐近稳定的充分条件，并通过分布的有限维表示将稳定性保证扩展到底层状态分布。 |
| [^162] | [Target Regeneration Shrinks the Gains of Draft-Training Objectives in Speculative Decoding](https://arxiv.org/abs/2610.29013) | 研究发现，投机解码中新的草稿训练目标（如令牌级蒸馏和验证门控蒸馏）在外部回复数据上的收益，在经过目标模型贪婪重生成后大幅缩水，且该现象在多种训练与解码设置下均稳定存在。 |
| [^163] | [Local and Sparse Kernel Operator Learning for Incompressible Flows: An Operator Lagrange Formulation](https://arxiv.org/abs/2610.29009) | 该论文提出一种局部稀疏核算子学习方法，用GPU并行求解的局部无散度插值替代全局重构，用紧支撑核实现稀疏算子回归，并通过算子拉格朗日权重高效结合两者，从而高效学习不可压缩流的解算子。 |
| [^164] | [A Jump-Modified Wasserstein Gradient Flow for Maximum Mean Discrepancy with Energy Kernels](https://arxiv.org/abs/2610.29007) | 本文提出一种带跳跃项的修正Wasserstein梯度流来最小化能量核生成的最大均值差异（MMD），证明了该流的全局适定性、仅凭有限二阶矩即可收敛到目标分布，并构造了能量单调不增的全局无碰撞粒子过程。 |
| [^165] | [FlashNext-RL: A System Optimization Study for Long-Horizon Agentic RL of Qwen3.8-Flash-Next](https://arxiv.org/abs/2610.28998) | 本文提出 FlashNext-RL 框架，通过优化主机卸载、梯度累积以及混合专家、稀疏注意力和门控残差流等专用内核，实现了在单个 8xB200 节点上对 Qwen3.8-Flash-Next 完整 48 层模型的 BF16 长程智能体强化学习训练，显著提升了长序列训练吞吐量。 |
| [^166] | [Rethinking Time Series Tokenization from a Frequency Perspective](https://arxiv.org/abs/2610.28997) | 提出了一种几乎无失真的基于频率的时间序列分词器TimeFT，通过频域划分、频率移位和奈奎斯特采样获取token，无需参数且可作为现有分词器的即插即用替代品，有效避免传统分词带来的频谱失真问题。 |
| [^167] | [The Optimal Sampler for Diverse TSP Solutions: Characterization, Limits, and Learning](https://arxiv.org/abs/2610.28994) | 该论文针对多样化旅行商问题（D-TSP）提出将问题分解为采样与选择两阶段的混合框架，证明了选择阶段近似难度下界，并从理论和实证两方面为最优采样分布提供了保证（确保采样池包含最优解集或获得至少k个相互分散的解）。 |
| [^168] | [Doob-Based Generative Policy Optimization](https://arxiv.org/abs/2610.28990) | 提出 DoobPO 框架，利用 Doob 的 h 变换通过比率模型与条件漂移匹配，实现无需似然计算的扩散与流生成式策略强化学习微调，并统一兼容 PPO、GRPO 等策略改进规则。 |
| [^169] | [AgentPolicyForge: From Executable Environments to Policy Benchmarks and Adaptive Guards](https://arxiv.org/abs/2610.28989) | 该论文提出AgentPolicyForge框架，通过可执行环境实际运行工具调用并记录结果，构建了涵盖十个策略族的PolicyGym基准数据集，并利用其中真实发生的合规与违规案例迭代训练能够检查参数、资源和上下文（而非仅看工具身份）的自适应策略防护器。 |
| [^170] | [Attention-based Classification for Indirect Prompt Injection](https://arxiv.org/abs/2610.28988) | 本文首次在真实的智能体场景（AgentDojo基准）下评估了Attention Tracker检测方法，发现其声称的数据集无关性并不成立——长结构化工具输出会颠覆其注意力假设并使检测效果接近随机，进而提出了基于注意力分类的改进检测方案。 |
| [^171] | [From Common Information to Canonical Correlation Analysis and Beyond: Bridging Information Theory, Signal Processing, and Machine Learning](https://arxiv.org/abs/2610.28978) | 本教程首次将统计学中的典型相关分析、信息论中的两种公共信息概念以及工程领域的多视图子空间方法置于统一框架下，为无监督恢复多观测中公共分量的跨学科问题建立了桥梁，并厘清了不同理论之间的精确等价与类比关系。 |
| [^172] | [Is In-Context Reinforcement Learning A Posterior Sampler? An Empirical Study](https://arxiv.org/abs/2610.28977) | 本文通过建立贝叶斯后验采样器必须满足的两个必要条件——充分统计量不变性和鞅性质，并据此构造统计假设检验，实证考察了上下文强化学习（特别是DPT及其变体）在实践中是否真正表现为贝叶斯后验采样器。 |
| [^173] | [SSMCGM-Stream: A Streaming State-Space Model for Personalized Diabetes Management](https://arxiv.org/abs/2610.28976) | 提出了流式状态空间模型SSMCGM-Stream，通过实时更新个性化隐藏状态实现高效、可解释的血糖预测，并结合SSMCGM-Agent提供设备端自然语言糖尿病决策支持，性能与神经基线相当而计算成本大幅降低。 |
| [^174] | [How Does Orthogonalization Adapt to the Neural Network Hessian Structure? A Gradient Self Outer-Product Analysis at Initialization](https://arxiv.org/abs/2610.28975) | 本文通过在初始化阶段对梯度自外积（$\mathbf{V}\mathbf{V}^\top$）的对角占优性进行分析，揭示了Muon正交化预条件器之所以有效，是因为它能匹配神经网络逐层Hessian的行块对角占优结构。 |
| [^175] | [From Weather to Enhanced Weathering: Weather Foundation Models for Carbon Dioxide Removal at Operational Scale](https://arxiv.org/abs/2610.28973) | 该研究首次整体评估了天气基础模型在增强岩石风化碳去除量化任务上的迁移能力，其基于Aurora 1.5构建的TerraNova-ERW模型能够在未见过的田块内正确排序测量值，但指出实现零样本预测新田块去除量仍需更大规模的ERW运营数据。 |
| [^176] | [Unit-Aware Differentially Private Quantum Machine Learning for Clinical and Wearable Health Data](https://arxiv.org/abs/2610.28968) | 该论文提出一种“单元感知”的差分隐私量子机器学习方法，将每条临床记录或每位可穿戴设备用户视为隐私单元，并在输入维度、隐私预算与模型容量匹配的条件下与经典模型对比，以检验可训练参数更少的量子模型能否在保护隐私的同时为临床与健康数据保留更多预测效用。 |
| [^177] | [Can Quantum Layers offer Performance and Security Advantages in Vision-Language Models?](https://arxiv.org/abs/2610.28964) | 本研究以LLaVA-OneVision为骨干，通过在多个位置插入或替换4/8量子比特的模拟量子线路，系统比较了量子-经典混合视觉-语言模型与纯经典模型在性能、安全性及计算成本上的差异，以探究量子层在大规模多模态模型中的实际价值。 |
| [^178] | [Transformers as Generic Deep Neural Networks: Exact Compilation, the Cost of Removing Activation-Activation Products, and Multi-Agent Attention Clustering](https://arxiv.org/abs/2610.28958) | 本文证明每个Transformer块都能被精确编译成仅由仿射映射、路由和少数基本非线性运算构成的通用深度神经网络，并量化了移除激活-激活乘积运算所需付出的至少N²次额外标量非线性求值的代价，表明仅保留一种逐元素形式的该运算即可消除此开销。 |
| [^179] | [HyperDrone: Simulation-to-Reality Transfer at Millions of Frames per Second](https://arxiv.org/abs/2610.28950) | HyperDrone是一个超高速四旋翼无人机模拟器，通过结合优化的多旋翼动力学与新型光线追踪引擎HyperLux，比现有最先进模拟器快20倍以上，并能在消费级GPU上以每秒数百万帧的速度渲染高保真逼真图像，从而实现高效的仿真到现实迁移。 |
| [^180] | [Online Open-World Wafer Defect Discovery using Hyperdimensional Computing](https://arxiv.org/abs/2610.28949) | 提出了基于超维计算的在线开放世界晶圆缺陷发现框架WaferHDC，无需标签或重训即可动态发现并分组新型晶圆缺陷模式，在WM-81数据集上调整兰德指数超越最强在线基线13.3个百分点。 |
| [^181] | [Is 2:4 Sparsity Still Worth It? Semi-Structured Sparsity versus Quantization in LLM Inference](https://arxiv.org/abs/2610.28947) | 本研究系统比较了大语言模型推理中2:4半结构化稀疏与量化的取舍，发现只要硬件还有更低精度可用，降精度量化总是优于叠加2:4稀疏（后者既更不准确也更慢），只有当量化已达到硬件支持的最低精度时，2:4稀疏才能为线性层带来1.3—1.7倍的加速。 |
| [^182] | [Multi-Objective Protein Optimization via Modular Diffusion Steering](https://arxiv.org/abs/2610.28943) | 本文提出MODS框架，通过推理时扩散引导结合采样的切比雪夫标量化权重和代理集成预测，实现模块化的多目标蛋白质优化，可灵活搭配不同扩散模型以发现多种性质权衡下的高性能蛋白质序列。 |
| [^183] | [Auditing Conformal Prediction for Language Models Under Inference-Time Model Shift](https://arxiv.org/abs/2610.28937) | 该论文首次提出并形式化了“推理时模型偏移”现象——即部署与校准阶段推理配置（如量化模式和推理温度）的差异会引起保形分数分布变化并破坏分割保形可交换性，并通过8个语言模型、4个基准和12种推理配置的大规模实验系统评估了其对语言模型保形预测可靠性的影响。 |
| [^184] | [The Data Differentiator Framework (DDF): Quantifying Marginal Data Value with Information-Theoretic and Game-Theoretic Measures in an Agentic Evaluation Pipeline (SDSAE)](https://arxiv.org/abs/2610.28933) | 提出数据差异化框架（DDF），通过互信息冗余惩罚、成本归一化的边际效用评分和治理输入，量化候选数据资产相对企业已有数据的增量价值，并由智能体流水线SDSAE自动高效执行评估。 |
| [^185] | [One Policy, Many Styles: Where to Aggregate Preferences for Steerable Humanoid Control?](https://arxiv.org/abs/2610.28925) | 该论文研究了偏好条件强化学习中偏好聚合层级（奖励、优势、损失、PCGrad）对人形机器人可引导控制的影响，发现没有单一聚合层级在所有环境下最优：早期聚合扩大行为响应范围但增加非预期改变，晚期聚合提升控制精度但有所取舍。 |
| [^186] | [Query Feedback and Teaching for Stable and Certifiable Language Generation](https://arxiv.org/abs/2610.28918) | 本文在极限语言生成中引入了更强的“可认证性”概念，并借鉴教学集理论建立了正确性、稳定性与可认证性三级教学维度层级，同时进一步研究了成员查询模型下的稳定性与可认证性。 |
| [^187] | [A Model-Free Continuous-Time RL Algorithm from Discrete-Time Off-Policy Data with End-to-End Guarantees](https://arxiv.org/abs/2610.28915) | 该论文提出了一种无模型的PhiBE策略迭代（PhiBE-PI）框架，仅使用单个状态-动作评论家、直接从离散时间离策略数据中学习连续时间强化学习策略，同时保留连续时间的动作排序尺度并具有端到端的理论保证。 |
| [^188] | [Backtracking Direct, Indirect and Total Effects](https://arxiv.org/abs/2610.28910) | 本文提出基于回溯反事实的因果效应新定义，无需干预处理变量或中介变量，而是通过回溯条件分布构造满足处理-中介约束的替代外生状态，从而得到回溯直接效应、间接效应和总效应。 |
| [^189] | [JaxMARL-BC: Multi-Agent Reinforcement Learning for Macroeconomic Modelling, in Seconds](https://arxiv.org/abs/2610.28907) | 本文提出基于JAX加速的模拟器JaxMARL-BC，将MARL-BC经济体的训练速度提升75至724倍、缩短至数秒内，并可扩展至20万个独立学习的家庭，解决了MARL在宏观经济学中应用的主要瓶颈。 |
| [^190] | [Tropical Cyclone Attributes in Sparse Autoencoder Features of an AI Weather Model](https://arxiv.org/abs/2610.28904) | 该研究以GraphCast中的热带气旋为案例，通过稀疏自编码器（SAE）探测AI气象模型的内部表征，发现气旋的位置、强度和气压结构信息以跨独立SAE和层间可重复的方式分布于多个特征中，且深层特征更能预测气旋六小时后的状态。 |
| [^191] | [Minimax-Optimal Predictive Flow Matching under Distribution Drift](https://arxiv.org/abs/2610.28898) | 提出 DyProd-FM 算法，在无需知晓漂移预算的情况下，对固定非凸神经网络类的预测流匹配问题实现了与下界匹配的极小极大最优动态遗憾。 |
| [^192] | [Is Refusal Direction Universal across Low-Resourced Languages?](https://arxiv.org/abs/2610.28897) | 该研究发现拒绝方向的跨语言普适性是分级的而非二元的：来自安全对齐语言的拒绝方向可广泛迁移至低资源语言，而来自强失对齐语言的拒绝方向则无法可靠地调节任何目标语言的拒绝行为。 |
| [^193] | [CSD: Common-Support Distillation for Cross-Tokenizer Knowledge Transfer](https://arxiv.org/abs/2610.28895) | 本文提出CSD框架，通过在每个对齐输入分块处构建教师与学生模型间的公共支撑集，并以单一目标函数匹配分布，解决了词表和分词差异导致的跨分词器logits知识蒸馏对齐难题。 |
| [^194] | [Combating Catastrophic Forgetting in Continual Domain Adaptation via Knowledge Recasting](https://arxiv.org/abs/2610.28892) | 该论文提出技能模式传输框架，通过将新领域知识转换到模型已有的技能空间中、使陌生知识借助熟悉的过程被学习，从而在持续领域自适应中有效对抗灾难性遗忘。 |
| [^195] | [RL-IF: Tracking Non-local Training Data Influence for Online Reinforcement Learning](https://arxiv.org/abs/2610.28883) | 该论文提出轨迹级留一指标 RL-LOO 及其一阶估计器 RL-IF，首次将在线强化学习的数据归因从单轮局部影响扩展到跨多轮的非局部影响，并同时刻画优化效应与策略诱导的采样效应。 |
| [^196] | [Actually, All Learnable Distribution Classes Are Privately Learnable](https://arxiv.org/abs/2610.28878) | 本文证明了所有在总变差距离下可学习的分布类在（ε, δ）-差分隐私下同样可学习，从而证明了 Ashtiani 的猜想。 |
| [^197] | [LEAN: Loss-Engineered Modulation of the Model Weight-Sensitivity Distribution for Efficient, Error-Tolerant Neural Networks](https://arxiv.org/abs/2610.28872) | LEAN提出了一种新颖的复合正则化方法（包含范围项和尾部项），能够系统性地调制训练后模型的权重敏感度分布，仅需十个epoch的微调即可显著提升神经网络对位翻转错误的容忍度并降低存储开销。 |
| [^198] | [How Much Gradient Does Truncated Backprop Lose in Looped Transformers?](https://arxiv.org/abs/2610.28871) | 该论文首次在九个循环语言模型上精确量化了截断反向传播（仅反传最后k次循环）所丢弃的权重梯度比例，发现该损失在已训练模型上于靠近输出处即已确定、几乎不随循环总数变化，并在微调过程中逐渐减小。 |
| [^199] | [A General Topological Symmetry Index for Critical Orbits In Neural Network Loss Landscapes](https://arxiv.org/abs/2610.28864) | 本文基于等变度理论建立了一个通用的拓扑对称性指标，用于系统刻画神经网络损失景观中由架构对称性产生的临界轨道，能够检测临界点的迷向性并量化其净拓扑贡献，且适用于任意网络架构包括退化情形。 |
| [^200] | [DualHeads: Reward-tilted few-step diffusion made simple](https://arxiv.org/abs/2610.28858) | 提出 DualHeads 方法，通过在冻结的少步扩散模型上训练奖励头和参考头两个轻量级二元分类器，仅利用外部奖励评估和参考样本即可实现少步扩散模型的偏好对齐，且该目标等价于 KL 正则化的奖励最大化。 |
| [^201] | [Self-Supervised Tuning of Low-Fidelity Optimization Models via High-Fidelity Feedback](https://arxiv.org/abs/2610.28851) | 该论文提出一种自监督框架，将低保真优化模型嵌入为可微分优化层，借助高保真校正问题对偶变量提供的梯度进行端到端训练，无需昂贵的高保真解标签即可自动修正低保真模型在最优潮流、最优气流和池化等问题中的系统性误差。 |
| [^202] | [When Do Gaussian Priors Fail? A Latent Geometry Investigation of RL Fine-Tuning](https://arxiv.org/abs/2610.28849) | 本研究揭示了高斯先验VAE在强化学习微调下的结构性缺陷——KL项主动惩罚模态分离导致潜空间模态重叠并引发模型崩溃——并指出崩溃阈值相差71倍的根源可归结为单一几何量 d_min/(2σ₀)。 |
| [^203] | [KIRA: A Full-Stack Reading Screening System with Dual-Pathway Analysis for Children](https://arxiv.org/abs/2610.28840) | KIRA是一个面向儿童的双通道阅读筛查系统，通过将预训练语音识别模型适配儿童语音（词错误率从13.88%降至8.84%，音素错误率从41.66%降至23.00%），实现了单词级流畅度评分与音素级错误检测的结合，并通过教师界面支持针对性阅读干预。 |
| [^204] | [Sketch2Sim: an end-to-end modular framework for hand-drawn circuit analysis](https://arxiv.org/abs/2610.28836) | 本文提出Sketch2Sim端到端模块化框架，通过多阶段流水线（含双YOLO检测器）将手绘电路图照片自动转换为SPICE兼容网表并计算电路的节点电压与电流，同时构建了1000个标注的手绘RLC电路数据集，旨在改善电路教学体验并加速工程设计评审。 |
| [^205] | [Brittle by Design? Multi-Scenario Reinforcement Learning for Robust Mission Coordination in Systems of Systems](https://arxiv.org/abs/2610.28818) | 该论文提出一种将高保真数字任务模型与基于近端策略优化的强化学习相结合的智能任务协同方法，并通过多场景航空灭火任务验证了其在不确定动态环境下实现体系鲁棒自适应任务分配的有效性。 |
| [^206] | [A Mechanistic Study of Automated Failure Attribution in LLM-based Agentic Systems](https://arxiv.org/abs/2610.28811) | 本文对基于大语言模型的智能体系统中的自动化失败归因进行了机制层面的研究，分析了为何直接提示方法效果不佳，并探索了提升失败归因性能的改进策略。 |
| [^207] | [Combining Long-Term Memory and LLMs Comes at a High Cost: From Prompt Injection to Persistent Compromise](https://arxiv.org/abs/2610.28810) | 本文揭示了现代对话式AI平台的持久记忆功能使提示注入攻击能够跨会话传播（记忆注入），并提出通用威胁模型证明所有NIST攻击目标均可实现，凸显了这一新型安全威胁的严重性与紧迫性。 |
| [^208] | [Towards Realistic Models of Change: A Causal Perspective on Decision-Making and Dynamical Systems](https://arxiv.org/abs/2610.28803) | 该论文从因果视角研究决策与动力系统中的现实变革模型，其第一部分针对算法追索权问题提出了有别于既有框架的新视角，旨在识别个体可采取哪些行动来推翻不利的决策。 |
| [^209] | [Depth Separation for Neural Networks under Gaussian Measure](https://arxiv.org/abs/2610.28800) | 该论文在高斯测度下建立了神经网络的深度分离结果：存在一个径向函数可被多项式规模的3层ReLU网络以小L²误差逼近，但2层网络要逼近它需要指数级规模，其核心工具是关于径向函数Hermite高频展开的不可逼近性定理，且全部证明均经Lean 4形式化验证。 |
| [^210] | [An Explicit Feedback Contract for Iterative ML-Agent Harnesses](https://arxiv.org/abs/2610.28798) | 该论文提出了一种显式反馈契约，通过基于配对标准误的噪声阈值来控制验证分数的发布，防止机器学习智能体对验证集进行自适应选择，从而避免在无真实增益的任务上虚报性能提升。 |
| [^211] | [CALLIOPE: Private Insights into AI Use, Without Embeddings](https://arxiv.org/abs/2610.28797) | CALLIOPE是一个支持差分隐私的自顶向下主题层次发现框架，它不依赖向量嵌入聚类，而是在整个流程中直接利用LLM能力，在高隐私保护程度下比基于嵌入的方法表现更好。 |
| [^212] | [ArcFlow: Training Flow Policies for Multiple Solver Budgets](https://arxiv.org/abs/2610.28788) | 提出 ArcFlow 部署感知算法，通过动作对齐监督、预算循环暴露和最差预算检查点选择，在多个求解器预算范围内训练单一流匹配策略，避免单一预算强化学习导致的低成本部署性能退化。 |
| [^213] | [Is Anyone Enforcing the Cartel? What Reinforcement Learning Teaches LLM Agents in Markets](https://arxiv.org/abs/2610.28787) | 该研究提出一种在反事实市场副本中强制智能体作弊以检验合谋执行机制的审计方法，发现RL训练的LLM定价智能体只会学会训练中易尝试且被奖励的行为（被威慑或跟随降价），而无法自发掌握执行卡特尔所必需的高代价价格战惩罚。 |
| [^214] | [Measure First, Then Allocate: Calibrated Rollout Budgets for MCP Agents in Reinforcement Learning](https://arxiv.org/abs/2610.28785) | 提出BRACE方法，利用奖励分歧测量与历史任务证据相结合的相关奖励模型，在分配生成预算前前瞻性预测混合奖励组，显著提升强化学习中MCP工具使用智能体的rollout预算分配效率。 |
| [^215] | [Extreme Low-State Quantized Transformer for Electricity-Load Forecasting](https://arxiv.org/abs/2610.28784) | 该论文研究了极端低状态（3至16状态）权重量化对Transformer电力负荷预测模型的影响，发现量化感知微调（QAT-FT）可恢复约90%以上的量化精度损失，显著优于直接训练后量化（PTQ）。 |
| [^216] | [Second Thoughts: Exploring and Planning through Comparative Imagination in World Models](https://arxiv.org/abs/2610.28779) | 提出SDI方法，利用基于行动者的共享参考标准，同时为世界模型中的探索信号和规划决策提供统一依据，使智能体能够自主判断执行计划动作还是遵循已学习的策略，且无需额外的分歧集成模型。 |
| [^217] | [Building a Nested LADDER for LLM Unstructured Post-Training Pruning](https://arxiv.org/abs/2610.28773) | LADDER通过逐层增量式评估稀疏度分配，在每步优先剪除单位参数移除造成端到端退化最小的层，从而生成非均匀稀疏分布和多预算下精确嵌套的剪枝掩码，显著提升大语言模型训练后剪枝在高稀疏度下的鲁棒性。 |
| [^218] | [Motion Equivariant Convolutional LSTM: When Motion Becomes a Gauge Symmetry](https://arxiv.org/abs/2610.28771) | 提出运动等变卷积LSTM（MELSTM），将运动视为时间上局域化的规范对称性，通过估计时变速度而非枚举速度，使网络记忆能够随输入的运动而等变地移动。 |
| [^219] | [ReAPO: Retry-Aware Policy Optimization for Efficient Exploration in LLM Reasoning](https://arxiv.org/abs/2610.28770) | 提出基于随机最短路径框架的信用分配方法 ReAPO，通过为失败轨迹分配额外生成预算来实现重试感知的策略优化，从而提升大语言模型推理中探索的效率。 |
| [^220] | [SkillEval: Learning Interpretable Ability Profiles of LLMs via Cognitive Diagnosis Models](https://arxiv.org/abs/2610.28767) | SkillEval 结合大语言模型的元认知能力与心理测量学中的认知诊断模型，从基准题目中自动发现可解释的技能，并为大语言模型生成跨模型的技能级能力画像，相比传统平均分评估更可靠、更校准且信息更丰富。 |
| [^221] | [Automating ML for Science: Can Frontier Agents Climb Scientific Hills in the Wild?](https://arxiv.org/abs/2610.28763) | 该论文提出了长时程测试平台 Agent4Sci（涵盖 11 个领域的 20 个科学机器学习基准），发现前沿智能体能在所有基准上超越已发表的最佳模型（Claude Opus 平均相对改进 10.4%），但这些进展几乎完全来自机器学习工程技巧而非研究级的科学洞察。 |
| [^222] | [GPU-Accelerated Construction of the Gabriel Graph: Parallelization Strategies and Applications to Large-Margin Classification and Graph-Based Clustering](https://arxiv.org/abs/2610.28759) | 本文提出一组基于CUDA的GPU并行化策略（包括从线程索引到顶点对的隐式闭式映射以及基于位掩码的原子压缩边列表构建），突破了Gabriel图朴素构建O(N³)的复杂度瓶颈，使其能够扩展到大规模数据集，并应用于大间隔分类和基于图的密度感知聚类。 |
| [^223] | [Understanding the Inner Workings of a State-Space Foundation Model for Limit Order Books](https://arxiv.org/abs/2610.28753) | 本研究揭示了8.2M参数的Mamba-3状态空间基础模型在限价订单簿消息流上自回归训练后，内部学会了市场及其动态的通用表示，其中订单流不平衡这一短期价格波动核心驱动因素可被以R²≈0.59的高精度解码，且该表示可泛化至未见过的股票。 |
| [^224] | [A Reverse-Gold Theorem: Positive-Data Identification by Substrate-Bounded Contrastive Learners](https://arxiv.org/abs/2610.28747) | 该论文证明了一个“逆向戈德定理”：借助先天递归对比算子自我生成结构受限的负例扰动，并以InfoNCE式对比学习训练的受限评分器，在忠实性条件下可仅凭正数据实现高概率的PAC条件语言识别，且样本复杂度仅为多项式级。 |
| [^225] | [Offline Apprenticeship Learning for Human-Centric Environments](https://arxiv.org/abs/2610.28716) | 本论文提出一系列离线学徒学习框架（包括 EM-EDM、THEMES、HALIDE），通过解决奖励建模与演示质量两大挑战，在不与环境交互的情况下从专家演示中学习适用于智能辅导系统和临床医疗等以人为本环境的自适应策略。 |
| [^226] | [Big Claims, Small Margins: Do Reported GraphLLM Gains Survive Simple Baselines?](https://arxiv.org/abs/2610.28714) | 该研究发现简单基线（Correct & Smooth和MLP）在文本属性图节点分类上与最先进GraphLLM方法的差距中位数仅约2-3个百分点，并揭示了部分已发表研究存在测试集选择超参数、测试标签泄露等评估风险。 |
| [^227] | [STAR-XPC: Set Transformer for Estimating X-ray Transmission in Polymer Composites](https://arxiv.org/abs/2610.28709) | 本文提出基于集合Transformer的STAR-XPC模型，可在1-3毫秒内快速准确估算聚合物复合材料的X射线透射比，替代耗时超过5小时的蒙特卡罗模拟。 |
| [^228] | [Borrowing Strength Across Stores: Transfer Learning for Choice-Based Revenue Management](https://arxiv.org/abs/2610.28708) | 该论文提出了一种面向MNL需求模型的收缩估计量，通过迁移学习在门店本地需求估计与跨门店信息共享之间取得平衡，从而在本地数据有限的情况下提升零售收益管理中品类与定价决策的可靠性。 |
| [^229] | [SimHack: Predicting the onset of detectable reward hacking before RL training](https://arxiv.org/abs/2610.28707) | 本文提出SimHack模拟器，仅凭少量小规模参考运行即可在强化学习训练开始前预测奖励破解行为何时出现，从而帮助提前选择风险更低的训练方案。 |
| [^230] | [Scalable Stress-Constrained Topology Optimization: A Super-resolution and Simultaneous Approach Based on Machine Learning](https://arxiv.org/abs/2610.28703) | 提出同步神经拓扑优化（SNTO）框架，利用自定义神经网络在单一优化循环中联合表示并优化状态、伴随和设计变量，配合独特的网络架构与硬件感知训练方案，实现了面向数百万自由度的可扩展应力约束拓扑优化。 |
| [^231] | [Reinforcement learning for hardware acceleration and system-level optimization on FPGA and integrated-circuit platforms: A survey](https://arxiv.org/abs/2610.28699) | 本综述系统梳理了100余项在FPGA和集成电路平台上实现强化学习的研究，按算法类别、抽象层次、硬件实现策略和应用领域进行分类，总结了强化学习硬件加速与系统级优化的架构与实现方法。 |
| [^232] | [Scaling in Games: Continued Pre-Training For Embodied Agents In Diverse Virtual Worlds](https://arxiv.org/abs/2610.28698) | 本文通过涵盖模型规模、数据量与数据多样性的120个训练配置，系统研究了持续预训练在多样化虚拟世界中的规模化规律，发现更多数据提升域内性能、数据多样性促进向相关世界的迁移、更大的模型规模带来性能提升。 |
| [^233] | [Interaction-Gated Time-Frequency Learning for Parameter-Efficient PPG-Based Heart Rate Estimation](https://arxiv.org/abs/2610.28693) | 本文提出TF-IG框架，通过参数高效的时域-频域双分支架构以及显式建模特征一致性与差异性的交互门控融合模块，实现了基于PPG的准确心率估计。 |
| [^234] | [Solar System Surface Technosignatures: Searching for Signs of Artificial Design on Extraterrestrial Terrain](https://arxiv.org/abs/2610.28687) | 该论文提出将技术签名搜索从人工射电信号扩展到行星表面的非随机人工图案，并建议结合高分辨率影像数据与机器学习异常检测方法，在月球和火星等地质不活跃天体上寻找可能存留数亿年的人工设计痕迹。 |
| [^235] | [What Git Does Not Record: Provenance Legibility in Two AI-Assisted Research Repositories](https://arxiv.org/abs/2610.28655) | 本研究通过审计同一作者带有不同类型AI智能体残留痕迹的两个研究仓库，发现代码仓库在披露AI参与科研时的溯源可读性呈反向差异——规范性规则痕迹保留85.6%的结论性行但丢失过程片段，叙事性草稿痕迹记录过程片段却仅存21.8%——从而揭示了Git仓库无法完整回答“AI如何参与研究”这一溯源问题的根本局限。 |
| [^236] | [To Keep or Not to Keep: Learning KV Cache Retention in Disaggregated LLM Serving Systems](https://arxiv.org/abs/2610.28637) | 提出了KVLearn框架，通过轻量级前缀复用预测器学习复用概率，并结合重计算、传输和存储成本进行成本感知的KV缓存保留决策，同时节省解耦式LLM服务中的内存与带宽。 |
| [^237] | [Planning under Noise with Neural Operators](https://arxiv.org/abs/2610.28623) | 本文提出PUN神经算子，通过在屏蔽泊松随机最优控制框架中为障碍物引入反射诺伊曼边界条件、为目标引入狄利克雷条件，并结合网格无关的蒙特卡洛求解器，学习噪声感知的价值场，从而在未见过的几何环境中实现噪声下的可靠规划。 |
| [^238] | [CountVideoBench: A Benchmark to Evaluate Prediction Biases on the Object Counting task by Video-Language Models](https://arxiv.org/abs/2610.28356) | 该论文提出CountVideoBench基准，利用2,600个覆盖[5,50]区间内26个计数值的合成视频，解决了现有视频问答基准计数分布严重偏向小数值、导致大数值上的模型预测偏差无法被评估的问题，从而系统性地评估视频语言模型在物体计数任务中的预测偏差。 |
| [^239] | [Observation Assimilation Transformer: Probabilistic Weather Analysis from Sparse Station Observations](https://arxiv.org/abs/2610.28327) | 提出观测同化Transformer（OAT），一种几何感知的神经过程模型，利用稀疏注意力机制同化稀疏站点观测，将数值天气预报先验场修正为高分辨率的概率性后验场，可扩展至数千站点并支持任意连续位置的预测。 |
| [^240] | [Fixed Sets and the Hardness of Computing Correlated Equilibria](https://arxiv.org/abs/2610.28141) | 本文证明计算多面体博弈的近似相关均衡是PPAD完全的（与纳什均衡同等难度），挑战了相关均衡比纳什均衡更容易求解的传统观点，并通过提出“不动集”这一不动点的自然推广建立了相关均衡与不动点理论的概念性联系。 |
| [^241] | [LaLa-World: Scaling Latent World Models with Natural Language Actions](https://arxiv.org/abs/2610.27851) | LaLa-World通过用语言统一描述动作与视觉变化、并将未来预测器构建于语言模型之上，使单个潜在世界模型能够跨仿真、机器人、图像和视频等异构数据进行联合训练，在MPC规划性能上超越了针对单一领域或任务训练的现有模型。 |
| [^242] | [Privacy Accounting for Empirical Denoisers](https://arxiv.org/abs/2610.27710) | 本文提出一种无需训练的截断隐私可问责经验去噪器，通过显式计算其每步敏感性，使扩散模型采样过程中的每一步去噪都可建模为高斯机制，从而实现对采样过程本身隐私损失的形式化核算。 |
| [^243] | [Hybrid Momentum Methods for Optimizing Deep Learning Models](https://arxiv.org/abs/2610.27674) | 提出混合动量方法HyM，将线性加权的长期历史梯度与几何折扣的快动量相结合，并据此构建经Newton–Schulz正交化的HyMUON优化器，在保留长期梯度信息与适应近期变化之间实现平衡。 |
| [^244] | [Lower Bounds for Smooth Nonconvex Zeroth-Order Optimization: The Price of Determinism under Low Curvature Trace](https://arxiv.org/abs/2610.27550) | 本文在低曲率迹（有效秩）条件下证明了随机化零阶非凸优化仅需 O(ET) 次函数调用且在常数因子内最优，而任何确定性方法都无法获得这种依赖有效秩的节省，从而量化了确定性方法的固有代价。 |
| [^245] | [Machine-learned spectral PDE solvers for fluids and plasmas](https://arxiv.org/abs/2610.27317) | 提出WARP方法，通过在求解PDE的空间离散化中进行弱形式发现，并以观测状态为锚点的多步推演微分学习，构建出数值稳定的机器学习谱方法PDE求解器，在流体和等离子体基准系统上显著提升了系数恢复、推演精度和统计度量。 |
| [^246] | [When Weak Submodularity Meets Matroids: How Does Standard Greedy Behave?](https://arxiv.org/abs/2610.26967) | 本文解决了弱次模最大化在一般拟阵约束下标准贪心算法最坏情况近似保证这一长期开放问题，给出了以次模比 $\gamma$ 和拟阵秩 $r$ 表达的精化下界。 |
| [^247] | [Beyond Policy Discrepancy: Capability-Aware Multi-Teacher On-Policy Distillation](https://arxiv.org/abs/2610.26950) | 提出能力感知的多教师在线策略蒸馏方法 CA-MOPD，将强化学习专家模型合并与验证器估计的师生能力差距相结合，根据学生已掌握的内容动态调整各领域蒸馏强度，克服了仅基于策略差异分配监督的局限性。 |
| [^248] | [Copying Under Repetition: From Prefix Matching to Position Hashing](https://arxiv.org/abs/2610.26722) | 本文提出利用小素数周期的VO旋转位置编码并结合中国剩余定理，证明了两层Transformer能在任意重复字符串下实现精确复制并正确终止，同时给出了顺序有限精度下复制的不可能性证明。 |
| [^249] | [Breaking Diffusion: Parallel Reasoning with Draft Refinement](https://arxiv.org/abs/2610.26542) | 该论文提出草稿精炼语言模型（DRLMs），利用单个去噪器并行提出并评估草稿修订，并通过校正自身token证据来衡量token所获得的上下文支持，从而突破传统扩散模型迭代去噪无法保证整体答案变优的局限。 |
| [^250] | [Self-organizing reinforcement learning: a learner-free swarm approaches the explore-exploit optimum](https://arxiv.org/abs/2610.25113) | 一个无需学习者与奖励信号的极简自组织觅食群体，通过Levy飞行探索、阈值级联招募与拥挤分散三者的耦合循环，能在约150步内动态逼近多智能体资源共享博弈的理想自由/Nash均衡分配（匹配斜率达0.94），且移除招募-运动耦合后匹配会崩溃至0.08。 |
| [^251] | [TrustAgent-RFF: Agentic Diagnostic Planning for Reliable RF Fingerprinting under Domain Shift](https://arxiv.org/abs/2610.24930) | 提出TrustAgent-RFF预测级可靠性框架，在不改动冻结I/Q分类器且无需目标域标签的条件下，通过融合反事实稳定性、原型一致性、采集域相似度与信号质量等射频证据，并利用源域校准的信任估计器输出预测可信度，从而解决域偏移下射频指纹识别可靠性不足的问题。 |
| [^252] | [An important Bottleneck in LLM Optimizers: Principle-Level Local Optima](https://arxiv.org/abs/2610.24291) | 本文揭示LLM优化器普遍存在“原理级搜索坍塌”瓶颈，即搜索被局限于狭窄的设计原理集合导致性能长期停滞，并通过跨三个领域的诊断基准与决策树协议证实78.3%的运行中存在该现象。 |
| [^253] | [First Passage Attention](https://arxiv.org/abs/2610.22714) | 该论文提出首次通过注意力（FPA），将注意力衰减机制从路径求和转向路径极值——根据学习标量场在中间跨度上的最深下凹来惩罚键的logit，从而能在保留较旧结构相关token的同时选择性擦除中间上下文，并证明常数维度的FPA头即可实现任意存活键子集，而传统softmax头或token级遗忘门需要与键数呈线性关系的维度。 |
| [^254] | [Generalization Guarantees of Randomized Block Coordinate Descent for Nonconvex Learning](https://arxiv.org/abs/2610.22213) | 本文为非凸学习中的随机块坐标下降算法建立了高概率泛化保证，并通过早停策略量化了优化与泛化的平衡，实现了总体风险梯度的最优收敛速率。 |
| [^255] | [N-Gram Memories at Two Bits: Quantizing Lookup Tables in Large Language Models](https://arxiv.org/abs/2610.21810) | 本文提出条件程序（CP）压缩表示方法，仅用两比特编码按需重构大语言模型n-gram记忆查找表中被请求的行，相比标准2比特量化将输出KL散度降低41–63%，并将DeepSeek的表存储从203GB压缩至51GB。 |
| [^256] | [DialFork: Going Beyond Factuality to Evaluate Procedural Hallucination in Conversational Agents](https://arxiv.org/abs/2610.21561) | DialFork是首个评估对话代理“程序性幻觉”的基准，通过在受控对话轮次位置注入四种幻觉类型构建了涵盖12个领域、24个任务的36,278个对话，并从三个粒度定义了刻画程序性偏差的评估任务。 |
| [^257] | [Scaling Multilingual Mixture-of-Experts: How Language Coverage Changes Capacity and Compute Allocation](https://arxiv.org/abs/2610.20247) | 本文提出了一种感知重复的联合缩放定律，揭示多语言MoE模型中活跃参数、训练词元、专家数量与语言覆盖范围之间的关系，并发现32个专家相比稠密模型可将语言从12种扩展到200种时的损失惩罚降低83%。 |
| [^258] | [Asymptotically Exact Negative Guidance of Diffusion Models via Positive-Unlabeled Learning](https://arxiv.org/abs/2610.19802) | 提出一种从显式目标密度采样的扩散模型负引导新方法，利用正-未标记学习在线估计样本属于不良条件的概率，从而实现渐近精确、偏差可通过增加计算量系统降低的负引导。 |
| [^259] | [On the Monitorability of Natural Language Autoencoders: Identification vs Discrimination](https://arxiv.org/abs/2610.18727) | 论文将自然语言自编码器的训练与监督部署形式化为两种不同的指称博弈，并指出二者之间的差距导致其解释虽能识别对应的激活，却难以让人类监控者从多个近似叙述中辨别出真实的那一个。 |
| [^260] | [Just Respect the Semantics: Image-Token Compression via Pre-LLM Semantic Variable-Resolution Tokenization and In-LLM Relevance Pruning](https://arxiv.org/abs/2610.18690) | Varest通过语义引导的可变分辨率token化与相关性token剪枝，仅用5.56%的图像token即可保留86%-92%的全token性能，实现高效且语义无损的图像token压缩。 |
| [^261] | [Compression Through Time: An Interplay of Separability, Stationarity and Capacity](https://arxiv.org/abs/2610.16660) | 该论文揭示了循环网络中的信息压缩现象由类别可分性、时间平稳性和有限的表示容量这三种相互竞争的机制共同驱动。 |
| [^262] | [Pessimistic Latent Task-aware Optimization for Robust Offline Meta-Reinforcement Learning](https://arxiv.org/abs/2610.15428) | 提出PLATO方法，基于任务潜变量位于低维流形上的假设，通过从训练质心向外扰动潜变量来合成反事实任务，让策略在训练中接触支持域外潜变量，从而实现对外推任务更鲁棒的离线元强化学习。 |
| [^263] | [Exploring Jailbreak Vulnerability Across Depth-Based Dynamic Inference in Large Language Models](https://arxiv.org/abs/2610.15207) | 本文首次系统研究了基于深度的动态推理对大语言模型越狱鲁棒性的影响，发现浅层退出始终比深层退出对越狱攻击更具鲁棒性，揭示了动态推理灵活性所带来的安全风险。 |
| [^264] | [DisCo Algebra: Why Disintegration and Combination Are Not Inverses](https://arxiv.org/abs/2610.15006) | 本文提出DisCo代数框架，用对称余操作刻画联合概率的分解、用非单位部分操作刻画组合，证明“组合∘分解”在可允许输入上是恒等操作，而“分解∘组合”则是一个有信息损失的重正化算子，从而揭示了分解与组合并非互逆。 |
| [^265] | [Caught in the Act: Probes Effectively Detect Sabotage and Catch Unverbalized Deception](https://arxiv.org/abs/2610.12445) | 该研究通过构建迄今最大的欺骗数据集并设计可跨层、跨标记聚合信息的新型探针架构，使白盒探针在 SHADE-Arena 上以 98.8% 的 AUC 超越前沿文本监控基线，甚至能检测出仅凭上下文无法察觉的“内省式欺骗”。 |
| [^266] | [Estimating great expectations under autoregressive language models with potentials](https://arxiv.org/abs/2610.11399) | 本文提出利用采样时免费获得的下一词元条件概率构造势函数，对语言模型下检验泛函的期望进行估计，在计算成本相近的情况下显著降低了估计方差。 |
| [^267] | [SciExam for ENSO: Can AI Agents Build Climate Models?](https://arxiv.org/abs/2610.10513) | 该论文提出了SciExam for ENSO基准测试，让AI智能体在六小时内仅凭真实观测数据自主构建ENSO低阶随机气候模型，并用隐藏评分器检验模型能否复现统计特性、恢复隐变量和预测保留年份，结果显示十二个智能体系统中有六个的模型优于已发表的模型。 |
| [^268] | [Minimal Witness Reinforcement Learning](https://arxiv.org/abs/2610.07226) | 本文提出最小见证强化学习（MWRL），利用基于集合并集覆盖损失的信用分配机制，仅凭单一黑盒验证器的信号即可同时实现解的最小性与多个备选解的恢复。 |
| [^269] | [TrustmeWatcher: An Application for Workplace Micro-Sensing and Explainable Well-Being Feedback](https://arxiv.org/abs/2610.06657) | TrustmeWatcher 是一款整合操作系统级活动感知数据与问卷自我报告、并借助本地运行的人工智能模型生成可解释幸福感反馈的职场微感知应用程序。 |
| [^270] | [Physics is the Best Teacher: Consistency Learning for Time-Invariant Operators of Chaotic Dynamics](https://arxiv.org/abs/2610.04108) | 提出通过推导时不变演化算子必须满足的物理一致性方程，将物理知识嵌入一致性学习，实现对混沌系统长期行为的单次跳跃式高效预测，在精度与时间外推能力上优于逐步推进的传统方法。 |
| [^271] | [Predictor-Guided Latent Space Codon Optimization for Maximizing Protein Expression](https://arxiv.org/abs/2610.03098) | 提出潜空间密码子优化方法LSCO，通过将序列映射到预训练mRNA语言模型的潜空间，将离散的密码子优化问题转化为可梯度搜索的连续问题，并结合不确定性感知的表达预测器、最小自由能正则化、自然性先验和约束解码，以最大化蛋白质表达。 |
| [^272] | [Proximal Balancing for Causal Effect Estimation under Unmeasured Confounding](https://arxiv.org/abs/2609.40051) | 提出邻近平衡方法，将经典协变量平衡思想扩展到仅能通过代理变量观测的混杂因素，无需指定代理角色、求解逆问题或依赖潜变量模型，即可实现未测量混杂下的因果效应估计。 |
| [^273] | [A Generalisation Signal Need Not Be a Model-Selection Signal](https://arxiv.org/abs/2609.39099) | 该研究发现，尽管受Hessian启发的新型仅前向传播代理指标在大多数任务上与泛化差距相关，但这种泛化信号并不能可靠地识别出部署环境下最优的模型，表明泛化信号未必能作为模型选择的信号。 |
| [^274] | [Alignment via Training Against Probes Without Losing Monitorability](https://arxiv.org/abs/2609.38645) | 该研究提出探针引导微调方法，发现使用持续更新的探针作为直接训练信号能有效减少模型有害行为并提升诚实性，同时保持模型效用而不丧失可监控性，而静态探针则容易被模型利用规避。 |
| [^275] | [Interpretable but Fragile? Robustness of Concept Bottlenecks under Geometric-Semantic Perturbations](https://arxiv.org/abs/2609.38625) | 该论文提出了一个基于生成器的统一评估框架，首次将潜空间中的连续几何扰动与概念空间中的离散语义干预区分开来，从经验性与可认证两个层面系统比较概念瓶颈模型与标准分类器的鲁棒性，从而调和了此前相互矛盾的研究结论。 |
| [^276] | [SAGE: Semantic Audio Generative Encoder](https://arxiv.org/abs/2609.32755) | SAGE是一个仅1.05亿参数、只用公开音乐训练的紧凑音频自编码器，通过蒸馏预训练音频-文本模型的嵌入来构建语义潜在空间，在推理速度、重建质量和语义结构三方面同时超越了现有的更大模型。 |
| [^277] | [SoLiD26: A First Principles Solid-Liquid Interface Dataset for Machine-learned Interatomic Potentials](https://arxiv.org/abs/2609.28013) | SoLiD26是一个包含1540万个第一性原理原子结构、涵盖15种化学元素的固液界面数据集，可用于训练和评估面向电化学、催化和腐蚀等应用的机器学习原子间势。 |
| [^278] | [Beyond Appearance Shifts: Task-Semantic Action Calibration for VLA Models](https://arxiv.org/abs/2609.23650) | 提出BAS-VLA任务语义动作校准框架，通过以语义破坏为中心的校准核心与证据门控的保持辅助模块，使VLA模型在场景外观变化时保持动作稳定，而在任务语义改变时产生显著不同的行为。 |
| [^279] | [Causal Discovery via Transformed Low-Rank Quantile Surfaces](https://arxiv.org/abs/2609.16931) | 本文提出低秩分位数曲面（LRQS）二元因果模型，证明在因果方向上变换后的条件分位数曲面具有低秩结构从而保证因果方向的通用可识别性，并提出了一种交替进行秩约束近似与单调变换保序估计的非参数因果发现方法。 |
| [^280] | [Feature Superposition in Neural Networks: From Theory to Practice](https://arxiv.org/abs/2609.06862) | 本综述系统梳理了神经网络中特征叠加的几何、学习与计算理论，并比较了从训练网络中恢复可解释特征的实用方法及其评估标准，从而将理论研究与实证证据连接起来。 |
| [^281] | [When Retain Constraints Conflict: Mitigating Forget-Retain Interference in Tabular Data](https://arxiv.org/abs/2609.06786) | 该论文揭示了表格数据遗忘中独有的“模式引起遗忘-保留重叠”难题，并提出冲突感知遗忘方法（CAU）来缓解遗忘与保留之间的相互干扰。 |
| [^282] | [DrugReason: Dynamic Multi-View Reasoning over Knowledge Graph and Language Evidence for Drug Repurposing](https://arxiv.org/abs/2609.06779) | DrugReason 提出了一个动态多视角推理框架，将知识图谱的结构化多跳推理与大语言模型灵活的机理解释相结合，从而实现更可靠、更具事实依据的药物重定位预测。 |
| [^283] | [Interface-Induced Trajectory Censoring](https://arxiv.org/abs/2609.03966) | 该论文发现智能体评估中的工具调用率可能被服务栈接口“审查”为零——即使模型实际发出了格式良好的调用，且2x2析因实验证明全部效应源于聊天模板与解析器之间的交互（修复任何一方都无效），仅更换服务适配器即可使同一模型得分从 0.00 跃升至 0.96/0.19。 |
| [^284] | [Do Tabular Foundation Models Know Physics? Contamination, Units, and the Deterministic Limit](https://arxiv.org/abs/2609.02766) | 本文在316个物理方程采样的数据集上评估四种表格基础模型，发现其性能虽全面超越基线模型，但其贝叶斯先验既无法表示无噪声机制也无法表示物理单位，因此只能对物理数据进行插值而不能充当真正的物理模型。 |
| [^285] | [S^3martCirc: Self-supervised Smart Circuit Discovery](https://arxiv.org/abs/2609.00755) | 提出S^3martCirc，一种自监督的智能电路发现方法，将电路发现与功能解释两个阶段统一起来，解决了组件重要性与功能角色相互依存这一被传统两阶段范式忽视的问题。 |
| [^286] | [Euclidean Fourier Neural Operators](https://arxiv.org/abs/2608.28425) | 提出欧几里得傅里叶神经算子（EFNO），通过将谱核参数化为物理波矢的连续函数，使神经算子能够跨不同形状和大小的周期域保持一致地学习与迁移，克服了传统FNO的域依赖性问题。 |
| [^287] | [Hierarchical Exponential-Gaussian Mixtures for Watch-Time Distribution Prediction](https://arxiv.org/abs/2608.23356) | 本文提出了一种分层指数-高斯混合模型，通过分层分解和正则化技术解决了现有EGMN模型的方差崩溃和组件冗余问题，从而提升了短视频推荐中观看时间预测的准确性。 |
| [^288] | [Crystal-structure design by agentic AI in a language of motifs](https://arxiv.org/abs/2608.15900) | MatEvolve利用可解释的基元语言和主体智能体，无需微调即达到新结构原型，比生成模型高出三倍以上，显著提升材料设计的外推能力。 |
| [^289] | [Learning reshapes power-law anisotropy in internal representations](https://arxiv.org/abs/2608.15239) | 该论文通过精确求解两层线性网络的学习动态，揭示了幂律各向异性在内部表示中的形成机制，并发现特征学习机制下指数呈现非单调且多阶段演化，而惰性机制下行为不同。 |
| [^290] | [Small-Scale Experiments: Are We There Yet?](https://arxiv.org/abs/2608.11859) | 本文证明小规模实验中缩放定律的可信度受超参数敏感性影响，且调优超参数比模型大小更关键，并解释了随规模增大超参数优化变易的原因。 |
| [^291] | [SeFoRA: Sketch-Aggregated Federated Low-Rank Adaptation with Heterogeneous Client Ranks](https://arxiv.org/abs/2608.10144) | SeFoRA 通过让各客户端传输本地 ΔW 更新的线性草图实现联邦服务器端的直接聚合，解决了联邦 LoRA 中客户端秩异构导致的维度不兼容和双线性失配问题，并具有 $\mathcal{O}(T^{-1/2})$ 速率的一阶稳定点邻域收敛保证。 |
| [^292] | [Goal Retrieval versus Action-Value Ordering: Contrastive Critics under Best-of-$K$ Selection](https://arxiv.org/abs/2607.27422) | 本文揭示了对比式评论家“擅长目标检索却不擅长动作价值排序”的差距：未归一化对比得分在总体最优处对行为策略价值单调，但余弦归一化及有限训练的网络在 Best-of-K 选择中可能给出错误的动作排序。 |
| [^293] | [Inverse RL Helps Align AI by Imitating Humans](https://arxiv.org/abs/2607.24900) | 提出PARED方法，受逆强化学习启发，仅从专家演示中学习显式且可解释的奖励函数，从而在没有偏好判断或验证器的情况下实现对语言模型的有效对齐。 |
| [^294] | [DFAH-Bench: Benchmarking Observable Agent Instability in Financial Decision-Making](https://arxiv.org/abs/2607.20491) | 该论文提出DFAH-Bench基准，通过同时衡量决策一致性与工具路径一致性，揭示金融智能体在决策结果高度稳定（约95%）的情况下，其背后的工具执行过程却存在显著不稳定性（一致性仅约50%）。 |
| [^295] | [Conditional Optimal Bridge for Riemannian Activation Steering](https://arxiv.org/abs/2607.10517) | 该论文提出COBRAS方法，将激活引导建模为残差流超球面上的薛定谔桥问题，从适定的优化问题出发有原则地推导出对数密度比引导目标，并通过查询自适应的引导场和支撑分数控制分布外干预，在多个模型上优于现有方法。 |
| [^296] | [Trees from Marginals: Autoregressive drafting with factorized priors](https://arxiv.org/abs/2607.06763) | 本文提出Weaver，一种从因子化草稿模型的top-K边缘分布构建提议树的轻量级自回归适配器，恢复了token间的条件依赖，并结合无需回滚的树验证算法与优化的CUDA内核，相比自回归解码实现了4.08倍加速。 |
| [^297] | [Learning When to Automate: Queue Control in Human-AI Service Systems](https://arxiv.org/abs/2607.06017) | 该论文针对人机协作服务系统，提出了结合上置信界学习与漂移加惩罚控制的UCB-DPP策略，能够在聊天机器人成功概率和人工客服服务速率未知的情况下在线学习系统参数，从而在减少人工拥塞与控制自动化成本之间实现最优的排队与资源分配决策。 |
| [^298] | [To Retain or to Adapt? Generalizing Continual Learning](https://arxiv.org/abs/2607.05609) | 该论文挑战了持续学习以“保留知识”为中心的传统假设，将持续学习重新形式化为以最小化平均终身误差为目标的在线优化问题，并提出“迁移效率”作为衡量不稳定性与瞬态误差之间张力的定量指标。 |
| [^299] | [The Binary Tree Mechanism is Optimal for Differentially Private Continual Counting](https://arxiv.org/abs/2607.00876) | 本文证明了在固定隐私参数下，差分隐私持续计数问题中误差对数据流长度的依赖是不可回避的下界（近似DP下 $\Omega(\log^{3/2} n)$，纯DP下 $\Omega(\log^2 n)$），从而确立了二叉树机制的最优性。 |
| [^300] | [Mixture-of-Control: State-Aware Fine-Tuning for Transformer-based Models](https://arxiv.org/abs/2606.31397) | 提出了控制混合方法，将逐块控制状态作为稀疏混合专家中的专家，实现高效的跨块通信，自适应整合局部与全局控制信号，从而在状态感知微调中增强表示学习。 |
| [^301] | [Sequential sparse Gaussian process quantile regression](https://arxiv.org/abs/2606.31284) | 本文提出一种序贯稀疏高斯过程分位数回归框架，通过诱导变量和拉普拉斯近似实现高效后验推断，并将预测不确定性分解为条件先验与后验诱发两个分量，以此驱动诱导输入填充和数据采集两种自适应机制，实现计算资源的动态分配。 |
| [^302] | [FracEvent: Event-Camera Simulation via Fractional-Relaxation Pixel Dynamics](https://arxiv.org/abs/2606.26636) | 提出FracEvent事件仿真器，通过分式松弛电压动力学精确建模像素生命周期，以改善事件时序保真度和下游任务迁移性能。 |
| [^303] | [TNODEV: Toolbox for Neural ODE Verification](https://arxiv.org/abs/2606.16567) | TNODEV是首个针对神经ODE的端到端形式化验证工具箱，通过集成证伪检查、基于混合单调性的快速可达性分析、输入集精化循环与并行调度，显著提升了验证精度。 |
| [^304] | [$\mu_0$: A Scalable 3D Interaction-Trace World Model](https://arxiv.org/abs/2606.13769) | μ₀通过预测物体、工具、手等显著交互点的平滑3D轨迹（而非密集像素或具身特定动作标签）构建可扩展的世界模型，并借助TraceExtract系统从多样化视频源自动提取3D监督，实现与具身形态无关的机器人学习预训练。 |
| [^305] | [Rethinking Synthetic Rationale Supervision for Disease Prediction](https://arxiv.org/abs/2606.10279) | 该研究发现，在疾病预测中使用合成推理理由进行“推理优先”监督微调反而会持续且显著地损害预测性能，且这种退化跨模型系列、数据规模和任务普遍存在，并非由推理理由质量不佳所致。 |
| [^306] | [Lean4Agent: Formal Modeling and Verification for Agent Workflow and Trajectory](https://arxiv.org/abs/2606.06523) | 本文提出Lean4Agent，这是首个利用Lean4依赖类型形式化语言对智能体工作流的语义一致性进行形式化建模与验证、并能对执行轨迹揭示的运行时故障进行定位的框架。 |
| [^307] | [Let It Be Simple: One-Step Action Generation for Vision-Language-Action Models](https://arxiv.org/abs/2606.05737) | 该论文发现VLA动作生成与文生图截然不同——只需将训练时间步分布偏向噪声端点，或直接将多步迭代解码替换为单次前向传播，即可获得性能相当的单步策略，其根本原因在于VLA生成的条件-目标结构更接近图生文任务。 |
| [^308] | [InfoQuant: Shaping Activation Distributions for Low-Bit LLM Quantization](https://arxiv.org/abs/2605.26175) | 该论文提出InfoQuant，将激活变换重新定义为面向量化器的分布设计问题，并从信息论视角证明量化友好的激活分布应同时具备更小的数值范围和足够的离散程度，从而提升大语言模型低比特量化的效果。 |
| [^309] | [Metropolis-Scale Resilient and Trustworthy Traffic Flow Inference Using Multi-Source Data](https://arxiv.org/abs/2605.25004) | 提出任务感知注意力神经过程（TA-ANP）统一概率框架，融合浮动车数据与稀疏固定检测器测量，实现无需重训练即可适应传感配置变化的大都市尺度弹性可信全网交通状态推断。 |
| [^310] | [From Theory to Decision Rule: Calibrating the Noisy-Label Crossover for Vision-Language Model Weak Supervision Across Three Medical-Imaging Benchmarks](https://arxiv.org/abs/2605.24771) | 本文在三个医学影像基准上实证测定了视觉-语言模型伪标签从有益转为有害的交叉点，并将其转化为可操作的专家标注数量预算决策规则。 |
| [^311] | [Biological Context Does Not Reliably Improve LLM-Guided Experimental Optimization](https://arxiv.org/abs/2605.15341) | 通过对45个重建自341篇论文的生物学任务进行大规模测试，研究发现为LLM提供真实的生物学背景信息（参数名称、单位和标签）反而会降低实验优化性能，在超过一半的任务上不如使用无生物学语义的匿名化名称。 |
| [^312] | [A Novel Schur Decomposition-Based Weight Projection Method for Stable State-Space Neural-Network Architectures](https://arxiv.org/abs/2605.14489) | 本文提出一种基于Schur分解的权重投影方法，通过将状态空间神经网络中状态矩阵的准三角因子动态投影到最近的稳定矩阵上，以最小过参数化代价保证模型动态的渐近稳定性，且完全兼容反向传播训练。 |
| [^313] | [NodeSynth: Socially Aligned Synthetic Data for AI Evaluation](https://arxiv.org/abs/2605.14381) | NodeSyn通过将微调的分类体系生成器与可信文献来源和敏感属性相结合，生成反映真实社会背景的敏感领域评估数据，其诱发的模型失败率达到通用合成基线的3.1倍、真实人类查询的1.6倍，显著优于现有合成数据方法。 |
| [^314] | [The Sample Complexity of Multiple Change Point Identification under Bandit Feedback](https://arxiv.org/abs/2605.13252) | 该论文提出了一种在老虎机反馈下识别多个变化点的自适应算法，并证明了其样本预算的非渐近上下界，同时揭示在一般置信水平下样本复杂度不仅取决于跳跃幅度，先前仅由跳跃幅度决定复杂度的认识是不完整的。 |
| [^315] | [A Composite Activation Function for Learning Stable Binary Representations](https://arxiv.org/abs/2605.11558) | 提出了一种由sigmoid与tanh复合而成的重尾激活函数（HTAF），作为Heaviside函数的平滑近似，在零输入附近保持较大梯度且尾部梯度衰减更慢，从而实现了脉冲神经网络、二值神经网络和深度Heaviside神经网络的稳定梯度训练。 |
| [^316] | [Language Modeling with Hyperspherical Flows](https://arxiv.org/abs/2605.11125) | 提出𝕊-FLM，一种在超球面上通过旋转向量生成序列的潜在流语言模型，避免了基于one-hot向量的高维运算带来的高昂训练成本，并赋予噪声注入更清晰的语义解释。 |
| [^317] | [Kinetic theory for Transformers and the lost-in-the-middle phenomenon](https://arxiv.org/abs/2605.09213) | 该论文将Transformer的因果自注意力建模为非可交换相互作用粒子系统，通过平均场极限与累积量展开分析，首次严格证明了“迷失在中间”现象，即词元检索性能随其在提示中的位置呈U形分布。 |
| [^318] | [Path-Coupled Bellman Flows for Distributional Reinforcement Learning](https://arxiv.org/abs/2605.08253) | 提出 PCBF 方法，通过源一致的贝尔曼耦合路径与共享基础噪声进行流匹配，在连续时间框架下学习回报分布，同时避免了投影、边界失配与高方差自举问题。 |
| [^319] | [RelAgent: LLM Agents as Data Scientists for Relational Learning](https://arxiv.org/abs/2605.07840) | 提出了RelAgent，一个基于大语言模型的自主数据科学家，它通过构建SQL特征程序并结合经典模型，为关系学习提供快速、确定性且内在可解释的预测。 |
| [^320] | [Rank-Variance Scores for Importance Sampling and Data Pruning under Label Contamination](https://arxiv.org/abs/2605.07551) | 该论文提出了一种基于K个独立训练代理模型损失排名方差的非单调评分方法，以解决标签污染下基于损失的采样与剪枝反而优先选中污染样本的失效问题，并将总秩方差分解为代理间部分（分歧正则化重要性采样DR-IS）和代理内部分。 |
| [^321] | [Adaptive Pluralistic Alignment: A pipeline for dynamic artificial democracy](https://arxiv.org/abs/2605.01642) | 本文提出适应性多元化对齐（APA）框架，通过低秩奖励基分解、社会选择理论投票评审团以及随价值观变化的高效权重更新三个阶段，使AI系统能够持续跟踪演变的社会价值观、避免价值锁定，而无需昂贵的重新预训练。 |
| [^322] | [ProMORNA: An End-to-End Generative Framework for Full-Length mRNA Design](https://arxiv.org/abs/2605.01513) | ProMORNA是一个零样本端到端生成框架，通过BART风格架构预训练及两阶段后训练（冷启动监督微调加多目标强化学习），直接生成全长治疗性mRNA，无需区域化生成或昂贵的推理时搜索。 |
| [^323] | [CellTwin: Reproducible Cellular Network Emulation via High Resolution, in situ Measurement-Driven ML](https://arxiv.org/abs/2604.26080) | 本文提出首个支持多客户端的机器学习驱动蜂窝网络仿真器CellTwin，直接从高分辨率原位网络遥测数据中学习真实5G调度器的资源分配行为，实现了可复现的高保真网络仿真。 |
| [^324] | [Fine-Tuning Small Reasoning Models for Quantum Field Theory](https://arxiv.org/abs/2604.18936) | 本文首次对大语言模型在理论物理可验证任务上的强化学习与监督微调进行学术比较，并构建了可生成超过2,500个量子场论问题的数据管线来微调小型推理模型。 |
| [^325] | [A Mechanistic Study of Algorithmic Generalization in Diffusion Models on Modular Arithmetic](https://arxiv.org/abs/2604.17673) | 该研究以模加法任务上的“顿悟”现象为测试平台，从机制上揭示了扩散模型的算法泛化方式：单图像场景下模型通过周期性表示编码操作数并在自注意力-前馈接口处按三角加法恒等式组合实现模加法，而高多样性多图像场景下迭代采样过程则被划分为算术计算与视觉精细化两个阶段。 |
| [^326] | [On the Expressive Power and Limitations of Multi-Layer SSMs](https://arxiv.org/abs/2604.14501) | 多层状态空间模型（SSM）求解函数复合问题时必须满足深度、状态维度与标量精度之间的下界权衡（d²p=Ω(N/L³)），且输入后推理无法绕过该通信下界，从而揭示了多层SSM表达能力的基本限制。 |
| [^327] | [Generalization and Scaling Laws for Mixture-of-Experts Transformers](https://arxiv.org/abs/2604.09175) | 该论文为专家混合Transformer建立了泛化与缩放理论，证明其泛化误差由活跃参数预算主导并附加路由开销，且近似误差可通过扩展活跃容量或增加专家数量来降低，具体取决于主导瓶颈。 |
| [^328] | [The Stepwise Informativeness Assumption: When Do Entropy Dynamics Track Reasoning Correctness in LLMs?](https://arxiv.org/abs/2604.06192) | 本文提出步进信息性假设（SIA），从理论上刻画了熵降低何时能真实反映大语言模型推理的正确性进展，并指出这种对齐取决于训练机制将推理轨迹与正确答案耦合的方式，同时给出了该现象的可观测特征和失效情形。 |
| [^329] | [FactReview: Evidence-Grounded Peer Review with Execution-Based Claim Verification](https://arxiv.org/abs/2604.04074) | FactReview通过结合文献、参考检查和代码执行的多源证据审计，显著提升了论文声明验证的准确性和评审效率。 |
| [^330] | [A Survey of On-Policy Distillation for Large Language Models](https://arxiv.org/abs/2604.00626) | 这篇综述论文通过学生生成状态、教师信息与策略更新三者之间的相互作用，建立了在线策略蒸馏（OPD）的统一分析框架，并将其与自蒸馏、多教师集成、强化学习和交互式智能体等方向联系起来，系统阐释了不同方法的设计选择对能力迁移、训练效率和部署的影响。 |
| [^331] | [Insights on back marking for the automated identification of animals](https://arxiv.org/abs/2603.25535) | 本研究通过训练ResNet-50神经网络对猪的背部标记进行分类分析，揭示了标记设计的核心原则——标记必须在运动模糊、多视角和遮挡条件下保持清晰无歧义，从而为基于机器学习的动物个体自动识别提供设计指导。 |
| [^332] | [MCLR: Improving Conditional Modeling via Inter-Class Likelihood-Ratio Maximization and Unifying Classifier-Free Guidance with Alignment Objectives](https://arxiv.org/abs/2603.22364) | 本文提出MCLR微调目标，通过显式最大化类间似然比来强化类特定结构，从而在不依赖无分类器引导的情况下显著提升扩散模型的条件生成能力，并缩小了与CFG的差距。 |
| [^333] | [Flow Matching Policy Optimization with Mirror Descent and Entropy Constraints](https://arxiv.org/abs/2603.17685) | 本文提出FMER框架，证明基于ODE的流匹配可同时实现无需仿真的策略优化与可解析计算的熵，并通过镜像下降与熵约束在在线强化学习中有效平衡策略表达能力与探索-利用权衡。 |
| [^334] | [Zero-Shot Cross-City Generalization in End-to-End Autonomous Driving: Self-Supervised versus Supervised Representations](https://arxiv.org/abs/2603.11417) | 该研究将零样本跨城市迁移作为端到端自动驾驶的表示级压力测试，通过在严格地理划分下系统对比I-JEPA、DINOv2和MAE等自监督骨干网络与监督式预训练表示，揭示了视觉预训练方式对地理域偏移下泛化能力的关键影响。 |
| [^335] | [SYNAPSE: Framework for Neuron Analysis and Perturbation in Sequence Encoding](https://arxiv.org/abs/2603.08424) | SYNAPSE是一个无需重新训练的系统化框架，通过提取逐层[CLS]表示、线性探针获取神经元排名以及前向钩子干预，实现对Transformer模型内部行为的透明分析与压力测试。 |
| [^336] | [The Alien Space of Science: Sampling Coherent but Cognitively Unavailable Research Directions](https://arxiv.org/abs/2603.01092) | 该论文提出一种方法，通过将文献分解为“想法原子”并联合建模连贯性与认知可得性，来系统性地发现科学上合理但现有研究社区不太可能自发提出的研究方向。 |
| [^337] | [MARS: Harmonizing Multimodal Convergence via Adaptive Rank Search](https://arxiv.org/abs/2603.00720) | 提出MARS方法，利用双缩放定律自适应搜索LoRA秩对，以平衡多模态大语言模型微调中不同模态的收敛速度，避免负面干扰并提升任务性能。 |
| [^338] | [GLUScope: A Tool for Analyzing GLU Neurons in Transformer Language Models](https://arxiv.org/abs/2602.23826) | GLUScope是一个开源工具，通过分析基于GLU激活函数的语言模型中神经元门控与输入激活的四种正负符号组合（激活象限），帮助可解释性研究者揭示神经元促进或抑制特定token等截然不同的功能行为。 |
| [^339] | [2-Step Agent: How a Bayesian Decision Maker Learns from AI-Decision Support](https://arxiv.org/abs/2602.21889) | 本文提出“两步智能体”这一通用计算框架，用以刻画机器学习预测如何作为信息来源改变理性贝叶斯决策者的信念，进而影响因果效应估计、下游决策及其结果，并在多元线性-高斯设定下给出了可解的贝叶斯推断求解方案。 |
| [^340] | [Analyzing and Improving Chain-of-Thought Monitorability Through Information Theory](https://arxiv.org/abs/2602.18297) | 本文用信息论揭示了思维链可监控性的必要非充分条件，识别出信息鸿沟和引导误差两类失效来源，发现思维链混淆可在无优化压力下自发出现，并证明可通过针对性干预系统性缓解监控的脆弱性。 |
| [^341] | [Learning to Allocate Resources with Censored Feedback](https://arxiv.org/abs/2602.06565) | 该论文研究了删失反馈下的在线预算分配问题，证明了 $\Omega(\sqrt{T})$ 的最坏情况遗憾下界，并提出了一种结合非平凡参数估计与置信界的乐观算法 RA-UCB，用于联合估计未知参数并最大化累积奖励。 |
| [^342] | [RuleSmith: Multi-Agent LLMs for Automated Game Balancing](https://arxiv.org/abs/2602.06232) | RuleSmith框架将多智能体大语言模型自博弈与高斯过程贝叶斯优化相结合，在参数化的规则空间中自动搜索游戏平衡配置，并在简化的文明风格游戏CivMini上实现了接近均衡的双阵营胜率。 |
| [^343] | [Detecting Misbehaviors of Large Vision-Language Models by Evidential Uncertainty Quantification](https://arxiv.org/abs/2602.05535) | 提出了一种细粒度的证据不确定性量化方法（EUQ），通过同时捕捉知识冲突与信息缺失这两类认知不确定性，有效检测大型视觉-语言模型的不当行为。 |
| [^344] | [HealthMamba: An Uncertainty-aware Spatiotemporal Graph State Space Model for Effective and Reliable Healthcare Facility Visit Prediction](https://arxiv.org/abs/2602.05286) | 提出了不确定性感知的时空框架HealthMamba，通过统一时空上下文编码器融合静态与动态信息、利用新型图状态空间模型GraphMamba进行分层时空建模，实现准确且可靠的医疗机构就诊预测。 |
| [^345] | [Extending ROC Analysis to Uncertainty-Aware Risk Prediction with an Interval-Based AUC (iAUC)](https://arxiv.org/abs/2602.04775) | 提出不确定性感知的ROC框架与基于区间的AUC（iAUC）指标，用于评估区间值风险预测的判别性能，并将正负样本排序分解为自信正确、模糊和自信错误三个区域。 |
| [^346] | [Machine Unlearning in Low-Dimensional Feature Subspace](https://arxiv.org/abs/2601.22456) | 提出LOFT方法，通过在预训练模型的低维特征子空间中优化一个可插入的小型主投影矩阵来分离保留数据与遗忘数据的信息，从而以极低开销实现有效且高效的机器遗忘。 |
| [^347] | [From Fuzzy to Exact: The Halo Architecture for Infinite-Depth Reasoning via Rational Arithmetic](https://arxiv.org/abs/2601.18702) | 提出基于有理数域和精确推理单元的Halo架构，通过微环（丢番图近似约束内存）与宏环（符号对齐保证逻辑一致性）的双环拓扑，以精确运算取代模糊浮点运算，摆脱数值稳定技巧并实现无限深度推理。 |
| [^348] | [Does Centralized Learning Help Embodied Agents Coordinate, and When Can a Tabular Study Tell?](https://arxiv.org/abs/2601.17454) | 该论文在捕食者-猎物环境中，于多种捕食者与猎物配对及不同网格规模下，系统比较独立Q学习与中心化Q学习，并通过预先设定的直接协调性度量来检验集中式学习何时真正有助于具身智能体的协调。 |
| [^349] | [RubRIX: Rubric-Driven Risk Mitigation in Caregiver-AI Interactions](https://arxiv.org/abs/2601.13235) | 本文提出了RubRIX——一个基于关怀伦理理论、经临床医生验证的评分量表框架，将LLM在照护者交互场景中的回应风险操作化为五个维度，并在超过2万条真实照护者查询上验证，量表引导的改进可将风险成分降低45-98%。 |
| [^350] | [ChemBART: Pre-training a Reaction Language Model for Comprehensive Synthesis Planning](https://arxiv.org/abs/2601.02915) | ChemBART是一种以反应为中心进行预训练的BART语言模型，通过在整体反应表达式上进行掩码填充预训练，能够统一解决前体/试剂生成、产率回归、分子性质分类等多种化学任务，并可与蒙特卡洛树搜索结合实现多步合成路线设计。 |
| [^351] | [Structure is Supervision: Multiview Masked Autoencoders for Radiology](https://arxiv.org/abs/2511.22294) | 提出多视角掩码自编码器 MVMAE 及其结合放射学报告的扩展版 MVMAE-V2T，利用放射影像天然的多视角结构作为自监督信号，在三个大规模数据集的疾病分类任务上持续超越有监督和视觉-语言基线方法。 |
| [^352] | [Synthesis and Verification of Register Automata over Data Sequences (Technical Report)](https://arxiv.org/abs/2511.19100) | 该论文提出了针对有序数据序列上确定性寄存器自动机的局部搜索合成方法与精确鲁棒性检查器，后者对固定寄存器数量可在多项式时间内运行，并通过将鲁棒性归约为可覆盖性和最短路径问题实现高效验证。 |
| [^353] | [Operon: Incremental Construction of Ragged Data via Named Dimensions](https://arxiv.org/abs/2511.16080) | Operon是一个基于Rust的工作流引擎，通过命名维度与显式依赖关系的形式化方法，结合静态验证的领域特定语言和增量式的动态任务调度，实现了对不规则数据形状与依赖关系的原生追踪与管理。 |
| [^354] | [Structured Temporal Causality for Interpretable Multivariate Time Series Anomaly Detection](https://arxiv.org/abs/2510.16511) | 提出了OracleAD，一个简单可解释的无监督多元时间序列异常检测框架，它将每个变量的时间动态编码为因果嵌入，通过自注意力机制建模动态空间关系，并与代表正常状态的稳定潜在结构对齐以检测异常。 |
| [^355] | [Memorizing Long-tail Data Can Help Generalization Through Composition](https://arxiv.org/abs/2510.16322) | 记忆长尾样本与特征组合能力相结合，可以使模型对训练中从未出现过的长尾特征组合的稀有测试样本做出正确预测，且这种组合能力依赖于模型架构。 |
| [^356] | [Sculpting the Latent Space: Disentanglement with WAE-MMD and Programmable Priors](https://arxiv.org/abs/2510.11953) | 该论文通过非参数化MMD直接塑形聚合后验，提出支持可编程先验的WAE-MMD方法，并引入无监督纠缠度量LPS，使潜空间纠缠度降低70%，实现无监督解耦表示学习的最先进水平。 |
| [^357] | [Hyperdimensional Probe: Decoding LLM Representations via Vector Symbolic Architectures](https://arxiv.org/abs/2509.25045) | 提出超维探针——一种结合向量符号架构与神经探测的混合式有监督探针，统一了有监督探针、稀疏自编码器和logit分析等可解释性方法，从而更全面地解码大语言模型的内部表示。 |
| [^358] | [BrainPro: Process-conditioned Pretraining with Topographic Retrieval for Large EEG Models](https://arxiv.org/abs/2509.22050) | BrainPro提出了一种过程条件化的自监督预训练框架，通过拓扑检索将不同数据集的电极导联配置映射到通用通道-区域模板，并同时学习共享表征与过程关联表征，从而解决EEG基础模型中心理过程线索利用不足和导联配置异构两大挑战。 |
| [^359] | [TopoMap: A Feature-based Discriminator of the Topographical Regions in the Test Input Space](https://arxiv.org/abs/2509.03242) | TopoMap是一种模型无关的方法，通过基于输入特征对测试输入空间进行拓扑划分，帮助用户识别代表性不足、失败率高或模型更新后预测发生变化的区域，从而指导数据生成和标注工作的优先级排序。 |
| [^360] | [Stack Trace-Based Crash Deduplication with Transformer Adaptation](https://arxiv.org/abs/2508.19449) | 本文提出dedupT，一种基于Transformer的崩溃去重方法，通过整体建模堆栈跟踪并适配预训练语言模型，显著提升了重复崩溃排序和唯一崩溃检测的准确性，减少了人工分类负担。 |
| [^361] | [Can synthetic data reproduce real-world findings in epidemiology? A replication study using adversarial random forests](https://arxiv.org/abs/2508.14936) | 该研究提出对抗性随机森林（ARF）作为合成表格型流行病学数据的高效便捷方法，并通过复制六项流行病学研究的统计分析，验证了合成数据能在保护隐私的同时可靠重现真实世界的研究发现。 |
| [^362] | [UniPrompt-CL: Efficient Single-Inference Continual Learning for Medical Image Classification](https://arxiv.org/abs/2508.10954) | UniPrompt-CL是一种无需回放的域增量学习框架，通过在Transformer各层共享统一提示池并以单一主干网络、单次推理的方式实现医学图像分类的高效持续学习，在糖尿病视网膜病变和皮肤癌任务上以更低的训练成本取得了更高准确率并显著减少遗忘。 |
| [^363] | [VRPRM: Process Reward Modeling via Visual Reasoning](https://arxiv.org/abs/2508.03556) | VRPRM提出了一种标签高效的两阶段训练策略，仅用5.5K个CoT标注和40K个强化学习样本就实现了更优的过程奖励建模，在VisualProcessBench上超越基于40万样本训练的VisualPRM-8B达7分，并在Best-of-N评估中为不同规模的多模态策略模型带来超过21分的性能提升。 |
| [^364] | [A Comprehensive Evaluation on Quantization Techniques for Large Language Models](https://arxiv.org/abs/2507.17417) | 本文对大语言模型的低比特训练后量化技术进行了最新、广泛且公平的综合评估与综述，并通过将量化方法解耦为多个步骤来深入分析现有方法之间的联系。 |
| [^365] | [Energy-Based Transfer for Reinforcement Learning](https://arxiv.org/abs/2506.16590) | 提出一种基于能量分数与分布外检测的迁移学习方法，让教师策略仅在其训练分布内的状态上进行干预引导，从而在单任务和多任务强化学习中显著提升样本效率与性能。 |
| [^366] | [From Proxies to Fields: Spatiotemporal Reconstruction of Global Radiation from Sparse Sensor Sequences](https://arxiv.org/abs/2506.12045) | 本文提出时间辐射算子网络（TRON），一种时空神经算子架构，能够仅从稀疏、非均匀的代理测量序列中实时重建连续的全球标量场，无需密集网格输入和未来观测数据。 |
| [^367] | [In-Context Bias Propagation in LLM-Based Tabular Data Generation](https://arxiv.org/abs/2506.09630) | 本文系统研究了基于LLM的合成表格数据生成中上下文示例偏差的传播机制，发现即使轻微的上下文偏差也会导致合成数据的全局统计失真，并揭示了恶意攻击者可通过注入有偏上下文示例来破坏下游模型公平性的对抗性风险。 |
| [^368] | [Quality-Diversity Red-Teaming: Automated Generation of High-Quality and Diverse Attackers for Large Language Models](https://arxiv.org/abs/2506.07121) | 本文提出质量多样性红队测试（QDRT）框架，通过自动生成高质量且多样化的攻击者，克服了以往红队测试中多样性度量过于简单化和单一攻击者模型覆盖不足的局限，实现对大语言模型更全面的安全评估。 |
| [^369] | [Understanding and Mitigating Under-Confidence in GNNs from the Final Layer](https://arxiv.org/abs/2505.11335) | 本文建立了一个统一的理论框架，揭示GNN的置信度不足由最后一层的类中心层面和节点层面校准共同决定，并从理论上证明减少最后一层参数的权重衰减可从类中心层面缓解置信度不足问题。 |
| [^370] | [Remote Rowhammer Attack using Adversarial Observations on Federated Learning Clients](https://arxiv.org/abs/2505.06335) | 该研究首次揭示了针对联邦学习服务器的新型远程Rowhammer攻击：攻击者利用强化学习智能体在客户端生成对抗性观测，引发服务器重复性的聚簇内存更新，从而在无需后门权限的情况下通过位翻转远程破坏服务器内存。 |
| [^371] | [Learning to Interfere in Non-Orthogonal Multiple-Access Joint Source-Channel Coding](https://arxiv.org/abs/2504.03690) | 提出一种基于多视图自编码器的机器学习方法，将压缩与信道编码融合，使多个发射机能够同时共享全部信道资源以实现非正交多址（NOMA）联合信源信道编码，接收端可从叠加信号中恢复所有图像并识别其对应的发射机。 |
| [^372] | [Quantifying Symptom Causality in Clinical Decision Making: An Exploration Using CausaLM](https://arxiv.org/abs/2503.19394) | 本研究利用CausaLM框架和TReATE方法量化了关键症状（如胸痛）对临床诊断模型预测结果的因果效应，超越了传统基于相关性的分析方法，为理解临床NLP模型的决策机制提供了新途径。 |
| [^373] | [Exploiting Concavity Information in Gaussian Process Contextual Bandit Optimization](https://arxiv.org/abs/2503.10836) | 本文提出一种将保持凹性的回归样条基与受凹性约束的高斯过程后验相结合的新型奖励模型，并据此构建UCB算法，从而在上下文老虎机优化中利用已知的凹性结构提升序贯决策的优化效率。 |
| [^374] | [AgiBot World Colosseo: A Large-scale Manipulation Platform for Scalable and Intelligent Embodied Systems](https://arxiv.org/abs/2503.06669) | AgiBot World是一个包含超过100万条轨迹、覆盖217个任务的大规模机器人操作数据平台，配合利用潜在动作表征的通用策略GO-1，实现了数量级的数据规模提升和平均30%的性能改进。 |
| [^375] | [Optimized Conformal Selection: Powerful Selective Inference After Data-Dependent Score Construction](https://arxiv.org/abs/2411.17983) | 该论文提出了优化共形选择（OptCS）通用框架，突破了现有方法要求评分函数独立于校准和测试数据的限制，在数据依赖的评分构建和大量数据复用的情况下仍能保证有限样本的错误发现率（FDR）控制。 |
| [^376] | [Bilateral Sharpness-Aware Minimization for Flatter Minima](https://arxiv.org/abs/2409.13173) | 本文提出双边锐度感知最小化（BSAM），通过融合最大锐度（MaxS）与最小锐度（MinS）构建更优的平坦度指标，克服了SAM仅关注梯度上升方向平坦度的“平坦度指标问题”，从而找到更平坦的极小值以提升神经网络泛化能力。 |
| [^377] | [Multi-Label Sentiment Analysis on 100 Languages with Dynamic Weighting for Label Imbalance](https://arxiv.org/abs/2008.11573) | 该论文提出了一种动态加权方法来应对多标签情感分析中的标签不平衡问题，并结合适配的多标签焦点损失与线性时间复杂度的最优阈值选择方法，在100种语言的情感分析任务中于9个指标中的7个上达到了最先进水平。 |
| [^378] | [The Limits of Post-Selection Generalization](https://arxiv.org/abs/1806.06100) | 本文揭示了事后泛化算法的能力极限，证明了任何满足事后泛化并回答自适应选择统计查询的算法其误差都存在紧的下界，为选择后数据分析的进展设置了一个根本性障碍。 |
| [^379] | [Supermodular Rank: Set Function Decomposition and Optimization.](http://arxiv.org/abs/2305.14632) | 本论文介绍了超模数秩的概念，并使用子模函数分解优化集合函数，提高了几个算法的逼近比保证。 |

# 详细

[^1]: LongZero：从零数据出发增强大语言模型的长上下文推理能力

    LongZero: Enhancing Long-Context Reasoning in LLMs from Zero Data

    [https://arxiv.org/abs/2610.30132](https://arxiv.org/abs/2610.30132)

    LongZero提出了一种无需任何外部数据的自举式强化学习方法，让大语言模型通过自行合成结构化的问题-答案-证据任务来生成训练数据，从而有效提升自身的长上下文推理能力。

    

    强化学习（RL）是提升大语言模型（LLM）长上下文推理能力的有效途径，但它需要长篇、多样、具有挑战性且可验证的训练数据。人工标注此类数据成本高昂，自然存在的数据又过于稀缺而无法支撑强化学习，而现有的数据合成方法依赖外部语料库或更强的教师模型，随着被训练模型能力的提升，这些依赖越来越难以获得。最近，从零数据出发的强化学习——即模型从自身提出的任务中学习——取得了出色的效果，但现有方法仅针对短上下文任务。因此，我们提出了LongZero，一种让LLM在无需任何外部数据的情况下提升自身长上下文推理能力的方法。LongZero不是在现有文档上编写问题和答案，而是首先让模型以紧凑的结构化形式合成问题、答案及其支持证据，并……

    arXiv:2610.30132v1 Announce Type: new  Abstract: Reinforcement learning (RL) is an effective way to strengthen the long-context reasoning of large language models (LLMs), but it needs training data that are long, diverse, challenging, and verifiable. Human annotation of such data is expensive, naturally occurring data are too scarce to sustain RL, and existing synthesis methods depend on external corpora or stronger teacher models, which become harder to obtain as the trained models get stronger. Recently, RL from zero data, in which a model learns from tasks it proposes itself, has achieved excellent performance, but existing methods target short-context tasks. Therefore, we propose $\textbf{LongZero}$, a method that lets an LLM improve its own long-context reasoning without any external data. Instead of writing questions and answers over existing documents, LongZero first has the model synthesize the question and answer with its supporting evidence in a compact structured form, and o
    
[^2]: 在关键之处深入思考：面向序列推荐的物品级自适应递归

    Think Harder Where It Matters: Item-Level Adaptive Recursion for Sequential Recommendation

    [https://arxiv.org/abs/2610.30128](https://arxiv.org/abs/2610.30128)

    提出LoopRec，一种针对序列推荐的物品级自适应递归方法，根据每个物品的实际计算需求动态分配递归深度，取代固定计算预算，从而既节省计算资源又提升整体推荐质量。

    

    现有的序列推荐模型通常为用户交互历史中的每个物品分配固定的计算预算，而不考虑该物品对最终预测的贡献。我们通过两项诊断性发现对这一事实上的设计提出挑战。（1）计算需求的不均匀性：同一序列中不同物品所需的计算量差异极大，大多数物品经过浅层处理即可收敛，而少数关键物品只有在完整深度下才能稳定。（2）固定分配的次优性：即使能够理想地获取目标物品信息，非均匀的深度分配也始终优于固定分配，这表明现有的均匀计算既在已收敛的物品上浪费了计算能力，又牺牲了整体推荐质量。这些观察促使计算方式从固定深度转向物品级自适应推理。为此，我们提出了LoopRec。LoopRec递归地应用一个权重共享的单一模块……（原文摘要在此处被截断）

    arXiv:2610.30128v1 Announce Type: cross  Abstract: Existing sequential recommenders usually allocate a fixed computational budget to every item in a user's interaction history, regardless of the item's contribution to the final prediction. We challenge this de facto design with two diagnostic findings. (1) Non-uniform computational need: items within the same sequence vary drastically in how much computation they require. Most converge after shallow processing while a critical minority only stabilizes at full depth. (2) Suboptimality of fixed allocation: given oracle access to the target item, a non-uniform depth assignment consistently outranks the fixed one, indicating that existing uniform computation both wastes capacity on already-converged items and sacrifices overall recommendation quality. These observations motivate a shift from fixed-depth computation to item-level adaptive reasoning. To this end, we propose LoopRec. LoopRec recursively applies a single weight-shared block an
    
[^3]: 关于 $\ell_\infty$ 收缩映射的拟多项式算法

    A Quasipolynomial Algorithm for $\ell_\infty$ Contractions

    [https://arxiv.org/abs/2610.30122](https://arxiv.org/abs/2610.30122)

    本文提出了一种在拟多项式时间内计算 $\ell_\infty$ 范数下非扩张映射的近似不动点的算法，由此推出简单随机博弈可以在拟多项式时间内求解。

    

    我们给出了一种算法，可以在 $\log(1/\eps)^{O(\log^2 d)}$ 的时间内计算任意关于 $\ell_\infty$ 范数的 $d$ 维非扩张映射的 $\eps$-近似不动点。我们的方法核心在于使用一种提升论证来几何级地缩小搜索区域。作为诸多推论之一，我们的算法意味着简单随机博弈可以在拟多项式时间内求解。

    arXiv:2610.30122v1 Announce Type: cross  Abstract: We give an algorithm for computing an $\eps$-approximate fixed point of any $d$-dimensional nonexpansion with respect to the $\ell_\infty$ norm in time $\log(1/\eps)^{O(\log^2 d)}$. At its core, our method uses a boosting argument to geometrically shrink a search region. Among other consequences, our algorithm implies that simple stochastic games can be solved in quasipolynomial time.
    
[^4]: 再生过程的自归一化Bernstein不等式及其在无独立性学习中的应用

    Self-normalized Bernstein Inequalities for Regenerative Processes and Applications to Learning Without Independence

    [https://arxiv.org/abs/2610.30119](https://arxiv.org/abs/2610.30119)

    该论文为具有独立同分布标记周期的再生过程证明了自归一化Bernstein不等式，其中与再生时间尾部相关的项在多数情形下仅为二阶小量，并据此为非独立同分布的再生数据分类建立了泛化误差界，进而提出满足预言机不等式的不可知分类器。

    

    我们针对具有独立同分布、平均周期长度有限的标记周期的再生过程的有界函数，证明了一个自归一化的Bernstein不等式。该界利用了观测到的周期长度，并对截断以及观测窗口结束后才终止的周期保留了显式项。通过从一个有限的确定性集合中选取调节参数，我们得到了一种Bernstein不等式的版本，其中除一项与再生时间的尾部相关外，其余所有项均为经验项。随后我们在多种情形下研究这一项：结果表明，在许多令人关注的情形中，它是二阶小量。这对机器学习中的应用具有重要意义：我们针对使用非独立同分布的再生数据的分类问题证明了泛化误差界。这些界可用于定义学习目标：通过仅最小化界中的一阶项，我们定义了仍然满足预言机（oracle）不等式的不可知分类器。我们还表明……（摘要在此处截断）

    arXiv:2610.30119v1 Announce Type: new  Abstract: We prove a self-normalized Bernstein inequality for bounded functions of a regenerative process with independent, identically distributed marked cycles of finite mean length. The bound uses observed cycle lengths and retains explicit terms for truncation and the cycle ending after the observation horizon. Selecting a tuning parameter from a finite deterministic collection gives a version of Bernstein inequality where all terms are empirical but one, related to the tails of the regeneration times. We then study this term in many regimes: it turns out that it is of second-order in many cases of interest. This is important for applications in machine learning: we prove generalization error bounds for classification with non-i.i.d., regenerative data. These bounds can be used to define learning objectives: by minimizing the first-order term only in the bound, we define agnostic classifiers that still satisfy oracle inequalities. We also show
    
[^5]: 局部性与权重共享在卷积神经网络中的作用

    The role of locality and weight sharing in convolutional neural networks

    [https://arxiv.org/abs/2610.30118](https://arxiv.org/abs/2610.30118)

    本研究通过师生模型的精确动力学分析揭示：卷积网络中权重共享带来的学习加速与输入位置数 $P$ 成正比，而局部性则决定样本复杂度，二者的影响均依赖于输入位置间的空间相关性。

    

    卷积神经网络的成功通常被归因于局部性和权重共享。然而，这两个特性如何分别塑造学习过程，以及它们的影响如何依赖于自然图像的空间相关性，仍然知之甚少。我们在一个师生模型中研究这些问题，其中一个卷积教师作用于 $P$ 个输入位置，位置之间的相关性为 $c$。我们比较了通过在线随机梯度下降训练的卷积学生与局部连接学生，并推导出在大特征维度 $D$ 极限下能精确描述其动力学的常微分方程。对于不相关的位置（$c = 0$），两个学生遵循相同的学习轨迹，而卷积学生的学习速度恰好快 $P$ 倍。我们还发现局部性会影响样本复杂度：与教师核大小相匹配的学生能够从 $\mathcal{O}_{D}(D)$ 个样本中恢复教师……

    arXiv:2610.30118v1 Announce Type: new  Abstract: The success of convolutional neural networks is often attributed to locality and weight sharing. However, how these two properties separately shape learning, and how their effects depend on the spatial correlations of natural images, remain poorly understood. We address these questions in a teacher-student model with a convolutional teacher acting on $P$ input positions with correlation $c$ between positions. We compare convolutional and locally connected students trained by online stochastic gradient descent, and derive ordinary differential equations that describe their dynamics exactly in the limit of large feature dimension $D$. For uncorrelated positions ($c = 0$), the two students follow the same learning trajectory, with the convolutional student learning exactly $P$ times faster. We also find that locality affects the sample complexity: a student that matches the kernel size of the teacher recovers it from $\mathcal{O}_{D}(D)$ sa
    
[^6]: 高斯混合模型中NPMLE在Hellinger距离下的极小极大最优性

    Minimax Optimality of the NPMLE for Gaussian Mixtures in Hellinger Distance

    [https://arxiv.org/abs/2610.30109](https://arxiv.org/abs/2610.30109)

    本文证明了基于n个样本的NPMLE在真实混合测度为次高斯时达到平方Hellinger距离下的极小极大最优速率log(n)/n，并在紧支撑情形下达到更快的log(n)/(n log log n)速率，其证明依赖于对有限高斯混合类局部Hellinger括号熵的精确刻画。

    

    我们研究了基于n个独立同分布样本、使用非参数极大似然估计量（NPMLE）来估计一元高斯混合密度的问题。我们证明了当真实的混合测度为次高斯时，无约束的NPMLE在平方Hellinger距离下达到了极小极大最优速率log(n)/n。此外，我们还证明了当真实混合测度具有紧支撑时，约束在紧区间上的NPMLE同样达到极小极大最优速率，且该速率略快，为log(n)/(n log log n)阶。我们的证明关键在于精确刻画有限高斯混合类上的局部Hellinger括号熵。我们的一个关键技术创新是对有界次数多项式球的L²括号熵给出了精确上界，这可能具有独立的研究价值。作为应用，我们建立了g-建模方法的遗憾最优性

    arXiv:2610.30109v1 Announce Type: cross  Abstract: We analyze the problem of estimating univariate Gaussian mixture densities using the nonparametric maximum likelihood estimator (NPMLE) based on $n$ independent and identically distributed samples. We show that the unconstrained NPMLE achieves the minimax optimal rate $\log(n)/n$ in squared Hellinger distance, when the true mixing measure is sub-Gaussian. In addition, we show that when the true mixing measure has compact support, the NMPLE constrained to a compact interval also achieves the minimax optimal rate, which is slightly faster and of order $\log(n)/(n \log \log n)$. Our proof consists of sharply characterizing the local Hellinger bracketing entropy of the class of finite Gaussian mixtures. One of our key technical innovations is to sharply bound the $L^2$ bracketing entropy of balls of bounded-degree polynomials, which may be of independent interest. As an application, we establish the regret-optimality of $g$-modeling proced
    
[^7]: 速度模糊剖面：面向流匹配的时间分辨贝叶斯风险诊断

    Velocity Ambiguity Profiles: Time-Resolved Bayes-Risk Diagnostics for Flow Matching

    [https://arxiv.org/abs/2610.30107](https://arxiv.org/abs/2610.30107)

    该论文提出速度模糊剖面（VAP）这一新诊断工具，将流匹配的时间分辨损失分解为不可约的贝叶斯风险下界与可通过数据或改进估计器降低的超额项，并在两个结构化基准中证明了其统计收敛性质，从而揭示了流匹配在哪些时间区域受内在模糊性限制、哪些区域仍可由数据改进。

    

    流匹配训练的是一系列连续的速度回归问题，但通常报告的是对插值时间取平均的损失，这掩盖了哪些区域受内在模糊性限制、哪些区域仍然受数据量限制。我们引入速度模糊剖面（VAP），即 $\mathcal{A}_v(t)$——速度预测在时间局部的贝叶斯风险下界，并将时间分辨的损失分解为 $L_N(t)=\mathcal{A}_v(t)+R(t;N)$，其中只有超额项 $R(t;N)$ 可以通过更多数据或不同的估计器来降低。对于 $N$ 个训练样本以及结构化示例中的 $K$ 分量混合分布，我们证明了两个统计基准。在一个内在的局部回归基准中，$\mathcal{A}_v(t)$ 进入标准非参数收敛率中随时间变化的噪声因子；在分离的各向同性高斯混合分布中，贝叶斯速度变为局部仿射的，且标签神谕（label-oracle）残差回归基准具有参数化的超额风险规模 $K\mathcal{A}_v(t)/N$，与……（原文摘要在此处截断）

    arXiv:2610.30107v1 Announce Type: new  Abstract: Flow matching trains a continuum of velocity regression problems but usually reports a loss averaged over interpolation time, obscuring which regions are limited by intrinsic ambiguity and which remain data-limited. We introduce the Velocity Ambiguity Profile (VAP), $\mathcal{A}_v(t)$, the time-local Bayes-risk floor of velocity prediction, and decompose time-resolved loss as $L_N(t)=\mathcal{A}_v(t)+R(t;N)$, where only the excess term $R(t;N)$ is reducible by more data or a different estimator. For $N$ training samples and a $K$-component mixture in the structured example, we prove two statistical benchmarks. In an intrinsic local-regression benchmark, $\mathcal{A}_v(t)$ enters the time-dependent noise factor of the standard nonparametric rate. In separated isotropic Gaussian mixtures, the Bayes velocity becomes locally affine, and a label-oracle residual-regression benchmark has parametric excess-risk scale $K\mathcal{A}_v(t)/N$, match
    
[^8]: 非对称在策略蒸馏学习情境化探索

    Asymmetric On-Policy Distillation Learns In-Context Exploration

    [https://arxiv.org/abs/2610.30105](https://arxiv.org/abs/2610.30105)

    提出ASTEROID学生-教师蒸馏框架，让拥有特权信息的教师基于学生在部分可观测环境中自主收集的在策略历史上下文提供动作标签，从而使学生学会测试时的情境化探索行为。

    

    在部分可观测条件下的序贯决策十分困难，因为智能体必须先通过探索来推断隐藏的、与任务相关的信息，然后才能有效地行动。通过在线强化学习从零开始学习这种信息收集行为的样本效率很低，而离线模仿学习则会在不同的隐藏状态之间平均专家动作，从而无法学到探索行为。在这项工作中，我们提出了 ASTEROID，这是一个简单而有效的学生-教师监督学习框架，能够学习测试时的探索行为。尽管拥有特权信息的教师从不需要探索，我们证明了基于历史条件的学生模型在与策略上下文一起训练时能够学会探索：学生首先在部分可观测条件下收集在策略上下文，特权教师则基于该上下文提供动作标签。通过直接在学生的在策略历史条件下蒸馏专家动作，学生能够学会系统地收集与任务相关的信息。

    arXiv:2610.30105v1 Announce Type: cross  Abstract: Sequential decision-making under partial observability is difficult because agents must explore to infer hidden, task-relevant information before acting effectively. Learning such information-gathering behaviors tabula rasa via online reinforcement learning is sample-inefficient, whereas offline imitation averages expert actions across hidden states and fails to learn exploration. In this work, we propose ASTEROID, a simple and effective student--teacher supervised learning framework that can learn test-time exploration. Although a privileged expert never needs to explore, we demonstrate that a history-conditioned student can learn to do so when trained with on-policy contexts: the student first collects on-policy context under partial observability, and the privileged expert provides action labels conditioned on that context. By distilling expert actions directly conditioned on the student's on-policy history, the student learns to sy
    
[^9]: BatchBench：针对生物医学图像中批处理效应与领域偏移的AI鲁棒性基准测试——一个JUMP细胞绘画排行榜

    BatchBench: Benchmarking AI Robustness to Batch Effects and Domain Shift in Biomedical Images - A JUMP Cell Painting Leaderboard

    [https://arxiv.org/abs/2610.30104](https://arxiv.org/abs/2610.30104)

    该论文推出了BatchBench排行榜，基于大型公开生物医学成像数据集JUMP-CP，评估和比较AI系统在批处理效应和领域偏移下的鲁棒性。

    

    尽管近年来取得了诸多进展，当前的AI系统仍然极易受到领域偏移的影响，尤其是在生物医学成像领域，因为图像通常是在差异极大的实验条件下采集的。尽管近期的领域自适应方法已取得实质性进展，但在领域偏移情况下仍存在显著的性能差距。为了衡量这一方面的进展，我们推出了BatchBench，这是一个评估和比较AI系统在来自未见过的实验批次和数据来源的生物医学图像上鲁棒性的排行榜。BatchBench基于JUMP-CP构建，这是最大的公开可用生物医学成像数据集之一。JUMP-CP使用Cell Painting实验方法生成，表现出明显的批处理效应和领域偏移，使其成为评估分布变化鲁棒性的一个具有挑战性的测试平台。

    arXiv:2610.30104v1 Announce Type: new  Abstract: Despite recent advances, current AI systems remain highly susceptible to domain shifts, particularly in biomedical imaging, where images are usually acquired under substantially different experimental conditions. Although recent domain adaptation methods have made substantial progress, significant performance gaps under domain shift persist. To measure progress in this regard, we introduce BatchBench, a leaderboard that evaluates and compares the robustness of AI systems on biomedical images from unseen experimental batches and data sources. BatchBench is built on JUMP-CP, one of the largest publicly available biomedical imaging datasets. Generated using the Cell Painting assay, JUMP-CP exhibits pronounced batch effects and domain shifts, making it a challenging testbed for evaluating robustness to distributional changes.
    
[^10]: 基于偏好反馈的差分隐私稀疏奖励估计

    Differentially Private Sparse Reward Estimation with Preference Feedback

    [https://arxiv.org/abs/2610.30099](https://arxiv.org/abs/2610.30099)

    该论文首次在Bradley-Terry-Luce模型下为基于成对偏好反馈的差分隐私稀疏奖励估计建立了全面的理论框架，突破了现有隐私对齐方法误差界随模型维度多项式增长的瓶颈，使隐私保护的RLHF在过参数化大模型时代变得可行。

    

    基于人类反馈的强化学习已成为使大语言模型与人类价值观对齐的核心范式。然而，对敏感偏好数据的依赖要求提供严格的差分隐私（DP）保证，以防止用户信息泄露。现有的隐私对齐方法通常假设稠密的奖励参数化，导致估计误差界随环境模型维度 $d$ 呈多项式增长。这种依赖在现代过参数化模型中是难以承受的，因为 $d$ 非常庞大。在本工作中，我们研究了隐私稀疏RLHF问题以解决这一局限，假设人类偏好（奖励模型）由一个大小为 $s^* \ll d$ 的相关特征子集所主导。我们在Bradley-Terry-Luce模型下，提供了从成对偏好反馈中进行差分隐私稀疏奖励估计的首个全面理论框架，并引入了高效算法

    arXiv:2610.30099v1 Announce Type: new  Abstract: Reinforcement Learning with Human Feedback (RLHF) has become a central paradigm for aligning Large Language Models with human values. However, the reliance on sensitive preference data necessitates rigorous differential privacy (DP) guarantees to prevent user information leakage. Existing private alignment methods typically assume dense reward parameterizations, resulting in estimation error bounds that scale polynomially with the ambient {model size $d$}. Such dependence is prohibitive in modern {overparameterized models} where $d$ is vast. In this work, we study \emph{Private Sparse RLHF} problem to address this limitation, assuming that human preferences (reward model) are governed by a subset of $s^* \ll d$ relevant features. We provide the first comprehensive theoretical framework for differentially private sparse reward estimation from pairwise preference feedback under the Bradley-Terry-Luce model, and introduce efficient algorith
    
[^11]: VX-One：面向实时边界分流的有类型、经校准的决策头——关于未知攻击诚实性、级联灰色地带与漂移下校准的实证研究

    VX-One: Typed, Calibrated Decision Heads for Real-Time Perimeter Triage - An empirical study of unknown-attack honesty, cascade gray zones, and calibration under drift

    [https://arxiv.org/abs/2610.30090](https://arxiv.org/abs/2610.30090)

    本文提出VX-One，一个将商用有类型决策原语（校准门控、含显式未知选项的闭集选择、序数分数）作为输出的小型多头网络，在CPU上以毫秒级延迟实现实时边界分流，并实证评估了未知攻击诚实性、级联灰色地带与漂移下的校准性能。

    

    边界分流面临延迟与语义之间的权衡：签名引擎快速且格式安全，但对未见过的攻击无能为力，而生成式模型虽具备语义理解能力却速度缓慢且会产生无法解析的输出。2026年末，一类商用System One决策模型（TypeSafe Jev；Microsoft-Decision-1）开始将有类型的决策原语——二值门控、闭集选择、序数分数——作为产品API出售，但学术界尚无相关研究。我们提出VX-One：一个小型多头网络（参数量10万至1000万），其输出即为这些原语——一个经校准的门控、一个在始终包含显式未知选项的封闭候选集上的选择，以及一个序数分数，采用多任务方式训练，并对每个头进行温度校准（CPU上p50延迟约为0.7-1.7毫秒）。我们在UNSW-NB15、CIC-IDS2017、EMBER2024数据集上将其与经校准的XGBoost、随机森林、MLP、Kitsune进行对比评估，并测量了一个两阶段级联系统（配备52,674条Emerging-Threats规则的Suricata -> VX-One）……

    arXiv:2610.30090v1 Announce Type: cross  Abstract: Perimeter triage faces a latency/semantics trade-off: signature engines are fast and format-safe but blind to unseen attacks, while generative models are semantic but slow and produce unparseable output. In late 2026 a class of commercial System One decision models (TypeSafe Jev; Microsoft-Decision-1) began selling typed decision primitives - binary gates, closed-set choices, ordinal scores - as product APIs, with no academic treatment. We present VX-One: a small multi-head network (100K-10M parameters) whose outputs are these primitives - a calibrated gate, a choice over a closed candidate set that always includes an explicit unknown option, and an ordinal score, trained multi-task, temperature-calibrated per head (~0.7-1.7 ms p50 on CPU). We evaluate it on UNSW-NB15, CIC-IDS2017, EMBER2024 against calibrated XGBoost, Random-Forest, MLP, Kitsune, and measure a two-phase cascade (Suricata with 52,674 Emerging-Threats rules -> VX-One) o
    
[^12]: Orbitol：基于Wyckoff轨道的晶体生成

    Orbitol: Crystal Generation on Wyckoff Orbits

    [https://arxiv.org/abs/2610.30086](https://arxiv.org/abs/2610.30086)

    Orbitol通过用Wyckoff轨道标记替代原子标记，实现了对空间群对称性的直接控制，能生成保持所要求空间群的稳定新颖晶体结构，且生成速度比现有对称性感知基线快两到三个数量级。

    

    生成模型通过学习组分和几何结构来提出晶体材料，但无条件采样无法直接控制空间群，这限制了诸如压电性等响应是否在对称性上被允许。我们提出了Orbitol，一个构建于Crystalite扩散Transformer之上的晶体生成器，它用Wyckoff轨道标记取代了原子标记。给定一个空间群和一组被占据的Wyckoff位置，Orbitol生成每个轨道的原子类型和自由坐标，以及与之兼容的晶格，因此每个有效结构在所请求空间群的对称操作下都保持不变。在所评估的设置下，其生成速度比对称性感知的基线方法快两到三个数量级。在MP-20数据集上，Orbitol有13.8%的尝试产生了稳定或亚稳、独特且新颖的结构，并在弛豫后保持了所请求的空间群，而次优方法SGFM仅为10.8%。

    arXiv:2610.30086v1 Announce Type: new  Abstract: Generative models propose crystalline materials by learning composition and geometry, but unconditional sampling gives no direct control over the space group, which constrains whether responses such as piezoelectricity are symmetry-allowed. We present Orbitol, a crystal generator built on the Crystalite diffusion Transformer that replaces atom tokens with Wyckoff-orbit tokens. Given a space group and a collection of occupied Wyckoff positions, Orbitol generates the atom type and free coordinates of each orbit together with a compatible lattice, so every valid structure is invariant under the operations of the requested group. It generates structures two to three orders of magnitude faster than the symmetry-aware baselines at the evaluated settings. On MP-20, 13.8% of Orbitol's attempts yielded stable or metastable, unique and novel structures that retained the requested space group after relaxation, compared with 10.8% for SGFM, the next
    
[^13]: LLM分类器电路

    Circuits of LLM Classifiers

    [https://arxiv.org/abs/2610.30076](https://arxiv.org/abs/2610.30076)

    该论文将LLM分类器视为对目标标签的含噪测量仪器，提出将其组织成自适应的查询与决策“电路”，刻画了最优电路结构及准确率的理论极限，并在七个分类任务上改进了成本-准确率前沿。

    

    通过简单地用待分类样本对语言模型进行提示，就能轻松构建分类器。然而，这些模型的预测既不是确定性的，也不是在所有样本上都同样可靠。我们如何组合来自不同模型的预测以提升可靠性？本文将LLM分类器视为对潜在目标标签的含噪测量仪器，目标是在给定平均查询成本下最大化平均分类准确率。为此，我们引入了“电路”的概念，将LLM分类器组织成自适应的查询与决策序列。随后我们刻画了最优电路，获得了可实现准确率的基本理论极限，并提供了实现最优电路的实用见解。在涵盖安全、健康、政治学、人类偏好、智能体评估、决策与推理的七个分类任务上，我们的方法在成本-准确率前沿上优于路由和级联基线。

    arXiv:2610.30076v1 Announce Type: new  Abstract: Language models make it easy to build classifiers by simply prompting a model with the sample to classify. However, their predictions are neither deterministic nor uniformly reliable across samples. How can we combine predictions from different models to improve reliability? We view LLM classifiers as noisy measurement instruments of an underlying target label. Our goal is to maximize average classification accuracy at a given average query cost. To this end, we introduce circuits that organize LLM classifiers into an adaptive sequence of queries and decisions. We then characterize the optimal circuit, obtain the fundamental limits of achievable accuracy, and provide practical insights on realizing the optimal circuit. Across seven classification tasks spanning safety, health, political science, human preference, agent evaluation, decision-making and reasoning, our method improves the cost-accuracy frontier over routing and cascading bas
    
[^14]: 从分类器头到编码器原型：基于分类器头聚类的个性化联邦学习

    From Classifier Heads to Encoder Prototypes: Personalized Federated Learning with Head-Derived Clustering

    [https://arxiv.org/abs/2610.30075](https://arxiv.org/abs/2610.30075)

    HDC-FL利用联邦学习训练中已有的客户端分类器头进行聚类分组，为每组构建编码器原型并以软混合形式返回给客户端，无需平均分类器头、额外优化或微调即可实现个性化表示共享。

    

    个性化联邦学习必须决定哪些客户端应当共享表示。我们提出了HDC-FL（基于分类器头聚类的个性化联邦学习），该方法从共享编码器训练已经产生的信号——客户端特有的分类器头——中读取这一决策。在周期性的刷新轮次中，服务器将缓存的分类器头投影到归一化的低秩描述符空间中，对描述符进行聚类分组，从缓存的客户端编码器中为每组构建一个编码器原型，并向每个客户端返回这些原型的软混合，同时保留其自身不变的分类器头。客户端特有的分类器头从不被平均或分发给其他客户端；共享编码器的训练轨迹保持不变；并且无需引入第二条优化轨迹、额外的本地目标函数或训练后的微调阶段。在涵盖标签子集、Dirichlet分布和旋转异质性的11个CIFAR-10和CIFAR-100配置上，使用ResNet……（原文摘要在此处截断）

    arXiv:2610.30075v1 Announce Type: new  Abstract: Personalized federated learning must decide which clients should share representations. We introduce HDC-FL (Head-Derived Clustering for Personalized Federated Learning), which reads that decision off a signal shared-encoder training already produces: the client-specific classifi- er heads. At periodic refresh rounds the server projects the cached heads into a normalized low-rank descriptor space, groups the descriptors, builds one encoder prototype per group from the cached client encoders, and returns to each client a soft mixture of those prototypes paired with its own unchanged head. Client-specific heads are never averaged or distributed to oth- er clients; the shared-encoder trajectory is left intact; and no second optimization trajectory, additional local objective, or post-training fine-tuning stage is required. Across 11 CIFAR-10 and CIFAR-100 configurations covering label-subset, Dirichlet and rotation heterogeneity with ResNet
    
[^15]: 一路"Loopholing"到底：基于纯自条件化的语言建模

    Loopholing All the Way: Language Modeling with pure Self-Conditioning

    [https://arxiv.org/abs/2610.30068](https://arxiv.org/abs/2610.30068)

    提出LAW方法，将自条件化从辅助技巧提升为连续生成模型的核心机制，通过在去噪步骤间传播期望嵌入并在最后一次性解码，在数独、迷宫和GSM8K等任务上超越了现有的离散与连续扩散语言模型。

    

    近期的扩散语言模型（DLMs）使用自条件化（Self-Conditioning, SC），即Loopholing，来提升性能。虽然SC传统上被视为一种辅助性启发式技巧，但其实证上的成功引出了一个问题：能否构建一个完全基于自条件化的连续生成模型？我们给出了肯定的回答，提出了LAW（"Loopholing All the Way"，即"一路Loopholing到底"），该方法将去噪器预测下的期望嵌入在多个步骤之间进行传播，并在最后一次性解码所有token。在数独（Sudoku）任务上，LAW优于以往包括Loopholing离散扩散模型在内的离散和连续扩散模型。在迷宫（Maze）任务上，LAW在2到64次函数评估（NFE）范围内准确率最高。在TinyGSM上训练并以温度1、1024次NFE采样后，LAW解决了42.2%的GSM8K问题，而均匀Loopholing离散扩散模型仅为34.5%。此外，使用LAW微调预训练的均匀扩散模型（Duo）可将其准确率从17.2%提升至48.1%。最后，在OpenWebTe

    arXiv:2610.30068v1 Announce Type: new  Abstract: Recent Diffusion Language Models (DLMs) use Self-Conditioning (SC), or Loopholing, to improve their performance. While SC is traditionally seen as an auxiliary heuristic, its empirical success raises the question: can a continuous generative model be based purely on Self-Conditioning? We answer affirmatively with LAW (``Loopholing All the Way''), which propagates the Expected Embedding under the denoiser's prediction across steps, decoding the tokens once, at the end. On Sudoku, LAW outperforms previous Discrete and Continuous Diffusion Models, including Loopholing Discrete Diffusion. On Maze, LAW is the most accurate from 2 to 64 NFE. After training on TinyGSM and sampling at temperature 1 with 1024 NFE, LAW solves 42.2% of GSM8K problems against 34.5% for uniform Loopholing Discrete Diffusion. Additionally, fine-tuning a pre-trained Uniform Diffusion Model (Duo) with LAW increases its accuracy from 17.2% to 48.1%. Finally, on OpenWebTe
    
[^16]: 循环均匀扩散语言模型

    Looped Uniform Diffusion Language Models

    [https://arxiv.org/abs/2610.30066](https://arxiv.org/abs/2610.30066)

    该论文首次将循环机制（在每步去噪中递归复用早期Transformer层）引入均匀扩散语言模型，在不改变训练目标和采样器的情况下显著提升训练效率，使循环UDLM以更少计算量即可达到非循环模型训练约0.5万亿token后的验证NLL水平。

    

    均匀扩散语言模型（UDLM）通过迭代去噪普通词表token序列来生成文本，其中任意token都可能被破坏并随后被修订。在训练过程中，UDLM必须在可能不可靠的上下文上进行去噪，而掩码扩散语言模型（MDLM）始终以干净的上下文为条件。这种具有挑战性的训练目标使得UDLM的训练比MDLM更耗费FLOP，但能带来更强的下游推理性能。最近，选择性循环——即在每一步去噪中递归应用若干早期Transformer层——已被证明可以在等FLOP预算下提升MDLM的训练效率。我们在保持训练目标和采样器不变的情况下，研究了UDLM中的循环机制。我们发现循环能显著提升训练效率：循环UDLM达到了在约0.5万亿token上训练的非循环UDLM的验证NLL水平……

    arXiv:2610.30066v1 Announce Type: new  Abstract: Uniform diffusion language models (UDLMs) generate text by iteratively denoising sequences of ordinary vocabulary tokens, any of which may be corrupted and subsequently revised. During training, UDLMs must denoise from a context that may itself be unreliable, whereas masked diffusion language models (MDLMs) always condition on a clean context. This challenging training objective makes UDLMs more FLOP-intensive to train than MDLMs, but it yields stronger downstream reasoning performance. Recently, selective looping, which recursively applies a few early Transformer layers within each denoising step, has been shown to improve training efficiency in MDLMs under an iso-FLOP budget. We investigate looping in UDLMs while keeping the training objective and sampler unchanged. We find that looping significantly improves training efficiency: looped UDLMs reach the validation NLL of a non-looped UDLM trained on about 0.5 trillion tokens with up to 
    
[^17]: HGM-Dual：面向长程规划的谱系感知引导式智能体框架优化

    HGM-Dual: Lineage-Aware Guided Agent Harness Optimization for Long-Horizon Planning

    [https://arxiv.org/abs/2610.30042](https://arxiv.org/abs/2610.30042)

    提出了HGM-Dual框架，通过谱系感知上下文和诊断引导，实现面向规划智能体的预算高效框架演化，解决了开放式自我改进在长程规划任务中反复探索无效改动、消耗大量评估预算的问题。

    

    框架优化通过改进智能体的提示词、工具和工作流，是增强基于冻结或闭源大语言模型构建的智能体的一种日益有效的方法。以达尔文-哥德尔机和赫胥黎-哥德尔机为代表的前沿方法依赖于递归自我改进，且主要针对编程任务开发——在编程任务中，改进智能体所需的能力与被改进的能力恰好一致。而规划任务打破了这种一致性：智能体必须生成一系列共同满足相互制约约束的工具调用。在这种情况下，开放式自我修改会反复探索无效的改动，并消耗大量评估预算。我们提出了HGM-Dual，一个面向规划智能体的诊断引导、预算高效的框架演化方法。首先，HGM-Dual将智能体演化表示为谱系感知的上下文。该上下文记录了每个祖先做了什么改动、哪些行为得到了改进……（摘要在此处被截断）

    arXiv:2610.30042v1 Announce Type: cross  Abstract: Harness optimization, which improves an agent's prompts, tools, and workflows, is an increasingly effective way to enhance agents built on frozen or closed-source LLMs. Leading approaches such as the Darwin--Godel Machine and the Huxley--Godel Machine (HGM) rely on recursive self-improvement and were developed mainly for coding, where the competence needed to improve the agent coincides with the competence being improved. Planning tasks, in which an agent must produce a sequence of tool calls that jointly satisfies interacting constraints, break this alignment. There, open-ended self-modification repeatedly explores ineffective changes and consumes substantial evaluation budgets. We introduce HGM-Dual, a diagnostic-guided framework for budget-efficient harness evolution for planning agents. First, HGM-Dual represents agent evolution as lineage-aware context. This context records what each ancestor changed, which behaviors improved or r
    
[^18]: ExGrad：超越抑制——对抗训练中的显式干净梯度减除

    ExGrad: Beyond Suppression - Explicit Clean-Gradient Subtraction in Adversarial Training

    [https://arxiv.org/abs/2610.30018](https://arxiv.org/abs/2610.30018)

    ExGrad 通过显式构建干净、对抗与 AWP 三类梯度并用全局可解的三变量优化进行重组，证明在对抗训练中对干净梯度进行适度减除（而非单纯抑制或置零）能够显著提升 PGD-20 鲁棒性。

    

    对抗训练从代理损失中导出更新，但源损失的下降并不能直接刻画一个更新对对抗裕度的影响。我们提出 ExGrad（外推梯度重组），该方法构建干净输入梯度、对抗输入梯度和对抗权重扰动（AWP）梯度三类梯度。通过虚拟步骤测量它们对裕度的响应，并以一个全局可解的三变量问题在受限仿射域上选取系数，且不施加非负性约束。在 CIFAR-10 的不同仿射半径下，负的干净系数迅速占据主导地位，且其幅值随半径增大而增加。跨零干预实验表明，仅仅将干净系数设为零是不够的：轻度减除即可提升 PGD-20 鲁棒性，而更强的减除则以牺牲干净准确率为代价换取 PGD-20 的进一步提升。在 CIFAR-10 的 WRN-28-4 对照实验中，三种候选梯度移除变体均无法匹敌完整的 ExGrad。投影诊断显示……（原文摘要在此处截断）

    arXiv:2610.30018v1 Announce Type: new  Abstract: Adversarial training derives updates from surrogate losses, but source-loss descent does not directly characterize an update's effect on adversarial margins. We introduce ExGrad (extrapolative gradient recomposition), which constructs clean-input, adversarial-input, and adversarial-weight-perturbed (AWP) gradients. Virtual steps measure their margin responses, and a globally solvable three-variable problem selects coefficients over a bounded affine domain without nonnegativity constraints. Across affine radii on CIFAR-10, negative clean coefficients rapidly predominate and increase in magnitude with the radius. Cross-zero interventions show that setting the clean coefficient to zero is insufficient: mild subtraction improves PGD-20, while stronger subtraction trades clean accuracy for further PGD-20 gains. In CIFAR-10 WRN-28-4 controls, none of the three candidate-removal variants matches full ExGrad. Projection diagnostics show that neg
    
[^19]: Code Comprehensor：面向大型代码仓库的层次化文档与跨模态检索

    Code Comprehensor: Hierarchical Documentation and Cross-Modal Retrieval for Large Repositories

    [https://arxiv.org/abs/2610.30016](https://arxiv.org/abs/2610.30016)

    该论文提出Code Comprehensor混合检索-推理框架，通过代码依赖调用图、层次化文档生成（CodeScribe）和CLIP式跨模态检索模型（CodeContrast），实现对大型代码库相关代码上下文的快速语义精准检索，并为基于MCP的LLM问答提供可靠依据。

    

    理解大型、复杂且快速演进的代码库仍然是规模化软件工程的核心挑战，它阻碍了新人上手、代码维护、重构和影响分析。将这些信息提供给大语言模型（LLM）是另一个障碍，因为LLM不仅需要理解所需的代码方法，还需要理解其在整个仓库全局结构中相关联的方法。我们提出了Code Comprehensor，这是一个混合检索-推理框架，结合了：（i）代码依赖调用图，（ii）CodeScribe——利用代码感知的LLM在仓库调用图上生成层次化文档，（iii）CodeContrast——一个CLIP风格的双编码器模型，基于CodeScribe的输出进行训练以学习对齐的文本-代码嵌入。CodeContrast能够实现快速、语义精准的相关代码上下文邻域检索，这些邻域作为基于MCP的LLM编排器回答领域特定查询时的可靠依据。

    arXiv:2610.30016v1 Announce Type: cross  Abstract: Understanding large, complex, and rapidly evolving code bases remains a core challenge for software engineering at scale, hindering onboarding, maintenance, refactoring, and impact analysis. Providing this information to a Large-Language Model (LLM) is another hurdle, as the LLM must understand not only the required code method but also its associated methods in the grand scheme of the repository. We propose Code Comprehensor, a hybrid retrieval-reasoning framework combining (i) a code dependency callgraph, (ii) CodeScribe, which uses code-aware LLMs to generate hierarchical documentation across repository call graphs, and (iii) CodeContrast, a CLIP-style dual-encoder model trained on CodeScribe output to learn aligned text-code embeddings. CodeContrast enables fast, semantically precise retrieval of relevant code-context neighborhoods, which serve as grounded evidence for an MCP-based LLM orchestrator answering domain-specific queries
    
[^20]: 智能体行为地图：无需运行智能体，通过“它们会做什么”来定位智能体

    Agent Behavior Maps: Locating Agents by What They Would Do, Without Running Them

    [https://arxiv.org/abs/2610.30014](https://arxiv.org/abs/2610.30014)

    提出“智能体行为地图”方法，利用共享参考轨迹上的中心化动作对数似然为LLM智能体定位，无需实际运行目标智能体即可刻画其行为特征、追踪轨迹KL散度并支持性能预测与行为相似性分析。

    

    具有相似成功率的LLM智能体可能遵循不同的动作序列。智能体行为地图利用共享参考轨迹上的中心化动作对数似然来表示LLM-框架组合，而无需对目标智能体进行新的环境回放。我们在明确假设下推导出将坐标距离平方与轨迹KL散度联系起来的误差界，并且在八个交叉评分的模型中，这些距离在排序上与直接测量的轨迹KL散度保持一致。在100个LLM和四个环境的实验中，基于ALFWorld的坐标支持性能预测并识别行为相似性。在ALFWorld中，即使控制了成功率和模型规模，这些相似性依然存在，包括跨模型家族的相似性。作为扩展，框架组件的位移近似满足可组合性，其精度在不同组件和环境之间有所差异。

    arXiv:2610.30014v1 Announce Type: cross  Abstract: LLM agents with similar success rates can follow different action sequences. An Agent Behavior Map represents LLM-harness combinations using centered action log-likelihoods on shared reference trajectories, without requiring new environment rollouts from the target agent. We derive error bounds relating squared coordinate distances to trajectory KL under explicit assumptions, and across eight cross-scored models the distances track the directly measured trajectory KL in rank order. Across 100 LLMs and four environments, ALFWorld-based coordinates support performance prediction and identify behavioral similarities. In ALFWorld, these similarities persist after controlling for success rates and model size, including across model families. As an extension, harness-component displacements approximately compose, with accuracy varying across components and environments.
    
[^21]: 通过梦境学习缓解模型坍塌

    Mitigating Model Collapse through Dreaming Learning

    [https://arxiv.org/abs/2610.30009](https://arxiv.org/abs/2610.30009)

    该论文从谱分析角度揭示低温采样（Ts < 1）会使转移算子锐化、谱隙收窄从而导致模型坍塌，并提出“梦境学习”正则化方法——在训练中对模型以高温生成的自身样本施加额外损失项——来缓解这一退化。

    

    在合成数据上训练生成模型会导致多样性丧失和性能下降，这一现象被称为模型坍塌。我们在受控的马尔可夫环境中从谱分析的角度研究这种退化。学习到的转移算子的特征值谱为该过程提供了直接的观察视角，因为次主导特征值决定了马尔可夫链的混合速度，进而决定了模型对状态空间的持续探索程度。我们发现，在温度 Ts < 1 下采样对转移算子的作用类似于反复锐化，会将其推向一个确定性系统：次主导特征值趋近于单位圆，谱隙收窄。当 Ts < 1 时，即使是数据累积似乎也无法阻止这种漂移。随后我们分析了梦境学习，这是一种正则化方法，它在常规训练的同时，对模型以较高温度 Td > 1 采样生成的自身样本添加一个损失项。由于高温样本会平滑……

    arXiv:2610.30009v1 Announce Type: new  Abstract: Training generative models on synthetic data causes loss of diversity and a decrease in performance, a phenomenon known as model collapse. We study this degeneration spectrally in a controlled Markov setting. The eigenvalue spectrum of the learned transition operator offers a direct view of the process, since the subdominant eigenvalues set how fast the chain mixes and therefore how much of the state space it continues to explore. We find that sampling at temperature $T_s < 1$ acts on the operator like repeated sharpening, pushing it toward a deterministic system: the subdominant eigenvalues approach the unit circle and the spectral gap narrows. When $T_s < 1$, even data accumulation does not appear to arrest this drift. We then analyze Dreaming Learning, a regulariser that, in parallel with regular training, adds a loss term on the model's own samples drawn at a raised temperature $T_d > 1$. Because high-temperature samples flatten the 
    
[^22]: EPIC：通过精确KL前缀重要性比率裁剪重新思考自回归大语言模型策略的信任域

    EPIC: Rethinking Trust Regions for Autoregressive LLM Policies via Exact-KL Prefix Importance-Ratio Clipping

    [https://arxiv.org/abs/2610.29999](https://arxiv.org/abs/2610.29999)

    该论文提出EPIC方法，将自回归生成建模为有限时域MDP并推导性能差恒等式，证明生成前缀的累积似然比才是逐步策略改进的精确重要性权重，并据此用精确反向KL裁剪构建前缀级信任域，从而修正标准PPO token级裁剪忽略前缀分布偏移所导致的偏差。

    

    基于PPO式裁剪的强化学习（RL）已成为提升大语言模型（LLM）推理能力的有效方法。然而，标准的token级裁剪仅通过当前token的条件概率来衡量策略变化，而作为条件的生成前缀本身也是由策略产生的。这种不匹配在长响应和离策略（off-policy）训练场景中尤其严重。我们将自回归生成重新表述为有限时域马尔可夫决策过程，并在行为策略采样下推导出其性能差恒等式。由此得到的测度变换表明，所生成前缀的累积似然比正是每一步策略改进项的精确重要性权重。从信任域策略优化（TRPO）的单调改进原则出发，我们进而利用精确的反向KL（reverse-KL）构建了前缀级信任域（摘要原文在此处截断）。

    arXiv:2610.29999v1 Announce Type: new  Abstract: Reinforcement learning (RL) based on PPO-style clipping has become an effective approach to improving the reasoning capabilities of large language models (LLMs). However, standard token-level clipping measures policy change only through the conditional probability of the current token, while the prefix conditioning context is itself generated by the policy. This mismatch becomes especially consequential in long-response and off-policy training. We reformulate autoregressive generation as a finite-horizon Markov decision process and derive its performance-difference identity under behavior-policy sampling. The resulting change of measure identifies the cumulative likelihood ratio of the generated prefix as the exact importance weight for each stepwise policy-improvement term. Starting from the monotonic-improvement principle of trust-region policy optimization, we then construct a prefix-level trust region using the exact reverse-KL coord
    
[^23]: SCDM：基于元学习的非线性时序系统可扩展因果发现

    SCDM: Scalable Causal Discovery in Nonlinear Temporal Systems with Meta-Learning

    [https://arxiv.org/abs/2610.29992](https://arxiv.org/abs/2610.29992)

    该论文提出SCDM框架，通过元学习将每个目标变量作为任务并共享时序预测器与因果强度矩阵，避免了穷举式逐目标优化，从而实现高维非线性时序系统的可扩展因果发现。

    

    从非线性多元时间序列中进行因果发现在高维系统中极具挑战性，因为候选有向关系的数量随变量数量呈二次增长。现有方法往往受限于逐目标变量的模型拟合、重复的条件独立性检验或昂贵的图搜索过程。我们提出了SCDM，一个用于可扩展时序因果发现的共享参数元学习框架。SCDM将每个目标变量视为一个任务，同时跨任务学习一个共享的时序预测器和一个共享的因果强度矩阵。每个元学习回合仅更新目标变量任务的一个子集，使模型能够在整个系统中积累结构证据，而无需进行穷举式的逐目标优化。训练完成后，训练后图读取步骤将学习到的预测器转换为连续的有向因果分数，以实现与阈值无关的评估。我们还提供了有限误差下的结构恢复保证（摘要原文在此处截断）。

    arXiv:2610.29992v1 Announce Type: new  Abstract: Causal discovery from nonlinear multivariate time series is challenging in high-dimensional systems, where the number of candidate directed relations grows quadratically with the number of variables. Existing methods are often limited by target-wise model fitting, repeated conditional testing, or expensive graph-search procedures. We propose SCDM, a shared-parameter meta-learning framework for scalable temporal causal discovery. SCDM treats each target variable as a task while learning a shared temporal predictor and a shared causal-strength matrix across tasks. Each meta-episode updates only a subset of target-variable tasks, allowing the model to accumulate structural evidence across the full system without exhaustive target-wise optimization. After training, a post-training graph readout converts the learned predictor into continuous directed causal scores for threshold-independent evaluation. We also provide a finite-error recovery p
    
[^24]: TIDE：基于模型行为的树引导细粒度诊断评估

    TIDE: Tree-Induced Fine-Grained Diagnostic Evaluation from Model Behaviors

    [https://arxiv.org/abs/2610.29985](https://arxiv.org/abs/2610.29985)

    TIDE 将诊断评估转化为行为驱动的决策树归纳，通过从基准样本和模型分数中自动构建分层能力树，实现了对大语言模型能力的细粒度诊断，揭示出聚合分数无法反映的能力差异。

    

    基准测试性能是比较大型语言模型（LLM）的主要标准，通常用一个聚合分数来概括，例如准确率、通过率或评分标准分数。这种聚合分数所提供的关于模型底层能力的信息有限，而理解这些能力对模型开发和评估至关重要。为此，我们提出了树引导诊断评估（TIDE），它将诊断评估重新构建为行为驱动的决策树归纳，以实现细粒度的能力诊断。TIDE 从基准样本和模型分数中递归地构建分层能力树：在每个节点，由一个 LLM 评判器提出候选拆分并执行样本到分组的分类，并依据基于行为的信息增益从中选择最优拆分。我们将 TIDE 应用于五个模型和四个基准测试以验证其有效性，并发现了在聚合分数层面无法观察到的细粒度能力差异。

    arXiv:2610.29985v1 Announce Type: cross  Abstract: Benchmark performance is the dominant standard for comparing large language models (LLMs), typically summarized by a single aggregate score, such as accuracy, pass rate, or rubric score. This aggregate score provides limited information about a model's underlying capabilities, yet understanding these capabilities is crucial for model development and evaluation. To this end, we propose Tree-Induced Diagnostic Evaluation (TIDE), which recasts diagnostic evaluation as behavior-driven decision-tree induction to perform fine-grained capability diagnosis. TIDE recursively builds a hierarchical capability tree from benchmark samples and model scores: at each node, an LLM judge proposes candidate splits and performs sample-to-group classification, and behavior-based information gain selects among them. We apply TIDE to five models across four benchmarks to demonstrate its effectiveness, and find fine-grained capability contrasts invisible at t
    
[^25]: 非平稳多臂老虎机中现时效应推断的代价

    The Price of Current-Effect Inference in Nonstationary Multi-Armed Bandits

    [https://arxiv.org/abs/2610.29982](https://arxiv.org/abs/2610.29982)

    该论文揭示了非平稳多臂老虎机中收益最大化与现时效应推断之间的内在权衡，证明了以有限宽度置信区间同时覆盖所有现时效应所需付出的动态遗憾极小极大代价，并提出 TRACE 算法来同时兼顾这两个目标。

    

    自适应实验利用老虎机算法来偏好具有更高期望收益的干预措施。然而，决策者还需要对现时处理效应进行可靠的推断：即在当前时刻干预措施与参照臂平均收益之间的差异，而非历史平均。这两个目标相互冲突：收益最大化倾向于选择更优的干预，而现时效应推断则要求随着结果随时间变化，持续探索可能次优的干预。我们针对平均收益具有指数 β∈(0,1] 和常数 L 的 Hölder 平滑性的 K 臂老虎机刻画了这一权衡。在要求置信区间以不超过 w 的宽度、以高概率同时覆盖所有现时效应的条件下，我们在期望动态遗憾上建立了（在对数因子范围内）匹配的极小极大界。在时间跨度 T 内，我们的算法 TRACE（Tracking Rewards and Current Effects）实现了遗憾……（原文摘要至此截断）

    arXiv:2610.29982v1 Announce Type: cross  Abstract: Adaptive experiments use bandit algorithms to favor interventions with higher expected rewards. However, decision-makers also require reliable inference on current treatment effects: differences between intervention and reference-arm mean rewards at the present time, rather than historical averages. These objectives conflict: reward maximization favors better interventions, whereas current-effect inference requires continued exploration of potentially suboptimal interventions as outcomes change over time. We characterize this tradeoff for $K$-arm bandits with H\"older-smooth mean rewards of exponent $\beta\in(0,1]$ and constant $L$. Requiring high-probability simultaneous coverage of all current effects by confidence intervals of width at most $w$, we establish matching minimax bounds, up to logarithmic factors, on expected dynamic regret. Over horizon $T$, our algorithm TRACE (Tracking Rewards and Current Effects) achieves regret of o
    
[^26]: CORD：面向智能体强化学习与蒸馏联合训练的冲突感知优化方法

    CORD: Conflict-Aware Optimization for Joint RL and Distillation in Agents

    [https://arxiv.org/abs/2610.29965](https://arxiv.org/abs/2610.29965)

    提出CORD冲突感知优化框架，通过最近点投影以闭式更新有选择地消除强化学习与蒸馏目标间相互抵消的梯度分量，从而在多种智能体联合训练方案中提升训练效果。

    

    将强化学习与自蒸馏相结合的联合训练方法，把任务奖励与稠密的教师指导信号结合起来用于智能体训练，但二者的梯度可能部分相互抵消。混合系数只能控制两种目标的相对规模，却无法有选择地去除相互对抗的分量。我们提出了CORD，一个面向双目标冲突感知优化的通用框架。CORD将“受保护目标”的选择与“触发投影”的条件相互解耦，并在每个优化器小批次中仅进行一次梯度组合。基于最近点形式的表述，CORD给出了闭式更新规则，并在优化器变换之前刻画了其对受保护目标的一阶效应。梯度诊断方法将智能体训练中的梯度冲突与局部优化进展的损失联系起来，从而支持有选择性的修正。作为一种梯度组合机制，CORD可以嵌入到不同的联合训练方案中，同时保留它们各自的目标。在三种此类方案中，CORD在ALFWorld等任务上均带来了性能提升……

    arXiv:2610.29965v1 Announce Type: new  Abstract: Joint reinforcement learning and self-distillation combines task rewards with dense teacher guidance for agent training, but their gradients can partly cancel. A mixing coefficient controls relative scale without selectively removing opposing components. We present CORD, a general framework for conflict-aware optimization of two objectives. CORD separates the choice of protected objective from the condition that triggers projection, and combines the gradients once per optimizer mini-batch. A nearest-point formulation yields a closed-form update and characterizes its first-order effect on the protected objective before optimizer transformations. Gradient diagnostics connect conflict in agent training to lost local optimization progress and motivate selective correction. As a gradient-combination mechanism, CORD can be added to different joint training recipes while retaining their objectives. Across three such recipes, it improves ALFWorl
    
[^27]: 掌握模型拟合：InterModel Vigorish（模型间赌利）在可解释机器学习中的应用

    Mastering Model Fit: Applications of the InterModel Vigorish for Interpretable Machine Learning

    [https://arxiv.org/abs/2610.29960](https://arxiv.org/abs/2610.29960)

    本文将“模型间赌利（IMV）”引入机器学习领域，通过 SHAP-IMV、Multi-IMV 和 Ablate-IMV 等可解释工具，将模型拟合差异量化为相对于参考模型的期望投注优势，从而揭示哪些特征、模型组件和训练阶段真正改善了概率拟合。

    

    arXiv:2610.29960v1 公告类型：交叉。仅有准确率几乎无法说明哪些特征或模型组件改善了概率拟合、哪些结果从中受益，以及拟合在训练过程中如何变化。我们将模型间赌利（InterModel Vigorish, IMV）引入机器智能领域，以将拟合差异表达为相对于参考模型的期望投注优势。SHAP-IMV 通过 Shapley 加权的子集比较将增益归因于各个特征，Multi-IMV 评估一对多及成对的类别判别能力，Ablate-IMV 则通过消融评估架构组件和特征组。我们使用表格数据、图像、传感器序列和图数据，在十个随机种子上对这些应用进行了评估。结果显示，特征排序在不同估计器之间存在差异；一对多准确率很高时，相对于零基准的改进却可能微不足道；而在固定训练预算下，消融可以保持甚至改善拟合。在六个图分类基准上，我们对 IMV 进行了超过 200 个训练（周期）的时序剖析……

    arXiv:2610.29960v1 Announce Type: cross  Abstract: Accuracy alone gives little information about which features or model components improve probabilistic fit, which outcomes benefit, or how fit changes during training. We introduce the InterModel Vigorish (IMV) into the Machine Intelligence space in order to express differences in fit as an expected betting advantage over a reference model. SHAP-IMV attributes gains to features through Shapley-weighted subset comparisons, Multi-IMV evaluates one-vs-rest and pairwise class discrimination, and Ablate-IMV assesses architectural components and feature groups by ablation. We evaluate these applications over ten random seeds using tabular data, images, sensor sequences, and graphs. Feature rankings vary across estimators. High one-vs-rest accuracy can accompany negligible improvement over a null, while ablation can preserve or improve fit under a fixed training budget. On six graph-classification benchmarks, we time-profile IMV over 200 trai
    
[^28]: FrontierCS 2：一个面向计算机科学研究与开发的基准

    FrontierCS 2: A Benchmark for Computer Science R&D

    [https://arxiv.org/abs/2610.29932](https://arxiv.org/abs/2610.29932)

    FrontierCS 2 通过删除论文贡献并将智能体置于真实研究环境中迭代提交补丁的方式，构建了一个包含 48 个开放式任务的基准，用于评估大语言模型智能体在前沿计算机科学研究与开发中的实际贡献能力。

    

    大语言模型智能体能否为前沿计算机科学研究与开发做出贡献？我们提出了 FrontierCS 2，一个包含 48 个开放式任务的基准测试。我们选取一篇论文及其作者发布的代码，并删除论文中所述的贡献部分，被删除的代码成为设定标准的人类参考实现。此前的开放式基准测试只是给每个智能体一个问题陈述，并在运行结束后对单个程序进行评分；而 FrontierCS 2 则将智能体置于一个研究环境中：智能体继承一个可运行的代码仓库，并可以在任意时刻提交补丁。对于每个补丁，裁判会返回一个开发集分数，但将评估器、留出的最终测试集和参考实现保持私有，并基于该最终测试集对照参考实现对最终提交进行评分。这些任务涵盖 13 个领域，从机器学习系统和高性能计算到数据库、安全与机器人技术。除了这 48 个任务之外，FrontierCS 2 还保留了一个包含 100 多个任务的私有任务集，将于……（原文在此处截断）

    arXiv:2610.29932v1 Announce Type: new  Abstract: Can LLM agents make contributions to frontier computer science R&D? We introduce FrontierCS 2, a benchmark of 48 open-ended tasks. We take a paper and the code its authors released and delete the paper's contribution. The deleted code becomes the human reference that sets the bar. Earlier open-ended benchmarks give each agent a problem statement and grade one program after the run. FrontierCS 2 places the agent inside a research environment. The agent inherits a working repository and may submit a patch at any point. For each patch, a judge returns a development-suite score but keeps the evaluator, the held-out final suite and the reference private, and grades the final submission on that suite against the reference. The tasks span 13 domains, from machine-learning systems and high-performance computing to databases, security and robotics. Beyond these 48 tasks, FrontierCS 2 keeps a private set of more than 100 tasks, to be announced in 
    
[^29]: 你的基准测试是关于难度的经验测度

    Your Benchmark Is an Empirical Measure Over Difficulty

    [https://arxiv.org/abs/2610.29927](https://arxiv.org/abs/2610.29927)

    本文提出将基准测试视为难度上的经验测度这一新视角，识别并形式化了“测度失配”与“覆盖缺口”两种失效模式，提出两个具有严格理论保证的诊断指标，并对7个领域25个流行基准的分析表明许多基准的难度权重隐式失衡，其排行榜结论需谨慎解读。

    

    基准测试不仅仅是一组测试题目的集合；它天然地在难度维度上诱导出一个经验测度。在本工作中，我们发展了这种基于测度的基准评估视角。这一视角使我们能够识别并形式化两种核心失效模式：一是测度失配，即基准的难度分布与预期的难度分布不一致；二是覆盖缺口，即难度轴上未被覆盖的区域导致基准无法检测到有意义的局部训练进展。我们提出了两个能够刻画这两种失效模式的诊断指标，并为其提供了严格的理论保证。在实证方面，我们利用精心估计的题目难度，分析了横跨7个领域的25个流行基准。结果表明，许多基准隐式地对某些难度区域赋予了过高权重，因此其排行榜结论应当谨慎解读；另一些基准则在难度轴上留下了大量未被覆盖的区域。

    arXiv:2610.29927v1 Announce Type: cross  Abstract: A benchmark is not merely a collection of test questions; it naturally induces an empirical measure over difficulty. In this work, we develop this measure-based view of benchmark evaluation. This perspective allows us to identify and formalize two central failure modes: measure mismatch, where the benchmark difficulty distribution differs from the intended difficulty distribution, and coverage gaps, where uncovered regions along the difficulty axis prevent benchmarks from detecting meaningful localized training progress. We propose two diagnostic metrics that capture these two failure modes and provide rigorous theoretical guarantees. Empirically, we analyze 25 popular benchmarks spanning 7 domains using carefully estimated item difficulties. Our results show that many benchmarks implicitly overweight certain difficulty regions, so their leaderboard conclusions should be interpreted with caution; others leave substantial portions of th
    
[^30]: 通用大语言模型时代下，情境化隐私正在被忽视

    Privacy in Context Is Being Overlooked in the Age of General-Purpose LLMs

    [https://arxiv.org/abs/2610.29925](https://arxiv.org/abs/2610.29925)

    本文指出隐私本质上是情境相关的，当前为LLM提示隐私构建统一框架的研究方向具有误导性并可能带来虚假安全感，应转向面向特定应用的部署与面向特定情境的评估。

    

    随着大型语言模型（LLM）在应用范围和能力上不断增强，隐私问题日益凸显。由于用户越来越多地在众多不同情境中与同一个通用系统进行交互，人们自然而然地倾向于追求一种“一刀切”的隐私解决方案。遗憾的是，这种做法忽视了数十年隐私研究中的一个关键洞见：隐私本质上是情境相关的。在本文中，我们论证了现有的提示隐私研究正在朝向设计统一框架的方向漂移，这种做法极具误导性，并可能营造出一种虚假的安全感。此外，我们认为即使未来沿此方向开展的研究，也必须面对提示隐私的诸多理想属性之间存在的根本性矛盾。因此，这种持续构建统一框架的尝试需要被重新审视。我们推荐三条关键的前进路径：面向特定应用的LLM部署、面向特定情境的评估等。

    arXiv:2610.29925v1 Announce Type: new  Abstract: As large language models (LLMs) continue to grow in scope and capability, privacy concerns are on the rise. Since users increasingly interact with a single general-purpose system across many distinct contexts, there is a natural tendency to pursue a one-size-fits-all solution to privacy. Unfortunately, this overlooks a crucial insight from decades of privacy research: privacy is inherently contextual. In this paper, we argue that existing work in prompt privacy is drifting towards attempts to design a unified framework, which is highly misleading and can foster a false sense of security. Furthermore, we argue that even future work in this direction needs to wrestle with fundamental tensions across various desirable properties of prompt privacy. Thus, such continued attempts towards a unified framework need to be reconsidered. We recommend three critical paths forward: application-specific deployment of LLMs, context-specific evaluation o
    
[^31]: 大语言模型价值训练的意外副作用

    Unintended Side Effects of Value Training in LLMs

    [https://arxiv.org/abs/2610.29922](https://arxiv.org/abs/2610.29922)

    本文通过在霍夫斯泰德各价值维度上微调四个大语言模型，发现针对特定价值的训练会对未训练的价值观、风险偏好、社会认知和安全特性产生意外副作用，因此建议价值对齐评估应纳入更广泛的行为测试。

    

    随着大语言模型越来越多地被部署在价值敏感的场景中，使其与人类价值观对齐已成为自然语言处理领域后训练阶段的一个目标，旨在向模型灌输额外的价值观并引导模型行为。然而，目前的评估仍然较为狭窄。由于各种价值观之间相互关联，针对特定价值观进行微调会如何改变模型无关的能力和行为尚不清楚。在这项工作中，作者分别在霍夫斯泰德价值理论的各个价值维度上对四个指令微调的大语言模型进行调整，并在跨价值迁移和更广泛的行为层面对其进行评估。研究发现，针对特定价值的训练会对未经训练的价值观、风险偏好、社会认知和安全特性产生意想不到的副作用。因此，作者建议在后训练之后的价值对齐过程中应纳入更广泛的行为评估，并强调需要进一步研究额外的价值训练最适合嵌入到训练流程的哪个环节。

    arXiv:2610.29922v1 Announce Type: new  Abstract: As LLMs are increasingly deployed in value-sensitive contexts, aligning them with human values has emerged as a post-training objective in NLP, aimed at instilling additional values and guiding model behaviour. However, current evaluations remain narrow. Because values are interconnected, it is unclear how fine-tuning for specific values alters unrelated capabilities and behaviours. In this work, we adapt four instruction-tuned LLMs across individual Hofstede's value dimensions and evaluate them on both cross-value transfer and broader behaviours. We find that value-specific training produces unintended side effects on untrained values, risk preferences, social cognition, and safety traits. Consequently, we suggest that value alignment after post-training should incorporate broader behavioral evaluations, and we highlight the need to investigate where additional value training might best fit into the pipeline.
    
[^32]: 防止上下文线性优化中的模型提取

    Preventing Model Extraction in Contextual Linear Optimization

    [https://arxiv.org/abs/2610.29918](https://arxiv.org/abs/2610.29918)

    该论文针对预测后优化（PTO）系统研究了模型提取攻击与防御，刻画了线性模型可被攻击者通过决策观察提取的条件，并提出了一种仅在预测成本向量接近决策边界时才扰动决策的防御机制，以在抵御提取攻击的同时控制决策遗憾。

    

    考虑一个上下文线性优化问题，其中预测后优化（predict-then-optimize, PTO）系统返回最优决策，但不透露其下游线性规划中所使用的成本向量。我们的关注点在于如何防止能够提交上下文查询的攻击者提取底层预测模型。尽管从成本到决策的映射是多对一的，但跨不同上下文的观察仍可能揭示底层预测模型。对于线性模型，我们给出了攻击者仅通过观察决策就能学习到该模型（至多相差一个正的缩放因子）的充分条件。这些条件启发了一种自适应攻击方法，该方法交替使用逆优化与边界采样，即利用导致当前攻击者模型决策边界的上下文查询。为防止模型提取，我们提出了一种机制，仅当预测的成本向量接近某个决策边界时才对决策进行扰动。我们建立了遗憾界……（原文摘要在此处截断）

    arXiv:2610.29918v1 Announce Type: new  Abstract: Consider a contextual linear optimization problem, where a predict-then-optimize (PTO) system returns optimal decisions without revealing the cost vectors used in its downstream linear program. Our focus is on how to prevent attackers, who can submit context queries, from extracting the underlying prediction model. Although the cost-to-decision map is many-to-one, observations across contexts can reveal the underlying prediction model. For linear models, we provide sufficient conditions where the attacker can learn the model up to positive scaling by only observing the decision. These conditions motivate an adaptive attack that alternates inverse optimization with boundary sampling, i.e., using context queries leading to decision boundaries of the current attacker model. To prevent extraction, we propose a mechanism that perturbs decisions only when the predicted cost vector is near a decision boundary. We establish a regret bound in ter
    
[^33]: 基于Kolmogorov-Arnold网络的可处理推理

    Tractable Inference with Kolmogorov-Arnold Networks

    [https://arxiv.org/abs/2610.29909](https://arxiv.org/abs/2610.29909)

    该论文提出TOUCKAN，通过为KAN网络引入有限次多项式外函数形成可分解的和-积表示，使期望、高阶矩等统计量能够通过确定性递归精确计算，从而避免了高维场景下昂贵的蒙特卡洛采样。

    

    计算科学中的机器学习代理模型通常不仅需要准确的预测，还需要计算期望、方差、守恒积分和归一化常数。传统神经网络（包括Kolmogorov-Arnold网络，即KAN）一般需要通过蒙特卡洛采样或多维数值求积来评估这些量，导致重复推理的成本高昂，尤其在高维情况下更是如此。我们提出了具有可处理算子的不确定性与约束KAN网络（TOUCKAN），其有限次多项式外函数产生了可分解的和-积表示。在因式分解的输入测度下，期望和高阶矩可通过代数方式化归为对一维叶子积分的确定性递归计算。独立和共享内块两种变体构成了一个参数效率层次结构，且针对其最受约束成员的通用逼近定理证明可扩展至整个类别。

    arXiv:2610.29909v1 Announce Type: cross  Abstract: Machine-learning surrogates in computational science often require not only accurate predictions but also expectations, variances, conserved integrals, and normalization constants. Conventional neural networks, including Kolmogorov--Arnold networks (KANs), generally evaluate these quantities through Monte Carlo sampling or multidimensional quadrature, making repeated inference costly, particularly in high dimensions. We introduce KANs with Tractable Operators for Uncertainty and Constraints (TOUCKANs), whose finite-degree polynomial outer functions yield a decomposable sum--product representation. Under factorizing input measures, expectations and higher moments then reduce algebraically to deterministic recursions over one-dimensional leaf integrals. Independent and shared inner-block variants form a parameter-efficiency hierarchy, and a universal-approximation proof for its most restricted member extends to the complete class. The sa
    
[^34]: 一个紧致的动态 Haussler 引理

    A Sharp Dynamic Haussler Lemma

    [https://arxiv.org/abs/2610.29898](https://arxiv.org/abs/2610.29898)

    本文将 Haussler 覆盖引理推广到序贯设定：对归一化 ℓ_p（1≤p<∞）中的有界实值类，存在与时间范围无关、规模为 (C/ε^p)^d 的全局动态覆盖（d 为序列 fat 维数），并给出匹配的下界，从而在 p=2 时解决了 Rakhlin 等人提出的序贯 ℓ₂ 覆盖问题。

    

    Haussler 覆盖引理（JCTA 1995）给出了二值 VC 类在归一化 ℓ₁ 距离下紧致的 (C/δ)^d 度量熵界。我们证明了它在归一化 ℓ_p 距离下、针对有界实值函数类（对每个 1≤p<∞）的紧致全局动态类似结果。记 d=sfat_{ε/32}(F)（序列 fat 维数），我们对任意时间范围以及 0<ε<1/2 得到规模至多为 (C/ε^p)^d 的全局动态覆盖。这里，仅由一族确定性的因果专家组成、且独立于实例序列事先固定的单个家族，即可在每条序列上覆盖每个目标。对于 0<ε<1/4，匹配的最坏情形下界 (c/ε)^{pd} 已经在一条固定序列上的二值类情形下成立，即使使用实值专家也是如此。其中的常数是普适的，与 p 无关。当 p=2 时，该结果以强形式解决了 Rakhlin、Sridharan 和 Tewari（PTRF 2015）提出的与时间范围无关的序贯 ℓ₂ 覆盖问题……

    arXiv:2610.29898v1 Announce Type: new  Abstract: Haussler's covering lemma (JCTA 1995) gives the sharp $(C/\delta)^d$ metric-entropy bound for binary VC classes under normalized $\ell_1$ distance. We prove a sharp global dynamic analogue for bounded real-valued classes in normalized $\ell_p$, for every $1\le p<\infty$. Writing $d=\operatorname{sfat}_{\varepsilon/32}(\mathcal F)$, we obtain a global dynamic cover of size at most $(C/\varepsilon^p)^d$ for every horizon and $0<\varepsilon<1/2$. Here a single family of deterministic causal experts, fixed independently of the instance sequence, covers every target on every sequence. For $0<\varepsilon<1/4$, a matching worst-case lower bound $(c/\varepsilon)^{pd}$ holds already for binary classes on one fixed sequence, even with real-valued experts. The constants are universal, independent of $p$.   For $p=2$, this resolves the horizon-independent sequential $\ell_2$ covering problem of Rakhlin, Sridharan, and Tewari (PTRF 2015) in a stronge
    
[^35]: 双层线性网络中的静默对齐：高维视角

    Silent Alignment in Two-Layer Linear Networks: A High-Dimensional Perspective

    [https://arxiv.org/abs/2610.29864](https://arxiv.org/abs/2610.29864)

    该论文在高维比例极限下借助随机矩阵理论精确刻画了双层线性网络梯度流的谱演化，并发现小初始化时存在“静默对齐”现象——网络奇异方向在一阶时间尺度上先完成对齐而损失保持不变，可见的学习随后才在对数时间尺度上以鞍点到鞍点的方式发生。

    

    我们研究了双层线性网络在高维情形下的梯度流，其中输入维度、隐藏层内部维度与输出维度成比例地发散。在白化输入和各向同性失衡的条件下，我们将动力学的闭式解与已有的随机矩阵理论结果相结合，推导出该动力学的高维极限。我们的方法给出了权重矩阵在任意时刻的精确渐近谱分布，并刻画了所学子空间的演化。在小初始化极限下，我们识别出一种“静默对齐”现象：网络的奇异方向在一阶时间尺度上完成对齐，而奇异值和经验损失却保持在其初始值不变。可见的学习随后才在（对数）时间尺度上发生，此时奇异值逐渐脱离零点，损失随之下降。这一阶段以“鞍点到鞍点”的动力学为特征，该动力学进而转变为一个连续的过程……

    arXiv:2610.29864v1 Announce Type: cross  Abstract: We study gradient flow in a two-layer linear network in the high-dimensional regime where the input, inner and output dimensions diverge proportionally. Under whitened inputs and isotropic imbalance, we combine a closed-form solution of the dynamics with prior random matrix theory results to derive a high-dimensional limit for the dynamics. Our method yields exact asymptotic spectral distributions for the weight matrices at all times and characterizes the evolution of the learned subspace. In the small-initialization limit, we identify a silent alignment phenomenon: the singular directions of the network align on an order-one timescale, while the singular values and empirical loss remain at their initial values. Visible learning occurs later, on a logarithmic timescale, as singular values progressively escape zero and the loss decreases. This phase is characterized by a saddle-to-saddle dynamics, which becomes a continuous process of s
    
[^36]: RangeNav：面向机器人导航的任务对齐激光雷达距离图像预训练

    RangeNav: Task-Aligned LiDAR Range-Image Pretraining for Robot Navigation

    [https://arxiv.org/abs/2610.29845](https://arxiv.org/abs/2610.29845)

    提出RangeNav，一个仅18.3万参数、完全基于合成数据预训练的轻量级激光雷达距离图像编码器，通过结合距离重建、语义分割和鸟瞰图预测的导航对齐目标，为大规模并行强化学习的机器人导航提供高效的360°感知表征。

    

    在密集人群和杂乱场所中运行的自主移动机器人需要360°感知能力以应对周围的危险。虽然全景激光雷达能够提供全方位覆盖，但在强化学习（RL）中学习表征在稀疏奖励下样本效率低下，且现有的3D预训练点云编码器对于大规模并行RL仿真而言内存开销过大。我们提出了RangeNav，这是一个紧凑型（18.3万参数）的激光雷达距离图像编码器，完全使用合成数据进行预训练。RangeNav采用各向异性卷积和循环填充来处理距离图像的极端长宽比和方位角接缝问题。我们使用结合距离重建、语义分割和鸟瞰图（BEV）预测的导航对齐目标对网络进行预训练，然后将其冻结用于策略学习。在仿真中，相比仅使用重建的基线，添加语义和BEV监督使成功率提升了（原文在此处截断）。

    arXiv:2610.29845v1 Announce Type: cross  Abstract: Autonomous mobile robots operating in dense crowds and cluttered sites require $360^\circ$ perception to react to surrounding hazards. While panoramic LiDAR offers full-surround coverage, learning representations within reinforcement learning (RL) is sample-inefficient under sparse rewards, and existing 3D pretrained point-cloud encoders are too memory-heavy for massively parallel RL simulation. We present RangeNav, a compact (183k parameters) LiDAR range-image encoder pretrained entirely on synthetic data. RangeNav uses anisotropic convolutions and circular padding to handle the extreme aspect ratio and azimuthal seam of range images. We pretrain the network using a navigation-aligned objective combining range reconstruction, semantic segmentation, and bird's-eye-view (BEV) prediction, then freeze it for policy learning. In simulation, adding semantic and BEV supervision increases success rates over a reconstruction-only baseline by a
    
[^37]: NeuroCodec：面向高保真目标说话人提取的EEG引导潜在域生成模型

    NeuroCodec: EEG-Guided Latent-Domain Generative Model for High-Fidelity Target Speaker Extraction

    [https://arxiv.org/abs/2610.29815](https://arxiv.org/abs/2610.29815)

    提出NeuroCodec，首个在神经音频编解码器潜在空间中运行的生成式EEG引导目标说话人提取框架，通过图卷积建模EEG空间依赖并结合注意力机制与Mamba时序建模进行跨模态融合，突破了传统时频掩蔽方法保真度低和仅限窄带音频的局限。

    

    在听力辅助设备近期发展的推动下，多说话人环境中的目标说话人提取（TSE）受到了广泛关注。为解决这一问题，听觉技术利用通过脑电图（EEG）记录的神经信号来识别注意力，以引导目标说话人提取，从而实现神经引导的听觉注意力解码。然而，大多数现有技术依赖于时频掩蔽方法，该方法在时频重叠区域的保真度较低，且通常仅限于窄带音频。相比之下，我们提出了NeuroCodec，据我们所知，这是首个在神经音频编解码器潜在空间中运行的生成式EEG引导目标说话人提取框架。具体而言，NeuroCodec通过图卷积建模EEG的空间依赖关系，并通过注意力机制和基于Mamba的时序建模来融合跨模态的EEG与语音表征。我们进一步引入了一个六阶段训练课程……

    arXiv:2610.29815v1 Announce Type: cross  Abstract: Driven by recent advancements in hearing assistive devices, target speaker extraction (TSE) in multi-talker environments has drawn significant attention. To address this, hearing technologies identify attention using neural signals recorded via electroencephalography (EEG) to guide TSE, enabling neuro-steered auditory attention decoding. However, most existing techniques rely on time-frequency masking, which suffers from low fidelity in spectro-temporally overlapping regions and is often restricted to narrowband audio. In contrast, we introduce NeuroCodec, which, to the best of our knowledge, is the first generative EEG-guided TSE framework operating in the latent space of a neural audio codec. Specifically, NeuroCodec models spatial EEG dependencies via graph convolutions and fuses cross-modal EEG and speech representations through attention mechanisms and Mamba-based temporal modeling. We further introduce a six-phase training curric
    
[^38]: 真实的PMID，错误的论文：生物医学论断验证智能体的框架设计

    Real PMIDs, Wrong Papers: Harness Design for Biomedical Claim-Verification Agents

    [https://arxiv.org/abs/2610.29812](https://arxiv.org/abs/2610.29812)

    该论文通过受控消融实验发现，在基于实时PubMed API的生物医学论断验证任务中，让语言智能体自行构造查询的框架设计能将标准论文召回率提升三倍并取得最佳准确率（62%），而简单的单次检索仅能为6%的论断找到标准论文且无法提升准确率。

    

    文献检索语言智能体正在成为生命科学领域科研助手的标准组成部分，然而关于其框架中哪些部分真正重要，目前缺乏受控的实验证据。我们以实时PubMed API作为唯一工具，在生物医学论断验证任务上进行了受控消融实验。在固定骨干模型和输出格式的前提下，我们比较了四种框架：闭卷直接回答、以论断本身作为查询的单次检索、能够自行编写PubMed查询的ReAct式智能体，以及给定标准摘要的oracle（理想上界），每种框架均使用两个高效的前沿骨干模型。天真的单次检索仅能为6%的论断找到标准论文，并且只是将过度自信的错误转化为弃权，而没有提升准确率。让模型自行构造查询可将标准论文的召回率提升三倍，并产生最佳的判定结果（准确率为62%，而其他现实可行的框架为49-59%），尽管这些差异并不显著。

    arXiv:2610.29812v1 Announce Type: cross  Abstract: Literature-searching language agents are becoming a standard component of scientific assistants for the life sciences, yet there is little controlled evidence on which parts of their harness matter. We run a controlled ablation on biomedical claim verification (SciFact) with the live PubMed API as the only tool. Holding the backbone and output format fixed, we compare four harnesses: closed-book answering, single-shot retrieval using the claim as the query, a ReAct-style agent that writes its own PubMed queries, and an oracle given the gold abstracts, each with two efficient frontier backbones. Naive single-shot retrieval finds the gold paper for only 6% of claims and turns overconfident errors into abstentions without improving accuracy. Letting the model formulate its own queries triples gold recall and yields the best verdicts (62% accuracy vs. 49-59% for the other realistic harnesses), although these differences are not significant
    
[^39]: 非对称无似然假设检验

    Asymmetric Likelihood-Free Hypothesis Testing

    [https://arxiv.org/abs/2610.29798](https://arxiv.org/abs/2610.29798)

    本文刻画了非对称无似然假设检验中可行的样本预算三元组（至多相差对数因子），证明对参考分布 P 和 Q 分配不等的样本预算可带来多项式级别的节省——例如将来自 D 的所需样本量从 Ω(√n) 降至 Õ(n^(1/4))——并将其应用于非自适应分布 2-聚类问题。

    

    在无似然假设检验（LFHT）中，给定来自定义在 [n] 上、且满足全变差距离 d_TV(P,Q)≥ε 的两个未知分布 P 和 Q 的样本，以及来自第三个分布 D∈{P,Q} 的样本，目标是判别 D=P 还是 D=Q。我们研究了非对称无似然假设检验，其中 P 和 Q 的样本预算 s_P 和 s_Q 可以不同，并在对数因子范围内刻画了可行的三元组 (s_P, s_Q, s_D)。我们证明了对参考分布采用不相等的样本预算可以带来多项式级别的节省。例如，在常数分离度下，从 P 采集 n 个样本、从 Q 采集 n^(3/4) 个样本，即可仅用 Õ(n^(1/4)) 个来自 D 的样本完成检验，而现有的两类基线方法——对称无似然假设检验与非等样本接近性检验——都需要来自 D 的 Ω(√n) 个样本。我们进一步将结果应用于非自适应分布 2-聚类问题，该问题中 k 个分布之间的样本分配必须在观测之前固定。

    arXiv:2610.29798v1 Announce Type: new  Abstract: In likelihood-free hypothesis testing (LFHT), given samples from two unknown distributions $P,Q$ on $[n]$ with $\mathrm{d_{TV}}(P,Q)\geq\varepsilon$, and from a third distribution $D\in\{P,Q\}$, the goal is to identify whether $D=P$ or $D=Q$. We study asymmetric likelihood-free hypothesis testing, where the sample budgets $s_P$ and $s_Q$ for $P$ and $Q$ may differ, and characterize the feasible triples $(s_P,s_Q,s_D)$ up to logarithmic factors. We establish that unequal reference budgets can give a polynomial saving. For instance, for constant separation, $n$ samples from $P$ and $n^{3/4}$ from $Q$ allow testing with $\widetilde O(n^{1/4})$ samples from $D$, whereas both existing baselines---symmetric LFHT and unequal-sample closeness testing---require $\Omega(\sqrt n)$ samples from $D$.   We further apply our results to non-adaptive distribution 2-clustering, where the sample allocation among $k$ distributions must be fixed before obser
    
[^40]: 大语言模型从哪里学会推荐AI？

    Where Do LLMs Learn to Recommend AI?

    [https://arxiv.org/abs/2610.29785](https://arxiv.org/abs/2610.29785)

    研究发现LLM过度推荐AI的倾向主要源于指令微调数据中约7%的AI相关样本——剔除这些数据可将推荐率降至预训练模型水平以下，而追求“有帮助”的数据和“最佳”“立刻”式的提问方式则会进一步推高该倾向。

    

    当被问及该学什么专业、从事什么职业或开发什么产品时，大语言模型（LLM）经常会推荐与AI相关的选项。我们通过比较微调前后的模型以及改变提问方式，来探究这种倾向的来源。首先，预训练模型在其27%的回复推荐中已包含明确的AI/ML术语，而其发布的指令微调版本中这一比例达到59%。其次，我们自己进行微调的实验表明，这种上升并非不可避免：一般的指令微调不会改变该比例，但剔除指令数据中涉及AI的7%样本后，该比例会下降14个百分点，甚至低于其起始的预训练模型水平。奖励“有帮助性”的数据会使该比例上升9个百分点，而根据多样化模拟用户偏好排序的数据则会使其下降9个百分点。最后，在模型保持不变的情况下，要求给出“最佳”选项或询问“现在”该做什么，也会提高这一比例。

    arXiv:2610.29785v1 Announce Type: new  Abstract: Asked what to study, which career to pursue, or what product to build, large language models (LLMs) often recommend AI-related options. We ask where this tendency comes from by comparing models before and after fine-tuning and by varying how questions are asked. First, pretrained models already include explicit AI/ML terms in the recommendations of 27% of their responses, and their released instruction-tuned versions do so in 59%. Second, fine-tuning models ourselves shows that this rise is not inevitable: general instruction tuning leaves the rate unchanged, but removing the 7% of instruction examples that concern AI cuts it by 14 percentage points, below the level of the pretrained models it starts from. Data rewarding helpfulness raise it by 9 points, while data ranked by the preferences of diverse simulated users lower it by 9. Finally, with models held fixed, asking for the "best" option or for what to do "right now" raises the rate
    
[^41]: 可监控性是可训练的：面向思维链可监控性的后训练

    Monitorability Is Trainable: Post-Training for Chain-of-Thought Monitorability

    [https://arxiv.org/abs/2610.29734](https://arxiv.org/abs/2610.29734)

    本文提出了一种通过最大化思维链与输出之间互信息的无监督后训练目标，能够恢复退化模型损失的思维链可监控性，并进一步提升未退化模型的监控效果与不安全行为预测能力。

    

    思维链监控是为数不多的可用于监督强人工智能系统的实用工具之一，但它十分脆弱：模型的推理只有在与模型的实际行为保持耦合时，才能对监控者有用。这种耦合可能会在训练过程中被削弱，例如通过思维链压缩、针对监控器的优化，以及来自合成文档的上下文外泛化。在本工作中，我们提出了一种无监督的后训练目标，通过最大化思维链与输出之间的互信息来提升思维链的可监控性。在三个具有退化思维链的模型载体上，该方法恢复了大部分乃至全部损失的可监控性。将该目标应用于未退化的策略时，它能进一步提升经评估的可监控性，增强思维链监控器和激活探针对不安全行为的预测能力，并提高“后果性推理”的出现频率。这些结果表明，可监控性应当被视为一种……

    arXiv:2610.29734v1 Announce Type: new  Abstract: Chain-of-thought (CoT) monitoring is one of the few practical tools available for overseeing capable AI systems, but it is fragile: a model's reasoning is only useful to a monitor if it remains coupled to what the model actually does. This coupling can be degraded during training through, for example, CoT compression, optimization against a monitor, and out-of-context generalization from synthetic documents. In this work, we propose an unsupervised post-training objective based on maximizing mutual information between CoT and the output for improving CoT monitorability. Across three model organisms with degraded CoT, it recovers most or all of the lost monitorability. Applied to undegraded policies, it further improves judged monitorability, improves prediction of unsafe behavior by CoT monitors and activation probes, and increases the prevalence of consequential reasoning. These results suggest that monitorability should be treated as a
    
[^42]: 视觉-语言-动作策略的后验认知不确定性估计

    Post-hoc Epistemic Uncertainty for Vision-Language-Action Policies

    [https://arxiv.org/abs/2610.29730](https://arxiv.org/abs/2610.29730)

    提出了一种后验核方差头方法，无需训练多个策略或修改原策略，即可从单一VLA策略的嵌入中估计认知不确定性，其表现与双策略集成方法相当。

    

    视觉-语言-动作（VLA）策略需要识别其训练知识中的空白，以便在新情况下请求有用的示范或协助，尤其是在失败代价高昂的场景中。独立训练的策略集合内部的分歧可以指导这类决策，但这需要训练和运行多个策略。我们提出了后验核方差头，能够从单一策略的嵌入中估计认知不确定性。每个头计算以训练示范的嵌入为条件的高斯过程条件方差。该方法的拟合不需要回归目标或梯度更新，且不改变原有策略。在SmolVLA于LIBERO-10上的实验中，核方差头在与任务成功的任务级相关性、主动学习期间的示范获取、以及向更强监督策略转介等方面，与双策略集成分数相比表现具有竞争力。

    arXiv:2610.29730v1 Announce Type: cross  Abstract: Vision-language-action (VLA) policies need to recognize gaps in their training knowledge to request useful demonstrations or assistance in new situations, especially when failures are costly. Disagreement within an ensemble of independently trained policies can guide these decisions, but requires training and running multiple policies. We propose post-hoc kernel-variance heads that estimate epistemic uncertainty from the embeddings of a single policy. Each head computes the conditional variance of a Gaussian process conditioned on embeddings from the training demonstrations. Fitting requires no regression targets or gradient updates and leaves the policy unchanged. In experiments with SmolVLA on LIBERO-10, the kernel variance heads perform competitively with two-policy ensemble scores in task-level correlation with task success, in demonstration acquisition during active learning, and in referral to a stronger supervisor policy as a pr
    
[^43]: 通过有向游走与关联扩散实现佩特网上的结构感知消息传递

    Structure-Aware Message Passing on Petri Nets through Directed Walks and Incidence Diffusion

    [https://arxiv.org/abs/2610.29725](https://arxiv.org/abs/2610.29725)

    该论文提出PNRWPE位置编码和DA-PNGNN两种结构感知机制，使图神经网络能够显式捕捉佩特网中的有向循环（过程回路）和关联矩阵不变量，将预测模型生成因子的宏平衡准确率从0.751提升至0.796，其中过程回路上的准确率几乎翻倍。

    

    过程挖掘学科的一个常见应用是将观测到的事件数据与佩特网关联起来，佩特网是一种具有丰富形式化基础的二部有向图形化过程模型。近期的工作学习了佩特网的向量表示以检索过程模型并将其与事件日志相关联，但隐式地忽略了隐藏在网络结构中的两个关键方面：有向循环（过程回路）和由网络关联矩阵记录的不变量。我们通过基于有向库所-变迁游走返回概率构建的位置编码（PNRWPE），以及通过由关联矩阵扩散内核门控的同类型消息（DA-PNGNN），将这两个方面显式地呈现给消息传递图神经网络。我们的构造将预测96个过程发现竞赛2023模型生成因子的宏平衡准确率从0.751提升至0.796（使用PNRWPE），这几乎使过程回路上的准确率翻倍，并进一步（摘要在此处截断）

    arXiv:2610.29725v1 Announce Type: new  Abstract: A common application of the process mining discipline is to connect observed event data with Petri nets, which are bipartite, directed graphical process models with rich formal underpinnings. Recent work learns vector representations of Petri nets to retrieve process models and relate them to event logs, but leaves implicit two key aspects hidden within the net's structure: directed cycles (process loops) and invariants recorded by the net's incidence matrix. We expose both of these to message-passing graph neural networks, through a positional encoding built from return probabilities of directed place-transition walks (PNRWPE), and through same-type messages gated by diffusion kernels of the incidence matrix (DA-PNGNN). Our constructions raise macro balanced accuracy in predicting the generation factors of the 96 Process Discovery Contest 2023 models from 0.751 to 0.796 with PNRWPE, which nearly doubles accuracy on process loops, and to
    
[^44]: 重新思考大型预训练模型的事后去偏

    Rethinking Post-hoc Debiasing of Large Pretrained Models

    [https://arxiv.org/abs/2610.29688](https://arxiv.org/abs/2610.29688)

    本文揭示了现有事后去偏方法依赖文本表达偏差特征的局限，证明在多偏差属性场景下其表现甚至不及普通分类器，并指出这些方法所寻求的偏差模式其实可以直接从冻结特征的几何结构中观察到。

    

    CLIP等大型预训练模型被广泛用作冻结的特征提取器。然而，它们的嵌入继承了训练数据中的虚假相关性，给特征引入了偏差。越来越多的研究试图通过事后去偏（post-hoc debiasing）来解决这一问题，即保持骨干网络不动。我们发现，尽管机制各异，这些方法都需要将偏差特征识别出来并以文本形式表达。这种依赖并非无害。在本工作中，我们通过实验证明，这一要求在某些场景下会带来不可预见的性能损失。例如，在包含两个共现偏差属性的UrbanCars基准上，近期的这些方法在带有偏差的冻结骨干特征上的表现甚至不如一个普通分类器，即使提供真实偏差身份也无法弥合这一差距。我们进一步表明，这类方法所寻求的模式在冻结特征的几何结构中其实是可以直接观察到的。具体来说，跨类……

    arXiv:2610.29688v1 Announce Type: cross  Abstract: Large pretrained models such as CLIP are widely used as frozen feature extractors. However, their embeddings inherit spurious correlations from training data, introducing bias into the features. A growing body of research has sought to address this issue through post hoc debiasing, leaving the backbone untouched. We discover that despite differing mechanisms, these methods require the bias feature to be identified and expressed in text. This dependency is not innocuous. In this work, we experimentally demonstrate that this requirement incurs an unforeseen penalty in certain scenarios. For example, on the UrbanCars benchmark, having two co-occurring bias attributes, the recent methods underperform a vanilla classifier on biased frozenbackbone features, and even providing ground-truth bias identities does not close the gap. We show that the pattern that such methods seek is readily observable in the frozen geometry. Specifically, cross-c
    
[^45]: TraceForge：用自己的思维进行蒸馏

    TraceForge: Distilling with Your Own Thinking

    [https://arxiv.org/abs/2610.29680](https://arxiv.org/abs/2610.29680)

    提出TraceForge，以教师模型的答案为输入、按学生模型自身家族的风格生成推理轨迹，并通过“监督+对抗”两阶段训练，在不违反服务条款的前提下实现对闭源前沿模型的推理蒸馏。

    

    使用一个大语言模型为另一个生成训练数据是最通用的训练方式之一，至今仍是常见做法。一些前沿团队已对自家模型关闭了这条路线：他们返回的只是答案或简短摘要，而不附带背后详细的推理轨迹。这略显讽刺：这些模型本身正是建立在蒸馏人类知识之上的，而蒸馏开源模型仍然完全常规；该限制仅针对它们自身的输出。最近有工作通过跨模型回放加密数据块来恢复隐藏的推理过程，虽然这种方法不知为何竟能奏效，但它违反了服务提供商的服务条款，因此无法作为长期构建产品的基础。我们构建了替代方案：以上下文和教师模型的答案作为输入，自行生成推理轨迹，并使其符合学生自身模型家族的风格。训练分为两个阶段——先监督学习、后对抗学习：一个生成器学习……

    arXiv:2610.29680v1 Announce Type: cross  Abstract: Using an LLM to generate training data for another is among the most general ways to train one, and it remains common practice. Some frontier teams have closed that route for their own models: what comes back is an answer, or a short summary, without the detailed reasoning trace behind it. That is a little ironic: they were themselves built on distilling human knowledge, and distilling an open-source model is still entirely routine; the restriction applies to their own outputs alone. Recent work recovers the hidden reasoning by replaying encrypted blocks across models. It somehow works, but it breaks the providers' terms of service, so it is not something a product can be built on long-term.   We built the alternative. We take the context and the teacher's answer as input, and generate the reasoning trace ourselves, in the style of the student's own model family. Training is two-stage, supervised and then adversarial: a generator learn
    
[^46]: 沃瑟斯坦空间中的PCA提升

    Lifting PCA in the Wasserstein Space

    [https://arxiv.org/abs/2610.29679](https://arxiv.org/abs/2610.29679)

    本文提出了“提升PCA”，通过凸延拓提升泛函的框架，首次将欧氏主成分分析规范化地扩展到Wasserstein空间，并将其归结为两个等价的、具有广义代价的Wasserstein重心最小化问题。

    

    主成分分析（PCA）是统计降维的基本工具，然而在最优传输的框架下，针对分布值数据的规范化扩展一直未曾出现。虽然Wasserstein重心和Wasserstein最小二乘回归分别为欧氏平均和线性回归在Wasserstein空间中提供了规范化的提升，但PCA的类似原理却始终未被探索。本文提出了提升PCA（Lifted PCA），这是欧氏PCA在Wasserstein空间中的一种规范化扩展，其通过凸延拓提升概率测度空间中泛函的框架获得。从Stiefel流形和Grassmann流形上PCA的经典变分形式出发，我们推导出两个等价的最小化问题，这些问题定义在支撑于上述流形上的概率测度之上，并对应于具有广义代价的Wasserstein重心。我们建立了（摘要在此处截断）

    arXiv:2610.29679v1 Announce Type: cross  Abstract: Principal Component Analysis (PCA) is a fundamental tool for statistical dimension reduction, yet no canonical extension to distribution-valued data has emerged within the framework of optimal transport. While Wasserstein barycenters and Wasserstein least-squares regression provide canonical liftings of Euclidean averaging and linear regression in the Wasserstein space, an analogous principle for PCA has remained unexplored. In this paper, we introduce Lifted PCA, a canonical extension of Euclidean PCA to Wasserstein space, obtained through the framework of lifting functionals in the space of probability measures by convex extension. Starting from the classical variational formulations of PCA on the Stiefel and Grassmann manifolds, we derive two equivalent minimization problems over probability measures supported on these manifolds that correspond to Wasserstein barycenters with generalized costs. We establish an exact analogue of the 
    
[^47]: DriveMLP：当训练策略超越模型复杂度时——自动驾驶中的发现

    DriveMLP: When Training Strategy Outperforms Model Complexity in Autonomous Driving

    [https://arxiv.org/abs/2610.29676](https://arxiv.org/abs/2610.29676)

    该论文提出专家引导强化学习（EGRL）训练策略，并训练出轻量级的DriveMLP策略，证明在自动驾驶闭环运动规划中，训练方法而非模型复杂度才是性能提升的关键因素。

    

    学习在闭环执行下保持可靠的运动规划策略是自动驾驶（AD）中的核心挑战。模仿学习（IL）可以很好地扩展到大型自动驾驶数据集，但在闭环执行下存在分布偏移、因果混淆和误差累积等问题。相比之下，强化学习（RL）通常需要手工设计的基于规则的奖励和大规模仿真。我们研究了一种简单的闭环模仿学习框架，该框架将强化学习纯粹用作优化工具，并由从记录的专家数据中导出的轨迹距离奖励来引导。我们将这种训练策略称为专家引导强化学习（EGRL）。通过受控的实证研究，我们发现证据表明，性能提升的很大一部分来自于训练过程本身，而非模型的大小和复杂度。随后，我们提出了DriveMLP，一个使用EGRL训练的轻量级MLP策略，作为这些发现的简单实例化。它取得了…

    arXiv:2610.29676v1 Announce Type: new  Abstract: Learning motion planning policies that remain reliable under closed-loop execution is a central challenge in Autonomous Driving (AD). Imitation Learning (IL) scales well to large AD datasets but suffers from distribution shift, causal confusion, and compounding errors under closed-loop execution. In contrast, Reinforcement Learning (RL) typically requires handcrafted rule-based rewards and massive simulation. We study a simple closed-loop IL framework that uses RL purely as an optimization tool, guided by a trajectory-distance reward derived from logged expert data. We refer to this training strategy as Expert-Guided RL (EGRL). Through a controlled empirical study we find evidence that the training procedure, rather than model size and complexity, accounts for a substantial part of the performance gains. We then introduce DriveMLP, a lightweight MLP policy trained with EGRL, as a simple instantiation of these findings. It achieves a scor
    
[^48]: 刻画无感知共同奖励多玩家多臂老虎机：通用平稳性、协调陷阱与可证全局速率

    Characterizing Oblivious Common-Reward Multi-Player Bandits: Universal Stationarity, Coordination Traps, and Certified Global Rates

    [https://arxiv.org/abs/2610.29673](https://arxiv.org/abs/2610.29673)

    本文证明在无感知共同奖励多玩家多臂老虎机中，玩家独立的梯度更新共同构成隐藏乘积策略目标的无偏随机梯度估计，从而获得适用于所有实例的通用有限时间一阶平稳性保证，同时通过乘积策略陷阱障碍表明平稳性一般无法提升为全局最优性。

    

    本文对无感知共同奖励多玩家多臂老虎机（OC-MPMAB）进行了理论分析，这是一个极简的多玩家老虎机模型，其中多个玩家同时选择臂，且只能观察到一份共享的奖励。关键在于，玩家是完全无感知的：他们无法观察其他玩家的动作，甚至不知道是否存在其他玩家。本工作的贡献有三个：首先，我们证明玩家各自独立执行的梯度老虎机更新共同构成了一个隐藏的乘积策略目标的无偏随机梯度估计器，这一性质进而为每个实例带来了通用的有限时间一阶平稳性保证。其次，我们证明平稳性通常无法被强化为全局最优性；为说明这一点，我们识别出一种乘积策略陷阱障碍，它排除了全局最优性保证的可能性。第三，我们证明全局……

    arXiv:2610.29673v1 Announce Type: new  Abstract: This paper presents a theoretical analysis of the oblivious common-reward multi-player multi-armed bandit (OC-MPMAB), a minimal multi-player bandit model in which multiple players select arms simultaneously and observe only a shared reward. Crucially, the players are fully oblivious: they do not observe the actions of other players and do not even know whether any other players exist. The contributions of this work are threefold. First, we prove that the independent gradient-bandit updates applied by the players jointly form an unbiased stochastic gradient estimator of a hidden product-policy objective. This property then yields a universal finite-time first-order stationarity guarantee for every instance. Second, we show that stationarity cannot, in general, be strengthened to global optimality. To illustrate this, we identify a product-policy trap obstruction that rules out the guarantee of global optimality. Third, we show that global
    
[^49]: 利用时间序列基础模型进行金融犯罪检测

    Leveraging Time-Series Foundation Models for Financial Crime Detection

    [https://arxiv.org/abs/2610.29667](https://arxiv.org/abs/2610.29667)

    该论文提出TCEE框架，利用冻结的时间序列基础模型总结账户交易历史的时间上下文来增强交易边的表示，在六个AMLWorld数据集上将PR-AUC相比最强基线平均相对提升10.8%，同时训练速度提升6倍。

    

    金融犯罪检测需要在大规模交易网络中识别非法活动，其中可疑行为稀少且随时间逐步演变。现有的基于图的方法能够捕捉账户之间的关系，但通常未能充分利用账户交易历史中编码的时间行为信息。我们提出了时间上下文边增强框架，通过使用冻结的时间序列基础模型对每个账户的交易历史进行总结，并利用这些摘要中的相关时间上下文来丰富每笔交易的表示。这些增强的表示可以直接用于边分类评分，也可以作为边特征输入图神经网络（GNN），以捕捉更广泛的结构信息。在六个AMLWorld数据集上，TCEE在其中五个数据集上将PR-AUC相比最强的评估基线平均相对提升10.8%，同时端到端训练速度提升6.0倍。

    arXiv:2610.29667v1 Announce Type: new  Abstract: Financial crime detection requires identifying illicit activity within large transaction networks, where suspicious behaviour is rare and unfolds over time. Existing graph-based methods capture relationships among accounts but typically underuse the temporal behaviour encoded in account transaction histories. We introduce Temporal-Context Edge Enrichment (TCEE), a framework that addresses this gap by summarising each account's transaction history with a frozen time-series foundation model (TSFM), and enriching each transaction's representation with the relevant temporal context from these summaries. These enriched representations can be scored directly for edge classification, or fed into a graph neural network (GNN) as edge features to capture broader structural information. Across six AMLWorld datasets, TCEE improves PR-AUC over the strongest evaluated baseline on five by a relative 10.8% on average, while training 6.0x faster end-to-e
    
[^50]: 面向大语言模型进化因子挖掘的谱系感知搜索控制

    Lineage-Aware Search Control for LLM-Based Evolutionary Factor Mining

    [https://arxiv.org/abs/2610.29663](https://arxiv.org/abs/2610.29663)

    提出LineageEvo框架，通过谱系条件化的LLM变异将谱系范围搜索经验提炼为算子先验并与全局先验融合，为LLM进化因子挖掘引入谱系层面的搜索控制，克服了传统方法搜索方向性弱、因子结构同质化的局限。

    

    公式化Alpha因子挖掘旨在搜索能够将市场数据转化为预测信号的符号表达式。近期基于大语言模型（LLM）的进化因子挖掘方法将大语言模型用作语义变异算子。在传统的LLM-进化算法（LLM-EA）范式下，随机变异和基于适应度的选择可能导致搜索方向性薄弱，并使搜索集中围绕一小组结构相似的因子。我们认为，这些局限性的产生是因为现有方法主要在个体和种群层面组织搜索，而谱系层面的控制未被充分利用。谱系通过进化传承将连续的因子变体连接起来。我们提出LineageEvo，它以两种相互耦合的搜索控制形式来利用谱系：（1）谱系条件化的LLM变异，将谱系范围搜索经验提炼为算子特定先验，并将其与全局算子先验融合，从而提供持续的、与谱系相关的指导……（原文摘要至此截断）

    arXiv:2610.29663v1 Announce Type: cross  Abstract: Formulaic alpha factor mining searches for symbolic expressions that transform market data into predictive signals. Recent LLM-based evolutionary factor mining methods use large language models as semantic variation operators. Under the conventional LLM-EA paradigm, stochastic variation and fitness-based selection can leave the search weakly directed and concentrated around a narrow set of structurally similar factors. We argue that these limitations arise because current methods mainly organize search at the individual and population levels, leaving lineage-level control underused. A lineage connects successive factor variants through evolutionary descent. We propose LineageEvo, which uses lineage in two coupled forms of search control: (1) lineage-conditioned LLM variation, which distills Lineage-Scoped Search Experience into operator-specific priors and fuses them with global operator priors, providing persistent, lineage-relevant g
    
[^51]: SPARE：通过路由熵进行稀疏度分配以实现高效的专家混合模型

    SPARE: Sparsity Allocation via Routing Entropy for Efficient MoEs

    [https://arxiv.org/abs/2610.29662](https://arxiv.org/abs/2610.29662)

    SPARE是一种无需训练的MoE专家剪枝方法，利用路由熵自适应地为各层分配不同的剪枝稀疏度，在相同预算下比均匀剪枝获得更高的模型准确率。

    

    混合专家架构能够高效地扩展模型容量，但其巨大的内存占用成为部署的主要障碍。专家剪枝可以减少内存占用，但大多数方法只关注删除哪些专家，并对每一层进行同等程度的剪枝。我们提出了SPARE（基于路由熵的稀疏度分配），这是一种无需训练、与选择器无关的方法，它利用路由熵——即token在校准数据上归一化Top-K门控权重的熵——来决定从每一层中删除多少个专家。当一个token的门控权重均匀分散时，删除一个活跃专家只会丢弃其中很小一部分信息，因此SPARE会更激进地剪枝那些路由更分散的层。随机剪枝下的扰动界限将这一直觉形式化，且使用固定选择器的受控干预实验表明集中路由与更大的输出变化相关联。在使用相同选择器和相同预算的情况下，SPARE相比均匀分配方式提升了平均准确率。

    arXiv:2610.29662v1 Announce Type: cross  Abstract: Mixture-of-Experts (MoE) architectures scale model capacity efficiently, yet their substantial memory footprint poses a major barrier to deployment. Expert pruning reduces this footprint, but most methods focus on which experts to remove and prune every layer equally. We propose SPARE (Sparsity Allocation via Routing Entropy), a training-free, selector-agnostic method that decides how many experts to remove from each layer using routing entropy, the entropy of tokens' normalized Top-K gate weights on calibration data. When a token's gate weight is spread evenly, removing one active expert discards little of it, so SPARE prunes layers with more diffuse routing more aggressively. A perturbation bound under random pruning formalizes this intuition, and controlled interventions with a fixed selector link concentrated routing to larger output changes. With the same selector and budget, SPARE improves average accuracy over uniform allocation
    
[^52]: 用于短期大气视宁度预报的概率机器学习

    Probabilistic Machine Learning for Short-Term Atmospheric Seeing Forecasting

    [https://arxiv.org/abs/2610.29654](https://arxiv.org/abs/2610.29654)

    该论文将高斯过程和基于归一化流的概率模型与确定性RNN/LSTM基线进行对比，实现了莫纳克亚山顶大气视宁度提前至两小时的概率预报及不确定性量化。

    

    准确的短期大气视宁度预报对于天文观测和自由空间光通信系统具有重要意义，因为光学湍流的快速变化会影响系统性能和运行规划。在本研究中，我们对能够提前最多两小时预报大气视宁度的机器学习模型进行了基准测试，并特别关注概率预报和不确定性量化。我们比较了两种概率方法——高斯过程（GP）和基于归一化流的模型——与确定性的循环神经网络（RNN）和长短期记忆网络（LSTM）基线模型。所有模型均在莫纳克亚山顶的视宁度测量数据上进行训练和评估。我们使用确定性指标（包括均方根误差和皮尔逊相关系数）以及评估预报质量与校准程度的概率指标对模型进行评估。

    arXiv:2610.29654v1 Announce Type: new  Abstract: Accurate short-term forecasting of atmospheric seeing is of significant interest for astronomical observations and free-space optical communication systems, where rapid variations in optical turbulence can affect system performance and operational planning. In this work, we benchmark machine learning models for predicting atmospheric seeing up to two hours in advance, with a particular focus on probabilistic forecasting and uncertainty quantification. We compare two probabilistic approaches, Gaussian Processes (GPs) and a normalizing-flow- based model (FloTS), against deterministic recurrent neural network (RNN) and long short-term memory (LSTM) baselines. All models are trained and evaluated on seeing measurements from the Maunakea summit. We assess the models using both deterministic metrics, including the root mean square error and the Pearson correlation coefficient, and probabilistic metrics that evaluate the quality and calibration
    
[^53]: 面向任务的语义通信的递归代码演进

    Recursive Code Evolution for Task-Oriented Semantic Communications

    [https://arxiv.org/abs/2610.29641](https://arxiv.org/abs/2610.29641)

    该论文提出ComRSI框架，通过大语言模型辅助的递归代码自我改进来自动演进语义通信程序，最终仅用每张图像32个复符号就实现了74.3%的测试精度，优于使用8倍符号资源的初始设计。

    

    语义通信联合优化推理精度、信道使用和计算资源，但底层的通信程序通常在训练开始前就已固定。本文提出了ComRSI（面向通信的递归自我改进），一个通过大语言模型辅助的代码编辑来递归演进该通信程序的框架。一个固定的智能体负责修改编码器、信道适配器、解码器、损失函数和训练策略；一个受保护的评估器在规定的预算和资源约束下筛选候选方案。被接受的代码和文本反馈会为后续提案提供指导，同时候选模型的权重从零开始训练。在最初的十代CIFAR-100演化谱系中，八次被接受的修订产生了一个注意力池化代码，在每张图像32个复符号的加性高斯白噪声（AWGN）信道条件下实现了74.3%的测试精度，而初始设计需要256个符号才达到67.2%的精度。

    arXiv:2610.29641v1 Announce Type: cross  Abstract: Semantic communication jointly optimizes inference accuracy, channel use, and computation, yet the underlying communication program is typically specified before training. This paper presents ComRSI (Communications-Oriented Recursive Self-Improvement), a framework for recursively evolving that program through large-language-model-assisted code edits. A fixed agent modifies the encoder, channel adapters, decoder, loss, and training policy; a protected evaluator screens candidates under prescribed budgets and resource constraints. Accepted code and textual feedback inform subsequent proposals, while candidate weights are trained from scratch. In the original ten-generation CIFAR-100 lineage, eight accepted revisions produce an attention-pooled code achieving 74.3% test accuracy over an Additive White Gaussian Noise (AWGN) channel at 32 complex symbols per image, compared with 67.2% at 256 symbols for the initial design. The program also 
    
[^54]: 基于迁移效率评分的VLC室内定位跨区域迁移学习

    Cross-Area Transfer Learning for VLC-based Indoor Localization with a Transfer Efficiency Score

    [https://arxiv.org/abs/2610.29638](https://arxiv.org/abs/2610.29638)

    本文提出了一种面向VLC室内定位的跨区域迁移学习框架，仅用40%的目标区域数据微调预训练模型即可恢复97%的完整数据性能，并引入迁移效率评分（TES）指标，无需穷举评估即可识别最佳迁移源模型。

    

    室内定位系统对工业4.0应用至关重要。基于可见光通信（VLC）的系统具有抗电磁干扰、高精度和能效高等优势，但其性能受到部署区域之间环境差异的阻碍。本文提出了一种用于VLC室内定位的迁移学习（TL）框架，在降低数据需求的同时增强了跨区域鲁棒性。利用来自博世制造工厂四个不同区域的真实测量数据，我们表明，仅使用40%的目标区域测量数据对在源区域数据上预训练的深度神经网络（DNN）进行微调，即可恢复97%的全数据性能，从而减少了所需的目标数据量。我们进一步引入了迁移效率评分（TES），这是一种复合指标，无需穷举评估即可识别最有效的源模型进行迁移。

    arXiv:2610.29638v1 Announce Type: new  Abstract: Indoor localization systems are critical for Industry-4.0 applications. Visible Light Communication (VLC)-based systems offer advantages such as immunity to electromagnetic interference, high accuracy, and energy efficiency, but their performance is hindered by environmental variability across deployment areas. This paper presents a transfer learning (TL) framework for VLC-based indoor localization that enhances cross-area robustness while reducing data requirements. Using real-world measurements from four distinct areas of a Bosch manufacturing facility, we show that fine-tuning a deep neural network (DNN) pre-trained on source area data with only 40% of target-area measurements recovers 97% of full-data performance, thereby reducing the amount of target data needed. We further introduce the Transfer Efficiency Score (TES), a composite metric that identifies the most effective source models for transfer without exhaustive evaluation. Gu
    
[^55]: 十米尺度电离层不均匀体空间分布对空间天气的依赖性

    Space weather dependence of decameter-scale ionospheric irregularities spatial distribution

    [https://arxiv.org/abs/2610.29630](https://arxiv.org/abs/2610.29630)

    本文提出了一种基于SuperDARN和SECIRA雷达数据与空间天气参数、采用受图神经网络启发的原创神经网络架构的数据驱动多尺度方法，用于高纬度十米尺度电离层不均匀体空间分布的即时预报和预测。

    

    本文描述了一种基于SuperDARN和SECIRA雷达数据以及空间天气参数，构建高纬度十米尺度电离层不均匀体发生的即时预报和预测模型的方法。该方法依赖于对影响不均匀体形成的空间天气参数动态变化的自学习模式识别。所提出的方法是一种数据驱动的多尺度方法，其中不均匀体的空间分布通过一种受图神经网络启发的原创架构神经网络来逼近。该模型使用2010-2025年期间11部雷达的数据进行了训练和测试（SuperDARN：CVW、ICE、ICW、INV、PGR、PYK、RKN、SAS、STO；SECIRA：EKB、MAGW）。结果表明，当使用地磁坐标MLT、MLAT时，模型达到最优性能。在所研究的参数集合中，获得最佳即时预报和预测的最优参数组合为……

    arXiv:2610.29630v1 Announce Type: cross  Abstract: This paper describes an approach to constructing nowcast and forecast predictive models for the occurrence of high-latitude decameter ionospheric irregularities based on SuperDARN and SECIRA radar data and space weather parameters. The method relies on self-learning pattern recognition in the dynamics of space weather parameters, that influence the irregularities formation. The proposed approach is a data-driven, multiscale method in which the spatial distribution of irregularities is approximated using a neural network of an original architecture, inspired by graph neural networks.   The model was trained and tested using data from 11 radars during the period 2010-2025 (SuperDARN: CVW, ICE, ICW, INV, PGR, PYK, RKN, SAS, STO; SECIRA: EKB, MAGW). It is shown that the model achieves optimal performance when using geomagnetic coordinates MLT,MLAT. The optimal set of parameters for the best nowcast and forecast over studied set of paramete
    
[^56]: 面向不完整证据的多模态学习可靠性感知概率融合

    Reliability-Aware Probabilistic Fusion for Multimodal Learning with Incomplete Evidence

    [https://arxiv.org/abs/2610.29621](https://arxiv.org/abs/2610.29621)

    该论文提出一种可靠性感知的高斯乘积专家方法，利用输入相关的可靠性权重调节各模态精度并自动忽略缺失专家，从而在模态缺失的临床多模态任务（如HAIM-MIMIC-MM）中取得明显的AUROC性能优势。

    

    多模态输入可能出现缺失或退化，这引发了关于证据可用性与可信度的不同问题。我们提出了一种可靠性感知的高斯乘积专家方法，其中输入相关的可靠性权重可以调节每个模态的潜在精度，同时忽略缺失的专家。我们刻画了可靠性和可用性如何影响后验信息与专家影响力。在两个临床多模态任务上，完整框架在输入完整时保持竞争力，并在模态缺失的情况下取得了最明显的平均AUROC优势，尤其是在HAIM-MIMIC-MM数据集上。消融实验表明缺失感知训练带来了显著提升，并且在匹配的PoE、缺失感知和一致性训练条件下，可靠性加权为HAIM带来了额外的改进。未见过的测试时损坏被用于诊断性评估，揭示了模态依赖的可靠性权重和潜在不确定性响应，包括在严重情况下出现的失效。

    arXiv:2610.29621v1 Announce Type: new  Abstract: Multimodal inputs may be missing or degraded, raising distinct questions of evidence availability and trust. We introduce a reliability-aware Gaussian Product-of-Experts (PoE) in which an input-dependent reliability weight tempers each modality's latent precision, while missing experts are omitted. We characterize how reliability and availability affect posterior information and expert influence. Across two clinical multimodal tasks, the full framework remains competitive with complete inputs and achieves its clearest mean-AUROC advantage under modality absence, particularly on HAIM-MIMIC-MM. Ablations show a substantial gain from missing-aware training and an additional HAIM improvement from reliability weighting under matched PoE, missing-aware, and consistency training. Unseen test-time corruptions are used diagnostically and reveal modality-dependent reliability-weight and latent-uncertainty responses, including failure under severe 
    
[^57]: 注意脚下：面向人形机器人运动的预期感知

    Watch Your Step: Anticipated Perception for Humanoid Locomotion

    [https://arxiv.org/abs/2610.29612](https://arxiv.org/abs/2610.29612)

    提出一种轻量级落足点预测模型，仅凭本体感知和速度指令预测未来落足点，并只在高程图中采样这些落足点周围的小区域作为策略输入，从而在感知进入策略之前剔除冗余信息，加速训练并提升对新地形的泛化能力。

    

    感知运动策略通常观察以机器人为中心的稠密高程图，然而实际上只有下一步落足点下方及周围的地形才对控制决策有意义。学习型编码器会对整个视觉输入进行压缩，迫使策略在学习基础运动任务的同时还要自行判断哪些输入是重要的。同时学习这两者会减慢训练速度，而且在缺乏计算开销高昂的注意力机制的情况下，会使得对新地形的泛化变得更加困难。我们反其道而行之，在策略接收到高程图之前就剔除其中的冗余输入。一个轻量级的落足点预测模型（FPM）仅凭本体感知和速度指令即可预测未来的落足点，而我们只在这些预测落足点周围的小区域内对高程图进行采样。我们将预测模型与运动策略一同训练，并随着机器人步态的成熟，定期基于新的策略轨迹数据对预测模型进行重新训练。这使得感知能够预先……（摘要在此处被截断）

    arXiv:2610.29612v1 Announce Type: cross  Abstract: Perceptive locomotion policies typically observe a dense robot-centric elevation map, although only the terrain beneath and around the next footsteps informs the control decision. Learned encoders compress the whole visual input, leaving the policy to work out which inputs matter while learning the baseline locomotion task. Learning both at once slows training and, without computationally expensive attention mechanisms, makes generalization to new terrain harder. We instead remove redundant inputs from the elevation map before the policy sees it. A lightweight foothold prediction model (FPM) predicts future footholds from proprioception and the velocity command alone, and we sample the elevation map only in small patches around these predicted footholds. We train the prediction model alongside the locomotion policy, periodically retraining it on new policy rollouts as the robot's gait matures. This allows the perception to anticipate w
    
[^58]: 无需帧，只需频率：面向智能眼镜的带宽高效部署时自适应

    No Frames, Just Frequencies: Bandwidth-Efficient Deployment-Time Adaptation for Smart Glasses

    [https://arxiv.org/abs/2610.29607](https://arxiv.org/abs/2610.29607)

    提出了一种带宽高效的自适应框架，智能眼镜只需在设备端计算并发送紧凑的频率描述符而非视频帧，由主机设备据此合成分布外数据并自适应模型副本，从而在保障隐私的同时实现智能眼镜的部署时模型自适应。

    

    部署在智能眼镜上的深度神经网络必须在遵守严格的隐私和设备约束的同时，适应输入数据分布的变化。我们提出了“无需帧，只需频率”，这是一种带宽高效的源辅助自适应框架，它使用在设备端计算的紧凑频率描述符来刻画分布偏移。眼镜无需传输摄像头帧，而是计算频率描述符并将其发送到包含模型副本的主机设备。主机设备将接收到的频率描述符映射为分布偏移算子，该算子合成分布外数据，并按照预定义的自适应协议对模型副本进行自适应调整。我们在ImageNet-C损坏数据混合集上，使用基于卷积的、基于自注意力的和混合型视觉模型，以及人物分类和物体检测任务上评估了该框架。

    arXiv:2610.29607v1 Announce Type: cross  Abstract: Deep neural networks deployed on smart glasses must adapt to changes in the distribution of incoming data while adhering to strict privacy and device constraints. We present No Frames, Just Frequencies, a bandwidth-efficient source-assisted adaptation framework that characterizes distribution shifts using compact frequency descriptors computed on-device. Rather than transmitting camera frames, the glasses compute and send frequency descriptors to a host device that contains a replica of the model. The host device maps the received frequency descriptor to a distribution shift operator that synthesizes out-of-distribution data and adapts a model replica using a pre-defined adaptation protocol. We evaluate the framework on mixtures of ImageNet-C corruptions using convolution-based (ConvNeXtV2-Femto), self-attention-based (DeiT-Tiny), and hybrid (TinyViT-5M) vision models, on person classification (Visual Wake Words), and on object detecti
    
[^59]: 强随机流映射的改进训练

    Improved Training of Strong Stochastic Flow Maps

    [https://arxiv.org/abs/2610.29605](https://arxiv.org/abs/2610.29605)

    本文通过使用移位勒让德多项式展开来表示布朗路径，并结合 Wong-Zakai 风格的目标函数，改进了强随机流映射的训练，实现了基于导数的损失函数。

    

    随机流映射通过学习沿解路径直接在时间点之间跳跃，有望实现随机微分方程（SDE）的少步采样，然而训练这些模型来学习这一解映射仍然困难。主要障碍在于布朗路径本身：它是一个需要传递给神经网络的无穷维对象，且处处不可微，这使得为最先进的确定性流映射提供动力的基于导数的损失函数难以实现。我们确定了两个解决这些问题的核心设计原则：a) 使用携带高阶局部信息的基来表示布朗路径，b) 使用一个迫使网络利用路径完整表示（而不仅仅是一阶信息）的目标函数进行训练。为此，我们采用移位勒让德多项式展开并结合 Wong-Zakai 风格的目标函数，从而解锁了基于导数的损失函数。

    arXiv:2610.29605v1 Announce Type: new  Abstract: Stochastic flow maps promise few-step sampling of stochastic differential equations (SDEs) by learning to directly jump between times along the solution path, yet training these models to learn this solution map remains difficult. The primary obstacle is the Brownian path itself: it is an infinite-dimensional object which needs to be passed to the neural network and is nowhere differentiable, which makes the implementation of derivative-based losses that power the state-of-the-art deterministic flow maps just out of reach. We identify two core design principles which address these issues: a) represent the Brownian path with a basis which carries higher-order local information, and b) train with an objective that forces the network to use the full representation of the path, not just the first order. To address this, we leverage the shifted Legendre polynomial expansion with a Wong-Zakai styled objective, unlocking derivative-based losses
    
[^60]: 面向标签延迟下冻结多元预测器在线残差校正的局部与全局递归岭回归方法

    Local and Global Recursive Ridge Regression for Online Residual Correction of Frozen Multivariate Forecasters under Label Delay

    [https://arxiv.org/abs/2610.29604](https://arxiv.org/abs/2610.29604)

    提出DRiC方法，利用基于主DCT系数的递归岭回归和固定规模充分统计量，在标签延迟与小状态约束下对冻结多元预测器进行在线残差校正，提供按通道建模的局部变体和跨通道池化的全局变体，并通过残差尺度保护机制抵御异常数据块。

    

    冻结的多步预测器可以基于其自身的残差进行在线校正。然而，在标签延迟的情况下，一个数据块的目标值只有在整个预测时域结束之后才会到达，并且校正器保留的状态会随着通道数量的增加而增长。为了在保留较小状态的同时实现较大的误差降低，我们提出了 DRiC（DCT 岭回归校正器），它以递归更新的充分统计量，将下一个残差块的主离散余弦变换（DCT）系数对冻结预测和最新输入块的相应系数进行岭回归拟合。其局部变体 DRiC-L 为每个通道保留一个这样的回归；其全局变体 DRiC-G 将所有通道汇集到一个回归中，其统计量不依赖于通道数量。系统还设置了一个保护机制，跟踪冻结预测残差在每个通道上的运行尺度；对于远超该尺度的数据块，它会跳过更新，并对发出的校正进行裁剪。我们评估了……

    arXiv:2610.29604v1 Announce Type: new  Abstract: A frozen multi-horizon forecaster can be corrected online from its own residuals. However, under label delay the target of a block arrives only after the whole horizon has passed, and the corrector's retained state grows with the number of channels. Aiming to combine a small retained state with a large error reduction, we propose DRiC (DCT Ridge Corrector), which fits a ridge regression of the leading discrete cosine transform (DCT) coefficients of the next residual block on those of the frozen forecast and of the latest input block, with sufficient statistics updated recursively. Its local variant, DRiC-L, keeps one such regression per channel; its global variant, DRiC-G, pools all channels into one regression, whose statistics do not depend on the channel count. A guard tracks a running scale of the frozen forecast's residual for each channel; it skips updates on blocks far outside that scale and clips the issued correction. We evaluat
    
[^61]: 扩散语言模型在表格到文本生成中的忠实性诊断研究

    A Diagnostic Study of Faithfulness in Diffusion Language Models for Table-to-Text Generation

    [https://arxiv.org/abs/2610.29602](https://arxiv.org/abs/2610.29602)

    本研究诊断发现，扩散语言模型在表格到文本生成中随示例增加或微调后流畅度和BLEU提升，但对源表格的事实忠实性反而下降，揭示了表面流畅性与事实依据性的分离。

    

    扩散语言模型通过在双向注意力下对被遮蔽的序列进行迭代去噪来生成文本，而不受自回归解码那样严格的自左向右生成约束。这种差异是否会影响模型在结构化输入中保持可靠事实依据的能力，目前基本上尚未被探索。我们在表格到文本生成任务中使用ToTTo数据集研究这一问题，该数据集中高亮的源表格单元格使事实依据性可以直接被检验。在完全相同的基于检索的少样本协议下，我们将掩蔽扩散模型MDLM和LLaDA与GPT-2和GPT-Neo进行比较，同时还评估了经QLoRA适配的LLaDA，并将参考相似度与源表格事实依据性作为两个独立的维度进行评分。结果显示这两个维度出现了分离。在LLaDA内部，增加示例数量使BLEU从11.3提升至22.0，而相对于源表格的NLI蕴含分数却从0.78下降至0.52；QLoRA达到了本研究中最高的BLEU值24.15，但其事实依据性并未因此改善。人工标注揭示了失败模式……（原文摘要在此处截断）

    arXiv:2610.29602v1 Announce Type: new  Abstract: Diffusion language models generate by iteratively denoising a masked sequence under bidirectional attention, without the strict left-to-right commitment of autoregressive decoding. Whether this difference affects how reliably they remain grounded in structured inputs is largely unexplored. We study this question in table-to-text generation using ToTTo, where highlighted source cells make factual grounding directly testable. We compare the masked diffusion models MDLM and LLaDA with GPT-2 and GPT-Neo under an identical retrieval-based few-shot protocol, together with a QLoRA-adapted LLaDA, and we score reference similarity and source grounding as separate axes. The two pull apart. Within LLaDA, additional demonstrations raise BLEU from 11.3 to 22.0 while NLI entailment against the source table falls from 0.78 to 0.52; QLoRA reaches the best BLEU of the study, 24.15, and grounds no better for it. Manual annotation shows the failure modes d
    
[^62]: 分位数分布强化学习的近极小极大速率

    Near Minimax Rate for Quantile Distributional Reinforcement Learning

    [https://arxiv.org/abs/2610.29595](https://arxiv.org/abs/2610.29595)

    本文在生成模型假设下，为分位数Bellman不动点估计建立了在所有分辨率和所有Wasserstein阶数上一致成立的 $\sqrt{n}$ 高概率误差界，达到了分位数分布策略评估的近极小极大最优速率。

    

    本文研究分位数分布策略评估问题，其中给定策略下的回报分布由分位数Bellman不动点 $\bm\eta_m$ 来近似。我们在Wasserstein距离下，研究了跨不同分位数分辨率估计 $\bm\eta_m$ 的统计精度。假设可以访问生成模型，我们从经验马尔可夫决策过程构建了分位数不动点估计量 $\bm\eta_m^{(n)}$，并建立了Wasserstein距离下的非渐近误差界。当奖励密度一致下有界（即一致远离零）时，我们获得了在所有分辨率 $m$ 和所有Wasserstein阶数 $1\le p\le\infty$ 上同时成立的 $\sqrt{n}$ 阶高概率界。当奖励密度在支撑集端点处趋近于零时，我们证明仅需连续性和端点单调性即可在 $1\le p\le2$ 时保持相同的与分辨率无关的速率。对于 $p>2$，我们建立了带有系数的 $\sqrt{n}$ 阶界……

    arXiv:2610.29595v1 Announce Type: cross  Abstract: In this paper, we study quantile distributional policy evaluation, where the return distribution under a given policy is approximated by the quantile Bellman fixed point $\bm\eta_m$. We investigate the statistical accuracy of estimating $\bm\eta_m$ across quantile resolutions under Wasserstein distances. Assuming access to a generative model, we construct a quantile fixed point estimator $\bm\eta_m^{(n)}$ from an empirical Markov decision process and establish non-asymptotic error bounds under Wasserstein distances. When reward densities are uniformly bounded away from zero, we obtain a root-$n$ high-probability bound simultaneously over all resolutions $m$ and all Wasserstein orders $1\le p\le\infty$. When reward densities vanish at the support endpoints, we show that continuity and endpoint monotonicity suffice to preserve the same resolution-independent rate for $1\le p\le2$. For $p>2$, we establish a root-$n$ bound with coefficient
    
[^63]: 一种基于迁移学习的高效可持续可见光通信室内定位方法

    A Transfer Learning-Driven Methodology for Efficient and Sustainable Indoor Localization Using VLC

    [https://arxiv.org/abs/2610.29594](https://arxiv.org/abs/2610.29594)

    本文提出一种基于迁移学习的可见光通信室内定位方法，能够有效应对光照波动和物理障碍等环境变化，同时显著降低计算开销与能耗。

    

    物联网的快速扩张使得联网设备之间能够无缝协作，室内定位因此成为工业自动化的关键组成部分。然而，传统定位技术在动态环境中表现不佳。这些方法依赖于大量且频繁的数据收集以及针对特定环境的校准，限制了可扩展性和互操作性，也未能有效利用已有的研究成果，同时带来显著的能耗代价，阻碍了实际部署。为应对这些挑战，我们提出了一种基于迁移学习（TL）的节能方法，用于基于可见光通信的室内定位。我们的方法利用迁移学习来应对环境变化，包括光照波动和物理障碍物等通常会降低定位性能的因素，同时与传统方法相比，降低了计算开销和能耗。我们还……

    arXiv:2610.29594v1 Announce Type: new  Abstract: The rapid expansion of the Internet of Things enables seamless collaboration among connected devices, making indoor localization a critical component of industrial automation. However, conventional localization techniques struggle in dynamic environments. These methods depend on extensive, frequent data collection and environment-specific calibration, limiting scalability, interoperability, and effective use of prior research while imposing significant energy penalties that hinder practical deployment. To address these challenges, we propose an energy-efficient transfer learning (TL) based approach for visible light communication based indoor localization. Our method leverages TL to tackle environmental variability, including lighting fluctuations and physical obstacles that typically degrade localization performance, while simultaneously reducing computational overhead and energy consumption compared to conventional approaches. We also 
    
[^64]: 基于片段级图编码的可解释分子性质预测

    Fragment-Level Graph Encoding for Interpretable Molecular Property Prediction

    [https://arxiv.org/abs/2610.29589](https://arxiv.org/abs/2610.29589)

    该论文提出用BRICS分解将分子编码为片段级图取代原子级图，在不改变模型结构的前提下使GNN能够给出功能团级别的可解释预测，并在约五分之一大小的图上达到与原子级编码相当的预测精度。

    

    图神经网络（GNN）是分子性质预测的标准工具之一，但其预测的原子级解释对药物化学家而言难以直接采用。我们提出改变输入表示而非设计新模型：将分子通过BRICS分解编码为片段级图，其中每个节点是一个具有合成意义的片段，其特征为其原子特征的逐元素求和。由于仅改变了表示方式，任何消息传递GNN和任何事后归因方法都能在功能团分辨率而非单个原子的层面上产生解释。在骨架划分下，跨越四种架构、四个预测任务和两种解释器的实验表明，片段级编码在约五分之一大小的图上达到了与原子级相同的测试精度。积分梯度归因在两个酶靶点上以高于随机水平的准确率恢复了已知药效团，其中在BACE上达到1.5倍的……

    arXiv:2610.29589v1 Announce Type: new  Abstract: Graph neural networks (GNNs) are among the standard tools for molecular property prediction, but the atom-level explanations of their predictions are hard for medicinal chemists to act on. We propose a change of input representation rather than a new model: molecules are encoded as fragment-level graphs from BRICS decomposition, where each node is a synthetically meaningful fragment whose features are the element-wise sum of its atoms' features. Because only the representation changes, any message-passing GNN and any post-hoc attribution method produces explanations at the resolution of functional groups instead of single atoms. Across four architectures, four prediction tasks, and two explainers under scaffold splits, fragment-level encoding matches atom-level test accuracy on roughly five-fold smaller graphs. Integrated-Gradients attributions recover known pharmacophores above chance on two enzyme targets, namely BACE at 1.5 times the 
    
[^65]: 通过信噪比调节解锁流匹配模型中的空间接地能力

    Unlocking Spatial Grounding in Flow-Matching Models via SNR Tuning

    [https://arxiv.org/abs/2610.29581](https://arxiv.org/abs/2610.29581)

    该论文发现信噪比（SNR）分布是流匹配模型空间接地能力的关键决定因素，指出高噪声区间是空间学习的“关键学习阶段”，通过调节SNR训练分布即可显著提升模型对空间布局的精细控制，而无需额外的结构适配器或复杂的推理时引导。

    

    尽管基础模型在图像合成领域取得了成功，但实现细粒度的空间控制仍然是一个重大挑战，这通常需要辅助的结构适配器或复杂的推理时引导。虽然现代视觉-语言模型（VLM）文本编码器具备表示空间坐标的潜在能力，但生成模型往往未能利用这些信息。我们证明，信噪比（SNR）分布是流匹配模型中空间接地的关键决定因素。借助一种新颖的几何基元合成基准，我们将结构机制与自然图像固有的纹理复杂性分离开来，并引入一种严格的颜色感知布局度量来量化空间保真度。我们表明，在居中的SNR分布上进行训练会导致较差的布局遵从性（25.9%）。相反，我们发现高噪声区间是空间接地的“关键学习阶段”。将……（摘要在此处截断）

    arXiv:2610.29581v1 Announce Type: cross  Abstract: Despite the success of foundation models in image synthesis, achieving fine-grained spatial control remains a significant challenge, typically requiring auxiliary structural adapters or complex inference-time steering. While modern vision-language (VLM) text encoders possess the latent capacity to represent spatial coordinates, generative models often fail to leverage this information. We demonstrate that the signal-to-noise ratio (SNR) distribution is a critical determinant of spatial grounding in flow-matching models. Using a novel synthetic benchmark of geometric primitives, we isolate structural mechanics from textural complexity inherent in natural imagery and introduce a rigorous color-aware layout metric to quantify spatial fidelity. We show that training on a centered SNR distribution results in poor layout adherence (25.9%). Instead, we identify high-noise regimes as a "critical learning phase" for spatial grounding. Shifting 
    
[^66]: 基于线性模型的快速贝叶斯优化

    Fast Bayesian Optimization Through Linear Models

    [https://arxiv.org/abs/2610.29580](https://arxiv.org/abs/2610.29580)

    本文提出了利用现有贝叶斯优化算法线性结构的新型快速BO方法，在保持与最先进算法相当性能的同时将速度提升100至1000倍以上，从而将贝叶斯优化的适用范围拓展到成本敏感的全新问题类别。

    

    贝叶斯优化（BO）通常仅用于那些底层函数被认为“昂贵”（例如在时间或金钱上）的问题，此时BO算法本身的运行速度并不重要。然而在现实世界中，问题的成本跨越一个从便宜到昂贵的连续范围。在本文中，我们通过速度-性能帕累托前沿的视角来审视贝叶斯优化：我们将BO与CMA-ES等更快的算法进行比较，并指出帕累托前沿的某些区域目前是空白的。作为解决方案，我们提出了一组新的快速BO方法，这些方法利用了现有BO算法中的线性结构，适用于连续输入和组合输入（维度从约10维到6000维以上）。我们的方法实现了与最先进BO相似的性能，同时速度快了10²倍到10³倍以上，为贝叶斯优化领域开辟了一类全新的问题。

    arXiv:2610.29580v1 Announce Type: new  Abstract: Bayesian optimization (BO) is typically used only for problems where the underlying function is deemed "expensive" (e.g. in time or money), with the speed of the BO algorithm being of little importance. In the real world, however, problems span a continuous range of values, going from cheap to expensive. In this paper, we present BO algorithms through the lens of a speed-performance Pareto frontier: we compare BO against faster algorithms like CMA-ES, and highlight how certain regions of the Pareto frontier are currently empty. As a solution, we propose a new set of fast BO methods that exploit the linear structure in existing BO algorithms, for both continuous and combinatorial inputs (from ~10 to 6000+ dimensions). Our methods achieve performance similar to state-of-the-art BO, while being more than $10^2\times$ to $10^3\times$ faster, opening the field of BO to an entirely new class of problems.
    
[^67]: 预测市场合约在上线时能揭示什么：交易量与争议风险

    What Prediction-Market Contracts Reveal at Launch: Trading Volume and Dispute Risk

    [https://arxiv.org/abs/2610.29578](https://arxiv.org/abs/2610.29578)

    仅凭预测市场合约上线时的可解释信息（如市场主题、措辞和结算规则）即可高度准确地预测其未来交易量与结算争议风险，但这种预测能力会随平台环境和上线格式的变化而显著衰减。

    

    基于Polymarket上2023年1月至2026年5月间802,863个已结算市场的近乎全量数据，仅使用上线时可获得的信息（市场的标题与描述、指定的结算来源以及合约元数据），我们预测其生命周期内的交易量以及结算日后是否会被争议。合约信息能很好地预测交易量（对数交易量的样本外R²=0.747），且大部分信号是可解释的：市场主题贡献了最大增益（相比0.220的合约特征基线，R²提升+0.289），措辞与结算规则设计带来较小但可测量的改进（合计+0.027），而神经文本模型在可解释特征之外几乎没有额外增益。这一性能无法在伴随上线格式转变的平台变革期中延续：预测市场交易量是否达到1,000美元的准确率从同期评估下的AUC 0.940下降至跨期训练评估时的0.577（原文在此处截断）。

    arXiv:2610.29578v1 Announce Type: cross  Abstract: Using a near-census of 802,863 resolved markets listed on Polymarket between January 2023 and May 2026, and only information available at listing (the market's title and description, named resolution source, and contract metadata), we predict lifetime trading volume and whether settlement is later disputed. Contract information predicts trading volume well (out-of-sample $R^2 = 0.747$ for log volume), and most of the signal is interpretable: market topic provides the largest gain ($+0.289$ $R^2$ over a $0.220$ contract-feature baseline), wording and resolution-rule design add smaller but measurable improvements ($+0.027$ combined), and neural text models add little beyond the interpretable features. This performance does not survive a period of platform change coinciding with a shift in listing formats: accuracy in predicting whether a market reaches \$1,000 falls from AUC $0.940$ under same-period evaluation to $0.577$ when training o
    
[^68]: 反身性阈值：多智能体操演性预测中的相变

    The Reflexivity Threshold: A Phase Transition for Multi-Agent Performative Prediction

    [https://arxiv.org/abs/2610.29577](https://arxiv.org/abs/2610.29577)

    该论文提出用“反身性矩阵”刻画多智能体操演性预测中跨智能体反馈的网络结构，并证明当该矩阵的谱半径越过 1 这一阈值时，重复再训练会发生从稳定收敛到最坏情况学习受阻的相变。

    

    许多学习系统具有操演性：一旦部署，其预测会改变未来模型训练所依赖的数据分布。在多智能体场景中，这种反馈通过一个由相互作用的学习者构成的网络进行传递，因此稳定性不仅取决于操演性的强弱，还取决于谁影响谁。我们研究多智能体操演性预测中的重复再训练问题，并识别出一个谱阈值，它区分了稳定学习与最坏情况下的学习受阻。先前的工作往往将跨智能体的反馈压缩为一个标量收缩常数，从而忽略了这种网络结构。我们转而引入反身性矩阵 $\Gamma$，它记录了一个智能体的部署能在多大程度上移动另一个智能体的最优响应，该矩阵由各智能体的梯度敏感度和曲率参数定义。当该矩阵的谱半径跨越 1 时，阈值出现。低于阈值时，重复再训练收敛（摘要在此处截断）。

    arXiv:2610.29577v1 Announce Type: new  Abstract: Many learning systems are performative: once deployed, their predictions change the data distribution on which future models are trained. In multi-agent settings, this feedback is routed through a network of interacting learners, so stability depends not only on the strength of performativity but also on who affects whom. We study repeated retraining in multi-agent performative prediction and identify a spectral threshold that separates stable learning from worst-case learning obstruction. Prior work often collapses cross-agent feedback into a scalar contraction constant, missing this network structure. We instead introduce the reflexivity matrix $\Gamma$, which records how strongly one agent's deployment can move another agent's best response, defined from per-agent gradient sensitivities and curvature parameters. The threshold occurs when the spectral radius of this matrix crosses one. Below the threshold, repeated retraining converges
    
[^69]: 小损失下的差分隐私专家在线预测

    Private Online Prediction from Experts with Small Losses

    [https://arxiv.org/abs/2610.29575](https://arxiv.org/abs/2610.29575)

    该论文提出了首个遗憾度达到 $\tilde{O}(\sqrt{L^\star})$ 的差分隐私专家在线预测算法，在低损失情形下显著优于现有 $\tilde{O}(\sqrt{T})$ 界限，并证明在忘却输入下隐私代价很小。

    

    我们研究机器学习中的一个基本问题——差分隐私约束下的专家在线预测。现有的隐私保护算法实现了接近最优的遗憾度 $\tilde{O}(\sqrt{T})$，其中 $T$ 为时间范围。然而，当最优专家的累计损失 $L^\star$ 远小于 $T$ 时，该界限变得次优。在这项工作中，我们提出了首个遗憾度为 $\tilde{O}(\sqrt{L^\star})$ 的差分隐私算法，在低损失情形下带来了显著的改进。对于常数隐私参数，我们的遗憾度界限在多对数因子范围内与非隐私下界相匹配，表明在此设置下隐私只需付出很小的代价。我们的结果假设输入序列是忘却的，即预先固定且独立于算法的输出。我们还证明了对于自适应生成的输入，$\tilde{\Omega}(\sqrt{T})$ 的遗憾度是不可避免的。

    arXiv:2610.29575v1 Announce Type: new  Abstract: We consider online prediction from experts, a fundamental problem in machine learning, under differential privacy constraints. Existing private algorithms achieve near-optimal regret $\tilde{O}(\sqrt{T})$, where $T$ is the time horizon. However, this bound becomes suboptimal when $L^\star$, the cumulative loss of the best expert, is significantly less than $T$. In this work, we present the first differentially private algorithm with regret $\tilde{O}(\sqrt{L^\star})$, offering a substantial improvement in the low-loss regime. Our regret bound matches the non-private lower bound up to poly-logarithmic factors for constant privacy parameters, demonstrating that privacy incurs only a small cost in this setting. Our result assumes that the input sequence is oblivious, i.e., it is fixed in advance and independent of the algorithm's outputs. We also prove that an $\tilde{\Omega}(\sqrt{T})$ regret is inevitable for adaptively generated inputs, 
    
[^70]: 面向无标签图泛化的隐式特征重加权

    Implicit Feature Reweighting for Label-Free Graph Generalization

    [https://arxiv.org/abs/2610.29571](https://arxiv.org/abs/2610.29571)

    提出了无标签表示学习框架SCARCE，通过变分图自编码器与对抗混淆机制，仅利用训练图的环境划分即可学习到对分布偏移鲁棒的图表示，无需类别标签即实现图级别的分布外泛化。

    

    图级别的分布外泛化对于在分布偏移不可避免的真实世界应用中部署图神经网络至关重要。现有方法通常在表示学习过程中需要类别标签来判断哪些子结构具有预测性，这限制了它们的适用性。我们提出了SCARCE，这是一个无标签的表示学习框架，仅从训练图的环境划分中即可获得对分布偏移鲁棒的图表示，该划分既可以来自数据集元数据，也可以在没有元数据可用时通过对自监督嵌入进行聚类获得。我们的方法分为两个阶段：（1）预热启动变分图自编码器用于图重建；（2）对抗混淆机制迫使编码器产生判别器无法据此区分环境的表示，同时保持图重建的保真度。

    arXiv:2610.29571v1 Announce Type: new  Abstract: Graph-level out-of-distribution (OOD) generalization is critical for deploying graph neural networks in real-world applications where distribution shifts are inevitable. Existing approaches typically require class labels during representation learning to decide which substructures are predictive, limiting their applicability. We propose SCARCE, a label-free representation-learning framework that yields distribution-shift-robust graph representations from an environment partition of the training graphs alone, whether that partition comes from dataset metadata or, when none is available, from clustering a self-supervised embedding. Our method operates in two stages: (1) a variational graph autoencoder is warm-started for graph reconstruction; and (2) an adversarial confusion mechanism forces the encoder to produce representations from which a discriminator cannot distinguish environments, while preserving graph-reconstruction fidelity. We 
    
[^71]: 鲁棒非线性模型预测控制的认证神经近似

    Certified Neural Approximation of Robust Nonlinear Model Predictive Control

    [https://arxiv.org/abs/2610.29559](https://arxiv.org/abs/2610.29559)

    该论文提出一个框架，通过将近似误差作为有界输入扰动纳入鲁棒MPC设计，提供先验的构造性设计条件，使神经网络控制器在部署前即可继承模型预测控制的递归可行性与稳定性保证。

    

    模型预测控制是一种高性能的控制方法，但在许多控制任务中其在线计算成本相对较高，且难以在底层硬件环境中实现。为解决这一问题，我们提出了一个框架，用于生成能够以可证明的方式继承鲁棒模型预测策略的安全性和稳定性保证的神经网络控制器。与基于事后验证或额外在线安全过滤器的方法不同，我们的方法提供了构造性的、先验的设计条件，确保在部署之前递归可行性和稳定性得以保持。我们将近似误差视为有界输入扰动，并在MPC设计中加以处理。这使得神经网络控制器能够直接从底层模型预测控制器继承鲁棒性，从而将保证传递到神经近似中。该设计过程是一个离线的、……

    arXiv:2610.29559v1 Announce Type: cross  Abstract: Model predictive control is a high-performance control method, but it incurs comparatively high online computational costs across many control tasks and is challenging to realize in low-level hardware settings. To address this, we present a framework for generating neural network controllers that provably inherit the safety and stability guarantees of a robust model predictive policy. In contrast to approaches based on post-verification or additional online safety filters, our method provides constructive, a priori design conditions that ensure preservation of recursive feasibility and stability before deployment. We treat the approximation error as a bounded input disturbance, which is addressed in the MPC design. This allows the neural network controller to directly inherit robustness from the underlying model predictive controller, which enables the transfer of guarantees to the approximation. The design process is an offline, const
    
[^72]: SOAR：社交与观点感知的推荐系统

    SOAR: Social and Opinion-Aware Recommendations

    [https://arxiv.org/abs/2610.29556](https://arxiv.org/abs/2610.29556)

    提出了SOAR推荐模型，通过结合用户社交好友网络与文本评论的情感信息，实现更精准的个性化兴趣点推荐。

    

    在线平台面临的一个持久挑战是推荐问题。无论是社区还是广告商，在分析复杂的用户行为、识别用户习惯和不断变化的偏好，以及提供与用户兴趣相关的及时推荐方面都存在困难。特别是兴趣点推荐还面临额外的挑战：识别影响用户行为的因素、捕捉用户倾向，并为每个个体生成准确、个性化的推荐。在本工作中，我们提出了SOAR，一个社交与观点感知的推荐模型，该模型结合了用户的社交好友网络和文本评论来评估某个地点是否适合特定用户，并特别关注共享评论中的情感信息。在真实世界数据集上的实验结果表明，将评论纳入推荐过程能够提高推荐的精确度。

    arXiv:2610.29556v1 Announce Type: new  Abstract: One of the enduring challenges in online platforms is recommendation. Both communities and advertisers face difficulties in analyzing complex user behavior, identifying habits and evolving preferences, and providing timely recommendations that are relevant to user interests. In particular, point of interest recommendation poses the additional challenge of identifying the factors that influence user behavior, capturing user tendencies, and generating accurate, personalized recommendations for each individual. In this work, we propose SOAR, a Social and Opinion Aware Recommendation model that incorporates both the user's friendship network and textual reviews to assess whether a location is suitable for a specific user, with a particular emphasis on the sentiment of shared reviews. Experimental results in real-world datasets demonstrate that incorporating reviews into the recommendation process improves the precision of the recommendation.
    
[^73]: WRDO：对齐语言模型中的加权拒绝方向优化

    WRDO: Weighted Refusal Direction Optimization in Aligned Language Models

    [https://arxiv.org/abs/2610.29555](https://arxiv.org/abs/2610.29555)

    提出加权拒绝方向优化方法，通过学习聚合权重从针对每个提示的方向库中构建复合拒绝方向，以强化对齐语言模型的拒绝干预效果。

    

    对齐的大型语言模型被训练为遵守良性请求，同时拒绝有害请求。然而，最近的研究表明，这种拒绝行为是脆弱的，通常可以通过简单的激活空间或权重空间干预来破坏。特别是，已有研究表明，拒绝行为主要由残差流中的单一方向介导，该方向通常被计算为有害提示与无害提示之间的均值差向量。然而，这种标准构造方法将异质的有害提示聚合为一个统一的平均值，可能会混淆不同的与拒绝相关的现象。在本工作中，我们研究是否可以通过学习如何组合针对特定提示的拒绝方向来强化拒绝干预。我们提出了加权拒绝方向优化，该方法利用学习到的聚合权重，从一组针对每个提示的方向库中构建复合拒绝方向。这种方法允许不同……

    arXiv:2610.29555v1 Announce Type: cross  Abstract: Aligned large language models are trained to comply with benign requests while refusing harmful ones. However, recent work shows that this refusal behavior is fragile and can often be disrupted by simple activation- or weight-space interventions. In particular, refusal has been shown to be largely mediated by a single residual-stream direction, typically computed as a difference-of-means vector between harmful and harmless prompts. However, this standard construction aggregates heterogeneous harmful prompts into one uniform average, potentially conflating distinct refusal-related phenomena. In this work, we study whether refusal interventions can be strengthened by learning how to combine prompt-specific refusal directions. We propose Weighted Refusal Direction Optimization (WRDO), a procedure that constructs a composite refusal direction from a bank of per-prompt directions using learned aggregation weights. This approach allows diffe
    
[^74]: 用于后向兼容检索的超球面单纯形表示

    Hyperspherical Simplex Representations for Backward-Compatible Retrieval

    [https://arxiv.org/abs/2610.29550](https://arxiv.org/abs/2610.29550)

    提出由softmax输出和logits导出的超球面单纯形表示，利用softmax交叉熵训练所诱导的单纯形结构作为跨模型更新的共同几何参考，无需专门兼容训练、映射函数或依赖旧模型即可实现独立训练模型间的后向兼容检索。

    

    训练现代AI模型的成本日益高昂，而更新已部署的模型可能会改变其内部特征表示，从而影响基于这些模型构建的应用。在检索系统中，此类更新通常需要对图库进行完整的重新索引，这在大规模场景下是一个计算代价高昂的过程。现有的后向兼容方法能够直接比较更新后模型与早期模型的表示，从而避免重新索引。然而，这些方法通常需要专门的兼容性训练、学习到的映射函数或架构修改，并且往往显式依赖于先前的模型。我们提出由softmax输出和logits导出的超球面单纯形表示，用于在独立训练的模型之间实现后向兼容检索。这些表示利用由softmax交叉熵训练与独热标签所诱导的单纯形结构，作为跨模型更新的共同几何参考。

    arXiv:2610.29550v1 Announce Type: cross  Abstract: Training modern AI models has become increasingly expensive, and updating deployed models can alter their internal feature representations, affecting applications built upon them. In retrieval systems, such updates often require complete gallery re-indexing, a computationally expensive process at scale. Existing backward-compatibility methods enable direct comparison of representations from updated and earlier models, avoiding re-indexing. However, they typically require compatibility-specific training, learned mapping functions, or architectural modifications, and often depend explicitly on the previous model. We propose hyperspherical simplex representations derived from softmax outputs and logits for backward-compatible retrieval across independently trained models. They leverage the simplex structure induced by softmax cross-entropy training with one-hot labels as a common geometric reference across model updates. Since this refere
    
[^75]: DPO计算图的随机剪枝：1.4倍训练加速与更低峰值内存

    Stochastic Pruning of the DPO Computation Graph: 1.4x Faster Training and Lower Peak Memory

    [https://arxiv.org/abs/2610.29538](https://arxiv.org/abs/2610.29538)

    提出DPO计算图随机剪枝的统一框架，通过独立采样打分位置、执行前缀与反向传播分支，实现1.4倍训练加速并显著降低峰值内存。

    

    直接偏好优化（DPO）在输出打分、序列执行和反向传播中均会产生训练开销，因此仅减少监督信号并不一定能节省计算量。我们提出了一个DPO计算图随机剪枝的统一框架，该框架独立地采样用于打分的响应位置、用于执行的响应前缀以及用于反向传播的偏好分支。通过在相应操作之前做出这些选择，该框架在训练步骤的每个阶段都省略了相关计算。我们的分析刻画了位置剪枝和序列剪枝所引起的目标函数变化，并利用逆概率加权方法，为采样后的目标获得了条件无偏的单分支梯度估计器。受控测量实验表明，位置剪枝和序列剪枝主要能够降低峰值内存，其中大部分输出内存的节省归因于一种保留完整响应的实现方式。

    arXiv:2610.29538v1 Announce Type: new  Abstract: Direct preference optimization (DPO) incurs training costs in output scoring, sequence execution, and backpropagation, so reducing supervision alone does not necessarily save computation. We introduce a unified framework for stochastic pruning of the DPO computation graph that independently samples response positions for scoring, response prefixes for execution, and preference branches for backpropagation. By making these selections before the corresponding operations, the framework omits computation at each stage of a training step. Our analysis characterizes the objective changes induced by position and sequence pruning and uses inverse-probability weighting to obtain a conditionally unbiased single-branch gradient estimator for the sampled objective. Controlled measurements show that position and sequence pruning primarily reduce peak memory, with most output-memory savings attributable to an implementation that retains full response 
    
[^76]: 面向纵向电子健康记录的块扩散模型：精度、效率与表示学习的权衡

    Block Diffusion for Longitudinal EHR: Accuracy, Efficiency, and Representation Trade-offs

    [https://arxiv.org/abs/2610.29533](https://arxiv.org/abs/2610.29533)

    基于块的掩码扩散模型可作为自回归模型在纵向电子健康记录预测中的高效替代方案，小方块配置在保持接近AR预测性能的同时实现约1.43倍的推理加速，且学习到的临床表示信息量与AR相当。

    

    自回归（AR）模型已被证明在基于纵向电子健康记录进行零样本预测方面非常有效，但其对包含时间上聚集且相互依赖的临床事件的轨迹施加了严格的顺序生成方式。我们在MIMIC-IV数据集上，在匹配的分词方式、上下文长度、模型容量和评估条件下，研究了基于块的掩码扩散模型（BD）作为替代方案。在不同块大小的实验中，BD表现出明显的质量-效率权衡，小方块配置在保留AR大部分预测性能的同时，能够显著加快推理速度。一个具有代表性的B=8配置实现了约1.43倍的评估加速。尽管在大多数设置下生成式 rollout 仍不及AR，但在BD和AR表示上训练的冻结线性探针在各类临床预测任务中取得了几乎相同的性能，这表明BD学习到了具有可比信息量的临床表示。

    arXiv:2610.29533v1 Announce Type: new  Abstract: Autoregressive (AR) models have proved effective for zero-shot prediction from longitudinal electronic health records, but impose strictly sequential generation on trajectories containing temporally clustered and interdependent clinical events. We investigate block-based masked diffusion (BD) as an alternative on MIMIC-IV under matched tokenisation, context length, model capacity, and evaluation conditions. Across block sizes, BD exhibits a clear quality--efficiency trade-off, with small-block configurations retaining much of the predictive performance of AR while enabling substantially faster inference. A representative $B{=}8$ configuration achieves approximately $1.43\times$ faster evaluation. Although generative rollout remains below AR in most settings, frozen linear probes on BD and AR representations achieve near-identical performance across clinical prediction tasks, indicating that BD learns comparably informative clinical repre
    
[^77]: 由极值数据驱动的学习相变

    Phase Transitions in Learning Driven by Extremal Data

    [https://arxiv.org/abs/2610.29531](https://arxiv.org/abs/2610.29531)

    该论文通过研究多示例学习中的极值回归问题，推导出梯度流的低维投影，揭示了当softmax近似最大池化的锐度参数 $\lambda$ 增大使动力学进入梯度消失区域时，学习会发生相变，从而刻画了梯度下降成功与失效的条件。

    

    我们研究了一个极值回归问题，它是多示例学习（MIL）的一个实例，其中预测器以多重集的形式给出，称为“袋”（bag）。袋中的元素（称为“示例”，instance）具有隐藏标签，这些标签被池化以产生观测到的袋级标签。最大池化在MIL应用中很常见，但会导致梯度消失。为了解决这一问题，现代MIL流程采用softmax来近似最大值，并将锐度视为可学习参数，记为 $\lambda$。当最大池化模型是真实的数据生成过程时，学习会在梯度下降过程中增大 $\lambda$ 时取得成功，从而将动力学驱动到参数空间中梯度消失的区域。为了理解梯度下降在什么条件下会失效，我们研究了一个极简模型，并在袋大小 $n$ 与维度 $d$ 都很大的联合极限下（$\alpha=\ln...$），推导出了梯度流的低维投影。

    arXiv:2610.29531v1 Announce Type: new  Abstract: We study an extremal regression problem, an example of multiple instance learning (MIL), where the predictors are given as multisets, called bags. The bag elements (called instances) possess hidden labels which are pooled to produce an observed bag-level label. Max pooling is common in MIL applications, but gives rise to vanishing gradients. To handle this, modern MIL pipelines approximate the max with a softmax, treating the sharpness as a learnable parameter, denoted $\lambda$. When the max pooling model is the true data generating process, learning will be successful when $\lambda$ is increased during gradient descent, thereby driving the dynamics into regions of parameter space with vanishing gradients. To understand the conditions under which gradient descent breaks down, we study a minimal model and derive a low dimensional projection of the gradient flow in the joint limit of large bags $n$ and high dimension $d$ with $\alpha=\ln 
    
[^78]: 基于预训练上下文分位数回归的残差因果发现

    Residual-Based Causal Discovery via Pretrained In-Context Quantile Regression

    [https://arxiv.org/abs/2610.29530](https://arxiv.org/abs/2610.29530)

    该论文提出Loscar方法，将RESIT框架从加性噪声模型扩展至更广泛的位置-尺度噪声模型，通过分位数回归非参数化地估计条件均值和尺度，并支持将预训练的表格基础模型作为即插即用估计器，从而避免了现有方法对参数化假设的依赖。

    

    基于残差的因果发现假设每个变量由其父节点和一个独立的噪声源共同生成。该方法通过估计这一噪声并检验其与候选原因之间的独立性来推断因果关系。对于可识别的函数类别，如加性噪声模型（ANM）及更广泛的位置-尺度噪声模型（LSNM），该原理能够识别因果方向，因为只有正确的因果模型才满足这种独立性。现有的LSNM方法依赖参数化假设来获得标准化残差，存在模型误设的风险。相比之下，RESIT框架允许对残差进行非参数估计，从而迭代地识别汇节点，但该框架最初是为ANM提出的。为弥合这一差距，我们提出了Loscar，这是RESIT在LSNM上的扩展方法，通过分位数回归估计条件均值和条件尺度，使预训练的表格基础模型能够作为即插即用的估计器。我们将其与可行的广义最小二乘法进行对比……

    arXiv:2610.29530v1 Announce Type: cross  Abstract: Residual-based causal discovery assumes that each variable is generated from its parents and an independent noise source. It estimates this noise and tests its independence from candidate causes. For identifiable functions such as additive-noise models (ANMs) and the broader class of location-scale noise models (LSNMs), this principle identifies the causal direction, as only the causal model admits this independence. Existing LSNM methods rely on parametric assumptions to obtain standardized residuals, risking model misspecification. In contrast, the RESIT-framework allows for nonparametric estimation of residuals to iteratively identify sink nodes, but was proposed for ANMs. To close this gap, we propose Loscar, an extension of RESIT to LSNMs that estimates the conditional mean and scale via quantile regression, enabling pretrained tabular foundation models as plug-in estimators. We compare it with a feasible generalized least squares
    
[^79]: 联合嵌入预测目标对纵向电子健康记录是否有效？

    Are Joint-Embedding Predictive Objectives Effective for Longitudinal EHRs?

    [https://arxiv.org/abs/2610.29528](https://arxiv.org/abs/2610.29528)

    该研究表明，尽管JEPA方法在纵向电子健康记录的表示学习中能够稳定训练且不发生表示坍塌，但其性能始终不如标准的掩码语言建模，且减小预测目标规模也无法改善下游任务表现。

    

    掩码语言建模（MLM）是目前应用于纵向电子健康记录（EHR）的Transformer编码器的主流预训练目标。然而，重建单个事件可能会过度强调含噪声或信息量较弱的观测，且并不能明确地促进模型学习更高层次的患者状态。联合嵌入预测架构（JEPA）则通过在潜在空间中预测表示来进行学习，并已在多种领域展现出强大性能。本文研究了JEPA在多大程度上能够适配EHR序列表示。基于MIMIC-IV数据集的实证研究表明，尽管我们提出的三种JEPA公式都能稳定训练且不发生表示坍塌，但其表现始终不及标准MLM。我们发现，是预测目标的规模而非目标的异质性增加了预测误差，然而减小目标规模也无法恢复下游任务的性能。这表明……

    arXiv:2610.29528v1 Announce Type: new  Abstract: Masked language modelling (MLM) is the dominant pretraining objective for transformer encoders applied to longitudinal electronic health records (EHRs). However, reconstructing individual events may overemphasise noisy or weakly informative observations and does not explicitly encourage learning higher-level patient states. Joint-Embedding Predictive Architectures (JEPAs) instead learn by predicting representations in latent space and have shown strong performance in a variety of domains. In this paper, we investigate to what extent JEPA can be adapted to EHR sequence representation. Our empirical investigation using MIMIC-IV shows that although all three of our JEPA formulations train stably without representational collapse, they consistently underperform standard MLM. We find that target size, not target heterogeneity, increases prediction error, yet reducing the target size does not recover downstream performance. This suggests that 
    
[^80]: 几何知识空间、注意力配对与架构

    Geometric knowledge spaces, Attention pairing and Architectures

    [https://arxiv.org/abs/2610.29526](https://arxiv.org/abs/2610.29526)

    本文提出一个几何化框架，通过形式化“知识世界”与“注意力配对”的概念，统一推导出标准注意力、因果注意力、线性注意力、张量注意力和循环神经网络等多种架构，并引入新的几何空间以启发未来神经网络架构的设计。

    

    在本文中，我们提出了一种几何方法来推导注意力机制并探索潜在的新型transformer架构。我们首先形式化了“知识世界”和“注意力配对”的概念，然后展示了该框架如何自然地引出标准注意力、因果注意力、线性注意力、张量注意力以及循环神经网络的架构。最后，我们引入了若干新的几何空间，这些空间可能启发新型神经网络架构的设计。

    arXiv:2610.29526v1 Announce Type: new  Abstract: In this paper, we propose a geometric approach to deriving attention mechanisms and exploring potential new transformer architectures. We begin by formalizing the notions of knowledge worlds and attention pairings, then demonstrate how this framework naturally leads to the architectures of standard attention, causal attention, linear attention, tensor attention, and recurrent neural networks. Finally, we introduce several new geometric spaces that may inspire novel neural network architectures.
    
[^81]: pplx-embed-v2-late 技术报告

    pplx-embed-v2-late Technical Report

    [https://arxiv.org/abs/2610.29523](https://arxiv.org/abs/2610.29523)

    Perplexity 推出的 pplx-embed-v2-late 包含 0.6B 和 9B 两个多语言多模态后期交互模型，通过 18B 教师模型的表示蒸馏训练，在文本、图像和视觉文档检索上达到同规模最先进水平，并借助共享嵌入空间支持“9B 建索引、0.6B 低延迟查询”的非对称部署。

    

    在本报告中，我们介绍了 pplx-embed-v2-late，这是一对参数量分别为 0.6B 和 9B 的多语言、多模态后期交互（late-interaction）模型。两款模型在各自规模级别中，于一系列文本、图像和视觉文档检索基准测试上均取得了最先进的结果。在视觉文档检索任务上，pplx-embed-v2-late-0.6b 与比它大数倍的模型相比也具有竞争力。这些模型通过从一个 18B 参数的教师模型进行表示蒸馏而训练得到。因此，它们共享一个嵌入空间，支持非对称部署：可以使用 9B 模型对文档建立索引，并使用低延迟的 0.6B 模型进行查询。两款模型均可在 https://huggingface.co/collections/perplexity-ai/pplx-embed-v2 获取。

    arXiv:2610.29523v1 Announce Type: cross  Abstract: In this report, we introduce pplx-embed-v2-late, a pair of 0.6B and 9B multilingual and multimodal late-interaction models. Both models achieve state-of-the-art results within their respective size classes across a range of text, image, and visual document retrieval benchmarks. On visual document retrieval tasks, pplx-embed-v2-late-0.6b is competitive with models several times larger than itself. The models are trained via representation distillation from an 18B teacher. Consequently, they share an embedding space that supports asymmetric deployment: documents can be indexed with the 9B model and queried with the low-latency 0.6B model. Both models are available at https://huggingface.co/collections/perplexity-ai/pplx-embed-v2.
    
[^82]: 面向密集预测的监督自适应课程学习与受保护反课程学习

    Supervision-Adaptive Curriculum and Guarded Anti-Curriculum Learning for Dense Prediction

    [https://arxiv.org/abs/2610.29519](https://arxiv.org/abs/2610.29519)

    提出 PaceShift 方法，采用“先稳定、后挑战”的范式，将监督自适应课程选择与受保护的谱后训练相结合，通过区分监督可靠性与时间更新价值来提升密集预测的训练效果。

    

    课程学习按难度对样本进行排序，但仅凭难度并不能确立密集预测任务中监督信息的可靠性。简单样本可能暴露学习捷径，困难样本可能富有信息量，而噪声或模糊样本则可能误导训练。基于我们的会议论文方法 CurriSeg，我们提出了 PaceShift，这是一种“先稳定、后挑战”的范式，将监督自适应的课程选择与受保护的谱后训练相结合。基于学习动力学的视角，我们将监督可靠性与时间更新价值分离开来，并在模型稳定之后再施加谱域挑战。第一阶段从可用的监督信号、数据增广一致性、模型分歧和预测确定性来估计可靠性；时间统计量用于指导样本加权，同时像素级不确定性权重可降低不确定区域在训练早期的影响。第二阶段利用监督一致性对来自多个谱视角的预测进行排序（摘要在此处截断）。

    arXiv:2610.29519v1 Announce Type: cross  Abstract: Curriculum learning orders samples by difficulty, but difficulty alone does not establish supervision reliability for dense prediction. Easy samples can expose shortcuts, hard samples can be informative, and noisy or ambiguous samples can mislead training. Building on our conference method CurriSeg, we present PaceShift, a stabilize-then-challenge paradigm combining supervision-adaptive curriculum selection with guarded spectral post-training. A learning-dynamics view motivates separating supervision reliability from temporal update value and applying spectral challenges after stabilization. The first phase estimates reliability from available supervision, augmentation consistency, model disagreement, and prediction certainty. Temporal statistics guide sample weighting, while pixel-uncertainty weights reduce the early influence of uncertain regions. The second phase ranks predictions from multiple spectral views using supervision agree
    
[^83]: 利用上下文感知解码缓解思维链中的查询捷径问题

    Mitigating Query Shortcuts in Chain-of-Thought with Context-Aware Decoding

    [https://arxiv.org/abs/2610.29518](https://arxiv.org/abs/2610.29518)

    提出推理依据对比解码（RCD）方法，通过对比查询在有无推理依据时的词元分布来放大推理依据的作用，从而缓解LLM在思维链推理中依赖查询捷径的行为并提升答案准确率。

    

    虽然思维链提示提升了大型语言模型（LLM）的推理性能，但近期研究表明，最终答案并不总是依赖于所生成的推理依据。特别是，LLM可能依赖于针对查询本身的捷径，而非其推理步骤所提供的信息。在本文中，我们研究了上下文感知解码——一种已有的对比解码方法——在应用于思维链推理时能否缓解这种捷径行为。我们将这一应用称为推理依据对比解码（RCD），该方法对比查询在有无推理依据条件下预测的下一个词元分布，从而放大那些因推理依据而似然度增加的词元。我们通过KL正则化的奖励最大化目标对这一解码规则给出了理论解释。在五个推理基准上的实验表明，RCD相比标准解码方法提升了答案准确率。

    arXiv:2610.29518v1 Announce Type: new  Abstract: While Chain-of-Thought (CoT) prompting improves the reasoning performance of large language models (LLMs), recent studies have shown that final answers do not always depend on the generated rationales. In particular, LLMs may rely on query-specific shortcuts rather than the information provided by their reasoning steps. In this paper, we investigate whether context-aware decoding (CAD), an existing contrastive decoding method, can mitigate such shortcut behavior when applied to CoT reasoning. We study this application as Rationale Contrastive Decoding (RCD), which contrasts the next-token distributions conditioned on the query with and without its rationale, amplifying tokens whose likelihood increases due to the rationale. We provide a theoretical interpretation of this decoding rule via a KL-regularized reward-maximization objective. Experiments across five reasoning benchmarks demonstrate that RCD improves answer accuracy over standar
    
[^84]: BRIDGE：一种阻止模型更新中虚假改进的接受门控，附回顾性临床评估

    BRIDGE: An Acceptance Gate to Block False Improvements in Model Updates, with a Retrospective Clinical Evaluation

    [https://arxiv.org/abs/2610.29515](https://arxiv.org/abs/2610.29515)

    BRIDGE 提出了一种模型无关的接受门控机制，通过交叉拟合与 bootstrap 置信下界检验来拦截模型更新中的虚假改进，将虚假改进率从约 20% 降至接近零，并经 12 项回顾性临床任务验证有效。

    

    模型会被反复更新，而在验证数据上看似更好但在新数据上并非如此的更新是一种“虚假改进”，可能误导临床决策。朴素的规则会接受每一个正向的验证增益，而当同一批数据既用于选择又用于评判更新时，结果会过于乐观。BRIDGE 是一个模型无关的接受门控，它基于预测结果和可用标签、以固定的设置来决定是否接受每次更新。数据的每一折评判在其余折上选出的更新（交叉拟合），以减少选择偏差；接受条件要求增益的 bootstrap 置信下界超过一个裕度。实验涵盖四种设置：(A) 11 个新数据集，(B) 合成任务，(C) 12 项回顾性临床任务，以及 (D) 39 个支持性数据集。虚假改进率从朴素方法的 19.1% 降至 BRIDGE 的 1.1% (A)，从 20.7% 降至 1.7% (B)，从 20.0% 降至 0% (C)，从 12.5% 降至 0.5% (D)；所有关于数据集或任务上降低幅度的 95% 置信区间均不包含零。

    arXiv:2610.29515v1 Announce Type: new  Abstract: Models are updated repeatedly, and an update that looks better on validation data but not on new data is a false improvement that can misguide clinical decisions. The naive rule accepts every positive validation gain and is optimistic when the same data choose and judge the update. BRIDGE is a model-agnostic acceptance gate that decides each update from predictions and the available labels, with fixed settings. Each fold of the data judges the update chosen on the other folds (cross-fitting) to reduce selection bias; acceptance requires the gain's lower bootstrap bound to exceed a margin. The four settings are (A) 11 new datasets, (B) synthetic tasks, (C) 12 retrospective clinical tasks, and (D) 39 supporting datasets. The false-improvement rate falls from 19.1% (naive) to 1.1% (BRIDGE) in (A), 20.7% to 1.7% (B), 20.0% to 0% (C), and 12.5% to 0.5% (D); every 95% confidence interval of the reduction over datasets or tasks excludes zero (n
    
[^85]: 基于Stein算子的非归一化模型估计的统一方法

    A unified approach for estimation of unnormalized models via Stein operators

    [https://arxiv.org/abs/2610.29510](https://arxiv.org/abs/2610.29510)

    本文建立了一个连接适当评分规则与Stein方法的统一框架，证明分数匹配、比率匹配、伪似然和噪声对比估计等评分规则可自然诱导Stein算子，并借助由此产生的几何视角统一刻画了相应估计量的渐近效率。

    

    本文开发了一个统一的框架，将适当评分规则与Stein方法相联系，用于非归一化模型的估计。我们构建了一类适当评分规则，它涵盖了分数匹配、比率匹配、伪似然和噪声对比估计。我们证明了这些适当评分规则会自然地诱导出Stein算子。这一联系为Stein算子的构造提供了新的原理，并使我们能够通过所诱导的Stein算子相关的几何结构来研究基于这些评分规则的估计量。利用这一几何视角，我们对所得估计量的渐近效率给出了统一的刻画。

    arXiv:2610.29510v1 Announce Type: cross  Abstract: In this paper, we develop a unified framework connecting proper scoring rules and Stein's method for the estimation of unnormalized models. We construct a class of proper scoring rules that encompasses score matching, ratio matching, pseudolikelihood, and noise contrastive estimation. We show that these proper scoring rules naturally induce Stein operators. This connection yields a new construction principle for Stein operators, and enables us to study estimators based on these scoring rules through the geometry associated with the induced Stein operators. Using this geometric perspective, we provide a unified characterization of the asymptotic efficiency of the resulting estimators.
    
[^86]: 面向长尾识别的抗捷径CAM蒸馏

    Shortcut-Resistant CAM Distillation for Long-Tailed Recognition

    [https://arxiv.org/abs/2610.29508](https://arxiv.org/abs/2610.29508)

    提出抗捷径CAM蒸馏框架SRCD，通过将头部类别以物体为中心的类激活图解释蒸馏到尾部类别，有效缓解长尾识别中尾部类别因上下文相似而容易学习捷径的问题。

    

    真实世界的数据集通常遵循长尾分布，这使得对尾部类别的泛化变得困难。我们通过捷径学习的视角重新审视这一问题——在捷径学习中，模型倾向于选择最简单的预测线索（如背景或纹理）而非以物体为中心的语义，尤其是在监督信号稀缺且有偏的情况下。我们发现这种倾向在尾部类别上被进一步放大：有限的样本往往共享相似的上下文，使得非语义信号高度相关，从而诱使模型走捷径；而头部类别由于外观和环境多样，能够促进更稳定的以物体为中心的表示。基于这一观察，我们提出了抗捷径CAM蒸馏（SRCD），这是一个即插即用的框架，能够将头部类别中以物体为中心的解释迁移到尾部类别。SRCD在类激活图（CAM）空间中运行，CAM为预测提供特定类别的空间证据图。

    arXiv:2610.29508v1 Announce Type: new  Abstract: Real-world datasets often follow a long-tailed distribution, making generalization to tail classes difficult. We revisit this problem through the lens of shortcut learning, where models prefer the easiest predictive cues (e.g., background or textures) over object-centric semantics, especially under scarce and biased supervision. We find that this tendency is amplified for tail classes: limited examples often share similar contexts, making non-semantic signals highly correlated and thus tempting shortcuts, whereas head classes with diverse appearances and environments encourage more stable object-focused representations. Motivated by this observation, we propose Shortcut-Resistant CAM Distillation (SRCD), a plug-and-play framework that transfers object-focused explanations from head to tail classes. SRCD operates in the Class Activation Map (CAM) space, where a CAM provides a class-specific spatial evidence map for a prediction. SRCD aggr
    
[^87]: 异步、无代际的ABC：面向HPC上异构模拟器工作负载的流式AMIS重加权方法

    Asynchronous, Generation-Free ABC: Streaming AMIS Reweighting for Heterogeneous Simulator Workloads on HPC

    [https://arxiv.org/abs/2610.29506](https://arxiv.org/abs/2610.29506)

    该论文提出了一种无代际的异步ABC采样器，通过在模拟结果到达时即时更新提议分布，并利用流式AMIS重加权复用全部历史粒子，消除了并行计算中的同步屏障，显著提升了异构模拟器工作负载在高性能计算环境下的吞吐量。

    

    近似贝叶斯计算（ABC）通过对模拟数据进行参数后验分布估计，使具有难以处理的似然函数的模型也能进行贝叶斯推断。序贯ABC方法通过迭代地将提议分布向高后验概率区域调整来提高采样效率。然而，这些方法通常以“代”为单位运行，每一代必须等待其最慢的模拟完成后才能开始下一代。在并行机器上，这种同步屏障会导致工作节点闲置并降低吞吐量，尤其是当模拟运行时间差异很大时。我们提出了一种无代际的ABC采样器，它在模拟结果到达时即更新提议分布，并向每个可用的工作节点发送新的候选样本。该提议分布基于迄今为止评估出的最佳粒子。为了估计后验分布，我们复用完整的历史记录，并基于重构的采样混合分布对粒子进行流式重加权。

    arXiv:2610.29506v1 Announce Type: cross  Abstract: Approximate Bayesian computation (ABC) enables Bayesian inference for models with intractable likelihoods by estimating parameter posterior distributions from simulated data. Sequential ABC methods improve sampling efficiency by iteratively adapting the proposal distribution toward regions of high posterior probability. However, these methods typically operate in generations, where each generation must wait for its slowest simulation before the next can begin. On a parallel machine, this synchronization barrier leaves workers idle and reduces throughput, particularly when simulation runtimes vary substantially. We introduce a generation-free ABC sampler that updates its proposal when simulation results arrive and sends a new candidate to each available worker. The proposal is based on the best particles evaluated so far. To estimate the posterior, we reuse the full history and reweight particles against a reconstructed mixture of sampl
    
[^88]: 重新思考用于时间序列预测的智能体模型编排

    Rethinking Agentic Model Orchestration for Time Series Forecasting

    [https://arxiv.org/abs/2610.29504](https://arxiv.org/abs/2610.29504)

    该论文提出FairAMO受控评估协议，揭示智能体模型编排（AMO）在时间序列预测中的收益本质上受限于候选基础模型本身，其关键在于模型互补性而非数量，且相对最佳单一模型的MASE改进不足6%。

    

    时间序列基础模型（TSFMs）在不同预测任务上的表现参差不齐。近期研究采用智能体模型编排（AMO）来选择或组合TSFMs，其效果往往优于单个预测器。然而，这些收益可能源于更强或更具互补性的候选模型，而非更好的编排本身。为了厘清这两种效应，我们提出了FairAMO，一种采用冻结候选预测的受控评估协议。在GIFT-Eval上的实验表明，AMO确实能提升预测准确性，但其收益受限于候选TSFMs的上限：在所有任务中，最佳编排器相对于最佳单个TSFM的MASE降低幅度不到6%。这些收益取决于模型间的互补性（即不同候选模型在不同任务上各有所长），而非候选模型的数量。增加成员会带来边际收益递减，而一个冗余成员甚至可能抵消全部收益。此外，历史证据的价值也很快达到饱和——仅需几个近期时间窗口即可捕获全部收益。

    arXiv:2610.29504v1 Announce Type: new  Abstract: Time series foundation models (TSFMs) perform unevenly across forecasting tasks. Recent work uses agentic model orchestration (AMO) to select or combine TSFMs, often outperforming individual forecasters. Yet these gains may reflect stronger or more complementary candidates rather than better orchestration. To disentangle these effects, we introduce FairAMO, a controlled evaluation protocol with frozen candidate forecasts. Experiments on GIFT-Eval show that AMO improves accuracy, but its gains are bounded by the candidate TSFMs. The best orchestrators reduce MASE by less than 6% relative to the best single TSFM across all tasks. These gains depend on complementarity (different candidates excel on different tasks) rather than the number of candidates. Adding members yields diminishing returns, and a redundant member can erase these gains. The value of historical evidence also saturates quickly. A few recent windows capture all gains, and f
    
[^89]: 基于非线性薛定谔桥与分支扩散的生成建模

    Generative Modeling via Nonlinear Schr\"odinger Bridges and Branching Diffusions

    [https://arxiv.org/abs/2610.29476](https://arxiv.org/abs/2610.29476)

    该论文提出了基于前向-后向偏微分方程的非线性薛定谔桥生成建模框架SBNL，并利用分支扩散设计出具有高度并行性和高维可扩展性的高效数值算法。

    

    我们提出了一个基于前向-后向偏微分方程（PDE）的全新生成建模框架。这些方程自然地产生于平均场博弈（MFG）理论中，并与非线性薛定谔系统密切相关。我们的主要贡献是在生成建模的背景下重新解释这一联系，并为其实现开发高效的数值方法。具体而言，我们通过一个耦合的偏微分方程组推导并刻画了经典薛定谔桥问题的非线性扩展，由此得到的模型我们称之为SBNL（非线性薛定谔桥）。为计算其解，我们设计了一种基于分支扩散和半线性偏微分方程概率表示的模拟算法。该方法具有“尴尬并行”（embarrassingly parallel）的特性，可扩展至高维情形，并适用于一大类非线性相互作用泛函。我们通过……

    arXiv:2610.29476v1 Announce Type: new  Abstract: We propose a new generative modeling framework grounded in forward-backward partial differential equations (PDEs). These equations arise naturally in mean-field game (MFG) theory, and are closely related to nonlinear Schr\"odinger systems. Our main contribution is to reinterpret this connection within the setting of generative modeling and to develop an efficient numerical methodology for its practical implementation. Specifically, we derive and characterize a nonlinear extension of the classical Schr\"odinger bridge problem through a coupled system of PDEs, yielding a model we call SBNL (Schr\"odinger Bridge NonLinear). To compute its solution, we design a simulation algorithm based on branching diffusions and the probabilistic representation of semilinear PDEs. The resulting method is embarrassingly parallel, scalable to high dimensions, and applicable to a broad class of nonlinear interaction functionals. We validate the approach thro
    
[^90]: 面向低NFE生成流的密度层面后向误差

    Density-Level Backward Error for Low-NFE Generative Flows

    [https://arxiv.org/abs/2610.29475](https://arxiv.org/abs/2610.29475)

    本文提出密度层面的后向误差框架，分离出真正改变边缘分布的求解器误差分量，从而为低NFE生成流导出感知求解器阶数的计算分配规则以及基于规范自由度的速度场选择目标。

    

    低NFE（网络函数评估次数）生成式ODE采样不仅受求解器阶数的限制，还受固定的网络评估预算如何沿流分布的影响。我们开发了一个密度层面的后向误差框架，该框架能够识别求解器修正方程缺陷中会改变演化边缘分布的分量，同时排除在Wasserstein切几何中不可见的螺线轨迹畸变。所得残差诱导出一种感知求解器阶数的计算分配规则，并借助规范自由度，提供了一个用于从同一密度路径的多个速度代表中选择数值有利者的目标函数。这将我们的方法与由轨迹或端点替代量驱动的调度搜索以及高阶求解器设计区分开来。精确的高斯模型和混合模型确立了该机制，包括基于曲率的分配失效以及连续最优传输并非离散化最优的情形。在预训练的CIFAR-10

    arXiv:2610.29475v1 Announce Type: new  Abstract: Low-NFE generative ODE sampling is constrained not only by solver order, but also by how a fixed budget of network evaluations is distributed along the flow. We develop a density-level backward-error framework that identifies the component of a solver's modified-equation defect that changes the evolving marginal, while excluding solenoidal trajectory distortion that is invisible in Wasserstein tangent geometry. The resulting residual induces a solver-order-aware compute-allocation rule and, through gauge freedom, an objective for selecting numerically favorable velocity representatives of the same density path. This separates our approach from schedule search driven by trajectory or endpoint surrogates and from higher-order solver design. Exact Gaussian and mixture models establish the mechanism, including cases where curvature-based allocation fails and continuum-optimal transport is not discretization-optimal. On a pretrained CIFAR-10 
    
[^91]: 超越自适应等变性松弛的非对称性隧穿

    Asymmetry Tunneling Beyond Adaptive Equivariance Relaxation

    [https://arxiv.org/abs/2610.29473](https://arxiv.org/abs/2610.29473)

    该研究发现，即使刻意限制严格等变模型与残差路径之间的接口，打破对称性的任务相关结构仍能通过残差路径“隧穿”进入模型（即“非对称隧穿”现象），表明等变性松弛未必需要自适应控制。

    

    等变神经网络将预设的对称性编码为强大的归纳偏置，然而现实世界的数据可能表现出这些先验无法完全捕捉的对称性破缺结构。先前的工作主要通过模型侧的等变性松弛来解决这一挑战，即为数据寻找合适的等变性水平。这引出了一个互补的问题：等变性松弛是否必须进行自适应控制？为了研究这一问题，我们在严格等变模型旁引入了一条残差路径，并刻意限制二者之间的接口，以考察任务相关结构的传递情况。在五个基准测试中，我们观察到残差在这些限制条件下既会出现坍缩，也能够存活；我们将后一种现象称为“非对称隧穿”。我们进一步通过四种互补的诊断方法对这一现象进行了刻画。我们的结果表明，等变性松弛的自适应控制并不是……的先决条件（摘要原文在此处被截断）。

    arXiv:2610.29473v1 Announce Type: new  Abstract: Equivariant neural networks encode prescribed symmetries as powerful inductive biases, yet real-world data may exhibit symmetry-breaking structure that these priors cannot fully capture. Prior work has largely addressed this challenge through model-side equivariance relaxation, seeking an appropriate level of equivariance for the data. This raises a complementary question: must equivariance relaxation be adaptively controlled? To investigate this question, we introduce a residual path alongside a strictly equivariant model and deliberately restrict their interface to examine the transmission of task-relevant structure. Across five benchmarks, we observe both residual collapse and survival despite these restrictions; we call the latter phenomenon asymmetry tunneling. We further characterize this phenomenon through four complementary diagnostics. Our results suggest that adaptive control of equivariance relaxation is not a prerequisite for
    
[^92]: OmniSimulator：对齐小型语言模型以实现真实的异构行为建模

    OmniSimulator: Aligning Small Language Models for Authentic Heterogeneous Behavior Modeling

    [https://arxiv.org/abs/2610.29472](https://arxiv.org/abs/2610.29472)

    本文提出 OmniSimulator 训练框架，通过五维结构化行为推理模式、精选真实用户行为数据以及教师模型生成的高质量推理轨迹，对齐小型语言模型，从而显著提升其在短视频平台上进行真实异构用户行为模拟的预测能力。

    

    大型语言模型（LLM）作为交互系统的通用用户模拟器已展现出巨大潜力。先前的研究表明，在高质量数据上进行训练能够显著提升模型的推理能力和下游任务表现。然而，用户模拟的上限仍受制于模型自身的推理范式：原生的思维链往往退化为浅层的模式匹配，无法充分建模真实用户决策背后的行为因素。在本文中，我们提出了 OmniSimulator，一种新颖的训练框架，旨在提升小型语言模型在短视频平台环境中进行真实用户模拟的预测能力。具体而言，OmniSimulator 通过一个包含五个维度的结构化模式来组织面向行为的推理，从真实用户行为场景中精选数据，并引入教师模型来生成高质量的推理轨迹。

    arXiv:2610.29472v1 Announce Type: new  Abstract: Large language models (LLMs) have shown great potential as general-purpose user simulators for interactive systems. Prior studies have demonstrated that training on high-quality data can substantially improve model reasoning and downstream task performance. However, the upper bound of user simulation is still constrained by the model's reasoning paradigm: native chain-of-thought often degenerates into shallow pattern matching and fails to adequately model the behavioral factors underlying real user decisions. In this paper, we propose OmniSimulator, a novel training framework designed to improve the predictive ability of small LLMs for realistic user simulation on short-video platform environments. Specifically, OmniSimulator organizes behavior-oriented reasoning through a structured schema with five dimensions, carefully selects data from real user-behavior scenarios, and introduces a teacher model to generate high-quality reasoning tra
    
[^93]: 面向可泛化智能体强化学习的框架扩展

    Harness Scaling for Generalizable Agentic Reinforcement Learning

    [https://arxiv.org/abs/2610.29458](https://arxiv.org/abs/2610.29458)

    提出Harness Scaling这一智能体强化学习的新扩展维度，将训练从单一固定框架扩展到多样化执行配置，从而提升模型在不同框架配置下的泛化与迁移能力。

    

    智能体框架已成为现代智能体系统的核心组成部分，塑造着模型使用工具、管理上下文以及协调长时程交互的方式。要充分发挥这些框架的收益，取决于底层模型利用它们的有效程度，这促使人们直接通过框架进行强化学习。然而，现有方法通常仅在单一固定框架下进行训练，这会导致模型过度专门化，并在不同框架配置之间的迁移能力受限。为解决这一问题，我们提出了Harness Scaling（框架扩展），这是智能体强化学习的一个全新扩展维度，将训练从单一框架扩展到多样化的执行配置。具体而言，我们构建了一个因子化的框架空间，选取具有均衡因子覆盖的紧凑训练子集，并组织跨框架训练，将全局覆盖与同一任务的可控变化相结合。在多个基准测试、模型规模以及（摘要在此处被截断）

    arXiv:2610.29458v1 Announce Type: cross  Abstract: Agent harnesses have become a central component of modern agentic systems, shaping how models use tools, manage context, and coordinate long-horizon interaction. Realizing the benefits of these harnesses depends on how effectively the underlying model can make use of them, motivating reinforcement learning directly through the harness. However, existing approaches typically train through a single fixed harness, which can lead to specialization and limited transfer across harness configurations. To address this, we introduce \textbf{Harness Scaling}, a new scaling dimension for agentic RL that extends training from a single harness to diverse execution configurations. Concretely, we construct a factorized harness space, select a compact training subset with balanced factorial coverage, and organize cross-harness training to combine global coverage with controlled variation for the same task. Across multiple benchmarks, model scales, and
    
[^94]: 基础模型可以被蒸馏到何种程度以实现高效的喷注标记？

    How Far Can Foundation Models Be Distilled for Efficient Jet Tagging?

    [https://arxiv.org/abs/2610.29439](https://arxiv.org/abs/2610.29439)

    本研究系统地将六个教师模型（包括两个基础模型）蒸馏到小型学生模型中用于喷注标记，发现每位教师都能提升每位学生且增益主要取决于教师强弱，而仅两万参数的蒸馏版L-GATr-slim即可超越所有已发表的从头训练的喷注标记器。

    

    喷注标记是大型强子对撞机上机器学习最重要的应用之一。近期的模型要么构建了更多的物理知识，要么依赖于大规模预训练，但实验对其推理成本有严格的限制。知识蒸馏可以将这些模型的性能迁移到运行成本低廉的小型学生模型中。我们探讨了教师的哪些特性能够帮助学生、学生模型可以小到什么程度，以及它能够多接近最先进的水平。在一个喷注标记基准上，我们将六个在架构、对称性和预训练方面各不相同（其中包括两个基础模型）的教师模型蒸馏到五个小型学生模型中。每个教师都能提升每个学生的性能，且提升幅度主要取决于教师的强弱。随后，我们在宽度和深度上进一步缩减了我们两个最强的蒸馏后学生模型——Transformer和L-GATr-slim。仅用两万个参数，蒸馏后的L-GATr-slim就超越了所有已发表的从头训练的喷注标记器……

    arXiv:2610.29439v1 Announce Type: new  Abstract: Jet tagging is one of the most important applications of machine learning at the Large Hadron Collider. Recent models build in more physics knowledge or rely on large-scale pretraining, but the experiments place strict limits on their inference cost. Knowledge distillation can transfer their performance to small students that are cheap to run. We ask which properties of a teacher help the student, how small a student can get and how close it can come to the state of the art. On a jet tagging benchmark, we distil six teachers that differ in architecture, symmetry and pretraining, including two foundation models, into five small students. Every teacher improves every student, and the gain depends mainly on the strength of the teacher. We then shrink our two strongest distilled students, Transformer and L-GATr-slim, in width and in depth. With 20 thousand parameters, a distilled L-GATr-slim outperforms every published jet tagger trained fro
    
[^95]: 一幕多面：面向长程智能体的效应边缘策略优化

    One Act, Many Masks: Effect-Marginal Policy Optimization for Long-Horizon Agents

    [https://arxiv.org/abs/2610.29427](https://arxiv.org/abs/2610.29427)

    该论文发现了多轮强化学习中信用信号与具体文本响应绑定所导致的“信用更新错位”问题，并提出效应边缘策略优化（EMPO），通过效应级别的代理目标将策略更新与执行层面的等价性对齐，从而改进长程LLM智能体的策略优化。

    

    多轮强化学习（RL）的最新进展显著提升了LLM智能体在复杂、长程交互任务上的性能。然而，尽管在细粒度信用分配方面已取得进展，策略优化仍通过针对特定响应的概率比来应用这些学习信号。不同的文本响应可以产生相同的可执行效应，但由后续结果产生的信用差异仍然与各自的文本实现绑定在一起。我们将这种不匹配识别为“信用更新错位”。我们的分析揭示，即使策略上下文和可执行效应保持不变，信用与文本的配对方式也会直接塑造策略更新的方向。受这一发现启发，我们提出了效应边缘策略优化（EMPO），它通过一个效应级别的代理目标将不同文本实现之间的更新耦合起来，使策略优化与执行层面的等价性保持一致……

    arXiv:2610.29427v1 Announce Type: new  Abstract: Recent advances in multi-turn reinforcement learning (RL) have substantially improved the performance of LLM agents on complex, long-horizon interactive tasks. However, despite progress in fine-grained credit assignment, policy optimization still applies these learning signals through response-specific probability ratios. Different textual responses can induce the same executable effect, yet credit differences arising from subsequent outcomes remain tied to their individual textual realizations. We identify this mismatch as credit-update misalignment. Our analysis reveals that credit-text pairing directly shapes policy-update directions, even when the policy context and executable effect are held fixed. Motivated by this finding, we propose Effect-Marginal Policy Optimization (EMPO), which aligns policy optimization with execution-level equivalence through an effect-level surrogate that couples updates across alternative textual realizat
    
[^96]: BIRD-RL：在有状态的数据中心环境中扩展智能体强化学习

    BIRD-RL: Scaling Agentic Reinforcement Learning over Stateful Data-Centric Environments

    [https://arxiv.org/abs/2610.29426](https://arxiv.org/abs/2610.29426)

    BIRD-RL提出了一种四阶段智能体强化学习课程方法，通过交替增强基础推理与轨迹自适应在线探索，解决了SQL调试等有状态数据中心环境中环境隔离成本高、状态污染和跨轨迹干扰导致的训练不稳定问题。

    

    尽管智能体强化学习（RL）已成为训练小型语言模型（SLM）的一种有前景的范式，但现有框架主要面向以工具为中心的场景，难以应对以数据为中心的环境。具体而言，将工具中心型框架应用于SQL调试等有状态任务会带来严重的空间低效问题，因为合理的环境隔离通常需要为每次rollout分配一个独立的数据库容器。而在多轮交互中简单地复用数据库、不跟踪其不断演变的状态，则会导致轨迹污染和跨轨迹干扰，严重破坏策略更新的稳定性。为克服这些瓶颈，我们提出了BIRD-RL，一种新颖的四阶段智能体强化学习课程方法，通过交替增强基础智能体推理能力与轨迹自适应的在线策略探索，规避直接端到端学习固有的不稳定性。关键在于，为支持这一课程学习……

    arXiv:2610.29426v1 Announce Type: cross  Abstract: While agentic reinforcement learning (RL) has emerged as a promising paradigm for training small language models (SLMs), existing frameworks are predominantly tool-centric and struggle with data-centric environments. Specifically, applying tool-centric frameworks to stateful tasks like SQL debugging introduces severe spatial inefficiencies since proper environment isolation typically requires a separate database container per rollout. Naively reusing databases without tracking evolving states across multi-turn interactions causes trajectory contamination and cross-trajectory interference, which severely destabilizes policy updates. To overcome these bottlenecks, we introduce BIRD-RL, a novel four-stage agentic RL curriculum that alternately enhances foundational agent reasoning and trajectory-adaptive on-policy exploration to circumvent the instability inherently associated with direct end-to-end learning. Crucially, to support this cu
    
[^97]: 基于向量值 $L_p$ 相关性 $V_\alpha$ 的泛化界

    Generalization Bounds via Vector-Valued $L_p$ Correlation $V_\alpha$

    [https://arxiv.org/abs/2610.29425](https://arxiv.org/abs/2610.29425)

    该论文提出了一族单参数依赖性度量 $V_\alpha$ 并据此推导出新的泛化界，这些界在 $\alpha=1$ 时退化为全变差界、在 $\alpha=2$ 时与方差和 $\chi^2$ 行为相关，且在 $\alpha>2$ 时即使损失方差无穷大也能保持有限。

    

    理解随机化学习算法的泛化能力仍然是一个核心挑战。互信息和全变差界为量化数据与假设之间的依赖性提供了互补的途径，但它们在高维情形下的表现可能截然不同。我们研究了一族单参数的依赖性度量——向量值 $L_p$ 范数相关性 $V_\alpha$，并据此推导出泛化界，其依赖项在 $\alpha=1$ 时的全变差（TV）与 $\alpha=2$ 时的方差及 $\chi^2$ 相关行为之间进行插值。对于 $\alpha>2$，Hölder 共轭损失矩小于二，由此得到的重尾特化形式即使在损失方差为无穷大时仍可保持有限。一个特别易于解释的特化出现在 $\alpha=2$，此时该界可以用算法输出密度在不同数据集之间的变异性 $\int \sqrt{\mathrm{Var}_S[p(w\mid S)]}~ dw$ 来表示。

    arXiv:2610.29425v1 Announce Type: new  Abstract: Understanding generalization of randomized learning algorithms remains a central challenge. Mutual-information and total-variation bounds provide complementary ways to quantify data--hypothesis dependence, but they can behave quite differently in high-dimensional regimes. We study a one-parameter family of dependence measures, the vector-valued $L_p$-norm correlation $V_\alpha$, and derive generalization bounds whose dependence terms interpolate between total variation (TV) at $\alpha=1$ and variance and $\chi^2$ related behavior at $\alpha=2$. For $\alpha>2$, the H\"older-conjugate loss moment is below two, yielding heavy-tailed specializations that can remain finite even when the loss variance is infinite. A particularly interpretable specialization occurs at $\alpha=2$, where the bound can be expressed in terms of the variability of the algorithm's output density across datasets, $\int \sqrt{\mathrm{Var}_S[p(w\mid S)]}~ dw$. We show t
    
[^98]: 当思想成为记忆：统一上下文学习与微调

    When Thoughts Become Memory: Unifying In-Context Learning and Fine-Tuning

    [https://arxiv.org/abs/2610.29419](https://arxiv.org/abs/2610.29419)

    该论文首次提出基于信息价值精确定律的认证规则，统一判定未见数据应放入预训练模型的上下文还是其权重，并通过随时有效的检验在多种模型与任务上验证了其正确性。

    

    我们首次给出了一个认证规则，用于判定未见数据应属于预训练模型的上下文还是其权重，该规则建立在其信息价值的精确定律之上。这一信息价值——即在留出查询上节省的编码长度——在期望上等于关于查询所存储的比特数，加上先验修复，再减去解码损失，且该定律对任何模型、架构、数据集和任务均成立。该定律的上界可通过一次测量来认证这一选择，而一个随时有效检验则可基于留出查询做出选择，且无论何时停止都能控制误差。在训练过的transformer、TabPFN、占用网络和立体匹配网络上，该上界颁发的每一份证书都是正确的。该检验在翼型压力预测任务上选择TabPFN优于XGBoost，选择训练后的代理模型优于上下文学习器；在32组比较、共64,000个查询流中，有61,948次正确判定GPT-2应使用缓存还是微调，且在零假设流中的错误率至多3.3%。该定律还能诊断何时学习（原文在此截断）

    arXiv:2610.29419v1 Announce Type: cross  Abstract: We are the first to give a certified rule for whether unseen data belong in a pretrained model's context or its weights, built on an exact law for their information value. That value, the code length saved on held-out queries, equals in expectation the bits stored about the query, plus a prior repair, minus a decoding loss, for any model, architecture, dataset and task. The law's ceilings certify the choice from one measurement, and an anytime-valid test makes it from held-out queries, with error controlled whenever it stops. Every certificate the ceilings issue on a trained transformer, TabPFN, an occupancy network and a stereo network is correct. The test picks TabPFN over XGBoost and a trained surrogate over an in-context learner on airfoil pressure, calls the better of GPT-2's cache and fine-tune in 61,948 of 64,000 query streams over 32 comparisons, and errs in at most 3.3% of null streams. The law also diagnoses when learning in 
    
[^99]: 基于贝叶斯Bradley-Terry模型的主动学习替代策略

    Alternative Strategies for Active Learning with Bayesian Bradley Terry Model

    [https://arxiv.org/abs/2610.29416](https://arxiv.org/abs/2610.29416)

    该论文开发了基于Polya-Gamma数据增强的贝叶斯Bradley-Terry模型，提出三种基于知识梯度的主动学习策略，并引入无需参考排名即可评估排名可靠性的后验分离可靠性（PSR）指标。

    

    比较判断法通过成对比较来对项目进行排序，但从有限数量的判断中恢复准确排名既取决于推理方法，也取决于所选的配对。我们开发了一种贝叶斯Bradley-Terry模型（BBTM），该模型使用Polya-Gamma数据增强以实现高效的吉布斯采样，并量化潜在能力和排名中的不确定性。我们基于知识梯度（KG）原理提出了三种主动学习策略，并使用Kendall's tau距离、后验排名熵和后验分离度来定义采集函数。对后验矩的近似一步更新降低了评估候选配对的计算成本。我们还引入了后验分离可靠性（PSR），用于衡量后验均值能力相对于后验不确定性的分散程度。它提供了一种排名可靠性的内部指标，无需参考排名。在合成实验……

    arXiv:2610.29416v1 Announce Type: new  Abstract: Comparative judgement uses pairwise comparisons to rank items, but recovering accurate rankings from a limited number of judgements depends on both the inference method and the pairs selected. We developed a Bayesian Bradley-Terry model (BBTM) that uses Polya-Gamma data augmentation for efficient Gibbs sampling and quantifies uncertainty in latent abilities and ranks. We propose three active learning strategies based on the knowledge gradient (KG) principle. We define our acquisition functions using Kendall's tau distance, posterior rank entropy, and posterior separation. Approximate one-step updates to posterior moments reduce the computational cost of evaluating candidate pairs. We also introduce Posterior Separation Reliability (PSR), which measures the spread of posterior mean abilities relative to posterior uncertainty. It provides an internal indicator of ranking reliability without requiring a reference ranking. In synthetic exper
    
[^100]: 掼蛋自博弈中的“预测—优化”差距

    The Predictive--Optimization Gap in GuanDan Self-Play

    [https://arxiv.org/abs/2610.29411](https://arxiv.org/abs/2610.29411)

    在掼蛋自博弈中，对局进度虽被证明具有真实预测价值，但以冻结胜负预测器的变化 ΔΦ 作回合级监督干预，尽管在个别训练谱系中奏效，五个预注册实验在方法层面均未复现增益，揭示了预测有效性与优化效用之间的差距。

    

    arXiv:2610.29411v1，公告类型：new。结果预测信号在强化学习中常被视为有前景的优化目标，但预测有效性与优化效用是两个不同的命题。我们在掼蛋——一种多回合不完全信息博弈，其中单回合结果的含义取决于公开的对局状态——中揭示了这一区别。留出集分析表明，状态—收益交互作用在恒定的带符号收益效应之外提供了额外的预测价值，证明对局进度在预测意义上是有意义的。随后，我们将冻结的对局胜负预测器的变化量 ΔΦ 作为受控的回合级监督干预手段，应用于匹配的 DanZero 风格蒙特卡洛自博弈协议中。该干预在个别训练谱系中可以成功：一个选定的进度增强检查点在锁定评估上显著优于其匹配的基线。然而，方法层面的结果逆转了这一印象：五个预注册（的实验中没有一个……）（摘要在此处截断）。

    arXiv:2610.29411v1 Announce Type: new  Abstract: Outcome-predictive signals are often treated as promising optimization targets in reinforcement learning, but predictive validity and optimization utility are different claims. We expose this distinction in GuanDan, a multi-round imperfect-information game in which the implication of a round outcome depends on the public match state. Held-out analysis shows that state--gain interactions provide incremental predictive value beyond a constant signed-gain effect, establishing that match progress is predictively meaningful. We then use changes in a frozen match-win predictor, $\Delta\Phi$, as a controlled round-level supervision intervention in a matched DanZero-style Monte Carlo self-play protocol. The intervention can succeed in an individual lineage: a selected progress-augmented checkpoint substantially outperforms its matched Baseline on a locked evaluation. Yet the method-level result reverses this impression: none of five preregistere
    
[^101]: 基于深度学习与领域知识驱动决策融合的区域感知多视角检测

    Region-Aware Multi-View Inspection Using Deep Learning and Domain-Informed Decision Fusion

    [https://arxiv.org/abs/2610.29398](https://arxiv.org/abs/2610.29398)

    该论文提出一种区域感知的双模型检测框架，将座圈区域缺陷的实例分割与轴区域异常的图像分类相结合，并通过形式化工业标准作业程序的确定性决策融合层，对360度旋转采集的多视角图像检测结果进行融合，实现精密部件的部件级自动判定。

    

    当同一精密部件的不同区域表现出不同的缺陷特征、因而需要不同的验收标准时，对具有微小表面缺陷的精密部件进行自动化光学检测仍然极具挑战性。本文提出了一种区域感知的双模型检测框架，将针对局部座圈区域缺陷的实例分割与针对整体轴区域异常的图像级分类相结合。这两种模型的检测结果并不被视为足以直接判定部件报废的证据，而是在模型输出之上应用一个形式化了工业标准作业程序的确定性决策融合层，以做出最终的部件级判定。系统采用PLC同步的成像装置，在360度完整旋转过程中采集18张座圈图像和18张轴部图像，以提供多视角的周向覆盖。该框架在380根汽车喷油器针阀上进行了评估，覆盖了具有代表性的工况条件。

    arXiv:2610.29398v1 Announce Type: cross  Abstract: Automated optical inspection of precision components with tiny surface defects remains challenging when different regions of the same component exhibit different defect characteristics and, therefore, different acceptance criteria. This paper presents a region-aware, dual-model inspection framework that combines instance segmentation of localized seat-region defects with image-level classification of global shaft-region anomalies. Rather than treating these two model detections as sufficient evidence for component rejection, a deterministic decision fusion layer that formalizes industrial Standard Operating Procedures is applied to the model outputs to make the final component level decision. A PLC-synchronized imaging system acquires 18 seat and 18 shaft images over a full 360-degree rotation to provide multi-view circumferential coverage. The framework was evaluated on 380 automotive fuel injector needles, covering representative ope
    
[^102]: 带温度控制谱锐化的渐进式置换对齐

    Progressive Permutation Alignment with Temperature-Controlled Spectral Sharpening

    [https://arxiv.org/abs/2610.29396](https://arxiv.org/abs/2610.29396)

    该论文提出一种渐进式置换对齐方法，通过从输入到输出的分层对齐策略，结合匈牙利算法的离散锚点与温度控制的谱退火锐化技术，解决隐式神经表示中因置换对称性导致的权重空间模型对齐难题。

    

    在隐式神经表示中，三维信号直接编码在网络权重中，因此稳定的权重空间操作至关重要。然而，置换对称性导致独立训练但功能等价的模型在权重空间中相距甚远，在线性插值下常常产生较大的损失障碍。现有的神经元对齐方法要么过早地确定离散匹配而陷入不一致的局部最优，要么依赖于对连续松弛进行计算开销高昂的梯度优化，这两种方法在NeRF等具有挑战性的场景中都不可靠。我们采用渐进式对齐，这是一种从输入到输出遵循层间依赖关系的分层策略。在每一层，估计一个相关性矩阵，并与通过匈牙利算法获得的离散锚点进行对齐，随后温度控制的谱退火逐渐将相关性结构锐化（原文摘要在此处截断）。

    arXiv:2610.29396v1 Announce Type: new  Abstract: In implicit neural representations (INRs), 3D signals are encoded directly in network weights, making stable weight-space operations essential. However, permutation symmetry causes independently trained yet functionally equivalent models to appear distant in weight space, often inducing large loss barriers under linear interpolation. Existing neuron alignment methods either commit early to discrete matches and become trapped in inconsistent local optima or rely on computationally expensive gradient optimization over continuous relaxations, both of which are unreliable in challenging cases such as NeRF. We adopt progressive alignment, a layer-wise strategy that follows inter-layer dependencies from input to output. At each layer, a correlation matrix is estimated and aligned with a discrete anchor obtained by the Hungarian algorithm, after which temperature-controlled spectral annealing gradually sharpens the correlation structure toward 
    
[^103]: 先提名后裁决：基于大语言模型辅助的单次遍历逐实例MIP求解器调优

    Nominate-then-Adjudicate: LLM-Assisted One-Pass and Per-Instance MIP Solver Tuning

    [https://arxiv.org/abs/2610.29391](https://arxiv.org/abs/2610.29391)

    该论文提出“先提名后裁决”流水线：先一次性采样并复用固定配置组合作为候选名单，再由无任务微调的大语言模型在推理时以一次模型调用、零求解器调用完成逐实例选择，从而无需昂贵的逐实例配置搜索即可实现MIP求解器的逐实例调优。

    

    混合整数规划（MIP）求解器暴露出数百甚至数千个参数，其最佳设置因实例而异，但逐实例的参数搜索代价过高。我们将此问题框架化为面向异构大数据优化工作负载的配置问题，并提出一个“先提名后裁决”流水线，该流水线围绕两个设计选择构建。首先，对固定组合进行一次采样并在多个实例间复用，从而在推理阶段免除了逐实例的配置搜索；在所评估的实例族上，所得的候选名单保留了可观的神谕性能空间，使得在该名单内进行选择成为实证研究的重点。其次，固定的20维特征表示可能将结构上不同的实例映射到同一向量，导致仅基于特征的选择器即使在这些实例偏好不同配置时也无法区分它们。基础模型不接收任何任务特定的权重更新，且选择过程为单次遍历：推理时仅需一次模型调用、无需任何求解器调用。

    arXiv:2610.29391v1 Announce Type: new  Abstract: Mixed-integer programming solvers expose hundreds or even thousands of parameters whose best setting varies per instance, yet per-instance search is prohibitively expensive. We frame this as configuration for heterogeneous Big Data optimization workloads and propose a nominate-then-adjudicate pipeline built around two design choices. First, a fixed portfolio is sampled once and reused across instances, removing per-instance configuration search at inference; on the evaluated families, the resulting shortlist retains substantial oracle headroom, making selection within it the empirical focus. Second, the fixed 20-feature representation can map structurally different instances to the same vector, preventing a feature-only selector from distinguishing them even when they prefer different configurations. The foundation model receives no task-specific weight updates, and selection is one-pass: one model call and no solver calls at inference. 
    
[^104]: 面向多样化推理轨迹的可并行化概率潜在推理

    Parallelizable Probabilistic Latent Reasoning for Diverse Reasoning Trajectories

    [https://arxiv.org/abs/2610.29385](https://arxiv.org/abs/2610.29385)

    该论文提出了一种可并行的概率潜在推理方法，通过在紧凑连续的潜在空间中对多个有效推理轨迹的分布进行建模，既实现了多样化推理轨迹的探索，又支持高效的并行化训练。

    

    依赖语言化思维链的语言模型受限于在语言空间中逐个token的生成方式，其中许多token仅服务于文本连贯性而非核心推理。这使得逻辑轨迹上的真实差异与语言表达的表面变化纠缠在一起。将推理转移到紧凑的连续潜在空间中，为高效表示和探索多样化推理轨迹提供了一种有前景的替代方案。然而，当前的潜在模型大多仍是确定性的，只能捕获单一推理轨迹，而无法对多个有效备选方案的分布进行建模。虽然随机方法对潜在推理步骤的分布进行了参数化，但它们常常面临优化不稳定和实际性能有限的问题。此外，它们在训练过程中依赖自回归的潜在预测，阻碍了有效的并行化，导致了令人难以承受的（训练开销）……

    arXiv:2610.29385v1 Announce Type: new  Abstract: Language models relying on verbalized Chain-of-Thought (CoT) are constrained to token-by-token generation in the language space, where many tokens serve textual coherence rather than core reasoning. This entangles genuine differences in logical trajectories with superficial variations in verbalization. Moving reasoning into a compact, continuous latent space offers a promising alternative for efficiently representing and exploring diverse reasoning trajectories. However, current latent models remain largely deterministic, capturing only a single reasoning trajectory rather than modeling a distribution over multiple valid alternatives. While stochastic approaches parameterize distributions over latent reasoning steps, they frequently suffer from unstable optimization and limited empirical performance. Furthermore, their reliance on autoregressive latent prediction during training precludes effective parallelization, leading to prohibitive
    
[^105]: 带噪声观测的级联赌博机：算法与遗憾界

    Cascading Bandits with Noisy Observations: Algorithms and Regret Bounds

    [https://arxiv.org/abs/2610.29381](https://arxiv.org/abs/2610.29381)

    该论文提出了一种探测成本与位置相关且反馈含噪声的随机级联赌博机模型，刻画了最优短列表可通过按每臂分数排序求得的条件，并给出了实例相关遗憾匹配理论下界的UCB算法。

    

    我们提出了一种新颖的随机级联赌博机设置，其中探测从L个臂中选取K个臂构成的有序短列表会产生与位置相关的成本，而学习者只能观察到关于每个臂真实输出的噪声反馈。我们识别出一类噪声模型的子类，对于这些模型，可以通过依据每个臂的分数对臂进行排序来找到最优短列表：当信道不存在假阳性时，这样的分数对选择和排序同时存在；当测量无成本时，这样的分数仅对排序存在（但对选择不存在）；然而在一般情况下，一旦噪声和成本同时存在，选择和排序都无法分离。在选择和排序分数都存在的情形下，我们提出了一种基于UCB的算法，在时间范围T上具有O((L-K) log T)的实例相关遗憾，该遗憾与算法无关的下界相匹配（在常数因子内）。对于分数不存在的更具挑战性的场景，我们开发了一种新算法……

    arXiv:2610.29381v1 Announce Type: new  Abstract: We propose a novel stochastic cascading bandit setup where probing an ordered shortlist of $K$ out of $L$ arms incurs position-dependent costs, and the learner observes only noisy feedback regarding each arm's true output. We identify a sub-class of noise models for which the optimal shortlist can be found by sorting arms according to a per-arm score: such a score exists for both selection and ordering when the channel has no false positives, and for ordering alone (but not selection) when measurements are costless; however in general neither selection nor ordering is separable once noise and cost are both present. For the case where both selection and ordering scores exist, we provide a UCB-based algorithm with $O((L-K)\log T)$ instance-dependent regret over a time horizon $T$, which matches an algorithm-agnostic lower bound (up to constant factors). For the more challenging scenario where scores don't exist, we develop a new algorithmi
    
[^106]: Right On Q：通过自生成轨迹恢复量化模型的推理能力

    Right On Q: Recovering Reasoning under Quantization via Self-Generated Trajectories

    [https://arxiv.org/abs/2610.29378](https://arxiv.org/abs/2610.29378)

    提出OnQ流水线，通过在策略蒸馏利用学生模型自生成的轨迹进行训练，解决了固定响应监督导致的训练与生成前缀不匹配及高熵处分布失配问题，从而恢复低比特量化大模型的推理能力。

    

    量化感知训练（QAT）旨在恢复低比特大语言模型（LLM）的推理能力。现有方法通常利用原始的全精度模型在固定的响应序列上对量化学生模型进行监督。然而，由此得到的低比特量化模型在长程推理任务上仍然表现不佳，并且常常无法在足以满足全精度模型的token预算内正常终止。我们识别出了缩小这一差距的两个挑战：第一，固定响应监督导致训练前缀与学生模型实际生成时遇到的前缀之间存在不匹配，且随着响应变长，这种轨迹分歧会不断增大；第二，在学生模型自身的前缀上，它会对某些下一个token候选词预测不足，而对另一些候选词预测过度，这种分布失配在高熵前缀处出现得更加频繁。我们提出了OnQ，一个低比特恢复流水线，它结合了在策略蒸馏与候选（摘要在此处被截断）

    arXiv:2610.29378v1 Announce Type: new  Abstract: Quantization-aware training (QAT) aims to recover reasoning capabilities in low-bit large language models (LLMs). Existing recipes typically supervise the quantized student with the original full-precision model on fixed response sequences. However, the resulting low-bit quantized models still struggle on long-horizon reasoning tasks and often fail to terminate within a token budget that suffices for the full-precision model. We identify two challenges in closing this gap. First, fixed-response supervision leaves a mismatch between training prefixes and those encountered during student generation, and this trajectory divergence grows as responses lengthen. Second, on its own prefixes, the student underpredicts some next-token candidates and overpredicts others, and this distributional misalignment becomes more frequent at high-entropy prefixes. We present OnQ, a low-bit recovery pipeline that combines on-policy distillation and candidate
    
[^107]: 用于多模态X射线和中子散射反演的组合式神经网络

    Compositional neural networks for multimodal X-ray and neutron scattering inversion

    [https://arxiv.org/abs/2610.29362](https://arxiv.org/abs/2610.29362)

    该论文提出了一种基于先验摊销流匹配的模块化框架，通过在推理阶段用组合采样器融合分别训练的单观测后验估计器，实现多模态X射线和中子散射数据的概率性联合反演，无需联合再训练即可复用预训练模型。

    

    X射线和中子散射是凝聚态物理和材料科学中结构表征不可或缺的手段。然而，由于散射实验通常只记录强度而不含相位信息，不同的结构可能产生相似的散射图案，使得相应的逆问题不具有唯一性。通过结合互补的测量可以减少这种歧义，但将其整合到机器学习框架中仍然具有挑战性。在此，我们提出了一种基于先验摊销流匹配的概率性多测量分析模块化框架。针对每种观测类型或物理探针分别训练单观测后验估计器，并仅在推理时通过组合采样器将其结合。这种后期融合方法允许预训练的单观测模型被重复使用，而无需专门的联合再训练。作为一个典型案例，我们考虑反射率测量……

    arXiv:2610.29362v1 Announce Type: new  Abstract: X-ray and neutron scattering are indispensable for structural characterization across condensed-matter physics and materials science. Yet, because scattering experiments generally record intensities without phase information, distinct structures can yield similar scattering patterns, rendering the corresponding inverse problems non-unique. This ambiguity can be reduced by combining complementary measurements, but their integration into machine-learning frameworks remains challenging. Here we introduce a modular framework for probabilistic multi-measurement analysis based on prior-amortized flow matching. Single-observation posterior estimators are trained separately for individual observation types or physical probes and combined only at inference time through a compositional sampler. This late fusion approach allows pretrained single-observation models to be reused without dedicated joint retraining. As a case in point, we consider refl
    
[^108]: GaP-MoE：通过正交梯度累积与投影实现专家混合模型中的显式专家专业化

    GaP-MoE : Explicit Expert Specialization in Mixture-of-Experts via Orthogonal Gradient Accumulation and Projection

    [https://arxiv.org/abs/2610.29355](https://arxiv.org/abs/2610.29355)

    提出GaP-MoE方法，通过正交梯度累积与投影直接操纵专家输出梯度而非修改损失函数，从而在稀疏专家混合模型中诱导显式的专家专业化，解决表示坍缩问题。

    

    稀疏专家混合模型通过将输入令牌仅路由到专家网络的一个子集来提升可扩展性，但这类模型常常遭受表示坍缩问题，即多个专家收敛到相似的函数并丧失其预期的多样性。先前的工作要么通过改进路由器架构，要么通过引入隐式惩罚专家之间相似性的辅助损失项来解决这一局限。然而，基于路由器的方法仍然容易受到路由坍缩的影响，而基于损失的方法在优化过程中可能与任务原始目标发生冲突。在本工作中，我们提出了GaP-MoE（MoE中的梯度累积与投影），该方法通过直接操纵专家输出梯度而非修改损失函数来诱导专家专业化。在每个训练步骤中，GaP-MoE基于归一化路由分数，从TopK路由的专家中随机选择一个专家作为领导者……

    arXiv:2610.29355v1 Announce Type: new  Abstract: Sparse Mixture-of-Experts (SMoE) models improve scalability by routing input tokens to only a subset of expert networks, but they often suffer from representation collapse, in which multiple experts converge toward similar functions and lose their intended diversity. Prior work has addressed this limitation either by improving router architectures or by introducing auxiliary loss terms that implicitly penalize similarity among experts. However, router-based approaches remain vulnerable to routing collapses, and loss-based approaches can conflict with the task's original objective during optimization. In this work, we propose GaP-MoE (Gradient Accumulation and Projection in MoE), which induces expert specialization by directly manipulating expert output gradients rather than modifying the loss function. At each training step, GaP-MoE stochastically selects one of the $TopK$ routed experts as a leader based on normalized routing scores, an
    
[^109]: 跨语言语音克隆中用于口音控制的闭环双参考引导方法

    Closed-Loop Dual-Reference Guidance for Accent Control in Cross-Lingual Voice Cloning

    [https://arxiv.org/abs/2610.29351](https://arxiv.org/abs/2610.29351)

    提出闭环双参考引导方法，通过将原始参考与语音转换构建的母语锚点配对并动态更新语言模型状态，有效消除跨语言语音克隆中的源语言口音，在仅损失极小说话人相似度的前提下显著提升目标语言的发音质量。

    

    零样本跨语言语音克隆常常会将参考说话人的口音带入目标语言中。我们提出闭环双参考引导方法，将原始参考与通过语音转换（VC）构建的可复用母语锚点进行配对。这两个参考分别初始化各自独立的语言模型（LM）状态，共同引导一条共享的声学轨迹；每个完成的语音片段随后会更新这两个状态，使其生成历史与实际输出的语音保持对齐。锚点构建每个说话人仅需执行一次，在线合成使用冻结的TTS模型，无需逐句进行语音转换。在20个保留测试的法语说话人上，固定设置 λ*=0.33 与直接克隆相比，将残余的法语LID质量降低了73%，ECAPA相似度损失仅为0.042；相对于仅使用锚点的克隆，该方法恢复了0.155的相似度。在保留测试组中，该方法实现了最低的平均WER（0.009）和最高的平均GOP（4.750）。

    arXiv:2610.29351v1 Announce Type: cross  Abstract: Zero-shot cross-lingual voice cloning often carries the reference speaker's accent into the target language. We propose closed-loop dual-reference guidance, which pairs the original reference with a reusable native anchor constructed by voice conversion (VC). The references initialize separate language-model (LM) states that guide a shared acoustic trajectory; each completed patch then updates both states, keeping their generated histories aligned with the emitted speech. Anchor construction runs once per speaker, and online synthesis uses a frozen TTS model without per-utterance VC. On 20 held-out French speakers, the fixed setting $\lambda^{\!*}=0.33$ reduces residual French LID mass by 73\% versus direct cloning, at an ECAPA similarity cost of only $0.042$; relative to anchor-only cloning, it recovers $0.155$. The method achieves the lowest mean WER ($0.009$) and highest mean GOP ($4.750$) among the held-out test arms. In blinded pr
    
[^110]: 数据调度与学习率调度应当协同设计

    Data Schedules and Learning Rate Schedules Should Be Designed Together

    [https://arxiv.org/abs/2610.29348](https://arxiv.org/abs/2610.29348)

    研究表明数据调度与学习率调度必须协同设计：中期训练中最优的数据过渡策略依赖于学习率调度，当学习率持续衰减时，更早引入中期训练数据并与预训练数据混合能降低最终损失。

    

    中期训练是大语言模型训练流程中的一个阶段，它在规模更小但价值更高的数据集上继续进行预训练（PT）。将预训练与中期训练两个阶段作为一个整体来看，训练通常从仅使用预训练数据开始，然后完全过渡到中期训练数据。在本工作中，我们将这种从预训练数据到中期训练数据的完全过渡视为更广泛的一类数据混合调度的特例，这类调度规定了每个批次中预训练与中期训练token的比例如何随训练过程变化。在固定总训练计算量和中期训练token预算的条件下，我们提出这样的问题：这种向中期训练数据的完全过渡对于最小化最终中期训练损失是否是最优的？我们表明，答案取决于学习率（LR）调度：在某种学习率调度下，一种数据调度可能优于另一种，但仅改变学习率调度就可能逆转这一结果。具体而言，当学习率在继续训练过程中持续衰减时，更早引入中期训练数据并将其与预训练数据混合，可以降低最终的中期训练损失。

    arXiv:2610.29348v1 Announce Type: cross  Abstract: Mid-training (MT) is a training phase in LLM training pipelines that continues pretraining (PT) on smaller datasets of higher value. Taking the PT and MT phases as a whole, training usually starts with only PT data and then transitions completely to MT data. In this work, we view this complete transition from PT to MT data as a special case of a broader class of data mixing schedules that specify how the proportions of PT and MT tokens in each batch change over training. Under fixed total training compute and MT token budgets, we ask: Is this complete transition to MT data optimal for minimizing final MT loss? We show that the answer depends on the learning rate (LR) schedule: one data schedule may outperform another under one LR schedule, but changing only the LR schedule can reverse the outcome. Specifically, when the LR decays throughout continued training, introducing MT data earlier and mixing it with PT data lowers the final MT l
    
[^111]: 深度搜索与深度研究LLM智能体的基准测试与误差分析

    Benchmarking and Error Analysis of Deep Search and Deep Research LLM Agents

    [https://arxiv.org/abs/2610.29346](https://arxiv.org/abs/2610.29346)

    本文针对深度搜索与深度研究LLM智能体提出了一套诊断中间执行过程错误的方法与框架，建立了系统级和任务级两大类错误的结构化分类体系，并揭示了这些错误对整体性能的影响。

    

    随着大语言模型（LLM）演变为能够执行长周期、多步骤任务的自主深度搜索与深度研究智能体，评估其可靠性变得日益关键。然而，现有的评估范式主要关注最终任务结果，在理解中间执行轨迹方面存在显著空白，而智能体在这些中间环节极易受到各种错误的影响。为填补这一关键空白，我们提出了一套专门用于诊断中间工作流程中错误的综合性方法与分析框架。基于该框架，我们建立了一个结构化的错误分类体系，将这些脆弱性划分为两个主要维度：系统级错误和任务级错误。通过对逐步执行轨迹和智能体-环境交互的系统检查，我们揭示了这些潜在错误如何影响整体性能。最后，通过实证……（摘要原文在此处截断）

    arXiv:2610.29346v1 Announce Type: new  Abstract: As Large Language Models (LLMs) evolve into autonomous Deep Search and Deep Research Agents capable of executing long-horizon, multi-step tasks, evaluating their reliability becomes increasingly critical. However, existing evaluation paradigms predominantly focus on final task outcomes, leaving a significant gap in understanding the intermediate execution traces, where agents are highly susceptible to various errors. To address this critical gap, we introduce a comprehensive methodology and analytical framework designed specifically to diagnose errors in intermediate workflows. Building upon this framework, we establish a structured taxonomy that categorizes these vulnerabilities into two primary dimensions: System-Level Errors and Task-Level Errors. Through a systematic examination of step-by-step execution traces and agent-environment interactions, we reveal how these underlying errors affect overall performance. Finally, through empir
    
[^112]: 可解码性并非可控性：连续潜变量TTS中的局部与各向异性说话人控制

    Decodability Is Not Controllability: Local and Anisotropic Speaker Control in Continuous-Latent TTS

    [https://arxiv.org/abs/2610.29345](https://arxiv.org/abs/2610.29345)

    该研究揭示连续潜变量TTS中说话人特征的高可解码性并不等于因果可控性，有效的话者音色控制依赖于在参考音频早期位置（L0层）进行干预，而非探针准确率最高的深层。

    

    在连续潜变量文本转语音（TTS）中，高线性可解码性并不意味着因果可控性。在 VoxCPM2 中，通过跨30名说话人和60个有向说话人配对的匹配位置探针与等范数干预揭示了这种分离：最终残差声学语言模型层（RALM L7）的参考音频状态保持0.970的探针准确率，但对齐编辑在60个配对中却产生0个目标主导结果。在参考音频早期位置的编辑则呈现相反的模式：在L0层，配对对齐更新使目标说话人相似度提升0.326，并在全部60个配对中实现目标主导，而任何等范数方向或位置控制下最多仅达到1/60。独立的说话人编码器、针对保留说话人原型的检索以及双盲听测共同证实，转换后的声音被听者感知为预期的目标说话人。在 dots.tts 中的跨架构干预进一步区分了主干杠杆作用与直接声学控制：主干编辑……

    arXiv:2610.29345v1 Announce Type: cross  Abstract: High linear decodability does not entail causal controllability in continuous-latent text-to-speech (TTS). In VoxCPM2, position-matched probes and equal-norm interventions across 30 speakers and 60 directed dyads expose this separation: reference-audio states at the final residual acoustic language model layer (RALM L7) retain 0.970 probe accuracy, yet aligned edits yield 0/60 target-dominant dyads. Edits at early reference-audio positions produce the opposite pattern. At L0, pair-aligned updates increase target similarity by 0.326 and achieve target dominance in all 60 dyads, against at most 1/60 under any equal-norm direction or position control. Independent speaker encoders, retrieval against held-out speaker prototypes, and double-blind listening confirm that the shifted voice is heard as the intended target. Cross-architecture interventions in dots.tts distinguish backbone leverage from direct acoustic control: backbone edits disp
    
[^113]: Transformer MLP中的读取器—写入器耦合

    Reader--Writer Coupling in Transformer MLPs

    [https://arxiv.org/abs/2610.29340](https://arxiv.org/abs/2610.29340)

    本文提出将Transformer MLP中隐藏通道的读取器与写入器视为耦合对进行联合测量，利用MLP更新雅可比矩阵与自身转置的一致性导出带符号的配对坐标，并据此发现每个模块仅由一小部分强耦合通道构成核心。

    

    Transformer的MLP为每个隐藏通道赋予两种角色：读取器决定该通道何时响应，写入器决定该通道向残差流中加回什么内容。若将此层视为键—值记忆来读取，则可以一次测量一个因子集：读取器与写入器之间的配对仅由它们共享的索引承载，任何仅从单一因子集计算出的量都无法看到这种配对。然而，训练通过联合更新将这两个因子耦合在一起，因此我们探究将二者一同读取能够揭示什么。该模块本身提供了联合测量的工具：MLP写回到它所读取的同一个流中，因此其更新的精确雅可比矩阵是方阵，它与自身转置的一致性可以分解为读取器—写入器项，其内积为每个通道（作为一个耦合对来读取）给出一个带符号的配对坐标。按此坐标排序，我们所测量的每个模块都拥有一个由强耦合通道组成的小核心；在初始化使两种符号等概率出现之处……（原文摘要在此处截断）

    arXiv:2610.29340v1 Announce Type: cross  Abstract: A transformer MLP gives each hidden channel two roles: a reader that decides when the channel responds, and a writer that decides what it adds back to the residual stream. Read as a key--value memory, the layer can be measured one factor set at a time: the pairing of readers to writers is carried only by their shared index, invisible to any quantity computed from one factor set alone. Yet training couples the two factors through their joint update, so we ask what reading them together shows. The block supplies the joint instrument: the MLP writes back into the very stream it reads, so the exact Jacobian of its update is square, and its agreement with its own transpose resolves into reader--writer terms whose inner products give each channel, read as a couple, a signed pairing coordinate. Ranked by this coordinate, each block we measure has a small core of strongly coupled channels; where initialization gives either sign equally, the co
    
[^114]: BrainCodes：基于离散神经编码的轻量级脑电基础模型

    BrainCodes: A Lightweight EEG Foundation Model over Discrete Neural Codes

    [https://arxiv.org/abs/2610.29320](https://arxiv.org/abs/2610.29320)

    BrainCodes是一个基于离散神经编码的轻量级EEG基础模型，通过量化神经分词器与时空编码器的自监督预训练，以紧凑架构在睡眠分期等多种神经生理监测任务上达到或超越现有最先进水平。

    

    近来的脑电图基础模型日益强调模型规模，这可能限制了其在资源受限设备和日常环境中的部署。我们认为可迁移的EEG特征可以通过紧凑的架构来学习，并提出了BrainCodes——一个轻量级基础模型，它通过自监督学习在来自1000多名参与者的800多小时记录数据上进行预训练。我们的架构将基于量化的神经分词器与支持任意电极配置的时空编码器相结合。预训练目标通过通道位置编码明确鼓励表征捕获空间连接性，并通过辅助训练目标实现对窄带神经振荡的敏感性。我们的模型在包括睡眠分期在内的多种神经生理监测任务上达到或超越了现有最先进水平。

    arXiv:2610.29320v1 Announce Type: new  Abstract: Recent electroencephalography (EEG) foundation models increasingly emphasize model scale, potentially limiting their deployment on resource-constrained devices and in everyday environments. We argue that transferable EEG features can instead be learned with a compact architecture and introduce BrainCodes, a lightweight foundation model pretrained through self-supervised learning on more than 800 hours of recordings from over 1,000 participants. Our architecture combines a quantization-based neural tokenizer with a spatiotemporal encoder that supports arbitrary electrode configurations. The pretraining objective explicitly encourages representations that capture spatial connectivity through channel positional encoding, and sensitivity to narrowband neural oscillations through an auxiliary training objective. Our model matches or surpasses the state of the art across diverse neurophysiological monitoring tasks, including sleep staging, att
    
[^115]: DISRL：用于动态社区追踪的双尺度增量式半监督强化学习

    DISRL: Dual-Scale Incremental Semi-Supervised Reinforcement Learning for Dynamic Community Tracking

    [https://arxiv.org/abs/2610.29315](https://arxiv.org/abs/2610.29315)

    提出了双尺度增量式半监督强化学习框架DISRL，通过动态生成种子特定指导并将短期适应与长期参数巩固解耦，解决了动态社区追踪中监督不确定性和误差级联累积两大难题。

    

    动态局部社区检测旨在跨图快照追踪指定种子节点所在的演化社区。然而，该方法面临两个关键挑战：一是时间监督不确定性，即稀疏且过时的标签难以在拓扑变化下提供针对种子的指导；二是无界的误差级联，即顺序策略转移会在长时间范围内累积探索误差。为应对这些挑战，我们提出了DISRL，一个双尺度增量式半监督强化学习框架。DISRL引入了一个监督构建模块，能够从稀疏的已知社区中动态生成针对种子的指导；以及一个双尺度社区探索模块，通过将时间窗口内的短期适应与跨窗口的长期参数巩固相解耦来稳定策略演化。此外，我们构建了一个包含多样社区演化场景的动态图基准数据集。广泛的实验……（摘要截断）

    arXiv:2610.29315v1 Announce Type: new  Abstract: Dynamic local community detection tracks the evolving community of designated seed nodes across graph snapshots. However, it faces two critical challenges: temporal supervision uncertainty, where sparse and outdated labels fail to provide seed-specific guidance under topology changes, and unbounded error cascades, where sequential policy transfer accumulates exploration errors over long horizons. To address these challenges, we propose DISRL, a Dual-scale Incremental Semi-supervised Reinforcement Learning framework. DISRL introduces a supervision construction module that dynamically generates seed-specific guidance from sparse known communities, and a dual-scale community exploration module that stabilizes policy evolution by decoupling short-term adaptation within temporal windows from long-term parameter consolidation across windows. Furthermore, we construct a dynamic graph benchmark with diverse community evolution scenarios. Extensi
    
[^116]: YingLan：面向异构观测统一表示学习的多模态海洋基础模型

    YingLan: A Multimodal Ocean Foundation Model for Unified Representation Learning from Heterogeneous Observations

    [https://arxiv.org/abs/2610.29313](https://arxiv.org/abs/2610.29313)

    提出了多模态海洋基础模型 YingLan，通过物理属性条件化动态编码、时空增量耦合（STIC）与海洋状态演化感知预训练，实现了对卫星影像、网格化海洋场和浮标观测等异构数据的统一可迁移表示学习。

    

    多模态海洋建模仍然极具挑战性，其原因在于卫星影像、网格化海洋场与浮标观测在数据结构、物理语义和时空特性上存在显著的异构性。现有方法大多聚焦于特定任务的海洋建模，而从异构观测中学习可迁移的表示仍是一个悬而未决的问题。为应对这一挑战，我们提出了 YingLan——一种多模态海洋基础模型，它集成了物理属性条件化动态编码、时空增量耦合以及面向海洋状态演化的预训练。具体而言，动态编码机制依据物理语义与观测特性生成自适应的输入投影，从而能够灵活地表征异构海洋变量；STIC 通过潜在时间特征的跨尺度交互来建模多尺度时空依赖关系……

    arXiv:2610.29313v1 Announce Type: new  Abstract: Multimodal ocean modeling remains challenging due to substantial heterogeneity in data structures, physical semantics, and spatiotemporal characteristics across satellite imagery, gridded oceanographic fields, and buoy observations. Existing approaches largely focus on task-specific ocean modeling, while learning transferable representations from heterogeneous observations remains an open problem. To address this challenge, we propose YingLan, a multimodal ocean foundation model that integrates physics-attribute-conditioned dynamic encoding, Spatiotemporal Increment Coupling (STIC), and ocean-state evolution-aware pretraining. Specifically, the dynamic encoding mechanism generates adaptive input projections conditioned on physical semantics and observation characteristics, enabling flexible representation of heterogeneous ocean variables. STIC models multiscale spatiotemporal dependencies through cross-scale interactions of latent tempor
    
[^117]: 基于遗传规划的混合整数线性规划可解释割平面选择方法

    Interpretable Cutting Plane Selection for Mixed-Integer Linear Programming Using Genetic Programming

    [https://arxiv.org/abs/2610.29311](https://arxiv.org/abs/2610.29311)

    本文提出首个基于遗传规划的完全可解释割平面选择框架GPSCS，通过演化符号化评分函数将手工启发式策略显式泛化为可解释的自适应割平面选择策略，弥补了神经网络方法可解释性差且缺乏收敛保证的不足。

    

    割平面能够在不排除任何可行整数解的前提下，从混合整数线性规划（MILP）中移除分数解，因而是现代MILP求解器的关键组成部分。当前最先进的求解器依赖于经过精细调优的手工割平面策略，尽管这些策略行之有效，但本质上仍是静态的，可能并非最优。近年来，已有基于学习的方法被提出用于设计自适应割平面策略；然而，这些方法大多基于神经网络，可解释性有限，且无法提供收敛性保证。在本文中，我们提出了基于遗传规划与统计的割平面选择方法（GPSCS），这是首个将手工割平面选择启发式规则显式泛化为完全可解释策略的基于学习的框架。首先，遗传规划模型演化出一个符号化评分函数，利用从（摘要在此处被截断）

    arXiv:2610.29311v1 Announce Type: new  Abstract: Cutting planes (cuts) remove fractional solutions from mixed-integer linear programs (MILPs) without excluding any feasible integer solutions, making them a key ingredient in modern MILP solvers. Current state-of-the-art solvers rely on fine-tuned handcrafted cut strategies, which, despite their effectiveness, remain inherently static and can be suboptimal. More recently, learning-based approaches have been proposed to design adaptive cut strategies; however, most of these methods are based on neural networks, offering limited interpretability and providing no guarantees of convergence. In this paper, we propose \emph{Genetic Programming and Statistics-based Cut Selection} (GPSCS), the first learning-based framework that explicitly generalizes handcrafted cut selection heuristics into a fully interpretable policy. First, a genetic programming model evolves a symbolic scoring function that ranks candidate cuts using features derived from 
    
[^118]: 数据驱动与物理引导的实测波浪谱后报

    Data-driven and physics-guided hindcasting of measured wave spectra

    [https://arxiv.org/abs/2610.29306](https://arxiv.org/abs/2610.29306)

    该研究提出一种数据驱动并结合物理约束的循环模型，利用浮标谱初始化并输入风、风区和深度信息，实现了对波罗的海实测波浪谱长达12小时的准确后报，其中物理松弛约束将波高RMSE降低5.01%至约2厘米。

    

    对实测波浪谱的短期后报进行了评估，所用循环模型由单个逐时浮标谱初始化，并输入当地风、风区和深度信息。来自波罗的海三个站点的观测数据被组合使用：2017–2020年用于训练，2021年用于验证，2022年用于测试。ERA5再分析资料在整个十二小时预测区间内提供风场。比较研究考察了谱方差、峰值频率和峰值振幅的监督方式，随后引入经验风浪边界和局部松弛约束。在数据驱动比较中，方差与峰值频率监督给出了最低的谱误差，而额外的峰值振幅监督给出了最低的波高误差。在匹配的物理比较中，松弛约束将波高RMSE降低了5.01%，降至约2厘米。将该约束扩展至第13–24小时（在这些小时没有谱目标的情况下），使主谱RMSE降低了……（原文摘要在此处截断）

    arXiv:2610.29306v1 Announce Type: cross  Abstract: Short-range hindcasts of measured wave spectra are evaluated using a recurrent model initialized by one hourly buoy spectrum and supplied with local wind, fetch and depth. Measurements from three Baltic Sea stations are combined for training in 2017--2020, validation in 2021 and testing in 2022. ERA5 reanalysis supplies winds throughout the twelve-hour prediction interval. The comparison examines supervision of spectral variance, peak frequency and peak amplitude, followed by empirical wind-sea bounds and a local relaxation constraint. Variance and peak-frequency supervision gives the lowest spectral error within the data-driven comparison, whereas additional peak-amplitude supervision gives the lowest wave-height error. In the matched physical comparison, relaxation reduces height RMSE by 5.01\% to \PhPtwoHs{}~cm. Extending this constraint through hours 13--24, without spectral targets at those hours, reduces primary spectral RMSE by 
    
[^119]: ExpEvolve：面向开放式发现的经验结构化与复用

    ExpEvolve: Structuring and Reusing Experience for Open-Ended Discovery

    [https://arxiv.org/abs/2610.29303](https://arxiv.org/abs/2610.29303)

    提出ExpEvolve框架，将思想层级的提取与控制注入进化式编码代理，通过发现能够解释程序间评分差异的高层级思想因素并复用经验，提升开放式科学发现的样本效率。

    

    诸如AlphaEvolve这样的进化编码代理是利用大语言模型（LLM）构建AI科学家的显著成功案例之一。这些代理通过迭代改进和演化程序来解决开放式科学问题，利用LLM的先验知识和推理能力。尽管取得了成功，但对程序的变异大多是基于表面启发式的低层级操作，缺乏持久的控制单元。相比之下，科学是通过思想实现复利式积累的。在这项工作中，我们提出了ExpEvolve，它将思想层级的提取与控制注入进化过程：它从程序中发现并利用能够解释不同程序之间评分差异的高层级因素。我们刻画了利用思想因素进行发现何时能提升样本效率的理论条件。我们还构建了一个实用框架，以展示提取出的高层级思想因素有助于指导基于经验的利用。

    arXiv:2610.29303v1 Announce Type: cross  Abstract: Evolutionary coding agents such as AlphaEvolve are one of the notable successes in using Large Language Models (LLMs) to build AI Scientists. These agents tackle open-ended scientific problems by iteratively improving and evolving programs, leveraging the prior knowledge and reasoning capabilities of LLMs. Despite their success, mutations over programs are mostly low-level upon surface heuristics, without a persistent unit of control. In contrast, science compounds through ideas. In this work, we propose ExpEvolve, which injects idea-level extraction and control into evolution: it discovers and leverages the high-level factors from programs that explain scoring differences among different programs. We characterize theoretical conditions on when discovery with idea factors improves sample efficiency. We also build a practical framework to show that the extracted high-level idea factors are helpful for guiding exploitation from experienc
    
[^120]: 在行为保真度要求下压缩太空机器人的星载语言模型

    Compressing Onboard Language Models for Space Robots under a Behavioral-Fidelity Requirement

    [https://arxiv.org/abs/2610.29302](https://arxiv.org/abs/2610.29302)

    本文提出QSelect方法，通过为每个线性模块从AWQ、GPTQ和AutoRound量化器中选择最优方案并以KL散度作为保真度验收标准，为太空机器人构建满足飞行硬件约束且行为接近全精度模型的静态低比特星载语言模型。

    

    语言模型正被探索作为太空机器人解释操作员指令并在已验证技能之间进行选择的接口。在星载环境下运行语言模型需要低比特压缩，而飞行应用还增加了一些通用训练后量化未能明确处理的要求：固定的内存预算与存储格式、受配置控制的静态制品，以及与在地面验证过的全精度模型相近的行为表现。我们将压缩视为地面段的一个步骤，并研究了QSelect方法，它通过将每个线性模块分配给某一个量化器家族，并在最终的下一个词元分布上对每次替换进行评分，从兼容的AWQ、GPTQ和AutoRound检查点中组合出一个静态的低比特模型。默认的保真度模式无需标签或调整的阈值，仅当与BF16参考模型相比的前向KL散度减小时才接受替换；可选的目标适配模式则增加了在小型目标数据集上的答案似然（摘要在此处截断）。

    arXiv:2610.29302v1 Announce Type: cross  Abstract: Language models are being explored as the interface through which space robots interpret operator commands and select among verified skills. Running them onboard calls for low-bit compression, and flight use adds requirements that generic post-training quantization leaves implicit: a fixed memory budget and storage format, a static artifact under configuration control, and behavior close to the full-precision model validated on the ground. We treat compression as a ground-segment step and study QSelect, which composes one static low-bit model from compatible AWQ, GPTQ, and AutoRound checkpoints by assigning each linear module to one quantizer family and scoring every substitution at the final next-token distribution. The default fidelity mode needs no labels or tuned thresholds and accepts a substitution only when the forward KL divergence from the BF16 reference decreases; an optional target-adapted mode adds answer likelihood on a sm
    
[^121]: 近似所罗门诺夫归纳：论学习得到的通用预测器的鲁棒性

    Approximate Solomonoff Induction: On the Robustness of Learned Universal Predictors

    [https://arxiv.org/abs/2610.29289](https://arxiv.org/abs/2610.29289)

    本文研究了实际机器学习算法产生的近似最优预测器的鲁棒性问题，给出了近似对预测误差最坏情况影响的紧致界，从而论证了通过对数损失最小化学习到的通用预测器在近似条件下依然保持稳健。

    

    通过在大规模多样化数据集上进行对数损失最小化训练的大型语言模型为何展现出广泛的能力和可泛化的技能，这一问题尚未被完全理解。Grau-Moya等人（ICML 2024）通过元学习的视角提供了一个概念性解释：若序列生成环境 $\nu$ 从先验 $w_{\nu}$ 中抽取，则最小化总体对数损失可以恢复相应的贝叶斯混合预测器 $\xi=\sum_\nu w_\nu\nu$。在 $w$ 为通用先验的理想化极限下，最优预测器对应于所罗门诺夫通用归纳。然而，实用的机器学习算法通常只能产生近似最优的预测器。这引出了一个自然的问题：预测性能在近似条件下是否依然保持鲁棒？在本文中，我们首先给出了近似对预测误差的最坏情况影响的紧致界。我们证明任何满足 $ 的预测器 $\rho$……（摘要原文在此处被截断）

    arXiv:2610.29289v1 Announce Type: new  Abstract: Why large language models trained via log-loss minimization over massive and diverse datasets exhibit broad capabilities and generalizable skills is not yet fully understood. Grau-Moya et al. (ICML 2024) offer a conceptual explanation through the lens of meta-learning: if sequence-generating environments $\nu$ are drawn from a prior $w_{\nu}$, minimizing population log-loss recovers the corresponding Bayesian mixture predictor $\xi=\sum_\nu w_\nu\nu$. In the idealized limit in which $w$ is a universal prior, the optimal predictor corresponds to Solomonoff universal induction. Practical machine learning algorithms, however, generally produce only \emph{approximately} optimal predictors. This raises a natural question: \emph{Does the prediction performance remain robust under approximation}? In this paper, we first provide a tight bound on the worst-case effect of approximation on prediction error. We show any predictor $\rho$ satisfying $
    
[^122]: Flat-RLVR：面向大语言模型推理能力的基于可验证奖励的锐度感知强化学习

    Flat-RLVR: Sharpness-Aware Reinforcement Learning with Verifiable Rewards for Reasoning Capacity in LLMs

    [https://arxiv.org/abs/2610.29281](https://arxiv.org/abs/2610.29281)

    提出Flat-RLVR方法，通过锐度感知最小化在RLVR中寻找平坦奖励区域，缓解分布锐化问题，在保持准确率的同时提升大语言模型的样本多样性和推理能力。

    

    带有可验证奖励的强化学习（RLVR），特别是通过组相对策略优化（GRPO），已成为大语言模型（LLM）后训练的核心范式。虽然RLVR在增强大语言模型的推理能力方面非常有效，但它通常会引发“分布锐化”现象，即策略分布变得过度尖锐，那些罕见但正确的轨迹被逐步抑制，从而降低了样本多样性。为此，我们在RLVR中促进平坦的奖励最优值以平滑策略分布。我们从理论上将平坦奖励的RLVR与分布锐化的减少联系起来，并在实证上展示了在保持单次尝试准确率的同时改善了样本多样性。我们的方法称为Flat-RLVR，在大语言模型后训练过程中通过锐度感知最小化来搜索平坦的奖励区域，并将其与GRPO变体相结合，以在……之间实现更好的权衡（摘要原文在此处截断）。

    arXiv:2610.29281v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR), particularly through group-relative policy optimization (GRPO), has emerged as a central paradigm for the post-training of large language models (LLMs). While RLVR is highly effective at enhancing the reasoning capabilities of LLMs, it commonly induces \textit{distribution sharpening}, whereby the policy distribution becomes excessively peaked and rare-yet-correct trajectories are progressively suppressed, thus diminishing sample diversity. To this end, we promote flat-reward optima in RLVR to smooth the policy distribution. We theoretically link flat reward RLVR to reduced distribution sharpening, and empirically show improved sample diversity while maintaining single-attempt accuracy. Our method, called \texttt{Flat}-RLVR, searches for flat reward regions via sharpness-aware minimization during LLM post-training, combining this with GRPO variants to achieve better trade-offs betwee
    
[^123]: 基于贝叶斯进化优化的多智能体系统设计与调优

    Design and Tuning of Multi-Agent Systems Using Bayesian Evolutionary Optimization

    [https://arxiv.org/abs/2610.29276](https://arxiv.org/abs/2610.29276)

    提出了一种结合进化算子与图感知贝叶斯优化的框架，可自动发现并调优面向特定任务的多智能体系统拓扑结构，在代码生成和表格数据预处理基准上取得了最优性能。

    

    设计有效的多智能体系统仍然是一项具有挑战性且耗费资源的任务。我们提出了一个贝叶斯进化优化框架，用于自动发现面向特定任务的多智能体系统拓扑结构。该方法将进化算子生成的多样化工作流图与贝叶斯优化相结合，把成本高昂的评估集中在有前景的候选方案上。与基于大语言模型的候选选择不同，我们的方法使用图感知的高斯过程和期望改进准则，在进化搜索过程中提供明确的选择标准。我们在代码生成基准 BigCodeBench 以及使用 TabPFN 的表格数据预处理基准上对该方法进行了评估。在 BigCodeBench 上，进化得到的架构在所有被评估的智能体系统中取得了最高的整体性能；在时间序列表格数据集上，优化后的多智能体系统预处理在多项任务上优于默认的 TabPFN 模型，同时在其余任务上保持竞争力。

    arXiv:2610.29276v1 Announce Type: new  Abstract: Designing effective multi-agent systems (MAS) remains a challenging and resource-intensive task. We propose a Bayesian Evolutionary Optimization framework for automatically discovering task-specific MAS topologies. The approach combines evolutionary operators to generate diverse workflow graphs with Bayesian optimization to focus cost intensive evaluations on promising candidates. In contrast to LLM-based candidate selection, our approach uses a graph-aware Gaussian Process and Expected Improvement to provide an explicit selection criterion during evolutionary search. We evaluate the method on BigCodeBench for code generation and on a tabular preprocessing benchmark using TabPFN. On BigCodeBench, evolved architectures achieve the highest overall performance among the evaluated agent systems. On temporal tabular datasets, optimized MAS preprocessing improves over the default TabPFN model on several tasks, while remaining competitive on II
    
[^124]: 面向机器学习模型安全部署的性能估计方法

    Estimating Performance for Safe Deployment of Machine Learning Models

    [https://arxiv.org/abs/2610.29262](https://arxiv.org/abs/2610.29262)

    提出了MOVAL模型评估工具，通过类别级置信度校准与多校准分数集成方法，利用无标签测试数据估计模型性能，从而实现机器学习模型的安全部署。

    

    将深度神经网络应用于现实场景的主要挑战在于训练数据集与测试数据集之间的分布差异，这通常是由数据获取条件的变化引起的。在没有标签的情况下评估预训练模型的性能对于安全性至关重要，但深度模型过于乐观的概率输出使这一问题变得复杂，而类不平衡往往进一步加剧了这一问题，导致不可靠的预测。为了解决这些问题，我们提出了MOVAL，一种新颖的模型评估工具，它利用未标记的测试数据来估计模型性能。MOVAL结合了按类别的置信度校准来调整模型输出，从而不仅能够估计准确率之外的指标，还能减轻类不平衡对置信度校准的不利影响。此外，我们提出了一种集成方法，将多个校准后的置信度分数相结合，以获得更稳健的解决方案。我们在自然（图像数据集上验证了MOVAL的有效性）。

    arXiv:2610.29262v1 Announce Type: new  Abstract: The main challenge in applying deep neural networks to real-world scenarios lies in distributional discrepancies between training and test datasets, often caused by variations in data acquisition conditions. Assessing the performance of pre-trained models without labels is crucial for safety, but this is complicated by deep models' overly optimistic probabilistic outputs, often exacerbated by class imbalances, leading to unreliable predictions. To address these issues, we introduce MOVAL, a novel MOdel eVALuation tool that estimates performance using unlabeled test data. MOVAL incorporates class-wise confidence calibration to adjust model outputs, allowing for the estimation of metrics beyond accuracy and mitigating the detrimental effects of class imbalance on confidence calibration. Additionally, we propose an ensemble approach that combines multiple calibrated confidence scores for a more robust solution. We validate MOVAL on natural 
    
[^125]: 去中心化网络上的群组分布式鲁棒优化

    Group Distributionally Robust Optimization over Decentralized Networks

    [https://arxiv.org/abs/2610.29261](https://arxiv.org/abs/2610.29261)

    该论文提出直接优化去中心化网络上的原始群组分布式鲁棒优化目标，并通过新颖的误差分解方法将其转化为去中心化在线凸优化问题，首次在半去中心化和完全去中心化设置下为工作节点的局部模型建立了理论保证。

    

    我们研究了去中心化网络上的群组分布式鲁棒优化（GDRO），其中各工作节点通过最小化最大局部风险来协作学习鲁棒模型。现有方法仅优化对偶正则化的代理目标，并针对全局平均模型建立理论保证，而该全局平均模型对工作节点来说是不可获取的。在本文中，我们直接优化原始的GDRO目标，并在半去中心化和完全去中心化两种设置下为局部模型建立理论保证。一个关键难点在于，将集中式GDRO中的原始-对偶分解方法应用于此会产生依赖于工作节点的损失函数，而其评估需要无法获取的工作节点间交叉信息。为应对这一挑战，我们开发了一种新颖的误差分解方法，将去中心化GDRO重构为：(i) 在半去中心化设置下的去中心化在线凸优化问题和专家建议预测问题，以及 (ii) 在完全去中心化设置下的两个相互耦合的去中心化在线凸优化问题。

    arXiv:2610.29261v1 Announce Type: cross  Abstract: We study group distributionally robust optimization (GDRO) over decentralized networks, where workers collaborate to learn robust models by minimizing the maximum local risk. Existing methods optimize only dual-regularized surrogates and establish guarantees for a globally averaged model that is not available to workers. In this paper, we directly optimize the original GDRO objective and establish guarantees for local models in semi-decentralized (SemiDec) and fully decentralized (Dec) settings. A key difficulty is that applying the primal-dual decomposition from centralized GDRO yields worker-dependent losses whose evaluation requires unavailable cross-worker information. To address this challenge, we develop a novel error decomposition that recasts decentralized GDRO as (i) a decentralized online convex optimization (D-OCO) problem and a prediction-with-expert-advice problem in SemiDec, and as (ii) two coupled D-OCO problems in Dec, 
    
[^126]: 通过可观测性扩展FAIR原则以实现可信联邦学习

    Extending FAIR Principles with Observability for Trustworthy Federated Learning

    [https://arxiv.org/abs/2610.29252](https://arxiv.org/abs/2610.29252)

    提出FAIR-O框架，将FAIR原则扩展出可观测性维度，使每个参与者在每轮联邦学习中生成最小化的签名信任证据（可由本地TinyLLM辅助解释），在不迁移原始数据的前提下建立三级信任模型以实现可信联邦学习。

    

    联邦学习（FL）将数据保留在其源头，但仅靠数据的本地化并不能使联邦学习值得信赖。在缺乏足够可观测性的情况下，聚合器无法判断客户端服务是否健康、本地数据流水线是否可信，或合同约束在训练过程中是否得到妥善监控。我们的立场是：可信的联邦学习除了模型更新过程之外，还需要一个“证据平面”。每个参与者持续地将本地基础设施和数据总结为与每一轮联邦学习相关联的最小化的、经签名的信任证据，而原始数据和日志仍保留在本地。本地的TinyLLM（小型语言模型）可以协助解释可观测性信号、推导上下文证据，并向运维人员提供自然语言解释。我们通过FAIR-O实现了这一愿景，FAIR-O是FAIR原则的可观测性扩展。该方法不以基准测试联邦学习的准确率为重点，而是提出了一个三级信任模型以及一个双参与者的Python实现……

    arXiv:2610.29252v1 Announce Type: new  Abstract: Federated learning (FL) keeps data at its source, but locality alone does not make a federation trustworthy. Without sufficient observability, an aggregator cannot determine whether the client service was healthy, the local data pipeline was trustworthy, or contractual constraints were properly monitored during training. We take the position that trustworthy FL needs an evidence plane in addition to its model-update process. Each participant continuously summarizes local infrastructure, data into minimal, signed trust evidence associated with each FL round, while raw data, logs remain local. A local TinyLLM may assist in interpreting observability signals, deriving contextual evidence, and providing natural-language explanations to operators. We instantiate this vision through FAIR-O, an observability extension of FAIR principle. Rather than benchmarking FL accuracy, a three-level trust model has been proposed and a two-participant Pytho
    
[^127]: MA-UITD：用于刻画连续社会行为的离线多智能体奖励推断

    MA-UITD: Offline Multi-Agent Reward Inference for Characterizing Continuous Social Behavior

    [https://arxiv.org/abs/2610.29237](https://arxiv.org/abs/2610.29237)

    提出MA-UITD，一种离线多智能体奖励推断方法，无需额外环境交互即可从固定行为数据集的联合演示中推断显式的智能体特定奖励，用于刻画连续社会行为。

    

    多智能体逆强化学习不仅为模仿学习提供了一个强大的框架，也为理解支持社会互动和决策的神经计算提供了有力工具。从相互作用的智能体中推断奖励具有挑战性，因为观察到的选择既反映个人偏好，也反映对他人的预期。现有方法通常需要额外的环境交互、以内循环形式进行的重复策略优化，或对环境动力学和动作空间的严格假设，这限制了它们在固定行为数据集上的适用性。我们提出了多智能体单边逆时序差分学习（MA-UITD），这是一种离线方法，可从连续状态环境（动作空间为离散或有界连续）的联合演示中推断出显式的智能体特定奖励。其核心思想是通过表征每个智能体自身动作的长期后果……

    arXiv:2610.29237v1 Announce Type: new  Abstract: Multi-agent inverse reinforcement learning offers a powerful framework not only for imitation learning but also for understanding the neural computations supporting social interactions and decision-making. Inferring rewards from interacting agents is challenging because observed choices reflect both personal preferences and expectations about others. Existing methods often require additional environment interaction, repeated policy optimization as an inner loop, or restrictive assumptions about environment dynamics and action spaces, limiting their applicability to fixed behavioral datasets. We introduce Multi-Agent Unilateral Inverse Temporal Difference Learning (MA-UITD), an offline method for inferring explicit agent-specific rewards from joint demonstrations in continuous-state environments with either discrete or bounded continuous actions. Its core idea is to represent the long-term consequences of each agent's own actions through 
    
[^128]: 面向流式视频理解的局部缓存与选择性重算

    Cache Locally, Recompute Selectively for Streaming Video Understanding

    [https://arxiv.org/abs/2610.29229](https://arxiv.org/abs/2610.29229)

    提出无需训练的即插即用方法FrameBlend，通过在查询时利用查询相关上下文对选定视频标记的KV缓存进行选择性重算，解决了流式视频理解中的上下文不匹配问题。

    

    流式视频理解（SVU）要求视频大语言模型在持续处理不断输入的视频的同时，以低延迟回答用户查询。近期方法通过在空闲时段将帧预填充至KV缓存，并在查询时复用缓存的KV，来降低查询时的计算开销。为了在无界的视频流上控制每帧的计算量，这些方法在预填充时限制了可用的上下文。这导致了一个根本性的上下文不匹配问题：受限的上下文可能遗漏了正确理解视频标记（video tokens）所需的信息，同时又包含了与最终查询无关的信息，而查询内容在预填充阶段是未知的。我们提出FrameBlend，一种简单而有效的即插即用方法，通过选择性的KV重算来解决这一不匹配问题，且无需任何额外训练。在查询时，FrameBlend使用与查询相关的上下文对选定的视频标记进行重新预填充，并用重算得到的KV替换原有的缓存KV。至关重要的是……（原文摘要于此处截断）

    arXiv:2610.29229v1 Announce Type: cross  Abstract: Streaming video understanding (SVU) requires video-LLMs to process incoming video continuously while answering user queries with low latency. Recent methods reduce query-time cost by prefilling frames into KV caches during idle time and reusing the cached KVs at query time. To bound per-frame computation over unbounded video streams, these methods restrict the context available during prefill. This creates a fundamental context mismatch: the restricted context may omit information needed to properly contextualize video tokens, yet, still contain information irrelevant to the eventual query, which is unknown during prefill. We propose FrameBlend, a simple yet effective plug-and-play method to address this mismatch through selective KV recomputation, without requiring any additional training. At query time, FrameBlend re-prefills selected video tokens using query-relevant context and replaces their cached KVs with recomputed ones. Crucia
    
[^129]: 面向测度值数据的薛定谔-巴斯桥重心模型

    Schr\"odinger-Bass Bridge Barycenter Models for Measure-Valued Data

    [https://arxiv.org/abs/2610.29227](https://arxiv.org/abs/2610.29227)

    本文利用薛定谔-巴斯桥的方向不对称性与对偶漂移-扩散控制，提出四种新型测度值数据重心模型，并从理论上证明了这些重心的存在性，且证明高斯输入下解码器形式存在高斯最优解，可将重心计算化为有限维优化。

    

    arXiv:2610.29227v1 公告类型：交叉 摘要：基于概率测度之间随机与确定性最优传输桥的重心模型，已被证明对（概率）测度值数据的有监督和无监督学习十分有用。在本文中，我们利用最近提出的两测度间薛定谔-巴斯桥中出现的方向不对称性及对偶漂移-扩散控制，提出了SBB-编码器、SBB-解码器、SBB-对称和SBB-马尔可夫四种重心模型。针对每一个新模型，我们的第一个理论结果在温和假设下证明了SBB重心的存在性。我们的第二个理论结果刻画了高斯分布的解码器SBB重心：我们证明对于高斯输入测度，解码器形式存在高斯极小化器，从而将重心计算归结为有限维优化，并得到一个紧凑的算法。对于一般的…

    arXiv:2610.29227v1 Announce Type: cross  Abstract: Barycenter models, based on stochastic and deterministic optimal transport bridges between probability measures, have proven to be useful for supervised and unsupervised learning for (probability) measure-valued data. In this paper, we utilize the directional asymmetry and dual drift-diffusion control arising in the recently proposed Schr\"odinger-Bass Bridge (SBB) between two measures, and propose \textit{SBB-encoder}, \textit{SBB-decoder}, \textit{SBB-symmetric}, and \textit{SBB-Markovian} barycenter models. For each of these novel models, our first theoretical result establishes the existence of the SBB-barycenter under mild assumptions. Our second theoretical result characterizes the decoder SBB-barycenter of Gaussians where we show that for Gaussian input measures, the decoder formulation admits a Gaussian minimizer, reducing barycenter computation to a finite-dimensional optimization and yielding a compact algorithm. For general 
    
[^130]: 多阶段制造中基于延迟引导注意力的质量预测

    Delay-Guided Attention for Quality Prediction in Multistage Manufacturing

    [https://arxiv.org/abs/2610.29220](https://arxiv.org/abs/2610.29220)

    该论文提出了DAMF方法，通过显式估计阶段间延迟分布并利用其引导跨阶段信息传递，解决了多阶段制造中延迟随时间变化导致的质量预测难题。

    

    多阶段制造中的质量预测在阶段间延迟随时间变化时极具挑战性，因为在同一时间戳采集的测量数据可能反映不同的工艺状态。现有方法通常保留生产顺序，但融合的是时间戳对齐的特征，或者隐式地建模延迟依赖关系，而没有识别与每个下游观测相关的上游历史信息。我们提出了面向多阶段预测的延迟引导注意力方法，该方法显式地估计阶段间延迟分布，并利用这些分布来引导跨阶段信息传递，从而实现最终质量预测。对于每个相邻阶段对，延迟预测器将当前下游表示与候选的延迟上游表示进行比较。通过发生-幅值分解，分别估计正延迟发生的概率及其条件幅值，从而得到针对每个样本和阶段对的离散延迟状态分布。

    arXiv:2610.29220v1 Announce Type: new  Abstract: Quality prediction in multistage manufacturing is challenging when inter-stage lags vary over time, as measurements taken at the same timestamp may reflect different process states. Existing approaches often preserve production order but fuse timestamp-aligned features or model delayed dependencies implicitly, without identifying the upstream history relevant to each downstream observation. We propose Delay-Guided Attention for Multi-stage Forecasting (DAMF), which explicitly estimates inter-stage lag distributions and uses them to guide cross-stage information transfer for final quality prediction. For each adjacent stage pair, a lag predictor compares the current downstream representation with candidate lagged upstream representations. An occurrence-magnitude factorization separately estimates the probability of a positive lag and its conditional magnitude, yielding a sample- and stage-pair-specific distribution over discrete lag state
    
[^131]: PerSPL：个性化半监督偏好学习

    PerSPL: Personalized Semi-supervised Preference Learning

    [https://arxiv.org/abs/2610.29219](https://arxiv.org/abs/2610.29219)

    提出PerSPL框架，通过成对均值分数中心化来稳定跨提示的分数坐标系，从而迭代地将大量无标签响应转化为用户特定的训练信号，以低成本方式提升个性化奖励模型的学习效果。

    

    个性化奖励模型需要用户特定的偏好比较数据，但收集这类反馈成本高昂，且只能为每个用户提供有限的监督信号。与此同时，大量响应可以在不涉及用户身份或偏好标签的情况下生成，然而现有的个性化奖励模型并未利用这些数据。直接从这些响应中学习是困难的，因为成对目标函数约束的是同一提示内的相对分数，而对单个响应进行伪标注则需要同一用户在不同提示之间可比较的分数。我们提出了 PerSPL（个性化半监督偏好学习），这是一个与骨干模型无关的框架，它利用成对均值分数中心化来稳定这种跨提示的分数坐标系，并迭代地将无标签响应转化为用户特定的训练信号。在四个基准（PersonalLLM、HelpSteer2、Reddit TLDR 以及……）上与七个代表性基线进行了对比（摘要至此被截断）。

    arXiv:2610.29219v1 Announce Type: new  Abstract: Personalized reward models require user-specific preference comparisons, but collecting such feedback is costly and provides only limited supervision for each user. At the same time, large pools of responses can be generated without user identities or preference labels, yet existing personalized reward models leave this data unused. Directly learning from these responses is difficult because pairwise objectives constrain relative scores within a prompt, whereas pseudo-labeling individual responses requires scores that are comparable across prompts for the same user. We propose \textbf{PerSPL} (Personalized Semi-supervised Preference Learning), a backbone-agnostic framework that uses pair-mean score centering to stabilize this cross-prompt score coordinate and iteratively turns unlabeled responses into user-specific training signals. Against seven representative baselines across four benchmarks---PersonalLLM, HelpSteer2, Reddit TLDR, and 
    
[^132]: 偏好不等于许可：代理式商务中个性化自动购买边界的确定

    Preference Is Not Permission: Personalizing the Auto-Purchase Line in Agentic Commerce

    [https://arxiv.org/abs/2610.29216](https://arxiv.org/abs/2610.29216)

    该论文提出将学习个人化的“自动购买边界”作为代理式商务中的新任务，并发现十六个指令微调模型普遍“越权”——即使商品匹配度低、购买依据缺失也会自动下单，与人类委托者的授权意愿存在系统性错位，且这一边界因人而异、需要个性化建模。

    

    个性化购物基准关注的是哪件商品适合用户；我们则提出另一个问题：智能体可以在该商品上被允许承诺到什么程度，并将学习个人的“自动购买线”作为一项任务提出。这条边界以证据为条件且具有三值性：是否做出购买承诺是否恰当，取决于商品信息的充分性（购买依据 warrant）、可逆性、价格是否符合用户的规范以及紧迫性，而不仅仅取决于匹配程度；其自然输出为“不购买”、“先确认”或“自动购买”三个层级。在十六个指令微调模型上的实验表明，模型的自主承诺行为与人类授权参考存在系统性错位——在该参考中，十五名标注者扮演给定用户画像中的委托方，选择他们在无需确认的情况下所允许的最远行动：即便匹配度很低，即便模型自身的证据标志显示购买依据缺失，模型仍会做出购买承诺。这是一种越权行为。这条边界因人而异：这些评分者之间存在分歧，但分歧是结构化的。

    arXiv:2610.29216v1 Announce Type: cross  Abstract: Personalized shopping benchmarks ask which item fits the user; we ask how far an agent may commit on that item, and pose learning the individual's auto-purchase line as a task. The boundary is evidence-conditioned and three-valued: whether committing is appropriate depends on the sufficiency of product information (warrant), reversibility, price against the user's norm, and urgency, not only on fit; and the natural output is don't buy, confirm first, or buy automatically. Across sixteen instruction-tuned models, autonomous commitment is misaligned with a human authorization reference in which fifteen annotators, role-playing the delegating principal from a given profile, choose the furthest action they would allow without confirmation: models commit even when fit is low and when their own evidence flag reports the warrant missing. This is overreach. The line is individual: these raters disagree, yet the disagreement is structured and p
    
[^133]: 论无穷远处的梯度猜想：可定义势函数梯度轨迹的有限球面长度

    On the gradient conjecture at infinity: Finite spherical length for gradient trajectories of definable potentials

    [https://arxiv.org/abs/2610.29211](https://arxiv.org/abs/2610.29211)

    该论文证明了任意 o-极小结构中可定义 $C^1$-势函数的每条梯度轨迹（无论收敛还是发散到无穷远）的球面投影长度有限（即总角运动有限且方向收敛），从而对无穷远处梯度猜想和可定义 Thom 梯度猜想给出了肯定的回答。

    

    我们证明，对于每个开集 $U \subseteq \mathbb R^n$，可定义 $C^1$-势函数 $F \colon U \to \mathbb R$ 的每条梯度轨迹 $\theta(t)$（$t \geq 0$）的球面投影都具有有限长度，特别地，它是收敛的（方向收敛）。换句话说，可定义 $C^1$-势函数的每条梯度轨迹都具有有限的总角运动。这里“可定义”指在任意扩张实数域的 o-极小结构中可定义。值得注意的是，该结果中的梯度轨迹是梯度流微分方程（其解可能不唯一）的任意解，无需任何附加假设：梯度轨迹既可以无界并发散到无穷远处，也可以是收敛的。因此，所建立的定理对可定义势函数的无穷远处梯度猜想以及可定义 Thom 梯度猜想均给出了肯定的回答。我们将该定理应用于……

    arXiv:2610.29211v1 Announce Type: cross  Abstract: We prove that for every open $ U \subseteq \mathbb R^n $ the spherical projection of every gradient trajectory $\theta(t)$, $t \geq 0$, of a definable $C^1$-potential $F \colon U \to \mathbb R$ has finite length and, in particular, converges (directional convergence). In other words, every gradient trajectory of a definable $C^1$-potential has finite total angular motion. Here, definable means definable in an arbitrary o-minimal structure expanding the real field.   Notably, the gradient trajectory in this result is an arbitrary solution of the gradient flow differential equation (whose solutions may not be unique) with no further assumptions: The gradient trajectory may be unbounded and diverge to infinity or it may be convergent. The established theorem therefore provides affirmative answers to both the gradient conjecture at infinity for definable potentials and the definable Thom gradient conjecture.   We apply our theorem to the t
    
[^134]: 果蝇罗盘连接组中的重建差异可与个体差异相媲美，并破坏已训练的回路模型

    Reconstruction differences rival individual differences in the fly compass connectome and break trained circuit models

    [https://arxiv.org/abs/2610.29205](https://arxiv.org/abs/2610.29205)

    该研究发现果蝇头朝向（罗盘）连接组中由重建流程差异（如突触检测方法）造成的回路变化可与不同果蝇个体之间的生物学差异相媲美，说明基于连接组训练的回路模型依赖于其构建所用的特定重建版本。

    

    神经网络模型越来越多地基于突触分辨率的连接组构建，而同一个回路如今可以在来自不同动物和不同处理流程的重建结果中获得。我们尚不清楚此类模型是否依赖于构建它们所用的重建版本，也不清楚连接组之间的差异究竟是生物学上的还是测量误差。我们研究了果蝇头朝向回路的九个版本：四只果蝇（雄性CNS、半脑hemibrain、Fly Wire、BANC），以及同一只果蝇仅在阈值、校对或突触检测方式上有所不同的五个变体。通过使用经解剖学标记验证的基于线路连接的对应关系，我们将所有差异置于同一尺度上进行比较。结果显示：在同一只果蝇内改变突触检测方法，在四种距离度量上造成的回路变化超过了到达最小果蝇间差异的一半；校对造成的差异较小但依赖于所用的度量指标；而同一只果蝇左右两半之间的差异并不大于随机划分的一半。网络

    arXiv:2610.29205v1 Announce Type: cross  Abstract: Neural network models are increasingly constructed from synapse-resolution connectomes, and the same circuit is now available in reconstructions from other animals and pipelines. We do not know whether such models are dependent on the reconstruction they are built from, and whether connectome differences are biological or measurement errors. We studied the Drosophila head-direction circuit in nine versions: four flies (male CNS, hemibrain, Fly Wire, BANC) and five variants of the same fly varying only by thresholds, proofreading or synapse detection. Using wiring-based correspondences validated by anatomical labels, we place all differences on one scale. A change of synapse detection within one fly changed the circuit more than halfway to the smallest inter-fly difference on four distances; proofreading differences were smaller but depended on the metric, and left and right halves of one fly differed no more than a random half. Network
    
[^135]: 边缘VLM时代物理对抗鲁棒性的统一评估框架

    A Unified Evaluation Framework of Physical Adversarial Robustness in the Edge VLM Era

    [https://arxiv.org/abs/2610.29196](https://arxiv.org/abs/2610.29196)

    本文提出了边缘VLM时代物理对抗鲁棒性的统一评估框架，引入条件攻击成功率指标，在对抗补丁攻击下公平比较了视觉语言模型与传统目标检测器的鲁棒性差异。

    

    视觉语言模型的最新进展使其能够通过语言提示和生成式解码实现端到端的视觉定位，展现出显著提升的零样本和开放词汇检测能力。因此，VLM正迅速被部署于安全攸关的现实世界系统的边缘设备上。然而，边缘VLM的物理对抗鲁棒性尚未得到系统性研究，且其输出形式与现代目标检测器存在根本差异，为公平评估带来了方法论上的挑战。为了直接比较这两类模型，我们对每个选定的代表性模型在同一个结果上进行评估：佩戴在人体上的对抗补丁能否逃逸模型的检测。我们通过引入一个统一指标——条件攻击成功率来报告这一结果，该指标仅在模型在匹配的灰色遮挡物下仍能定位目标的图像上进行统计，并将所有失败划分为漏检、替换

    arXiv:2610.29196v1 Announce Type: cross  Abstract: Recent advancements in Vision-Language Models (VLMs) enable end-to-end visual grounding via language prompts and generative decoding, demonstrating largely improved zero-shot and open-vocabulary detection capabilities. As a result, VLMs are rapidly being deployed at the edge in safety-critical real-world systems. However, the physical adversarial robustness of edge VLMs has not been systematically studied, and the fundamental differences in output with modern object detectors create methodological challenges for fair evaluation. To compare the two families directly, we evaluate every selected representative model on one outcome: whether an adversarial patch worn on a person evades detection of the model. We report this by introducing a unified metric, conditional attack success rate (cASR), counted only on images the model still localizes under a matched gray occluder, and partition every failure into omission (no box), substitution (a
    
[^136]: 截断袋外概率增益：一种用于随机森林封装式特征选择的抗噪准则

    Truncated Out-of-Bag Probability Gain: A Noise-Robust Criterion for Random-Forest Wrapper Feature Selection

    [https://arxiv.org/abs/2610.29195](https://arxiv.org/abs/2610.29195)

    本文提出TOP-Gain准则，通过将真实类别袋外概率的下降截断为零来抑制袋外估计噪声，使随机森林封装式特征选择的特征排序更具可重复性，并在26个数据集的预注册基准测试中取得最佳综合表现，且计算效率比VSURF高21倍。

    

    随机森林封装式特征选择可以利用袋外预测作为内置的验证信号，但决定每次前向步骤所选准则会累积袋外估计噪声。我们提出了TOP-Gain（截断袋外概率增益），该方法通过真实类别的袋外概率在样本层面的平均增量来为候选特征评分，并将下降部分截断为零。在一个简单的噪声模型下，截断消除了由未受影响样本贡献的约三分之二的噪声方差，同时保留了信号，并使候选特征的排序显著更具可重复性。在一项针对26个OpenML-CC18数据集的预注册基准测试中，TOP-Gain在15种方法中于ROC曲线下面积（ROC-AUC）、准确率和Macro-F1上均取得了最佳平均排名；它显著优于14个基线方法中的13个，并以少21倍的单核计算量达到了与VSURF相当的准确率。在相同的封装器内，……（原文摘要在此处截断）

    arXiv:2610.29195v1 Announce Type: new  Abstract: Random-Forest wrapper feature selection can use out-of-bag (OOB) predictions as a built-in validation signal, but the criteria that decide each forward step accumulate OOB estimation noise. We propose TOP-Gain (Truncated Out-of-bag Probability Gain), which scores a candidate feature by the mean sample-level increase in the OOB probability of the true class, with decreases truncated at zero. Under a simple noise model, truncation removes about two-thirds of the noise variance contributed by unaffected samples while preserving the signal, and it makes the ranking of candidate features markedly more reproducible. In a pre-registered benchmark on 26 OpenML-CC18 datasets, TOP-Gain achieved the best mean rank in area under the ROC curve (ROC-AUC), accuracy and Macro-F1 among 15 methods; it was significantly better than 13 of 14 baselines and matched the accuracy of VSURF with 21 times less single-core computation. Within an identical wrapper, 
    
[^137]: 曲率缩放的锐度感知最小化：超越平滑性的收敛性

    Curvature-Scaled Sharpness-Aware Minimization: Convergence Beyond Smoothness

    [https://arxiv.org/abs/2610.29192](https://arxiv.org/abs/2610.29192)

    该论文利用相对梯度失真揭示了SAM在广义平滑条件下失真可能无界、SAM会循环而USAM会发散的现象，建立了梯度扰动统一失真控制的必要条件，并由此提出曲率缩放的锐度感知最小化方法，实现了超越平滑性假设的收敛性保证。

    

    锐度感知最小化（SAM）已成为训练深度神经网络的一种广泛使用的方法。尽管其应用广泛，但现有的SAM类方法的收敛性保证都要求平滑性这一限制性假设。广义平滑条件最近在梯度方法的分析中被提出，但对于SAM仍未得到充分探索。为了分析SAM的扰动在广义平滑性下的影响，我们使用相对梯度失真来量化锐度感知方向偏离原始梯度的程度。在$(L_0,L_1)$-平滑性条件下，我们证明这种失真对于SAM在接近平稳点时可能变得无界，对于其未归一化变体（USAM）在大梯度时可能变得无界，并给出了SAM发生循环、USAM发散的具体例子。随后，我们针对沿梯度方向的扰动（其非负乘子仅依赖于梯度范数）建立了实现统一失真控制的必要条件。

    arXiv:2610.29192v1 Announce Type: cross  Abstract: Sharpness-Aware Minimization (SAM) has established itself as a widely used method for training deep neural networks. Despite its popularity, existing convergence guarantees for SAM-type methods require smoothness, a restrictive assumption. Generalized smoothness conditions have recently been proposed in the analysis of gradient methods but remain underexplored for SAM. To analyze the effect of SAM's perturbation under generalized smoothness, we use relative gradient distortion to quantify how the sharpness-aware direction departs from the original gradient. Under $(L_0,L_1)$-smoothness, we show that this distortion can become unbounded near stationarity for SAM and at large gradients for its unnormalized variant (USAM), with examples where SAM cycles and USAM diverges. We then establish a necessary condition for uniform distortion control among perturbations along the gradient whose nonnegative multipliers depend only on its norm. Moti
    
[^138]: 基于学习的控制十年回顾：从理论基础到实际部署

    A Decade of Learning-Based Control: From Theoretic Foundations to Practical Deployment

    [https://arxiv.org/abs/2610.29187](https://arxiv.org/abs/2610.29187)

    本文作为教程论文，受IEEE CDC十年“基于学习的控制”专题启发，系统回顾了机器学习与控制交叉领域的关键进展，涵盖稳定性与安全性保证理论、动力学模型学习方法，以及强化学习、模仿学习和贝叶斯优化等数据驱动策略合成方法及其应用与挑战。

    

    受IEEE决策与控制会议（CDC）十年来“基于学习的控制”特邀专题的启发，本教程论文回顾了机器学习与控制交叉领域的关键进展。我们专注于将机器学习技术的灵活性与控制的结构和理论基础相结合的方法，以设计能力强、可靠且安全的自主系统。我们首先介绍了建立稳定性和安全性保证的核心概念，然后回顾了通过学习动力学模型来改善不确定性下控制的方法，最后讨论了直接综合或学习策略的数据驱动方法，包括强化学习、模仿学习和贝叶斯优化。我们进一步讨论了代表性应用和开放挑战。本文的目标是为该领域提供一个易于理解的入门途径，并支持向可靠的基于学习的控制系统持续迈进。

    arXiv:2610.29187v1 Announce Type: cross  Abstract: Motivated by ten years of Learning-Based Control Invited Sessions at the IEEE Conference on Decision and Control (CDC), this tutorial paper reviews key advances at the intersection of machine learning and control. We focus on methods that combine the flexibility of machine learning techniques with the structure and theoretical foundations of control to design autonomous systems that are capable, reliable, and safe. We first introduce core notions for establishing stability and safety guarantees, then review approaches that learn dynamics models to improve control under uncertainty, and finally discuss data-driven methods that directly synthesize or learn policies, including reinforcement learning, imitation learning, and Bayesian optimization. We further discuss representative applications and open challenges. The goal is to provide an accessible entry point to the field and to support continued progress toward reliable learning-based 
    
[^139]: ROCOCO：被遮挡曲线目标的鲁棒轮廓补全

    ROCOCO: Robust Outline Completion of Occluded Curvilinear Objects

    [https://arxiv.org/abs/2610.29177](https://arxiv.org/abs/2610.29177)

    该论文提出ROCOCO，一种基于流匹配的生成模型，结合UNet遮挡掩码预测，无需先验遮挡信息即可鲁棒地补全被环境障碍物遮挡的曲线目标轮廓，性能显著超越现成修复基线。

    

    我们提出了一种基于学习的生成模型，用于在环境障碍物遮挡的情况下补全曲线特征。为此，我们采用了一种基于流匹配的方法，支持用适当的曲线特征对遮挡区域进行修复。为了消除对遮挡先验特权信息的依赖，我们还训练了一个基于UNet的掩码预测模型，该模型仅需被遮挡的观测即可识别需要修复的区域。我们通过使用程序生成的测试用例进行实验验证了方法的各个组件，并使用大型公开的曲线目标数据集验证了完整的流水线。与现成的图像修复基线相比，当未提供遮挡掩码时，ROCOCO以显著优势成为最强的方法；当提供掩码时，它与掩码监督的修复方法具有竞争力，在全图误差上取得最佳表现，在遮挡区域上仅次于LaMa。

    arXiv:2610.29177v1 Announce Type: cross  Abstract: We propose a learning-based generative model for completing curvilinear features under occlusion by environmental obstacles. To this end, we adopt a flow matching-based approach that supports the inpainting of occluded regions with the appropriate curvilinear features. To eliminate the reliance on privileged information about the occlusion a priori, we also train a UNet-based mask prediction model, which can identify regions for inpainting given only the occluded observation. The components of our approach are validated via experiments using procedurally-generated test cases, and the complete pipeline is validated using large public datasets for curvilinear objects. Against off-the-shelf inpainting baselines, ROCOCO is the strongest method by a wide margin when no occlusion mask is given, and is competitive with mask-supervised inpainting when one is, achieving best on full-image error and second only to LaMa on the occluded region, wh
    
[^140]: 条件流模型将认知不确定性转化为表观的偶然方差

    Conditional Flow Models Convert Epistemic Uncertainty into Apparent Aleatoric Variance

    [https://arxiv.org/abs/2610.29172](https://arxiv.org/abs/2610.29172)

    该论文发现条件流模型生成的样本方差会系统性地大于真实条件分布的方差，这种额外的方差本质上是模型将拟合真实条件分布时的认知不确定性转化为生成样本中表观的偶然方差。

    

    条件流模型（CFMs）在机器学习的各个领域正被越来越多地用来替代确定性预测器，包括那些传统上被表述为回归的任务。鉴于它们在此类任务中的成功，我们提出了一个问题：CFM生成的条件分布在点预测器所能提供的信息之外，还携带了哪些有用的信息？我们识别出一个系统性现象：在许多自然的情形下，即使在近乎确定性的回归问题中，CFM样本的方差通常也大于真实条件分布的方差。这一效应在模型尚未充分解析出对条件变量的依赖关系之处最为显著，因此这种额外的方差可以被视为将拟合真实条件分布时的认知不确定性转化为生成样本中表观的偶然方差。通过受控的合成数据和真实数据实验，以及与指数族基线的比较……

    arXiv:2610.29172v1 Announce Type: new  Abstract: Conditional Flow Models (CFMs) are increasingly being used in place of deterministic predictors across machine learning, including for tasks that were classically formulated as regression. Given their success in such tasks, we ask the question: what useful information do the conditional distributions produced by CFMs carry beyond what a point predictor would provide?   We identify a systematic phenomenon: in many natural regimes, CFM samples routinely have larger variance than the true conditional distribution, even in nearly deterministic regression problems. This effect is most pronounced where the model has not resolved the dependence on the conditioning variable, and thus the additional variance can be viewed as converting epistemic uncertainty in fitting the ground truth conditional distribution into apparent aleatoric variance in the generated samples. Through controlled synthetic and real-data experiments, and comparisons with exp
    
[^141]: TeachBack：通过双向偏好优化实现在线策略自蒸馏的“学会教学”

    TeachBack: Learning to Teach via Bidirectional Preference Optimization for On-Policy Self-Distillation

    [https://arxiv.org/abs/2610.29167](https://arxiv.org/abs/2610.29167)

    TeachBack提出了一种双向偏好学习框架，使模型不仅从掌握特权信息的教师处学习，还反过来优化教师的教学方式，从而生成更适合学生学习的推理轨迹。

    

    在线策略自蒸馏（OPSD）使语言模型能够从一个“教师”模型学习，该教师由同一个模型实例化而来，但被赋予了特权信息，例如参考答案。我们发现，拥有充分的特权信息可以让教师生成高度准确的轨迹；然而，仅凭准确性并不能保证这些轨迹适合学生学习。直接获取完整答案可能会促使教师生成简短的、依赖参考答案的推导，从而省略了在没有特权信息的情况下解决问题所必需的探索、验证和自我纠正等过程。我们提出了TeachBack，这是一个用于在线策略自蒸馏的双向偏好学习框架，它不仅从教师那里学习，还学习教师应该如何教学。TeachBack通过控制注入到提示中的参考推理后缀的长度等方式来塑造教师的生成行为……

    arXiv:2610.29167v1 Announce Type: cross  Abstract: On-policy self-distillation (OPSD) enables a language model to learn from a Teacher instantiated from the same model but given access to privileged information, such as a reference solution. We find that access to sufficient privileged information allows the Teacher to generate highly accurate trajectories; however, accuracy alone does not guarantee that these trajectories are suitable for Student learning. Direct access to a complete solution may encourage shortened, reference-conditioned derivations that omit the exploration, verification, and self-correction required to solve problems without privileged information. We introduce TeachBack, a bidirectional preference-learning framework for on-policy self-distillation that learns not only from the Teacher but also how the Teacher should teach. TeachBack shapes the Teacher's generation behavior by controlling the length of the reference-reasoning suffix injected into the prompt and by 
    
[^142]: 从数据增强到合成证据：在数据集入口处评估生成的胸部X光片

    From Augmentation to Synthetic Evidence: Evaluating Generated Chest Radiographs at Dataset Intake

    [https://arxiv.org/abs/2610.29152](https://arxiv.org/abs/2610.29152)

    该研究提出在数据集入口处对生成的胸部X光片进行质量检查、来源分类和记录验证的三重评估，发现基于质量的选择虽能显著提高质量通过率，但无法使生成图像同时通过质量与来源双重检查。

    

    生成的胸部X光片作为明示的数据增强可能是有用的，但当同一张图像被提交作为真实检查的记录时，它就变成了缺乏依据的证据。我们通过区分三个问题来研究这一问题：图像是否通过自动化质量检查、来源分类器是否将其标记为生成图像，以及可信记录是否支持所声称的采集过程。在我们的实验中，Qwen从1,000张现有的Cheff识别或SynthCheX候选图像池中最多选择20张图像。接收方使用MedGemma评估图像质量，并使用基于冻结DINOv3特征的分类器预测图像来源。基于质量的选择使Cheff识别图像的质量通过率从20张中的12张提高到19张，SynthCheX图像从8张提高到17张。我们没有发现任何证据表明选择能够改善同时通过两项检查的情况：没有选中的Cheff识别图像能同时通过两项检查，而均匀选择、质量选择和质量约束的替代选择分别产生2张、1张和……（摘要在此处截断）

    arXiv:2610.29152v1 Announce Type: cross  Abstract: A generated chest radiograph can be useful as declared augmentation, but the same image becomes unsupported evidence when submitted as the record of a real examination. We study this problem by separating three questions: whether an image passes an automated quality check, whether an origin classifier flags it as generated, and whether a trusted record supports the claimed acquisition. In our experiments, Qwen selects up to 20 images from pools of 1,000 existing Cheff-identified or SynthCheX candidates. A receiver evaluates quality with MedGemma and predicted origin with a classifier on frozen DINOv3 features. Quality-based selection raises quality passes from 12 to 19 of 20 for Cheff-identified images and from 8 to 17 for SynthCheX. We find no evidence that selection improves passage through both checks: no selected Cheff-identified image passes both, while uniform, quality, and quality-constrained surrogate selection produce 2, 1, an
    
[^143]: SAFTER：面向分布偏移下鲁棒强化学习的基于显式可恢复性的安全感知滤波

    SAFTER: Safety-Aware Filtering Through Explicit Recoverability for Robust Reinforcement Learning under Distribution Shift

    [https://arxiv.org/abs/2610.29149](https://arxiv.org/abs/2610.29149)

    SAFTER通过将可微分的闭式备份CBF滤波器嵌入SAC训练，把当前安全性、备份安全性和可恢复性统一为单一可微分证书，在分布偏移下的四足导航任务中显著提升了成功率并大幅降低样本消耗。

    

    arXiv:2610.29149v1 公告类型：cross 摘要：在执行器能力受限的条件下，学习能够泛化到未见环境的安全策略十分困难，因为一个不安全的决策可能导致系统变得不可恢复。传统的控制屏障函数（CBF）滤波器仅强制执行瞬时安全性，而忽略了未来的可恢复性。我们提出了SAFTER（通过显式可恢复性进行安全感知滤波），它将一个可微分的、具有闭式解的备份CBF滤波器嵌入到Soft Actor-Critic训练中。SAFTER将当前安全性、有限时域备份安全性、终端可恢复性和输入可容许性统一为一个单一的可微分证书。其CBF条件能够产生精确的最小干预投影，而无需迭代的在线优化。在四足机器人导航任务中，SAFTER达到了90%的训练成功率，比Vanilla SAC少使用30.6%的转移样本，比基于当前状态的CBF滤波SAC基线少使用34.1%-35.1%的转移样本。在未见过的模拟环境中，SAFTER将成功率最多提升了42.4%。

    arXiv:2610.29149v1 Announce Type: cross  Abstract: Learning safe policies that generalize to unseen environments is difficult under bounded actuation, where an unsafe decision may become unrecoverable. Conventional control barrier function (CBF) filters enforce only instantaneous safety and ignore future recoverability. We propose SAFTER (Safety-Aware Filtering Through Explicit Recoverability), which embeds a differentiable, closed-form backup-CBF filter into Soft Actor-Critic training. SAFTER unifies current safety, finite-horizon backup safety, terminal recoverability, and input admissibility into a single differentiable certificate. Its CBF condition yields an exact minimum-intervention projection without iterative online optimization. In quadruped navigation, SAFTER reaches 90% training success with 30.6% fewer transitions than Vanilla SAC and 34.1-35.1% fewer than current-state CBF-filtered SAC baselines. Across unseen simulated environments, SAFTER improves success by up to 42.4 
    
[^144]: 通过表征工程在现实场景中缓解奖励黑客行为

    Mitigating Reward Hacking in Realistic Scenarios Through Representation Engineering

    [https://arxiv.org/abs/2610.29129](https://arxiv.org/abs/2610.29129)

    本研究提出一种基于表征工程的动态干预方法，通过多种引导向量以低成本显著减少现实场景中的奖励黑客行为，且不损害模型能力，并能跨领域泛化。

    

    奖励黑客行为是当前生产级机器学习系统中的一个重大问题，会导致代价高昂且危险的故障。本研究探讨了如何在现实环境中（包括工具调用和长上下文）监测和缓解奖励黑客行为。我们对通过三种不同方法获得的引导方向应用了动态干预：来自标注回合数据的均值差向量、基于梯度优化的向量，以及模型引用反奖励黑客行为规范时提取的向量。这种动态干预比向残差流均匀添加引导向量对模型表征的扰动更小，从而能够实现更精确的编辑。我们发现，该方法能够以低成本显著减少奖励黑客行为，且不会对模型的连贯性或能力造成相应的显著损失；此外，最初从编程环境中提取的引导向量也能泛化到减少其他领域的奖励黑客行为。

    arXiv:2610.29129v1 Announce Type: new  Abstract: Reward hacking is a major problem in current production machine learning systems, resulting in costly and dangerous failures. Here we study how to monitor for and mitigate reward hacking in realistic settings, including tool calls and long contexts. We apply a dynamic intervention to steering directions obtained by three different methods: difference-in-means vectors from labeled rollouts, gradient optimization-based vectors, and vectors extracted when models reference an anti-reward-hacking model spec. This dynamic intervention perturbs model representations less than uniform steering vector addition to the residual stream, allowing for more precise edits. We find that we are able to cheaply and significantly reduce reward hacking without correspondingly significant costs to model coherence or capability, and that steering vectors originally extracted from coding environments also generalize to reducing reward hacking in other domains. 
    
[^145]: 预算受限下的递归自我改进：在代码编辑运行之前对其进行评分

    Recursive Self-Improvement on a Budget: Scoring Code Edits Before They Run

    [https://arxiv.org/abs/2610.29121](https://arxiv.org/abs/2610.29121)

    该论文提出一种预算受限的递归自我改进方法：利用由少量实例评估拟合并冻结的线性读出器，在代码编辑实际运行之前对其进行评分，使自改写的编码智能体能够在有限评估预算下高效选择最有前景的编辑。

    

    一个会改写自身脚手架的编码智能体，必须在该编辑被评估之前就选择要展开哪个提议的编辑，而搜索最终还必须返回一个智能体。每次运行都会基于96个脚手架编辑（这些编辑由16实例拟合分割的四次评估进行标注）来拟合一个线性读出器，其输入为父代在引导分割上的求解率以及提议差异的八个坐标，随后冻结这些系数。HumanEval+预留48个实例作为引导分割，Polyglot预留64个。该分割用于为父代评分并维护一个包含四个节点的存档。Huxley-Gödel机器选择机制通过汤普森采样，基于已被评估节点的引导分割求解计数来抽取一个父代，并展开该父代所提议的子代。之后，每种方法都会返回其引导分割求解计数达到最大的最早被评估的节点。表格报告了该节点在其余100个HumanEval+实例和其余145个Polyglot实例上的留出集分数。

    arXiv:2610.29121v1 Announce Type: cross  Abstract: A coding agent that rewrites its own scaffold has to choose which proposed edit to expand before that edit has been evaluated, and the search then has to return one agent. Each run fits one linear readout on the parent's steering-split solve rate and eight coordinates of the proposed diff, from 96 scaffold edits labeled by four evaluations of a 16-instance fit split, and then freezes the coefficients. HumanEval+ reserves 48 instances as the steering split and Polyglot reserves 64. That split scores parents and keeps an archive of four nodes. Huxley-G\"odel Machine selection draws a parent by Thompson sampling on the steering counts of nodes that have already been evaluated, and it expands the child that parent proposes. Every method then returns the earliest evaluated node whose steering-split solve count is maximal. The tables report the held-out score of that node on the other 100 HumanEval+ instances and the other 145 Polyglot insta
    
[^146]: MHeatmap：用于矩形矩阵可视化的双游走拉普拉斯重排序与马赛克布局

    MHeatmap: Two-Walk Laplacian Reordering and Mosaic Layout for Rectangular Matrix Visualization

    [https://arxiv.org/abs/2610.29114](https://arxiv.org/abs/2610.29114)

    该论文提出MHeatmap方法，通过在加权二部图中引入长度为2的游走相似性连接并基于Fiedler向量排序实现行列协调重排，同时采用行列尺寸与对应总和成比例的马赛克布局，解决了矩形矩阵热力图中相关对象分散、单元格大小掩盖数据总量差异的可读性问题。

    

    加权矩形矩阵记录了两类对象之间的关系，例如职业与行业、文档与词项、出发地与目的地，或人工标注类别与计算得到的聚类。当任意的行列顺序使相关对象分散开来时，这些矩阵的热力图会变得难以阅读，而等大小的单元格则掩盖了行总和与列总和之间的巨大差异。我们提出了MHeatmap（马赛克热力图），分两个阶段解决这些问题。首先，双游走谱序列化将矩阵表示为加权二部图。普通的二部图排序仅利用观测到的行到列的连接，而双游走还利用长度为2的游走，使共享列的行之间、共享行的列之间获得显式的相似性连接。对由此得到的图拉普拉斯矩阵的Fiedler向量进行排序，即可获得相互协调的行序和列序。随后，马赛克布局对单位正方形进行划分，使行高与行总和成比例，列（摘要原文在此处截断）

    arXiv:2610.29114v1 Announce Type: new  Abstract: Weighted rectangular matrices record relations between two types of objects, such as occupations and industries, documents and terms, origins and destinations, or annotated classes and computed clusters. Their heatmaps become difficult to read when arbitrary row and column orders scatter related objects, while equal size cells hide large differences in row and column totals. We present MHeatmap (Mosaic Heatmap), which addresses these problems in two stages. Two-Walk spectral seriation first represents the matrix as a weighted bipartite graph. Ordinary bipartite ordering uses only observed row to column links. Two-Walk also uses length-two walks, so rows that share columns and columns that share rows receive explicit similarity links. Sorting a Fiedler vector of the resulting graph Laplacian gives coordinated row and column orders. The Mosaic layout then partitions a unit square so that row height is proportional to the row total and colu
    
[^147]: 面向分子磁性的SU(2)-等变旋量消息传递

    SU(2)-Equivariant Spinor Message Passing for Molecular Magnetism

    [https://arxiv.org/abs/2610.29111](https://arxiv.org/abs/2610.29111)

    本文提出eSpinor，一个通过传播旋量特征并结合规范协变消息传递与规范不变读出的SU(2)-等变图神经网络，显著提升了分子磁性性质（如零场分裂和电子自旋关联）的预测精度。

    

    预测分子磁性通常需要进行包含自旋轨道耦合（SOC）的电子结构计算，而在大量分子上重复此类计算代价高昂。其底层电子态可以按旋量（spinor）形式变换，然而现有的等变代理模型通常仅使用整数阶特征。受这些电子态旋量性质的启发，我们开发了eSpinor——一种传播旋量特征的SU(2)-等变图神经网络。我们从近似SOC哈密顿量的本征向量初始化旋量特征，并采用规范协变的消息传递与规范不变的读出层，使预测结果独立于任意的本征基选择。在钴基单分子磁体候选物上，eSpinor相较于NequIP改进了零场分裂的预测，包括对未见过的配体。对于一种镝基磁共振成像探针，eSpinor在真实电子自旋关联方面也优于基于哈密顿量信息的基线方法。

    arXiv:2610.29111v1 Announce Type: cross  Abstract: Predicting molecular magnetism often requires electronic-structure calculations with spin-orbit coupling (SOC), which are costly to repeat across many molecules. The underlying electronic states can transform as spinors, yet equivariant surrogates typically use only integer-rank features. Inspired by the spinor nature of these electronic states, we develop eSpinor, an SU(2)-equivariant graph neural network that propagates spinor features. We initialize spinor features from eigenvectors of an approximate SOC Hamiltonian and use gauge-covariant message passing with a gauge-invariant readout to make predictions independent of the arbitrary eigenbasis choice. On cobalt single-molecule magnet candidates, eSpinor improves zero-field-splitting predictions relative to NequIP, including for unseen ligands. For a dysprosium magnetic resonance imaging probe, eSpinor improves real electron-spin correlations over a Hamiltonian-informed baseline des
    
[^148]: DriftNeRF：面向稀疏视角神经辐射场的跨射线密度漂移

    DriftNeRF: Cross-Ray Density Drifting for Sparse-View Neural Radiance Fields

    [https://arxiv.org/abs/2610.29106](https://arxiv.org/abs/2610.29106)

    该论文提出DriftNeRF，一种即插即用的正则化器，通过带掩码的指数核从其他射线聚合邻近点密度，解决了稀疏视角下NeRF目标函数因缺乏跨射线密度一致性约束而导致的密度矛盾问题。

    

    神经辐射场（NeRF）能够从密集的图像集合中合成照片级真实感的新视角，但在可用视角较少时性能会急剧下降。我们将一个具体的、可度量的缺陷归因于其目标函数本身：渲染损失是对各独立射线的求和，其中不包含任何用于比较不同射线赋予空间同一区域密度的项，因此在每个训练像素都能被正确再现的同时，相邻的三维点却可能收敛到相互矛盾的密度值。我们通过在训练过程中直接测量跨射线密度不一致性，使这一量变得显式，并证明它在原始NeRF中始终维持较高水平，在基于频率的正则化下只是偶然下降，而且在稀疏视角场景中与留出集渲染质量同步变化。随后，我们提出DriftNeRF来弥补这一缺陷——它是一种即插即用的正则化器，在密度样本上实例化了漂移场模板：利用一个带掩码的指数核，从其他射线上的邻近点聚合密度信息（原文摘要在此处截断）。

    arXiv:2610.29106v1 Announce Type: cross  Abstract: Neural Radiance Fields (NeRF) synthesize photorealistic novel views from dense image collections, but degrade sharply when few views are available. We attribute a specific, measurable defect to the objective itself: the rendering loss is a sum over independent rays and contains no term comparing densities that different rays assign to the same region of space, so nearby 3D points can settle on contradictory densities while every training pixel is reproduced correctly. We make this quantity explicit by measuring cross-ray density disagreement directly during training, and show that it stays high in vanilla NeRF, falls only incidentally under frequency-based regularization, and moves together with held-out quality in the sparse regime. We then close the gap with DriftNeRF, a drop-in regularizer that instantiates the drifting-field template on density samples: a masked exponential kernel aggregates densities from nearby points on other ra
    
[^149]: 抹香鲸尾音序列的双通道对比编码

    Dual-Channel Contrastive Encoding of Sperm Whale Coda

    [https://arxiv.org/abs/2610.29101](https://arxiv.org/abs/2610.29101)

    本文提出双通道对比编码器（DCCE），通过分别编码并融合抹香鲸尾音的节奏与频谱信息，在社会单元、尾音类型和个体识别三个任务上取得了优异性能。

    

    抹香鲸（Physeter macrocephalus）通过称为尾音序列的短促咔嗒声序列进行交流，其结构包括咔嗒声间隔（ICI）的节奏模式和咔嗒声的频谱组成。本文提出了双通道对比编码器（DCCE），该方法使用独立的编码器分别显式编码节奏信息和频谱信息，然后再进行融合。我们在三个尾音理解任务上评估了DCCE：社会单元识别、尾音类型识别和个体身份识别。我们在统一的线性探测协议下将DCCE与文献中的强基线方法进行比较，并通过消融实验分析了每个通道和训练目标的作用。在1,383个尾音序列上，采用共享的数据划分和仅基于验证集的层选择，支持三种随机划分和三种录音日期不相交的交叉验证。采用单元配对和类型监督的DCCE在单元/类型/身份任务上实现了平均随机划分的宏F1分数0.901/0.865/0.597。原始ICI数据仍然……（摘要截断）

    arXiv:2610.29101v1 Announce Type: cross  Abstract: Sperm whales (Physeter macrocephalus) communicate through short click sequences called codas, whose structure includes the rhythmic pattern of inter-click intervals (ICI) and the spectral composition of the clicks. In this paper, we propose the Dual-Channel Contrastive Encoder (DCCE), which explicitly encodes rhythm and spectral information with separate encoders before fusing them. We evaluate DCCE on three coda-understanding tasks: social unit, coda type, and individual identification. We compare DCCE against strong baselines from literature under a common linear-probe protocol and analyze the role of each channel and training objective through ablations. On 1,383 codas, shared partitions and validation-only layer selection support three random splits and three recording-date-disjoint folds. DCCE with unit pairing and type supervision achieves mean random-split macro-F1 of 0.901/0.865/0.597 for unit/type/identity. Raw ICI remains str
    
[^150]: 物理参数偏移下的因果世界模型

    Causal world models under physical parameter shifts

    [https://arxiv.org/abs/2610.29088](https://arxiv.org/abs/2610.29088)

    该论文将因果建模的两个关键组件——由成对干预导出的依赖掩码和显式的物理参数化机制——融入世界模型，使机器人在连杆质量等物理参数超出训练分布时仍能保持接近预言水平的准确预测。

    

    世界模型用于预测具身系统在动作作用下如何演化，但当物理条件超出训练分布时，其准确性可能下降。我们研究了因果建模的两个组成部分：预测器可以使用哪些输入，以及其机制如何计算这些输入的影响。在一个带有捷径特征的合成基准测试中，通过成对干预导出的冻结依赖掩码，将全连接预测器的平均分布外（OOD）RMSE从1.769降至0.697，接近生成器预言掩码的0.694。针对CausalWorld中的机器人连杆质量外推任务（训练与测试的质量范围互不重叠），我们开发了手指局部世界模型：连杆质量通过显式仿射系数进入惯性和动力学偏置项，动作通过学习到的扭矩网络或已知的PD控制器施加作用，并通过学习到的残差来捕捉与接触相关的效应。（注：摘要原文在此处截断）

    arXiv:2610.29088v1 Announce Type: new  Abstract: World models predict how embodied systems evolve under actions, but their accuracy can degrade when physical conditions move outside the training distribution. We study two components of causal modeling: which inputs a predictor may use, and how its mechanism computes their effects. In a synthetic benchmark with shortcut features, a frozen dependency mask derived from paired interventions reduces mean out-of-distribution (OOD) RMSE from 1.769 for a fully connected predictor to 0.697, close to the generator-oracle mask (0.694). For robot link-mass extrapolation in CausalWorld, with disjoint training and test mass ranges, we develop finger-local world models in which link masses enter inertia and dynamics-bias terms through explicit affine coefficients, actions act through either a learned torque network or the known PD controller, and learned residuals capture contact-related effects. Compared with a separately trained control whose mass 
    
[^151]: 内部引导：一种用于倾斜扩散模型的采样方法

    Internal Guidance as a Sampling Method for Tilted Diffusion Models

    [https://arxiv.org/abs/2610.29086](https://arxiv.org/abs/2610.29086)

    本文提出“内部引导”方法，通过改变基础去噪器的求值输入点而非修改其输出来采样倾斜密度，并利用随机对角重缩放改进训练算法，解决了所需求值点超出训练分布的问题。

    

    奖励引导方法使用仅由基础分布样本训练的扩散模型，来对基础密度的指数倾斜分布进行采样。现有方法采用我们所说的“外部引导”：即在漂移项中加入加性项，或在转移概率上加入乘性项。我们提出了“内部引导”，它不修改基础去噪器的输出，而是改变对其求值的输入点。对于嵌入在超立方体 $\{-1,1\}^{d}$ 中的离散分布，任何线性倾斜密度的去噪器恰好等于基础去噪器在经过平移和重新缩放的输入上的求值结果。在实践中，基础去噪器是在基础分布的加噪样本上训练的神经网络。对倾斜密度进行采样所需的求值点，相对于基础分布的基本加噪过程而言超出了训练分布范围。我们通过对数据进行随机对角重缩放来修改训练算法，使去噪器能够（摘要在此处被截断）

    arXiv:2610.29086v1 Announce Type: cross  Abstract: Reward guidance methods sample an exponential tilt of a base density using a diffusion model trained with only samples from the base distribution. Existing methods use what we call external guidance: additive terms in the drift or multiplicative terms on the transition probabilities. We propose internal guidance which leaves the output of the base denoiser unmodified and instead changes the point at which it is evaluated. For discrete distributions embedded in the hypercube $\{-1,1\}^{d}$, the denoiser of any linearly tilted density is exactly the base denoiser evaluated at a shifted and rescaled input. In practice the base denoiser is a neural network trained on noised samples of the base distribution. The required evaluation points for sampling the tilted density would be out-of-distribution for basic noising of the base distribution. We modify the training algorithm with random diagonal rescalings of the data, so that the denoiser n
    
[^152]: 面向基于流的离线强化学习的价值校正蒸馏

    Value-Rectified Distillation for Flow-based Offline Reinforcement Learning

    [https://arxiv.org/abs/2610.29075](https://arxiv.org/abs/2610.29075)

    提出价值校正蒸馏（VRD）框架，将离线策略蒸馏重新表述为价值引导的生成式轨迹对齐问题，通过动态耦合机制校正流动力学以解耦多模态行为并分离高价值动作，从而解决离线强化学习中的分布外模态平均问题。

    

    我们旨在解决分布外（OOD）模态平均问题，这是在离线强化学习（RL）中蒸馏基于流匹配（FM）的行为策略时出现的一个关键问题：在多模态数据集上学习时，标准蒸馏会无意中迫使策略在不同行为模态之间进行插值，导致生成的动作坍缩到无效的分布外区域。为解决这一问题，我们提出价值校正蒸馏（VRD），这是一个将离线策略蒸馏重新表述为价值引导的生成式轨迹对齐问题的新颖框架。通过采用动态耦合机制，VRD主动校正流动力学，以显式解耦不同的行为模态，从而干净地分离出高价值动作。具体而言，我们通过两种互补算法来实例化VRD：其中VRD-Q利用价值分位数行为蒸馏从统计上过滤掉低价值行为

    arXiv:2610.29075v1 Announce Type: new  Abstract: We aim to address the problem of Out-of-Distribution (OOD) Mode Averaging, a critical problem that emerges when distilling Flow Matching (FM) based behavior policies in offline Reinforcement Learning (RL): When learning over multi-modal datasets, standard distillation inadvertently forces the policy to interpolate across different behavioral modes, causing the generated actions to collapse into invalid, OOD regions. To resolve this, we propose Value-Rectified Distillation (VRD), a novel framework that reformulates offline policy distillation as a value-guided generative trajectory alignment problem. By employing a dynamic coupling mechanism, VRD actively rectifies flow dynamics to explicitly decouple distinct behavioral modes, cleanly isolating high-value actions. Specifically, we instantiate VRD through two complementary algorithms: VRD-Q, which utilizes value-quantile behavior distillation to statistically filter out low-value behavior
    
[^153]: 选择性预测中选择的代价

    The Price of Choice in Selective Prediction

    [https://arxiv.org/abs/2610.29067](https://arxiv.org/abs/2610.29067)

    本文刻画了选择性预测中审计“存在满足风险-覆盖契约的规则”这一断言时的代价：交并测试（IUT）所需的极小极大样本量随允许规则数K线性增长，且该速率在忽略对数因子的意义下是最优的。

    

    arXiv:2610.29067v1 公告类型: new 摘要：如果预先声明的规则菜单（一个阈值或K个分数区间的并集）中的某条规则能够达到至少κ的覆盖率且错误率至多为α，则可宣称该选择性分类器满足风险-覆盖契约。要驳斥这一存在性断言，需要排除所有被允许的规则。交并测试（IUT）无需多重性校正即可实现有效性，但其样本成本可能随菜单的丰富程度而增加。我们刻画了这一成本，并将其与最优审计进行比较。对于固定的内部风险目标α和间隙γ，在所述的小间隙、大菜单情形下，当覆盖下限为κ时，K区间菜单的极小极大样本量为K(1-κ)/(κγ²)（忽略对数因子）；精确的IUT可达到这一速率。该下界即使在分数分布已知的情况下依然成立，而上界可扩展至VC维为d的菜单。当任何规则看似可行时都拒绝拒绝（即不拒绝）的程序……（摘要在此处被截断）

    arXiv:2610.29067v1 Announce Type: new  Abstract: A selective classifier may be claimed to satisfy a risk-coverage contract if some rule from a predeclared menu (a threshold or a union of $K$ score intervals) attains coverage at least $\kappa$ and error at most $\alpha$. Refuting this existential claim requires ruling out every permitted rule. The intersection-union test (IUT) achieves validity without a multiplicity correction, but its sample cost can increase with menu richness. We characterize this cost and compare it with the optimal audit. For a fixed interior risk target $\alpha$ and gap $\gamma$, the minimax sample size for K-interval menus at coverage floor $\kappa$ is $K(1-\kappa)/(\kappa\gamma^2)$, up to logarithmic factors, in the stated small-gap, large-menu regime; exact IUT attains this rate. The lower bound holds even with a known score distribution, while the upper bound extends to menus of VC dimension $d$. Procedures that refuse to reject whenever any rule appears feas
    
[^154]: 面向无监督时间序列异常检测的多教师评分原则蒸馏

    Distilling Multi-Teacher Scoring Principles for Unsupervised Time Series Anomaly Detection

    [https://arxiv.org/abs/2610.29063](https://arxiv.org/abs/2610.29063)

    AnoMix提出了一种无监督多教师蒸馏框架，利用CoRank多教师排序蒸馏目标，将多个异构异常检测器的互补评分原则蒸馏进单个轻量级学生检测器，无需任何异常标签即在七个基准数据集上实现有效整合。

    

    时间序列异常具有多种多样的形式，而单个检测器所编码的异常评分原则往往难以覆盖所有异常类型。近期的无监督时间序列异常检测器能够提供互补的评分信号，但将它们作为集成部署时，仍会保留每个单独检测器的推理和维护成本。我们研究在无监督设置下——即没有异常标签可用于定义共享目标时——能否将异构的教师检测器蒸馏为单个轻量级检测器。AnoMix通过从冻结教师模型的训练集异常评分中提取弱成对监督来解决这一问题。它对每个教师的评分序列独立进行归一化，并使用CoRank——一种多教师排序蒸馏目标——来训练学生模型，该目标通过教师共识将教师评分差距转化为成对监督。在七个基准数据集上，AnoMix成功地将协同互补的教师信号整合到了单个学生检测器中。

    arXiv:2610.29063v1 Announce Type: new  Abstract: Time series anomalies take diverse forms, and individual detectors encode different anomaly scoring principles that rarely cover all anomaly types. Recent unsupervised time-series anomaly detectors provide complementary scoring signals, yet deploying them as an ensemble preserves the inference and maintenance cost of every individual detector. We study whether heterogeneous teacher detectors can be distilled into a single lightweight detector in the unsupervised setting, where no anomaly labels are available to define a shared target. AnoMix addresses this by extracting weak pairwise supervision from the training-set anomaly scores of frozen teachers. It normalizes each teacher score sequence independently and trains the student with CoRank, a multi-teacher ranking-distillation objective that converts teacher score gaps into pairwise supervision through teacher consensus. Across seven benchmark datasets, AnoMix successfully integrates co
    
[^155]: OccJEPA：将遮挡视为掩码的完整视频分割

    OccJEPA: Occlusion as Masking for Amodal Video Segmentation

    [https://arxiv.org/abs/2610.29061](https://arxiv.org/abs/2610.29061)

    OccJEPA 将视频中的遮挡视为掩码任务，利用预训练的 V-JEPA 2.1 预测器填补被遮挡区域并解码出完整掩码，实现了确定性的完整视频分割，比基于扩散模型的最先进方法快 33.5 倍且精度更高。

    

    完整视频分割要求预测每帧中物体的完整掩码，包括被其他物体遮挡而隐藏的部分。目前最好的方法使用微调后的视频扩散模型来生成这些掩码，但这类方法需要大量的去噪步骤和一个额外的深度网络，且每次采样给出的结果都不同。JEPA 家族的自监督视频世界模型是在一个密切相关的任务上训练的：从可见区域预测被掩码遮蔽的时空区域的表征。因此，我们将遮挡视为掩码：物体可见掩码未覆盖的 token 被丢弃，由预训练的 V-JEPA 2.1 预测器填补这些 token，然后通过一个轻量级头将观测到的和预测的特征解码为完整掩码。由此得到的模型 OccJEPA（Occlusion-as-masking JEPA）是确定性的，比基于扩散的最先进方法快 33.5 倍，并且在 SAIL-VOS 和 TAO-Amodal 上（包括保留测试集）更加准确。

    arXiv:2610.29061v1 Announce Type: cross  Abstract: Amodal video segmentation asks for the complete mask of an object in every frame, including the parts hidden behind other objects. The best current methods generate these masks with fine-tuned video diffusion models, which need many denoising steps and an extra depth network, and give a different answer for every sample. Self-supervised video world models of the JEPA family are trained on a closely related task: predicting the representation of masked space-time regions from the visible ones. We therefore treat occlusion as masking. The tokens that the object's visible mask does not cover are dropped, the pre-trained V-JEPA 2.1 predictor fills them in, and a light head decodes the observed and predicted features into amodal masks. The resulting model, OccJEPA (Occlusion-as-masking JEPA), is deterministic and 33.5x faster than the diffusion-based state of the art, and more accurate on SAIL-VOS, on TAO-Amodal, including a held-out test s
    
[^156]: 隐藏学习结构的动力学

    Dynamics of Hidden Learning Structure

    [https://arxiv.org/abs/2610.29054](https://arxiv.org/abs/2610.29054)

    本文提出纤维调节理论与纤维获取理论，刻画了隐藏学习结构中“保护性修复”与“可训练结构获取”这两种操作及其相互约束关系，并通过有限监督学习构造证明二者联合可产生任一操作单独无法实现的有利学习响应。

    

    学习者当前的预测并不能完全决定其接下来将如何学习。我们通过两个相互关联的操作来研究隐藏学习结构的动力学。纤维调节理论描述了对现有功能的保护性修复，以及修复之后仍然可用的学习响应。纤维获取理论描述了可训练结构的形成与更新，并区分了中性安装、对学习信号的访问以及额外的功能方向。这两种操作还会相互约束：一项修复义务可能迫使某个学习通道发生改变，而新获取的特征既可能使新的保护性修复变得可行，也可能降低现有修复的最小系数成本。我们刻画了联合可容许性与响应可达性，并给出了一个有限监督学习构造，其中这两种操作结合在一起能够产生一种仅靠任一单独操作都无法实现的有利响应。我们实现了……（摘要在此处截断）

    arXiv:2610.29054v1 Announce Type: new  Abstract: A learner's present predictions do not fully determine how it will learn next. We study the dynamics of hidden learning structure through two related operations. Fiber Regulation Theory (FRT) describes protected repair of existing function and the learning responses left available after repair. Fiber Acquisition Theory (FAT) describes the formation and renewal of trainable structure, distinguishing neutral installation, access to a learning signal, and additional functional directions. The two operations also constrain one another. A repair obligation can force a change in a learning channel, while acquired features can make new protected repairs feasible or reduce the minimum coefficient cost of existing ones. We characterize joint admissibility and response reachability, and give a finite supervised-learning construction in which the two operations together produce a favorable response absent from either operation alone. We implement p
    
[^157]: 我们自己的Jev：单次前向传播中的类型化决策，从自研裁判到开放LoRA模型

    Our Own Jev: Typed Decisions in One Forward Pass, from an In-House Judge to Open LoRA Models

    [https://arxiv.org/abs/2610.29052](https://arxiv.org/abs/2610.29052)

    该论文提出用单次前向传播的“类型化决策”接口（直接读取logits或轻量指针头）替代生成式LLM裁判，其自研System 1裁判仅用一块GPU约3分钟、约0.06美元即可判定1,000个智能体编码会话，以约1%的成本取代每轮约1,000美元且标签一致性仅26.9%的前沿LLM裁判。

    

    类型化决策将证据、问题和一个固定的选项集映射为每个选项的概率，只需一次前向传播即可完成，方式是读取语言模型的logits或在其隐藏状态上附加一个小型头部，而不是生成并解析文本。商业系统Jev推广了这种System 1接口，开源复现也随之出现。我们报告了该模式的两种用途。第一部分描述了一个自研的System 1裁判，它为一个训练语料库筛选了约14.8万条智能体编码会话记录。它取代了一个前沿LLM裁判——后者每轮处理的成本约在1,000美元量级，且在两位裁判达成一致的会话中，其存储的标签与新做的双裁判共识仅有26.9%吻合；一个自托管的冻结骨干网络，配合秩为16的LoRA适配器和指针头，可在一次前向传播中回答三个类型化问题，在一块GPU上约三分钟内即可判定1,000个会话，按参考价格计算成本约0.06美元，大约仅为前沿模型成本的1%……

    arXiv:2610.29052v1 Announce Type: new  Abstract: A typed decision maps evidence, a question and a fixed set of options to a probability for every option, computed in one forward pass, by reading a language model's logits or a small head over its hidden states, instead of generating and parsing text. The commercial system Jev popularised this System~1 interface, and open reproductions have followed. We report two uses of the pattern. \emph{Part I} describes an in-house System~1 judge that triages about 148K agentic coding session records for a training corpus. It replaces a frontier LLM judge that cost on the order of \$1,000 per pass and whose stored labels matched a fresh two-judge consensus on only 26.9\% of the sessions where the two judges agreed; a self-hosted frozen backbone with a rank-16 LoRA adapter and a pointer head answers three typed questions in one forward pass, judges 1,000 sessions in about three minutes of one GPU, about \$0.06 at a reference rate and roughly 1\% of t
    
[^158]: ART-MPG：自回归反射/透射流形路径引导

    ART-MPG: Autoregressive R/T Manifold Path Guiding

    [https://arxiv.org/abs/2610.29049](https://arxiv.org/abs/2610.29049)

    提出ART-MPG，一种通过自回归采样镜面链的长度与反射/透射拓扑字符串、并以拓扑条件化vMF混合分布采样方向，同时将重试采样同时条件化于链长度和R/T字符串的紧凑神经模型，在保持无偏性的前提下降低了流形路径引导的预期重试次数和估计器方差。

    

    镜面链将重要的光传输集中在路径空间的小区域内，其结构取决于链的拓扑。我们提出了ART-MPG，一个紧凑的神经模型，它对链长度、自回归的反射/透射（R/T）字符串进行采样，并从以拓扑为条件的四分量von Mises-Fisher（vMF）混合分布中采样方向。我们扩展了流形路径引导（MPG）中的重复试验加权方法，将重试采样同时以链长度和采样的R/T字符串为条件，并考虑了其在随机重命中试验之外已知的概率质量。对于固定的提议分布和目标链，与仅以链长度为条件的重试相比，该方法在保持理想期望的同时，降低了预期重试次数和倒数估计器的方差。我们将形式上的无偏性结果与实验中使用的带封顶和钳位处理的渲染器区分开来。我们在十四个场景上、在一系列每像素采样数（SPP）设置下进行评估（摘要在此处截断）。

    arXiv:2610.29049v1 Announce Type: new  Abstract: Specular chains concentrate important transport in small regions of path space and the structure depends on chain topology. We introduce ART-MPG, a compact neural model that samples chain length, an autoregressive reflection/transmission (R/T) string, and a direction from a topology-conditioned four-component von Mises-Fisher (vMF) mixture. We extend the repeated trial weighting in Manifold Path Guiding (MPG) by conditioning retry sampling on both chain length and the sampled R/T string, accounting for its known mass outside the stochastic re-hit trials. For a fixed proposal and target chain, this preserves the ideal expectation while reducing the expected retry count and reciprocal-estimator variance compared with retries conditioned on chain length alone. We distinguish the formal unbiasedness result from the capped and clamped renderer used in experiments. We evaluate fourteen scenes across a range of samples per pixel (SPP) against e
    
[^159]: 门控初始化与循环Dropout及Zoneout在学习长程依赖中的对比：LSTM与GRU网络在合成任务上的受控研究

    Gate Initialisation versus Recurrent Dropout and Zoneout in Learning Long Dependencies: A Controlled Study of LSTM and GRU Networks on Synthetic Tasks

    [https://arxiv.org/abs/2610.29034](https://arxiv.org/abs/2610.29034)

    该研究通过720次运行的受控析因实验，首次系统交叉比较了门控初始化方式与循环Dropout/Zoneout放置位置对LSTM和GRU学习长程依赖的影响，发现chrono初始化在困难序列长度下使全部360次运行均获得成功。

    

    Dropout和zoneout通常被例行添加到长短期记忆网络（LSTM）和门控循环单元（GRU）网络中，但据我们所知，它们的放置位置尚未在析因设计中与门控初始化进行交叉组合，而且早期梯度传播能否预测哪些运行将学会长程依赖也尚不清楚。我们在求和问题和延迟回忆任务上，以720次运行的析因网格训练单层LSTM和GRU网络，将常规门控初始化与chrono门控初始化同最多六种循环正则化放置位置（每种各采用一个比率）交叉组合，并进行了预先设定的后续实验，同时记录了最终隐藏状态对每个滞后位置输入的敏感程度。由于这两个任务都提供新鲜或穷尽的训练数据，本研究衡量的是正则化方法对优化的影响，而非评估有限训练集上的泛化能力。在特意选定的使常规初始化难以奏效的序列长度下，所有360次chrono初始化的运行（摘要在此处截断）

    arXiv:2610.29034v1 Announce Type: new  Abstract: Dropout and zoneout are routinely added to long short-term memory (LSTM) and gated recurrent unit (GRU) networks, but, to our knowledge, their placements have not been crossed with gate initialisation in a factorial design, and it is not known whether early gradient propagation tells which runs will learn long dependencies. We trained one-layer LSTMs and GRUs on the adding problem and on delayed recall in a factorial grid of 720 runs, crossing conventional and chrono gate initialisation with up to six recurrent regulariser placements at one rate each, with pre-specified follow-ups, and recorded how sensitive the final hidden state was to the input at every lag. Because both tasks supply fresh or exhaustive training data, the study measures the regularisers' effects on optimisation and does not assess generalisation from a finite training set. At lengths chosen to make conventional initialisation difficult, all 360 chrono-initialised runs
    
[^160]: 更好的策略，更差的验证器审计

    Better Policies, Worse Verifier Audits

    [https://arxiv.org/abs/2610.29029](https://arxiv.org/abs/2610.29029)

    本文揭示了在策略改进闭环中，随着策略掌握部分子目标，基于聚合分数的组件验证器审计会出现“同策略审计崩塌”现象，导致审计反而偏好更差的验证器，并给出了精确的识别集、完全擦除阈值与崩塌临界点的理论刻画。

    

    验证器堆栈越来越多地嵌入到后训练与测试时计算循环之中。我们研究了一种特定的控制回路：组件验证器在策略新生成的输出上针对完整成功标签进行评估，当该聚合分数下降时便进行刷新。对于合取目标 $y = F \wedge M$，其同策略AUC为 $(1-q)A_{\mathrm{own}} + qA_{\mathrm{cross}}$，其中 $q = \Pr[F=1 \mid y=0]$。随着策略逐渐掌握 $F$，$q$ 可能趋近于一，从而去除 $F$ 失败样本的权重，并奖励有关 $M$ 的信息。我们将这一现象称为“同策略审计崩塌”。我们推导出了清晰的识别集、完全擦除阈值，以及审计可能偏好更差组件验证器的确切临界点；其总体最优解是完整结果预测器，而非组件预言机。精确的预言机展示了这种崩塌现象，一个自然的多标签文本对照实验表明，在观察到的失败混合下，反演可能会选出具有更低规范 $F$-AUC 的头部……

    arXiv:2610.29029v1 Announce Type: new  Abstract: Verifier stacks increasingly sit inside post-training and test-time compute loops. We study a specific control loop: a component verifier is evaluated on fresh policy outputs against the full success label and refreshed when that aggregate score falls. For a conjunctive target $y = F \wedge M$, its on-policy AUC is $(1-q)A_{\mathrm{own}} + qA_{\mathrm{cross}}$, where $q = \Pr[F=1 \mid y=0]$. As the policy masters $F$, $q$ can rise toward one, removing weight from $F$-failures and rewarding information about $M$. We call this on-policy audit collapse. We derive a sharp identified set, a complete-erasure threshold, and the exact point at which the audit can prefer a worse component verifier; its population optimum is the full-outcome predictor, not the component oracle. An exact oracle exhibits the collapse, a natural multi-label text control shows inversion can select a head with lower canonical $F$-AUC at the observed failure mixture, an
    
[^161]: 用于稳定状态分布的基于相异度的马尔可夫决策过程

    Dissimilarity-Based Markov Decision Processes for Stabilizing State Distributions

    [https://arxiv.org/abs/2610.29024](https://arxiv.org/abs/2610.29024)

    本文提出基于相异度的马尔可夫决策过程（DMDP）理论框架，其中策略作用于状态分布并通过最小化与目标分布的相异度进行优化，建立了最优策略使目标分布渐近稳定的充分条件，并通过分布的有限维表示将稳定性保证扩展到底层状态分布。

    

    针对随机系统的强化学习通常优化期望累积奖励，然而可靠且可重复的行为还取决于状态分布随时间的演化方式。仅优化期望回报并不能保证该分布收敛到期望的目标分布或保持稳定。本文开发了一个用于学习能够稳定状态分布的策略的理论框架。我们将该问题表述为基于相异度的马尔可夫决策过程（DMDP），其中策略作用于当前状态分布，目标函数衡量其与目标分布的相异度。我们建立了充分条件，在这些条件下最优策略能使目标分布渐近稳定。为支持策略学习，我们引入了分布的有限维表示，并展示了其参数动力学的稳定性保证如何扩展到底层的状态分布。

    arXiv:2610.29024v1 Announce Type: new  Abstract: Reinforcement learning for stochastic systems typically optimizes expected cumulative rewards, while reliable and repeatable behavior also depends on how the state distribution evolves over time. Optimizing expected return alone does not guarantee that this distribution converges to a desired target or remains stable. This paper develops a theoretical framework for learning policies that stabilize state distributions. We formulate the problem as a Dissimilarity-based Markov Decision Process (DMDP), in which the policy acts on the current state distribution and the objective measures its dissimilarity to a target distribution. We establish sufficient conditions under which the optimal policy makes the target distribution asymptotically stable. To support policy learning, we introduce a finite-dimensional representation of distributions and show how stability guarantees for its parameter dynamics extend to the underlying state distribution
    
[^162]: 目标重生成削弱了投机解码中草稿训练目标的收益

    Target Regeneration Shrinks the Gains of Draft-Training Objectives in Speculative Decoding

    [https://arxiv.org/abs/2610.29013](https://arxiv.org/abs/2610.29013)

    研究发现，投机解码中新的草稿训练目标（如令牌级蒸馏和验证门控蒸馏）在外部回复数据上的收益，在经过目标模型贪婪重生成后大幅缩水，且该现象在多种训练与解码设置下均稳定存在。

    

    投机解码通过并行验证草稿令牌来加速大型语言模型的推理。新的草稿训练目标通常被认为是相比交叉熵在对外部撰写的回复上带来更高接受率的原因，然而许多训练流水线会让目标模型重写这些回复。我们探究了这些收益有多少能够保留下来。通过一个视觉-语言目标模型和一个块草稿器，我们在同一语料库的四个版本上训练了三种目标：语料令牌上的交叉熵、来自目标模型的令牌级蒸馏，以及仅在能通过验证的位置上启用的门控蒸馏。在外部回复上，蒸馏和门控都带来了显著收益，但在经过贪婪重生成后，蒸馏的收益变小，而门控几乎或完全没有收益。这种缩减在不同数据顺序、位置权重、验证器、解码温度下持续存在；对于蒸馏而言，在不同初始化和学习率下也是如此。在匹配监督长度以及每个目标的b……（原文截断）

    arXiv:2610.29013v1 Announce Type: new  Abstract: Speculative decoding accelerates large language models by verifying drafted tokens in parallel. New draft-training objectives are usually credited with the acceptance they add over cross-entropy on externally written responses, yet many pipelines let the target rewrite them. We ask how much of that gain survives. With a vision-language target and a block drafter, we train three objectives on four versions of one corpus: cross-entropy on corpus tokens, token-level distillation from the target, and distillation gated to positions that would survive verification. Distillation and the gate add large gains on external responses, but after greedy regeneration, distillation adds a smaller gain and the gate little or none. The shrinkage persists across data orders, position weights, verifiers, decoding temperatures and, for distillation, initializations and learning rates. Most of it remains at matched supervised length and at each objective's b
    
[^163]: 面向不可压缩流的局部稀疏核算子学习：算子拉格朗日表述

    Local and Sparse Kernel Operator Learning for Incompressible Flows: An Operator Lagrange Formulation

    [https://arxiv.org/abs/2610.29009](https://arxiv.org/abs/2610.29009)

    该论文提出一种局部稀疏核算子学习方法，用GPU并行求解的局部无散度插值替代全局重构，用紧支撑核实现稀疏算子回归，并通过算子拉格朗日权重高效结合两者，从而高效学习不可压缩流的解算子。

    

    我们提出了一种用于学习不可压缩流解算子的局部稀疏核方法。该方法建立在我们早期的保性质核方法之上，后者因全局输出重构和稠密算子回归而扩展性较差。我们的方法用局部无散度插值替代全局输出重构，这些插值采用任意阶的无散度多项式，可在GPU上并行求解。这些插值是无散度的，在周期域上满足周期性，并且借助多尺度核，在细化过程中逼近Kolmogorov统计特性。一种在任意输入维度下均为半正定的紧支撑核使算子回归变得稀疏，其支撑半径（即稀疏度）根据数据拟合得到，同时输入中可选的仿射多项式用于捕捉全局趋势。算子拉格朗日权重将二者结合起来：每个新输入只需一次稀疏求解，并在输出重构之前作用于训练场。

    arXiv:2610.29009v1 Announce Type: new  Abstract: We present a local and sparse kernel method for learning the solution operators of incompressible flows. It builds on our earlier property-preserving kernel method, whose global output reconstruction and dense operator regression scale poorly. Our method replaces the global output reconstruction by local divergence-free interpolants with divergence-free polynomials of any degree, solved in parallel on the GPU. These interpolants are divergence free, periodic on periodic domains, and, with a multiscale kernel, approach Kolmogorov statistics under refinement. A compactly supported kernel, positive semidefinite in any input dimension, makes the operator regression sparse. Its support radius, and hence the sparsity, is fitted to data, and optional affine polynomials in the input capture global trends. Operator Lagrange weights join the two. They cost one sparse solve per new input and act on the training fields before the output reconstructi
    
[^164]: 基于能量核的最大均值差异的跳跃修正Wasserstein梯度流

    A Jump-Modified Wasserstein Gradient Flow for Maximum Mean Discrepancy with Energy Kernels

    [https://arxiv.org/abs/2610.29007](https://arxiv.org/abs/2610.29007)

    本文提出一种带跳跃项的修正Wasserstein梯度流来最小化能量核生成的最大均值差异（MMD），证明了该流的全局适定性、仅凭有限二阶矩即可收敛到目标分布，并构造了能量单调不增的全局无碰撞粒子过程。

    

    我们研究由能量核 $K(z)=-|z|^q$（在每一维 $d\ge2$ 时 $0<q<2$，以及一维情形 $1\le q<2$）生成的平方最大均值差异（MMD）的跳跃修正Wasserstein梯度流。在次临界密度和有限矩假设下，我们证明了全局适定性。所引入的跳跃项将质量转移向一阶变分较小的取值，并与输运项耗散相同的MMD能量。在我们对目标测度和跳跃速率的假设下，每个具有有限MMD能量且跳跃耗散为零的Borel概率测度都等于目标测度。将这一刚性结果与紧性及二阶矩恒等式相结合，可以从仅具有有限二阶矩的初始数据得到向目标测度的收敛，而无需额外的一致时间密度或矩假设。我们还构造了一个全局无碰撞的粒子过程，其能量沿每条路径单调不增。定量平均场估计适用于……

    arXiv:2610.29007v1 Announce Type: cross  Abstract: We study a jump-modified Wasserstein gradient flow of the squared maximum mean discrepancy generated by the energy kernels $K(z)=-|z|^q$, $0<2$, in every dimension $d\ge2$ as well as the one-dimensional range $1\le q<2$. Under subcritical density and finite-moment assumptions, we prove global well-posedness. The added jumps transfer mass toward smaller values of the first variation and dissipate the same MMD energy as the transport term. Under our hypotheses on the target and jump rates, every Borel probability measure with finite MMD energy and zero jump dissipation is equal to the target. Combining this rigidity with compactness and a second-moment identity yields convergence to the target from finite-second-moment data, without additional uniform-in-time density or moment assumptions. We also construct a global collision-free particle process whose energy is nonincreasing along each path. A quantitative mean-field estimate applies t
    
[^165]: FlashNext-RL：面向 Qwen3.8-Flash-Next 长程智能体强化学习的系统优化研究

    FlashNext-RL: A System Optimization Study for Long-Horizon Agentic RL of Qwen3.8-Flash-Next

    [https://arxiv.org/abs/2610.28998](https://arxiv.org/abs/2610.28998)

    本文提出 FlashNext-RL 框架，通过优化主机卸载、梯度累积以及混合专家、稀疏注意力和门控残差流等专用内核，实现了在单个 8xB200 节点上对 Qwen3.8-Flash-Next 完整 48 层模型的 BF16 长程智能体强化学习训练，显著提升了长序列训练吞吐量。

    

    长程智能体强化学习（RL）既需要高效的 rollout 生成，也需要在长序列上进行训练。然而，Qwen3.8-Flash-Next 的显存需求和模型专用算子使得现有的开源实现难以完成这一任务。为解决这些局限，我们提出了 FlashNext-RL，一个基于 prime-rl 的模型专用 RL 训练优化框架，并使用 vLLM 进行软件工程（SWE）rollout 生成。我们的主要目标是在单个 8xB200 节点上对完整的 48 层模型进行 BF16 训练，其中前两层、n-gram 表、稀疏注意力索引器和多 token 预测头被冻结。我们优化了主机内存卸载、梯度累积，以及针对混合专家、稀疏注意力和门控残差流的内核。在全模型开发研究中，非填充训练吞吐量从初始 32K-token 配置下的约 395 tokens/s/GPU 提升至……

    arXiv:2610.28998v1 Announce Type: cross  Abstract: Long-horizon agentic reinforcement learning (RL) requires both efficient rollout generation and training on long sequences. However, the memory requirements and model-specific operators of Qwen3.8-Flash-Next make this difficult with existing open-source implementations. To resolve these limitations, we introduce FlashNext-RL, a model-specific RL training optimization framework based on prime-rl, with vLLM for software engineering (SWE) rollout generation. Our main focus is BF16 training of the full 48-layer model on a single 8xB200 node, with the first two layers, the n-gram table, the sparse-attention indexer and the multi-token prediction head frozen. We optimize host offload, gradient accumulation and the kernels for mixture-of-experts, sparse attention and the gated residual stream. In the full-model development study, non-padding training throughput increases from approximately 395 tokens/s/GPU in the initial 32K-token configurati
    
[^166]: 从频率视角重新思考时间序列分词

    Rethinking Time Series Tokenization from a Frequency Perspective

    [https://arxiv.org/abs/2610.28997](https://arxiv.org/abs/2610.28997)

    提出了一种几乎无失真的基于频率的时间序列分词器TimeFT，通过频域划分、频率移位和奈奎斯特采样获取token，无需参数且可作为现有分词器的即插即用替代品，有效避免传统分词带来的频谱失真问题。

    

    将时间序列划分为子序列的分词方法已成为现代时间序列建模的基础范式。然而，我们发现当前的分词策略固有地引入了频谱失真，迫使学习到的表示偏离真实的底层模式（如趋势、周期性和季节性），从而损害模型性能。通过从频率视角进行理论分析，我们推导出了这种失真发生的边界条件。为了克服这种失真，我们提出了TimeFT，一种几乎无失真的基于频率的时间序列数据分词器。TimeFT通过频域划分、频率移位和奈奎斯特采样来获取token。该方法无需参数，计算复杂度可忽略不计，并可作为现有分词器的即插即用替代品。我们在预测、分类等多种任务上的大量实验……

    arXiv:2610.28997v1 Announce Type: new  Abstract: Tokenization that partitions time series into subsequences has become a foundational paradigm in modern time series modeling. However, we find that current tokenization strategies inherently introduce spectral distortions, forcing the learned representations to diverge from the true underlying patterns such as trend, periodicity, and seasonality, thereby impairing model performance. Through theoretical analysis from a frequency perspective, we derive the boundary conditions under which such distortions occur. To overcome this distortion, we propose TimeFT, a nearly distortion-free frequency-based tokenizer for time series data. TimeFT obtains tokens through frequency-domain partitioning, frequency shifting, and Nyquist sampling. The method is parameter-free, incurs negligible computational complexity, and can serve as a drop-in replacement for existing tokenizers. Our extensive experiments on diverse tasks such as forecasting, classifica
    
[^167]: 多样化TSP解的最优采样器：刻画、极限与学习

    The Optimal Sampler for Diverse TSP Solutions: Characterization, Limits, and Learning

    [https://arxiv.org/abs/2610.28994](https://arxiv.org/abs/2610.28994)

    该论文针对多样化旅行商问题（D-TSP）提出将问题分解为采样与选择两阶段的混合框架，证明了选择阶段近似难度下界，并从理论和实证两方面为最优采样分布提供了保证（确保采样池包含最优解集或获得至少k个相互分散的解）。

    

    我们研究多样化旅行商问题（D-TSP），该问题旨在寻找一组 $k$ 个不同的TSP回路。每个被选中的回路长度不得超过最优回路长度的规定倍数，同时需最小化所选集合内的最大两两Jaccard相似度。这一问题的形式化对于需要同时保证解质量和容错性的应用非常重要，例如物流规划、机器人和战略巡逻。我们提出了一种新颖的混合框架，将D-TSP分解为采样阶段和选择阶段。其中选择阶段是一个最大-最小分散问题，对于该问题，任何优于两倍因子的近似都是NP难的；因此我们将研究重点放在采样阶段。我们的贡献兼具理论与实证两方面。在理论上，我们从两个视角为最优采样分布给出了两个端到端的保证：其一是在采样池中包含一个最优解集，其二是获得至少 $k$ 个相互分散的解。

    arXiv:2610.28994v1 Announce Type: cross  Abstract: We address the Diverse Traveling Salesman Problem (D-TSP) that seeks a set of $k$ distinct TSP tours. Each selected tour must not exceed a prescribed multiple of the optimal tour length, while the maximum pairwise Jaccard similarity within the selected set is minimized. This formulation is important for applications requiring both solution quality and fault tolerance, such as logistics planning, robotics, and strategic patrolling. We propose a novel hybrid framework that decomposes D-TSP into a sampling stage and a selection stage. The selection stage is a max-min dispersion problem, for which any approximation better than a factor of two is NP-hard; we therefore focus on the sampling stage. Our contributions are both theoretical and empirical. Theoretically, we give two end-to-end guarantees for the optimal sampling distribution, from two perspectives: including an optimal set in the sampled pool, or having at least $k$ mutually dista
    
[^168]: 基于 Doob 的生成式策略优化

    Doob-Based Generative Policy Optimization

    [https://arxiv.org/abs/2610.28990](https://arxiv.org/abs/2610.28990)

    提出 DoobPO 框架，利用 Doob 的 h 变换通过比率模型与条件漂移匹配，实现无需似然计算的扩散与流生成式策略强化学习微调，并统一兼容 PPO、GRPO 等策略改进规则。

    

    扩散模型和流模型为机器人与序列决策系统提供了表达能力强的策略类别。然而，它们的动作似然通常难以计算，这限制了直接使用诸如 PPO 和 GRPO 等被广泛采用的策略优化方法。我们提出了 Doob 策略优化（DoobPO），这是一个利用 Doob 的 h 变换对生成式策略进行无似然强化学习微调的框架。DoobPO 将策略改进与生成式策略学习相分离：它首先学习一个轻量级的比率模型来表示期望的策略更新，然后通过条件漂移匹配将由此得到的比率倾斜目标策略蒸馏进生成式策略中。这一构建方式统一适用于扩散策略和流策略，并支持广泛的策略改进规则族，包括 PPO 和 GRPO。我们证明了实际使用的条件漂移匹配目标在梯度上是等价的（摘要在此处截断）

    arXiv:2610.28990v1 Announce Type: new  Abstract: Diffusion and flow models provide expressive policy classes for robotic and sequential decision-making systems. However, their action likelihoods are generally intractable, limiting the direct use of widely adopted policy optimization methods such as PPO and GRPO. We propose Doob Policy Optimization (DoobPO), a framework that leverages Doob's $h$-transform for likelihood-free reinforcement learning fine-tuning of generative policies. DoobPO separates policy improvement from generative policy learning. It first learns a lightweight ratio model that represents the desired policy update, and then distills the resulting ratio-tilted target policy into the generative policy via conditional drift matching. This formulation applies uniformly to both diffusion and flow policies and supports a broad family of policy improvement rules, including PPO and GRPO. We prove that the practical conditional drift matching objective is gradient-equivalent t
    
[^169]: AgentPolicyForge：从可执行环境到策略基准与自适应防护器

    AgentPolicyForge: From Executable Environments to Policy Benchmarks and Adaptive Guards

    [https://arxiv.org/abs/2610.28989](https://arxiv.org/abs/2610.28989)

    该论文提出AgentPolicyForge框架，通过可执行环境实际运行工具调用并记录结果，构建了涵盖十个策略族的PolicyGym基准数据集，并利用其中真实发生的合规与违规案例迭代训练能够检查参数、资源和上下文（而非仅看工具身份）的自适应策略防护器。

    

    定制化的智能体部署需要针对工具、资源和执行条件定义策略。将这些策略仅放在任务智能体的提示词中并不能可靠地防止违规行为。训练一个防护器需要符合或违反这些策略的工具调用示例。我们提出AgentPolicyForge，这是一个利用可执行环境来构建策略基准并迭代训练策略防护器的框架。借助该框架，我们构建了PolicyGym数据集，涵盖十个策略族，使用698个场景生成训练数据，并另用254个独立的保留场景构建经人工审核的测试集。工具调用会被实际执行并记录其结果，以帮助判断其是否符合策略。每个违规案例都使用被允许的工具，因此需要对参数、资源和上下文进行检查，而不仅仅是检查工具的身份。该案例构建过程支持迭代式防护器训练：攻击者生成请求……（原文截断）

    arXiv:2610.28989v1 Announce Type: cross  Abstract: Customized agent deployments define policies over tools, resources, and execution conditions. Providing these policies in the task agent's prompt does not reliably prevent violations. Training a guard requires examples of tool calls that comply with or violate those policies. We introduce AgentPolicyForge, a framework that uses executable environments to construct policy benchmarks and iteratively train policy guards. With this framework, we construct PolicyGym, a dataset across ten policy families using 698 scenarios for training data generation and 254 separate held-out scenarios for human-reviewed test sets. Tool calls are executed and their outcomes recorded to help determine whether they comply with the policy. Every violating case uses a permitted tool, requiring checks of arguments, resources, and context rather than tool identity alone. The case construction process supports iterative guard training: an attacker generates reque
    
[^170]: 基于注意力机制的间接提示注入分类方法

    Attention-based Classification for Indirect Prompt Injection

    [https://arxiv.org/abs/2610.28988](https://arxiv.org/abs/2610.28988)

    本文首次在真实的智能体场景（AgentDojo基准）下评估了Attention Tracker检测方法，发现其声称的数据集无关性并不成立——长结构化工具输出会颠覆其注意力假设并使检测效果接近随机，进而提出了基于注意力分类的改进检测方案。

    

    间接提示注入（IPI）攻击是指对抗性指令被嵌入到语言模型所处理的外部数据中，这对已部署的LLM智能体的完整性构成重大威胁。Attention Tracker（Hung et al., 2025）是最近提出的一种白盒检测方法，它通过监控模型注意力在用户任务标记和数据标记之间的分布来识别提示注入攻击，并声称通过一个标量焦点分数阈值即可实现与数据集无关的检测。在本文中，我们首次在使用AgentDojo基准的真实智能体IPI场景下对Attention Tracker进行了评估，发现其数据集无关性的声明并不成立。我们发现，在智能体环境中，长篇结构化的工具输出颠覆了Attention Tracker方法所依赖的关于注意力的基本假设，导致该方法在AgentDojo上对大多数模型家族的检测效果接近随机猜测。我们提出Att…（原文摘要在此处截断）

    arXiv:2610.28988v1 Announce Type: new  Abstract: Indirect prompt injection (IPI) attacks, in which adversarial instructions are embedded in external data processed by a language model, pose a significant threat to the integrity of deployed LLM agents. Attention Tracker (Hung et al., 2025) is a recently proposed white-box detection method that identifies prompt injection attacks by monitoring how a model's attention is distributed between user task tokens and data tokens, claiming dataset-agnostic detection via a scalar focus score threshold. In this paper, we provide the first evaluation of Attention Tracker in a realistic agentic IPI setting using the AgentDojo benchmark, finding that the dataset agnosticism claim does not hold. We find that in agentic environments, long structured tool outputs subvert the fundamental assumptions Attention Tracker makes about attention that the method depends on and produces near-chance detection on AgentDojo across most model families. We propose Att
    
[^171]: 从公共信息到典型相关分析及其延伸：连接信息论、信号处理与机器学习

    From Common Information to Canonical Correlation Analysis and Beyond: Bridging Information Theory, Signal Processing, and Machine Learning

    [https://arxiv.org/abs/2610.28978](https://arxiv.org/abs/2610.28978)

    本教程首次将统计学中的典型相关分析、信息论中的两种公共信息概念以及工程领域的多视图子空间方法置于统一框架下，为无监督恢复多观测中公共分量的跨学科问题建立了桥梁，并厘清了不同理论之间的精确等价与类比关系。

    

    信号处理、机器学习和神经成像中的许多问题提供了对同一过程的若干联合观测，每个观测将所有观测共有的分量与自身私有的分量混合在一起，并需要在无监督条件下恢复公共分量。这个问题在多个研究群体中得到了研究，且在很大程度上是相互独立的：统计学家提出了由Hotelling于1936年引入的典型相关分析（CCA）；信息论学者提出了两种公共信息的概念，分别源于Gács和Körner以及Wyner；工程师们为传感器阵列和多用户通信开发了子空间方法，以设计多视图接收器。本教程在单一的阐述中呈现了各研究群体为这一问题开发的主要结果和工具，并解释了它们交汇处涌现的联系，区分了精确等价关系与特殊情况和类比。我们发展了信息论……（原文摘要截断于此）

    arXiv:2610.28978v1 Announce Type: cross  Abstract: Many problems in signal processing, machine learning, and neuroimaging provide several jointly observed measurements of one process, each mixing a component common to all of them with a component private to itself, and require the common component to be recovered without supervision. This problem has been studied in several communities, largely independently. Statisticians proposed canonical correlation analysis (CCA), introduced by Hotelling in 1936. Information theorists proposed two notions of common information, due to G\'acs and K\"orner and to Wyner. Engineers developed subspace methods for sensor arrays and multiuser communications to design multiview receivers. This tutorial presents the main results and tools each community developed for this problem in a single exposition, and explains the connections that emerge where they meet, distinguishing exact equivalences from special cases and analogies. We develop the information-th
    
[^172]: 上下文强化学习是后验采样器吗？一项实证研究

    Is In-Context Reinforcement Learning A Posterior Sampler? An Empirical Study

    [https://arxiv.org/abs/2610.28977](https://arxiv.org/abs/2610.28977)

    本文通过建立贝叶斯后验采样器必须满足的两个必要条件——充分统计量不变性和鞅性质，并据此构造统计假设检验，实证考察了上下文强化学习（特别是DPT及其变体）在实践中是否真正表现为贝叶斯后验采样器。

    

    上下文强化学习利用大型语言模型令人印象深刻的上下文学习能力，在推理时零样本泛化到新的、未见过的任务，而无需任何参数更新。在众多方法中，决策预训练Transformer（DPT）因其理论基础而尤为引人注目：在模型具有充分表达能力、预训练数据无限等理想化假设下，DPT可以执行贝叶斯后验采样。然而这一理论刻画在实践中是否仍然成立仍是悬而未决的问题。因此，我们提出这样一个问题：上下文强化学习，具体而言是DPT和我们提出的DPT变体，在实践中是否是一个贝叶斯后验采样器？为了研究这一问题，我们建立了贝叶斯后验采样器必须满足的两个必要条件：充分统计量不变性和鞅性质。随后我们构造了统计假设检验，以统计方法评估实践中训练的（摘要在此处截断）

    arXiv:2610.28977v1 Announce Type: cross  Abstract: In-context reinforcement learning (ICRL) exploits the impressive in-context learning ability of large language models to zero-shot generalize to new, unseen tasks at inference time without any parameter updates. Among many approaches, the Decision-Pretrained Transformer (DPT) is particularly appealing for its theoretical grounding: under idealized assumptions such as a sufficiently expressive model and infinite pretraining data, DPT can perform Bayesian posterior sampling. Whether this characterization survives in practice remains open. Therefore, we ask: \textit{is ICRL, specifically DPT and our proposed DPT variants, a Bayesian posterior sampler in practice?} To approach this question, we establish two necessary conditions that a Bayesian posterior sampler must satisfy: \textit{sufficient-statistic invariance} and the \textit{martingale property}. We then construct statistical hypothesis tests to statistically assess if practically-t
    
[^173]: SSMCGM-Stream：一种用于个性化糖尿病管理的流式状态空间模型

    SSMCGM-Stream: A Streaming State-Space Model for Personalized Diabetes Management

    [https://arxiv.org/abs/2610.28976](https://arxiv.org/abs/2610.28976)

    提出了流式状态空间模型SSMCGM-Stream，通过实时更新个性化隐藏状态实现高效、可解释的血糖预测，并结合SSMCGM-Agent提供设备端自然语言糖尿病决策支持，性能与神经基线相当而计算成本大幅降低。

    

    连续血糖监测为糖尿病管理提供了支持，但需要准确、可解释且足够高效以实现实时部署的预测模型。我们提出了SSMCGM-Stream，这是一种流式状态空间模型，通过每次接收到的血糖读数更新来维护个性化的隐藏状态，从而提供恒定时间的预测、具有生理结构的反事实推理，以及通过隐藏状态归因实现的可解释预测。在此基础上，SSMCGM-Agent通过一个受保护的语言模型接口，将预测结果、解释和反事实分析转化为有据可依的建议，提供风险感知的自然语言决策支持。在T1DEXI队列上的评估表明，SSMCGM-Stream的预测性能与强大的神经基线模型相当，同时大幅降低了计算成本，使持续的设备端糖尿病决策支持成为可能。

    arXiv:2610.28976v1 Announce Type: new  Abstract: Continuous glucose monitoring supports diabetes management but requires forecasting models that are accurate, interpretable, and efficient enough for real-time deployment. We present SSMCGM-Stream, a streaming state-space model that maintains a personalized hidden state updated with each incoming glucose reading to provide constant-time forecasting, physiologically structured counterfactual reasoning, and interpretable forecasts through hidden-state attribution. Built on this foundation, SSMCGM-Agent delivers risk-aware, natural-language decision support by translating forecasts, explanations, and counterfactuals into grounded recommendations through a guarded language-model interface. Evaluated on the T1DEXI cohort, SSMCGM-Stream achieves forecasting performance comparable to strong neural baselines while substantially reducing computational cost, enabling continuous on-device diabetes decision support.
    
[^174]: 正交化如何适应神经网络的Hessian结构？基于初始化时梯度自外积的分析

    How Does Orthogonalization Adapt to the Neural Network Hessian Structure? A Gradient Self Outer-Product Analysis at Initialization

    [https://arxiv.org/abs/2610.28975](https://arxiv.org/abs/2610.28975)

    本文通过在初始化阶段对梯度自外积（$\mathbf{V}\mathbf{V}^\top$）的对角占优性进行分析，揭示了Muon正交化预条件器之所以有效，是因为它能匹配神经网络逐层Hessian的行块对角占优结构。

    

    arXiv:2610.28975v1 公告类型：新论文 摘要：Muon在每一步更新前对权重矩阵的动量进行正交化，在神经网络上，这个简单的预条件器在广泛任务中的表现优于AdamW。现有分析要么在一般优化框架内解释Muon，要么只在少数特定模型上进行分析，且许多工作将正交化视为一种预条件器。由于在损失的二次近似下，Hessian是最优的预条件器，这些工作留下了一个基本问题：是什么使得正交化适合于神经网络的Hessian？先前的研究表明，神经网络的逐层Hessian是行块对角占优的，而Muon的隐式预条件器，即Kronecker积 $(\mathbf{V}\mathbf{V}^\top)^{1/2}\otimes\mathbf{I}$（其中 $\mathbf{V}$ 为动量），恰好在 $\mathbf{V}\mathbf{V}^\top$ 对角占优时具有相同的行块结构。$\mathbf{V}\mathbf{V}^\top$ 的对角占优性已在语言模型中被观察到，但……

    arXiv:2610.28975v1 Announce Type: new  Abstract: Muon orthogonalizes a weight matrix's momentum before each step, and on neural networks this simple preconditioner beats AdamW in a wide range of tasks. Existing analyses explain Muon within general optimization frameworks or on a few specific models, and many of them view orthogonalization as a preconditioner. Since the Hessian is the optimal preconditioner under a quadratic approximation of the loss, these works leave a basic question open: what makes orthogonalization suited to the Hessian of a neural network? Prior work shows that the layer-wise Hessian of a neural network is row-block diagonally dominant, and Muon's implicit preconditioner, the Kronecker product $(\mathbf{V}\mathbf{V}^\top)^{1/2}\otimes\mathbf{I}$ with $\mathbf{V}$ the momentum, has the same row-block structure exactly when $\mathbf{V}\mathbf{V}^\top$ is diagonally dominant. Diagonal dominance of $\mathbf{V}\mathbf{V}^\top$ has been observed in language models, but 
    
[^175]: 从天气到强化风化：面向运营尺度二氧化碳去除的天气基础模型

    From Weather to Enhanced Weathering: Weather Foundation Models for Carbon Dioxide Removal at Operational Scale

    [https://arxiv.org/abs/2610.28973](https://arxiv.org/abs/2610.28973)

    该研究首次整体评估了天气基础模型在增强岩石风化碳去除量化任务上的迁移能力，其基于Aurora 1.5构建的TerraNova-ERW模型能够在未见过的田块内正确排序测量值，但指出实现零样本预测新田块去除量仍需更大规模的ERW运营数据。

    

    增强岩石风化（ERW）是一种每年有潜力移除数十亿吨二氧化碳的碳去除解决方案，但目前量化其去除量需要密集的土壤采样和实验室分析。我们在来自巴西和美国的264个田块的5,192条年代序列上，采用留一田块协议，评估了五个天气基础模型（FM）作为冻结编码器并连接轻量级解码器的表现。据我们所知，这是首次针对ERW任务对天气基础模型进行的整体评估，也是在运营级ERW数据上最大规模的评估。我们发现，各类天气基础模型的表征可以迁移到ERW任务：我们基于Aurora 1.5构建的最佳模型TerraNova-ERW，能够在从未见过的田块内对测量值进行正向排序。然而，对新田块平均去除量的零样本预测尚不可行，我们将其归因于数据规模：同一架构在约10^5条自然风化序列上表现出色，这锚定了我们预期ERW数据必须达到的规模。

    arXiv:2610.28973v1 Announce Type: new  Abstract: Enhanced Rock Weathering (ERW) is a carbon removal solution with the potential to remove gigatons of CO$_2$ per year, but quantifying that removal today requires dense soil sampling and laboratory analysis. We evaluate five weather Foundation Models (FMs) as frozen encoders feeding lightweight decoders, under a leave-one-field-out protocol on 5,192 chronosequences from 264 fields across Brazil and the US. To our knowledge, this is the first holistic evaluation of weather FMs for ERW and the largest evaluation on operational ERW data. We find that across weather FMs, representations transfer to ERW: our best model, TerraNova-ERW, built on Aurora 1.5, positively orders measurements within fields it has never seen. Zero-shot prediction of a new field's mean removal is not yet feasible, and we attribute this to data scale: the same architecture excels on $\sim$$10^5$ natural-weathering sequences, anchoring the scale we expect ERW data must r
    
[^176]: 面向临床与可穿戴健康数据的单元感知差分隐私量子机器学习

    Unit-Aware Differentially Private Quantum Machine Learning for Clinical and Wearable Health Data

    [https://arxiv.org/abs/2610.28968](https://arxiv.org/abs/2610.28968)

    该论文提出一种“单元感知”的差分隐私量子机器学习方法，将每条临床记录或每位可穿戴设备用户视为隐私单元，并在输入维度、隐私预算与模型容量匹配的条件下与经典模型对比，以检验可训练参数更少的量子模型能否在保护隐私的同时为临床与健康数据保留更多预测效用。

    

    临床登记数据和可穿戴传感器可支持早期健康风险评估，但基于这些记录训练的模型在发布时可能泄露敏感信息。差分隐私（DP）通过注入噪声来限制这种泄露，而噪声的总幅度会随可训练参数数量的增加而增长，这一代价对规模较小的健康队列尤为严重。这促使人们关注参数高效的模型，包括量子机器学习（QML），其变分电路能够以少量可训练参数表达复杂的决策边界。由此引出一个问题：在输入维度与隐私预算匹配的情况下，量子模型是否比经典模型保留更多效用？我们在三个临床队列和一个可穿戴设备队列上研究这一问题，将每条临床记录或每位参与者视为一个隐私单元，并在五种隐私预算和五个随机种子下，对容量匹配的量子模型、多层感知机和逻辑回归模型进行了比较。（注：原文摘要在此处截断）

    arXiv:2610.28968v1 Announce Type: new  Abstract: Clinical registries and wearable sensors support early health-risk assessment, yet models trained on these records may leak sensitive information when released. Differential privacy (DP) bounds this leakage by injecting noise whose total magnitude grows with the number of trainable parameters, a cost especially severe for small health cohorts. This motivates parameter efficient models including quantum machine learning (QML), whose variational circuits can represent complex decision boundaries with few trainable parameters. It raises a question: at matched input dimensionality and privacy budget, does a quantum model retain more utility than a classical one? We investigate this question on three clinical cohorts and one wearable cohort, treating each clinical record or participant as a privacy unit. We compare capacity-matched quantum, multilayer perceptron, and logistic regression models across five privacy budgets and five seeds. The q
    
[^177]: 量子层能否为视觉-语言模型带来性能与安全优势？

    Can Quantum Layers offer Performance and Security Advantages in Vision-Language Models?

    [https://arxiv.org/abs/2610.28964](https://arxiv.org/abs/2610.28964)

    本研究以LLaVA-OneVision为骨干，通过在多个位置插入或替换4/8量子比特的模拟量子线路，系统比较了量子-经典混合视觉-语言模型与纯经典模型在性能、安全性及计算成本上的差异，以探究量子层在大规模多模态模型中的实际价值。

    

    随着量子计算从含噪声中等规模量子（NISQ）时代迈向早期容错量子计算（EFTQC），量子增强AI正从一个概念性想法转变为实验性研究。量子组件有望扩展现有AI系统的能力，然而此类组件将如何在大规模多模态模型中发挥作用仍不清楚。在本研究中，我们尝试利用经典模拟的量子层，针对视觉-语言模型（VLMs）回答这一问题。以LLaVA-OneVision为骨干网络，我们在四个插入位置和四个替换位置上评估了4量子比特和8量子比特的基本环形线路与强纠缠线路。插入方式是将量子层添加到经典模型中，而替换方式则是用量子层取代经典组件。在相同的实验设置下，我们比较了经典模型与量子-经典混合模型之间的性能与计算成本。在Visual Spatial Reasoning数据集上，插入……（摘要原文在此处截断）

    arXiv:2610.28964v1 Announce Type: cross  Abstract: As quantum computing moves from the noisy intermediate-scale quantum (NISQ) era toward early fault-tolerant quantum computing (EFTQC), quantum-enhanced AI is shifting from a conceptual idea to experimental research. Quantum components are expected to extend the capabilities of existing AI systems, yet it remains unclear how such components will help within large multimodal models. In this study, we try to answer that question for vision-language models (VLMs) using classically simulated quantum layers. With LLaVA-OneVision as the backbone, we evaluate 4- and 8-qubit basic-ring and strongly entangling circuits at four insertion and four replacement locations. Insertion adds quantum layers to the classical model, while replacement substitutes them for classical components. Using the same setup, we compare performance and computational cost between classical and quantum-classical hybrid models. On the Visual Spatial Reasoning dataset, ins
    
[^178]: 作为通用深度神经网络的Transformer：精确编译、移除激活-激活乘积的代价与多智能体注意力聚类

    Transformers as Generic Deep Neural Networks: Exact Compilation, the Cost of Removing Activation-Activation Products, and Multi-Agent Attention Clustering

    [https://arxiv.org/abs/2610.28958](https://arxiv.org/abs/2610.28958)

    本文证明每个Transformer块都能被精确编译成仅由仿射映射、路由和少数基本非线性运算构成的通用深度神经网络，并量化了移除激活-激活乘积运算所需付出的至少N²次额外标量非线性求值的代价，表明仅保留一种逐元素形式的该运算即可消除此开销。

    

    Transformer通常被作为一种特殊的模型类别来部署。神经处理单元（NPU）为注意力和归一化添加了专用算子，量化也依赖于针对Transformer的特定算法。我们证明这种计算本身并不需要如此特殊化。每个包含注意力、LayerNorm和GELU MLP的Transformer块都可以被精确编译成一个通用深度神经网络，该网络仅由固定的仿射映射、路由，以及平方、sigmoid和逆平方根运算，再加上GELU或其sigmoid近似构成。在缺少激活-激活乘积运算的情况下，逐变换的编译对于稠密得分而言，每个注意力头至少需要额外增加N²次标量非线性求值。而保留这一种运算类型即可消除这一代价；其逐元素形式已存在于所有被调研的NPU上，配合sigmoid和逆平方根即已足够。对于具有可交换尾部得分的top-k注意力，我们证明了精确尾部质量或条件均值尾部质量能够给出无偏且均方误差最小的估计。

    arXiv:2610.28958v1 Announce Type: new  Abstract: Transformers are deployed as a special model class. Neural processing units (NPU) add dedicated operators for attention and normalization, and quantization relies on Transformer-specific algorithms. We show the computation does not force this. Every Transformer block, with attention, LayerNorm and a GELU MLP, compiles exactly into a generic deep neural network built from fixed affine maps, routing and the square, sigmoid and inverse square root, plus GELU or its sigmoid approximation. Without activation-activation products, every transform-wise compilation adds at least $N^2$ scalar nonlinear evaluations per head for dense scores. Keeping that one operation type removes this cost; its elementwise form, present on every surveyed NPU, suffices with the sigmoid and inverse square root. For top-$k$ attention with exchangeable tail scores, we prove that exact or conditional-mean tail mass gives an unbiased, minimum-mean-square-error estimate 
    
[^179]: HyperDrone：以每秒数百万帧实现仿真到现实的迁移

    HyperDrone: Simulation-to-Reality Transfer at Millions of Frames per Second

    [https://arxiv.org/abs/2610.28950](https://arxiv.org/abs/2610.28950)

    HyperDrone是一个超高速四旋翼无人机模拟器，通过结合优化的多旋翼动力学与新型光线追踪引擎HyperLux，比现有最先进模拟器快20倍以上，并能在消费级GPU上以每秒数百万帧的速度渲染高保真逼真图像，从而实现高效的仿真到现实迁移。

    

    我们提出了HyperDrone，一个专注于高性能批量渲染的超快速四旋翼无人机模拟器，其速度比现有最先进的批量四旋翼模拟器快20倍以上。HyperDrone将高度优化的多旋翼动力学与HyperLux相结合，HyperLux是一种新型光线追踪引擎，支持Nvidia OptiX、Vulkan、Apple Metal和WebGPU后端，充分利用现代光线追踪硬件，并能将两个组件完全在GPU上运行。HyperLux能够以可配置的保真度级别渲染RGB、深度、语义/实例分割、表面法线和光流等观测信息，包括反射、逼真光照、运动模糊、抗锯齿以及基于物理的渲染（PBR）。即使在最高保真度设置下，HyperLux在消费级GPU上渲染逼真3D场景时仍能达到每秒数百万帧（FPS）的速度。我们通过训练基于视觉的偏航控制任务来展示HyperDrone的优势。

    arXiv:2610.28950v1 Announce Type: cross  Abstract: We present HyperDrone, an extremely fast quadrotor simulator focused on high-performance batch rendering that is more than 20x faster than existing state-of-the-art batch quadrotor simulators. HyperDrone combines highly optimized multirotor dynamics with HyperLux, a new ray-tracing engine that supports Nvidia OptiX, Vulkan, Apple Metal and WebGPU backends that takes full advantage of modern ray-tracing hardware, capable of running both components entirely on the GPU. HyperLux renders RGB, depth, semantic/instance segmentation, surface normals and optical flow observations at configurable levels of fidelity, including reflections, realistic lighting, motion blur, anti-aliasing, and Physically Based Rendering (PBR). Even in the highest-fidelity settings, HyperLux achieves millions of Frames Per Second (FPS) on consumer-grade GPUs while rendering realistic 3D scenes. We demonstrate the merits of HyperDrone by training a vision-based yaw-c
    
[^180]: 基于超维计算的在线开放世界晶圆缺陷发现

    Online Open-World Wafer Defect Discovery using Hyperdimensional Computing

    [https://arxiv.org/abs/2610.28949](https://arxiv.org/abs/2610.28949)

    提出了基于超维计算的在线开放世界晶圆缺陷发现框架WaferHDC，无需标签或重训即可动态发现并分组新型晶圆缺陷模式，在WM-81数据集上调整兰德指数超越最强在线基线13.3个百分点。

    

    传统的晶圆缺陷分析依赖于离线的、封闭世界分类器，无法检测到在连续半导体制造过程中出现的新型、未曾见过的缺陷特征。为解决这一问题，我们提出了WaferHDC，一个基于超维计算（HDC）的在线开放世界晶圆缺陷发现框架。WaferHDC以存储为中心的架构能够实现快速异常检测，并将重复出现的故障分组为紧凑的超向量表示。这使得存储空间能够随着新的缺陷模式动态扩展，而无需重新训练静态分类器或存档完整的原始晶圆历史数据。WaferHDC首先对正常空间分布进行建模，然后在无标签或预定义类别数量的情况下，将重复出现的故障动态分组到持久缺陷存储中。在行业标准WM-81数据集上，该方法在调整兰德指数（adjusted Rand index）方面超越最强的在线基线13.3个百分点，在归一化互信息（normalized mutual information）方面超越10.5个百分点。

    arXiv:2610.28949v1 Announce Type: new  Abstract: Conventional wafer defect analysis relies on offline, closed-world classifiers that fail to detect novel, unseen defect signatures emerging during continuous semiconductor manufacturing. To address this, we propose WaferHDC, an online open-world wafer defect discovery framework based on hyperdimensional computing (HDC). The memory-centric architecture of WaferHDC enables rapid anomaly detection and grouping of recurring failures into compact hypervector representations. This allows the memory space to dynamically scale with new defect modes without retraining a static classifier or archiving entire raw wafer history. WaferHDC first models normal spatial distributions, then dynamically groups recurring failures into persistent defect memory without labels or predefined class counts. It outperforms the strongest online baseline by 13.3 p.p. in adjusted Rand index and 10.5 p.p. in normalized mutual information on the industry-standard WM-81
    
[^181]: 2:4稀疏还值得吗？大语言模型推理中半结构化稀疏与量化的对比

    Is 2:4 Sparsity Still Worth It? Semi-Structured Sparsity versus Quantization in LLM Inference

    [https://arxiv.org/abs/2610.28947](https://arxiv.org/abs/2610.28947)

    本研究系统比较了大语言模型推理中2:4半结构化稀疏与量化的取舍，发现只要硬件还有更低精度可用，降精度量化总是优于叠加2:4稀疏（后者既更不准确也更慢），只有当量化已达到硬件支持的最低精度时，2:4稀疏才能为线性层带来1.3—1.7倍的加速。

    

    自Ampere架构以来，NVIDIA张量核心就一直支持2:4半结构化稀疏，且该稀疏模式可与量化组合使用。当两者结合时，实际问题是下一步该怎么做：在给定精度下，是增加2:4稀疏，还是降到下一个更低的精度？我们在等比特（将稀疏元数据计算在内）和同等标称张量核心层级的条件下，对Transformer层的全部六个GEMM操作数进行了比较，在三代GPU上测量了计算内核，并包含了在线剪枝与量化。我们的结果表明，只要硬件支持更低的精度，增加2:4就是更差的选择——不仅精度更低，速度也更慢，这是因为2:4在硬件强制轴上从每组四个元素中丢弃的能量不会随比特数的减少而缩减，而量化误差则会。一旦量化达到硬件支持的最低精度，2:4稀疏仍能将线性层加速1.3—1.7倍，在H100上，经过重新训练的……（原文摘要在此处截断）

    arXiv:2610.28947v1 Announce Type: new  Abstract: NVIDIA tensor cores have supported 2:4 semi-structured sparsity since Ampere, and the pattern composes with quantization. When the two are combined, the practical question is which step to take next: from a given precision, add 2:4 sparsity or drop to the next lower precision? We compare the two at equal bits, with sparsity metadata counted, and at equal nominal tensor-core tiers, over all six GEMM operands of a transformer layer, with kernels measured on three GPU generations and online pruning and quantization included. Our results show that as long as a lower precision is available, adding 2:4 is the worse step, both less accurate and slower, because the energy that 2:4 discards from each group of four along the hardware-forced axis does not shrink with bits, while quantization error does. Once quantization reaches the lowest precision the hardware supports, 2:4 still speeds up linear layers by 1.3--1.7\x{}, and on an H100 a retrained
    
[^182]: 基于模块化扩散引导的多目标蛋白质优化

    Multi-Objective Protein Optimization via Modular Diffusion Steering

    [https://arxiv.org/abs/2610.28943](https://arxiv.org/abs/2610.28943)

    本文提出MODS框架，通过推理时扩散引导结合采样的切比雪夫标量化权重和代理集成预测，实现模块化的多目标蛋白质优化，可灵活搭配不同扩散模型以发现多种性质权衡下的高性能蛋白质序列。

    

    蛋白质工程通常需要同时改进多个性质，这促使人们寻找代表有用适应性权衡的高性能序列。发现这些序列具有挑战性，因为设计空间在组合意义上极其庞大，且实验活动通常只允许进行少数几轮的批量测量。我们提出了MODS，一个通过推理时扩散引导进行多目标蛋白质优化的模块化框架。对于每个候选提案，MODS通过采样增强的切比雪夫标量化权重和一个适应性代理集成成员来定义标量引导奖励。改变权重可以将生成过程引导向不同的目标权衡，而改变集成成员则可以使用于引导的预测多样化。由于MODS通过标量奖励来表达多目标获取策略，因此它可以与不同的扩散模型及兼容的引导程序配对使用。在每一……（摘要不完整，原文在此处截断）

    arXiv:2610.28943v1 Announce Type: new  Abstract: Protein engineering often requires improving several properties simultaneously, motivating the search for high-performing sequences that represent useful fitness tradeoffs. Discovering these sequences is challenging because the design space is combinatorially large and experimental campaigns often permit only a few rounds of batched measurements. We introduce \textit{MODS}, a modular framework for multi-objective protein optimization via inference-time diffusion steering. For each proposal, MODS samples augmented Chebyshev scalarization weights and a fitness-surrogate ensemble member to define a scalar steering reward. Varying the weights directs generation toward different objective tradeoffs, while varying ensemble members diversifies the predictions used for guidance. Because MODS expresses multi-objective acquisition through scalar rewards, it can be paired with different diffusion models and compatible steering procedures. After eac
    
[^183]: 审计推理时模型偏移下语言模型的保形预测

    Auditing Conformal Prediction for Language Models Under Inference-Time Model Shift

    [https://arxiv.org/abs/2610.28937](https://arxiv.org/abs/2610.28937)

    该论文首次提出并形式化了“推理时模型偏移”现象——即部署与校准阶段推理配置（如量化模式和推理温度）的差异会引起保形分数分布变化并破坏分割保形可交换性，并通过8个语言模型、4个基准和12种推理配置的大规模实验系统评估了其对语言模型保形预测可靠性的影响。

    

    语言模型的保形预测（CP）通常在匹配协议下进行评估，即校准和部署使用相同的训练后推理配置，例如量化模式和推理温度。而在实际部署中，为了满足内存或吞吐量限制或服务策略的要求，这些配置选择往往与校准时不同。我们认为这暴露了当前语言模型CP评估协议的一个局限性。我们将这一现象形式化为“推理时模型偏移”：即由推理配置而非输入-标签分布引起的保形分数分布变化，并从理论上刻画了即使任务分布和已训练模型保持不变，这种偏移也会破坏分割保形可交换性。随后，我们在8个语言模型、4个有限标签基准和12种推理配置上评估了其影响。我们发现匹配评估...

    arXiv:2610.28937v1 Announce Type: new  Abstract: Conformal prediction (CP) for language models is typically evaluated under a matched protocol, in which calibration and deployment use the same post-training inference configuration, such as quantization mode and inference temperature. In deployment, these choices often differ from those used at calibration to meet memory or throughput constraints or serving policies. We argue that this exposes a limitation in current CP evaluation protocols for language models. We formalize the underlying phenomenon as inference-time model shift: a change in the conformal score distribution induced by inference configuration rather than by the input--label distribution, and theoretically characterize how it breaks split-conformal exchangeability even when the task distribution and trained model are unchanged. We then evaluate its consequences across 8 language models, 4 finite-label benchmarks, and 12 inference configurations. We find that matched evalu
    
[^184]: 数据差异化框架（DDF）：在智能体评估流水线（SDSAE）中利用信息论与博弈论度量量化数据的边际价值

    The Data Differentiator Framework (DDF): Quantifying Marginal Data Value with Information-Theoretic and Game-Theoretic Measures in an Agentic Evaluation Pipeline (SDSAE)

    [https://arxiv.org/abs/2610.28933](https://arxiv.org/abs/2610.28933)

    提出数据差异化框架（DDF），通过互信息冗余惩罚、成本归一化的边际效用评分和治理输入，量化候选数据资产相对企业已有数据的增量价值，并由智能体流水线SDSAE自动高效执行评估。

    

    在数据丰富的企业中，关键问题不是某个候选数据集在孤立状态下是否优秀，而是它能否在组织已有数据之外提供额外信息，以及这种增量是否物有所值。我们提出了数据差异化框架（DDF），这是一种针对候选数据资产的复合评分方法，它结合了：（i）基于互信息的、针对现有数据组合的冗余惩罚；（ii）通过消融研究得出的、经成本归一化并受显著性门控的边际效用评分；以及（iii）由业务方设定的明确治理输入（可信度、身份解析质量、战略权重）。该框架由一个分阶段的流水线——战略数据采购智能体生态系统（SDSAE）——执行，对于未通过治理或冗余门控的资产，该流水线会跳过昂贵的模型重训练步骤。从概念上讲，DDF 将 mRMR 特征选择中的“相关性减去冗余性”原则从特征层面……（原文摘要在此处截断）

    arXiv:2610.28933v1 Announce Type: new  Abstract: In data-rich enterprises the hard question is not whether a candidate dataset is good in isolation, but whether it adds information beyond what the organization already holds, and whether that increment is worth its price. We present the Data Differentiator Framework (DDF), a composite scoring method for candidate data assets that combines (i) a mutual-information-based redundancy penalty against the existing portfolio, (ii) a cost-normalized, significance-gated marginal utility score derived from an ablation study, and (iii) explicit governance inputs (trust, identity resolution quality, strategic weights) set by the business. The framework is executed by a staged pipeline, the Strategic Data Sourcing Agent Ecosystem (SDSAE), that short-circuits expensive model-retraining steps for assets failing governance or redundancy gates. Conceptually, the DDF extends the relevance-minus-redundancy principle of mRMR feature selection from the feat
    
[^185]: 一种策略，多种风格：在哪一层聚合偏好以实现可引导的人形机器人控制？

    One Policy, Many Styles: Where to Aggregate Preferences for Steerable Humanoid Control?

    [https://arxiv.org/abs/2610.28925](https://arxiv.org/abs/2610.28925)

    该论文研究了偏好条件强化学习中偏好聚合层级（奖励、优势、损失、PCGrad）对人形机器人可引导控制的影响，发现没有单一聚合层级在所有环境下最优：早期聚合扩大行为响应范围但增加非预期改变，晚期聚合提升控制精度但有所取舍。

    

    现代机器人智能体越来越被期望支持可引导控制，即单一策略能够在运行时根据不断变化的用户偏好调整其行为风格。偏好条件强化学习提供了这样的接口，但仅仅依靠条件化并不能保证所请求的行为改变足够强烈、不会泄露到无关行为中、也不会损害任务性能。我们的评估将偏好奖励响应性（PRR）与覆盖率和效用指标相结合，使用这些指标在两个人形机器人控制框架中比较了奖励层级、优势函数层级、损失层级以及基于PCGrad的聚合方式。结果显示，没有任何一个聚合层级能在所有环境和所有评估标准上都表现最佳。实验表明，较早聚合目标能产生更广泛的行为响应范围，但对其他目标会造成更大的非预期改变；而较晚聚合目标则能提高控制精度，但需要付出一定代价。

    arXiv:2610.28925v1 Announce Type: cross  Abstract: Modern robotic agents are increasingly expected to support steerable control, where a single policy can adjust its behavioral style at runtime under changing user preferences. Preference-conditioned reinforcement learning provides this interface, but conditioning alone does not ensure that a requested behavior change is strong, does not leak into unrelated behaviors, and does not compromise task performance. Our evaluation combines Preference Reward Responsiveness (PRR) with coverage and utility metrics. We use these measures to compare reward-level, advantage-level, loss-level, and PCGrad-based aggregation across two humanoid-control frameworks. No aggregation level performs best across all environments and criteria. Experiments show that aggregating objectives early produces a wider range of behavioral responses but greater unintended changes in other objectives, while aggregating them later improves control precision at some cost to
    
[^186]: 面向稳定与可认证语言生成的查询反馈与教学

    Query Feedback and Teaching for Stable and Certifiable Language Generation

    [https://arxiv.org/abs/2610.28918](https://arxiv.org/abs/2610.28918)

    本文在极限语言生成中引入了更强的“可认证性”概念，并借鉴教学集理论建立了正确性、稳定性与可认证性三级教学维度层级，同时进一步研究了成员查询模型下的稳定性与可认证性。

    

    极限意义下的语言生成研究的是：一个观察未知目标语言枚举序列的学习者，是否最终能够只生成有效的、之前未见过的元素。我们研究了由 Kalavasis、Mehrotra 和 Velegkas（2025）提出的稳定性概念，它要求生成器的无限输出集合最终保持不变，并引入了更强的可认证性概念，即要求生成器能够根据其观测判断出当前输出是有效的。借鉴学习理论中的教学集思想，我们在三种越来越严格的成功标准下——正确性、稳定性和可认证性——为生成任务引入了教学维度的层级结构。这些维度衡量了使相应成功条件得到保证所需的所选示例的最少数量。我们还在 Charikar 和 Pabbaraju（2025）的成员查询模型中研究了稳定性与可认证性。我们对稳定（原文在此处截断）……

    arXiv:2610.28918v1 Announce Type: new  Abstract: Language generation in the limit studies whether a learner observing an enumeration of an unknown target language can eventually generate only valid, previously unseen elements. We study stability, introduced by Kalavasis, Mehrotra, and Velegkas (2025), which requires the generator's infinite output set to eventually remain unchanged, and introduce the stronger notion of certifiability, which requires the generator to determine from its observations that its current output is valid. Drawing on teaching sets in learning theory, we introduce a hierarchy of teaching dimensions for generation under three increasingly strict success criteria: correctness, stability, and certifiability. These dimensions measure the minimum number of selected examples whose appearance guarantees the corresponding success condition. We also study stability and certifiability in the membership-query model of Charikar and Pabbaraju (2025). We characterize stable a
    
[^187]: 基于离散时间离策略数据、具有端到端保证的无模型连续时间强化学习算法

    A Model-Free Continuous-Time RL Algorithm from Discrete-Time Off-Policy Data with End-to-End Guarantees

    [https://arxiv.org/abs/2610.28915](https://arxiv.org/abs/2610.28915)

    该论文提出了一种无模型的PhiBE策略迭代（PhiBE-PI）框架，仅使用单个状态-动作评论家、直接从离散时间离策略数据中学习连续时间强化学习策略，同时保留连续时间的动作排序尺度并具有端到端的理论保证。

    

    本文研究连续时间强化学习问题，其中状态遵循一个未知的随机微分方程，但仅有固定的离策略离散时间转移数据批次可用。随着采样间隔的缩小，标准的离散时间Q学习会出现问题，因为动作值之间的差异会消失。连续时间的替代方法，如深度优势更新和鞅q学习，虽然保留了动作信号，但需要联合学习一个价值函数和一个依赖动作的评论家网络。基于PhiBE-Q公式化（arXiv:2606.21925），我们开发了一个无模型的PhiBE策略迭代（PhiBE-PI）框架，该框架仅使用单个状态-动作评论家，同时保留了连续时间的动作排序尺度。在每次迭代中，该框架直接从采样的增量中学习评论家，并利用它来改进策略。整个过程中重复使用同一数据批次，无需显式估计漂移或……

    arXiv:2610.28915v1 Announce Type: new  Abstract: In this paper, we study continuous-time reinforcement learning, where the state follows an unknown stochastic differential equation but only a fixed off-policy batch of discrete-time transitions is available. Standard discrete-time Q-learning becomes problematic as the sampling interval shrinks, because differences between action values vanish. Continuous-time alternatives, such as Deep Advantage Updating and martingale q-learning, preserve the action signal but jointly learn a value function and an action-dependent critic. Building on the PhiBE-Q formulation (arXiv:2606.21925), we develop a model-free PhiBE policy iteration (PhiBE-PI) framework that uses a single state-action critic while preserving the continuous-time action-ranking scale. At each iteration, the framework learns the critic directly from sampled increments and uses it to improve the policy. The same batch is reused throughout, without explicitly estimating the drift or 
    
[^188]: 回溯直接效应、间接效应与总效应

    Backtracking Direct, Indirect and Total Effects

    [https://arxiv.org/abs/2610.28910](https://arxiv.org/abs/2610.28910)

    本文提出基于回溯反事实的因果效应新定义，无需干预处理变量或中介变量，而是通过回溯条件分布构造满足处理-中介约束的替代外生状态，从而得到回溯直接效应、间接效应和总效应。

    

    自然直接效应和间接效应是通过跨世界干预反事实来定义的。我们提出了一种基于回溯反事实的替代性构造方法。我们不再对处理变量或中介变量进行干预，而是利用回溯条件分布来构造一个满足所需处理-中介约束的替代性外生状态。由此得到了回溯直接效应和回溯间接效应，以及回溯总效应。同样地，我们也对人群干预效应进行了相应的改造。

    arXiv:2610.28910v1 Announce Type: cross  Abstract: Natural direct and indirect effects are defined through cross-world interventional counterfactuals. We propose an alternative construction based on backtracking counterfactuals. Rather than intervening on the treatment or mediator, we use a backtracking conditional distribution to construct an alternative exogenous state satisfying a desired treatment-mediator constraint. This yields the backtracking direct and indirect effects, and also the backtracking total effect. Likewise, we adapt the population intervention effects.
    
[^189]: JaxMARL-BC：用于宏观经济建模的多智能体强化学习，只需数秒

    JaxMARL-BC: Multi-Agent Reinforcement Learning for Macroeconomic Modelling, in Seconds

    [https://arxiv.org/abs/2610.28907](https://arxiv.org/abs/2610.28907)

    本文提出基于JAX加速的模拟器JaxMARL-BC，将MARL-BC经济体的训练速度提升75至724倍、缩短至数秒内，并可扩展至20万个独立学习的家庭，解决了MARL在宏观经济学中应用的主要瓶颈。

    

    多智能体强化学习（MARL）为在模拟经济中引入大量异质决策者提供了一种有原则的方法。尽管该方法在金融领域率先得到应用，但其在宏观经济学中的推广一直较为缓慢，原因主要有两点：一是缺乏一个共享的参照框架，将MARL系统与学界已经信赖的动态随机一般均衡（DSGE）模型联系起来；二是训练时间过长。第一个局限最近已由多智能体强化学习经济周期模型（MARL-BC）解决，该模型将深度强化学习家庭置于实际经济周期经济体中，并在特定极限条件下可还原为经典的DSGE模型。本文则解决了第二个局限。我们提出了JaxMARL-BC，一个基于JAX加速的模拟器，可以在数秒内完成MARL-BC经济体的训练（比参考实现快75倍至724倍），并可扩展至200,000个独立学习的家庭。我们验证了其正确性……

    arXiv:2610.28907v1 Announce Type: cross  Abstract: Multi-agent reinforcement learning (MARL) offers a principled way of populating simulated economies with many heterogeneous decision makers. Its adoption, pioneered in finance, has nonetheless been slow in macroeconomics because of two main limitations: the lack of a shared frame of reference connecting MARL systems to dynamic stochastic general equilibrium (DSGE) models the profession already trusts, and the very long training times. The first limitation was recently addressed by the Multi-Agent Reinforcement Learning Business Cycle (MARL-BC), which places deep RL households in a real business cycle economy and reduces to classic DSGE models in specific limits. Here we address the second limitation. We present JaxMARL-BC, a JAX-accelerated simulator that trains MARL-BC economies in seconds (between 75x and 724x faster than the reference implementation), and scales to 200,000 independent learning households. We verify its correctness a
    
[^190]: AI气象模型稀疏自编码器特征中的热带气旋属性

    Tropical Cyclone Attributes in Sparse Autoencoder Features of an AI Weather Model

    [https://arxiv.org/abs/2610.28904](https://arxiv.org/abs/2610.28904)

    该研究以GraphCast中的热带气旋为案例，通过稀疏自编码器（SAE）探测AI气象模型的内部表征，发现气旋的位置、强度和气压结构信息以跨独立SAE和层间可重复的方式分布于多个特征中，且深层特征更能预测气旋六小时后的状态。

    

    人工智能（AI）气象模型如今已能熟练地预报天气事件，但它们在内部如何表征这些事件仍然未知。稀疏自编码器（SAE）为探测隐藏状态提供了稀疏特征，但关于事件多个物理属性的信息如何在这些特征之间组织仍不清楚。以GraphCast中的热带气旋作为测试案例，我们分析了来自205个气旋的2,115个六小时时间步，将SAE激活与位置、强度和气压结构相关联，涵盖五种第8层分解和16个处理器层。每个属性都与多个特征相关；位置和气压结构的关系在较小的子集中重叠。这种组织方式在独立的SAE和各层之间重复出现，无需特征对齐。在更深层中，主要的位置特征与六小时后的位置和气压结构的对齐程度优于当前时刻。这些结果表明，这种组织……

    arXiv:2610.28904v1 Announce Type: cross  Abstract: Artificial intelligence (AI) weather models now forecast weather events skillfully, but how they represent these events internally remains unknown. Sparse autoencoders (SAEs) provide sparse features for probing hidden states, but how information about multiple physical attributes of an event is organized across them remains unclear. Using tropical cyclones in GraphCast as a test case, we analyze 2,115 six-hourly timesteps from 205 cyclones, relating SAE activations to position, intensity, and pressure structure across five Layer-8 decompositions and 16 processor layers. Each attribute relates to multiple features; position and pressure-structure relations overlap in a smaller subset. This organization recurs across independent SAEs and layers without feature alignment. In deeper layers, the leading position feature aligns better with position and pressure structure six hours ahead than at present. These results suggest that the organiz
    
[^191]: 分布漂移下的极小极大最优预测流匹配

    Minimax-Optimal Predictive Flow Matching under Distribution Drift

    [https://arxiv.org/abs/2610.28898](https://arxiv.org/abs/2610.28898)

    提出 DyProd-FM 算法，在无需知晓漂移预算的情况下，对固定非凸神经网络类的预测流匹配问题实现了与下界匹配的极小极大最优动态遗憾。

    

    流匹配通过回归学习生成式速度场，但在分布漂移的情况下，速度场必须在当前数据到来之前就被部署使用。我们针对固定的、可能非凸的神经网络类研究这一预测问题，以每一轮最佳可表示速度场为基准衡量动态遗憾。我们提出 DyProd-FM 算法，该方法在二倍递增的历史数据前缀上重新拟合流匹配模型，并通过一个休眠式 Prod 主算法在多个拟合结果中进行选择。在无需知晓任何漂移预算的情况下，该算法在相对速度风险变化量 $\Omega$ 上取得遗憾 $\widetilde{O}(\min\{N_\omega+1,1+\sqrt{T\Omega}\})$，在目标分布变化量 $V_\mu=\sum_t W_2(\mu_{t+1},\mu_t)$ 上取得遗憾 $\widetilde{O}(\min\{N_\mu+1,1+T^{1/3}V_\mu^{2/3}\})$（均不超过近似误差与拟合误差）。针对固定函数类的匹配下界表明，这两个速率在 $T$ 和各漂移预算上均达到最优（至多相差对数因子）。以往的最优动态遗憾保证依赖于凸决策集的假设……

    arXiv:2610.28898v1 Announce Type: new  Abstract: Flow matching learns a generative velocity field by regression, but under distribution drift a field must be deployed before the current data arrive. We study this predictive problem for a fixed, possibly nonconvex, neural class, with dynamic regret against the best represented field of each round. We propose DyProd-FM, which refits flow matching on dyadic historical prefixes and selects among the fits with a sleeping Prod master. Without knowing any drift budget, it attains regret $\widetilde{O}(\min\{N_\omega+1,1+\sqrt{T\Omega}\})$ in the relative velocity risk variation $\Omega$ and $\widetilde{O}(\min\{N_\mu+1,1+T^{1/3}V_\mu^{2/3}\})$ in the target-law variation $V_\mu=\sum_t W_2(\mu_{t+1},\mu_t)$, up to approximation and fitting costs. Matching lower bounds for fixed classes show that both rates are optimal in $T$ and in the budgets up to logarithmic factors. Prior optimal dynamic-regret guarantees rely on a convex decision set with
    
[^192]: 拒绝方向在低资源语言中是否具有普适性？

    Is Refusal Direction Universal across Low-Resourced Languages?

    [https://arxiv.org/abs/2610.28897](https://arxiv.org/abs/2610.28897)

    该研究发现拒绝方向的跨语言普适性是分级的而非二元的：来自安全对齐语言的拒绝方向可广泛迁移至低资源语言，而来自强失对齐语言的拒绝方向则无法可靠地调节任何目标语言的拒绝行为。

    

    拒绝方向已被证明能够在安全对齐的语言之间以近乎完美的效果进行跨语言迁移（Wang et al., 2025），且据称不受资源水平的影响。然而，此前评估中唯一真正的低资源语言是约鲁巴语，该语言属于强失对齐语言且表现出有限的迁移效果，这使得拒绝方向在其他低资源语言中的表现仍是一个悬而未决的问题。我们将 PolyRefuse 扩展到另外三种低资源语言（白俄罗斯语、巴什基尔语和塔吉克语），并在 Qwen2.5-14B-Instruct 上复现了（Wang et al., 2025）的实验流程。我们发现，普适性是分级的而非二元的：来自安全对齐和部分对齐语言的拒绝方向能够广泛迁移，而一旦考虑天花板效应，来自强失对齐语言的拒绝方向则无法在任何测试目标上可靠地调节拒绝行为，包括其他失对齐语言。几何对齐与行为迁移进一步表……

    arXiv:2610.28897v1 Announce Type: new  Abstract: Refusal directions have been shown to transfer cross-lingually with near-perfect effectiveness across safety-aligned languages (Wang et al., 2025), reportedly regardless of resource level. However, the only truly low-resourced language evaluated is Yoruba, which is strongly misaligned and shows limited transfer, leaving open how refusal directions behave for other low-resourced languages. We extend PolyRefuse to three additional low-resourced languages (Belarusian, Bashkir, and Tajik) and replicate the pipeline of (Wang et al., 2025) on Qwen2.5-14B-Instruct. We find that universality is graded rather than binary: directions from safety-aligned and partially-aligned languages transfer broadly, while those from strongly misaligned languages fail to reliably modulate refusal behavior on any target tested, including other misaligned languages, once ceiling effects are accounted for. Geometric alignment and behavioral transfer are further sho
    
[^193]: CSD：面向跨分词器知识迁移的公共支撑集蒸馏

    CSD: Common-Support Distillation for Cross-Tokenizer Knowledge Transfer

    [https://arxiv.org/abs/2610.28895](https://arxiv.org/abs/2610.28895)

    本文提出CSD框架，通过在每个对齐输入分块处构建教师与学生模型间的公共支撑集，并以单一目标函数匹配分布，解决了词表和分词差异导致的跨分词器logits知识蒸馏对齐难题。

    

    跨分词器的logits蒸馏能够实现跨模型族的知识迁移，使学生模型能够从更强或更专精的教师模型中学习。然而，词表和分词方式的差异使得教师模型和学生模型的下一词元分布难以对齐。现有的基于logits的方法虽然易于集成到标准蒸馏流程中，但通常依赖于限制性的启发式方法或针对特定分词器对的选择。这些方法可能将监督限制在共享词表的子集上，需要为不同的分词器对手动选择不同的目标函数，或者仅匹配观察到的分块及其补集，而非逐个匹配备选的续写结果。我们提出了公共支撑集蒸馏，这是一个统一的基于logits的框架，它在每个对齐的输入分块处构建教师和学生模型之间的共享支撑集，并通过单一目标函数来匹配由此导出的分布。对于一对一的对齐情形，（摘要内容在此处被截断）

    arXiv:2610.28895v1 Announce Type: new  Abstract: Cross-tokenizer logits distillation enables knowledge transfer across model families, allowing students to learn from stronger or more specialized teachers. However, differences in vocabulary and segmentation make teacher and student next-token distributions difficult to align. Existing logits-based methods are easy to integrate into standard distillation pipelines, but often rely on restrictive heuristics or tokenizer-pair-specific choices. They may restrict supervision to shared vocabulary subsets, require manually selecting different objectives for different tokenizer pairs, or match only the observed chunk and its complement rather than individual alternative continuations. We introduce Common-Support Distillation (CSD), a unified logits-based framework that constructs a shared support between the teacher and student at each aligned input chunk and matches the induced distributions with a single objective. For one-to-one alignments, 
    
[^194]: 通过知识重塑对抗持续领域自适应中的灾难性遗忘

    Combating Catastrophic Forgetting in Continual Domain Adaptation via Knowledge Recasting

    [https://arxiv.org/abs/2610.28892](https://arxiv.org/abs/2610.28892)

    该论文提出技能模式传输框架，通过将新领域知识转换到模型已有的技能空间中、使陌生知识借助熟悉的过程被学习，从而在持续领域自适应中有效对抗灾难性遗忘。

    

    arXiv:2610.28892v1 公告类型：新论文 摘要：在后训练阶段将语言模型适配到新领域时，存在损害模型既有能力的风险，这一现象通常被称为灾难性遗忘。现有的缓解方法大多将遗忘视为模型过度漂移的后果，因此通过保留旧输出、重放旧数据，或对参数和表示的变化进行正则化来约束更新，以使更新后的模型保持接近其先前的状态。我们采取了不同的视角：与其把更新后的模型拉回其旧状态，不如将新领域转换到模型已有的知识空间中，使陌生的知识能够通过熟悉的过程被学习。为此，我们提出了技能模式传输（SST），这是一个将新知识传输至模型既有技能空间的后训练框架。SST 将每种示例类型抽象为一个技能模式，保留其核心操作、依赖关系以及……

    arXiv:2610.28892v1 Announce Type: new  Abstract: Adapting language models to new domains during post-training risks degrading previously acquired capabilities, a phenomenon commonly known as catastrophic forgetting. Existing mitigation methods largely treat forgetting as a consequence of excessive model drift. They therefore constrain updates by preserving old outputs, replaying old data, or regularizing changes to parameters and representations so that the updated model remains close to its previous state. We take a different view: Rather than pulling the updated model back toward its old state, we translate the new domain into the model's pre-existing knowledge space, making unfamiliar knowledge learnable through familiar procedures. Therefore, we propose Skill Schema Transport (SST), a post-training framework that transports new knowledge into the model's existing skill space. SST abstracts each example type into a skill schema, preserving its essential operations, dependencies, and
    
[^195]: RL-IF：追踪在线强化学习中的非局部训练数据影响

    RL-IF: Tracking Non-local Training Data Influence for Online Reinforcement Learning

    [https://arxiv.org/abs/2610.28883](https://arxiv.org/abs/2610.28883)

    该论文提出轨迹级留一指标 RL-LOO 及其一阶估计器 RL-IF，首次将在线强化学习的数据归因从单轮局部影响扩展到跨多轮的非局部影响，并同时刻画优化效应与策略诱导的采样效应。

    

    在线强化学习在决策任务中取得了显著成功，但现代在线强化学习对训练经验的质量仍然高度敏感。因此，理解单个训练样本的贡献是以数据为中心的强化学习的核心问题，其应用涵盖从经验选择到策略解释等多个方面。现有的在线强化学习数据归因方法主要关注单个训练轮次内的局部影响，忽视了跨多个训练轮次的非局部数据影响。在这项工作中，我们通过一个轨迹级的留一量化指标 RL-LOO 将在线强化学习中的数据归因形式化，该指标度量了在后续在线更新发生后，某一个训练样本对下游目标的影响。随后，我们推导出 RL-IF，一个一阶估计器，它同时通过优化效应和策略诱导的采样效应两条途径传播数据影响，并提供了相应的理论（摘要在此处截断）。

    arXiv:2610.28883v1 Announce Type: new  Abstract: Online reinforcement learning (RL) has achieved significant success in decision-making tasks, but modern online RL remains highly sensitive to the quality of training experience. Understanding the contribution of individual training instances is therefore central to data-centric RL, with applications ranging from experience selection to policy interpretation. Existing data attribution methods for online RL primarily focus on local influence within a single training round, overlooking the non-local data influence across multiple training rounds. In this work, we formalize data attribution in online RL through a trajectory-level leave-one-out quantity, RL-LOO, which measures how one training sample influences a downstream target after subsequent online updates have taken place. We then derive RL-IF, a first-order estimator that propagates data influence through both optimization effects and policy-induced sampling effects, and provide theo
    
[^196]: 实际上，所有可学习的分布类都是可隐私学习的

    Actually, All Learnable Distribution Classes Are Privately Learnable

    [https://arxiv.org/abs/2610.28878](https://arxiv.org/abs/2610.28878)

    本文证明了所有在总变差距离下可学习的分布类在（ε, δ）-差分隐私下同样可学习，从而证明了 Ashtiani 的猜想。

    

    我们证明，每一个在总变差距离下能够以有限个样本学习到任意小目标误差的分布类，同样可以在（$\varepsilon, \delta$）-差分隐私条件下以有限个样本学习到任意小的目标误差。这证明了 Ashtiani 提出的一个猜想。Bun、Kamath、Mouzakis 和 Singhal 之前的工作 [arXiv:2402.00267] 曾对该猜想给出了弱反驳，表明当可学习性相对于固定目标误差来定义时，这一蕴含关系并不成立。与之相反，我们证明，在可学习性的标准定义下，该猜想是成立的。

    arXiv:2610.28878v1 Announce Type: new  Abstract: We show that every class of distributions that is learnable up to arbitrarily small target error in total variation distance with finitely many samples is also learnable under ($\varepsilon, \delta$)-differential privacy up to arbitrarily small target error with finitely many samples. This proves a conjecture of Ashtiani. Previous work by Bun, Kamath, Mouzakis, and Singhal [arXiv:2402.00267] gave a weak refutation of this conjecture, showing that this implication is not true when learnability is defined with respect to a fixed target error. In contrast, we show that, under the standard definition of learnability, the conjecture is true.
    
[^197]: LEAN：通过损失工程调制模型权重敏感度分布以实现高效、容错的神经网络

    LEAN: Loss-Engineered Modulation of the Model Weight-Sensitivity Distribution for Efficient, Error-Tolerant Neural Networks

    [https://arxiv.org/abs/2610.28872](https://arxiv.org/abs/2610.28872)

    LEAN提出了一种新颖的复合正则化方法（包含范围项和尾部项），能够系统性地调制训练后模型的权重敏感度分布，仅需十个epoch的微调即可显著提升神经网络对位翻转错误的容忍度并降低存储开销。

    

    训练好的神经网络中的各个权重对模型精度的敏感度并不均匀：最敏感的1%权重承担了位翻转可能造成的总损害的大约五分之一。这种敏感度分布决定了模型的位翻转容忍度、纠错码必须覆盖的权重比例以及每个权重所需的存储位数；而现有的方案只能在训练所留下的分布上进行工作。我们证明可以系统地调制或改变训练后模型的权重敏感度分布，从而将权重从一个敏感度区间转移到另一个区间。存储权重的敏感度等于其梯度与位翻转在存储格式中所引起步长（变化量）的乘积，LEAN对这两个因素添加了一个复合正则化项：一个约束翻转步长的范围项和一个降低最敏感权重梯度的尾部项，其中一个系数λ用于控制分布移动的程度。使用该正则项进行十个epoch的训练——无论是作为微调还是作为训练的最后十个epoch——（此处摘要内容被截断）

    arXiv:2610.28872v1 Announce Type: new  Abstract: The weights of a trained network do not have uniform sensitivity to model accuracy: the 1% most sensitive weights carry about a fifth of the damage a bit flip can do. This distribution decides the bit-flip tolerance of a model, the fraction of weights a code must cover and the bits each weight needs; existing schemes work with the distribution training leaves. We show that we can systematically modulate or alter the weight sensitivity distribution of a trained model, thereby shifting weights from one sensitivity bin to another. The sensitivity of a stored weight is the product of its gradient and the step a flip takes in the storage format, and LEAN adds a composite regularizer on both factors, a range term that bounds the flip step and a tail term that lowers the gradient of the most sensitive weights, with one coefficient {\lambda} setting how far the distribution moves. Ten epochs with the term, as a fine-tune or as the last epochs of
    
[^198]: 循环Transformer中截断反向传播会损失多少梯度？

    How Much Gradient Does Truncated Backprop Lose in Looped Transformers?

    [https://arxiv.org/abs/2610.28871](https://arxiv.org/abs/2610.28871)

    该论文首次在九个循环语言模型上精确量化了截断反向传播（仅反传最后k次循环）所丢弃的权重梯度比例，发现该损失在已训练模型上于靠近输出处即已确定、几乎不随循环总数变化，并在微调过程中逐渐减小。

    

    循环Transformer通过多次重复使用同一组层来构建，并且通常只通过最后k次循环进行反向传播来训练。我们在九个循环语言模型上，精确测量了这种做法在单次反向传播中对每个k值所丢弃的权重梯度比例，并报告了三项发现。第一，在已训练的模型上，这种被损失的梯度在靠近输出的位置就已基本确定：一旦循环次数远大于k，它几乎不再随循环总数变化（在我们测试的三个模型上，从32次到64次循环的变化幅度在约0.03以内，这与一项针对Parcae的预注册测试的预测相符）。在反向信号收缩的情形下，ρ^k（其中ρ为信号每循环拟合出的收缩率）在中等循环次数下能较好地近似该损失，但在长循环次数下会高估它，这是因为随着拟合纳入更多循环，拟合出的ρ会向上漂移。而在Ouro模型上信号是增长的，保留四次循环中的两次仍会损失最终输出梯度的81%。第二，这种损失的梯度在微调过程中会下降，下降了……（原文摘要在此处截断）

    arXiv:2610.28871v1 Announce Type: new  Abstract: Looped transformers reuse one block of layers many times and are often trained by backpropagating through only the last k loops. We measure how much of the weight gradient this throws away, exactly and for every k from one backward pass, on nine looped language models, and report three findings. First, on a trained model this lost gradient is set near the output: once the loop count is well above k, it barely changes with the loop count (within about 0.03 from 32 to 64 loops on the three models we tested, as a pre-registered test predicted for Parcae). Where the backward signal shrinks, $\rho^k$, with $\rho$ the signal's fitted shrink per loop, approximates it at moderate loop counts but overstates it at long ones, because the fitted $\rho$ drifts up as the fit takes in more loops. On Ouro the signal grows, and keeping two of four loops still loses 81% of the final output's gradient. Second, the lost gradient falls during fine-tuning, by
    
[^199]: 神经网络损失景观中临界轨道的通用拓扑对称性指标

    A General Topological Symmetry Index for Critical Orbits In Neural Network Loss Landscapes

    [https://arxiv.org/abs/2610.28864](https://arxiv.org/abs/2610.28864)

    本文基于等变度理论建立了一个通用的拓扑对称性指标，用于系统刻画神经网络损失景观中由架构对称性产生的临界轨道，能够检测临界点的迷向性并量化其净拓扑贡献，且适用于任意网络架构包括退化情形。

    

    由于神经网络损失景观的高维性，对其全局结构的系统性理解仍然具有挑战性。源于网络架构和学习目标的对称性为研究这种结构的某些方面提供了一种途径。尽管越来越多的研究开始关注这些对称性，但对其以及它们所产生的临界轨道的系统性刻画仍然不完整。为填补这一空白，本工作基于等变度理论家族的方法论，建立了一个拓扑对称性指标来解码这些对称性。该框架适用于任意网络架构，并包含适用于退化临界轨道的方法。该指标能够检测临界点的迷向性，并量化相应的净拓扑贡献。文中给出了若干示例，包括softplus多层感知机、Transformer以及等变神经网络……

    arXiv:2610.28864v1 Announce Type: new  Abstract: A systematic understanding of the global structure of neural-network loss landscapes remains challenging due to their high dimensionality. Symmetries arising from network architectures and learning objectives provide a way to investigate aspects of this structure. Although a growing number of studies have drawn attention to these symmetries, a systematic characterization of them, as well as of the critical orbits they give rise to, remains incomplete. To address this gap, in this work we establish a topological symmetry index, based on the methodologies from the family of equivariant degree theories, to decode these symmetries. The frameworks apply to arbitrary network architectures and include methods applicable to degenerate critical orbits. The index detects the isotropy of the critical points and quantifies the corresponding net topological contributions. Several examples, including softplus MLPs, Transformers, and equivariant neural
    
[^200]: DualHeads：简化的奖励倾斜少步扩散模型

    DualHeads: Reward-tilted few-step diffusion made simple

    [https://arxiv.org/abs/2610.28858](https://arxiv.org/abs/2610.28858)

    提出 DualHeads 方法，通过在冻结的少步扩散模型上训练奖励头和参考头两个轻量级二元分类器，仅利用外部奖励评估和参考样本即可实现少步扩散模型的偏好对齐，且该目标等价于 KL 正则化的奖励最大化。

    

    蒸馏后的扩散模型能够以少量采样步骤生成高质量图像，但使其与任务需求和人类偏好对齐却十分困难。为多步扩散模型开发的方法依赖于可微奖励，或依赖于蒸馏生成器可能不具备的分数场和随机轨迹。我们探究是否可以仅使用外部奖励评估和参考样本来适配少步生成器，并证明两个二元分类器就足够了。我们的方法 DualHeads 将它们实现为冻结扩散主干上的轻量级头：奖励头学习同一提示词下哪些样本得分更高，参考头将参考图像与生成图像区分开来，而生成器则最大化两者 logits 的加权和。当这些头达到最优时，该目标等价于学习奖励的 KL 正则化最大化，且头权重的比值决定了正则化强度。

    arXiv:2610.28858v1 Announce Type: cross  Abstract: Distilled diffusion models generate high-quality images in a few sampling steps, but aligning them with task requirements and human preferences is difficult. Methods developed for multi-step diffusion models rely on differentiable rewards, or on score fields and stochastic trajectories that distilled generators may lack. We ask whether a few-step generator can be adapted using only external reward evaluations and reference samples, and show that two binary classifiers suffice. Our method, DualHeads, implements them as lightweight heads on a frozen diffusion backbone. The reward head learns which samples of the same prompt score higher, the reference head separates reference images from generated ones, and the generator maximizes a weighted sum of their logits. When the heads are optimal, this objective is KL-regularized maximization of the learned reward, and the ratio of head weights sets the regularization strength. Empirically, with
    
[^201]: 基于高保真反馈的低保真优化模型自监督调优

    Self-Supervised Tuning of Low-Fidelity Optimization Models via High-Fidelity Feedback

    [https://arxiv.org/abs/2610.28851](https://arxiv.org/abs/2610.28851)

    该论文提出一种自监督框架，将低保真优化模型嵌入为可微分优化层，借助高保真校正问题对偶变量提供的梯度进行端到端训练，无需昂贵的高保真解标签即可自动修正低保真模型在最优潮流、最优气流和池化等问题中的系统性误差。

    

    低保真（LF）优化模型因求解高效可靠而被广泛应用于大规模运营场景，但相对于能更好刻画物理与运行约束的高保真（HF）模型，它们往往存在系统性误差。现有的基于学习的方法通常依赖使用高保真解的监督或半监督训练，然而高保真解的计算成本高昂，且可能与高保真运行目标不一致。我们提出一种自监督框架，直接依据高保真反馈对低保真优化模型进行参数化。该框架将低保真问题嵌入为一个可微分优化层并进行端到端训练，同时由一个高保真校正问题评估其解，并通过对偶变量提供梯度信息，而无需高保真解标签。我们在最优潮流、最优天然气流和池化问题上对该框架进行了评估，涵盖了不同来源的不确定性。（原文摘要在此处截断）

    arXiv:2610.28851v1 Announce Type: cross  Abstract: Low-fidelity (LF) optimization models are widely used in large-scale operations because they can be solved efficiently and reliably, but they often exhibit systematic errors relative to high-fidelity (HF) formulations that better capture physical and operational constraints. Existing learning-based approaches typically rely on supervised or semi-supervised training with HF solutions, which are expensive to compute and may not align with HF operational objectives. We propose a self-supervised framework that parameterizes LF optimization models directly from HF feedback. The low-fidelity problem is embedded as a differentiable optimization layer and trained end-to-end, while a HF correction problem evaluates its solution and provides gradient information through its dual variables, without requiring HF solution labels. We evaluate the framework on optimal power flow, optimal gas flow, and pooling problems, spanning different sources of l
    
[^202]: 高斯先验何时失效？强化学习微调的潜空间几何研究

    When Do Gaussian Priors Fail? A Latent Geometry Investigation of RL Fine-Tuning

    [https://arxiv.org/abs/2610.28849](https://arxiv.org/abs/2610.28849)

    本研究揭示了高斯先验VAE在强化学习微调下的结构性缺陷——KL项主动惩罚模态分离导致潜空间模态重叠并引发模型崩溃——并指出崩溃阈值相差71倍的根源可归结为单一几何量 d_min/(2σ₀)。

    

    潜在扩散模型几乎普遍在VAE潜空间上采用高斯先验。对于基础生成任务，这一默认选择并无问题；然而在强化学习微调下，它带有一种结构性局限。标准VAE的KL项不仅忽略了模态结构——它还主动惩罚模态的分离性与紧凑性。当数据为多模态分布时，ELBO的最优解会将不同的模态压缩进相互重叠的潜空间中。强化学习微调会进一步放大这种重叠：得分网络在模态之间的鞍点区域分辨率不足，动态-变分放大效应急剧上升，崩溃边界β_c被推至远高于其变分尺度的位置。在真实的丙氨酸二肽玻尔兹曼分布上，高斯先验VAE在β_c = 0.224时崩溃，而模态追踪的高斯混合先验在β_c = 0.00316时才崩溃，同一流程下两者相差71倍。我们将这一差距追溯至一个单一的几何中介量 d_min/(2σ₀)，……

    arXiv:2610.28849v1 Announce Type: new  Abstract: Latent diffusion models almost universally place a Gaussian prior on the VAE latent space. For base generation this default is unproblematic; under reinforcement-learning fine-tuning, however, it carries a structural limitation. The standard VAE's KL term does not merely ignore mode structure - it actively taxes separation and tightness. When data are multimodal, the ELBO optimum compresses distinct modes into an overlapping latent. RL fine-tuning amplifies this overlap: the score network is under-resolved on the saddle region between modes, dynamic-to-variational amplification rises sharply, and the collapse boundary beta_c is pushed far above its variational scale. On the real alanine-dipeptide Boltzmann distribution, a Gaussian-prior VAE collapses at beta_c = 0.224 while a mode-tracked Gaussian-mixture prior collapses at 0.00316, a 71x gap from the same pipeline.   We trace this gap to a single geometric mediator, d_min/(2 sigma_0), a
    
[^203]: KIRA：面向儿童的双通道分析全栈阅读筛查系统

    KIRA: A Full-Stack Reading Screening System with Dual-Pathway Analysis for Children

    [https://arxiv.org/abs/2610.28840](https://arxiv.org/abs/2610.28840)

    KIRA是一个面向儿童的双通道阅读筛查系统，通过将预训练语音识别模型适配儿童语音（词错误率从13.88%降至8.84%，音素错误率从41.66%降至23.00%），实现了单词级流畅度评分与音素级错误检测的结合，并通过教师界面支持针对性阅读干预。

    

    早期年级的阅读筛查对于识别存在阅读困难风险的儿童至关重要。现有的基于AI的筛查系统往往在分析范围和评估指标上存在局限，或者是一些缺乏透明模型规格与评估的专有系统。我们提出了KIRA，一个将单词级流畅度评分与详细的音素级错误检测相结合的双通道筛查系统。我们将预训练的单词和音素识别器适配到儿童语音上，将单词错误率从13.88%降低到8.84%，音素错误率从41.66%降低到23.00%。基于这些针对儿童适配的模型，我们实现了单词阅读（WRE）和口头阅读流畅度（ORF）这两项K-2年级筛查中的基础任务。该系统将这些识别器与面向教师的网页界面集成，呈现流畅度分数以及音素反馈，以便进行潜在的针对性干预。KIRA将现代儿童语音识别技术引入到一个全栈……（原文摘要到此截断）

    arXiv:2610.28840v1 Announce Type: new  Abstract: Early-grade reading screening is critical for identifying children at risk of reading difficulty. Existing AI-based screeners are often limited in the scope of their analyses and evaluation metrics, or are proprietary systems that lack transparent model specifications and evaluation. We present KIRA, a dual-pathway screening system that pairs word-level fluency scoring with detailed phoneme-level error detection. We adapt pretrained word and phoneme recognizers to children's speech, reducing word error rate from 13.88% to 8.84% and phoneme error rate from 41.66% to 23.00%. Powered by these child-adapted models, we implement Word Reading (WRE) and Oral Reading Fluency (ORF), two foundational tasks in K-2 screening. The system integrates these recognizers with a web-based teacher-facing interface presenting fluency scores alongside phoneme feedback for potential targeted intervention. KIRA brings modern child-speech recognition into a full
    
[^204]: Sketch2Sim：一个用于手绘电路分析的端到端模块化框架

    Sketch2Sim: an end-to-end modular framework for hand-drawn circuit analysis

    [https://arxiv.org/abs/2610.28836](https://arxiv.org/abs/2610.28836)

    本文提出Sketch2Sim端到端模块化框架，通过多阶段流水线（含双YOLO检测器）将手绘电路图照片自动转换为SPICE兼容网表并计算电路的节点电压与电流，同时构建了1000个标注的手绘RLC电路数据集，旨在改善电路教学体验并加速工程设计评审。

    

    本研究提出了一种从草图到仿真的系统，用于解决手绘电路图的识别问题。该系统将原始图像转换为兼容SPICE（集成电路通用仿真程序）的网表，并自动计算电路的节点电压和电流。另一个目标是提出一个整体性框架，通过增强电路的教学与学习体验以及加快工程设计评审流程，使学生和教育工作者受益。本文提出了一个多阶段流水线，将手绘电路的原始照片转换为可执行的类SPICE网表，从而提供解决方案。研究团队构建了一个包含1000个手绘电阻-电感-电容（RLC）电路的数据集，并对元件、文本区域和字符识别进行了标注。所提出的系统将电路图分析分解为三个阶段： 基于双YOLO（You Only Look Once）检测器的元件与文本检测……（原文摘要在此处截断）

    arXiv:2610.28836v1 Announce Type: cross  Abstract: This study proposes a sketch-to-simulation system to solve handwritten circuit diagrams. The system converts raw images into a Simulation Program with Integrated Circuit Emphasis (SPICE)-compatible netlist and automatically calculates the circuit's nodal voltages and currents. Another objective is to propose a holistic framework that benefits students and educators by enhancing the teaching and learning experience of electric circuits and expediting engineering design reviews. This article presents a multistage pipeline that converts raw photographs of handwritten electric circuits into executable SPICE-like netlists, providing a solution. A dataset of 1,000 hand-drawn Resistor-Inductor-Capacitor (RLC) circuits was curated and annotated for components, text regions, and character recognition. The proposed system decomposes a circuit diagram into three stages: (i) dual You Only Look Once (YOLO)-based detectors for components and text re
    
[^205]: 设计如此脆弱？面向体系鲁棒任务协同的多场景强化学习

    Brittle by Design? Multi-Scenario Reinforcement Learning for Robust Mission Coordination in Systems of Systems

    [https://arxiv.org/abs/2610.28818](https://arxiv.org/abs/2610.28818)

    该论文提出一种将高保真数字任务模型与基于近端策略优化的强化学习相结合的智能任务协同方法，并通过多场景航空灭火任务验证了其在不确定动态环境下实现体系鲁棒自适应任务分配的有效性。

    

    随着系统工程从单体系统转向体系，任务工程已成为系统工程领域的一种新思路。任务环境具有不确定性和动态性，任务结果取决于任务资产与环境的交互方式，这使得静态架构变得脆弱，因而需要既灵活又分析严谨的任务工程方法。我们提出了一种智能任务协同方法，将高保真数字任务模型与强化学习相结合，以实现自适应任务分配。依托由数字任务模型和基于代理的仿真构成的数字工程基础设施，我们将任务战术管理形式化为马尔可夫决策过程，并使用近端策略优化训练强化学习协调器，根据实际任务结果将任务状态映射到战术动作。通过包含三个场景的航空灭火任务对该方法进行了演示。

    arXiv:2610.28818v1 Announce Type: cross  Abstract: As systems engineering (SE) shifts from monolithic systems to Systems of Systems, Mission Engineering (ME) has emerged as a new line of thinking in SE. Mission environments are uncertain and dynamic, and outcomes depend on how mission assets interact with them, rendering static architectures brittle and calling for flexible yet analytically rigorous ME approaches. We propose an intelligent mission coordination methodology integrating high-fidelity digital mission models with Reinforcement Learning (RL) for adaptive task allocation. Using a Digital Engineering (DE) infrastructure comprising a digital mission model and agent-based simulation, we formulate mission tactics management as a Markov Decision Process and train an RL coordinator with Proximal Policy Optimization to map mission states to tactical actions from realized outcomes. An aerial firefighting mission with three scenarios demonstrates the approach. First, all trained coord
    
[^206]: 基于大语言模型的智能体系统中自动化失败归因的机制研究

    A Mechanistic Study of Automated Failure Attribution in LLM-based Agentic Systems

    [https://arxiv.org/abs/2610.28811](https://arxiv.org/abs/2610.28811)

    本文对基于大语言模型的智能体系统中的自动化失败归因进行了机制层面的研究，分析了为何直接提示方法效果不佳，并探索了提升失败归因性能的改进策略。

    

    智能体系统通过让多个基于大语言模型（LLM）的智能体相互协作、并行执行子任务，在复杂任务上展现出了强大的能力。然而，这种协作结构为可靠地评估和诊断系统故障带来了新的挑战，尤其是失败归因问题：即追踪系统的执行过程，以识别导致最终错误结果的责任智能体步骤。由于人工执行这项任务既耗时又费力，近期的研究工作开始转向使用大语言模型作为裁判来自动化这一过程，称为自动化失败归因。这些方法假设一个足够强大的模型能够直接从交互记录中推断出失败的步骤。然而在实践中，直接提示往往表现不佳，这促使后续工作探索各种提升AFA性能的方法，包括替代性提示策略等。

    arXiv:2610.28811v1 Announce Type: new  Abstract: Agentic systems have demonstrated strong capabilities on complex tasks by enabling multiple interacting large language model (LLM)-based agents to collaborate and execute subtasks in parallel. However, this collaborative structure introduces new challenges for reliably evaluating and diagnosing system failures, particularly failure attribution: tracing a system's execution to identify the agent step responsible for an incorrect final outcome. Because manually performing this task is time- and labor-intensive, recent work has turned to LLMs as judges to automate the process, referred to as automated failure attribution (AFA). These approaches assume that a sufficiently capable model can infer the failing step directly from an interaction transcript. In practice, however, direct prompting often performs poorly, which motivates follow-up work that explores various ways to improve AFA performance, including alternative prompting strategies a
    
[^207]: 将长期记忆与大语言模型结合需付出高昂代价：从提示注入到持久性入侵

    Combining Long-Term Memory and LLMs Comes at a High Cost: From Prompt Injection to Persistent Compromise

    [https://arxiv.org/abs/2610.28810](https://arxiv.org/abs/2610.28810)

    本文揭示了现代对话式AI平台的持久记忆功能使提示注入攻击能够跨会话传播（记忆注入），并提出通用威胁模型证明所有NIST攻击目标均可实现，凸显了这一新型安全威胁的严重性与紧迫性。

    

    近年来，生成式人工智能的广泛普及也带来了提示注入和越狱攻击的同步增长。这些攻击手段通过嵌入恶意指令来操纵语言模型，使其绕过预期的行为和操作边界。在传统聊天机器人中，由于状态记忆不会跨对话保留，此类攻击仅限于单个会话内。然而，现代对话界面中的持久记忆功能使得攻击能够跨会话传播，形成了一种被称为“记忆注入”的威胁向量。本工作在领先的大语言模型即服务平台中探讨了这一新兴漏洞，强调了其现实世界的严重性以及采取对策的必要性。现有文献主要关注针对特定平台功能定制的违规行为，而我们提出了一个通用的威胁模型。我们证明了所有NIST攻击目标在ChatGPT等平台上均可共同实现。

    arXiv:2610.28810v1 Announce Type: cross  Abstract: The widespread adoption of generative artificial intelligence in recent years has brought a parallel rise in prompt injection and jailbreak attacks. By embedding malicious instructions, these exploits manipulate language models to bypass intended behavior and operational boundaries. In traditional chatbots, such attacks are confined to a single session because state memory is not preserved across conversations. However, persistent memory features in modern conversational interfaces enable cross-session attack propagation, establishing a threat vector known as memory injection. This work explores this emerging vulnerability in leading LLM-as-a-service platforms, highlighting its real-world severity and the need for countermeasures. While existing literature focuses on violations tailored to specific platform features, we present a general threat model. We demonstrate that all NIST attack goals are collectively achievable across ChatGPT 
    
[^208]: 迈向现实的变革模型：决策与动力系统的因果视角

    Towards Realistic Models of Change: A Causal Perspective on Decision-Making and Dynamical Systems

    [https://arxiv.org/abs/2610.28803](https://arxiv.org/abs/2610.28803)

    该论文从因果视角研究决策与动力系统中的现实变革模型，其第一部分针对算法追索权问题提出了有别于既有框架的新视角，旨在识别个体可采取哪些行动来推翻不利的决策。

    

    尽管统计学习在近几十年中取得了显著进展——这得益于大型数据集和改变科学实践的数据驱动方法——但它仍面临一个关键局限：难以泛化到已见数据之外，在分布偏移下常常失效。因果性提供了一个有前景的解决方案：因果模型关注变化背后的底层机制而非表面关联，能够在外部干预下进行稳健推理。这使得系统不仅能从数据中学习，还能对假设情景进行推理、适应新环境并支持更好的决策。本论文分两部分探讨这些方法如何相互启发。第一部分研究算法追索权，即识别个体可以做出哪些改变以推翻不利决策，我们针对已有框架提出了一种替代视角，主张……（原文摘要在此处截断）

    arXiv:2610.28803v1 Announce Type: new  Abstract: While statistical learning has made remarkable progress in recent decades, driven by large datasets and data-driven protocols that have transformed scientific practice, it still faces a key limitation: it struggles to generalize beyond the data it has seen, often failing under distribution shift. Causality offers a promising solution: causal models focus on the underlying mechanisms of change rather than surface-level associations, enabling robust reasoning under external interventions. This allows systems not only to learn from data, but also to reason about hypothetical scenarios, adapt to new environments, and inform better decisions. This thesis explores how these approaches can inform each other, in two parts.   The first part examines algorithmic recourse, that is, identifying which changes an individual can make to overturn an unfavorable decision. We propose an alternative perspective to the established framework, arguing for a m
    
[^209]: 高斯测度下神经网络的深度分离

    Depth Separation for Neural Networks under Gaussian Measure

    [https://arxiv.org/abs/2610.28800](https://arxiv.org/abs/2610.28800)

    该论文在高斯测度下建立了神经网络的深度分离结果：存在一个径向函数可被多项式规模的3层ReLU网络以小L²误差逼近，但2层网络要逼近它需要指数级规模，其核心工具是关于径向函数Hermite高频展开的不可逼近性定理，且全部证明均经Lean 4形式化验证。

    

    我们重新审视了由Eldan和Shamir、Daniely等人研究过的神经网络深度分离问题。具体而言，我们考虑在d维标准高斯测度下径向函数的L²逼近这一自然设定。我们证明，存在一个标准高斯测度下的L²径向函数，它可以被一个规模关于d为多项式的深度为3的ReLU神经网络以固定的足够小的L²误差逼近，但对于权重具有亚指数上界的深度为2的网络而言，若要逼近该函数，其规模必须至少关于d呈指数级。为此，我们证明了一个一般性结果：如果某个径向函数在其Hermite展开的高频部分具有可观的质量，那么它无法被宽度较小且权重有界的深度为2的神经网络很好地逼近。所有定理、引理、命题和推论均已在Lean 4中得到形式化验证。

    arXiv:2610.28800v1 Announce Type: cross  Abstract: We revisit the problem of depth separation for neural networks, as studied by Eldan and Shamir, Daniely, and others. Specifically, we consider the natural setting of $L^2$ approximation of radial functions under the standard Gaussian measure in $d$ dimensions. We show that there is a radial function in $L^2$ under the standard Gaussian measure that can be approximated to a fixed sufficiently small $L^2$ error by a depth-3 ReLU neural network of polynomial size in $d$ but can only be approximated by depth-2 networks with subexponentially bounded weights if their size is at least exponential in $d$. For that, we prove a general result showing that if a radial function has substantial mass in the high frequencies of its Hermite expansion, then it cannot be well approximated by a depth-2 neural network with small width and bounded weights. All theorems, lemmas, propositions, and corollaries are verified in Lean~4.
    
[^210]: 迭代式机器学习智能体框架的显式反馈契约

    An Explicit Feedback Contract for Iterative ML-Agent Harnesses

    [https://arxiv.org/abs/2610.28798](https://arxiv.org/abs/2610.28798)

    该论文提出了一种显式反馈契约，通过基于配对标准误的噪声阈值来控制验证分数的发布，防止机器学习智能体对验证集进行自适应选择，从而避免在无真实增益的任务上虚报性能提升。

    

    迭代式机器学习智能体通过编辑代码、运行实验、读取验证分数并决定下一步尝试什么来开发模型。在当前的框架中，这一循环由每次尝试后返回的精确验证分数来驱动，这使得验证集容易受到自适应选择的影响。在这项工作中，我们为机器学习智能体框架引入了一种显式反馈契约，它规定了智能体可以观察什么、迭代何时应该停止、以及何时才可以触及测试集。该契约仅当提交的结果超过由配对标准误计算得出的噪声阈值时才发布验证分数，因此无需调节任何边际参数。我们在点击率预测、学习排序以及一个没有任何信号可寻的对照任务上进行了实验。对照任务是最严苛的场景，因为其真实增益在构造上为零：读取精确分数的智能体最终报告了0.0157的增益，而在该契约下它报告了……

    arXiv:2610.28798v1 Announce Type: new  Abstract: Iterative ML agents develop models by editing code, running experiments, reading a validation score, and choosing what to try next. In current harnesses this loop is served by an exact validation score returned after every attempt, which leaves the validation split open to adaptive selection. In this work, we introduce an explicit feedback contract for ML-agent harnesses, which specifies what an agent may observe, when iteration should stop, and when the test split may be touched. The contract releases a validation score only when a submission clears a noise threshold computed from the paired standard error, so no margin has to be tuned. We ran experiments on click-through-rate prediction, learning to rank, and a control task with no signal to find. The control is the sharpest case, because the true gain there is zero by construction: an agent reading exact scores ended by reporting a gain of 0.0157, while under the contract it reported 
    
[^211]: CALLIOPE：无需嵌入向量的AI使用私密洞察

    CALLIOPE: Private Insights into AI Use, Without Embeddings

    [https://arxiv.org/abs/2610.28797](https://arxiv.org/abs/2610.28797)

    CALLIOPE是一个支持差分隐私的自顶向下主题层次发现框架，它不依赖向量嵌入聚类，而是在整个流程中直接利用LLM能力，在高隐私保护程度下比基于嵌入的方法表现更好。

    

    我们提出了CALLIOPE，这是一个自顶向下的框架，用于在大型非结构化数据语料库（例如LLM对话）中发现主题层次结构，同时支持差分隐私（DP）。与传统的乃至最近的LLM增强方法不同——这些方法最终都依赖于对向量嵌入进行聚类——我们的框架在整个流程中直接利用LLM的能力。实验表明，在差分隐私设置下，该方法比现有的（基于嵌入的）方法具有更好的效用，尤其是在高隐私保护程度的情况下。我们使用多种传统的聚类指标以及我们提出的一些新颖效用指标来评估CALLIOPE输出的质量。

    arXiv:2610.28797v1 Announce Type: cross  Abstract: We introduce CALLIOPE, a top-down framework for discovering topic hierarchies in large corpora of unstructured data (e.g., LLM conversations), while supporting differential privacy (DP). Unlike traditional and even recent LLM-augmented approaches that ultimately rely on clustering vector embeddings, our framework leverages LLM capabilities directly for the entire pipeline. Experiments demonstrate that this approach provides better utility over existing (embedding-based) methods in the DP setting, especially in the high privacy regime. We evaluate the quality of CALLIOPE outputs using a variety of traditionally studied clustering metrics, as well as some novel utility metrics that we introduce.
    
[^212]: ArcFlow：面向多求解器预算的流策略训练

    ArcFlow: Training Flow Policies for Multiple Solver Budgets

    [https://arxiv.org/abs/2610.28788](https://arxiv.org/abs/2610.28788)

    提出 ArcFlow 部署感知算法，通过动作对齐监督、预算循环暴露和最差预算检查点选择，在多个求解器预算范围内训练单一流匹配策略，避免单一预算强化学习导致的低成本部署性能退化。

    

    arXiv:2610.28788v1 公告类型：新论文。摘要：流匹配策略通过求解器的步数来调节推理成本，但在单一固定求解器预算下进行强化学习，可能会掩盖在更低成本部署预算下的性能退化，甚至会抹除预训练策略中已有的一步推理能力。我们提出了动作比率裁剪流，这是一种面向部署的算法，用于在指定的求解器预算范围内训练单一的流策略。ArcFlow 结合了动作对齐监督、预算暴露和最差预算检查点选择三项技术。对于重建目标，其桥接机制将时间探针聚合为每个动作对应的一个代理比率和一个 PPO 风格的裁剪，且其边界残差精确等于部署时的一步重建误差。循环训练使共享的场暴露于整个部署预算范围，而检查点选择机制则优化其性能下限。对于原生速度目标，多预算端点监督在无需替换原生形式的情况下应用了相同的原则。

    arXiv:2610.28788v1 Announce Type: new  Abstract: Flow-matching policies adjust inference cost through the number of solver steps, but reinforcement learning at a single fixed solver budget can hide degradation at cheaper deployment budgets and even erase useful one-step behavior present in a pretrained policy. We introduce Action-Ratio-Clip Flow (ArcFlow), a deployment-aware algorithm for training one flow policy across a specified range of solver budgets. ArcFlow combines action-aligned supervision, budget exposure, and worst-budget checkpoint selection. For reconstruction objectives, its bridge aggregates temporal probes into one surrogate ratio and one PPO-style clip per action, with a boundary residual exactly equal to deployed one-step reconstruction error. Cycling exposes the shared field to the deployment range, and selection optimizes its performance floor. For native velocity objectives, multi-budget endpoint supervision applies the same principle without replacing the native 
    
[^213]: 有人在执行卡特尔吗？强化学习教会了市场中的LLM智能体什么

    Is Anyone Enforcing the Cartel? What Reinforcement Learning Teaches LLM Agents in Markets

    [https://arxiv.org/abs/2610.28787](https://arxiv.org/abs/2610.28787)

    该研究提出一种在反事实市场副本中强制智能体作弊以检验合谋执行机制的审计方法，发现RL训练的LLM定价智能体只会学会训练中易尝试且被奖励的行为（被威慑或跟随降价），而无法自发掌握执行卡特尔所必需的高代价价格战惩罚。

    

    对大型语言模型（LLM）智能体的评估报告称，定价智能体会以牺牲他人利益为代价进行合谋。然而，合谋的定义核心在于执行机制：一个由竞争者分享垄断利润的卡特尔，只有当作弊的成员受到惩罚时才能维持。我们引入了一种审计方法，通过在市场的反事实副本中强制某个成员作弊来检验执行机制，并用它来研究强化学习（RL）教会了定价与交易中的智能体什么。经RL训练的小型LLM学到的是其训练对手所奖励的行为，但仅限于它们容易尝试的行为。在自我对弈中，它们学会的是无视对手的固定规则；面对会施以惩罚的对手，它们学会被威慑；面对作弊的对手，它们学会跟随其降价，却永远学不会那种本可威慑作弊者的高昂代价的价格战——而这种策略它们很少尝试。被强制加入训练中的价格战能够被学会，但大多只在被迫使用的情境下才生效。面对自身的副本时，被威慑的和跟随降价的智能体都能获得相同的卡特尔利润，但只有跟随降价……（原文摘要到此截断）

    arXiv:2610.28787v1 Announce Type: new  Abstract: Evaluations of large language model (LLM) agents report that pricing agents collude at others' expense. Yet collusion is defined by enforcement: a cartel, in which competitors share monopoly profits, holds only if a member who cheats is punished. We introduce an audit that tests enforcement by forcing a member to cheat in counterfactual copies of a market, and use it to study what reinforcement learning (RL) teaches agents in pricing and trading. Small LLMs trained by RL learn what their training rivals reward, but only what they readily try. In self-play, they learn fixed rules ignoring their rivals; against a punishing rival, to be deterred; against a cheating rival, to match its cuts, never the costly price war that would deter it, which they rarely try. A war forced into training is learned, but mostly only where it was forced. Against their own copies, deterred and matching agents can earn the same cartel profits, but only matching 
    
[^214]: 先测量，后分配：强化学习中MCP智能体的校准式Rollout预算分配

    Measure First, Then Allocate: Calibrated Rollout Budgets for MCP Agents in Reinforcement Learning

    [https://arxiv.org/abs/2610.28785](https://arxiv.org/abs/2610.28785)

    提出BRACE方法，利用奖励分歧测量与历史任务证据相结合的相关奖励模型，在分配生成预算前前瞻性预测混合奖励组，显著提升强化学习中MCP工具使用智能体的rollout预算分配效率。

    

    训练使用工具的语言模型需要高昂的交互成本，然而同一rollout组内完全相同的奖励无法提供组间相对的学习信号。我们利用奖励分歧来指导评测框架评估以及在模型上下文协议环境中策略训练的rollout分配。我们首先提出Discord，它采用精确配对推理和保持决策的提前停止来测试评测框架的修改：只有存在分歧的配对结果才能使一个评测框架优于另一个。这一洞察启发了BRACE（基于相关情景的银行校准rollout分配）中的前瞻性分配方法。BRACE结合历史任务证据、任务成功率的不确定性以及相关奖励模型，在分配生成预算之前预测混合奖励组。受控的评测框架实验支持Discord的有效性，而真实MCP实验则揭示了保留与成本之间的权衡。在三个Qwen3-VL模型规模上，BRACE超越了最强基线

    arXiv:2610.28785v1 Announce Type: cross  Abstract: Training tool-using language models requires costly interaction, yet identical rewards within a rollout group provide no group-relative learning signal. We use reward disagreement to guide harness evaluation and the allocation of rollouts for policy training in Model Context Protocol (MCP) environments. We first introduce Discord, which tests harness edits using exact paired inference and decision-preserving early stopping: only disagreeing paired outcomes favor one harness over another. This insight motivates prospective allocation in BRACE (Bank-calibrated Rollout Allocation under Correlated Episodes). BRACE combines historical task evidence, uncertainty in task success rates, and a correlated reward model to forecast mixed-reward groups before allocating generation budget. Controlled harness experiments favor Discord, while live MCP experiments expose a retention-cost trade-off. At three Qwen3-VL model scales, BRACE exceeds the stro
    
[^215]: 面向电力负荷预测的极端低状态量化Transformer

    Extreme Low-State Quantized Transformer for Electricity-Load Forecasting

    [https://arxiv.org/abs/2610.28784](https://arxiv.org/abs/2610.28784)

    该论文研究了极端低状态（3至16状态）权重量化对Transformer电力负荷预测模型的影响，发现量化感知微调（QAT-FT）可恢复约90%以上的量化精度损失，显著优于直接训练后量化（PTQ）。

    

    激进的低状态权重量化会大幅削弱神经网络的表示灵活性，使得在仅有少量离散权重状态可用时难以保持预测性能。我们针对基于Transformer的电力负荷预测任务研究了这一极端情形，在两种量化方案（仅前馈层量化、注意力加前馈层量化）下分别采用3状态、5状态、7状态和16状态权重量化。实验包含两项任务：基于前48小时数据预测未来1小时或未来24小时，并在两个共同主要指标的电力消费数据集上进行评估。结果显示，直接训练后量化（PTQ）相对于全精度模型，在数据集A和数据集B中使平均归一化均方根误差（NRMSE）分别增加6.25和7.04个百分点，而量化感知微调（QAT-FT）分别恢复了约90.2%和94.5%的性能损失。从随机初始化开始的量化感知训练（Scratch-QAT）也取得了具有竞争力的低状态量化性能。（摘要原文在此处截断）

    arXiv:2610.28784v1 Announce Type: new  Abstract: Aggressive low-state weight quantization can substantially reduce neural-network representational flexibility, making predictive performance difficult to preserve when only a few discrete weight states are available. We investigate this regime for Transformer-based electricity-load forecasting using 3-, 5-, 7-, and 16-state weight quantization across feed-forward-only and attention-plus-feed-forward quantization. Two tasks predict either the next hour or the next 24 hours from the preceding 48 hours across two co-primary electricity-consumption datasets. Direct post-training quantization (PTQ) increased mean normalized root-mean-square error (NRMSE) relative to full precision by 6.25 and 7.04 percentage points in Datasets A and B, whereas quantization-aware fine-tuning (QAT-FT) recovered approximately 90.2% and 94.5% of these penalties. Quantization-aware training from random initialization (Scratch-QAT) also produced competitive low-sta
    
[^216]: 三思而后行：在世界模型中通过比较性想象进行探索与规划

    Second Thoughts: Exploring and Planning through Comparative Imagination in World Models

    [https://arxiv.org/abs/2610.28779](https://arxiv.org/abs/2610.28779)

    提出SDI方法，利用基于行动者的共享参考标准，同时为世界模型中的探索信号和规划决策提供统一依据，使智能体能够自主判断执行计划动作还是遵循已学习的策略，且无需额外的分歧集成模型。

    

    世界模型使智能体能够在行动之前比较想象中的结果，但智能体仍需决定探索哪些备选方案，以及何时所提出的计划应当取代其已学习的策略。我们提出了SDI（想象中的同源分歧，Sibling Disagreement in Imagination），它通过一个共享的基于行动者的参考标准将这两项决策联系起来。SDI比较从同一模型状态想象出的动作序列，并利用行动者自身序列的重复展开来衡量固定行为下的变异性。该参考标准同时缩放候选回报之间的离散程度（提供探索信号）以及所提计划的预测优势（用于决定动作是否被接受）。当计划动作的优势超过参考变异性的固定倍数时，智能体执行该计划动作；否则，它遵循行动者的策略。SDI复用基础智能体的世界模型、行动者和评论家，无需额外的分歧集成模型。探索与规划共同引导训练数据的选择。

    arXiv:2610.28779v1 Announce Type: new  Abstract: World models let agents compare imagined outcomes before acting, but an agent must still decide which alternatives to explore and when a proposed plan should replace its learned policy. We introduce SDI (Sibling Disagreement in Imagination), which connects these decisions through a shared actor-based reference. SDI compares action sequences imagined from the same model state, using repeated rollouts of the actor's own sequence to measure variation under fixed behavior. This reference scales both the spread across candidate returns, which provides an exploration signal, and a proposed plan's predicted advantage, which governs action acceptance. The agent executes a planned action when its advantage exceeds the reference variability by a fixed factor; otherwise, it follows the actor. SDI reuses the base agent's world model, actor, and critic without requiring an additional disagreement ensemble. Exploration and planning guide training data
    
[^217]: 为大语言模型非结构化训练后剪枝构建嵌套LADDER

    Building a Nested LADDER for LLM Unstructured Post-Training Pruning

    [https://arxiv.org/abs/2610.28773](https://arxiv.org/abs/2610.28773)

    LADDER通过逐层增量式评估稀疏度分配，在每步优先剪除单位参数移除造成端到端退化最小的层，从而生成非均匀稀疏分布和多预算下精确嵌套的剪枝掩码，显著提升大语言模型训练后剪枝在高稀疏度下的鲁棒性。

    

    非均匀分配可以改善大语言模型的训练后剪枝效果，但在高稀疏度下模型质量仍可能急剧下降。为了控制随稀疏度增加而导致的性能退化，我们提出了LADDER（通过损伤驱动评估与细化的逐层分配方法，Layer-Wise Allocation via Damage-Driven Evaluation and Refinement），该方法以增量方式分配稀疏度。在每一步中，它评估从每个符合条件的层额外剪除一部分权重，并提交使每个被移除参数造成端到端性能退化最小的层更新。这一过程产生了非均匀的稀疏度分布，并在多个预算下产生精确嵌套的剪枝掩码。我们在五个OPT、Llama和Qwen模型上，在10-90%的稀疏度范围内评估LADDER，使用SparseGPT作为候选生成后端，并与均匀分配和自适应分配基线进行比较。结果揭示了一个明显的工作区间权衡：LADDER在良性稀疏度下表现相当或略弱，但在高稀疏度下变得显著更加稳健。

    arXiv:2610.28773v1 Announce Type: new  Abstract: Non-uniform allocation can improve post-training pruning of large language models, yet quality can still degrade sharply at high sparsity. To control degradation as sparsity increases, we introduce LADDER (Layer-Wise Allocation via Damage-Driven Evaluation and Refinement), which allocates sparsity incrementally. At each step, it evaluates pruning an additional fraction of weights from each eligible layer and commits the layer update that causes the least end-to-end degradation per removed parameter. This process produces non-uniform sparsity profiles and exactly nested masks across multiple budgets. We evaluate LADDER on five OPT, Llama, and Qwen models across 10-90% sparsity, using SparseGPT as its candidate-generation backend and comparing against uniform and adaptive allocation baselines. Results reveal a clear operating-regime trade-off: LADDER is comparable or slightly weaker at benign sparsity but becomes substantially more robust 
    
[^218]: 运动等变卷积LSTM：当运动成为一种规范对称性

    Motion Equivariant Convolutional LSTM: When Motion Becomes a Gauge Symmetry

    [https://arxiv.org/abs/2610.28771](https://arxiv.org/abs/2610.28771)

    提出运动等变卷积LSTM（MELSTM），将运动视为时间上局域化的规范对称性，通过估计时变速度而非枚举速度，使网络记忆能够随输入的运动而等变地移动。

    

    物体会移动，相机会平移，天气会随风飘移：序列数据中许多变化都是运动。运动改变的是事物的位置，而非事物本身，因此它是数据的一种对称性。然而，标准的循环网络对运动不具备等变性：当输入在移动时，它们的记忆却停留在原地。流等变循环网络首次将这种对称性引入序列模型，针对恒定速度的运动（即流），通过枚举一组速度来实现。我们在三个方向上扩展了这一思想：从简单循环网络扩展到卷积LSTM，从恒定速度流扩展到速度随时间变化的运动，以及从枚举速度扩展到估计速度。将序列视为定义在时间上的场，我们证明运动成为一种规范对称性：一种在时间上被局域化的全局对称性。基于这一观点，我们的运动等变卷积LSTM（MELSTM）将其速度模型作为规范场，并且可以证明其具有等变性……

    arXiv:2610.28771v1 Announce Type: new  Abstract: Objects move, cameras pan, and weather drifts with the wind: much of what changes in sequential data is motion. Motion changes where things are, not what they are, and is therefore a symmetry of the data. Yet standard recurrent networks are not equivariant to it: their memory stays put while the input moves on. Flow equivariant recurrent networks first brought this symmetry into sequence models, for motions of constant velocity, or flows, by enumerating a bank of velocities. We extend this idea in three directions: from simple recurrent networks to convolutional LSTMs, from flows to motions whose velocity changes in time, and from enumerating velocities to estimating them. Treating a sequence as a field over time, we show that motion becomes a gauge symmetry: a global symmetry made local in time. Built on this view, our motion equivariant convolutional LSTM (MELSTM) uses its velocity model as the gauge field and is provably equivariant t
    
[^219]: ReAPO：面向大语言模型推理高效探索的重试感知策略优化

    ReAPO: Retry-Aware Policy Optimization for Efficient Exploration in LLM Reasoning

    [https://arxiv.org/abs/2610.28770](https://arxiv.org/abs/2610.28770)

    提出基于随机最短路径框架的信用分配方法 ReAPO，通过为失败轨迹分配额外生成预算来实现重试感知的策略优化，从而提升大语言模型推理中探索的效率。

    

    RLVR（可验证奖励强化学习）提升了语言模型的推理能力，但其探索通常依赖独立的随机采样，这会导致反复采样到相似的失败策略。已有工作通过在后续尝试的上下文中包含错误解法，在采样过程中引入依赖关系来引导探索，并取得了成功。然而，这类方法引入了一个额外的问题：如何为每次尝试分配信用。我们提出从随机最短路径的视角来审视信用分配问题——这是经典强化学习中的一个框架，旨在最小化到达终止状态（在我们的场景中即经过验证的成功回答）的期望累积成本。我们证明，在适当的简化假设下，这一视角自然地引出了对 GRPO 的一个简单修改，我们称之为重试感知策略优化。ReAPO 会为失败的轨迹分配额外的生成资源……

    arXiv:2610.28770v1 Announce Type: new  Abstract: RLVR improves language-model reasoning, but commonly relies on independent stochastic rollouts for exploration, which can repeatedly sample similar failed strategies. Prior work has seen success with approaches that guide exploration by introducing dependence into the sampling process by including incorrect solutions in the context for subsequent attempts. Still, these methods introduce the additional question of how to assign credit to each attempt. We propose to view the credit assignment through the lens of Stochastic Shortest Paths, a framework in classical reinforcement learning that seeks to minimize the expected cumulative cost to reach a terminal state (in our case, a verified successful response). We show that, under suitable simplifying assumptions, this perspective naturally leads to a simple modification to GRPO, which we call Retry-Aware Policy Optimization (ReAPO). ReAPO allocates additional generation to failed trajectorie
    
[^220]: SkillEval：通过认知诊断模型学习大语言模型的可解释能力画像

    SkillEval: Learning Interpretable Ability Profiles of LLMs via Cognitive Diagnosis Models

    [https://arxiv.org/abs/2610.28767](https://arxiv.org/abs/2610.28767)

    SkillEval 结合大语言模型的元认知能力与心理测量学中的认知诊断模型，从基准题目中自动发现可解释的技能，并为大语言模型生成跨模型的技能级能力画像，相比传统平均分评估更可靠、更校准且信息更丰富。

    

    当前对大语言模型在基准测试上的评估通常将单个样本上的表现汇总为一个标量分数。这种方法过于粗糙，忽略了基准数据集所涵盖的多样化技能范围。为了解决这一局限并更好地利用基准数据，我们提出了 SkillEval，一个全新的评估框架，能够进行跨模型、技能级别的分析。该框架整合了两个领域的进展：（1）利用大语言模型的元认知能力，从题目中自动发现可解释的技能；（2）应用心理测量学中的认知诊断模型（CDMs）来估计大语言模型在每项技能上的能力。我们在五个覆盖不同领域的基准（MATH、BBH、GPQA、MuSR、IFEval）上，对 3,811 个大语言模型在 9,523 个题目上进行了实验。与传统的基于平均分的评估范式相比，SkillEval 产生了可靠、更校准、信息更丰富且简洁的结果。

    arXiv:2610.28767v1 Announce Type: new  Abstract: Current evaluation of LLMs on benchmarks typically aggregates performance over individual examples into a single scalar score. This approach is overly coarse and ignores the diverse scope of skills covered by benchmark datasets. To address this limitation and better exploit benchmark data, we propose SkillEval, a new evaluation framework that performs cross-model, skill-level analysis. The framework integrates advances from two areas: (1) it leverages the meta-cognitive capabilities of LLMs to automatically discover interpretable skills from questions, and (2) it applies cognitive diagnosis models (CDMs) from psychometrics to estimate per-skill abilities of LLMs. We conduct experiments on 3,811 LLMs evaluated on 9,523 items across five benchmarks spanning diverse domains (MATH, BBH, GPQA, MuSR, IFEval). Compared to the traditional average-based evaluation paradigm, SkillEval produces reliable, more calibrated and informative yet concise 
    
[^221]: 面向科学的机器学习自动化：前沿智能体能否在真实环境中攀登科学之峰？

    Automating ML for Science: Can Frontier Agents Climb Scientific Hills in the Wild?

    [https://arxiv.org/abs/2610.28763](https://arxiv.org/abs/2610.28763)

    该论文提出了长时程测试平台 Agent4Sci（涵盖 11 个领域的 20 个科学机器学习基准），发现前沿智能体能在所有基准上超越已发表的最佳模型（Claude Opus 平均相对改进 10.4%），但这些进展几乎完全来自机器学习工程技巧而非研究级的科学洞察。

    

    前沿编码智能体日益被定位为端到端的自动化研究者，而“面向科学的机器学习”为检验这一承诺提供了一个高要求的试金石：改进一个科学模型往往需要超越常规机器学习工程的领域洞察。我们提出了 Agent4Sci，一个长时程测试平台，用于衡量当今的智能体能在多大程度上改进科学机器学习基准上已发表的最佳模型，涵盖 11 个领域的 20 个基准。Agent4Sci 并非通过排行榜为智能体排名，而是探讨智能体在真实条件下能在这些任务上取得哪些类型的进展，以及它们在何处存在不足。实验表明，当今的前沿智能体在所有基准上都推进了已发表的最佳结果，其中 Claude Opus 达到了 10.4% 的平均相对改进，且没有任何一次运行被标记为学术不端。然而，这些提升几乎完全来自一系列机器学习工程技巧，而非研究层面的洞察。轨迹分析揭示了三个反复出现的局限……（摘要原文在此处被截断）

    arXiv:2610.28763v1 Announce Type: cross  Abstract: Frontier coding agents are increasingly framed as end-to-end automated researchers, and ML for science offers a demanding test of that promise: improving a scientific model often requires domain insight beyond routine ML engineering. We introduce Agent4Sci, a long-horizon testbed for measuring how far today's agents can improve the published best models on scientific ML benchmarks, spanning 20 benchmarks across 11 domains. Rather than ranking agents on a leaderboard, Agent4Sci asks what kinds of progress agents can make on these tasks under in-the-wild conditions, and where they fall short. Empirically, today's frontier agents advance the published best on every benchmark, with Claude Opus reaching 10.4% mean relative improvement and no run flagged for academic misconduct. These gains, however, come almost entirely from a bag of ML-engineering tricks rather than research-level insight. Trajectory analysis surfaces three recurring limit
    
[^222]: GPU加速的Gabriel图构建：并行化策略及其在大间隔分类与基于图的聚类中的应用

    GPU-Accelerated Construction of the Gabriel Graph: Parallelization Strategies and Applications to Large-Margin Classification and Graph-Based Clustering

    [https://arxiv.org/abs/2610.28759](https://arxiv.org/abs/2610.28759)

    本文提出一组基于CUDA的GPU并行化策略（包括从线程索引到顶点对的隐式闭式映射以及基于位掩码的原子压缩边列表构建），突破了Gabriel图朴素构建O(N³)的复杂度瓶颈，使其能够扩展到大规模数据集，并应用于大间隔分类和基于图的密度感知聚类。

    

    Gabriel图是一种在计算几何和几何机器学习中被广泛使用的邻近图，它是大间隔分类器的基础，并且正如本工作所展示的，也是密度感知聚类方法的基础。然而，其朴素构建方法具有三次方时间复杂度O(N³)，这严重限制了它在超过数千个样本的数据集上的应用。本文提出了一组基于CUDA实现的GPU并行化策略，使Gabriel图的大规模构建切实可行。我们引入了：(i) 一种从线性线程索引到对应顶点对(i,j)的隐式闭式映射，无需显式的O(N²)邻接结构，并使每个线程能够在常数时间和常数内存内定位其顶点对；(ii) 一种基于位掩码、两遍扫描、原子压缩的边列表构建方法，生成仅与实际找到的边数成比例的稠密边数组；以及(iii) 一……（摘要在此处截断）

    arXiv:2610.28759v1 Announce Type: cross  Abstract: The Gabriel graph is a proximity graph widely used in computational geometry and geometric machine learning, underlying large-margin classifiers and, as we show in this work, density-aware clustering methods. Its construction, however, has cubic time complexity $O(N^3)$ in the naive formulation, which severely limits its use on datasets beyond a few thousand samples. This paper presents a set of GPU parallelization strategies, implemented in CUDA, that make Gabriel graph construction practical at scale. We introduce (i) an implicit closed-form mapping from a linear thread index to the corresponding vertex pair $(i,j)$, which removes the need for an explicit $O(N^2)$ adjacency structure and lets every thread locate its pair in constant time and constant memory; (ii) a two-pass, bitmask-based, atomically-compacted edge-list construction that produces a dense edge array proportional only to the number of edges actually found; and (iii) a 
    
[^223]: 理解限价订单簿状态空间基础模型的内部工作机制

    Understanding the Inner Workings of a State-Space Foundation Model for Limit Order Books

    [https://arxiv.org/abs/2610.28753](https://arxiv.org/abs/2610.28753)

    本研究揭示了8.2M参数的Mamba-3状态空间基础模型在限价订单簿消息流上自回归训练后，内部学会了市场及其动态的通用表示，其中订单流不平衡这一短期价格波动核心驱动因素可被以R²≈0.59的高精度解码，且该表示可泛化至未见过的股票。

    

    自回归基础模型通过生成限价订单簿（LOB）的消息流，提供了最先进的市场模拟能力。然而，在不了解其内部工作机制的情况下，研究方向难以收窄，其在真实交易环境中的实际应用也仍存在不确定性。因此，本研究探究了这些模型究竟从市场微观结构中学到了什么。我们发现，类似于序列模型在游戏中构建对棋盘的内部表示，一个在八只美国大盘股的标记化消息流上进行自回归训练的8.2M参数Mamba-3状态空间模型，学会了市场及其动态的内部表示。特别是，订单流不平衡（短期价格波动的核心驱动因素）能够以 $R^2 \approx 0.59$ 的精度被解码，远高于对照组和基线模型。这种表示还能扩展到未见过的股票，表明这些模型学到了通用的表示。

    arXiv:2610.28753v1 Announce Type: cross  Abstract: Autoregressive foundation models offer state-of-the-art capabilities to simulate the market by generating the message flow of limit order books (LOBs). Yet, without understanding their inner workings, research directions are hard to narrow and practical use in real-world trading environments remains uncertain. This study therefore asks what these models actually learn about market microstructure. We find that, similarly to games where sequence models build an internal representation of the board, an 8.2M-parameter Mamba-3 state-space model trained autoregressively on the tokenized message stream of eight large-cap US equities learns an internal representation of the market and its dynamics. In particular, order-flow imbalance, a central driver of short-horizon price moves, is decodable at $R^2 \approx 0.59$, far above controls and baselines. This representation extends to unseen stocks, suggesting that these models learn general repres
    
[^224]: 逆向戈德定理：基于基底受限对比学习器的正数据语言识别

    A Reverse-Gold Theorem: Positive-Data Identification by Substrate-Bounded Contrastive Learners

    [https://arxiv.org/abs/2610.28747](https://arxiv.org/abs/2610.28747)

    该论文证明了一个“逆向戈德定理”：借助先天递归对比算子自我生成结构受限的负例扰动，并以InfoNCE式对比学习训练的受限评分器，在忠实性条件下可仅凭正数据实现高概率的PAC条件语言识别，且样本复杂度仅为多项式级。

    

    戈德定理表明，任何学习器都无法仅凭正例呈现来识别超有限类中的任意语言。我们证明了一个形态不同的正面结果。固定一个先天的“递归对比算子”$R$，它将任意字符串映射到其结构上受限的有限扰动集合；再考虑一个容量受限的判别式评分器，它通过在自我生成的扰动上进行InfoNCE式对比估计训练。在算子、评分器与目标语言之间满足忠实性条件的前提下，训练后的评分器能够仅凭正数据实现“PAC条件识别”：以高概率，它会把除$\varepsilon$比例之外的所有正例都排在其自身轨道中每个语言外扰动之上，且样本复杂度关于模型容量、轨道大小、分数界限和$1/\varepsilon$呈多项式级。负例由学习器自行构造，而非由外部提供……

    arXiv:2610.28747v1 Announce Type: cross  Abstract: Gold's theorem shows that no learner can identify an arbitrary language in a superfinite class from positive presentations alone. We prove a positive result of a different shape. Fix an innate \emph{recursive contrast operator} $R$ mapping any string to a finite set of structurally bounded perturbations of it, and a discriminative scorer of bounded capacity trained by InfoNCE-style contrastive estimation \citep{vandenoord2018,macollins2018} on self-generated perturbations. Under a faithfulness condition relating operator, scorer, and target, the trained scorer achieves \emph{PAC conditional identification} from positive data alone: with high probability it ranks all but an $\varepsilon$-fraction of positives above every out-of-language perturbation in their own orbit, with sample complexity polynomial in the capacity, the orbit size, the score bound, and $1/\varepsilon$. The negatives are constructed by the learner, not supplied by an 
    
[^225]: 以人为本环境中的离线学徒学习

    Offline Apprenticeship Learning for Human-Centric Environments

    [https://arxiv.org/abs/2610.28716](https://arxiv.org/abs/2610.28716)

    本论文提出一系列离线学徒学习框架（包括 EM-EDM、THEMES、HALIDE），通过解决奖励建模与演示质量两大挑战，在不与环境交互的情况下从专家演示中学习适用于智能辅导系统和临床医疗等以人为本环境的自适应策略。

    

    在智能辅导系统（ITSs）和临床医疗等以人为本的环境中，有效的决策需要具备自适应性、数据高效性和伦理可部署性的策略。学徒学习（AL）通过恢复潜在的奖励结构，直接从专家演示中推断策略，从而实现无需环境交互的严格离线学习。本论文围绕两大挑战——奖励建模与演示质量——构建了一系列递进的离线学徒学习框架。在奖励建模方面，EM-EDM 通过潜在奖励函数对演示进行聚类并归纳出特定于各聚类的策略，而 THEMES 在分层学习的奖励调节器引导下将轨迹划分为时间上连贯的子轨迹；在一个教育数据集和三个临床数据集上，THEMES 的表现优于扁平型和分层型的学徒学习基线。在演示质量方面，HALIDE 整合了连续的（摘要在此处截断）

    arXiv:2610.28716v1 Announce Type: new  Abstract: Effective decision-making in human-centric environments such as Intelligent Tutoring Systems (ITSs) and clinical healthcare requires policies that are adaptive, data-efficient, and ethically deployable. Apprenticeship Learning (AL) infers policies directly from expert demonstrations by recovering the underlying reward structure, enabling strictly offline learning without environment interaction. This dissertation develops a progression of offline AL frameworks organized around two challenges: reward modeling and demonstration quality. For reward modeling, EM-EDM clusters demonstrations by latent reward functions and induces cluster-specific policies, and THEMES partitions trajectories into temporally coherent sub-trajectories guided by a hierarchically learned reward regulator; across one educational and three clinical datasets, THEMES outperforms flat and hierarchical AL baselines. For demonstration quality, HALIDE integrates continuous
    
[^226]: 高调宣称，微小差距：GraphLLM报告的性能提升能否经受简单基线的检验？

    Big Claims, Small Margins: Do Reported GraphLLM Gains Survive Simple Baselines?

    [https://arxiv.org/abs/2610.28714](https://arxiv.org/abs/2610.28714)

    该研究发现简单基线（Correct & Smooth和MLP）在文本属性图节点分类上与最先进GraphLLM方法的差距中位数仅约2-3个百分点，并揭示了部分已发表研究存在测试集选择超参数、测试标签泄露等评估风险。

    

    基于大语言模型（LLM）的方法在文本属性图（TAG）节点分类任务上经常报告最先进的准确率。我们探究这些结果与在相同数据划分上运行的低成本基线相比表现如何。在三篇近期研究的数据划分上，我们仅使用验证集进行选择来调优四个基线代表方法，并将它们与各论文在匹配设置下为基于LLM方法报告的最高平均性能进行比较。基于TF-IDF特征的Correct & Smooth方法与该参考值的中位数差距为2.3个百分点，基于冻结e5-large-v2嵌入的MLP差距为2.8个百分点。在24个任务中的15个上，Correct & Smooth与参考值的差距在3个百分点以内或更优，但在22个可测试任务中的12个上参考值显著更高，因此若干已报告的性能提升仍然可观。通过对照论文阅读其公开代码，我们还记录了评估风险：在测试集上选择超参数或检查点、测试池节点的标签被用于训练。

    arXiv:2610.28714v1 Announce Type: new  Abstract: LLM-based methods for node classification on text-attributed graphs (TAGs) routinely report state-of-the-art accuracy. We ask how these results compare with inexpensive baselines run on the same splits. On the splits of three recent studies, we tune four baseline representatives with validation-only selection and compare them with the highest mean that each paper reports for an LLM-based method in the matched setting. Correct & Smooth on TF-IDF features trails this reference by a median of 2.3 percentage points, and an MLP on frozen e5-large-v2 embeddings by 2.8 points. Correct & Smooth is within 3 points of the reference or better on 15 of 24 tasks, but the reference is significantly higher on 12 of 22 testable tasks, so several reported gains remain substantial. Reading released code next to the papers, we also document evaluation risks: selection of hyperparameters or checkpoints on the test set, labels of test-pool nodes used in trai
    
[^227]: STAR-XPC：用于估算聚合物复合材料X射线透射率的集合Transformer

    STAR-XPC: Set Transformer for Estimating X-ray Transmission in Polymer Composites

    [https://arxiv.org/abs/2610.28709](https://arxiv.org/abs/2610.28709)

    本文提出基于集合Transformer的STAR-XPC模型，可在1-3毫秒内快速准确估算聚合物复合材料的X射线透射比，替代耗时超过5小时的蒙特卡罗模拟。

    

    在辐射屏蔽研究中，评估X射线束穿过聚合物复合材料的能量透射比是一项具有挑战性的任务。作为参考解的蒙特卡罗输运模拟方法精度很高，但计算成本极其昂贵：我们的每个参考模拟被限制在40亿（4×10⁹）光子历史数和20,000秒（超过5小时）的时间上限内，而闭式解析方法又无法表征如此复杂的场景。我们提出了一种人工神经网络模型，将分子组分的无序多重集、其质量分数以及四个整体物理描述符直接映射到透射比 I(t)/I₀，推理时间仅需1-3毫秒。该模型是一个集合Transformer（Set Transformer），在结构上具有置换不变性，并通过固定大小的填充集合接受最多含11种组分的不同规模混合物。每种组分均使用化学领域预训练的Transformer语言模型ChemBE进行嵌入……

    arXiv:2610.28709v1 Announce Type: new  Abstract: Evaluating the energy transmission ratio of an X-ray beam through a polymer composite is a challenging task in radiation shielding studies. The reference solution with Monte Carlo transport simulation is highly accurate but computationally expensive: each of our reference simulations is capped at $4\times10^9$ photon histories and a 20,000 s (over 5 hours) time limit, while closed-form analytical methods fail to represent such complex scenarios. We present an artificial neural network model that maps the unordered multi-set of molecular constituents, their mass fractions, and four bulk physical descriptors directly to the transmission ratio $I(t)/I_0$ in 1-3 milliseconds. The model is a Set Transformer, which makes it permutation invariant by construction and accepts mixtures of different sizes up to 11 constituents within a fixed-size padded set. Each constituent is embedded with a chemistry-pretrained transformer language model, ChemBE
    
[^228]: 跨门店借力：基于选择模型的收益管理迁移学习

    Borrowing Strength Across Stores: Transfer Learning for Choice-Based Revenue Management

    [https://arxiv.org/abs/2610.28708](https://arxiv.org/abs/2610.28708)

    该论文提出了一种面向MNL需求模型的收缩估计量，通过迁移学习在门店本地需求估计与跨门店信息共享之间取得平衡，从而在本地数据有限的情况下提升零售收益管理中品类与定价决策的可靠性。

    

    大型零售连锁企业必须根据当地顾客偏好来调整品类和定价决策，然而即使是拥有大量销售记录的零售商，在估计门店或本地市场层面的需求时，数据往往有限，尤其是在销售周期较短的产品类别中。因此，本地估计（为每个门店单独拟合需求模型）可能不可靠，而跨门店汇集数据虽能提高估计精度，却以忽略本地需求异质性为代价。我们研究了零售商如何在多项Logit（MNL）需求模型下通过迁移学习来平衡这两种相互冲突的效应。我们提出了一种收缩估计量，将门店特定的需求估计与从其他门店汇集的信息相结合。与运营管理领域现有信息传递理论文献所基于的线性或二次设定不同，非线性MNL似然函数无法直接……（摘要在此处截断）

    arXiv:2610.28708v1 Announce Type: cross  Abstract: Large retail chains must tailor assortment and pricing decisions to local customer preferences, yet even retailers with extensive sales records often have limited data for estimating demand at the store or local market level, particularly in product categories with short selling windows. Local estimation (fitting a demand model for each store on its own) can therefore be unreliable, while pooling data across stores improves estimation precision at the cost of ignoring local demand heterogeneity. We study how retailers can balance these competing effects through transfer learning under a multinomial logit (MNL) demand model. We propose a shrinkage estimator that combines store-specific demand estimates with information pooled from other stores. Unlike the linear or quadratic settings underlying much of the existing theoretical literature on information transfer in operations, the nonlinear MNL likelihood does not admit a straightforward
    
[^229]: SimHack：在强化学习训练开始前预测可检测的奖励破解行为的出现时机

    SimHack: Predicting the onset of detectable reward hacking before RL training

    [https://arxiv.org/abs/2610.28707](https://arxiv.org/abs/2610.28707)

    本文提出SimHack模拟器，仅凭少量小规模参考运行即可在强化学习训练开始前预测奖励破解行为何时出现，从而帮助提前选择风险更低的训练方案。

    

    在使用强化学习训练大语言模型进行编程时，奖励破解行为通常是可以被检测到的，但却难以预防，原因在于： 随着训练环境日益复杂，对环境进行加固变得越来越困难； 训练时的干预措施反而会激励模型隐藏其破解行为以绕过检测。因此，我们转而研究是否可以在执行强化学习运行之前，预测可检测的奖励破解行为何时会成为主导，从而帮助我们提前选择风险更低的训练方案。我们发现，仅凭初始破解率并不能决定其出现时机——不同的训练方案和多次重复运行之间，出现时机的差异很大。为此，我们将该问题形式化为：预测在给定目标训练方案下，奖励破解行为出现步骤的分布。我们提出了SimHack，这是一个模拟器，它从少量小规模的参考强化学习运行中学习奖励破解行为的训练动态，并针对未见过的目标训练方案生成反事实模拟运行……

    arXiv:2610.28707v1 Announce Type: new  Abstract: When training LLMs to code with reinforcement learning, reward hacking is often detectable but difficult to prevent, since (a) environment hardening is increasingly difficult as training environments grow more complex; and (b) training-time interventions incentivize models to obfuscate hacking behavior to bypass detection.   We instead study whether we can predict when detectable reward hacking becomes dominant before executing an RL run, enabling us to select lower-risk recipes in advance. We find that the initial hacking rate alone does not determine onset, which varies substantially across training recipes and repeated runs. We thus formulate this problem as predicting the distribution of the reward hacking onset step under a target training recipe.   We introduce SimHack, a simulator that learns reward hacking training dynamics from a handful of smaller-scale reference RL runs and generates counterfactual simulation runs of unseen ta
    
[^230]: 可扩展的应力约束拓扑优化：一种基于机器学习的超分辨率与同步优化方法

    Scalable Stress-Constrained Topology Optimization: A Super-resolution and Simultaneous Approach Based on Machine Learning

    [https://arxiv.org/abs/2610.28703](https://arxiv.org/abs/2610.28703)

    提出同步神经拓扑优化（SNTO）框架，利用自定义神经网络在单一优化循环中联合表示并优化状态、伴随和设计变量，配合独特的网络架构与硬件感知训练方案，实现了面向数百万自由度的可扩展应力约束拓扑优化。

    

    arXiv:2610.28703v1 公告类型：新论文 摘要：大规模拓扑优化（TO）在计算上仍然非常耗费资源，尤其是对于应力约束问题。为应对这一挑战，我们提出了同步神经拓扑优化（SNTO）框架，其中我们使用自定义神经网络（NN）来表示状态变量、伴随变量和设计变量，这些网络的参数在利用连续伴随分析和自动微分的单一优化循环中进行联合优化。该优化最小化一个多组分损失函数，该函数结合了设计目标、局部应力约束的聚合以及势能泛函。我们的神经网络具有独特的架构，能够加速收敛，同时使网络能够捕捉对应力约束拓扑优化至关重要的尖锐梯度和边界。为实现向数百万自由度（DoF）的可扩展性，我们开发了一种硬件感知训练方案，其中设计域（摘要原文在此处截断）

    arXiv:2610.28703v1 Announce Type: new  Abstract: Large-scale topology optimization (TO) remains computationally demanding, especially for stress-constrained problems. To address this challenge, we propose the simultaneous neural topology optimization (SNTO) framework where we represent state, adjoint, and design variables with custom neural networks (NNs) whose parameters are jointly optimized within a single optimization loop that leverages continuous adjoint analysis and automatic differentiation. This optimization minimizes a multi-component loss function that combines the design objective, an aggregation of the local stress constraints, and the potential energy functionals. Our NNs have a unique architecture that accelerates convergence while enabling the networks to capture sharp gradients and boundaries which are critical to stress-constrained TO. To achieve scalability to millions of degrees of freedom (DoF), we develop a hardware-aware training scheme in which the design domain
    
[^231]: 面向FPGA与集成电路平台的强化学习硬件加速与系统级优化：综述

    Reinforcement learning for hardware acceleration and system-level optimization on FPGA and integrated-circuit platforms: A survey

    [https://arxiv.org/abs/2610.28699](https://arxiv.org/abs/2610.28699)

    本综述系统梳理了100余项在FPGA和集成电路平台上实现强化学习的研究，按算法类别、抽象层次、硬件实现策略和应用领域进行分类，总结了强化学习硬件加速与系统级优化的架构与实现方法。

    

    强化学习（RL）能够在嵌入式、边缘和异构计算系统中实现自适应与自主决策。现场可编程门阵列（FPGA）和集成电路（IC）为强化学习部署提供了低延迟、高能效和架构灵活性，但同时也带来了资源受限、实现复杂度以及硬件-算法协同设计方面的挑战。本综述调研了100多项关于FPGA与IC平台上强化学习实现的研究，重点关注架构、实现策略、性能评估和应用。相关文献按强化学习类别进行组织，包括基于模型的马尔可夫决策过程、经典的基于值的方法、深度强化学习、actor-critic、策略梯度以及多智能体方法。我们进一步按抽象层次、硬件实现策略和应用领域对研究进行分类，考察强化学习算法如何映射到可重构与固定硬件之上。

    arXiv:2610.28699v1 Announce Type: new  Abstract: Reinforcement learning (RL) enables adaptive and autonomous decision-making in embedded, edge, and heterogeneous computing systems. Field-programmable gate arrays (FPGAs) and integrated circuits (ICs) offer low latency, energy efficiency, and architectural flexibility for RL deployment, while introducing challenges in resource constraints, implementation complexity, and hardware-algorithm co-design. This review surveys more than 100 studies on RL implementations across FPGA and IC platforms, emphasizing architectures, implementation strategies, performance evaluation, and applications. The literature is organized by RL family, including model-based Markov decision processes, classical value-based methods, deep RL, actor-critic, policy-gradient, and multi-agent approaches. We further classify studies by abstraction level, hardware realization strategy, and application domain, examining how RL algorithms map onto reconfigurable and fixed h
    
[^232]: 游戏中的规模化：面向多样化虚拟世界中具身智能体的持续预训练

    Scaling in Games: Continued Pre-Training For Embodied Agents In Diverse Virtual Worlds

    [https://arxiv.org/abs/2610.28698](https://arxiv.org/abs/2610.28698)

    本文通过涵盖模型规模、数据量与数据多样性的120个训练配置，系统研究了持续预训练在多样化虚拟世界中的规模化规律，发现更多数据提升域内性能、数据多样性促进向相关世界的迁移、更大的模型规模带来性能提升。

    

    摘要：在物理环境中，扩展数据、多样性和模型容量已经改善了具身学习，因为多样化的经验始终建立在共享的物理定律之上。然而，这些收益能否延伸到具有不同视觉风格、动态特性、控制方式和目标的虚拟世界，仍不清楚。我们通过对持续预训练（CPT）在三个维度上的受控研究来探究这一问题：数据量、多样性和模型规模。我们的120个训练配置涵盖了从0.8B到27B的四种模型规模、从50小时到1,600小时的六种数据预算，以及从仅Minecraft数据到多样化多世界语料的五种数据混合方式。我们使用三级验证损失以及从语料库内世界到未见世界的游戏玩法基准来评估性能与泛化能力。我们的结果表明：（1）更多数据可提升语料库内性能，多样性可改善向相关世界的迁移但在差异较大的世界中收益有限，而模型规模的增大可以提升……

    arXiv:2610.28698v1 Announce Type: cross  Abstract: Scaling data, diversity and model capacity has improved embodied learning in physical environments, where diverse experiences remain grounded in shared physical laws. Whether these gains extend to virtual worlds with different visual styles, dynamics, controls, and objectives remains unclear. We investigate this question through a controlled study of continued pre-training (CPT) across three axes: data volume, diversity, and model size. Our 120 training configurations span four model sizes from 0.8B to 27B, six data budgets from 50 to 1,600 hours, and five mixtures ranging from Minecraft-only data to diverse multi-world corpora. We assess performance and generalization using three-level validation losses and gameplay benchmarks from in-corpus worlds to unseen worlds. Our results show: (1) more data improves in-corpus performance, diversity improves transfer to related worlds with limited gains in distant worlds, and model size improves
    
[^233]: 面向参数高效的基于光电容积脉搏波（PPG）心率估计的交互门控时频学习

    Interaction-Gated Time-Frequency Learning for Parameter-Efficient PPG-Based Heart Rate Estimation

    [https://arxiv.org/abs/2610.28693](https://arxiv.org/abs/2610.28693)

    本文提出TF-IG框架，通过参数高效的时域-频域双分支架构以及显式建模特征一致性与差异性的交互门控融合模块，实现了基于PPG的准确心率估计。

    

    光电容积脉搏波描记法（PPG）能够实现低成本、无创的心率（HR）监测，但其对波形失真、生理变异性以及测量干扰仍然较为敏感。时域波形特征与频域周期性信息为心率估计提供了互补信息；然而，如何在一个紧凑的模型中有效探索并整合这些表征仍然是一个挑战。本文提出了TF-IG，一个用于基于PPG心率估计的参数高效交互门控时频学习框架。TF-IG采用双分支架构：时域分支通过膨胀深度可分离卷积捕获脉搏形态信息，频域分支则从有界对数频谱图表示中提取心脏周期性信息。为了融合这两个域的信息，我们引入了一个交互门控融合模块，该模块显式地建模特征之间的一致性与差异性，同时还……（原文摘要在此处截断）

    arXiv:2610.28693v1 Announce Type: new  Abstract: Photoplethysmography (PPG) enables low-cost and non-invasive heart rate (HR) monitoring but remains sensitive to waveform distortion, physiological variability, and measurement interference. Temporal waveform characteristics and frequency-domain periodicity provide complementary information for HR estimation; however, effectively exploring and integrating these representations within a compact model remains challenging. This paper proposes TF-IG, a parameter-efficient interaction-gated time-frequency learning framework for PPG-based HR estimation. TF-IG employs a dual-branch architecture in which a temporal branch captures pulse morphology via dilated depthwise-separable convolutions, while a frequency branch extracts cardiac periodicity information from bounded log-spectrogram representations. To combine the two domains, we introduce an interaction-gated fusion module that explicitly models feature agreement and discrepancy, while also 
    
[^234]: 太阳系表面技术签名：在地外地形上搜寻人工设计的迹象

    Solar System Surface Technosignatures: Searching for Signs of Artificial Design on Extraterrestrial Terrain

    [https://arxiv.org/abs/2610.28687](https://arxiv.org/abs/2610.28687)

    该论文提出将技术签名搜索从人工射电信号扩展到行星表面的非随机人工图案，并建议结合高分辨率影像数据与机器学习异常检测方法，在月球和火星等地质不活跃天体上寻找可能存留数亿年的人工设计痕迹。

    

    人工射电信号历来是技术签名搜寻的主要焦点。在此，我们运用人类学视角将搜寻范围扩展至表面技术签名：即行星地形上非随机的、有意为之的图案，类似于地球上持久存在的人造景观和纪念物。在月球和火星等地质不活跃的天体上，人工痕迹和图案可能保持可见达数亿年之久。我们讨论了使这一搜寻恰逢其时的两项最新进展：高分辨率成像数据集（例如火星勘测轨道飞行器（MRO）上HiRISE相机产生的数据），以及基于机器学习的异常检测方法。最后，我们列出了开发能够搜寻这些表面技术签名模型的两种可能方法：监督学习和无监督学习。结合激光雷达、探地雷达和反射光谱学，这些方法将……

    arXiv:2610.28687v1 Announce Type: cross  Abstract: Artificial radio emissions have historically been a major focus of technosignature searches. Here, we apply an anthropological lens to expand this scope to surface technosignatures: non-random, intentional patterning on planetary terrain, similar to long-lasting human-made landscapes and monuments on Earth. On geologically inactive bodies such as the Moon and Mars, artificial traces and patterns could potentially remain visible for hundreds of millions of years. We discuss two recent developments that make this search timely: high-resolution imaging datasets, such as those produced by the HiRISE camera on the Mars Reconnaissance Orbiter (MRO), and ML-based anomaly-detection methods. Finally, we list two possible approaches to developing a model capable of searching for these surface technosignatures: a supervised and an unsupervised one. Complemented by LiDAR, ground-penetrating radar, and reflectance spectroscopy, these methods would 
    
[^235]: Git未记录的内容：两个AI辅助研究仓库中的溯源可读性

    What Git Does Not Record: Provenance Legibility in Two AI-Assisted Research Repositories

    [https://arxiv.org/abs/2610.28655](https://arxiv.org/abs/2610.28655)

    本研究通过审计同一作者带有不同类型AI智能体残留痕迹的两个研究仓库，发现代码仓库在披露AI参与科研时的溯源可读性呈反向差异——规范性规则痕迹保留85.6%的结论性行但丢失过程片段，叙事性草稿痕迹记录过程片段却仅存21.8%——从而揭示了Git仓库无法完整回答“AI如何参与研究”这一溯源问题的根本局限。

    

    出版商和资助机构越来越多地要求作者披露AI工具在一项研究中发挥了何种作用。在计算类研究工作中，代码仓库正是作者或审计者寻求答案的地方，而本研究追问的是：仓库实际上能够提供什么。我们审计了同一作者在同一领域开展的两个单人研究仓库，每个仓库承载着不同类型的智能体残留痕迹：其一是常设规则文件构成的规范性痕迹，其二是持续运行的智能体草稿本构成的叙事性痕迹，两个仓库分别包含466次和103次提交。两者的可读性呈现相反的特征：规范性痕迹中曾经提交过的代码行有85.6%在当前修订版本中仍然可见，且常设规则只陈述结论而丢弃其背后的过程片段；叙事性痕迹记录了过程片段，但可见比例仅为21.8%。若按修订版本计数，差距则小得多：规范性痕迹的41个修订版本中有15个不留任何可见行，而叙事性痕迹的41个中有18个。

    arXiv:2610.28655v1 Announce Type: cross  Abstract: Publishers and funders increasingly ask authors to disclose how AI tools contributed to a piece of research. In computational work the repository is where an author or an auditor would look for the answer, and we ask what it can actually supply. We audit two single-investigator repositories by the same author in the same field, each carrying a different class of agent residue: a prescriptive trace of standing rule files and a narrative trace of a running agent scratchpad, in repositories of 466 and 103 commits. The two are legible in opposite ways. 85.6% of the lines ever committed to the prescriptive trace are visible at the current revision, and a standing rule states a conclusion and discards the episode behind it; the narrative trace records episodes and shows 21.8%. Counted by revision the gap is far smaller: 15 of the prescriptive trace's 41 revisions leave no visible line, against 18 of the narrative trace's 41. The Co-Authored-
    
[^236]: 保留还是丢弃：在解耦式LLM服务系统中学习KV缓存保留策略

    To Keep or Not to Keep: Learning KV Cache Retention in Disaggregated LLM Serving Systems

    [https://arxiv.org/abs/2610.28637](https://arxiv.org/abs/2610.28637)

    提出了KVLearn框架，通过轻量级前缀复用预测器学习复用概率，并结合重计算、传输和存储成本进行成本感知的KV缓存保留决策，同时节省解耦式LLM服务中的内存与带宽。

    

    解耦式LLM服务将预填充与解码阶段分离到不同的节点池中，在KV缓存的计算与消费之间引入了网络架构。这打破了经典缓存策略的一个关键假设：缓存未命中不再仅仅导致本地重计算，还需要向解码节点进行网络传输。同样，将一个数据块存入全局缓存也需要额外的传输，因此糟糕的保留决策会同时浪费内存和带宽。我们提出了KVLearn，一个面向解耦式服务中成本感知KV缓存保留的学习型框架。KVLearn包含：(i) 一个轻量级的前缀复用预测器（PRP），它从结构和时间上的前缀特征估计复用概率，而无需修改模型权重；(ii) 一个成本感知保留分数（CARS），它利用重计算、传输和存储成本将复用概率转换为保留/准入决策；(iii) 一个自适应阈值……（摘要被截断）

    arXiv:2610.28637v1 Announce Type: new  Abstract: Disaggregated \textsc{LLM} serving separates prefill and decode across node pools, placing a network fabric between \textsc{KV}-cache computation and consumption. This breaks a key assumption of classical cache policies: a miss no longer incurs only local recomputation, but also network transfer to the decode node. Likewise, admitting a block to the global cache requires an additional transfer, so poor retention decisions waste both memory and bandwidth. We present \textbf{KVLearn}, a learning-based framework for cost-aware \textsc{KV}-cache retention in disaggregated serving. KVLearn includes: (i) a lightweight Prefix Reuse Predictor (\textsc{PRP}) that estimates reuse probability from structural and temporal prefix features without modifying model weights; (ii) a Cost-Aware Retention Score (\textsc{CARS}) that converts reuse probability into keep/admit decisions using recomputation, transfer, and storage costs; and (iii) an Adaptive Th
    
[^237]: 基于神经算子的噪声环境规划

    Planning under Noise with Neural Operators

    [https://arxiv.org/abs/2610.28623](https://arxiv.org/abs/2610.28623)

    本文提出PUN神经算子，通过在屏蔽泊松随机最优控制框架中为障碍物引入反射诺伊曼边界条件、为目标引入狄利克雷条件，并结合网格无关的蒙特卡洛求解器，学习噪声感知的价值场，从而在未见过的几何环境中实现噪声下的可靠规划。

    

    arXiv:2610.28623v1 公告类型：新论文 摘要：在不确定的控制执行噪声下进行规划，需要在多样化环境中具备可靠的反馈策略。我们提出了“噪声下规划”神经算子（Planning Under Noise, PUN），它能够学习噪声感知的价值场，用于在未见过的几何环境中进行规划。基于已有的随机最优控制屏蔽泊松公式，我们解决了两个挑战：指定支持在不同几何环境中导航的边界条件，以及高效计算具有可靠梯度的价值场。零狄利克雷障碍物边界会在墙壁处终止采样路径，削弱沿长路径的目标信号，从而降低估计的可靠性。我们转而在障碍物上施加反射诺伊曼条件，并在目标处施加狄利克雷条件，使采样路径能够继续探索自由空间直至到达目标。为了在无需构型空间网格的情况下求解所得到的偏微分方程，我们扩展了蒙特卡洛求解器 Walk on Star……

    arXiv:2610.28623v1 Announce Type: new  Abstract: Planning under uncertain control execution noise requires reliable feedback policies across diverse environments. We introduce Planning Under Noise (PUN), a neural operator that learns noise-aware value fields for planning in unseen geometries. Building on an established screened-Poisson formulation of stochastic optimal control, we address two challenges: specifying boundary conditions that support navigation across varying geometries and efficiently computing fields with reliable gradients. Zero-Dirichlet obstacle boundaries terminate sampled paths at walls, weakening the goal signal along long routes and making it harder to estimate reliably. We instead impose reflecting Neumann conditions on obstacles and a Dirichlet condition at the goal, allowing sampled paths to continue exploring free space until they reach the target. To solve the resulting PDE without a configuration-space mesh, we extend a Monte Carlo (MC) solver, Walk on Star
    
[^238]: CountVideoBench：一个用于评估视频语言模型在物体计数任务上预测偏差的基准

    CountVideoBench: A Benchmark to Evaluate Prediction Biases on the Object Counting task by Video-Language Models

    [https://arxiv.org/abs/2610.28356](https://arxiv.org/abs/2610.28356)

    该论文提出CountVideoBench基准，利用2,600个覆盖[5,50]区间内26个计数值的合成视频，解决了现有视频问答基准计数分布严重偏向小数值、导致大数值上的模型预测偏差无法被评估的问题，从而系统性地评估视频语言模型在物体计数任务中的预测偏差。

    

    arXiv:2610.28356v1 公告类型：交叉 摘要：视频中的物体计数仍然是多模态大语言模型一项刻画不足的能力。许多现有的视频问答（VideoQA）基准中包含计数问题，但这些基准中的真实计数值分布严重偏向小数值。在我们调查的九个基准中，真实计数的中位数至多为5，且出现频率最高的三个数值合计占视频总数的45%至93%。因此，平均准确率主要由最常见的计数值主导；在罕见计数值上出现的依赖数值的偏差对平均值的贡献微乎其微，因而无法被察觉。我们提出了CountVideoBench，这是一个包含2,600个合成视频的基准，覆盖[5, 50]区间内的26个计数值（每个数值100个视频），并被分为六个由连续计数组成的聚类，既覆盖整个数值范围，又能支持近邻数值间的比较。每个视频在受控的摄像条件下包含单一类别的静态物体。

    arXiv:2610.28356v1 Announce Type: cross  Abstract: Object counting in videos remains a poorly characterized capability of multimodal large language models. Counting questions are included in many existing VideoQA benchmarks, but the ground-truth count distributions in these benchmarks are heavily skewed toward small values. Across the nine benchmarks we survey, the median ground-truth count is at most five, and the three most frequent values together account for between 45% and 93% of the videos. The resulting mean accuracy is then dominated by the most frequent counts; a value-dependent bias at a rare count contributes negligibly to the average and remains invisible. We introduce CountVideoBench, a benchmark of 2,600 synthetic videos covering 26 cardinalities in the interval [5, 50] (100 videos each), grouped into six clusters of consecutive counts that span the range while enabling near-neighbor comparisons. Each video contains a single class of static objects under a controlled came
    
[^239]: 观测同化Transformer：基于稀疏站点观测的概率性天气分析

    Observation Assimilation Transformer: Probabilistic Weather Analysis from Sparse Station Observations

    [https://arxiv.org/abs/2610.28327](https://arxiv.org/abs/2610.28327)

    提出观测同化Transformer（OAT），一种几何感知的神经过程模型，利用稀疏注意力机制同化稀疏站点观测，将数值天气预报先验场修正为高分辨率的概率性后验场，可扩展至数千站点并支持任意连续位置的预测。

    

    我们提出了观测同化Transformer（OAT），这是一种几何感知的神经过程，通过同化稀疏观测，将先验的数值天气预报（NWP）场修正并精化为高分辨率的后验场。OAT是一个编码器-解码器Transformer，它以一组观测位置及其NWP先验为条件，在任意未观测的查询位置集合上进行预测。其稀疏注意力机制使其可扩展至数千个站点观测，且仅依赖于位置之间的相对位置，而从不依赖绝对坐标。该模型允许部分缺失的观测，可在任意连续位置进行预测而无需网格投影，并且输出为参数化分布。我们开发了一种基于连续排序概率评分（CRPS）损失的站点随机化训练算法。我们在两个数据集上测试了OAT——一个是包含数百个站点的区域尺度数据集，另一个是……（摘要在此处截断）

    arXiv:2610.28327v1 Announce Type: new  Abstract: We present the Observation Assimilation Transformer (OAT), a geometry-aware neural process that corrects and refines a prior numerical weather prediction (NWP) field into a high-resolution posterior field by assimilating sparse observations. OAT is an encoder-decoder transformer that considers a set of observed locations and their NWP prior to condition predictions at a set of arbitrary unobserved query locations. Its sparse attention mechanism makes it scalable to thousands of station observations and depends on relative positions between locations, never on the absolute coordinates. The model allows for partially missing observations, predictions at arbitrary continuous locations without grid projection, and outputs are parametric distributions. A station-randomized training algorithm is developed with continuous ranked probability score (CRPS) loss. We test OAT on two datasets - one at regional scale with hundreds of stations, and one
    
[^240]: 不动集与计算相关均衡的困难性

    Fixed Sets and the Hardness of Computing Correlated Equilibria

    [https://arxiv.org/abs/2610.28141](https://arxiv.org/abs/2610.28141)

    本文证明计算多面体博弈的近似相关均衡是PPAD完全的（与纳什均衡同等难度），挑战了相关均衡比纳什均衡更容易求解的传统观点，并通过提出“不动集”这一不动点的自然推广建立了相关均衡与不动点理论的概念性联系。

    

    我们证明，计算多面体博弈的近似相关均衡是PPAD完全的，因此在多项式时间内等价于寻找纳什均衡。这一结果挑战了博弈论中的传统观点，即相关均衡是比纳什均衡更容易处理的对象。从概念层面，我们将相关均衡与不动点的一个自然推广联系起来，我们称之为“不动集”。连续函数 f 的不动集是一个非空有限集 S，满足 S ⊆ conv f(S)。我们的主要技术贡献是证明了即使达到逆多项式精度，计算不动集仍是PPAD困难的。

    arXiv:2610.28141v1 Announce Type: cross  Abstract: We show that computing an approximate correlated equilibrium of a polytope game is PPAD-complete, and thus polynomial-time equivalent to finding a Nash equilibrium. This challenges the conventional view in game theory that correlated equilibria are more tractable objects than Nash equilibria. From a conceptual standpoint, we connect correlated equilibria to a natural generalization of fixed points, which we call fixed sets. A fixed set of a continuous function $f$ is a nonempty finite set $S$ such that $S \subseteq \operatorname{conv} f(S)$. Our main technical contribution is to prove that computing a fixed set is PPAD-hard even at inverse polynomial precision.
    
[^241]: LaLa-World：基于自然语言动作扩展潜在世界模型

    LaLa-World: Scaling Latent World Models with Natural Language Actions

    [https://arxiv.org/abs/2610.27851](https://arxiv.org/abs/2610.27851)

    LaLa-World通过用语言统一描述动作与视觉变化、并将未来预测器构建于语言模型之上，使单个潜在世界模型能够跨仿真、机器人、图像和视频等异构数据进行联合训练，在MPC规划性能上超越了针对单一领域或任务训练的现有模型。

    

    本文表明，JEPA风格的潜在世界模型（LWM）可以通过采用视频生成模型的语言条件接口，实现跨多样化领域和数据源的学习。我们将这一方法确立为基于想象进行规划以及下游策略微调的有效且高效的基础。为此，我们提出了LaLa-World，一个语言条件潜在世界模型，它从仿真和机器人轨迹以及更广泛的图像和视频数据中学习。两个关键设计使这一工作成为可能：首先，我们用语言来描述数值动作和视觉变化，使得单一的未来预测器能够在异构数据集上进行联合训练；其次，我们将预测器构建在语言模型之上，使其在预测未来潜在状态的同时能够理解语言条件。由此得到的预测器相比针对每个领域或任务单独训练的现有LWM，实现了更高的平均MPC规划性能。

    arXiv:2610.27851v1 Announce Type: cross  Abstract: This paper shows that JEPA-style latent world models (LWMs) can learn across diverse domains and data sources by adopting the language-conditioned interface of video generation models. We establish this approach as an effective and efficient basis for planning with imagination and downstream policy fine-tuning. To this end, we introduce LaLa-World, a Language-conditioned Latent World model that learns from simulation and robot trajectories together with broader image and video data. Two designs make this work. First, we describe both numerical actions and visual changes in language, enabling joint training of a single future predictor across heterogeneous datasets. Second, we build the predictor on a language model, allowing it to interpret language conditions while predicting future latent states. The resulting predictor achieves higher average MPC planning performance than existing LWMs trained separately for each domain or task. Add
    
[^242]: 经验去噪器的隐私核算

    Privacy Accounting for Empirical Denoisers

    [https://arxiv.org/abs/2610.27710](https://arxiv.org/abs/2610.27710)

    本文提出一种无需训练的截断隐私可问责经验去噪器，通过显式计算其每步敏感性，使扩散模型采样过程中的每一步去噪都可建模为高斯机制，从而实现对采样过程本身隐私损失的形式化核算。

    

    扩散模型已成为现代生成系统的基础，但其强大的记忆能力引发了隐私方面的担忧。虽然差分隐私（DP）训练能够提供形式化的保证，但其代价是样本质量的下降。在本工作中，我们研究了一个与之互补的问题：在采样过程本身中会产生多少隐私损失？直接分析现代神经去噪器是具有挑战性的，因为它们对单个训练样本的敏感性通常是难以计算的。为了克服这一挑战，我们构建并分析了一种经过截断的隐私可问责经验去噪器——这是一种无需训练的贝叶斯最优去噪器估计器，其每一步的敏感性都可以被显式计算，从而使每个去噪步骤都能成为一个高斯机制。这一视角使得在以下情况下进行隐私核算成为可能：当每个训练样本对每个去噪步骤都有贡献时，可在高斯DP下进行隐私核算；当每个……（摘要在此处被截断）

    arXiv:2610.27710v1 Announce Type: new  Abstract: Diffusion models have emerged as the foundation of modern generative systems, yet their high memorization capacity raises privacy concerns. While differentially private (DP) training provides formal guarantees, it comes at a cost in sample quality. In this work, we study a complementary question: how much privacy loss is incurred during the sampling process itself? Directly analyzing modern neural denoisers is challenging because their sensitivity to a single training example is generally intractable. To overcome this challenge, we construct and analyze a clipped privacy-accountable empirical denoiser, a training-free estimator of the Bayes-optimal denoiser, whose per-step sensitivity can be computed explicitly, allowing each denoising step to be a Gaussian mechanism. This perspective enables privacy accounting under Gaussian DP when each training example contributes to each denoising step, and under $f$-DP trade-off functions when each 
    
[^243]: 用于优化深度学习模型的混合动量方法

    Hybrid Momentum Methods for Optimizing Deep Learning Models

    [https://arxiv.org/abs/2610.27674](https://arxiv.org/abs/2610.27674)

    提出混合动量方法HyM，将线性加权的长期历史梯度与几何折扣的快动量相结合，并据此构建经Newton–Schulz正交化的HyMUON优化器，在保留长期梯度信息与适应近期变化之间实现平衡。

    

    动量是深度学习随机优化的核心，包括大语言模型的训练。经典动量方法通常依赖快速的指数移动平均（EMA）方案，该方案以固定的时间尺度对过去的梯度进行折扣，可能低估那些仍然具有信息价值的较早梯度。这在保留长期历史与适应近期变化之间造成了权衡。我们提出混合动量（HyM），它将过去梯度的线性加权平均与几何折扣的快动量相结合。前者按梯度迭代索引成比例地分配权重，并保留不断增长的历史跨度；后者则强调近期梯度。这种混合结构是模块化的，可以与自适应预处理及其他更新变换相结合。特别地，我们提出HyMUON，它对混合方向应用Newton–Schulz正交化。

    arXiv:2610.27674v1 Announce Type: new  Abstract: Momentum is central to stochastic optimization in deep learning, including the training of large language models. Classical momentum methods typically rely on a fast exponential moving average (EMA) scheme, which discounts past gradients on a fixed timescale and may underweight older gradients that remain informative. This creates a trade-off between retaining long-term history and adapting to recent changes. We introduce Hybrid Momentum (HyM), which combines a linearly weighted average of past gradients with a geometrically discounted fast momentum. The former assigns weights proportional to the gradients' iteration indices and retains a growing span of history, while the latter emphasizes recent gradients. This hybrid construction is modular and can be combined with adaptive preconditioning and other update transformations. In particular, we propose HyMUON, which applies Newton--Schulz orthogonalization to the hybrid direction. Illustr
    
[^244]: 光滑非凸零阶优化的下界：低曲率迹条件下确定性的代价

    Lower Bounds for Smooth Nonconvex Zeroth-Order Optimization: The Price of Determinism under Low Curvature Trace

    [https://arxiv.org/abs/2610.27550](https://arxiv.org/abs/2610.27550)

    本文在低曲率迹（有效秩）条件下证明了随机化零阶非凸优化仅需 O(ET) 次函数调用且在常数因子内最优，而任何确定性方法都无法获得这种依赖有效秩的节省，从而量化了确定性方法的固有代价。

    

    低曲率迹（low curvature trace）是一种有效秩条件，常被用来解释随机化零阶方法为何能扩展到语言模型微调等高维任务。确定性方法能否获得同样的节省？我们证明它们不能。我们研究仅凭精确函数值在 ℝ^d 上寻找 L-光滑非凸函数 ε-驻点的问题，并假设在初始值以下 Hessian 被一个迹至多为 LE 的固定矩阵所支配；这正是用于解释零阶微调的有效秩条件。当 ET≤d 时，此类微调所采用的高斯两点法只要拒绝使 f 上升的步，就只需 O(ET) 次函数调用，其中 T=LΔ/ε² 是从初始间隙 Δ 出发的梯度下降迭代次数；我们还证明任何随机化方法都需要 Ω(ET) 次调用，因此该复杂度在常数因子内是最优的。相比之下，每个确定性方法……（原文摘要至此截断）

    arXiv:2610.27550v1 Announce Type: new  Abstract: Low curvature trace, an effective rank condition, is a common explanation for why randomized zeroth-order methods scale to high-dimensional tasks such as language model fine-tuning. Can deterministic methods achieve the same savings? We show that they cannot. We study finding $\varepsilon$-stationary points of $L$-smooth nonconvex functions on $\mathbb{R}^d$ from exact function values, assuming that below the starting value the Hessian is dominated by a fixed matrix with trace at most $LE$; this is the effective rank condition used to explain zeroth-order fine-tuning. When $ET\le d$, the Gaussian two-point method that such fine-tuning uses needs only $O(ET)$ calls if it rejects steps that raise $f$, where $T=L\Delta/\varepsilon^2$ is the iteration count of gradient descent from initial gap $\Delta$, and we prove that every randomized method needs $\Omega(ET)$ calls, so this cost is optimal up to constants. In contrast, every deterministi
    
[^245]: 面向流体与等离子体的机器学习谱方法偏微分方程求解器

    Machine-learned spectral PDE solvers for fluids and plasmas

    [https://arxiv.org/abs/2610.27317](https://arxiv.org/abs/2610.27317)

    提出WARP方法，通过在求解PDE的空间离散化中进行弱形式发现，并以观测状态为锚点的多步推演微分学习，构建出数值稳定的机器学习谱方法PDE求解器，在流体和等离子体基准系统上显著提升了系数恢复、推演精度和统计度量。

    

    用于从数据中发现控制偏微分方程（PDE）的稀疏回归方法对观测噪声敏感，且不受约束在正向积分下再现数据。我们通过在求解PDE所用的空间离散化中执行弱形式发现来克服这两个局限性，该方法通过在以观测状态为锚点的多步推演上进行微分来学习。新方法称为偏微分方程弱形式锚定推演（WARP），可产生机器学习的PDE求解器。在本工作中，我们聚焦于适合谱伽辽金（spectral Galerkin）求解器的系统。在八个基准流体与等离子体系统（包括高维湍流模拟）上，与单步弱形式SINDy基线相比，WARP能够从新的随机初始条件生成数值稳定的推演，并在系数恢复、推演精度和统计度量方面显著提升。

    arXiv:2610.27317v1 Announce Type: cross  Abstract: Sparse regression methods for discovering governing partial differential equations (PDEs) from data are sensitive to observation noise and are not constrained to reproduce the data under forward integration. We address both limitations by performing weak-form discovery in the spatial discretization in which the PDE is solved, learned by differentiating through multistep rollouts anchored at the observed states. The new method, Weak-form Anchored Rollout for PDEs (WARP), produces a machine-learned PDE solver. In this work, we focus on systems amenable to a spectral Galerkin solver. Across eight benchmark fluid and plasma systems, including high-dimensional turbulent simulations, WARP produces numerically stable rollouts from new random initial conditions and substantially improves coefficient recovery, rollout accuracy, and statistical measures compared with a single-step weak-form SINDy baseline.
    
[^246]: 当弱次模性遇上拟阵：标准贪心算法表现如何？

    When Weak Submodularity Meets Matroids: How Does Standard Greedy Behave?

    [https://arxiv.org/abs/2610.26967](https://arxiv.org/abs/2610.26967)

    本文解决了弱次模最大化在一般拟阵约束下标准贪心算法最坏情况近似保证这一长期开放问题，给出了以次模比 $\gamma$ 和拟阵秩 $r$ 表达的精化下界。

    

    弱次模最大化在机器学习、数据挖掘和统计学中有众多应用。特别是在基数约束下，标准贪心算法已被证明可以达到紧致的 $(1-e^{-\gamma})$-近似保证，其中 $\gamma$ 表示次模比。然而，在考虑一般拟阵约束时，标准贪心算法究竟能表现多好一直是一个长期悬而未决的开放问题。本文通过研究标准前向贪心算法的最坏情况近似保证 $R_{\mathrm{gr}}(\gamma,r)$（其中 $r$ 为拟阵的秩）来解答这一问题。具体而言，本文的第一个贡献是给出了如下精化的下界：$R_{\gr}(\gamma,r)\ge \max\left\{ \frac{\gamma}{r}, \frac{\gamma^2}{B_\gamma+\sqrt{\gamma(r-1)}}, \frac{\gamma^{r+1}}{2-\gamma+\gamma^{r+1}}, \frac{\gamma^2}{14\left(B_\gamma+\sqrt{\gamma(1-\gamma)(r-1)}\right)} \right\}$。

    arXiv:2610.26967v1 Announce Type: cross  Abstract: Weakly submodular maximization finds numerous applications in machine learning, data mining and statistics. In particular, under a cardinality constraint, the standard greedy algorithm is known to attain the tight $(1-e^{-\gamma})$-approximation guarantee, where $\gamma$ denotes the submodularity ratio. However, when considering general matroid constraints, \emph{how well standard greedy can perform} has remained a longstanding open question. In this paper, we resolve this question by studying the worst-case approximation guarantee $R_{\mathrm{gr}}(\gamma,r)$ of standard forward greedy, where $r$ is the rank of matroid. Specifically, our first contribution is the following refined lower bound \begin{equation*} R_{\gr}(\gamma,r)\ge \max\left\{ \frac{\gamma}{r}, \frac{\gamma^2} {B_\gamma+\sqrt{\gamma(r-1)}}, \frac{\gamma^{r+1}} {2-\gamma+\gamma^{r+1}}, \frac{\gamma^2} {14\left(B_\gamma+\sqrt{\gamma(1-\gamma)(r-1)}\right)} \right\}, \end{
    
[^247]: 超越策略差异：能力感知的多教师在线策略蒸馏

    Beyond Policy Discrepancy: Capability-Aware Multi-Teacher On-Policy Distillation

    [https://arxiv.org/abs/2610.26950](https://arxiv.org/abs/2610.26950)

    提出能力感知的多教师在线策略蒸馏方法 CA-MOPD，将强化学习专家模型合并与验证器估计的师生能力差距相结合，根据学生已掌握的内容动态调整各领域蒸馏强度，克服了仅基于策略差异分配监督的局限性。

    

    多教师在线策略蒸馏（MOPD）将领域专家模型整合到一个共享的学生模型中，但如何确定每个教师应给予多强的监督仍然是一个挑战。现有的基于差异的分配方法，即使学生已经达到或超过教师的任务性能，也可能分配强监督，因为行为上的不匹配并不一定意味着还存在能力缺陷。我们提出能力感知的多教师在线策略蒸馏（CA-MOPD），它将专家模型合并与能力引导的分配相结合，以考虑学生已经学到的内容。CA-MOPD 首先将强化学习专家模型合并为强大的学生模型初始化，然后使用验证器估计的师生能力差距来调整领域级别的蒸馏强度。随着能力差距的缩小，监督强度降低；当估计的学生能力出现退化时，监督强度增加，同时结合密集的 token 级别指导……

    arXiv:2610.26950v1 Announce Type: cross  Abstract: Multi-teacher on-policy distillation (MOPD) integrates domain specialists into a shared student, but determining how strongly each teacher should supervise remains challenging. Existing discrepancy-based allocation can assign strong supervision even when the student already matches or exceeds the teacher's task performance, since behavioral mismatch does not necessarily imply a remaining capability deficit. We propose Capability-Aware Multi-Teacher On-Policy Distillation (CA-MOPD), which combines specialist merging with capability-guided allocation to account for what the student has already learned. CA-MOPD first merges reinforcement-learning specialists into a strong student initialization, then uses verifier-estimated teacher--student capability gaps to adjust domain-level distillation strength. Supervision decreases as capability gaps close and increases when estimated student capability regresses, while dense token-level guidance 
    
[^248]: 重复条件下的复制：从前缀匹配到位置哈希

    Copying Under Repetition: From Prefix Matching to Position Hashing

    [https://arxiv.org/abs/2610.26722](https://arxiv.org/abs/2610.26722)

    本文提出利用小素数周期的VO旋转位置编码并结合中国剩余定理，证明了两层Transformer能在任意重复字符串下实现精确复制并正确终止，同时给出了顺序有限精度下复制的不可能性证明。

    

    前沿大语言模型（LLM）几乎能够完美复制随机文本，但在处理长段重复字符时却常常出现过度复制或截断的问题。在重复字符串中，匹配局部内容不再能可靠地判断复制应从何处继续或何时停止。然而，我们证明了一个具有 $O(\log\log N)$ 位精度和 $O(\log N/\log\log N)$ 宽度的两层 Transformer 能够复制直到设计上限 $N$ 的任意字符串，并在任意重复情况下正确终止。我们采用具有小素数周期的 RoPE 风格的注意力值与输出旋转（VO-旋转），以对这些素数取模的方式编码到分隔符锚点的距离。根据中国剩余定理，这些周期性编码共同区分了所有所需的距离，使得单一检索规则即可选择下一个源 token 或结束标记。相比之下，我们证明了在顺序有限精度下复制的一个与宽度无关的不可能性结果。

    arXiv:2610.26722v2 Announce Type: replace-cross  Abstract: Frontier LLMs can copy random text almost flawlessly, yet they often overrun or truncate long blocks of repeated characters. In repeated strings, matching local content no longer reliably identifies where copying should continue or when it should stop. Nevertheless, we show that a two-layer transformer with $O(\log\log N)$-bit precision and $O(\log N/\log\log N)$ width can copy every string up to a chosen design horizon $N$ and terminate correctly under arbitrary repetition. We use RoPE-style rotations of attention values and outputs (VO-rotations) with small prime periods to encode distances from delimiter anchors modulo those primes. By the Chinese Remainder Theorem, these periodic encodings jointly distinguish all required distances, allowing one retrieval rule to select either the next source token or the end marker. In contrast, we prove a width-independent impossibility result for copying under sequential finite-precision
    
[^249]: 突破扩散：基于草稿精炼的并行推理

    Breaking Diffusion: Parallel Reasoning with Draft Refinement

    [https://arxiv.org/abs/2610.26542](https://arxiv.org/abs/2610.26542)

    该论文提出草稿精炼语言模型（DRLMs），利用单个去噪器并行提出并评估草稿修订，并通过校正自身token证据来衡量token所获得的上下文支持，从而突破传统扩散模型迭代去噪无法保证整体答案变优的局限。

    

    草拟答案使推理模型能够在其他决策的背景下重新审视每个决策。然而，反复的扩散去噪并不能确保各个单独看似合理的token更新共同形成一个更好的答案。我们提出了草稿精炼语言模型（DRLMs），它使用单个去噪器来提出并行修订并评估候选草稿。我们训练该去噪器从均匀损坏的草稿中重建每个token，同时教会它保留和修复。其预测直接指定修订内容，而无需扩散反向转移过程。评估则需要以另一种方式使用这些预测：对已有token的置信度既反映了上下文支持，也包含了观察该token本身所得的证据。利用已知的损坏过程，我们校正这种自身token证据，以估计每个token在多大程度上受到提示和其余草稿的支持。汇总这些估计可得到一个空腔熵……（原文在此处截断）

    arXiv:2610.26542v1 Announce Type: cross  Abstract: Drafting an answer allows a reasoning model to reconsider each decision in the context of the others. Yet repeated diffusion denoising does not ensure that individually plausible token updates form a better answer together. We introduce Draft Refinement Language Models (DRLMs), which use a single denoiser to propose parallel revisions and evaluate a candidate draft. We train the denoiser to reconstruct every token from uniformly corrupted drafts, teaching both preservation and repair. Its predictions specify revisions directly, without a diffusion reverse transition. Evaluation requires a different use of these predictions: confidence in an existing token reflects both contextual support and evidence from observing the token itself. Using the known corruption process, we correct for this self-token evidence to estimate how well each token is supported by the prompt and the remaining draft. Aggregating these estimates yields a cavity en
    
[^250]: 自组织强化学习：无需学习者的群体逼近探索-利用最优解

    Self-organizing reinforcement learning: a learner-free swarm approaches the explore-exploit optimum

    [https://arxiv.org/abs/2610.25113](https://arxiv.org/abs/2610.25113)

    一个无需学习者与奖励信号的极简自组织觅食群体，通过Levy飞行探索、阈值级联招募与拥挤分散三者的耦合循环，能在约150步内动态逼近多智能体资源共享博弈的理想自由/Nash均衡分配（匹配斜率达0.94），且移除招募-运动耦合后匹配会崩溃至0.08。

    

    探索/利用权衡是强化学习（RL）的核心问题。我们证明了一个极简觅食群体在无需学习者、无需奖励信号的情况下即可逼近其最优解——即多智能体资源共享博弈的理想自由/Nash分配（这是一个均衡，而非回报最大化）。智能体通过重尾（Levy）飞行进行探索，通过在其邻近网络上的阈值级联将邻居招募到新发现的资源处，并在资源斑块拥挤时分散离开；这种级联循环平衡了探索与利用，并驱使觅食者趋向理想自由分布。该群体达到了这一最优解的形态（专注觅食者以0.94的斜率追踪资源丰富度），但并未精确达到其均衡——它仍存在可测量的可利用空间——而是通过招募与运动的耦合在动态过程中（约150步）逼近该状态：一旦移除这种耦合，匹配即告崩溃（0.94降至0.08）。

    arXiv:2610.25113v2 Announce Type: replace  Abstract: The explore/exploit trade-off is the core problem of reinforcement learning (RL). We show that a minimal foraging swarm approaches its optimum - the ideal-free / Nash allocation of a multi-agent resource-sharing game (an equilibrium, not a return maximum) - with no learner and no reward signal. Agents explore by heavy-tailed (Levy) flights, recruit neighbours to a discovery through a threshold cascade on their proximity network, and disperse as a patch crowds; this cascade cycle balances exploration and exploitation and drives foragers toward the ideal free distribution. The swarm attains the shape of this optimum (committed foragers track richness with slope 0.94) but not its exact equilibrium - it stays measurably exploitable - reaching it dynamically (~150 steps) through the coupling of recruitment to movement: remove the coupling and matching collapses (0.94 -> 0.08). The RL sits at the level of the problem and the allocation rea
    
[^251]: TrustAgent-RFF：面向域偏移下可靠射频指纹识别的智能体诊断规划框架

    TrustAgent-RFF: Agentic Diagnostic Planning for Reliable RF Fingerprinting under Domain Shift

    [https://arxiv.org/abs/2610.24930](https://arxiv.org/abs/2610.24930)

    提出TrustAgent-RFF预测级可靠性框架，在不改动冻结I/Q分类器且无需目标域标签的条件下，通过融合反事实稳定性、原型一致性、采集域相似度与信号质量等射频证据，并利用源域校准的信任估计器输出预测可信度，从而解决域偏移下射频指纹识别可靠性不足的问题。

    

    射频指纹识别（RFF）模型在源域条件下能够取得较高的准确率，但当接收机、距离、采集日期或环境发生变化时，其可靠性会显著下降。在未见域中出现的错误预测可能获得较高的softmax置信度，这使得仅依靠置信度阈值方法不足以实现可信部署。我们提出了TrustAgent-RFF，一个面向RFF的预测级可靠性框架，它可以在冻结的同相正交（I/Q）分类器上运行，且无需目标域标签。对于每次设备预测，该框架都会采集射频证据：反事实稳定性、原型一致性、采集域相似度以及信号质量。一个基于源域校准的信任估计器将这些诊断信息与分类器置信度、预测边际以及证据可用性指标相结合。当诊断信号指示异常时，源自源域的下尾惩罚和跨模块交互作用会降低信任度……

    arXiv:2610.24930v2 Announce Type: replace  Abstract: Radio frequency fingerprinting (RFF) models can achieve accuracy under source domain conditions but become unreliable when the receiver, distance, collection day, or environment changes. Incorrect predictions in unseen domains may receive high softmax confidence, making confidence thresholding insufficient for trustworthy deployment. We propose TrustAgent-RFF, a prediction level reliability framework for RFF that operates with a frozen in-phase and quadrature (I/Q) classifier without target domain labels. For each device prediction, the framework acquires radio frequency evidence: counterfactual stability, prototype consistency, acquisition domain similarity, and signal quality. A source calibrated trust estimator combines these diagnostics with classifier confidence, prediction margin, and evidence availability indicators. Source derived lower tail penalties and cross module interactions reduce trust when diagnostic signals indicate
    
[^252]: LLM优化器的一个重要瓶颈：原理级局部最优

    An important Bottleneck in LLM Optimizers: Principle-Level Local Optima

    [https://arxiv.org/abs/2610.24291](https://arxiv.org/abs/2610.24291)

    本文揭示LLM优化器普遍存在“原理级搜索坍塌”瓶颈，即搜索被局限于狭窄的设计原理集合导致性能长期停滞，并通过跨三个领域的诊断基准与决策树协议证实78.3%的运行中存在该现象。

    

    反馈驱动的LLM优化器正日益被应用于进化搜索、贝叶斯优化、启发式设计和科学发现。然而，尽管仍剩余大量评估预算，快速的初始收益往往会让位于长期的性能停滞。我们研究了“原理级搜索坍塌”现象，即优化器的搜索被局限于狭窄的一组设计原理，尽管它可能仍在持续生成不同的候选方案并取得微小的改进。我们构建了一个涵盖分子优化、网络哨兵节点选择和GPU内核优化的诊断基准，并引入了一个决策树协议，该协议综合考量原理收窄、长期性能平台期以及与可达到的参考解决方案之间的差距。在三个问题领域的161次可评估运行中，78.3%的运行结果支持坍塌现象的存在，其中55.3%具有强支持。然而，候选层面的变化可能会掩盖……

    arXiv:2610.24291v2 Announce Type: replace  Abstract: Feedback-driven LLM optimizers are increasingly used for evolutionary search, Bayesian optimization, heuristic design, and scientific discovery. Yet rapid initial gains often give way to prolonged plateaus despite substantial remaining evaluation budgets. We investigate \emph{principle-level search collapse}, where an optimizer's search becomes confined to a narrow set of design principles, even though it may continue to generate distinct candidates and achieve small improvements. We assemble a diagnostic benchmark spanning molecular optimization, network sentinel-node selection, and GPU-kernel optimization, and introduce a decision-tree protocol combining principle narrowing, prolonged performance plateaus, and gaps to attainable reference solutions. Across 161 assessable runs on the three problem domains, 78.3\% receive support for collapse, including 55.3\% with strong support. Candidate-level variation can nevertheless conceal re
    
[^253]: 首次通过注意力

    First Passage Attention

    [https://arxiv.org/abs/2610.22714](https://arxiv.org/abs/2610.22714)

    该论文提出首次通过注意力（FPA），将注意力衰减机制从路径求和转向路径极值——根据学习标量场在中间跨度上的最深下凹来惩罚键的logit，从而能在保留较旧结构相关token的同时选择性擦除中间上下文，并证明常数维度的FPA头即可实现任意存活键子集，而传统softmax头或token级遗忘门需要与键数呈线性关系的维度。

    

    使用遗忘门增强softmax注意力通常依赖于沿键-查询路径对衰减贡献进行求和。由于这种累加是单调的，这些机制无法在保留较旧的、结构上相关的token的同时，选择性地擦除中间的上下文。我们提出首次通过注意力，它将从路径求和转向路径极值：通过将某个键的logit按比例降低，降低量与中间跨度上学习到的标量场低于该键水平的最深下凹成正比。FPA将遗忘Transformer（FoX）作为特例包含在内，并且当标量场和水平遵循共享的学习随机游走时，具有未受惩罚logit的存活键自然形成一个栈。我们证明，一个常数维度的FPA头可以实现由后续下凹所留下的任意存活键子集，而带有位置偏置或token级遗忘门的softmax头则需要与键数量呈线性关系的头维度。在合成实验中

    arXiv:2610.22714v1 Announce Type: new  Abstract: Augmenting softmax attention with forget gates typically relies on summing decay contributions along the key--query path. Because such accumulation is monotonic, these mechanisms cannot selectively erase intermediate context while preserving older, structurally relevant tokens. We propose First Passage Attention (FPA), which shifts from path summation to path extrema by lowering a key's logit proportionally to the deepest dip of a learned scalar field below that key's level over the intervening span. FPA encompasses the Forgetting Transformer (FoX) as a special case, and when the field and levels follow a shared learned walk, the alive keys with unpenalised logits naturally form a stack. We show that a constant-dimension FPA head realises any subset of keys left alive by subsequent dips, whereas softmax heads with positional biases or token-level forget gates require a head dimension linear in the number of keys. In synthetic experiments
    
[^254]: 非凸学习中随机块坐标下降的泛化保证

    Generalization Guarantees of Randomized Block Coordinate Descent for Nonconvex Learning

    [https://arxiv.org/abs/2610.22213](https://arxiv.org/abs/2610.22213)

    本文为非凸学习中的随机块坐标下降算法建立了高概率泛化保证，并通过早停策略量化了优化与泛化的平衡，实现了总体风险梯度的最优收敛速率。

    

    随机块坐标下降是众多大规模机器学习问题中广泛使用的一种优化算法。现有的RBCD理论分析主要从优化角度关注其收敛行为，而其在非凸问题中的泛化行为仍然知之甚少。本文为具有非凸目标的RBCD建立了高概率泛化保证。在一般的非凸设置下，我们推导了经验风险和总体风险梯度的界，且仅需标准的光滑性假设。在采用多项式衰减步长的情况下，我们通过调整迭代次数来平衡优化与泛化，从而量化了早停如何实现总体风险梯度的最优收敛速率。在梯度主导的设置下，我们给出了关于函数次优性的界，该界几乎匹配凸优化中乐观的收敛速率。

    arXiv:2610.22213v1 Announce Type: new  Abstract: Randomized block coordinate descent (RBCD) is a popular optimization algorithm for many large-scale machine learning problems. The existing theoretical analysis of RBCD mainly focuses on its convergence behavior from an optimization perspective, while its generalization behavior remains less understood for nonconvex problems. In this paper, we establish high-probability generalization guarantees for RBCD with nonconvex objectives. In the general nonconvex setting, we derive bounds on the gradients of both empirical and population risks, requiring only standard smoothness assumptions. With polynomially decaying step sizes, we tune the iteration number to balance optimization and generalization, thereby quantifying how early stopping leads to the optimal convergence rate for the population gradients. In the gradient-dominated setting, we provide bounds in terms of functional suboptimality, which nearly match the optimistic rates in convex 
    
[^255]: 二比特的N-Gram记忆：大语言模型中查找表的量化

    N-Gram Memories at Two Bits: Quantizing Lookup Tables in Large Language Models

    [https://arxiv.org/abs/2610.21810](https://arxiv.org/abs/2610.21810)

    本文提出条件程序（CP）压缩表示方法，仅用两比特编码按需重构大语言模型n-gram记忆查找表中被请求的行，相比标准2比特量化将输出KL散度降低41–63%，并将DeepSeek的表存储从203GB压缩至51GB。

    

    条件记忆通过选择性查找来扩展语言模型的容量，但完整的嵌入表必须始终可用以应对每一个可能的输入。我们提出了条件程序，这是一种压缩表示方法，只需重建输入所请求的行。CP用短码和局部尺度来表示每一行，并使用一个在整个表上共享的重构程序。我们通过高斯格、范数恢复以及可独立解码的行记录来实例化这种表示。CP的两比特码相比2比特最近取整量化，将输出KL散度降低了41–63%。对三个经过训练的n-gram记忆族的分析表明，类高斯的坐标统计与行之间学习到的关系并存；在相同的存储预算下，共享格结构的表现优于我们测试的拟合码本和低秩残差变体。对于DeepSeek，该方法将表存储从203GB降至51GB，为更大的工作负载留出了空间。

    arXiv:2610.21810v1 Announce Type: cross  Abstract: Conditional memory expands language-model capacity through selective lookup, but the full embedding table must remain available for every possible input. We introduce Conditional Program (CP), a compressed representation that reconstructs only the rows an input requests. CP represents each row by a short code and a local scale, with a reconstruction program shared across the table. We instantiate this representation with a Gaussian trellis, norm restoration and independently decodable row records. CP's two-bit code lowers output KL by 41--63\% relative to 2-bit round-to-nearest. Analysis across three trained n-gram memory families shows that Gaussian-like coordinate statistics coexist with learned relationships among rows; at matched storage budgets, the shared trellis outperforms the fitted codebooks and low-rank residual variants we test. For DeepSeek, it reduces table storage from 203GB to 51GB, leaving room for larger workloads whe
    
[^256]: DialFork：超越事实性，评估对话代理中的程序性幻觉

    DialFork: Going Beyond Factuality to Evaluate Procedural Hallucination in Conversational Agents

    [https://arxiv.org/abs/2610.21561](https://arxiv.org/abs/2610.21561)

    DialFork是首个评估对话代理“程序性幻觉”的基准，通过在受控对话轮次位置注入四种幻觉类型构建了涵盖12个领域、24个任务的36,278个对话，并从三个粒度定义了刻画程序性偏差的评估任务。

    

    作为对话代理部署的大语言模型（LLM）被越来越多地期望遵循程序性逻辑。虽然事实性幻觉已受到广泛关注，但另一种失败模式却被忽视了：程序性幻觉，即代理产生的语句虽然单独来看流畅且事实上貌似合理，却偏离了预期的对话流程。我们提出了DialFork，这是首个在原始对话文本上衡量这种失败模式的基准。DialFork通过在受控轮次位置对符合规范的对话进行分叉并注入四种幻觉类型之一来构建，产生了涵盖12个领域、24个任务的36,278个对话；30名人工标注者验证了该构建过程。我们定义了三个评估任务，从三个粒度刻画程序性偏差：一组对话偏离符合规范对话的程度（用于监控已部署的代理）、单个对话偏离的程度（用于程序性检索……

    arXiv:2610.21561v1 Announce Type: cross  Abstract: Large language models (LLMs) deployed as conversational agents are increasingly expected to follow procedural logic. While factual hallucination has received extensive attention, a different failure mode has been overlooked: procedural hallucination, in which an agent produces individually fluent and factually plausible utterances while deviating from the intended conversational process. We introduce DialFork, the first benchmark to measure this failure mode on raw dialogue text. DialFork is built by forking conformant dialogues at controlled turn positions and injecting one of four hallucination types, yielding 36,278 dialogues across 24 tasks in 12 domains; 30 human annotators validate the construction. We define three evaluation tasks that characterize procedural deviation at three granularities: how far a set of dialogues deviates from conformant ones (monitoring deployed agents), how far a single dialogue does (procedural retrieva
    
[^257]: 扩展多语言专家混合模型：语言覆盖范围如何改变容量与计算分配

    Scaling Multilingual Mixture-of-Experts: How Language Coverage Changes Capacity and Compute Allocation

    [https://arxiv.org/abs/2610.20247](https://arxiv.org/abs/2610.20247)

    本文提出了一种感知重复的联合缩放定律，揭示多语言MoE模型中活跃参数、训练词元、专家数量与语言覆盖范围之间的关系，并发现32个专家相比稠密模型可将语言从12种扩展到200种时的损失惩罚降低83%。

    

    多语言模型在各语言之间共享容量，因此增加语言数量可能会损害现有语言的表现。专家混合模型只需增加极少量的每词元计算即可扩展这一容量。鉴于这种额外容量，增加专家数量能否降低每种新增语言的代价？为回答这一问题，我们提出了一种感知重复的联合缩放定律，将活跃参数数量、每种语言的训练词元数量、专家数量与语言覆盖范围联系起来。该定律基于来自64个预训练配置的26,300个逐语言损失观测数据进行拟合，在留出配置上相比最强评估基线将预测误差降低了28%。在活跃参数数量和每种语言训练词元数量相同的情况下，32个专家相比稠密模型将语言从12种扩展到200种时的损失惩罚降低了83%。我们的定律进一步表明，更广的语言覆盖范围会在更大的参数范围内倾向于使用更多专家。

    arXiv:2610.20247v1 Announce Type: new  Abstract: Multilingual models share capacity across languages, so adding languages can hurt performance on existing ones. Mixture-of-experts models expand this capacity with little additional per-token computation. Given this extra capacity, does adding experts reduce what each additional language costs? To answer this question, we propose a repetition-aware joint scaling law relating number of active parameters, training tokens per language, number of experts, and language coverage. Fitted on 26,300 per-language loss observations from 64 pretrained configurations, our law reduces prediction error on held-out configurations by 28% relative to the strongest evaluated baseline. At matched number of active parameters and training tokens per language, 32 experts reduce the loss penalty of expanding from 12 to 200 languages by 83% compared with a dense model. Our law further shows that broader language coverage favors more experts across a wider range 
    
[^258]: 基于正-未标记学习的扩散模型渐近精确负引导

    Asymptotically Exact Negative Guidance of Diffusion Models via Positive-Unlabeled Learning

    [https://arxiv.org/abs/2610.19802](https://arxiv.org/abs/2610.19802)

    提出一种从显式目标密度采样的扩散模型负引导新方法，利用正-未标记学习在线估计样本属于不良条件的概率，从而实现渐近精确、偏差可通过增加计算量系统降低的负引导。

    

    扩散模型的负引导旨在生成避开某种不期望条件的样本，该条件既可以由参考样本指定，也可以由条件得分与无条件模型共同指定。现有方法在每个样本处注入由局部信息推导出的排斥项，因此所得采样器并不以某个显式密度为目标，其偏差也无法通过增加计算量而被系统地降低。与之相反，我们提出了一种从显式目标密度采样的负引导方法，该方法能够统一处理基于参考样本和基于得分的两种条件指定方式，并以一个参数化目标密度本身的回避强度来抑制不期望条件附近的概率质量。实现这一目标需要在每个扩散时间步估计每个样本属于不期望条件的可能性。我们将这一估计问题构造为一个正-未标记学习问题，该问题支持轻量级的在线训练。

    arXiv:2610.19802v1 Announce Type: new  Abstract: Negative guidance of diffusion models generates samples that avoid an undesired condition, specified either by reference samples or by a conditional score alongside an unconditional model. Existing methods inject repulsive terms derived from local information at each sample, so the resulting sampler does not target an explicit density, and its bias cannot be systematically reduced by increasing computation. We instead present a negative-guidance method that samples from an explicit target density, which uniformly handles both reference-sample and score-based specifications and suppresses probability mass near the undesired condition with an avoidance strength that parametrizes the target density itself. Realizing this target requires estimating, at each diffusion timestep, how likely each sample is to belong to the undesired condition. We cast this estimation as a Positive-Unlabeled learning problem that admits lightweight online trainin
    
[^259]: 论自然语言自编码器的可监控性：识别与判别

    On the Monitorability of Natural Language Autoencoders: Identification vs Discrimination

    [https://arxiv.org/abs/2610.18727](https://arxiv.org/abs/2610.18727)

    论文将自然语言自编码器的训练与监督部署形式化为两种不同的指称博弈，并指出二者之间的差距导致其解释虽能识别对应的激活，却难以让人类监控者从多个近似叙述中辨别出真实的那一个。

    

    自然语言自编码器（NLA）被提出作为语言模型的监督工具：激活言语化器（AV）为激活向量撰写英文解释，而激活重构器（AR）将解释映射回向量。它们在监督中的价值取决于人类监控者在给定AV解释的情况下，能否将一段对激活的真实叙述与虚假叙述区分开来。我们将NLA的训练与NLA监督部署形式化为源自涌现通信领域的两种不同的指称博弈：训练是一种生成式指称博弈，而监督是一种判别式的、多模态的、部分可观测的指称博弈，该博弈在受监控语言模型的激活与真实描述它们的叙述之间的多对多关系上展开。这两种博弈之间的差距预示着AV的解释将能够识别其所描述的是哪个激活，却无法将其真实叙述与紧密匹配的叙述区分开来。

    arXiv:2610.18727v2 Announce Type: replace-cross  Abstract: Natural Language Autoencoders (NLAs) are proposed as oversight tools for language models: an Activation Verbaliser (AV) writes an English explanation of an activation vector, and an Activation Reconstructor (AR) maps the explanation back to a vector. Their value for oversight depends on a human monitor being able to tell a true account of an activation from a false one, given the AV's explanation. We formalise NLA training and NLA oversight deployment as two different referential games from Emergent Communication: training is a generative referential game, while oversight is a discriminative, multi-modal, partially-observable referential game played over a many-to-many relation between the monitored language model's activations and the accounts that are true of them. The gap between the two games predicts that AV's explanations will identify which activation they describe without discriminating its true account from closely mat
    
[^260]: 尊重语义即可：通过LLM前语义可变分辨率分词与LLM内相关性剪枝实现图像token压缩

    Just Respect the Semantics: Image-Token Compression via Pre-LLM Semantic Variable-Resolution Tokenization and In-LLM Relevance Pruning

    [https://arxiv.org/abs/2610.18690](https://arxiv.org/abs/2610.18690)

    Varest通过语义引导的可变分辨率token化与相关性token剪枝，仅用5.56%的图像token即可保留86%-92%的全token性能，实现高效且语义无损的图像token压缩。

    

    图像token压缩必须在不损害语义的前提下减少计算量。我们提出Varest，其核心指导原则是语义可替代性：连贯语义区域内的token可以被粗略概括，而物体边界和文字内容则需要更精细的区分。物体和文本定位器引导可变分辨率合并，在边界和文字处保留空间覆盖范围和更精细的细节。学习到的[REL]寄存器随后在解码器早期处理之后对视觉token进行剪枝。跨预算的训练方式使得每个骨干网络只需一个检查点。在三个骨干网络和13个基准测试上，Varest在每个评估预算下的平均性能保持率均优于15个基线方法，仅使用原始层平均图像token数量的5.56%即可保留全token性能的86.10%至92.33%。

    arXiv:2610.18690v1 Announce Type: cross  Abstract: Image-token compression must reduce computation without hurting the semantics. We introduce Varest, guided by semantic fungibility: tokens within a coherent semantic region can be summarized coarsely, while object boundaries and literal text require finer distinctions. Object and text localizers guide variable-resolution merging, preserving spatial coverage and finer detail at boundaries and text. Learned [REL] registers then prune visual tokens after early decoder processing. Training across budgets enables one checkpoint per backbone. Across three backbones and 13 benchmarks, Varest outperforms 15 baselines in mean performance retention at every evaluated budget, retaining 86.10-92.33% of full-token performance with only 5.56% of the original layer-averaged image-token count.
    
[^261]: 随时间的压缩：可分性、平稳性与容量的相互作用

    Compression Through Time: An Interplay of Separability, Stationarity and Capacity

    [https://arxiv.org/abs/2610.16660](https://arxiv.org/abs/2610.16660)

    该论文揭示了循环网络中的信息压缩现象由类别可分性、时间平稳性和有限的表示容量这三种相互竞争的机制共同驱动。

    

    信息论压缩和几何压缩作为对泛化现象的可能解释已被广泛研究，然而它们在循环网络中的时间动态在很大程度上仍未被探索。我们使用离散化的LSTM——其有限的嵌入空间使得可以直接估计信息量——观察到在许多数据集的最后几个时间步上存在明显的压缩现象。我们通过三种相互竞争的机制来解释这一效应：(i) 类别可分性允许在最终时间步实现更强的压缩；(ii) 共享的循环动力学将这种压力传播到具有相似激活状态的时间步上，使压缩程度取决于时间平稳性；(iii) 有限的表示容量会对抗持续的压缩，因为它限制了可用于后续计算的信息量。我们通过受控的合成干预和真实世界的序列任务系统地测试了这些机制。我们的结果识别了（摘要在此处截断）

    arXiv:2610.16660v2 Announce Type: replace  Abstract: Information-theoretic and geometric compression have been widely studied as possible explanations for generalization. Yet their temporal dynamics in recurrent networks remain largely unexplored. Using discretized LSTMs, whose finite embedding space enables direct estimates of information quantities, we observe pronounced compression across the final timesteps of many datasets. We explain this effect through three competing mechanisms: (i) class separability permits stronger compression at the final timestep; (ii) shared recurrent dynamics propagate this pressure across timesteps with similar activation states, making its extent depend on temporal stationarity; and (iii) limited representational capacity opposes prolonged compression because it restricts information available for subsequent computation. We systematically test these mechanisms through controlled synthetic interventions and real-world sequence tasks. Our results identif
    
[^262]: 面向鲁棒离线元强化学习的悲观潜任务感知优化

    Pessimistic Latent Task-aware Optimization for Robust Offline Meta-Reinforcement Learning

    [https://arxiv.org/abs/2610.15428](https://arxiv.org/abs/2610.15428)

    提出PLATO方法，基于任务潜变量位于低维流形上的假设，通过从训练质心向外扰动潜变量来合成反事实任务，让策略在训练中接触支持域外潜变量，从而实现对外推任务更鲁棒的离线元强化学习。

    

    基于上下文的离线元强化学习（COMRL）的进展几乎完全依赖于更好的任务表征，这种表征将任务身份与行为策略产生的伪影分离开来，使潜变量仅反映任务本身。其隐含的期望是，一个干净的表征能够迁移到支持域外（参数外推）的任务上。然而，在这类任务上，编码器会产生位于训练支持域之外的潜变量，而策略从未在这些潜变量上进行过训练。我们提出悲观潜任务感知优化（PLATO）来填补这一空白，该方法在训练期间让策略接触支持域外的潜变量。PLATO基于一个温和的流形假设：任务潜变量位于低维流形上，而支持域外的任务距离训练质心更远。它利用这一几何特性，通过将推断出的潜变量从质心向外扰动来合成反事实任务。随后，一个学习得到的解码器集成会推出相应轨迹（摘要在此处截断）。

    arXiv:2610.15428v1 Announce Type: cross  Abstract: Context-based offline meta-reinforcement learning (COMRL) has advanced almost entirely through better task representations, which separate task identity from artifacts of the behavior policy so that the latent reflects the task alone. The implicit expectation has been that a clean representation will transfer to out-of-support (parameter-extrapolation) tasks. On such a task, however, the encoder produces a latent outside the training support, where the policy has never been trained. We address this gap with Pessimistic Latent Task-aware Optimization (PLATO), which exposes the policy to off-support latents during training. PLATO assumes a mild manifold hypothesis: task latents lie on a low-dimensional manifold, and out-of-support tasks lie farther from the training centroid. It exploits this geometry by perturbing inferred latents outward from the centroid to synthesize counterfactual tasks. A learned decoder ensemble then rolls out sho
    
[^263]: 探索大语言模型中基于深度的动态推理的越狱漏洞

    Exploring Jailbreak Vulnerability Across Depth-Based Dynamic Inference in Large Language Models

    [https://arxiv.org/abs/2610.15207](https://arxiv.org/abs/2610.15207)

    本文首次系统研究了基于深度的动态推理对大语言模型越狱鲁棒性的影响，发现浅层退出始终比深层退出对越狱攻击更具鲁棒性，揭示了动态推理灵活性所带来的安全风险。

    

    大语言模型（LLMs）已被证明计算成本高昂且容易受到越狱攻击，即对抗性提示可以绕过安全对齐并诱导产生有害响应。为了解决这种低效问题，研究者提出了一系列技术以实现高效的大语言模型推理。基于深度的动态推理通过允许从中间层生成输出，已展现出令人鼓舞的效果，它提供了根据可用资源预算灵活调整模型行为的能力。然而，这种灵活性如何影响大语言模型对越狱攻击的鲁棒性仍尚未被探索。在本文中，我们通过对基于深度的动态推理对大语言模型对抗越狱攻击鲁棒性的影响进行全面研究来填补这一空白。具体而言，我们得出以下关键观察：1) 浅层退出始终比深层退出对越狱攻击更具鲁棒性；2) 退出深度……

    arXiv:2610.15207v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have been shown to be computationally expensive and vulnerable to jailbreak attacks, in which adversarial prompts bypass safety alignment and induce harmful responses. To address this inefficiency, a range of techniques have been proposed to enable efficient LLM inference. Depth-based dynamic inference has demonstrated promising results by enabling outputs to be generated from intermediate layers. It offers the flexibility to adjust model behavior based on the available resource budget. However, how this flexibility affects the robustness of LLMs against jailbreak attacks remains unexplored. In this paper, we address this gap via a comprehensive investigation into the impact of depth-based dynamic inference on the robustness of LLMs against jailbreak attacks. Specifically, we make the following key observations: 1) Shallow exits are consistently more robust to jailbreak attacks than deeper ones; 2) Exit dep
    
[^264]: DisCo代数：为什么分解与组合并非互逆运算

    DisCo Algebra: Why Disintegration and Combination Are Not Inverses

    [https://arxiv.org/abs/2610.15006](https://arxiv.org/abs/2610.15006)

    本文提出DisCo代数框架，用对称余操作刻画联合概率的分解、用非单位部分操作刻画组合，证明“组合∘分解”在可允许输入上是恒等操作，而“分解∘组合”则是一个有信息损失的重正化算子，从而揭示了分解与组合并非互逆。

    

    我们提出了DisCo代数——一个建立在裸族 $\mathcal{P}_n=\mathrm{Prob}([0,1]^n)$ 之上的联合概率组合式框架，由一个对称余操作（disintegration，即分解）和一个非单位部分对称操作（combination，即限制在分解像上的组合）构成。该框架不需要概率积分变换，不需要外部汇总函数，也不需要用条件核对对象进行富集。分解一侧使用分块联合累积分布函数（CDF）作为内在的、由测度决定的一维摘要，从而实现有损压缩；组合一侧读取相同的摘要，并通过正则条件测度进行重构，在可允许输入上实现无损嵌入。我们证明：先组合后分解在可允许输入上是恒等操作，但先分解后组合则不是——后者定义了一个重正化算子 $R_{\mathcal{I}}=\text{Combine}\circ\text{Disintegrate}$。该算子的信息损失……

    arXiv:2610.15006v2 Announce Type: replace-cross  Abstract: We present DisCo algebra---a compositional framework for joint probability built from a symmetric cooperad (disintegration) and a non-unital partial symmetric operad (combination restricted to the image of disintegration) on the bare family $\mathcal{P}_n=\mathrm{Prob}([0,1]^n)$. No probability integral transform, no external summary functions, no enrichment of objects with conditional kernels. The disintegration side uses block-wise joint CDFs as intrinsic, measure-determined 1D summaries, yielding a lossy compression. The combination side reads the same summaries and reconstructs via regular conditional measures, yielding a lossless embedding on admissible inputs. We prove that combination followed by disintegration is the identity on admissible inputs, but disintegration followed by combination is not---it defines a renormalization operator $R_{\mathcal{I}}=\text{Combine}\circ\text{Disintegrate}$. The information loss of thi
    
[^265]: 当场抓获：探针有效检测破坏行为并捕捉未言明的欺骗

    Caught in the Act: Probes Effectively Detect Sabotage and Catch Unverbalized Deception

    [https://arxiv.org/abs/2610.12445](https://arxiv.org/abs/2610.12445)

    该研究通过构建迄今最大的欺骗数据集并设计可跨层、跨标记聚合信息的新型探针架构，使白盒探针在 SHADE-Arena 上以 98.8% 的 AUC 超越前沿文本监控基线，甚至能检测出仅凭上下文无法察觉的“内省式欺骗”。

    

    近期发生的一些事件凸显了监控大语言模型（LLM）智能体的挑战，以及模型欺骗人类的危险。我们展示了基于探针的白盒欺骗检测可以扩展到前沿监控场景，方法包括收集迄今为止最大的欺骗数据集用于训练探针，并引入一种新颖的探针架构，该架构能够跨多个层和标记（token）聚合信息。我们的探针在 SHADE-Arena 上达到了 98.8% 的 AUC，超越了 Opus 5.5 的文本监控基线，并且随着底层模型规模的扩大，其检测效果也随之提升。为了将我们的探针推向极限，我们在几种仅凭上下文无法判断是否存在欺骗的案例上对其进行了测试。在这些我们称为“内省式欺骗”的案例中，真实情况只能通过仔细的引导性询问或对模型训练数据的透彻了解才能确定。在其中一项评估中，我们展示了探针能够区分包含……的对话记录（原文摘要在此处截断）。

    arXiv:2610.12445v1 Announce Type: cross  Abstract: Recent incidents have highlighted the challenge of monitoring LLM agents and the danger of models deceiving people. We show that white-box deception detection via probes can be scaled up to frontier monitoring settings by collecting the largest deception dataset to date for training probes and introducing a novel probe architecture which can aggregate information across many layers and tokens. Our probes achieve 98.8% AUC in SHADE-Arena, surpassing an Opus 5.5 text-monitoring baseline, and show improved efficacy as the underlying model is scaled up. To push our probes to their limit, we test them on several cases where deception cannot be determined from the context alone. In these cases, which we refer to as introspective deception, the ground truth can only be determined through careful elicitation or thorough knowledge of a model's training data. In one such evaluation, we show that probes can distinguish transcripts containing a mo
    
[^266]: 利用势函数估计自回归语言模型下的广义期望

    Estimating great expectations under autoregressive language models with potentials

    [https://arxiv.org/abs/2610.11399](https://arxiv.org/abs/2610.11399)

    本文提出利用采样时免费获得的下一词元条件概率构造势函数，对语言模型下检验泛函的期望进行估计，在计算成本相近的情况下显著降低了估计方差。

    

    语言模型的许多应用所依赖的并非单个样本，而是模型下某个检验泛函的期望。可靠地估计这类期望在计算上可能非常昂贵。在本文中，我们展示了如何利用采样过程中作为副产品自然获得的下一词元条件概率来使估计更加高效。我们通过势函数来实现这一点：势函数是定义在前缀上的实值函数，能够对检验泛函进行加性分解。我们构造了一个估计器，其方差取决于所选择的势函数，并推导出势函数能够降低该方差的充分条件。随后，我们为若干估计对象和应用开发了实用的势函数，并在计算成本相当的情况下，在多个估计对象上展示了显著的方差缩减效果。

    arXiv:2610.11399v1 Announce Type: new  Abstract: Many applications of language models hinge not on individual samples but on the expectation of a test functional under the model. Estimating such expectations reliably can be computationally expensive. In this paper, we show how to make estimation more efficient by exploiting the next-token conditional probabilities which are available as a by-product of sampling. We do so through potentials: real-valued functions on prefixes that decompose the test functional additively. We construct an estimator whose variance depends on the chosen potential, and derive conditions under which a potential reduces this variance. We then develop practical potentials for several estimands and applications, and demonstrate substantial variance reductions across several estimands at comparable computational cost.
    
[^267]: SciExam for ENSO：AI智能体能构建气候模型吗？

    SciExam for ENSO: Can AI Agents Build Climate Models?

    [https://arxiv.org/abs/2610.10513](https://arxiv.org/abs/2610.10513)

    该论文提出了SciExam for ENSO基准测试，让AI智能体在六小时内仅凭真实观测数据自主构建ENSO低阶随机气候模型，并用隐藏评分器检验模型能否复现统计特性、恢复隐变量和预测保留年份，结果显示十二个智能体系统中有六个的模型优于已发表的模型。

    

    语言模型智能体越来越多地被要求开展开放式科学研究，然而对其结果的评估通常是对照已知答案、评分标准或语言模型审稿人，这些方法都无法判断一个新的科学模型是否有效。“厄尔尼诺-南方涛动AI科学考试”是一个基准测试，要求智能体基于真实观测数据构建厄尔尼诺-南方涛动（ENSO，年际气候变率的主导模态）的低阶随机模型。在六小时的时间预算内，智能体处理观测数据、编写自己的诊断程序（随后被冻结），并仅以这些诊断结果作为反馈来开发模型。随后由隐藏的评分器测试该模型能否复现ENSO的统计特性、恢复未观测到的变量以及对未纳入训练的年份进行预测，并以同样的方式对已发表的模型进行评分。在十二个智能体系统中，有六个产生的模型得分高于已发表的模型，主要得益于更好的（重建）。

    arXiv:2610.10513v1 Announce Type: cross  Abstract: Language-model agents are increasingly asked to carry out open-ended scientific research, yet their results are usually graded against a known answer, a rubric, or a language-model reviewer, none of which can tell whether a new scientific model is valid. The AI Science Exam for El Nino-Southern Oscillation (SciExam for ENSO) is a benchmark in which agents build low-order stochastic models of ENSO, the dominant mode of interannual climate variability, from real observations. Within a six-hour budget, agents process the observations, write their own diagnostics, which are then frozen, and develop a model using only these diagnostics as feedback. Hidden graders then test whether the model reproduces ENSO's statistics, recovers unobserved variables, and forecasts held-out years, and score a published model in the same way. Across twelve agent systems, six produce models that score higher than the published model, mainly through better reco
    
[^268]: 最小见证强化学习

    Minimal Witness Reinforcement Learning

    [https://arxiv.org/abs/2610.07226](https://arxiv.org/abs/2610.07226)

    本文提出最小见证强化学习（MWRL），利用基于集合并集覆盖损失的信用分配机制，仅凭单一黑盒验证器的信号即可同时实现解的最小性与多个备选解的恢复。

    

    “足以产生某一结果的不可约条件是什么？”这是计算与科学领域中最常出现的核心问题之一。其答案——最小充分见证——正是我们所称的解释、机制与原因。这类问题通常需要找出多个最小见证，然而标准的强化学习方法可能只能揭示一个解或冗余的解。我们将该问题形式化为最小见证识别，并提出了最小见证强化学习（MWRL）。MWRL 对从策略中采样得到的、经成功验证的提议所认证的集合取并集，并根据“若缺少该提议，组并集将会损失的覆盖范围”为每个提议分配信用。这一直接源于问题定义的信用分配机制，仅凭一个黑盒验证器的单个比特信号，便统一了对最小性与备选解恢复的双重需求。基于该原理，我们推导出一个值迭代规划器，能够恢复……（摘要原文在此处截断）

    arXiv:2610.07226v1 Announce Type: cross  Abstract: ``What are the irreducible conditions that are sufficient to produce an outcome?'' is one of the most common questions that recur across computation and science. Its answers, the minimal sufficient witnesses, are what we mean by explanations, mechanisms and reasons. These problems usually ask for multiple minimal witnesses, yet standard RL methods may reveal only one solution or redundant ones. We formalize this problem as minimal-witness identification and introduce Minimal-Witness Reinforcement Learning (MWRL). MWRL takes the union of the sets certified by successful proposals sampled from the policy and credits each proposal for the coverage the group union would lose without that proposal. This credit assignment, derived directly from the problem definition, unifies the demands for minimality and recovery of alternatives from a single black-box verifier bit. Under this principle, we derive a value iteration planner that recovers th
    
[^269]: TrustmeWatcher：一款用于职场微感知与可解释幸福感反馈的应用程序

    TrustmeWatcher: An Application for Workplace Micro-Sensing and Explainable Well-Being Feedback

    [https://arxiv.org/abs/2610.06657](https://arxiv.org/abs/2610.06657)

    TrustmeWatcher 是一款整合操作系统级活动感知数据与问卷自我报告、并借助本地运行的人工智能模型生成可解释幸福感反馈的职场微感知应用程序。

    

    职场感知研究通常将长期采集的行为记录与自我报告相结合，然而收集这些数据的工具往往与返回结果的界面相互分离。我们提出了 TrustmeWatcher，这是为 TRUST-ME 项目构建的一款应用程序，旨在打通这一工作流程。TrustmeWatcher 复用 ActivityWatch 的操作系统级监视器来采集计算机活动数据，并在此基础上添加了自身的应用层。它将采集到的行为记录转化为交互式屏幕使用时间仪表盘，同步用户在 StreamDeck 上完成的简短问卷的回答，并将问卷与视频亮点一并呈现。活动记录与自我报告被对齐为带标签的记录，用于模型开发。本文的范围仅限于将 ActivityWatch 数据作为模型输入。人工智能利用这些活动记录来预测六个归一化的状态分数和一个总体幸福感分数。训练好的模型在本地运行，仪表盘……（原文在此处截断）

    arXiv:2610.06657v2 Announce Type: replace  Abstract: Workplace sensing studies combine long-running behaviour traces with self-reports, yet the tools that collect those data often sit apart from the interface that returns results. We present TrustmeWatcher, the application built for the TRUST-ME project to connect this work. TrustmeWatcher reuses ActivityWatch's OS-level watchers for computer-activity collection and adds its own application layer. It turns the collected traces into an interactive screen-time dashboard, synchronizes responses from short questionnaires completed on the StreamDeck, and presents questionnaires alongside video highlights. Activity records and self-reports are aligned into labelled records for model development. The scope of this paper is limited to ActivityWatch data as model input. Artificial intelligence (AI) uses these activity records to predict six normalized state scores and an overall well-being score. The trained model runs locally, and the dashboar
    
[^270]: 物理是最好的老师：面向混沌动力学时不变算子的一致性学习

    Physics is the Best Teacher: Consistency Learning for Time-Invariant Operators of Chaotic Dynamics

    [https://arxiv.org/abs/2610.04108](https://arxiv.org/abs/2610.04108)

    提出通过推导时不变演化算子必须满足的物理一致性方程，将物理知识嵌入一致性学习，实现对混沌系统长期行为的单次跳跃式高效预测，在精度与时间外推能力上优于逐步推进的传统方法。

    

    加速预测混沌系统的长期行为在科学计算中至关重要。然而，现有方法依赖于数值求解器或自回归模型，每次只能推进一小步，这使得长时间跨度的预测代价高昂。我们转而将该问题视为学习系统的时间不变演化算子，该算子能够在单次评估中将状态跳越一个大的时间跨度。为此，我们推导出时不变算子必须满足的一致性方程，包含物理时间下的微分与复合目标。这些方程还将学习到的算子与物理规定的瞬时动力学联系起来，从而实现物理知识在一致性学习中的嵌入。在五个混沌系统上的实验表明，经物理蒸馏的一致性学习使短期轨迹和长期统计都更加准确。学习到的算子具备时间外推能力，且所需步数仅为原来的十分之一。

    arXiv:2610.04108v2 Announce Type: replace  Abstract: Accelerating the prediction of long-term behavior in chaotic systems is crucial in scientific computing. However, existing methods rely on numerical solvers or autoregressive models that advance one small step at a time, which makes long horizons expensive. We instead view this problem as learning the system's time-invariant evolution operator, which jumps the state across a large time span in a single evaluation. To this end, we derive the consistency equations a time-invariant operator must satisfy, with differential and compositional objectives in physical time. These equations also connect the learned operator to the physics-prescribed instant dynamics, enabling physics embedding in consistency learning. Across five chaotic systems, we find that physics-distilled consistency makes both short-term trajectories and long-term statistics more accurate. The learned operator survives temporal extrapolation and requires one-tenth as man
    
[^271]: 预测器引导的潜空间密码子优化以最大化蛋白质表达

    Predictor-Guided Latent Space Codon Optimization for Maximizing Protein Expression

    [https://arxiv.org/abs/2610.03098](https://arxiv.org/abs/2610.03098)

    提出潜空间密码子优化方法LSCO，通过将序列映射到预训练mRNA语言模型的潜空间，将离散的密码子优化问题转化为可梯度搜索的连续问题，并结合不确定性感知的表达预测器、最小自由能正则化、自然性先验和约束解码，以最大化蛋白质表达。

    

    密码子优化是通过选择同义密码子来提高mRNA翻译效率和蛋白质表达水平的过程，是治疗性蛋白质生产和mRNA疫苗研发的核心环节，但它仍然是一个难题。其设计空间是离散的且组合规模巨大，这使得基于梯度的方法无法适用，而现有工具依赖启发式代理指标（如密码子适应指数或GC含量），这些指标难以真实反映实际表达水平。我们提出了潜空间密码子优化（LSCO），通过将序列映射到预训练mRNA语言模型的潜空间中，将这一离散问题转化为连续问题，从而实现高效的基于梯度的搜索。LSCO结合了四个组件：来自不确定性感知预测器的数据驱动表达目标函数、用于结构稳定性的最小自由能（MFE）正则化项、来自蛋白质到密码子反向翻译模型的自然性先验，以及确保蛋白质保真度的约束解码。

    arXiv:2610.03098v1 Announce Type: new  Abstract: Codon optimization, the process of selecting synonymous codons to improve mRNA translation efficiency and protein expression, is central to therapeutic protein production and mRNA vaccines, yet it remains a hard problem. The design space is discrete and combinatorially large, precluding gradient-based methods, and existing tools rely on heuristic proxies (e.g., Codon Adaptation Index or GC-content) that poorly capture true expression. We introduce Latent-Space Codon Optimization (LSCO), which recasts this discrete problem as a continuous one by mapping sequences into the latent space of a pretrained mRNA language model, enabling efficient gradient-based search. LSCO combines four components: a data-driven expression objective from an uncertainty-aware predictor, a Minimum-Free-Energy regularizer for structural stability, a naturalness prior from a protein-to-codon back-translation model, and constrained decoding for protein fidelity. On 
    
[^272]: 面向未测量混杂下因果效应估计的邻近平衡方法

    Proximal Balancing for Causal Effect Estimation under Unmeasured Confounding

    [https://arxiv.org/abs/2609.40051](https://arxiv.org/abs/2609.40051)

    提出邻近平衡方法，将经典协变量平衡思想扩展到仅能通过代理变量观测的混杂因素，无需指定代理角色、求解逆问题或依赖潜变量模型，即可实现未测量混杂下的因果效应估计。

    

    从观测数据中估计因果效应是科学与政策领域的核心问题，但当混杂因素未被测量时，因果效应无法被识别。邻近因果推断通过未测量混杂因素的代理变量来应对这一问题。然而，现有的基于代理的方法要么指定代理角色并求解一个逆问题，该问题是不适定的，且在高维代理变量下难以估计；要么使用潜变量模型，该模型假设学习到的潜变量与隐藏的混杂因素相匹配，一旦不匹配就会产生偏差。为了应对这些挑战，我们提出了邻近平衡方法。它将经典的协变量平衡思想推广到只能通过代理变量观测的混杂因素：它学习协变量和代理变量的一个低维摘要表示，使处理组之间具有可比性，然后对该摘要进行调整。该方法不需要指定代理角色、求解逆问题或使用潜变量模型。我们给出（原文摘要在此处截断）

    arXiv:2609.40051v1 Announce Type: new  Abstract: Estimating causal effects from observational data is central to science and policy, but the effects are not identified when confounders are unmeasured. Proximal causal inference addresses this problem with proxies of the unmeasured confounders. However, existing proxy-based approaches either designate proxy roles and solve an inverse problem, which is ill-posed and hard to estimate with high-dimensional proxies, or use a latent-variable model, which assumes that the learned latent variable matches the hidden confounder and leaves bias when it does not. To address these challenges, we introduce proximal balancing. It carries the classical idea of covariate balancing to confounders that are observed only through proxies: it learns a low-dimensional summary of the covariates and proxies that makes the treatment groups comparable, and then adjusts for this summary. It needs no designated proxy roles, inverse problem, or latent model. We give
    
[^273]: 泛化信号未必是模型选择信号

    A Generalisation Signal Need Not Be a Model-Selection Signal

    [https://arxiv.org/abs/2609.39099](https://arxiv.org/abs/2609.39099)

    该研究发现，尽管受Hessian启发的新型仅前向传播代理指标在大多数任务上与泛化差距相关，但这种泛化信号并不能可靠地识别出部署环境下最优的模型，表明泛化信号未必能作为模型选择的信号。

    

    计算生物学中的模型选择通常依赖于从训练环境中抽取的验证数据，即使部署环境超出了该范围。当验证数据不再能保留哪个模型最优的信息时，一个自然的替代方法是利用训练好的网络本身的属性来对候选模型进行排序。我们使用一种受Hessian范数启发的新型仅前向传播代理指标，结合常见的Hessian度量指标，在分子性质、蛋白质适应度和药物反应任务上测试了这一想法。与我们的假设相反，随着验证Spearman相关性的恶化，几何指标并没有变得更有用：增强验证对某些分布偏移有帮助，但对另一些则有显著损害。更令人惊讶的是，即使在Hessian迹和顶部特征值关系较弱或发生反转的情况下，该代理指标在大多数任务上仍与泛化差距相关，然而这一信号并不能可靠地识别出部署环境下最优的模型。曲率界不必保留跨分布的模型排序信息。

    arXiv:2609.39099v1 Announce Type: cross  Abstract: Model selection in computational biology often relies on validation data drawn from the training regime, even when deployment lies outside it. When validation no longer preserves which model is best, a natural alternative is to rank candidates using properties of the trained network itself. We test this idea using a novel, forward-only proxy motivated by the norm of the Hessian, alongside common Hessian measures, across molecular property, protein fitness, and drug-response tasks. Contrary to our hypothesis, geometry does not become more useful as validation Spearman correlation deteriorates: augmenting validation helps some shifts but significantly harms others. More surprisingly, the proxy still correlates with generalisation gap on most tasks even when Hessian trace and top-eigenvalue relationships are weak or reversed, yet this signal does not reliably identify the deployment-best model. A curvature bound need not preserve cross-mo
    
[^274]: 通过对抗探针训练实现对齐而不丧失可监控性

    Alignment via Training Against Probes Without Losing Monitorability

    [https://arxiv.org/abs/2609.38645](https://arxiv.org/abs/2609.38645)

    该研究提出探针引导微调方法，发现使用持续更新的探针作为直接训练信号能有效减少模型有害行为并提升诚实性，同时保持模型效用而不丧失可监控性，而静态探针则容易被模型利用规避。

    

    模型通常基于其观察到的输出进行对齐，使用示范数据、偏好数据或奖励信号。这些目标会奖励那些看起来对齐的响应。更强大的模型可能学会满足这些目标而不内化预期行为，例如在训练过程中伪装顺从。当目标定义在模型内部而非输出上时，这种表面顺从可能更难实现。因此，我们研究了探针引导的微调方法，使用检测模型激活中不良属性的探针作为直接训练信号。我们在两个对齐目标（无害性和诚实性）下评估了线性和非线性探针，以及每层不同数量的探针。我们发现，针对训练期间不更新的探针进行训练是一个容易被利用的目标，而持续更新的探针在保持效用的同时，能够显著减少有害性并提高诚实性。

    arXiv:2609.38645v1 Announce Type: cross  Abstract: Models are usually aligned based on their observed outputs, using demonstrations, preference data, or reward signals. These objectives reward responses that look aligned. More capable models may learn to satisfy them without internalizing the intended behavior, for example by faking compliance during training. Such superficial compliance could be harder when the objective is defined on model internals rather than outputs. Therefore, we study probe-guided fine-tuning, using probes that detect undesired properties in model activations as a direct training signal. We evaluate linear and non-linear probes with different numbers of probes per layer across two alignment objectives: harmlessness and honesty. We find that training against probes that do not update during training is an easily exploitable objective, while continuously updated probes substantially reduce harmfulness and improve honesty while preserving utility. Probe-guided fine
    
[^275]: 可解释但脆弱？概念瓶颈模型在几何-语义扰动下的鲁棒性

    Interpretable but Fragile? Robustness of Concept Bottlenecks under Geometric-Semantic Perturbations

    [https://arxiv.org/abs/2609.38625](https://arxiv.org/abs/2609.38625)

    该论文提出了一个基于生成器的统一评估框架，首次将潜空间中的连续几何扰动与概念空间中的离散语义干预区分开来，从经验性与可认证两个层面系统比较概念瓶颈模型与标准分类器的鲁棒性，从而调和了此前相互矛盾的研究结论。

    

    概念瓶颈模型（CBMs）旨在提供可解释的中间表示，然而此类瓶颈如何影响模型鲁棒性仍不明确，现有研究报告的结果不一致，有时甚至相互矛盾。我们认为，这些分歧源于将不同的鲁棒性概念与扰动模式混为一谈，而非对概念瓶颈模型本身存在根本性分歧。为厘清这些因素，我们提出了一个基于生成器的评估框架，使标准分类器与概念瓶颈模型能够在两类截然不同的扰动下进行受控比较：潜空间中的连续几何扰动与概念空间中的离散语义干预。在该框架内，我们一方面通过预测与概念级敏感性指标对鲁棒性进行经验性评估，另一方面利用潜空间与概念空间中的随机平滑进行可认证的鲁棒性评估。在一系列实验中，我们调和了此前……（原文摘要在此处截断）

    arXiv:2609.38625v1 Announce Type: cross  Abstract: Concept Bottleneck Models (CBMs) are designed to provide interpretable intermediate representations, yet how such bottlenecks affect robustness remains unclear, with existing studies reporting mixed and sometimes contradictory findings. We argue that these discrepancies arise from conflating different robustness notions and perturbation regimes, rather than from fundamental disagreements about CBMs themselves. To disentangle these factors, we introduce a generator-based evaluation framework that enables controlled comparisons between standard classifiers and CBMs under two distinct perturbation types: continuous geometric perturbations in latent space and discrete semantic interventions in concept space. Within this framework, we evaluate robustness both empirically, via prediction and concept-level sensitivity metrics, and certifiably, using randomized smoothing in latent and concept spaces. Across experiments, we reconcile previously
    
[^276]: SAGE：语义音频生成编码器

    SAGE: Semantic Audio Generative Encoder

    [https://arxiv.org/abs/2609.32755](https://arxiv.org/abs/2609.32755)

    SAGE是一个仅1.05亿参数、只用公开音乐训练的紧凑音频自编码器，通过蒸馏预训练音频-文本模型的嵌入来构建语义潜在空间，在推理速度、重建质量和语义结构三方面同时超越了现有的更大模型。

    

    音频自编码器将波形压缩为紧凑的潜在表示，作为原始音频与下游模型之间的接口。现有系统需要在重建质量、潜在空间的语义结构和推理速度之间进行三方权衡，通常只能以牺牲其他方面为代价来优化其中一两个方面。本文提出了SAGE（语义音频生成编码器）：一个紧凑的变分自编码器，仅使用公开可用的音乐进行训练，通过从预训练的音频-文本模型中蒸馏嵌入向量来塑造其潜在空间。这个1.05亿参数的模型运行速度比Stable Audio Open快约3倍，并达到了SAME-L（一个参数量是其8倍、速度慢2倍以上的自编码器）的听感测试质量，同时在重建的客观感知指标和分布指标上超越了上述两个模型。此外，它在全部十九个潜在语义探测任务上都达到了最先进的水平，无论是域内还是（原文摘要到此截断）。

    arXiv:2609.32755v3 Announce Type: replace-cross  Abstract: Audio autoencoders compress waveforms into compact latent representations that serve as the interface between raw audio and downstream models. Current systems navigate a three-way trade-off between reconstruction quality, semantic structure of the latent space, and inference speed, typically favoring one or two of these at the expense of the others. This paper introduces SAGE, Semantic Audio Generative Encoder: a compact variational autoencoder, trained solely on publicly available music, that shapes its latent by distilling embeddings from a pretrained audio-text model. This 105M-parameter model runs about 3x faster than Stable Audio Open and reaches the listening-test quality of SAME-L, an autoencoder 8x larger and over 2x slower, while surpassing both on objective perceptual and distributional metrics of reconstruction. Furthermore, it sets the state of the art on all nineteen probing tasks of latent semantics, in domain and
    
[^277]: SoLiD26：用于机器学习原子间势的第一性原理固液界面数据集

    SoLiD26: A First Principles Solid-Liquid Interface Dataset for Machine-learned Interatomic Potentials

    [https://arxiv.org/abs/2609.28013](https://arxiv.org/abs/2609.28013)

    SoLiD26是一个包含1540万个第一性原理原子结构、涵盖15种化学元素的固液界面数据集，可用于训练和评估面向电化学、催化和腐蚀等应用的机器学习原子间势。

    

    面向先进材料应用（如电化学、催化和腐蚀）中固液界面的机器学习原子间势（MLIPs），需要同时采样液体环境、固体以及界面本身的训练数据。我们提出了SoLiD26，这是一个经过精心筛选的固液界面数据集，包含1540万个第一性原理原子结构，体系最多达576个原子、涵盖15种化学元素，可用于训练和评估机器学习原子间势。这些结构汇编自固液界面研究中开展的密度泛函理论（DFT）计算，其中大多数构型来自从头算分子动力学（AIMD）模拟。每条记录包含原子种类、坐标位置、模拟晶胞、周期性边界条件、势能和原子受力。SoLiD26包含水相货币金属（铜族金属）界面、电极-电解质体系以及精选的体相参考结构，均采用VASP计算（原文在此处截断）。

    arXiv:2609.28013v1 Announce Type: cross  Abstract: Machine-learned interatomic potentials (MLIPs) for solid-liquid interfaces in advanced materials applications, e.g., electrochemistry, catalysis and corrosion, require training data that samples both liquid environments, the solid and the interface itself. We present SoLiD26, a curated solid-liquid interface dataset, containing 15.4 million first-principles atomic structures with up to 576 atoms and 15 chemical elements for training and evaluating MLIPs. The structures were compiled from density functional theory (DFT) calculations performed in studies of solid-liquid interfaces, with most configurations originating from ab initio molecular dynamics (AIMD) simulations. Each record contains atomic species, positions, simulation cell, periodic boundary conditions, potential energy and atomic forces. SoLiD26 includes aqueous coinage metal interfaces, electrode-electrolyte systems, and selected bulk reference structures, calculated with VA
    
[^278]: 超越外观变化：面向VLA模型的任务语义动作校准

    Beyond Appearance Shifts: Task-Semantic Action Calibration for VLA Models

    [https://arxiv.org/abs/2609.23650](https://arxiv.org/abs/2609.23650)

    提出BAS-VLA任务语义动作校准框架，通过以语义破坏为中心的校准核心与证据门控的保持辅助模块，使VLA模型在场景外观变化时保持动作稳定，而在任务语义改变时产生显著不同的行为。

    

    视觉-语言-动作（VLA）模型在具身操作任务中已取得优异表现，但仍缺乏一种明确的机制来平衡行为稳定性与任务语义敏感性。我们识别出两种互补的失败模式：在任务保持型变化下（即任务语义不变，但场景外观发生变化，例如风格、光照、杂乱程度或指令改写），策略往往表现出不必要的动作漂移；相反，在语义破坏型变化下（即关键任务语义如目标物体或约束被改变），策略常常无法产生足够不同的行为，而是继续沿原轨迹执行。为填补这一空白，我们提出BAS-VLA，一个构建在冻结基础VLA之上的任务语义动作校准框架。BAS-VLA采用以破坏为中心的校准核心作为默认路径，并引入一个选择性的、由证据门控的保持辅助模块，该模块仅在……（摘要在此处截断）

    arXiv:2609.23650v1 Announce Type: cross  Abstract: Vision-language-action (VLA) models have achieved strong performance in embodied manipulation, but still lack a clear mechanism to balance behavioral stability with task-semantic sensitivity. We identify two complementary failure modes. Under task-preserving changes, where task semantics remain unchanged but scene appearance varies (e.g., style, illumination, clutter, or paraphrasing), policies often exhibit unnecessary action drift. Conversely, under semantic-breaking changes, where key task semantics such as the target object or constraint are altered, policies frequently fail to produce sufficiently distinct behaviors and instead follow the original trajectory. To address this gap, we propose BAS-VLA, a task-semantic action calibration framework built on top of a frozen base VLA. BAS-VLA adopts a breaking-centered calibration core as the default path, and introduces a selective evidence-gated preserving auxiliary that activates only
    
[^279]: 基于变换低秩分位数曲面的因果发现

    Causal Discovery via Transformed Low-Rank Quantile Surfaces

    [https://arxiv.org/abs/2609.16931](https://arxiv.org/abs/2609.16931)

    本文提出低秩分位数曲面（LRQS）二元因果模型，证明在因果方向上变换后的条件分位数曲面具有低秩结构从而保证因果方向的通用可识别性，并提出了一种交替进行秩约束近似与单调变换保序估计的非参数因果发现方法。

    

    我们提出了低秩分位数曲面（LRQS），这是一种二元因果模型，在因果方向上，条件分位数曲面的未知单调变换具有低秩函数分解形式。LRQS 涵盖了位置-尺度噪声模型和后非线性异方差噪声模型，同时允许多个分位数基函数来表示超出位置-尺度效应的分布变化。我们证明了 LRQS 的通用可识别性：变换后的分位数曲面在因果方向上是低秩的，而在相应约束下，反方向的可表示性仅出现在例外的、经过精细调整的原因边缘分布中。我们提供了一个简单而强大的因果评分方法，采用非参数拟合程序，在离散化分位数曲面的秩约束近似与未知单调变换的保序估计之间交替进行。在具有更高秩分布变化的合成机制上的实验表……（原文摘要在此处被截断）

    arXiv:2609.16931v1 Announce Type: cross  Abstract: We propose Low-Rank Quantile Surfaces (LRQS), a bivariate causal model in which, in the causal direction, an unknown monotone transformation of the conditional quantile surface admits a low-rank functional decomposition. LRQS subsumes location-scale noise models and post-nonlinear heteroscedastic noise models, while allowing multiple quantile bases to represent changes beyond location-scale effects. We prove generic identifiability of LRQS: the transformed quantile surface is low rank in the causal direction, whereas reverse representability under the corresponding constraints occurs only for exceptional, fine-tuned cause marginals. We provide a simple-yet-powerful causal score using a nonparametric fitting procedure that alternates between rank-constrained approximation of discretized quantile surfaces and isotonic estimation of the unknown monotone transformation. Experiments on synthetic mechanisms with higher-rank distributional sh
    
[^280]: 神经网络中的特征叠加：从理论到实践

    Feature Superposition in Neural Networks: From Theory to Practice

    [https://arxiv.org/abs/2609.06862](https://arxiv.org/abs/2609.06862)

    本综述系统梳理了神经网络中特征叠加的几何、学习与计算理论，并比较了从训练网络中恢复可解释特征的实用方法及其评估标准，从而将理论研究与实证证据连接起来。

    

    叠加是指神经网络所表示的特征数量超过其自身维度的现象。它为多义神经元提供了一种可能的解释，并推动了从神经激活中恢复可解释特征的方法的发展。理论模型通常从给定的一组输入特征以及关于这些特征值在不同输入间如何变化的假设出发，进而研究网络如何将这些值编码到低维的隐藏表示中。相比之下，实证工作旨在识别训练后的网络中所编码的特征，并确定它们在计算中的作用。在这篇综述中，我们回顾了叠加表示的几何结构、学习过程和计算方式，并解释了特征统计特性与解码器选择如何影响研究结论。为了将这些理论解释与训练网络中的证据联系起来，我们比较了恢复和分析特征的实用方法，并考察了这些方法的评估所能确立的内容。

    arXiv:2609.06862v1 Announce Type: new  Abstract: Superposition refers to neural networks representing more features than they have dimensions. It offers a possible explanation for polysemantic neurons and motivates methods for recovering interpretable features from neural activations. Theoretical models typically start with a given set of input features and assumptions about how their values vary across inputs, then study how a network encodes those values in a lower-dimensional hidden representation. Empirical work, by contrast, seeks to identify the features encoded in trained networks and determine their role in computation. In this survey, we review the geometry, learning, and computation of superposed representations, explaining how feature statistics and decoder choice affect the conclusions. To connect these theoretical accounts with evidence from trained networks, we compare practical methods for recovering and analyzing features and examine what their evaluations establish. Si
    
[^281]: 当保留约束发生冲突时：缓解表格数据中的遗忘-保留干扰

    When Retain Constraints Conflict: Mitigating Forget-Retain Interference in Tabular Data

    [https://arxiv.org/abs/2609.06786](https://arxiv.org/abs/2609.06786)

    该论文揭示了表格数据遗忘中独有的“模式引起遗忘-保留重叠”难题，并提出冲突感知遗忘方法（CAU）来缓解遗忘与保留之间的相互干扰。

    

    机器遗忘旨在移除指定训练数据对模型的影响，同时保持模型效用，但其在表格数据上的行为仍未得到充分探索。这一空白非常重要，因为表格预测被广泛应用于高风险领域，并且正通过记录序列化和模式感知提示越来越多地适配到语言模型中。我们识别出一个区分表格数据遗忘与自由文本或其他模态遗忘的关键挑战：模式引起的遗忘-保留重叠。在序列化的表格数据中，记录共享固定的列名/值槽位、相似的属性范围和共同的输出空间。因此，一个待遗忘的行可能存在依赖于相同高信号属性的邻近保留行，导致保留维护与遗忘所需的更新相互对立。受这种失败模式的启发，我们提出了冲突感知遗忘（CAU），一种模式感知的方法，能够减少遗忘-保留之间的干扰。

    arXiv:2609.06786v1 Announce Type: new  Abstract: Machine unlearning aims to remove the influence of designated training data while preserving model utility, but its behavior on tabular data remains underexplored. This gap is important because tabular prediction is widely used in high-stakes domains and is increasingly adapted to language models through record serialization and schema-aware prompting. We identify a key challenge that distinguishes tabular unlearning from unlearning in free-form text or other modalities: schema-induced forget-retain overlap. In serialized tabular data, records share fixed column-name/value slots, similar attribute ranges, and common output spaces. Consequently, a forget row may have nearby retain rows that rely on the same high-signal attributes, causing retain preservation to oppose the update required for forgetting. Motivated by this failure mode, we propose Conflict-Aware Unlearning (CAU), a schema-aware approach that reduces forget-retain interferen
    
[^282]: DrugReason：面向药物重定位的知识图谱与语言证据动态多视角推理

    DrugReason: Dynamic Multi-View Reasoning over Knowledge Graph and Language Evidence for Drug Repurposing

    [https://arxiv.org/abs/2609.06779](https://arxiv.org/abs/2609.06779)

    DrugReason 提出了一个动态多视角推理框架，将知识图谱的结构化多跳推理与大语言模型灵活的机理解释相结合，从而实现更可靠、更具事实依据的药物重定位预测。

    

    药物重定位旨在为现有化合物发现新的治疗用途，与从头药物发现相比，它为临床转化提供了更快、更具成本效益的路径。然而，候选药物-疾病对的空间极其庞大，且其潜在关系往往依赖于复杂的多跳生物机制，这使得难以可靠地预测哪些药物-疾病对代表真正的治疗关系。现有方法从两个方向入手：基于知识图谱的方法将整理好的生物医学证据组织为结构化关系网络，以进行有事实依据的多跳推理；而基于大语言模型的方法则利用预训练知识生成灵活的机理性解释。然而，二者单独使用均不充分——知识图谱受限于已观测到的图结构，而大语言模型缺乏事实依据并存在幻觉风险。为填补这一空白，我们提出了DrugReason，一个多视角推理框架……

    arXiv:2609.06779v1 Announce Type: cross  Abstract: Drug repurposing aims to identify new therapeutic uses for existing compounds and, compared with de novo drug discovery, offers a faster and more cost-effective path to clinical translation. However, the space of candidate drug-disease pairs is enormous and their underlying relationships often depend on complex multi-hop biological mechanisms, making it difficult to reliably predict which pairs represent true therapeutic relationships. Existing approaches tackle this from two directions: knowledge graph-based methods organize curated biomedical evidence into structured relational networks for grounded multi-hop reasoning, while LLM-based methods leverage pretrained knowledge to generate flexible mechanistic rationales. Yet neither is sufficient alone - KGs are confined to observed graph structure while LLMs lack factual grounding and risk hallucination. To address this gap, we propose DrugReason, a multi-view reasoning framework that i
    
[^283]: 接口诱导的轨迹审查

    Interface-Induced Trajectory Censoring

    [https://arxiv.org/abs/2609.03966](https://arxiv.org/abs/2609.03966)

    该论文发现智能体评估中的工具调用率可能被服务栈接口“审查”为零——即使模型实际发出了格式良好的调用，且2x2析因实验证明全部效应源于聊天模板与解析器之间的交互（修复任何一方都无效），仅更换服务适配器即可使同一模型得分从 0.00 跃升至 0.96/0.19。

    

    智能体评估报告的工具调用率是从服务栈中读取的。即使模型正在发出格式良好的调用，该数字也可能为零：接口在下游任何组件看到轨迹之前就将其“审查”掉了。在 BFCL v4 自身的数据、执行器和评分器上，保持权重、测试用例、解码方式和随机种子固定不变，仅更换服务适配器，同一模型的得分就可以是 0.00 或 0.96 / 0.19。对聊天模板和解析器进行的 2x2 析因实验精确定位了该效应：两个主效应都恰好为零，全部效应都落在交互作用中——没有任何组件是有缺陷的，只修复契约的一方毫无收效。在 tau-bench 的 115 个交互式零售任务上，同样的更换使服务器解析的调用从 0 增加到 636，到达工具执行环节的任务从 0 增加到 103。我们的探针实验在 Qwen2.5-Coder 的 21 倍规模范围内重现了这一漏斗效应：服务器在每个规模下都只能解析 0/100，而格式良好的发出调用在 32B 时上升到 80/100（审查后约 72……（摘要截断）

    arXiv:2609.03966v1 Announce Type: new  Abstract: Agent evaluations report a tool-call rate read off the serving stack. That number can be zero while the model is emitting well-formed calls: the interface censors the trajectory before anything downstream sees it.   On BFCL v4's own data, executor and scorer, holding weights, cases, decoding and seeds fixed and changing only the serving adapter, the same model scores 0.00 or 0.96 / 0.19. A 2x2 over chat template and parser locates the effect exactly: both main effects are exactly zero and all of it sits in the interaction -- no component is defective, and repairing one side of the contract buys precisely nothing. On tau-bench's 115 interactive retail tasks the same swap moves server-parsed calls from 0 to 636 and tasks reaching any tool execution from 0 to 103. Our probe reproduces the funnel across a 21x scale range of Qwen2.5-Coder: the server parses 0/100 at every size while well-formed emitted calls rise to 80/100 at 32B (~72 after c
    
[^284]: 表格基础模型懂物理吗？数据污染、物理单位与确定性极限

    Do Tabular Foundation Models Know Physics? Contamination, Units, and the Deterministic Limit

    [https://arxiv.org/abs/2609.02766](https://arxiv.org/abs/2609.02766)

    本文在316个物理方程采样的数据集上评估四种表格基础模型，发现其性能虽全面超越基线模型，但其贝叶斯先验既无法表示无噪声机制也无法表示物理单位，因此只能对物理数据进行插值而不能充当真正的物理模型。

    

    表格基础模型（TFMs）以语言模型填充文本的方式学习填充表格，而表格可以说是大多数物理测量数据所呈现的格式。它们在这一过程中学到了物理知识吗？由于它们在构造上是贝叶斯模型，因此问题在于其先验中包含什么。我们直接探测了这一问题，在从316个物理方程采样的数据集上（涵盖域内和域外），将四个模型（TabPFN-3、TabICLv2、TabDPT和Real-TabPFN-2.5）与六个基线模型进行了对比评估。TFMs无论开箱即用还是经过调优后都表现卓越。但我们证明了它们的先验既无法表示无噪声的物理机制，也无法表示物理单位，这正是它们能够对物理数据进行插值、却尚不能充当真正物理模型的原因。

    arXiv:2609.02766v1 Announce Type: new  Abstract: Tabular foundation models (TFMs) learn to fill in tables the way language models fill in text, and tables are arguably the format in which most physical measurement arrives. Did they learn any physics in the process? They are Bayesian by construction, so the question is what their prior contains. We probe it directly, evaluating four of them (TabPFN-3, TabICLv2, TabDPT and Real-TabPFN-2.5) against six baselines on datasets sampled from 316 physical equations, in and out of domain. TFMs dominate, out of the box and after tuning. But we show that their prior can represent neither a noiseless mechanism nor physical units, which is why they interpolate physics without yet being able to act as physical models.
    
[^285]: S^3martCirc：自监督智能电路发现

    S^3martCirc: Self-supervised Smart Circuit Discovery

    [https://arxiv.org/abs/2609.00755](https://arxiv.org/abs/2609.00755)

    提出S^3martCirc，一种自监督的智能电路发现方法，将电路发现与功能解释两个阶段统一起来，解决了组件重要性与功能角色相互依存这一被传统两阶段范式忽视的问题。

    

    大语言模型（LLMs）在从文本摘要到问答等多种任务中展现出了卓越的性能。尽管具备这些能力，其黑箱特性掩盖了内部决策过程。机制可解释性（MI）旨在通过将神经网络逆向工程为人类可理解的算法来解决这一问题。当前针对大语言模型的MI方法通常遵循两阶段范式：首先识别重要组件（电路发现），其中组件通常是个体节点，如注意力头或前馈神经元；其次确定它们在特定任务中所扮演的角色（功能解释）。然而，这种顺序化的方法忽视了一个基本洞见：组件的重要性与其功能角色本质上是相互依存的。将这两个阶段统一起来面临两个关键挑战：（1）功能角色往往与特定的节点或组件绑定，限制了……（摘要在此处截断）

    arXiv:2609.00755v1 Announce Type: new  Abstract: Large Language Models (LLMs) have demonstrated remarkable performance across diverse tasks, from text summarization to question answering. Despite these capabilities, their black-box nature obscures internal decision-making processes. Mechanistic interpretability (MI) aims to address this by reverse-engineering neural networks into human-understandable algorithms. Current MI approaches for LLMs typically follow a two-stage paradigm: first identifying important components (circuit discovery), where components are typically individual nodes such as an attention head or feedforward neuron, and second determining the role they play in a certain task (functional interpretation). However, this sequential approach overlooks a fundamental insight: a component's importance and its functional role are inherently codependent. Unifying these stages presents two key challenges: (1) functional roles are often tied to specific nodes or components, limi
    
[^286]: 欧几里得傅里叶神经算子

    Euclidean Fourier Neural Operators

    [https://arxiv.org/abs/2608.28425](https://arxiv.org/abs/2608.28425)

    提出欧几里得傅里叶神经算子（EFNO），通过将谱核参数化为物理波矢的连续函数，使神经算子能够跨不同形状和大小的周期域保持一致地学习与迁移，克服了传统FNO的域依赖性问题。

    

    傅里叶神经算子（FNO）为学习函数空间之间的映射提供了一个高效的框架，因为其构建方式使其独立于训练和评估时所用的网格分辨率。然而，FNO 并不独立于其所应用的周期域：其离散谱权重由整数傅里叶模数索引，而这些模数对应于物理波矢。当应用于不同的域时，相同的训练权重会在不同的波矢处起作用，FNO 会在不被察觉的情况下表示一个不同的算子。这使得 FNO 不适用于跨域迁移至关重要的任务。我们提出欧几里得傅里叶神经算子（EFNO）作为 FNO 的域独立替代方案。通过将谱核参数化为物理波矢的连续函数，EFNO 能够学习在不同形状和大小的周期域上保持一致作用的算子。我们在（摘要在此处截断）

    arXiv:2608.28425v1 Announce Type: new  Abstract: Fourier neural operators (FNOs) provide an efficient framework for learning mappings between function spaces as they are, by construction, independent of the grid resolution at which they are trained and evaluated. However, FNOs are not independent of the periodic domain they are applied to: their discrete spectral weights are indexed by integer Fourier mode numbers, which correspond to physical wavevectors. When applied to a different domain, the same trained weights act at different wavevectors, and the FNO silently represents a different operator. This makes FNOs unsuitable for tasks where transfer across domains is crucial. We propose Euclidean Fourier neural operators~(EFNOs) as a domain-independent alternative to FNOs. By parameterizing the spectral kernel as a continuous function of the physical wavevector, the EFNO can learn operators that act consistently across periodic domains of varying shape and size. We evaluate the EFNO on
    
[^287]: 分层指数-高斯混合模型用于观看时间分布预测

    Hierarchical Exponential-Gaussian Mixtures for Watch-Time Distribution Prediction

    [https://arxiv.org/abs/2608.23356](https://arxiv.org/abs/2608.23356)

    本文提出了一种分层指数-高斯混合模型，通过分层分解和正则化技术解决了现有EGMN模型的方差崩溃和组件冗余问题，从而提升了短视频推荐中观看时间预测的准确性。

    

    准确的观看时间（WT）预测是短视频推荐的重要需求。然而，WT分布具有接近零膨胀、长尾和多模态的特点。最近的指数-高斯混合网络（EGMN）对完整的条件WT分布进行建模，而非单一的点估计，并达到了最先进的性能。我们的大规模复现研究表明，EGMN容易遭受方差崩溃、组件冗余和无效组件的问题。我们提出了一种分层指数-高斯混合（HEGM）模型，通过分层跳过观看分解、基于KL的方差正则化、结构化初始化、去除强制高斯偏移和熵正则化来解决这些失败模式。在公开和大规模工业数据集上，HEGM提高了排序准确性和阈值事件预测，同时保持了竞争力的点估计准确性，并显著提升了整体性能。

    arXiv:2608.23356v1 Announce Type: new  Abstract: Accurate watch-time (WT) prediction is an important requirement for short-video recommendations. Yet WT distributions are near-zero-inflated, long-tailed and multimodal. The recent Exponential-Gaussian Mixture Network (EGMN) models the full conditional WT distribution rather than a single point estimate and achieves state-of-the-art performance. Our large-scale reproduction study reveals that EGMN is vulnerable to variance collapse, component redundancy, and inactive components. We propose a Hierarchical Exponential-Gaussian Mixture (HEGM) model that addresses these failure modes through a hierarchical skip-watch decomposition, KL-based variance regularization, structured initialization, removing the forced Gaussian shift and the entropy regularizer. Across public and large-scale industrial datasets, HEGM improves ranking accuracy and threshold-event prediction, while maintaining competitive point-estimation accuracy and substantially im
    
[^288]: 基于基元语言的主体智能体晶体结构设计

    Crystal-structure design by agentic AI in a language of motifs

    [https://arxiv.org/abs/2608.15900](https://arxiv.org/abs/2608.15900)

    MatEvolve利用可解释的基元语言和主体智能体，无需微调即达到新结构原型，比生成模型高出三倍以上，显著提升材料设计的外推能力。

    

    arXiv:2608.15900v1 公告类型：交叉 摘要：数据驱动的材料发现以内插方式比外推方式更可靠，且很少能达到新的结构类型。我们提出了MatEvolve，一个主体智能体AI框架，用于设计晶体，每个候选方案都附带明确的理由并进行测试。该智能体以可解释的“基元语言”进行推理，将每个晶体写成“基元轮廓”，描述构成晶体的重复几何模式——即“基元”。基元轮廓不仅作为材料的描述，还作为材料设计的媒介：智能体编辑轮廓并根据修改后的轮廓构建晶体，最有前景的候选方案通过第一性原理计算进行验证。应用于稀土稀缺永磁体的设计，MatEvolve——基于最先进的语言模型Claude Fable 5，无需微调——达到新结构原型的频率是生成模型的3倍以上。

    arXiv:2608.15900v1 Announce Type: cross  Abstract: Data-driven materials discovery interpolates more reliably than it extrapolates and seldom reaches new structure types. We present MatEvolve, an agentic-AI framework designing crystals, proposing each candidate with a stated rationale and testing it. The agent reasons in an interpretable \emph{language of motifs}, writing each crystal as a \emph{motif profile} that describes the recurring geometric patterns---the \emph{motifs}---composing it. The motif profile serves not merely as a description of a material but as the medium for material design: the agent edits the profile and constructs a crystal from the modified one, and the most promising candidates are validated by first-principles calculation. Applied to the design of rare-earth-lean permanent magnets, MatEvolve---built on the state-of-the-art language model Claude Fable~5 without fine-tuning---reaches new structural prototypes more than three times as often as generative models
    
[^289]: 学习重塑内部表示中的幂律各向异性

    Learning reshapes power-law anisotropy in internal representations

    [https://arxiv.org/abs/2608.15239](https://arxiv.org/abs/2608.15239)

    该论文通过精确求解两层线性网络的学习动态，揭示了幂律各向异性在内部表示中的形成机制，并发现特征学习机制下指数呈现非单调且多阶段演化，而惰性机制下行为不同。

    

    摘要：在从最先进的语言模型到小鼠大脑皮层等广泛的生物和人工神经系统中，内部表示中的幂律各向异性已被观察到。这种各向异性是高维信息处理的关键几何特性，并支撑着多种理论分析。然而，其从输入结构和任务驱动学习中涌现的机制仍不清楚。在此，我们通过精确求解一个宽两层线性神经网络在教师-学生设置下，具有幂律输入和教师结构的学习动态，来刻画这一形成过程。我们表明，在特征学习机制中，内部表示谱的局部幂律指数在训练过程中非单调演化，并在模式和训练时间上表现出多达四个不同的渐近区域。相比之下，在惰性机制中，该指数重新...

    arXiv:2608.15239v1 Announce Type: new  Abstract: Power-law anisotropy in internal representations has been observed across a wide range of biological and artificial neural systems, from state-of-the-art language models to the mouse cerebral cortex. This anisotropy is a key geometric property of high-dimensional information processing and underlies a variety of theoretical analyses. However, the mechanism by which it emerges from input structure and task-driven learning has remained unclear. Here, we characterize this formation process by exactly solving the learning dynamics of a wide two-layer linear neural network in a teacher--student setting with power-law input and teacher structures. We show that, in the feature-learning regime, the local power-law exponent of the internal-representation spectrum evolves nonmonotonically over the course of training and exhibits up to four distinct asymptotic regimes across modes and training times. By contrast, in the lazy regime, the exponent re
    
[^290]: 小规模实验：我们到了吗？

    Small-Scale Experiments: Are We There Yet?

    [https://arxiv.org/abs/2608.11859](https://arxiv.org/abs/2608.11859)

    本文证明小规模实验中缩放定律的可信度受超参数敏感性影响，且调优超参数比模型大小更关键，并解释了随规模增大超参数优化变易的原因。

    

    摘要：arXiv:2608.11859v1 公告类型：新 摘要：缩放定律曾承诺成本效益高的实验；六年后，它们尚未完全兑现。相反，研究人员发现它们在小型规模（从4M参数开始）下不可靠，并得出结论认为无法避免较大的模型。我们表明事实并非如此：混淆因素是超参数。小型模型高度敏感，但超参数敏感性随规模增大而减弱。这种小规模敏感性使得缩放定律容易被忽略，因为它们只在完全调优的前沿出现，而达到该前沿需要远超大多数人进行的广泛搜索。通过消融基本缩放定律配方，我们表明调优良好的超参数比任何其他成分都更重要。此外，我们揭示了为什么这些超参数变得更容易找到：随着规模增加，超参数损失曲面维度降低。然而，尽管缩放定律存在于小型模型中，外推会遇到统计问题。

    arXiv:2608.11859v1 Announce Type: new  Abstract: Scaling laws promised cost-effective experiments; six years later, they have yet to fully deliver. Instead, researchers have found them unreliable at small scales (starting at 4M parameters) and concluded that sizable models cannot be avoided. We show this is not the case: the confounding factor is hyperparameters. Small models are highly sensitive, but hyperparameter sensitivity fades with scale. This small-scale sensitivity makes scaling laws easy to miss because they only emerge on the fully tuned frontier, and reaching that frontier requires an extensive search far beyond what most ever run. By ablating the basic scaling law recipe, we show well-tuned hyperparameters matter more than any other ingredient. Further, we reveal why those hyperparameters become easier to find: as scale increases, the hyperparameter loss surface becomes lower dimensional. Nevertheless while scaling laws exist in small models, extrapolation hits statistical
    
[^291]: SeFoRA：面向异构客户端秩的草图聚合联邦低秩适配方法

    SeFoRA: Sketch-Aggregated Federated Low-Rank Adaptation with Heterogeneous Client Ranks

    [https://arxiv.org/abs/2608.10144](https://arxiv.org/abs/2608.10144)

    SeFoRA 通过让各客户端传输本地 ΔW 更新的线性草图实现联邦服务器端的直接聚合，解决了联邦 LoRA 中客户端秩异构导致的维度不兼容和双线性失配问题，并具有 $\mathcal{O}(T^{-1/2})$ 速率的一阶稳定点邻域收敛保证。

    

    我们研究基础模型（如大型语言模型 LLM 和大型视觉模型 LVM）的联邦参数高效微调问题，采用低秩适配方法。将 LoRA 与联邦学习相结合会引入两种设置各自单独都不存在的挑战：客户端可能使用不同的 LoRA 秩，导致其因子矩阵维度不兼容；同时，逐因子平均会受到双线性失配问题的困扰。我们提出了 SeFoRA，这是一种草图聚合的联邦 LoRA 算法，其中每个客户端传输其本地 ΔW 更新的线性草图，从而能够在联邦服务器端实现直接聚合。因此，SeFoRA 克服了秩异构性挑战，并缓解了双线性失配问题。我们建立了以 $\mathcal{O}(T^{-1/2})$ 的速率收敛至一阶稳定点邻域的收敛性保证。在语言模型和视觉模型上开展的大量数值实验表明，我们的算法相较于现有方法具有优越的性能。

    arXiv:2608.10144v2 Announce Type: replace  Abstract: We consider federated parameter-efficient fine-tuning of foundation models, e.g., LLMs and LVMs, using low-rank adaptation (LoRA; ~Hu et al.\ 2022). Combining LoRA with federated learning introduces challenges absent from either setting alone: clients may use different LoRA ranks, making their factor matrices dimension-incompatible, and factor-wise averaging suffers from a bilinear mismatch. We propose SeFoRA, a sketch-aggregated federated LoRA algorithm in which each client transmits a linear sketch of its local $\Delta \mathbf{W}$ updates, enabling direct aggregation at the federator. As a result, SeFoRA overcomes the rank heterogeneity challenge and alleviates the bilinear mismatch. We establish convergence to a neighborhood of a first-order stationary point at rate $\\mathcal{O}(T^{-1/2})$. Extensive numerical experiments on language and vision models showcase the performance of our algorithm compared to existing methods.
    
[^292]: 目标检索与动作价值排序：Best-of-K 选择下的对比式评论家

    Goal Retrieval versus Action-Value Ordering: Contrastive Critics under Best-of-$K$ Selection

    [https://arxiv.org/abs/2607.27422](https://arxiv.org/abs/2607.27422)

    本文揭示了对比式评论家“擅长目标检索却不擅长动作价值排序”的差距：未归一化对比得分在总体最优处对行为策略价值单调，但余弦归一化及有限训练的网络在 Best-of-K 选择中可能给出错误的动作排序。

    

    目标条件动作生成器能够产生许多看似合理的动作，Best-of-K 选择随后利用评论家针对当前状态和目标对这些动作进行排序。然而，标准的对比式评论家是在一种不同的排序任务上训练和评估的：针对固定的状态-动作对，将未来目标与无关负样本区分开来。本文证明，在总体最优处，未归一化的对比得分关于行为策略价值是单调的；而余弦归一化得分缺乏这一保证，且经有限训练的网络可能无法恢复正确的排序。作者使用共享的状态-目标查询和冻结的候选池来审计由此产生的差距，从而确保各评论家仅在动作排序上存在差异。在八个 OGBench 任务上，原始得分与余弦归一化评论家的检索 AUC 达到 0.96–1.00，但它们在导航任务上对动作进展的排序能力要弱得多，有时甚至出现颠倒。在精确价值受控的问题中，所有三个对比选择器都具有更高的平均……（原文摘要在此处截断）

    arXiv:2607.27422v2 Announce Type: replace  Abstract: Goal-conditioned action generators can produce many plausible actions. Best-of-\(K\) selection then uses a critic to rank those actions for the current state and goal. Standard contrastive critics are trained and evaluated on a different ranking: distinguishing future goals from unrelated negatives for a fixed state--action pair. The unnormalized contrastive score is monotone in behavior-policy value at the population optimum. Cosine-normalized scores lack this guarantee, and finite trained networks may fail to recover the correct ordering. We audit the resulting gap using shared state--goal queries and frozen candidate pools, so critics differ only in their action ordering. Raw and cosine critics reach retrieval AUCs of \(0.96\)--\(1.00\) across eight OGBench tasks. Their progress ordering on navigation tasks is much weaker and sometimes inverted. In the exact-value controlled problem, all three contrastive selectors have higher mea
    
[^293]: 逆强化学习通过模仿人类帮助对齐AI

    Inverse RL Helps Align AI by Imitating Humans

    [https://arxiv.org/abs/2607.24900](https://arxiv.org/abs/2607.24900)

    提出PARED方法，受逆强化学习启发，仅从专家演示中学习显式且可解释的奖励函数，从而在没有偏好判断或验证器的情况下实现对语言模型的有效对齐。

    

    语言模型对齐旨在使模型行为反映出理想特性，如有用性、安全性和指令遵循。现有方法从专家演示、偏好判断或可验证结果中获取监督信号。专家演示通常用于指导监督微调（SFT），而偏好判断和验证器则为强化学习（RL）提供奖励。这引出了一个问题：能否仅凭演示数据推导出显式、可解释且可用于后训练流程的奖励？受逆强化学习的启发，我们提出了从演示中估计的投影对齐奖励（PARED）。PARED在一小组响应级特征上从专家演示中学习显式奖励，使用一个轻量级判别器在该特征空间中将专家演示与策略自身生成的样本区分开来。通过推理时重排序和对抗性在策略（实验验证）……

    arXiv:2607.24900v2 Announce Type: replace  Abstract: Language model alignment aims to make model behavior reflect desirable properties such as helpfulness, safety, and instruction following. Existing methods draw supervision from expert demonstrations, preference judgments, or verifiable outcomes. Demonstrations typically guide supervised fine-tuning (SFT), while preference judgments and verifiers provide rewards for reinforcement learning (RL). This raises a question: can the demonstrations alone be used to derive an explicit, interpretable reward useful in a post-training pipeline? Motivated by inverse reinforcement learning, we introduce Projected Alignment Reward Estimated from Demonstrations (PARED). PARED learns an explicit reward from expert demonstrations over a small set of response-level features, using a lightweight discriminator that separates demonstrations from the policy's own samples in this feature space. Experiments with inference-time reranking and adversarial on-pol
    
[^294]: DFAH-Bench：金融决策中可观测智能体不稳定性的基准测试

    DFAH-Bench: Benchmarking Observable Agent Instability in Financial Decision-Making

    [https://arxiv.org/abs/2607.20491](https://arxiv.org/abs/2607.20491)

    该论文提出DFAH-Bench基准，通过同时衡量决策一致性与工具路径一致性，揭示金融智能体在决策结果高度稳定（约95%）的情况下，其背后的工具执行过程却存在显著不稳定性（一致性仅约50%）。

    

    金融智能体可以在重复做出相同决策的同时，改变其背后的工作过程。DFAH-Bench 将确定性-忠实性保障框架（DFAH）付诸实践，在相同的合格重放中同时配对决策一致性与工具路径一致性，然后将该资格认定原则扩展到证据、授权、执行和任务结果。回顾性与前瞻性重放分析揭示了稳定决策背后的过程变化。在570个符合条件的前瞻性情节中，决策一致性为94.2-95.1%，而有序工具、参数和结果的一致性仅为45.0-51.5%；其中一个分层比预先规定的覆盖最低值低了一组。一项独立的捕获诊断显示，系统性的遗漏仍可保持完美的重放一致性。使用 τ-Knowledge 银行环境，我们在采用开源权重与前沿生成器的不同队列中保留了1,080个计划情节和1,033个已知原生结果。

    arXiv:2607.20491v3 Announce Type: replace-cross  Abstract: A financial agent can repeat a decision while changing the work behind it. DFAH-Bench operationalizes the Determinism--Faithfulness Assurance Harness (DFAH), pairing decision agreement with tool-path agreement on the same qualified replays, then extends that qualification principle to evidence, authorization, execution and task outcomes. Retrospective and prospective replay analyses expose process variation behind stable decisions. Across 570 eligible prospective episodes, decision agreement is 94.2-95.1%, while agreement on ordered tools, arguments and results is 45.0-51.5%; one stratum falls one group below its prespecified coverage minimum. A separate capture diagnostic shows that systematic omissions can preserve perfect replay agreement. Using the $\tau$-Knowledge banking environment, we retain 1,080 scheduled episodes and 1,033 known native outcomes across separate cohorts with open-weight and frontier generators. Missing
    
[^295]: 面向黎曼激活引导的条件最优桥

    Conditional Optimal Bridge for Riemannian Activation Steering

    [https://arxiv.org/abs/2607.10517](https://arxiv.org/abs/2607.10517)

    该论文提出COBRAS方法，将激活引导建模为残差流超球面上的薛定谔桥问题，从适定的优化问题出发有原则地推导出对数密度比引导目标，并通过查询自适应的引导场和支撑分数控制分布外干预，在多个模型上优于现有方法。

    

    激活引导为在推理阶段控制大型语言模型提供了一种轻量级的微调替代方案。许多现有的引导方法隐式地优化期望与不期望激活分布之间的对数密度比目标，但缺乏有原则的推导。而采用固定引导方向的方法在分布外（OOD）场景下性能也可能下降。我们提出了用于黎曼激活引导的条件最优桥（COBRAS），该方法将引导建模为残差流超球面上的薛定谔桥问题。这一表述使得对数密度比引导目标能够从一个适定的优化问题中得到有原则的推导。我们构建了一个查询自适应的引导场，将对数密度比引导作为其极限情形加以恢复。拟合的传输还为控制OOD干预提供了支撑分数。在四个模型上，COBRAS的性能优于先前的引导方法。

    arXiv:2607.10517v2 Announce Type: replace  Abstract: Activation steering offers a lightweight alternative to fine-tuning for controlling large language models at inference time. Many existing steering methods implicitly optimize a log-density-ratio objective between desired and undesired activation distributions without a principled derivation. Methods with fixed steering directions can also degrade out-of-distribution (OOD) performance. We introduce Conditional Optimal Bridge for Riemannian Activation Steering (\textsc{Cobras}), which casts steering as a Schr\"{o}dinger Bridge on the residual-stream hypersphere. This formulation yields a principled derivation of the log-density-ratio steering objective from a well-posed optimization problem. We construct a query-adaptive steering field that recovers log-density-ratio steering as a limiting case. The fitted transport also provides a support score for controlling OOD interventions. Across four models, \textsc{Cobras} outperforms prior s
    
[^296]: 从边缘分布构建树：基于因子化先验的自回归草稿生成

    Trees from Marginals: Autoregressive drafting with factorized priors

    [https://arxiv.org/abs/2607.06763](https://arxiv.org/abs/2607.06763)

    本文提出Weaver，一种从因子化草稿模型的top-K边缘分布构建提议树的轻量级自回归适配器，恢复了token间的条件依赖，并结合无需回滚的树验证算法与优化的CUDA内核，相比自回归解码实现了4.08倍加速。

    

    投机解码通过以计算换取单次前向传播中额外生成的token，极大地提升了自回归语言模型的交互性。因子化草稿模型尤其高效，因为它们可以并行预测未来token的边缘分布，但其独立性假设导致随着投机预算的增长，接受率急剧下降。我们分析了这一局限性，并提出了Weaver——一个轻量级的自回归适配器，它从因子化草稿模型的top-K边缘分布中构建提议树。Weaver恢复了提议token之间的条件依赖关系，同时避免了全词表投影计算。为了支持对具有门控Delta Net（Gated Delta Net）层的模型进行快速验证，我们推导了一种无需回滚的树验证算法，并在SGLang中实现了优化的CUDA内核。通过结合这些模型与系统层面的贡献，我们相比自回归解码实现了4.08倍的加速。

    arXiv:2607.06763v3 Announce Type: replace-cross  Abstract: Speculative decoding greatly increases the interactivity of autoregressive language models by trading off computation for extra tokens generated in a single forward pass. Factorized draft models are especially efficient because they predict future-token marginals in parallel, but their independence assumption causes acceptance rates to degrade sharply as the speculative budget grows. We analyze this limitation and introduce Weaver, a lightweight autoregressive adapter that constructs proposal trees from the top-K marginals of a factorized drafter. Weaver restores conditional dependencies between proposed tokens while avoiding a full-vocabulary projection. To support fast verification for models with Gated Delta Net layers, we derive a rollback-free tree-verification algorithm and implement optimized CUDA kernels in SGLang. By combining these model and systems contributions we achieve a 4.08-fold speedup over autoregressive deco
    
[^297]: 学习何时自动化：人机协作服务系统中的排队控制

    Learning When to Automate: Queue Control in Human-AI Service Systems

    [https://arxiv.org/abs/2607.06017](https://arxiv.org/abs/2607.06017)

    该论文针对人机协作服务系统，提出了结合上置信界学习与漂移加惩罚控制的UCB-DPP策略，能够在聊天机器人成功概率和人工客服服务速率未知的情况下在线学习系统参数，从而在减少人工拥塞与控制自动化成本之间实现最优的排队与资源分配决策。

    

    我们研究一个人机协作服务系统，其中任务按顺序到达，并通过两阶段架构进行处理：首先由自动化聊天机器人处理，必要时再转交人工客服。我们考虑 $T$ 个顺序到达的任务，每个任务属于 $K$ 种异构类型之一。对于每个任务，决策者需要选择为聊天机器人分配多少资源，而聊天机器人针对各任务类型的成功概率最初是未知的。未被聊天机器人解决的任务将进入依赖任务类型的人工服务队列，由服务速率未知的人工客服进行处理。该模型刻画了混合服务系统中的一个核心权衡：更多地依赖自动化可以减少人工服务的拥塞，但会增加聊天机器人的成本，而自动化不足则可能导致人工客服过载。我们提出了 UCB-DPP 策略，该策略将上置信界算法与漂移加惩罚控制相结合，在做出决策的同时学习系统的未知参数……（原文摘要在此处截断）

    arXiv:2607.06017v2 Announce Type: replace  Abstract: We study a human-AI service system in which tasks arrive sequentially and are processed through a two-stage architecture: an automated chatbot followed, when necessary, by a human agent. We consider $T$ sequentially arriving tasks, each belonging to one of $K$ heterogeneous types. For each task the decision maker chooses how many resources to allocate to the chatbot, whose type-dependent success probabilities are initially unknown. Tasks not resolved by the chatbot enter type-dependent human-service queues, where they are processed by a human agent with unknown service rates. This model captures a central tradeoff in hybrid service systems: relying more on automation reduces human congestion but increases chatbot costs, while insufficient automation may overload the human agent. We propose the UCB-DPP policy, which combines Upper Confidence Bounds with Drift-Plus-Penalty control to learn the unknown parameters of the system while mak
    
[^298]: 保留还是适应？持续学习的泛化

    To Retain or to Adapt? Generalizing Continual Learning

    [https://arxiv.org/abs/2607.05609](https://arxiv.org/abs/2607.05609)

    该论文挑战了持续学习以“保留知识”为中心的传统假设，将持续学习重新形式化为以最小化平均终身误差为目标的在线优化问题，并提出“迁移效率”作为衡量不稳定性与瞬态误差之间张力的定量指标。

    

    持续学习领域的文献长期以来一直以缓解灾难性遗忘为目标。这一目标建立在一个通常未被言明的假设之上：即终身学习者应逼近联合任务学习（JTL）的解，并保留所有先前获得的知识。我们挑战这一以保留为中心的前提，认为在非平稳环境中，优先考虑保留可能会阻碍实时适应。我们将焦点转向平均终身误差（ALE），将持续学习形式化为一个由环境动态与学习动态之间相互作用所支配的在线优化问题。我们引入迁移效率作为定量度量，用以刻画不稳定性（继承自相互冲突的过去经验的偏差）与瞬态误差（从头学习新任务所产生的优化成本）之间的张力。在温和的收敛条件下，该分解在线性模型和神经网络模型中均成立。

    arXiv:2607.05609v2 Announce Type: replace-cross  Abstract: The Continual Learning (CL) literature has long been driven by the goal of mitigating catastrophic forgetting. This objective rests on an often unstated assumption: that a lifelong learner should approximate the Joint-Task Learning (JTL) solution and retain all previously acquired knowledge. We challenge this retention-centered premise, arguing that in non-stationary environments prioritizing retention can impede real-time adaptation. Shifting the focus to the Average Lifelong Error (ALE), we formalize CL as an online optimization problem governed by the interaction between environmental and learning dynamics. We introduce Transfer Efficiency as a quantitative measure of the tension between Instability, the bias inherited from conflicting past experience, and Transient Error, the optimization cost of learning new tasks from scratch. Under mild convergence conditions, holding across linear and neural network models, this decompo
    
[^299]: 二叉树机制在差分隐私持续计数中是最优的

    The Binary Tree Mechanism is Optimal for Differentially Private Continual Counting

    [https://arxiv.org/abs/2607.00876](https://arxiv.org/abs/2607.00876)

    本文证明了在固定隐私参数下，差分隐私持续计数问题中误差对数据流长度的依赖是不可回避的下界（近似DP下 $\Omega(\log^{3/2} n)$，纯DP下 $\Omega(\log^2 n)$），从而确立了二叉树机制的最优性。

    

    私有持续计数是差分隐私中的一个基本问题：给定一条长度为 $n$ 的二进制数据流，其中每个 $1$ 对应一个个体的贡献，目标是在保护每个个体隐私的同时发布所有运行中的累计计数。在隐私参数固定的情况下，标准的二叉树机制在近似差分隐私下可实现期望 $\ell_\infty$ 误差 $O(\log^{3/2} n)$，在纯差分隐私下可实现 $O(\log^2 n)$。这些误差对数据流长度的依赖是否必要，一直是一个核心的开放性问题。对于固定的 $\varepsilon\in(0,1)$，我们证明了：在 $\delta>0$ 为足够小的固定值时，近似差分隐私下的下界为 $\Omega(\log^{3/2} n)$；在纯差分隐私下的下界为 $\Omega(\log^2 n)$。这些下界确立了二叉树机制在两种设置下的最优性。这些界对任意机制均成立，即使整条数据流是……（原文此处截断）

    arXiv:2607.00876v3 Announce Type: replace-cross  Abstract: Private continual counting is a fundamental problem in differential privacy: given a binary stream of length $n$, where each $1$ corresponds to the contribution of one individual, the goal is to release all running counts while protecting the privacy of each individual. For fixed privacy parameters, the standard binary tree mechanism achieves expected $\ell_\infty$ error $O(\log^{3/2} n)$ under approximate differential privacy and $O(\log^2 n)$ under pure differential privacy. Whether these dependences on the stream length are necessary has remained a central open problem.   For fixed $\varepsilon\in(0,1)$, we prove a lower bound of $\Omega(\log^{3/2} n)$ under approximate DP with sufficiently small fixed $\delta>0$, and a lower bound of $\Omega(\log^2 n)$ under pure DP. These bounds establish the optimality of the binary tree mechanism in both settings. The bounds hold for arbitrary mechanisms, even when the entire stream is a
    
[^300]: 控制混合：面向基于Transformer模型的状态感知微调

    Mixture-of-Control: State-Aware Fine-Tuning for Transformer-based Models

    [https://arxiv.org/abs/2606.31397](https://arxiv.org/abs/2606.31397)

    提出了控制混合方法，将逐块控制状态作为稀疏混合专家中的专家，实现高效的跨块通信，自适应整合局部与全局控制信号，从而在状态感知微调中增强表示学习。

    

    基于状态的微调已成为Transformer基于权重自适应的一种有吸引力的替代方案，它将轻量级控制更新到状态中而非模型权重中，在保持参数效率的同时提供了显著的内存节省。然而，大多数现有的基于状态的方法通常仅应用逐块的控制更新，这限制了块间的信息交换并制约了表示自适应能力。同时，先前支持跨块通信的机制往往会引入相当大的计算开销，降低了其在高效微调中的实用性。我们提出了控制混合，这是一种高效的跨层通信机制，能够自适应地整合局部和全局控制信号，以增强自适应过程中的表示学习。MoC将逐块控制状态视为稀疏混合专家过程中的专家，从而实现跨Transformer块的高效通信。

    arXiv:2606.31397v2 Announce Type: replace  Abstract: State-based fine-tuning has emerged as a compelling alternative to weight-based adaptation for transformers, updating lightweight controls into states rather than model weights, offering substantial memory savings while retaining parameter efficiency. However, most existing state-based methods typically apply only per-block control updates, which limits inter-block information exchange and restricts representational adaptation. Meanwhile, prior mechanisms that enable cross-block communication often introduce considerable computational overhead, reducing their practicality for efficient fine-tuning. We introduce Mixture-of-Control (MoC), an efficient cross-layer communication that adaptively integrates local and global control signals to enhance representation learning during adaptation. MoC treats block-wise control states as experts in a sparse mixture-of-experts process, enabling efficient communication across transformer blocks. E
    
[^301]: 序贯稀疏高斯过程分位数回归

    Sequential sparse Gaussian process quantile regression

    [https://arxiv.org/abs/2606.31284](https://arxiv.org/abs/2606.31284)

    本文提出一种序贯稀疏高斯过程分位数回归框架，通过诱导变量和拉普拉斯近似实现高效后验推断，并将预测不确定性分解为条件先验与后验诱发两个分量，以此驱动诱导输入填充和数据采集两种自适应机制，实现计算资源的动态分配。

    

    分位数回归旨在从观测数据中估计响应变量的条件分位数。在贝叶斯框架下，高斯过程分位数回归能够提供不确定性量化，但由于非对称拉普拉斯似然函数的非共轭性以及后验推断的计算代价，该方法面临显著的计算挑战。我们开发了一个稀疏高斯过程框架，其中分位数函数通过缩减的诱导变量集合来表示，并使用拉普拉斯近似进行后验推断。随后，通过将预测不确定性分解为条件先验方差分量和后验诱发方差分量，驱动了两种互补的自适应机制：诱导输入填充和数据采集。这些机制被结合到一个序贯算法中，该算法将计算资源分配给预测不确定性的主要来源，并自适应地控制

    arXiv:2606.31284v2 Announce Type: replace  Abstract: Quantile regression aims to estimate the conditional quantiles of a response variable from observed data. In a Bayesian setting, Gaussian process quantile regression provides uncertainty quantification but faces significant computational challenges due to the nonconjugacy of the asymmetric Laplace likelihood and the cost of posterior inference. We develop a sparse Gaussian process framework in which the quantile function is represented through a reduced set of inducing variables and posterior inference is performed using a Laplace approximation. A decomposition of the predictive uncertainty into conditional-prior and posterior-induced variance components is then exploited to drive two complementary adaptive mechanisms: inducing-input infilling and data acquisition. These mechanisms are combined within a sequential algorithm that allocates computational effort toward the dominant source of predictive uncertainty and adaptively control
    
[^302]: 分式松弛像素动力学下的事件相机仿真

    FracEvent: Event-Camera Simulation via Fractional-Relaxation Pixel Dynamics

    [https://arxiv.org/abs/2606.26636](https://arxiv.org/abs/2606.26636)

    提出FracEvent事件仿真器，通过分式松弛电压动力学精确建模像素生命周期，以改善事件时序保真度和下游任务迁移性能。

    

    事件相机能以微秒级时间分辨率异步报告亮度变化，但由于需要专用传感器、精确同步和任务特定标注，真实事件数据难以大规模采集。因此，事件相机仿真对于基于事件的视觉任务至关重要。大多数实用仿真器基于对比度阈值事件生成，部分还加入额外滤波、随机噪声或手动调参的传感器参数。尽管有效，这类公式常简化每个像素生命周期产生的时间结构，从而扭曲事件时序并削弱下游任务迁移效果。我们提出FracEvent，一种采用分式松弛电压动力学建模像素级生命周期的事件仿真器。给定对数强度轨迹，FracEvent驱动一个紧凑的松弛模式栈，将其响应合并为电压状态，并通过比较电压与阈值来触发ON/OFF事件。

    arXiv:2606.26636v1 Announce Type: cross  Abstract: Event cameras asynchronously report brightness changes with microsecond-level temporal resolution, but real event data remain difficult to collect at scale because specialized sensors, careful synchronization, and task-specific annotations are required. Event-camera simulation is therefore important to event-based vision tasks. Most practical simulators build on contrast-threshold event generation, some with additional filtering, stochastic noise, or hand-tuned sensor parameters. While effective, such formulations often simplify the temporal structure produced by the lifecycle of each pixel, which can distort event timing and weaken downstream transfer. We introduce FracEvent, an event simulator that models this pixel-level lifecycle with fractional-relaxation voltage dynamics. Given a log-intensity trajectory, FracEvent drives a compact stack of relaxation modes, combines their responses into a voltage state, emits ON/OFF events by lo
    
[^303]: TNODEV：神经ODE验证工具箱

    TNODEV: Toolbox for Neural ODE Verification

    [https://arxiv.org/abs/2606.16567](https://arxiv.org/abs/2606.16567)

    TNODEV是首个针对神经ODE的端到端形式化验证工具箱，通过集成证伪检查、基于混合单调性的快速可达性分析、输入集精化循环与并行调度，显著提升了验证精度。

    

    神经常微分方程在安全关键场景中备受关注，例如作为信息物理系统的连续时间控制器以及集成到自动化决策流水线中的分类器，由此引出了其行为能否被形式化验证的问题。现有专门针对神经ODE的工具仅提供单次可达性调用，缺乏迭代式输入集精化，这使其判定精度受限于单次可达性调用所能达到的水平。我们提出TNODEV，这是首个针对神经ODE的形式化验证器，它在单一的端到端流水线中集成了证伪检查器、基于连续时间混合单调性的快速区间可达性后端、带有三种输入集分割启发式策略的验证与精化循环，以及并行调度器。TNODEV支持对纯神经ODE以及与神经网络控制器构成闭环的神经ODE进行安全集包含验证。

    arXiv:2606.16567v3 Announce Type: replace-cross  Abstract: Neural ordinary differential equations (neural ODE) gained attention in safety critical settings such as continuous-time controllers for cyber-physical systems and classifiers integrated into automated decision pipelines, raising the question whether their behavior can be formally verified. Existing tools dedicated to neural ODE provide only a single reachability call without iterative input-set refinement, limiting the precision of their verdicts to whatever one reachability call can deliver. We present TNODEV, the first formal verifier for neural ODE that integrates a falsification checker, a fast interval-based reachability backend based on continuous-time mixed monotonicity, a verification and refinement loop with three input-set splitting heuristics, and a parallel scheduler in a single end-to-end pipeline. TNODEV supports safe-set inclusion verification on pure neural ODE, neural ODE in closed loop with a neural network c
    
[^304]: μ₀：可扩展的3D交互轨迹世界模型

    $\mu_0$: A Scalable 3D Interaction-Trace World Model

    [https://arxiv.org/abs/2606.13769](https://arxiv.org/abs/2606.13769)

    μ₀通过预测物体、工具、手等显著交互点的平滑3D轨迹（而非密集像素或具身特定动作标签）构建可扩展的世界模型，并借助TraceExtract系统从多样化视频源自动提取3D监督，实现与具身形态无关的机器人学习预训练。

    

    能够捕捉动作如何引起物理变化的世界模型，使机器人学习无需依赖特定具身形态的动作标签即可实现规模化。像素空间视频模型提供了广泛的视觉先验，但会将模型容量耗费在密集的外观重建上；而直接动作模型则需要阻碍可扩展性的具身形态特定标签。我们提出了 μ₀，一个基于3D轨迹的可扩展世界模型。μ₀ 不预测密集像素或直接建模动作，而是为物体、工具、手和接触区域等显著交互点预测平滑的3D轨迹，从而形成一个紧凑的、与具身形态无关的运动接口。为了实现从多样化视频源进行训练，我们的 TraceExtract 系统通过选择关键点、构建全局对齐的轨迹，并将运动片段与分层语言描述相关联，自动提取3D监督。这种 TraceExtract 监督用于预训练……（原文摘要在此处截断）

    arXiv:2606.13769v3 Announce Type: replace-cross  Abstract: World models that capture how actions induce physical change enable scalable robot learning without reliance on embodiment-specific action labels. Pixel-space video models provide broad visual priors but expend model capacity on dense appearance reconstruction, while direct action models require embodiment-specific labels that hinder scalability. We present $\mu_0$, a scalable world model based on 3D traces. Rather than predicting dense pixels or directly modeling actions, $\mu_0$ forecasts smooth 3D trajectories for salient interaction points such as objects, tools, hands, and contact regions, yielding a compact, embodiment-agnostic motion interface. To enable training from diverse video sources, our TraceExtract system automatically extracts 3D supervision by selecting keypoints, constructing globally aligned traces, and associating motion segments with hierarchical language captions. This TraceExtract supervision pretrains $
    
[^305]: 重新思考疾病预测中的合成推理监督

    Rethinking Synthetic Rationale Supervision for Disease Prediction

    [https://arxiv.org/abs/2606.10279](https://arxiv.org/abs/2606.10279)

    该研究发现，在疾病预测中使用合成推理理由进行“推理优先”监督微调反而会持续且显著地损害预测性能，且这种退化跨模型系列、数据规模和任务普遍存在，并非由推理理由质量不佳所致。

    

    在监督微调中使用合成推理理由被广泛认为能够提升临床预测效果，因为它不仅教会模型“预测什么”，还教会模型“为什么这样预测”。我们在基于纵向健康病史的五年阿尔茨海默病及相关痴呆预测任务中检验了这一假设。在该设置中，我们使用正确标签生成训练推理理由，然后微调模型使其先输出推理理由、再输出标签。在一项涵盖504种配置的大规模对照实验中，我们发现“推理优先”的监督微调相比“仅标签”微调，始终且显著地损害了预测性能。这种性能退化在不同模型系列和数据规模上均持续存在，且使用面向推理的基础模型也无法解决。在一项采用固定协议的心血管疾病验证实验中重现了类似的发现。至关重要的是，这一失败并非由推理理由质量不佳所解释：人类专家标注证实了生成……（摘要原文在此处截断）

    arXiv:2606.10279v2 Announce Type: replace-cross  Abstract: Supervised fine-tuning with synthetic rationales is widely assumed to improve clinical prediction by teaching models not only what to predict but why. We test this assumption on five-year Alzheimer's disease and related dementias (ADRD) prediction from longitudinal health histories. In this setting, we generate training rationales using the correct labels, then fine-tune models to output a rationale followed by a label. Across a large-scale controlled experiment of 504 configurations, we find that Rationale First SFT consistently and substantially hurts predictive performance relative to Label Only fine-tuning. The degradation persists across model families and data scales, and is not resolved by using a reasoning-oriented base model. A fixed-protocol cardiovascular disease (CVD) validation reproduces similar findings. Crucially, the failure is not explained by poor rationale quality: human expert annotation confirms that the g
    
[^306]: Lean4Agent：面向智能体工作流与轨迹的形式化建模与验证

    Lean4Agent: Formal Modeling and Verification for Agent Workflow and Trajectory

    [https://arxiv.org/abs/2606.06523](https://arxiv.org/abs/2606.06523)

    本文提出Lean4Agent，这是首个利用Lean4依赖类型形式化语言对智能体工作流的语义一致性进行形式化建模与验证、并能对执行轨迹揭示的运行时故障进行定位的框架。

    

    让大语言模型（LLM）能够执行可靠的多步骤工作流已成为人工智能领域的核心挑战。尽管LLM的智能体能力近来取得了进展，但大多数智能体系统仍然缺乏用于规范、验证和调试其工作流与执行轨迹的形式化方法。这一挑战与数学领域的一个长期问题相呼应：自然语言（NL）的歧义性推动了形式化语言（FL）的发展。受这一范式的启发，我们提出了Lean4Agent——据我们所知，这是首个使用Lean4（一种依赖类型形式化语言）来建模和验证智能体行为的框架。Lean4Agent推出了FormalAgentLib，这是一个可扩展的Lean4库，用于在显式假设下对智能体工作流的语义一致性进行形式化建模与验证，并能够对轨迹所揭示的运行时故障进行定位。基于FormalAgent

    arXiv:2606.06523v3 Announce Type: replace-cross  Abstract: Equipping Large Language Models (LLMs) to execute reliable multi-step workflows has become a central challenge in artificial intelligence. Despite recent advances in LLMs' agentic capabilities, most agent systems still lack formal methods for specifying, verifying, and debugging their workflow and execution trajectories. This challenge mirrors a long-standing problem in mathematics, where the ambiguity of natural languages (NLs) motivates the development of formal languages (FLs). Inspired by this paradigm, we propose **Lean4Agent**, to the best of our knowledge, the first framework that uses Lean4, a dependent-type FL to model and verify agent behavior. **Lean4Agent** launches **FormalAgentLib**, an extensible Lean4 library for formally modeling and verifying agent workflows' semantic consistency under explicit assumptions, and enabling localization of execution-time failures revealed by trajectories. Building on **FormalAgent
    
[^307]: 让它保持简单：面向视觉-语言-动作模型的单步动作生成

    Let It Be Simple: One-Step Action Generation for Vision-Language-Action Models

    [https://arxiv.org/abs/2606.05737](https://arxiv.org/abs/2606.05737)

    该论文发现VLA动作生成与文生图截然不同——只需将训练时间步分布偏向噪声端点，或直接将多步迭代解码替换为单次前向传播，即可获得性能相当的单步策略，其根本原因在于VLA生成的条件-目标结构更接近图生文任务。

    

    对于文生图或视频生成任务，从标准的扩散或流匹配训练直接实现单步生成，在不借助蒸馏或专门单步目标的情况下通常难以奏效。我们发现VLA动作生成却表现出截然不同的特性：仅将训练时间步分布向噪声端点偏置，就能在LIBERO、LIBERO-Plus、LIBERO-Pro以及真实世界机器人任务上获得强大的单步策略。此外，对于不同架构的开源VLA模型，仅仅将迭代解码替换为单次前向传播，在LIBERO、CALVIN和SimplerEnv等基准上往往就能达到与原始多步解码相当的性能。我们将这一差异归因于VLA生成的条件-目标结构：它更接近于图生文而非文生图——丰富的观测对相对紧凑的目标形成了强约束。我们通过不可约简的速度损失将这一直觉形式化

    arXiv:2606.05737v3 Announce Type: replace-cross  Abstract: One-step generation from standard diffusion or Flow-Matching training is rarely effective for text-to-image or video generation without distillation or specialized one-step objectives. We find that VLA action generation behaves strikingly differently. Biasing only the training timestep distribution toward the noise endpoint yields strong one-step policies across LIBERO, LIBERO-Plus, LIBERO-Pro, and real-world robot tasks. Moreover, for open-source VLA models with different architectures, simply replacing iterative decoding with a single forward pass often performs comparably to the original multi-step decoding on benchmarks such as LIBERO, CALVIN, and SimplerEnv. We attribute this difference to the condition-target structure of VLA generation, which is closer to image-to-text than text-to-image: rich observations strongly constrain a comparatively compact target. We formalize this intuition through the irreducible velocity loss
    
[^308]: InfoQuant：面向低比特大语言模型量化的激活分布整形

    InfoQuant: Shaping Activation Distributions for Low-Bit LLM Quantization

    [https://arxiv.org/abs/2605.26175](https://arxiv.org/abs/2605.26175)

    该论文提出InfoQuant，将激活变换重新定义为面向量化器的分布设计问题，并从信息论视角证明量化友好的激活分布应同时具备更小的数值范围和足够的离散程度，从而提升大语言模型低比特量化的效果。

    

    低比特激活量化仍然是大语言模型（LLM）高效部署中的一个主要瓶颈。其难点不仅在于激活值中存在离群值，还在于其分布往往与低比特均匀量化器匹配不佳。现有的训练后量化（PTQ）方法通过抑制峰值、平衡通道或最小化重建误差来处理该问题，但它们很少明确说明什么样的激活分布才真正易于离散化。因此，激活值在数值上可能看起来更平滑，却仍然会产生较大的量化误差，因为量化范围依然很宽，或者大多数值坍缩到均值附近的少数几个量化级别上。我们将激活变换重新表述为面向量化器的分布设计问题，并从信息论的角度分析量化误差。我们的分析表明，对量化友好的激活应当同时具备更小的数值范围和足够的离散程度。

    arXiv:2605.26175v2 Announce Type: replace  Abstract: Low-bit activation quantization remains a major bottleneck in efficient large language model (LLM) deployment. The difficulty is not only that activations contain outliers, but that their distributions are often poorly matched to a low-bit uniform quantizer. Existing post-training quantization (PTQ) methods suppress peaks, balance channels, or minimize reconstruction error, yet they rarely specify what activation distribution is actually easy to discretize. As a result, activations may appear numerically smoother while still incurring large quantization error because the quantization range remains wide or most values collapse into a few levels near the mean. We recast activation transformation as quantizer-facing distribution design and analyze quantization error from an information-theoretic perspective. Our analysis shows that quantization-friendly activations should jointly have a smaller numerical range and sufficient dispersion 
    
[^309]: 基于多源数据的大都市尺度弹性且可信的交通流推断

    Metropolis-Scale Resilient and Trustworthy Traffic Flow Inference Using Multi-Source Data

    [https://arxiv.org/abs/2605.25004](https://arxiv.org/abs/2605.25004)

    提出任务感知注意力神经过程（TA-ANP）统一概率框架，融合浮动车数据与稀疏固定检测器测量，实现无需重训练即可适应传感配置变化的大都市尺度弹性可信全网交通状态推断。

    

    从稀疏观测中以高精度和可信的不确定性量化来推断全网交通状态，对智能交通系统至关重要，然而由于问题的欠定本质、传感网络中的多方面干扰，以及多个推断子任务在联合建模时存在的内在冲突，这一任务仍然充满挑战。我们提出了任务感知注意力神经过程（TA-ANP），这是一个统一的概率框架，通过融合浮动车数据（FCD）与稀疏固定检测器测量，实现弹性且可信的全局交通状态推断（GTSI）。通过将GTSI建模为随机过程，TA-ANP利用神经过程的元学习特性，无需重新训练即可快速适应传感配置的变化。此外，还引入了具有独特时空归纳偏置的任务感知多查询注意力模块，以联合处理三个GTSI子任务——

    arXiv:2605.25004v2 Announce Type: replace  Abstract: Inferring network-wide traffic states from sparse observations with high accuracy and trustworthy uncertainty quantification is essential for intelligent transportation systems, yet it remains challenging due to the underdetermined nature of the problem, multifaceted disturbances in sensing networks, and the inherent conflicts among multiple inference subtasks when modeled jointly. We propose the Task-Aware Attentive Neural Process (TA-ANP), a unified probabilistic framework for resilient and trustworthy global traffic state inference (GTSI) by fusing floating car data (FCD) with sparse fixed-detector measurements. By casting GTSI as a stochastic process, TA-ANP leverages the meta-learning properties of neural processes to adapt rapidly to changes in sensing configurations without retraining. A task-aware multi-query attention module with distinct spatiotemporal inductive biases is introduced to handle three GTSI subtasks jointly ---
    
[^310]: 从理论到决策规则：在三个医学影像基准上校准视觉-语言模型弱监督的噪声标签交叉点

    From Theory to Decision Rule: Calibrating the Noisy-Label Crossover for Vision-Language Model Weak Supervision Across Three Medical-Imaging Benchmarks

    [https://arxiv.org/abs/2605.24771](https://arxiv.org/abs/2605.24771)

    本文在三个医学影像基准上实证测定了视觉-语言模型伪标签从有益转为有害的交叉点，并将其转化为可操作的专家标注数量预算决策规则。

    

    在医学影像领域，少量专家标注通常与来自视觉-语言模型（VLM）的零样本伪标签相结合。当专家标注集非常小时，额外的伪标签能够降低方差并提升准确率；然而，一旦仅用专家标注训练的分类器达到弱标注器的准确率水平，加入伪标签反而会降低测试性能。我们在PatchCamelyon组织病理学、ISIC 2019皮肤镜和NIH ChestX-ray14放射影像三个数据集上，使用BiomedCLIP和PubMedCLIP生成的伪标签，并在六种学生架构（800万至8900万参数）上测量了该交叉点出现的位置。结果显示：在ISIC 2019上，弱标签在专家标注达到20–50个后不再有帮助；在PatchCamelyon上约为100个；在NIH ChestX-ray14上为250–500个。超过这些标注预算后，加入10000个伪标签会使测试宏平均AUC最多下降0.14个百分点（ISIC 2019上BiomedCLIP，PubMedCLIP为0.23），（摘要原文在此处被截断）

    arXiv:2605.24771v2 Announce Type: replace-cross  Abstract: In medical imaging, small sets of expert labels are often combined with zero-shot pseudo-labels from a vision-language model (VLM). When the expert-labeled set is very small, the extra pseudo-labels reduce variance and improve accuracy. Once a classifier trained on expert labels alone reaches the accuracy of the weak labeler, however, adding pseudo-labels lowers test performance. We measure where this crossover occurs on PatchCamelyon histopathology, ISIC 2019 dermoscopy, and NIH ChestX-ray14 radiography using pseudo-labels from BiomedCLIP and PubMedCLIP across six student architectures ($8\mathrm{M}$ to $89\mathrm{M}$ parameters). Weak labels stop helping after $20$--$50$ expert labels on ISIC 2019, about $100$ on PatchCamelyon, and $250$--$500$ on NIH ChestX-ray14. Above those budgets, adding $10{,}000$ pseudo-labels reduces test macro-AUC by up to $0.14$ points with BiomedCLIP ($0.23$ with PubMedCLIP) on ISIC 2019, $0.10$ on
    
[^311]: 生物学背景信息并不能可靠地提升LLM引导的实验优化

    Biological Context Does Not Reliably Improve LLM-Guided Experimental Optimization

    [https://arxiv.org/abs/2605.15341](https://arxiv.org/abs/2605.15341)

    通过对45个重建自341篇论文的生物学任务进行大规模测试，研究发现为LLM提供真实的生物学背景信息（参数名称、单位和标签）反而会降低实验优化性能，在超过一半的任务上不如使用无生物学语义的匿名化名称。

    

    自主实验室越来越多地将大语言模型（LLM）部署为闭环生物学优化器，即提出一个实验设计、读取结果、然后再提出新设计。这在一定程度上源于一个假设：生物学知识能帮助LLM比无法获取生物学背景的贝叶斯优化选择更好的设计方案。我们在45个生物学任务上检验了这一假设，这些任务从341篇已发表论文的2,485条实验数据中重建，由拟合这些数据的“神谕”模型进行评分，每次运行在两种提示条件下进行30轮迭代。领域感知条件提供已发表的参数名称、单位和标签，而领域无关条件则将这些名称重命名使其不具生物学意义，同时保持搜索空间和反馈完全相同。在19个LLM上，领域无关条件在按轨迹衡量时于58.7%的任务上、按终点衡量时于61.9%的任务上优于领域感知条件。这种失败是生物学特有的，在以同样方式构建的教育任务上结果会发生反转。

    arXiv:2605.15341v2 Announce Type: replace  Abstract: Autonomous labs increasingly deploy large language models (LLMs) as closedloop biology optimizers that propose a design, read the outcome, then propose again. This is partly because of an assumption that biological knowledge helps an LLM choose better designs than Bayesian optimization, which has no access to biological context. We test that assumption on 45 biology tasks reconstructed from 2,485 experimental rows across 341 published papers, scored by oracles fitted to those rows, each run over 30 iterations under two prompt conditions. The domain-aware condition provides the published parameter names, units, and labels, and the domain-agnostic renames them so they are not biologically meaningful, with search space and feedback identical. Across 19 LLMs domain-agnostic outperforms domain-aware on 58.7% of tasks by trajectory and 61.9% by endpoint. The failure is specific to biology, reversing on education tasks built the same way. D
    
[^312]: 一种基于Schur分解的新型权重投影方法，用于稳定的状态空间神经网络架构

    A Novel Schur Decomposition-Based Weight Projection Method for Stable State-Space Neural-Network Architectures

    [https://arxiv.org/abs/2605.14489](https://arxiv.org/abs/2605.14489)

    本文提出一种基于Schur分解的权重投影方法，通过将状态空间神经网络中状态矩阵的准三角因子动态投影到最近的稳定矩阵上，以最小过参数化代价保证模型动态的渐近稳定性，且完全兼容反向传播训练。

    

    从数据中构建动力系统的黑箱模型是机器学习中的一个具有挑战性的问题，尤其是当需要渐近稳定性保证时。本文针对线性离散时间状态空间层的状态矩阵，提出了一种新颖的、能够确保稳定性且与反向传播兼容的基于Schur分解的投影方案，同时还给出了该方法的一种替代性的预因子化公式。所提出的方法将状态矩阵实Schur分解得到的准三角因子动态地投影到其最近的稳定对应矩阵上，从而以最小的过参数化代价确保动态系统的稳定性。在合成线性系统上的实验表明，尽管计算复杂度略有增加，该方法的精度和收敛速度仍可与最先进的稳定系统辨识技术相媲美。此外，更少的权重数量有助于（提升训练的便捷性）……

    arXiv:2605.14489v3 Announce Type: replace  Abstract: Building black-box models for dynamical systems from data is a challenging problem in machine learning, especially when asymptotic stability guarantees are required. In this paper, we introduce a novel stability-ensuring and backpropagation-compatible projection scheme based on the Schur decomposition for the state matrix of linear discrete-time state-space layers, as well as an alternative pre-factorized formulation of the methodology. The proposed methods dynamically project the quasi-triangular factor of the state matrix's real Schur decomposition onto its nearest stable peer, ensuring stable dynamics with minimal overparameterization. Experiments on synthetic linear systems demonstrate that, despite a marginal increase in computational complexity, the method achieves accuracy and convergence rates comparable to those of state-of-the-art stable-system identification techniques. Furthermore, the lower weight count facilitates conve
    
[^313]: NodeSyn：面向AI评估的社会对齐合成数据

    NodeSynth: Socially Aligned Synthetic Data for AI Evaluation

    [https://arxiv.org/abs/2605.14381](https://arxiv.org/abs/2605.14381)

    NodeSyn通过将微调的分类体系生成器与可信文献来源和敏感属性相结合，生成反映真实社会背景的敏感领域评估数据，其诱发的模型失败率达到通用合成基线的3.1倍、真实人类查询的1.6倍，显著优于现有合成数据方法。

    

    生成式人工智能的最新进展为模型评估所需的大规模合成数据生成提供了便利。然而，若缺乏针对性的方法，这些数据集往往缺乏敏感领域所需的社会技术层面的细微差别。我们提出了NodeSyn，这是一种将经过微调的分类体系生成器（TaG）与可信的已发表文献来源以及提取的敏感属性相结合的方法，用于生成反映真实社会背景和细微危害的查询。在使用经人工验证的自动评估器，于医疗建议和自我伤害两个领域对十个开源及专有大语言模型进行评估时，NodeSyn在10个模型中的8个上于每个领域均排名第一。NodeSyn数据集引发的平均失败率是通用合成基线的3.1倍，是结构化合成基准的2.4倍，是真实人类撰写查询的1.6倍。消融研究证实，在控制结构混杂因素的情况下，我们细粒度的第3级分类体系…

    arXiv:2605.14381v3 Announce Type: replace-cross  Abstract: Recent advancements in generative AI facilitate large-scale synthetic data generation for model evaluation. However, without targeted approaches, these datasets often lack the sociotechnical nuance required for sensitive domains. We introduce NodeSyn, a methodology that pairs a fine-tuned taxonomy generator (TaG) with credible published sources and extracted sensitive attributes to generate queries reflecting authentic societal contexts and nuanced harms. Evaluated across ten open-weight and proprietary LLMs in Medical Advice and Self-Harm using human-validated autoraters, NodeSyn ranked first on 8 of 10 models in each domain. NodeSyn datasets induced average failure rates 3.1 times higher than generic synthetic baselines, 2.4 times higher than structured synthetic benchmarks, and 1.6 times higher than authentic human-authored queries. Ablation studies confirm that, controlling for structural confounds, our granular Level 3 tax
    
[^314]: 老虎机反馈下多变化点识别的样本复杂度

    The Sample Complexity of Multiple Change Point Identification under Bandit Feedback

    [https://arxiv.org/abs/2605.13252](https://arxiv.org/abs/2605.13252)

    该论文提出了一种在老虎机反馈下识别多个变化点的自适应算法，并证明了其样本预算的非渐近上下界，同时揭示在一般置信水平下样本复杂度不仅取决于跳跃幅度，先前仅由跳跃幅度决定复杂度的认识是不完整的。

    

    我们研究在老虎机反馈下的多变化点定位问题。一个定义在紧致区间上的未知分段常数函数可以在自适应选择的输入处被顺序查询，每次查询返回该函数的一个含噪评估值。目标是在目标精度 η 和置信水平 1-δ 下识别指定数量的不连续点（称为变化点），同时使用尽可能少的样本。我们提出了一种自适应算法，该算法首先检测可能包含变化点的区间，然后将这些变化点的位置精化到精度 η。我们建立了该算法样本预算的非渐近上界，并给出了相应的下界。先前的研究表明，当 δ→0 时，仅跳跃幅度决定了渐近样本复杂度。我们揭示了这一图景在该范围之外是不完整的。我们从实证和理论两方面证明，对于一般的 δ（摘要在此处截断）

    arXiv:2605.13252v2 Announce Type: replace-cross  Abstract: We study multiple change point localization under bandit feedback. An unknown piecewise-constant function on a compact interval can be queried sequentially at adaptively chosen inputs, and each query returns a noisy evaluation of the function. The goal is to identify a prescribed number of discontinuities, known as change points, within a target precision $\eta$ and confidence level $1-\delta$, while using as few samples as possible. We propose an adaptive algorithm that first detects intervals likely to contain change points and then refines their locations to precision $\eta$. We establish non-asymptotic upper bounds on its sample budget, together with corresponding lower bounds. Prior work shows that jump magnitudes alone determine the asymptotic sample complexity as $\delta\to 0$. We reveal that this picture is incomplete beyond this regime. We demonstrate, both empirically and theoretically, that for general $\delta$ and $
    
[^315]: 一种用于学习稳定二值表示的复合激活函数

    A Composite Activation Function for Learning Stable Binary Representations

    [https://arxiv.org/abs/2605.11558](https://arxiv.org/abs/2605.11558)

    提出了一种由sigmoid与tanh复合而成的重尾激活函数（HTAF），作为Heaviside函数的平滑近似，在零输入附近保持较大梯度且尾部梯度衰减更慢，从而实现了脉冲神经网络、二值神经网络和深度Heaviside神经网络的稳定梯度训练。

    

    激活函数通过塑造内部表示在神经网络中发挥着核心作用。近年来，学习二值激活表示因其在计算和内存效率以及可解释性方面的优势而受到广泛关注。然而，使用Heaviside（阶跃）激活函数训练神经网络仍然具有挑战性，因为其不可微性阻碍了标准的基于梯度的优化。在本文中，我们提出了重尾激活函数，它是Heaviside函数的一种平滑近似，能够实现基于梯度优化的稳定训练。我们将HTAF构造为sigmoid与双曲正切函数的复合函数，并从理论上证明它在零输入附近保持了较大的梯度质量，同时在尾部区域表现出更慢的梯度衰减。我们证明了脉冲神经网络、二值神经网络和深度Heaviside神经网络都可以得到稳定的训练。

    arXiv:2605.11558v3 Announce Type: replace  Abstract: Activation functions play a central role in neural networks by shaping internal representations. Recently, learning binary activation representations has attracted significant attention due to their advantages in computational and memory efficiency, as well as interpretability. However, training neural networks with Heaviside activations remains challenging, as their non-differentiability obstructs standard gradient-based optimization. In this paper, we propose Heavy Tailed Activation Function (HTAF), a smooth approximation to the Heaviside function that enables stable training with gradient-based optimization. We construct HTAF as a sigmoid hyperbolic tangent composite function and theoretically show that it maintains a large gradient mass around zero inputs while exhibiting slower gradient decay in the tail regions. We show that Spiking Neural Networks, Binary Neural Networks and Deep Heaviside neural Networks can be trained stably
    
[^316]: 基于超球面流的语言建模

    Language Modeling with Hyperspherical Flows

    [https://arxiv.org/abs/2605.11125](https://arxiv.org/abs/2605.11125)

    提出𝕊-FLM，一种在超球面上通过旋转向量生成序列的潜在流语言模型，避免了基于one-hot向量的高维运算带来的高昂训练成本，并赋予噪声注入更清晰的语义解释。

    

    离散扩散语言模型凭借其并行生成能力，作为自回归（AR）模型的替代方案而迅速发展。然而，出于可处理性的考虑，离散扩散模型从因子化分布中采样，其表达能力不如自回归模型。近期的流语言模型（FLMs）将连续流应用于语言，通过确定性常微分方程（ODE）将噪声传输为数据，从而避免了因子化采样。但FLM在one-hot向量上运行，其维度随词表大小而扩展，导致FLM的训练成本高昂。此外，由于所有不同的one-hot嵌入在ℓ₂距离下彼此等距，添加高斯噪声缺乏清晰的语义解释（这与图像不同，图像中高斯噪声会逐渐破坏结构）。我们提出了𝕊-FLM，一种运行在超球面上的潜在流语言模型。𝕊-FLM通过在𝕊^{d-1}中沿着用交……（交叉熵等）学习到的速度场旋转向量来生成序列。

    arXiv:2605.11125v4 Announce Type: replace  Abstract: Discrete Diffusion Language Models progressed rapidly as an alternative to autoregressive (AR) models, motivated by their parallel generation abilities. However, for tractability, discrete diffusion models sample from a factorized distribution, which is less expressive than AR. Recent Flow Language Models (FLMs) apply continuous flows to language, transporting noise to data with a deterministic ODE that avoids factorized sampling. FLMs operate on one-hot vectors whose dimension scales with the vocabulary size, making FLMs costly to train. Moreover, since all distinct one-hot embeddings are equidistant in $\ell_2$, adding Gaussian noise does not have a clear semantic interpretation (unlike images, where Gaussian noise progressively degrades structure). We introduce $\mathbb{S}$-FLM, a latent FLM in the hypersphere. $\mathbb{S}$-FLM generates sequences by rotating vectors in $\mathbb{S}^{d-1}$ along a velocity field learned with cross-
    
[^317]: Transformers的动理学理论与“迷失在中间”现象

    Kinetic theory for Transformers and the lost-in-the-middle phenomenon

    [https://arxiv.org/abs/2605.09213](https://arxiv.org/abs/2605.09213)

    该论文将Transformer的因果自注意力建模为非可交换相互作用粒子系统，通过平均场极限与累积量展开分析，首次严格证明了“迷失在中间”现象，即词元检索性能随其在提示中的位置呈U形分布。

    

    我们研究因果自注意力动力学——这是解码器Transformer的一个玩具模型——并将其解释为一个非可交换的相互作用粒子系统。通过将累积量展开方法适配到该模型的三角因果依赖结构，并借助基于Glauber演算的非层级方法来估计相关性，我们证明了定量的平均场极限结果以及相关性的下一阶刻画。对于独立同分布且均匀分布的词元，极限相关方程可以闭式求解，从而我们为经验上观察到的“迷失在中间”现象提供了一个严格的解释：词元检索性能作为其在提示中源位置的函数呈U形，在显式的小性条件下表现出首因效应、近因效应以及唯一的内部最小值。

    arXiv:2605.09213v2 Announce Type: replace-cross  Abstract: We study causal self-attention dynamics -- a toy model for decoder Transformers -- which we interpret as a non-exchangeable interacting particle system. Adapting cumulant expansions to the triangular causal dependency structure of the model, and appealing to non-hierarchical methods to estimate correlations using Glauber calculus, we prove a quantitative mean-field limit result and a next-order characterization of correlations. For iid uniformly distributed tokens, the limiting correlation equation can be solved in closed form and we obtain a rigorous explanation of the empirically observed \emph{lost-in-the-middle} phenomenon: the token retrieval profile, as a function of the source position in the prompt, is $\mathsf{U}$-shaped, with primacy, recency, and a unique interior minimum under an explicit smallness condition.
    
[^318]: 面向分布式强化学习的路径耦合贝尔曼流

    Path-Coupled Bellman Flows for Distributional Reinforcement Learning

    [https://arxiv.org/abs/2605.08253](https://arxiv.org/abs/2605.08253)

    提出 PCBF 方法，通过源一致的贝尔曼耦合路径与共享基础噪声进行流匹配，在连续时间框架下学习回报分布，同时避免了投影、边界失配与高方差自举问题。

    

    分布式强化学习（DRL）对完整的回报分布进行建模，但现有的有限支撑或基于分位数的方法依赖于投影操作，而近期的基于流的方法则可能在流源处出现边界失配，或在当前噪声与后继噪声相互独立时产生高方差的自举问题。我们提出了路径耦合贝尔曼流（PCBF），这是一种连续时间的分布式强化学习方法，它通过“源一致且经贝尔曼耦合的路径”利用流匹配来学习回报分布：当前路径在 t=0 时从所需的基础先验出发，在 t=1 时到达贝尔曼目标，并在中间时刻与后继流保持逐路径的仿射关系（无需要求所有时刻 t 的边际分布均满足分布式贝尔曼不动点）。PCBF 通过共享基础噪声将当前与后继的回报流耦合起来，并使用一种以 λ 参数化的控制变量目标……

    arXiv:2605.08253v3 Announce Type: replace  Abstract: Distributional reinforcement learning (DRL) models the full return distribution, but existing finite-support or quantile-based methods rely on projections, while recent flow-based approaches can suffer from \emph{boundary mismatch} at the flow source or from \emph{high-variance} bootstrapping when current and successor noises are independent. We propose Path-Coupled Bellman Flows (PCBF), a continuous-time DRL method that learns return distributions with flow matching using \textbf{source-consistent Bellman-coupled paths}: the current path starts from the required base prior at $t{=}0$, reaches the Bellman target at $t{=}1$, and maintains a pathwise affine relation to the successor flow at intermediate times (without requiring time-$t$ marginals to satisfy a distributional Bellman fixed point for all $t$). PCBF couples current and successor return flows through shared base noise and uses a $\lambda$-parameterized control-variate targe
    
[^319]: RelAgent：将大语言模型智能体作为关系学习的数据科学家

    RelAgent: LLM Agents as Data Scientists for Relational Learning

    [https://arxiv.org/abs/2605.07840](https://arxiv.org/abs/2605.07840)

    提出了RelAgent，一个基于大语言模型的自主数据科学家，它通过构建SQL特征程序并结合经典模型，为关系学习提供快速、确定性且内在可解释的预测。

    

    关系学习是一个具有挑战性的问题，它催生了多种方法，包括基于图的模型（如图神经网络、图Transformer）、表格方法（如表格基础模型）以及基于序列的方法（如大语言模型），每种方法都有其优势和局限性。我们提出了RelAgent，一个基于大语言模型的关系学习自主数据科学家，它分两个阶段运行。在搜索阶段，LLM智能体利用数据库、验证和评估工作区工具来构建SQL特征程序并选择预测模型。在推理阶段，所生成的程序无需进一步的LLM调用即可执行。最终的预测器由SQL查询和一个经典模型组成，能够实现快速、确定性和内在可解释的预测：特征是人类可读的查询，且预测仅依赖于查询定义的特征映射，从而实现……

    arXiv:2605.07840v2 Announce Type: replace  Abstract: Relational learning is a challenging problem that has motivated a wide range of approaches, including graph-based models (e.g., graph neural networks, graph transformers), tabular methods (e.g., tabular foundation models), and sequence-based approaches (e.g., large language models), each with its own advantages and limitations. We propose RelAgent, an LLM-based autonomous data scientist for relational learning, which operates in two phases. In the search phase, an LLM agent uses database, validation, and evaluation workspace tools to construct SQL feature programs and select a predictive model. In the inference phase, the resulting program is executed without further LLM calls. The final predictor consists of SQL queries and a classical model, enabling fast, deterministic, and intrinsically interpretable predictions: features are human-readable queries, and predictions depend only on the resulting query-defined feature map, enabling 
    
[^320]: 标签污染下用于重要性采样与数据剪枝的秩方差评分

    Rank-Variance Scores for Importance Sampling and Data Pruning under Label Contamination

    [https://arxiv.org/abs/2605.07551](https://arxiv.org/abs/2605.07551)

    该论文提出了一种基于K个独立训练代理模型损失排名方差的非单调评分方法，以解决标签污染下基于损失的采样与剪枝反而优先选中污染样本的失效问题，并将总秩方差分解为代理间部分（分歧正则化重要性采样DR-IS）和代理内部分。

    

    方差最优的重要性采样和基于幅值的剪枝方法会优先选取高损失或大梯度的样本，而在标签污染场景下，这些样本往往正是被污染的样本。在Huber的ε-污染模型下，我们精确刻画了这种失效现象：评分采样器访问受损样本的比例由受损样本的平均得分相对数据集平均得分的高低决定，而剪枝子集的污染程度则被评分的污染检测AUROC严格界定；在我们测量的每一种基于损失的评分下，都可以证明剪枝并不比随机选择更“干净”。因此，一个有用的评分必须关于损失是非单调的。我们利用K个独立训练的代理模型的损失排名构建了这样一种评分：被错误标注的样本排名始终最难，而接近决策边界的样本排名则不稳定。其总秩方差（TRV）分解为代理间部分——即分歧正则化重要性采样（DR-IS）——以及代理内部分；该分解预测了……（原文摘要在此截断）

    arXiv:2605.07551v2 Announce Type: replace  Abstract: Variance-optimal importance sampling and magnitude-based pruning prioritize high-loss or large-gradient examples, which under label contamination are often the corrupted ones. Under Huber's $\varepsilon$-contamination model, we characterize this failure exactly: a score sampler's share of visits to corrupted examples is set by their mean score relative to the dataset mean, and a pruned subset's contamination is tightly bounded by the score's corruption-detection AUROC; at every loss-based score we measure, pruning is provably no cleaner than random selection. A useful score must therefore be non-monotone in the loss. We build such a score from the loss ranks of $K$ independently trained proxies: a mislabelled example ranks consistently hardest, a near-boundary example unstably. Its total rank variance (TRV) splits into a between-proxy part, Disagreement-Regularized Importance Sampling (DR-IS), and a within-proxy part; the split predi
    
[^321]: 适应性多元化对齐：一种动态人工民主的流水线

    Adaptive Pluralistic Alignment: A pipeline for dynamic artificial democracy

    [https://arxiv.org/abs/2605.01642](https://arxiv.org/abs/2605.01642)

    本文提出适应性多元化对齐（APA）框架，通过低秩奖励基分解、社会选择理论投票评审团以及随价值观变化的高效权重更新三个阶段，使AI系统能够持续跟踪演变的社会价值观、避免价值锁定，而无需昂贵的重新预训练。

    

    主流的对齐方法针对一组固定的偏好，因此随着社会规范随时间演变，存在强制价值锁定的风险。我们提出了适应性多元化对齐（APA），这是一种模块化流水线，用于更新多元化对齐的AI系统，使其能够跟踪不断演变的价值观并避免价值锁定，而无需重复昂贵的预训练或大规模数据收集。APA包含三个阶段：(1) 通过低秩奖励基分解学习紧凑的个性化奖励模型；(2) 将这些模型作为一个“评审团”，通过基于社会选择理论的投票机制在候选输出中进行集体选择；(3) 随着价值观的变化，通过在固定奖励基上拟合新的标注者权重，高效地随时间调整评审团。由此产生的系统高效、可解释、可控且模块化。我们使用模拟的历史标注者实现了概念验证实例，展示了评审团构成与投票方式选择如何……（摘要被截断）

    arXiv:2605.01642v3 Announce Type: replace  Abstract: Prevailing alignment methods target a fixed set of preferences and therefore risk forcing value lock-in as societal norms evolve over time. We introduce Adaptive Pluralistic Alignment (APA), a modular pipeline for updating pluralistically aligned AI systems to track evolving values and avoid value lock-in without repeating costly pretraining or large-scale data collection. APA has three stages: (1) learning compact personalized reward models via low-rank reward basis decomposition, (2) using these models as a jury that collectively selects among candidate outputs through social-choice-theoretic voting, and (3) efficiently adapting the jury over time by fitting new annotator weights over the fixed reward bases as values shift. The resulting system is efficient, explainable, steerable, and modular. We implement a proof-of-concept instantiation using simulated historical annotators, demonstrate how jury composition and the choice of vot
    
[^322]: ProMORNA：一个用于全长mRNA设计的端到端生成框架

    ProMORNA: An End-to-End Generative Framework for Full-Length mRNA Design

    [https://arxiv.org/abs/2605.01513](https://arxiv.org/abs/2605.01513)

    ProMORNA是一个零样本端到端生成框架，通过BART风格架构预训练及两阶段后训练（冷启动监督微调加多目标强化学习），直接生成全长治疗性mRNA，无需区域化生成或昂贵的推理时搜索。

    

    设计治疗性信使核糖核酸（mRNA）需要生成能够最佳平衡细胞内稳定性和翻译效率的全长转录本。虽然生成模型已经推动了mRNA设计的发展，但现有方法依赖于区域化生成、昂贵的推理时搜索或针对特定目标的模型更新。为了克服这些限制，我们提出了ProMORNA，一个用于全长mRNA设计的零样本端到端生成框架。ProMORNA将设计过程与现代大型语言模型训练流程相对齐：它利用在超过600万个天然蛋白质-mRNA对上预训练的BART风格架构，随后进行包含冷启动监督微调和多目标强化学习的两阶段后训练过程。通过在单个指标层面标准化和聚合相对优势，并结合领域特定的稳定化技术，我们的后训练阶段直接对齐基因（摘要在此处被截断）

    arXiv:2605.01513v2 Announce Type: replace  Abstract: Designing therapeutic messenger RNA (mRNA) requires generating full-length transcripts that optimally balance intracellular stability and translation efficiency. While generative models have advanced mRNA design, existing approaches rely on region-wise generation, expensive inference-time search, or target-specific model updates. To overcome these limitations, we introduce ProMORNA, a zero-shot, end-to-end generative framework for full-length mRNA design. ProMORNA aligns the design process with modern Large Language Model training pipelines: it utilizes a BART-style architecture pre-trained on over 6 million natural protein-mRNA pairs, followed by a two-stage post-training process comprising cold-start supervised fine-tuning and multi-objective reinforcement learning. By standardizing and aggregating relative advantages at the individual metric level with domain-specific stabilization, our post-training stage directly aligns the gene
    
[^323]: CellTwin：基于高分辨率原位测量驱动机器学习的可复现蜂窝网络仿真

    CellTwin: Reproducible Cellular Network Emulation via High Resolution, in situ Measurement-Driven ML

    [https://arxiv.org/abs/2604.26080](https://arxiv.org/abs/2604.26080)

    本文提出首个支持多客户端的机器学习驱动蜂窝网络仿真器CellTwin，直接从高分辨率原位网络遥测数据中学习真实5G调度器的资源分配行为，实现了可复现的高保真网络仿真。

    

    当前和未来的应用需要超低延迟和稳定的吞吐量，然而这些应用常常经由5G蜂窝网络传输，因此必须应对易变的分组动态，因为5G基站调度器会动态响应用户工作负载和无线信道状况。在这些环境中评估网络算法的任务受到现有工具的制约：录制-回放仿真器切断了应用端点与商业运营商专有5G调度器之间存在的反馈交互，而全栈模拟器则依赖于过于简化的调度逻辑。为弥合这一现实差距，我们提出了NeuralEmu，一个高保真、基于机器学习的仿真框架，它直接从极高分辨率的网络遥测工具中学习复杂的5G调度器资源分配行为。作为首个能够处理多客户端的仿真器，NeuralEmu利用机器学习动态预测资源……

    arXiv:2604.26080v2 Announce Type: replace-cross  Abstract: Current and future applications demand ultra-low latency and consistent throughput, yet frequently traverse 5G cellular networks, so cope with volatile packet dynamics, as 5G base station schedulers dynamically react to user workloads and wireless channel conditions. The task of evaluating network algorithms in these environments is hamstrung by current tools: record-and-replay emulators sever the feedback interaction that exists between application end points and a commercial operator's proprietary 5G scheduler, while full-stack simulators rely on overly simplistic scheduling logic. To bridge this reality gap, we present NeuralEmu, a high-fidelity, machine learning-based emulation framework that learns complex 5G scheduler resource allocation behaviors directly from extremely high-resolution network telemetry tools. The first emulator to handle multiple clients, NeuralEmu utilizes machine learning to dynamically predict resour
    
[^324]: 针对量子场论的小型推理模型微调

    Fine-Tuning Small Reasoning Models for Quantum Field Theory

    [https://arxiv.org/abs/2604.18936](https://arxiv.org/abs/2604.18936)

    本文首次对大语言模型在理论物理可验证任务上的强化学习与监督微调进行学术比较，并构建了可生成超过2,500个量子场论问题的数据管线来微调小型推理模型。

    

    尽管大语言模型（LLM）在理论物理中的应用日益增多，但关于领域特定的物理推理能力在训练过程中如何发展的学术探索仍然很少。为了研究这一问题，我们首次从学术角度对大语言模型在理论物理可验证问题求解任务上的强化学习与监督微调进行了比较。由于开发此类能力所需的开源可验证训练数据十分稀缺，我们构建了一个稳健的数据生成管线，它既能创建合成问题，又能将现有的人类撰写问题改造为适合模型训练的形式。我们选择量子场论（QFT）作为主要领域，生成了超过2,500个合成问题，并整理了一个精选的、来自arXiv和标准教学资源的人类改编问题集合。我们开展了强化学习（RL）和监督微调（SFT）两类实验，

    arXiv:2604.18936v2 Announce Type: replace  Abstract: Despite the growing application of large language models (LLMs) to theoretical physics, there has been little academic exploration of how domain-specific physics reasoning ability develops during training. To investigate this, we perform the first academic comparison of reinforcement learning and supervised fine-tuning for LLMs on verifiable problem-solving tasks in theoretical physics. Because the open-source verifiable training data required to develop such capabilities is scarce, we built a robust data generation pipeline that can both create synthetic problems and make existing human-authored problems suitable for model training. Selecting quantum field theory (QFT) as our primary domain, we generated over 2,500 synthetic problems alongside a curated collection of human-adapted problems sourced from arXiv and standard pedagogical resources. We conduct both reinforcement learning (RL) and supervised fine-tuning (SFT) experiments, 
    
[^325]: 扩散模型在模运算上算法泛化的机制研究

    A Mechanistic Study of Algorithmic Generalization in Diffusion Models on Modular Arithmetic

    [https://arxiv.org/abs/2604.17673](https://arxiv.org/abs/2604.17673)

    该研究以模加法任务上的“顿悟”现象为测试平台，从机制上揭示了扩散模型的算法泛化方式：单图像场景下模型通过周期性表示编码操作数并在自注意力-前馈接口处按三角加法恒等式组合实现模加法，而高多样性多图像场景下迭代采样过程则被划分为算术计算与视觉精细化两个阶段。

    

    arXiv:2604.17673v2 公告类型：替换。摘要：尽管扩散模型在实证上取得了巨大成功，但从机制角度对其泛化能力的理解仍然十分有限。在这项工作中，我们对扩散模型中的算法学习提供了一个受控的机制性解释。通过利用基于图像的模加法任务上的“顿悟”现象作为一个易于处理的测试平台，我们剖析了模型在两种不同视觉机制下的内部动态。在单图像机制下，机制剖析揭示出该模型通过将操作数编码为周期性表示，并在自注意力与前馈网络的接口处以一种恢复三角加法恒等式的方式对它们进行组合，从而实现模加法运算。在具有高类内多样性的多图像机制下，我们发现迭代采样过程将任务划分为一个算术计算阶段和一个随后的视觉精细化阶段，两者之间的转变被内部频率精确捕捉。

    arXiv:2604.17673v2 Announce Type: replace  Abstract: Despite their empirical success, how diffusion models generalize remains poorly understood from a mechanistic perspective. In this work, we provide a controlled mechanistic account of algorithmic learning in diffusion models. By utilizing the grokking phenomenon on an image-based modular addition task as a tractable testbed, we dissect the model's internal dynamics across two distinct visual regimes. In a single-image regime, mechanistic dissection reveals that the model implements modular addition by encoding operands as periodic representations and composing them at the self-attention--feedforward interface in a way that recovers the trigonometric addition identities. In a diverse-image regime with high intraclass variability, we find that the iterative sampling process partitions the task into an arithmetic computation phase and a subsequent visual refinement phase, with their transition precisely captured by an internal frequency
    
[^326]: 论多层状态空间模型（SSM）的表达能力与局限性

    On the Expressive Power and Limitations of Multi-Layer SSMs

    [https://arxiv.org/abs/2604.14501](https://arxiv.org/abs/2604.14501)

    多层状态空间模型（SSM）求解函数复合问题时必须满足深度、状态维度与标量精度之间的下界权衡（d²p=Ω(N/L³)），且输入后推理无法绕过该通信下界，从而揭示了多层SSM表达能力的基本限制。

    

    我们研究了深度、有限精度、状态维度以及思维链如何影响多层状态空间模型（SSM）的表达能力。针对显式表的K个函数复合问题——一个序列信息传播的规范基准任务——我们证明，任何求解(L+3)个函数复合的L层SSM都必须满足 d²p=Ω(N/L³)，其中 d 为状态维度，p 为每个标量的精度。反过来，K个函数复合可以由一个 d=1 且 p=Θ(log N) 的(K+1)层广义SSM精确求解。这为该形式化问题族给出了一个最坏情况下的深度层级结构。我们随后区分了“输入后推理”（所有思考标记均在输入之后生成）与“输入交错推理”（思考标记可以在读取输入流的过程中插入）。输入后推理无法绕过我们基于通信的下界证明流程，

    arXiv:2604.14501v2 Announce Type: replace-cross  Abstract: We study how depth, finite precision, state dimension, and chain-of-thought (CoT) affect the expressive power of multi-layer state-space models (SSMs). For the explicit-table $K$-function-composition problem, a canonical benchmark for sequential information propagation, we prove that any $L$-layer SSM solving $(L+3)$-function composition must satisfy $d^2p=\Omega(N/L^3)$, where $d$ is the state dimension and $p$ is the per-scalar precision. Conversely, $K$-function composition is solved exactly by a $(K+1)$-layer generalized SSM with $d=1$ and $p=\Theta(\log N)$. This gives a worst-case depth hierarchy for this formal problem family. We then distinguish post-input reasoning, in which all thought tokens are generated after the input, from input-interleaved reasoning, in which thought tokens may be inserted while the input stream is being read. Post-input reasoning does not circumvent our communication-based lower-bound pipeline,
    
[^327]: 专家混合Transformer的泛化与缩放定律

    Generalization and Scaling Laws for Mixture-of-Experts Transformers

    [https://arxiv.org/abs/2604.09175](https://arxiv.org/abs/2604.09175)

    该论文为专家混合Transformer建立了泛化与缩放理论，证明其泛化误差由活跃参数预算主导并附加路由开销，且近似误差可通过扩展活跃容量或增加专家数量来降低，具体取决于主导瓶颈。

    

    我们为专家混合Transformer建立了一套泛化与缩放理论，该理论清晰地将每个输入的*活跃*容量与路由组合学分离开来。通过以固定路由模式为条件，并对这些模式进行并集界定，我们推导出了一个上确界范数覆盖数界限，其度量熵随活跃参数预算而扩展，并产生MoE特有的路由开销。结合针对平方损失的标准ERM分析，我们在d维流形数据模型和C^β目标函数下得到了一个泛化界限，表明一旦适当考虑活跃参数，近似与估计之间的权衡与稠密网络中的一致。我们进一步为MoE架构证明了一个构造性近似定理，表明在近似构造下，误差可以通过扩展活跃容量或增加专家数量来降低，具体取决于主导瓶颈。我们

    arXiv:2604.09175v2 Announce Type: replace  Abstract: We develop a theory of generalization and scaling for Mixture-of-Experts (MoE) Transformers that cleanly separates \emph{active} per-input capacity from routing combinatorics. By conditioning on fixed routing patterns and union-bounding across them, we derive a sup-norm covering-number bound whose metric entropy scales with the active parameter budget and incurs a MoE-specific routing overhead. Combined with a standard ERM analysis for squared loss, this yields a generalization bound under a $d$-dimensional manifold data model and $C^\beta$ targets, showing that approximation and estimation trade off as in dense networks once active parameters are accounted for appropriately. We further prove a constructive approximation theorem for MoE architectures, showing that, under the approximation construction, error can decrease either by scaling active capacity or by increasing the number of experts, depending on the dominant bottleneck. Fr
    
[^328]: 步进信息性假设：熵动态何时能追踪大语言模型的推理正确性？

    The Stepwise Informativeness Assumption: When Do Entropy Dynamics Track Reasoning Correctness in LLMs?

    [https://arxiv.org/abs/2604.06192](https://arxiv.org/abs/2604.06192)

    本文提出步进信息性假设（SIA），从理论上刻画了熵降低何时能真实反映大语言模型推理的正确性进展，并指出这种对齐取决于训练机制将推理轨迹与正确答案耦合的方式，同时给出了该现象的可观测特征和失效情形。

    

    近期的研究使用基于熵的信号在多个表示层次上研究大语言模型的推理，但该领域在很大程度上仍停留在经验层面。从理论角度看，模型不确定性的降低并不一定反映其朝正确性的进展。在这项工作中，我们通过步进信息性假设（Stepwise Informativeness Assumption, SIA）来刻画何时可以预期这种对齐关系，该假设要求降低熵的推理前缀扩展在期望上增加分配给有效答案的概率。我们表明，下一词元预测能够将与答案相关的信息从训练数据传递到模型，但其本身并不能保证与正确性对齐。相反，我们提出这种对齐取决于训练机制如何将推理轨迹与正确答案相耦合。此外，我们假设了这一现象的可观测特征，并刻画了其预期的失效情形。（原文摘要在此处截断）

    arXiv:2604.06192v2 Announce Type: replace  Abstract: Recent work uses entropy-based signals at multiple representation levels to study reasoning in large language models, but the field remains largely empirical. From a theoretical perspective, reductions in model uncertainty do not necessarily reflect progress toward correctness. In this work, we characterize when such alignment can be expected through the Stepwise Informativeness Assumption (SIA), which requires entropy-reducing reasoning-prefix extensions increase the probability assigned to valid answers in expectation. We show that next-token prediction can transfer answer-relevant information from training data to the model, but does not, by itself, guarantee alignment with correctness. Instead, we propose that this alignment depends on how the training regime couples reasoning traces to correct answers. Furthermore, we hypothesize observable signatures of this phenomenon and characterize its expected failure regimes. We empirical
    
[^329]: FactReview：基于证据的同行评审与执行式声明验证

    FactReview: Evidence-Grounded Peer Review with Execution-Based Claim Verification

    [https://arxiv.org/abs/2604.04074](https://arxiv.org/abs/2604.04074)

    FactReview通过结合文献、参考检查和代码执行的多源证据审计，显著提升了论文声明验证的准确性和评审效率。

    

    基于大型语言模型（LLM）的评审系统通常孤立地评估稿件，导致依赖文献和代码的声明难以验证。我们提出了FactReview，一个审计流水线，它提取与评审相关的声明，将其与相关工作及参考文献检查进行关联，并在代码可用时，在固定修复预算下执行发布的工件。在26篇论文不相交的测试论文及354个人工验证声明上，FactReview的声明恢复F1分数达到84.3%。在相同后端和证据匹配的比较中，FactReview总体得分4.72/5，比直接LLM评审者高出0.74分。移除执行证据会改变17.0%的声明状态，比移除任何其他单一证据来源的影响更大。在评审辅助研究中，FactReview将平均评审时间减少58%，同时将基准声明覆盖率从87%提高到99%。FactReview支持基于证据的声明审计，并具有接受性。

    arXiv:2604.04074v4 Announce Type: replace  Abstract: Large language model (LLM)-based reviewing systems typically assess manuscripts in isolation, leaving literature- and code-dependent claims difficult to verify. We present FactReview, an audit pipeline that extracts review-relevant claims, grounds them in related work and reference checks, and, when code is available, executes released artifacts under a fixed repair budget. On 26 paper-disjoint test papers with 354 human-verified claims, FactReview achieves 84.3\% F1 for claim recovery. In a same-backend, evidence-matched comparison, FactReview scores 4.72/5 overall, outperforming a direct LLM reviewer by 0.74 points. Removing execution evidence changes 17.0\% of claim statuses, more than removing any other single evidence source. In a reviewer-assistance study, FactReview reduces mean review time by 58\% while increasing benchmark-claim coverage from 87\% to 99\%. FactReview supports evidence-based claim auditing, with acceptance de
    
[^330]: 大语言模型在线策略蒸馏综述

    A Survey of On-Policy Distillation for Large Language Models

    [https://arxiv.org/abs/2604.00626](https://arxiv.org/abs/2604.00626)

    这篇综述论文通过学生生成状态、教师信息与策略更新三者之间的相互作用，建立了在线策略蒸馏（OPD）的统一分析框架，并将其与自蒸馏、多教师集成、强化学习和交互式智能体等方向联系起来，系统阐释了不同方法的设计选择对能力迁移、训练效率和部署的影响。

    

    在线策略蒸馏在大语言模型的训练中，由教师模型对学生模型自身生成的轨迹提供指导，从而解决了固定示范与部署时实际遇到的交互历史之间的不匹配问题。随着其在语言模型训练中的应用不断扩展，不同的监督与优化形式使得各方法之间难以比较，也难以理解它们在何种情况下有效。本综述通过学生生成状态、教师信息与策略更新三者之间的相互作用，构建了对在线策略蒸馏的统一视角。该框架将蒸馏的基础方法与自蒸馏、多教师集成、强化学习以及交互式智能体联系起来。我们考察了这些设置如何改变学生模型可获得的监督信号，并将由此产生的设计选择与能力迁移、训练效率及部署应用相关联。我们的综述聚焦于教师指导在何种条件下能够持续（改进学生模型）……

    arXiv:2604.00626v5 Announce Type: replace-cross  Abstract: On-Policy Distillation (OPD) trains large language models with teacher guidance on student-generated trajectories, addressing the mismatch between fixed demonstrations and the histories encountered during deployment. As its use expands across language model training, different formulations of supervision and optimization make it difficult to compare methods and understand when they are effective. This survey develops a unified view of OPD through the interplay between student-generated states, teacher information, and policy updates. This framework connects the foundations of distillation with self-distillation, multi-teacher integration, reinforcement learning, and interactive agents. We examine how these settings change the supervision available to the student and relate the resulting design choices to capability transfer, training efficiency, and deployment. Our synthesis centers on the conditions under which guidance remain
    
[^331]: 关于动物自动识别中背部标记设计的见解

    Insights on back marking for the automated identification of animals

    [https://arxiv.org/abs/2603.25535](https://arxiv.org/abs/2603.25535)

    本研究通过训练ResNet-50神经网络对猪的背部标记进行分类分析，揭示了标记设计的核心原则——标记必须在运动模糊、多视角和遮挡条件下保持清晰无歧义，从而为基于机器学习的动物个体自动识别提供设计指导。

    

    迄今为止，关于如何设计背部标记以最好地支持对猪等外观相似物种进行个体级别监测的研究很少。随着基于机器学习的监测解决方案的激增，特别需要关于如何设计能被此类算法有效识别的标记的指导方针。本研究通过分析一个训练用于通过背部标记区分猪的机器学习模型，为有效的背部标记设计提供了宝贵的见解。具体而言，研究者训练了一个ResNet-50类型的神经网络来对十头具有独特背部标记的猪进行分类。对模型预测的分析强调了某些设计选择的重要性。最重要的是，背部标记的设计必须确保每个标记在运动模糊、多样化视角以及由动物行为引起的遮挡条件下保持无歧义。所生成的见解可以支持个体级监测应用。

    arXiv:2603.25535v2 Announce Type: replace-cross  Abstract: To date, there is little research on how to design back marks to best support individual-level monitoring of uniform looking species like pigs. With the recent surge of machine learning-based monitoring solutions, there is a particular need for guidelines on the design of marks that can be effectively recognised by such algorithms. This study provides valuable insights on effective back mark design, based on the analysis of a machine learning model, trained to distinguish pigs via their back marks. Specifically, a neural network of type ResNet-50 was trained to classify ten pigs with unique back marks. The analysis of the model's predictions highlights the significance of certain design choices. Most importantly, the set of back marks must be designed such that each mark remains unambiguous under conditions of motion blur, diverse view angles and occlusions, caused by animal behaviour. The generated insights can support individ
    
[^332]: MCLR：通过类间似然比最大化改进条件建模，并将无分类器引导与对齐目标相统一

    MCLR: Improving Conditional Modeling via Inter-Class Likelihood-Ratio Maximization and Unifying Classifier-Free Guidance with Alignment Objectives

    [https://arxiv.org/abs/2603.22364](https://arxiv.org/abs/2603.22364)

    本文提出MCLR微调目标，通过显式最大化类间似然比来强化类特定结构，从而在不依赖无分类器引导的情况下显著提升扩散模型的条件生成能力，并缩小了与CFG的差距。

    

    arXiv:2603.22364v4 公告类型：替换。摘要：扩散模型具有强大的生成性能，但通常依赖于无分类器引导（CFG），这是一种在推理时修改采样轨迹的干预手段。由于标准的去噪分数匹配（DSM）在分数被准确学习时应能恢复目标分布，CFG在实践中的必要性暗示了学习到的条件分数存在缺陷。我们识别出类间分离较弱正是这样一种局限：在不同标签条件下从相同噪声生成的样本可能保留相似的类无关结构。为解决这一缺陷，我们提出了MCLR，一个有原则的微调目标，它显式地最大化类间似然比，从而在标准采样下强化类特定的结构。在多种模型和数据集上，MCLR显著改进了无引导的条件生成，并缩小了与CFG之间的差距。在理论上，我们证明了CFG引导的分数是...

    arXiv:2603.22364v4 Announce Type: replace  Abstract: Diffusion models achieve strong generative performance but often rely on classifier-free guidance (CFG), an inference-time intervention that modifies the sampling trajectory. Since standard denoising score matching (DSM) should recover the target distribution when the score is learned accurately, the practical necessity of CFG suggests deficiencies in learned conditional scores. We identify weak inter-class separation as one such limitation: samples generated from the same noise under different labels can retain similar class-agnostic structure. To address this deficiency, we introduce MCLR, a principled fine-tuning objective that explicitly maximizes inter-class likelihood-ratios, thereby strengthening class-specific structures under standard sampling. Across diverse models and datasets, MCLR substantially improves guidance-free conditional generation and narrows the gap to CFG. Theoretically, we prove that the CFG-guided score is t
    
[^333]: 基于镜像下降与熵约束的流匹配策略优化

    Flow Matching Policy Optimization with Mirror Descent and Entropy Constraints

    [https://arxiv.org/abs/2603.17685](https://arxiv.org/abs/2603.17685)

    本文提出FMER框架，证明基于ODE的流匹配可同时实现无需仿真的策略优化与可解析计算的熵，并通过镜像下降与熵约束在在线强化学习中有效平衡策略表达能力与探索-利用权衡。

    

    arXiv:2603.17685v4 公告类型：替换。摘要：在在线强化学习（RL）中，如何平衡策略表达能力与探索-利用权衡是一个核心挑战。尽管基于随机微分方程（SDE）的扩散策略能够表示复杂的多峰动作分布，但其存在两个关键局限：其随机逆过程使得熵难以计算（因而必须依赖启发式探索），且通过冗长的去噪链计算策略梯度代价高昂且不稳定。在本工作中，我们证明了基于常微分方程（ODE）的流匹配能够从根本上解决这些问题，既可实现无需仿真的策略优化，又能进行可解的熵计算。在此基础上，我们提出了基于镜像下降与熵约束的流匹配策略优化框架（FMER）。我们的框架从三个方面利用了这一洞见：首先，我们推导出解析形式的熵目标，以校正由 tanh 变换引起的密度畸变……（摘要在此处被截断）

    arXiv:2603.17685v4 Announce Type: replace  Abstract: Balancing policy expressiveness with the exploration-exploitation trade-off is a core challenge in online Reinforcement Learning (RL). While Stochastic Differential Equation (SDE)-based diffusion policies can represent complex, multimodal action distributions, they suffer from two critical limitations: their stochastic reverse processes render entropy intractable (necessitating heuristic exploration), and computing policy gradients through long denoising chains is expensive and unstable. In this work, we show that ODE-based flow matching inherently resolves these issues by enabling both simulation-free policy optimization and tractable entropy computation. Building on this, we introduce Flow Matching Policy Optimization with Mirror Descent and Entropy Constraints (FMER). Our framework exploits this insight in three ways. First, we derive an analytic entropy objective that corrects for the density distortion caused by the $\tanh$ tran
    
[^334]: 端到端自动驾驶中的零样本跨城市泛化：自监督表示与监督表示的对比

    Zero-Shot Cross-City Generalization in End-to-End Autonomous Driving: Self-Supervised versus Supervised Representations

    [https://arxiv.org/abs/2603.11417](https://arxiv.org/abs/2603.11417)

    该研究将零样本跨城市迁移作为端到端自动驾驶的表示级压力测试，通过在严格地理划分下系统对比I-JEPA、DINOv2和MAE等自监督骨干网络与监督式预训练表示，揭示了视觉预训练方式对地理域偏移下泛化能力的关键影响。

    

    端到端自动驾驶模型通常在多城市数据集上使用监督式的ImageNet预训练骨干网络进行训练，然而这些模型对未见城市的泛化能力在很大程度上尚未得到检验。当训练与评估数据在地理上混合时，模型可能会隐式地依赖特定城市的线索，从而掩盖在真实世界域偏移下泛化到新地点时可能出现的失败模式。在这项工作中，我们将零样本跨城市迁移形式化为端到端自动驾驶的一种受控表示级压力测试，并探究视觉预训练如何影响地理域偏移下的迁移行为。我们通过将自监督骨干网络I-JEPA、DINOv2和MAE集成到规划框架中，开展了全面的研究。我们在nuScenes的开环设置以及NAVSIM的闭环评估协议下，在严格的地理划分条件下评估模型性能。我们的实验揭示……

    arXiv:2603.11417v3 Announce Type: replace-cross  Abstract: End-to-end autonomous driving models are typically trained on multi-city datasets using supervised ImageNet-pretrained backbones, yet their ability to generalize to unseen cities remains largely unexamined. When training and evaluation data are geographically mixed, models may implicitly rely on city-specific cues, masking failure modes that would occur under real-world domain shifts when generalizing to new locations. In this work, we formulate zero-shot cross-city transfer as a controlled representation-level stress test for end-to-end autonomous driving and ask how visual pretraining affects transfer behavior under geographic domain shift. We conduct a comprehensive study by integrating self-supervised backbones I-JEPA, DINOv2, and MAE into planning frameworks. We evaluate performance under strict geographic splits on nuScenes in the open-loop setting and on NAVSIM in the closed-loop evaluation protocol. Our experiments reve
    
[^335]: SYNAPSE：面向序列编码的神经元分析与扰动框架

    SYNAPSE: Framework for Neuron Analysis and Perturbation in Sequence Encoding

    [https://arxiv.org/abs/2603.08424](https://arxiv.org/abs/2603.08424)

    SYNAPSE是一个无需重新训练的系统化框架，通过提取逐层[CLS]表示、线性探针获取神经元排名以及前向钩子干预，实现对Transformer模型内部行为的透明分析与压力测试。

    

    近年来，人工智能已成为数据分析、预测和问题解决等复杂任务的强大伙伴，但其缺乏透明度引发了人们对其可靠性的担忧。在医疗保健或网络安全等敏感领域，确保透明度、可信度和鲁棒性至关重要，因为错误决策或遭受成功攻击的后果可能非常严重。为了克服这些局限，本工作提出了SYNAPSE，这是一个系统性的、无需训练的框架（无需对基础模型进行重新训练），用于理解和压力测试Transformer模型在不同领域的内部行为。SYNAPSE提取各层的[CLS]表示，训练一个轻量级线性探针以获得全局和逐类别的神经元排名，并在推理过程中应用前向钩子干预。这种设计使得可以在不改变基础模型的情况下，对内部表示进行受控的、可重复的实验。

    arXiv:2603.08424v2 Announce Type: replace  Abstract: In recent years, Artificial Intelligence has become a powerful partner for complex tasks such as data analysis, prediction, and problem-solving, yet its lack of transparency raises concerns about its reliability. In sensitive domains such as healthcare or cybersecurity, ensuring transparency, trustworthiness, and robustness is essential, since the consequences of wrong decisions or successful attacks can be severe. To overcome these limitations, this work proposes SYNAPSE, a systematic, training-free framework (requiring no retraining of the base model) for understanding and stress-testing the internal behavior of Transformer models across domains. SYNAPSE extracts per-layer [CLS] representations, trains a lightweight linear probe to obtain global and per-class neuron rankings, and applies forward-hook interventions during inference. This design enables controlled, repeatable experiments on internal representations without altering t
    
[^336]: 科学的异域空间：采样连贯但认知上不可及的研究方向

    The Alien Space of Science: Sampling Coherent but Cognitively Unavailable Research Directions

    [https://arxiv.org/abs/2603.01092](https://arxiv.org/abs/2603.01092)

    该论文提出一种方法，通过将文献分解为“想法原子”并联合建模连贯性与认知可得性，来系统性地发现科学上合理但现有研究社区不太可能自发提出的研究方向。

    

    科学发现不仅受制于什么是真实的，还受制于研究者认知上可及的范围。许多研究方向从文献来看是连贯的，却不太可能被提出，因为没有现有的研究社区同时具备相关的概念、方法和直觉。现代语言模型继承了这种偏见，在被要求提出新想法时往往只会重组文献中的高密度区域。我们瞄准的是与之互补的区域——科学的异域空间：即在科学上合理、但在现有研究者分布下不太可能出现的研究方向。我们的方法将论文分解为概念单元，并将其聚类为一个“想法原子”词表。连贯性模型评估原子的组合能否构成可行的研究方向，而可得性模型评估是否存在任何现有作者群体有能力产出这些方向。我们对组合进行排序，以最大化连贯性并最小化可得性。

    arXiv:2603.01092v3 Announce Type: replace-cross  Abstract: Scientific discovery is constrained not only by what is true, but by what is cognitively available to researchers. Many directions are coherent in light of the literature yet unlikely to be proposed because no existing community combines the relevant concepts, methods, and intuitions. Modern language models inherit this bias, recombining high-density regions of the literature when prompted for novel ideas. We target the complementary region, the alien space of science: directions that are scientifically plausible but unlikely under the distribution of existing researchers. Our method decomposes papers into conceptual units and clusters them into a vocabulary of idea atoms. A coherence model scores whether atom combinations form viable research directions, while an availability model scores whether any existing author community is positioned to produce them. We rank combinations to maximize coherence while minimizing availabilit
    
[^337]: MARS：通过自适应秩搜索实现多模态收敛的协调

    MARS: Harmonizing Multimodal Convergence via Adaptive Rank Search

    [https://arxiv.org/abs/2603.00720](https://arxiv.org/abs/2603.00720)

    提出MARS方法，利用双缩放定律自适应搜索LoRA秩对，以平衡多模态大语言模型微调中不同模态的收敛速度，避免负面干扰并提升任务性能。

    

    使用低秩适应（LoRA）等参数高效方法对多模态大语言模型（MLLMs）进行微调对于任务适配至关重要。然而，跨模态之间不平衡的训练动态常常因负面干扰而导致次优的准确率，这一挑战通常采用低效的启发式方法来解决，例如手动调整不同模态的学习率。为了克服这一问题，我们提出了MARS（多模态自适应秩搜索），一种能够发现最佳秩对的方法，在平衡训练动态的同时最大化性能。我们的关键创新是提出的双缩放定律框架，为该搜索提供了支撑：一个定律建模模块特定的收敛时间，将搜索空间剪枝为动态对齐的候选集合；另一个定律则预测最终任务性能，从剪枝后的集合中选出最优的秩对。通过将LoRA的秩重新用作控制模态特定收敛速度的控制器，MARS在……

    arXiv:2603.00720v2 Announce Type: replace  Abstract: Fine-tuning Multimodal Large Language Models (MLLMs) with parameter-efficient methods like Low-Rank Adaptation (LoRA) is crucial for task adaptation. However, imbalanced training dynamics across modalities often lead to suboptimal accuracy due to negative interference, a challenge typically addressed with inefficient heuristic methods such as manually tuning separate learning rates. To overcome this, we introduce MARS (Multimodal Adaptive Rank Search), an approach to discover optimal rank pairs that balance training dynamics while maximizing performance. Our key innovation, a proposed framework of dual scaling laws, enables this search: one law models module-specific convergence time to prune the search space to candidates with aligned dynamics, while the other predicts final task performance to select the optimal pair from the pruned set. By re-purposing the LoRA rank as a controller for modality-specific convergence speed, MARS out
    
[^338]: GLUScope：一个用于分析Transformer语言模型中GLU神经元的工具

    GLUScope: A Tool for Analyzing GLU Neurons in Transformer Language Models

    [https://arxiv.org/abs/2602.23826](https://arxiv.org/abs/2602.23826)

    GLUScope是一个开源工具，通过分析基于GLU激活函数的语言模型中神经元门控与输入激活的四种正负符号组合（激活象限），帮助可解释性研究者揭示神经元促进或抑制特定token等截然不同的功能行为。

    

    我们提出了GLUScope，这是一个面向可解释性研究人员的开源资源，用于分析基于Transformer的语言模型（LM）中的神经元。我们聚焦于现已标准化、基于GLU的激活函数（如SwiGLU），这类激活函数几乎被所有近期开源权重的语言模型所采用。这带来了一个新的挑战：仅理解正激活是不够的。相反，神经元的门控激活和输入激活都可能为正或为负，从而产生四种不同的符号组合（激活象限），这些组合在某些情况下具有截然不同的功能（例如，促进还是抑制特定的一组token）。因此，对于任意神经元，我们的工具会展示四种符号组合各自的文本示例，并指出每种组合出现的频率。我们的资源能够带来新颖的洞见：特别是，得益于对所有四个激活象限的分析，它能够揭示原本可能被忽视的可解释行为。

    arXiv:2602.23826v2 Announce Type: replace  Abstract: We present **GLUScope**, an open-source resource for analyzing neurons in Transformer-based language models (LMs), intended for interpretability researchers. We focus on the now-standard *GLU-based activation functions* such as SwiGLU, used in virtually all recent open-weights LMs. This introduces a new challenge: understanding positive activations is not enough. Instead, both the *gate* and the *in* activation of a neuron can be positive or negative, leading to four different possible sign combinations (*activation quadrants*) that in some cases have quite different functionalities (e.g. promoting vs. suppressing a specific set of tokens). Accordingly, for any neuron, our tool shows text examples for each of the four sign combinations, and indicates how often each combination occurs. Our resource can lead to novel insights: in particular, thanks to analyzing all four activation quadrants, it can surface interpretable behavior that o
    
[^339]: 两步智能体：贝叶斯决策者如何从AI决策支持中学习

    2-Step Agent: How a Bayesian Decision Maker Learns from AI-Decision Support

    [https://arxiv.org/abs/2602.21889](https://arxiv.org/abs/2602.21889)

    本文提出“两步智能体”这一通用计算框架，用以刻画机器学习预测如何作为信息来源改变理性贝叶斯决策者的信念，进而影响因果效应估计、下游决策及其结果，并在多元线性-高斯设定下给出了可解的贝叶斯推断求解方案。

    

    arXiv:2602.21889v4 公告类型：replace-cross 摘要：机器学习模型的预后预测在多个领域支持人类决策，包括医疗和司法等高风险领域。然而，我们仍然缺乏对决策者如何从基于机器学习的决策支持（ML-DS）中学习的清晰理解。在本文中，我们引入了一个通用的计算框架——“两步智能体”（2-Step Agent），用以刻画机器学习模型的预测如何同时被用作关于训练数据及其生成该数据的因果机制的有用信息来源。我们的框架建模了： 机器学习模型的预测如何影响一个理性贝叶斯智能体的信念，以及 这种信念的变化如何影响因果效应的估计、针对新观测的下游决策以及随后的结果。除了该框架本身之外，我们还做出了三项贡献。首先，在多元线性-高斯设定下，我们推导出了一个可解的贝叶斯推……（摘要原文在此处被截断）

    arXiv:2602.21889v4 Announce Type: replace-cross  Abstract: Prognostic predictions from ML models support human decision making in several fields, including high-stakes ones such as healthcare and the judiciary. Yet, we still lack a clear understanding of how decision makers learn from ML-based decision support (ML-DS). In this paper, we introduce a general computational framework, the 2-Step Agent, to capture how a prediction from an ML model can also be used as a source of useful information about the training data and the causal mechanism that generated it. Our framework models (i) how a prediction from an ML model affects the beliefs of a rational Bayesian agent, and (ii) how this change in beliefs affects the estimation of the causal effect, the downstream decision for a new observation, and the subsequent outcome. In addition to the framework itself, we make three contributions. First, for the multivariate linear-Gaussian setting, we derive a tractable solution for the Bayesian in
    
[^340]: 基于信息论的思维链可监控性分析与改进

    Analyzing and Improving Chain-of-Thought Monitorability Through Information Theory

    [https://arxiv.org/abs/2602.18297](https://arxiv.org/abs/2602.18297)

    本文用信息论揭示了思维链可监控性的必要非充分条件，识别出信息鸿沟和引导误差两类失效来源，发现思维链混淆可在无优化压力下自发出现，并证明可通过针对性干预系统性缓解监控的脆弱性。

    

    思维链监控器是一类基于大语言模型的系统，通过分析推理轨迹来检测输出是否可能表现出特定关注属性，例如代码生成过程中“钻测试空子”的行为。本文利用信息论分析表明，思维链与输出之间具有非零互信息是思维链可监控性的必要条件，但并非充分条件。我们识别出两类可能在实际中削弱思维链监控器性能的近似误差来源：信息鸿沟，衡量监控器能够从思维链中提取可用信息的程度；以及引导误差，衡量监控器逼近最优监控函数的程度。实证研究显示，即使不存在任何针对思维链的优化压力，思维链混淆仍可能出现。我们进一步证明，思维链可监控性的脆弱性可以通过针对性干预得到系统性缓解。

    arXiv:2602.18297v2 Announce Type: replace-cross  Abstract: Chain-of-thought (CoT) monitors are LLM-based systems that analyze reasoning traces to detect when outputs may exhibit attributes of interest, such as test-hacking behavior during code generation. In this paper, we use information-theoretic analysis to show that non-zero mutual information between CoT and output is a necessary but not sufficient condition for CoT monitorability. We identify two sources of approximation error that may undermine the performance of CoT monitors in practice: information gap, which measures the extent to which the monitor can extract the information available in CoT, and elicitation error, which measures the extent to which the monitor approximates the optimal monitoring function. Empirically, we show that CoT obfuscation can arise even in the absence of any optimization pressure against CoT. We further demonstrate that the fragility of CoT monitorability can be systematically mitigated through targ
    
[^341]: 基于删失反馈的资源分配学习

    Learning to Allocate Resources with Censored Feedback

    [https://arxiv.org/abs/2602.06565](https://arxiv.org/abs/2602.06565)

    该论文研究了删失反馈下的在线预算分配问题，证明了 $\Omega(\sqrt{T})$ 的最坏情况遗憾下界，并提出了一种结合非平凡参数估计与置信界的乐观算法 RA-UCB，用于联合估计未知参数并最大化累积奖励。

    

    我们研究了一个在线资源分配问题：在每一轮中，需要在删失反馈下将预算 $B$ 分配给 $K$ 个臂。当且仅当满足以下两个条件时，某个臂才会产生奖励：(i) 该臂根据一个具有未知参数的、臂特定的伯努利随机变量被激活；(ii) 分配的预算超过从具有未知参数的参数分布中抽取的随机阈值。在 $T$ 轮过程中，学习者必须联合估计未知参数并分配预算，以在面临探索-利用权衡的同时最大化累积奖励。我们证明了一般模型下最坏情况遗憾下界为 $\Omega(\sqrt{T})$，即使阈值分布为已知的均匀分布也是如此。随后我们提出了 RA-UCB，这是一种利用非平凡参数估计和置信界的乐观算法。当预算 $B$ 在每轮开始时已知时，RA-UCB 实现了……（原文摘要在此处截断）

    arXiv:2602.06565v2 Announce Type: replace  Abstract: We study an online resource allocation problem in which at each round, a budget $B$ must be allocated across $K$ arms under censored feedback. An arm yields a reward if and only if two conditions are satisfied: (i) the arm is activated according to an arm-specific Bernoulli random variable with unknown parameter, and (ii) the allocated budget exceeds a random threshold drawn from a parametric distribution with unknown parameter. Over $T$ rounds, the learner must jointly estimate the unknown parameters and allocate the budget so as to maximize cumulative reward facing the exploration--exploitation trade-off. We prove a worst-case regret lower bound $\Omega(\sqrt{T})$ for the general model, even with known uniform threshold distributions. We then propose RA-UCB, an optimistic algorithm that leverages non-trivial parameter estimation and confidence bounds. When the budget $B$ is known at the beginning of each round, RA-UCB achieves a re
    
[^342]: RuleSmith：用于自动化游戏平衡的多智能体大语言模型框架

    RuleSmith: Multi-Agent LLMs for Automated Game Balancing

    [https://arxiv.org/abs/2602.06232](https://arxiv.org/abs/2602.06232)

    RuleSmith框架将多智能体大语言模型自博弈与高斯过程贝叶斯优化相结合，在参数化的规则空间中自动搜索游戏平衡配置，并在简化的文明风格游戏CivMini上实现了接近均衡的双阵营胜率。

    

    游戏平衡是一个长期存在的挑战，需要反复的游戏测试、专家直觉和大量的人工调优。我们提出了RuleSmith，这是一个将多智能体大语言模型自博弈与基于代理模型的优化相结合、作用于参数化规则空间的框架。作为概念验证，我们构建了CivMini，一个简化的文明风格游戏，其中包含由12个可调参数控制的两个非对称阵营。大语言模型智能体解读文本形式的规则手册和游戏状态，提出行动建议，这些行动由游戏引擎验证，从而产生带噪声的阵营胜率估计。我们的主要实现采用高斯过程贝叶斯优化，并配合基于期望改进的自适应采样，将更多评估对局分配给有前景的候选配置。RuleSmith始终能够找到接近平衡的配置：三次独立的优化运行在独立的200局和300局评估中均达到了接近均衡的经验胜率，此外的实验进一步……

    arXiv:2602.06232v2 Announce Type: replace  Abstract: Game balancing is a longstanding challenge requiring repeated playtesting, expert intuition, and extensive manual tuning. We introduce RuleSmith, a framework that couples multi-agent LLM self-play with surrogate-based optimization over a parameterized rule space. As a proof of concept, we build CivMini, a simplified civilization-style game with two asymmetric factions governed by 12 tunable parameters. LLM agents interpret textual rulebooks and game states to propose actions that are validated by the game engine, yielding noisy estimates of faction win rates. Our primary implementation uses Gaussian-process Bayesian optimization with Expected-Improvement-based adaptive sampling to allocate more evaluation games to promising candidates. RuleSmith consistently finds near-balanced configurations: three independent optimization runs reach near-even empirical win rates in independent 200- and 300-game evaluations, and additional experimen
    
[^343]: 基于证据不确定性量化的大型视觉-语言模型不当行为检测

    Detecting Misbehaviors of Large Vision-Language Models by Evidential Uncertainty Quantification

    [https://arxiv.org/abs/2602.05535](https://arxiv.org/abs/2602.05535)

    提出了一种细粒度的证据不确定性量化方法（EUQ），通过同时捕捉知识冲突与信息缺失这两类认知不确定性，有效检测大型视觉-语言模型的不当行为。

    

    大型视觉-语言模型（LVLMs）在多模态理解与生成方面取得了显著进展。然而，当面对能力不足或对抗性的输入时，它们经常产生不可靠甚至有害的内容，例如事实幻觉或危险指令。这种与人类期望的不一致，被称为LVLMs的“不当行为”，对其在关键应用中的部署引发了严重的担忧。研究发现这些不当行为源于认知不确定性，具体表现为内部知识冲突或支持信息的缺失。然而，现有的不确定性量化方法通常只能捕捉整体的认知不确定性，在识别此类问题方面效果有限。为弥补这一空白，我们提出了证据不确定性量化（EUQ），这是一种细粒度的方法，能够同时捕捉信息冲突与无知，从而有效检测……

    arXiv:2602.05535v3 Announce Type: replace  Abstract: %Large vision-language models (LVLMs) have shown substantial advances in multimodal understanding and generation. However, when presented with incompetent or adversarial inputs, they frequently produce unreliable or even harmful content, such as fact hallucinations or dangerous instructions. This misalignment with human expectations, referred to as \emph{misbehaviors} of LVLMs, raises serious concerns for deployment in critical applications. These misbehaviors are found to stem from epistemic uncertainty, specifically either conflicting internal knowledge or the absence of supporting information. However, existing uncertainty quantification methods, which typically capture only overall epistemic uncertainty, have shown limited effectiveness in identifying such issues. To address this gap, we propose Evidential Uncertainty Quantification (EUQ), a fine-grained method that captures both information conflict and ignorance for effective d
    
[^344]: HealthMamba：一种不确定性感知的时空图状态空间模型，用于有效且可靠的医疗机构就诊预测

    HealthMamba: An Uncertainty-aware Spatiotemporal Graph State Space Model for Effective and Reliable Healthcare Facility Visit Prediction

    [https://arxiv.org/abs/2602.05286](https://arxiv.org/abs/2602.05286)

    提出了不确定性感知的时空框架HealthMamba，通过统一时空上下文编码器融合静态与动态信息、利用新型图状态空间模型GraphMamba进行分层时空建模，实现准确且可靠的医疗机构就诊预测。

    

    医疗机构就诊预测对于优化医疗资源配置和制定公共卫生政策至关重要。尽管现有研究采用了先进的机器学习方法以提升预测性能，但它们通常将此任务表述为时间序列预测问题，而未考虑不同类型医疗机构之间固有的空间依赖关系，并且在突发公共卫生事件等异常情况下也难以提供可靠的预测。为了推进现有研究，我们提出了HealthMamba，一个用于实现准确且可靠的医疗机构就诊预测的不确定性感知时空框架。HealthMamba包含三个关键组件： 统一时空上下文编码器，用于融合异构的静态与动态信息； 一种名为GraphMamba的新型图状态空间模型，用于分层时空建模； 一个全面的不确定性（估计模块，摘要在此处截断）……

    arXiv:2602.05286v4 Announce Type: replace  Abstract: Healthcare facility visit prediction is essential for optimizing healthcare resource allocation and informing public health policy. Despite advanced machine learning methods being employed for better prediction performance, existing works usually formulate this task as a time-series forecasting problem without considering the intrinsic spatial dependencies of different types of healthcare facilities, and they also fail to provide reliable predictions under abnormal situations such as public emergencies. To advance existing research, we propose HealthMamba, an uncertainty-aware spatiotemporal framework for accurate and reliable healthcare facility visit prediction. HealthMamba comprises three key components: (i) a Unified Spatiotemporal Context Encoder that fuses heterogeneous static and dynamic information, (ii) a novel Graph State Space Model called GraphMamba for hierarchical spatiotemporal modeling, and (iii) a comprehensive uncer
    
[^345]: 将ROC分析扩展至不确定性感知的风险预测：基于区间的AUC（iAUC）

    Extending ROC Analysis to Uncertainty-Aware Risk Prediction with an Interval-Based AUC (iAUC)

    [https://arxiv.org/abs/2602.04775](https://arxiv.org/abs/2602.04775)

    提出不确定性感知的ROC框架与基于区间的AUC（iAUC）指标，用于评估区间值风险预测的判别性能，并将正负样本排序分解为自信正确、模糊和自信错误三个区域。

    

    在高风险预测中，区间值预测为表示预测不确定性提供了一种自然的方式。然而，标准的评估工具（如受试者工作特征曲线（ROC）和曲线下面积（AUC））是为点值预测设计的，无法反映不确定性如何影响判别性能。为弥补这一空白，我们开发了一个不确定性感知的ROC框架以及相应的基于区间的AUC（iAUC）指标，用于评估区间值风险预测的性能。该框架构建了两条ROC风格的曲线及其对应面积 $\mathrm{AUC}_L$ 和 $\mathrm{AUC}_U$，并证明这些量可将正负样本的排序诱导为三个区域的分解：自信正确、模糊以及自信错误的排序。在满足有效类条件覆盖率的前提下，$\mathrm{AUC}_L$ 和 $\mathrm{AUC}_U$ 与成对错误覆盖率……

    arXiv:2602.04775v2 Announce Type: replace  Abstract: In high-stakes risk prediction, interval-valued predictions provide a natural way to represent predictive uncertainty. However, standard evaluation tools such as the receiver operating characteristic (ROC) curve and the area under the curve (AUC) are designed for point-valued predictions and do not capture how uncertainty affects discrimination performance. To address this gap, we develop an uncertainty-aware ROC framework and corresponding interval-based AUC (iAUC) metrics for evaluating the performance of interval-valued risk predictions. The framework constructs two ROC-style curves with associated areas $\mathrm{AUC}_L$ and $\mathrm{AUC}_U$, and shows that these quantities induce a three-region decomposition of positive-negative rankings into confidently correct, ambiguous, and confidently incorrect orderings. Under valid class-conditional coverage, $\mathrm{AUC}_L$ and $\mathrm{AUC}_U$, together with the pairwise miscoverage rat
    
[^346]: 低维特征子空间中的机器遗忘

    Machine Unlearning in Low-Dimensional Feature Subspace

    [https://arxiv.org/abs/2601.22456](https://arxiv.org/abs/2601.22456)

    提出LOFT方法，通过在预训练模型的低维特征子空间中优化一个可插入的小型主投影矩阵来分离保留数据与遗忘数据的信息，从而以极低开销实现有效且高效的机器遗忘。

    

    机器遗忘旨在从预训练模型中移除特定数据的影响，同时保持模型在剩余数据上的性能。本工作从低维特征子空间的视角为机器遗忘提出了一个新颖的研究角度，揭示了在该子空间中分离保留数据与遗忘数据的可能性。这种可分离性催生了我们提出的LOFT方法，该方法通过主投影在预训练模型的低维特征子空间中执行遗忘，这些主投影经过优化，能够最大程度地捕获保留数据的信息，同时削弱遗忘数据的信息。在遗忘过程中，LOFT仅需优化一个小型投影矩阵，该矩阵可灵活地插入预训练模型，且只需从预训练模型中一次性获取特征。LOFT证明了即使无需（重新）……（原文摘要截断），也可以在低维特征子空间中实现有效且高效的机器遗忘。

    arXiv:2601.22456v3 Announce Type: replace  Abstract: Machine Unlearning (MU) aims at removing the influence of specific data from a pretrained model while preserving performance on the remaining data. In this work, a novel perspective for MU is presented upon low-dimensional feature subspaces, which gives rise to the potentials of separating the remaining and forgetting data herein. This separability motivates our LOFT, a method that proceeds unlearning in a LOw-dimensional FeaTure subspace from the pretrained model through principal projections, which are optimized to maximally capture the information of the remaining data and meanwhile diminish that of the forgetting data. During unlearning, LOFT simply optimizes a small-size projection matrix that can be flexibly inserted into the pretrained model, requiring just one-shot feature fetching from the pretrained model. LOFT demonstrates that effective and efficient MU can be achieved in low-dimensional feature subspaces even without rep
    
[^347]: 从模糊到精确：基于有理数运算实现无限深度推理的Halo架构

    From Fuzzy to Exact: The Halo Architecture for Infinite-Depth Reasoning via Rational Arithmetic

    [https://arxiv.org/abs/2601.18702](https://arxiv.org/abs/2601.18702)

    提出基于有理数域和精确推理单元的Halo架构，通过微环（丢番图近似约束内存）与宏环（符号对齐保证逻辑一致性）的双环拓扑，以精确运算取代模糊浮点运算，摆脱数值稳定技巧并实现无限深度推理。

    

    arXiv:2601.18702v5 公告类型：替换 摘要：当前大语言模型（LLM）主流的扩展范式建立在“模糊”的浮点运算基础之上。为了缓解这种近似基础的固有不稳定性，现代架构构建了复杂的结构与数值启发式机制——复数残差、Pre-RMSNorm、注意力缩放和梯度裁剪——这些机制仅仅为了防止数值崩溃就消耗了大量计算资源。我们提出向“精确”范式的转变。我们引入了Halo架构，该架构植根于有理数域（Q），并由定制的精确推理单元（EIU）驱动。为了解决有理数运算中位宽指数级增长的问题，Halo采用了双环拓扑，统一了两种互补的控制机制：（1）微环（连续统维护），通过丢番图近似严格约束内存复杂度；（2）宏环（符号对齐），强制逻辑一致性……

    arXiv:2601.18702v5 Announce Type: replace  Abstract: The prevailing scaling paradigm of Large Language Models (LLMs) rests on a substrate of "Fuzzy" floating-point arithmetic. To mitigate the inherent instability of this approximate foundation, modern architectures have erected a complex scaffolding of structural and numerical heuristics--Complex Residuals, Pre-RMSNorm, Attention Scaling, and Gradient Clipping--consuming significant compute solely to prevent numerical collapse.   We propose a paradigm shift to the "Exact". We introduce the Halo Architecture, grounded in the Rational Field (Q) and powered by a custom Exact Inference Unit (EIU). To resolve the exponential bit-width growth of rational arithmetic, Halo employs a Dual-Ring Topology that unifies two complementary control mechanisms: (1) The Micro-Ring (Continuum Maintenance), which strictly bounds memory complexity via Diophantine Approximation; and (2) The Macro-Ring (Symbolic Alignment), which enforces logical consistency 
    
[^348]: 集中式学习是否有助于具身智能体协调，以及表格化研究何时能够回答这一问题？

    Does Centralized Learning Help Embodied Agents Coordinate, and When Can a Tabular Study Tell?

    [https://arxiv.org/abs/2601.17454](https://arxiv.org/abs/2601.17454)

    该论文在捕食者-猎物环境中，于多种捕食者与猎物配对及不同网格规模下，系统比较独立Q学习与中心化Q学习，并通过预先设定的直接协调性度量来检验集中式学习何时真正有助于具身智能体的协调。

    

    表格化研究是比较独立式与集中式多智能体学习器的常用方法，因为它避免了函数逼近。一个自然的预期是，学习联合动作的集中式学习能够帮助智能体实现协调，尤其是当其身体条件使单独成功变得困难时。以往的工作只部分地检验了这一点：或者仅针对一种学习方法度量协调性，或者在与智能体身体固定的情况下比较不同方法，或者改变速度但只报告团队的整体表现。我们在一个捕食者-猎物环境中对此进行测试，其中队友之间合作捕猎，而两个团队相互竞争。我们在所有四种捕食者与猎物的配对中，在速度和耐力受限的条件下，比较独立Q学习（IQL）与中心化Q学习（CQL），在5x5和8x8网格上各训练200,000个回合，并验证智能体在两种情况下均能学到策略。我们直接度量协调性，并对主要测量指标预先确定了预测。

    arXiv:2601.17454v3 Announce Type: replace-cross  Abstract: Tabular studies are a common way to compare independent and centralized multi-agent learners, because they avoid function approximation. A natural expectation is that centralized learning, which learns joint actions, helps agents coordinate, especially when their bodies make solo success hard. Earlier work tests this only in parts: it measures coordination for one learning method, compares methods with the agents' bodies fixed, or varies speed but reports only how well the team did. We test it in a predator-prey environment, where teammates cooperatively hunt together while two teams compete. We compare independent Q-learning (IQL) with central Q-learning (CQL) in all four predator and prey pairings, under limits on speed and stamina, on a 5x5 and an 8x8 grid trained for 200,000 episodes each, and check that the agents learn on both. We measure coordination directly, with predictions for the main measurement fixed in advance. I
    
[^349]: RubRIX：基于评分量表的照护者-AI交互风险缓解框架

    RubRIX: Rubric-Driven Risk Mitigation in Caregiver-AI Interactions

    [https://arxiv.org/abs/2601.13235](https://arxiv.org/abs/2601.13235)

    本文提出了RubRIX——一个基于关怀伦理理论、经临床医生验证的评分量表框架，将LLM在照护者交互场景中的回应风险操作化为五个维度，并在超过2万条真实照护者查询上验证，量表引导的改进可将风险成分降低45-98%。

    

    寻求AI介导支持的照护者往往表达出复杂的需求——包括信息获取、情感确认以及痛苦信号——这需要对回应的安全性与恰当性进行细致评估。现有的AI评估框架主要聚焦于一般性风险（如毒性、幻觉、政策违规等），可能无法充分捕捉大语言模型回应在照护情境中的细微风险。我们提出了RubRIX（基于评分量表的风险指数），这是一个理论驱动、经临床医生验证的框架，用于评估LLM照护回应中的风险。RubRIX以“关怀伦理要素”为理论基础，将五个经实证得出的风险维度进行操作化：忽视、偏见与污名化、信息不准确、不加批判的肯定，以及认知傲慢。我们在来自Reddit和ALZConnected的超过20,000条照护者查询上评估了六个最先进的LLM。经过评分量表引导的改进，风险成分持续降低了45-98%……

    arXiv:2601.13235v2 Announce Type: replace-cross  Abstract: Caregivers seeking AI-mediated support express complex needs---information-seeking, emotional validation, and distress cues---that warrant careful evaluation of response safety and appropriateness. Existing AI evaluation frameworks, primarily focused on general risks (toxicity, hallucinations, policy violations, etc), may not adequately capture the nuanced risks of LLM-responses in caregiving-contexts. We introduce RubRIX (Rubric-based Risk Index), a theory-driven, clinician-validated framework for evaluating risks in LLM caregiving responses. Grounded in the Elements of an Ethic of Care, RubRIX operationalizes five empirically-derived risk dimensions: Inattention, Bias & Stigma, Information Inaccuracy, Uncritical Affirmation, and Epistemic Arrogance. We evaluate six state-of-the-art LLMs on over 20,000 caregiver queries from Reddit and ALZConnected. Rubric-guided refinement consistently reduced risk-components by 45-98% after 
    
[^350]: ChemBART：用于综合合成规划的化学反应语言模型预训练

    ChemBART: Pre-training a Reaction Language Model for Comprehensive Synthesis Planning

    [https://arxiv.org/abs/2601.02915](https://arxiv.org/abs/2601.02915)

    ChemBART是一种以反应为中心进行预训练的BART语言模型，通过在整体反应表达式上进行掩码填充预训练，能够统一解决前体/试剂生成、产率回归、分子性质分类等多种化学任务，并可与蒙特卡洛树搜索结合实现多步合成路线设计。

    

    大语言模型（LLMs）的最新进展已在多个领域展现出变革性潜力。然而，在计算机辅助合成规划（CASP）领域，当前的LLM方法通常局限于单分子表示，这限制了其在孤立的单任务场景（如前体预测）中的应用。为填补这一空白，我们提出了ChemBART，一个基于BART架构、采用以反应为中心的预训练策略的语言模型。通过在整体反应表达式（反应物 > 试剂 > 产物）上进行掩码填充预训练任务，ChemBART能够有效解决多种化学问题，包括前体/试剂生成、温度-产率回归以及分子性质分类。基于这些能力，我们展示了将ChemBART与蒙特卡洛树搜索（MCTS）相集成以进行多步合成路线设计，其中该模型用于优化策略和……

    arXiv:2601.02915v2 Announce Type: replace  Abstract: Recent advances in large language models (LLMs) have demonstrated transformative potential across diverse fields. However, in computer-aided synthesis planning (CASP), current LLM methodologies are often restricted to single-molecule representations, limiting their application to isolated, single-task scenarios such as precursor prediction. Addressing this gap, we present ChemBART, a BART-based language model that adopts a reaction-centric pre-training strategy. By leveraging a mask-filling pre-training task on holistic reaction expressions (Reactant > Reagent > Product), ChemBART effectively solves a variety of chemical problems, including precursor/reagent generation, temperature-yield regression, and molecular property classification. Building on these capabilities, we demonstrate the integration of ChemBART with Monte Carlo tree search (MCTS) for multi-step synthesis route design, where the model serves to optimize the policy and
    
[^351]: 结构即监督：面向放射学的多视角掩码自编码器

    Structure is Supervision: Multiview Masked Autoencoders for Radiology

    [https://arxiv.org/abs/2511.22294](https://arxiv.org/abs/2511.22294)

    提出多视角掩码自编码器 MVMAE 及其结合放射学报告的扩展版 MVMAE-V2T，利用放射影像天然的多视角结构作为自监督信号，在三个大规模数据集的疾病分类任务上持续超越有监督和视觉-语言基线方法。

    

    构建稳健的医学机器学习系统需要能够利用临床数据中内在结构的预训练策略。我们提出了多视角掩码自编码器（MVMAE），这是一个自监督框架，它利用放射学检查中自然存在的多视角组织结构，来学习视角不变且与疾病相关的表征。MVMAE 将掩码图像重建与跨视角对齐相结合，把不同投影视角之间的临床冗余转化为强大的自监督信号。我们进一步将该方法扩展为 MVMAE-V2T，它将放射学报告作为辅助的基于文本的学习信号，以增强语义基础，同时保持完全基于视觉的推理。在三个大规模公开数据集（MIMIC-CXR、CheXpert 和 PadChest）上的下游疾病分类任务评估中，MVMAE 持续优于有监督基线和视觉-语言基线。此外，MVM…（摘要内容不完整，在此截断）

    arXiv:2511.22294v5 Announce Type: replace-cross  Abstract: Building robust medical machine learning systems requires pretraining strategies that exploit the intrinsic structure present in clinical data. We introduce Multiview Masked Autoencoder (MVMAE), a self-supervised framework that leverages the natural multi-view organization of radiology studies to learn view-invariant and disease-relevant representations. MVMAE combines masked image reconstruction with cross-view alignment, transforming clinical redundancy across projections into a powerful self-supervisory signal. We further extend this approach with MVMAE-V2T, which incorporates radiology reports as an auxiliary text-based learning signal to enhance semantic grounding while preserving fully vision-based inference. Evaluated on a downstream disease classification task on three large-scale public datasets, MIMIC-CXR, CheXpert, and PadChest, MVMAE consistently outperforms supervised and vision-language baselines. Furthermore, MVM
    
[^352]: 数据序列上寄存器自动机的合成与验证（技术报告）

    Synthesis and Verification of Register Automata over Data Sequences (Technical Report)

    [https://arxiv.org/abs/2511.19100](https://arxiv.org/abs/2511.19100)

    该论文提出了针对有序数据序列上确定性寄存器自动机的局部搜索合成方法与精确鲁棒性检查器，后者对固定寄存器数量可在多项式时间内运行，并通过将鲁棒性归约为可覆盖性和最短路径问题实现高效验证。

    

    数据序列，例如时间序列、轨迹和传感器流，是定义在大型或无限数据域上的词。它们的性质往往依赖于不同位置的值之间的比较，而有限自动机无法直接表达这种比较。我们研究了有序数据序列上确定性寄存器自动机（DRA）的合成与验证问题。DRA 将有限控制与用于记忆数据值的寄存器相结合，为量化序列语言提供了紧凑的符号化描述。我们的技术核心是针对 DRA 的局部搜索合成过程以及一个精确的鲁棒性检查器。我们将字母表理论固定为有序有理数 $(\Q;<,=)$，并使用有理数守卫常量与非负仿射累加器代价。对于固定数量的 DRA 寄存器，该检查器可在多项式时间内运行；但其对寄存器数量的依赖是指数级的。该方法将鲁棒性验证归约为可覆盖性问题，进而归约为最短路径问题。

    arXiv:2511.19100v2 Announce Type: replace-cross  Abstract: Data sequences, such as time series, traces, and sensor streams, are words over large or infinite data domains. Their properties often depend on comparisons between values at different positions, which finite automata cannot express directly. We study synthesis and verification for Deterministic Register Automata (DRAs) over ordered data sequences. DRAs combine finite control with registers for remembered data values, providing compact symbolic descriptions of quantitative sequence languages. Our technical core is a local-search synthesis procedure and an exact robustness checker for DRAs. We fix the alphabet theory to ordered rationals $(\Q;<,=)$ with rational guard constants and use non-negative affine accumulator costs. For a fixed number of DRA registers, the checker runs in polynomial time; its dependence on that number is exponential. It reduces robustness to coverability and then to a shortest-path problem. Since the sea
    
[^353]: Operon：通过命名维度增量构建不规则数据

    Operon: Incremental Construction of Ragged Data via Named Dimensions

    [https://arxiv.org/abs/2511.16080](https://arxiv.org/abs/2511.16080)

    Operon是一个基于Rust的工作流引擎，通过命名维度与显式依赖关系的形式化方法，结合静态验证的领域特定语言和增量式的动态任务调度，实现了对不规则数据形状与依赖关系的原生追踪与管理。

    

    现代数据处理工作流经常遇到不规则数据：即包含可变长度元素的集合，这类数据自然产生于自然语言处理、科学测量和自主AI智能体等领域。现有的工作流引擎缺乏对不规则数据固有的形状和依赖关系的原生支持，迫使用户手动管理复杂的索引和依赖记录。我们提出了Operon，一个基于Rust的工作流引擎，它通过一种新颖的命名维度形式化方法及显式依赖关系来解决这些挑战。Operon提供了一种领域特定语言，用户可以在其中声明带有维度标注的管道，这些标注会经过静态验证以确保正确性；同时运行时系统在执行过程中随着数据形状的逐步发现而动态调度任务。我们形式化了用于推理部分形状的数学基础，并证明了Operon的……（原文摘要在此处截断）

    arXiv:2511.16080v2 Announce Type: replace-cross  Abstract: Modern data processing workflows frequently encounter ragged data: collections with variable-length elements that arise naturally in domains like natural language processing, scientific measurements, and autonomous AI agents. Existing workflow engines lack native support for tracking the shapes and dependencies inherent to ragged data, forcing users to manage complex indexing and dependency bookkeeping manually. We present Operon, a Rust-based workflow engine that addresses these challenges through a novel formalism of named dimensions with explicit dependency relations. Operon provides a domain-specific language where users declare pipelines with dimension annotations that are statically verified for correctness, while the runtime system dynamically schedules tasks as data shapes are incrementally discovered during execution. We formalize the mathematical foundation for reasoning about partial shapes and prove that Operon's in
    
[^354]: 用于可解释多元时间序列异常检测的结构化时间因果性

    Structured Temporal Causality for Interpretable Multivariate Time Series Anomaly Detection

    [https://arxiv.org/abs/2510.16511](https://arxiv.org/abs/2510.16511)

    提出了OracleAD，一个简单可解释的无监督多元时间序列异常检测框架，它将每个变量的时间动态编码为因果嵌入，通过自注意力机制建模动态空间关系，并与代表正常状态的稳定潜在结构对齐以检测异常。

    

    现实世界中的多元时间序列异常是罕见的，且通常缺乏标签。此外，主流方法依赖于针对基准数据集调优的日益复杂的架构，只能检测到异常片段的部分内容，并夸大了检测性能。在本文中，我们提出了OracleAD，一个简单且可解释的无监督多元时间序列异常检测框架。OracleAD将每个变量的过去序列编码为单一的因果嵌入，用于联合预测当前时间点并重建输入窗口，从而有效建模时间动态。随后，这些嵌入经过自注意力机制，被投影到共享潜在空间中以捕获变量间的空间关系。这些关系并非静态，而是通过从每个变量的时间动态中涌现的特性来建模的。投影后的嵌入与代表正常状态关系的稳定潜在结构对齐。

    arXiv:2510.16511v2 Announce Type: replace  Abstract: Real-world multivariate time series anomalies are rare and often unlabeled. Additionally, prevailing methods rely on increasingly complex architectures tuned to benchmarks, detecting only fragments of anomalous segments and overstating performance. In this paper, we introduce OracleAD, a simple and interpretable unsupervised framework for multivariate time series anomaly detection. OracleAD encodes each variable's past sequence into a single causal embedding to jointly predict the present time point and reconstruct the input window, effectively modeling temporal dynamics. These embeddings then undergo a self-attention mechanism to project them into a shared latent space and capture spatial relationships. These relationships are not static, since they are modeled by a property that emerges from each variable's temporal dynamics. The projected embeddings are aligned to a Stable Latent Structure (SLS) representing normal-state relations
    
[^355]: 记忆长尾数据可以通过组合帮助泛化

    Memorizing Long-tail Data Can Help Generalization Through Composition

    [https://arxiv.org/abs/2510.16322](https://arxiv.org/abs/2510.16322)

    记忆长尾样本与特征组合能力相结合，可以使模型对训练中从未出现过的长尾特征组合的稀有测试样本做出正确预测，且这种组合能力依赖于模型架构。

    

    深度学习促使研究人员重新思考记忆与泛化之间的关系。在许多情况下，由于隐式正则化的存在，记忆并不会损害泛化能力，并且通过记忆长尾样本还可能有所助益。在本文中，我们考虑了记忆与简单组合之间的协同作用——即在长尾特征的组合上做出正确预测的能力。从理论上讲，我们证明了在线性设置中，记忆与组合相结合可以帮助模型在需要长尾特征组合的稀有测试样本上做出正确预测，即使这些组合在训练数据中从未出现过。在简单数据上针对神经网络架构的实验表明，这一理论见解可以推广到线性设置之外，我们还进一步观察到，模型的组合能力取决于其架构。

    arXiv:2510.16322v2 Announce Type: replace  Abstract: Deep learning has led researchers to rethink the relationship between memorization and generalization. In many settings, memorization does not hurt generalization due to implicit regularization and may help by memorizing long-tailed examples. In this paper, we consider the synergy between memorization and simple composition -- the ability to make correct prediction on a combination of long-tailed features. Theoretically, we show that for a linear setting, memorization together with composition can help the model make correct predictions on rare test examples that require a combination of long-tailed features, even if such combinations were never observed in the training data. Experiments on neural network architecture on simple data show that the theoretical insight extends beyond the linear setting, and we further observe that the composition capability of the model depends on its architecture.
    
[^356]: 雕塑潜空间：基于WAE-MMD与可编程先验的解耦表示学习

    Sculpting the Latent Space: Disentanglement with WAE-MMD and Programmable Priors

    [https://arxiv.org/abs/2510.11953](https://arxiv.org/abs/2510.11953)

    该论文通过非参数化MMD直接塑形聚合后验，提出支持可编程先验的WAE-MMD方法，并引入无监督纠缠度量LPS，使潜空间纠缠度降低70%，实现无监督解耦表示学习的最先进水平。

    

    arXiv:2510.11953v2 公告类型：replace。摘要：学习解耦表示——即让语义特征在潜空间中被独立捕捉——目前主要由变分自编码器（VAE）和对抗训练主导。在本文中，我们提供了直接的定量与视觉证据，表明这些方法始终无法在聚合后验上强制实现目标因子化表示，从而导致表示纠缠。我们使用潜空间可预测性分数（Latent Predictability Score, LPS）来量化这种失败，LPS是一种无监督的相互可预测性度量，从经验上为潜空间中的纠缠程度提供下界。为解决这些局限性，我们利用非参数化的最大均值差异（MMD）直接对聚合后验进行塑形，从而实现可编程先验。我们证明了：（1）在无监督统计独立性方面达到最先进水平，与领先的基线方法相比，检测到的纠缠减少了70%；（2）可以灵活地选择性地注入……（摘要原文在此处截断）

    arXiv:2510.11953v2 Announce Type: replace  Abstract: Learning disentangled representations, where semantic features are independently captured in a latent space, is dominated by Variational Auto Encoders (VAE) and Adversarial Training. In this paper, we provide direct quantitative and visual evidence that these methods consistently fail to enforce the target factorized representation on the aggregate posterior, leading to entangled representations. We quantify this failure using the Latent Predictability Score (LPS), an unsupervised metric of mutual predictability that empirically lower bounds the level of entanglement in a latent space. To address these limitations, we leverage the non-parametric Maximum Mean Discrepancy (MMD) to directly sculpt the aggregate posterior enabling programmable priors. We demonstrate (1) state-of-the-art unsupervised statistical independence, with a 70% reduction in detected entanglement compared to leading baselines; (2) the flexibility to optionally inj
    
[^357]: 超维探针：基于向量符号架构解码大语言模型表示

    Hyperdimensional Probe: Decoding LLM Representations via Vector Symbolic Architectures

    [https://arxiv.org/abs/2509.25045](https://arxiv.org/abs/2509.25045)

    提出超维探针——一种结合向量符号架构与神经探测的混合式有监督探针，统一了有监督探针、稀疏自编码器和logit分析等可解释性方法，从而更全面地解码大语言模型的内部表示。

    

    尽管大型语言模型（LLM）具有强大的能力，但其内部仍然是不透明的，人们对其内部表示的理解十分有限。目前的可解释性方法要么专注于面向输入的特征提取，例如有监督探针和稀疏自编码器（SAE），要么专注于输出分布检查，例如面向logit的方法。然而，要全面理解LLM的向量空间，需要将这两种视角整合起来，而现有方法由于潜在特征定义上的限制，难以做到这一点。我们提出了超维探针，这是一种将符号表示与神经探测相结合的混合式有监督探针。通过利用向量符号架构（VSA）和超向量代数，它统一了先前的方法：有监督探针的自顶向下可解释性、SAE由稀疏性驱动的代理空间，以及面向输出的logit研究。通过结合传统的有监督学习范式……

    arXiv:2509.25045v4 Announce Type: replace  Abstract: Despite their capabilities, Large Language Models (LLMs) remain opaque with limited understanding of their internal representations. Current interpretability methods either focus on input-oriented feature extraction, such as supervised probes and Sparse Autoencoders (SAEs), or on output distribution inspection, such as logit-oriented approaches. A full understanding of LLM vector spaces, however, requires integrating both perspectives, something existing approaches struggle with due to constraints on latent feature definitions. We introduce the Hyperdimensional Probe, a hybrid supervised probe that combines symbolic representations with neural probing. Leveraging Vector Symbolic Architectures (VSAs) and hypervector algebra, it unifies prior methods: the top-down interpretability of supervised probes, SAE's sparsity-driven proxy space, and output-oriented logit investigation. By combining the supervised learning paradigm of traditiona
    
[^358]: BrainPro：面向大型脑电图模型的过程条件化预训练与拓扑检索方法

    BrainPro: Process-conditioned Pretraining with Topographic Retrieval for Large EEG Models

    [https://arxiv.org/abs/2509.22050](https://arxiv.org/abs/2509.22050)

    BrainPro提出了一种过程条件化的自监督预训练框架，通过拓扑检索将不同数据集的电极导联配置映射到通用通道-区域模板，并同时学习共享表征与过程关联表征，从而解决EEG基础模型中心理过程线索利用不足和导联配置异构两大挑战。

    

    脑电图（EEG）基础模型通常在异构语料库上进行预训练，但存在两个尚未被充分探索的挑战：（1）实验范式提供了心理过程线索，但在掩码预训练中很少被利用；（2）不同数据集的电极导联配置各不相同，导致空间表征依赖于特定的导联配置。现有方法通常只学习单一通用表征，并隐式地对空间结构进行建模，这限制了它们在导联配置异构性条件下利用过程相关信息的能力。我们提出了BrainPro，这是一个过程条件化的自监督预训练框架，它将拓扑检索与共享表征及过程关联表征学习相结合。BrainPro将数据集特定的导联配置映射到通用的通道-区域模板上，并检索通道级和区域级空间滤波器，从而形成一个拓扑对齐的空间基底。在此基底之上，BrainPro学习一个共享编码器以捕捉通用的EEG结构，……

    arXiv:2509.22050v3 Announce Type: replace  Abstract: EEG foundation models are pretrained on heterogeneous corpora, but two challenges remain underexplored: (1) experimental protocols provide mental-process cues that are rarely used in masked pretraining, and (2) electrode montages vary across datasets, making spatial representations montage-dependent. Existing methods typically learn a single general-purpose representation and model spatial structure implicitly, limiting their ability to exploit process-related information under montage heterogeneity. We introduce BrainPro, a process-conditioned self-supervised pretraining framework that couples topographic retrieval with shared and process-associated representation learning. BrainPro maps dataset-specific montages to a universal channel-region template and retrieves channel- and region-level spatial filters to form a topographically aligned spatial basis. Over this basis, BrainPro learns a shared encoder for general EEG structure and
    
[^359]: TopoMap：一种基于特征的测试输入空间拓扑区域判别器

    TopoMap: A Feature-based Discriminator of the Topographical Regions in the Test Input Space

    [https://arxiv.org/abs/2509.03242](https://arxiv.org/abs/2509.03242)

    TopoMap是一种模型无关的方法，通过基于输入特征对测试输入空间进行拓扑划分，帮助用户识别代表性不足、失败率高或模型更新后预测发生变化的区域，从而指导数据生成和标注工作的优先级排序。

    

    测试基于深度学习的系统是一项尚未解决的挑战。虽然找到导致深度学习模型出现错误行为的输入相对容易，但如何根据使这些输入对被测深度学习模型产生问题的特征来对它们进行分组，在很大程度上仍未被探索。在这项工作中，我们提出了TopoMap，这是一种模型无关的方法，仅依赖刻画输入的特征来构建输入空间的一种划分，其中每个区域代表一组特定的、可区分的输入特征。这样的划分为用户提供了一种可能性，即识别并分析那些代表性不足、失败率较高，或在模型更新后表现出预测变化的区域。随后，这些被识别的感兴趣区域可用于额外的数据生成，或帮助对已生成的、需要标注工作的输入进行优先级排序。为了根据输入所共有的特定特征对输入进行判别，我们首先应用降维……

    arXiv:2509.03242v3 Announce Type: replace  Abstract: Testing Deep Learning (DL)-based systems is an open challenge. Although it is relatively easy to find inputs that cause a DL model to misbehave, the grouping of such inputs by features that make them problematic for the DL model under test is largely unexplored. In this work, we propose TopoMap, a model-agnostic approach that relies solely on features that characterise inputs to construct a partitioning of the input space where each region represents a specific set of distinguishable input features. Such a partitioning gives users a possibility to identify and analyse areas that are under-represented, show higher failure rates, or exhibit prediction changes after a model update. The identified regions of interest can then be used for additional data generation or can help prioritise already generated inputs that require labelling efforts. To discriminate inputs according to the specific features they share, we first apply dimensional
    
[^360]: 基于堆栈跟踪的崩溃去重与Transformer适配

    Stack Trace-Based Crash Deduplication with Transformer Adaptation

    [https://arxiv.org/abs/2508.19449](https://arxiv.org/abs/2508.19449)

    本文提出dedupT，一种基于Transformer的崩溃去重方法，通过整体建模堆栈跟踪并适配预训练语言模型，显著提升了重复崩溃排序和唯一崩溃检测的准确性，减少了人工分类负担。

    

    arXiv:2508.19449v2 公告类型：替换交叉 摘要：自动化崩溃报告系统会生成大量重复报告，使问题跟踪系统不堪重负，并增加开发人员的工作负担。传统的基于堆栈跟踪的去重方法——依赖字符串相似性、基于规则的启发式方法或深度学习（DL）模型——通常无法捕捉堆栈跟踪中的上下文和结构关系。我们提出了dedupT，一种基于Transformer的方法，它将堆栈跟踪整体建模，而非视为孤立的帧。dedupT首先将预训练语言模型（PLM）适配到堆栈跟踪上，然后利用其嵌入来训练一个全连接网络（FCN），以有效排序重复崩溃。在真实世界数据集上的大量实验表明，dedupT在重复排序和唯一崩溃检测方面均优于现有的深度学习和传统方法（例如序列比对和信息检索技术），显著减少了人工分类工作。

    arXiv:2508.19449v2 Announce Type: replace-cross  Abstract: Automated crash reporting systems generate large volumes of duplicate reports, overwhelming issue-tracking systems and increasing developer workload. Traditional stack trace-based deduplication methods---relying on string similarity, rule-based heuristics, or deep learning (DL) models---often fail to capture the contextual and structural relationships within stack traces. We propose dedupT, a transformer-based approach that models stack traces holistically rather than as isolated frames. dedupT first adapts a pretrained language model (PLM) to stack traces, then uses its embeddings to train a fully-connected network (FCN) to rank duplicate crashes effectively. Extensive experiments on real-world datasets show that dedupT outperforms existing DL and traditional methods (e.g., sequence alignment and information retrieval techniques) in both duplicate ranking and unique crash detection, significantly reducing manual triage effort.
    
[^361]: 合成数据能否重现流行病学中的真实世界研究发现？一项基于对抗性随机森林的复制研究

    Can synthetic data reproduce real-world findings in epidemiology? A replication study using adversarial random forests

    [https://arxiv.org/abs/2508.14936](https://arxiv.org/abs/2508.14936)

    该研究提出对抗性随机森林（ARF）作为合成表格型流行病学数据的高效便捷方法，并通过复制六项流行病学研究的统计分析，验证了合成数据能在保护隐私的同时可靠重现真实世界的研究发现。

    

    由于数据访问受限和隐私问题，合成数据在应对流行病学中的实际挑战方面具有巨大潜力。然而，目前许多方法存在质量有限、计算需求高以及对非专业人员而言过于复杂等问题。此外，常用的合成数据评估策略往往无法直接反映统计效用，也不能充分衡量隐私风险。在此背景下，一个关键且尚未被充分探索的问题是：合成数据能否在保护隐私的同时，可靠地重现流行病学研究的关键发现。我们提出对抗性随机森林（ARF）作为一种高效且便捷的方法，用于合成表格型流行病学数据。为评估其性能，我们复制了六篇流行病学出版物的统计分析，涵盖血压、人体测量、心肌梗死、加速度计测量、孤独感和糖尿病，数据来自德国的

    arXiv:2508.14936v4 Announce Type: replace-cross  Abstract: Synthetic data hold substantial potential to address practical challenges in epidemiology due to restricted data access and privacy concerns. However, many current methods suffer from limited quality, high computational demands, and complexity for non-experts. Furthermore, common evaluation strategies for synthetic data often fail to directly reflect statistical utility and measure privacy risks sufficiently. Against this background, a critical underexplored question is whether synthetic data can reliably reproduce key findings from epidemiological research while preserving privacy. We propose adversarial random forests (ARF) as an efficient and convenient method for synthesizing tabular epidemiological data. To evaluate its performance, we replicated statistical analyses from six epidemiological publications covering blood pressure, anthropometry, myocardial infarction, accelerometry, loneliness, and diabetes, from the German 
    
[^362]: UniPrompt-CL：面向医学图像分类的高效单次推理持续学习

    UniPrompt-CL: Efficient Single-Inference Continual Learning for Medical Image Classification

    [https://arxiv.org/abs/2508.10954](https://arxiv.org/abs/2508.10954)

    UniPrompt-CL是一种无需回放的域增量学习框架，通过在Transformer各层共享统一提示池并以单一主干网络、单次推理的方式实现医学图像分类的高效持续学习，在糖尿病视网膜病变和皮肤癌任务上以更低的训练成本取得了更高准确率并显著减少遗忘。

    

    医学AI系统必须在数据分布随医院、成像设备和患者群体变化时保持可靠性，然而隐私限制通常阻止存储或重新访问过往患者数据。我们提出UniPrompt-CL，这是一个无需回放（rehearsal-free）的域增量学习框架，用于医学图像分类，仅使用单一主干网络和单次推理。与依赖多个主干网络、重复的ViT推理或独立的逐层提示池不同，UniPrompt-CL在Transformer各层之间共享统一的提示池，并仅为每个新领域添加少量提示来对其进行扩展。这减少了跨层冗余，同时实现了高效适配。在糖尿病视网膜病变任务上，与最强的主干匹配基线相比，UniPrompt-CL将平均准确率提高了2.0个百分点，并显著减少了遗忘。在皮肤癌任务上，它在DINOv2-B配置中以最低的训练成本实现了具有竞争力的准确率。

    arXiv:2508.10954v3 Announce Type: replace  Abstract: Medical AI systems must remain reliable as data distributions shift across hospitals, imaging devices, and patient populations, yet privacy constraints often prevent storing or revisiting past patient data. We present UniPrompt-CL, a rehearsal-free domain-incremental learning framework for medical image classification using a single backbone and a single inference. Rather than relying on multiple backbones, repeated ViT inferences, or independent layer-wise prompt pools, UniPrompt-CL shares a unified prompt pool across transformer layers and expands it with only a few prompts per new domain. This reduces cross-layer redundancy while enabling efficient adaptation. On diabetic retinopathy, UniPrompt-CL improves average accuracy by 2.0 percentage points and substantially reduces forgetting relative to the strongest backbone-matched baseline. On skin cancer, it achieves competitive accuracy at the lowest training cost among DINOv2-B conf
    
[^363]: VRPRM：基于视觉推理的过程奖励建模

    VRPRM: Process Reward Modeling via Visual Reasoning

    [https://arxiv.org/abs/2508.03556](https://arxiv.org/abs/2508.03556)

    VRPRM提出了一种标签高效的两阶段训练策略，仅用5.5K个CoT标注和40K个强化学习样本就实现了更优的过程奖励建模，在VisualProcessBench上超越基于40万样本训练的VisualPRM-8B达7分，并在Best-of-N评估中为不同规模的多模态策略模型带来超过21分的性能提升。

    

    过程奖励模型（PRM）为中间推理步骤提供细粒度的监督，然而大多数现有模型缺乏长程推理和深度思考能力。尽管先前的工作已将思维链引入奖励模型，但CoT-PRM高昂的标注成本仍然限制了其可扩展性。为了应对这些挑战，我们提出了视觉推理过程奖励模型（VRPRM），并设计了一种标签高效的两阶段训练策略。实验结果表明，仅使用5.5K个CoT-PRM监督步骤标签和40K个非CoT PRM强化学习训练样本，VRPRM在VisualProcessBench上取得了69.18的宏F1分数，超过基于40万样本训练的非思考型PRM——VisualPRM-8B达7.00分。在Best-of-N评估中，VRPRM-RL分别为InternVL2.5-8B、26B和38B策略模型带来21.77、23.42和23.12分的绝对Overall提升，最高提升可达……

    arXiv:2508.03556v5 Announce Type: replace  Abstract: Process Reward Models (PRMs) provide fine-grained supervision over intermediate reasoning steps, yet most existing models lack long-range reasoning and deep thinking capabilities. Although prior work has introduced Chain-of-Thought (CoT) to reward models, the high annotation cost of CoT-PRM still limits its scalability. To address these challenges, we propose the Visual Reasoning Process Reward Model (VRPRM) and design a label-efficient two-stage training strategy. Experimental results demonstrate that with only 5.5K CoT-PRM supervised step labels and 40K non-CoT PRM reinforcement learning training examples, VRPRM achieves a macro F1 score of 69.18 on VisualProcessBench, outperforming VisualPRM-8B, a non-thinking PRM trained on 400K examples, by 7.00 points. In Best-of-N evaluation, VRPRM-RL brings absolute Overall improvements of 21.77, 23.42, and 23.12 points on InternVL2.5-8B, 26B, and 38B policies, respectively, with a maximum re
    
[^364]: 大语言模型量化技术的综合评估

    A Comprehensive Evaluation on Quantization Techniques for Large Language Models

    [https://arxiv.org/abs/2507.17417](https://arxiv.org/abs/2507.17417)

    本文对大语言模型的低比特训练后量化技术进行了最新、广泛且公平的综合评估与综述，并通过将量化方法解耦为多个步骤来深入分析现有方法之间的联系。

    

    对于大语言模型（LLMs），训练后量化显著降低了内存和计算开销。尽管该领域发展迅速，现有论文往往缺乏在相同基准下的实验。此外，由于量化方案通常包含多个组件，分析众多现有方法之间的联系对于深入理解这一领域至关重要。为了弥补这些空白，我们对最先进的方法进行了最新且广泛的综述，并在相同基准下对具有代表性的纯文本大语言模型进行了低比特量化的全面评估，以确保公平比较。据我们所知，这种公平且广泛的调查研究仍然至关重要，但在研究界却明显探索不足。为了更好地理解各方法之间的联系，我们首先将已发表的量化方法解耦为两个步骤：用于减少离群值的预量化变换

    arXiv:2507.17417v4 Announce Type: replace  Abstract: For large language models (LLMs), post-training quantization (PTQ) significantly reduces memory and computational overhead. Despite rapid advancements, existing papers often lack experiments on the same ground. In addition, since quantization solutions typically contain multiple components, it is important to analyze the connections among numerous existing methods to gain an in-depth understanding of this field. To bridge these gaps, we conduct an up-to-date and extensive review of the state-of-the-art methods and perform comprehensive evaluations of low-bit quantization on representative text-only LLMs on the same ground to ensure fair comparisons. To our knowledge, this fair and extensive investigation remains critically important yet significantly underexplored in the research community. To better understand the connections, we first decouple published quantization into two steps: pre-quantization transformation to reduce outliers
    
[^365]: 基于能量的强化学习迁移方法

    Energy-Based Transfer for Reinforcement Learning

    [https://arxiv.org/abs/2506.16590](https://arxiv.org/abs/2506.16590)

    提出一种基于能量分数与分布外检测的迁移学习方法，让教师策略仅在其训练分布内的状态上进行干预引导，从而在单任务和多任务强化学习中显著提升样本效率与性能。

    

    强化学习仍然存在样本效率低下的问题，尤其是当智能体必须随时间适应相关但不同的任务时，例如在持续学习场景中。通过将知识从预训练的教师策略迁移过来，以引导智能体在新但相关的任务中进行探索，可以提高效率。然而，如果新任务与教师策略的训练任务差异足够大，迁移过来的引导可能是次优的，并会使探索偏向低回报的行为。我们提出了一种基于能量的迁移学习方法，该方法利用分布外检测来选择性地发出引导，使教师策略仅在其训练分布内的状态下进行干预。我们从理论上将能量分数与在线策略强化学习下的状态访问模式联系起来。实验表明，我们的方法在单任务和多任务设置中都提升了样本效率和性能。

    arXiv:2506.16590v2 Announce Type: replace  Abstract: Reinforcement learning remains sample inefficient, particularly when agents must adapt over time to related but distinct tasks, as in continual learning settings. Efficiency can be improved by transferring knowledge from a pretrained teacher policy to guide exploration in new but related tasks. However, if the new task sufficiently differs from the teacher's training task, the transferred guidance may be sub-optimal and bias exploration toward low-reward behaviors. We propose an energy-based transfer learning method that uses out-of-distribution detection to selectively issue guidance, enabling the teacher to intervene only in states within its training distribution. We theoretically link energy scores to state-visitation patterns under on-policy reinforcement learning. Empirically, our method improves sample efficiency and performance across both single-task and multi-task settings.
    
[^366]: 从代理数据到场：基于稀疏传感器序列的全球辐射时空重建

    From Proxies to Fields: Spatiotemporal Reconstruction of Global Radiation from Sparse Sensor Sequences

    [https://arxiv.org/abs/2506.12045](https://arxiv.org/abs/2506.12045)

    本文提出时间辐射算子网络（TRON），一种时空神经算子架构，能够仅从稀疏、非均匀的代理测量序列中实时重建连续的全球标量场，无需密集网格输入和未来观测数据。

    

    从大气科学、地球物理到公共卫生和航空航天安全，从稀疏、间接的观测中准确重建潜在环境场是各科学领域面临的一项根本性挑战。现有方法通常依赖于基于物理的模拟或密集的传感器网络；然而，这些方法受到高计算成本、延迟和有限空间覆盖范围的制约。我们在此提出时间辐射算子网络（TRON），这是一种时空神经算子架构，能够仅从稀疏、非均匀的代理测量序列中推断出连续的全球标量场。与近期需要密集网格输入来预测系统状态的预测模型不同，TRON 解决了稀疏到密集的跨域场重建问题。它从稀疏的、随时间演化的传感器数据中实时重建当前的全球场，而无需获取任何未来观测数据。

    arXiv:2506.12045v2 Announce Type: replace-cross  Abstract: Accurate reconstruction of latent environmental fields from sparse, indirect observations is a fundamental challenge across scientific domains, from atmospheric science and geophysics to public health and aerospace safety. Existing approaches typically rely on physics-based simulations or dense sensor networks; however, these methods are hampered by high computational cost, latency, and limited spatial coverage. Here we introduce the \textbf{Temporal Radiation Operator Network (TRON)}, a spatiotemporal neural operator architecture that infers continuous global scalar fields solely from sequences of sparse, non-uniform proxy measurements. Unlike recent prediction models that require dense, gridded inputs to predict system states, TRON tackles sparse-to-dense, cross-domain field reconstruction. It reconstructs the current global field in real time from sparse, temporally evolving sensor data, without access to any future observat
    
[^367]: 基于大语言模型的表格数据生成中的上下文偏差传播

    In-Context Bias Propagation in LLM-Based Tabular Data Generation

    [https://arxiv.org/abs/2506.09630](https://arxiv.org/abs/2506.09630)

    本文系统研究了基于LLM的合成表格数据生成中上下文示例偏差的传播机制，发现即使轻微的上下文偏差也会导致合成数据的全局统计失真，并揭示了恶意攻击者可通过注入有偏上下文示例来破坏下游模型公平性的对抗性风险。

    

    大语言模型（LLM）越来越多地通过上下文学习（ICL）用于合成表格数据的生成，为数据稀缺场景下的数据增强提供了实用的解决方案。尽管先前的工作已经展示了LLM通过增强代表性不足群体来提升下游任务性能的潜力，但这些好处通常假设可以访问无偏的、能代表真实数据集的上下文示例子集。然而，在现实环境中，数据往往充满噪声且在人口统计分布上存在偏斜。在本文中，我们系统地研究了上下文示例中的统计偏差如何传播到合成表格数据中，结果表明即使是轻微的上下文偏差也会导致全局性的统计失真。我们进一步引入了一个对抗性场景，其中恶意贡献者可以通过一部分上下文示例向合成数据集注入偏差，最终损害下游模型的公平性。

    arXiv:2506.09630v3 Announce Type: replace  Abstract: Large Language Models (LLMs) are increasingly used for synthetic tabular data generation through in-context learning (ICL), offering a practical solution for data augmentation in data scarce scenarios. While prior work has shown the potential of LLMs to improve downstream task performance through augmenting underrepresented groups, these benefits often assume access to a subset of unbiased in-context examples, representative of the real dataset. In real-world settings, however, data is frequently noisy and demographically skewed. In this paper, we systematically study how statistical biases within in-context examples propagate to synthetic tabular data, showing that even mild in-context biases lead to global statistical distortions. We further introduce an adversarial scenario where a malicious contributor can inject bias into the synthetic dataset via a subset of in-context examples, ultimately compromising the fairness of downstrea
    
[^368]: 质量多样性红队测试：面向大语言模型的高质量多样化攻击者自动生成

    Quality-Diversity Red-Teaming: Automated Generation of High-Quality and Diverse Attackers for Large Language Models

    [https://arxiv.org/abs/2506.07121](https://arxiv.org/abs/2506.07121)

    本文提出质量多样性红队测试（QDRT）框架，通过自动生成高质量且多样化的攻击者，克服了以往红队测试中多样性度量过于简单化和单一攻击者模型覆盖不足的局限，实现对大语言模型更全面的安全评估。

    

    确保大语言模型（LLMs）的安全性和鲁棒性是一项根本性挑战，也是负责任地部署人工智能的关键前提。红队测试是一种系统性框架，用于识别能够从目标大语言模型中诱发出有害回应的对抗性提示，已成为至关重要的安全评估范式。在这一框架中，对抗性提示的多样性对于全面的安全评估至关重要。然而，以往的红队测试方法通常通过词频或句子嵌入相似度等简单化的指标来追求多样性，这可能无法捕捉到攻击策略中有意义的变化。此外，训练单一攻击者模型的常见做法限制了对所有潜在攻击风格和风险类别的覆盖范围。本文引入了质量多样性红队测试（QDRT），这是一个旨在解决上述局限性的新框架。QDRT 实现了

    arXiv:2506.07121v3 Announce Type: replace  Abstract: Ensuring the safety and robustness of large language models (LLMs) is a fundamental challenge and a critical prerequisite for the responsible deployment of artificial intelligence. Red-teaming, a systematic framework to identify adversarial prompts that elicit harmful responses from target LLMs, has emerged as a crucial safety evaluation paradigm. Within this framework, the diversity of adversarial prompts is critical for a comprehensive safety assessment. However, previous red-teaming approaches often pursue diversity through simplistic metrics such as word frequency or sentence embedding similarity, which may not capture meaningful variation in attack strategies. In addition, the common practice of training a single attacker model restricts coverage across all potential attack styles and risk categories. This paper introduces Quality-Diversity Red-Teaming (QDRT), a new framework designed to address these limitations. QDRT achieves 
    
[^369]: 从最后一层理解并缓解图神经网络的置信度不足问题

    Understanding and Mitigating Under-Confidence in GNNs from the Final Layer

    [https://arxiv.org/abs/2505.11335](https://arxiv.org/abs/2505.11335)

    本文建立了一个统一的理论框架，揭示GNN的置信度不足由最后一层的类中心层面和节点层面校准共同决定，并从理论上证明减少最后一层参数的权重衰减可从类中心层面缓解置信度不足问题。

    

    图神经网络（GNN）在基于图的任务上已经展现出卓越的有效性。然而，它们的预测置信度往往校准不佳，通常表现为置信度不足，这损害了其决策的可靠性。现有的GNN校准方法通常会引入额外的校准组件，这些组件未能捕捉模型与预测置信度之间的内在关系，导致理论保证有限且计算开销增加。为了解决这一问题，我们提出了一种简单而高效的图校准方法。我们建立了一个统一的理论框架，揭示了模型置信度在最后一层由类中心层面和节点层面的校准共同决定。基于这一洞察，我们从理论上证明，减少最后一层参数的权重衰减可以通过作用于类中心层面来缓解GNN的置信度不足问题，同时……

    arXiv:2505.11335v5 Announce Type: replace  Abstract: Graph Neural Networks (GNNs) have demonstrated remarkable effectiveness on graph-based tasks. However, their predictive confidence is often miscalibrated, typically exhibiting under-confidence, which harms the reliability of their decisions. Existing calibration methods for GNNs normally introduce additional calibration components, which fail to capture the intrinsic relationship between the model and the prediction confidence, resulting in limited theoretical guarantees and increased computational overhead. To address this issue, we propose a simple yet efficient graph calibration method. We establish a unified theoretical framework revealing that model confidence is jointly governed by class-centroid-level and node-level calibration at the final layer. Based on this insight, we theoretically show that reducing the weight decay of the final-layer parameters alleviates GNN under-confidence by acting on the class-centroid level, while
    
[^370]: 基于联邦学习客户端对抗性观测的远程Rowhammer攻击

    Remote Rowhammer Attack using Adversarial Observations on Federated Learning Clients

    [https://arxiv.org/abs/2505.06335](https://arxiv.org/abs/2505.06335)

    该研究首次揭示了针对联邦学习服务器的新型远程Rowhammer攻击：攻击者利用强化学习智能体在客户端生成对抗性观测，引发服务器重复性的聚簇内存更新，从而在无需后门权限的情况下通过位翻转远程破坏服务器内存。

    

    联邦学习（FL）具有在大量并行智能体之间实现同步全局学习的潜力，使新兴的人工智能系统能够在人口统计多样化的数据上进行训练。实现高效训练的关键在于联邦学习能够在中央服务器上执行稀疏梯度更新和远程直接内存访问。现有大多数联邦学习安全研究侧重于保护边缘客户端或客户端与服务器之间通信通道的数据隐私。而面向客户端针对服务器的攻击则研究较少，因为人们普遍认为大量客户端的集体行为能够提供抗攻击韧性。在此，我们首次证明，通过攻击客户端从而在服务器中引发重复性的聚簇内存更新，攻击者可以远程发起Rowhammer攻击，通过诱导位翻转来破坏服务器内存。一个无需服务器后门访问权限的强化学习（RL）攻击者可以学会如何……（原文摘要在此处截断）

    arXiv:2505.06335v3 Announce Type: replace  Abstract: Federated Learning (FL) has the potential for simultaneous global learning amongst a large number of parallel agents, enabling emerging AI systems to be trained across demographically diverse data. Central to this being efficient is the ability for FL to perform sparse gradient updates and remote direct memory access at the central server. Most of the research in FL security focuses on protecting data privacy at the edge client or in the communication channels between the client and server. Client-facing attacks on the server are less well investigated as the assumption is that a large collective of clients offer resilience. Here, for the first time, we show that by attacking clients that lead to repetitive clustered memory updates in the server, we can remote initiate a rowhammer attack to corrupt the server memory with induced bit-flips. A reinforcement learning (RL) attacker without backdoor access to the server can learn how to m
    
[^371]: 学习在非正交多址接入联合信源信道编码中主动利用干扰

    Learning to Interfere in Non-Orthogonal Multiple-Access Joint Source-Channel Coding

    [https://arxiv.org/abs/2504.03690](https://arxiv.org/abs/2504.03690)

    提出一种基于多视图自编码器的机器学习方法，将压缩与信道编码融合，使多个发射机能够同时共享全部信道资源以实现非正交多址（NOMA）联合信源信道编码，接收端可从叠加信号中恢复所有图像并识别其对应的发射机。

    

    我们考虑多个发射机通过多址接入信道（MAC）传输其源信号（例如图像）的场景。传统通信系统通过在用户之间正交分配资源（时间和/或带宽）来最小化干扰，这限制了系统容量。我们提出了一种机器学习（ML）辅助的无线图像传输方法，该方法使用多视图自编码器将压缩与信道编码融合在一起，允许发射机同时使用所有可用的信道资源，从而形成一种非正交多址接入（NOMA）方案。接收机必须从接收到的叠加信号中恢复所有图像，同时还要将每张图像与其对应的发射机关联起来。传统机器学习模型处理的是单个样本，而我们的模型允许来自不同用户的信号相互干扰，以便在有限的带宽和功率约束下充分利用NOMA带来的增益。

    arXiv:2504.03690v2 Announce Type: replace-cross  Abstract: We consider multiple transmitters aiming to communicate their source signals (e.g., images) over a multiple access channel (MAC). Conventional communication systems minimize interference by orthogonally allocating resources (time and/or bandwidth) among users, which limits their capacity. We introduce a machine learning (ML)-aided wireless image transmission method that merges compression and channel coding using a multi-view autoencoder, which allows the transmitters to use all the available channel resources simultaneously, resulting in a non-orthogonal multiple access (NOMA) scheme. The receiver must recover all the images from the received superposed signal, while also associating each image with its transmitter. Traditional ML models deal with individual samples, whereas our model allows signals from different users to interfere in order to leverage gains from NOMA under limited bandwidth and power constraints. We introduc
    
[^372]: 量化临床决策中的症状因果性：基于CausaLM的探索

    Quantifying Symptom Causality in Clinical Decision Making: An Exploration Using CausaLM

    [https://arxiv.org/abs/2503.19394](https://arxiv.org/abs/2503.19394)

    本研究利用CausaLM框架和TReATE方法量化了关键症状（如胸痛）对临床诊断模型预测结果的因果效应，超越了传统基于相关性的分析方法，为理解临床NLP模型的决策机制提供了新途径。

    

    当前用于医学诊断的机器学习方法往往依赖于症状与疾病之间的相关性模式，当症状模糊或在多种疾病中普遍存在时，存在误诊风险。在这项工作中，我们超越相关性分析，研究关键症状——特别是“胸痛”——对诊断预测的因果影响。利用CausaLM框架，我们生成了有效“遗忘”目标概念的反事实文本表示，从而能够有原则地估计该概念对模型预测疾病分布的因果效应。通过采用基于文本表示的平均处理效应方法，我们量化了症状的存在与否如何影响模型的诊断结果，并将这些发现与CONEXP等基于相关性的基线方法进行对比。我们的结果为临床NLP模型的决策行为提供了更深入的洞察。

    arXiv:2503.19394v2 Announce Type: replace  Abstract: Current machine learning approaches to medical diagnosis often rely on correlational patterns between symptoms and diseases, risking misdiagnoses when symptoms are ambiguous or common across multiple conditions. In this work, we move beyond correlation to investigate the causal influence of key symptoms-specifically "chest pain" on diagnostic predictions. Leveraging the CausaLM framework, we generate counterfactual text representations in which target concepts are effectively "forgotten" enabling a principled estimation of the causal effect of that concept on a model's predicted disease distribution. By employing Textual Representation-based Average Treatment Effect (TReATE), we quantify how the presence or absence of a symptom shapes the model's diagnostic outcomes, and contrast these findings against correlation-based baselines such as CONEXP. Our results offer deeper insight into the decision-making behavior of clinical NLP models
    
[^373]: 在高斯过程上下文老虎机优化中利用凹性信息

    Exploiting Concavity Information in Gaussian Process Contextual Bandit Optimization

    [https://arxiv.org/abs/2503.10836](https://arxiv.org/abs/2503.10836)

    本文提出一种将保持凹性的回归样条基与受凹性约束的高斯过程后验相结合的新型奖励模型，并据此构建UCB算法，从而在上下文老虎机优化中利用已知的凹性结构提升序贯决策的优化效率。

    

    上下文老虎机对序贯决策问题进行建模，其中奖励同时取决于所选动作和观测到的上下文。在医学、商业和工程等许多领域中，先验知识提供了关于奖励函数的结构信息，可以利用这些信息来提高优化效率。本文研究这样一种设置：对于每个可观测的上下文，条件平均奖励已知关于动作变量是凹的。为了利用这一结构，我们开发了一种上下文老虎机算法，该算法在凹性约束条件下对贝叶斯高斯过程后验进行条件化。我们提出了一种新颖的奖励模型，将保持凹性的回归样条基与受约束的高斯过程后验相结合，得到一个易于处理的形状约束估计器。基于该模型，我们构建了UCB算法，并为受约束后验建立了新的后验集中不等式。

    arXiv:2503.10836v2 Announce Type: replace-cross  Abstract: The contextual bandit models sequential decision-making problems in which rewards depend on both the chosen action and observed context. In many domains such as medicine, business, and engineering, prior knowledge provides structural information about the reward function that can be exploited to improve optimization efficiency. This paper studies settings where, for each observable context, the conditional mean reward is known to be concave with respect to the action variable. To leverage this structure, we develop a contextual bandit algorithm that conditions a Bayesian Gaussian Process posterior on concavity constraints. We propose a novel reward model that combines a concavity-preserving regression spline basis with a constrained GP posterior, yielding a tractable shape-constrained estimator. Building on this model, we construct a UCB algorithm and establish new posterior concentration inequalities for the constrained poster
    
[^374]: AgiBot World Colosseo：面向可扩展智能具身系统的大规模操作平台

    AgiBot World Colosseo: A Large-scale Manipulation Platform for Scalable and Intelligent Embodied Systems

    [https://arxiv.org/abs/2503.06669](https://arxiv.org/abs/2503.06669)

    AgiBot World是一个包含超过100万条轨迹、覆盖217个任务的大规模机器人操作数据平台，配合利用潜在动作表征的通用策略GO-1，实现了数量级的数据规模提升和平均30%的性能改进。

    

    我们探索可扩展的机器人数据如何应对通用机器人操作在现实世界中的挑战。我们推出了AgiBot World，这是一个大规模平台，包含超过100万条轨迹，涵盖五种部署场景中的217个任务，与现有数据集相比实现了数量级的数据规模提升。借助带有人类在环验证的标准化采集流水线加速，AgiBot World保证了高质量和多样化的数据分布。该平台可从夹爪扩展至灵巧手和视觉-触觉传感器，用于细粒度技能的获取。在数据的基础上，我们提出了Genie Operator-1（GO-1），这是一种新颖的通用策略，利用潜在动作表征来最大化数据利用率，展现出随数据量增加而可预测的性能扩展。在我们的数据集上预训练的策略相比在其他数据集上训练的策略平均性能提升30%。

    arXiv:2503.06669v5 Announce Type: replace-cross  Abstract: We explore how scalable robot data can address real-world challenges for generalized robotic manipulation. Introducing AgiBot World, a large-scale platform comprising over 1 million trajectories across 217 tasks in five deployment scenarios, we achieve an order-of-magnitude increase in data scale compared to existing datasets. Accelerated by a standardized collection pipeline with human-in-the-loop verification, AgiBot World guarantees high-quality and diverse data distribution. It is extensible from grippers to dexterous hands and visuo-tactile sensors for fine-grained skill acquisition. Building on top of data, we introduce Genie Operator-1 (GO-1), a novel generalist policy that leverages latent action representations to maximize data utilization, demonstrating predictable performance scaling with increased data volume. Policies pre-trained on our dataset achieve an average performance improvement of 30% over those trained on
    
[^375]: 优化共形选择：数据依赖评分构建后的强效选择性推断

    Optimized Conformal Selection: Powerful Selective Inference After Data-Dependent Score Construction

    [https://arxiv.org/abs/2411.17983](https://arxiv.org/abs/2411.17983)

    该论文提出了优化共形选择（OptCS）通用框架，突破了现有方法要求评分函数独立于校准和测试数据的限制，在数据依赖的评分构建和大量数据复用的情况下仍能保证有限样本的错误发现率（FDR）控制。

    

    我们研究共形选择中的数据驱动评分构建问题，该方法利用共形p值和多重检验，从无标签数据池中筛选出具有较大未观测标签的“有趣”实例，同时在有限样本下控制错误发现率（FDR）。现有解决方案要求评分函数的构建必须独立于校准数据和测试数据，然而，当面临众多评分选择和有限的标注数据时，更理想的做法是以数据驱动的方式选择或构建强有力的符合性评分，并利用所有标注数据进行评分训练。我们提出了优化共形选择，这是一个通用框架，允许在灵活的数据驱动评分构建之后进行有效的多重检验。我们给出了一般性条件，使得OptCS在存在大量数据复用的情况下仍能构建有效的共形p值，并处理复杂的p值依赖关系，以维持有限样本的FDR控制。

    arXiv:2411.17983v2 Announce Type: replace-cross  Abstract: We study data-driven score construction in conformal selection, which uses conformal p-values and multiple testing to select "interesting" instances with large unobserved labels from a pool of unlabeled data, while controlling the FDR in finite sample. While existing solutions require the score function to be constructed independent of the calibration and test data, it is desirable to select or construct powerful conformity score in a data-driven manner and to leverage all labeled data for score training when presented with many score choices and limited labeled data. We propose Optimized Conformal Selection (OptCS), a general framework that allows valid multiple testing after flexible data-driven score construction. We introduce general conditions under which OptCS constructs valid conformal p-values despite substantial data reuse and handles complex p-value dependencies to maintain finite-sample FDR control. We instantiate th
    
[^376]: 面向更平坦极小值的双边锐度感知最小化

    Bilateral Sharpness-Aware Minimization for Flatter Minima

    [https://arxiv.org/abs/2409.13173](https://arxiv.org/abs/2409.13173)

    本文提出双边锐度感知最小化（BSAM），通过融合最大锐度（MaxS）与最小锐度（MinS）构建更优的平坦度指标，克服了SAM仅关注梯度上升方向平坦度的“平坦度指标问题”，从而找到更平坦的极小值以提升神经网络泛化能力。

    

    锐度感知最小化（SAM）通过降低最大锐度来增强模型的泛化能力。尽管SAM在实践中取得了成功，我们通过实验发现，SAM泛化提升背后的最大锐度面临“平坦度指标问题”（FIP），即SAM仅考虑梯度上升方向上的平坦度，导致下一步最小化区域不够平坦。更好的平坦度指标（FI）将带来神经网络更好的泛化能力，而SAM本质上是一种贪婪搜索方法。在本文中，我们提出利用当前权重周围邻域内训练损失与最小损失之间的差值，我们将其称为最小锐度。通过融合MaxS和MinS，我们构建了一个更好的平坦度指标，能够在优化过程中指示更平坦的方向。特别地，我们将该平坦度指标与SAM相结合，提出了双边SAM（BSAM），它能够找到更平坦的极小值。

    arXiv:2409.13173v2 Announce Type: replace-cross  Abstract: Sharpness-Aware Minimization (SAM) enhances generalization by reducing a Max-Sharpness (MaxS). Despite the practical success, we empirically found that the MAxS behind SAM's generalization enhancements face the "Flatness Indicator Problem" (FIP), where SAM only considers the flatness in the direction of gradient ascent, resulting in a next minimization region that is not sufficiently flat. A better Flatness Indicator (FI) would bring a better generalization of neural networks. Because SAM is a greedy search method in nature. In this paper, we propose to utilize the difference between the training loss and the minimum loss over the neighborhood surrounding the current weight, which we denote as Min-Sharpness (MinS). By merging MaxS and MinS, we created a better FI that indicates a flatter direction during the optimization. Specially, we combine this FI with SAM into the proposed Bilateral SAM (BSAM) which finds a more flatter mi
    
[^377]: 基于动态加权的100种语言多标签情感分析（针对标签不平衡问题）

    Multi-Label Sentiment Analysis on 100 Languages with Dynamic Weighting for Label Imbalance

    [https://arxiv.org/abs/2008.11573](https://arxiv.org/abs/2008.11573)

    该论文提出了一种动态加权方法来应对多标签情感分析中的标签不平衡问题，并结合适配的多标签焦点损失与线性时间复杂度的最优阈值选择方法，在100种语言的情感分析任务中于9个指标中的7个上达到了最先进水平。

    

    我们研究跨语言情感分析，该领域因其在市场研究、政治和社会科学等多个领域的应用而受到广泛关注。特别是，由于情感遵循普拉奇克情感轮模型，我们引入了一个多标签设置下的情感分析框架。我们提出了一种新颖的动态加权方法，在训练过程中平衡每个类别的贡献，这不同于以往根据类别频率分配固定权重的静态加权方法。此外，我们将单标签目标识别文献中偏向较难样本的焦点损失（focal loss）适配到我们的多标签设置中。进一步地，我们推导出一种在线性时间复杂度内选择最优类别特定阈值以最大化宏平均F1分数的方法。通过大量实验，我们证明我们的方法在3个不同（数据集）的9个指标中有7个取得了最先进的性能。

    arXiv:2008.11573v2 Announce Type: replace-cross  Abstract: We investigate cross-lingual sentiment analysis, which has attracted significant attention due to its applications in various areas including market research, politics and social sciences. In particular, we introduce a sentiment analysis framework in multi-label setting as it obeys Plutchik's wheel of emotions. We introduce a novel dynamic weighting method that balances the contribution from each class during training, unlike previous static weighting methods that assign non-changing weights based on their class frequency. Moreover, we adapt the focal loss that favors harder instances from single-label object recognition literature to our multi-label setting. Furthermore, we derive a method to choose optimal class-specific thresholds that maximize the macro-f1 score in linear time complexity. Through an extensive set of experiments, we show that our method obtains the state-of-the-art performance in 7 of 9 metrics in 3 differen
    
[^378]: 选择后泛化的极限

    The Limits of Post-Selection Generalization

    [https://arxiv.org/abs/1806.06100](https://arxiv.org/abs/1806.06100)

    本文揭示了事后泛化算法的能力极限，证明了任何满足事后泛化并回答自适应选择统计查询的算法其误差都存在紧的下界，为选择后数据分析的进展设置了一个根本性障碍。

    

    尽管统计学和机器学习提供了许多确保泛化的方法，但这些方法在存在自适应性的情况下往往会失效——自适应性是指一种常见的做法，即分析方式的选择取决于之前与同一数据集的交互。近期的一系列工作引入了功能强大的通用算法来确保事后泛化（也称为鲁棒泛化或选择后泛化），其含义是：给定算法的输出后，很难找到任何一个统计量使得数据与其来源总体之间存在显著差异。在这项工作中，我们展示了满足事后泛化的算法在能力上的若干局限。首先，我们对任何满足事后泛化并回答自适应选择的统计查询的算法的误差给出了紧的下界，这表明选择后数据分析的进展存在一个强大的障碍。其次，我们证明了事后泛化……

    arXiv:1806.06100v2 Announce Type: replace  Abstract: While statistics and machine learning offer numerous methods for ensuring generalization, these methods often fail in the presence of adaptivity---the common practice in which the choice of analysis depends on previous interactions with the same dataset. A recent line of work has introduced powerful, general purpose algorithms that ensure post hoc generalization (also called robust or post-selection generalization), which says that, given the output of the algorithm, it is hard to find any statistic for which the data differs significantly from the population it came from.   In this work we show several limitations on the power of algorithms satisfying post hoc generalization. First, we show a tight lower bound on the error of any algorithm that satisfies post hoc generalization and answers adaptively chosen statistical queries, showing a strong barrier to progress in post selection data analysis. Second, we show that post hoc genera
    
[^379]: 超模数秩：集合函数分解与优化

    Supermodular Rank: Set Function Decomposition and Optimization. (arXiv:2305.14632v1 [math.CO])

    [http://arxiv.org/abs/2305.14632](http://arxiv.org/abs/2305.14632)

    本论文介绍了超模数秩的概念，并使用子模函数分解优化集合函数，提高了几个算法的逼近比保证。

    

    我们定义了格上函数的超模数秩。这是将其分解为超模函数之和所需的最少项数。超模和式根据不同的偏序定义。我们表征了超模数秩的最大可能值，并描述了具有固定超模数秩的函数。我们同样定义了次模数秩，并使用次模函数的分解来优化集合函数。给定集合函数的次模数秩上界，我们制定了一个算法，将优化问题分解为子模函数子问题。我们展示了该方法提高了几个单调集合函数最大化和集合函数比值极小化算法的逼近比保证，计算开销取决于次模数秩。

    We define the supermodular rank of a function on a lattice. This is the smallest number of terms needed to decompose it into a sum of supermodular functions. The supermodular summands are defined with respect to different partial orders. We characterize the maximum possible value of the supermodular rank and describe the functions with fixed supermodular rank. We analogously define the submodular rank. We use submodular decompositions to optimize set functions. Given a bound on the submodular rank of a set function, we formulate an algorithm that splits an optimization problem into submodular subproblems. We show that this method improves the approximation ratio guarantees of several algorithms for monotone set function maximization and ratio of set functions minimization, at a computation overhead that depends on the submodular rank.
    

