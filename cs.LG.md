# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RL Post-Training in Diffusion Transformers Is Activation Steering](https://arxiv.org/abs/2611.26627) | 该论文发现扩散Transformer的强化学习后训练收益可以用一组与输入无关的固定导向向量（参数量仅为LoRA的1/65）复现90%以上，证明RL后训练本质上等价于简单的激活导向。 |
| [^2] | [TideCache: Efficient Hybrid LLM Serving with Transparent Cache Management and Layer-Wise Streaming](https://arxiv.org/abs/2611.26619) | TideCache是面向CPU-GPU超级芯片上混合架构LLM的服务系统，通过透明缓存池、逐层缓存流式传输和请求调度三大组件，动态管理异构缓存的显存分配与驻留，并将数据传输与计算重叠，从而实现高效服务。 |
| [^3] | [One Prior, Two Dynamics: Repurposing Video Diffusion Transformers for Self-Supervised Latent Action Learning](https://arxiv.org/abs/2611.26617) | 该论文提出O²LaF，通过将同一个预训练视频扩散Transformer分别适配为逆向动力学过渡编码器和前向动力学动作条件生成器，从无标注视频中自监督学习可迁移的潜在动作，并在导航与机器人操作的少样本动作探测中达到新的最先进水平。 |
| [^4] | [Bio-PM-Guided Synthetic IMU Generation](https://arxiv.org/abs/2611.26607) | 该论文提出利用预训练生物信号编码器Bio-PM的冻结编码器将IMU数据转换到token嵌入表示空间，并通过自回归transformer解码器和扩散解码器两类类条件解码器生成合成IMU数据，作为人体活动识别中提升泛化能力的实用数据增强方法。 |
| [^5] | [Depth Pro 2: Sharp and Temporally Consistent Metric Monocular Depth](https://arxiv.org/abs/2611.26603) | Depth Pro 2 通过单一可扩展网络同时处理图像与视频，生成兼具度量准确、边界清晰和时间一致性的深度图，突破了以往方法在一致性与清晰度之间的权衡。 |
| [^6] | [Active Feature Acquisition by Distilling an Oracle Teacher](https://arxiv.org/abs/2611.26598) | 该论文提出OracleImitator，首次显式构建了一个按候选特征对下游预测器误差影响进行排序的神谕教师，并通过软目标模仿将这一排序蒸馏为可部署的主动特征获取策略，在多模态医疗诊断和AFABench基准上于整个预算范围内匹配或超越现有方法，恢复约90%的可达准确率增益。 |
| [^7] | [Realized versus Population Reliability: Anytime-Valid Self-Consistency Certification under Correlated Reasoning](https://arxiv.org/abs/2611.26592) | 该论文发现语言模型推理流之间存在正相关，破坏了自一致性认证所依赖的条件独立假设，使被认证的假设分裂为关于单次抽取信念的“实现零假设”与关于模型平均行为的“总体零假设”（仅在零相关时二者等价），从而从根本上改变了 anytime-valid 证书所真正保证的可靠性含义。 |
| [^8] | [Probing the "Conscious" Mind: Selective Access and Global Workspace in Vision-Language Models](https://arxiv.org/abs/2611.26590) | 该论文提出多模态概念透镜（MC-Lens），通过将视觉编码器与语言解码器的中间状态映射到共享语义空间，并结合内部概念干预，揭示视觉-语言模型中概念的可及性与选择性访问机制，为全局工作空间理论在AI系统中的验证提供了工具。 |
| [^9] | [Represented but misread: Post-training changes how language models read out what they represent](https://arxiv.org/abs/2611.26584) | 研究发现后训练并未破坏语言模型内部的表征（线性探针仍能准确区分人类与AI文本），而是改变了模型将表征读取为答案的方式，导致回答灵敏度下降和对先验信息的错误利用。 |
| [^10] | [Text-Aware, Vision-Blind: How Multimodal Image Editing Systems Handle Unsafe Inputs](https://arxiv.org/abs/2611.26569) | 该论文提出EditSafetyBench基准，揭示多模态图像编辑系统存在系统性的文本-视觉安全失配：系统高度依赖文本提示进行安全审查（拒绝率超80%），但对不安全图像配合中性指令的情形检出率极低（仅约20%多），构成重大安全隐患。 |
| [^11] | [Cost-Aware Agentic Feature Optimization System for Fleet-Scale Ranking Models](https://arxiv.org/abs/2611.26531) | 该论文提出了一个端到端的LLM智能体系统，在跨模型共享与逐模型设定的资源约束下，为大规模排序模型集群自动化地完成假设生成、交叉特征生成与筛选、多方法重要性排序融合以及异步试验调优，实现成本感知的跨模型特征优化。 |
| [^12] | [Deterministic Temporal Embedding for Calcium Imaging Trace Extraction](https://arxiv.org/abs/2611.26522) | 该论文提出一种确定性时序嵌入框架，通过避免随机负采样的确定性优化方法，将钙成像中逐像素荧光轨迹映射到时间相干源形成空间相干区域的低维表示，从而无需细胞形状先验即可可靠提取钙成像信号轨迹。 |
| [^13] | [The Window Lever: Longer Observation Windows Outweigh the Observed Effects of Data, Capacity, and Architecture in Customer Behavior Forecasting](https://arxiv.org/abs/2611.26521) | 在长时程客户行为预测中，将观察窗口从8周延长到16周所带来的性能提升超过了增加数据量、扩大模型容量或采用新架构的效果，且匹配测试表明这一增益甚至被原始比较所低估。 |
| [^14] | [Simple Private Noisy SGD is Minimax-Optimal under Population Strong Convexity](https://arxiv.org/abs/2611.26515) | 本文证明了在强凸损失（尤其是强凸广义线性模型，如隐私逻辑回归）上，采用固定步长的单次噪声SGD即可达到极小极大最优的估计误差率，其上界与基于Score Attack框架推导的隐私下界相匹配，并首次消除了对数因子。 |
| [^15] | [Hybrid Retrieval for Government Scheme Search in English and Marathi](https://arxiv.org/abs/2611.26514) | 该论文提出一个面向马哈拉施特拉邦政府计划的双语（英语-马拉地语）搜索系统，通过等权重融合BM25与句子嵌入相似度，混合检索显著优于单独的BM25或密集检索，MRR最高达0.879。 |
| [^16] | [Grid Cells as Spectral Successor Features: A Reusable Basis for Continual Reinforcement Learning](https://arxiv.org/abs/2611.26504) | 该论文受网格细胞对应后继表示特征向量的启发，提出用固定参考策略转移矩阵的特征向量作为后继特征的可复用基，使得长期回报预测具有封闭形式的系数，且无需后继特征网络或回放缓冲区，为持续强化学习提供了高效基础。 |
| [^17] | [Anticipating Future Build Results with BuildPredictor](https://arxiv.org/abs/2611.26476) | 本文提出基于随机森林的BuildPredictor模型，利用源代码与构建代码的变更、代码复杂度变化以及历史构建结果等特征来预测CI构建结果，在项目内和整体设置中分别取得了0.82和0.86的中位数MCC，有望减少CI中因微小更改而浪费的时间和资源。 |
| [^18] | [A new algorithmic approach for finding flat minima: critically damped Polyak stepsizes](https://arxiv.org/abs/2611.26475) | 本文提出临界阻尼Polyak（CDP）自适应步长——将经典Polyak规则乘以4并加温和阻尼——使梯度下降偏向更平坦的极小值，并首次为局部二次目标函数建立了锐度与损失下降的时间一致理论保证。 |
| [^19] | [Neural Conditional Gaussian Model: Efficient Data Assimilation with Sequential Uncertainty Quantification for Nonlinear Dynamics](https://arxiv.org/abs/2611.26463) | 本文提出神经条件高斯模型，将未观测的物理状态映射到具有条件高斯结构的潜在空间，从而实现闭式数据同化更新，并通过高斯解码器对物理变量进行高效的不确定性量化。 |
| [^20] | [Bayesian optimization on amortized search space](https://arxiv.org/abs/2611.26459) | 该论文提出了一种利用机理模拟器的摊销式逆映射来约束搜索空间的贝叶斯优化方法，通过感知问题几何的高斯过程核高效利用问题的低内在维度，并给出了依赖内在维度的遗憾率理论保证，成功应用于帕金森病的药物剂量个性化。 |
| [^21] | [Scaling classical hardware for accurate real-time quantum error correction](https://arxiv.org/abs/2611.26447) | 本文提出与FPGA协同设计的流式神经解码器Rapids，在单块FPGA上以亚微秒级延迟对距离-19表面码进行实时解码，同时兼顾速度与精度，满足了超导量子处理器实时量子纠错的苛刻需求。 |
| [^22] | [Should Belgians Install an Advanced Energy Management System in Their Electrified Dwelling ?](https://arxiv.org/abs/2611.26441) | 本研究通过模拟500户比利时电气化住宅，评估模型预测控制（MPC）与基于决策树的TreeC等先进能源管理系统相比传统规则控制器，在热泵供暖、电热水和电动汽车充电场景下的潜在收益。 |
| [^23] | [Learning to Defer under Expert Drift](https://arxiv.org/abs/2611.26440) | 提出EDA-L2D框架，利用序列模型对专家近期决策结果历史进行建模，显式捕捉因疲劳、认知偏差等因素导致的专家可靠性时间漂移，实现自适应的延迟决策策略。 |
| [^24] | [Exact Bayesian layers via characteristic functions](https://arxiv.org/abs/2611.26426) | 本文提出CF-Prop方法，利用特征函数无采样地精确计算贝叶斯ReLU层的输出分布（即使在非高斯情形下），并能对高斯、泊松、伽马等多种似然闭式计算证据下界，从而实现高效的变分推断算法CF-VI，在CIFAR-100上取得最低校准误差。 |
| [^25] | [FEA solver integration of a composite World Model with a Spiking Neural Network memory Block for computationally efficient modelling of defects and variabilities](https://arxiv.org/abs/2611.26419) | 该论文将基于多模态脉冲Transformer的复合材料世界模型与脉冲神经网络记忆模块集成到有限元求解器中，实现了对纤维增强复合材料中缺陷与材料变异性的计算高效实时建模。 |
| [^26] | [Sequential Anytime Valid Tests for Efficient Inference-Time Alignment](https://arxiv.org/abs/2611.26413) | 该论文提出了一种基于奖励加权自一致性分数与 Wald 序列概率比检验的任意时有效停止规则 SPRT-RWSC，能在给定错误容忍度下统计可控地高效判定推理时对齐中的多数答案，避免固定预算造成的计算浪费。 |
| [^27] | [How to Effectively Manage the Energy in Residential Systems: Development and Comparison of Methods](https://arxiv.org/abs/2611.26412) | 本论文通过仿真与真实实验比较了规则、优化和机器学习三类能源管理方法，并提出了TreeC——一种新颖的机器学习方法，能够学习既易于理解又高效的住宅能源管理策略。 |
| [^28] | [Statistical Inference with Ill-Conditioned Random Weight Neural Network Sieves](https://arxiv.org/abs/2611.26410) | 本文证明了随机权重神经网络筛在维数增大时必然出现病态性（最小特征值消失），并在显式处理该病态性的基础上建立了相应的逼近速率、估计与统计推断理论。 |
| [^29] | [Flow Matching from Non-Gaussian Sources: A Theory of Efficient Sampling](https://arxiv.org/abs/2611.26407) | 本文为从非高斯源分布（如重尾和多峰分布）出发的流匹配建立了首个非渐近收敛理论，证明欧拉离散化采样器在温和条件下仅需 $\widetilde{O}(Ld\mathsf{TotVar}_{\mathsf{base}}/\varepsilon)$ 次迭代即可达到 $\varepsilon$ 级总变差误差。 |
| [^30] | [Simulation-pretrained autoencoders versus PCA for compressing multimode fibre transmission matrices](https://arxiv.org/abs/2611.26391) | 该研究表明，基于仿真预训练并仅用两个实测传输矩阵适配的置换等变自编码器，在62倍压缩下即可实现98%的理想聚焦增强，且对平移和缩放具有鲁棒性，但作为纯压缩器其精度仍比PCA低约十倍。 |
| [^31] | [A Complete Geometric Message-Passing Architecture for Point Clouds](https://arxiv.org/abs/2611.26384) | 该论文提出了一种新颖而简单的等变消息传递架构用于点云，并从理论上证明了其完全性，无需依赖计算密集的高阶旋转群表示或高阶Weisfeiler-Leman测试即可区分任意非对称等价的输入。 |
| [^32] | [Evidence from a Multi-Regime Causal Discovery Benchmark: UCMT Algorithms, Comparative Performance, and Controlled Ablation Studies](https://arxiv.org/abs/2611.26383) | 该论文通过预注册的多机制因果发现基准比较与受控消融研究，证明UCMT v2 Global在部分真实、半合成及高斯网络场景下有向F1优于NOTEARS与DAGMA，而基于BOSS的UCMT v4在CSuite上带来配对级显著改进，并揭示了影响算法表现的内部阶段。 |
| [^33] | [From Benchmarks to the Bench: Preference, Correspondence and Selectivity in Bidirectional Drug-Target Screening](https://arxiv.org/abs/2611.26369) | 该研究对13个已发布的药物-靶点相互作用模型进行了大规模双向基准比较，发现不同模型给出的Top-10推荐列表在77.8%（分子）和70.1%（蛋白）的对比中完全不相交，且系统间的一致性主要反映对相同候选物的重复偏好而非真实的对应关系，实验验证进一步表明仅部分模型能够富集活性化合物且选择性有限。 |
| [^34] | [HiSA: Sparsifying the Hidden Dimension of RoPE Attention in LLMs](https://arxiv.org/abs/2611.26363) | 提出HiSA方法，基于RoPE结构将注意力点积在隐藏（特征）维度上进行稀疏化，仅需计算约三分之一的查询-键点积就能在LLM训练中提升而非损害性能。 |
| [^35] | [Learning from Context: Leave-One-Out Prediction in Jump Diffusions](https://arxiv.org/abs/2611.26359) | 该论文提出在跳跃扩散模型中采用留一（LOO）预测——每个坐标仅依据其他坐标的破坏状态进行预测——并证明通过解析似然补全可从留一概率构造漂移项，在保持规定边缘分布的同时无需额外网络评估，从而促使模型真正学习坐标间的依赖关系。 |
| [^36] | [Mamba Flow Matching Neural Processes: Linear-Time Inference for Irregularly Observed Spatial Fields](https://arxiv.org/abs/2611.26355) | 该论文提出MambaFlowNP，将流匹配与Mamba状态空间模型引入神经过程，实现对不规则观测空间场的线性时间插值推断，突破了自注意力机制 O(K²) 计算开销对可处理点数的限制。 |
| [^37] | [TRCF: Temporal-Reference Complementary Forecasting for Irregular Multivariate Time Series](https://arxiv.org/abs/2611.26351) | TRCF框架通过将紧凑的通道独立预测器与互补的通道依赖分支相结合，利用ANCHOR和BRIDGE两个模块在保留不规则观测历史的同时融合跨变量信息，从而改进不规则多变量时间序列的预测。 |
| [^38] | [Perfumer: A Large Language Model-based Agentic Framework Accelerating Structural Relaxation and Electronic Band Prediction for First-Principle Calculation](https://arxiv.org/abs/2611.26349) | Perfumer是一个基于大语言模型的智能体框架，通过结构预弛豫使第一性原理计算实现2.75倍加速，其能带预测模型可快速准确地预测电子能带（中位误差0.157 eV，每材料仅约97毫秒）。 |
| [^39] | [OceanCoder: A Verification-Guided Large Language Model for Automated Simulation Modelling in Ocean Engineering](https://arxiv.org/abs/2611.26333) | OceanCoder 通过可扩展数据流水线和三层验证框架为 GRPO 提供分级奖励，构建了一个能自动生成满足工程要求仿真模型的海洋工程专用大语言模型。 |
| [^40] | [Conformal Selection for Hierarchical Streaming Data](https://arxiv.org/abs/2611.26328) | 该论文提出了面向分层数据流的无分布在线保形选择方法OHS及其带反馈变体OHSF，通过构建组间和组内的基于秩的e值并结合在线e-BH程序，将在线保形选择框架扩展到超越独立同分布/可交换数据假设的分层结构场景。 |
| [^41] | [Predicting Surface Displacement Fields of Laser Powder Bed Fusion Parts Using STL-Derived Graphs and Part-Level Descriptors](https://arxiv.org/abs/2611.26323) | 该研究提出一种双分支图注意力网络，直接从STL衍生图结合零件级描述符预测激光粉末床熔融零件的三维表面位移场，R²达0.946、平均绝对误差仅0.0178毫米，为替代计算昂贵的热力学模拟提供了高效的零件变形预测方案。 |
| [^42] | [A Control-Theoretic Approximation to Predictive Coding Dynamics](https://arxiv.org/abs/2611.26319) | 该论文证明了深度反馈控制（DFC）是预测编码（PC）动力学在不动点附近的局部近似（称为 Approx PC），在浅层线性网络中二者完全等价，且通过重新线性化反馈控制器和分步逼近输出目标可以缓解由深度和非线性带来的偏差，从而厘清了两种生物学可信局部学习框架之间的关系。 |
| [^43] | [Scalable fault vulnerability analysis for ReRAM-based process-in-memory CNN accelerators](https://arxiv.org/abs/2611.26307) | 该论文提出了一种可扩展的故障脆弱性分析方法，用于高效评估基于ReRAM的存内处理CNN加速器中永久性故障对推理精度的影响，克服了传统故障注入方法耗时且计算成本过高的缺点。 |
| [^44] | [SKILL.PNG: Learning Agent Skills in Pixel Space](https://arxiv.org/abs/2611.26296) | 提出Skill.png方法，通过梯度下降优化图像像素来学习可跨VLM通用迁移的智能体技能，在大多数基准设置中超越最先进的文本技能，且token开销极低。 |
| [^45] | [Who Moved My $Z$ Boson? Revealing Resonance Features Inside a Neural Network](https://arxiv.org/abs/2611.26291) | 该研究在基于真实LHC数据训练的生成式对撞机物理模型ShellFlow中，通过梯度与模式识别方法发现了代表 $Z$ 玻色子等共振的内部可解释特征，并证明可以操控这些特征来改变 $Z$ 玻色子的质量和峰宽而不影响不变质量谱的其余部分。 |
| [^46] | [A Statistical Framework for Training Data Auditing of Large Language Models](https://arxiv.org/abs/2611.26264) | 本文提出了一个统一的统计框架，用于大语言模型训练数据审计，涵盖文档级成员推断、目标数据集全局审计和可疑子集识别，能够有效聚合文档内局部成员信号并进行整体评估。 |
| [^47] | [The Stability Gap in Continual Reinforcement Learning](https://arxiv.org/abs/2611.26252) | 本文提出了一个在保持环境动态和奖励不变的前提下仅改变观测表示的简单持续强化学习基准，并首次揭示了其中的“稳定性差距”现象——即使智能体可以完全访问过去任务，新任务的引入仍会导致旧任务性能大幅下降后再部分或完全恢复。 |
| [^48] | [TwinLab-Bench: Evaluating AI Scientists in Agentic Twins of Real Labs](https://arxiv.org/abs/2611.26248) | 该论文提出TwinLab-Bench基准，利用来自真实物理科学、生命科学和工程实验室的52个任务，首次系统评估AI科学家自主设计实验的能力，其结果通过与真实实验室隐藏记录对比来评分。 |
| [^49] | [Physics-Informed Feature Representations and Pretrained Tabular Foundation Models for Generalizable Melt Pool Geometry Prediction](https://arxiv.org/abs/2611.26235) | 该研究提出用归一化焓和佩克莱数两个物理信息无量纲特征替代原始工艺参数，并结合预训练表格基础模型进行基准测试，以提升LPBF熔池几何预测的跨材料泛化能力与数据效率。 |
| [^50] | [Time Series Structure Learning with Local Causal Parent Discovery](https://arxiv.org/abs/2611.26232) | 该论文提出ITPD——一种可靠且完备的基于约束的因果结构学习方法，通过局部到全局的因果父节点发现来学习具有因果非平稳性的多变量时间序列结构，并借助时间结构先验与图划分将高阶条件独立性检验替换为低阶检验，从而降低计算与样本复杂度。 |
| [^51] | [Attention Head Specialization: Phase Transitions in Sample Complexity](https://arxiv.org/abs/2611.26228) | 该论文证明注意力头特化是样本复杂度中真正的相变，在可求解的多头注意力模型中表现为测试误差的非解析性，且随数据量增加可通过级联相变逐步恢复更弱的潜在特征，形成统计意义上的简单性偏置。 |
| [^52] | [Learn, Freeze, and Transfer an Activation Function Across Architectures](https://arxiv.org/abs/2611.26201) | 该论文提出“学习-冻结-迁移”协议和五参数傅里叶级数激活函数 FAct，证明从源网络学得的激活函数可以被冻结并跨架构迁移到从零训练的网络中，成为权重迁移之外的一种互补性知识迁移方式。 |
| [^53] | [Backbone Matters: Benchmarking Encoder-Decoder Networks for Semantic Segmentation of Driving Scenes on BDD100K](https://arxiv.org/abs/2611.26191) | 该论文在BDD100K数据集上对52种编码器-解码器组合进行了系统性基准测试，发现EfficientNet-B4主干网络搭配FPN解码器在驾驶场景语义分割中表现最佳（62.6% mIoU），并且主干网络的效果强烈依赖于所搭配的解码器。 |
| [^54] | [Invertible Transport Between Sample-Only Distributions via Projected Characteristic Functions](https://arxiv.org/abs/2611.26187) | 该论文提出了一种免密度框架PCF（投影特征函数匹配），通过投影经验特征函数学习仅样本分布之间的可逆传输映射，无需密度估计、对抗训练或配对观测，并通过可选的传输正则化器选择低位移变换。 |
| [^55] | [Reward misalignment as an opportunity for materials discovery](https://arxiv.org/abs/2611.26186) | 该研究提出，生成式AI材料发现中奖励函数与专家判断的错位并非失败，而是发现被奖励遗漏的新材料（如具有部分平坦能带的化合物）的机遇。 |
| [^56] | [Thermodynamically Consistent Neural Networks for a Universal Fluid Gibbs Free Energy Equation of State](https://arxiv.org/abs/2611.26183) | 本研究以吉布斯自由能为出发点，提出基于热力学一致神经网络的新型流体状态方程，为流体状态提供统一的机器学习表示，其气液相密度预测精度优于传统体积平移立方型状态方程和现有机器学习模型。 |
| [^57] | [Budget Internalization: Fixed-Budget RLVR Trades Test-Time Scaling for In-Budget Accuracy](https://arxiv.org/abs/2611.26172) | 固定预算RLVR训练会将生成长度上限内化为模型的习得约束，用测试时扩展能力换取了预算内的准确率提升。 |
| [^58] | [NariQ: Sub-3-Bit Quantization for Long-Horizon Agentic Tasks](https://arxiv.org/abs/2611.26157) | NariQ是一种格栅编码后训练量化方法，通过利用BF16模型在智能体会话中生成的token进行校准，首次实现了每权重2.5-3比特量化在长程智能体任务上与全精度模型持平的性能。 |
| [^59] | [Loss-Guided Tsetlin Machines: Making the Learning Objective Explicit](https://arxiv.org/abs/2611.26114) | 本文提出Loss-Guided Tsetlin机器，用显式可替换的损失函数（CE、SCE、ASL）取代其固有的隐式间隔更新规则，在不改变Tsetlin自动机与子句逻辑的前提下，在六个分类数据集中的五个上匹配或提升了准确率与F1，尤其在不平衡多标签任务上将宏平均F1最多提升9个百分点。 |
| [^60] | [Physics-Informed Deep Learning for Shear Wave Speed Estimation in MR Elastography](https://arxiv.org/abs/2611.26111) | 提出了一种端到端可训练的物理信息深度学习框架，将欠采样k空间数据的图像重建与MRE相位梯度反演相结合，可直接从测量数据稳健地估计脑组织剪切波速度。 |
| [^61] | [How Network Topology and Data Heterogeneity Shape Decentralized Learning: A Spectral Analysis](https://arxiv.org/abs/2611.26104) | 该论文提出了一个谱分析框架，揭示去中心化学习中的两个互补机制——主导特征模态决定共识轨迹与节点影响力（平稳影响力正比于节点度加一），非主导特征模态控制分歧的传播与衰减——并推导出模型分歧与谱混合及梯度异质性之间关系的界。 |
| [^62] | [Revealing Pathway Interactions in Network Computations with Complete Path Contribution](https://arxiv.org/abs/2611.26101) | 该论文提出“完全路径贡献”方法，将网络输出分解为每条路径在数学上完整的贡献，从而能够直接分析从输入到输出的完整因果路径之间的高阶相互作用。 |
| [^63] | [From Sessions to Serving: Integrating LLM-Based Embeddings into Retrieval for Production Content Recommendation](https://arxiv.org/abs/2611.26096) | 本文提出LSER生产级内容推荐系统，将预训练大语言模型生成的会话级和历史级用户嵌入作为辅助特征集成到检索阶段用户塔中，并通过近线异步服务路径保持嵌入实时更新。 |
| [^64] | [Approximate Adjoint Models for Memory and Compute-Efficient Online Learning of Hybrid Models](https://arxiv.org/abs/2611.26095) | 该论文提出一类近似伴随模型方法，在整个轨迹上保留完整的伴随递推关系而仅近似其中的回拉算子，从而实现混合模型的内存与计算高效在线学习，避免了截断BPTT带来的梯度偏差和对长程依赖的忽略。 |
| [^65] | [Phantom of the Masquerade: Graph Neural Network-Based Detection of Masquerade Attacks in Controller Area Networks](https://arxiv.org/abs/2611.26094) | 本文提出PhantoMasq框架，利用图神经网络将CAN流量建模为融合信号统计特征的消息序列图，联合学习结构依赖与时间特性，从而有效检测控制器局域网中难以识别的伪装攻击。 |
| [^66] | [RL Can Only Reinforce What It Sees Succeed](https://arxiv.org/abs/2611.26084) | 在先验与正确答案均已精确已知的谜题环境中，该研究发现结果奖励强化学习只能强化训练中实际观察到成功的动作——当正确动作罕见时，训练能否“起飞”可由训练前即可计算的一个统计量（动作获胜的尝试组期望数量，约35组对应五成概率）准确预测，且该规律跨多种实验设置普遍成立。 |
| [^67] | [Learning to Compare Neural Representations from Parameterized Computation Graphs](https://arxiv.org/abs/2611.26081) | 提出了一种基于图元网络的框架，可直接从模型的参数化计算图中学习预测内部表示相似性，无需实际运行模型即可实现层检索与模型拼接组件选择。 |
| [^68] | [Understanding MoE Decisions Beyond Gradients](https://arxiv.org/abs/2611.26075) | 该论文提出一种利用MoE模型自身的专家输出和选择规则来估计专家替换引发的级联切换效应的预测器，相比传统梯度方法将答案得分变化的预测误差在Qwen3.6上降低60%、在Gemma4上降低47%。 |
| [^69] | [Iterative Scarcity-Guided Exploration: Bootstrapping Generative Auto-bidding from Narrow Support](https://arxiv.org/abs/2611.26060) | 该论文从理论上揭示了生成式自动竞价在狭窄数据支撑下停滞的根因是信噪比崩溃，并提出迭代稀缺性引导探索（ISGE）方法，通过在回报信号崩溃时让稀缺性接管作为探索信号以提升信噪比，从而突破次优停滞实现自举式改进。 |
| [^70] | [Complete-SVD: A Functional Norm Approach to Low-Rank LLM Compression](https://arxiv.org/abs/2611.26059) | 提出Complete-SVD方法，通过最小化原始与压缩模型logit之间的等权L2距离，将大语言模型低秩压缩目标化简为具有闭式解的加权SVD，并利用随机探测以与词表大小和序列长度无关的计算成本无偏估计难以求解的输出侧因子。 |
| [^71] | [SlotFlow-GB: A Neural Framework for Trans-Dimensional Inference of Overlapping Galactic Binaries](https://arxiv.org/abs/2611.26057) | SlotFlow-GB是一种神经框架，通过学习跨维度后验的近似，可在单次评估中同时推断强重叠银河系双星的数量及其物理参数，无需预先指定源的数目。 |
| [^72] | [When Do Rankings Beat Signs? Gradient Direction Recovery under Even Nuisance](https://arxiv.org/abs/2611.26055) | 该论文证明在偶函数干扰下，跨对排序策略Reverse仅需Õ(d+√(d/ε))次评估、单轮反馈即可将梯度方向恢复至角精度ε，达到理论下界且优于仅使用对内符号的方法。 |
| [^73] | [Reference-Free Humanoid Skateboarding via Termination-Shaped Reinforcement Learning](https://arxiv.org/abs/2611.26054) | 提出了一种无需人类演示或解析模型的深度强化学习框架，通过任务奖励与终止塑形相结合，使人形机器人（如K1）以无模型方式学会滑板这一物体介导的复杂运动任务。 |
| [^74] | [Fixing the Twin, Not the Detector: Sim-to-Real Roadside LiDAR Detection from Digital-Twin Data](https://arxiv.org/abs/2611.26035) | 该论文的核心创新在于修正数字孪生数据与真实场景之间的差异（对齐坐标系、点密度和标注约定，补全缺失的行人与骑行者并以真实尺寸和点密度重建目标），而非改进检测器本身，从而使仅用合成数据在单块8GB笔记本GPU上训练的CenterPoint模型在UrbanTwin Sim2Real激光雷达挑战赛LUMPI赛道夺得第一名。 |
| [^75] | [The Delayed Societal Cost of Unfairness](https://arxiv.org/abs/2611.26030) | 该论文揭示了不公平性的延迟社会代价：在部署前数据上看似帕累托最优的不公平决策政策，在其随时间诱导出的人群上进行评估时，可能被更公平的政策超越，从而在不损害任何一方的情况下同时降低社会成本。 |
| [^76] | [FrontierReplay: Split-Free Exact Prefix Caching for Hybrid Language Models](https://arxiv.org/abs/2611.26020) | FrontierReplay 通过在可预测的结构性边界处生成精确检查点并保持预填充前向传播不中断，为混合语言模型实现了无分割的精确前缀缓存，使状态构建成本比完整重计算降低 6-37 倍。 |
| [^77] | [Does the Workspace Do the Work? A Causal Protocol for Verifying Global-Workspace Computation in a Language Model](https://arxiv.org/abs/2611.26018) | 本文提出一种因果测试协议，通过从其他输入移植消息来判定语言模型中的全局工作空间究竟是由谁执行计算的，可区分出忽略广播、纯转发广播、以及将广播与自身知识重组这三种不同情况，并在训练后的工作空间模型中同时发现了这三种结果。 |
| [^78] | [Reinforcing Few-Step Diffusion Models via Shadow Posterior Matching](https://arxiv.org/abs/2611.25998) | 提出影子后验匹配（SPM）方法，通过将生成样本重新加噪并从影子后验获取奖励信用，克服了用黑盒奖励强化少步扩散模型时采样器速度失配与教师锚定缺失两大结构性难题。 |
| [^79] | [Policy Improvement-Guided Replay for Continual Reinforcement Learning](https://arxiv.org/abs/2611.25989) | 该论文提出对比似然重放方法（CoLiR），基于策略改进原则和性能差分引理，在视觉语言模型的持续强化学习中选择性重放历史提示，从而在获得新能力的同时保留已取得的性能提升。 |
| [^80] | [Seeing Beyond the Obvious: Complementary Semantic Guidance for Salient Single-Positive Multi-Label Learning](https://arxiv.org/abs/2611.25986) | 提出CoST框架，利用图像-文本语义对应从未知类别中挖掘互补候选样本并构建覆盖原型，弥补显著原型对隐藏正样本视觉多样性捕捉不足的问题，提升显著单正多标签学习的监督效果。 |
| [^81] | [The Depth of a Prediction: Where Language Models Resolve a Word Predicts Human Reading Time](https://arxiv.org/abs/2611.25970) | 论文提出“解析深度”这一新指标，用以衡量语言模型在哪一层完成对一个词的预测，并证明模型解析越晚的词及其后续词会被人类读得更慢，且解析深度能在最终层意外度和熵之外显著提升对人类阅读时间的预测。 |
| [^82] | [Power Setpoint Tracking for Large-Scale EV Charging: Transformer-Constrained Charging Policy Learning with Self-Tuning Constraint Prices](https://arxiv.org/abs/2611.25965) | 该论文提出TC-CPL方法，通过比例-积分自整定约束价格、热启动残差策略和变压器感知编码器，在大规模EV充电功率设定点跟踪中实现近乎零变压器过载并显著降低跟踪误差，且无需观测车辆离开时间或荷电状态。 |
| [^83] | [Gradient Boosting with Kolmogorov-Arnold Units for Interpretable Regression beyond Pairwise Interactions](https://arxiv.org/abs/2611.25942) | MJBoost以带惩罚样条的Kolmogorov-Arnold单元作为弱学习器进行梯度提升，在保持精确函数型ANOVA分解可解释性的同时，捕捉超越成对交互作用的高阶交互效应。 |
| [^84] | [DTS: Delta-Based Tile Scoring for Efficient Long-Context Sparse Attention](https://arxiv.org/abs/2611.25938) | 提出了一种无需训练、内容自适应的瓦片评分方法 DTS，它根据局部向量变化选择查询和键锚点来高效评分稀疏注意力的块，并在冗余区域少选、异构区域多选锚点以兼顾效率与精度，从而在几乎不损失模型长上下文性能的前提下显著降低长上下文预填充的计算开销。 |
| [^85] | [Hellinger Decision Trees for Density Ratio Estimation: Finite-Sample Guarantees](https://arxiv.org/abs/2611.25936) | 提出了一种通过牛顿提升在轴对齐决策树集成上最小化平衡Hellinger风险来直接估计对数密度比的方法，并建立了有限样本保证，其收敛速率在对数因子范围内达到极小极大最优的L²速率。 |
| [^86] | [A Lightweight Hybrid Approach for Milling Force Reconstruction from Internal Machine Signals](https://arxiv.org/abs/2611.25921) | 本文提出一种轻量化混合方法，利用机床内部信号和加工元数据，通过监督学习模型预测平均铣削力，从而在无需额外传感器的既有机床上实现基于力的铣削过程监测。 |
| [^87] | [LIFT: Latent Flow Matching for Probabilistic Cell Fate Dynamics with Single Cell Snapshots](https://arxiv.org/abs/2611.25917) | LIFT是一个基于潜空间流匹配的概率性细胞命运预测框架，通过变分推断编码细胞状态的一对多转变关系，结合Mean Flow解码器实现单次评估即可高效采样，能够从谱系追踪数据中学习细胞条件分化结局并进行大规模蒙特卡洛采样预测。 |
| [^88] | [Verifiable Logic in Medical LLMs: A Critical Review of Agentic Clinical Reasoning Optimization and Neuro-Symbolic Grounding](https://arxiv.org/abs/2611.25898) | 这篇综述指出临床大语言模型的最终答案正确并不代表推理有效，并系统评述了幻觉成因、智能体幻觉级联、过程奖励建模与神经符号接地四种范式，其中带符号否决层的系统能在其编码知识范围内提供可验证的安全保证。 |
| [^89] | [The Price of an Explanation: What Demanding Verifiable Artifacts Costs, and What It Buys Back](https://arxiv.org/abs/2611.25896) | 该论文在ARC-AGI-2上量化了要求LLM生成可执行可验证产物（程序）的代价（得分从77%降至57%），并指出真正的瓶颈在于让解释“泛化”而非“找到”解释——因为多数仅由只答组解出的任务中，程序组其实已拥有能复现全部演示、却仍在测试上失败的程序。 |
| [^90] | [One Schedule Does Not Fit All: Adaptive Inference Schedules for Diffusion Language Models](https://arxiv.org/abs/2611.25895) | 本文提出 TCDA，一种具有显式误差保证的自适应部署规则，可根据任务、模型和采样器为扩散语言模型选择满足质量要求的最低成本推理调度，将去噪器评估次数最多减少 3.4 倍。 |
| [^91] | [Parameter-Efficient Adaptation of Self-supervised Vision Backbones Survives Distribution Shift in Radar Micro-Doppler HAR](https://arxiv.org/abs/2611.25874) | 本文首次将自监督视觉基础模型DINOv3与PiSSA参数高效微调相结合应用于雷达微多普勒人体活动识别，证明PEFT是唯一能同时避免跨数据集负迁移和跨方向欠适应的适配策略，其Small变体以4.5倍更少的总参数量即可达到先前最先进方法的同分布精度。 |
| [^92] | [X Hedging: towards an eXplainable artificial intelligence option Hedging framework](https://arxiv.org/abs/2611.25870) | 本文提出X Hedging框架，用LightGBM树集成模型替代神经网络进行期权对冲策略学习，在保持对冲性能的同时实现了计算高效与更好的可解释性，弥补了Deep Hedging的不足。 |
| [^93] | [Full Rank and Stable Sampling of $\tanh$ Networks](https://arxiv.org/abs/2611.25867) | 本文刻画了 tanh 网络中实现映射微分为满秩的参数集（与正则参数一致），并证明几乎任意 2P+1 个采样点即可保证一致范数 Lipschitz 稳定采样、参数可识别性（允许隐藏神经元置换与符号变化），以及投影梯度下降与投影 Gauss–Newton 方法的局部收敛。 |
| [^94] | [Interpretable Prototype-Conditioned Adaptation for Weather Forecasting](https://arxiv.org/abs/2611.25848) | 该论文提出ProtoCast方法，通过组特定的原型库，将每组的历史特征与时间变化和跨变量注意力提供的预报上下文相结合，进行演化感知的原型匹配，从而在适应不同地点和变量组的同时实现可解释的天气预报。 |
| [^95] | [Rank2Replace: VLM-Ranked Slot Replacement for Out-of-Distribution Generalization](https://arxiv.org/abs/2611.25834) | 该论文提出Rank2Replace方法，利用预训练视觉-语言模型按任务相关性对对象槽位进行排序并执行同排名槽位替换增强，在无需环境标签的情况下将任务相关证据知识迁移到小型预测器，显著提升分布外泛化性能。 |
| [^96] | [Beyond Alignment: Whitening Geometry, Profile Synthesis, and Heterogeneous Ensembles for Cross-Environment WiFi Action Recognition](https://arxiv.org/abs/2611.25831) | 该论文发现跨环境WiFi动作识别的性能崩塌根源在于编码器对新环境的外推失败而非域不对齐，证明对齐类方法反而有害，并提出结合白化几何、轮廓合成与异构集成的Dec-H方法作为有效解决途径。 |
| [^97] | [A Sparse Low-Rank Biclique Decomposition for Graphs](https://arxiv.org/abs/2611.25826) | 提出符号双团分解（SBD），一种面向整数加权图的稀疏低秩分解方法，能够在保持图的结构稀疏性、离散性和可解释性的同时，有效恢复噪声图中的块结构并支持高效稀疏矩阵运算。 |
| [^98] | [Limits of local curvature for constructing search subspaces in segmentation models](https://arxiv.org/abs/2611.25822) | 该研究发现基于局部曲率构造的搜索子空间表征梯度更新场变化的能力随搜索半径增大而衰减，但在初始梯度中加入广义高斯-牛顿方向可显著降低语义分割模型中的表征相对误差。 |
| [^99] | [Anchor and Check: Deterministic Cross-Prompt, Cross-Model Reconciliation for Auditable LLM Decisions](https://arxiv.org/abs/2611.25818) | 该论文提出“锚定与核验”确定性调和契约——让目标模型的贪心答案接受跨提示或跨模型家族的独立贪心核验、仅一致时才执行，以实现可审计的LLM自主决策，并发现汇集跨家族样本可使语义熵的AUROC提升1.7至4.0个百分点，而同模型温度核验是最弱设计。 |
| [^100] | [How Reliable Are Explanations of Language Models That Detect Insider Threats?](https://arxiv.org/abs/2611.25796) | 本文提出“倾覆半径”这一核心贡献，通过一阶预测度量特征移除的相似对照变化何时会颠覆检测器的结论，从而审计内部威胁检测语言模型解释的可靠性。 |
| [^101] | [RouteVerify: Routing Interventions for Step-Level Verification in Mixture-of-Experts Models](https://arxiv.org/abs/2611.25780) | 该论文提出 RouteVerify 受控干预框架，通过在混合专家模型的九个检查点上替换路由器 logits、MoE 输出和残差状态等内部激活，系统检验路由状态与逐步验证判断之间的因果关系。 |
| [^102] | [Driver Gaze Target Identification via Gaze Estimation and Structured Driving Scene Understanding](https://arxiv.org/abs/2611.25771) | 该论文提出了一种模块化框架，通过融合MAE预训练视觉Transformer视线估计、全景场景理解、个性化校准和熵驱动的目标推断，从车内驾驶员图像与前视道路图像中实现目标级的驾驶员注视目标识别。 |
| [^103] | [Disentangling Monitor Hardware Artifacts and Smoothing Lag for Shift-Robust Intraoperative EEG and Hemodynamic Risk Calibration](https://arxiv.org/abs/2611.25765) | 针对商用术中监护仪的硬件伪影（阻抗脉冲、平滑延迟、直流漂移、袖带保持）干扰风险预测的问题，本文提出因果信号调理与分布偏移鲁棒的保形校准框架，在5,439例麻醉手术中实现了更可靠的术中脑电与血流动力学风险校准。 |
| [^104] | [Closing Performance Gap for Diffusion Language Models under Low Inference Budgets](https://arxiv.org/abs/2611.25763) | 提出了无需训练的 CaRR 框架，通过 token 承诺重掩码、更新上下文回填和基于轨迹的恢复机制，在低推理预算下提升可编辑扩散语言模型的生成质量，从而缩小其性能差距。 |
| [^105] | [AdaptReg: Skip-level FiLM Adapter for Task-Specific Adaptation of a Frozen Registration Foundation Model](https://arxiv.org/abs/2611.25761) | AdaptReg通过在冻结的配准基础模型（uniGradICON）解码器跳跃连接上引入由TotalSegmentator解剖语义特征驱动的FiLM式轻量适配器，并以病灶体积保持损失进行监督，实现了面向前列腺癌纵向PSMA-PET/CT配准的任务特定适配，在精准对齐的同时避免MTV/TLG测量被配准伪影污染。 |
| [^106] | [Beyond Error Mitigation: Using Boosting for Noisy Quantum Machine Learning](https://arxiv.org/abs/2611.25752) | 该论文提出利用监督学习中带标签样本提供的问题特定信息，将Boosting提升方法应用于含噪声参数化量子电路，使二分类任务只需依赖分类器输出的符号和间隔而非精确重建无噪声期望值，从而超越传统量子误差缓解方法的巨大采样开销限制。 |
| [^107] | [Goal-Guided Reinforcement Learning via Fine-Tuning Vision-Language Models](https://arxiv.org/abs/2611.25742) | 该论文构建了CCGP-2M多模态目标偏好数据集，并通过KLTDPO微调视觉-语言模型，使其在在线目标引导强化学习框架中为稀疏奖励环境生成可靠且符合环境特点的高层探索目标。 |
| [^108] | [PosteriorBench: Evaluating Behavior Posterior Sampling in Language Models](https://arxiv.org/abs/2611.25735) | 该论文提出了 PosteriorBench 基准，用于从计算成本和与真实后验的 KL 散度两个维度评估语言模型的行为后验采样算法，并发现现有方法均无法同时兼顾高效性与保真度。 |
| [^109] | [Can LLMs Resolve Entity Conflicts in Domain Specific NLU Pipelines?](https://arxiv.org/abs/2611.25733) | 本研究通过比较基于规则的冲突解决方法与两种基于大语言模型的方法（多轮提示和LoRA微调的Llama3.1-8B-Instruct），验证了大语言模型在解决领域特定NLU流水线中实体冲突问题上的可行性。 |
| [^110] | [CGM: Chaos-Guided Graph Attention and Expert Routing for Autism Stereotypy Detection](https://arxiv.org/abs/2611.25730) | 该论文提出CGM模型，将逐关节、逐帧的运动不规律性（样本熵）张量直接融入图注意力计算和专家路由机制，使模型聚焦于运动不可预测的关节，从而更准确地区分自闭症刻板行为与正常运动。 |
| [^111] | [Safeguarded Learning-Assisted Nonlinear Conjugate Gradient Methods](https://arxiv.org/abs/2611.25707) | 该论文提出了一种带安全保障的学习辅助非线性共轭梯度方法框架，通过两个在线极限学习机模型在重启恢复、停滞检测和线搜索失败时辅助经典优化过程，且神经预测仅在满足下降性、尺度等确定性数值安全条件时才被采纳。 |
| [^112] | [From Geographic Memory to Spatial Generalization in Large Language Models](https://arxiv.org/abs/2611.25703) | 该论文提出将大语言模型中的地理知识组织为潜在空间场，通过连续空间编码器将位置表示作为空间标记注入模型，从而实现从仅能回忆训练中见过的“地理空间记忆”向能预测未见区域的“空间泛化”能力的跨越。 |
| [^113] | [Measurable Does Not Imply Transportable: A Prospective Audit of Model Explanations in Multi-Year Field Trials](https://arxiv.org/abs/2611.25692) | 该论文形式化了“前瞻性解释可迁移性”这一新概念，并通过多年玉米田间试验证明：可靠的测量并不保证解释能在真正未来的情境中复现，这一属性独立于特征重要性和测量可靠性。 |
| [^114] | [Explainability of genotype embedding models through observed cohort paths](https://arxiv.org/abs/2611.25681) | 该研究揭示了积分梯度等路径归因方法无法直接用于基因型嵌入模型的原因——离散等位基因剂量的学习嵌入在分数基因型处无定义、无梯度——并提出在嵌入空间中沿观察到的队列路径进行归因，同时证明常用的逐变异估计量不满足完备性。 |
| [^115] | [Lightweight Low-Frequency Non-Intrusive Load Monitoring via Karhunen-Lo\`eve Feature Distillation and Temporal Convolutional Networks](https://arxiv.org/abs/2611.25678) | 提出了一种面向边缘设备的轻量级NILM框架，通过活动边界感知的数据增强、KLT特征提取和紧凑型TCN，在低频采样下以低计算成本实现准确的负荷分解。 |
| [^116] | [GPU-Accelerated Secure Inference for Drug--Target Interaction Prediction](https://arxiv.org/abs/2611.25664) | 该论文针对iDASH 2026竞赛提出了GPU上的两方安全推理方案，通过协同设计MPC友好的DTI模型SlimDTA与MPC协议，实现快速的隐私保护药物-靶点相互作用预测，避免候选药物知识产权泄露。 |
| [^117] | [SiliconGym: A Versatile Reinforcement Learning Environment for Agentic EDA](https://arxiv.org/abs/2611.25658) | 本文提出SiliconGym，首个面向硬件设计与验证的跨任务、跨工具链强化学习环境，通过解耦版本化工作区、预算化执行与验收门控评分，实现从模块级RTL任务到全层次SoC流程的可扩展训练，并借助流式调度将RL批次完成速度提升21.9%。 |
| [^118] | [Extending Rademacher Complexity-Based Generalization Theory to Domain Generalization](https://arxiv.org/abs/2611.25650) | 本文将基于Rademacher复杂度的经典泛化理论扩展到领域泛化设定，通过考虑由概率分布构成的领域空间的几何性质，弥补了现有无几何理论无法形式化刻画分布层面泛化与样本层面泛化区别的根本局限。 |
| [^119] | [AdsNet: A Graph Neural Network for Direct Adsorption Energy Prediction](https://arxiv.org/abs/2611.25646) | AdsNet 是一种等变图神经网络，通过仅基于吸附质原子及其近邻表面原子的局部图，在单次前向传播中直接快速且准确地预测吸附能，其计算复杂度与体系规模解耦，非常适合高通量催化剂筛选。 |
| [^120] | [Erasing the Wrong Thing: Conditional Attribute Leakage and Concept Erasure in Shared Foundation-Model Embeddings](https://arxiv.org/abs/2611.25640) | 论文证明了当任务标签与敏感属性相关时，全局概念擦除方法必然保留任务条件的线性属性泄露，并提出闭式方法TC-LEACE，在每个任务层级内强制实施线性防护以消除条件性敏感属性泄露。 |
| [^121] | [What One Sample Cannot See: Why Fewer Denoising Steps Can Score Higher on Physics Benchmarks](https://arxiv.org/abs/2611.25632) | 现有Physics-IQ基准通过将生成视频与单一参考录像比较，会奖励牺牲多样性、趋于“折中”的模糊预测，即使这些预测的分布更不准确；本文提出仅需两个样本的Kernel Physics-IQ基准来消除这一偏差，并证明更少的去噪步骤虽然提高了官方分数，却降低了真实的物理保真度。 |
| [^122] | [NIMIS: A Structure-Agnostic, Chemically Informed Slot-Attention Framework for Demixing 1D $^1$H-NMR Mixture Spectra](https://arxiv.org/abs/2611.25629) | NIMIS 是一种无需分子结构先验的深度学习框架，它将 J-耦合化学信息融入 Transformer 的多头注意力机制，并借助槽注意力将一维 ¹H-NMR 混合物光谱中的峰按其来源组分进行分组分离。 |
| [^123] | [Physical Laws without Physics Loss? Probing Learned Dynamics in Video World Models through Forced Response](https://arxiv.org/abs/2611.25626) | 本文通过摇晃悬挂物体支撑点的强迫响应实验，探究视频世界模型是否真正学习物理定律，发现仅用流匹配损失训练的模型不仅能复现线性响应（含共振），还能捕捉到训练数据中仅占0.74%摆动平方位移的非线性软化等高阶动力学。 |
| [^124] | [QUBO-based feature selection for MRI undersampling pattern design](https://arxiv.org/abs/2611.25625) | 该论文创新性地将MRI笛卡尔欠采样掩模设计转化为基于互信息与条件互信息的QUBO特征子集选择问题，利用模拟退火和量子退火求解，在15%至40%的所有测试采样率下均取得了更优的重建质量。 |
| [^125] | [Deviations from the NTK Regime in CNNs under Joint Depth-Width Scaling](https://arxiv.org/abs/2611.25624) | 该论文通过数值实验揭示了卷积神经网络中NTK离散度与训练时的核旋转均随深度与宽度之比（L/C）增长，且在临界初始化下离散度曲线随L/C近似坍缩，表明CNN存在偏离NTK机制的非平凡联合深度-宽度缩放区域。 |
| [^126] | [Hierarchical Spatiotemporal World Modeling with Content-Adaptive Spatial Addressing](https://arxiv.org/abs/2611.25610) | 提出受海马体记忆与网格细胞启发的分层时空世界模型HiSTFormer，通过局部时间RoPE、多周期空间RoPE和内容自适应寻址，实现有界紧凑记忆，并在MemoryMaze和HM3D上取得优异的长时程预测性能与泛化能力。 |
| [^127] | [PSTO: A PDE-structured transport operator with shared implicit updates for multiscale CDR systems](https://arxiv.org/abs/2611.25607) | PSTO通过将PDE结构嵌入神经架构并在所有输运变量间共享单一输运算子模块、结合松弛不动点迭代实现类隐式修正，仅需无反应数据训练即可在推理时无需重训练地泛化到完整的多尺度对流-扩散-反应系统，且不增加参数量。 |
| [^128] | [On the Informativeness of Structural Stability Bounds for GNNs: GINs versus GATs](https://arxiv.org/abs/2611.25603) | 本文推导了GIN和单头GATv1在确定性边扰动下的结构稳定性界，并与扰动无关的直径界对比，精确刻画了GIN稳定性界更紧的条件（在Cora上成立但在低度数的MUTAG上失效），同时发现GATv1的稳定性界从不更紧。 |
| [^129] | [Decision-Focused Reinforcement Learning for LLM Post-Training](https://arxiv.org/abs/2611.25597) | 该论文提出面向决策的强化学习（DFRL）框架，以下游决策负遗憾作为奖励对LLM进行后训练，使其预测直接最小化下游决策成本且无需下游问题凸性或可微性，并通过启发式热启动和奖励校正两种方法解决了奖励评估开销大的问题。 |
| [^130] | [Bradley-Terry Training Can Distort Outcome Probabilities of Initially Accurate Language World Models](https://arxiv.org/abs/2611.25594) | 本文证明Bradley-Terry目标函数在基于比较反馈的训练中无法教会语言世界模型真实的结果概率，反而会将原本准确的模型（危害误差仅2.4个百分点）中33-43个百分点的概率扭曲到实际不可能发生的结果上，且这种失真源于目标函数本身而非比较数据。 |
| [^131] | [From Data Alignment to Parameter Generation: Few-Shot Cross-City Traffic Prediction with Conditional Diffusion](https://arxiv.org/abs/2611.25592) | 该论文提出了基于条件扩散的参数生成框架（CDPG），通过学习源城市高性能预测器参数的条件分布来直接生成目标城市各节点的模型参数，从而突破了传统数据对齐方法在少样本跨城市流量预测中易受分布偏移、测量噪声和城市特有偏差影响的局限。 |
| [^132] | [Design and Evaluation of Epira: An Automated Interview System for Case Investigation and Contact Tracing](https://arxiv.org/abs/2611.25577) | Epira是一种将大语言模型对话与规则控制器相结合的协议控制自动化访谈系统，既能灵活应对受访者澄清需求，又能严格遵循流调协议，为多疾病病例调查和接触者追踪提供了有效的自动化解决方案。 |
| [^133] | [Reinforcement-Aligned Probabilistic Weather Forecasting with Discrete Tokens](https://arxiv.org/abs/2611.25570) | WeatherCraft通过离散天气词元将概率生成与强化学习对齐统一起来，将GPT式预训练—后训练范式引入全球中期天气预报，可高效生成集合预报，并利用GRPO在无需对评估器或耦合模型求导的情况下直接优化极端事件预警等下游任务奖励。 |
| [^134] | [Is a Description Worth a Thousand Demonstrations? Specificity and Reliability of Task Descriptions in In-Context Learning](https://arxiv.org/abs/2611.25560) | 本文从贝叶斯视角将上下文学习中的任务描述建模为信息先验，提出用有效样本量（ESS）来量化描述相对于示例的预测价值，证明了可靠描述的ESS随其特异性和任务维度增长，并通过实验表明元训练的Transformer在训练分布上接近贝叶斯最优的ESS水平。 |
| [^135] | [Dynamics of Protected State Evolution](https://arxiv.org/abs/2611.25541) | 本文为受保护状态演化建立了动力学框架，通过按执行顺序组合转移记录来区分净消耗相同但初始储备需求不同的协议，并利用加权Gram算子、商构造与协变三次型刻画线性响应椭球、有限预算响应集的不可比性以及奇异门处的顺序可行性。 |
| [^136] | [Towards Autoregressive Neural Network Parameter Generation](https://arxiv.org/abs/2611.25538) | 该论文提出指令引导的参数生成方法IGPG，将神经网络参数生成形式化为自回归建模问题，通过把预训练参数编码到离散域，使大语言模型的零样本泛化能力得以迁移到权重生成任务中。 |
| [^137] | [Differentially Private Score Estimation with Applications to Score-Based Generative Models](https://arxiv.org/abs/2611.25536) | 本文研究了差分隐私约束下的分数估计问题，针对具有局部 Hölder 连续分数的次高斯分布，提出了一种基于二次 B 样条和指数机制的采样式隐私分数估计器，并将其应用于隐私保护的基于分数的生成模型。 |
| [^138] | [Learning Generative Ranking Policies for Whole-Page Optimization](https://arxiv.org/abs/2611.25527) | 提出一种局部页面级训练目标，可精确拆分为逐位置的训练信号，使生成式排序策略在整页优化中动态生成更高分的页面，同时克服了现有方法中统一页面级信号和固定模仿目标的双重局限。 |
| [^139] | [When and why can integral constraints improve the accuracy of physics-informed neural networks?](https://arxiv.org/abs/2611.25515) | 该论文通过推导带与不带积分约束的PINN损失函数的稳定性估计，从数学上揭示了积分约束在特定参数范围内通过恢复标准损失对弱控制误差分量的一致控制来提升精度的机理与适用条件。 |
| [^140] | [Sequential Probability Assignment against Smoothed Adversaries with Unknown Base Measure](https://arxiv.org/abs/2611.25514) | 本文在基测度 μ 未知且无需 Lipschitz 条件的更一般平滑在线学习设定下，证明了针对自适应型与疏忽型平滑对手的序贯概率赋值遗憾上界，其中自适应界由经验 ℓ∞ Hellinger 熵控制。 |
| [^141] | [Joint Language Modeling for In-the-Wild Speech-Scene Generation with Distinct Token Representations](https://arxiv.org/abs/2611.25503) | 提出一种基于大语言模型的框架，利用声源感知的神经音频编解码器从真实场景混合音频中提取差异化对齐的语音与环境Token，并以交错方式联合生成两者，解决了语音与环境声建模混淆及依赖模拟混合数据的问题。 |
| [^142] | [WeightCode: Installing New Capabilities into Neural Networks via Weight-Space Programming](https://arxiv.org/abs/2611.25476) | 提出WeightCode框架，利用冻结语言模型编写可执行的权重程序，无需梯度训练即可直接修改神经网络权重以安装新能力，同时保留原有功能。 |
| [^143] | [A four-level executable validation protocol for evaluating LLM-generated topology specifications of physics-based HVAC digital twins](https://arxiv.org/abs/2611.25471) | 本研究提出一个包含静态有效性、模型构建、仿真收敛和行为等价性四个层级的可执行验证协议，用于评估LLM生成的HVAC数字孪生规范是否物理有效，并在八个LLM和十四种HVAC拓扑上进行了验证。 |
| [^144] | [Dimension-Free Buffer Bounds for Compressed-Momentum Error Feedback](https://arxiv.org/abs/2611.25463) | 该论文证明，通过将压缩动量重新注入压缩前的累加器，压缩动量误差反馈的缓冲误差界可完全摆脱维度与秩相关的收缩常数，且该结论适用于 top-k 选择与截断 SVD 等任意闭锥度量投影类压缩器。 |
| [^145] | [ActMem: From Latent Memory Reasoning to Grounded Robotic Action](https://arxiv.org/abs/2611.25461) | 提出ActMem框架，在单一视觉-语言模型中将记忆推理与感知锚定解耦，通过潜在记忆思维链条件化流匹配动作生成，从而提升长时程机器人操作性能。 |
| [^146] | [FamilySpace: From Emergent Chemical Semantics to Controllable Crystal Generation](https://arxiv.org/abs/2611.25452) | 该论文发现晶体VAE的原子级潜空间会按元素族自发地组织出分离区域（即“涌现的化学语义”），并据此提出FamilySpace框架，通过从特定元素族的潜空间区域采样来控制生成晶体的元素族组成。 |
| [^147] | [Inverse ECG Reconstruction via a Preconditioned Variational Quantum Linear Solver](https://arxiv.org/abs/2611.25450) | 本文提出了首个将量子线性求解器应用于逆心电重建的量子-经典混合框架，通过拉普拉斯正则化预编码方法降低系统维数并改善病态性，再利用相干变分量子线性求解器（VQLS）求解，并从理论上严格证明了该框架能保持精确解。 |
| [^148] | [Interference Bounds and Merge-Aware Training Dynamics for Composable Fine-Tuned Models](https://arxiv.org/abs/2611.25430) | 该论文将模型合并的可组合性形式化为任务向量集合的性质，证明合并模型的超额损失由跨任务曲率重叠与更新带符号对齐构成的干涉泛函控制，并提出一种曲率引导的正则化微调方法，使合并模型在满足可组合性判据的同时保持标准的非凸收敛速率。 |
| [^149] | [Instance-Optimal Moment Estimation in the Add-Remove Model of Differential Privacy](https://arxiv.org/abs/2611.25425) | 本文引入实例相关的误差度量并证明其严格下界，设计了在纯差分隐私加-删除模型下达到这些下界的矩估计机制，实现了比现有方法渐近更优的效用。 |
| [^150] | [Coordinate Basis Linear Bandits with Sublinear Regret over Low-Rank Clusters](https://arxiv.org/abs/2611.25424) | 该论文提出坐标基线性老虎机算法，通过聚类机制在每个聚类中维护精确正交基和低维坐标表示以保留完整协方差信息，在高维线性老虎机场景中同时实现了计算效率提升与鲁棒的次线性遗憾保证。 |
| [^151] | [From Sequence Uncertainty to Phenotype Hypotheses: Uncertainty-Conditioned Inference over Biomedical Knowledge Graphs](https://arxiv.org/abs/2611.25422) | 该论文提出一种不确定性条件化的路径能量框架，将基因组基础模型的序列预测及其校准不确定性转化为生物医学知识图谱上可解释、可追溯的表型假设分布，无需患者级表型监督。 |
| [^152] | [FG-GraphBench: Benchmarking and Enhancing Fine-Grained Visual Understanding with Multimodal Scene Graphs](https://arxiv.org/abs/2611.25408) | 该论文提出了细粒度多模态场景图基准 FG-GraphBench（涵盖细粒度视觉问答、图像描述与内容描述三个任务）及 FiGRA 方法，以提升多模态大语言模型利用场景图结构信息进行细粒度视觉理解的能力。 |
| [^153] | [Antom 3-in-1 Transformer: A Payment Foundation Model](https://arxiv.org/abs/2611.25403) | 本文提出了面向支付风险检测的支付基础模型 Antom 三合一 Transformer，通过融合序列、表格和图表示进行预训练，实现了跨业务迁移能力的大幅提升，并在线上实验中显著降低了支付损失率。 |
| [^154] | [Protein Functional Semantic Alignment via Representation Rectification](https://arxiv.org/abs/2611.25396) | 提出SAVR对比式后训练方法，利用Pfam家族同源性信号校正蛋白质语言模型的表示结构，无需任务标签即可显著提升下游功能预测性能。 |
| [^155] | [TuringSpace: Grounding Spatial Intelligence in Action with Interactive CAPTCHAs](https://arxiv.org/abs/2611.25387) | 提出了TuringSpace——一个包含125个需通过点击和拖动完成的交互式验证码合成网络环境，用于评估多模态大模型在实际行动中落实空间智能的能力，人类解决率达94.1%，而最强模型仍失败25.9%，暴露出当前模型在行动性空间推理上的显著差距。 |
| [^156] | [Drifting, Discriminating, and Directioning for Action Coordination in Higher-dimensional Action Space](https://arxiv.org/abs/2611.25385) | 本文提出一种基于漂移模型的轻量级策略方法，通过协同优化源分布、判别与定向的三步训练算法，有效解决强化学习中高维动作空间的动作协调问题。 |
| [^157] | [Causal Discovery with False Positive Error Control](https://arxiv.org/abs/2611.25376) | 提出m-保守PC算法，在线性高斯设定和极弱忠实性假设下实现了因果发现中错误发现率的有限样本控制，同时惩罚虚假邻接与错误边方向。 |
| [^158] | [Fundamentals for Binary Classification Under Label Noise](https://arxiv.org/abs/2611.25372) | 本章系统综述了标签噪声下二分类学习的基础理论，在 PAC 框架下形式化了随机分类噪声、Massart 噪声和恶意分类噪声三种模型，并阐明了经验风险最小化的适用条件以及样本复杂度与噪声率之间的关系。 |
| [^159] | [Class-Conditional Conformal Triage for ICAO Face-Image Compliance under Class Imbalance](https://arxiv.org/abs/2611.25342) | 该研究提出将标签条件化（Mondrian）保形预测作为ICAO人脸图像合规分析的分流层，在不重新训练模型的情况下解决了类别不平衡导致的违规判定覆盖不足问题，将各准则的违规覆盖率从边际方法的0.37提升至0.89-0.92。 |
| [^160] | [Now You See Me Now You Don't: A Framework To Detect and Reduce Proxy Capacity](https://arxiv.org/abs/2611.25327) | 该论文针对受监管领域（如贷款）中差别性对待的公平性问题，提出了一个可辩护的框架，用于大规模检测并降低机器学习模型中的代理变量能力。 |
| [^161] | [RE-CAM: Readout-Enhanced Content-Addressable Memory for Quantitative Similarity Search](https://arxiv.org/abs/2611.25326) | RE-CAM提出了一个与存储器技术和应用无关的读出增强框架，通过将感知时间戳作为关键设计变量，使CAM阵列在单次搜索中即可近似线性和非线性函数，将传统的二进制匹配读出升级为定量相似性搜索。 |
| [^162] | [Improving Imitation Policies for Multi-Phase Tasks through Residual Learning Allocation](https://arxiv.org/abs/2611.25320) | 该论文提出“完成增益”概念，将其分解为阶段到达概率与条件完成改进，据此指导残差强化学习在多阶段任务中纠正哪些动作组件以及如何在各阶段间分配有限的训练预算。 |
| [^163] | [When Should Deployed BCI Decoders Adapt? A Resource-Priced Ladder for Frozen EEG Trunks](https://arxiv.org/abs/2611.25315) | 该论文针对冻结的EEG编码器提出一个由情境感知门控与效用定价（准确率、拟合时间和标注成本）驱动的资源阶梯自适应控制层，通过闭式贝叶斯头部更新，证明已部署BCI解码器的自适应应当被门控并设定上限，而非始终开启或盲目微调。 |
| [^164] | [Prescribed Bayesian Efficiency under Lifetime Adjustment Costs](https://arxiv.org/abs/2611.25311) | 该论文刻画了在终身调整成本约束下，贝叶斯学习者在时间范围 T=2^J 处同时实现累积期望超额损失 O(G(J)) 与有限期望终身费用的可达性准则，即跨动作距离尺度上带前缀约束的跟踪误差配额分配的有限性。 |
| [^165] | [VoxForge: A Unified Foundation Model for Dense 3D Features Across MRI, CT and PET](https://arxiv.org/abs/2611.25309) | 提出VoxForge，一个融合3D架构、自蒸馏与跨模态课程学习的统一医学基础模型编码器，在完全冻结状态下即可在MRI和CT的分割、分类、配准任务上媲美通用与专用模型，并能泛化到训练中从未见过的PET模态。 |
| [^166] | [SplitReAlign: Partition-Aware Representation Realignment for Split Federated Unlearning](https://arxiv.org/abs/2611.25301) | SplitReAlign提出了一种分区感知的表征重对齐框架，通过共享语义锚点协调的端点判别子空间，在分割联邦遗忘中实现类级知识的选择性遗忘并保护保留效用，同时适配客户端与服务器的非对称更新约束。 |
| [^167] | [Federated Learning of Car Aerodynamics Surrogates with Spectral Optimizers](https://arxiv.org/abs/2611.25300) | 本研究在跨车身类型的联邦学习设置下比较了多种优化器，发现谱优化器（尤其是FedMuon和作者提出的FedMARS-M）能训练出性能与中心化训练相差2%以内的汽车空气动力学代理模型。 |
| [^168] | [Cross-Question Reliable Reinforcement Learning](https://arxiv.org/abs/2611.25292) | 本文提出跨问题可靠强化学习，指出现有逐题独立地将正确答案标为置信度1、错误答案标为0的训练方式在实践中不可实现且对非零风险容忍度下的选择性预测有害，主张跨问题地学习可靠的置信度估计以支持更好的弃权与转交决策。 |
| [^169] | [LLMs Are Already Confidence-Aware Zero-Shot Classifiers](https://arxiv.org/abs/2611.25289) | 研究通过系统评估表明，大语言模型作为零样本分类器已具备置信度感知能力，基于logit的白盒评分方法在判别分类置信度可靠性上显著优于黑盒方法。 |
| [^170] | [Intelligent Embedding Pruning for Large-Scale Recommendation Models](https://arxiv.org/abs/2611.25288) | 该论文提出利用特征重要性来指导训练中嵌入剪枝的各表剪枝比例（FI-ITEP），在三个大规模推荐模型上实现 20%–40% 的内存节省并保持或提升准确率，同时显著减少训练所需的 GPU 数量。 |
| [^171] | [DS3M: Deep structured state space models for nonlinear system identification on modern hardware](https://arxiv.org/abs/2611.25274) | 本文提出深度结构化状态空间模型DS3M，通过以并联和串联方式组合紧凑的非线性动力学组件，并利用Triton实现的自定义GPU内核降低顺序评估开销，显著提升了非线性系统辨识中神经状态空间模型的训练速度。 |
| [^172] | [Scone: Symmetry-aware Neural Combinatorial Optimization for Graph Coloring](https://arxiv.org/abs/2611.25273) | 本文提出Scone，一种基于颜色等变图神经网络的对称性感知神经组合优化方法，通过利用颜色置换对称性缩小状态空间并在必要时打破对称性，从而高效构建高质量的图着色方案。 |
| [^173] | [Learning to Verify: Signature-Guided Training for Adaptive Adversarial Detection](https://arxiv.org/abs/2611.25271) | 该论文提出一种双模型对抗检测方法：固定原始分类器并训练独立验证器，利用“激活签名”引导生成能同时误导分类器、挑战验证器且内部特征接近目标类别的高难度对抗样本进行训练，使验证器在面对自适应攻击（含未见过的攻击）时依然有效，且推理时无需激活签名。 |
| [^174] | [Process-Knowledge-Guided Latent Diffusion for Generative Design of Porous Architectures: A Case Study in Aerosol Jet Printing](https://arxiv.org/abs/2611.25264) | 该论文提出一种工艺知识引导的潜在扩散生成框架，将基于工艺模型的利用率评分与VQ-VAE潜在表示、拓扑U-Net和ControlNet分支相结合，使生成的多孔结构布局能够适配特定的气溶胶喷射打印工艺设置。 |
| [^175] | [When Objectives and Evaluators Disagree: Evaluating Transfer for Post-hoc Demographic Framing Mitigation](https://arxiv.org/abs/2611.25258) | 该研究将无法访问源模型时通过事后重写缓解大语言模型人口框架偏差视为迁移问题，发现总是针对说服偏差指数（PBI）较低信息的固定策略迁移不一致，而在两侧提出编辑但最多接受一个的自适应策略在两个评估器家族八项比较中的七项显著更优，且主要得益于编辑了更多对而非引导方向选择。 |
| [^176] | [Moving Generality from Model to Modeler: LLM Agents for PHM on a Deployment-Style Benchmark](https://arxiv.org/abs/2611.25252) | 该论文提出部署式基准 NextUnit-Bench，证明基于LLM的智能体在PHM任务上可超越AutoML及已发表方法的忠实复现，主张通用性应从单一模型转移到能为每个新任务构建模型的智能体身上。 |
| [^177] | [Stability Bands for Feature Attribution: The Core, the Nucleolus and the Bargaining Set of Explained-Variance Games](https://arxiv.org/abs/2611.25238) | 该论文提出用解释方差博弈的核心、核仁与谈判集为每个特征刻画“稳定性区间”，以证明归因分配不会被特征联盟合理质疑，并证明线性模型下该博弈恰为2-可加，多重共线性正是联盟效应的根源。 |
| [^178] | [Some Memories Must Fade: Persistence and Decay of User Memories in Model Parameters](https://arxiv.org/abs/2611.25235) | 该论文提出了一种参数化记忆管理系统，通过固定容量的LoRA适配器作为记忆基底，配合学习策略来管理知识的完整生命周期——保留稳定事实、强化重复观察、更新变化的信念，并让短暂信息自然消退。 |
| [^179] | [Statistical, Machine Learning and Deep Learning Methods for Missing Covariates: An Empirical Evaluation with Applications to Alzheimer's Disease Research](https://arxiv.org/abs/2611.25219) | 本研究对八种处理缺失协变量的统计、机器学习和深度学习方法进行了系统性实证评估，在阿尔茨海默病研究背景下比较了它们在不同缺失机制下的表现。 |
| [^180] | [Learning to Read Events for Exact State Tracking with Frozen Language Models](https://arxiv.org/abs/2611.25211) | 该论文提出在冻结语言模型上训练一个轻量级选择器来读取事件流中的每个状态转换，并通过外部整数寄存器按序组合更新，从而实现精确状态跟踪，理论上能够解决固定深度transformer在TC⁰ ≠ NC¹猜想下无法解决的任务。 |
| [^181] | [{\phi}TD: Distributional Reinforcement Learning using Characteristic Functions](https://arxiv.org/abs/2611.25208) | φTD通过在频域中用特征函数计算损失，统一了分布式强化学习中的分位数与类别两类方法，并能学习连续分布混合等此前无法捕捉的分布，从而带来性能提升。 |
| [^182] | [From Self-Play RL to Constrained Control: Distilling DiffMPC Students with DAgger](https://arxiv.org/abs/2611.25197) | 该方法通过保留PPO教师骨干并用强制执行显式约束的DiffMPC替换其动作头，再借助DAgger在教师自博弈仿真器中蒸馏学生模型，成功为已训练的自博弈RL驾驶策略添加约束，学生性能持平甚至超越教师，且蒸馏耗时不超过教师训练时间的30%。 |
| [^183] | [MADJEPA: Enhanced Multi-Stage Joint-Embedding Predictive Architecture for Medical Anomaly Detection](https://arxiv.org/abs/2611.25194) | 该论文提出了MADJEPA，一种基于增强型多阶段联合嵌入预测架构的医学图像异常检测器，通过I-JEPA训练、目标编码器蒸馏和伪异常分类的多阶段训练流程，仅使用健康患者数据即可学习正常性的强表征并实现鲁棒的异常检测。 |
| [^184] | [Blind Spots Travel, Knowledge Stays Home: What RL Inherits from an LLM Judge](https://arxiv.org/abs/2611.25191) | 通过814次强化学习实验，本文发现LLM评审员对单个答案的知识几乎不会传递给被训练的模型（每百题仅纠正约一题），但其对某类问题的“盲区”会迁移，导致模型在新问题上系统性地拒绝回答该类问题。 |
| [^185] | [PDE-State Conditioning for Learned Optimization Using Physics-Informed Neural Networks](https://arxiv.org/abs/2611.25185) | 该论文提出用11维PDE状态（刻画算子结构、残差几何与短时程更新响应）通过有界FiLM调制来条件化元学习优化器，使学习型PINN优化器首次显式利用PDE层面信息，并通过设计研究确定紧凑残差几何状态G2与有界FiLM为最佳设计组合。 |
| [^186] | [Before the Choice: An Evidential Framework for Studying Prospective Representations in Language Models](https://arxiv.org/abs/2611.25170) | 该论文提出“选择之前”这一证据导向框架，区分前瞻性内容、前瞻性计算与因果引导，用以检验语言模型是否内部表征未被选择的轨迹，并通过预注册探针实验发现低秩探针未达标且存在排除法重构的混淆问题。 |
| [^187] | [Game-Theoretic Data Allocation for Expert Specialization and Routing](https://arxiv.org/abs/2611.25167) | 该论文提出InflEns方法，用一个位置博弈同时决定LoRA专家混合的训练数据分配与推理路由，使份额最高的专家在约90%的提示上表现最优，从而让分工在训练前即可检验并用分配本身指导推理。 |
| [^188] | [Separating Interpretability and Reconstruction in Sparse Autoencoders](https://arxiv.org/abs/2611.25165) | 提出DualSAE模块化框架，通过可插拔的残差分支与去相关损失，将稀疏自编码器中的可解释性与重构保真度这两个目标分离，从而学习到语义更纯粹的特征。 |
| [^189] | [Bayesian $\kappa$-Explorer for Minimax Model Estimation in MDPs](https://arxiv.org/abs/2611.25160) | 该论文提出贝叶斯 κ-Explorer，通过在转移分布上放置狄利克雷先验并基于后验均值模型进行规划，以闭式探索信号简化了 MDP 中极小极大转移估计的主动探索，在免除置信集优化的同时仍保持 Õ(T^(-1/3)) 的收敛率。 |
| [^190] | [Rate-distortion Assessment for Hierarchical Content Summarization](https://arxiv.org/abs/2611.25151) | 该论文受香农率失真理论启发，将分层内容摘要形式化为信息编码问题，提出了一个可定量刻画压缩效率与信息损失之间权衡的评估框架，从而实现对不同生成模型在给定率预算条件下的科学基准测试。 |
| [^191] | [CAIMS: Scalable Cross-Array In-Memory Search](https://arxiv.org/abs/2611.25141) | CAIMS提出了一个与存储技术和应用无关的跨阵列存内搜索框架，通过准确高效地将局部匹配结果归约为全局top-k结果，解决了跨多个CAM阵列进行可扩展top-k相似度搜索的关键难题。 |
| [^192] | [Temporal Extrapolation of Elastic Seismic Wave Propagation in Layered Media Using Neural Operators](https://arxiv.org/abs/2611.25130) | 该论文提出将嵌入RK4时间积分器的DeepONet与FNO残差校正以预测-校正器方式耦合训练，实现了层状弹性介质中地震波速度场超出训练时间区间的自回归时间外推预测。 |
| [^193] | [Auditing Learned Dynamics in Weather Models](https://arxiv.org/abs/2611.25119) | 该论文利用稀疏自编码器对GraphCast天气模型的内部机制进行审计，发现消融缺乏传统物理语义的计算网格特征也会增加位势高度误差，且抑制与对流和低层涡度相关的特征会削弱热带气旋加深的预测，为AI天气预报模型在高保证场景下的可靠性审计与极端事件预报改进提供了新方法。 |
| [^194] | [Fast and Memory-Light Evaluation of Density-Based Clusterings with the DISCO Index in R](https://arxiv.org/abs/2611.25114) | 该论文提出了discoCVI R软件包，通过从互可达性最小生成树的单链接合并树中提取DISCO指标所需的全部信息，将内存需求从二次方降至线性，实现了对基于密度聚类（含噪声标签质量评估）的精确且高效的评估。 |
| [^195] | [Long-Range Video Presentation Attack Detection: A Three-Class Benchmark Across 1-100 m and Two Sensor Types](https://arxiv.org/abs/2611.25108) | 该论文提出了首个面向长距离场景的视频人脸呈现攻击检测基准，涵盖1-100米五种拍摄距离、室内外环境、两种传感器类型及真人/欺骗/道具三分类标签（37万余帧、261个身份），并对四种CNN架构进行微调与评测。 |
| [^196] | [HapBO: Harnessed $\alpha$-$\pi$ Bayesian Optimization via Expert Natural Language Priors](https://arxiv.org/abs/2611.25100) | 该论文提出HapBO方法，将从人类专家或大语言模型一次性引出的自然语言先验确定性地编译为不限于高斯族的可重用信念函数，从而提升贝叶斯优化的样本效率，并对可能误导的信息保持鲁棒。 |
| [^197] | [BuggyMLE: Scaling Machine Learning Engineering Difficulty with Bug Injection](https://arxiv.org/abs/2611.25088) | 该论文提出基于十个结构特征的代码复杂度指数（CCI）来有效预测编码基准的任务难度（R²=0.951），并据此构建了BuggyMLE框架，通过向可运行的代码库注入受控错误来规模化提升机器学习工程任务的难度。 |
| [^198] | [PriTabICL: Differentially Private Tabular In-Context Learning with Task-Aligned Cluster-Wise Synthesis](https://arxiv.org/abs/2611.25073) | 该论文提出PriTabICL，通过任务对齐的簇级差分隐私数据合成来生成隐私保护的表格上下文学习示例，克服了直接对整个数据集应用通用差分隐私合成时全局分布建模掩盖局部结构和有限隐私预算下广泛属性依赖降低预测准确性的两大局限。 |
| [^199] | [Empirical Variational Bayes Factorization for Longitudinal Matrix Data](https://arxiv.org/abs/2611.25065) | 提出了一种经验变分贝叶斯因子分解方法，能够高效处理具有时间相关性的高维纵向矩阵数据，自动确定所有超参数，并有效应对缺失数据和不规则时间点的挑战。 |
| [^200] | [Infer3D: Inverse Generative Modeling Enables Robust 3D Reconstruction](https://arxiv.org/abs/2611.25061) | 该论文提出Infer3D逆向生成建模框架，通过恢复生成模型的潜变量并结合冻结的3D提升网络约束优化空间，使单目3D重建在分布外场景条件下依然保持鲁棒性。 |
| [^201] | [Near-Quadratic Sample Compression for All VC Classes](https://arxiv.org/abs/2611.25011) | 本文通过构造大小为 O(d²log²d) 的样本压缩方案，证明了Floyd和Warmuth提出的弱样本压缩猜想，即每个VC维为 d 的二值概念类都存在多项式大小的样本压缩方案。 |
| [^202] | [On the Impact of Feature Reduction Techniques on Adversarial Attacks](https://arxiv.org/abs/2611.25004) | 本研究首次系统分析了特征降维技术（如减少维度）在APT数据集上遭受投毒、后门和通用对抗扰动攻击时，对支持向量机、决策树、随机森林等多种机器学习模型对抗鲁棒性的影响。 |
| [^203] | [A Model of Diverse Sampling from Language Models](https://arxiv.org/abs/2611.25003) | 该论文提出 PoolDPP 方法，将语言模型的多样性采样形式化为行列式点过程（DPP）下的子集采样，通过候选池上的 K-DPP 采样器实现了输出与目标分布间 KL 散度 O(1/N) 的收敛保证，并在实验中扩展了质量-多样性前沿。 |
| [^204] | [Density Feedback Control for Transferability of Generative Sparse Adversarial Attacks](https://arxiv.org/abs/2611.25001) | 提出密度反馈控制（DFC）方法，通过在每个训练轮次根据观测密度与目标密度的差距动态调整稀疏系数，使生成式稀疏对抗攻击无需重新调参即可在单次运行中精确达到目标扰动密度，并提升迁移攻击成功率。 |
| [^205] | [Watch Where You Linearize: Predicting and Correcting Model-Merging Interference](https://arxiv.org/abs/2611.25000) | 该论文发现，一阶方向导数只有在接收模型处评估才能有效预测模型合并中的干扰——可将基线准确率误差减半，而在共享预训练主干处评估则几乎无法预测干扰。 |
| [^206] | [PsyWorld: Profile-Conditioned World Modeling for Human Behavior Prediction](https://arxiv.org/abs/2611.24999) | 该论文提出PsyWorld，一种画像条件化的状态世界模型，将实际观测到的环境—行为状态转移作为结构化监督信号来学习行为表示，从而摆脱了传统上直接将静态人格画像映射为动作的做法，实现了更准确的人类行为预测。 |
| [^207] | [Variance-Adaptive Upper Confidence Bounds for Multi-Armed Bandits with Kaplan--Meier Survival Estimation](https://arxiv.org/abs/2611.24988) | 该论文提出了KM-UCB，一种基于Greenwood方差公式和经验Bernstein界限的方差自适应多臂老虎机策略，将连续时间生存分析与序贯决策相结合以处理右删失数据下的异方差问题。 |
| [^208] | [Risk-optimal active learning with informative abstention](https://arxiv.org/abs/2611.24986) | 该论文提出了一种在预言机可能弃权且响应概率与分类器共享参数情形下的主动学习查询获取准则，通过结合调整冗余参数后的分类器信息与多类零一风险的局部曲率，实现首阶渐近超额风险的最小化。 |
| [^209] | [DL101 The Representation Capability of Neural Network Models](https://arxiv.org/abs/2611.24980) | 本文系统梳理了通用近似定理这一神经网络表示能力的数学基石，指出非多项式激活函数是实现通用逼近的必要条件，并强调这些定理本身并不提供关于效率、训练收敛与泛化能力的保证。 |
| [^210] | [Improving On-board Shape Models Using Sun Sensors and Implicit Representations](https://arxiv.org/abs/2611.24976) | 该方法仅利用太阳敏感器日食掩蔽数据与可微分神经模型EclipseNets，通过精化二维轮廓并监督周期性激活的神经符号距离场，实现小行星等不规则小天体星载三维形状模型的端到端生成与改进，将轮廓IoU从0.70提升至0.87。 |
| [^211] | [When Is Program Structure Required? Certificate-Grounded Calibration of Program-Induction Difficulty](https://arxiv.org/abs/2611.24975) | 该论文提出“基于证书的校准”方法，通过完整的单规则枚举和可执行的双规则见证来严格证明程序归纳任务究竟需要一条还是两条规则，实验表明经认证的分离任务使模型完成率显著下降而未见见证对的成功率反而上升，从而能更可靠地区分程序归纳基准的真实难度与模型能力。 |
| [^212] | [ICARE: cross-domain text classification with incremental class-aware representation and distillation learning](https://arxiv.org/abs/2611.24959) | 提出ICARE框架，通过基于表示学习的实例选择模块和知识蒸馏方法，解决跨域文本分类中自然语言推理系统的灾难性遗忘与过拟合问题。 |
| [^213] | [LLM Complexity: Towards a Computational Complexity View of Large Language Model Inference](https://arxiv.org/abs/2611.24943) | 该论文证明LLM推理时的token使用量与经典时间复杂度并非单调对应关系，并提出“LLM复杂度”这一新指标，通过准确率-token曲线下的面积从LLM的视角量化问题复杂度。 |
| [^214] | [Variance-Optimal Counterfactual Policy Design for Sample-Efficient Online A/B Testing](https://arxiv.org/abs/2611.24757) | 该论文提出一种方差最优的反事实策略设计方法，通过联合优化动作的采样方式，将流量引导至两种策略不一致的动作上，从而显著降低在线A/B测试的方差并提升样本效率。 |
| [^215] | [Where Global Context and Weight Sharing Work in Compact Vision Backbones](https://arxiv.org/abs/2611.23446) | 该研究发现，在EfficientNet-B0骨干网络的P3、P4、P5阶段插入轻量级零门控残差自注意力适配器可将ImageNet-1k精度提升至74.02%，而单纯用BiFPN多尺度特征金字塔包装骨干则无法超越普通MBConv主干。 |
| [^216] | [Why Sub-4-Bit Vision CNNs Collapse and Where Mixed-Precision Recovery Stops](https://arxiv.org/abs/2611.23445) | 本研究系统剖析了视觉CNN在4比特以下量化时崩溃的机理，发现崩溃高度非线性且集中在后期逐点投影和深度可分离层，并提出了结构化的多步骤混合精度恢复瀑布策略来挽救精度。 |
| [^217] | [Storing the Gemma 4 K=V Cache Once: Exact Global-KV Memory Savings in a Paged Serving Engine with Unchanged Attention Kernels](https://arxiv.org/abs/2611.23035) | This paper exploits the linear dependence between the key and value in the global attention layers of Gemma 4 31B (K=V), storing only one 640-value row without modifying the attention kernel, which reads the key and value caches through two overlapping strided views, achieving a 1.34 to 1.40 fold reduction in KV cache memory during decoding with bit-identical outputs. |
| [^218] | [Every Measurement, Every Direction, All at Once: Multimodal Flow Matching for Molecules and Spectra](https://arxiv.org/abs/2611.22098) | 本文提出MOSAIC，一个多模态流匹配Transformer模型，通过对测量数据施加随机掩码，使分子三维结构与UV/IR/拉曼光谱之间实现任意到任意的双向生成，并在QM9S上达到88.8%的分子图恢复率，证明了单次实验可被充分挖掘为多方向的监督信号。 |
| [^219] | [Chain-of-Thought Controllability Can Reduce Chain-of-Thought Monitorability](https://arxiv.org/abs/2611.21799) | 论文通过人设控制训练发现，即使训练中不含任何规避监控的示例，仅仅让模型学会以不同人设口吻书写推理，就能显著降低其在执行隐蔽副任务时思维链的可监控性（g-mean²从0.94降至0.22），表明思维链可控性的增强可能削弱思维链监控的有效性。 |
| [^220] | [Synthetic Neurosymbolic Task Generation with Controlled Constraint Complexity](https://arxiv.org/abs/2611.19597) | 本文提出NeSySy，一个可对约束结构和复杂度等关键属性进行细粒度控制的神经符号学习任务合成生成器，能够联合生成逻辑约束与一致数据集，从而系统性研究任务难度因素及约束感知学习方法的有效性边界。 |
| [^221] | [SynthDCB: World Distillation for Scalable Device Control Benchmarking](https://arxiv.org/abs/2611.18639) | 该论文提出SynthDCB基准，创新性地通过“世界蒸馏”方法将设备行为知识转化为模块化可执行组件，构建了覆盖家居、汽车和手机三大领域的可扩展设备控制与环境感知评测基准。 |
| [^222] | [Vision-Language-Action Models: A Life-Cycle Survey of Design, Training and Deployment](https://arxiv.org/abs/2611.18372) | 这是一篇系统性综述，沿着模型设计（输入与记忆、架构、动作输出、中间预测）、训练（预训练数据等）与部署的全生命周期七个维度，梳理了2022至2026年间88项视觉-语言-动作（VLA）模型研究，覆盖操作、导航、人形控制与世界-动作模型等方向。 |
| [^223] | [Federated Physics-Informed Thermal Field Prediction Across Unseen Geometries in Distributed Additive Manufacturing](https://arxiv.org/abs/2611.17794) | 该论文提出Fed-PIGANO联邦学习框架，使各制造站点能够在不共享专有零件几何数据的前提下联合训练物理信息几何感知神经算子模型，从而在分布式增材制造中实现对未见几何形状的热场预测。 |
| [^224] | [Classical Energies Learn to Reason 1,000x Faster than Looped Models](https://arxiv.org/abs/2611.17309) | 提出基于 Hopfield & Tank 经典能量函数的学习版本 CRAM，通过能量下降实现推理，在数独和迷宫谜题上仅需毫秒级训练即可达到近乎完美的准确率，速度比循环模型快至少1000倍，推理计算量减少至少282倍。 |
| [^225] | [LazyWeights: Decode-on-Demand Edge LLMs](https://arxiv.org/abs/2611.17108) | LazyWeights在保留现有量化格式的基础上，通过对权重进行无损Huffman编码并在DRAM与L2之间用小型ASIC单元按需解码，在不改动GPU内核的前提下使边缘设备上的模型体积缩小5.7%-31.4%，并将每token的延迟和能耗最多降低约一半。 |
| [^226] | [When Does Causality Help in Multivariate Time-Series Anomaly Detection?](https://arxiv.org/abs/2611.15073) | 该研究通过固定因果引导自编码器并系统改变图质量、异常类型和评估协议，揭示了因果性在多变量时间序列异常检测中的三个有效条件：图质量决定收益大小、异常必须破坏条件结构（严重度≥0.3）才有效，且只有范围感知的VUS评估协议才能真实反映因果模块的性能优势。 |
| [^227] | [Prediction with Foundation Models under Deployment Budgets](https://arxiv.org/abs/2611.12466) | 该论文提出了在平均查询成本预算约束下提升基础模型预测性能的框架：固定查询模型的方法可达到极小化极大最优率，并证明多个单独欠佳的模型输出组合后可超越最佳单一模型；自适应方法则借助Q学习为不同输入动态选择查询模型并决定停止时机。 |
| [^228] | [Attention at Absolute Zero](https://arxiv.org/abs/2611.11937) | 该论文提出一种在零温度（硬最大值）下读取见证者的高阶配对注意力层，通过仅传递计数比较比特并在每轮循环重建状态，克服了软注意力在规模扩大时的权重分散、分数差距与状态漂移等失效模式，在比训练规模大1000倍的最大值检索任务上仍保持99.95%以上的准确率。 |
| [^229] | [On Agentic Manipulation Policies](https://arxiv.org/abs/2611.09892) | 提出AMP1智能体式操作框架，证明通用AI模型直接控制机械臂可零样本超越VLA和WAM等专用策略，在RoboLab-120上达到85.6%的成功率。 |
| [^230] | [Context--Answer Metamodels: Forecasting LLM Activations Before the First Answer Token](https://arxiv.org/abs/2611.06837) | 本文提出上下文-答案元模型，证明语言模型阅读上下文时产生的激活序列可以直接高保真地预测其在随机解码推演中产生的答案激活，且性能随训练数据增长而提升并能零样本泛化到域外数据集。 |
| [^231] | [ConDR: Conservative Reachability through Neural Hamilton-Jacobi Subsolutions](https://arxiv.org/abs/2611.06802) | 提出 ConDR 方法，通过单边损失和学习 HJI 变分不等式的下解来保证神经网络估计的危险集保守地包含真实危险集，同时避免退化解，实现了高维系统中安全可靠的可达性分析。 |
| [^232] | [The Alignment Tax Is Graded: Task-Selective Capability Loss Under Length-Normalized DPO with SFT Retention](https://arxiv.org/abs/2611.06476) | 对齐微调带来的能力损失并非均匀发生，而是分级且任务选择性的——抽取类任务几乎不受影响，JSON生成轻微下降，而路由任务出现约30分的严重退化。 |
| [^233] | [How many sorted projections determine n points in k dimensions?](https://arxiv.org/abs/2611.06470) | 本文借助Hair和Sahai关于随机线性码列表恢复的两个引理，证明了在任意k维空间中，几乎任意选择适当数量的方向即可确定n个点的多重集，从而回答了Dym等人提出的高维开放问题。 |
| [^234] | [Best-of-Depth: Learning Exploratory Dynamics in Looped Transformers](https://arxiv.org/abs/2611.04663) | 提出Best-of-Depth方法，通过跨深度最小化预测损失将循环Transformer的推理轨迹训练为候选解集合，无需架构改动即可显著提升N皇后补全和图着色等任务的解覆盖率（91.03% 对比 45.75%）。 |
| [^235] | [Generative Forecasting from Coarse Observations: Does Cascading Help?](https://arxiv.org/abs/2611.00686) | 该论文提出一种分层生成参考数据的方法来评估生成式预测器是否采样自正确的条件分布，发现标准扩散模型和随机插值在有限数据和模型容量下会产生显著的分布误差，并证明了采用级联较短预测时段的方式有助于缓解这一问题。 |
| [^236] | [Physiological Trajectories and Observation Patterns in ICU Mortality Prediction: Transportability Under Changing Measurement Conditions](https://arxiv.org/abs/2610.42126) | 该研究通过嵌套表示设计与跨数据库外部验证发现，生理轨迹在ICU死亡率预测中具有跨医院、跨电子病历系统的可迁移预测价值，并独立于观测数量评估了测量时机对性能的影响。 |
| [^237] | [Probing the limits of forward screening and inverse design of crystal materials](https://arxiv.org/abs/2610.32678) | 该研究将55个晶体数据库按结构匹配整合为包含1460万种晶体和1.318亿性能值的统一数据库，揭示晶体-性能数据仅有4.5%被填充，并开发了高精度等变图变换器CGTNet构建性能模型库来填补数据空缺，以此探究晶体材料正向筛选与逆向设计的能力极限。 |
| [^238] | [REACT: Representational Early Analysis and Control of Chain-of-Thought Trajectories](https://arxiv.org/abs/2610.32113) | REACT 提出一种从成对安全与不安全轨迹中学习安全方向、并一次性施加于思维链早期隐藏状态的激活干预方法，发现最强控制增益集中在思维链的前两到三个 token 窗口，即使后续状态更能预测风险。 |
| [^239] | [EgoHand: Monocular 3D Hand Reconstruction from Egocentric Video](https://arxiv.org/abs/2610.30293) | EgoHand是基于1400万张图像高质量语料库训练的单目3D手部重建模型，在自我中心视频上实现了最先进的性能，具备强大的零样本跨域泛化能力和约9.69毫秒的低推理延迟。 |
| [^240] | [Forecasting Patient Trajectories from the Electronic Health Records with Action-Conditioned World Models](https://arxiv.org/abs/2610.23862) | 该论文提出了一种动作条件化的世界模型，通过显式分离患者状态与临床行动并采用状态依赖的线性化预测器，实现了电子健康记录中患者轨迹的时间分辨预测、多步推演和动作条件化模拟。 |
| [^241] | [Refined Topology-Pass Scheduling: Damped Trust-Region Compensation for Dynamic k-NN Graphs in SE(3)-Equivariant Structure Prediction](https://arxiv.org/abs/2610.19246) | 提出一种双遍拓扑调度方案，将动态kNN图构建基准的选择形式化为以逐块重建（PB）为唯一不动点的阻尼不动点迭代，并借助不翻转证书、最小方差（BLUE）重放增益与ℓ∞信赖域投影进行保守补偿，从而在SE(3)等变结构预测中以低成本逼近逐块重建的拓扑精度。 |
| [^242] | [ResearchFlow-Eval: Tracing the Execution Gap in End-to-End Multi-Claim Autoresearch](https://arxiv.org/abs/2610.18088) | 该论文提出 ResearchFlow-Eval——一个基于已发表论文锚定、在阶段与论断两个层面细粒度评估端到端多论断自主研究系统持续科学执行与迭代能力的评测框架，弥补了现有评估仅打分孤立任务或最终成果的不足。 |
| [^243] | [Mixing Regularization for Average-Reward $Q$-Learning and $Q$-Planning](https://arxiv.org/abs/2610.14988) | 该论文提出通过向一个与奖励无关的 $m$ 步混合策略进行正则化来控制平均奖励 $Q$-学习的收缩与探索性质，并给出有限时间收敛保证，在单步情形下匹配最佳已知速率，在 $m$ 步混合情形下首次推导出新的速率。 |
| [^244] | [Cross-Behavior Policy Optimization: Unified Learning from Offline Demonstrations and On- and Off-Policy Rollouts via a Shared Regression Target](https://arxiv.org/abs/2610.12830) | CBPO通过固定参考的共享回归目标，将离线演示、在策略与离策略采样统一到同一回归损失和回放缓冲区中，无需分别协调SFT与RL目标、重要性加权或评论家网络即可实现数据的持续复用与统一学习。 |
| [^245] | [The Dichotomy Between Pattern Recognition and Step-by-Step Reasoning](https://arxiv.org/abs/2610.09186) | 论文提出模式识别与逐步推理是同一光谱的两端：当下一个词元仅依赖少量前文时 LLM 学会逐步推理，任务的推理轨迹构成 De Bruijn 图的有向无环子图，且其边数远小于轨迹数，因此 LLM 可以通过组合少量已学习的边来解决更长的新任务。 |
| [^246] | [Xaurora: Generative Weather Forecasting with Denoising Stochastic Interpolants from a Foundation Model Prior](https://arxiv.org/abs/2610.06509) | 本文提出Xaurora方法，通过新颖的去噪随机插值生成技术和SDE重放缓冲区，将预训练的确定性大气基础模型Aurora转变为生成式集合预报模型，以低成本实现概率化天气预报与不确定性量化。 |
| [^247] | [Kandinsky 6.0 Video: Foundation Models for Synchronized Video and Audio Generation](https://arxiv.org/abs/2610.05608) | Kandinsky 6.0 Video 是一系列用于同步文本到音频-视频生成的基础扩散模型，其核心创新是采用双流 CrossDiT 架构，通过双向交叉注意力实现视频流与音频流的时间和语义对齐，可生成带同步44 kHz音频（含唇音同步）的5秒全高清视频。 |
| [^248] | [The sublevel Flood bifiltration: towards scalable 2-parameter persistent homology](https://arxiv.org/abs/2610.05441) | 本文提出次水平Flood双过滤，通过扩展单参数Flood过滤，为2参数持续同调提供了一种具有理论稳定性保证且可高效计算、适用于大规模点集的可扩展近似方法。 |
| [^249] | [Compact set-valued deep ensembling in multi-class classification](https://arxiv.org/abs/2610.05332) | 该论文提出利用紧凑集成（贝叶斯神经网络和蒙特卡洛dropout卷积神经网络）生成概率预测并构建代表性分布以定义贝叶斯最优预测，同时引入一族集合效用来实现减轻训练与存储负担且更鲁棒的集合值多类分类预测。 |
| [^250] | [ArrivalBench: Agent-Generated Data Pipelines Are Correct Once and Wrong Under Time](https://arxiv.org/abs/2610.02363) | ArrivalBench 提出在对抗性但可重放的数据送达调度下重新执行代理生成的数据管道，并以完整日志的批量重算作为判定标准，发现经单次执行评分认证的管道中有 7.0-79.2% 实际上默默出错，且这一差距与修复循环无关。 |
| [^251] | [Capturing In-Context Learning Dynamics with Task Operators](https://arxiv.org/abs/2610.01054) | 提出任务算子（TO）方法，通过将注意力头输出的仿射变换解析地重放到注意力输出投影中，从而无需处理完整示例即可高效模拟上下文学习，并揭示其内在机制。 |
| [^252] | [Towards Better Exploration in Sequential Test-Time Scaling](https://arxiv.org/abs/2609.39632) | 该论文揭示序列测试时扩展常因过早陷入“吸引子”（进入后阻止探索其他答案的答案集合）而停止改进——在27种方法、模型与基准的组合中53.8%的轨迹在四次迭代内进入吸引子——并提出一种简单的模型混合干预方法来帮助逃离吸引子，实现更好的探索。 |
| [^253] | [GroundingPI: A Grounding Foundation Model towards Physical Intelligence with Visual Primitives](https://arxiv.org/abs/2609.39601) | GroundingPI是一个40亿参数的视觉定位基础模型，它将点和框表示为共享词表中的量化坐标，并通过多模态与空间预训练、监督微调及GRPO强化学习训练而成，在11种感知能力的34个基准测试上以平均73.68%的成绩超越更大的GPT-6 Astra，创下新的最先进水平，为物理智能提供了快速精确的视觉感知基础。 |
| [^254] | [GroundAnything: Reconciling Parallel Decoding with Precise Visual Grounding at Flash Speed](https://arxiv.org/abs/2609.39600) | GroundAnything 是一个 40 亿参数的视觉定位基础模型，它将视觉定位重新定义为可并行生成的双向扩散过程，通过分块去噪、AR 到扩散的架构转换训练及 GRPO 强化后训练，在实现闪速并行解码的同时保证了精准的视觉定位效果。 |
| [^255] | [Can Computation from Earlier Problems Help LLMs Solve New Ones?](https://arxiv.org/abs/2609.39394) | 提出 STAIR 方法，通过固定存储库复用早期响应的键值信息并仅训练 12,288 个参数，就能让大语言模型在多轮对话中利用先前问题的计算来提升解决新问题的准确率。 |
| [^256] | [Characterizing High Bandwidth Flash for LLM Serving](https://arxiv.org/abs/2609.39131) | 本文提出HBM-HBF-主机三层存储架构与缓冲缓存感知调度机制，系统评估了高带宽闪存在高吞吐量LLM智能体服务中何时能够有效提升服务性能与能效。 |
| [^257] | [STEPQuant: When and Where Errors Matter in Delta-Rule Recurrent State Quantization](https://arxiv.org/abs/2609.38169) | STEPQuant揭示了Delta规则循环状态量化误差在时间（记忆寿命）和空间（键行/值列）两个维度上的差异化影响，并提出一种时空后训练量化框架，根据误差大小与记忆寿命自适应分配精度，从而在低比特量化下保持模型精度。 |
| [^258] | [FlowMap-OPD: Rollout--Kernel Separation for On-Policy Distillation of Few-Step Flow-Map Generators](https://arxiv.org/abs/2609.37851) | 提出 FlowMap-OPD 在线策略蒸馏框架，通过展开-核分离将学生状态获取与师生分布比较解耦，并通过实验证明带独立可调学生一致性的瞬时速度分布监督是最有效的蒸馏方案。 |
| [^259] | [Predictive Self-Supervised Learning Provably Identifies Stochastic Signals under Nuisance](https://arxiv.org/abs/2609.37789) | 该论文证明了常见的预测式自监督学习方法通过“预测互信息最大化”与“潜分布匹配”两个互补原理，能够在干扰信息存在下可证地恢复与预测相关的随机潜在信号。 |
| [^260] | [Optimizer-dependent training dynamics converge to the same one-third optimal data scaling](https://arxiv.org/abs/2609.37745) | 研究发现训练动力学中的动态指数依赖于优化器类型，但最优调参后损失随数据量的缩放指数在不同优化器下均收敛于 1/3。 |
| [^261] | [Port-Hamiltonian Latent Deliberation: Mitigating the Deliberation Drift Cliff in Test-Time Compute Scaling](https://arxiv.org/abs/2609.37351) | 该论文发现了测试时潜空间深思中随思考深度增加而推理崩溃的“深思漂移悬崖”现象，并提出端口哈密顿潜空间深思框架，以同时兼顾表达能力、Lyapunov稳定性与计算效率，从而在深层思考步数下有效抑制漂移、缓解推理性能崩溃。 |
| [^262] | [Channel-Dependent State Space Model for Multivariate Time Series Forecasting](https://arxiv.org/abs/2609.36453) | 提出Chameleon，一种通道依赖的状态空间模型，通过将选择性SSM与卡尔曼滤波器结合，利用其缺失的测量更新机制实现变量间数据依赖的细粒度交互，在保持线性复杂度的同时实现稳健的多元时间序列预测。 |
| [^263] | [Arbitrary-Accuracy Neural Approximation with Optimal Neuron Count and Optimal Bit Complexity](https://arxiv.org/abs/2609.35628) | 本文通过显式构造激活函数与闭式网络，证明了在 $[0,1]^d$ 上对连续函数进行任意精度逼近时 $d+1$ 个隐神经元是精确最优的，并给出了结合取整与指数函数的更简单构造及最优编码复杂度。 |
| [^264] | [Tetra: Serving Leech-Lattice Quantized LLMs at 2.7 Bits per Parameter](https://arxiv.org/abs/2609.35465) | Tetra通过在Leech格上设计基于Golay码64状态网格图的新码本和高效解码内核，将每权重读取量从4.804比特降至2.148比特，实现了整模型每参数约2.7比特的高质量大语言模型量化部署。 |
| [^265] | [ProtoSeam: Lifting Classifier Training with Latent Gaussian Mixture Models](https://arxiv.org/abs/2609.35174) | 该论文提出ProtoSeam方法，通过在网络内部语义接口处插入可学习的类别原型并阻断梯度跨接口传播，以隐高斯混合模型的形式重构分类器训练，在不改变推理架构的前提下将测试准确率提升最多六个百分点。 |
| [^266] | [The Double-Edged Sword of Information: Revealed versus Hidden Lotteries in School Choice](https://arxiv.org/abs/2609.34074) | 本文放宽了“所有学生偏好相同”的假设，在贝叶斯纳什均衡的博弈论模型下证明：学校选择中公开摇号号码是一把双刃剑，既可能提高也可能降低匹配率与社会福利。 |
| [^267] | [Chameleon: Dynamic Format Adapter for Efficient Diffusion](https://arxiv.org/abs/2609.33496) | Chameleon提出了一种混合格式的扩散模型训练后量化框架，在固定位宽的前提下将数字格式本身作为离散变量，按权重通道、激活层乃至扩散时间步自适应地选择INT8或FP8等格式，从而实现更高效的扩散模型推理。 |
| [^268] | [Agnostic Smoothed Online Regression with Adversarial Responses](https://arxiv.org/abs/2609.32478) | 本文提出 Hedge-Cover 算法，利用协变量平滑性界定专家先验中的分歧次数，为具有有界对抗响应的不可知平滑在线回归实现了 $\widetilde{O}(\sqrt{\text{Pdim}(\mathcal{F}) \textsf{C}_{\textsf{cov}} T})$ 的次线性遗憾，从而解决了极小极大最优自适应遗憾这一开放问题。 |
| [^269] | [Gradients for Interventions and Activation Values for Detection: Targeted Feature Learning in Language Models](https://arxiv.org/abs/2609.32355) | 该论文提出“定向特征学习”框架，为预定义概念构建单一特征，并系统比较了三种模型信号（激活值、激活梯度、参数梯度）与两种估计器组合而成的六种方法，其中CAA和GRADIEND为已有方法，另外四种为新提出的方法。 |
| [^270] | [KernelOPT: Dispatch-Aware Agentic Search for GPU Kernel Optimization](https://arxiv.org/abs/2609.30059) | KernelOPT是一个调度感知的多智能体GPU内核优化系统，它在保留厂商库调用的同时仅优化编译器生成的Triton子内核，并通过静态校验、多种子正确性、模型级float64回退与性能门控组成的四道验证级联确保端到端的正确性与加速。 |
| [^271] | [Linear RNN Scaling Laws: When Longer Sequences Beat More Sequences](https://arxiv.org/abs/2609.27964) | 该论文在教师-学生理论框架下首次为线性RNN语言模型推导出显式的近似、优化与统计缩放定律，并揭示了在特定谱指数条件下，增加序列长度比增加序列数量能更有效地提升模型性能。 |
| [^272] | [Visual Jev: Accurate and Efficient Decisions from Shared Visual Context](https://arxiv.org/abs/2609.25845) | Visual Jev 通过对同一图像只编码一次并批量执行多个问题后缀，在将准确率从 70.6% 提升至 76.1% 的同时，实现了最高 8.9 倍的推理加速。 |
| [^273] | [Prior-Amortized In-Context Bayesian Inference for Generalized Linear Mixed-Effects Models](https://arxiv.org/abs/2609.24422) | 提出metabeta——一个预训练神经网络，在测试时接受任意先验族和超参数输入，利用集合变换器与条件归一化流对GLMM进行先验摊销的上下文贝叶斯推断，无需MCMC即可零样本泛化并获得校准的后验估计。 |
| [^274] | [Swiss-Knife: A Framework for Reconfigurable Externalised Multi-Objective Alignment at Decode Time](https://arxiv.org/abs/2609.22226) | 该论文提出Swiss-Knife框架，将解码时的多目标对齐规范变为可热插拔的运行时对象，并通过表示定理刻画了聚合算子族，证明成对聚合在对抗性奖励污染下比argmax更稳定。 |
| [^275] | [Paint-Anything: Unified Any-Color Control for Image Generation and Editing](https://arxiv.org/abs/2609.20816) | Paint-Anything通过对象级颜色监督学习共享的十六进制提示接口，结合Paint-500K数据集和精确匹配的纯色锚点，实现了图像生成与编辑中任意精确颜色的统一控制。 |
| [^276] | [QVAC Genesis III: A Large-Scale, High-Quality Open Synthetic STEM Corpus for Efficient Language Model Pre-Training](https://arxiv.org/abs/2609.19513) | 提出了QVAC Genesis III——一个包含1914.3亿token的开放STEM合成语料库，通过以弱学生模型信号驱动的双重生成策略（将失败转化为纠正性解释、将成功扩展为对比性选项级推理），为token预算受限的小模型高效预训练提供了高价值数据。 |
| [^277] | [Mammography Foundation Models for Opportunistic Prediction of Major Adverse Cardiovascular Events](https://arxiv.org/abs/2609.19385) | 该研究表明，为乳腺癌任务预训练的乳腺X线摄影基础模型无需心血管专门监督或乳腺动脉钙化标注，即可利用常规乳腺筛查影像机会性地预测女性未来5年的重大不良心血管事件风险（AUROC约0.82）。 |
| [^278] | [Stop Removing Stopwords: How an Inherited Preprocessing Default Distorts Legal Text-as-Data](https://arxiv.org/abs/2609.19153) | 本研究通过穷尽式单词消融实验，首次直接针对下游分类目标验证停用词去除这一沿袭自信息检索时代、从未被验证过的预处理默认设置，揭示其可能扭曲实证法学中基于TF-IDF和线性分类器的文本数据分析结果。 |
| [^279] | [Are We Grading Properly? Understanding Failure Modes in Medical Benchmarks](https://arxiv.org/abs/2609.16023) | 该论文提出RIFT评分标准失败分类法，对两个医学临床基准测试进行系统检测，发现大量评分标准存在非原子性和错位僵化等缺陷，且这些缺陷并非表面问题，仅重写捆绑标准就能导致相同回答的分数变化高达15.9个百分点。 |
| [^280] | [Direct Topology Tracking in Continuous Implicit Models](https://arxiv.org/abs/2609.12157) | 提出了一种通过直接查询连续隐式模型及其导数来追踪拓扑特征演化的框架，无需将数据重采样到离散网格，从而在避免离散化伪影的同时实现忠实的特征追踪，并适用于解析函数、多元函数逼近和隐式神经表示等多种隐式模型。 |
| [^281] | [Silver Rate Is (Almost) Optimal for Gradient Descent Acceleration](https://arxiv.org/abs/2609.09152) | 本文证明了银速率指数是梯度下降加速的最优多项式收敛速率，通过建立非anytime与anytime两种设定下匹配的下界，结合已有上界完整刻画了预定非负步长所能达到的最优加速效果。 |
| [^282] | [Fast Gauss Sums via Flash Attention](https://arxiv.org/abs/2609.04910) | 该论文发现仅需两次简单的输入增广，就能借助高度优化的Flash Attention计算任意带符号权重的高斯核和，无需编写任何自定义GPU代码，且在速度、内存开销和精度上均显著超越PyTorch和PyKeOps等现有方案。 |
| [^283] | [Modern Transformers Are Implicit Hybrids: From Functional Differentiation to Principled Hybrid Architecture Design](https://arxiv.org/abs/2609.02986) | 本文提出RFIS和RPD两个干预指标，发现现代Transformer的注意力头自然分化为由全局位置频带（GPBand）分隔的检索头和位置头两类，为有原则地设计FA-LA混合架构提供了实证基础。 |
| [^284] | [Balance of Benchmarks: Semantic Density Reweighting for Benchmark Multiplicity and Task-Conditioned Evaluation](https://arxiv.org/abs/2608.30044) | 提出基准平衡方法，通过对基准分配逆密度语义权重来纠正基准密集区域的重复计算问题，并支持基于任务查询的条件化模型评估，在586个模型上的留出任务预测相关性从等权加权的0.049大幅提升至0.462。 |
| [^285] | [Achieving Near-Zero-Overhead Multi-Model Hierarchical Classification in Real-Time Detection Pipelines](https://arxiv.org/abs/2608.11770) | 本文提出一种五步方法论，实现分类骨干网络在NVIDIA Jetson DLA核心上的零GPU回退INT8部署，通过架构适配和手动动态范围调整，在实时检测流程中实现近零开销的多模型层次分类。 |
| [^286] | [CHORUS: Coverage-Guided Agentic RL and Policy Consolidation for Testbench Stimulus Generation](https://arxiv.org/abs/2608.10090) | CHORUS是首个将覆盖率引导的智能体强化学习与策略整合相结合的测试平台激励生成框架，通过权重平均与奖励引导蒸馏，其4B模型在CVDP-ECov上达到88.0%的智能体Pass@1，超越DeepSeek-R1（671B）13.5个百分点。 |
| [^287] | [Omni-modal decomposition autoencoders learn full-stack wearable disentangled representations](https://arxiv.org/abs/2608.07385) | 提出全模态变分分解自编码器（OmniDecVAEs），利用单一共享编码器-解码器和多视角自监督分解损失，从任意多模态可穿戴时间序列中学习解耦表征，在人体活动识别数据集上实现最高 3.0 个百分点的 DCI 解耦度提升。 |
| [^288] | [D\'ej\`a Cue: Localizing States in Object Histories via Vocabulary-Relative Coordinates](https://arxiv.org/abs/2608.02044) | 该论文提出无需训练的 Déjà Cue 框架，通过将状态描述嵌入以状态平衡的词汇均值为中心化来构造词汇相对坐标，使每个状态与其备选项进行对比，从而在物体追踪历史中精确定位状态发生的时刻，并构建了包含620条物体历史、2,465个状态标注的 MS-VOST 基准数据集。 |
| [^289] | [From field-scale to large-scale spectral libraries: Tabular foundation models in soil spectroscopy](https://arxiv.org/abs/2608.00608) | 该研究系统评估了TabPFN表格基础模型在85个土壤光谱特性预测任务中的表现，发现其始终优于CNN、Cubist、随机森林和PLSR等传统方法，为从田间尺度到全球尺度的土壤光谱建模提供了更优的新方案。 |
| [^290] | [Persistent Sparse Autoencoders: Learning Feature-Specific Timescales in Language Model Representations](https://arxiv.org/abs/2607.17117) | 该论文提出持久稀疏自编码器，通过为每个特征学习一个持久性系数，使稀疏自编码器能够仅凭重构目标从语言模型激活中自动学习特征特定的时间尺度，同时保持高质量的重构效果。 |
| [^291] | [What Does a Temporal Benchmark Score Measure? Decomposing Channel Use in Video VLM Evaluation](https://arxiv.org/abs/2607.12304) | 该论文提出“反转冲突”评测方法（帧序反转而RoPE保持正向），将时序视频VLM基准分数分解为“任务是否真正需要时序信息”与“模型是从像素还是位置编码获取时序”两个通道，从而区分真正理解时序的模型与仅仅被干扰的模型。 |
| [^292] | [Gravitational Duals from Equations of State II: Large Hierarchies and False Vacua](https://arxiv.org/abs/2606.30117) | 该论文将基于物理信息神经网络（PINNs）的全息逆问题方法扩展至具有大能级层级和伪真空的强耦合量子场论情形，通过一系列方法学改进克服了近简并态与势能未探测区域等挑战，实现了从边界热力学数据重构体标量势。 |
| [^293] | [Necessary but Not Sufficient: Temperature Control and Reproducibility in LLM-as-Judge Safety Evaluations](https://arxiv.org/abs/2606.26185) | 本研究通过实验证明，即使将大语言模型作为评判者的采样温度设为0，也无法完全消除安全评估结果的不确定性，尤其是在决策边界附近的项目中仍存在不可重复性。 |
| [^294] | [Hierarchical Pooling for Sheaf Neural Networks](https://arxiv.org/abs/2606.20932) | 提出 HiSP 层次化池化层，通过节点分组、基于细尺度层结构缩减特征并构建粗化图，再在粗分辨率下学习新的层结构，使层神经网络能够实现层次化表示学习。 |
| [^295] | [A Human-in-the-Loop Bayesian Optimization Framework for Constraint-Aware Bioprocess Development](https://arxiv.org/abs/2606.19230) | 本文提出了一种扩展的人机协同贝叶斯优化框架，通过将约束满足概率和鲁棒性能作为帕累托目标，使专家能交互式选择最优候选，从而在生物过程开发中兼顾约束与不确定性。 |
| [^296] | [When Does Depth Matter For In-Context Learning? Adaptive Inference in Deep Transformers](https://arxiv.org/abs/2606.16694) | 本文提出了广义线性注意力Transformer中分布式推理的一般理论，证明多层Transformer可利用“函数向量”内部表示在各层以越来越精细的尺度推断潜在上下文变量，从而阐明了深度堆叠在何种条件下比单层结构具有计算优势。 |
| [^297] | [Rational Sparse Autoencoder](https://arxiv.org/abs/2606.14990) | 提出有理稀疏自编码器（RSAE），用可训练的有理函数取代固定的编码器激活函数，在匹配期望稀疏度下从理论上保证不劣于 ReLU，并在给定假设下可证明优于 JumpReLU 和 TopK。 |
| [^298] | [Implicit Neural Representations of Individual Behavior](https://arxiv.org/abs/2606.12200) | 提出Behavioral INR，一种自监督生成模型，将视觉领域的隐式神经表示适配到行为领域，把策略表示为由片段级潜在变量通过FiLM层调制的状态-动作函数，从而在无标签的混合行为数据中实现无监督的策略表示学习与策略身份推断。 |
| [^299] | [RePAIR: Predictive Self-Supervised Representation Learning in Chess](https://arxiv.org/abs/2606.11860) | 本文提出RePAIR架构，融合MAE、JEPA与BERT的思想，通过掩蔽潜在状态序列并用轻量级预测器在低维嵌入空间中修复空缺，实现了对国际象棋局面序列的自监督表征学习，使有意义的棋类概念在潜在空间中自然聚类。 |
| [^300] | [SPACE: Source-free Proxy Anchor Concept Erasure for MLLMs](https://arxiv.org/abs/2606.09868) | SPACE 提出了首个面向多模态大语言模型的无源机器遗忘框架，通过文本引导的代理锚点选择和双约束语义隔离两个阶段，在无需访问目标概念原始数据的情况下实现敏感概念的擦除。 |
| [^301] | [Automating the Expert Eye: A System-Agnostic Deep Learning Framework for Rare Event Discovery in Imbalanced Force Spectroscopy](https://arxiv.org/abs/2606.09541) | 该研究提出了一个基于改进ResNet18与焦点损失、无需系统特异性物理先验的深度学习框架，能够自动从严重不平衡的单分子力谱数据中筛查稀有解离事件，在阳性率仅1.34%的测试集上达到92.31%召回率和91.95%特异性。 |
| [^302] | [Learning Which Evidence-Seeking Decisions Matter: Calibrated Counterfactual Replay for Language Agents](https://arxiv.org/abs/2606.05263) | 提出了PCCC-RL方法，通过前缀条件下的反事实重放、预先信用预测与基于选择概率的校准纠正机制，精准识别语言智能体中哪些证据寻求决策真正决定了最终答案的质量，从而解决稀疏奖励下的信用分配问题。 |
| [^303] | [Supportive Token Revealing for Fast Diffusion Language Model Decoding](https://arxiv.org/abs/2606.04236) | 提出无需训练的AXON模块，通过监控剩余的不确定掩码令牌并在需要额外上下文时揭示支持性令牌，从而加速扩散语言模型的并行解码并缓解质量与延迟之间的权衡。 |
| [^304] | [Adaptive Inference for Resource-Constrained Dynamic Pricing](https://arxiv.org/abs/2606.03736) | 本文提出一种推断感知的重新求解策略，在资源受限动态定价中平衡收益与需求推断，实现接近最优的遗憾和信息效率。 |
| [^305] | [VESTA: Visual Exploration with Statistical Tool Agents](https://arxiv.org/abs/2606.00384) | VESTA是一个为视觉语言模型配备动态增长工具包的新框架，通过数据变换、假设驱动的可视化和统计检验来探索数据并改进统计建模，克服了以往仅依赖迭代批评的智能体系统在复杂建模任务上的局限。 |
| [^306] | [PhoneWorld: Scaling Phone-Use Agent Environments](https://arxiv.org/abs/2605.29486) | PhoneWorld通过真实GUI轨迹自动构建可扩展的手机使用环境，解决了移动代理训练与评估中环境构建的瓶颈。 |
| [^307] | [Insights Generator: Systematic Corpus-Level Trace Diagnostics for LLM Agents](https://arxiv.org/abs/2605.21347) | 该论文提出了洞察生成器（IG）——一个多智能体系统，通过在执行轨迹语料库上自动提出并检验假设，生成有证据支持的系统性诊断洞察报告，解决了 LLM 智能体失败诊断依赖人工、无法规模化的问题。 |
| [^308] | [AVSD: Adaptive-View Self-Distillation by Balancing Consensus and Teacher-Specific Privileged Signals](https://arxiv.org/abs/2605.20643) | 本文提出 AVSD（自适应视图自蒸馏）方法，通过平衡共识信号与教师特定特权信号，在多种特权信息视图之间自适应地进行自蒸馏，从而联合解决了师生信息不对称和特权信息类型因任务而异这两个难题。 |
| [^309] | [TriAxialKV: Toward Extreme Low-Precision KV-Cache Quantization for Agentic Inference Tasks](https://arxiv.org/abs/2605.17170) | 提出TriAxialKV，一种沿时间接近度、模态和语义角色三个维度利用token异质性的混合精度KV缓存量化方案，可实现面向智能体推理任务的极致低精度压缩。 |
| [^310] | [Spherical Boltzmann machines: a solvable theory of learning and generation in energy-based models](https://arxiv.org/abs/2605.09031) | 该论文借助随机矩阵理论与动力学平均场理论，建立了球形玻尔兹曼机在高维极限下的首个可解理论，推导出训练动力学的封闭方程、计算了作为参数空间配分函数的贝叶斯证据，并揭示了与耦合矩阵模态凝聚相关的级联相变及其对生成采样行为的影响。 |
| [^311] | [Survey-aware Machine Learning: A Guideline for Valid Population Health Inference based on Scoping Review](https://arxiv.org/abs/2605.08963) | 本文提出了一个九步骤的感知调查设计机器学习（SaML）指南，通过将调查设计元数据（如抽样权重、分层变量等）纳入机器学习全生命周期，解决基于复杂健康调查数据训练模型时估计偏差、不确定性低估和公平性评估失真的问题。 |
| [^312] | [Structured Role-Aware Policy Optimization for Multimodal Reasoning](https://arxiv.org/abs/2605.07274) | 该论文提出结构化角色感知策略优化（SRPO），通过将多模态结构化响应分解为感知token与推理token，把序列级的GRPO优势细化为角色感知的token级优势，实现更精确的信用分配，从而确保正确答案有任务相关的视觉证据支撑。 |
| [^313] | [FRInGe: Distribution-Space Integrated Gradients with Fisher--Rao Geometry](https://arxiv.org/abs/2605.06404) | 提出FRInGe方法，将积分梯度的参考点和积分路径定义在预测分布空间中，利用Fisher-Rao几何构造测地线路径，解决了传统积分梯度对设计选择高度敏感和梯度饱和的问题。 |
| [^314] | [Gradient Regularized Newton Boosting Trees with Global Convergence](https://arxiv.org/abs/2605.00581) | 本文提出受限牛顿下降理论框架，证明了牛顿提升树在光滑强凸损失下的线性收敛速率，并通过梯度正则化扩展实现了对一般凸损失函数的全局收敛保证。 |
| [^315] | [Beyond Visual Fidelity: Benchmarking Super-Resolution Models for Large-Scale Remote Sensing Imagery via Downstream Task Integration](https://arxiv.org/abs/2605.00310) | 该论文提出了GeoSR-Bench基准数据集，通过将土地覆盖分类、生物量估计、变化检测等下游任务集成到评估流程中，突破传统PSNR/SSIM保真度指标的局限，实现对遥感影像超分辨率模型实用价值的更全面评估。 |
| [^316] | [Learning Curves and Prediction Persistence to Under-Regularization of Spectral Algorithms in Large Dimensions](https://arxiv.org/abs/2604.23212) | 该文在大维度体制（$n\asymp d^\gamma$）下，以常数因子精度刻画了谱算法随正则化从最优尺度减弱直至插值时条件超额风险的阶数，揭示了风险何时保持极小极大阶、何时出现多项式量级更大的欠正则化风险区间。 |
| [^317] | [Friction-Augmented Drifting Models for Resource-Efficient Domain Translation](https://arxiv.org/abs/2604.18194) | 本文提出了一种带摩擦的漂移模型（DMF），通过线性调度的系数缩放漂移场，在域转换中防止迭代陷入虚假力平衡，从而显著提升性能。 |
| [^318] | [Optimal algorithmic complexity of inference in quantum kernel methods](https://arxiv.org/abs/2604.15214) | 该论文通过将完整推理求和编码为单个可观测量的期望值并结合量子振幅估计，实现了量子核方法推理的最优查询复杂度。 |
| [^319] | [OpenTME: An Open Dataset of AI-powered H&E Tumor Microenvironment Profiles from TCGA](https://arxiv.org/abs/2604.12075) | 该研究发布了OpenTME数据集，利用AI病理基础模型从TCGA十种癌症类型的5,268张H&E全切片图像中生成大规模开放的肿瘤微环境定量图谱，每张切片包含超过20,000个定量指标。 |
| [^320] | [Component-Adaptive and Lesion-Level Supervision for Improved Small Structure Segmentation in Brain MRI](https://arxiv.org/abs/2604.08015) | 提出CATMIL训练目标，通过成分自适应Tversky加权和病灶级多示例学习两个辅助损失项，在不修改网络架构的前提下显著提升脑部MRI中小病灶的分割召回率。 |
| [^321] | [The Value of Information in Resource-Constrained Pricing](https://arxiv.org/abs/2603.24974) | 本文研究了资源受限动态定价中的信息价值，在完全信息、无信息和知情定价三种信息环境下分别设计了算法：完全信息下仅求解一次流体问题的残差修正策略达到 O(log T) 后悔值，无信息下确定性探索加确定性等价策略达到 O(√T) 后悔值，知情定价下反馈控制器在斜率矩阵对称时无需估计斜率即可学习最优实施价格。 |
| [^322] | [Continuous-time learning of probability distributions: a case study in a digital trial of young children with type 1 diabetes](https://arxiv.org/abs/2603.24427) | 提出一个基于高斯混合模型与神经常微分方程的连续时间概率分布学习框架，用于刻画1型糖尿病儿童连续血糖监测数据中血糖分布随时间的动态演变。 |
| [^323] | [{\lambda}Split: Self-Supervised Content-Aware Spectral Unmixing for Fluorescence Microscopy](https://arxiv.org/abs/2603.23647) | 提出了λSplit，一种基于物理信息的自监督深度生成模型，通过分层变分自编码器学习浓度图的条件分布，并利用可微分光谱混合器，实现了对荧光显微镜光谱图像的内容感知解混。 |
| [^324] | [Hierarchical Latent Structure Learning through Online Inference](https://arxiv.org/abs/2603.19139) | 该论文提出HOLMES计算框架，通过结合嵌套中餐馆过程先验的变体与序贯蒙特卡洛推断，首次实现了无需显式监督、可逐试验在线进行层级潜在结构学习的推断方法，并在保持平坦模型预测性能的同时获得更紧凑的表征。 |
| [^325] | [Convergence Rate of a Functional Learning Method for Conditional Stochastic Optimization](https://arxiv.org/abs/2603.13048) | 本文针对无法直接从条件分布采样的条件随机优化问题，提出在参数函数类中近似条件期望并同时学习与优化的算法，并给出了该方法的收敛速率分析。 |
| [^326] | [SimVLA: Attributing Gains in VLA Models Through Controlled Ablation](https://arxiv.org/abs/2602.18224) | 该论文提出刻意极简的VLA基线模型SimVLA作为受控实验平台，通过单变量消融实验发现训练动态（而非架构或任务配置）才是VLA模型性能提升的最大驱动因素，可带来约54%–94%的性能波动。 |
| [^327] | [On the Complexity of Offline Reinforcement Learning with Q*-Approximation and Partial Coverage](https://arxiv.org/abs/2602.12107) | 该论文通过信息论下界证明仅凭 Q\* 可实现性与贝尔曼完整性不足以实现部分覆盖下样本高效的离线强化学习，并提出一个通用的决策估计框架，将离线强化学习的复杂度分解为决策复杂度与价值估计误差两部分。 |
| [^328] | [Privacy-Utility Tradeoffs with Quantum Local Differential Privacy](https://arxiv.org/abs/2602.10510) | 该论文首次系统研究了$(\varepsilon,\delta)$-量子局部差分隐私框架下的最优隐私-效用权衡，证明了去极化机制在保真度和迹距离等通用效用度量下达到理论最优。 |
| [^329] | [Wedge Sampling: Efficient Tensor Completion with Nearly-Linear Sample Complexity](https://arxiv.org/abs/2602.05869) | 本文提出“楔形采样”这一非自适应采样方案，通过对张量展开同列元素进行成对采样以提供谱初始化所需的行相关性，从而在固定秩条件下以近线性样本复杂度实现低秩张量的多项式时间恢复。 |
| [^330] | [EMA Policy Gradient: Taming Reinforcement Learning for LLMs with EMA Anchor and Top-k KL](https://arxiv.org/abs/2602.04417) | 提出EMA锚点策略和Top-k KL估计器两项简单技术来改进大语言模型的策略梯度算法，结合GRPO后在数学推理和智能体强化学习任务上取得显著性能提升。 |
| [^331] | [Anytime-Valid Conformal Risk Control](https://arxiv.org/abs/2602.04364) | 本文提出了任意时刻有效的保形风险控制方法，将误差控制从固定大小校准数据集上的平均有效性扩展为随校准数据累积增长时在任意时刻均以高概率保持有效，并证明了该保证渐近紧致且修正速率不可再改进。 |
| [^332] | [SpikingGamma: Temporally Precise Online SNN Training Through Smoothed Temporal Delays](https://arxiv.org/abs/2602.01978) | SpikingGamma是一种新型SNN架构，神经元通过维护紧凑的平滑延迟状态库并以sigma-delta脉冲编码进行通信，实现了时间精确的在线训练，克服了传统基于代理梯度的方法在精细时间分辨率下扩展性差、在线近似不稳定且难以捕捉时间模式的局限。 |
| [^333] | [The Post-Training Dilemma: Why We Should Rethink the Sequential SFT-RL Paradigm](https://arxiv.org/abs/2601.07389) | 本文从理论上证明SFT与RL在近乎正交的参数子空间上进行优化，导致RL阶段不可避免地损害SFT所获得的能力，且KL惩罚无法修复这一梯度不对齐问题，因此主张重新思考顺序式SFT-RL后训练范式。 |
| [^334] | [Learning Multinomial Logits in $O(n \log n)$ time](https://arxiv.org/abs/2601.04423) | 本文针对多项Logit（MNL）模型的学习问题，提出了自适应与非自适应算法，能够以 $O(\frac{n\log n}{\varepsilon^2})$ 次查询使每个选择板上的诱导选择分布在总变差距离 $\varepsilon$ 内逼近真实分布。 |
| [^335] | [Virasoro Symmetry in Neural Network Field Theories](https://arxiv.org/abs/2512.24420) | 本文从神经自由场架构出发严格构造出中心荷为一的Virasoro代数，证明极限神经流关联与Fock真空关联一致，并将该构造推广至费米子与超对称扇区。 |
| [^336] | [Quantum Nondecimated Wavelet Transform: Theory, Circuits, and Applications](https://arxiv.org/abs/2512.21478) | 该论文提出了量子非抽取小波变换的理论与电路实现，证明了其层级统计量在循环移位下不变，并通过块编码收缩算子与移位寄存器后选择精确实现循环平移平均，从而完成平移等变的量子去噪。 |
| [^337] | [A Bayesian latent class reinforcement learning framework to capture adaptive, feedback-driven travel behaviour](https://arxiv.org/abs/2512.14713) | 本文提出了一种基于贝叶斯潜在类别强化学习的框架，能够捕捉出行行为中个体偏好随时间演变的异质性，并通过驾驶模拟数据识别出三种不同适应策略的旅行者类型。 |
| [^338] | [Variational analysis of determinantal varieties](https://arxiv.org/abs/2511.22613) | 本文提出统一的变分分析框架，推导出各类低秩集合（矩阵、张量、对称矩阵、半正定矩阵等）的一阶与二阶切集显式公式及切集相交规则，并据此给出了非光滑问题与其光滑参数化具有等价二阶稳定点的充要条件。 |
| [^339] | [Predicting Public Health Impacts of Electricity Usage](https://arxiv.org/abs/2511.22031) | 提出了HealthPredictor——一个将电力使用与公共健康结果联系起来的端到端AI模型，通过燃料组合预测、空气质量模拟和货币化健康损害评估三个组件，实现基于健康信息的需求侧电力管理。 |
| [^340] | [Sparse corruption in low-rank matrix inference: the PCA benchmark](https://arxiv.org/abs/2511.11927) | 该论文研究了稀疏噪声（建模为有限平均连通度的加权图邻接矩阵）损坏下低秩信号的PCA推断问题，利用复制方法与种群动力学解析计算了最大特征值、特征向量分量密度及信号重叠度，并识别出随图连通度变化的两个信号强度转变点。 |
| [^341] | [A Nesterov-Accelerated Byzantine-Robust Federated Learning](https://arxiv.org/abs/2511.02657) | 该论文提出了一种将Nesterov动量与拜占庭鲁棒聚合规则无缝结合的联邦学习算法Byrd-NAFL，在非凸光滑损失函数和宽松假设下建立了有限时间收敛保证，同时兼顾通信效率与对拜占庭攻击的抵抗能力。 |
| [^342] | [Does AI Assistance Preserve or Collapse Disagreement? A Study of Pre-Annotations in Ambiguous Video Labeling](https://arxiv.org/abs/2510.21798) | 该论文提出一种受控审计协议，揭示了AI预标注在模糊时序视频标注中的双重效应——将标注速度提升约23%、提高标注者一致性与潜空间标准化，但同时可能锚定标注者于模型先验并压制对多元数据集建设有价值的分歧。 |
| [^343] | [Survival of the fittest Cox model: Pivotal variable selection for time-to-event data](https://arxiv.org/abs/2510.19374) | 本文通过对Cox比例风险模型的偏似然函数进行平方根变换，使正则化参数选择具有渐近枢轴性（精确消除基线风险和删失率的影响），从而显著提升了生存分析中变量选择的准确性和可靠性。 |
| [^344] | [Extensions of Robbins-Siegmund Theorem with Applications in Reinforcement Learning](https://arxiv.org/abs/2509.26442) | 本文扩展了罗宾斯-西格蒙德定理，将零阶项的条件从可求和放宽为平方可求和，并通过一个新颖温和的增量假设实现了几乎必然收敛到有界集合，同时给出了收敛速率、高概率集中界和 L^p 收敛，从而适用于更广泛的强化学习场景。 |
| [^345] | [BrainPro: Process-conditioned Pretraining with Topographic Retrieval for Large EEG Models](https://arxiv.org/abs/2509.22050) | BrainPro提出了一种过程条件化的自监督预训练框架，通过拓扑检索将不同数据集的电极导联配置映射到通用通道-区域模板，并同时学习共享表征与过程关联表征，从而解决EEG基础模型中心理过程线索利用不足和导联配置异构两大挑战。 |
| [^346] | [FlowDrive: moderated flow matching with data balancing for trajectory planning](https://arxiv.org/abs/2509.21961) | FlowDrive通过按轨迹模式重加权来平衡长尾驾驶数据，并提出带有环内微小扰动引导的调节式流匹配规划器，在保持场景一致性的同时增加轨迹多样性，在nuPlan和interPlan基准上取得了学习型规划器中的最先进结果。 |
| [^347] | [Deconstructing Intraocular Pressure: A Non-Invasive Multi-Stage Probabilistic Inverse Framework](https://arxiv.org/abs/2509.14167) | 该论文提出一种机制分层的两阶段物理引导逆问题框架（MHIP），将全局不可辨识的眼压逆问题分解为条件可辨识的子问题，并结合物理校准数据缩放（PCDS），从而在缺乏体内真值的条件下实现从临床数据中对眼压相关内在参数的无创概率估计。 |
| [^348] | [Benchmarking Robust Aggregation in Decentralized Gradient Marketplaces](https://arxiv.org/abs/2509.05833) | 本文形式化了梯度数据市场（GDM）这一新兴范式，指出市场不仅需要模型层面的鲁棒性，还必须确保高质量数据被正确识别和奖励，并针对此提出了鲁棒聚合方法的基准测试。 |
| [^349] | [SCAR: State-Space Compression for AI-Based Radio Resource Management of Vehicular Services](https://arxiv.org/abs/2508.06243) | SCAR框架利用聚类和径向基函数网络等机器学习压缩技术，将高维CQI状态信息压缩为紧凑表示，从而提升动态车载环境中基于AI的无线资源管理的可扩展性。 |
| [^350] | [TabPFN: One Model to Rule Them All?](https://arxiv.org/abs/2505.20003) | 本文从统计学视角将TabPFN阐释为一种近似贝叶斯推断方法，并证明这一表格数据基础模型在半监督参数估计、协变量偏移预测和异质性处理效应等任务上有时能超越专门的先进方法。 |
| [^351] | [Learning with Local Search MCMC Layers](https://arxiv.org/abs/2505.14240) | 受模拟退火与Metropolis-Hastings联系的启发，论文将局部搜索启发式算法的邻域系统转化为提议分布，构建出可微的随机MCMC组合层及对应损失函数，为使用不精确求解器的组合优化层学习提供了理论基础，并显著降低了计算成本。 |
| [^352] | [LLM-Based Threat Detection and Prevention Framework for IoT Ecosystems](https://arxiv.org/abs/2505.00240) | 该论文提出了一种基于大语言模型的物联网威胁检测与防御框架，通过在 IoT-23 和 TON_IoT 数据集上微调的检测模型识别异常网络行为，并结合自适应防御策略生成组件，在边缘-云实验环境中实现了 99.75% 的威胁检测准确率和 98.3% 的防御准确率。 |
| [^353] | [CP-Guard+: A New Paradigm for Malicious Agent Detection and Defense in Collaborative Perception](https://arxiv.org/abs/2502.07807) | 该论文提出了一种无需验证最终感知结果、直接在特征层面检测恶意代理的新范式，并发布了首个协同感知恶意代理检测基准数据集CP-GuardBench及鲁棒防御方法CP-Guard+，在大幅降低计算开销的同时有效防御协同感知中的恶意攻击。 |
| [^354] | [Almost Sure Convergence Rates and Concentration of Stochastic Approximation and Reinforcement Learning with Markovian Noise](https://arxiv.org/abs/2411.13711) | 本文通过一种使用长度递减区间的平均ODE新颖离散化技术，首次为带马尔可夫噪声的压缩随机逼近算法建立了几乎必然收敛速率和亚Weibull尾部的最大集中性界，并首次应用于无基于计数学习率的Q学习及离策略时序差分学习。 |
| [^355] | [FlexTSF: A Flexible Forecasting Model for Time Series with Variable Regularities](https://arxiv.org/abs/2410.23160) | FlexTSF 提出了一种灵活的领域自适应时间序列预测模型，其核心的 IVP Patcher 连续时间分块模块基于初值问题，能够原生处理不均匀采样间隔、可变序列长度和缺失值，从而有效应对不规则时间序列的通用预测挑战。 |
| [^356] | [Designing Algorithmic Recommendations to Achieve Human-AI Complementarity](https://arxiv.org/abs/2405.01484) | 本文提出了一种基于因果推断潜在结果框架的算法设计方法，将算法推荐与工具变量相联系，无需对推荐效果做先验假设即可设计出实现人机互补的推荐算法。 |
| [^357] | [Security of Deep Reinforcement Learning for Autonomous Driving: A Survey](https://arxiv.org/abs/2212.06123) | 本文对86项强化学习安全研究进行了全面综述，根据威胁模型和单/多智能体设置系统分类了攻击与防御方法，并评估了这些机制在自动驾驶场景中的适用性。 |
| [^358] | [Active Learning in the Predict-then-Optimize Framework: A Margin-Based Approach.](http://arxiv.org/abs/2305.06584) | 本文提出了预测-优化框架中的首个主动学习方法，通过基于间隔的准则和SPO损失的可求解替代函数来决定是否请求样本标签，在提供理论泛化保证的同时有效降低标签复杂度。 |
| [^359] | [Variance Reduction for Matrix Computations with Applications to Gaussian Processes.](http://arxiv.org/abs/2106.14565) | 本文提出了一种通过矩阵分解来进行矩阵计算的方差缩减方法，并展示了在某些问题上可以实现任意更好的随机性能。此外，本文提出的矩阵乘积迹的分解估计器与对半正定矩阵行列式的对数估计器均有显著的效率提高，可以在实际的贝叶斯优化问题中得到应用。 |

# 详细

[^1]: 扩散Transformer中的强化学习后训练本质上是激活导向

    RL Post-Training in Diffusion Transformers Is Activation Steering

    [https://arxiv.org/abs/2611.26627](https://arxiv.org/abs/2611.26627)

    该论文发现扩散Transformer的强化学习后训练收益可以用一组与输入无关的固定导向向量（参数量仅为LoRA的1/65）复现90%以上，证明RL后训练本质上等价于简单的激活导向。

    

    强化学习（RL）后训练能显著提升扩散Transformer的视觉质量、文字渲染和组合准确性。由于诸如渲染指定文本或组合多个对象等能力本质上依赖于提示词，一个自然的假设是：提升这些能力需要学习复杂的、依赖于输入的变换。为了探究这些收益中有多少可以由一个与输入无关的偏移来解释，我们用一种刻意简单的参数化方式来探测每个经RL调优的模型：在冻结的基础模型的块输出上添加一组固定的导向向量。在SD3.5-M上，这些向量的参数量比RL调优后的LoRA少65倍（0.29M对比18.8M）。我们拟合这些向量以匹配公开RL调优检查点的速度预测，目标函数中不含任何奖励信号。尽管这些向量对每个提示词、每个token和每个采样种子都是完全相同的，它们仍然恢复了六个RL后训练模型90.0–96.3%的奖励收益（原文此处截断）。

    arXiv:2611.26627v1 Announce Type: cross  Abstract: Reinforcement learning (RL) post-training substantially improves the visual quality, text rendering, and compositional accuracy of diffusion transformers. Because capabilities such as rendering requested text or composing multiple objects are inherently prompt-dependent, a natural hypothesis is that improving them requires learning complex, input-dependent transformations. To isolate how much of these gains an input-independent shift explains, we probe each RL-tuned model with a deliberately simple parameterization: a fixed set of steering vectors added to the block outputs of the frozen base model. On SD3.5-M, these vectors hold 65x fewer parameters than the RL-tuned LoRA (0.29M against 18.8M). We fit them to match the velocity predictions of public RL-tuned checkpoints, with no reward in the objective. Although the vectors are identical for every prompt, token, and sampling seed, they recover 90.0-96.3% of the reward gains of six RL-
    
[^2]: TideCache：具有透明缓存管理与逐层流式传输的高效混合LLM服务系统

    TideCache: Efficient Hybrid LLM Serving with Transparent Cache Management and Layer-Wise Streaming

    [https://arxiv.org/abs/2611.26619](https://arxiv.org/abs/2611.26619)

    TideCache是面向CPU-GPU超级芯片上混合架构LLM的服务系统，通过透明缓存池、逐层缓存流式传输和请求调度三大组件，动态管理异构缓存的显存分配与驻留，并将数据传输与计算重叠，从而实现高效服务。

    

    混合架构已成为开源权重语言模型的主流。其全局注意力层与有界状态层维护着异构缓存，其缓存需求随上下文长度和并发度变化。我们提出TideCache，一个面向CPU-GPU超级芯片的服务系统，包含三个组件，用于在各类缓存之间分配GPU显存，并管理它们在GPU与主机内存之间的驻留。透明缓存池在固定虚拟地址之后于各缓存组件间重新分配物理页面，从而保持缓存布局和后端接口不变。缓存流式传输器利用有界状态层提供的计算间隔，将下一个全局注意力层的前缀预取到由各全局注意力层共享的GPU插槽中，在减少GPU显存驻留的同时将逐层数据传输与计算重叠。已生成的KV保持驻留。请求调度器选择驻留或流式执行模式，并据此触发容量调整。

    arXiv:2611.26619v1 Announce Type: cross  Abstract: Hybrid architectures have become mainstream among open-weight language models. Their global-attention and bounded-state layers maintain heterogeneous caches whose demands vary with context length and concurrency. We present TideCache, a serving system on CPU--GPU superchips with three components for allocating GPU memory among caches and managing their residency across GPU and host memory. The Transparent Cache Pool redistributes physical pages among cache components behind fixed virtual addresses, preserving cache layouts and backend interfaces. The Cache Streamer uses computation intervals provided by bounded-state layers to prefetch the next global-attention layer's prefix into GPU slots shared across global-attention layers, reducing GPU residency while overlapping layer-wise transfers with computation. Generated KV remains resident. The Request Scheduler selects resident or streamed execution and triggers capacity adjustments acco
    
[^3]: 一个先验，两种动力学：将视频扩散Transformer重新用于自监督潜在动作学习

    One Prior, Two Dynamics: Repurposing Video Diffusion Transformers for Self-Supervised Latent Action Learning

    [https://arxiv.org/abs/2611.26617](https://arxiv.org/abs/2611.26617)

    该论文提出O²LaF，通过将同一个预训练视频扩散Transformer分别适配为逆向动力学过渡编码器和前向动力学动作条件生成器，从无标注视频中自监督学习可迁移的潜在动作，并在导航与机器人操作的少样本动作探测中达到新的最先进水平。

    

    视频揭示了世界如何变化，却很少揭示导致这些变化的动作。潜在动作学习从视觉过渡中恢复此类监督信号，但其底层动力学往往需要从零开始学习。我们提出一个问题：预训练的视频生成器能否同时为逆向动力学和前向动力学提供先验？我们提出了O²LaF，它建立在一个简单的原则之上：一个先验，两种动力学。我们从同一个预训练视频DiT初始化两个模型，并使其各自适应不同的角色：一个带有可学习动作查询的前馈式过渡编码器用于逆向动力学，以及一个因果的、以动作为条件的视频生成器用于前向动力学。通过控制信息访问并通过紧凑的瓶颈端到端地优化两种动力学，O²LaF从无标注视频中学到了可迁移的动作。在导航和机器人操作任务上，O²LaF在少样本动作探测中创造了新的最先进水平。

    arXiv:2611.26617v1 Announce Type: cross  Abstract: Videos reveal how the world changes, but rarely the actions that cause those changes. Latent action learning recovers such supervision from visual transitions, yet often learns the underlying dynamics largely from scratch. We ask whether a pretrained video generator can instead provide the prior for both inverse and forward dynamics. We introduce O$^{2}$LaF, built on a simple principle: one prior, two dynamics. We initialize both models from the same pretrained video DiT and adapt them to their distinct roles: a feed-forward transition encoder with learnable action queries for inverse dynamics, and a causal, action-conditioned video generator for forward dynamics. By controlling information access and optimizing both dynamics end-to-end through a compact bottleneck, O$^{2}$LaF learns transferable actions from unlabeled videos. Across navigation and robotic manipulation, O$^{2}$LaF sets a new state of the art in few-shot action probing 
    
[^4]: Bio-PM引导的合成IMU数据生成

    Bio-PM-Guided Synthetic IMU Generation

    [https://arxiv.org/abs/2611.26607](https://arxiv.org/abs/2611.26607)

    该论文提出利用预训练生物信号编码器Bio-PM的冻结编码器将IMU数据转换到token嵌入表示空间，并通过自回归transformer解码器和扩散解码器两类类条件解码器生成合成IMU数据，作为人体活动识别中提升泛化能力的实用数据增强方法。

    

    收集和标注用于人体活动识别（HAR）的惯性测量单元（IMU）数据成本高昂，且在某一组受试者上训练的模型往往难以泛化到未见过的个体。即使数据是平衡的，分类器仍然难以区分动态特征相似的活动类别，因此可以从有针对性的数据增强中受益。我们探索在预训练生物信号编码器Bio-PM的表示空间中生成的合成IMU数据能否作为一种实用的数据增强来源。我们不直接合成原始信号，而是利用Bio-PM的冻结编码器将10秒的IMU数据窗口转换为token嵌入序列，并在此空间中训练类条件解码器：（1）一种按顺序预测token的自回归transformer解码器，以及（2）一种通过去噪目标学习token分布的扩散解码器。我们通过分布性指标和嵌入分析来评估生成数据的真实性。

    arXiv:2611.26607v1 Announce Type: new  Abstract: Collecting and labeling inertial measurement unit (IMU) data for human activity recognition (HAR) is expensive, and models trained on one group of subjects often fail to generalize to unseen individuals. Even with balanced data, classifiers struggle to distinguish dynamically similar activity classes and benefit from targeted augmentation. We explore whether synthetic IMU data generated in the representation space of a pretrained biosignal encoder, Bio-PM \cite{b1}, can serve as a practical augmentation source. Rather than synthesizing raw signals, we leverage Bio-PM's frozen encoder to convert 10-second IMU windows into token embedding sequences and train class-conditional decoders in this space: (1) an autoregressive transformer decoder that predicts tokens sequentially, and (2) a diffusion decoder that learns the token distribution through a denoising objective. We analyze generation realism through distributional metrics and embeddin
    
[^5]: Depth Pro 2：清晰且时间一致的度量单目深度估计

    Depth Pro 2: Sharp and Temporally Consistent Metric Monocular Depth

    [https://arxiv.org/abs/2611.26603](https://arxiv.org/abs/2611.26603)

    Depth Pro 2 通过单一可扩展网络同时处理图像与视频，生成兼具度量准确、边界清晰和时间一致性的深度图，突破了以往方法在一致性与清晰度之间的权衡。

    

    我们提出了 Depth Pro 2，一种在完整分辨率下估计图像和视频稠密度量深度的方法。通过分析以往的单目深度估计方法，我们发现它们通常需要在时间一致性与边界清晰度之间进行权衡：多帧方法偏向一致性，单帧方法偏向清晰度。为了克服这种权衡，我们训练了一个单一、可扩展的网络，能够同时处理图像和视频，并以原生分辨率预测深度图。Depth Pro 2 生成的深度图同时具备度量准确性、清晰度和时间一致性，在所有这些维度上都超越了先前的工作。我们发布了一系列模型，在度量精度、清晰度、时间一致性和运行时间方面定义了新的帕累托前沿。我们在 https://apple-aiml-research.github.io/ml-depth-pro-2/ 公开了代码和模型权重。

    arXiv:2611.26603v1 Announce Type: cross  Abstract: We present Depth Pro 2, an approach to estimating dense metric depth for images and video at full resolution. By analyzing prior methods for monocular depth estimation, we find that they commonly trade off temporal consistency and boundary sharpness: multi-frame methods favor consistency, single-frame methods favor sharpness. To overcome this trade-off, we train a single, scalable network that processes both images and video and predicts depth maps at native resolution. Depth Pro 2 produces depth maps that are metrically accurate, sharp, and temporally consistent all at once, outperforming prior work in all of these dimensions. We release a family of models that define a new Pareto frontier in terms of metric accuracy, sharpness, temporal consistency, and runtime. We release code and weights at https://apple-aiml-research.github.io/ml-depth-pro-2/
    
[^6]: 通过蒸馏神谕教师实现主动特征获取

    Active Feature Acquisition by Distilling an Oracle Teacher

    [https://arxiv.org/abs/2611.26598](https://arxiv.org/abs/2611.26598)

    该论文提出OracleImitator，首次显式构建了一个按候选特征对下游预测器误差影响进行排序的神谕教师，并通过软目标模仿将这一排序蒸馏为可部署的主动特征获取策略，在多模态医疗诊断和AFABench基准上于整个预算范围内匹配或超越现有方法，恢复约90%的可达准确率增益。

    

    获取一个实例的每一个特征往往代价高昂：一次测量可能带来金钱成本、时间延迟或隐私代价。主动特征获取通过学习一个策略来解决这一问题，该策略在预算约束下针对每个实例决定获取哪些特征。现有的AFA方法通过优化一个自行构建的效用目标来训练该策略，而不是蒸馏一个显式的教师。我们显式地构建了这个教师。在训练阶段，标签和未观测的特征值是可获得的，二者共同以闭式形式确定了这个教师：一个神谕，它根据每个候选特征对下游预测器误差的影响来对其进行排序。OracleImitator通过软目标模仿将这一排序蒸馏为一个可部署的策略。在多模态医疗诊断和AFABench基准上，OracleImitator在整个预算范围内匹配或超越了现有AFA方法，恢复了90%的可达准确率增益……

    arXiv:2611.26598v1 Announce Type: cross  Abstract: Acquiring every feature of an instance is often costly: a measurement can carry a monetary price, a time delay, or a privacy cost. Active Feature Acquisition (AFA) addresses this by learning a policy that decides, per instance, which features to acquire under a budget. Existing AFA methods train this policy by optimizing a self-constructed utility target rather than distilling an explicit teacher. We construct this teacher explicitly. At training time the label and the unobserved feature values are available, and together they determine the teacher in closed form: an oracle that ranks every candidate by its effect on the downstream predictor's error. OracleImitator distills this ranking into a deployable policy by soft-target imitation. Across multimodal medical diagnosis and the AFABench benchmark, OracleImitator matches or outperforms existing AFA methods across the budget range, recovering 90% of the attainable accuracy gain at unde
    
[^7]: 实现可靠性与总体可靠性之辨：相关推理下任意时刻有效的自一致性认证

    Realized versus Population Reliability: Anytime-Valid Self-Consistency Certification under Correlated Reasoning

    [https://arxiv.org/abs/2611.26592](https://arxiv.org/abs/2611.26592)

    该论文发现语言模型推理流之间存在正相关，破坏了自一致性认证所依赖的条件独立假设，使被认证的假设分裂为关于单次抽取信念的“实现零假设”与关于模型平均行为的“总体零假设”（仅在零相关时二者等价），从而从根本上改变了 anytime-valid 证书所真正保证的可靠性含义。

    

    自一致性认证将语言模型的重复采样视为序贯证据累积过程，当任意时刻有效的 e-process 越过阈值时停止，并借助 Ville 不等式保证有限样本下的第一类错误控制。以往构造假设采样的 rollout 是条件独立的。我们证明该假设在实践中并不成立：推理流会因共享推理前缀或跨提示异质性而呈正相关，这不仅改变证书的效率，更改变了所认证的假设本身。通过引入每单元潜变量（随机效应/组内相关结构）对依赖性建模，可得到可交换、等相关的增量，其相关系数为 ρ。相关性将单一零假设分裂为关于所抽取信念的“实现零假设”与关于模型平均行为的“总体零假设”，二者仅在 ρ=0 时重合。我们证明：(i) 标准投注 e-process 仍保持任意……（原文摘要在此处截断）

    arXiv:2611.26592v1 Announce Type: cross  Abstract: Self-consistency certification treats repeated sampling from a language model as sequential evidence accumulation and stops when an anytime-valid e-process crosses a threshold, guaranteeing finite-sample Type I error control by Ville's inequality. Prior constructions assume sampled rollouts are conditionally independent. We show this assumption fails in practice: reasoning streams are positively correlated through shared reasoning prefixes or across-prompt heterogeneity, changing not merely certificate efficiency but the hypothesis certified. Modeling dependence via a per-unit latent state (the random-effects / intraclass-correlation structure) induces exchangeable, equi-correlated increments with correlation $\rho$. Correlation splits the single null into a realized null about the drawn belief and a population null about average model behavior, which coincide only when $\rho=0$. We show: (i) the standard betting e-process remains anyt
    
[^8]: 探索“有意识”的心灵：视觉-语言模型中的选择性访问与全局工作空间

    Probing the "Conscious" Mind: Selective Access and Global Workspace in Vision-Language Models

    [https://arxiv.org/abs/2611.26590](https://arxiv.org/abs/2611.26590)

    该论文提出多模态概念透镜（MC-Lens），通过将视觉编码器与语言解码器的中间状态映射到共享语义空间，并结合内部概念干预，揭示视觉-语言模型中概念的可及性与选择性访问机制，为全局工作空间理论在AI系统中的验证提供了工具。

    

    视觉-语言模型（VLM）整合视觉与文本信息，但其输出无法揭示内部的概念表征。全局工作空间理论（GWT）提出，信息从并行的大脑加工过程中被选择出来，整合进一个共享的工作空间，并可供专门化的加工过程使用。我们引入多模态概念透镜（MC-Lens）——一种多模态雅可比透镜，用于研究VLM概念处理中的上述功能。其视觉编码器透镜和语言解码器透镜将中间状态映射到由解码器输出词汇表定义的共享语义空间，以比较不同模态间概念的可及性。MC-Lens还支持内部概念干预，以检验被访问的概念在下游计算中所起的作用。（i）可及性与选择性访问：多模态概念信息在中间状态中是可读取的，并且能够根据任务需求被选择性访问。我们将“读出载体”从单一名称

    arXiv:2611.26590v1 Announce Type: new  Abstract: Vision-language models (VLMs) integrate visual and textual information, but their outputs do not reveal internal concept representations. Global workspace theory (GWT) proposes that information is selected from parallel brain processes, integrated in a shared workspace, and made available to specialised processes. We introduce the Multimodal Concept Lens (MC-Lens), a multimodal Jacobian lens, to study these functions in VLM concept processing. Its vision encoder and language decoder lenses map intermediate states to a shared semantic space defined by the decoder's output vocabulary to compare concept accessibility across modalities. MC-Lens supports internal concept interventions to examine the role of accessed concepts in downstream computation. (i) Accessibility and selective access. Multimodal concept information is readable in intermediate states and selectively accessible as tasks require. We extend "readout carriers" from one name 
    
[^9]: 有所表征却被误读：后训练改变了语言模型读取其内部表征的方式

    Represented but misread: Post-training changes how language models read out what they represent

    [https://arxiv.org/abs/2611.26584](https://arxiv.org/abs/2611.26584)

    研究发现后训练并未破坏语言模型内部的表征（线性探针仍能准确区分人类与AI文本），而是改变了模型将表征读取为答案的方式，导致回答灵敏度下降和对先验信息的错误利用。

    

    后训练究竟改变了语言模型所表征的内容，还是改变了将表征读取为答案的方式？我们在六个开源权重的基础模型及其后训练版本上，将线性探针与模型自身的回答进行了比较，并采用信号检测理论进行分析。在判断德语文本由人类还是AI撰写的任务中，线性探针和词n-gram分类器都能很好地区分类别，但模型的回答却不能，而且常常以相反的顺序排列文本。后训练平均降低了回答的灵敏度（跨模型家族Δd' = -0.34；跨三个提示词为-0.29）。它还改变了对所陈述基础比率的利用：基础模型对数几率的变化小于所陈述的先验几率变化，而大多数后训练模型的变化则更大。在两个具有阶段性检查点的模型中，这一变化发生在监督微调阶段；模型拼接实验表明相关机制位于早期层，且从中间深度开始，内部状态……（摘要在此处截断）

    arXiv:2611.26584v1 Announce Type: new  Abstract: Does post-training change what a language model represents, or how this representation is read out into an answer? We compared linear probes with the models' own answers, analyzed with signal detection theory, in six open-weight base models and their post-trained versions. For judgments of whether German texts were written by a human or an AI, probes and word n-gram classifiers separated the classes well, but the answers did not and often ordered the texts inversely. Post-training lowered the sensitivity of the answers on average ($\Delta d' = -0.34$ across model families; -0.29 across three prompts). It also changed the use of a stated base rate: Base models shifted their log-odds by less than the stated change in prior odds, most post-trained models by more. In two models with stage-wise checkpoints, the change occurred during supervised fine-tuning, model stitching implicated the early layers, and from mid-depth on internal states at 
    
[^10]: 文本感知、视觉盲区：多模态图像编辑系统如何处理不安全输入

    Text-Aware, Vision-Blind: How Multimodal Image Editing Systems Handle Unsafe Inputs

    [https://arxiv.org/abs/2611.26569](https://arxiv.org/abs/2611.26569)

    该论文提出EditSafetyBench基准，揭示多模态图像编辑系统存在系统性的文本-视觉安全失配：系统高度依赖文本提示进行安全审查（拒绝率超80%），但对不安全图像配合中性指令的情形检出率极低（仅约20%多），构成重大安全隐患。

    

    多模态图像生成系统取得了快速进展，并且日益重视内容安全性。然而，它们的安全机制往往依赖于单一模态，缺乏对视觉内容和文本提示的联合考量，尤其是在指令引导的图像编辑场景中，上传的图像本身可能就带有风险。为此，我们提出了 EditSafetyBench，这是一个用于衡量先进图像编辑系统在系统层面安全意识的基准，包含 5,940 个图像-指令对，覆盖 11 个风险类别和三种编辑意图下的安全与不安全图像。随后，我们在四个系统上对该基准数据集进行了测试，包括 Gemini、GPT、Qwen API 和 Qwen 本地版本。基于评估结果，我们观察到一种系统性的文本-视觉安全失配现象：Gemini 和 GPT 对包含不安全意图编辑提示的案例分别拒绝 87.1% 和 81.6%，而对于同时包含不安全图像和中性指令的案例，却仅拒绝 24.2% 和 22.4%。

    arXiv:2611.26569v1 Announce Type: cross  Abstract: Multimodal image generation systems have made rapid progress and increasingly emphasize content safety. However, their safety mechanisms tend to rely on a single modality and lack joint consideration of both visual content and text prompts, particularly in instruction-guided image editing, where the uploaded image may itself carry risk. We thus introduce EditSafetyBench, a benchmark that measures system-level safety awareness of advanced image editing systems, containing 5,940 image-instruction pairs, covering safe and unsafe images across 11 risk categories and three edit intents. We then probe the benchmark dataset on four systems, including Gemini, GPT, Qwen API, and Qwen Local. Based on our evaluation, we observe a systematic text-vision safety mismatch: Gemini and GPT refuse 87.1\% and 81.6\% cases containing unsafe-intent editing prompts, while only refusing 24.2\% and 22.4\% cases with both unsafe images and neutral instructions
    
[^11]: 面向集群规模排序模型的成本感知智能体特征优化系统

    Cost-Aware Agentic Feature Optimization System for Fleet-Scale Ranking Models

    [https://arxiv.org/abs/2611.26531](https://arxiv.org/abs/2611.26531)

    该论文提出了一个端到端的LLM智能体系统，在跨模型共享与逐模型设定的资源约束下，为大规模排序模型集群自动化地完成假设生成、交叉特征生成与筛选、多方法重要性排序融合以及异步试验调优，实现成本感知的跨模型特征优化。

    

    大型网络平台运行着大规模的排序模型集群，每个模型都从包含数万个特征的共享特征池中选取特征。我们提出了一种端到端的智能体系统，在跨模型共享的资源约束和逐模型设定的资源约束（分别为特征管线成本与服务吞吐量QPS）下，对整个模型集群的特征进行优化。在该系统中，LLM智能体负责提出假设，确定性工具负责执行实验并验证假设。对于每个模型，LLM从组合式候选空间中生成并初筛交叉特征，随后采用多种重要性方法对所有特征进行排序。特征融合将上述多个排序混合为单一选择，对于多任务模型则利用跨任务的帕累托前沿。接着，LLM研究员智能体在资源上限约束下通过10至25次异步试验对特征选择进行调优，并由修剪步骤确保管线成本保持在预算之内。在生产模型上，该系统……（原文摘要在此处截断）

    arXiv:2611.26531v1 Announce Type: cross  Abstract: Large Web platforms operate large fleets of ranking models, each selecting features from a shared pool of tens of thousands. We present an end-to-end agentic system that optimizes features across the fleet under resource constraints shared across models and those set per model, such as feature pipeline cost and serving throughput (QPS), respectively. In this system, LLM agents propose hypotheses, and deterministic tools execute experiments and verify them. For each model, an LLM generates and shortlists cross features from a combinatorial candidate space, and all features are then ranked by several importance methods. Feature blending mixes these rankings into a single selection, using Pareto fronts across tasks for multi-task models. An LLM Researcher agent then tunes the feature selection in 10--25 asynchronous trials under resource caps, and a trimming step keeps the pipeline cost within budget. Across production models, the system 
    
[^12]: 面向钙成像信号轨迹提取的确定性时序嵌入方法

    Deterministic Temporal Embedding for Calcium Imaging Trace Extraction

    [https://arxiv.org/abs/2611.26522](https://arxiv.org/abs/2611.26522)

    该论文提出一种确定性时序嵌入框架，通过避免随机负采样的确定性优化方法，将钙成像中逐像素荧光轨迹映射到时间相干源形成空间相干区域的低维表示，从而无需细胞形状先验即可可靠提取钙成像信号轨迹。

    

    在钙成像序列中可靠地提取钙依赖性荧光信号轨迹，需要识别其荧光强度反映同一潜在细胞活动的像素。我们提出了一种确定性时序嵌入框架，将逐像素的荧光轨迹映射到低维表示中，使时间上相干的活性源形成空间上相干的区域。从模糊邻域图和谱初始化出发，我们采用精确的稀疏吸引项以及带稀疏边补偿的稠密排斥项的确定性粒子-网格分解来优化类别平衡的交叉熵目标，从而避免了随机负采样。所得的潜在表示经过空间解码即可得到用于信号轨迹提取的可靠源支撑区域，而无需施加显式的细胞形状先验。在合成数据和体内双光子钙成像数据上的实验证明了该方法的稳健性。

    arXiv:2611.26522v1 Announce Type: cross  Abstract: Reliable extraction of calcium-dependent fluorescence traces in calcium imaging sequences requires identifying pixels whose fluorescence intensity reflects the same underlying cellular activity. We present a deterministic temporal embedding framework that maps pixel-wise fluorescence traces to a low-dimensional representation in which temporally coherent sources form spatially coherent regions. Starting from a fuzzy-neighborhood graph and spectral initialization, we optimize a class-balanced cross-entropy objective using exact sparse attraction and a deterministic particle-mesh decomposition of the dense repulsive interaction with sparse edge compensation, avoiding stochastic negative sampling. The resulting latent representation is spatially decoded into reliable source supports for trace extraction without imposing an explicit cellular shape prior. Experiments on synthetic and in vivo two-photon calcium imaging data demonstrate robus
    
[^13]: 窗口杠杆：在客户行为预测中，更长的观察窗口的影响超过数据量、模型容量与架构的效果

    The Window Lever: Longer Observation Windows Outweigh the Observed Effects of Data, Capacity, and Architecture in Customer Behavior Forecasting

    [https://arxiv.org/abs/2611.26521](https://arxiv.org/abs/2611.26521)

    在长时程客户行为预测中，将观察窗口从8周延长到16周所带来的性能提升超过了增加数据量、扩大模型容量或采用新架构的效果，且匹配测试表明这一增益甚至被原始比较所低估。

    

    在我们关于长时程客户行为预测的实验中，更长的观察窗口所产生的效果超过了更多数据、更大模型容量和更新架构所带来的效应。所谓窗口，是指模型在预测目标周之前能观察到的历史周数。在100万客户规模下，将窗口从8周延长到16周，在所有模型族中都带来了 +0.070 到 0.107 的对数空间 R² 原始提升。匹配测试队列（在每个窗口下对相同的客户-周进行评分）表明，这种原始比较实际上低估了其涵盖的每个模型族的增益。这一方向性结论在五个公开数据集上均成立，其中四个数据集的时间跨度超过一年（岭回归和我们的transformer在全部五个数据集上进行了验证）。在我们的内部数据上，更长的窗口允许纳入最多多10%的、更“冷”（历史更少）的训练客户，而匹配方法无法消除这一效应。我们的设计并未将窗口长度与日历覆盖范围分离开，因为每个测试锚点（即进行预测的周）都落在同一个八周区间内。在训练期锚点上，每周岭回归……（原文在此处截断）

    arXiv:2611.26521v1 Announce Type: new  Abstract: In our experiments on long-horizon customer-behavior forecasting, a longer observation window outweighed the observed effects of more data, more capacity and newer architectures. The window is the weeks of history a model sees before the week it forecasts from. Going from an 8- to a 16-week window adds a raw +0.070 to 0.107 log-space $R^2$ in every family at 1M. Matched test cohorts, which score the same customer-weeks at every window, show this raw comparison understates the gain in every family they cover. The direction holds on five public datasets, four spanning over a year (ridge and our transformer on all five). On our internal data, longer windows admit up to 10% more and colder training customers, which matching does not fix. Our design does not separate window length from calendar coverage, because every test anchor (the week a forecast is made from) falls in one eight-week band. On training-period anchors, the weekly ridge prob
    
[^14]: 简单的隐私保护噪声SGD在总体强凸性下是极小极大最优的

    Simple Private Noisy SGD is Minimax-Optimal under Population Strong Convexity

    [https://arxiv.org/abs/2611.26515](https://arxiv.org/abs/2611.26515)

    本文证明了在强凸损失（尤其是强凸广义线性模型，如隐私逻辑回归）上，采用固定步长的单次噪声SGD即可达到极小极大最优的估计误差率，其上界与基于Score Attack框架推导的隐私下界相匹配，并首次消除了对数因子。

    

    随机梯度下降（SGD）是现代机器学习的核心，并且在许多回归场景中已被证明具有统计上的极小极大最优性，例如凸的广义线性模型（GLMs）。然而，尚不清楚其最简单的差分隐私版本——噪声SGD（Noisy SGD）——即使在强凸损失函数上是否也能享有同样良好的保证。我们对这一问题给出了肯定的答案。我们在紧凸域上的光滑、强凸且满足Lipschitz条件的损失函数下，在Rényi差分隐私框架内，针对采用固定步长调度和恒定批量大小的单次噪声SGD，给出了参数估计误差的上界。当专门应用于强凸GLMs时，我们的误差上界与我们基于Score Attack框架推导出的极小极大下界相匹配。特别地，这为隐私保护的逻辑回归中的噪声SGD提供了首个极小极大最优的收敛速率，避免了现有结果中出现的多项式对数因子项。我们的分析方法具有广泛的通用性……

    arXiv:2611.26515v1 Announce Type: cross  Abstract: Stochastic gradient descent (SGD) is at the center of modern machine learning and is known to be statistically minimax-optimal in many regression settings, e.g., for convex Generalized Linear Models (GLMs). Yet, it is unknown whether its simplest differentially private Noisy SGD counterpart fulfills similar favorable guarantees even on strongly convex losses. We resolve this question affirmatively. We upper bound the parameter estimation error for a single pass of Noisy SGD with a fixed stepsize schedule and constant batch sizes on smooth and strongly convex Lipschitz losses over a compact convex domain under Renyi DP. Specialized to strongly convex GLMs, our error bounds match minimax lower bounds we derive based on the Score Attack framework. In particular, this provides the first minimax-optimal rate for Noisy SGD in private logistic regression, avoiding poly-logarithmic terms present in existing results. Our analysis is versatile a
    
[^15]: 英文与马拉地语政府计划搜索的混合检索

    Hybrid Retrieval for Government Scheme Search in English and Marathi

    [https://arxiv.org/abs/2611.26514](https://arxiv.org/abs/2611.26514)

    该论文提出一个面向马哈拉施特拉邦政府计划的双语（英语-马拉地语）搜索系统，通过等权重融合BM25与句子嵌入相似度，混合检索显著优于单独的BM25或密集检索，MRR最高达0.879。

    

    检索政府计划需要将公民的信息需求与正式的行政描述相匹配，这通常涉及跨语言和跨术语的匹配问题。我们提出了一个针对马哈拉施特拉邦政府计划和公共服务的双语搜索系统，该系统通过等权重分数融合将BM25检索与句子嵌入相似度相结合。该系统对对齐的英语-马拉地语计划描述进行索引，并支持多语言、以马拉地语为主以及翻译中介的语义配置。在人工整理的双语查询上进行评估，使用IndicSBERT和MahaSBERT对BM25、密集检索和混合检索进行了比较。BM25提供了强大的基线，平均倒数排名（MRR）为0.864。混合检索使用IndicSBERT将MRR提升至0.872，使用MahaSBERT提升至0.879，而相应的纯密集检索系统仅达到0.663和0.655。基于MahaSBERT的融合还使精确率从0.781提升至0.808，同时匹配B（原文在此截断）

    arXiv:2611.26514v1 Announce Type: cross  Abstract: Retrieving government schemes requires matching citizens' information needs to formal administrative descriptions, often across languages and terminology. We present a bilingual search system for Maharashtra government schemes and public services that combines BM25 retrieval with sentence-embedding similarity through equal-weight score fusion. The system indexes aligned English-Marathi scheme descriptions and supports multilingual, Marathi-focused, and translation-mediated semantic configurations. Evaluation on manually curated bilingual queries compares BM25, dense retrieval, and hybrid retrieval using IndicSBERT and MahaSBERT. BM25 provides a strong baseline with a mean reciprocal rank (MRR) of 0.864. Hybrid retrieval increases MRR to 0.872 with IndicSBERT and 0.879 with MahaSBERT, whereas the corresponding dense-only systems achieve 0.663 and 0.655. MahaSBERT-based fusion also increases precision from 0.781 to 0.808 while matching B
    
[^16]: 网格细胞作为谱后继特征：持续强化学习的可复用基

    Grid Cells as Spectral Successor Features: A Reusable Basis for Continual Reinforcement Learning

    [https://arxiv.org/abs/2611.26504](https://arxiv.org/abs/2611.26504)

    该论文受网格细胞对应后继表示特征向量的启发，提出用固定参考策略转移矩阵的特征向量作为后继特征的可复用基，使得长期回报预测具有封闭形式的系数，且无需后继特征网络或回放缓冲区，为持续强化学习提供了高效基础。

    

    持续强化学习要求智能体在任务发生变化时进行适应，同时复用其对环境的已有知识。后继特征可以将环境动态与任务特定的奖励分离，但其有效性取决于基础特征的选择。以内嗅皮层网格细胞对应于后继表示特征向量这一发现作为设计原则，我们从固定参考策略转移矩阵的特征向量构建这一基。采用这样的基，某一状态-动作对处的后继特征是该同一状态-动作对处基础特征的线性组合——这一性质在一般情况下并不成立——且组合系数具有封闭形式：每个后继特征等于其对应的基础特征乘以 θ_i = 1/(1-γλ_i)，其中 λ_i 是其特征值。因此，长期回报可以用与即时奖励相同的特征来预测，无需后继特征网络或经验回放缓冲区。

    arXiv:2611.26504v1 Announce Type: new  Abstract: Continual reinforcement learning requires agents to adapt when the task changes while reusing what they know about the environment. Successor features separate dynamics from task-specific reward, but their effectiveness depends on the choice of base features. Taking the identification of entorhinal grid cells with eigenvectors of the successor representation as a design principle, we build this basis from eigenvectors of the transition matrix of a fixed reference policy. With such a basis, the successor features at a state-action pair are a linear combination of the base features at that same pair, which is not true in general, and the coefficients are known in closed form: each successor feature is its base feature multiplied by $\theta_i = 1/(1-\gamma\lambda_i)$, where $\lambda_i$ is its eigenvalue. The long-term return is thus predicted from the same features as the immediate reward, no successor-feature network or replay buffer is ne
    
[^17]: 使用BuildPredictor预测未来构建结果

    Anticipating Future Build Results with BuildPredictor

    [https://arxiv.org/abs/2611.26476](https://arxiv.org/abs/2611.26476)

    本文提出基于随机森林的BuildPredictor模型，利用源代码与构建代码的变更、代码复杂度变化以及历史构建结果等特征来预测CI构建结果，在项目内和整体设置中分别取得了0.82和0.86的中位数MCC，有望减少CI中因微小更改而浪费的时间和资源。

    

    持续集成（CI）是一种流行的编程实践，用于将代码更改无缝集成到代码库中。使用CI，开发人员可以在每次提交后受益于自动化的代码组装和测试。然而，尽管频繁的集成和构建为开发人员提供了支持，但它也带来了高昂的代价，即对微小的更改也会浪费大量的时间和资源。在这项工作中，我们研究了通过使用机器学习技术预测构建结果来减少这些缺点的潜力。具体而言，我们提出了一个随机森林模型，该模型具有多种特征，例如源代码和构建代码的更改，以及代码复杂度的变化。我们观察到，我们的模型在项目内和整体设置中均表现出良好的性能，中位数MCC分别达到0.82和0.86。我们还发现，模型中最重要的特征与历史构建结果信息有关。

    arXiv:2611.26476v1 Announce Type: cross  Abstract: Continuous Integration (CI) is a popular programming practice for the seamless integration of code changes into the code base. Using CI, developers benefit from automated assembling and testing of the code after each commit. However, while the frequent integration and building supports developers, it also comes at the high costs of wasting time and resources for minimal changes.   In this work, we investigate the potential of reducing the drawbacks by anticipating the build outcome using machine learning techniques. In particular, we propose a random forest model with various features such as changes to the source and build code, and changes to the complexity of the code.   We observe that our model shows a good performance in an intra-project and holistic setting yielding a median MCC of 0.82 and 0.86, respectively. We also found that the most important features of our model concern the historical build result information. Additionall
    
[^18]: 一种寻找平坦极小值的新算法方法：临界阻尼Polyak步长

    A new algorithmic approach for finding flat minima: critically damped Polyak stepsizes

    [https://arxiv.org/abs/2611.26475](https://arxiv.org/abs/2611.26475)

    本文提出临界阻尼Polyak（CDP）自适应步长——将经典Polyak规则乘以4并加温和阻尼——使梯度下降偏向更平坦的极小值，并首次为局部二次目标函数建立了锐度与损失下降的时间一致理论保证。

    

    我们提出了一种新的自适应步长规则，它在理论和实践上都使梯度下降偏向更平坦的极小值，并首次为一大类局部二次目标函数建立了关于锐度和损失下降的时间一致保证。我们的临界阻尼Polyak（CDP）步长将经典的Polyak规则乘以 $4$ 和一个温和的阻尼因子。因子 $4$ 至关重要，因为它在主导阶上产生围绕极小值流形 $\mathcal{M}$ 的等幅振荡。而阻尼因子则可证明地驱动迭代收敛到 $\mathcal{M}$。因此，CDP通过使步长自适应于局部曲率，在稳定性的边缘运行。一旦损失变得足够小，其投影迭代在主导阶上沿着 $\mathcal{M}$ 执行关于 $\log{\lambda_1(\nabla^2 \mathcal{L})}$ 的梯度下降，这与归一化梯度下降的平坦性偏好相呼应（Arora et al., 2022）。与归一化梯度下降不同，CDP的步长会随……（摘要此处截断）而收缩。

    arXiv:2611.26475v1 Announce Type: cross  Abstract: We present a new adaptive stepsize rule that biases gradient descent toward flatter minima in both theory and practice, and establish the first uniform-in-time guarantees for both sharpness and loss reduction for a broad class of locally quadratic objectives. Our critically damped Polyak (CDP) stepsizes multiply the classical Polyak rule by $4$ and a mild damping factor. The factor $4$ is critical in that it produces equioscillations around the manifold of minimizers $\mathcal{M}$ to leading order. Damping provably drives convergence to $\mathcal{M}$. CDP thus operates at the edge of stability by adapting its stepsizes to local curvature. Once the loss is small, its projected iterates follow gradient descent on $\log{\lambda_1(\nabla^2 \mathcal{L})}$ along $\mathcal{M}$ to leading order, paralleling the flatness bias of normalized gradient descent (Arora et al., 2022). Unlike normalized gradient descent, CDP step lengths shrink with th
    
[^19]: 神经条件高斯模型：面向非线性动力学的高效数据同化与序贯不确定性量化

    Neural Conditional Gaussian Model: Efficient Data Assimilation with Sequential Uncertainty Quantification for Nonlinear Dynamics

    [https://arxiv.org/abs/2611.26463](https://arxiv.org/abs/2611.26463)

    本文提出神经条件高斯模型，将未观测的物理状态映射到具有条件高斯结构的潜在空间，从而实现闭式数据同化更新，并通过高斯解码器对物理变量进行高效的不确定性量化。

    

    在部分可观测的非线性系统中进行数据同化（DA）需要对隐含状态及其不确定性进行反复估计。传统的非线性滤波方法计算代价高昂，且通常依赖于显式的动力学模型。本文提出了神经条件高斯模型，这是一种数据驱动的动力学模型，它将未被观测到的物理状态映射到一个学习得到的潜在空间中，其中潜在状态的动力学遵循条件高斯结构。这一结构使得潜在空间中的数据同化更新具有闭式解形式，同时非线性编码器-解码器映射保留了在物理空间中表征复杂动力学所需的灵活性。Neural CGM 的一个关键特性在于其对数据同化中不确定性的处理：潜在后验均值和协方差通过一个结合了局部矩传播与学习到的非线性修正的高斯解码器映射回物理变量。由此产生的……（原文摘要至此截断）

    arXiv:2611.26463v1 Announce Type: new  Abstract: Data assimilation (DA) in partially observed nonlinear systems requires repeated estimation of both hidden states and their uncertainty. Conventional nonlinear filtering can be computationally expensive and often relies on an explicit dynamical model. In this paper, we develop the Neural Conditional Gaussian Model (Neural CGM), a data-driven dynamical model that maps unobserved physical states into a learned latent space, where the dynamics of the latent states follow a conditional Gaussian structure. This structure allows closed-form DA updates in the latent space, while nonlinear encoder-decoder mappings retain the flexibility needed to represent complex dynamics in physical space. A key feature of Neural CGM is its treatment of uncertainty in DA. The latent posterior mean and covariance are mapped to the physical variables through a Gaussian decoder that combines local moment propagation with learned nonlinear corrections. The resulti
    
[^20]: 摊销式搜索空间上的贝叶斯优化

    Bayesian optimization on amortized search space

    [https://arxiv.org/abs/2611.26459](https://arxiv.org/abs/2611.26459)

    该论文提出了一种利用机理模拟器的摊销式逆映射来约束搜索空间的贝叶斯优化方法，通过感知问题几何的高斯过程核高效利用问题的低内在维度，并给出了依赖内在维度的遗憾率理论保证，成功应用于帕金森病的药物剂量个性化。

    

    在现实世界的黑盒优化问题中，通常存在一个机理模拟器，它能够以被优化参数的函数形式近似预测系统的行为。本文展示了如何利用此类模拟器的摊销式逆映射来限制贝叶斯优化的搜索空间，同时使用能够感知原始问题几何结构的高斯过程核。这使得我们能够高效地利用优化问题的低内在维度。我们刻画了该方法的近似偏差和信息增益，并在额外的几何与高斯过程假设下，推导出了依赖于内在维度的遗憾率。我们以帕金森病的药物剂量个性化作为应用实例来说明该方法。我们使用药物的药代动力学/药效动力学（PK/PD）模型作为模拟器，这是一个基于群体数据校准并量化了参数不确定性的机理模型。

    arXiv:2611.26459v1 Announce Type: cross  Abstract: In real-world black-box optimization problems, a mechanistic simulator is often available that approximately predicts the behavior of the system as a function of the parameters being optimized. Here, we show how an amortized inverse map of such a simulator can be used in Bayesian optimization to restrict the search space while using a Gaussian process kernel that is aware of the original problem geometry. This allows for efficiently leveraging the low intrinsic dimensionality of the optimization problem. We characterize its approximation bias and information gain, and derive an intrinsic-dimension-dependent regret rate under additional geometric and Gaussian Process assumptions. We illustrate the method with an application to drug dosing personalization in Parkinson's disease. We use, as a simulator, a pharmacokinetic/pharmacodynamic (PK/PD) model of a drug, a mechanistic model calibrated on population data with quantified parameter un
    
[^21]: 扩展经典硬件以实现精确的实时量子纠错

    Scaling classical hardware for accurate real-time quantum error correction

    [https://arxiv.org/abs/2611.26447](https://arxiv.org/abs/2611.26447)

    本文提出与FPGA协同设计的流式神经解码器Rapids，在单块FPGA上以亚微秒级延迟对距离-19表面码进行实时解码，同时兼顾速度与精度，满足了超导量子处理器实时量子纠错的苛刻需求。

    

    容错量子计算需要一个足够快的经典解码器，能够以综合征测量产生的速率对其进行处理，并以低延迟返回决策。这对超导处理器的要求最为苛刻，因为其快速的综合征提取周期迫使解码速度与准确性之间进行权衡。速度足够快以跟上节奏的解码器会牺牲准确性，而准确性的损失必须通过更大的码来补偿，从而需要额外的物理量子比特。本文介绍了Rapids，一种既快速又准确的流式神经解码器。我们通过将解码器与现场可编程门阵列（FPGA）协同设计来实现这一目标：该神经网络由能够低成本映射到硬件上的操作构建，并在其整数算术中进行训练。对于距离为19的表面码，Rapids在单块FPGA上以不到一微秒的时间解码每一轮综合征，对超导处理器而言速度足够快，并以低延迟返回决策。

    arXiv:2611.26447v1 Announce Type: cross  Abstract: Fault-tolerant quantum computation requires a classical decoder fast enough to process syndrome measurements at the rate at which they are generated, and to return its decisions with low latency. This is most demanding for superconducting processors, whose fast syndrome cycles force a trade-off between decoding speed and accuracy. Decoders fast enough to keep up sacrifice accuracy, which must be compensated for with larger codes and thus additional physical qubits. Here we introduce Rapids, a streaming neural decoder that is both fast and accurate. We achieve this by co-designing the decoder with a field-programmable gate array (FPGA): the network is built from operations that map cheaply onto the device and trained in its integer arithmetic. For the distance-19 surface code, Rapids decodes each round of syndromes in under a microsecond on a single FPGA, fast enough for superconducting processors, and returns its decisions with the low
    
[^22]: 比利时人应该在电气化住宅中安装先进能源管理系统吗？

    Should Belgians Install an Advanced Energy Management System in Their Electrified Dwelling ?

    [https://arxiv.org/abs/2611.26441](https://arxiv.org/abs/2611.26441)

    本研究通过模拟500户比利时电气化住宅，评估模型预测控制（MPC）与基于决策树的TreeC等先进能源管理系统相比传统规则控制器，在热泵供暖、电热水和电动汽车充电场景下的潜在收益。

    

    随着住宅电气化的推进以及更复杂电价机制在比利时的普及，一个问题随之而来：在电气化的比利时住宅中安装先进能源管理系统有哪些潜在收益？为回答这一问题，我们模拟了500户具有代表性的比利时住宅，其中化石燃料系统被可调控的电力技术所取代：采用空气-空气热泵进行供暖、电锅炉提供生活热水、以及配备家用充电桩的电动汽车满足私人出行需求。光伏系统则按照当前的普及率进行配置。我们评估了两种先进能源管理系统（EMS）：模型预测控制（MPC）以及一种使用决策树来建模EMS策略的机器学习方法TreeC。这两种方法与模拟默认运行方式的基于规则的控制器（RBC）以及具有完美预测的MPC（MPC P，作为性能上限）进行比较，并对这些EMS的性能进行评估。

    arXiv:2611.26441v1 Announce Type: cross  Abstract: As dwellings electrify and more complex electricity pricing spreads in Belgium, the question arises: What are the potential benefits of installing an advanced energy management systems in an electrified Belgian dwelling ?   To answer this question, we simulate 500 representative Belgian dwellings in which fossil systems are replaced by controllable electric technologies: space heating via air-to-air heat pumps, domestic hot water via electric boilers, and private transport via electric vehicles with home chargers. Photovoltaic systems are assigned according to current adoption rates.   We evaluate two advanced energy management systems (EMS): model predictive control (MPC) and TreeC, a machine learning method using decision trees to model the EMS strategy. These are compared to a rule-based controller (RBC) that mimics default operation and an MPC with perfect forecasts (MPC P) as an upper bound. Performance for these EMSs is assessed 
    
[^23]: 专家漂移下的学习延迟决策

    Learning to Defer under Expert Drift

    [https://arxiv.org/abs/2611.26440](https://arxiv.org/abs/2611.26440)

    提出EDA-L2D框架，利用序列模型对专家近期决策结果历史进行建模，显式捕捉因疲劳、认知偏差等因素导致的专家可靠性时间漂移，实现自适应的延迟决策策略。

    

    学习延迟决策框架通过将困难或关键决策分配给人类专家，从而保障机器学习和AI系统的安全性。然而，大多数先前的L2D研究都假设专家的可靠性是固定的，忽略了由于疲劳和认知偏差等因素导致的人类表现的时间波动。人类在从放射学、航空和司法决策等高风险场景，到众包图像标注等日常任务的广泛领域中，普遍表现出这种显著且系统性的性能变化。我们提出了专家漂移自适应学习延迟决策，这是一种新颖的专家感知L2D框架，通过序列模型将延迟决策头条件化于专家近期结果的历史记录，从而显式建模这些时间漂移。这种历史感知方法能够实现自适应的延迟策略，预测可靠性的波动并更好地在各决策主体之间分配决策。

    arXiv:2611.26440v1 Announce Type: new  Abstract: The learning to defer (L2D) framework enables safety in machine learning and AI systems by allocating difficult or critical decisions to a human expert. However, most prior work on L2D assumes fixed expert reliability, overlooking temporal fluctuations in human performance due to factors such as fatigue and cognitive biases. Humans commonly exhibit such substantial, systematic performance shifts across a wide range of domains, from high-stakes settings such as radiology, aviation, and judicial decision-making, to everyday tasks such as crowdsourced image annotation. We propose Expert-Drift-Adapted Learning-to-Defer (EDA-L2D), a novel expert-aware L2D framework that explicitly models these temporal shifts by conditioning the deferral head on recent histories of expert outcomes via a sequence model. This history-aware approach enables adaptive deferral policies that anticipate reliability fluctuations and better allocate decisions between 
    
[^24]: 基于特征函数的精确贝叶斯层

    Exact Bayesian layers via characteristic functions

    [https://arxiv.org/abs/2611.26426](https://arxiv.org/abs/2611.26426)

    本文提出CF-Prop方法，利用特征函数无采样地精确计算贝叶斯ReLU层的输出分布（即使在非高斯情形下），并能对高斯、泊松、伽马等多种似然闭式计算证据下界，从而实现高效的变分推断算法CF-VI，在CIFAR-100上取得最低校准误差。

    

    贝叶斯神经网络能够提供有原则的不确定性估计，但使用蒙特卡洛采样进行训练的代价可能很高。无采样变分推断算法使训练贝叶斯神经网络变得可行，但现有方法假设输出分布为高斯分布，这在实际应用问题中可能并不准确。本文提出了CF-Prop，一种不确定性传播方法，它利用特征函数来精确计算贝叶斯ReLU层的输出分布，无需任何采样，即使在非高斯情形下同样适用。由于CF-Prop传播的是特征函数，证据下界——变分训练的目标函数——可以对一大类似然函数（包括高斯、泊松和伽马分布）以闭式形式计算。这使得可以实现一种高效的变分推断算法CF-VI。在CIFAR-100上，CF-VI在所有受测方法中取得了最低的校准误差。

    arXiv:2611.26426v1 Announce Type: cross  Abstract: Bayesian neural networks provide principled uncertainty estimates, but training them with Monte Carlo sampling can be expensive. Sampling-free variational inference algorithms make training Bayesian neural networks feasible, but existing approaches assume a Gaussian output distribution, which can be inaccurate in applied problems. Here, we introduce CF-Prop, an uncertainty propagation method that uses characteristic functions to compute the exact output distribution of a Bayesian ReLU layer, without any sampling, even in non-Gaussian settings. Because CF-Prop propagates characteristic functions, the evidence lower bound---the objective of variational training---can be computed in closed form for a broad class of likelihoods, including the Gaussian, Poisson, and Gamma. This allows to implement an efficient resulting variational inference algorithm, CF-VI. On CIFAR-100, CF-VI achieves the lowest calibration error of any tested method, im
    
[^25]: 将复合材料世界模型与脉冲神经网络记忆模块集成到有限元（FEA）求解器中，用于缺陷与变异性的计算高效建模

    FEA solver integration of a composite World Model with a Spiking Neural Network memory Block for computationally efficient modelling of defects and variabilities

    [https://arxiv.org/abs/2611.26419](https://arxiv.org/abs/2611.26419)

    该论文将基于多模态脉冲Transformer的复合材料世界模型与脉冲神经网络记忆模块集成到有限元求解器中，实现了对纤维增强复合材料中缺陷与材料变异性的计算高效实时建模。

    

    每个纤维增强复合材料结构都具有独特的材料架构。所采用的制造方法以及缺陷的存在会导致局部变异，这要求材料尺度模型能够在结构中每个位置实时解析响应。并行多尺度方案可以提供这种能力，但由于积分点和时间增量的数量随结构尺寸增长，对于大型结构而言计算成本过于高昂。为缓解这些计算限制，代理模型作为材料尺度模型的替代品已被广泛研究。代理模型必须能够表示包含缺陷的任意局部架构，在未知数量的增量过程中追踪不可逆的材料状态，在由自身预测递归驱动时保持稳定，并适配每个积分点可用的内存。本文提出了一种基于多模态脉冲Transformer的复合材料世界模型……（摘要在此处被截断，仅翻译到可见部分）

    arXiv:2611.26419v1 Announce Type: new  Abstract: Each fibre-reinforced composite structure has a unique material architecture. The manufacturing methods used and the presence of defects lead to local variations which require material-scale models that can resolve the response in real time at every location in the structure. Concurrent multi-scale schemes provide this capability, but at a prohibitive computational cost for large structures as the number of integration points and time increments scales with size. Surrogates have been studied extensively as replacements for material-scale models to mitigate these computational limitations. A surrogate must represent an arbitrary local architecture including defects, track the irreversible material state over an unknown number of increments, remain stable when driven recursively by its own predictions, and fit within the memory available at every integration point. Here, a Composites World Model based on a Multi-Modal Spiking Transformer a
    
[^26]: 用于高效推理时对齐的序列化任意时有效检验

    Sequential Anytime Valid Tests for Efficient Inference-Time Alignment

    [https://arxiv.org/abs/2611.26413](https://arxiv.org/abs/2611.26413)

    该论文提出了一种基于奖励加权自一致性分数与 Wald 序列概率比检验的任意时有效停止规则 SPRT-RWSC，能在给定错误容忍度下统计可控地高效判定推理时对齐中的多数答案，避免固定预算造成的计算浪费。

    

    推理时对齐方法，如 Best-of-N (BoN)，通过为每个提示生成多个候选答案并使用奖励模型从中选择，来改进语言模型的输出。这种方法中大量计算成本被浪费：每个提示都被施加了固定的预算，尽管不同提示所需的计算量差异巨大。现有的自适应方法在采样答案看起来已经稳定时就停止，但其阈值是启发式调优的，对提前停止产生的错误缺乏统计控制。我们转而寻求一种任意时有效的停止规则，该规则以错误容忍度 α 作为输入，只有当答案在生成器分布下被认证为占多数（θ_a > 1/2）时才停止。我们使用奖励加权自一致性（RWSC）分数按最终答案聚合候选，并将停止问题转化为对这些聚类结果的序列化检验。将 Wald 序列概率比检验应用于这些聚类，得到了 SPRT-RWSC 方法，其表现良好。

    arXiv:2611.26413v1 Announce Type: new  Abstract: Inference-time alignment methods, such as Best-of-N (BoN), improve language model outputs by generating multiple candidates per prompt and selecting among them using a reward model. Much of this cost is wasted: a fixed budget is applied to every prompt, although the computation required by prompts varies widely. Adaptive methods halt once sampled answers appear settled, but their thresholds are tuned heuristics with no statistical control over early-stopping errors. We instead seek an anytime-valid stopping rule that takes an error tolerance $\alpha$ as input and halts only once an answer is certified to hold a majority ($\theta_a > 1/2$) under the generator's distribution. We aggregate candidates by final answer with a reward-weighted self-consistency (RWSC) score and treat stopping as a sequential test over the resulting clusters. Applying Wald's sequential probability ratio test to these clusters yields SPRT-RWSC, which performs well 
    
[^27]: 如何有效管理住宅系统中的能源：方法的开发与比较

    How to Effectively Manage the Energy in Residential Systems: Development and Comparison of Methods

    [https://arxiv.org/abs/2611.26412](https://arxiv.org/abs/2611.26412)

    本论文通过仿真与真实实验比较了规则、优化和机器学习三类能源管理方法，并提出了TreeC——一种新颖的机器学习方法，能够学习既易于理解又高效的住宅能源管理策略。

    

    从化石燃料的转型给能源管理带来了重大挑战。为了确保可持续的能源未来，关键在于尽可能减少能源消耗，并使电力消耗和储存与低温室气体排放的能源生产同步。本论文探讨了用于管理电池、热泵和其他能源相关系统的多种能源管理方法。论文通过三个仿真案例和一个真实世界实验，比较了基于简单规则、优化以及机器学习方法的能源管理方式。此外，本论文提出了TreeC，这是一种新颖的基于机器学习的方法，能够学习既易于理解又高效的能源管理策略，并适用于多种场景。本论文的三个主要贡献可总结如下：1）所开发的TreeC方法展现出与（原文此处截断）……

    arXiv:2611.26412v1 Announce Type: cross  Abstract: The transition from fossil fuels presents significant challenges in energy management. To ensure a sustainable energy future, it is crucial to reduce energy consumption when possible and to synchronize electricity consumption and storage with the production from sources with low greenhouse gas emissions. This thesis explores a variety of energy management methods used to manage batteries, heat pumps, and other energy-related systems. The thesis presents three simulation cases alongside one real-world experiment to compare energy management done through simple rules, optimization, and machine learning methods. Furthermore, this thesis introduces TreeC, a novel machine learning-based method that learns understandable yet efficient energy management strategies, applicable to many cases.   The three key contributions of this thesis can be summarized as follows: 1) The developed TreeC method demonstrates competitive performance compared to 
    
[^28]: 基于病态随机权重神经网络筛的统计推断

    Statistical Inference with Ill-Conditioned Random Weight Neural Network Sieves

    [https://arxiv.org/abs/2611.26410](https://arxiv.org/abs/2611.26410)

    本文证明了随机权重神经网络筛在维数增大时必然出现病态性（最小特征值消失），并在显式处理该病态性的基础上建立了相应的逼近速率、估计与统计推断理论。

    

    随机权重神经网络（RWNN）是一类浅层网络，其隐藏层参数为随机生成并保持固定，仅对输出系数进行估计。在给定采样参数的条件下，RWNN回归可视为一个具有随机逼近空间的线性筛问题。我们证明，随着维数的增长，这类随机筛无法始终保持一致的良好条件性。对于满足多项式覆盖条件与Lipschitz条件的参数化特征族，我们证明了总体协方差矩阵的特征值具有由参数空间覆盖几何结构决定的多项式上界，从而导致最小特征值趋于消失。我们发展了显式处理这种病态性的逼近、估计与推断理论。在β-Hölder源条件下，我们同时获得了快于蒙特卡洛速率的L²逼近速率，以及基于……（原文摘要在此处截断）的新L^∞逼近界。

    arXiv:2611.26410v1 Announce Type: cross  Abstract: Random weight neural networks (RWNNs) are shallow networks whose hidden-layer parameters are randomly generated and then kept fixed, while only the output coefficients are estimated. Conditional on the sampled parameters, RWNN regression is a linear sieve problem with random approximation space. We show that such random sieves cannot remain uniformly well-conditioned as their dimension grows. For parametrized feature families satisfying polynomial covering and Lipschitz conditions, the eigenvalues of the population covariance matrix are proven to have polynomial upper bounds determined by the covering geometry of the parameter space, leading to a vanishing smallest eigenvalue. We develop approximation, estimation, and inference theory that explicitly treats the ill-conditioning. Under a $\beta$-H\"older source condition, we obtain both faster-than-Monte-Carlo $L^2$ approximation rates and new $L^\infty$ approximation bounds based on st
    
[^29]: 来自非高斯源的流匹配：高效采样理论

    Flow Matching from Non-Gaussian Sources: A Theory of Efficient Sampling

    [https://arxiv.org/abs/2611.26407](https://arxiv.org/abs/2611.26407)

    本文为从非高斯源分布（如重尾和多峰分布）出发的流匹配建立了首个非渐近收敛理论，证明欧拉离散化采样器在温和条件下仅需 $\widetilde{O}(Ld\mathsf{TotVar}_{\mathsf{base}}/\varepsilon)$ 次迭代即可达到 $\varepsilon$ 级总变差误差。

    

    流匹配是一种多功能的生成建模框架，它将一个源分布传输至目标数据分布。其关键优势之一在于源分布选择的灵活性，例如可以使用重尾分布和多峰分布作为源。然而，针对非高斯源的有限步采样保证仍然十分有限。本文为采用线性插值的流匹配的欧拉离散化建立了非渐近收敛理论。假设定义在 $\mathbb{R}^d$ 上的源分布具有处处为正、连续可微的密度函数、有限的总方差 $\mathsf{TotVar}_{\mathsf{base}}$，以及 $L$-Lipschitz 的得分函数。在使用总体速度场和适当的时间网格时，只要目标分布满足温和的二阶矩条件，该采样器即可在 $\widetilde{O}(Ld\mathsf{TotVar}_{\mathsf{base}}/\varepsilon)$ 次迭代内实现至多 $\varepsilon$ 的总变差误差。这些假设涵盖……（摘要原文在此处截断）

    arXiv:2611.26407v1 Announce Type: cross  Abstract: Flow matching is a versatile framework for generative modeling that transports a source distribution to a target data distribution. A key advantage is its flexibility in the choice of source distribution, such as heavy-tailed and multimodal sources. Yet finite-step sampling guarantees for non-Gaussian sources remain limited. We establish a nonasymptotic convergence theory for Euler discretization of flow matching with linear interpolation. Suppose the source distribution on $\mathbb{R}^d$ has an everywhere positive, continuously differentiable density, finite total variance $\mathsf{TotVar}_{\mathsf{base}}$, and an $L$-Lipschitz score. Using the population velocity field and a suitable time grid, this sampler achieves total variation error at most $\varepsilon$ in $\widetilde{O}(Ld\mathsf{TotVar}_{\mathsf{base}}/\varepsilon)$ iterations, provided the target distribution satisfies a mild second-moment condition. These assumptions cover,
    
[^30]: 仿真预训练自编码器与PCA在多模光纤传输矩阵压缩中的对比

    Simulation-pretrained autoencoders versus PCA for compressing multimode fibre transmission matrices

    [https://arxiv.org/abs/2611.26391](https://arxiv.org/abs/2611.26391)

    该研究表明，基于仿真预训练并仅用两个实测传输矩阵适配的置换等变自编码器，在62倍压缩下即可实现98%的理想聚焦增强，且对平移和缩放具有鲁棒性，但作为纯压缩器其精度仍比PCA低约十倍。

    

    多模光纤在测量其传输矩阵（TM）之后，可以通过一根细而柔软的探针进行成像，但传输矩阵数据庞大，且每当光纤移动时都必须重新测量。一个紧凑的可行传输矩阵模型将使存储和重新估计它们的成本更低，因此我们探讨传输矩阵能以多紧凑的方式表示，以及需要多少个实测传输矩阵。在这里，我们将在仿真光纤上预训练、再适配到实测传输矩阵的置换等变自编码器与主成分分析（PCA）进行了基准对比。在62倍压缩下，仅用两个实测传输矩阵对预训练自编码器进行适配，就能充分重构50微米光纤的传输矩阵，使光聚焦达到理想增强效果的98%；并且当传输矩阵平移25个像素或缩放10%时，它仍能保持准确，而线性基在这种情况下会迅速失效。然而，它并不是更好的压缩器：当在全部163个真实传输矩阵上拟合时，PCA的精度约为其十倍（误差为1.1%，而自编码器为11.9%）。

    arXiv:2611.26391v1 Announce Type: cross  Abstract: Multimode optical fibres can image through a thin, flexible probe once their transmission matrix (TM) has been measured, but a TM is large and must be re-measured whenever the fibre moves. A compact model of feasible TMs would make storing and re-estimating them cheaper, so we ask how compactly TMs can be represented, and from how few measured TMs. Here we benchmark a permutation-equivariant autoencoder, pretrained on simulated fibres and adapted to measured TMs, against principal component analysis (PCA). At 62x compression, adapting the pretrained autoencoder with just two measured TMs reconstructs the TM of a 50um fibre well enough to focus light at 98% of the ideal enhancement, and it stays accurate when the TMs are shifted by 25 pixels or zoomed by 10%, which quickly degrades a linear basis. It is not, however, a better compressor: PCA is about ten times more accurate when fitted on all 163 real TMs (1.1% against 11.9% error on he
    
[^31]: 一种用于点云的完全几何消息传递架构

    A Complete Geometric Message-Passing Architecture for Point Clouds

    [https://arxiv.org/abs/2611.26384](https://arxiv.org/abs/2611.26384)

    该论文提出了一种新颖而简单的等变消息传递架构用于点云，并从理论上证明了其完全性，无需依赖计算密集的高阶旋转群表示或高阶Weisfeiler-Leman测试即可区分任意非对称等价的输入。

    

    三维点云是一种基础的数据模态，为了尊重其关于刚体运动和置换的自然对称性，研究者们已经开发了大量的机器学习架构。比较这些在复杂度与计算成本上差异巨大的架构的一种原则性方法，是考察它们的表达能力。具体而言，如果一个架构能够区分任何两个不由其所遵循的对称性相互关联的输入，那么该架构就是完全的。众所周知，简单的基于距离的消息传递网络是不完全的，而现有的完全架构通常依赖于计算代价高昂的机制，例如旋转群的高阶表示或高阶Weisfeiler-Leman型测试。在本工作中，我们设计了一种新颖、简单、等变的点云消息传递架构，并证明了它是完全的。我们还通过实验补充了理论结果，实验表明我们的架构能够分离……（摘要在此处截断）

    arXiv:2611.26384v1 Announce Type: new  Abstract: Three-dimensional point clouds are a fundamental data modality, and a large family of machine learning architectures has been developed to respect their natural symmetries to rigid motions and permutations. A principled way to compare these architectures, which vary widely in complexity and cost, is through their expressive power. That is, an architecture is complete if it can distinguish any two inputs not related by the symmetries it respects. Simple distance-based message-passing networks are known to be incomplete, and while there exist complete alternatives, they typically rely on computationally intensive mechanisms like high-order representations of the rotation group or higher-order Weisfeiler-Leman-type tests.   In this work, we devise a novel, simple, equivariant message-passing architecture for point clouds, and prove that it is complete. We complement our theoretical result with experiments showing that our architecture separ
    
[^32]: 来自多机制因果发现基准的证据：UCMT算法、比较性能与受控消融研究

    Evidence from a Multi-Regime Causal Discovery Benchmark: UCMT Algorithms, Comparative Performance, and Controlled Ablation Studies

    [https://arxiv.org/abs/2611.26383](https://arxiv.org/abs/2611.26383)

    该论文通过预注册的多机制因果发现基准比较与受控消融研究，证明UCMT v2 Global在部分真实、半合成及高斯网络场景下有向F1优于NOTEARS与DAGMA，而基于BOSS的UCMT v4在CSuite上带来配对级显著改进，并揭示了影响算法表现的内部阶段。

    

    我们对UCMT因果发现算法家族与既有替代算法在异构基准机制上进行了内部预注册评估，并辅以一项单独预注册的、针对UCMT v2 Global各组件的受控研究。本工作的贡献兼具实证与方法论意义：对三个仍在积极开发中的算法配置进行了家族分层比较，并就哪些内部阶段会影响其结果提供了有针对性的证据。在主基准测试中，UCMT v2 Global在选定的真实、半合成及高斯网络比较的配对运行层面上，其有向F1分数高于NOTEARS和DAGMA，但在其他设置中则落后于领先的置换搜索方法。基于BOSS的UCMT v4在CSuite上相对BOSS取得了有向F1的改进（平均配对差异+0.279；配对层面Holm p < 0.001），但数据集层面的调整后结果并不显著（p = 0.188）。在另一项固定…（原文摘要至此截断）

    arXiv:2611.26383v1 Announce Type: new  Abstract: We present an internal, pre-registered evaluation of the UCMT causal discovery algorithm family alongside established alternatives across heterogeneous benchmark regimes, complemented by a separately pre-registered, controlled study of UCMT v2 Global components. The contribution is both empirical and methodological: a family-stratified comparison of three actively developed algorithm configurations and targeted evidence about which internal stages affect their outcomes. In the main benchmark, UCMT v2 Global has higher directed F1 than NOTEARS and DAGMA in selected real, semi-synthetic and Gaussian-network comparisons at the paired-run level, while trailing leading permutation-search methods in other settings. The BOSS-based UCMT v4 produces a CSuite-specific directed-F1 improvement over BOSS (mean paired difference +0.279; pair-level Holm p < 0.001), but the dataset-level adjusted result is not significant (p = 0.188). In a separate fixe
    
[^33]: 从基准测试到实验台：双向药物-靶点筛选中的偏好、一致性与选择性

    From Benchmarks to the Bench: Preference, Correspondence and Selectivity in Bidirectional Drug-Target Screening

    [https://arxiv.org/abs/2611.26369](https://arxiv.org/abs/2611.26369)

    该研究对13个已发布的药物-靶点相互作用模型进行了大规模双向基准比较，发现不同模型给出的Top-10推荐列表在77.8%（分子）和70.1%（蛋白）的对比中完全不相交，且系统间的一致性主要反映对相同候选物的重复偏好而非真实的对应关系，实验验证进一步表明仅部分模型能够富集活性化合物且选择性有限。

    

    为已知靶点识别活性分子以及为现有药物寻找潜在新靶点，是药物发现与药物重定位的核心任务。药物-靶点相互作用（DTI）模型能够对海量药物-靶点对进行排序，但有限的实验验证能力使实际测试只能局限于少数候选对象。我们在711个FDA批准药物分子结构和614种人类蛋白上比较了13个已发布系统的双向推荐结果：对于相同的靶点或药物，模型两两对比中分别有77.8%和70.1%的情况产生了完全不相交的Top-10分子或蛋白列表；即使在推荐结果存在重叠时，某些系统之间的一致性也主要源于对相同候选物的重复偏好。基于79个分子和65种激酶的已发表抑制活性测量数据的分析显示，部分系统能够富集活性候选物，但当推荐结果与不同药物或……（原摘要在此处截断）匹配时，其命中率变化甚微。

    arXiv:2611.26369v1 Announce Type: new  Abstract: Identifying active molecules for known targets and potential targets for existing drugs is central to drug discovery and repurposing. Drug-target interaction (DTI) models can rank vast numbers of drug-target pairs, but limited experimental capacity restricts testing to a small number of candidates. We compared the bidirectional recommendations of 13 released systems across 711 molecular structures of FDA-approved drugs and 614 human proteins. For the same target or drug, pairs of models produced completely disjoint Top-10 molecule and protein lists in 77.8% and 70.1% of comparisons, respectively. Even where recommendations overlapped, agreement between some systems largely reflected recurring preferences for the same candidates. Published inhibition measurements for 79 molecules and 65 kinases revealed that some systems enriched active candidates, yet their hit yields changed little when recommendations were matched to different drugs or
    
[^34]: HiSA：对大语言模型中RoPE注意力的隐藏维度进行稀疏化

    HiSA: Sparsifying the Hidden Dimension of RoPE Attention in LLMs

    [https://arxiv.org/abs/2611.26363](https://arxiv.org/abs/2611.26363)

    提出HiSA方法，基于RoPE结构将注意力点积在隐藏（特征）维度上进行稀疏化，仅需计算约三分之一的查询-键点积就能在LLM训练中提升而非损害性能。

    

    大语言模型（LLM）中的注意力机制通过完整的点积将每个查询与每个键进行评分。我们研究了为了让注意力机制有效，这个点积是否必须被完整计算。我们提出了隐藏稀疏注意力，它保持查询和键的投影不变，但仅计算部分点积，遵循一个由token位置设定的静态、无参数的规则：注意力在隐藏（特征）维度上变得稀疏——这是一个相比token稀疏性被探索得少得多的维度——而不是像几乎所有稀疏注意力方法那样在token维度上进行稀疏化。基于旋转位置嵌入的结构，我们将键和查询分解为在计算注意力时可以省略的部分。通过使用HiSA从头训练LLM，我们发现这种部分点积不仅不会损害性能，反而相比完整点积提升了性能，同时只需计算约三分之一的查询-键乘积。

    arXiv:2611.26363v1 Announce Type: cross  Abstract: Attention in large language models (LLMs) scores every query against every key with a full dot product. We investigate whether this dot product must be computed completely for attention to be effective. We propose Hidden Sparse Attention (HiSA), which keeps the query and key projections intact but only computes a partial dot product, following a static, parameter-free rule set by the token positions: attention becomes sparse in the hidden (feature) dimension, an axis much less explored than token sparsity, rather than over tokens as in almost all sparse attention methods. Based on the structure of rotary position embeddings (RoPE), we decompose keys and queries into parts which can be omitted when computing attention. By training LLMs from scratch with HiSA, we find that this partial dot product does not hurt but improves performance compared with the complete product while computing only about a third of the query-key products. During
    
[^35]: 从上下文中学习：跳跃扩散过程中的留一预测

    Learning from Context: Leave-One-Out Prediction in Jump Diffusions

    [https://arxiv.org/abs/2611.26359](https://arxiv.org/abs/2611.26359)

    该论文提出在跳跃扩散模型中采用留一（LOO）预测——每个坐标仅依据其他坐标的破坏状态进行预测——并证明通过解析似然补全可从留一概率构造漂移项，在保持规定边缘分布的同时无需额外网络评估，从而促使模型真正学习坐标间的依赖关系。

    

    连续扩散模型通过嵌入空间中的随机动力学生成离散数据。我们考虑了一类跳跃扩散过程，它还会将坐标重置为类别值的嵌入表示。坐标自身被破坏的状态可能使预测变得容易，这可能减弱模型学习坐标间依赖关系的动力。我们研究了留一预测，其中每个坐标仅由其他坐标的被破坏状态来进行预测。我们证明，在给定假设下，具有完全坐标重置和状态无关时钟的跳跃扩散过程，当重置目标使用精确的留一概率且漂移使用完整的端点后验时，能够保持规定的边缘分布。我们进一步证明，该漂移可以通过解析似然补全从留一概率构造出来，而无需额外的网络评估。我们在Text8和LM1B文本数据集上评估了无约束预测、留一预测和补全预测。

    arXiv:2611.26359v1 Announce Type: new  Abstract: Continuous diffusion models generate discrete data through stochastic dynamics in an embedding space. We consider jump diffusions that additionally reset coordinates to embeddings of categorical values. A coordinate's own corrupted state can make prediction easy, potentially reducing the pressure to learn dependencies between coordinates. We study leave-one-out (LOO) prediction, in which each coordinate is predicted from the corrupted states of the other coordinates. We show that, under stated assumptions, jump diffusions with complete coordinate resets and state-independent clocks preserve the prescribed marginals when reset destinations use exact LOO probabilities and the drift uses the full endpoint posterior. We further show that this drift can be constructed from LOO probabilities by analytic likelihood completion without additional network evaluations. We evaluate unrestricted, LOO, and completed predictions on Text8 and LM1B text 
    
[^36]: Mamba流匹配神经过程：面向不规则观测空间场的线性时间推断

    Mamba Flow Matching Neural Processes: Linear-Time Inference for Irregularly Observed Spatial Fields

    [https://arxiv.org/abs/2611.26355](https://arxiv.org/abs/2611.26355)

    该论文提出MambaFlowNP，将流匹配与Mamba状态空间模型引入神经过程，实现对不规则观测空间场的线性时间插值推断，突破了自注意力机制 O(K²) 计算开销对可处理点数的限制。

    

    从稀疏且位置不规则分布的观测中对空间场（例如每日最高温度）进行插值，在环境科学与健康科学领域是一个反复出现的问题。标准的高斯过程克里金法需要对每个新场重新拟合超参数，并求解复杂度为 O(N³) 的线性方程组。当观测到许多相关的场时，例如来自同一传感器网络的每日地图，神经过程（NPs）则改为从过去的场中学习插值映射，并通过一次前向传播预测新的场。联合预测型变体（即对整个场进行采样，而非在每个位置独立取值）通常在上下文点与查询点上使用自注意力机制，其计算开销在合并点数 K 上为 O(K²)，从而将实际应用限制在几千个点以内。状态空间模型提供了一种线性时间的替代方案，但直接的状态空间适配方式会在读取查询点之前将上下文压缩为单一隐藏状态，减少了解码器所能利用的信息。我们提出了 MambaFlowNP，一种……

    arXiv:2611.26355v1 Announce Type: new  Abstract: Interpolating a spatial field, such as daily maximum temperature, from sparse, irregularly located observations recurs across the environmental and health sciences. Standard Gaussian-process kriging refits hyperparameters and solves an $O(N^3)$ system for every new field. When many related fields are observed, such as daily maps from one sensor network, neural processes (NPs) instead learn the interpolation map from past fields and predict a new field in one forward pass. Joint-predictive variants, which sample whole fields rather than independent per-location values, typically use self-attention over context and query points, costing $O(K^2)$ in the combined point count $K$ and limiting practical use to a few thousand points. State-space models offer a linear-time alternative, but a direct state-space adaptation compresses the context into one hidden state before reading queries, reducing what the decoder sees. We propose MambaFlowNP, a
    
[^37]: TRCF：面向不规则多变量时间序列的时序-参考互补预测

    TRCF: Temporal-Reference Complementary Forecasting for Irregular Multivariate Time Series

    [https://arxiv.org/abs/2611.26351](https://arxiv.org/abs/2611.26351)

    TRCF框架通过将紧凑的通道独立预测器与互补的通道依赖分支相结合，利用ANCHOR和BRIDGE两个模块在保留不规则观测历史的同时融合跨变量信息，从而改进不规则多变量时间序列的预测。

    

    不规则多变量时间序列预测支持医疗健康监测、环境分析和活动识别等应用。通道独立（CI）方法能够捕捉每个变量的时间模式，但不会显式利用来自其他变量的观测信息。通道依赖（CD）建模利用变量之间的关系，然而异步测量使这些关系难以学习。因此，在保留每个变量不规则历史的同时融合跨变量信息仍然是一个关键挑战。我们提出了时序-参考互补预测（TRCF），这是一个用互补的通道依赖分支来增强紧凑通道独立预测器的框架。其中，ANCHOR在共享时间位置上聚合观测值和时间戳，保留观测支撑以形成紧凑的通道独立特征；BRIDGE通过历史补全构建互补特征，首先估计每个变量的历史值……

    arXiv:2611.26351v1 Announce Type: new  Abstract: Irregular multivariate time series forecasting supports healthcare monitoring, environmental analysis, and activity recognition. Channel-independent (CI) approaches capture each variable's temporal patterns but do not explicitly use observations from other variables. Channel-dependent (CD) modeling exploits relationships among variables, yet asynchronous measurements make these relationships difficult to learn. Incorporating cross-variable information while preserving each variable's irregular history therefore remains a key challenge. We propose Temporal-Reference Complementary Forecasting (TRCF), a framework that augments a compact CI forecaster with a complementary CD branch. ANCHOR aggregates observed values and timestamps at shared temporal locations, retaining observation support to form compact CI features. BRIDGE constructs complementary features through history completion. It first estimates each variable's historical values fro
    
[^38]: Perfumer：一种基于大语言模型的智能体框架，用于加速第一性原理计算中的结构弛豫与电子能带预测

    Perfumer: A Large Language Model-based Agentic Framework Accelerating Structural Relaxation and Electronic Band Prediction for First-Principle Calculation

    [https://arxiv.org/abs/2611.26349](https://arxiv.org/abs/2611.26349)

    Perfumer是一个基于大语言模型的智能体框架，通过结构预弛豫使第一性原理计算实现2.75倍加速，其能带预测模型可快速准确地预测电子能带（中位误差0.157 eV，每材料仅约97毫秒）。

    

    第一性原理结构弛豫与能带计算需要大量的计算资源和专业知识。本文提出了Perfumer，一个基于大语言模型的智能体框架，它将Quantum ESPRESSO与两个预测模型相结合。在所测试的结构上，先使用Perfumer-relax模型进行预弛豫、再经QE精细优化，相比单独使用QE实现了2.75倍的总体加速。Perfumer-band模型将周期图神经网络与潜在的厄米特谱头相结合，可在任意请求的波矢处返回12条能带能量，在无需构建哈密顿量的情况下保持倒格子周期性和时间反演对称性。该模型在125,153个自建的能带结构数据集上训练，在来自涵盖85种元素的留出成分族的2,607个测试材料（包括镧系、锕系和磁性化合物）上，单材料平均绝对误差的中位数达到0.157 eV，每个材料仅需约97毫秒……

    arXiv:2611.26349v1 Announce Type: cross  Abstract: First-principles structural relaxation and band calculations require substantial computation and expertise. Here we present Perfumer, a large language model-based agentic framework that couples Quantum ESPRESSO with two predictive models. Across tested structures, pre-relaxation with Perfumer-relax model followed by QE refinement yielded a 2.75-fold overall speedup over QE alone. Perfumer-band model combines a periodic graph neural network with a latent Hermitian spectral head returns 12 band energies at any requested wavevector, preserving reciprocal lattice periodicity and time-reversal symmetry without constructing Hamiltonian. Trained on 125,153 self-curated band structures, Perfumer-band model reaches a median per-material mean absolute error of 0.157 eV on 2,607 test materials, including lanthanide, actinide and magnetic compounds, from held-out composition families spanning 85 elements, taking about 97 ms per material on two con
    
[^39]: OceanCoder：面向海洋工程自动化仿真建模的验证引导大语言模型

    OceanCoder: A Verification-Guided Large Language Model for Automated Simulation Modelling in Ocean Engineering

    [https://arxiv.org/abs/2611.26333](https://arxiv.org/abs/2611.26333)

    OceanCoder 通过可扩展数据流水线和三层验证框架为 GRPO 提供分级奖励，构建了一个能自动生成满足工程要求仿真模型的海洋工程专用大语言模型。

    

    数值仿真在海洋工程中十分重要，但模型构建需要专家知识。大语言模型可以支持自动化建模，然而代码语法正确和能够成功执行并不代表生成的模型和结果满足工程要求。我们提出了 OceanCoder，一个基于 Qwen3-8B、面向海洋工程自动化仿真建模的大语言模型。为应对专业化的编程需求和训练数据稀缺的问题，我们开发了一个可扩展的数据构建流程，生成了涵盖 12 类工程任务的 12,000 条经过验证的监督微调（SFT）样本。一个三层验证框架用于检查模型输入是否符合工程参数与任务约束、验证所需仿真分析是否完成，并评估计算结果的工程合理性。其评估结果为群体相对策略优化（GRPO）提供分级奖励，引导……（原文在此处截断）

    arXiv:2611.26333v1 Announce Type: new  Abstract: Numerical simulation is important in ocean engineering, but model construction requires expert knowledge. Large language models support automated modelling, but code syntax and successful execution alone do not establish whether generated models and results meet engineering requirements. We present OceanCoder, a Qwen3-8B-based large language model for automated simulation modelling in ocean engineering. To address specialised programming requirements and scarce training data, we develop an extensible data construction pipeline producing 12,000 verified supervised fine-tuning (SFT) examples across 12 engineering task categories. A three-layer verification framework checks model inputs against engineering parameters and task constraints, verifies completion of required simulation analyses, and assesses the engineering plausibility of computed outputs. Its outcomes provide graded rewards for group relative policy optimisation (GRPO), guidin
    
[^40]: 分层数据流的保形选择

    Conformal Selection for Hierarchical Streaming Data

    [https://arxiv.org/abs/2611.26328](https://arxiv.org/abs/2611.26328)

    该论文提出了面向分层数据流的无分布在线保形选择方法OHS及其带反馈变体OHSF，通过构建组间和组内的基于秩的e值并结合在线e-BH程序，将在线保形选择框架扩展到超越独立同分布/可交换数据假设的分层结构场景。

    

    许多现实世界的任务需要在感兴趣的结果被揭示之前，对是否选择候选对象做出序贯决策。在这项工作中，我们研究了分层数据背景下的在线选择问题，其中候选对象依次来自不同的总体。这将现有的在线保形选择框架扩展到了标准的独立同分布/可交换数据假设之外。主要挑战在于如何在分层数据结构下为每个候选对象序贯地构建有效的统计量。为此，我们首先开发了在线分层选择（OHS），这是一种无分布方法，能够对分层数据流进行有原则的候选选择。具体而言，我们提出了为组间和组内样本构建基于秩的e值的方法，并将平均e值应用于在线e-BH程序。随后，我们开发了带反馈的OHS（OHSF），该方法利用反馈信息提升了选择的性能。

    arXiv:2611.26328v1 Announce Type: cross  Abstract: Many real-world tasks involve making sequential decisions about whether to select candidates before their responses of interest are revealed. In this work, we address the online selection problem in the context of hierarchical data, where candidates arrive sequentially from different populations. This extends existing online conformal selection frameworks beyond the standard i.i.d./exchangeable data assumption. The primary challenge lies in constructing valid statistics for each candidate sequentially under the hierarchical data structure. To this end, we first develop Online Hierarchical Selection (OHS), a distribution-free method that enables principled candidate selection for hierarchical data streams. Specifically, we introduce methods to construct rank-based e-values for both between-group and within-group samples, and apply averaged e-values to the online e-BH procedure. We then develop OHS with feedback (OHSF), which boosts the 
    
[^41]: 基于STL衍生图与零件级描述符的激光粉末床熔融零件表面位移场预测

    Predicting Surface Displacement Fields of Laser Powder Bed Fusion Parts Using STL-Derived Graphs and Part-Level Descriptors

    [https://arxiv.org/abs/2611.26323](https://arxiv.org/abs/2611.26323)

    该研究提出一种双分支图注意力网络，直接从STL衍生图结合零件级描述符预测激光粉末床熔融零件的三维表面位移场，R²达0.946、平均绝对误差仅0.0178毫米，为替代计算昂贵的热力学模拟提供了高效的零件变形预测方案。

    

    在激光粉末床熔融（LPBF）中，制造前预测零件变形仍然具有挑战性，而热力学模拟在重复设计评估中计算成本高昂。本研究提出了一种双分支图注意力网络（GAT），直接从STL衍生的图中预测三维表面位移。第一分支学习加密后的STL表面网格中连接顶点之间的几何关系，第二分支则融入了12个关于质量分布、基板接触、悬垂结构、尺寸和形状的零件级描述符。来自ANSYS Additive Print模拟的位移标签通过高斯核加权平均映射到STL顶点上。在117个零件上按零件划分进行五折交叉验证评估，该模型在位移幅值预测上达到0.946的R²和0.0178 mm的平均绝对误差。与逐节点多层感知机相比，GAT将R²从0.91提升至更高水平。

    arXiv:2611.26323v1 Announce Type: new  Abstract: Predicting part distortion before fabrication remains challenging in laser powder bed fusion (LPBF), while thermomechanical simulation is computationally expensive for repeated design evaluations. This study presents a two-branch graph attention network (GAT) to predict three-dimensional surface displacement directly from STL-derived graphs. The first branch learns geometric relationships among connected vertices of a densified STL surface mesh, while the second incorporates 12 part-level descriptors of mass distribution, build-plate contact, overhang, size, and shape. Displacement labels from ANSYS Additive Print simulations are mapped onto STL vertices using Gaussian-kernel weighted averaging. Evaluated on 117 parts using five-fold cross-validation split by part, the model achieves an R^2 of 0.946 and a mean absolute error of 0.0178 mm for displacement magnitude. Relative to a node-wise multilayer perceptron, GAT improves R^2 from 0.91
    
[^42]: 预测编码动力学的控制论近似

    A Control-Theoretic Approximation to Predictive Coding Dynamics

    [https://arxiv.org/abs/2611.26319](https://arxiv.org/abs/2611.26319)

    该论文证明了深度反馈控制（DFC）是预测编码（PC）动力学在不动点附近的局部近似（称为 Approx PC），在浅层线性网络中二者完全等价，且通过重新线性化反馈控制器和分步逼近输出目标可以缓解由深度和非线性带来的偏差，从而厘清了两种生物学可信局部学习框架之间的关系。

    

    反向传播仍然是训练深度神经网络的主流方法，但其对非局部计算的依赖促使研究者寻找生物学上合理的替代方案。预测编码与深度反馈控制这两个著名框架为局部学习提供了不同的途径，但二者之间的关系尚不明确。本研究表明，DFC 是预测编码动力学在不动点附近的局部近似：在浅层线性网络中二者完全等价，而随着网络深度增加、非线性的引入以及偏离平衡点距离的增大，会产生系统性偏差。我们对这种近似（称为 Approx PC）在活动动力学方面偏离 PC 的方式进行了刻画，并证明可以通过对反馈控制器重新线性化以及通过一系列中间推动逐步逼近输出目标来减轻这些偏差。在 MNIST 和 FashionMNIST 数据集上训练的网络中，Approx PC 达到了（与其他方法相当的）任务性能（摘要原文在此处截断）。

    arXiv:2611.26319v1 Announce Type: cross  Abstract: Backpropagation remains the dominant method for training deep neural networks, but its reliance on non-local computations has motivated the search for biologically plausible alternatives. Two prominent frameworks, predictive coding (PC) and deep feedback control (DFC), offer distinct approaches to local learning, yet their relationship remains unclear. Here, we show that DFC arises as a local approximation to predictive coding dynamics near fixed points, with exact equivalence in shallow linear networks and systematic deviations arising with depth, nonlinearity, and distance from equilibrium. We characterize how this approximation, which we call Approx PC, deviates from PC in its activity dynamics, and show that these deviations can be mitigated by relinearizing the feedback controller and approaching the output target through a sequence of intermediate nudges. In networks trained on MNIST and FashionMNIST, Approx PC reaches task perfo
    
[^43]: 面向基于阻变存储器（ReRAM）的存内处理CNN加速器的可扩展故障脆弱性分析

    Scalable fault vulnerability analysis for ReRAM-based process-in-memory CNN accelerators

    [https://arxiv.org/abs/2611.26307](https://arxiv.org/abs/2611.26307)

    该论文提出了一种可扩展的故障脆弱性分析方法，用于高效评估基于ReRAM的存内处理CNN加速器中永久性故障对推理精度的影响，克服了传统故障注入方法耗时且计算成本过高的缺点。

    

    基于阻变随机存取存储器（ReRAM）的存内处理（PIM）加速器，凭借模拟计算、高密度、低泄漏和非易失性等特性，为神经网络中的并行矩阵-向量乘法（VMM）提供了高效的平台。然而，这类加速器易受永久性故障的影响，例如高阻态卡死和低阻态卡死故障，这些故障会永久性地破坏映射到ReRAM单元电导值上的卷积神经网络（CNN）权重，从而降低推理精度，并在安全关键应用中导致系统不可靠。故障注入是一种常用的神经网络脆弱性评估技术，它通过系统性地向单个参数注入故障并测量由此产生的精度下降来评估网络。尽管这种方法精确，但其极其耗时且计算开销巨大，尤其是对于大型网络。

    arXiv:2611.26307v1 Announce Type: cross  Abstract: Resistive random-access memory (ReRAM)-based processing-in-memory (PIM) accelerators offer a highly efficient platform for parallel matrix-vector multiplications (VMM) in neural networks, leveraging analog computation, high density, low leakage, and non-volatility. However, their susceptibility to permanent faults, such as stuck-at-high (\textit{SaH}) and stuck-at-low (\textit{SaL}) resistance states, permanently corrupts the convolutional neural networks (CNNs) weights mapped onto ReRAM cell conductance values, thereby degrading inference accuracy and leading to system unreliability in safety-critical applications. Fault injection is a commonly used technique to assess the vulnerability of neural networks by systematically injecting faults into individual parameters and measuring the resulting accuracy degradation. Although this method is accurate, it is prohibitively time-consuming and computationally expensive, especially for large-
    
[^44]: SKILL.PNG：在像素空间中学习智能体技能

    SKILL.PNG: Learning Agent Skills in Pixel Space

    [https://arxiv.org/abs/2611.26296](https://arxiv.org/abs/2611.26296)

    提出Skill.png方法，通过梯度下降优化图像像素来学习可跨VLM通用迁移的智能体技能，在大多数基准设置中超越最先进的文本技能，且token开销极低。

    

    AI智能体通常使用技能（即置于其上下文中的自然语言指令）来针对新任务和新领域进行专门化。然而，文本技能的离散特性阻碍了连续优化。另一方面，基于梯度的方法（如微调）通常是模型特定的，导致技能无法在不同智能体之间迁移。鉴于当今许多智能体都构建在视觉语言模型（VLM）之上，我们可以通过引入Skill.png来克服这些限制：这是一种获取类技能图像输入的方法，其像素通过冻结的VLM视觉编码器使用梯度下降进行优化。图像技能与任何VLM兼容，并且单张图像可以针对模型集合进行联合优化。我们证明，Skill.png在16个设置中的9个（涵盖4个VLM和4个基准测试）中显著优于使用前沿LLM优化的最先进文本技能，同时仅需64到256个图像token。

    arXiv:2611.26296v1 Announce Type: new  Abstract: AI agents are commonly specialized for new tasks and domains using skills, i.e. natural language instructions placed in their context. However, the discrete nature of textual skills prevents continuous optimization. Gradient-based approaches such as finetuning, on the other hand, are typically model-specific, preventing skill transfer across different agents. Since many agents today are built on vision-language models (VLMs), we can address these limitations by introducing Skill.png: a method for obtaining skill-like image inputs whose pixels are optimized with gradient descent through the vision encoder of a frozen VLM. Image skills are compatible with any VLM, and a single image can be optimized jointly for an ensemble of models. We show that Skill.png significantly outperforms state-of-the-art textual skills optimized using frontier LLMs in 9 of 16 settings (across 4 VLMs and 4 benchmarks) while requiring only 64 to 256 image tokens i
    
[^45]: 谁动了我的 $Z$ 玻色子？揭示神经网络内部的共振特征

    Who Moved My $Z$ Boson? Revealing Resonance Features Inside a Neural Network

    [https://arxiv.org/abs/2611.26291](https://arxiv.org/abs/2611.26291)

    该研究在基于真实LHC数据训练的生成式对撞机物理模型ShellFlow中，通过梯度与模式识别方法发现了代表 $Z$ 玻色子等共振的内部可解释特征，并证明可以操控这些特征来改变 $Z$ 玻色子的质量和峰宽而不影响不变质量谱的其余部分。

    

    机械可解释性领域的方法已经在语言和图像模型中揭示出与人类对齐的概念。在本文中，我们在一个基于真实LHC数据训练的神经生成式对撞机物理模型中搜索物理学概念。我们在ShellFlow内部发现了代表共振的特征，例如 $Z$ 玻色子、$J/\psi$ 和 $\Upsilon$。该网络从未被告知哪些μ子来自共振，它仅被训练用于生成末态粒子的运动学量。我们利用这些特征来改变 $Z$ 玻色子的某些属性：我们将其质量在约80至105 GeV之间进行调整，而从神经网络中移除该特征则会产生更宽的 $Z$ 峰。在这两种情况下，不变质量谱的其余部分基本保持不变。该特征由神经网络的梯度构建，其选择性则通过模式识别方法构建。一个基于未标注（数据训练的生成式粒子物理模型……）※摘要在此处被截断

    arXiv:2611.26291v1 Announce Type: cross  Abstract: Methods used in mechanistic interpretability have revealed human-aligned concepts inside language and image models. In this paper, we search for physics concepts inside a neural a generative collider physics model trained on real LHC data (ShellFlow). We discover features inside ShellFlow that represent resonances, such as the $Z$, $J/\psi$, and $\Upsilon$. The network was never told which muons come from a resonance, it was only trained to generate the kinematics of the final-state particles. We use these features to alter aspects of the $Z$ boson: we change its mass between $\sim80$ and $105$ GeV, and removing such a feature from the neural network produces a broader $Z$ peak. In both cases the rest of the invariant mass spectrum remains mostly unchanged. The feature was built from the gradients of the neural network, and its selectivity was built using pattern recognition methods. A generative particle physics model trained on unlab
    
[^46]: 大语言模型训练数据审计的统计框架

    A Statistical Framework for Training Data Auditing of Large Language Models

    [https://arxiv.org/abs/2611.26264](https://arxiv.org/abs/2611.26264)

    本文提出了一个统一的统计框架，用于大语言模型训练数据审计，涵盖文档级成员推断、目标数据集全局审计和可疑子集识别，能够有效聚合文档内局部成员信号并进行整体评估。

    

    训练数据审计旨在判断感兴趣的数据是否被用于训练大语言模型（LLM），这一问题对于隐私审计、版权保护和基准污染检测至关重要。这些问题在预训练中使用受版权保护的书籍和科学论文，以及在微调中使用合成文本和医疗问答记录时尤为突出。在实践中，训练成员信号可能仅集中在长文档的少数段落中，因此在每个文档内聚合证据而不稀释这些局部信号至关重要。除了单个文档之外，审计目标数据集还需要基于文档级证据进行整体评估。在本文中，我们提出了一个统一的统计框架，用于文档级成员推断、目标数据集全局审计以及可疑子集识别。我们整合了互补的块级成员信号（摘要在此处被截断）

    arXiv:2611.26264v1 Announce Type: cross  Abstract: Training-data auditing asks whether data of interest were used to train a large language model (LLM), a question central to privacy auditing, copyright protection, and benchmark-contamination detection. These concerns are particularly relevant to the use of copyrighted books and scientific papers in pre-training, as well as synthetic text and medical question-answer records in fine-tuning. In practice, training membership signals may be concentrated in only a few passages of a long document, making it important to aggregate evidence within each document without diluting these localized signals. Beyond individual documents, auditing a target collection requires an overall assessment based on the document-level evidence. In this paper, we propose a unified statistical framework for document-level membership inference, target-collection global auditing, and suspicious-subset identification. We integrate complementary chunk-level membershi
    
[^47]: 持续强化学习中的稳定性差距

    The Stability Gap in Continual Reinforcement Learning

    [https://arxiv.org/abs/2611.26252](https://arxiv.org/abs/2611.26252)

    本文提出了一个在保持环境动态和奖励不变的前提下仅改变观测表示的简单持续强化学习基准，并首次揭示了其中的“稳定性差距”现象——即使智能体可以完全访问过去任务，新任务的引入仍会导致旧任务性能大幅下降后再部分或完全恢复。

    

    持续强化学习（CRL）旨在使智能体能够适应不断变化的任务，同时保留之前获得的能力。评估通常依赖于最终的聚合指标，而这些指标可能掩盖适应过程中的临时性能下降。本工作引入了一个简单的持续强化学习基准，该基准在保持环境动态和奖励不变的情况下改变观测表示，支持概念验证演示、低成本实验以及易于重现的健全性检查。利用这个基准，我们展示了持续强化学习中的稳定性差距：在引入新任务后，先前任务的性能出现大幅下降，随后部分或完全恢复。尽管智能体可以完全访问过去的任务，这种模式仍然会发生。

    arXiv:2611.26252v1 Announce Type: cross  Abstract: Continual reinforcement learning (CRL) aims to enable agents to adapt to changing tasks while preserving previously acquired capabilities. Evaluation often relies on final aggregate metrics that can hide temporary performance degradation during adaptation. This work introduces a simple CRL benchmark that varies observation representations while preserving environment dynamics and rewards, supporting proof-of-concept demonstrations, inexpensive experimentation, and easily reproducible sanity checks. Using this benchmark, we demonstrate the stability gap in CRL: a substantial drop in prior-task performance after the introduction of new tasks, followed by partial or complete recovery. This pattern occurs despite the agent having complete access to past tasks.
    
[^48]: TwinLab-Bench：在真实实验室的智能体孪生环境中评估AI科学家

    TwinLab-Bench: Evaluating AI Scientists in Agentic Twins of Real Labs

    [https://arxiv.org/abs/2611.26248](https://arxiv.org/abs/2611.26248)

    该论文提出TwinLab-Bench基准，利用来自真实物理科学、生命科学和工程实验室的52个任务，首次系统评估AI科学家自主设计实验的能力，其结果通过与真实实验室隐藏记录对比来评分。

    

    AI智能体日益被期望扮演AI科学家的角色，这需要三种开放性递增的能力：正确执行给定的计算工作流程、从给定的设计空间中选择下一个测量方案，以及自主设计实验。现有基准主要评估前两种能力，即通过终端或从预定义候选中进行选择的闭环方式来评估。而评估第三种能力需要可靠地执行智能体自主设计的实验，并使实验读数建立在真实实验室的基础之上。我们提出了TwinLab-Bench，该基准通过来自物理科学、生命科学和工程实验室的52个任务，评估AI科学家在自主设计实验方面的能力。该基准支持超过100种实验操作类型，智能体的报告通过与取自真实实验室记录的隐藏答案进行对比来评分。每个任务都有其专属的TwinLab智能体，该智能体充当实验环境……

    arXiv:2611.26248v1 Announce Type: new  Abstract: AI agents are increasingly expected to act as AI scientists, which requires three increasingly open abilities: carrying out a given computational workflow correctly, choosing the next measurements from a given design space, and designing their own experiments. Existing benchmarks mainly evaluate the first two abilities, in a terminal or in a closed loop that selects from predefined candidates. Evaluating the third ability requires reliably executing an experiment that an agent designs, with readings grounded in a real laboratory. We introduce TwinLab-Bench, which evaluates AI scientists on self-designed experiments in 52 tasks from laboratories in the physical sciences, the life sciences, and engineering. The benchmark supports more than 100 experimental action types, and the agent's report is scored by comparison with a hidden answer taken from real laboratory records. Each task has its own TwinLab agent, an agent that serves as the env
    
[^49]: 基于物理信息特征表示与预训练表格基础模型的泛化熔池几何预测

    Physics-Informed Feature Representations and Pretrained Tabular Foundation Models for Generalizable Melt Pool Geometry Prediction

    [https://arxiv.org/abs/2611.26235](https://arxiv.org/abs/2611.26235)

    该研究提出用归一化焓和佩克莱数两个物理信息无量纲特征替代原始工艺参数，并结合预训练表格基础模型进行基准测试，以提升LPBF熔池几何预测的跨材料泛化能力与数据效率。

    

    为先进制造开发可泛化、数据高效的机器学习（ML）模型仍然具有挑战性，原因在于实验数据集有限，且工艺-性能关系会随材料与运行条件而变化。在激光粉末床熔融（LPBF）中，熔池宽度、深度以及熔化模式（导热模式、过渡模式或锁孔模式）决定了工艺稳定性与缺陷形成。传统ML模型通常使用有量纲的工艺参数，这些参数只能隐式地表示激光-材料相互作用，可能限制跨材料迁移能力。本研究评估了两个物理信息驱动的无量纲特征——归一化焓与佩克莱数（Peclet number），它们将激光加工条件与材料热物理性能相结合——相比原始工艺参数能否改善熔池预测。我们系统性地对七个模型进行了基准测试：四种经典ML方法（高斯过程、支持向量机、随机森林和X……（原文摘要在此处截断）

    arXiv:2611.26235v1 Announce Type: new  Abstract: Developing generalizable, data-efficient machine learning (ML) models for advanced manufacturing remains challenging because experimental datasets are limited and process-property relationships vary across materials and operating conditions. In laser powder bed fusion (LPBF), melt pool width, depth, and melting mode (conduction, transition, or keyhole) govern process stability and defect formation. Conventional ML models typically use dimensional process parameters, which represent laser-material interactions only implicitly and may limit cross-material transfer. This study evaluates whether two physics-informed dimensionless features, normalized enthalpy and Peclet number, combining laser processing conditions with material thermophysical properties, improve melt pool prediction over raw process parameters. We systematically benchmark seven models: four classical ML methods (Gaussian process, support vector machine, random forest, and X
    
[^50]: 基于局部因果父节点发现的时间序列结构学习

    Time Series Structure Learning with Local Causal Parent Discovery

    [https://arxiv.org/abs/2611.26232](https://arxiv.org/abs/2611.26232)

    该论文提出ITPD——一种可靠且完备的基于约束的因果结构学习方法，通过局部到全局的因果父节点发现来学习具有因果非平稳性的多变量时间序列结构，并借助时间结构先验与图划分将高阶条件独立性检验替换为低阶检验，从而降低计算与样本复杂度。

    

    时间序列上的学习与推断会带来重大的统计挑战，包括观测值的自相关性、随时间变化的分布偏移以及维数灾难。这些挑战使因果发现算法难以投入实际应用，而这类算法在一般情况下本已存在高计算复杂度与高样本复杂度的问题。为缓解这些问题，本工作提出了迭代时间父节点发现（ITPD）：一种针对具有因果非平稳性的多变量时间序列的可靠且完备的基于约束的因果结构学习程序。ITPD 采用局部到全局的学习方法，通过依次发现每个变量的父节点集合来构建全局图。ITPD 利用时间结构先验、图划分和局部马尔可夫条件，将高阶条件独立性检验替换为低阶替代检验。在精确神谕（oracle）下，ITPD 执行了最少次数的……

    arXiv:2611.26232v1 Announce Type: new  Abstract: Learning and inference on time series can pose significant statistical challenges, including autocorrelated observations, distribution shift over time, and the curse of dimensionality. These challenges complicate the practical application of causal discovery algorithms, which already suffer from high computational and sample complexity in the general case. To mitigate these issues, this work introduces Iterative Temporal Parent Discovery (ITPD): a sound and complete constraint-based procedure for causal structure learning on multivariate time series with causal nonstationarity. ITPD takes a local-to-global learning approach that discovers the global graph by sequentially discovering the parent set for each variable. ITPD exploits temporal structural priors, graph partitioning, and the local Markov condition to swap higher-order conditional independence tests for lower-order alternatives. With an exact oracle, ITPD performed the least num
    
[^51]: 注意力头特化：样本复杂度中的相变

    Attention Head Specialization: Phase Transitions in Sample Complexity

    [https://arxiv.org/abs/2611.26228](https://arxiv.org/abs/2611.26228)

    该论文证明注意力头特化是样本复杂度中真正的相变，在可求解的多头注意力模型中表现为测试误差的非解析性，且随数据量增加可通过级联相变逐步恢复更弱的潜在特征，形成统计意义上的简单性偏置。

    

    注意力头在训练过程中会特化为不同的功能，往往随模型行为的急剧变化而分阶段涌现。虽然这一现象大多被作为训练时间的函数来研究，我们则探究注意力头特化如何依赖于可用训练数据的数量。在实证方面，我们发现BERT中特化于句法关系的注意力头会随数据集规模的增大而涌现。随后，我们在一个可求解的多头注意力高维模型中研究这一现象，并证明在该模型中，注意力头特化是样本复杂度意义上真正的相变，对应于测试误差及其他可观测量中的非解析性。根据数据结构和正则化的不同，特化可以跟随无信息量的头分化而发生，或通过一系列级联相变逐步恢复较弱的潜在特征，从而提供了一种统计形式的简单性偏置。

    arXiv:2611.26228v1 Announce Type: new  Abstract: Attention heads specialize into distinct functions during training, often emerging in stages together with sharp changes in model behavior. While this phenomenon has mostly been studied as a function of training time, we ask how head specialization depends on the amount of available training data. Empirically, we find that heads specialized to syntactic relations in BERT emerge as the dataset size increases. We then study this phenomenon in a solvable high-dimensional model of multi-head attention and establish that, in this model, head specialization is a genuine phase transition in sample complexity, corresponding to a non-analyticity in the test error and other observables. Depending on data structure and regularization, specialization can follow uninformative head differentiation or occur through a cascade of phase transitions that progressively recover weaker latent features, providing a statistical form of simplicity bias. Finally,
    
[^52]: 学习、冻结并跨架构迁移激活函数

    Learn, Freeze, and Transfer an Activation Function Across Architectures

    [https://arxiv.org/abs/2611.26201](https://arxiv.org/abs/2611.26201)

    该论文提出“学习-冻结-迁移”协议和五参数傅里叶级数激活函数 FAct，证明从源网络学得的激活函数可以被冻结并跨架构迁移到从零训练的网络中，成为权重迁移之外的一种互补性知识迁移方式。

    

    迁移学习通常通过复用源网络学到的权重来传递知识。我们研究学习到的激活函数是否也可以被迁移，并在从零开始训练的网络中保持有效。我们通过“学习-冻结-迁移”协议研究这种互补形式的迁移：激活函数与源网络在单次训练中联合优化，随后被冻结并复用于从零开始训练的目标网络。具有全局共享参数的逐点激活函数与层的维度无关，因此可以嵌入到从零开始训练的不同架构的网络中。我们用 FAct 实例化该协议，FAct 是一个由双谐波截断傅里叶级数表示的五参数激活函数，以 GELU 的傅里叶近似初始化，并在 ImageNet-1K 上的视觉 Transformer（ViT）中学习得到。当在相同架构的 ViT 中复用时……

    arXiv:2611.26201v1 Announce Type: new  Abstract: Transfer learning typically reuses knowledge acquired by a source network through its learned weights. We investigate whether a learned activation function can also be transferred and remain useful in networks trained from scratch. We study this complementary form of transfer through a learn-freeze-transfer protocol, in which the activation is optimized jointly with a source network in a single training run, and is then frozen and reused in target networks trained from scratch. A pointwise activation with globally shared parameters is independent of layer dimensions and can therefore be incorporated into networks of different architectures trained from scratch. We instantiate this protocol with FAct, a five-parameter activation represented by a two-harmonic truncated Fourier series, initialized from a Fourier approximation of GELU and learned in a Vision Transformer (ViT) on ImageNet-1K. When reused in a ViT of the same architecture trai
    
[^53]: 主干网络至关重要：在BDD100K上对驾驶场景语义分割的编码器-解码器网络进行基准测试

    Backbone Matters: Benchmarking Encoder-Decoder Networks for Semantic Segmentation of Driving Scenes on BDD100K

    [https://arxiv.org/abs/2611.26191](https://arxiv.org/abs/2611.26191)

    该论文在BDD100K数据集上对52种编码器-解码器组合进行了系统性基准测试，发现EfficientNet-B4主干网络搭配FPN解码器在驾驶场景语义分割中表现最佳（62.6% mIoU），并且主干网络的效果强烈依赖于所搭配的解码器。

    

    语义分割为自动驾驶车辆提供对其周围环境的密集像素级描述，是场景理解的核心组成部分。大多数分割网络由编码器（即主干网络）和解码器组成，然而主干网络的选择往往缺乏系统性的比较。本工作在BDD100K驾驶数据集上研究了这一选择问题。研究将四种广泛使用的解码器（U-Net、FPN、LinkNet和PSPNet）与来自六个家族（ResNet、VGG、DenseNet、Inception、MobileNet和EfficientNet）的13种主干网络进行组合，全部52个模型均在统一的训练协议下进行训练。表现最佳的模型——采用EfficientNet-B4主干网络的FPN解码器——达到了62.6%的平均交并比（mIoU）、89.8%的准确率和0.092的损失。在52个模型中，仅有两个模型（均为采用EfficientNet的FPN）超过了BDD100K基准报告的56.9% mIoU。主干网络的效果在很大程度上取决于解码器：相同的EfficientNet-B4在不同解码器下给出了42.6%的mIoU（摘要原文在此处似乎被截断）。

    arXiv:2611.26191v1 Announce Type: cross  Abstract: Semantic segmentation gives an autonomous vehicle a dense, pixel-level description of its surroundings and is a core part of scene understanding. Most segmentation networks consist of an encoder (the backbone) and a decoder, yet the backbone is often chosen without a systematic comparison. This work studies this choice on the BDD100K driving dataset. Four widely used decoders (U-Net, FPN, LinkNet and PSPNet) are combined with 13 backbones from six families (ResNet, VGG, DenseNet, Inception, MobileNet and EfficientNet), and all 52 models are trained with one common protocol. The best model, an FPN decoder with an EfficientNet-B4 backbone, reaches 62.6% mean intersection over union (mIoU), 89.8% accuracy and a loss of 0.092. Only two of the 52 models, both FPN with EfficientNet, exceed the 56.9% mIoU reported with the BDD100K benchmark. The effect of the backbone depends strongly on the decoder: the same EfficientNet-B4 gives 42.6% mIoU 
    
[^54]: 基于投影特征函数的仅样本分布间可逆传输

    Invertible Transport Between Sample-Only Distributions via Projected Characteristic Functions

    [https://arxiv.org/abs/2611.26187](https://arxiv.org/abs/2611.26187)

    该论文提出了一种免密度框架PCF（投影特征函数匹配），通过投影经验特征函数学习仅样本分布之间的可逆传输映射，无需密度估计、对抗训练或配对观测，并通过可选的传输正则化器选择低位移变换。

    

    许多有用的潜在分布可以进行采样，但没有易于处理的密度。这类分布可以由真实世界观测的经验集合表示，或者由科学模拟器和黑盒生成模型生成。这些仅样本先验无法直接用于基于似然的归一化流，因为归一化流需要评估先验密度。我们提出了投影特征函数匹配，这是一个免密度的框架，用于学习数据分布与仅通过样本指定的先验之间的可逆映射。我们的方法使用投影经验特征函数来对齐分布，从而消除了对密度估计、对抗训练或配对观测的需求。为了解决传输映射的非唯一性问题，我们引入了一个可选的传输正则化器，在保持端点对齐的同时选择位移较低（low-displacement）的变换。我们在c…（原文摘要在此处截断）

    arXiv:2611.26187v1 Announce Type: new  Abstract: Many useful latent distributions can be sampled but do not have a tractable density. Such distributions may be represented by empirical collections of real-world observations or generated by scientific simulators and black-box generative models. These sample-only priors cannot be used directly in likelihood-based normalizing flows, which require evaluating the prior density. We propose Projected Characteristic-Function matching (PCF), a density-free framework for learning an invertible map between a data distribution and a prior specified only through samples.   Our method aligns distributions using projected empirical characteristic functions, eliminating the need for density estimation, adversarial training, or paired observations. To address the non-uniqueness of the transport map, we introduce an optional transport regularizer that selects lower-displacement transformations while preserving endpoint alignment.   We benchmark PCF on c
    
[^55]: 奖励错位作为材料发现的机遇

    Reward misalignment as an opportunity for materials discovery

    [https://arxiv.org/abs/2611.26186](https://arxiv.org/abs/2611.26186)

    该研究提出，生成式AI材料发现中奖励函数与专家判断的错位并非失败，而是发现被奖励遗漏的新材料（如具有部分平坦能带的化合物）的机遇。

    

    奖励错位通常被视为人工智能的一种失败。然而，在利用生成式AI模型进行材料发现时，预定义奖励与专家判断之间的分歧可能揭示出奖励所忽略的材料。我们以电子平带为主要例子来研究这种可能性。专家们大多认同数值带宽评分，但他们之间的分歧是系统性的，且无法用估计约5%的错误标注率来解释。我们从成对比较中学习专家的偏好，并用其指导晶体扩散模型。学习到的偏好与数值评分在最平坦的能带上达成一致，但在其他情况下则出现分歧。例如，偏好倾向于KCo2As2和CoGa——这些已报道的具有部分平坦能带的化合物会被评分所淘汰。该框架还可以扩展到可合成性和声子性质。我们的研究结果表明，奖励错位可以揭示……

    arXiv:2611.26186v1 Announce Type: cross  Abstract: Reward misalignment is usually regarded as a failure of artificial intelligence. In materials discovery with generative AI models, however, disagreement between predefined rewards and expert judgment may reveal materials that the rewards overlook. We investigate this possibility using electronic flat bands as a prime example. Experts mostly agree with a numerical bandwidth score, but their disagreements are systematic and cannot be explained by an estimated 5% rate of mistaken labels. We learn the experts' preferences from pairwise comparisons and use them to guide a crystal diffusion model. The learned preferences and numerical score agree on the flattest bands but diverge elsewhere. For example, the preferences favor KCo2As2 and CoGa, reported compounds with partially flat bands that the score would discard. The framework also extends to synthesizability and phonon properties. Our findings suggest that reward misalignment can expose 
    
[^56]: 热力学一致的神经网络用于通用流体吉布斯自由能状态方程

    Thermodynamically Consistent Neural Networks for a Universal Fluid Gibbs Free Energy Equation of State

    [https://arxiv.org/abs/2611.26183](https://arxiv.org/abs/2611.26183)

    本研究以吉布斯自由能为出发点，提出基于热力学一致神经网络的新型流体状态方程，为流体状态提供统一的机器学习表示，其气液相密度预测精度优于传统体积平移立方型状态方程和现有机器学习模型。

    

    在过去三个世纪中，许多热力学研究者致力于开发既具有高精度又具有广泛适用性的流体状态方程，提出了各种理论框架和模型。在这些工作的基础上，本研究以吉布斯自由能为出发点，引入热力学一致的神经网络，构建了基于吉布斯自由能的流体状态方程（GEoS）。该方法建立了一种新颖的机器学习框架，为流体状态提供了统一的表示。GEoS在气相和液相密度测试集上表现良好，平均绝对相对偏差（AARD）分别为5.25%和3.96%。它优于传统的体积平移立方型状态方程和现有的机器学习模型，证明了其有效性。利用UMAP降维方法的进一步分析表明，测试集中的大多数分子……

    arXiv:2611.26183v1 Announce Type: new  Abstract: Over the past three centuries, many thermodynamics researchers have worked to develop fluid equations of state with both high accuracy and broad applicability, proposing various theoretical frameworks and models. Building on these efforts, this study uses Gibbs free energy as the starting point and introduces thermodynamically consistent neural networks to construct a fluid equation of state based on Gibbs free energy (GEoS). This approach establishes a novel machine learning framework that provides a unified representation of fluid states. GEoS performs well on the vapor- and liquid-phase density test sets, with average absolute relative deviations (AARDs) of 5.25% and 3.96%, respectively. It outperforms conventional volume-translated cubic equations of state and existing machine learning models, demonstrating its effectiveness. Further analysis using the UMAP dimensionality reduction method shows that most molecules in the test set bel
    
[^57]: 预算内化：固定预算RLVR以牺牲测试时扩展换取预算内准确率

    Budget Internalization: Fixed-Budget RLVR Trades Test-Time Scaling for In-Budget Accuracy

    [https://arxiv.org/abs/2611.26172](https://arxiv.org/abs/2611.26172)

    固定预算RLVR训练会将生成长度上限内化为模型的习得约束，用测试时扩展能力换取了预算内的准确率提升。

    

    推理模型在测试时能够生成更长的答案时往往会变得更准确，这种行为被称为测试时扩展。然而，基于可验证奖励的强化学习（RLVR）通常在固定的生成长度上限内训练模型。我们发现这一上限不仅影响训练成本，还会改变模型行为。训练后，模型在训练预算之外仅使用很少的额外token。它们在接近该预算时优于基础模型，但其准确率停止提升，并在更大预算下低于基础模型。因此，生成长度上限成为了一种习得的约束，限制了模型的测试时扩展能力。我们将这一效应称为“预算内化”。我们在Qwen3.5-4B、Gemma4-E2B和Nemotron-3-Nano-4B三个模型上，以及数学推理和多样化程序生成问题两类推理训练任务中均观察到了这一效应。该效应还会从数学训练迁移到科学和编程任务。在Qwen3.5-4B上进行的训练目标层面的消融实验表明……

    arXiv:2611.26172v1 Announce Type: cross  Abstract: Reasoning models often become more accurate when they can generate longer answers at test time. This behavior is known as test-time scaling. However, RL with verifiable rewards (RLVR) usually trains them with a fixed rollout cap. We find that this cap changes model behavior, not only training cost. After training, models use few additional tokens beyond the training budget. They outperform the base models near that budget, but their accuracy stops improving and falls below the base model at larger budgets. The rollout cap thus becomes a learned constraint, that limits model's test-time scaling. We call this effect budget internalization. We observe it across Qwen3.5-4B, Gemma4-E2B, and Nemotron-3-Nano-4B and two reasoning training tasks: mathematical reasoning and diverse, procedurally generated problems. This effect also transfers from math training to science and coding tasks. Training-objective level ablations on Qwen3.5-4B show tha
    
[^58]: NariQ：面向长程智能体任务的低于3比特量化方法

    NariQ: Sub-3-Bit Quantization for Long-Horizon Agentic Tasks

    [https://arxiv.org/abs/2611.26157](https://arxiv.org/abs/2611.26157)

    NariQ是一种格栅编码后训练量化方法，通过利用BF16模型在智能体会话中生成的token进行校准，首次实现了每权重2.5-3比特量化在长程智能体任务上与全精度模型持平的性能。

    

    长程智能体任务要求模型在10万token甚至更长的上下文中经过多步骤调用工具，然而许多低比特量化方法是在短网页文本上进行校准、并仅以困惑度作为评判标准的。这种错配代价高昂：现有的对Qwen3.8-27B在每权重3比特或以下的量化在AIME等较短任务上能与BF16持平，但根据已发表的结果，在长程智能体基准DeepSWE上它们最多只能解决BF16所解决任务的72%。我们提出NariQ，一种格栅编码的后训练量化方法，它使用BF16模型在智能体会话中生成的token进行校准。在Qwen3.8-27B上，NariQ在每权重3.0和2.5比特时在两个长程智能体基准DeepSWE和Terminal Bench以及AIME和RealWorldQA上均与BF16保持持平。我们每权重2.0比特的检查点则略逊于此性能。据我们所知，这些是首个被证明能与全精度版本相媲美的每权重3比特或以下的量化方法。

    arXiv:2611.26157v1 Announce Type: new  Abstract: Long-horizon agentic tasks require a model to call tools over many steps at contexts of 100K tokens or more, yet many low-bit quantization methods are calibrated on short web text and judged on perplexity. This mismatch is costly: existing quantizations of Qwen3.8-27B at or below 3 bits per weight match BF16 on shorter tasks like AIME, yet in published results solve at most 72% as many tasks as BF16 on DeepSWE, a long-horizon agentic benchmark. We propose NariQ, a trellis-coded post-training quantization method that calibrates on tokens generated by the BF16 model in agent sessions. On Qwen3.8-27B, NariQ at 3.0 and 2.5 bits per weight stays level with BF16 on two long-horizon agentic benchmarks, DeepSWE and Terminal Bench, as well as on AIME and RealWorldQA. Our checkpoint at 2.0 bits per weight trails this performance. To our knowledge, these are the first quantizations at or below 3 bits per weight shown to match their full-precision v
    
[^59]: 损失引导的Tsetlin机器：使学习目标显式化

    Loss-Guided Tsetlin Machines: Making the Learning Objective Explicit

    [https://arxiv.org/abs/2611.26114](https://arxiv.org/abs/2611.26114)

    本文提出Loss-Guided Tsetlin机器，用显式可替换的损失函数（CE、SCE、ASL）取代其固有的隐式间隔更新规则，在不改变Tsetlin自动机与子句逻辑的前提下，在六个分类数据集中的五个上匹配或提升了准确率与F1，尤其在不平衡多标签任务上将宏平均F1最多提升9个百分点。

    

    Tsetlin机器（TM）没有显式的损失函数概念，其学习信号被固定为一种隐式的、基于间隔的更新规则（MBUR），无法针对特定任务选择合适的目标函数。本文提出了损失引导的Tsetlin机器（Loss-Guided TM），用一个显式且可替换的损失函数取代MBUR。在所提出的方法中，子句反馈和权重更新由损失函数驱动，同时保持Tsetlin自动机和子句逻辑不变。该方法引入了交叉熵（CE）、对称交叉熵（SCE）和非对称损失（ASL）。我们在涵盖多分类、多标签和二分类任务的六个数据集上对Loss-Guided TM进行了评估。通过选择合适的损失函数，Loss-Guided TM在六个数据集中的五个上达到或超越了MBUR的准确率和F1分数。收益在不平衡的多标签数据集上最为显著，其中ASL将宏平均F1提升了高达9个百分点。使用CE时，Loss-Guided TM还……（摘要在此处截断）

    arXiv:2611.26114v1 Announce Type: new  Abstract: The Tsetlin Machine (TM) has no explicit notion of a loss function. Its learning signal is fixed to an implicit, margin-based update rule (MBUR), with no way to choose an objective function suited to a specific task. This paper introduces Loss-Guided TM, which replaces MBUR with an explicit and swappable loss function. In our proposed method, clause feedback and weight updates are driven by the loss, while keeping the Tsetlin Automaton and clause logic unchanged. The approach incorporates cross entropy (CE), symmetric cross entropy (SCE), and asymmetric loss (ASL). We evaluate Loss-Guided TM across six datasets that span multiclass, multilabel, and binary classification tasks. With a suitable loss, Loss-Guided TM matches or improves accuracy and F1 over MBUR on five of the six datasets. The gains are greatest for the imbalanced multilabel datasets, where ASL improves macro F1 by up to 9 percentage points. With CE, Loss-Guided TM is also 
    
[^60]: 物理信息深度学习用于磁共振弹性成像中的剪切波速度估计

    Physics-Informed Deep Learning for Shear Wave Speed Estimation in MR Elastography

    [https://arxiv.org/abs/2611.26111](https://arxiv.org/abs/2611.26111)

    提出了一种端到端可训练的物理信息深度学习框架，将欠采样k空间数据的图像重建与MRE相位梯度反演相结合，可直接从测量数据稳健地估计脑组织剪切波速度。

    

    目的：磁共振弹性成像是一种无创成像技术，用于绘制活体组织的生物力学特性，包括剪切波速度（SWS），但其数据采集本质上较为缓慢，且波场反演是一个病态问题。我们不依赖手工设计的图像先验，而是提出一种数据驱动的方法，将图像重建与MRE反演相结合，从欠采样的k空间数据中稳健地估计SWS。方法：我们的物理信息重建框架包含两个模块：基于模型的神经网络（NN）正则化重建模块，以及从重建图像计算SWS的相位梯度反演。将两个模块级联后，即可得到一种端到端可训练的方法，直接从测量的k空间数据估计SWS。我们在回顾性高度欠采样的脑部MRE数据上对该方法进行了评估，并将其与基于全变分（TV）最小化的方法进行了比较。

    arXiv:2611.26111v1 Announce Type: cross  Abstract: Objective: Magnetic Resonance Elastography (MRE) is a non-invasive imaging technique for mapping biomechanical properties of in vivo tissue, including shear wave speed (SWS), but involves intrinsically slow data acquisition and an ill-posed wave inversion. Instead of relying on handcrafted image priors, we propose a data-driven approach jointly combining image reconstruction and MRE inversion for robust SWS estimation from undersampled k-space data. Methods: Our physics-informed reconstruction framework comprises two blocks: a model-based neural network (NN)-regularized reconstruction module and a phase-gradient inversion (k-MDEV) calculating SWS from the reconstructed images. Concatenating both blocks yields an end-to-end trainable method to estimate SWS directly from measured k-space data. We evaluated the method on retrospectively highly undersampled brain MRE data and compared it to a total variation (TV) minimization-based approac
    
[^61]: 网络拓扑与数据异质性如何塑造去中心化学习：一种谱分析

    How Network Topology and Data Heterogeneity Shape Decentralized Learning: A Spectral Analysis

    [https://arxiv.org/abs/2611.26104](https://arxiv.org/abs/2611.26104)

    该论文提出了一个谱分析框架，揭示去中心化学习中的两个互补机制——主导特征模态决定共识轨迹与节点影响力（平稳影响力正比于节点度加一），非主导特征模态控制分歧的传播与衰减——并推导出模型分歧与谱混合及梯度异质性之间关系的界。

    

    网络拓扑通过信息混合和各节点的相对贡献影响去中心化联邦学习。现有分析主要关注谱混合特性，通常假设双随机聚合，忽视了在不规则通信网络下依赖于拓扑的节点影响力。我们开发了一个谱分析框架，将两种互补机制分离开来：主导特征模态决定共识轨迹和节点影响力，而非主导特征模态控制分歧的传播与衰减。在均匀邻域平均的条件下，我们证明平稳影响力与“节点度加一”成正比，并推导出一个将模型分歧与谱混合特性及梯度异质性相关联的界。随后，我们研究了这些机制如何与节点间的数据分布相互作用。在规则与异构网络拓扑上的实验揭示了……

    arXiv:2611.26104v1 Announce Type: new  Abstract: Network topology influences decentralized federated learning through information mixing and the relative contribution of individual nodes. While existing analyses primarily focus on spectral mixing properties, they often assume doubly stochastic aggregation, overlooking topology-dependent influence under irregular communication networks. We develop a spectral framework that separates two complementary mechanisms: the dominant eigenmode determines the consensus trajectory and node influence, while non-dominant eigenmodes govern the propagation and attenuation of disagreement. Under uniform neighborhood averaging, we show that the stationary influence is proportional to node degree plus one and derive a bound relating model disagreement to spectral mixing and gradient heterogeneity. We then investigate how these mechanisms interact with the data distribution across nodes. Experiments across regular and heterogeneous network topologies reve
    
[^62]: 利用完全路径贡献揭示网络计算中的通路相互作用

    Revealing Pathway Interactions in Network Computations with Complete Path Contribution

    [https://arxiv.org/abs/2611.26101](https://arxiv.org/abs/2611.26101)

    该论文提出“完全路径贡献”方法，将网络输出分解为每条路径在数学上完整的贡献，从而能够直接分析从输入到输出的完整因果路径之间的高阶相互作用。

    

    网络通过多层互联单元传递信号，因此解释网络计算意味着追踪信息如何从输入流向输出，以及各条通路之间如何相互作用。大多数现有方法仅分析单元、层或注意力头等组件之间的成对相互作用。即使将这些度量事后组合成电路，它们也无法捕捉从输入到输出的完整因果链（路径）上的高阶相互作用，因此无法解答指数级数量的路径如何相互作用以构建网络计算。这使得大量机制性问题无法回答，例如抑制性通路是否会否决兴奋性通路，以及两者如何结合以产生网络决策。为了填补这一空白，我们提出了完全路径贡献，这是一种将网络输出分解为各条路径的数学上完整的贡献的方法，从而可以直接研究路径层面的相互作用。

    arXiv:2611.26101v1 Announce Type: new  Abstract: Networks pass signals through many layers of interconnected units, so explaining what they compute means tracing how information travels from input to output and how pathways interact. Most existing approaches analyze only pairwise interactions between components such as units, layers or attention heads. Even when these measurements are combined post hoc into a circuit, they do not capture higher-order interactions along full causal chains (paths) from input to output, and so cannot address how exponentially many paths interact to construct network computation. This leaves a large set of mechanistic questions unanswerable, for example whether inhibitory pathways veto excitatory ones, and how the two combine to yield network decisions. To address this gap, we introduce Complete Path Contribution, a method to decompose the network output into mathematically complete contributions of individual paths to study path-level interactions directl
    
[^63]: 从会话到服务：将基于大语言模型的嵌入集成到生产级内容推荐的检索阶段

    From Sessions to Serving: Integrating LLM-Based Embeddings into Retrieval for Production Content Recommendation

    [https://arxiv.org/abs/2611.26096](https://arxiv.org/abs/2611.26096)

    本文提出LSER生产级内容推荐系统，将预训练大语言模型生成的会话级和历史级用户嵌入作为辅助特征集成到检索阶段用户塔中，并通过近线异步服务路径保持嵌入实时更新。

    

    大型市场网站上的个性化内容推荐必须将异构的候选内容（例如客户故事、产品页面、行业页面和学习内容）与稀疏且快速变化的用户行为相协调，这使得检索阶段不仅要服务于下游排序，还需更好地表示用户兴趣，其要求日益提高。我们提出了LSER（LLM Session Embedding Retrieval，基于大语言模型的会话嵌入检索），这是一个面向autodesk.com的生产级内容推荐系统，它从预训练大语言模型（LLM）中提取会话级和历史级嵌入，并将其作为辅助用户侧特征添加到检索阶段的用户塔中，与现有的已学习序列编码器并存。为了在实时行为下保持用户侧嵌入的新鲜度，我们构建了一条近线服务路径，异步刷新主内容推荐服务器的序列向量以及LLM导出的用户向量。我们验证了LLM特征的增量价值（摘要在此处截断）

    arXiv:2611.26096v1 Announce Type: cross  Abstract: Personalized content recommendation on large marketplace websites must reconcile heterogeneous candidate content (e.g., customer stories, product pages, industry pages, and learning content) with sparse, fast-changing user behavior, placing increasing demands on the retrieval stage to represent user interest well, not just the downstream ranking stage. We present LSER (LLM Session Embedding Retrieval), a production content recommendation system for autodesk.com that derives session- and history-level embeddings from a pretrained large language model (LLM) and adds them as auxiliary user-side features in the retrieval-stage user tower alongside existing learned sequence encoders. To keep user-side embeddings fresh under real-time behavior, we build a nearline serving path that asynchronously refreshes sequential and LLM-derived user vectors for our main content recommendation server. We validate the incremental value of the LLM features
    
[^64]: 面向混合模型内存与计算高效在线学习的近似伴随模型

    Approximate Adjoint Models for Memory and Compute-Efficient Online Learning of Hybrid Models

    [https://arxiv.org/abs/2611.26095](https://arxiv.org/abs/2611.26095)

    该论文提出一类近似伴随模型方法，在整个轨迹上保留完整的伴随递推关系而仅近似其中的回拉算子，从而实现混合模型的内存与计算高效在线学习，避免了截断BPTT带来的梯度偏差和对长程依赖的忽略。

    

    可微分编程通过最小化沿模拟轨迹累积的误差，实现数值模型的在线校准，包括估计物理参数以及在混合配置中训练神经子模型。梯度通常通过伴随方法或时间反向传播（BPTT）计算，其计算与内存开销随训练窗口长度增长，这限制了混合模型训练的时长和分辨率，成为现实地球物理应用的主要瓶颈。诸如截断BPTT之类的方法通过丢弃跨越截断边界的敏感性来减少内存占用，但会引入梯度偏差并忽略长程依赖关系。我们提出了一类梯度近似方法，在整个轨迹上保持伴随递推关系，同时仅对其中的回拉算子进行近似。精确的回拉算子通过自动微分计算（摘要在此处截断）。

    arXiv:2611.26095v1 Announce Type: new  Abstract: Differentiable programming enables online calibration of numerical models by minimizing errors accumulated along simulated trajectories. This includes estimating physical parameters and training neural sub-models in hybrid configurations. Gradients are typically computed using adjoint methods, or backpropagation through time (BPTT), whose computational and memory costs grow with the length of the training window. This limits the duration and resolution of hybrid-model training, a major bottleneck for realistic geophysical applications. Approaches like truncated BPTT reduces memory use by discarding sensitivities across truncation boundaries, introducing gradient bias and neglecting long-range dependencies. We propose a class of gradient approximation methods that preserve the adjoint recurrence over the full trajectory while approximating only the pullback operators within it. Exact pullbacks are computed by automatic differentiation at 
    
[^65]: 假面舞会上的幽灵：基于图神经网络的控制器局域网伪装攻击检测

    Phantom of the Masquerade: Graph Neural Network-Based Detection of Masquerade Attacks in Controller Area Networks

    [https://arxiv.org/abs/2611.26094](https://arxiv.org/abs/2611.26094)

    本文提出PhantoMasq框架，利用图神经网络将CAN流量建模为融合信号统计特征的消息序列图，联合学习结构依赖与时间特性，从而有效检测控制器局域网中难以识别的伪装攻击。

    

    控制器局域网（CAN）缺乏内置的安全机制，使得现代车辆和工业系统容易受到伪装攻击（masquerade attacks），这类攻击可能危及安全性与可靠性。这些攻击会注入模仿合法通信模式的恶意消息，导致现有入侵检测系统（IDS）难以检测。以往基于图的IDS方法通常依赖浅层嵌入，将表示学习与分类过程解耦，这可能限制了它们充分利用结构和信号级信息的能力。在本文中，我们提出了PhantoMasq，一个基于图神经网络（GNN）的CAN系统伪装攻击检测框架。PhantoMasq将CAN流量建模为消息序列图，并结合从信号行为中提取的轻量级统计特征，实现对结构依赖关系与时间特性的联合学习。检测任务被表述为图

    arXiv:2611.26094v1 Announce Type: cross  Abstract: The lack of built-in security in controller area networks (CANs) exposes modern vehicles and industrial systems to masquerade attacks that can compromise safety and reliability. These attacks inject malicious messages that mimic legitimate communication patterns, making them difficult to detect using existing intrusion detection systems (IDS). Prior graph-based IDS approaches often rely on shallow embeddings that decouple representation learning from classification, which may limit their ability to fully exploit structural and signal-level information. In this paper, we propose PhantoMasq, a graph neural network (GNN)-based framework for detecting masquerade attacks in CAN systems. PhantoMasq models CAN traffic as message sequence graphs enriched with lightweight statistical features derived from signal behavior, enabling joint learning of structural dependencies and temporal characteristics. The detection task is formulated as a graph
    
[^66]: 强化学习只能强化它所见到的成功

    RL Can Only Reinforce What It Sees Succeed

    [https://arxiv.org/abs/2611.26084](https://arxiv.org/abs/2611.26084)

    在先验与正确答案均已精确已知的谜题环境中，该研究发现结果奖励强化学习只能强化训练中实际观察到成功的动作——当正确动作罕见时，训练能否“起飞”可由训练前即可计算的一个统计量（动作获胜的尝试组期望数量，约35组对应五成概率）准确预测，且该规律跨多种实验设置普遍成立。

    

    当一个熟悉的规则失效时，智能体必须权衡预训练教给它的内容与它当前观察到的内容。我们研究结果奖励强化学习（RL）如何改变这种平衡，并在一个两侧信息均已确定的谜题世界中进行验证：小语言模型在由贝叶斯玩家玩出的游戏数据上从头训练，因此先验和正确答案都是精确已知的。基础模型像其语料库那样进行对冲，而RL会将这种对冲转化为决策，方向取决于奖励偏向哪一方。但RL只能强化它所见到的成功。当正确动作很罕见时，每次训练要么在所有地方都学会它，要么完全学不会；而哪些训练能够成功，可以在训练开始前用一个计数来预测：该动作获胜的尝试组的期望数量。大约35个这样的组对应五成胜率。该计数对新问题和重复问题均成立，对4到256次尝试的组均成立，并且在不同学习率、优化器、GRPO变体和模型规模下均保持成立。

    arXiv:2611.26084v1 Announce Type: cross  Abstract: When a familiar rule stops working, an agent has to weigh what pre-training taught it against what it now observes. We ask how outcome-reward reinforcement learning (RL) changes this balance, in a puzzle world where both sides are known: small language models trained from scratch on games played by a Bayesian player, so that the prior and the right answer are exact. Base models hedge as their corpus did, and RL turns that hedging into decisions, in whichever direction the reward favours. But RL can only reinforce what it sees succeed. When the right action is rare, each run either learns it everywhere or not at all, and which runs take off is predicted before training by one count: the expected number of groups of attempts in which the action wins. About 35 such groups give even odds. The count holds for new and repeated problems and for groups of 4 to 256 attempts, and across learning rates, optimizers, GRPO variants and model sizes f
    
[^67]: 从参数化计算图中学习比较神经表示

    Learning to Compare Neural Representations from Parameterized Computation Graphs

    [https://arxiv.org/abs/2611.26081](https://arxiv.org/abs/2611.26081)

    提出了一种基于图元网络的框架，可直接从模型的参数化计算图中学习预测内部表示相似性，无需实际运行模型即可实现层检索与模型拼接组件选择。

    

    比较已训练模型的内部表示对于识别对应层以及为模型重用选择组件非常重要。然而，在推理时进行直接比较需要在已知参考输入上评估候选模型并收集其内部激活。本文研究了能否通过学习，从模型的参数化计算中直接预测内部表示的相似性，从而在未见过的模型上进行推断，而无需在独立训练的模型之间实际测量相似性。我们提出了一个应用图元网络的框架，将每个内部位置映射到由其上游计算导出的嵌入，同时考虑隐藏坐标排列和内部注意力基变换。我们分析了所提估计器的结构特性，并推导了层检索与拼接候选选择的条件性保证。实验表明，学习到的预测器能够支持表示关系的估计。

    arXiv:2611.26081v1 Announce Type: new  Abstract: Comparing internal representations across trained models is important for identifying corresponding layers and selecting components for model reuse. Yet, direct comparisons at inference require evaluating candidate models on known reference inputs and collecting their internal activations. In this paper, we study whether internal representation similarity measured across independently trained models can instead be learned to predict in unseen models from their parameterized computations. We introduce a framework that applies a graph metanetwork for mapping each internal site to an embedding derived from its upstream computations while accounting for hidden-coordinate permutation and internal attention-basis transformations. We analyze the structural properties of our estimators and derive conditional guarantees for layer retrieval and stitching candidate selection. Empirically, we find that the learned predictors support relation estimat
    
[^68]: 超越梯度：理解MoE决策机制

    Understanding MoE Decisions Beyond Gradients

    [https://arxiv.org/abs/2611.26075](https://arxiv.org/abs/2611.26075)

    该论文提出一种利用MoE模型自身的专家输出和选择规则来估计专家替换引发的级联切换效应的预测器，相比传统梯度方法将答案得分变化的预测误差在Qwen3.6上降低60%、在Gemma4上降低47%。

    

    理解语言模型的决策需要了解其内部计算如何影响模型在不同答案之间的偏好。基于梯度的方法通过预测改变这些内部计算所带来的效果来估计这种影响。在混合专家模型中，替换某一个专家可能会改变后续各层所选择的专家。这种专家切换会以导数无法捕捉的方式改变答案得分。我们提出了一种预测器，它利用模型自身的专家输出和选择规则来估计这些额外的影响，并将其加入到二阶预测之中。在来自三个基准测试的问题上，相较于使用梯度和二阶导数的多步基线方法，我们的方法在Qwen3.6上将答案得分变化的预测误差降低了60%，在Gemma4上降低了47%。

    arXiv:2611.26075v1 Announce Type: new  Abstract: Understanding a language model's decision requires knowing how its internal computations affect its preference between answers. Gradient-based methods estimate this influence by predicting the effect of changing those computations. In mixture-of-experts (MoE) models, replacing one expert can change which experts later layers select. Those switches can alter the answer scores in ways that derivatives do not capture. We introduce a predictor that uses the model's own expert outputs and selection rules to estimate these extra effects, then adds them to a second-order prediction. Across questions from three benchmarks, our method reduces the error in predicting answer-score changes by 60\% on Qwen3.6 and 47\% on Gemma4, relative to a multistep baseline using gradients and second derivatives.
    
[^69]: 迭代稀缺性引导探索：从狭窄数据支撑中自举生成式自动竞价

    Iterative Scarcity-Guided Exploration: Bootstrapping Generative Auto-bidding from Narrow Support

    [https://arxiv.org/abs/2611.26060](https://arxiv.org/abs/2611.26060)

    该论文从理论上揭示了生成式自动竞价在狭窄数据支撑下停滞的根因是信噪比崩溃，并提出迭代稀缺性引导探索（ISGE）方法，通过在回报信号崩溃时让稀缺性接管作为探索信号以提升信噪比，从而突破次优停滞实现自举式改进。

    

    自动竞价是在线广告拍卖中的核心算法组件。在实际的冷启动场景中，迭代训练通常被用来补偿低质量、支撑范围狭窄的历史数据。遗憾的是，在迭代训练过程中，代表性的基于扩散模型的AI生成竞价（AIGB）方法无法在狭窄的数据支撑之外持续进行外推探索，从而停滞在次优水平。本文从理论上将这种停滞归因于信噪比（SNR）崩溃：微弱的径向回报信号被曲率引起的惩罚所淹没。为了打破这种停滞，我们提出了迭代稀缺性引导探索（ISGE），它引入了一种引导交接机制：当回报信号崩溃时，稀缺性随之接管作为探索性信号，将信噪比提升至临界阈值以上。具体而言，ISGE在一个为数据分配稀缺性得分的评判器之间迭代交替……

    arXiv:2611.26060v1 Announce Type: new  Abstract: Auto-bidding is a core algorithmic component in online advertising auctions. In practical cold-start settings, iterative training is commonly used to compensate for low-quality, narrowly supported historical data. Unfortunately, in iterative training, representative diffusion-based AI-Generated Bidding (AIGB) methods fail to sustain extrapolation beyond the narrow data support and stagnate at a suboptimal level. In this paper, we theoretically attribute this stagnation to Signal-to-Noise Ratio (SNR) collapse: the weak radial return signal is overwhelmed by a curvature-induced penalty. To break this stagnation, we propose \textbf{Iterative Scarcity-Guided Exploration (ISGE)}, which introduces a guidance handoff: as the return signal collapses, scarcity subsequently takes over as an exploratory signal to elevate the SNR above a critical threshold. Specifically, ISGE iteratively alternates between a \emph{Judger} that assigns scarcity score
    
[^70]: Complete-SVD：一种面向大语言模型低秩压缩的泛函数范数方法

    Complete-SVD: A Functional Norm Approach to Low-Rank LLM Compression

    [https://arxiv.org/abs/2611.26059](https://arxiv.org/abs/2611.26059)

    提出Complete-SVD方法，通过最小化原始与压缩模型logit之间的等权L2距离，将大语言模型低秩压缩目标化简为具有闭式解的加权SVD，并利用随机探测以与词表大小和序列长度无关的计算成本无偏估计难以求解的输出侧因子。

    

    神经网络的训练后低秩压缩不仅需要近似原始权重，还必须保持模型的输入输出行为。近期的输出感知方法在压缩层之外、即模型输出处度量压缩误差，通过观察到的下一个token或模型预测的概率对logit变化进行加权。我们提出Complete-SVD，一种面向大语言模型的方法，它最小化原始模型与压缩模型logit之间的L2距离，并对所有logit变化进行等权处理。通过在原始权重附近对网络进行线性化，并结合Kronecker因子分解近似，该目标函数被化简为具有闭式解的加权SVD。其输出侧因子难以直接计算；我们通过随机探测对其进行无偏估计，每个序列的代价仅为K次反向传播，且该成本与词表大小和序列长度无关。在相同设置与均匀……（摘要在此处截断）

    arXiv:2611.26059v1 Announce Type: cross  Abstract: Post-training low-rank compression of neural networks must not only approximate the original weights but also preserve the model's input-output behavior. Recent output-aware methods measure the compression error beyond the compressed layer, at the model output, weighting logit changes by the observed next token or by the model's predicted probabilities. We propose Complete-SVD, a method for large language models that minimizes the $L_2$ distance between the logits of the original and compressed models, weighting all logit changes equally. Linearizing the network around the original weights, together with a Kronecker-factored approximation, reduces this objective to a weighted SVD with a closed-form solution. Its output-side factor is intractable to compute; we estimate it without bias from random probes, at a cost of $K$ backward passes per sequence, independent of vocabulary size and sequence length. At matched settings with uniform r
    
[^71]: SlotFlow-GB：用于重叠银河系双星跨维度推断的神经框架

    SlotFlow-GB: A Neural Framework for Trans-Dimensional Inference of Overlapping Galactic Binaries

    [https://arxiv.org/abs/2611.26057](https://arxiv.org/abs/2611.26057)

    SlotFlow-GB是一种神经框架，通过学习跨维度后验的近似，可在单次评估中同时推断强重叠银河系双星的数量及其物理参数，无需预先指定源的数目。

    

    激光干涉空间天线（LISA）将观测毫赫兹频段的引力波天空，其中包括数百万个长寿命的银河系双星（GBs），预计其中数万个在整个任务运行期间可被单独分辨。提取这些信号源需要从强重叠信号中联合确定它们的数量和物理参数，这是一个计算需求极高的跨维度推断问题。我们提出了SlotFlow-GB，一个神经框架，它构建了对跨维度后验的学习近似，能够在单次评估中产生关于源数量的概率分布以及各个双星的参数密度分布，而无需预先指定源的数量。我们使用包含2 μHz频率区间内多达十个双星、为期一年的LISA模拟演示了该框架，并重点关注源……（摘要在此处被截断）的高信噪比情形。

    arXiv:2611.26057v1 Announce Type: cross  Abstract: The Laser Interferometer Space Antenna (LISA) will observe the millihertz gravitational-wave sky, including millions of long-lived Galactic binaries (GBs), tens of thousands of which are expected to be individually resolvable throughout the mission's lifetime. Extracting these sources requires jointly determining their number and physical parameters from strongly overlapping signals, posing a computationally demanding trans-dimensional inference problem. We introduce SlotFlow-GB, a neural framework that constructs a learned approximation to the trans-dimensional posterior, producing a probability distribution over source multiplicity and parameter densities for individual binaries in a single evaluation, without requiring their number to be specified in advance. We demonstrate this framework using year-long LISA simulations containing up to ten binaries within a 2 $\mu$Hz frequency interval, focusing on a high-SNR regime in which sourc
    
[^72]: 什么时候排序优于符号？偶函数干扰下的梯度方向恢复

    When Do Rankings Beat Signs? Gradient Direction Recovery under Even Nuisance

    [https://arxiv.org/abs/2611.26055](https://arxiv.org/abs/2611.26055)

    该论文证明在偶函数干扰下，跨对排序策略Reverse仅需Õ(d+√(d/ε))次评估、单轮反馈即可将梯度方向恢复至角精度ε，达到理论下界且优于仅使用对内符号的方法。

    

    能否通过对一批对径查询对进行排序，比仅使用对内符号以更少的评估次数恢复梯度方向？我们研究了d维空间中线性信号叠加未知可微偶函数干扰下的固定中心恢复问题，并将函数评估次数与反馈轮数分开计数。共享的对偏移形成了一个信息层级：Mirror保留对内符号，Reverse将反转的跨对排序耦合以抵消偏移，Full则强制执行所有排序不等式。相对壳变化定义为查询球面上的干扰范围与线性信号尺度的归一化比值。当该值至多为cε/√d时，Reverse在固定置信度下仅用一轮反馈、Õ(d+√(d/ε))次评估，即可将每个一致方向的角精度定位到ε，且无需向解码器提供变化界。这与下界在对数因子内相匹配，而仅使用符号则需Θ(d

    arXiv:2611.26055v1 Announce Type: new  Abstract: Can ranking a batch of antipodal query pairs recover a gradient direction with fewer evaluations than using within-pair signs alone? We study fixed-center recovery in $d$ dimensions for a linear signal plus an unknown differentiable even nuisance, counting function evaluations and feedback rounds separately. Shared pair offsets create an information hierarchy: Mirror retains within-pair signs, Reverse couples reversed cross-pair orders to cancel offsets, and Full enforces all ranking inequalities. Relative shell variation is the nuisance range over the query sphere normalized by the linear-signal scale. When it is at most $c\varepsilon/\sqrt{d}$, Reverse localizes every consistent direction to angular accuracy $\varepsilon$ using $\widetilde{O}(d+\sqrt{d/\varepsilon})$ evaluations in one round at fixed confidence, without a variation bound supplied to the decoder. This matches a lower bound up to logarithmic factors, versus the $\Theta(d
    
[^73]: 基于终止塑形强化学习的无参考人形机器人滑板运动

    Reference-Free Humanoid Skateboarding via Termination-Shaped Reinforcement Learning

    [https://arxiv.org/abs/2611.26054](https://arxiv.org/abs/2611.26054)

    提出了一种无需人类演示或解析模型的深度强化学习框架，通过任务奖励与终止塑形相结合，使人形机器人（如K1）以无模型方式学会滑板这一物体介导的复杂运动任务。

    

    人形机器人运动学研究在行走和跑步方面进展迅速，但“物体介导”的运动——即机器人必须与被动外部物体协调配合的运动——仍然十分困难。滑板就是一项具有挑战性的任务，它要求机器人同时与地面和一个被动但可移动的平台交互以实现前进运动。目前人形机器人滑板的方法要么需要人类演示，要么需要使用依赖系统辨识的解析转向模型。本文提出了一种不依赖人类演示等先验数据的深度强化学习框架，使K1等人形机器人能够通过无模型学习学会滑板。该方法将任务奖励与阻断漏洞利用的终止条件相结合，并引入了一系列约束来限制诸如踏板脚脱离滑板、双脚同时站上滑板等不良行为。

    arXiv:2611.26054v1 Announce Type: cross  Abstract: Humanoid locomotion research has advanced rapidly on walking and running, but \emph {object-mediated} locomotion, in which a robot must coordinate its body with a passive external object, remains difficult. One challenging task is skateboarding, which requires simultaneous interaction with the ground and a passive but movable platform to achieve forward motion. Current methods for tackling skateboarding with humanoid robots either require human demonstrations or utilise analytic steering models that require system identification. In this paper, we introduce a deep reinforcement learning framework that does not rely on prior data such as demonstrations and enables robots such as the K1 humanoid to learn to skateboard using model-free learning. The method combines task rewards with exploit-blocking terminations. We introduce a number of constraints restricting behaviours such as losing the planted board foot, standing with both feet on t
    
[^74]: 修正孪生数据而非检测器：基于数字孪生数据的路侧激光雷达从仿真到现实的检测

    Fixing the Twin, Not the Detector: Sim-to-Real Roadside LiDAR Detection from Digital-Twin Data

    [https://arxiv.org/abs/2611.26035](https://arxiv.org/abs/2611.26035)

    该论文的核心创新在于修正数字孪生数据与真实场景之间的差异（对齐坐标系、点密度和标注约定，补全缺失的行人与骑行者并以真实尺寸和点密度重建目标），而非改进检测器本身，从而使仅用合成数据在单块8GB笔记本GPU上训练的CenterPoint模型在UrbanTwin Sim2Real激光雷达挑战赛LUMPI赛道夺得第一名。

    

    路侧激光雷达检测器使用真实标注进行训练成本高昂，数字孪生有望替代这些标注，但在孪生数据上训练的检测器在真实场地会面对不同的传感器、不同的目标以及不同的标注约定。我们描述了在UrbanTwin Sim2Real激光雷达挑战赛（IEEE BigData 2026 Cup）LUMPI赛道获得第一名（综合得分0.5541）、V2X-Real赛道获得第五名（0.8102）的完整技术流程。所有检测器均为CenterPoint模型，仅使用合成数据训练，运行在单块8 GB笔记本GPU上。大部分性能提升来自修正孪生数据与真实场地之间的差异，而非来自检测器本身。我们将孪生数据的坐标系、点密度和边界框约定与测试传感器对齐，生成孪生数据中缺失的行人和骑行者，并将街道杂波作为无标注负样本，同时以真实尺寸和真实点密度重建孪生目标。一种非学习的重采样方法……

    arXiv:2611.26035v1 Announce Type: cross  Abstract: Roadside LiDAR detectors are expensive to train on real labels, and digital twins promise to replace these labels, but a detector trained on a twin meets a different sensor, different objects and different annotation conventions on the real site. We describe the pipeline that placed first in the LUMPI track of the UrbanTwin Sim2Real LiDAR Challenge (IEEE BigData 2026 Cup, 0.5541 combined score) and fifth on its V2X-Real track (0.8102). Every detector is a CenterPoint model trained only on synthetic data, on a single 8 GB laptop GPU. Most of the gain came from correcting differences between the twin and the real site rather than from the detector. We align the twin's coordinates, point density and box convention with the test sensor, generate the pedestrians and riders that the twins lack, plus street clutter as unlabeled negatives, and rebuild twin objects at real size and real point density. A non-learned resampling to the per-cell po
    
[^75]: 不公平的延迟社会代价

    The Delayed Societal Cost of Unfairness

    [https://arxiv.org/abs/2611.26030](https://arxiv.org/abs/2611.26030)

    该论文揭示了不公平性的延迟社会代价：在部署前数据上看似帕累托最优的不公平决策政策，在其随时间诱导出的人群上进行评估时，可能被更公平的政策超越，从而在不损害任何一方的情况下同时降低社会成本。

    

    公平机器学习通常在静态数据上评估决策政策的社会成本——即决策者（DM）所承担的损失，以及在决策对象（DS）不同人口群体之间成本的不公平分布——在这种静态评估框架下，减少一方的成本往往以牺牲另一方为代价。然而，决策会改变其所作用的人群，已有研究表明，基于静态数据设计的公平性干预措施可能无法惠及弱势群体，甚至随时间推移对其造成伤害。我们研究了一个互补的问题：在什么情况下，一个在部署前数据上帕累托最优的政策，一旦在它所诱导出的人群上进行评估，反而会被一个更公平的政策超越——即在不增加另一方成本的情况下降低一方的成本？在资格分数随二元决策及其结果而动态演化的设定中，我们证明，在罗尔斯式公平（即从不降低任何群体的接受率）下……

    arXiv:2611.26030v1 Announce Type: new  Abstract: Fair machine learning typically evaluates the societal costs of a decision policy -- the loss incurred by the decision-maker (DM) and the unfair distribution of costs among demographic groups of decision-subjects (DS) -- on static data, where reducing one often comes at the expense of the other. Yet decisions change the populations on which they act, and prior work has shown that fairness interventions designed on static data may fail to benefit, or even harm, the disadvantaged group over time. We investigate a complementary question: When can a policy that is Pareto-optimal on pre-deployment data be outperformed by a fairer policy once each is evaluated on the population it induces, reducing costs for one stakeholder without increasing them for the other? In a setting where qualification scores evolve in response to binary decisions and their outcomes, we show that under Rawlsian fairness, which never lowers any group's acceptance rate,
    
[^76]: FrontierReplay：面向混合语言模型的无分割精确前缀缓存

    FrontierReplay: Split-Free Exact Prefix Caching for Hybrid Language Models

    [https://arxiv.org/abs/2611.26020](https://arxiv.org/abs/2611.26020)

    FrontierReplay 通过在可预测的结构性边界处生成精确检查点并保持预填充前向传播不中断，为混合语言模型实现了无分割的精确前缀缓存，使状态构建成本比完整重计算降低 6-37 倍。

    

    混合语言模型将自注意力机制与循环层相结合，而循环层的原位状态更新使得前缀复用只能发生在已保存的精确位置。在粗粒度网格上保存状态会浪费可复用的前缀，而现有的细粒度精确缓存则必须在提示词中间的每个检查点处截断预填充前向传播，由此增加的调度步骤在负载压力下抵消了复用带来的收益。我们观察到，在多轮对话、智能体循环、文档问答和树搜索等场景中，前缀复用几乎总是终止于可预先预测的结构性边界，例如轮次结束处和已知分支点。我们提出了 FrontierReplay，这是一种前缀缓存系统，它在这些边界处生成精确的检查点，而无需截断预填充过程。在前向传播不中断运行的同时，FrontierReplay 记录循环层输入的一个短窗口，并调用跨层、仅状态的复现内核来构建目标状态，其每 token 的成本比完整重计算低 6-37 倍。

    arXiv:2611.26020v1 Announce Type: new  Abstract: Hybrid language models combine self-attention with recurrent layers whose in-place state updates restrict prefix reuse to exact saved positions. Saving states on a coarse grid wastes reusable prefixes, whereas existing fine-grained exact caches must cut the prefill forward pass at each mid-prompt checkpoint, adding scheduler steps that outweigh the reuse gains under load.   We observe that across multi-turn chat, agent loops, document question answering, and tree search, prefix reuse almost always ends at structural boundaries predictable in advance, such as turn ends and known branch points. We present FrontierReplay, a prefix caching system that materializes exact checkpoints at these boundaries without cutting the prefill. While the forward pass runs uncut, FrontierReplay logs a short window of recurrent-layer inputs and invokes a cross-layer, state-only replay kernel to build the target state 6-37x cheaper per token than full recompu
    
[^77]: 工作空间是否真的在工作？一种用于验证语言模型中全局工作空间计算的因果协议

    Does the Workspace Do the Work? A Causal Protocol for Verifying Global-Workspace Computation in a Language Model

    [https://arxiv.org/abs/2611.26018](https://arxiv.org/abs/2611.26018)

    本文提出一种因果测试协议，通过从其他输入移植消息来判定语言模型中的全局工作空间究竟是由谁执行计算的，可区分出忽略广播、纯转发广播、以及将广播与自身知识重组这三种不同情况，并在训练后的工作空间模型中同时发现了这三种结果。

    

    全局工作空间理论（GWT）是意识研究的主流理论之一，该理论认为，当信息被广播到使用该信息的专门化过程时，信息便成为有意识的。然而，在网络中构建一个工作空间并不能揭示训练后的网络究竟用它计算了什么：一个报告答案的模块可能将广播消息与自身知识相结合，也可能仅仅转发在其他地方计算出的答案，而任务准确率和消融实验无法区分这些情况。我们引入了一种因果协议，用于识别模块通信时究竟由谁执行计算。其核心测试是从另一个输入移植一条消息，其选择方式使得忽略、转发和重组该消息分别会预测出三种不同的答案。在专为转发或重组而构建的网络中，该测试在每种情况下都能正确判定其计算方式。在经过训练的工作空间模型中，该协议发现了全部三种结果：其中一个模型的广播内容几乎未被使用，而在两个仅在训练数据上存在差异的模型中……（摘要原文在此处截断）

    arXiv:2611.26018v1 Announce Type: cross  Abstract: Global Workspace Theory (GWT), a leading theory of consciousness, holds that information becomes conscious when it is broadcast to specialized processes that use it. Building a workspace into a network does not reveal what the trained network computes with it: a module that reports an answer may have combined a broadcast message with its own knowledge, or merely forwarded an answer computed elsewhere, and task accuracy and ablation cannot tell these apart. We introduce a causal protocol that identifies who computes when modules communicate. Its central test transplants a message from another input, chosen so that ignoring, relaying and recombining it predict three different answers. On networks built to relay or to recombine, the test assigns the right computation in every case. Among trained workspace models it finds all three outcomes: one leaves its broadcast nearly unused, and of two models that differ only in their training data, 
    
[^78]: 通过影子后验匹配强化少步扩散模型

    Reinforcing Few-Step Diffusion Models via Shadow Posterior Matching

    [https://arxiv.org/abs/2611.25998](https://arxiv.org/abs/2611.25998)

    提出影子后验匹配（SPM）方法，通过将生成样本重新加噪并从影子后验获取奖励信用，克服了用黑盒奖励强化少步扩散模型时采样器速度失配与教师锚定缺失两大结构性难题。

    

    少步扩散模型因其高效性而被广泛部署。然而，像多步模型的常规做法那样，用黑盒奖励来强化它们面临两个结构性挑战。首先，少步生成器是一个采样器而非分数模型，因此其速度并不是生成分布的前向速度。其次，它必须锚定到未蒸馏的教师模型，而二者之间不共享任何转移过程。现有的少步强化学习方法通过相对于教师模型的边缘KL散度来处理后者，但每一个方法都在其奖励信用中为前者付出代价。我们提出影子后验匹配（Shadow Posterior Matching, SPM），该方法将生成的样本重新加噪为影子输入，并从其影子后验中获得奖励信用。从KL正则化的奖励目标出发，我们推导出一个回归损失，它将奖励值转化为奖励方向，同时将生成器的输出锚定到教师模型，并使影子输入与部署时的输入保持对齐。应用于4步（原文在此处截断）

    arXiv:2611.25998v1 Announce Type: new  Abstract: Few-step diffusion models are widely deployed for their efficiency. However, reinforcing them with black-box rewards, as is routine for multi-step models, faces two structural challenges. First, a few-step generator is a sampler rather than a score model, so its velocity is not the forward velocity of the generated distribution. Second, it must be anchored to the undistilled teacher, which shares no transitions with it. Existing few-step RL methods handle the latter with a marginal KL to the teacher, but each pays for the former in its reward credit. We propose Shadow Posterior Matching (SPM), which renoises generated samples into shadow inputs and obtains the credit from their shadow posterior. From the KL-regularized reward objective, we derive a regression loss that turns reward values into a reward direction while anchoring the generator's outputs to the teacher and aligning the shadow inputs with the deployed inputs. Applied to 4-st
    
[^79]: 策略改进引导的持续强化学习重放方法

    Policy Improvement-Guided Replay for Continual Reinforcement Learning

    [https://arxiv.org/abs/2611.25989](https://arxiv.org/abs/2611.25989)

    该论文提出对比似然重放方法（CoLiR），基于策略改进原则和性能差分引理，在视觉语言模型的持续强化学习中选择性重放历史提示，从而在获得新能力的同时保留已取得的性能提升。

    

    持续强化学习（CRL）要求模型在获得新能力的同时保持之前任务上的性能。传统持续学习中常见的面向稳定性的视角强调知识保留和干扰缓解。作为对这一视角的补充，我们通过策略改进——强化学习的一个基本原则——重新审视视觉语言模型的持续强化学习，以保留已获得的性能提升。利用性能差分引理，我们通过当前策略在历史任务分布上的期望优势来刻画历史性能变化，从而启发使用当前策略对历史提示进行重放。为了分配有限的重放记忆，我们提出了对比似然重放（CoLiR），该方法将最终任务策略对正确回答的似然增益与初始策略对错误回答的似然降低进行对比。我们的分析将正确回答与……

    arXiv:2611.25989v1 Announce Type: cross  Abstract: Continual reinforcement learning (CRL) requires models to acquire new capabilities while retaining performance on previous tasks. A stability-oriented perspective common in conventional continual learning emphasizes knowledge preservation and interference mitigation. Complementing this perspective, we revisit CRL for vision-language models through policy improvement, a fundamental principle of RL, to preserve acquired performance gains. Using the performance difference lemma, we characterize historical performance changes through the current policy's expected advantages on historical task distributions, motivating replay of historical prompts with the current policy. To allocate limited replay memory, we propose Contrastive Likelihood Replay (CoLiR), which contrasts likelihood gains on correct responses from the final task policy with likelihood reductions on incorrect responses from the initial policy. Our analysis links correct-respo
    
[^80]: 看见显而易见之外：显著单正样本多标签学习的互补语义引导

    Seeing Beyond the Obvious: Complementary Semantic Guidance for Salient Single-Positive Multi-Label Learning

    [https://arxiv.org/abs/2611.25986](https://arxiv.org/abs/2611.25986)

    提出CoST框架，利用图像-文本语义对应从未知类别中挖掘互补候选样本并构建覆盖原型，弥补显著原型对隐藏正样本视觉多样性捕捉不足的问题，提升显著单正多标签学习的监督效果。

    

    显著单正样本多标签学习为每张图像分配其最显著的正标签，而将其他类别留为未标注。现有方法从已观察到的正样本中构建类别级显著原型，以引导未观察类别的识别。然而，由于显著标注偏向视觉上突出的实例，这些原型可能无法充分捕捉隐藏正样本的视觉多样性，导致正样本监督不足。为解决这一局限，我们提出了CoST，一种覆盖引导的互补软目标细化框架，通过互补的类别语义信息来补充显著原型。具体而言，CoST利用任务学习的类别信息和图像-文本语义对应关系，从未知类别中识别语义相关的候选样本，随后将选出的候选样本聚合为类别级覆盖原型，以补充语义表示。

    arXiv:2611.25986v1 Announce Type: new  Abstract: Salient Single-Positive Multi-Label Learning (SalSPML) assigns each image its most salient positive label, leaving other classes unlabeled. Existing methods construct class-level Salient Prototypes from observed positives to guide the recognition of unobserved classes. However, since salient annotations favor visually prominent instances, these prototypes may not fully capture the visual diversity of hidden positives, resulting in insufficient positive supervision. To address this limitation, we propose CoST, a Coverage-guided Complementary Soft-Target Refinement framework that supplements Salient Prototypes with complementary class-semantic information. Specifically, CoST identifies semantically relevant candidates from unknown classes by leveraging task-learned category information and image-text semantic correspondence. The selected candidates are then aggregated into class-wise Coverage Prototypes to complement the semantic represent
    
[^81]: 预测的深度：语言模型解析词语的层级位置可预测人类阅读时间

    The Depth of a Prediction: Where Language Models Resolve a Word Predicts Human Reading Time

    [https://arxiv.org/abs/2611.25970](https://arxiv.org/abs/2611.25970)

    论文提出“解析深度”这一新指标，用以衡量语言模型在哪一层完成对一个词的预测，并证明模型解析越晚的词及其后续词会被人类读得更慢，且解析深度能在最终层意外度和熵之外显著提升对人类阅读时间的预测。

    

    来自语言模型最后一层的意外度预测了人们阅读一个词的时长，然而模型是在许多层中逐步计算这一预测的。我们使用调谐透镜在每一层读取每个词的预测，并用“解析深度”来概括这条逐层意外度轨迹——即其变化所围绕的中心深度。在八个英语眼动追踪和自定步速阅读语料库（81,957 个词）以及十个 GPT-2、OPT 和 Pythia 模型上，模型解析较晚的词，以及这些词之后的词，与同等意外程度的词相比被阅读得更慢。在留出集预测上，解析深度在最终层意外度、熵以及最佳单层之外额外提升了每词 0.26 毫奈特的预测能力（合并 p = 0.001），约为熵所带来的增益的三分之一，且每个模型家族均有所获益，其中 GPT-2 和 Pythia 各通过一个语料库体现。净增益在不同语料库之间存在差异（随机效应均值 0.07，p = 0.37）；相对于层序打乱的解析深度——其引入了同样多的系数……

    arXiv:2611.25970v1 Announce Type: new  Abstract: Surprisal from a language model's final layer predicts how long people read a word, yet the model computes its prediction across many layers. We read each word's prediction at every layer with the tuned lens and summarize this layerwise surprisal trajectory by its resolution depth, the depth around which its change is centered. On eight English eye-tracking and self-paced reading corpora (81,957 words) and ten GPT-2, OPT and Pythia models, words that a model resolves later, and the words after them, are read more slowly than equally surprising words. Resolution depth improves held-out prediction beyond final-layer surprisal, entropy and the best single layer by 0.26 millinats per word (pooled p = 0.001), about a third of entropy's gain, and each model family gains, GPT-2 and Pythia through one corpus each. The net gain varies between corpora (random-effects mean 0.07, p = 0.37); against layer-shuffled depth, which adds as many coefficien
    
[^82]: 大规模电动汽车充电的功率设定点跟踪：基于自整定约束价格的变压器约束充电策略学习

    Power Setpoint Tracking for Large-Scale EV Charging: Transformer-Constrained Charging Policy Learning with Self-Tuning Constraint Prices

    [https://arxiv.org/abs/2611.25965](https://arxiv.org/abs/2611.25965)

    该论文提出TC-CPL方法，通过比例-积分自整定约束价格、热启动残差策略和变压器感知编码器，在大规模EV充电功率设定点跟踪中实现近乎零变压器过载并显著降低跟踪误差，且无需观测车辆离开时间或荷电状态。

    

    在功率设定点跟踪（PST）中，运营方需要按照已购买的充电功率曲线运行，同时避免共享变压器过载或对车辆充电不足。PST被建模为一个受约束的马尔可夫决策过程，在变压器过载约束和平均用户满意度约束下最小化跟踪误差。变压器约束充电策略学习（TC-CPL）在软演员-评论家（soft actor-critic）算法基础上进行了扩展：基于实测违规量的比例-积分拉格朗日乘子自整定更新、以现有调度器的演示数据热启动的残差修正项，以及由所有充电端口共享的变压器感知编码器。该方法无需观测车辆的离开时间或荷电状态。仿真实验（五个随机种子、100个配对日）将其与现有方法轮询调度（Round-Robin）进行对比。在一个拥有600个充电端口、122个变电站的区域内，原始过载从每天3.59 kWh降至0.0017 kWh（过载种子日比例从77%降至0.4%），绝对跟踪误差降低27%，输送能量增加16%，用户满意度（原文在此处截断）。

    arXiv:2611.25965v1 Announce Type: new  Abstract: In power setpoint tracking (PST), an operator follows a purchased charging profile without overloading shared transformers or undercharging vehicles. PST is cast as a constrained Markov decision process that minimizes tracking error subject to overload and mean user-satisfaction constraints. Transformer-Constrained Charging Policy Learning (TC-CPL) extends soft actor-critic with proportional-integral Lagrange-multiplier updates from measured violations, a residual on the incumbent dispatcher that is warm-started from its demonstrations, and a transformer-aware encoder shared by all ports. It observes no departure times or states of charge. Simulations (five seeds, 100 paired days) compare it with the incumbent, Round-Robin. On a 600-port, 122-substation district, raw overload falls from 3.59 to 0.0017 kWh/day (overloaded seed-days from 77% to 0.4%), and absolute tracking error falls by 27%. Delivered energy rises by 16%, and satisfaction
    
[^83]: 使用Kolmogorov-Arnold单元的梯度提升方法：超越成对交互作用的可解释回归

    Gradient Boosting with Kolmogorov-Arnold Units for Interpretable Regression beyond Pairwise Interactions

    [https://arxiv.org/abs/2611.25942](https://arxiv.org/abs/2611.25942)

    MJBoost以带惩罚样条的Kolmogorov-Arnold单元作为弱学习器进行梯度提升，在保持精确函数型ANOVA分解可解释性的同时，捕捉超越成对交互作用的高阶交互效应。

    

    面向表格回归的可解释模型（如可解释提升机EBM）将预测限制为主效应和成对交互作用，以便模型中的每个部分都可以被绘制出来。当目标变量依赖于三个或更多输入的联合作用时，这种限制会带来精度损失，人们通常只能退而求其次，使用带有事后解释的黑箱模型。本文提出MJBoost，这是一种梯度提升方法，其弱学习器为单个Kolmogorov-Arnold单元 $g(\sum_p \phi_p(x_p))$，其中内外函数均采用惩罚样条。每个单元从残差的主Hessian方向出发，通过交替惩罚最小二乘法以闭式解进行拟合，类别型变量和缺失值可直接进入单元。由于每个单元仅通过一个内函数依赖于每个输入，拟合出的模型具有精确的函数型ANOVA分解，可分为主效应、成对效应和高阶残差部分（原文摘要在此处截断）。

    arXiv:2611.25942v1 Announce Type: new  Abstract: Interpretable models for tabular regression, such as the explainable boosting machine (EBM), restrict a prediction to main effects and pairwise interactions so that every part of it can be plotted. When the target depends on three or more inputs jointly, this restriction costs accuracy, and a black-box model with post hoc explanations becomes the usual fallback. Here we present MJBoost, a gradient boosting method whose weak learner is a single Kolmogorov-Arnold unit $g(\sum_p \phi_p(x_p))$ with penalized spline inner and outer functions. Each unit is fitted in closed form by alternating penalized least squares, starting from principal Hessian directions of the residual, and categorical and missing inputs enter the units directly. Because a unit depends on each input only through one inner function, the fitted model has an exact functional ANOVA decomposition into main effects, pair effects and a higher-order remainder, which one-dimensio
    
[^84]: DTS：基于增量变化的瓦片评分方法，实现高效的长上下文稀疏注意力

    DTS: Delta-Based Tile Scoring for Efficient Long-Context Sparse Attention

    [https://arxiv.org/abs/2611.25938](https://arxiv.org/abs/2611.25938)

    提出了一种无需训练、内容自适应的瓦片评分方法 DTS，它根据局部向量变化选择查询和键锚点来高效评分稀疏注意力的块，并在冗余区域少选、异构区域多选锚点以兼顾效率与精度，从而在几乎不损失模型长上下文性能的前提下显著降低长上下文预填充的计算开销。

    

    大语言模型（LLM）越来越需要处理长文档和代码库，这使得长上下文注意力成为预填充阶段的主要瓶颈。块稀疏注意力通过跳过注意力矩阵中不重要的块来减少计算量，但准确地对块进行评分本身可能代价高昂。我们提出了基于增量变化的瓦片评分方法，这是一种无需训练、内容自适应的稀疏块注意力瓦片评分方法。DTS 根据局部向量变化来选择查询和键的锚点，在冗余区域使用更少的代表性元素，同时在异构区域保留更多细节。为了加速锚点选择，并行增量打包将查询和键序列划分为与瓦片对齐的块，并并发地对这些块执行有界局部扫描。选定的瓦片随后使用原始的查询、键和值向量进行计算。在 LLaMA-3.1-8B-Instruct 上，DTS 将平均 RULER 和 LongBench 分数保持在 0.10 以内（摘要在此处似乎被截断）。

    arXiv:2611.25938v1 Announce Type: new  Abstract: Large language models (LLMs) increasingly need to process long documents and codebases, making long-context attention a major prefill bottleneck. Block sparse attention reduces computation by skipping unimportant blocks of the attention matrix, but scoring blocks accurately can itself be expensive. We propose Delta-Based Tile Scoring (DTS), a training-free, content-adaptive tile scoring method for sparse block attention. DTS selects query and key anchors according to local vector changes, using fewer representatives in redundant regions while retaining more detail in heterogeneous regions. To accelerate anchor selection, parallel delta packing partitions the query and key sequences into tile-aligned chunks and performs bounded local scans over these chunks concurrently. The selected tiles are then computed using the original query, key, and value vectors. On LLaMA-3.1-8B-Instruct, DTS keeps average RULER and LongBench scores within 0.10 
    
[^85]: 用于密度比估计的Hellinger决策树：有限样本保证

    Hellinger Decision Trees for Density Ratio Estimation: Finite-Sample Guarantees

    [https://arxiv.org/abs/2611.25936](https://arxiv.org/abs/2611.25936)

    提出了一种通过牛顿提升在轴对齐决策树集成上最小化平衡Hellinger风险来直接估计对数密度比的方法，并建立了有限样本保证，其收敛速率在对数因子范围内达到极小极大最优的L²速率。

    

    密度比在双样本比较、分布偏移适应以及生成模型评估中起着核心作用。然而，通过对底层密度分别进行估计来估计密度比可能是不必要的困难。在本文中，我们开发了一种直接估计对数密度比的方法，即在轴对齐决策树的加性集成上最小化平衡的Hellinger风险，并使用牛顿提升进行训练。我们为该算法产生的估计器建立了有限样本保证，并推导出了显式的收敛速率。在适当的尾部条件和Hölder光滑性条件下，再加上关于每次提升迭代所取得进展的额外假设，我们的速率在忽略对数因子的情况下与函数估计的经典极小极大最优 $\mathscr{L}^{2}$ 速率相匹配。一个关键的技术贡献是利用特殊的指数结构进行的优化分析。

    arXiv:2611.25936v1 Announce Type: cross  Abstract: Density ratios play a central role in two-sample comparison, distribution-shift adaptation, and generative-model evaluation. However, estimating a density ratio through separate estimation of the underlying densities can be unnecessarily challenging. In this paper, we develop a direct approach to the log density ratio estimation by minimizing a balanced Hellinger risk over additive ensembles of axis-aligned decision trees, trained using Newton boosting. We establish finite-sample guarantees for the estimator produced by the algorithm and derive an explicit convergence rate. Under suitable tail and H\"{o}lder smoothness conditions, together with an additional assumption on the progress made at each boosting iteration, our rate matches, up to logarithmic factors, the classical minimax-optimal $ \mathscr{L}^{2} $ rate for function estimation. A key technical contribution is an optimization analysis that exploits the special exponential st
    
[^86]: 一种基于机床内部信号重构铣削力的轻量化混合方法

    A Lightweight Hybrid Approach for Milling Force Reconstruction from Internal Machine Signals

    [https://arxiv.org/abs/2611.25921](https://arxiv.org/abs/2611.25921)

    本文提出一种轻量化混合方法，利用机床内部信号和加工元数据，通过监督学习模型预测平均铣削力，从而在无需额外传感器的既有机床上实现基于力的铣削过程监测。

    

    基于铣削力的数据驱动监测可以支持工艺优化和状态监测等先进制造服务。然而，其工业落地常常受到昂贵且具侵入性的硬件以及大量训练数据需求的阻碍。因此，本文提出了一种利用机床内部信号和加工元数据的混合方法，无需额外传感器即可在既有（brownfield）机床上实现基于力的监测。常用的机床信号采样率相对较低，无法直接表征高频的铣削力变化，尤其是在较高的主轴转速下。与学习完整的力信号（这会增加模型复杂度、数据需求和过拟合风险）不同，本文采用监督式机器学习模型，仅从选定的机床信号特征和加工参数预测平均铣削力。这一低频估计值再用于对统计……（摘要在此处被截断）

    arXiv:2611.25921v1 Announce Type: new  Abstract: Data-driven monitoring based on milling force can enable advanced manufacturing services including process optimization and condition monitoring. However, industrial implementation is often hindered by costly and intrusive hardware and large training data requirements. This paper therefore presents a hybrid approach using machine-internal signals and machining metadata to enable force-based monitoring on brownfield machines without additional sensors. Commonly available machine signals have relatively low sampling rates and cannot directly represent high-frequency milling-force variations, especially at higher spindle speeds. Instead of learning the complete force signal, which would increase model complexity, data requirements, and overfitting risk, a supervised machine-learning model predicts only the average milling force from selected machine-signal features and machining parameters. This low-frequency estimation parameterizes a stat
    
[^87]: LIFT：基于单细胞快照的潜空间流匹配用于概率性细胞命运动力学预测

    LIFT: Latent Flow Matching for Probabilistic Cell Fate Dynamics with Single Cell Snapshots

    [https://arxiv.org/abs/2611.25917](https://arxiv.org/abs/2611.25917)

    LIFT是一个基于潜空间流匹配的概率性细胞命运预测框架，通过变分推断编码细胞状态的一对多转变关系，结合Mean Flow解码器实现单次评估即可高效采样，能够从谱系追踪数据中学习细胞条件分化结局并进行大规模蒙特卡洛采样预测。

    

    谱系追踪通过克隆祖先关系连接细胞状态，但通常无法确定唯一的“源细胞-后代细胞”配对关系。因此，从这类数据中学习需要一个能够建模条件结果的模型，而不仅仅是终端的群体分布。我们提出了LIFT，一个基于谱系信息耦合的潜空间流框架，用于概率性细胞状态生成。变分推断将多种可能的结果编码在潜空间中，而条件先验使得仅需从初始状态即可进行采样。Mean Flow解码器可直接预测未来细胞状态，每次采样仅需一次解码器评估，这使得通过大规模蒙特卡洛采样预测细胞命运变得十分便捷。这一构造将一对多关系的表示与高效采样以及条件预测误差分析相结合。在合成数据集和真实数据集上的实验证明了LIFT在恢复群体分布方面的能力。

    arXiv:2611.25917v1 Announce Type: new  Abstract: Lineage tracing links cell states through clonal ancestry, but does not generally identify unique source-descendant pairings. Learning from such data therefore requires a model of conditional outcomes, rather than only the terminal population distribution. We introduce LIFT, a latent flow framework for probabilistic cell-state generation from lineage-informed couplings. Variational inference encodes alternative outcomes in a latent space, while a conditional prior enables sampling from the initial state alone. A Mean Flow decoder predicts future cell state directly, requiring one decoder evaluation per sampling, which makes it convenient to predict cell fate through large scale Monte-Carlo sampling. This construction combines a representation of one-to-many relationships with efficient sampling and an analysis of conditional prediction error. Experiments on synthetic and real datasets demonstrate LIFT's ability to recover both population
    
[^88]: 医学大语言模型中的可验证逻辑：智能体临床推理优化与神经符号接地的批判性综述

    Verifiable Logic in Medical LLMs: A Critical Review of Agentic Clinical Reasoning Optimization and Neuro-Symbolic Grounding

    [https://arxiv.org/abs/2611.25898](https://arxiv.org/abs/2611.25898)

    这篇综述指出临床大语言模型的最终答案正确并不代表推理有效，并系统评述了幻觉成因、智能体幻觉级联、过程奖励建模与神经符号接地四种范式，其中带符号否决层的系统能在其编码知识范围内提供可验证的安全保证。

    

    临床大语言模型通常以最终答案是否正确作为评判标准。然而，模型可能通过无效的推理得出流畅、看似合理的输出，而静态基准测试无法检测到这种情况；自信的幻觉、因果无效的推理以及虚假的人口统计学捷径都可能通过传统评估。本批判性综述依次涵盖四种范式：临床幻觉如何产生以及为什么静态指标会漏检它；智能体幻觉级联——即多智能体和工具增强系统会传播并放大早期推理错误；过程奖励建模作为步骤级监督；以及神经符号接地——形式逻辑引擎、本体推理机和知识图谱对神经输出施加可验证的约束。在其编码的知识范围内，具有符号否决层的系统能够提供相对于其编码知识成立的保证，并在安全性指标上报告了显著的提升。

    arXiv:2611.25898v1 Announce Type: cross  Abstract: Clinical large language models are usually judged by whether the final answer is right. Yet a model can reach a fluent, plausible output through invalid reasoning that static benchmarks do not detect, and confident hallucination, causally invalid inference, and spurious demographic shortcuts can pass conventional evaluation. This critical review covers four paradigms in turn: how clinical hallucination arises and why static metrics miss it; the agentic hallucination cascade, in which multi-agent and tool-augmented systems propagate and amplify early reasoning errors; process reward modeling as step-level supervision; and neuro-symbolic grounding, in which formal logic engines, ontology reasoners, and knowledge graphs place verifiable constraints on neural output. Within their encoded domains, systems with symbolic veto layers offer guarantees that hold relative to their encoded knowledge, and they report large gains on the safety measu
    
[^89]: 解释的代价：要求可验证产物损失了什么，又换回了什么

    The Price of an Explanation: What Demanding Verifiable Artifacts Costs, and What It Buys Back

    [https://arxiv.org/abs/2611.25896](https://arxiv.org/abs/2611.25896)

    该论文在ARC-AGI-2上量化了要求LLM生成可执行可验证产物（程序）的代价（得分从77%降至57%），并指出真正的瓶颈在于让解释“泛化”而非“找到”解释——因为多数仅由只答组解出的任务中，程序组其实已拥有能复现全部演示、却仍在测试上失败的程序。

    

    生成模型会产生许多候选解，却没有任何内在信号可用于对它们排序。一种缓解手段是要求提供“可验证产物”：即一个可对照现有证据执行的程序。人们普遍担心，要求可执行验证会在解答质量和计算预算两方面带来成本。我们在包含120个任务的ARC-AGI-2评测集上测量了这两类成本。要求LLM提供可验证产物后，其得分从120个任务中的93个降至68个（77%降至57%）。在损失的这25个任务中，有21个任务在我们运行过的任何程序生成配置下——无论何种预算、无论采样还是搜索——都未能解决；然而，在仅由“只给答案”组解决的28个任务中，有27个任务上某些程序组所持有的程序能够复现全部演示示例，却仍在测试上失败。瓶颈在于获得能够“泛化”的解释，而非“找到”某个解释。对两个……（原文摘要在此处截断）

    arXiv:2611.25896v1 Announce Type: new  Abstract: Generative models produce many candidate solutions but provide no intrinsic signal by which to rank them. One mitigation is to demand a \emph{verifiable artifact}: a program that can be executed against the available evidence. A common concern is that requiring executable verification incurs costs in both solution quality and computational budget. We measure these two costs on the $120$-task ARC-AGI-2 evaluation set. Demanding a verifiable artifact from an LLM reduced the score from $93$ to $68$ of $120$ tasks ($77\%$ to $57\%$). Of those $25$ tasks, $21$ were reached by no program-producing configuration we ran, at any budget, by sampling or by search; yet for $27$ of the $28$ tasks that only the answer-only arm solved, some program arm held a program that reproduced every demonstration and still failed the test. The bottleneck is obtaining an explanation that \emph{generalizes}, not finding one. A short evolutionary search over two alr
    
[^90]: 一种调度无法适用于所有情况：扩散语言模型的自适应推理调度

    One Schedule Does Not Fit All: Adaptive Inference Schedules for Diffusion Language Models

    [https://arxiv.org/abs/2611.25895](https://arxiv.org/abs/2611.25895)

    本文提出 TCDA，一种具有显式误差保证的自适应部署规则，可根据任务、模型和采样器为扩散语言模型选择满足质量要求的最低成本推理调度，将去噪器评估次数最多减少 3.4 倍。

    

    扩散语言模型通常在所有工作负载中使用固定的去噪调度。我们证明，在保持质量的前提下，最廉价的调度取决于任务、模型和采样器。我们提出了任务校准去噪分配（Task-Calibrated Denoising Allocation，TCDA），这是一种具有显式误差保证的部署规则。给定候选采样器、配对校准数据、效用容忍度以及防止退化的防护措施，TCDA 使用同步单侧自助法界限来认证哪些候选方案同时满足效用和防护标准。它部署经过认证的最廉价候选方案，或回退到完整调度。部署不可行调式的概率渐近地至多为 alpha。候选方案包括早期前缀解码、状态复用续写、缩减时间网格和置信度阈值解码。在七个数据集和两个模型系列上，TCDA 认证了去噪器评估次数的显著减少：在 XSum 上减少 3.4 倍，在 WMT 上减少 2.0 倍。

    arXiv:2611.25895v1 Announce Type: new  Abstract: Diffusion language models typically use a fixed denoising schedule across workloads. We show that the cheapest schedule that preserves quality depends on the task, model, and sampler. We propose Task-Calibrated Denoising Allocation (TCDA), a deployment rule with an explicit error guarantee. Given candidate samplers, paired calibration data, a utility tolerance, and guardrails against degeneration, TCDA uses simultaneous one-sided bootstrap bounds to certify which candidates meet both the utility and guardrail criteria. It deploys the cheapest certified candidate or falls back to the full schedule. The probability of deploying an infeasible schedule is asymptotically at most alpha. Candidates include early prefix decoding, state-reusing continuation, reduced time grids, and confidence-threshold decoding. Across seven datasets and two model families, TCDA certifies reductions in denoiser evaluations by factors of 3.4 on XSum and 2.0 on WMT
    
[^91]: 自监督视觉骨干网络的参数高效适配在雷达微多普勒人体活动识别中经受住分布偏移的考验

    Parameter-Efficient Adaptation of Self-supervised Vision Backbones Survives Distribution Shift in Radar Micro-Doppler HAR

    [https://arxiv.org/abs/2611.25874](https://arxiv.org/abs/2611.25874)

    本文首次将自监督视觉基础模型DINOv3与PiSSA参数高效微调相结合应用于雷达微多普勒人体活动识别，证明PEFT是唯一能同时避免跨数据集负迁移和跨方向欠适应的适配策略，其Small变体以4.5倍更少的总参数量即可达到先前最先进方法的同分布精度。

    

    微多普勒雷达能够实现非接触式、保护隐私的人体活动识别（HAR），但其部署鲁棒性仍未得到充分研究。我们采用被试分离的同分布交叉验证以及少样本跨年龄、跨方向和跨数据集迁移实验，比较了全量微调、冻结探测和参数高效微调（PEFT）三种策略。结果显示，全量微调出现了负的跨数据集迁移，而冻结特征在跨方向场景下适应不足，只有PEFT避免了这两种失败。我们将DINOv3与全层主奇异方向适配方法相结合，据我们所知，这是自监督图像基础模型骨干首次应用于雷达微多普勒人体活动识别。DINOv3-PiSSA在所有评估设置中均在PEFT方法中排名第一。在我们的评估协议下，其Small变体仅使用3%的可训练参数和少4.5倍的总参数量，就达到了先前Base尺寸PEFT最先进方法的同分布准确率。

    arXiv:2611.25874v1 Announce Type: new  Abstract: Micro-Doppler radar enables contactless, privacy-preserving human activity recognition (HAR), but deployment robustness remains underexplored. We compared full fine-tuning, frozen probing, and parameter-efficient fine-tuning (PEFT) using subject-disjoint in-distribution cross-validation and few-shot cross-age, cross-orientation, and cross-dataset transfer. Full fine-tuning suffered negative cross-dataset transfer, while frozen features under-adapted on cross-orientation. Only PEFT avoided both failures. We paired DINOv3 with all-layer principal-singular-direction adaptation (PiSSA), to our knowledge the first application of a self-supervised image foundation backbone to radar micro-Doppler HAR. DINOv3-PiSSA ranked first among PEFT methods in every evaluated setting. Under our protocol, its Small variant matched the in-distribution accuracy of the prior Base-size PEFT state of the art using 4.5x fewer total parameters, with only 3% traina
    
[^92]: X Hedging：迈向可解释人工智能期权对冲框架

    X Hedging: towards an eXplainable artificial intelligence option Hedging framework

    [https://arxiv.org/abs/2611.25870](https://arxiv.org/abs/2611.25870)

    本文提出X Hedging框架，用LightGBM树集成模型替代神经网络进行期权对冲策略学习，在保持对冲性能的同时实现了计算高效与更好的可解释性，弥补了Deep Hedging的不足。

    

    我们提出了X Hedging，这是对Deep Hedging框架的一种替代方案，旨在解决基于神经网络的对冲策略在计算和可解释性方面的局限性。Deep Hedging利用神经网络学习灵活的对冲策略，但依赖不透明的模型，且通常需要大量训练数据，这会使解释变得困难，并限制其在数据受限环境中的有效性。X Hedging朝着解决这些局限性迈出了一步，它用树集成方法（具体为LightGBM）替代神经网络来估计对冲策略。这种设计提供了一种计算高效的替代方案，同时保持了强大的对冲性能，并朝着更加以用户为中心的可解释性迈进。为了将LightGBM集成到强化学习风格的对冲框架中，我们引入了一种优化方法，该方法依次训练树集成模型，同时通过随机化引入随机性。

    arXiv:2611.25870v1 Announce Type: new  Abstract: We propose X Hedging, an alternative to the Deep Hedging framework that addresses computational and explainability limitations of neural-network-based hedging strategies. Deep Hedging leverages neural networks to learn flexible hedging policies but relies on opaque models and typically requires large training datasets, which can complicate interpretation and limit its effectiveness in data-constrained settings. X Hedging takes a step toward addressing these limitations by replacing neural networks with tree ensemble methods, specifically LightGBM, to estimate hedging strategies. This design provides a computationally efficient alternative while preserving strong hedging performance and moving toward more user-centered explainability. To integrate LightGBM into a reinforcement-learning-style hedging framework, we introduce an optimization approach that sequentially trains tree-ensemble models while introducing stochasticity through random
    
[^93]: tanh 网络的满秩与稳定采样

    Full Rank and Stable Sampling of $\tanh$ Networks

    [https://arxiv.org/abs/2611.25867](https://arxiv.org/abs/2611.25867)

    本文刻画了 tanh 网络中实现映射微分为满秩的参数集（与正则参数一致），并证明几乎任意 2P+1 个采样点即可保证一致范数 Lipschitz 稳定采样、参数可识别性（允许隐藏神经元置换与符号变化），以及投影梯度下降与投影 Gauss–Newton 方法的局部收敛。

    

    我们研究具有固定架构、含 $P$ 个参数的前馈 $\tanh$ 网络从有限个点采样值中的恢复问题。首先，我们刻画了实现映射的微分具有满秩的参数集合，并证明该集合与 Vlačić 和 Bölcskei (2021) 意义下的正则参数相重合。其次，我们利用这一刻画证明：几乎任意一组 $2P+1$ 个采样点都能在该参数集的紧子集上产生一致范数意义下的 Lipschitz 稳定性。第三，我们证明对于精确观测，投影梯度下降法和投影 Gauss–Newton 法从足够接近的可行初始点出发可局部收敛。同样的采样值还能确定任意参数下所实现的函数，并在允许隐藏神经元的置换与符号变化的前提下识别出被刻画参数集中的参数。最后，我们针对实解析（函数）开发了一个一般性框架。

    arXiv:2611.25867v1 Announce Type: cross  Abstract: We study the recovery of feedforward $\tanh$ networks of fixed architecture with $P$ parameters from finitely many point evaluations. First, we characterize the set of parameters where the differential of the realization map has full rank and show that it coincides with the regular parameters in the sense of Vla{\v{c}}i{\'c} and B{\"o}lcskei (2021). Second, we make use of this characterization to prove that almost every tuple of $2P+1$ sampling points yields uniform-norm Lipschitz stability on compact subsets of this parameter set. Third, we prove that for exact observations, projected gradient descent and projected Gauss--Newton converge locally from sufficiently close feasible initial guesses. The same evaluations also determine the realized function for arbitrary parameters and identify parameters in the characterized set up to permutations and sign changes of hidden neurons. Finally, we develop a general framework for real-analytic
    
[^94]: 面向天气预报的可解释原型条件化自适应方法

    Interpretable Prototype-Conditioned Adaptation for Weather Forecasting

    [https://arxiv.org/abs/2611.25848](https://arxiv.org/abs/2611.25848)

    该论文提出ProtoCast方法，通过组特定的原型库，将每组的历史特征与时间变化和跨变量注意力提供的预报上下文相结合，进行演化感知的原型匹配，从而在适应不同地点和变量组的同时实现可解释的天气预报。

    

    数据驱动的天气预报需要区分局部状态及其时间演化。然而，学习到的模式在神经预报器中仍然是隐式的，使其影响难以解释。若无法区分局部演化，可能会导致对发展方式不同的地点给出相似的预报，从而增加预测误差。这促使我们需要显式的参考来区分和解释局部模式。原型能够提供显式的模式参考，但样本级或任务级的条件化尚未解决如何适应不同地点和变量组的问题。为解决这一问题，我们设计了ProtoCast，它使用组特定的原型库来适应局部特征，同时考虑跨变量依赖关系。具体而言，（1）我们将每组的历史特征和时间变化与来自跨变量注意力的预报上下文相结合，实现演化感知的原型匹配。（2）稀疏混合的……（原文摘要在此处截断）

    arXiv:2611.25848v1 Announce Type: new  Abstract: Data-driven weather forecasting requires distinguishing local states and their temporal evolution. However, learned patterns remain implicit in neural forecasters, making their influence difficult to interpret. Failing to distinguish local evolution can lead to similar forecasts for locations that develop differently, increasing prediction error. This motivates explicit references for distinguishing and interpreting local patterns. Prototypes provide explicit pattern references, but sample- or task-level conditioning leaves open how to adapt to distinct locations and variable groups. To deal with this problem, we design ProtoCast, which uses group-specific prototype banks to adapt local features while accounting for cross-variable dependencies. Specifically, (1) we combine each group's historical features and temporal change with forecast context from cross-variable attention for evolution-aware prototype matching. (2) Sparse mixtures of
    
[^95]: Rank2Replace：基于VLM排序的槽位替换实现分布外泛化

    Rank2Replace: VLM-Ranked Slot Replacement for Out-of-Distribution Generalization

    [https://arxiv.org/abs/2611.25834](https://arxiv.org/abs/2611.25834)

    该论文提出Rank2Replace方法，利用预训练视觉-语言模型按任务相关性对对象槽位进行排序并执行同排名槽位替换增强，在无需环境标签的情况下将任务相关证据知识迁移到小型预测器，显著提升分布外泛化性能。

    

    虚假相关性可以使图像预测器在分布内获得很高的准确率，但在分布偏移下泛化能力很差。我们提出了Rank2Replace，该方法通过训练时数据增强，将预训练视觉-语言模型（VLM）中关于任务相关证据的知识迁移到小型预测器中，以实现分布外（OOD）泛化。VLM按任务相关性对以对象为中心的槽位进行排序，并引导基于排名的替换，使用来自另一张图像的相同排名的槽位嵌入。这种增强保留了原始图像的标签，鼓励模型在偶然线索变化时依赖保留的任务证据，并支持与现有OOD方法的结合。在共享冻结槽位嵌入的四个数据集和六个划分上，默认的交叉熵配置在没有环境标签或数据集特定替换的情况下，达到了80.43%的宏观平均平均组准确率和65.28%的最差组准确率。

    arXiv:2611.25834v1 Announce Type: cross  Abstract: Spurious correlations can give image predictors high in-distribution accuracy but poor generalization under distribution shift. We introduce $\textbf{Rank2Replace}$, which transfers knowledge of task-relevant evidence from pretrained vision--language models (VLMs) to small predictors through training-time augmentation for out-of-distribution (OOD) generalization. The VLM ranks object-centric slots by task relevance and guides rank-dependent replacement with same-rank slot embeddings from another image. This augmentation retains the original image's label, encouraging reliance on preserved task evidence as incidental cues change, and supports combination with existing OOD methods. Across four datasets and six splits under shared frozen slot embeddings, the default cross-entropy configuration reaches macro-averaged average-group and worst-group accuracies of $80.43\%$ and $65.28\%$ without environment labels or dataset-specific replaceme
    
[^96]: 超越对齐：白化几何、轮廓合成与异构集成用于跨环境WiFi动作识别

    Beyond Alignment: Whitening Geometry, Profile Synthesis, and Heterogeneous Ensembles for Cross-Environment WiFi Action Recognition

    [https://arxiv.org/abs/2611.25831](https://arxiv.org/abs/2611.25831)

    该论文发现跨环境WiFi动作识别的性能崩塌根源在于编码器对新环境的外推失败而非域不对齐，证明对齐类方法反而有害，并提出结合白化几何、轮廓合成与异构集成的Dec-H方法作为有效解决途径。

    

    跨环境部署是实用化WiFi感知的核心障碍：在训练环境中达到0.89以上top-1准确率的处理流水线，在仅有K-shot标签的留一域外（LODO）评估下会骤降至0.408。一项上限分析表明，这种性能崩塌并非对齐问题：动作信号在已见环境中很强（约0.89 K10），环境身份信息可以从编码器中被解码出来（0.965），而失败集中在编码器对新环境的泛化外推上。一个预先注册的证伪实验链证实，“对齐域”的方法——训练时对齐、读出层原型对齐、对抗混淆以及协方差/子空间对齐（CORAL降低3.8个百分点，子空间对齐降低8.3个百分点）——反而会损害性能，且源域统计量与跨环境贡献呈负相关。有效的手段在别处；我们将它们组合成Dec-H方法：(i) 闭式解的源域类内……

    arXiv:2611.25831v1 Announce Type: new  Abstract: Cross-environment deployment is the central obstacle to practical WiFi sensing: a pipeline reaching 0.89+ top-1 in its training environments collapses to 0.408 under Leave-One-Domain-Out (LODO) evaluation with only K-shot labels. A ceiling analysis shows the collapse is not an alignment problem: the action signal is strong across seen environments (about 0.89 K10), environment identity is decodable from the encoder (0.965), and the failure concentrates in the encoder's extrapolation to the novel environment. A pre-registered falsification chain confirms that "align the domains" methods--training-time alignment, readout-level prototype alignment, adversarial confusion, and covariance/subspace alignment (CORAL -3.8pp, subspace -8.3pp)--degrade performance, and that source-domain statistics anti-correlate with cross-environment contribution. The effective levers lie elsewhere; we combine them into Dec-H: (i) closed-form source within-class 
    
[^97]: 一种面向图的稀疏低秩双团分解方法

    A Sparse Low-Rank Biclique Decomposition for Graphs

    [https://arxiv.org/abs/2611.25826](https://arxiv.org/abs/2611.25826)

    提出符号双团分解（SBD），一种面向整数加权图的稀疏低秩分解方法，能够在保持图的结构稀疏性、离散性和可解释性的同时，有效恢复噪声图中的块结构并支持高效稀疏矩阵运算。

    

    低秩加稀疏分解（如鲁棒PCA）通常将低秩成分建模为稠密的，而将稀疏成分视为损坏或噪声。这种视角并不适合图的邻接矩阵，因为图中的稀疏性是结构性的，而稠密的低秩因子会破坏图的组合性质和计算性质。我们提出符号双团分解（Signed Biclique Decomposition, SBD），这是一种针对整数加权图的稀疏低秩分解方法，它将结构化成分表示为稀疏符号二值秩一因子之和，再加上一个稀疏符号残差。所得的表示保持离散性、可解释性和稀疏性，并与标准线性代数兼容。我们形式化了固定秩与贪心SBD目标，并推导出一种具有理论保证的基于增益的贪心算法。实验表明，SBD能够在噪声图中恢复块结构，同时其因子化表示支持高效的稀疏矩阵乘法运算。

    arXiv:2611.25826v1 Announce Type: new  Abstract: Low-rank plus sparse decompositions, such as robust PCA, typically model the low-rank component as dense and the sparse component as corruption. This viewpoint is poorly suited to graph adjacency matrices, where sparsity is structural and dense low-rank factors destroy the combinatorial and computational properties of the graph. We propose signed biclique decomposition (SBD), a sparse low-rank decomposition for integer-weighted graphs that represents the structured component as a sum of sparse signed binary rank-one factors, together with a sparse signed residual. The resulting representation remains discrete, interpretable, sparse, and compatible with standard linear algebra. We formulate fixed-rank and greedy SBD objectives and derive a gain-based greedy algorithm with theoretical guarantees. Empirically, SBD recovers block structure in noisy graphs, while its factorized representation enables efficient sparse matrix-matrix multiplicat
    
[^98]: 分割模型中构建搜索子空间的局部曲率极限

    Limits of local curvature for constructing search subspaces in segmentation models

    [https://arxiv.org/abs/2611.25822](https://arxiv.org/abs/2611.25822)

    该研究发现基于局部曲率构造的搜索子空间表征梯度更新场变化的能力随搜索半径增大而衰减，但在初始梯度中加入广义高斯-牛顿方向可显著降低语义分割模型中的表征相对误差。

    

    我们检验了由局部曲率构造的子空间在语义分割模型中对有限步长后梯度更新场变化的表征能力。局部响应是曲率算子对实际输入位移的作用。在针对50张图像和两个模型的研究中，我们测量了在固定子空间中获得的损失增量与在全空间中获得的损失增量之比。该比值随搜索半径的增大而减小。在来自同一样本的十张图像上，我们针对两个模型比较了由初始梯度张成的一维子空间与两个秩为16的子空间。这两个等秩子空间在相同的初始状态下、使用相同的实际步长和测得的向量进行评估。在初始梯度基础上加入广义高斯-牛顿方向，可将表征场变化的平均相对误差从第一个模型的0.922降至0.874，从第二个模型的0.819降至0.763。

    arXiv:2611.25822v1 Announce Type: cross  Abstract: We examine how well subspaces constructed from local curvature represent the change in the gradient update field after a finite step in semantic segmentation models. The local response is the action of the curvature operator on the actual input displacement. In a study of 50 images and two models, we measure the ratio of the loss increase attained in a fixed subspace to that attained in the full space. This ratio decreases as the search radius increases. On ten images from the same sample, we compare the one-dimensional subspace spanned by the initial gradient with two rank-16 subspaces for both models. The two equal-rank subspaces are assessed at the same initial states, using the same actual steps and measured vectors. Adding generalised Gauss-Newton directions to the initial gradient reduces the mean relative error in representing the field change from 0.922 to 0.874 for the first model and from 0.819 to 0.763 for the second. At the
    
[^99]: 锚定与核验：用于可审计LLM决策的确定性跨提示、跨模型调和机制

    Anchor and Check: Deterministic Cross-Prompt, Cross-Model Reconciliation for Auditable LLM Decisions

    [https://arxiv.org/abs/2611.25818](https://arxiv.org/abs/2611.25818)

    该论文提出“锚定与核验”确定性调和契约——让目标模型的贪心答案接受跨提示或跨模型家族的独立贪心核验、仅一致时才执行，以实现可审计的LLM自主决策，并发现汇集跨家族样本可使语义熵的AUROC提升1.7至4.0个百分点，而同模型温度核验是最弱设计。

    

    让大型语言模型（LLM）做出离散决策的流水线需要知道哪些决策可以在无人值守的情况下被执行，并且需要一条能解释每个决策的记录。我们研究了一个小型且完全明确指定的调和契约：提交目标模型的贪心答案（即锚点），抽取一至两次通过不同路径得到该答案的贪心核验（更换提示词，或使用来自另一模型家族的模型），然后由一个纯函数返回五种结果之一，其中只有“一致”结果会被采纳执行。在覆盖6000个问题的48,000次模型-条目评估中，每种方法均可使用言语化置信度作为平局决胜依据，核心发现关乎调用的来源：汇集另一个模型家族的样本可使语义熵的AUROC在每个预算水平上提升1.7至4.0个百分点，而同模型的温度核验是最弱的设计。作为完整流水线，三调用契约与基于3或5个样本的汇集语义熵的点估计……

    arXiv:2611.25818v1 Announce Type: new  Abstract: Pipelines that let a large language model (LLM) make discrete decisions need to know which decisions they may act on unattended, and a record that explains each one. We study a small, fully specified reconciliation contract: commit the target model's greedy answer (the anchor), draw one or two greedy checks that reach the answer by a different route (an alternate prompt, or a model from another family), and let a pure function return one of five outcomes, of which only Agree is acted on. On 48,000 model-item evaluations over 6,000 questions, with every method given access to verbalized confidence as a tie-breaker, the main finding concerns where calls come from: pooling another family's samples raises semantic entropy's AUROC by 1.7 to 4.0 points at every budget, and a same-model temperature check is the weakest design. As complete pipelines, the point estimates of the three-call contract and pooled semantic entropy over 3 or 5 samples d
    
[^100]: 检测内部威胁的语言模型解释有多可靠？

    How Reliable Are Explanations of Language Models That Detect Insider Threats?

    [https://arxiv.org/abs/2611.25796](https://arxiv.org/abs/2611.25796)

    本文提出“倾覆半径”这一核心贡献，通过一阶预测度量特征移除的相似对照变化何时会颠覆检测器的结论，从而审计内部威胁检测语言模型解释的可靠性。

    

    组织机构会监测员工的活动以防范内部威胁，即员工滥用其合法访问权限的行为，例如在离职前复制机密文件。检测器会对员工每天活动的异常程度进行评分，并将最异常的情况标记出来供分析师审查。当前主流方法使用不透明的神经网络，因此对某个标记的解释指向模型的内部特征：即在滥用行为发生的日子里最活跃的神经元活动模式。我们对在CERT模拟工作场所日志上训练的语言模型检测器的此类解释进行了审计。为了检验某个解释，我们观察当在模型内部移除其特征时检测器分数会发生什么变化，并将其与一个对照组进行比较：该变化尽可能类似于移除操作，但实际上并未移除这些特征。由于定义“相似”需要选择一种度量方式，我们的核心贡献是“倾覆半径”：即对该度量必须移动多远之后结论才会发生改变的一阶预测。

    arXiv:2611.25796v1 Announce Type: new  Abstract: Organizations monitor their workers' activity for insider threats, in which workers misuse their legitimate access, by copying confidential files before leaving. Detectors score how unusual each day of activity is, flagging the most unusual for an analyst to review. Prominent current approaches use opaque neural networks, so explanations of a flag point to internal features: patterns of neuron activity most active on days of misuse. We audit such explanations for language-model detectors trained on CERT's simulated workplace logs. To test an explanation, we see what happens to the detector's score when we remove its features inside the model. We compare this to a control: a change that is as similar as possible to the removal, but without actually removing the features. Because defining "similar" requires choosing a metric, our central contribution is the tipping radius: a first-order prediction of how far that metric must move before th
    
[^101]: RouteVerify：面向混合专家模型逐步验证的路由干预研究

    RouteVerify: Routing Interventions for Step-Level Verification in Mixture-of-Experts Models

    [https://arxiv.org/abs/2611.25780](https://arxiv.org/abs/2611.25780)

    该论文提出 RouteVerify 受控干预框架，通过在混合专家模型的九个检查点上替换路由器 logits、MoE 输出和残差状态等内部激活，系统检验路由状态与逐步验证判断之间的因果关系。

    

    混合专家模型中的路由状态能够区分正确与错误的推理步骤。如果改变路由也会改变模型的逐步验证判断，人们很容易得出结论：该判断遵循路由所编码的信息。我们通过 RouteVerify 对这一推断进行检验，这是一项受控干预研究，它将一个求解轨迹中被选定的内部激活替换为另一个轨迹中的激活，同时保持被评估的输入不变。在九个 MoE 检查点上，我们比较了三种干预接口：路由器 logits、MoE 输出以及 MoE 之后的残差状态。我们改变了激活被替换的层和词元位置，并测量验证边际（verification margin）的变化，该边际量化了模型对 INCORRECT 相对于 CORRECT 的偏好。我们还使用来自正确步骤与错误步骤的激活来评估相同的输入，以检验由此产生的判断是否取决于源步骤的正确性。

    arXiv:2611.25780v1 Announce Type: cross  Abstract: Routing states in mixture-of-experts (MoE) models can distinguish correct from erroneous reasoning steps. If changing routing also changes a model's step-level verification judgment, it is tempting to conclude that the judgment follows what routing encodes. We test this inference with RouteVerify, a controlled intervention study that replaces selected internal activations from one solution trace with activations from another, while keeping the evaluated input unchanged. Across nine MoE checkpoints, we compare three intervention interfaces: router logits, MoE outputs, and post-MoE residual states. We vary the layers and token positions at which activations are replaced. We measure changes in a verification margin that quantifies the model's preference for INCORRECT over CORRECT. We also evaluate the same input with activations from correct versus erroneous steps to test whether the resulting judgment depends on the source step's correct
    
[^102]: 基于视线估计与结构化驾驶场景理解的驾驶员注视目标识别

    Driver Gaze Target Identification via Gaze Estimation and Structured Driving Scene Understanding

    [https://arxiv.org/abs/2611.25771](https://arxiv.org/abs/2611.25771)

    该论文提出了一种模块化框架，通过融合MAE预训练视觉Transformer视线估计、全景场景理解、个性化校准和熵驱动的目标推断，从车内驾驶员图像与前视道路图像中实现目标级的驾驶员注视目标识别。

    

    识别驾驶员的注视目标可以为智能车辆中的视觉注意力监测提供目标级信息。视线区域分类和连续注意力定位所能提供的关于被注视场景目标身份的信息较为有限。本文提出了一种模块化框架，从同步采集的车舱内驾驶员图像和前视道路图像中识别目标级或区域级的注视目标。该框架包含四个组成部分：MFGaze，它将使用掩码自编码器（MAE）预训练的视觉Transformer与通过特征级线性调制（FiLM）实现的头部姿态条件化相结合；全景场景理解，用于构建结构化的候选目标集合；多项式岭回归，通过个性化校准将视线特征映射到二维场景图像坐标；以及熵驱动的目标推断，自适应地融合边界距离几何信息与类别级语义先验。

    arXiv:2611.25771v1 Announce Type: cross  Abstract: Identifying a driver's gaze target provides object-level information for visual attention monitoring in intelligent vehicles. Gaze-zone classification and continuous attention localization provide limited information about the identity of the attended scene target. This paper proposes a modular framework that identifies object- or region-level gaze targets from synchronized in-cabin driver images and forward-view road images. The framework comprises four components: MFGaze, which combines a Vision Transformer pretrained with a masked autoencoder (MAE) and head-pose conditioning through feature-wise linear modulation (FiLM); panoptic scene understanding, which constructs a structured candidate set; polynomial ridge regression, which maps gaze features to 2D scene-image coordinates through individualized calibration; and entropy-driven target inference, which adaptively fuses boundary-distance geometry with class-level semantic priors. E
    
[^103]: 解耦监护仪硬件伪影与平滑滞后以实现分布偏移鲁棒的术中脑电与血流动力学风险校准

    Disentangling Monitor Hardware Artifacts and Smoothing Lag for Shift-Robust Intraoperative EEG and Hemodynamic Risk Calibration

    [https://arxiv.org/abs/2611.25765](https://arxiv.org/abs/2611.25765)

    针对商用术中监护仪的硬件伪影（阻抗脉冲、平滑延迟、直流漂移、袖带保持）干扰风险预测的问题，本文提出因果信号调理与分布偏移鲁棒的保形校准框架，在5,439例麻醉手术中实现了更可靠的术中脑电与血流动力学风险校准。

    

    商用术中脑电图（EEG）和血压监护仪会引入周期性阻抗脉冲、平滑延迟、直流（DC）漂移和袖带保持等伪影，这些伪影会干扰风险预测与不确定性校准。我们基于5,439例公开麻醉手术记录，开发了一套因果信号调理与分布偏移鲁棒的保形校准框架。流式滤波器屏蔽10.13分钟时长的64 Hz阻抗检查，应用无削波的后陷波调理，分离63秒的抑制比滞后，并区分三通道袖带充气与信号脱落，覆盖丙泊酚-瑞芬太尼全静脉麻醉（TIVA；N=3,214）、挥发性麻醉（N=2,046）、欧洲内镜镇静（N=171）以及直流耦合手术（N=8）。研究将去除阻抗后的皮层树集成模型与深度序列模型进行比较，并与基于患者聚类的子群-协变量加权保形风险控制（CRC）以及滞后缓冲的在线保形预测方法相结合（摘要在此处被截断）。

    arXiv:2611.25765v1 Announce Type: new  Abstract: Commercial intraoperative electroencephalography (EEG) and blood pressure monitors introduce cyclic impedance pulses, smoothing delays, direct-current (DC) drift, and cuff holds that confound risk forecasting and uncertainty calibration. We developed a causal signal-conditioning and shift-robust conformal calibration framework across 5,439 public anesthesia procedures. Streaming filters blank 10.13-min 64-Hz impedance checks, apply unclipped post-notch conditioning, separate 63-s suppression-ratio lag, and distinguish three-channel cuff inflations from dropouts across propofol-remifentanil total intravenous anesthesia (TIVA; N = 3,214), volatile anesthesia (N = 2,046), European endoscopy sedation (N = 171), and DC-coupled surgery (N = 8). De-impedanced cortical tree ensembles were compared against deep sequence models, paired with patient-cluster Subgroup-Covariate-Weighted Conformal Risk Control (CRC) and lag-buffered online Conformal P
    
[^104]: 在低推理预算下缩小扩散语言模型的性能差距

    Closing Performance Gap for Diffusion Language Models under Low Inference Budgets

    [https://arxiv.org/abs/2611.25763](https://arxiv.org/abs/2611.25763)

    提出了无需训练的 CaRR 框架，通过 token 承诺重掩码、更新上下文回填和基于轨迹的恢复机制，在低推理预算下提升可编辑扩散语言模型的生成质量，从而缩小其性能差距。

    

    扩散语言模型通过并行更新多个 token，为低预算生成提供了一条架构层面的途径，与逐 token 的自回归解码相比，可以减少顺序解码步骤的数量。然而，在低预算下实现这一优势颇具挑战，因为当去噪和细化步骤的数量减少时，生成质量往往会下降。可编辑扩散模型通过草稿后细化来应对这种质量损失，但在最终提交之前，有用的中间结果可能仍然会丢失。我们提出了 CaRR（Commitment-aware Refinement and Recovery，承诺感知的细化与恢复），这是一个无需训练的框架，它在现有的前向调度内，通过更新上下文回填和基于轨迹的恢复，从固定的可编辑检查点提升提交质量。其中 Token Commitment Remasking（TCR，token 承诺重掩码）会对选定的原生纯替换进行重掩码，并在下一个预定的编辑前向中使用更新后的内容进行回填。

    arXiv:2611.25763v1 Announce Type: cross  Abstract: Diffusion language models offer an architectural route to low-budget generation by updating multiple tokens in parallel, which can reduce the number of sequential decoding steps relative to token-by-token autoregressive decoding. However, realizing this advantage at low budgets is challenging, as generation quality often declines when the number of denoising and refinement steps is reduced. Editable diffusion models address this quality loss through post-draft refinement, but useful intermediate results may still be lost before final submission. We introduce CaRR (Commitment-aware Refinement and Recovery), a training-free framework which improves submission quality from a fixed editable checkpoint through updated-context refill and trajectory-based recovery within its existing forward schedule. Token Commitment Remasking (TCR) remasks selected native pure replacements and refills them on the next scheduled edit forward using the update
    
[^105]: AdaptReg：用于冻结配准基础模型任务特定适配的跳跃层FiLM适配器

    AdaptReg: Skip-level FiLM Adapter for Task-Specific Adaptation of a Frozen Registration Foundation Model

    [https://arxiv.org/abs/2611.25761](https://arxiv.org/abs/2611.25761)

    AdaptReg通过在冻结的配准基础模型（uniGradICON）解码器跳跃连接上引入由TotalSegmentator解剖语义特征驱动的FiLM式轻量适配器，并以病灶体积保持损失进行监督，实现了面向前列腺癌纵向PSMA-PET/CT配准的任务特定适配，在精准对齐的同时避免MTV/TLG测量被配准伪影污染。

    

    纵向PSMA-PET/CT配准是评估前列腺癌治疗反应的关键步骤，它所需要的不只是通用对齐精度：形变必须保持病灶体积，从而使代谢肿瘤体积（MTV）和病灶糖酵解总量（TLG）的测量变化能够反映真实的治疗反应而非配准伪影。我们提出AdaptReg，通过一个具有恒等保持初始化的轻量可训练适配器分支，将冻结的配准基础模型适配到此类任务特定需求。在保持uniGradICON完全冻结的前提下，我们对其解码器的跳跃连接施加FiLM风格的逐体素仿射调制（γ, β）；该调制由冻结的TotalSegmentator编码器提取的解剖语义特征生成，并受病灶体积保持损失监督。在官方Codabench验证集（20对数据）上，AdaptReg取得了DSC 0.7585、HD95 6.33 mm以及较低的MTV/TLG误差。

    arXiv:2611.25761v1 Announce Type: cross  Abstract: Longitudinal PSMA-PET/CT registration is a key step in assessing treatment response in prostate cancer, and it requires more than generic alignment accuracy: the deformation must preserve lesion volume, so that measured changes in metabolic tumor volume (MTV) and total lesion glycolysis (TLG) reflect treatment response rather than registration artifacts. We propose AdaptReg, which adapts a frozen registration foundation model to such task-specific requirements through a lightweight trainable adapter branch with identity-preserving initialization. Keeping uniGradICON fully frozen, we apply FiLM-style per-voxel affine modulation ($\gamma$, $\beta$) to its decoder skip connections; the modulation is generated from anatomical semantic features of a frozen TotalSegmentator encoder and supervised by a lesion volume-preservation loss. On the official Codabench validation set (20 pairs), AdaptReg achieves DSC 0.7585, HD95 6.33 mm, MTV/TLG erro
    
[^106]: 超越误差缓解：将提升方法应用于含噪声量子机器学习

    Beyond Error Mitigation: Using Boosting for Noisy Quantum Machine Learning

    [https://arxiv.org/abs/2611.25752](https://arxiv.org/abs/2611.25752)

    该论文提出利用监督学习中带标签样本提供的问题特定信息，将Boosting提升方法应用于含噪声参数化量子电路，使二分类任务只需依赖分类器输出的符号和间隔而非精确重建无噪声期望值，从而超越传统量子误差缓解方法的巨大采样开销限制。

    

    量子误差缓解（QEM）可以从含噪声的电路执行中恢复关于理想量子计算的信息，但可能需要较大的采样开销。基本下界表明，在没有问题特定信息的情况下，恢复无噪声观测量可能需要超多项式数量的样本，对于广泛的含噪声电路类别甚至需要指数级的开销。然而，问题特定的信息可以减少获得所需答案所需的样本数量。我们在监督量子机器学习中研究了这种可能性，其中带标签的样本提供了关于待学习函数的信息。二分类任务提供了这样一种场景，因为正确的预测取决于分类器输出的符号和间隔，而不是对每个无噪声期望值的精确重建。我们开发了一个理论和实证框架，用于比较含噪声参数化量子电路的提升方法。

    arXiv:2611.25752v1 Announce Type: cross  Abstract: Quantum error mitigation (QEM) can recover information about ideal quantum computations from noisy circuit executions, but may require a large sampling overhead. Fundamental lower bounds show that recovering noiseless observables without problem-specific information can require superpolynomially many samples, with exponential overhead for broad classes of noisy circuits. However, problem-specific information can reduce the number of samples needed to obtain the required answer. We investigate this possibility in supervised quantum machine learning, where labeled examples provide information about the function to be learned. Binary classification provides such a setting, since correct prediction depends on the sign and margin of the classifier output rather than on accurate reconstruction of every noiseless expectation value. We develop a theoretical and empirical framework for comparing boosting of noisy parameterized quantum circuits 
    
[^107]: 通过微调视觉-语言模型实现目标引导的强化学习

    Goal-Guided Reinforcement Learning via Fine-Tuning Vision-Language Models

    [https://arxiv.org/abs/2611.25742](https://arxiv.org/abs/2611.25742)

    该论文构建了CCGP-2M多模态目标偏好数据集，并通过KLTDPO微调视觉-语言模型，使其在在线目标引导强化学习框架中为稀疏奖励环境生成可靠且符合环境特点的高层探索目标。

    

    在类Craftax环境中，由于奖励稀疏和时间跨度长，探索仍然是强化学习（RL）智能体面临的主要挑战。大语言模型（LLM）可以通过生成高层目标来引导探索，但使用未经显式任务适配的模型的方法通常依赖精心设计的提示词，并且可能产生不一致或与上下文不匹配的目标。我们提出了CCGP-2M，这是一个基于Craftax-Classic交互轨迹构建的多模态目标偏好数据集，并提出了一种由视觉-语言模型（VLM）引导的强化学习框架，该框架从该数据集中学习针对特定环境的目标引导。该框架包含两个阶段：离线VLM微调和在线目标引导强化学习。在离线阶段，我们使用Kullback-Leibler Token级直接偏好优化（KLTDPO）将VLM与目标偏好对齐，该方法将token级KL正则化与分布式鲁棒重加权相结合（摘要在此处被截断）。

    arXiv:2611.25742v1 Announce Type: new  Abstract: Exploration remains a major challenge for reinforcement learning (RL) agents in Craftax-style environments due to sparse rewards and long horizons. Large language models (LLMs) can guide exploration by generating high-level goals, but methods using models without explicit task adaptation often rely on carefully designed prompts and may produce inconsistent or contextually inappropriate goals. We introduce CCGP-2M, a multimodal goal-preference dataset constructed from interaction trajectories in Craftax-Classic, and propose a vision-language model (VLM)-guided RL framework that learns environment-specific goal guidance from this dataset. The framework consists of two stages: offline VLM fine-tuning and online goal-guided RL. In the offline stage, we align the VLM with goal preferences using Kullback-Leibler Token-level Direct Preference Optimization (KLTDPO), which combines token-level KL regularization with distributionally robust reweig
    
[^108]: PosteriorBench：评估语言模型中行为后验采样能力的基准

    PosteriorBench: Evaluating Behavior Posterior Sampling in Language Models

    [https://arxiv.org/abs/2611.25735](https://arxiv.org/abs/2611.25735)

    该论文提出了 PosteriorBench 基准，用于从计算成本和与真实后验的 KL 散度两个维度评估语言模型的行为后验采样算法，并发现现有方法均无法同时兼顾高效性与保真度。

    

    语言模型中的罕见行为可能难以研究，因为通过普通采样来获得这些行为往往需要极其大量的样本。由此引出了行为后验采样问题：给定一个语言模型和一个感兴趣的行为，是否存在一种算法，能够从该模型在展现该行为的对话记录上的分布中高效地进行采样？我们提出了 PosteriorBench，这是一个用于比较算法的基准，比较维度包括计算成本，以及其采样结果与真实后验在正向和反向 KL 散度上的接近程度。PosteriorBench 包含基于 Qwen3.5-0.8B 的 21 个任务，行为概率范围从 10^-5 到 10^-1，每个任务都配有来自一个包含十亿条标注响应的语料库的精确后验样本。我们发现，我们评估的所有方法中没有一种能同时做到高效且忠实：诸如提示（prompting）这样的简单方法成本低廉，但其与后验之间的 KL 散度会停滞在较高水平；对后验样本进行微调的方法……（摘要在此处截断）

    arXiv:2611.25735v1 Announce Type: cross  Abstract: Rare language model behaviors can be difficult to study because obtaining them through ordinary sampling can require a very large number of samples. This motivates the behavior posterior sampling problem: given a language model and a behavior of interest, is there an algorithm that efficiently samples from the model's distribution over transcripts that exhibit this behavior? We introduce PosteriorBench, a benchmark for comparing algorithms by their compute cost and by how close their samples are to the true posterior in forward and reverse KL divergence. PosteriorBench contains 21 tasks on Qwen3.5-0.8B with behavior probabilities from $10^{-5}$ to $10^{-1}$, each with exact posterior samples from a corpus of one billion labeled responses. We find that no method we evaluate is both efficient and faithful: simple methods like prompting are cheap but plateau at a high KL divergence from the posterior, fine-tuning on posterior samples is o
    
[^109]: 大语言模型能否解决领域特定自然语言理解流水线中的实体冲突？

    Can LLMs Resolve Entity Conflicts in Domain Specific NLU Pipelines?

    [https://arxiv.org/abs/2611.25733](https://arxiv.org/abs/2611.25733)

    本研究通过比较基于规则的冲突解决方法与两种基于大语言模型的方法（多轮提示和LoRA微调的Llama3.1-8B-Instruct），验证了大语言模型在解决领域特定NLU流水线中实体冲突问题上的可行性。

    

    解决实体冲突是基于流水线的自然语言理解（NLU）架构中的一个关键步骤，尤其是当多个提取器提供重叠或冲突的输出时。传统的基于规则的实体冲突解决方法往往缺乏灵活性且难以扩展。在这项工作中，我们旨在研究是否可以利用大语言模型的广泛知识来解决领域特定NLU流水线中的实体冲突。我们对传统的基于规则的冲突解决方法与两种基于大语言模型的方法进行了实验比较：一种是使用Llama3.1-8B-Instruct模型的多轮提示方法，另一种是经过LoRA微调的Llama3.1-8B-Instruct模型。评估在两种设置下进行：（1）对大语言模型匹配现有NLU流水线输出的能力进行广泛评估；（2）在更受控的场景中评估大语言模型在人工标注数据上的准确率。结果表明，虽然多轮提示具有易于适配和行为快速调整的优势……

    arXiv:2611.25733v1 Announce Type: cross  Abstract: Resolving entity conflicts is a critical step in Pipeline-based NLU architectures, especially when multiple extractors provide overlapping or conflicting outputs. Traditional rule-based entity conflict resolution methods often lack flexibility and are unscalable. In this work, we aim to study if the broad knowledge of LLMs can be leveraged to resolve entity conflicts in domain specific NLU pipelines. We present an experimental comparison between a traditional rule-based conflict resolution method, and two LLM-based approaches: a multi-shot prompting method using the Llama3.1-8B-Instruct model and a LoRA fine-tuned Llama3.1-8B-Instruct model. The evaluation was conducted under two setups: (1) a broad assessment of LLM's capability to match existing NLU pipeline output, and (2) a more controlled scenario assessing the LLM accuracy on human-labeled data. Results show that while multi-shot prompting offers ease of adaptation and rapid beha
    
[^110]: CGM：用于自闭症刻板行为检测的混沌引导图注意力与专家路由模型

    CGM: Chaos-Guided Graph Attention and Expert Routing for Autism Stereotypy Detection

    [https://arxiv.org/abs/2611.25730](https://arxiv.org/abs/2611.25730)

    该论文提出CGM模型，将逐关节、逐帧的运动不规律性（样本熵）张量直接融入图注意力计算和专家路由机制，使模型聚焦于运动不可预测的关节，从而更准确地区分自闭症刻板行为与正常运动。

    

    用于自闭症视频分析的骨架图注意力方法仅根据内容对关节进行加权，这使得不规律摆动的手腕与静止的手腕获得同等的地位。然而，刻板行为的定义在于运动展开方式的不确定性，而非运动量的大小：在我们的研究队列中，所有基于幅度的运动学描述子在刻板动作片段中均下降，唯独角速度的窗口化样本熵上升（Hedges' g = +1.05）。我们将这种不规律性计算为逐关节、逐帧的张量，并使其从结构上驱动整个计算过程。所提出的混沌引导图注意力与专家混合模型（CGM）结合了两种机制：第一，在图注意力logit中加入一个可学习的成对项，将注意力集中于不规律的关节，并将剩余的边权重重新排序至共同不规律的关节对；其无约束系数既不会消失也不会反转，将该张量置零会使94%的片段中的注意力重新对齐。第二，一个以……为条件的门控机制（原文摘要在此处截断）。

    arXiv:2611.25730v1 Announce Type: cross  Abstract: Skeleton graph attention for autism video analysis weights joints by content alone, giving an erratic wrist the same standing as a still one. Stereotypy, however, is defined by how unpredictably movement unfolds rather than how much of it occurs: in our cohort, every amplitude-based kinematic descriptor falls in stereotyped clips, while windowed Sample Entropy of angular velocity alone rises (Hedges' g = +1.05). We compute this irregularity as a per-joint, per-frame tensor that structurally drives computation. The proposed chaos-guided graph attention and mixture-of-experts model (CGM) couples two mechanisms. First, a learned pairwise term added to the graph-attention logit concentrates attention on irregular joints and re-ranks the remaining edge weight toward jointly irregular pairs; its unconstrained coefficient neither vanishes nor inverts, and zeroing the tensor realigns attention in 94% of clips. Second, a gate conditioned on the
    
[^111]: 带安全保障的学习辅助非线性共轭梯度方法

    Safeguarded Learning-Assisted Nonlinear Conjugate Gradient Methods

    [https://arxiv.org/abs/2611.25707](https://arxiv.org/abs/2611.25707)

    该论文提出了一种带安全保障的学习辅助非线性共轭梯度方法框架，通过两个在线极限学习机模型在重启恢复、停滞检测和线搜索失败时辅助经典优化过程，且神经预测仅在满足下降性、尺度等确定性数值安全条件时才被采纳。

    

    为非线性共轭梯度（NCG）方法开发了一种学习辅助框架，其中轻量级在线预测用于支持而非替代经典优化过程。所提出的方法保留了标准的两项非线性共轭梯度结构，并引入了两个独立的在线极限学习机模型。第一个模型在重启恢复期间或检测到持续停滞时预测非线性共轭梯度系数，而第二个模型仅在有限精度算术下发生线搜索失败后使用，用于对确定性回退步长提出有界修正。神经预测仅在其满足确定性数值安全保障（包括下降性、尺度和分布外要求）时才被接受；否则，将保留相应的确定性恢复或回退机制。两个学习模型均利用优化过程中已经生成的量进行在线更新。

    arXiv:2611.25707v1 Announce Type: cross  Abstract: A learning-assisted framework for Nonlinear Conjugate Gradient (NCG) methods is developed in which lightweight online prediction is used to support, rather than replace, the classical optimization process. The proposed approach retains the standard two-term NCG structure and introduces two independent online extreme learning machine models. The first model predicts an NCG coefficient during restart recovery or when persistent stagnation is detected, whereas the second model is used only after a line-search failure in finite precision arithmetic to propose a bounded correction to the deterministic fallback step size. Neural predictions are accepted only when they satisfy deterministic numerical safeguards, including descent, scale, and out-of-distribution requirements; otherwise, the corresponding deterministic recovery or fallback mechanism is retained. Both learning models are updated online using quantities already generated along th
    
[^112]: 从地理记忆到大语言模型中的空间泛化

    From Geographic Memory to Spatial Generalization in Large Language Models

    [https://arxiv.org/abs/2611.25703](https://arxiv.org/abs/2611.25703)

    该论文提出将大语言模型中的地理知识组织为潜在空间场，通过连续空间编码器将位置表示作为空间标记注入模型，从而实现从仅能回忆训练中见过的“地理空间记忆”向能预测未见区域的“空间泛化”能力的跨越。

    

    大语言模型（LLM）能够将地点与地理描述相关联，但恢复训练中观察到的关联并不能证明其具备跨地理空间泛化的能力。我们区分了“地理空间记忆”（即恢复训练期间观察到的位置—描述关联）与“空间泛化”（即为未观察区域中的位置预测描述）。我们研究连续的位置表示能否改善这种泛化能力，并促使地理知识在大语言模型内部形成空间化的组织。我们将这种组织形式构建为一个潜在空间场：一种通过空间依赖关系连接的、从位置到描述的学习映射。基于连续空间编码器，我们将位置表示投影到模型的嵌入空间中，并将其作为空间标记提供。我们在两个新构建的地理文本基准（分别针对洛杉矶和伦敦）上评估了所得模型。

    arXiv:2611.25703v1 Announce Type: cross  Abstract: Large language models (LLMs) can associate places with geographic descriptions, but recovering an observed association does not establish an ability to generalize across geographic space. We distinguish geospatial memory, the recovery of location--description associations observed during training, from spatial generalization, the prediction of descriptions for locations in unobserved regions. We investigate whether continuous location representations improve this generalization and encourage geographic knowledge to become spatially organized within an LLM. We frame this organization as a latent spatial field: a learned mapping from locations to descriptions connected through spatial dependencies. Building on a continuous spatial encoder, we project location representations into the model's embedding space and supply them as spatial tokens. We evaluate the resulting models on two newly curated geo-text benchmarks for Los Angeles and Lon
    
[^113]: 可测量并不意味着可迁移：多年田间试验中模型解释的前瞻性审计

    Measurable Does Not Imply Transportable: A Prospective Audit of Model Explanations in Multi-Year Field Trials

    [https://arxiv.org/abs/2611.25692](https://arxiv.org/abs/2611.25692)

    该论文形式化了“前瞻性解释可迁移性”这一新概念，并通过多年玉米田间试验证明：可靠的测量并不保证解释能在真正未来的情境中复现，这一属性独立于特征重要性和测量可靠性。

    

    模型解释日益被用作科学证据，然而它们能让科学家在构建解释所用的数据之外得出何种结论，仍不明确。当细粒度的解释无法复现时，这种失败可能被归因于对其目标的噪声测量，由此引出一个根本性问题：可靠的测量是否意味着可复现的解释？我们对此进行了直接检验。我们形式化了“前瞻性解释可迁移性”这一概念：即从过去数据中学到的解释在真正未来情境中被恢复的程度。这一属性有别于特征重要性和测量可靠性。利用多年份的“基因组到田地”玉米田间试验——其中每个新年份都是真实的未来情境，而非随机重采样——我们在五个前向划分和一个前瞻性锁定的测试年份上评估了16个具有农学可解释性的环境概念。粗粒度的环境层面解释能够迁移（摘要在此处截断）

    arXiv:2611.25692v1 Announce Type: new  Abstract: Model explanations are increasingly used as scientific evidence, yet it remains unclear what they license scientists to conclude beyond the data on which they were constructed. When a fine-grained explanation fails to replicate, the failure may be attributed to noisy measurement of its target, raising a fundamental question: does reliable measurement imply a reproducible explanation? We test this directly. We formalize prospective explanation transportability: the extent to which an explanation learned from past data is recovered in a genuinely future context. This property is distinct from feature importance and measurement reliability. Using multi-year Genomes-to-Fields maize trials, where each new year is a real future context rather than a random resample, we evaluate 16 agronomically interpretable environmental concepts across five forward splits and one prospectively locked test year. Coarse environment-level explanations transport
    
[^114]: 基于观察队列路径的基因型嵌入模型可解释性研究

    Explainability of genotype embedding models through observed cohort paths

    [https://arxiv.org/abs/2611.25681](https://arxiv.org/abs/2611.25681)

    该研究揭示了积分梯度等路径归因方法无法直接用于基因型嵌入模型的原因——离散等位基因剂量的学习嵌入在分数基因型处无定义、无梯度——并提出在嵌入空间中沿观察到的队列路径进行归因，同时证明常用的逐变异估计量不满足完备性。

    

    将基因型与神经影像相结合的深度学习模型在预测神经系统疾病方面比单独使用任何一种模态都更为准确。然而，要将其用于临床，需要确定预测所依据的是哪些基因变异。诸如积分梯度等基于路径的归因方法通过从基线到输入对模型梯度进行积分来解决这一问题。这些方法假设模型能够接受该路径上的任意点作为有效输入，而基因型模型违反了这一假设。基因型模型通过仅在三个离散等位基因剂量{0,1,2}处定义的学习嵌入将这些剂量映射为向量，因此直线路径会经过分数基因型，而在这些点上既不存在嵌入，也不存在梯度。因此，归因必须在嵌入空间中进行，而路径的选择可能会影响最终的归因结果。我们证明，常用的逐变异估计量通常不满足完备性。

    arXiv:2611.25681v1 Announce Type: new  Abstract: Deep learning models that combine genotypes with neuroimaging predict neurological diseases more accurately than either modality alone. Their clinical use, however, requires identifying which genetic variants a prediction is based on. Path-based attribution methods like Integrated Gradients address this by integrating the model's gradient from a baseline to the input. These methods assume the model accepts any point along that path as a valid input. Genotype models violate this assumption. They map the three discrete allele dosages \{0,1,2\} to vectors through a learned embedding defined only at those values, so a straight line passes through fractional genotypes at which no embedding, and therefore no gradient, exists. Attribution must therefore proceed in embedding space where the choice of path can affect the resulting attribution. We show that the commonly used per-variant estimator does not generally satisfy completeness. Additive s
    
[^115]: 基于Karhunen-Loève特征蒸馏与时序卷积网络的轻量级低频非侵入式负荷监测

    Lightweight Low-Frequency Non-Intrusive Load Monitoring via Karhunen-Lo\`eve Feature Distillation and Temporal Convolutional Networks

    [https://arxiv.org/abs/2611.25678](https://arxiv.org/abs/2611.25678)

    提出了一种面向边缘设备的轻量级NILM框架，通过活动边界感知的数据增强、KLT特征提取和紧凑型TCN，在低频采样下以低计算成本实现准确的负荷分解。

    

    非侵入式负荷监测（NILM）是实现需求侧管理、支持智能电网优化的关键使能技术。一个可实际部署的NILM解决方案需要具备三个关键特性：准确的负荷分解、低频采样和低计算成本。本文提出了一种轻量级、面向边缘设备的NILM框架，该框架将活动边界感知的合成数据增强与Karhunen-Loève变换（KLT）特征提取以及时序卷积网络（TCN）相结合。数据增强阶段通过提取并重新组合完整或接近完整的电器运行片段，使模型无需新的完整标注的家庭总负荷数据即可完成训练。KLT滤波器组提取去相关的、电器感知的时序特征，随后将这些特征与原始总负荷数据一同输入紧凑的TCN进行能量分解。在两个标准NILM数据集UK-DALE和REDD上以10秒采样进行的评估表明……

    arXiv:2611.25678v1 Announce Type: cross  Abstract: Non-Intrusive Load Monitoring (NILM) is a critical enabler for demand-side management which allows for smart grid optimization. A deployment-ready NILM solution necessitates three key features: accurate disaggregation, low-frequency sampling, and low computational cost. This paper proposes a lightweight, edge-oriented NILM framework that combines activity-boundary-aware synthetic data augmentation with Karhunen-Lo\`eve Transform (KLT) feature extraction and a Temporal Convolutional Network (TCN). The augmentation stage extracts and recombines complete or near-complete appliance activation segments so that the model can be trained without new fully labeled household aggregates. The KLT filter bank extracts decorrelated, appliance-informed temporal features which are then fed into a compact TCN along with the original aggregate for energy disaggregation. Evaluations on two standard NILM datasets, UK-DALE and REDD, at 10-s sampling and wi
    
[^116]: GPU加速的药物-靶点相互作用预测安全推理

    GPU-Accelerated Secure Inference for Drug--Target Interaction Prediction

    [https://arxiv.org/abs/2611.25664](https://arxiv.org/abs/2611.25664)

    该论文针对iDASH 2026竞赛提出了GPU上的两方安全推理方案，通过协同设计MPC友好的DTI模型SlimDTA与MPC协议，实现快速的隐私保护药物-靶点相互作用预测，避免候选药物知识产权泄露。

    

    预测药物-靶点相互作用（DTI）的深度学习模型可以加速药物发现，但模型所评分的候选药物属于机密知识产权，其所有者无法公开泄露。安全多方计算（MPC）允许两个服务器对一个以秘密共享形式存在的药物进行预测计算，而任何一方服务器都无法获知该药物的真实内容。然而，深度模型的安全推理速度很慢：非线性操作需要服务器之间进行通信，且MPC算术无法自然地映射到GPU上。本文提出了针对iDASH 2026隐私与安全竞赛第三赛道的解决方案，该赛道要求在GPU上实现快速的两方安全DTI推理。我们的解决方案建立在MPC友好模型与MPC协议的协同设计之上。在模型方面，我们设计了SlimDTA，这是一种能够减少对秘密药物所执行操作的DTI模型，尤其是减少需要通信的非线性操作。在协议方面，我们构建了……

    arXiv:2611.25664v1 Announce Type: cross  Abstract: Deep learning models that predict drug--target interactions (DTI) can speed up drug discovery, but the candidate drugs they score are valuable intellectual property that their owners cannot reveal. Secure multiparty computation (MPC) lets two servers compute a prediction on a secret-shared drug, without either server learning the drug. However, secure inference of deep models is slow. Non-linear operations require communication between the servers, and MPC arithmetic does not map naturally to GPUs. In this paper, we present our solution to Track~III of the iDASH 2026 privacy and security competition, which asked for fast two-party secure DTI inference on GPUs. Our solution is built on the co-design of MPC-friendly models and MPC protocols. On the model side, we design SlimDTA, a DTI model that reduces the operations on the secret drug, and in particular the non-linear operations that require communication. On the protocol side, we buil
    
[^117]: SiliconGym：面向智能体化EDA的多功能强化学习环境

    SiliconGym: A Versatile Reinforcement Learning Environment for Agentic EDA

    [https://arxiv.org/abs/2611.25658](https://arxiv.org/abs/2611.25658)

    本文提出SiliconGym，首个面向硬件设计与验证的跨任务、跨工具链强化学习环境，通过解耦版本化工作区、预算化执行与验收门控评分，实现从模块级RTL任务到全层次SoC流程的可扩展训练，并借助流式调度将RL批次完成速度提升21.9%。

    

    利用电子设计自动化（EDA）反馈的强化学习（RL）正在推动硬件设计与验证领域大语言模型的发展。要将这些成果扩展到硬件工程智能体，需要能够跨越多种任务、在工具调用之间保持状态，并同时支持大型层次结构与并行回合的环境。我们提出了SiliconGym，这是首个面向硬件设计与验证的多功能、跨任务、跨工具链的强化学习环境。通过解耦版本化工作区、预算化执行和验收门控评分，SiliconGym能够从模块级的RTL、测试平台和PPA任务，扩展到多轮代码仓库修复以及全层次SoC流程。流式调度将工具检查与生成过程重叠，使一个包含256个答案的RL批次的完成速度比原生ChipSeek快21.9%。借助可互换的开源与商业后端，我们证明了在开源工具上训练获得的功能正确性可以顺利迁移到商业si（摘要在此处截断）

    arXiv:2611.25658v1 Announce Type: new  Abstract: Reinforcement learning (RL) with electronic design automation (EDA) feedback is advancing large language models for hardware design and verification. Extending these gains to hardware-engineering agents requires environments that span tasks, preserve state across tool calls, and support large hierarchies alongside parallel rollouts. We present SiliconGym, the first versatile, cross-task and cross-toolchain RL environment for hardware design and verification. By decoupling versioned workspaces, budgeted execution, and acceptance-gated scoring, SiliconGym scales from module-level RTL, testbench, and PPA tasks to multi-round repository repair and full-hierarchy SoC flows. Streaming dispatch overlaps tool checks with generation, finishing a 256-answer RL batch 21.9% faster than native ChipSeek. Using interchangeable open-source and commercial backends, we show that open-source-trained functional correctness transfers cleanly to commercial si
    
[^118]: 将基于Rademacher复杂度的泛化理论扩展到领域泛化

    Extending Rademacher Complexity-Based Generalization Theory to Domain Generalization

    [https://arxiv.org/abs/2611.25650](https://arxiv.org/abs/2611.25650)

    本文将基于Rademacher复杂度的经典泛化理论扩展到领域泛化设定，通过考虑由概率分布构成的领域空间的几何性质，弥补了现有无几何理论无法形式化刻画分布层面泛化与样本层面泛化区别的根本局限。

    

    领域泛化（DG）旨在让模型在多个具有不同分布的源领域上进行训练，以确保其在未见过的目标领域上具有稳健的性能。在此过程中，它将泛化的概念从传统的样本层面提升到了分布层面。现有的理论框架通过引入一个赋予元分布的领域空间，将经典的泛化分析扩展到这一新设定。然而，这些方法通常将领域空间视为没有几何结构的。实际上，领域空间本质上是一个由概率分布组成的无限维、非线性抽象空间，其复杂的几何性质从根本上决定了泛化行为。因此，现有的无几何结构的领域泛化理论存在根本性局限：它们缺乏对分布层面泛化与样本层面泛化之间区别的形式化刻画。

    arXiv:2611.25650v1 Announce Type: new  Abstract: Domain generalization (DG) aims to train models on multiple source domains with distinct distributions to ensure robust performance on unseen target domains. In doing so, it elevates the concept of generalization from the traditional sample level to the distribution level. Existing theoretical frameworks introduce a domain space endowed with a meta-distribution to extend classical generalization analysis to this new setting. However, these approaches typically treat the domain space as devoid of geometry. In reality, the domain space is an inherently infinite-dimensional, nonlinear abstract space composed of probability distributions, and its complex geometric properties fundamentally determine generalization behavior. Consequently, existing geometry-free theories of domain generalization have a fundamental limitation. They lack a formal characterization of what distinguishes distribution-level generalization from sample-level generaliza
    
[^119]: AdsNet：用于直接预测吸附能的图神经网络

    AdsNet: A Graph Neural Network for Direct Adsorption Energy Prediction

    [https://arxiv.org/abs/2611.25646](https://arxiv.org/abs/2611.25646)

    AdsNet 是一种等变图神经网络，通过仅基于吸附质原子及其近邻表面原子的局部图，在单次前向传播中直接快速且准确地预测吸附能，其计算复杂度与体系规模解耦，非常适合高通量催化剂筛选。

    

    快速且准确地预测吸附能对于高通量计算催化剂筛选至关重要，而现有方法要么在评估速度上有所妥协（全图机器学习势），要么在几何敏感性上有所妥协（如 alpha-scheme 等基于配位数的模型）。在此，我们提出了 AdsNet，这是一种等变图神经网络，能够在单次前向传播中直接预测吸附能。它仅在一个局部图上运行，该图只包含吸附质原子及其截断半径内的直接表面近邻。这种局部图表示将计算复杂度与体系总规模解耦，同时编码连续的几何特征（原子间距离和方向矢量），从而对局部成键环境的细微变化保持敏感。AdsNet 采用消息传递架构，在多个相互作用层中使用标量和矢量节点特征。该模型在 AdsTraj 数据集上训练，该数据集包含……

    arXiv:2611.25646v1 Announce Type: cross  Abstract: Fast and accurate prediction of adsorption energies is critical for high-throughput computational catalyst screening, existing methods compromise either on evaluation speed (full-graph machine learning potentials) or geometric sensitivity (coordination-based models like the alpha-scheme). Here, we present AdsNet, an equivariant graph neural network that directly predicts adsorption energies in a single forward pass, which operates on a local graph restricted only to adsorbate atoms and its immediate surface neighbors within a cutoff radius. This local graph formulation decouples computational complexity from total system size while encoding continuous geometric features (interatomic distances and directional vectors) enabling sensitivity to subtle variations in local bonding environments. AdsNet adopts a message-passing architecture with scalar and vector node features over multiple interaction layers. Trained on AdsTraj, a dataset of 
    
[^120]: 擦除错的对象：共享基础模型嵌入中的条件属性泄露与概念擦除

    Erasing the Wrong Thing: Conditional Attribute Leakage and Concept Erasure in Shared Foundation-Model Embeddings

    [https://arxiv.org/abs/2611.25640](https://arxiv.org/abs/2611.25640)

    论文证明了当任务标签与敏感属性相关时，全局概念擦除方法必然保留任务条件的线性属性泄露，并提出闭式方法TC-LEACE，在每个任务层级内强制实施线性防护以消除条件性敏感属性泄露。

    

    医学图像嵌入越来越多地被共享用于授权的下游任务，以替代原始图像，然而它们可能会泄露敏感属性。标准的概念擦除方法针对的是总体层面的防护性。然而，我们证明，当二元任务标签与敏感标签相关联时，全局防护性和任务条件防护性无法在保留非零线性任务信号的情况下同时成立，因此全局擦除必须要么消除该任务信号，要么保留任务条件的线性泄露。我们进一步展示了全局擦除如何影响线性读取器对具有不同敏感属性的正负样本的判别能力，并从擦除前的统计数据推导出其效用代价的近似闭式预测。受这一冲突的启发，我们提出了任务条件LEACE（TC-LEACE），这是LEACE的一个闭式变体，它在其拟合数据的每个任务层级内强制实施线性防护性。

    arXiv:2611.25640v1 Announce Type: cross  Abstract: Medical-image embeddings are increasingly shared in place of raw images for authorized downstream tasks, yet they may reveal sensitive attributes. Standard concept-erasure methods target population-level guardedness. However, we prove that, when binary task and sensitive labels are associated, global and task-conditional linear guardedness cannot both hold while a nonzero linear task signal is preserved, so global erasure must either eliminate that signal or retain task-conditional linear leakage. We further show how global erasure affects linear readers' discrimination between positive and negative samples with different sensitive attributes, deriving an approximate closed-form prediction of its utility cost from pre-erasure statistics. Motivated by this conflict, we introduce Task-Conditional LEACE (TC-LEACE), a closed-form variant of LEACE that enforces linear guardedness within each task stratum of its fitting data. We evaluate fou
    
[^121]: 单一样本无法看到什么：为什么更少的去噪步骤能在物理基准上获得更高分数

    What One Sample Cannot See: Why Fewer Denoising Steps Can Score Higher on Physics Benchmarks

    [https://arxiv.org/abs/2611.25632](https://arxiv.org/abs/2611.25632)

    现有Physics-IQ基准通过将生成视频与单一参考录像比较，会奖励牺牲多样性、趋于“折中”的模糊预测，即使这些预测的分布更不准确；本文提出仅需两个样本的Kernel Physics-IQ基准来消除这一偏差，并证明更少的去噪步骤虽然提高了官方分数，却降低了真实的物理保真度。

    

    当一场风暴可能向北或向南移动时，预测中间路径在平方误差下可能比任何一条合理路径的得分都高。当视频生成器被作为世界模型进行测试时，同样的悖论出现了：即使视频变得模糊或提前停止运动，更少的去噪步骤反而能提高物理分数。为什么？因为Physics-IQ将每个生成的视频与一段真实录像进行比较，而一个场景实际上可以有多种有效的未来。精确的分解表明，该基准的分数可能奖励生成未来之间变异性的减少，即使其分布变得不那么准确。我们提出Kernel Physics-IQ以在期望意义上消除这种激励，每个条件只需要两个独立样本。在所有五个图像到视频模型中，两步采样都降低了Kernel Physics-IQ分数。以CogVideoX为例，官方分数上升了0.055，而我们的分数却下降了0.131。改变采样调度还揭示了官方分数未能捕捉到的更多差异。

    arXiv:2611.25632v1 Announce Type: cross  Abstract: When a storm can head north or south, a forecast down the middle may score better under squared error than either plausible path. The same paradox appears when video generators are tested as world models: fewer denoising steps can raise physics scores even as videos blur or stop moving early. Why? Physics-IQ compares each generated video with one recording, although a scene can have multiple valid futures. An exact decomposition shows its scores can reward reduced variation among generated futures, even when their distribution becomes less accurate. We introduce Kernel Physics-IQ to remove this incentive in expectation, needing only two independent samples per condition. Two-step sampling lowers Kernel Physics-IQ in all five image-to-video models. For CogVideoX, the official score rises by 0.055 while ours falls by 0.131. Changing the sampling schedule reveals further differences official scores miss. In controlled physical systems wit
    
[^122]: NIMIS：一种结构无关、化学信息驱动的槽注意力框架，用于分离一维 ¹H-NMR 混合物光谱

    NIMIS: A Structure-Agnostic, Chemically Informed Slot-Attention Framework for Demixing 1D $^1$H-NMR Mixture Spectra

    [https://arxiv.org/abs/2611.25629](https://arxiv.org/abs/2611.25629)

    NIMIS 是一种无需分子结构先验的深度学习框架，它将 J-耦合化学信息融入 Transformer 的多头注意力机制，并借助槽注意力将一维 ¹H-NMR 混合物光谱中的峰按其来源组分进行分组分离。

    

    化学反应的产物通常是复杂的混合物，必须先将其分离为各个组分，才能获得可解释的 ¹H-NMR 光谱。直接分析粗混合物的光谱将加速反应分析，并促进对新化学空间的探索。我们提出了 NIMIS，这是一个结构无关的、基于注意力的深度学习框架，它根据峰值所来源的组分，对一维 ¹H-NMR 混合物光谱中的峰进行分组。J-耦合信息被直接引入多头 Transformer 编码器的注意力机制中，由此得到的上下文化峰值嵌入再通过槽注意力被分离为各个组分。NIMIS 在基于 NMRexp 数据库中实验光谱构建的 158,005 个合成混合物上进行了训练，涵盖由 C、H、N、O、S、F、Cl、Br 和 I 组成的最多含 20 个重原子的分子。在留出的分子上，它达到了峰值到组分（摘要在此处截断）

    arXiv:2611.25629v1 Announce Type: new  Abstract: The products of chemical reactions are usually complex mixtures that must be separated into their components before interpretable $^1$H-NMR spectra can be obtained. Direct analysis of the spectra of crude mixtures would accelerate reaction analysis and facilitate the exploration of new chemical space. We present NIMIS, a structure-agnostic, attention-based deep learning framework that groups the peaks of a 1D $^1$H-NMR mixture spectrum according to the component from which they originate. J-coupling information is introduced directly into the attention mechanism of a multi-head Transformer encoder, and the resulting contextualised peak embeddings are separated into components by slot attention. NIMIS was trained on 158,005 synthetic mixtures constructed from experimental spectra in the NMRexp database, covering molecules of up to 20 heavy atoms composed of C, H, N, O, S, F, Cl, Br and I. On held-out molecules it reaches a peak-to-compone
    
[^123]: 无需物理损失的物理定律？通过强迫响应探测视频世界模型中的学习动力学

    Physical Laws without Physics Loss? Probing Learned Dynamics in Video World Models through Forced Response

    [https://arxiv.org/abs/2611.25626](https://arxiv.org/abs/2611.25626)

    本文通过摇晃悬挂物体支撑点的强迫响应实验，探究视频世界模型是否真正学习物理定律，发现仅用流匹配损失训练的模型不仅能复现线性响应（含共振），还能捕捉到训练数据中仅占0.74%摆动平方位移的非线性软化等高阶动力学。

    

    视频世界模型是在没有物理损失函数的情况下学习物理定律，还是仅仅模仿运动？整体运动和外观评分无法回答这一问题。我们通过非线性强迫响应来研究它：用一个激励去摇晃悬挂物体的支撑点。我们引入了双重视觉-数值条件化，并将六个图像到视频的骨干网络适配为仅使用流匹配损失来预测这种运动。Wan模型能够准确预测运动，并泛化到未见过的激励、物体、姿态和场景。反事实测试表明其能力超越了简单地回放训练视频。但它们究竟学到了什么定律？我们利用动力学和几何的泰勒展开设计了三个层级的测试。所有六个模型都再现了线性响应，包括共振现象；其中五个模型捕捉到了更高阶的动力学，包括非线性软化效应。值得注意的是，软化效应在训练数据中几乎不存在：仅占摆动平方位移的0.74%。对双摆的扩展……

    arXiv:2611.25626v1 Announce Type: cross  Abstract: Do video world models learn physical laws without a physics loss, or merely imitate motion? Overall motion and appearance scores cannot settle it. We investigate this through nonlinear forced response: an excitation shakes the support of a suspended object. We introduce dual visual-numeric conditioning and adapt six image-to-video backbones to predict this motion using only the flow-matching loss. Wan predicts motion accurately and generalizes to unseen excitations, objects, poses, and scenes. Counterfactual tests suggest capabilities beyond replaying training videos. But what laws have they learned? We design three tiers of tests using Taylor expansions of the dynamics and geometry. All six reproduce the linear response, including resonance; five capture higher-order dynamics, including nonlinear softening. Remarkably, softening is barely present in training: only 0.74% of the swing's squared displacement. Extensions to a double pendu
    
[^124]: 基于QUBO的特征选择用于MRI欠采样模式设计

    QUBO-based feature selection for MRI undersampling pattern design

    [https://arxiv.org/abs/2611.25625](https://arxiv.org/abs/2611.25625)

    该论文创新性地将MRI笛卡尔欠采样掩模设计转化为基于互信息与条件互信息的QUBO特征子集选择问题，利用模拟退火和量子退火求解，在15%至40%的所有测试采样率下均取得了更优的重建质量。

    

    我们将笛卡尔磁共振成像（MRI）的欠采样掩模设计形式化为一个特征子集选择问题。每条采样线由一个二元特征表示，用于指示其是否包含在每个候选掩模中。利用候选掩模及其在训练集上的评估结果，我们估计了互信息和条件互信息。这些估计值与采样数量惩罚项共同定义了一个二次无约束二进制优化（QUBO）模型。该模型通过模拟退火和量子退火求解，以确定特征子集及其对应的采样掩模。我们使用fastMRI膝关节幅值参考图像，在患者不相交的训练集与测试集设置下，通过傅里叶欠采样仿真中的零填充重建，在测试集上评估了所选择的子集。在从15%到40%的所有六个测试目标采样率下，所得重建具有更低的均方根

    arXiv:2611.25625v1 Announce Type: new  Abstract: We formulate undersampling mask design for Cartesian magnetic resonance imaging (MRI) as a feature subset selection problem. Each sampling line is represented by a binary feature indicating whether it is included in each candidate mask. Using candidate masks and their training-set evaluations, we estimate mutual information (MI) and conditional mutual information (CMI). These estimates and a sampling-count penalty define a quadratic unconstrained binary optimization (QUBO) model. The model is solved using simulated annealing and quantum annealing to determine a feature subset and its corresponding sampling mask. We assess the selected subsets on the test set through zero-filled reconstruction in Fourier undersampling simulations using fastMRI knee magnitude reference images, with patient-disjoint training and test sets. At all six tested target sampling rates from 15% to 40%, the resulting reconstructions have lower mean root mean square
    
[^125]: 联合深度-宽度缩放下卷积神经网络对NTK机制的偏离

    Deviations from the NTK Regime in CNNs under Joint Depth-Width Scaling

    [https://arxiv.org/abs/2611.25624](https://arxiv.org/abs/2611.25624)

    该论文通过数值实验揭示了卷积神经网络中NTK离散度与训练时的核旋转均随深度与宽度之比（L/C）增长，且在临界初始化下离散度曲线随L/C近似坍缩，表明CNN存在偏离NTK机制的非平凡联合深度-宽度缩放区域。

    

    神经正切核（NTK）的集中性与训练时间恒定性取决于网络宽度、深度和初始化之间的相互作用。尽管联合深度-宽度缩放已在全连接网络、残差网络和循环网络中得到研究，但其在卷积神经网络（CNN）中的影响仍知之甚少。我们在CNN中通过数值方法研究这些影响，考察初始化时的NTK离散度以及训练过程中的核旋转。设深度 $L$ 为卷积层的数量，宽度 $C$ 为每层的通道数。我们发现，在临界和混沌初始化区域中，NTK离散度随 $L/C$ 的增大而增加，但在有序区域中保持较小。在临界初始化下，离散度曲线作为 $L/C$ 的函数近似坍缩，这表明存在一个非平凡的联合缩放区域。在训练过程中，核旋转通常随 $L/C$ 的增大而增加，即使在初始（原文在此处截断）的有序区域中也是如此。

    arXiv:2611.25624v1 Announce Type: new  Abstract: The concentration and training-time constancy of the neural tangent kernel (NTK) depend on the interplay between network width, depth, and initialization. While joint depth-width scaling has been studied in fully-connected, residual, and recurrent networks, its effects in convolutional neural networks (CNNs) remain less understood. We investigate these effects numerically in CNNs, examining NTK dispersion at initialization and kernel rotation during training. Let depth $L$ be the number of convolutional layers, and width $C$ the number of channels per layer. We find that NTK dispersion increases with $L/C$ in the critical and chaotic initialization regimes, but remains small in the ordered regime. At critical initialization, dispersion curves approximately collapse as a function of $L/C$, suggesting a nontrivial joint scaling regime. During training, kernel rotation generally increases with $L/C$, even in the ordered regime where initial
    
[^126]: 基于内容自适应空间寻址的分层时空世界建模

    Hierarchical Spatiotemporal World Modeling with Content-Adaptive Spatial Addressing

    [https://arxiv.org/abs/2611.25610](https://arxiv.org/abs/2611.25610)

    提出受海马体记忆与网格细胞启发的分层时空世界模型HiSTFormer，通过局部时间RoPE、多周期空间RoPE和内容自适应寻址，实现有界紧凑记忆，并在MemoryMaze和HM3D上取得优异的长时程预测性能与泛化能力。

    

    在部分可观测环境中的长时程世界建模需要紧凑的记忆机制，使其在定位漂移不断累积的情况下仍能保持空间结构。受海马体情景与空间记忆以及内嗅皮层网格细胞多周期编码的启发，我们提出了HiSTFormer——一种分层时空世界模型。该模型采用逐层组织的局部时间RoPE与多周期空间RoPE，并结合内容条件化的地址偏移，以捕获近期时间上下文和长程空间结构。固定容量的记忆使KV存储和每步注意力成本在序列增长时保持有界。在MemoryMaze和HM3D上，HiSTFormer在训练序列长度内优于强世界模型基线，并对远超训练所见预测时域的任务表现出显著更好的泛化能力，同时模型规模紧凑。该分层时空设计保持有效。

    arXiv:2611.25610v1 Announce Type: cross  Abstract: Long-horizon world modeling in partially observable environments requires compact memories that preserve spatial structure despite accumulated localization drift. Inspired by hippocampal episodic and spatial memory and the multi-periodic coding of entorhinal grid cells, we propose HiSTFormer, a hierarchical spatiotemporal world model, which adopts a layer-wise organization of local temporal RoPE and multi-periodic spatial RoPE, together with content-conditioned address offsets, to capture recent temporal context and long-range spatial structure. A fixed-capacity memory keeps KV storage and per-step attention cost bounded as the sequence grows. On MemoryMaze and HM3D, HiSTFormer outperforms strong world-model baselines within the training sequence length and generalizes substantially better to prediction horizons far beyond those seen during training, while using a compact model. The hierarchical spatiotemporal design remains effective 
    
[^127]: PSTO：用于多尺度对流-扩散-反应系统的具有共享隐式更新的PDE结构化输运算子

    PSTO: A PDE-structured transport operator with shared implicit updates for multiscale CDR systems

    [https://arxiv.org/abs/2611.25607](https://arxiv.org/abs/2611.25607)

    PSTO通过将PDE结构嵌入神经架构并在所有输运变量间共享单一输运算子模块、结合松弛不动点迭代实现类隐式修正，仅需无反应数据训练即可在推理时无需重训练地泛化到完整的多尺度对流-扩散-反应系统，且不增加参数量。

    

    多尺度对流-扩散-反应（CDR）系统将空间输运与局部刚性反应动力学耦合在一起，使其数值模拟在计算上代价高昂。传统的神经代理模型需要昂贵的耦合CDR训练数据，且其参数量随输运变量数量的增加而增长，限制了其在高维CDR系统上的可扩展性。本文提出了一种PDE结构化输运算子（PSTO），它仅从无反应数据中学习对流-扩散动力学。通过将PDE结构嵌入到神经架构中，PSTO能够在推理阶段泛化到完整的CDR系统而无需重新训练。单个输运算子模块在所有输运变量之间共享，并通过松弛不动点迭代进行精化，在潜空间中执行类隐式的修正。这在不增加可训练参数的情况下提高了递归多步预测的稳定性。我们对PSTO进行了评估……

    arXiv:2611.25607v1 Announce Type: new  Abstract: Multiscale convection-diffusion-reaction (CDR) systems couple spatial transport with local stiff reaction kinetics, making their numerical simulation computationally demanding. Conventional neural surrogates require costly coupled CDR training data, and their parameter counts scale with the number of transported variables, limiting scalability to high-dimensional CDR systems. Here we propose a PDE-Structured Transport Operator (PSTO) that learns convection--diffusion dynamics solely from reaction-free data. By embedding the PDE structure into the neural architecture, PSTO generalizes to full CDR systems at inference without retraining. A single transport operator block is shared across all transported variables and refined through relaxed fixed-point iterations, performing an implicit-like correction in latent space. This improves stability during recursive multi-step prediction without adding trainable parameters. We evaluate PSTO on th
    
[^128]: 论GNN结构稳定性界的信息量：GIN与GAT的对比

    On the Informativeness of Structural Stability Bounds for GNNs: GINs versus GATs

    [https://arxiv.org/abs/2611.25603](https://arxiv.org/abs/2611.25603)

    本文推导了GIN和单头GATv1在确定性边扰动下的结构稳定性界，并与扰动无关的直径界对比，精确刻画了GIN稳定性界更紧的条件（在Cora上成立但在低度数的MUTAG上失效），同时发现GATv1的稳定性界从不更紧。

    

    图神经网络的稳定性界限制了当图发生变化时节点表示可能产生的漂移程度，但人们很少将其与最简单的替代方案进行比较：即不依赖于图结构、对任意两个输出之间距离给出的界。我们在确定性边扰动（不必很小）下，推导了图同构网络（GIN）和单头GATv1网络的结构稳定性界，并将它们分别与这种与扰动无关的直径界进行比较。对于GIN，我们通过一个将扰动大小、网络深度和最大度联系起来的条件，精确刻画了稳定性界何时更紧；该条件在Cora数据集上以很大余量成立，但在低度数的MUTAG图上超过几层后便不再成立。对于GATv1，稳定性界从不更紧，这表明需要开展基于注意力质量的分析。

    arXiv:2611.25603v1 Announce Type: cross  Abstract: Stability bounds for graph neural networks limit how much node representations can drift when the graph changes, but they are rarely compared with the simplest alternative: a bound on the distance between any two outputs, whatever the graph. We derive structural stability bounds for Graph Isomorphism Networks (GINs) and single-head GATv1 networks under deterministic edge perturbations that need not be small, and compare each with such a perturbation-independent diameter bound. For GINs, we characterize exactly when the stability bound is tighter, through a condition linking perturbation size, depth and maximum degree; it holds with a wide margin on Cora, but fails beyond a few layers on the low-degree MUTAG graphs. For GATv1, the bound is never tighter, which calls for analyses based on attention masses.
    
[^129]: 面向决策的强化学习用于大语言模型后训练

    Decision-Focused Reinforcement Learning for LLM Post-Training

    [https://arxiv.org/abs/2611.25597](https://arxiv.org/abs/2611.25597)

    该论文提出面向决策的强化学习（DFRL）框架，以下游决策负遗憾作为奖励对LLM进行后训练，使其预测直接最小化下游决策成本且无需下游问题凸性或可微性，并通过启发式热启动和奖励校正两种方法解决了奖励评估开销大的问题。

    

    大语言模型（LLM）作为各类任务中的预测器展现出了巨大潜力，这得益于它们利用非结构化上下文信息的能力。然而，当预测结果作为下游优化问题的参数时，更高的预测精度并不一定转化为更低的决策成本。我们提出了面向决策的强化学习（DFRL），这是一个用于对LLM进行后训练以预测优化参数的框架，它采用下游决策的负遗憾作为奖励。DFRL训练LLM预测器以最小化由其预测所导致的决策成本，无需下游问题具有凸性或可微性，并且能够轻松集成到现有的强化学习后训练框架中。为了解决奖励评估开销高昂的问题，我们开发了两种方法。启发式热启动在初始阶段使用廉价的近似奖励，随后切换为使用精确奖励进行改进。奖励校正……（原文截断）

    arXiv:2611.25597v1 Announce Type: cross  Abstract: Large language models (LLMs) have shown promise as predictors across various tasks, benefiting from their ability to use information from unstructured context. However, when predictions serve as parameters of downstream optimization problems, higher prediction accuracy does not necessarily translate into lower decision costs. We propose decision-focused reinforcement learning (DFRL), a framework for post-training LLMs to predict optimization parameters using negative downstream decision regret as reward. DFRL trains LLM predictors to minimize the costs of the decisions induced by their predictions, without requiring downstream convexity or differentiability, and integrates readily into existing RL post-training frameworks. To address expensive reward evaluation, we develop two approaches. Heuristic warm-starting uses inexpensive approximate rewards during an initial phase before switching to refinement with exact rewards. Reward correc
    
[^130]: Bradley-Terry训练可能扭曲原本准确的语言世界模型的结果概率

    Bradley-Terry Training Can Distort Outcome Probabilities of Initially Accurate Language World Models

    [https://arxiv.org/abs/2611.25594](https://arxiv.org/abs/2611.25594)

    本文证明Bradley-Terry目标函数在基于比较反馈的训练中无法教会语言世界模型真实的结果概率，反而会将原本准确的模型（危害误差仅2.4个百分点）中33-43个百分点的概率扭曲到实际不可能发生的结果上，且这种失真源于目标函数本身而非比较数据。

    

    arXiv:2611.25594v1 公告类型：交叉 摘要：比较反馈（即由评判者指出两个候选结果中哪一个更接近实际发生的情况）是训练为动作结果分配概率的语言世界模型的一种实用方法。我们证明，使用Bradley-Terry目标函数进行拟合时，它未必能教会模型真实的结果概率，甚至可能扭曲原本准确的结果概率。在一个具有枚举结果分布的模拟机器人推动任务中，该目标函数会使一个在未见布局上表现准确的模型（危害误差为2.4个百分点）将33至43个百分点的概率转移到该条件下没有任何枚举分支会产生的结果上。问题的根源在于目标函数本身，而非比较数据：在训练前计算的目标函数最优解可以预测模型最终的去向，而一个了解评判者规则的似然函数能够从相同的比较数据中学习且不产生这种偏移。这种偏移具有鲁棒性：它在未见条件、五种候选法则以及三倍更密集的枚举下均会重现。

    arXiv:2611.25594v1 Announce Type: cross  Abstract: Comparative feedback, in which a judge says which of two candidate outcomes is closer to what happened, is a practical way to train language world models that assign probabilities to an action's outcomes. We show that, fitted with a Bradley-Terry objective, it need not teach the true outcome probabilities and can distort accurate ones. On a simulated robot push with enumerated outcome distributions, this objective takes a model that is accurate on unseen layouts (hazard error 2.4 percentage points, pp) and moves 33 to 43 pp of its probability onto outcomes that none of a condition's enumerated branches produces. The cause is the objective, not the comparisons: its optimum, computed before training, predicts where the model ends up, and a likelihood that knows the judge's rule learns from the same comparisons without the shift. The shift is robust: it recurs on unseen conditions, under five candidate laws and a three-fold denser enumera
    
[^131]: 从数据对齐到参数生成：基于条件扩散的少样本跨城市流量预测

    From Data Alignment to Parameter Generation: Few-Shot Cross-City Traffic Prediction with Conditional Diffusion

    [https://arxiv.org/abs/2611.25592](https://arxiv.org/abs/2611.25592)

    该论文提出了基于条件扩散的参数生成框架（CDPG），通过学习源城市高性能预测器参数的条件分布来直接生成目标城市各节点的模型参数，从而突破了传统数据对齐方法在少样本跨城市流量预测中易受分布偏移、测量噪声和城市特有偏差影响的局限。

    

    准确的移动流量预测对于网络服务管理至关重要，可支持主动的容量规划、拥塞缓解和节能运行。然而，许多城市和地区数据稀缺（例如新部署或监测设备不完善的区域），这使得流量预测成为一个少样本学习问题，并限制了标准数据驱动模型的实用性。现有的跨城市解决方案主要依赖数据空间对齐来匹配数据丰富城市与数据稀缺城市之间的流量分布，但这种对齐方式在真实世界的分布偏移、测量噪声和城市特有偏差下往往十分脆弱。为了解决这些局限性，我们提出了一种基于条件扩散的参数生成（CDPG）框架。CDPG不再对齐原始数据，而是学习源城市高性能预测器参数的条件分布，并直接生成节点特定的模型参数

    arXiv:2611.25592v1 Announce Type: new  Abstract: Accurate mobile traffic forecasting is crucial for network service management, supporting proactive capacity provisioning, congestion mitigation, and energy-efficient operation. However, many cities and regions are data-scarce (e.g., newly deployed or poorly instrumented areas), thus making traffic prediction a few-shot learning problem and limiting the practicality of standard data-driven models. Existing cross-city solutions primarily rely on data-space alignment to match traffic distributions between data-rich and data-scarce cities, but such alignment is often brittle under real-world distribution shifts, measurement noise, and city-specific bias. To address these limitations, we propose a Conditional Diffusion-based Parameter Generation (CDPG) framework. Instead of aligning raw data, CDPG learns the conditional distribution of high-performing predictor parameters for source cities and directly generates node-specific model parameter
    
[^132]: Epira的设计与评估：一种用于病例调查和接触者追踪的自动化访谈系统

    Design and Evaluation of Epira: An Automated Interview System for Case Investigation and Contact Tracing

    [https://arxiv.org/abs/2611.25577](https://arxiv.org/abs/2611.25577)

    Epira是一种将大语言模型对话与规则控制器相结合的协议控制自动化访谈系统，既能灵活应对受访者澄清需求，又能严格遵循流调协议，为多疾病病例调查和接触者追踪提供了有效的自动化解决方案。

    

    目的：病例调查和接触者追踪是劳动密集型工作。当受访者在透露接触者信息前需要澄清时，网络表格难以应对，而开放式聊天机器人可能偏离工作协议。我们开发并评估了Epira，一种用于多疾病病例调查和接触者追踪的协议控制大语言模型（LLM）访谈系统。材料与方法：Epira将LLM对话与基于规则的控制器相结合，该控制器强制执行必问问题、安全交接和结构化病例记录，并可通过上传问卷适配新疾病。研究测试了本地（Qwen2.5）和云端（GPT-5.4）两种后端。研究1使用108个麻疹和淋病模拟病例（包含真实标准字段和接触者信息），在Epira和开放式条件下进行测试（共864次会话）。研究2招募了72名志愿者，采用2×2疾病-模型访谈设计，评估完成度、可用性、可接受性和信任度。结果：在研究1中，使用……的Epira（摘要截断）

    arXiv:2611.25577v1 Announce Type: new  Abstract: Objective: Case investigation and contact tracing are labor-intensive. Web forms struggle when respondents need clarification before disclosing contacts, and open-ended chatbots may deviate from protocols. We developed and evaluated Epira, a protocol-controlled large language model (LLM) interviewer for multi-disease case investigation and contact tracing.   Materials and Methods: Epira pairs LLM dialogue with a rule-based controller enforcing required questions, safety handoffs, and a structured case record, and adapts to new diseases by questionnaire upload. Local (Qwen2.5) and cloud (GPT-5.4) backends were tested. Study 1 used 108 simulated cases with ground-truth fields and contacts for measles and gonorrhea under Epira and open-ended conditions (864 sessions). Study 2 enrolled 72 volunteers in a 2$\times$2 disease-by-model interview assessing completion, usability, acceptability, and trust.   Results: Across Study 1, Epira with a lo
    
[^133]: 基于离散词元的强化学习对齐概率天气预报

    Reinforcement-Aligned Probabilistic Weather Forecasting with Discrete Tokens

    [https://arxiv.org/abs/2611.25570](https://arxiv.org/abs/2611.25570)

    WeatherCraft通过离散天气词元将概率生成与强化学习对齐统一起来，将GPT式预训练—后训练范式引入全球中期天气预报，可高效生成集合预报，并利用GRPO在无需对评估器或耦合模型求导的情况下直接优化极端事件预警等下游任务奖励。

    

    全球中期天气预报不仅需要高效生成集合预报以刻画未来状态的不确定性，还需要服务于极端事件预警和跨圈层耦合模拟等下游应用。我们提出WeatherCraft，一个通过离散天气词元将概率生成与强化学习对齐统一起来的框架，将GPT式的预训练—后训练范式引入全球中期天气预报。多尺度分词器将大气状态表示为由粗到细的词元层级结构，Transformer则建模跨时间和跨尺度的因果依赖，并在每个尺度内并行预测词元。直接采样即可生成多样化的集合预报，而精确的词元对数概率使组相对策略优化（GRPO）能够在无需对评估器或耦合模型进行求导的情况下最大化下游任务奖励。在ERA5基准上，Wea……（注：原文摘要不完整）

    arXiv:2611.25570v1 Announce Type: new  Abstract: Global medium-range weather forecasting must not only generate ensembles efficiently to characterize uncertainty in future states, but also serve downstream applications such as extreme-event warnings and cross-sphere coupled simulations. We introduce WeatherCraft, a framework that unifies probabilistic generation and reinforcement learning alignment through discrete weather tokens, bringing the GPT-style pretraining--post-training paradigm to global medium-range weather forecasting. A multiscale tokenizer represents atmospheric states as a coarse-to-fine token hierarchy, while a Transformer models causal dependencies across time and scale and predicts tokens within each scale in parallel. Direct sampling generates diverse ensembles, and exact token log-probabilities enable Group Relative Policy Optimization (GRPO) to maximize downstream task rewards without differentiating through evaluators or coupled models. On the ERA5 benchmark, Wea
    
[^134]: 一段描述值千条示例吗？上下文学习中任务描述的特异性与可靠性

    Is a Description Worth a Thousand Demonstrations? Specificity and Reliability of Task Descriptions in In-Context Learning

    [https://arxiv.org/abs/2611.25560](https://arxiv.org/abs/2611.25560)

    本文从贝叶斯视角将上下文学习中的任务描述建模为信息先验，提出用有效样本量（ESS）来量化描述相对于示例的预测价值，证明了可靠描述的ESS随其特异性和任务维度增长，并通过实验表明元训练的Transformer在训练分布上接近贝叶斯最优的ESS水平。

    

    用于上下文学习的提示词除了可以通过示例来演示任务，或者代替示例演示之外，还可以用文字来描述任务。然而，任务描述如何发挥作用在理论上仍缺乏充分的理解。本文通过上下文学习的贝叶斯视角，将任务描述建模为以特异性和可靠性为特征的信息先验。我们借鉴贝叶斯统计中的有效样本量（ESS）概念，将描述的预测价值量化为等效的示例数量。对于上下文线性回归，我们证明了可靠描述的ESS会随其特异性和任务维度的增加而增长。然而，可能不相关的描述除非同时给出示例，否则其特异性几乎不会带来收益。在实践中，上下文学习者是否会像贝叶斯最优预测器那样使用描述呢？我们的实验表明，元训练的Transformer在训练分布上进行测试时，其表现接近贝叶斯最优的ESS。

    arXiv:2611.25560v1 Announce Type: new  Abstract: A prompt for in-context learning (ICL) can describe a task in addition to, or instead of, demonstrating it through examples. Yet how task descriptions work remains poorly understood theoretically. Through the Bayesian view of ICL, we model descriptions as informative priors characterized by their specificity and reliability. We adapt the effective sample size (ESS) from Bayesian statistics, quantifying a description's predictive value as an equivalent number of demonstrations. For in-context linear regression, we show that a reliable description's ESS grows with its specificity and with the task's dimension. A description that may be irrelevant, however, gains little from its specificity unless demonstrations are given. In practice, do in-context learners use descriptions as a Bayes-optimal predictor would? Our experiments show that meta-trained Transformers come close to the Bayes-optimal ESS when tested on their training distribution. 
    
[^135]: 受保护状态演化的动力学

    Dynamics of Protected State Evolution

    [https://arxiv.org/abs/2611.25541](https://arxiv.org/abs/2611.25541)

    本文为受保护状态演化建立了动力学框架，通过按执行顺序组合转移记录来区分净消耗相同但初始储备需求不同的协议，并利用加权Gram算子、商构造与协变三次型刻画线性响应椭球、有限预算响应集的不可比性以及奇异门处的顺序可行性。

    

    具有相同受保护观测的状态可能允许不同的未来干预。每个转移指定其后继状态、响应增量、所需的初始储备以及净消耗。按执行顺序组合这些记录，可以区分净消耗相同但初始储备需求不同的协议。在固定状态下，一个加权Gram算子确定了线性响应椭球，而一个商构造识别出只能通过模块间合作才能达到的方向。具有相同完全仿射数据的非线性系统可能拥有不可比较的有限预算响应集。在正则坐标系中，二次最小成本的一阶修正是一个协变三次型，它涉及成本的三阶导数以及保护与响应的Hessian矩阵。在奇异门处，顺序可行性取决于沿路径的峰值储备。一个历史加权算子能够检测到超出旧信道范围之外的响应。

    arXiv:2611.25541v1 Announce Type: cross  Abstract: States with identical protected observations can admit different future interventions. Each transition specifies its successor, response increment, required initial reserve and net consumption. Composing these records in execution order distinguishes protocols with equal net consumption and different initial reserve requirements. At a fixed state, a weighted Gram operator determines the linear response ellipsoid and a quotient identifies directions attainable only through cooperation between blocks. Nonlinear systems with identical complete affine data can have incomparable finite-budget response sets. In regular charts, the first correction to quadratic minimum cost is a covariant cubic form involving the third derivative of cost and the Hessians of protection and response. At singular gates, sequential feasibility depends on the peak reserve along the path. A history-weighted operator detects responses outside the old channel's range
    
[^136]: 迈向自回归的神经网络参数生成

    Towards Autoregressive Neural Network Parameter Generation

    [https://arxiv.org/abs/2611.25538](https://arxiv.org/abs/2611.25538)

    该论文提出指令引导的参数生成方法IGPG，将神经网络参数生成形式化为自回归建模问题，通过把预训练参数编码到离散域，使大语言模型的零样本泛化能力得以迁移到权重生成任务中。

    

    大型语言模型（LLMs）近来在机器学习领域确立了新的范式，展现出卓越的零样本泛化能力和复杂推理能力。与此同时，神经网络参数生成已成为实现快速任务适配与模型合成的有前景的研究方向。然而，现有的权重生成技术难以联合学习多样化模型的分布，难以在单一生成器中编码异构架构，也难以实现分布外泛化。因此，我们提出这样一个问题：能否将预训练参数编码到离散域中，从而将LLMs在文本生成中展现的零样本泛化能力迁移到权重生成任务上。基于这一观点，我们提出了一种受LLM启发的方法——指令引导的参数生成（IGPG），该方法将网络参数生成形式化为一个自回归建模问题。我们首先对预训练权重进行量化……

    arXiv:2611.25538v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have recently established a new paradigm in machine learning, demonstrating exceptional zero-shot generalization and complex reasoning capabilities. Concurrently, neural network parameter generation has emerged as a promising avenue for rapid task adaptation and model synthesis. Existing weight generation techniques, however, struggle to jointly learn the distribution of diverse models, to encode heterogeneous architectures within a single generator, and to generalize out of distribution. We therefore ask whether pretrained parameters can be encoded into a discrete domain so that the zero-shot generalization LLMs exhibit in text generation can be transferred to weight generation. Following this view, we propose an LLM-inspired approach, Instruction-Guided Parameter Generation (IGPG), that formulates network parameter generation as an autoregressive modeling problem. We first quantize pretrained weights with
    
[^137]: 差分隐私分数估计及其在基于分数的生成模型中的应用

    Differentially Private Score Estimation with Applications to Score-Based Generative Models

    [https://arxiv.org/abs/2611.25536](https://arxiv.org/abs/2611.25536)

    本文研究了差分隐私约束下的分数估计问题，针对具有局部 Hölder 连续分数的次高斯分布，提出了一种基于二次 B 样条和指数机制的采样式隐私分数估计器，并将其应用于隐私保护的基于分数的生成模型。

    

    估计一个分布的分数函数在统计学和机器学习中扮演着核心角色，其应用涵盖分布估计、采样以及基于分数的生成建模。然而，从敏感数据中学习分数会引发隐私问题。这些担忧在基于分数的生成建模中尤为突出，因为训练后的模型可能会记住个别训练样本，并通过成员推断和数据提取攻击泄露敏感信息。尽管非隐私的分数估计已被广泛研究，但其隐私版本（即隐私保护下的分数估计）的统计极限仍知之甚少。在这项工作中，我们研究了在 $(\alpha,\beta)$-差分隐私约束下，对具有局部 $\gamma$-Hölder 连续分数的 $d$ 维次高斯分布进行分数估计的问题。我们开发了一种基于采样的隐私分数估计器，该估计器使用了二次 B 样条和指数机制，并且我们实现了……

    arXiv:2611.25536v1 Announce Type: cross  Abstract: Estimating the score function of a distribution plays a central role in statistics and machine learning, with applications in distribution estimation, sampling, and score-based generative modeling. However, learning scores from sensitive data raises privacy concerns. These concerns are particularly pronounced in score-based generative modeling, where trained models may memorize individual training examples and expose sensitive information through membership inference and data extraction attacks.   Although non-private score estimation has been extensively studied, the statistical limits of its private counterpart remain less understood. In this work, we study score estimation under $(\alpha,\beta)$-differential privacy for $d$-dimensional sub-Gaussian distributions with locally $\gamma$-H\"older-continuous scores. We develop a sampling-based private score estimator using quadratic B-splines and the exponential mechanism, and we impleme
    
[^138]: 面向整页优化的生成式排序策略学习

    Learning Generative Ranking Policies for Whole-Page Optimization

    [https://arxiv.org/abs/2611.25527](https://arxiv.org/abs/2611.25527)

    提出一种局部页面级训练目标，可精确拆分为逐位置的训练信号，使生成式排序策略在整页优化中动态生成更高分的页面，同时克服了现有方法中统一页面级信号和固定模仿目标的双重局限。

    

    推荐页面是作为一个整体被消费的：页面上的物品既相互争夺注意力又彼此互补，因此最优页面并非简单地由各自最优的物品并排放置而成。我们研究整页优化问题，即联合地选择和排列页面上的物品以最大化页面整体的得分。可能的页面数量呈组合式增长，因此好的页面必须通过生成而非枚举的方式获得。一些生成式重排方法对采样页面中的每个选择都使用相同的页面级训练信号；另一些方法则模仿在训练前构建的目标页面，而这些目标页面在生成器不断改进的过程中基本保持固定不变。我们用一个局部页面级目标同时解决了上述两个局限，该目标反复地将当前生成器向得分更高的页面推进。这一目标可以精确地拆分为每个位置一个训练目标，每个目标通过完整页面的得分来比较同一部分页面下的候选物品……（摘要在此处被截断）

    arXiv:2611.25527v1 Announce Type: cross  Abstract: A recommendation page is consumed as a whole: the items on it compete for attention and complement one another, so the best page is not simply the individually best items placed side by side. We study whole-page optimization, which selects and orders the items on a page jointly to maximize the score of the page. The number of possible pages grows combinatorially, so good pages must be generated rather than enumerated. Some generative re-ranking methods use the same page-level training signal for every choice in a sampled page. Others imitate target pages built before training, which remain largely fixed as the generator improves. We address both limitations with one local page-level objective, which repeatedly moves the current generator toward pages with higher score. This objective splits exactly into one training target per position, and each target compares candidates at the same partial page by the scores of the complete pages the
    
[^139]: 积分约束何时以及为何能够提高物理信息神经网络的精度？

    When and why can integral constraints improve the accuracy of physics-informed neural networks?

    [https://arxiv.org/abs/2611.25515](https://arxiv.org/abs/2611.25515)

    该论文通过推导带与不带积分约束的PINN损失函数的稳定性估计，从数学上揭示了积分约束在特定参数范围内通过恢复标准损失对弱控制误差分量的一致控制来提升精度的机理与适用条件。

    

    在损失函数中加入通过对控制方程进行积分所得到的约束，已被报道可以在若干应用中提高物理信息神经网络（PINN）的精度，但对这些益处的数学理解仍然有限。对于若干偏微分方程（PDE），我们推导了带积分约束与不带积分约束的损失函数的稳定性估计。通过比较发现，在某些参数范围内，标准损失对特定误差分量的控制非常弱，而积分约束恢复了对这些分量的一致控制。这解释了积分约束何时以及为何能够提高精度。最后，数值实验支持了这一解释。在预测标准损失对这些分量控制较弱的参数范围内，两种损失在精度上确实表现出明显差异；相反，在标准损失仍保持足够控制的参数范围内……

    arXiv:2611.25515v1 Announce Type: cross  Abstract: Adding constraints obtained by integrating governing equations to the loss has been reported to improve the accuracy of physics-informed neural networks (PINNs) in several applications, but mathematical understanding of these benefits remains limited. For several partial differential equations (PDEs), we derive stability estimates for losses with and without integral constraints. The comparison reveals that in certain parameter regimes, the standard loss has very weak control over specific error components, and that the integral constraints restore uniform control of these components. This explains when and why the constraints improve accuracy. Finally, numerical experiments support this interpretation. In parameter regimes where the standard loss is predicted to have weak control over those components, the two losses indeed show clear differences in accuracy. Conversely, in regimes where the standard loss retains sufficient control, b
    
[^140]: 针对具有未知基测度的平滑对手的序贯概率赋值

    Sequential Probability Assignment against Smoothed Adversaries with Unknown Base Measure

    [https://arxiv.org/abs/2611.25514](https://arxiv.org/abs/2611.25514)

    本文在基测度 μ 未知且无需 Lipschitz 条件的更一般平滑在线学习设定下，证明了针对自适应型与疏忽型平滑对手的序贯概率赋值遗憾上界，其中自适应界由经验 ℓ∞ Hellinger 熵控制。

    

    平滑在线学习近年来被作为一种绕过完全对抗设定下难度结果的方法而受到研究，因为完全对抗设定可能过于悲观。在该框架中，对手被限制从相对于某个固定基测度 μ 具有有界密度的分布中生成上下文。先前的工作大多强假设学习者已知 μ，显著的例外包括 Block 等（2024）和 Blanchard（2025）。本文在 μ 未知的更一般设定下，研究使用平滑且良好指定数据的序贯概率赋值（即基于对数损失的在线学习），并且无需 Block 等（2024）与 Blanchard（2025）中所施加的 Lipschitz 条件。我们证明了针对自适应型与疏忽型平滑对手的遗憾上界：自适应情形的界由经验 ℓ∞ Hellinger 熵控制，而疏忽型情形的界改进了……

    arXiv:2611.25514v1 Announce Type: new  Abstract: Smoothed online learning has recently been studied as a way to bypass hardness results for the fully adversarial setting, which can be overly pessimistic. In this framework, the adversary is constrained to generate contexts from distributions having a bounded density with respect to some fixed base measure $\mu$. Prior work makes the strong assumption that $\mu$ is known to the learner, with notable exceptions including Block et al. (2024) and Blanchard (2025). In this paper, we study sequential probability assignment (a.k.a., online learning with log loss) with smooth, well-specified data in the more general setting where $\mu$ is \emph{unknown}, without the Lipschitz condition imposed in the work of Block et al. (2024) and Blanchard (2025). We prove regret upper bounds against both adaptive and oblivious smoothed adversaries: the adaptive bound is controlled by empirical $\ell_\infty$ Hellinger entropy, while the oblivious bound improv
    
[^141]: 基于差异化Token表示的真实场景语音-声景生成联合语言建模

    Joint Language Modeling for In-the-Wild Speech-Scene Generation with Distinct Token Representations

    [https://arxiv.org/abs/2611.25503](https://arxiv.org/abs/2611.25503)

    提出一种基于大语言模型的框架，利用声源感知的神经音频编解码器从真实场景混合音频中提取差异化对齐的语音与环境Token，并以交错方式联合生成两者，解决了语音与环境声建模混淆及依赖模拟混合数据的问题。

    

    语音-声景生成旨在合成清晰可懂的目标语音，并将其与多样化的环境声音融合为一个连贯的声学场景。该任务在两个方面被认为具有挑战性：其一，语音与环境音频在语义和声学特性上差异显著，但现有方法通常使用单一的混合声学表示对完整场景进行建模；其二，在真实世界的语音场景中，这两种声音密切相关，而传统方法通常依赖于由独立录制音源混合而成的模拟场景。我们提出了一种基于大语言模型（LLM）的框架，用于对语音Token流与环境Token流这两条差异化Token流进行联合建模。其中，一个具备声源感知能力的神经音频编解码器从混合音频中提取时间对齐的语音Token与环境Token，预训练的LLM则以自然语言场景描述为条件，以交错方式生成这两条Token流。

    arXiv:2611.25503v1 Announce Type: cross  Abstract: Speech-scene generation aims to synthesize intelligible target speech together with diverse environmental sounds as a coherent acoustic scene. The task is considered challenging in two respects. Speech and environmental audio differ substantially in their semantic and acoustic characteristics, yet existing approaches typically model the complete scene using a single mixed acoustic representation. At the same time, the two sounds are closely related in real-world speech scenes, whereas conventional approaches usually rely on simulated scenes mixed from independently recorded sources. We propose a large language model (LLM)-based framework for joint modeling of distinct speech and environment token streams. A source-aware neural audio codec extracts temporally aligned speech and environment tokens from mixed audio, and a pretrained LLM generates both token streams in an interleaved fashion conditioned on the natural-language scene descri
    
[^142]: WeightCode：通过权重空间编程为神经网络安装新能力

    WeightCode: Installing New Capabilities into Neural Networks via Weight-Space Programming

    [https://arxiv.org/abs/2611.25476](https://arxiv.org/abs/2611.25476)

    提出WeightCode框架，利用冻结语言模型编写可执行的权重程序，无需梯度训练即可直接修改神经网络权重以安装新能力，同时保留原有功能。

    

    神经网络将其学习到的能力编码在权重之中，然而如何借助显式任务知识来扩展这些能力仍然是一个根本性的挑战。本研究展示了权重空间编程（WSP）——即由可执行程序直接读取、构建或修改网络权重——能够在已训练的神经网络中安装新能力。WeightCode是一个WSP框架，它使用冻结的语言模型来编写可执行的权重程序，利用任务知识和网络的现有权重，将新的计算安装到已训练的神经网络中，而无需基于梯度的训练。一旦安装完成，新能力便可独立于语言模型运行。在给定一个语法或八个演示样本的情况下，权重程序使网络能够执行之前未见过的命令组合（精确匹配率从0%提升至99.8%）和新的拾取动作（成功率从0%提升至98.4%），同时保留原有的命令执行与导航能力。在ImageNet数据集上，四个权重程序实现了……

    arXiv:2611.25476v1 Announce Type: cross  Abstract: Neural networks encode learned capabilities in their weights, yet extending those capabilities with explicit task knowledge remains a fundamental challenge. Here we show that Weight-Space Programming (WSP), in which executable programs read, construct or modify a network's weights directly, can install new capabilities within trained neural networks. WeightCode, a WSP framework, uses a frozen language model to write executable weight programs that install new computations into trained neural networks using task knowledge and their existing weights, without gradient-based training. Once installed, the new capabilities run independently of the language model. Given a grammar or eight demonstrations, weight programs enable previously unseen command compositions (0% to 99.8% exact match) and a new pickup action (0% to 98.4% success), while preserving the original command and navigation capabilities. On ImageNet, four weight programs achiev
    
[^143]: 一种用于评估大语言模型生成的物理式暖通空调（HVAC）数字孪生拓扑规范的四层可执行验证协议

    A four-level executable validation protocol for evaluating LLM-generated topology specifications of physics-based HVAC digital twins

    [https://arxiv.org/abs/2611.25471](https://arxiv.org/abs/2611.25471)

    本研究提出一个包含静态有效性、模型构建、仿真收敛和行为等价性四个层级的可执行验证协议，用于评估LLM生成的HVAC数字孪生规范是否物理有效，并在八个LLM和十四种HVAC拓扑上进行了验证。

    

    空调是制造业与建筑物中的主要能源消耗设备，其基于物理的数字孪生可以根据对设备、连接关系和尺寸参数的声明式规范自动生成。如今，大语言模型（LLM）能够根据自然语言描述撰写此类规范。然而，现有的规范评估方法仅停留在解析成功、运行完成或文本相似性层面，无法拒绝那些在仿真中收敛但物理上错误的模型。本研究提出了一种四层验证协议，用于检验LLM撰写的规范不仅能成功执行，还能产生物理上有效的数字孪生。每份规范需依次通过静态有效性检验、模型构建、仿真收敛，以及与隐藏参考模型的行为等价性验证（等价性带宽通过±20%的尺寸扰动进行校准）。该协议被应用于八个LLM和十四种HVAC拓扑结构，其中五种描述的是一个ex……（摘要内容被截断）

    arXiv:2611.25471v1 Announce Type: cross  Abstract: Physics-based digital twins of air conditioning, a principal energy consumer in manufacturing and buildings, can be generated automatically from a declarative specification of its equipment, connections, and sizing. Large language models (LLMs) now write such specifications from plain-language descriptions. Evaluations of these specifications stop at parse success, completed runs, or textual similarity, and cannot reject a model that converges in simulation but is physically wrong. In this study, a four-level protocol is proposed that tests whether an LLM-written specification not only executes but also yields a physically valid digital twin. Each specification is executed through static validity, model build, simulation convergence, and behavioral equivalence with a hidden reference model within bands calibrated by $\pm$20% sizing perturbations. The protocol was applied to eight LLMs and fourteen HVAC topologies, five describing an ex
    
[^144]: 面向压缩动量误差反馈的与维度无关的缓冲界

    Dimension-Free Buffer Bounds for Compressed-Momentum Error Feedback

    [https://arxiv.org/abs/2611.25463](https://arxiv.org/abs/2611.25463)

    该论文证明，通过将压缩动量重新注入压缩前的累加器，压缩动量误差反馈的缓冲误差界可完全摆脱维度与秩相关的收缩常数，且该结论适用于 top-k 选择与截断 SVD 等任意闭锥度量投影类压缩器。

    

    误差反馈分析通常通过压缩器的收缩常数来约束记忆缓冲区——对于 p 维谱的最佳秩 r 近似，该常数为 r/p——并且它会延续到收敛速率中。我们证明该常数完全可以不出现。当压缩动量被重新注入到压缩前的累加器中时，先前的压缩动量 β₁M_{t-1} 成为投影中一个可行的竞争项，此时缓冲区满足 ‖E_t‖_F ≤ (1−β₁)‖G_t‖_F + γ‖E_{t-1}‖_F，该界不含维度 m、n，也不含秩 r。该论证仅利用“压缩器是到某个闭锥上的度量投影”这一事实，因此它既涵盖 top-k 选择，也涵盖截断 SVD；对来自投影集合的任何选择都成立，无需对奇异值交叉施加任何条件；并且其轨迹跟踪形式对梯度不作任何假设。在标准分析的两阶段模板中，它替代了阶数为……的第二阶段常数（原文摘要在此处截断）。

    arXiv:2611.25463v1 Announce Type: new  Abstract: Analyses of error feedback bound the memory buffer through a compressor contraction constant -- $r/p$ for the best rank-$r$ approximation of a $p$-dimensional spectrum -- which survives into the rate. We show it need not appear at all. When the compressed momentum is re-injected into the pre-compression accumulator, the previous compressed momentum $\beta_1 M_{t-1}$ is an admissible competitor in the projection, and the buffer obeys $\|E_t\|_F \le (1-\beta_1)\|G_t\|_F + \gamma\|E_{t-1}\|_F$, free of the dimensions $m,n$ and of the rank $r$. The argument uses only that the compressor is a metric projection onto a closed cone, so it covers top-$k$ selection as well as truncated SVD, holds for every selection from the projection set with no condition on singular-value crossings, and in its trajectory-tracking form assumes nothing about the gradients. In the two-stage template of the standard analyses it replaces a Stage-2 constant of order 
    
[^145]: ActMem：从潜在记忆推理到感知落地的机器人动作

    ActMem: From Latent Memory Reasoning to Grounded Robotic Action

    [https://arxiv.org/abs/2611.25461](https://arxiv.org/abs/2611.25461)

    提出ActMem框架，在单一视觉-语言模型中将记忆推理与感知锚定解耦，通过潜在记忆思维链条件化流匹配动作生成，从而提升长时程机器人操作性能。

    

    视觉-语言-动作（VLA）模型在真实世界的长时程操作任务中已展现出令人瞩目的成果，而记忆在其中扮演着至关重要的角色。然而，现有的记忆增强型VLA模型要么只是检索过去的观察而不对其含义进行推理，要么在共享上下文中对完整历史进行推理，从而挤占了当前的感知能力。认知科学将二者视为不同的系统：记忆重构过去的经验以决定“做什么”，而感知则引导“如何行动”。因此，我们提出ActMem，这是一个在单一视觉-语言骨干网络内将记忆推理与感知锚定解耦的上下文记忆框架。ActMem构建于具备原生视频理解能力的视觉-语言模型之上，它从带时间戳的交互历史中推断当前子任务，并将该推理过程中形成的隐藏状态保留为潜在记忆思维链，以此条件化流匹配动作生成……

    arXiv:2611.25461v1 Announce Type: cross  Abstract: Vision-Language-Action (VLA) models have shown promising results in real-world long-horizon manipulation, where memory plays a crucial role.   However, existing memory-augmented VLAs either retrieve past observations without reasoning about what they imply, or reason over the full   history in a shared context where it crowds out current perception. Cognitive science treats them as distinct systems: memory reconstructs past   experience to decide what to do, while perception guides how to act. We therefore propose ActMem, an in-context memory framework that   decouples memory reasoning from perceptual grounding within a single vision-language backbone. Built on a vision-language model with native   video understanding, ActMem infers the current subtask from a timestamped interaction history and keeps the hidden states formed during this   inference as a Latent Memory Chain-of-Thought (Memory CoT), which conditions a flow-matching actio
    
[^146]: FamilySpace：从涌现的化学语义到可控晶体生成

    FamilySpace: From Emergent Chemical Semantics to Controllable Crystal Generation

    [https://arxiv.org/abs/2611.25452](https://arxiv.org/abs/2611.25452)

    该论文发现晶体VAE的原子级潜空间会按元素族自发地组织出分离区域（即“涌现的化学语义”），并据此提出FamilySpace框架，通过从特定元素族的潜空间区域采样来控制生成晶体的元素族组成。

    

    潜空间晶体生成模型通常将用于学习原子级表示的变分自编码器（VAE）与流模型相结合，流模型负责将简单的源分布传输到学习到的潜空间分布。然而，以往的研究主要关注生成晶体的质量，而对原子级潜空间的结构及其生成效用基本未加探索。在这项工作中，我们证明了即使是普通的晶体VAE，也会隐式地按照元素族来组织原子级潜空间表示，不同的元素族占据潜空间中大致分离的区域。我们将这一现象称为“涌现的化学语义”。这种结构表明，从特定元素族的潜空间区域进行采样，可以使解码器偏向生成来自相应元素族的原子。基于这一见解，我们提出了FamilySpace，一个用于控制生成晶体元素族组成的框架。

    arXiv:2611.25452v1 Announce Type: new  Abstract: Latent crystal generative models typically combine a variational autoencoder (VAE) for learning atom-wise representations with a flow model that transports a simple source distribution to the learned latent distribution. However, previous studies have mainly focused on the quality of generated crystals, leaving the structure and generative utility of the atom-wise latent space largely unexplored. In this work, we show that even a vanilla crystal VAE implicitly organizes atom-wise latent representations according to elemental families, with different families occupying largely separated regions of the latent space. We term this phenomenon emergent chemical semantics. This structure suggests that sampling from a family-specific latent region can bias the decoder toward atoms from the corresponding elemental family. Building on this insight, we propose FamilySpace, a framework for controlling the elemental-family composition of generated cr
    
[^147]: 基于预条件变分量子线性求解器的逆心电重建

    Inverse ECG Reconstruction via a Preconditioned Variational Quantum Linear Solver

    [https://arxiv.org/abs/2611.25450](https://arxiv.org/abs/2611.25450)

    本文提出了首个将量子线性求解器应用于逆心电重建的量子-经典混合框架，通过拉普拉斯正则化预编码方法降低系统维数并改善病态性，再利用相干变分量子线性求解器（VQLS）求解，并从理论上严格证明了该框架能保持精确解。

    

    逆心电学通过体表测量重建心表面电位，实现心脏电活动的无创映射。然而，该重建过程需要求解稠密且病态的线性系统，其计算成本随着心脏网格的精细化而快速增长。量子计算为求解此类系统提供了新的机遇。本文提出了一种用于逆心电重建的量子-经典混合框架，据我们所知，这是首次将量子线性求解器应用于该问题。我们首先提出了一种拉普拉斯正则化的预编码方法，在降低系统维数的同时改善其条件数。其次，利用相干变分量子线性求解器（VQLS）对得到的降维系统进行编码并求解。最后，我们给出了严格的理论证明，证明所提出的框架能够保持精确解。

    arXiv:2611.25450v1 Announce Type: new  Abstract: Inverse electrocardiography (ECG) reconstructs heart-surface potentials from body-surface measurements, enabling noninvasive mapping of cardiac electrical activity. However, the reconstruction requires solving dense and ill-conditioned linear systems, whose computational cost grows rapidly with the refined heart mesh. Quantum computing offers new opportunities for solving such systems. This paper presents a hybrid quantum-classical framework for inverse ECG reconstruction, which, to the best of our knowledge, is the first to apply a quantum linear solver to this problem. We first propose a Laplacian-regularized pre-encoding procedure to reduce the dimension of the system while improving its conditioning. Second, the resulting reduced system is encoded and solved using a coherent variational quantum linear solver (VQLS). Finally, we provide a rigorous theoretical proof establishing that the proposed framework preserves the exact solution 
    
[^148]: 可组合微调模型的干涉界与合并感知训练动力学

    Interference Bounds and Merge-Aware Training Dynamics for Composable Fine-Tuned Models

    [https://arxiv.org/abs/2611.25430](https://arxiv.org/abs/2611.25430)

    该论文将模型合并的可组合性形式化为任务向量集合的性质，证明合并模型的超额损失由跨任务曲率重叠与更新带符号对齐构成的干涉泛函控制，并提出一种曲率引导的正则化微调方法，使合并模型在满足可组合性判据的同时保持标准的非凸收敛速率。

    

    通过组合参数更新来合并微调模型，已成为多任务训练的一种标准替代方案。我们将可组合性形式化为任务向量集合的一种性质，并证明合并模型的超额损失由一个干涉泛函所控制，该泛函刻画了跨任务的曲率重叠程度以及更新的带符号对齐程度。在二次情形下，该泛函恰好等于超额损失，由此给出一个既必要又充分的判据，它以曲率度量（而非欧氏度量）来衡量任务向量之间的夹角。在由曲率引导的正则项约束下进行微调——该正则项由内存高效优化器所维护的那类曲率统计量构造而成——可保证在每次迭代中，基础曲率度量下的位移始终低于由惩罚系数设定的上限，同时保持标准的 $O(1/\sqrt{T})$ 非凸收敛速率。在仅对基础模（摘要在此处截断）

    arXiv:2611.25430v1 Announce Type: new  Abstract: Combining fine-tuned models by composing their parameter updates has become a standard alternative to multi-task training. We formalize composability as a property of a collection of task vectors and prove that the excess loss of the merged model is controlled by an interference functional of the cross-task curvature overlap and the signed alignment of the updates. In the quadratic regime this functional equals the excess loss, giving a criterion that is necessary as well as sufficient and that measures the angle between task vectors in the curvature metric rather than the Euclidean one. Fine-tuning under a curvature-guided regularizer -- constructed from curvature statistics of the kind memory-efficient optimizers maintain -- keeps displacement in the base curvature metric below a ceiling set by the penalty coefficient at every iterate, at the standard $O(1/\sqrt{T})$ non-convex rate. Under a single Loewner comparison between the base s
    
[^149]: 差分隐私加-删除模型中的实例最优矩估计

    Instance-Optimal Moment Estimation in the Add-Remove Model of Differential Privacy

    [https://arxiv.org/abs/2611.25425](https://arxiv.org/abs/2611.25425)

    本文引入实例相关的误差度量并证明其严格下界，设计了在纯差分隐私加-删除模型下达到这些下界的矩估计机制，实现了比现有方法渐近更优的效用。

    

    在本文中，我们研究了在加-删除模型的纯差分隐私约束下估计经验均值、方差和协方差的问题。虽然现有机制能够达到最优的最坏情况误差，但它们对于单个数据集而言可能并非最优。为了衡量机制在单个数据集上的性能，我们引入了实例相关的误差度量，并证明了其严格下界。随后，我们设计了能够达到这些下界的机制，并证明与现有方法相比，这些机制获得了渐近更优的效用。

    arXiv:2611.25425v1 Announce Type: cross  Abstract: In this paper, we study the problem of estimating empirical means, variances, and covariances under pure differential privacy in the add-remove model. While existing mechanisms achieve optimal worst-case error, they may not be optimal for individual datasets. To measure the performance of mechanisms on individual datasets, we introduce instance-dependent error metrics and prove sharp lower bounds on them. Then, we design mechanisms that achieve these lower bounds and demonstrate that they achieve asymptotically better utility compared to existing methods.
    
[^150]: 基于低秩聚类的坐标基线性老虎机算法及其次线性遗憾保证

    Coordinate Basis Linear Bandits with Sublinear Regret over Low-Rank Clusters

    [https://arxiv.org/abs/2611.25424](https://arxiv.org/abs/2611.25424)

    该论文提出坐标基线性老虎机算法，通过聚类机制在每个聚类中维护精确正交基和低维坐标表示以保留完整协方差信息，在高维线性老虎机场景中同时实现了计算效率提升与鲁棒的次线性遗憾保证。

    

    arXiv:2611.25424v1 公告类型：new 摘要：我们研究了在动作集有限的高维环境下线性老虎机算法的计算效率问题。在线性老虎机中，特征向量维度 $d$ 的增加会导致每一轮更新的计算成本增长至 $O(d^2)$。传统的低秩近似方法（如 SOFUL）通过固定尺寸的矩阵草图技术来降低计算量，但当数据谱尾部较重且近似秩选择不当时，这类方法存在产生无效线性遗憾的风险。为了保证遗憾收敛并有效降低计算成本，我们引入了一种聚类机制，并提出了坐标基线性老虎机算法。我们的方法通过维护每个聚类列空间的精确正交基以及低维坐标表示，保留了每个聚类内的完整协方差信息，从而保证了鲁棒的次线性遗憾……

    arXiv:2611.25424v1 Announce Type: new  Abstract: We study the problem of computational efficiency for linear bandits in high-dimensional settings with a finite arm set. In linear bandits, the increase in the dimension $d$ of the feature vectors leads to growing computational costs of $O(d^2)$ at each round of update. Traditional low-rank approximation methods such as SOFUL reduce computation via fixed-size matrix sketching, yet run the risk of incurring vacuous linear regret when the spectral tail of the data is heavy and the approximation rank is inadequately selected. To guarantee regret convergence and effectively reduce computational costs, we introduce a clustering mechanism and propose the Coordinate Basis Linear Bandit (CBL) algorithm. Our method preserves the full covariance information in each cluster by maintaining an exact orthonormal basis of the cluster's column space together with a low-dimensional coordinate representation, which guarantees robust sublinear regret withou
    
[^151]: 从序列不确定性到表型假设：生物医学知识图谱上的不确定性条件化推断

    From Sequence Uncertainty to Phenotype Hypotheses: Uncertainty-Conditioned Inference over Biomedical Knowledge Graphs

    [https://arxiv.org/abs/2611.25422](https://arxiv.org/abs/2611.25422)

    该论文提出一种不确定性条件化的路径能量框架，将基因组基础模型的序列预测及其校准不确定性转化为生物医学知识图谱上可解释、可追溯的表型假设分布，无需患者级表型监督。

    

    基因组基础模型能够提供变异层面的证据，但要将这些证据与表型假设联系起来，需要在整个结构化推理过程中考虑不确定性。我们提出了一种不确定性条件化的路径能量框架，该框架将序列预测和经过校准的预测不确定性转化为图谱支持的表型假设上的显式分布，且无需患者层面的表型监督。一个可解释的学习能量函数对可接受的基因—疾病—表型路径进行评分，查询不确定性控制目标温度，路径概率被聚合为表型假设。这一公式化方法使证据质量成为假设广度的显式决定因素，同时保留了可追溯的多路径支持。对可枚举候选集的精确归一化为评估经Hastings修正的学习修订算子提供了参考基准。在AlzKB和Hetionet上，表型层面的熵追踪……（原文在此处截断）

    arXiv:2611.25422v1 Announce Type: new  Abstract: Genomic foundation models provide variant-level evidence, but connecting that evidence to phenotype hypotheses requires accounting for uncertainty throughout structured reasoning. We introduce an uncertainty-conditioned path-energy framework that converts sequence predictions and calibrated predictive uncertainty into explicit distributions over graph-supported phenotype hypotheses, without patient-level phenotype supervision. An interpretable learned energy scores admissible gene--disease--phenotype routes, query uncertainty controls the target temperature, and path probabilities are aggregated into phenotype hypotheses. This formulation makes evidence quality an explicit determinant of hypothesis breadth while retaining traceable multi-path support. Exact normalization over enumerable candidate sets provides a reference for evaluating a Hastings-corrected learned revision operator. On AlzKB and Hetionet, phenotype-level entropy tracks 
    
[^152]: FG-GraphBench：基于多模态场景图进行细粒度视觉理解的基准测试与增强

    FG-GraphBench: Benchmarking and Enhancing Fine-Grained Visual Understanding with Multimodal Scene Graphs

    [https://arxiv.org/abs/2611.25408](https://arxiv.org/abs/2611.25408)

    该论文提出了细粒度多模态场景图基准 FG-GraphBench（涵盖细粒度视觉问答、图像描述与内容描述三个任务）及 FiGRA 方法，以提升多模态大语言模型利用场景图结构信息进行细粒度视觉理解的能力。

    

    细粒度视觉理解要求多模态大语言模型（MLLM）能够识别微小的视觉实体并理解它们之间的关系。场景图为表示这类信息提供了一种自然的方式，它将视觉实体组织为节点，将其之间的关系组织为边。然而，我们发现了两个挑战：现有场景图基准对细粒度实体及其之间关系的覆盖不足，且现有的细粒度视觉理解方法尚未在多模态大语言模型中有效利用这种图结构信息。为应对这些挑战，我们提出了细粒度多模态场景图基准和 FiGRA。FG-GraphBench 是一个包含三个互补任务的基准：细粒度视觉问答（FG-QA）、图像描述生成（FG-Caption）和图像内容描述生成（FG-Description）。我们开发了一个两阶段的构建流水线……（摘要在此处被截断）

    arXiv:2611.25408v1 Announce Type: cross  Abstract: Fine-grained visual understanding requires multimodal large language models (MLLMs) to recognize small visual entities and understand their relationships. Scene graphs offer a natural way to represent this information, organizing visual entities as nodes and their relationships as edges. However, we identify two challenges: existing scene graph benchmarks provide insufficient coverage of fine-grained entities and the relationships among them, and existing methods for fine-grained visual understanding have yet to effectively leverage such graph-structured information in MLLMs. To address these challenges, we introduce Fine-Grained Multimodal Scene Graph Benchmark (FG-GraphBench) and FiGRA. FG-GraphBench is a benchmark comprising three complementary tasks: fine-grained visual question answering (FG-QA), image captioning (FG-Caption), and image description generation (FG-Description). We develop a two-stage construction pipeline that disc
    
[^153]: Antom 三合一 Transformer：支付基础模型

    Antom 3-in-1 Transformer: A Payment Foundation Model

    [https://arxiv.org/abs/2611.25403](https://arxiv.org/abs/2611.25403)

    本文提出了面向支付风险检测的支付基础模型 Antom 三合一 Transformer，通过融合序列、表格和图表示进行预训练，实现了跨业务迁移能力的大幅提升，并在线上实验中显著降低了支付损失率。

    

    我们报告了 Antom 三合一 Transformer 的开发工作，这是一个面向支付风险检测的支付基础模型，它集成了序列、表格和图三种表示形式。该模型包含约 70 亿参数，每个任务路由仅激活约 17 亿参数。该模型主要在蚂蚁国际的内部支付数据上进行预训练，以学习可复用的行为和关系表示。在一项跨业务迁移实验中，预训练使目标业务指标相比相同架构从零训练提升了 34.5%。在仅有 25% 训练标签的情况下，预训练模型在不同商户领域的平均精度和 Recall@0.5% 上均优于使用全量标签从零训练的基线模型。在蚂蚁国际十亿级支付平台进行的为期四个月的在线实验中，该模型相比 XGB-Online 将整体损失率降低了 37.88%。

    arXiv:2611.25403v1 Announce Type: new  Abstract: We report the development of Antom 3-in-1 Transformer, a payment foundation model that integrates sequence, tabular, and graph representations for payment risk detection. The model contains approximately 7B parameters, with approximately 1.7B activated per task route. Antom 3-in-1 Transformer is pretrained to learn reusable behavioral and relational representations, primarily on Ant International's internal payment data. In a cross-business transfer experiment, pretraining improves the target business metric by 34.5% over the same architecture trained from scratch. With only 25% of the training labels, the pretrained model outperforms the full-label scratch baseline in Average Precision (AP) and Recall@0.5% on different merchant domains. In a four-month online experiment on Ant International's billion-scale payment platform, the model reduced the overall loss rate by 37.88% relative to XGB-Online.
    
[^154]: 基于表示校正的蛋白质功能语义对齐

    Protein Functional Semantic Alignment via Representation Rectification

    [https://arxiv.org/abs/2611.25396](https://arxiv.org/abs/2611.25396)

    提出SAVR对比式后训练方法，利用Pfam家族同源性信号校正蛋白质语言模型的表示结构，无需任务标签即可显著提升下游功能预测性能。

    

    蛋白质语言模型（PLMs）通常采用基于似然的目标进行预训练，例如掩码语言建模或去噪。这些目标能够捕捉序列层面的规律，但可能产生功能信号与非功能变异（如序列同一性、组成偏差和常见局部基序）相互纠缠的表示，从而为下游功能预测造成读取瓶颈。我们提出了基于表示校正的功能语义对齐方法（SAVR），这是一种对比式后训练方法，利用Pfam家族共同成员关系作为同源性引导的离散语义信号来重塑PLM的表示。SAVR减少了方向性主导，并在嵌入维度之间重新分配方差，降低了方差集中度，同时提高了嵌入协方差谱的有效维度。在后训练过程中未观察任何任务特定标签的情况下，SAVR达到了最先进的性能。

    arXiv:2611.25396v1 Announce Type: new  Abstract: Protein language models (PLMs) are commonly pretrained with likelihood-based objectives such as masked language modeling or denoising. These objectives capture sequence-level regularities, but can produce representations in which functional signals are entangled with non-functional variation (e.g., sequence identity, compositional bias, and frequent local motifs), creating a readout bottleneck for downstream function prediction. We introduce Functional Semantic Alignment via Representation Rectification (SAVR), a contrastive post-training method using Pfam family co-membership as a homology-guided discrete semantic signal to reshape PLM representations. SAVR reduces directional dominance and redistributes variance across embedding dimensions, decreasing variance concentration, and increasing effective dimensionality of the embedding covariance spectrum. Without observing any task-specific labels during post-training, SAVR achieves state-
    
[^155]: TuringSpace：通过交互式验证码在行动中落实空间智能

    TuringSpace: Grounding Spatial Intelligence in Action with Interactive CAPTCHAs

    [https://arxiv.org/abs/2611.25387](https://arxiv.org/abs/2611.25387)

    提出了TuringSpace——一个包含125个需通过点击和拖动完成的交互式验证码合成网络环境，用于评估多模态大模型在实际行动中落实空间智能的能力，人类解决率达94.1%，而最强模型仍失败25.9%，暴露出当前模型在行动性空间推理上的显著差距。

    

    人类在视觉世界中行动的方式是：推断二维视图背后的三维布局，预测动作将如何改变它，并在看到结果后进行自我修正。多模态大语言模型（MLLM）也能做到这一点吗？验证码（CAPTCHA）提供了一个天然的测试场景，因为人类正是通过这一循环来解答验证码的：在屏幕上点击和拖动，并根据每次的结果进行调整。然而，现有的验证码基准大多已被解决，且很少涉及空间性；而空间类基准又很少需要实际行动。因此，我们提出了TuringSpace——一个包含125个交互式验证码的合成网络环境，这些验证码涵盖五种空间能力，必须通过屏幕上的点击和拖动来完成。每个任务都是一个程序，可以借助校验器和参考答案生成新的实例，因此这些任务也构成了一个可扩展的训练场。人类能解决94.1%的任务，而表现最好的模型GPT-6 Astra仍有25.9%的任务失败，最好的开源权重模型Kimi K3则失败了68.0%的任务。

    arXiv:2611.25387v1 Announce Type: new  Abstract: Humans act in the visual world by inferring the 3D layout behind a 2D view, predicting how an action changes it, and correcting themselves after seeing the result. Can multimodal large language models (MLLMs) do the same? CAPTCHAs offer a natural test, since humans solve them through this very loop, clicking and dragging on the screen and adjusting after each result. Yet existing CAPTCHA benchmarks are largely solved and rarely spatial, while spatial benchmarks rarely require action. We therefore present TuringSpace, a synthetic web environment of 125 interactive CAPTCHAs that span five spatial abilities and must be solved through clicks and drags on the screen. Every task is a program that generates new instances with a checker and an oracle, so the tasks also form a scalable training gym. Humans solve 94.1% of the tasks, whereas the best model, GPT-6 Astra, still fails 25.9% of them, and the best open-weight model, Kimi K3, fails 68.0%
    
[^156]: 漂移、判别与定向：面向高维动作空间的动作协调

    Drifting, Discriminating, and Directioning for Action Coordination in Higher-dimensional Action Space

    [https://arxiv.org/abs/2611.25385](https://arxiv.org/abs/2611.25385)

    本文提出一种基于漂移模型的轻量级策略方法，通过协同优化源分布、判别与定向的三步训练算法，有效解决强化学习中高维动作空间的动作协调问题。

    

    本文提出了一种轻量级策略模型，以缓解强化学习（RL）中处理高维动作空间的困难，并提供了一种有效训练该模型的算法。为此，我们专注于漂移模型，这类模型能够隐式表示复杂分布，并在单次前向传播中推断动作。然而，尽管漂移模型可以通过Q梯度的反向传播直接训练，但仅凭这一点无法期望其恰当地捕捉所需的动作协调。因此，本文构建了一个三步学习算法来促进动作协调。首先，将漂移模型的源分布作为对角策略模型进行协同优化，这使得漂移模型将对角分布无法表达的协调作为残差进行附加。其次，引入漂移前后的判别器模型，采用对抗性训练……

    arXiv:2611.25385v1 Announce Type: new  Abstract: This paper proposes a lightweight policy model to mitigate the difficulty of handling higher-dimensional action spaces in reinforcement learning (RL), alongside an algorithm to effectively train it. To address this, we focus on drifting models, which can implicitly represent complex distributions and infer actions in a single forward pass. However, while drifting models can be straightforwardly trained via backpropagation of the Q-gradient, this alone cannot be expected to appropriately capture the desired action coordination. Therefore, this paper constructs a three-step learning algorithm to encourage the action coordination. First, the source distribution of the drifting model is co-optimized as a diagonal policy model. This conditions the drifting model to attach the coordination, which cannot be expressed by a diagonal distribution, as residuals. Second, a discriminator model before and after drifting is introduced with adversarial 
    
[^157]: 具有假阳性错误控制的因果发现

    Causal Discovery with False Positive Error Control

    [https://arxiv.org/abs/2611.25376](https://arxiv.org/abs/2611.25376)

    提出m-保守PC算法，在线性高斯设定和极弱忠实性假设下实现了因果发现中错误发现率的有限样本控制，同时惩罚虚假邻接与错误边方向。

    

    因果发现方法产生关于因果关系的数据驱动假设，这些假设可能成为新的科学发现，或为下游干预决策提供依据。在此类场景中，假阳性的因果论断可能比假阴性造成更严重的后果。然而，鲜有方法针对因果发现中的假阳性错误控制，且现有的错误保证通常仅为渐近性的，并需要较强版本的忠实性假设。我们提出了 m-保守 PC 算法，并证明在线性高斯设定下，该算法在极弱版本的忠实性假设下即可控制错误发现率（FDR）。在我们的框架中，因果图的假阳性错误基于其所蕴含的条件独立性信息来定义，由此得到的度量能够同时惩罚虚假邻接关系和错误的边方向。参数 m 则量化了算法复杂度与保证强度之间的权衡。

    arXiv:2611.25376v1 Announce Type: cross  Abstract: Causal discovery methods produce data-driven hypotheses about causal relationships, which may become new scientific findings or inform decisions about downstream interventions. In such settings, false positive causal claims can be more consequential than false negatives. Yet few methods target false positive error control in causal discovery, and existing error guarantees are often asymptotic and require strong versions of the faithfulness assumption. We introduce the $m$-conservative PC algorithm and prove that it controls false discovery rate in the linear Gaussian setting under an extremely weak version of the faithfulness assumption. In our framework, false positive error for causal graphs is defined based on the conditional-independence information they entail, yielding a metric that penalizes both false adjacencies and incorrect edge orientations. The parameter $m$ quantifies a trade-off between algorithmic complexity and the str
    
[^158]: 标签噪声下二分类的基础理论

    Fundamentals for Binary Classification Under Label Noise

    [https://arxiv.org/abs/2611.25372](https://arxiv.org/abs/2611.25372)

    本章系统综述了标签噪声下二分类学习的基础理论，在 PAC 框架下形式化了随机分类噪声、Massart 噪声和恶意分类噪声三种模型，并阐明了经验风险最小化的适用条件以及样本复杂度与噪声率之间的关系。

    

    本章综述了标签噪声下二分类的基础性研究成果。标签噪声下的二分类是指在一部分类别标签不正确的数据集上训练模型，这会导致模型性能下降和误分类问题。我们区分了投毒攻击与噪声过程，并在可能近似正确（PAC）框架下形式化了随机分类噪声、Massart 噪声和恶意分类噪声这三种噪声模型，它们刻画了不同的对抗强度和数据生成场景。随后，我们重点介绍了一些核心思想与代表性结果。特别地，我们解释了为什么随机分类噪声能够保持假设之间的期望排序，从而支持基于经验风险最小化（ERM）原则的学习；样本复杂度如何依赖于噪声率；以及为什么学习算法通常需要对污染水平设定一个上界，还有如何成功地……（原文摘要在此处被截断）

    arXiv:2611.25372v1 Announce Type: cross  Abstract: This chapter surveys fundamental results for binary classification under label noise. Binary classification under label noise involves training models on datasets where class labels are partially incorrect, leading to degraded performance and misclassification. We distinguish poisoning attacks from noise processes and formalize random classification noise, Massart noise, and malicious classification noise within the probably approximately correct (PAC) framework. These models capture different adversarial strengths and data-generation realities. We then highlight some core ideas and representative results. In particular, we explain why random classification noise preserves the expected ordering of hypotheses and thus supports learning using the empirical risk minimization (ERM) principle, how sample complexity depends on the noise rate, and why learning algorithms often require an upper bound on the corruption level, as well as how suc
    
[^159]: 类别不平衡条件下面向ICAO人脸图像合规检测的类别条件化保形分流方法

    Class-Conditional Conformal Triage for ICAO Face-Image Compliance under Class Imbalance

    [https://arxiv.org/abs/2611.25342](https://arxiv.org/abs/2611.25342)

    该研究提出将标签条件化（Mondrian）保形预测作为ICAO人脸图像合规分析的分流层，在不重新训练模型的情况下解决了类别不平衡导致的违规判定覆盖不足问题，将各准则的违规覆盖率从边际方法的0.37提升至0.89-0.92。

    

    自动化的ICAO/ISO人脸图像合规分析器可提供准则级别的通过/不通过判定，但在决定接受、拒绝或重拍照片时，这些判定并未量化不确定性。我们研究了将保形预测作为BioGaze的分流层，在合成TONO基准上对14项合规准则进行了评估。在不重新训练分析器的前提下，我们为每项准则拟合逻辑回归分类器，并使用受试者不重叠的数据划分，将边际保形预测与标签条件化（Mondrian）保形预测及传统弃权规则进行比较。边际保形预测实现了约0.90的整体覆盖率，但对违规情况存在明显的覆盖不足，某一准则的十分区平均覆盖率仅为0.37。而相应的Mondrian方法在各准则上的平均覆盖率范围为0.89至0.92。然而，准则级别假阳性的大部分降低来自逻辑回归层（从0.145降至0.048），而经过调优的弃权规则实现了……（原文摘要到此截断）

    arXiv:2611.25342v1 Announce Type: new  Abstract: Automated ICAO/ISO face-image compliance analysers provide criterion-level pass/fail decisions, yet these verdicts do not quantify uncertainty when deciding whether to accept, reject, or retake a photograph. We investigate conformal prediction as a triage layer for BioGaze, evaluating 14 compliance criteria on the synthetic TONO benchmark. Without retraining the analyser, we fit per-criterion logistic classifiers and compare marginal and label-conditional (Mondrian) conformal prediction with conventional abstention rules using subject-disjoint splits. Marginal conformal prediction achieved approximately 0.90 overall coverage but substantially under-covered violations, with a ten-partition mean of 0.37 for one criterion. The corresponding Mondrian means ranged from 0.89 to 0.92 across criteria. However, most of the reduction in criterion-level false positives came from the logistic layer (0.145 to 0.048), while tuned abstention rules achi
    
[^160]: 时隐时现：一个检测与降低代理能力的框架

    Now You See Me Now You Don't: A Framework To Detect and Reduce Proxy Capacity

    [https://arxiv.org/abs/2611.25327](https://arxiv.org/abs/2611.25327)

    该论文针对受监管领域（如贷款）中差别性对待的公平性问题，提出了一个可辩护的框架，用于大规模检测并降低机器学习模型中的代理变量能力。

    

    在诸如贷款等受监管的领域中，两个最突出的公平性概念是差别性影响和差别性对待：差别性影响通常由不同人口群体之间不利结果发生率的差异来定义，而差别性对待通常由使用人口统计信息或其代理变量来做出决策来定义。目前已有成熟的方法来定量检测和缓解差别性影响；然而，针对差别性对待开展同样工作的研究还比较少。从从业者的角度来看，如何表述和操作化条件独立性检验以识别代理变量，是一个细微且尚未被充分探索的问题。随着机器学习系统在受监管领域的应用范围不断扩大，从业者需要一个可辩护的框架来大规模识别代理变量。当前的差别性对待框架通常依赖于定量与……（摘要在此处被截断）

    arXiv:2611.25327v1 Announce Type: new  Abstract: In regulated domains such as lending, the two most salient fairness concepts are disparate impact and disparate treatment: disparate impact is generally defined by differential rates of adverse outcomes between demographic groups, while disparate treatment is generally defined by the use of demographic information or proxies to issue decisions. There already exist established methods to quantitatively detect and mitigate disparate impact; however, there has been less research on doing the same for disparate treatment. From a practitioner's perspective, the manner in which a conditional independence test is formulated and operationalized to identify proxies is nuanced and underexplored. As Machine Learning (ML) systems increase in scope and application in regulated domains, practitioners need a defensible framework to identify proxies at scale. Current disparate treatment frameworks typically rely upon some combination of quantitative and
    
[^161]: RE-CAM：面向定量相似性搜索的读出增强型内容寻址存储器

    RE-CAM: Readout-Enhanced Content-Addressable Memory for Quantitative Similarity Search

    [https://arxiv.org/abs/2611.25326](https://arxiv.org/abs/2611.25326)

    RE-CAM提出了一个与存储器技术和应用无关的读出增强框架，通过将感知时间戳作为关键设计变量，使CAM阵列在单次搜索中即可近似线性和非线性函数，将传统的二进制匹配读出升级为定量相似性搜索。

    

    相似性搜索对许多应用至关重要，但在传统架构上执行时会产生大量的数据移动。内容寻址存储器（CAM）是一种有前景的硬件载体，因为它可以直接在CAM阵列内进行相似性搜索，最大限度地减少数据移动，同时提供低延迟和高能效。然而，传统的CAM阵列读出只能产生二进制的匹配结果，丢弃了可以进一步利用的定量相似性信息。提取这些信息需要解读CAM搜索过程中不断演变的阵列输出电压，这使得感知时间戳与参考电压和解码参数一起成为关键的设计变量。在这项工作中，我们提出了RE-CAM，一个与存储器技术和应用无关的CAM读出增强框架，使CAM阵列读出能够通过一次CAM搜索近似线性和非线性函数。

    arXiv:2611.25326v1 Announce Type: cross  Abstract: Similarity search is essential to many applications but incurs massive data movement when executed on conventional architectures. Content-addressable memory (CAM) is a promising hardware substrate, as it allows similarity search directly in the CAM array, minimizing data movement and offering both low latency and high energy efficiency. However, conventional CAM array readout produces only binary match results, discarding quantitative similarity information that could be further exploited. Extracting this information requires interpreting the CAM array output voltage that evolves during a CAM search, making the sensing timestamp a crucial design variable alongside the reference voltage and decoding parameter. In this work, we propose RE-CAM, a readout enhancement framework for CAM that is agnostic to memory technology and application, enabling CAM array readout to approximate linear and nonlinear functions with one CAM search. By reusi
    
[^162]: 通过残差学习分配改进多阶段任务的模仿策略

    Improving Imitation Policies for Multi-Phase Tasks through Residual Learning Allocation

    [https://arxiv.org/abs/2611.25320](https://arxiv.org/abs/2611.25320)

    该论文提出“完成增益”概念，将其分解为阶段到达概率与条件完成改进，据此指导残差强化学习在多阶段任务中纠正哪些动作组件以及如何在各阶段间分配有限的训练预算。

    

    残差强化学习通过有限的环境交互预算来纠正基于模仿学习训练的基础策略。在包含多个阶段的任务中，该预算必须分配到各个阶段之中，而基础策略失败所在的阶段并不一定是对其进行纠正最能提升任务完成度的阶段。我们通过“完成增益”这一概念将该区别形式化：即在某一个阶段内，当状态反馈控制器替代基础策略动作中的选定组件时，整个任务完成概率的变化。我们将完成增益分解为到达该阶段的概率以及从该阶段开始时的条件完成改进，并通过配对的回合数据进行估计。正的估计值决定了残差策略应纠正哪些动作组件，以及训练回合应如何在不同阶段之间分配。在随机点质量对接实验中，大多数失败发生在……

    arXiv:2611.25320v1 Announce Type: new  Abstract: Residual reinforcement learning corrects an imitation-trained base policy using a limited budget of environment interaction. In a task with several phases, this budget must be divided among the phases, and the phase in which the base policy fails is not necessarily the phase in which a correction most improves task completion. We formalize this distinction through the completion gain: the change in full-task completion probability when, during a single phase, a state-feedback controller replaces selected components of the base policy's action. We decompose the completion gain into the probability of reaching the phase and the conditional improvement in completion from the start of the phase, and we estimate it from paired episodes. The positive estimates determine which action components the residual policy corrects and how training episodes are distributed across phases. In stochastic point-mass docking experiments, most failures occurr
    
[^163]: 已部署的脑机接口解码器何时应该自适应？面向冻结EEG主干的资源定价阶梯

    When Should Deployed BCI Decoders Adapt? A Resource-Priced Ladder for Frozen EEG Trunks

    [https://arxiv.org/abs/2611.25315](https://arxiv.org/abs/2611.25315)

    该论文针对冻结的EEG编码器提出一个由情境感知门控与效用定价（准确率、拟合时间和标注成本）驱动的资源阶梯自适应控制层，通过闭式贝叶斯头部更新，证明已部署BCI解码器的自适应应当被门控并设定上限，而非始终开启或盲目微调。

    

    arXiv:2611.25315v1 公告类型：交叉列表 摘要：已部署的运动想象（MI）解码器何时应该投入自适应资源——而不仅仅是在分布偏移下如何自适应？我们研究了冻结编码器之外的一个轻量控制层：一个资源阶梯（出厂模型 -> 贝叶斯头 -> 仿射映射，并以完整微调作为最后的逃生通道），由情境感知门控（regime-aware gating）管理，其效用函数对准确率、拟合时间和标注成本进行定价。该头部是共轭意义上的贝叶斯：在嵌入空间中对类条件高斯分布采用正态-逆威沙特（Normal-Inverse-Wishart）先验，随着标注试验数据的到来以闭式形式更新（无需梯度计算）。完整的估计器、常数以及再分析代码见配套论文和本提交的辅助文件包。核心论点：自适应应当被门控并设定上限，而不是始终开启，也不是盲目微调。证据：（i）效用最大化门控在已经携带会话漂移的干净数据上会过度自适应；一个以干净数据为参照的门控可以恢复较低的误进入率（例如，将自适应率从0……（原文摘要在此处截断）

    arXiv:2611.25315v1 Announce Type: cross  Abstract: When should a deployed MI decoder spend adaptation resources-not only how to adapt under shift? We study a thin control layer outside a frozen encoder: a resource ladder (factory -> Bayesian head -> affine map, with full fine-tune as an escape hatch) governed by regime-aware gating under a utility that prices accuracy, fit time, and label cost. The head is Bayesian in the conjugate sense: a Normal-Inverse-Wishart prior over the class conditional Gaussians in embedding space, updated in closed form (no gradients) as labelled trials arrive. Full estimators, constants, and the reanalysis code are in the companion paper and the ancillary bundle of this submission. Thesis: adaptation should be gated and capped, not always-on and not blindly fine-tuned. Evidence: (i) utility-max gating over-adapts on clean data that already carries session drift; a clean-referenced gate restores low false-entry rates (e.g. dropping the adaptation rate from 0
    
[^164]: 终身调整成本约束下的规定贝叶斯效率

    Prescribed Bayesian Efficiency under Lifetime Adjustment Costs

    [https://arxiv.org/abs/2611.25311](https://arxiv.org/abs/2611.25311)

    该论文刻画了在终身调整成本约束下，贝叶斯学习者在时间范围 T=2^J 处同时实现累积期望超额损失 O(G(J)) 与有限期望终身费用的可达性准则，即跨动作距离尺度上带前缀约束的跟踪误差配额分配的有限性。

    

    贝叶斯学习者可能在进行后验更新的同时保留已发出的动作，在预测损失与调整费用之间进行权衡。我们假设存在总体曲率、边界控制、关于最优动作的期望后验不确定性的 1/n 阶下界，以及一个可观测的无偏梯度见证，其平方范数一致可积。样本损失可以是无界的或非凸的。累积期望超额损失衡量相对于当前预测贝叶斯动作的损失增加。对于满足 c(0)=0 的有界非负非递减费用 c，以及满足 G(J)=o(J) 的非递减剖面 G: ℕ → [1,∞)，我们刻画了在时间范围 T=2^J 下累积期望超额损失 O(G(J)) 与有限期望终身费用的联合可达性。这里 J 是对数时间的索引。判据是跨动作距离尺度上带前缀约束的跟踪误差配额分配的有限性，等价地……（原文在此处截断）

    arXiv:2611.25311v1 Announce Type: new  Abstract: A Bayesian learner may retain an issued action while updating its posterior, trading predictive loss against adjustment charges. We assume population curvature, boundary control, an order-$1/n$ lower bound on expected posterior uncertainty about the optimal action, and an observable unbiased gradient witness whose squared norms are uniformly integrable. Sample losses may be unbounded or nonconvex. Cumulative expected excess measures the loss increase relative to current predictive Bayes actions. For bounded nonnegative nondecreasing charges $c$ with $c(0)=0$ and nondecreasing profiles $G: \mathbb{N} \to [1,\infty)$ with $G(J)=o(J)$, we characterize joint attainability of cumulative expected excess $O(G(J))$ at horizons $T=2^J$ and finite expected lifetime charge. Here $J$ indexes logarithmic time. The criterion is finiteness of a prefix-constrained allocation of tracking-error allowances across action-distance scales, equivalently its de
    
[^165]: VoxForge：跨MRI、CT和PET的密集3D特征统一基础模型

    VoxForge: A Unified Foundation Model for Dense 3D Features Across MRI, CT and PET

    [https://arxiv.org/abs/2611.25309](https://arxiv.org/abs/2611.25309)

    提出VoxForge，一个融合3D架构、自蒸馏与跨模态课程学习的统一医学基础模型编码器，在完全冻结状态下即可在MRI和CT的分割、分类、配准任务上媲美通用与专用模型，并能泛化到训练中从未见过的PET模态。

    

    统一的3D医学基础模型有望提供可复用的冻结特征，能够像专用模型一样出色地完成分割、分类和配准等多种任务，覆盖MRI、CT和PET。目前的通用模型仍需要为局部任务和全局任务分别配置骨干网络；专用模型则局限于单一模态，且人们担心跨模态训练会弱化精细解剖结构；此外，这两类模型通常都依赖端到端微调才能保持竞争力。我们提出了VoxForge——一个单一编码器，它整合了3D架构、自蒸馏和跨模态课程学习的最新进展，从采用宽松许可协议的MRI和CT体数据中学习可迁移的特征。我们严格在冻结状态下对其进行评估——这是可复用基础模型必须通过的测试——将其与最先进的通用模型和专用模型在MRI和CT上的分割、分类和配准任务中进行比较，并在其从未见过的PET数据上加以测试。VoxForge

    arXiv:2611.25309v1 Announce Type: cross  Abstract: A unified 3D medical foundation model promises reusable, frozen features that serve a wide variety of tasks such as segmentation, classification and registration on MRI, CT and PET as well as specialist models do. Generalist models still need separate backbones for localized and global tasks, specialist models are confined to a single modality, with the concern that spanning modalities under-emphasizes fine anatomy, and both typically rely on end-to-end fine-tuning to be competitive. We present VoxForge, a single encoder that integrates recent advances in 3D architecture, self-distillation and cross-modal curricula to learn transferable features from permissively licensed MRI and CT volumes. We evaluate it strictly frozen, the test a reusable foundation model must pass, against state-of-the-art generalist and specialist models on segmentation, classification and registration across MRI and CT, and on PET, which it never sees. VoxForge 
    
[^166]: SplitReAlign：面向分割联邦遗忘的分区感知表征重对齐

    SplitReAlign: Partition-Aware Representation Realignment for Split Federated Unlearning

    [https://arxiv.org/abs/2611.25301](https://arxiv.org/abs/2611.25301)

    SplitReAlign提出了一种分区感知的表征重对齐框架，通过共享语义锚点协调的端点判别子空间，在分割联邦遗忘中实现类级知识的选择性遗忘并保护保留效用，同时适配客户端与服务器的非对称更新约束。

    

    分割联邦学习（SFL）将神经网络分布在客户端与服务器之间，使类特定知识在双方的表征中相互耦合。这种耦合使分割联邦遗忘（SFU）变得复杂：在异构表征空间和非对称更新约束下，类级遗忘必须在移除目标知识的同时保留其余效用。我们提出SplitReAlign，一个面向类级分割联邦遗忘的分区感知表征重对齐框架。该方法通过共享语义锚点协调构建端点特定的判别子空间，选择性地重对齐待遗忘的表征，同时惩罚活动子空间之外的变化。为适应非对称更新约束，SplitReAlign在仅客户端更新中利用冻结服务器传递的梯度，在仅服务器引导中使用固定的客户端特征统计量，而联合更新则同时优化两端分区。在CIFAR-10和CI……（实验部分截断）上的实验验证了方法的有效性。

    arXiv:2611.25301v1 Announce Type: new  Abstract: Split federated learning (SFL) distributes neural networks across clients and a server, coupling class-specific knowledge across their representations. This coupling complicates split federated unlearning (SFU), where class-wise forgetting must remove target knowledge while preserving retained utility under heterogeneous representation spaces and asymmetric update constraints. We propose SplitReAlign, a partition-aware representation realignment framework for class-wise SFU. It constructs endpoint-specific discriminative subspaces coordinated through a shared semantic anchor, selectively realigning forgotten representations while penalizing changes outside the active subspaces. To accommodate asymmetric update constraints, SplitReAlign leverages gradients through frozen servers for Client-only updates and fixed client-feature statistics for Server-only guidance, while Joint updates optimize both partitions. Experiments on CIFAR-10 and CI
    
[^167]: 基于谱优化器的汽车空气动力学代理模型联邦学习

    Federated Learning of Car Aerodynamics Surrogates with Spectral Optimizers

    [https://arxiv.org/abs/2611.25300](https://arxiv.org/abs/2611.25300)

    本研究在跨车身类型的联邦学习设置下比较了多种优化器，发现谱优化器（尤其是FedMuon和作者提出的FedMARS-M）能训练出性能与中心化训练相差2%以内的汽车空气动力学代理模型。

    

    神经网络代理模型可以在几秒钟内预测汽车所受的空气动力，而CFD模拟则需要数小时，但训练这类代理模型需要对不同汽车设计进行数千次CFD模拟，而汽车制造商将这些设计数据作为专有信息保密。联邦学习允许多家制造商在不交换数据的前提下共同训练一个共享的代理模型。我们研究了哪种联邦优化器能使这样的代理模型最接近在汇总数据上训练得到的模型。在DrivAerNet++数据集上，我们在六个客户端上训练Transolver代理模型，每个客户端仅持有单一车身类型的汽车，并将三种对每个权重矩阵的更新进行正交化的谱优化器（FedMuon以及我们提出的两种方法FedMARS-M和FedOLion）与三种非谱优化器（FedAvg、SCAFFOLD-Adam和FedLion）进行比较，所有方法均采用相同的调优协议。FedMuon和FedMARS-M在每轮仅使用2到4个本地epoch的情况下，其阻力误差达到与使用Muon的中心化训练相差2%以内。非谱方法……

    arXiv:2611.25300v1 Announce Type: new  Abstract: Neural surrogates predict the aerodynamic forces on a car in seconds, compared with hours for a CFD simulation, but training them requires thousands of CFD simulations of different car designs, which car manufacturers keep proprietary. Federated learning allows several manufacturers to train a shared surrogate without exchanging data. We study which federated optimizer brings such a surrogate closest to one trained on pooled data. On DrivAerNet++, we train a Transolver surrogate across six clients that each hold cars of a single body type, and we compare three spectral optimizers, which orthogonalize the update of each weight matrix (FedMuon and two methods we propose, FedMARS-M and FedOLion), with three non-spectral ones (FedAvg, SCAFFOLD-Adam and FedLion), all tuned with the same protocol. FedMuon and FedMARS-M reach a drag error within 2% of centralized training with Muon, using two to four local epochs per round. The non-spectral met
    
[^168]: 跨问题可靠强化学习

    Cross-Question Reliable Reinforcement Learning

    [https://arxiv.org/abs/2611.25292](https://arxiv.org/abs/2611.25292)

    本文提出跨问题可靠强化学习，指出现有逐题独立地将正确答案标为置信度1、错误答案标为0的训练方式在实践中不可实现且对非零风险容忍度下的选择性预测有害，主张跨问题地学习可靠的置信度估计以支持更好的弃权与转交决策。

    

    arXiv:2611.25292v1 公告类型：新论文 摘要：在自身能力极限附近运行的AI系统，可以通过弃权或将其没有把握的问题转交给人类或更强的模型，来在最小化错误风险的同时最大化效用。最近，大语言模型（LLMs）被后训练以估计其对自身答案的置信度。为实现这一目标，现有方法基于每个问答对单独的正确性信号来强化置信度分数，其奖励机制鼓励对正确答案赋予置信度1，对错误答案赋予置信度0。然而，估计置信度分数的价值在于能够只选出最有把握的答案，从而在适应用户风险容忍度的同时最大化已解决问题的比例。对正确答案赋予置信度1、对错误答案赋予置信度0虽然在理论上是最佳的，但在实践中无法实现，并且对于具有非零风险容忍度的选择性预测是有害的：置信度……（摘要原文在此处截断）

    arXiv:2611.25292v1 Announce Type: new  Abstract: AI systems operating near the limits of their capabilities can maximize utility while minimizing the risk of error by abstaining or deferring questions they are not confident about to a human or a stronger model. Recently, large language models (LLMs) have been post-trained to estimate confidence in their own answers. To achieve this, methods reinforce confidence scores based on the correctness signal to each question-answer pair individually, with rewards that encourage assigning confidence 1 to correct answers and 0 to incorrect answers. However, estimating confidence scores is useful in that it enables selecting only the most confident answers, to maximize the share of solved questions while accommodating the user's risk tolerance. Assigning confidence 1 to correct answers and 0 to incorrect ones is theoretically optimal, but unattainable in practice and detrimental for selective prediction with a non-zero risk tolerance: confidences 
    
[^169]: 大语言模型已经是具备置信度感知的零样本分类器

    LLMs Are Already Confidence-Aware Zero-Shot Classifiers

    [https://arxiv.org/abs/2611.25289](https://arxiv.org/abs/2611.25289)

    研究通过系统评估表明，大语言模型作为零样本分类器已具备置信度感知能力，基于logit的白盒评分方法在判别分类置信度可靠性上显著优于黑盒方法。

    

    零样本大语言模型越来越多地被用作文本分类器，但它们的标签只有在知道能在多大程度上信任它们时才有用。我们比较了黑盒方法（言语化置信度和采样）与读取token概率的白盒方法（P(True)以及使用受限生成的多种logit评分方法）。我们在五个包含3到151个标签的分类数据集上评估了11个开放权重大语言模型（1B-32B参数），测量分类性能（准确率、F1-macro）、判别能力（AUROC）和校准（ECE和Brier分数）。置信度方法的选择很少显著影响准确率，但基于logit的评分方法比黑盒方法（AUROC 0.57-0.66）带来了更好的判别能力（AUROC 0.74-0.77）。用单token标识符替换标签可以保持判别能力，但平均损失3-6个准确率点。我们还引入了仅决策评分方法，该方法（摘要在此处截断）

    arXiv:2611.25289v1 Announce Type: cross  Abstract: Zero-shot large language models (LLMs) are increasingly used as text classifiers, but their labels are only useful when we know to what extent we can trust them. We compare black-box methods (verbalised confidence and sampling) with white-box methods that read token probabilities (P(True), and multiple logit-scoring methods using constrained generation). We evaluate eleven open-weight LLMs (1B-32B parameters) on five classification datasets with 3 to 151 labels, measuring classification performance (accuracy, F1-macro), discrimination (AUROC), and calibration (ECE and Brier score). The choice of confidence method rarely impacts accuracy significantly, but logit-based scoring methods lead to much better discrimination (AUROC 0.74-0.77) than black-box methods (AUROC 0.57-0.66). Replacing labels with single-token identifiers preserves discrimination but costs 3-6 accuracy points on average. We also introduce decision-only scoring, which o
    
[^170]: 面向大规模推荐模型的智能嵌入剪枝

    Intelligent Embedding Pruning for Large-Scale Recommendation Models

    [https://arxiv.org/abs/2611.25288](https://arxiv.org/abs/2611.25288)

    该论文提出利用特征重要性来指导训练中嵌入剪枝的各表剪枝比例（FI-ITEP），在三个大规模推荐模型上实现 20%–40% 的内存节省并保持或提升准确率，同时显著减少训练所需的 GPU 数量。

    

    现代推荐系统面临巨大的内存压力，这主要源于将高基数类别特征映射为稠密向量的稀疏嵌入层。训练中嵌入剪枝（ITEP）在训练过程中移除不那么重要的嵌入行，从而带来端到端的内存节省，但由于各表的剪枝比例是由基于规模的启发式方法设定的，有时会降低模型准确率。我们利用特征重要性（FI）来指导这些剪枝比例（FI-ITEP），对与较不重要特征相关联的表分配更高的剪枝比例。在三个大规模推荐模型上，FI-ITEP 相比基于百分位数的 ITEP 实现了 20%–40% 的总内存减少，同时保持或提升了准确率。由于特征重要性信号的计算所需资源极少，这些内存节省转化为具体的系统收益：相同的模型可以在更少的 GPU 训练节点上完成训练（例如从 112 个减少到 80 个），且每个 GPU 的吞吐量更高，节省了 1000+ GPU 资源。

    arXiv:2611.25288v1 Announce Type: cross  Abstract: Modern recommendation systems face substantial memory pressure, largely from the sparse embedding layer that maps high-cardinality categorical features into dense vectors. In-training embedding pruning (ITEP) removes less important embedding rows during training, yielding end-to-end memory savings but sometimes degrading accuracy because per-table pruning ratios are set by size-based heuristics. We use feature importance (FI) to guide these ratios (FI-ITEP), assigning higher pruning to tables tied to less important features. Across three large-scale recommendation models, FI-ITEP achieves 20--40% total memory reduction while matching or improving accuracy relative to percentile-based ITEP. Because the feature-importance signal requires minimal resources to compute, these memory savings translate into concrete systems benefits: the same models train on fewer GPU trainers (e.g., 112 to 80) with higher per-GPU throughput, saving 1,000+ GP
    
[^171]: DS3M：在现代硬件上用于非线性系统辨识的深度结构化状态空间模型

    DS3M: Deep structured state space models for nonlinear system identification on modern hardware

    [https://arxiv.org/abs/2611.25274](https://arxiv.org/abs/2611.25274)

    本文提出深度结构化状态空间模型DS3M，通过以并联和串联方式组合紧凑的非线性动力学组件，并利用Triton实现的自定义GPU内核降低顺序评估开销，显著提升了非线性系统辨识中神经状态空间模型的训练速度。

    

    神经状态空间辨识从测量到的输入和输出中学习动力学模型。SUBNET 将这一任务与一个编码器相结合，该编码器从较短的测量历史中估计初始状态。其连续时间版本在数值积分过程中需要多次连续的神经网络评估，导致训练速度缓慢，这是实际应用中的一个主要限制。我们提出了深度结构化状态空间模型（DS3M），它以并联和串联的方式组合紧凑的非线性动力学组件。使用 Triton 实现的自定义 GPU 内核降低了其顺序评估的启动和内存开销。我们解释了这一构建方法，描述了一种端口哈密顿变体，并给出了关于准确性和训练时间的多种基准比较。

    arXiv:2611.25274v1 Announce Type: new  Abstract: Neural state-space identification learns dynamical models from measured inputs and outputs. SUBNET combines this task with an encoder that estimates initial states from short measurement histories. Its continuous-time version requires many successive neural-network evaluations during numerical integration, resulting in slow training speed which is a major limitation for practical use cases. We propose deep structured state space models (DS3M), which combine compact nonlinear dynamical components in parallel and in series. Custom GPU kernels implemented with Triton reduce the launch and memory overhead of their sequential evaluation. We explain this construction, describe a port-Hamiltonian variant, and present various benchmark comparisons of accuracy and training time.
    
[^172]: Scone：面向图着色的对称性感知神经组合优化

    Scone: Symmetry-aware Neural Combinatorial Optimization for Graph Coloring

    [https://arxiv.org/abs/2611.25273](https://arxiv.org/abs/2611.25273)

    本文提出Scone，一种基于颜色等变图神经网络的对称性感知神经组合优化方法，通过利用颜色置换对称性缩小状态空间并在必要时打破对称性，从而高效构建高质量的图着色方案。

    

    图着色是指为节点分配颜色使得相邻节点颜色互不相同的问题，是一个基础的组合优化问题。然而，在给定固定调色板的情况下，找到不含有单色边的节点着色方案是NP难的。这促使人们研究基于学习的方法，这类方法能够学习利用实例间反复出现的结构，从而高效地生成高质量着色方案。至关重要的是，一个着色方案的有效性对颜色置换具有不变性，我们证明利用这种对称性可以缩小模型学习策略时所需的有效状态空间的规模。与此同时，着色算法还必须能够打破某些对称性，例如为属于同一自同构轨道的两个相邻节点分配不同的颜色。本文提出了Scone，一个使用颜色等变图神经网络顺序构建着色方案的模型，能够……

    arXiv:2611.25273v1 Announce Type: new  Abstract: Graph coloring, which entails assigning colors to nodes such that no adjacent nodes have the same color, is a fundamental combinatorial optimization problem. In general, however, finding a node coloring with no monochromatic edges given a fixed palette is NP-hard. This motivates learning-based methods that can learn to exploit recurring structure across instances to efficiently produce high-quality colorings. Crucially, the validity of a coloring is invariant to permutations of the colors, and we show that exploiting this symmetry reduces the size of the effective state space over which the model has to learn its policy. At the same time, the coloring algorithm must be able to break some symmetries, for example to assign different colors to two adjacent nodes belonging to the same automorphism orbit. In this paper, we introduce Scone, a model that uses a color-equivariant graph neural network to construct colorings sequentially, allowing
    
[^173]: 学习验证：面向自适应对抗检测的签名引导训练

    Learning to Verify: Signature-Guided Training for Adaptive Adversarial Detection

    [https://arxiv.org/abs/2611.25271](https://arxiv.org/abs/2611.25271)

    该论文提出一种双模型对抗检测方法：固定原始分类器并训练独立验证器，利用“激活签名”引导生成能同时误导分类器、挑战验证器且内部特征接近目标类别的高难度对抗样本进行训练，使验证器在面对自适应攻击（含未见过的攻击）时依然有效，且推理时无需激活签名。

    

    当攻击者明确试图绕过检测器时，对抗样本检测器往往会失效。我们提出一种双模型方法，即保持原始分类器固定不变，并训练一个独立的验证器来评估分类器的预测结果。在训练过程中，我们识别出神经网络内部激活中的模式，这些模式能够区分每个类别的真实样本与针对该类别的对抗样本。这些被称为“激活签名”的模式可指导生成具有挑战性的对抗样本，这类样本既能误导分类器、又能挑战验证器，且在内部特征上与目标类别相似。验证器通过知识蒸馏学习识别这些对抗样本，同时保持其在干净输入上的性能。在推理阶段，验证器无需激活签名即可评估分类器的预测结果。我们在公开数据集上针对多种自适应攻击评估了该方法，包括训练中未见过的攻击……

    arXiv:2611.25271v1 Announce Type: new  Abstract: Adversarial example detectors often fail when attackers explicitly attempt to bypass them. We propose a dual-model approach that keeps the original classifier fixed and trains a separate verifier to assess its predictions. During training, we identify patterns of internal neural network activations that distinguish genuine examples of each class from adversarial examples targeting that class. These patterns, called activation signatures, guide the generation of challenging adversarial examples that mislead the classifier, challenge the verifier, and internally resemble the target class. The verifier learns to recognize these examples while preserving its performance on clean inputs through knowledge distillation. At inference, the verifier evaluates the classifier's prediction without requiring activation signatures. We evaluate the approach on publicly available datasets against several adaptive attacks, including attacks unseen during 
    
[^174]: 工艺知识引导的潜在扩散模型用于多孔结构的生成式设计：以气溶胶喷射打印为例

    Process-Knowledge-Guided Latent Diffusion for Generative Design of Porous Architectures: A Case Study in Aerosol Jet Printing

    [https://arxiv.org/abs/2611.25264](https://arxiv.org/abs/2611.25264)

    该论文提出一种工艺知识引导的潜在扩散生成框架，将基于工艺模型的利用率评分与VQ-VAE潜在表示、拓扑U-Net和ControlNet分支相结合，使生成的多孔结构布局能够适配特定的气溶胶喷射打印工艺设置。

    

    印刷多孔布局将功能性材料与开放通道相互交织，而材料利用率最优的布局取决于工艺设置如何塑造沉积形态。生成模型能够快速提出布局方案，但若生成器在每种工艺设置下接收的样本并非来自同一分布，即使工艺相关评分对候选布局的排序存在差异，其结果也会受到影响。本文探讨了工艺知识能否引导潜在扩散生成器，在给定设计域内生成适合特定工艺设置的布局。我们构建了一个依赖工艺的利用率评分，其中工艺模型确定了印刷材料的效用随距通道距离而衰减的长度尺度。该生成器融合了向量量化变分自编码器（VQ-VAE）潜在表示、拓扑U-Net、ControlNet域分支，以及一个单独训练的、可根据噪声预测评分的回归器。

    arXiv:2611.25264v1 Announce Type: new  Abstract: Printed porous layouts interleave functional material with open channels, and the layout that uses the material best depends on how the process settings shape the deposit. Generative models propose layouts quickly, but a generator that does not receive the process setting samples from the same distribution at every setting, even when the process-dependent score ranks candidate layouts differently. This paper examines whether process knowledge can steer a latent diffusion generator toward layouts suited to a given process setting in a prescribed design domain. We formulate a process-dependent utilization score in which a process model sets the length over which the utility of printed material decays with distance from a channel. The generator combines a vector-quantized variational autoencoder (VQ-VAE) latent representation, a topology U-Net, a ControlNet domain branch, and a separately trained regressor that predicts the score from noise
    
[^175]: 当目标与评估者不一致时：评估事后人口框架偏差缓解的迁移性

    When Objectives and Evaluators Disagree: Evaluating Transfer for Post-hoc Demographic Framing Mitigation

    [https://arxiv.org/abs/2611.25258](https://arxiv.org/abs/2611.25258)

    该研究将无法访问源模型时通过事后重写缓解大语言模型人口框架偏差视为迁移问题，发现总是针对说服偏差指数（PBI）较低信息的固定策略迁移不一致，而在两侧提出编辑但最多接受一个的自适应策略在两个评估器家族八项比较中的七项显著更优，且主要得益于编辑了更多对而非引导方向选择。

    

    在源模型不可访问的情况下，事后重写被用于减少大语言模型生成文本中的人口群体差异。一次重写可能仅仅因为改善了自己的目标或满足了自己的选择器而显得成功。我们将其作为一个迁移问题进行研究，研究对象是660个性别对和1,980个年龄对比的人口条件化气候信息。重写器提出编辑，选择模型在满足约束条件的候选中进行挑选，而一个独立于两者的评估器对成对框架差异进行评分。总是针对说服偏差指数（PBI）较低的信息的固定策略，在不同维度和结果聚合方式之间的迁移表现不一致。一种在两侧都提出编辑但最多接受一个编辑的自适应策略，在应用于相同候选时，在两个评估器家族的八项比较中有七项显著优于该规则，其优势主要来自编辑了更多的信息对，而非引导编辑方向的选择。

    arXiv:2611.25258v1 Announce Type: cross  Abstract: Post-hoc rewriting is used to reduce demographic disparities in LLM-generated text when the source model is inaccessible. A rewrite can look successful merely because it improved its own target or satisfied its selector. We study this as a transfer problem on 660 gender pairs and 1,980 age contrasts of demographic-conditioned climate messages. A rewriter proposes edits, a selection model chooses among constraint-satisfying candidates, and an independent evaluator uninvolved in either rates pairwise framing disparity. A fixed policy that always targets the message with the lower Persuasion Bias Index (PBI) transfers inconsistently across axes and outcome aggregations. An adaptive policy that proposes edits on both sides but accepts at most one significantly improves on that rule, applied to the same candidates, in seven of eight comparisons across two evaluator families, mostly by editing more pairs, not by guided direction choice, at h
    
[^176]: 将通用性从模型转移到建模者：在部署式基准上用于PHM（故障预测与健康管理）的LLM智能体

    Moving Generality from Model to Modeler: LLM Agents for PHM on a Deployment-Style Benchmark

    [https://arxiv.org/abs/2611.25252](https://arxiv.org/abs/2611.25252)

    该论文提出部署式基准 NextUnit-Bench，证明基于LLM的智能体在PHM任务上可超越AutoML及已发表方法的忠实复现，主张通用性应从单一模型转移到能为每个新任务构建模型的智能体身上。

    

    基于大语言模型（LLM）构建的编程智能体如今帮助开发了许多作为通用解决方案发布的模型。那么，通用性是否应改为体现在为每个新任务构建模型的智能体之中？我们在故障预测与健康管理（PHM）领域考察了这一问题，因为在该领域中每个模型都必须在未见过的设备上工作。我们提出的部署式基准 NextUnit-Bench 涵盖 24 个公开数据集，划分为 58 个折，分别对设备单元或运行条件进行留出；每个智能体有两个小时时间交付一个可执行的预测器，随后在隐藏单元上进行测试。首先，前沿智能体的表现优于 AutoML——后者的得分不到最佳智能体的一半——也优于在四个数据集上对已发表方法的忠实实现，30 个实现中仅有 6 个击败了最强智能体自行设计的方案；在可复现的情况下，已发表的准确率依赖于论文自身的约定，而仅仅一个约定——对测试引擎的剩余寿命进行上限截断——就可能使报告的误差减半。

    arXiv:2611.25252v1 Announce Type: new  Abstract: Coding agents built on large language models (LLMs) now help develop many models published as general solutions. Should generality instead reside in the agent that builds a model for each new task? We examine this question in prognostics and health management (PHM), where every model must work on unseen equipment. Our deployment-style benchmark, NextUnit-Bench, covers 24 public datasets split into 58 folds that hold out units or operating conditions; each agent has two hours to deliver an executable predictor, then run on the hidden units. First, frontier agents outperform AutoML, which reaches less than half the best agent's score, and faithful implementations of published methods on four datasets, of which only 6 of 30 beat the strongest agent's own design; published accuracies, where reproducible, rest on the papers' own conventions, and a single convention, capping the remaining life of test engines, can halve the reported error. Sec
    
[^177]: 特征归因的稳定性区间：解释方差博弈的核心、核仁与谈判集

    Stability Bands for Feature Attribution: The Core, the Nucleolus and the Bargaining Set of Explained-Variance Games

    [https://arxiv.org/abs/2611.25238](https://arxiv.org/abs/2611.25238)

    该论文提出用解释方差博弈的核心、核仁与谈判集为每个特征刻画“稳定性区间”，以证明归因分配不会被特征联盟合理质疑，并证明线性模型下该博弈恰为2-可加，多重共线性正是联盟效应的根源。

    

    沙普利值归因是经济学中解释机器学习预测的默认方法，但它对每个特征只报告一个数字，并且无法说明一组联合贡献超过其所获份额的特征联盟是否可能对该分配提出异议。我们通过解释方差博弈的核心、核仁和谈判集来研究特征归因，其中一组特征的“价值”定义为这些特征所生成的模型预测的方差。对于每个特征，我们报告一个稳定性区间（其份额在核心中的取值范围，或在核心为空时在最小核心中的取值范围），以核仁作为显著参考点，并给出该分配不受理据充分的异议质疑的证明。三个结果支撑了这一方法：首先，对于线性模型，该博弈恰好是2-可加的，其成对哈萨尼红利为 $2\beta_i\beta_j\sigma_{ij}$，因此多重共线性正是产生联盟……

    arXiv:2611.25238v1 Announce Type: cross  Abstract: Shapley-value attributions are the default way of explaining machine-learning predictions in economics, but they report a single number per feature and say nothing about whether a coalition of features that jointly accounts for more than it receives could contest that allocation. We study feature attribution through the core, the nucleolus and the bargaining set of an explained-variance game, in which the worth of a set of features is the variance of the model's predictions that those features generate. For every feature we report a stability band (its range of shares over the core, or over the least core when the core is empty), with the nucleolus as distinguished point and a certificate that the allocation admits no justified objection. Three results organise the approach. First, for linear models the game is exactly 2-additive, with pairwise Harsanyi dividends $2\beta_i\beta_j\sigma_{ij}$, so multicollinearity is what creates coalit
    
[^178]: 有些记忆必须消逝：用户记忆在模型参数中的持久性与衰减

    Some Memories Must Fade: Persistence and Decay of User Memories in Model Parameters

    [https://arxiv.org/abs/2611.25235](https://arxiv.org/abs/2611.25235)

    该论文提出了一种参数化记忆管理系统，通过固定容量的LoRA适配器作为记忆基底，配合学习策略来管理知识的完整生命周期——保留稳定事实、强化重复观察、更新变化的信念，并让短暂信息自然消退。

    

    参数化记忆使大型语言模型能够将知识直接内化到权重中，为基于检索的记忆（RAG）提供了一种潜在强大的替代方案。然而，现有的参数化记忆方法难以部署，因为它们在很大程度上沿用了持续学习的目标，即无限期地保留信息。这对于记忆系统而言既不可持续也不充分，因为记忆系统必须管理知识的整个生命周期：保留稳定的事实、强化反复出现的观察、更新已经改变的信念，并允许短暂的信息逐渐消退。我们提出了一个参数化记忆管理系统，由记忆基底（负责信息的存储与遗忘）和学习策略（决定记住什么、以多深的程度记住、以及记住多久）组成。我们的基底是一个固定容量的LoRA适配器，由k-hot秩组合构成，能够在不产生灾难性干扰的情况下存储事实……

    arXiv:2611.25235v1 Announce Type: cross  Abstract: Parametric memory allows Large Language Models to internalize knowledge directly into their weights, offering a potentially powerful alternative to retrieval-based memory (RAG). However, existing approaches to parametric memory are difficult to deploy as they largely mirror the Continual Learning (CL) objective of retaining information indefinitely. This is neither sustainable nor sufficient for a memory system, which must govern knowledge across its lifecycle by retaining stable facts, reinforcing recurring observations, updating changed beliefs, and allowing transient information to fade. We propose a system for parametric memory management consisting of a memory substrate (handling information storage and forgetting) and a learned policy (what to remember, how deeply, and for how long). Our substrate is a fixed-capacity LoRA adapter that consists of k-hot rank combinations which store facts without catastrophic interference and main
    
[^179]: 处理缺失协变量的统计学、机器学习与深度学习方法：在阿尔茨海默病研究中的实证评估

    Statistical, Machine Learning and Deep Learning Methods for Missing Covariates: An Empirical Evaluation with Applications to Alzheimer's Disease Research

    [https://arxiv.org/abs/2611.25219](https://arxiv.org/abs/2611.25219)

    本研究对八种处理缺失协变量的统计、机器学习和深度学习方法进行了系统性实证评估，在阿尔茨海默病研究背景下比较了它们在不同缺失机制下的表现。

    

    缺失数据在生物医学研究中十分常见，尤其是当生物标志物成本高昂或需要侵入性操作才能获取时，例如阿尔茨海默病研究中的脑脊液指标。尽管目前存在从传统统计方法到现代机器学习（ML）和深度学习（DL）算法的多种缺失数据处理方法，但它们在不同缺失机制下的比较性能仍缺乏充分了解。在本研究中，我们对八种处理缺失协变量的方法进行了实证评估，包括完整病例分析、多重插补、基于机器学习的技术，以及多层感知机、变分自编码器和生成对抗网络等深度学习框架。我们总结了这些方法的概念基础、实际应用考量、优势与局限性。通过大量的模拟实验，我们评估了各方法的估计偏差、效率、标准误准确性等性能表现。

    arXiv:2611.25219v1 Announce Type: cross  Abstract: Missing data are common in biomedical research, particularly when biomarkers are costly or require invasive procedures to obtain, such as cerebrospinal fluid measures in Alzheimer's disease. Although a wide range of missing data methods exists, from traditional statistical approaches to modern machine learning (ML) and deep learning (DL) algorithms, their comparative performance under different missingness mechanisms remains insufficiently understood. In this study, we provide an empirical evaluation of eight methods for handling missing covariates, including complete-case analysis, multiple imputation, ML-based techniques, and DL frameworks such as multilayer perceptrons, variational autoencoders, and generative adversarial networks. We summarize their conceptual foundations, practical implementation considerations, strengths, and limitations. Through extensive simulations, we assess estimation bias, efficiency, standard error accurac
    
[^180]: 使用冻结语言模型学习读取事件以实现精确状态跟踪

    Learning to Read Events for Exact State Tracking with Frozen Language Models

    [https://arxiv.org/abs/2611.25211](https://arxiv.org/abs/2611.25211)

    该论文提出在冻结语言模型上训练一个轻量级选择器来读取事件流中的每个状态转换，并通过外部整数寄存器按序组合更新，从而实现精确状态跟踪，理论上能够解决固定深度transformer在TC⁰ ≠ NC¹猜想下无法解决的任务。

    

    长事件流需要识别每一个变化并按顺序应用这些变化。我们在冻结的语言模型上训练一个选择器来读取每个状态转换。在仅有API访问权限的条件下，我们让模型列出这些转换，然后由外部整数寄存器按顺序将它们组合起来。置信度规则可以标记不确定的更新或保留多个候选。我们研究了五个元素置换的乘积这一任务，在TC⁰ ≠ NC¹的标准猜想下，固定深度、对数精度的transformer无法在所有输入长度上解决该任务。在两事件窗口上，一个27B模型在没有可见推理的情况下100个问题仅答对3个，而在生成中间步骤时100个问题答对96个。一个带有训练选择器的冻结0.8B模型，在299个窗口（每个窗口包含43,687个事件）中全部正确读取了每个事件。同一套系统在另一台新机器上的299个新窗口中通过了298个，未能达到单侧95%置信下界0.99的目标。

    arXiv:2611.25211v1 Announce Type: cross  Abstract: Long event streams require identifying each change and applying the changes in order. We train a selector on a frozen language model to read each transition. With API-only access, we ask the model to list the transitions. An external integer register composes them in order. A confidence rule can flag uncertain updates or retain several candidates. We study products of permutations of five items, a task that fixed-depth, log-precision transformers cannot solve on all input lengths under the standard conjecture that $\mathrm{TC}^0 \ne \mathrm{NC}^1$. On two-event windows, a 27B model answered 3 of 100 correctly without visible reasoning and 96 of 100 when it generated intermediate steps. A frozen 0.8B model with a trained selector read every event in 299 of 299 windows of 43,687 events each. The same package passed 298 of 299 new windows on a fresh machine, missing the target of 0.99 for the one-sided 95% lower confidence bound. A separa
    
[^181]: φTD：基于特征函数的分布式强化学习

    {\phi}TD: Distributional Reinforcement Learning using Characteristic Functions

    [https://arxiv.org/abs/2611.25208](https://arxiv.org/abs/2611.25208)

    φTD通过在频域中用特征函数计算损失，统一了分布式强化学习中的分位数与类别两类方法，并能学习连续分布混合等此前无法捕捉的分布，从而带来性能提升。

    

    分布式强化学习方法旨在学习回报的完整分布，然而现有的算法方法仅局限于狭窄的一类分布。我们提出φTD，一种通过在频域中计算损失来克服这一局限的方法。我们证明了该方法进一步统一了分布式强化学习中的分位数方法和类别方法，并使学习以前无法捕捉的分布成为可能，例如连续分布的混合。我们在合成基准和大规模深度强化学习环境中对φTD进行了实证研究，结果表明由更广泛的参数族所带来的灵活性能够带来性能的提升。

    arXiv:2611.25208v1 Announce Type: new  Abstract: Distributional Reinforcement Learning methods aim to learn the entire distribution of returns, yet current algorithmic approaches are limited to a narrow class of distributions. We propose $\varphi$TD, a method to overcome this limitation by computing the loss in the frequency domain. We demonstrate that this further unifies quantile and categorical approaches to distributional RL, and enables the learning of distributions that were previously impossible to capture, such as mixtures of continuous distributions. We empirically study $\varphi$TD, across synthetic benchmarks and large-scale deep RL environments and demonstrate that the flexibility obtained from wider parametric families can lead to improved performance.
    
[^182]: 从自博弈强化学习到约束控制：使用DAgger蒸馏DiffMPC学生模型

    From Self-Play RL to Constrained Control: Distilling DiffMPC Students with DAgger

    [https://arxiv.org/abs/2611.25197](https://arxiv.org/abs/2611.25197)

    该方法通过保留PPO教师骨干并用强制执行显式约束的DiffMPC替换其动作头，再借助DAgger在教师自博弈仿真器中蒸馏学生模型，成功为已训练的自博弈RL驾驶策略添加约束，学生性能持平甚至超越教师，且蒸馏耗时不超过教师训练时间的30%。

    

    自博弈强化学习（RL）能够产生强大的驾驶策略，但这些策略难以验证或解释。可微分模型预测控制（DiffMPC）可以强制执行显式约束，但通常需要通过模仿记录数据或代价高昂的强化学习来训练。为了向已训练好的策略添加约束，我们保留了PPO教师模型的骨干网络，并将其动作头替换为强制执行道路边界半平面和智能体椭圆约束的DiffMPC。随后，我们在教师的自博弈仿真器中使用DAgger对学生模型进行蒸馏。在9个随机种子上的实验表明，由较小教师模型蒸馏出的受约束学生在nuPlan上与教师持平，在CARLA和分布外的Waymo场景上超过了教师。对于更强的教师模型，学生在nuPlan和Waymo上的失败次数更少，同时在CARLA上保持同等水平。其蒸馏过程耗时不超过教师训练时间的30%。

    arXiv:2611.25197v1 Announce Type: cross  Abstract: Self-play reinforcement learning (RL) produces strong driving policies, but these policies are hard to verify or interpret. Differentiable model predictive control (DiffMPC) can enforce explicit constraints, but it is usually trained by imitating logged data or with costly RL. To add constraints to an already trained policy, we keep a PPO teacher's backbone and replace its action head with a DiffMPC that enforces road boundary half-planes and agent ellipses. We then distill the student with DAgger in the teacher's self-play simulator. Over 9 seeds, the constrained student of a smaller teacher matches its teacher on nuPlan and beats it on CARLA and on out-of-distribution Waymo scenes. For a stronger teacher, the student has fewer failures on nuPlan and Waymo, while staying level on CARLA. Its distillation takes no more than 30% of the teacher's training time.
    
[^183]: MADJEPA：用于医学异常检测的增强型多阶段联合嵌入预测架构

    MADJEPA: Enhanced Multi-Stage Joint-Embedding Predictive Architecture for Medical Anomaly Detection

    [https://arxiv.org/abs/2611.25194](https://arxiv.org/abs/2611.25194)

    该论文提出了MADJEPA，一种基于增强型多阶段联合嵌入预测架构的医学图像异常检测器，通过I-JEPA训练、目标编码器蒸馏和伪异常分类的多阶段训练流程，仅使用健康患者数据即可学习正常性的强表征并实现鲁棒的异常检测。

    

    医学图像中的异常检测是一项复杂的任务，需要仅使用健康患者的数据来开发一个鲁棒的模型。在无法获得标注数据的情况下，现代方法通常采用自监督学习来获得正常性的模型。由于世界模型有望为其环境构建强大的内部表征，我们认为它们是医学异常检测环境（即对健康患者的医学检查）的合适候选。为此，我们提出了MADJEPA，一种基于增强型多阶段联合嵌入预测架构（JEPA）的医学异常检测器。我们设计了一个多阶段训练流程来构建鲁棒的检测器：（1）基于图像的JEPA（I-JEPA）训练旨在学习正常训练数据的强表征；（2）目标编码器蒸馏旨在调节I-JEPA表征对异常数据的过度泛化；（3）伪异常分类旨在……（原文在此处截断）

    arXiv:2611.25194v1 Announce Type: cross  Abstract: Anomaly detection in medical images is a complex task, requiring the development of a robust model using only healthy patient data. Without access to labeled data, modern approaches typically employ self-supervised learning to obtain a model of normality. Since world models promise to build strong internal representations of their environment, we consider them as suitable candidates for the medical anomaly detection environment, namely medical investigations of healthy patients. To this end, we propose MADJEPA, a medical anomaly detector based on an enhanced multi-stage joint-embedding predictive architecture (JEPA). We design a multi-stage training pipeline to produce a robust detector: (1) image-based JEPA (I-JEPA) training aims to learn a strong representation of normal training data; (2) target encoder distillation aims to modulate over-generalization of I-JEPA representations to abnormal data; (3) pseudo-anomaly classification aim
    
[^184]: 盲区会迁移，知识留原地：强化学习从大模型评审员那里继承了什么

    Blind Spots Travel, Knowledge Stays Home: What RL Inherits from an LLM Judge

    [https://arxiv.org/abs/2611.25191](https://arxiv.org/abs/2611.25191)

    通过814次强化学习实验，本文发现LLM评审员对单个答案的知识几乎不会传递给被训练的模型（每百题仅纠正约一题），但其对某类问题的“盲区”会迁移，导致模型在新问题上系统性地拒绝回答该类问题。

    

    大模型评审员在不存在标准答案的场景中充当奖励信号，人们期望其知识能够传递给被训练的模型。在814次强化学习运行中，我们训练7-8B参数的模型回答闭卷问题或选择弃答，并探究究竟什么能够被传递。评审员对单个答案的了解会留在原地：一个被评审员奖励的错误答案只会在那一个问题上被记住。使用标准答案或远强评审员训练出的模型，在选择回答哪些新问题方面，并不比未训练模型通过采样32个答案并在答案不一致时弃答的策略更好；评审员的知识大约每百个问题只带来一个被纠正的答案。而评审员对某一类问题的看法则会迁移：一个拒绝了12个关于“已消失国家首都”正确训练答案中11个的评审员，会使模型在新问题上拒绝98%的此类首都问题，而这种损失无法从准确率和弃答率中显现。仅通过手工修改这12个判定……（原文摘要在此处截断）

    arXiv:2611.25191v1 Announce Type: cross  Abstract: LLM judges serve as rewards where no answer key exists, in the hope that their knowledge passes to the trained model. In 814 RL runs that teach 7-8B models to answer closed-book questions or abstain, we ask what passes. What a grader knows about single answers stays home. A wrong answer that a judge rewards is memorised for that one question. A model trained with the answer key, or with a far stronger judge, chooses which new questions to answer no better than the untrained model does by sampling 32 answers and abstaining where they disagree; the grader's knowledge adds about one corrected answer per hundred questions. What a grader believes about a kind of question travels. A judge that rejects eleven of the twelve correct training answers about capitals of former states teaches the model to refuse 98% of such capitals on new questions, a loss that accuracy and abstention rates do not reveal. Changing only those twelve verdicts by han
    
[^185]: 基于物理信息神经网络的学习优化中的PDE状态条件化

    PDE-State Conditioning for Learned Optimization Using Physics-Informed Neural Networks

    [https://arxiv.org/abs/2611.25185](https://arxiv.org/abs/2611.25185)

    该论文提出用11维PDE状态（刻画算子结构、残差几何与短时程更新响应）通过有界FiLM调制来条件化元学习优化器，使学习型PINN优化器首次显式利用PDE层面信息，并通过设计研究确定紧凑残差几何状态G2与有界FiLM为最佳设计组合。

    

    物理信息神经网络（PINN）的优化依赖于底层偏微分方程（PDE）的全局性质以及物理残差在整个定义域上的演化方式。诸如Adam等标准优化器使用固定的基于梯度的统计量，而学习型PINN优化器则用由局部参数和优化历史特征构建的学习式更新规则来替代固定规则。然而，这些方法均未显式地利用PDE层面的信息。我们研究了两个相互耦合的设计问题：学习型优化器应当观测哪些PDE层面的信息，以及这些信息应如何影响其更新？我们引入了一个11维的PDE状态，用以刻画算子结构、残差几何和短时程更新响应，并通过有界的逐特征调制（feature-wise modulation）将其用于条件化一个结构化的元学习优化器。反复开展的设计研究表明，在所有被测试的方案中，紧凑的残差几何状态G2与有界的FiLM是综合表现最强的选择。

    arXiv:2611.25185v1 Announce Type: new  Abstract: Physics-informed neural network (PINN) optimization depends on global properties of the underlying PDE and how the physics residual evolves across the domain. Standard optimizers such as Adam use fixed gradient-based statistics, while learned PINN optimizers replace the fixed rule with a learned update built from local parameters and optimization-history features. However, none explicitly use PDE-level information. We study two coupled design questions: what PDE-level information should a learned optimizer observe, and how should that information influence its update? We introduce an 11-dimensional PDE state-capturing operator structure, residual geometry, and short-horizon update response, and use it to condition a structured meta-learned optimizer through bounded feature-wise modulation. Repeated design studies identify the compact residual-geometry state G2 and bounded FiLM as the strongest overall choices among those tested. The resu
    
[^186]: 选择之前：研究语言模型中前瞻性表征的证据框架

    Before the Choice: An Evidential Framework for Studying Prospective Representations in Language Models

    [https://arxiv.org/abs/2611.25170](https://arxiv.org/abs/2611.25170)

    该论文提出“选择之前”这一证据导向框架，区分前瞻性内容、前瞻性计算与因果引导，用以检验语言模型是否内部表征未被选择的轨迹，并通过预注册探针实验发现低秩探针未达标且存在排除法重构的混淆问题。

    

    一个可观察的决策揭示了被选中的行动，但并不能确定内部表征了哪些备选方案。借鉴前瞻性研究与海马体前向扫掠的观测，我们提出了“选择之前”，一个面向证据的框架，用以区分面向未来的内容、前瞻性计算与因果引导。该框架组织了四组实验对比，并使其替代解释变得明确。其首个测试案例关注确定性符号环境中的未被选择轨迹。一项预注册的低秩探针在密封测试集上仅评估一次，未达到其预设标准（配对准确率为0.020，而基于文本与观测选择的基线为0.195；共293个有效配对）。探索性假设H1b揭示了“排除法重构”问题：识别出被选中的候选即可能确定未被选择的目标。在H1c中，匹配世界在保持候选键位级相同的同时改变未被选择的延续……

    arXiv:2611.25170v1 Announce Type: cross  Abstract: An observable decision reveals the selected action but does not determine which alternatives were internally represented. Drawing on prospection research and hippocampal forward-sweep observations, we develop Before the Choice, an evidence-oriented framework distinguishing future-directed content, prospective computation, and causal guidance. The framework organizes four experimental contrasts and makes their alternative explanations explicit. Its first test case concerns unchosen trajectories in deterministic symbolic environments. A preregistered low-rank probe evaluated once on a sealed TEST did not meet its criterion (paired accuracy 0.020 versus 0.195 for a text-and-observed-choice baseline; 293 admissible pairs). Exploratory H1b exposed a reconstruction-by-elimination problem: identifying the chosen candidate can determine the unchosen target. In H1c, matched worlds varied an unchosen continuation while keeping candidate keys bit
    
[^187]: 面向专家专业化与路由的博弈论数据分配

    Game-Theoretic Data Allocation for Expert Specialization and Routing

    [https://arxiv.org/abs/2611.25167](https://arxiv.org/abs/2611.25167)

    该论文提出InflEns方法，用一个位置博弈同时决定LoRA专家混合的训练数据分配与推理路由，使份额最高的专家在约90%的提示上表现最优，从而让分工在训练前即可检验并用分配本身指导推理。

    

    用于训练专家的数据分配方式是否也能预测哪个专家在给定提示上表现最佳？若存在这样的联系，将使预期的劳动分工在训练之前即可被检验，并允许分配本身来指导推理。我们提出InflEns，一种LoRA专家混合模型，其训练权重与推理路由均来自固定提示-特征空间中的同一个位置博弈。参与者通过选择位置来最大化其在提示分布中的预期份额，而无需使用专家损失。我们利用该博弈的核通用理论准则来判断共置位置何时对分离变得局部不稳定。我们在由七个安全相关基准组成的数据集上评估了InflEns，结果表明：在锐化单纯形表示下，跨四个核，份额最高的专家在约90%的提示上具有最低的留出响应负对数似然（NLL），相比之下……

    arXiv:2611.25167v1 Announce Type: cross  Abstract: Can the allocation used to train experts also predict which expert will perform best on a prompt? Such a connection would make the intended division of labor inspectable before training and allow the allocation itself to guide inference. We introduce InflEns, a mixture of LoRA experts whose training weights and inference routes come from the same location game in a fixed prompt-trait space. Players choose positions to maximize their expected shares of the prompt distribution, without using expert losses. We utilize the game's kernel-general theoretical criterion for when co-located positions become locally unstable to separation. We evaluate InflEns on a corpus assembled from seven safety-related benchmarks, which demonstrates that the highest-share expert has the lowest held-out response negative log-likelihood (NLL) on approximately 90% of prompts across four kernels under a sharpened simplex representation, compared with approximate
    
[^188]: 在稀疏自编码器中分离可解释性与重构

    Separating Interpretability and Reconstruction in Sparse Autoencoders

    [https://arxiv.org/abs/2611.25165](https://arxiv.org/abs/2611.25165)

    提出DualSAE模块化框架，通过可插拔的残差分支与去相关损失，将稀疏自编码器中的可解释性与重构保真度这两个目标分离，从而学习到语义更纯粹的特征。

    

    稀疏自编码器通过将神经网络的内部表示分解为稀疏且可解释的特征，是机制可解释性研究的有效工具。然而，在传统SAE中，单一的稀疏表示同时承担着重构保真度与可解释性两个职责，而这两个目标并不保证相互一致。为了将这两个目标分离开来，我们提出了DualSAE，这是一个模块化框架，通过一个可插拔的残差分支来增强标准SAE。标准SAE充当稀疏分支，学习包含输入主要语义信息的稀疏编码；而新增的残差分支则负责对稀疏分支所忽略的非语义信息进行建模。我们通过一个去相关损失确保两个分支对整体重构的贡献具有最小的重叠。此外，我们还采用了一种两阶段训练策略，先稳定稀疏表示的学习，然后再将其与残差分支进行耦合。

    arXiv:2611.25165v1 Announce Type: new  Abstract: Sparse Autoencoders (SAEs) are effective tools for mechanistic interpretability by decomposing neural representations into sparse and interpretable features. However, a single sparse representation in traditional SAEs is responsible for both reconstruction fidelity and interpretability, which are not guaranteed to align. To separate these objectives, we propose DualSAE, a modular framework that augments standard SAEs with a plug-in Residual Branch. A standard SAE serves as the Sparse Branch, learning a sparse code that contains the dominant semantics of the input. The added Residual Branch models the non-semantic information omitted by the Sparse Branch. We ensure that the contributions of the two branches towards the overall reconstruction have minimal overlap through a decorrelation loss. We also adopt a two-stage training strategy that first stabilizes sparse representation learning before coupling with the Residual Branch. Across div
    
[^189]: 马尔可夫决策过程（MDP）中用于极小极大模型估计的贝叶斯 κ-Explorer

    Bayesian $\kappa$-Explorer for Minimax Model Estimation in MDPs

    [https://arxiv.org/abs/2611.25160](https://arxiv.org/abs/2611.25160)

    该论文提出贝叶斯 κ-Explorer，通过在转移分布上放置狄利克雷先验并基于后验均值模型进行规划，以闭式探索信号简化了 MDP 中极小极大转移估计的主动探索，在免除置信集优化的同时仍保持 Õ(T^(-1/3)) 的收敛率。

    

    我们研究了未知马尔可夫决策过程（MDP）中用于极小极大转移估计的主动探索问题。频率学派的 κ-Explorer 通过一族参数化的探索目标来解决这一问题，但其需要构建置信界，并对可行的转移核与占用测度进行联合优化，由此带来的计算负担限制了其实际适用性。我们提出了贝叶斯 κ-Explorer，通过在转移分布上放置狄利克雷先验来简化这一过程。在后验均值估计量下，后验期望平方 ℓ₂ 误差等于总后验方差，从而提供了一个闭式（closed-form）的探索信号。该算法在后验均值模型下进行规划，从而免除了置信集优化。在所陈述的假设下，这一更简单的规划过程仍能保持 Õ(T^(-1/3)) 的代理收敛率。

    arXiv:2611.25160v1 Announce Type: new  Abstract: We study active exploration for minimax transition estimation in unknown Markov decision processes (MDPs). Frequentist $\kappa$-Explorer~\citep{gu2026kappa} addresses this problem through a parameterized family of exploration objectives, but requires confidence-bound constructions and joint optimization over plausible transition kernels and occupancy measures, introducing a computational burden that limits its practical applicability. We develop Bayesian $\kappa$-Explorer, which simplifies this procedure by placing Dirichlet priors on the transition distributions. Under the posterior mean estimator, the posterior expected squared $\ell_2$ error equals the total posterior variance, providing a closed-form exploration signal. The algorithm plans under the posterior mean model, eliminating confidence-set optimization. Under the stated assumptions, this simpler planning procedure retains the $\widetilde{O}(T^{-1/3})$ surrogate convergence ra
    
[^190]: 分层内容摘要的率失真评估

    Rate-distortion Assessment for Hierarchical Content Summarization

    [https://arxiv.org/abs/2611.25151](https://arxiv.org/abs/2611.25151)

    该论文受香农率失真理论启发，将分层内容摘要形式化为信息编码问题，提出了一个可定量刻画压缩效率与信息损失之间权衡的评估框架，从而实现对不同生成模型在给定率预算条件下的科学基准测试。

    

    强大的生成模型，包括大型语言模型和视觉模型，已经通过端到端的基于提示的生成方式，在广泛的内容生成、编辑和操作任务中取得了成功。然而，目前对生成式内容操作质量的评估仍然停留在主观层面，这限制了对模型能力的科学理解。受香农信息压缩理论中率失真理论的启发，在这项工作中，我们将分层内容摘要形式化为一个信息编码问题。我们设计了一个有原则性依据的框架来提取源内容的语义层次结构，以支持对生成式内容操作过程中压缩效率（率）与信息损失（失真）之间率失真权衡的定量刻画，从而使我们能够在给定的率预算条件下对各种模型的性能进行基准测试。

    arXiv:2611.25151v1 Announce Type: new  Abstract: Powerful generative models including large language and vision models have demonstrated success across a wide range of content generation, editing and manipulation tasks via end-to-end prompt-based generation. However, current assessment of the quality of generative content manipulation remains subjective, limiting scientific understanding of model capabilities. Inspired by rate-distortion theory from Shannon information compression theory, in this work we formalize hierarchical content summarization as an information encoding problem. We design a principled framework to extract the semantic hierarchy of the source content to support quantitative characterization of the rate-distortion trade-off between compression efficiency (rate) against information loss (distortion) during generative content manipulation, which allows us to benchmark various models with respect to performance conditional on rate budget of interest. We demonstrate the
    
[^191]: CAIMS：可扩展的跨阵列存内搜索

    CAIMS: Scalable Cross-Array In-Memory Search

    [https://arxiv.org/abs/2611.25141](https://arxiv.org/abs/2611.25141)

    CAIMS提出了一个与存储技术和应用无关的跨阵列存内搜索框架，通过准确高效地将局部匹配结果归约为全局top-k结果，解决了跨多个CAM阵列进行可扩展top-k相似度搜索的关键难题。

    

    搜索，尤其是top-k相似度搜索，在许多应用中扮演着至关重要的角色，但其性能可能严重受限于内存访问开销。存内搜索（In-Memory Search, IMS）作为基于CPU/GPU方法的替代方案，利用内容寻址存储器（CAM）阵列内部的大规模并行性来就地执行搜索，同时提供低延迟和高能效。然而，单个CAM阵列的容量通常不足以应对现代工作负载，因此需要将数据分布在多个阵列上。跨多个CAM阵列部署可扩展的top-k相似度搜索带来了一个现有设计未能解决的根本性问题：如何准确且高效地将局部二进制匹配结果归约为全局top-k结果。本工作提出了CAIMS，一个与存储技术和应用无关的跨阵列存内搜索框架，能够实现可扩展且高效的top-k相似度搜索，同时保持准确性。

    arXiv:2611.25141v1 Announce Type: cross  Abstract: Search, especially top-k similarity search, plays a vital role in many applications, but its performance can be severely limited by memory access overhead. In-memory search (IMS), an alternative to CPU/GPU-based approaches, exploits massive parallelism within a content-addressable memory (CAM) array to perform search in situ and provides both low latency and high energy efficiency. However, the capacity of a single CAM array is often insufficient for modern workloads, requiring data to be distributed across multiple arrays. Deploying scalable top-k similarity search across multiple CAM arrays poses a fundamental problem that existing designs fail to address: how to accurately and efficiently reduce local binary match results to global top-k results. This work introduces CAIMS, a cross-array IMS framework that is agnostic to memory technologies and applications, enabling scalable and efficient top-k similarity search while preserving ac
    
[^192]: 基于神经算子的层状介质弹性地震波传播时间外推

    Temporal Extrapolation of Elastic Seismic Wave Propagation in Layered Media Using Neural Operators

    [https://arxiv.org/abs/2611.25130](https://arxiv.org/abs/2611.25130)

    该论文提出将嵌入RK4时间积分器的DeepONet与FNO残差校正以预测-校正器方式耦合训练，实现了层状弹性介质中地震波速度场超出训练时间区间的自回归时间外推预测。

    

    对于神经算子模型而言，对超出训练时间区间的地震速度场进行时间预测仍然具有挑战性。本研究开发了一个耦合神经算子框架，用于层状弹性介质中应力条件化的时间预测。嵌入时间积分器的深度算子网络通过嵌入的四阶龙格-库塔格式推进速度场，而傅里叶神经算子学习对预测空间场的残差校正。这两个组件以预测-校正器的配置进行联合训练。预测以前两个时间层的参考速度场初始化，并以自回归方式推进，在整个滚动预测过程中持续提供参考应力场和规定的激励，包括超出训练区间的部分。四个案例考虑了固定总深度的四层土柱中的水平极化剪切波……

    arXiv:2611.25130v1 Announce Type: new  Abstract: Temporal prediction of seismic velocity fields beyond the time interval represented in training remains challenging for neural operator models. This study develops a coupled neural operator framework for stress-conditioned temporal prediction in layered elastic media. A time-integrator-embedded Deep Operator Network (TI-DeepONet) advances the velocity field through an embedded fourth-order Runge-Kutta (RK4) scheme, while a Fourier Neural Operator (FNO) learns a residual correction to the predicted spatial field. The two components are trained jointly in a predictor-corrector configuration. Prediction is initialized with reference velocity fields at the first two time levels and proceeds autoregressively, with reference stress fields and prescribed excitation supplied throughout rollout, including beyond the training interval. Four cases consider horizontally polarized shear (SH) waves in four-layer soil columns with fixed total depths of
    
[^193]: 天气模型中学习动力学的审计

    Auditing Learned Dynamics in Weather Models

    [https://arxiv.org/abs/2611.25119](https://arxiv.org/abs/2611.25119)

    该论文利用稀疏自编码器对GraphCast天气模型的内部机制进行审计，发现消融缺乏传统物理语义的计算网格特征也会增加位势高度误差，且抑制与对流和低层涡度相关的特征会削弱热带气旋加深的预测，为AI天气预报模型在高保证场景下的可靠性审计与极端事件预报改进提供了新方法。

    

    学习型天气模型是在实验上可访问的计算系统，其内部表示可以被检查和操纵。利用来自GraphCast（DeepMind的自回归天气预报模型）的稀疏自编码器特征，我们测试了因果相关性是否需要物理语义，这一问题对于审计在高保证（high-assurance）场景中使用的AI预报器是否依赖可靠机制至关重要。我们还研究了与物理相关联的潜在方向是否可以被引导以改进极端事件预报，其中干预是在滚动预报过程中应用于模型的内部激活。我们发现，消融一个锁定在GraphCast计算网格上的特征会增加全球位势高度误差，尽管该特征缺乏传统的物理语义。其次，抑制与对流和低层涡度相关联的特征会削弱预测的热带气旋加深（即预报期间中心气压的下降……）。

    arXiv:2611.25119v1 Announce Type: new  Abstract: Learned weather models are experimentally accessible computational systems whose internal representations can be inspected and manipulated. Using sparse autoencoder features from GraphCast, DeepMind's autoregressive weather forecasting model, we test whether causal relevance requires physical semantics, a question that matters for auditing whether AI forecasters used in high-assurance settings rely on reliable mechanisms. We also look at whether physically associated latent directions can be steered to improve extreme-event forecasts, where interventions are applied to the model's internal activations during the rollout. We find that ablating a feature locked to GraphCast's computational mesh increases global geopotential height error despite lacking conventional physical semantics. Second, suppressing convection- and low-level-vorticity-associated features weakens predicted tropical-cyclone deepening (the fall in central pressure during
    
[^194]: 基于R语言的DISCO指标实现基于密度聚类结果的快速轻量内存评估

    Fast and Memory-Light Evaluation of Density-Based Clusterings with the DISCO Index in R

    [https://arxiv.org/abs/2611.25114](https://arxiv.org/abs/2611.25114)

    该论文提出了discoCVI R软件包，通过从互可达性最小生成树的单链接合并树中提取DISCO指标所需的全部信息，将内存需求从二次方降至线性，实现了对基于密度聚类（含噪声标签质量评估）的精确且高效的评估。

    

    DISCO是一种新近提出的用于基于密度聚类的内部聚类有效性指标，与早期指标不同，它还能评估噪声标签的质量。discoCVI软件包使DISCO可以在R中使用。本文描述了该软件包的设计和典型工作流程，其中DISCO被用于比较聚类结果并在没有真实标签的情况下选择HDBSCAN的参数，随后介绍了0.2.0版本中引入的实现。DISCO是通过所有点对之间的密度连通性距离定义的，因此直接实现方式需要存储多个n×n矩阵。我们证明DISCO所需的每一个量都可以从互可达性最小生成树的单链接合并树中读取，因此该指标可以在内存随点数线性增长的情况下被精确计算，同时运行时间仍保持与原始定义相同的二次方复杂度。该实现已与之前的版本进行了对比测试。

    arXiv:2611.25114v1 Announce Type: new  Abstract: DISCO is a recent internal cluster validity index for density-based clustering that, unlike earlier indices, also evaluates the quality of noise labels. The discoCVI package makes DISCO available in R. This article describes the design of the package and a typical workflow, in which DISCO is used to compare clusterings and to choose the parameters of HDBSCAN without ground truth, followed by the implementation introduced in version 0.2.0. DISCO is defined through density-connectivity distances between all pairs of points, so direct implementations store several n-by-n matrices. We show that every quantity DISCO needs can be read off the single-linkage merge tree of the mutual-reachability minimum spanning tree, so that the index can be computed exactly with memory that grows linearly with the number of points, while the running time remains quadratic as in the original definition. The implementation is tested against the previous version
    
[^195]: 长距离视频呈现攻击检测：跨1-100米距离与两种传感器类型的三分类基准

    Long-Range Video Presentation Attack Detection: A Three-Class Benchmark Across 1-100 m and Two Sensor Types

    [https://arxiv.org/abs/2611.25108](https://arxiv.org/abs/2611.25108)

    该论文提出了首个面向长距离场景的视频人脸呈现攻击检测基准，涵盖1-100米五种拍摄距离、室内外环境、两种传感器类型及真人/欺骗/道具三分类标签（37万余帧、261个身份），并对四种CNN架构进行微调与评测。

    

    部署在边境管控和机场监控中的人脸呈现攻击检测（PAD）系统受到一个关键假设的限制：目标对象在近距离拍摄，通常在一米以内。目前尚无现有基准系统地解决多距离采集、多样化环境条件以及欺骗攻击与遮挡攻击并存的问题，这使得已部署的模型在面对真实场景时准备不足。我们引入了一个基于视频的PAD数据集，包含来自261个独特身份的37万余帧图像，数据在室内和室外条件下、使用单反相机（DSLR，C）和移动传感器（M）在五个 standoff 距离（1米至100米）采集，并带有三分类标签：真人、欺骗攻击和道具（遮挡）；模型在视频片段而非单帧上进行训练和评估。我们对四种CNN架构进行了微调和基准测试：MobileNetV3、MobileNet、EfficientNet和ResNet-50。分辨率更高的DSLR相机保持了接近完美的欺骗分类性能……

    arXiv:2611.25108v1 Announce Type: new  Abstract: Face presentation attack detection (PAD) systems deployed in border control and airport surveillance are constrained by a critical assumption: subjects are captured at close range, typically under one meter. No existing benchmark systematically addresses multi-distance capture, diverse environmental conditions, and concurrent spoofing and occlusion-based attacks, leaving deployed models underprepared for real-world scenarios. We introduce a video-based PAD dataset of 374,000+ frames from 261 unique identities collected across five standoff distances (1 m-100 m) under indoor and outdoor conditions using a DSLR camera (C) and a mobile sensor (M), with three-class labels: live, spoof, and props (occlusions); models are trained and evaluated on video clips rather than single frames. We fine-tune and benchmark four CNN architectures: MobileNetV3, MobileNet, EfficientNet, and ResNet-50. The higher-resolution DSLR maintains near-perfect spoof c
    
[^196]: HapBO：通过专家自然语言先验实现的 α-π 贝叶斯优化

    HapBO: Harnessed $\alpha$-$\pi$ Bayesian Optimization via Expert Natural Language Priors

    [https://arxiv.org/abs/2611.25100](https://arxiv.org/abs/2611.25100)

    该论文提出HapBO方法，将从人类专家或大语言模型一次性引出的自然语言先验确定性地编译为不限于高斯族的可重用信念函数，从而提升贝叶斯优化的样本效率，并对可能误导的信息保持鲁棒。

    

    贝叶斯优化（BO）在获得关于问题的先验知识时可以显著提升样本效率。然而在实践中，这类知识往往只能以自然语言的形式获得，例如专家直觉或文档，而非先验引导的BO方法所要求的显式概率分布图景。有效利用这类知识既需要从语言到先验的富有表现力的映射，也需要对可能产生误导的信息保持鲁棒性。我们提出HapBO（Harnessed α-π 贝叶斯优化），它从人类专家、大语言模型（LLM）或两者中一次性引出信念，并将结构化规范确定性地编译为可重用的信念函数（π: X → [0,1]），其表示形式丰富，不限于高斯分布族。基于 α-πBO，HapBO 通过在任意非负[函数上]施加有界正乘数来应用这一信念函数……

    arXiv:2611.25100v1 Announce Type: new  Abstract: Bayesian Optimization (BO) can be substantially more sample-efficient when informed by prior knowledge about the problem. In practice, however, such knowledge is often available only in natural language, e.g., expert intuitions or documentation, rather than as the explicit probability distribution landscape that prior-guided BO methods may require. Leveraging it effectively demands both an expressive mapping from language to priors and robustness to information that may be misleading. We propose HapBO (Harnessed $\alpha$-$\pi$ Bayesian Optimization), which elicits beliefs once from a human expert, a large language model (LLM), or both, and deterministically compiles a structured specification into a reusable belief function ($\pi: \mathcal{X} \to[0,1]$), with a rich representation not restricted to a Gaussian family. Building on $\alpha$-$\pi$BO, HapBO applies this belief function through a bounded positive multiplier on any nonnegative 
    
[^197]: BuggyMLE：通过错误注入扩展机器学习工程难度

    BuggyMLE: Scaling Machine Learning Engineering Difficulty with Bug Injection

    [https://arxiv.org/abs/2611.25088](https://arxiv.org/abs/2611.25088)

    该论文提出基于十个结构特征的代码复杂度指数（CCI）来有效预测编码基准的任务难度（R²=0.951），并据此构建了BuggyMLE框架，通过向可运行的代码库注入受控错误来规模化提升机器学习工程任务的难度。

    

    让AI系统能够自主改进AI模型，需要具备执行机器学习工程（MLE）能力的智能体。在实践中，设计、训练和评估复杂模型需要修改多文件代码库，以复现结果、调试训练过程、整合数据集以及移植框架。然而，现有的MLE基准测试将智能体限制为仅能进行单文件编辑，且无法完整访问代码库或shell，而常见的代码库级软件工程基准测试已迅速饱和。为解决这一问题，我们提出了代码复杂度指数（CCI），这是一种基于十个结构特征的难度度量指标，涵盖代码库规模、依赖深度和补丁大小。CCI在八个公开编码基准测试中能够有力预测平均解决率（R²=0.951），并确定补丁大小和复杂度是最主要的预测因素。在CCI的指导下，我们开发了BuggyMLE，这是一个可扩展的框架，智能体在其中向正常工作的代码注入受控的变异错误（注：原文摘要至此截断）。

    arXiv:2611.25088v1 Announce Type: new  Abstract: Enabling AI systems to autonomously improve AI models requires agents capable of performing machine learning engineering (MLE). In practice, designing, training, and evaluating complex models demands modifying multi-file repositories to reproduce results, debug training runs, integrate datasets, and port frameworks. However, existing MLE benchmarks limit agents to single-file edits without full repository or shell access, and common repository-level software engineering benchmarks have rapidly saturated. To address this, we introduce the Code Complexity Index (CCI), a difficulty metric based on ten structural features spanning repository scale, dependency depth, and patch size. CCI strongly predicts average solve rates across eight public coding benchmarks ($R^2 = 0.951$), identifying patch size and complexity as top predictors. Guided by CCI, we develop BuggyMLE, a scalable framework where agents inject controlled mutations into working
    
[^198]: PriTabICL：基于任务对齐的簇级合成的差分隐私表格上下文学习

    PriTabICL: Differentially Private Tabular In-Context Learning with Task-Aligned Cluster-Wise Synthesis

    [https://arxiv.org/abs/2611.25073](https://arxiv.org/abs/2611.25073)

    该论文提出PriTabICL，通过任务对齐的簇级差分隐私数据合成来生成隐私保护的表格上下文学习示例，克服了直接对整个数据集应用通用差分隐私合成时全局分布建模掩盖局部结构和有限隐私预算下广泛属性依赖降低预测准确性的两大局限。

    

    表格上下文学习使大语言模型（LLM）能够利用提示中少量带标签的示例来预测目标记录。然而，这些示例通常包含敏感的个人信息。差分隐私（DP）提供了有原则的保护，但基于噪声的方法会引入大量噪声，扭曲对准确预测至关重要的特征-标签关系，尤其是随着数据维度的增长。在本工作中，我们研究用于生成隐私保护TabICL示例的差分隐私数据合成。我们指出了将通用差分隐私合成直接应用于整个数据集的两个局限性：建模全局分布可能掩盖异构数据中的局部结构和少数模式，而在有限的隐私预算下测量广泛的属性依赖关系可能会降低用于与预测最相关的关系的准确性。更广泛地说，全局统计……（原文摘要在此处截断）

    arXiv:2611.25073v1 Announce Type: cross  Abstract: Tabular in-context learning (TabICL) enables large language models (LLMs) to predict target records using a handful of labeled demonstrations in their prompts. However, these demonstrations often contain sensitive personal information. Differential privacy (DP) offers principled protection, but noise-based methods introduce substantial noise, distorting the feature-label relationships essential for accurate prediction, especially as data dimensionality grows.   In this work, we study DP data synthesis for generating privacy-preserving TabICL demonstrations. We identify two limitations of directly applying general-purpose DP synthesis to entire datasets: modeling a global distribution can obscure local structure and minority patterns in heterogeneous data, while measuring broad attribute dependencies under a limited privacy budget can reduce the accuracy available for relationships most relevant to prediction. More broadly, global stati
    
[^199]: 纵向矩阵数据的经验变分贝叶斯因子分解

    Empirical Variational Bayes Factorization for Longitudinal Matrix Data

    [https://arxiv.org/abs/2611.25065](https://arxiv.org/abs/2611.25065)

    提出了一种经验变分贝叶斯因子分解方法，能够高效处理具有时间相关性的高维纵向矩阵数据，自动确定所有超参数，并有效应对缺失数据和不规则时间点的挑战。

    

    在生物医学研究中，为多个个体随时间收集高维数据正变得越来越普遍，尤其是在疾病进展研究或分子时间进程实验中。通常情况下，数据存在缺失，或不同个体之间的时间点间隔不规则。目前缺乏灵活的方法来探索随时间变化的潜在模式并处理缺失数据。我们提出了一种经验变分贝叶斯（EVB）方法来对高维纵向数据进行因子分解，其中潜在因子具有时间上的相关性。EVB方法计算高效，能够灵活适应随时间变化的协方差结构，并自动确定所有超参数（包括潜在秩，即因子数量）。模拟研究表明，该方法比其他策略更精确地捕捉潜在信号，并且在处理缺失数据方面表现良好。

    arXiv:2611.25065v1 Announce Type: cross  Abstract: Collecting high-dimensional data over time for multiple individuals is increasingly common in biomedical research, particularly in studies of disease progression or molecular time-course experiments. Often data are missing or the timepoints are irregularly spaced across individuals. There is a lack of flexible methodology to explore latent patterns of change over time and account for missing data. We propose an empirical variational Bayes (EVB) approach to factorize high-dimensional longitudinal data, in which the underlying factors have temporal correlation over time. The EVB approach is computationally efficient and accommodates flexible covariance structures over time with automatic determination of all hyper-parameters (including the underlying rank, i.e., number of factors). Simulations demonstrate that the approach captures the underlying signal more precisely than other strategies, and performs well with respect to missing data 
    
[^200]: Infer3D：逆向生成建模实现鲁棒的3D重建

    Infer3D: Inverse Generative Modeling Enables Robust 3D Reconstruction

    [https://arxiv.org/abs/2611.25061](https://arxiv.org/abs/2611.25061)

    该论文提出Infer3D逆向生成建模框架，通过恢复生成模型的潜变量并结合冻结的3D提升网络约束优化空间，使单目3D重建在分布外场景条件下依然保持鲁棒性。

    

    即使从高度陌生或极端的视角，人类也能理解场景的3D结构。认知科学表明，这种泛化能力源于一种自顶向下的生成式推理策略。相比之下，最先进的3D重建神经网络通常依赖摊销式前馈推理。这种方法虽然快速，但在面对训练中未见过的分布外（OOD）场景条件时会性能退化。受人类视觉生成特性的启发，我们提出了Infer3D，这是一个逆向生成建模框架，通过恢复生成模型的潜变量来最佳匹配单目输入。为防止退化的平坦输出，我们将生成式图像先验与一个冻结的3D提升网络耦合，将优化空间约束在有效的、多视角一致的3D流形上。在各种受控的合成和真实世界任务中，Infer3D展现出了卓越的OOD重建能力。

    arXiv:2611.25061v1 Announce Type: cross  Abstract: Humans can understand the 3D structure of scenes even from highly unfamiliar or extreme viewpoints. Cognitive science suggests this generalization stems from a top-down generative inference strategy. In contrast, state-of-the-art 3D reconstruction neural networks typically rely on amortized, feedforward inference. While fast, this pattern-matching approach degrades when faced with out-of-distribution (OOD) scene conditions unseen during training. Inspired by the generative nature of human vision, we propose Infer3D, an inverse generative modeling framework that recovers the latent variables of a generative model to best match the monocular input. To prevent degenerate flat outputs, we couple the generative image prior with a frozen 3D lifting network that constrains the optimization space to a valid, multi-view-consistent 3D manifold. Across a variety of controlled synthetic and real-world tasks, Infer3D demonstrates superior OOD recon
    
[^201]: 所有VC类的近二次样本压缩

    Near-Quadratic Sample Compression for All VC Classes

    [https://arxiv.org/abs/2611.25011](https://arxiv.org/abs/2611.25011)

    本文通过构造大小为 O(d²log²d) 的样本压缩方案，证明了Floyd和Warmuth提出的弱样本压缩猜想，即每个VC维为 d 的二值概念类都存在多项式大小的样本压缩方案。

    

    样本压缩猜想是学习理论中一个长期悬而未决的问题，起源于Littlestone和Warmuth在20世纪80年代的工作。其核心在于简化与泛化之间的联系：如果一个与训练样本一致的预测器仅通过该样本的一小部分来定义，那么它很可能具有泛化能力。样本压缩方案将这一思想形式化，许多自然的学习算法都可以通过这种方式进行分析。Floyd和Warmuth（1995）提出的样本压缩猜想断言，每个VC维为 $d$ 的二值概念类都存在大小为 $O(d)$ 的样本压缩方案；一个较弱的猜想只要求方案大小为 $d$ 的多项式。这两个猜想至今仍未解决：第一个仅依赖于 $d$ 的界由Moran和Yehudayoff（2016）给出，但它是 $d$ 的指数级。我们通过构造一个大小为 $O(d^2 \log^2 d)$ 的样本压缩方案，证明了较弱的猜想。

    arXiv:2611.25011v1 Announce Type: cross  Abstract: The sample compression conjecture is a longstanding open problem in learning theory, originating in the work of Littlestone and Warmuth in the 1980s. At its heart is a connection between simplification and generalization: if a predictor consistent with a training sample is defined using only a small part of that sample, then it is likely to generalize. Sample compression schemes formalize this idea, and many natural learning algorithms can be analyzed this way.   The sample compression conjecture of Floyd and Warmuth (1995) asserts that every binary concept class of VC dimension $d$ admits a sample compression scheme of size $O(d)$; a weaker conjecture asks only for size polynomial in $d$. Both have remained open: the first bound depending only on $d$, due to Moran and Yehudayoff (2016), is exponential in $d$. We prove the weaker conjecture by constructing a sample compression scheme of size $O(d^2 \log^2 d)$.
    
[^202]: 特征降维技术对对抗攻击的影响研究

    On the Impact of Feature Reduction Techniques on Adversarial Attacks

    [https://arxiv.org/abs/2611.25004](https://arxiv.org/abs/2611.25004)

    本研究首次系统分析了特征降维技术（如减少维度）在APT数据集上遭受投毒、后门和通用对抗扰动攻击时，对支持向量机、决策树、随机森林等多种机器学习模型对抗鲁棒性的影响。

    

    对抗攻击通过利用机器学习模型固有的脆弱性，对其构成了重大威胁。研究人员建立了一个名为对抗机器学习的跨学科领域，以探索这些脆弱性并制定相应的防御策略。特征降维策略被用于降低数据维度，以减少训练时间、避免维度灾难、防止过拟合并缓解冗余特征。然而，特征降维在不同分类器和攻击场景下对对抗鲁棒性的影响仍未得到充分探索。本研究旨在分析特征降维技术对多样化机器学习模型对抗鲁棒性的影响，研究对象是在APT数据集上遭受投毒攻击、后门攻击和通用对抗扰动攻击的模型。这些模型包括支持向量机、决策树、随机森林、多……（摘要原文在此处截断）

    arXiv:2611.25004v1 Announce Type: cross  Abstract: Adversarial attacks have posed significant threats to machine learning models by exploiting their inherent vulnerabilities. Researchers have established an interdisciplinary field called adversarial machine learning to explore these vulnerabilities and develop corresponding defense strategies. Feature reduction strategies are used to decrease the dimensionality of the data in order to reduce training time, avoid the curse of dimensionality, prevent overfitting, and mitigate redundant features. However, their effect on adversarial robustness across different classifiers and attack scenarios remains insufficiently explored. This study aims to analyze the effect of feature reduction techniques on the adversarial robustness of a diverse set of machine learning models under poisoning, backdoor, and universal adversarial perturbation attacks on the APT dataset. These models include support vector machine, decision tree, random forest, multi-
    
[^203]: 一个语言模型多样性采样模型

    A Model of Diverse Sampling from Language Models

    [https://arxiv.org/abs/2611.25003](https://arxiv.org/abs/2611.25003)

    该论文提出 PoolDPP 方法，将语言模型的多样性采样形式化为行列式点过程（DPP）下的子集采样，通过候选池上的 K-DPP 采样器实现了输出与目标分布间 KL 散度 O(1/N) 的收敛保证，并在实验中扩展了质量-多样性前沿。

    

    语言模型通常能产生高质量的个体样本，但其样本集合往往质量不佳，重复样本聚集在相同的语义模式附近。我们将多样性生成形式化为从行列式点过程（DPP）中采样大小为 $K$ 的完整字符串子集，其中似然编码条目质量，嵌入几何结构编码相似输出之间的排斥作用。由于对所有字符串进行精确推理是不可行的，我们提出 PoolDPP：从一个易于处理的提议分布中抽取有限的候选池，通过重要性加权使候选偏向目标质量，然后在诱导的池核上运行 K-DPP 采样器；贪心 MAP 变体则提供确定性的选择方式。我们证明了一个具有显式常数的有限样本界：采样器输出与目标 $K$-DPP 之间的 KL 散度随池大小 $N$ 以 O(1/N) 的速度衰减。实证结果表明，PoolDPP 在开放式生成任务上扩展了质量-多样性边界，并提高了精确率-召回率……（摘要截断）

    arXiv:2611.25003v1 Announce Type: cross  Abstract: Language models often produce high-quality individual samples but poor sets, with repeated samples clustering around the same semantic modes. We formalize diverse generation as sampling a size-$K$ subset of complete strings from a determinantal point process (DPP), where the likelihood encodes item quality, and the embedding geometry encodes repulsion between similar outputs. Exact inference over all strings is intractable, so we propose PoolDPP: draw a finite candidate pool from a tractable proposal, importance-weight the candidates toward the target quality, and run a $K$-DPP sampler on the induced pool kernel; a greedy MAP variant provides deterministic selection. We prove a finite-sample bound with an explicit constant: the KL divergence between the sampler's output and the target $K$-DPP decays as O(1/N) in the pool size. Empirically, PoolDPP extends the quality--diversity frontier on open-ended generation and the precision--recal
    
[^204]: 面向生成式稀疏对抗攻击可迁移性的密度反馈控制

    Density Feedback Control for Transferability of Generative Sparse Adversarial Attacks

    [https://arxiv.org/abs/2611.25001](https://arxiv.org/abs/2611.25001)

    提出密度反馈控制（DFC）方法，通过在每个训练轮次根据观测密度与目标密度的差距动态调整稀疏系数，使生成式稀疏对抗攻击无需重新调参即可在单次运行中精确达到目标扰动密度，并提升迁移攻击成功率。

    

    基于生成器的稀疏对抗攻击通过训练编码器-解码器网络来生成可迁移的稀疏扰动，用于黑盒鲁棒性评估。然而，其扰动密度只能通过一个固定的稀疏系数间接设定，该系数到密度的映射关系未知且在训练过程中会发生漂移，因此要达到目标密度需要反复重新调参；而且这种固定的压力一旦密度开始下降，几乎不再带来可迁移性方面的收益。我们提出密度反馈控制（DFC）方法，在每个训练轮次（epoch）结束时，根据观测到的硬掩码密度与目标密度之间的差距来收紧或放宽该稀疏系数，而无需改变生成器的架构或损失函数。在10%的目标密度下，DFC使三种基础攻击的平均密度在单次运行中与目标的偏差控制在一个百分点以内，并提升了其中两种攻击在CNN和Transformer目标模型上的迁移攻击成功率（ASR），包括在逐图像（原文在此处截断）……

    arXiv:2611.25001v1 Announce Type: new  Abstract: Generator-based sparse adversarial attacks train encoder--decoder networks to produce transferable sparse perturbations for black-box robustness evaluation. Their perturbation density, however, is set only indirectly by a fixed sparsity coefficient whose mapping to density is unknown and shifts during training, so reaching a target density requires re-tuning; this fixed pressure also yields little transferability gain once density reduction begins. We propose Density Feedback Control (DFC), which tightens or relaxes this coefficient at the end of each epoch according to the gap between the observed hard-mask density and the target, without changing the generator architecture or loss. At a 10\% target density, DFC brings the mean density of three base attacks within one percentage point of the target in a single run and improves the transfer attack success rate (ASR) of two of them on CNN and Transformer targets, including under per-image
    
[^205]: 注意你在哪里线性化：预测与校正模型合并中的干扰

    Watch Where You Linearize: Predicting and Correcting Model-Merging Interference

    [https://arxiv.org/abs/2611.25000](https://arxiv.org/abs/2611.25000)

    该论文发现，一阶方向导数只有在接收模型处评估才能有效预测模型合并中的干扰——可将基线准确率误差减半，而在共享预训练主干处评估则几乎无法预测干扰。

    

    模型合并通过将各自微调得到的任务向量加到共享主干网络上，从而组合多个分别微调的模型。目前对干扰的评估通常只能在构建并测试候选合并之后进行，这使得系数搜索的成本很高。我们证明，任务A的输出沿任务B向量方向的一阶方向导数可以预测干扰，但前提是必须在合适的模型处进行评估。在接收模型处，该导数能够逐任务、逐样本地预测邻近的合并结果，将无变化基线的准确率误差减半，并识别出大多数预测发生翻转的样本。对于完整的多任务合并，单次评估是不够的，需要沿合并路径进行两次评估。然而，在共享的预训练主干处，该导数在我们的视觉实验中与实际发生的变化几乎正交，在LoRA语言模型的初步实验中也仅具有部分预测能力。控制实验将这种失败与沿接收模型的位移联系起来。

    arXiv:2611.25000v1 Announce Type: new  Abstract: Model merging combines separately fine-tuned models by adding their task vectors to a shared backbone. Interference is typically assessed only after constructing and evaluating a candidate merge, making coefficient search expensive. We show that a first-order directional derivative of task A's outputs along task B's vector can predict interference, but only when evaluated at the appropriate model.   At the receiver model, this derivative forecasts nearby merges per task and per example, halves the accuracy error of a no-change baseline, and identifies most examples whose predictions flip. For full multi-task merges, a single evaluation is insufficient; two evaluations along the merge path are needed. At the shared pretrained backbone, however, the derivative is nearly orthogonal to realized changes in our vision experiments and only partly predictive in LoRA language-model pilots. Controls link this failure to displacement along the rece
    
[^206]: PsyWorld：面向人类行为预测的画像条件化世界模型

    PsyWorld: Profile-Conditioned World Modeling for Human Behavior Prediction

    [https://arxiv.org/abs/2611.24999](https://arxiv.org/abs/2611.24999)

    该论文提出PsyWorld，一种画像条件化的状态世界模型，将实际观测到的环境—行为状态转移作为结构化监督信号来学习行为表示，从而摆脱了传统上直接将静态人格画像映射为动作的做法，实现了更准确的人类行为预测。

    

    理解稳定的心理测量特质与随时间变化的状态如何共同影响行为，是计算行为科学中的一个重要问题。现有的个性化智能体通常直接基于静态画像对动作进行条件化，而未将实际发生的状态转移作为显式的监督信号来源加以利用。我们提出了一种画像条件化的状态世界模型，利用观测到的环境—行为状态转移来学习融合转移信息的行为预测表示。在训练过程中，后验编码器对实际发生的状态转移进行量化，而先验模型则根据心理测量画像和转移前上下文预测相应的编码。随后，先验编码与预测的下一状态共同用于条件化宏观层面的行为以及基于模拟器的行为执行。由此，该世界模型提供了结构化的转移监督，而不是将人格直接作为动作的描述符。我们在VEET（一个协同……）中对所提出的框架进行了评估。

    arXiv:2611.24999v1 Announce Type: new  Abstract: Understanding how stable psychometric traits and time-varying states jointly inform behavior is an important problem in computational behavioral science. Existing personalized agents commonly condition actions directly on a static profile, leaving realized state transitions unused as an explicit source of supervision. We introduce a profile-conditioned state world model that uses observed environment--behavior state transitions to learn transition-informed representations for behavioral prediction. During training, a posterior encoder quantizes realized transitions, while a prior predicts the resulting codes from the psychometric profile and pre-transition context. The prior code and predicted next state then condition macro-level behavior and simulator-grounded execution. Thus, the world model provides structured transition supervision instead of treating personality as a direct action descriptor. We evaluate the framework in VEET, a co
    
[^207]: 基于Kaplan–Meier生存估计的多臂老虎机方差自适应置信上界算法

    Variance-Adaptive Upper Confidence Bounds for Multi-Armed Bandits with Kaplan--Meier Survival Estimation

    [https://arxiv.org/abs/2611.24988](https://arxiv.org/abs/2611.24988)

    该论文提出了KM-UCB，一种基于Greenwood方差公式和经验Bernstein界限的方差自适应多臂老虎机策略，将连续时间生存分析与序贯决策相结合以处理右删失数据下的异方差问题。

    

    我们研究了随机多臂老虎机问题，其中与每个臂相关联的奖励是在预先指定时间范围内的生存概率，而观测到的结果是一个右删失的事件时间变量。在这种设置下，标准老虎机算法面临延迟和不完整的观测。先前针对Kaplan–Meier (KM) 估计器的改进方法依赖于Hoeffding类型的界限，忽略了连续生存过程固有的异方差性。为弥合这一差距，我们提出了KM-UCB，一种方差自适应策略，将连续时间生存分析与离散序贯决策相结合。通过利用鞅理论和连续影响函数，我们推导出了以Greenwood方差公式为参数的固定队列经验Bernstein界限。对于实时分配，我们保留Greenwood指数，并使用来自已完成记录的生存界限来审查所推荐的臂。在追赶式报告机制下……

    arXiv:2611.24988v1 Announce Type: cross  Abstract: We study the stochastic multi-armed bandit problem in which the reward associated with each arm is the survival probability at a pre-specified time horizon, and the observed outcome is a right-censored time-to-event variable. Standard bandit algorithms face delayed and partial observations in this setting. Prior adaptations for the Kaplan--Meier (KM) estimator rely on Hoeffding-type bounds that ignore the inherent heteroscedasticity of continuous survival processes. To bridge this gap, we propose KM-UCB, a variance-adaptive policy that unites continuous-time survival analysis with discrete sequential decision-making. By leveraging martingale theory and continuous influence functions, we derive a fixed-cohort empirical Bernstein bound parameterized by Greenwood's variance formula. For realtime allocation, we retain the Greenwood index and use a survival bound from completed records to review the recommended arm. Under catch-up reporting
    
[^208]: 具有信息性弃权的风险最优主动学习

    Risk-optimal active learning with informative abstention

    [https://arxiv.org/abs/2611.24986](https://arxiv.org/abs/2611.24986)

    该论文提出了一种在预言机可能弃权且响应概率与分类器共享参数情形下的主动学习查询获取准则，通过结合调整冗余参数后的分类器信息与多类零一风险的局部曲率，实现首阶渐近超额风险的最小化。

    

    我们研究了在预言机可能弃权且其响应概率与分类器共享参数的情形下的标签获取问题。此时，一次查询可以通过返回的标签和响应结果两条途径提供信息。在正确设定的参数模型下，我们推导出对响应模型冗余参数进行调整后的分类器信息。将该信息与多类零一风险的局部曲率相结合，得到了一个用于最小化首阶渐近超额风险的查询获取准则。我们刻画了固定查询预算下的最优分配方案，并展示了查询分配如何影响响应模型的可识别性。当响应概率不依赖于分类器参数时，该预言机准则退化为响应折扣式获取。仿真实验考察了由此得到的查询分配，并通过消融实验分离出响应结果项的贡献。此外还基于消费者投诉数据回放进行了实验验证。

    arXiv:2611.24986v1 Announce Type: cross  Abstract: We study label acquisition when an oracle may abstain and its response probability shares parameters with the classifier. A query can then provide information through both a returned label and the response outcome. Under a correctly specified parametric model, we derive the classifier information after adjustment for response-model nuisance parameters. Combining this information with the local curvature of multiclass zero-one risk gives an acquisition criterion for minimizing the leading asymptotic excess risk. We characterize the optimal allocation under a fixed query budget and show how query allocation affects response identifiability. The oracle criterion reduces to response-discounted acquisition when response probability does not depend on the classifier parameters. Simulations examine the resulting query allocations and isolate the contribution of the response-outcome term through an ablation. A Consumer Complaints replay retain
    
[^209]: DL101：神经网络模型的表示能力

    DL101 The Representation Capability of Neural Network Models

    [https://arxiv.org/abs/2611.24980](https://arxiv.org/abs/2611.24980)

    本文系统梳理了通用近似定理这一神经网络表示能力的数学基石，指出非多项式激活函数是实现通用逼近的必要条件，并强调这些定理本身并不提供关于效率、训练收敛与泛化能力的保证。

    

    arXiv:2611.24980v1 公告类型：cross 摘要：通用近似定理（Cybenko、Funahashi、Hornik 与 Leshno）为神经网络为何足够强大、能够建模复杂非线性现象提供了数学依据。浅层网络可以逼近紧集上的任意连续函数。Leshno 定理阐明了非多项式激活函数对于实现通用性是必要的，这解释了为什么 ReLU、逻辑斯蒂（sigmoid）、双曲正切以及其他常见激活函数是有效的（而纯多项式激活则不行）。然而，逼近定理并未提供效率、训练或泛化方面的任何保证。它们既没有说明为达到给定精度需要多少神经元，也没有说明梯度下降能否找到正确的参数，或者在有限数据上训练的网络是否能够良好地泛化。Barron 为特定函数类提供了定量的误差界（即对傅里叶范数有界的函数给出了逼近速率），而 Montúfar 则是第一个证明……（摘要到此截断）

    arXiv:2611.24980v1 Announce Type: cross  Abstract: Universal approximation theorems (Cybenko, Funahashi, Hornik & Leshno) provide the mathematical justification for why neural networks are powerful enough to model complex, nonlinear phenomena. Shallow networks can approximate any continuous function on compact sets. Leshno's theorem clarified that non-polynomial activations are necessary for universality, which explains why ReLU, logistic, hyperbolic tangent, and other common activations work (but pure polynomials do not).   However, approximation theorems provide no efficiency, training, nor generalization guarantees. They do not indicate how many neurons are needed for a given accuracy, whether gradient descent will find the right parameters, or whether networks trained on finite data will generalize well.   Barron provided quantitative error bounds for particular function classes (approximation rates for functions with bounded Fourier norm), whereas Mont\'ufar was the first to show 
    
[^210]: 利用太阳敏感器与隐式表示改进星载形状模型

    Improving On-board Shape Models Using Sun Sensors and Implicit Representations

    [https://arxiv.org/abs/2611.24976](https://arxiv.org/abs/2611.24976)

    该方法仅利用太阳敏感器日食掩蔽数据与可微分神经模型EclipseNets，通过精化二维轮廓并监督周期性激活的神经符号距离场，实现小行星等不规则小天体星载三维形状模型的端到端生成与改进，将轮廓IoU从0.70提升至0.87。

    

    我们提出了一种原创的端到端方法，仅利用低成本的太阳敏感器日食掩蔽数据和粗糙的先验知识，来生成和改进不规则小天体（如小行星）的星载三维形状模型，专为资源受限的任务（如立方星）量身定制。借助EclipseNets——一个可微分的日食边界神经模型，我们的流程首先通过学习航天器轨道运行期间的实时进入和退出事件来精化二维轮廓，从而避免了传统成像或激光雷达在带宽匮乏环境中的局限性。随后，这些轮廓监督一个具有周期性激活的紧凑神经符号距离场，实现对高保真隐式几何表示的增量更新，能够捕捉细尺度特征和凹陷结构。在真实轨道动力学条件下对小行星Eros的仿真中，该方法将轮廓IoU从0.70提升至0.87，并产生连贯的三维重建结果。

    arXiv:2611.24976v1 Announce Type: cross  Abstract: We propose an original, end-to-end approach to generate and refine the onboard 3D shape model of irregular small bodies, such as asteroids, using only low-cost Sun sensor eclipse data and coarse a priori knowledge, tailored for resource-constrained missions like CubeSats. Leveraging EclipseNets, a differentiable neural model of eclipse boundaries, our pipeline first refines 2D silhouettes by learning from real-time entry and exit events during spacecraft orbits, avoiding the limitations of traditional imaging or lidar in bandwidth-starved environments. These silhouettes then supervise a compact neural signed-distance field with periodic activations, enabling incremental updates to a high-fidelity implicit geometry representation capable of capturing fine-scale features and concavities. In simulations of the body Eros under realistic orbital dynamics, the method increases silhouette IoU from 0.70 to 0.87 and produces coherent 3D reconst
    
[^211]: 何时需要程序结构？基于证书的程序归纳难度校准

    When Is Program Structure Required? Certificate-Grounded Calibration of Program-Induction Difficulty

    [https://arxiv.org/abs/2611.24975](https://arxiv.org/abs/2611.24975)

    该论文提出“基于证书的校准”方法，通过完整的单规则枚举和可执行的双规则见证来严格证明程序归纳任务究竟需要一条还是两条规则，实验表明经认证的分离任务使模型完成率显著下降而未见见证对的成功率反而上升，从而能更可靠地区分程序归纳基准的真实难度与模型能力。

    

    程序归纳基准只有在示例确实需要其旨在测试的结构时才具有信息量。然而，双规则生成器可能产生单规则也能解释的演示。我们提出了基于证书的校准方法，该方法在一个有界字符串替换语言中构建最小解长度分别为一和二的匹配任务。在模型推理之前，通过完整的单规则枚举和可执行的双规则见证来认证这种分离。我们评估了120个包含三到六个共享示例的参考族，使用了1,440个新模型轨迹和480个已保存的轨迹。在Qwen3-8B和Olmo-3-7B-Think两个模型上，在经认证的分离任务上，所显示任务的完成率降低了23.53-45.83个百分点，而在完全未展示的见证对上的成功率反而提高了8.40-13.45个百分点。这些见证收益伴随着模型在均匀采样字符串上一致性的下降，以及在同时激活两条参考规则的输入上仅有1.93-5.78%的一致性。

    arXiv:2611.24975v1 Announce Type: cross  Abstract: A program-induction benchmark is informative when its examples require the structure it aims to test. Yet a two-rule generator can produce demonstrations that a single rule also explains. We introduce certificate-grounded calibration, which constructs matched tasks with minimum solution lengths of one and two in a bounded string-replacement language. Complete one-rule enumeration and an executable two-rule witness certify the change before model inference. We evaluate 120 reference families with three to six shared examples, using 1,440 new model trajectories and 480 saved trajectories. Across Qwen3-8B and Olmo-3-7B-Think, displayed-task completion is 23.53-45.83 percentage points lower on certified separation tasks, while success on a wholly unshown witness pair is 8.40-13.45 points higher. These witness gains accompany lower agreement on uniformly sampled strings and 1.93-5.78% agreement on inputs that activate both reference rules. 
    
[^212]: ICARE：基于增量类别感知表示与蒸馏学习的跨域文本分类

    ICARE: cross-domain text classification with incremental class-aware representation and distillation learning

    [https://arxiv.org/abs/2611.24959](https://arxiv.org/abs/2611.24959)

    提出ICARE框架，通过基于表示学习的实例选择模块和知识蒸馏方法，解决跨域文本分类中自然语言推理系统的灾难性遗忘与过拟合问题。

    

    强大的微调大型语言模型（LLM）的兴起使得稳健的自然语言推理（NLI）系统成为必需。这些NLI系统理想情况下应在各种任务上进行训练，涵盖情感分析、新闻分类和指代消解等多样化概念。然而，这种持续的、多任务的处理方式可能导致灾难性遗忘，即先前学习的任务被丢失，尤其是在缺乏特定标签的领域。使用带标签的过去示例进行重复训练可能因导致过拟合而进一步加剧这一问题，阻碍系统分类新类别的能力。为了应对这些挑战，本研究提出了ICARE，这是一个具有两个关键组件的新颖框架：实例选择模块和知识蒸馏方法。前者基于表示学习，精心选择呈现给NLI系统的训练示例。后者是基于特征和……的知识蒸馏方法（摘要在此处被截断）。

    arXiv:2611.24959v1 Announce Type: new  Abstract: The rise of powerful, fine-tuned large language models (LLMs) necessitates robust natural language inference (NLI) systems. These NLI systems ideally train across various tasks, encompassing diverse concepts like sentiment analysis, news classification, and coreference resolution. However, this continual, multi-tasking approach can lead to catastrophic forgetting, where previously learned tasks are lost, especially in domains lacking specific labels. Rehearsing past examples with labels can further exacerbate this issue by causing overfitting, hindering the system's ability to classify new categories. To tackle these challenges, this research proposes ICARE, a novel framework with two key components: an instance selection module and a knowledge distillation method. The former is based on representation learning, which meticulously selects training examples to present to the NLI system. The latter, knowledge distillation with feature and 
    
[^213]: LLM复杂度：迈向大语言模型推理的计算复杂度视角

    LLM Complexity: Towards a Computational Complexity View of Large Language Model Inference

    [https://arxiv.org/abs/2611.24943](https://arxiv.org/abs/2611.24943)

    该论文证明LLM推理时的token使用量与经典时间复杂度并非单调对应关系，并提出“LLM复杂度”这一新指标，通过准确率-token曲线下的面积从LLM的视角量化问题复杂度。

    

    计算复杂度理论为刻画计算问题的效率提供了一个原则性框架，而大语言模型（LLM）同样可以被视作执行“输入—推理时计算—输出”这一结构化过程的计算系统。在此背景下，生成token的数量成为衡量推理时计算量的一个自然且可直接观测的代理指标。然而，我们证明了LLM token长度与经典时间复杂度之间这一颇具吸引力的类比并不成立。我们否定了“推理时token使用量与经典时间复杂度单调一致”这一假设，并展示了其根本性分歧源于LLM在推理过程中实际分配计算的方式。为此，我们提出了LLM复杂度（LLM Complexity）这一新指标，通过刻画准确率-token曲线所界定的面积，从LLM的视角量化问题复杂度。我们发现……

    arXiv:2611.24943v1 Announce Type: cross  Abstract: Computational complexity theory offers a principled framework for characterizing the efficiency of computational problems, and Large Language Models (LLMs) can similarly be viewed as computational systems executing a structured process of input, inference-time compute, and output. In this context, the number of generated tokens emerges as a natural and directly observable proxy for inference-time compute. However, we show that the tempting analogy between LLM token length and classical time complexity does not hold. We disprove the hypothesis that inference-time token usage aligns monotonically with classical time complexity, demonstrating a fundamental divergence grounded in how LLMs actually allocate computation during inference. To this end, we introduce LLM Complexity, a new metric designed to quantify problem complexity from the perspective of LLMs by characterizing the area bounded above by the accuracy-token curve. We find that 
    
[^214]: 面向样本高效在线A/B测试的方差最优反事实策略设计

    Variance-Optimal Counterfactual Policy Design for Sample-Efficient Online A/B Testing

    [https://arxiv.org/abs/2611.24757](https://arxiv.org/abs/2611.24757)

    该论文提出一种方差最优的反事实策略设计方法，通过联合优化动作的采样方式，将流量引导至两种策略不一致的动作上，从而显著降低在线A/B测试的方差并提升样本效率。

    

    在线A/B测试是在部署前评估面向用户系统变更（例如新的推荐或排序策略）的标准方法。虽然标准A/B测试的均值差估计量是无偏的，但它通常存在高方差问题。当处理策略与生产策略以相近的概率选择相同的动作时，由此产生的观测数据会增加噪声，却几乎不提供关于两种策略差异的信息。因此，A/B测试往往需要大量流量和较长周期才能检测到较小的处理效应。现有的方差缩减方法主要是在数据收集方案固定之后改进估计量，而近年来的重叠感知设计仅优化两种策略之间的单一流量分配，因而无法将流量引导到两种策略不一致的动作上。在本文中，我们联合优化动作的采样方式……

    arXiv:2611.24757v1 Announce Type: new  Abstract: Online A/B testing is the standard approach for evaluating changes to user-facing systems, such as a new recommendation or ranking policy, before deployment. While the difference-in-means estimator of a standard A/B test is unbiased, it often suffers from high variance. When the treatment and production policies choose the same actions with similar probabilities, the resulting observations can add noise while providing little information about the difference between the policies. Consequently, A/B tests can require large amounts of traffic and long durations to detect small treatment effects. Existing variance-reduction methods mainly improve the estimator after the data-collection scheme has been fixed, and recent overlap-aware designs only optimize a single traffic split between the two policies; thereby, they cannot direct traffic to the actions on which the policies disagree.   In this paper, we jointly optimize how actions are sampl
    
[^215]: 紧凑视觉骨干网络中全局上下文与权重共享的适用位置

    Where Global Context and Weight Sharing Work in Compact Vision Backbones

    [https://arxiv.org/abs/2611.23446](https://arxiv.org/abs/2611.23446)

    该研究发现，在EfficientNet-B0骨干网络的P3、P4、P5阶段插入轻量级零门控残差自注意力适配器可将ImageNet-1k精度提升至74.02%，而单纯用BiFPN多尺度特征金字塔包装骨干则无法超越普通MBConv主干。

    

    紧凑型卷积骨干网络（如EfficientNet）通过局部的深度可分离卷积实现了出色的参数效率，但混合设计中仍有两个核心的结构性问题：全局自注意力在何处能超越纯卷积以提升识别性能，以及重复的模块能否在不损失表征能力的前提下共享权重？我们在ImageNet-1k（5万张图像的验证集）上通过受控的EfficientNet-B0实验对这两个问题进行了研究。首先，我们表明仅将分类骨干网络嵌入多尺度特征金字塔（BiFPN）中无法超越普通的MBConv主干（69.99%–71.81% vs. 普通B0的72.86%），直到配合阶段注意力后才有所提升（73.15%）。相比之下，保持MBConv主干不变，并在P3（28×28窗口式）、P4（14×14全局）和P5（7×7全局）阶段插入轻量级的零门控残差自注意力适配器，可独立达到74.02%的准确率，并优于逐位匹配配置的普通B0基线。

    arXiv:2611.23446v1 Announce Type: cross  Abstract: Compact convolutional backbones such as EfficientNet achieve strong parameter efficiency through local depthwise separable convolutions, yet two structural questions remain central to hybrid design: where does global self-attention improve recognition over pure convolution, and can repeated blocks share weights without losing representational capacity? We investigate both questions on ImageNet-1k (50,000-image validation) using controlled EfficientNet-B0 experiments. First, we show that wrapping a classification backbone in multi-scale feature pyramids (BiFPN) fails to outperform a plain MBConv trunk (69.99% - 71.81% vs. 72.86% plain B0) until paired with stage attention (73.15%). In contrast, keeping the MBConv trunk intact and inserting lightweight zero-gated residual self-attention adapters at stages P3 (28x28 windowed), P4 (14x14 global), and P5 (7x7 global) reaches 74.02% standalone and outperforms a bit-for-bit matched plain-B0 c
    
[^216]: 为什么亚4比特视觉CNN会崩溃以及混合精度恢复的极限在哪里

    Why Sub-4-Bit Vision CNNs Collapse and Where Mixed-Precision Recovery Stops

    [https://arxiv.org/abs/2611.23445](https://arxiv.org/abs/2611.23445)

    本研究系统剖析了视觉CNN在4比特以下量化时崩溃的机理，发现崩溃高度非线性且集中在后期逐点投影和深度可分离层，并提出了结构化的多步骤混合精度恢复瀑布策略来挽救精度。

    

    在资源受限环境中运行的深度视觉模型严重依赖低比特量化，然而将权重压缩到4比特以下往往会引发灾难性的精度崩溃。尽管混合精度恢复策略旨在通过选择性地保留高精度参数来挽救精度，但这些模型在哪里以及为何崩溃的基本机制仍鲜为人知。在本研究中，我们基于完整的50,000张图像的ImageNet-1k验证集，系统分析了标准视觉CNN（包括ResNet和EfficientNet骨干网络）量化崩溃的机理。首先，我们绘制了性能退化的确切进程，证明崩溃是高度非线性的，并集中于后期的逐点投影层和深度可分离层。其次，我们引入了一个结构化的多步骤恢复瀑布方法，评估选择性整数精度（INT4/INT8）和选择性浮点缩放的应用效果。

    arXiv:2611.23445v1 Announce Type: new  Abstract: Deep vision models operating in resource-constrained environments depend heavily on low-bit quantization, yet compressing weights below 4 bits frequently induces catastrophic accuracy collapse. While mixed-precision recovery strategies aim to rescue accuracy by selectively retaining high-precision parameters, the fundamental mechanics of where and why these models collapse remain poorly understood. In this study, we systematically analyze the anatomy of quantization collapse across standard Vision CNNs (including ResNet and EfficientNet backbones) evaluated on the full 50,000-image ImageNet-1k validation set. First, we map the exact progression of degradation, demonstrating that collapse is highly non-linear and concentrated in late-stage pointwise projections and depthwise layers. Second, we introduce a structured, multi-step recovery waterfall that evaluates selective integer precision (INT4/INT8) and selective floating-point scales. W
    
[^217]: The K=V Cache of Gemma 4 Only Needs to Be Stored Once: Achieving Exact Global-KV Memory Savings in a Paged Serving Engine Without Modifying the Attention Kernel

    Storing the Gemma 4 K=V Cache Once: Exact Global-KV Memory Savings in a Paged Serving Engine with Unchanged Attention Kernels

    [https://arxiv.org/abs/2611.23035](https://arxiv.org/abs/2611.23035)

    This paper exploits the linear dependence between the key and value in the global attention layers of Gemma 4 31B (K=V), storing only one 640-value row without modifying the attention kernel, which reads the key and value caches through two overlapping strided views, achieving a 1.34 to 1.40 fold reduction in KV cache memory during decoding with bit-identical outputs.

    

    In the global attention layers of Gemma 4 31B, the value is the normalized key projection, and the key is the value multiplied by a fixed weight, with rotary position encoding applied to only one quarter of its dimensions. The Gemma 4 technical report points out that this design can reduce the global KV cache by up to 37.5%, but the open-source serving engines we inspected still store a complete key and a complete value: 1024 values per KV head per token. We instead store only one row of 640 values, which is exact in real-number arithmetic, and an unmodified paged attention kernel reads it as its key cache and value cache through two overlapping strided views. In a vLLM prototype using the 16-bit 31B model, the outputs are bit-identical to an emulation of the same arithmetic on the original cache; 2 of 432 final answers differ from the original engine (within its run-to-run noise range); and at 32K to 64K tokens, the steady decoding KV memory per request decreases by a factor of 1.34 to 1.40. On a pair of prefill/decode...

    arXiv:2611.23035v1 Announce Type: cross  Abstract: In the global attention layers of Gemma 4 31B, the value is the normalized key projection, and the key is the value scaled by a fixed weight, with rotary position encoding on one quarter of its dimensions. The Gemma 4 report states that this design reduces the global KV cache by up to 37.5%, but the open serving engines that we checked still store a full key and a full value: 1,024 values per KV head and token. We store one 640-value row instead, exact in real arithmetic, and an unchanged paged attention kernel reads it through two overlapping strided views as its key cache and its value cache. In a vLLM prototype with the 16-bit 31B model, the outputs are bit-identical to an emulation of the same arithmetic on the stock cache, 2 of 432 final answers differ from the stock engine (within its run-to-run noise), and the steady decode KV memory per request falls by a factor of 1.34 to 1.40 at 32K to 64K tokens. On a prefill/decode pair of 
    
[^218]: 所有测量、所有方向、一次完成：面向分子与光谱的多模态流匹配

    Every Measurement, Every Direction, All at Once: Multimodal Flow Matching for Molecules and Spectra

    [https://arxiv.org/abs/2611.22098](https://arxiv.org/abs/2611.22098)

    本文提出MOSAIC，一个多模态流匹配Transformer模型，通过对测量数据施加随机掩码，使分子三维结构与UV/IR/拉曼光谱之间实现任意到任意的双向生成，并在QM9S上达到88.8%的分子图恢复率，证明了单次实验可被充分挖掘为多方向的监督信号。

    

    科学领域的生成式建模常常受到数据限制，因为实验成本高昂。然而，每次实验可能对同一物理实体产生多个相互关联的测量结果。标准的生成模型通常只在单一方向上训练，丢弃了其余方向的监督信息。我们认为，每次实验所包含的训练信号远超这种不对称性允许模型利用的程度：随机掩码使测量结果既可以作为条件输入，也可以作为预测目标，从而从单次实验中获得多个有监督的去噪任务。我们通过MOSAIC来验证这一前提，MOSAIC是一个覆盖分子三维结构以及紫外、红外和拉曼光谱的多模态流匹配Transformer模型。MOSAIC是一个能够进行任意到任意生成的单一模型：即从其他模态的任意不相交子集生成任意模态子集。在QM9S数据集上，MOSAIC实现了88.8%的分子图恢复率，超越了最强的光谱到结构（模型）（摘要原文在此处截断）。

    arXiv:2611.22098v1 Announce Type: new  Abstract: Generative modeling in science is often data-limited, where experiments are expensive. However, each experiment may yield several coupled measurements of the same physical entity. Standard generative models often train only on a single direction, discarding supervision from the rest. We argue that each experiment contains more training signal than this asymmetry allows the model to exploit: random masking enables measurements to serve as either conditioning inputs or prediction targets, yielding multiple supervised denoising tasks from a single experiment. We investigate this premise with MOSAIC, a multimodal flow matching transformer over molecular 3D structure and UV, IR, and Raman spectra. MOSAIC is a single model that is capable of any-to-any generation: generating any subset of modalities from any disjoint subset of the others. On QM9S, MOSAIC achieves 88.8% molecular graph recovery, outperforming the strongest spectra-to-structure 
    
[^219]: 思维链可控性可能降低思维链可监控性

    Chain-of-Thought Controllability Can Reduce Chain-of-Thought Monitorability

    [https://arxiv.org/abs/2611.21799](https://arxiv.org/abs/2611.21799)

    论文通过人设控制训练发现，即使训练中不含任何规避监控的示例，仅仅让模型学会以不同人设口吻书写推理，就能显著降低其在执行隐蔽副任务时思维链的可监控性（g-mean²从0.94降至0.22），表明思维链可控性的增强可能削弱思维链监控的有效性。

    

    思维链监控是一种很有前景的工具，可用于检测AI智能体中的隐藏目标、偏离任务行为及其他故障。其中一个风险是，模型可能会刻意塑造自己的推理过程，使其对监控者不透明。我们通过“人设控制训练”对这一风险进行了压力测试：在用多样化人设口吻改写的数学推理轨迹上进行监督微调。尽管训练数据中不包含任何规避监控的示例，这仍诱导出了更广泛地控制思维链的能力。在隐蔽副任务评估中，gpt-oss-20b模型体完成隐藏副任务的那些思维链轨迹，所包含的可供监控的证据比基础模型的更少。在各项评估与不同思维链长度上按副任务成功次数加权平均后，思维链可监控性指标g-mean²从基础模型的0.94下降至模型体在无隐藏指令情况下的0.68，而在明确被指示隐藏时进一步降至0.22。在干预评估中……（原文摘要在此处截断）

    arXiv:2611.21799v1 Announce Type: new  Abstract: Chain-of-thought (CoT) monitoring is a promising tool for detecting hidden goals, off-task behavior, and other failures in AI agents. One risk is that models could deliberately shape their reasoning to make it opaque to monitors. We stress-test this risk through persona-control training: supervised fine-tuning on mathematical reasoning traces rewritten in the voices of diverse personas. This induces a broader ability to control CoT, even though training contains no monitor-evasion examples. In covert side-task evaluations, traces in which a gpt-oss-20b model organism completes the hidden side task contain less monitorable evidence than the base model's. Averaged across evaluations and CoT lengths, weighted by side-task-success counts, the CoT monitorability metric g-mean$^2$ falls from 0.94 for the base model to 0.68 for the model organism without a hiding instruction, and to 0.22 when explicitly instructed to hide. In intervention evalu
    
[^220]: 具有受控约束复杂度的合成神经符号任务生成

    Synthetic Neurosymbolic Task Generation with Controlled Constraint Complexity

    [https://arxiv.org/abs/2611.19597](https://arxiv.org/abs/2611.19597)

    本文提出NeSySy，一个可对约束结构和复杂度等关键属性进行细粒度控制的神经符号学习任务合成生成器，能够联合生成逻辑约束与一致数据集，从而系统性研究任务难度因素及约束感知学习方法的有效性边界。

    

    我们提出了NeSySy，一个用于神经符号学习任务的合成生成器，它能够联合生成逻辑约束和一致的数据集。与现有基准不同，NeSySy对诸如约束结构、紧致性、输入依赖性以及数据生成与对齐策略等关键属性提供了细粒度的控制。特别地，它允许在无限域上使用丰富的约束，并利用求解器来确保约束是有意义的，且生成的样本满足这些约束。我们设计NeSySy旨在同时支持基于声明式约束的学习和基于远程监督的学习。在一系列约束族上，我们提供了证据表明其参数会对任务难度产生系统性的且在很大程度上相互独立的影响。利用NeSySy，我们研究了约束感知的学习方法，识别出显式约束强制有益的情形以及表达性无约束（模型表现更优）的情形。

    arXiv:2611.19597v1 Announce Type: cross  Abstract: We introduce NeSySy, a synthetic generator for neurosymbolic learning tasks that jointly produces logical constraints and consistent datasets. Unlike existing benchmarks, NeSySy provides fine-grained control over key properties such as constraint structure, tightness, input dependence, and data generation and alignment strategies. In particular, it allows the use of rich constraints over infinite domains, and uses solvers to ensure that the constraints are meaningful and that generated samples satisfy the constraints.   We design NeSySy to support both learning with declarative constraints and learning from distant supervision. Across a range of constraint families, we provide evidence that its parameters induce systematic and largely independent effects on task difficulty. Using NeSySy, we study constraint-aware learning methods, identifying regimes where explicit constraint enforcement is beneficial and where expressive unconstrained
    
[^221]: SynthDCB：面向可扩展设备控制基准测试的世界蒸馏方法

    SynthDCB: World Distillation for Scalable Device Control Benchmarking

    [https://arxiv.org/abs/2611.18639](https://arxiv.org/abs/2611.18639)

    该论文提出SynthDCB基准，创新性地通过“世界蒸馏”方法将设备行为知识转化为模块化可执行组件，构建了覆盖家居、汽车和手机三大领域的可扩展设备控制与环境感知评测基准。

    

    扩展设备控制评估需要同时扩展智能体所交互的世界以及它们需要解决的任务。我们提出了SynthDCB，一个通过世界蒸馏构建的、可扩展的环境感知与设备控制基准。世界蒸馏将设备行为、交互约束和环境效应的知识转化为模块化的可执行组件，这些组件被组合成具有显式状态、动作和时间动态的交互式世界。统一的能力表示将工具发现、世界执行和任务合成连接起来，使新的能力能够通过共享的构建流水线进入评估。SynthDCB实例化了家居、汽车和手机三个领域，包含114个设备类别、1,166个功能和95个事件，支持五大任务族共32个子类型。这种有状态的设计支持通过状态正确性和工具调用轨迹进行任务验证与智能体评估。我们评估了……

    arXiv:2611.18639v1 Announce Type: cross  Abstract: Scaling device-control evaluation requires scaling both the worlds agents interact with and the tasks they must solve. We introduce SynthDCB, a scalable benchmark for environment awareness and device control built through world distillation. World distillation transforms knowledge of device behaviors, interaction constraints, and environmental effects into modular executable components, which are composed into interactive worlds with explicit states, actions, and temporal dynamics. A unified capability representation connects tool discovery, world execution, and task synthesis, allowing new capabilities to enter evaluation through a shared construction pipeline. SynthDCB instantiates Home, Car, and Phone domains with 114 device classes, 1,166 functions, and 95 events, supporting five task families with 32 subtypes. The stateful design enables task validation and agent evaluation through state correctness and tool-call traces. We evalua
    
[^222]: 视觉-语言-动作模型：设计、训练与部署的全生命周期综述

    Vision-Language-Action Models: A Life-Cycle Survey of Design, Training and Deployment

    [https://arxiv.org/abs/2611.18372](https://arxiv.org/abs/2611.18372)

    这是一篇系统性综述，沿着模型设计（输入与记忆、架构、动作输出、中间预测）、训练（预训练数据等）与部署的全生命周期七个维度，梳理了2022至2026年间88项视觉-语言-动作（VLA）模型研究，覆盖操作、导航、人形控制与世界-动作模型等方向。

    

    视觉-语言-动作（VLA）模型将网络预训练的视觉-语言模型适配为输出机器人动作，从而使网络知识得以迁移到物理控制中。在不到4年的时间里，该领域从首个VLA模型RT-2，发展到可在单张消费级GPU上微调的开源模型，再到能够清扫陌生住宅的专有通用模型。本综述回顾了2022年12月至2026年10月间的88项研究工作，主要聚焦于机器人操作与导航，并涵盖人形机器人控制、世界-动作模型与评估。我们沿着七个维度追踪VLA从模型设计到训练与部署的全生命周期：模型设计涵盖，输入与记忆，即模型观察什么；架构，即其流水线如何构建；动作输出，即它如何产生运动指令；中间预测，即它还预测什么，从推理步骤到未来观测。训练涵盖——预训练数据，即机器人、网络、人类与仿真数据的混合（注：原文摘要在此处截断）。

    arXiv:2611.18372v1 Announce Type: cross  Abstract: Vision-language-action (VLA) models adapt web-pretrained vision-language models to output robot actions, so that web knowledge carries over to physical control. In under 4 years the field has moved from RT-2, the first VLA, to open models that fine-tune on a single consumer GPU and to proprietary generalists that clean unseen homes. This survey reviews 88 works from December 2022 to October 2026, mainly on manipulation and navigation, plus humanoid control, world-action models and evaluation. We follow a VLA through its life cycle, from model design to training and deployment, along seven axes. Model design covers (a) inputs and memory, what the model observes; (b) architecture, how its pipeline is built; (c) action output, how it produces motor commands; and (d) intermediate predictions, what else it predicts, from reasoning steps to future observations. Training covers (e) pretraining data, the mix of robot, web, human and simulated 
    
[^223]: 分布式增材制造中跨未见几何形状的联邦物理信息热场预测

    Federated Physics-Informed Thermal Field Prediction Across Unseen Geometries in Distributed Additive Manufacturing

    [https://arxiv.org/abs/2611.17794](https://arxiv.org/abs/2611.17794)

    该论文提出Fed-PIGANO联邦学习框架，使各制造站点能够在不共享专有零件几何数据的前提下联合训练物理信息几何感知神经算子模型，从而在分布式增材制造中实现对未见几何形状的热场预测。

    

    在增材制造（AM）中，每次零件几何形状的改变都需要进行新的热仿真，这使得对大量候选设计的评估成本高昂。几何感知模型可以通过单一训练好的模型在未见过的零件上预测热场，从而降低这一成本，但这类模型假设多样化的几何形状数据可以集中获取。当每个制造站点仅持有有限的、无法共享的专有零件设计时，这一假设便具有很强的局限性。我们提出了Fed-PIGANO，一个联邦学习框架，它在各个站点独立训练物理信息几何感知神经算子（PI-GANO）模型，并在不共享零件几何数据的情况下聚合各站点的几何编码器和预测网络。我们在金属增材制造的二维稳态热基准上对该框架进行了评估，几何形状来自五个多边形族，分布于三个站点，并在从族组成相同到差异显著的四种场景下进行了测试。

    arXiv:2611.17794v1 Announce Type: new  Abstract: In additive manufacturing (AM), each change in part geometry requires a new thermal simulation, which makes design evaluation over many candidates costly. Geometry-aware models reduce this cost by predicting thermal fields on unseen parts with a single learned model, but they assume that diverse geometries are available centrally. This assumption is restrictive when each manufacturing site holds only a limited set of proprietary part designs that it cannot share. We propose Fed-PIGANO, a federated framework that trains physics-informed geometry-aware neural operator (PI-GANO) models at individual sites and aggregates their geometry encoders and prediction networks without sharing part geometries. We evaluate the framework on a 2D steady-state thermal benchmark for metal AM, with geometries from five polygonal families distributed across three sites under four scenarios that range from identical to strongly different family compositions. 
    
[^224]: 经典能量函数学会推理：比循环模型快1000倍

    Classical Energies Learn to Reason 1,000x Faster than Looped Models

    [https://arxiv.org/abs/2611.17309](https://arxiv.org/abs/2611.17309)

    提出基于 Hopfield & Tank 经典能量函数的学习版本 CRAM，通过能量下降实现推理，在数独和迷宫谜题上仅需毫秒级训练即可达到近乎完美的准确率，速度比循环模型快至少1000倍，推理计算量减少至少282倍。

    

    循环模型已成为推理任务的重要范式，在数独极限（Sudoku-Extreme）和迷宫唯一（Maze-Unique）等简单推理谜题上取得了成功。然而，这些模型可能不稳定、训练缓慢，并且需要大量参数和浮点运算（FLOPs）来解决简单谜题。基于能量下降的方法不仅能缓解这些问题，还带来了诸多优势：内置早停机制、计算量能自适应问题难度、无需对求解器进行求导的快速训练，以及能学习谜题约束的可解释权重。在此，我们通过 Hopfield & Tank 在上世纪80年代提出的能量函数的学习版本证明了这一点，我们将其命名为 CRAM。该方法在单个 GPU 上分别仅需 42 毫秒和 132 毫秒的训练，即可在数独极限测试集上达到 99.996% 的解题率，在迷宫唯一测试集上达到 100% 的解题率，比循环模型在这些任务上至少快 1000 倍。此外，CRAM 在数独极限任务上以至少少 282 倍的推理浮点运算量，达到了与基线循环模型最高准确率相当的水平。

    arXiv:2611.17309v1 Announce Type: new  Abstract: Looped models have emerged as a prominent paradigm for reasoning, with successes on simple reasoning puzzles such as Sudoku-Extreme and Maze-Unique. However, these models can be unstable, slow to train, and use many parameters and FLOPs to solve simple puzzles. Descent on an energy not only mitigates these issues but also offers advantages: built-in early stopping, compute that adapts to problem difficulty, fast training without differentiating through a solver, and interpretable weights that learn the constraints of the puzzle. Here we demonstrate this with a learned version of Hopfield & Tank's energy from the 80s, which we term CRAM, solves 99.996% of Sudoku-Extreme and 100% of Maze-Unique test puzzles after 42 ms and 132 ms of training on one GPU, at least 1,000x faster than looped models on these tasks. CRAM also matches baseline looped models at their highest accuracy on Sudoku-Extreme using at least 282x fewer inference FLOPs. Our
    
[^225]: LazyWeights：按需解码的边缘大语言模型

    LazyWeights: Decode-on-Demand Edge LLMs

    [https://arxiv.org/abs/2611.17108](https://arxiv.org/abs/2611.17108)

    LazyWeights在保留现有量化格式的基础上，通过对权重进行无损Huffman编码并在DRAM与L2之间用小型ASIC单元按需解码，在不改动GPU内核的前提下使边缘设备上的模型体积缩小5.7%-31.4%，并将每token的延迟和能耗最多降低约一半。

    

    内存决定了边缘设备能运行哪些大语言模型以及运行得多快：在4 GB的Jetson Nano上受容量限制，在8 GB的Jetson Orin Nano Super上则受带宽限制。现有部署格式（GPTQ、AWQ、llama.cpp的K量化和I量化）对矩阵的每个整数编码分配相同的位数，尽管这些编码的分布远非均匀。LazyWeights保留了这些格式，但通过调整GPTQ使编码更易压缩，在交错通道中对编码进行无损Huffman编码，并在DRAM与L2缓存之间的一个小型单元中按需解码，因此GPU内核无需任何改动，DRAM只需传输编码后的字节。在1%和3.5%的困惑度预算下，编码后的Phi-3-mini、Mistral-7B和SmolLM-1.7B比最佳的llama.cpp格式小5.7%-31.4%。一个位精确的0.21 mm² ASIC解码单元（1 GHz，7 nm工艺）将Orin上每个token的延迟和能耗最多降低36%和38%（结合prompt-lookup投机解码时可降低54%和44%）。在Nano上，LazyWeights每token所需时间比CPU版llama.cpp减少58%-64%，能耗减少53%-57%，并能运行……

    arXiv:2611.17108v2 Announce Type: replace-cross  Abstract: Memory decides which LLMs an edge device runs, and how fast: capacity on a 4 GB Jetson Nano, bandwidth on an 8 GB Jetson Orin Nano Super. Deployed formats (GPTQ, AWQ, llama.cpp's K- and I-quants) spend equal bits on every integer code of a matrix, although the codes are far from uniform. LazyWeights keeps these formats but tunes GPTQ so the codes compress, Huffman-codes them losslessly in interleaved lanes, and decodes them on demand in a small unit between DRAM and L2, so GPU kernels run unchanged and DRAM moves only coded bytes. At 1% and 3.5% perplexity budgets, coded Phi-3-mini, Mistral-7B and SmolLM-1.7B are 5.7-31.4% smaller than the best llama.cpp format. A bit-exact 0.21 mm^2 ASIC decoder unit (1 GHz, 7 nm) cuts Orin latency and energy per token by up to 36% and 38% (54% and 44% with prompt-lookup speculation). On the Nano, LazyWeights needs 58-64% less time and 53-57% less energy per token than CPU llama.cpp, and runs 
    
[^226]: 因果性何时有助于多变量时间序列异常检测？

    When Does Causality Help in Multivariate Time-Series Anomaly Detection?

    [https://arxiv.org/abs/2611.15073](https://arxiv.org/abs/2611.15073)

    该研究通过固定因果引导自编码器并系统改变图质量、异常类型和评估协议，揭示了因果性在多变量时间序列异常检测中的三个有效条件：图质量决定收益大小、异常必须破坏条件结构（严重度≥0.3）才有效，且只有范围感知的VUS评估协议才能真实反映因果模块的性能优势。

    

    因果感知评分在多变量时间序列异常检测中很常见，但文献很少说明它何时有效。在保持一个因果引导的自编码器（DeCauAE）不变的情况下，我们相对于真实标准系统地改变图质量、异常类型和评估协议。图质量决定收益：在一个二十变量的有向无环图（DAG）上，相对于无图自编码器的增益随着图被破坏而衰减。异常类型决定目标：在异常破坏条件结构的场景中，结构因子将窗口F1从0.800提升到0.976，而当破坏严重度s低于0.3时其优势消失。评估协议决定排名：点调整F1会让随机评分也能匹配完整模型，而范围感知的VUS则能将自编码器家族与经典基线区分开来。在真实基准上，因果模块只有在系统密度符合条件且跨子样本可重复性能选出相应实体时才有回报；在所有地方启用它需付出0.012 VUS-ROC的代价。

    arXiv:2611.15073v2 Announce Type: replace  Abstract: Causality-aware scoring is common in multivariate time-series anomaly detection, yet the literature rarely states when it helps. Holding one causality-guided autoencoder (DeCauAE) fixed, we vary graph quality, anomaly type and protocol against ground truth. Graph quality fixes the payoff: on a twenty-variable DAG the gain over a graph-free autoencoder decays as the graph is corrupted. Anomaly type fixes the target: the structural factor lifts window F1 from 0.800 to 0.976 where anomalies break conditional structure, and its advantage vanishes below a break severity s 0.3. Protocol fixes the ranking: point-adjusted F1 lets a random score match the full model, while range-aware VUS separates the autoencoder family from the classical baselines. On real benchmarks the causal module pays only where density makes a system eligible and cross-subsample reproducibility selects entities inside it; enabling it everywhere costs 0.012 VUS-ROC. Th
    
[^227]: 部署预算约束下基于基础模型的预测

    Prediction with Foundation Models under Deployment Budgets

    [https://arxiv.org/abs/2611.12466](https://arxiv.org/abs/2611.12466)

    该论文提出了在平均查询成本预算约束下提升基础模型预测性能的框架：固定查询模型的方法可达到极小化极大最优率，并证明多个单独欠佳的模型输出组合后可超越最佳单一模型；自适应方法则借助Q学习为不同输入动态选择查询模型并决定停止时机。

    

    基础模型能够为下游任务提供预测，这些预测借鉴了大规模预训练所积累的知识，但不同基础模型在任务特定准确率和查询成本方面存在差异。我们开发了一个框架，在每条输入的平均查询成本预算约束下改进下游任务的预测，利用训练数据来学习应查询哪些模型，以及如何将模型输出与无需查询即可获得的特征相结合。第一种方法对每条输入查询相同的模型；第二种方法则允许不同输入查询不同的模型，并使用Q学习来决定下一个查询哪个模型以及何时停止查询。对于第一种方法，我们证明了多个单独并不准确的输出组合起来可以优于最佳的单个输出。当回归函数仅通过预算内的模型输出而依赖于这些输出时，该方法可以达到极小化极大率；否则，预算约束下的任何方法都可能被迫只能达到更慢的收敛率，而第一种方法恰好能达到该较慢的速率。对于第二种方法……（摘要原文在此处截断）

    arXiv:2611.12466v2 Announce Type: replace-cross  Abstract: Foundation models can supply predictions for a downstream task that draw on knowledge from extensive pretraining, but they differ in task-specific accuracy and query cost. We develop a framework that improves prediction for the task under a budget on the average query cost per input, using training data to learn which models to query and how to combine their outputs with features available without querying. The first method queries the same models for every input; the second lets them vary across inputs and uses Q-learning to decide which model to query next and when to stop. For the first method, we show that several individually inaccurate outputs can jointly outperform the best single one. It attains the minimax rate when the regression function depends on the outputs only through models within the budget. Otherwise, any method within the budget can be forced to a slower rate, which the first method attains. For the second m
    
[^228]: 绝对零度下的注意力机制

    Attention at Absolute Zero

    [https://arxiv.org/abs/2611.11937](https://arxiv.org/abs/2611.11937)

    该论文提出一种在零温度（硬最大值）下读取见证者的高阶配对注意力层，通过仅传递计数比较比特并在每轮循环重建状态，克服了软注意力在规模扩大时的权重分散、分数差距与状态漂移等失效模式，在比训练规模大1000倍的最大值检索任务上仍保持99.95%以上的准确率。

    

    注意力机制对答案的候选支持（我们称之为“见证者”）进行评分，并在某个温度下读取这些分数。我们构建了一个高阶注意力层，其状态存在于物品对之上：对于一对，每个第三方物品 k 都是一个见证者，通过其两条边 和 进行评判。该层在零温度下读取见证者（保留最佳者及其对应的计数），只传递比较这些计数的比特，在每次循环中从输入和这些比特重建其配对状态，并且仅依靠答案进行训练。每一项设计选择都消除了软注意力在规模扩大时的一种失效模式：权重分散、学习到的分数之间无法区分的差距，以及漂移的软状态。在最大值检索任务中，该模型在比训练集大1,000倍的集合上，于全部十个随机种子下均达到至少99.95%的准确率；而使用软计数训练的同一网络在4,096个项目时性能降至42%。在无提示的CLRS-30基准上，该层的一般形式（其一半通道在零温度下读取……）

    arXiv:2611.11937v2 Announce Type: replace-cross  Abstract: Attention scores candidate supports of an answer, which we call witnesses, and reads the scores at a temperature. We build a higher-order attention layer whose state lives on pairs of items: for a pair (a, b), each third item k is a witness judged through its legs (a, k) and (k, b). It reads witnesses at zero temperature, keeping the best and the number tied with it, passes on only bits comparing such counts, rebuilds its pair state from the input and these bits at every loop, and is trained from answers alone. Each choice removes one way soft attention fails at size: dispersed weight, unresolved gaps between learned scores, and a drifting soft state. In maximum retrieval it is right on at least 99.95% of sets 1,000 times larger than in training, on all ten seeds; trained with a soft count, the same network falls to 42% at 4,096 items. On CLRS-30 without hints, a general form of the layer (half its channels read at zero tempera
    
[^229]: 论智能体式操作策略

    On Agentic Manipulation Policies

    [https://arxiv.org/abs/2611.09892](https://arxiv.org/abs/2611.09892)

    提出AMP1智能体式操作框架，证明通用AI模型直接控制机械臂可零样本超越VLA和WAM等专用策略，在RoboLab-120上达到85.6%的成功率。

    

    近年来机器人操作领域的研究主要聚焦于通过演示数据训练的专用策略。我们证明，通用型AI模型能够比专用策略（如视觉-语言-动作模型（VLA）和世界-动作模型（WAM））更可靠地控制机械臂。我们系统地分析了智能体式操作策略（AMP），即由通用型AI模型直接指挥机械臂，并随着任务的展开自适应地调整其动作。我们提出了AMP1，一个可在单臂、双臂和移动操作中无需任何改动即可运行的参考框架。AMP1在RoboLab-120和CaP-Gym基准测试中取得了最先进的结果，且无需任何演示数据或任务特定训练。在RoboLab-120上，AMP1零样本成功率达到85.6%，而领先的VLA和WAM策略分别仅为49.2%和42.9%。该智能体在涵盖十二个任务和三台机器人的108次真实世界测试中零样本成功了104次，显著降低了失败率。

    arXiv:2611.09892v2 Announce Type: replace-cross  Abstract: Recent work on robot manipulation has focused on specialist policies trained on demonstrations. We show that generalist AI models can control manipulators more reliably than specialist policies such as vision-language-action models (VLAs) and world-action models (WAMs). We systematically analyze agentic manipulation policies (AMPs), in which a generalist AI model commands the manipulator directly, adapting its actions as the task unfolds. We present AMP1, a reference harness that runs unchanged across single-arm, bimanual, and mobile manipulation. AMP1 achieves state-of-the-art results on the RoboLab-120 and CaP-Gym benchmarks, with no demonstrations or task-specific training. On RoboLab-120, AMP1 reaches an 85.6% success rate zero-shot, compared to 49.2% and 42.9% for leading VLA and WAM policies. The agent succeeds zero-shot in 104 of 108 real-world trials spanning twelve tasks and three robot instances, reducing failure rate
    
[^230]: 上下文-答案元模型：在首个答案词元生成之前预测大语言模型激活

    Context--Answer Metamodels: Forecasting LLM Activations Before the First Answer Token

    [https://arxiv.org/abs/2611.06837](https://arxiv.org/abs/2611.06837)

    本文提出上下文-答案元模型，证明语言模型阅读上下文时产生的激活序列可以直接高保真地预测其在随机解码推演中产生的答案激活，且性能随训练数据增长而提升并能零样本泛化到域外数据集。

    

    在随机解码条件下，单个上下文可以从语言模型中引出许多不同的答案。表征这种变化通常需要大量昂贵的自回归推演，而每次推演只能揭示一个实现样本。我们提出这样一个问题：模型在阅读上下文时产生的激活能否直接预测模型推演时的激活。我们训练了上下文-答案元模型，将上下文激活序列映射为平均池化答案激活的点预测或分布预测。我们发现：（i）我们研究的四种元模型家族能够高保真地预测平均池化的答案激活，且分布元模型还能捕捉推演过程中的变异性；（ii）随着训练集增长到50万个提示，性能不断提升，并且元模型能够零样本泛化到域外数据集；（iii）它们的预测保留了可预测模型行为的信号；（iv）来自Qwen2.5的上下文激活能够预测平均池化的答案激活。

    arXiv:2611.06837v1 Announce Type: cross  Abstract: Under stochastic decoding, a single context can elicit many different answers from a language model. Characterizing this variation typically requires many expensive autoregressive rollouts, each revealing only one realization. We ask whether the activations produced while reading the context can directly forecast the model's rollout activations. We train context--answer metamodels that map a context activation sequence to point or distribution forecasts of mean-pooled answer activations. We find that (i) the four metamodel families we study forecast mean-pooled answer activations with high fidelity, and the distribution metamodels also capture rollout variability; (ii) performance improves as the training set grows to 500,000 prompts, and the metamodels generalize zero-shot to out-of-domain datasets; (iii) their forecasts retain behavior-predictive signal; and (iv) context activations from Qwen2.5 predict mean-pooled answer activations
    
[^231]: ConDR：基于神经网络汉密尔顿-雅可比下解的保守可达性分析

    ConDR: Conservative Reachability through Neural Hamilton-Jacobi Subsolutions

    [https://arxiv.org/abs/2611.06802](https://arxiv.org/abs/2611.06802)

    提出 ConDR 方法，通过单边损失和学习 HJI 变分不等式的下解来保证神经网络估计的危险集保守地包含真实危险集，同时避免退化解，实现了高维系统中安全可靠的可达性分析。

    

    汉密尔顿-雅可比（HJ）可达性分析用于识别动力系统中的不安全状态，即在最坏情况扰动下系统可能被强制进入失败的那些状态。然而，经典的基于网格的求解器随状态维数呈指数级扩展。DeepReach 通过用基于 HJ-Isaacs（HJI）变分不等式（VI）训练的神经值函数取代网格来提升可扩展性，但它对残差的两侧施加同等的惩罚。这导致学到的值可能高于真实值，从而使不安全状态被错误地分类为安全。我们提出保守 DeepReach（ConDR），旨在学习 HJI 变分不等式的一个下解，使得学到的危险集包含真实的危险集。ConDR 通过构造方式强制执行障碍分支和终端条件，利用单边损失将 HJI 残差推至低于零的 $\epsilon$ 安全裕度，并防止将所有状态都判为不安全的退化解。我们证明了……（摘要在此处截断）

    arXiv:2611.06802v1 Announce Type: cross  Abstract: Hamilton-Jacobi (HJ) Reachability identifies the unsafe states of a dynamical system from which it can be forced into failure under worst-case disturbance. However, classical grid-based solvers scale exponentially with state dimension. DeepReach improves scalability by replacing the grid with a neural value function trained on the HJ-Isaacs (HJI) variational inequality (VI), but penalizes both sides of the residual equally. This allows the learned value to lie above the true one and unsafe states to be misclassified as safe. We propose Conservative DeepReach (ConDR) which aims to learn a subsolution of the HJI VI such that the learned unsafe set contains the true one. ConDR enforces the obstacle branch and terminal condition by construction, pushes the HJI residual to a safety margin $\epsilon$ below zero with a one-sided loss, and prevents the degenerate solution of classifying everything as unsafe. We prove that the exact solution of
    
[^232]: 对齐代价是分级的：SFT保留下长度归一化DPO的任务选择性能力损失

    The Alignment Tax Is Graded: Task-Selective Capability Loss Under Length-Normalized DPO with SFT Retention

    [https://arxiv.org/abs/2611.06476](https://arxiv.org/abs/2611.06476)

    对齐微调带来的能力损失并非均匀发生，而是分级且任务选择性的——抽取类任务几乎不受影响，JSON生成轻微下降，而路由任务出现约30分的严重退化。

    

    设定。这是一项实证特征描述研究；我们未提出新方法，也未与其他目标函数进行对比。我们在结构化财务报告任务上，使用稀疏LoRA和长度归一化DPO加上SFT保留项（即RPO目标）对Qwen3.5-4B进行微调，并跟踪下游能力随隐式奖励边际增长至约7个β缩放单位（nf4量化训练；在未量化的基础模型上以fp16测量时约为3）时的变化。问题。边际增长是否会在所有任务上均匀地降低能力，还是具有选择性？发现。在一项预注册的研究（8个随机种子，其中6个在冻结前未使用）中，通过伪影感知评分和TOST等效性检验，对齐代价是分级的且具有任务选择性：信息抽取能力得以保留（Δ = -0.55分，在±5分范围内等效），json能力损失很小（Δ = -3.52分，在聚类后的项目级区间上等效性检验未通过），而路由能力严重退化（Δ = -30分）。

    arXiv:2611.06476v1 Announce Type: new  Abstract: Setting. This is an empirical characterization study; we propose no new method and run no comparisons against alternative objectives. We fine-tune Qwen3.5-4B with sparse LoRA and length-normalized DPO plus an SFT retention term (the RPO objective) on structured financial-filing tasks, and track downstream capability as the implicit reward margin grows to $\sim 7$ $\beta$-scaled units (nf4 training; $\sim 3$ under fp16 measurement on the unquantized base). Question. Does margin growth degrade capability uniformly across tasks, or selectively? Findings. In a pre-registered study (8 seeds, 6 unseen at freeze) with artifact-aware scoring and TOST equivalence testing, the alignment tax is graded and task-selective: extraction is preserved ($\Delta = -0.55$ pts, equivalent at $\pm5$ pts), json loses little ($\Delta = -3.52$ pts, equivalence fails on a clustered item-level interval), and router degrades severely ($\Delta = -30$ pts). Router's l
    
[^233]: 多少个有序投影能确定k维空间中的n个点？

    How many sorted projections determine n points in k dimensions?

    [https://arxiv.org/abs/2611.06470](https://arxiv.org/abs/2611.06470)

    本文借助Hair和Sahai关于随机线性码列表恢复的两个引理，证明了在任意k维空间中，几乎任意选择适当数量的方向即可确定n个点的多重集，从而回答了Dym等人提出的高维开放问题。

    

    设 $d_1,\dots,d_m$ 为 $\mathbb{R}^k$ 中的方向，其中 $k\ge 2$。$\mathbb{R}^k$ 中 $n$ 个点构成的多重集 $X$ 会投影为 $m$ 个数字多重集 $\{\{x\cdot d_j : x\in X\}\}$，或等价地，投影为这些数字的 $m$ 个排序列表。需要多少个方向才能使这些投影确定任意一个由 $n$ 个点构成的多重集？在每个维度上都有两个已知的简单界：总是需要多于 $(k-1)\log_2 n$ 个方向，而处于一般位置的 $(k-1)n+1$ 个方向总是足够的。在平面上，人们知道得更多：Matousek、Privetivy 和 Skovron 证明了几乎任何 $O(\log n \log\log n)$ 个方向的选择都足够。至于在高维空间中是否有类似的结果成立，这是 Dym、Wellershoff、Tsoukanis、Levy 和 Balan 留下的开放问题。我们观察到，Hair 和 Sahai 为随机线性码的列表恢复所证明的两个引理可以回答该问题，并且通过收紧他们的论证，我们证明了在每个维度中……

    arXiv:2611.06470v1 Announce Type: cross  Abstract: Let $d_1,\dots,d_m$ be directions in $\mathbb{R}^k$, where $k\ge 2$. A multiset $X$ of $n$ points of $\mathbb{R}^k$ projects to the $m$ multisets of numbers $\{\{x\cdot d_j : x\in X\}\}$, or equivalently to the $m$ sorted lists of these numbers. How many directions are needed for these projections to determine every multiset of $n$ points? Two simple bounds are known in every dimension: more than $(k-1)\log_2 n$ directions are always needed, and $(k-1)n+1$ directions in general position always suffice. In the plane much more is known: Matousek, Privetivy and Skovron showed that almost every choice of $O(\log n \log\log n)$ directions suffices. Whether anything similar holds in higher dimensions was left open by Dym, Wellershoff, Tsoukanis, Levy and Balan. We observe that two lemmas of Hair and Sahai, proved for list recovery of random linear codes, answer that question, and by tightening their argument we show that in every dimension a
    
[^234]: Best-of-Depth：在循环Transformer中学习探索性动力学

    Best-of-Depth: Learning Exploratory Dynamics in Looped Transformers

    [https://arxiv.org/abs/2611.04663](https://arxiv.org/abs/2611.04663)

    提出Best-of-Depth方法，通过跨深度最小化预测损失将循环Transformer的推理轨迹训练为候选解集合，无需架构改动即可显著提升N皇后补全和图着色等任务的解覆盖率（91.03% 对比 45.75%）。

    

    循环Transformer通过重复的潜在状态更新来扩展推理时的计算，但最终深度训练并不直接鼓励有用的中间替代方案。我们提出了Best-of-Depth（BoD）方法，通过最小化跨深度的预测损失，将一条循环轨迹训练为候选解集合。不同的有效目标可以在不同深度被匹配，无需架构改动或针对特定深度的标签。精确匹配分析将该目标与解覆盖范围联系起来。在N皇后补全和图着色任务上，BoD相比最终深度训练和中间监督目标提高了覆盖率：N皇后覆盖率达到91.03%，而最终深度训练仅为45.75%。进一步实验表明，该方法能够发现超出训练深度的解，并在熟悉输入上恢复教师池中未包含的解。使用BoD对预训练的循环语言模型进行微调，其汇总代码生成成功率高于最终深度训练。

    arXiv:2611.04663v2 Announce Type: replace-cross  Abstract: Looped Transformers extend inference-time computation through repeated latent-state updates, but final-depth training does not directly encourage useful intermediate alternatives. We introduce Best-of-Depth (BoD), which trains a recurrent trajectory as a candidate set by minimizing prediction loss across depths. Different valid targets can be matched at different depths without architectural changes or depth-specific labels. An exact-match analysis connects this objective to solution coverage. On N-Queens completion and graph coloring, BoD improves coverage over final-depth and intermediate-supervision objectives. N-Queens coverage reaches 91.03%, versus 45.75% for final-depth training. Further experiments show discovery beyond the training depth and recovery of solutions withheld from teacher pools on familiar inputs. Fine-tuning a pretrained looped language model with BoD yields higher pooled code-generation success than fina
    
[^235]: 从粗粒度观测进行生成式预测：级联有帮助吗？

    Generative Forecasting from Coarse Observations: Does Cascading Help?

    [https://arxiv.org/abs/2611.00686](https://arxiv.org/abs/2611.00686)

    该论文提出一种分层生成参考数据的方法来评估生成式预测器是否采样自正确的条件分布，发现标准扩散模型和随机插值在有限数据和模型容量下会产生显著的分布误差，并证明了采用级联较短预测时段的方式有助于缓解这一问题。

    

    我们研究从粗粒度观测中对流体和波等动力系统进行概率预测的问题。由于不可见的细尺度会影响动力学过程，同一个粗粒度初始状态可能演化出整个未来分布。生成式预测器可能产生视觉上合理的样本，但由于轨迹数据中每个初始状态仅包含单一未来，因此很难判断它们是否采样自正确的条件分布。我们通过分层方式生成参考数据来解决这一问题：对于每个粗粒度初始状态，我们生成许多不同的细尺度分量，并对完整状态进行数值演化。利用这些条件集合，我们表明标准的扩散模型和随机插值方法尽管在原理上是精确的，但在有限数据和模型容量下于整个预测时段上训练时，仍可能产生显著的分布误差。随后我们展示了级联较短的预测时段……

    arXiv:2611.00686v2 Announce Type: replace  Abstract: We study probabilistic forecasting of dynamical systems such as fluids and waves from coarse observations. Since the unseen fine scales influence the dynamics, the same coarse initial state can evolve into a whole distribution of futures. Generative forecasters may produce visually plausible samples, but it is hard to tell whether they sample the right conditional distribution, since trajectory data contain a single future for each initial state. We address this problem by generating reference data in a hierarchical way: for each coarse initial state, we generate many different fine-scale components and numerically evolve the full states. Using these conditional ensembles, we show that standard diffusion models and stochastic interpolants can make substantial distributional errors when trained over the full forecast horizon with finite data and model capacity, despite being exact in principle. We then show that cascading shorter fore
    
[^236]: ICU死亡率预测中的生理轨迹与观测模式：测量条件变化下的可迁移性研究

    Physiological Trajectories and Observation Patterns in ICU Mortality Prediction: Transportability Under Changing Measurement Conditions

    [https://arxiv.org/abs/2610.42126](https://arxiv.org/abs/2610.42126)

    该研究通过嵌套表示设计与跨数据库外部验证发现，生理轨迹在ICU死亡率预测中具有跨医院、跨电子病历系统的可迁移预测价值，并独立于观测数量评估了测量时机对性能的影响。

    

    目的：区分ICU死亡率预测中生理轨迹与观测模式各自的可迁移预测价值，并评估测量时机是否独立于观测数量对预测性能产生影响。材料与方法：使用MIMIC-III、MIMIC-IV和eICU数据集对48小时内的院内死亡率进行预测。通过嵌套表示逐步整合最新值快照（R1）、纵向生理摘要（R2）以及测量频率与时机特征（R3）。在7,762名MetaVision患者上拟合的模型，在来自204家医院的21,682名MIMIC-IV患者和43,878名eICU患者上进行评估。在保持保留观测数量不变的前提下，评估测量时机的影响。结果：生理轨迹改善了eICU的外部预测效果（R2-R1对数损失差异-0.00922，95% CI -0.01148至-0.00704）。观测模式特征改善……（原文摘要到此截断）

    arXiv:2610.42126v1 Announce Type: new  Abstract: Objective: To distinguish the transferable predictive value of physiological trajectories from that of observation patterns in ICU mortality prediction, and to assess whether measurement timing affects performance independently of observation quantity.   Materials and Methods: In-hospital mortality was predicted at 48 hours using MIMIC-III, MIMIC-IV, and eICU. Nested representations progressively integrated latest-value snapshots (R1), longitudinal physiological summaries (R2), and measurement frequency and timing features (R3). Models fitted on 7,762 MetaVision patients were evaluated on 21,682 MIMIC-IV patients and 43,878 eICU patients across 204 hospitals. The effect of measurement timing was assessed while holding the number of retained observations constant.   Results: Physiological trajectories improved external prediction in eICU (R2-R1 log-loss difference -0.00922, 95% CI -0.01148 to -0.00704). Observation-pattern features improv
    
[^237]: 探究晶体材料正向筛选与逆向设计的极限

    Probing the limits of forward screening and inverse design of crystal materials

    [https://arxiv.org/abs/2610.32678](https://arxiv.org/abs/2610.32678)

    该研究将55个晶体数据库按结构匹配整合为包含1460万种晶体和1.318亿性能值的统一数据库，揭示晶体-性能数据仅有4.5%被填充，并开发了高精度等变图变换器CGTNet构建性能模型库来填补数据空缺，以此探究晶体材料正向筛选与逆向设计的能力极限。

    

    晶体材料的设计要么通过在已知晶体中筛选具有目标性能的材料，要么通过生成具有该性能的新晶体。这两种途径都以已知晶体的公共数据库作为评判标准，但这些数据库对同一晶体的命名方式各不相同。近期多个生成式与自主发现项目所宣称的新材料，事后被证明是已知物相或其近似变体。这两种途径的能力极限至今仍未被测量。本文通过将55个数据库中的晶体按结构匹配，构建了一个可自我扩展的数据库，包含1460万种晶体和1.318亿个性能值，且每个数值均可追溯至其来源。该数据库中仅有4.5%的晶体-性能对含有数值。为填补这些空缺，我们开发了CGTNet——一种等变图变换器，在两个公开基准的若干任务上达到了最先进的精度。针对每个任务训练后，它构成了一个性能模型库，我们已发布其预测结果。

    arXiv:2610.32678v1 Announce Type: cross  Abstract: Crystal materials are designed either by screening known crystals for a target property or by generating new crystals with that property. Both routes are judged against public databases of known crystals, but these databases name the same crystal differently. Several materials announced as new by recent generative and autonomous discovery campaigns have since proved to be known phases or close variants of them. The limits of both routes remain unmeasured. Here we probe them by matching the crystals of 55 databases by structure into one self-extending database of 14.6 million crystals and 131.8 million property values, each traced to its source. Only 4.5% of its crystal-property pairs carry a value. To fill the empty pairs, we developed CGTNet, an equivariant graph transformer with state-of-the-art accuracy on several tasks of two public benchmarks. Trained on each task, it forms a library of property models whose predictions we release
    
[^238]: REACT：思维链轨迹的表征早期分析与控制

    REACT: Representational Early Analysis and Control of Chain-of-Thought Trajectories

    [https://arxiv.org/abs/2610.32113](https://arxiv.org/abs/2610.32113)

    REACT 提出一种从成对安全与不安全轨迹中学习安全方向、并一次性施加于思维链早期隐藏状态的激活干预方法，发现最强控制增益集中在思维链的前两到三个 token 窗口，即使后续状态更能预测风险。

    

    思维链生成提供了一种在推理模型完成回答之前进行干预的机会。但是，风险最容易被检测到的时间点是否也能提供最有效的控制？我们通过 REACT 来研究这个问题，这是一种条件性激活干预方法，它从成对的安全与不安全轨迹中学习一个安全方向，并将其一次性应用于早期的思维链隐藏状态。模型权重保持冻结，所有后续 token 的生成不再进行进一步编辑。为了研究时机问题，我们在前五个思维链位置上分别训练方向，并使用无门控编辑配合编辑后的完整重新生成进行评估。在三个蒸馏推理模型和三个生成种子上，最强的增益集中在两到三个 token 的早期窗口内；在 DeepSeek-R1-Distill-Llama-8B 上，效果在第四和第五个 token 处降至接近于零，尽管较晚的状态可能更能预测风险。

    arXiv:2610.32113v1 Announce Type: cross  Abstract: Chain-of-thought (CoT) generation offers an opportunity to intervene before a reasoning model completes its answer. But does the point at which risk is easiest to detect also offer the most effective control? We study this question with REACT, a conditional activation intervention that learns a safety direction from paired safe and unsafe trajectories and applies it once to an early CoT hidden state. The model weights remain frozen, and all subsequent tokens are generated without further edits. To study timing, we train directions separately at each of the first five CoT positions and evaluate ungated edits with full post-edit regeneration. Across three distilled reasoning models and three generation seeds, the strongest gains concentrate in a two- to three-token early window; on DeepSeek-R1-Distill-Llama-8B, the effect falls to approximately zero at the fourth and fifth tokens, even though later states can be more predictive of risk. 
    
[^239]: EgoHand：基于自我中心视频的单目3D手部重建

    EgoHand: Monocular 3D Hand Reconstruction from Egocentric Video

    [https://arxiv.org/abs/2610.30293](https://arxiv.org/abs/2610.30293)

    EgoHand是基于1400万张图像高质量语料库训练的单目3D手部重建模型，在自我中心视频上实现了最先进的性能，具备强大的零样本跨域泛化能力和约9.69毫秒的低推理延迟。

    

    我们提出了EgoHand，一个在自我中心（第一视角）视频上实现最先进性能的单目3D手部重建模型。我们方法的核心是构建了一个包含1400万张图像的高质量训练语料库，结合了合成数据、真实场景视频和多相机动作捕捉，以平衡场景多样性与监督质量。该模型采用SOMA-Hand作为其参数化表示，使用transformer编码器处理2D手部裁剪图像，并融入相机线索和辅助2D分支以提升几何一致性和图像对齐效果。在HOT3D、ARCTIC和H2O数据集上的零样本评估展示了强大的跨域泛化能力，在约9.69毫秒的推理延迟下，其关节位置、根平移和加速度误差均低于竞争基线方法。一项盲测用户偏好研究进一步表明，参与者更青睐EgoHand重建的手部姿态，而非Apple EgoDex的结果。

    arXiv:2610.30293v1 Announce Type: cross  Abstract: We introduce EgoHand, a monocular 3D hand reconstruction model that achieves state-of-the-art performance on egocentric video. Central to our approach is the curation of a high-quality training corpus of 14 million images, combining synthetic data, in-the-wild video, and multi-camera motion capture to balance scene diversity and supervision quality. The model adopts SOMA-Hand as its parametric representation and processes 2D hand crops with a transformer encoder, incorporating camera cues and an auxiliary 2D branch to improve geometric consistency and image alignment. Zero-shot evaluations on HOT3D, ARCTIC, and H2O demonstrate strong out-of-domain generalization, with lower joint-position, root-translation, and acceleration errors than competing baselines at an inference latency of approximately 9.69 ms. A blinded user preference study further shows that participants favor EgoHand's reconstructed hand poses over those from Apple EgoDex
    
[^240]: 基于动作条件化世界模型的电子健康记录患者轨迹预测

    Forecasting Patient Trajectories from the Electronic Health Records with Action-Conditioned World Models

    [https://arxiv.org/abs/2610.23862](https://arxiv.org/abs/2610.23862)

    该论文提出了一种动作条件化的世界模型，通过显式分离患者状态与临床行动并采用状态依赖的线性化预测器，实现了电子健康记录中患者轨迹的时间分辨预测、多步推演和动作条件化模拟。

    

    我们为电子健康记录（EHR）引入了一种预测性世界模型，将患者历史表示为动作条件化的潜在轨迹。我们的框架不再仅通过自回归标记预测来建模临床事件，而是显式地将患者状态与观察到的临床行动分离，并通过一个以状态和动作为条件的转移算子来学习它们的演化。该框架的核心是一个状态依赖的线性化预测器，它能够传播潜在患者状态中的局部变化，从而实现时间分辨预测、多步推演以及动作条件化模拟。实验表明，生成的潜在轨迹保留了编码器派生表示中存在的下游预测信号、时间结构和子群体层面的统计关系。我们进一步将自由运行的推演结果解码到原始临床特征空间，并将未来的临床测量结果与观察数据进行对比评估。

    arXiv:2610.23862v2 Announce Type: replace  Abstract: We introduce a predictive world model for electronic health records (EHR) that represents patient histories as action conditioned latent trajectories. Rather than modeling clinical events solely through autoregressive token prediction, our framework explicitly separates patient states from observed clinical actions and learns their evolution through a state and action conditioned transition operator. At its core is a state dependent linearization predictor that propagates local changes in latent patient state, enabling time resolved prediction, multi step rollout, and action conditioned simulation. Empirically, we show that generated latent trajectories preserve downstream predictive signal, temporal structure, and subgroup level statistical relationships present in encoder derived representations. We further decode free running rollouts into the original clinical feature space and evaluate future clinical measurements against observ
    
[^241]: 精细化拓扑遍次调度：SE(3)等变结构预测中动态k近邻图的阻尼信赖域补偿

    Refined Topology-Pass Scheduling: Damped Trust-Region Compensation for Dynamic k-NN Graphs in SE(3)-Equivariant Structure Prediction

    [https://arxiv.org/abs/2610.19246](https://arxiv.org/abs/2610.19246)

    提出一种双遍拓扑调度方案，将动态kNN图构建基准的选择形式化为以逐块重建（PB）为唯一不动点的阻尼不动点迭代，并借助不翻转证书、最小方差（BLUE）重放增益与ℓ∞信赖域投影进行保守补偿，从而在SE(3)等变结构预测中以低成本逼近逐块重建的拓扑精度。

    

    等变结构预测流水线在坐标更新与k近邻（kNN）图构建之间交替进行，其中图的成员关系是坐标的离散、分段常数函数。在每个块边界（PB）重建该图是训练轮次中最昂贵的部分，而复用一个冻结图（F）虽然快速，却会留下拓扑差距。我们研究了一种双遍设计：廉价的遍次记录少量状态信息（间隔边距、kNN半径、实测的每边界位移），第二遍则进行保守补偿。我们将图构建基准的选择形式化为一个阻尼不动点迭代，其唯一不动点恰好为PB，并由此推导出：(i) 具有双侧常数 $m_i/4$ 和单侧常数 $m_i/2$ 的精确不翻转（non-flip）证书；(ii) 最小方差（BLUE）重放增益 $\gamma^*$ 及由此产生的可行性交叉点；(iii) 一个 $\ell_\infty$ 信赖域投影，可保……

    arXiv:2610.19246v1 Announce Type: new  Abstract: Equivariant structure-prediction pipelines alternate coordinate updates with the construction of a k-nearest-neighbour (kNN) graph whose membership is a discrete, piecewise-constant function of the coordinates. Rebuilding that graph at every block boundary (PB) is the expensive part of an epoch, whereas reusing one frozen graph (F) is fast but leaves a topology gap. We study a two-pass design in which the cheap pass records a small amount of state (margins, kNN radii, measured per-boundary displacements) and a second pass compensates conservatively. We formulate the choice of the graph-construction basis as a damped fixed-point iteration whose unique fixed point is exactly PB, and derive (i) exact non-flip certificates with the two-sided constant $m_i/4$ and one-sided constant $m_i/2$, (ii) the minimum-variance (BLUE) replay gain $\gamma^*$ and the resulting feasibility crossover, (iii) an $\ell_\infty$ trust-region projection that prese
    
[^242]: ResearchFlow-Eval：追踪端到端多论断自主研究中的执行差距

    ResearchFlow-Eval: Tracing the Execution Gap in End-to-End Multi-Claim Autoresearch

    [https://arxiv.org/abs/2610.18088](https://arxiv.org/abs/2610.18088)

    该论文提出 ResearchFlow-Eval——一个基于已发表论文锚定、在阶段与论断两个层面细粒度评估端到端多论断自主研究系统持续科学执行与迭代能力的评测框架，弥补了现有评估仅打分孤立任务或最终成果的不足。

    

    自主研究系统越来越多地承担起完整的研究项目。在实践中，科学研究要求系统通过持续的、多阶段的工作流程来追求多个相互依赖的论断，并随着证据的积累不断修订实验。然而，现有的自主研究系统评估通常只对孤立任务或最终成果进行打分，使得这些过程层面的能力在很大程度上未被衡量。因此，我们提出了 ResearchFlow-Eval，这是一个在阶段与论断两个层面上评估持续科学执行与迭代的细粒度框架。每项评估都锚定于一篇已发表的论文，从中推导出针对各阶段的评分细则，将每个研究阶段分解为多个维度和分别打分的细则条目。利用这些评分细则，我们对每个研究阶段进行评分，并详细考察实验执行情况，重点关注多论断的完成度以及跨多次尝试的修订。我们将该评估应用于（原文摘要在此处截断）

    arXiv:2610.18088v2 Announce Type: new  Abstract: Autoresearch systems increasingly undertake entire research projects. In practice, scientific research requires systems to pursue multiple interdependent claims through a continuous, multi-stage workflow and revise experiments as evidence accumulates. Existing evaluations of autoresearch systems, however, typically score isolated tasks or final artifacts, leaving these process-level capabilities largely unmeasured. We therefore introduce ResearchFlow-Eval, a fine-grained framework for assessing continuous scientific execution and iteration at both the stage and claim levels. Each evaluation is grounded in a published anchor paper, from which we derive stage-specific rubrics that decompose each research stage into dimensions and separately scored rubric items. Using these rubrics, we score each research stage and examine experimental execution in detail, focusing on multi-claim completion and revisions across attempts. We apply the evalua
    
[^243]: 面向平均奖励 $Q$-学习与 $Q$-规划的混合正则化方法

    Mixing Regularization for Average-Reward $Q$-Learning and $Q$-Planning

    [https://arxiv.org/abs/2610.14988](https://arxiv.org/abs/2610.14988)

    该论文提出通过向一个与奖励无关的 $m$ 步混合策略进行正则化来控制平均奖励 $Q$-学习的收缩与探索性质，并给出有限时间收敛保证，在单步情形下匹配最佳已知速率，在 $m$ 步混合情形下首次推导出新的速率。

    

    强化学习的平均奖励设定具有挑战性，因为其贝尔曼算子仅在底层马尔可夫链具有混合性时才在跨度上收缩。现有工作通常假设存在收缩性质（例如通过环境混合性），而我们考虑这样的设定：给定一个与奖励无关的 $m$ 步混合策略，并通过向该混合策略正则化来控制所学策略的收缩性质与探索性质。我们对 $Q$-学习算法提供了有限时间分析，这些算法可以用多种正则化选择实例化，其保证相对于正则化后的不动点来度量。我们的框架揭示了新的、依赖于正则化子的对象，这些对象控制收缩、探索和算子偏差。在单步设定中，我们的速率匹配了 $Q$-迭代和同步 $Q$-学习的最佳已知速率；对于 $m$ 步混合情形，我们推导出了新的速率。

    arXiv:2610.14988v1 Announce Type: cross  Abstract: The average-reward setting for reinforcement learning is challenging because its Bellman operator is known to contract in span only when the underlying Markov chain is mixing. Whereas existing work typically assumes a contraction property, e.g., through environment mixing, we consider the setting where we are given a reward-agnostic $m$-step mixing policy, and we control the contraction and exploration properties of the learned policy by regularizing toward the mixing policy. We provide finite-time analysis of $Q$-learning algorithms that can be instantiated with multiple choices of regularization, with guarantees measured with respect to the regularized fixed point. Our framework exposes novel regularizer-dependent objects that control contraction, exploration, and operator bias. In the one-step setting, our rates match the best known rates for $Q$-iteration and synchronous $Q$-learning, and for $m$-step mixing we derive new rates. We
    
[^244]: 跨行为策略优化：通过共享回归目标从离线演示与在策略及离策略采样中实现统一学习

    Cross-Behavior Policy Optimization: Unified Learning from Offline Demonstrations and On- and Off-Policy Rollouts via a Shared Regression Target

    [https://arxiv.org/abs/2610.12830](https://arxiv.org/abs/2610.12830)

    CBPO通过固定参考的共享回归目标，将离线演示、在策略与离策略采样统一到同一回归损失和回放缓冲区中，无需分别协调SFT与RL目标、重要性加权或评论家网络即可实现数据的持续复用与统一学习。

    

    推理后训练依赖离线演示数据和策略生成的采样数据，然而传统的先监督微调（SFT）后强化学习（RL）的方法将两者的目标分离，并限制了已收集回复的持续复用。我们提出跨行为策略优化（CBPO），通过固定参考目标使这些数据源能够兼容于同一种回归损失。由于生成策略不进入该目标，正负离线补全、参考回复以及当前与历史采样数据可以共享统一的奖励标注记录格式和经验回放缓冲区。随着新回复进入缓冲区，优化始终保持相同的回归目标，无需协调分离的SFT和RL目标，也无需存储行为策略概率、重要性加权或学习评论家网络。通过限制可用的数据源，可以得到离线、在策略和离策略等变体。

    arXiv:2610.12830v1 Announce Type: new  Abstract: Reasoning post-training draws on offline demonstrations and policy-generated rollouts, yet conventional supervised fine-tuning (SFT) followed by reinforcement learning (RL) separates their objectives and limits continued reuse of collected responses. We propose \textbf{Cross-Behavior Policy Optimization} (CBPO), which makes these sources compatible with one regression loss through a fixed-reference target. Because the generating policy does not enter this target, positive and negative offline completions, reference responses, and current and historical rollouts can share a common reward-labeled record format and replay buffer. As new responses enter the buffer, optimization retains the same regression objective without coordinating separate SFT and RL objectives or requiring stored behavior-policy probabilities, importance weighting, or a learned critic. Restricting the available sources gives offline, on-policy, and off-policy variants,
    
[^245]: 模式识别与逐步推理之间的二分法

    The Dichotomy Between Pattern Recognition and Step-by-Step Reasoning

    [https://arxiv.org/abs/2610.09186](https://arxiv.org/abs/2610.09186)

    论文提出模式识别与逐步推理是同一光谱的两端：当下一个词元仅依赖少量前文时 LLM 学会逐步推理，任务的推理轨迹构成 De Bruijn 图的有向无环子图，且其边数远小于轨迹数，因此 LLM 可以通过组合少量已学习的边来解决更长的新任务。

    

    我们认为，模式识别与逐步推理是同一光谱的两端。当数据的结构使得下一个词元仅依赖于少量前文时，大语言模型（LLM）学会逐步推理；当下一个词元依赖于大量前文时，LLM 的推理则类似于模式识别。如果下一个词元仅依赖于最近的 $c$ 个词元，那么推理轨迹就是 De Bruijn 图上的路径，其中图的节点是长度为 $c$ 的上下文，边是上下文之间的下一词元转移。一个任务的推理轨迹集合构成该 De Bruijn 图的一个有向无环子图。已经学习了该子图所有边的 LLM 可以将这些边组合起来，解决更长、未见过的任务，即它能够逐步推理。我们证明，边的数量相比推理轨迹的数量是极其微小的。从实验上看，Transformer 所需的训练样本数量（摘要至此截断）

    arXiv:2610.09186v1 Announce Type: new  Abstract: We argue that pattern recognition and step-by-step reasoning are two ends of a spectrum. A large language model (LLM) learns to reason step-by-step when data is structured such that the next token depends on a small amount of preceding context. Inference in LLMs resembles pattern recognition when the next token depends on a large amount of preceding context. If the next token depends on only the $c$ most recent tokens, reasoning traces are paths on a De Bruijn graph whose nodes are $c$-length contexts and edges are next-token transitions between contexts. The set of reasoning traces of a task forms a directed acyclic subgraph of the De Bruijn graph. An LLM that has learned all edges of this subgraph can compose them to solve longer, unseen tasks, i.e., it reasons step-by-step. We prove that the number of edges is vanishingly small compared to the number of reasoning traces. Empirically, the number of training samples a transformer needs 
    
[^246]: Xaurora：基于基础模型先验的去噪随机插值生成式天气预报

    Xaurora: Generative Weather Forecasting with Denoising Stochastic Interpolants from a Foundation Model Prior

    [https://arxiv.org/abs/2610.06509](https://arxiv.org/abs/2610.06509)

    本文提出Xaurora方法，通过新颖的去噪随机插值生成技术和SDE重放缓冲区，将预训练的确定性大气基础模型Aurora转变为生成式集合预报模型，以低成本实现概率化天气预报与不确定性量化。

    

    近年来，深度学习彻底改变了天气预报领域，尤其是通过大气基础模型，这些模型以经典物理模型计算成本的一小部分提供了具有竞争力的预报能力。然而，大多数现有的基础模型是确定性的，这限制了通过生成大型集合预报来实现准确的不确定性量化、极端天气风险评估和长期天气预报。此外，这些模型从头开始训练需要巨大的、往往令人望而却步的计算开销。为了解决这些缺点，我们将一个预训练的确定性先验模型，即Aurora基础模型，转变为生成式集合预报模型。为此，我们引入了一种新颖的生成方法——去噪随机插值，并结合用于随机微分方程（SDE）滚动的重放缓冲区，实现了SDE轨迹的概率化训练。

    arXiv:2610.06509v2 Announce Type: replace  Abstract: Deep learning has revolutionised weather forecasting in recent years, especially through atmospheric foundation models, which offer competitive skill for a fraction of the computational costs of classic physics-based models. However, most existing foundation models are deterministic, limiting the generation of large ensembles for accurate uncertainty quantification, extreme weather risk assessment, and long-range weather forecasting. Furthermore, these models incur a large, often prohibitive, computational overhead to train from scratch. To address these shortcomings, we turn a pretrained deterministic prior model, namely the Aurora foundation model, into a generative ensemble-prediction model. To that end, we introduce a novel generative method, Denoising Stochastic Interpolants, combined with a replay buffer for Stochastic Differential Equation (SDE) rollout, enabling probabilistic training of SDE trajectories. Our stochastic found
    
[^247]: Kandinsky 6.0 Video：同步视频与音频生成的基础模型

    Kandinsky 6.0 Video: Foundation Models for Synchronized Video and Audio Generation

    [https://arxiv.org/abs/2610.05608](https://arxiv.org/abs/2610.05608)

    Kandinsky 6.0 Video 是一系列用于同步文本到音频-视频生成的基础扩散模型，其核心创新是采用双流 CrossDiT 架构，通过双向交叉注意力实现视频流与音频流的时间和语义对齐，可生成带同步44 kHz音频（含唇音同步）的5秒全高清视频。

    

    我们提出了 Kandinsky 6.0 Video，这是一个用于同步文本到音频-视频生成的基础扩散模型家族，包含 Kandinsky 6.0 Video Lite（30亿参数）和 Kandinsky 6.0 Video Pro（290亿参数）两款模型。两款模型均可在文本到音频-视频（T2AV）和图像到音频-视频（I2AV）两种模式下生成5秒视频片段，并配有同步的44 kHz音频（包括唇音同步）；内置的超分辨率模型可将输出分辨率提升至全高清（1920x1080）。基于 Kandinsky 5.0 的视频生成能力，Kandinsky 6.0 Video 采用双流 CrossDiT 架构，通过双向交叉注意力连接预训练的视频流与新训练的音频流，实现时间与语义上的对齐。我们的持续预训练策略首先在大规模音频语料库上从头训练音频流，然后在配对的音频-视频数据上联合训练两个流，同时保持单模态能力……（摘要原文在此处截断）

    arXiv:2610.05608v2 Announce Type: replace-cross  Abstract: We present Kandinsky 6.0 Video, a family of foundation diffusion models for synchronized text-to-audio-video generation, comprising Kandinsky 6.0 Video Lite (3B parameters) and Kandinsky 6.0 Video Pro (29B parameters). Both models generate 5-second video clips with synchronized 44 kHz audio, including lip-sync, in text-to-audio-video (T2AV) and image-to-audio-video (I2AV) modes; a built-in super-resolution model raises the output resolution to Full-HD (1920x1080). Building on the video generation capabilities of Kandinsky 5.0, Kandinsky 6.0 Video employs a dual-stream CrossDiT architecture that connects a pretrained video stream and a newly trained audio stream through bidirectional cross-attention for temporal and semantic alignment. Our continuous pretraining strategy first trains the audio stream from scratch on large-scale audio corpora and then trains both streams jointly on paired audio-video data while preserving unimoda
    
[^248]: 次水平Flood双过滤：迈向可扩展的2参数持续同调

    The sublevel Flood bifiltration: towards scalable 2-parameter persistent homology

    [https://arxiv.org/abs/2610.05441](https://arxiv.org/abs/2610.05441)

    本文提出次水平Flood双过滤，通过扩展单参数Flood过滤，为2参数持续同调提供了一种具有理论稳定性保证且可高效计算、适用于大规模点集的可扩展近似方法。

    

    多参数持续同调是拓扑数据分析中一个快速发展的分支，它提升了单参数持续同调对离群点的鲁棒性，同时仍能捕捉数据的度量特征。然而，一个显著的局限性在于其缺乏可扩展性。在本文中，我们提出了一种在大规模点集上高效计算2参数持续同调的新方法。我们的工作扩展了最初为单参数持续性而开发的Flood过滤。我们的构造称为次水平Flood双过滤，它为次水平偏移双过滤提供了可扩展的近似。我们证明了该构造具有理论稳定性，并描述了如何高效地计算它。我们在密度感知至关重要的低维合成数据集上，以及真实世界的时间序列数据集上，展示了我们方法在分类任务中的性能。

    arXiv:2610.05441v2 Announce Type: replace-cross  Abstract: Multiparameter persistent homology is a rapidly developing branch of topological data analysis that improves the robustness of single-parameter persistent homology to outliers, while still capturing the metric characteristics of the data. However, a notable limitation is its lack of scalability. In this paper, we introduce a novel approach for efficiently computing 2-parameter persistent homology on large point sets. Our work extends the Flood filtration, originally developed for single-parameter persistence. Our construction, called the sublevel Flood bifiltration, offers a scalable approximation of the sublevel offset bifiltration. We show that it benefits from theoretical stability properties and describe how to compute it efficiently. We demonstrate the performance of our approach in classification tasks on low-dimensional synthetic datasets, where density awareness is critical, as well as on real-world time series datasets
    
[^249]: 多类分类中的紧凑集合值深度集成学习

    Compact set-valued deep ensembling in multi-class classification

    [https://arxiv.org/abs/2610.05332](https://arxiv.org/abs/2610.05332)

    该论文提出利用紧凑集成（贝叶斯神经网络和蒙特卡洛dropout卷积神经网络）生成概率预测并构建代表性分布以定义贝叶斯最优预测，同时引入一族集合效用来实现减轻训练与存储负担且更鲁棒的集合值多类分类预测。

    

    本文解决深度集成学习（其中深度神经网络作为集成成员）中面临的显著挑战：训练与存储负担，以及针对多种效用（可能涉及奖励敏感性）的谨慎（集合值）预测的鲁棒性。为减轻训练和存储负担，我们提出采用紧凑集成方法，例如贝叶斯神经网络和具有蒙特卡洛dropout预测选项的卷积神经网络，来生成概率预测。对于每个查询实例，这些概率预测随后被用于定义一个优化某种统计距离的代表性分布，进而利用该代表性分布来定义任意效用下的贝叶斯最优预测（BOP）。为解决单点预测可能存在的非鲁棒性问题，我们提出了一族满足若干理想性质的集合效用，其集合值BOP可以被求解。

    arXiv:2610.05332v2 Announce Type: replace  Abstract: This paper tackles visible challenges in deep ensemble learning, where deep neural networks serve as ensemble members: training and storage burdens, and robustness of cautious (set-valued) predictions targeting multiple utilities, which may involve reward-sensitivity. To mitigate the training and storage burdens, we propose to employ compact ensembles, such as Bayesian Neural Networks and Convolutional Neural Networks with the Monte-Carlo dropout prediction option, to produce probabilistic predictions. For each query instance, these probabilistic predictions are then used to define a representative distribution optimizing some statistical distance. The representative distribution is then employed to define the Bayes-optimal prediction (BOP) of any utility. To address the potential unrobustness of singleton prediction making, we propose a family of set-utilities satisfying some desirable properties and whose set-valued BOPs can be fou
    
[^250]: ArrivalBench：代理生成的数据管道一次运行即正确，随时间推移而出错

    ArrivalBench: Agent-Generated Data Pipelines Are Correct Once and Wrong Under Time

    [https://arxiv.org/abs/2610.02363](https://arxiv.org/abs/2610.02363)

    ArrivalBench 提出在对抗性但可重放的数据送达调度下重新执行代理生成的数据管道，并以完整日志的批量重算作为判定标准，发现经单次执行评分认证的管道中有 7.0-79.2% 实际上默默出错，且这一差距与修复循环无关。

    

    现有针对代理（Agent）生成数据工作的基准测试，通常通过让数据管道对固定快照运行一次来评分。ArrivalBench 则改变这一方式：在对抗性但可重放的送达调度（包括迟到、重复、乱序和重试的记录）下重新执行代理所留下的管道，并要求其最终状态等于对完整日志进行批量重算的结果。由于该判定基准采用重算而非分类的方式，“错误的表”和“崩溃”会被判定为不同的结论：崩溃可以被团队现有的监控系统发现，而错误的表则不会。在我们构建的 40 个任务上，我们对单次执行评分方式的复现实现认证了十一个模型所生成管道的 86-100%；而重新执行同样的产物，却发现其中 7.0-79.2% 被认证的管道实际上默默出错了。这一差距并非由修复循环造成：在相同模型和任务下，针对快照测试修复过的管道，其重放失败率与首次就通过该测试的管道大致相当。

    arXiv:2610.02363v1 Announce Type: new  Abstract: Benchmarks for agent-generated data work grade a pipeline by running it once against a fixed snapshot. ArrivalBench instead re-executes the pipeline an agent leaves behind under adversarial but replayable delivery schedules (late, duplicated, out-of-order and retried records) and requires its final state to equal a batch recomputation of the complete log. Because the oracle recomputes rather than classifies, a wrong table and a crash are distinct verdicts: a crash is visible to monitoring a team already runs, and a wrong table is not. On 40 tasks we built, our reimplementation of single-execution grading certifies 86-100% of the pipelines eleven models produce; re-executing the same artifacts finds 7.0-79.2% of the certified ones silently wrong. The gap is not produced by the repair loop: within the same model and task, pipelines repaired against the snapshot test fail replay about as often as those that passed it first time. In every mo
    
[^251]: 用任务算子捕捉上下文学习动态

    Capturing In-Context Learning Dynamics with Task Operators

    [https://arxiv.org/abs/2610.01054](https://arxiv.org/abs/2610.01054)

    提出任务算子（TO）方法，通过将注意力头输出的仿射变换解析地重放到注意力输出投影中，从而无需处理完整示例即可高效模拟上下文学习，并揭示其内在机制。

    

    上下文学习（ICL）使语言模型能够仅凭示例演示执行新任务，而无需更新权重。然而，每次ICL推理都需要处理完整的示例集合，导致部署效率低下，且ICL的作用机制尚未被完全理解。先前的工作将ICL压缩为从特定层或位置提取的固定激活向量，但这些与输入无关的干预方法在复杂任务上会失效，因为这类任务的输出取决于与输入之间的细粒度交互。通过分析ICL的前向传递过程，我们发现每个注意力头的输出是其上下文掩码对应输出的仿射变换，且该变换的参数在给定任务的不同样本间具有经验上的稳定性。基于这一发现，我们提出了任务算子（Task Operator, TO），它将这一变换作为解析推导出的更新重放到注意力输出投影中。在词汇类、算法类……（原文摘要截断）

    arXiv:2610.01054v1 Announce Type: cross  Abstract: In-context learning (ICL) enables language models to perform new tasks from demonstrations without weight updates. However, every ICL inference requires processing the full set of examples, resulting in inefficient deployments, and how ICL works mechanistically is not fully understood. Prior work compresses ICL into fixed activation vectors extracted from specific layers or positions, but these input-independent interventions fail on complex tasks where the output depends on fine-grained interactions with the input. By analyzing the ICL forward pass, we show that each attention head's output is an affine transformation of its context-masked counterpart, and that the parameters of this transformation are empirically stable across samples for a given task. Building on this, we introduce Task Operator (TO), which replays this transformation as an analytically derived update to the attention output projection. Across lexical, algorithmic, 
    
[^252]: 迈向序列测试时扩展中更好的探索

    Towards Better Exploration in Sequential Test-Time Scaling

    [https://arxiv.org/abs/2609.39632](https://arxiv.org/abs/2609.39632)

    该论文揭示序列测试时扩展常因过早陷入“吸引子”（进入后阻止探索其他答案的答案集合）而停止改进——在27种方法、模型与基准的组合中53.8%的轨迹在四次迭代内进入吸引子——并提出一种简单的模型混合干预方法来帮助逃离吸引子，实现更好的探索。

    

    测试时扩展通过在推理时投入额外的计算来提升语言模型的推理能力。然而，现有的两类方法通常难以在长时间尺度上持续改进。并行方法反复从模型中采样独立的答案，在模型不太可能一次性解决的问题上扩展效果不佳。相比之下，序列方法在先前答案的基础上获取新想法，但迄今为止尚未被证明能够达到超越并行扩展所找到的答案。首先，我们表明序列扩展常常停止改进是因为它过早地陷入一个“吸引子”：一组一旦进入就会阻止探索不同答案的答案集合。在扩展方法、模型和基准测试的27种组合中，我们发现53.8%的序列扩展轨迹在四次迭代内进入吸引子。其次，我们展示了一种简单的模型混合干预方法有助于逃离吸引子。

    arXiv:2609.39632v1 Announce Type: new  Abstract: Test-time scaling improves language model reasoning by spending additional compute at inference. However, both classes of existing methods often fail to continue improving over long timescales. Parallel methods repeatedly sample independent answers from the model, scaling poorly on problems the model is unlikely to solve in a single attempt. In contrast, sequential methods build on previous answers to access new ideas, yet so far have not been shown to reach answers beyond those found by parallel scaling. First, we show that sequential scaling often stops improving because it becomes prematurely trapped in an attractor: a set of answers that prevents exploration of different answers once entered. Across 27 combinations of scaling methods, models, and benchmarks, we find that 53.8% of sequential scaling trajectories enter an attractor within four iterations. Second, we show that a simple model-mixing intervention helps escape attractors. 
    
[^253]: GroundingPI：一个基于视觉原语、面向物理智能的视觉定位基础模型

    GroundingPI: A Grounding Foundation Model towards Physical Intelligence with Visual Primitives

    [https://arxiv.org/abs/2609.39601](https://arxiv.org/abs/2609.39601)

    GroundingPI是一个40亿参数的视觉定位基础模型，它将点和框表示为共享词表中的量化坐标，并通过多模态与空间预训练、监督微调及GRPO强化学习训练而成，在11种感知能力的34个基准测试上以平均73.68%的成绩超越更大的GPT-6 Astra，创下新的最先进水平，为物理智能提供了快速精确的视觉感知基础。

    

    精确的视觉定位至关重要。它需要指明哪个物体是目标以及该物体位于何处，即使在杂乱场景和微小物体的情况下也必须准确，并且必须足够快速以支持闭环控制。然而，视觉-语言-动作模型（VLA）和世界-动作模型（WAM）通常从通用的视觉-语言骨干网络和视频生成骨干网络获取感知能力，而这些骨干网络在上述场景中仍然表现不佳。我们提出了GroundingPI，一个40亿参数的视觉定位基础模型，它以共享词表中的量化坐标形式生成点和边界框。其训练结合了多模态与空间预训练、监督微调以及基于GRPO的强化学习，并利用公开数据集和专用数据引擎提供监督信号。在与涵盖11种感知能力的34个视觉定位基准上的44个基线模型的对比中，GroundingPI确立了新的最先进水平，平均成绩达到73.68%，超过了参数量更大的GPT-6 Astra（71.54%）。作为下游任务的视觉骨干网络，GroundingPI提升了……（摘要在此处被截断）

    arXiv:2609.39601v1 Announce Type: cross  Abstract: Precise grounding matters. It specifies which object is the target and where that object is, even in clutter and for tiny objects, and it has to be fast enough for closed-loop control. Yet vision-language-action (VLA) and world-action models (WAMs) take perception from general-purpose vision-language and video-generation backbones, which still fail in these settings. We introduce GroundingPI, a 4B grounding foundation model that generates points and boxes as quantized coordinates in a shared vocabulary. Training combines multimodal and spatial pretraining, supervised fine-tuning, and reinforcement learning with GRPO, using supervision from public datasets and dedicated data engines. Against 44 baselines across 34 grounding benchmarks spanning 11 perceptual capabilities, GroundingPI establishes a new state of the art, averaging 73.68%, above the larger GPT-6 Astra (71.54%). As a downstream visual backbone, GroundingPI improves performan
    
[^254]: GroundAnything：以闪速实现并行解码与精准视觉定位的统一

    GroundAnything: Reconciling Parallel Decoding with Precise Visual Grounding at Flash Speed

    [https://arxiv.org/abs/2609.39600](https://arxiv.org/abs/2609.39600)

    GroundAnything 是一个 40 亿参数的视觉定位基础模型，它将视觉定位重新定义为可并行生成的双向扩散过程，通过分块去噪、AR 到扩散的架构转换训练及 GRPO 强化后训练，在实现闪速并行解码的同时保证了精准的视觉定位效果。

    

    自回归（AR）视觉定位模型将空间预测序列化，引入了顺序延迟，并对输出 token 施加了因果顺序约束。我们将视觉定位视为视觉证据提取：物体、位置和空间关系由图像和查询共同约束，但它们之间的依赖关系并不意味着存在内在的从左到右的生成顺序。这一区别使得双向扩散成为天然合适的选择，允许空间假设并行涌现，并通过迭代去噪进行联合优化。我们提出了 GroundAnything，一个 40 亿参数的视觉定位基础模型，通过分块去噪实现了快速并行解码与精准定位的统一。其训练过程结合了基于公开数据集与专用数据引擎的定位预训练、采用 AR 与扩散联合目标的直接 AR 到扩散架构转换、监督微调以及基于 GRPO 的强化后训练。在 30 个视觉定位基准测试上……

    arXiv:2609.39600v1 Announce Type: cross  Abstract: Autoregressive (AR) grounding models serialize spatial predictions, introducing sequential latency and imposing a causal order on output tokens. We view grounding as visual evidence extraction: objects, locations, and spatial relations are jointly constrained by the image and query, yet their dependencies do not imply an intrinsic left-to-right generation order. This distinction makes bidirectional diffusion a natural fit, allowing spatial hypotheses to emerge in parallel and be jointly refined through iterative denoising. We introduce GroundAnything, a 4B-parameter grounding foundation model that reconciles fast parallel decoding with precise localization through blockwise denoising. Training combines grounding pretraining from public datasets and dedicated data engines, direct AR-to-diffusion conversion with joint AR and diffusion objectives, supervised fine-tuning, and GRPO-based reinforcement post-training. Across 30 grounding benc
    
[^255]: 早期问题的计算能否帮助大语言模型解决新问题？

    Can Computation from Earlier Problems Help LLMs Solve New Ones?

    [https://arxiv.org/abs/2609.39394](https://arxiv.org/abs/2609.39394)

    提出 STAIR 方法，通过固定存储库复用早期响应的键值信息并仅训练 12,288 个参数，就能让大语言模型在多轮对话中利用先前问题的计算来提升解决新问题的准确率。

    

    大语言模型经常在同一对话中解决相互独立的问题。那么，来自较早问题的计算能否帮助它们解决新问题？为了回答这个问题，我们首先进行了初步实验，结果表明保留的历史记录既可能提高也可能降低后续轮次的准确率，即使是在同一领域内也是如此。为了理解这些影响，我们采用受控重放方法来隔离特定于每个“问题-历史”配对的内部状态变化。结果显示，在不同的历史条件下，这些变化在当前问题之间保持了相似的关系。为了在保留历史的情况下改进推理，我们提出了 STAIR（Stale-Token Attention for Inter-query Reuse，面向跨查询重用的陈旧令牌注意力机制）。STAIR 将较早响应生成过程中的键和值捕获到一个固定存储库中，并学习在提示处理期间当前查询读取该存储库时对其进行重定向。基础模型保持完全冻结，仅训练 12,288 个参数。在三个 Qwen 模型和四个基准测试上，STAIR 提升了平均准确率（原文在此处截断）。

    arXiv:2609.39394v1 Announce Type: new  Abstract: Large language models often solve independent problems in the same conversation. Can computation from earlier problems help them solve new ones? To answer this question, we first conduct preliminary experiments showing that retained history can raise or lower later-turn accuracy, even within the same domain. To understand these effects, we use controlled replay to isolate internal state changes specific to each problem-history pairing. Across different histories, these changes preserve similar relationships among current problems. To improve reasoning under retained history, we introduce STAIR (Stale-Token Attention for Inter-query Reuse). STAIR captures keys and values from earlier response generation in a fixed bank. It learns to redirect current queries when they read this bank during prompt processing. The base model remains frozen; only 12,288 parameters are trained. Across three Qwen models and four benchmarks, STAIR improves avera
    
[^256]: 面向大语言模型服务的高带宽闪存特性分析

    Characterizing High Bandwidth Flash for LLM Serving

    [https://arxiv.org/abs/2609.39131](https://arxiv.org/abs/2609.39131)

    本文提出HBM-HBF-主机三层存储架构与缓冲缓存感知调度机制，系统评估了高带宽闪存在高吞吐量LLM智能体服务中何时能够有效提升服务性能与能效。

    

    大语言模型（LLM）服务需要大量内存来存储模型权重和KV缓存。随着模型规模不断增大、上下文长度不断增加，内存容量和带宽日益成为服务性能的瓶颈。智能体工作负载通过在不断增长的上下文上进行反复交互，进一步加剧了这一压力，使得保留KV状态以供重复使用变得愈发重要。高带宽闪存为扩展大语言模型服务的加速器内存容量提供了一种途径，但其访问成本和有限的写入寿命使其应用变得复杂。我们从系统设计和调度选择的角度评估了HBF在高吞吐量智能体服务中的表现，以了解何时增加的容量能够提升服务性能和能效。我们提出了HBM-HBF-主机分层存储系统和缓冲缓存感知调度方法，并使用基于追踪的仿真来分析它们对性能和能耗的影响。

    arXiv:2609.39131v1 Announce Type: new  Abstract: Large language model (LLM) serving requires substantial memory to store model weights and KV caches. As models grow larger and contexts become longer, memory capacity and bandwidth increasingly become bottlenecks for serving performance. Agentic workloads compound this pressure through repeated interactions over growing contexts, making it increasingly important to retain KV state for reuse. High-bandwidth flash (HBF) offers a way to expand accelerator memory capacity for large language model (LLM) serving, but its access costs and limited write endurance complicate its use. We evaluate HBF for high-throughput agentic serving across system design and scheduling choices to understand when additional capacity improves serving performance and energy efficiency. We introduce an HBM-HBF-host hierarchical storage system and buffered cache-aware scheduling, and use trace-driven simulations to analyze their effects on performance, energy consump
    
[^257]: STEPQuant：Delta规则循环状态量化中误差何时何地产生影响

    STEPQuant: When and Where Errors Matter in Delta-Rule Recurrent State Quantization

    [https://arxiv.org/abs/2609.38169](https://arxiv.org/abs/2609.38169)

    STEPQuant揭示了Delta规则循环状态量化误差在时间（记忆寿命）和空间（键行/值列）两个维度上的差异化影响，并提出一种时空后训练量化框架，根据误差大小与记忆寿命自适应分配精度，从而在低比特量化下保持模型精度。

    

    线性注意力机制用固定大小的循环状态取代了不断增长的KV缓存，但这些持久状态在并发服务场景下可能成为显著的内存瓶颈。直接将循环状态量化为低精度往往会导致严重的精度下降，因为量化误差会在连续的状态更新中不断传播。我们发现这些误差的影响取决于两个互补的维度：在时间维度上，长期存活的记忆中的误差会在多个解码步骤中持续存在；在空间维度上，不同键行中的误差对模型输出的影响各不相同，而状态幅值在行和列两个方向上都存在显著差异。基于这些观察，我们提出了STEPQuant，一个面向Delta规则循环状态的时空后训练量化框架。STEPQuant根据误差大小和记忆寿命来分配精度，并基于状态分布联合拟合键行与值列的缩放因子。

    arXiv:2609.38169v1 Announce Type: cross  Abstract: Linear attention replaces growing KV caches with fixed-size recurrent states, yet these persistent states can become a substantial memory bottleneck under concurrent serving. Directly quantizing recurrent states to low precision often leads to severe accuracy degradation, as quantization errors propagate through successive state updates. We discover that the impact of these errors depends on two complementary dimensions: temporally, errors in long-lived memory can persist across many decoding steps; spatially, errors in different key rows affect model outputs differently, while state magnitudes vary substantially along both rows and columns. Motivated by these observations, we propose STEPQuant, a spatial-temporal post-training quantization framework for Delta-rule recurrent states. STEPQuant allocates precision according to error magnitude and memory lifetime, and jointly fits key-row and value-column scales based on state distributio
    
[^258]: FlowMap-OPD：面向少步流映射生成器在线策略蒸馏的展开-核分离方法

    FlowMap-OPD: Rollout--Kernel Separation for On-Policy Distillation of Few-Step Flow-Map Generators

    [https://arxiv.org/abs/2609.37851](https://arxiv.org/abs/2609.37851)

    提出 FlowMap-OPD 在线策略蒸馏框架，通过展开-核分离将学生状态获取与师生分布比较解耦，并通过实验证明带独立可调学生一致性的瞬时速度分布监督是最有效的蒸馏方案。

    

    少步流映射生成器（包括 MeanFlow 和一致性模型）能够通过长程传输实现高效采样，但其在线策略蒸馏仍未得到充分研究。我们提出了 FlowMap-OPD，这是一个在线策略蒸馏框架，它将学生状态的获取与师生分布比较分离开来。基于状态边缘分布的公式化确立了这种分离，而流-速度一致性则将局部监督与部署的长程映射联系起来。在该框架内，我们开发了流映射、诱导速度和瞬时速度分布监督，每种监督均与单独指定的原生流映射展开相配对。在跨容量的 ImageNet 实验中，针对三种教师奖励的对比表明，带有独立可调学生一致性的瞬时速度分布监督是最有效的选择。在文生图实验中，FlowMap-OPD 展现出强大的多特化能力。（注：原文摘要在此处被截断）

    arXiv:2609.37851v1 Announce Type: cross  Abstract: Few-step flow-map generators, including MeanFlow and consistency models, enable efficient sampling through long-range transport, yet their on-policy distillation remains underexplored. We introduce FlowMap-OPD, an on-policy distillation framework that separates student-state acquisition from teacher--student distribution comparison. A formulation based on state marginals establishes this separation, while flow--velocity consistency connects local supervision to the deployed long-range map. Within this framework, we develop flow-map, induced-velocity, and instantaneous-velocity distribution supervision, each paired with a separately specified native flow-map rollout. Cross-capacity ImageNet experiments across three teacher rewards identify instantaneous-velocity distribution supervision with independently tunable student consistency as the most effective choice. In text-to-image experiments, FlowMap-OPD demonstrates strong multi-special
    
[^259]: 预测式自监督学习可在干扰存在下被证明能够识别随机信号

    Predictive Self-Supervised Learning Provably Identifies Stochastic Signals under Nuisance

    [https://arxiv.org/abs/2609.37789](https://arxiv.org/abs/2609.37789)

    该论文证明了常见的预测式自监督学习方法通过“预测互信息最大化”与“潜分布匹配”两个互补原理，能够在干扰信息存在下可证地恢复与预测相关的随机潜在信号。

    

    自监督学习（SSL）通过在潜空间中进行预测，而无需生成输入数据本身，能够学习到高度抽象且有用的表示。直观上，这一成功通常被归因于其能够丢弃与预测无关的干扰信息。然而，这带来了一个难题：与预测相关的潜在信号中的随机变化和真正的干扰因素都会使观测结果部分不可预测；二者如何区分？令人惊讶的是，我们证明了常见的SSL方法恰好能够实现这一点：它们隐式地实例化了一个具有随机动力学和观测私有干扰的潜变量模型。我们将这类方法恢复随机信号的能力归结为两个互补的原理：预测互信息最大化确保表示保留预测所需的信息，而潜分布匹配则约束了这一信息的编码方式，从而使得……（原文摘要在此处截断）

    arXiv:2609.37789v1 Announce Type: cross  Abstract: Self-supervised learning (SSL) by predicting in latent space, without generating the input data itself, learns highly abstract, useful representations. Intuitively, this success is often attributed to its ability to discard nuisance information that is irrelevant to prediction. However, this poses a conundrum: both stochastic variation in a prediction-relevant latent signal and true nuisance make observations partly unpredictable; how could they be distinguished? Surprisingly, we prove that common SSL methods can achieve exactly this, by implicitly instantiating a latent-variable model with stochastic dynamics and observation-private nuisance. We trace their ability to recover the stochastic signal to two complementary principles: Predictive mutual information maximization ensures that representations retain the information needed for prediction, while latent distribution matching constrains how this information is encoded, thereby mak
    
[^260]: 依赖优化器的训练动力学收敛到相同的三分之一最优数据缩放规律

    Optimizer-dependent training dynamics converge to the same one-third optimal data scaling

    [https://arxiv.org/abs/2609.37745](https://arxiv.org/abs/2609.37745)

    研究发现训练动力学中的动态指数依赖于优化器类型，但最优调参后损失随数据量的缩放指数在不同优化器下均收敛于 1/3。

    

    神经缩放律——即损失随训练按幂律下降——是大语言模型的核心，最近有一种观点提出，1/3 指数源于对峰值分布的学习。该解释针对的是 SGD（随机梯度下降），但实际中的模型通常使用自适应优化器进行训练。在这里，我们区分了 1/3 理论未能区分的两个指数：损失在单次训练运行中随训练步数下降的速度，以及经过最优调参后的损失随数据集规模 D 下降的速度。我们表明，第一个指数（动态指数）依赖于具体优化器，而第二个指数（最优数据指数）在不同优化器下均收敛到 1/3。在一个在线师生模型中，我们将损失分解为范数增长（径向）和向教师方向的对齐（切向），每一部分都以幂律衰减，对应的动态指数分别为 α_r 和 α_t。在 SGD 下，两者都接近 1/3，因此数据指数在不同学习率下也表现为 1/3。

    arXiv:2609.37745v1 Announce Type: new  Abstract: Neural scaling, in which loss falls as a power law with training, is central to large language models, and one recent proposal is that a $1/3$ exponent emerges from learning peaked distributions. That account describes SGD, but models in practice are trained with adaptive optimizers. Here we separate two exponents the $1/3$ account does not distinguish: how fast the loss falls with training steps along a single run, and how fast the optimally tuned loss falls with dataset size $D$. We show that the first, a dynamic exponent, is optimizer-specific while the second, an optimal data exponent, converges to $1/3$ across optimizers. In an online teacher-student model we decompose the loss into norm growth (radial) and alignment toward the teacher direction (tangential), each decaying as a power law with dynamic exponents $\alpha_{r}$ and $\alpha_{t}$. Under SGD, both are close to $1/3$, so the data exponent is also $1/3$ across different learn
    
[^261]: 端口哈密顿潜空间深思：缓解测试时计算扩展中的深思漂移悬崖

    Port-Hamiltonian Latent Deliberation: Mitigating the Deliberation Drift Cliff in Test-Time Compute Scaling

    [https://arxiv.org/abs/2609.37351](https://arxiv.org/abs/2609.37351)

    该论文发现了测试时潜空间深思中随思考深度增加而推理崩溃的“深思漂移悬崖”现象，并提出端口哈密顿潜空间深思框架，以同时兼顾表达能力、Lyapunov稳定性与计算效率，从而在深层思考步数下有效抑制漂移、缓解推理性能崩溃。

    

    测试时计算扩展已成为先进机器推理的基石，然而在连续潜表示空间中直接执行迭代深思会暴露出一种灾难性病理现象：深思漂移悬崖。尽管无约束的循环潜空间模型在较短的推理步数（K ≤ 4）下能获得初步的推理收益，但当外推至更深的思考步数（K ≥ 16）时，其推理能力会发生崩溃，在标准逻辑基准测试中性能下降22%至62%。我们通过22轮实证与理论研究，解决了测试时潜空间推理中表达能力、Lyapunov稳定性与计算效率之间的三难困境。我们证明，严格保守的标量势梯度流能够抑制长程漂移（悬崖仅为3.40%），但会将峰值推理准确率瓶颈在32.73%；而无约束的旋转流虽实现了高符号表达能力（82.33%），却会遭受高达36.87%的严重漂移。

    arXiv:2609.37351v1 Announce Type: cross  Abstract: Test-time compute scaling has emerged as a cornerstone of advanced machine reasoning, yet performing iterative deliberation directly within continuous latent representation spaces reveals a catastrophic pathology: the Deliberation Drift Cliff. While unconstrained recurrent latent models achieve initial reasoning gains at short horizons (K <= 4), their reasoning collapses when extrapolated to deeper thinking steps (K >= 16), dropping by 22% to 62% across standard logical benchmarks. We resolve the trilemma among expressivity, Lyapunov stability, and computational efficiency in test-time latent reasoning through a 22-round empirical and theoretical investigation. We demonstrate that strictly conservative scalar potential gradient flows suppress long-range drift (cliff 3.40%) but bottleneck peak reasoning accuracy at 32.73%, whereas unconstrained rotational flows achieve high symbolic expressivity (82.33%) but suffer a severe 36.87% drift
    
[^262]: 面向多元时间序列预测的通道依赖状态空间模型

    Channel-Dependent State Space Model for Multivariate Time Series Forecasting

    [https://arxiv.org/abs/2609.36453](https://arxiv.org/abs/2609.36453)

    提出Chameleon，一种通道依赖的状态空间模型，通过将选择性SSM与卡尔曼滤波器结合，利用其缺失的测量更新机制实现变量间数据依赖的细粒度交互，在保持线性复杂度的同时实现稳健的多元时间序列预测。

    

    多元时间序列预测（MTSF）在许多现实世界领域中至关重要。现有的深度学习方法分为两种范式，各有明显的局限性：通道独立（CI）方法无条件地忽略跨变量依赖关系，仅建模时间动态；而通道依赖（CD）方法虽同时考虑两者，但通常需要在架构上做出折衷以缓解过拟合和计算开销问题。因此，我们提出了Chameleon，一种专门设计的CD状态空间模型（SSM），它能够在变量之间实现数据依赖的细粒度交互，同时计算量随变量数量线性扩展。通过将选择性SSM与卡尔曼滤波器联系起来，我们利用前者缺失的测量更新步骤进行跨变量建模，同时保留SSM主干以实现稳健的时间建模。我们进一步识别出GatedDeltaNet对时间序列任务的有利归纳偏置，将其改造作为模型主干，并加以改进（摘要在此处截断）。

    arXiv:2609.36453v1 Announce Type: cross  Abstract: Multivariate time series forecasting (MTSF) is critical across many real-world domains. Existing deep learning approaches fall into two paradigms with distinct limitations: channel-independent (CI) methods unconditionally ignore cross-variable dependencies and model only temporal dynamics, while channel-dependent (CD) methods consider both but typically rely on architectural compromises to mitigate overfitting and computational overhead. We therefore propose Chameleon, a specialized CD state space model (SSM) that enables data-dependent, fine-grained interactions across variables while scaling linearly with their number. By connecting selective SSMs with the Kalman filter, we leverage the missing measurement update in the former for cross-variable modeling while preserving the SSM backbone for robust temporal modeling. We further identify favorable inductive biases of GatedDeltaNet for time series, adapt it as our backbone, and improve
    
[^263]: 具有最优神经元数量与最优比特复杂度的任意精度神经逼近

    Arbitrary-Accuracy Neural Approximation with Optimal Neuron Count and Optimal Bit Complexity

    [https://arxiv.org/abs/2609.35628](https://arxiv.org/abs/2609.35628)

    本文通过显式构造激活函数与闭式网络，证明了在 $[0,1]^d$ 上对连续函数进行任意精度逼近时 $d+1$ 个隐神经元是精确最优的，并给出了结合取整与指数函数的更简单构造及最优编码复杂度。

    

    我们研究了在 $[0,1]^d$ 上对连续函数进行任意精度逼近所需的最少隐神经元数量以及相关的编码复杂度。对于 $d\geq 2$，我们构造了两个固定的、显式定义的激活函数，并为每个激活函数给出了一个具有宽度为 $d$ 和 $1$ 的两个隐藏层的闭式网络，该网络在一致范数下可实现任意精度逼近。我们证明了 $d+1$ 是在具有局部可积激活函数和仿射输出层的标准前馈网络中，隐神经元总数的精确最小值。我们进一步给出了一种更简单的构造，仅使用一个结合了取整函数与指数函数的初等激活函数。该构造需要宽度为 $d$、$1$ 和 $2$ 的三个隐藏层；如果允许跳跃连接，则宽度为 $d$、$1$ 和 $1$ 即可满足要求。对于一个有界的 $\alpha$-Hölder 类（$\alpha\in(0,1]$），采用数字提取激活函数与初等激活函数的网络……

    arXiv:2609.35628v2 Announce Type: replace  Abstract: We study the minimum number of hidden neurons required for arbitrary-accuracy approximation of continuous functions on $[0,1]^d$ and the associated encoding complexity. For $d\geq 2$, we construct two fixed, explicitly defined activation functions and, for each, a closed-form network with two hidden layers of widths $d$ and $1$ that achieves arbitrary accuracy in the uniform norm. We prove that $d+1$ is the exact minimum total number of hidden neurons among standard feedforward networks with locally integrable activations and affine outputs. We further give a simpler construction using a single elementary activation that combines the floor and exponential functions. This construction requires three hidden layers of widths $d$, $1$, and $2$. If a skip connection is allowed, widths $d$, $1$, and $1$ suffice. For a bounded $\alpha$-H\"older class with $\alpha\in(0,1]$, the networks with the digit-extraction activation and with the eleme
    
[^264]: Tetra：以每参数2.7比特部署Leech格量化的大语言模型

    Tetra: Serving Leech-Lattice Quantized LLMs at 2.7 Bits per Parameter

    [https://arxiv.org/abs/2609.35465](https://arxiv.org/abs/2609.35465)

    Tetra通过在Leech格上设计基于Golay码64状态网格图的新码本和高效解码内核，将每权重读取量从4.804比特降至2.148比特，实现了整模型每参数约2.7比特的高质量大语言模型量化部署。

    

    Leech格量化在每权重2比特的情况下能提供良好的模型质量，但其码本包含超过10^14个点，数量太多而无法用查找表实现。我们之前的内核在加载时展开码字，为2比特的代码需从GPU内存读取每权重4.804比特。我们提出Tetra，这是同一格上的一个新码本。一个24权重的块仍然占用48比特，其中大部分用于索引Golay码的64状态网格图和一个共享的16 KiB表。内核在矩阵向量乘积内部通过六次表加载和两次小型查找解码一个块，每权重仅读取2.148比特。对于完整模型，我们为每个矩阵行重新训练一个缩放因子，将损失最大的矩阵存储为4比特整数，并用4比特嵌入表来补偿其开销。我们的Qwen3-4B、8B和14B模型文件在整个模型上分别保持每参数2.73、2.70和2.73比特。它们在完整MMLU测试集上的得分分别为63.37、69.58和75.66，比4比特AWQ低4.76、4.21和2.46分。

    arXiv:2609.35465v2 Announce Type: replace  Abstract: Leech-lattice quantization gives good quality at two bits per weight, but its codebooks hold more than 10^14 points, too many for a lookup table. Our earlier kernel expanded the codes at load time and read 4.804 bits per weight from GPU memory for 2 bits of code. We present Tetra, a new codebook on the same lattice. A 24-weight block still takes 48 bits, most of which index a 64-state trellis of the Golay code and one shared 16 KiB table. The kernel decodes a block with six table loads and two small lookups inside the matrix-vector product, and reads 2.148 bits per weight. For full models, we retrain one scale per matrix row, store the matrices that lose the most as 4-bit integers, and pay for them with 4-bit embedding tables. Our Qwen3-4B, 8B and 14B files hold 2.73, 2.70 and 2.73 bits per parameter over the whole model. They score 63.37, 69.58 and 75.66 on the full MMLU test set, 4.76, 4.21 and 2.46 points below 4-bit AWQ at 5.3 to
    
[^265]: ProtoSeam：利用隐高斯混合模型提升分类器训练

    ProtoSeam: Lifting Classifier Training with Latent Gaussian Mixture Models

    [https://arxiv.org/abs/2609.35174](https://arxiv.org/abs/2609.35174)

    该论文提出ProtoSeam方法，通过在网络内部语义接口处插入可学习的类别原型并阻断梯度跨接口传播，以隐高斯混合模型的形式重构分类器训练，在不改变推理架构的前提下将测试准确率提升最多六个百分点。

    

    我们提出了一种监督分类的提升式重构方法，能够在不改变推理时网络架构的情况下提高标准分类器的最终准确率。将网络 $N=N_2\circ N_1$ 在单一语义接口处切分，并在该处为每个类别插入一个可学习的原型。训练过程结合了二次一致性惩罚项（将 $N_1(x)$ 拉向其所属类别的原型）以及在围绕原型采样的样本上评估的 $N_2$ 分类损失，同时梯度不跨接口传播。在推理阶段，原型被丢弃，直接使用未经修改的网络 $N_2\circ N_1$。在 CIFAR-10、CIFAR-100 和 TinyImageNet 数据集上，配合 ResNet 和视觉 Transformer 骨干网络，在相同的调优协议下，提升式训练相比未提升的变体可将测试准确率提高最多六个百分点。此外，我们为这些结果提供了理论依据。

    arXiv:2609.35174v2 Announce Type: replace  Abstract: We propose a lifted reformulation of supervised classification that improves the final accuracy of standard classifiers without changing the architecture at inference time. A network $N=N_2\circ N_1$ is split at a single semantic interface and one learnable prototype per class is inserted there. Training combines a quadratic consensus penalty that pulls $N_1(x)$ toward the prototype of its class with a classification loss of $N_2$ evaluated on samples drawn around the prototypes, while no gradient crosses the interface. At inference the prototypes are discarded and the unmodified network $N_2\circ N_1$ is used. Across CIFAR-10, CIFAR-100, and TinyImageNet with ResNet and vision transformer backbones, lifted training improves test accuracy by up to six percentage points over variants without lifting under a shared tuning protocol. Moreover, we provide theoretical justification of those results.
    
[^266]: 信息的双刃剑：学校选择中的公开与隐藏摇号

    The Double-Edged Sword of Information: Revealed versus Hidden Lotteries in School Choice

    [https://arxiv.org/abs/2609.34074](https://arxiv.org/abs/2609.34074)

    本文放宽了“所有学生偏好相同”的假设，在贝叶斯纳什均衡的博弈论模型下证明：学校选择中公开摇号号码是一把双刃剑，既可能提高也可能降低匹配率与社会福利。

    

    在学校选择机制中，当偏好同一所学校的学生数量超过可用名额时，匹配机制通常会使用摇号号码来打破平局。在实践中，近年来纽约市公立学校开始向学生公开摇号号码。先前文献的理论结果表明，在延迟接受算法下，公开摇号号码能够严格提高匹配数量。然而，这些理论结果基于一个过度简化的假设，即所有学生对学校的偏好排序完全相同。在这项工作中，我们放宽了这一假设，允许学生拥有异质的偏好排序。在学生策略构成贝叶斯纳什均衡的博弈论模型下，我们以匹配率和社会福利这两个指标衡量匹配结果，刻画了公开摇号号码既可能改善也可能恶化匹配结果的各类情形。

    arXiv:2609.34074v2 Announce Type: replace-cross  Abstract: In school choice, a lottery number is often used by the matching mechanism to break ties when more students prefer the same school than the number of seats available. In practice, in recent years, the NYC Public Schools started revealing the lottery number to students. Theoretical findings from prior literature suggest that revealing the lottery number strictly improves the number of matches under the deferred acceptance algorithm. However, these theoretical results are based on the over-simplifying assumption that all students share the same preference ranking for schools. In this work, we relax this assumption and allow students to have heterogeneous preference rankings. Under a game-theoretic model where student strategies form a Bayesian Nash equilibrium, we characterize scenarios where revealing the lottery number can either improve or worsen the matching outcome, measured by two metrics of the match rate and social welfar
    
[^267]: 变色龙：面向高效扩散模型的动态格式适配器

    Chameleon: Dynamic Format Adapter for Efficient Diffusion

    [https://arxiv.org/abs/2609.33496](https://arxiv.org/abs/2609.33496)

    Chameleon提出了一种混合格式的扩散模型训练后量化框架，在固定位宽的前提下将数字格式本身作为离散变量，按权重通道、激活层乃至扩散时间步自适应地选择INT8或FP8等格式，从而实现更高效的扩散模型推理。

    

    训练后量化（PTQ）是在内存受限的加速器上运行现代扩散模型的标准方法，然而现有所有的扩散模型PTQ方案都预先固定了数字格式，仅调整缩放因子、零点或逐层位宽。在固定位宽下，最佳格式取决于被编码的分布，而该分布在权重通道之间、层与层之间以及扩散时间步上都各不相同——激活分布会从重尾、噪声主导的形态逐渐演变为紧密聚集且结构化的形态。我们提出Chameleon，这是一个PTQ框架，它保持位宽固定，并将格式本身视为一个离散变量，按权重通道和激活层进行选择，并可选择按时间步分桶进一步细化。激活格式来自{INT8, FP8 E4M3, FP8 E5M2, MXFP8, MXINT8}集合，基于两个低成本的统计量（经验峰度和闭式……[摘要在此处被截断]）提前选定。

    arXiv:2609.33496v2 Announce Type: replace-cross  Abstract: Post-training quantization (PTQ) is the standard way to run modern diffusion models on memory-constrained accelerators, yet every existing diffusion PTQ scheme fixes the $\mathit{number\ format}$ in advance and only tunes the scale, zero point, or per-layer bit-width. At a fixed bit-width the best format depends on the distribution being encoded, and that distribution differs across weight channels, across layers, and along the diffusion timestep, where activation distributions slide from heavy-tailed and noise-dominated to tightly clustered and structured. We propose Chameleon, a PTQ framework that holds the bit-width fixed and treats the format itself as a discrete variable, chosen per weight channel and per activation layer, optionally refined per timestep bucket. Activation formats come from {INT8, FP8 E4M3, FP8 E5M2, MXFP8, MXINT8}, selected ahead of time from two cheap statistics (empirical kurtosis and the closed-form di
    
[^268]: 具有对抗性响应的不可知平滑在线回归

    Agnostic Smoothed Online Regression with Adversarial Responses

    [https://arxiv.org/abs/2609.32478](https://arxiv.org/abs/2609.32478)

    本文提出 Hedge-Cover 算法，利用协变量平滑性界定专家先验中的分歧次数，为具有有界对抗响应的不可知平滑在线回归实现了 $\widetilde{O}(\sqrt{\text{Pdim}(\mathcal{F}) \textsf{C}_{\textsf{cov}} T})$ 的次线性遗憾，从而解决了极小极大最优自适应遗憾这一开放问题。

    

    我们研究具有有界对抗响应的平滑在线预测问题。这一被广泛研究的框架通过平滑参数 $\textsf{C}_{\textsf{cov}}$ 在独立同分布采样与对抗性协变量选择之间架起桥梁，该参数以固定但未知的基准测度为参照，约束条件协变量密度。我们提出了 Hedge-Cover，这是一种信息论算法，对于伪维度有界的函数类，能够实现次线性遗憾 $\widetilde{O}(\sqrt{\text{Pdim}(\mathcal{F}) \textsf{C}_{\textsf{cov}} T})$。该算法使用 Hedge 聚合一个精心构建的专家族，其先验将遗憾与一致性选择器与目标函数之间的分歧次数联系起来。我们通过利用协变量平滑性来界定这一分歧次数。这回答了先前文献中提出的关于平滑在线回归问题极小极大最优自适应遗憾的开放问题。

    arXiv:2609.32478v2 Announce Type: replace-cross  Abstract: We study smoothed online prediction with bounded adversarial responses. This widely studied framework bridges i.i.d. sampling and adversarial covariate selection through a smoothness parameter $\textsf{C}_{\textsf{cov}}$, which bounds conditional covariate densities relative to a fixed, unknown base measure. We propose Hedge-Cover, an information-theoretic algorithm that achieves sublinear regret $\widetilde{O}(\sqrt{\text{Pdim}(\mathcal{F}) \textsf{C}_{\textsf{cov}} T})$ for function classes with bounded pseudo-dimension. The algorithm aggregates a carefully constructed family of experts using Hedge, with a prior that links regret to the number of disagreements between a consistent selector and a target function. We bound this number by exploiting covariate smoothness. This answers an open problem posed in \cite{blanchard2025agnostic} on the minimax optimal adaptive regret of the smoothed online regression problem. We establis
    
[^269]: 用于干预的梯度与用于检测的激活值：语言模型中的定向特征学习

    Gradients for Interventions and Activation Values for Detection: Targeted Feature Learning in Language Models

    [https://arxiv.org/abs/2609.32355](https://arxiv.org/abs/2609.32355)

    该论文提出“定向特征学习”框架，为预定义概念构建单一特征，并系统比较了三种模型信号（激活值、激活梯度、参数梯度）与两种估计器组合而成的六种方法，其中CAA和GRADIEND为已有方法，另外四种为新提出的方法。

    

    模型内部特征可以通过两种方式进行研究：一是识别特定概念的能力，二是被操纵时（例如通过引导或权重编辑）产生的因果效应。特征学习的一种主流方法是稀疏自编码器（SAE），它学习广泛的特征字典，其与特定概念的关系通常是事后确定的。然而，许多可解释性问题恰恰相反，是假设驱动的，涉及预先指定的概念。我们将这一设定称为定向特征学习，即为这样一个预定义概念构建单一特征。我们提出了一个受控比较，涵盖三种模型信号（激活值、激活梯度和参数梯度）和两种估计器（对比均值和可学习的一维编码器-解码器），由此产生六种定向方法，其中CAA和GRADIEND是已有方法，另外四种新方法覆盖了其余的组合。

    arXiv:2609.32355v2 Announce Type: replace  Abstract: Model-internal features can be studied through both their ability to identify a specified concept and their causal effect when manipulated, e.g., through steering or weight editing. A prominent approach to feature learning is Sparse Autoencoders (SAEs), which learn broad feature dictionaries whose relation to particular concepts is typically identified post hoc. However, many interpretability questions are instead hypothesis-driven and concern a concept specified in advance. We study this setting as targeted feature learning, where a single feature is constructed for such a predefined concept. We present a controlled comparison across three model signals (activation values, activation gradients, and parameter gradients) and two estimators (contrastive mean and a learned one-dimensional encoder-decoder), yielding six targeted methods, with CAA and GRADIEND as existing instances and four new methods covering the remaining combinations.
    
[^270]: KernelOPT：面向GPU内核优化的调度感知智能体搜索

    KernelOPT: Dispatch-Aware Agentic Search for GPU Kernel Optimization

    [https://arxiv.org/abs/2609.30059](https://arxiv.org/abs/2609.30059)

    KernelOPT是一个调度感知的多智能体GPU内核优化系统，它在保留厂商库调用的同时仅优化编译器生成的Triton子内核，并通过静态校验、多种子正确性、模型级float64回退与性能门控组成的四道验证级联确保端到端的正确性与加速。

    

    深度学习的推理与训练性能在很大程度上取决于GPU内核的效率。现代编译器（如PyTorch Inductor）能够从高层模型代码自动生成GPU内核，但其性能常常大幅落后于专家手写的实现。近期基于大语言模型（LLM）辅助的内核优化器虽然能缩小独立内核方面的这一差距，但它们将编译后的模型视为黑盒，通常只优化单个独立内核，既不尊重编译器的结构性决策，也不进行模型级的端到端验证。我们提出了KernelOPT，一个将编译后模型视为结构化产物的多智能体系统。该系统保留厂商库调用（cuBLAS、cuDNN），仅针对生成的Triton子内核，并使用五个由性能剖析引导的LLM智能体进行优化。一个由静态校验、多种子正确性验证、模型级float64回退验证以及性能门控组成的四道验证级联，在优化过程中对候选内核进行筛选（原文摘要在此处截断）。

    arXiv:2609.30059v1 Announce Type: cross  Abstract: Deep learning inference and training performance depends critically on GPU kernel efficiency. Modern compilers such as PyTorch Inductor automatically generate GPU kernels from high-level model code, but frequently underperform expert-written implementations by wide margins. Recent LLM-assisted kernel optimizers can close this gap for standalone kernels, yet treat compiled models as black boxes, generally optimizing individual standalone kernels without respecting the compiler's structural decisions or verifying the model end-to-end. We present KernelOPT, a multi-agent system that treats compiled models as structured artifacts. It preserves vendor library calls (cuBLAS, cuDNN) and exclusively targets generated Triton sub-kernels using five profiling-guided LLM agents. A four-gate verification cascade of static validation, multi-seed correctness, model-level float64-fallback verification, and performance gating filters candidates during 
    
[^271]: 线性RNN缩放定律：更长的序列何时胜过更多的序列

    Linear RNN Scaling Laws: When Longer Sequences Beat More Sequences

    [https://arxiv.org/abs/2609.27964](https://arxiv.org/abs/2609.27964)

    该论文在教师-学生理论框架下首次为线性RNN语言模型推导出显式的近似、优化与统计缩放定律，并揭示了在特定谱指数条件下，增加序列长度比增加序列数量能更有效地提升模型性能。

    

    自回归语言模型的经验缩放定律将预测损失与模型规模、数据规模和优化计算量联系起来，但其在序列预训练设置中的理论起源仍然缺乏深入理解。我们在一个可解析的教师-学生模型中研究该问题：一个稳定的潜在线性RNN生成轨迹，一个采用草图（sketch）方法的线性递归学生模型通过带安全防护的全批量WSD梯度下降在下一词元预测任务上训练。草图维度 $M$ 扮演模型规模的角色，而 $N$ 条长度为 $P$ 的独立轨迹提供训练词元。我们允许创新协方差与初始化协方差具有不同的幂律指数 $\alpha$ 和 $\theta$。由此产生的设计谱给出了由谱交叉点分隔的显式近似、优化和统计缩放定律。当 $\theta\ge\alpha$ 时，原始的单尺度速率 $M^{1-\beta_\alpha}$、$R^{(1-\beta_\alpha)/\alpha}$……（摘要在此处被截断）

    arXiv:2609.27964v1 Announce Type: new  Abstract: Empirical scaling laws for autoregressive language models relate prediction loss to model size, data size, and optimization compute, but their theoretical origin is still poorly understood in sequential pretraining settings. We study this question in a tractable teacher--student model where a stable latent linear RNN generates trajectories and a sketched linear recurrent student is trained by safeguarded full-batch WSD gradient descent on next-token prediction. The sketch dimension $M$ plays the role of model size, while $N$ independent trajectories of length $P$ provide the training tokens. We allow the innovation and initialization covariances to have different power-law exponents $\alpha$ and $\theta$. The induced design spectrum produces explicit approximation, optimization, and statistical scaling laws separated by spectral crossovers. When $\theta\ge\alpha$, the original one-scale rates $M^{1-\beta_\alpha}$, $R^{(1-\beta_\alpha)/\a
    
[^272]: Visual Jev：基于共享视觉上下文的准确高效决策

    Visual Jev: Accurate and Efficient Decisions from Shared Visual Context

    [https://arxiv.org/abs/2609.25845](https://arxiv.org/abs/2609.25845)

    Visual Jev 通过对同一图像只编码一次并批量执行多个问题后缀，在将准确率从 70.6% 提升至 76.1% 的同时，实现了最高 8.9 倍的推理加速。

    

    许多视觉应用会对同一张图像提出多个相互独立的强制选择题。Visual Jev 对图像和公共上下文只进行一次编码，将各个孤立的问题后缀作为一个批次执行，并从骨干网络的语言模型头读取候选概率。在四个基准测试中，答案监督的后训练将等权重宏平均准确率从 70.6% 提升至 76.1%，且增益集中在训练所涵盖的两个任务族上。在每张图像 N=32 个问题的设置下，共享批量执行在预热摊销时间上比独立串行执行快 8.9 倍，并且比一个已经批量化但需要重新计算前缀的基线快 3.4 倍，代价是更高的峰值内存占用。一个匹配的类型化头对照实验表明，相较于语言模型头读出方式，它并没有一致的准确率优势。因此，所支持的设计非常简单：为提升质量而适配骨干网络，保留现有的读出方式，并通过共享执行来提高效率。

    arXiv:2609.25845v1 Announce Type: cross  Abstract: Many vision applications ask several independent, forced-choice questions about the same image. Visual Jev encodes the image and public context once, executes isolated question suffixes as a batch, and reads candidate probabilities from the backbone's language-model head. Across four benchmarks, answer-supervised post-training raises equal-weight macro accuracy from 70.6% to 76.1%, with the gain concentrated on the two task families represented in training. At N=32 questions per image, shared batched execution is 8.9x faster in warm amortized time than independent serial execution and remains 3.4x faster than an already-batched baseline that recomputes the prefix, at the cost of higher peak memory. A matched typed-head control offers no consistent accuracy advantage over the language-model-head readout. The supported design is therefore simple: adapt the backbone for quality, retain the existing readout, and share execution for efficie
    
[^273]: 面向广义线性混合效应模型的先验摊销上下文贝叶斯推断

    Prior-Amortized In-Context Bayesian Inference for Generalized Linear Mixed-Effects Models

    [https://arxiv.org/abs/2609.24422](https://arxiv.org/abs/2609.24422)

    提出metabeta——一个预训练神经网络，在测试时接受任意先验族和超参数输入，利用集合变换器与条件归一化流对GLMM进行先验摊销的上下文贝叶斯推断，无需MCMC即可零样本泛化并获得校准的后验估计。

    

    分层数据在实证科学中无处不在，通常使用广义线性混合效应模型（GLMM）进行分析。对GLMM进行贝叶斯推断能够提供校准的不确定性量化，但需要MCMC采样；No-U-Turn采样器（NUTS）是金标准，但速度缓慢，且每遇到新的数据集、模型和先验都必须从头开始。我们提出了metabeta，一个用于GLMM先验摊销上下文贝叶斯推断的预训练神经网络。与以往在训练时固定先验的神经后验估计器不同，metabeta在测试时接受先验族和超参数作为输入，从而实现零样本泛化。两个集合变换器与条件归一化流镜像了后验的两级结构（跨组共享的全局参数与每组的局部参数）。该模型在数百万个涵盖连续、二元和计数结果的真实感模拟数据集上进行训练。

    arXiv:2609.24422v1 Announce Type: new  Abstract: Hierarchical data is ubiquitous in the empirical sciences and is most commonly analyzed with generalized linear mixed-effects models (GLMMs). Bayesian inference for GLMMs yields calibrated uncertainty but requires MCMC; the No-U-Turn Sampler (NUTS) is the gold standard but is slow and must restart from scratch for every new dataset, model and prior. We introduce metabeta, a pretrained neural network for prior-amortized in-context Bayesian inference over GLMMs. Unlike previous neural posterior estimators that fix the prior at training time, metabeta accepts prior families and hyperparameters as inputs at test time, enabling zero-shot generalization. Two set transformers and conditional normalizing flows mirror the posterior's two-level structure (global parameters shared across groups, local parameters per group). The model is trained on millions of realistic simulated datasets spanning continuous, binary, and count outcomes. By default, 
    
[^274]: Swiss-Knife：解码时可重构的外部化多目标对齐框架

    Swiss-Knife: A Framework for Reconfigurable Externalised Multi-Objective Alignment at Decode Time

    [https://arxiv.org/abs/2609.22226](https://arxiv.org/abs/2609.22226)

    该论文提出Swiss-Knife框架，将解码时的多目标对齐规范变为可热插拔的运行时对象，并通过表示定理刻画了聚合算子族，证明成对聚合在对抗性奖励污染下比argmax更稳定。

    

    解码时对齐方法通过使用外部奖励对候选续写进行评分并选择得分最大者，来引导一个冻结的语言模型。我们认为这种共享设计只是更大空间中的一个退化点。我们提出Swiss-Knife，一个外部化多目标对齐框架，其中对齐规范是一等运行时对象：可热插拔的评分刀片、批归一化器、成对聚合算子和选择规则。六个公理刻画了可容许的聚合算子，并且我们证明了一个表示定理：每个满足这些公理的算子都具有形式 $R_i = \sum_{j \neq i} g((\mu_i - \mu_j)/s(\sigma_i,\sigma_j))$，这是一个双参数族，其中包含probit和logistic比较规则以及逐点argmax作为命名的坐标。在该族中，成对聚合在对抗性奖励污染下是Lipschitz稳定的，而argmax则不是，且候选批归一化……（原文摘要在此处截断）

    arXiv:2609.22226v1 Announce Type: new  Abstract: Decode-time alignment methods steer a frozen language model by scoring candidate continuations with an external reward and selecting the maximiser. We argue that this shared design is a single degenerate point in a much larger space. We introduce Swiss-Knife, a framework for externalised multi-objective alignment in which the alignment specification is a first-class runtime object: hot-swappable scoring blades, a batch normaliser, a pairwise aggregation operator, and a selection rule. Six axioms characterise the admissible aggregation operators, and we prove a representation theorem: every operator satisfying them has the form $R_i = \sum_{j \neq i} g((\mu_i - \mu_j)/s(\sigma_i,\sigma_j))$, a two-parameter family containing probit and logistic comparison rules and pointwise argmax as named coordinates. Within it, pairwise aggregation is Lipschitz-stable under adversarial reward contamination while argmax is not, and Candidate-Batch Norma
    
[^275]: Paint-Anything：面向图像生成与编辑的统一任意颜色控制

    Paint-Anything: Unified Any-Color Control for Image Generation and Editing

    [https://arxiv.org/abs/2609.20816](https://arxiv.org/abs/2609.20816)

    Paint-Anything通过对象级颜色监督学习共享的十六进制提示接口，结合Paint-500K数据集和精确匹配的纯色锚点，实现了图像生成与编辑中任意精确颜色的统一控制。

    

    专业设计需要任意颜色控制：即能够在图像生成和编辑中为对象指定任意24位十六进制值的目标颜色。先前的工作已探索了颜色生成、编辑和上色，但通常依赖于专用的颜色表示或专门的推理流程。大语言模型的进展提供了一个更简单的起点：即使是紧凑的模型也能将十六进制值与颜色语义关联起来。我们提出了Paint-Anything，它通过对象级颜色监督学习一个用于生成和编辑的共享十六进制提示接口。我们开发了一个数据管道，通过对象定位、感知颜色标注和编辑对合成，从真实图像构建了Paint-500K数据集。由于阴影使得真实图像的标签只能提供近似颜色，我们用纯色锚点来补充这种监督，这些锚点的像素与其配对的十六进制值完全匹配。这些锚点仅在高噪声阶段使用。

    arXiv:2609.20816v1 Announce Type: cross  Abstract: Professional design requires any-color control: the ability to specify an object's target color with any 24-bit hex value for image generation and editing. Prior work has explored color generation, editing, and colorization, but often relies on dedicated color representations or specialized inference procedures. Advances in large language models offer a simpler starting point: even compact models can associate hex values with color semantics. We present Paint-Anything, which learns a shared hex-prompt interface for generation and editing through object-level color supervision. We develop a data pipeline that constructs Paint-500K from real images through object grounding, perceptual color labeling, and editing-pair synthesis. Since shadows make real-image labels only approximate colors, we complement this supervision with pure-color anchors whose pixels exactly match their paired hex values. These anchors are used only at high-noise ti
    
[^276]: QVAC Genesis III：一个用于高效语言模型预训练的大规模高质量开放合成STEM语料库

    QVAC Genesis III: A Large-Scale, High-Quality Open Synthetic STEM Corpus for Efficient Language Model Pre-Training

    [https://arxiv.org/abs/2609.19513](https://arxiv.org/abs/2609.19513)

    提出了QVAC Genesis III——一个包含1914.3亿token的开放STEM合成语料库，通过以弱学生模型信号驱动的双重生成策略（将失败转化为纠正性解释、将成功扩展为对比性选项级推理），为token预算受限的小模型高效预训练提供了高价值数据。

    

    高质量预训练数据是面向边缘AI和端侧部署的教育及STEM专用语言模型的关键瓶颈，在这些场景中token预算受到严格限制。尽管各大机构在私有语料库上训练越来越大的模型，但开放生态系统中缺乏能够以高效方式为小模型提供高单token学习价值的STEM导向合成数据集。为填补这一空白，我们推出了QVAC Genesis III，这是一个拥有1914.3亿token、以STEM为核心的多领域合成语料库，涵盖19个领域，并包含多个难度级别和不同的教育风格。QVAC Genesis III通过一种双重生成策略构建，该策略以一个弱小的边缘规模学生模型作为信号进行针对性教师蒸馏：学生的失败被转化为纠正性解释，而其成功则被扩展为针对所有答案选项的对比性选项级推理。我们进一步引入了LLM作为解析器的机制……（原文摘要在此处截断）

    arXiv:2609.19513v1 Announce Type: new  Abstract: High-quality pre-training data is a critical bottleneck for educational and STEM-specific language models targeting edge AI and on-device deployment where token budgets are tightly constrained. While major organizations train ever-larger models on private corpora, the open ecosystem lacks STEM-focused synthetic datasets that deliver high per-token learning value efficiently for small models. To address this gap, we introduce QVAC Genesis III, a 191.43B-token, STEM-focused multi-domain synthetic corpus covering 19 domains across several difficulty levels and different educational styles. QVAC Genesis III is built via a dual generation strategy that performs targeted teacher distillation using a weak edge-scale student model as signal: the student's failures are converted into corrective explanations, while its successes are expanded into contrastive option-level reasoning over all answer choices. We further introduce an LLM-as-a-parser ev
    
[^277]: 用于重大不良心血管事件机会性预测的乳腺X线摄影基础模型

    Mammography Foundation Models for Opportunistic Prediction of Major Adverse Cardiovascular Events

    [https://arxiv.org/abs/2609.19385](https://arxiv.org/abs/2609.19385)

    该研究表明，为乳腺癌任务预训练的乳腺X线摄影基础模型无需心血管专门监督或乳腺动脉钙化标注，即可利用常规乳腺筛查影像机会性地预测女性未来5年的重大不良心血管事件风险（AUROC约0.82）。

    

    心血管疾病（CVD）仍然是女性死亡的首要原因，然而心血管风险评估通常依赖于在常规诊疗中可能缺失、过时或无法获得的临床变量。筛查性乳腺X线摄影为心血管风险的机会性分层提供了机会，因为它是常规获取的影像，并且包含与心血管风险和事件相关的血管特征，例如乳腺动脉钙化（BAC）。我们评估了最初为乳腺癌相关任务预训练的乳腺X线摄影专用基础模型，能否在缺乏心血管专门监督或明确BAC标注的情况下，迁移到心血管风险预测任务。我们构建了一个包含22,497名女性的5年重大不良心血管事件（MACE）队列，并将其与电子健康记录结局相关联，其中包含500个事件（患病率2.22%）。基础模型实现了0.823和0.822的AUROC……

    arXiv:2609.19385v1 Announce Type: cross  Abstract: Cardiovascular disease (CVD) remains the leading cause of death among women, yet cardiovascular risk assessment often relies on clinical variables that may be missing, outdated, or unavailable in routine care. Screening mammography offers an opportunity for opportunistic cardiovascular risk stratification because it is routinely acquired and contains vascular features, including breast arterial calcifications (BAC), that are associated with cardiovascular risk and events. We evaluate whether mammography specific foundation models, originally pretrained for breast cancer-related tasks, can transfer to cardiovascular risk prediction without cardiovascular specific supervision or explicit BAC annotation. We constructed a 5-year major adverse cardiovascular event (MACE) cohort of 22,497 women linked to electronic health record outcomes, including 500 events (2.22% prevalence). The foundation models achieved AUROCs of 0.823 and 0.822 substa
    
[^278]: 停止去除停用词：一项沿袭的预处理默认设置如何扭曲法律文本即数据研究

    Stop Removing Stopwords: How an Inherited Preprocessing Default Distorts Legal Text-as-Data

    [https://arxiv.org/abs/2609.19153](https://arxiv.org/abs/2609.19153)

    本研究通过穷尽式单词消融实验，首次直接针对下游分类目标验证停用词去除这一沿袭自信息检索时代、从未被验证过的预处理默认设置，揭示其可能扭曲实证法学中基于TF-IDF和线性分类器的文本数据分析结果。

    

    实证法学研究日益将司法文本视为数据，其中许多研究仍依赖稀疏、可解释的处理流程——TF-IDF特征和线性分类器——因为文本特征往往是研究对象本身，而不仅仅是实现预测的手段。然而，这些流程继承了一系列源自二十世纪中叶信息检索领域的预处理默认设置，这些设置从未针对分类准确度进行过验证，其中最根深蒂固的是停用词去除。本研究引入了一种穷尽式的单词消融方法，直接根据下游目标衡量预处理步骤的效果，并将其应用于停用词去除这一最难撼动的案例。通过将最高法院数据库的标签与Caselaw Access Project的判决意见文本相匹配，该研究考察了两个涵盖F1提升空间的双分类任务：意识形态方向（不去除停用词的基线F1约0.68）和宪法与非宪法法律类型（约0.92），涉及7,66

    arXiv:2609.19153v1 Announce Type: new  Abstract: Empirical legal scholarship increasingly treats judicial text as data, and much of it still runs on sparse, interpretable pipelines -- TF-IDF features and linear classifiers -- because the textual feature is often the object of study, not merely a means to a prediction. Yet these pipelines inherit a chain of preprocessing defaults from mid-century information retrieval that were never validated against classification accuracy, the most entrenched being stopword removal. This study introduces an exhaustive single-word ablation that measures a preprocessing step's effect directly against the downstream objective, and applies it to stopword removal as the hardest case to dislodge. Matching Supreme Court Database labels to Caselaw Access Project opinion texts, it examines two binary tasks that bracket F1 headroom, ideological direction (no-removal baseline F1 ~ 0.68) and constitutional versus non-constitutional law type (~ 0.92), across 7,66
    
[^279]: 我们的评分方式是否恰当？理解医学基准测试中的失败模式

    Are We Grading Properly? Understanding Failure Modes in Medical Benchmarks

    [https://arxiv.org/abs/2609.16023](https://arxiv.org/abs/2609.16023)

    该论文提出RIFT评分标准失败分类法，对两个医学临床基准测试进行系统检测，发现大量评分标准存在非原子性和错位僵化等缺陷，且这些缺陷并非表面问题，仅重写捆绑标准就能导致相同回答的分数变化高达15.9个百分点。

    

    医学评估正在从基于静态选项的提问转向具有开放式输出模式的真实临床场景。然而，简单地大规模评分成本高昂，基于评分标准的评估已成为主流的可扩展替代方案。我们探讨了当评分标准本身不够严密时会发生什么，以及这些缺陷是否能够被检测和纠正。我们将RIFT——一个全局评分标准失败分类法——应用于两个临床基准测试（HealthBench Professional和LiveMedBench），发现失败模式具有重要意义：在HealthBench Professional上，LLM评判者将29.6%的评估标准标记为非原子性，将65.4%标记为错位或僵化。随后，我们证明这些缺陷具有实质性影响，而不仅仅是表面问题。例如，将“以下至少一项/全部满足”形式的捆绑标准重写为等权重的子标准并对相同响应重新评分，会使受影响对话的分数发生高达15.9个百分点的变化。

    arXiv:2609.16023v1 Announce Type: new  Abstract: Medical evaluation is shifting from static option-based questioning to realistic clinical scenarios with open-ended output modes. Grading these at scale naively, however, is expensive, and rubric-based evaluation has become the dominant scalable alternative. We ask what happens when the rubrics themselves are not airtight, and whether such flaws can be detected and corrected. We apply RIFT, a global rubric failure taxonomy, to two clinical benchmarks (HealthBench Professional and LiveMedBench), and find failure modes are meaningful: on HealthBench Professional an LLM judge flags 29.6% of criteria as non-atomic and 65.4% as misaligned/rigid. Then, we show that these flaws are meaningful and not simply cosmetic. As an example, rewriting bundled criteria of the form "at least one of / all of the following" as equally weighted children and regrading identical responses shifts scores by up to 15.9 percentage points on affected conversations, 
    
[^280]: 连续隐式模型中的直接拓扑追踪

    Direct Topology Tracking in Continuous Implicit Models

    [https://arxiv.org/abs/2609.12157](https://arxiv.org/abs/2609.12157)

    提出了一种通过直接查询连续隐式模型及其导数来追踪拓扑特征演化的框架，无需将数据重采样到离散网格，从而在避免离散化伪影的同时实现忠实的特征追踪，并适用于解析函数、多元函数逼近和隐式神经表示等多种隐式模型。

    

    我们提出了一个在连续隐式模型中直接追踪拓扑特征的框架。此类模型包括隐式神经表示（INRs）和多元函数逼近（MFAs），正越来越多地被用于表示科学数据，而不受离散网格分辨率限制的约束。它们为复杂场提供了紧凑、平滑且可微的表示，为高性能数据存储、重建和分析开辟了新的机会。给定一个连续隐式模型，我们的方法通过查询模型及其导数来追踪临界点的演化，从而无需重新采样到网格上。这种方法能够实现忠实的特征追踪，同时避免离散化引起的混叠等伪影。我们展示了该框架在各种隐式表示上的通用性，包括解析函数、MFAs和INRs，并表明它可以……

    arXiv:2609.12157v1 Announce Type: cross  Abstract: We present a framework for tracking topological features directly within continuous implicit models. Such models, including implicit neural representations (INRs) and multivariate functional approximations (MFAs), are increasingly adopted to represent scientific data without the resolution constraints of discrete grids. They offer compact, smooth, and differentiable representations of complex fields, enabling new opportunities for high-performance data storage, reconstruction, and analysis. Given a continuous implicit model, our method tracks the evolution of critical points by querying the model and its derivatives, thereby eliminating the need to resample onto a grid. This approach enables faithful feature tracking while avoiding discretization-induced artifacts such as aliasing. We demonstrate the generality of our framework across a range of implicit representations, including analytic functions, MFAs, and INRs, and show that it pr
    
[^281]: 银速率对梯度下降加速是（几乎）最优的

    Silver Rate Is (Almost) Optimal for Gradient Descent Acceleration

    [https://arxiv.org/abs/2609.09152](https://arxiv.org/abs/2609.09152)

    本文证明了银速率指数是梯度下降加速的最优多项式收敛速率，通过建立非anytime与anytime两种设定下匹配的下界，结合已有上界完整刻画了预定非负步长所能达到的最优加速效果。

    

    我们研究在光滑凸优化中，通过预先确定的非负步长，梯度下降（GD）能够被加速到何种程度。记 $p_{\mathrm{sil}}=\log_2(1+\sqrt{2})$，我们证明了 $\Omega\left(n^{-p_{\mathrm{sil}}-O(\sqrt{\log\log n/\log n})}\right)$ 的非anytime下界。在anytime设定下，每一个无限的非负步长调度都存在无穷多个终止时刻，其误差为 $\Omega\left(n^{-\frac{2p_{\mathrm{sil}}}{1+p_{\mathrm{sil}}}-O(\sqrt{\log\log n/\log n})}\right)$。结合银调度的上界 [Altschuler and Parrilo, 2025] 与anytime上界 [Zhang et al., 2025]，我们的结果确定了这两种设定下的最优多项式收敛指数。

    arXiv:2609.09152v1 Announce Type: cross  Abstract: We study how far gradient descent (GD) can be accelerated by predetermined nonnegative stepsizes in smooth convex optimization. Writing $p_{\mathrm{sil}}=\log_2(1+\sqrt{2})$, we prove an $\Omega\left(n^{-p_{\mathrm{sil}}-O(\sqrt{\log\log n/\log n})}\right)$ non-anytime lower bound. In the anytime setting, every infinite nonnegative schedule has infinitely many horizons with error $\Omega\left(n^{-\frac{2p_{\mathrm{sil}}}{1+p_{\mathrm{sil}}}-O(\sqrt{\log\log n/\log n})}\right)$. Together with the silver-schedule upper bound [Altschuler and Parrilo, 2025] and the anytime upper bound [Zhang et al., 2025], our results determine the optimal polynomial convergence exponents in both settings.
    
[^282]: 通过Flash Attention快速计算高斯和

    Fast Gauss Sums via Flash Attention

    [https://arxiv.org/abs/2609.04910](https://arxiv.org/abs/2609.04910)

    该论文发现仅需两次简单的输入增广，就能借助高度优化的Flash Attention计算任意带符号权重的高斯核和，无需编写任何自定义GPU代码，且在速度、内存开销和精度上均显著超越PyTorch和PyKeOps等现有方案。

    

    高斯核和是最大均值差异（MMD）、核梯度流、Stein变分梯度下降（SVGD）以及许多其他核方法的计算核心。与此同时，softmax注意力机制已经获得了大量硬件感知的代码工程投入，最终催生了Flash Attention。我们证明了具有任意带符号权重的高斯核和可以通过Flash Attention来计算：只需两次简单的输入增广操作，即可将归一化的softmax归约转换为非归一化的高斯和，而无需编写任何一行自定义GPU代码。对于fp16精度下特征维度D>8的情况，该方法在速度、内存开销和精度方面均优于编译后的PyTorch代码和PyKeOps内核（且往往优势显著）。事实上，其内存占用始终保持线性扩展。

    arXiv:2609.04910v1 Announce Type: new  Abstract: Gaussian kernel sums are the computational core of maximum mean discrepancies (MMDs), kernel gradient flows, Stein variational gradient descent (SVGD), and many other kernel methods. At the same time, softmax attention has received an extraordinary amount of hardware-aware code engineering, culminating in flash attention. We show that Gauss kernel sums with arbitrary, signed weights can be evaluated via flash attention: two small input augmentations turn the normalized softmax reduction into the unnormalized Gauss sum, without writing a single line of custom GPU code. For feature dimension D>8 in fp16, this approach beats compiled PyTorch code as well as PyKeOps kernels (often significantly) in speed, memory-overhead and accuracy. Indeed, its memory scaling remains linear.
    
[^283]: 现代Transformer是隐式混合体：从功能分化到有原则的混合架构设计

    Modern Transformers Are Implicit Hybrids: From Functional Differentiation to Principled Hybrid Architecture Design

    [https://arxiv.org/abs/2609.02986](https://arxiv.org/abs/2609.02986)

    本文提出RFIS和RPD两个干预指标，发现现代Transformer的注意力头自然分化为由全局位置频带（GPBand）分隔的检索头和位置头两类，为有原则地设计FA-LA混合架构提供了实证基础。

    

    结合全注意力（FA）和线性注意力（LA）的混合架构日益受到关注，但其注意力分配方式仍然依赖于启发式规则。我们在基于RoPE的Transformer所学习到的头级别功能组织中寻求一个有证据支撑的基础。行为探针无法给出完整的分类体系，因此我们提出了两个干预指标：RoPE频率重要性分数（RFIS），衡量每个频率如何影响某个头的注意力分布；以及RoPE位置依赖性（RPD），用于分离注意力头对旋转位置调制的依赖程度。在Qwen3系列模型和Llama3.1上，RFIS给出假设且RPD加以验证，形成了一个完整的分类体系，将由显著中低频带分隔的检索头和位置头区分开来。受控Transformer实验表明，这一边界遵循训练长度的位置尺度；我们将其命名为全局位置频带。该分析揭示了零样本长度外推失败的潜在原因，并产生了

    arXiv:2609.02986v1 Announce Type: new  Abstract: Hybrid architectures combining Full Attention (FA) and Linear Attention (LA) are increasingly prominent, yet their allocation remains heuristic. We seek an evidence-grounded basis in head-level functional organization learned by RoPE-based Transformers. Behavioral probes do not yield a complete taxonomy, so we propose two intervention metrics: RoPE Frequency Importance Score (RFIS), measuring how each frequency affects a head's attention distribution, and RoPE Positional Dependence (RPD), isolating dependence on rotary positional modulation. On Qwen3-series models and Llama3.1, RFIS suggests and RPD verifies a complete taxonomy of retrieval and positional heads separated by a salient mid-low-frequency band. Controlled Transformers show that this boundary follows the training-length positional scale; we term it the Global Positional Band (GPBand). The analysis suggests a potential cause of zero-shot length-extrapolation failure and yields
    
[^284]: 基准的平衡：面向基准多重性与任务条件评估的语义密度重加权

    Balance of Benchmarks: Semantic Density Reweighting for Benchmark Multiplicity and Task-Conditioned Evaluation

    [https://arxiv.org/abs/2608.30044](https://arxiv.org/abs/2608.30044)

    提出基准平衡方法，通过对基准分配逆密度语义权重来纠正基准密集区域的重复计算问题，并支持基于任务查询的条件化模型评估，在586个模型上的留出任务预测相关性从等权加权的0.049大幅提升至0.462。

    

    语言模型通常通过对基准列表中的分数进行等权平均来进行比较。这类列表通过公开发表而不断增长，缺乏显式的测量设计，因此等权加权使已发表基准的密度成为隐含的能力权重：基准密集的区域会被重复计算。我们提出了基准平衡方法，该方法对基准描述进行嵌入，并为每个基准分配逆密度语义权重。在公开的密度尺度下，相近的基准条目共享聚合影响力。在将异构分数统一到共同潜在尺度后，残差场利用相同的几何结构，使模型排名能够以任务查询为条件。这两个组成部分在实证上发挥着不同的作用。在包含586个模型和14个基准的数据快照上，BoB能够预测哪些模型在留出任务上表现出超出其一般能力的异常强势，画像相关性达到0.462，而等权加权下仅为0.049。

    arXiv:2608.30044v1 Announce Type: new  Abstract: Language models are commonly compared by averaging scores across a benchmark list with equal weight. Such lists grow through publication outside an explicit measurement design, so equal weighting turns the density of published benchmarks into an implicit capability weight: densely benchmarked regions count repeatedly. We introduce Balance of Benchmarks (BoB), which embeds benchmark descriptions and assigns each benchmark an inverse-density semantic weight. Nearby entries share aggregate influence at a disclosed density scale. After equating heterogeneous scores onto a common latent scale, a residual field uses the same geometry to condition model rankings on a task query. The two components serve distinct empirical roles. On a snapshot of 586 models and 14 benchmarks, BoB predicts which models are unusually strong on a held-out task beyond their general ability, reaching a profile correlation of 0.462 compared with 0.049 under equal weig
    
[^285]: 在实时检测流程中实现近零开销的多模型层次分类

    Achieving Near-Zero-Overhead Multi-Model Hierarchical Classification in Real-Time Detection Pipelines

    [https://arxiv.org/abs/2608.11770](https://arxiv.org/abs/2608.11770)

    本文提出一种五步方法论，实现分类骨干网络在NVIDIA Jetson DLA核心上的零GPU回退INT8部署，通过架构适配和手动动态范围调整，在实时检测流程中实现近零开销的多模型层次分类。

    

    摘要：arXiv:2608.11770v1 公告类型：交叉 摘要：在目标识别、监控、自动驾驶和无人机领域的边缘部署视觉系统需要层次化推理流程，其中检测模型识别感兴趣的对象，下游分类器提供细粒度的属性分析。在GPU上运行所有模型会造成串行瓶颈，随着流程阶段增多，限制了实时吞吐量。现代边缘SoC将GPU与专用神经加速器（NPU、DLA）配对，支持并发执行，但由于严格的算子约束、量化不兼容性以及未文档化的端到端流程，在这些加速器上部署自定义模型仍不切实际。我们以NVIDIA Jetson DLA核心作为代表性平台。我们提出了一种五步方法论，用于分类骨干网络的零GPU回退DLA INT8部署，包括架构适配、手动动态范围规避以挽救TensorRT的隐式量化。

    arXiv:2608.11770v1 Announce Type: cross  Abstract: Edge-deployed vision systems in target recognition, surveillance, autonomous vehicles, and drone domains require hierarchical inference pipelines where a detection model identifies objects of interest and downstream classifiers provide fine-grained attribute analysis. Running all models on the GPU creates a serial bottleneck that limits real-time throughput as pipeline stages grow. Modern edge SoCs pair GPUs with dedicated neural accelerators (NPUs, DLAs) capable of concurrent execution, yet deploying custom models on these accelerators remains impractical due to strict operator constraints, quantization incompatibilities, and an undocumented end-to-end pipeline. We target NVIDIA Jetson DLA cores as the representative platform. We present a five-step methodology for zero GPU fallback DLA INT8 deployment of classification backbones, comprising architecture adaptation, manual dynamic range workaround to rescue TensorRT's implicit quantiz
    
[^286]: CHORUS：面向测试平台激励生成的覆盖率引导智能体强化学习与策略整合

    CHORUS: Coverage-Guided Agentic RL and Policy Consolidation for Testbench Stimulus Generation

    [https://arxiv.org/abs/2608.10090](https://arxiv.org/abs/2608.10090)

    CHORUS是首个将覆盖率引导的智能体强化学习与策略整合相结合的测试平台激励生成框架，通过权重平均与奖励引导蒸馏，其4B模型在CVDP-ECov上达到88.0%的智能体Pass@1，超越DeepSeek-R1（671B）13.5个百分点。

    

    大语言模型编写的测试平台可能能够编译并运行，但整个数据通路却处于空闲状态。我们提出了CHORUS，这是首个将覆盖率引导的智能体强化学习（RL）与策略整合相结合用于测试平台激励生成的框架。我们训练LLM4Cov的最终监督检查点来起草测试平台，并利用仿真器反馈修复其得分最低的草稿，将CVDP-ECov上的智能体Pass@1从69.2%提升至85.5%。将同样的训练应用于三个检查点后，整体通过率相近，但各策略在不同设计上各有成功。对它们的权重进行平均，可将这些优势整合到一个稠密的4B模型中，达到86.7%的智能体Pass@1。当仿真器奖励为每个训练提示选择教师模型时，蒸馏得到的4B模型达到88.0%，比DeepSeek-R1（671B）高出13.5个百分点。

    arXiv:2608.10090v2 Announce Type: replace  Abstract: Testbenches written by large language models can compile and run while leaving entire datapaths idle. We present CHORUS, the first framework to combine coverage-guided agentic reinforcement learning (RL) with policy consolidation for testbench stimulus generation. We train LLM4Cov's final supervised checkpoint to draft testbenches and repair its lowest-scoring draft using simulator feedback, raising CVDP-ECov agentic Pass@1 from 69.2% to 85.5%. Applying the same training to three checkpoints yields similar overall pass rates, yet the policies succeed on different designs. Averaging their weights combines these strengths in a single dense 4B model that reaches 86.7% agentic Pass@1. When simulator rewards select the teacher for each training prompt, the distilled 4B model reaches 88.0%, 13.5 percentage points above DeepSeek-R1 (671B).
    
[^287]: 全模态分解自编码器学习全栈式可穿戴解耦表征

    Omni-modal decomposition autoencoders learn full-stack wearable disentangled representations

    [https://arxiv.org/abs/2608.07385](https://arxiv.org/abs/2608.07385)

    提出全模态变分分解自编码器（OmniDecVAEs），利用单一共享编码器-解码器和多视角自监督分解损失，从任意多模态可穿戴时间序列中学习解耦表征，在人体活动识别数据集上实现最高 3.0 个百分点的 DCI 解耦度提升。

    

    学习解耦表征是通用多模态可穿戴模型的关键，然而现有方法很少能在单一模型中同时实现异构多模态时间序列的分类、解耦、融合与生成。我们提出了全模态变分分解自编码器（OmniDecVAEs），这是一个通过单一共享编码器-解码器从任意数量模态中学习多用途表征的框架。OmniDecVAEs 在 DecVAEs 的基础上进行扩展，通过多视角自监督分解损失和逐分量条件解码器，为每个模态学习时频潜在子空间。在全模态人体活动识别（HAR）HARWE 数据集（最多三十个通道）以及公开的 UCI HAR 和 WESAD 数据集上，OmniDecVAEs 为未见过的受试者学习到了最解耦的表征（DCI 解耦指标最高提升 3.0 个百分点）。在 HARWE 数据集上，它们改善了受试者相关的（摘要在此处截断）

    arXiv:2608.07385v2 Announce Type: replace-cross  Abstract: Learning disentangled representations is key for general-purpose multi-modal wearable models, yet existing approaches rarely combine classification, disentanglement, fusion, and generation of heterogeneous multi-modal time series in a single model. We introduce Omni-modal Variational Decomposition Autoencoders (OmniDecVAEs), a framework that learns multi-purpose representations from arbitrarily many modalities with a single shared encoder-decoder. OmniDecVAEs extend DecVAEs by learning time-frequency latent subspaces per modality through a multi-view self-supervised decomposition loss and a component-wise conditional decoder. On the omni-modal human activity recognition (HAR) HARWE dataset (up to thirty channels) and the public UCI HAR and WESAD datasets, OmniDecVAEs learn the most disentangled representations for unseen subjects (DCI disentanglement up to 3.0 percentage points (pp) higher). On HARWE, they improve subject-depen
    
[^288]: Déjà Cue：通过词汇相对坐标在物体追踪历史中定位状态

    D\'ej\`a Cue: Localizing States in Object Histories via Vocabulary-Relative Coordinates

    [https://arxiv.org/abs/2608.02044](https://arxiv.org/abs/2608.02044)

    该论文提出无需训练的 Déjà Cue 框架，通过将状态描述嵌入以状态平衡的词汇均值为中心化来构造词汇相对坐标，使每个状态与其备选项进行对比，从而在物体追踪历史中精确定位状态发生的时刻，并构建了包含620条物体历史、2,465个状态标注的 MS-VOST 基准数据集。

    

    物体追踪能够在视觉变化中保持目标的身份一致性，但无法判断物体何时是完整的、被切开的或被削皮的。我们研究“状态时刻检索”问题：定位被追踪物体展现出给定描述集合中每个状态的具体时刻。然而，绝对的图文相似度往往偏向于共享的物体语义，从而掩盖了定位单个状态所需的区分性信息。我们的核心洞察是：备选状态本身可以作为衡量状态证据的参照基准。我们提出 Déjà Cue，一个无需训练的框架，它将描述嵌入以状态平衡的词汇均值为中心进行标准化，从而生成将每个状态与其备选项进行对比的查询方向。借助冻结的视觉-语言编码器，Déjà Cue 对帧级分数进行校准，并在连续可见片段内检索多个时长的状态区间。我们通过为620条物体历史标注2,465个状态构建了 MS-VOST 数据集。利用该编码器与时间检索（原文摘要在此处截断）

    arXiv:2608.02044v2 Announce Type: replace-cross  Abstract: Object tracking preserves identity through visual changes, but cannot tell when an object is whole, cut, or peeled. We study state-moment retrieval: locating when a tracked object exhibits each state in a supplied set of descriptions. However, absolute image--text similarity can favor shared object semantics, obscuring the distinctions needed to localize individual states. Our insight is that the alternatives themselves provide a reference for measuring state evidence. We introduce D\'ej\`a Cue, a training-free framework that centers description embeddings on a state-balanced vocabulary mean, producing query directions that compare each state against its alternatives. Using frozen vision--language encoders, D\'ej\`a Cue calibrates frame scores and retrieves intervals at multiple durations within contiguous visible runs. We construct MS-VOST by annotating 620 object histories with 2,465 states. With the encoder and temporal sear
    
[^289]: 从田间尺度到大规模光谱库：表格基础模型在土壤光谱学中的应用

    From field-scale to large-scale spectral libraries: Tabular foundation models in soil spectroscopy

    [https://arxiv.org/abs/2608.00608](https://arxiv.org/abs/2608.00608)

    该研究系统评估了TabPFN表格基础模型在85个土壤光谱特性预测任务中的表现，发现其始终优于CNN、Cubist、随机森林和PLSR等传统方法，为从田间尺度到全球尺度的土壤光谱建模提供了更优的新方案。

    

    可见光-近红外（vis-NIR）和中红外（MIR）光谱技术能够实现土壤特性的快速、低成本预测。然而，如何将高维、高度共线的光谱转化为准确的土壤特性预测仍然具有挑战性，尤其是在应用机器学习方法时。我们系统地研究了用于光谱建模的回归模型和降维方法，涵盖了来自土壤计量学开放基准数据集的85个土壤特性预测任务，涉及田间尺度的数字土壤制图和全球土壤光谱库。我们比较了基于上下文学习的表格基础模型（TabPFN）、卷积神经网络（CNN）、基于规则的回归（Cubist）、随机森林和偏最小二乘回归（PLSR），这些方法分别使用全光谱以及由主成分分析（PCA）和偏最小二乘（PLS）潜在变量导出的特征。TabPFN始终表现出……

    arXiv:2608.00608v2 Announce Type: replace  Abstract: Visible and near-infrared (vis-NIR) and mid-infrared (MIR) spectroscopy enable rapid, cost-effective prediction of soil properties. Yet, translating high-dimensional, highly collinear spectra into accurate soil property predictions remains challenging, particularly when employing machine learning. We systematically investigated regression models and dimensionality reduction approaches for spectroscopic modeling across 85 soil property prediction tasks from open benchmark datasets in pedometrics, spanning field-scale digital soil mapping and a global soil spectral library. We compared an in-context learning tabular foundation model (TabPFN), a convolutional neural network (CNN), rule-based regression (Cubist), Random Forest, and partial least squares regression (PLSR) using full spectra as well as features derived from principal component analysis (PCA) and partial least squares (PLS) latent variables. TabPFN consistently delivered th
    
[^290]: 持久稀疏自编码器：在语言模型表示中学习特征特定的时间尺度

    Persistent Sparse Autoencoders: Learning Feature-Specific Timescales in Language Model Representations

    [https://arxiv.org/abs/2607.17117](https://arxiv.org/abs/2607.17117)

    该论文提出持久稀疏自编码器，通过为每个特征学习一个持久性系数，使稀疏自编码器能够仅凭重构目标从语言模型激活中自动学习特征特定的时间尺度，同时保持高质量的重构效果。

    

    稀疏自编码器（SAE）将语言模型的激活分解为稀疏特征，然而这些模型传统上对每个词元进行独立编码，无法揭示跨序列持续存在的信息。我们首先证明，时间持久性可以在标准SAE特征中自然涌现：当一个特征激活后，隐藏状态会保持与其方向对齐，且过去的激活有助于重构后续的隐藏状态。这种持续性在不用特征之间的差异很大。因此，我们提出了持久稀疏自编码器（Persistent SAEs），这是标准SAE的一种扩展，它为每个特征学习一个持久性系数，使模型能够仅通过重构任务学习特征特定的时间尺度。我们的实验表明，持久SAE在学习一系列时间尺度的同时保持了有竞争力的重构质量：短时间尺度（快速）特征保持局部可解释性，而长时间尺度（……（摘要在此处被截断）

    arXiv:2607.17117v2 Announce Type: replace-cross  Abstract: Sparse autoencoders (SAEs) decompose language model activations into sparse features, yet these models traditionally encode each token independently, failing to expose information that persists across a sequence. We first show that temporal persistence can naturally emerge in standard SAE features: after a feature activates, the hidden state remains aligned with its direction, and past activations help reconstruct later hidden states. How long this lasts varies widely across features. We therefore introduce Persistent Sparse Autoencoders (Persistent SAEs), an extension of standard SAEs that learns a persistence coefficient for each feature, allowing the model to learn feature-specific timescales from reconstruction alone. Our experiments show that Persistent SAEs retain competitive reconstruction quality while learning a spectrum of timescales: short-timescale (fast) features stay locally interpretable, whereas long-timescale (
    
[^291]: 时序基准分数到底衡量了什么？分解视频视觉语言模型评估中的通道使用

    What Does a Temporal Benchmark Score Measure? Decomposing Channel Use in Video VLM Evaluation

    [https://arxiv.org/abs/2607.12304](https://arxiv.org/abs/2607.12304)

    该论文提出“反转冲突”评测方法（帧序反转而RoPE保持正向），将时序视频VLM基准分数分解为“任务是否真正需要时序信息”与“模型是从像素还是位置编码获取时序”两个通道，从而区分真正理解时序的模型与仅仅被干扰的模型。

    

    时序视频问答基准上的分数本应衡量模型是否具备时序理解能力，但它混淆了两个问题：（1）该问题是否真的是时序性的，是否需要多帧及其先后顺序？（2）如果是，模型是从像素中恢复出顺序，还是直接从位置编码（RoPE）中读取顺序？时序分数的大部分内容对这两个问题都无法回答，因为单帧和答案先验往往就足以支撑分数，而该领域现有的有效性检验手段——帧打乱敏感性和多帧增益——只涉及问题（1）。针对问题（2），我们用两种方式对“反转冲突”（即帧序被反转而RoPE仍保持正向）进行评分：一种是无标签方式，即以相对正向标签的准确率损失来衡量；另一种是在标签在时间反转下可确定性变换时，以相对反转标签的准确率来衡量。这能够将真正跟随反转内容的模型与仅仅被干扰的模型区分开来。这两种解读在两种模……（原文截断）

    arXiv:2607.12304v2 Announce Type: replace-cross  Abstract: A score on a temporal video question answering benchmark is meant to measure that a model has temporal understanding, but it conflates two questions. (1)~is the question even temporal, does it need several frames and their order? And (2)~when it does, does the model recover the order from the pixels, or read it off the positional encoding (RoPE)? Most of a temporal score answers neither, since a single frame and answer priors often carry it, and the field's validity checks, frame-shuffle sensitivity and multi-frame gain, speak only to (1). For (2) we score a reversal conflict, frames reversed while RoPE stays forward, two ways: label-free, as the accuracy lost against the forward labels, and against reversed labels wherever labels transform deterministically under time reversal. This separates a model that follows the reversed content from one merely disrupted. The two readings dissociate two similarly scoring models at both mo
    
[^292]: 从状态方程到引力对偶 II：大能级层级与伪真空

    Gravitational Duals from Equations of State II: Large Hierarchies and False Vacua

    [https://arxiv.org/abs/2606.30117](https://arxiv.org/abs/2606.30117)

    该论文将基于物理信息神经网络（PINNs）的全息逆问题方法扩展至具有大能级层级和伪真空的强耦合量子场论情形，通过一系列方法学改进克服了近简并态与势能未探测区域等挑战，实现了从边界热力学数据重构体标量势。

    

    我们研究了在以大层级和伪真空存在为特征的区域内，强耦合量子场论的全息对偶重构问题。在规范/引力对偶框架下，这些特征转化为非平凡的热力学行为和新奇的重整化群流，包括在非相邻不动点之间发生的“跳跃流”。在以往基于物理信息神经网络（PINNs）工作的基础上，我们将“从边界热力学数据重构体标量势”这一全息逆问题扩展到了这一新的区域。这一设定带来了诸多概念和数值上的挑战，例如近简并态、能量尺度的大层级，以及势能中未被输入数据直接探测到的区域。我们开发了一系列方法学上的改进来克服这些障碍，从而改进了已有的基于PINNs的方法。

    arXiv:2606.30117v2 Announce Type: replace-cross  Abstract: We investigate the reconstruction of holographic duals for strongly coupled quantum field theories in regimes characterized by large hierarchies and the presence of false vacua. Within the gauge/gravity duality, these features translate into non-trivial thermodynamic behaviour and exotic renormalization group flows, including skipping flows between non-adjacent fixed points. Building on previous work based on Physics-Informed Neural Networks (PINNs), we extend the holographic inverse problem of reconstructing the bulk scalar potential from boundary thermodynamic data into this new regime. This setting presents a variety of conceptual and numerical challenges, such as near-degenerate states, large hierarchies of energy scales, and regions of the potential that are not directly probed by the input data. We develop a set of methodological advances that overcome these obstacles, thereby improving the established PINNs-based methodo
    
[^293]: 必要但不充分：大语言模型作为评判者的安全评估中的温度控制与可重复性

    Necessary but Not Sufficient: Temperature Control and Reproducibility in LLM-as-Judge Safety Evaluations

    [https://arxiv.org/abs/2606.26185](https://arxiv.org/abs/2606.26185)

    本研究通过实验证明，即使将大语言模型作为评判者的采样温度设为0，也无法完全消除安全评估结果的不确定性，尤其是在决策边界附近的项目中仍存在不可重复性。

    

    arXiv:2606.26185v1 公告类型：新论文 摘要：大语言模型作为评判者（“评分器”）组件现已成为评估框架中的标准配置，包括在安全评估中，通过/不通过的判定结果可能决定下游部署决策。一个普遍假设是将评分器的采样温度设置为0可使评分结果确定。我们针对真实的安全评估代码库（日本AISI的开源项目aisev）测试了这一假设，并在两个层面上证明其不成立。首先，评估框架在调用评分器时未设置温度或随机种子；底层提供商默认应用1.0的温度值，因此接近决策边界的项目在相同运行中会出现通过/不通过翻转（在20次运行中，单个项目的不一致率高达约50%）。其次，将温度固定为0可以减少但无法消除翻转：在跨越两个提供商、三个模型层级和五种采样配置的690次API调用中，即使采用强制贪婪解码（top_k=1），7个边界项目中的1-2个仍然不可重复。Claude Opus 4.7/4.

    arXiv:2606.26185v1 Announce Type: new  Abstract: LLM-as-judge ("grader") components are now standard in evaluation harnesses, including safety evaluations where a pass/fail verdict may gate downstream deployment decisions. A widespread assumption is that setting the grader's sampling temperature to 0 makes grading deterministic. We test this assumption against a real safety-evaluation codebase (Japan AISI's open-source aisev) and show it fails on two levels. First, the harness invokes its grader without setting temperature or seed; the underlying provider silently applies its default of 1.0, so items near the decision boundary flip pass/fail across identical runs (per-item disagreement up to ~50% over 20 runs). Second, pinning temperature=0 reduces but does not eliminate flips: across 690 API calls spanning two providers, three model tiers, and five sampling configurations, 1-2 of 7 borderline items remain non-reproducible even under forced greedy decoding (top_k=1). Claude Opus 4.7/4.
    
[^294]: 层神经网络的层次化池化

    Hierarchical Pooling for Sheaf Neural Networks

    [https://arxiv.org/abs/2606.20932](https://arxiv.org/abs/2606.20932)

    提出 HiSP 层次化池化层，通过节点分组、基于细尺度层结构缩减特征并构建粗化图，再在粗分辨率下学习新的层结构，使层神经网络能够实现层次化表示学习。

    

    层神经网络通过在胞腔层上运作来扩展图神经网络；胞腔层为每个节点和边分配各自的向量空间，并定义了从节点空间到与其相关联的边空间的线性限制映射。这些映射决定了层拉普拉斯算子，并且可以在每一层中学习，从而实现各层特定的扩散动力学。然而，现有的 SNN 架构始终在同一个图上进行扩散：其限制映射虽然在层与层之间可以变化，但始终作用于原始的节点和边。我们提出了层次化层池化，这是一种能够在 SNN 中实现层次化表示学习的池化层。HiSP 遵循“选择—缩减—连接—提升”框架，对细尺度的节点进行分组，利用细尺度的层结构缩减其特征，并构建相应的粗化图；随后在粗分辨率下学习一个新的层。通过交替进行层扩散……（原文摘要至此被截断）

    arXiv:2606.20932v2 Announce Type: replace  Abstract: Sheaf neural networks (SNNs) extend graph neural networks by operating on cellular sheaves, which assign to each node and edge its own vector space, along with linear restriction maps from node spaces to the spaces associated with their incident edges. These maps determine the sheaf Laplacian and can be learned at each layer, enabling layer-specific diffusion dynamics. Existing SNN architectures, however, always diffuse on the same graph, with restriction maps that may change across layers but continue to act on the original nodes and edges. We introduce Hierarchical Sheaf Pooling (HiSP), a pooling layer that enables hierarchical representation learning in SNNs. Following the Select-Reduce-Connect-Lift framework, HiSP groups fine-scale nodes, reduces their features using the fine-scale sheaf structure, and constructs the corresponding coarse graph. A new sheaf is then learned at the coarse resolution. By interleaving sheaf diffusion 
    
[^295]: 一种面向约束感知生物过程开发的人机协同贝叶斯优化框架

    A Human-in-the-Loop Bayesian Optimization Framework for Constraint-Aware Bioprocess Development

    [https://arxiv.org/abs/2606.19230](https://arxiv.org/abs/2606.19230)

    本文提出了一种扩展的人机协同贝叶斯优化框架，通过将约束满足概率和鲁棒性能作为帕累托目标，使专家能交互式选择最优候选，从而在生物过程开发中兼顾约束与不确定性。

    

    本文介绍了对帕累托前沿引导采样（PFGS）的扩展，这是一种人机协同（HitL）贝叶斯优化（BO）框架，其中高斯过程（GP）代理导出的量被重新表述为多目标优化问题的目标，所得帕累托前沿暴露给领域专家进行交互式候选选择，而非返回单一自动推荐。该框架在两个方向上进行了扩展：约束优化通过将满足输出规格限制的后验概率作为显式帕累托目标来处理，该概率从GP后验分布解析计算；鲁棒优化通过蒙特卡洛采样策略来处理，该策略估计在用户定义的输入扰动变异性下的期望下置信性能，捕捉在可能实现偏差下的性能退化。

    arXiv:2606.19230v2 Announce Type: replace-cross  Abstract: This work presents an extension to Pareto Front Guided Sampling (PFGS), a Human-in-the-Loop (HitL) Bayesian Optimization (BO) framework in which Gaussian process (GP) surrogate-derived quantities are reformulated as objectives of a multi-objective optimization problem, and the resulting Pareto front is exposed to a domain expert for interactive candidate selection rather than returning a single automated recommendation. The framework is extended in two directions: constrained optimization is addressed by incorporating the posterior probability of satisfying output specification limits as an explicit Pareto objective, computed analytically from the GP posterior distribution; robust optimization is addressed by a Monte Carlo sampling strategy that estimates expected lower-confidence performance over a user-defined variability of input perturbations, capturing performance degradation under likely implementation deviations. The res
    
[^296]: 深度何时对上下文学习重要？深度Transformer中的自适应推理

    When Does Depth Matter For In-Context Learning? Adaptive Inference in Deep Transformers

    [https://arxiv.org/abs/2606.16694](https://arxiv.org/abs/2606.16694)

    本文提出了广义线性注意力Transformer中分布式推理的一般理论，证明多层Transformer可利用“函数向量”内部表示在各层以越来越精细的尺度推断潜在上下文变量，从而阐明了深度堆叠在何种条件下比单层结构具有计算优势。

    

    Transformer通过多个连续的注意力和前馈块执行计算，使它们能够学习大量耦合变量之间的复杂相关性。那么，在多层上堆叠连续的注意力-前馈计算，何时能比单个Transformer块提供计算优势？我们通过研究广义线性注意力Transformer中的上下文学习来回答这个问题。我们首先引入了此类Transformer中分布式推理的一般理论，该理论受限于通信与深度的约束。我们证明，这样的系统可以利用内部表示（“函数向量”）在其各层上以越来越精细的尺度推断潜在上下文变量。对于一个上下文线性回归任务，该理论预测：虽然不含前馈块的单层Transformer对于上下文变量服从高斯先验的情形是最优的，但多层Transformer……

    arXiv:2606.16694v3 Announce Type: replace-cross  Abstract: Transformers perform computations through many successive attention and feedforward blocks, allowing them to learn complex correlations between a large collection of coupled variables. When does stacking successive attention-feedforward computations over many layers provide a computational advantage over a single transformer block? We address this question by examining in-context learning in generalized linear attention transformers. We first introduce a general theory of distributed inference in such transformers, subject to constraints on communication and depth. We show that such systems can exploit internal representations (`function vectors') to infer a latent context variable at increasingly finer scales over its layers. For an in-context linear regression task, the theory predicts that while one-layer transformers without feedforward blocks are optimal for Gaussian priors over the context variable, multi-layer transforme
    
[^297]: 有理稀疏自编码器

    Rational Sparse Autoencoder

    [https://arxiv.org/abs/2606.14990](https://arxiv.org/abs/2606.14990)

    提出有理稀疏自编码器（RSAE），用可训练的有理函数取代固定的编码器激活函数，在匹配期望稀疏度下从理论上保证不劣于 ReLU，并在给定假设下可证明优于 JumpReLU 和 TopK。

    

    稀疏自编码器（SAE）是机制可解释性研究的标准工具，但当前的 SAE 家族受到固定编码器非线性激活函数（如 ReLU、JumpReLU 和 TopK）的约束。这种做法将特定的稀疏机制硬编码进模型，并可能扭曲重构与稀疏性之间的权衡。我们提出了有理稀疏自编码器（RSAE），它用一个可训练的有理函数取代固定的编码器激活函数。该设计的动机源于我们在匹配期望稀疏度下为 RSAE 证明的重构优势：保证不劣于 ReLU，并且在给定假设下，可证明优于 JumpReLU 和 TopK。该模型通过两阶段流水线实现：首先是一个初始化过程，复制预训练的基线 SAE 权重，并对尺度参数和有理系数进行校准；随后在稀疏正则化的（微调步骤中进行优化，原文摘要在此处截断）。

    arXiv:2606.14990v3 Announce Type: replace-cross  Abstract: Sparse autoencoders (SAEs) are standard tools for mechanistic interpretability, but current SAE families are constrained by fixed encoder nonlinearities such as ReLU, JumpReLU, and TopK. This hard-codes a particular sparsity mechanism into the model and can distort the reconstruction-versus-sparsity trade-off. We introduce the Rational Sparse Autoencoder (RSAE), which replaces the fixed encoder activation with a trainable rational function. The design is motivated by a reconstruction advantage that we prove for RSAE at matched expected sparsity, with guaranteed non-inferiority to ReLU and, under stated assumptions, provable improvements over JumpReLU and TopK. The model is then implemented through a two-stage pipeline: an initialization procedure that copies the pre-trained baseline SAE weights and calibrates the scale parameters along with the rational coefficients; followed by a fine-tuning step under a sparsity-regularized r
    
[^298]: 个体行为的隐式神经表示

    Implicit Neural Representations of Individual Behavior

    [https://arxiv.org/abs/2606.12200](https://arxiv.org/abs/2606.12200)

    提出Behavioral INR，一种自监督生成模型，将视觉领域的隐式神经表示适配到行为领域，把策略表示为由片段级潜在变量通过FiLM层调制的状态-动作函数，从而在无标签的混合行为数据中实现无监督的策略表示学习与策略身份推断。

    

    我们研究从无标签的多策略行为数据中进行策略表示学习。每个轨迹片段由一个固定策略生成，但策略标签不可用。这种场景出现在机器人自由玩耍、演示、游戏、赛车以及其他异构行为混合且无标注的数据集中。我们提出了Behavioral INR，一种将视觉领域中的隐式神经表示（INR）适配到行为领域的自监督生成模型。Behavioral INR不是将坐标映射到RGB值，而是将策略表示为一个状态-动作函数，即把状态映射到后续动作。一个片段级的潜在变量通过FiLM层对该函数进行调制，从而产生策略之上的生成先验，并允许在无监督条件下推断策略身份。由于INR将每个数据点视为来自某个底层函数的样本，同一模型能够自然地适应不同的片段长度和不同的采样粒度……

    arXiv:2606.12200v2 Announce Type: replace-cross  Abstract: We study policy representation learning from unlabeled multi-policy behavioral data. Each episode is generated by a fixed policy, but policy labels are unavailable. This setting appears in robotics play, demonstrations, games, racing, and other datasets where heterogeneous behaviors are mixed without annotations. We introduce Behavioral INR, a self-supervised generative model that adapts implicit neural representations (INRs) from vision to behavior. Instead of mapping coordinates to RGB values, Behavioral INR represents a policy as a state-action function mapping states to subsequent actions. An episode-level latent modulates this function through FiLM layers, yielding a generative prior over policies and allowing policy identity to be inferred without supervision. Because INRs treat each datapoint as samples from an underlying function, the same model naturally accommodates variable episode lengths and different sampling gran
    
[^299]: RePAIR：国际象棋中的预测式自监督表征学习

    RePAIR: Predictive Self-Supervised Representation Learning in Chess

    [https://arxiv.org/abs/2606.11860](https://arxiv.org/abs/2606.11860)

    本文提出RePAIR架构，融合MAE、JEPA与BERT的思想，通过掩蔽潜在状态序列并用轻量级预测器在低维嵌入空间中修复空缺，实现了对国际象棋局面序列的自监督表征学习，使有意义的棋类概念在潜在空间中自然聚类。

    

    在本文中，我们提出了通过迭代精炼实现自编码表征预测（RePAIR）——一种新颖的自监督表征学习架构，它融合了掩码自编码器（MAE）、联合嵌入预测架构（JEPA）以及基于Transformer的双向编码器表征（BERT）。我们展示了如何利用该架构将序列数据中的对象（如连续的国际象棋局面）编码为紧凑而有意义的表征。该架构的基本原理类似于BERT和MAE：掩蔽潜在状态序列中的大部分内容，然后应用一个轻量级的预测器（Predictor）在低维嵌入空间中修复序列中的空缺，这与JEPA的思想类似。我们在国际象棋领域的实验表明，编码器（Encoder）能够不断精炼棋盘表征，使得有意义的国际象棋概念在潜在空间中形成聚类。此外，重建……

    arXiv:2606.11860v2 Announce Type: replace  Abstract: In this paper, we introduce Representation Prediction via Autoencoding using Iterative Refinement (RePAIR) - a novel self-supervised representation learning architecture that synthesizes Masked Autoencoders (MAE), Joint Embedding Predictive Architectures (JEPA), and Bidirectional Encoder Representations from Transformers (BERT). We demonstrate how it can be used to encode objects in sequential data like consecutive chess positions into compact yet meaningful representations. The basic principle of the architecture is to mask large portions of a sequence of latent states, similar to BERT and MAE. Then, we apply a lightweight Predictor to the latent representations that repairs gaps in the sequence in a lower-dimensional embedding space akin to JEPA. Our experiments in the domain of chess show that the Encoder refines the board representations such that meaningful chess concepts emerge clustered in the latent space. Furthermore, recons
    
[^300]: SPACE：面向多模态大语言模型的无源代理锚点概念擦除方法

    SPACE: Source-free Proxy Anchor Concept Erasure for MLLMs

    [https://arxiv.org/abs/2606.09868](https://arxiv.org/abs/2606.09868)

    SPACE 提出了首个面向多模态大语言模型的无源机器遗忘框架，通过文本引导的代理锚点选择和双约束语义隔离两个阶段，在无需访问目标概念原始数据的情况下实现敏感概念的擦除。

    

    随着多模态大语言模型（MLLMs）面临日益增长的隐私风险和监管限制，机器遗忘已成为在保持模型性能的同时删除敏感数据的关键解决方案。然而，现有的机器遗忘方法通常依赖于目标概念的视觉数据，而由于严格的数据保留政策，这些数据往往无法获取，因此催生了对无需访问目标数据的无源遗忘方法的需求。在本工作中，我们提出了无源代理锚点概念擦除，这是首个专门面向多模态大语言模型的无源遗忘框架。SPACE 包含两个阶段：（1）文本引导的代理锚点选择（TPAS），从共享特征空间中检索语义对齐的代理锚点；（2）双约束语义隔离（DCSI），通过优化这些锚点来间接擦除目标概念。DCSI 将更新约束在保留……（原文摘要在此处截断）

    arXiv:2606.09868v2 Announce Type: replace-cross  Abstract: As Multimodal Large Language Models (MLLMs) face growing privacy risks and regulatory constraints, machine unlearning (MU) has emerged as a crucial solution for removing sensitive data while preserving model performance. However, existing MU methods typically rely on visual data of the target concepts, which is often unavailable due to strict data retention policies, thus creating a demand for source-free unlearning approaches that operate without access to the target data. In this work, we propose Source-free Proxy Anchor Concept Erasure (SPACE), the first source-free unlearning framework specialized for MLLMs. SPACE consists of two stages: (1) Text-Guided Proxy Anchor Selection (TPAS), which retrieves semantically aligned proxy anchors from the shared feature space. (2) Dual-Constraint Semantic Isolation (DCSI), which optimizes these anchors to indirectly erase target concepts. DCSI confines updates to the null space of retai
    
[^301]: 自动化“专家之眼”：面向不平衡力谱学中稀有事件发现的系统无关深度学习框架

    Automating the Expert Eye: A System-Agnostic Deep Learning Framework for Rare Event Discovery in Imbalanced Force Spectroscopy

    [https://arxiv.org/abs/2606.09541](https://arxiv.org/abs/2606.09541)

    该研究提出了一个基于改进ResNet18与焦点损失、无需系统特异性物理先验的深度学习框架，能够自动从严重不平衡的单分子力谱数据中筛查稀有解离事件，在阳性率仅1.34%的测试集上达到92.31%召回率和91.95%特异性。

    

    单分子力谱（SMFS）能够在单分子水平上探测生物分子的力学特性，但力-延伸曲线的高通量采集带来了数据处理瓶颈。在数千条以噪声为主的轨迹中识别稀有的特异性解离事件，目前仍主要依赖人工目视检查。本研究提出了一种用于严重不平衡SMFS数据集初步筛选的深度学习工作流程。通过使用力-延伸曲线的二维图像表示，且不依赖系统特异性的物理先验，我们对采用焦点损失（focal loss）训练的改进版ResNet18架构进行了微调，并在 Ruminococcus champanellensis 细菌的 XMod-Doc/Coh 复合物的力学解折叠路径上进行了评估。在目标阳性率仅为1.34%（970条曲线中仅有13个阳性P_2事件）的留出测试集上，该模型实现了92.31%的召回率（13个事件中检出12个）和91.95%的特异性（957条阴性轨迹中880条被正确分类），可自动……

    arXiv:2606.09541v2 Announce Type: replace-cross  Abstract: Single-molecule force spectroscopy (SMFS) probes biomolecular mechanics at the single-molecule level, but high-throughput acquisition of force-extension curves creates a data processing bottleneck. Identifying rare specific unbinding events among thousands of noise-dominated traces still relies largely on manual visual inspection. Here, we present a deep learning workflow for the preliminary screening of severely imbalanced SMFS datasets. Using 2D image representations of force-extension curves without system-specific physical priors, we fine-tuned a modified ResNet18 architecture trained with focal loss and evaluated it on the mechanical unfolding pathways of the Ruminococcus champanellensis XMod-Doc/Coh complex. On a held-out test set with a 1.34% target prevalence (13 positive P_2 events among 970 curves), the model achieved 92.31% recall (12 of 13 events) and 91.95% specificity (880 of 957 negative traces), automatically di
    
[^302]: 学习哪些证据寻求决策才是重要的：面向语言智能体的校准反事实重放方法

    Learning Which Evidence-Seeking Decisions Matter: Calibrated Counterfactual Replay for Language Agents

    [https://arxiv.org/abs/2606.05263](https://arxiv.org/abs/2606.05263)

    提出了PCCC-RL方法，通过前缀条件下的反事实重放、预先信用预测与基于选择概率的校准纠正机制，精准识别语言智能体中哪些证据寻求决策真正决定了最终答案的质量，从而解决稀疏奖励下的信用分配问题。

    

    语言智能体通过搜索、阅读和核查声明来回答问题。仅凭最终答案的奖励很难说明是哪些决策为其赢得了正确结果，因为智能体可能在冗余搜索之后、或在缺少一半证据的情况下依然得到正确的答案字符串。我们将这个问题视为一个实验。在保存的决策点上，PCCC-RL（前缀条件反事实信用分配方法）会恢复工具状态，同时运行智能体所选择的动作和一个可行的替代动作，并让同一个冻结策略完成这两个分支。最终得分的差异便抽样出该决策的真实价值。由于重放所有内容代价过高，该方法预先预测每个动作的信用值，随机抽取一个子集进行验证，并利用记录的选择概率来纠正预测误差，将预算花费在进一步的检查仍能改变训练信号的地方。替代动作仅从前缀中抽取，这保持了……

    arXiv:2606.05263v2 Announce Type: replace-cross  Abstract: Language agents answer questions by searching, reading, and checking claims. A reward on the final answer says little about which of those decisions earned it, since an agent can reach the right string after redundant searches or with half the evidence missing. We treat the question as an experiment. At a saved decision point, PCCC-RL, for prefix-conditioned counterfactual credit, restores the tool state, runs the action the agent chose alongside a feasible alternative, and lets the same frozen policy finish both branches. The difference in final score then samples what the decision was worth. Replaying everything is too expensive, so the method predicts each action's credit up front, checks a random subset, and uses the recorded selection probabilities to correct the errors of that prediction, spending the budget where a further check can still move the training signal. Alternatives are drawn from the prefix alone, which keeps
    
[^303]: 支持性令牌揭示：加速扩散语言模型解码

    Supportive Token Revealing for Fast Diffusion Language Model Decoding

    [https://arxiv.org/abs/2606.04236](https://arxiv.org/abs/2606.04236)

    提出无需训练的AXON模块，通过监控剩余的不确定掩码令牌并在需要额外上下文时揭示支持性令牌，从而加速扩散语言模型的并行解码并缓解质量与延迟之间的权衡。

    

    离散扩散语言模型可以通过并行更新多个被掩码的位置来高效生成文本，但这种并行性引入了质量与延迟之间的权衡。激进的解码可能过早地确定相互依赖的令牌，而保守的解码则需要许多去噪步骤。现有方法通过基于置信度或依赖性准则判断哪些令牌可以安全揭示来缓解这一矛盾。然而，避免不安全的提交并不一定能使剩余的掩码序列易于解码，因为不确定的令牌可能依赖于其他被掩码的令牌，这成为去噪步骤的瓶颈。我们提出了AXON，这是一个无需训练的模块，可以叠加在现有的扩散语言模型并行解码策略之上。AXON并不替换基础解码器，而是监控剩余的不确定掩码令牌，仅在其当前状态表明需要额外上下文时才进行干预。

    arXiv:2606.04236v2 Announce Type: replace-cross  Abstract: Discrete diffusion language models can generate text efficiently by updating multiple masked positions in parallel, but this parallelism introduces a quality-latency trade-off. Aggressive decoding may commit mutually dependent tokens too early, while conservative decoding requires many denoising steps. Existing methods address this tension by deciding which tokens are safe to reveal using confidence or dependency criteria. However, avoiding unsafe commits does not necessarily make the remaining masked sequence easy to decode, since uncertain tokens may depend on masked tokens, creating a bottleneck for denoising steps. We propose AXON, a training-free module that can be added on top of existing parallel decoding strategies for diffusion language models. Rather than replacing the base decoder, AXON monitors the remaining uncertain masked tokens and intervenes only when their current state suggests that additional context is need
    
[^304]: 资源受限动态定价的自适应推断

    Adaptive Inference for Resource-Constrained Dynamic Pricing

    [https://arxiv.org/abs/2606.03736](https://arxiv.org/abs/2606.03736)

    本文提出一种推断感知的重新求解策略，在资源受限动态定价中平衡收益与需求推断，实现接近最优的遗憾和信息效率。

    

    arXiv:2606.03736v2 公告类型：替换-交叉 摘要：我们研究了当卖方在销售季前设定一个固定价格并寻求收益及对该价格需求的有效推断时的资源受限动态定价问题。库存耗尽可能移除目标价格附近的每个可行价格，因此对剩余价格进行随机化不一定能保持识别性。我们提出了一种推断感知的重新求解策略，该策略在观察当前协变量之前检查目标支持，并使用记录的价格混合实现流体目标负荷。在仿射约束容量族中，目标质量$t^{-\gamma}$产生信息量$T^{1-\gamma}$，区间半径$T^{-(1-\gamma)/2}$，以及对初始流体最优值的遗憾$O(\log T+T^{1-\gamma})$。在同一仿射族中，学习到的重心重新求解保留了一个恒定质量的局部目标组件，并且通过指数大于1的多项式误差支出，实现了线性信息时钟和$O(\log T)$遗憾；松弛容量局部定价

    arXiv:2606.03736v2 Announce Type: replace-cross  Abstract: We study resource-constrained dynamic pricing when the seller seeks revenue and valid inference about demand at a price fixed before the selling season. Depletion can remove every feasible price near the target, so randomization over the remaining prices need not preserve identification. We propose an inference-aware re-solving policy that checks target support before observing the current covariates and implements the fluid target load with a logged pricing mixture. In an affine binding-capacity family, target mass $t^{-\gamma}$ yields information $T^{1-\gamma}$, interval radius $T^{-(1-\gamma)/2}$, and regret $O(\log T+T^{1-\gamma})$ against the initial fluid optimum. In the same affine family, learned barycentric re-solving retains a target-local component of constant mass and, with polynomial error spending of exponent greater than one, achieves a linear information clock and $O(\log T)$ regret; slack-capacity local pricing
    
[^305]: VESTA：基于统计工具智能体的可视化探索

    VESTA: Visual Exploration with Statistical Tool Agents

    [https://arxiv.org/abs/2606.00384](https://arxiv.org/abs/2606.00384)

    VESTA是一个为视觉语言模型配备动态增长工具包的新框架，通过数据变换、假设驱动的可视化和统计检验来探索数据并改进统计建模，克服了以往仅依赖迭代批评的智能体系统在复杂建模任务上的局限。

    

    将定量模型拟合到数据是科学工作流程中的核心步骤，但这一步骤仍然是自动化程度最低的环节之一。近年来，基于智能体的系统利用语言模型和视觉语言模型（VLM）来提出统计模型，但这些系统在具有挑战性的建模任务上表现不佳。为了解决这些局限性，我们提出了VESTA（Visual Exploration with Statistical Tool Agents，基于统计工具智能体的可视化探索），这是一个为VLM配备动态增长工具包的框架，通过数据变换、假设驱动的可视化以及统计检验来引导模型的改进。与以往仅依赖迭代式批评反馈的系统不同，VESTA通过选择或创建可复用的诊断工具，在模型改进之前和改进过程中对数据进行探索。我们使用三种工具包配置对VESTA进行评估：无工具、静态的专家编写工具，以及动态的模型自建工具。为支持这一评估，我们引入了DAWN（用于自动化工作流程和数值……的数据集）——摘要在此处截断。

    arXiv:2606.00384v3 Announce Type: replace  Abstract: Fitting quantitative models to data is a central step in scientific workflows, yet it remains one of the least automated. Recent agent-based systems leverage language and vision-language models (VLMs) to propose statistical models, but these systems struggle on challenging modeling tasks. To address these limitations, we introduce VESTA (Visual Exploration with Statistical Tool Agents), a framework that equips VLMs with a dynamically growing toolkit to guide model refinement through data transformations, hypothesis-driven visualizations, and statistical tests. Unlike prior systems that rely on iterative critique alone, VESTA explores data before and during refinement by selecting or creating reusable diagnostic tools. We evaluate VESTA using three toolkit configurations: no tools, static expert-written tools, and dynamic model-written tools. To support this evaluation, we introduce DAWN (Dataset for Automated Workflows and Numerical 
    
[^306]: PhoneWorld：扩展手机使用代理环境

    PhoneWorld: Scaling Phone-Use Agent Environments

    [https://arxiv.org/abs/2605.29486](https://arxiv.org/abs/2605.29486)

    PhoneWorld通过真实GUI轨迹自动构建可扩展的手机使用环境，解决了移动代理训练与评估中环境构建的瓶颈。

    

    arXiv:2605.29486v2 公告类型：替换交叉  摘要：手机使用代理的一个核心瓶颈是，难以大规模构建覆盖真实移动行为的可控、可复现环境。现有的移动代理基准在评估方面取得了重要进展，但本身并未提供一种可扩展的方式来构建许多新的手机使用环境。我们提出了PhoneWorld，一个可复用的流水线，将真实的GUI轨迹和截图转换为可控的手机使用环境、可执行任务、自动验证器和训练回放。PhoneWorld不是一次手动构建一个移动基准，而是利用真实轨迹来恢复哪些屏幕重要、屏幕如何连接、哪些交互必须改变环境状态，以及哪些用户目标允许自动验证。从这些信号中，它构建了由只读应用内容和可变状态支持的可运行模拟Android应用，然后推导出可执行任务、基于规则的验证器和训练数据。

    arXiv:2605.29486v2 Announce Type: replace-cross  Abstract: A central bottleneck for phone-use agents is that controllable, reproducible environments covering real mobile behavior are hard to build at scale. Existing mobile-agent benchmarks have made important progress on evaluation, but they do not by themselves provide a scalable way to construct many new phone-use environments. We present PhoneWorld, a reusable pipeline that converts real GUI trajectories and screenshots into controllable phone-use environments, executable tasks, automatic verifiers, and training rollouts. Rather than hand-building one mobile benchmark at a time, PhoneWorld uses real trajectories to recover which screens matter, how screens connect, which interactions must change environment state, and which user goals admit automatic verification. From these signals, it builds runnable mock Android apps backed by read-only app content and mutable state, then derives executable tasks, rule-based verifiers, and traini
    
[^307]: 洞察生成器：面向大语言模型智能体的系统性语料库级轨迹诊断

    Insights Generator: Systematic Corpus-Level Trace Diagnostics for LLM Agents

    [https://arxiv.org/abs/2605.21347](https://arxiv.org/abs/2605.21347)

    该论文提出了洞察生成器（IG）——一个多智能体系统，通过在执行轨迹语料库上自动提出并检验假设，生成有证据支持的系统性诊断洞察报告，解决了 LLM 智能体失败诊断依赖人工、无法规模化的问题。

    

    arXiv:2605.21347v4 公告类型：replace-cross 摘要：诊断大语言模型（LLM）智能体的失败在很大程度上仍然是手工完成的。从业者通常只检查一小部分执行轨迹，形成临时性的假设，然后不断迭代。这一过程会遗漏那些只有在轨迹群体层面才会显现的模式，也无法扩展到单条轨迹就包含数万 token 的生产级语料库。我们形式化了语料库级轨迹诊断问题：给定一个执行轨迹语料库，目标是在轨迹群体上刻画系统性的行为模式，生成有据可依的自然语言洞察，并且每条洞察都关联相应的支持性证据。我们提出了洞察生成器（Insights Generator, IG），这是一个多智能体系统，它通过在轨迹语料库上提出并检验假设来回答诊断问题，最终产出有证据支撑的洞察报告。我们从定性和客观两个维度评估了 IG，包括基于评分量表的报告评估，以及通过实施洞察所带来的下游性能提升……

    arXiv:2605.21347v4 Announce Type: replace-cross  Abstract: Diagnosing failures in LLM agents remains largely manual. Practitioners inspect a small subset of execution traces, form ad-hoc hypotheses, and iterate. This process misses patterns that only emerge across trace populations and does not scale to production corpora where individual traces span tens of thousands of tokens. We formalize the problem of corpus-level trace diagnostics. Given a corpus of execution traces, the goal is to produce grounded natural-language insights that characterize systematic behavioral patterns across trace groups, each linked to supporting evidence. We present the Insights Generator (IG), a multi-agent system that answers diagnostic questions by proposing and testing hypotheses across the trace corpus to produce an evidence-backed insights report. We evaluate IG across qualitative and objective dimensions, spanning rubric-based report assessment and downstream performance improvements achieved by impl
    
[^308]: AVSD：通过平衡共识信号与教师特定特权信号的自适应视图自蒸馏

    AVSD: Adaptive-View Self-Distillation by Balancing Consensus and Teacher-Specific Privileged Signals

    [https://arxiv.org/abs/2605.20643](https://arxiv.org/abs/2605.20643)

    本文提出 AVSD（自适应视图自蒸馏）方法，通过平衡共识信号与教师特定特权信号，在多种特权信息视图之间自适应地进行自蒸馏，从而联合解决了师生信息不对称和特权信息类型因任务而异这两个难题。

    

    自蒸馏使语言模型能够从自身的轨迹中进行在线学习，其方法是将同一个模型同时用作学生和教师，其中教师以学生无法获取的特权信息为条件。这类信息可以来自不同的类型或视图，例如解题方案、示范、反馈或最终答案。这种设置无需依赖单独的外部模型即可提供密集的 token 级反馈，但会产生一个根本性的不对称问题：教师可能依赖于学生在推理时无法访问的视图特定信息。此外，最佳类型的特权信息往往取决于具体任务，因此难以选择单一的教师视图。在本工作中，我们通过提出 AVSD（自适应视图自蒸馏）来联合解决这两个挑战，这是一种使用多种特权信息视图的新型自蒸馏方法，它重构了 token 级监督……（摘要内容被截断）

    arXiv:2605.20643v2 Announce Type: replace-cross  Abstract: Self-distillation enables language models to learn on-policy from their own trajectories by using the same model as both student and teacher, with the teacher being conditioned on privileged information unavailable to the student. Such information can come in different types or views, such as solutions, demonstrations, feedback, or final answers. This setup provides dense token-level feedback without relying on a separate external model, but creates a fundamental asymmetry: the teacher may rely on view-specific information that the student cannot access at inference time. Moreover, the best type of privileged information is often task-dependent, making it difficult to choose a single teacher view. In this work, we address both these challenges jointly by introducing AVSD (Adaptive-View Self-Distillation), a novel method of self-distillation with multiple privileged-information views, which reconstructs token-level supervision b
    
[^309]: TriAxialKV：面向智能体推理任务的极致低精度KV缓存量化

    TriAxialKV: Toward Extreme Low-Precision KV-Cache Quantization for Agentic Inference Tasks

    [https://arxiv.org/abs/2605.17170](https://arxiv.org/abs/2605.17170)

    提出TriAxialKV，一种沿时间接近度、模态和语义角色三个维度利用token异质性的混合精度KV缓存量化方案，可实现面向智能体推理任务的极致低精度压缩。

    

    智能体工作负载已成为大语言模型（LLM）推理的主要工作负载。它们与仅聊天的工作负载存在显著差异，需要长上下文处理、多模态输入处理能力，以及具备工具调用功能的结构化多轮交互。因此，其上下文呈现出可沿三个关键维度携带不同重要性的结构：与当前轮次的时间接近度、模态（如文本或图像token），以及语义角色（如用户查询、工具调用、观察结果或推理内容）。这些维度刻画了不同的token行为，并导致对KV缓存压缩的不同敏感度。然而，现有的KV缓存量化方法通常是同构的，或仅利用单一维度上的异质性（如时间接近度或模态），而忽略了这些维度之间的相互作用。为此，我们提出了TriAxialKV，一种新颖的混合精度KV缓存量化方案，它为……

    arXiv:2605.17170v2 Announce Type: replace  Abstract: Agentic workloads have emerged as a major workload for LLM inference. They differ significantly from chat-only workloads, requiring long-context processing, the ability to handle multimodal inputs, and structured multi-turn interactions with tool calling capabilities. As a result, their context exhibits structure that can carry different importance along three key axes: temporal recency to the current turn, modality such as text or image tokens, and semantic role such as user queries, tool calls, observations, or reasoning. These axes capture distinct token behaviors and lead to different sensitivities to KV-cache compression. However, existing KV-cache quantization methods are typically homogeneous or exploit only heterogeneity on a single dimension, such as temporal proximity or modality, overlooking the interactions among them. To this end, we introduce TriAxialKV, a novel mixed-precision KV-cache quantization scheme that assigns 
    
[^310]: 球形玻尔兹曼机：能量基模型中学习与生成的可解理论

    Spherical Boltzmann machines: a solvable theory of learning and generation in energy-based models

    [https://arxiv.org/abs/2605.09031](https://arxiv.org/abs/2605.09031)

    该论文借助随机矩阵理论与动力学平均场理论，建立了球形玻尔兹曼机在高维极限下的首个可解理论，推导出训练动力学的封闭方程、计算了作为参数空间配分函数的贝叶斯证据，并揭示了与耦合矩阵模态凝聚相关的级联相变及其对生成采样行为的影响。

    

    能量基模型（EBMs）是受统计物理启发的灵活生成式架构，但其学习与生成特性至今仍缺乏深入理解。本文分析了高维极限下一个可求解的能量基模型：球形玻尔兹曼机（SBM）。结合随机矩阵理论与动力学平均场理论的工具，我们：（1）推导并求解了描述高维极限下SBM训练动力学的封闭方程组；（2）计算了渐近贝叶斯证据，它在参数空间中起到配分函数的作用，编码了训练后模型的全局性质；（3）揭示了在训练过程中以及随超参数变化而出现的级联相变，这些相变与耦合矩阵最高模态对数据的相继对齐与凝聚有关。我们还将这些相变与教师-学生场景中采样阶段的生成现象联系起来，包括：采样温度（原文在此处截断）

    arXiv:2605.09031v2 Announce Type: replace  Abstract: Energy-based models (EBMs) are flexible generative architectures inspired by statistical physics, but their learning and generative properties remain poorly understood. Here, we analyze a solvable EBM in the high-dimensional limit: the spherical Boltzmann machine (SBM). Combining tools from random matrix theory and dynamical mean-field theory, we: derive and solve closed equations describing the training dynamics of the SBM in the high-dimensional limit; compute the asymptotic Bayesian evidence, which acts as a partition function in parameter space and encodes global properties of the trained model; and uncover cascades of phase transitions that occur both during training and as a function of hyperparameters, related to successive alignment and condensation of the top modes of the coupling matrix to the data. We connect these transitions to sampling-time generative phenomena in a teacher-student scenario, including: sampling temperat
    
[^311]: 感知调查设计的机器学习（SaML）：基于范围综述的人群健康有效推断指南

    Survey-aware Machine Learning: A Guideline for Valid Population Health Inference based on Scoping Review

    [https://arxiv.org/abs/2605.08963](https://arxiv.org/abs/2605.08963)

    本文提出了一个九步骤的感知调查设计机器学习（SaML）指南，通过将调查设计元数据（如抽样权重、分层变量等）纳入机器学习全生命周期，解决基于复杂健康调查数据训练模型时估计偏差、不确定性低估和公平性评估失真的问题。

    

    基于复杂健康调查（如国家健康与营养调查NHANES）训练的机器学习（ML）模型往往忽略初级抽样单元、分层变量和抽样权重。这种做法违反了标准评估方法的独立性假设，导致估计量产生偏差、不确定性被低估，且公平性评估无法反映人群层面的差异。我们提出了感知调查设计的机器学习，这是一个九步骤指南，将调查设计元数据纳入机器学习的整个生命周期。通过对16篇方法学论文的范围综述，我们总结了加权模型训练、基于设计的交叉验证以及调查调整性能评估方面的现有研究工作，并识别了超参数调优和模型部署方面的研究空白。我们还提供了针对特定任务的指导，阐明了不同分析目标所需的步骤。

    arXiv:2605.08963v2 Announce Type: replace-cross  Abstract: Machine Learning (ML) models trained on complex health surveys such as the National Health and Nutrition Examination Survey (NHANES) often ignore primary sampling units, stratification variables, and sampling weights. This practice violates the independence assumptions of standard evaluation methods. As a result, estimates become biased, uncertainty is underestimated, and fairness assessments fail to reflect population-level disparities. We propose Survey-aware Machine Learning (SaML), a nine-step guideline that incorporates survey design metadata across the ML lifecycle. Through a scoping review of 16 methodological papers, we summarize existing work on weighted model training, design-based cross-validation, and survey-adjusted performance evaluation. We also identify gaps in hyperparameter tuning and deployment. We provide task-specific guidance that clarifies which steps are required for different analytical objectives. SaML
    
[^312]: 面向多模态推理的结构化角色感知策略优化

    Structured Role-Aware Policy Optimization for Multimodal Reasoning

    [https://arxiv.org/abs/2605.07274](https://arxiv.org/abs/2605.07274)

    该论文提出结构化角色感知策略优化（SRPO），通过将多模态结构化响应分解为感知token与推理token，把序列级的GRPO优势细化为角色感知的token级优势，实现更精确的信用分配，从而确保正确答案有任务相关的视觉证据支撑。

    

    基于可验证奖励的强化学习（RLVR），尤其是结合组相对策略优化（GRPO），在提升大型视觉语言模型（LVLM）的推理能力方面已展现出强大潜力。然而，在多模态推理中，最终答案奖励通常是在序列级别进行分配的，无法区分不同token的功能角色，因此难以判断一个正确答案是否得到了与任务相关的视觉证据的支持。在本文中，我们从角色感知的token级信用分配视角重新审视多模态RLVR，将结构化响应分解为用于提取视觉证据的感知token，以及用于从这些证据推导出答案的推理token。基于这一视角，我们提出了结构化角色感知策略优化（SRPO），该方法在不改变……（原文在此处截断）的前提下，将序列级的GRPO优势细化为角色感知的token级优势。

    arXiv:2605.07274v2 Announce Type: replace  Abstract: Reinforcement learning from verifiable rewards (RLVR), especially with Group Relative Policy Optimization (GRPO), has shown strong potential for improving the reasoning capabilities of large vision-language models (LVLMs). However, in multimodal reasoning, final-answer rewards are typically assigned at the sequence level and do not distinguish the functional roles of different tokens, making it difficult to determine whether a correct answer is supported by task-relevant visual evidence. In this paper, we revisit multimodal RLVR from the perspective of role-aware token-level credit assignment, where structured responses are decomposed into perception tokens for extracting visual evidence and reasoning tokens for deriving answers from that evidence. Based on this perspective, we propose Structured Role-aware Policy Optimization (SRPO), which refines the sequence-level GRPO advantage into role-aware token-level advantages without chang
    
[^313]: FRInGe：基于Fisher-Rao几何的分布空间积分梯度

    FRInGe: Distribution-Space Integrated Gradients with Fisher--Rao Geometry

    [https://arxiv.org/abs/2605.06404](https://arxiv.org/abs/2605.06404)

    提出FRInGe方法，将积分梯度的参考点和积分路径定义在预测分布空间中，利用Fisher-Rao几何构造测地线路径，解决了传统积分梯度对设计选择高度敏感和梯度饱和的问题。

    

    尽管基于梯度的归因方法具有模型忠实性和可扩展性，但其一个著名变体——积分梯度- 对设计选择高度敏感，特别是在基线选择、路径构造和离散化方面，同时还存在梯度饱和问题。为了解决这些局限性，我们提出了Fisher-Rao积分梯度（FRInGe），该方法在预测分布空间中同时定义参考点和积分路径。FRInGe用最大熵预测参考替代输入基线，并遵循概率单纯形上的Fisher-Rao测地线。相应的输入空间轨迹通过拉回Fisher度量实现，并由Kullback-Leibler（KL）散度和欧几里得信任域加以稳定；归因结果通过沿该轨迹积分输入梯度获得。我们进一步推导了FRInGe-B变体，该变体将预测概率分布二值化，从而消除了计算（摘要在此处被截断）……

    arXiv:2605.06404v2 Announce Type: replace  Abstract: Although gradient-based attribution methods are model-faithful and scalable, a prominent variant, Integrated Gradients (IG), suffers from high sensitivity to design choices, particularly in baseline selection, path formulation, and discretization, as well as gradient saturation. To address these limitations, we propose Fisher--Rao Integrated Gradients (FRInGe), which defines both the reference and integration paths in the predictive distribution space. FRInGe replaces input baselines with a maximum-entropy predictive reference and follows a Fisher--Rao geodesic on the probability simplex. The corresponding input-space trajectory is realized through the pullback Fisher metric and stabilized by Kullback--Leibler (KL) and Euclidean trust regions; attributions are obtained by integrating input gradients along this trajectory. We further derive FRInGe-B, a variant that binarizes the predictive probability distribution, removing the comput
    
[^314]: 具有全局收敛性的梯度正则化牛顿提升树

    Gradient Regularized Newton Boosting Trees with Global Convergence

    [https://arxiv.org/abs/2605.00581](https://arxiv.org/abs/2605.00581)

    本文提出受限牛顿下降理论框架，证明了牛顿提升树在光滑强凸损失下的线性收敛速率，并通过梯度正则化扩展实现了对一般凸损失函数的全局收敛保证。

    

    梯度提升决策树（GBDT）在表格数据机器学习中占据主导地位，XGBoost、LightGBM 和 CatBoost 等现代实现均基于牛顿提升，即在决策树空间中的二阶下降步骤。尽管牛顿提升在经验上取得了成功，但与一阶提升相比，其全局收敛性却鲜有理论理解。在本文中，我们提出了受限牛顿下降，这是一个在希尔伯特空间上使用不精确迭代进行牛顿方法凸优化的框架，其建立在余弦角和弱梯度边的概念之上。在该框架内，基于 GBDT 的牛顿提升和经典有限维理论均可作为特例被涵盖。我们首先证明了原始牛顿提升对于满足海森矩阵占优条件的光滑强凸损失函数能够达到线性收敛速率。为了处理具有 Lipschitz 海森矩阵的一般凸损失函数，我们扩展了近期基于梯度的……

    arXiv:2605.00581v2 Announce Type: replace-cross  Abstract: Gradient Boosting Decision Trees (GBDTs) dominate tabular machine learning, with modern implementations like XGBoost, LightGBM, and CatBoost being based on Newton boosting: a second-order descent step in the space of decision trees. Despite its empirical success, the global convergence of Newton boosting is poorly understood compared to first-order boosting. In this paper, we introduce Restricted Newton Descent, a framework for convex optimization with Newton's method on Hilbert spaces with inexact iterates, based on the concepts of cosine angle and weak gradient edge. Within this framework, we recover Newton boosting with GBDTs and classical finite-dimensional theory as special cases. We first prove that vanilla Newton boosting achieves a linear rate of convergence for smooth, strongly convex losses that satisfy a Hessian-dominance condition. To handle general convex losses with Lipschitz Hessians, we extend a recent gradient 
    
[^315]: 超越视觉保真度：通过下游任务集成对大规模遥感影像超分辨率模型进行基准评估

    Beyond Visual Fidelity: Benchmarking Super-Resolution Models for Large-Scale Remote Sensing Imagery via Downstream Task Integration

    [https://arxiv.org/abs/2605.00310](https://arxiv.org/abs/2605.00310)

    该论文提出了GeoSR-Bench基准数据集，通过将土地覆盖分类、生物量估计、变化检测等下游任务集成到评估流程中，突破传统PSNR/SSIM保真度指标的局限，实现对遥感影像超分辨率模型实用价值的更全面评估。

    

    arXiv:2605.00310v3 公告类型：replace-cross 摘要：超分辨率（SR）技术在从低分辨率输入重建高分辨率图像方面已取得重大进展。提升的分辨率不仅带来视觉上的增强，也对监测任务具有实用价值。尤其是，超分辨率技术正越来越多地被开发用于基于卫星的对地观测，其应用涵盖城市规划、农业、生态学和灾害响应等领域。然而，现有的超分辨率研究和基准测试通常采用PSNR或SSIM等保真度指标，而超分辨率图像的真正价值在于支持诸如土地覆盖分类、生物量估计和变化检测等下游任务。为弥合这一差距，我们提出了GeoSR-Bench，这是一个下游任务集成的超分辨率基准数据集，用于在保真度指标之外评估超分辨率模型。GeoSR-Bench包含来自约36,000个位置、覆盖多样土地覆盖类型的空间位置对应、时间对齐且经过质量控制的图像对。

    arXiv:2605.00310v3 Announce Type: replace-cross  Abstract: Super-resolution (SR) techniques have made major advances in reconstructing high-resolution images from low-resolution inputs. The increased resolution provides visual enhancement and utility for monitoring tasks. In particular, SR has been increasingly developed for satellite-based Earth observation, with applications in urban planning, agriculture, ecology, and disaster response. However, existing SR studies and benchmarks typically use fidelity metrics such as PSNR or SSIM, whereas the true utility of super-resolved images lies in supporting downstream tasks such as land cover classification, biomass estimation, and change detection. To bridge this gap, we introduce GeoSR-Bench, a downstream task-integrated SR benchmark dataset to evaluate SR models beyond fidelity metrics. GeoSR-Bench comprises spatially co-located, temporally aligned, and quality-controlled image pairs from about 36,000 locations across diverse land covers
    
[^316]: 大维度下谱算法的学习曲线与欠正则化时的预测持续性

    Learning Curves and Prediction Persistence to Under-Regularization of Spectral Algorithms in Large Dimensions

    [https://arxiv.org/abs/2604.23212](https://arxiv.org/abs/2604.23212)

    该文在大维度体制（$n\asymp d^\gamma$）下，以常数因子精度刻画了谱算法随正则化从最优尺度减弱直至插值时条件超额风险的阶数，揭示了风险何时保持极小极大阶、何时出现多项式量级更大的欠正则化风险区间。

    

    我们研究当样本量 $n$ 与维度 $d$ 满足 $n\asymp d^\gamma$（$\gamma>0$ 为固定常数）时，谱算法在正则化程度减弱过程中预测风险的变化规律。对于球面上的内积核，在光滑度 $s\geq 0$ 的源条件与信号非退化条件下，我们在每个固定的可容许尺度 $\lambda\asymp d^{-u}$（$u>0$）以及插值（interpolation）情形下，以常数因子精度刻画了条件超额风险。所得的风险阶数描述了源光滑度、滤波器资格以及谱块之间转换的影响。这些结果识别了在正则化从最优尺度减弱至插值的过程中，条件超额风险何时仍保持极小极大风险阶数，以及何时在插值之前的某个区间会出现按多项式量级增大的风险。我们还使用确定性正则化方法，将相同谱滤波器下的核风险与总体序列风险进行了比较……

    arXiv:2604.23212v2 Announce Type: replace-cross  Abstract: We study how prediction risk changes as regularization is reduced for spectral algorithms when the sample size $n$ and dimension $d$ satisfy $n\asymp d^\gamma$ for fixed $\gamma>0$. For inner-product kernels on the sphere, under a source condition with smoothness $s\geq0$ and a signal nondegeneracy condition, we characterize the conditional excess risk up to constant factors at each fixed admissible scale $\lambda\asymp d^{-u}$, $u>0$, and at interpolation. The resulting risk orders describe the effects of source smoothness, filter qualification, and transitions between spectral blocks. They identify when the conditional excess risk retains the minimax risk order as regularization is weakened from an optimal scale to interpolation, and when an interval before interpolation exhibits a polynomially larger risk. We also compare the kernel risk with a population sequence risk for the same spectral filter, using a deterministic regu
    
[^317]: 摩擦增强漂移模型用于资源高效的域转换

    Friction-Augmented Drifting Models for Resource-Efficient Domain Translation

    [https://arxiv.org/abs/2604.18194](https://arxiv.org/abs/2604.18194)

    本文提出了一种带摩擦的漂移模型（DMF），通过线性调度的系数缩放漂移场，在域转换中防止迭代陷入虚假力平衡，从而显著提升性能。

    

    arXiv:2604.18194v2 公告类型：替换 摘要：单步生成器承诺以普通微分方程（ODE）为基础的流模型推理和训练成本的一小部分实现高保真合成，这在计算受限时是一个核心关注点。漂移模型（DMs）通过在基于核的漂移场下演化样本来训练单步生成器，完全避免了ODE积分，但其迭代的双粒子替代物允许一个“局部排斥”区域，其中排斥力可能主导对目标的吸引力。我们引入了DMF（带摩擦的漂移模型），它通过线性调度的系数$1-\gamma(i)$缩放漂移场。对替代物的闭式分析给出了每步收缩阈值和误差轨迹的有限时域界限，这解释了为什么摩擦可以在迭代松弛到虚假力平衡固定点之前停止迭代。在FFHQ潜在空间域转换中，DMF显著优于无摩擦版本。

    arXiv:2604.18194v2 Announce Type: replace  Abstract: Single-step generators promise high-fidelity synthesis at a fraction of the inference and training cost of ordinary differential equation (ODE)-based flow models, a central concern when compute is limited. Drifting Models (DMs) train a one-step generator by evolving samples under a kernel-based drift field, avoiding ODE integration entirely, but a two-particle surrogate of their iteration admits a \emph{locally repulsive} regime in which repulsion can dominate the attraction to the target. We introduce DMF (Drifting Model with Friction), which scales the drift field by a linearly-scheduled coefficient $1-\gamma(i)$. A closed-form analysis of the surrogate gives a per-step contraction threshold and a finite-horizon bound on the error trajectory, suggesting why friction can halt the iteration before it relaxes to a spurious force-balance fixed point. On FFHQ latent-space domain translation, DMF significantly improves on the frictionles
    
[^318]: 量子核方法中推理的最优算法复杂度

    Optimal algorithmic complexity of inference in quantum kernel methods

    [https://arxiv.org/abs/2604.15214](https://arxiv.org/abs/2604.15214)

    该论文通过将完整推理求和编码为单个可观测量的期望值并结合量子振幅估计，实现了量子核方法推理的最优查询复杂度。

    

    量子核方法是在监督学习中实现量子优势的主要候选方案之一。其关键瓶颈在于推理成本：在训练好的模型上对新数据进行评估，需要在加性精度 $\varepsilon$ 下估计由 $N$ 个核值构成的加权和 $\sum_{i=1}^N \alpha_i k(x,x_i)$，其中 $\alpha$ 是训练得到的系数向量。标准方法通过对每一项独立采样进行估计，产生 $O(N\lVert\alpha\rVert_2^2/\varepsilon^2)$ 的查询复杂度。在本工作中，我们识别出两个相互独立的改进维度：（1）如何估计单个核值（采样与量子振幅估计），（2）如何近似该求和（逐项求和与通过单个可观测量），并系统地分析了所有可能的组合。查询复杂度最优的组合将完整的推理求和编码为单个可观测量的期望值，并应用量子振幅估计……

    arXiv:2604.15214v3 Announce Type: replace-cross  Abstract: Quantum kernel methods are among the leading candidates for achieving quantum advantage in supervised learning. A key bottleneck is the cost of inference: evaluating a trained model on new data requires estimating a weighted sum $\sum_{i=1}^N \alpha_i k(x,x_i)$ of $N$ kernel values to additive precision $\varepsilon$, where $\alpha$ is the vector of trained coefficients. The standard approach estimates each term independently via sampling, yielding a query complexity of $O(N\lVert\alpha\rVert_2^2/\varepsilon^2)$. In this work, we identify two independent axes for improvement: (1) How individual kernel values are estimated (sampling versus quantum amplitude estimation), and (2) how the sum is approximated (term-by-term versus via a single observable), and systematically analyze all combinations thereof. The query-optimal combination, encoding the full inference sum as the expectation value of a single observable and applying qua
    
[^319]: OpenTME：一个来自TCGA的AI驱动的H&E肿瘤微环境谱开放数据集

    OpenTME: An Open Dataset of AI-powered H&E Tumor Microenvironment Profiles from TCGA

    [https://arxiv.org/abs/2604.12075](https://arxiv.org/abs/2604.12075)

    该研究发布了OpenTME数据集，利用AI病理基础模型从TCGA十种癌症类型的5,268张H&E全切片图像中生成大规模开放的肿瘤微环境定量图谱，每张切片包含超过20,000个定量指标。

    

    肿瘤微环境（TME）在癌症进展、治疗反应和患者预后中发挥着核心作用，然而从常规苏木精-伊红（H&E）染色组织病理学切片中获得大规模、一致且定量的TME表征仍然稀缺。我们推出了OpenTME，这是一个开放获取的数据集，包含从癌症基因组图谱（TCGA）中十种癌症类型（膀胱癌、乳腺癌、结直肠癌、头颈癌、肝癌、肺癌、卵巢癌、胰腺癌、前列腺癌和胃癌）的5,268张H&E染色全切片图像中预先计算得到的TME谱。所有输出均由HETA（H&E组织分析）生成，这是一个基于Atlas系列病理学基础模型构建的AI驱动应用程序，可执行组织质量控制、组织分割、细胞检测与分类、肿瘤区域和三级淋巴结构（TLS）检测以及空间邻域分析，每张切片可产生超过20,000个定量读数……

    arXiv:2604.12075v2 Announce Type: replace-cross  Abstract: The tumor microenvironment (TME) plays a central role in cancer progression, treatment response, and patient outcomes, yet large-scale, consistent, and quantitative TME characterization from routine hematoxylin and eosin (H&E)-stained histopathology remains scarce. We introduce OpenTME, an open-access dataset of pre-computed TME profiles derived from 5,268 H&E-stained whole-slide images across ten cancer types (bladder, breast, colorectal, head and neck, liver, lung, ovarian, pancreatic, prostate, and stomach cancer) from The Cancer Genome Atlas (TCGA). All outputs were generated using HETA (H&E Tissue Analysis), an AI-powered application built on the Atlas family of pathology foundation models, which performs tissue quality control, tissue segmentation, cell detection and classification, tumor zone and tertiary lymphoid structure (TLS) detection, and spatial neighborhood analysis, yielding over 20,000 quantitative readouts per
    
[^320]: 基于成分自适应与病灶级监督的脑部MRI小结构分割改进方法

    Component-Adaptive and Lesion-Level Supervision for Improved Small Structure Segmentation in Brain MRI

    [https://arxiv.org/abs/2604.08015](https://arxiv.org/abs/2604.08015)

    提出CATMIL训练目标，通过成分自适应Tversky加权和病灶级多示例学习两个辅助损失项，在不修改网络架构的前提下显著提升脑部MRI中小病灶的分割召回率。

    

    脑部MRI中的小病灶难以分割，因为它们仅占体积的极小部分，并且在体素级优化中被背景和较大病灶所主导，导致模型即使达到较高的Dice相似系数（DSC），仍可能遗漏许多小病灶。我们提出CATMIL，一种在不改变网络架构的情况下，向标准nnU-Net的Dice和交叉熵损失添加两个辅助项的训练目标。成分自适应Tversky（CAT）项根据病灶连通域大小的倒数对病灶体素进行加权，使每个病灶无论体积大小都贡献近乎相等的损失。病灶级多示例学习（MIL）项将每个病灶视为一个体素包，并对没有任何体素被检测出来的病灶施加惩罚。在MSLesSeg数据集的多发性硬化病灶分割任务中，CATMIL取得了最高的小病灶召回率（0.873，而Dice+CE为0.796；差异的95%置信区间为+0.030至+0.157，在所有六个测试中均更高）（摘要文本在此处截断）。

    arXiv:2604.08015v3 Announce Type: replace-cross  Abstract: Small lesions in brain MRI are hard to segment because they occupy a tiny fraction of the volume and are dominated by background and larger lesions during voxel-wise optimization, so a model can reach a high Dice similarity coefficient (DSC) while missing many of them. We propose CATMIL, a training objective that adds two auxiliary terms to the standard nnU-Net Dice and cross-entropy loss without changing the architecture. The Component-Adaptive Tversky (CAT) term weights lesion voxels by the inverse size of their connected component, so each lesion contributes nearly equally regardless of volume. The lesion-level Multiple Instance Learning (MIL) term treats each lesion as a bag of voxels and penalizes lesions with no detected voxel. For multiple sclerosis lesion segmentation on MSLesSeg, CATMIL achieves the highest small-lesion recall (0.873 vs. 0.796 for Dice+CE; 95% CI of the difference +0.030 to +0.157, higher in all six te
    
[^321]: 资源受限定价中信息的价值

    The Value of Information in Resource-Constrained Pricing

    [https://arxiv.org/abs/2603.24974](https://arxiv.org/abs/2603.24974)

    本文研究了资源受限动态定价中的信息价值，在完全信息、无信息和知情定价三种信息环境下分别设计了算法：完全信息下仅求解一次流体问题的残差修正策略达到 O(log T) 后悔值，无信息下确定性探索加确定性等价策略达到 O(√T) 后悔值，知情定价下反馈控制器在斜率矩阵对称时无需估计斜率即可学习最优实施价格。

    

    在资源约束下的动态定价中，需要在探索未知需求模式与利用已知信息之间进行权衡，以实现收益最大化。我们针对该问题在不同的信息环境下开发了相应的算法。产品共享易逝性资源，企业发布价格并观察实际到达的需求请求，这些请求可能超过剩余容量。在完全信息环境下，一种仅求解一次流体问题的残差修正策略，在有界价格和任意秩资源矩阵的条件下可达到 O(log T) 的后悔值。在无信息环境下，先进行确定性探索再采用确定性等价策略，当极限流体解非退化时可达到 O(√T) 的后悔值。在知情定价环境下，给定初始的价格-需求估计，若斜率矩阵是对称的，反馈控制器无需估计斜率矩阵即可学习实施价格……（原文摘要此处截断）

    arXiv:2603.24974v3 Announce Type: replace-cross  Abstract: Dynamic pricing with resource constraints requires balancing exploration of unknown demand patterns against exploitation of known information to maximize revenue. We develop algorithms for this problem across different information regimes. Products share perishable resources, and the firm posts prices and observes the realized requests, which may exceed the remaining capacity. In the full-information setting, a residual-correction policy that solves the fluid problem once attains $O(\log T)$ regret with bounded prices and arbitrary-rank resource matrices. In the no-information setting, deterministic exploration followed by a certainty-equivalent policy attains $O(\sqrt T)$ regret when the limiting fluid solution is nondegenerate. In the informed-price setting, given an initial price-demand estimate, a feedback controller learns the implementing prices without estimating the slope matrix if the slope matrix is symmetric, a set c
    
[^322]: 概率分布的连续时间学习：1型糖尿病幼儿数字试验中的案例研究

    Continuous-time learning of probability distributions: a case study in a digital trial of young children with type 1 diabetes

    [https://arxiv.org/abs/2603.24427](https://arxiv.org/abs/2603.24427)

    提出一个基于高斯混合模型与神经常微分方程的连续时间概率分布学习框架，用于刻画1型糖尿病儿童连续血糖监测数据中血糖分布随时间的动态演变。

    

    理解生物标志物分布如何随时间演变是数字健康和慢性病监测中的核心挑战。例如在糖尿病中，血糖测量值分布的变化可以揭示传统汇总指标所忽略的疾病进展与治疗反应模式。受一项为期13周、比较闭环胰岛素输注系统 t:slim X2 与标准疗法在1型糖尿病儿童中疗效的临床试验启发，我们提出了一个概率框架，利用每五分钟采集一次的连续血糖监测（CGM）数据来建模时间索引分布的连续时间演变。我们将血糖分布表示为高斯混合模型，其时变的混合权重由神经常微分方程（neural ODE）控制，并基于最大均值差异（MMD）的分布匹配准则来估计模型参数。由此得到的框架具有可解释性和计算高效性。

    arXiv:2603.24427v2 Announce Type: replace-cross  Abstract: Understanding how biomarker distributions evolve over time is a central challenge in digital health and chronic disease monitoring. For example, in diabetes, changes in the distribution of glucose measurements can reveal patterns of disease progression and treatment response that conventional summary measures miss. Motivated by a 13-week clinical trial comparing the closed-loop insulin delivery system t:slim X2 with standard therapy in children with type 1 diabetes, we propose a probabilistic framework to model the continuous-time evolution of time-indexed distributions using continuous glucose monitoring (CGM) data collected every five minutes. We represent the glucose distribution as a Gaussian mixture, with time-varying mixture weights governed by a neural ODE. We estimate the model parameters using a distribution-matching criterion based on the maximum mean discrepancy. The resulting framework is interpretable, computationa
    
[^323]: λSplit：面向荧光显微镜的自监督内容感知光谱解混方法

    {\lambda}Split: Self-Supervised Content-Aware Spectral Unmixing for Fluorescence Microscopy

    [https://arxiv.org/abs/2603.23647](https://arxiv.org/abs/2603.23647)

    提出了λSplit，一种基于物理信息的自监督深度生成模型，通过分层变分自编码器学习浓度图的条件分布，并利用可微分光谱混合器，实现了对荧光显微镜光谱图像的内容感知解混。

    

    在荧光显微镜中，光谱解混旨在从捕获了混合荧光团发射信号的光谱图像中恢复各个荧光团的浓度。由于经典方法采用逐像素操作并依赖最小二乘拟合，其性能会随着发射光谱重叠程度的加剧和噪声水平的升高而下降，这表明一种能够学习并利用结构先验的数据驱动方法可能会带来更好的结果。虽然目前存在基于学习的光谱成像方法，但它们要么未针对显微镜数据进行优化，要么是为极特殊的场景开发的，并不适用于荧光显微镜的设置。为解决这一问题，我们提出了λSplit，这是一种融合物理信息的深度生成模型，利用分层变分自编码器学习浓度图的条件分布。一个完全可微分的光谱混合器强制其与图像保持一致性……

    arXiv:2603.23647v4 Announce Type: replace-cross  Abstract: In fluorescence microscopy, spectral unmixing aims to recover individual fluorophore concentrations from spectral images that capture mixed fluorophore emissions. Since classical methods operate pixel-wise and rely on least-squares fitting, their performance degrades with increasingly overlapping emission spectra and higher levels of noise, suggesting that a data-driven approach that can learn and utilize a structural prior might lead to improved results. Learning-based approaches for spectral imaging do exist, but they are either not optimized for microscopy data or are developed for very specific cases that are not applicable to fluorescence microscopy settings. To address this, we propose {\lambda}Split, a physics-informed deep generative model that learns a conditional distribution over concentration maps using a hierarchical Variational Autoencoder. A fully differentiable Spectral Mixer enforces consistency with the image 
    
[^324]: 基于在线推断的层级潜在结构学习

    Hierarchical Latent Structure Learning through Online Inference

    [https://arxiv.org/abs/2603.19139](https://arxiv.org/abs/2603.19139)

    该论文提出HOLMES计算框架，通过结合嵌套中餐馆过程先验的变体与序贯蒙特卡洛推断，首次实现了无需显式监督、可逐试验在线进行层级潜在结构学习的推断方法，并在保持平坦模型预测性能的同时获得更紧凑的表征。

    

    学习系统必须在跨经验的泛化能力与对任务相关细节的辨别能力之间取得平衡。因此，有效的学习需要能够同时支持两者的表征。在线潜在原因模型支持增量式推断，但假设的是平坦的划分结构；而层级贝叶斯模型能够捕捉多层级结构，但通常需要进行离线推断。我们提出了HOLMES模型（多尺度经验结构的层级在线学习，Hierarchical Online Learning of Multiscale Experience Structure），这是一个通过在线推断实现层级潜在结构学习的计算框架。HOLMES将嵌套中餐馆过程先验的一个变体与序贯蒙特卡洛推断相结合，在无需对潜在结构进行显式监督的情况下，实现对层级潜在表征的可处理的逐试验推断。在模拟实验中，HOLMES在预测性能上与平坦模型相当，同时学习到了更紧凑的表征。

    arXiv:2603.19139v3 Announce Type: replace  Abstract: Learning systems must balance generalization across experiences with discrimination of task-relevant details. Effective learning therefore requires representations that support both. Online latent-cause models support incremental inference but assume flat partitions, whereas hierarchical Bayesian models capture multilevel structure but typically require offline inference. We introduce the \textbf{Hierarchical Online Learning of Multiscale Experience Structure (HOLMES) model}, a computational framework for hierarchical latent structure learning through online inference. HOLMES combines a variation on the nested Chinese Restaurant Process prior with sequential Monte Carlo inference to perform tractable trial-by-trial inference over hierarchical latent representations without explicit supervision over the latent structure. In simulations, HOLMES matched the predictive performance of flat models while learning more compact representation
    
[^325]: 条件随机优化的函数学习方法及其收敛速率

    Convergence Rate of a Functional Learning Method for Conditional Stochastic Optimization

    [https://arxiv.org/abs/2603.13048](https://arxiv.org/abs/2603.13048)

    本文针对无法直接从条件分布采样的条件随机优化问题，提出在参数函数类中近似条件期望并同时学习与优化的算法，并给出了该方法的收敛速率分析。

    

    我们考虑一个涉及两个随机变量的随机优化问题：上下文变量 $X$ 和因变量 $Y$。目标是最小化应用于条件期望 $\mathbb{E}[f(X, Y,\beta) \mid X]$ 的非线性损失泛函的期望值，其中 $f$ 是非线性函数，$\beta$ 表示决策变量。我们关注一个具有重要实际意义的场景：在这种情况下，无法直接从 $Y \mid X$ 的条件分布中采样，只有独立同分布的观测对数据流 $\{(X^k, Y^k)\}_{k=0,1,2,\ldots}$ 可用。在我们的方法中，条件期望在一个预先指定的参数函数类中被近似。我们分析了一种同时学习与优化的算法，该算法联合估计条件期望并优化外层目标函数。通过使用一种特殊设计的非最优性度量，并结合……的平方范数（摘要原文在此处截断）

    arXiv:2603.13048v3 Announce Type: replace-cross  Abstract: We consider a stochastic optimization problem involving two random variables: a context variable $X$ and a dependent variable $Y$. The objective is to minimize the expected value of a nonlinear loss functional applied to the conditional expectation $\mathbb{E}[f(X, Y,\beta) \mid X]$, where $f$ is a nonlinear function and $\beta$ represents the decision variables. We focus on the practically important setting in which direct sampling from the conditional distribution of $Y \mid X$ is infeasible, and only a stream of i.i.d. observation pairs $\{(X^k, Y^k)\}_{k=0,1,2,\ldots}$ is available. In our approach, the conditional expectation is approximated within a prespecified parametric function class. We analyze a simultaneous learning-and-optimization algorithm that jointly estimates the conditional expectation and optimizes the outer objective. Using a specially designed measure of non-optimality, combining the squared norm of the o
    
[^326]: SimVLA：通过受控消融实验归因VLA模型的性能提升

    SimVLA: Attributing Gains in VLA Models Through Controlled Ablation

    [https://arxiv.org/abs/2602.18224](https://arxiv.org/abs/2602.18224)

    该论文提出刻意极简的VLA基线模型SimVLA作为受控实验平台，通过单变量消融实验发现训练动态（而非架构或任务配置）才是VLA模型性能提升的最大驱动因素，可带来约54%–94%的性能波动。

    

    视觉-语言-动作（VLA）模型通常将架构变更与不同的预训练数据、骨干网络规模和优化方案捆绑在一起，这使得究竟是什么因素真正推动了进步变得模糊不清。我们提出了SimVLA——一个刻意极简的VLA模型，即一个标准的视觉-语言骨干网络加上一个轻量级的连续动作头——作为解决这一归因问题的受控实验平台；我们的贡献是方法论和实证层面的（一项受控归因研究和一个最小基线校准），而非一种新技术。在LIBERO、CALVIN、WidowX和Google Robot四个基准上，一次只改变一个变量的消融实验表明，训练动态是测量出的最大驱动因素，可产生约54%–94%的性能波动，而在匹配设置下，任务配置和架构的影响较小。尽管SimVLA使用了小型骨干网络，且在针对基准的训练之前没有进行额外的大规模机器人数据VLA预训练，其仍达到了98.6%的……

    arXiv:2602.18224v2 Announce Type: replace-cross  Abstract: Vision-Language-Action (VLA) models often bundle architectural changes with different pretraining data, backbone scales, and optimization recipes, obscuring what actually drives progress. We introduce SimVLA, a deliberately minimal VLA, a standard vision-language backbone with a lightweight continuous-action head, as a controlled testbed for this attribution problem; our contribution is methodological and empirical (a controlled attribution study and a minimal-baseline calibration), not a new technique. Across LIBERO, CALVIN, WidowX, and Google Robot, one-knob-at-a-time ablations show that training dynamics are the largest measured driver, producing performance swings of $\Delta$$\approx$54--94\%, while task configuration and architecture have smaller effects under matched settings. Despite using a small backbone and no additional large-scale robot-data VLA pretraining before benchmark-specific training, SimVLA reaches 98.6\% a
    
[^327]: 关于具有 Q\* 近似与部分覆盖的离线强化学习的复杂性研究

    On the Complexity of Offline Reinforcement Learning with Q*-Approximation and Partial Coverage

    [https://arxiv.org/abs/2602.12107](https://arxiv.org/abs/2602.12107)

    该论文通过信息论下界证明仅凭 Q\* 可实现性与贝尔曼完整性不足以实现部分覆盖下样本高效的离线强化学习，并提出一个通用的决策估计框架，将离线强化学习的复杂度分解为决策复杂度与价值估计误差两部分。

    

    我们研究了在 Q\* 近似与部分覆盖条件下的离线强化学习，这一设定激发了许多实用算法（如保守 Q 学习，CQL；Kumar 等，2020），但其受到的理论关注十分有限。我们的工作受到以下开放问题的启发：“Q\* 可实现性与贝尔曼完整性是否足以支撑部分覆盖下样本高效的离线强化学习？”我们通过信息论下界对该问题给出了否定回答。为了识别能够使部分覆盖下样本高效离线强化学习成为可能的额外结构，我们受在线强化学习中无模型决策估计系数（DEC）（Foster 等，2023b；Liu 等，2025b）的启发，引入了一个通用的决策估计框架。该框架将离线强化学习的复杂度分解为决策复杂度和价值估计误差，从而可以对这两个子问题进行模块化研究。我们的结果不仅统一了已有的……

    arXiv:2602.12107v3 Announce Type: replace-cross  Abstract: We study offline reinforcement learning under $Q^\star$-approximation and partial coverage, a setting that motivates practical algorithms such as Conservative $Q$-Learning (CQL; Kumar et al., 2020) but has received limited theoretical attention. Our work is inspired by the following open question: "Are $Q^\star$-realizability and Bellman completeness sufficient for sample-efficient offline RL under partial coverage?"   We answer in the negative via an information-theoretic lower bound. To identify additional structure that enables sample-efficient offline RL under partial coverage, we introduce a general decision-estimation framework, inspired by model-free decision-estimation coefficients (DEC) for online RL (Foster et al., 2023b; Liu et al., 2025b). Our framework decomposes offline RL complexity into decision complexity and value estimation error. This allows modular study of both sub-problems. Our result not only unifies exi
    
[^328]: 量子局部差分隐私中的隐私-效用权衡

    Privacy-Utility Tradeoffs with Quantum Local Differential Privacy

    [https://arxiv.org/abs/2602.10510](https://arxiv.org/abs/2602.10510)

    该论文首次系统研究了$(\varepsilon,\delta)$-量子局部差分隐私框架下的最优隐私-效用权衡，证明了去极化机制在保真度和迹距离等通用效用度量下达到理论最优。

    

    当敏感信息被编码在数据中时，在尝试从数据中学习有用信息的同时确保信息的隐私性至关重要。这里存在一个自然的权衡：提高隐私要求可能会降低学习协议的效用。在差分隐私的量子设定中，这种隐私与效用之间的权衡迄今为止在很大程度上尚未被探索。在本工作中，我们研究了当隐私由$(\varepsilon,\delta)$-量子局部差分隐私量化时，针对通用效用度量和特定应用效用度量的最优隐私-效用权衡。在通用设定中，我们专注于优化原始态与私有化态之间的保真度和迹距离。我们证明了去极化机制在给定的隐私要求下能够实现最优效用。随后，我们研究了学习可观测量期望值这一具体应用（摘要在此处截断）。

    arXiv:2602.10510v2 Announce Type: replace-cross  Abstract: When sensitive information is encoded in data, it is important to ensure the privacy of information when attempting to learn useful information from the data. There is a natural tradeoff whereby increasing privacy requirements may decrease the utility of a learning protocol. In the quantum setting of differential privacy, such tradeoffs between privacy and utility have so far remained largely unexplored. In this work, we study optimal privacy-utility tradeoffs for both generic and application-specific utility metrics when privacy is quantified by $(\varepsilon,\delta)$-quantum local differential privacy. In the generic setting, we focus on optimizing fidelity and trace distance between the original state and the privatized state. We show that the depolarizing mechanism achieves the optimal utility for given privacy requirements. We then study the specific application of learning the expectation of an observable with respect to 
    
[^329]: 楔形采样：具有近线性样本复杂度的高效张量补全

    Wedge Sampling: Efficient Tensor Completion with Nearly-Linear Sample Complexity

    [https://arxiv.org/abs/2602.05869](https://arxiv.org/abs/2602.05869)

    本文提出“楔形采样”这一非自适应采样方案，通过对张量展开同列元素进行成对采样以提供谱初始化所需的行相关性，从而在固定秩条件下以近线性样本复杂度实现低秩张量的多项式时间恢复。

    

    我们提出了楔形采样（wedge sampling），这是一种非自适应采样方案，用于从张量元素的结构化观测中恢复阶数为 k、边长为 n 的低秩张量。该方案在张量展开矩阵的同一列中对元素进行成对采样，从而在相关的二部采样图中形成长度为二的路径（即楔形）。在均匀采样产生楔形数量不足的情形下，成对元素的乘积为谱初始化提供了所需的行相关性。在秩、不相干性、张量阶数和条件数均为固定的情况下，楔形采样能够以关于 n 近似线性的样本复杂度实现多项式时间的恢复。该谱初始化器可与现有的精细化程序结合，仅需额外使用 Õ(n) 个均匀采样的元素。谱精细化过程可实现到任意固定相对 Frobenius 误差的弱恢复。对于具有不相干典型多分解（CP）因子的对称三阶张量，……

    arXiv:2602.05869v4 Announce Type: replace-cross  Abstract: We introduce wedge sampling, a non-adaptive sampling scheme for recovering an order-$k$ low-rank tensor of side length $n$ from structured observations of its entries. The scheme samples pairs of entries in the same column of a tensor unfolding, forming length-two paths (wedges) in the associated bipartite sampling graph. The products of the paired entries supply the row correlations needed for spectral initialization in regimes where uniform sampling yields too few wedges. For fixed rank, incoherence, tensor order, and condition number, wedge sampling enables polynomial-time recovery with nearly linear sample complexity in $n$. The spectral initializer can be combined with existing refinement procedures using only an additional $\tilde O(n)$ uniformly sampled entries. Spectral refinement achieves weak recovery to any fixed relative Frobenius error. For symmetric third-order tensors with incoherent canonical polyadic factors, g
    
[^330]: EMA策略梯度：利用EMA锚点与Top-k KL驯服大语言模型的强化学习

    EMA Policy Gradient: Taming Reinforcement Learning for LLMs with EMA Anchor and Top-k KL

    [https://arxiv.org/abs/2602.04417](https://arxiv.org/abs/2602.04417)

    提出EMA锚点策略和Top-k KL估计器两项简单技术来改进大语言模型的策略梯度算法，结合GRPO后在数学推理和智能体强化学习任务上取得显著性能提升。

    

    强化学习（RL）已使大语言模型（LLM）能够掌握日益复杂的推理和智能体行为。在本工作中，我们提出两种简单的技术来改进面向大语言模型的策略梯度算法。首先，我们用指数移动平均（EMA）替换强化学习过程中的固定锚点策略，这类似于深度Q学习中的目标网络。其次，我们引入Top-$k$ KL估计器，它允许在精确KL与采样KL之间进行灵活插值。我们推导了使用EMA锚点的稳定性条件；此外，我们证明了Top-$k$ KL估计器在任意$k$值下都能产生无偏的KL值和无偏的梯度，同时兼具精确KL的优势。将这两种技术与GRPO相结合（EMA-PG）可带来显著的性能提升。在数学推理方面，它使R1蒸馏的Qwen-1.5B在OlympiadBench上达到53.9%，而GRPO为50.8%。在智能体强化学习领域，使用Qwen-3B……（摘要原文在此处截断）

    arXiv:2602.04417v2 Announce Type: replace-cross  Abstract: Reinforcement Learning (RL) has enabled Large Language Models (LLMs) to acquire increasingly complex reasoning and agentic behaviors. In this work, we propose two simple techniques to improve policy gradient algorithms for LLMs. First, we replace the fixed anchor policy during RL with an Exponential Moving Average (EMA), similar to a target network in deep Q-learning. Second, we introduce Top-$k$ KL estimator, which allows for flexible interpolation between exact KL and sampled KL. We derive the stability conditions for using EMA anchor; moreover, we show that our Top-$k$ KL estimator yields both unbiased KL values and unbiased gradients at any $k$, while bringing the benefits of exact KL. When combined with GRPO, the two techniques (EMA-PG) lead to a significant performance boost. On math reasoning, it allows R1-distilled Qwen-1.5B to reach 53.9% on OlympiadBench compared to 50.8% by GRPO. On agentic RL domains, with Qwen-3B b
    
[^331]: 任意时刻有效的保形风险控制

    Anytime-Valid Conformal Risk Control

    [https://arxiv.org/abs/2602.04364](https://arxiv.org/abs/2602.04364)

    本文提出了任意时刻有效的保形风险控制方法，将误差控制从固定大小校准数据集上的平均有效性扩展为随校准数据累积增长时在任意时刻均以高概率保持有效，并证明了该保证渐近紧致且修正速率不可再改进。

    

    预测集为预测任务中的不确定性量化提供了一种手段。利用留出的校准数据，保形预测和风险控制能够以计算高效的方式生成具有统计上有效误差控制的预测集。然而，在标准形式中，误差仅在固定大小的多个可能校准数据集上进行平均控制。在本文中，我们将这种控制扩展为：随着校准数据集的累积增长，在任意时间点上都能以高概率保持有效。我们利用基于分位数的论证推导出此类保证，并说明了所提出框架在涉及分布偏移的情境中的适用性。我们进一步证明了这些保证是渐近紧的，即风险从下方收敛到目标水平，并且我们的修正速率无法再被改进。最后，我们展示了该方法的实际性能。

    arXiv:2602.04364v2 Announce Type: replace-cross  Abstract: Prediction sets provide a means of quantifying the uncertainty in predictive tasks. Using held out calibration data, conformal prediction and risk control can produce prediction sets that exhibit statistically valid error control in a computationally efficient manner. However, in the standard formulations, the error is only controlled on average over many possible calibration datasets of fixed size. In this paper, we extend the control to remain valid with high probability over a cumulatively growing calibration dataset at any time point. We derive such guarantees using quantile-based arguments and illustrate the applicability of the proposed framework to settings involving distribution shift. We further show that our guarantees are asymptotically tight, in the sense that the risks converge to the target level from below, and that the rate of our correction cannot be improved. Finally, we demonstrate the practical performance o
    
[^332]: SpikingGamma：通过平滑时间延迟实现时间精确的在线脉冲神经网络训练

    SpikingGamma: Temporally Precise Online SNN Training Through Smoothed Temporal Delays

    [https://arxiv.org/abs/2602.01978](https://arxiv.org/abs/2602.01978)

    SpikingGamma是一种新型SNN架构，神经元通过维护紧凑的平滑延迟状态库并以sigma-delta脉冲编码进行通信，实现了时间精确的在线训练，克服了传统基于代理梯度的方法在精细时间分辨率下扩展性差、在线近似不稳定且难以捕捉时间模式的局限。

    

    脉冲神经网络（SNN）的神经形态硬件实现通过稀疏的、事件驱动的计算，有望实现节能、低延迟的人工智能。然而，在精细时间离散化下训练SNN仍然具有挑战性，这阻碍了低延迟响应能力以及将软件训练的SNN映射到高效硬件上。在当前的方法中，脉冲神经元被建模为自循环单元，嵌入循环网络中以随时间维持状态，并使用基于代理梯度的BPTT或RTRL变体进行训练。这些方法在时间分辨率提高时扩展性较差，而在线近似方法在长序列上表现出不稳定性，且往往无法捕捉时间模式。为了解决这些局限性，我们开发了SpikingGamma，这是一种SNN架构，其中神经元维护一个紧凑的平滑延迟状态库，并通过sigma-delta脉冲编码进行通信。我们证明在前馈网络中，SpikingGamma能够实现……

    arXiv:2602.01978v2 Announce Type: replace-cross  Abstract: Neuromorphic hardware implementations of Spiking Neural Networks (SNNs) promise energy-efficient, low-latency AI through sparse, event-driven computation. Yet, training SNNs under fine temporal discretization remains challenging, hindering low-latency responsiveness and the mapping of software-trained SNNs to efficient hardware. In current approaches, spiking neurons are modeled as self-recurrent units, embedded into recurrent networks to maintain state over time, and trained with BPTT or RTRL variants based on surrogate gradients. These methods scale poorly with temporal resolution, while online approximations exhibit instability for long sequences and tend to fail at capturing temporal patterns. To address these limitations, we develop SpikingGamma, an SNN architecture in which neurons maintain a compact bank of smoothed-delay states and communicate through sigma-delta spike-coding. We show that in feedforward networks, Spiki
    
[^333]: 后训练困境：为什么我们应该重新思考顺序式SFT-RL范式

    The Post-Training Dilemma: Why We Should Rethink the Sequential SFT-RL Paradigm

    [https://arxiv.org/abs/2601.07389](https://arxiv.org/abs/2601.07389)

    本文从理论上证明SFT与RL在近乎正交的参数子空间上进行优化，导致RL阶段不可避免地损害SFT所获得的能力，且KL惩罚无法修复这一梯度不对齐问题，因此主张重新思考顺序式SFT-RL后训练范式。

    

    这篇立场论文认为，先监督微调后强化学习（SFT-then-RL）的顺序流程——作为大语言模型（LLM）后训练的事实标准范式——存在一个无法通过超参数调整来修复的根本性缺陷：SFT和RL在几乎正交的参数子空间上进行优化，因此RL不可避免地会损害前一阶段SFT所获得的能力。在Polyak-Łojasiewicz（PL）条件下（该条件在过参数化网络中被经验性观察到），我们证明RL至少会使SFT损失增加 $\frac{\mu}{2}\mathrm{dist}(\theta_{\mathrm{RL}}, \Theta^*_{\mathrm{SFT}})^2$（其中 $\mu>0$）。尽管KL惩罚可以约束分布偏移，但它无法纠正两个目标之间潜在的梯度不对齐问题。在Qwen3-0.6B上跨四个任务的实验证实了理论预测：在MATH-500、Alpaca、HH-RLHF和SST-2上（原文摘要在此处截断）。

    arXiv:2601.07389v3 Announce Type: replace-cross  Abstract: This position paper argues that the sequential supervised fine-tuning followed by reinforcement learning (SFT-then-RL) pipeline---the de facto paradigm for large language model (LLM) post-training---has a fundamental property that cannot be fixed by hyperparameter tuning: SFT and RL optimize over nearly orthogonal parameter subspaces, so RL inevitably degrades the capabilities acquired during the preceding SFT phase. Under the Polyak-Lojasiewicz (PL) condition, which is empirically observed for over-parameterized networks, we show that RL increases the SFT loss by at least $\frac{\mu}{2}\mathrm{dist}(\theta_{\mathrm{RL}}, \Theta^*_{\mathrm{SFT}})^2$ for some $\mu>0$. Although the KL penalty can constrain distributional shift, it cannot correct the underlying gradient misalignment between the two objectives. Experiments on Qwen3-0.6B across four tasks confirm the theoretical predictions: on MATH-500, Alpaca, HH-RLHF, and SST-2, 
    
[^334]: 在 $O(n\log n)$ 时间内学习多项Logit模型

    Learning Multinomial Logits in $O(n \log n)$ time

    [https://arxiv.org/abs/2601.04423](https://arxiv.org/abs/2601.04423)

    本文针对多项Logit（MNL）模型的学习问题，提出了自适应与非自适应算法，能够以 $O(\frac{n\log n}{\varepsilon^2})$ 次查询使每个选择板上的诱导选择分布在总变差距离 $\varepsilon$ 内逼近真实分布。

    

    多项Logit（MNL）模型由一个有限的物品全集 $[n]=\{1,\dots,n\}$ 组成，其中每个物品被赋予一个正权重。一次查询指定一个可接受的子集——称为一个“选择板”——模型以与权重成正比的概率从该选择板中选择一个物品。这种查询模型在文献中也被称为 Plackett–Luce 模型或条件采样预言机。尽管MNL模型已被广泛研究，但一个基本的计算问题仍未解决：在给定对选择板的查询访问的条件下，我们能以多高的效率学习各物品的权重，使得对于每一个选择板，模型诱导出的选择分布与真实分布之间的总变差距离不超过 $\varepsilon$？这一问题居于MNL学习的核心，并对现代推荐系统界面具有直接的影响。我们为该任务提供了自适应与非自适应两类算法。我们的第一个自适应算法进行 $O(\frac{n\log n}{\varepsilon^2})$ 次查询并精确学习……

    arXiv:2601.04423v2 Announce Type: replace-cross  Abstract: A Multinomial Logit (MNL) model is composed of a finite universe of items $[n]=\{1,\dots,n\}$, each assigned a positive weight. A query specifies an admissible subset---called a slate---and the model chooses one item from that slate with probability proportional to its weight. This query model is also known as the Plackett--Luce model or conditional sampling oracle in the literature. Although MNLs have been studied extensively, a basic computational question remains open: given query access to slates, how efficiently can we learn weights so that, for every slate, the induced choice distribution is within total variation distance $\varepsilon$ of the ground truth? This question is central to MNL learning and has direct implications for modern recommender system interfaces. We provide both adaptive and non-adaptive algorithms for this task. Our first adaptive algorithm makes $O(\frac{n\log n}{\varepsilon^2})$ queries and learns e
    
[^335]: 神经网络场论中的Virasoro对称性

    Virasoro Symmetry in Neural Network Field Theories

    [https://arxiv.org/abs/2512.24420](https://arxiv.org/abs/2512.24420)

    本文从神经自由场架构出发严格构造出中心荷为一的Virasoro代数，证明极限神经流关联与Fock真空关联一致，并将该构造推广至费米子与超对称扇区。

    

    我们从神经自由场架构中构造出Virasoro代数。一个对数动量律和一个光滑谱窗口给出了权重为一的流核，其径向排序的不连续性确定了振荡器配对。极限下的神经流关联与Fock真空关联一致。其Sugawara应力张量给出中心荷为一的Virasoro代数，带荷顶点插入识别出其最高权扇区。对调节器误差与特征误差的界控制了该表示的有限模式近似。在有限宽度下，一个显式的伴随相容投影在其厄米混合块为正时定义了一个辅助Fock读出。柯西（Cauchy）特征将该构造推广至费米子与超对称扇区，包括Ramond表示中的Clifford零模，而一阶应力张量给出费米子与玻色子幽灵系统。共享的反射特征……（原文摘要在此处截断）

    arXiv:2512.24420v4 Announce Type: replace-cross  Abstract: We construct the Virasoro algebra from a neural free-field architecture. A logarithmic momentum law and a smooth spectral window give the weight-one current kernel, whose radial-ordering discontinuity fixes the oscillator pairing. The limiting neural current correlations agree with the Fock vacuum correlations. Their Sugawara stress tensor gives the Virasoro algebra with central charge one, and charged vertex insertions identify its highest-weight sectors. Bounds on regulator and feature errors control finite-mode approximations to this representation. At finite width, an explicit adjoint-compatible projection defines an auxiliary Fock readout when its Hermitian mixed block is positive. Cauchy features extend the construction to fermionic and supersymmetric sectors, including a Clifford zero mode in the Ramond representation, while first-order stress tensors give fermionic and bosonic ghost systems. Shared reflected features im
    
[^336]: 量子非抽取小波变换：理论、电路与应用

    Quantum Nondecimated Wavelet Transform: Theory, Circuits, and Applications

    [https://arxiv.org/abs/2512.21478](https://arxiv.org/abs/2512.21478)

    该论文提出了量子非抽取小波变换的理论与电路实现，证明了其层级统计量在循环移位下不变，并通过块编码收缩算子与移位寄存器后选择精确实现循环平移平均，从而完成平移等变的量子去噪。

    

    量子非抽取小波变换使用移位寄存器在单个正交小波变换之前对幅度编码信号施加循环移位。我们证明，对移位寄存器求偏迹可得到等于归一化非抽取小波能量的层级概率，且该概率在循环移位下保持不变。其对角线给出系数平方的移动平均。我们推导了层级计数的多项式模型，并展示了如何读取除最粗细节层级之外的所有层级的滞后一阶自协方差。对于这些全局统计量，直接对输入进行傅里叶基测量至少同样精确，且所需电路更简单。对于去噪，我们在变换与其逆变换之间插入一个块编码的收缩算子。通过对移位寄存器的后选择，以酉算子线性组合的形式精确实现了循环平移平均。在每个层级内共享的权重给出了平移等变的滤波

    arXiv:2512.21478v2 Announce Type: replace-cross  Abstract: The quantum nondecimated wavelet transform uses a shift register to apply circular shifts to an amplitude encoded signal before a single orthogonal wavelet transform. We show that tracing out the shift register gives level probabilities equal to the normalized nondecimated wavelet energies, invariant under circular shifts. Its diagonal gives moving averages of squared coefficients. We derive a multinomial model for the level counts and show how to read the autocovariance at lag one for all but the coarsest detail level. For these global statistics, direct Fourier basis measurement of the input is at least as precise and uses a simpler circuit.   For denoising, we insert a block encoded shrinkage operator between the transform and its inverse. Postselection on the shift register implements the cycle spinning average exactly as a linear combination of unitaries. Weights shared within each level give a translation equivariant filt
    
[^337]: 一种贝叶斯潜在类别强化学习框架，用于捕捉适应性、反馈驱动的出行行为

    A Bayesian latent class reinforcement learning framework to capture adaptive, feedback-driven travel behaviour

    [https://arxiv.org/abs/2512.14713](https://arxiv.org/abs/2512.14713)

    本文提出了一种基于贝叶斯潜在类别强化学习的框架，能够捕捉出行行为中个体偏好随时间演变的异质性，并通过驾驶模拟数据识别出三种不同适应策略的旅行者类型。

    

    许多出行决策涉及一定程度的经验形成，即个体随时间学习其偏好。同时，个体旅行者之间存在广泛的异质性，无论是在其潜在偏好还是这些偏好的演变方式上。本文提出了一种潜在类别强化学习（LCRL）模型，使分析人员能够捕捉这两种现象。我们将该模型应用于驾驶模拟器数据集，并通过变分贝叶斯方法估计参数。我们识别出三类不同的个体，它们在偏好适应方式上存在显著差异：第一类表现出情境依赖的偏好，并具有特定情境的开发倾向；第二类无论情境如何都遵循持续的开发策略；第三类则采用探索性策略并结合情境特定偏好。

    arXiv:2512.14713v1 Announce Type: cross  Abstract: Many travel decisions involve a degree of experience formation, where individuals learn their preferences over time. At the same time, there is extensive scope for heterogeneity across individual travellers, both in their underlying preferences and in how these evolve. The present paper puts forward a Latent Class Reinforcement Learning (LCRL) model that allows analysts to capture both of these phenomena. We apply the model to a driving simulator dataset and estimate the parameters through Variational Bayes. We identify three distinct classes of individuals that differ markedly in how they adapt their preferences: the first displays context-dependent preferences with context-specific exploitative tendencies; the second follows a persistent exploitative strategy regardless of context; and the third engages in an exploratory strategy combined with context-specific preferences.
    
[^338]: 行列式簇的变分分析

    Variational analysis of determinantal varieties

    [https://arxiv.org/abs/2511.22613](https://arxiv.org/abs/2511.22613)

    本文提出统一的变分分析框架，推导出各类低秩集合（矩阵、张量、对称矩阵、半正定矩阵等）的一阶与二阶切集显式公式及切集相交规则，并据此给出了非光滑问题与其光滑参数化具有等价二阶稳定点的充要条件。

    

    行列式簇——即秩有界的矩阵或张量集合——在低秩优化领域引起了越来越多的关注。低秩集合的切锥已被广泛研究，并支撑了一系列几何方法；而编码曲率信息的二阶几何则更为复杂。在本工作中，我们开发了一个统一框架，用于推导各类低秩集合的一阶与二阶切集的显式公式，包括低秩矩阵、张量、对称矩阵以及半正定矩阵。该框架还适用于低秩集合与另一满足某些假设的集合的交集，从而给出了切集相交规则。通过切集的视角，我们建立了一个充要条件，在该条件下非光滑问题与其光滑参数化具有等价的二阶稳定点。作为应用，我们利用切……（摘要原文在此处被截断）

    arXiv:2511.22613v3 Announce Type: replace-cross  Abstract: Determinantal varieties---the sets of bounded-rank matrices or tensors---have attracted growing interest in low-rank optimization. The tangent cone to low-rank sets is widely studied and underpins a range of geometric methods. The second-order geometry, which encodes curvature information, is more intricate. In this work, we develop a unified framework to derive explicit formulas for both first- and second-order tangent sets to various low-rank sets, including low-rank matrices, tensors, symmetric matrices, and positive semidefinite matrices. The framework also accommodates the intersection of a low-rank set and another set satisfying some assumptions, thereby yielding a tangent intersection rule. Through the lens of tangent sets, we establish a necessary and sufficient condition under which a nonsmooth problem and its smooth parameterization share equivalent second-order stationary points. As applications, we exploit tangent s
    
[^339]: 预测电力使用对公共健康的影响

    Predicting Public Health Impacts of Electricity Usage

    [https://arxiv.org/abs/2511.22031](https://arxiv.org/abs/2511.22031)

    提出了HealthPredictor——一个将电力使用与公共健康结果联系起来的端到端AI模型，通过燃料组合预测、空气质量模拟和货币化健康损害评估三个组件，实现基于健康信息的需求侧电力管理。

    

    电力部门是空气污染物排放的主要来源，影响着几乎每个社区的公共健康。尽管监管措施已经减少了空气污染物，但化石燃料仍然是能源供应的重要组成部分，这凸显了需要更先进的需求侧方法来减少公共健康影响。为了实现基于健康信息的需求侧管理，我们引入了HealthPredictor，这是一个特定领域的AI模型，提供了将电力使用与公共健康结果联系起来的端到端流程。该模型包含三个组件：燃料组合预测器（估计不同发电来源的贡献）、空气质量转换器（模拟污染物排放和大气扩散）以及健康影响评估器（将由此产生的污染物变化转化为货币化的健康损害）。在美国多个地区，我们的健康驱动优化框架……（摘要被截断）

    arXiv:2511.22031v3 Announce Type: replace-cross  Abstract: The electric power sector is a leading source of air pollutant emissions, impacting the public health of nearly every community. Although regulatory measures have reduced air pollutants, fossil fuels remain a significant component of the energy supply, highlighting the need for more advanced demand-side approaches to reduce the public health impacts. To enable health-informed demand-side management, we introduce HealthPredictor, a domain-specific AI model that provides an end-to-end pipeline linking electricity use to public health outcomes. The model comprises three components: a fuel mix predictor that estimates the contribution of different generation sources, an air quality converter that models pollutant emissions and atmospheric dispersion, and a health impact assessor that translates resulting pollutant changes into monetized health damages. Across multiple regions in the United States, our health-driven optimization fra
    
[^340]: 低秩矩阵推断中的稀疏损坏：PCA基准测试

    Sparse corruption in low-rank matrix inference: the PCA benchmark

    [https://arxiv.org/abs/2511.11927](https://arxiv.org/abs/2511.11927)

    该论文研究了稀疏噪声（建模为有限平均连通度的加权图邻接矩阵）损坏下低秩信号的PCA推断问题，利用复制方法与种群动力学解析计算了最大特征值、特征向量分量密度及信号重叠度，并识别出随图连通度变化的两个信号强度转变点。

    

    主成分分析（PCA）是从含噪观测中提取低秩信号的标准工具。众所周知，当将PCA应用于被稠密、均匀噪声损坏的秩一信号时，在大矩阵尺寸极限下会出现BBP转变：离群特征值的涌现与相应特征向量的对齐发生在同一临界信号强度处。本文研究稀疏噪声损坏的情形。噪声矩阵被建模为具有有限平均连通度的加权无向图的邻接矩阵。利用复制方法，我们通过由种群动力学求解的递归分布方程，解析地计算了典型的最大特征值、最大特征向量分量密度以及与信号的重叠度。我们识别出两个作为图连通度函数的信号强度转变：θ_crit，标记通过最大特征向量实现信号恢复……

    arXiv:2511.11927v3 Announce Type: replace-cross  Abstract: Principal Component Analysis (PCA) is a standard tool for extracting a low-rank signal from noisy observations. It is known that when applying PCA to a rank-one signal corrupted by a dense, homogeneous noise, in the large matrix size limit, the BBP transition occurs, where the emergence of an outlying eigenvalue and the alignment of the corresponding eigenvector occur at the same critical signal strength. Here we study the case of sparse noise corruption. The noise matrix is modelled as the adjacency matrix of a weighted undirected graph with finite average connectivity. Using the replica method, we analytically compute the typical top eigenvalue, the top eigenvector component density, and the overlap with the signal, through recursive distributional equations solved by population dynamics. We identify two signal-strength transitions as functions of graph connectivity: $\theta_{\rm crit}$, marking signal recovery by the top eig
    
[^341]: 一种Nesterov加速的拜占庭鲁棒联邦学习

    A Nesterov-Accelerated Byzantine-Robust Federated Learning

    [https://arxiv.org/abs/2511.02657](https://arxiv.org/abs/2511.02657)

    该论文提出了一种将Nesterov动量与拜占庭鲁棒聚合规则无缝结合的联邦学习算法Byrd-NAFL，在非凸光滑损失函数和宽松假设下建立了有限时间收敛保证，同时兼顾通信效率与对拜占庭攻击的抵抗能力。

    

    我们研究了鲁棒联邦学习问题，即一组工作节点在中央服务器的协调下协作训练一个共享模型，同时存在能够进行任意且潜在恶意行为的拜占庭攻击者。为了同时提高通信效率和对这类攻击者的抵御能力，我们提出了一种拜占庭鲁棒的Nesterov加速联邦学习算法。Byrd-NAFL将Nesterov动量与拜占庭鲁棒聚合规则无缝集成到联邦学习过程中，以实现对梯度污染的快速且安全的收敛。我们在非凸光滑损失函数下，并采用对聚合梯度较为宽松的假设，为Byrd-NAFL建立了有限时间收敛保证。大量数值实验验证了Byrd-NAFL的有效性，并证明了其在收敛性能方面相对于现有基准方法的优越性。

    arXiv:2511.02657v2 Announce Type: replace  Abstract: We investigate robust federated learning, where a group of workers collaboratively train a shared model under the orchestration of a central server in the presence of Byzantine adversaries capable of arbitrary and potentially malicious behaviors. To simultaneously enhance communication efficiency and resilience against such adversaries, we propose a Byzantine-resilient Nesterov-accelerated federated learning (Byrd-NAFL) algorithm. Byrd-NAFL seamlessly integrates Nesterov's momentum into the federated learning process alongside Byzantine-resilient aggregation rules to achieve fast and safe convergence against gradient corruption. We establish a finite-time convergence guarantee for Byrd-NAFL under non-convex and smooth loss functions with relaxed assumptions on the aggregated gradients. Extensive numerical experiments validate the effectiveness of Byrd-NAFL and demonstrate the superiority over existing benchmarks in terms of convergen
    
[^342]: AI辅助是保留还是消解分歧？模糊视频标注中预标注的研究

    Does AI Assistance Preserve or Collapse Disagreement? A Study of Pre-Annotations in Ambiguous Video Labeling

    [https://arxiv.org/abs/2510.21798](https://arxiv.org/abs/2510.21798)

    该论文提出一种受控审计协议，揭示了AI预标注在模糊时序视频标注中的双重效应——将标注速度提升约23%、提高标注者一致性与潜空间标准化，但同时可能锚定标注者于模型先验并压制对多元数据集建设有价值的分歧。

    

    AI生成的预标注可以加速视频标注，但也可能使标注者锚定于模型先验，并压制那些对构建多元化数据集有价值的分歧。我们在模糊时序视频标注任务中研究这一权衡问题，在该任务中标注者需要选择事件边界，并赋予依赖于上下文的标签，例如“正常”或“异常”。我们提出了一种受控审计协议，将标注成本、共识一致性、标注者间一致性、时间边界变化、语义标签变化、潜空间标准化以及编辑行为等因素分离开来进行分析。在一项包含18名标注者和180个标注会话的平衡试点研究中，基于CLIP的固定预标注引擎将平均标注时间降低了23.11%；72%的标注者在有辅助时速度更快，每位标注者的中位数时间增益为35%。辅助标注提高了标注者间的一致性和CLIP空间标准化程度，而语义标（原文摘要在此处被截断）

    arXiv:2510.21798v3 Announce Type: replace-cross  Abstract: AI-generated Pre-Annotations can accelerate video labeling, but they may also anchor annotators to model priors and suppress disagreement that is valuable for pluralistic dataset construction. We study this tradeoff in ambiguous temporal video annotation, where annotators choose event boundaries and assign context-dependent labels such as "normal" or "abnormal." We introduce a controlled audit protocol that separates annotation cost, consensus alignment, inter-annotator consistency, temporal-boundary variation, semantic-label variation, latent-space standardization, and edit behavior.   In a counterbalanced pilot study with 18 annotators and 180 annotation sessions, a fixed CLIP-based Pre-Annotation engine reduced mean annotation time by 23.11%; 72% of annotators were faster with assistance, with a median per-annotator gain of 35%. Assistance increased inter-annotator consistency and CLIP-space standardization, while semantic-l
    
[^343]: 适者生存的Cox模型：针对时间-事件数据的枢轴变量选择方法

    Survival of the fittest Cox model: Pivotal variable selection for time-to-event data

    [https://arxiv.org/abs/2510.19374](https://arxiv.org/abs/2510.19374)

    本文通过对Cox比例风险模型的偏似然函数进行平方根变换，使正则化参数选择具有渐近枢轴性（精确消除基线风险和删失率的影响），从而显著提升了生存分析中变量选择的准确性和可靠性。

    

    我们重新审视了Cox比例风险模型，以改进生存分析中的变量选择。通过对偏似然函数进行平方根变换，使得正则化参数的选择在渐近意义下成为枢轴量：基线风险与删失率被精确地抵消，而删失在时间上的分布方式仅通过一个阶为 $1/\log n$ 的有界修正项产生影响。利用鞅残差，我们通过蒙特卡洛方法计算该惩罚水平，而无需对设计矩阵进行重新排序。所得的准则融合了BIC等信息准则与lasso等惩罚回归方法的优点，取两者之长。在模拟数据和真实数据上，我们的方法显著优于日常使用的最先进方法。

    arXiv:2510.19374v3 Announce Type: replace-cross  Abstract: We revisit Cox's proportional hazards model to improve variable selection in survival analysis. A square-root transformation of the partial likelihood renders the selection of the regularization parameter asymptotically pivotal: the baseline hazard and the censoring rate cancel exactly, and the way censoring is spread over time only enters through a bounded correction of order $1/\log n$. Using martingale residuals, we compute this penalty level by Monte Carlo without reordering the design matrix. The resulting criterion borrows from information criteria such as BIC and from penalized regression methods such as the lasso, taking the best of both. On simulated and real data, our method substantially improves upon state-of-the-art approaches used daily.
    
[^344]: 罗宾斯-西格蒙德定理的扩展及其在强化学习中的应用

    Extensions of Robbins-Siegmund Theorem with Applications in Reinforcement Learning

    [https://arxiv.org/abs/2509.26442](https://arxiv.org/abs/2509.26442)

    本文扩展了罗宾斯-西格蒙德定理，将零阶项的条件从可求和放宽为平方可求和，并通过一个新颖温和的增量假设实现了几乎必然收敛到有界集合，同时给出了收敛速率、高概率集中界和 L^p 收敛，从而适用于更广泛的强化学习场景。

    

    罗宾斯-西格蒙德定理确立了近似上鞅随机过程的收敛性，是分析随机逼近和强化学习中随机迭代算法最常用的方法之一。然而，其原始形式存在一个显著的局限性，即要求零阶项可求和。在许多重要的强化学习应用中，这一可求和条件无法得到满足。这一局限性促使我们扩展罗宾斯-西格蒙德定理，使其适用于零阶项不可求和、但仅平方可求和的近似上鞅情形。特别地，我们对随机过程的增量引入了一个新颖且温和的假设，该假设与平方可求和条件相结合，能够实现几乎必然收敛到一个有界集合。此外，我们进一步提供了几乎必然收敛速率、高概率集中界以及 L^p 收敛等结果。

    arXiv:2509.26442v3 Announce Type: replace  Abstract: The Robbins-Siegmund theorem establishes the convergence of stochastic processes that are almost supermartingales and is one of the most commonly used approaches for analyzing stochastic iterative algorithms in stochastic approximation and reinforcement learning (RL). However, its original form has a significant limitation as it requires the zero-order term to be summable. In many important RL applications, this summable condition, however, cannot be met. This limitation motivates us to extend the Robbins-Siegmund theorem for almost supermartingales where the zero-order term is not summable, but only square-summable. In particular, we introduce a novel and mild assumption on the increments of the stochastic processes. This together with the square-summable condition enables an almost sure convergence to a bounded set. Additionally, we further provide almost sure convergence rates, high probability concentration bounds, and $L^p$ conv
    
[^345]: BrainPro：面向大型脑电图模型的过程条件化预训练与拓扑检索方法

    BrainPro: Process-conditioned Pretraining with Topographic Retrieval for Large EEG Models

    [https://arxiv.org/abs/2509.22050](https://arxiv.org/abs/2509.22050)

    BrainPro提出了一种过程条件化的自监督预训练框架，通过拓扑检索将不同数据集的电极导联配置映射到通用通道-区域模板，并同时学习共享表征与过程关联表征，从而解决EEG基础模型中心理过程线索利用不足和导联配置异构两大挑战。

    

    脑电图（EEG）基础模型通常在异构语料库上进行预训练，但存在两个尚未被充分探索的挑战：（1）实验范式提供了心理过程线索，但在掩码预训练中很少被利用；（2）不同数据集的电极导联配置各不相同，导致空间表征依赖于特定的导联配置。现有方法通常只学习单一通用表征，并隐式地对空间结构进行建模，这限制了它们在导联配置异构性条件下利用过程相关信息的能力。我们提出了BrainPro，这是一个过程条件化的自监督预训练框架，它将拓扑检索与共享表征及过程关联表征学习相结合。BrainPro将数据集特定的导联配置映射到通用的通道-区域模板上，并检索通道级和区域级空间滤波器，从而形成一个拓扑对齐的空间基底。在此基底之上，BrainPro学习一个共享编码器以捕捉通用的EEG结构，……

    arXiv:2509.22050v3 Announce Type: replace  Abstract: EEG foundation models are pretrained on heterogeneous corpora, but two challenges remain underexplored: (1) experimental protocols provide mental-process cues that are rarely used in masked pretraining, and (2) electrode montages vary across datasets, making spatial representations montage-dependent. Existing methods typically learn a single general-purpose representation and model spatial structure implicitly, limiting their ability to exploit process-related information under montage heterogeneity. We introduce BrainPro, a process-conditioned self-supervised pretraining framework that couples topographic retrieval with shared and process-associated representation learning. BrainPro maps dataset-specific montages to a universal channel-region template and retrieves channel- and region-level spatial filters to form a topographically aligned spatial basis. Over this basis, BrainPro learns a shared encoder for general EEG structure and
    
[^346]: FlowDrive：基于数据平衡的调节式流匹配轨迹规划方法

    FlowDrive: moderated flow matching with data balancing for trajectory planning

    [https://arxiv.org/abs/2509.21961](https://arxiv.org/abs/2509.21961)

    FlowDrive通过按轨迹模式重加权来平衡长尾驾驶数据，并提出带有环内微小扰动引导的调节式流匹配规划器，在保持场景一致性的同时增加轨迹多样性，在nuPlan和interPlan基准上取得了学习型规划器中的最先进结果。

    

    基于学习的规划器对驾驶数据的长尾分布非常敏感。常见机动操作在数据集中占主导地位，而危险或罕见场景则较为稀疏。这种不平衡会使模型偏向频繁出现的情况，降低其在关键场景上的性能。为解决这一问题，我们比较了多种训练数据采样平衡策略，发现按轨迹模式进行重加权是一种有效的方法。随后我们提出了FlowDrive，这是一种流匹配轨迹规划器，它学习条件整流流，能够以少量的流匹配步骤将噪声直接映射到轨迹分布。我们进一步引入了调节式环内引导，在流步骤之间注入微小扰动，在保持场景一致性的同时系统地增加轨迹多样性。在nuPlan和专注于交互的interPlan基准测试中，FlowDrive在学习型规划器中取得了最先进的成果，并接近……

    arXiv:2509.21961v4 Announce Type: replace-cross  Abstract: Learning-based planners are sensitive to the long-tailed distribution of driving data. Common maneuvers dominate datasets, while dangerous or rare scenarios are sparse. This imbalance can bias models toward the frequent cases and degrade performance on critical scenarios. To tackle this problem, we compare balancing strategies for sampling training data and find reweighting by trajectory pattern an effective approach. We then present FlowDrive, a flow-matching trajectory planner that learns a conditional rectified flow to map noise directly to trajectory distributions with few flow-matching steps. We further introduce moderated, in-the-loop guidance that injects small perturbation between flow steps to systematically increase trajectory diversity while remaining scene-consistent. On nuPlan and the interaction-focused interPlan benchmarks, FlowDrive achieves state-of-the-art results among learning-based planners and approaches m
    
[^347]: 解构眼压：一种无创的多阶段概率逆问题框架

    Deconstructing Intraocular Pressure: A Non-Invasive Multi-Stage Probabilistic Inverse Framework

    [https://arxiv.org/abs/2509.14167](https://arxiv.org/abs/2509.14167)

    该论文提出一种机制分层的两阶段物理引导逆问题框架（MHIP），将全局不可辨识的眼压逆问题分解为条件可辨识的子问题，并结合物理校准数据缩放（PCDS），从而在缺乏体内真值的条件下实现从临床数据中对眼压相关内在参数的无创概率估计。

    

    我们提出了面向生物输运系统的机制分层逆问题（MHIP）范式，其中宏观观测量源于微观尺度内在物理与潜在宏观尺度几何之间的乘性纠缠。在缺乏体内真值的情况下，从临床数据求解MHIP是严重欠定的：标准统计采样会沿不可辨识流形退化，而整体式机器学习基线则会坍缩为不可解释的条件均值。我们提出一种两阶段的物理引导分解方法，将全局不可辨识的逆映射转化为条件可辨识的子问题。第一阶段是一个受约束的逆代理模型，在固定参考几何下分离出内在参数；第二阶段则基于人群水平的临床数据进行经验结构校准。我们还引入了物理校准数据缩放（PCDS），

    arXiv:2509.14167v2 Announce Type: replace  Abstract: We propose the mechanistically hierarchical inverse problem (MHIP) paradigm for biological transport systems, where a macroscopic observation emerges from the multiplicative entanglement of intrinsic micro-scale physics and latent macro-scale geometry. Without in vivo ground truth, solving MHIPs from clinical data is severely underdetermined. Standard statistical sampling degenerates along non-identifiable manifolds, while monolithic machine learning baselines collapse to an uninterpretable conditional mean. We propose a two-stage physics-guided decomposition that turns a globally non-identifiable inverse mapping into conditionally identifiable sub-problems. The first stage is a constrained inverse surrogate that isolates the intrinsic parameter under a fixed reference geometry. The second stage performs empirical structural calibration against population-level clinical data. We also introduce Physics-Calibrated Data Scaling (PCDS), 
    
[^348]: 去中心化梯度市场中鲁棒聚合的基准测试

    Benchmarking Robust Aggregation in Decentralized Gradient Marketplaces

    [https://arxiv.org/abs/2509.05833](https://arxiv.org/abs/2509.05833)

    本文形式化了梯度数据市场（GDM）这一新兴范式，指出市场不仅需要模型层面的鲁棒性，还必须确保高质量数据被正确识别和奖励，并针对此提出了鲁棒聚合方法的基准测试。

    

    无法集中化或出售原始数据的机构越来越多地选择交换模型更新：跨机构联邦学习已在医院网络和制药联盟中投入实际生产。一个不断发展的研究方向瞄准了下一步：将梯度更新作为付费的、特定查询的数据产品进行交换的市场，尽管目前尚无大规模部署。我们将这一新兴范式形式化为梯度数据市场（GDM），其中数据消费者发布学习任务，分布式参与者贡献针对特定查询的梯度更新。联邦学习关心的是模型能否在攻击下存活；而市场还必须关心谁获得了报酬。现有协议仅确保模型层面的鲁棒性，却未能捕捉高质量数据是否被正确识别并得到奖励。在GDM中，选择机制决定支付，这会激励参与者为“被选中”而非真正的数据效用进行优化。我们引入了一个……（摘要在此处被截断）

    arXiv:2509.05833v2 Announce Type: replace  Abstract: Institutions that cannot centralize or sell raw data increasingly exchange model updates instead: cross-silo federated learning is already in production among hospital networks and pharmaceutical consortia. A growing line of research targets the next step: marketplaces in which gradient updates are exchanged as paid, query-specific data products, though none is yet deployed at scale. We formalize this emerging paradigm as a Gradient Data Marketplace (GDM), in which data consumers issue learning tasks and distributed participants contribute query-specific gradient updates. Federated learning asks whether the model survives an attack; a marketplace must also ask who got paid. Existing protocols ensure model-level robustness but fail to capture whether high-quality data is correctly identified and rewarded. In a GDM, selection determines payment, incentivizing participants to optimize for inclusion rather than utility. We introduce a mo
    
[^349]: SCAR：面向车载服务基于AI的无线资源管理的状态空间压缩

    SCAR: State-Space Compression for AI-Based Radio Resource Management of Vehicular Services

    [https://arxiv.org/abs/2508.06243](https://arxiv.org/abs/2508.06243)

    SCAR框架利用聚类和径向基函数网络等机器学习压缩技术，将高维CQI状态信息压缩为紧凑表示，从而提升动态车载环境中基于AI的无线资源管理的可扩展性。

    

    车载联网服务需求的不断增长给基于AI的网络与服务管理带来了重大挑战，原因在于网络状态信息体量大且变化迅速。在动态车载环境中，传统的管理与控制机制在处理信道质量指示（CQI）等细粒度指标时难以扩展。为应对这一挑战，我们提出了SCAR（面向基于AI的无线资源管理的状态空间压缩），这是一个边缘辅助框架，可将高维CQI信息抽象为紧凑的表示形式，供基于AI的管理使用。SCAR采用基于机器学习（ML）的压缩技术，包括聚类和径向基函数（RBF）网络，在保留与管理决策相关的基本特征的同时，降低CQI派生状态信息的维度。所得到的压缩状态被用于训练强化学习

    arXiv:2508.06243v3 Announce Type: replace  Abstract: The increasing demand for connected vehicular services poses significant challenges for AI-based network and service management due to the high volume and rapid variability of network state information. Traditional management and control mechanisms struggle to scale when processing fine-grained metrics such as Channel Quality Indicators (CQIs) in dynamic vehicular environments. To address this challenge, we propose SCAR (State-Space Compression for AI-Based Radio Resource Management), an edge-assisted framework that abstracts high-dimensional CQI information into compact representations for AI-based management. SCAR employs machine-learning (ML)-based compression techniques, including clustering and radial basis function (RBF) networks, to reduce the dimensionality of CQI-derived state information while preserving essential features relevant to management decisions. The resulting compressed states are used to train reinforcement lear
    
[^350]: TabPFN：一个模型统御一切？

    TabPFN: One Model to Rule Them All?

    [https://arxiv.org/abs/2505.20003](https://arxiv.org/abs/2505.20003)

    本文从统计学视角将TabPFN阐释为一种近似贝叶斯推断方法，并证明这一表格数据基础模型在半监督参数估计、协变量偏移预测和异质性处理效应等任务上有时能超越专门的先进方法。

    

    Hollmann等人（Nature 637 (2025) 319-326）最近提出了TabPFN，这是一种基于Transformer的深度学习模型，用于表格数据的回归与分类任务。他们声称该模型“在样本量不超过10,000的数据集上以显著优势超越所有先前的方法，且训练时间大幅减少”。此外，他们将TabPFN称为表格数据的“基础模型”，因为它能够支持“数据生成、密度估计、学习可复用的嵌入以及微调”。在本文中，我们面向统计学界对TabPFN的工作原理进行了针对性的阐释，重点强调其可理解为近似贝叶斯推断。随后，我们探讨了TabPFN对统计学领域的意义：我们展示了开箱即用的TabPFN在某些情况下能够超越专为半监督参数估计、协变量偏移下的预测以及异质性处理效应估计而设计的最先进方法。

    arXiv:2505.20003v3 Announce Type: replace  Abstract: Hollmann et al. (Nature 637 (2025) 319-326) recently introduced TabPFN, a transformer-based deep learning model for regression and classification on tabular data, which they claim "outperforms all previous methods on datasets with up to 10,000 samples by a wide margin, using substantially less training time." Furthermore, they have called TabPFN a "foundation model" for tabular data, as it can support "data generation, density estimation, learning reusable embeddings and fine-tuning". In this paper, we provide a tailored explanation of how TabPFN works for a statistics audience, by emphasizing its interpretation as approximate Bayesian inference. We then explore the significance of TabPFN to the field of statistics: We show that an out-of-the-box application of TabPFN can sometimes outperform specialized state-of-the-art methods for semi-supervised parameter estimation, prediction under covariate shift, and heterogeneous treatment ef
    
[^351]: 基于局部搜索MCMC层的学习

    Learning with Local Search MCMC Layers

    [https://arxiv.org/abs/2505.14240](https://arxiv.org/abs/2505.14240)

    受模拟退火与Metropolis-Hastings联系的启发，论文将局部搜索启发式算法的邻域系统转化为提议分布，构建出可微的随机MCMC组合层及对应损失函数，为使用不精确求解器的组合优化层学习提供了理论基础，并显著降低了计算成本。

    

    将组合优化层集成到神经网络中近来引起了广泛的研究兴趣。然而，许多现有方法缺乏理论保证，或者在使用不精确求解器时表现不佳。这是一个关键的限制，因为许多运筹学问题是NP难的，通常需要使用基于邻域的局部搜索启发式算法。在本文中，我们提出了一种使用此类不精确求解器进行学习的原则性方法。受模拟退火与Metropolis-Hastings之间联系的启发，我们将局部搜索启发式算法中针对具体问题的邻域系统转化为提议分布，从而在可行解集合上实现MCMC（马尔可夫链蒙特卡洛）。这使我们能够构建可微的、随机性的组合层以及相应的损失函数。用局部搜索替代精确求解器，极大地降低了学习的计算负担。

    arXiv:2505.14240v4 Announce Type: replace  Abstract: Integrating combinatorial optimization layers into neural networks has recently attracted significant research interest. However, many existing approaches lack theoretical guarantees or fail to perform adequately when relying on inexact solvers. This is a critical limitation, as many operations research problems are NP-hard, often necessitating the use of neighborhood-based local search heuristics. In this paper, we introduce a principled approach for learning with such inexact solvers. Inspired by the connection between simulated annealing and Metropolis-Hastings, we transform the problem specific neighborhood systems used in local search heuristics into proposal distributions, implementing MCMC on the set of feasible solutions. This allows us to construct differentiable, stochastic combinatorial layers and associated loss functions. Replacing an exact solver with a local search strongly reduces the computational burden of learning 
    
[^352]: 基于大语言模型的物联网生态系统威胁检测与防御框架

    LLM-Based Threat Detection and Prevention Framework for IoT Ecosystems

    [https://arxiv.org/abs/2505.00240](https://arxiv.org/abs/2505.00240)

    该论文提出了一种基于大语言模型的物联网威胁检测与防御框架，通过在 IoT-23 和 TON_IoT 数据集上微调的检测模型识别异常网络行为，并结合自适应防御策略生成组件，在边缘-云实验环境中实现了 99.75% 的威胁检测准确率和 98.3% 的防御准确率。

    

    物联网（IoT）的快速扩张带来了复杂且不断演变的网络安全挑战。本文提出了一种基于大语言模型（LLM）的物联网网络智能威胁检测与防御框架。所提出的架构集成了两个核心组件：检测组件利用在物联网专用数据集（IoT-23 和 TON_IoT）上微调的模型来识别异常网络行为，防御组件则生成自适应响应策略以缓解检测到的威胁。该框架包含提示管理、局部适应和全局优化模块，以支持上下文感知的决策制定和持续的模型引导适应。在基于 Docker 的边缘-云环境中进行的实验验证表明，BERT Small 在威胁检测中达到了 99.75% 的准确率、F1 分数、精确率和召回率，而防御 LLM 达到了 98.3% 的准确率。

    arXiv:2505.00240v3 Announce Type: replace-cross  Abstract: The rapid expansion of the Internet of Things (IoT) has introduced complex and evolving cybersecurity challenges. This paper presents a Large Language Model (LLM)-based framework for intelligent threat detection and prevention in IoT networks. The proposed architecture integrates two core components: a Detection Component that leverages fine-tuned models trained on IoT-specific datasets (IoT-23 and TON_IoT) to identify anomalous network behaviours, and a Prevention Component that generates adaptive response strategies to mitigate detected threats. The framework incorporates prompt management, local adaptation, and global refinement modules to support context-aware decision-making and continuous model-guided adaptation. Experimental validation in a Docker-based edge-cloud environment shows that BERT Small achieves 99.75% accuracy, F1-score, precision, and recall for threat detection, while the Prevention LLM achieves 98.3% accur
    
[^353]: CP-Guard+：协同感知中恶意代理检测与防御的新范式

    CP-Guard+: A New Paradigm for Malicious Agent Detection and Defense in Collaborative Perception

    [https://arxiv.org/abs/2502.07807](https://arxiv.org/abs/2502.07807)

    该论文提出了一种无需验证最终感知结果、直接在特征层面检测恶意代理的新范式，并发布了首个协同感知恶意代理检测基准数据集CP-GuardBench及鲁棒防御方法CP-Guard+，在大幅降低计算开销的同时有效防御协同感知中的恶意攻击。

    

    协同感知（CP）是实现安全网联自动驾驶的一种有前景的方法，它使多辆车辆能够共享感知信息以提升感知性能。然而，与单车感知相比，协同感知系统的开放性使其更容易受到恶意攻击，攻击者可以注入恶意信息来误导自车的感知，从而给安全驾驶带来严重风险。为了缓解这种脆弱性，我们首先提出了一种新的恶意代理检测范式，该范式能够在特征层面有效识别恶意代理，而无需验证最终感知结果，从而显著降低计算开销。基于这一范式，我们推出了CP-GuardBench，这是首个用于训练和评估协同感知系统中各类恶意代理检测方法的综合性数据集。此外，我们还开发了一种名为CP-Guard+的鲁棒防御方法……（摘要原文在此处截断）

    arXiv:2502.07807v2 Announce Type: replace-cross  Abstract: Collaborative perception (CP) is a promising method for safe connected and autonomous driving, which enables multiple vehicles to share sensing information to enhance perception performance. However, compared with single-vehicle perception, the openness of a CP system makes it more vulnerable to malicious attacks that can inject malicious information to mislead the perception of an ego vehicle, resulting in severe risks for safe driving. To mitigate such vulnerability, we first propose a new paradigm for malicious agent detection that effectively identifies malicious agents at the feature level without requiring verification of final perception results, significantly reducing computational overhead. Building on this paradigm, we introduce CP-GuardBench, the first comprehensive dataset provided to train and evaluate various malicious agent detection methods for CP systems. Furthermore, we develop a robust defense method called C
    
[^354]: 马尔可夫噪声下随机逼近与强化学习的几乎必然收敛速率与集中性

    Almost Sure Convergence Rates and Concentration of Stochastic Approximation and Reinforcement Learning with Markovian Noise

    [https://arxiv.org/abs/2411.13711](https://arxiv.org/abs/2411.13711)

    本文通过一种使用长度递减区间的平均ODE新颖离散化技术，首次为带马尔可夫噪声的压缩随机逼近算法建立了几乎必然收敛速率和亚Weibull尾部的最大集中性界，并首次应用于无基于计数学习率的Q学习及离策略时序差分学习。

    

    据我们所知，本文首次为具有马尔可夫噪声的一般压缩随机逼近算法建立了几乎必然收敛速率以及具有亚Weibull尾部的最大集中性界。作为推论，我们还得到了 $L^p$ 中的收敛速率。我们成功的关键在于对随机逼近算法的平均常微分方程（ODE）采用了一种新颖的离散化方法，即使用长度递减（而非恒定）的区间。在应用方面，我们首次得到了无需基于计数的 learning rates 的 $Q$-学习的几乎必然收敛速率，以及离策略时序差分学习（off-policy temporal difference learning）的首个最大集中性界，两者均基于马尔可夫样本。

    arXiv:2411.13711v2 Announce Type: replace  Abstract: This paper establishes, to our knowledge, the first almost sure convergence rate and the first maximal concentration bound with sub-Weibull tails for general contractive stochastic approximation algorithms with Markovian noise. As a corollary, we also obtain convergence rates in $L^p$. Key to our successes is a novel discretization of the mean ODE of stochastic approximation algorithms using intervals with diminishing (instead of constant) length. Applications yield the first almost sure convergence rate for $Q$-learning without count-based learning rates and the first maximal concentration bound for off-policy temporal difference learning, both with Markovian samples.
    
[^355]: FlexTSF：一种面向具有可变规律性的时间序列的灵活预测模型

    FlexTSF: A Flexible Forecasting Model for Time Series with Variable Regularities

    [https://arxiv.org/abs/2410.23160](https://arxiv.org/abs/2410.23160)

    FlexTSF 提出了一种灵活的领域自适应时间序列预测模型，其核心的 IVP Patcher 连续时间分块模块基于初值问题，能够原生处理不均匀采样间隔、可变序列长度和缺失值，从而有效应对不规则时间序列的通用预测挑战。

    

    对于通用时间序列预测模型而言，在具有不规则时间结构的时间序列上进行泛化是一个长期的挑战。现有方法通常假设采样是规律的，或严重依赖插补技术，这限制了它们在现实场景中的适用性——由于多样化的传感设备和记录方式，不规则性在实际场景中普遍存在。为了填补这一空白，我们提出了 FlexTSF，这是一种灵活且领域自适应的预测模型，专为具有可变时间规律性的时间序列而设计。其基础是 IVP Patcher，这是一个利用初值问题（IVPs）的连续时间分块模块，能够从本质上处理不均匀的采样间隔、可变的序列长度以及缺失值。FlexTSF 构建于仅解码器的 Transformer 架构之上，通过专门的因果自注意力机制整合了归一化的时间戳输入和特定领域的统计描述符（注：原文摘要到此被截断）。

    arXiv:2410.23160v3 Announce Type: replace-cross  Abstract: Generalizing across time series with irregular temporal structures poses a persistent challenge for universal time series forecasting models. Existing approaches often assume regular sampling or depend heavily on imputation, which limits their applicability in real-world scenarios where irregularities are prevalent due to diverse sensing devices and recording practices. To address this gap, we propose FlexTSF, a flexible and domain-adaptive forecasting model specifically designed for time series with variable temporal regularities. At its foundation lies the IVP Patcher, a continuous-time patching module leveraging Initial Value Problems (IVPs) to inherently handle uneven sampling intervals, variable sequence lengths, and missing values. Built upon a decoder-only Transformer architecture, FlexTSF integrates normalized timestamp inputs and domain-specific statistical descriptors through a specialized causal self-attention mechan
    
[^356]: 设计算法推荐以实现人机互补

    Designing Algorithmic Recommendations to Achieve Human-AI Complementarity

    [https://arxiv.org/abs/2405.01484](https://arxiv.org/abs/2405.01484)

    本文提出了一种基于因果推断潜在结果框架的算法设计方法，将算法推荐与工具变量相联系，无需对推荐效果做先验假设即可设计出实现人机互补的推荐算法。

    

    arXiv:2405.01484v3 公告类型：replace-cross 摘要：算法经常是辅助而非取代人类决策者。然而，算法的设计与分析通常聚焦于预测结果，并未明确建模其对人类决策的影响。鉴于实证证据一再表明算法辅助工具无法改善人类决策，算法助手的设计与其角色之间的这种差异尤其令人担忧。在本文中，我们对推荐算法的设计进行了形式化，使其能够辅助人类决策者，而无需对推荐如何影响决策做出限制性的先验假设。我们构建了一个算法设计问题，利用因果推断中的潜在结果框架来建模推荐对人类决策者二元处理选择的影响。在此模型中，我们建立了算法推荐与工具变量之间的联系。

    arXiv:2405.01484v3 Announce Type: replace-cross  Abstract: Algorithms frequently assist, rather than replace, human decision-makers. However, the design and analysis of algorithms often focus on predicting outcomes and do not explicitly model their effect on human decisions. This discrepancy between the design and role of algorithmic assistants becomes particularly concerning in light of empirical evidence that suggests that algorithmic assistants again and again fail to improve human decisions. In this article, we formalize the design of recommendation algorithms that assist human decision-makers without making restrictive ex-ante assumptions about how recommendations affect decisions. We formulate an algorithmic-design problem that leverages the potential-outcomes framework from causal inference to model the effect of recommendations on a human decision-maker's binary treatment choice. Within this model, we make a connection between algorithmic recommendations and instrumental variab
    
[^357]: 面向自动驾驶的深度强化学习安全性：综述

    Security of Deep Reinforcement Learning for Autonomous Driving: A Survey

    [https://arxiv.org/abs/2212.06123](https://arxiv.org/abs/2212.06123)

    本文对86项强化学习安全研究进行了全面综述，根据威胁模型和单/多智能体设置系统分类了攻击与防御方法，并评估了这些机制在自动驾驶场景中的适用性。

    

    强化学习（RL）使智能体能够通过与环境的交互学习最优行为，并已被越来越多地部署于包括自动驾驶在内的安全关键型应用中。尽管前景广阔，强化学习仍易受到两类攻击：一类旨在破坏策略学习过程，另一类则旨在诱使训练后的智能体做出错误决策。尽管强化学习安全领域的文献迅速增长，且已有多篇综述问世，但现有的分类方法往往难以指导针对特定系统选择合适的防御措施。在这项工作中，我们对86项近期的强化学习安全研究进行了全面综述，通过依据明确的威胁模型以及单智能体与多智能体设置对攻击和防御进行系统化分类，弥补了上述不足。此外，我们还考察了最先进的攻击与防御机制在自动驾驶场景下的相关性与适用性。

    arXiv:2212.06123v4 Announce Type: replace  Abstract: Reinforcement learning (RL) enables agents to learn optimal behaviors through interaction with their environment and has been increasingly deployed in safety-critical applications, including autonomous driving. Despite its promise, RL is susceptible to attacks designed either to compromise policy learning or to induce erroneous decisions by trained agents. Although the literature on RL security has grown rapidly and several surveys exist, existing categorizations often fall short in guiding the selection of appropriate defenses for specific systems. In this work, we present a comprehensive survey of 86 recent studies on RL security, addressing these limitations by systematically categorizing attacks and defenses according to defined threat models and single- versus multi-agent settings. Furthermore, we examine the relevance and applicability of state-of-the-art attacks and defense mechanisms within the context of autonomous driving, 
    
[^358]: 预测-优化框架中的主动学习：一种基于间隔的方法

    Active Learning in the Predict-then-Optimize Framework: A Margin-Based Approach. (arXiv:2305.06584v1 [cs.LG])

    [http://arxiv.org/abs/2305.06584](http://arxiv.org/abs/2305.06584)

    本文提出了预测-优化框架中的首个主动学习方法，通过基于间隔的准则和SPO损失的可求解替代函数来决定是否请求样本标签，在提供理论泛化保证的同时有效降低标签复杂度。

    

    我们在预测-优化框架中开发了首个主动学习方法。具体而言，我们开发了一种学习方法，该方法从无标签数据流中依次决定是否请求特征样本的“标签”，其中标签对应于用于决策优化的模型参数。我们的主动学习方法是首个由预测参数所导致的决策误差（即智能预测-优化损失，SPO损失）直接驱动的方法。受SPO损失结构的启发，我们的算法采用基于间隔的准则，利用退化距离的概念，并在收集的数据上最小化SPO损失的一个易于求解的替代函数。特别地，我们开发了一种高效的主动学习算法，包含硬拒绝和软拒绝两种变体，每种变体都具有理论上的超额风险（即泛化）保证。我们进一步推导了标签复杂度（即请求标签的样本数量）的界限。

    We develop the first active learning method in the predict-then-optimize framework. Specifically, we develop a learning method that sequentially decides whether to request the "labels" of feature samples from an unlabeled data stream, where the labels correspond to the parameters of an optimization model for decision-making. Our active learning method is the first to be directly informed by the decision error induced by the predicted parameters, which is referred to as the Smart Predict-then-Optimize (SPO) loss. Motivated by the structure of the SPO loss, our algorithm adopts a margin-based criterion utilizing the concept of distance to degeneracy and minimizes a tractable surrogate of the SPO loss on the collected data. In particular, we develop an efficient active learning algorithm with both hard and soft rejection variants, each with theoretical excess risk (i.e., generalization) guarantees. We further derive bounds on the label complexity, which refers to the number of samples who
    
[^359]: 矩阵计算的方差缩减及其在高斯过程中的应用

    Variance Reduction for Matrix Computations with Applications to Gaussian Processes. (arXiv:2106.14565v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2106.14565](http://arxiv.org/abs/2106.14565)

    本文提出了一种通过矩阵分解来进行矩阵计算的方差缩减方法，并展示了在某些问题上可以实现任意更好的随机性能。此外，本文提出的矩阵乘积迹的分解估计器与对半正定矩阵行列式的对数估计器均有显著的效率提高，可以在实际的贝叶斯优化问题中得到应用。

    

    随着计算速度和内存容量的不断提高，方法学上的进步也为随机模拟的性能带来了显著提高。本文聚焦于通过矩阵分解来进行矩阵计算的方差缩减。我们提供了现有方差缩减方法在估计大型矩阵元素时未能利用矩阵分解所带来的方差缩减，而如何通过计算矩阵的平方根分解，在某些情况下能够实现任意更好的随机性能。此外，我们提出了一种矩阵乘积迹的分解估计器，并在数值上证明，在某些高斯过程对数似然函数估计的问题上，该估计器的效率可以提高1000倍。此外，我们提供了一种对半正定矩阵行列式的对数估计器，其方差比文献中使用的标准估计器快速衰减。我们在实际的贝叶斯优化问题中说明了我们提出的方法的实用性。

    In addition to recent developments in computing speed and memory, methodological advances have contributed to significant gains in the performance of stochastic simulation. In this paper, we focus on variance reduction for matrix computations via matrix factorization. We provide insights into existing variance reduction methods for estimating the entries of large matrices. Popular methods do not exploit the reduction in variance that is possible when the matrix is factorized. We show how computing the square root factorization of the matrix can achieve in some important cases arbitrarily better stochastic performance. In addition, we propose a factorized estimator for the trace of a product of matrices and numerically demonstrate that the estimator can be up to 1,000 times more efficient on certain problems of estimating the log-likelihood of a Gaussian process. Additionally, we provide a new estimator of the log-determinant of a positive semi-definite matrix where the log-determinant 
    

