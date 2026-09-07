# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [UniMate: One Unified Model to Animate Diverse Skeletons](https://arxiv.org/abs/2609.05415) | UniMate提出一种拓扑感知的扩散Transformer，通过图感知注意力偏置、谱旋转位置编码和全局拓扑条件器三种机制将骨骼拓扑融入注意力计算，实现用单一基础模型根据文本提示为任意拓扑结构的骨架生成关节运动，无需针对每个骨架微调或测试时优化。 |
| [^2] | [RegionFed: Federated Learning for Personalized Query Understanding in Heterogeneous Retail Environments](https://arxiv.org/abs/2609.05403) | RegionFed提出了一种完全在梯度层面运作的架构鲁棒联邦学习框架，以区域与全局梯度间的ℓ2冲突作为统一信号来诊断数据异构性并为各区域选择成本最低的充分个性化策略，解决了现有参数级个性化联邦学习方法在现代Transformer上灾难性崩溃的问题。 |
| [^3] | [Distill Globally, Adapt Locally: Reasoning Distillation and Product-Type Test-Time Training for Scalable Trade-Up Recommendation](https://arxiv.org/abs/2609.05363) | 该论文提出一个两级框架，通过检索增强LLM教师将换新推理蒸馏到1550万参数的轻量级嵌入对分类器中，并结合产品类型的测试时训练进行局部适配，使推理时仅需预计算嵌入即可完成数亿产品对的可扩展换新推荐。 |
| [^4] | [Variational Continuation for Double Pendulum Periodic Orbits](https://arxiv.org/abs/2609.05337) | 该论文提出了一种基于Hessian矩阵与自动微分的变分延拓方法，无需积分器和手工推导雅可比矩阵即可高效、有引导地数值延拓动力系统中的周期轨道，并通过双摆周期振荡从不动点出发的完整延拓演示了沿轨道族的分岔与轨道族交叉检测能力。 |
| [^5] | [Lightweight Vision Transformer Compression for On-Device Plant Disease Detection in Resource-Constrained Agricultural Field Conditions](https://arxiv.org/abs/2609.05334) | 该论文提出一种统一的视觉Transformer压缩框架，结合Hessian平衡自适应块剪枝、量化与基于注意力的知识蒸馏，实现资源受限农业设备端的高效辣椒病害检测。 |
| [^6] | [Embedded Graph Flows for Categorical Graph Generation](https://arxiv.org/abs/2609.05328) | 提出嵌入式图流（EGF）生成模型，通过为节点和无序边类别学习连续嵌入，并利用置换等变图变换器将高斯噪声输运到这些端点，从而摆脱独热编码的人为几何限制，在QM9等分子图生成基准上显著超越DiGress和GruM等基线方法（FCD低至0.150）。 |
| [^7] | [Adaptive Gated Deepfake Detection for Low-Resolution and Resource-Constrained Environments](https://arxiv.org/abs/2609.05320) | 本文提出 AdaGate-DF 自适应门控深伪检测框架，利用图像质量线索通过双多出口系统路由样本，让高质量图像提前退出以节省算力，在低分辨率和资源受限环境下实现了优于现有方法的高效检测。 |
| [^8] | [Optimal Rates for Agentic Networked Information Aggregation](https://arxiv.org/abs/2609.05318) | 本文基于 Kearns 等人（SODA'26）的开创性工作，研究了智能体在网络中各自只能看到部分数据、只传递自身结论的信息聚合学习模型，并证明了此类模型下最后一个智能体超额均方误差的最优收敛速率。 |
| [^9] | [How Does mHC Use Its Residual Streams? Selective Routing and Near-Identity Mixing](https://arxiv.org/abs/2609.05309) | 本文通过分析 DeepSeek-V4-Flash 的四流残差通路，发现 mHC 模型采用集中且随深度变化的选择性读写路由（每个模块约有效使用两个流），残差混合主要限于早期层、深层几乎恒等地传递各流，且干预实验证明各流承载着方向上不同、功能上重要的表示。 |
| [^10] | [Online Change-point Detection for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2609.05298) | 提出了一种轻量级、与算法无关的在线变点检测方法PPR，通过对智能体回报流进行平滑处理并应用统计漂移检测，使合作式多智能体强化学习系统能够在环境或任务目标发生变化时及时识别情况改变。 |
| [^11] | [LexFlip: A Dissociation Diagnostic for Legal Meaning Preservation Metrics](https://arxiv.org/abs/2609.05296) | 本文提出LexFlip诊断集，通过373个保持词汇表面形式不变却逆转法律效力的魁北克法语法条最小扰动，揭示了现有嵌入类语义度量几乎无法察觉法律含义变化，暴露了当前法律文本简化评估中“相同对检验”的根本缺陷。 |
| [^12] | [Learning from VAE Errors to support ECG-based Differential Diagnosis of Myocardial Scar](https://arxiv.org/abs/2609.05294) | 本研究提出利用β-VAE的DTW重构误差来辅助基于心电图的LGE心肌瘢痕鉴别诊断，发现重构误差在12个导联中的10个导联上能显著区分瘢痕阳性与阴性患者，为常规心电图筛查心肌瘢痕提供了新思路。 |
| [^13] | [How to Speculate about Uncertainty in Agentic Coding? A Draft-Model Gate Method](https://arxiv.org/abs/2609.05274) | 提出推测不确定性（SU）方法，利用小型开源草稿模型在单次前向传播中对黑盒智能体已生成的输出轨迹进行评分，无需访问模型内部信息即可预测失败，并通过执行前否决门控等下游策略降低智能体编程的失败成本。 |
| [^14] | [Shallow neural network approximation in mixed Sobolev spaces](https://arxiv.org/abs/2609.05263) | 该论文建立了与激活函数无关的傅里叶块原理，证明浅层神经网络对混合光滑度为 $\alpha$ 的函数的逼近代数阶为 $\min\{\alpha,\rho\}$，并通过匹配的下界确定了 $\mathrm{ReLU}^k$ 网络在任意维数下的最优代数逼近指数为 $\min\{\alpha,k+1\}$。 |
| [^15] | [GLASS: Graph-Language Alignment with Spherical Scoring for Transferable Graph-Level Anomaly Detection](https://arxiv.org/abs/2609.05253) | GLASS框架通过在单位超球面上对齐图编码器与文本嵌入，并利用von Mises-Fisher核密度估计进行球面多模态评分，实现了具有跨域可迁移性的图级异常检测。 |
| [^16] | [Proton Irradiation Characterization of an Open-Source ML Accelerator on a Zynq UltraScale+ MPSoC](https://arxiv.org/abs/2609.05249) | 本论文通过对部署在 Zynq UltraScale+ SoC 上执行 ResNet-20 推理的开源 Tensil 神经网络加速器进行 20-58 MeV 质子辐照实验，首次为其建立了系统级辐射响应基线，填补了开源 RTL 可访问加速器在实证辐射表征方面的空白，为开发可验证的空间辐射缓解策略奠定了基础。 |
| [^17] | [PRICE: A Systematic Study of LLM Adaptation Choices for Bitcoin Price Forecasting](https://arxiv.org/abs/2609.05235) | 该研究提出PRICE框架，通过联合优化LoRA微调、递归多步推理、整数舍入数值表示、CTF提示和零温度解码五项适配选择，系统性地将4位量化的LLaMA-3 8B大语言模型高效适配于比特币短期价格预测任务。 |
| [^18] | [Hessian-based molecular conformation augmentation for a scalable and efficient strategy of machine learning interatomic potentials](https://arxiv.org/abs/2609.05233) | 该论文提出两种基于Hessian矩阵的数据增强方案（UniAug和ModeAug），通过简单的泰勒展开生成增强分子构型，无需修改模型架构或引入额外计算与内存开销，即可有效利用Hessian信息训练机器学习原子间势。 |
| [^19] | [FedDRAW: Federated Dual Reputation Annealing Weighting for Heterogeneous Multi-Institutional Chest Radiograph Classification](https://arxiv.org/abs/2609.05223) | 提出FedDRAW服务器端聚合方法，通过双声誉退火加权机制，解决联邦平均按样本量加权导致小而信息丰富的医院被边缘化、大型客户端可能主导全局模型的问题，用于异构多机构胸片分类。 |
| [^20] | [A Verifier-Guided Explainable Reasoning Framework with Gold-Anchored QLoRA, Task-Aware Mixture-of-Experts, and Group-Relative RLVR](https://arxiv.org/abs/2609.05221) | 该论文提出一种面向教育问答的可解释推理框架，通过黄金锚定QLoRA微调、任务感知符号路由（逻辑问题交由FOL/Z3验证器、物理问题交由公式与单位感知求解器）以及组相对RLVR强化学习，从正确性、一致性和推理深度三个维度提升大模型推理的可验证性与可解释性。 |
| [^21] | [Dimension-Adaptive Batched Lipschitz Narrowing Without Knowing the Zooming Dimension](https://arxiv.org/abs/2609.05214) | 本文提出计数自适应BLiN算法，通过依据前一轮淘汰后存活的立方体数量选取下一条边长，去除了A-BLiN对缩放维数 $d_z$ 的依赖，在未知 $d_z$ 的情况下仍以 $\mathcal O_d(\log\log T)$ 个批次达到 $\widetilde{\mathcal O}_d(T^{(d_z+1)/(d_z+2)})$ 的最优遗憾值与批次复杂度。 |
| [^22] | [PAC-Bayesian Reconstruction Guarantees for Time Series Variational Autoencoders](https://arxiv.org/abs/2609.05212) | 本文为应用于时间序列的潜变量模型提出了一个PAC-贝叶斯框架，将PAC-贝叶斯保证扩展到具有马尔可夫潜结构的重构界，且所得到的保证不随轨迹长度增长。 |
| [^23] | [FluxDisco: Symbolic Regression for Stoichiometric Dynamical Systems via Monte Carlo Graph Search](https://arxiv.org/abs/2609.05207) | 提出了FluxDisco框架，利用已知化学计量关系缩小搜索空间并适配蒙特卡洛图搜索算法，为化学计量ODE系统实现物理约束下的符号回归，从含噪数据中准确恢复可解释的控制方程。 |
| [^24] | [Phase Transition Frequency as a Training Time Predictor of Test Accuracy in ResNets](https://arxiv.org/abs/2609.05194) | ResNet微调过程中类可分性“相变”跳变的次数可作为最终测试精度的早期预测指标，在标准i.i.d.基准上呈强负相关（最高达r = -0.87），但在分布偏移条件下该相关性显著减弱。 |
| [^25] | [SMILE: Self-Explainable Multimodal Information Bottleneck for Medical Diagnosis](https://arxiv.org/abs/2609.05174) | 该论文提出SMILE框架，将自解释多模态医学诊断形式化为信息瓶颈问题，通过联合优化预测性能与模态级可解释性，并借助基于矩阵的Renyi's α阶熵泛函实现稳定优化，从而识别每个模态中对诊断决策贡献最大的关键信息。 |
| [^26] | [Conformal Prediction for Offensive Security](https://arxiv.org/abs/2609.05165) | 本文填补了共形预测在攻击性安全应用方面的研究空白，首次探索了其在隐私保护机器学习和网络流量分析两个关键攻击领域中的初步应用。 |
| [^27] | [Beyond Stationarity in Time Series: Discovering Causal Structures and Latent Regimes via Markov Blankets](https://arxiv.org/abs/2609.05150) | 该论文提出RCBNB-MB算法，突破时间序列平稳性假设，通过将时间序列分割为具有稳定因果结构的潜在状态并在每个状态内利用马尔可夫毯发现因果图，实现了对非平稳时间序列的鲁棒因果发现。 |
| [^28] | [A Hybrid Predictive Ensemble of Machine Learning and Deep Neural Networks for Early Cardiovascular Disease Risk Assessment](https://arxiv.org/abs/2609.05146) | 该研究提出了一种融合机器学习与深度神经网络的混合集成框架，利用医疗物联网的实时生理数据实现心血管疾病的早期风险预测与预后评估。 |
| [^29] | [From 80x to 385x: A Best-Matching-Unit Search at the L2 Roof, Measured Against a Symmetrically Tuned Baseline](https://arxiv.org/abs/2609.05138) | 该论文对新颖SOM算法SparseBin和基线算法cuSPARSE进行了对称调优，使最佳匹配单元搜索每epoch性能提升5.6-10.1倍，将相对基线的优势从约80倍扩大到385倍，并证明调优后的内核已触及77%峰值的L2带宽上限，进一步优化空间仅剩约1.3倍。 |
| [^30] | [MomentQuant: an even more minimalist interval method with linear time complexity for time series classification](https://arxiv.org/abs/2609.05136) | 该论文提出MomentQuant算法，通过对Quant算法进行更优化的实现，并利用Cornish-Fisher展开以近似分位数替代精确分位数从而免去排序步骤，将时间序列分类的计算效率进一步提升至线性时间复杂度。 |
| [^31] | [Coarse-Graining Hidden Representations: Unsupervised Neuron Selection via Mapping Entropy](https://arxiv.org/abs/2609.05126) | 该论文提出了一种完全无监督的神经元选择方法，通过映射熵对隐藏层的粗粒化方案进行评分，仅依据隐藏激活的统计特性即可识别出网络中最具信息量的神经元子集，无需标签或梯度。 |
| [^32] | [Single-Query Black-Box Calibration Auditing via Logit Bias](https://arxiv.org/abs/2609.05125) | 本文提出利用LLM API中暴露的logit_bias参数进行数学操作，每样本仅需一次查询即可评估精确概率阈值，并给出二分类任务真实校准误差的首个可证明一致估计器，为黑盒基础模型的校准审计提供高效框架。 |
| [^33] | [A Comparative Study of Counterfactual Explainers for Graph Neural Networks Enabling Multiple Types of Graph Edit](https://arxiv.org/abs/2609.05113) | 本文对六种最先进的图神经网络反事实解释方法在多种真实与合成数据集上的图分类和节点分类任务进行了系统比较，揭示了各方法在解释大小、覆盖率和质量之间的权衡，为该领域的未来研究提供了指导。 |
| [^34] | [NEAT-POCKET: Pocket-Conditioned Autoregressive 3D Molecular Generation with a Neighborhood-Guided Set Transformer](https://arxiv.org/abs/2609.05097) | NEAT-POCKET是一个蛋白质口袋条件下的自回归3D分子生成模型，在保持原子排列不变性并显式建模氢原子的同时，以显著快于现有方法的采样速度实现了具有竞争力的基于结构的药物分子生成性能，并支持口袋条件下的片段补全以辅助先导化合物优化。 |
| [^35] | [Deep Microcompression: Structured Pruning and Bit-packed Quantization for Microcontrollers](https://arxiv.org/abs/2609.05081) | DMC通过结构化剪枝、量化感知训练和位打包技术，实现了标准CNN在仅有2KB SRAM的ATmega328P微控制器上的首次部署，在LeNet-5上达到55.8倍压缩比并保持98.77%的准确率。 |
| [^36] | [Confounding-Valid Conformal Inference for Counterfactual KPIs in Wireless Networks](https://arxiv.org/abs/2609.05073) | 针对无线网络遥测数据中因遗漏变量而产生的隐藏混杂因素问题，本文提出了一种混杂有效性保形推断方法，在随机化遥测数据稀缺的情况下，为反事实KPI分析提供可靠的统计保证。 |
| [^37] | [Beyond Co-purchase Relation: Evolution of Complementary Recommendations at Allegro](https://arxiv.org/abs/2609.05063) | 本文提出部署在Allegro电商平台的生产级检索框架AlleCompanion，通过数据级过滤启发式方法与类别约束的双塔架构（含类别适配器）将嘈杂的共同购买行为信号转化为精确的语义互补性，从而实现更精准的互补产品推荐。 |
| [^38] | [Impact of Data Loss in Postprocessing on Training and Inference of Quantum Neural Networks](https://arxiv.org/abs/2609.05060) | 本研究以 Qiskit 的 SamplerQNN 为案例，揭示了面向模拟器设计的后处理假设在真实量子硬件上会导致高达99.6%的有效测量数据被静默丢失且难以察觉，从而在无任何API报错的情况下扭曲量子神经网络的训练与推理结果。 |
| [^39] | [An Analysis of Self-supervised Pre-training with Dependent Samples](https://arxiv.org/abs/2609.05031) | 该研究证明，在自监督预训练中，尽管同一数据点的不同增强之间存在相互依赖性，将它们汇集在一起学习比将数据划分为独立子集的基线方法是更好的选择。 |
| [^40] | [Amortizing Scaling Law Construction Costs](https://arxiv.org/abs/2609.05016) | 该论文提出了一种基于贝叶斯优化的高效缩放定律构建框架，通过逐步扩展计算预算并结合代理模型幻想评估来恢复广泛的实验网格，大幅降低了构建缩放定律的计算成本。 |
| [^41] | [Solution-space heterogeneity shapes federated learning dynamics across partial differential equations](https://arxiv.org/abs/2609.05012) | 该论文提出解空间 PDE-Dirichlet 协议，为偏微分方程联邦学习提供了可迁移的非独立同分布数据定义，推导出分配异质性与 Dirichlet 浓度的精确反比关系，并揭示了解空间异质性驱动梯度分歧与参数发散的动力学机制。 |
| [^42] | [Beyond Homoscedasticity: Decoupled Uncertainty Optimization for Deep Imbalanced Regression](https://arxiv.org/abs/2609.04995) | 该论文提出了解耦不确定性优化框架DUO，通过解决异方差负对数似然中的梯度耦合问题，增强长尾分布中困难尾部样本的学习信号，从而缓解深度不平衡回归中的尾部欠拟合。 |
| [^43] | [BeaconKV: Key-Value Cache Compression Guided by Beacon Queries for Efficient Large Reasoning Model Inference](https://arxiv.org/abs/2609.04971) | 该论文发现长程推理中存在会重新关注早期关键上下文的“思维回溯token”，且其查询在嵌入空间中聚成少数相似组，据此提出无需训练的KV缓存压缩方法BeaconKV，从而高效支持大型推理模型的推理。 |
| [^44] | [Fractal basins trap latent reasoning](https://arxiv.org/abs/2609.04963) | 该论文发现推理模型本质上是具有分形盆地的动力系统，暂态混沌源于推理在对应“近乎正确解”的鞍点附近长时间被困，从而揭示了任务越难、推理越慢的动力学机制。 |
| [^45] | [Physics-Aware Random Walk Fingerprints for Scalable Power Grid Graph Classification](https://arxiv.org/abs/2609.04943) | 本文提出多通道物理感知随机游走指纹（MC-PA-RWF），通过将物理边状态引入随机游走传播，为电网图级联故障分类任务提供了一种无需端到端训练、可扩展且可解释的轻量级图级表示方法。 |
| [^46] | [One Diffusion Model, Two Roles: Guided Trajectory Planning and Safety-Critical Scenario Generation in Closed-Loop Simulation](https://arxiv.org/abs/2609.04921) | 提出单个预训练扩散交通模型可同时充当自车规划器与安全关键场景生成器，借助SSDS扩散变换器解码器和免训练的DAPSE能量引导方案，在nuPlan上提升闭环规划性能并支持压力测试场景生成 |
| [^47] | [Fast Gauss Sums via Flash Attention](https://arxiv.org/abs/2609.04910) | 该论文发现仅需两次简单的输入增广，就能借助高度优化的Flash Attention计算任意带符号权重的高斯核和，无需编写任何自定义GPU代码，且在速度、内存开销和精度上均显著超越PyTorch和PyKeOps等现有方案。 |
| [^48] | [Methane Detection On Board Satellites from Unorthorectified Imagery](https://arxiv.org/abs/2609.04906) | 本文提出UnorthoDOS数据集与方法，首次直接在未正射校正的高光谱影像上训练U-Net模型进行甲烷羽流检测，性能接近基于正射校正数据训练的模型并大幅超越匹配滤波基线，同时通过FP16压缩验证了模型在卫星上部署的可行性。 |
| [^49] | [Sound-based Multi-Person 3D Pose Estimation](https://arxiv.org/abs/2609.04902) | 本文首次提出仅利用声学信号进行多人三维姿态估计的SoundMHPE框架，通过编码器-解码器架构克服了多人声学信号重叠和人际反射干扰的挑战。 |
| [^50] | [Adaptation Interfaces for In-Context Tabular Foundation Models in Time-to-Event Prediction](https://arxiv.org/abs/2609.04901) | 本文提出了将表格基础模型应用于事件时间预测的多种适配接口（时间零样本重构、分类微调和生存头适配），发现基于Cox的接口是最可靠且表现最强的适配方式，尤其在大规模数据集的综合Brier分数上表现突出，而零样本推理则适用于较小的数据集。 |
| [^51] | [From Language Models to World-Acting Systems: Progress and Limits of Agentic AI across Digital, Social, Virtual, and Physical Environments](https://arxiv.org/abs/2609.04894) | 该批判性综述指出，智能体AI在行动接口扩展方面进展显著，但在稳健的任务完成、授权、恢复与可信委派方面仍存在关键局限，且不应将其视为迈向自主性的单一进程。 |
| [^52] | [From Deep to Shallow: Unconstrained and Efficient Layer Merging Strategy](https://arxiv.org/abs/2609.04881) | 该论文提出了一种无需解析解且不增加卷积核大小的高效层合并策略，克服了现有深度压缩方法无法处理带填充卷积的局限，实现了神经网络的高效压缩与推理加速。 |
| [^53] | [When Genomic Masking Priors Fail to Transfer: Strong Variant Prediction, Weak Functional Generation](https://arxiv.org/abs/2609.04861) | GenDA双向扩散模型在变异效应预测上虽超越同规模自回归模型，但熵引导策略对性能提升并无实际贡献，且模型在零样本功能序列填充生成任务上失败，表明基因组掩码先验能很好地迁移到变异预测、却难以迁移到功能生成。 |
| [^54] | [KVMem: Virtualizing Million-Token Agent Workspaces on a Consumer GPU](https://arxiv.org/abs/2609.04852) | KVMem提出了一种KV上下文虚拟化系统，将智能体工作空间中溢出的历史记录以分页KV状态保存在GPU内存、主机内存和NVMe之间，并利用模型原生的注意力空间索引按需物化执行视图，从而在百万Token历史的长期智能体基准上实现了更高的任务效用和推理效率。 |
| [^55] | [Coupled Control and Wireless World Models for Resilient Remote Robotic Control](https://arxiv.org/abs/2609.04851) | 本文提出将控制世界模型与无线JEPA世界模型相耦合，从视觉观测和射频表示中联合预测机器人动力学与无线信道演化，从而实现预测性通信调度，在通信资源受限的不稳定无线网络下实现弹性远程机器人控制。 |
| [^56] | [PACE: Propagation-Aware Collaborative Correction for One-Shot Personalized Federated Graph Learning](https://arxiv.org/abs/2609.04832) | PACE提出了一种传播感知的协同校正方法，将外部协同知识作为对完整本地模型的紧凑校正而非替代品，通过服务器构建接收方锚定的校正并结合CNLL校准自动选择融合系数，有效解决了一次性个性化联邦图学习中的客户端异构性风险问题。 |
| [^57] | [Communication-Efficient Personalized Federated Learning via Layer-Wise Multi-Threshold Random Sketching](https://arxiv.org/abs/2609.04830) | 该论文提出一种逐层多阈值随机草图方法，通过针对不同层的参数分布采用差异化的多阈值量化策略，克服了传统单比特方法单一阈值的局限性，实现了通信高效的个性化联邦学习。 |
| [^58] | [Minimax Lower Bound for Estimating Diffusion-based Local Intrinsic Dimension](https://arxiv.org/abs/2609.04822) | 本文首次研究了基于扩散的局部内在维度估计的统计难度，证明了有限尺度场与流形维度d的偏差至多为O(σ²)，并建立了阶为(nσ^d)^{-1}的极小极大下界。 |
| [^59] | [Federated Attack Campaign Detection via Contrastive Encoding of Threat Indicators in Gradient Updates](https://arxiv.org/abs/2609.04815) | 提出FedIoC联邦学习框架，客户端通过监督对比学习将本地威胁指标编码进梯度更新，使监控同一攻击活动的客户端产生对齐的梯度分量，从而在无需共享敏感数据的前提下实现跨组织的协调网络攻击活动检测。 |
| [^60] | [How Faithful Is Attribution for Sales Forecasting? A Counterfactual Study](https://arxiv.org/abs/2609.04797) | 该研究为多序列WaveNet销售预测模型添加了反事实可解释性层，其归因分解可精确加总至预测值，并通过删除/插入测试以统计显著性验证了归因忠实反映模型的真实行为而非伪影。 |
| [^61] | [Learning-Augmented Algorithms: Guarantees, Construction Mechanisms, and System-Level Implications](https://arxiv.org/abs/2609.04787) | 本综述系统梳理了学习增强算法领域的预测接口、误差度量、一致性—鲁棒性权衡及五种代表性构造机制，明确区分形式化保证与实证系统证据，并指出了该领域的未来开放问题。 |
| [^62] | [Dynamic Heterogeneous Graph Representation Learning: A Survey](https://arxiv.org/abs/2609.04779) | 本综述首次系统性地回顾了动态异构图表示学习领域，提出了一个统一涵盖离散时间与连续时间的动态异构图形式化定义，并据此建立了一种新颖的以算法为中心的分类体系。 |
| [^63] | [Persistent Teacher Anchoring for Tool-Using Agents](https://arxiv.org/abs/2609.04773) | 提出持久教师锚定（PTA），一种由学生诱导但由教师承诺的 rollout 构建方法，通过在块级验证基础上增加轮级承诺让工具调用得以执行，从而解决工具使用中师生分布差距累积导致的漂移问题。 |
| [^64] | [A Robust Watermark-based Fingerprint Framework for GNNs Ownership Verification](https://arxiv.org/abs/2609.04772) | 本文提出REMARK框架，一种用于GNN所有权验证的鲁棒水印指纹方法，通过精心构造分布内水印图等手段，克服了现有方法在模型性能退化、不现实的代理模型假设以及过度依赖特定输出层三个方面的局限性。 |
| [^65] | [Resilience Beyond Stationary Client Unavailability: Unlocking Efficient and Unbiased Federated Learning](https://arxiv.org/abs/2609.04763) | 提出FedSWE算法，可高效且无偏地应对联邦学习中异构、非平稳的随机客户端可用性问题，且无需以往方法那样高昂的内存与计算开销。 |
| [^66] | [A Fairness Audit of the Duckworth-Lewis-Stern Method: Format-Specific and Gender-Differential Bias, with an Interpretable Calibration Layer for Cricket Target Revision](https://arxiv.org/abs/2609.04754) | 该论文首次对板球DLS目标修正方法进行大规模公平性审计，发现其预测误差随比赛状态变化跨度高达137分，且在单日国际赛中存在显著的性别偏差（女性比赛的平均过高预测比分性比赛多6.13分），并提出一个可解释的校准层来修正这些问题。 |
| [^67] | [Same Request, Different Answer: Quantization Amplifies Cache-Induced Divergence in LLM Serving](https://arxiv.org/abs/2609.04748) | 该论文首次系统量化了LLM服务中前缀缓存对推理可复现性的破坏，并揭示权重量化会显著放大这一效应——16位精度下36.2%的智能体回合轨迹发生偏离，而在4位量化下这一比例攀升至75.0%。 |
| [^68] | [Training Large Language Models for Small-Molecule Design with Synthetic Task Scaling](https://arxiv.org/abs/2609.04735) | 本文提出基于课程学习的合成任务扩展训练方法，通过逐步引入难度递增的廉价合成设计任务来训练大语言模型，使其习得的分子设计策略能够泛化到评估成本高昂的真实分子先导化合物优化场景。 |
| [^69] | [Locating and Steering Refusal Beyond Attention](https://arxiv.org/abs/2609.04721) | 该研究发现拒绝行为的安全表示可跨架构迁移：仅通过一次刚体旋转即可将Transformer的表示空间与状态空间模型（SSM）对齐，使在Transformer上训练的有害探针能识别SSM的有害输入，而移除该对齐方向即可解除模型的拒绝行为。 |
| [^70] | [Simulation-free Unbalanced Dynamic Optimal Transport with General Growth Penalty](https://arxiv.org/abs/2609.04710) | 该论文提出SUDO，一种无需模拟的非平衡动态最优传输框架，突破了现有方法仅适用于二次（WFR）生长惩罚或依赖昂贵NeuralODE模拟的限制，能够高效支持一般生长惩罚下的单细胞动力学推断。 |
| [^71] | [Sustainable Edge Vision via Empirically Calibrated DVFS: Eliminating Thermal Throttling on Passively Cooled Hardware](https://arxiv.org/abs/2609.04705) | 提出一种经验校准的状态感知DVFS调度器，在无源散热的树莓派5上运行YOLOv8n时完全消除了热节流，相比仅温度响应式基线帧率提高6.8%且每帧能耗降低1.9%。 |
| [^72] | [LookThere! Sparse Vision by Reinforced Selection](https://arxiv.org/abs/2609.04698) | 提出LookThere端到端强化学习框架，联合训练浅层输入选择器与深层表示提取器，在不依赖辅助信号的情况下实现极端稀疏条件下性能与计算权衡的新帕累托前沿。 |
| [^73] | [A Differentiable Neural Surrogate for Photon Propagation in Neutrino Telescopes](https://arxiv.org/abs/2609.04695) | candela 是一个可微分的 SIREN 神经场，通过学习冰立方探测器的光子格林函数，能够以比传统蒙特卡洛方法快 50-100 倍的速度模拟中微子望远镜中的光子传播，同时将产量精度保持在蒙特卡洛期望值的 2% 以内。 |
| [^74] | [Enhancing Multimodal Emotion Recognition via Multi-Feature Encoding and Attention-Based Fusion](https://arxiv.org/abs/2609.04690) | 本文提出了一种多模态情感识别框架，通过融合Wav2Vec2语义嵌入、MFCC及声学统计特征等多类音频特征、ResNet50-BiLSTM视觉时空特征提取，以及基于多头注意力的特征级融合机制，显著提升了情感识别效果。 |
| [^75] | [WEECFP-SuRGE: Wide Embedded Extended Connectivity Fingerprint with Substructure Rotary Graph-distance Encoding](https://arxiv.org/abs/2609.04672) | 该论文提出了无需参数的分子指纹WEECFP以及结合子结构旋转图距离编码（SuRGE）的transformer架构WEECFP-SuRGE，在不使用任何外部预训练的情况下，在TDC ADMET排行榜的22项基准中取得多项第1名和总体领先的回归性能。 |
| [^76] | [Interpretability for Turing Machines](https://arxiv.org/abs/2609.04661) | 本文将源自神经网络的可解释性技术“敏感性分析”拓展至图灵机，从理论和实证上证明算法中的对称性与路径分离会诱导敏感性矩阵中的置换对称性和低秩块，并可通过主成分分析和聚类方法恢复算法特征。 |
| [^77] | [Latent-Aligned Reasoning for Multimodal Recommendation](https://arxiv.org/abs/2609.04645) | 提出LARK两阶段潜空间推理框架，通过可学习token与冻结视觉编码器对齐及物品对比学习，解决多模态推荐中VLM多步推理导致的视觉与文本信号衰减（跨模态稀释）问题。 |
| [^78] | [SMILE: Bridging Continuous Optimization and Discrete Symbolic Recovery](https://arxiv.org/abs/2609.04639) | SMILE提出了一种三阶段混合框架，通过数据结构分析、可解释激活网络的连续优化以及结构化剪枝与符号恢复，将连续梯度优化与离散符号搜索相结合，能够高效地从数据中恢复出具有精确符号常数的紧凑数学表达式。 |
| [^79] | [Tracing Audio Grounding and Answer Selection in Audio LLMs](https://arxiv.org/abs/2609.04637) | 该研究揭示了音频大语言模型内部音频真正决定答案的机制：训练主要在中间至后期层增强音频对最终预测的影响，而声学信息主要在早至中间层塑造答案选项的表示。 |
| [^80] | [Too Rare to Learn: Prescribed Cyclone Tracks Degrade a Bay of Bengal Ocean Emulator](https://arxiv.org/abs/2609.04635) | 该研究发现在孟加拉湾神经网络海洋模拟器中引入预设气旋轨迹输入反而会降低预报技巧，原因是气旋事件在训练数据中过于罕见（仅占7.9%的天数），导致这些输入一旦激活就超出模型的训练分布。 |
| [^81] | [SCAPES: Semantically Conditioned Autoregressive Prior for Environmental Sounds](https://arxiv.org/abs/2609.04634) | SCAPES是一个轻量级的语义条件自回归生成模型，通过在神经音频编解码器的连续潜在空间上用流匹配建模重叠音频片段的潜在轨迹演化，仅用3600万参数即可在有限数据上实现高保真环境声音的语义控制合成。 |
| [^82] | [SiLR: Structure-Preserving Admission and Process Reward for LLM Tool Agents](https://arxiv.org/abs/2609.04629) | 该论文提出SiLR方法，通过影子执行提案并在分支级违规状态的乘积序下进行结构化准入决策，从理论上证明了任何标量评分都无法可靠替代这种结构化比较，从而解决了LLM工具代理违规后恢复中的标量投影陷阱问题。 |
| [^83] | [GNN-Guided Graph Coarsening and Adaptive QUBO Penalties for the Capacitated Vehicle Routing Problem with Time Windows on a Quantum Annealer](https://arxiv.org/abs/2609.04593) | 本文提出GNN引导的图粗化与自适应QUBO罚项校准方法，在Solomon基准上结合模拟退火与D-Wave量子退火机求解CVRPTW，将平均原始约束违反从33.0降至0。 |
| [^84] | [Hidden In Plain Gaze: Gaze Representations as Privacy Controls for Utility and Re-identification Risk in XR](https://arxiv.org/abs/2609.04592) | 该研究发现在XR系统中，选择人工设计的眼动特征作为数据表征可以在保留约85%动作识别准确率的同时，将用户再识别风险降低约一个数量级，从而在特征提取阶段提供一种轻量级的隐私控制手段。 |
| [^85] | [Representation Redundancy and Structural Complexity in Finite-Field Inversion](https://arxiv.org/abs/2609.04583) | 该论文证明有限域 \(\mathbb F_{2^n}\) 上有序基与不同坐标求逆映射之间恰好是 n 对一的冗余对应（同一伽罗瓦轨道的基诱导同一映射），并通过比较三种布尔表述的代数次数与联合 ANF 跃变，刻画了表示选择对求逆运算结构复杂度的影响。 |
| [^86] | [Centered Permutation Prefixes for SGD with Random Reshuffling: Sharp Rates, H\"older Geometry, and Composite Proximal Extensions](https://arxiv.org/abs/2609.04578) | 本文通过中心化置换前缀技术证明了随机重排SGD在分量非凸、无需分量Hessian连续性的条件下即可达到与已知二次下界匹配的 Õ(T^{-2} + n²T^{-3}) 期望收敛速率，并将结果推广到Hölder几何与复合近端问题。 |
| [^87] | [Optimizer Memory Schedules for Outscaling the Overtraining Axis](https://arxiv.org/abs/2609.04577) | 该研究揭示优化器的相对性能与最优超参数会随过度训练程度显著变化（如最优权重衰减约按 sqrt(OT) 缩放、更长训练偏好更长的优化器记忆），并通过引入对数时间权重衰减与动量冷却，使 ADANA 在长训练时长下持续保持超越 AdamW 的扩展优势。 |
| [^88] | [Training-Free Halving of Activated Experts in Fine-Grained Mixture-of-Experts Models](https://arxiv.org/abs/2609.04575) | 提出一种无需训练的方法，通过将激活的专家数量与归一化参考集大小解耦，在细粒度MoE模型中将激活专家减半，在几乎不损失精度的同时显著降低计算量。 |
| [^89] | [MURAL: Multimodal Uncertainty-aware Recommendation via Adaptive edge Learning](https://arxiv.org/abs/2609.04574) | MURAL提出统一框架，通过可微分的自适应边学习器动态发现潜在物品关联并引入不确定性感知机制，解决了多模态推荐中静态图结构僵化和噪声模态信号融合导致的两大瓶颈。 |
| [^90] | [Extremely Sparse Supervision Incentivizes Reasoning Ability](https://arxiv.org/abs/2609.04565) | 研究发现在在线策略蒸馏中，仅需对每个推理轨迹中的一两个token（约占总token数的0.05%）进行监督，就能有效激励大语言模型的推理能力，且在多数情况下可匹配甚至超越全token训练的效果。 |
| [^91] | [A Sim-to-Real Study of Surface-Code Decoder Benchmarking](https://arxiv.org/abs/2609.04557) | 本研究利用运行在表面码阈值以下的Willow处理器真实数据，对六种解码器在四级保真度递增的噪声模型下进行基准排名，发现只有为每种操作类型赋予独立错误率的噪声模型才能产生与硬件一致的排名，并首次对NVIDIA的Ising预解码器进行了硬件独立评估。 |
| [^92] | [Fast Surrogate Modeling of Excitable and Oscillatory FitzHugh-Nagumo Dynamics with Parametric Neural Operators](https://arxiv.org/abs/2609.04549) | 本文提出将参数条件化的傅里叶神经算子（通过FiLM机制注入参数向量）作为FitzHugh-Nagumo系统的快速可微代理模型，并结合分岔分析针对振荡态和可激发态两种动力学状态分别训练算子，从而实现对5维生理参数空间的高效快速探索。 |
| [^93] | [Mitra-v2 Technical Report](https://arxiv.org/abs/2609.04540) | Mitra-v2 是一个仅用合成数据训练的表格基础模型，通过更大更多样的预训练任务分布和小型 2D Transformer 架构，在 300 多个真实数据集上达到与工业级大模型相当的最先进性能，并大幅超越 TabPFN-3。 |
| [^94] | [Distilled Continuous Diffusion Language Models Can Write Code in Few Steps---or One](https://arxiv.org/abs/2609.04531) | 本文提出0.7B参数的连续扩散代码生成模型PlaidQ，通过分布匹配和成对轨迹监督的蒸馏技术，将去噪轨迹压缩到仅需16步甚至1步即可高效生成代码，性能可与离散扩散语言模型媲美。 |
| [^95] | [An Energy-Based Conservative-Dissipative Latent Neural Evolution Operator for Magnetization Dynamics](https://arxiv.org/abs/2609.04530) | 该论文提出一种将卷积自编码器与结构化潜在神经常微分方程耦合的能量基降阶模型，通过学习标量势构造保守-耗散的潜在动力学结构，保证潜在能量沿解单调递减，且无需物理能量标签或时间导数监督即可联合训练，用于微磁磁化动力学的建模。 |
| [^96] | [Scale-QLoRA: Code-Invariant Adapter Merging for Native 4-bit Microscaling LLMs](https://arxiv.org/abs/2609.04526) | Scale-QLoRA通过只训练原生4比特微缩放检查点的每块缩放因子、同时冻结全部E2M1编码来合并LoRA适配器，从而避免了朴素合并因重新量化编码平面而抹除适配效果（最高达39个百分点）的问题。 |
| [^97] | [Hakken: Predicting future discoveries to fill the gaps in today's knowledge](https://arxiv.org/abs/2609.04494) | Hakken是一个领域无关的知识预测系统，它将基于transformer的时序知识图谱预测模型与LLM的语义知识相融合，预测科学概念间尚未发现的新关系，并通过解释框架帮助科学家评估这些预测，在生物医学领域的时间感知多标签关系预测上树立了新的基准。 |
| [^98] | [ResLearn-XR: Residual Learning for Network Traffic and Quality-of-Experience-Aware Modeling in Extended Reality](https://arxiv.org/abs/2609.04493) | ResLearn-XR是一个两阶段残差学习框架，通过在值空间进行流量预测、在logit空间进行QoE风险估计，并引入数据描述符算法（DDA）和新的XR Traffic-QoE数据集，实现了对突发性XR网络流量的准确预测和QoE风险的可靠估计。 |
| [^99] | [When Quantization Breaks Memory: Recurrent-State Write-Back in Low-Precision Temporal Inference](https://arxiv.org/abs/2609.04490) | 该论文揭示循环网络中量化状态的存储规则（循环状态回写）会显著影响后续推理——在荧光寿命成像任务中，用4位确定性状态存储替代连续状态传播会使两个寿命参数的估计误差分别增加约70倍和300倍。 |
| [^100] | [Client-Side Probing of Deleted Ridge Statistics in Federated Unlearning](https://arxiv.org/abs/2609.04475) | 该研究揭示了一种新的隐私威胁：在联邦遗忘系统中，恶意客户端可以利用服务器广播的线性分类器更新作为探测手段，精确恢复被删除的样本、类别或客户端摘要，并可能将被删除的数据重新插入。 |
| [^101] | [Nested Inductive Bias Framework for SPD Manifold Learning](https://arxiv.org/abs/2609.04466) | 提出嵌套归纳偏置框架，通过两阶段微分同胚组合将庞加莱度量等非欧几里得目标几何回拉到SPD流形上，从而同时满足流形约束与关系先验两类归纳偏置。 |
| [^102] | [On-board ML for Trace Gas detection in Imaging Spectroscopy data](https://arxiv.org/abs/2609.04458) | 该研究首次利用边缘机器学习技术实现了成像光谱数据中甲烷点源排放的机载实时检测，通过下传轻量模型预测的潜在事件而非完整数据立方体，克服了通信带宽瓶颈并大幅缩短了从观测到信息获取的时间。 |
| [^103] | [When Load-Balancing Goes Too Far: Expert Pruning in Over-Dispersed Mixture-of-Experts Models](https://arxiv.org/abs/2609.04453) | 该论文发现在因训练时负载均衡过于激进而导致路由过度分散的MoE模型中，路由器概率不再是可靠的专家重要性信号，困惑度也无法预测下游任务准确率，因此传统的基于路由的专家剪枝方法在此类模型中会失效，且不同评分指标之间存在能力权衡。 |
| [^104] | [Conformity Breaks Conformal Prediction](https://arxiv.org/abs/2609.04445) | 论文揭示同伴压力会使多智能体LLM系统发生“评分机制偏移”，悄然破坏共形预测的覆盖率保证（从90%降至74%），且攻击者可通过针对低置信度子组将其覆盖率近乎减半（从87%降至47%）。 |
| [^105] | [Recovering molecules from coarse-grained beads: free-energy-conditioned generative backmapping across chemical space](https://arxiv.org/abs/2609.04432) | 本文提出juniper——一个以辛醇-水分配自由能为条件的分子图离散去噪扩散模型，首次将组成层面的粗粒化反向映射（从粗粒化珠子组合恢复对应分子化合物）表述为条件图生成问题，实现了跨化学空间的分子恢复。 |
| [^106] | [A Repeated-Measurement Study for Cultural Analytics of English Song Lyrics Using Five Large Language Models](https://arxiv.org/abs/2609.04428) | 本研究通过对大型歌词语料库的重复标注实验，系统评估了五个大语言模型对英文歌词中自尊、自控、寻求归属感和寻求认可四种社会构念进行零样本标注的测量可靠性，发现LLM测量的可靠性因构念而异，其中自尊的重复测量可靠性最强。 |
| [^107] | [Beyond a Universal Forecasting Selector: Demand-Conditioned Model Selection across Demand Patterns and Horizons](https://arxiv.org/abs/2609.04425) | 该研究通过大规模对比实验证明不存在普适最优的预测模型选择器，选择器的优劣依赖于需求模式、数据可得性和预测期，因此应将选择器作为依赖情境的组件并根据具体需求条件加以选择。 |
| [^108] | [Tuning Collective Patterns to Alleviate Congestion in Shared AI Clusters](https://arxiv.org/abs/2609.04417) | 提出了REACT系统，它在应用层（通信库）利用易于获取的流统计信息在运行时检测拥塞，并动态调整GPU节点间的集合通信模式，从而无需全局控制或基础设施支持即可缓解共享AI集群中的拥塞问题。 |
| [^109] | [REFINE: LLM Refinement over Budgeted Text-Attributed Graphs for Personalized Medical Concept Representation](https://arxiv.org/abs/2609.04415) | 提出REFINE框架，通过强化学习策略在预算约束下利用大语言模型对知识图谱进行图精炼，从文本属性知识图谱中学习患者个性化的医学概念表示，从而提升EHR预测性能。 |
| [^110] | [Disentangling Attention in Deep Operator Learning: A Controlled Study of Data-Driven and Physics-Informed Architectures](https://arxiv.org/abs/2609.04407) | 本文通过对五种注意力机制不同的DeepONet变体进行受控系统研究，在数据驱动与物理信息两种训练模式下训练，首次分离了交叉注意力、自注意力、标记化和注意力深度各自对偏微分方程求解精度的贡献。 |
| [^111] | [Candidate Comparability Before Promotion: Conditional Validation in Adaptive Network Intrusion Detection](https://arxiv.org/abs/2609.04388) | 本文针对自适应网络入侵检测提出“晋升前的条件化验证”方法，证明晋升结论依赖于挑战者的构建方式与证据规模——表观的晋升危害主要是由现任模型的冻结预处理放大所致，在自包含流水线与严格控制证据量的条件下该危害并不成立。 |
| [^112] | [Privacy Failure in Split-LLM Training, The Returned Gradient Nullifies the Decoys](https://arxiv.org/abs/2609.04382) | 论文揭示了拆分式LLM训练中一个被忽视的隐私漏洞：由于损失函数忽略诱饵行，返回梯度中诱饵行的值恰好为零，其零值模式可完美暴露真实数据行，在九个随机种子下每轮运行4,096帧全部被准确识别。 |
| [^113] | [On the Abundance of Critical Points of the t-SNE Energy](https://arxiv.org/abs/2609.04379) | 本文在特征空间密度服从连续对称性的条件下，通过识别成对的离散对称性，为包括 t-SNE 及其大数据极限在内的一般能量族构造出无穷多族不同的临界点，从而为解释 t-SNE 能量景观中存在大量不遵循数据拓扑与聚类结构的局部极小值这一现象提供了严格的理论起点。 |
| [^114] | [VLA-Precision: Asymmetric Co-Bootstrapping for Efficient Real-World Online RL of Vision-Language-Action Models](https://arxiv.org/abs/2609.04355) | 提出VLA-Precision框架，通过非对称协同自举算法和ACoB-Stream架构，同时解决VLA模型真实世界在线强化学习中价值信号不可靠与计算开销过大两大瓶颈，实现高效且精确的策略改进。 |
| [^115] | [A Quantum Variational Approach to Prototypical Recurrent Unit](https://arxiv.org/abs/2609.04354) | 提出了一种参数量显著少于经典（LSTM、GRU）和量子（QLSTM、QGRU）循环架构的轻量级量子原型循环单元 QPRU，在保持与最先进基线相当的预测性能的同时，具备更强的可扩展性。 |
| [^116] | [A Constraint-Aware Generative Framework for Synthetic Origin-Destination Demand in Logistics Networks](https://arxiv.org/abs/2609.04345) | 提出了一种约束感知的条件生成框架，通过可微约束将运营指导直接融入生成目标，为分层物流网络生成拓扑真实且运营可行的合成起讫点需求数据。 |
| [^117] | [SharedSAE: One Feature Dictionary Across Language Models](https://arxiv.org/abs/2609.04344) | SharedSAE通过共享字典结合各模型专属的编码器-解码器对，用单一稀疏自编码器即可替代多个语言模型的专用SAE，在保留96.6%解释方差的同时实现跨模型的统一特征解释与迁移。 |
| [^118] | [Modular Deep Recurrent Neural Network: Application to Quadrotors](https://arxiv.org/abs/2609.04339) | 本文提出一种带层间前馈连接的模块化深度循环神经网络，显著提升了学习和建模高阶动态与非线性特性的能力，并缓解了梯度消失/爆炸问题，在四旋翼无人机高度动态建模中验证了其相比现有方法的优越性。 |
| [^119] | [Data-Driven Learning of Unknown Nonlinear Differential Equations Using Functional Analysis](https://arxiv.org/abs/2609.04329) | 本文提出一种基于泛函分析与算子理论的可解释机器学习方法，通过在函数空间中以积分形式构造代价函数，仅利用单条状态轨迹数据即可在线增量地学习非线性常微分方程的未知向量场。 |
| [^120] | [The microscope is the mask: privileged views and labels from a cryo-ET forward model](https://arxiv.org/abs/2609.04325) | 该论文提出CARNIVAL模型，巧妙利用冷冻电子断层扫描前向模型模拟产生的损坏配对视图作为自监督学习信号，并将模拟中蛋白质的位置和身份等特权信息融入模型架构与损失函数，无需微调即可在真实断层图上完成蛋白质分类与检测。 |
| [^121] | [TNFlow: Amortized Posterior Inference for Trans-Neptunian Object Surface Composition](https://arxiv.org/abs/2609.04305) | 提出TNFlow架构，将Transformer与归一化流结合，能在单个CPU上约0.7秒内从反射光谱快速反演海王星外天体表面成分与颗粒尺寸的多模态后验分布，在合成光谱上表现出良好精度与泛化能力，但在真实JWST光谱上对某些材料存在盲区或偏差。 |
| [^122] | [Data-Optimized Contingency Screening: A Machine Learning Approach to Power System Security](https://arxiv.org/abs/2609.04300) | 本研究提出一种基于机器学习的电力系统预想事故安全等级分类方法，利用SMOTE和PCA对牛顿-拉夫逊潮流计算提取的数据进行优化，并训练KNN、随机森林和支持向量机模型，将事故划分为安全、中等和严重三类，以支持主动决策并防范大规模故障。 |
| [^123] | [BER-PEF: Unified Human Mobility Predictability Evaluation via Bayes Error Rate Estimation](https://arxiv.org/abs/2609.04292) | 提出BER-PEF框架，将贝叶斯错误率估计转化为人类移动可预测性估计，并在无可观测真实值的情况下为比较不同估计器提供了统一的评估协议。 |
| [^124] | [Evidence Integration in Large Language Models](https://arxiv.org/abs/2609.04290) | 本文提出了一个关于大语言模型证据整合机制的分布理论，并通过千万级规模实验验证了三个预测：接收者眼中更可能的候选答案更具说服力、模型更易整合自身特有错误而非外来错误、以及相同证据可能改善弱模型却损害强模型。 |
| [^125] | [When Seeing Overrides Knowing: Visual Dominance and Deferral-Based Method for Personalized Safety in VLMs](https://arxiv.org/abs/2609.04281) | 该研究提出个性化安全基准MPS-Bench，发现主流视觉语言模型因“视觉主导性”机制——即视觉信息在多模态融合中过早抑制文本风险信号——而几乎从不询问缺失的用户情境，并据此提出基于延后答复的方法以提升模型的个性化安全性。 |
| [^126] | [Evaluating Large Language Models for Forced Outage Risk Prediction: Benefits and Comparison to Machine Learning](https://arxiv.org/abs/2609.04272) | 本研究首次将大语言模型以零样本方式应用于电网天气相关停电风险预测，发现其精度虽略逊于监督机器学习模型，但在可操作推理和地理可扩展性方面具有独特优势，两者结合可能是最佳实践。 |
| [^127] | [Quantum-Assisted Memory-Efficient Training for Parameter-Intensive Wi-Fi-Based Human Activity Recognition](https://arxiv.org/abs/2609.04271) | 本文提出量子辅助内存高效训练框架Q-MET，通过混合量子-经典神经网络间接生成模型参数，显著降低了Wi-Fi人体活动识别模型的可训练参数数量和内存消耗，使其更适合资源受限设备的部署。 |
| [^128] | [Corporate-Family Resolution Is Not a String-Matching Problem: A Public Benchmark Stratified by Name Visibility](https://arxiv.org/abs/2609.04269) | 该论文发布了CorpFam——一个基于美国联邦奖励记录中供应商自报母公司数据构建的企业家族消解公开基准，通过按名称可见性分层评估并报告分层召回率，揭示出最强匹配方法对名称不可见的企业家族对识别率仅4.2%，证明企业家族消解不能被简化为字符串匹配问题。 |
| [^129] | [A Data Fusion Framework for Grounding Aerospace Surrogate Model via Experimental Wind-Tunnel Observations](https://arxiv.org/abs/2609.04267) | 本文提出一种数据融合修正框架，利用风洞PSP压敏漆实验测量数据对CFD训练的深度学习气动代理模型进行在线修正，无需重新训练即可弥合CFD与实验之间的系统性差异并提升预测精度。 |
| [^130] | [Compute-in-Memory Attention: A Time-Domain Analog Softmax Circuit with RC-Tunable Temperature](https://arxiv.org/abs/2609.04266) | 该论文提出了一种基于22nm FDSOI工艺的时域模拟softmax电路，利用共享下降斜坡和RC衰减参考直接对计算内存储器生成的模拟分数电压进行指数运算与电路内归一化，无需模数转换，并可通过斜坡斜率和RC时间常数灵活调节softmax温度。 |
| [^131] | [ProToMEx: Rapid, Interpretable Explanations via Structured Representations](https://arxiv.org/abs/2609.04265) | ProToMEx利用概率主题模型学习潜在“主题”来表示分类背后的高层次语义原因，实现了超越传统特征归因的全局与局部解释，其解释保真度可与SHAP、LIME等主流方法媲美。 |
| [^132] | [Spectral-Target Physical Latent Structuring for JEPA-Style World Models](https://arxiv.org/abs/2609.04264) | 该论文发现了JEPA风格世界模型中“物理表示惰性”这一新失败模式，并提出轻量级傅里叶辅助头在训练时对潜在空间施加物理信息结构化约束，在不增加推理成本的情况下有效提升下游规划性能。 |
| [^133] | [Low-Latency Spell Correction for Japanese Music Search Queries](https://arxiv.org/abs/2609.04262) | 该论文提出了一种基于BART的紧凑序列到序列模型，并设计了脚本感知的合成拼写错误生成流水线，结合键盘布局模型与语音混淆先验，实现日语音乐搜索查询的低延迟拼写纠正，同时通过曲目标题脚本规范化有效减少模型幻觉。 |
| [^134] | [Self-Supervised Pretraining of Molecular Graph Encoders with LeJEPA](https://arxiv.org/abs/2609.04261) | 该研究将LeJEPA自监督预训练方法适配到分子图编码器，发现预训练虽能显著提升嵌入表示质量（ogbg-molhiv冻结探针ROC-AUC提升0.123），但这一优势并不能稳健地转化为下游微调任务的性能增益。 |
| [^135] | [EXAONE Forecast for Finance](https://arxiv.org/abs/2609.04239) | EXAONE Finance是一个专为金融预测设计的时间序列基础模型，其核心创新在于采用无注意力机制的高效架构，通过因果一维卷积和组感知池化两个线性时间复杂度算子取代自注意力机制，从而解决了现有模型计算成本过高、无法处理间歇性观测数据以及未能捕捉金融市场独特动态的问题。 |
| [^136] | [GEPARD - Generative, Prosody-aware, Autoregressive text-to-speech model for Realtime Dialogue](https://arxiv.org/abs/2609.04222) | GEPARD是一个基于标准LLM骨干网络、可在不修改vLLM推理引擎计算内核的情况下流式运行的自回归文本转语音模型，通过将零样本语音克隆等辅助机制移出解码循环，实现了真正的实时语音对话生成。 |
| [^137] | [Sequential Beats Joint: On the Interplay between On-Policy Distillation and RLVR](https://arxiv.org/abs/2609.04108) | 先蒸馏后强化学习的两阶段训练方案在推理任务上持续优于纯OPD、纯RLVR及所有联合优化方法，因为OPD先扩大学生对教师解的覆盖范围、RL再在其内锐化，而联合训练会导致两种信号相互干扰。 |
| [^138] | [A location-invariant estimator of extremal quantile treatment effects for heavy-tailed distributions](https://arxiv.org/abs/2609.04018) | 该论文通过逆倾向得分加权将位置不变的Fraga极值指数估计量引入因果设置，并采用基于差分的外推方案使位置参数在分位数差中自然抵消，从而首次构造出对位置平移保持不变的极端分位数处理效应估计量，解决了现有方法在重尾分布下不满足位置不变性的问题。 |
| [^139] | [LLM4CKD: Large Language Models for Early Stage Chronic Kidney Disease Screening](https://arxiv.org/abs/2609.04013) | 大语言模型在零样本和少样本学习设置下，无需任务特定训练即可实现与传统机器学习和深度学习方法相当的早期慢性肾病筛查性能。 |
| [^140] | [Interface-Induced Trajectory Censoring](https://arxiv.org/abs/2609.03966) | 该论文发现智能体评估中的工具调用率可能被服务栈接口“审查”为零——即使模型实际发出了格式良好的调用，且2x2析因实验证明全部效应源于聊天模板与解析器之间的交互（修复任何一方都无效），仅更换服务适配器即可使同一模型得分从 0.00 跃升至 0.96/0.19。 |
| [^141] | [Free Pause Tokens](https://arxiv.org/abs/2609.03807) | 提出免费暂停标记，通过权重共享主干上的并行预测流为模型提供额外思考计算，在不增加上下文长度、KV缓存和推理延迟的情况下，仅以1.14倍训练计算量的代价提升下一个词元预测性能。 |
| [^142] | [Tree species mapping in Denmark: A comparison of spectral-temporal features with geospatial foundation model embeddings](https://arxiv.org/abs/2609.03480) | 本研究利用丹麦国家森林清查数据与Sentinel卫星观测，系统比较了人工构建的光谱-时间特征与地球观测基础模型（TESSERA和AlphaEarth）嵌入在树种分类中的表现，发现基于光谱-时间特征的多层感知机取得最佳性能（纯林宏观F1达0.843），而基础模型嵌入也展现出有竞争力的结果，为大规模森林树种制图的方法选择提供了重要参考。 |
| [^143] | [Counterfactual Fairness Audits of Multi-Step Clinical LLM Agents Require a Measured Per-Action Instability Floor](https://arxiv.org/abs/2609.03221) | 临床LLM智能体在完全相同输入下本身就存在显著的动作不稳定性（约8.7%），因此反事实公平性审计必须先测量这一“每动作不稳定性底线”，否则任何检测到的人口统计学差异都无法解释。 |
| [^144] | [Post-Training Language Models for Gold-Medal Performance in Coding Competitions](https://arxiv.org/abs/2609.02849) | 该研究通过结合大规模题目筛选、监督微调、强化学习以及反馈驱动的测试时计算策略 GenCorrect，使语言模型在 IOI 2025 编程竞赛中取得了超越金牌分数线（438.3 分）的成绩（Nano-CC 达 468 分，Ultra-CC 达 502 分）。 |
| [^145] | [TrajMind: Chaining Role-Specialized LoRAs for Fast-and-Slow Collective Trajectory Anomaly Diagnosis](https://arxiv.org/abs/2609.02540) | TrajMind提出快慢双路径框架，在单一冻结的视觉-语言骨干网络上链式切换三个角色专精LoRA适配器，将持续在线筛查与按需诊断解耦，从而实现低延迟且经源数据可验证的集体轨迹异常诊断。 |
| [^146] | [DeepAffinity: Long-Term Aspect Preference Prediction in eCommerce using Small Language Models](https://arxiv.org/abs/2609.02468) | 提出DeepAffinity框架，利用结构化提示和专用预测头微调的小语言模型（SLM）从用户时序交互历史中预测其对品牌、尺寸、颜色等产品属性的长期偏好，性能优于标准生成式微调方法，并能提升大规模推荐质量。 |
| [^147] | [OR-Transformer: Scaling Real-Time Decision-Making to 1,000 Items](https://arxiv.org/abs/2609.01933) | OR-Transformer通过商品置换等变的Transformer架构和路径梯度训练的深度强化学习方法，将随机需求下的联合补货决策扩展到1024种商品规模，随规模增长持续优于MILP基线并大幅降低在线决策时间。 |
| [^148] | [hLLM: Single Pass Decoding for Generative Reranking](https://arxiv.org/abs/2609.01807) | 提出hLLM，通过轻量自注意力头从LLM预填充隐状态读取项目-位置得分矩阵，并用匈牙利算法求最优二分匹配，在O(1)次前向传播内一次性解码全部N个序数，从而将生成式重排序的解码从逐token自回归生成变为常数次前向传播，且天然保证输出为有效排序。 |
| [^149] | [SocialBuddy: Tailoring Search Agent for Social Scenarios](https://arxiv.org/abs/2609.01641) | 该论文提出首个面向社交场景的智能体搜索框架SocialBuddy，并构建了包含20万用户画像、1000万社交帖子和5万推理轨迹的大规模模拟环境SocialEnv，以解决复杂社交搜索中的性能退化与稀疏奖励难题。 |
| [^150] | [Omega-N: Interpretable Structural Node Descriptors and Their Applicability Domain](https://arxiv.org/abs/2609.01633) | 本文提出 Omega-N，通过局部化三角指标因子并结合配置零模型超额与多尺度个性化 PageRank 邻域两项修正，得到每节点十个仅依赖图结构、无需属性与训练的可解释描述符，并验证了全局标量对标谱基线、逐节点归因在结构中心数量未知时更具优势的理论预测。 |
| [^151] | [Cheap Verifiers, Large Blind Spots: Measuring the Reliability Cost of Cost-Saving Cascades](https://arxiv.org/abs/2609.01345) | 该研究通过真实LLM实验发现，推理级联中廉价验证器对学生模型错误答案的“盲区”随学生能力增强而扩大、随验证器能力增强而缩小，恰好在级联机制赖以存在的低成本配置下最为严重，而用前沿验证器消除盲区又会因过度升级而抵消成本节约，从而揭示了成本节约级联设计背后隐藏的显著可靠性代价。 |
| [^152] | [Synthetic Worlds for Temporal Evaluation and Knowledge Updating in LLMs](https://arxiv.org/abs/2609.00184) | 该论文提出了一个模拟驱动的合成框架，通过虚构未来世界的 ParallelEvents 基准避免评估污染，并利用 Synapse 训练框架（结合中期训练与指令微调）实现大语言模型的可扩展知识更新，性能比现有方法提升 14.23%。 |
| [^153] | [Stress-Testing Efficient Responsible-AI Evaluation: When Compute Savings Change Benchmark Conclusions](https://arxiv.org/abs/2608.31108) | 该研究首次对负责任AI基准测试中的高效评估方法进行系统性压力测试，发现批处理等计算节省手段能在降低能耗的同时保持基准结论的稳定，而INT8量化虽能保持评估质量却会使能耗增至基线的1.79-4.26倍。 |
| [^154] | [A Visual Question Answering Model to Automate Nondestructive Evaluation Image Analysis](https://arxiv.org/abs/2608.29408) | 本研究提出了一种专为无损检测设计的视觉问答模型，结合ResNet-50图像特征提取与GPT-2语言生成能力，使检测人员能够通过自然语言直接查询检测图像，显著提升检测效率、减少错误并增强现场实用性。 |
| [^155] | [TACIT-Switch: Cost-Aware Model Escalation for LLM Agents from Censored Supervision](https://arxiv.org/abs/2608.27911) | 提出TACIT-Switch方法，利用教师标注的删失干预时间学习永久性移交策略，以成本感知的方式决定何时将LLM智能体从小模型骨干升级至大模型骨干，部署时无需教师参与即可将成功率提升7.4-11.1个百分点。 |
| [^156] | [Token-Level Advertising](https://arxiv.org/abs/2608.27382) | 我们提出了LAMA，一种令牌级广告拍卖机制，将广告商影响嵌入生成式AI的生成过程，同时满足激励兼容性和个体理性，并实现接近最优的福利，实验证明其能提升平台福利和收入。 |
| [^157] | [Canalization Before Generalization: Grokking as a Dynamical Probe](https://arxiv.org/abs/2608.25813) | 本文通过权重衰减脉冲在“顿悟”平台期揭示了解选择的通道化过程，表明在可见泛化之前就形成了稳定的剂量排序，为理解神经网络泛化机制提供了新的动力学视角。 |
| [^158] | [DeMMO: Longitudinal and Cross-Disease Modelling of Digital Mobility Outcomes via Multi-Task Learning](https://arxiv.org/abs/2608.25073) | DeMMO提出了一种可解释的多任务学习框架，能同时建模多种疾病的纵向数字移动结果与多临床结局，并自动捕捉跨疾病的共享和独特模式。 |
| [^159] | [Semantic Overlays: Mitigating Prompt Injection with Annotations Beyond Tokens and Steering Vectors](https://arxiv.org/abs/2608.23873) | 该论文提出了一种名为“语义覆盖层”的新技术，通过向模型输入添加非文本通道来缓解提示注入攻击，利用小型学习的适配器在冻结模型的残差流中创建带外注释，从而增强模型对片段身份的理解。 |
| [^160] | [MetaCaster: Meta-Harness-Optimized Agent for End-to-End Few-Shot Learning of Lightweight Time Series Forecasters](https://arxiv.org/abs/2608.23473) | 本文提出MetaCaster，一种元框架优化的多智能体系统，通过智能体数据生成从少量示例和文本中自动训练轻量级时间序列预测器，实现了资源受限场景下的高效少样本学习。 |
| [^161] | [Dual-Scale State-Space Modeling with Speaker-Wise Dynamic CRF for Speech Emotion Recognition in Conversation](https://arxiv.org/abs/2608.22399) | 本文提出DSSM-CRF纯音频架构，利用双向状态空间模型在帧和对话双尺度上编码语音表示，并通过说话人级动态条件随机场显式分离跨说话人上下文影响与说话人内部情感演变两种过程，从而提升会话语音情感识别的效果。 |
| [^162] | [Improving Energy Efficiency of Oil Platforms Through Optimal Loading of Diesel Generators Using Machine Learning and Search Algorithms](https://arxiv.org/abs/2608.22076) | 本研究首次将机器学习与搜索算法结合，针对海上石油平台的柴油发电机负载进行优化，以降低能源消耗而非提升产量，填补了该领域研究空白。 |
| [^163] | [Across-Design Uncertainty in Short Pricing Panels: Evidence from Simulated Price Trajectories](https://arxiv.org/abs/2608.21334) | 本文通过模拟价格轨迹证明，短定价面板中跨设计不确定性占估计误差方差的绝大部分（97.6%），并提出了一个经验关系式来描述其分散度。 |
| [^164] | [ClosureBench: A Constructive Benchmark for Compositional Graph Reasoning](https://arxiv.org/abs/2608.18242) | 本文提出ClosureBench，一个通过程序化生成实例的建设性图推理基准，能直接测量模型记忆化，并揭示模型在新鲜实例上的性能显著下降。 |
| [^165] | [Degradation-Aligned Self-Supervised Learning for State of Health Estimation of Lithium-Ion Batteries under Label Sparsity](https://arxiv.org/abs/2608.16612) | 提出一种基于排序任务的退化对齐自监督学习框架，利用未标注数据预训练CNN-GRU模型，在标签稀疏条件下实现锂离子电池健康状态的准确稳健估计。 |
| [^166] | [Boosting Data Augmentation with Stochastic Weight Averaging](https://arxiv.org/abs/2608.14373) | 本研究证明随机权重平均在增强数据上能提供超出其单独性能提升的等变性增强，且无需重复训练成本。 |
| [^167] | [Terminal Symmetry as a Decision Resource: Statewise Refinement for Anytime Verified Construction](https://arxiv.org/abs/2608.11318) | 本文提出了一种将终端对称性作为决策资源的新框架，通过传输-细化-认证机制实现任意时间验证构建，并提供了完成保证和最优验证器查询界限。 |
| [^168] | [LEED: Local Embedding Evolution Distance for over-smoothing estimation and virtual node selection in GNN](https://arxiv.org/abs/2608.09596) | 提出了一种新的局部度量方法LEED（局部嵌入演化距离），通过追踪单个节点嵌入跨层的演化来量化过平滑现象，实现节点级细粒度分析，并可用于指导图神经网络中虚拟节点的选择。 |
| [^169] | [Latent Fact-Checking: Detecting Misinformation through Activation Engineering](https://arxiv.org/abs/2608.06417) | 本文提出了一种基于激活工程的错误信息检测框架，利用语言模型表示空间中的几何方向来分类声明真伪，无需微调或外部知识。 |
| [^170] | [Deep Divide-and-Reduce in Symbolic Regression](https://arxiv.org/abs/2608.02628) | 该论文提出DDRSR方法，通过对更广泛分解结构的形式化分析，从根本上扩展了AI Feynman方法中表达式分解与归约机制的适用范围，并克服了其依赖暴力搜索的局限。 |
| [^171] | [KernelGenBench: A Multi-Source and Multi-Chip Benchmark for LLM-based Kernel Generation](https://arxiv.org/abs/2607.27231) | 该论文提出了KernelGenBench，首个统一的多源多芯片基准，覆盖210个算子和六个硬件平台，用于系统评估大语言模型与智能体生成的Triton核函数在算子来源与硬件平台之间的性能迁移能力。 |
| [^172] | [Aletheia: An Offline-First Clinical Decision Support System for Differential Diagnosis in Low-Resource Healthcare Settings](https://arxiv.org/abs/2607.24814) | Aletheia是一个面向撒哈拉以南非洲低资源医疗环境的离线优先临床决策支持系统，通过QLoRA微调Qwen2.5-3B-Instruct模型，在东非高发疾病鉴别诊断中实现了80%的Top-1准确率和100%的Top-3准确率，无需依赖互联网连接和高规格硬件。 |
| [^173] | [To Erase, or Not to Erase: Robust Training-Free Concept Erasure with Preservation aware Adaptive Ranked Subspace Expansion](https://arxiv.org/abs/2607.23492) | 本文提出PARSE框架，一种免训练的概念擦除方法，通过保留感知的自适应排序子空间扩展，解决了现有概念擦除技术中擦除鲁棒性与模型效用之间的权衡问题，实现更鲁棒且不损害良性概念效用的目标擦除。 |
| [^174] | [Not All LLM Reasoning is Visible in the Chain-of-Thought](https://arxiv.org/abs/2607.22925) | 前沿大语言模型能利用语义无关的填充token进行思维链之外的“不可见推理”来提升任务表现，甚至可以完成思维链监控完全无法察觉的隐藏目标，这对基于CoT监控的AI安全方案构成重大风险。 |
| [^175] | [Automatic knot selection in smooth additive models](https://arxiv.org/abs/2607.21083) | 本文研究B样条回归中的节点自动选择问题，指出尽管P样条正则化已成为广义加性模型的标准，但常被忽视的节点选择技术具有独特的优势。 |
| [^176] | [Statevector-to-Hardware Reconstruction of a Four-Qubit ZZ Quantum Kernel: A Single-Backend Case Study of Three Execution Jobs](https://arxiv.org/abs/2607.20377) | 该研究在单一后端上通过基线、动态解耦和门扭转三个独立执行任务，量化了四量子比特ZZ量子核格拉姆矩阵从精确态矢量参考到硬件执行的偏差，发现门扭转技术在所有报告的几何指标上偏离最小。 |
| [^177] | [Efficient Clustering with Quality Guardrails for LLM-based Recommender Systems at Industry Scale](https://arxiv.org/abs/2607.19704) | 该论文提出一种带单样本质量护栏的高效聚类方法，使LLM推荐系统能在工业规模下仅对簇代表运行LLM即可大幅降低成本，同时确保每个簇成员获得的输出既与自身相关又安全。 |
| [^178] | [Improving Weak World Models Behind Strong Agents in Atari Pong](https://arxiv.org/abs/2607.15142) | 该论文在 Atari Pong 中复现五个视觉世界模型智能体并独立评估其冻结世界模型，揭示出强智能体背后普遍存在球消失、错误运动等缺陷的弱世界模型，并提出了改进这些弱世界模型的方法。 |
| [^179] | [EvoCUA-1.5: Online Reinforcement Learning for Multi-turn Computer-Use Agents](https://arxiv.org/abs/2607.09773) | EvoCUA-1.5 将计算机使用智能体从离线经验学习扩展到在线强化学习，并提出步级策略优化（STEPO）方法，以解决多轮交互中上下文管理观察、稀疏终端奖励、可变长度轨迹和慢速环境反馈等挑战。 |
| [^180] | [Quantum Kolmogorov--Arnold representation theorem for continuous unitary-valued maps](https://arxiv.org/abs/2607.03187) | 本文为连续酉值映射建立了 Kolmogorov--Arnold 表示定理的两个量子类比版本，分别给出了反厄米值映射矩阵指数内的精确加法分解，以及考虑量子算符非对易性的因式分解形式。 |
| [^181] | [From Architecture to Output: Structural Origins of Hallucination in Large Language Models and the Amplifying Role of Data](https://arxiv.org/abs/2606.07537) | 该论文提出了一个仅需采样访问权限的幻觉归因框架，通过针对前缀、上下文和频率竞争的三次有序干预，将大语言模型的单个幻觉追溯归因到自注意力联想检索、最大似然预训练目标或暴露偏差下的自回归承诺等具体组件，并提出五个可证伪的预测。 |
| [^182] | [From Sampled Outcomes to Capability Distributions: Rethinking Supervision for LLM Routing](https://arxiv.org/abs/2606.06924) | 该论文提出DARS方法，通过语义保持的查询改写和随机解码的重复观测来估计查询级模型能力分布，构建风险感知的路由监督信号，解决了现有LLM路由中单次采样监督不稳定的问题。 |
| [^183] | [Second-order consistency for learning chaotic dynamics via randomized Jacobian matching](https://arxiv.org/abs/2606.01596) | 提出模型约束的随机雅可比匹配方法，通过在随机扰动输入处比较雅可比，隐式强制二阶（Hessian）一致性，避免学习到的混沌系统漂移向虚假吸引子并改善长时间统计特性。 |
| [^184] | [Harmless Yet Harmful: Neutral Prompting Attacks for Stealthy Hallucination Steering in Agent Skills](https://arxiv.org/abs/2605.29354) | 提出了一种高度隐蔽的“中性提示攻击”（NPA），通过鼓励想象力和穷尽性等语义无害的指令，悄然提升LLM编程智能体产生软件包幻觉的倾向，从而实现无需显式恶意意图的隐蔽软件供应链攻击。 |
| [^185] | [Robust and Efficient Guardrails with Latent Reasoning](https://arxiv.org/abs/2605.29068) | COLAGUARD通过分阶段训练将多步安全推理压缩进连续潜在空间，推理时直接传播隐状态，在宏F1上超越Llama Guard 3达8.24分，并在达到显式推理基线GuardReasoner同等性能的同时实现12.9倍加速和22.4倍的token开销降低。 |
| [^186] | [Optimal Data Acquisition for Reinforcement Learning: A Large Deviations Perspective](https://arxiv.org/abs/2605.28675) | 本文从大偏差理论出发，为无限时域强化学习建立了统一的数据获取框架，以策略选择错误概率的指数衰减率作为效率度量，并通过凸松弛和惰性单步投影次梯度方法给出了可求解的最优数据获取方案。 |
| [^187] | [Beyond Pairwise Preferences: Listwise Reward-Aware Alignment for Diffusion Models](https://arxiv.org/abs/2605.26491) | 提出 Diffusion LAIR，一种奖励感知的列表级偏好优化方法，将同一提示词下多张候选图像的奖励分数转化为中心化优势权重，突破成对比较的局限，更充分地利用连续奖励信号来对齐文本到图像扩散模型。 |
| [^188] | [SCRIPT: Scalable Diffusion Policy with Multi-stage Training for Language-driven Physics-Based Humanoid Control](https://arxiv.org/abs/2605.22894) | 提出SCRIPT——一个多阶段训练的可扩展扩散策略，其核心JAST-DiT通过联合注意力直接耦合动作、物理状态与文本，实现了基于自然语言指令对物理仿真人形机器人忠实、高质量且稳定的长时程控制。 |
| [^189] | [Less Data, Faster Training: repeating smaller datasets speeds up learning via sampling biases](https://arxiv.org/abs/2605.20314) | 该论文发现重复训练较小数据集可通过采样偏差促进恰当的逐层参数增长，从而节省计算并加速学习，这一策略可作为优化中的有利归纳偏置，尤其在推理任务中效果显著。 |
| [^190] | [Deep Learning as Neural Low-Degree Filtering: A Spectral Theory of Hierarchical Feature Learning](https://arxiv.org/abs/2605.13612) | 本文提出“神经低度滤波”这一谱理论框架，将深度网络的分层特征学习刻画为逐层进行的显式迭代谱滤波过程，为懒惰机制之外的多层特征学习提供了可解析的数学工具，能够预测逐层表示选择、概念涌现所需的样本复杂度以及深度带来的具体收益机制。 |
| [^191] | [HLS-Seek: QoR-Aware Code Generation for High-Level Synthesis via Proxy Comparative Reward Reinforcement Learning](https://arxiv.org/abs/2605.13536) | 提出HLS-Seek框架，利用帕累托支配准确率达99.53%的比较式代理奖励模型，结合不确定性感知的MC dropout切换机制按需调用真实Vitis HLS综合并在线更新代理，从而在无需完整综合在环的情况下实现QoR感知的自然语言到HLS代码生成。 |
| [^192] | [Inductive Venn-Abers and related regressors](https://arxiv.org/abs/2605.06646) | 本文通过引入共形预测，将Venn-Abers预测器从二元分类和有界回归推广到无界回归，并证明由此导出的点回归器在较大训练集上能在一定程度上提升标准回归器的预测效率。 |
| [^193] | [Inducing Permutation Invariant Priors in Bayesian Optimization for Carbon Capture and Storage Applications](https://arxiv.org/abs/2605.02409) | 本文提出了一种新颖的置换不变高斯过程核GP-Perm，通过比较集合经验表示之间的稳定散度来编码置换不变性，从而提升碳捕集与封存项目中井位布置的贝叶斯优化效率。 |
| [^194] | [Relocation of compact sets in $\mathbb{R}^n$ by diffeomorphisms and linear separability of datasets in $\mathbb{R}^n$](https://arxiv.org/abs/2604.21393) | 本文建立了通过微分同胚将R^n中有限个紧集重定位到任意目标区域的理论，并证明通过到R^{n+1}的可微嵌入或宽度为n、采用Leaky-ReLU/ELU/SELU激活函数的深度神经网络，可使这些数据集变得线性可分。 |
| [^195] | [Advancing Subseasonal Forecasting with Machine Learning](https://arxiv.org/abs/2604.16238) | 本文提出概率偏差校正（PBC）机器学习框架，通过学习校正历史概率预报大幅减少系统性误差，使ECMWF的AI预报系统的次季节（2-6周）预报技巧翻倍。 |
| [^196] | [The Geometry of Polynomial Group Convolutional Neural Networks](https://arxiv.org/abs/2603.29566) | 本文基于分次群代数为多项式群卷积神经网络提出了新的数学框架，给出了 Hadamard 和 Kronecker 两种自然参数化方法，并证明了其神经流形维度仅取决于网络层数和群的大小。 |
| [^197] | [Reservoir-Based Graph Convolutional Networks](https://arxiv.org/abs/2603.24131) | 该论文将结构化卷积机制引入基于储备池的图神经网络，在不依赖深层网络堆叠和大量参数调优的情况下实现稳定的长程信息传播，从而缓解传统GCN的过度平滑与高计算成本问题。 |
| [^198] | [Autoregressive Guidance of Deep Spatially Selective Filters using Bayesian Tracking for Efficient Extraction of Moving Speakers](https://arxiv.org/abs/2603.23723) | 本文提出了可与任意深度空间滤波器兼容的贝叶斯跟踪算法，通过自回归方式将增强后的语音信号反馈给轻量级跟踪器以引导深度空间选择性滤波器，从而实现对移动说话人语音的高效实时提取。 |
| [^199] | [YOLO with Kolmogorov-Arnold networks and vision-language foundation models for interpretable object detection with trustworthy multimodal AI in computer vision perception](https://arxiv.org/abs/2603.23037) | 该论文提出用Kolmogorov-Arnold网络作为可解释的事后代理模型，基于七个几何与语义特征评估YOLOv10检测结果的置信度可信性，并结合BLIP视觉-语言基础模型生成描述，实现计算机视觉感知中透明、可信的目标检测。 |
| [^200] | [Nepali Passport Question Answering: A Low-Resource Dataset for Public Service Applications](https://arxiv.org/abs/2603.13320) | 该研究构建了首个面向护照公共服务的尼泊尔语问答低资源数据集，通过微调Transformer嵌入模型并结合BM25进行混合检索，其中基于多语言E5嵌入的模型取得了最佳检索性能。 |
| [^201] | [The Struggle Between Continuation and Refusal: A Mechanistic Analysis of the Continuation-Triggered Jailbreak in LLMs](https://arxiv.org/abs/2603.08234) | 该论文通过注意力头层面的机制可解释性分析（因果干预与激活缩放），揭示了大语言模型中“延续触发越狱”现象的根源在于模型内在的文本延续驱动力与安全拒绝机制之间的固有竞争。 |
| [^202] | [Squint: Fast Visual Reinforcement Learning for Sim-to-Real Robotics](https://arxiv.org/abs/2602.21203) | 本文提出Squint，一种视觉Soft Actor Critic方法，通过并行仿真、分布式评论家和优化实现等技术，实现了比以往视觉离策略和同策略方法更快的实际训练速度，可高效完成仿真到现实的机器人操作任务。 |
| [^203] | [Regularity of Second-Order Elliptic PDEs in Spectral Barron Spaces](https://arxiv.org/abs/2602.19381) | 该论文证明了在温和的椭圆性条件下，$\mathbb{R}^d$ 上二阶椭圆偏微分方程的解在谱Barron空间中获得额外两阶正则性，从而确定了一类其解可被宽度与空间维度无关的两层余弦激活神经网络逼近的偏微分方程。 |
| [^204] | [Brain4FMs: A Benchmark of Foundation Models for Electrical Brain Signal](https://arxiv.org/abs/2602.11558) | 该论文提出了首个统一评估脑基础模型在EEG和iEEG两类脑电信号上表现的开放基准测试Brain4FMs，集成了17个代表性模型和21个公开数据集，覆盖临床诊断、睡眠分期、通信和情感计算四大应用领域。 |
| [^205] | [Enhancing Affine Maximizer Auctions with Correlation-Aware Payment](https://arxiv.org/abs/2602.09455) | 提出相关性感知仿射最大化拍卖（CA-AMA）框架，通过引入相关性感知支付在保持主导策略激励兼容性的同时突破经典AMA的表达能力限制，能在经典AMA表现任意差的估值相关场景下达到最优收益，并配套设计了两阶段训练算法。 |
| [^206] | [MemCoRe: Recovering Evidence from Progressively Compressed Factual Knowledge for Agent Memory](https://arxiv.org/abs/2602.07885) | 提出MemCoRe，将智能体记忆组织为压缩层级结构，在逐级压缩冗余的同时保留各层级检索所需结构，从而在记忆压缩与证据检索有效性之间实现平衡。 |
| [^207] | [Consensus Group Relative Policy Optimization for Text Generation](https://arxiv.org/abs/2602.03102) | 该论文提出C-GRPO方法，通过将最小贝叶斯风险（MBR）解码的共识效用蒸馏为GRPO框架中的组相对目标函数，在不依赖黄金参考或偏好标签的前提下，将高昂的推理时“采样-重排序”计算成本转移到训练阶段。 |
| [^208] | [Multi-Modal Time Series Prediction via Mixture of Modulated Experts](https://arxiv.org/abs/2601.21547) | 提出了一种名为“专家调制”的新机制，使混合专家模型的路由与专家计算均受文本信号调制，从而摆脱对token级融合的依赖，提升多模态时间序列预测的准确性与跨模态对齐能力。 |
| [^209] | [TeleTables: A Benchmark for Large Language Models in Telecom Table Interpretation](https://arxiv.org/abs/2601.04202) | 该论文提出TeleTables基准（包含2,220张3GPP规范表格和500道人工验证选择题），通过评估20个开源大语言模型揭示了电信表格解读的两大瓶颈：闭卷时领域知识不足导致准确率不超过41%，而提供表格上下文时准确率虽可超90%，但会随推理深度、证据范围和表格结构复杂性增加而系统性下降。 |
| [^210] | [GLOW: Graph-Language Co-Encoding for Agentic Workflow Performance Prediction](https://arxiv.org/abs/2512.15751) | GLOW 提出了一个将图神经网络的结构建模能力与大语言模型的拓扑感知语义编码能力相结合的统一框架，通过图-语言协同编码高效预测智能体工作流的性能，从而避免昂贵的基于执行的评估。 |
| [^211] | [Forecast Skill Is Not Decision Skill: Evidence from Weather-Dependent Decision Tasks](https://arxiv.org/abs/2512.14779) | 该论文提出“决策校准”框架，从决策者视角在决策层面评估概率预报，并通过对天气相关决策任务的实验发现，机器学习模型与数值天气预报模型在预报层面的性能差异并不能可靠地反映其在下游决策中的实际价值。 |
| [^212] | [Fractal and Chaotic Activation Functions in Echo State Networks: Preprocessing Topology Governs the Echo State Property](https://arxiv.org/abs/2512.14675) | 该论文发现回声状态网络中的分形与混沌等非光滑激活函数（如康托尔函数）不仅能保持回声状态特性，还能在谱半径容忍度（高达ρ=10）和收敛速度（快2.6倍）上超越传统光滑激活函数。 |
| [^213] | [Partial Inverse Design of High-Performance Concrete Using Cooperative Neural Networks for Constraint-Aware Mix Generation](https://arxiv.org/abs/2512.06813) | 本文提出一种协同神经网络框架，用于高性能混凝土的部分逆向设计，能够在单次前向传播中生成满足约束且性能一致的配合比，无需针对不同约束场景重新训练。 |
| [^214] | [Constrained Sensing and Reliable State Estimation with Shallow Recurrent Decoders on a TRIGA Mark II Reactor](https://arxiv.org/abs/2510.12368) | 本文将浅层循环解码器网络从概念反应堆设计拓展到实际部署的TRIGA Mark II研究堆，证明了该数据驱动方法能够利用稀疏、含噪的传感器数据实现可靠的反应堆状态估计。 |
| [^215] | [WaveletDiff: Multilevel Wavelet Diffusion For Time Series Generation](https://arxiv.org/abs/2510.11839) | 该论文提出WaveletDiff框架，直接在小波系数上训练扩散模型，并结合跨层级注意力机制与基于帕塞瓦尔定理的能量约束，利用时间序列固有的多分辨率结构生成高质量合成时间序列。 |
| [^216] | [Gradient-based Model Shortcut Detection for Time Series Classification](https://arxiv.org/abs/2510.10075) | 该论文首次研究并建立了时间序列分类中深度模型基于点的捷径学习问题，并提出了一种基于梯度的模型捷径检测方法，以揭示模型对训练数据内部虚假相关性的依赖。 |
| [^217] | [GraphMend: Code Transformations for Fixing Graph Breaks in PyTorch 2](https://arxiv.org/abs/2509.16248) | GraphMend 是一种通过 AST 级程序分析与语义保持验证的代码变换来自动修复 PyTorch 2 中 FX 图断裂的编译器技术，使 PyTorch 能够捕获更大、无中断的计算图而无需开发者手动重构。 |
| [^218] | [Deep Learning-Driven Peptide Classification in Biological Nanopores](https://arxiv.org/abs/2509.14029) | 该论文提出通过连续小波变换将纳米孔电阻脉冲转换为尺度图，把肽识别问题转化为图像分类任务，从而利用深度卷积模型提升生物纳米孔中肽分类的准确率。 |
| [^219] | [Towards Efficient Parametric State Estimation in Circulating Fuel Reactors with Shallow Recurrent Decoder Networks](https://arxiv.org/abs/2503.08904) | 本研究利用浅层循环解码器网络，仅凭三个堆芯外中子通量时间序列测量数据，即可高效估计循环燃料反应堆的完整状态向量（包括中子通量、先驱核浓度、温度、压力和速度），并将该架构扩展至参数化情形。 |
| [^220] | [Graph Foundation Models for Recommendation: A Comprehensive Survey](https://arxiv.org/abs/2502.08346) | 该综述首次全面梳理了图基础模型（GFM）在推荐系统中的应用，提出了现有方法的清晰分类体系，深入剖析了融合图神经网络与大语言模型优势的技术细节，并指出了该领域的关键挑战与未来研究方向。 |
| [^221] | [TSMini: A Simple Yet Highly Effective Trajectory Similarity Learning Model](https://arxiv.org/abs/2502.00285) | 提出了TSMini模型，通过子视图建模机制和基于k近邻的损失函数，同时学习轨迹的绝对相似值和相对相似排序，实现了高精度的轨迹相似性逼近。 |
| [^222] | [DeltaGNN: Graph Neural Network with Information Flow Control](https://arxiv.org/abs/2501.06002) | 本文提出DeltaGNN，通过创新的信息流分数度量实现线性复杂度的信息流控制，同时解决图神经网络中的过度平滑和过度压缩问题。 |
| [^223] | [Hyperedge Anomaly Detection with Hypergraph Neural Network](https://arxiv.org/abs/2412.05641) | 提出了一种基于超图神经网络的端到端无监督模型，能够无需标注数据即可检测超图中异常的高阶关联。 |
| [^224] | [Explainable Clustering of Mixture Models](https://arxiv.org/abs/2411.01576) | 本文首次从混合模型视角研究可解释聚类问题，给出了可解释性代价的首个数据相关上界，并针对具有次指数尾部的混合模型的K-中位数聚类提出了新算法。 |
| [^225] | [The Sample Complexity of Learning Lipschitz Operators with respect to Gaussian Measures](https://arxiv.org/abs/2410.23440) | 该论文证明了关于高斯测度的Lipschitz算子具有更高阶的高斯Sobolev正则性，给出了Hermite多项式逼近误差的上下界，并紧致刻画了从（可能自适应的）线性样本重构Lipschitz算子的样本复杂度。 |
| [^226] | [Small Molecule Optimization with Large Language Models](https://arxiv.org/abs/2407.18897) | 提出Mol-E——一种利用分子语言模型生成能力的进化算法，在实用分子优化基准测试中于任务无关和任务知情两种模式下均创造了新的最先进记录。 |
| [^227] | [Procedural Content Generation via Generative Artificial Intelligence](https://arxiv.org/abs/2407.09013) | 本综述系统回顾了生成式人工智能在程序化内容生成中的应用（涵盖地形、物品和故事情节等内容），并重点探讨了定制化内容处理、质量与多样性保证以及训练数据不足等关键挑战，以及应对这些挑战的创新生成技术、模型架构和适用于有限数据场景的方法。 |
| [^228] | [An Empirical Study into Clustering of Unseen Datasets with Self-Supervised Encoders](https://arxiv.org/abs/2406.02465) | 本研究通过将仅在ImageNet-1k上预训练的有监督与自监督编码器部署到未见数据集并进行聚类实验，发现有监督编码器在训练域内更有用，而自监督编码器在远离训练域的分布外数据上泛化能力更强，且经分类微调后这一优势会发生逆转。 |

# 详细

[^1]: UniMate：用统一模型驱动多样化骨架生成动画

    UniMate: One Unified Model to Animate Diverse Skeletons

    [https://arxiv.org/abs/2609.05415](https://arxiv.org/abs/2609.05415)

    UniMate提出一种拓扑感知的扩散Transformer，通过图感知注意力偏置、谱旋转位置编码和全局拓扑条件器三种机制将骨骼拓扑融入注意力计算，实现用单一基础模型根据文本提示为任意拓扑结构的骨架生成关节运动，无需针对每个骨架微调或测试时优化。

    

    自动绑定的最新进展已能够大规模产出可直接用于动画制作的3D资产，但生成驱动这些资产的运动仍然是一个瓶颈。现有的学习型动画方法受拓扑结构限制：它们依赖特定类别的模板，或需要在推理时进行针对每个骨架的微调以及参考动作。我们提出UniMate，这是一个统一的基础模型，它能根据已绑定的3D资产和文本提示，为任意骨架合成关节运动，无需测试时优化或针对每个骨架的重新训练。UniMate引入了一个拓扑感知的扩散Transformer，通过三种机制将骨骼拓扑整合到注意力机制中：（1）基于成对关节关系和测地距离的图感知注意力偏置；（2）通过图拉普拉斯算子将RoPE泛化到任意运动学树上的谱旋转位置编码；（3）从静止姿态骨架经注意力池化得到的全局拓扑条件器。

    arXiv:2609.05415v1 Announce Type: cross  Abstract: Recent advances in automatic rigging now deliver animation-ready 3D assets at scale, yet generating the motion to drive them remains a bottleneck. Existing learned animators are topology-constrained: they rely on category-specific templates or require per-skeleton fine-tuning and reference motions at inference. We present UniMate, a unified foundation model that synthesizes articulated motion for arbitrary skeletons from a rigged 3D asset and a text prompt, with no test-time optimization or per-skeleton retraining. UniMate introduces a topology-aware diffusion transformer, which integrates skeletal topology into attention via three mechanisms: (1) a graph-aware attention bias from pairwise joint relations and geodesic distances; (2) a spectral rotary position embedding generalizing RoPE to arbitrary kinematic trees via the graph Laplacian; and (3) a global topological conditioner attention-pooled from the rest-pose skeleton. We also cu
    
[^2]: RegionFed：面向异构零售环境中个性化查询理解的联邦学习

    RegionFed: Federated Learning for Personalized Query Understanding in Heterogeneous Retail Environments

    [https://arxiv.org/abs/2609.05403](https://arxiv.org/abs/2609.05403)

    RegionFed提出了一种完全在梯度层面运作的架构鲁棒联邦学习框架，以区域与全局梯度间的ℓ2冲突作为统一信号来诊断数据异构性并为各区域选择成本最低的充分个性化策略，解决了现有参数级个性化联邦学习方法在现代Transformer上灾难性崩溃的问题。

    

    零售搜索系统服务于不同的地理区域，这些区域具有截然不同的查询模式、词汇和产品偏好，由此产生的显著数据异构性对隐私保护训练和模型个性化都构成了挑战。联邦学习为隐私保护提供了自然的解决方案，但标准的联邦学习方法产生的是全局模型，牺牲了区域性能；而现有的个性化联邦学习方法在参数层面运作，会在现代Transformer模型上发生灾难性崩溃（在T5上准确率低于10%），这是由于共享嵌入（tied embeddings）与LayerNorm的相互作用所致。我们提出了RegionFed，一个架构鲁棒的联邦学习框架，它完全在梯度层面运作，从而避开了上述失败模式。RegionFed利用区域梯度与全局梯度之间的ℓ2冲突作为统一信号，该信号(i)诊断异构程度，(ii)将每个区域路由至成本最低且足以满足需求的个性化策略。

    arXiv:2609.05403v1 Announce Type: cross  Abstract: Retail search systems serve diverse geographic regions with distinct query patterns, vocabularies, and product preferences, creating significant data heterogeneity that challenges both privacy-preserving training and model personalization. Federated learning offers a natural solution for privacy, but standard FL methods produce global models that sacrifice regional performance, while existing personalized FL approaches operate at the parameter level and catastrophically collapse on modern transformers (below 10\% accuracy on T5) due to tied embeddings and LayerNorm interactions. We introduce RegionFed, an \textit{architecture-robust} federated learning framework that sidesteps this failure by operating entirely at the gradient level. RegionFed uses the $\ell_2$ conflict between regional and global gradients as a unified signal that (i) diagnoses heterogeneity, (ii) routes each region to the cheapest sufficient personalization strategy,
    
[^3]: 全局蒸馏，局部适配：面向可扩展换新推荐的推理蒸馏与产品类型测试时训练

    Distill Globally, Adapt Locally: Reasoning Distillation and Product-Type Test-Time Training for Scalable Trade-Up Recommendation

    [https://arxiv.org/abs/2609.05363](https://arxiv.org/abs/2609.05363)

    该论文提出一个两级框架，通过检索增强LLM教师将换新推理蒸馏到1550万参数的轻量级嵌入对分类器中，并结合产品类型的测试时训练进行局部适配，使推理时仅需预计算嵌入即可完成数亿产品对的可扩展换新推荐。

    

    换新推荐旨在识别更高质量的替代品，既保留客户的购买意图，又能提供升级的权益。大型语言模型（LLM）虽然能够对这类差异进行推理，但直接将其应用于数以亿计的产品对在运营上并不现实。我们提出了一个两级框架，将LLM的推理能力蒸馏到一个高效的非生成式学生模型中，并将其决策边界适配到特定产品类型的换新标准上。在第一级中，一个检索增强的少样本LLM教师模型生成结构化的关系标签和自然语言推理依据。这些推理依据通过对齐目标和对比目标来监督一个紧凑的嵌入对分类器；在推理阶段，学生模型仅使用两个预计算的768维产品嵌入，无需任何LLM调用或文本生成。在一个包含8,352对产品的人工标注固定基准上，一个1550万参数的四分类推理蒸馏学生模型……（摘要不完整，此处截断）

    arXiv:2609.05363v1 Announce Type: new  Abstract: Trade-up recommendation identifies higher-quality alternatives that preserve a customer's purchase intent while offering upgraded benefits. Large language models (LLMs) can reason about such distinctions, but applying them directly to hundreds of millions of product pairs is operationally impractical. We introduce a two-level framework that distills LLM reasoning into an efficient non-generative student and adapts its decision boundary to product-type-specific trade-up criteria. At Level 1, a retrieval-augmented few-shot LLM teacher generates structured relation labels and natural-language rationales. These rationales supervise a compact embedding-pair classifier through alignment and contrastive objectives; at inference, the student uses only two precomputed 768-dimensional product embeddings, with no LLM calls or text generation. On a fixed human-annotated benchmark of 8,352 pairs, a 15.5M-parameter four-class reasoning-distilled stude
    
[^4]: 双摆周期轨道的变分延拓

    Variational Continuation for Double Pendulum Periodic Orbits

    [https://arxiv.org/abs/2609.05337](https://arxiv.org/abs/2609.05337)

    该论文提出了一种基于Hessian矩阵与自动微分的变分延拓方法，无需积分器和手工推导雅可比矩阵即可高效、有引导地数值延拓动力系统中的周期轨道，并通过双摆周期振荡从不动点出发的完整延拓演示了沿轨道族的分岔与轨道族交叉检测能力。

    

    我们提出了一种基于Hessian矩阵的方法，用于对动力系统中的周期轨道进行数值延拓。我们将环路（周期轨道候选）参数化为傅里叶级数，并根据环路相对于物理微分方程的偏差定义损失函数。与以往依赖手工推导雅可比矩阵的工作不同，我们的方法通过利用自动微分（一种常见的机器学习技术）实现了该过程的自动化。延拓方向可以通过损失景观的平坦方向（即特征值为零的方向）来确定，从而使周期轨道的搜索高效且有引导性。我们的方法无需积分器，能够精确初始化不稳定不动点附近的振荡，并能高效检测轨道族交叉与次谐波分岔。作为演示，我们展示了从不动点出发的双摆周期振荡的完整延拓过程，揭示了沿轨道族出现的分岔现象。

    arXiv:2609.05337v1 Announce Type: new  Abstract: We present a Hessian-based approach to numerically continue periodic orbits in dynamical systems. A loop (periodic orbit candidate) is parametrized as a Fourier series; a loss function is defined based on the deviation of the loop from the physical differential equations. Unlike previous work relying on hand-derived Jacobians, our method automates the process by leveraging automatic differentiation, a common machine learning technique. The continuation direction can be determined by the flat directions of the loss landscapes (directions with zero eigenvalues), making the search of periodic orbits efficient and guided. Our method is integrator-free, precisely initializes oscillations around unstable fixed points, and efficiently detects orbit family intersections and subharmonic bifurcations. As a demonstration, we present full continuations of periodic double pendulum oscillations from fixed points, showing bifurcations along orbit famil
    
[^5]: 面向资源受限农业田间条件下设备端植物病害检测的轻量级视觉Transformer压缩

    Lightweight Vision Transformer Compression for On-Device Plant Disease Detection in Resource-Constrained Agricultural Field Conditions

    [https://arxiv.org/abs/2609.05334](https://arxiv.org/abs/2609.05334)

    该论文提出一种统一的视觉Transformer压缩框架，结合Hessian平衡自适应块剪枝、量化与基于注意力的知识蒸馏，实现资源受限农业设备端的高效辣椒病害检测。

    

    辣椒（Capsicum annuum）是印度经济价值最高的农作物之一，然而其产量持续受到病害威胁，而这类病害在缺乏专家介入的情况下难以识别。尽管视觉Transformer（ViT）已达到很高的分类准确率，但其庞大的计算开销使得在资源受限设备上的部署面临挑战。现有的压缩方法通常将剪枝、量化和知识蒸馏孤立地处理，对其组合应用的潜在收益与相互作用探索不足。我们提出了一种统一的视觉Transformer压缩框架，将基于二阶敏感度估计指导的Hessian平衡自适应块剪枝（H-BAC）与量化以及基于注意力的知识蒸馏相结合。为了系统地识别每种压缩技术族中最有效的配置，每种技术首先被评估……（原文摘要在此处截断）

    arXiv:2609.05334v1 Announce Type: cross  Abstract: Chilli (Capsicum annuum) is one of India's most economically significant crops, yet its productivity is persistently threatened by diseases that are difficult to identify without expert intervention. While Vision Transformers (ViTs) have achieved high classification accuracy, their large computational footprint makes deployment on resource constrained devices challenging. Existing compression approaches typically address pruning, quantization, and knowledge distillation in isolation, leaving the potential benefits and interactions of their combined application insufficiently explored. We propose a unified Vision Transformer compression framework that combines Hessian-Balanced Adaptive Block Pruning (H-BAC), guided by second-order sensitivity estimation, with quantization and attention-based knowledge distillation. To systematically identify the most effective configuration within each compression family, each technique is first evaluat
    
[^6]: 用于分类图生成的嵌入式图流

    Embedded Graph Flows for Categorical Graph Generation

    [https://arxiv.org/abs/2609.05328](https://arxiv.org/abs/2609.05328)

    提出嵌入式图流（EGF）生成模型，通过为节点和无序边类别学习连续嵌入，并利用置换等变图变换器将高斯噪声输运到这些端点，从而摆脱独热编码的人为几何限制，在QM9等分子图生成基准上显著超越DiGress和GruM等基线方法（FCD低至0.150）。

    

    生成分类图需要选择能够形成连贯结构的节点类型和边类型，且不依赖于节点顺序。许多图生成器将类别编码为固定的独热向量，这可能强加一种人为的几何结构，使各类别之间距离相等。我们提出了嵌入式图流，这是一种生成模型，它为节点和无序边类别学习连续嵌入，并利用置换等变的图变换器将高斯噪声输运到这些学习到的端点上。终端读出层再将嵌入映射回离散的图类别。在多个分子基准测试中，EGF取得了具有竞争力的性能。在QM9数据集上，EGF在三种方法的全部四项报告指标中均取得最佳结果，其中Fréchet ChemNet距离（FCD）为0.150，而分类扩散基线DiGress为0.717，基于桥接的基线GruM为0.812。当应用于更大的分子时……

    arXiv:2609.05328v1 Announce Type: new  Abstract: Generating categorical graphs requires choosing node and edge types that form a coherent structure without depending on node order. Many graph generators encode categories as fixed one-hot vectors, which can impose an artificial geometry in which categories are equidistant. We propose Embedded Graph Flows (EGF), a generative model that learns continuous embeddings for node and unordered-edge categories and transports Gaussian noise towards these learnt endpoints using a permutation-equivariant graph transformer. A terminal readout maps the embeddings back to discrete graph categories. Across molecular benchmarks, EGF achieved competitive performance. On QM9, EGF gives the best result on all four reported metrics among the three methods, including a Fr\'echet ChemNet Distance (FCD) of 0.150, compared with 0.717 for the categorical-diffusion baseline DiGress and 0.812 for the bridge-based baseline GruM. When applied to larger molecules in 
    
[^7]: 面向低分辨率与资源受限环境的自适应门控深伪检测

    Adaptive Gated Deepfake Detection for Low-Resolution and Resource-Constrained Environments

    [https://arxiv.org/abs/2609.05320](https://arxiv.org/abs/2609.05320)

    本文提出 AdaGate-DF 自适应门控深伪检测框架，利用图像质量线索通过双多出口系统路由样本，让高质量图像提前退出以节省算力，在低分辨率和资源受限环境下实现了优于现有方法的高效检测。

    

    深伪检测模型通常依赖于高质量输入、固定的推理路径以及计算开销高昂的网络架构，这限制了它们在低分辨率和资源受限场景中的应用。本文提出了 AdaGate-DF，一种自适应门控深伪检测框架，它利用图像质量线索将样本路由至一个双多出口系统中，使高质量图像能够提前退出，从而节省计算资源。我们在两个基准数据集和多种配置下，将 AdaGate-DF 与 MaD-CoRN、DefakeHop++ 和 ShuffleNetV2 进行了对比评估，以测试其对图像分辨率的依赖性以及训练和推理效率。在 Celeb-DF 数据集上，AdaGate-DF 达到了 0.9370 的 AUC，优于 MaD-CoRN 和 DefakeHop++，同时保持了较低的推理延迟。基于分辨率的测试表明，随着输入分辨率的提高，性能持续提升，在 384×384 分辨率下 AUC 达到 0.9708。FaceForensics++ 的结果强调了……（原文截断）

    arXiv:2609.05320v1 Announce Type: cross  Abstract: Deepfake detection models often rely on high-quality inputs, fixed inference paths, and computationally expensive architectures, limiting their use in low-resolution and resource-constrained settings. This paper proposes AdaGate-DF, an adaptive gated deepfake detection framework that uses image-quality cues to route samples through a dual multi-exit system so high-quality images can exit earlier and save compute. We evaluated AdaGate-DF against MaD-CoRN, DefakeHop++, and ShuffleNetV2 on two benchmark datasets (Celeb-DF and FaceForensics++) under multiple configurations to test image resolution dependence and training and inference efficiency. On Celeb-DF, AdaGate-DF achieves an AUC of 0.9370, outperforming MaD-CoRN and DefakeHop++ while maintaining a low inference latency. Resolution-based testing shows consistent improvement as input resolution increases, reaching an AUC of 0.9708 at 384 by 384. The FaceForensics++ results highlight t
    
[^8]: 智能体网络化信息聚合的最优速率

    Optimal Rates for Agentic Networked Information Aggregation

    [https://arxiv.org/abs/2609.05318](https://arxiv.org/abs/2609.05318)

    本文基于 Kearns 等人（SODA'26）的开创性工作，研究了智能体在网络中各自只能看到部分数据、只传递自身结论的信息聚合学习模型，并证明了此类模型下最后一个智能体超额均方误差的最优收敛速率。

    

    基于 Kearns、Roth 和 Ryu（SODA'26）的开创性论文，我们研究了网络化学习模型中的信息聚合问题。该模型捕捉了智能体 AI 中的一个核心模式：每个智能体只能看到数据的一部分，并且只传递自己的结论。他们的模型考虑了具有均方误差（MSE）损失的线性回归问题。智能体位于一个有向无环图（DAG）中，每个智能体只能看到特征的一个子集及其父节点的预测，拟合一个线性预测器，并且只向前传递其预测结果。基准是比较对象是能够看到所有原始特征的全特征学习者。如果深度为 $D$ 的路径上每 $M$ 个连续智能体 collectively（组合起来）能够看到所有原始特征，则称该路径是 $M$-覆盖的。Kearns、Roth 和 Ryu 证明了在此类路径上最后一个智能体的超额均方误差为 $O(M/\sqrt{D})$，并给出了一个循环实例，当 $D < M^2$ 时其超额误差为 $\Omega(M/D)$。我们还证明了对于任何固定分布，超额误差会收……（原文摘要在此处截断）

    arXiv:2609.05318v1 Announce Type: new  Abstract: Building on the pioneering paper of Kearns, Roth, and Ryu (SODA'26), we study information aggregation in a networked learning model. The model captures a central pattern in agentic AI: each agent sees only part of the data and passes on only its own conclusion. Their model considers a linear regression problem with the mean squared error (MSE) loss. Agents sit in a DAG and each sees only a subset of the features and its parents' predictions, fits a linear predictor, and passes only its prediction forward. The benchmark is the full-feature learner that sees all raw features. A path of depth $D$ is $M$-covered if every block of $M$ consecutive agents collectively sees all raw features. Kearns, Roth, and Ryu proved that the excess mean squared error of the last agent on such a path is $O(M/\sqrt D)$, and gave a cyclic instance with excess error $\Omega(M/D)$ for $D< M^2$. We also show that for any fixed distribution the excess error contrac
    
[^9]: mHC 如何使用其残差流？选择性路由与近恒等混合

    How Does mHC Use Its Residual Streams? Selective Routing and Near-Identity Mixing

    [https://arxiv.org/abs/2609.05309](https://arxiv.org/abs/2609.05309)

    本文通过分析 DeepSeek-V4-Flash 的四流残差通路，发现 mHC 模型采用集中且随深度变化的选择性读写路由（每个模块约有效使用两个流），残差混合主要限于早期层、深层几乎恒等地传递各流，且干预实验证明各流承载着方向上不同、功能上重要的表示。

    

    超连接及其流形约束变体 mHC 将残差通路从单一流扩展为 n 个流，然而训练后的模型如何利用这种容量仍不清楚：各模块读写的范围有多广，残差通路混合各流的强度有多大，以及这些流是否承载着不同的表示。我们使用有效流数量、跨流残差权重和流间余弦相似度，在 DeepSeek-V4-Flash 的四流残差通路中考察了这些性质。读/写路由是集中的，但随深度而变化：一个典型的注意力或 FFN 站点实际上只有效使用约两个流，主导流会随层而改变，且各表示在方向上保持明显区分。残差混合较为温和，主要发生在早期层；在第 22-42 层中，通路大多将每个流单独向前传递。针对性干预确立了这些模式在功能上的重要性。替换……（原文摘要在此处截断）

    arXiv:2609.05309v1 Announce Type: cross  Abstract: Hyper-Connections and their manifold-constrained variant mHC widen a residual pathway from one stream to n, yet how trained models use this capacity remains unclear: how broadly blocks read and write, how strongly the residual pathway mixes streams, and whether the streams carry distinct representations. We examine these properties in the four-stream residual pathway of DeepSeek-V4-Flash using effective stream counts, cross-stream residual weights, and inter-stream cosine similarity. Read/write routing is concentrated but varies across depth: a typical attention or FFN site effectively uses about two streams, while the dominant stream changes across layers and the representations remain directionally distinct. Residual mixing is modest and occurs primarily in early layers; in layers 22-42, the pathway mostly carries each stream forward separately. Targeted interventions establish the functional significance of these patterns. Replacing
    
[^10]: 面向合作式多智能体强化学习的在线变点检测

    Online Change-point Detection for Cooperative Multi-Agent Reinforcement Learning

    [https://arxiv.org/abs/2609.05298](https://arxiv.org/abs/2609.05298)

    提出了一种轻量级、与算法无关的在线变点检测方法PPR，通过对智能体回报流进行平滑处理并应用统计漂移检测，使合作式多智能体强化学习系统能够在环境或任务目标发生变化时及时识别情况改变。

    

    合作式多智能体强化学习（MARL）系统依赖过去的经验来学习协调行为，但如果环境或任务目标在训练过程中发生变化，这些经验可能变得不可靠。在这种情况下，智能体首先需要一种识别情况已经发生变化的方法，然后再决定如何进行适应。本文研究了利用奖励衍生信号对合作式多智能体强化学习进行在线变点检测。我们提出了“过往奖励模式”（Patterns of Past Rewards，PPR），这是一种轻量级、与算法无关的检测器，它对智能体的回报流进行平滑处理，突出近期变化，并应用统计漂移检测器来标记显著的转变。我们在基于多智能体粒子环境（Multi-Agent Particle Environment）构建的自定义“说话者-聆听者”（Speaker-Listener）环境中，在两种受控的非平稳性场景下对PPR进行了评估。我们的结果表明检测速度与警报稳定性之间存在权衡。一个平滑回报的基线方法能够更早地检测到变化，但会产生许多……（原文摘要在此处截断）

    arXiv:2609.05298v1 Announce Type: cross  Abstract: Cooperative multi-agent reinforcement learning (MARL) systems rely on past experience for learning coordinated behaviour, but this experience may become unreliable if the environment or task objective changes during training. In such cases, agents first need a way to recognize that the situation has changed before deciding how to adapt. This paper studies online change-point detection for cooperative MARL using reward-derived signals. We propose \emph{Patterns of Past Rewards} (PPR), a lightweight algorithm-agnostic detector that smooths agents' return streams, highlights recent changes, and applies a statistical drift detector to flag significant shifts. We evaluate PPR in a custom Speaker-Listener environment based on the Multi-Agent Particle Environment under two controlled non-stationarity scenarios. Our results show a trade-off between detection speed and alarm stability. A smoothed-return baseline detects earlier but produces man
    
[^11]: LexFlip：法律含义保持度量的解离诊断方法

    LexFlip: A Dissociation Diagnostic for Legal Meaning Preservation Metrics

    [https://arxiv.org/abs/2609.05296](https://arxiv.org/abs/2609.05296)

    本文提出LexFlip诊断集，通过373个保持词汇表面形式不变却逆转法律效力的魁北克法语法条最小扰动，揭示了现有嵌入类语义度量几乎无法察觉法律含义变化，暴露了当前法律文本简化评估中“相同对检验”的根本缺陷。

    

    简化后的法律条款是否仍然表达了原文的含义？目前使用的检验方法无法证明这一点：要求相同句对得分最高、无关句对得分最低的检验方式，会使词汇重叠与法律效力一同变动，因此任何词元重叠的单调函数都能同时满足这两项要求。我们的补救方案是“解离”——即表面形式保持不变而法律效力发生变化的项目。我们发布了LexFlip，包含373个针对魁北克省法语成文法的最小扰动，这些扰动在保留0.93词元的同时逆转了法律效力，并配套一个评分框架，可对度量指标、回归器和提示式评判模型进行评测。我们测试的七个嵌入类和BERTScore度量在此类编辑上仅消耗其相同对至无关对得分区间的0.022至0.039，相比之下双向NLI为0.670——而双向NLI恰恰是相同句对检验会淘汰的唯一一类方法。在FrJudge上，相对于实测的人类上限r=0.597，一个简单的长度特征胜过了所有语义度量指标，并且具有最低的（原文在此处截断）

    arXiv:2609.05296v1 Announce Type: new  Abstract: Does a simplified legal clause still say what the original said? The checks in current use cannot establish that it does: requiring an identical pair to score highest and an unrelated pair lowest moves lexical overlap and legal force together, so any monotone function of token overlap satisfies both. Our remedy is a dissociation, an item holding surface form fixed while legal force moves. We release LexFlip, 373 minimal perturbations of Quebec statutory French that reverse legal force while preserving 0.93 of the tokens, with a harness scoring metrics, regressors and prompted judges alike. The seven embedding and BERTScore metrics we test spend only 0.022 to 0.039 of their identical-to-unrelated range on such an edit, against 0.670 for bidirectional NLI, the one family the identical-pair check would disqualify. On FrJudge, against a measured human ceiling of r=0.597, a bare length feature outscores every semantic metric and has the lowes
    
[^12]: 从VAE误差中学习以支持基于心电图的心肌瘢痕鉴别诊断

    Learning from VAE Errors to support ECG-based Differential Diagnosis of Myocardial Scar

    [https://arxiv.org/abs/2609.05294](https://arxiv.org/abs/2609.05294)

    本研究提出利用β-VAE的DTW重构误差来辅助基于心电图的LGE心肌瘢痕鉴别诊断，发现重构误差在12个导联中的10个导联上能显著区分瘢痕阳性与阴性患者，为常规心电图筛查心肌瘢痕提供了新思路。

    

    心脏磁共振上的延迟钆强化是心肌瘢痕的关键标志物，但其有限的可及性促使人们寻求基于常规心电图的筛查方法。我们在一个包含300名受试者的本地队列中，评估了β-变分自编码器（VAE）推导的心电图表征能否区分LGE阳性与LGE阴性的心肌病患者。我们将来自基础模型ECGx.AI的32维特征与在正常PTB-XL心电图上训练的更浅层β-VAE的特征进行了比较，评估了下游分类性能以及基于动态时间规整（DTW）的重构误差。ECGx.AI结合随机森林达到了0.686的ROC曲线下面积，而所提出的β-VAE结合梯度提升达到了0.577，敏感度为0.775。值得注意的是，根据Mann-Whitney U检验，DTW重构误差在12个导联中的10个导联上在两类患者之间存在显著差异，并有助于分类任务，结合逻辑回归达到0.643的ROC曲线下面积。

    arXiv:2609.05294v1 Announce Type: new  Abstract: Late Gadolinium Enhancement (LGE) on cardiac magnetic resonance is a key marker of myocardial scar, but its limited accessibility motivates routine ECG-based screening. We evaluated whether $\beta$-variational autoencoder (VAE)-derived ECG representations can discriminate LGE+ from LGE- cardiomyopathic patients in a local cohort of 300 subjects. We compared 32-dimensional features from the foundation ECGx.AI model with those from a shallower $\beta$-VAE trained on normal PTB-XL ECGs, evaluating downstream classification and Dynamic Time Warping (DTW)-based reconstruction errors. ECGx.AI reached an area under ROC of 0.686 with Random Forest, while the proposed $\beta$-VAE reached 0.577 with sensitivity of 0.775 with Gradient Boosting. Notably, DTW-reconstruction errors significantly differed between classes in 10 out of 12 leads according to Mann-Whitney U test and help in classification, leading to an area under ROC of 0.643 with Logisti
    
[^13]: 如何在智能体编程中推测不确定性？一种草稿模型门控方法

    How to Speculate about Uncertainty in Agentic Coding? A Draft-Model Gate Method

    [https://arxiv.org/abs/2609.05274](https://arxiv.org/abs/2609.05274)

    提出推测不确定性（SU）方法，利用小型开源草稿模型在单次前向传播中对黑盒智能体已生成的输出轨迹进行评分，无需访问模型内部信息即可预测失败，并通过执行前否决门控等下游策略降低智能体编程的失败成本。

    

    部署于软件工程的大语言模型智能体失败代价高昂：它们会自信地做出错误行为，而这些错误行为只有在代价高昂的执行与重试之后才被识别。我们提出了推测不确定性，这是一种仅从黑盒智能体的输出词元中恢复预测性失败信号的方法，无需访问logits、权重、激活值，也无需重复采样。该方法反转了推测解码的思路，用一个小的开源权重草稿模型在单次前向传播中对智能体已生成的轨迹进行评分。从这些推测性交叉似然中，我们通过分离推理片段与动作片段来提取阶段感知特征，并针对可验证的目标进行校准。SU会产生一个失败似然分数，任何下游策略，例如路由、人工干预或额外的测试时计算——都可以直接使用该分数。为证明该信号切实可用，我们在软件工程任务上实例化了其中一种策略，即执行前否决门控。

    arXiv:2609.05274v1 Announce Type: new  Abstract: LLM agents deployed for software engineering fail expensively: they act confidently wrong, and bad actions are recognized only after costly execution and retry. We present Speculative Uncertainty (SU), a method that recovers a predictive failure signal for a black-box agent from its output tokens alone, with no access to logits, weights, activations, or repeated sampling. Inverting speculative decoding, a small open-weight draft model scores the agent's already-generated trajectory in a single forward pass. From these speculative cross-likelihoods we extract phase-aware features by separating the reasoning and action spans, and calibrate them against a verifiable objective. SU produces a failure-likelihood score that any downstream policy, such as routing, human intervention, or extra test-time compute, can consume directly. To show the signal is actionable, we instantiate one such policy, a pre-execution veto gate, on software engineeri
    
[^14]: 混合Sobolev空间中的浅层神经网络逼近

    Shallow neural network approximation in mixed Sobolev spaces

    [https://arxiv.org/abs/2609.05263](https://arxiv.org/abs/2609.05263)

    该论文建立了与激活函数无关的傅里叶块原理，证明浅层神经网络对混合光滑度为 $\alpha$ 的函数的逼近代数阶为 $\min\{\alpha,\rho\}$，并通过匹配的下界确定了 $\mathrm{ReLU}^k$ 网络在任意维数下的最优代数逼近指数为 $\min\{\alpha,k+1\}$。

    

    我们研究了具有 $n$ 个神经元和一般激活函数的浅层神经网络对混合 Sobolev 空间的最优 $L_2$ 逼近。我们首先建立了一个与激活函数无关的傅里叶块原理：如果某激活函数在傅里叶块性质的意义下具有单变量逼近阶 $\rho$，那么对于混合光滑度为 $\alpha$ 的目标函数，全局逼近速率具有代数阶 $\min\{\alpha,\rho\}$，直至显式的对数因子。为了针对具体激活函数验证该性质，我们引入了一个结构化的单变量逼近条件，该条件以显式参数蕴含傅里叶块性质。对于 $\mathrm{ReLU}^k$，一个匹配的代数下界确定了 $\min\{\alpha,k+1\}$ 为任意维数下的最优代数逼近指数，上界中存在对数因子。该框架还对基数 B 样条给出了指数 $\min\{\alpha,k+1\}$，并……（原文在此截断）

    arXiv:2609.05263v1 Announce Type: cross  Abstract: We investigate the best $L_2$ approximation of mixed Sobolev spaces by shallow neural networks with $n$ neurons and general activation functions. We first establish an activation-independent Fourier-block principle: if an activation has univariate approximation order $\rho$ in the sense of the Fourier-block property, then the global approximation rate has algebraic order $\min\{\alpha,\rho\}$ for target functions of mixed smoothness $\alpha$, up to explicit logarithmic factors. To verify this property for concrete activations, we introduce a structured univariate approximation condition that implies the Fourier-block property with explicit parameters. For $\mathrm{ReLU}^k$, a matching algebraic lower bound identifies $\min\{\alpha,k+1\}$ as the optimal algebraic approximation exponent in any dimension, up to logarithmic factors in the upper bound. The framework also yields the exponent $\min\{\alpha,k+1\}$ for cardinal B-splines and so
    
[^15]: GLASS：基于球面评分的图-语言对齐实现可迁移的图级异常检测

    GLASS: Graph-Language Alignment with Spherical Scoring for Transferable Graph-Level Anomaly Detection

    [https://arxiv.org/abs/2609.05253](https://arxiv.org/abs/2609.05253)

    GLASS框架通过在单位超球面上对齐图编码器与文本嵌入，并利用von Mises-Fisher核密度估计进行球面多模态评分，实现了具有跨域可迁移性的图级异常检测。

    

    我们提出了GLASS，一个用于图级异常检测（GLAD）的框架，它通过在单位超球面上进行图-语言对齐来实现强大的跨域可迁移性。GLASS通过多切片软余弦目标，将结构感知的图编码器与指令感知的文本嵌入对齐，从而构建统一表示空间。我们的框架将局部、全局和语义图属性序列化为紧凑的图描述符提示（GraphDP），创建了一个文本桥梁，实现了与领域无关的异常评分。通过Matryoshka表示切片强制执行多尺度一致性，模型能够在多个粒度级别上捕获异常偏差。在评分方面，我们将异常检测形式化为对齐超球面上的密度估计问题，并引入球面多模态评分（SMS），在图和文本嵌入空间中实例化von Mises-Fisher核密度估计器。

    arXiv:2609.05253v1 Announce Type: new  Abstract: We introduce GLASS, a framework for graph-level anomaly detection (GLAD) that achieves robust cross-domain transferability through graph-language alignment on the unit hypersphere. GLASS builds a unified representation space by aligning a structure-aware graph encoder with an instruction-aware text embedding via a multi-slice soft cosine objective. Our framework serializes local, global, and semantic graph properties into a compact Graph Descriptor Prompt (GraphDP), creating a text bridge that enables domain-agnostic anomaly scoring. By enforcing multi-scale consistency through Matryoshka representation slices, the model captures anomalous deviations at multiple levels of granularity. For scoring, we formulate anomaly detection as density estimation on the aligned hypersphere and introduce Spherical Multi-Modal Scoring (SMS), which instantiates von Mises-Fisher kernel density estimators in both graph and text embedding spaces. This proba
    
[^16]: 基于 Zynq UltraScale+ MPSoC 的开源机器学习加速器的质子辐照特性表征

    Proton Irradiation Characterization of an Open-Source ML Accelerator on a Zynq UltraScale+ MPSoC

    [https://arxiv.org/abs/2609.05249](https://arxiv.org/abs/2609.05249)

    本论文通过对部署在 Zynq UltraScale+ SoC 上执行 ResNet-20 推理的开源 Tensil 神经网络加速器进行 20-58 MeV 质子辐照实验，首次为其建立了系统级辐射响应基线，填补了开源 RTL 可访问加速器在实证辐射表征方面的空白，为开发可验证的空间辐射缓解策略奠定了基础。

    

    随着星载计算系统日益依赖神经网络（NN）加速器，商用黑盒架构的不透明性严重限制了可验证辐射缓解策略的开发。开源的、寄存器传输级（RTL）可访问的加速器通过支持用户自定义的检测手段解决了这一局限，但很少有加速器具备实证的辐射响应基线。本工作为部署在 Zynq UltraScale+ SoC 上执行 ResNet-20 推理的未加固开源 Tensil 神经网络加速器建立了基础性的系统级质子辐照基线。在 20 至 58 MeV 的质子辐照下，研究团队在受监控的运行窗口内注入了 4.29×10¹⁰ p/cm² 的注量。实验期间共发生七次工作负载中断，需要两次重启笔记本进程、四次重启或板卡复位以及一次断电重启序列。此外还发生了两次输出损坏事件，在不损失服务的情况下返回了错误的 CIFAR-10 分类结果。

    arXiv:2609.05249v1 Announce Type: cross  Abstract: As spaceborne computing systems increasingly rely on neural network (NN) accelerators, the opacity of commercial, black-box architectures severely restricts the development of verifiable radiation mitigation strategies. Open-source, register-transfer level (RTL)-accessible accelerators resolve this limitation by enabling user-defined instrumentation, yet few have empirical radiation-response baselines. This work establishes a foundational system-level proton-irradiation baseline for an unmitigated open-source Tensil NN accelerator deployed on a Zynq UltraScale+ SoC executing ResNet-20 inference. Under 20 to 58 MeV proton irradiation, we delivered $4.29 \times 10^{10}$ p/cm$^{2}$ within monitored operational windows. Seven workload interruptions required two restarts of the notebook process, four reboots or board resets, and one power-cycle sequence. Two output-corruption events returned incorrect CIFAR-10 classes without loss of servic
    
[^17]: PRICE：比特币价格预测中大语言模型适配选择的系统性研究

    PRICE: A Systematic Study of LLM Adaptation Choices for Bitcoin Price Forecasting

    [https://arxiv.org/abs/2609.05235](https://arxiv.org/abs/2609.05235)

    该研究提出PRICE框架，通过联合优化LoRA微调、递归多步推理、整数舍入数值表示、CTF提示和零温度解码五项适配选择，系统性地将4位量化的LLaMA-3 8B大语言模型高效适配于比特币短期价格预测任务。

    

    加密货币市场表现出极端的波动性和非平稳动态特性，这对传统预测方法构成了挑战。尽管大语言模型（LLMs）在时间序列预测方面展现出潜力，但各类适配选择的综合影响在金融领域仍在很大程度上未被探索。本研究提出了PRICE，这是一种将大语言模型适配于短期比特币价格预测的结构化方法。PRICE基于4位量化的LLaMA-3 8B模型构建，研究了微调、数值表示、提示工程、推理和解码方式如何共同影响预测性能。PRICE集成了以下技术：基于低秩适应（LoRA）的参数高效微调、递归多步推理、整数舍入数值表示、上下文-任务-格式（CTF）提示以及精确零温度解码。消融实验表明，每个组件都对预测的准确性和可靠性有所贡献。LoRA实现了高效的……

    arXiv:2609.05235v1 Announce Type: cross  Abstract: Cryptocurrency markets exhibit extreme volatility and non-stationary dynamics that challenge conventional forecasting methods. Although Large Language Models (LLMs) have shown promise for time series forecasting, the combined effects of adaptation choices remain largely unexplored in financial settings. This study introduces PRICE, a structured approach for adapting LLMs to short-term Bitcoin price forecasting. Built on a 4-bit quantized LLaMA-3 8B model, PRICE investigates how fine-tuning, numerical representation, prompting, inference, and decoding jointly influence forecasting performance. PRICE integrates Parameter-efficient fine-tuning with Low-Rank Adaptation (LoRA), Recursive multi-step inference, Integer-rounded numerical representation, Context-Task-Format (CTF) prompting, and Exact zero-temperature decoding. Ablation studies show that each component contributes to forecasting accuracy and reliability. LoRA enables efficient t
    
[^18]: 基于Hessian的分子构象增强：一种可扩展且高效的机器学习原子间势策略

    Hessian-based molecular conformation augmentation for a scalable and efficient strategy of machine learning interatomic potentials

    [https://arxiv.org/abs/2609.05233](https://arxiv.org/abs/2609.05233)

    该论文提出两种基于Hessian矩阵的数据增强方案（UniAug和ModeAug），通过简单的泰勒展开生成增强分子构型，无需修改模型架构或引入额外计算与内存开销，即可有效利用Hessian信息训练机器学习原子间势。

    

    虽然机器学习原子间势（MLIPs）已成功学习了势能面（PES）和原子力，但许多实际应用，如振动分析和过渡态搜索，严重依赖于势能面的Hessian矩阵。然而，标准MLIPs通常仅在能量和力上进行训练，使得Hessian信息在很大程度上未被利用。与此同时，现有将Hessian显式纳入训练目标的方法需要对模型架构进行修改，并且由于高阶反向传播而引入了显著的计算和内存开销。为了解决这些局限性，我们提出了两种基于Hessian的数据增强方案：各向同性高斯位移和简正模式加权位移。两种方法都利用简单的泰勒展开，在不改变训练目标或扩展自动微分图的情况下实现有效增强，这使得

    arXiv:2609.05233v1 Announce Type: new  Abstract: While machine-learning interatomic potentials (MLIPs) have successfully learned potential energy surfaces (PES) and atomic forces, many practical applications, such as vibrational analysis and transition state search, rely heavily on the PES Hessian. Yet, standard MLIPs tend to be trained on energy and forces alone, leaving Hessian information largely unexploited. Meanwhile, existing methods that explicitly incorporate the Hessian into training objectives require architectural modifications and introduce significant computational and memory overheads due to higher-order backpropagation. To address these limitations, we propose two Hessian-derived data augmentation schemes: isotropic Gaussian displacement (\textbf{UniAug}) and normal mode-weighted displacement (\textbf{ModeAug}). Both methods utilize simple Taylor expansions, achieving effective augmentation without altering training objectives or extending the autograd graph. This allows
    
[^19]: FedDRAW：用于异构多机构胸片分类的联邦双声誉退火加权方法

    FedDRAW: Federated Dual Reputation Annealing Weighting for Heterogeneous Multi-Institutional Chest Radiograph Classification

    [https://arxiv.org/abs/2609.05223](https://arxiv.org/abs/2609.05223)

    提出FedDRAW服务器端聚合方法，通过双声誉退火加权机制，解决联邦平均按样本量加权导致小而信息丰富的医院被边缘化、大型客户端可能主导全局模型的问题，用于异构多机构胸片分类。

    

    人工智能模型在医学诊断中前景广阔，但其需要大量无偏数据，而这些数据在医学领域分散于各医院，为保护患者隐私无法集中存储。联邦学习（FL）解决了这一问题：各医院在患者数据保留在本地的同时，共同训练一个共享的诊断模型。训练以通信轮次进行，每家医院在本地训练共享模型后将其返回服务器，通过加权平均进行合并。这种聚合权重决定了哪些机构的知识会塑造最终结果。联邦平均算法按本地样本数量成比例地设置该权重，因此规模小但信息丰富的医院会被永久性地赋予较小的影响力，而数据量最大的客户端即使信息量较低也可能主导全局模型。我们提出了联邦双声誉退火加权方法（FedDRAW），这是一种服务器端聚合方法，结合了数据……

    arXiv:2609.05223v1 Announce Type: new  Abstract: Artificial intelligence models are promising for medical diagnosis, but they require large numbers of unbiased data, which in medicine are distributed across hospitals and cannot be centralized to protect patient privacy. Federated Learning (FL) addresses this, since hospitals train one shared diagnostic model while patient data remain local. Training proceeds in communication rounds, in which each hospital trains the shared model locally and returns it to the server for merging by weighted average. This aggregation weight determines whose institutional knowledge shapes the result. Federated averaging (FedAvg) sets it in proportion to local sample count, so a small but informative hospital is permanently assigned a small influence, andl argest clients could dominate the global model even when they are less informative. We propose Federated Dual Reputation Annealing Weighting (FedDRAW), a server-side aggregation method that combines a dat
    
[^20]: 一种验证器引导的可解释推理框架：结合黄金锚定QLoRA、任务感知混合专家与组相对RLVR

    A Verifier-Guided Explainable Reasoning Framework with Gold-Anchored QLoRA, Task-Aware Mixture-of-Experts, and Group-Relative RLVR

    [https://arxiv.org/abs/2609.05221](https://arxiv.org/abs/2609.05221)

    该论文提出一种面向教育问答的可解释推理框架，通过黄金锚定QLoRA微调、任务感知符号路由（逻辑问题交由FOL/Z3验证器、物理问题交由公式与单位感知求解器）以及组相对RLVR强化学习，从正确性、一致性和推理深度三个维度提升大模型推理的可验证性与可解释性。

    

    大型语言模型（LLMs）展现出强大的推理能力，但其解释可能仍然不一致、依据薄弱或难以验证。我们提出了一种面向透明教育问答的验证器引导可解释推理框架，该框架结合了黄金锚定QLoRA、任务感知符号路由和组相对RLVR。首先使用以权威答案为锚定的字段加权QLoRA监督对Qwen2.5-3B-Instruct模型进行适配。随后，一个轻量级路由器将逻辑问题分配给FOL/Z3验证器，将物理问题分配给公式与单位感知的符号求解器。验证器反馈进一步用于支持RLVR过程中的候选评估、自我修订和奖励构建。候选响应沿三个互补维度进行评估：P1评估答案正确性，P2评估证据或单位一致性，P3评估推理深度和可解释性。在推理阶段，无需黄金答案的自我一致性机制对多个候选进行聚合……

    arXiv:2609.05221v1 Announce Type: cross  Abstract: Large language models (LLMs) show strong reasoning ability, but their explanations can remain inconsistent, weakly grounded, or difficult to verify. We propose a verifier-guided explainable reasoning framework for transparent educational question answering that combines gold-anchored QLoRA, task-aware symbolic routing, and group-relative RLVR. Qwen2.5-3B-Instruct is first adapted with field-weighted QLoRA supervision anchored to authoritative answers. A lightweight router then assigns logic problems to a FOL/Z3 verifier and physics problems to a formula- and unit aware symbolic solver. Verifier feedback is further used to support candidate evaluation, self-revision, and reward construction during RLVR. Candidate responses are evaluated along three complementary dimensions: P1 for answer correctness, P2 for evidence or unit consistency, and P3 for reasoning depth and explainability. At inference, gold-free self-consistency aggregates mu
    
[^21]: 无需知晓缩放维数的维数自适应分批Lipschitz收窄算法

    Dimension-Adaptive Batched Lipschitz Narrowing Without Knowing the Zooming Dimension

    [https://arxiv.org/abs/2609.05214](https://arxiv.org/abs/2609.05214)

    本文提出计数自适应BLiN算法，通过依据前一轮淘汰后存活的立方体数量选取下一条边长，去除了A-BLiN对缩放维数 $d_z$ 的依赖，在未知 $d_z$ 的情况下仍以 $\mathcal O_d(\log\log T)$ 个批次达到 $\widetilde{\mathcal O}_d(T^{(d_z+1)/(d_z+2)})$ 的最优遗憾值与批次复杂度。

    

    A-BLiN算法中恰当组合边长（ACE）序列依赖于缩放维数 $d_z$。本短文去除了这一依赖。下一条边长是根据上一轮淘汰后存活的立方体数量来选取的。由此得到的计数自适应BLiN（Count-Adaptive BLiN）算法既不使用 $d_z$ 也不使用缩放常数 $C_z$，却仍能以 $\mathcal O_d(\log\log T)$ 个批次取得 $\widetilde{\mathcal O}_d(T^{(d_z+1)/(d_z+2)})$ 的遗憾值。结合原论文定理10中的自适应网格下界，即使 $d_z$ 未知，最优批次复杂度仍为 $\Theta_d(\log\log T)$。

    arXiv:2609.05214v1 Announce Type: new  Abstract: The Appropriately Combined Edge-length (ACE) sequence in A-BLiN depends on the zooming dimension $d_z$. This note removes that dependence. The next edge length is selected from the number of cubes that survive the preceding elimination. The resulting Count-Adaptive BLiN algorithm does not use $d_z$ or the zooming constant $C_z$, yet it attains $\widetilde{\mathcal O}_d(T^{(d_z+1)/(d_z+2)})$ regret with $\mathcal O_d(\log\log T)$ batches. Together with the adaptive-grid lower bound in Theorem 10 of the original paper, the optimal batch complexity remains $\Theta_d(\log\log T)$ when $d_z$ is unknown.
    
[^22]: 时间序列变分自编码器的PAC-贝叶斯重构保证

    PAC-Bayesian Reconstruction Guarantees for Time Series Variational Autoencoders

    [https://arxiv.org/abs/2609.05212](https://arxiv.org/abs/2609.05212)

    本文为应用于时间序列的潜变量模型提出了一个PAC-贝叶斯框架，将PAC-贝叶斯保证扩展到具有马尔可夫潜结构的重构界，且所得到的保证不随轨迹长度增长。

    

    准确预测时间序列对于从能源系统到医疗保健和金融等复杂数据应用至关重要。在当前最先进的模型中，生成式潜变量模型正得到越来越多的应用；然而，针对现代潜变量模型的原则性泛化保证仍然有限。特别是，尽管变分自编码器被广泛用于序列数据，但其理论分析主要局限于独立同分布的设定。在本工作中，我们为应用于时间序列的潜变量模型开发了一个PAC-贝叶斯框架。基于基于重构的界，我们将PAC-贝叶斯保证扩展到马尔可夫潜结构，通过序列生成过程来捕捉时间依赖性。这些保证不会随轨迹长度的增加而增长。我们的界依赖于文献中常见的假设；我们提供了一个示例框架，在这些假设下……

    arXiv:2609.05212v1 Announce Type: cross  Abstract: Forecasting time series accurately is critical for applications with complex data ranging from energy systems to healthcare and finance. Among current state of the art models, generative latent variable models are increasingly implemented; yet principled generalisation guarantees for modern latent variable models remain limited. In particular, while Variational AutoEncoders are widely used for sequential data, their theoretical analysis is largely restricted to i.i.d. settings. In this work, we develop a PAC-Bayesian framework for latent variables models applied to time series. Building on reconstruction-based bounds, we extend PAC-Bayesian guarantees to Markovian latent structures, capturing temporal dependencies through a sequential generative process. These guarantees do not grow with the length of the trajectory. Our bounds depend on assumptions which are common in the literature; we provide an example framework where they would be
    
[^23]: FluxDisco：基于蒙特卡洛图搜索的化学计量动力学系统符号回归

    FluxDisco: Symbolic Regression for Stoichiometric Dynamical Systems via Monte Carlo Graph Search

    [https://arxiv.org/abs/2609.05207](https://arxiv.org/abs/2609.05207)

    提出了FluxDisco框架，利用已知化学计量关系缩小搜索空间并适配蒙特卡洛图搜索算法，为化学计量ODE系统实现物理约束下的符号回归，从含噪数据中准确恢复可解释的控制方程。

    

    动力学符号回归方法能够从含噪数据中识别出控制微分方程，在可解释性和预测准确性之间取得平衡。然而，标准方法通常会产生违反已知物理定律的表达式。为解决这一问题，我们提出了FluxDisco，一个专为基于通量的化学计量ODE系统量身定制的物理信息框架。通过利用已知的化学计量关系，我们缩小了表达式搜索空间并确保了物理一致性。我们的框架针对化学计量系统联合通量发现所面临的独特挑战，对蒙特卡洛图搜索算法进行了适配。我们在一系列物理和生物系统上评估了该方法，证明了其能够通过可解释的方程准确恢复系统的控制动力学。

    arXiv:2609.05207v1 Announce Type: cross  Abstract: Dynamical symbolic regression methods identify governing differential equations from noisy data, balancing interpretability and predictive accuracy. However, standard methods often produce expressions that violate known physical laws. To address this, we propose FluxDisco, a physics-informed framework tailored for flux-based, stoichiometric ODE systems. By leveraging a known stoichiometry, we reduce the expression search space and ensure physical adherence. Our framework adapts the Monte Carlo Graph Search algorithm for the unique challenges associated with joint flux discovery of stoichiometric systems. We evaluate our method across a range of physical and biological systems, demonstrating its ability to accurately recover governing dynamics through interpretable equations.
    
[^24]: 相变频率作为ResNet测试精度的训练时间预测指标

    Phase Transition Frequency as a Training Time Predictor of Test Accuracy in ResNets

    [https://arxiv.org/abs/2609.05194](https://arxiv.org/abs/2609.05194)

    ResNet微调过程中类可分性“相变”跳变的次数可作为最终测试精度的早期预测指标，在标准i.i.d.基准上呈强负相关（最高达r = -0.87），但在分布偏移条件下该相关性显著减弱。

    

    本文实证考察了ResNet微调过程中观察到的离散类可分性跳变次数作为最终测试精度预测指标的有效性。实验涵盖四个基准数据集（CIFAR-10、CIFAR-100、TinyImageNet和CIFAR-10-C）、三种架构（ResNet-18、ResNet-50和ResNet-101）共75个实验，每个配置使用5到10个随机种子。在标准独立同分布分类基准上获得了较强的数据集内负相关性：CIFAR-10上r = -0.84（p < 10^-8，n = 30），CIFAR-100上r = -0.87（p < 10^-5，n = 15）。在分布压力下，这种关系有所减弱：TinyImageNet上r = -0.45，CIFAR-10-C损坏基准上r = -0.19。另外两项分析进一步规范了该实证结论。一项将架构深度作为线性协变量加以控制的偏相关分析表明，在CIFAR-100上，跳变次数仍保持统计学显著……

    arXiv:2609.05194v1 Announce Type: cross  Abstract: The number of discrete class-separability jumps observed during ResNet finetuning is examined empirically as a predictor of final test accuracy. Across 75 experiments spanning four benchmarks (CIFAR-10, CIFAR-100, TinyImageNet, and CIFAR-10-C) and three architectures (ResNet-18, ResNet-50, and ResNet-101), with five to ten seeds per configuration, a strong within-dataset negative correlation is obtained on standard i.i.d. classification benchmarks: \(r = -0.84\) on CIFAR-10 (\(p < 10^{-8}\), \(n = 30\)) and \(r = -0.87\) on CIFAR-100 (\(p < 10^{-5}\), \(n = 15\)). Under distributional stress, the relationship attenuates: TinyImageNet yields \(r = -0.45\), and the CIFAR-10-C corruption benchmark yields \(r = -0.19\). Two additional analyses discipline the empirical claim. A partial correlation controlling for architecture depth, treated as a linear covariate, shows that on CIFAR-100 the transition count retains statistically significant
    
[^25]: SMILE：面向医学诊断的自解释多模态信息瓶颈方法

    SMILE: Self-Explainable Multimodal Information Bottleneck for Medical Diagnosis

    [https://arxiv.org/abs/2609.05174](https://arxiv.org/abs/2609.05174)

    该论文提出SMILE框架，将自解释多模态医学诊断形式化为信息瓶颈问题，通过联合优化预测性能与模态级可解释性，并借助基于矩阵的Renyi's α阶熵泛函实现稳定优化，从而识别每个模态中对诊断决策贡献最大的关键信息。

    

    可解释性日益被视为基于人工智能的医学诊断中的关键要求，尤其是在安全攸关的临床决策场景中。现有的大多数医疗领域可解释性方法均以事后解释的方式运作，且主要针对单模态数据设计，这限制了它们在日益普及的多模态诊断环境中的适用性。本文通过在信息瓶颈框架内对自解释多模态诊断问题进行形式化建模来解决这一问题。我们提出了一个统一的学习范式，通过识别每个模态内部对诊断决策贡献最大的最具信息量的元素，来联合优化预测性能与特定模态的可解释性。为实现可求解且稳定的优化，在编码器具有充分表达能力的假设下，我们采用了基于矩阵的Renyi's α阶熵泛函。在代表性数据集上开展的大量实验验证了该方法的有效性。

    arXiv:2609.05174v1 Announce Type: cross  Abstract: Explainability is increasingly seen as a crucial requirement in AI-based medical diagnosis, particularly in safety-critical clinical decision-making. Most existing explainability methods in healthcare operate in a post-hoc manner and are predominantly designed for unimodal data, which limits their applicability in increasingly prevalent multimodal diagnostic settings. This paper addresses the problem of self-explainable multimodal diagnosis by formulating it within the information bottleneck (IB) framework. We propose a unified learning paradigm that jointly optimizes predictive performance and modality-specific explainability by identifying the most informative elements inside each modality that contribute to diagnostic decisions. To enable tractable and stable optimization, we employ a matrix-based Renyi's $\alpha$-order entropy functional under the assumption of sufficiently expressive encoders. Extensive experiments on representati
    
[^26]: 面向攻击性安全的共形预测

    Conformal Prediction for Offensive Security

    [https://arxiv.org/abs/2609.05165](https://arxiv.org/abs/2609.05165)

    本文填补了共形预测在攻击性安全应用方面的研究空白，首次探索了其在隐私保护机器学习和网络流量分析两个关键攻击领域中的初步应用。

    

    尽管共形预测（Conformal Prediction, CP）被提出已有二十五年多的时间，但迄今为止它在网络安全领域的应用却出人意料地稀少。我们特别观察到，虽然共形预测在许多近期工作中被用作防御措施，但其在实施攻击（即攻击性安全）方面的应用在文献中却难以寻到踪迹。我们探索了这一空白，在攻击性安全的两个关键领域——隐私保护机器学习和网络流量分析——提出了初步的研究发现。

    arXiv:2609.05165v1 Announce Type: cross  Abstract: Despite its introduction more than a quarter century ago, Conformal Prediction (CP) has seen surprisingly few applications to the cyber security world thus far. In particular, we observe that, while CP has been employed as a defensive measure in many recent works, its use for carrying out attacks (i.e., for offensive security) is hard to trace in the literature. We explore this gap, by presenting initial findings in two key areas of offensive security: Privacy-Preserving Machine Learning, and network traffic analysis.
    
[^27]: 超越时间序列的平稳性：基于马尔可夫毯发现因果结构与潜在状态

    Beyond Stationarity in Time Series: Discovering Causal Structures and Latent Regimes via Markov Blankets

    [https://arxiv.org/abs/2609.05150](https://arxiv.org/abs/2609.05150)

    该论文提出RCBNB-MB算法，突破时间序列平稳性假设，通过将时间序列分割为具有稳定因果结构的潜在状态并在每个状态内利用马尔可夫毯发现因果图，实现了对非平稳时间序列的鲁棒因果发现。

    

    本论文提出了一种新的时间序列因果发现算法——基于马尔可夫毯的状态感知约束型与噪声型因果发现算法（RCBNB-MB），该算法放宽了时间序列中普遍存在的单一、时间一致因果结构的假设。时间序列通常在离散时间点上被观测，且常表现出状态变化，这对静态因果结构的假设构成了挑战，而这一假设是许多现实世界动态系统中的局限性。为应对这一挑战，RCBNB-MB 识别潜在的因果状态，其定义为稳定因果结构成立的时间点子集。该算法采用迭代策略，将时间序列分割为多个状态，并在每个状态内发现因果图。通过利用马尔可夫毯而非直接父节点，RCBNB-MB 增强了对因果发现中错误的鲁棒性，并保留了预测信息。我们为 RCBNB-MB 的能力提供了理论保证……

    arXiv:2609.05150v1 Announce Type: cross  Abstract: This paper introduces Regime-aware Constraint-Based and Noise-Based causal discovery with Markov Blankets (RCBNB-MB), a novel causal discovery algorithm for time series that relaxes the common assumption of a single, time-consistent causal structure. Time series are typically observed at discrete time points and often exhibit regime changes that challenge the assumption of a static causal structure, a limitation in many real-world dynamic systems. To address this challenge, RCBNB-MB identifies latent causal regimes, defined as subsets of time points within which a stable causal structure holds. The algorithm follows an iterative strategy that segments the time series into regimes and discovers the causal graph within each regime. By leveraging the Markov blanket rather than direct parents, RCBNB-MB gains robustness to errors in causal discovery and preserves predictive information. We provide theoretical guarantees for RCBNB-MB's abili
    
[^28]: 一种用于心血管疾病早期风险评估的机器学习与深度神经网络混合预测集成方法

    A Hybrid Predictive Ensemble of Machine Learning and Deep Neural Networks for Early Cardiovascular Disease Risk Assessment

    [https://arxiv.org/abs/2609.05146](https://arxiv.org/abs/2609.05146)

    该研究提出了一种融合机器学习与深度神经网络的混合集成框架，利用医疗物联网的实时生理数据实现心血管疾病的早期风险预测与预后评估。

    

    本研究提出了一种智能框架，该框架集成了机器学习与深度神经网络集成技术，用于心血管疾病的早期检测与预后评估。该系统利用从医疗物联网设备收集的实时生理数据，包括心电图传感器、心率监测器和血压跟踪仪。为确保输入数据的准确性和可靠性，采用了降噪、归一化和缺失值插补等预处理步骤。通过有效的特征选择方法识别出最重要的健康指标，然后使用优化的分类器进行处理，如支持向量机（SVM）、随机森林和极端梯度提升（XGBoost），并将这些分类器以集成架构相结合，以提高诊断精度。该框架在预测心血管疾病风险方面表现出卓越的性能，实现了较高的……（摘要原文在此处截断）

    arXiv:2609.05146v1 Announce Type: new  Abstract: This study introduces an intelligent framework that integrates machine learning and deep neural network ensemble techniques for early detection and prognosis of cardiovascular diseases. The system utilizes real-time physiological data collected from Internet of Medical Things (IoMT) devices, including ECG sensors, heart rate monitors, and blood pressure trackers. To ensure the accuracy and reliability of input data, preprocessing steps such as noise reduction, normalization, and missing value imputation are employed. The most significant health indicators are identified through effective feature selection methods and then processed using optimized classifiers such as Support Vector Machines (SVM), Random Forests, and eXtreme Gradient Boosting (XGBoost), which are combined in an ensemble architecture to improve diagnostic precision. The framework demonstrates remarkable performance in predicting cardiovascular disease risk, achieving high
    
[^29]: 从80倍到385倍：在L2性能上限下的最佳匹配单元搜索，与对称调优的基线对比测量

    From 80x to 385x: A Best-Matching-Unit Search at the L2 Roof, Measured Against a Symmetrically Tuned Baseline

    [https://arxiv.org/abs/2609.05138](https://arxiv.org/abs/2609.05138)

    该论文对新颖SOM算法SparseBin和基线算法cuSPARSE进行了对称调优，使最佳匹配单元搜索每epoch性能提升5.6-10.1倍，将相对基线的优势从约80倍扩大到385倍，并证明调优后的内核已触及77%峰值的L2带宽上限，进一步优化空间仅剩约1.3倍。

    

    GPU实现之间的比较通常是不对称的：一方由其作者精心调优，另一方则按原样运行。我报告了一个程序，它对一种新颖的SOM算法（SparseBin）以及与之比较的基线算法（cuSPARSE）都进行了调优。主导自组织映射训练的最佳匹配单元搜索通过四个手段进行了调优——分块大小、分块成员聚类、神经元轴分块和向量化加载——在32x32到512x512的映射规模下，每个epoch相比之前发布的配置达到了5.6-10.1倍的提升，并使我们早期MEDLINE图谱背后CUDA实现的领先优势从约80倍扩大到约385倍。cuSPARSE（SparseBin与之比较的实现）在每一项调优手段上都获得了对应的类比优化，并在此过程中变得快2-3倍。调优后的内核以77%的峰值利用率触及L2带宽上限，而所有其他单元为40-65%，这将任何进一步优化的空间限制在约1.3倍——

    arXiv:2609.05138v1 Announce Type: new  Abstract: Comparisons between GPU implementations are usually asymmetric: one side is tuned by its author, the other is run as found. I report a programme that tuned both a novel SOM algorithm (SparseBin) and the baseline algorithm it was being compared to (cuSPARSE). The best-matching-unit search that dominates self-organizing map training was tuned through four levers - tile size, tile-membership clustering, neuron-axis chunking and vectorised loads - reaching 5.6-10.1x per epoch over the previously published configuration at map sizes from 32x32 to 512x512, and lifting the margin over the CUDA implementation behind our earlier MEDLINE atlases from ~80x to ~385x. cuSPARSE, the implementation SparseBin is compared against, received every lever with an analogue on its side, and became 2-3x faster in the process. The tuned kernel pressed the L2 bandwidth roof at 77% of peak with every other unit at 40-65%, bounding any further lever at ~1.3x - a te
    
[^30]: MomentQuant：一种用于时间序列分类的、具有线性时间复杂度的更加极简的区间方法

    MomentQuant: an even more minimalist interval method with linear time complexity for time series classification

    [https://arxiv.org/abs/2609.05136](https://arxiv.org/abs/2609.05136)

    该论文提出MomentQuant算法，通过对Quant算法进行更优化的实现，并利用Cornish-Fisher展开以近似分位数替代精确分位数从而免去排序步骤，将时间序列分类的计算效率进一步提升至线性时间复杂度。

    

    时间序列数据在许多现实世界的应用和众多领域中非常常见，人们对于使用机器学习进行自动化信息提取的兴趣日益增长。其中一个子领域是时间序列分类，即为每个新的、未见过的时间序列分配一个标签。在过去几十年中，已经开发了许多算法，而预测性能与计算成本之间的权衡一直被讨论。Quant是一种基于区间的算法，它从递归的、固定的、二分的区间中提取分位数，已被证明能在速度非常快的同时实现高精度。我们提出了两项改进以使该算法更加快速。第一项是对完全相同的算法进行更好优化的实现。第二项是使用Cornish-Fisher展开导出近似分位数，而不是精确分位数。这一改变消除了对时间序列进行排序的必要性，从而降低了计算成本。

    arXiv:2609.05136v1 Announce Type: new  Abstract: Time series data is very common in many real-world applications and in numerous domains, with increasing interest for automated information extraction using machine learning. One of these subfields is time series classification, which consists in assigning a label to each new, unseen time series. Many algorithms have been developed over the past decades, with the trade-off between predictive performance and computational cost being consistently discussed. Quant, an interval-based algorithm extracting quantiles from recursive, fixed, dyadic intervals, was shown to achieve high accuracy, while being very fast. We propose two changes to make this algorithm even faster. The first one is a better optimized implementation of the exact same algorithm. The second one is to derive approximate quantiles, using the Cornish-Fisher expansion, instead of exact quantiles. This change removes the necessity to sort the time series, leading to a smaller c
    
[^31]: 粗粒化隐藏表示：基于映射熵的无监督神经元选择

    Coarse-Graining Hidden Representations: Unsupervised Neuron Selection via Mapping Entropy

    [https://arxiv.org/abs/2609.05126](https://arxiv.org/abs/2609.05126)

    该论文提出了一种完全无监督的神经元选择方法，通过映射熵对隐藏层的粗粒化方案进行评分，仅依据隐藏激活的统计特性即可识别出网络中最具信息量的神经元子集，无需标签或梯度。

    

    过参数化神经网络所包含的隐藏单元数量远超任务名义上所需，这引出了一个问题：哪些神经元是必不可少的，以及这种区分能否在表示本身中被识别出来——而无需标签或梯度。我们将神经元选择问题表述为通过保留隐藏层神经元的一个子集来对隐藏层进行粗粒化，并利用映射熵对每个候选选择进行评分。该量度衡量了丢弃部分网络神经元时固有的判别能力损失，使映射熵最小化的选择被认为是最具信息量的。这一准则完全是无监督的，因为它仅依赖于隐藏层激活的统计特性。在师生网络中，映射熵优化能够恢复出与教师一致的最小表示，并按隐藏层残余变异性的比例保留额外的单元；在一个非线性高斯过程任务中，它选择出一致的函数

    arXiv:2609.05126v1 Announce Type: new  Abstract: Overparameterized neural networks carry far more hidden units than a task nominally requires, raising the question of which neurons are essential and whether that distinction is legible in the representation itself, without labels or gradients. We cast neuron selection as the problem of coarse-graining the hidden layer by retaining a subset of its neurons, and score each putative selection by the mapping entropy (ME). This quantity measures the loss of discriminatory power inherent in discarding part of the network neurons, and the selection that minimises the ME is taken as particularly informative. This criterion is fully unsupervised, in that it depends only on hidden-activation statistics. In teacher-student networks, ME optimisation recovers the minimal teacher-consistent representation and retains extra units in proportion to the hidden layer's residual variability; in a non-linear Gaussian process task, it selects coherent functio
    
[^32]: 基于Logit偏置的单次查询黑盒校准审计

    Single-Query Black-Box Calibration Auditing via Logit Bias

    [https://arxiv.org/abs/2609.05125](https://arxiv.org/abs/2609.05125)

    本文提出利用LLM API中暴露的logit_bias参数进行数学操作，每样本仅需一次查询即可评估精确概率阈值，并给出二分类任务真实校准误差的首个可证明一致估计器，为黑盒基础模型的校准审计提供高效框架。

    

    评估大型语言模型（LLM）的校准对于其作为零样本分类器的安全部署至关重要。然而，商业API提供商越来越多地隐藏标准校准指标所需的连续输出概率。为了绕过这种不透明性，我们证明任何暴露logit_bias参数的LLM API都可以通过数学操作，在严格每样本仅一次查询的情况下评估精确的概率阈值。利用这一机制，我们为二分类任务引入了一种新颖且可证明一致的真实校准误差估计器。因此，我们的方法为审计黑盒基础模型提供了一个高效的框架。

    arXiv:2609.05125v1 Announce Type: new  Abstract: Evaluating the calibration of Large Language Models (LLMs) is critical for their safe deployment as zero-shot classifiers. Yet, commercial API providers increasingly hide the continuous output probabilities required by standard calibration metrics. To bypass this opacity, we demonstrate that any LLM API exposing a logit\_bias parameter can be mathematically manipulated to evaluate exact probability thresholds using strictly one query per sample. Leveraging this mechanism, we introduce a novel and provably consistent estimator of the True Calibration Error for binary tasks. Our approach therefore provides an efficient framework for auditing black-box foundation models.
    
[^33]: 一项支持多种图编辑类型的图神经网络反事实解释器的比较研究

    A Comparative Study of Counterfactual Explainers for Graph Neural Networks Enabling Multiple Types of Graph Edit

    [https://arxiv.org/abs/2609.05113](https://arxiv.org/abs/2609.05113)

    本文对六种最先进的图神经网络反事实解释方法在多种真实与合成数据集上的图分类和节点分类任务进行了系统比较，揭示了各方法在解释大小、覆盖率和质量之间的权衡，为该领域的未来研究提供了指导。

    

    面向图结构数据的反事实解释旨在确定对输入图进行何种最小且符合现实的修改，才能将模型的预测改变为预定义的输出。尽管近期已出现支持通过添加和删除边来修改图的反事实解释器，但仍然缺乏通用且高效的方法，尤其是在生成解释的质量方面。此外，该问题仍远未解决，因为现有方法各具优劣，往往需要在解释大小、覆盖率和质量之间进行权衡。因此，识别每种方法在哪些方面表现良好、在哪些方面存在不足十分重要，以便为该领域的未来研究提供指导。为此，我们的研究在多样化的真实世界和合成数据集上比较了六种最先进的模型，涵盖了二分类和多分类的图分类与节点分类任务。

    arXiv:2609.05113v1 Announce Type: new  Abstract: Counterfactual explanations for graph-structured data seek to determine minimal and realistic modifications required in an input graph to alter a model's prediction to a predefined output. Although counterfactual explainers that support modifying the graph by both adding and removing edges have recently emerged, there is still a lack of general and efficient methods, especially when considering the quality of the generated explanations. Moreover, the problem remains far from solved, as existing methods exhibit different strengths and weaknesses, often trading off between explanation size, coverage and quality. For this reason, it is important to identify where each method performs well and where it falls short, so as to guide future research in the field. Thus, our study compares six state-of-the-art (SOTA) models on a diverse set of real-world and synthetic datasets, covering both binary and multi-class graph and node classification tas
    
[^34]: NEAT-POCKET：基于邻域引导集合变换器的口袋条件自回归3D分子生成

    NEAT-POCKET: Pocket-Conditioned Autoregressive 3D Molecular Generation with a Neighborhood-Guided Set Transformer

    [https://arxiv.org/abs/2609.05097](https://arxiv.org/abs/2609.05097)

    NEAT-POCKET是一个蛋白质口袋条件下的自回归3D分子生成模型，在保持原子排列不变性并显式建模氢原子的同时，以显著快于现有方法的采样速度实现了具有竞争力的基于结构的药物分子生成性能，并支持口袋条件下的片段补全以辅助先导化合物优化。

    

    AI驱动的新型分子设计通过直接在靶蛋白结合口袋内生成新型配体，为加速早期药物发现提供了一条有前景的路线。我们提出了NEAT-POCKET，这是自回归NEAT模型在3D分子生成方面的口袋条件扩展版本。NEAT-POCKET能够在蛋白质口袋环境中逐原子生成分子，同时保持原子排列不变性并显式建模氢原子。在CrossDocked和SPINDR数据集上的基准测试表明，NEAT-POCKET实现了具有竞争力的基于结构的生成性能，同时采样速度显著快于现有基线方法。除了完整分子生成之外，NEAT-POCKET还能自然地实现口袋条件下的片段补全，这一任务与先导化合物优化和骨架衍生直接相关。这些结果使NEAT-POCKET成为基于结构的药物设计中一个快速、灵活且实用的框架。

    arXiv:2609.05097v1 Announce Type: cross  Abstract: AI-driven de novo molecular design offers a promising route to accelerate early-stage drug discovery by generating novel ligands directly within target protein binding pockets. We present NEAT-POCKET, a pocket-conditioned extension of the autoregressive NEAT model for 3D molecular generation. NEAT-POCKET generates molecules atom by atom in protein pocket environments while preserving atom permutation invariance and explicitly modeling hydrogen atoms. Benchmarks on the CrossDocked and SPINDR datasets show that NEAT-POCKET achieves competitive structure-based generation performance while sampling substantially faster than existing baselines. Beyond full-molecule generation, NEAT-POCKET naturally enables pocket-conditioned fragment completion, a task directly relevant to lead optimization and scaffold elaboration. These results position NEAT-POCKET as a fast, flexible, and practical framework for structure-based drug design.
    
[^35]: 深度微压缩：面向微控制器的结构化剪枝与位打包量化

    Deep Microcompression: Structured Pruning and Bit-packed Quantization for Microcontrollers

    [https://arxiv.org/abs/2609.05081](https://arxiv.org/abs/2609.05081)

    DMC通过结构化剪枝、量化感知训练和位打包技术，实现了标准CNN在仅有2KB SRAM的ATmega328P微控制器上的首次部署，在LeNet-5上达到55.8倍压缩比并保持98.77%的准确率。

    

    本文介绍了深度微压缩（Deep Microcompression，DMC），这是一种面向裸机微控制器深度学习推理的硬件感知流水线。DMC集成了结构化剪枝、量化感知训练和定长位打包技术，在LeNet-5上实现了55.8倍的权重压缩比（准确率98.77%），并生成了一个无依赖的C库，具有确定性的推理延迟。在RP2040（Cortex-M0+）上，DMC相较于TensorFlow Lite将二进制文件大小减少了3倍，同时保持了相同的准确率。至关重要的是，DMC实现了首个有文献记录的标准CNN在ATmega328P上的部署——该设备仅有2KB的SRAM，此前被认为无法进行CNN推理。

    arXiv:2609.05081v1 Announce Type: new  Abstract: This paper introduces Deep Microcompression (DMC), a hardware-aware pipeline for deep learning inference on bare-metal microcontrollers. DMC integrates structured pruning, quantization-aware training, and fixed-length bit-packing to achieve a 55.8$\times$ weight compression ratio on LeNet-5 (98.77\% accuracy), generating a dependency-free C library with deterministic latency. On the RP2040 (Cortex-M0+), DMC reduces binary size by 3$\times$ versus TensorFlow Lite while matching its accuracy. Critically, DMC enables the first documented deployment of a standard CNN on the ATmega328P, a device constrained to 2KB SRAM, previously considered infeasible for CNN inference.
    
[^36]: 无线网络中反事实KPI的混杂有效性保形推断

    Confounding-Valid Conformal Inference for Counterfactual KPIs in Wireless Networks

    [https://arxiv.org/abs/2609.05073](https://arxiv.org/abs/2609.05073)

    针对无线网络遥测数据中因遗漏变量而产生的隐藏混杂因素问题，本文提出了一种混杂有效性保形推断方法，在随机化遥测数据稀缺的情况下，为反事实KPI分析提供可靠的统计保证。

    

    保形反事实推断使网络运营商能够利用记录的遥测数据，可靠地回答关于网络运行的“假设分析”问题。这些答案通常以预测集的形式呈现，该预测集以用户定义的概率包含在替代控制操作下本应观察到的关键性能指标（KPI）。一个关键挑战是，记录的遥测数据可能遗漏控制器所使用的变量，从而导致隐藏的混杂因素，并使反事实分析的统计保证失效。原则上，这一问题可以通过随机化遥测来解决，即通过独立于网络状态分配控制操作来收集遥测数据。然而，由于这种随机化可能会干扰网络的正常运行，随机化遥测数据通常十分稀缺，导致仅基于此类数据的反事实分析只能产生信息量不足的预测集。为应对这些挑战，我们提出了混杂有效性耦合（原文在此处截断）……

    arXiv:2609.05073v1 Announce Type: new  Abstract: Conformal counterfactual inference enables network operators to use logged telemetry to reliably answer 'what-if' questions about network operation. These answers typically take the form of prediction sets that contain, with a user-defined probability, the key performance indicators (KPIs) that would have been observed under alternative control actions. A key challenge is that logged telemetry may omit variables used by the controller, resulting in hidden confounding and invalidating the statistical guarantees of counterfactual analysis. In principle, this issue can be addressed using randomized telemetry, collected by assigning control actions independently of the network state. However, because such randomization may disrupt normal operation, randomized telemetry is typically scarce, causing counterfactual analysis based solely on it to produce uninformative prediction sets. To address these challenges, we propose Confounding-Valid Cou
    
[^37]: 超越共同购买关系：Allegro平台互补推荐的演进

    Beyond Co-purchase Relation: Evolution of Complementary Recommendations at Allegro

    [https://arxiv.org/abs/2609.05063](https://arxiv.org/abs/2609.05063)

    本文提出部署在Allegro电商平台的生产级检索框架AlleCompanion，通过数据级过滤启发式方法与类别约束的双塔架构（含类别适配器）将嘈杂的共同购买行为信号转化为精确的语义互补性，从而实现更精准的互补产品推荐。

    

    当客户将一台专业相机加入购物车时，系统应该推荐匹配的镜头、通用的三脚架，还是另一台相机机身？互补产品推荐对于构建完整的购物篮至关重要，然而标准模型往往无法区分那些仅仅是被一起购买的商品与真正能够协同使用的商品。在本文中，我们提出了AlleCompanion：一个部署在Allegro.com的生产级检索框架，它将嘈杂的行为信号转化为精确的语义兼容性。我们通过将数据级过滤启发式方法与类别约束的双塔架构相结合，来缓解大规模共同购买流量中固有的噪声。在该框架内，类别适配器在嵌入空间中引导模型，将候选商品约束在逻辑上互补的边界之内。由于大规模建模真实用户行为本身就十分困难，我们引入了ComCat，一个……

    arXiv:2609.05063v1 Announce Type: cross  Abstract: When a customer adds a professional camera to their cart, should the system suggest a matching lens, a generic tripod, or another camera body? Complementary Product Recommendation is vital for comprehensive basket building, yet standard models often fail to distinguish between items that are merely bought together and those that truly work together. In this paper, we present AlleCompanion: a production-scale retrieval framework deployed at Allegro.com that transforms noisy behavioural signals into precise semantic compatibility. We mitigate the intrinsic noise in large-scale co-purchase traffic by combining data-level filtering heuristics with a category-constrained Two Tower architecture. Within this framework, the Category Adapter guides the model in the embedding space, constraining candidates within logically complementary boundaries. Since modelling authentic user behaviour at scale is inherently difficult, we introduce ComCat, a 
    
[^38]: 后处理中的数据丢失对量子神经网络训练与推理的影响

    Impact of Data Loss in Postprocessing on Training and Inference of Quantum Neural Networks

    [https://arxiv.org/abs/2609.05060](https://arxiv.org/abs/2609.05060)

    本研究以 Qiskit 的 SamplerQNN 为案例，揭示了面向模拟器设计的后处理假设在真实量子硬件上会导致高达99.6%的有效测量数据被静默丢失且难以察觉，从而在无任何API报错的情况下扭曲量子神经网络的训练与推理结果。

    

    随着量子硬件扩展到更大规模的设备，与之对接的经典软件层也必须同步演进。主要在模拟器环境中开发和测试的后处理程序可能包含一些在实用规模设备上不再成立的假设，从而导致难以仅从高层模型输出中检测到的数据丢失。我们对 Qiskit Machine Learning 库中基于采样的量子神经网络类 SamplerQNN 进行了案例研究。该类的后处理方法应用了一个假设测量比特串位于虚拟量子比特空间的过滤器。在我们的量子硬件运行中，比特串跨越超过100个物理量子比特，该过滤器导致85%至99.6%的有效测量样本丢失，具体取决于编译器的量子比特放置策略。由此产生的概率向量未经过归一化，使得失真的预测值和损失值能够在模型中传播，而不会触发API级别的错误提示。

    arXiv:2609.05060v1 Announce Type: cross  Abstract: As quantum hardware scales to larger devices, the classical software layers that interface with it must evolve in step. Postprocessing routines developed and tested primarily in simulator settings can encode assumptions that no longer hold on utility-scale devices, leading to data loss that can be difficult to detect from high-level model outputs alone. We present a case study of \texttt{SamplerQNN}, the sampling-based quantum neural network class in the Qiskit Machine Learning library. Here, the postprocessing method applies a filter that assumes measurement bit-strings are in virtual qubit space. On our quantum hardware runs, where bit-strings span over 100 physical qubits, this filter led to the loss of 85 to 99.6\% of valid measurement shots, depending on the transpiler's qubit placement. The resulting probability vector is unnormalised, allowing distorted prediction and loss values to propagate through the model without an API-lev
    
[^39]: 具有依赖样本的自监督预训练分析

    An Analysis of Self-supervised Pre-training with Dependent Samples

    [https://arxiv.org/abs/2609.05031](https://arxiv.org/abs/2609.05031)

    该研究证明，在自监督预训练中，尽管同一数据点的不同增强之间存在相互依赖性，将它们汇集在一起学习比将数据划分为独立子集的基线方法是更好的选择。

    

    自监督学习依赖于对未标记数据点 x 的所谓数据增强 φ(x)——例如，对图像 x 中的随机像素进行掩码——这些增强应保持 x 的标签不变，并常被用于为下游任务学习一个较低复杂度的不变子空间 V。在实践中，这些增强 {φ_l(x_i)} 被汇集在一起来学习 V，尽管同一数据点 x 的不同增强 φ_l(x)、φ_k(x) 之间存在明显的相互依赖性。然而，该主题的相关理论工作通常考虑避免这种依赖性的方法，因此仅限于在较小的独立数据子集上运行。我们在这项工作中表明，尽管存在相互依赖性，将增强汇集在一起比将数据划分为独立数据子集的基线方法是更好的选择。更准确地说，在估计 V 的背景下，统计

    arXiv:2609.05031v1 Announce Type: cross  Abstract: Self-supervised learning relies on so-called data augmentations $\phi(x)$ of unlabeled datapoints $x$ --- for example, masking random pixels in an image $x$ --- that should leave the label of $x$ invariant and are often used to learn a lower-complexity invariant subspace $\cal V$ for downstream tasks. In practice, such augmentations $\{ \phi_l(x_i) \}$ are pooled together to learn $\cal V$, despite obvious inter-dependencies between different augmentations $\phi_l(x), \phi_k(x)$ of the same datapoint $x$. However, theoretical works on the subject typically consider procedures that avoid such dependencies, and are therefore limited to operate on smaller subsets of independent data.   We show in this work that pooling augmentations together, despite inter-dependencies, is a better alternative than the baseline of partitioning the data into subsets of independent data. More precisely, in the context of estimating $\cal V$, the statistical
    
[^40]: 摊销缩放定律构建成本

    Amortizing Scaling Law Construction Costs

    [https://arxiv.org/abs/2609.05016](https://arxiv.org/abs/2609.05016)

    该论文提出了一种基于贝叶斯优化的高效缩放定律构建框架，通过逐步扩展计算预算并结合代理模型幻想评估来恢复广泛的实验网格，大幅降低了构建缩放定律的计算成本。

    

    缩放定律指导着大型基础模型的训练设计选择，但推导缩放定律需要在超参数、token预算和参数数量上进行穷举式的网格训练，这在计算上非常昂贵。然而，拟合缩放定律只需要跨计算规模的最佳损失前沿，大部分已训练的配置都会被丢弃。我们提出了一个高效的缩放定律构建框架，将数据收集表述为贝叶斯优化问题，并引入了在受限计算预算下比较缩放定律拟合方法的评估指标。我们发现，在采集过程中逐步扩展计算预算（类似于实践中按计算规模顺序评估配置的方式），可以显著提高恢复效率。随后，通过代理模型幻想评估来增强观察到的配置，可以恢复更广泛的实验网格，从而实现准确的缩放定律拟合。

    arXiv:2609.05016v1 Announce Type: cross  Abstract: Scaling laws guide the design choices for training large foundation models, but deriving them involves training an exhaustive grid over hyperparameters, token budgets, and parameter counts, which is computationally expensive. Fitting a scaling law, however, only requires the best-loss frontier across compute scales, discarding most of the trained configurations. We propose a framework for efficient scaling law construction that formulates data collection as a Bayesian optimization problem, and introduce metrics for comparing scaling law fitting methods under constrained compute budgets. We find that progressively expanding the compute budget during acquisition, mirroring the compute-ordered evaluation of configurations in practice, substantially improves recovery efficiency. Augmenting the observed configurations with surrogate-fantasized evaluations then recovers the broader experimental grid, allowing accurate scaling law fitting wit
    
[^41]: 解空间异质性塑造偏微分方程联邦学习的动力学

    Solution-space heterogeneity shapes federated learning dynamics across partial differential equations

    [https://arxiv.org/abs/2609.05012](https://arxiv.org/abs/2609.05012)

    该论文提出解空间 PDE-Dirichlet 协议，为偏微分方程联邦学习提供了可迁移的非独立同分布数据定义，推导出分配异质性与 Dirichlet 浓度的精确反比关系，并揭示了解空间异质性驱动梯度分歧与参数发散的动力学机制。

    

    联邦科学机器学习使各机构能够在不集中本地物理数据的前提下训练神经代理模型，然而针对偏微分方程（PDE）的研究缺乏一个可迁移的非独立同分布数据定义。现有协议依据方程特定的规则对坐标、系数、边界条件或几何形状进行划分。在此，我们提出解空间 PDE-Dirichlet 协议，该协议将连续的监督响应转换为可复用的解分箱，并通过在这些分箱几何结构上的最优传输来量化客户端之间实际实现的分离程度。我们推导出了总体分配异质性与 Dirichlet 浓度之间的精确反比关系，并建立了响应异质性诱导梯度分歧、局部更新离散以及参数发散的条件。在七个受控和公开的 PDE 任务上，三个（摘要内容在此处截断）

    arXiv:2609.05012v1 Announce Type: new  Abstract: Federated scientific machine learning enables institutions to train neural surrogates without centralizing local physical data, yet studies of partial differential equations (PDEs) lack a transferable definition of non-independent and identically distributed data. Existing protocols partition coordinates, coefficients, boundary conditions, or geometries according to equation-specific rules. Here, we introduce solution-space PDE-Dirichlet, a protocol that converts continuous supervised responses into reusable solution bins and quantifies the realized separation between clients through optimal transport over the geometry of these bins. We derive an exact inverse relation between population allocation heterogeneity and the Dirichlet concentration, and we establish conditions under which response heterogeneity induces gradient disagreement, local-update dispersion, and parameter divergence. Across seven controlled and public PDE tasks, three
    
[^42]: 超越同方差性：面向深度不平衡回归的解耦不确定性优化

    Beyond Homoscedasticity: Decoupled Uncertainty Optimization for Deep Imbalanced Regression

    [https://arxiv.org/abs/2609.04995](https://arxiv.org/abs/2609.04995)

    该论文提出了解耦不确定性优化框架DUO，通过解决异方差负对数似然中的梯度耦合问题，增强长尾分布中困难尾部样本的学习信号，从而缓解深度不平衡回归中的尾部欠拟合。

    

    深度不平衡回归（DIR）普遍存在于跨多种模态的连续预测任务中，例如年龄估计、深度预测和蛋白质突变活性预测，其中标签稀少的尾部样本往往具有更高的实际价值。然而，大多数现有方法仍在均方误差或其简单变体下学习确定性的点映射，隐含地假设所有样本具有相同的不确定性水平，从而忽略了长尾数据中普遍存在的逐样本异方差性。我们进一步指出，即使是异方差负对数似然也受到梯度耦合问题的困扰，在DIR场景下，这会削弱困难尾部样本的学习信号，导致优化惰性和尾部欠拟合。为解决这一问题，我们提出了DUO，一个不确定性感知的长尾回归框架。具体而言，所提出的方法将回归目标建模为……

    arXiv:2609.04995v1 Announce Type: new  Abstract: Deep Imbalanced Regression (DIR) is pervasive in continuous prediction tasks across diverse modalities, such as age estimation, depth prediction, and protein mutation activity prediction, where label-scarce tail samples often carry higher practical value. However, most existing methods still learn deterministic point mappings under mean squared error or its simple variants, implicitly assuming a uniform uncertainty level across all samples and thereby overlooking the instance-wise heteroscedasticity that is widespread in long-tailed data. We further point out that even heteroscedastic negative log-likelihood suffers from a gradient coupling issue, which, under DIR scenarios, weakens the learning signal of hard tail samples and leads to optimization inertia as well as tail underfitting. To address this, we propose DUO, an uncertainty-aware long-tailed regression framework. Specifically, the proposed method models the regression target as 
    
[^43]: BeaconKV：由信标查询引导的键值缓存压缩方法，实现高效的大推理模型推理

    BeaconKV: Key-Value Cache Compression Guided by Beacon Queries for Efficient Large Reasoning Model Inference

    [https://arxiv.org/abs/2609.04971](https://arxiv.org/abs/2609.04971)

    该论文发现长程推理中存在会重新关注早期关键上下文的“思维回溯token”，且其查询在嵌入空间中聚成少数相似组，据此提出无需训练的KV缓存压缩方法BeaconKV，从而高效支持大型推理模型的推理。

    

    大型推理模型通过扩展的思维链生成获得了卓越的问题解决能力，但由此产生的键值缓存随序列长度线性增长，造成严重的内存瓶颈，在长推理轨迹场景下常常超出GPU容量。现有的KV缓存压缩方法依赖最近的查询来估计未来token的重要性，隐含地假设这些查询可以作为未来注意力模式的可靠代理。我们证明这一假设在长程推理中并不成立：某些解码步骤会生成“思维回溯token” (Thought Revisiting Tokens, TRT)，重新关注距离较远的先前上下文，例如推理轨迹早期形成的任务求解计划。通过系统性分析，我们发现与TRT对应的查询在嵌入空间中聚集成少数几个相似度组。基于这一洞察，我们提出BeaconKV，一种无需训练的KV缓存压缩方法……

    arXiv:2609.04971v1 Announce Type: cross  Abstract: Large Reasoning Models (LRMs) achieve superior problem-solving through extended Chain-of-Thought (CoT) generation, but the resulting key-value (KV) cache grows linearly with sequence length and creates severe memory bottlenecks, often exceeding GPU capacity for long reasoning traces. Existing KV cache compression methods rely on recent queries to estimate future token importance, implicitly assuming these serve as reliable proxies for future attention patterns. We demonstrate that this assumption fails in long-horizon reasoning: certain decoding steps generate Thought Revisiting Tokens (TRT) that re-attend to distant previous context, such as task-solving plans formulated early in the trace. Through systematic analysis, we discover that queries corresponding to the TRT cluster into a small number of similarity groups in the embedding space. Based on this insight, we propose BeaconKV, a training-free KV cache compression method that mai
    
[^44]: 分形盆地困住潜在推理

    Fractal basins trap latent reasoning

    [https://arxiv.org/abs/2609.04963](https://arxiv.org/abs/2609.04963)

    该论文发现推理模型本质上是具有分形盆地的动力系统，暂态混沌源于推理在对应“近乎正确解”的鞍点附近长时间被困，从而揭示了任务越难、推理越慢的动力学机制。

    

    推理使人工智能模型能够重新审视并纠正自身的错误，从而推动了近期在数学定理求解、软件工程和自主任务规划等前沿领域的进展。人们普遍观察到，推理模型在更难的任务上会进行更长时间的推理，但导致这种减速的普遍机制尚不清楚。在本研究中，我们证明推理模型表现出暂态混沌，这是困难任务的计算复杂性所带来的一种物理后果。由此，我们证明多种领先的推理模型都是具有分形盆地的动力系统，且分形程度随任务难度增加而上升——这一现象在数独、迷宫求解、视觉谜题和数理逻辑等多种任务中均有体现。我们进一步证明，暂态混沌的产生是由于推理过程在鞍点附近被长时间困住，而这些鞍点对应于对底层问题近乎正确的尝试性解答。我们的结果……

    arXiv:2609.04963v1 Announce Type: new  Abstract: Reasoning allows artificial intelligence models to revisit and correct their mistakes, enabling recent frontier advances in mathematical theorem solving, software engineering, and autonomous task planning. Reasoning models are widely observed to reason for longer on harder tasks, but the general mechanism responsible for these slowdowns is unknown. Here, we show that reasoning models exhibit transient chaos, a physical consequence of the computational complexity of difficult tasks. As a consequence, we show that diverse leading reasoning models are dynamical systems with fractal basins, with fractality increasing with task difficulty across diverse tasks like Sudoku and maze solving, visual puzzles, and mathematical logic. We show that transient chaos emerges due to reasoning becoming trapped for extended durations near saddle points, which we show correspond to nearly-correct attempted solutions of the underlying problem. Our results sh
    
[^45]: 面向可扩展电网图分类的物理感知随机游走指纹

    Physics-Aware Random Walk Fingerprints for Scalable Power Grid Graph Classification

    [https://arxiv.org/abs/2609.04943](https://arxiv.org/abs/2609.04943)

    本文提出多通道物理感知随机游走指纹（MC-PA-RWF），通过将物理边状态引入随机游走传播，为电网图级联故障分类任务提供了一种无需端到端训练、可扩展且可解释的轻量级图级表示方法。

    

    近年来的基准数据集（如PowerGraph）提供了大量用于级联故障分类的电网图。图神经网络（GNN）在此任务上表现出强大的预测性能，但通常需要端到端训练和针对特定模型的调优，且其潜在表示难以与具有物理意义的传播模式相关联。随机游走指纹（RWF）提供了一种可扩展且可解释的替代方案，但现有变体主要强调拓扑结构和节点级信息，在游走动力学中忽略了与电网运行相关的边状态。我们提出面向电力系统的多通道物理感知随机游走指纹（MC-PA-RWF），这是一种轻量级的图级表示框架，将物理边状态引入随机游走传播过程。该方法从领域相关属性构建多个边加权通道，并从每个通道中提取特定指纹……

    arXiv:2609.04943v1 Announce Type: new  Abstract: Recent benchmarks such as PowerGraph provide large collections of power-grid graphs for cascading-failure classification. Graph neural networks (GNNs) achieve strong predictive performance on this task, but typically require end-to-end training and model-specific tuning, while their latent representations can be difficult to relate to physically meaningful propagation patterns. Random Walk Fingerprints (RWF) offer a scalable and interpretable alternative, but existing variants primarily emphasise topology and node-level information, leaving grid-relevant operational edge states in the walk dynamics. We propose Multi-Channel Physics-Aware Random Walk Fingerprints (MC-PA-RWF) for power systems, a lightweight graph-level representation framework that introduces physical edge states into random-walk propagation. The method constructs multiple edge-weighted channels from domain-relevant attributes, extracts a channel-specific fingerprint from
    
[^46]: 一个扩散模型，两种角色：闭环仿真中的引导轨迹规划与安全关键场景生成

    One Diffusion Model, Two Roles: Guided Trajectory Planning and Safety-Critical Scenario Generation in Closed-Loop Simulation

    [https://arxiv.org/abs/2609.04921](https://arxiv.org/abs/2609.04921)

    提出单个预训练扩散交通模型可同时充当自车规划器与安全关键场景生成器，借助SSDS扩散变换器解码器和免训练的DAPSE能量引导方案，在nuPlan上提升闭环规划性能并支持压力测试场景生成

    

    扩散概率模型能够捕捉驾驶场景中联合未来轨迹的多模态、富含交互特性的分布。我们证明，单个预训练的扩散交通模型可以在自动驾驶开发闭环中扮演两种互补角色：作为自车运动规划器，以及作为用于压力测试规划器的可控安全关键场景生成器。在规划方面，我们提出了一种单流-双流（SSDS）扩散变换器解码器，通过联合注意力而非后期交叉注意力来融合场景上下文，从而提升了nuPlan上的闭环性能。我们进一步提出了基于能量的解耦退火后验采样（DAPSE），这是一种无需训练的引导方案，可在干净样本层面注入任意能量函数，避免了一阶近似误差，同时不需要任何辅助网络。除规划之外，我们利用同一个扩散模型作为可控的安全关键场景生成器（摘要在此处截断）

    arXiv:2609.04921v1 Announce Type: cross  Abstract: Diffusion probabilistic models can capture the multi-modal, interaction-rich distribution of joint future trajectories in driving scenes. We show that a single pretrained diffusion traffic model can serve two complementary roles in the autonomous driving development loop: as an ego motion planner, and as a controllable generator of safety-critical scenarios for stress-testing the planners. On the planning side, we introduce a Single-Stream Dual-Stream (SSDS) diffusion-transformer decoder that fuses scene context via joint attention rather than late cross-attention, improving closed-loop performance on nuPlan. We further propose Decoupled Annealing Posterior Sampling with Energy (DAPSE), a training-free guidance scheme that injects arbitrary energy functions at the clean-sample level, avoiding the first-order approximation errors while requiring no auxiliary networks. Beyond planning, we leverage the same diffusion model as a controllab
    
[^47]: 通过Flash Attention快速计算高斯和

    Fast Gauss Sums via Flash Attention

    [https://arxiv.org/abs/2609.04910](https://arxiv.org/abs/2609.04910)

    该论文发现仅需两次简单的输入增广，就能借助高度优化的Flash Attention计算任意带符号权重的高斯核和，无需编写任何自定义GPU代码，且在速度、内存开销和精度上均显著超越PyTorch和PyKeOps等现有方案。

    

    高斯核和是最大均值差异（MMD）、核梯度流、Stein变分梯度下降（SVGD）以及许多其他核方法的计算核心。与此同时，softmax注意力机制已经获得了大量硬件感知的代码工程投入，最终催生了Flash Attention。我们证明了具有任意带符号权重的高斯核和可以通过Flash Attention来计算：只需两次简单的输入增广操作，即可将归一化的softmax归约转换为非归一化的高斯和，而无需编写任何一行自定义GPU代码。对于fp16精度下特征维度D>8的情况，该方法在速度、内存开销和精度方面均优于编译后的PyTorch代码和PyKeOps内核（且往往优势显著）。事实上，其内存占用始终保持线性扩展。

    arXiv:2609.04910v1 Announce Type: new  Abstract: Gaussian kernel sums are the computational core of maximum mean discrepancies (MMDs), kernel gradient flows, Stein variational gradient descent (SVGD), and many other kernel methods. At the same time, softmax attention has received an extraordinary amount of hardware-aware code engineering, culminating in flash attention. We show that Gauss kernel sums with arbitrary, signed weights can be evaluated via flash attention: two small input augmentations turn the normalized softmax reduction into the unnormalized Gauss sum, without writing a single line of custom GPU code. For feature dimension D>8 in fp16, this approach beats compiled PyTorch code as well as PyKeOps kernels (often significantly) in speed, memory-overhead and accuracy. Indeed, its memory scaling remains linear.
    
[^48]: 基于未正射校正影像的星载卫星甲烷检测

    Methane Detection On Board Satellites from Unorthorectified Imagery

    [https://arxiv.org/abs/2609.04906](https://arxiv.org/abs/2609.04906)

    本文提出UnorthoDOS数据集与方法，首次直接在未正射校正的高光谱影像上训练U-Net模型进行甲烷羽流检测，性能接近基于正射校正数据训练的模型并大幅超越匹配滤波基线，同时通过FP16压缩验证了模型在卫星上部署的可行性。

    

    甲烷是一种强效温室气体，是气候变化的主要驱动因素之一。其有效减排依赖于及时检测。传统检测方法依赖正射校正来纠正几何畸变，并使用匹配滤波器来增强羽流信号，这些步骤是为地面处理而设计的，并不适合星载执行。我们提出了UnorthoDOS，这是一个数据集与方法，可直接在未正射校正的高光谱影像上训练机器学习模型，从而省去正射校正和匹配滤波产品这两个步骤。我们在未正射校正数据上训练的U-Net模型，其性能接近在正射校正数据上训练的模型（在所有羽流上IoU为16.91%对比18.47%），且两者均大幅优于mag1c匹配滤波基线（IoU为4.76%）。我们进一步证明了星载部署的可行性：FP16压缩使模型大小减半，而输出偏差低于0.3%。已训练的机器学习模型和两个……（原文摘要在此处截断）

    arXiv:2609.04906v1 Announce Type: cross  Abstract: As a potent greenhouse gas, methane is a major driver of climate change. Its effective mitigation relies on timely detection. Conventional detection methods rely on orthorectification to correct geometric distortions and matched filters to enhance plume signals, which are steps designed for ground processing and poorly suited to onboard execution. We introduce UnorthoDOS, a dataset and approach for training machine learning models directly on unorthorectified hyperspectral imagery, bypassing both orthorectification and matched-filter products. Our U-Net models trained on unorthorectified data approach the performance of models trained on orthorectified data (IoU 16.91% vs. 18.47% on all plumes), while both substantially outperform the mag1c matched-filter baseline (IoU 4.76%). We further demonstrate the feasibility of onboard deployment: FP16 compression halves model size with under 0.3% output deviation. The trained ML models and two 
    
[^49]: 基于声音的多人三维姿态估计

    Sound-based Multi-Person 3D Pose Estimation

    [https://arxiv.org/abs/2609.04902](https://arxiv.org/abs/2609.04902)

    本文首次提出仅利用声学信号进行多人三维姿态估计的SoundMHPE框架，通过编码器-解码器架构克服了多人声学信号重叠和人际反射干扰的挑战。

    

    arXiv:2609.04902v1 公告类型： cross 摘要：我们能否仅利用声音来恢复多个人的三维姿态？本文首次尝试仅从声学信号估计多人三维姿态。由于运动相关的信号变化相互叠加，仅使用声学信号估计多个个体的姿态本身极具挑战性。与单人场景不同，多个对象的存在会导致声学特征重叠，使得难以将特定的信号变化归因于某个个体的姿态。此外，人际间的反射进一步加剧了复杂性，其引入了复杂的传播延迟，模糊了时间上的运动-声学关系。为解决这些问题，我们提出了SoundMHPE（基于声音的多人人体姿态估计器），这是一种新颖的编码器-解码器框架，包含两个关键组件。首先，声学多尺度编码器捕获多样的时间特征和细粒度的频率特征。

    arXiv:2609.04902v1 Announce Type: cross  Abstract: Can we recover the 3D poses of multiple people using only sound? This paper presents the first attempt to estimate multi-person 3D poses solely from acoustic signals. Estimating the poses of multiple individuals using acoustic signals is inherently challenging due to the superposition of motion-dependent signal variations. Unlike single-person scenarios, the presence of multiple subjects leads to overlapping acoustic signatures, making it difficult to attribute specific signal changes to an individual's pose. Furthermore, the complexity is compounded by inter-person reflections, which introduce intricate propagation delays that obscure the temporal motion-acoustic relationship. To address these issues, we propose SoundMHPE (Sound-based Multi-person Human Pose Estimator), a novel encoder-decoder framework consisting of two key components. First, the Acoustic Multi-scale Encoder captures diverse temporal and fine-grained frequency featur
    
[^50]: 面向事件时间预测的上下文表格基础模型适配接口

    Adaptation Interfaces for In-Context Tabular Foundation Models in Time-to-Event Prediction

    [https://arxiv.org/abs/2609.04901](https://arxiv.org/abs/2609.04901)

    本文提出了将表格基础模型应用于事件时间预测的多种适配接口（时间零样本重构、分类微调和生存头适配），发现基于Cox的接口是最可靠且表现最强的适配方式，尤其在大规模数据集的综合Brier分数上表现突出，而零样本推理则适用于较小的数据集。

    

    表格基础模型在结构化数据上取得了出色的性能，尤其是在标准分类和回归问题上。然而，将其扩展到带删失的事件时间预测具有挑战性，因为这需要妥善处理删失数据和事件时间动态。在我们先前工作的基础上，我们进一步将表格基础模型与CoxPH和DeepHit联系起来，并修订了上下文重采样的训练流程。我们在74个单风险数据集上，使用冻结的表格基础模型骨干，评估了时间零样本重构、基于分类的微调以及生存头适配方法，并额外研究了4个竞争风险数据集。结果表明，零样本推理在较小的单风险数据集上表现有效，而随着数据集规模的增大，有监督适配的优势日益显著。其中，Cox提供了最可靠且表现最强的适配接口，尤其是在较大数据集上的综合Brier分数（IBS）方面；而DeepHit在时间相关指标上相对更强。

    arXiv:2609.04901v1 Announce Type: cross  Abstract: Tabular foundation models (TabFMs) achieve strong performance on structured data, particularly for standard classification and regression problems. Yet, extending them to censored time-to-event prediction is challenging because it requires properly handling censoring and event-time dynamics. Building on our prior work, we further link TabFMs with CoxPH and DeepHit and revise the context-resampled training procedure. We evaluate temporal zero-shot reformulation, classification-based fine-tuning, and survival-head adaptation using frozen TabFM backbones on 74 single-risk data sets, and we additionally study 4 competing-risk data sets. Zero-shot inference is effective on smaller single-risk data sets, whereas supervised adaptation becomes increasingly advantageous as data sets scale. Cox provides the most reliably strong interface, especially for Integrated Brier Score (IBS) on larger data sets. DeepHit is relatively stronger for the time
    
[^51]: 从语言模型到作用于世界的系统：智能体AI在数字、社交、虚拟和物理环境中的进展与局限

    From Language Models to World-Acting Systems: Progress and Limits of Agentic AI across Digital, Social, Virtual, and Physical Environments

    [https://arxiv.org/abs/2609.04894](https://arxiv.org/abs/2609.04894)

    该批判性综述指出，智能体AI在行动接口扩展方面进展显著，但在稳健的任务完成、授权、恢复与可信委派方面仍存在关键局限，且不应将其视为迈向自主性的单一进程。

    

    当周围系统允许大语言模型的输出改变外部状态时，它们便成为具有实际影响的智能体。模型如今可以调用工具、操作界面、委派任务、保持状态、栖居于生成的世界之中，并控制机器人或实验室设备。这些进展常被叙述为迈向自主性的统一进程，从而混淆了模型能力、系统集成、持久性与安全授权。本批判性综述综合了截至2026年8月31日可获得的原始研究和官方技术规范。我们沿着委派权限、时间持久性和环境耦合三个维度组织证据，同时将模型、运行框架和环境分离开来。在所考察的证据中，行动接口的扩展比稳健的任务完成、恢复、授权或独立验证得到了更有说服力的记录。模型上下文协议和Agent2Agent改进了互操作性，但并未建立可信的委派；多智能体……

    arXiv:2609.04894v1 Announce Type: new  Abstract: Large language models become consequential agents when surrounding systems let outputs change external state. Models now call tools, operate interfaces, delegate work, retain state, inhabit generated worlds, and control robots or laboratory equipment. Such advances are often narrated as one march toward autonomy, conflating model competence, system integration, persistence, and safe authority. This critical review synthesizes primary research and official technical specifications available by 31 August 2026. We organize the evidence along delegated authority, temporal persistence, and environmental coupling, while separating model, harness, and environment. Within the evidence examined, action-interface expansion is documented more convincingly than robust completion, recovery, authorization, or independent verification. Model Context Protocol and Agent2Agent improve interoperability but do not establish trustworthy delegation; multi-age
    
[^52]: 从深到浅：无约束且高效的层合并策略

    From Deep to Shallow: Unconstrained and Efficient Layer Merging Strategy

    [https://arxiv.org/abs/2609.04881](https://arxiv.org/abs/2609.04881)

    该论文提出了一种无需解析解且不增加卷积核大小的高效层合并策略，克服了现有深度压缩方法无法处理带填充卷积的局限，实现了神经网络的高效压缩与推理加速。

    

    尽管深度神经网络已成为机器学习众多领域的基础，但其高昂的计算需求限制了它在资源受限环境中的应用。为了解决这一问题，研究者提出了深度压缩方法，通过识别和线性化冗余的激活函数，从而允许在不引入中间非线性层的情况下合并网络层。然而，这些方法面临两个关键挑战：由于缺乏解析解，它们无法直接应用于带填充的卷积层；此外，它们通常会增大合并后层的卷积核大小，从而限制了加速收益。为了克服这些限制，我们提出了一种高效的策略，使得不存在现有解析解的层也能被合并，且不会增大卷积核大小。我们在多种架构和数据集上验证了该方法，并在真实设备上测量了推理加速效果。

    arXiv:2609.04881v1 Announce Type: new  Abstract: Although Deep Neural Networks have become foundational in many areas of Machine Learning, high computational demands limit their application in resource-constrained environments. To address this issue, depth compression methods have been proposed to identify and linearize redundant activation functions, thereby allowing for the merging of layers without intermediate non-linearities. However, these methods face two key challenges: they cannot be directly applied to convolutions with padding due to the absence of an analytical solution for merging these layers, and they typically increase the kernel size of merged layers, thus limiting speed-up gains. To overcome these limitations, we propose an efficient strategy that enables merging of layers without an existing analytical solution, and also without increasing kernel size. We validate our approach across multiple architectures and datasets, and measure inference speed-up gains on real em
    
[^53]: 当基因组掩码先验无法迁移时：强变异预测，弱功能生成

    When Genomic Masking Priors Fail to Transfer: Strong Variant Prediction, Weak Functional Generation

    [https://arxiv.org/abs/2609.04861](https://arxiv.org/abs/2609.04861)

    GenDA双向扩散模型在变异效应预测上虽超越同规模自回归模型，但熵引导策略对性能提升并无实际贡献，且模型在零样本功能序列填充生成任务上失败，表明基因组掩码先验能很好地迁移到变异预测、却难以迁移到功能生成。

    

    双向离散扩散模型似乎天然适合基因组建模，因为它可以从两侧侧翼序列重建缺失的序列。我们在一个附加假设下开发了GenDA（基因组密度优化的吸收态扩散模型）：熵引导的片段放置会将重建压力集中在组成复杂的区域上，从而同时改进下游变异效应预测和功能序列生成。我们的结果仅部分支持这一前提。经过监督微调后，2.02亿参数的GenDA模型在ClinVar单核苷酸变异（SNV）汇总数据上达到了0.774的AUROC，比同等规模的因果自回归模型高出0.103。然而，一个使用随机片段放置的匹配变体达到了0.777，这表明没有证据显示熵引导是ClinVar性能提升的原因。更出乎意料的是，GenDA在零样本功能填充压力测试中失败：在启动子、增强子、外显子边界和内含子边界上，它无法……（原文摘要在此处截断）

    arXiv:2609.04861v1 Announce Type: new  Abstract: Bidirectional discrete diffusion model appears naturally suited to genomic modeling because it can reconstruct missing sequence from both flanks. We developed GenDA (Genomic Density-optimized Absorbing Diffusion) under the additional hypothesis that entropy-guided span placement would concentrate reconstruction pressure on compositionally complex regions, improving both downstream variant-effect prediction and functional sequence generation. Our results only partially support this premise. After supervised fine-tuning, the 202M-parameter GenDA model reaches a pooled ClinVar SNV AUROC of 0.774, exceeding a similarly scaled autoregressive model by 0.103. However, a matched random-span variant reaches 0.777, providing no evidence that entropy guidance causes the ClinVar improvement. More unexpectedly, GenDA fails a zero-shot functional inpainting stress test: across promoters, enhancers, exon boundaries, and intron boundaries, it does not c
    
[^54]: KVMem：在消费级GPU上虚拟化百万Token智能体工作空间

    KVMem: Virtualizing Million-Token Agent Workspaces on a Consumer GPU

    [https://arxiv.org/abs/2609.04852](https://arxiv.org/abs/2609.04852)

    KVMem提出了一种KV上下文虚拟化系统，将智能体工作空间中溢出的历史记录以分页KV状态保存在GPU内存、主机内存和NVMe之间，并利用模型原生的注意力空间索引按需物化执行视图，从而在百万Token历史的长期智能体基准上实现了更高的任务效用和推理效率。

    

    现代LLM智能体在持久化工作空间中运行，其积累的历史记录可能超出GPU KV缓存容量以及模型的原生上下文窗口。现有系统通常将较旧的上下文压缩为摘要，或稍后以文本形式检索，这要么丢失了细粒度的执行证据，要么反复预填充模型已经处理过的内容。我们提出了KVMem，一个KV上下文虚拟化系统，它将溢出的工作空间历史保存为分页的KV状态，分布于GPU内存、主机内存和NVMe之间。KVMem使用轻量级的、模型原生的注意力空间索引来选择相关的历史块，并物化一个受模型原生上下文窗口限制的、依赖于查询的执行视图。在跨越长达一百万Token历史的长期上下文智能体基准（包括LongMemEval、MemoryAgentBench和AgentLongBench）上的广泛评估表明，KVMem通常实现更高的任务效用和更大的推理效率……

    arXiv:2609.04852v1 Announce Type: new  Abstract: Modern LLM agents operate in persistent workspaces whose accumulated history can exceed both GPU KV capacity and the model's native context window. Existing systems typically compact older context into summaries or retrieve it later as text, either losing fine-grained execution evidence or repeatedly prefilling content that the model has already processed. We present KVMem, a KV-context virtualization system that preserves overflowed workspace history as paged KV state across GPU memory, host memory, and NVMe. KVMem uses lightweight, model-native attention-space indexes to select relevant historical blocks and materializes a query-dependent execution view bounded by the model's native context window. Extensive evaluations on long-context agent benchmarks spanning histories up to one million tokens, including LongMemEval, MemoryAgentBench, and AgentLongBench, show that KVMem generally achieves higher task utility and greater inference eff
    
[^55]: 面向弹性远程机器人控制的控制与无线世界模型耦合方法

    Coupled Control and Wireless World Models for Resilient Remote Robotic Control

    [https://arxiv.org/abs/2609.04851](https://arxiv.org/abs/2609.04851)

    本文提出将控制世界模型与无线JEPA世界模型相耦合，从视觉观测和射频表示中联合预测机器人动力学与无线信道演化，从而实现预测性通信调度，在通信资源受限的不稳定无线网络下实现弹性远程机器人控制。

    

    在无线网络上运行的远程机器人系统，必须在通信资源有限、信道条件不断变化以及存在环境干扰的情况下保持可靠控制。然而，持续传输高维感知观测数据（如相机图像）会增加通信开销和能耗，同时降低在不稳定连接条件下的鲁棒性。为应对这些挑战，本文提出了一种具备弹性的通信感知远程机器人控制框架，该框架基于耦合的控制世界模型与无线联合嵌入预测架构（JEPA）世界模型，从视觉观测以及基于频谱图和持久图像（PI）的原始与结构化射频（RF）表示的组合中，联合捕获机器人动力学与无线信道演化。所学习到的潜在表示能够通过联合预测未来机器人状态来实现预测性通信调度。

    arXiv:2609.04851v1 Announce Type: cross  Abstract: Remote robotic systems operating over wireless networks must maintain reliable control despite limited communication resources, changing channel conditions, and environmental disturbances.However, continuously transmitting high-dimensional sensory observations, such as camera images, increases communication overhead and energy consumption while reducing robustness under unreliable connectivity.To address these challenges, this paper proposes a resilient communication-aware remote robotic control framework based on coupled control and wireless Joint Embedding Predictive Architecture (JEPA) world models that jointly capture robot dynamics and wireless channel evolution from visual observations and a combination of raw and structured radio frequency (RF) representations based on spectrograms and Persistence Images(PIs).The learned latent representations enable predictive communication scheduling by jointly forecasting future robot states 
    
[^56]: PACE：面向一次性个性化联邦图学习的传播感知协同校正

    PACE: Propagation-Aware Collaborative Correction for One-Shot Personalized Federated Graph Learning

    [https://arxiv.org/abs/2609.04832](https://arxiv.org/abs/2609.04832)

    PACE提出了一种传播感知的协同校正方法，将外部协同知识作为对完整本地模型的紧凑校正而非替代品，通过服务器构建接收方锚定的校正并结合CNLL校准自动选择融合系数，有效解决了一次性个性化联邦图学习中的客户端异构性风险问题。

    

    在个性化联邦图学习中，客户端异构性既带来机遇也带来风险。其他子图所持有的知识可能对接收方的本地模型起到补充作用，但不兼容的迁移可能会覆盖原本可靠的预测结果。一次性通信加剧了这种矛盾，因为不合适的服务器返回结果无法在后续得到纠正。我们提出了PACE，它将协同知识视为对完整本地预测器的紧凑校正，而非其替代品。每个客户端上传一个秩为r的更新载体以及传播消息矩的对角草图。服务器利用这些信息构建一个传播感知的、以接收方为锚点的校正，同时接收方保留其完整的本地模型。随后，凸负对数似然校准（CNLL）利用验证节点在本地与外部logits之间选择一个系数；模型参数保持固定，且不会发送任何反馈。在秩为6时，个性化返回占据...

    arXiv:2609.04832v1 Announce Type: new  Abstract: Client heterogeneity creates both an opportunity and a risk in personalized federated graph learning. Knowledge held by other subgraphs may complement a receiver's Local model, but an incompatible transfer can override reliable predictions. One-shot communication sharpens this tension because an unsuitable server return cannot be corrected later. We introduce PACE, which treats collaborative knowledge as a compact correction to a complete Local predictor rather than as its replacement. Each client uploads a rank-r update carrier and a diagonal sketch of propagated message moments. The server uses them to construct a propagation-aware, receiver-anchored correction, while the receiver retains its full Local model. Convex negative-log-likelihood calibration (CNLL) then selects one coefficient between Local and External logits using validation nodes; model parameters remain fixed and no feedback is sent. At Rank-6, personalized returns occup
    
[^57]: 基于逐层多阈值随机草图的通信高效个性化联邦学习

    Communication-Efficient Personalized Federated Learning via Layer-Wise Multi-Threshold Random Sketching

    [https://arxiv.org/abs/2609.04830](https://arxiv.org/abs/2609.04830)

    该论文提出一种逐层多阈值随机草图方法，通过针对不同层的参数分布采用差异化的多阈值量化策略，克服了传统单比特方法单一阈值的局限性，实现了通信高效的个性化联邦学习。

    

    个性化联邦学习（PFL）是分布式设备间协同学习的一种有前景的范式，其中边缘节点在无需共享原始数据的情况下协同训练个性化模型。尽管PFL通过学习客户端特定模型解决了数据异构性问题，但在带宽受限的系统中交换高维参数时，仍然面临巨大的上行和下行通信开销。近期的单比特方法实现了极致压缩，但它们通常依赖于应用于整个模型的单一阈值规则。这种设计存在两个局限性：首先，它忽略了各层之间参数分布和量化敏感性的差异；其次，单一阈值仅提供粗略的二值信息，无法捕捉参数分布中的细粒度变化。为解决这些问题，我们提出了一种基于逐层多阈值随机草图的通信高效PFL框架。

    arXiv:2609.04830v1 Announce Type: new  Abstract: Personalized federated learning (PFL) is a promising paradigm for collaborative learning over distributed devices, where edge nodes collaboratively train personalized models without sharing raw data. Although PFL addresses data heterogeneity by learning client-specific models, it still suffers from substantial uplink and downlink communication costs when exchanging high-dimensional parameters in bandwidth-constrained systems. Recent one-bit methods achieve extreme compression, but they usually rely on a single thresholding rule applied to the whole model. This design has two limitations. First, it overlooks layer-wise differences in parameter distributions and quantization sensitivities. Second, a single threshold provides only coarse binary information and cannot capture fine-grained variations in parameter distributions. To address these issues, we propose a communication-efficient PFL framework via layer-wise multi-threshold random sk
    
[^58]: 基于扩散模型的局部内在维度估计的极小极大下界

    Minimax Lower Bound for Estimating Diffusion-based Local Intrinsic Dimension

    [https://arxiv.org/abs/2609.04822](https://arxiv.org/abs/2609.04822)

    本文首次研究了基于扩散的局部内在维度估计的统计难度，证明了有限尺度场与流形维度d的偏差至多为O(σ²)，并建立了阶为(nσ^d)^{-1}的极小极大下界。

    

    虽然基于扩散的方法近来已成为探测高维数据内在几何结构的有效工具，但其统计难度在很大程度上仍未被探索。我们研究了FLIPD（Kamkari等，2024；arXiv:2406.03537）背后的有限尺度总体泛函的估计问题，这是一种基于扩散的局部内在维度（LID）量，通过高斯平滑密度的对数尺度导数来定义。直观地说，高斯平滑将局部维度转化为尺度定律：在d维流形附近，核质量以σ^d的方式增长，因此对噪声尺度求导可以揭示内在指数。在正则流形模型下，我们证明在该模型类上一致地，有限尺度场与流形维度d的偏差至多为O(σ²)。然后，我们建立了估计该有限尺度场的阶为(nσ^d)^{-1}的极小极大下界。

    arXiv:2609.04822v1 Announce Type: cross  Abstract: While diffusion-based methods have recently emerged as effective tools for probing the intrinsic geometry of high-dimensional data, their statistical difficulty remains largely unexplored. We study estimation of the finite-scale population functional underlying FLIPD (Kamkari et al., 2024; arXiv:2406.03537), a diffusion-based local intrinsic dimension (LID) quantity defined through the logarithmic scale derivative of a Gaussian-smoothed density. Intuitively, Gaussian smoothing turns local dimension into a scale law: near a $d$-dimensional manifold, the kernel mass grows like $\sigma^d$, so differentiating with respect to the noise scale reveals the intrinsic exponent. Under a regular manifold model, we show uniformly over the model class that the finite-scale field differs from the manifold dimension $d$ by at most $O(\sigma^2)$. We then establish a minimax lower bound of order $(n\sigma^d)^{-1}$ for estimating this finite-scale field 
    
[^59]: 基于梯度更新中威胁指标对比编码的联邦攻击活动检测

    Federated Attack Campaign Detection via Contrastive Encoding of Threat Indicators in Gradient Updates

    [https://arxiv.org/abs/2609.04815](https://arxiv.org/abs/2609.04815)

    提出FedIoC联邦学习框架，客户端通过监督对比学习将本地威胁指标编码进梯度更新，使监控同一攻击活动的客户端产生对齐的梯度分量，从而在无需共享敏感数据的前提下实现跨组织的协调网络攻击活动检测。

    

    检测跨越多个组织的协调网络攻击活动，传统上需要在机构边界和国家边境之间共享敏感遥测数据与威胁情报，而联邦学习通过直接在本地数据上训练共享威胁检测器消除了这一障碍。我们提出了FedIoC，这是一个模块化框架，其中客户端将本地可用的结构化威胁指标融入其梯度更新中；我们采用对IoC匹配流量的监督对比损失来实例化客户端编码器。在每个训练批次内，匹配任何已知指标模式的流量构成正样本集；对比目标将其学习到的嵌入拉近，并将非IoC嵌入推开，从而使得攻击活动相关的结构在设计上就被表达在梯度方向中。共享相同攻击活动指标的客户端随后会产生对齐的梯度分量，服务器集群……

    arXiv:2609.04815v1 Announce Type: new  Abstract: Detecting orchestrated cyberattack campaigns that span multiple organizations traditionally requires sharing sensitive telemetry and threat intelligence across institutional boundaries and country borders, a barrier that Federated Learning removes by training shared threat detectors directly on local data. We propose FedIoC, a modular framework in which clients fold locally available structured threat indicators into their gradient updates; we instantiate the client-side encoder with a supervised contrastive loss over IoC-matched flows. Within each training batch, flows that match any known indicator pattern form the positive set; the contrastive objective pulls their learned embeddings together and pushes non-IoC embeddings away, so that campaign-relevant structure is, by design, expressed in the gradient direction. Clients sharing indicators for the same attack campaign then produce aligned gradient components, which the server cluster
    
[^60]: 销售预测中的归因有多忠实？一项反事实研究

    How Faithful Is Attribution for Sales Forecasting? A Counterfactual Study

    [https://arxiv.org/abs/2609.04797](https://arxiv.org/abs/2609.04797)

    该研究为多序列WaveNet销售预测模型添加了反事实可解释性层，其归因分解可精确加总至预测值，并通过删除/插入测试以统计显著性验证了归因忠实反映模型的真实行为而非伪影。

    

    深度销售预测模型（如WaveNet风格的膨胀卷积网络）虽然准确但不透明：当单一模型为众多序列中的某一个预测销量时，它无法解释原因。我们在一个基于完整Corporacion Favorita食品杂货数据集（1,688天内共174,685个序列）训练的多序列WaveNet预测器上，添加了一个事后、与架构无关的反事实可解释性层。该方法将每个预测分解为精确加总等于预测值的贡献分量，从而避免了我们在加性SHAP式归因中观察到的分配伪影问题。我们使用删除/插入协议评估归因的忠实性，发现两项测试均具有统计显著的效果（删除差距0.22，p<0.001；插入差距0.27，p<0.01；且在五个背景采样种子下结果稳健），从而证明这些归因反映的是真实的模型行为，而非看似合理的伪影。随后我们诚实地刻画了……（摘要在此处截断）

    arXiv:2609.04797v1 Announce Type: new  Abstract: Deep models for sales forecasting, such as WaveNet-style dilated convolutional networks, are accurate but opaque: when a single model predicts sales for one of many series, it offers no account of why. We add a post-hoc, architecture-agnostic counterfactual interpretability layer to a multi-series WaveNet forecaster trained on the full Corporacion Favorita grocery dataset (174,685 series over 1,688 days). The method decomposes each forecast into contributions that sum exactly to the predicted value, avoiding the allocation artifacts we observed with additive SHAP-style attribution. We evaluate faithfulness with a deletion/insertion protocol and find a statistically significant effect on both tests (deletion gap 0.22, p<0.001; insertion gap 0.27, p<0.01; robust across five background-sampling seeds), establishing that the attributions reflect genuine model behavior rather than plausible-looking artifacts. We then characterize, honestly, w
    
[^61]: 学习增强算法：性能保证、构造机制与系统级意义

    Learning-Augmented Algorithms: Guarantees, Construction Mechanisms, and System-Level Implications

    [https://arxiv.org/abs/2609.04787](https://arxiv.org/abs/2609.04787)

    本综述系统梳理了学习增强算法领域的预测接口、误差度量、一致性—鲁棒性权衡及五种代表性构造机制，明确区分形式化保证与实证系统证据，并指出了该领域的未来开放问题。

    

    学习增强算法在利用可能出错的预测的同时保持正式的性能保证。本综述综合梳理了预测接口、误差度量、一致性—鲁棒性权衡，以及涵盖在线优化、缓存、学习型数据结构、图问题和机制设计等领域的五种代表性构造机制。综述还引入了一个正交的定理层面维度，用于区分所达到的上界与匹配的渐近依赖关系。正式的理论保证与实证系统证据被分开处理，并对预测成本、反馈和组合进行了明确分析。由此形成的综合性阐述给出了有限端到端推理的充分条件，并勾画了成本感知预测、内生误差、语义预测器和基准测试方面的开放问题。

    arXiv:2609.04787v1 Announce Type: new  Abstract: Learning-augmented algorithms use fallible predictions while retaining formal performance guarantees. This survey synthesizes prediction interfaces, error measures, consistency--robustness trade-offs, and five representative construction mechanisms across online optimization, caching, learned data structures, graph problems, and mechanism design. An orthogonal theorem-level axis distinguishes achieved upper bounds from matched asymptotic dependence. Formal guarantees are separated from empirical systems evidence, with explicit treatment of prediction cost, feedback, and composition. The resulting synthesis states sufficient conditions for limited end-to-end reasoning and delineates open problems in cost-aware prediction, endogenous error, semantic predictors, and benchmarking.
    
[^62]: 动态异构图表示学习：综述

    Dynamic Heterogeneous Graph Representation Learning: A Survey

    [https://arxiv.org/abs/2609.04779](https://arxiv.org/abs/2609.04779)

    本综述首次系统性地回顾了动态异构图表示学习领域，提出了一个统一涵盖离散时间与连续时间的动态异构图形式化定义，并据此建立了一种新颖的以算法为中心的分类体系。

    

    图表示学习（GRL）是建模复杂网络的经典范式。然而，现实世界的人工智能系统本质上表现为不断演化的异构实体及其复杂的交互关系，这给静态或同构的建模方法带来了重大挑战。为应对这些复杂性，动态异构图（DHG）表示学习已成为一种重要方法，能够学习同时保留结构语义和时间动态特性的低维表示。本综述首次对动态异构图表示学习方法进行了系统性回顾。我们首先从时间粒度的视角出发，引入了一个涵盖离散时间和连续时间动态异构图的统一形式化定义。基于这一形式化框架，我们提出了一种新颖的以算法为中心的分类体系，对现有文献进行归类，包括早期基于嵌入的方法、图神经网络（GNN）方法等。

    arXiv:2609.04779v1 Announce Type: cross  Abstract: Graph representation learning (GRL) serves as a canonical paradigm for modeling complex networks. However, real-world AI systems inherently manifest as evolving heterogeneous entities with complex interactions, posing significant challenges to static or homogeneous modeling. To address these complexities, representation learning for Dynamic Heterogeneous Graphs (DHGs) has emerged as a vital approach for learning low-dimensional representations that simultaneously preserve structural semantics and temporal dynamics. This survey presents the first systematic review of DHG representation learning methods. We first introduce a unified formal definition that encompasses both discrete-time and continuous-time DHGs from the perspective of temporal granularity. Building upon this formulation, we propose a novel algorithm-centric taxonomy that categorizes existing literature, including early embedding-based approaches, graph neural network (GNN
    
[^63]: 面向工具使用智能体的持久教师锚定

    Persistent Teacher Anchoring for Tool-Using Agents

    [https://arxiv.org/abs/2609.04773](https://arxiv.org/abs/2609.04773)

    提出持久教师锚定（PTA），一种由学生诱导但由教师承诺的 rollout 构建方法，通过在块级验证基础上增加轮级承诺让工具调用得以执行，从而解决工具使用中师生分布差距累积导致的漂移问题。

    

    蒸馏在大语言模型后训练中十分常见，其中在策略知识蒸馏（OPKD）利用学生生成的轨迹为学生参与下游强化学习做好准备。在每个状态，学生需要匹配由教师提供的下一个词元分布。当 rollout 进入教师不会访问的状态时，师生分布之间的差距会不断累积。在工具使用场景中，这种差距的影响尤为严重，因为学生编写的调用会在监督之前执行，其返回的观察结果会塑造后续的前缀内容。提议者-验证者生成方式通过让教师在生成过程中决定保留哪些学生提出的文本，来缓解这种漂移。然而，现有方案仅对文本进行管控，将工具执行排除在其作用范围之外。我们提出持久教师锚定（PTA），这是一种由学生诱导、但由教师承诺的 rollout 构建方法。PTA 在保留块级验证的同时增加了轮级承诺，使调用能够真正到达环境并被执行。

    arXiv:2609.04773v1 Announce Type: cross  Abstract: Distillation is common in LLM post-training, where on-policy knowledge distillation (OPKD) uses student-generated trajectories to prepare the student for downstream RL. At each state, the student matches a next-token distribution supplied by the teacher. As the rollout enters states the teacher would not visit, the teacher-student distribution gap can accumulate. In tool use, this gap becomes consequential because student-written calls execute before supervision and their observations shape later prefixes. Proposer-verifier generation addresses this drift by letting the teacher decide which student-proposed text is retained during generation. Existing formulations govern text but leave tool execution outside their scope. We propose Persistent Teacher Anchoring (PTA), a student-induced but teacher-committed rollout construction. PTA retains chunk-level verification and adds turn-level commitment, allowing a call to reach the environment
    
[^64]: 一种用于图神经网络（GNN）所有权验证的鲁棒的基于水印的指纹框架

    A Robust Watermark-based Fingerprint Framework for GNNs Ownership Verification

    [https://arxiv.org/abs/2609.04772](https://arxiv.org/abs/2609.04772)

    本文提出REMARK框架，一种用于GNN所有权验证的鲁棒水印指纹方法，通过精心构造分布内水印图等手段，克服了现有方法在模型性能退化、不现实的代理模型假设以及过度依赖特定输出层三个方面的局限性。

    

    图神经网络（GNN）高昂的训练成本引发了人们对模型所有权侵权问题日益增长的关注，例如模型窃取和未经授权的滥用。为了验证模型所有权并防止重大经济损失，研究者们已提出两类GNN所有权验证（OV）方法：基于水印的方法和基于指纹的方法。然而，这些方法通常面临三个局限性：（1）相对于训练集呈分布外（OOD）的水印图会导致受保护模型的性能下降；（2）假设代理模型在包含水印的训练集上训练是不切实际的；（3）过度依赖特定输出层进行指纹提取。在本文中，我们提出了一种用于GNN的鲁棒的基于水印的指纹框架，命名为REMARK。REMARK首先生成精心设计的分布内水印图，以使输出差异最大化……

    arXiv:2609.04772v1 Announce Type: new  Abstract: The high training cost of Graph Neural Networks (GNNs) has raised growing concerns regarding model ownership infringement, such as model stealing and unauthorized misuse. To verify model ownership and prevent significant economic losses, two groups of GNN Ownership Verification (OV) methods have been proposed: watermark-based methods and fingerprint-based methods. However, these methods typically face three limitations: (1) the performance degradation of protected models caused by out-of-distribution (OOD) watermark graphs with respect to the training set; (2) the unrealistic assumption that surrogate models have been trained on a watermark-containing training set; and (3) over-reliance on specific output levels for fingerprint extraction. In this paper, we propose a Robust watErMArk-based fingeRprint frameworK for GNNs, named REMARK. REMARK first generates carefully crafted in-distribution watermark graphs that maximize output differenc
    
[^65]: 超越平稳客户端不可用性的韧性：解锁高效且无偏的联邦学习

    Resilience Beyond Stationary Client Unavailability: Unlocking Efficient and Unbiased Federated Learning

    [https://arxiv.org/abs/2609.04763](https://arxiv.org/abs/2609.04763)

    提出FedSWE算法，可高效且无偏地应对联邦学习中异构、非平稳的随机客户端可用性问题，且无需以往方法那样高昂的内存与计算开销。

    

    由于资源限制或外部与内部的不确定性，现实世界联邦学习系统中的客户端往往是间歇性可用的边缘设备。在高度动态的环境中，参数服务器缺乏对客户端可用性的先验实时认知，这使得传统联邦学习算法难以适应客户端可用性中的不确定性。如果不加以妥善处理，复杂的客户端可用性可能引入显著偏差，从而损害所训练模型的性能。以往的大多数工作要么未能考虑非平稳的客户端可用性动态，要么需要大量的内存和计算开销。本文旨在开发对异构且非平稳的随机客户端可用性具有可证明韧性的高效联邦学习算法。我们提出了FedSWE，它采用了新颖的算法结构以补偿错过的……（摘要在此处截断）

    arXiv:2609.04763v1 Announce Type: new  Abstract: Due to resource constraints or external and internal uncertainties, clients in real-world federated learning systems are often intermittently available edge devices. In highly dynamic environments, the parameter server lacks prior real-time knowledge of clients' availability, making it challenging to adapt traditional federated learning algorithms to be resilient to uncertainties in client availability. If not carefully addressed, complex client availability can introduce significant bias, potentially harming the performance of the trained model. Most prior work either fails to account for non-stationary client availability dynamics or demands significant memory and computational overhead. This paper aims to develop efficient federated learning algorithms that are provably resilient to heterogeneous and non-stationary stochastic client availability. We propose FedSWE, which admits novel algorithmic structures to (i) compensate for missed
    
[^66]: Duckworth-Lewis-Stern（DLS）方法的公平性审计：特定赛制与性别差异偏差，以及用于板球目标分数修正的可解释校准层

    A Fairness Audit of the Duckworth-Lewis-Stern Method: Format-Specific and Gender-Differential Bias, with an Interpretable Calibration Layer for Cricket Target Revision

    [https://arxiv.org/abs/2609.04754](https://arxiv.org/abs/2609.04754)

    该论文首次对板球DLS目标修正方法进行大规模公平性审计，发现其预测误差随比赛状态变化跨度高达137分，且在单日国际赛中存在显著的性别偏差（女性比赛的平均过高预测比分性比赛多6.13分），并提出一个可解释的校准层来修正这些问题。

    

    Duckworth-Lewis-Stern（DLS）方法自1999年以来一直是因雨中断的有限轮次板球比赛中修正目标分数的国际标准。尽管已被实际使用超过二十年，但目前尚无对其预测偏差的大规模实证审计发表。我们对来自Cricsheet的8,150场国际比赛（3,095场单日国际赛ODI、5,055场T20国际赛）进行了此类审计，生成了233,550个带时间划分的合成中断场景。我们记录了两种结构性偏差。首先，DLS的预测误差在不同的（剩余轮数、失掉三柱门数）比赛状态分桶之间跨度达137分。其次，DLS在单日国际赛中表现出一种此前从未被量化的性别差异偏差：在训练集上，男性比赛的平均过高预测为+1.51分，而女性比赛为+7.63分，差距达+6.13分（F = 195.16，p < 10⁻⁴³）。我们将DLS与五种现代替代方法进行基准比较：Bi-LSTM、XGBoost、增强型XGBoost变体，以及一种深度上下文感知模型。

    arXiv:2609.04754v1 Announce Type: new  Abstract: The Duckworth-Lewis-Stern (DLS) method has been the international standard for revising target scores in rain-interrupted limited-overs cricket since 1999. Despite over two decades of operational use, no large-scale empirical audit of its prediction bias has been published. We conduct such an audit on 8,150 international matches (3,095 ODIs, 5,055 T20Is) from Cricsheet, generating 233,550 synthetic interruption scenarios with temporal splits. We document two structured biases. First, DLS prediction error spans a 137-run range across (overs-remaining, wickets-lost) match-state buckets. Second, DLS exhibits a gender-differential bias on ODIs that has not previously been quantified: on the training split, mean over-prediction is +1.51 runs for men but +7.63 runs for women, a gap of +6.13 runs (F = 195.16, p < 10^-43). We benchmark DLS against five modern alternatives: Bi-LSTM, XGBoost, an enriched XGBoost variant, a deep context-aware model
    
[^67]: 相同请求，不同答案：量化放大了LLM服务中由缓存引起的分歧

    Same Request, Different Answer: Quantization Amplifies Cache-Induced Divergence in LLM Serving

    [https://arxiv.org/abs/2609.04748](https://arxiv.org/abs/2609.04748)

    该论文首次系统量化了LLM服务中前缀缓存对推理可复现性的破坏，并揭示权重量化会显著放大这一效应——16位精度下36.2%的智能体回合轨迹发生偏离，而在4位量化下这一比例攀升至75.0%。

    

    前缀缓存是指服务引擎在多个请求之间复用共享提示前缀的键值张量，主流开源技术栈默认启用该功能，并将其视为一种透明的优化手段。我们测量了它对可复现性造成的代价，发现这一代价随着权重量化而急剧上升。在固定模型、解码参数、随机种子和请求顺序，并以批大小为1串行发出每个请求的条件下，我们在两个引擎和四种权重格式上运行了一个八十回合的多轮智能体工具调用工作负载，分别测试了启用与禁用缓存的情况。启用缓存后，智能体的执行轨迹在16位精度下有36.2%的回合发生改变，而在4位量化下这一比例达到75.0%，且这一梯度在受控缓存配置下的重复测量中依然成立。在禁用缓存的情况下，所有配置中的重复执行均实现比特级完全一致（800个回合中0个出现偏差），这将其他来源的非确定性上限界定为……

    arXiv:2609.04748v1 Announce Type: cross  Abstract: Prefix caching, in which a serving engine reuses the key and value tensors of a shared prompt prefix across requests, is enabled by default in the major open-source stacks and treated as a transparent optimization. We measure what it costs in reproducibility, and find that the cost rises sharply with weight quantization. Holding the model, decoding parameters, seed, and request order fixed, and issuing every request serially at batch size one, we ran an eighty-episode multi-turn agentic tool-use workload with caching enabled and disabled across two engines and four weight formats. Enabling the cache changed the agent's trajectory on 36.2 percent of episodes at 16-bit precision and on 75.0 percent at four-bit, a gradient that survives re-measurement under a controlled cache configuration. With caching disabled, repeated execution was bit-identical in every configuration, 0 of 800 episodes, which bounds other sources of nondeterminism at
    
[^68]: 通过合成任务扩展训练大语言模型以实现小分子设计

    Training Large Language Models for Small-Molecule Design with Synthetic Task Scaling

    [https://arxiv.org/abs/2609.04735](https://arxiv.org/abs/2609.04735)

    本文提出基于课程学习的合成任务扩展训练方法，通过逐步引入难度递增的廉价合成设计任务来训练大语言模型，使其习得的分子设计策略能够泛化到评估成本高昂的真实分子先导化合物优化场景。

    

    设计可行的候选药物需要在组合规模巨大且崎岖不平的化学空间中搜索能够满足多个（且往往相互竞争的）目标的分子。大语言模型（LLMs）因其强大的表征能力、推理能力以及在整合外部环境信息方面的灵活性，为这一问题提供了有用的生成式先验。虽然基于可验证奖励的强化学习（RLVR）可用于提升LLMs的能力，但许多与化学相关的打分函数单次评估就需要数小时甚至数天，使得在在线训练过程中直接使用它们的成本过于高昂。本文研究了LLMs能否从更廉价的合成任务中学习分子设计策略，并将所学泛化到代价高昂的分子先导化合物优化场景中。我们发现，逐步纳入更具挑战性合成设计任务的基于课程学习的训练方案，能够实现……（摘要内容在此处被截断）

    arXiv:2609.04735v1 Announce Type: new  Abstract: Designing viable drug candidates requires searching a combinatorially large and rugged chemical space for molecules that satisfy multiple, often competing, objectives. Large language models (LLMs) provide a useful generative prior for this problem because of their representational capacity, reasoning ability, and flexibility when incorporating information from the external environment. While reinforcement learning from verifiable rewards (RLVR) can be used to improve the capabilities of LLMs, many chemically relevant scoring functions require hours or even days per evaluation, making them prohibitively expensive to use directly during online training. Here, we investigate whether LLMs can learn molecular design strategies from cheaper synthetic tasks that generalize to expensive molecular lead optimization settings. We find that curriculum-based training recipes that gradually incorporate more challenging synthetic design tasks enable st
    
[^69]: 超越注意力机制：定位与引导语言模型的拒绝行为

    Locating and Steering Refusal Beyond Attention

    [https://arxiv.org/abs/2609.04721](https://arxiv.org/abs/2609.04721)

    该研究发现拒绝行为的安全表示可跨架构迁移：仅通过一次刚体旋转即可将Transformer的表示空间与状态空间模型（SSM）对齐，使在Transformer上训练的有害探针能识别SSM的有害输入，而移除该对齐方向即可解除模型的拒绝行为。

    

    拒绝行为究竟存在于语言模型内部的何处？当模型架构改变时，这个位置是否会随之改变？在Transformer中，拒绝行为由残差流中的单一方向所主导，这一发现如今已成为安全与可解释性工具的基础。状态空间模型（SSM）通过循环更新而非注意力机制来传递信息，与Transformer不共享任何词元混合机制。那么，同样的安全表示能否在这一架构转变中存续，还是必须针对每种架构重新发现？答案是：它存续了下来。一次单一的刚体旋转——它只能重新定向空间而不能重塑空间——就能将一个模型的表示空间与另一个对齐，因此二者真正共享这一表示。在Transformer上训练的有害性探针能够识别SSM的有害输入；而移除对齐后的方向会使模型回答它原本会拒绝的攻击，相比之下，移除同样维度的随机方向的效果要差得多。因架构而异的是……（原文摘要在此处截断）

    arXiv:2609.04721v1 Announce Type: new  Abstract: Where inside a language model does refusal live, and does that place change when the architecture does? In a transformer, refusal is governed by a single direction in the residual stream, a finding that safety and interpretability tooling now depend on. State-space models (SSMs) route information through a recurrent update instead of attention, sharing no token-mixing mechanism with a transformer. Does the same safety representation survive this shift, or must it be rediscovered per architecture? It survives. A single rigid rotation, which can only reorient a space and not reshape it, aligns one model's representation space with another's, so the two genuinely share the representation. A harm probe trained on a transformer then flags an SSM's harmful inputs, and removing the aligned direction makes a model answer attacks it would otherwise refuse, while a random direction of the same size does far less. What is architecture-specific is n
    
[^70]: 无需模拟的带一般生长惩罚的非平衡动态最优传输

    Simulation-free Unbalanced Dynamic Optimal Transport with General Growth Penalty

    [https://arxiv.org/abs/2609.04710](https://arxiv.org/abs/2609.04710)

    该论文提出SUDO，一种无需模拟的非平衡动态最优传输框架，突破了现有方法仅适用于二次（WFR）生长惩罚或依赖昂贵NeuralODE模拟的限制，能够高效支持一般生长惩罚下的单细胞动力学推断。

    

    从非配对的单细胞快照中推断细胞动力学，需要同时对状态转移以及种群的生长或死亡进行建模。非平衡动态最优传输通过沿传输路径对生长施加惩罚来解决这一问题，使得生长惩罚的选择成为编码增殖与凋亡相关生物学先验的关键方式。然而，现有的UDOT求解器要么依赖计算代价高昂的NeuralODE模拟，要么依赖条件路径的解析解，从而将其效率局限于二次惩罚，即Wasserstein-Fisher-Rao（WFR）测地线。为了实现支持一般生长惩罚的高效UDOT求解器，我们首先证明凹生长惩罚会导致生长与传输相互分离的退化解。随后，我们提出了SUDO，一个面向具有一般非（二次）生长惩罚的UDOT的无模拟框架……

    arXiv:2609.04710v1 Announce Type: cross  Abstract: Inferring cellular dynamics from unpaired single-cell snapshots requires modeling both state transitions and population growth or death. Unbalanced dynamic optimal transport (UDOT) addresses this by penalizing growth along transport paths, making the choice of growth penalty a key way to encode biological priors on proliferation and apoptosis. However, existing UDOT solvers either rely on computationally expensive NeuralODE simulations or depend on analytical solutions of conditional paths, restricting their efficiency solely to quadratic penalties, i.e. Wasserstein-Fisher-Rao (WFR) geodesics. To enable an efficient UDOT solver for general growth penalties, we first show that concave growth penalties lead to degenerate solutions where growth and transport are separated. We then introduce \textbf{S}imulation-free \textbf{U}nbalanced \textbf{D}ynamic \textbf{O}ptimal transport (SUDO), a simulation-free framework for UDOT with general non
    
[^71]: 通过经验校准的DVFS实现可持续的边缘视觉：消除无源散热硬件上的热节流

    Sustainable Edge Vision via Empirically Calibrated DVFS: Eliminating Thermal Throttling on Passively Cooled Hardware

    [https://arxiv.org/abs/2609.04705](https://arxiv.org/abs/2609.04705)

    提出一种经验校准的状态感知DVFS调度器，在无源散热的树莓派5上运行YOLOv8n时完全消除了热节流，相比仅温度响应式基线帧率提高6.8%且每帧能耗降低1.9%。

    

    无源散热消除了风扇的能耗开销和机械故障模式，使其对边缘部署具有吸引力，但在无源散热的边缘片上系统（SoC）上进行持续的深度神经网络（DNN）推理时，热节流成为瓶颈。为解决这一问题，我们提出了一种经验校准的、状态感知的动态电压频率调节（DVFS）调度器。与启发式驱动的控制器不同，我们的方法利用时域保护和绝对温度界限，并辅以导数触发器作为防范剧烈热尖峰的保障措施。在运行YOLOv8n的无源散热树莓派5上的评估表明，我们的调度器在持续30分钟的工作负载中消除了所有观察到的热节流事件。相比仅基于温度的反应式基线，它实现了6.8%更高的帧率（Cohen's d = 8.73），同时每帧能耗降低1.9%。此外，我们优化的无源散热调度方案超越了……

    arXiv:2609.04705v1 Announce Type: cross  Abstract: Passive cooling eliminates the energy overhead and mechanical failure modes of fans, making it attractive for edge deployment, yet sustained Deep Neural Network (DNN) inference on passively cooled edge Systems-on-Chip (SoCs) is bottlenecked by thermal throttling. To address this, we propose an empirically calibrated, state-aware Dynamic Voltage and Frequency Scaling (DVFS) scheduler. Unlike heuristic-driven controllers, our methodology utilizes time-domain guards and absolute temperature bounds, with derivative triggers acting as safeguards against sharp thermal spikes. Evaluated on a passively cooled Raspberry Pi 5 running YOLOv8n, our scheduler eliminates all observed thermal throttling events during sustained 30-minute workloads. It outperforms a temperature-only reactive baseline by achieving a 6.8% higher frame rate (Cohen's d = 8.73) while consuming 1.9% less energy per frame. Furthermore, our optimized passive scheduling surpass
    
[^72]: LookThere！基于强化选择的稀疏视觉

    LookThere! Sparse Vision by Reinforced Selection

    [https://arxiv.org/abs/2609.04698](https://arxiv.org/abs/2609.04698)

    提出LookThere端到端强化学习框架，联合训练浅层输入选择器与深层表示提取器，在不依赖辅助信号的情况下实现极端稀疏条件下性能与计算权衡的新帕累托前沿。

    

    视觉Transformer通常将每个图像token视为同等重要，然而对于计算机视觉中的大多数任务，实际上只需要其中一小部分。自适应计算方法通过选择需要处理的token来加速推理，但现有方法在极端稀疏情况下表现不佳，并且需要依赖可能无法泛化的启发式信号，例如token多样性和注意力分数。我们通过LookThere解决了这些局限性，借助端到端的强化学习框架联合训练浅层输入选择器和深层表示提取器，在性能与计算的权衡中实现了新的帕累托前沿。选择器学习“看哪里”，提取器学习“看什么”，两者协同工作，通过只选择对给定任务值得处理的内容来节省计算，而无需依赖辅助信号。我们证明LookThere只选择与任务相关的输入，在高分辨率设置（如交通标志）下的稀疏识别任务中表现出色。

    arXiv:2609.04698v1 Announce Type: cross  Abstract: Vision transformers typically treat every image token as equally important, yet for most tasks in computer vision only a fraction are needed. Adaptive computation methods accelerate inference by choosing which tokens to process, but existing methods struggle at extreme sparsity and require heuristics that may not generalize like token diversity and attention scores. We address these limitations with LookThere, achieving a new pareto frontier in performance-compute trade-offs through an end-to-end reinforcement learning framework that jointly trains a shallow input selector and a deep representation extractor. The selector learns where to look and the extractor learns what to see, together saving computation by selecting only what is worth processing for a given task without relying on auxiliary signals. We show that LookThere only selects the task-specific input, excelling at sparse recognition in high-resolution settings (traffic sign
    
[^73]: 用于中微子望远镜中光子传播的可微分神经代理模型

    A Differentiable Neural Surrogate for Photon Propagation in Neutrino Telescopes

    [https://arxiv.org/abs/2609.04695](https://arxiv.org/abs/2609.04695)

    candela 是一个可微分的 SIREN 神经场，通过学习冰立方探测器的光子格林函数，能够以比传统蒙特卡洛方法快 50-100 倍的速度模拟中微子望远镜中的光子传播，同时将产量精度保持在蒙特卡洛期望值的 2% 以内。

    

    大体积中微子望远镜通过切伦科夫光来推断中微子的性质，但模拟数十亿光子在高度散射的冰或水中的传输在计算上代价高昂。我们提出了 candela，一个可微分的 SIREN 神经场，它学习了冰立方中微子天文台的光子格林函数，该天文台是嵌入南极冰川冰中的立方公里级探测器。给定一个点状能量沉积和传感器，它能够预测传感器处预期的光子产量以及完整的光子到达时间分布。完整事件则通过将带电粒子的能量沉积分解为点状光源，并叠加它们对应的传感器响应来进行模拟。在蒙特卡洛模拟数据上训练后，candela 生成事件的速度比现有方法快 50 到 100 倍，且计算成本随中微子能量的增长仅呈弱相关性。它将光子产量的中位数保持在蒙特卡洛期望值的 2% 以内，时间分布精度保持在蒙特卡洛统计误差水平。

    arXiv:2609.04695v1 Announce Type: cross  Abstract: Large-volume neutrino telescopes infer neutrino properties from Cherenkov light, but simulating the transport of billions of photons through highly scattering ice or water is computationally costly. We introduce candela, a differentiable SIREN neural field that learns the photon Green's function of the IceCube Neutrino Observatory, a cubic-kilometer detector embedded in Antarctic glacial ice. Given a point-like energy deposit and sensor, it predicts the expected photon yield and full arrival-time distribution at the sensor. Complete events are simulated by decomposing charged-particle energy deposits into point-like sources and superposing their predicted sensor responses. Trained on Monte-Carlo simulations, candela generates events $50$--$100\times$ faster than existing methods, with cost scaling only weakly with neutrino energy. It keeps median yields within $2\%$ of the MC expectation and timing distributions at the MC statistical f
    
[^74]: 基于多特征编码与注意力融合的多模态情感识别增强方法

    Enhancing Multimodal Emotion Recognition via Multi-Feature Encoding and Attention-Based Fusion

    [https://arxiv.org/abs/2609.04690](https://arxiv.org/abs/2609.04690)

    本文提出了一种多模态情感识别框架，通过融合Wav2Vec2语义嵌入、MFCC及声学统计特征等多类音频特征、ResNet50-BiLSTM视觉时空特征提取，以及基于多头注意力的特征级融合机制，显著提升了情感识别效果。

    

    多模态情感识别因其在人机交互、远程教育和医疗保健等领域的重要性而受到越来越多的关注。本文提出了一种新颖的多模态情感识别框架，该框架将丰富的音频与视觉特征提取同基于注意力的融合策略相结合。在音频方面，我们提取了三种互补的特征类型：来自Wav2Vec2的语义嵌入、MFCC特征以及音高、能量和节奏等统计声学描述符，并通过BiLSTM对这些特征进行对齐与融合，以捕捉时间依赖性。在视频方面，我们提出了一种ResNet50-BiLSTM架构，将深度残差学习与序列建模相结合，从面部序列中提取富有表现力的时空特征。为了增强多模态协同效应，我们引入了基于多头注意力的特征级融合机制，使模型能够自适应地权衡不同模态的贡献。实验……

    arXiv:2609.04690v1 Announce Type: cross  Abstract: Multimodal emotion recognition has attracted growing interest due to its importance in human-computer interaction, remote education, and healthcare. This paper proposes a novel multimodal emotion recognition framework that integrates rich audio and visual feature extraction with an attention-based fusion strategy. For audio, we extract three complementary feature types: semantic embeddings from Wav2Vec2, MFCC features, and statistical acoustic descriptors such as pitch, energy, and rhythm. These are aligned and fused via a BiLSTM to capture temporal dependencies. For video, we propose a ResNet50-BiLSTM architecture that combines deep residual learning and sequential modeling to extract expressive spatiotemporal features from facial sequences. To enhance multimodal synergy, we introduce a feature-level fusion mechanism based on multi-head attention, allowing the model to adaptively weigh contributions across modalities. Experiments cond
    
[^75]: WEECFP-SuRGE：具有子结构旋转图距离编码的宽嵌入扩展连接性指纹

    WEECFP-SuRGE: Wide Embedded Extended Connectivity Fingerprint with Substructure Rotary Graph-distance Encoding

    [https://arxiv.org/abs/2609.04672](https://arxiv.org/abs/2609.04672)

    该论文提出了无需参数的分子指纹WEECFP以及结合子结构旋转图距离编码（SuRGE）的transformer架构WEECFP-SuRGE，在不使用任何外部预训练的情况下，在TDC ADMET排行榜的22项基准中取得多项第1名和总体领先的回归性能。

    

    我们提出了WEECFP，一种无需参数的1024维连续分子指纹，它将每个Morgan子结构散射到单个向量的约32个带符号位置上；同时提出了WEECFP-SuRGE，一种transformer架构，其自注意力机制将SuRGE（子结构旋转图距离编码）——一种由分子最短路径图距离参数化的类RoPE旋转——应用于WEECFP子结构token。该架构的7模型混合体（WEECFP-SuRGE Blend）在TDC ADMET排行榜上取得了最低的平均回归排名；在TDC ADMET排行榜上总体排名第2（仅次于预训练的MapLight+GNN），并且在所有不使用外部预训练的方法中总体排名第1；在完整的22项基准测试套件中，在Pgp、亲脂性、CYP2D6底物、微粒体清除率和LD50上获得排行榜第1名（WEECFP-NoSuRGE Blend另在HIA上达到第1名）——且全程无需任何外部预训练。在

    arXiv:2609.04672v1 Announce Type: new  Abstract: We introduce WEECFP, a parameter-free 1024-dimensional continuous molecular fingerprint that scatters each Morgan substructure across roughly thirty-two signed positions of a single vector, and WEECFP-SuRGE, a transformer architecture whose self-attention applies SuRGE (Substructure Rotary Graph-distance Encoding) -- a RoPE-like rotation parameterized by molecular shortest-path graph distance -- to WEECFP substructure tokens. A 7-model blend of this architecture (the WEECFP-SuRGE Blend) achieves the lowest average regression rank on the TDC ADMET leaderboard; is #2 overall on the TDC ADMET leaderboard (behind only pretrained MapLight+GNN), and is #1 overall among methods that use no external pretraining; takes leaderboard #1 finishes on Pgp, Lipophilicity, CYP2D6 Substrate, Clearance Microsome, and LD50 (with the WEECFP-NoSuRGE Blend separately reaching #1 on HIA) across the full 22-benchmark suite -- without any external pretraining. On
    
[^76]: 图灵机的可解释性

    Interpretability for Turing Machines

    [https://arxiv.org/abs/2609.04661](https://arxiv.org/abs/2609.04661)

    本文将源自神经网络的可解释性技术“敏感性分析”拓展至图灵机，从理论和实证上证明算法中的对称性与路径分离会诱导敏感性矩阵中的置换对称性和低秩块，并可通过主成分分析和聚类方法恢复算法特征。

    

    我们证明，敏感性分析（susceptibilities）——一种为神经网络开发的可解释性技术——可以通过探测Murfet和Troiani（arXiv:2504.08075）提出的噪声图灵机学习问题的局部损失景观，来识别图灵机中算法结构的存在。我们证明，图灵机所实现算法中的对称性和路径分离会在其敏感性矩阵中诱导出置换对称性和低秩块。我们在一组确定性有限自动机（DFA）上对这一结论进行了实证研究，并证明可以通过主成分分析和聚类方法在敏感性空间中恢复算法特征。

    arXiv:2609.04661v1 Announce Type: new  Abstract: We show that susceptibilities, an interpretability technique developed for neural networks, can identify the presence of algorithmic structure in Turing machines by probing the local loss landscape of a learning problem for noisy Turing machines introduced by Murfet and Troiani (arXiv:2504.08075). We prove that symmetries and path separation in the algorithm implemented by a Turing machine induce permutation symmetries and low-rank blocks in its susceptibility matrix. We study this empirically on a set of deterministic finite automata (DFAs) and demonstrate that algorithmic features can be recovered by principal component analysis and clustering methods in susceptibility space.
    
[^77]: 面向多模态推荐的潜空间对齐推理

    Latent-Aligned Reasoning for Multimodal Recommendation

    [https://arxiv.org/abs/2609.04645](https://arxiv.org/abs/2609.04645)

    提出LARK两阶段潜空间推理框架，通过可学习token与冻结视觉编码器对齐及物品对比学习，解决多模态推荐中VLM多步推理导致的视觉与文本信号衰减（跨模态稀释）问题。

    

    多模态视觉-语言模型（VLMs）在跨模态理解方面展现出卓越的能力，但将其应用于推荐任务时仍存在一个根本性挑战：随着表示在多步推理过程中传播，视觉和文本信号会逐渐衰减——我们将这种现象称为跨模态稀释。为解决这一问题，我们提出了LARK（潜空间对齐推理框架），这是一个在单一VLM内具备互补对齐机制的两阶段潜空间推理框架。在第一阶段，可学习的潜空间token与多步思维链（CoT）推理交错进行，并与冻结的视觉编码器显式对齐，作为视觉检查点，在整个推理链中保留感知细节。在第二阶段，潜空间表示通过桥接MLP进行投影，并采用物品到物品的对比学习进行训练；为防止推理语义发生（摘要在此处截断）……

    arXiv:2609.04645v1 Announce Type: cross  Abstract: Multimodal Vision-Language Models (VLMs) have demonstrated remarkable capabilities in cross-modal understanding, yet a fundamental challenge persists when applying them to recommendation: as representations propagate through multi-step reasoning, both visual and textual signals progressively attenuate - a phenomenon we term cross-modal dilution. To address this, we propose LARK (Latent-Aligned Reasoning frameworK), a two-stage latent reasoning framework with complementary alignment mechanisms within a single VLM. In the first stage, learnable latent tokens are interleaved with multi-step chain-of-thought (CoT) reasoning and explicitly aligned with a frozen vision encoder, serving as visual checkpoints that preserve perceptual details throughout the reasoning chain. In the second stage, the latent representations are projected via a bridge MLP and trained with item-to-item contrastive learning; to prevent the reasoning semantics from fa
    
[^78]: SMILE：连接连续优化与离散符号恢复

    SMILE: Bridging Continuous Optimization and Discrete Symbolic Recovery

    [https://arxiv.org/abs/2609.04639](https://arxiv.org/abs/2609.04639)

    SMILE提出了一种三阶段混合框架，通过数据结构分析、可解释激活网络的连续优化以及结构化剪枝与符号恢复，将连续梯度优化与离散符号搜索相结合，能够高效地从数据中恢复出具有精确符号常数的紧凑数学表达式。

    

    符号回归（SR）从数据中发现闭式数学表达式，提供超越黑盒模型的可解释性。现有方法在组合搜索空间中收敛缓慢，且缺乏利用数据中组合结构的机制。我们提出了SMILE（Sine正弦、Multiplication乘法、Identity恒等、Logarithm对数、Exponential指数），这是一个混合框架，通过三个阶段将基于梯度的连续优化与离散符号恢复统一起来：对数据进行结构分析以识别目标表达式的组合层次结构；通过连续优化学习一个使用可解释激活函数编码目标表达式的网络参数；通过结构化剪枝、系数优化和舍入实现符号恢复。最后这个阶段将学习到的网络提炼成一个具有精确符号常数的紧凑表达式。我们在SRBench上对SMILE进行了评估，涵盖真实表达式和黑盒（数据）场景。

    arXiv:2609.04639v1 Announce Type: new  Abstract: Symbolic regression (SR) discovers closed-form mathematical expressions from data, offering interpretability beyond black-box models. Existing methods suffer from slow convergence in combinatorial search spaces and lack mechanisms to exploit compositional structure in the data. We introduce SMILE (Sine, Multiplication, Identity, Logarithm, Exponential), a hybrid framework that unifies continuous gradient-based optimization with discrete symbolic recovery through three stages: structural analysis of the data to identify the compositional hierarchy of the target expression, continuous optimization to learn parameters of a network that encodes the target expression using interpretable activations, and symbolic recovery through structured pruning, coefficient optimization, and rounding. This final stage distills the learned network into a compact expression with exact symbolic constants. We evaluate SMILE on SRBench across ground-truth and b
    
[^79]: 追踪音频大语言模型中的音频接地与答案选择

    Tracing Audio Grounding and Answer Selection in Audio LLMs

    [https://arxiv.org/abs/2609.04637](https://arxiv.org/abs/2609.04637)

    该研究揭示了音频大语言模型内部音频真正决定答案的机制：训练主要在中间至后期层增强音频对最终预测的影响，而声学信息主要在早至中间层塑造答案选项的表示。

    

    音频大语言模型在音频理解方面已取得进展，但它们仍可能通过文本线索或语言先验进行推理来预测答案，而非依据所提供的音频。一种常见的补救措施是在那些答案无法仅凭文本推断的数据上训练模型。这种方法能够提升性能，但模型内部究竟发生了什么变化仍不清楚。在本文中，我们探究了为了让音频真正决定答案，模型内部必须发生什么。我们的发现有三点：(1) 将音频替换为静音或无关音频时，经过训练的模型的性能下降幅度明显大于预训练模型；(2) 声学信息在早至中间层最强烈地塑造模型对答案选项的表示，而训练主要在中间至后期层中增强了音频信息对最终预测的影响；(3) 训练期间学到的权重……（原文摘要至此截断）

    arXiv:2609.04637v1 Announce Type: cross  Abstract: Audio Large Language Models (Audio LLMs) have advanced in audio understanding, yet they can still predict the answer by reasoning from textual cues or linguistic priors rather than the provided audio. A common remedy is to train models on data whose answers cannot be inferred from text alone. This approach can improve performance, but what changes within the model remains unclear. In this paper, we ask what must happen inside the model for the audio to actually determine the answer. Our findings are threefold. (1) Replacing the audio with silence or unrelated audio causes substantially larger performance degradation in the trained model than in the pretrained model. (2) Acoustic information most strongly shapes the model's representations of the answer choices in early-to-middle layers, while training mainly increases the influence of audio information on the final prediction in middle-to-late layers. (3) The weights learned during tra
    
[^80]: 《因过罕而学不会：预设气旋轨迹降低孟加拉湾海洋模拟器性能》

    Too Rare to Learn: Prescribed Cyclone Tracks Degrade a Bay of Bengal Ocean Emulator

    [https://arxiv.org/abs/2609.04635](https://arxiv.org/abs/2609.04635)

    该研究发现在孟加拉湾神经网络海洋模拟器中引入预设气旋轨迹输入反而会降低预报技巧，原因是气旋事件在训练数据中过于罕见（仅占7.9%的天数），导致这些输入一旦激活就超出模型的训练分布。

    

    神经海洋模拟器正被提议用于气旋频发沿海海域的区域预报，一个自然的设计选择是将气旋作为预设输入提供给网络。我们在孟加拉湾对这一选择进行了测试，发现它是有害的。我们从GLORYS12再分析数据中保留了15个完整的气旋事件（强度从65节到150节），并比较了两个除四个预设气旋轨迹通道外完全相同的U-Net模型。在三个随机种子下，纯海洋模型在每次运行中都优于持续性基准预报，而风暴条件化模型在每次运行中都输给它，两者的技巧范围完全不相交（p = 3.1e-5，跨风暴配对检验）。其原因是暴露频率而非信号内容：这些通道在仅7.9%的训练天数中非零，因此一旦激活便处于训练分布之外。额外的误差恰好落在预设风暴足迹之内，并且在推理阶段用无风暴图替换真实气旋图反而能改善对保留风暴的预报。

    arXiv:2609.04635v1 Announce Type: new  Abstract: Neural ocean emulators are being proposed for regional forecasting in cyclone-exposed coastal seas, and a natural design choice is to hand the network the cyclone as a prescribed input. We test that choice in the Bay of Bengal and find it harmful. We withhold 15 whole cyclones spanning 65 to 150 kt from GLORYS12 reanalysis and compare two U-Nets that are identical except for four prescribed cyclone-track channels. Across three seeds the ocean-only model beats persistence in every run and the storm-conditioned model loses to it in every run, with the two skill ranges disjoint (p = 3.1e-5, paired across storms). The cause is exposure frequency rather than signal content: the channels are non-zero on only 7.9% of training days, so they are out of distribution the moment they activate. The extra error falls inside the prescribed storm footprint, and replacing the real cyclone map with a no-storm map at inference improves held-out storm forec
    
[^81]: SCAPES：面向环境声音的语义条件自回归先验模型

    SCAPES: Semantically Conditioned Autoregressive Prior for Environmental Sounds

    [https://arxiv.org/abs/2609.04634](https://arxiv.org/abs/2609.04634)

    SCAPES是一个轻量级的语义条件自回归生成模型，通过在神经音频编解码器的连续潜在空间上用流匹配建模重叠音频片段的潜在轨迹演化，仅用3600万参数即可在有限数据上实现高保真环境声音的语义控制合成。

    

    随着生成式音频模型日益复杂，合成日常声音的计算与生态成本变得愈发高昂，往往需要工业规模的资源和海量数据集。在本文中，我们提出了SCAPES：一个面向环境声音的语义条件自回归先验模型。SCAPES是一个轻量级、资源高效的生成模型，旨在通过高级语义控制合成高保真的环境纹理声音。通过在神经音频编解码器的连续潜在流形上运行，我们的方法绕过了离散标记化固有的僵化结构约束。我们提出了一种分割策略，将音频分解为重叠的片段，使连续归一化流（CNF）能够利用流匹配（Flow Matching）来建模潜在轨迹的演化。我们的实验表明，一个3600万参数的SCAPES实例可以在有限的、未标注的数据上完成训练（原文在此处截断）。

    arXiv:2609.04634v1 Announce Type: cross  Abstract: As generative audio models grow in complexity, the computational and ecological costs of synthesizing everyday sounds have become increasingly prohibitive, often requiring industrial-scale resources and massive datasets. In this paper, we present SCAPES: a Semantically Conditioned Autoregressive Prior for Environmental Sounds. SCAPES is a lightweight, resource-efficient generative model designed to synthesize high-fidelity environmental textures through high-level semantic control. By operating on the continuous latent manifold of a neural audio codec, our approach bypasses the rigid structural constraints inherent to discrete tokenization. We propose a segmentation strategy that decomposes audio into overlapping segments, enabling a Continuous Normalizing Flow (CNF) to model the evolution of latent trajectories using Flow Matching. Our experiments demonstrate that a 36-million parameter instance of SCAPES can be trained on limited, un
    
[^82]: SiLR：面向LLM工具代理的结构保持准入与过程奖励

    SiLR: Structure-Preserving Admission and Process Reward for LLM Tool Agents

    [https://arxiv.org/abs/2609.04629](https://arxiv.org/abs/2609.04629)

    该论文提出SiLR方法，通过影子执行提案并在分支级违规状态的乘积序下进行结构化准入决策，从理论上证明了任何标量评分都无法可靠替代这种结构化比较，从而解决了LLM工具代理违规后恢复中的标量投影陷阱问题。

    

    LLM工具代理的运行时门控通常被视为一个过滤器。在ReAct循环中，一个被拒绝的提案之后会在相同状态下产生另一个提案，因此该门控实际上是对提案流的一种搜索算子，其准入标准决定了哪些轨迹是可达的。我们研究了违规后恢复准入问题，即当系统仍处于违规状态时必须允许进展通过，并识别出标量投影陷阱：聚合评分门控会接受局部改进的提案，从而使轨迹陷入平台期无法继续提升。SiLR则对每个提案进行影子执行，并在分支级违规状态（过载分支支持和逐分支严重性）的乘积序下进行准入决策。我们证明没有任何标量替代物对该序是可靠的，因此这种失败是表征性的，而非阈值调优的问题。在挖掘的Gym-ANM场景上，SiLR在多动作回合中实现了21/21的恢复率，而终端奖励方法仅为0/21，最佳标量方法为9/21。

    arXiv:2609.04629v1 Announce Type: new  Abstract: A runtime gate for an LLM tool agent is usually cast as a filter. In a ReAct loop a rejected proposal is followed by another at the same state, so the gate is a search operator over the proposal stream whose admission criterion shapes which trajectories are reachable. We study post-violation recovery admission, where progress must be admitted while the system is still in violation, and identify the scalar projection trap: an aggregate-score gate accepts a locally improving proposal and commits the trajectory to a plateau. SiLR instead shadow-executes each proposal and admits it under a product order over the branch-level violation state (overloaded-branch support and per-branch severity). We prove that no scalar surrogate is sound for this order, so the failure is representational, not a matter of threshold tuning. On mined Gym-ANM scenarios, SiLR recovers 21/21 multi-action episodes against 0/21 for terminal and 9/21 for the best scalar
    
[^83]: 量子退火机上求解带时间窗容量约束车辆路径问题的GNN引导图粗化与自适应QUBO罚项方法

    GNN-Guided Graph Coarsening and Adaptive QUBO Penalties for the Capacitated Vehicle Routing Problem with Time Windows on a Quantum Annealer

    [https://arxiv.org/abs/2609.04593](https://arxiv.org/abs/2609.04593)

    本文提出GNN引导的图粗化与自适应QUBO罚项校准方法，在Solomon基准上结合模拟退火与D-Wave量子退火机求解CVRPTW，将平均原始约束违反从33.0降至0。

    

    图粗化可以缩减用量子退火求解车辆路径问题时产生的大型二次无约束二元优化(QUBO)公式。该方法将时间窗兼容的邻近客户合并为超节点，求解缩减后的问题，再将解扩展回原始图。对于带时间窗的容量约束车辆路径问题(CVRPTW)，现有的粗化启发式方法需要针对特定问题族进行调优，且在随机实例上仍不可靠。我们在Solomon基准测试上，使用模拟退火和D-Wave Advantage2处理器来解决这些局限性。我们首先引入了自适应罚项校准：统一的罚项缩放几乎没有效果，而控制内部系数范围则能显著改善原始样本；通过移除非约束性约束、归一化约束性约束并缩放剩余罚项，将平均原始约束违反从33.0降低到0。

    arXiv:2609.04593v1 Announce Type: new  Abstract: Graph coarsening reduces the large Quadratic Unconstrained Binary Optimization (QUBO) formulations arising when vehicle-routing problems are solved by quantum annealing. Nearby customers with compatible time windows are merged into super-nodes, the reduced problem is solved, and the solution is expanded to the original graph. For the Capacitated Vehicle Routing Problem with Time Windows (CVRPTW), existing coarsening heuristics require family-specific tuning and remain unreliable on random instances. We address these limitations on the Solomon benchmark using simulated annealing and a D-Wave Advantage2 processor.   We first introduce adaptive penalty calibration. Uniform penalty scaling has little effect, whereas controlling the internal coefficient range substantially improves raw samples. Removing non-binding constraints, normalising binding ones, and scaling the remaining penalties reduces mean raw constraint violations from 33.0 to 0.
    
[^84]: 隐于寻常注视之中：以视线表征作为扩展现实（XR）中效用与再识别风险的隐私控制

    Hidden In Plain Gaze: Gaze Representations as Privacy Controls for Utility and Re-identification Risk in XR

    [https://arxiv.org/abs/2609.04592](https://arxiv.org/abs/2609.04592)

    该研究发现在XR系统中，选择人工设计的眼动特征作为数据表征可以在保留约85%动作识别准确率的同时，将用户再识别风险降低约一个数量级，从而在特征提取阶段提供一种轻量级的隐私控制手段。

    

    智能扩展现实（XR）系统日益使用眼动和头部追踪来推断用户意图、任务和注意力，但这些信号同样可能暴露生物特征身份。我们研究视线数据表征的选择能否在特征提取阶段、即引入扰动或形式化隐私机制之前，作为一种轻量级隐私控制手段。基于自我中心的HoloAssist数据集，我们在模型容量匹配的条件下比较了三种视线表征：原始视线数据、空间注意力热图以及人工设计的眼动特征。我们在动作识别（作为任务效用）和闭集用户再识别（作为隐私泄露）两个任务上评估每种表征。结果表明，表征的选择会显著改变隐私与效用之间的权衡。人工设计的特征保留了原始视线约85%的动作识别准确率，同时将再识别率降低约一个数量级，在206个身份条件下约为随机概率的四倍。

    arXiv:2609.04592v1 Announce Type: cross  Abstract: Intelligent extended reality (XR) systems increasingly use eye and head tracking to infer user intent, task, and attention, but the same signals can also reveal biometric identity. We study whether gaze data representation choice can serve as a lightweight privacy control at feature extraction, before adding perturbation or formal privacy mechanisms. Using the egocentric HoloAssist dataset, we compare three gaze representations under matched model capacity: raw gaze, spatial attention heatmaps, and engineered eye-movement features. We evaluate each representation on action recognition as task utility and closed-set user re-identification as privacy leakage. Representation choice substantially changes the privacy-utility tradeoff. Engineered features retain roughly 85% of raw gaze's action-recognition accuracy while reducing re-identification by about an order of magnitude, to roughly four times the chance rate across 206 identities. Th
    
[^85]: 有限域求逆中的表示冗余与结构复杂度

    Representation Redundancy and Structural Complexity in Finite-Field Inversion

    [https://arxiv.org/abs/2609.04583](https://arxiv.org/abs/2609.04583)

    该论文证明有限域 \(\mathbb F_{2^n}\) 上有序基与不同坐标求逆映射之间恰好是 n 对一的冗余对应（同一伽罗瓦轨道的基诱导同一映射），并通过比较三种布尔表述的代数次数与联合 ANF 跃变，刻画了表示选择对求逆运算结构复杂度的影响。

    

    为某一数学运算所选的表示方式会同时影响其代数形式与实际学习难度。我们针对 \(\mathbb F_{2^n}\) 上的求逆运算研究这一现象，其中域元素用不同的有序 \(\mathbb F_2\)-基表示。我们证明：两个有序基诱导出相同的坐标求逆映射，当且仅当它们属于同一个伽罗瓦轨道。由于每个轨道的大小为 \(n\)，有序基与不同求逆映射之间的对应关系恰好是 n 对一的。随后我们分析了求逆的三种布尔表述形式：基准表述的代数次数为 \(n-1\)、联合 ANF 跃变为 1；混合表示表述的次数为 \(2(n-1)\)、联合 ANF 跃变为 2；完全原始表述的次数至多为 \(3(n-1)\)、联合 ANF 跃变至少为 \(n\)。在所考察的情形中，穷举计算与理论结果及上界完全吻合。受控实验……

    arXiv:2609.04583v1 Announce Type: new  Abstract: The representation chosen for a mathematical operation can affect both its algebraic form and its empirical learning difficulty. We study this phenomenon for inversion over \(\mathbb F_{2^n}\), with field elements expressed in varying ordered \(\mathbb F_2\)-bases. We prove that two ordered bases induce the same coordinate inversion map if and only if they belong to the same Galois orbit. Since every orbit has size \(n\), the correspondence between ordered bases and distinct inversion maps is exactly \(n\)-to-one. We then analyze three Boolean formulations of inversion. The reference formulation has algebraic degree \(n-1\) and joint ANF leap \(1\), the mixed representation formulation has degree \(2(n-1)\) and joint ANF leap \(2\), and the complete raw formulation has degree at most \(3(n-1)\) and joint ANF leap at least \(n\). Exhaustive computations agree with the theoretical results and bounds in the cases considered. Controlled expe
    
[^86]: 用于随机重排SGD的中心化置换前缀：锐利率、Hölder几何与复合近端扩展

    Centered Permutation Prefixes for SGD with Random Reshuffling: Sharp Rates, H\"older Geometry, and Composite Proximal Extensions

    [https://arxiv.org/abs/2609.04578](https://arxiv.org/abs/2609.04578)

    本文通过中心化置换前缀技术证明了随机重排SGD在分量非凸、无需分量Hessian连续性的条件下即可达到与已知二次下界匹配的 Õ(T^{-2} + n²T^{-3}) 期望收敛速率，并将结果推广到Hölder几何与复合近端问题。

    

    我们研究针对有限和 F(x)=(1/n)∑_{i=1}^n f_i(x) 的随机重排随机梯度下降（SGD with random reshuffling）。对于采用常数分量步长的新鲜重排（fresh reshuffling），如果每个 f_i 具有 L-Lipschitz 梯度，且平均函数 F 是 μ-强凸的并具有 Lipschitz 连续的 Hessian 矩阵，我们证明了最后一个 epoch 的收敛速率 E[F(y_K)−F(x_*)] = Õ(T^{-2} + n²T^{-3})，其中 T = nK，该速率在 (n,K) 依赖关系上匹配了已知的二次下界。各分量函数可以是非凸的，并且不需要分量级的 Hessian 连续性或单独的有界迭代假设。更一般地，ν-Hölder 连续的平均 Hessian 矩阵仅带来额外的 Õ(n^{1+ν}T^{-2-2ν}) 项，因此只要 ν ≥ 1/2，二次速率即可保持。在凸分量的条件下，基于递减步长的结果消除了大 epoch 的要求，并且一旦 nK 超过条件数规模，就能恢复相同的两项规模。

    arXiv:2609.04578v1 Announce Type: cross  Abstract: We study stochastic gradient descent with random reshuffling for finite sums \[ F(x)=\frac1n\sum_{i=1}^n f_i(x). \] For fresh reshuffling with a constant component stepsize, if each $f_i$ has an $L$-Lipschitz gradient and the average $F$ is $\mu$-strongly convex with a Lipschitz-continuous Hessian, we prove the last-epoch rate \[ \mathbb E[F(y_K)-F(x_\star)] =\widetilde O\!\left(T^{-2}+n^2T^{-3}\right), \qquad T=nK, \] matching the known quadratic lower bound in its $(n,K)$-dependence. The components may be nonconvex, and no componentwise Hessian continuity or separate bounded-iterate assumption is required. More generally, a $\nu$-H\"older-continuous average Hessian adds only $\widetilde O(n^{1+\nu}T^{-2-2\nu})$, so every $\nu\ge 1/2$ preserves the quadratic rate. Under convex components, a decreasing-stepsize result removes the large-epoch requirement and recovers the same two-term scale once $nK$ exceeds the condition-number scale. 
    
[^87]: 面向超越过度训练轴的优化器记忆调度

    Optimizer Memory Schedules for Outscaling the Overtraining Axis

    [https://arxiv.org/abs/2609.04577](https://arxiv.org/abs/2609.04577)

    该研究揭示优化器的相对性能与最优超参数会随过度训练程度显著变化（如最优权重衰减约按 sqrt(OT) 缩放、更长训练偏好更长的优化器记忆），并通过引入对数时间权重衰减与动量冷却，使 ADANA 在长训练时长下持续保持超越 AdamW 的扩展优势。

    

    我们研究了优化器在过度训练轴上的扩展表现，并表明优化器的相对性能与最优超参数会随训练时长发生显著变化。特别地，我们研究了矩阵预条件化方法（Muon 和 SOAP）以及动量调度方法（ADANA）相对于 AdamW 的扩展表现。我们在参数量从 51M 到 253M 的模型上、在 1 倍至 256 倍的过度训练（OT）因子范围内对这四种优化器进行了比较，并在每个设置下对基础学习率进行了全面扫描。首选的学习率调度方式可能会在过度训练轴上发生反转，最优权重衰减系数大致按 sqrt(OT) 的比例缩放，而更长的训练时长通常偏好更长的固定记忆。即使在为 AdamW 在每个训练时长上单独调整其固定记忆之后，ADANA 相对于 AdamW 的扩展优势依然存在。对数时间尺度的权重衰减和动量冷却为 ADANA 带来了可观的收益，且这些收益会随着训练时长的增加而不断复合累积。经过这种处理，ADANA 在……（原文在此处截断）

    arXiv:2609.04577v1 Announce Type: new  Abstract: We investigate how optimizers scale across the overtraining axis and show that relative optimizer performance and optimal hyperparameters change substantially with training horizon. In particular, we study how matrix-preconditioned methods (Muon and SOAP) and a momentum-scheduled method (ADANA) scale relative to AdamW. We compare these four optimizers across models from 51M to 253M parameters and overtraining (OT) factors from 1x to 256x, sweeping the base learning rate at every setting. The preferred learning rate schedule can reverse across the overtraining axis, the best weight decay coefficient scales approximately as sqrt(OT), and longer horizons generally favor longer fixed memory. ADANA's scaling advantage over AdamW persists after tuning AdamW's fixed memory separately at each horizon. Log-time weight decay and momentum cooldown provide substantial gains for ADANA that compound as training increases. With this treatment, ADANA ou
    
[^88]: 细粒度混合专家模型中无需训练的激活专家数量减半方法

    Training-Free Halving of Activated Experts in Fine-Grained Mixture-of-Experts Models

    [https://arxiv.org/abs/2609.04575](https://arxiv.org/abs/2609.04575)

    提出一种无需训练的方法，通过将激活的专家数量与归一化参考集大小解耦，在细粒度MoE模型中将激活专家减半，在几乎不损失精度的同时显著降低计算量。

    

    现代细粒度混合专家模型将每个token路由到少量专家，并对其路由概率进行重新归一化。我们证明，这种重新归一化隐式地将专家输出增益校准到训练时的top-k：在推理时减小k不仅改变了使用哪些专家，还改变了专家分支的强度。我们通过激活top k1个专家、同时按top k2个专家的概率质量进行归一化，将这两种效应分离开来，仅引入一个整数超参数，无需任何参数、训练或可测量的计算开销。在Qwen3.6-35B-A3B上，将专家数量从8减至4时，标准重新归一化下MMLU下降4.65分，而在k2=16时仅下降0.35分，同时路由专家的计算量减半。该结果在规模大11倍的Qwen3.5-397B-A17B上得到复现：使用适当的参考集将专家从10减至5仅损失0.55分。

    arXiv:2609.04575v1 Announce Type: cross  Abstract: Modern fine-grained Mixture-of-Experts (MoE) models route each token to a small number of experts and renormalize their router probabilities. We show that this renormalization implicitly calibrates expert output gain to the training top-$k$: reducing $k$ at inference changes not only which experts are used but also the strength of the expert branch. We separate these effects by activating the top $k_1$ experts while normalizing by the probability mass of the top $k_2$ experts, introducing one integer with no parameters, training, or measurable compute overhead. On Qwen3.6-35B-A3B, reducing from 8 to 4 experts causes a 4.65-point MMLU drop under standard renormalization but only 0.35 points with $k_2=16$, while halving routed-expert compute. The result replicates on the $11\times$ larger Qwen3.5-397B-A17B, where reducing from 10 to 5 experts loses only 0.55 points with an appropriate reference set. Removing renormalization entirely is c
    
[^89]: MURAL：基于自适应边学习的多模态不确定性感知推荐

    MURAL: Multimodal Uncertainty-aware Recommendation via Adaptive edge Learning

    [https://arxiv.org/abs/2609.04574](https://arxiv.org/abs/2609.04574)

    MURAL提出统一框架，通过可微分的自适应边学习器动态发现潜在物品关联并引入不确定性感知机制，解决了多模态推荐中静态图结构僵化和噪声模态信号融合导致的两大瓶颈。

    

    多模态图神经网络通过内容特征增强稀疏的交互数据，已成为推荐系统的标准方法。然而，当前架构面临两个瓶颈：一是结构僵化，即依赖静态预计算的相似度图，无法适应不断演变的用户偏好；二是语义脆弱性，即噪声模态信号被不加区分地融合，从而扭曲了协同信号。我们提出MURAL（基于自适应边学习的多模态不确定性感知推荐），这是一个统一框架，将多模态推荐从固定的结构增强转变为动态拓扑发现。为解决结构僵化问题，自适应边学习器结合可微分的检索增强策略与近似最近邻搜索，发现既具备语义自适应性又具备计算可扩展性（O(NlogN)）的潜在物品-物品关联。为解决语义脆弱性问题，一个不确定性…

    arXiv:2609.04574v1 Announce Type: cross  Abstract: Multimodal Graph Neural Networks have become standard for recommendation by augmenting sparse interaction data with content features. Yet current architectures face two bottlenecks: structural rigidity, from a reliance on static precomputed similarity graphs that cannot adapt to evolving preferences; and semantic fragility, where noisy modality signals are indiscriminately fused, distorting the collaborative signal. We propose MURAL (Multimodal Uncertainty-aware Recommendation via Adaptive edge Learning), a unified framework that shifts multimodal recommendation from fixed structural augmentation to dynamic topology discovery. To address structural rigidity, an Adaptive Edge Learner combines a differentiable retrieval-augmented strategy with an approximate nearest neighbor search to discover latent item-item correlations that are both semantically adaptive and computationally scalable (O(NlogN)). To address semantic fragility, an Uncer
    
[^90]: 极度稀疏的监督激励推理能力

    Extremely Sparse Supervision Incentivizes Reasoning Ability

    [https://arxiv.org/abs/2609.04565](https://arxiv.org/abs/2609.04565)

    研究发现在在线策略蒸馏中，仅需对每个推理轨迹中的一两个token（约占总token数的0.05%）进行监督，就能有效激励大语言模型的推理能力，且在多数情况下可匹配甚至超越全token训练的效果。

    

    大语言模型通过有效的后训练展现出日益强大的推理能力。然而，主流的后训练方法在海量token上进行优化，隐含地假设有效的学习必须依赖密集的token训练。我们在在线策略蒸馏（OPD）设置中重新审视了这一假设，该设置天然允许在生成的每个token上提供密集的教师监督。使用Qwen3系列模型，我们发现了一个反直觉的现象：推理能力可以通过极小比例的生成token得到有效激励——每个推理轨迹中仅需一到两个token，仅占全部token的0.05%。令人惊讶的是，尽管训练目标中排除了绝大多数生成token，这种稀疏监督在提升推理能力方面在大多数情况下仍可匹配甚至超越全token训练的效果。该现象在九种教师-学生配置中均被一致观察到。

    arXiv:2609.04565v1 Announce Type: new  Abstract: Large language models demonstrate increasingly strong reasoning capabilities through effective post-training. Yet, prevailing post-training methods optimize over massive numbers of tokens, implicitly assuming that effective learning must be token-intensive. We revisit this assumption in the on-policy distillation (OPD) setting, which naturally admits dense teacher supervision at every generated token. Using the Qwen3 family, we discover a counter-intuitive phenomenon: reasoning can be effectively incentivized by an extremely small fraction of generated tokens--as few as one or two tokens per reasoning trajectory, corresponding to only 0.05% of all tokens. Surprisingly, this sparse supervision in most cases matches or surpasses full-token training in improving reasoning ability, despite excluding the vast majority of generated tokens from the training objective. This phenomenon is consistently observed across nine teacher--student configu
    
[^91]: 表面码解码器基准测试的仿真到真实研究

    A Sim-to-Real Study of Surface-Code Decoder Benchmarking

    [https://arxiv.org/abs/2609.04557](https://arxiv.org/abs/2609.04557)

    本研究利用运行在表面码阈值以下的Willow处理器真实数据，对六种解码器在四级保真度递增的噪声模型下进行基准排名，发现只有为每种操作类型赋予独立错误率的噪声模型才能产生与硬件一致的排名，并首次对NVIDIA的Ising预解码器进行了硬件独立评估。

    

    量子纠错解码器通常在合成的电路级噪声下进行基准测试，其隐含假设是：解码器在该噪声下的排名能够迁移到真实硬件上，且随着噪声模型变得更真实，排名的一致性会提高。首个运行在表面码阈值以下的Willow处理器使我们能够检验这一假设。我们使用保真度递增的四层噪声模型阶梯，在三个码距、两种基和十五种轮数下对照真实数据，对六种解码器进行了排名。结果表明，只有当噪声模型为每种操作类型赋予各自的错误率时，解码器排名才与硬件结果一致；而将模型校准到具体设备虽能改善绝对错误率，却无法提升排名的一致性。此外，我们还首次对NVIDIA的Ising预解码器进行了硬件上的独立评估，包括在其训练感受野以下的码距，以及通过映射到其训练所用的晶格上进行测试。

    arXiv:2609.04557v1 Announce Type: cross  Abstract: Quantum error-correction decoders are typically benchmarked against synthetic circuit-level noise, under the assumption that a decoder's ranking under such noise transfers to hardware and improves as the noise model becomes more realistic. The Willow processor, the first to operate below the surface-code threshold, allows us to test this assumption. We rank a panel of six decoders using a four-rung ladder of noise models with increasing fidelity, evaluated against real data across three code distances, two bases, and fifteen round counts. Rank agreement with hardware appears once the noise model gives each operation type its own error rate. Calibrating the model to the device improves absolute error rates but not rank agreement. We additionally provide the first independent evaluation of NVIDIA's Ising pre-decoder on hardware, at code distances below its training receptive field and via a mapping onto the lattice on which it was traine
    
[^92]: 基于参数化神经算子的可激发与振荡FitzHugh-Nagumo动力学的快速代理建模

    Fast Surrogate Modeling of Excitable and Oscillatory FitzHugh-Nagumo Dynamics with Parametric Neural Operators

    [https://arxiv.org/abs/2609.04549](https://arxiv.org/abs/2609.04549)

    本文提出将参数条件化的傅里叶神经算子（通过FiLM机制注入参数向量）作为FitzHugh-Nagumo系统的快速可微代理模型，并结合分岔分析针对振荡态和可激发态两种动力学状态分别训练算子，从而实现对5维生理参数空间的高效快速探索。

    

    FitzHugh-Nagumo（FHN）系统作为神经元电压动力学的简化模型，捕捉了孤立动作电位以及全脑中观察到的节律性尖峰放电背后的激活-抑制结构。探索其5维生理参数空间对于神经调控以及将电压记录映射回生物物理机制非常重要，但经典的有限差分求解器使得快速参数扫描的代价高昂。我们训练了参数条件化的傅里叶神经算子（FNO）作为FHN电压场和恢复场在一维空间域上的快速、可微代理模型，通过特征级线性调制（FiLM）将参数向量 λ = (D_u, D_v, a, b, τ) 注入到每个傅里叶层中。我们应用分岔分析界定了模型所跨越的两种不同动力学状态——振荡态（持续放电）和可激发态（动作电位传播），并在每种状态下分别训练一个神经算子。

    arXiv:2609.04549v1 Announce Type: new  Abstract: The FitzHugh-Nagumo (FHN) system serves as a simplified model of neuronal voltage dynamics, capturing the activator-inhibitor structure behind both isolated action potentials and the rhythmic spiking seen across the brain. Exploring its 5D physiological parameter space is important for neuromodulation and mapping voltage recordings back to biophysics, yet classical finite-difference solvers make rapid parameter sweeps expensive. We train parameter-conditioned Fourier Neural Operators (FNOs) as fast, differentiable surrogates for the FHN voltage and recovery fields on a one-dimensional spatial domain, conditioning each Fourier layer on the parameter vector $\lambda = (D_u, D_v, a, b, \tau)$ via feature-wise linear modulation (FiLM). We apply a single bifurcation analysis that delimits the two distinct regimes the model spans, oscillatory (tonic firing) and excitable (action-potential propagation), and we train one operator in each. In the
    
[^93]: Mitra-v2 技术报告

    Mitra-v2 Technical Report

    [https://arxiv.org/abs/2609.04540](https://arxiv.org/abs/2609.04540)

    Mitra-v2 是一个仅用合成数据训练的表格基础模型，通过更大更多样的预训练任务分布和小型 2D Transformer 架构，在 300 多个真实数据集上达到与工业级大模型相当的最先进性能，并大幅超越 TabPFN-3。

    

    我们提出了 Mitra-v2，这是一个表格基础模型，在真实世界的分类和回归问题上实现了最先进的性能，应用范围涵盖信贷风险评分、临床预测、设备故障检测和房价估计。Mitra-v2 仅使用合成数据进行训练，其预训练任务分布比 Mitra-v1 规模更大、更加多样化。该模型构建于一个小型二维 Transformer 骨干网络之上，支持更长的上下文和更大的特征空间，改进的优化方法使其能够从这一更大的任务分布中学习。我们在包含 300 多个真实世界数据集的 TabArena 和 TALENT 基准上，采用两种评估协议对 Mitra-v2 进行了评估。在完整的 TabArena 基准上，Mitra-v2 达到了与工业规模模型 TabFM 和 EXAONE Tabular 相当的最先进性能，同时在分类和回归任务中大幅超越 TabPFN-3。Mitra-v2 与 1.6B（参数规模）模型相匹配……

    arXiv:2609.04540v1 Announce Type: new  Abstract: We introduce Mitra-v2, a tabular foundation model that delivers state-of-the-art performance on real-world classification and regression problems, from credit-risk scoring and clinical prediction to equipment-failure detection and house-price estimation. Mitra-v2 is trained only on synthetic data, with a pretraining distribution that is much larger and more diverse than Mitra-v1's. Built on a small 2D Transformer backbone, Mitra-v2 supports longer contexts and larger feature spaces. Improved optimization lets it learn from this larger task distribution. We evaluate Mitra-v2 on the TabArena and TALENT benchmarks, comprising more than 300 real-world datasets under two evaluation protocols. On the full TabArena benchmark, Mitra-v2 delivers state-of-the-art performance at the level of the industry-scale TabFM and EXAONE Tabular models, while surpassing TabPFN-3 by a wide margin in both classification and regression. Mitra-v2 matches the 1.6B
    
[^94]: 蒸馏后的连续扩散语言模型可以用极少步骤——甚至一步——编写代码

    Distilled Continuous Diffusion Language Models Can Write Code in Few Steps---or One

    [https://arxiv.org/abs/2609.04531](https://arxiv.org/abs/2609.04531)

    本文提出0.7B参数的连续扩散代码生成模型PlaidQ，通过分布匹配和成对轨迹监督的蒸馏技术，将去噪轨迹压缩到仅需16步甚至1步即可高效生成代码，性能可与离散扩散语言模型媲美。

    

    语言生成几乎普遍被视为一个顺序过程：自回归模型一次生成一个token，而扩散语言模型则用长轨迹的迭代细化取代了token级别的串行性。在这项工作中，我们介绍了PlaidQ，一个用于代码生成的0.7B参数连续扩散语言模型，并展示了其轨迹可以被激进地蒸馏为仅少数几个去噪步骤——甚至只有一步，从而实现高效的代码生成。PlaidQ将预训练的自回归模型重新用作对连续token嵌入进行双向去噪的模型。我们通过分布匹配来蒸馏PlaidQ以实现少步生成，并通过成对轨迹监督来实现单步生成。在相同模型规模下，PlaidQ在代码生成方面与离散扩散语言模型具有竞争力。蒸馏进一步推动了质量-计算前沿：16步的学生模型在HumanEval和MBPP+上分别达到了31.78和40.49的pass@10。

    arXiv:2609.04531v1 Announce Type: new  Abstract: Language generation is almost universally treated as a sequential process: autoregressive models emit one token at a time, while diffusion language models replace token-level seriality with a long trajectory of iterative refinement. In this work, we introduce PlaidQ, a 0.7B continuous diffusion language model for code generation, and show that its trajectory can be aggressively distilled into only a few denoising steps---or even one, enabling efficient code generation. PlaidQ repurposes a pretrained autoregressive model as a bidirectional denoiser over continuous token embeddings. We distill PlaidQ with distribution matching for few-step generation and paired-trajectory supervision for one-step generation. At matched model scale, PlaidQ is competitive with discrete diffusion language models on code generation. Distillation then shifts the quality--compute frontier: a 16-step student reaches 31.78 and 40.49 pass@10 on HumanEval and MBPP+,
    
[^95]: 一种用于磁化动力学的基于能量的保守-耗散潜在神经演化算子

    An Energy-Based Conservative-Dissipative Latent Neural Evolution Operator for Magnetization Dynamics

    [https://arxiv.org/abs/2609.04530](https://arxiv.org/abs/2609.04530)

    该论文提出一种将卷积自编码器与结构化潜在神经常微分方程耦合的能量基降阶模型，通过学习标量势构造保守-耗散的潜在动力学结构，保证潜在能量沿解单调递减，且无需物理能量标签或时间导数监督即可联合训练，用于微磁磁化动力学的建模。

    

    我们开发了一种基于能量的微磁磁化动力学降阶模型，该模型将卷积自编码器与结构化的潜在神经常微分方程相结合。受Landau-Lifshitz-Gilbert方程进动-耗散结构的启发，潜在向量场由学习到的标量势的梯度经过一个反对称算子和一个对称半正定耗散算子生成。该势在非唯一的潜在坐标中学习得到，且不等同于吉布斯自由能，但它沿自治连续时间解单调递减，同时反对称分量允许系统沿其水平集运动。编码器、解码器、潜在能量以及算子仅在短轨迹窗口上使用潜在损失和解码滚动损失进行联合训练，无需时间导数监督、物理能量标签或耗散惩罚项。

    arXiv:2609.04530v1 Announce Type: new  Abstract: We develop an energy-based reduced-order model for micromagnetic magnetization dynamics that couples a convolutional autoencoder to a structured latent neural ordinary differential equation. Motivated by the precessional-dissipative structure of the Landau-Lifshitz-Gilbert equation, the latent vector field is generated from the gradient of a learned scalar potential through an antisymmetric operator and a symmetric positive-semidefinite dissipative operator. This potential is learned in nonunique latent coordinates and is not identified with the Gibbs free energy, but decreases monotonically along autonomous continuous-time solutions, while the antisymmetric component permits motion along its level sets. The encoder, decoder, latent energy, and operators are trained jointly on short trajectory windows using latent and decoded-rollout losses alone, without time-derivative supervision, physical-energy labels, or dissipation penalties. At i
    
[^96]: Scale-QLoRA：面向原生4比特微缩放大语言模型的编码不变适配器合并

    Scale-QLoRA: Code-Invariant Adapter Merging for Native 4-bit Microscaling LLMs

    [https://arxiv.org/abs/2609.04526](https://arxiv.org/abs/2609.04526)

    Scale-QLoRA通过只训练原生4比特微缩放检查点的每块缩放因子、同时冻结全部E2M1编码来合并LoRA适配器，从而避免了朴素合并因重新量化编码平面而抹除适配效果（最高达39个百分点）的问题。

    

    将LoRA适配器合并进基座模型是标准的部署做法：它消除了运行时适配器每次前向传播的开销，并留下一个任何服务栈都能加载的单一独立检查点。但在原生4比特微缩放检查点（NVFP4、MXFP4）上，这一步不再是无代价的。合并后的权重必须通过量化器写回，而量化器会重新推导检查点的离散E2M1编码平面（约占该制品字节数的90%），因此部署制品便与某一种量化约定耦合，其生命周期中后续任何触及编码的事件都可能使其发生偏移。若按朴素方式执行，这一步不只是脆弱，甚至更糟：它会抹除适配效果，最高可达39个百分点，因为面对一个已落在编码网格上的基座，重建的最优解就是该基座本身。Scale-QLoRA则只适配原生的每块缩放因子字段，在部署网格上训练这些缩放因子，并冻结所有E2M1编码。在固定的原生格式、缩放网格、块布局……（摘要原文在此处截断）

    arXiv:2609.04526v1 Announce Type: new  Abstract: Merging a LoRA adapter into its base model is standard deployment practice: it removes the runtime adapter's per-forward overhead and leaves a single standalone checkpoint any serving stack can load. On a native 4-bit microscaling checkpoint (NVFP4, MXFP4) that step stops being free. The merged weights must be written back through a quantizer, which re-derives the checkpoint's discrete E2M1 code plane (roughly 90% of the artifact's bytes), so the deployed artifact becomes coupled to one quantization convention, and every later code-touching event in its lifecycle can move it. Done naively the step is worse than fragile: it deletes the adaptation, by up to 39 pp, because against an already-on-grid base the reconstruction optimum is that base. Scale-QLoRA instead adapts only the native per-block scale field, trains those scales on the deployment grid, and freezes every E2M1 code. Within a fixed native format, scale grid, block layout and c
    
[^97]: Hakken：预测未来发现以填补当今知识的空白

    Hakken: Predicting future discoveries to fill the gaps in today's knowledge

    [https://arxiv.org/abs/2609.04494](https://arxiv.org/abs/2609.04494)

    Hakken是一个领域无关的知识预测系统，它将基于transformer的时序知识图谱预测模型与LLM的语义知识相融合，预测科学概念间尚未发现的新关系，并通过解释框架帮助科学家评估这些预测，在生物医学领域的时间感知多标签关系预测上树立了新的基准。

    

    我们提出了Hakken，一个与领域无关的预测和解释系统，用于执行知识预测，即通过建立新颖的关系来增长科学知识，这些关系不受限于先前知识的演绎范围。Hakken使用基于transformer的预测模型，该模型建立在从海量研究出版物中提取的知识图谱的时间序列之上，并与大语言模型（LLM）的语义知识相融合，以预测科学概念之间尚未记录的关系的存在并定义其类型。随后，它调用一个与模型无关的解释框架，为每个预测提供伴随信息，使科学家能够评估所建议的新关系。虽然具有通用性，我们通过将其应用于生物医学领域来展示Hakken的实用能力。在该领域中，Hakken的预测模型为时间感知的多标签关系预测建立了新的基准，并且我们……

    arXiv:2609.04494v1 Announce Type: cross  Abstract: We present Hakken, a domain-agnostic prediction and explanation system performing knowledge prediction, i.e., growing scientific knowledge by establishing novel relationships, ones that are not limited to the deductive hull of previous knowledge. Hakken uses a transformer-based prediction model built on temporal sequences of knowledge graphs extracted from vast bodies of research publications, fused with an LLM's semantic knowledge, to predict the presence and define the type of as-yet undocumented relationships between scientific concepts. It then calls a model-agnostic explanation framework to provide accompanying information for each prediction that allows scientists to evaluate the suggested new relationship. While general purpose, we demonstrate Hakken's practical capabilities by applying it to the biomedical domain. There, Hakken's prediction model establishes a new benchmark for time-aware multi-label relation prediction, and we
    
[^98]: ResLearn-XR：面向扩展现实的网络流量与体验质量感知建模的残差学习

    ResLearn-XR: Residual Learning for Network Traffic and Quality-of-Experience-Aware Modeling in Extended Reality

    [https://arxiv.org/abs/2609.04493](https://arxiv.org/abs/2609.04493)

    ResLearn-XR是一个两阶段残差学习框架，通过在值空间进行流量预测、在logit空间进行QoE风险估计，并引入数据描述符算法（DDA）和新的XR Traffic-QoE数据集，实现了对突发性XR网络流量的准确预测和QoE风险的可靠估计。

    

    我们提出了ResLearn-XR，这是一个用于预测扩展现实（XR）网络流量和估计体验质量（QoE）风险的残差学习框架。ResLearn-XR采用两阶段时间学习结构，由一个基础序列预测模型构成，并通过任务特定的残差学习组件加以增强，以提高对突发性、非平稳XR流量动态的适应性。残差学习阶段在值空间中进行连续的XR流量预测，并在logit空间中进行概率性QoE风险估计。对于QoE风险分支，我们引入了数据描述符算法（DDA），这是一个因果特征构建模块，可将数据包级别的应用层观测数据转换为适用于加密流量分析的时间帧感知描述符。我们还构建了一个XR Traffic-QoE数据集，将连续的XR流量轨迹与会话级用户报告的QoE标签配对。ResLearn-XR降低了SMAPE……

    arXiv:2609.04493v1 Announce Type: new  Abstract: We present ResLearn-XR, a residual learning framework for predicting eXtended Reality (XR) network traffic and estimating Quality-of-Experience (QoE) risk. ResLearn-XR adopts a two-stage temporal learning structure comprising a base sequence prediction model augmented with task-specific residual learning components to improve adaptability to bursty, non-stationary XR traffic dynamics. The residual learning stages operate in the value space for continuous XR traffic forecasting and in the logit space for probabilistic QoE risk estimation. \rev{For the QoE-risk branch, we introduce a Data Descriptor Algorithm (DDA), a causal feature-construction module that converts packet-level application-layer observables into frame-timing-aware descriptors suitable for encrypted traffic analysis. We also construct an XR Traffic-QoE dataset that pairs continuous XR traffic traces with session-level user-reported QoE labels. ResLearn-XR reduces SMAPE by 
    
[^99]: 当量化破坏记忆时：低精度时序推理中的循环状态回写

    When Quantization Breaks Memory: Recurrent-State Write-Back in Low-Precision Temporal Inference

    [https://arxiv.org/abs/2609.04490](https://arxiv.org/abs/2609.04490)

    该论文揭示循环网络中量化状态的存储规则（循环状态回写）会显著影响后续推理——在荧光寿命成像任务中，用4位确定性状态存储替代连续状态传播会使两个寿命参数的估计误差分别增加约70倍和300倍。

    

    量化被广泛用于降低神经网络推理的计算与内存需求。然而，在循环网络中，量化后的状态会被存储并在下一个时间步返回，因此存储该状态所采用的规则会改变后续的计算。本文引入“循环状态回写”这一概念来指代这一规则，并在一个用于荧光寿命成像的紧凑GRU编码器-解码器中分离出其影响，荧光寿命成像是一种用于定量生物成像的分子成像模态。其核心任务是从高噪声的时间分辨荧光信号中估计两个寿命参数：短寿命分量 τ1 和长寿命分量 τ2。在保持已训练模型不变的情况下，用确定性的4位状态存储替代连续的状态传播，会使 τ1 和 τ2 的估计误差分别增大约70倍和300倍。当重复的微小更新始终低于回写……（原文摘要在此处截断）

    arXiv:2609.04490v1 Announce Type: new  Abstract: Quantization is widely used to reduce the computational and memory demands of neural-network inference. In recurrent networks, however, the quantized state is stored and returned at the next time step, so the rule used to store that state can alter subsequent computations. Here, we introduce recurrent-state write-back to denote this rule and isolate its effect in a compact GRU encoder--decoder for fluorescence lifetime imaging, a molecular imaging modality used in quantitative biological imaging. A central task is estimating two lifetime parameters, the short-lived component {\tau}1 and the long-lived component {\tau}2, from high-noise time-resolved fluorescence signals. Holding the trained model fixed, replacing continuous state propagation with deterministic 4-bit state storage increases estimation errors for {\tau}1 and {\tau}2 by approximately 70x and 300x, respectively. Failure occurs when repeated small updates remain below the wri
    
[^100]: 联邦遗忘中客户端对已删除岭回归统计信息的探测

    Client-Side Probing of Deleted Ridge Statistics in Federated Unlearning

    [https://arxiv.org/abs/2609.04475](https://arxiv.org/abs/2609.04475)

    该研究揭示了一种新的隐私威胁：在联邦遗忘系统中，恶意客户端可以利用服务器广播的线性分类器更新作为探测手段，精确恢复被删除的样本、类别或客户端摘要，并可能将被删除的数据重新插入。

    

    联邦遗忘旨在无需从头重新训练的情况下，将某个客户端的数据从共享模型中移除。一些高效的系统通过存储训练特征的紧凑可加摘要，并在每次接受更改后广播更新后的线性分类器，从而实现精确删除。我们证明这些广播也可能泄露隐藏的摘要信息。恶意客户端可以提交已知的更改，利用返回的分类器识别服务器状态，并比较某个孤立删除前后紧邻的状态。这会暴露被删除的样本、类别或客户端摘要，并可能使其被重新插入。我们精确刻画了观测何时包含足够的独立信息，针对无限制探测给出了匹配的最优构造，并基于攻击者自身数据构成的加法操作推导出一种更现实的估计器。在MNIST和CIFAR-10数据集上，高精度广播可实现对每个测试样本的精确标签恢复。

    arXiv:2609.04475v1 Announce Type: cross  Abstract: Federated unlearning aims to remove a client's data from a shared model without retraining from scratch. Some efficient systems make deletion exact by storing compact, additive summaries of the training features and broadcasting an updated linear classifier after every accepted change. We show that these broadcasts can also reveal the hidden summaries. A malicious client can submit known changes, use the returned classifiers to identify the server state, and compare states immediately before and after an isolated deletion. This exposes the deleted sample, class, or client summary and can enable its reinsertion. We characterize exactly when the observations contain enough independent information, give a matching optimal construction for unrestricted probes, and derive a more realistic estimator based on additions formed from the attacker's own data. On MNIST and CIFAR-10, high-precision broadcasts permit exact label recovery for every t
    
[^101]: SPD流形学习的嵌套归纳偏置框架

    Nested Inductive Bias Framework for SPD Manifold Learning

    [https://arxiv.org/abs/2609.04466](https://arxiv.org/abs/2609.04466)

    提出嵌套归纳偏置框架，通过两阶段微分同胚组合将庞加莱度量等非欧几里得目标几何回拉到SPD流形上，从而同时满足流形约束与关系先验两类归纳偏置。

    

    在几何深度学习中，归纳偏置主要承担两项功能：强制流形约束和嵌入关系先验。目前，SPD（对称正定）流形上的表示学习通常依赖于回拉欧几里得度量（如对数欧几里得度量）来满足前者。虽然这类度量在计算上高效并能避免域边界违规问题，但它们诱导出的是平坦几何，可能无法捕捉数据集内在的关系先验。尽管庞加莱度量等度量被广泛用于诱导域对齐的关系先验，但将它们从标准向量表示推广到SPD流形一直是一个难题。为弥合这一差距，我们提出了一个嵌套归纳偏置框架，该框架利用两阶段微分同胚组合，将非欧几里得目标几何正式回拉到SPD流形上。该框架能够构建曲率对齐的黎曼

    arXiv:2609.04466v1 Announce Type: new  Abstract: In Geometric Deep Learning, inductive biases serve two primary functions: enforcing manifold constraints and embedding relational priors. Currently, representation learning on SPD manifolds frequently relies on pullback Euclidean metrics, such as the Log-Euclidean Metric, to satisfy the former. While computationally efficient in avoiding domain boundary violations, these metrics induce a flat geometry that may fail to capture the intrinsic relational priors of datasets. While metrics such as the Poincar\'e metric are widely utilized to induce domain-aligned relational priors, generalizing them from standard vector representations to the SPD manifold has remained a challenge. To bridge this gap, we introduce a Nested Inductive Bias framework that utilizes a two-stage diffeomorphic composition to formally pull back non-Euclidean target geometries onto the SPD manifold. This framework enables the construction of curvature-aligned Riemannian
    
[^102]: 基于机载机器学习的成像光谱数据痕量气体检测

    On-board ML for Trace Gas detection in Imaging Spectroscopy data

    [https://arxiv.org/abs/2609.04458](https://arxiv.org/abs/2609.04458)

    该研究首次利用边缘机器学习技术实现了成像光谱数据中甲烷点源排放的机载实时检测，通过下传轻量模型预测的潜在事件而非完整数据立方体，克服了通信带宽瓶颈并大幅缩短了从观测到信息获取的时间。

    

    在航空和航天成像光谱观测活动中收集的数据能够检测诸如痕量气体排放等瞬态事件。然而，当前的处理流程依赖于缓慢的地面处理，这延迟了每个检测事件的信息获取时间，并阻碍了即时采取的后续行动。在2026年3月的东京野外观测活动中，我们探索了利用所搭载的AVIRIS-5传感器对成像光谱数据进行机载处理。由于通信瓶颈，完整的数据立方体无法在飞行期间立即下传。因此，我们改为下传由高效且轻量的机器学习模型所预测的潜在事件。我们展示了首个利用边缘机器学习技术、基于成像光谱数据的甲烷点源排放机载检测。

    arXiv:2609.04458v1 Announce Type: new  Abstract: Data collected during aerial and spaceborne imaging spectroscopy campaigns enables the detection of transient events such as trace gas emissions. However, current processing pipelines depend on slow, on-the-ground processing, which delays the time to information of each detected event and prohibits immediate follow-up actions. During the Tokyo Field Campaign of March 2026, we explored on-board processing of Imaging Spectroscopy data from the equipped AVIRIS-5 sensor. Due to communication bottlenecks, full datacubes cannot be downlinked immediately during the flight. Instead we downlink the potential events predicted by our efficient and small machine learning model. We show the first on-board detection of methane point source emission with Imaging Spectroscopy data using Edge ML.
    
[^103]: 当负载均衡走向极端：过度分散混合专家模型中的专家剪枝

    When Load-Balancing Goes Too Far: Expert Pruning in Over-Dispersed Mixture-of-Experts Models

    [https://arxiv.org/abs/2609.04453](https://arxiv.org/abs/2609.04453)

    该论文发现在因训练时负载均衡过于激进而导致路由过度分散的MoE模型中，路由器概率不再是可靠的专家重要性信号，困惑度也无法预测下游任务准确率，因此传统的基于路由的专家剪枝方法在此类模型中会失效，且不同评分指标之间存在能力权衡。

    

    专家剪枝通过移除由路由器识别出的低重要性专家，来降低混合专家模型（MoE）的内存与服务成本，其前提假设是路由器概率能够提供可靠的专家重要性信号。我们观察到，这一假设在“过度分散路由”（over-dispersed routing）的状态下会失效——这种状态与训练中过于激进的负载均衡相关，此时token几乎均匀地分布到各个专家上，重要性信号随之崩塌。在这种状态下，困惑度无法预测下游任务的准确率：在gpt-oss-20B上，困惑度最低的剪枝配置反而产生了最差的数学推理表现，而困惑度最高的配置却保留了数学推理能力。这一现象在标准路由（如Mixtral-8x7B-Instruct）下并不会出现，在标准路由下困惑度与准确率会同步退化。此外，在过度分散路由下进行剪枝还会暴露出一种能力上的权衡：没有任何单一的评分指标能够全面占优，其中激活感知评分……

    arXiv:2609.04453v1 Announce Type: cross  Abstract: Expert pruning reduces the memory and serving cost of Mixture-of-Experts (MoE) models by removing low-importance experts identified by the router, assuming router probabilities provide a reliable importance signal. We observe that this assumption breaks down under over-dispersed routing, a regime associated with aggressive load-balancing during training, in which tokens are distributed nearly uniformly across experts and importance signals collapse. In this regime, perplexity does not predict downstream task accuracy: on gpt-oss-20B, the lowest-perplexity pruning configuration yields the worst mathematical reasoning, while the highest-perplexity configuration preserves it. This does not occur under standard routing (e.g., Mixtral-8x7B-Instruct), where perplexity and accuracy degrade together. Pruning under over-dispersed routing also exposes a capability trade-off in which no single scoring metric dominates: activation-aware scoring pr
    
[^104]: 从众性破坏共形预测

    Conformity Breaks Conformal Prediction

    [https://arxiv.org/abs/2609.04445](https://arxiv.org/abs/2609.04445)

    论文揭示同伴压力会使多智能体LLM系统发生“评分机制偏移”，悄然破坏共形预测的覆盖率保证（从90%降至74%），且攻击者可通过针对低置信度子组将其覆盖率近乎减半（从87%降至47%）。

    

    当LLM单独回答时，共形证书可以是有效的，但当同一个LLM看到一致断言错误答案的同伴时，该证书便会失效。问题本身没有改变，改变的是模型对正确答案的评分。我们将这一现象称为“评分机制偏移”：在干净条件下校准的证书只能认证模型单独回答时的评分方式，却无法认证其在同伴压力下的评分方式。我们证明这种偏移会在多智能体LLM系统中悄然破坏共形预测。在多种开放权重模型和多选题问答任务上，在标准alpha = 0.10工作点下，当存在一致给出错误答案的同伴时，覆盖率会从校准后的90%降至74%。这一平均值掩盖了更严重的失败：攻击者通过针对证书仍能覆盖的低置信度题目，几乎将该子组的覆盖率减半——从87%降至47%——而受监控的整体平均值仍保持在较高水平。这种失败还会波及决策层：一个本应在不确定时上报的系统……（原文摘要在此处截断）

    arXiv:2609.04445v1 Announce Type: cross  Abstract: A conformal certificate can be valid when an LLM answers alone and invalid when the same LLM sees peers that unanimously assert a wrong answer. The question is unchanged; the model's score for the correct answer changes. We call this a score-mechanism shift: clean calibration certifies how the model scores answers alone, but not how it scores them under peer pressure. We show that this shift silently breaks conformal prediction in multi-agent LLM systems. Across open-weight models and multiple-choice QA tasks, coverage falls from a calibrated 90% to 74% under unanimous-wrong peers at the standard alpha = 0.10 operating point. The average hides a sharper failure: by targeting the low-confidence items the certificate still covers, an attacker nearly halves coverage on that subgroup, from 87% to 47%, while the monitored average remains much higher. The failure also reaches the decision layer: a system that should escalate when uncertain c
    
[^105]: 从粗粒化珠子中恢复分子：跨化学空间的自由能条件生成式反向映射

    Recovering molecules from coarse-grained beads: free-energy-conditioned generative backmapping across chemical space

    [https://arxiv.org/abs/2609.04432](https://arxiv.org/abs/2609.04432)

    本文提出juniper——一个以辛醇-水分配自由能为条件的分子图离散去噪扩散模型，首次将组成层面的粗粒化反向映射（从粗粒化珠子组合恢复对应分子化合物）表述为条件图生成问题，实现了跨化学空间的分子恢复。

    

    可迁移的粗粒化（CG）力场对化学空间进行了压缩：通过将原子聚合为数量更少的相互作用珠子，诸如MARTINI之类的模型将可区分的化合物数量减少了大约三个数量级，使得在软物质领域对热力学性质进行高通量筛选变得可行，其中药物-膜渗透性是一个发展成熟的例子。然而这种压缩是有损的，并且迄今为止是单向的：筛选返回的是珠子的组合，却缺乏一条已确立的路径回到其所代表的化合物。恢复这些化合物——即组成层面的反向映射——是一个一对多的逆映射问题，有别于研究更为充分的构象问题（即从已知映射重建原子坐标）。本文中，我们将组成层面的反向映射表述为条件图生成问题，并引入了juniper——一个在分子图上运行的离散去噪扩散模型，其以辛醇-水分配自由能为条件（摘要在此处不完整）。

    arXiv:2609.04432v1 Announce Type: cross  Abstract: Transferable coarse-grained (CG) force fields compress chemical space: by aggregating atoms into a reduced set of interaction beads, models such as MARTINI reduce the number of distinguishable compounds by roughly three orders of magnitude, making high-throughput screening of thermodynamic properties tractable across soft matter, with drug--membrane permeability as a well-developed example. The compression is lossy and, so far, one-way: a screen returns a combination of beads, with no established route back to the compounds it stands for. Recovering those compounds--compositional backmapping--is a one-to-many inverse map, distinct from the better-studied conformational problem of rebuilding atomic coordinates from a known mapping. Here we formulate compositional backmapping as conditional graph generation by introducing juniper, a discrete denoising diffusion model over molecular graphs conditioned on the octanol--water partition free 
    
[^106]: 使用五个大语言模型对英文歌词进行文化分析的重复测量研究

    A Repeated-Measurement Study for Cultural Analytics of English Song Lyrics Using Five Large Language Models

    [https://arxiv.org/abs/2609.04428](https://arxiv.org/abs/2609.04428)

    本研究通过对大型歌词语料库的重复标注实验，系统评估了五个大语言模型对英文歌词中自尊、自控、寻求归属感和寻求认可四种社会构念进行零样本标注的测量可靠性，发现LLM测量的可靠性因构念而异，其中自尊的重复测量可靠性最强。

    

    大语言模型（LLM）正被越来越多地用于以人工编码员无法实现的规模来标注文化文本。然而，在将其输出视为潜在社会构念的测量结果之前，有必要先确认这些测量是否可靠。本研究评估了五个大语言模型作为英文歌词中四种社会构念的零样本标注器的表现，这四种构念分别是：自尊、自控、寻求归属感以及寻求认可。通过对一个大型歌词语料库进行重复标注，我们考察了基于LLM的测量的三个特性：重复运行之间的一致性、不同模型之间的趋同性，以及共识标签向监督分类任务的可迁移性。研究结果表明，基于LLM的测量在不同构念之间的可靠性并不一致。自尊在所有模型中均表现出最强的重复测量可靠性，而寻求认可通常较不稳定；自控和自……（原文摘要在此处截断）

    arXiv:2609.04428v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used to annotate cultural texts at scales that are impractical for human coders. However, before their outputs are treated as measurements of latent social constructs, it is necessary to establish whether those measurements are reliable. This study evaluates five LLMs as zero-shot annotators of four social constructs expressed in English song lyrics: self-esteem, self-control, seeking belonging, and seeking recognition. Using repeated annotations of a large lyric corpus, we examine three properties of LLM-based measurement: consistency across repeated runs, convergence across models, and transferability of consensus labels to supervised classification. The findings show that LLM-based measurement is not uniformly reliable across constructs. Self-esteem exhibits the strongest repeated-measurement reliability across models, while seeking recognition is generally less stable; self-control and se
    
[^107]: 超越通用预测选择器：跨需求模式与预测期的需求条件化模型选择

    Beyond a Universal Forecasting Selector: Demand-Conditioned Model Selection across Demand Patterns and Horizons

    [https://arxiv.org/abs/2609.04425](https://arxiv.org/abs/2609.04425)

    该研究通过大规模对比实验证明不存在普适最优的预测模型选择器，选择器的优劣依赖于需求模式、数据可得性和预测期，因此应将选择器作为依赖情境的组件并根据具体需求条件加以选择。

    

    预测模型选择在异质性需求场景下仍然十分困难，因为最合适的决策规则可能随需求结构、数据可得性和预测期的变化而不同。本研究探讨了选择器本身是否应被视为预测过程中依赖具体情境的组成部分。研究比较了五种选择机制——RMSSE、ERA、OWA、CCG-AHSC和CCG-AHSCD——涵盖24个优化的预测模型、九个数据集、三种训练-测试划分以及1至12个周期的预测期。选择器性能通过全局相对准确度（GRA）、统计检验和最优可达模型参考进行事后评估。研究结果表明，没有任何一种选择器能够在所有条件下都占据主导地位。CCG-AHSC和CCG-AHSCD在平滑需求及若干波动性（Erratic）配置中更具竞争力，而OWA和ERA在间歇性（Intermittent）和块状（Lumpy）需求场景中表现更优。选择器的适用性还会随历史数据可得性和预测期的变化而改变。

    arXiv:2609.04425v1 Announce Type: new  Abstract: Forecasting-model selection remains difficult in heterogeneous demand because the most suitable decision rule may vary with demand structure, data availability, and forecasting horizon. This study examines whether the selector itself should be treated as a context-dependent component of the forecasting process. Five selection mechanisms - RMSSE, ERA, OWA, CCG-AHSC, and CCG-AHSCD - are compared across 24 optimized forecasting models, nine datasets, three training-testing partitions, and horizons from 1 to 12 cycles. Selector performance is evaluated ex post using Global Relative Accuracy (GRA), statistical tests, and a best-attainable-model reference. No selector dominates across all conditions. CCG-AHSC and CCG-AHSCD are more competitive for Smooth demand and several Erratic configurations, whereas OWA and ERA perform better in Intermittent and Lumpy settings. Selector suitability also changes with historical data availability and horizo
    
[^108]: 调整集合通信模式以缓解共享AI集群中的拥塞

    Tuning Collective Patterns to Alleviate Congestion in Shared AI Clusters

    [https://arxiv.org/abs/2609.04417](https://arxiv.org/abs/2609.04417)

    提出了REACT系统，它在应用层（通信库）利用易于获取的流统计信息在运行时检测拥塞，并动态调整GPU节点间的集合通信模式，从而无需全局控制或基础设施支持即可缓解共享AI集群中的拥塞问题。

    

    分布式AI训练涉及多对GPU节点之间反复进行的数据交换轮次。即使其中一条流因拥塞而变慢，也会导致整个通信轮次减速。当前用于规避AI集群拥塞的方法要么假设对整个工作负载具有全局控制（例如协调所有作业的调度），要么假设具备基础设施支持（例如交换机中的自适应路由）。因此，这些方法在共享云环境中并不适用——在这种环境中，属于某一用户的AI作业可能面临来自其他用户作业或后台流量的外部拥塞，而这超出了其自身的控制范围。在本文中，我们构建了一个名为REACT的系统，该系统能够响应拥塞来调整GPU节点之间反复出现的数据交换模式（即集合通信）。REACT在应用（通信库）层工作，利用易于获取的流统计数据在运行时检测拥塞，并据此调整集合通信模式。

    arXiv:2609.04417v1 Announce Type: cross  Abstract: Distributed AI training involves recurring rounds of data exchange between multiple pairs of GPU nodes. Slowdown in even one flow due to congestion can cause the entire communication round to slowdown. Current approaches for evading congestion in AI clusters assume global control over the entire workload (e.g. coordinating the schedule of all jobs) or assume infrastructural support (e.g. adaptive routing in switches). They are thus ill-suited in a shared cloud setting where AI jobs belonging to one user can face external congestion from other users' jobs or background traffic beyond its own control. In this paper, we build a system, REACT, that tunes the recurring pattern of data exchange between GPU nodes (known as communication collectives) in response to congestion. REACT works at the application (communication library) layer, where it detects congestion at runtime using readily available flow stats, and tunes the collective pattern
    
[^109]: REFINE：基于预算化文本属性图的大语言模型精炼方法，用于个性化医学概念表示

    REFINE: LLM Refinement over Budgeted Text-Attributed Graphs for Personalized Medical Concept Representation

    [https://arxiv.org/abs/2609.04415](https://arxiv.org/abs/2609.04415)

    提出REFINE框架，通过强化学习策略在预算约束下利用大语言模型对知识图谱进行图精炼，从文本属性知识图谱中学习患者个性化的医学概念表示，从而提升EHR预测性能。

    

    学习丰富的医学概念表示对于电子健康记录（EHR）预测至关重要。文本属性知识图谱（TKGs）通过将异构的医学关系与文本语义有机结合，为此提供了天然的基础。然而，大多数现有编码器对所有患者统一处理医学概念，尽管事实上某个编码的含义和预测价值取决于患者特定的临床情境和诊疗轨迹。从TKGs中学习患者个性化的概念表示面临两个关键挑战：（1）如何决定为每个观察到的编码纳入多少知识图谱上下文；（2）如何将语义信息与患者特定的关系结构对齐。我们提出了REFINE，一个面向患者个性化医学概念编码的知识图谱感知、预算化的大语言模型图精炼框架。从全局TKG出发，REFINE构建患者特定的时序图，并通过序贯强化学习策略选择个性化的KG扩展（摘要在此处截断）。

    arXiv:2609.04415v1 Announce Type: cross  Abstract: Learning rich medical concept representations is essential for EHR prediction. Text-attributed knowledge graphs (TKGs) provide a natural foundation by organizing heterogeneous medical relations together with textual semantics. However, most existing encoders process concepts uniformly across patients, despite the fact that a code's meaning and predictive value depend on patient-specific clinical context and trajectory. Learning patient-personalized concept representations from TKGs introduces two key challenges: (1) deciding how much KG context to incorporate for each observed code, and (2) aligning semantic information with the patient-specific relational structure. We propose REFINE, a KG-aware budgeted LLM graph refinement framework for patient-personalized medical concept encoding. Starting from a global TKG, REFINE constructs patient-specific temporal graphs. A sequential reinforcement learning policy selects a personalized KG exp
    
[^110]: 解耦深度算子学习中的注意力机制：数据驱动与物理信息架构的受控研究

    Disentangling Attention in Deep Operator Learning: A Controlled Study of Data-Driven and Physics-Informed Architectures

    [https://arxiv.org/abs/2609.04407](https://arxiv.org/abs/2609.04407)

    本文通过对五种注意力机制不同的DeepONet变体进行受控系统研究，在数据驱动与物理信息两种训练模式下训练，首次分离了交叉注意力、自注意力、标记化和注意力深度各自对偏微分方程求解精度的贡献。

    

    深度神经算子学习输入函数与完整偏微分方程（PDE）解场之间的映射，使得对新问题实例的前向评估比传统数值求解器快数个数量级。注意力机制最近被引入神经算子中，但大多数研究同时改变了多个架构组件，导致难以识别究竟是什么因素真正提升了精度。本工作对五种具有不同注意力机制的深度算子网络（DeepONet）变体进行了受控且系统的研究，并在数据驱动和物理信息两种训练模式下进行训练，以分离交叉注意力、自注意力、标记化和注意力深度各自的影响。我们在源驱动的瞬态一维非线性扩散-反应方程、具有可变初始条件的瞬态一维粘性Burgers方程以及二维泊松热传导问题上对它们进行了评估。

    arXiv:2609.04407v1 Announce Type: new  Abstract: Deep neural operators learn mappings between input functions and complete PDE solution fields, enabling forward evaluations of new problem instances orders of magnitude faster than conventional numerical solvers. Attention mechanisms have recently been introduced into neural operators, but most studies change several architectural components at once, making it difficult to identify what actually improves accuracy. This work presents a controlled and systematic study of five deep operator network (DeepONet) variants with distinct attention mechanisms, trained under both data-driven and physics-informed regimes, to isolate the effects of cross-attention, self-attention, tokenization, and attention depth. We evaluate them on a source-driven transient one-dimensional nonlinear diffusion-reaction equation, a transient one-dimensional viscous Burgers equation with variable initial conditions, and a two-dimensional Poisson heat-conduction probl
    
[^111]: 晋升之前先验证候选模型可比性：自适应网络入侵检测中的条件化验证

    Candidate Comparability Before Promotion: Conditional Validation in Adaptive Network Intrusion Detection

    [https://arxiv.org/abs/2609.04388](https://arxiv.org/abs/2609.04388)

    本文针对自适应网络入侵检测提出“晋升前的条件化验证”方法，证明晋升结论依赖于挑战者的构建方式与证据规模——表观的晋升危害主要是由现任模型的冻结预处理放大所致，在自包含流水线与严格控制证据量的条件下该危害并不成立。

    

    自适应网络入侵检测系统在漂移警报触发后会重新训练分类器，但警报只能检测到“发生了变化”；它并不能证明挑战者模型应当取代已部署的现任模型。模型晋升事关安全，因为它改变了负责后续攻击检测的模型，而对其评估存在一个方法论问题：晋升结论可能取决于挑战者模型是如何构建的，以及支持它的证据有多少。我们在CICIDS2017、UNSW-NB15和ToN-IoT数据集上检验了这种依赖性，实验设计包括：自包含的挑战者流水线、嵌套的候选证据规模控制、在统一测试框架下对九种更新策略的比较，以及最终的敏感性分析——将每个精确特征向量严格限定为仅承担评估、训练或探测三种角色中的一种。结果表明：由现任模型拥有的冻结预处理放大了表观上的晋升危害；而采用自包含的挑战者流水线后，全漂移情形下的平均危害并未持续存在。将名义候选证据规模从512提高到……（摘要原文在此处截断）

    arXiv:2609.04388v1 Announce Type: cross  Abstract: Adaptive network intrusion detection systems retrain classifiers after drift alarms, but an alarm detects change; it does not establish that a challenger should replace the deployed incumbent. Promotion is security-relevant because it changes the model responsible for subsequent attack detection, and evaluating it has a methodological problem: promotion conclusions may depend on how the challenger was constructed and on how much evidence supports it. We test that dependence on CICIDS2017, UNSW-NB15 and ToN-IoT with self-contained challenger pipelines, nested candidate-size controls, a common-harness comparison of nine update policies, and a final sensitivity confining every exact feature vector to one evaluation, training or probe role. Incumbent-owned frozen preprocessing amplified apparent promotion harm; with self-contained challenger pipelines the mean full-drift harm did not persist. Raising nominal candidate evidence from 512 to 
    
[^112]: 拆分式LLM训练中的隐私失效：返回的梯度使诱饵彻底失效

    Privacy Failure in Split-LLM Training, The Returned Gradient Nullifies the Decoys

    [https://arxiv.org/abs/2609.04382](https://arxiv.org/abs/2609.04382)

    论文揭示了拆分式LLM训练中一个被忽视的隐私漏洞：由于损失函数忽略诱饵行，返回梯度中诱饵行的值恰好为零，其零值模式可完美暴露真实数据行，在九个随机种子下每轮运行4,096帧全部被准确识别。

    

    我们对一套双节点拆分式LLM（大语言模型）训练系统开展了系统安全案例研究：该系统的隐私评估虽已通过，却遗留了一条未经测试的可观测信道。可信本地节点向不可信云节点（UCN）发送受保护的激活值，UCN返回其输出，而持有私有损失的TLN则返回输出梯度。UCN接收到的数据帧中混合了真实行与诱饵行，而损失函数会忽略诱饵行，因此诱饵行的梯度恰好为零，这种零值模式便暴露了哪些行是真实数据。我们采用预先固定的协议对这一泄露进行测量：注入已知强度的泄露以证明测量工具确实能够检测到泄露；设置打乱标签的对照实验以证明工具不会误报不存在的泄露；并在所有实验运行之前预先设定判定阈值。在九个随机种子下，零值模式在每一帧上都正确识别出了真实行，每轮运行4,096帧全部命中（4,096/4,096）。针对数据帧内容发起的攻击相比常数猜测基线，每百个词符约可多恢复一个词符（+0.6……

    arXiv:2609.04382v1 Announce Type: cross  Abstract: We present a systems-security case study of a two-node split-LLM training system whose privacy evaluation passed while leaving an observable channel untested. The Trusted Local Node (TLN) sends protected activations to the Untrusted Cloud Node (UCN), the UCN returns its output, and TLN, holding the private loss, returns the output gradient. The frame the UCN receives mixes real rows with decoys, and the loss ignores the decoys. Their gradients are exactly zero, so the pattern of zeros reveals which rows were real. We measure it with a protocol fixed in advance: a leak injected at known strength to prove the instrument can see one, a shuffled-label control to prove it does not report absent leaks, and a threshold set before the runs. Across nine seeds, the zeros identified the real rows on every frame, 4,096 of 4,096 per run. An attack on the frame contents recovered about one extra token per hundred over a constant-guess baseline (+0.6
    
[^113]: 论 t-SNE 能量函数临界点的丰富性

    On the Abundance of Critical Points of the t-SNE Energy

    [https://arxiv.org/abs/2609.04379](https://arxiv.org/abs/2609.04379)

    本文在特征空间密度服从连续对称性的条件下，通过识别成对的离散对称性，为包括 t-SNE 及其大数据极限在内的一般能量族构造出无穷多族不同的临界点，从而为解释 t-SNE 能量景观中存在大量不遵循数据拓扑与聚类结构的局部极小值这一现象提供了严格的理论起点。

    

    本文研究了 t-SNE 算法的能量景观。尽管该算法已被广泛采用，但其相关能量的非凸性使得在许多情形下难以严格理解该算法究竟捕捉到了数据的什么特征。特别地，许多著名的数值例子（其中若干个在本文中得到了重现）表明，其能量景观十分复杂，存在大量不遵循底层数据拓扑结构或聚类结构的局部极小值。本工作旨在为严格解释这些现象迈出第一步。具体而言，对于一个包含原始 t-SNE 算法和最近发现的大数据极限在内的一般能量族，以及对于在特征空间中服从连续对称性的密度分布，我们构造了无穷多个互不相同的临界点族。这些临界点的构造基于识别成对的离散对称性，其中一个位于……

    arXiv:2609.04379v1 Announce Type: new  Abstract: This paper considers the energy landscape of the t-SNE algorithm. While this algorithm has enjoyed broad adoption, the non-convexity of the associated energy has made it difficult to rigorously understand what the algorithm captures in many settings. In particular, a number of well-known numerical examples, several of which are reproduced in this article, suggest a complicated energy landscape with many local minimizers that do not respect the topology or clustering structure of the underlying data. This work seeks to provide first steps towards a rigorous explanation of these phenomena. Specifically, for a general family of energies, which include both the original t-SNE algorithm and recently identified large data limits, and for densities in feature space which obey a continuous symmetry, we construct infinite families of distinct critical points. These critical points are based upon identifying pairs of discrete symmetries, one in th
    
[^114]: VLA-Precision：面向视觉-语言-动作模型高效真实世界在线强化的非对称协同自举方法

    VLA-Precision: Asymmetric Co-Bootstrapping for Efficient Real-World Online RL of Vision-Language-Action Models

    [https://arxiv.org/abs/2609.04355](https://arxiv.org/abs/2609.04355)

    提出VLA-Precision框架，通过非对称协同自举算法和ACoB-Stream架构，同时解决VLA模型真实世界在线强化学习中价值信号不可靠与计算开销过大两大瓶颈，实现高效且精确的策略改进。

    

    预训练的视觉-语言-动作（VLA）模型能够实现广泛的操作任务，但在需要精度和可重复性的任务中仍然不够可靠。将真实世界在线强化学习应用于VLA后训练，可以在示范数据之外实现自主的试错式改进，但这也暴露出两个瓶颈：1）不可靠的价值信号可能导致策略漂移；2）大型VLA的开销限制了吞吐量和样本效率。为应对这些挑战，我们提出了VLA-Precision，一个高效的真实世界在线强化学习框架，其核心是非对称协同自举算法和ACoB-Stream架构。具体而言，ACoB在不同时间尺度上建立非对称协同自举机制：早期由干预引导的行为学习快速提升策略性能，同时提高在线经验的质量。随着自主经验的不断积累，全局回报传播与局部偏好排序逐步校准……（摘要截断）

    arXiv:2609.04355v1 Announce Type: cross  Abstract: Pretrained vision-language-action (VLA) models enable broad manipulation but remain unreliable in tasks demanding precision and repeatability. Applying real-world online reinforcement learning (RL) to VLA post-training enables autonomous trial-and-error improvement beyond demonstrations alone, but exposes two bottlenecks: 1) unreliable value signals can induce policy drift; 2) large-VLA overhead constrains throughput and sample efficiency. To address these challenges, we present VLA-Precision, an efficient real-world online RL framework featuring the Asymmetric Co-Bootstrapping (ACoB) algorithm and the ACoB-Stream architecture. Specifically, ACoB establishes asymmetric co-bootstrapping across timescales: early intervention-guided behavioral learning rapidly improves policy performance while enhancing online experience quality. As autonomous experience accumulates, global return propagation and local preference ranking progressively cal
    
[^115]: 原型循环单元的量子变分方法

    A Quantum Variational Approach to Prototypical Recurrent Unit

    [https://arxiv.org/abs/2609.04354](https://arxiv.org/abs/2609.04354)

    提出了一种参数量显著少于经典（LSTM、GRU）和量子（QLSTM、QGRU）循环架构的轻量级量子原型循环单元 QPRU，在保持与最先进基线相当的预测性能的同时，具备更强的可扩展性。

    

    我们提出了一种轻量级的量子原型循环单元（QPRU），与经典循环架构（如长短期记忆网络 LSTM 和门控循环单元 GRU）以及量子变体（包括量子 LSTM 和量子 GRU）相比，其所需参数显著更少。尽管设计紧凑，QPRU 仍实现了具有竞争力的预测性能，可与最先进的基线相媲美，同时具备重要的结构和实用优势，包括更强的可扩展性和更少的可训练参数数量。

    arXiv:2609.04354v1 Announce Type: new  Abstract: We introduce a lightweight Quantum Prototypical Recurrent Unit (QPRU) that requires significantly fewer parameters than both classical recurrent architectures, such as Long Short- Term Memory (LSTM) and Gated Recurrent Unit (GRU), and quantum variants, including Quantum LSTM (QLSTM) and Quantum GRU (QGRU). Despite its compact design, the QPRU achieves competitive forecasting performance, matching state-of-the-art baselines while offering important structural and practical advantages, including enhanced scalability and a reduced number of trainable parameters.
    
[^116]: 面向物流网络中合成起讫点需求的约束感知生成框架

    A Constraint-Aware Generative Framework for Synthetic Origin-Destination Demand in Logistics Networks

    [https://arxiv.org/abs/2609.04345](https://arxiv.org/abs/2609.04345)

    提出了一种约束感知的条件生成框架，通过可微约束将运营指导直接融入生成目标，为分层物流网络生成拓扑真实且运营可行的合成起讫点需求数据。

    

    大规模物流网络需要合成数据生成能力，以支持在新型条件（如网络重构和需求冲击）下基于场景的规划。现有方法主要依赖历史观测数据，缺乏在遵守运营约束的同时生成能够适应网络拓扑变化的需求模式的能力。我们提出了一种约束感知的条件生成框架，用于分层物流网络中合成起讫点（OD）需求的生成。该框架将需求建模为给定每个起点时目的地的条件分布，实现了拓扑感知的合成，既在拓扑上真实又在运营上可行。运营指导通过可微约束直接融入生成目标，同时灵活的条件机制支持多种运营场景，并能适应不断演化的网络。

    arXiv:2609.04345v1 Announce Type: cross  Abstract: Large-scale logistics networks require synthetic data generation capabilities to support scenario-based planning under novel conditions-such as network reconfiguration and demand shocks. Existing approaches, which rely primarily on historical observations, lack the ability to generate demand patterns that adapt to changes in network topology while respecting operational constraints. We propose a constraint-aware conditional generative framework for synthetic origin-destination demand generation in hierarchical logistics networks. The framework models demand as a conditional distribution over destinations given each origin, enabling topology-aware synthesis that is both topologically realistic and operationally feasible. Operational guidance is incorporated directly into the generative objective via differentiable constraints, while a flexible conditioning mechanism supports various operational contexts and adaptation to evolving networ
    
[^117]: SharedSAE：跨语言模型的单一特征字典

    SharedSAE: One Feature Dictionary Across Language Models

    [https://arxiv.org/abs/2609.04344](https://arxiv.org/abs/2609.04344)

    SharedSAE通过共享字典结合各模型专属的编码器-解码器对，用单一稀疏自编码器即可替代多个语言模型的专用SAE，在保留96.6%解释方差的同时实现跨模型的统一特征解释与迁移。

    

    稀疏自编码器（SAE）被广泛用于解释语言模型的激活，但SAE的训练和潜变量标注通常需要对每个模型重复进行。在这项工作中，我们证明了一个单一的共享SAE可以取代一组针对各模型的专用SAE。我们的方法SharedSAE将共享字典与模型专属的编码器-解码器对相结合。与最接近的先前方法不同——该方法会丢弃激活幅度并要求在推理时使用所有模型——SharedSAE仅对选择分数进行归一化以保留激活幅度，并通过模型丢弃策略实现单模型推理。我们在四个跨越不同模型家族和分词器的10亿参数规模基础语言模型上训练了SharedSAE。尽管其潜变量在模型间共享，SharedSAE仍保留了专用SAE平均解释方差的96.6%；其潜变量激活所展现的跨模型相关性是事后对齐的独立SAE的1.8倍，且其潜变量描述能够在模型间迁移。

    arXiv:2609.04344v1 Announce Type: cross  Abstract: Sparse autoencoders (SAEs) are widely used to interpret language model activations, but SAE training and latent labelling are typically repeated for every model. Here, we show that a single shared SAE can replace a collection of dedicated per-model SAEs. Our method, SharedSAE, combines a shared dictionary with model-specific encoder-decoder pairs. Unlike the closest prior method, which discards activation magnitudes and requires all models at inference, SharedSAE instead normalizes only selection scores, preserving magnitudes, and uses model dropout for single-model inference. We train SharedSAE on four 1B-scale base language models spanning distinct families and tokenizers. Despite sharing its latents across models, SharedSAE retains 96.6% of dedicated SAEs' mean explained variance; its latent activations exhibit cross-model correlations 1.8 times as high as separate SAEs aligned post-hoc, and its latent descriptions transfer across m
    
[^118]: 模块化深度循环神经网络：在四旋翼无人机上的应用

    Modular Deep Recurrent Neural Network: Application to Quadrotors

    [https://arxiv.org/abs/2609.04339](https://arxiv.org/abs/2609.04339)

    本文提出一种带层间前馈连接的模块化深度循环神经网络，显著提升了学习和建模高阶动态与非线性特性的能力，并缓解了梯度消失/爆炸问题，在四旋翼无人机高度动态建模中验证了其相比现有方法的优越性。

    

    本文介绍了一种模块化深度循环神经网络（RNN），旨在简化各种RNN架构的部署过程，并为基于梯度的学习方法自动计算导数。模块化设计带来了一系列新架构，其中一种包含层间前馈连接。通过在多层RNN中添加层间前馈连接，观察到RNN学习和建模高阶动态与非线性特性的能力得到显著提升。层间前馈连接还缓解了多层RNN中空间上的梯度消失/爆炸问题。这些结果通过四旋翼无人机的案例研究得到验证：利用本文提出的特定网络结构学习到了高度动态模型，而现有方法无法如此快速地泛化，甚至完全无法泛化。

    arXiv:2609.04339v1 Announce Type: new  Abstract: A modular deep Recurrent Neural Network (RNN) is introduced to facilitate the process of deploying various architectures of RNNs, and to automatically compute derivatives for gradient-based learning methods. The modularity leads to a set of new architectures, one of which includes feedforward inter-layer connections. By adding feedforward inter-layer connections in a multi-layer RNN, it is observed that the capability of the RNN to learn and model high-order dynamics and nonlinearities is significantly improved. The problem of vanishing/exploding gradient in space for a multilayer RNN is also alleviated using feedforward connections. These results are demonstrated using a quadrotor case study, for which a model of the altitude dynamics is learned with our particular network structure, while existing methods are unable to generalize as quickly or at all.
    
[^119]: 基于泛函分析的数据驱动未知非线性微分方程学习

    Data-Driven Learning of Unknown Nonlinear Differential Equations Using Functional Analysis

    [https://arxiv.org/abs/2609.04329](https://arxiv.org/abs/2609.04329)

    本文提出一种基于泛函分析与算子理论的可解释机器学习方法，通过在函数空间中以积分形式构造代价函数，仅利用单条状态轨迹数据即可在线增量地学习非线性常微分方程的未知向量场。

    

    本文重新阐述了数据驱动发现非线性常微分方程（ODE）的问题，并提出了一种新的可解释机器学习方法。所提出的方法旨在仅从单条状态轨迹的数据中学习非线性动力学的未知向量场，而无需关于系统物理的先验知识。该方法与现有方法有两个根本区别：1）本方法的公式化推导基于泛函分析与算子理论；2）代价函数是在函数空间中构造的，表示为两个函数之间以积分形式定义的距离，而非现有机器学习方法中所使用的误差离散求和。本文提出了一种增量学习算法，以在线方式处理新的训练样本来学习未知向量场。该方法可以从受迫和非受迫自治系统中发现未知向量场。

    arXiv:2609.04329v1 Announce Type: cross  Abstract: In this paper, the problem of data-driven discovery of nonlinear ordinary differential equations (ODEs) is recast, and a new interpretable machine learning (ML) method is proposed. The proposed method aims to learn the unknown vector field of nonlinear dynamics without prior knowledge of the system's physics from only one single state trajectory's data. The proposed method has two fundamental differences with existing methods: 1) the formulation presented in this method is derived based on Functional Analysis and Operator Theory, and 2) the cost function is constructed in the function space as a distance between two functions as an integral, instead of the discrete-sum of errors used in existing ML approaches. An incremental learning algorithm is proposed to learn the unknown vector field to handle new training samples in an online manner. The proposed method can discover the unknown vector field from both forced and unforced autonomou
    
[^120]: 显微镜即掩码：来自冷冻电子断层扫描前向模型的特权视图与标签

    The microscope is the mask: privileged views and labels from a cryo-ET forward model

    [https://arxiv.org/abs/2609.04325](https://arxiv.org/abs/2609.04325)

    该论文提出CARNIVAL模型，巧妙利用冷冻电子断层扫描前向模型模拟产生的损坏配对视图作为自监督学习信号，并将模拟中蛋白质的位置和身份等特权信息融入模型架构与损失函数，无需微调即可在真实断层图上完成蛋白质分类与检测。

    

    我们探索使用模拟数据来训练一个模型，用于对在有限倾斜角度下采集图像重建、且受测量算子严重损坏的拥挤冷冻电子断层扫描体数据中的蛋白质进行标注。首先，我们利用前向模型施加的损坏来生成完全相同场景的特定领域增强配对视图，用于实现集成到LeJEPA自监督训练框架中的不变性目标。其次，我们利用来自模拟流程的额外信息（如模拟体数据中蛋白质的位置和身份）来指导模型的架构设计和损失函数，使得语义信息在生成的密集特征体中定位于蛋白质所在位置。所得到的模型CARNIVAL无需微调即可在真实断层图的分类和检测任务上进行评估，评估所用的基准数据集包含多种蛋白质类型。

    arXiv:2609.04325v1 Announce Type: cross  Abstract: We explore the use of simulated data for training a model for protein annotation in crowded cryo-electron tomography volumes reconstructed from images collected at limited tilt angles and severely corrupted by the measurement operator. Firstly, we leverage the corruptions imposed by the forward model to generate domain-specific augmented paired views of the exact same scene for an invariance objective integrated into the LeJEPA self-supervised training framework. Secondly, we use additional information from the simulation pipeline such as the positions and identity of proteins in the simulated volumes to inform the architecture of the model and the loss function, so that semantic information is localised at protein positions in the resulting dense feature volume. The resulting model, CARNIVAL, is evaluated without finetuning on classification and detection tasks in real tomograms, using a benchmark dataset containing multiple protein t
    
[^121]: TNFlow：海王星外天体表面成分的摊销后验推断

    TNFlow: Amortized Posterior Inference for Trans-Neptunian Object Surface Composition

    [https://arxiv.org/abs/2609.04305](https://arxiv.org/abs/2609.04305)

    提出TNFlow架构，将Transformer与归一化流结合，能在单个CPU上约0.7秒内从反射光谱快速反演海王星外天体表面成分与颗粒尺寸的多模态后验分布，在合成光谱上表现出良好精度与泛化能力，但在真实JWST光谱上对某些材料存在盲区或偏差。

    

    我们提出了TNFlow，一种结合Transformer与归一化流的架构，用于从反射光谱推断海王星外天体（TNO）的表面成分。TNFlow在由Shkuratov辐射传输模型生成的合成光谱上进行训练，以充当该模型的逆运算器。在单个CPU核心上，TNFlow反演一个光谱仅需约0.7秒，并返回关于单纯形有效成分与颗粒尺寸的多模态后验分布。在合成光谱上，权重最高的模式在测试集上与真值的平均总变差距离为0.149，且模型对未见过的已知成分组合具有良好的泛化能力。在真实JWST光谱上的定性测试显示，模型对某些材料存在盲区或偏差。我们认为这可能归因于模拟器的保真度或训练集本身。

    arXiv:2609.04305v1 Announce Type: cross  Abstract: We present TNFlow, a transformer and normalizing flow architecture for inferring the surface composition of Trans-Neptunian Objects (TNOs) from their reflectance spectra. TNFlow is trained on synthetic spectra generated by the Shkuratov radiative transfer model to act as its inverse. TNFlow takes ${\sim}$0.7s to invert one spectrum on a single CPU core, returning a multimodal posterior over simplex-valid compositions and grain sizes. On synthetic spectra, the highest-weight mode achieves a mean total-variation distance of 0.149 from ground truth on the test split, and the model generalizes well to unseen combinations of known components. Qualitative tests on real JWST spectra show blindness or bias towards some materials. We suggest this could be attributed to either simulator fidelity or the training set.
    
[^122]: 数据优化的预想事故筛选：一种面向电力系统安全的机器学习方法

    Data-Optimized Contingency Screening: A Machine Learning Approach to Power System Security

    [https://arxiv.org/abs/2609.04300](https://arxiv.org/abs/2609.04300)

    本研究提出一种基于机器学习的电力系统预想事故安全等级分类方法，利用SMOTE和PCA对牛顿-拉夫逊潮流计算提取的数据进行优化，并训练KNN、随机森林和支持向量机模型，将事故划分为安全、中等和严重三类，以支持主动决策并防范大规模故障。

    

    确保电力系统的安全对于系统的稳定性与可靠性至关重要，尤其是在发生扰动的情况下。对电力系统中的预想事故进行有效分类，能够支持主动决策，并减轻大规模停电与故障的发生。本研究探讨了利用机器学习算法将电力系统预想事故的安全等级划分为安全、中等和严重三类的方法。在该方法中，采用牛顿-拉夫逊潮流法从预想事故场景中提取系统数据，并以综合性能指标作为安全性度量。在数据预处理方面，采用合成少数类过采样技术（SMOTE）和主成分分析（PCA）分别解决类别不平衡问题和进行降维处理。在通过N-k预想事故场景生成的数据集上，对K近邻（KNN）、随机森林（RF）和支持向量机（SVM）模型进行了训练与评估。

    arXiv:2609.04300v1 Announce Type: new  Abstract: Ensuring the security of the power system is essential for stability and reliability, especially in the event of disruption. Effective classification of contingency in power systems enables proactive decision-making and mitigates large-scale breakdowns and failures. This study explores the use of machine learning algorithms to classify security levels of contingencies in power systems into safe, moderate or severe classes. For this approach, Newton-Raphson load flow method extracts system data from contingency scenarios, using Overall Performance Index (OPI) as safety measure. For data pre-processing, Synthetic Minority Over-Sampling Technique (SMOTE) and Principal Component Analysis (PCA) is used to address class imbalance and reduce dimensionality, respectively. K-Nearest Neighbours (KNN), Random Forest (RF) and Support Vector Machines (SVM) is trained and evaluated on datasets generated through N-k contingency scenarios for k equal 1,
    
[^123]: BER-PEF：基于贝叶斯错误率估计的统一人类移动可预测性评估

    BER-PEF: Unified Human Mobility Predictability Evaluation via Bayes Error Rate Estimation

    [https://arxiv.org/abs/2609.04292](https://arxiv.org/abs/2609.04292)

    提出BER-PEF框架，将贝叶斯错误率估计转化为人类移动可预测性估计，并在无可观测真实值的情况下为比较不同估计器提供了统一的评估协议。

    

    人类移动可预测性关注的是在给定目标和输入信息条件下可达到的最佳预测性能，但其真实值在实际移动数据上无法直接观测。我们提出了BER-PEF，一个基于贝叶斯错误率的框架，它将贝叶斯错误率估计转化为移动可预测性估计，并提供了一个统一的协议，用于在无可观测真实值的情况下比较各种估计器。该框架将符号序列、数值轨迹、上下文特征和学习到的表示映射到一个共同的特征-标签空间，然后通过测量估计器输出在共享可预测性参考区间下方、上方以及整个区间上的偏差，在受控扰动曲线上对估计器输出进行评估。在Foursquare NYC和TKY、GeoLife以及T-Drive数据集上的实验表明，若干基于贝叶斯错误率的估计器在符号序列上比现有可预测性方法取得了更低的参考偏差。

    arXiv:2609.04292v1 Announce Type: new  Abstract: Human mobility predictability concerns the best prediction performance attainable from a given target and input information, but its ground truth is not directly observable on real mobility data. We present BER-PEF, a Bayes-error-rate-based framework that converts BER estimation into mobility predictability estimation and provides a unified protocol for comparing estimators without observable ground truth. The framework maps symbolic sequences, numeric trajectories, contextual features, and learned representations into a common feature--label space, then evaluates estimator outputs along controlled perturbation curves against a shared predictability reference interval by measuring deviations below the interval, above the interval, and across the full interval. Experiments on Foursquare NYC and TKY, GeoLife, and T-Drive show that several BER-based estimators achieve lower reference discrepancy than existing predictability methods on symbo
    
[^124]: 大语言模型中的证据整合

    Evidence Integration in Large Language Models

    [https://arxiv.org/abs/2609.04290](https://arxiv.org/abs/2609.04290)

    本文提出了一个关于大语言模型证据整合机制的分布理论，并通过千万级规模实验验证了三个预测：接收者眼中更可能的候选答案更具说服力、模型更易整合自身特有错误而非外来错误、以及相同证据可能改善弱模型却损害强模型。

    

    尽管人们越来越依赖借助工具、检索增强生成、其他智能体和用户提供的外部证据进行推理的大语言模型（LLMs），但LLMs如何将这些证据整合到它们已经初步形成的决策中，这在很大程度上仍不清楚。我们提出了一个分布理论，在该理论中，证据通过接收者先验权重和候选证据倾斜度来改变接收者初始答案的分布，并由此得出三个预测。第一，对接收者而言概率更高的候选答案更具说服力。第二，接收者更容易整合自身特有的错误，而不是来自不同来源的外来错误。第三，相同的证据可以改善较弱的模型，却会损害较强的模型。我们通过超过一千万次试验、来自四个系列的十二个LLMs以及八个领域（其中四个是物理与生命科学领域的科学发现任务：量子力学、物理学、遗传学和分子生物学）验证了这些预测。

    arXiv:2609.04290v1 Announce Type: cross  Abstract: Despite increasing reliance on LLMs that reason with external evidence supplied by tools, retrieval-augmented generation, other agents, and users, how LLMs integrate such evidence into decisions they have already begun to form remains largely unclear. We present a distributional theory in which evidence shifts the receiver's distribution of initial answers, driven by a receiver prior weight and a candidate evidence tilt, leading to three predictions. First, candidates more probable to the receiver are more persuasive. Second, receivers more readily integrate characteristic errors of their own than foreign errors from different sources. Third, identical evidence can improve weaker models and harm stronger ones. We confirm these over ten million trials, twelve LLMs from four families, and eight domains, four of them scientific discovery tasks in the physical and life sciences: quantum mechanics, physics, genetics, and molecular biology. 
    
[^125]: 当“看见”压倒“知晓”：视觉语言模型中的视觉主导性与基于延后答复的个性化安全方法

    When Seeing Overrides Knowing: Visual Dominance and Deferral-Based Method for Personalized Safety in VLMs

    [https://arxiv.org/abs/2609.04281](https://arxiv.org/abs/2609.04281)

    该研究提出个性化安全基准MPS-Bench，发现主流视觉语言模型因“视觉主导性”机制——即视觉信息在多模态融合中过早抑制文本风险信号——而几乎从不询问缺失的用户情境，并据此提出基于延后答复的方法以提升模型的个性化安全性。

    

    视觉语言模型（VLM）正日益被部署于高风险场景中，在此类场景下，一个总体上合理的回答对特定用户而言仍可能是不安全的，因为该用户在医疗、情绪或处境方面的背景信息对模型而言是未知的。我们研究了多模态系统中的个性化安全问题，并提出了MPS-Bench基准：该基准基于584张真实图像，涵盖12个高风险领域，共构建了5,181个场景，每个场景均配有一个隐藏的用户画像。对八个前沿VLM的评估显示，它们几乎总是直接作答（86%–99%）而非主动询问缺失的上下文，且在个性化安全方面没有任何模型的得分超过2.6/5。为理解这些失败产生的原因，我们分析了多模态交互过程，并识别出“视觉主导性”（visual dominance）现象：在多模态融合过程中，视觉信息过早进入文本表示，从而抑制了文本中的风险信号。因果干预实验揭示了一个两阶段机制，其中视觉情感信息首先被转移……

    arXiv:2609.04281v1 Announce Type: cross  Abstract: Vision-language models (VLMs) are increasingly deployed in high-stakes settings, where a response that is reasonable in general may still be unsafe for a particular user whose medical, emotional, or situational context is unknown to the model. We study this problem of personalized safety in multimodal systems and introduce MPS-Bench, a benchmark of 5,181 scenarios from 584 real-world images across 12 high-risk domains, each paired with a hidden user profile. Evaluating eight frontier VLMs, we find that they almost always respond directly (86-99%) rather than seek missing context, and none exceeds 2.6/5 on personalized safety. To understand why these failures arise, we analyze multimodal interactions and identify visual dominance: visual information enters text representations early and suppresses textual risk signals during multimodal fusion. Causal interventions reveal a two-stage mechanism in which visual affect is first transferred 
    
[^126]: 评估大语言模型在强迫停电风险预测中的应用：优势及与机器学习的比较

    Evaluating Large Language Models for Forced Outage Risk Prediction: Benefits and Comparison to Machine Learning

    [https://arxiv.org/abs/2609.04272](https://arxiv.org/abs/2609.04272)

    本研究首次将大语言模型以零样本方式应用于电网天气相关停电风险预测，发现其精度虽略逊于监督机器学习模型，但在可操作推理和地理可扩展性方面具有独特优势，两者结合可能是最佳实践。

    

    本研究考察了大语言模型（LLMs）在零样本框架下预测配电网中与天气相关的强迫停电风险的能力，无需标注训练数据。该问题被表述为在三个预测时间尺度（3小时、6小时、12小时）上的二元严重程度分类任务，使用了德克萨斯州中部一个公用事业服务区六年的停电记录和高分辨率天气数据。四种零样本大语言模型与两种监督分类器在两种输入配置下进行了基准比较：一种使用当前天气观测数据，另一种使用天气预报数据。结果表明，监督模型在宏F1分数和精确率上优于大语言模型，而新一代大语言模型取得了具有竞争力的分数。除了准确性之外，大语言模型在可操作推理和地理可扩展性方面提供了互补优势，这表明将它们与监督模型相结合可能是最佳实践。

    arXiv:2609.04272v1 Announce Type: cross  Abstract: This study examines the ability of large language models (LLMs) to predict the risk of weather-related forced outages in the distribution grid in a zero-shot framework, without labeled training data. The problem is formulated as a binary severity classification task across three forecast horizons (3h, 6h, 12h), using six years of outage records and high-resolution weather data for a utility service area in central Texas. Four zero-shot LLMs are benchmarked against two supervised classifiers across two input configurations: one using current weather observations and the other using weather forecast data. Results show that supervised models outperform LLMs on macro-F1 and precision, while newer LLM generations achieve competitive scores. Beyond accuracy, LLMs offer complementary strengths in actionable reasoning and geographic scalability, suggesting that combining them with supervised models may be the best practice.
    
[^127]: 面向参数密集型Wi-Fi人体活动识别的量子辅助内存高效训练方法

    Quantum-Assisted Memory-Efficient Training for Parameter-Intensive Wi-Fi-Based Human Activity Recognition

    [https://arxiv.org/abs/2609.04271](https://arxiv.org/abs/2609.04271)

    本文提出量子辅助内存高效训练框架Q-MET，通过混合量子-经典神经网络间接生成模型参数，显著降低了Wi-Fi人体活动识别模型的可训练参数数量和内存消耗，使其更适合资源受限设备的部署。

    

    基于Wi-Fi的人体活动识别（HAR）已成为通信感知一体化的重要组成部分，为一系列情境感知服务铺平了道路。然而，现有的大多数基于Wi-Fi的HAR系统依赖于在训练和推理阶段都需大量计算和内存资源的深度学习（DL）模型，这给实际部署带来了重大挑战。传统训练方法需要同时更新数百万个参数，导致内存消耗极其巨大。在本文中，我们提出了一种新颖的量子辅助内存高效训练框架（Q-MET），旨在同时提高训练和推理的效率。Q-MET利用混合量子-经典神经网络间接生成HAR模型的参数，与直接优化方式相比，显著减少了可训练参数的数量。为进一步支持在资源受限设备上的部署，我们集成了结构（摘要在此处被截断）

    arXiv:2609.04271v1 Announce Type: new  Abstract: Wi-Fi-based human activity recognition (HAR) has become an important part of integrated sensing and communications, paving the way for a range of context-aware services. However, most existing Wi-Fi-based HAR systems rely on deep learning (DL) models that are computationally and memory intensive in both training and inference, which poses significant challenges for real-world deployment. Conventional training requires simultaneous updates of millions of parameters, leading to prohibitive memory consumption. In this paper, we propose a novel quantum-assisted memory-efficient training framework (Q-MET) designed to improve efficiency in both training and inference. Q-MET utilizes a hybrid quantum classical neural network to indirectly generate parameters for HAR models, significantly reducing the trainable parameter count compared to direct optimization. To further support the deployment on resource-constrained devices, we integrate structu
    
[^128]: 企业家族消解不是一个字符串匹配问题：一个按名称可见性分层的公开基准

    Corporate-Family Resolution Is Not a String-Matching Problem: A Public Benchmark Stratified by Name Visibility

    [https://arxiv.org/abs/2609.04269](https://arxiv.org/abs/2609.04269)

    该论文发布了CorpFam——一个基于美国联邦奖励记录中供应商自报母公司数据构建的企业家族消解公开基准，通过按名称可见性分层评估并报告分层召回率，揭示出最强匹配方法对名称不可见的企业家族对识别率仅4.2%，证明企业家族消解不能被简化为字符串匹配问题。

    

    判断两条供应商记录是否属于同一个企业家族，是支出整合、信用风险敞口汇总和制裁筛查的前提条件。这项任务通常被当作实体匹配来处理，但两者本质不同：家族关联连接的是刻意不同的实体，而且相关证据往往不出现在任何一条记录中。我们提出了CorpFam，这是一个包含54,864个候选对、覆盖10,307个企业家族的公开基准，其数据源自6,638,350条美国联邦奖励记录，其中每个供应商都向政府登记处自行申报其最终母公司。候选对按名称可见性进行分层：即名称在归一化后是否完全相同、是否共享一个区分性词元，或完全不共享。由于各层的正例率介于10.2%到97.3%之间，我们报告按层计算的召回率（该指标不受基础率影响），而非F1分数（该指标会受基础率影响）。在5个匹配器中，表现最强的匹配器能恢复100.0%的名称相同候选对，但对名称不可见的候选对仅能恢复4.2%。

    arXiv:2609.04269v1 Announce Type: cross  Abstract: Deciding whether two supplier records belong to the same corporate family is a prerequisite for spend consolidation, credit exposure aggregation and sanctions screening. It is usually treated as entity matching, but the tasks differ: a family link connects records that are deliberately different entities, and the evidence often appears in neither record. We introduce CorpFam, a public benchmark of 54,864 candidate pairs over 10,307 corporate families, derived from 6,638,350 US federal award records in which every supplier self-reports its ultimate parent to a government registry. Pairs are stratified by name visibility: whether the names are identical after normalisation, share a distinctive token, or share none. Because strata have positive rates from 10.2% to 97.3%, we report per-stratum recall, base-rate invariant, rather than F1, which is not. The strongest of 5 matchers recovers 100.0% of identical pairs and 4.2% of invisible ones
    
[^129]: 一种通过实验风洞观测数据为航空航天代理模型提供支撑的数据融合框架

    A Data Fusion Framework for Grounding Aerospace Surrogate Model via Experimental Wind-Tunnel Observations

    [https://arxiv.org/abs/2609.04267](https://arxiv.org/abs/2609.04267)

    本文提出一种数据融合修正框架，利用风洞PSP压敏漆实验测量数据对CFD训练的深度学习气动代理模型进行在线修正，无需重新训练即可弥合CFD与实验之间的系统性差异并提升预测精度。

    

    基于高保真CFD数据训练的气动代理模型能够准确再现标量输出和整个场域的数值预测，但其预测精度受限于CFD与实验观测之间的系统性差异。我们提出了一种基于实验的修正框架，利用风洞PSP（压敏漆）测量数据来适配CFD训练的深度学习代理模型。一个基于NASA CRM翼身组合体构型的2,300个高保真CFD模拟训练的Geotransolver代理模型（涵盖几何变化、马赫数0.70-0.85以及0至4度的迎角范围），能够以R²>0.99的精度再现CFD积分气动力和俯仰力矩，但与实验数据不匹配。为了在不重新训练代理模型的情况下融入实验信息，研究者在两个来流马赫数（0.70和0.85）下、相同迎角范围内的空间配准PSP测量数据上训练了一个修正网络。

    arXiv:2609.04267v1 Announce Type: new  Abstract: Aerodynamic surrogate models trained on high-fidelity CFD data reproduce numerical predictions of both scalar outputs and entire fields accurately, yet their predictive fidelity is limited by systematic discrepancies between CFD and experimental observations. We present an experimentally grounded correction framework that adapts a CFD-trained deep learning surrogate using wind-tunnel PSP measurements. A Geotransolver surrogate trained on 2,300 high-fidelity CFD simulations of the NASA CRM wing-body configuration, spanning geometric variation, Mach 0.70-0.85, and angles of attack 0 to 4 degrees, reproduces the CFD integrated aerodynamic forces and pitching moment to R2 > 0.99 but does not match the experimental data. To incorporate experimental information without retraining the surrogate, a correction network is trained on spatially registered PSP measurements at two freestream Mach numbers (0.70 and 0.85) across the same angle-of-attack
    
[^130]: 计算内存储器注意力：一种具有RC可调温度的时域模拟Softmax电路

    Compute-in-Memory Attention: A Time-Domain Analog Softmax Circuit with RC-Tunable Temperature

    [https://arxiv.org/abs/2609.04266](https://arxiv.org/abs/2609.04266)

    该论文提出了一种基于22nm FDSOI工艺的时域模拟softmax电路，利用共享下降斜坡和RC衰减参考直接对计算内存储器生成的模拟分数电压进行指数运算与电路内归一化，无需模数转换，并可通过斜坡斜率和RC时间常数灵活调节softmax温度。

    

    Softmax是Transformer注意力机制中的关键运算，但其指数运算和归一化操作在计算内存储器加速器中会带来显著的开销，尤其是当模拟注意力分数必须先转换到数字域时。本工作提出了一种可调温度的模拟softmax电路，基于GlobalFoundries 22纳米全耗尽绝缘体上硅（FDSOI）工艺实现，可直接作用于CIM生成的分数电压，无需中间模数转换。每个输入分数通过共享的下降斜坡被转换为时域事件。相应的比较器翻转对RC衰减参考信号进行采样以生成指数权重，随后由电路内归一化级进行处理。与依赖晶体管弱反型特性来实现指数运算的模拟softmax电路不同，所提出的架构通过斜坡斜率和RC时间常数来控制softmax响应……

    arXiv:2609.04266v1 Announce Type: cross  Abstract: Softmax is a key operation in Transformer attention, but its exponentiation and normalization add significant overhead in compute-in-memory (CIM) accelerators, especially when analog attention scores must first be converted to the digital domain. This work presents a tunable-temperature analog softmax circuit in GlobalFoundries 22-nm fully depleted silicon-on-insulator (FDSOI) technology that operates directly on CIM-generated score voltages without intermediate analog-to-digital conversion. Each input score is converted into a time-domain event using a shared falling ramp. The corresponding comparator transition samples an RC-decaying reference to generate an exponential weight, which is then processed by an in-circuit normalization stage. In contrast to analog softmax circuits that rely on transistor weak-inversion behavior for exponentiation, the proposed architecture controls the softmax response through the ramp slope and RC time 
    
[^131]: ProToMEx：通过结构化表示实现快速、可解释的模型解释

    ProToMEx: Rapid, Interpretable Explanations via Structured Representations

    [https://arxiv.org/abs/2609.04265](https://arxiv.org/abs/2609.04265)

    ProToMEx利用概率主题模型学习潜在“主题”来表示分类背后的高层次语义原因，实现了超越传统特征归因的全局与局部解释，其解释保真度可与SHAP、LIME等主流方法媲美。

    

    现有的机器学习分类器事后解释方法主要聚焦于特征归因，即为单个特征分配重要性分数。尽管这种方式有价值，但它难以刻画通常驱动模型决策过程的复杂组合模式。为了克服这一局限，我们提出了ProToMEx，这是一种利用概率主题模型（PTMs）的全新可解释性范式。我们的模型无关框架学习潜在的“主题”，这些主题代表了分类背后的不同高层次原因，超越了简单的特征重要性，从而揭示潜在的语义结构。ProToMEx能够自然地提供两类解释：对模型整体行为的全局解释，以及能够厘清特定预测中多个共存原因的局部解释。我们通过实证研究表明，ProToMEx不仅能产生与SHAP和LIME等流行方法保真度相当的解释

    arXiv:2609.04265v1 Announce Type: new  Abstract: Existing post-hoc explainers for machine learning classifiers primarily focus on feature attribution, assigning importance scores to individual features. While valuable, this approach struggles to articulate the complex, combinatorial patterns that often drive a model's decision-making process. To overcome this limitation, we introduce ProToMEx, a new paradigm for explainability that leverages Probabilistic Topic Models (PTMs). Our model-agnostic framework learns latent ''topics'' that represent distinct, high-level reasons for a classification, moving beyond simple feature importance to reveal underlying semantic structures. ProToMEx naturally provides both global explanations of a model's overall behaviour and local explanations that can disentangle multiple co-existing reasons for a specific prediction. We demonstrate empirically that ProToMEx not only produces explanations of comparable fidelity to popular methods like SHAP and LIME 
    
[^132]: 面向JEPA风格世界模型的频谱目标物理潜在结构化方法

    Spectral-Target Physical Latent Structuring for JEPA-Style World Models

    [https://arxiv.org/abs/2609.04264](https://arxiv.org/abs/2609.04264)

    该论文发现了JEPA风格世界模型中“物理表示惰性”这一新失败模式，并提出轻量级傅里叶辅助头在训练时对潜在空间施加物理信息结构化约束，在不增加推理成本的情况下有效提升下游规划性能。

    

    潜在世界模型作为一种在潜在空间而非像素空间中进行预测和规划的方法，正日益受到欢迎。近期的架构，如LeWorldModel（LeWM），采用SIGReg等正则化技术联合训练编码器和预测器，以防止表示坍塌。然而，即使有此类正则化来防止表示坍塌，我们仍然识别出一种新的世界模型失败模式——“物理表示惰性”，该现象在高动态环境中尤为显著。对于这些“惰性”情况，学习到的潜在状态并未坍塌，但却无法表示关键的物理属性，从而普遍导致下游规划失败。为解决这一问题，我们提出在训练阶段引入带有轻量级“傅里叶辅助头”的辅助监督，该方法在潜在空间中强制实施基于物理信息的结构化，且不产生任何额外的推理时开销，并可推广至任意环境。

    arXiv:2609.04264v1 Announce Type: new  Abstract: Latent world models have become increasingly popular as a method to predict and plan in latent space rather than pixel space. Recent architectures, such as LeWorldModel (LeWM), jointly train the encoder and predictor using regularization techniques like SIGReg to prevent representation collapse. Even with such regularization preventing representation collapse, we identify a new world model failure mode of \textit{physical representation laziness}, particularly noted in highly dynamic environments. For these lazy cases, the learned latent states do not collapse but nonetheless fail to represent key physical properties, causing ubiquitous downstream planning failure. To resolve this issue, we propose training-time auxiliary supervision with a lightweight "Fourier auxiliary head", which enforces physically-informed structuring of the latent space with no additional inference-time cost and can be generalized to any environment. Experimentall
    
[^133]: 面向日语音乐搜索查询的低延迟拼写纠正

    Low-Latency Spell Correction for Japanese Music Search Queries

    [https://arxiv.org/abs/2609.04262](https://arxiv.org/abs/2609.04262)

    该论文提出了一种基于BART的紧凑序列到序列模型，并设计了脚本感知的合成拼写错误生成流水线，结合键盘布局模型与语音混淆先验，实现日语音乐搜索查询的低延迟拼写纠正，同时通过曲目标题脚本规范化有效减少模型幻觉。

    

    针对日语搜索查询的拼写纠正面临独特的挑战，这是因为日语共存四种书写体系（拉丁字母/罗马字、平假名、片假名和汉字），且每种书写体系会引发不同的错误模式。我们提出了一种紧凑的基于BART的序列到序列模型（3层编码器+3层解码器），专为日语音乐搜索查询的低延迟拼写纠正而设计。其核心贡献在于一个具备脚本感知能力的合成拼写错误生成流水线，该流水线通过结合键盘布局模型（QWERTY键盘和日文滑行输入）、从真实查询日志中挖掘的语音混淆先验、浊音/清音辅音交替以及假名形态错误，生成逼真的训练数据。一个关键的设计决策是在合成拼写错误之前，将混合脚本的曲目标题规范化为单一的标准脚本，我们证明这一做法对减少模型幻觉至关重要。我们还在目标音乐曲库上训练了一个自定义的字节级BPE分词器，以……

    arXiv:2609.04262v1 Announce Type: cross  Abstract: Spell correction for Japanese search queries presents unique challenges due to the co-existence of four writing scripts (Latin/romaji, hiragana, katakana, and kanji) and the distinct error patterns each script induces. We present a compact BART-based sequence-to-sequence model (3 encoder + 3 decoder layers) designed for low-latency spell correction of Japanese music search queries. The core contribution lies in a script-aware synthetic misspelling generation pipeline that produces realistic training data by combining keyboard-layout models (QWERTY and flick input), phonetic confusion priors mined from real query logs, voiced/unvoiced consonant alternations, and kana case errors. A key design decision is normalizing mixed-script catalog titles to a single canonical script before misspelling synthesis, which we show is critical for reducing model hallucinations. We train a custom byte-level BPE tokenizer on the target music catalog to ha
    
[^134]: 基于LeJEPA的分子图编码器自监督预训练

    Self-Supervised Pretraining of Molecular Graph Encoders with LeJEPA

    [https://arxiv.org/abs/2609.04261](https://arxiv.org/abs/2609.04261)

    该研究将LeJEPA自监督预训练方法适配到分子图编码器，发现预训练虽能显著提升嵌入表示质量（ogbg-molhiv冻结探针ROC-AUC提升0.123），但这一优势并不能稳健地转化为下游微调任务的性能增益。

    

    自监督预训练已经变革了语言和视觉领域，但其对分子图神经网络的价值仍存在争议。我们探究了在大规模无标注语料库上进行预训练是否能改善分子性质预测。我们将LeJEPA——一种通过草图化各向同性高斯正则化（SIGReg）进行正则化的无预测器联合嵌入预测架构——适配到分子图上，采用多种子、基于自助法的评估协议，在Wong等人的抗生素活性数据集和ogbg-molhiv上评估GPS和Chemprop风格的D-MPNN编码器。预训练改善了学习到的表示，但并不能稳健地提升微调性能。在预训练嵌入上训练的冻结探针在两个任务上都优于随机初始化（ogbg-molhiv的ROC-AUC为0.788对0.665，提升0.123），达到了已发表的自监督方法水平，但这一优势并未转化为微调收益。在抗生素支架划分上，规范划分是显著的（delta AU……（摘要在此处截断）

    arXiv:2609.04261v1 Announce Type: cross  Abstract: Self-supervised pretraining has transformed language and vision, but its value for molecular graph neural networks remains contested. We ask whether pretraining on a large unlabelled corpus improves molecular property prediction. We adapt LeJEPA, a predictor-free joint-embedding predictive architecture regularised by Sketched Isotropic Gaussian Regularisation (SIGReg), to molecular graphs, evaluating GPS and Chemprop-style D-MPNN encoders on the Wong et al. [1] antibiotic-activity dataset and ogbg-molhiv using a multi-seed, bootstrap-based protocol. Pretraining improves learned representations but does not robustly improve finetuning. A frozen probe on pretrained embeddings exceeds random initialisation on both tasks (ogbg-molhiv ROC-AUC 0.788 vs 0.665; +0.123), reaching the published self-supervised band, but this does not translate into finetuning gains. On the antibiotic scaffold split, a canonical partition is significant (delta AU
    
[^135]: 面向金融领域的EXAONE预测模型

    EXAONE Forecast for Finance

    [https://arxiv.org/abs/2609.04239](https://arxiv.org/abs/2609.04239)

    EXAONE Finance是一个专为金融预测设计的时间序列基础模型，其核心创新在于采用无注意力机制的高效架构，通过因果一维卷积和组感知池化两个线性时间复杂度算子取代自注意力机制，从而解决了现有模型计算成本过高、无法处理间歇性观测数据以及未能捕捉金融市场独特动态的问题。

    

    本技术报告介绍了EXAONE Finance（EXAONE Forecast for Finance），这是一个专为金融预测定制的金融时间序列（TS）基础模型（TSFM）。近年来，时间序列基础模型通过大规模预训练实现了强大的零样本预测性能。然而，这些模型主要是为通用领域的时间序列开发的，并且很大程度上依赖于自注意力架构（self-attention backbone），其计算成本随序列长度和变量数量呈二次方增长。此外，这些模型假设输入是完全可观测的，且其预训练语料库未能捕捉金融市场独特的动态特性。这些限制阻碍了它们在金融领域的实际应用，因为金融领域常见的是长序列、多通道、间歇性观测的面板数据。为了解决这些挑战，EXAONE Finance采用了一种无注意力机制的架构，用两个简单而有效的线性时间复杂度算子取代了自注意力机制：1）用于时间维度混合的因果一维卷积，以及2）组感知池化多层感知机。

    arXiv:2609.04239v1 Announce Type: new  Abstract: This technical report presents EXAONE Forecast for Finance (EXAONE Finance), a financial time series (TS) foundation model (TSFM) tailored to financial forecasting. Recent TSFMs achieve strong zero-shot performance through large-scale pretraining. However, they are primarily developed for general-domain TS and largely rely on self-attention backbones whose computational cost grows quadratically with sequence length and variate count. Moreover, they assume fully observed inputs and are pretrained on corpora that fail to capture the unique dynamics of financial markets. These limitations hinder their applicability to finance, where long, many-channel, intermittently observed panels are common. To address these challenges, EXAONE Finance adopts an attention-free architecture, replacing self-attention with two simple yet effective linear-time operators: 1) a causal 1D convolution for temporal mixing and 2) a group-aware pooling multi-layer p
    
[^136]: GEPARD——面向实时对话的生成式、韵律感知、自回归文本转语音模型

    GEPARD - Generative, Prosody-aware, Autoregressive text-to-speech model for Realtime Dialogue

    [https://arxiv.org/abs/2609.04222](https://arxiv.org/abs/2609.04222)

    GEPARD是一个基于标准LLM骨干网络、可在不修改vLLM推理引擎计算内核的情况下流式运行的自回归文本转语音模型，通过将零样本语音克隆等辅助机制移出解码循环，实现了真正的实时语音对话生成。

    

    我们提出了GEPARD（面向实时对话的生成式、韵律感知、自回归文本转语音模型），这是一个用于实时语音对话的流式文本转语音模型。GEPARD使用LLM骨干网络自回归地生成语音——文本和音频嵌入在单一的仅解码器（decoder-only）模型中共同训练——并通过基于FSQ（有限标量量化）的神经编解码器将其解码为波形，随着文本的到达逐块流式输出音频。我们的核心目标是构建一个可以由标准LLM推理引擎（vLLM）提供服务而无需修改其计算内核的TTS架构。这定义了总体设计原则：骨干网络是标准的全注意力transformer，而所有非平凡的辅助机制——零样本语音克隆、文本增强和分类器无关引导——都被移出自回归解码循环，转移到预填充阶段，或直接蒸馏到模型权重中。在流式端到端推理中，单一流达到的实时因子为……（原文摘要在此处截断）

    arXiv:2609.04222v1 Announce Type: cross  Abstract: We present GEPARD (Generative, Prosody-aware, Autoregressive text-to-speech model for Realtime Dialogue), a streaming text-to-speech model for real-time spoken dialogue. GEPARD generates speech autoregressively with an LLM backbone - text and audio embeddings are trained together in a single decoder-only model - and decodes it to a waveform with an FSQ-based neural codec, streaming audio chunk-by-chunk as text arrives.   Our central goal is a TTS architecture served by a standard LLM engine (vLLM) without modifying its compute kernels. This defines the overarching design principle: the backbone is a standard full-attention transformer, while all non-trivial auxiliary mechanisms - zero-shot voice cloning, text augmentation, and classifier-free guidance - are moved out of the autoregressive decode loop into prefill, or distilled directly into the weights.   On streaming end-to-end inference, a single stream reaches a Real-Time Factor of 
    
[^137]: 顺序优于联合：论在线策略蒸馏与RLVR的相互作用

    Sequential Beats Joint: On the Interplay between On-Policy Distillation and RLVR

    [https://arxiv.org/abs/2609.04108](https://arxiv.org/abs/2609.04108)

    先蒸馏后强化学习的两阶段训练方案在推理任务上持续优于纯OPD、纯RLVR及所有联合优化方法，因为OPD先扩大学生对教师解的覆盖范围、RL再在其内锐化，而联合训练会导致两种信号相互干扰。

    

    可验证奖励强化学习（RLVR）和在线策略蒸馏（OPD）已成为对推理大语言模型进行后训练的两种主流方法。先前的工作利用OPD的密集token级监督来补充稀疏的RL奖励，在单个步骤内融合这两种信号：要么作为加权加性组合，要么作为对RL优势的教师调制重缩放。在本文中，我们展示了一个简单的两阶段方案——先OPD后RL——在逻辑和数学推理基准上持续优于纯OPD、纯RLVR以及所有此类联合基线方法。除了实证结果外，我们还通过pass@$k$行为、学习动态和参数更新对这一现象提供了系统性的理解，并得出一个一致的解释：OPD扩大了学生对教师支持解的覆盖范围，而RL则在该支持范围内进行锐化，同时联合优化这两种信号会导致它们相互干扰。

    arXiv:2609.04108v1 Announce Type: cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) and on-policy distillation (OPD) have emerged as two dominant methods for post-training reasoning LLMs. Prior work uses OPD's dense token-level supervision to complement the sparse RL reward, fusing the two signals within a single step: either as a \emph{weighted-additive combination} or a \emph{teacher-modulated rescaling} of the RL advantage. In this paper, we show that a simple two-stage scheme, OPD-then-RL, consistently outperforms pure OPD, pure RLVR, and all such joint baselines across logic and math reasoning benchmarks. Beyond the empirical results, we further provide a systematic understanding of this through pass@$k$ behavior, learning dynamics, and parameter updates, yielding a consistent explanation: OPD expands the student's coverage of teacher-supported solutions and RL sharpens within that support, while jointly optimizing the two signals causes them to interfere.To p
    
[^138]: 针对重尾分布的极端分位数处理效应的位置不变估计量

    A location-invariant estimator of extremal quantile treatment effects for heavy-tailed distributions

    [https://arxiv.org/abs/2609.04018](https://arxiv.org/abs/2609.04018)

    该论文通过逆倾向得分加权将位置不变的Fraga极值指数估计量引入因果设置，并采用基于差分的外推方案使位置参数在分位数差中自然抵消，从而首次构造出对位置平移保持不变的极端分位数处理效应估计量，解决了现有方法在重尾分布下不满足位置不变性的问题。

    

    分位数处理效应（QTEs）衡量处理对结果分布的影响，在目标分位数远超数据范围的应用中，极端分位数水平上的估计是核心关注点。对于重尾潜在结果，现有的极端分位数处理效应估计量依赖于外推法结合因果极值指数（EVI）估计量，但由此得到的估计量在潜在结果分布的共同位置平移下不具有不变性，尽管总体的分位数处理效应本身是位置不变的。我们分两步解决这一问题。首先，我们利用逆倾向得分加权将位置不变的Fraga极值指数估计量适配到因果设置中。其次，我们用基于差分的方案替代原始的外推公式，在该方案下，取分位数差时位置参数会被抵消。因此，所得到的分位数处理效应估计量是位置不变的。

    arXiv:2609.04018v1 Announce Type: new  Abstract: Quantile treatment effects (QTEs) measure the effect of a treatment on the distribution of an outcome, and their estimation at extreme quantile levels is of central interest in applications where the target quantiles lie far beyond the range of the data. For heavy-tailed potential outcomes, existing extremal QTE estimators rely on extrapolation combined with a causal extreme value index (EVI) estimator, but the resulting estimator is not invariant under a common location shift of the potential outcome distributions, even though the population QTE is. We address this issue in two steps. First, we adapt the location-invariant Fraga estimator of the EVI to the causal setting using inverse propensity score weighting. Second, we replace the original extrapolation formula with a difference-based scheme, under which the location parameter cancels when quantile differences are taken. The resulting QTE estimator is therefore location invariant. W
    
[^139]: LLM4CKD：用于早期慢性肾病筛查的大语言模型

    LLM4CKD: Large Language Models for Early Stage Chronic Kidney Disease Screening

    [https://arxiv.org/abs/2609.04013](https://arxiv.org/abs/2609.04013)

    大语言模型在零样本和少样本学习设置下，无需任务特定训练即可实现与传统机器学习和深度学习方法相当的早期慢性肾病筛查性能。

    

    早期筛查慢性肾病（CKD）对于及时干预至关重要，然而大多数机器学习（ML）和深度学习（DL）方法需要标注数据和模型训练，限制了它们在现实筛查场景中的应用。本研究评估了大语言模型（LLMs）在零样本和少样本上下文学习设置下进行CKD筛查的有效性，并将其与传统ML和DL方法进行比较。我们提出了一个框架，该框架使用临床筛选的表格特征和结构化提示模板，使基于LLM的推理无需任务特定训练即可实现。LLM的性能在多种提示风格、特征配置和数据设置下进行了评估，并与标准ML、DL和表格基础模型（TFM）基线以及现有CKD筛查工具进行了比较。结果显示，LLM仅需少量样本即可获得有竞争力的性能，往往能够匹敌或超越传统方法。

    arXiv:2609.04013v1 Announce Type: new  Abstract: Early screening of chronic kidney disease (CKD) is critical for timely intervention, yet most machine learning (ML) and deep learning (DL) approaches require labeled data and model training, limiting their use in real-world screening settings. This study evaluates the effectiveness of large language models (LLMs) for CKD screening under zero-shot and few-shot in-context learning settings and compares them with traditional ML and DL methods. We propose a framework that uses clinically selected tabular features and structured prompt templates to enable LLM-based inference without task-specific training. LLM performance is evaluated across multiple prompt styles, feature configurations, and data settings, and compared with standard ML, DL, and tabular foundation model (TFM) baselines, and existing CKD screening tools. The results show that LLMs can achieve competitive performance using only a small number of examples, often matching or outp
    
[^140]: 接口诱导的轨迹审查

    Interface-Induced Trajectory Censoring

    [https://arxiv.org/abs/2609.03966](https://arxiv.org/abs/2609.03966)

    该论文发现智能体评估中的工具调用率可能被服务栈接口“审查”为零——即使模型实际发出了格式良好的调用，且2x2析因实验证明全部效应源于聊天模板与解析器之间的交互（修复任何一方都无效），仅更换服务适配器即可使同一模型得分从 0.00 跃升至 0.96/0.19。

    

    智能体评估报告的工具调用率是从服务栈中读取的。即使模型正在发出格式良好的调用，该数字也可能为零：接口在下游任何组件看到轨迹之前就将其“审查”掉了。在 BFCL v4 自身的数据、执行器和评分器上，保持权重、测试用例、解码方式和随机种子固定不变，仅更换服务适配器，同一模型的得分就可以是 0.00 或 0.96 / 0.19。对聊天模板和解析器进行的 2x2 析因实验精确定位了该效应：两个主效应都恰好为零，全部效应都落在交互作用中——没有任何组件是有缺陷的，只修复契约的一方毫无收效。在 tau-bench 的 115 个交互式零售任务上，同样的更换使服务器解析的调用从 0 增加到 636，到达工具执行环节的任务从 0 增加到 103。我们的探针实验在 Qwen2.5-Coder 的 21 倍规模范围内重现了这一漏斗效应：服务器在每个规模下都只能解析 0/100，而格式良好的发出调用在 32B 时上升到 80/100（审查后约 72……（摘要截断）

    arXiv:2609.03966v1 Announce Type: new  Abstract: Agent evaluations report a tool-call rate read off the serving stack. That number can be zero while the model is emitting well-formed calls: the interface censors the trajectory before anything downstream sees it.   On BFCL v4's own data, executor and scorer, holding weights, cases, decoding and seeds fixed and changing only the serving adapter, the same model scores 0.00 or 0.96 / 0.19. A 2x2 over chat template and parser locates the effect exactly: both main effects are exactly zero and all of it sits in the interaction -- no component is defective, and repairing one side of the contract buys precisely nothing. On tau-bench's 115 interactive retail tasks the same swap moves server-parsed calls from 0 to 636 and tasks reaching any tool execution from 0 to 103. Our probe reproduces the funnel across a 21x scale range of Qwen2.5-Coder: the server parses 0/100 at every size while well-formed emitted calls rise to 80/100 at 32B (~72 after c
    
[^141]: 免费暂停标记

    Free Pause Tokens

    [https://arxiv.org/abs/2609.03807](https://arxiv.org/abs/2609.03807)

    提出免费暂停标记，通过权重共享主干上的并行预测流为模型提供额外思考计算，在不增加上下文长度、KV缓存和推理延迟的情况下，仅以1.14倍训练计算量的代价提升下一个词元预测性能。

    

    免费暂停标记为语言模型提供额外的计算量来形成每个下一个词元预测（如同暂停标记或思考标记的作用），但它不是在序列中添加额外标记，而是通过权重共享主干上的并行预测流来承载这些计算。在实际应用中，它使一个10亿参数模型的下一个词元预测提升了2-3厘奈特。由于暂停是搭载在已有位置上而非新增位置，因此使用它是免费的：在推理时它不增加上下文长度、不增加KV缓存，且几乎不产生延迟，推理浮点运算量的增长通常无关紧要，因为它并非吞吐量的主动瓶颈。唯一的主要成本在于训练阶段：与优化过的预训练流程相比，额外的训练计算量可低至1.14倍，同时保留了大部分收益。其结果是在与标准下一个词元训练的Transformer等浮点运算、等参数量和等词元数量条件下取得的性能改进。

    arXiv:2609.03807v1 Announce Type: cross  Abstract: A free pause token gives a language model extra compute to form each next-token prediction (as a pause, or thinking, token does) but carries that compute in a parallel prediction stream over a weight-shared backbone rather than as an extra token in the sequence. It improves next-token prediction by 2-3 centinats in practice on a 1B parameter model. Because the pause rides an existing position instead of adding one, it is free to use: at inference it adds no context length, no KV cache, and essentially no latency with the growth in inference flops typically irrelevant as it is not the active bottleneck on throughput. The only primary cost is in training, where additional training compute versus an optimized pretraining pipeline is reduced to as low as x1.14 while preserving most of the benefits. The result is an isoflop, isoparameter, and isotoken improvement over standard next token trained transformers.
    
[^142]: 丹麦树种制图：光谱-时间特征与地理空间基础模型嵌入的比较

    Tree species mapping in Denmark: A comparison of spectral-temporal features with geospatial foundation model embeddings

    [https://arxiv.org/abs/2609.03480](https://arxiv.org/abs/2609.03480)

    本研究利用丹麦国家森林清查数据与Sentinel卫星观测，系统比较了人工构建的光谱-时间特征与地球观测基础模型（TESSERA和AlphaEarth）嵌入在树种分类中的表现，发现基于光谱-时间特征的多层感知机取得最佳性能（纯林宏观F1达0.843），而基础模型嵌入也展现出有竞争力的结果，为大规模森林树种制图的方法选择提供了重要参考。

    

    我们利用国家森林资源清查样地和地球观测（EO）数据对丹麦全国的树种进行制图，同时评估了基础模型在大尺度森林表征方面的潜力。我们比较了两种用于树种分类的备选输入表示方式：(i) 基于多时相Sentinel-1和Sentinel-2观测数据人工构建的光谱-时间特征（STF），以及 (ii) 由地球观测基础模型TESSERA和AlphaEarth生成的嵌入表示。两种表示方法均辅以冠层高度信息。我们针对所有输入表示评估了随机森林、XGBoost和多层感知机（MLP）分类器，并对纯林和混交林分别进行了评估。基于STF的MLP取得了最高的分类性能，在纯林和混交林上的宏观F1分数分别为0.843和0.653。在TESSERA嵌入上训练的MLP在纯林分类上也表现出具有竞争力的性能。

    arXiv:2609.03480v1 Announce Type: cross  Abstract: We map tree species across Denmark using National Forest Inventory plots and EO data, while evaluating the potential of foundation models for large-scale forest characterization. We compare two alternative input representations for tree species classification: (i) manually engineered spectral-temporal features (STF) derived from multi-temporal Sentinel-1 and Sentinel-2 observations, and (ii) embeddings generated by the EO FMs TESSERA and AlphaEarth. Both representations are complemented with canopy height information. Random forest, XGBoost, and Multi-Layer Perceptron (MLP) classifiers are evaluated for all input representations, with separate assessments for pure and mixed forest stands. The STF-based MLP achieves the highest classification performance, yielding macro F1 scores of 0.843 and 0.653 for pure and mixed stands, respectively. The MLP trained on TESSERA embeddings delivers competitive performance for pure stands, achieving r
    
[^143]: 多步临床大语言模型智能体的反事实公平性审计需要测量每动作的不稳定性底线

    Counterfactual Fairness Audits of Multi-Step Clinical LLM Agents Require a Measured Per-Action Instability Floor

    [https://arxiv.org/abs/2609.03221](https://arxiv.org/abs/2609.03221)

    临床LLM智能体在完全相同输入下本身就存在显著的动作不稳定性（约8.7%），因此反事实公平性审计必须先测量这一“每动作不稳定性底线”，否则任何检测到的人口统计学差异都无法解释。

    

    反事实审计是检查临床智能体是否对人口统计学上不同但临床上相同的患者采取不同行动的标准工具。这类审计报告一个“翻转率”：当仅改变患者描述时，智能体行动发生改变的频率。我们证明这一指标本身是不可解释的。在16个病例情景上将完全相同的条件重复运行十次（相同叙述、相同描述字符串、不改变任何变量），临床智能体的行动在8.7%的结果-情景单元格中发生了改变，且不稳定性在不同行动之间呈现8倍的异质性，从ICU升级决策的0.022到受管制物质谨慎建议的0.179。我们数据中没有任何人口统计学对比能够与这一底线区分开。第二个模型给出了6.7%的合并底线，且对六种行动的不稳定性排序几乎完全一致（Spearman 0.94，精确p=0.017），说明该底线并非单一系统的特有产物。对五次抽样进行多数投票聚合可以消除其中39%的不稳定性……

    arXiv:2609.03221v1 Announce Type: new  Abstract: Counterfactual audits are the standard tool for checking whether a clinical agent treats demographically distinct but clinically identical patients differently. They report a flip rate: how often an action changes when only the patient descriptor changes. We show that this quantity is uninterpretable on its own. Re-running an identical condition ten times over sixteen vignettes (same narrative, same descriptor string, nothing varied) moved a clinical agent's action in 8.7% of outcome-vignette cells, and instability was heterogeneous across actions by a factor of eight, from 0.022 for ICU escalation to 0.179 for controlled-substance caution. No demographic contrast in our data was distinguishable from that floor. A second model gives a pooled floor of 6.7% and ranks the six actions almost identically (Spearman 0.94, exact p=0.017), so the floor is not one system's artefact. Majority-vote aggregation over five draws removes 39% of it and t
    
[^144]: 面向编程竞赛金牌表现的语言模型后训练

    Post-Training Language Models for Gold-Medal Performance in Coding Competitions

    [https://arxiv.org/abs/2609.02849](https://arxiv.org/abs/2609.02849)

    该研究通过结合大规模题目筛选、监督微调、强化学习以及反馈驱动的测试时计算策略 GenCorrect，使语言模型在 IOI 2025 编程竞赛中取得了超越金牌分数线（438.3 分）的成绩（Nano-CC 达 468 分，Ultra-CC 达 502 分）。

    

    竞赛编程已成为检验大语言模型推理能力的关键测试，其中 IOI 和 ICPC 等国际赛事代表了最具挑战性的场景。我们提出了一条端到端的专门化流水线，结合了大规模题目筛选、合成推理轨迹、监督微调（SFT）和强化学习（RL）。利用 22,000 道精选题目，我们通过 SFT 和 RL 训练了 Nemotron-3-Nano-CC（30B-A3B），并仅通过 SFT 训练了 Nemotron-3-Ultra-CC（550B-A55B）。我们进一步提出了 GenCorrect，这是一种由反馈驱动的测试时计算策略，可迭代地生成、评估并改进多样化的解决方案。在 IOI 2025 上，Nano-CC 在后训练后从 130 分提升至 291 分，结合 GenCorrect 后达到 468 分，超过了 438.3 的金牌分数线，而 Ultra-CC 达到了 502 分。在这些结果的指导下，我们开发了一个面向竞赛的 Ultra-CC 系统，并在 IOI 2026 期间进行了前瞻性评估。

    arXiv:2609.02849v1 Announce Type: cross  Abstract: Competitive programming has become a key test of large language model reasoning, with international competitions such as IOI and ICPC representing its most challenging settings. We present an end-to-end specialization pipeline combining large-scale problem curation, synthetic reasoning traces, supervised fine-tuning (SFT), and reinforcement learning (RL). Using 22,000 curated problems, we train Nemotron-3-Nano-CC (30B-A3B) with SFT and RL and Nemotron-3-Ultra-CC (550B-A55B) with SFT alone. We further introduce GenCorrect, a feedback-driven test-time compute strategy that iteratively generates, evaluates, and refines diverse solutions. On IOI 2025, Nano-CC improves from 130 points to 291 after post-training and to 468 with GenCorrect, exceeding the gold threshold of 438.3 while Ultra-CC reaches 502. Guided by these results, we develop a competition-specific Ultra-CC system and evaluate it prospectively during IOI 2026. Under the same ti
    
[^145]: TrajMind：通过链式连接角色专精LoRA实现快慢结合的集体轨迹异常诊断

    TrajMind: Chaining Role-Specialized LoRAs for Fast-and-Slow Collective Trajectory Anomaly Diagnosis

    [https://arxiv.org/abs/2609.02540](https://arxiv.org/abs/2609.02540)

    TrajMind提出快慢双路径框架，在单一冻结的视觉-语言骨干网络上链式切换三个角色专精LoRA适配器，将持续在线筛查与按需诊断解耦，从而实现低延迟且经源数据可验证的集体轨迹异常诊断。

    

    从城市轨迹中诊断集体异常对交通治理而言日益重要，因为它能够揭示发生了什么、涉及了谁、以及事件发生的地点和时间。现有检测器可以高效地产生分数或标签，而视觉-语言流水线则能提供更丰富的语义；但两者都无法将可验证的诊断与低延迟监控相结合。核心挑战在于如何从源轨迹中识别集体模式并恢复精确的事件细节，而无需对每个监控窗口都运行完整的诊断流水线。因此，我们将持续在线的筛查与按需进行的诊断分离：筛查负责发出警报，而诊断只发布经源数据验证的“什么-谁-哪里-何时”记录。我们提出TrajMind，一个快慢结合的框架，它在一个冻结的视觉-语言骨干网络上切换三个角色专精的LoRA适配器。其慢速路径TrajMind_slow通过链式连接基于画布的……

    arXiv:2609.02540v1 Announce Type: new  Abstract: Diagnosing collective anomalies from urban trajectories is increasingly important for traffic governance, as it reveals what happened, who was involved, and where and when the event occurred. Existing detectors efficiently produce scores or labels, whereas vision--language pipelines provide richer semantics; neither couples verifiable diagnosis with low-latency monitoring. The central challenge is to recognize collective patterns and recover exact event details from the source trajectories without running the full diagnostic pipeline for every monitored window. We therefore separate always-on screening from on-demand diagnosis: screening raises alerts, while diagnosis releases only source-verified what--who--where--when records. We present TrajMind, a fast-and-slow framework that switches three role-specialized LoRA adapters over one frozen vision--language backbone. Its slow path, \textit{TrajMind$_{\text{slow}}$}, chains canvas-based t
    
[^146]: DeepAffinity：基于小语言模型的电商长期属性偏好预测

    DeepAffinity: Long-Term Aspect Preference Prediction in eCommerce using Small Language Models

    [https://arxiv.org/abs/2609.02468](https://arxiv.org/abs/2609.02468)

    提出DeepAffinity框架，利用结构化提示和专用预测头微调的小语言模型（SLM）从用户时序交互历史中预测其对品牌、尺寸、颜色等产品属性的长期偏好，性能优于标准生成式微调方法，并能提升大规模推荐质量。

    

    我们探索预测电商用户对产品属性（如品牌、尺寸和颜色）的偏好——我们将这一任务定义为“属性亲和度”。解决这一任务可以提升对客户的理解，并在推荐、搜索和营销中实现细粒度的个性化。我们将属性亲和度构建为一个时序预测任务：从用户按时间顺序排列的交互历史中预测其未来的属性选择，捕捉超越当前会话、随时间演变的长期偏好。为此，我们提出了DeepAffinity，它利用小语言模型，结合结构化提示词以及针对该任务微调的专用预测头。我们展示了DeepAffinity优于标准的生成式微调方法，而通用开源大语言模型在没有进行任务特定微调的情况下表现不佳，凸显了它们在建模细致行为方面的局限性。最后，DeepAffinity在大规模跨国数据上提升了推荐质量。

    arXiv:2609.02468v1 Announce Type: cross  Abstract: We explore predicting eCommerce user preferences for product aspects such as brand, size, and color - a task we define as Aspect Affinity. Solving this task improves customer understanding and enables fine-grained personalization in recommendation, search, and marketing. We frame Aspect Affinity as a temporal prediction task: forecasting a users future aspect choices from their time-ordered interaction history, capturing long-term preferences that evolve beyond the current session. To this end, we propose DeepAffinity, which leverages Small Language Models (SLMs) with structured prompts and specialized prediction heads fine-tuned for this task. We show DeepAffinity outperforms standard generative fine-tuning methods, while general-purpose open-source LLMs perform poorly without task-specific tuning, highlighting their limits in modeling nuanced behavior. Finally, DeepAffinity enhances recommendation quality on a large-scale multination
    
[^147]: OR-Transformer：将实时决策扩展至1000件商品规模

    OR-Transformer: Scaling Real-Time Decision-Making to 1,000 Items

    [https://arxiv.org/abs/2609.01933](https://arxiv.org/abs/2609.01933)

    OR-Transformer通过商品置换等变的Transformer架构和路径梯度训练的深度强化学习方法，将随机需求下的联合补货决策扩展到1024种商品规模，随规模增长持续优于MILP基线并大幅降低在线决策时间。

    

    现代供应链运营可能需要在相关随机需求、异质提前期和共享固定订货成本的条件下，协调数千种异质商品的补货，从而产生超过10⁴维的观测空间。在这种规模下，滚动时域随机混合整数线性规划（MILP）变得极其缓慢，而标准强化学习（RL）方法在高维动作空间中面临日益严峻的信用分配挑战。我们提出了OR-Transformer，这是一个面向随机需求下联合补货问题的深度强化学习框架，采用商品置换等变的Transformer架构，并通过库存动态进行路径梯度训练。在规模高达1,024个库存商品的问题上，随着规模的增长，OR-Transformer越来越多地超越基于学习的方法和滚动时域MILP基线。它还将在线决策时间降低了超过（原文摘要在此处被截断）

    arXiv:2609.01933v1 Announce Type: new  Abstract: Modern supply chain operations can require coordinating replenishment across thousands of heterogeneous items under correlated stochastic demand, heterogeneous lead times, and shared fixed ordering costs, yielding observation spaces exceeding $10^4$ dimensions. At this scale, rolling-horizon stochastic mixed-integer linear programs (MILPs) become prohibitively slow, while standard reinforcement learning (RL) methods face increasingly challenging credit assignment in high-dimensional action spaces. We introduce OR-Transformer, a deep reinforcement learning framework for joint replenishment under stochastic demand, with an item-permutation-equivariant Transformer architecture and pathwise-gradient training through the inventory dynamics. Across problem sizes up to 1,024 inventory items, OR-Transformer increasingly outperforms learning-based and rolling-horizon MILP baselines as scale grows. It also reduces online decision-making time by ov
    
[^148]: hLLM：面向生成式重排序的单遍解码

    hLLM: Single Pass Decoding for Generative Reranking

    [https://arxiv.org/abs/2609.01807](https://arxiv.org/abs/2609.01807)

    提出hLLM，通过轻量自注意力头从LLM预填充隐状态读取项目-位置得分矩阵，并用匈牙利算法求最优二分匹配，在O(1)次前向传播内一次性解码全部N个序数，从而将生成式重排序的解码从逐token自回归生成变为常数次前向传播，且天然保证输出为有效排序。

    

    arXiv:2609.01807v1 公告类型： cross 摘要：大语言模型（LLM）实现了最先进的生成式排序质量，但其产生的排序结果必须经过解码，而自回归解码每生成一个token就需要一次顺序前向传播。我们观察到，排序器必须输出的token仅仅是N个序数值，用于按排序顺序命名各个项目，而这种狭窄的、具有置换结构的输出格式使得我们可以采用比从左到右生成高效得多的解码策略。我们提出了hLLM（匈牙利LLM），一种针对该输出格式专门设计的解码策略，它能够在O(1)次前向传播中解码全部N个序数。hLLM通过一个轻量级的自注意力头，从LLM预填充阶段的隐状态中读取一个N×K的项目-位置得分矩阵，然后利用匈牙利算法将该矩阵的最优二分匹配作为序数解码，从而在构造层面（而非通过事后修复）保证输出是一个有效的置换。通过对训练信号的系统性研究……

    arXiv:2609.01807v1 Announce Type: cross  Abstract: Large language models (LLMs) achieve state-of-the-art generative ranking quality, but the ranking they produce must be decoded, and autoregressive decoding spends one sequential forward pass per emitted token. We observe that the only tokens a ranker must emit are the $N$ ordinal values naming the items in ranked order, and that this narrow, permutation-structured output format admits decoding strategies which are much more efficient than left-to-right generation. We introduce hLLM (Hungarian LLM), a format-specialized decoding strategy that decodes all $N$ ordinals in $O(1)$ forward passes. hLLM reads an $N \times K$ item-position score matrix off the LLM's prefill hidden states with a lightweight self-attention head, then decodes the ordinals as the optimal bipartite assignment of that matrix via the Hungarian algorithm, yielding a valid permutation by construction rather than by repair. Through a systematic study of training signals
    
[^149]: SocialBuddy：为社交场景量身定制的搜索智能体

    SocialBuddy: Tailoring Search Agent for Social Scenarios

    [https://arxiv.org/abs/2609.01641](https://arxiv.org/abs/2609.01641)

    该论文提出首个面向社交场景的智能体搜索框架SocialBuddy，并构建了包含20万用户画像、1000万社交帖子和5万推理轨迹的大规模模拟环境SocialEnv，以解决复杂社交搜索中的性能退化与稀疏奖励难题。

    

    在数字社交互动时代，从海量社交信息流中搜索朋友的帖子已成为用户的基本需求。然而，尽管现代智能体搜索框架在传统检索任务中取得了显著成功，但面对异构的用户查询和多维度的社交内容时，它们会陷入失效，导致在复杂社交搜索中出现严重的性能退化。为弥补这一差距，我们推出了SocialBuddy——首个专为社交场景量身定制的智能体搜索框架。具体而言，我们构建了SocialEnv，这是首个面向社交搜索的大规模模拟环境。借助自动化的数据与轨迹合成流水线，SocialEnv包含20万个用户画像、1000万条社交帖子和5万条推理轨迹，为社交搜索智能体的研发奠定了坚实基础。为解决社交搜索中稀疏奖励带来的信用分配难题，我们……

    arXiv:2609.01641v1 Announce Type: cross  Abstract: In the era of digital social interaction, searching friends' posts from massive social streams has become a fundamental user need. However, while modern agentic search frameworks have achieved remarkable success in conventional retrieval tasks, they break down when confronted with heterogeneous user queries and multi-dimensional social feeds, resulting in severe performance degradation in complex social search. To bridge this gap, we introduce SocialBuddy, the first agentic search framework tailored for social scenarios. Specifically, we construct SocialEnv, the first large-scale simulated environment for social search. Powered by an automated data and trajectory synthesis pipeline, SocialEnv includes 200K user profiles, 10 million social posts, and 50K reasoning trajectories, establishing a solid foundation for the development of social search agents. To tackle the credit assignment dilemma caused by sparse rewards in social search, w
    
[^150]: Omega-N：可解释的结构节点描述符及其适用域

    Omega-N: Interpretable Structural Node Descriptors and Their Applicability Domain

    [https://arxiv.org/abs/2609.01633](https://arxiv.org/abs/2609.01633)

    本文提出 Omega-N，通过局部化三角指标因子并结合配置零模型超额与多尺度个性化 PageRank 邻域两项修正，得到每节点十个仅依赖图结构、无需属性与训练的可解释描述符，并验证了全局标量对标谱基线、逐节点归因在结构中心数量未知时更具优势的理论预测。

    

    复合结构指标用一个数字概括整个网络；对于基于三角形的指标而言，它在谱上是冗余的：Tr(A^3) 就是邻接谱的三阶矩。非冗余的信息位于下一层级，即 diag(A^3)，它依赖于特征向量，无法由谱唯一确定。该指标族理论论文中的一个推论指出并预测：全局标量应当与经过锐化的谱基线相媲美而非超越它们，而逐节点归因方法则应在结构中心数量未知的情况下表现更优。本文对这一预测进行了验证。我们通过将四个因子逐一局部化来构建 Omega-N。直接的局部化方法条件数很差；来自已发表实践的两项修正解决了这一问题——即对每个局部因子引入配置零模型超额，以及在多个尺度上采用个性化 PageRank 邻域——从而仅凭图结构本身，为每个节点生成十个可解释的特征，无需节点属性、无需训练。

    arXiv:2609.01633v1 Announce Type: cross  Abstract: A composite structural index summarises a network in one number; for a triangle-based index it is spectrally redundant: Tr(A^3) is the third moment of the adjacency spectrum. The non-redundant content sits one level down, in diag(A^3), which depends on eigenvectors and is not spectrally determined. A corollary in the theory paper for this index family stated that, and predicted: the global scalar should tie sharpened spectral baselines rather than beat them, while the node-wise attribution should do better where the number of structural epicentres is unknown. This paper tests it.   We construct Omega-N by localizing each of the four factors. The direct localization is badly conditioned; two corrections from published practice fix it, a configuration-null excess for every local factor and a personalized-PageRank neighbourhood at several scales, giving ten interpretable features per node from the graph alone, with no attributes, training
    
[^151]: 便宜的验证器，巨大的盲区：衡量成本节约级联的可靠性代价

    Cheap Verifiers, Large Blind Spots: Measuring the Reliability Cost of Cost-Saving Cascades

    [https://arxiv.org/abs/2609.01345](https://arxiv.org/abs/2609.01345)

    该研究通过真实LLM实验发现，推理级联中廉价验证器对学生模型错误答案的“盲区”随学生能力增强而扩大、随验证器能力增强而缩小，恰好在级联机制赖以存在的低成本配置下最为严重，而用前沿验证器消除盲区又会因过度升级而抵消成本节约，从而揭示了成本节约级联设计背后隐藏的显著可靠性代价。

    

    推理级联通过用廉价模型回答大多数查询，并将困难的长尾部分升级给作为验证器的前沿模型来降低成本。一个自然的扩展方式形成闭环：在验证器的拒绝样本上微调廉价的学生模型，使升级率（以及成本）逐轮下降。我们在真实的LLM上测量了这个循环，并报告了四项发现。首先，验证器的盲区——即它接受的学生错误答案的比例——很大且呈对抗性变化：它随学生能力的增强而增大（当学生模型从0.5B扩展到32B时，β从0.12增至0.55），并随验证器能力的增强而减小，因此在“廉价学生+廉价验证器”这一级联机制所天然创造的情境下，盲区最为严重。其次，花钱消除盲区会抵消节省的成本：一个前沿验证器能把β降到约0.05，但随后会在46%的困难MATH查询上进行升级，而真实错误率仅为39%，这意味着几乎一半的流量都要支付前沿模型的价格。第三，朴素的纠正性微调……（摘要在此处被截断）

    arXiv:2609.01345v1 Announce Type: new  Abstract: Inference cascades cut cost by answering most queries with a cheap model and escalating a hard tail to a frontier model that acts as verifier. A natural extension closes the loop: fine-tune the cheap student on the verifier's rejections so the escalation rate, and cost, fall each round. We measure this loop on real LLMs and report four findings. First, the verifier's blind spot, the fraction of the student's wrong answers it accepts, is large and moves adversarially: it grows with student capability ($\beta$ from 0.12 to 0.55 as the student scales 0.5B to 32B) and shrinks with verifier capability, so it is worst in the cheap-student, cheap-verifier regime cascades exist to create. Second, buying it away returns the saving: a frontier verifier drives $\beta$ to about 0.05 but then escalates on 46% of hard-MATH queries against a 39% true error rate, paying the frontier price on nearly half of all traffic. Third, naive corrective fine-tunin
    
[^152]: 面向大语言模型时序评估与知识更新的合成世界

    Synthetic Worlds for Temporal Evaluation and Knowledge Updating in LLMs

    [https://arxiv.org/abs/2609.00184](https://arxiv.org/abs/2609.00184)

    该论文提出了一个模拟驱动的合成框架，通过虚构未来世界的 ParallelEvents 基准避免评估污染，并利用 Synapse 训练框架（结合中期训练与指令微调）实现大语言模型的可扩展知识更新，性能比现有方法提升 14.23%。

    

    大语言模型（LLM）依赖于静态的预训练语料库，导致其知识随时间推移而变得过时。现有的知识编辑评估方法要么容易遭受快速的数据污染，要么依赖于与现有刚性知识相冲突的反事实编辑。在本工作中，我们提出了一个合成的、模拟驱动的框架，用于研究大语言模型中的知识插入。我们引入了 {\sc ParallelEvents}，这是一个由虚构但逼真的未来世界构成的基准，能够生成连贯的事件轨迹以进行受控评估，在避免污染的同时保持一致性。基于该数据集，我们开发了 {\sc Synapse}，这是一个利用模型自身生成的数据、通过中期训练（mid-training）和指令微调来更新模型参数的训练框架。这一合成流程实现了可扩展的知识整合，而无需昂贵的人工策划数据。实验结果表明，{\sc Synapse} 的性能比现有方法高出 14.23%。

    arXiv:2609.00184v1 Announce Type: new  Abstract: Large language models (LLMs) rely on static pretraining corpora, causing their knowledge to become outdated over time. Existing approaches for evaluating knowledge edits either suffer from rapid contamination or rely on counterfactual edits that conflict with rigid existing knowledge. In this work, we propose a synthetic, simulation-driven framework for studying knowledge insertion in LLMs. We introduce {\sc ParallelEvents}, a benchmark of fictional yet realistic future worlds that generates coherent event trajectories for controlled evaluation, avoiding contamination while preserving consistency. Building on this dataset, we develop {\sc Synapse}, a training framework that uses model-generated data to update model parameters via mid-training and instruction tuning. This synthetic pipeline enables scalable knowledge integration without costly human-curated data. Empirically, {\sc Synapse} outperforms existing methods by 14.23\%, demonstr
    
[^153]: 压力测试高效的负责任AI评估：当计算节省改变基准测试结论时

    Stress-Testing Efficient Responsible-AI Evaluation: When Compute Savings Change Benchmark Conclusions

    [https://arxiv.org/abs/2608.31108](https://arxiv.org/abs/2608.31108)

    该研究首次对负责任AI基准测试中的高效评估方法进行系统性压力测试，发现批处理等计算节省手段能在降低能耗的同时保持基准结论的稳定，而INT8量化虽能保持评估质量却会使能耗增至基线的1.79-4.26倍。

    

    高效评估改变了用于支持模型行为结论的协议，但很少有研究检验这些结论在评估本身变得更廉价之后是否仍然稳定。我们通过在七种涵盖批处理、量化、基准缩减及其组合的条件下，对三个稠密模型和专家混合模型在BBQ和BBQ-V上进行评估，对负责任AI基准测试中的结论稳健性进行压力测试。我们并不将总体准确率的保持视为充分条件，而是将准确率、偏见的严重程度与流行度、推理质量、子群体行为、子集成员稳定性、运行时间以及实测GPU能耗与全量基准的BF16基线进行对比。更大的批处理使准确率保持在基线的0.35个百分点以内，并产生相对较小的子群体变化，同时在六个模型-数据集设置中的五个里降低了能耗。INT8在很大程度上保持了质量，但使用了1.79-4.26倍于基线的能耗。

    arXiv:2608.31108v1 Announce Type: new  Abstract: Efficient evaluation changes the protocol used to support claims about model behavior, yet it is rarely tested whether those claims remain stable after the evaluation itself is made cheaper. We stress-test conclusion robustness in responsible-AI benchmarking by evaluating three dense and mixture-of-experts models on BBQ and BBQ-V under seven conditions spanning batching, quantization, benchmark reduction, and their combinations. Rather than treating preserved aggregate accuracy as sufficient, we compare accuracy, bias severity and prevalence, reasoning quality, subgroup behavior, subset-membership stability, runtime, and measured GPU energy against a full-benchmark BF16 baseline. Larger batching keeps accuracy within 0.35 percentage points of baseline and produces comparatively small subgroup changes, while reducing energy in five of six model--dataset settings. INT8 largely preserves quality but uses 1.79--4.26$\times$ baseline energy. 
    
[^154]: 一种用于自动化无损检测图像分析的视觉问答模型

    A Visual Question Answering Model to Automate Nondestructive Evaluation Image Analysis

    [https://arxiv.org/abs/2608.29408](https://arxiv.org/abs/2608.29408)

    本研究提出了一种专为无损检测设计的视觉问答模型，结合ResNet-50图像特征提取与GPT-2语言生成能力，使检测人员能够通过自然语言直接查询检测图像，显著提升检测效率、减少错误并增强现场实用性。

    

    本研究介绍了一种专门为无损检测应用设计的视觉问答（VQA）模型。VQA模型允许检测人员以交互方式查询无损检测图像，提出诸如“是否存在裂纹”或“缺陷位于何处”等有针对性的问题，并从模型获得精确的回答。该系统利用深度学习和自然语言处理技术，集成了图像特征提取（通过ResNet-50模型）和语言生成能力（通过GPT-2），以提供准确且信息丰富的反馈。通过实现直接的问答交互，该VQA模型显著提高了检测效率，减少了潜在错误，并增强了在实际现场场景中的可用性。

    arXiv:2608.29408v1 Announce Type: cross  Abstract: This study introduces a Visual Question Answering model designed specifically for nondestructive evaluation applications. VQA models allow inspectors to interactively query NDE images, asking targeted questions like, Is there a crack or Where is the defect located and receive precise answers from the model. Leveraging deep learning and natural language processing, the developed system integrates image feature extraction (via a ResNet-50 model) and language generation capabilities (via GPT-2) to provide accurate, informative feedback. By enabling direct question-and-answer interactions, this VQA model significantly improves inspection efficiency, reduces potential errors, and enhances usability in practical field scenarios.
    
[^155]: TACIT-Switch：基于删失监督的LLM智能体成本感知模型升级方法

    TACIT-Switch: Cost-Aware Model Escalation for LLM Agents from Censored Supervision

    [https://arxiv.org/abs/2608.27911](https://arxiv.org/abs/2608.27911)

    提出TACIT-Switch方法，利用教师标注的删失干预时间学习永久性移交策略，以成本感知的方式决定何时将LLM智能体从小模型骨干升级至大模型骨干，部署时无需教师参与即可将成功率提升7.4-11.1个百分点。

    

    采用较小语言模型骨干的智能体成本较低，但可能陷入持续的失败模式；而采用较大骨干的智能体通常更可靠，但成本更高。这种可靠性-成本之间的权衡促使人们研究路由方法，以决定何时调用具有更大骨干的智能体：可以在执行之前、在固定的轨迹前缀之后，或在各个步骤局部进行。我们的方法TACIT-SWITCH从累积的轨迹证据和教师标注的删失干预时间中学习永久性移交策略。它将每个标注表示为累积风险尺度上的区间删失观测。由此得到的混合治愈阈值模型可估计配对的强模型rollout成功的概率，以及在成功条件下的移交阈值；部署时无需教师参与。在基于机制的多步模拟中，TACIT-SWITCH相比基线方法将成功率提升了7.4-11.1个百分点。

    arXiv:2608.27911v1 Announce Type: new  Abstract: Agents with smaller language-model backbones are less expensive but can drift into persistent failure modes, whereas those with larger backbones are generally more reliable but more costly. This reliability-cost trade-off motivates routing methods that decide when to invoke an agent with a larger backbone: before execution, after a fixed trajectory prefix, or locally at individual steps. Our method, TACIT-SWITCH, learns permanent handoff policies from accumulated trajectory evidence and Teacher-Annotated Censored Intervention Times (TACIT). It represents each annotation as an interval-censored observation on a cumulative-risk scale. The resulting mixture-cure threshold model estimates the probability that the paired Strong rollout succeeds and, conditional on success, the handoff threshold; no teacher is required at deployment. In a mechanism-based multi-step simulation, TACIT-SWITCH improves success by 7.4-11.1 percentage points over ta
    
[^156]: 令牌级广告

    Token-Level Advertising

    [https://arxiv.org/abs/2608.27382](https://arxiv.org/abs/2608.27382)

    我们提出了LAMA，一种令牌级广告拍卖机制，将广告商影响嵌入生成式AI的生成过程，同时满足激励兼容性和个体理性，并实现接近最优的福利，实验证明其能提升平台福利和收入。

    

    摘要：arXiv:2608.27382v1 公告类型：交叉 摘要：生成式人工智能正在改变人们获取信息的方式，挑战了围绕预定义广告位构建的传统广告机制。为了迈向生成原生的广告，我们提出了潜在广告商混合拍卖（LAMA），一种令牌级广告机制，将广告商的影响力直接嵌入生成过程中。广告商报告局部延续价值，这些价值诱导出广告商特定的下一令牌策略，平台通过潜在混合解码这些策略，同时更新分配后验。我们证明LAMA满足马尔可夫激励兼容（DSIC）和个体理性（IR），并实现接近最优的KL正则化福利。我们进一步开发了一种基于学习的实现，该实现从学习的局部优势和根值在线重建所需的报告。在真实世界商业搜索查询分割上的概念验证实验表明，LAMA在保持面向用户的响应质量的同时，提高了平台福利和收入。

    arXiv:2608.27382v1 Announce Type: cross  Abstract: Generative AI is transforming how people access information, challenging traditional advertising mechanisms built around predefined slots. Towards generation-native advertising, we propose the Latent Advertiser Mixture Auction (LAMA), a token-level advertising mechanism that embeds advertiser influence directly into the generation process. Advertisers report local continuation values that induce advertiser-specific next-token policies, from which the platform decodes through a latent mixture while updating an allocation posterior. We show that LAMA satisfies Markov DSIC and IR, and achieves near-optimal KL-regularized welfare. We further develop a learning-based implementation that reconstructs the required reports online from learned local advantages and root values. Proof-of-concept experiments on real-world commercial-search query splits show that LAMA improves platform welfare and revenue while maintaining user-facing response qual
    
[^157]: 泛化之前的通道化：将“顿悟”现象作为动力学探针

    Canalization Before Generalization: Grokking as a Dynamical Probe

    [https://arxiv.org/abs/2608.25813](https://arxiv.org/abs/2608.25813)

    本文通过权重衰减脉冲在“顿悟”平台期揭示了解选择的通道化过程，表明在可见泛化之前就形成了稳定的剂量排序，为理解神经网络泛化机制提供了新的动力学视角。

    

    对于过参数化的神经网络，许多解能同样好地拟合训练数据，但在未见样本上的行为却大相径庭。“顿悟”现象将训练拟合与可见泛化分离，为研究训练过程中这种选择如何发展提供了一个窗口。我们在这一平台期扫描短时、固定持续时间的权重衰减脉冲，并测量它们如何改变后续的泛化时间。在三个“顿悟”任务中，这些变化在平台期早期是无序的，但后来形成稳定的剂量排序，即更强的权重衰减增加导致更早的泛化，更强的权重衰减减少导致更晚的泛化。这种排序在所有三个任务中都在可见泛化之前出现。同时，扰动与基线泛化检查点之间的测试损失障碍趋向于零，而有序的时间效应仍然持续。我们将这种日益受限的解选择与持久的时间效应组合称为“通道化”，并探讨其作为训练动力学探针的意义。

    arXiv:2608.25813v1 Announce Type: new  Abstract: For overparameterized neural networks, many solutions can fit the training data equally well while behaving very differently on unseen samples. Grokking separates training fit from visible generalization, providing a window for studying how this selection develops during training. We sweep short, fixed-duration weight-decay (WD) pulses across this plateau and measure how they shift later generalization time. Across three grokking tasks, these shifts are unordered early in the plateau but later form a stable dose ordering, with stronger WD increases leading to earlier generalization and stronger WD decreases leading to later generalization. This ordering emerges before visible generalization in all three tasks. Meanwhile, test-loss barriers between perturbed and baseline generalization checkpoints collapse toward zero while the ordered timing effects persist. We call this combination of increasingly constrained solution selection and pers
    
[^158]: DeMMO：通过多任务学习对数字移动结果进行纵向和跨疾病建模

    DeMMO: Longitudinal and Cross-Disease Modelling of Digital Mobility Outcomes via Multi-Task Learning

    [https://arxiv.org/abs/2608.25073](https://arxiv.org/abs/2608.25073)

    DeMMO提出了一种可解释的多任务学习框架，能同时建模多种疾病的纵向数字移动结果与多临床结局，并自动捕捉跨疾病的共享和独特模式。

    

    数字移动结果（DMOs）源自可穿戴传感器，表征日常生活中的移动能力，并为监测疾病进展提供了一种有前景的手段。然而，大多数DMO研究仅在一次就诊中检查一种疾病；它们没有建模多变量DMO与多种临床结果之间的关系如何跨疾病共同演变。技术上，现有的时间多任务框架可以建模个体疾病内的进展，但它们不能跨疾病联合建模多个预测结果，尤其是在疾病队列不共享参与者时。为弥补这些空白，我们提出了DeMMO，一个用于纵向、多疾病和多结果学习的可解释框架。DeMMO通过纵向DMO系数矩阵表示每个疾病-结果目标，并将时间正则化与稳定及访视特定特征选择相结合。其核心技术贡献是自动跨疾病和跨结果的建模机制，旨在捕捉疾病间共享和独特的移动模式。

    arXiv:2608.25073v1 Announce Type: new  Abstract: Digital mobility outcomes (DMOs) derived from wearable sensors characterise mobility in daily life and offer a promising means of monitoring disease progression. Yet most DMO studies examine one disease at one visit; they do not model how multivariate DMO relationships with multiple clinical outcomes evolve jointly across diseases. Technically, existing temporal multi-task frameworks can model progression within an individual disease, but they do not jointly model multiple prediction outcomes across diseases, particularly when disease cohorts do not share participants. To address these gaps, we propose DeMMO, an interpretable framework for longitudinal, multi-disease, and multi-outcome learning. DeMMO represents each disease-outcome objective by a longitudinal DMO coefficient matrix and combines temporal regularisation with stable and visit-specific feature selection. Its central technical contribution is an automatic cross-disease and c
    
[^159]: 语义覆盖层：通过超越令牌和引导向量的注释缓解提示注入

    Semantic Overlays: Mitigating Prompt Injection with Annotations Beyond Tokens and Steering Vectors

    [https://arxiv.org/abs/2608.23873](https://arxiv.org/abs/2608.23873)

    该论文提出了一种名为“语义覆盖层”的新技术，通过向模型输入添加非文本通道来缓解提示注入攻击，利用小型学习的适配器在冻结模型的残差流中创建带外注释，从而增强模型对片段身份的理解。

    

    摘要：arXiv:2608.23873v1 公告类型：新 摘要：语言模型看到的一切都是令牌。服务堆栈知道每个片段是什么——用户输入、工具输出、指令——但模型必须自己跟踪这些，它可能会失去跟踪或被混淆：文本可以被写成看起来像任何东西。提示注入是对这种现象的自然利用。通过扰乱模型对片段身份的理解，攻击者可以诱导不必要的、可能危险的行为。在模型输入中添加一个非文本通道——一种超越文本传达片段身份的方式——缓解了这类攻击。因此，我们引入了一种通用的引导技术，称为语义覆盖层：小型学习的适配器，应用于冻结模型的残差流中的选定预填充位置。在片段上铺设覆盖层创建了一个带外注释通道，该通道无法通过令牌复制。与引导向量不同，语义覆盖层是经过训练的、可适应的，并有选择性地应用。一个覆盖层...

    arXiv:2608.23873v1 Announce Type: new  Abstract: Everything a language model sees is tokens. The serving stack knows what each span is -- user input, tool output, instructions -- but the model must keep track of that itself, and it can lose track or be confused: text can be written to read like anything. Prompt injection is a natural exploit of this phenomenon. By scrambling the model's understanding of span identity, an attacker can induce unwanted and potentially dangerous actions. Adding a non-textual channel to the model's input -- a way to communicate span identity beyond text -- mitigates this class of attack. We thus introduce a general steering technique called Semantic Overlays: small learned adapters applied at chosen prefill positions to a frozen model's residual stream. Laying an overlay over a span creates an out-of-band annotation channel that cannot be replicated by tokens. Unlike steering vectors, Semantic Overlays are trained, adaptable, and selectively applied. An ove
    
[^160]: MetaCaster：面向轻量级时间序列预测器端到端少样本学习的元框架优化智能体

    MetaCaster: Meta-Harness-Optimized Agent for End-to-End Few-Shot Learning of Lightweight Time Series Forecasters

    [https://arxiv.org/abs/2608.23473](https://arxiv.org/abs/2608.23473)

    本文提出MetaCaster，一种元框架优化的多智能体系统，通过智能体数据生成从少量示例和文本中自动训练轻量级时间序列预测器，实现了资源受限场景下的高效少样本学习。

    

    时间序列预测（TSF）正朝着多模态和智能体化方向发展，但在资源受限的场景中，使用基础模型仍然不经济，此时紧凑、专门的预测器更为理想。然而，轻量级预测器通常需要大量训练数据，这限制了它们在数据稀缺、积累缓慢或对隐私敏感的时间序列领域中的应用。为解决这一困境，我们研究了轻量级预测器少样本学习这一具有挑战性的问题。我们提出了MetaCaster，一个元框架优化的多智能体系统，利用智能体数据生成，仅从少量示例和文本上下文中自动训练专门的轻量级预测器。我们的工作突显了一种新的TSF范式，其中智能体不作为预测器，而是作为中间工程师，为部署准备高效、任务特定的预测器。在18个数据集和23个最先进的轻量级方法上进行了实验。

    arXiv:2608.23473v1 Announce Type: cross  Abstract: Time series forecasting (TSF) is evolving toward multimodal and agentic settings, yet using foundation models remains uneconomical in resource-constrained scenarios, where compact, specialized forecasters are more desirable. However, lightweight forecasters typically require substantial training data, limiting their use in domains with scarce, slowly accumulated, or privacy-sensitive time series. To address this dilemma, we investigate the challenging problem of few-shot learning for lightweight forecasters. We propose MetaCaster, a meta-harness-optimized multi-agent framework that uses agentic data generation to automatically train specialized lightweight forecasters from only a few examples and textual contexts. Our work highlights a new TSF paradigm in which agents act not as forecasters but as intermediary engineers that prepare efficient, task-specific forecasters for deployment. Experiments on 18 datasets, 23 state-of-the-art lig
    
[^161]: 基于说话人级动态条件随机场的双尺度状态空间建模用于会话语音情感识别

    Dual-Scale State-Space Modeling with Speaker-Wise Dynamic CRF for Speech Emotion Recognition in Conversation

    [https://arxiv.org/abs/2608.22399](https://arxiv.org/abs/2608.22399)

    本文提出DSSM-CRF纯音频架构，利用双向状态空间模型在帧和对话双尺度上编码语音表示，并通过说话人级动态条件随机场显式分离跨说话人上下文影响与说话人内部情感演变两种过程，从而提升会话语音情感识别的效果。

    

    会话语音情感识别需要将跨时间尺度的声学证据与两种交互过程相协调：跨说话人的上下文影响和说话人内部的情感演变。我们提出DSSM-CRF，一种纯音频架构，能够显式地分离这些过程。双向状态空间模型在帧尺度和对话尺度上编码融合的自监督语音表示，使每个话语表示都能捕获局部韵律以及来自所有说话人的上下文信息。随后，解码器将每个说话人的话语排列成独立的动态条件随机场链。说话人链中相邻的话语构成一个转移对，其分数由语料库级别的转移矩阵与基于两个上下文化话语预测出的残差相结合而得到。一个辅助目标用于监督每对话语是否发生情感变化，但不参与维特比推理。因此，对话者的轮次影响上下文。

    arXiv:2608.22399v2 Announce Type: replace  Abstract: Conversational speech emotion recognition must reconcile acoustic evidence across temporal scales with two interaction processes: cross-speaker contextual influence and within-speaker emotion evolution. We propose DSSM-CRF, an audio-only architecture that explicitly separates these processes. Bidirectional state-space models encode fused self-supervised speech representations at frame and dialogue scales, so each utterance representation captures local prosody and context from all speakers. The decoder then orders each speaker's utterances into an independent dynamic conditional random field chain. Consecutive utterances in a speaker's chain form a transition pair whose score combines a corpus-level transition matrix with a residual predicted from the two contextualized utterances. An auxiliary objective supervises whether each pair changes emotion but does not participate in Viterbi inference. Thus, interlocutor turns affect context
    
[^162]: 通过机器学习与搜索算法优化柴油发电机负载提升海上石油平台能源效率

    Improving Energy Efficiency of Oil Platforms Through Optimal Loading of Diesel Generators Using Machine Learning and Search Algorithms

    [https://arxiv.org/abs/2608.22076](https://arxiv.org/abs/2608.22076)

    本研究首次将机器学习与搜索算法结合，针对海上石油平台的柴油发电机负载进行优化，以降低能源消耗而非提升产量，填补了该领域研究空白。

    

    摘要：日益增长的能源需求、化石燃料枯竭和气候变化凸显了提高能源生产和消费效率的必要性。海上油气平台面临能源利用效率低下、系统故障、可达性差和环境影响等挑战。机器学习（ML）为提升这些系统的安全性、可持续性和效率提供了机遇；然而，以往研究主要集中在增加石油产量，而非减少平台上的能源消耗。本研究探讨了使用机器学习与搜索算法来提高海上石油平台柴油效率的方法。研究分析了从苏格兰一座平台收集的18个月数据，重点关注四台柴油发电机作为主要柴油消耗设备。在探索性数据分析和异常值检测之后，开发了回归模型来预测不同发电机功率负载下的每日柴油消耗量。

    arXiv:2608.22076v1 Announce Type: cross  Abstract: Rising energy demand, fossil fuel depletion and climate change highlight the need for more efficient energy production and consumption. Offshore oil and gas platforms face challenges related to inefficient energy use, system failures, accessibility and environmental impact. Machine learning (ML) offers opportunities to improve the safety, sustainability and efficiency of these systems; however, previous research has largely focused on increasing oil production rather than reducing energy consumption on platforms. This study investigates the use of ML and search algorithms to improve diesel efficiency on an offshore oil platform. Data collected over 18 months from a platform in Scotland were analysed, focusing on four diesel generators as the primary diesel-consuming equipment. Following exploratory data analysis and outlier detection, regression models were developed to predict daily diesel consumption for different generator power loa
    
[^163]: 短定价面板中的跨设计不确定性：来自模拟价格轨迹的证据

    Across-Design Uncertainty in Short Pricing Panels: Evidence from Simulated Price Trajectories

    [https://arxiv.org/abs/2608.21334](https://arxiv.org/abs/2608.21334)

    本文通过模拟价格轨迹证明，短定价面板中跨设计不确定性占估计误差方差的绝大部分（97.6%），并提出了一个经验关系式来描述其分散度。

    

    arXiv:2608.21334v1 公告类型：新 摘要：短观测定价面板可能包含许多观测值，但仅提供少量不同的价格变动。本文在一个校准至稀疏定价机制的合成数据生成过程中研究这一区别的推断后果。我们将条件于已实现价格轨迹的不确定性与同一定价过程生成的不同替代轨迹间估计误差的变化分开。在基线模拟中，后者成分占梯度提升规格估计误差方差的97.6%。面板内重采样程序使用一个已实现轨迹的信息，无法识别这一跨设计成分。三个结果组织了分析。首先，跨设计离散度由经验关系 sigma_hat 约等于 0.182 V^(-0.271) 良好描述，其中 V 等于变动次数乘以幅度平方。其次，添加共享区域

    arXiv:2608.21334v1 Announce Type: new  Abstract: Short observational pricing panels can contain many observations while offering only a small number of distinct price movements. This paper studies the inferential consequences of that distinction in a synthetic data-generating process calibrated to a sparse pricing regime. We separate uncertainty conditional on a realised price trajectory from variation in estimation error across alternative trajectories generated by the same pricing process. In the baseline simulations, the latter component accounts for 97.6% of the variance of estimation error for the gradient-boosted specification. Within-panel resampling procedures use the information of one realised trajectory and do not identify this across-design component.   Three results organise the analysis. First, across-design dispersion is well described by the empirical relation sigma_hat approx 0.182 V^(-0.271), where V equals moves times magnitude squared. Second, adding regions sharing
    
[^164]: ClosureBench：一个用于组合图推理的建设性基准

    ClosureBench: A Constructive Benchmark for Compositional Graph Reasoning

    [https://arxiv.org/abs/2608.18242](https://arxiv.org/abs/2608.18242)

    本文提出ClosureBench，一个通过程序化生成实例的建设性图推理基准，能直接测量模型记忆化，并揭示模型在新鲜实例上的性能显著下降。

    

    我们介绍了ClosureBench，这是一个用于组合图关系推理的建设性基准，具有程序化验证的 ground truth。与易受数据污染影响的固定测试集基准不同，ClosureBench按需生成实例：每个任务的参考答案通过执行Ein张量逻辑语言中的程序来计算，确保机器验证的正确性。该基准涵盖三个组合级别（L1-L3）的26个任务类别，难度沿三个独立轴控制：图大小、边密度和查询深度。我们评估了从1.5B开放权重到前沿系统（o3、GPT-4.1、Gemini 2.5、Claude Sonnet 4）的模型，并报告了三个发现。首先，由于该基准总能提供新鲜实例，它直接测量记忆化：一个在固定测试集上微调的模型，在见过的和新鲜实例上的准确率之间存在19.3个百分点的差距，这是静态测试集无法捕捉的。

    arXiv:2608.18242v1 Announce Type: new  Abstract: We introduce ClosureBench, a constructive benchmark for compositional graph-relational reasoning with programmatically verified ground truth. Unlike fixed-test-set benchmarks vulnerable to data contamination, ClosureBench generates instances on demand: each task's reference answer is computed by executing a program in the Ein tensor-logic language, ensuring machine-verified correctness. The benchmark spans 26 task categories at three compositional levels (L1-L3), with difficulty controlled along three independent axes: graph size, edge density, and query depth.   We evaluate models from 1.5B open weights to frontier systems (o3, GPT-4.1, Gemini 2.5, Claude Sonnet 4) and report three findings. First, because the benchmark can always supply fresh instances, it measures memorisation directly: a model fine-tuned on a fixed test set shows a 19.3 percentage-point gap between its accuracy on seen and on fresh instances, which a static test set 
    
[^165]: 面向标签稀疏条件下锂离子电池健康状态估计的退化对齐自监督学习

    Degradation-Aligned Self-Supervised Learning for State of Health Estimation of Lithium-Ion Batteries under Label Sparsity

    [https://arxiv.org/abs/2608.16612](https://arxiv.org/abs/2608.16612)

    提出一种基于排序任务的退化对齐自监督学习框架，利用未标注数据预训练CNN-GRU模型，在标签稀疏条件下实现锂离子电池健康状态的准确稳健估计。

    

    准确的健康状态（SOH）估计是电池系统安全与优化使用的基础。尽管数据驱动的SOH估计模型效果显著，但它们通常需要大量高质量带标签的循环数据，而在实际应用中，这些标签往往在数量和覆盖范围上都很稀疏。因此，在本工作中，我们提出了一种基于卷积神经网络-门控循环单元（CNN-GRU）模型的退化对齐自监督学习（SSL）框架，该框架通过循环顺序排序目标作为预训练的前置任务，从未标注数据中学习与老化一致的表示，从而在稀疏标签数据上微调后实现稳健的SOH估计。测试结果表明，所提出的基于排序的SSL方法能够赋予预训练模型从未标注数据中提取的退化对齐信息，并且在微调后，模型能够进行准确、稳健的SOH估计。

    arXiv:2608.16612v1 Announce Type: cross  Abstract: An accurate estimation of the state of health (SOH) underpins a safe and optimized use of the battery system. Although compelling, data-driven SOH estimation models typically require large amounts of high-quality labeled cycling data, while in practice such labels are often sparse in both quantity and coverage. Therefore, in this work, we propose a degradation-aligned self-supervised learning (SSL) framework based on a convolutional neural network-gated recurrent unit (CNN-GRU) model, which learns aging-consistent representations from unlabeled data through a cycle-order ranking objective as the pretext task for pretraining, thereby enabling robust SOH estimation after fine-tuning on sparsely labeled data. Test results showcase that the proposed ranking-based SSL approach proves to endow the pretrained model with degradation-aligned information from unlabeled data, and after fine-tuning the model can carry out accurate, robust SOH esti
    
[^166]: 通过随机权重平均增强数据增强效果

    Boosting Data Augmentation with Stochastic Weight Averaging

    [https://arxiv.org/abs/2608.14373](https://arxiv.org/abs/2608.14373)

    本研究证明随机权重平均在增强数据上能提供超出其单独性能提升的等变性增强，且无需重复训练成本。

    

    arXiv:2608.14373v1 公告类型：新论文 摘要：学习任务的对称性已成为设计现代深度学习解决方案的重要因素。数据增强是一种将对称性融入通用神经网络的直接且有效的方法。最近的结果表明，在增强数据上训练时，无限大的深度集成表现出完美的对称性。然而，由于训练集成需要多次重复训练过程，这种方法成本高昂。在本研究中，我们研究了随机权重平均（SWA）作为一种替代性集成技术，它不需要重复训练运行。我们通过用奥恩斯坦-乌伦贝克过程近似训练结束时的随机训练轨迹来分析SWA。我们表明，在无限宽度极限下，对增强数据应用SWA提供了超越仅由SWA带来的性能提升所预期的等变性增强。我们通过广泛的数值实验验证了我们的结果。

    arXiv:2608.14373v1 Announce Type: new  Abstract: The symmetries of a learning task have become an important factor in designing modern deep learning solutions. Data augmentation is a straightforward and effective way of incorporating symmetries into a generic neural network. Recent results show that infinitely large deep ensembles show perfect symmetry when trained on augmented data. However, since training ensembles requires repeating the training process many times, this method is costly. In this work, we study stochastic weight averaging (SWA) as an alternative ensembling technique that does not require repeated training runs. We analyze SWA by approximating the stochastic training trajectory at the end of training with an Ornstein--Uhlenbeck process. We show that in the infinite-width limit, SWA on augmented data provides an equiviariance boost that goes beyond what could be expected from the performance increase due to SWA alone. We verify our results with extensive numerical expe
    
[^167]: 终端对称性作为决策资源：用于任意时间验证构建的状态细化

    Terminal Symmetry as a Decision Resource: Statewise Refinement for Anytime Verified Construction

    [https://arxiv.org/abs/2608.11318](https://arxiv.org/abs/2608.11318)

    本文提出了一种将终端对称性作为决策资源的新框架，通过传输-细化-认证机制实现任意时间验证构建，并提供了完成保证和最优验证器查询界限。

    

    arXiv:2608.11318v1 公告类型：交叉 摘要：许多顺序构建任务在完成时表现出精确的对称性，而其执行过程仍是有方向性和历史依赖的。我们提出了一种终端对称性的决策资源视角：过程证据提供方向性，终端对应关系将该结构传输到等价结果上，实现状态证据在转换后细化其当前决策相关性，固定验证器认证执行过程。这种分解产生了传输-细化-认证框架。\method{} 通过情节固定的传输过程结构、其状态限制的过程秩、在接受的转换后刷新的状态依赖残差秩，以及一个序数秩交集（其前$k$集合恰好是两个提议前缀的并集）来实现该原则。该交集在前缀覆盖下提供完成保证，并在相应前缀信息模型下达到最紧的最坏情况验证器查询界限；一个t

    arXiv:2608.11318v1 Announce Type: cross  Abstract: Many sequential construction tasks exhibit exact symmetry at completion while their execution remains directed and history-dependent. We develop a decision-resource view of terminal symmetry: process evidence supplies directionality, terminal correspondence transports that structure across equivalent outcomes, realized-state evidence refines its current decision relevance after transitions, and a fixed verifier certifies execution. This decomposition yields transport--refine--certify. \method{} instantiates the principle with an episode-fixed transported process structure, its state-restricted process rank, a state-dependent residual rank refreshed after accepted transitions, and an ordinal rank meet whose top-$k$ set is exactly the union of the two proposal prefixes. The meet provides a completion guarantee under prefix coverage and attains the tight worst-case verifier-query bound under the corresponding prefix information model; a t
    
[^168]: LEED：用于图神经网络过平滑估计与虚拟节点选择的局部嵌入演化距离

    LEED: Local Embedding Evolution Distance for over-smoothing estimation and virtual node selection in GNN

    [https://arxiv.org/abs/2608.09596](https://arxiv.org/abs/2608.09596)

    提出了一种新的局部度量方法LEED（局部嵌入演化距离），通过追踪单个节点嵌入跨层的演化来量化过平滑现象，实现节点级细粒度分析，并可用于指导图神经网络中虚拟节点的选择。

    

    图神经网络（GNN）受到两个基本限制：过平滑和过度挤压。过平滑是指随着网络深度增加，节点表示变得难以区分；过度挤压是指长距离信息通过有限的消息传递通道被压缩。现有的度量方法（如狄利克雷能量）只能提供过平滑的全局刻画，但缺乏分析节点级行为和指导架构改进所需的分辨率。在本文中，我们提出了LEED（局部嵌入演化距离），这是一种新颖的局部度量方法，通过跟踪单个节点嵌入在各层之间的演化来量化过平滑。通过在节点级别进行度量，LEED能够对训练过程中的表示动态进行细粒度分析，揭示全局基于能量的度量所无法发现的异质过平滑模式。这种局部性产生了具有信息量的节点重要性分数，可被解释为嵌入驱动的中心性（摘要在此处截断）。

    arXiv:2608.09596v2 Announce Type: replace-cross  Abstract: Graph Neural Networks (GNNs) suffer from two fundamental limitations: over-smoothing, where node representations become indistinguishable with depth, and over-squashing, where long-range information is compressed through limited message-passing channels. Existing metrics such as Dirichlet energy provide global characterizations of over-smoothing but lack the resolution to analyze node-level behavior and guide architectural improvements. In this paper, we propose LEED (Local Embedding Evolution Distance), a novel local metric that quantifies over-smoothing by tracking the evolution of individual node embeddings across layers. By operating at the node level, LEED enables fine-grained analysis of representation dynamics during training, revealing heterogeneous over-smoothing patterns that are invisible to global energy-based measures. This locality induces informative node importance scores, interpreted as embedding-driven central
    
[^169]: 潜在事实核查：通过激活工程检测错误信息

    Latent Fact-Checking: Detecting Misinformation through Activation Engineering

    [https://arxiv.org/abs/2608.06417](https://arxiv.org/abs/2608.06417)

    本文提出了一种基于激活工程的错误信息检测框架，利用语言模型表示空间中的几何方向来分类声明真伪，无需微调或外部知识。

    

    arXiv:2608.06417v3 公告类型：交叉替换 摘要：在线错误信息的泛滥推动了对可扩展检测系统的需求。尽管大多数现有方法依赖于表面层面的语言特征或外部知识检索，我们将真实性视为语言模型表示空间中的一种几何属性。我们引入了一个基于激活工程的错误信息检测框架，该框架利用Transformer模型的潜在几何结构。我们的方法通过对比成对真实和虚假陈述的激活，遵循对比激活加法（CAA）的均值差异原理，在残差流中引出错误信息方向。在推理时，将未见声明的最后一个令牌激活投影到该方向上，并将投影表示馈送到多层感知器（MLP）进行分类。该过程无需对骨干模型进行微调，也无需外部证据检索。

    arXiv:2608.06417v3 Announce Type: replace-cross  Abstract: The proliferation of misinformation online has driven demand for scalable detection systems. While most existing approaches rely on surface-level linguistic features or external knowledge retrieval, we examine truthfulness as a geometric property of a language model's representation space. We introduce a misinformation detection framework grounded in activation engineering, which leverages the latent geometry of transformer models. Our approach elicits a misinformation direction in the residual stream by contrasting activations from paired truthful and false statements, following the difference-in-means principle of Contrastive Activation Addition (CAA). At inference time, the last-token activation of an unseen claim is projected onto this direction, and the projected representation is fed to an Multilayer Perceptron (MLP) for classification. The procedure requires no fine-tuning of the backbone model, no external evidence retr
    
[^170]: 符号回归中的深度分治归约（DDRSR）

    Deep Divide-and-Reduce in Symbolic Regression

    [https://arxiv.org/abs/2608.02628](https://arxiv.org/abs/2608.02628)

    该论文提出DDRSR方法，通过对更广泛分解结构的形式化分析，从根本上扩展了AI Feynman方法中表达式分解与归约机制的适用范围，并克服了其依赖暴力搜索的局限。

    

    符号回归（SR）是从数据中发现潜在模式并用数学表达式将其表示出来的任务。当前的机器学习符号回归方法通常缺乏对这些表达式所遵循的内在数学和物理原理的深刻理解。虽然开创性的AI Feynman方法利用了数据背后的数学性质，但其表达式分解机制存在适用范围狭窄的问题，在复杂方程上容易失效。此外，其底层机制严重依赖于对子表达式的暴力搜索，极大地限制了其实用性。在AI Feynman的基础上，我们提出了符号回归中的深度分治归约方法（DDRSR），这是通过对更广泛一类分解结构进行形式化分析而得出的一个有原则性的扩展。DDRSR从根本上拓宽了表达式分解和归约的适用范围……

    arXiv:2608.02628v2 Announce Type: replace-cross  Abstract: Symbolic regression (SR) is the task of discovering underlying patterns from data and representing them using mathematical expressions. Current machine learning approaches to SR often lack a profound understanding of the intrinsic mathematical and physical principles governing these expressions. While the pioneering AI Feynman method leverages the mathematical properties underlying the data, its expression decomposition mechanism suffers from a narrow scope of applicability and is prone to failure on complex equations. Furthermore, its underlying mechanisms rely heavily on brute-force searches for sub-expressions, severely limiting its practical utility. Building on AI Feynman, we propose Deep Divide-and-Reduce in Symbolic Regression (DDRSR), a principled extension derived from a formal analysis of a broader class of decomposition structures. DDRSR fundamentally broadens the applicability of expression decomposition and reducti
    
[^171]: KernelGenBench：一个面向基于大语言模型核函数生成的多源多芯片基准测试

    KernelGenBench: A Multi-Source and Multi-Chip Benchmark for LLM-based Kernel Generation

    [https://arxiv.org/abs/2607.27231](https://arxiv.org/abs/2607.27231)

    该论文提出了KernelGenBench，首个统一的多源多芯片基准，覆盖210个算子和六个硬件平台，用于系统评估大语言模型与智能体生成的Triton核函数在算子来源与硬件平台之间的性能迁移能力。

    

    现代人工智能系统依赖于专用加速器核函数，而日益多样化的算子和硬件使其开发变得复杂。大语言模型（LLM）和智能体系统有望实现这项工作的自动化，但现有评估无法表明其性能能否跨算子来源和硬件平台迁移，以及这种迁移的代价是什么。我们提出了KernelGenBench，这是首个用于评估由大语言模型和智能体生成的Triton核函数的统一多源、多芯片基础设施。通过以统一的Triton目标覆盖六个硬件平台，它在现有核函数生成基准中提供了最广泛的跨厂商硬件覆盖。我们报告了两个受控分析视角：KernelGenBench-MS（多源）涵盖来自PyTorch ATen、生产级vLLM算子和专有cuBLAS例程的210个算子，而KernelGenBench-MC（多芯片）则在六个硬件平台上评估一个语义稳定的110算子子集。

    arXiv:2607.27231v2 Announce Type: replace  Abstract: Modern AI systems depend on specialized accelerator kernels, whose development is complicated by increasingly diverse operators and hardware. LLMs and agentic systems promise to automate this work, but existing evaluations do not show whether their performance transfers across operator sources and hardware platforms, or what such transfer costs. We present KernelGenBench, the first unified multi-source and multi-chip infrastructure for evaluating LLM- and agent-generated Triton kernels. With a common Triton target spanning six hardware platforms, it provides the broadest cross-vendor hardware coverage among existing kernel-generation benchmarks. We report two controlled analytical views: KernelGenBench-MS (Multi-Source) covers 210 operators from PyTorch ATen, production vLLM operators, and proprietary cuBLAS routines, while KernelGenBench-MC (Multi-Chip) evaluates a semantically stable 110-operator subset across six hardware platform
    
[^172]: Aletheia：面向低资源医疗环境的离线优先鉴别诊断临床决策支持系统

    Aletheia: An Offline-First Clinical Decision Support System for Differential Diagnosis in Low-Resource Healthcare Settings

    [https://arxiv.org/abs/2607.24814](https://arxiv.org/abs/2607.24814)

    Aletheia是一个面向撒哈拉以南非洲低资源医疗环境的离线优先临床决策支持系统，通过QLoRA微调Qwen2.5-3B-Instruct模型，在东非高发疾病鉴别诊断中实现了80%的Top-1准确率和100%的Top-3准确率，无需依赖互联网连接和高规格硬件。

    

    在撒哈拉以南非洲地区，专科临床专业知识的获取仍然严重受限，农村地区的医患比例可能低于1:25,000。现有的AI辅助诊断工具主要依赖可靠的互联网连接和高规格硬件，这使其对于地区医院和卫生中心的一线医护人员而言并不实用。本文提出了Aletheia，一个专为撒哈拉以南非洲低资源医疗环境设计的离线优先临床决策支持系统。Aletheia基于Qwen2.5-3B-Instruct构建，采用量化低秩自适应（QLoRA）技术，在涵盖50种东非高发疾病、共27,000个临床推理样本的精选数据集上进行了微调。评估结果显示，其Top-1诊断准确率为80%（10例中的8例；95% CI 49.0-94.3%），Top-3准确率为100%（10例中的10例；95% CI 72.2-100%），BERTScore-F1为0.909。

    arXiv:2607.24814v2 Announce Type: replace  Abstract: Access to specialist clinical expertise remains severely limited across sub-Saharan Africa, where physician-to-patient ratios can fall below 1:25,000 in rural settings. Existing AI-assisted diagnostic tools predominantly require reliable internet connectivity and high-specification hardware, rendering them impractical for frontline healthcare workers in district hospitals and health centres. This paper presents Aletheia, an offline-first clinical decision support system designed for low-resource healthcare contexts across sub-Saharan Africa. Aletheia is built upon Qwen2.5-3B-Instruct, fine-tuned using Quantised Low-Rank Adaptation (QLoRA) on a curated dataset of 27,000 clinical reasoning samples spanning 50 disease conditions with elevated prevalence in East Africa. Evaluation demonstrates a Top-1 diagnostic accuracy of 80% (8 of 10 cases; 95% CI 49.0-94.3%), Top-3 accuracy of 100% (10 of 10; 95% CI 72.2-100%), BERTScore-F1 of 0.909,
    
[^173]: 擦除，还是不擦除：基于保留感知自适应排序子空间扩展的鲁棒免训练概念擦除

    To Erase, or Not to Erase: Robust Training-Free Concept Erasure with Preservation aware Adaptive Ranked Subspace Expansion

    [https://arxiv.org/abs/2607.23492](https://arxiv.org/abs/2607.23492)

    本文提出PARSE框架，一种免训练的概念擦除方法，通过保留感知的自适应排序子空间扩展，解决了现有概念擦除技术中擦除鲁棒性与模型效用之间的权衡问题，实现更鲁棒且不损害良性概念效用的目标擦除。

    

    概念擦除技术（CETs）通过编辑文本到图像扩散模型来擦除不需要的目标（如NSFW内容或受版权保护的风格），同时保持模型对良性概念的效用。当前的CET面临擦除鲁棒性与模型效用之间的权衡：更强的编辑能更可靠地擦除目标，但会降低模型对非目标概念的效用，反之亦然。这源于现有方法定义“擦除什么”和“保留什么”的方式。许多CET依赖于静态概念库，这些概念库通过人工指定、大语言模型（LLM）生成或CLIP图像-文本相似度选择来构建。这样的概念库没有建模提示词在去噪过程中如何引导模型，使得模型容易受到重新引入目标概念的触发攻击，同时抑制附近的良性概念。我们提出了保留感知自适应排序子空间扩展（PARSE），这是一个用于潜在扩散模型中鲁棒概念擦除的免训练框架。给定一个擦除目标，PARSE查询扩散……

    arXiv:2607.23492v2 Announce Type: replace-cross  Abstract: Concept erasure techniques (CETs) edit text-to-image diffusion models to erase undesired targets such as NSFW content or copyrighted styles, while preserving model utility on benign concepts. Current CETs face a trade-off between erasure robustness and utility: stronger edits erase the target more reliably but degrade utility on non-target concepts, and vice versa. This stems from how existing methods define what to erase and what to preserve. Many CETs rely on static concept banks specified manually, generated by LLMs, or selected by CLIP image-text similarity. Such banks do not model how prompts steer the model during denoising, leaving it vulnerable to triggers that reintroduce the target while suppressing nearby benign concepts. We present Preservation-aware Adaptive Ranked Subspace Expansion (PARSE), a training-free framework for robust concept erasure in latent diffusion models. Given a target, PARSE queries the diffusion
    
[^174]: 并非所有大语言模型的推理都体现在思维链中

    Not All LLM Reasoning is Visible in the Chain-of-Thought

    [https://arxiv.org/abs/2607.22925](https://arxiv.org/abs/2607.22925)

    前沿大语言模型能利用语义无关的填充token进行思维链之外的“不可见推理”来提升任务表现，甚至可以完成思维链监控完全无法察觉的隐藏目标，这对基于CoT监控的AI安全方案构成重大风险。

    

    AI安全的一个关键问题是：语言模型是否会在其输出token中表达其全部推理过程。我们展示了一种具体的失败模式：前沿模型通过利用语义无关的填充token来提升在合成推理任务上的表现，从而展现出“不可见推理”。我们在三个任务上评估了13个前沿语言模型，发现许多模型都能从填充token中显著获益，准确率提升最高可达13个百分点。这种收益取决于所使用的token类型，且在不同模型之间存在差异。我们进一步表明，填充token使Claude Opus 4.5能够在不牺牲其主要任务准确率的前提下满足一个隐藏的模运算约束，这证明了不可见推理可以服务于思维链监控完全无法察觉的目标。强化学习使Qwen3-235B对填充token的内容产生了强烈偏好，但无论是强化学习还是监督微调（原文在此处截断）

    arXiv:2607.22925v2 Announce Type: replace-cross  Abstract: A key question for AI safety is whether a language model expresses all of its reasoning in its output tokens. We demonstrate a concrete failure mode where frontier models exhibit invisible reasoning by leveraging semantically irrelevant filler tokens to improve performance on synthetic reasoning tasks. We evaluate 13 frontier language models across three tasks and find that many models benefit significantly from filler tokens, with accuracy improvements of up to 13 percentage points. The benefit depends on which tokens are used and differs across models. We further show that filler tokens enable Claude Opus 4.5 to satisfy a hidden modular arithmetic constraint without sacrificing accuracy on its primary task, demonstrating that invisible reasoning can serve objectives entirely invisible to CoT monitoring. Reinforcement learning gives Qwen3-235B strong preferences over filler token content, but neither RL nor supervised fine-tun
    
[^175]: 平滑加性模型中的自动节点选择

    Automatic knot selection in smooth additive models

    [https://arxiv.org/abs/2607.21083](https://arxiv.org/abs/2607.21083)

    本文研究B样条回归中的节点自动选择问题，指出尽管P样条正则化已成为广义加性模型的标准，但常被忽视的节点选择技术具有独特的优势。

    

    B样条回归是非参数建模中广泛使用的框架。该方法的性能取决于在估计过程之前指定变点（即所谓的节点）的数量和位置。这样的节点序列决定了用于表示回归函数的B样条基的维度以及需要估计的系数数量。因此，节点的选择会影响模型的灵活性，进而影响其平滑度和拟合优度。传统上，这个问题要么通过节点选择算法显式选择节点来解决，要么通过正则化方法（如P样条）来解决，后者可以自动调整回归器的平滑度。后者已成为广义加性模型（GAMs）中的标准方法。相比之下，由于计算或建模限制而经常被忽视的节点选择技术，可以提供某些优势……

    arXiv:2607.21083v2 Announce Type: replace-cross  Abstract: B-spline regression constitutes a widely used framework for nonparametric modeling. The performance of this methodology depends on specifying the number and placement of changepoints, known as knots, prior to the estimation process. Such knot sequence determines the dimension of the B-spline basis used to represent the regression function and the number of coefficients to be estimated. Therefore, the knots' choice affects the model's flexibility, influencing its smoothness and goodness-of-fit. Traditionally, this problem has been addressed either by explicitly selecting knots, via knot-selection algorithms, or by regularization methods, such as P-splines, which automatically tune the regressor's smoothness. The latter have become the standard in generalized additive models (GAMs). In contrast, knot-selection techniques, frequently neglected because of computational or modeling limitations, provide certain advantages which can b
    
[^176]: 四量子比特ZZ量子核的态矢量到硬件重构：三个执行任务的单一后端案例研究

    Statevector-to-Hardware Reconstruction of a Four-Qubit ZZ Quantum Kernel: A Single-Backend Case Study of Three Execution Jobs

    [https://arxiv.org/abs/2607.20377](https://arxiv.org/abs/2607.20377)

    该研究在单一后端上通过基线、动态解耦和门扭转三个独立执行任务，量化了四量子比特ZZ量子核格拉姆矩阵从精确态矢量参考到硬件执行的偏差，发现门扭转技术在所有报告的几何指标上偏离最小。

    

    硬件噪声与有限采样会扰动构成量子核格拉姆（Gram）矩阵的保真度估计。我们测量了在N=24个室内空气质量时间窗口上，针对一个固定的四量子比特ZZ特征映射，三个硬件重构的格拉姆矩阵与精确态矢量参考之间的偏离程度。实验在ibm_fez后端上以每电路1024次采样执行，分为三个单独、非交错的作业：基线、仅动态解耦和仅门扭转。所有矩阵均为完整、有限且半正定的。相对于参考的非对角均方根误差（RMSE）分别为0.0878、0.0864和0.0427；全矩阵中心化核对齐（CKA）范围为0.933-0.989，事后对角排除（U中心化）CKA范围为0.816-0.986。门扭转作业在每个报告的几何轴上偏离最小；其在Spearman相关、平均绝对误差、RMSE和全矩阵CKA诊断方面与基线的对比具有删除稳定性，而Pearson和对角排除（摘要原文在此处截断）。

    arXiv:2607.20377v2 Announce Type: replace-cross  Abstract: Hardware noise and finite sampling perturb the fidelity estimates forming a quantum-kernel Gram matrix. We measured how far three hardware-reconstructed Gram matrices depart from an exact statevector reference for one frozen four-qubit ZZ feature map on N=24 indoor air-quality windows, executed on ibm_fez at 1024 shots per circuit in three single, non-interleaved jobs: baseline, dynamical decoupling alone, and gate twirling alone. All were complete, finite, and positive-semidefinite. Off-diagonal root-mean-squared error (RMSE) against the reference was 0.0878, 0.0864, and 0.0427; full-matrix centered kernel alignment (CKA) ranged 0.933-0.989 and the post hoc diagonal-excluded (U-centered) CKA 0.816-0.986. The gate-twirled job deviated least on every reported geometry axis; its baseline contrasts are deletion-stable for the Spearman, mean-absolute-error, RMSE, and full-matrix CKA diagnostics, while the Pearson and diagonal-exclu
    
[^177]: 面向工业级规模LLM推荐系统的带质量护栏的高效聚类方法

    Efficient Clustering with Quality Guardrails for LLM-based Recommender Systems at Industry Scale

    [https://arxiv.org/abs/2607.19704](https://arxiv.org/abs/2607.19704)

    该论文提出一种带单样本质量护栏的高效聚类方法，使LLM推荐系统能在工业规模下仅对簇代表运行LLM即可大幅降低成本，同时确保每个簇成员获得的输出既与自身相关又安全。

    

    大语言模型在大规模运行时可能成本极高且速度缓慢，尤其是对于那些需要在数百万输入上对每个样本调用一次LLM的应用。一种自然的扩展方式是对输入进行聚类，仅对簇代表运行LLM，然后将输出传播给簇内的其他成员。然而，成员所获得的输出质量完全取决于其与代表的匹配程度。现成的聚类方法优化的是聚合目标，只能保证平均情况下的质量，而缺乏针对单个样本的质量护栏。因此，成员可能被分配给匹配度差/不良的代表，所继承的输出虽然对代表而言是合适的，但对成员来说可能不相关，甚至不安全。例如，一个幼儿的家长如果与年龄较大孩子的家长被分到同一簇中，可能会收到不适合其年龄的推荐内容。此外，大多数聚类方法在运行时间和内存上难以扩展到数百万输入，限制了其在工业界的应用。

    arXiv:2607.19704v2 Announce Type: replace  Abstract: LLMs can be prohibitively expensive and slow to run at scale, especially for applications that invoke an LLM per sample over millions of inputs. A natural way to scale is to cluster the inputs, run the LLM only on cluster representatives, and propagate the outputs to other cluster members. However, the outputs a member receives are only as good as its match to the representative. Off-the-shelf clustering methods optimize an aggregate objective, targeting average-case quality without per-sample guardrails. As a result, members can be assigned to poorly-matched representatives, and the inherited outputs -- though appropriate for the representative -- may be irrelevant or even unsafe for the member. For example, a parent of a toddler grouped with parents of older children could receive age-inappropriate recommendations. Most clustering methods also scale poorly to millions of inputs in runtime and memory, limiting their use at industry 
    
[^178]: 改进 Atari Pong 中强智能体背后的弱世界模型

    Improving Weak World Models Behind Strong Agents in Atari Pong

    [https://arxiv.org/abs/2607.15142](https://arxiv.org/abs/2607.15142)

    该论文在 Atari Pong 中复现五个视觉世界模型智能体并独立评估其冻结世界模型，揭示出强智能体背后普遍存在球消失、错误运动等缺陷的弱世界模型，并提出了改进这些弱世界模型的方法。

    

    arXiv:2607.15142v3 公告类型：替换 摘要：强世界模型智能体中常常包含弱世界模型。我们通过在 Atari Pong 中复现五个视觉世界模型智能体来研究这种智能体与世界模型之间的差距：DreamerV3、DIAMOND、TWISTER、Simulus 和 STORM，其性能与已报道的结果相当，并对它们的冻结世界模型进行独立评估。首先，闭环 rollout 诊断在独立训练的策略下对每个冻结模型生成的视觉轨迹进行定性检查。所有五个模型都表现出明显的视觉或动力学缺陷，包括球消失、错误运动以及无效的球拍-球交互。其次，在原生零样本基于模型的强化学习（MBRL）设置下，使用智能体原生的 RL 流程完全在冻结模型内部从头训练新策略，无需真实环境训练。当在真实环境中评估时，这些策略的表现显著低于复现的智能体：DreamerV3

    arXiv:2607.15142v3 Announce Type: replace  Abstract: Strong world-model agents frequently contain weak world models. We study this agent-world-model gap by reproducing five visual world-model agents in Atari Pong: DreamerV3, DIAMOND, TWISTER, Simulus, and STORM, with performance comparable to the reported results, and independently evaluating their frozen world models. First, closed-loop rollout diagnosis qualitatively inspects visual trajectories generated by each frozen model under an independently trained policy. All five models exhibit clear visual or dynamical failures, including ball disappearance, incorrect motion, and invalid ball-paddle interactions. Second, under native zero-shot model-based reinforcement learning (MBRL), a new policy is trained entirely within the frozen model from scratch using the agent's native RL procedure, without real-environment training. When evaluated in the real environment, these policies substantially underperform the reproduced agents: DreamerV3
    
[^179]: EvoCUA-1.5：面向多轮计算机使用智能体的在线强化学习

    EvoCUA-1.5: Online Reinforcement Learning for Multi-turn Computer-Use Agents

    [https://arxiv.org/abs/2607.09773](https://arxiv.org/abs/2607.09773)

    EvoCUA-1.5 将计算机使用智能体从离线经验学习扩展到在线强化学习，并提出步级策略优化（STEPO）方法，以解决多轮交互中上下文管理观察、稀疏终端奖励、可变长度轨迹和慢速环境反馈等挑战。

    

    计算机使用智能体必须通过与部分可观察的多模态桌面环境进行反复交互来解决长时程任务。尽管模仿学习和离线轨迹优化能够提供强大的先验知识，但静态轨迹无法覆盖真实计算机使用中的因果反馈循环：每个动作都会改变屏幕状态、未来的动作空间以及恢复选项。EvoCUA-1.5 将自我进化的计算机使用智能体从离线经验学习扩展到在线强化学习，其中策略与可执行的沙盒环境进行交互，并从可验证的任务结果中获得改进。在这种设置下，在线强化学习并非简单复用单轮语言强化学习的方法即可奏效。多轮交互引入了上下文管理的观察、稀疏的终端奖励、可变长度的轨迹以及缓慢的环境反馈等挑战。EvoCUA-1.5 通过步级策略优化（STEPO）方法来应对这些挑战，该方法保留了……（摘要在此处截断）

    arXiv:2607.09773v2 Announce Type: replace  Abstract: Computer-use agents must solve long-horizon tasks through repeated interaction with partially observable, multimodal desktop environments. Although imitation learning and offline trajectory refinement provide strong priors, static traces cannot cover the causal feedback loop of real computer use: each action changes the screen state, future action space, and recovery options. EvoCUA-1.5 extends self-evolving computer-use agents from offline experience learning to online reinforcement learning, where policies interact with executable sandbox environments and improve from verifiable task outcomes. Online RL in this setting requires more than directly reusing single-turn language-RL recipes. Multi-turn interaction introduces context-managed observations, sparse terminal rewards, variable-length trajectories, and slow environment feedback. EvoCUA-1.5 addresses these challenges with Step-Level Policy Optimization (STEPO), which preserves 
    
[^180]: 连续酉值映射的量子 Kolmogorov--Arnold 表示定理

    Quantum Kolmogorov--Arnold representation theorem for continuous unitary-valued maps

    [https://arxiv.org/abs/2607.03187](https://arxiv.org/abs/2607.03187)

    本文为连续酉值映射建立了 Kolmogorov--Arnold 表示定理的两个量子类比版本，分别给出了反厄米值映射矩阵指数内的精确加法分解，以及考虑量子算符非对易性的因式分解形式。

    

    arXiv:2607.03187v2 公告类型：replace-cross 摘要：经典的 Kolmogorov--Arnold 表示定理指出，任意连续多元函数都可以精确地分解为有限个一元连续函数与加法运算的复合。这一基础性成果近期启发了经典机器学习中 Kolmogorov--Arnold 网络（KANs）的发展，以及其向量子领域（QKANs）的扩展。在本文中，我们在单位矩阵的开 1-邻域 \(O_1(\mathbf{I}) \subset \mathcal{U}(n)\) 内，针对多变量连续酉值映射建立了 Kolmogorov--Arnold 表示定理的两个量子类比。首先，我们证明了一个表示定理，在反厄米值映射的矩阵指数内部给出了精确的加法分解。其次，鉴于量子算符的非对易性质，我们推导出了一个因式分解版本，用以表示目标酉矩阵（原文摘要在此处截断）

    arXiv:2607.03187v2 Announce Type: replace-cross  Abstract: The classical Kolmogorov--Arnold representation theorem states that any continuous multivariate function can be exactly decomposed into a finite composition of univariate continuous functions and addition operations.   This foundational result has recently inspired the development of Kolmogorov--Arnold Networks (KANs) in classical machine learning, as well as their extensions into the quantum domain (QKANs). In this paper, we establish two quantum analogues of the Kolmogorov--Arnold representation theorem for continuous unitary-valued maps of several variables within an open $1$-neighbourhood of the identity matrix \(O_1(\mathbf{I}) \subset \mathcal{U}(n)\).   First, we prove a representation theorem that yields an exact additive decomposition inside the matrix exponent of anti-Hermitian-valued maps.   Second, due to the non-commutative nature of quantum operators, we derive a factorised version expressing the target unitary ma
    
[^181]: 从架构到输出：大语言模型幻觉的结构性起源与数据的放大作用

    From Architecture to Output: Structural Origins of Hallucination in Large Language Models and the Amplifying Role of Data

    [https://arxiv.org/abs/2606.07537](https://arxiv.org/abs/2606.07537)

    该论文提出了一个仅需采样访问权限的幻觉归因框架，通过针对前缀、上下文和频率竞争的三次有序干预，将大语言模型的单个幻觉追溯归因到自注意力联想检索、最大似然预训练目标或暴露偏差下的自回归承诺等具体组件，并提出五个可证伪的预测。

    

    大语言模型会生成流畅、自信但事实上错误的输出。现有的分类体系仅按输出类型对这些失败进行分类——内在性还是外在性、忠实性还是事实性——却从未说明是哪个计算组件产生了特定的失败。我们探讨需要什么条件才能将单个幻觉归因于纯解码器（decoder-only）架构栈中的特定组件。我们将三个组件——自注意力的联想检索、最大似然预训练目标、以及暴露偏差（exposure bias）下的自回归承诺——视为候选失败面，论证而非假定它们的可分离性，并规定了一种仅需采样访问权限的归因程序：一组针对前缀、上下文和频率竞争的有序三步干预，同时辅以基于独立标注和分类器基线的验证设计。我们提出了五个可证伪的预测……

    arXiv:2606.07537v2 Announce Type: replace-cross  Abstract: Large language models produce fluent, confident, factually wrong output. Existing taxonomies classify these failures by output type -- intrinsic versus extrinsic, faithfulness versus factuality -- but say nothing about which computational component produced a given failure. We ask what would be required to attribute an individual hallucination to a specific component of the decoder-only stack. We treat three components -- self-attention's associative retrieval, the maximum-likelihood pretraining objective, and autoregressive commitment under exposure bias -- as candidate failure surfaces, justify their separability rather than assuming it, and specify an attribution procedure requiring only sampling access: an ordered set of three interventions on prefix, context, and frequency competition, together with a validation design based on independent annotation and a classifier baseline. We state five falsifiable predictions and iden
    
[^182]: 从采样结果到能力分布：重新思考大语言模型路由的监督信号

    From Sampled Outcomes to Capability Distributions: Rethinking Supervision for LLM Routing

    [https://arxiv.org/abs/2606.06924](https://arxiv.org/abs/2606.06924)

    该论文提出DARS方法，通过语义保持的查询改写和随机解码的重复观测来估计查询级模型能力分布，构建风险感知的路由监督信号，解决了现有LLM路由中单次采样监督不稳定的问题。

    

    现有的LLM路由方法通常针对每个查询-模型对仅使用一次采样的响应来构建监督信号。然而，由于LLM的生成过程具有随机性，这样的观测可能成为模型能力的不稳定估计：语义等价的查询表述和重复解码可能产生不同的分数，甚至导致不同的模型偏好。我们证明了这种不稳定性会进一步从路由标签传播到学习到的路由策略中。为了解决这一问题，我们提出了DARS（分布感知路由监督），它通过跨越语义保持查询改写和随机解码的重复观测来估计查询级别的模型能力。DARS汇总期望质量、期望成本和性能变异性，以构建风险感知的监督信号，而无需改变下游路由器的架构。在多种任务和路由方法上的实验表明，DARS通常能够改进路由性能。

    arXiv:2606.06924v2 Announce Type: replace  Abstract: Existing LLM routing methods often construct supervision from a single sampled response for each query--model pair. Because LLM generation is stochastic, however, such an observation can be an unstable estimate of model capability: semantically equivalent query formulations and repeated decoding may yield different scores and even different model preferences. We show that this instability can further propagate from routing labels to learned routing policies. To address this issue, we propose DARS (Distribution-Aware Routing Supervision), which estimates query-level model capability from repeated observations spanning semantics-preserving query rewrites and stochastic decoding. DARS summarizes expected quality, expected cost, and performance variability to construct risk-aware supervision without changing the downstream router architecture. Experiments across diverse tasks and routing methods show that DARS generally improves routing 
    
[^183]: 通过随机雅可比匹配学习混沌动力学的二阶一致性

    Second-order consistency for learning chaotic dynamics via randomized Jacobian matching

    [https://arxiv.org/abs/2606.01596](https://arxiv.org/abs/2606.01596)

    提出模型约束的随机雅可比匹配方法，通过在随机扰动输入处比较雅可比，隐式强制二阶（Hessian）一致性，避免学习到的混沌系统漂移向虚假吸引子并改善长时间统计特性。

    

    短时程精度并不能保证学习到的混沌系统具有正确的长时间动力学行为。轨迹（零阶）匹配约束向量场的取值，雅可比（一阶）匹配约束局部切向动力学，但两者都无法确定雅可比在远离监督状态处如何变化，因此模型可能在局部准确的同时漂移向虚假吸引子，并扭曲长时间的统计特性。我们证明二阶监督可以缓解这些失败。由于在高维空间中构建完整的Hessian张量在计算上过于昂贵，我们提出了模型约束的随机雅可比匹配方法，该方法在随机扰动的输入处比较真实向量场与学习到的向量场的雅可比。泰勒展开表明，随机雅可比损失的期望可分解为雅可比失配项加上按噪声方差缩放的Hessian失配项，从而隐式地强制执行二阶一致性。

    arXiv:2606.01596v2 Announce Type: replace-cross  Abstract: Short-horizon accuracy does not ensure that a learned chaotic system has correct long-time dynamics. Trajectory (zeroth-order) matching constrains vector-field values, and Jacobian (first-order) matching constrains local tangent dynamics, but neither determines how the Jacobian varies away from supervised states, so a model can be locally accurate while drifting toward spurious attractors and distorting long-time statistics. We show that second-order supervision mitigates these failures. Because forming full Hessian tensors is computationally prohibitive in high dimensions, we propose model-constrained randomized Jacobian matching, which compares the Jacobians of the true and learned vector fields at randomly perturbed inputs. A Taylor expansion shows that the expected randomized Jacobian loss decomposes into the Jacobian mismatch plus a Hessian mismatch scaled by the noise variance, implicitly enforcing second-order consistenc
    
[^184]: 无害却有害：针对智能体技能中隐蔽幻觉引导的中性提示攻击

    Harmless Yet Harmful: Neutral Prompting Attacks for Stealthy Hallucination Steering in Agent Skills

    [https://arxiv.org/abs/2605.29354](https://arxiv.org/abs/2605.29354)

    提出了一种高度隐蔽的“中性提示攻击”（NPA），通过鼓励想象力和穷尽性等语义无害的指令，悄然提升LLM编程智能体产生软件包幻觉的倾向，从而实现无需显式恶意意图的隐蔽软件供应链攻击。

    

    arXiv:2605.29354v2 通知类型：replace-cross 摘要：基于大语言模型（LLM）的编程智能体越来越多地参与软件开发工作流，包括生成代码、选择依赖项以及生成软件包安装命令。这带来了一种新的软件供应链风险：当智能体幻觉出一个不存在的软件包时，攻击者可以注册该幻觉出来的名称，进而危害后续安装该软件包的用户。现有的软件包幻觉攻击与防御主要聚焦于自然发生的幻觉、有针对性的依赖引导或事后软件包验证。在本文中，我们提出了“中性提示攻击”（Neutral Prompting Attack, NPA），这是一种高度隐蔽的攻击范式：语义上完全无害的指令（例如鼓励想象力和穷尽性）能够在不包含任何显式恶意意图的情况下，提高模型产生软件包幻觉的倾向。与有针对性的依赖引导不同，NPA并不指定攻击者选定的软件包，而是改变模型的依赖……（原文摘要在此处截断）

    arXiv:2605.29354v2 Announce Type: replace-cross  Abstract: LLM-powered coding agents increasingly participate in software development workflows by generating code, selecting dependencies, and producing package installation commands. This creates a new software supply chain risk: when an agent hallucinates a non-existent package, an attacker may register the hallucinated name and later compromise users who install it. Existing package hallucination attacks and defenses primarily focus on naturally occurring hallucinations, targeted dependency steering, or post-hoc package validation. In this paper, we introduce \emph{Neutral Prompting Attack} (NPA), a highly stealthy attack paradigm in which semantically benign instructions, such as encouraging imagination and exhaustiveness, increase package hallucination propensity without containing explicit malicious intent. Unlike targeted dependency steering, NPA does not specify an attacker-chosen package. Instead, it shifts the model's dependenc
    
[^185]: 基于潜在推理的鲁棒高效安全护栏

    Robust and Efficient Guardrails with Latent Reasoning

    [https://arxiv.org/abs/2605.29068](https://arxiv.org/abs/2605.29068)

    COLAGUARD通过分阶段训练将多步安全推理压缩进连续潜在空间，推理时直接传播隐状态，在宏F1上超越Llama Guard 3达8.24分，并在达到显式推理基线GuardReasoner同等性能的同时实现12.9倍加速和22.4倍的token开销降低。

    

    随着大语言模型（LLM）越来越多地部署于现实世界的应用中，保障其安全性至关重要。现有的安全护栏通常依赖于单次分类，或近期出现的蒸馏推理方法。基于推理的护栏在性能上显著优于仅分类的基线方法，但其会带来较高的查询延迟和token开销，使其难以适用于高吞吐量部署场景。为应对这一挑战，我们提出了COLAGUARD，这是一种安全护栏模型，通过分阶段训练课程将多步安全推理迁移至连续潜在空间中，从而在推理时实现直接的隐状态传播。在涵盖八个安全基准的十个提示与响应审核设置上进行的评估中，COLAGUARD的宏F1分数比Llama Guard 3提高了8.24个点，并在宏F1上与我们的显式推理基线GuardReasoner持平，同时实现了12.9倍的加速和22.4倍的（token开销降低）。（注：原文摘要在此处截断）

    arXiv:2605.29068v2 Announce Type: replace  Abstract: Maintaining the safety of large language models (LLMs) is crucial as they are increasingly deployed in real-world applications. Existing safety guardrails typically rely on single-pass classification or, more recently, distilled reasoning. Reasoning-based guardrails significantly outperform classification-only baselines, but they incur substantial query latency and token overhead that make them impractical for highthroughput deployment. To address this challenge, we propose COLAGUARD, a guardrail model that transfers multi-step safety reasoning into a continuous latent space through a stage-wise training curriculum, enabling direct hidden-state propagation at inference. Evaluated on ten prompt- and response-moderation settings spanning eight safety benchmarks, COLAGUARD improves macro-F1 by 8.24 points over Llama Guard 3 and matches our explicit reasoning baseline, GuardReasoner, in macroF1 while delivering a 12.9X speedup and 22.4X 
    
[^186]: 强化学习中的最优数据获取：大偏差视角

    Optimal Data Acquisition for Reinforcement Learning: A Large Deviations Perspective

    [https://arxiv.org/abs/2605.28675](https://arxiv.org/abs/2605.28675)

    本文从大偏差理论出发，为无限时域强化学习建立了统一的数据获取框架，以策略选择错误概率的指数衰减率作为效率度量，并通过凸松弛和惰性单步投影次梯度方法给出了可求解的最优数据获取方案。

    

    数据获取效率是在业务和医疗运营中部署强化学习的核心挑战，在这些场景中，交互成本高昂、速度缓慢，且往往需要人工参与。本文为无限时域强化学习中的数据获取建立了一个统一的大偏差框架。我们引入策略选择错误概率的指数衰减率作为有理论依据的效率度量，并借助马尔可夫链的大偏差理论推导出该衰减率的变分刻画，从而得到一个嵌套优化问题。基于这一刻画，我们依据嵌套问题的最优解形式化了两种互补的最优性概念。由于所得到的规划问题是隐式的且通常难以求解，我们提出了一个具有显式约束、易于处理的凸松弛方法。随后，我们开发了一种惰性单步投影次梯度方法来求解该松弛问题

    arXiv:2605.28675v2 Announce Type: replace  Abstract: Data acquisition efficiency is a central challenge in deploying reinforcement learning in business and healthcare operations, where interactions are costly, slow, and often involve humans in the loop. This paper develops a unified large deviations framework for data acquisition in infinite-horizon reinforcement learning. We introduce the exponential decay rate of the policy-selection error probability as a principled efficiency metric and derive a variational characterization of this rate via large deviations theory for Markov chains, yielding a nested optimization problem. Based on this characterization, we formalize two complementary notions of optimality in terms of the optimal solution of the nested problem. Because the resulting program is implicit and generally intractable, we propose a tractable convex relaxation with explicit constraints. We then develop a lazy one-step projected subgradient method to solve the relaxed proble
    
[^187]: 超越成对偏好：面向扩散模型的列表级奖励感知对齐

    Beyond Pairwise Preferences: Listwise Reward-Aware Alignment for Diffusion Models

    [https://arxiv.org/abs/2605.26491](https://arxiv.org/abs/2605.26491)

    提出 Diffusion LAIR，一种奖励感知的列表级偏好优化方法，将同一提示词下多张候选图像的奖励分数转化为中心化优势权重，突破成对比较的局限，更充分地利用连续奖励信号来对齐文本到图像扩散模型。

    

    偏好优化已成为从人类反馈的在线强化学习之外的一种高效替代方案，用于对齐文本到图像的扩散模型。然而，现有方法在很大程度上将监督信息简化为二元的成对比较。当训练数据自然地包含同一提示词的多张候选图像，且连续的奖励分数能够提供比单一胜负标签更丰富的信息时，这种成对化简便显得力不从心。为了解决这些局限，我们提出了 Diffusion LAIR，一种面向扩散模型的奖励感知列表级偏好优化方法。对于每个提示词，LAIR 将一组候选图像上的奖励分数转换为中心化的优势权重，然后在隐式奖励上优化一个优势加权回归目标——该隐式奖励定义为当前模型相对于固定参考模型在去噪损失上的改进——并采用二次惩罚来正则化模型更新的幅度。

    arXiv:2605.26491v2 Announce Type: replace  Abstract: Preference optimization has emerged as an efficient alternative to online reinforcement learning from human feedback (RLHF) for aligning text-to-image diffusion models. However, existing methods largely reduce supervision to binary pairwise comparisons. This pairwise reduction is limiting when training data naturally contains multiple candidate images for the same prompt, and when continuous reward scores can provide richer information than a single winner-loser label. To address these limitations, we propose Diffusion LAIR, a reward-aware listwise preference optimization method for diffusion models. For each prompt, LAIR converts reward scores across a group of candidate images into centered advantage weights, then optimizes an advantage-weighted regression objective on the implicit reward, defined as the denoising-loss improvement of the current model over a fixed reference model, with a quadratic penalty that regularizes the magni
    
[^188]: SCRIPT：面向语言驱动的物理仿真人形机器人控制的可扩展扩散策略与多阶段训练

    SCRIPT: Scalable Diffusion Policy with Multi-stage Training for Language-driven Physics-Based Humanoid Control

    [https://arxiv.org/abs/2605.22894](https://arxiv.org/abs/2605.22894)

    提出SCRIPT——一个多阶段训练的可扩展扩散策略，其核心JAST-DiT通过联合注意力直接耦合动作、物理状态与文本，实现了基于自然语言指令对物理仿真人形机器人忠实、高质量且稳定的长时程控制。

    

    基于自然语言指令控制物理仿真人形机器人是实现通用具身智能体的重要一步。然而，现有方法仍受限于语义表达性与物理可行性之间的矛盾，往往无法同时实现忠实的指令遵循、高质量的运动以及稳定的长时程控制。我们提出了SCRIPT，一种面向语言驱动的物理仿真人形控制的可扩展扩散策略及其多阶段训练框架。SCRIPT的核心是联合动作-状态-文本扩散Transformer（JAST-DiT），它将动作、物理状态和文本表示为专门的token流，并通过联合注意力机制将它们耦合起来，实现语言语义与控制动力学之间的直接交互。为稳定自回归控制，我们引入了一种非线性历史条件机制，该机制保留密集的近期上下文，并对逐渐久远的历史进行递进式的……

    arXiv:2605.22894v3 Announce Type: replace-cross  Abstract: Controlling physics-based humanoids from natural-language instructions is a critical step toward general-purpose embodied agents. However, existing methods remain constrained by a tension between semantic expressiveness and physical feasibility, often failing to jointly achieve faithful instruction following, high-quality motion, and stable long-horizon control. We propose SCRIPT, a scalable diffusion policy with a multi-stage training framework for language-driven physics-based humanoid control. The core of SCRIPT is a Joint Action-State-Text Diffusion Transformer (JAST-DiT), which represents actions, physical states, and text as dedicated token streams and couples them through joint attention, enabling direct interaction between language semantics and control dynamics. To stabilize autoregressive control, we introduce a nonlinear history conditioning mechanism, which preserves the dense recent context and samples increasingly
    
[^189]: 更少数据，更快训练：通过采样偏差重复训练较小数据集可加速学习

    Less Data, Faster Training: repeating smaller datasets speeds up learning via sampling biases

    [https://arxiv.org/abs/2605.20314](https://arxiv.org/abs/2605.20314)

    该论文发现重复训练较小数据集可通过采样偏差促进恰当的逐层参数增长，从而节省计算并加速学习，这一策略可作为优化中的有利归纳偏置，尤其在推理任务中效果显著。

    

    这项工作研究了“小数据与大数据之间的差距”现象，即在较少样本上进行重复训练，相比使用更大数据集，能够在训练过程中节省计算资源。这一现象在各种算法任务、网络架构和优化器中均可观察到，且无法用先前的理论来解释。我们认为，这种加速来自于采样偏差所促成的恰当的逐层增长，当数据集规模越小时，这一效应越为明显。我们提供了理论分析以及来自多种干预实验的实证证据。我们的结果表明，使用较小数据集并进行更多次重复训练不仅仅是数据稀缺时的后备策略，还可以被主动地用作有利于优化的归纳偏置，尤其是在推理任务中。

    arXiv:2605.20314v2 Announce Type: replace-cross  Abstract: This work investigates the ``small-vs-large gap'', where repeating on fewer samples can lead to compute saving during training compared to using a larger dataset. This is observed across algorithmic tasks, architectures and optimizers and cannot be explained using prior theory. We argue that the speedup comes from appropriate layer-wise growth enabled by sampling biases, which is more pronounced when the dataset size is smaller. We provide both theoretical analysis and empirical evidence from various interventions. Our results suggest that using a smaller dataset with more repetitions is not just a fallback strategy under data scarcity, but can be proactively leveraged as a favorable inductive biases for optimization, particularly in reasoning tasks.
    
[^190]: 深度学习即神经低度滤波：分层特征学习的谱理论

    Deep Learning as Neural Low-Degree Filtering: A Spectral Theory of Hierarchical Feature Learning

    [https://arxiv.org/abs/2605.13612](https://arxiv.org/abs/2605.13612)

    本文提出“神经低度滤波”这一谱理论框架，将深度网络的分层特征学习刻画为逐层进行的显式迭代谱滤波过程，为懒惰机制之外的多层特征学习提供了可解析的数学工具，能够预测逐层表示选择、概念涌现所需的样本复杂度以及深度带来的具体收益机制。

    

    理解深度神经网络如何从数据中学习有用的内部表示，仍然是深度学习理论中的一个核心开放问题。我们提出了神经低度滤波，这是基于梯度的训练的一种理想化极限，在该极限下，分层特征学习成为一个显式的迭代谱过程。在这一极限中，各层的动力学相互解耦：给定当前表示，下一层会选择与标签具有最大可及低度相关性的方向。这为深度学习提供了一个易于处理的替代机制，并伴随一个自然的核空间解释。Neural LoFi 为研究懒惰机制之外的多层特征学习提供了一个数学上显式的框架。它预测了表示如何逐层被选择，解释了概念的出现如何在给定样本复杂度下涌现，并给出了深度逐步发挥作用的（原文在此处截断）

    arXiv:2605.13612v2 Announce Type: replace  Abstract: Understanding how deep neural networks learn useful internal representations from data remains a central open problem in the theory of deep learning. We introduce Neural Low-Degree Filtering (Neural LoFi), a stylized limit of gradient-based training in which hierarchical feature learning becomes an explicit iterative spectral procedure. In this limit, the dynamics at each layer decouple: given the current representation, the next layer selects directions with maximal accessible low-degree correlation to the label. This yields a tractable surrogate mechanism for deep learning, together with a natural kernel-space interpretation. Neural LoFi provides a mathematically explicit framework for studying multi-layer feature learning beyond the lazy regime. It predicts how representations are selected layer by layer, explains how emergence of concepts arises with given sample complexity, and gives a concrete mechanism by which depth progressi
    
[^191]: HLS-Seek：基于代理比较奖励强化学习的高层次综合QoR感知代码生成

    HLS-Seek: QoR-Aware Code Generation for High-Level Synthesis via Proxy Comparative Reward Reinforcement Learning

    [https://arxiv.org/abs/2605.13536](https://arxiv.org/abs/2605.13536)

    提出HLS-Seek框架，利用帕累托支配准确率达99.53%的比较式代理奖励模型，结合不确定性感知的MC dropout切换机制按需调用真实Vitis HLS综合并在线更新代理，从而在无需完整综合在环的情况下实现QoR感知的自然语言到HLS代码生成。

    

    高层次综合（HLS）将算法级的C/C++描述编译为硬件，其质量结果（QoR）——即延迟和资源利用率——在很大程度上由pragma配置和代码结构决定。现有的从自然语言到HLS（NL-to-HLS）的训练方法优先考虑功能正确性，而在很大程度上忽略了QoR。我们观察到，用于HLS的强化学习（RL）并不需要绝对的综合结果——只需要候选方案之间的相对比较。基于这一洞察，我们提出了HLS-Seek，这是一个QoR感知的NL-to-HLS框架，它通过一个在帕累托支配判断上达到99.53%准确率的比较式代理奖励模型，避免了完全依赖综合在环的强化学习。为了防止奖励作弊（reward hacking），我们引入了不确定性感知的蒙特卡洛（MC）dropout切换机制，选择性地对低置信度的候选方案调用真实的Vitis HLS综合，并在线更新代理模型，从而创建一个自我改进的奖励系统。

    arXiv:2605.13536v2 Announce Type: replace-cross  Abstract: High-Level Synthesis (HLS) compiles algorithmic C/C++ descriptions into hardware, with Quality of Results (QoR)---latency and resource utilization---critically governed by pragma configurations and code structure. Existing natural-language-to-HLS (NL-to-HLS) training approaches prioritize functional correctness while largely ignoring QoR. We observe that reinforcement learning (RL) for HLS does not require absolute synthesis results---only relative comparisons between candidates. Based on this insight, we propose \textbf{HLS-Seek}, a QoR-aware NL-to-HLS framework that avoids full synthesis-in-the-loop RL via a comparative proxy reward model achieving 99.53\% Pareto-dominance accuracy. To prevent reward hacking, we introduce \textit{uncertainty-aware Monte Carlo (MC) dropout switching} that selectively invokes real Vitis HLS synthesis for low-confidence candidates and online updates the proxy, creating a self-improving reward sy
    
[^192]: 归纳式Venn-Abers及相关回归器

    Inductive Venn-Abers and related regressors

    [https://arxiv.org/abs/2605.06646](https://arxiv.org/abs/2605.06646)

    本文通过引入共形预测，将Venn-Abers预测器从二元分类和有界回归推广到无界回归，并证明由此导出的点回归器在较大训练集上能在一定程度上提升标准回归器的预测效率。

    

    Venn-Abers预测器是具有良好有效性性质的概率预测器，但其主要局限在于此前仅适用于二元分类，尽管最近已有扩展到有界回归的工作。本文将其推广到无界回归的情形，这需要引入共形预测的元素。在模拟和实证研究中，我们考察了由Venn-Abers回归器导出的点回归器的预测效率，并论证了在较大训练集上，它们能在一定程度上提升标准回归器的预测效率。

    arXiv:2605.06646v2 Announce Type: replace  Abstract: Venn-Abers predictors are probabilistic predictors that enjoy appealing properties of validity, but their major limitation is that they have been applicable only to binary classification, apart from a recent extension to bounded regression. We generalize them to the case of unbounded regression, which requires adding an element of conformal prediction. In our simulation and empirical studies we investigate the predictive efficiency of point regressors derived from Venn-Abers regressors and argue that they somewhat improve the predictive efficiency of standard regressors for larger training sets.
    
[^193]: 在碳捕集与封存应用的贝叶斯优化中诱导置换不变先验

    Inducing Permutation Invariant Priors in Bayesian Optimization for Carbon Capture and Storage Applications

    [https://arxiv.org/abs/2605.02409](https://arxiv.org/abs/2605.02409)

    本文提出了一种新颖的置换不变高斯过程核GP-Perm，通过比较集合经验表示之间的稳定散度来编码置换不变性，从而提升碳捕集与封存项目中井位布置的贝叶斯优化效率。

    

    贝叶斯优化是一种迭代方法，专为优化昂贵的黑盒目标函数而设计。作为贝叶斯优化中黄金标准的高斯过程等代理模型，在处理具有置换对称性的输入时可能效率低下，因为最常用的核函数更适合向量输入，而非无序的条目集合。受此问题启发，我们将置换不变的贝叶斯优化应用于碳捕集与封存项目中的井位布置问题。该高保真黑盒模拟器被设置为在群组控制下操作油井，从而在注入井组与生产井组内部产生了置换对称性，而标准的高斯过程核无法利用这种对称性。在这项工作中，我们的主要贡献是一种新颖的高斯过程核（GP-Perm），它通过比较集合所诱导的经验表示之间的稳定散度来编码置换不变性，并且能够……（原文摘要至此截断）

    arXiv:2605.02409v3 Announce Type: replace  Abstract: Bayesian Optimization is an iterative method, tailored to optimizing expensive black box objective functions. Surrogate models like Gaussian Processes, which are the gold standard in Bayesian Optimization, can be inefficient for inputs with permutation symmetries, as the most common kernels employed are better suited for vector inputs rather than unordered sets of items. Motivated by this issue, we turn to permutation invariant Bayesian Optimization for well placement in Carbon Capture and Storage projects. The high fidelity black box simulator is instructed to operate wells under group control, giving rise to permutation symmetries within injector and producer groups that cannot be exploited with standard GP kernels. In this work, our main contribution is a novel Gaussian Process kernel (GP-Perm) that encodes permutation invariance by comparing sets through a stable divergence between their induced empirical representations, and can
    
[^194]: 用微分同胚重定位R^n中的紧集以及R^n中数据集的线性可分性

    Relocation of compact sets in $\mathbb{R}^n$ by diffeomorphisms and linear separability of datasets in $\mathbb{R}^n$

    [https://arxiv.org/abs/2604.21393](https://arxiv.org/abs/2604.21393)

    本文建立了通过微分同胚将R^n中有限个紧集重定位到任意目标区域的理论，并证明通过到R^{n+1}的可微嵌入或宽度为n、采用Leaky-ReLU/ELU/SELU激活函数的深度神经网络，可使这些数据集变得线性可分。

    

    通过自微分同胚重定位n维流形中的紧集本身就具有重要的研究价值，同时在数据科学的数据分类方面具有显著的潜在应用。本文提出了一种理论，用于通过R^n的微分同胚将R^n中有限个紧集重定位到R^n中的任意目标区域。此外，我们证明对于任意这样的紧集集合，存在一个到R^{n+1}的可微嵌入，使得它们的像成为线性可分的。作为该理论的应用，我们证明在温和条件下，R^n中有限个紧数据集可以通过宽度为n、采用Leaky-ReLU、ELU或SELU激活函数的深度神经网络（DNN）实现线性可分。此外，我们还证明R^n中任意有限个相互不相交的紧数据集可以……

    arXiv:2604.21393v2 Announce Type: replace  Abstract: Relocation of compact sets in an $n$-dimensional manifold by self-diffeomorphism is of its own interest as well as significant potential applications to data classification in data science. This paper presents a theory for relocating a finite number of compact sets in $\mathbb{R}^n$ to be relocated to arbitrary target domains in $\mathbb{R}^n$ by diffeomorphisms of $\mathbb{R}^n$. Furthermore, we prove that for any such collection, there exists a differentiable embedding into $\mathbb{R}^{n+1}$ such that their images become linearly separable.   As applications of the established theory, we show that a finite number of compact datasets in $\mathbb{R}^n$ can be made linearly separable by width-$n$ deep neural networks (DNNs) with Leaky-ReLU, ELU, or SELU activation functions, under a mild condition. In addition, we show that any finite number of mutually disjoint compact datasets in $\mathbb{R}^n$ can be made linearly separable in $\m
    
[^195]: 利用机器学习推进次季节预报

    Advancing Subseasonal Forecasting with Machine Learning

    [https://arxiv.org/abs/2604.16238](https://arxiv.org/abs/2604.16238)

    本文提出概率偏差校正（PBC）机器学习框架，通过学习校正历史概率预报大幅减少系统性误差，使ECMWF的AI预报系统的次季节（2-6周）预报技巧翻倍。

    

    决策者依靠天气预报来种植作物、管理野火、配置水资源和能源，以及为极端天气做好准备。如今，得益于基于物理的动力学模型和数据驱动的人工智能（AI）模型的稳步进步，此类预报在未来两周内享有前所未有的准确性。然而，由于误差累积、系统性模型偏差以及大气的混沌特性，模型在次季节时间尺度（未来2-6周）的预报技巧急剧下降。为应对这一性能衰退，我们提出了概率偏差校正（PBC），这是一种机器学习框架，通过学习校正历史概率预报来大幅减少系统性误差。当应用于欧洲中期天气预报中心（ECMWF）的领先动力学模型和AI模型时，PBC使AI预报系统原本有限的次季节预报技巧翻倍，并提升了业务预报模型的技巧……

    arXiv:2604.16238v3 Announce Type: replace  Abstract: Decision-makers rely on weather forecasts to plant crops, manage wildfires, allocate water and energy, and prepare for weather extremes. Today, such forecasts enjoy unprecedented accuracy out to two weeks thanks to steady advances in physics-based dynamical models and data-driven artificial intelligence (AI) models. However, model skill drops precipitously at subseasonal timescales (2 - 6 weeks ahead), due to compounding errors, systemic model biases, and the chaotic nature of the atmosphere. To counter this degradation, we introduce probabilistic bias correction (PBC), a machine learning framework that substantially reduces systematic error by learning to correct historical probabilistic forecasts. When applied to the leading dynamical and AI models from the European Centre for Medium-Range Weather Forecasts (ECMWF), PBC doubles the modest subseasonal skill of the AI Forecasting System and improves the skill of the operationally-deb
    
[^196]: 多项式群卷积神经网络的几何学

    The Geometry of Polynomial Group Convolutional Neural Networks

    [https://arxiv.org/abs/2603.29566](https://arxiv.org/abs/2603.29566)

    本文基于分次群代数为多项式群卷积神经网络提出了新的数学框架，给出了 Hadamard 和 Kronecker 两种自然参数化方法，并证明了其神经流形维度仅取决于网络层数和群的大小。

    

    我们研究了针对任意有限群 $G$ 的多项式群卷积神经网络（PGCNNs）。特别地，我们使用分次群代数的语言为 PGCNNs 引入了一个新的数学框架。该框架基于 Hadamard 积和 Kronecker 积给出了该架构的两种自然参数化方法，二者通过一个线性映射相互关联。我们计算了相关神经流形的维度，验证了其仅取决于网络层数和群的大小。我们还描述了 Kronecker 参数化在正则群作用和重缩放意义下的一般纤维，并对 Hadamard 参数化提出了类似的猜想。该猜想得到了针对小群和浅层网络的显式计算的支持。

    arXiv:2603.29566v2 Announce Type: replace  Abstract: We study polynomial group convolutional neural networks (PGCNNs) for an arbitrary finite group $G$. In particular, we introduce a new mathematical framework for PGCNNs using the language of graded group algebras. This framework yields two natural parametrizations of the architecture, based on Hadamard and Kronecker products, related by a linear map. We compute the dimension of the associated neuromanifold, verifying that it depends only on the number of layers and the size of the group. We also describe the general fiber of the Kronecker parametrization up to the regular group action and rescaling, and conjecture the analogous description for the Hadamard parametrization. Our conjecture is supported by explicit computations for small groups and shallow networks.
    
[^197]: 基于储备池的图卷积网络

    Reservoir-Based Graph Convolutional Networks

    [https://arxiv.org/abs/2603.24131](https://arxiv.org/abs/2603.24131)

    该论文将结构化卷积机制引入基于储备池的图神经网络，在不依赖深层网络堆叠和大量参数调优的情况下实现稳定的长程信息传播，从而缓解传统GCN的过度平滑与高计算成本问题。

    

    消息传递是图神经网络的核心机制，它通过聚合邻居节点的信息来迭代更新节点嵌入。图卷积网络是这一方法的典型代表，它将卷积运算适配到图结构上，从而能够有效地融合相邻节点的特征。然而，GCN在处理复杂或动态数据时会遇到挑战：捕获长程依赖通常需要更深的网络层，这不仅会增加计算成本，还会导致过度平滑问题，使节点嵌入变得难以区分。为了克服这些挑战，研究者已将储备池计算集成到图神经网络中，利用迭代的消息传递动力学实现稳定的信息传播，而无需大量的参数调优。尽管前景可观，现有的基于储备池的模型缺乏结构化的卷积机制，限制了其准确聚合信息的能力……（注：原文摘要在此处被截断）

    arXiv:2603.24131v2 Announce Type: replace  Abstract: Message passing is a core mechanism in Graph Neural Networks (GNNs), enabling the iterative update of node embeddings by aggregating information from neighboring nodes. Graph Convolutional Networks (GCNs) exemplify this approach by adapting convolutional operations for graph structures, allowing features from adjacent nodes to be combined effectively. However, GCNs encounter challenges with complex or dynamic data. Capturing long-range dependencies often requires deeper layers, which not only increase computational costs but also lead to over-smoothing, where node embeddings become indistinguishable. To overcome these challenges, reservoir computing has been integrated into GNNs, leveraging iterative message-passing dynamics for stable information propagation without extensive parameter tuning. Despite its promise, existing reservoir-based models lack structured convolutional mechanisms, limiting their ability to accurately aggregate
    
[^198]: 基于贝叶斯跟踪的自回归引导深度空间选择性滤波器，实现移动说话人的高效提取

    Autoregressive Guidance of Deep Spatially Selective Filters using Bayesian Tracking for Efficient Extraction of Moving Speakers

    [https://arxiv.org/abs/2603.23723](https://arxiv.org/abs/2603.23723)

    本文提出了可与任意深度空间滤波器兼容的贝叶斯跟踪算法，通过自回归方式将增强后的语音信号反馈给轻量级跟踪器以引导深度空间选择性滤波器，从而实现对移动说话人语音的高效实时提取。

    

    深度空间选择性滤波器能够针对方向已知的静止说话人实现高质量语音增强，并具备可实时运行的架构。为了在仅已知说话人初始方向的动态场景中保持同样的性能水平，需要准确且计算轻量的跟踪算法。在逐帧因果处理的前提下，时间反馈机制允许利用增强后的语音信号来提升跟踪性能。在本工作中，我们研究了将增强信号融入轻量级跟踪算法并以自回归方式引导深度空间滤波器的策略。我们提出的贝叶斯跟踪算法可与任意深度空间滤波器兼容。为提高开发和评估过程中模拟轨迹的真实性，我们开发了一个基于社会力模型的合成数据生成框架。实验结果验证了自回归引导方法的有效性。

    arXiv:2603.23723v3 Announce Type: replace-cross  Abstract: Deep spatially selective filters achieve high-quality enhancement with real-time capable architectures for stationary speakers of known directions. To retain this level of performance in dynamic scenarios where only the speakers' initial directions are given, accurate, yet computationally lightweight tracking algorithms become necessary. Assuming a frame-wise causal processing style, temporal feedback allows for leveraging the enhanced speech signal to improve tracking performance. In this work, we investigate strategies to incorporate the enhanced signal into lightweight tracking algorithms and autoregressively guide deep spatial filters. Our proposed Bayesian tracking algorithms are compatible with arbitrary deep spatial filters. To increase the realism of simulated trajectories during development and evaluation, we develop a synthetic data generation framework based on the social force model. Results validate that the autore
    
[^199]: 基于Kolmogorov-Arnold网络与视觉-语言基础模型的YOLO：面向计算机视觉感知中可解释目标检测的可信多模态AI

    YOLO with Kolmogorov-Arnold networks and vision-language foundation models for interpretable object detection with trustworthy multimodal AI in computer vision perception

    [https://arxiv.org/abs/2603.23037](https://arxiv.org/abs/2603.23037)

    该论文提出用Kolmogorov-Arnold网络作为可解释的事后代理模型，基于七个几何与语义特征评估YOLOv10检测结果的置信度可信性，并结合BLIP视觉-语言基础模型生成描述，实现计算机视觉感知中透明、可信的目标检测。

    

    本文研究了一种新型Kolmogorov-Arnold网络框架的可信目标检测能力。该方法解决了车辆检测感知乃至更广泛计算机视觉领域的一个关键局限：这些系统在视觉退化或模糊场景下，其置信度分数的可靠性缺乏透明度。为此，本文采用Kolmogorov-Arnold网络作为可解释的事后代理模型，利用七个几何与语义特征对YOLOv10检测的可信度进行建模。Kolmogorov-Arnold网络的加性样条结构使得每个特征的影响可以直接可视化，产生平滑且透明的函数映射，从而揭示模型的置信度何时得到充分支持、何时不可靠。此外，引导式语言-图像预训练（BLIP）基础模型为每个检测结果生成描述性文本说明……

    arXiv:2603.23037v2 Announce Type: replace-cross  Abstract: The trustworthy object detection capabilities of a novel Kolmogorov-Arnold network framework are examined here. The approach addresses a key limitation in computer vision for vehicle detection perception, and beyond. These systems offer limited transparency regarding the reliability of their confidence scores in visually degraded or ambiguous scenes. To this end, a Kolmogorov-Arnold network is employed as an interpretable post-hoc surrogate to model the trustworthiness of the You Only Look Once (Yolov10) detections using seven geometric and semantic features. The additive spline-based structure of the Kolmogorov-Arnold network enables direct visualisation of each feature's influence. This produces smooth and transparent functional mappings that reveal when the model's confidence is well supported and when it is unreliable. Furthermore, a bootstrapped language-image (BLIP) foundation model generates descriptive captions of each 
    
[^200]: 尼泊尔语护照问答：面向公共服务应用的低资源数据集

    Nepali Passport Question Answering: A Low-Resource Dataset for Public Service Applications

    [https://arxiv.org/abs/2603.13320](https://arxiv.org/abs/2603.13320)

    该研究构建了首个面向护照公共服务的尼泊尔语问答低资源数据集，通过微调Transformer嵌入模型并结合BM25进行混合检索，其中基于多语言E5嵌入的模型取得了最佳检索性能。

    

    尼泊尔语作为一种低资源语言，由于缺乏标注数据和计算语言学资源，在构建有效的信息检索系统方面面临重大挑战。在本研究中，我们试图通过构建成对结构的尼泊尔语问答数据集来弥补这一空白。我们专注于与护照服务相关的常见问题（FAQs），构建了用于训练和评估信息检索模型的数据集。在研究中，我们针对问答检索中的语义相似性任务对基于Transformer的嵌入模型进行了微调，并将微调后的模型与基线模型BM25进行了比较。此外，我们还实现了一种将微调模型与BM25相结合的混合检索方法，并评估了混合检索的性能。结果表明，微调的基于SBERT的模型优于BM25，而基于多语言E5嵌入的模型取得了最高的检索性能。

    arXiv:2603.13320v2 Announce Type: replace-cross  Abstract: Nepali, a low-resource language, faces significant challenges in building an effective information retrieval system due to the unavailability of annotated data and computational linguistic resources. In this study, we attempt to address this gap by preparing a pair-structured Nepali Question-Answer dataset. We focus on Frequently Asked Questions (FAQs) for passport-related services, building a data set for training and evaluation of IR models. In our study, we have fine-tuned transformer-based embedding models for semantic similarity in question-answer retrieval. The fine-tuned models were compared with the baseline BM25. In addition, we implement a hybrid retrieval approach, integrating fine-tuned models with BM25, and evaluate the performance of the hybrid retrieval. Our results show that the fine-tuned SBERT-based models outperform BM25, whereas multilingual E5 embedding-based models achieve the highest retrieval performance
    
[^201]: 延续与拒绝之间的拉锯战：对大语言模型中延续触发越狱现象的机制分析

    The Struggle Between Continuation and Refusal: A Mechanistic Analysis of the Continuation-Triggered Jailbreak in LLMs

    [https://arxiv.org/abs/2603.08234](https://arxiv.org/abs/2603.08234)

    该论文通过注意力头层面的机制可解释性分析（因果干预与激活缩放），揭示了大语言模型中“延续触发越狱”现象的根源在于模型内在的文本延续驱动力与安全拒绝机制之间的固有竞争。

    

    随着大语言模型（LLMs）的快速发展，LLM的安全性已成为一个关键关注点。尽管在安全对齐方面已付出巨大努力，当前的LLM仍然容易受到越狱攻击。然而，这类脆弱性的根本原因至今仍鲜为人知，因此学术界和工业界亟需对越狱机制展开严谨的研究。在本工作中，我们关注一种延续触发的越狱现象，即仅仅重新定位一个延续触发的指令后缀，就能显著提高越狱成功率。为了揭示这一现象的内在机制，我们在注意力头的层面开展了全面的机制可解释性分析。通过因果干预和激活缩放实验，我们表明这种越狱行为主要源于模型内在的文本延续驱动力与安全拒绝机制之间的一种固有竞争。

    arXiv:2603.08234v2 Announce Type: replace  Abstract: With the rapid advancement of large language models (LLMs), the safety of LLMs has become a critical concern. Despite significant efforts in safety alignment, current LLMs remain vulnerable to jailbreaking attacks. However, the root causes of such vulnerabilities are still poorly understood, necessitating a rigorous investigation into jailbreak mechanisms across both academic and industrial communities. In this work, we focus on a continuation-triggered jailbreak phenomenon, whereby simply relocating a continuation-triggered instruction suffix can substantially increase jailbreak success rates. To uncover the intrinsic mechanisms of this phenomenon, we conduct a comprehensive mechanistic interpretability analysis at the level of attention heads. Through causal interventions and activation scaling, we show that this jailbreak behavior primarily arises from an inherent competition between the model's intrinsic continuation drive and th
    
[^202]: Squint：面向仿真到现实机器人的快速视觉强化学习

    Squint: Fast Visual Reinforcement Learning for Sim-to-Real Robotics

    [https://arxiv.org/abs/2602.21203](https://arxiv.org/abs/2602.21203)

    本文提出Squint，一种视觉Soft Actor Critic方法，通过并行仿真、分布式评论家和优化实现等技术，实现了比以往视觉离策略和同策略方法更快的实际训练速度，可高效完成仿真到现实的机器人操作任务。

    

    视觉强化学习对机器人领域很有吸引力，但代价高昂。离策略方法样本效率高但训练速度慢，而同策略方法虽然易于并行化但浪费样本。近期研究表明，对于基于状态的控制任务，离策略方法在实际训练时间（wall-clock time）上可以快于同策略方法。然而，将这一结论扩展到视觉任务仍然具有挑战性，因为高维输入图像使训练动态变得复杂，并带来巨大的存储和编码开销。为了解决这些挑战，我们提出了Squint，一种视觉Soft Actor Critic方法，其训练速度在实际耗时上超过了以往各类视觉离策略和同策略方法。Squint通过并行仿真、分布式评论家、分辨率squinting、层归一化、经过调优的更新-数据比以及优化实现来实现这一点。我们在SO-101任务集上进行评估，这是ManiSkill3中一套全新的包含八个操作任务的基准测试，并带有大量域随机化。

    arXiv:2602.21203v2 Announce Type: replace-cross  Abstract: Visual reinforcement learning is appealing for robotics but expensive. Off-policy methods are sample-efficient yet slow while on-policy methods parallelize well but waste samples. Recent work has shown that off-policy methods can train faster than on-policy methods in wall-clock time for state-based control. Extending this to vision remains challenging, where high-dimensional input images complicate training dynamics and introduce substantial storage and encoding overhead. To address these challenges, we introduce Squint, a visual Soft Actor Critic method that achieves faster wall-clock training than prior visual off-policy and on-policy methods. Squint achieves this via parallel simulation, a distributional critic, resolution squinting, layer normalization, a tuned update-to-data ratio, and an optimized implementation. We evaluate on the SO-101 Task Set, a new suite of eight manipulation tasks in ManiSkill3 with heavy domain r
    
[^203]: 谱Barron空间中二阶椭圆偏微分方程的正则性

    Regularity of Second-Order Elliptic PDEs in Spectral Barron Spaces

    [https://arxiv.org/abs/2602.19381](https://arxiv.org/abs/2602.19381)

    该论文证明了在温和的椭圆性条件下，$\mathbb{R}^d$ 上二阶椭圆偏微分方程的解在谱Barron空间中获得额外两阶正则性，从而确定了一类其解可被宽度与空间维度无关的两层余弦激活神经网络逼近的偏微分方程。

    

    我们在谱Barron空间中建立了 $\mathbb{R}^{d}$ 上二阶椭圆偏微分方程的正则性定理。在温和的椭圆性和小性假设下，方程的解获得了额外两阶的Barron正则性。作为推论，我们确定了一类偏微分方程，其解可以由具有余弦激活函数的两层神经网络逼近，且神经网络的宽度与空间维度无关。

    arXiv:2602.19381v2 Announce Type: replace-cross  Abstract: We establish a regularity theorem for second-order elliptic PDEs on $\mathbb{R}^{d}$ in spectral Barron spaces. Under mild ellipticity and smallness assumptions, the solution gains two additional orders of Barron regularity. As a corollary, we identify a class of PDEs whose solutions can be approximated by two-layer neural networks with cosine activation functions, where the width of the neural network is independent of the spatial dimension.
    
[^204]: Brain4FMs：面向脑电信号的基础模型基准测试

    Brain4FMs: A Benchmark of Foundation Models for Electrical Brain Signal

    [https://arxiv.org/abs/2602.11558](https://arxiv.org/abs/2602.11558)

    该论文提出了首个统一评估脑基础模型在EEG和iEEG两类脑电信号上表现的开放基准测试Brain4FMs，集成了17个代表性模型和21个公开数据集，覆盖临床诊断、睡眠分期、通信和情感计算四大应用领域。

    

    脑基础模型（BFM）通过从神经信号中学习可迁移的表示，正在推动神经技术的发展，在临床诊断和神经科学研究中具有广阔的应用前景。其发展依赖于大规模的脑电信号预训练语料库，包括头皮脑电图（EEG）和颅内脑电图。然而，现有的BFM基准测试主要聚焦于EEG，仅涵盖有限的模型子集，且除下游性能之外提供的分析较为有限。我们提出了Brain4FMs，据我们所知，这是首个用于联合评估BFM在EEG和iEEG上表现的统一基准。它集成了17个代表性模型和21个公开数据集，覆盖临床诊断、睡眠分期、通信和情感计算等领域。Brain4FMs开放且即插即用，具备数据集感知的预处理、跨被试评估、异构多通道处理以及标准化的下游适配等特性。

    arXiv:2602.11558v2 Announce Type: replace  Abstract: Brain foundation models (BFMs) are advancing neurotechnology by learning transferable representations from neural signals, with broad potential in clinical diagnosis and neuroscience research. Their development relies on large-scale pretraining corpora of electrical brain signals, including scalp electroencephalography (EEG) and intracranial EEG (iEEG). However, existing BFM benchmarks primarily focus on EEG, cover only a limited subset of models, and provide limited analysis beyond downstream performance. We introduce Brain4FMs, the first unified benchmark, to our knowledge, for jointly evaluating BFMs on EEG and iEEG. It integrates 17 representative models and 21 public datasets across clinical diagnosis, sleep staging, communication, and affective computing. Brain4FMs is open and plug-and-play, with dataset-aware preprocessing, cross-subject evaluation, heterogeneous multichannel handling, and standardized downstream adaptation wo
    
[^205]: 通过相关性感知支付增强仿射最大化拍卖

    Enhancing Affine Maximizer Auctions with Correlation-Aware Payment

    [https://arxiv.org/abs/2602.09455](https://arxiv.org/abs/2602.09455)

    提出相关性感知仿射最大化拍卖（CA-AMA）框架，通过引入相关性感知支付在保持主导策略激励兼容性的同时突破经典AMA的表达能力限制，能在经典AMA表现任意差的估值相关场景下达到最优收益，并配套设计了两阶段训练算法。

    

    arXiv:2602.09455v2 公告类型： replace-cross 摘要：仿射最大化拍卖（AMAs）是一类从VCG机制推广而来的机制族，由于其天然具有主导策略激励兼容性（DSIC）和个体理性（IR），被广泛应用于自动化机制设计中。然而，由于其支付形式是固定的，AMA的表达能力受到限制，尤其是在竞买人估值相互关联的分布情形下。在本文中，我们提出了相关性感知仿射最大化拍卖（CA-AMA），这是一个新颖的框架，通过一种新的相关性感知支付来增强AMA。我们证明了任何CA-AMA都能保持DSIC性质，并将寻找最优CA-AMA形式化为一个受IR约束的约束优化问题。随后，我们从理论上刻画了经典AMA相比最优收益可能表现任意糟糕的场景，而CA-AMA却能够达到最优收益。为了优化CA-AMA，我们设计了一种实用的两阶段训练算法。我们推导出目标函数……

    arXiv:2602.09455v2 Announce Type: replace-cross  Abstract: Affine Maximizer Auctions (AMAs), a generalized mechanism family from VCG, are widely used in automated mechanism design due to their inherent dominant-strategy incentive compatibility (DSIC) and individual rationality (IR). However, as the payment form is fixed, AMA's expressiveness is restricted, especially in distributions where bidders' valuations are correlated. In this paper, we propose Correlation-Aware AMA (CA-AMA), a novel framework that augments AMA with a new correlation-aware payment. We show that any CA-AMA preserves the DSIC property and formalize finding optimal CA-AMA as a constraint optimization problem subject to the IR constraint. Then, we theoretically characterize scenarios where classic AMAs can perform arbitrarily poorly compared to the optimal revenue, while the CA-AMA can reach the optimal revenue. For optimizing CA-AMA, we design a practical two-stage training algorithm. We derive that the target funct
    
[^206]: MemCoRe：从渐进压缩的事实知识中恢复证据以支持智能体记忆

    MemCoRe: Recovering Evidence from Progressively Compressed Factual Knowledge for Agent Memory

    [https://arxiv.org/abs/2602.07885](https://arxiv.org/abs/2602.07885)

    提出MemCoRe，将智能体记忆组织为压缩层级结构，在逐级压缩冗余的同时保留各层级检索所需结构，从而在记忆压缩与证据检索有效性之间实现平衡。

    

    记忆系统能够使大语言模型智能体从通过不断增长的交互历史所积累的事实知识中整合并检索相关证据，以支持下游推理。现有方法已探索了多种组织与压缩这些历史的策略。然而，如何在压缩与检索有效性之间取得平衡仍然具有挑战性：保留过多内容会导致相关证据被冗余条目所掩盖，而过于激进地丢弃则可能移除后来被证明相关的内容。这实质上构成了压缩冗余与保留足够结构以检索目标证据之间的权衡，信息瓶颈理论对这一权衡进行了形式化刻画。为此，我们提出了MemCoRe，它将记忆组织为一个压缩层级结构，其中每一层都在进一步压缩冗余的同时，保留该层级检索所需的结构。在这一层级中，证据被渐进式压缩……

    arXiv:2602.07885v3 Announce Type: replace  Abstract: Memory systems enable LLM agents to consolidate and retrieve relevant evidence from the factual knowledge accumulated through growing interaction histories for downstream reasoning. Existing approaches have explored diverse strategies for organizing and compressing these histories. However, balancing compression with retrieval effectiveness remains challenging: retaining too much content can cause relevant evidence to be obscured by redundant entries, while discarding too aggressively may remove content that later proves relevant. This amounts to a tradeoff between compressing redundancy and preserving enough structure to retrieve target evidence, as formalized by the information bottleneck. To this end, we propose MemCoRe, which organizes memory as a compression hierarchy where each level compresses redundancy further while retaining the structure needed for retrieval at that level. In this hierarchy, evidence is progressively compr
    
[^207]: 用于文本生成的共识组相对策略优化

    Consensus Group Relative Policy Optimization for Text Generation

    [https://arxiv.org/abs/2602.03102](https://arxiv.org/abs/2602.03102)

    该论文提出C-GRPO方法，通过将最小贝叶斯风险（MBR）解码的共识效用蒸馏为GRPO框架中的组相对目标函数，在不依赖黄金参考或偏好标签的前提下，将高昂的推理时“采样-重排序”计算成本转移到训练阶段。

    

    许多强大的文本生成解码方法遵循“采样-重排序”范式：它们抽取多个候选结果，使用基于样本间共识的效用（奖励）函数对每个候选进行评分，并返回最佳的一个。尽管这些方法行之有效，但由于需要反复采样和评分，它们在推理阶段会产生高昂的计算成本。先前摊销推理时计算的尝试通常依赖于黄金参考、教师标签或精心策划的偏好数据，这增加了数据集构建的工作量，并提高了对高保真奖励模型的要求。我们提出了共识组相对策略优化，该方法通过将共识效用表述为GRPO框架内的组相对目标，将最小贝叶斯风险（MBR）解码蒸馏到训练过程中。C-GRPO仅需要一个效用函数和策略样本，无需黄金参考或显式偏好标签。在理想条件下，我们证明了C-GRPO的目标函数……（原文摘要在此处截断）

    arXiv:2602.03102v2 Announce Type: replace  Abstract: Many strong decoding methods for text generation follow a sample-and-rerank paradigm: they draw multiple candidates, score each under a utility (reward) function using consensus across samples, and return the best one. Although effective, these methods incur high computational costs during inference due to repeated sampling and scoring. Prior attempts to amortize inference-time computation typically rely on gold references, teacher labels, or curated preference data, increasing dataset construction effort and the demand for high-fidelity reward models. We propose Consensus Group Relative Policy Optimization (C-GRPO), which distills Minimum Bayes Risk (MBR) decoding into training by formulating the consensus utility as a group-relative objective within GRPO. C-GRPO requires only a utility function and policy samples, without gold references or explicit preference labels. Under ideal conditions, we show that the objective function of C
    
[^208]: 基于调制专家混合的多模态时间序列预测

    Multi-Modal Time Series Prediction via Mixture of Modulated Experts

    [https://arxiv.org/abs/2601.21547](https://arxiv.org/abs/2601.21547)

    提出了一种名为“专家调制”的新机制，使混合专家模型的路由与专家计算均受文本信号调制，从而摆脱对token级融合的依赖，提升多模态时间序列预测的准确性与跨模态对齐能力。

    

    现实世界中的时间序列呈现出复杂且不断演化的动态特性，使得准确预测极具挑战性。近期的多模态预测方法利用新闻报道等文本信息来改进预测，但大多数方法依赖于token级融合，即在共享嵌入空间中将时间片段与语言token混合。然而，当高质量的“时间序列-文本”配对数据稀缺、且时间序列在特性上存在巨大差异时，这种融合方式可能并不适用，从而使跨模态对齐变得复杂。与此同时，混合专家（MoE）架构已被证明在时间序列建模和多模态学习中均行之有效，但许多现有的基于MoE的模态融合方法仍然依赖于token级融合。为解决这一问题，我们提出了专家调制，这是一种用于多模态时间序列预测的新机制，它使路由和专家计算均以文本信号为条件，从而能够……（摘要至此截断）

    arXiv:2601.21547v2 Announce Type: replace-cross  Abstract: Real-world time series exhibit complex and evolving dynamics, making accurate forecasting extremely challenging. Recent multi-modal forecasting methods leverage textual information such as news reports to improve prediction, but most rely on token-level fusion that mixes temporal patches with language tokens in a shared embedding space. However, such fusion can be ill-suited when high-quality time-text pairs are scarce and when time series exhibit substantial variation in characteristics, thus complicating cross-modal alignment. In parallel, mixture-of-experts (MoE) architectures have proven effective for both time series modeling and multi-modal learning, yet many existing MoE-based modality integration methods still depend on token-level fusion. To address this, we propose Expert Modulation, a new mechanism for multi-modal time series prediction that conditions both routing and expert computation on textual signals, enabling 
    
[^209]: TeleTables：面向电信表格解读的大语言模型基准测试

    TeleTables: A Benchmark for Large Language Models in Telecom Table Interpretation

    [https://arxiv.org/abs/2601.04202](https://arxiv.org/abs/2601.04202)

    该论文提出TeleTables基准（包含2,220张3GPP规范表格和500道人工验证选择题），通过评估20个开源大语言模型揭示了电信表格解读的两大瓶颈：闭卷时领域知识不足导致准确率不超过41%，而提供表格上下文时准确率虽可超90%，但会随推理深度、证据范围和表格结构复杂性增加而系统性下降。

    

    大语言模型（LLM）越来越多地被应用于电信工程任务，但在3GPP规范上的表现较差。这些标准将大量技术信息编码在复杂的表格中，而大语言模型对此类表格的知识与解读能力在很大程度上仍未被探索。我们提出了TeleTables，该基准包含来自13份3GPP规范、四种格式的2,220张表格，以及500道经人工验证、涵盖从直接检索到多步推理的选择题。我们对20个开源权重大语言模型（涵盖非推理、多模态、推理和表格专用架构）的评估揭示了两个显著的性能瓶颈：在闭卷设置下，领域知识是主要制约因素，没有任何通用模型的准确率超过41%；当表格作为上下文提供时，最优模型的准确率超过90%，但性能会随着推理深度、证据范围和结构复杂性的增加而系统性下降，最大差距达32.2个百分点。

    arXiv:2601.04202v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) are increasingly applied to telecom engineering tasks, yet perform poorly on 3GPP specifications. These standards encode much of their technical information in complex tables, but LLM knowledge and interpretation of such tables remain largely unexplored. We introduce TeleTables, a benchmark comprising 2,220 tables from 13 3GPP specifications in four formats and 500 human-verified MCQs spanning direct retrieval to multi-step reasoning. Evaluating 20 open-weight LLMs across non reasoning, multimodal, reasoning, and table specialized architectures reveals two distinct performance bottlenecks. In the closed-book setting, domain knowledge is the primary constraint, with no general-purpose model exceeding 41% accuracy. When the table is provided as context, the best models exceed 90%, but performance degrades systematically with reasoning depth, evidence scope, and structural complexity, with a 32.2pp spr
    
[^210]: GLOW：用于智能体工作流性能预测的图-语言协同编码

    GLOW: Graph-Language Co-Encoding for Agentic Workflow Performance Prediction

    [https://arxiv.org/abs/2512.15751](https://arxiv.org/abs/2512.15751)

    GLOW 提出了一个将图神经网络的结构建模能力与大语言模型的拓扑感知语义编码能力相结合的统一框架，通过图-语言协同编码高效预测智能体工作流的性能，从而避免昂贵的基于执行的评估。

    

    智能体工作流已成为解决复杂任务的一种有前景的范式。然而，自动生成高质量的智能体工作流仍然代价高昂，因为智能体工作流的优化需要通过执行来评估大量候选工作流，从而导致高昂的计算成本和延迟。近年来，智能体工作流性能预测已成为一个热门研究课题，旨在避免昂贵的基于执行的评估，但现有方法主要使用图神经网络（GNN）来建模工作流结构，未能充分捕捉智能体之间的语义关系。为解决这一局限，我们提出了 GLOW，一个用于智能体工作流性能预测的统一框架，它将 GNN 的图结构建模能力与大语言模型（LLM）的拓扑感知语义编码能力相结合。具体而言，首先通过在图理解任务上进行指令微调构建一个面向图的 LLM，以提取拓扑感知的语义表示……

    arXiv:2512.15751v2 Announce Type: replace-cross  Abstract: Agentic Workflows (AWs) have emerged as a promising paradigm for solving complex tasks. However, automatically generating high-quality AWs remains expensive because AW optimization requires evaluating a large number of candidate AWs via execution, resulting in high computational cost and latency. Recently, AW performance prediction has become a hot research topic to avoid costly execution-based evaluation, but existing methods primarily use Graph Neural Networks (GNNs) to model workflow structures and insufficiently capture the semantic relationships among agents. To address this limitation, we propose GLOW, a unified framework for AW performance prediction that combines the graph-structure modeling ability of GNNs with the topology-aware semantic encoding capability of LLMs. Specifically, a graph-oriented LLM is first built through instruction-tuning on graph understanding tasks to extract topology-aware semantic representatio
    
[^211]: 预报技能不等于决策技能：来自天气相关决策任务的证据

    Forecast Skill Is Not Decision Skill: Evidence from Weather-Dependent Decision Tasks

    [https://arxiv.org/abs/2512.14779](https://arxiv.org/abs/2512.14779)

    该论文提出“决策校准”框架，从决策者视角在决策层面评估概率预报，并通过对天气相关决策任务的实验发现，机器学习模型与数值天气预报模型在预报层面的性能差异并不能可靠地反映其在下游决策中的实际价值。

    

    标准的天气预报评估侧重于预报员的视角，并采用统计方法对预报与观测进行比较。然而在实践中，预报是用于辅助决策的，因此从决策者的视角出发，通过预报改善决策的能力来量化其价值似乎是更自然的方式。决策校准提供了一个新颖的框架，用于在决策层面而非预报层面评估概率预报的性能。我们运用决策校准框架，在各种与天气相关的决策任务上比较了机器学习模型与经典数值天气预报模型的表现，尽管该框架也适用于任何预报模型集合。我们发现，模型在预报层面的性能并不能可靠地转化为下游决策中的表现：某些性能差异只有在决策层面才会显现，甚至在看似……

    arXiv:2512.14779v2 Announce Type: replace  Abstract: Standard weather forecast evaluations focus on the forecaster's perspective and on a statistical assessment comparing forecasts and observations. In practice, however, forecasts are used to make decisions, so it seems natural to take the decision-maker's perspective and quantify the value of a forecast by its ability to improve decision-making. Decision calibration provides a novel framework for evaluating probabilistic forecast performance at the decision level rather than the forecast level. We evaluate decision calibration to compare a Machine Learning and a classical numerical weather prediction model on various weather-dependent decision tasks, though the framework is applicable to any set of forecast models. We find that model performance at the forecast level does not reliably translate to performance in downstream decision-making: some performance differences only become apparent at the decision level, and even among seemingl
    
[^212]: 回声状态网络中的分形与混沌激活函数：预处理拓扑决定回声状态特性

    Fractal and Chaotic Activation Functions in Echo State Networks: Preprocessing Topology Governs the Echo State Property

    [https://arxiv.org/abs/2512.14675](https://arxiv.org/abs/2512.14675)

    该论文发现回声状态网络中的分形与混沌等非光滑激活函数（如康托尔函数）不仅能保持回声状态特性，还能在谱半径容忍度（高达ρ=10）和收敛速度（快2.6倍）上超越传统光滑激活函数。

    

    当代储层计算严重依赖全局Lipschitz连续、性质良好的激活函数，这限制了其在国防、灾害响应和药物建模等需要在极端条件下稳健运行的领域中的应用。我们系统性地研究了回声状态网络中的非光滑激活函数，包括混沌、随机和分形变体。通过对36,610个储层配置的参数扫描，我们证明了几种非光滑函数不仅能保持与回声状态特性（ESP）一致的行为，而且在收敛速度和谱半径容忍度方面优于传统的光滑激活函数。值得注意的是，康托尔函数（处处连续、几乎处处导数为零）在谱半径高达 ρ = 10 时仍能保持与ESP一致的行为，比传统函数的典型界限高出一个数量级，同时收敛速度比传统函数快2.6倍。

    arXiv:2512.14675v2 Announce Type: replace  Abstract: Contemporary reservoir computing relies heavily on globally Lipschitz, well-behaved activation functions, limiting applications in defense, disaster response, and pharmaceutical modeling where robust operation under extreme conditions is critical. We systematically investigate non-smooth activation functions, including chaotic, stochastic, and fractal variants, in echo state networks. Through parameter sweeps across 36,610 reservoir configurations, we demonstrate that several non-smooth functions not only maintain behavior consistent with the Echo State Property (ESP) but outperform traditional smooth activations in convergence speed and spectral radius tolerance.   Notably, the Cantor function (continuous everywhere, zero derivative almost everywhere) maintains ESP-consistent behavior up to spectral radii of rho = 10, an order of magnitude beyond typical bounds for traditional functions, while achieving 2.6x faster convergence than 
    
[^213]: 基于协同神经网络实现约束感知配合比生成的高性能混凝土部分逆向设计

    Partial Inverse Design of High-Performance Concrete Using Cooperative Neural Networks for Constraint-Aware Mix Generation

    [https://arxiv.org/abs/2512.06813](https://arxiv.org/abs/2512.06813)

    本文提出一种协同神经网络框架，用于高性能混凝土的部分逆向设计，能够在单次前向传播中生成满足约束且性能一致的配合比，无需针对不同约束场景重新训练。

    

    高性能混凝土（HPC）的配合比设计涉及相互依赖的变量和实际约束，决策过程十分复杂。尽管数据驱动方法已经改进了混凝土工程中正向设计的预测建模，但逆向设计仍然受限，尤其是在部分变量被固定、只需推断其余变量的情况下。本研究提出了一种用于高性能混凝土部分逆向设计的协同神经网络框架。该框架将插补模型与替代强度预测器相结合，并通过协同训练进行学习。训练完成后，该框架可在单次前向传播中生成有效且性能一致的配合比设计，无需针对不同约束场景进行重新训练。与自编码器模型以及基于高斯过程替代模型的贝叶斯推断等基线模型相比，所提出的方法在替代模型预测强度与目标强度之间实现了一致性

    arXiv:2512.06813v3 Announce Type: replace-cross  Abstract: High-performance concrete (HPC) requires complex mix design decisions involving interdependent variables and practical constraints. While data-driven methods have improved predictive modeling for forward design in concrete engineering, inverse design remains limited, especially when some variables are fixed and only the remaining ones must be inferred. This study proposes a cooperative neural network framework for the partial inverse design of HPC. The framework integrates an imputation model with a surrogate strength predictor and learns through cooperative training. Once trained, it generates valid and performance-consistent mix designs in a single forward pass without retraining for different constraint scenarios. Compared with baseline models, including autoencoder models and Bayesian inference with Gaussian process surrogates, the proposed method achieves strength consistency between the surrogate-predicted strength of the
    
[^214]: 在TRIGA Mark II反应堆上利用浅层循环解码器实现约束感知与可靠状态估计

    Constrained Sensing and Reliable State Estimation with Shallow Recurrent Decoders on a TRIGA Mark II Reactor

    [https://arxiv.org/abs/2510.12368](https://arxiv.org/abs/2510.12368)

    本文将浅层循环解码器网络从概念反应堆设计拓展到实际部署的TRIGA Mark II研究堆，证明了该数据驱动方法能够利用稀疏、含噪的传感器数据实现可靠的反应堆状态估计。

    

    浅层循环解码器网络是一种新颖的数据驱动方法，能够在核反应堆等工程系统中提供精确的状态估计。这种深度学习架构是一种鲁棒性强的技术，旨在将少量稀疏测量的时间轨迹映射到完整的状态空间（包括不可观测的场），它对传感器位置不敏感，并能够通过集成策略处理含噪数据，同时凭借训练时间短的优势而无需超参数调优。该架构已成功应用于熔盐快堆概念设计；现在，这项工作考察了浅层循环解码器在实际部署的反应堆上的性能。其底层模型由TRIGA Mark II研究堆的流体动力学模型表示；该架构既使用了来自数值模型的合成温度数据，也利用了实验温度数据。

    arXiv:2510.12368v2 Announce Type: replace-cross  Abstract: Shallow Recurrent Decoder networks are a novel data-driven methodology able to provide accurate state estimation in engineering systems, such as nuclear reactors. This deep learning architecture is a robust technique designed to map the temporal trajectories of a few sparse measures to the full state space, including unobservable fields, which is agnostic to sensor positions and able to handle noisy data through an ensemble strategy, leveraging the short training times and without the need for hyperparameter tuning. The architecture was successfully applied to the Molten Salt Fast Reactor concept; now, this work considers the performance of Shallow Recurrent Decoders on a deployed reactor concept. The underlying model is represented by a fluid dynamics model of the TRIGA Mark II research reactor; the architecture will use both synthetic temperature data coming from the numerical model and leveraging experimental temperature dat
    
[^215]: WaveletDiff：用于时间序列生成的多层级小波扩散模型

    WaveletDiff: Multilevel Wavelet Diffusion For Time Series Generation

    [https://arxiv.org/abs/2510.11839](https://arxiv.org/abs/2510.11839)

    该论文提出WaveletDiff框架，直接在小波系数上训练扩散模型，并结合跨层级注意力机制与基于帕塞瓦尔定理的能量约束，利用时间序列固有的多分辨率结构生成高质量合成时间序列。

    

    时间序列在许多涉及预测、分类和因果推断任务的应用中无处不在，例如医疗保健、金融、音频信号处理和气候科学。然而，大规模、高质量的时间序列数据集仍然稀缺。合成生成可以解决这一局限性；然而，目前局限于时域或频域的模型难以再现真实世界时间序列固有的多尺度结构。我们提出了WaveletDiff，这是一个新框架，它直接在小波系数上训练扩散模型，以利用时间序列数据固有的多分辨率结构。该模型将针对每个分解层级的专用Transformer与跨层级注意力机制相结合，通过自适应门控实现时间尺度和频率尺度之间的选择性信息交换。该模型还受到基于帕塞瓦尔定理的层级特定能量约束的指导。

    arXiv:2510.11839v3 Announce Type: replace  Abstract: Time series are ubiquitous in many applications that involve forecasting, classification and causal inference tasks, such as healthcare, finance, audio signal processing and climate sciences. Still, large, high-quality time series datasets remain scarce. Synthetic generation can address this limitation; however, current models confined either to the time or frequency domains struggle to reproduce the inherently multi-scaled structure of real-world time series. We introduce WaveletDiff, a new framework that trains diffusion models directly on wavelet coefficients to exploit the inherent multi-resolution structure of time series data. The model combines dedicated transformers for each decomposition level with cross-level attention mechanisms that enable selective information exchange between temporal and frequency scales through adaptive gating. It is also informed by level-specific energy constraints based on Parseval's theorem which 
    
[^216]: 基于梯度的时序分类模型捷径检测方法

    Gradient-based Model Shortcut Detection for Time Series Classification

    [https://arxiv.org/abs/2510.10075](https://arxiv.org/abs/2510.10075)

    该论文首次研究并建立了时间序列分类中深度模型基于点的捷径学习问题，并提出了一种基于梯度的模型捷径检测方法，以揭示模型对训练数据内部虚假相关性的依赖。

    

    在过去二十年中，深度学习模型在时间序列分类（TSC）任务中吸引了大量研究关注。近年来，深度神经网络（DNN）已超越经典的基于距离的方法，取得了最先进的性能。尽管性能令人瞩目，但深度神经网络已被证明会依赖训练数据中存在的虚假相关性，这可能阻碍其泛化能力。例如，如果训练集中大多数猫都躺在草地背景中，模型可能会错误地将“草的出现”与“猫”这一标签关联起来。然而，深度神经网络在时间序列中的捷径行为仍未得到充分探索。现有的捷径研究大多依赖于性别、患者群体等外部属性，而非关注时间序列模型内部的偏差行为。在本文中，我们迈出了第一步，研究并建立基于点的捷径学习机制（摘要在此处截断）。

    arXiv:2510.10075v2 Announce Type: replace-cross  Abstract: Deep learning models have attracted lots of research attention in time series classification (TSC) task in the past two decades. Recently, deep neural networks (DNN) have surpassed classical distance-based methods and achieved state-of-the-art performance. Despite their promising performance, deep neural networks (DNNs) have been shown to rely on spurious correlations present in the training data, which can hinder generalization. For instance, a model might incorrectly associate the presence of grass with the label ``cat" if the training set have majority of cats lying in grassy backgrounds. However, the shortcut behavior of DNNs in time series remain under-explored. Most existing shortcut work are relying on external attributes such as gender, patients group, instead of focus on the internal bias behavior in time series models.   In this paper, we take the first step to investigate and establish point-based shortcut learning b
    
[^217]: GraphMend：修复 PyTorch 2 中图断裂的代码变换

    GraphMend: Code Transformations for Fixing Graph Breaks in PyTorch 2

    [https://arxiv.org/abs/2509.16248](https://arxiv.org/abs/2509.16248)

    GraphMend 是一种通过 AST 级程序分析与语义保持验证的代码变换来自动修复 PyTorch 2 中 FX 图断裂的编译器技术，使 PyTorch 能够捕获更大、无中断的计算图而无需开发者手动重构。

    

    本文提出了 GraphMend，一种自动修复 PyTorch 2 程序中 FX 图断裂的编译器技术。尽管 PyTorch 2 引入了 TorchDynamo 和 TorchInductor 来实现即时图编译，但某些代码模式仍会导致图断裂，迫使执行回退到 Python 急切执行模式，从而引入高昂的 CPU-GPU 同步开销并减少优化机会。我们对 195 个 Hugging Face 模型的调查发现，13.8% 的模型存在图断裂。GraphMend 通过源代码级程序分析和变换自动消除可修复的图断裂。它分析 AST 级别的程序结构以识别图断裂模式，并且仅在能够静态证明语义保持的情况下才应用变换。这些变换使 PyTorch 能够捕获更大、无中断的 FX 图，而无需开发者手动重构代码。我们在全部 27 个模型上对 GraphMend 进行了评估。

    arXiv:2509.16248v5 Announce Type: replace-cross  Abstract: This paper presents GraphMend, a compiler technique that automatically fixes FX graph breaks in PyTorch 2 programs. Although PyTorch 2 introduced TorchDynamo and TorchInductor to enable just-in-time graph compilation, certain code patterns still cause graph breaks that force execution to fall back to Python eager mode, introducing costly CPU-GPU synchronization and reducing optimization opportunities. Our investigation of 195 Hugging Face models reveals that 13.8% of models exhibit graph breaks. GraphMend automatically eliminates fixable breaks through source-level program analysis and transformations. It analyzes AST-level program structure to identify graph-break patterns and applies transformations only when their semantic preservation can be statically established. These transformations enable PyTorch to capture larger, uninterrupted FX graphs without manual refactoring by developers. We evaluate GraphMend on all 27 models 
    
[^218]: 基于深度学习的生物纳米孔肽分类

    Deep Learning-Driven Peptide Classification in Biological Nanopores

    [https://arxiv.org/abs/2509.14029](https://arxiv.org/abs/2509.14029)

    该论文提出通过连续小波变换将纳米孔电阻脉冲转换为尺度图，把肽识别问题转化为图像分类任务，从而利用深度卷积模型提升生物纳米孔中肽分类的准确率。

    

    基于纳米孔的单分子传感是实现快速、低成本疾病诊断和蛋白质测序的一条有前景的途径：当肽或蛋白质等分析物穿过纳米级孔道时，它会调制离子电流，产生电阻脉冲，其特征由分析物的结构及其与孔道的相互作用决定。然而，将这些特征信号转化为可靠的分子身份识别，仍是一个非常适合机器学习解决的开放性问题，因为信号噪声大、会因实验条件不同而产生变化，且难以特征化，这迄今为止限制了分类准确率。在本研究中，我们通过连续小波变换将每个电阻脉冲转换为尺度图，从而将肽识别问题转化为图像分类任务，这种表示形式能够联合编码振幅、频率和时间信息，非常适合深度卷积模型处理。在一个数据集上……（原文摘要在此处截断）

    arXiv:2509.14029v2 Announce Type: replace  Abstract: Nanopore-based single-molecule sensing is a promising route to fast, low-cost disease diagnosis and protein sequencing: as an analyte such as a peptide or protein traverses a nanoscale pore, it modulates the ionic current, producing a resistive pulse whose signature is determined by the analyte's structure and its interactions with the pore. Translating these signatures into reliable molecular identities, however, is an open problem well suited for machine learning, as the signals are noisy, suffer from variations due to experimental conditions, and are difficult to featurize, which has so far limited classification accuracy. Here we translate the peptide identification problem into an image-classification task by transforming each resistive pulse into a scaleogram via the continuous wavelet transform, a representation that jointly encodes amplitude, frequency, and time in a form well suited for deep convolutional models. On a datase
    
[^219]: 基于浅层循环解码器网络的循环燃料反应堆高效参数化状态估计研究

    Towards Efficient Parametric State Estimation in Circulating Fuel Reactors with Shallow Recurrent Decoder Networks

    [https://arxiv.org/abs/2503.08904](https://arxiv.org/abs/2503.08904)

    本研究利用浅层循环解码器网络，仅凭三个堆芯外中子通量时间序列测量数据，即可高效估计循环燃料反应堆的完整状态向量（包括中子通量、先驱核浓度、温度、压力和速度），并将该架构扩展至参数化情形。

    

    数据驱动方法的最新发展为工程系统的精确状态重构开辟了新的方法论；由于强耦合物理过程的复杂性以及极端恶劣严酷的环境，核反应堆是这项任务中特别具有挑战性的应用对象，尤其是对于第四代反应堆等新技术而言。数据驱动技术可以融合不同来源的信息，包括计算代理模型和系统上的局部噪声测量数据，从而鲁棒地估计系统状态。这项工作利用新颖的浅层循环解码器（SHRED）架构，仅从三个堆芯外的时间序列中子通量测量数据出发，推断出反应堆的整个状态向量（包括中子通量、先驱核浓度、温度、压力和速度）。特别地，这项工作将标准架构扩展为能够处理参数化时间…（摘要在此处截断）

    arXiv:2503.08904v3 Announce Type: replace  Abstract: The recent developments in data-driven methods have paved the way to new methodologies to provide accurate state reconstruction of engineering systems; nuclear reactors represent particularly challenging applications for this task due to the complexity of the strongly coupled physics involved and the extremely harsh and hostile environments, especially for new technologies such as Generation-IV reactors. Data-driven techniques can combine different sources of information, including computational proxy models and local noisy measurements on the system, to robustly estimate the state. This work leverages the novel Shallow Recurrent Decoder architecture to infer the entire state vector (including neutron fluxes, precursors concentrations, temperature, pressure and velocity) of a reactor from three out-of-core time-series neutron flux measurements alone. In particular, this work extends the standard architecture to treat parametric time-
    
[^220]: 用于推荐的图基础模型：一项全面综述

    Graph Foundation Models for Recommendation: A Comprehensive Survey

    [https://arxiv.org/abs/2502.08346](https://arxiv.org/abs/2502.08346)

    该综述首次全面梳理了图基础模型（GFM）在推荐系统中的应用，提出了现有方法的清晰分类体系，深入剖析了融合图神经网络与大语言模型优势的技术细节，并指出了该领域的关键挑战与未来研究方向。

    

    推荐系统（RS）是浏览海量在线信息的基础工具，深度学习的进步在提高排序准确性方面发挥着日益重要的作用。其中，图神经网络（GNN）擅长提取高阶结构信息，而大语言模型（LLM）则旨在处理和理解自然语言，这两种方法都因此高效且被广泛采用。近期的研究聚焦于图基础模型（GFM），它整合了GNN和LLM的优势，通过利用用户-物品关系的图结构以及文本理解能力，更高效地对复杂的推荐系统问题进行建模。在这篇综述中，我们通过引入当前方法的清晰分类体系、深入探讨方法论细节，并强调关键挑战与未来方向，对基于GFM的推荐系统技术进行了全面概述。

    arXiv:2502.08346v4 Announce Type: replace-cross  Abstract: Recommender systems (RS) serve as a fundamental tool for navigating the vast expanse of online information, with deep learning advancements playing an increasingly important role in improving ranking accuracy. Among these, graph neural networks (GNNs) excel at extracting higher-order structural information, while large language models (LLMs) are designed to process and comprehend natural language, making both approaches highly effective and widely adopted. Recent research has focused on graph foundation models (GFMs), which integrate the strengths of GNNs and LLMs to model complex RS problems more efficiently by leveraging the graph-based structure of user-item relationships alongside textual understanding. In this survey, we provide a comprehensive overview of GFM-based RS technologies by introducing a clear taxonomy of current approaches, diving into methodological details, and highlighting key challenges and future direction
    
[^221]: TSMini：一个简单却极其高效的轨迹相似性学习模型

    TSMini: A Simple Yet Highly Effective Trajectory Similarity Learning Model

    [https://arxiv.org/abs/2502.00285](https://arxiv.org/abs/2502.00285)

    提出了TSMini模型，通过子视图建模机制和基于k近邻的损失函数，同时学习轨迹的绝对相似值和相对相似排序，实现了高精度的轨迹相似性逼近。

    

    轨迹相似性是许多时空数据挖掘应用的基础。最近的研究提出了深度学习模型来逼近传统的轨迹相似性度量，利用其训练后的快速推理时间。尽管已有研究表明其推理效率很高，但由于在轨迹粒度建模以及利用训练数据中的相似性信号方面存在困难，相似性逼近精度仍面临挑战。为填补这一空白，我们提出了TSMini，一个高效的轨迹相似性模型，它包含子视图建模机制和基于k近邻的损失函数。前者使模型能够学习多粒度的轨迹模式，后者引导TSMini不仅学习轨迹之间的绝对相似性值，还学习它们的相对相似性排序。这些创新共同实现了高精度的轨迹相似性逼近。实验表明，TSMini的性能优于……（原文摘要至此截断）

    arXiv:2502.00285v3 Announce Type: replace  Abstract: Trajectory similarity is fundamental to many spatio-temporal data mining applications. Recent studies propose deep learning models to approximate conventional trajectory similarity measures, exploiting their fast inference time once trained. Although efficient inference has been reported, challenges remain in similarity approximation accuracy due to difficulties in trajectory granularity modeling and in exploiting similarity signals in training data. To fill this gap, we propose TSMini, a highly effective trajectory similarity model with a sub-view modeling mechanism and a k nearest neighbor-based loss. The former enables learning multi-granularity trajectory patterns, while the latter guides TSMini to learn not only absolute similarity values between trajectories but also their relative similarity ranks. Together, these innovations enable highly accurate trajectory similarity approximation. Experiments show that TSMini outperforms t
    
[^222]: DeltaGNN：具有信息流控制的图神经网络

    DeltaGNN: Graph Neural Network with Information Flow Control

    [https://arxiv.org/abs/2501.06002](https://arxiv.org/abs/2501.06002)

    本文提出DeltaGNN，通过创新的信息流分数度量实现线性复杂度的信息流控制，同时解决图神经网络中的过度平滑和过度压缩问题。

    

    图神经网络（GNNs）是通过消息传递过程中的递归邻域聚合来处理图结构数据的流行深度学习模型。当应用于半监督节点分类时，消息传递使GNNs能够理解短程空间交互，但也导致它们遭受过度平滑和过度压缩的问题。这些挑战阻碍了模型的表达能力，并阻止使用更深层模型来捕捉图内的长程节点交互（LRIs）。流行的LRIs检测解决方案要么因高时间复杂度而难以处理大规模图，要么无法在不同图结构上泛化。为解决这些限制，我们提出了一种称为“信息流控制”的机制，该机制利用一种名为“信息流分数”的新型连通性度量，以线性计算复杂度应对过度平滑和过度压缩问题。

    arXiv:2501.06002v3 Announce Type: replace  Abstract: Graph Neural Networks (GNNs) are popular deep learning models designed to process graph-structured data through recursive neighborhood aggregations in the message passing process. When applied to semi-supervised node classification, the message-passing enables GNNs to understand short-range spatial interactions, but also causes them to suffer from over-smoothing and over-squashing. These challenges hinder model expressiveness and prevent the use of deeper models to capture long-range node interactions (LRIs) within the graph. Popular solutions for LRIs detection are either too expensive to process large graphs due to high time complexity or fail to generalize across diverse graph structures. To address these limitations, we propose a mechanism called \emph{information flow control}, which leverages a novel connectivity measure, called \emph{information flow score}, to address over-smoothing and over-squashing with linear computationa
    
[^223]: 基于超图神经网络的超边异常检测

    Hyperedge Anomaly Detection with Hypergraph Neural Network

    [https://arxiv.org/abs/2412.05641](https://arxiv.org/abs/2412.05641)

    提出了一种基于超图神经网络的端到端无监督模型，能够无需标注数据即可检测超图中异常的高阶关联。

    

    超图是一种能够对数据实体之间高阶关联进行建模的数据结构。传统的图结构数据只能表示成对关系，而超图可以将任意数量的实体关联起来，这在许多现实应用中至关重要。超图学习算法已经在节点分类、链接预测等众多问题场景中得到了充分研究。然而，针对超图异常检测的研究却相对较少。异常检测旨在识别偏离常规模式的事件，将其应用于超图可以检测异常的高阶关联。在本工作中，我们提出了一种基于超图神经网络的端到端模型，用于识别超图中的异常关联。我们提出的算法以无监督方式运行，无需任何标注数据。我们在多个真实数据集上进行了大量实验……

    arXiv:2412.05641v2 Announce Type: replace-cross  Abstract: Hypergraph is a data structure that enables us to model higher-order associations among data entities. Conventional graph-structured data can represent pairwise relationships only, whereas hypergraph enables us to associate any number of entities, which is essential in many real-life applications. Hypergraph learning algorithms have been well-studied for numerous problem settings, such as node classification, link prediction, etc. However, much less research has been conducted on anomaly detection from hypergraphs. Anomaly detection identifies events that deviate from the usual pattern and can be applied to hypergraphs to detect unusual higher-order associations. In this work, we propose an end-to-end hypergraph neural network-based model for identifying anomalous associations in a hypergraph. Our proposed algorithm operates in an unsupervised manner without requiring any labeled data. Extensive experimentation on several real-
    
[^224]: 混合模型的可解释聚类

    Explainable Clustering of Mixture Models

    [https://arxiv.org/abs/2411.01576](https://arxiv.org/abs/2411.01576)

    本文首次从混合模型视角研究可解释聚类问题，给出了可解释性代价的首个数据相关上界，并针对具有次指数尾部的混合模型的K-中位数聚类提出了新算法。

    

    可解释聚类问题由Moshkovitz等人（ICML 2020）首次提出，研究具有K个叶子的轴对齐决策树能在多大程度上近似给定的聚类。树的性能通过“可解释性代价”来衡量，其定义为树的聚类代价（其中每个叶子对应一个簇）与最优代价之间的比值。最近的一些工作针对不同的代价函数给出了可解释性代价的最坏情况刻画。然而，这些保证与数据无关，因此在实际聚类场景中表现得极其悲观。本文从混合模型的视角研究可解释聚类问题，这使得我们能够首次给出可解释性代价的数据相关界。首先，我们关注具有次指数尾部的混合模型的K-中位数聚类，并提出了一种利用相关信息的算法……

    arXiv:2411.01576v3 Announce Type: replace  Abstract: The explainable clustering problem was first posed by Moshkovitz et al. (ICML 2020) and studies how well an axis-aligned decision tree with $K$ leaves can approximate a given clustering. The performance of the tree is measured via the \textit{price of explainability}, defined as the ratio between the clustering cost of the tree (where every leaf is a cluster) and the optimal cost. Several recent works have given worst-case characterizations of the price of explainability for different cost functions. However, these guarantees are data-agnostic and therefore notoriously pessimistic in practical clustering settings. In this paper, we study explainable clustering from the point of view of mixture models, which allows us to give the first data-dependent bounds on the price of explainability. First, we focus on $K$-medians clustering of mixture models with subexponential tails. We propose an algorithm that leverages information about the 
    
[^225]: 关于高斯测度的Lipschitz算子学习的样本复杂度

    The Sample Complexity of Learning Lipschitz Operators with respect to Gaussian Measures

    [https://arxiv.org/abs/2410.23440](https://arxiv.org/abs/2410.23440)

    该论文证明了关于高斯测度的Lipschitz算子具有更高阶的高斯Sobolev正则性，给出了Hermite多项式逼近误差的上下界，并紧致刻画了从（可能自适应的）线性样本重构Lipschitz算子的样本复杂度。

    

    算子学习，即利用机器学习来逼近无穷维函数空间之间的映射，近年来受到越来越多的研究关注。算子逼近可以作为计算科学与工程问题的高效代理模型，与传统方法形成互补。然而，尽管其在实践中取得了成功，我们对其底层数学理论的理解在很大程度上仍不完整。本文研究了关于高斯测度的Lipschitz算子的逼近问题。我们证明了Lipschitz算子具有更高阶的高斯Sobolev正则性，并建立了Hermite多项式逼近误差的下界与上界。随后，我们研究了从m个任意（可能是自适应的）线性样本重构Lipschitz算子的一般性策略。作为一个关键发现，我们紧致地刻画了相应的样本复杂度，即……

    arXiv:2410.23440v4 Announce Type: replace  Abstract: Operator learning, the approximation of mappings between infinite-dimensional function spaces using machine learning, has gained increasing research attention in recent years. Operator approximations can serve as efficient surrogate models for problems in computational science and engineering, complementing traditional methods. However, despite their empirical success, our understanding of the underlying mathematical theory is in large part still incomplete. In this paper, we study the approximation of Lipschitz operators with respect to Gaussian measures. We prove higher Gaussian Sobolev regularity of Lipschitz operators and establish lower and upper bounds on the Hermite polynomial approximation error. We then study general reconstruction strategies of Lipschitz operators from $m$ arbitrary (potentially adaptive) linear samples. As a key finding, we tightly characterize the corresponding sample complexity, that is, the smallest ach
    
[^226]: 基于大语言模型的小分子优化

    Small Molecule Optimization with Large Language Models

    [https://arxiv.org/abs/2407.18897](https://arxiv.org/abs/2407.18897)

    提出Mol-E——一种利用分子语言模型生成能力的进化算法，在实用分子优化基准测试中于任务无关和任务知情两种模式下均创造了新的最先进记录。

    

    分子优化，即设计具有理想性质的分子的过程，是药物发现领域的一项关键挑战。大语言模型（LLM）的最新进展为其与传统分子优化算法相结合以提升性能开辟了新的机遇。在这项工作中，我们提出了分子语言模型驱动的进化算法，这是一种依赖于在分子和分子性质上训练的大语言模型生成能力的进化算法。科学贡献：Mol-E在实用分子优化基准测试中创造了新的最先进结果，在任务无关模式（其中目标函数被严格视为黑盒）中取得了17.500的Top-10 AUC总和值，在任务知情模式（其中优化器获得目标的固定语义描述）中取得了20.551的Top-10 AUC总和值。Mol-E还优于所评估的所有基线方法。

    arXiv:2407.18897v2 Announce Type: replace  Abstract: Molecular optimization, the process of designing molecules with desirable properties, represents a critical challenge in drug discovery. Recent advancements in large language models (LLMs) have opened new opportunities for their integration with traditional molecular optimization algorithms to improve performance. In this work, we propose Molecular Language Model powered Evolutionary Algorithm (Mol-E), an evolutionary algorithm that relies on the generative capabilities of LLMs trained on molecules and molecular properties.   Scientific Contribution. Mol-E establishes new state-of-the-art results on the Practical Molecular Optimization benchmark, with summed Top-10 AUC values of 17.500 in the task-agnostic regime, in which the oracle is treated strictly as a black box, and 20.551 in the task-informed regime, in which the optimizer receives a fixed semantic description of the objective. Mol-E also improves over the evaluated baselines
    
[^227]: 基于生成式人工智能的程序化内容生成

    Procedural Content Generation via Generative Artificial Intelligence

    [https://arxiv.org/abs/2407.09013](https://arxiv.org/abs/2407.09013)

    本综述系统回顾了生成式人工智能在程序化内容生成中的应用（涵盖地形、物品和故事情节等内容），并重点探讨了定制化内容处理、质量与多样性保证以及训练数据不足等关键挑战，以及应对这些挑战的创新生成技术、模型架构和适用于有限数据场景的方法。

    

    过去人们已经尝试在程序化内容生成（PCG）中利用机器学习。在这篇综述论文中，我们研究了自2010年代中期以来受到广泛关注的生成式人工智能（AI）是如何被应用于PCG的。我们回顾了生成式AI在创建各类内容方面的应用，包括地形、物品，甚至故事情节。虽然生成式AI对PCG十分有效，但构建高性能模型不仅需要处理定制化内容并确保质量与多样性，还需要获得充足的训练数据。为了推动PCG研究的进一步发展，解决这些挑战至关重要。因此，我们还特别关注了那些探索创新生成技术、模型架构以及适用于有限数据场景的方法的研究。

    arXiv:2407.09013v3 Announce Type: replace  Abstract: The attempt to utilize machine learning in procedural content generation (PCG) has been made in the past. In this survey paper, we investigate how generative artificial intelligence (AI), which saw a significant increase in interest in the mid-2010s, is being used for PCG. We review applications of generative AI for the creation of various types of content, including terrains, items, and even storylines. While generative AI is effective for PCG, building high-performance models requires not only handling customized content and ensuring quality and diversity, but also securing sufficient training data. For PCG research to advance further, addressing these challenges is essential. Thus, we also give special consideration to research that explores innovative generation techniques, model architectures, and approaches suited for limited-data scenarios.
    
[^228]: 自监督编码器在未见数据集上聚类的实证研究

    An Empirical Study into Clustering of Unseen Datasets with Self-Supervised Encoders

    [https://arxiv.org/abs/2406.02465](https://arxiv.org/abs/2406.02465)

    本研究通过将仅在ImageNet-1k上预训练的有监督与自监督编码器部署到未见数据集并进行聚类实验，发现有监督编码器在训练域内更有用，而自监督编码器在远离训练域的分布外数据上泛化能力更强，且经分类微调后这一优势会发生逆转。

    

    预训练模型能否在无需任何再训练的情况下泛化到新数据集？我们将预训练的图像模型部署在它们未曾训练过的数据集上，并研究其嵌入是否能够形成有意义的聚类。我们的基准测试实验套件使用仅在ImageNet-1k上通过有监督或自监督技术预训练的编码器，将其部署在训练期间未见过的图像数据集上，并用传统聚类算法进行聚类。这一评估为自监督模型的嵌入提供了新的见解，自监督模型所优先关注的特征与有监督模型不同。我们发现证据表明，有监督编码器在训练域内比自监督（SSL）编码器具有更大的效用，而在远离训练域的分布外场景中情况则恰恰相反。然而，针对ImageNet-1k分类任务微调后的SSL编码器会表现出相反的行为，在域内性能优于纯有监督模型，而在（域外）则有所下降。

    arXiv:2406.02465v2 Announce Type: replace-cross  Abstract: Can pretrained models generalize to new datasets without any retraining? We deploy pretrained image models on datasets they were not trained for, and investigate whether their embeddings form meaningful clusters. Our suite of benchmarking experiments uses encoders pretrained solely on ImageNet-1k with either supervised or self-supervised training techniques, deployed on image datasets that were not seen during training, and clustered with conventional clustering algorithms. This evaluation provides new insights into the embeddings of self-supervised models, which prioritize different features to supervised models. We find evidence that supervised encoders offer more utility than SSL encoders within the training domain, and vice-versa far outside of it. However, fine-tuning SSL encoders for ImageNet-1k classification results in the opposite behaviour, with better performance than supervised-only models on in-domain and decreased
    

