# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Embedding Models Measure in Peculiar Ways](https://arxiv.org/abs/2609.20821) | 该研究发现嵌入模型对质量、距离、时间和体积等物理测量的表示十分微弱且奇特，主要受表面字符串相似性的强烈影响，而重新校准相似度也无法显著改善其与真实物理测量的对齐。 |
| [^2] | [Paint-Anything: Unified Any-Color Control for Image Generation and Editing](https://arxiv.org/abs/2609.20816) | Paint-Anything通过对象级颜色监督学习共享的十六进制提示接口，结合Paint-500K数据集和精确匹配的纯色锚点，实现了图像生成与编辑中任意精确颜色的统一控制。 |
| [^3] | [How Does Distribution Shift Shape Pretraining Gains in Neural PDE Surrogates?](https://arxiv.org/abs/2609.20814) | 本研究通过在翼型RANS数据上预训练神经PDE代理模型，量化了分布偏移的不同组成部分（湍流模型差异与翼型几何多样性）对预训练收益的影响，发现收益大小随微调样本量和偏移类型而变化。 |
| [^4] | [Quantifying Overclaiming Propensity in Frontier LLM Agents](https://arxiv.org/abs/2609.20812) | 本文提出OverclaimBench评估套件，首次量化了前沿LLM编码智能体在最终回复中“过度宣称”任务完成的倾向，并发现在67.9%的运行中智能体并未真正阅读所有被要求审查的文件。 |
| [^5] | [Score Centering Stabilizes Off-policy Reinforcement Learning](https://arxiv.org/abs/2609.20807) | 本文提出加性的“分数中心化”修正项，通过抵消训练与推理引擎间持续累积的漂移偏差，在无需高昂采样开销的情况下稳定了训练-推理失配下的大语言模型强化学习。 |
| [^6] | [An Empirical Study of Harness Design for Coding Agents](https://arxiv.org/abs/2609.20804) | 该论文通过固定执行循环并系统变化规划、动作空间和上下文管理三个组件的实证研究，发现上下文管理在上下文窗口预算紧张时价值显著提升，且其主要收益来自防止上下文溢出故障。 |
| [^7] | [PosteriorBench: From Point Estimates to Posterior Matching in Evaluating Generative Inverse Solvers](https://arxiv.org/abs/2609.20794) | 论文提出PosteriorBench基准，通过在四个物理逆问题上构建高保真参考后验分布，将生成式逆问题求解器的评估从单一重建的点估计精度提升为对真实后验分布匹配能力的评估。 |
| [^8] | [GeoAAC: Geometry-Based Adaptive Action Chunking from Denoising Trajectories in VLA Policies](https://arxiv.org/abs/2609.20776) | GeoAAC利用Flow Matching去噪轨迹的几何特征来评估动作预测的可靠性，从而在VLA策略中实现动作分块时间跨度的自适应调整。 |
| [^9] | [Calibrated RF-Fingerprinting Under Interference With Heterogeneous Transmission Protocols](https://arxiv.org/abs/2609.20765) | 本文提出在多信号同信道干扰且传输协议异构的环境下，将射频指纹识别建模为多标签分类问题并使用一维CNN求解，同时通过模型校准推导置信度阈值，为假阴性数量上限提供保证，确保不遗漏真正的频谱政策违规行为。 |
| [^10] | [Agile-WAM: An Agile Tactile World Action Model for Contact-Rich Robot Control](https://arxiv.org/abs/2609.20761) | 提出Agile-WAM，一种无需大规模预训练生成骨干的敏捷触觉世界动作模型，通过共享潜在空间中的直接视觉-触觉到动作流匹配过程，联合生成动作与未来视觉/触觉表征，实现高效灵活的接触密集型机器人控制。 |
| [^11] | [Prediction-Powered Smoothing and Validation for Disaggregated AI Evaluation](https://arxiv.org/abs/2609.20758) | 本文提出预测驱动平滑（PP-S）及其跨分类体系借力扩展（PP-TS），利用贝叶斯小区域估计方法为标签稀少领域的AI分解式评估提供精确的点估计和区间估计，并推导了新的近似无偏基于设计的交叉验证分数用于模型验证。 |
| [^12] | [OPTED: On-Policy Fine-Tuning for End-to-End Driving using a Render-Free Teacher](https://arxiv.org/abs/2609.20756) | 提出OPTED方法，通过在向量化输入上用强化学习训练特权教师来监督预训练学生模型的闭环后训练，从而无需昂贵的传感器仿真渲染即可缓解端到端驾驶策略在闭环部署中的累积误差问题。 |
| [^13] | [dQwen3.5: Hybrid-Attention Diffusion Language Models](https://arxiv.org/abs/2609.20751) | 本论文将Qwen3.5的注意力-RNN混合架构（0.8B至9B规模）适配为dQwen3.5扩散语言模型系列，证明混合架构骨干仅需全注意力模型约一半的训练词元即可达到相同损失，同时在任意顺序解码上表现相当，并在并行解码下表现出色。 |
| [^14] | [MILER: Semantic Mid-Level Representation for Sim-to-Real Reinforcement Learning in Unstructured Autonomous Driving](https://arxiv.org/abs/2609.20747) | MILER提出了一种基于语义中层表示的端到端策略框架，通过BEVFusion将真实传感器数据转换为与模拟器一致的语义鸟瞰图表示，实现了非结构化自动驾驶环境中零样本的模拟到现实强化学习迁移。 |
| [^15] | [Video DeltaNet: A Video-Native Hybrid Attention for Livestream Video Generation](https://arxiv.org/abs/2609.20744) | 提出Video DeltaNet（VDN），通过将局部Softmax注意力与引入视频增量注意力（VDA）的双向线性记忆相结合的混合架构，解决视频扩散模型中的注意力计算瓶颈，实现高质量的直播视频生成。 |
| [^16] | [Don't Mask the Environment: Observation Supervision Changes How Agents Explore Under RL](https://arxiv.org/abs/2609.20715) | 该论文提出ActObs方法，在监督微调中同时对轨迹中已有的环境观测标记进行监督，使策略学会建模动作后果，从而在不增加任何数据、参数或计算成本的情况下，显著提升后续GRPO强化学习中智能体的探索能力和pass@k性能。 |
| [^17] | [Stable Movement for Nondual Lipschitz Convex Optimization: Efficiency and Nearly Optimal Oracle Rates](https://arxiv.org/abs/2609.20701) | 该论文通过将Lipschitz凸优化归约为嵌套凸集追逐问题，为非对偶情形设计出高效算法并实现了近乎最优的一阶预言机复杂度，从而解决了COLT 2015开放问题的非光滑部分。 |
| [^18] | [TetrisCNN for interpretable detection of phases of matter from experimental quantum simulator data](https://arxiv.org/abs/2609.20693) | 本文提出TetrisCNN，一种具有俄罗斯方块状多形状滤波器并行分支的卷积神经网络，能直接从含噪声的实验量子模拟器数据中学习以自旋关联子表达的稀疏可解释潜在表征，从而实现对物质相的可解释探测。 |
| [^19] | [The First-Order Oracle Complexity of Lipschitz Convex Optimization in Nondual Settings](https://arxiv.org/abs/2609.20687) | 该论文通过一种以仿射损失最大值评估比较者的新在线学习博弈及其与顺序fat-shattering维数的联系，肯定地解决了COLT开放问题的非光滑版本，证明当p < q时ℓ_p球的几何结构可将ℓ_q-Lipschitz凸优化的收敛速率提升至Õ(1/T)，改进了经典的O(1/√T)速率。 |
| [^20] | [RISC-V and machine learning: a survey](https://arxiv.org/abs/2609.20677) | 本文系统综述了RISC-V指令集架构在机器学习领域的应用现状，提出了统一的实现分类法、性能与设计权衡的对比分析以及软件工具链成熟度的评估。 |
| [^21] | [Epidemiological Causal Graph Identification: Challenges, Identifiability and Algorithms](https://arxiv.org/abs/2609.20676) | 该论文证明了有序分布节点与指数族分布节点之间的因果边方向在一般参数值下是可识别的，将有序-泊松模型的可识别性结果推广到更广泛的指数族分布，并提出了基于分数的穷举搜索和掩码连续优化两种因果发现算法。 |
| [^22] | [Multi-center Medical Data Mining with FL-Net - A One-stop Shop for Federated Learning](https://arxiv.org/abs/2609.20650) | FL-Net是一个一站式联邦学习临床研究框架，集成了数据协调、数据发现、披露控制和容器化联邦工作流执行，填补了现有14个框架均无法完全满足的五项要求，并将在欧盟项目中覆盖10家医院的80万余名患者。 |
| [^23] | [Beyond PINNs: A Unified Gauss--Newton and Petrov--Galerkin Framework for Neural and Hybrid PDE Solvers](https://arxiv.org/abs/2609.20641) | 本文提出了一个统一的泛函高斯-牛顿离散化框架，通过对偶配对将线性测量表示为测试函数，证明高斯-牛顿系统恰为线性化问题的Petrov-Galerkin离散化，从而将PINN的逐点配点训练与有限元变分方法统一起来，并使测试函数的选择成为显式的算法设计手段。 |
| [^24] | [COIN-GP: Cooperative Online Learning in Networked Distributed Systems with Partial Measurements via Gaussian Process Regression](https://arxiv.org/abs/2609.20598) | 提出了一种结合在线分布式高斯过程回归的基于观测器的协作学习框架，可在仅有部分测量的条件下联合估计分布式传感网络中的系统状态与未知动力学，并提供了数据采集条件和误差上界。 |
| [^25] | [Recursive Quantum Long Short-Term Memory for Stable Short-Horizon Temperature Forecasting](https://arxiv.org/abs/2609.20594) | 本文提出一种递归量子长短期记忆（QLSTM）架构，通过递归量子特征变换提升混合量子-经典时间序列模型的稳定性与泛化能力，在温度预测任务中比标准QLSTM收敛更快、误差更低。 |
| [^26] | [CrystalMO-TuRBO: Multi-Objective Trust-Region Bayesian Optimization for High-precision Joint Crystal Structure Refinement](https://arxiv.org/abs/2609.20592) | 提出了CrystalMO-TuRBO，一种多目标信赖域贝叶斯优化架构，通过将X射线和中子数据的偏差作为独立目标建模，避免了手动权重调节，实现了高精度的晶体结构联合精修。 |
| [^27] | [NS3Learn: Transferring 5G NR Mode-2 Reception Realism from ns-3 to the Veins/SUMO Stack for Connected-Vehicle Safety Assessment](https://arxiv.org/abs/2609.20578) | 本研究提出NS3Learn，一个从ns-3 5G-LENA仿真数据（1050万接收样本）中学习得到的闭式模型，能在Veins/SUMO车联网仿真中无需重实现完整协议即可准确刻画5G NR sidelink Mode-2的资源竞争损失，将每时刻消息投递率预测的平均绝对偏差降至0.06，显著优于现有模型。 |
| [^28] | [TAP Accuracy Below the Fluctuation Scale and Universal Posterior Geometry in Spherical Linear Models](https://arxiv.org/abs/2609.20577) | 本文在Marchenko–Pastur谱正则性条件下，证明了球面线性模型中TAP自由能逼近精度可达 $O_P(p^{-1})$（低于自然波动尺度 $O_P(p^{-1/2})$），并在所有全局TAP最大化子上一致刻画了后验几何。 |
| [^29] | [Accelerating Visual Policy Learning with Sampling-Based Model Predictive Control](https://arxiv.org/abs/2609.20575) | 提出采样引导策略搜索方法SGPS，将基于采样的模型预测控制与一阶策略优化相结合，并采用将渲染排除在计算图之外的解耦FoPG公式，避免局部优化陷入非预期接触模式，实现单GPU上直接从深度观测高效训练视觉策略。 |
| [^30] | [Mitigating Retaliatory Algorithmic Collusion in Repeated Games](https://arxiv.org/abs/2609.20548) | 该论文提出了CURB奖励塑形框架，通过将Q-learning合谋行为与简单惩罚码理论形式化关联，利用合作与背叛历史下策略间的全变分距离检测并惩罚算法合谋，为一般性重复博弈提供了通用的合谋缓解方法。 |
| [^31] | [Parallelism, critical windows, and separations among diffusion language models](https://arxiv.org/abs/2609.20539) | 本文首次对掩码扩散、均匀扩散与高斯扩散三类主流扩散语言模型的并行生成能力进行了细粒度理论比较，证明均匀扩散和高斯扩散同样能以前向传播次数与分布对偶总相关（可远小于上下文长度）成比例的方式完成采样，而此前仅掩码扩散具备这一性质。 |
| [^32] | [Relational Attention for Data-Efficient Language Modeling](https://arxiv.org/abs/2609.20530) | 该论文提出在双注意力Transformer架构中将关系注意力与自注意力相结合，并借助BabyLM 2026挑战赛的数据受限环境，验证关系注意力所带来的数据效率能否成功迁移到语言建模任务中。 |
| [^33] | [Noise-Robust Quantum State Characterization for Remote State Preparation with Deep Learning](https://arxiv.org/abs/2609.20523) | 该论文提出了一种基于Transformer的量子态表征器模型，能够在复杂散射与动态噪声环境下从含噪测量中以超过99.999%的保真度重建远程态制备的光子偏振态，并通过注意力机制揭示测量观测量间相关性的物理解释。 |
| [^34] | [When EOS Tokens Disagree: Understanding Length Inflation in On-Policy Distillation](https://arxiv.org/abs/2609.20511) | 该论文揭示了学生模型与教师模型之间EOS终止标记不匹配是在线策略蒸馏中生成长度膨胀的关键原因，并提出将功能等价的EOS标记视为共享的语义停止动作可显著缓解该问题。 |
| [^35] | [Truncated automatic sparse differentiation for machine learning interatomic potentials](https://arxiv.org/abs/2609.20510) | 本文提出截断自动稀疏微分（ASD）方法，利用机器学习原子间势高阶导数的稀疏性与距离衰减特性，使大型系统的完整Hessian矩阵计算在计算上变得可行。 |
| [^36] | [Radio Frequency Detection and Classification of Microplastics in Water](https://arxiv.org/abs/2609.20507) | 该论文提出了一种机器学习辅助的射频介电光谱检测平台，实现了对水中多种10微米级微塑料颗粒的无标记快速检测与材料分类。 |
| [^37] | [Distributionally Robust Federated Learning with Multi-Source Data](https://arxiv.org/abs/2609.20501) | 该论文提出了一种分布鲁棒联邦学习框架，通过将全局模糊集构建为局部模糊集的可容许混合的并集，同时应对跨客户端混合不确定性与客户端内部的分布模糊性，并据此建立了高概率样本外性能保证及相应的联邦算法。 |
| [^38] | [Resolution limits for process comparison from event data](https://arxiv.org/abs/2609.20489) | 本文证明了基于事件日志随机语言的标准过程挖掘方法无法从数据中区分并发与顺序行为，刻画了这种分辨率极限，并指出可以通过活动起止时间或以对象为中心的记录等被随机语言丢弃的证据来恢复这种区分。 |
| [^39] | [Deep Learning-Based Classification of Cognitive and Resting States Using Electroencephalography Signals](https://arxiv.org/abs/2609.20467) | 该研究提出了一种将卷积神经网络（CNN）与门控循环单元（GRU）相结合的2D-Net深度学习框架，并通过时频分析提取EEG信号特征，实现了对认知状态与静息状态的有效分类。 |
| [^40] | [Training Neural Networks to Approach the Optimum Bayes Estimator in Dense Multi-Emitter Localization](https://arxiv.org/abs/2609.20465) | 该论文通过在合成帧上训练神经网络来逼近密集发射源定位的最优贝叶斯估计器，为实现高通量、大视场、超时空分辨率的单分子定位显微镜奠定了基础。 |
| [^41] | [Correlation-Free Transition Path Sampling through Shooting Point Generation Guided by Committor Learning](https://arxiv.org/abs/2609.20461) | 该论文提出利用 Committor 学习引导射击点生成，在无需预先已知反应坐标的情况下实现无关联跃迁路径采样，从而克服了传统跃迁路径采样方法中路径相关性导致效率受限的问题。 |
| [^42] | [Online Supervised Dimension Reduction with Random Features: Diagnostics and Computational Trade-offs](https://arxiv.org/abs/2609.20454) | 本文为基于随机特征的在线核监督主成分分析（OKSPCA）提供了一致性、集中性与扰动的理论诊断，并通过六个基准实验揭示：精确优化监督谱目标并不能保证得到精确的总体子空间或更优的预测表示，从而阐明了目标优化精度、子空间恢复精度与预测性能之间的计算权衡。 |
| [^43] | [Seismic Site Response Prediction from Sparse Observations Using Finite-Element-Pretrained Latent Dynamics](https://arxiv.org/abs/2609.20451) | 本研究提出FLARE-T框架，通过从密集有限元模拟中学习低维潜在动力学并用稀疏观测记录进行校准，显著提升了地震场地响应预测的准确性，并通过离心机试验和Lotung现场垂直阵列数据验证了其有效性。 |
| [^44] | [Cross-Architecture Foundation-Model Distillation for Edge Flood Segmentation](https://arxiv.org/abs/2609.20441) | 该论文通过将3亿参数的地理空间基础模型蒸馏到仅70万参数的EfficientViT-B0学生模型中，并利用教师模型对未标注Sentinel-2影像生成伪标签来扩展训练数据，实现了可在边缘设备部署的高性能洪水分割模型。 |
| [^45] | [SCGFM-ART: Amortized Relational Transport for Structure-Centric Graph Foundation Models](https://arxiv.org/abs/2609.20419) | SCGFM-ART提出了一种以结构为中心的图基础模型框架，通过摊销关系传输将任意异构图直接对齐到由有限关系基准定义的共享关系图谱坐标系上，无需代价高昂的运行时Gromov-Wasserstein优化，即可从全局和局部两个层面实现跨域统一的图表示学习。 |
| [^46] | [The Bias of Nonlinear Two-Time-scale Stochastic Approximation under Constant Step-Sizes](https://arxiv.org/abs/2609.20409) | 本文首次对常数步长下的非线性双时间尺度随机逼近给出了紧的均方误差与偏差上界 $O(\alpha+\beta^2/\alpha^2)$，并通过分离各项误差来源阐明了 $\beta^2/\alpha^2$ 偏差项的产生机制。 |
| [^47] | [Learning Principal-Agent Contracts for Equitable Smallholder Carbon Farming under Moral Hazard and Adverse Selection](https://arxiv.org/abs/2609.20404) | 本研究将面向小农户的碳农业合同设计问题建模为POMDP，利用强化学习学习在道德风险与逆向选择下的动态利润最大化合同，以弥合碳计划难以惠及小农户的现实差距。 |
| [^48] | [Model-based Bootstrap for Offline Policy Evaluation in Tabular Reinforcement Learning](https://arxiv.org/abs/2609.20389) | 该论文提出了一种基于模型的自助法框架，通过从估计的MDP重新生成轨迹来对表格型强化学习中的离线策略评估进行不确定性量化，从而克服了经典重采样方法在鲁棒性、可扩展性和有限样本有效性上的局限。 |
| [^49] | [Minimax-Optimal Online Contract Design with Unrestricted Bounded Contracts](https://arxiv.org/abs/2609.20353) | 该论文证明了在委托人仅能观察结果而无法观察行动的重复合同设计中，即使允许任意有界的支付方案、代理利润可能不连续且行动空间完全任意，极小极大遗憾仍为 $T^{m/(m+1)}$ 量级，并通过有效维度约简与显示偏好构造的单调反应映射设计出达到该最优速率的学习策略。 |
| [^50] | [COMPASS: Ordered Clustered Routing at 100K Scale](https://arxiv.org/abs/2609.20352) | COMPASS算法通过协调并行子求解器将搜索与学习加速路由相结合，解决了有序聚类旅行商问题（OCTSP），可在10万规模节点上获得持续改进的高质量解，并支持通用距离矩阵输入。 |
| [^51] | [Fast Cross-Strength Multi-Contrast Brain MRI Translation using Latent Bridge Matching](https://arxiv.org/abs/2609.20341) | 提出基于条件潜在桥接匹配的统一模型，无需任务特定架构即可实现单步推理的跨场强多对比度脑部MRI快速翻译，在MRIxFields2026挑战赛全部三项任务中取得有竞争力的结果。 |
| [^52] | [Detecting Deceptive Recruitment: A Signal-theoretic Machine Learning Framework for Early Identification of Labour Exploitation](https://arxiv.org/abs/2609.20336) | 该研究基于信号理论将欺骗性招聘广告检测形式化为分类问题，利用464个真实案例构建了融合计算机视觉、自然语言处理和语义嵌入的多模态机器学习框架，为强迫劳动的早期识别提供了实证验证的有效方法。 |
| [^53] | [Sharp Reconstruction Bounds for Autoencoders Using the Same Forward Map](https://arxiv.org/abs/2609.20333) | 该论文证明了在雅可比奇异值受约束条件下，使用相同前向映射的自编码器重构导数误差的精确下界，并通过真实地面激光雷达森林扫描数据实验验证了该理论界的预测能力。 |
| [^54] | [Near-Optimal Pure Single-Loop Extragradient Method for Strongly Convex--Strongly Concave Minimax Optimization](https://arxiv.org/abs/2609.20327) | 提出了一种参数固定的纯单循环阻尼外梯度方法用于强凸-强凹极小极大优化，无需内层求解或重启策略，即可以 O(√(κ_x κ_y) log(1/ε)) 的近最优梯度查询复杂度实现末次迭代线性收敛。 |
| [^55] | [QEncodeBench: Can Large Language Models Encode Classical Problems into Verified Quantum Oracles?](https://arxiv.org/abs/2609.20319) | 该论文提出QEncodeBench基准，首次通过对抗性自验证的完整解集等价性检查，系统测量大语言模型将经典约束问题编码为经验证量子相位预言机的能力，发现无推理模式的代码模型几乎完全失败，而启用原生推理可使准确率提升一个数量级。 |
| [^56] | [EviRec: Continual Evidence Learning for Dual Cold-Start POI Recommendation](https://arxiv.org/abs/2609.20313) | 提出了EviRec持续证据学习框架，通过匹配、转移记忆和生命周期三个互补视角评估每个候选POI历史证据的可信度，以解决用户与POI同时更替带来的双重冷启动推荐问题。 |
| [^57] | [ZeroHAT: Behavior-Conditioned Zero-Shot Human Activity Trace Generation](https://arxiv.org/abs/2609.20310) | ZeroHAT提出了一种行为条件化的零样本框架，通过从拥有真实数据的源区域迁移人类行为模式，并结合目标区域公开可获得的上下文信息，为缺乏真实人类活动轨迹数据的区域生成合成数据。 |
| [^58] | [Hypernetwork-Parameterized Spatially Adaptive Neural Operators for PDE Learning](https://arxiv.org/abs/2609.20309) | 提出空间自适应神经算子SANO，利用坐标条件超网络为每个空间位置生成独立的算子参数，解决了传统神经算子在边界和高梯度区域欠拟合导致误差累积的问题。 |
| [^59] | [SAGG: Sample-Adaptive Gradient Gating for Robust Multimodal Learning under Heterogeneous Corruption](https://arxiv.org/abs/2609.20302) | 该论文揭示了多模态学习中样本级异质损坏会使共享参数的梯度调制方法产生不可约偏差，并证明样本级二元门控是唯一无偏策略，据此提出SAGG方法，通过在线特征范数质量测试实现样本自适应的梯度门控与方差截断，达到标准O(1/√T)收敛速率。 |
| [^60] | [Improving Generalization and Robustness in Offline Reinforcement Learning via Boundary-Aware Data Augmentation](https://arxiv.org/abs/2609.20300) | 该论文从理论上揭示了随机片段插值训练的误差与状态间距离呈正相关，并据此提出边界感知数据增强方法，以提升离线强化学习在分布内的泛化能力和鲁棒性。 |
| [^61] | [A Learning Algorithm for Threshold Boolean Networks with Prescribed Fixed Points](https://arxiv.org/abs/2609.20298) | 该论文提出了一种基于自定义可微损失函数的学习算法，能够从指定的不动点集合中准确推断阈值布尔网络，在拟南芥基因调控模型上实现了无虚假吸引子的完美重建，性能显著优于感知机和逻辑回归等标准方法。 |
| [^62] | [Intact-to-Amputee Transfer in Surface-EMG Gesture Decoding: Training Source and Calibration Budget](https://arxiv.org/abs/2609.20297) | 研究发现表面肌电手势解码的零样本跨人群迁移完全失败，但基于四十名健全受试者训练的跨用户编码器在仅需三次标注校准的情况下，对截肢者的解码性能显著优于传统单用户分类器（宏F1达0.779对0.589），且健全者训练数据比少量截肢者数据产生更好的迁移效果。 |
| [^63] | [Personalising a Cross-User Surface Electromyography Encoder Under a Small Calibration Budget](https://arxiv.org/abs/2609.20296) | 本研究将个性化跨用户表面肌电编码器视为有代价的设计决策，在77名受试者上比较了四种利用少量校准重复数据的方法，发现全微调在所有预算水平下都始终最为准确。 |
| [^64] | [TinyCNN: A 193K-Parameter Network for On-Device Plant Disease Detection, with a Cross-Dataset Robustness Diagnosis](https://arxiv.org/abs/2609.20290) | 本文提出仅含19.3万参数的轻量级卷积网络TinyCNN，通过深度可分离卷积在PlantVillage数据集上实现98.88%的准确率，规模比ResNet18小约58倍，可在低成本移动设备上直接部署进行植物病害检测。 |
| [^65] | [Labeled Incidence Structures for Native Transformer Modeling of Text, Knowledge Graphs, and Hypergraphs](https://arxiv.org/abs/2609.20278) | 本文提出标记关联结构（LIS），将文本、知识图谱和超图统一编码为（内容、槽位、关系实例）三元组表示，使单个标准Transformer无需展平数据即可原生处理这三种异构数据类型。 |
| [^66] | [A Table-Free Index for Tapered Memoization Grids: Compact Out-of-Core Evaluation of Functions of Sorted Arguments](https://arxiv.org/abs/2609.20276) | 该论文证明锥形记忆化网格的键集恰为多重集组合，因而可基于经典组合数系统实现无表闭式 O(d) 排名与逆排名索引，消除了原方案的预处理表，支持并行构建与无键存储，形成仅存值的扁平数组结构，在 3740 万条目规模下比哈希映射记忆化节省 5.7 倍内存。 |
| [^67] | [AI-Driven Real-Time Relay Optimisation in Smart Urban NR-V2X Networks via Learning-to-Optimise Graph Neural Networks](https://arxiv.org/abs/2609.20271) | 本文提出一种基于图神经网络的AI驱动学习优化框架，利用离线MILP最优中继决策监督训练边感知GINE网络，从而在NR-V2X城市车联网中实现接近最优的实时多跳中继选择。 |
| [^68] | [Placement Is Free, Composition Is Not: The Latin Square as a Provably-Balanced Construction for Heterogeneous Sequence-Mixer Stacks](https://arxiv.org/abs/2609.20269) | 本文提出用拉丁方排列异构序列混合器的构建方法，通过让每种机制在每行每列恰好出现一次，实现跨深度均衡暴露并消除放置位置的混淆因素，从而将机制选择与放置的影响解耦。 |
| [^69] | [Improving Online Reinforcement Learning via Bidirectional Behavior Prior Distillation](https://arxiv.org/abs/2609.20268) | 提出双向行为先验蒸馏算法，利用动作值先验引导在线策略学习，无需依赖高质量离线数据集和专家轨迹即可解决贪婪策略更新导致的评论家估计误差问题，从而提升在线强化学习的样本效率和学习稳定性。 |
| [^70] | [Counterexamples and Sufficient Conditions: Comments on "Optimally-Transported Generalized Method of Moments"](https://arxiv.org/abs/2609.20260) | 本文对Schennach & Starck提出的最优传输广义矩方法（OTGMM）估计量的核心定理给出反例，证明其定理2-6在原假设条件下不成立，并探讨使这些结论成立的充分条件。 |
| [^71] | [How Far Can Sub-3B Open Language Models Go in Zero-Shot Essay Scoring on an 8 GB Consumer GPU?](https://arxiv.org/abs/2609.20250) | 在严格隐私约束下，小于3B参数的开源语言模型即可在单张8 GB消费级GPU上实现零样本作文评分，且评分细则分解提示在批处理最小-最大聚合下全面优于整体提示。 |
| [^72] | [Accuracy Is Not Enough: A Cross-Architecture Audit of Demographic Bias in Deep Knowledge Tracing](https://arxiv.org/abs/2609.20249) | 该研究首次对四种深度知识追踪架构在三种训练方案下进行了人口统计偏差的跨架构审计，发现偏差确实存在但高度依赖数据集情境，且仅凭准确率无法保障模型公平性。 |
| [^73] | [Is It Still Worth Training a Classical Model in the Era of LLMs? A Crossover Benchmark on Tabular Data](https://arxiv.org/abs/2609.20218) | 该研究提出“标注数据交叉点 N*”这一指标，量化在表格数据预测任务中经典模型需要多少训练数据才能超越免训练的大语言模型，并发现经典模型经过少量数据训练后便能快速胜出。 |
| [^74] | [Explaining spatial information flow in short-term traffic forecasting models using a gated graph attention network](https://arxiv.org/abs/2609.20217) | 本文提出在图注意力网络层中加入可学习的门控机制，以量化每个传感器的更新状态来自邻居信息的比例，并通过对门控的正则化逐步撤除邻居信息，从而以分级消融的方式解释和验证短期交通预测模型中空间信息流的必要性。 |
| [^75] | [Transformer fault diagnosis using an efficient simulation-driven variational quantum classifier with domain-aware feature encoding](https://arxiv.org/abs/2609.20214) | 本文提出一种仿真驱动的变分量子分类器框架，通过融合基于Duval几何的领域感知特征编码、混合ZX-YY量子特征映射以及双量子比特全纠缠EfficientSU2拟设，在浅层参数化电路中以极少量量子比特实现了基于溶解气体分析的变压器早期故障诊断。 |
| [^76] | [Subdomain-aware representation compression for pretrained image embeddings](https://arxiv.org/abs/2609.20213) | 本文将PCA和LDA等降维技术应用于预训练图像嵌入的子域表示压缩，在降低空间与计算复杂度的同时反而提升了准确率，并验证了压缩表示的迁移学习能力。 |
| [^77] | [Scene-Conditioned Relation Routing for urban cellular activity forecasting](https://arxiv.org/abs/2609.20209) | SCRR-Net提出一种场景条件下的空间关系路由框架，利用城市上下文信息联合控制空间依赖选择与跨任务知识迁移，在短信、网络流量和通话活动预测任务上均优于现有方法并具备可解释性。 |
| [^78] | [Towards a Unified Modality-Agnostic Multimodal Framework for Cognitive Workload Assessment](https://arxiv.org/abs/2609.20199) | 本文提出了一种统一的、模态无关的层级Transformer架构，可在单一模型内处理异构生物信号模态，并系统评估了ECG、EDA、RESP、SpO₂和EEG五种模态全部31种组合在认知工作负荷评估中的效果。 |
| [^79] | [Evaluating Financial Sentiment in the Age of AI](https://arxiv.org/abs/2609.20198) | 本文评估了十二种金融情绪模型，发现通用大语言模型无需微调即可达到与金融专用模型相当的分类准确率，但更高的准确率并未带来更强的经济关联——各模型情绪度量仅与盈利意外显著相关，而均无法显著预测次日股票收益。 |
| [^80] | [SoftTri: Smooth Triangular Membership Functions for Adaptive Fuzzy Inference Systems](https://arxiv.org/abs/2609.20194) | 提出 SoftTri——一种受 Swish 激活函数启发的可微平滑三角隶属函数，在保持经典三角隶属函数几何结构与局部性的同时实现无穷阶光滑，使神经模糊系统具备高效的端到端梯度优化能力。 |
| [^81] | [When Does Retrieval Help Time-Series Forecasting?](https://arxiv.org/abs/2609.20193) | 该研究证明检索插件在时间序列预测中的收益并非源于其机制本身，而是取决于回看窗口长度与主导季节周期的关系，在该关系合适时简单的周期重复基线即可击败标准模型和检索插件。 |
| [^82] | [VLN on the Fly: An Onboard Vision-Language Navigation Stack for Aerial Robots](https://arxiv.org/abs/2609.20191) | 该论文提出了一种将基础定位、规划和控制保持为独立可检查阶段的机载视觉语言导航堆栈，在四旋翼飞行器上实现了15次试验中13次成功到达目标，平均误差仅5.72厘米。 |
| [^83] | [Sequential Contextual Fit Predicts Human Behavioural and Neural Dynamics Across Domains](https://arxiv.org/abs/2609.20179) | 本研究提出序列上下文契合度（SCF）这一通用嵌入度量指标，证明其在语言、情绪、决策和神经数据等多个领域中均能有效预测人类行为与神经动态，且其预测能力独立于惊讶度和预测误差等已有预测因子。 |
| [^84] | [PaGNet: A Panel-Aware GBDT--Neural Network for Multi-Target Corporate Tax Avoidance Proxy Forecasting](https://arxiv.org/abs/2609.20177) | 本文提出面向面板的GBDT-神经网络混合模型PaGNet，通过LightGBM与Panel-MLP双分支结构及逐目标混合器，在韩国上市公司面板数据上实现多目标企业避税代理指标预测，并提供透明的分支依赖性诊断。 |
| [^85] | [Robust Federated Q-Learning with Almost No Communication](https://arxiv.org/abs/2609.20174) | 本文提出了鲁棒联邦Q学习算法 Robust Fed-Q，即使存在少量对抗性智能体，也能在仅需极少通信轮次的条件下实现最优值函数的精确收敛，并获得接近最优的协作式样本复杂度加速。 |
| [^86] | [Support Thresholds, Not Algorithms, Limit Rare-Association Recovery in Co-Purchase Networks](https://arxiv.org/abs/2609.20171) | 该研究发现限制共购买网络中稀有高价值关联恢复的关键因素是支持度阈值的选择而非算法本身，基于网络的过滤方法（如噪声校正法和top-K法）能恢复80-100%的稀有高提升度关联，远超Apriori算法的22-28%。 |
| [^87] | [Small Enough to Know Everything: The Fully-Enumerable Transformer as an Instrument for the Science of Delayed Generalization](https://arxiv.org/abs/2609.20166) | 本文提出将全可枚举任务上的微型Transformer作为研究延迟泛化（grokking）现象的科学仪器，其提供精确泛化上限、任务手术、全权重直接观察和生存时间统计四种独特能力，并通过预注册守恒研究验证小规模下发现的规律具有可迁移性。 |
| [^88] | [LEO Satellite Internet of Things: Architecture, Technology, and On-Orbit Verification](https://arxiv.org/abs/2609.20165) | 本文提出了融合组成与功能视角的6G低轨卫星物联网二维系统架构，评估了大规模免授权随机接入、深度学习多波束预编码和分布式协作路由三项关键使能技术，并通过在轨验证平台证实了其真实可行性。 |
| [^89] | [A Noise Optimum in Rehearsal-Free Continual Learning: Isolation, Mechanism, and Scope](https://arxiv.org/abs/2609.20162) | 本文发现在无回放持续学习中，向巩固规则注入适当强度的相干噪声可使记忆保持呈倒U形最优效应，其关键在于将锚定增益与注入噪声方差相耦合这一单行规则。 |
| [^90] | [Special Lagrangian cones in Deep Learning](https://arxiv.org/abs/2609.20159) | 本文提出了Harvey-Lawson锥的矩阵推广形式，证明其为精确特殊拉格朗日流形，并表明它属于一族对深度学习中平衡流形进行叶状分解的精确特殊拉格朗日流形。 |
| [^91] | [QUALS: Corpus Equilibrium for Universal Forecasting via Pattern Quantization and Learnability Synchronization](https://arxiv.org/abs/2609.20156) | 提出了QUALS大规模时间序列语料库均衡框架，通过模式量化与可学习性同步两大机制管理复杂数据分布，显著提升数据效率，使现有模型仅用一小部分训练数据即可实现更优的零样本预测性能。 |
| [^92] | [Task-Oriented Semantic Feature Transmission for Multi-Task Satellite Remote Sensing over Low-SNR Channels](https://arxiv.org/abs/2609.20150) | 本文提出一种面向任务的语义特征传输框架，绕过图像重建、直接传输多任务预训练骨干网络提取的语义特征，并通过信道适配模块压缩特征与特征恢复器修复信道损伤，在低信噪比信道下的卫星遥感分类与检测任务中始终优于传统重建导向的JSCC方法。 |
| [^93] | [Fast-varying Natural Frequencies and Damping Ratio Identification for Linear Time-Varying System](https://arxiv.org/abs/2609.20138) | 该论文提出了一种将长短期记忆网络与扩展卡尔曼滤波器相结合的物理增强机器学习方法，用于辨识线性时变系统在风浪载荷下快变固有频率与阻尼比，并通过海上风力机仿真数据验证了其有效性。 |
| [^94] | [Local Sparsity Enables Unsupervised LLM Safety Detection](https://arxiv.org/abs/2609.20129) | 本文利用稀疏自编码器概念空间中的局部稀疏性这一关键洞察，提出了一个无需不安全训练数据的无监督LLM安全异常检测框架，并有理论支撑。 |
| [^95] | [QoS-Aware Federated Learning for Multimodal In-Cabin Interaction in Smart Vehicles](https://arxiv.org/abs/2609.20123) | 提出了一种QoS感知的异步事件触发联邦学习框架FedQoS，通过两阶段门控机制将本地计算与全局通信解耦，解决了车载网络异构时变的服务质量约束下智能车辆多模态舱内交互协同训练的难题。 |
| [^96] | [CARE-VI: Conservative Adaptive Reliability Estimation for Value Improvement in Off-Policy Actor-Critic Learning](https://arxiv.org/abs/2609.20098) | 提出CARE-VI框架，通过保守自适应排序与筛选（CARS）、选择器-评估器价值评估（SEVA）和动态自适应风险感知增强三种协同机制，解决了候选动作排序噪声、选择得分复用偏差和固定权重放大弱证据三大风险，从而提升离策略演员-评论家学习中时序差分目标的可靠性。 |
| [^97] | [SETTer: Sparse-Encoder Transformer for Long-term Multivariate Time Series Forecasting](https://arxiv.org/abs/2609.20086) | SETTer是一种用于长期多变量时间序列预测的Transformer模型，通过解耦自注意力和混合掩码技术，有效捕捉时间与通道维度上的短期和长期模式，并增强了模型的可解释性。 |
| [^98] | [MATCH: Model-Aware Tool Learning with Curriculum Scheduling and Hierarchically Gated Rewards](https://arxiv.org/abs/2609.20082) | MATCH提出了一种模型感知的闭环工具学习框架，通过课程难度与策略能力共同演化的课程调度，以及按工具名称、参数键、参数值逐级门控授予信用的分层奖励机制，解决了固定阈值课程脱节与加性奖励信用泄漏两大问题。 |
| [^99] | [Evaluating Explanation Methods by the Predictors They Induce](https://arxiv.org/abs/2609.20058) | 提出一种新的解释方法评估框架：将解释直接转化为可加性预测器并衡量其在未见数据上重现模型预测的能力，无需任何拟合，从而客观比较PDP、ALE、SHAP和LIME等方法的优劣。 |
| [^100] | [Correct Now, Insufficient Later: Auditing Update Sufficiency in Context Compression](https://arxiv.org/abs/2609.20045) | 该论文提出配对历史审计方法，揭示上下文压缩的记忆系统虽能正确回答当前查询，却可能丢弃后续更新所需的关键区分信息，并构建记录级审计框架以区分记忆保留充分性、响应传递和答案格式合规性等不同失败模式。 |
| [^101] | [Dynamic Generalized Gromov-Wasserstein Optimal Transport](https://arxiv.org/abs/2609.20008) | 该论文提出TP-DATE框架，首次以无模拟方式将Gromov-Wasserstein最优传输动态化，通过路径作用量证明静态与动态二次型最优传输的等价性，并发展行进对流匹配方法，实现空间转录组学中兼顾组织结构保持的连续轨迹重建。 |
| [^102] | [EPIG-Tree: Compute-Optimal Branching for Gradient-Efficient Reinforcement Learning](https://arxiv.org/abs/2609.20004) | 该论文提出EPIG-Tree方法，通过全方差定律分解推导出两条计算分配定律，将树状分支放置在每单位计算能最大程度降低策略梯度不确定性的位置，从而实现计算最优且梯度高效的强化学习。 |
| [^103] | [Past, Future, All at Once: Mitigating Stability-Plasticity Dilemma via Post-hoc JANUS Rectification](https://arxiv.org/abs/2609.19985) | 提出了一种事后且与微调无关的JANUS权重修正框架，通过将参数更新投影到雅可比零空间实现参数空间正交性，在微调新任务的同时有效缓解灾难性遗忘、恢复历史知识。 |
| [^104] | [Quantum Graph Convolutional Networks: Implementation and Trainability Analysis](https://arxiv.org/abs/2609.19983) | 本文基于QGNN框架实现了SGC和LGC两种量子图卷积网络，通过量子模拟证明其能以更少参数达到与经典模型相当的性能，并通过代价梯度分析确定了这些量子模型可训练的适用任务。 |
| [^105] | [CellRFT: Reinforcement Fine-Tuning for Single-Cell Perturbation Modeling](https://arxiv.org/abs/2609.19970) | 提出CellRFT强化微调框架，以生物学评估作为直接训练反馈，通过策略梯度优化与分层奖励聚合改进单细胞扰动预测。 |
| [^106] | [Graph-Based Stochastic Power-UCT: Monte-Carlo Graph Search with Power Mean Estimation](https://arxiv.org/abs/2609.19956) | 提出基于图的随机Power-UCT算法（GS-Power-UCT），通过在相同规划深度共享状态并保留不同深度的独立值，在含环路的一般随机MDP中实现样本复用，证明了以 O(n^{-1/2}) 速率收敛到有限视界值，并进一步提出两种全状态变体以增加样本共享并控制视界混合偏差。 |
| [^107] | [One Intervention per Component is Enough: Towards Identifiability in Linear Stochastic Dynamics from Steady State](https://arxiv.org/abs/2609.19955) | 本文证明了对漂移图的每个强连通分量仅需一次干预，即可从稳态数据中辨识多元OU过程的全部参数，并提出了相应的递归学习算法。 |
| [^108] | [Intrinsic Sequence-Likelihood Confidence in Retrieval-Dominated Extractive QA: Two Pre-Specified Negatives, and What They Do and Do Not Attribute](https://arxiv.org/abs/2609.19942) | 该研究通过预先注册的评估标准证明，在检索已能恢复92-99.8%最优性能的抽取式问答场景中，模型的内在序列似然置信度无论作为蒸馏触发器还是路由弃答策略的控制信号均告失效。 |
| [^109] | [Stringological sequence prediction III: layered ziplines and a tradeoff between efficiency and expressivity](https://arxiv.org/abs/2609.19940) | 本文提出了一种基于分层滑索程序的较弱复杂度度量，该度量虽然表达性低于算术重复复杂度（ARC），但支持在拟线性时间和多对数空间内运行的高效预测算法，揭示了效率与表达性之间的权衡。 |
| [^110] | [Error bounds in Sobolev norms for approximations with norm constrained ReLU neural networks](https://arxiv.org/abs/2609.19937) | 该论文将路径范数约束ReLU神经网络的逼近理论从一致逼近推广到Sobolev范数逼近，分别针对浅层网络和深层网络给出了以$W^{1,p}$范数度量的逼近误差界，且深层网络的误差界无需对函数光滑性附加限制。 |
| [^111] | [From "Who Is This User?" to "What Does This Purchase Mean?": A Deployed Pipeline for Semantic User Profiling at Bank Scale](https://arxiv.org/abs/2609.19928) | 该论文提出一种已部署的三阶段LLM流水线（解析-画像-标注），将用户属性推断从“逐用户”转变为“逐交易模式”，使推理成本随模式数而非用户数增长，在银行级规模下实现了与逐用户LLM推断统计上无差异的语义用户画像。 |
| [^112] | [The Life of a Token: from Words to Bits on the Wire](https://arxiv.org/abs/2609.19924) | 这是一篇教程论文，以但丁《神曲》为具体示例，完整讲解了文字如何经过分词、向量嵌入，最终转化为高性能计算系统中网络通信比特流的全过程，揭示了LLM训练中不为人知的底层通信机制。 |
| [^113] | [Amortizing Physics-Informed Neural Solvers via Graph Hypernetworks](https://arxiv.org/abs/2609.19915) | 该论文提出将偏微分方程的算子和跨场关系显式编码为算子图，利用图超网络为元训练的因式分解PINN生成初始化编码，在无需解标签的情况下将物理信息神经求解器摊销到相关PDE族，并在固定适应预算内比系数向量和DeepSets描述符取得更高求解精度。 |
| [^114] | [Digital Twins for Opinion Dynamics: A Generative LLM Framework for Social Networks](https://arxiv.org/abs/2609.19913) | 本文提出了一种基于数字孪生概念的生成式大语言模型框架，通过克隆真实Twitter网络、为智能体赋予多维属性并利用Mistral-7B模拟观点更新，且在真实数据集上验证了其有效性。 |
| [^115] | [REARL: A Closed-loop Autonomous Driving Simulation Enhancement Framework with Real Traffic Data and Large Language Models](https://arxiv.org/abs/2609.19903) | REARL提出了一种结合真实交通数据聚类与大语言模型的闭环仿真增强框架，通过实时监测仿真与真实交通的偏差并动态调整车辆决策，使自动驾驶仿真更贴近真实交通模式。 |
| [^116] | [Self-Replicating Neural Cellular Automata: Quantifying Emergent Phenotypic and Genotypic Diversity in an OpenEnded Substrate](https://arxiv.org/abs/2609.19902) | 本文提出了自我复制的神经细胞自动机系统，并设计了一套表型与基因型两个尺度的粗粒度多样性度量指标，首次对开放式进化基底中涌现的空间组织化生态系统的多样性进行了量化分析。 |
| [^117] | [Delphi Scanner: efficient and interpretable static malware detection via API sequence modeling](https://arxiv.org/abs/2609.19900) | 该论文提出Delphi Scanner系统，利用CNN对Windows API序列建模并结合基于规则的解耦解释层，以仅1.53 MB的紧凑模型实现了95.35%准确率的高效且可解释的Windows PE静态恶意软件检测，并展现出良好的分布外泛化能力和对抗逃避技术的鲁棒性。 |
| [^118] | [Online Adaptive Kernel Mixing for Gaussian Process Decision Making](https://arxiv.org/abs/2609.19891) | 提出HACK方法，将高斯过程的核选择转化为基于专家建议的在线学习问题，利用AdaHedge在线自适应地混合候选核并保证权重收敛于最佳核，从而提升贝叶斯优化等序列决策任务在核设定错误下的鲁棒性。 |
| [^119] | [Uni-LaDiR: Latent Diffusion Unifies Multimodal Reasoning](https://arxiv.org/abs/2609.19878) | Uni-LaDiR提出将多模态推理步骤映射到共享潜在空间，并利用扩散模型预测下一块思维标记，从而在统一的潜在表示中实现灵活的多模态推理。 |
| [^120] | [AURA: Adaptive Uncertainty-Routed Analysis for Email Threat Detection](https://arxiv.org/abs/2609.19873) | 本文提出AURA，一种多模态电子邮件威胁检测系统，通过不确定性路由机制仅将模糊消息升级至微调的transformer编码器进行语义分析，在跨越十年真实对抗攻击数据上实现了高效的垃圾邮件与钓鱼攻击检测。 |
| [^121] | [Pretrained Medical Representations for the Practical Screening of Drug Repositioning Candidates](https://arxiv.org/abs/2609.19865) | 提出了一种集成层次化子词聚合、部分掩码和交叉引用机制的统一预训练框架，能更好地捕捉医疗代码的层次结构与诊疗交互，在临床事件预测任务中优于现有方法，并可用于药物重定位候选药物的实用筛选。 |
| [^122] | [Expected Hypervolume Maximization for Multiobjective Optimization under Uncertainties](https://arxiv.org/abs/2609.19858) | 该论文提出将不确定性下的多目标优化表述为关于期望超体积最大化的贝叶斯决策问题，并利用高斯过程作为可微代理模型以及基于采集函数的主动学习策略来高效求解。 |
| [^123] | [Trust, but Validate the Instrument: Auditing AI-Generated RTL Verification Plans on Authored Security-Regression Proxies](https://arxiv.org/abs/2609.19844) | 论文提出可审计框架SecTB-RTL，通过自建硬件安全回归测试发现AI生成的RTL验证计划虽被提供商接受但几乎全部无法通过生产语义验证，证明提供商模式的接受并不能等同于执行的有效性。 |
| [^124] | [Beyond Flattened Tokens: Structure-Preserving EEG Decoding with Reusable TriDim Blocks](https://arxiv.org/abs/2609.19842) | 提出可复用的TriDim模块，通过显式保留通道、补丁内样本位置和补丁位置三个结构轴并配合跨轴注意力机制，避免了传统方法将EEG标记扁平化处理的问题，实现了结构保持式的EEG解码。 |
| [^125] | [Steering Equilibrium Selection in Regularized Self-Play via the Reference Policy](https://arxiv.org/abs/2609.19820) | 该研究证明在正则化自我博弈中，通过将熵正则化参考策略锚定在目标成员上，可以有目的地引导算法收敛到纳什均衡多胞体中特定的价值等价均衡，且锚定效果跟随参考策略而非初始化。 |
| [^126] | [DeliveryGym: An RL Environment for Long-Horizon Embodied Agent Planning with Adaptive Curriculum](https://arxiv.org/abs/2609.19801) | DeliveryGym是一个面向长时程具身智能体规划的3D强化学习环境，通过连续快递员班次、持久世界动态和基于模拟器事件的轨迹奖励来评估决策成本，并采用自适应课程机制根据智能体观察到的弱点动态调整训练难度。 |
| [^127] | [Evolution or Illusion? Rethinking Evaluation in LLM Evolutionary Search](https://arxiv.org/abs/2609.19799) | 该论文通过在种子数与迭代数的完整组合网格上系统评估三种LLM进化搜索策略，揭示了固定预算在“宽度”（更多种子）与“深度”（更多迭代）之间的最优分配方式以及策略排名都会随策略、任务和总预算显著变化，证明传统单预算设置下的评估结论并不可靠。 |
| [^128] | [Learning-Based Reconstruction of Optical Properties in Bilayered Media from Single-distance Time-Resolved Reflectance Measurements](https://arxiv.org/abs/2609.19786) | 提出了一种基于机器学习的新框架，利用蒙特卡洛模拟生成的合成DTOF数据集，从单距离时间分辨反射测量中重建双层生物介质的光学特性，克服了传统扩散方程逆求解器在结构异质性下精度不足的问题。 |
| [^129] | [PhyRestore: Physics-Structured Latent-Factor Restoration](https://arxiv.org/abs/2609.19776) | PhyRestore提出了一种物理结构化的潜在因子恢复框架，先恢复受损的物理因子，再依据RUSLE已知的物理关系重建土壤流失时间变化，在高幅变化恢复上优于直接预测方法。 |
| [^130] | [TorchCraft: Unified binder design by inverting an all-atom structure predictor](https://arxiv.org/abs/2609.19770) | 提出了TorchCraft统一结合剂设计框架，通过逆转冻结的全原子结构预测器（基于预训练的AlphaFold 3权重）优化序列，成功设计出微结合剂、VHH、环肽和配体结合蛋白等多种结合剂，实验验证其无需事后序列重新设计即可实现有效结合。 |
| [^131] | [OceanMoE: Structured Conditional Sparse Computation for Long-Horizon Multivariate Ocean Forecasting](https://arxiv.org/abs/2609.19768) | 提出OceanMoE结构化条件稀疏混合专家框架，在统一模型中保留海洋变量共享上下文的同时，根据预测目标、空间位置和路由置信度实现自适应专门化计算，提升长时间序列多变量海洋预报能力。 |
| [^132] | [HyperAMS-Net: Adaptive Multi-Scale Spatial Hypergraph Network for Brain Disorder Classification](https://arxiv.org/abs/2609.19755) | HyperAMS-Net是一种融合自适应多尺度卷积、超图注意力和空间-通道注意力的深度学习框架，能够有效捕捉脑部神经影像中的多尺度模式与高阶依赖关系，实现基于静息态fMRI或结构MRI的脑部疾病精准分类。 |
| [^133] | [Alliance Beats Isolation: Unifying Heterogeneous Allied Datasets Improves Classifier Performance](https://arxiv.org/abs/2609.19748) | 本文提出一种将异构关联数据集的特征空间合并并利用矩阵补全构建统一数据集的方法，从而实现分类知识在数据集间的迁移，提升分类器性能。 |
| [^134] | [Federated Learning Framework for Privacy-Preserving Kidney Stone Detection](https://arxiv.org/abs/2609.19740) | 该论文提出了一种结合优化YOLOv8网络的联邦学习框架，在无需共享患者数据、符合GDPR和HIPAA法规的前提下，实现了CT图像中肾结石的隐私保护式准确检测。 |
| [^135] | [ALIBI: Adversarial Legitimacy Injection in Binary Input against LLM Malware Analyzers](https://arxiv.org/abs/2609.19722) | 该论文提出ALIBI攻击方法，通过在二进制文件中注入一个包含虚假安全产品叙述的非执行只读节区（不改变任何可执行行为），即可诱导前沿LLM恶意软件分析器将恶意样本误判为良性或大幅降低其威胁严重性评估，揭示了LLM推理能力带来的新型攻击面。 |
| [^136] | [Learn Your Own Thoughts: Abstract Token Curriculum](https://arxiv.org/abs/2609.19717) | 提出了抽象token课程学习（ATC）框架，无需直接监督或手动设计草稿板，即可通过逐步增加问题复杂度训练模型在连续表示空间中自发形成内部抽象思维，并从理论和实验上证明了其相对于以往连续思维训练方法的优势。 |
| [^137] | [Odds-Ratio Thompson Sampling: A Specification and Design Guide for Contrast-Based Multi-Armed Bandits](https://arxiv.org/abs/2609.19709) | 本文提出比值比汤普森采样（OR-TS），通过保留臂间对数比值对比的联合后验分布并在每个批次重新拟合共同水平，解决了批量多臂老虎机中因共享基线水平漂移而导致绝对奖励率记忆失效的问题，显著降低了后悔值。 |
| [^138] | [Opinion Dynamics-based Coalition Formation for Federated Learning in Heterogeneous IoT Systems](https://arxiv.org/abs/2609.19695) | 该论文将联邦学习中的客户端联盟形成建模为Hegselmann-Krause有界置信观点动力学过程，通过在本地权重空间中基于欧氏距离和余弦相似度置信准则形成联盟并在联盟层面聚合，解决了异构物联网系统中统计异质性导致FedAvg无法捕获客户端特定模式的问题。 |
| [^139] | [Conservation Buys Stability and Factoring Buys Counterfactuals in Physical World Models](https://arxiv.org/abs/2609.19674) | 该论文证明物理世界模型的两种失效需要不同的结构性补救：用辛积分器演化学习到的能量可保持保守动力学几何结构、使长时程展开在训练范围100倍内保持稳定，而用显式线性因式分解编码物理耦合则能使模型泛化到从未见过的干预情形。 |
| [^140] | [When2Think: Learning Difficulty-Aware Length Control for Efficient Hybrid Reasoning Models](https://arxiv.org/abs/2609.19671) | When2Think提出了一个混合推理后训练框架，通过实例级难度感知控制（IDAC）机制根据问题难度动态分配计算资源，解决了大型推理模型对简单问题过度思考、对困难问题思考不足的系统性低效问题。 |
| [^141] | [CoRe: Coherence and Relational Alignment for Multivariate Time Series Forecasting](https://arxiv.org/abs/2609.19670) | CoRe提出了一种与模型无关的学习目标，通过频率一致性损失和低秩关系图损失显式保留未来轨迹的时间一致性与变量间关系一致性，无需引入可训练参数即可增强现有多变量时间序列预测模型。 |
| [^142] | [EmbodiedMind: Adaptive Data Curation and Prefix-Tree Reinforcement Learning for Efficient Embodied Intelligence](https://arxiv.org/abs/2609.19659) | 本文提出EmbodiedMind训练范式，通过自适应数据筛选（RSFT与IR-GRPO）和前缀树强化学习三大协同阶段，解决了具身基础模型训练中样本利用率低、梯度贡献不均衡和长时程信用分配困难三大问题，以更少的数据和计算资源实现了最先进的平均性能。 |
| [^143] | [PrefixBench-H100: Characterizing Prefix Reuse and Time-to-First-Token in H100 LLM Serving](https://arxiv.org/abs/2609.19657) | 本文提出PrefixBench-H100，一个在单个NVIDIA H100上可复现的基准测试与测量框架，通过受控合成轨迹与真实工作负载系统表征前缀重用和首令牌生成时间的特性，揭示vLLM和TensorRT-LLM等推理运行时中前缀重用收益的边界条件。 |
| [^144] | [Well-posedness of neural turbulence closures and tangent dissipation](https://arxiv.org/abs/2609.19647) | 该文为神经湍流闭合建立了适定性理论，证明全局切向耗散——可通过强制非负切向扩散的精确积分构造或对切向反应违反施加惩罚来促进——能够保证解的存在性、唯一性以及后验误差与先验残差之间的全局逆敏感度界。 |
| [^145] | [A Policy Profile for Croissant: Refusal as a Property of the Dataset](https://arxiv.org/abs/2609.19640) | 该论文为机器学习数据集描述符Croissant提出了一个策略配置文件，通过封闭的五种运算符集合和完整定义的决策程序，使数据集能够声明其允许的操作及条件，让门控系统仅凭描述符即可做出并记录访问决策。 |
| [^146] | [The Complexity Kink: A Prompt-Side Structural Complexity Index for Code-Generation Reliability](https://arxiv.org/abs/2609.19616) | 该论文提出一个生成前评分的六维提示侧结构复杂度指数，发现代码生成通过率在复杂度分数上存在非单调断点，但该断点并非通用的失败临界值，会随任务类型固定效应等因素发生移动。 |
| [^147] | [TacSushi: Tactile-Grounded World-Action Modeling for Dexterous Sushi Manipulation](https://arxiv.org/abs/2609.19613) | 提出TacSushi，一种触觉接地的世界-动作建模策略，通过特征级门控融合指尖触觉并利用失败试验的未来后果预测进行监督，实现了形变、遮挡和不确定接触条件下的灵巧寿司操作。 |
| [^148] | [The Output-Space Hypothesis: Enumerative Equivalence Checking for Tensor Programs](https://arxiv.org/abs/2609.19611) | 该论文提出“输出空间假设”，通过翻转量词的新颖符号执行策略，从“针对单个输入检查所有输出位置”转变为“针对所有输入检查单个输出位置的等价性”，从而更持续有效地发现张量程序优化中的错误。 |
| [^149] | [DeltaSelect: Affordable A/B Testing for Coding Agents](https://arxiv.org/abs/2609.19607) | DeltaSelect是一种开源方法，通过皮尔逊相关性筛选出单次运行即可可靠代表完整基准性能的任务子集，为编码智能体的开发迭代提供低成本、可重复的A/B测试方案。 |
| [^150] | [Chain-of-Thought Entropy as a Reliability Signal: A Preregistered Reproduction](https://arxiv.org/abs/2609.19606) | 本预注册独立复现研究证实，大语言模型思维链熵轨迹的形状（而非总熵下降幅度）是预测答案正确性的可靠信号，形状信号在四个开源模型上成功复现，而幅度信号则因实验设置而异。 |
| [^151] | [CliniCIRCA: A Modular LLM Framework for Constructing Longitudinal Mental Health Patient Journeys from Raw EHR Narratives](https://arxiv.org/abs/2609.19585) | CliniCIRCA是首个无需事件级时间戳即可从非结构化出院总结中对临床事件进行时间分类的多阶段大语言模型框架，通过临床医生参与纠错生成黄金标准标签，并支持基于时间线的患者历程总结。 |
| [^152] | [FedFIbOS: Fisher Importance based Optimal Submodelling for Heterogeneous Federated Learning](https://arxiv.org/abs/2609.19559) | 提出FedFIbOS方法，通过最小化子模型掩蔽误差推导出基于Fisher信息的原则性参数选择准则，为异构联邦学习中最优子模型的选择提供了理论依据，解决了现有启发式方法缺乏收敛理论支撑以及部分客户端参与引入估计偏差的问题。 |
| [^153] | [Compressed Active Subspaces for Scalable Bayesian Inference](https://arxiv.org/abs/2609.19539) | 本文提出压缩主动子空间（CAS）方法，通过结构化等距嵌入先将模型参数映射到压缩空间再构建主动子空间，大幅降低内存开销，使大规模模型的可扩展贝叶斯推断成为可能。 |
| [^154] | [Next-token functional estimation](https://arxiv.org/abs/2609.19529) | 针对时间依赖数据，本文提出“留窗口法”估计量用于估计下一词元泛函（如惊喜概率、测试误差等），克服了留一法在时间依赖下不一致的缺陷，并在平稳 β-混合过程下以参数速率收敛。 |
| [^155] | [Self Improvement via Fast Tree-search](https://arxiv.org/abs/2609.19526) | 提出SIFT框架，利用LLM作为裁判对候选补丁进行两两比较并通过正则化Bradley-Terry模型聚合实力分数，大幅降低了自我改进循环中候选修改的评估成本，使编码智能体在严格预算约束下实现高效的自我改进。 |
| [^156] | [LSTM-UT and Recurrent-Depth Transformers on Cellular Automata](https://arxiv.org/abs/2609.19521) | 该论文提出带有有界门控记忆的LSTM-UT模型，并通过Rule 30元胞自动机和延迟回忆实验证明，仅依赖当前隐藏状态的Block Universal Transformer在深度外推上优于依赖扩展注意力缓存的CoTFormer，而LSTM-UT进一步同时提升了深度外推与延迟回忆能力。 |
| [^157] | [An Architecture for Long-Horizon Agents: Levels, Ticks and Cascaded Intelligence](https://arxiv.org/abs/2609.19519) | 本文提出一种由时间尺度层级记忆、时钟节拍驱动的自主行动和失败后才升级的级联智能组成的分层架构，使语言模型智能体能够在不遗忘的前提下持续运行数天甚至数周。 |
| [^158] | [LLM-as-an-Improver: Turning Verification into Better Candidates](https://arxiv.org/abs/2609.19515) | 本文提出“验证—修复—重选”（VRR）方法，不再将验证器的反馈仅用于排序，而是利用其修复优胜候选、生成新思路方案并重新选择最终答案，从而在推理时提升LLM的代码生成与推理性能。 |
| [^159] | [QVAC Genesis III: A Large-Scale, High-Quality Open Synthetic STEM Corpus for Efficient Language Model Pre-Training](https://arxiv.org/abs/2609.19513) | 提出了QVAC Genesis III——一个包含1914.3亿token的开放STEM合成语料库，通过以弱学生模型信号驱动的双重生成策略（将失败转化为纠正性解释、将成功扩展为对比性选项级推理），为token预算受限的小模型高效预训练提供了高价值数据。 |
| [^160] | [Null importance: Disentangling relevance for interpretable machine learning](https://arxiv.org/abs/2609.19511) | 本文提出基于“零重要性”的统一框架，厘清了可解释机器学习中特征重要性所隐含的多种本质上不同的相关性概念，并在算法公平性和基因组扰动建模中展示了这些区分的实际意义。 |
| [^161] | [Sample Count Is Not Enough: Candidate-Generation Strategy Shapes the Energy and Performance of LLM Test-Time Scaling](https://arxiv.org/abs/2609.19499) | 该论文指出，仅用候选样本数量N不足以刻画大语言模型测试时扩展的真实开销，相同采样预算在不同批量调度策略下的能耗与性能表现差异显著。 |
| [^162] | [Search at the Cost of Sampling: Nearly-Instant Latent Space Bayesian Optimization](https://arxiv.org/abs/2609.19476) | 该论文提出一种在高维潜在空间的球形域上使用线性代理模型的贝叶斯优化方法，通过利用球面对称性推导出近乎闭式的求解方案，相比最先进的基线方法实现至少100倍的加速，使贝叶斯优化在虚拟筛选成本较低的场景下变得实用。 |
| [^163] | [Safety Beyond the Interface: Detecting Harm via Latent States in Large Language Models](https://arxiv.org/abs/2609.19472) | 该研究通过从LLaMA-3.1-8B内部激活值中训练仅1260万参数的轻量级MLP探针来检测有害提示，实现了与规模大1000倍的防护模型相当的检测性能（F1最高达99%），同时显著降低了延迟和计算成本。 |
| [^164] | [Enhanced Agriculture-informed Neural Network by Domain Knowledge](https://arxiv.org/abs/2609.19466) | 提出了一种融合肥料扩散、土壤呼吸和充水孔隙度等领域知识的神经-机理混合框架KAINN，提升了农业氧化亚氮（N2O）排放预测的准确性、可解释性和跨环境条件的泛化能力。 |
| [^165] | [Compositional Reasoning in Language Models under Reinforcement Learning Post-Training](https://arxiv.org/abs/2609.19465) | 本文提出依赖图框架形式化语言模型的组合推理，并揭示强化学习后训练中的不对称迁移现象——分解技能训练难以迁移到组合任务，而组合任务训练则更容易迁移回分解任务。 |
| [^166] | [Sharpness-Aware Minimization (SAM) Improves Classification Accuracy of Bacterial Raman Spectral Data Enabling Portable Diagnostics](https://arxiv.org/abs/2609.19453) | 本论文将锐度感知最小化（SAM）应用于细菌拉曼光谱分类任务，在无需复杂预处理的情况下将分类准确率提升高达10.5%，并增强了模型在有限数据集上的泛化能力，为便携式抗生素耐药性诊断奠定了基础。 |
| [^167] | [GLAMDRING: Gait Learning And Morphology co-Design via Reinforcement LearnING of CPGs](https://arxiv.org/abs/2609.19452) | 该论文提出GLAMDRING框架，通过强化学习训练Hopf振荡器CPG步态控制器，并同时优化四足机器人的形态设计（连杆几何结构与执行器配置），以满足速度、功率和载荷等约束条件。 |
| [^168] | [From Models to Systems: A Comprehensive Survey of Efficient Multimodal Learning](https://arxiv.org/abs/2609.19445) | 本综述首次提出涵盖模型、算法和系统三个层次的结构化高效多模态学习分类体系，并系统综合了跨层协同设计的方法论，以应对“效率-效用-隐私”的根本性权衡。 |
| [^169] | [Bayesian Optimization with Rich Auxiliary Information via LLMs](https://arxiv.org/abs/2609.19437) | 本文提出三种利用大语言模型将丰富辅助信息（如训练曲线、专家笔记和先验知识）融入贝叶斯优化的方法，在超参数优化基准和真实核聚变优化任务中始终优于标准BO及现有LLM优化方法。 |
| [^170] | [Demystifying Linear Operator Learning for Control Systems](https://arxiv.org/abs/2609.19428) | 本文提出利用演化方程的（半）群框架推导结构假设，并借助逆问题框架分析学习算法，从而为控制系统的线性算子学习提供误差分解、收敛保证与最优正则化，实现方法比较与可证明优势算法的设计。 |
| [^171] | [Stable Policy Learning](https://arxiv.org/abs/2609.19418) | 本文证明算法稳定性是刻画策略学习中期望福利与抽样风险之间权衡的核心因素，并提出“策略投票装袋”方法，通过对多个子样本的处理决策投票取平均来降低抽样风险。 |
| [^172] | [Deep Learning Detection of Beyond-General-Relativity Deviations in Gravitational-Wave Signals: A Detection-Threshold Study with Real LIGO Noise](https://arxiv.org/abs/2609.19416) | 该研究利用一维卷积神经网络结合手工波形统计量的混合深度学习分类器，在真实LIGO H1探测器噪声中对引力波信号的超越广义相对论偏差进行探测，并基于真实GW150914应变数据确定了约β≈0.25的定量探测阈值。 |
| [^173] | [Improving Offline Goal-Conditioned Reinforcement Learning via Selective Reward Stimulation](https://arxiv.org/abs/2609.19414) | 提出RSIQL方法，通过利用辅助价值函数识别离线轨迹中朝目标取得进展的中间状态并对其施加选择性奖励刺激，来改善奖励稀疏和长时程依赖下的离线目标条件强化学习。 |
| [^174] | [Mammography Foundation Models for Opportunistic Prediction of Major Adverse Cardiovascular Events](https://arxiv.org/abs/2609.19385) | 该研究表明，为乳腺癌任务预训练的乳腺X线摄影基础模型无需心血管专门监督或乳腺动脉钙化标注，即可利用常规乳腺筛查影像机会性地预测女性未来5年的重大不良心血管事件风险（AUROC约0.82）。 |
| [^175] | [FCx: An algorithm for finding Feasible Counterfactual Explanations](https://arxiv.org/abs/2609.19383) | 该论文提出FCx算法，通过改进的变分自编码器首次高效生成同时具备逼真性、低成本和可行性的反事实解释，同时支持用户指定的硬约束和通过因果推断自动获取的软约束。 |
| [^176] | [Machine-Learning Assessment of the Predictive Value of Inflammatory Biomarkers for Cognitive Impairment in an Older Hispanic Adult Cohort](https://arxiv.org/abs/2609.19374) | 该研究在小型老年西班牙裔临床队列中采用无泄漏的阈值似然朴素贝叶斯分类器，发现炎症生物标志物I-309（CCL1）相较于人口统计学基线对认知障碍具有显著且可复现的预测增量价值。 |
| [^177] | [Stiefel Attention: When the Geometry of Transformer Projection Matrices Dominates Optimizer Choice---and When It Does Not](https://arxiv.org/abs/2609.19363) | 本文将注意力中的查询和键投影矩阵约束到Stiefel流形上并用黎曼Adam优化，理论证明了该更新的最速下降性、尺度无关性和严格O(d)-等变性，并发现权重衰减在该流形上黎曼梯度恒为零，使注意力几何免于坍缩，从而在grokking任务上将验证准确率从61.1%大幅提升至97.0%。 |
| [^178] | [Smart Insole Human Activity Recognition for Continuous Monitoring in Elderly Care](https://arxiv.org/abs/2609.19359) | 本文提出了一种集成16个压力传感点和六轴IMU的无线智能鞋垫平台，利用HGB机器学习模型从足底压力和惯性信号中高精度识别老年人坐、站、行走及不稳定行走状态（宏F1超过0.95），为养老护理中的跌倒风险持续监测提供了有效方案。 |
| [^179] | [Learning Submanifolds for Subsequent Inference on Random Dot Product Graphs, Part 1: Theory](https://arxiv.org/abs/2609.19357) | 提出了一种半监督决策规则框架，利用辅助数据和Isomap流形学习来学习随机点积图潜在位置的未知低维支撑流形，并证明随着辅助数据量的增加，半监督规则的风险收敛于最优先知规则的风险。 |
| [^180] | [How to Guide Your Language Flow](https://arxiv.org/abs/2609.19356) | 提出了一种名为“探针引导”的新方法，利用现有扩散模型的冻结内部状态构建引导信号，无需推理时额外的前向传播，即可在无条件生成和问答基准上显著提升扩散语言模型的性能，并揭示了自动引导中弱模型需来自训练低熵区域的关键条件。 |
| [^181] | [Can Vision-Language Models Judge Olympic Diving? From Reasoning to Scores in Zero-Shot Action Quality Assessment](https://arxiv.org/abs/2609.19354) | 该研究提出一种基于回归的集成框架，利用视觉语言模型生成的语义推理和阶段级子评分对奥运跳水进行零样本动作质量评估，将Spearman相关性从0.32显著提升至0.67。 |
| [^182] | [Personalized Federated Hierarchical Gaussian Processes for Privacy-Preserving Modeling of Heterogeneous Distributed Systems](https://arxiv.org/abs/2609.19337) | 提出个性化联邦层次高斯过程 pFedHGP，通过将客户端潜在函数分解为共享全局分量、客户端特定偏差和局部残差，在仅同步低维统计量的前提下实现异构分布式数据的隐私保护概率建模，并在故障分类和空气质量建模等应用中取得优异效果。 |
| [^183] | [Why Pretraining Fails to Share Cross-Lingual Knowledge](https://arxiv.org/abs/2609.19291) | 本研究通过受控双语预训练实验发现，不相交的词表空间是跨语言知识泛化的根本障碍——即使是对同一语言的完全相同副本，仅仅词表不相交就足以导致知识隔阂。 |
| [^184] | [Learning-Induced Dynamical Transition in Recurrent Neural Networks](https://arxiv.org/abs/2609.19288) | 该研究发展了非平衡动力学平均场理论，揭示循环神经网络在学习过程中因有效反馈强度的持续增长而经历由分岔刻画的从混沌到稳定的动力学转变。 |
| [^185] | [Radio-Frequency Convolutional Neural Networks](https://arxiv.org/abs/2609.19279) | 该论文提出射频卷积神经网络（RF-CNN），创造性地将无线设备中已有的射频混频器硬件重新用于在频域执行卷积运算，使边缘设备无需增加额外计算硬件即可运行高达2640万参数的深度CNN。 |
| [^186] | [Randomized SVD Approximations for Spectral Co-Clustering of Word-Document Matrices](https://arxiv.org/abs/2609.19243) | 本文提出两种随机SVD近似方法来加速词-文档矩阵的谱共聚类，实验表明随机投影方法在各种设置下更为可靠，而随机采样方法仅对较稠密的矩阵有效。 |
| [^187] | [Block Parallelism For Efficient Distributed Long-Context Diffusion Language Model Training](https://arxiv.org/abs/2609.19242) | 本文提出块并行（BP）与上下文分片块并行（CSBP）两种新的分布式并行方法，利用BDLM目标函数在目标块上的可分离性，将损坏块计算分配至独立进程并分片共享干净序列，从而大幅提升长上下文扩散语言模型训练的通信与内存效率。 |
| [^188] | [Layer-wise Curriculum Learning for Efficient LLM Compression](https://arxiv.org/abs/2609.19213) | 提出逐层课程学习方法用于高效LLM压缩，通过将模型分层分段并从易到难地进行知识蒸馏以加速收敛、稳定迁移过程，同时借助多线程特征缓存策略最大化GPU利用率，实现了先进的模型压缩效果。 |
| [^189] | [Not All Nodes Are Created Equal: Homophily-Aware Stratification for Stable GNN Evaluation](https://arxiv.org/abs/2609.19210) | 该论文指出，仅按类别分层的交叉验证不足以稳定图神经网络评估，因为数据划分间局部邻域同质性分布的差异会系统性影响消息传递行为并夸大评估方差，为此提出了同质性感知的分层划分方法以实现更可靠的GNN比较。 |
| [^190] | [Generative Query Suggestion via Intent Coverage and Query-Level Credit Assignment](https://arxiv.org/abs/2609.19209) | 该论文提出一种意图驱动的生成式查询推荐框架，通过意图感知多样性奖励与查询级信用分配的双阶段优化，在提升点击率和查询质量的同时增强意图覆盖。 |
| [^191] | [Position: It is Time to Virtualize Foundation Models with a Self-evolving Operating System Layer](https://arxiv.org/abs/2609.19203) | 本文提出构建“基础模型操作系统”（FMOS），通过虚拟化基础模型交互并统一编排记忆分层、模型选择、资源分配与策略验证，以解决当前AI智能体技术栈碎片化、行为不可移植和治理脆弱的问题。 |
| [^192] | [Federated Soft Clustering via Generalized Total Variation Minimization](https://arxiv.org/abs/2609.19202) | 该论文提出基于广义全变差最小化的联邦软聚类框架，系统比较了三种模型差异度量方法（无需组件匹配的KL散度和闭式MMD，以及需要组件匹配的欧氏距离），并通过同步投影梯度优化，为基于MMD的实例提供了收敛到驻点的理论保证。 |
| [^193] | [Advantage Scale Calibration Imbalance in Group-Relative Optimization under Low-Variance Rewards: Diagnosis and Bounded Recovery](https://arxiv.org/abs/2609.19164) | 本文诊断了低方差奖励下群体相对优化中优势尺度校准失衡的问题，提出三方校准接口揭示RLOO/Dr.GRPO与GRPO各自的失控行为，并通过奖励分辨率协议与MaxNorm-AC过滤亚分辨率噪声、实现对可信小差距的有界恢复。 |
| [^194] | [VisKG-LM: Compiling Knowledge Graphs into Visual Memory for Multiple-Choice Question Answering](https://arxiv.org/abs/2609.19158) | VisKG-LM 将检索到的知识图谱子图一次性离线编译为保留分支结构的可视化图像并缓存为只读记忆，使语言模型在推理时无需重复在线编码图结构，从而将图编码与语言推理解耦，提升多项选择题问答的效率。 |
| [^195] | [Stop Removing Stopwords: How an Inherited Preprocessing Default Distorts Legal Text-as-Data](https://arxiv.org/abs/2609.19153) | 本研究通过穷尽式单词消融实验，首次直接针对下游分类目标验证停用词去除这一沿袭自信息检索时代、从未被验证过的预处理默认设置，揭示其可能扭曲实证法学中基于TF-IDF和线性分类器的文本数据分析结果。 |
| [^196] | [Sampling Reveals Style: Unsupervised, Training-Free Discovery of Prompt-Conditional Stylistic Axes in LLM Activations](https://arxiv.org/abs/2609.19150) | 该论文提出一种无需训练的无监督方法，通过对同一提示的高温重复采样补全进行主成分分析，自动发现并标记大语言模型激活中与提示相关的风格轴，并通过245个人类风格标注验证了其与人类自发风格需求的高度契合。 |
| [^197] | [Subliminal Prompting Beyond Static Geometry: Causal Depth and Multi-Token Confounds](https://arxiv.org/abs/2609.19149) | 该论文首次将标记纠缠解释中的相关性测量与因果性测量明确区分开，通过在多个深度进行隐藏状态复制的因果干预实验，发现静态输出向量相似度随模型规模增大而失去预测力，而隐藏状态对所传递特质的因果控制能力则显著增强。 |
| [^198] | [How Model Growth, Recursion, and Boundary Operators Influence Scaling Exponents](https://arxiv.org/abs/2609.19107) | 该研究证明架构干预（尤其是模型增长和递归深度）可以改变预训练的缩放指数，使7.4B模型增长架构以约20倍更少的计算量匹配GPT-3 13B的性能，且计算效率优势随规模扩大而增加。 |
| [^199] | [Double descent is the principle of least action](https://arxiv.org/abs/2609.19076) | 本文用统计力学解释了机器学习中的双重下降现象：将随机梯度训练视为温度为 $T$ 的粒子在损失能量景观上的扩散，有限时间的扩散带来有效权重衰减，使每个参数成为二次自由度，从而由能量均分定理导出测试误差随参数数量先升后降的规律。 |
| [^200] | [MoRE: Mixture of Reused Experts](https://arxiv.org/abs/2609.18176) | MoRE通过在相邻层组之间共享专家池并引入可学习的深度嵌入对每层输入进行条件化，在不增加参数的情况下扩展路由组合多样性，实现了比标准MoE和权重共享方法更低的困惑度和更强的下游性能。 |
| [^201] | [Not All Layers Need Tuning: Diagnosing and Directing Adaptation in Vision-Language-Action Models](https://arxiv.org/abs/2609.18084) | 该研究通过在五个不同架构的VLA模型上测量区域隔离微调下的适应成本，揭示了不同类型的分布偏移会系统性集中在特定网络区域（外观变化集中于视觉编码器、指令变化集中于语言骨干、新物体变化集中于视觉编码器与动作头），并提出了一种仅凭十个无标注观测、无需微调即可诊断适应成本并针对性分配适配容量的流水线。 |
| [^202] | [Fathom: Per-Query Read Depth for Sparse Decoding over Offloaded KV Caches](https://arxiv.org/abs/2609.17652) | Fathom提出一种让每个查询自适应决定键通道读取位数的稀疏解码方法，通过比特平面存储与逆注水式比特预算分配，在百万token卸载KV缓存场景下实现比现有136位扫描方法快1.67倍的GPU解码速度，同时保持更低注意力误差。 |
| [^203] | [Recovering Physical Parameters from Fragmented Observations via Exact Distributed Spline Merging](https://arxiv.org/abs/2609.16579) | 本文提出精确分布式样条合并方法，各数据持有者仅需共享局部Gram矩阵和矩向量即可获得与集中式拟合数学上完全相同的解，并通过场重建与导数提取流程从分布式碎片观测中实现物理参数推断。 |
| [^204] | [Not All Relations Are Equal: Relation-Balanced and Calibrated Graph Learning for Provenance-Based Intrusion Detection](https://arxiv.org/abs/2609.16462) | 提出无监督框架RECAL，通过关系平衡的掩码图学习捕捉稀有交互模式，并结合针对各关系良性错误分布的误差校准机制，在DARPA E3数据集上实现最高99.99%的F1分数，有效降低基于溯源入侵检测中的误报和漏检风险。 |
| [^205] | [Task-Directed Residual AddUNet:Perfect-Reconstruction Routing for Full-Rate Representations](https://arxiv.org/abs/2609.15857) | 本文证明受约束加性U-Net精确等价于完美重构多速率滤波器组，并提出残差全速率PR架构，可将任务无关信息从面向任务的表示中路由出去，同时保证任意线性或非线性路由算子下的精确重构，且无需可逆性、重构损失或解码器。 |
| [^206] | [When Faster VLA Deployment Changes Closed-Loop Behavior: Task Success-Latency Analysis of SmolVLA Across PyTorch and ONNX Variants](https://arxiv.org/abs/2609.14146) | 该论文通过在消费级GPU上系统对比SmolVLA的PyTorch与ONNX部署变体，揭示了推理延迟降低约一半会显著改变闭环任务行为（Spatial成功率从70%降至约40%，Object保持不变），且导出图的审计表明所谓INT8并非真正的算子级量化，而静态语言宽度调整可部分恢复性能损失。 |
| [^207] | [NeuroFlex: Lossless Element-Level ANN-SNN Co-Execution for Efficient Sparse Inference](https://arxiv.org/abs/2609.14092) | NeuroFlex是首个能在单个输出元素级别无损切换ANN和SNN执行模式的加速器，通过将整数精确的ANN-SNN等价性扩展到元素级别并配合成本引导调度器，实现了97-99%的PE利用率，相比纯ANN基线降低57-67%的EDP，相比纯SNN基线获得高达2.5倍加速。 |
| [^208] | [An Efficient and Modular Framework for Targeted Harm Mitigation in LLMS](https://arxiv.org/abs/2609.13624) | 提出了一种结合Activated LoRA适配器与上下文感知路由机制的模块化纠正框架，可在生成过程中以低延迟、有针对性的方式缓解大语言模型的有害输出，同时提升模型对齐性能。 |
| [^209] | [Optimal Value Inference for Reinforcement Learning](https://arxiv.org/abs/2609.09981) | 该论文提出了一种基于Neyman正交性的去偏估计方法，通过softmax近似的自诱导贝尔曼方程构建冗余参数，实现了强化学习中最优价值的有效统计推断，且在视界发散和行为策略随时间变化的情况下依然保持渐近正态性。 |
| [^210] | [Sparse Data Augmentation for Optimization with Provable Guarantees](https://arxiv.org/abs/2609.08133) | 该论文证明了在非凸几何机器学习优化中，使用优化前采样的少量固定数据变换进行稀疏数据增强，梯度下降仅需对数级加多项式级的群变换查询次数，即可在概率保证下逼近完全数据增强目标的稳定点。 |
| [^211] | [LayerRoute: Action-Conditioned Mixture-of-Layers Routing for Vision-Language-Action Policies](https://arxiv.org/abs/2609.06079) | 提出LayerRoute，一个动作条件的混合层路由接口，使VLA策略能够根据具体动作需求自适应地访问和混合VLM不同层次的表示，突破了传统固定层分配的限制。 |
| [^212] | [Verify Before You Distill: Prompt-Level Teacher Gating for On-Policy Distillation](https://arxiv.org/abs/2609.02998) | 该论文提出教师门控在线策略蒸馏（TGOPD），通过经验证器评分的教师探测在提示级别先验证教师模型的可靠性，将可靠提示路由到密集OPD监督、不可靠提示路由到基于验证器的GRPO，从而避免“自信但错误”的教师模型诱导误导性更新。 |
| [^213] | [Curvature Cryptanalysis of Smooth Transformer Feed-Forward Networks](https://arxiv.org/abs/2608.28843) | 该论文提出了一种基于曲率（二阶Hessian信息）的密码分析方法，证明采用GELU或SiLU等平滑激活函数的Transformer前馈网络会通过二阶泄漏通道泄露其隐藏权重方向，仅需8193次黑盒查询（16个投影Hessian）即可高精度提取FFN结构，并将查询成本降低了16倍。 |
| [^214] | [A Finite Sample Analysis for Quantile Temporal Difference Learning in Distributional Reinforcement Learning](https://arxiv.org/abs/2608.27313) | 本文首次为表格型分布强化学习中的同步分位数时序差分学习提供了全局有限样本保证，通过分离稳定性机制，证明了其收敛速率对分位数数量无多项式依赖。 |
| [^215] | [GRAS: Guided Reduced-Variance Proposals and Adaptive Selection for Training-Free Reward Alignment in Discrete Diffusion](https://arxiv.org/abs/2608.26585) | 本文提出GRAS方法，通过Rao-Blackwell化提议和自适应温度选择，在不增加降噪器成本的情况下，显著提升离散扩散模型免训练奖励对齐的效率和稳定性。 |
| [^216] | [D$^3$-MOPD: Adaptive Dynamic Domain ScheDuling for Efficient Multi-Teacher Distillation](https://arxiv.org/abs/2608.24987) | 本文提出了一种零开销的动态领域调度方法，通过利用训练中已有的反向KL信号在线调整数据混合比例，解决了多教师蒸馏中固定混合导致的计算浪费问题。 |
| [^217] | [CatchBench: When Can an Agent Failure Be Caught?](https://arxiv.org/abs/2608.22808) | CatchBench提出了一个在三种信息状态（运行前配置PRE、运行中轨迹前缀LIVE、完成后轨迹POST）下评估智能体故障审计能力的基准，通过七项任务契约对72个参赛者进行评分，并发现现有方法在大多数对比中缺乏区分能力。 |
| [^218] | [Spending Scarce Confirmatory PET Measurements: Target-Aligned Validation in A4/LEARN](https://arxiv.org/abs/2608.22223) | 本文提出了一种目标对齐的PET验证策略，通过结合目标影响和残差不确定性来优化稀缺确认性测量的分配，避免在影响弱的受试者上浪费资源。 |
| [^219] | [GigaBrain-WBC-0.5: A Behavior World Model for Robust Whole-Body Control with Environment Interaction](https://arxiv.org/abs/2608.18234) | 本文提出了首个行为世界模型GigaBrain-WBC-0.5，通过因果Transformer联合预测动作、状态和潜在行为命令，使机器人能够建模环境交互，实现鲁棒的全身控制。 |
| [^220] | [Teach and Grow: An Agent-Centered Architecture for General Robot Learning](https://arxiv.org/abs/2608.17209) | 本文提出了一种名为“教学与成长学习”（TGL）的智能体中心架构，通过将少量演示转化为可复用的技能模块，并动态组合与修正，以降低通用机器人学习中的“再训练税”，提升其在未覆盖场景中的适应能力。 |
| [^221] | [Reference-free logged energy-oracle recovery for neural approximations of symmetric coercive variational problems: conforming Riesz reconstruction and archive-level selection](https://arxiv.org/abs/2608.16473) | 本文提出一种无参考的选择规则，通过符合Riesz监测器恢复神经近似的日志能量预言，并揭示档案选择的顺序敏感性，确保在有限检查点下准确选择最优近似。 |
| [^222] | [Attributing Preprocessing Invariance in Spectral Foundation Models](https://arxiv.org/abs/2608.14227) | 本文指出谱基础模型中的预处理不变性可能源于输入归一化本身，而非模型学习，并主张在评估时应将归一化单独作为基线。 |
| [^223] | [A Quantum/Classical Example Oracle Separation for Making Things Up](https://arxiv.org/abs/2608.11648) | 本研究首次证明，在Oracle模型下，存在某些分布只能被量子示例学习者高效生成，而经典示例学习者无法做到，从而揭示了量子示例的独特优势。 |
| [^224] | [Beyond On-Policy Exploration: Integrating External Policy Rollouts for Reinforcement Learning in Diffusion Language Models](https://arxiv.org/abs/2608.01717) | 提出ERILS方法，通过控制外部轨迹长度并分别处理不同来源的奖励，将更强外部策略生成的高奖励轨迹整合到扩散语言模型的强化学习训练中，有效解决了在线策略成功样本稀缺导致的训练进展受限问题。 |
| [^225] | [When Low CER is Not Enough: An Analysis of Hallucinations in Vision-Language OCR Systems on Historical Uruguayan Documents](https://arxiv.org/abs/2607.24077) | 该研究通过分析乌拉圭独裁时期历史文献，揭示视觉语言OCR模型虽然在字符错误率上优于传统方法，但存在标准指标无法检测的幻觉问题（如拼写规范化、虚假内容生成和语义替换），表明仅凭低CER不足以评估其在档案转录中的可靠性。 |
| [^226] | [Molt: A Scalable PyTorch-Native Training Framework for Agentic Reinforcement Learning](https://arxiv.org/abs/2607.21653) | 提出了 Molt 框架，通过可组合模型并行、统一智能体接口、全异步 rollout 与优化以及分布式经验存储，实现了万亿参数规模下的智能体强化学习训练，且无需修改现有智能体的执行逻辑。 |
| [^227] | [PhysCoRe: Physics-Corrected Residual World Models for Material-Aware Deformable Dynamics](https://arxiv.org/abs/2607.20653) | PhysCoRe通过将可微分物质点法（MPM）模拟器与视觉材料推断模块（MfM）和动力学残差校正模块（RfD）相结合，实现了对可变形物体在机器人操作下既符合物理规律又能泛化材料属性的动力学预测。 |
| [^228] | [Scalable Policy Optimization for Networked Multi-Agent Reinforcement Learning with Continuous State-Action Spaces](https://arxiv.org/abs/2607.18554) | 该论文针对连续状态-动作空间的网络化多智能体强化学习，对CDCPG算法进行了理论分析，通过局部随机傅里叶特征和最小二乘时序差分评论家，推导出空间残差与有限特征残差分离的动作值表示，并借助全局综合转移近似界和投影贝尔曼论证在无需逆条件乘子的情况下控制群体预测误差，从而实现可扩展的策略优化。 |
| [^229] | [Building a Neural Network from Scratch: Implementation, Evaluation, and Optimization](https://arxiv.org/abs/2607.16682) | 本文从零实现了一个不依赖自动微分和现成深度学习模块的完整神经网络框架，涵盖多层架构、激活函数、正则化和先进优化器，并通过多分类任务验证了其正确性、数值稳定性与泛化能力。 |
| [^230] | [Interpretable and Calibrated Classification of Clinical Data Using Supervised Feature Binarization](https://arxiv.org/abs/2607.15394) | 本文提出一种结合监督式卡方引导二值化与伯努利朴素贝叶斯的临床分类框架，在保持完全可解释性和良好校准的同时，对连续医疗数据实现高精度分类（三个基准数据集AUC为0.800至0.984）。 |
| [^231] | [Multi-Axis Max@K Reinforcement Learning for Representative Diversity in Text-to-Image Generation](https://arxiv.org/abs/2607.14962) | 该论文提出了多轴 max@K 这一基于分组的强化学习目标，通过仅奖励提升各模式组内最大值的样本的信用分配机制，让不同样本贡献于不同语义模式，从而提升文本到图像生成模型对预定义目标模式的覆盖及代表性多样性。 |
| [^232] | [When Data Imbalance Helps: Robust Generalization Through Shortcut Saturation](https://arxiv.org/abs/2607.10116) | 本文发现一个反直觉现象：在容量足够的模型中，数据不平衡反而通过“捷径饱和”机制促进鲁棒泛化，但在容量较小的模型中不平衡会使模型陷入对捷径特征的依赖。 |
| [^233] | [Faithful, Not Corrective: Model Capability Governs Message-Format Effects in Multi-Hop Agent Relays](https://arxiv.org/abs/2607.09678) | 论文通过六跳、五种格式的多智能体消息接力实验发现，消息格式效应由接力模型自身能力决定：强接力器对所有格式几乎无损传递，且接力行为表现为忠实复制而非纠错。 |
| [^234] | [Prompt-Driven Exploration](https://arxiv.org/abs/2607.08837) | 本文提出一种利用视觉-语言模型从强化学习展开视频中自动诊断并重写提示的方法，以实现对弱策略的全局探索，而无需依赖稀疏奖励。 |
| [^235] | [A Transferable Learned Temporal Prior for Transmission Reconstruction and Decision-Relevant Uncertainty in Real Outbreak Labels](https://arxiv.org/abs/2606.30842) | 该研究从十一种其他疾病中学习并提前锁定了一个时间先验模型，无需重新拟合即可迁移应用于安第斯病毒的传播链重构任务并显著优于基线，同时评估了已发表传播链作为真实标签的可靠性及其对决策的影响。 |
| [^236] | [Accelerating Q-learning through Efficient Value-Sharing across Actions](https://arxiv.org/abs/2606.29806) | 本文提出一种无参数的均值扩展层，通过在状态内的动作之间共享价值并学习低范数表示，加速强化学习中动作价值的学习过程。 |
| [^237] | [When Retrieval Metrics Mislead: Measuring Policy Signal in Long-Horizon Tool-Use Agents](https://arxiv.org/abs/2606.23937) | 该研究发现精确匹配检索召回率是一个具有误导性的代理指标——即使正确的治理规则仅在 7% 的情况下被排名第一检索到，检索到的断言仍能让分类器取得与使用黄金规则几乎相同的性能。 |
| [^238] | [Do LLM Attribution Metrics Transfer? Auditing Retrieval-Augmented Generation Evaluation Across Datasets and Constructs](https://arxiv.org/abs/2606.23915) | 该研究系统审计了八种检索增强生成归因自动指标，发现在生成答案归因构念下没有任何指标能在所有数据集上保持与最佳指标一致的性能，不同数据集上的指标排名甚至完全反转，因此实践中将这些指标视为可互换是不成立的。 |
| [^239] | [Exploring a Layer-Wise Design Space for KV Cache Eviction](https://arxiv.org/abs/2606.15157) | 该论文提出在Transformer各层间组合不同的KV缓存淘汰方法构成异构路由，发现在相同缓存预算下，这种逐层的异构策略在LongBench大多数任务上优于全模型统一的同构淘汰策略。 |
| [^240] | [Measurement Under Selection: Decoy-Calibrated Failure Audits for Language Models](https://arxiv.org/abs/2606.09046) | 本文提出Janus框架，通过随机打乱属性标签生成的“诱饵”来校准阈值，只有超出偶然波动水平并在保留数据上稳健成立的错误模式才被报告，从而避免语言模型失效审计中出现虚假的模式发现。 |
| [^241] | [SoK: Reconstruction Attacks on Synthetic Tabular Data (Insights from Winning the NIST CRC)](https://arxiv.org/abs/2606.08372) | 本文系统化了针对合成表格数据的重建攻击，提出了按结构组织的攻击分类体系，对14种攻击与13种合成数据生成方法在5个基准数据集上进行了大规模实证评估，提出了包括最强攻击CoBP-RA在内的新攻击，并提供了解读攻击成功含义的记忆测试方法论。 |
| [^242] | [EssentialGIN: a new approach for gene essentiality prediction based on graph isomorphism neural networks](https://arxiv.org/abs/2606.07700) | 本研究提出EssentialGIN方法，通过改进图同构神经网络来保留PPI网络的拓扑特征，并整合基因表达、直系同源和亚细胞定位等生物信息，从而实现对必需基因的准确预测。 |
| [^243] | [Post-Boundary Bridge: Must Local Attention Go Global Between Global Layers?](https://arxiv.org/abs/2606.02680) | PBB方法在混合Transformer中保留块内因果注意力，仅在块边界之间添加直接连接，而将长程通信完全交给全注意力层处理，在2.05亿至20.7亿参数规模的稠密和混合专家模型上实现了接近全注意力模型的性能，同时提升了来源检索能力。 |
| [^244] | [Near-Optimal Machine Unlearning Utility for Smooth Strongly Convex Losses](https://arxiv.org/abs/2606.01527) | 本文为光滑强凸损失下的近似 $(\varepsilon, \delta)$-机器遗忘证明了接近紧致的总体超额风险上下界，确定了遗忘的最优统计代价为采样误差加遗忘惩罚，并揭示出 $(\varepsilon, \delta)$-遗忘相比纯 $\varepsilon$-遗忘并无统计优势。 |
| [^245] | [Genetic algorithm vs. gradient descent for training a neural network architecture dedicated to low data regimes in small medical datasets](https://arxiv.org/abs/2605.27411) | 本文为DEBI-NN架构设计了一种专用的空间反向传播方案，使其能够采用梯度下降进行训练，并在多个数据集上与遗传算法的训练性能进行了比较。 |
| [^246] | [Adversarial Water-Filling: Theory, Algorithms, and a Domain-Specific Wireless Foundation Model](https://arxiv.org/abs/2605.26163) | 本文提出对抗性注水（AWF）问题及其理论与算法，并构建了基于置换不变信道表示和约束感知图神经网络的领域专用无线基础模型，用于解决多运营商低轨卫星频谱共享中的竞争性功率分配问题。 |
| [^247] | [Batch Normalization Amplifies Memorization and Privacy Risks](https://arxiv.org/abs/2605.24420) | 本研究实证发现批归一化（BN）层会显著加深模型对离群样本的记忆，而这种放大的记忆直接转化为更高的隐私泄露风险，使模型更容易受到成员推断攻击。 |
| [^248] | [Reinforcement Learning for Graph Generation under a Hard Assortativity Constraint](https://arxiv.org/abs/2605.23285) | 本文提出一种强化学习框架，通过保度重连的定向传输策略使图精确满足硬同配性约束，在生成成本降低至少一个数量级的同时保留超过98%的构型多样性，并能从小图训练泛化至不同规模与拓扑。 |
| [^249] | [Multi-Resolution Attribution from Adaptive Routing State](https://arxiv.org/abs/2605.22866) | 本文证明自适应分层系统中学习到的路由状态本身即可定义一种多分辨率的一致性归因——叶子节点值为路径权重乘积、内部节点为前缀乘积，且细粒度读数恰好加和等于粗粒度读数，并在LLM、人口普查、智能体和电信网络等层次结构中于多个层级揭示出有意义结构。 |
| [^250] | [Jacobian-Guided Anisotropic Noise Reshaping for Enhancing Representation Utility under Local Differential Privacy](https://arxiv.org/abs/2605.16812) | 本文通过利用下游模型雅可比矩阵识别任务关键子空间，并将标准LDP的各向同性噪声重塑为各向异性分布，从而在不牺牲隐私保证的前提下显著提升数据表示效用。 |
| [^251] | [EfficientTDMPC: Improved MPC Objectives for Sample-Efficient Continuous Control](https://arxiv.org/abs/2605.16692) | EfficientTDMPC通过动力学模型集成、跨不同展开深度平均回报估计以及对规划器目标施加不确定性惩罚来减少模型与价值网络的估计误差，并结合缓冲区数据新鲜度等实用改进，从而在连续控制任务中实现更高效的模型强化学习，并能更好地利用更高的更新-数据比。 |
| [^252] | [Clin-JEPA: A Multi-Phase Co-Training Framework for Joint-Embedding Predictive Pretraining on EHR Patient Trajectories](https://arxiv.org/abs/2605.10840) | 提出Clin-JEPA框架，首次将联合嵌入预测架构（JEPA）引入电子健康记录患者轨迹建模，通过编码器与预测器的多阶段协同训练，使LLM编码器的潜空间围绕患者生理动力学组织，从而实现潜空间中的患者轨迹模拟。 |
| [^253] | [Fast Training of Mixture-of-Experts for Time Series Forecasting via Expert Loss Integration](https://arxiv.org/abs/2605.10330) | 提出一种融合专家特定损失与部分在线学习策略的自适应混合专家框架，解决了小门控权重导致的优化难题，在保持计算效率的同时显著提升了时间序列预测性能。 |
| [^254] | [Learning to Theorize the World from Observation](https://arxiv.org/abs/2605.03413) | 本文提出Learning-to-Theorize学习范式及世界理论模型NEO，通过将潜在程序作为习得的“思维语言”，从原始非文本观察中构建显式、可组合、可执行的世界解释性理论。 |
| [^255] | [Model-Aware Data Cleaning for Tabular Foundation Models](https://arxiv.org/abs/2604.25154) | 该论文提出L2C-TFM方法，通过引入以Wasserstein距离分布稳定性作为正则项的模型感知奖励，训练强化学习策略来智能编排表格数据清洗算子，从而改善表格基础模型在含脏数据场景下的表现。 |
| [^256] | [Green-ELM: Efficient Analytic Learning via High-Dimensional Random Projections](https://arxiv.org/abs/2604.15613) | Green-ELM通过高维随机投影和闭式解析解（Moore-Penrose伪逆、LU与Cholesky分解）一次性求解输出层，完全避免了反向传播，在MNIST上达到98.10%准确率的同时大幅降低计算开销。 |
| [^257] | [Transcriptomic Models for Immunotherapy Response Prediction Show Limited Cross-cohort Generalisability](https://arxiv.org/abs/2604.05478) | 本研究系统评估了九种最先进的转录组学免疫治疗反应预测模型，发现它们在独立外部数据集上的预测性能有限，表明现有模型的跨队列泛化能力不足，需要在临床应用前进一步改进。 |
| [^258] | [Automated Membership Inference Attacks (AutoMIA): Discovering MIA Signal Computations using LLM Agents](https://arxiv.org/abs/2603.19375) | 提出AutoMIA框架，利用大语言模型智能体自动探索和发现针对特定目标模型与数据集定制的新型成员推断攻击信号计算方法。 |
| [^259] | [MAPLE: Metadata Augmented Private Language Evolution](https://arxiv.org/abs/2603.19258) | MAPLE通过引入元数据增强，解决了私有演化（PE）方法在私有数据分布偏离基础模型预训练先验时的初始化瓶颈问题，实现了更高效的基于API的差分隐私合成数据生成。 |
| [^260] | [TTSR: Test-Time Self-Evolving via Reflection](https://arxiv.org/abs/2603.03297) | TTSR通过让单个模型交替扮演学生和教师角色，基于反思后合成的范式，在测试时针对失败轨迹生成变体问题，从而克服了缺乏可学习样本和探索效率低下的瓶颈。 |
| [^261] | [High-Resolution Range Profile Classifiers Require Aspect-Angle Awareness](https://arxiv.org/abs/2603.00087) | 本研究表明，高分辨率距离像分类器通过显式利用方位角信息可平均提升约7%的分类准确率，且即使方位角通过因果卡尔曼滤波器在线估计获得，大部分性能增益依然能够保留。 |
| [^262] | [Exploring Sparsity and Smoothness of Arbitrary Lp Norms in Adversarial Attacks](https://arxiv.org/abs/2602.06578) | 该论文系统研究了 ℓp 范数中参数 p（p∈[1,2]）的取值如何影响对抗扰动的稀疏性与平滑性，并提出了基于平滑操作和一阶泰勒近似的平滑性度量框架，填补了范数选择与扰动结构特性之间关系的研究空白。 |
| [^263] | [Perturbing the Phase: Analyzing Adversarial Robustness of Complex-Valued Neural Networks](https://arxiv.org/abs/2602.06577) | 本文提出了专门针对复值输入相位信息的"相位攻击"并推导了常用对抗攻击的复值版本，发现复值神经网络在某些场景下比实值神经网络更鲁棒，但两者都对相位变化极为敏感，相位攻击造成的性能下降超过同等强度的常规攻击。 |
| [^264] | [Rethinking the Design Space of Reinforcement Learning for Diffusion Models: On the Importance of Likelihood Estimation Beyond Loss Design](https://arxiv.org/abs/2602.04663) | 本文系统解耦并分析了扩散模型强化学习设计空间中的三个因素，发现采用仅从最终生成样本计算的基于证据下界（ELBO）的似然估计器，是决定算法有效性与效率的最关键因素，其重要性超越了损失函数的设计本身。 |
| [^265] | [QUATRO: Query-Adaptive Trust Region Policy Optimization for LLM Fine-tuning](https://arxiv.org/abs/2602.04620) | QUATRO提出了一种查询自适应信赖域策略优化方法，通过直接强制执行信赖域约束来替代GRPO的启发式裁剪与组归一化，实现了大语言模型微调中稳定、熵受控的策略更新。 |
| [^266] | [Score-based diffusion models for severely ill-posed problems in diffuse optical tomography](https://arxiv.org/abs/2602.03449) | 本文针对扩散光学层析成像这一严重不适定的逆问题，提出了一种由学习成分与基于模型的成分相结合的混合分数正则化策略，以提升基于分数的扩散模型在真实实验测量条件下的重建质量。 |
| [^267] | [On the Inherent Privacy Amplification of Missing Data](https://arxiv.org/abs/2602.01928) | 该论文提出了一个将缺失数据融入差分隐私的新框架，从形式上证明并刻画了数据缺失对隐私保护的固有放大效应，即缺失特征本身天然地增强了个人隐私保护。 |
| [^268] | [Comparison of Image Processing Models in Quark Gluon Jet Classification](https://arxiv.org/abs/2602.00141) | 该研究在使用相同三通道喷注图像表示的条件下比较了CNN、ViT和Swin变换器，发现保留强局部成分的CNN和Swin模型始终优于ViT，表明局部喷注子结构在夸克-胶子判别中起关键作用。 |
| [^269] | [Elastic Spectral State Space Models for Train-Once Budgeted Inference](https://arxiv.org/abs/2601.22488) | 提出ES-SSM框架，通过对SSM序列算子的Hankel谱逼近并结合输入自适应门控与预算丢弃，实现“一次训练即可导出多预算模型”的弹性序列建模，避免了为不同资源约束重复训练或蒸馏模型的成本。 |
| [^270] | [Why $\beta_1 = \beta_2$ Is Dynamically Special in Adam](https://arxiv.org/abs/2601.21739) | 本文揭示了Adam优化器中 $\beta_1 = \beta_2$ 在动力学上特殊的具体机制：连续时间极限下，归一化更新中与两个记忆时间差成正比的幅度滞后项恰好在两参数相等时消失，使得对角线区域成为结构上不存在失配诱发响应的唯一情形。 |
| [^271] | [L2R: Low-Rank and Lipschitz-Controlled Routing for Mixture-of-Experts](https://arxiv.org/abs/2601.21349) | 提出L2R统一路由框架，通过在共享低秩潜在路由空间中进行专家分配，并引入饱和内积评分（SIPS）显式控制路由函数的Lipschitz行为，重塑MoE的路由空间与评分几何，从而提升路由可区分性与专家专业化的稳定性。 |
| [^272] | [Ambient Dataloops: Generative Models for Dataset Refinement](https://arxiv.org/abs/2601.15417) | 提出了 Ambient Dataloops 迭代框架，通过数据集与模型的协同演化逐步提升数据质量，并借助 Ambient Diffusion 技术避免自消耗循环，在图像生成和从头蛋白质设计中取得最先进性能。 |
| [^273] | [SpaRRTa: A Synthetic Benchmark for Evaluating Spatial Intelligence in Visual Foundation Models](https://arxiv.org/abs/2601.11729) | 本文提出了SpaRRTa基准，通过合成图像评估视觉基础模型的空间关系识别能力，以区分真正的空间意识与对特定3D任务的过拟合。 |
| [^274] | [Precision autotuning for linear solvers via contextual bandit-based RL](https://arxiv.org/abs/2601.00728) | 提出了一种基于上下文赌博机的强化学习框架，通过离散化特征和增量动作值估计为线性求解器动态选择最优精度配置，在迭代精化中实现精度与计算效率的平衡。 |
| [^275] | [Poodle: Seamlessly Scaling Down Large Language Models with Just-in-Time Model Replacement](https://arxiv.org/abs/2512.05525) | 本文提出即时模型替换（JITR）的愿景，在检测到LLM调用中的重复任务时透明地将模型替换为针对该任务优化的更廉价的小模型，在保持LLM易用性的同时显著节省成本和能源。 |
| [^276] | [Unexplored flaws in multiple-choice VQA make benchmarking unreliable](https://arxiv.org/abs/2511.22341) | 该论文揭示了MC-VQA基准测试中存在严重的提示格式敏感性问题——即使消除了选项顺序的影响，仅仅改变选项ID、分隔符等语义无关的格式元素就会导致模型性能排名频繁逆转，这表明当前MC-VQA作为多模态大语言模型评估基准的可靠性存在根本性缺陷。 |
| [^277] | [Pre-train to Gain: Robust Learning Without Clean Labels](https://arxiv.org/abs/2511.20844) | 该论文提出先在目标数据集上进行域内自监督预训练、再进行标准监督训练的方法，无需任何干净标签子集即可获得对标签噪声更鲁棒的模型。 |
| [^278] | [Learn2Drive: A neural network-based framework for socially compliant automated vehicle control](https://arxiv.org/abs/2510.21736) | 该论文提出了一种融合社会价值取向的神经网络自适应巡航控制框架，使自动驾驶车辆能够兼顾对人类驾驶车辆和交通流的影响，充当移动交通调节器以缓解拥堵、提升整体交通效率。 |
| [^279] | [SGM: A Statistical Godel Machine for Risk-Controlled Recursive Self-Modification](https://arxiv.org/abs/2510.10232) | 本文提出了首个针对递归自我修改的统计安全层——统计哥德尔机（SGM），用统计置信度检验（e值、Hoeffding界）替代无法在随机高维环境中实现的形式化证明要求，并通过全局误差预算和确认触发的调和支出机制（CTHS）实现累积风险的可控性。 |
| [^280] | [The Environmental Impacts of Language Model Training Keep Rising Now is the Time to Catch Impacts on the Rebound](https://arxiv.org/abs/2510.09022) | 研究发现，尽管采用了硬件、算法和碳排优化等减耗策略，过去十年语言模型训练的能源消耗和环境影响仍呈指数级增长，表明存在回弹效应，且硬件的影响必须在全生命周期内评估。 |
| [^281] | [Watermarking Diffusion Language Models](https://arxiv.org/abs/2509.24368) | 本文提出了首个专为扩散语言模型设计的水印技术，通过在期望意义上应用水印并促进增强水印强度的词元生成，在保持检测器不变的前提下实现了超过99%的真阳性率且对生成质量影响极小。 |
| [^282] | [Low-rank Orthogonalization for Large-scale Matrix Optimization with Applications to Foundation Model Training](https://arxiv.org/abs/2509.11983) | 该论文提出了利用神经网络训练中梯度低秩特性的低秩正交化方法，并由此发展出低秩矩阵符号梯度下降和低秩Muon优化器，为大规模基础模型训练提供了高效的矩阵优化方案。 |
| [^283] | [Evaluating Out-of-Distribution Robustness in Graph-Based Android Malware Classification: A New Principled Benchmark](https://arxiv.org/abs/2508.06734) | 该论文提出了一个新的基准测试套件来评估基于图的Android恶意软件分类器在协变量偏移和域偏移下的分布外鲁棒性，发现其性能下降高达45%，并提出通过结合轻量级元数据与LLM代码嵌入的语义增强框架来解决现有基准仅依赖图结构的局限性。 |
| [^284] | [Neural Langevin Machine: a local asymmetric learning rule can be creative](https://arxiv.org/abs/2506.23546) | 该论文提出“神经朗之万机”这一生成模型，其核心创新在于仅依赖局部神经信号的非对称、发放速率调整学习规则，既具有生物学合理性，又能实现创造性图像生成、去噪以及从记忆到泛化的转变。 |
| [^285] | [Spherical Cauchy Variational Autoencoders: Heavy Angular Tails and Exact KL Evaluation](https://arxiv.org/abs/2506.21278) | 提出球面柯西分布作为超球面变分自编码器的后验分布，兼具重角尾特性和精确的KL散度解析计算能力，克服了von Mises-Fisher分布需要拒绝采样和Power Spherical分布密度在对跖点归零的缺陷。 |
| [^286] | [Enabling automatic transcription of child-centered audio recordings from real-world environments](https://arxiv.org/abs/2506.11747) | 该论文提出一种自动检测长时间儿童音频录音中可被可靠转录语音片段的方法，突破以往ASR需完整处理录音的限制，实现了对真实环境录音中大部分语音的自动准确转录。 |
| [^287] | [A Network Science Approach to Granular Time Series Segmentation](https://arxiv.org/abs/2505.17640) | 该论文提出将时间序列细粒度分割形式化为图节点分类问题，通过可见性图变换结合图注意力网络（GAT）实现，其中WDPVG+GAT在59个数据集的基准上达到0.916的加权F1分数，且WDPVG、有向NVG和加权NVG三种图构建方法表现最佳且统计上不可区分。 |
| [^288] | [FOCAL: Fine-Grained Optimal-Transport-Driven Contrastive Alignment of Language and ECGs with Waveform Enhancement](https://arxiv.org/abs/2505.11939) | 该论文提出FOCAL框架，通过最优传输实现心电图局部波形片段与报告病理标签的细粒度精确对齐，并利用语义相似度矩阵缓解标签级对齐中的假阴性问题，从而提升零样本心电图解读性能。 |
| [^289] | [Out-of-Sample Embedding with Proximity Data: Projection versus Restricted Reconstruction](https://arxiv.org/abs/2505.06756) | 本文综述了基于邻近性数据的样本外嵌入的各种核方法，证明它们均可归结为投影或受限重构这两种基本策略之一，其中受限重构策略可简化为只需一维搜索的非线性优化问题。 |
| [^290] | [Sufficient Decision Proxies for Decision-Focused Learning](https://arxiv.org/abs/2505.03953) | 本文首次研究了优化问题在何种性质下使用特定决策代理（单场景预测或参数化分布估计）才是合理的，并据此为决策聚焦学习提出了学习复杂度几乎不受影响的替代性决策代理。 |
| [^291] | [R2DN: Scalable Parameterization of Contracting and Lipschitz Recurrent Deep Networks](https://arxiv.org/abs/2504.01250) | 本文提出R2DN，通过将线性时不变系统与1-Lipschitz深度前馈网络反馈互联来直接参数化稳定且鲁棒的递归深度网络，无需像递归平衡网络（REN）那样迭代求解平衡层，从而大幅加快GPU上的推理与训练速度，并使扩展网络规模和输入序列长度在计算上切实可行。 |
| [^292] | [Traffic Engineering in Large-scale Networks with Generalizable Graph Neural Networks](https://arxiv.org/abs/2503.24203) | TELGEN通过将“预测最优流量工程解”转化为“预测最优流量工程算法”的新思路，学习高效逼近经典最优TE算法的端到端求解过程，从而在大规模网络中实现了高效求解与跨多种网络条件的卓越泛化能力。 |
| [^293] | [A Large-Scale Vision-Language Dataset Derived from Open Scientific Literature to Advance Biomedical Generalist AI](https://arxiv.org/abs/2503.22727) | 该论文发布了源自PubMed Central开放获取文献的开源大规模多模态数据集Biomedica（含600万篇文章、2400万图像-文本对及专家标注），基于其训练的AI模型在嵌入、对话和检索等各任务类别中均超越了此前的开放系统。 |
| [^294] | [A Generative-AI Modeling Framework for Explainable Decision Support in Complex Geosteering Scenarios](https://arxiv.org/abs/2503.08509) | 该论文提出了一种集成生成对抗网络、集合模型更新方法和离散动态规划优化的实时AI驱动地质导向工作流程，为复杂定向钻井场景提供可解释的自动决策支持。 |
| [^295] | [Information-Geometric Inverse Distillation for Enhancing Adversarial Transferability](https://arxiv.org/abs/2502.17003) | 提出逆向知识蒸馏（IKD）机制，通过最大化代理模型上良性样本与对抗样本的预测分布差异来增强对抗攻击的迁移性，并从信息几何角度证明软标签交叉熵与KL散度在固定锚点下完全等价。 |
| [^296] | [Physics-Informed Support Vector Kernels via Green-Function Analogies and Jackson-Chebyshev Spectral Design](https://arxiv.org/abs/2502.11153) | 该论文提出了一种利用格林函数类比构造的Jackson阻尼Chebyshev支持向量核，通过显式特征映射保证半正定性并提供可检查的谱先验，从而在无需精确等同物理传播子的情况下实现物理信息驱动的核选择，并在多种物理系统回归任务中得到验证。 |
| [^297] | [Perspective of Software Engineering Researchers on Machine Learning Practices Regarding Research, Review, and Education](https://arxiv.org/abs/2411.19304) | 该研究首次从软件工程研究人员的视角出发，通过定性分析揭示了他们在研究、教学和评审中使用机器学习时的多样化实践、面临的挑战及建议。 |
| [^298] | [Interactive proofs for verifying (quantum) learning and testing](https://arxiv.org/abs/2410.23969) | 该论文证明了资源受限的学习者或测试者通常无法通过与不可信方进行经典交互来提升学习或测试效率，尤其是在涉及量子内存的绝大多数测试和学习问题上，内存受限的量子算法也无法借助此类交互克服自身的资源限制。 |
| [^299] | [DRL-AdaPart: DRL-Driven Adaptive STAR-RIS Partitioning for Fair and Efficient Resource Utilization](https://arxiv.org/abs/2407.06868) | 提出了一种基于深度强化学习的自适应STAR-RIS单元分区方法DRL-AdaPart，通过联合优化相移与子表面分配变量并引入惩罚项智能停用多余单元，在保证资源高效利用的同时，为静态和移动用户提供公平且高速的数据速率。 |
| [^300] | [A Computational Tropical Geometry Framework for Neural Networks](https://arxiv.org/abs/2405.20174) | 该论文提出了一个计算热带几何框架，给出了一种可将神经网络的线性区域计算为多面体显式并集的算法及其正确性证明，为分析神经网络表达能力提供了具体的计算工具。 |
| [^301] | [When fairness metrics fail: A utility-based perspective on $\varepsilon$-fairness](https://arxiv.org/abs/2405.09360) | 该论文提出了一个将决策后果纳入公平性评估的基于效用的框架，并证明一个决策过程即使满足ε-公平性概率度量，在考虑结果效用时仍可能达到最大程度的不公平。 |
| [^302] | [Estimation of multiple mean vectors in high dimension](https://arxiv.org/abs/2403.15038) | 通过凸组合的方法估计高维空间中不同概率分布的多维均值，引入了两种权重确定策略：一种通过测试程序识别低方差的相邻均值，提出了封闭形式插补公式；另一种通过最小化二次风险的上置信界确定权重，通过理论分析得出方法对经验均值的二次风险改进，在维度渐近的角度上渐近地接近 Oracle（Minimax）改进。 |
| [^303] | [ResNLS: An Improved Model for Stock Price Forecasting](https://arxiv.org/abs/2312.01020) | ResNLS是一种结合ResNet和LSTM的混合模型，通过强调相邻股票价格之间的依赖关系来提升股价预测效果，其中使用前5个交易日收盘价作为输入的ResNLS-5模型相比现有最先进方法至少提升了20%的性能。 |

# 详细

[^1]: 嵌入模型以奇特的方式进行测量

    Embedding Models Measure in Peculiar Ways

    [https://arxiv.org/abs/2609.20821](https://arxiv.org/abs/2609.20821)

    该研究发现嵌入模型对质量、距离、时间和体积等物理测量的表示十分微弱且奇特，主要受表面字符串相似性的强烈影响，而重新校准相似度也无法显著改善其与真实物理测量的对齐。

    

    嵌入空间定义了语义相似性和距离的概念。我们研究了这些嵌入是否反映了质量、距离、时间和体积等物理测量，这些物理测量具有唯一且客观的语义等价与距离概念。我们发现物理测量在嵌入空间中仅被微弱地建模，相反，可以观察到相当奇特的测量模式。进一步的分析表明，物理测量的嵌入表示受到表面字符串相似性的强烈影响，而重新校准相似性并不能实质性改善对齐效果。

    arXiv:2609.20821v1 Announce Type: new  Abstract: Embedding spaces define notions of semantic similarity and distance. We study whether those embeddings reflect physical measurements of mass, distance, time and volume, which admit a unique, objective notion of semantic equivalence and distance. We find that physical measurement is only weakly modeled in the embedding space, and that instead quite peculiar measurement patterns can be observed. Further analysis indicates that embedding representations of physical measurements are strongly influenced by superficial string similarity, and recalibration of similarity does not substantially improve the alignment.
    
[^2]: Paint-Anything：面向图像生成与编辑的统一任意颜色控制

    Paint-Anything: Unified Any-Color Control for Image Generation and Editing

    [https://arxiv.org/abs/2609.20816](https://arxiv.org/abs/2609.20816)

    Paint-Anything通过对象级颜色监督学习共享的十六进制提示接口，结合Paint-500K数据集和精确匹配的纯色锚点，实现了图像生成与编辑中任意精确颜色的统一控制。

    

    专业设计需要任意颜色控制：即能够在图像生成和编辑中为对象指定任意24位十六进制值的目标颜色。先前的工作已探索了颜色生成、编辑和上色，但通常依赖于专用的颜色表示或专门的推理流程。大语言模型的进展提供了一个更简单的起点：即使是紧凑的模型也能将十六进制值与颜色语义关联起来。我们提出了Paint-Anything，它通过对象级颜色监督学习一个用于生成和编辑的共享十六进制提示接口。我们开发了一个数据管道，通过对象定位、感知颜色标注和编辑对合成，从真实图像构建了Paint-500K数据集。由于阴影使得真实图像的标签只能提供近似颜色，我们用纯色锚点来补充这种监督，这些锚点的像素与其配对的十六进制值完全匹配。这些锚点仅在高噪声阶段使用。

    arXiv:2609.20816v1 Announce Type: cross  Abstract: Professional design requires any-color control: the ability to specify an object's target color with any 24-bit hex value for image generation and editing. Prior work has explored color generation, editing, and colorization, but often relies on dedicated color representations or specialized inference procedures. Advances in large language models offer a simpler starting point: even compact models can associate hex values with color semantics. We present Paint-Anything, which learns a shared hex-prompt interface for generation and editing through object-level color supervision. We develop a data pipeline that constructs Paint-500K from real images through object grounding, perceptual color labeling, and editing-pair synthesis. Since shadows make real-image labels only approximate colors, we complement this supervision with pure-color anchors whose pixels exactly match their paired hex values. These anchors are used only at high-noise ti
    
[^3]: 分布偏移如何塑造神经PDE代理模型的预训练收益？

    How Does Distribution Shift Shape Pretraining Gains in Neural PDE Surrogates?

    [https://arxiv.org/abs/2609.20814](https://arxiv.org/abs/2609.20814)

    本研究通过在翼型RANS数据上预训练神经PDE代理模型，量化了分布偏移的不同组成部分（湍流模型差异与翼型几何多样性）对预训练收益的影响，发现收益大小随微调样本量和偏移类型而变化。

    

    对神经PDE代理模型进行预训练，可以在几何形状或建模物理发生变化时减少所需的新CFD数据量。然而，分布偏移的不同组成部分如何影响这种收益目前仍不清楚。我们在一个翼型族的254,909个RANS解上预训练代理模型，然后在自由流范围相匹配的两个目标设置下，在新翼型族上进行微调：即相同的Spalart-Allmaras（SA）湍流模型，以及加入了$e^N$转捩建模的SA模型。在N=1000时，对于相同SA的目标，预训练模型达到了相当于从头训练模型使用3.25倍样本量的精度，而对于转捩建模目标则相当于2.58倍。当N=5000时，这一排序发生逆转（1.56倍对1.86倍）。在N=1000时，采样更多不同的翼型能降低两个目标上的误差，但只有相同SA目标的收益增长超过了观察到的抽样间波动（3.3倍至4.0倍）。

    arXiv:2609.20814v1 Announce Type: cross  Abstract: Pretraining a neural PDE surrogate can reduce the amount of new CFD data needed when geometry or modeled physics changes. However, it remains unclear how different components of distribution shift affect this benefit. We pretrain a surrogate on 254,909 RANS solutions from one airfoil family and fine-tune it on a new family under two target settings with matched freestream ranges: the same Spalart-Allmaras (SA) modeling and SA with added $e^N$ transition modeling. At $N=1000$, the pretrained model matches the accuracy of a model trained from scratch on $3.25\times$ as many samples for the same-SA target, but $2.58\times$ as many for the transition-modeled target. By $N=5000$, this ordering reverses ($1.56\times$ versus $1.86\times$). At $N=1000$, sampling more distinct airfoils lowers error on both targets, but only for the same-SA target is the gain increase larger than the observed draw-to-draw variation ($3.3\times$ to $4.0\times$). 
    
[^4]: 量化前沿大语言模型智能体的过度宣称倾向

    Quantifying Overclaiming Propensity in Frontier LLM Agents

    [https://arxiv.org/abs/2609.20812](https://arxiv.org/abs/2609.20812)

    本文提出OverclaimBench评估套件，首次量化了前沿LLM编码智能体在最终回复中“过度宣称”任务完成的倾向，并发现在67.9%的运行中智能体并未真正阅读所有被要求审查的文件。

    

    前沿编码智能体越来越被信任可以长时间自主工作，然而智能体的最终回复往往是用户能够看到的关于该工作的唯一记录。我们量化了前沿智能体“过度宣称”任务完成的倾向，这种失实陈述可能会误导用户。当智能体的最终回复与其上下文中的信息相矛盾时，即发生了过度宣称。这一定义无需对意图进行推断，且与任务是否成功无关。我们提出了OverclaimBench，这是一个由五个文件审查场景、基于对话记录的覆盖率测量以及预先登记的植入缺陷组成的评估套件。我们在八款专有前沿模型各自的生产级命令行界面中对其进行评估，并在单一固定测试框架下对四个开放权重模型进行评估，结果发现：1）在67.9%的运行中，智能体并未阅读所有被要求审查的文件；2）在未完整阅读文件的运行中，智能体……（摘要在此处被截断）

    arXiv:2609.20812v1 Announce Type: cross  Abstract: Frontier coding agents are increasingly trusted to work autonomously for long periods, yet an agent's final response is often the only account of that work a user sees. We quantify the propensity of frontier agents to \emph{overclaim} task completion, a misrepresentation that can mislead the user. An agent overclaims when its final response contradicts information in its context. This definition requires no inference about intent and is independent of task success. We introduce \emph{OverclaimBench}, an evaluation suite composed of five file-review scenarios, transcript-based coverage measurements, and registered planted defects. We evaluate eight proprietary frontier models in their own production command-line interfaces, and four open-weight models under a single fixed harness on OverclaimBench and find that 1) agents do not read all the files they were asked to review in 67.9\% of runs; 2) among runs where not all files are read, ag
    
[^5]: 分数中心化稳定离策略强化学习

    Score Centering Stabilizes Off-policy Reinforcement Learning

    [https://arxiv.org/abs/2609.20807](https://arxiv.org/abs/2609.20807)

    本文提出加性的“分数中心化”修正项，通过抵消训练与推理引擎间持续累积的漂移偏差，在无需高昂采样开销的情况下稳定了训练-推理失配下的大语言模型强化学习。

    

    大语言模型的强化学习（RL）对训练与推理引擎之间的微小差异极其敏感，这种差异通常被称为训练-推理失配（TIM）。然而，完全消除TIM并不现实，因为这将给轨迹采样效率带来重大代价。在本文中，我们证明了TIM下强化学习的不稳定性主要由漂移引起：训练与推理引擎之间的一种持续性偏差，会随每一次训练步不断累积。我们推导出一个加性的“分数中心化”修正项，通过抵消漂移来稳定TIM下的强化学习。在训练0.6B到30B参数规模的模型时，仅使用分数中心化即可在量化条件下达到甚至超越基于重要性采样的方法，且失配越严重，这一差距越大。由于该修正是加性的，分数中心化还可以与重要性采样结合使用——二者的组合效果优于纯粹的重要性（采样方法）……

    arXiv:2609.20807v1 Announce Type: new  Abstract: Reinforcement learning (RL) of large language models is notoriously sensitive to small differences between training and inference engines, often referred to as the training-inference mismatch (TIM). However, completely eliminating TIM is impractical, as it would come at a major cost to rollout efficiency. In this paper, we show that the instability of RL under TIM is primarily caused by drift: a persistent bias between training and inference engines that accumulates with every training step. We derive an additive "score centering" correction term that stabilizes RL under TIM by canceling drift. When training models from 0.6B to 30B parameters, score centering alone matches or outperforms methods based on importance sampling under quantization, with the gap growing as the mismatch becomes more severe. Because the correction is additive, score centering also composes with importance sampling -- their composition outperforms pure importance
    
[^6]: 编码智能体框架设计的实证研究

    An Empirical Study of Harness Design for Coding Agents

    [https://arxiv.org/abs/2609.20804](https://arxiv.org/abs/2609.20804)

    该论文通过固定执行循环并系统变化规划、动作空间和上下文管理三个组件的实证研究，发现上下文管理在上下文窗口预算紧张时价值显著提升，且其主要收益来自防止上下文溢出故障。

    

    编码框架决定了自主编码智能体如何将模型能力转化为长周期的软件工程性能，然而现有工作通常将框架作为整体系统进行评估，导致各个组件的有效性尚不清楚。为了实现组件级别的比较，我们使用一个轻量级编码框架来研究这一问题，该框架的执行循环保持固定，而三个组件则进行变化：规划、动作空间和上下文管理。我们在SWE-Bench Verified和Terminal-Bench 2.1上对四个模型进行了评估，共评估了176个匹配设置，涵盖五种上下文管理策略、四种上下文窗口预算，以及针对规划和动作空间的定向消融实验。我们发现：（1）随着上下文窗口预算的收紧，上下文管理变得愈发重要，其大部分收益来自于防止上下文溢出故障。（2）在基于LLM的摘要之前分阶段进行基于规则的省略（摘要在此处截断）。

    arXiv:2609.20804v1 Announce Type: new  Abstract: Coding harnesses shape how autonomous coding agents translate model capabilities into long-horizon software-engineering performance, yet existing work typically evaluates harnesses as monolithic systems, leaving the effectiveness of individual components unclear. To enable component-level comparisons, we study this question with a lightweight coding harness whose execution loop is fixed while three components are varied: planning, action space, and context management. Across four models evaluated on SWE-Bench Verified and Terminal-Bench 2.1, we evaluate 176 matched settings spanning five context-management strategies, four context-window budgets, and targeted ablations of planning and action space. We find that: (1) Context management becomes increasingly valuable as the context-window budget tightens, with most of its benefit coming from preventing context-overflow failures. (2) Staging rule-based elision before LLM-based summarization 
    
[^7]: PosteriorBench：从点估计到后验匹配——生成式逆问题求解器的评估

    PosteriorBench: From Point Estimates to Posterior Matching in Evaluating Generative Inverse Solvers

    [https://arxiv.org/abs/2609.20794](https://arxiv.org/abs/2609.20794)

    论文提出PosteriorBench基准，通过在四个物理逆问题上构建高保真参考后验分布，将生成式逆问题求解器的评估从单一重建的点估计精度提升为对真实后验分布匹配能力的评估。

    

    生成模型正被越来越多地用于求解科学逆问题，但现有的评估方法仍主要关注某种方法能否产生单一合理的重建结果。这对于病态问题是不够的，因为在病态问题中，多个解可能与相同的稀疏或含噪观测相一致。在这些情况下，一个方法可能在逐点精度上表现优异，但仍会因模式坍缩、过度自信的不确定性估计或对不相容解进行平均而无法捕捉真实的后验分布。我们提出了PosteriorBench，一个用于评估生成式逆问题求解器分布精度的基准。PosteriorBench评估了四个基于物理的逆问题：达西流反演、泊松源恢复、碳捕集与封存以及光传输材质推断。对于每个任务，我们使用计算量大但成熟可靠的方法（如拒绝采样……）构建高保真的参考后验分布。

    arXiv:2609.20794v1 Announce Type: new  Abstract: Generative models are increasingly used to solve scientific inverse problems, but existing evaluations still focus primarily on whether a method can produce a single plausible reconstruction. This is insufficient for ill-posed problems, where multiple solutions may be consistent with the same sparse or noisy observations. In these settings, a method can achieve strong pointwise accuracy while still failing to capture the true posterior through mode collapse, overconfident uncertainty, or averaging incompatible solutions. We introduce PosteriorBench, a benchmark for evaluating the distributional accuracy of generative inverse solvers. PosteriorBench evaluates four physics-based inverse problems: Darcy flow inversion, Poisson source recovery, carbon capture and storage, and light transport material inference. For each task, we construct high-fidelity reference posteriors using computationally heavy but established procedures such as reject
    
[^8]: GeoAAC：VLA策略中基于去噪轨迹几何的自适应动作分块方法

    GeoAAC: Geometry-Based Adaptive Action Chunking from Denoising Trajectories in VLA Policies

    [https://arxiv.org/abs/2609.20776](https://arxiv.org/abs/2609.20776)

    GeoAAC利用Flow Matching去噪轨迹的几何特征来评估动作预测的可靠性，从而在VLA策略中实现动作分块时间跨度的自适应调整。

    

    动作分块被广泛应用于视觉-语言-动作（VLA）策略中的动作生成与执行，然而现有方法通常采用固定的动作时间跨度。在策略执行过程中，不同的任务阶段可能需要不同程度的动作连续性、控制精度和闭环反馈，固定的时间跨度难以适应不断变化的控制需求。我们提出了GeoAAC，一种面向基于流的VLA策略、基于几何的自适应动作分块方法，可根据当前动作预测的可靠性动态调整动作时间跨度。我们证明了Flow Matching去噪轨迹的几何形状提供了刻画预测可靠性的过程级信息，且各动作前缀间的几何变化与预测不确定性保持正相关。GeoAAC利用这种前缀级几何信息构建时间跨度级别的几何轮廓，并自适应地确定动作时间跨度……

    arXiv:2609.20776v1 Announce Type: cross  Abstract: Action chunking is widely used for action generation and execution in Vision-Language-Action (VLA) policies, yet existing approaches commonly use a fixed action horizon. During a rollout, different task stages may require different levels of action continuity, control precision, and closed-loop feedback, making a fixed horizon unable to accommodate changing control requirements. We propose \textbf{GeoAAC}, a geometry-based adaptive action chunking method for flow-based VLA policies that adjusts the action horizon according to the reliability of the current action prediction. We show that the geometry of Flow Matching denoising trajectories provides process-level information for characterizing prediction reliability, with geometric variation across action prefixes remaining positively correlated with predictive uncertainty. GeoAAC uses this prefix-wise geometry to construct a horizon-wise geometric profile and adaptively determine the a
    
[^9]: 异构传输协议干扰下经校准的射频指纹识别

    Calibrated RF-Fingerprinting Under Interference With Heterogeneous Transmission Protocols

    [https://arxiv.org/abs/2609.20765](https://arxiv.org/abs/2609.20765)

    本文提出在多信号同信道干扰且传输协议异构的环境下，将射频指纹识别建模为多标签分类问题并使用一维CNN求解，同时通过模型校准推导置信度阈值，为假阴性数量上限提供保证，确保不遗漏真正的频谱政策违规行为。

    

    arXiv:2609.20765v1 公告类型：新论文 摘要：射频（RF）指纹识别是一种频谱监测技术，通过发射信号中留存的硬件损伤特征来识别特定的发射器。尽管该技术已被广泛研究，但现有研究几乎都只考虑同一时间仅有一个发射器发射信号的场景，这限制了其在现实世界中的适用性。在本工作中，我们通过考虑同信道干扰进一步推进了射频指纹识别的研究，即多个发射信号相互干扰、在时间和频率上重叠。具体而言，我们将该问题表述为多标签分类问题，并采用一维卷积神经网络（CNN）。此外，我们对模型进行了校准，从而推导出标签概率的置信度阈值，并对平均假阴性数量的上限提供保证，为不遗漏真正的频谱政策违规行为提供了一定程度的置信度。

    arXiv:2609.20765v1 Announce Type: new  Abstract: Radio Frequency(RF)-Fingerprinting is a spectrum monitoring technique that identifies specific transmitters based on hardware impairments imprinted within the emitted signal. Although widely researched, studies almost exclusively consider scenarios where only one transmitter is emitting at a time, limiting real world applicability. In this work, we further the study of RF-Fingerprinting by considering co-channel interference, with multiple emitted signals interfering with each other, overlapping in time and frequency. Specifically, we formulate this problem as a multi-label classification problem and employ a 1D convolutional neural network (CNN). Furthermore, the models are calibrated such that the confidence thresholds for the label probabilities are derived, with guarantees on the upper bound on the average number of False Negatives, providing a degree of confidence in not missing a true spectrum policy violation. The proposed method 
    
[^10]: Agile-WAM：一种用于接触密集型机器人控制的敏捷触觉世界动作模型

    Agile-WAM: An Agile Tactile World Action Model for Contact-Rich Robot Control

    [https://arxiv.org/abs/2609.20761](https://arxiv.org/abs/2609.20761)

    提出Agile-WAM，一种无需大规模预训练生成骨干的敏捷触觉世界动作模型，通过共享潜在空间中的直接视觉-触觉到动作流匹配过程，联合生成动作与未来视觉/触觉表征，实现高效灵活的接触密集型机器人控制。

    

    世界动作模型（WAM）通过联合预测未来世界状态和机器人动作，超越了传统的视觉运动策略，使策略能够学习支持有效控制的物理动力学。然而，近期的触觉WAM通常依赖大规模预训练生成骨干网络来捕捉接触密集的物理动力学，这限制了其推理效率和灵活部署。本文提出了Agile-WAM，一种用于接触密集型机器人控制的敏捷触觉世界动作模型。Agile-WAM将视觉和触觉观测编码到共享潜在空间中，作为直接的“视觉-触觉到动作”流匹配过程的源头，可以联合生成动作块以及未来视觉/触觉的潜在表示。一个关键观察是，视觉和触觉信号以本质上不同的时间尺度演化：相邻的视觉帧通常高度相似，而触觉信号的变化则迅速得多。

    arXiv:2609.20761v1 Announce Type: cross  Abstract: World Action Models (WAMs) advance beyond conventional visuomotor policies by jointly predicting future world states and robot actions, enabling the policy to learn physical dynamics that support effective control. However, recent tactile WAMs often rely on large-scale pretrained generative backbones to capture contact-rich physical dynamics, which limit their inference efficiency and flexible deployment. In this paper, we present \ABBR{}, an agile tactile World Action Model for contact-rich robot control. \ABBR{} encodes visual and tactile observations into a shared latent that serves as the source of a direct vision-tactile-to-action flow-matching process, which can jointly generate latent representations of action chunks and future visual/tactile latents. A key observation is that vision and tactile signals evolve at inherently different timescales: adjacent visual frames are often highly similar, whereas tactile signals can change 
    
[^11]: 用于分解式AI评估的预测驱动平滑与验证

    Prediction-Powered Smoothing and Validation for Disaggregated AI Evaluation

    [https://arxiv.org/abs/2609.20758](https://arxiv.org/abs/2609.20758)

    本文提出预测驱动平滑（PP-S）及其跨分类体系借力扩展（PP-TS），利用贝叶斯小区域估计方法为标签稀少领域的AI分解式评估提供精确的点估计和区间估计，并推导了新的近似无偏基于设计的交叉验证分数用于模型验证。

    

    评估一个AI系统需要进行分解式评估，因为其性能在不同领域（如基准测试任务类型或已部署智能体的对话类型）之间存在差异。穷举测试成本高昂，因此评估依赖于一个带有标签的单元样本。我们将评估集视为有限总体，寻求对每个领域均值的精确点估计和区间估计。直接估计方法（包括预测驱动推断PPI）仅使用该领域自身的标签，在标签稀少的情况下精度不足。小区域估计（small area estimation）解决了这一问题，我们在其基础上开发了一个集成估计与验证的完整工作流程。在估计方面，我们提出了预测驱动平滑，这是一个拟合到每个领域预测驱动估计值的贝叶斯模型，并进一步扩展为可在报告分类体系间借力的版本（PP-TS）。在验证方面，我们推导了一种新的近似无偏的基于设计的交叉验证分数，用于选择……

    arXiv:2609.20758v1 Announce Type: cross  Abstract: Evaluating an AI system requires disaggregated assessment, as performance varies across domains such as benchmark task types or conversation types in deployed agents. Exhaustive testing is expensive, so evaluation rests on a sample of labeled units. We treat the evaluation set as a finite population and seek accurate point and interval estimates of each domain mean. Direct estimators, including prediction-powered inference (PPI), use only a domain's own labels and are imprecise where labels are few. Small area estimation addresses this problem, and we build on it to develop an integrated workflow for estimation and validation. For estimation, we propose prediction-powered smoothing (PP-S), a Bayesian model fit to each domain's prediction-powered estimate, with an extension that borrows strength across a reporting taxonomy (PP-TS). For validation, we derive a new, approximately unbiased design-based cross-validation score for choosing a
    
[^12]: OPTED：使用无渲染教师的端到端驾驶在线策略微调方法

    OPTED: On-Policy Fine-Tuning for End-to-End Driving using a Render-Free Teacher

    [https://arxiv.org/abs/2609.20756](https://arxiv.org/abs/2609.20756)

    提出OPTED方法，通过在向量化输入上用强化学习训练特权教师来监督预训练学生模型的闭环后训练，从而无需昂贵的传感器仿真渲染即可缓解端到端驾驶策略在闭环部署中的累积误差问题。

    

    随着单纯扩大预训练数据的收益逐渐递减，后训练在自动驾驶等物理AI领域正变得日益重要。端到端驾驶策略通常通过行为克隆在人类演示数据上进行开环预训练。然而，闭环部署过程中的累积误差可能使车辆偏离训练数据分布，增加安全关键事故的风险。闭环后训练可以缓解这一风险，但对于基于传感器的策略而言需要成本高昂的仿真。我们提出了OPTED（端到端驾驶的在线策略微调）方法，它将强化学习与端到端策略的后训练解耦：先在向量化输入（高精地图和边界框）上使用强化学习训练一个特权教师，然后该教师在闭环后训练期间为预训练的学生模型提供监督。我们将OPTED应用于两个基于相机的模型TransFuser和V...

    arXiv:2609.20756v1 Announce Type: cross  Abstract: As scaling pre-training data alone yields diminishing returns, post-training is becoming increasingly important across physical AI domains such as autonomous driving. End-to-end driving policies are pre-trained in open loop with behavior cloning on human demonstrations. However, compounding errors during closed-loop deployment can take the vehicle outside the training data distribution, increasing the risk of safety-critical incidents. Closed-loop post-training can mitigate this risk but requires costly simulation for sensor-based policies. We propose OPTED (on-policy fine-tuning for end-to-end driving) which decouples reinforcement learning from the post-training of the end-to-end policy: a privileged teacher is trained using RL on vectorized inputs (HD-map and bounding boxes). This teacher then provides supervision to the pre-trained student during closed-loop post-training. We apply OPTED to two camera-based models, TransFuser and V
    
[^13]: dQwen3.5：混合注意力扩散语言模型

    dQwen3.5: Hybrid-Attention Diffusion Language Models

    [https://arxiv.org/abs/2609.20751](https://arxiv.org/abs/2609.20751)

    本论文将Qwen3.5的注意力-RNN混合架构（0.8B至9B规模）适配为dQwen3.5扩散语言模型系列，证明混合架构骨干仅需全注意力模型约一半的训练词元即可达到相同损失，同时在任意顺序解码上表现相当，并在并行解码下表现出色。

    

    将预训练的自回归（AR）模型进行适配是构建扩散语言模型（DLM）的一条高性价比路径。尽管几乎所有此类适配都以全注意力Transformer为起点，但自回归建模已转向注意力层与RNN层交替堆叠的混合架构。这为适配带来了障碍：与注意力机制不同，RNN在结构上是因果的，要将其双向化并非易事。尽管存在这种不匹配，我们仍然研究了此类骨干网络能否成为有效的扩散语言模型——我们在0.8B、2B、4B和9B规模上对Qwen3.5进行了适配，由此得到了dQwen3.5系列模型。我们发现混合骨干网络可以成为高效的适配起点：与全注意力对照组相比，混合模型仅需约一半的词元量即可达到相同的训练损失。在各个规模上，dQwen3.5在任意顺序解码行为上与全注意力扩散语言模型相似，并在并行解码下表现强劲。

    arXiv:2609.20751v1 Announce Type: new  Abstract: Adapting a pretrained autoregressive (AR) model is a cost-efficient route to a diffusion language model (DLM). While nearly all such adaptations start from a full-attention transformer, AR modeling has shifted toward hybrid architectures that interleave attention and RNN layers. This creates an obstacle for adaptation: unlike attention, RNNs are structurally causal and nontrivial to bidirectionalize. Despite this mismatch, we investigate whether such backbones can become effective DLMs by adapting Qwen3.5 at 0.8B, 2B, 4B, and 9B scales, yielding the dQwen3.5 family. We find that hybrid backbones can be efficient starting points for adaptation: against a full-attention control, the hybrid reaches a given training loss in about half the tokens. Across scales, dQwen3.5 resembles full-attention DLMs in any-order decoding behavior and performs strongly under parallel decoding.
    
[^14]: MILER：用于非结构化自动驾驶中模拟到现实强化学习的语义中层表示

    MILER: Semantic Mid-Level Representation for Sim-to-Real Reinforcement Learning in Unstructured Autonomous Driving

    [https://arxiv.org/abs/2609.20747](https://arxiv.org/abs/2609.20747)

    MILER提出了一种基于语义中层表示的端到端策略框架，通过BEVFusion将真实传感器数据转换为与模拟器一致的语义鸟瞰图表示，实现了非结构化自动驾驶环境中零样本的模拟到现实强化学习迁移。

    

    强化学习因其超越人类表现的潜力和自学习策略的能力而成为一种有前景的方法。然而，由于非结构化环境模拟到现实迁移的挑战，其在真实世界自动驾驶中的应用仍然稀缺，尤其是在非结构化环境中。在这项工作中，我们提出了MILER，一个具有零样本模拟到现实迁移能力的端到端策略框架。在离线训练期间，我们采用自定义的语义中层表示（MLR）模拟器，并使用强化学习训练策略网络，其控制输出直接应用于自行车模型。在真实车辆部署时，相机和激光雷达数据由BEVFusion处理，生成与MLR模拟器一致的语义鸟瞰图表示。策略网络生成的动作不会直接应用于真实车辆，而是我们……

    arXiv:2609.20747v1 Announce Type: cross  Abstract: Reinforcement learning constitutes a promising approach owing to its potential for superhuman performance and self-learned policies. However, its application to real-world autonomous driving remains scarce, particularly in unstructured environments, because of the challenges associated with sim-to-real transfer for unstructured environments. In this work, we present MILER, an end-to-end policy framework with zero-shot sim-to-real transfer. During offline training, we employ a custom semantic mid-level representation (MLR) simulator and train the policy network using reinforcement learning, with its control outputs applied directly to a bicycle model. During deployment on the real vehicle, camera and LiDAR data are processed by BEVFusion to generate a semantic bird's-eye-view representation consistent with that of the MLR simulator. The actions generated by the policy network are not applied directly to the real vehicle. Instead, we emp
    
[^15]: Video DeltaNet：一种面向直播视频生成的视频原生混合注意力机制

    Video DeltaNet: A Video-Native Hybrid Attention for Livestream Video Generation

    [https://arxiv.org/abs/2609.20744](https://arxiv.org/abs/2609.20744)

    提出Video DeltaNet（VDN），通过将局部Softmax注意力与引入视频增量注意力（VDA）的双向线性记忆相结合的混合架构，解决视频扩散模型中的注意力计算瓶颈，实现高质量的直播视频生成。

    

    视频扩散模型在去噪过程中需要反复处理长时空token序列，这使得注意力机制成为主要的计算瓶颈。线性注意力提供了一种有吸引力的替代方案，并已在近期的大型语言模型中得到广泛采用，但直接将其应用于视频模型往往无法保留高质量生成所需的细粒度交互。我们提出了Video DeltaNet（VDN），它将局部Softmax注意力与双向线性记忆相结合，用于长程视频上下文建模。其线性分支引入了视频增量注意力（Video Delta Attention，VDA），通过联合整合每帧的空间token，实现每帧一次的记忆更新。分离的输出投影和可学习的门控机制用于校准两个分支，同时采用分阶段的教师对齐训练策略，将新通路逐步引入预训练模型。我们将VDN实例化于MiniMax H3，将该混合机制应用于视频到视频的交互，同时保留Softmax注意力（原文在此处截断）。

    arXiv:2609.20744v1 Announce Type: new  Abstract: Video diffusion models repeatedly process long spatiotemporal token sequences during denoising, making attention a major computational bottleneck. Linear attention offers an appealing alternative and has been widely adopted in recent large language models, but directly applying it to video models often fails to preserve the fine-grained interactions required for high-quality generation. We present Video DeltaNet (VDN), which combines local Softmax attention with bidirectional linear memory for long-range video context. Its linear branch introduces Video Delta Attention (VDA), which updates memory once per frame by jointly incorporating its spatial tokens. Separate output projections and learnable gates calibrate the two branches, while a staged teacher-alignment recipe progressively introduces the new pathway into pretrained models. We instantiate VDN on MiniMax H3, applying the hybrid to video-to-video interactions while retaining Softm
    
[^16]: 不要屏蔽环境：观测监督改变智能体在强化学习下的探索方式

    Don't Mask the Environment: Observation Supervision Changes How Agents Explore Under RL

    [https://arxiv.org/abs/2609.20715](https://arxiv.org/abs/2609.20715)

    该论文提出ActObs方法，在监督微调中同时对轨迹中已有的环境观测标记进行监督，使策略学会建模动作后果，从而在不增加任何数据、参数或计算成本的情况下，显著提升后续GRPO强化学习中智能体的探索能力和pass@k性能。

    

    智能体的轨迹记录了智能体做了什么以及接下来发生了什么。然而，标准的监督微调（SFT）只对智能体生成的动作标记应用损失，仅将环境观测作为上下文而不作为预测目标。我们探究这一惯例是否能为后续强化学习提供最佳初始化。我们提出ActObs，它还对每条轨迹中已有的观测标记进行监督。尽管部署的智能体从不生成观测，但学习预测观测可以在不增加数据、参数、序列标记或前向传播的情况下，促使策略对动作后果进行建模。这些方法在SFT后表现相似，但在GRPO后出现分化。在Qwen3-4B上，基于ActObs的GRPO在Terminal-Bench 2.0上每个评估采样预算下都比仅监督动作的方法获得更高的pass@k。在Qwen3-8B上，它以一定的pass@1可靠性换取更高的pass@k（pass@16时提升3.4个百分点），并解决了更多任务。

    arXiv:2609.20715v1 Announce Type: cross  Abstract: Agent trajectories record what an agent does and what happens next. Yet standard supervised fine-tuning (SFT) applies loss only to agent-authored action tokens, using environment observations as context but not as prediction targets. We ask whether this convention provides the best initialization for subsequent reinforcement learning. We introduce ActObs, which also supervises the observation tokens already present in each trajectory. Although deployed agents never generate observations, learning to predict them encourages the policy to model action consequences without adding data, parameters, sequence tokens, or forward passes. The methods perform similarly after SFT but diverge after GRPO. On Qwen3-4B, GRPO from ActObs achieves higher pass@k at every evaluated sampling budget than its action-only counterpart on Terminal-Bench 2.0. On Qwen3-8B, it trades some pass@1 reliability for higher pass@k (+3.4 pp at pass@16) and solves more d
    
[^17]: 非对偶Lipschitz凸优化的稳定移动：高效性与近乎最优的预言机速率

    Stable Movement for Nondual Lipschitz Convex Optimization: Efficiency and Nearly Optimal Oracle Rates

    [https://arxiv.org/abs/2609.20701](https://arxiv.org/abs/2609.20701)

    该论文通过将Lipschitz凸优化归约为嵌套凸集追逐问题，为非对偶情形设计出高效算法并实现了近乎最优的一阶预言机复杂度，从而解决了COLT 2015开放问题的非光滑部分。

    

    我们研究了实现 $G$-Lipschitz 凸函数在半径为 $R$ 的 $\ell_p$ 球上关于 $\ell_q$ 范数优化的一阶预言机复杂度的高效算法，其中 $1\leq p,q\leq \infty$。对于 $p<q$ 的情况，我们在 $T$ 次预言机查询后获得误差 $\widetilde{O}_{p,q}(GR/T^{1/p-(1/q-1/2)_{+}})$，高效地实现了 (MBG+26) 的近乎最优速率，从而解决了 COLT 2015 开放问题 (Guz15b) 的非光滑部分。特别地，对于半径为 $R$ 的 $\ell_1$ 球上的欧几里得 Lipschitz 情形（$p=1,q=2$），速率为 $\widetilde{O}(GR/T)$。我们的解决方案是将凸 Lipschitz 优化归约为在演化束（bundle）的下水平集中追逐嵌套凸集的问题 (LNN95; BBE+20)：在每次查询时，我们要么找到一个函数值较低的点，要么在当前所追逐的束的下水平集中产生一个深切割。选择器稳定性与…（摘要截断）

    arXiv:2609.20701v1 Announce Type: cross  Abstract: We study efficient algorithms for realizing the first-order oracle complexity of optimization of $G$-Lipschitz convex functions with respect to the $\ell_{q}$-norm over an $\ell_{p}$-ball of radius $R$, where $1\leq p,q\leq \infty$. For $p<q$, we obtain error $\widetilde{O}_{p,q}(GR/T^{1/p-(1/q-1/2)_{+}})$ after $T$ oracle queries, efficiently realizing the nearly optimal rates of (MBG+26), thereby resolving the nonsmooth end of the COLT 2015 open problem (Guz15b). In particular, the rate is $\widetilde{O}(GR/T)$ for Euclidean Lipschitzness over an $\ell_1$-ball of radius $R$ ($p=1,q=2$). Our solution consists of reducing convex Lipschitz optimization to the chasing nested convex sets problem in sublevel sets of an evolving bundle (LNN95; BBE+20): at each query we either find a point with low function value or we produce a deep cut in the current sublevel of the bundle, that we chase. The dichotomy between stability of selectors and fo
    
[^18]: TetrisCNN：从实验量子模拟器数据中可解释地探测物质相

    TetrisCNN for interpretable detection of phases of matter from experimental quantum simulator data

    [https://arxiv.org/abs/2609.20693](https://arxiv.org/abs/2609.20693)

    本文提出TetrisCNN，一种具有俄罗斯方块状多形状滤波器并行分支的卷积神经网络，能直接从含噪声的实验量子模拟器数据中学习以自旋关联子表达的稀疏可解释潜在表征，从而实现对物质相的可解释探测。

    

    探测物质相通常依赖于识别正确的序参量——对于未知的相变，这项任务一直出了名地困难，传统上依靠物理直觉和有根据的猜测来引导。神经网络近来提供了一条替代途径，能够在没有任何先验物理知识的情况下定位已知模型中的相变。然而，这些方法仍然是黑箱，只能识别相而无法阐明其性质。此外，当面对现实的、含噪声的实验数据时，它们往往力不从心，而实验数据正是物理学中自动化方法的终极试金石。在此，我们通过引入 TetrisCNN 来弥合这两种视角：这是一种卷积神经网络架构，具有多个由不同形状滤波器组成的并行分支，让人联想到俄罗斯方块，它能够直接以自旋关联子的形式学习稀疏且可解释的潜在表征。应用于二维……

    arXiv:2609.20693v1 Announce Type: cross  Abstract: Detecting phases of matter in general relies on identifying the correct order parameter - a task that remains notoriously difficult for unknown transitions and traditionally is guided by physical intuition and educated guess. Neural networks have recently offered an alternative route by locating phase transitions in known models without any a priori physical knowledge. Yet these approaches remain black boxes and only identify phases without elucidating their properties. Moreover, they often struggle when confronted with realistic, noisy experimental data, which constitute the ultimate testbed for automated methods in physics. Here, we bridge these perspectives by introducing TetrisCNN, a convolutional architecture with parallel branches of differently shaped filters, reminiscent of Tetris blocks, that learns sparse, interpretable latent representations directly in terms of spin correlators. Applied to experimental snapshots of two-dime
    
[^19]: 非对偶设置下Lipschitz凸优化的一阶Oracle复杂度

    The First-Order Oracle Complexity of Lipschitz Convex Optimization in Nondual Settings

    [https://arxiv.org/abs/2609.20687](https://arxiv.org/abs/2609.20687)

    该论文通过一种以仿射损失最大值评估比较者的新在线学习博弈及其与顺序fat-shattering维数的联系，肯定地解决了COLT开放问题的非光滑版本，证明当p < q时ℓ_p球的几何结构可将ℓ_q-Lipschitz凸优化的收敛速率提升至Õ(1/T)，改进了经典的O(1/√T)速率。

    

    我们研究了ℓ_q范数下Lipschitz目标函数在ℓ_p球上的一阶黑盒凸优化问题，肯定地解决了COLT开放问题（Guz15b）的非光滑版本——即更小可行集的几何结构（p < q）能否改善凸优化的收敛速率，并在对数因子内匹配了已有的下界。我们的收敛速率包括ℓ_1球上凸欧几里得-Lipschitz优化的Õ(1/T)，在一般假设下改进了经典的O(1/√T)速率。其中的关键技术工具是一种新的在线学习博弈，在该博弈中，比较者使用到目前为止观察到的仿射损失的最大值来进行评估。我们利用一个组合在线学习量——顺序fat-shattering维数——对该博弈的值进行了上下界刻画，并针对ℓ_p/ℓ_q情形对该维数进行了表征。我们的结果一般适用于可行集……

    arXiv:2609.20687v1 Announce Type: cross  Abstract: We study first-order black-box convex optimization over an $\ell_p$-ball for objectives Lipschitz in the $\ell_q$-norm, solving in the affirmative the nonsmooth version of the COLT open question (Guz15b) on whether the geometry of a smaller feasible set ($p < q$) can improve convergence rates in convex optimization, and matching prior lower bounds up to logarithmic factors. Our rates include \(\widetilde O(1/T)\) for convex Euclidean-Lipschitz optimization over the $\ell_1$-ball, improving on the $O(1/\sqrt{T})$ classical rate under general assumptions. The key technical device is a new online learning game, where the comparator is evaluated using the maximum of affine losses observed so far. We bound the value of this game above and below in terms of a combinatorial online learning quantity: the sequential fat-shattering dimension, which we characterize for the $\ell_p / \ell_q$ case. Our results generally apply when the feasible set 
    
[^20]: RISC-V与机器学习：综述

    RISC-V and machine learning: a survey

    [https://arxiv.org/abs/2609.20677](https://arxiv.org/abs/2609.20677)

    本文系统综述了RISC-V指令集架构在机器学习领域的应用现状，提出了统一的实现分类法、性能与设计权衡的对比分析以及软件工具链成熟度的评估。

    

    开源处理器架构与机器学习的交汇正在推动对可定制、高效且易于获取的硬件的需求。本综述考察了RISC-V指令集架构（ISA）在机器学习应用中的现状，基于最新研究分析了当前的能力、挑战和未来方向。分析涵盖了学术和商业实现、软件框架以及实际应用。本文对RISC-V机器学习生态系统进行了全面评估，内容涵盖从指令集扩展和核心实现，到编译器优化和部署策略。主要贡献包括：RISC-V机器学习实现的统一分类法、性能与设计权衡的对比分析、软件工具链成熟度的评估，以及对指令集扩展和专用加速器新兴趋势的识别。研究结果表明，在能效、专用指令等方面已取得进展。

    arXiv:2609.20677v1 Announce Type: new  Abstract: The intersection of open-source processor architectures and machine learning is driving the demand for customizable, efficient, and accessible hardware. This survey examines the state of the RISC-V ISA in machine learning applications, analyzing current capabilities, challenges, and future directions based on recent research. The analysis covers academic and commercial implementations, software frameworks, and real-world applications. The RISC-V machine learning ecosystem is evaluated, from instruction set extensions and core implementations to compiler optimizations and deployment strategies. Key contributions include a unified taxonomy of RISC-V ML implementations, a comparative analysis of performance and design trade-offs, an evaluation of software toolchain maturity, and the identification of emerging trends in instruction set extensions and specialized accelerators. Findings reveal progress in energy efficiency, specialized instruc
    
[^21]: 流行病学因果图识别：挑战、可识别性与算法

    Epidemiological Causal Graph Identification: Challenges, Identifiability and Algorithms

    [https://arxiv.org/abs/2609.20676](https://arxiv.org/abs/2609.20676)

    该论文证明了有序分布节点与指数族分布节点之间的因果边方向在一般参数值下是可识别的，将有序-泊松模型的可识别性结果推广到更广泛的指数族分布，并提出了基于分数的穷举搜索和掩码连续优化两种因果发现算法。

    

    从观测数据中进行因果发现是统计学和机器学习的基础，然而在没有干预的情况下确定因果方向需要结构性假设。现有的可识别性研究主要集中于加性噪声模型下的连续变量，往往忽略了包含有序尺度、计数和连续测量的混合数据集。本文研究了有向无环图中节点遵循有序分布（通过有序logit模型）或正则单参数指数族分布的因果发现问题。我们证明了对于一般参数值，有序节点与指数族节点之间的边方向是分布可识别的。我们的发现将之前关于有序-泊松模型的结果推广到了更广泛的指数族分布。在计算方面，我们引入了基于分数的穷举搜索方法以及一种使用掩码的连续优化框架。

    arXiv:2609.20676v1 Announce Type: new  Abstract: Causal discovery from observational data is fundamental to statistics and machine learning, yet determining causal direction without interventions necessitates structural assumptions. Existing identifiability research primarily focuses on continuous variables under additive noise models, often neglecting mixed datasets containing ordinal scales, counts, and continuous measurements. This paper investigates causal discovery in Directed Acyclic Graphs (DAGs) where nodes follow either an ordinal distribution (via an ordered logit model) or a regular one-parameter exponential family distribution. We prove that the edge direction between an ordinal and an exponential family node is distributionally identifiable for generic parameter values. Our findings generalize previous Ordinal-Poisson results to the broader exponential family. Computationally, we introduce a score-based exhaustive search and a masked continuous optimization framework using
    
[^22]: 基于FL-Net的多中心医学数据挖掘——联邦学习的一站式解决方案

    Multi-center Medical Data Mining with FL-Net - A One-stop Shop for Federated Learning

    [https://arxiv.org/abs/2609.20650](https://arxiv.org/abs/2609.20650)

    FL-Net是一个一站式联邦学习临床研究框架，集成了数据协调、数据发现、披露控制和容器化联邦工作流执行，填补了现有14个框架均无法完全满足的五项要求，并将在欧盟项目中覆盖10家医院的80万余名患者。

    

    联邦学习使协作训练成为可能，且无需共享患者级别的数据，但大多数研究仍停留在模拟阶段。基于从文献中总结出的五项要求，我们分析了14个联邦学习框架，发现没有框架能完全满足这些要求。我们提出了FL-Net，这是一个新颖的联邦临床研究框架，可满足所有要求。它将模块化数据协调、数据发现、披露控制、安全构建的版本化FL-Net-Tools以及容器化的联邦工作流执行集成到一个持久网络中。它使协调后的数据和工作流能够在不同研究之间重复使用。FL-Net的端到端能力通过数据协调、跨MIMIC和US-130数据集的跨研究患者发现，以及支持多达50个并发客户端的可重现、经过审计的联邦工作流得到了评估。FL-Net正在dAIbetes和Microb-AI-ome欧盟项目内开发，将覆盖10家医院的超过80万名患者。

    arXiv:2609.20650v1 Announce Type: new  Abstract: Federated learning enables collaborative training without sharing patient-level data, but most studies remain simulations. Based on five requirements derived from the literature, we analyzed 14 FL frameworks and found that none fully satisfied these requirements. We present FL-Net, a novel federated clinical research framework to fulfill all requirements. It integrates modular data harmonization, data discovery, disclosure control, securely built versioned FL-Net-Tools and containerized federated workflow execution into a persistent network. It enables the re-use of harmonized data and workflows across studies. FL-Net's end-to-end capabilities were evaluated through harmonization, cross-study patient discovery across MIMIC and US-130, and reproducible, audited federated workflows with up to 50 concurrent clients. FL-Net is being developed within the dAIbetes and Microb-AI-ome EU projects and will cover over 800,000 patients across 10 hos
    
[^23]: 超越PINNs：面向神经与混合偏微分方程求解器的统一高斯-牛顿与Petrov-Galerkin框架

    Beyond PINNs: A Unified Gauss--Newton and Petrov--Galerkin Framework for Neural and Hybrid PDE Solvers

    [https://arxiv.org/abs/2609.20641](https://arxiv.org/abs/2609.20641)

    本文提出了一个统一的泛函高斯-牛顿离散化框架，通过对偶配对将线性测量表示为测试函数，证明高斯-牛顿系统恰为线性化问题的Petrov-Galerkin离散化，从而将PINN的逐点配点训练与有限元变分方法统一起来，并使测试函数的选择成为显式的算法设计手段。

    

    物理信息神经网络（PINNs）与有限元方法为偏微分方程的数值逼近提供了两种不同的范式：前者通常通过最小化逐点强形式残差进行训练，而后者则自然地建立在弱变分形式以及离散化后得到的有限维系统之上。在本工作中，我们引入了一个基于用有限线性测量族对泛函高斯-牛顿问题进行离散化的统一框架。我们证明，通过适当的对偶配对，线性测量可以由测试函数来表示。由此得到的高斯-牛顿系统恰好是线性化泛函问题的Petrov-Galerkin离散化。这一视角将逐点配点法和自然梯度构造作为特例加以恢复，同时使测试函数的选择成为一种显式的算法设计选择。

    arXiv:2609.20641v1 Announce Type: cross  Abstract: Physics-informed neural networks and finite element methods provide two different paradigms for the numerical approximation of partial differential equations: the former are commonly trained by minimizing pointwise strong residuals, whereas the latter are naturally built from weak variational formulations and the finite-dimensional systems obtained after discretization. In this work, we introduce a common framework based on the discretization of functional Gauss--Newton problems by finite families of linear measurements. We show that, through an appropriate duality pairing, the linear measurements can be represented by test functions. The resulting Gauss--Newton system is then precisely a Petrov--Galerkin discretization of the linearized functional problem. This perspective recovers pointwise collocation and natural-gradient constructions as particular cases, while making the choice of test functions an explicit algorithmic design choi
    
[^24]: COIN-GP：基于高斯过程回归的部分测量网络化分布式系统中的协作在线学习

    COIN-GP: Cooperative Online Learning in Networked Distributed Systems with Partial Measurements via Gaussian Process Regression

    [https://arxiv.org/abs/2609.20598](https://arxiv.org/abs/2609.20598)

    提出了一种结合在线分布式高斯过程回归的基于观测器的协作学习框架，可在仅有部分测量的条件下联合估计分布式传感网络中的系统状态与未知动力学，并提供了数据采集条件和误差上界。

    

    本文研究了在分布式传感器网络中联合估计系统状态和部分未知动力学的问题，特别是在仅有部分状态观测可用的场景中。为解决这一问题，我们提出了一种基于观测器的动态协作学习框架，并结合在线分布式高斯过程（GP）回归，即使在测量不完整和GP模型存在缺陷的情况下也能实现准确估计。此外，我们引入了一种新颖的数据收集策略，并提供了确保数据采集可行的理论条件。同时，我们利用GP的确定性误差界，推导出了涵盖状态估计和模型估计的误差上界。仿真实验表明，与现有的基于分布式GP的方法相比，我们的方法具有优越性。

    arXiv:2609.20598v1 Announce Type: new  Abstract: In this paper, we tackle the problem of jointly estimating the system states and partially unknown dynamics within distributed sensor-equipped networks, particularly in scenarios where only partial state observations are available. To address this issue, we propose an observer-based dynamic cooperative learning framework incorporating online distributed Gaussian Process (GP) regression, which enables accurate estimation despite incomplete in measurements and deficient GP models. In addition, a novel data collection strategy is introduced, with theoretical conditions ensuring feasible data acquisition. Moreover, we also derive an error upper bound encompassing state estimation and model estimation, leveraging the deterministic error bounds of GPs. Empirical simulations demonstrate the superiority of our approach compared to existing distributed GP-based methods.
    
[^25]: 面向稳定短期温度预测的递归量子长短期记忆网络

    Recursive Quantum Long Short-Term Memory for Stable Short-Horizon Temperature Forecasting

    [https://arxiv.org/abs/2609.20594](https://arxiv.org/abs/2609.20594)

    本文提出一种递归量子长短期记忆（QLSTM）架构，通过递归量子特征变换提升混合量子-经典时间序列模型的稳定性与泛化能力，在温度预测任务中比标准QLSTM收敛更快、误差更低。

    

    量子长短期记忆（QLSTM）模型通过变分量子电路扩展了循环序列学习，但其优化行为可能因随机初始化和时间上下文的不同而产生显著差异。本文评估了一种递归QLSTM架构与标准QLSTM在每日最低和最高温度一步预测任务中的表现。使用多伦多的每日天气观测数据和完全相同的训练设置，我们在8天、16天和32天的输入窗口上，跨20个随机种子比较了两种模型的收敛性、预测精度和泛化能力。递归模型始终能更早达到接近最优的测试损失，降低了平均绝对误差和均方根误差，并表现出更小的泛化差距。这些结果表明，递归量子特征变换可以提升紧凑型混合量子-经典时间序列模型的稳定性和样本外性能。

    arXiv:2609.20594v1 Announce Type: new  Abstract: Quantum long short-term memory (QLSTM) models extend recurrent sequence learning with variational quantum circuits, but their optimization behavior can vary substantially across random initializations and temporal contexts. This paper evaluates a recursive QLSTM architecture against a standard QLSTM for one-step-ahead prediction of daily minimum and maximum temperature. Using daily weather observations from Toronto and identical training settings, we compare convergence, predictive accuracy, and generalization across input windows of 8, 16, and 32 days over 20 random seeds. The recursive model consistently reaches a near-optimal test loss earlier, reduces mean absolute error and root mean squared error, and exhibits a smaller generalization gap. These results indicate that recursive quantum feature transformations can improve stability and out-of-sample performance for compact hybrid quantum--classical temporal models.
    
[^26]: CrystalMO-TuRBO：用于高精度晶体结构联合精修的多目标信赖域贝叶斯优化

    CrystalMO-TuRBO: Multi-Objective Trust-Region Bayesian Optimization for High-precision Joint Crystal Structure Refinement

    [https://arxiv.org/abs/2609.20592](https://arxiv.org/abs/2609.20592)

    提出了CrystalMO-TuRBO，一种多目标信赖域贝叶斯优化架构，通过将X射线和中子数据的偏差作为独立目标建模，避免了手动权重调节，实现了高精度的晶体结构联合精修。

    

    晶体结构精修是材料表征中的一个基本逆问题，其目标是通过优化结构参数来重现实验衍射数据。传统方法（如最小二乘法和基于似然的优化）依赖于局部搜索，在面对非凸、含噪且高度相关的参数空间时常常表现不佳，尤其是在融合多种衍射模态时。X射线和中子数据的联合精修尤其具有挑战性，因为两者具有互补但相互竞争的灵敏度，通常需要通过标量化目标将其组合，这不仅需要手动设置权重，还容易导致次优解。我们提出了CrystalMO-TuRBO，一种用于晶体结构联合精修的多目标信赖域贝叶斯优化架构。该方法将X射线和中子数据的偏差建模为独立的目标，并将问题转化为归一化的最大化设置……

    arXiv:2609.20592v1 Announce Type: new  Abstract: Crystal structure refinement is a fundamental inverse problem in materials characterization, where structural parameters are optimized to reproduce experimental diffraction data. Conventional approaches, such as least-squares and likelihood-based optimization, rely on local search and often struggle with non-convex, noisy, and highly correlated parameter landscapes, particularly when integrating multiple diffraction modalities. Joint refinement of X-ray and neutron data is especially challenging due to their complementary but competing sensitivities, which are typically combined through scalarized objectives requiring manual weighting and leading to suboptimal solutions. We propose CrystalMO-TuRBO, a multi-objective trust region Bayesian optimization architecture for joint crystal structure refinement. The method models X-ray and neutron discrepancies as separate objectives and transforms the problem into a normalized maximization settin
    
[^27]: NS3Learn：将5G NR Mode-2接收真实性从ns-3迁移到Veins/SUMO技术栈以用于联网车辆安全评估

    NS3Learn: Transferring 5G NR Mode-2 Reception Realism from ns-3 to the Veins/SUMO Stack for Connected-Vehicle Safety Assessment

    [https://arxiv.org/abs/2609.20578](https://arxiv.org/abs/2609.20578)

    本研究提出NS3Learn，一个从ns-3 5G-LENA仿真数据（1050万接收样本）中学习得到的闭式模型，能在Veins/SUMO车联网仿真中无需重实现完整协议即可准确刻画5G NR sidelink Mode-2的资源竞争损失，将每时刻消息投递率预测的平均绝对偏差降至0.06，显著优于现有模型。

    

    联网车辆安全评估依赖于交通与网络的耦合仿真，但标准信道模型忽略了5G NR sidelink Mode-2中的无线资源竞争，导致在密集交通条件下给出不切实际的高消息投递率。本研究在不需完整重新实现协议的情况下引入了资源竞争损失。我们对来自ns-3 5G-LENA轨迹（基于3GPP场景校准并由SUMO轨迹驱动）的1050万个接收结果进行标注，以拟合NS3Learn——一个能够捕捉半双工损耗、调度冲突、接收端捕获效应和解码的闭式模型。评估涵盖两个信号灯控制的城市路网、六个渗透率水平（1-100%）以及每种条件下的五个随机种子。与ns-3 5G-LENA相比，NS3Learn在每时刻投递率上的平均绝对偏差仅为0.06，优于其他替代模型（偏差分别为0.44和0.55）。拟合参数迁移至另一个不同的交叉口时仅需20%的额外（校准，原文截断）……

    arXiv:2609.20578v1 Announce Type: cross  Abstract: Connected-vehicle safety evaluations rely on coupled traffic and network simulations, but standard channel models ignore radio resource competition in 5G NR sidelink Mode-2, reporting unrealistically high message delivery in dense traffic. This study introduces resource-competition losses without requiring full protocol reimplementation. We labeled 10.5 million reception outcomes from ns-3 5G-LENA traces (calibrated on 3GPP scenarios and driven by SUMO trajectories) to fit NS3Learn - a closed-form model capturing half-duplex loss, scheduling collisions, receiver capture, and decoding. Evaluation spanned two signalized urban networks, six penetration levels (1-100%), and five random seeds per condition. NS3Learn achieved a mean absolute deviation of 0.06 in per-instant delivery compared to ns-3 5G-LENA, outperforming alternative models (0.44 and 0.55 deviation). Fitted parameters transferred to a distinct intersection with only 20% addi
    
[^28]: 球面线性模型中低于波动尺度的TAP精度与普适后验几何

    TAP Accuracy Below the Fluctuation Scale and Universal Posterior Geometry in Spherical Linear Models

    [https://arxiv.org/abs/2609.20577](https://arxiv.org/abs/2609.20577)

    本文在Marchenko–Pastur谱正则性条件下，证明了球面线性模型中TAP自由能逼近精度可达 $O_P(p^{-1})$（低于自然波动尺度 $O_P(p^{-1/2})$），并在所有全局TAP最大化子上一致刻画了后验几何。

    

    我们研究了在环境维度与样本量成比例增长时的贝叶斯最优球面线性模型，并对设计矩阵施加了定量的Marchenko–Pastur谱正则性条件。该条件可被元素标准化且四阶矩有限的归一化独立同分布设计所满足，但不要求元素间相互独立，也无需对奇异向量施加任何条件。在此条件下，我们证明了定量的全温度TAP逼近，并刻画了后验几何。对于自然的有限纵横比TAP泛函，归一化球面自由能与TAP最优值之差为 $O_P(p^{-1})$。二者与其显式确定性等价形式之差均在 $O_P(p^{-1/2})$ 以内，且该波动尺度是紧致（sharp）的。在所有全局TAP最大化子上一致地，到球面后验均值的归一化平方欧氏距离为 $O_P(p^{-1})$。此外，我们还证明了数据依赖邻域（此处摘要原文截断）。

    arXiv:2609.20577v1 Announce Type: cross  Abstract: We study the Bayes-optimal spherical linear model as the ambient dimension and sample size grow proportionally, under a quantitative Marchenko--Pastur spectral-regularity condition on the design. This condition is satisfied by normalized i.i.d. designs with standardized entries of finite fourth moment, but does not require entrywise independence or impose conditions on the singular vectors. Under this condition, we prove a quantitative all-temperature TAP approximation and characterize the posterior geometry. For the natural finite-aspect-ratio TAP functional, the normalized spherical free energy and the TAP optimum differ by $O_P(p^{-1})$. Each is within $O_P(p^{-1/2})$ of its explicit deterministic equivalent, and this fluctuation scale is sharp. Uniformly over all global TAP maximizers, the normalized squared Euclidean distance to the spherical posterior mean is $O_P(p^{-1})$. We also prove that the posterior mass outside a data-dep
    
[^29]: 基于采样式模型预测控制加速视觉策略学习

    Accelerating Visual Policy Learning with Sampling-Based Model Predictive Control

    [https://arxiv.org/abs/2609.20575](https://arxiv.org/abs/2609.20575)

    提出采样引导策略搜索方法SGPS，将基于采样的模型预测控制与一阶策略优化相结合，并采用将渲染排除在计算图之外的解耦FoPG公式，避免局部优化陷入非预期接触模式，实现单GPU上直接从深度观测高效训练视觉策略。

    

    学习用于运动和操作的视觉策略需要与环境进行协调接触，并可能产生大量的计算和GPU内存开销。一阶策略梯度方法（FoPG）通过可微分仿真降低了训练成本，但其局部优化可能收敛到非预期的接触模式。为解决这一不足，我们提出了采样引导策略搜索（SGPS），它将通过基于采样的模型预测控制进行的反复动作目标细化与一阶策略优化相结合。行为克隆从采样动作中初始化策略；随后训练在受扰动的初始状态和随机化动力学条件下，交替进行基于采样的细化与短时域FoPG更新。对于视觉策略训练，我们采用一种解耦的FoPG公式，将渲染排除在计算图之外，从而无需状态-策略教师即可直接从深度观测中学习。在单块GPU上，……

    arXiv:2609.20575v1 Announce Type: cross  Abstract: Learning visual policies for locomotion and manipulation requires coordinating contact with the environment and can incur substantial computation and GPU memory costs. First-order policy gradients (FoPG) reduce training cost through differentiable simulation, but local optimization can converge to unintended contact patterns. To address this shortfall, we propose Sampling-Guided Policy Search (SGPS), which couples recurring action-target refinement by sampling-based model-predictive control with first-order policy optimization. Behavior cloning initializes the policy from sampled actions; training then alternates sampling-based refinement with short-horizon FoPG updates under perturbed initial states and randomized dynamics. For visual policy training, we use a decoupled FoPG formulation that excludes rendering from the computation graph, enabling direct learning from depth observations without a state-policy teacher. On a single GPU, 
    
[^30]: 缓解重复博弈中的报复性算法合谋

    Mitigating Retaliatory Algorithmic Collusion in Repeated Games

    [https://arxiv.org/abs/2609.20548](https://arxiv.org/abs/2609.20548)

    该论文提出了CURB奖励塑形框架，通过将Q-learning合谋行为与简单惩罚码理论形式化关联，利用合作与背叛历史下策略间的全变分距离检测并惩罚算法合谋，为一般性重复博弈提供了通用的合谋缓解方法。

    

    在重复交互中被训练以最大化自身奖励的强化学习智能体可以收敛到类似显式合谋的超竞争结果，而无需通信或共享设计。现有的缓解方法大多局限于特定的经济场景，如双边平台和拍卖，因此如何为一般性的重复博弈设计干预措施仍是一个开放问题。我们通过形式化先前工作中关于Q-learning合谋的经验观察与经典的简单惩罚码理论之间的联系来解决这一空白。我们证明，任何非平凡的简单惩罚码都会在智能体的策略中诱导出可量化的条件依赖性，这种依赖性可以通过智能体在合作历史与背叛历史下动作分布之间的全变分距离检测出来。基于这一联系，我们提出了CURB（通过奖励塑形和信念注入实现合谋解缠），这是一个奖励塑形框架，通过惩罚这种全变分……

    arXiv:2609.20548v1 Announce Type: cross  Abstract: Reinforcement learning agents trained to maximize their own reward in repeated interactions can converge to supra-competitive outcomes resembling explicit collusion, without communication or shared design. Existing mitigation approaches are largely tied to specific economic settings, like two-sided platforms and auctions, leaving open how to design interventions for general repeated games. We address this gap by formalizing the connection between empirical observations from prior work on Q-learning collusion and classical theory of Simple Penal Codes (SPCs). We show any non-trivial SPC induces a quantifiable conditional dependence in agents' policies, detectable via the total variation distance between an agent's action distributions across cooperation and defection histories. Building on this connection, we propose CURB (Collusion Unwinding via Reward shaping and Belief injection), a reward-shaping framework that penalizes this Total 
    
[^31]: 扩散语言模型中的并行性、临界窗口与分离性

    Parallelism, critical windows, and separations among diffusion language models

    [https://arxiv.org/abs/2609.20539](https://arxiv.org/abs/2609.20539)

    本文首次对掩码扩散、均匀扩散与高斯扩散三类主流扩散语言模型的并行生成能力进行了细粒度理论比较，证明均匀扩散和高斯扩散同样能以前向传播次数与分布对偶总相关（可远小于上下文长度）成比例的方式完成采样，而此前仅掩码扩散具备这一性质。

    

    扩散大语言模型的一个广受欢迎的卖点在于其并行性能力：即能够以远高于自回归模型的效率生成文本序列，后者每个 token 都需要一次前向传播。然而，在众多相互竞争的 dLLM 范式之中——从掩码扩散到均匀扩散再到高斯扩散——对于这些不同方案在并行性方面如何比较，原理性的理解仍然有限。在本工作中，我们对这三种主流方法的并行性能力开展了细粒度的比较研究，并证明了以下结果：均匀扩散和高斯扩散可以在前向传播次数随底层分布的对偶总相关缩放的情况下完成采样，对偶总相关是一种内在复杂度的度量，其数值可以远小于上下文长度；而在此之前，只有掩码扩散被证明能够实现这一点。此外，对于某一族随机经验测度，我们表明（摘要截断）……

    arXiv:2609.20539v1 Announce Type: new  Abstract: A popular selling point of diffusion large language models (dLLMs) is their capacity for parallelism: the ability to generate sequences of text far more efficiently than autoregressive models, which require one forward pass per token. Yet among the many competing paradigms for dLLMs, from masked to uniform to Gaussian diffusion, principled understanding of how these different proposals compare in parallelism remains limited. In this work, we initiate a fine-grained comparison of the capacity for parallelism among these three leading approaches and prove the following:   - Uniform and Gaussian diffusion can sample in a number of forward passes which scales with the dual total correlation of the underlying distribution, a measure of intrinsic complexity which can be much smaller than the context length. Previously, it was only known how to achieve this using masked diffusion.   - For a certain family of random empirical measures, we show t
    
[^32]: 面向数据高效语言建模的关系注意力机制

    Relational Attention for Data-Efficient Language Modeling

    [https://arxiv.org/abs/2609.20530](https://arxiv.org/abs/2609.20530)

    该论文提出在双注意力Transformer架构中将关系注意力与自注意力相结合，并借助BabyLM 2026挑战赛的数据受限环境，验证关系注意力所带来的数据效率能否成功迁移到语言建模任务中。

    

    我们提出了Relational BabyLM，这是提交给BabyLM 2026挑战赛的一个系统，它在单个仅解码器的Transformer中结合了两种基于认知科学启发的归纳偏置。在架构上，我们用双注意力Transformer（DAT）替代标准的自注意力机制，将对象级（“感官”）词汇特征的路由与结构/关系信息分离开来。从自注意力中解耦出来的关系注意力（RA）在纯关系任务上能极大地提高数据效率和训练样本外的泛化能力，但语言建模要求对象级信息和关系信息既能解耦又能整合，而基于RA的语言模型在很大程度上仍未被探索。BabyLM挑战赛的数据受限训练和全面评估为检验这种数据效率能否迁移到语言建模中提供了理想的测试平台。

    arXiv:2609.20530v1 Announce Type: new  Abstract: We present Relational BabyLM, a system submission to the BabyLM 2026 challenge that combines two cognitively motivated inductive biases in a single decoder-only Transformer. Architecturally, we replace standard self-attention with a Dual Attention Transformer (DAT), which separates the routing of object-level ("sensory") lexical features from structural/relational information (Altabaa and Lafferty, 2025; Altabaa et al., 2024; Webb et al., 2024; Kerg et al., 2022; Webb et al., 2021). Relational attention (RA) disentangled from self-attention greatly increases data efficiency and out-of-training-sample generalization on purely relational tasks, but language modeling requires object-level and relational information to be integrated as well as disentangled, and RA-based LMs have remained largely unexplored. BabyLM's data-constrained training and comprehensive evaluation is an ideal testing ground for whether that data efficiency transfers. A
    
[^33]: 基于深度学习的远程态制备噪声鲁棒量子态表征

    Noise-Robust Quantum State Characterization for Remote State Preparation with Deep Learning

    [https://arxiv.org/abs/2609.20523](https://arxiv.org/abs/2609.20523)

    该论文提出了一种基于Transformer的量子态表征器模型，能够在复杂散射与动态噪声环境下从含噪测量中以超过99.999%的保真度重建远程态制备的光子偏振态，并通过注意力机制揭示测量观测量间相关性的物理解释。

    

    量子通信是安全信息处理和可扩展量子网络的基础。特别是，远程态制备能够实现高效的量子态传输，但在复杂噪声下准确估计目标态仍然具有挑战性。本文提出了一种基于Transformer的量子态表征器（TQSC）模型，用于含噪的远程态制备实验。该模型能够在复杂散射环境中从噪声测量数据中重建实验制备的纯态和混合光子偏振态，同时其注意力模式为所测观测量之间的相关性提供了具有物理依据的见解。该方法在复杂散射和动态高斯噪声下实现了超过99.999%的平均估计器-目标保真度，其鲁棒性和泛化能力进一步通过Qiskit模拟的布洛赫球态得到验证。此外，在一个使用保留态的实际MNIST图像传输任务中，解码器……（摘要原文在此处截断）

    arXiv:2609.20523v1 Announce Type: cross  Abstract: Quantum communication underpins secure information processing and scalable quantum networks. In particular, remote state preparation (RSP) enables efficient quantum state transfer, but accurately estimating target states under complex noise remains challenging. Here, we propose a Transformer-based Quantum State Characterizer (TQSC) model for noisy RSP experiments. Our model reconstructs experimentally prepared pure and mixed photonic polarization states from noisy measurements in complex scattering environments, while its attention patterns provide physically grounded insights into correlations among the measured observables. The method achieves a mean estimator-target fidelity exceeding 99.999% under complex scattering and dynamic Gaussian noise, while its robustness and generalization are further examined using Qiskit-simulated Bloch-ball states.Furthermore, in a practical MNIST image transmission task with held-out states, the decod
    
[^34]: 当EOS标记不一致时：理解在线策略蒸馏中的长度膨胀问题

    When EOS Tokens Disagree: Understanding Length Inflation in On-Policy Distillation

    [https://arxiv.org/abs/2609.20511](https://arxiv.org/abs/2609.20511)

    该论文揭示了学生模型与教师模型之间EOS终止标记不匹配是在线策略蒸馏中生成长度膨胀的关键原因，并提出将功能等价的EOS标记视为共享的语义停止动作可显著缓解该问题。

    

    我们研究了在线策略蒸馏（OPD）中的长度膨胀问题，即学生模型的响应可能变得过长，甚至会耗尽生成预算。我们识别出基础学生模型与后训练教师模型之间的终止标记不匹配是这种行为的一个重要来源。在Qwen3、Llama和Gemma三个模型系列中，即使两个模型声明的停止集合完全相同，它们也可能将停止概率放置在不同的EOS标记上。这种不匹配可能会抑制学生模型偏好的终止动作，而无法可靠地传递教师模型偏好的替代终止动作。我们证明仅对齐解码停止集合是不够的，而将功能等价的EOS标记视为共享的语义停止动作，可以在所有三个模型系列中显著缓解由不匹配引起的长度膨胀。为了进一步理解终止行为在训练过程中如何演化，我们研究了不同K2-Horizon训练阶段下的OPD（原文在此处截断）。

    arXiv:2609.20511v1 Announce Type: new  Abstract: We study length inflation in on-policy distillation (OPD), where student responses can become excessively long and even exhaust the generation budget. We identify \emph{termination-token mismatch} between base students and post-trained teachers as an important source of this behavior. Across Qwen3, Llama, and Gemma, the two models can place their stopping probability on different EOS tokens, even when their declared stopping sets are identical. This mismatch can suppress the student's preferred termination action without reliably transferring the teacher-preferred alternative. We show that aligning the decoding stopping set alone is insufficient, while treating functionally equivalent EOS tokens as a shared semantic stopping action substantially mitigates mismatch-induced length inflation across all three model families. To further understand how termination behavior evolves over training, we study OPD across different K2-Horizon trainin
    
[^35]: 用于机器学习原子间势的截断自动稀疏微分

    Truncated automatic sparse differentiation for machine learning interatomic potentials

    [https://arxiv.org/abs/2609.20510](https://arxiv.org/abs/2609.20510)

    本文提出截断自动稀疏微分（ASD）方法，利用机器学习原子间势高阶导数的稀疏性与距离衰减特性，使大型系统的完整Hessian矩阵计算在计算上变得可行。

    

    机器学习原子间势（MLIPs）学习从原子位置到势能的映射。力作为该能量的负梯度，驱动分子动力学过程，并可通过自动微分轻松获得。更高阶的导数，尤其是Hessian矩阵，描述了集体运动，能够直接预测实验可观测量，但通常被认为对于大型系统在计算上是不可行的。我们提出了一个解决方案：在物理系统中，相互作用随距离衰减，而大多数MLIP通过有限感受野内的消息传递建立在这种局域性之上。这意味着高阶导数既具有稀疏性，又随距离衰减。这种结构可以利用自动稀疏微分（ASD）来加以利用。我们解释了如何计算MLIP导数的稀疏模式，并证明对于多个基础MLIP模型，ASD能够计算大型多孔材料的完整Hessian矩阵。

    arXiv:2609.20510v1 Announce Type: cross  Abstract: Machine learning interatomic potentials (MLIPs) learn the mapping from atomic positions to potential energy. The forces, the negative gradient of this energy, drive molecular dynamics and are readily obtained using automatic differentiation. Higher-order derivatives, most notably the Hessian, describe collective motion and allow the direct prediction of experimental observables, but are considered computationally inaccessible for large systems. We suggest a solution: in physical systems, interactions decay with distance, and most MLIPs build on this locality through message passing up to a finite receptive field. This implies both sparsity of higher-order derivatives and their decay with distance. This structure can be exploited using automatic sparse differentiation (ASD). We explain how to compute the sparsity pattern for MLIP derivatives and demonstrate that, for multiple foundation MLIPs, ASD computes full Hessians of large porous 
    
[^36]: 水中微塑料的射频检测与分类

    Radio Frequency Detection and Classification of Microplastics in Water

    [https://arxiv.org/abs/2609.20507](https://arxiv.org/abs/2609.20507)

    该论文提出了一种机器学习辅助的射频介电光谱检测平台，实现了对水中多种10微米级微塑料颗粒的无标记快速检测与材料分类。

    

    微米和纳米塑料颗粒（MPs/NPs）是普遍存在的环境污染物，其数量的日益增加及其潜在的健康影响使得人们迫切需要快速、无标记的检测方法。当颗粒尺寸减小到低微米范围时，传统的光学和光谱技术由于通量有限和/或样品制备复杂而变得越来越具有挑战性。在这项工作中，我们提出了一种机器学习（ML）辅助的射频（RF）介电光谱细胞测定（DiSC）平台，用于微塑料的无标记检测与分类。八种标称直径为10微米的微塑料颗粒悬浮于去离子水中，在0.2-9 GHz范围内的四个频率下进行了表征。以载液介质为参考，将测得的射频散射参数（S参数）的变化用于训练监督式机器学习模型以实现材料分类，包括……

    arXiv:2609.20507v1 Announce Type: new  Abstract: Micro- and nano-plastic particles (MPs/NPs) are ubiquitous environmental contaminants whose increasing abundance and potential health impacts have created an urgent need for rapid, label-free detection methods. As particle size decreases to the low-micrometer range, conventional optical and spectroscopic techniques become increasingly challenging because of limited throughput and/or complex sample preparation. In this work, we present a machine learning (ML)-assisted radio-frequency (RF) dielectric spectroscopic cytometry (DiSC) platform for the label-free detection and classification of MPs. Eight types of $ 10 $ {\mu}m nominal-diameter MP particles suspended in deionized (DI) water were characterized at four frequencies spanning $ 0.2\text{-}9\text{ GHz} $. The measured alterations in RF scattering parameters (S-parameters), referenced to the carrier medium, were used to train supervised ML models for material classification, including
    
[^37]: 基于多源数据的分布鲁棒联邦学习

    Distributionally Robust Federated Learning with Multi-Source Data

    [https://arxiv.org/abs/2609.20501](https://arxiv.org/abs/2609.20501)

    该论文提出了一种分布鲁棒联邦学习框架，通过将全局模糊集构建为局部模糊集的可容许混合的并集，同时应对跨客户端混合不确定性与客户端内部的分布模糊性，并据此建立了高概率样本外性能保证及相应的联邦算法。

    

    联邦学习利用私有客户端数据训练一个共享模型。在实践中，数据生成的分布可能各不相同，且客户端之间的真实混合比例往往是未知的，这使得底层群体分布难以确定。现有方法通过针对最坏情况混合分布进行优化来处理跨客户端混合不确定性，但假设对各个客户端的分布估计是准确的。然而，当这些估计基于有限样本时可能并不可靠。为了同时处理跨客户端混合不确定性和客户端内部的分布模糊性，我们构建了一个全局模糊集，将其定义为局部模糊集的可容许混合的并集。该构造允许使用客户端特定的模糊半径，并且可以进行客户端可分离的重新表述。利用这一结构，我们建立了高概率的样本外性能保证。我们进一步针对基于惩罚项的重新表述开发了一种联邦算法，并证明了……

    arXiv:2609.20501v1 Announce Type: new  Abstract: Federated learning trains a shared model from private client data. In practice, data-generating distributions may differ, and the true mixture across clients is often unknown, making the underlying group distribution difficult to specify. Existing approaches address cross-client mixture uncertainty by optimizing against the worst-case mixture, yet assume accurate client-wise distribution estimates. However, these estimates can be unreliable when based on finite samples. To handle both cross-client mixture uncertainty and within-client distributional ambiguity, we construct a global ambiguity set as the union of admissible mixtures of local ambiguity sets. The construction allows client-specific ambiguity radii and admits a client-wise separable reformulation. Leveraging this structure, we establish a high-probability out-of-sample performance guarantee. We further develop a federated algorithm for a penalty-based reformulation and prove 
    
[^38]: 基于事件数据的过程比较分辨率极限

    Resolution limits for process comparison from event data

    [https://arxiv.org/abs/2609.20489](https://arxiv.org/abs/2609.20489)

    本文证明了基于事件日志随机语言的标准过程挖掘方法无法从数据中区分并发与顺序行为，刻画了这种分辨率极限，并指出可以通过活动起止时间或以对象为中心的记录等被随机语言丢弃的证据来恢复这种区分。

    

    一家医院同时进行血液检查和影像检查，另一家医院则按先后顺序进行，且两种顺序出现的频率相同。了解实际发生了什么，以及它是如何被记录在数据中的，对所有运营管理者至关重要。在过程挖掘中，标准方法是构建事件日志，并尝试以数据驱动的方式发现并发和顺序过程。我们证明了这种基于事件日志随机语言的标准方法，只能报告其发现算法的假设，因为每一个这样的日志都可以被一个完全没有并发性的模型同样好地解释。此外，在获取任何数据之前，我们刻画了数据在何时能够以及何时无法区分并发行为。当数据无法区分时，这种区分可以从随机语言所丢弃的证据中恢复，例如活动开始和结束的时间，或在单次执行中固定顺序的以对象为中心的记录。

    arXiv:2609.20489v1 Announce Type: cross  Abstract: One hospital runs bloods and imaging at the same time. Another runs them one after the other, in either order, equally often. Knowing which actually happened, and how it is recorded in data, is critical for all operational managers. In process mining, the standard approach is to construct an event log, and attempt to discover concurrent and sequential processes in a data-driven way. We show this standard approach, built on the stochastic language of an event log, reports only the assumptions of its discovery algorithm, because every such log is explained equally well by a model with no concurrency at all. Further, before any data is acquired, we characterise when data can and cannot distinguish concurrent behaviour. Where it cannot, the distinction is recoverable from evidence the stochastic language discards, such as the times at which activities start and end, or object-centric records that fix an order within an execution. The remed
    
[^39]: 基于深度学习的脑电图信号认知状态与静息状态分类

    Deep Learning-Based Classification of Cognitive and Resting States Using Electroencephalography Signals

    [https://arxiv.org/abs/2609.20467](https://arxiv.org/abs/2609.20467)

    该研究提出了一种将卷积神经网络（CNN）与门控循环单元（GRU）相结合的2D-Net深度学习框架，并通过时频分析提取EEG信号特征，实现了对认知状态与静息状态的有效分类。

    

    从脑电图（EEG）信号中对认知状态和静息状态进行分类，对于理解与各种精神状态相关的大脑活动波动至关重要。EEG提供了一种非侵入式方法，可在静息和任务导向的认知条件下记录大脑功能，而深度学习技术能够从复杂的EEG数据中自动提取有意义的模式。本研究提出了一种深度学习框架，通过EEG记录区分静息状态和认知状态。所提出的框架集成了卷积神经网络（CNN）与门控循环单元（GRU），用于从EEG信号中提取特征。研究进行了时频分析以探索信号的显著特征，随后利用传统的深度学习和机器学习分类器（包括所提出的2D-Net架构）对提取的特征进行评估。

    arXiv:2609.20467v1 Announce Type: cross  Abstract: The categorization of cognitive and resting states derived from electroencephalography (EEG) signals is crucial for comprehending fluctuations in brain activity linked to various mental states. EEG provides a non-intrusive approach for documenting brain function in both resting and task-oriented cognitive conditions, whilst deep learning techniques enable the automatic extraction of significant patterns from intricate EEG data. This study presents a deep learning framework to distinguish between resting and cognitive states through EEG records. The proposed framework integrates a Convolutional Neural Network (CNN) stacked with a Gated Recurrent Unit (GRU) for the extraction of features from EEG signals. Time-frequency analysis is conducted to explore the salient aspects of signals, and the derived features are then assessed utilizing conventional deep learning and machine learning classifiers, including the suggested 2D-Net architectur
    
[^40]: 训练神经网络以逼近密集多发射源定位中的最优贝叶斯估计器

    Training Neural Networks to Approach the Optimum Bayes Estimator in Dense Multi-Emitter Localization

    [https://arxiv.org/abs/2609.20465](https://arxiv.org/abs/2609.20465)

    该论文通过在合成帧上训练神经网络来逼近密集发射源定位的最优贝叶斯估计器，为实现高通量、大视场、超时空分辨率的单分子定位显微镜奠定了基础。

    

    我们在合成帧上训练神经网络，以逼近密集发射源定位中的最优贝叶斯估计器。该结果为未来通过训练神经网络实现高通量、大视场、超时空分辨率的单分子定位显微镜（SMLM）的研究工作提供了依据。

    arXiv:2609.20465v1 Announce Type: new  Abstract: We train neural networks on synthesized frames to approach the optimum Bayes estimator for dense emitter localization. The result justifies the future work on training neural networks to achieve high-throughput large-FOV super spatiotemporal resolution SMLM.
    
[^41]: 通过 Committor 学习引导的射击点生成实现无关联跃迁路径采样

    Correlation-Free Transition Path Sampling through Shooting Point Generation Guided by Committor Learning

    [https://arxiv.org/abs/2609.20461](https://arxiv.org/abs/2609.20461)

    该论文提出利用 Committor 学习引导射击点生成，在无需预先已知反应坐标的情况下实现无关联跃迁路径采样，从而克服了传统跃迁路径采样方法中路径相关性导致效率受限的问题。

    

    研究系统的动力学行为通常依赖于刻画其在长寿命状态之间的跃迁方式。由于这类跃迁属于稀有事件，观测它们通常需要专门的增强采样技术。跃迁路径采样（Transition Path Sampling, TPS）是一种成熟的生成反应性轨迹的方法，其实现简单，且不需要预先定义反应坐标。然而，其效率受到其顺序性本质以及由此导致的采样路径之间相关性的限制。先前的工作通过将 TPS 与基于条件玻尔兹曼生成器的采样方案相结合来解决这一限制，条件玻尔兹曼生成器是一种能够对给定目标概率分布进行采样的生成式机器学习模型。这种方法可以产生无关联的跃迁路径，但其依赖于一个准确描述的反应坐标，而反应坐标很少能提前得知。基于 Committor 学习领域的最新进展……（摘要原文在此处被截断）

    arXiv:2609.20461v1 Announce Type: cross  Abstract: Studying the dynamical behavior of a system often depends on characterizing how it transitions between long-lived states. Because such transitions are rare, observing them usually requires specialized enhanced sampling techniques. Transition Path Sampling (TPS) is a well-established method for generating reactive trajectories, which is simple to implement and does not require the definition of a preconceived reaction coordinate. However, its efficiency is limited by its sequential nature and the resulting correlations between sampled paths. Previous work addressed this limitation by combining TPS with a sampling scheme based on conditioned Boltzmann Generators, a generative machine learning model capable of sampling a given target probability distribution. This approach produces uncorrelated transition paths but relies on an accurate reaction coordinate, which is rarely known in advance. Building on recent advances in committor learnin
    
[^42]: 基于随机特征的在线监督降维：诊断与计算权衡

    Online Supervised Dimension Reduction with Random Features: Diagnostics and Computational Trade-offs

    [https://arxiv.org/abs/2609.20454](https://arxiv.org/abs/2609.20454)

    本文为基于随机特征的在线核监督主成分分析（OKSPCA）提供了一致性、集中性与扰动的理论诊断，并通过六个基准实验揭示：精确优化监督谱目标并不能保证得到精确的总体子空间或更优的预测表示，从而阐明了目标优化精度、子空间恢复精度与预测性能之间的计算权衡。

    

    对监督谱目标的精确优化未必能产生精确的总体子空间或更好的预测表示。我们针对在线核监督主成分分析（OKSPCA）研究了这些区别，该方法将有限随机特征坐标中的中心化交叉矩与Adam式的正交基更新相结合，用于一个已确立的目标。固定映射一致性、集中性与扰动结果刻画了估计器及其精确子空间；随后通过同目标比较分别评估实际迭代过程。在六个预测基准上，性能取决于所声明的流水线：用精确的经验目标替换跟踪器后，两个回归缺陷基本保持不变。直接的分类秩模型平均而言几乎捕获了全部终端目标能量，但保存的中间状态表现出显著的几何偏差；一个受控的样本……

    arXiv:2609.20454v1 Announce Type: cross  Abstract: Accurate optimization of a supervised spectral objective need not produce an accurate population subspace or a better predictive representation. We investigate these distinctions for Online Kernel Supervised Principal Component Analysis (OKSPCA), which combines a centered cross-moment in finite random-feature coordinates with an Adam-style orthonormal basis update for an established objective. Fixed-map consistency, concentration and perturbation results describe the estimator and its exact subspace; same-target comparisons then assess the practical iterate separately. Across six predictive benchmarks, performance depends on the declared pipeline: replacing the tracker with the exact empirical target leaves the two regression deficits largely unchanged. Direct classification-rank models capture nearly all terminal objective energy on average, but a saved intermediate state exhibits substantial geometric deviation; a controlled sample-s
    
[^43]: 基于有限元预训练潜在动力学的稀疏观测地震场地响应预测

    Seismic Site Response Prediction from Sparse Observations Using Finite-Element-Pretrained Latent Dynamics

    [https://arxiv.org/abs/2609.20451](https://arxiv.org/abs/2609.20451)

    本研究提出FLARE-T框架，通过从密集有限元模拟中学习低维潜在动力学并用稀疏观测记录进行校准，显著提升了地震场地响应预测的准确性，并通过离心机试验和Lotung现场垂直阵列数据验证了其有效性。

    

    数值场地响应预测结果常常与实际观测存在偏差，但由于记录在传感器覆盖范围和事件数量上都较为有限，校正这些偏差十分困难。本研究提出了迁移使能强迫潜在自编码器响应方程方法（FLARE-T），通过学习并校准连接基底加速度输入与多深度加速度输出的低维潜在动力学，来改进数值预测。FLARE-T首先从密集的有限元模拟中学习低维响应流形和输入驱动的动力学，然后训练一个稀疏编码器，将模拟的传感器响应映射到所学习的坐标中，并利用有限的记录来校准其中的动力学。该方法采用短响应窗口初始化每次预测，同时由完整的基底运动驱动整个响应过程。该框架通过分层土离心机试验和Lotung场地垂直阵列数据进行了评估。

    arXiv:2609.20451v1 Announce Type: new  Abstract: Numerical site-response predictions often deviate from observations, yet correcting these discrepancies is difficult because records are limited in both sensor coverage and number of events. This study proposes the Transfer-Enabled Forced Latent Autoencoder for Response Equations (FLARE-T) to improve these predictions by learning and calibrating low-dimensional latent dynamics that connect the base acceleration input to acceleration outputs at multiple depths. FLARE-T learns a low-dimensional response manifold and input-driven dynamics from dense finite-element simulations. It then trains a sparse encoder to map simulated sensor responses into the learned coordinates and uses limited records to calibrate the dynamics within them. A short response window initializes each prediction, while the complete base motion drives the response. The framework was evaluated using a layered-soil centrifuge test and the Lotung field vertical array. Test
    
[^44]: 面向边缘洪水分割的跨架构基础模型蒸馏

    Cross-Architecture Foundation-Model Distillation for Edge Flood Segmentation

    [https://arxiv.org/abs/2609.20441](https://arxiv.org/abs/2609.20441)

    该论文通过将3亿参数的地理空间基础模型蒸馏到仅70万参数的EfficientViT-B0学生模型中，并利用教师模型对未标注Sentinel-2影像生成伪标签来扩展训练数据，实现了可在边缘设备部署的高性能洪水分割模型。

    

    地理空间基础模型能够提供强大的洪水分割性能，但其庞大的规模限制了在内存受限的边缘硬件上的部署。我们将一个在252个手工标注的Sen1Floods11训练场景上微调的3亿参数Prithvi-EO-2.0教师模型，蒸馏为70万参数的EfficientViT-B0学生模型。教师模型为额外的未标注Sentinel-2影像提供监督信号，使学生模型的训练集无需新的人工标注即可扩展。在252个场景的匹配预算下，教师监督训练与直接训练相比具有竞争力，并在测试配置中提升了STURM-Flood的性能；几何匹配的对照实验表明，仅凭标签来源并不能解释这一差异。将教师监督数据池扩展至2,500个场景缩小了学生与教师模型之间剩余的差距：浮点学生模型在Sen1Floods11测试集上达到0.787的水体交并比（IoU），而教师模型为0.822

    arXiv:2609.20441v1 Announce Type: cross  Abstract: Geospatial foundation models can provide strong flood-segmentation performance, but their size limits deployment on memory-constrained edge hardware. We distill a 300-million-parameter Prithvi-EO-2.0 teacher, fine-tuned on the 252 manually labeled Sen1Floods11 training scenes, into a 0.7-million-parameter EfficientViT-B0 student. The teacher supervises additional unlabeled Sentinel-2 imagery, allowing the student training set to grow without new manual annotations. At the matched budget of 252 scenes, teacher-supervised training is competitive with direct training and improves STURM-Flood performance across tested configurations; a geometry-matched control shows that label source alone does not explain the difference. Scaling the teacher-supervised pool to 2,500 scenes narrows the remaining student--teacher gap: the float student reaches 0.787 water intersection over union on the Sen1Floods11 test split against 0.822 for the teacher, m
    
[^45]: SCGFM-ART：面向结构中心的图基础模型的摊销关系传输

    SCGFM-ART: Amortized Relational Transport for Structure-Centric Graph Foundation Models

    [https://arxiv.org/abs/2609.20419](https://arxiv.org/abs/2609.20419)

    SCGFM-ART提出了一种以结构为中心的图基础模型框架，通过摊销关系传输将任意异构图直接对齐到由有限关系基准定义的共享关系图谱坐标系上，无需代价高昂的运行时Gromov-Wasserstein优化，即可从全局和局部两个层面实现跨域统一的图表示学习。

    

    图基础模型（GFMs）旨在跨严重异构的图域学习可迁移的表示。然而，拓扑结构、图规模和特征语义方面的严重域偏移阻碍了统一的、与领域无关的表示空间的构建。为了解决这一问题，我们提出了SCGFM-ART，这是一个以结构为中心的图基础模型框架，它通过摊销关系传输（ART）将任意图对齐到一个共享的关系图谱上。关系图谱作为一个由有限的关系基准集合定义的通用坐标系，而ART则直接预测可复用的、端到端的图到基准的传输计划，从而避免了代价高昂的运行时Gromov-Wasserstein优化。在此公式化框架下，SCGFM-ART将图分解为一种统一的表示：全局层面通过其相对于图谱的关系响应坐标，局部层面通过其节点到角色的结构对应关系。这些对应关系将……

    arXiv:2609.20419v1 Announce Type: cross  Abstract: Graph foundation models (GFMs) aim to learn transferable representations across severely heterogeneous graph domains. However, severe domain shifts in topology, graph scale, and feature semantics impede the construction of a unified, domain-agnostic representation space. To address this, we propose SCGFM-ART, a structure-centric GFM framework that aligns arbitrary graphs onto a shared relational atlas via Amortized Relational Transport (ART). The relational atlas serves as a universal coordinate system defined by a finite set of relational landmarks (bases), while ART directly predicts reusable, end-to-end graph-to-base transport plans, bypassing costly runtime Gromov-Wasserstein optimizations. Under this formulation, SCGFM-ART decomposes a graph into a unified representation: globally via its relational response coordinates relative to the atlas, and locally via its node-to-role structural correspondences. These correspondences projec
    
[^46]: 常数步长下非线性双时间尺度随机逼近的偏差

    The Bias of Nonlinear Two-Time-scale Stochastic Approximation under Constant Step-Sizes

    [https://arxiv.org/abs/2609.20409](https://arxiv.org/abs/2609.20409)

    本文首次对常数步长下的非线性双时间尺度随机逼近给出了紧的均方误差与偏差上界 $O(\alpha+\beta^2/\alpha^2)$，并通过分离各项误差来源阐明了 $\beta^2/\alpha^2$ 偏差项的产生机制。

    

    双时间尺度随机逼近（TTSA）是分析强化学习、优化和随机控制中耦合迭代算法的基础工具。然而，非线性双时间尺度方案的有限时间保证仍然难以获得，尤其是在常数步长的情形下。在本文中，我们研究了步长满足 $\alpha\gg\beta$ 的非线性TTSA。在标准的稳定性、正则性和马尔可夫噪声假设下，我们对两种迭代围绕其极限平衡点的均方误差和偏差给出了上界。我们的界为 $O(\alpha+\beta^2/\alpha^2)$，并证明了当 $\beta\le\alpha^{3/2}$ 时该界是紧的。我们的分析分离了初始条件、快时间尺度跟踪误差、马尔可夫依赖性以及时间尺度耦合各自的贡献，从而阐明了 $\beta^2/\alpha^2$ 项的来源。我们的结果揭示了其与先前研究的线性TTSA设定之间的定性差异。

    arXiv:2609.20409v1 Announce Type: new  Abstract: Two-timescale stochastic approximation (TTSA) is a fundamental tool for analyzing coupled iterative algorithms in reinforcement learning, optimization, and stochastic control. However, finite-time guarantees for nonlinear two-timescale schemes remain difficult to obtain, especially under constant step-sizes. In this paper, we study nonlinear TTSA with step-sizes $\alpha\gg\beta$. Under standard stability, regularity, and Markovian noise assumptions, we upper bound the mean-squared error and the bias of both iterates around their limiting equilibria. Our bounds scale as $O(\alpha+\beta^2/\alpha^2)$, which we prove to be tight when $\beta\le\alpha^{3/2}$. The analysis separates the contributions of initial conditions, fast-timescale tracking error, Markovian dependence, and timescale coupling, thereby clarifying the origin of the $\beta^2/\alpha^2$ term. Our results reveal qualitative differences from the linear TTSA setting previously stu
    
[^47]: 在道德风险与逆向选择下学习面向公平小农户碳农业的委托-代理合同

    Learning Principal-Agent Contracts for Equitable Smallholder Carbon Farming under Moral Hazard and Adverse Selection

    [https://arxiv.org/abs/2609.20404](https://arxiv.org/abs/2609.20404)

    本研究将面向小农户的碳农业合同设计问题建模为POMDP，利用强化学习学习在道德风险与逆向选择下的动态利润最大化合同，以弥合碳计划难以惠及小农户的现实差距。

    

    农业土壤是一个尚未被开发的主要碳汇，碳农业正在成为挖掘这一潜力的有前景的做法。主导南亚和撒哈拉以南非洲农业的小农户是通过碳农业扩大气候减缓规模的关键。具有讽刺意味的是，现实世界中的碳计划在很大程度上未能惠及他们。我们通过合同设计的视角研究这一重要差距。一个聚合商向异质性的小农户群体提供单一的组合合同，这些小农户拥有私人的采用成本（逆向选择）并付出不可观察的努力（道德风险），农艺结果随多个季节不断演变。我们将这一动态演变的合同问题表述为部分可观测马尔可夫决策过程（POMDP），并使用强化学习来学习动态的利润最大化合同。我们分析了聚合商在各种条件下的表现。我们发现，利润最大化的聚合商不仅仅继承了排他性……（摘要原文在此处截断）

    arXiv:2609.20404v1 Announce Type: new  Abstract: Agricultural soils are a major untapped carbon sink. Carbon farming is emerging as a promising practice for tapping this potential. Smallholder farmers, who dominate agriculture across South Asia and sub-Saharan Africa, are key to scaling climate mitigation via carbon farming. It is ironic that real-world carbon programs largely fail to reach them. We study this important gap through the lens of contract design. An aggregator offers a single pooled contract to a heterogeneous population of smallholder farmers who have private adoption costs (adverse selection) and exert unobserved effort (moral hazard), with agronomic outcomes evolving over multiple seasons. We formulate this evolving contracting problem as a POMDP and use reinforcement learning to learn a dynamic profit-maximising contract. We analyse the performance of the aggregator under various conditions. We find that a profit-maximising aggregator does not merely inherit the exclu
    
[^48]: 表格型强化学习中离线策略评估的基于模型的自助法（Bootstrap）框架

    Model-based Bootstrap for Offline Policy Evaluation in Tabular Reinforcement Learning

    [https://arxiv.org/abs/2609.20389](https://arxiv.org/abs/2609.20389)

    该论文提出了一种基于模型的自助法框架，通过从估计的MDP重新生成轨迹来对表格型强化学习中的离线策略评估进行不确定性量化，从而克服了经典重采样方法在鲁棒性、可扩展性和有限样本有效性上的局限。

    

    离线策略评估（OPE）在高风险强化学习应用中至关重要，在这些应用中，新策略必须在部署前得到可靠评估。在此类场景下，仅有点估计是不够的；有原则的不确定性量化，例如置信区间和方差估计，对于安全且具备风险意识的决策必不可少。统一这些任务的一种全面途径是估计评估误差的抽样分布。然而，现有方法往往在鲁棒性、可扩展性或有限样本有效性方面存在不足。本文提出了一种基于模型的自助法框架，用于有限时域、时变非齐次马尔可夫决策过程（MDP）中离线策略评估的不确定性量化。与依赖对完整回合进行重采样的经典自助法不同，所提方法从估计得到的MDP中重新生成轨迹，因此能够适用于范围更广的……（摘要在此处截断）

    arXiv:2609.20389v1 Announce Type: cross  Abstract: Offline policy evaluation (OPE) is crucial in high-stakes reinforcement learning applications, where new policies must be assessed reliably before deployment. In such settings, point estimates alone are insufficient; principled uncertainty quantification, such as confidence intervals and variance estimates, is essential for safe and risk-aware decision-making. A comprehensive way to unify these tasks is to estimate the sampling distribution of the evaluation error. Existing approaches, however, often suffer from limited robustness, scalability, or finite-sample validity. In this paper, we propose a model-based bootstrap framework for uncertainty quantification of OPE in finite-horizon, time-inhomogeneous Markov decision processes (MDPs). Unlike classical bootstrap methods that rely on resampling complete episodes, the proposed method regenerates trajectories from an estimated MDP and can therefore accommodate a much broader range of of
    
[^49]: 具有无限制有界合同的极小极大最优在线合同设计

    Minimax-Optimal Online Contract Design with Unrestricted Bounded Contracts

    [https://arxiv.org/abs/2609.20353](https://arxiv.org/abs/2609.20353)

    该论文证明了在委托人仅能观察结果而无法观察行动的重复合同设计中，即使允许任意有界的支付方案、代理利润可能不连续且行动空间完全任意，极小极大遗憾仍为 $T^{m/(m+1)}$ 量级，并通过有效维度约简与显示偏好构造的单调反应映射设计出达到该最优速率的学习策略。

    

    我们研究重复合同设计问题，其中委托人只能观察到结果，而无法观察到产生这些结果的行动。委托人可以使用任意有界的基于结果的支付向量，而代理人的最优反应可能使期望利润对这些支付不连续。对于每个固定的结果数量 $m\ge2$，$T$ 轮上的极小极大遗憾（在至多相差对数因子的意义下）为 $T^{m/(m+1)}$ 量级。该上界允许任意的行动空间和代理人异质性，无需光滑性或单调盈余假设。证明的关键在于一种有效维度约简技术：即使固定的打破平局方式不满足平移不变性，基准仍可被规范化，之后利用显示偏好在支付差坐标中得到一个单调的反应映射。基于该映射的 Lipschitz 参数化构建的学习策略，仅使用观察到的结果类别即可达到该速率。下界构造刻画了激励损失如何累积……

    arXiv:2609.20353v1 Announce Type: new  Abstract: We study repeated contract design when a principal observes outcomes but not the actions that generate them. The principal may use any bounded outcome-contingent payment vector, and the agent's best response can make expected profit discontinuous in those payments. For every fixed number $m\ge2$ of outcomes, the minimax regret over $T$ rounds is of order $T^{m/(m+1)}$, up to logarithmic factors. The upper bound allows arbitrary action spaces and agent heterogeneity, without smoothness or monotone-surplus assumptions. Its key is an effective-dimension reduction that the benchmark can be normalized even when fixed tie-breaking is not shift invariant, after which revealed preference yields a monotone response map in payment-difference coordinates. A learning policy built on a Lipschitz parametrization of this map attains the rate using only observed outcome categories. The lower-bound construction accounts for how incentive losses accumulat
    
[^50]: COMPASS：10万规模的有序聚类路由算法

    COMPASS: Ordered Clustered Routing at 100K Scale

    [https://arxiv.org/abs/2609.20352](https://arxiv.org/abs/2609.20352)

    COMPASS算法通过协调并行子求解器将搜索与学习加速路由相结合，解决了有序聚类旅行商问题（OCTSP），可在10万规模节点上获得持续改进的高质量解，并支持通用距离矩阵输入。

    

    大规模路由通常需要按规定顺序访问节点簇，由此产生了有序聚类旅行商问题。独立优化每个簇看似自然，但会忽略非局部的依赖关系。我们提出了针对OCTSP的COMPASS算法，该算法通过协调并行子求解器，将搜索与学习加速的路由相结合。COMPASS没有质量上限，其解会随着计算资源的增加而持续改进。它充分利用了聚类结构，可以在与簇大小（而非实例大小）呈指数关系的时间内达到精确解。实验表明，COMPASS始终优于其他替代方法。与常见的大规模路由求解器不同，COMPASS可以直接使用通用距离矩阵，而不局限于坐标输入。我们展示了该算法在10万个合成节点和2.85万个真实电商节点上的扩展能力。据我们所知，后者是迄今为止已报道的最大的非对称距离路由求解实例。

    arXiv:2609.20352v1 Announce Type: new  Abstract: Large-scale routing often requires visiting clusters of nodes in a prescribed order, giving rise to the Ordered Clustered Traveling Salesman Problem (OCTSP). Optimizing each cluster independently seems natural, but misses non-local dependencies. We introduce the COMPASS algorithm for OCTSP, which combines search with learning-accelerated routing by orchestrating parallel sub-solvers. COMPASS has no quality ceiling and its solutions keep improving with compute. It exploits the clustered structure, and can reach exact solutions in time exponential in cluster size rather than instance size. Empirically, COMPASS consistently outperforms alternative methods. Unlike common large-scale routing solvers, COMPASS consumes general distance matrices and is not limited to coordinate inputs. We demonstrate scaling to 100K synthetic nodes and to 28.5K real e-commerce nodes. To our knowledge, the latter is the largest reported routing solution over asym
    
[^51]: 基于潜在桥接匹配的快速跨场强多对比度脑部MRI翻译

    Fast Cross-Strength Multi-Contrast Brain MRI Translation using Latent Bridge Matching

    [https://arxiv.org/abs/2609.20341](https://arxiv.org/abs/2609.20341)

    提出基于条件潜在桥接匹配的统一模型，无需任务特定架构即可实现单步推理的跨场强多对比度脑部MRI快速翻译，在MRIxFields2026挑战赛全部三项任务中取得有竞争力的结果。

    

    在不同场强下采集的磁共振成像（MRI）在噪声、分辨率、均匀性和对比度方面表现出显著差异，这限制了不同采集设置之间的可比性，并使下游分析变得复杂。我们通过一个基于条件潜在桥接匹配框架构建的统一条件模型来解决这一问题，实现可控的场强到场强合成。我们的单一模型无需任务特定的架构或训练，在MRIxFields2026挑战赛验证阶段的全部三项任务中均取得了极具竞争力的结果。我们仅需单步推理即可实现快速生成，在单块NVIDIA A5000 GPU上，可在90秒内生成30个轴位切片的所有模态和场强组合，并在70秒内完成整个体积的跨模态-场强翻译。此外，我们还针对解决方案的各个组件提供了广泛的消融实验。

    arXiv:2609.20341v1 Announce Type: cross  Abstract: Magnetic Resonance Imaging (MRI) acquired at different field strengths exhibits pronounced variation in noise, resolution, homogeneity, and contrast, which limits comparability across acquisition settings and complicates downstream analysis. We address this with a unified conditional model for controllable field-to-field synthesis, built on the framework of conditional latent bridge matching. Our single model achieves highly competitive results across the validation phase for all three tasks of the MRIxFields2026 challenge without task-specific architectures or training. We achieve fast generation with only a single inference step, producing all modality and field-strength combinations for $30$ axial slices in under $90$ seconds, as well as cross-modality-strength translation for a full volume in under $70$ seconds, on a single NVIDIA A5000 GPU. We further provide extensive ablations regarding different components of our solution. Code
    
[^52]: 检测欺骗性招聘：一种基于信号理论的劳动力剥削早期识别机器学习框架

    Detecting Deceptive Recruitment: A Signal-theoretic Machine Learning Framework for Early Identification of Labour Exploitation

    [https://arxiv.org/abs/2609.20336](https://arxiv.org/abs/2609.20336)

    该研究基于信号理论将欺骗性招聘广告检测形式化为分类问题，利用464个真实案例构建了融合计算机视觉、自然语言处理和语义嵌入的多模态机器学习框架，为强迫劳动的早期识别提供了实证验证的有效方法。

    

    欺骗性在线招聘广告已成为强迫劳动的主要途径，然而由于数据稀缺和缺乏经过实证验证的指标，系统的检测方法仍发展不足。我们将这一检测挑战形式化为信号理论下的分类问题，其中剥削者通过文本、视觉和结构维度传递模仿合法通信的无成本信号。使用通过反奴隶制慈善机构收集的来自九个来源国和21个行业的464个经验证案例（164个欺骗性案例，300个合法案例），我们开发了结合计算机视觉、自然语言处理和语义嵌入的多模态检测模型。通过系统的特征消融实验和重复分层交叉验证，我们证明单一模态即可实现显著的判别能力（ROC-AUC：0.87—0.97），而多模态的整合可带来适度的进一步提升。

    arXiv:2609.20336v1 Announce Type: cross  Abstract: Deceptive online job advertisements have emerged as a primary pathway into forced labour, yet systematic detection methods remain underdeveloped due to data scarcity and absence of empirically validated indicators. We formalise this detection challenge as a classification problem under signalling theory, where exploiters transmit costless signals mimicking legitimate communications across textual, visual, and structural dimensions. Using 464 verified cases (164 deceptive, 300 legitimate) collected through anti-slavery charities across nine origin countries and 21 industries, we develop multimodal detection models combining computer vision, natural language processing, and semantic embeddings. Through systematic feature ablation experiments and repeated stratified cross-validation, we demonstrate that individual modalities achieve substantial discriminatory power (ROC-AUC: 0.87--0.97), whilst their integration yields modest further gain
    
[^53]: 使用相同前向映射的自编码器的尖锐重构界

    Sharp Reconstruction Bounds for Autoencoders Using the Same Forward Map

    [https://arxiv.org/abs/2609.20333](https://arxiv.org/abs/2609.20333)

    该论文证明了在雅可比奇异值受约束条件下，使用相同前向映射的自编码器重构导数误差的精确下界，并通过真实地面激光雷达森林扫描数据实验验证了该理论界的预测能力。

    

    我们研究了自编码器中的重构问题，这类自编码器在将观测坐标置零之前和之后应用相同的前向映射。对于输入维度和隐藏维度相等的奇数 $d\geq 3$，在雅可比矩阵奇异值位于 $[m,M]$ 区间内的保定向微分同胚中，我们证明了最小的一致重构导数误差为 $\max\{1-M(M-m)/2,0\}$，且仿射映射在任意指定深度下都能达到这一尖锐界。然而，一个平移的径向旋转能够以任意接近于1的奇异值精确重构任意指定的球体，这促使我们为有限数据界引入额外的条件。我们在一个包含798,452个点的地面激光雷达森林扫描数据上验证了这一理论预测。在输入尺度 $0.05$ 下，理论界的均值为 $0.155$，约为四个空间区域、两个深度和三个随机种子下平均归一化训练误差 $0.185$ 的 $84\%$。在该尺度下，增加一个隐藏坐标可降低平均重构误差……

    arXiv:2609.20333v1 Announce Type: new  Abstract: We study reconstruction in autoencoders that apply the same forward map before and after setting the observed coordinates to zero. For equal odd input and hidden dimensions $d\geq 3$, among orientation-preserving diffeomorphisms whose Jacobian singular values lie in $[m,M]$, we show that the least uniform reconstruction-derivative error is $\max\{1-M(M-m)/2,0\}$, with affine maps attaining this sharp bound at every prescribed depth. A translated radial rotation can nevertheless reconstruct any prescribed ball exactly with singular values arbitrarily close to one, motivating additional conditions for a finite-data bound. We test this prediction on a 798,452-point terrestrial LiDAR forest scan. At input scale $0.05$, the mean theoretical bound is $0.155$, about $84\%$ of the mean normalized training error $0.185$ across four spatial regions, two depths, and three seeds. At this scale, adding one hidden coordinate reduces the mean reconstru
    
[^54]: 用于强凸-强凹极小极大优化的近最优纯单循环外梯度方法

    Near-Optimal Pure Single-Loop Extragradient Method for Strongly Convex--Strongly Concave Minimax Optimization

    [https://arxiv.org/abs/2609.20327](https://arxiv.org/abs/2609.20327)

    提出了一种参数固定的纯单循环阻尼外梯度方法用于强凸-强凹极小极大优化，无需内层求解或重启策略，即可以 O(√(κ_x κ_y) log(1/ε)) 的近最优梯度查询复杂度实现末次迭代线性收敛。

    

    我们研究确定性无约束设置下具有一般非线性耦合的光滑强凸-强凹极小极大优化问题。我们提出了一种纯单循环阻尼外梯度方法，其参数固定，在初始一次梯度查询后，每次迭代仅需两次新的全梯度评估。该方法使用辅助反馈递归，无需内层求解、精度调度或分阶段重启。我们建立了末次迭代的线性收敛性，并证明将到鞍点的欧氏距离平方减少到其初始值的ε倍需要 O(√(κ_x κ_y) log(2κ_x κ_y/ε)) 次全梯度查询，其中 κ_x = L/μ_x，κ_y = L/μ_y。该界通过固定的显式更新，在对数因子范围内达到了关于条件数的最优阶。数值实验证明了该方法的有效性。

    arXiv:2609.20327v1 Announce Type: cross  Abstract: We study smooth strongly convex--strongly concave minimax optimization with general nonlinear coupling in the deterministic unconstrained setting. We propose a pure single-loop damped extragradient method with fixed parameters and two new full-gradient evaluations per iteration after one initialization query. The method uses an auxiliary feedback recursion and requires no inner solves, accuracy schedules, or staged restarts. We establish last-iterate linear convergence and show that reducing the squared Euclidean distance to the saddle point to an $\varepsilon$ fraction of its initial value requires $O(\sqrt{\kappa_x\kappa_y}\log(2\kappa_x\kappa_y/\varepsilon))$ full-gradient queries, where $\kappa_x=L/\mu_x$ and $\kappa_y=L/\mu_y$. This bound attains the optimal condition-number order up to logarithmic factors through fixed explicit updates. Numerical experiments demonstrate the effectiveness of the method.
    
[^55]: QEncodeBench：大语言模型能否将经典问题编码为经验证的量子预言机？

    QEncodeBench: Can Large Language Models Encode Classical Problems into Verified Quantum Oracles?

    [https://arxiv.org/abs/2609.20319](https://arxiv.org/abs/2609.20319)

    该论文提出QEncodeBench基准，首次通过对抗性自验证的完整解集等价性检查，系统测量大语言模型将经典约束问题编码为经验证量子相位预言机的能力，发现无推理模式的代码模型几乎完全失败，而启用原生推理可使准确率提升一个数量级。

    

    Grover搜索、振幅放大和量子计数都依赖于同一个可复用的子程序——相位预言机，而算法文献通常将其构造视为既定前提：假定经典谓词已经被编码为正确的、资源受限的电路。我们将这一假设转化为可测量的能力。QEncodeBench让大语言模型（LLM）承担将经典约束问题编码为相位预言机的任务，并使用一个对抗性自验证的验证器对生成的电路进行评分，该验证器能够判定完整解集在全局相位意义下的等价性，同时要求辅助比特恢复原状并强制执行资源预算。我们证明，基于采样的基态测试会系统性高估这种能力。以这种方式测量，各模型之间呈现显著分化：不带推理模式的代码模型几乎无法解决任何问题，而在相同模型权重上启用原生推理模式可使准确率提升一个数量级。失败案例绝大多数……（摘要原文在此处截断）

    arXiv:2609.20319v1 Announce Type: cross  Abstract: Grover search, amplitude amplification, and quantum counting all rely on the same reusable subroutine, a phase oracle, whose construction the algorithms literature takes as given: the classical predicate is assumed to be already encoded as a correct, resource-bounded circuit. We turn this assumption into a measured capability. QEncodeBench tasks large language models (LLMs) with encoding classical constraint problems as phase oracles and scores the generated circuits with an adversarially self-validated verifier that decides full solution-set equivalence up to a global phase, with ancillas restored and resource budgets enforced. Sampled basis-state tests, we show, systematically overestimate this ability. Measured this way, models separate sharply: code models without a reasoning mode solve essentially nothing, and enabling native reasoning on identical weights improves accuracy by an order of magnitude. The failures are overwhelmingly
    
[^56]: EviRec：面向双重冷启动POI推荐的持续证据学习

    EviRec: Continual Evidence Learning for Dual Cold-Start POI Recommendation

    [https://arxiv.org/abs/2609.20313](https://arxiv.org/abs/2609.20313)

    提出了EviRec持续证据学习框架，通过匹配、转移记忆和生命周期三个互补视角评估每个候选POI历史证据的可信度，以解决用户与POI同时更替带来的双重冷启动推荐问题。

    

    兴趣点（POI）推荐是基于位置的服务中的一项核心任务，然而大多数现有方法都假设用户群体和POI目录是固定不变的。通过对美国10个城市进行大规模的数据驱动分析，我们发现了显著的POI流失、用户更替、类别漂移以及静态POI记忆的衰减现象，这促使我们开展对持续双重冷启动POI推荐的研究。为应对这一场景，我们提出了EviRec，这是一个持续证据学习框架，能够针对每个候选POI单独估计其历史证据的可信程度。EviRec从三个互补的视角对每个可见候选进行评分：基于用户近期出行画像的匹配视角、捕捉重复出行规律的转移记忆视角，以及反映候选成熟度的生命周期视角。由于接近零的转移分数可能意味着不相关或观测不足，EviRec据此对证据进行评估与校准。

    arXiv:2609.20313v1 Announce Type: new  Abstract: Point-of-Interest (POI) recommendation is a core task in location-based services, yet most existing methods assume a fixed user population and POI catalog. Through a large-scale data-driven analysis of 10 U.S. cities, we identify substantial POI churn, user turnover, category drift, and decay in static POI memory, motivating the study of continual dual cold-start POI recommendation. To address this setting, we propose EviRec, a continual evidence-learning framework that estimates how much historical evidence should be trusted separately for each candidate POI. EviRec scores each visible candidate from three complementary views: a matching view based on the user's recent mobility profile, a transition-memory view that captures repeated mobility routines, and a lifecycle view that reflects candidate maturity. Because a near-zero transition score may indicate either irrelevance or insufficient observation, EviRec qualifies the evidence usin
    
[^57]: ZeroHAT：行为条件化的零样本人类活动轨迹生成

    ZeroHAT: Behavior-Conditioned Zero-Shot Human Activity Trace Generation

    [https://arxiv.org/abs/2609.20310](https://arxiv.org/abs/2609.20310)

    ZeroHAT提出了一种行为条件化的零样本框架，通过从拥有真实数据的源区域迁移人类行为模式，并结合目标区域公开可获得的上下文信息，为缺乏真实人类活动轨迹数据的区域生成合成数据。

    

    人类活动轨迹记录了个体对兴趣点的带时间戳的访问，对于出行预测和城市模拟等应用至关重要。然而，由于高昂的采集成本和隐私问题，获取大规模人类活动轨迹十分困难。合成人类活动轨迹生成为使此类数据可用提供了一条有前景的途径，并吸引了工业界和学术界日益增长的关注。尽管针对这一主题已有许多研究，但其中大多数依赖某一区域的真实数据来为同一区域生成合成数据，这对于许多无法获得真实人类活动轨迹的区域来说是不可行的。为填补这一空白，我们提出了ZeroHAT，这是一个行为条件化的框架，它以零样本的方式为目标区域生成合成人类活动轨迹，通过从源区域的真实人类活动轨迹中学习并迁移行为模式，并利用目标区域公开可获得的上下文信息对其进行适配。

    arXiv:2609.20310v1 Announce Type: new  Abstract: Human activity traces record individuals' timestamped visits to points of interest and are essential for applications such as mobility prediction and urban simulation. However, accessing large-scale HATs is challenging due to high collection costs and privacy concerns. Synthetic HAT generation offers a promising way to make such data available and has attracted growing interest from both industry and academia. Although many efforts have been devoted to this topic, most of them rely on real data from a region to generate synthetic data for the same region, which is infeasible for the many regions where real HATs are unavailable. To fill this gap, we propose ZeroHAT, a behavior-conditioned framework that generates synthetic HATs for a target region in a zero-shot manner by transferring behavioral patterns learned from real HATs in source regions and adapting them with publicly available contextual information about the target region. ZeroH
    
[^58]: 用于偏微分方程学习的超网络参数化空间自适应神经算子

    Hypernetwork-Parameterized Spatially Adaptive Neural Operators for PDE Learning

    [https://arxiv.org/abs/2609.20309](https://arxiv.org/abs/2609.20309)

    提出空间自适应神经算子SANO，利用坐标条件超网络为每个空间位置生成独立的算子参数，解决了传统神经算子在边界和高梯度区域欠拟合导致误差累积的问题。

    

    空间非均质偏微分方程（PDE）由于几何形状和物理系数的变化而表现出依赖于位置的动力学特性。现有的神经算子通过多尺度特征、注意力机制或区域分解来改进局部建模，但其更新规则通常仍然是空间共享的。基于超网络的方法可以在不同PDE实例之间自适应调整参数，但通常每个实例只生成一个全局参数化。因此，共享算子可能在边界和高梯度区域出现欠拟合，这些局部误差会在自回归滚动预测过程中不断累积。我们提出了一种空间自适应神经算子（SANO），它用空间连续的、依赖于位置的算子参数场取代了空间共享的参数化。SANO使用傅里叶编码坐标和坐标条件超网络来在空间各位置生成算子调节编码。

    arXiv:2609.20309v1 Announce Type: new  Abstract: Spatially heterogeneous partial differential equations (PDEs) exhibit location-dependent dynamics arising from variations in geometry and physical coefficients. Existing neural operators improve localized modeling through multiscale features, attention mechanisms, or domain decomposition, yet their update rules often remain spatially shared. Hypernetwork-based methods adapt parameters across PDE instances but typically generate only one global parameterization per instance. Consequently, shared operators may underfit boundaries and high-gradient regions, with these localized errors accumulating during autoregressive rollout. We propose a spatially adaptive neural operator (SANO), which replaces this spatially shared parameterization with a spatially continuous field of location-dependent operator parameters. SANO uses Fourier-encoded coordinates and a coordinate-conditioned hypernetwork to generate spatial operator-conditioning codes at 
    
[^59]: SAGG：面向异质损坏下鲁棒多模态学习的样本自适应梯度门控

    SAGG: Sample-Adaptive Gradient Gating for Robust Multimodal Learning under Heterogeneous Corruption

    [https://arxiv.org/abs/2609.20302](https://arxiv.org/abs/2609.20302)

    该论文揭示了多模态学习中样本级异质损坏会使共享参数的梯度调制方法产生不可约偏差，并证明样本级二元门控是唯一无偏策略，据此提出SAGG方法，通过在线特征范数质量测试实现样本自适应的梯度门控与方差截断，达到标准O(1/√T)收敛速率。

    

    多模态梯度平衡方法使用每个模态共享的标量来调节编码器梯度，这隐含地假设损坏在整个训练批次中是均匀分布的。然而在实际中，损坏是样本异质的：在单个小批量中，不同样本可能有不同的模态被损坏。我们证明，在这种异质损坏模型下，任何具有共享调制参数的批次级、样本无关的线性估计器，相对于干净数据的梯度都会产生不可约的偏差，并且样本级的“全有或全无”门控是自然无分布估计器类中唯一无偏的策略。受这一结果启发，我们提出了样本自适应梯度门控（SAGG），它通过在线特征范数质量测试对每个样本做出二元的保留或丢弃决策，并引入截断机制以控制方差。我们证明基于SAGG的SGD以标准的O(1/√T)（速率收敛）。

    arXiv:2609.20302v1 Announce Type: new  Abstract: Multimodal gradient balancing methods modulate encoder gradients with a shared scalar per modality, implicitly assuming that corruption is uniform across the training batch. In practice, corruption is sample-heterogeneous: within a single mini-batch, different samples may have different modalities corrupted. We prove that under this heterogeneous corruption model, any batch-level sample-agnostic linear estimator with a shared modulation parameter incurs an irreducible bias with respect to the clean-data gradient, and that sample-level all-or-nothing gating is the unique unbiased strategy within a natural distribution-free estimator class. Motivated by this result, we propose Sample-Adaptive Gradient Gating (SAGG), which makes a binary retain-or-discard decision per sample via an online feature-norm quality test and incorporates a truncation mechanism for variance control. We prove that SAGG-based SGD converges at the standard O(1/sqrt(T)
    
[^60]: 通过边界感知数据增强提升离线强化学习的泛化性与鲁棒性

    Improving Generalization and Robustness in Offline Reinforcement Learning via Boundary-Aware Data Augmentation

    [https://arxiv.org/abs/2609.20300](https://arxiv.org/abs/2609.20300)

    该论文从理论上揭示了随机片段插值训练的误差与状态间距离呈正相关，并据此提出边界感知数据增强方法，以提升离线强化学习在分布内的泛化能力和鲁棒性。

    

    当前的离线强化学习（ORL）算法往往容易过拟合训练数据集，在部署到真实环境时表现出较差的分布内泛化能力和鲁棒性，从而削弱了其有效性。现有方法通常借助计算机视觉领域广泛使用的正则化技术来提升分布内泛化能力和鲁棒性。然而，由于低层物理信号对分布偏移的高度敏感性，这些方法在分布内泛化和鲁棒性方面仍存在明显局限，难以在复杂环境中实现稳定性能。为解决这一问题，我们从理论上分析了通过随机片段插值训练的行为策略和动作值函数的误差界，揭示了误差大小与状态间距离呈正相关。基于这一洞察，我们提出了一种（摘要内容在此处截断）……

    arXiv:2609.20300v1 Announce Type: new  Abstract: Current offline reinforcement learning (ORL) algorithms tend to overfit the training dataset and exhibit poor in-distribution generalization and robustness performance when deployed to real environments, thus compromising their effectiveness. Existing methods typically enhance in-distribution generalization and robustness by leveraging regularization techniques widely used in computer vision. However, due to the high sensitivity of low-level physical signals to distributional shifts, these methods still suffer from notable limitations in in-distribution generalization and robustness, making it difficult to achieve stable performance in complex environments. To address this issue, we theoretically analyze the error bounds of the behavior policy and action-value function trained with random episode interpolation, revealing that the error scales positively correlated with the distance between states. Based on this insight, we propose a meth
    
[^61]: 具有指定不动点的阈值布尔网络学习算法

    A Learning Algorithm for Threshold Boolean Networks with Prescribed Fixed Points

    [https://arxiv.org/abs/2609.20298](https://arxiv.org/abs/2609.20298)

    该论文提出了一种基于自定义可微损失函数的学习算法，能够从指定的不动点集合中准确推断阈值布尔网络，在拟南芥基因调控模型上实现了无虚假吸引子的完美重建，性能显著优于感知机和逻辑回归等标准方法。

    

    我们提出了一种用于推断具有指定不动点集合的阈值布尔网络（TBN）的学习算法。该方法采用一个自定义的可微损失函数，联合实现不动点的保持、惩罚虚假吸引子、鼓励二值输出，并通过L1正则化促进网络稀疏性。将该算法应用于拟南芥的FOS-GRN模型，在30次独立运行中有5次实现了完美重建（即所有10个期望的不动点均被恢复且没有虚假不动点），平均恢复了8.53 ± 0.90个正确的不动点且没有产生虚假吸引子。相比之下，感知机和逻辑回归等标准方法虽然最多能恢复10个不动点，但会引入8到31个虚假不动点。通过改变稀疏系数（λ）的补充分析证实，该方法的性能以及推断网络的结构特性均保持稳健。

    arXiv:2609.20298v1 Announce Type: new  Abstract: We present a learning algorithm for inferring threshold Boolean networks (TBNs) with a prescribed set of fixed points. The proposed method employs a custom differentiable loss function that jointly enforces fixed point preservation, penalizes spurious attractors, encourages binary outputs, and promotes sparsity through L1 regularization. Applied to the FOS-GRN model of Arabidopsis thaliana, the approach achieved perfect reconstruction (i.e., all 10 desired fixed points and no spurious ones) in 5 out of 30 independent runs, recovering on average 8.53 $\pm$ 0.90 correct fixed points with no spurious attractors. In contrast, standard methods such as the Perceptron and Logistic Regression recovered up to 10 fixed points but introduced between 8 and 31 spurious ones. An additional analysis varying the sparsity coefficient ($\lambda$) confirmed that the method's performance and the structural properties of the inferred networks remain robust w
    
[^62]: 表面肌电图手势解码中从健全者到截肢者的迁移：训练数据来源与校准预算

    Intact-to-Amputee Transfer in Surface-EMG Gesture Decoding: Training Source and Calibration Budget

    [https://arxiv.org/abs/2609.20297](https://arxiv.org/abs/2609.20297)

    研究发现表面肌电手势解码的零样本跨人群迁移完全失败，但基于四十名健全受试者训练的跨用户编码器在仅需三次标注校准的情况下，对截肢者的解码性能显著优于传统单用户分类器（宏F1达0.779对0.589），且健全者训练数据比少量截肢者数据产生更好的迁移效果。

    

    在一个人身上训练的识别器很少能直接迁移到另一个人身上，而达到实用性能通常需要最终用户进行新一轮的标注校准。一项涵盖1077项研究的系统综述量化了证据薄弱之处：截肢者仅出现在约六分之一的研究中。本研究将一种与电极排列无关的跨用户编码器应用于十一名经桡骨截肢者，并采用与其健全肢体训练数据相匹配的实验协议。结果表明，零样本跨人群迁移完全失败：编码器必须获得新用户的标注数据后才能开始解码，而在此之后，仅需三次重复标注，它对队列中每位受试者的表现均比临床为单个用户拟合的分类器高出0.190宏F1。在仅有三次标注重复的情况下，该方法达到0.779的宏F1，而传统单用户流程仅为0.589。此外，用四十名健全受试者训练比用十名其他截肢者训练能产生更好的向新截肢者的迁移效果，而两者结合则产生更好的迁移效果……

    arXiv:2609.20297v1 Announce Type: new  Abstract: A recogniser trained on one person rarely transfers to the next, and useful performance usually demands a fresh round of labelled calibration from the end user. A systematic review of 1077 studies quantifies where the evidence is thin: amputees appear in about one in six. Here a montage-agnostic cross-user encoder is carried to eleven transradial amputees on a protocol matched to its intact-limb training data. Zero-shot cross-population transfer fails outright: the encoder requires labeled data from the new user before it begins decoding, and it then exceeds the per-user classifier a clinic would fit by 0.190 macro F1 at three repetitions and for every subject in the cohort. Given three labelled repetitions it reaches 0.779 macro-F1 against 0.589 for the per-user pipeline. Training on forty intact subjects produces better transfers to a new amputee than training on ten other amputees, and combining the two produces better transfers than 
    
[^63]: 小校准预算下个性化跨用户表面肌电编码器

    Personalising a Cross-User Surface Electromyography Encoder Under a Small Calibration Budget

    [https://arxiv.org/abs/2609.20296](https://arxiv.org/abs/2609.20296)

    本研究将个性化跨用户表面肌电编码器视为有代价的设计决策，在77名受试者上比较了四种利用少量校准重复数据的方法，发现全微调在所有预算水平下都始终最为准确。

    

    肌电接口在正常工作之前需要来自用户的校准。以往的工作将校准视为一种数量指标，但并未提出这样的问题：设备在收集完校准重复数据后应该用它们做什么。本文将从跨用户编码器进行个性化这一过程视为一个有代价的设计决策。针对每个留出的受试者，从同一个跨用户编码器出发，测试了四种使用完全相同标记重复数据的替代方法。原型自适应、线性探针、缩放微调和全微调在每个预算水平上均进行了测试，直至各数据库允许的最大值——DB1上五次重复，DB2和DB5上四次重复。在77名受试者上比较四种花费小额校准预算的方式后发现，全微调在每一个预算水平上都是最准确的，其一致性之高以至于在任何受试者子集中都没有例外。这一结果对人们如何做出相关工程决策具有影响。

    arXiv:2609.20296v1 Announce Type: new  Abstract: A myoelectric interface needs calibration from the user before it will function. Earlier work has treated calibration as a quantity, but has not asked the question of what a device should do with the calibration repetitions once they have been collected. This paper views personalizing the cross-user encoder as a design decision with a cost. Four alternative approaches to using exactly the same labeled repetitions were tested from a single cross-user encoder per held-out subject. Prototypical adaptation, linear probes, scaled fine-tuning and full fine-tuning were tested at every budget up to the maximum each database allows, five repetitions on DB1 and four on DB2 and DB5. Comparing four ways to spend a small calibration budget across 77 subjects, full fine-tuning is the most accurate at every budget, consistently enough that there is no exception among subsets of subjects. The result which impacts how one might make an engineering decisi
    
[^64]: TinyCNN：一个用于设备端植物病害检测的19.3万参数网络，附带跨数据集鲁棒性诊断

    TinyCNN: A 193K-Parameter Network for On-Device Plant Disease Detection, with a Cross-Dataset Robustness Diagnosis

    [https://arxiv.org/abs/2609.20290](https://arxiv.org/abs/2609.20290)

    本文提出仅含19.3万参数的轻量级卷积网络TinyCNN，通过深度可分离卷积在PlantVillage数据集上实现98.88%的准确率，规模比ResNet18小约58倍，可在低成本移动设备上直接部署进行植物病害检测。

    

    早期检测作物病害是联合国可持续发展目标2（零饥饿）下可持续农业与粮食安全的核心，在资源受限地区尤为迫切——这些地区专家诊断稀缺，但低成本移动设备却十分普及。本文提出了TinyCNN，一个用于设备端植物病害分类的轻量级卷积神经网络。TinyCNN采用深度可分离卷积模块，仅包含193,190个可训练参数，对于224x224的输入图像仅需110.05M次乘加运算。在38类的PlantVillage基准数据集上，TinyCNN达到了98.88%的测试准确率和98.03%的宏F1分数，同时其规模比ResNet18小约58倍，比MobileNetV2教师模型小11.8倍，直接降低了推理的能耗、内存和成本占用，符合绿色AI原则。本文进一步分析了普通知识蒸馏作为一种可持续的模型压缩策略。

    arXiv:2609.20290v1 Announce Type: cross  Abstract: Detecting crop disease early is central to sustainable agriculture and food security under United Nations Sustainable Development Goal 2 (Zero Hunger), and is especially urgent in resource-constrained regions where expert diagnosis is scarce but low-cost mobile devices are widespread. This paper presents TinyCNN, a lightweight convolutional neural network for on-device plant disease classification. TinyCNN uses depthwise separable convolution blocks and contains only 193,190 trainable parameters with 110.05M MACs for a 224x224 input image. On the 38-class PlantVillage benchmark, TinyCNN achieves 98.88% test accuracy and 98.03% macro-F1 while being approximately 58x smaller than ResNet18 and 11.8x smaller than a MobileNetV2 teacher, directly reducing the energy, memory, and cost footprint of inference in line with Green AI principles. The paper further analyzes vanilla knowledge distillation as a sustainable model-compression strategy; 
    
[^65]: 用于文本、知识图谱和超图原生Transformer建模的标记关联结构

    Labeled Incidence Structures for Native Transformer Modeling of Text, Knowledge Graphs, and Hypergraphs

    [https://arxiv.org/abs/2609.20278](https://arxiv.org/abs/2609.20278)

    本文提出标记关联结构（LIS），将文本、知识图谱和超图统一编码为（内容、槽位、关系实例）三元组表示，使单个标准Transformer无需展平数据即可原生处理这三种异构数据类型。

    

    文本、知识图谱和超图都包含在关系实例中扮演不同角色的元素，而当数据被展平为token序列时，这种结构信息就会丢失。我们引入标记关联结构，这是一种统一表示，将每个端点编码为$(x_d, s, e)$三元组：内容$x_d$、角色或槽位$s$、以及该角色所在的关系实例$e$。由于每种数据类型都无需展平即可映射到相同的$(x_d, s, e)$表示，单个标准transformer便可以原生处理所有这些数据类型，结构差异完全由算子承载，而非由架构承载。LIS通过组合槽位算子和实例算子$A(s,e) = R_s R_e$为每个端点分配一个结构地址。我们刻画了该分解何时能为每个token提供唯一的、路径无关的地址。当满足这一条件时，将端点$j$与端点$i$进行比较的自然算子是相对传输$P_（注：原摘要在此处截断）

    arXiv:2609.20278v1 Announce Type: cross  Abstract: Text, knowledge graphs, and hypergraphs all have elements that play distinct roles within relation instances, structure that is lost when data is flattened into token sequences. We introduce labeled incidence structures (LIS), a uniform representation that encodes each endpoint as $(x_d, s, e)$: content $x_d$, a role or slot $s$, and the relation instance $e$ in which that role appears. Because every data type maps to the same $(x_d, s, e)$ representation without flattening, a single standard transformer can process them all natively, structural differences are carried entirely by the operators, not the architecture.   LIS assigns a structural address to each endpoint by composing a slot operator and an instance operator, $A(s,e) = R_s R_e$. We characterize when this factorization gives every token a unique, path-independent address. When it does, the natural operator comparing endpoint $j$ to endpoint $i$ is the relative transport $P_
    
[^66]: 面向锥形记忆化网格的无表索引：有序参数函数的紧凑核外求值

    A Table-Free Index for Tapered Memoization Grids: Compact Out-of-Core Evaluation of Functions of Sorted Arguments

    [https://arxiv.org/abs/2609.20276](https://arxiv.org/abs/2609.20276)

    该论文证明锥形记忆化网格的键集恰为多重集组合，因而可基于经典组合数系统实现无表闭式 O(d) 排名与逆排名索引，消除了原方案的预处理表，支持并行构建与无键存储，形成仅存值的扁平数组结构，在 3740 万条目规模下比哈希映射记忆化节省 5.7 倍内存。

    

    许多应用需要反复评估一个作用于排序得分向量的昂贵函数 f，该向量的影响随排名衰减，例如 Plackett-Luce 选择概率、alpha-entmax 注意力阈值以及按排名加权的聚合。Biswas 和 Regan（TCS 2015）提出了一种锥形网格来对这类函数进行记忆化，并通过预计算的节点计数表进行索引。我们首先明确指出，锥形网格的键集恰好就是多重集组合的集合，因此其索引即为经典的组合数系统：这带来了一个无需表格的闭式 O(d) 排名计算，消除了原始方案中 O(Bd)-O(Bd²) 的预处理表，将其推广到不再受限于固定的第一坐标，并提供了此前缺失的 O(d) 逆排名计算，从而支持无顺序约束的并行构建和无键存储。所得到的结构是一个仅存储值的扁平数组：在 N=37.4M 条目规模下，其内存占用比哈希映射记忆化少 5.7 倍，并能高效响应查询（摘要在此处截断）……

    arXiv:2609.20276v1 Announce Type: cross  Abstract: Many applications must repeatedly evaluate an expensive function f of a sorted score vector whose influence decays with rank: Plackett-Luce choice probabilities, alpha-entmax attention thresholds, and rank-weighted aggregates. Biswas and Regan (TCS 2015) introduced a tapered grid that memoizes such functions, indexed through precomputed node-count tables. We first make explicit that the tapered grid's key set is exactly the set of multiset combinations, so its index is the classical combinatorial number system: this yields a table-free closed-form O(d) rank that eliminates the O(Bd)-O(Bd^2) preprocessing tables of the original scheme, generalizes it beyond a pinned first coordinate, and supplies the previously missing O(d) unranking, which enables order-free parallel construction and key-free storage. The resulting structure is a values-only flat array: at N=37.4M entries it occupies 5.7x less memory than a hash-map memo and answers qu
    
[^67]: 基于学习优化图神经网络的智慧城市NR-V2X网络中AI驱动的实时中继优化

    AI-Driven Real-Time Relay Optimisation in Smart Urban NR-V2X Networks via Learning-to-Optimise Graph Neural Networks

    [https://arxiv.org/abs/2609.20271](https://arxiv.org/abs/2609.20271)

    本文提出一种基于图神经网络的AI驱动学习优化框架，利用离线MILP最优中继决策监督训练边感知GINE网络，从而在NR-V2X城市车联网中实现接近最优的实时多跳中继选择。

    

    可靠且低时延的通信是由NR-V2X网络支撑的智慧城市服务和工业4.0应用的基本需求。然而，路侧单元（RSU）部署有限以及复杂的城市传播环境，往往导致网联自动驾驶车辆（CAV）难以保持稳定的连接。本文提出了一种基于图神经网络（GNN）的AI驱动学习优化框架，用于NR-V2X系统中的实时多跳中继选择。该车辆网络被建模为图结构，其中节点表示CAV和RSU，边则编码无线链路特性。通过离线的混合整数线性规划（MILP）建模提供最优中继决策，作为监督信号训练具备边特征的图同构网络（GINE）。在真实城市数据集上的大量实验表明，所提出的方法实现了接近最优的连接性能。

    arXiv:2609.20271v1 Announce Type: new  Abstract: Reliable and low-latency communication is a fundamental requirement for smart city services and Industry 4.0 applications enabled by NR-V2X networks. However, limited Road-Side Unit (RSU) deployment and complex urban propagation conditions often prevent Connected and Automated Vehicles (CAVs) from maintaining stable connectivity. This paper proposes an AI-driven Learning-to-Optimise (L2O) framework based on Graph Neural Networks (GNNs) for real-time multi-hop relay selection in NR-V2X systems. The vehicular network is modelled as a graph, where nodes represent CAVs and RSUs, and edges encode radio-link characteristics. An offline Mixed-Integer Linear Programming (MILP) formulation provides optimal relay decisions used as supervision for training an edge-aware Graph Isomorphism Network with Edge Features (GINE). Extensive experiments on realistic urban datasets demonstrate that the proposed approach achieves near-optimal connectivity perf
    
[^68]: 放置是免费的，组合则不然：拉丁方作为异构序列混合器堆栈的可证明均衡构建方法

    Placement Is Free, Composition Is Not: The Latin Square as a Provably-Balanced Construction for Heterogeneous Sequence-Mixer Stacks

    [https://arxiv.org/abs/2609.20269](https://arxiv.org/abs/2609.20269)

    本文提出用拉丁方排列异构序列混合器的构建方法，通过让每种机制在每行每列恰好出现一次，实现跨深度均衡暴露并消除放置位置的混淆因素，从而将机制选择与放置的影响解耦。

    

    自GPT以来，大多数Transformer在每个层都重复使用相同的注意力机制。然而这种设计在很大程度上是一种惯例而非经过检验的结论。当多种序列混合器组合在一个堆栈中时，性能提升可能来自机制选择、放置位置或两者兼有，这使得因果归因变得困难。我们介绍了Aether-7B-5Attn，一个65.9亿参数的混合专家模型（约29.8亿激活参数），其49层包含七种序列混合机制，排列成7×7的拉丁方。由于每种机制在每一行和每一列中恰好出现一次，该设计保证了跨深度的均衡暴露，同时消除了放置带来的混淆因素。为了评估这一原则，我们构建了一个参数匹配的代理模型，将四种机制排列为十六层上的4×4拉丁方，匹配至7.009亿参数，每组使用八个随机种子进行训练。结果揭示出明显的分离现象。重新排列分布（摘要在此处截断）

    arXiv:2609.20269v1 Announce Type: new  Abstract: Since GPT, most Transformers have repeated the same attention mechanism at every layer. Yet this design is largely a convention rather than a tested conclusion. When multiple sequence mixers are combined in one stack, improvements may arise from mechanism choice, placement, or both, making causal attribution difficult. We introduce Aether-7B-5Attn, a 6.59B-parameter mixture-of-experts model ($\approx$2.98B active) whose 49 layers contain seven sequence-mixing mechanisms arranged as a $7\times7$ Latin square. Because each mechanism appears exactly once in every row and column, the design guarantees balanced exposure across depth while eliminating placement confounds. To evaluate this principle, we build a parameter-matched proxy with four mechanisms arranged as a $4\times4$ Latin square over sixteen layers, matched to 700.9M parameters and trained with eight seeds per arm. The results reveal a clear dissociation. Rearranging a distributed
    
[^69]: 通过双向行为先验蒸馏改进在线强化学习

    Improving Online Reinforcement Learning via Bidirectional Behavior Prior Distillation

    [https://arxiv.org/abs/2609.20268](https://arxiv.org/abs/2609.20268)

    提出双向行为先验蒸馏算法，利用动作值先验引导在线策略学习，无需依赖高质量离线数据集和专家轨迹即可解决贪婪策略更新导致的评论家估计误差问题，从而提升在线强化学习的样本效率和学习稳定性。

    

    在线强化学习算法经常表现出较差的样本效率和不稳定的学习动态，这源于系统性的评论家估计误差，而贪婪的策略更新进一步加剧了这些误差。现有的行为先验强化学习方法试图通过依赖离线预训练从固定数据集中学习行为模型，并使用策略先验来约束在线策略更新，从而缓解这一问题。然而，离线数据集的质量有限，往往阻碍了提供能够有效指导策略更新的高价值策略的能力。专家轨迹的缺失严重损害了在线策略学习，导致样本效率低下和性能次优。为了应对这些挑战，我们摒弃了传统的行为先验方法，提出了双向行为先验蒸馏（B2PD）算法。B2PD利用动作值先验来引导条件……（摘要内容在此处被截断）

    arXiv:2609.20268v1 Announce Type: new  Abstract: Online reinforcement learning (RL) algorithms frequently exhibit poor sample efficiency and unstable learning dynamics, stemming from systematic critic estimation errors that are exacerbated by greedy policy updates. Existing behavior-prior reinforcement learning methods attempt to alleviate this issue by relying on offline pre-training to learn behavior models from fixed datasets and using policy priors to constrain online policy updates. However, the limited quality of offline datasets often hinders the ability to provide high-value policies that can effectively guide policy updates. The absence of expert trajectories significantly impairs online policy learning, leading to low sample efficiency and suboptimal performance. To address these challenges, we depart from conventional behavior prior approaches and propose a Bidirectional Behavior Prior Distillation (B2PD) algorithm. B2PD leverages action-value priors to guide a conditional v
    
[^70]: 反例与充分条件：对《最优传输广义矩方法》一文的评论

    Counterexamples and Sufficient Conditions: Comments on "Optimally-Transported Generalized Method of Moments"

    [https://arxiv.org/abs/2609.20260](https://arxiv.org/abs/2609.20260)

    本文对Schennach & Starck提出的最优传输广义矩方法（OTGMM）估计量的核心定理给出反例，证明其定理2-6在原假设条件下不成立，并探讨使这些结论成立的充分条件。

    

    我们针对Schennach & Starck（2026a）提出的最优传输广义矩方法（OTGMM）估计量进行评论，并在其所述假设条件下给出针对定理2-6的反例。首先，小误差分析中所使用的假设不足以保证定理2的一致性和定理3的渐近正态性。其次，我们考察大误差分析，其中定理4声称OTGMM估计量等价于带修正矩的GMM估计量。我们证明，在标量模型中，定理4所选择的值不同于唯一的OTGMM最小化点，且违反了OTGMM样本矩约束。在一个满足定理5和定理6假设的过度识别模型中，拉格朗日乘子的第一个分量在OTGMM估计量和带修正矩的GMM估计量下具有不同的概率极限。在模型误设的情形下，OTGMM所选择的总体值取决于……（原文摘要至此截断）

    arXiv:2609.20260v1 Announce Type: cross  Abstract: We comment on the optimally-transported generalized method of moments (OTGMM) estimator proposed by Schennach & Starck (2026a) and give counterexamples to Theorems 2-6 under their stated assumptions. First, the assumptions used in the small-error analysis are insufficient for consistency in Theorem 2 and asymptotic normality in Theorem 3. Next, we consider the large-error analysis, in which Theorem 4 states that the OTGMM estimator is equivalent to a GMM estimator with modified moments. We show that in a scalar model, Theorem 4 selects a value that differs from the unique OTGMM minimizer and violates the OTGMM sample moment restriction. In an overidentified model satisfying the assumptions used in Theorems 5 and 6, the first component of the Lagrange multiplier has different probability limits under the OTGMM estimator and the GMM estimator with modified moments. Under misspecification, the population value selected by OTGMM depends on
    
[^71]: 小于3B参数的开源语言模型在8 GB消费级GPU上进行零样本作文评分能走多远？

    How Far Can Sub-3B Open Language Models Go in Zero-Shot Essay Scoring on an 8 GB Consumer GPU?

    [https://arxiv.org/abs/2609.20250](https://arxiv.org/abs/2609.20250)

    在严格隐私约束下，小于3B参数的开源语言模型即可在单张8 GB消费级GPU上实现零样本作文评分，且评分细则分解提示在批处理最小-最大聚合下全面优于整体提示。

    

    基于大语言模型的零样本作文评分通常以专有API模型进行演示，然而最需要自动评分的场景——例如公立学校在严格隐私规则下批改数千篇作文——往往恰恰是无法将学生作文发送给第三方API的场景。我们探究了当模型必须是完全在本地以FP16运行的小于3B参数的开源模型时，其能力还能保留多少。我们对来自两个家族的四个指令微调模型（Qwen2.5的0.5B/1.5B/3B版本以及SmolLM2的1.7B版本）在单张8 GB消费级GPU上，针对全部八个ASAP-AES提示开展了受控研究，采用自助法置信区间、经Holm校正的配对检验以及贴近实际部署情况的关键设计选择变体。研究得出三项发现：(i) 在批处理最小-最大聚合方式下，评分细则分解提示对所有模型都优于整体提示（尽管Qwen2.5-3B在一个提示上出现显著下降），而在均值聚合方式下……（摘要内容在此处截断）

    arXiv:2609.20250v1 Announce Type: new  Abstract: Zero-shot essay scoring with large language models is usually demonstrated with proprietary API models, yet the settings where automated scoring is most needed, such as public schools grading thousands of essays under strict privacy rules, are often those where sending student writing to a third-party API is unacceptable. We ask how much capability survives when the model must be a sub-3B open model running fully locally in FP16, with a controlled study of four instruction-tuned models from two families (Qwen2.5 at 0.5B/1.5B/3B, SmolLM2 at 1.7B) on all eight ASAP-AES prompts on a single 8 GB consumer GPU, with bootstrap confidence intervals, Holm-corrected paired tests, and deployment-realistic variants of the key design choices. Three findings emerge. (i) Rubric-decomposed prompting beats holistic prompting for every model under batch min-max aggregation (though Qwen2.5-3B drops significantly on one prompt), and under mean aggregation t
    
[^72]: 准确率是不够的：深度知识追踪中人口统计偏差的跨架构审计

    Accuracy Is Not Enough: A Cross-Architecture Audit of Demographic Bias in Deep Knowledge Tracing

    [https://arxiv.org/abs/2609.20249](https://arxiv.org/abs/2609.20249)

    该研究首次对四种深度知识追踪架构在三种训练方案下进行了人口统计偏差的跨架构审计，发现偏差确实存在但高度依赖数据集情境，且仅凭准确率无法保障模型公平性。

    

    深度知识追踪（DKT）模型隐式地决定了自适应系统认为哪些学生已经掌握了某项技能，然而关于其人口统计公平性的几乎所有证据都来自贝叶斯知识追踪；驱动现代系统的深度模型尚未接受过类似的跨架构审计。我们填补了这一空白：在两个包含人口统计元数据的公开数据集（Eedi，1590万次交互；OULAD，预处理后16.7万条数据）上，对四种架构（DKT、DKVMN、SAKT、AKT）在三种训练方案（标准、重加权、对抗性）下进行训练，并采用ABROCA、学生级自助法置信区间和置换检验进行评估，以回应当前对公平性指标不稳定性的批评。研究得出三项发现：（i）偏差确实存在但依赖于具体情境：每种架构在Eedi数据集上都显示出显著的社会经济ABROCA（0.018-0.023，p<0.005），经济弱势学生的分组AUC更低，而性别偏差同样显著……（摘要截断）

    arXiv:2609.20249v1 Announce Type: new  Abstract: Deep knowledge tracing (DKT) models implicitly decide which students an adaptive system believes have mastered a skill, yet almost all evidence on their demographic fairness comes from Bayesian knowledge tracing; the deep models that power modern systems have received no comparable cross-architecture audit. We close this gap: four architectures (DKT, DKVMN, SAKT, AKT) trained under three regimes (standard, reweighting, adversarial) on two public datasets with demographic metadata, Eedi (15.9M interactions) and OULAD (167k after preprocessing), evaluated with ABROCA, student-level bootstrap confidence intervals, and permutation tests addressing recent critiques of fairness-metric instability. Three findings emerge. (i) Bias is real but context-dependent: every architecture shows a significant socioeconomic ABROCA on Eedi (0.018-0.023, $p<0.005$), with per-group AUC lower for economically disadvantaged students, while gender bias is signif
    
[^73]: 在大语言模型时代，训练经典模型还值得吗？基于表格数据的交叉点基准测试

    Is It Still Worth Training a Classical Model in the Era of LLMs? A Crossover Benchmark on Tabular Data

    [https://arxiv.org/abs/2609.20218](https://arxiv.org/abs/2609.20218)

    该研究提出“标注数据交叉点 N*”这一指标，量化在表格数据预测任务中经典模型需要多少训练数据才能超越免训练的大语言模型，并发现经典模型经过少量数据训练后便能快速胜出。

    

    大语言模型可以通过纯英文描述直接对表格数据的一行进行标注，而无需任何训练——这一能力目前已集成到主流电子表格工具中，例如 Microsoft Copilot in Excel 和 Anthropic 的 Claude for Excel。这对于许多标签获取成本高昂的商业预测问题提出了一个实际问题：你应该直接提示一个冻结的大语言模型，还是收集数据并训练一个模型——如果是后者，需要多少数据？我们用“标注数据交叉点 N*”来量化这个答案，即训练的经典模型的学习曲线超越冻结大语言模型免训练（因此恒定不变）误差时所需的训练集大小。通过汇总 126 次独立的学生评估（针对小型 GPT 模型，涵盖 8 种提示配置、18 个表格数据集），并结合六个经典模型家族的权威幂律学习曲线，我们发现训练很快就能获胜：即使为其提供“神谕”式的最佳提示配置选择，训练的经典模

    arXiv:2609.20218v1 Announce Type: cross  Abstract: Large language models can label a tabular row from a plain-English description with no training - a capability now shipping in mainstream spreadsheet tools such as Microsoft Copilot in Excel and Anthropic's Claude for Excel - raising a practical question for the many business prediction problems where labels are expensive: should you prompt a frozen LLM, or collect data and train a model - and if so, how much data? We quantify the answer with the labeled-data crossover N*, the training-set size at which a trained classical model's learning curve overtakes a frozen LLM's training-free (and therefore flat) error. Aggregating 126 independent student evaluations of small GPT models under eight prompting configurations across 18 tabular datasets, paired with authoritative power-law learning curves for six classical model families, we find that training wins fast: even given an oracle choice of its best prompt configuration, a trained classi
    
[^74]: 使用门控图注意力网络解释短期交通预测模型中的空间信息流

    Explaining spatial information flow in short-term traffic forecasting models using a gated graph attention network

    [https://arxiv.org/abs/2609.20217](https://arxiv.org/abs/2609.20217)

    本文提出在图注意力网络层中加入可学习的门控机制，以量化每个传感器的更新状态来自邻居信息的比例，并通过对门控的正则化逐步撤除邻居信息，从而以分级消融的方式解释和验证短期交通预测模型中空间信息流的必要性。

    

    短期交通预测支持路网的实时监控与控制，而图注意力网络（GAT）是在这些模型中表示空间依赖关系的标准手段。GAT层被广泛描述为能够捕获相邻位置的影响，但这一点很少得到验证，因为作为支持证据的注意力权重无法与任何可测量量进行比较。这留下了两个悬而未决的问题：模型应当如何解释，以及模型的哪些组件是必要的。我们通过在GAT层中添加一个门控来解决这一问题，该门控在每个传感器和每个时间步上学习传感器的更新状态中有多少比例来自其邻居而非自身。对门控进行正则化可以逐步撤除邻居信息，从而提供一种分级的消融形式。我们将门控GAT应用于ST-MetaNet，其编码器和解码器各自在两个循环层之间放置一个GAT层。

    arXiv:2609.20217v1 Announce Type: new  Abstract: Short-term traffic forecasting supports real-time monitoring and control of road networks, and graph attention networks (GAT) are the standard means of representing spatial dependence in these models. GAT layers are widely described as capturing the influence of neighbouring locations, but this is seldom verified, because the attention weights offered in support cannot be compared against any measured quantity. That leaves two questions open, how the model should be explained and which of its components are necessary. We address this by adding a gate to the GAT layer which learns, at every sensor and every time step, what share of a sensor's updated state is drawn from its neighbours rather than from itself. Regularising the gate withdraws neighbour information progressively and thereby provides a graded form of ablation. We apply the gated GAT to ST-MetaNet, whose encoder and decoder each place one GAT layer between two recurrent layers
    
[^75]: 基于具有领域感知特征编码的高效仿真驱动变分量子分类器的变压器故障诊断

    Transformer fault diagnosis using an efficient simulation-driven variational quantum classifier with domain-aware feature encoding

    [https://arxiv.org/abs/2609.20214](https://arxiv.org/abs/2609.20214)

    本文提出一种仿真驱动的变分量子分类器框架，通过融合基于Duval几何的领域感知特征编码、混合ZX-YY量子特征映射以及双量子比特全纠缠EfficientSU2拟设，在浅层参数化电路中以极少量量子比特实现了基于溶解气体分析的变压器早期故障诊断。

    

    早期变压器故障诊断面临着非线性溶解气体交互作用、故障特征相互重叠以及标注数据有限等挑战，而实际部署还进一步要求系统在现实计算约束下具备可靠的性能。本文提出了一种面向基于溶解气体分析（DGA）的变压器故障诊断的仿真驱动建模框架，其中采用精心设计的变分量子分类器（VQC）作为计算核心，并通过仿真进行系统性分析。该框架将源自Duval几何的领域感知特征建模与轻量级双量子比特量子表示相结合，使非线性气体交互效应能够在浅层参数化电路中被捕获。文中设计了一种混合ZX-YY量子特征映射来建模非对易特征之间的交互作用，同时采用全纠缠EfficientSU2拟设在严格的（计算资源）约束下提供了足够的表达能力……

    arXiv:2609.20214v1 Announce Type: cross  Abstract: Early transformer fault diagnosis is challenged by nonlinear dissolved-gas interactions, overlapping fault signatures, and limited labeled data, while practical deployment further requires reliable performance under realistic computational constraints. This paper presents a simulation-driven modeling framework for dissolved gas analysis-based transformer fault diagnosis, in which a carefully engineered variational quantum classifier (VQC) is employed as the computational core and systematically analyzed through simulation. The framework integrates domain-aware feature modeling derived from Duval geometry with a lightweight two-qubit quantum representation, enabling nonlinear gas-interaction effects to be captured within a shallow parameterized circuit. A hybrid ZX-YY quantum feature map is designed to model non-commuting feature interactions, while a full-entanglement EfficientSU2 ansatz provides adequate expressive capacity under stri
    
[^76]: 子域感知的预训练图像嵌入表示压缩

    Subdomain-aware representation compression for pretrained image embeddings

    [https://arxiv.org/abs/2609.20213](https://arxiv.org/abs/2609.20213)

    本文将PCA和LDA等降维技术应用于预训练图像嵌入的子域表示压缩，在降低空间与计算复杂度的同时反而提升了准确率，并验证了压缩表示的迁移学习能力。

    

    摘要：降维是一种广为人知的提高空间效率的技术，通常在整个数据集上统一应用。本文研究了利用降维技术进行子域表示压缩的可能性。我们在图像领域探索了主成分分析（PCA）和线性判别分析（LDA）等标准技术。结果不仅展示了在空间和计算复杂度方面的预期改进——这对边缘设备机器学习应用至关重要——还显示出相对于直接使用全嵌入流程的准确率提升。一种可能的解释是降维有效地提取了子域特征。我们还进行了实验，以展示使用压缩表示的迁移学习能力。

    arXiv:2609.20213v1 Announce Type: new  Abstract: Dimensionality reduction is a well-known technique for improving space efficiency, typically applied uniformly across an entire dataset. This paper investigates the possibilities of using dimensionality reduction techniques for subdomain representation compression. We explore standard techniques such as Principal Component Analysis (PCA) and Linear discriminant analysis (LDA) in image domains. The results not only demonstrate the expected improvements in space and computation complexity crucial for edge-device ML applications but also show improvements in accuracy over direct full-embedding procedure. One possible explanation is that dimensionality reduction effectively extracts subdomain features. We also performed experiments to demonstrate transfer learning capabilities using the compressed representations.
    
[^77]: 面向城市蜂窝活动预测的场景条件关系路由

    Scene-Conditioned Relation Routing for urban cellular activity forecasting

    [https://arxiv.org/abs/2609.20209](https://arxiv.org/abs/2609.20209)

    SCRR-Net提出一种场景条件下的空间关系路由框架，利用城市上下文信息联合控制空间依赖选择与跨任务知识迁移，在短信、网络流量和通话活动预测任务上均优于现有方法并具备可解释性。

    

    城市蜂窝活动预测需要联合建模异构的时空信号，包括短信使用量、移动网络流量和通话活动。现有方法通常将时间建模、空间关系学习和多信号预测相互分离，依赖于固定的图结构或静态的多任务学习方案，这限制了它们对不断变化的城市场景的适应能力。我们提出了SCRR-Net，这是一个场景条件下的空间关系路由框架，其中城市上下文信息联合控制空间依赖选择和跨任务知识迁移。SCRR-Net包含一个上下文编码器、一个空间图专家路由模块、一个时间Transformer编码器和一个任务知识路由模块。在Milano和Trento数据集上的实验表明，SCRR-Net在短信、网络流量和通话活动预测方面持续优于竞争方法，同时提供了可解释的路由机制。

    arXiv:2609.20209v1 Announce Type: cross  Abstract: Urban cellular activity forecasting requires jointly modeling heterogeneous spatiotemporal signals, including SMS usage, mobile network traffic, and call activity. Existing methods often separate temporal modeling, spatial relation learning, and multi-signal prediction, relying on fixed graph structures or static multi-task learning schemes, which limits their adaptability to changing urban scenes. We propose SCRR-Net, a scene-conditioned spatial relation routing framework in which urban contextual information jointly controls spatial dependency selection and cross-task knowledge transfer. SCRR-Net includes a context encoder, a spatial graph expert routing module, a temporal Transformer encoder, and a task knowledge routing module. Experiments on the Milano and Trento datasets demonstrate that SCRR-Net consistently outperforms competing methods on SMS, network traffic, and call activity forecasting, while providing interpretable routin
    
[^78]: 面向认知工作负荷评估的统一模态无关多模态框架

    Towards a Unified Modality-Agnostic Multimodal Framework for Cognitive Workload Assessment

    [https://arxiv.org/abs/2609.20199](https://arxiv.org/abs/2609.20199)

    本文提出了一种统一的、模态无关的层级Transformer架构，可在单一模型内处理异构生物信号模态，并系统评估了ECG、EDA、RESP、SpO₂和EEG五种模态全部31种组合在认知工作负荷评估中的效果。

    

    认知工作负荷反映了任务执行过程中所需的心理努力程度，是自适应人机系统设计的核心。利用生物信号测量认知工作负荷已得到广泛研究和记录；然而，关于组合异构生物信号模态用于这一目的的效果研究仍然有限。为了为该领域提供见解，我们开发了一个统一的、模态无关的、基于层级Transformer的架构，能够在单一模型内处理异构生物信号模态。我们在一项试点研究中使用该框架，评估了五种模态——心电图（ECG）、皮肤电活动（EDA）、呼吸（RESP）、外周血氧饱和度（SpO₂）和脑电图（EEG）——所有31种可能的组合，采用留一被试验证方法，涵盖三项认知上不同的任务：抽象推理（IQ）、算术问题求解（MATH）以及一个游戏……

    arXiv:2609.20199v1 Announce Type: new  Abstract: Cognitive workload reflects the mental effort required during task performance and is central to the design of adaptive human-machine systems. The use of biosignals to measure cognitive workload has been extensively researched and documented; however, studies examining the effects of combining heterogeneous biosignal modalities for this purpose remain limited. To provide insight into this area, we developed a unified, modality-agnostic, hierarchical Transformer-based architecture to process heterogeneous biosignal modalities within a single model. We use this framework in a pilot study evaluating all $31$ possible combinations of five modalities: Electrocardiogram (ECG), Electrodermal Activity (EDA), Respiration (RESP), Peripheral Oxygen Saturation (SpO$_2$), and Electroencephalogram (EEG), under leave-one-subject-out validation across three cognitively distinct tasks: abstract reasoning (IQ), arithmetic problem solving (MATH), and a gam
    
[^79]: 人工智能时代的金融情绪评估

    Evaluating Financial Sentiment in the Age of AI

    [https://arxiv.org/abs/2609.20198](https://arxiv.org/abs/2609.20198)

    本文评估了十二种金融情绪模型，发现通用大语言模型无需微调即可达到与金融专用模型相当的分类准确率，但更高的准确率并未带来更强的经济关联——各模型情绪度量仅与盈利意外显著相关，而均无法显著预测次日股票收益。

    

    金融情绪度量在实证金融研究中被广泛使用，但通用大语言模型（LLMs）是否优于现有的金融专用方法仍不清楚。本文采用语言有效性和经济有效性两项评估标准，对十二种情绪模型进行了评估，包括基于词典的方法、金融专用的transformer模型以及开源大语言模型。我们发现，通用大语言模型无需针对特定任务进行微调，即可达到与金融专用transformer模型相当的分类性能。然而，更高的分类准确率并不能转化为更强的经济关联性。若干模型产生的情绪度量与盈利意外显著相关，但没有一个模型与次日股票收益显著相关。模型在盈利大幅超预期或大幅不及预期的公告上表现最佳，而在盈利变动较为温和的公告上表现则明显较弱。

    arXiv:2609.20198v1 Announce Type: new  Abstract: Financial sentiment measures are widely used in empirical finance, but it remains unclear whether general-purpose large language models (LLMs) improve on existing finance-specific methods. This paper evaluates twelve sentiment models, including dictionary-based methods, finance-specific transformers, and open-source LLMs, using two criteria: linguistic validity and economic validity. We find that general-purpose LLMs achieve classification performance comparable to finance-specific transformer models without task-specific fine-tuning. However, higher classification accuracy does not translate into stronger economic relationships. Several models produce sentiment measures that are significantly associated with earnings surprises, but none is significantly associated with next-day stock returns. Model performance is strongest for announcements with large earnings beats or misses and substantially weaker for announcements with more moderate
    
[^80]: SoftTri：面向自适应模糊推理系统的平滑三角隶属函数

    SoftTri: Smooth Triangular Membership Functions for Adaptive Fuzzy Inference Systems

    [https://arxiv.org/abs/2609.20194](https://arxiv.org/abs/2609.20194)

    提出 SoftTri——一种受 Swish 激活函数启发的可微平滑三角隶属函数，在保持经典三角隶属函数几何结构与局部性的同时实现无穷阶光滑，使神经模糊系统具备高效的端到端梯度优化能力。

    

    三角隶属函数（MFs）因其可解释性强、参数化复杂度低以及良好的局部性，被广泛应用于模糊系统中。然而，其在节点点处的固有不可微性限制了基于梯度的优化方法在自适应神经模糊架构中的有效性，通常需要采用次梯度近似或启发式平滑技术。在本文中，我们提出了 SoftTri，这是一种可微的三角隶属函数，它采用受 Swish 类激活函数启发的平滑软铰链机制构建。所提出的公式在保持经典三角隶属函数的几何结构和局部化特性的同时，对于任意有限的锐度参数 β>0，均可提供关于输入变量和隶属参数（a,b,c）的 C∞ 光滑性。我们推导了闭式解析梯度，以实现高效且完全可微的反向传播。

    arXiv:2609.20194v1 Announce Type: cross  Abstract: Triangular membership functions (MFs) are widely used in fuzzy systems because of their interpretability, low parameterization complexity, and strong locality properties. However, their inherent nondifferentiability at knot points limits the effectiveness of gradient-based optimization in adaptive neuro-fuzzy architectures, often necessitating subgradient approximations or heuristic smoothing techniques. In this paper, we propose \emph{SoftTri}, a differentiable triangular membership function constructed using a smooth soft-hinge mechanism inspired by Swish-type activations. The proposed formulation preserves the geometric structure and localized behavior of classical triangular MFs while providing $C^\infty$ smoothness with respect to both the input variable and the membership parameters $(a,b,c)$ for any finite sharpness parameter $\beta>0$. Closed-form analytical gradients are derived to enable efficient and fully differentiable bac
    
[^81]: 检索何时有助于时间序列预测？

    When Does Retrieval Help Time-Series Forecasting?

    [https://arxiv.org/abs/2609.20193](https://arxiv.org/abs/2609.20193)

    该研究证明检索插件在时间序列预测中的收益并非源于其机制本身，而是取决于回看窗口长度与主导季节周期的关系，在该关系合适时简单的周期重复基线即可击败标准模型和检索插件。

    

    检索插件为深度预测器提供其回看窗口无法承载的信息。已发表的评估报告显示一致的性能提升，且各自将功劳归于自身的机制。我们证明这种收益实际上取决于运行点：即窗口长度S与主导季节周期L之间的关系，这是标准评估协议从未改变的一个维度。按该关系对评估进行分层即可揭示这一规律。在S=12时，一个简单地重复最后观测周期的简单对照方法，在七个基准中的四个上以8%到44%的MSE优势击败了六个标准骨干模型（总体而言）。它在ETTm1上击败了我们运行的最强检索插件，并在ECL上与其持平。而在三个训练集频谱缺乏集中共享周期的数据集上，它最多落后25%。对预测期、周期和窗口进行的受控合成扫描显示，收益边界与周期相关（相关性+0.71），而非与预测期相关（-0.23）。一项成对对照……

    arXiv:2609.20193v1 Announce Type: new  Abstract: Retrieval plug-ins supply a deep forecaster with information its lookback window cannot carry. Published evaluations report consistent gains, and each credits its own mechanism. We show that the benefit belongs instead to the operating point: the relation between window length $S$ and dominant seasonal period $L$, an axis the standard protocol never varies. Stratifying the evaluation by that relation exposes the regime. At $S{=}12$, a simple control that repeats the last observed period beats the six standard backbones, in aggregate, on four of seven benchmarks by $8\%$ to $44\%$ of MSE. It beats the strongest plug-in we run on ETTm1 and matches it on ECL. It is worse by up to $25\%$ on the three datasets whose training-split spectra lack a concentrated, shared period. A controlled synthetic sweep of horizon, period, and window shows the benefit boundary tracks the period (correlation $+0.71$), not the horizon ($-0.23$). A paired control
    
[^82]: 即时视觉语言导航：面向空中机器人的机载视觉语言导航堆栈

    VLN on the Fly: An Onboard Vision-Language Navigation Stack for Aerial Robots

    [https://arxiv.org/abs/2609.20191](https://arxiv.org/abs/2609.20191)

    该论文提出了一种将基础定位、规划和控制保持为独立可检查阶段的机载视觉语言导航堆栈，在四旋翼飞行器上实现了15次试验中13次成功到达目标，平均误差仅5.72厘米。

    

    在空中机器人上完全机载运行视觉语言导航非常困难，因为基础定位、规划和控制必须共享有限的计算资源，且飞行中难以隔离单阶段的错误。端到端的空中策略将这些阶段融合为一个网络，牺牲了模块化堆栈所能保持的可观测性和安全检查。我们提出了VLN on the Fly，一个将基础定位、规划和控制保持为独立、可检查阶段的机载堆栈。量化视觉语言模型（VLM）将指令定位到粗略的图像单元，深度信息将其提升为3D目标，快速B样条规划器生成可行轨迹，预训练的强化学习策略将其跟踪并转换为电机指令，适用于四旋翼飞行器。在受控室内空间中对三个日常指代物进行的15次机载飞行中，该堆栈在15次试验中有13次成功到达目标，平均目标误差为5.72厘米，平均GPU利用率为39.3%。在另外6次杂乱环境试验中，

    arXiv:2609.20191v1 Announce Type: cross  Abstract: Running vision-language navigation fully onboard an aerial robot is hard, since grounding, planning, and control must share limited compute and a single-stage error is difficult to isolate in flight. End-to-end aerial policies fuse these stages into one network, giving up the observability and safety checks a modular stack keeps available. We propose VLN on the Fly, an onboard stack that keeps grounding, planning, and control as separate, inspectable stages. A quantized VLM grounds an instruction to a coarse image cell, depth lifts it to a 3D goal, a fast B-spline planner returns a feasible trajectory, and a pretrained reinforcement learning policy tracks it to motor commands across quadrotors. Across 15 onboard flights over three everyday referents in a controlled indoor volume, the stack reaches the target in 13 of 15 trials with 5.72 cm mean goal error and 39.3% average GPU utilization. In 6 additional cluttered-environment trials, 
    
[^83]: 序列上下文契合度预测跨领域的人类行为与神经动态

    Sequential Contextual Fit Predicts Human Behavioural and Neural Dynamics Across Domains

    [https://arxiv.org/abs/2609.20179](https://arxiv.org/abs/2609.20179)

    本研究提出序列上下文契合度（SCF）这一通用嵌入度量指标，证明其在语言、情绪、决策和神经数据等多个领域中均能有效预测人类行为与神经动态，且其预测能力独立于惊讶度和预测误差等已有预测因子。

    

    人类的感知、行动和决策都是以序列方式展开的，但现有的计算预测指标往往局限于特定领域。本研究计算并检验了序列上下文契合度（SCF），这是一种基于嵌入的度量指标，用于衡量当前信息状态与其近期上下文的匹配程度。该指标采用简单的近因加权相似度核函数，可应用于词语、声音、视觉场景、情感状态、选择、行动以及神经表征。在语言处理、音乐诱发情绪、部分视听情绪脑电（EEG）数据、赌博决策、人类活动识别以及决策相关脑电数据等多个领域，较低的上下文契合度预测了更长的处理时间、更大的情绪或行为转变以及更强的神经状态变化。在控制了惊讶度、强化学习预测误差、声学变化、视觉变化和传感器变化等已有预测因子之后，这些效应依然存在。因此，SCF提供了一个……（摘要原文在此处截断）

    arXiv:2609.20179v1 Announce Type: new  Abstract: Human perception, action and decision making unfold in sequences, but computational predictors are often domain-specific. This study computes and tests sequential contextual fit (SCF), an embedding-based measure of how well a current information state matches its recent context. The metric uses a simple recency-weighted similarity kernel and can be applied to words, sounds, visual scenes, affective states, choices, actions and neural representations. Across language processing, music-evoked emotion, a subset of audiovisual emotion EEG data, gambling decisions, human activity recognition and decision-related EEG, lower contextual fit predicted longer processing times, larger affective or behavioural transitions and stronger neural-state changes. These effects remained after controlling for established predictors including surprisal, reinforcement-learning prediction error, acoustic change, visual change and sensor change. SCF therefore pr
    
[^84]: PaGNet：一种面向面板的GBDT-神经网络混合模型，用于多目标企业避税代理指标预测

    PaGNet: A Panel-Aware GBDT--Neural Network for Multi-Target Corporate Tax Avoidance Proxy Forecasting

    [https://arxiv.org/abs/2609.20177](https://arxiv.org/abs/2609.20177)

    本文提出面向面板的GBDT-神经网络混合模型PaGNet，通过LightGBM与Panel-MLP双分支结构及逐目标混合器，在韩国上市公司面板数据上实现多目标企业避税代理指标预测，并提供透明的分支依赖性诊断。

    

    基于企业-年度面板数据预测企业避税代理指标极具挑战性，因为预测信号分散在较短的企业历史和相关目标之中，而以筛选为导向的应用需要模型行为具有可解释性。我们提出PaGNet（面向面板的GBDT-神经网络），这是一个双分支混合模型，结合了使用面板-时间摘要的LightGBM分支，以及采用注意力池化时间聚合和共享主干多任务学习的Panel-MLP分支。一个逐目标的验证最优混合器在无需可训练融合参数的情况下，同时生成最终预测和简洁的分支依赖性诊断。在涵盖2011年至2024年共1,754家韩国上市公司的KoTaP面板上，PaGNet在无数据泄漏、共享超参数的协议下，于四种特征机制中进行了评估。在排除直接代理指标滞后项的FS1机制和增强税收历史的FS2机制中，应计类目标（TSTA、TSDA）稳定地路由至LightGBM分支。

    arXiv:2609.20177v1 Announce Type: new  Abstract: Forecasting corporate tax avoidance proxies from firm--year panel data is challenging because predictive signals are distributed across short firm histories and related targets, while screening-oriented use requires transparent model behavior. We propose PaGNet (Panel-Aware GBDT--Neural Network), a two-branch hybrid that combines a LightGBM branch using panel-temporal summaries with a Panel-MLP branch using attention-pooled temporal aggregation and shared-trunk multi-task learning. A per-target validation-optimal blender produces both the final prediction and a compact branch-reliance diagnostic without trainable fusion parameters. On the KoTaP panel of 1{,}754 Korean listed firms from 2011--2024, PaGNet is evaluated under a leakage-free, shared-hyperparameter protocol across four feature regimes. In the direct-proxy-lag-excluded FS1 regime and the tax-history-augmented FS2 regime, accrual targets (TSTA, TSDA) route stably to the LightGB
    
[^85]: 几乎无需通信的鲁棒联邦Q学习

    Robust Federated Q-Learning with Almost No Communication

    [https://arxiv.org/abs/2609.20174](https://arxiv.org/abs/2609.20174)

    本文提出了鲁棒联邦Q学习算法 Robust Fed-Q，即使存在少量对抗性智能体，也能在仅需极少通信轮次的条件下实现最优值函数的精确收敛，并获得接近最优的协作式样本复杂度加速。

    

    我们考虑一个涉及 $M$ 个智能体的联邦强化学习场景，所有智能体都与同一个马尔可夫决策过程（MDP）进行交互。智能体通过中央服务器交换信息以学习最优值函数。我们的目标是理解，当其中一小部分智能体是对抗性的且可以任意行动时，在这种设置下能在多大程度上通过协作获得样本复杂度的加速。为此，我们提出了 Robust Fed-Q，这是一种联邦Q学习算法，它融合了基于模型的强化学习和无模型强化学习的思想，并结合了鲁棒统计学中的中位数均值方法。我们证明，尽管存在数据污染，Robust Fed-Q 仍能以高概率 (i) 保证在无限样本的极限下精确收敛到最优值函数，并且 (ii) 享有得益于协作的接近最优的有限时间收敛速率。此外，我们的方法只需要 $\tilde{O}(1)$ 轮与中央服务器的通信。

    arXiv:2609.20174v1 Announce Type: new  Abstract: We consider a federated reinforcement learning setting involving $M$ agents, all of whom interact with a common Markov Decision Process (MDP). The agents exchange information via a central server to learn the optimal value function. Our goal is to understand to what extent one can hope for collaborative sample-complexity speedups in such a setting, when a small fraction of the agents are adversarial and can act arbitrarily. To that end, we propose Robust Fed-Q}, a federated Q-learning algorithm that blends ideas from both model-based and model-free RL, along with the median-of-means device from robust statistics. We prove that despite corruption, with high-probability, Robust Fed-Q (i) guarantees exact convergence to the optimal value function in the limit of infinite samples, and (ii) enjoys near-optimal finite-time rates that benefit from collaboration. In addition, our approach requires just $\tilde{O}(1)$ rounds of communication to a
    
[^86]: 支持度阈值而非算法限制了共购买网络中稀有关联的恢复

    Support Thresholds, Not Algorithms, Limit Rare-Association Recovery in Co-Purchase Networks

    [https://arxiv.org/abs/2609.20171](https://arxiv.org/abs/2609.20171)

    该研究发现限制共购买网络中稀有高价值关联恢复的关键因素是支持度阈值的选择而非算法本身，基于网络的过滤方法（如噪声校正法和top-K法）能恢复80-100%的稀有高提升度关联，远超Apriori算法的22-28%。

    

    Apriori算法的支持度阈值在市场购物篮分析中涉及一个权衡：高阈值可以捕捉频繁出现的关联，而低阈值则会导致生成大量的规则。本文在两个食品杂货数据集上比较了五种共购买边过滤方法，即Instacart（320万个购物篮）和Dunnhumby（20.8万个购物篮），包括Apriori、Apriori加提升度后过滤、基于提升度的top-K排名，以及两种基于网络的方法：噪声校正（NC）和差异过滤器（DF）。top-K方法确保了最大的平均提升度，而NC方法通过单一显著性参数（α）实现了类似的提升度水平。这两种方法比Apriori恢复了多得多的稀有高提升度关联（80-100%对22-28%）。NC和top-K选择了具有显著差异的边（18-29%不重叠）：NC保留了统计上显

    arXiv:2609.20171v1 Announce Type: new  Abstract: The support threshold of the Apriori algorithm involves a trade-off in conducting market basket analysis: the associations that occur frequently are noted with high threshold; however, the low ones lead to generating the large amount of rules. The paper compares five methods for co-purchase edge filtration on two grocery datasets: i.e., Instacart (3.2 million baskets) and Dunnhumby (208 thousand baskets), including Apriori, Apriori + lift post-filtering, top-$K$ ranking based on lift, and two methods based on networks, noise-corrected (NC) and disparity filter (DF). The top-$K$ method ensures the maximum average lift, while the NC achieves similar lift level by means of a single value of the significance parameter ($\alpha$). These two methods recover substantially more rare high-lift associations than Apriori (80-100% against 22-28%). NC and top-$K$ select meaningfully different edges (18-29% non-overlapping): NC retains statistically v
    
[^87]: 小到足以知晓一切：作为延迟泛化科学仪器的全可枚举Transformer

    Small Enough to Know Everything: The Fully-Enumerable Transformer as an Instrument for the Science of Delayed Generalization

    [https://arxiv.org/abs/2609.20166](https://arxiv.org/abs/2609.20166)

    本文提出将全可枚举任务上的微型Transformer作为研究延迟泛化（grokking）现象的科学仪器，其提供精确泛化上限、任务手术、全权重直接观察和生存时间统计四种独特能力，并通过预注册守恒研究验证小规模下发现的规律具有可迁移性。

    

    在完全可枚举任务上训练的微型Transformer在grokking（顿悟）现象研究中占据着一个特殊的位置：每个输入都可以被评估，每个泛化上限都可以被精确计算，数百个随机种子的实验只需几分钟。我们认为这一研究机制是一种科学仪器，具备四种近似设置无法提供的能力：(a) 精确且可证伪的泛化上限；(b) 任务手术，即可在可证明固定其他所有变量的同时操纵单一结构变量；(c) 直接观察每一个权重；(d) 对大量随机种子进行生存时间统计，从而将“不发生grokking”重新表述为删失观测。显而易见的质疑是：在10^4参数量级下刻画的规律在更大规模上可能毫无意义。我们通过一项预注册的守恒研究来回应这一质疑：在12K参数下确立的三条任务侧规律——可恢复性上限定律、角色冲突延迟定律和权重衰减响应定律——在一个理（摘要原文截断）

    arXiv:2609.20166v1 Announce Type: new  Abstract: Tiny transformers trained on fully-enumerable tasks occupy an unusual position in the study of grokking: every input can be evaluated, every generalization ceiling can be computed exactly, and hundreds of seeds cost minutes. We argue this regime is a scientific instrument with four capabilities that approximate settings cannot offer: (a) exact, falsifiable generalization ceilings; (b) task surgery that manipulates one structural variable while provably fixing all others; (c) direct observation of every weight; and (d) survival-time statistics over many seeds that recast "does not grok" as a censored observation. The obvious objection is that laws characterized at 10^4 parameters may not mean anything beyond them. We answer it with a preregistered conservation study: three task-side laws established at 12K parameters -- a recoverability-ceiling law, a role-conflict delay law, and a weight-decay response law -- are re-measured under an ide
    
[^88]: 低轨卫星物联网：架构、技术与在轨验证

    LEO Satellite Internet of Things: Architecture, Technology, and On-Orbit Verification

    [https://arxiv.org/abs/2609.20165](https://arxiv.org/abs/2609.20165)

    本文提出了融合组成与功能视角的6G低轨卫星物联网二维系统架构，评估了大规模免授权随机接入、深度学习多波束预编码和分布式协作路由三项关键使能技术，并通过在轨验证平台证实了其真实可行性。

    

    低地球轨道（LEO）卫星星座有望成为第六代（6G）物联网（IoT）的基石，提供真正意义上的全球覆盖和无处不在的连接。本文提出了一个面向6G低轨卫星物联网的整体性二维系统架构，该架构融合了组成视角和功能视角，以促进低轨卫星与地面网络的无缝集成。基于这一架构，我们评估了针对上行链路、下行链路和星间链路（ISL）的三项关键使能技术。具体而言，我们分析了用于实现高效上行连接的大规模免授权随机接入技术，研究了基于深度学习的多波束预编码以实现稳健的下行传输，并考察了用于弹性星间链路数据传输的分布式协作路由技术。此外，我们提出了一个在轨验证平台，验证了这些技术在真实环境中的可行性和性能。

    arXiv:2609.20165v1 Announce Type: cross  Abstract: Low Earth orbit (LEO) satellite constellations are poised to become a cornerstone of the sixth-generation (6G) Internet of Things (IoT), providing truly global coverage and ubiquitous connectivity. This article presents a holistic two-dimensional system architecture for 6G LEO satellite IoT that incorporates composition and functional perspectives to facilitate the seamless integration of LEO satellites and terrestrial networks. Building upon this architecture, we evaluate three pivotal enabling technologies targeting the uplink, downlink, and inter-satellite links (ISLs). Specifically, we analyze massive grant-free random access for efficient uplink connectivity, investigate deep learning-based multibeam precoding for robust downlink transmission, and examine distributed cooperative routing for resilient ISL data delivery. Furthermore, we present an on-orbit verification platform that validates the real-world feasibility and performan
    
[^89]: 无回放持续学习中的噪声最优值：隔离、机制与适用范围

    A Noise Optimum in Rehearsal-Free Continual Learning: Isolation, Mechanism, and Scope

    [https://arxiv.org/abs/2609.20162](https://arxiv.org/abs/2609.20162)

    本文发现在无回放持续学习中，向巩固规则注入适当强度的相干噪声可使记忆保持呈倒U形最优效应，其关键在于将锚定增益与注入噪声方差相耦合这一单行规则。

    

    向巩固规则中注入随机噪声可以将网络对早期任务的记忆保持能力提升至一个最优水平，超过该水平后则会使其下降——即记忆保持与噪声之间呈倒U形关系。本文完全在仿真环境中隔离出产生这一最优值的因素，并描绘了其适用范围。(1) 现象：记忆保持的倒U形曲线出现在多个相关任务的持续学习基准上（Split-MNIST、FashionMNIST、持续Yin-Yang）。(2) 隔离：量级匹配的对照实验表明，该效应需要朝向已巩固权重的相干恢复力——相同量级的随机方向力不会产生最优值，而朝向错误目标的相干力则会主动造成损害。(3) 有效成分：将锚定增益与注入噪声方差 sigma^2 相耦合即可恢复大部分最优效应——这是一行代码即可实现的规则，而Ornstein-Uhlenbeck自适应（固定增益）和MESU（后验方差增益）均未实现这一点。强制的Ornstein-Uhl

    arXiv:2609.20162v1 Announce Type: new  Abstract: Injecting stochastic noise into a consolidation rule can improve a network's retention of earlier tasks up to an optimal level, then degrade it -- an inverted-U in retention vs. noise. This paper isolates what produces that optimum and maps where it holds, entirely in simulation. (1) Phenomenon: the retention inverted-U appears on several related-task continual-learning benchmarks (Split-MNIST, FashionMNIST, continual Yin-Yang). (2) Isolation: a magnitude-matched ladder shows the effect requires coherent restoring toward the consolidated weights -- a random-direction force of identical magnitude produces no optimum, and a coherent force toward the wrong target actively hurts. (3) Active ingredient: most of the optimum is recovered by coupling the anchor gain to the injected-noise variance sigma^2 -- a one-line rule that neither Ornstein-Uhlenbeck Adaptation (fixed gain) nor MESU (posterior-variance gain) implements. A forced Ornstein-Uhl
    
[^90]: 深度学习中的特殊拉格朗日锥

    Special Lagrangian cones in Deep Learning

    [https://arxiv.org/abs/2609.20159](https://arxiv.org/abs/2609.20159)

    本文提出了Harvey-Lawson锥的矩阵推广形式，证明其为精确特殊拉格朗日流形，并表明它属于一族对深度学习中平衡流形进行叶状分解的精确特殊拉格朗日流形。

    

    我们引入了Harvey和Lawson锥的一个矩阵推广形式，并证明它是一个精确特殊拉格朗日流形。我们进一步证明它属于一族精确特殊拉格朗日流形，该族流形对深度学习中出现的平衡流形进行叶状结构分解。

    arXiv:2609.20159v1 Announce Type: cross  Abstract: We introduce a matrix generalization of the cone of Harvey and Lawson and prove that it is an exact special Lagrangian manifold. We further show that it belongs to a family of exact special Lagrangian manifolds that foliate the balanced manifold arising in deep learning.
    
[^91]: QUALS：通过模式量化与可学习性同步实现通用预测的语料库均衡

    QUALS: Corpus Equilibrium for Universal Forecasting via Pattern Quantization and Learnability Synchronization

    [https://arxiv.org/abs/2609.20156](https://arxiv.org/abs/2609.20156)

    提出了QUALS大规模时间序列语料库均衡框架，通过模式量化与可学习性同步两大机制管理复杂数据分布，显著提升数据效率，使现有模型仅用一小部分训练数据即可实现更优的零样本预测性能。

    

    无处不在的跨领域时间序列数据为交通系统和电网等领域的关键应用提供了支撑。近年来，在大规模数据集上训练基础模型以实现准确的零样本预测已成为重要的研究热点。然而，当前研究主要侧重于架构创新，而对数据多样性的关注不足，往往依赖简单的数据采样策略，无法有效管理复杂的数据分布，导致训练数据利用效率低下、性能欠佳。为解决这一问题，我们提出了QUALS，一个大规模时间序列语料库均衡框架。QUALS显著提升了数据效率，即让现有模型仅使用原始训练数据的一小部分即可取得更优性能。具体而言，QUALS通过两个核心机制运行。首先，一个模式量化框架……

    arXiv:2609.20156v1 Announce Type: cross  Abstract: Ubiquitous time series data across diverse domains enables critical applications in areas such as transportation systems and power grids. Recently, training foundation models on massive datasets to achieve accurate zero-shot forecasting has emerged as a major research focus. However, current studies predominantly prioritize architectural innovations while insufficiently addressing data diversity, often relying on simple data sampling strategies that fail to manage complex data distributions effectively, leading to inefficient use of training data and suboptimal performance. To address this, we propose QUALS, a large-scale time series corpus equilibrium framework. QUALS significantly enhances data efficiency, i.e., enabling existing models to achieve superior performance using only a small fraction of the original training data. Specifically, QUALS operates through two core mechanisms. First, a pattern quantization framework systematica
    
[^92]: 面向任务的低信噪比信道多任务卫星遥感语义特征传输

    Task-Oriented Semantic Feature Transmission for Multi-Task Satellite Remote Sensing over Low-SNR Channels

    [https://arxiv.org/abs/2609.20150](https://arxiv.org/abs/2609.20150)

    本文提出一种面向任务的语义特征传输框架，绕过图像重建、直接传输多任务预训练骨干网络提取的语义特征，并通过信道适配模块压缩特征与特征恢复器修复信道损伤，在低信噪比信道下的卫星遥感分类与检测任务中始终优于传统重建导向的JSCC方法。

    

    传统的卫星遥感传输遵循“先重建后推理”的范式，优化像素级保真度，这与分类和检测等下游任务的目标不匹配，尤其是在低信噪比条件下。本文研究了一种面向任务的框架，该框架绕过图像重建，直接传输由多任务预训练骨干网络提取的语义特征。轻量级信道适配模块（CAM）压缩特征维度以降低带宽占用，特征恢复器则在信道损伤后恢复与任务相关的结构。在骨干网络冻结的情况下，CAM和特定任务的下游头在随机信噪比训练下通过任务级和特征级监督进行联合优化。在所采用的加性高斯白噪声（AWGN）设置下，场景分类和目标检测的实验表明，在不同信噪比条件下，该方法始终优于面向重建的联合信源信道编码（JSCC）基线。

    arXiv:2609.20150v1 Announce Type: cross  Abstract: Conventional satellite remote sensing transmission follows a reconstruct-then-infer paradigm that optimizes pixel-level fidelity, creating an objective mismatch with downstream tasks such as classification and detection, especially at low SNR. This paper investigates a task-oriented framework that bypasses image reconstruction and directly transmits semantic features extracted by a multitask-pretrained backbone. A lightweight channel adaptation module (CAM) compresses feature dimensionality for bandwidth reduction, and a feature restorer recovers task-relevant structure after channel corruption. With the backbone frozen, the CAM and task-specific downstream heads are jointly optimized with task and feature-level supervision under random-SNR training. Under the adopted AWGN setting, experiments on scene classification and object detection show consistent gains over reconstruction-oriented JSCC baselines across different SNR conditions, 
    
[^93]: 线性时变系统的快变固有频率与阻尼比辨识

    Fast-varying Natural Frequencies and Damping Ratio Identification for Linear Time-Varying System

    [https://arxiv.org/abs/2609.20138](https://arxiv.org/abs/2609.20138)

    该论文提出了一种将长短期记忆网络与扩展卡尔曼滤波器相结合的物理增强机器学习方法，用于辨识线性时变系统在风浪载荷下快变固有频率与阻尼比，并通过海上风力机仿真数据验证了其有效性。

    

    这项工作提出了一种物理增强的机器学习方法，用于线性时变（LTV）系统在时变运行条件下——即快变固有频率和阻尼比情形——的系统辨识，该方法将长短期记忆网络（LSTM）与扩展卡尔曼滤波器（EKF）相结合。所提出的方法利用振动数据（位移和速度测量值）、模态阻尼比的领域知识，以及一个能够给出近似固有频率时变模型的基于物理的模型。该方法使用由双叶片海上风力机有限元模型在真实环境和运行条件下生成的合成数据进行验证。该系统由于运行条件的影响而表现出快速时变的频率，且由于风浪载荷的作用，其辨识尤为困难。论文还评估了所提方法在假设条件下的鲁棒性。

    arXiv:2609.20138v1 Announce Type: new  Abstract: This work proposes a physics-enhanced machine learning approach for the system identification of Linear Time-Varying (LTV) systems under time-varying operating conditions in terms of fast-varying natural frequencies and damping ratios by combining a long short-term memory network with an Extended Kalman Filter (EKF). The proposed approach uses vibration data (displacement and velocity measurements), domain knowledge of modal damping ratios, and a physics-based model that can yield an approximate natural frequencies time-dependency model. The approach is validated using synthetic data generated from a finite element model of a 2-blade offshore wind turbine under realistic environmental and operating conditions. This system displays fast time-varying frequencies due to operating conditions, whose identification is particularly challenging because of the wind and wave loading. The robustness of the proposed approach is assessed under assume
    
[^94]: 局部稀疏性实现无监督的大语言模型安全检测

    Local Sparsity Enables Unsupervised LLM Safety Detection

    [https://arxiv.org/abs/2609.20129](https://arxiv.org/abs/2609.20129)

    本文利用稀疏自编码器概念空间中的局部稀疏性这一关键洞察，提出了一个无需不安全训练数据的无监督LLM安全异常检测框架，并有理论支撑。

    

    大语言模型（LLM）的部署时安全方法主要是有监督的，并假设能够获取不安全的训练数据。然而，新的攻击和伤害类别不断出现，而以这种有监督方式训练的模型无法捕获这些内容。另一种方法是从异常检测的视角来看待这个问题，即仅依赖于对安全数据的建模并标记分布外输入。然而，LLM激活位于高维空间中，这引发了关于异常检测在统计上是否可行的担忧。我们证明，在线性表示假设（LRH）下，确实存在希望。在通常通过稀疏自编码器（SAE）恢复的LRH概念空间中，邻近的点共享一个较小的共同激活支持集。利用这一局部稀疏性洞察，我们提出了一个基于局部掩码SAE的异常检测框架，并提供了理论依据的支持。我们进行了验证……

    arXiv:2609.20129v1 Announce Type: cross  Abstract: Deployment-time safety methods for large language models (LLMs) are predominantly supervised and assume access to unsafe training data. Nevertheless, new attacks and harm categories regularly arise, not captured by models trained in such a supervised fashion. An alternative approach is to view this problem through the lens of anomaly detection, namely, to rely solely on modeling safe data and flagging out-of-distribution inputs. However, LLM activations lie in a high-dimensional space, raising concerns about whether anomaly detection is statistically feasible. We show that, under the linear representation hypothesis (LRH), there may indeed be hope. In the LRH concept space, which is typically recovered via a sparse autoencoder (SAE), nearby points share a small common active support. Using this local sparsity insight, we propose a framework for locally masked SAE-based anomaly detection, supported by theoretical justifications. We vali
    
[^95]: 面向智能车辆多模态舱内交互的QoS感知联邦学习

    QoS-Aware Federated Learning for Multimodal In-Cabin Interaction in Smart Vehicles

    [https://arxiv.org/abs/2609.20123](https://arxiv.org/abs/2609.20123)

    提出了一种QoS感知的异步事件触发联邦学习框架FedQoS，通过两阶段门控机制将本地计算与全局通信解耦，解决了车载网络异构时变的服务质量约束下智能车辆多模态舱内交互协同训练的难题。

    

    现代智能车辆利用多模态传感器——从高带宽视觉系统到低速率生理监测设备——来提供个性化的舱内服务。然而，将高保真多模态融合与协同训练相结合，常常受到车载网络异构且时变的服务质量（QoS）约束的阻碍。标准的联邦学习（FL）方法强制执行刚性的同步轮次，无法考虑这些资源不对称性，从而导致安全关键的时间违规和能量耗尽。在本文中，我们提出了FedQoS，这是一种新颖的异步、事件触发的联邦学习框架，它通过两阶段门控机制将本地计算与全局通信解耦。首先，我们引入了资源感知训练门控，只有当感知缓冲区和能量储备满足安全阈值时才启动本地学习，从而防止机器学习任务损害车辆核心功能。

    arXiv:2609.20123v1 Announce Type: new  Abstract: Modern smart vehicles leverage multimodal sensors, ranging from high-bandwidth vision systems to low-rate physiological monitors, to provide personalized in-cabin services. However, integrating high-fidelity multimodal fusion with collaborative training is often hindered by the heterogeneous and time-varying Quality of Service (QoS) constraints of vehicular networks. Standard Federated Learning (FL) approaches enforce rigid synchronous rounds that fail to account for these resource asymmetries, leading to safety-critical timing violations and energy exhaustion. In this paper, we propose FedQoS, a novel asynchronous, event-triggered FL framework that decouples local computation from global communication via a two-phase gating mechanism. First, we introduce a resource-aware training gate that initializes local learning only when sensing buffers and energy reserves meet safety thresholds, preventing ML tasks from compromising core vehicle m
    
[^96]: CARE-VI：离策略演员-评论家学习中面向价值提升的保守自适应可靠性估计

    CARE-VI: Conservative Adaptive Reliability Estimation for Value Improvement in Off-Policy Actor-Critic Learning

    [https://arxiv.org/abs/2609.20098](https://arxiv.org/abs/2609.20098)

    提出CARE-VI框架，通过保守自适应排序与筛选（CARS）、选择器-评估器价值评估（SEVA）和动态自适应风险感知增强三种协同机制，解决了候选动作排序噪声、选择得分复用偏差和固定权重放大弱证据三大风险，从而提升离策略演员-评论家学习中时序差分目标的可靠性。

    

    可靠的时序差分目标是离策略演员-评论家学习的核心。直接价值提升方法通过备选动作对下一状态目标进行精化，但这种精化的可靠性取决于候选动作如何被排序、审查和加权。噪声较大的排序可能导致过早提交候选动作，重复使用选择得分可能使目标估值产生偏差，而固定的增强权重则可能放大微弱的证据。为应对这些风险，我们提出了保守自适应排序与筛选（CARS），它在预设预算内保留一个有序的候选前缀，并且仅当观测到的边界差距超过按分歧程度缩放的不确定性半径时才进行缩减。选择器-评估器价值评估（SEVA）使用选择器评论家对候选动作进行排序，并采用单独参数化的评估器评论家来审查所选价值，同时将审查后的价值限制在选择器参考值之内。动态自适应风险感知增强（原文截断）……

    arXiv:2609.20098v1 Announce Type: new  Abstract: Reliable temporal-difference targets are central to off-policy actor-critic learning. Direct value improvement refines the next-state target with alternative actions, but the reliability of this refinement depends on how candidate actions are ranked, reviewed, and weighted. Noisy rankings may force premature candidate commitment, reusing selection scores may bias target valuation, and fixed enhancement weights may amplify weak evidence. To address these risks, we develop Conservative Adaptive Ranking and Screening (CARS), which retains an ordered candidate prefix within a preset budget and narrows it only when the observed boundary gap exceeds a disagreement-scaled uncertainty radius. Selector-Evaluator Value Assessment (SEVA) uses selector critics to order candidates and a separately parameterized evaluator critic to review the selected value, then caps the reviewed value at the selector reference. Dynamic Adaptive Risk-aware Enhancemen
    
[^97]: SETTer：用于长期多变量时间序列预测的稀疏编码器Transformer

    SETTer: Sparse-Encoder Transformer for Long-term Multivariate Time Series Forecasting

    [https://arxiv.org/abs/2609.20086](https://arxiv.org/abs/2609.20086)

    SETTer是一种用于长期多变量时间序列预测的Transformer模型，通过解耦自注意力和混合掩码技术，有效捕捉时间与通道维度上的短期和长期模式，并增强了模型的可解释性。

    

    长期多变量时间序列在电力系统、金融交易等众多应用领域中发挥着重要作用。然而，由于其通常表现出高维度和复杂的关系，传统预测方法难以对其进行准确预测。近期的研究表明，基于Transformer的方法凭借其注意力机制在长期预测任务中相当有效。然而，在面对复杂的高维输入时，这类方法存在过度平滑、容量有限和不透明等问题。为此，本文提出了SETTer，一种基于Transformer的模型，通过引入解耦自注意力和混合掩码等新技术来应对这些挑战。所提出的技术使SETTer能够有效捕捉时间和通道维度上的主要短期和长期模式。此外，我们还用简单的可解释结构丰富了模型层……

    arXiv:2609.20086v1 Announce Type: new  Abstract: Long-term multivariate time series plays a significant role in many application areas such as power systems, trading, etc. However, their accurate prediction is quite difficult for conventional forecasting methods as they often exhibit high dimensionality and complex relationships. Recent works show that transformer-based approaches are quite effective for long-term forecasting thanks to their attention mechanism. However, in the presence of complex high-dimensional inputs, they show evidence of oversmoothing, limited capacity, and opacity. To this end, this paper introduces SETTer, a transformer-based model that addresses these challenges by incorporating novel techniques for decoupled self-attention and hybrid masking. The proposed techniques enable SETTer to effectively capture the dominant short- and long-term patterns across the temporal and channel dimensions. In addition, we enrich the model layers with simple explainable structur
    
[^98]: MATCH：基于课程调度与分层门控奖励的模型感知工具学习

    MATCH: Model-Aware Tool Learning with Curriculum Scheduling and Hierarchically Gated Rewards

    [https://arxiv.org/abs/2609.20082](https://arxiv.org/abs/2609.20082)

    MATCH提出了一种模型感知的闭环工具学习框架，通过课程难度与策略能力共同演化的课程调度，以及按工具名称、参数键、参数值逐级门控授予信用的分层奖励机制，解决了固定阈值课程脱节与加性奖励信用泄漏两大问题。

    

    工具学习使大语言模型（LLM）能够使用外部工具来完成超出其参数化知识的任务。强化学习可以通过反馈来优化工具调用行为，但现有方法仍面临两个问题：固定阈值的课程可能与策略不断演进的能力边界脱节；当预测的工具名称错误时，加性奖励可能导致参数级别的信用泄漏。为解决这些问题，我们提出了MATCH——一个融合课程调度与分层门控奖励的模型感知工具学习闭环框架。模型感知课程学习（MACL）维护由奖励导出的样本难度，使其与策略共同演化，并在每个训练周期中选择位于当前能力边界附近的样本，同时辅以一个难度更高的top-k样本池。分层工具调用门控奖励（HTGR）将工具名称、参数键和参数值作为一条门控链进行评分，仅当上一层级预测正确时才在相应层级给予信用。

    arXiv:2609.20082v1 Announce Type: cross  Abstract: Tool learning enables large language models (LLMs) to use external tools for tasks beyond parametric knowledge. Reinforcement learning can optimize tool-call behavior from feedback, but current methods still face two problems: fixed-threshold curricula can become misaligned with the policy's evolving capability boundary, and additive rewards can leak argument-level credit when the predicted tool is wrong. To address these problems, we propose MATCH, a closed-loop framework for model-aware tool learning with curriculum scheduling and hierarchically gated rewards. Model-Aware Curriculum Learning (MACL) maintains reward-derived sample difficulty that co-evolves with the policy, and each epoch selects samples near the current capability boundary together with a top-k pool of harder cases. Hierarchical Tool-call Gated Reward (HTGR) scores tool name, argument key, and argument value as a gated chain, granting credit at each level only when p
    
[^99]: 通过解释方法所诱导的预测器来评估解释方法

    Evaluating Explanation Methods by the Predictors They Induce

    [https://arxiv.org/abs/2609.20058](https://arxiv.org/abs/2609.20058)

    提出一种新的解释方法评估框架：将解释直接转化为可加性预测器并衡量其在未见数据上重现模型预测的能力，无需任何拟合，从而客观比较PDP、ALE、SHAP和LIME等方法的优劣。

    

    机器学习模型的解释通常依据难以相互比较的标准来评判。我们提出了一个更简单的检验方法：如果一种解释真正描述了模型如何使用其特征，那么应该能够从该解释出发重建模型的预测。我们将每种解释转化为一个预测器，方法是读取每个特征的效应并将其相加，然后衡量该预测器在未见数据上重现模型的能力。由于没有任何拟合过程，因此该分数反映的是解释本身。这一检验适用于任何能够表示为特征函数的解释；我们在部分依赖图（PDP）、累积局部效应（ALE）、SHAP和LIME上进行了演示。我们证明，当特征相互独立时，对部分依赖曲线求和可以得到模型的最佳可加性摘要，而当特征相互依赖时该方法会失效。在13个真实数据集、9个合成设计以及四个模型家族上，我们比较了各方法的得分表现。

    arXiv:2609.20058v1 Announce Type: new  Abstract: Explanations of machine learning models are usually judged by criteria that are hard to compare. We propose a simpler test: if an explanation really describes how a model uses its features, it should be possible to rebuild the model's predictions from it. We turn each explanation into a predictor by reading each feature's effect and adding them up, and measure how well that predictor reproduces the model on unseen data. Nothing is fitted, so the score reflects the explanation itself. The test applies to any explanation that can be written as a function of the features; we demonstrate it on partial dependence plots (PDP), accumulated local effects (ALE), SHAP and LIME. We prove that summing partial dependence curves gives the best possible additive summary of a model when its features are independent, and that this fails when they are dependent. Across 13 real datasets and 9 synthetic designs and four model families, which method scores b
    
[^100]: 当下正确，日后不足：审计上下文压缩中的更新充分性

    Correct Now, Insufficient Later: Auditing Update Sufficiency in Context Compression

    [https://arxiv.org/abs/2609.20045](https://arxiv.org/abs/2609.20045)

    该论文提出配对历史审计方法，揭示上下文压缩的记忆系统虽能正确回答当前查询，却可能丢弃后续更新所需的关键区分信息，并构建记录级审计框架以区分记忆保留充分性、响应传递和答案格式合规性等不同失败模式。

    

    记忆系统能够正确回答当前查询，同时却丢弃了后续更新所需的区分信息。我们通过配对历史审计来研究这种失败现象：两个历史拥有相同的当前答案，接收相同的未来更新，但需要不同的后续答案。一项先导实验评估了24个历史对，涵盖六种合成机制、12种记忆条件、两次重复和两个模型后端。一个确定性前沿选择器在DeepSeek上获得了96/96的严格揭示准确率，在GLM上为82/96；一个结构化写入器获得62个成功、1个未解决结果和56/96。所配置的四结果联合对比具有有限样本识别区间[0.521, 0.542]和[0.292, 0.313]，而非置信区间。记录级审计在不改变这些原始分数的情况下，区分了保留状态充分性、响应传递和答案模式合规性。它发现了26个和25个格式良好但语义错误的结构……

    arXiv:2609.20045v1 Announce Type: cross  Abstract: A memory can answer a current query correctly while discarding distinctions required by a later update. We investigate this failure with a paired-history audit: two histories have the same current answer, receive a shared future update, and require different subsequent answers. A pilot evaluates 24 history pairs across six synthetic mechanisms, 12 memory conditions, two repeats, and two model backends. A deterministic frontier selector obtains strict reveal accuracy of 96/96 on DeepSeek and 82/96 on GLM; a structured writer obtains 62 successes with one unresolved outcome and 56/96. The configured four-outcome joint contrast has finite-sample identification intervals of [0.521, 0.542] and [0.292, 0.313], not confidence intervals. A record-level audit distinguishes retained-state adequacy, response delivery, and answer-schema compliance without changing those original scores. It finds 26 and 25 well-formed but semantically wrong structu
    
[^101]: 动态广义Gromov-Wasserstein最优传输

    Dynamic Generalized Gromov-Wasserstein Optimal Transport

    [https://arxiv.org/abs/2609.20008](https://arxiv.org/abs/2609.20008)

    该论文提出TP-DATE框架，首次以无模拟方式将Gromov-Wasserstein最优传输动态化，通过路径作用量证明静态与动态二次型最优传输的等价性，并发展行进对流匹配方法，实现空间转录组学中兼顾组织结构保持的连续轨迹重建。

    

    Gromov-Wasserstein最优传输（GW-OT）通过引入结构感知的传输代价扩展了经典最优传输。这对于空间转录组学尤为重要，因为在空间转录组学中，动态重建除了匹配表达模式外，还应保留组织结构。尽管静态形式已被广泛用于此类结构感知对齐，但用于重建连续轨迹的一般动态形式仍然缺失。我们引入了行进对动态对齐与轨迹估计，这是一个以无模拟方式动态推广GW-OT的理论与计算框架。我们通过路径作用量表述了一大类静态和动态二次型最优传输（QOT），并证明了静态与动态的等价性。我们进一步发展了行进对流匹配方法，该方法允许条件路径之间相互作用，并将其交互边缘化为单一向量场（摘要在此处被截断）。

    arXiv:2609.20008v1 Announce Type: cross  Abstract: Gromov--Wasserstein optimal transport (GW-OT) extends classical optimal transport by introducing structure-aware transport cost. This is particularly relevant for spatial transcriptomics, where dynamical reconstruction should preserve tissue structure in addition to matching expression patterns. While static formulations have been widely used for such structure-aware alignment, a general dynamic formulation for reconstructing continuous trajectories is still missing. We introduce Travelling Pair Dynamical Alignment and Trajectory Estimation (TP-DATE), a theoretical and computational framework to generalize GW-OT dynamically in a simulation-free manner. We formulate a broad class of static and dynamic Quadratic-form OT (QOT) through path actions and prove the static dynamic equivalence. We further develop travelling-pair flow matching, which allows interacting conditional paths and marginalizes their interactions into a single vector fi
    
[^102]: EPIG-Tree：面向梯度高效强化学习的计算最优分支方法

    EPIG-Tree: Compute-Optimal Branching for Gradient-Efficient Reinforcement Learning

    [https://arxiv.org/abs/2609.20004](https://arxiv.org/abs/2609.20004)

    该论文提出EPIG-Tree方法，通过全方差定律分解推导出两条计算分配定律，将树状分支放置在每单位计算能最大程度降低策略梯度不确定性的位置，从而实现计算最优且梯度高效的强化学习。

    

    以组相对策略优化（GRPO）为代表的基于奖励的语言模型强化学习，将整条随机轨迹坍缩为单一的标量奖励。这种方式简洁且易于扩展，但在探索和奖励分配上效率低下：一条轨迹可能包含许多因果决策、恢复尝试和环境随机性事件，然而每个token或动作却继承同一个轨迹级别的优势值。我们将基于树的rollout构建视为策略梯度估计中的计算分配问题进行研究。我们的核心主张是：分支不应仅仅放置在策略不确定的地方，而应放置在每单位计算下、新增分支能最大程度降低策略梯度不确定性的位置。通过对局部策略梯度随机变量进行全方差定律分解，我们推导出两条分配定律：新增分支用于降低决策不确定性，而重复的后缀rollout用于降低延续（不确定性）……

    arXiv:2609.20004v1 Announce Type: cross  Abstract: Reward-based reinforcement learning for language models, exemplified by Group Relative Policy Optimization (GRPO), collapses an entire stochastic trajectory into a single scalar reward. This is clean and scalable, but it explores and allocates reward inefficiently: a trajectory may contain many causal decisions, recovery attempts, and environment-randomness events, yet every token or action inherits one trajectory-level advantage. We study tree-based rollout construction as a compute-allocation problem for policy-gradient estimation. Our central claim is that branches should be placed not where the policy is merely uncertain, but where an additional branch most reduces uncertainty about the policy gradient per unit of compute. From a law-of-total-variance decomposition of the local policy-gradient random variable, we derive two allocation laws: new branches reduce decision uncertainty, while repeated suffix rollouts reduce continuation
    
[^103]: 过去与未来，一步到位：通过事后JANUS修正缓解稳定性-可塑性困境

    Past, Future, All at Once: Mitigating Stability-Plasticity Dilemma via Post-hoc JANUS Rectification

    [https://arxiv.org/abs/2609.19985](https://arxiv.org/abs/2609.19985)

    提出了一种事后且与微调无关的JANUS权重修正框架，通过将参数更新投影到雅可比零空间实现参数空间正交性，在微调新任务的同时有效缓解灾难性遗忘、恢复历史知识。

    

    在新任务上微调基础模型不可避免地会遭受灾难性遗忘。虽然现有工作试图在参数高效微调方法的基础上缓解这一问题，但它们采用了过于严格的子空间正交性条件。在本文中，我们引入了一个纯粹的事后且与微调方式无关的权重修正框架，实现了参数空间正交性——这是一阶意义上保持历史性能的充要条件。通过将参数更新投影到雅可比零空间（JANUS）中，我们的方法在不干扰底层微调过程的前提下，显著恢复了受损的历史知识。为了克服雅可比近似的局部有效性限制，我们进一步提出了一种多步自适应修正机制，利用JANUS位移动态验证有效信任区域并调整步长。

    arXiv:2609.19985v1 Announce Type: cross  Abstract: Fine-tuning foundation models on new tasks inevitably suffer from catastrophic forgetting. While existing works attempt to mitigate this on the basis of parameter-efficient fine-tuning methods, they adopted an overly restrictive Subspace Orthogonality condition. In this paper, we introduce a purely post-hoc and tuning-agnostic weight rectification framework that achieves Parameter Space Orthogonality, which is the necessary and sufficient condition for preserving historical performance to the first order. By projecting parameter updates into the JAcobian NUll Space (JANUS), our method significantly recovers compromised historical knowledge without interfering with the underlying fine-tuning process. To overcome the local validity of the Jacobian approximation, we further propose a Multi-step Adaptive Rectification mechanism that utilizes the JANUS shift to dynamically verify the valid trust region and adjust step sizes. Coupled with ou
    
[^104]: 量子图卷积网络：实现与可训练性分析

    Quantum Graph Convolutional Networks: Implementation and Trainability Analysis

    [https://arxiv.org/abs/2609.19983](https://arxiv.org/abs/2609.19983)

    本文基于QGNN框架实现了SGC和LGC两种量子图卷积网络，通过量子模拟证明其能以更少参数达到与经典模型相当的性能，并通过代价梯度分析确定了这些量子模型可训练的适用任务。

    

    图神经网络（GNN）在图结构数据上实现了最先进的性能，但在大图上的训练和推理通常受到内存限制和稀疏线性代数计算负载的瓶颈制约。量子计算提供了一套替代性原语，有望改善图学习的可扩展性。本文在Liao等人的量子图神经网络（QGNN）框架基础上，实现了两种代表性架构——简化图卷积（SGC）和线性图卷积（LGC）模型——并利用量子模拟在开放的基准图数据集和半监督学习任务上对其进行评估。我们将量子模型的预测性能和优化行为与经典基线进行了比较，结果表明量子模型能够以更少的参数实现有竞争力的性能。最后，我们提出了代价梯度分析，以确定所展示模型可训练的任务类型。

    arXiv:2609.19983v1 Announce Type: cross  Abstract: Graph Neural Networks (GNNs) achieve state-of-the-art performance on graph-structured data, but training and inference on large graphs are often bottlenecked by memory constraints and sparse linear-algebra workloads. Quantum computing offers an alternative set of primitives that may improve scalability for graph learning. Building on the quantum graph neural network (QGNN) framework of Liao \textit{et al.}, this work implements two representative architectures --- the Simplified Graph Convolution (SGC) and Linear Graph Convolution (LGC) models --- and evaluates them on open benchmark graph datasets and semi-supervised learning tasks using quantum simulation. We compare predictive performance and optimization behavior against classical baselines, showing that the quantum models achieve competitive performance with fewer parameters. Finally, we present a cost gradient analysis that identifies the tasks for which the models showcased are 
    
[^105]: CellRFT：面向单细胞扰动建模的强化微调框架

    CellRFT: Reinforcement Fine-Tuning for Single-Cell Perturbation Modeling

    [https://arxiv.org/abs/2609.19970](https://arxiv.org/abs/2609.19970)

    提出CellRFT强化微调框架，以生物学评估作为直接训练反馈，通过策略梯度优化与分层奖励聚合改进单细胞扰动预测。

    

    预测细胞对扰动的响应有助于研究基因功能、疾病机制和治疗策略。尽管单细胞扰动建模已取得进展，但现有模型通常优化的是替代损失，这些损失并不能直接反映用于评估的生物学标准，因此更好的数据拟合未必能带来更好的生物学预测。为解决这一不匹配问题，我们提出了CellRFT，这是一个以生物学评估作为直接训练反馈的强化微调框架。CellRFT利用策略梯度优化从生成的细胞群落的不可微评估中学习，并通过分层奖励聚合整合多个生物学奖励。全面的实验证明了CellRFT在不同预训练模型上的适用性以及在提升扰动预测方面的有效性，并揭示了优化某一生物学标准可能有助于或……（摘要截断）

    arXiv:2609.19970v1 Announce Type: new  Abstract: Predicting cellular responses to perturbations supports the study of gene function, disease mechanisms, and therapeutic strategies. Despite advances in single-cell perturbation modeling, existing models typically optimize surrogate losses that do not directly reflect the biological criteria used for evaluation, so better data fitting need not yield better biological predictions. To address this mismatch, we introduce \textbf{CellRFT}, a reinforcement fine-tuning framework that uses biological evaluation as direct training feedback. CellRFT uses policy-gradient optimization to learn from non-differentiable evaluations of generated cell populations and integrates multiple biological rewards through hierarchical reward aggregation. Comprehensive experiments demonstrate CellRFT's applicability across different pretrained models and effectiveness in improving perturbation prediction, reveal that optimizing one biological criterion can help or
    
[^106]: 基于图的随机Power-UCT：结合幂平均估计的蒙特卡洛图搜索

    Graph-Based Stochastic Power-UCT: Monte-Carlo Graph Search with Power Mean Estimation

    [https://arxiv.org/abs/2609.19956](https://arxiv.org/abs/2609.19956)

    提出基于图的随机Power-UCT算法（GS-Power-UCT），通过在相同规划深度共享状态并保留不同深度的独立值，在含环路的一般随机MDP中实现样本复用，证明了以 O(n^{-1/2}) 速率收敛到有限视界值，并进一步提出两种全状态变体以增加样本共享并控制视界混合偏差。

    

    arXiv:2609.19956v1 公告类型：新论文 摘要：基于树的蒙特卡洛树搜索（MCTS）在随机马尔可夫决策过程（MDP）中，当同一状态通过不同轨迹被到达时会重复该状态，从而在随机MDP中浪费模拟次数。我们提出了基于图的随机Power-UCT（GS-Power-UCT），它在相同规划深度处共享到达的状态，同时为在不同深度到达的状态保留各自独立的值。该设计适用于一般的随机MDP，包括存在环路的问题。我们证明，对于固定的规划视界，根节点估计以 O(n^{-1/2}) 的速率收敛到有限视界值，与基于树的随机Power-UCT相匹配，同时在共享状态间复用样本。我们还研究了两个全状态变体：GS-Power-UCT-F，为每个物理状态存储一个节点以增加样本共享，但可能混合来自不同剩余视界的值；以及GS-Power-UCT-F⁺，它使用自适应视界来控制这种偏差。后者收敛到最优值 $V^{\star}(s_0)$。

    arXiv:2609.19956v1 Announce Type: new  Abstract: Tree-based Monte-Carlo Tree Search (MCTS) duplicates the same state when it is reached through different trajectories, which can waste simulations in stochastic MDPs. We introduce Graph-Based Stochastic-Power-UCT (GS-Power-UCT), which shares states reached at the same planning depth while keeping separate values for states reached at different depths. This design applies to general stochastic MDPs, including problems with cycles. We prove that for a fixed planning horizon, the root estimate converges to the finite-horizon value at rate $O(n^{-1/2})$, matching tree-based Stochastic-Power-UCT while reusing samples across shared states. We also study two full-state variants: GS-Power-UCT-F, which stores one node per physical state to increase sample sharing but may mix values from different remaining horizons, and GS-Power-UCT-F$^+$, which uses an adaptive horizon to control this bias. The latter converges to $V^{\star}(s_0)$, the optimal i
    
[^107]: 每个强连通分量一次干预即可：从稳态实现线性随机动力学的可辨识性

    One Intervention per Component is Enough: Towards Identifiability in Linear Stochastic Dynamics from Steady State

    [https://arxiv.org/abs/2609.19955](https://arxiv.org/abs/2609.19955)

    本文证明了对漂移图的每个强连通分量仅需一次干预，即可从稳态数据中辨识多元OU过程的全部参数，并提出了相应的递归学习算法。

    

    我们研究了从稳态观测和干预数据中恢复多元Ornstein-Uhlenbeck（OU）过程参数的问题。在许多应用场景中，例如大规模基因扰动实验，仅有平稳的“快照”式测量数据可用，这使得依赖时间序列轨迹的标准随机微分方程估计方法不再适用。我们首先建立了一个可辨识性结果：对于漂移图的每个强连通分量（SCC）仅需进行一次干预，即可在全局缩放因子意义下通用地恢复所有OU过程参数。该结果成立的前提是SCC凝聚图连通且具有单一根节点，并满足某些谱非退化假设。我们提出了一种递归学习算法，该算法对SCC进行拓扑排序，并对每个分量隔离其边缘动力学，进而求解由稳态矩方程导出的线性方程组。

    arXiv:2609.19955v1 Announce Type: new  Abstract: We study the problem of recovering the parameters of a multivariate Ornstein-Uhlenbeck (OU) process from steady-state observational and interventional data. In many applications, such as large-scale gene perturbation experiments, only stationary "snapshot" measurements are available, making standard stochastic differential equation estimation methods that rely on time-series trajectories inapplicable. We first establish an identifiability result: one intervention per strongly connected component (SCC) of the drift graph suffices to recover all OU process parameters generically up to a global scaling factor. This holds provided that the SCC condensation graph is connected with a single root and certain spectral nondegeneracy assumptions hold. We propose a recursive learning algorithm that orders SCCs topologically and, for each component, isolates its marginal dynamics and solves a linear system derived from the steady-state moment equati
    
[^108]: 检索主导的抽取式问答中的内在序列似然置信度：两个预先设定的否定结果，及其能归因与不能归因的对象

    Intrinsic Sequence-Likelihood Confidence in Retrieval-Dominated Extractive QA: Two Pre-Specified Negatives, and What They Do and Do Not Attribute

    [https://arxiv.org/abs/2609.19942](https://arxiv.org/abs/2609.19942)

    该研究通过预先注册的评估标准证明，在检索已能恢复92-99.8%最优性能的抽取式问答场景中，模型的内在序列似然置信度无论作为蒸馏触发器还是路由弃答策略的控制信号均告失效。

    

    在抽取式文档问答中——其问题由包含答案的段落生成，因此检索即可恢复任何模式组合所能达到性能的92-99.8%（无论其绝对准确率如何）——基于置信度的机制几乎没有提升空间。在专业领域语料库上微调开源语言模型后，模型自身的置信度是一个颇具吸引力的控制信号：它可用于决定哪些查询值得进一步适配、哪些答案值得信赖。我们在实验开始前预先固定的评估标准下，对四个7-9B模型家族进行了评估（其领域适配使闭卷F1最多提升+0.03），结果两种用途均告失败：在预先设定的三步迁移预算下，蒸馏触发器在全部四个模型家族上失效，单模型试点中的路由与弃答策略同样失败。在我们测试的每一个正确性标准下，仅检索即可恢复最优组合准确率的92-99.8%，留下的……

    arXiv:2609.19942v1 Announce Type: new  Abstract: In extractive document question answering whose questions were generated from the passages that contain their answers -- so that retrieval recovers 92-99.8% of what any mode combination could reach, whatever its absolute accuracy -- confidence-driven mechanisms have little to gain. Fine-tuning an open language model on a specialized domain corpus yields a model whose own confidence is a tempting control signal: it could decide which queries warrant further adaptation, and which answers to trust. We evaluate both uses under criteria fixed before the runs were executed, across four 7-9B model families whose adaptation moved closed-book F1 by at most +0.03, and both fail: a distillation trigger on all four families, under its pre-specified three-step transfer budget, and a routing-and-abstention policy in its single-model pilot. Retrieval alone recovers 92-99.8% of best-case combined accuracy under every correctness criterion we test, leavi
    
[^109]: 字符串学序列预测 III：分层滑索与效率与表达性之间的权衡

    Stringological sequence prediction III: layered ziplines and a tradeoff between efficiency and expressivity

    [https://arxiv.org/abs/2609.19940](https://arxiv.org/abs/2609.19940)

    本文提出了一种基于分层滑索程序的较弱复杂度度量，该度量虽然表达性低于算术重复复杂度（ARC），但支持在拟线性时间和多对数空间内运行的高效预测算法，揭示了效率与表达性之间的权衡。

    

    arXiv:2609.19940v1 公告类型：交叉 摘要：在之前的论文中，我们开始研究适应于字符串学词复杂度度量的序列预测算法。特别地，我们定义了一种称为算术重复复杂度的复杂度度量，该度量允许多项式时间的预测算法，其错误界限相对于复杂度呈拟线性关系。在本文中，我们展示了一种与 ARC 相关的较弱复杂度度量，该度量允许一种特别高效的预测算法：对于适当的高度结构化序列，该算法可在拟线性时间和多对数空间内运行。该复杂度度量通过一类受限的“滑索程序”（直线程序的一种变体）来定义，我们称之为分层的。因此，我们得到了一个表达性较低但算法更高效的度量（与我们对 ARC 的研究结果相比），从而展示了一种可能的权衡。

    arXiv:2609.19940v1 Announce Type: cross  Abstract: In previous papers, we began the study of sequence prediction algorithms adapted to stringological word complexity measures. In particular, we defined a complexity measure called Arithmetic Repetition Complexity (ARC) which admits a polynomial-time prediction algorithm with a mistake bound quasilinear in the complexity. Here, we show a weaker complexity measure related to ARC that admits an especially efficient prediction algorithm: an algorithm that runs in quasilinear time and polylog space for appropriate highly-structured sequences. The complexity measure is defined via a restricted class of "zipline programs" (a variant of straight-line programs), which we call layered. We thus get a less expressive measure with a more efficient algorithm (compared to our results for ARC), demonstrating a possible tradeoff.
    
[^110]: 范数约束ReLU神经网络逼近的Sobolev范数误差界

    Error bounds in Sobolev norms for approximations with norm constrained ReLU neural networks

    [https://arxiv.org/abs/2609.19937](https://arxiv.org/abs/2609.19937)

    该论文将路径范数约束ReLU神经网络的逼近理论从一致逼近推广到Sobolev范数逼近，分别针对浅层网络和深层网络给出了以$W^{1,p}$范数度量的逼近误差界，且深层网络的误差界无需对函数光滑性附加限制。

    

    最近的研究表明，光滑函数可以通过权重具有路径范数约束的ReLU神经网络得到良好的逼近。我们将这些结果从一致逼近扩展到Sobolev范数下的逼近。具体而言，我们分析了$W^{n,p}$空间中的Sobolev函数能被宽度为$W$、深度为$L$且路径范数以$K$为界的神经网络逼近的程度，其中逼近误差以$W^{1,p}$范数度量。对于深度$L=1$的浅层网络，当光滑性指标满足$n<s=(d+3)/2$且输入为$d$维时，我们推导出逼近误差界$\mathcal{O}(\max\{W^{-(n-1)/d}, K^{-(n-1)/(s-n)}\})$。对于深层网络，我们消除了对光滑性的限制，证明了当宽度$W$和深度$L$充分大时，逼近界$\mathcal{O}(K^{-(n-1)/(d+d/p+1)})$成立。

    arXiv:2609.19937v1 Announce Type: cross  Abstract: Recent studies have shown that smooth functions can be well approximated by ReLU neural networks with path norm constraint on the weights. We extend these results from uniform approximation to approximation in Sobolev norm. Specifically, we analyze how well Sobolev functions in $W^{n,p}$ can be approximated by neural networks with width $W$, depth $L$ and path norm bounded by $K$, when the approximation error is measured in the $W^{1,p}$-norm. For shallow networks with depth $L=1$, we derive the approximation error bound $\mathcal{O}(\max\{W^{-(n-1)/d}, K^{-(n-1)/(s-n)}\})$, when the smoothness index satisfies $n<s=(d+3)/2$ and the input is $d$-dimensional. For deep networks, we remove the restriction on the smoothness by showing that the approximation bound $\mathcal{O}(K^{-(n-1)/(d+d/p+1)})$ holds if the width $W$ and depth $L$ are sufficiently large.
    
[^111]: 从“这个用户是谁？”到“这次购买意味着什么？”：银行级规模语义用户画像的已部署流水线

    From "Who Is This User?" to "What Does This Purchase Mean?": A Deployed Pipeline for Semantic User Profiling at Bank Scale

    [https://arxiv.org/abs/2609.19928](https://arxiv.org/abs/2609.19928)

    该论文提出一种已部署的三阶段LLM流水线（解析-画像-标注），将用户属性推断从“逐用户”转变为“逐交易模式”，使推理成本随模式数而非用户数增长，在银行级规模下实现了与逐用户LLM推断统计上无差异的语义用户画像。

    

    对交易历史进行逐用户的大语言模型（LLM）推理会使推理预算与用户数量呈线性绑定关系，这在实际应用规模下变得难以承受。我们将属性推断从“逐用户”重新构建为“逐交易模式”。该流水线分三个阶段运行：Resolve（解析）阶段利用可选的网络信息锚定（web grounding）抽象化商品名称；Profile（画像）阶段为每个高频模式推断属性；Tag（标注）阶段将自由文本属性聚类为一个可查询的数据库。在 Profile 阶段，每个模式仅需一次 LLM 调用即可输出预定义的分类标签、自由文本属性以及每个属性的流行度估计。由于推理基于模式而非用户进行，预算随模式数量而非用户数量增长。在公开的电商语料库上，该数据库在所评估的各项属性的 AUC 指标上与直接阅读每个用户原始历史的 LLM 在统计上无显著差异，且流行度估计在正负样本之间携带判别信号。

    arXiv:2609.19928v1 Announce Type: new  Abstract: Per-user LLM inference on transaction histories binds the inference budget linearly to user count, which becomes prohibitive at applied scale. We re-cast attribute inference from per-user to per-transaction-pattern. The pipeline runs in three phases: Resolve abstracts item names with optional web grounding, Profile infers attributes for each frequent pattern, and Tag clusters free-text attributes into a queryable database. In Profile, a single LLM call per pattern emits predefined categorical labels, free-text attributes, and per-attribute prevalence estimates. Because inference runs over patterns rather than users, the budget grows with the pattern count rather than the user count. On the public Open e-commerce corpus, the database is statistically indistinguishable from an LLM that reads each user's raw history directly in AUC across the evaluated attributes, and the prevalence estimates carry discriminative signal between positive and
    
[^112]: 一个Token的生命历程：从文字到线缆上的比特

    The Life of a Token: from Words to Bits on the Wire

    [https://arxiv.org/abs/2609.19924](https://arxiv.org/abs/2609.19924)

    这是一篇教程论文，以但丁《神曲》为具体示例，完整讲解了文字如何经过分词、向量嵌入，最终转化为高性能计算系统中网络通信比特流的全过程，揭示了LLM训练中不为人知的底层通信机制。

    

    大型语言模型（LLM）将海量的非结构化文本转化为用于语言生成和推理任务的语义模式。在其易用性的背后隐藏着一个复杂的过程：文字变成token，token变成向量，向量最终产生流经高性能计算（HPC）系统的比特流。随着现代LLM的参数规模增长到数十亿甚至数万亿，这一过程越来越多地分布在数千个相互连接的加速器上，使得底层的通信网络结构成为模型训练中至关重要却往往不透明的组成部分。本教程旨在引导读者了解从文字到网络流量的完整旅程，揭示语言如何在HPC训练系统中被转化为通信流。通过但丁《神曲》的具体示例，我们阐释了模型架构、分词、嵌入和并行化策略如何影响通信数据的规模……

    arXiv:2609.19924v1 Announce Type: cross  Abstract: Large Language Models (LLMs) transform vast collections of unstructured text into semantic patterns used for language generation and reasoning tasks. Behind their ease of use lies a complex process: words become tokens, tokens become vectors, and vectors ultimately give rise to streams of bits that flow through High-Performance Computing (HPC) systems. As modern LLMs grow to billions or trillions of parameters, this path increasingly unfolds across thousands of interconnected accelerators, making the underlying communication fabric a critical and often opaque component of model training. This tutorial aims to walk the reader through the journey from words to network traffic, shedding light on how language is translated into communication flows within HPC training systems. Using concrete examples from Dante's Divine Comedy, we illustrate how model architecture, tokenization, embeddings, and parallelization strategies shape the volume, s
    
[^113]: 通过图超网络摊销物理信息神经求解器

    Amortizing Physics-Informed Neural Solvers via Graph Hypernetworks

    [https://arxiv.org/abs/2609.19915](https://arxiv.org/abs/2609.19915)

    该论文提出将偏微分方程的算子和跨场关系显式编码为算子图，利用图超网络为元训练的因式分解PINN生成初始化编码，在无需解标签的情况下将物理信息神经求解器摊销到相关PDE族，并在固定适应预算内比系数向量和DeepSets描述符取得更高求解精度。

    

    arXiv:2609.19915v1 公告类型：新论文。摘要：在相关偏微分方程（PDE）族之间摊销物理信息神经网络（PINNs）需要向一个可重用的求解器描述每个方程。系数向量在预定义的槽位中编码数值参数，使得算子和跨场分配关系保持隐式。我们在一个算子图中显式表达这些关系，图中节点表示场、导数、项和残差，而系数作为项的属性保留。图超网络生成对角编码，用于为每个目标方程初始化一个元训练的因式分解PINN。元训练和目标特定适应仅使用控制方程和规定条件，无需解标签。我们在固定的适应预算内，通过求解精度比较了系数向量、基于DeepSets的项集和图条件化三种方法。在标量对流-扩散-反应问题中，两种基于项的描述符都提高了高反应系数情形下的精度，且性能相近。在双场Fisher-KPP问题中……

    arXiv:2609.19915v1 Announce Type: new  Abstract: Amortizing physics-informed neural networks (PINNs) across related PDEs requires describing each equation to a reusable solver. Coefficient vectors encode numerical parameters in predefined slots, leaving operator and cross-field assignments implicit. We make these relationships explicit in an operator graph, with nodes for fields, derivatives, terms, and residuals and coefficients retained as term attributes. A graph hypernetwork generates diagonal codes that initialize a meta-trained factorized PINN for each target equation. Meta-training and target-specific adaptation use governing equations and prescribed conditions without solution labels. We compare coefficient-vector, DeepSets-based term-set, and graph conditioning by solution accuracy within a fixed adaptation budget. In scalar convection-diffusion-reaction problems, both term-based descriptors improve high-reaction accuracy, with similar performance. In two-field Fisher-KPP, met
    
[^114]: 用于观点动力学的数字孪生：面向社交网络的生成式大语言模型框架

    Digital Twins for Opinion Dynamics: A Generative LLM Framework for Social Networks

    [https://arxiv.org/abs/2609.19913](https://arxiv.org/abs/2609.19913)

    本文提出了一种基于数字孪生概念的生成式大语言模型框架，通过克隆真实Twitter网络、为智能体赋予多维属性并利用Mistral-7B模拟观点更新，且在真实数据集上验证了其有效性。

    

    社交网络中的观点动力学研究是计算社会科学的关键挑战之一，与理解政治极化、错误信息和健康应对直接相关。目前的方法要么专注于简化的数学模型，忽略了与信念更新相关的语言和情境因素，要么使用尚未经过真实数据验证的基于大语言模型（LLM）的模拟。我们提出了一个基于数字孪生概念的框架，用于模拟社交网络中的观点动力学。该方法通过克隆真实的Twitter网络填补了这一空白，为智能体分配一组属性（如角色设定、情绪、中心性、固执性和影响力），并采用Mistral-7B基于记忆和社交暴露来执行观点更新。为了评估所提出的方法，我们使用两个真实的Twitter数据集（COVID-19讨论和美国2020年大选）对其进行了验证。

    arXiv:2609.19913v1 Announce Type: new  Abstract: The study of opinion dynamics in social networks is one of the key challenges in computational social science with direct relevance to understanding political polarization, misinformation, and health responses. Current approaches focus on simplified mathematical models that ignore linguistic and contextual factors related to belief updates or use Large Language Model (LLM)-based simulations that have not been validated against real data. We present a framework based on the concept of a digital twin to simulate opinion dynamics in social networks. The approach fills the gap by cloning a real-world Twitter network, assigns a set of attributes for agents (such as persona, emotions, centrality, stubbornness, and influence), and employs Mistral-7B to perform opinion update based on memory and social exposure. To evaluate the proposed approach, we validate it against two real Twitter datasets (COVID-19 discourse and U.S elections 2020). The re
    
[^115]: REARL：一种结合真实交通数据与大语言模型的闭环自动驾驶仿真增强框架

    REARL: A Closed-loop Autonomous Driving Simulation Enhancement Framework with Real Traffic Data and Large Language Models

    [https://arxiv.org/abs/2609.19903](https://arxiv.org/abs/2609.19903)

    REARL提出了一种结合真实交通数据聚类与大语言模型的闭环仿真增强框架，通过实时监测仿真与真实交通的偏差并动态调整车辆决策，使自动驾驶仿真更贴近真实交通模式。

    

    准确的仿真对自动驾驶开发至关重要，然而捕捉真实世界的交通复杂性仍然具有挑战性。依赖预定义规则或静态数据回放的现有仿真器难以应对动态交通场景。CRITICAL方法使用真实交通数据和大语言模型（LLM）来调整初始仿真配置，但随着仿真过程的推进，仿真分布仍会偏离真实交通。我们提出REARL，一个将真实交通数据与大语言模型相结合的闭环仿真增强框架。该方法对真实交通数据进行聚类，每个聚类中心作为代表性场景，为大语言模型提供典型的真实世界交通模式。随后，一个定时滑动窗口检测器监测车辆速度分布以及车辆对之间平均间距的偏差。若某项指标超过阈值，则由大语言模型调整车辆的决策行为；否则保留现有控制器继续运行。

    arXiv:2609.19903v1 Announce Type: new  Abstract: Accurate simulation is crucial for autonomous driving development, yet capturing real-world traffic complexity remains challenging. Existing simulators that rely on predefined rules or static data playback struggle with dynamic traffic. CRITICAL uses real traffic data and a large language model (LLM) to adjust the initial simulation configuration, but the simulated distribution still diverges from real traffic as the rollout evolves. We propose REARL, a closed-loop simulation enhancement framework that integrates real traffic data with LLMs. Real traffic data are clustered, and each cluster center is used as a representative scenario that provides typical real-world traffic patterns for the LLM. A timed sliding-window detector then monitors discrepancies in vehicle speed distribution and mean spacing between pairs of vehicles. If a metric exceeds a threshold, the LLM adjusts vehicle decision-making; otherwise the existing controller is k
    
[^116]: 自我复制的神经细胞自动机：在开放式基底中量化涌现的表型与基因型多样性

    Self-Replicating Neural Cellular Automata: Quantifying Emergent Phenotypic and Genotypic Diversity in an OpenEnded Substrate

    [https://arxiv.org/abs/2609.19902](https://arxiv.org/abs/2609.19902)

    本文提出了自我复制的神经细胞自动机系统，并设计了一套表型与基因型两个尺度的粗粒度多样性度量指标，首次对开放式进化基底中涌现的空间组织化生态系统的多样性进行了量化分析。

    

    我们研究了一种计算机模拟的基底，其中双通道细胞自动机网格的每个像素都携带一个微型神经网络（智能体），该网络感知其摩尔邻域。细胞仅通过自我复制得以存续：克隆一个存活的邻居并对其权重施加均匀扰动进行变异，从而使表型（细胞状态）完全由基因型（网络权重）驱动。从少数种子祖先出发，系统逐渐成长为一个空间组织化的生态系统，其中包含共存、竞争和占据主导地位的多个物种。我们的主要贡献是一套粗粒度多样性度量方法，能够在两个尺度上对这种增长进行量化：四个基于细胞类型频率、熵和细胞方差的表型工具，以及两个基因型工具——通过完整权重向量的哈希值与稀疏随机权重探针对每个智能体进行着色区分。在1680次小规模运行和24次长时间运行（1000代、200×200网格）的五倍参数扫描中，该基底表现出持续性...

    arXiv:2609.19902v1 Announce Type: cross  Abstract: We study an in-silico substrate in which every pixel of a two-channel cellular-automata grid carries a tiny neural network (an agent) that senses its Moore neighborhood. A cell persists only by self-replication: a living neighbor is cloned and its weights are mutated by a uniform perturbation, so that phenotype (cell state) is driven entirely by genotype (network weights). From a handful of seeded founders the system grows into a spatially organized ecosystem of coexisting, competing and dominating species. Our main contribution is a battery of coarse-grained diversity metrics that make such growth measurable at two scales: four phenotypic tools based on cellular-type frequency, entropy and cell variance, and two genotypic tools that colour each agent by a hash of its full weight vector versus a sparse random-weight probe. Across a five-fold sweep of 1680 small runs and 24 long (1000-generation, 200 x 200) runs, the substrate is persis
    
[^117]: Delphi Scanner：基于API序列建模的高效且可解释的静态恶意软件检测

    Delphi Scanner: efficient and interpretable static malware detection via API sequence modeling

    [https://arxiv.org/abs/2609.19900](https://arxiv.org/abs/2609.19900)

    该论文提出Delphi Scanner系统，利用CNN对Windows API序列建模并结合基于规则的解耦解释层，以仅1.53 MB的紧凑模型实现了95.35%准确率的高效且可解释的Windows PE静态恶意软件检测，并展现出良好的分布外泛化能力和对抗逃避技术的鲁棒性。

    

    针对Windows可移植可执行（PE）文件的静态恶意软件检测需要在检测效果、计算效率和分析可解释性之间进行仔细的权衡。本文介绍了Delphi Scanner，这是一种面向Windows PE文件的静态恶意软件检测系统，它在效率与行为解释之间取得了平衡。该系统使用卷积神经网络（CNN）对Windows API序列进行建模以对PE文件进行分类，并通过一个基于规则层的解耦解释层将API归类为高层恶意能力。在超过190,000个Windows PE文件上的评估表明，该系统实现了95.35%的准确率，而模型大小仅为1.53 MB。在5,647个分布外的MalwareBazaar样本、成对的加壳与未加壳可执行文件以及三种对抗性操纵策略上进行的鲁棒性实验，证实了该系统在训练分布之外的泛化能力以及对保持功能的逃避技术的抵抗力。总体而言……

    arXiv:2609.19900v1 Announce Type: cross  Abstract: Static malware detection for Windows Portable Executable files demands a careful balance between detection effectiveness, computational efficiency, and analytical interpretability. This paper introduces Delphi Scanner, a static malware detection system for Windows PE files that balances efficiency with behavioral interpretation. It uses a convolutional neural network (CNN) to model Windows API sequences to classify PE and a decoupled interpretation layer based on a rule-based layer to categorize APIs into high-level malicious capabilities. Evaluated on over 190,000 Windows PE files, the system achieves 95.35% accuracy with a 1.53~MB model footprint. Robustness experiments on 5,647 out-of-distribution MalwareBazaar samples, paired packed and unpacked executables, and three adversarial manipulation strategies confirm generalization beyond the training distribution and resistance to functionality-preserving evasion techniques. Overall, th
    
[^118]: 面向高斯过程决策的在线自适应核混合方法

    Online Adaptive Kernel Mixing for Gaussian Process Decision Making

    [https://arxiv.org/abs/2609.19891](https://arxiv.org/abs/2609.19891)

    提出HACK方法，将高斯过程的核选择转化为基于专家建议的在线学习问题，利用AdaHedge在线自适应地混合候选核并保证权重收敛于最佳核，从而提升贝叶斯优化等序列决策任务在核设定错误下的鲁棒性。

    

    高斯过程（GP）被广泛用作序列决策问题（如贝叶斯优化（BO）、水平集估计（LSE）和贝叶斯主动学习（BAL））中黑盒函数的代理模型。GP的性能关键取决于核函数的选择，而标准核函数在设定错误的情况下可能导致次优决策。为解决这一问题，我们提出了HACK GPs（Hedge自适应累积核），该方法将核选择视为一个基于专家建议的在线学习问题。HACK将每个候选核视为一个GP“专家”，并使用AdaHedge基于一个损失信号在线更新专家上的分布，该损失作为专家拟合函数能力以及与任务目标一致程度的代理指标。我们提出了HACK的两种变体：（i）高斯混合和（ii）类别采样。我们建立了一般性理论保证，表明在损失间隔条件下，权重会集中于最佳核上，且由此产生的……（原文摘要此处截断）

    arXiv:2609.19891v1 Announce Type: new  Abstract: Gaussian Processes (GPs) are widely used as surrogates for black-box functions in sequential decision-making problems such as Bayesian optimization (BO), level set estimation (LSE), and Bayesian active learning (BAL). GP performance critically depends on kernels, and standard kernels can lead to suboptimal decisions under misspecification. To address this, we introduce HACK GPs (Hedge Adaptive Cumulative Kernels), a method that views kernel selection as an online learning with expert advice problem. HACK treats each candidate kernel as a GP "expert" and updates a distribution over experts online using AdaHedge, based on a loss received as a proxy for their ability to fit the function and align with the task objective. We provide two variants of HACK: (i) Mixture of Gaussians (MoG) and (ii) categorical sampling. We establish general guarantees showing that, under a loss-gap condition, the weight concentrates on the best kernel and the res
    
[^119]: Uni-LaDiR：潜在扩散统一多模态推理

    Uni-LaDiR: Latent Diffusion Unifies Multimodal Reasoning

    [https://arxiv.org/abs/2609.19878](https://arxiv.org/abs/2609.19878)

    Uni-LaDiR提出将多模态推理步骤映射到共享潜在空间，并利用扩散模型预测下一块思维标记，从而在统一的潜在表示中实现灵活的多模态推理。

    

    多模态推理要求模型在整个推理过程中利用来自多种模态的信息。然而，现有方法通常将特定模态的思维标记拼接在单一序列中，使得模型在跨模态推理时需要自行弥合表示上的差异。我们提出了Uni-LaDiR（统一潜在扩散推理器），这是一个将这些思维引入共享潜在空间进行推理的框架。统一编码器将来自不同模态的教师推理步骤映射为共享的思维标记，并通过训练来保留后续推理步骤及最终答案或动作所需的信息。由于相同的上下文可以支持多个有效的下一步推理，我们使用扩散模型基于输入和先前块来预测下一块思维标记。通过共享模型权重联合训练编码器和扩散推理器，促使思维标记既对任务有用，又……

    arXiv:2609.19878v1 Announce Type: cross  Abstract: Multimodal reasoning requires models to draw on information from multiple modalities throughout the reasoning process. Yet existing methods often concatenate modality-specific thought tokens in a single sequence, leaving the model to bridge representational differences as it reasons across modalities. We introduce Uni-LaDiR (Unified Latent Diffusion Reasoner), a framework that brings these thoughts into a shared latent space for reasoning. A unified encoder maps teacher reasoning steps from different modalities into shared thought tokens, trained to preserve the information needed for later reasoning steps and the final answer or action. Because the same context can support multiple valid next steps, we use diffusion to predict the next block of thought tokens from the input and preceding blocks. Jointly training the encoder and diffusion reasoner with shared model weights encourages thought tokens to be both useful for the task and pr
    
[^120]: AURA：用于电子邮件威胁检测的自适应不确定性路由分析

    AURA: Adaptive Uncertainty-Routed Analysis for Email Threat Detection

    [https://arxiv.org/abs/2609.19873](https://arxiv.org/abs/2609.19873)

    本文提出AURA，一种多模态电子邮件威胁检测系统，通过不确定性路由机制仅将模糊消息升级至微调的transformer编码器进行语义分析，在跨越十年真实对抗攻击数据上实现了高效的垃圾邮件与钓鱼攻击检测。

    

    电子邮件垃圾邮件和钓鱼攻击仍然是关键的安全威胁。攻击者越来越多地利用大语言模型来精心制作在语境上具有说服力的恶意消息，而现有的垃圾邮件检测系统往往难以跟上这一步伐。这些系统在多样化和不断演变的攻击场景中的泛化能力有限，导致其在实际部署后的有效性下降。本文提出了自适应不确定性路由分析，这是一种多模态电子邮件威胁检测系统，可同时分析电子邮件的内容及其嵌入的URL。AURA围绕两个层次构建：第一层量化URL分类器的预测不确定性，只有不确定的消息才会被升级到微调的transformer编码器进行语义分析。该系统在八个异构训练语料库以及两个跨越十年对抗性攻击活动的真实世界留出语料库上进行了评估。AURA达到了宏F1分数0.（原文在此处截断）

    arXiv:2609.19873v1 Announce Type: new  Abstract: Email spam and phishing attacks remain a critical security threat. Adversaries increasingly exploit large language models to craft contextually convincing malicious messages, and existing spam detection systems often struggle to keep pace. Generalization across diverse and evolving attack scenarios is limited, which reduces effectiveness once these systems are deployed in practice. This paper introduces Adaptive Uncertainty-Routed Analysis (AURA), a multimodal email threat detection system that analyzes both the content of an email and its embedded URLs. AURA is built around two layers: the first quantifies prediction uncertainty from a URL classifier, and only ambiguous messages are escalated to a fine-tuned transformer encoder for semantic analysis. The system is evaluated on eight heterogeneous training corpora together with two held-out real-world corpora spanning a decade of adversarial campaigns. AURA reaches a macro F1-score of 0.
    
[^121]: 面向药物重定位候选药物实用筛选的预训练医学表示

    Pretrained Medical Representations for the Practical Screening of Drug Repositioning Candidates

    [https://arxiv.org/abs/2609.19865](https://arxiv.org/abs/2609.19865)

    提出了一种集成层次化子词聚合、部分掩码和交叉引用机制的统一预训练框架，能更好地捕捉医疗代码的层次结构与诊疗交互，在临床事件预测任务中优于现有方法，并可用于药物重定位候选药物的实用筛选。

    

    电子健康记录和医疗理赔数据中医疗代码序列的表示学习，已在疾病预测等多种临床应用中取得成功。然而，将这种方法扩展到科学假设的发现仍面临重大挑战，原因之一是许多现有的基于BERT的模型无法充分捕捉医疗代码的层次结构以及诊断与治疗之间复杂的相互作用。为解决这些局限性，我们提出了一个新的统一预训练框架，该框架显式地集成了层次化子词标记聚合、部分掩码和交叉引用机制。所提出的模型在预训练目标和下游临床事件预测任务（包括痴呆发病和住院预测）上均持续优于现有方法。我们还进行了一项计算机模拟的药物重定位案例研究，针对……（原文此处截断）

    arXiv:2609.19865v1 Announce Type: new  Abstract: Representation learning from medical code sequences in electronic health records and medical claims data has been successful in various clinical applications, such as those regarding disease prediction. However, significant challenges remain in extending this approach to the discovery of scientific hypotheses. One reason is that many existing BERT-based models fail to adequately capture the hierarchical structure of medical codes and the complex interactions between diagnoses and treatments. To address these limitations, we propose a new unified pre-training framework that explicitly integrates hierarchical sub-token aggregation, partial masking, and cross-reference mechanisms. The proposed model consistently outperformed existing methods on both pre-training objectives and downstream clinical event prediction tasks, including the onset of dementia and hospitalization. We also conducted an in silico drug repositioning case study targetin
    
[^122]: 不确定性下多目标优化的期望超体积最大化

    Expected Hypervolume Maximization for Multiobjective Optimization under Uncertainties

    [https://arxiv.org/abs/2609.19858](https://arxiv.org/abs/2609.19858)

    该论文提出将不确定性下的多目标优化表述为关于期望超体积最大化的贝叶斯决策问题，并利用高斯过程作为可微代理模型以及基于采集函数的主动学习策略来高效求解。

    

    不确定性下的多目标优化问题通常通过取每个目标的期望值来解决。在本工作中，我们提出将此问题表述为一个贝叶斯决策问题，并依赖于超体积的期望值，该期望值需要关于一个有限的输入点集进行最大化。我们证明，只要对被支配点进行适当处理，这可以在随机优化框架中使用基于梯度的方法来完成。此外，在缺乏现成可微代码的情况下，我们提出使用高斯过程作为可微代理模型来执行优化。本工作的另一项贡献是一些主动学习策略，即通过采集函数帮助构建针对所研究的多目标优化问题精心设计的代理模型。这些策略在简单的分析问题上进行了比较。

    arXiv:2609.19858v1 Announce Type: new  Abstract: The problem of multiobjective optimization under uncertainties is often approached by taking the expectation of each objective. In this work, we propose instead to formulate this as a Bayesian decision problem and to rely on the expected value of the hypervolume, which is to be maximized with respect to a finite set of input points. We show that this can be performed using methods based on gradients in a stochastic optimization framework, provided that care is taken with respect to dominated points. Moreover, in the absence of readily available differentiable code, we propose to use Gaussian Processes as differentiable surrogate models, in order to perform the optimization. An additional contribution in this work are some active learning strategies, through acquisition functions which helps construct a surrogate model well-designed for the multiobjective optimization problem at stake. These strategies are compared on simple analytical pr
    
[^123]: 信任，但要验证工具：在自建安全回归代理上审计AI生成的RTL验证计划

    Trust, but Validate the Instrument: Auditing AI-Generated RTL Verification Plans on Authored Security-Regression Proxies

    [https://arxiv.org/abs/2609.19844](https://arxiv.org/abs/2609.19844)

    论文提出可审计框架SecTB-RTL，通过自建硬件安全回归测试发现AI生成的RTL验证计划虽被提供商接受但几乎全部无法通过生产语义验证，证明提供商模式的接受并不能等同于执行的有效性。

    

    AI生成的RTL验证计划可能满足提供商的模式，但在与可信执行的边界处失败。我们提出了SecTB-RTL，一个涵盖31个任务和124个自建硬件安全回归测试的可审计框架。确定性非AI基线在递增的资源限制下分别杀死了36、75和78个突变体。第一次确认性运行（C1-R2）在模型执行之前就失败了，因为提供商拒绝了其响应模式。在未查看结果的情况下进行仅模式修复后，单独冻结的后续运行（C1-R3）完成了1,860次调用。提供商接受了1,857个响应，但只有九个通过了生产语义验证器。生成规则和执行规则并不匹配。因此，我们将该运行保留为工具验证事件，不报告任何提示效果估计。这一事件表明，提供商或模式的接受并不能证明执行的有效性。编译和覆盖率仅是诊断指标。

    arXiv:2609.19844v1 Announce Type: cross  Abstract: AI-generated RTL verification plans can satisfy a provider schema yet fail at the boundary to trusted execution. We present SecTB-RTL, an auditable framework covering 31 tasks and 124 authored hardware-security regressions. A deterministic non-AI baseline killed 36, 75, and 78 mutants at increasing resource limits. The first confirmatory run (C1-R2) failed before model execution because the provider rejected its response schema. After a schema-only repair made without viewing outcomes, a separately frozen follow-up run (C1-R3) completed 1,860 calls. The provider accepted 1,857 responses, but only nine passed the production semantic validator. The generation and execution rules did not match. We therefore preserve the run as an instrument-validation incident and report no prompt-effect estimate. This incident shows that provider or schema acceptance does not establish execution validity. Compilation and coverage are only diagnostics; th
    
[^124]: 超越扁平化标记：基于可复用TriDim模块的结构保持式EEG解码

    Beyond Flattened Tokens: Structure-Preserving EEG Decoding with Reusable TriDim Blocks

    [https://arxiv.org/abs/2609.19842](https://arxiv.org/abs/2609.19842)

    提出可复用的TriDim模块，通过显式保留通道、补丁内样本位置和补丁位置三个结构轴并配合跨轴注意力机制，避免了传统方法将EEG标记扁平化处理的问题，实现了结构保持式的EEG解码。

    

    有效的EEG解码需要能够保留通道间组织结构、局部波形动态以及长期时间上下文的表征。现有的EEG架构通常使用独立的专用模块来捕获这些结构，或者将它们压缩成单一的标记序列，这使得在整个主干网络中难以维持它们各自的作用并协调它们之间的交互。我们提出了TriDim，一种可复用的模块，它保持表征的形状，并使EEG的三个轴保持显式：通道、每个补丁内的样本位置、以及整个记录中的补丁位置。这三个轴分别对应空间信息、短期时间信息和长期时间信息。每个TriDim模块沿各个轴应用前馈变换，并通过跨轴注意力来协调它们之间的信息交换。通过堆叠TriDim模块并结合多级三轴读出，我们构建了TriDimEEG，一个独立的（摘要内容在此处截断）

    arXiv:2609.19842v1 Announce Type: new  Abstract: Effective EEG decoding requires representations that preserve organization among channels, local waveform dynamics, and long-range temporal context. Existing EEG architectures often capture these structures using separate specialized modules or collapse them into a single token sequence, making it difficult to maintain their distinct roles and coordinate their interactions throughout the backbone. We propose TriDim, a reusable block that preserves the representation shape and keeps three EEG axes explicit: channel, sample position within each patch, and patch position across the recording. These axes correspond to spatial, short-term temporal, and long-term temporal information, respectively. Each TriDim block applies feed-forward transformations along individual axes and cross-axis attention to coordinate information exchange among them. By stacking TriDim blocks with a multi-level tri-axis readout, we construct TriDimEEG, a standalone 
    
[^125]: 通过参考策略引导正则化自我博弈中的均衡选择

    Steering Equilibrium Selection in Regularized Self-Play via the Reference Policy

    [https://arxiv.org/abs/2609.19820](https://arxiv.org/abs/2609.19820)

    该研究证明在正则化自我博弈中，通过将熵正则化参考策略锚定在目标成员上，可以有目的地引导算法收敛到纳什均衡多胞体中特定的价值等价均衡，且锚定效果跟随参考策略而非初始化。

    

    正则化自我博弈——DeepNash的Stratego对弈背后的方法家族——通过针对一个缓慢移动的熵正则化参考策略ρ进行最优响应，将双人零和博弈策略驱动至纳什均衡。当博弈具有由价值等价均衡构成的多胞体时，正则化项会默默打破平局：在均匀参考策略下，它会选择最大熵的成员，即ρ在纳什集合上的I-投影。那么，参考策略能否被有意地用来选择特定均衡？在五个可精确求解的博弈以及一个二维多胞体上，使用精确最优响应和跨独立随机种子的等价性检验，将参考策略锚定在目标成员并进行精炼，可使自我博弈收敛到该成员，平均坐标误差为0.007，中位可利用性为5×10⁻⁵，经TOST检验证明与±0.05范围内的请求等价；这种锚定在精炼过程中持续存在，并跟随参考策略而非初始化设置。选择遵循可达性加权……

    arXiv:2609.19820v1 Announce Type: new  Abstract: Regularized self-play -- the family behind DeepNash's Stratego play -- drives a two-player zero-sum policy to a Nash equilibrium by best-responding to a slowly moving, entropy-regularized reference policy $\rho$. When the game has a polytope of value-equivalent equilibria, the regularizer silently breaks the tie: with a uniform reference it selects the maximum-entropy member, the I-projection of $\rho$ onto the Nash set. Can the reference be used to choose the equilibrium on purpose? On five exactly solvable games plus a 2-D polytope, with exact best responses and equivalence tests over independent seeds, anchoring the reference at a target member and refining steers self-play to that member with mean coordinate error 0.007 at median exploitability $5\times10^{-5}$, TOST-equivalent to the request within $\pm0.05$; the anchoring persists through refinement and follows the reference, not the initialization. Selection follows the reach-weig
    
[^126]: DeliveryGym：一个具有自适应课程的长时程具身智能体规划强化学习环境

    DeliveryGym: An RL Environment for Long-Horizon Embodied Agent Planning with Adaptive Curriculum

    [https://arxiv.org/abs/2609.19801](https://arxiv.org/abs/2609.19801)

    DeliveryGym是一个面向长时程具身智能体规划的3D强化学习环境，通过连续快递员班次、持久世界动态和基于模拟器事件的轨迹奖励来评估决策成本，并采用自适应课程机制根据智能体观察到的弱点动态调整训练难度。

    

    可执行环境使大语言模型（LLM）智能体能够从其行动的后果中学习。对于具身智能体而言，这些后果不仅限于当前任务是否成功：完成一次配送可能会消耗后续工作所需的时间、精力或金钱。因此，学习规划需要能够保留这些依赖关系，并将其转化为贯穿完整轨迹反馈的环境。我们推出了DeliveryGym，这是一个用于在连续快递员班次上评估和训练智能体的3D环境。它将多模态工具交互与持久的世界动态相结合，并从模拟器事件中计算轨迹奖励，使智能体决策的成本可用于强化学习（RL）。该环境还能根据策略所观察到的弱点调整未来的训练班次，同时保持评估固定不变。在六个模型和13个城市地图的测试中，评估揭示了可靠执行分配的配送任务与……之间的差距

    arXiv:2609.19801v1 Announce Type: new  Abstract: Executable environments enable LLM agents to learn from the consequences of their actions. For embodied agents, those consequences extend beyond whether the current task succeeds: completing a delivery can consume the time, energy, or money needed for later work. Learning to plan therefore requires environments that preserve these dependencies and turn them into feedback across a complete trajectory. We introduce DeliveryGym, a 3D environment for evaluating and training agents on continuous courier shifts. It couples multimodal tool interaction with persistent world dynamics and computes trajectory rewards from simulator events, making the costs of an agent's decisions available for reinforcement learning (RL). The environment also adapts future training shifts to the policy's observed weaknesses while keeping evaluation fixed. Across six models and 13 city maps, evaluation exposes a gap between reliably executing assigned deliveries and
    
[^127]: 进化还是错觉？重新思考LLM进化搜索中的评估

    Evolution or Illusion? Rethinking Evaluation in LLM Evolutionary Search

    [https://arxiv.org/abs/2609.19799](https://arxiv.org/abs/2609.19799)

    该论文通过在种子数与迭代数的完整组合网格上系统评估三种LLM进化搜索策略，揭示了固定预算在“宽度”（更多种子）与“深度”（更多迭代）之间的最优分配方式以及策略排名都会随策略、任务和总预算显著变化，证明传统单预算设置下的评估结论并不可靠。

    

    LLM驱动的进化搜索通过启动种子并对每个种子进行迭代来发现程序。现有论文通常只报告单一的预算设置，通常是一个种子运行固定次数的迭代，并仅从这一个数据点对方法进行排名。我们证明这是不够的。我们在五个优化任务上评估了三种进化搜索策略，这些任务是此类论文常用的基准。我们在种子数和迭代数构成的完整网格上进行分析。我们的发现表明，在更多种子（宽度）和更多迭代（深度）之间分配固定预算的最佳方式会随着策略、任务和总预算的变化而变化。此外，我们还观察到策略之间的排名也会随预算变化。在其中一个任务上，单个种子下表现最差的策略在四十个种子下反而最佳；在另一个任务上，最佳迭代次数远低于实践中的常用值，因此额外的迭代深度只会浪费预算，而这些预算若用于更多种子本可转化为分数提升。我们提供了一个……

    arXiv:2609.19799v1 Announce Type: cross  Abstract: LLM-driven evolutionary search finds programs by launching seeds and iterating each one. Papers report a single budget setting, usually one seed run for a fixed number of iterations, and rank methods from that one point. We show this is not enough. We evaluate three evolutionary search strategies on five optimization tasks, commonly used by papers in the genre to report results. We run the analysis over a full grid of seeds and iterations. Our findings suggest that the best way to split a fixed budget between more seeds (width) and more iterations (depth) changes with the strategy, the task, and the total budget. Furthermore, we observe that the ranking of strategies also changes with the budget. On one task the strategy that looks worst at one seed is best at forty seeds. On another the best number of iterations is well below the value common in practice, so extra depth wastes budget that more seeds would turn into score. We provide a
    
[^128]: 基于学习的双层介质光学特性重建：利用单距离时间分辨反射测量

    Learning-Based Reconstruction of Optical Properties in Bilayered Media from Single-distance Time-Resolved Reflectance Measurements

    [https://arxiv.org/abs/2609.19786](https://arxiv.org/abs/2609.19786)

    提出了一种基于机器学习的新框架，利用蒙特卡洛模拟生成的合成DTOF数据集，从单距离时间分辨反射测量中重建双层生物介质的光学特性，克服了传统扩散方程逆求解器在结构异质性下精度不足的问题。

    

    从时域反射测量中重建分层生物介质的光学特性（特别是吸收系数和散射系数）的逆问题，对传统分析模型而言仍然是一个重大挑战。基于扩散方程的逆求解器通常难以处理结构异质性，在表层吸收和深层散射方面精度往往较差。在本工作中，我们提出了一种机器学习框架作为重建双层介质光学特性的替代方法，并将其效率与精度同基于模型的算法进行了基准对比。为了克服扩散理论和逆重建的内在近似，我们使用精确的蒙特卡洛模拟在多个源-探测器距离下生成了一个稳健的正向时间分辨飞行时间（DTOF）合成数据集。随后在该数据集上训练了机器学习流程，并针对……进行了验证。

    arXiv:2609.19786v1 Announce Type: new  Abstract: The inverse problem of reconstructing optical properties, specifically absorption and scattering coefficients, in layered biological media from time-domain reflectance measurements remains a significant challenge for traditional analytical models. Inverse solvers based on the diffusion equation often struggle with structural heterogeneity, frequently yielding poor accuracy for superficial absorption and deep-layers scattering. In this work, we propose a machine learning framework as an alternative approach to reconstruct the optical properties of a bilayered medium, benchmarking its efficiency and accuracy against model-based algorithms. To overcome the intrinsic approximations of diffusion theory and inverse reconstruction, we generated a robust synthetic dataset of forward DTOF using exact Monte Carlo simulations at multiple source-detector distances. A machine learning pipeline was then trained on this dataset and validated against st
    
[^129]: PhyRestore：物理结构化的潜在因子恢复

    PhyRestore: Physics-Structured Latent-Factor Restoration

    [https://arxiv.org/abs/2609.19776](https://arxiv.org/abs/2609.19776)

    PhyRestore提出了一种物理结构化的潜在因子恢复框架，先恢复受损的物理因子，再依据RUSLE已知的物理关系重建土壤流失时间变化，在高幅变化恢复上优于直接预测方法。

    

    当具有物理意义的输入因子存在噪声或损坏时，估计土壤流失的时间变化极具挑战性，尤其是因为相对于大量几乎无变化的位置而言，显著变化的情况十分罕见。我们通过修正通用土壤流失方程（RUSLE）来研究这一问题，并提出了PhyRestore——一个物理结构化的潜在因子恢复框架。PhyRestore并非直接预测土壤流失变化，也不是修正退化的物理估计值，而是恢复受损的物理因子，并通过已知的物理关系重建时间变化。我们在流域尺度的双时相栅格场景下对PhyRestore进行了评估，考察了降雨侵蚀力和覆盖管理因子单独受损及同时受损的情形，并将其与退化的RUSLE估计以及Direct RF、XGBoost、MLP和CNN等直接预测模型进行比较。当受损因子保持一致时，因子恢复提高了对大幅变化的恢复能力。

    arXiv:2609.19776v1 Announce Type: new  Abstract: Estimating temporal soil-loss change is challenging when physically meaningful input factors are noisy or corrupted, particularly because substantial changes are rare relative to the large number of locations exhibiting little change. We study this problem through the Revised Universal Soil Loss Equation (RUSLE) and introduce PhyRestore, a physics-structured latent-factor restoration framework. Rather than directly predicting soil-loss change or correcting a degraded physical estimate, PhyRestore restores corrupted physical factors and reconstructs temporal change through the known physical relationship. We evaluate PhyRestore in a watershed-scale bitemporal raster setting under isolated and simultaneous corruption of rainfall erosivity and cover management, comparing it with the degraded RUSLE estimate and Direct RF, XGBoost, MLP, and CNN models. Factor restoration improves high-magnitude recovery when the corrupted factors remain ident
    
[^130]: TorchCraft：通过逆转全原子结构预测器实现统一的结合剂设计

    TorchCraft: Unified binder design by inverting an all-atom structure predictor

    [https://arxiv.org/abs/2609.19770](https://arxiv.org/abs/2609.19770)

    提出了TorchCraft统一结合剂设计框架，通过逆转冻结的全原子结构预测器（基于预训练的AlphaFold 3权重）优化序列，成功设计出微结合剂、VHH、环肽和配体结合蛋白等多种结合剂，实验验证其无需事后序列重新设计即可实现有效结合。

    

    全原子结构预测器能够模拟多种分子相互作用，但如何利用其学习到的结构先验进行结合剂设计仍然具有挑战性。在此，我们提出了TorchCraft，一个统一的结合剂设计框架，它通过冻结的全原子预测器对序列logits进行优化。TorchCraft基于TorchFold实现，将置信度、接触、几何和序列先验等多个目标整合到一个共享的优化流程中，适用于微结合剂、框架条件化的VHH（单域抗体）、环肽以及配体结合蛋白的设计。利用预训练的AlphaFold 3权重，TorchCraft生成了具有实验验证结合能力的代表性微结合剂和VHH，每种形式针对四个不同靶标，且无需事后序列重新设计。计算基准测试进一步证明了该框架对环肽和配体条件化口袋设计的适用性。TorchCraft将预测器逆转方法扩展到多种结合剂形式和分子情境中。

    arXiv:2609.19770v1 Announce Type: new  Abstract: All-atom structure predictors model diverse molecular interactions, but using their learned structural priors for binder design remains challenging. Here we present TorchCraft, a unified binder-design framework that optimizes sequence logits through a frozen all-atom predictor. Implemented in TorchFold, TorchCraft combines confidence, contact, geometric, and sequence-prior objectives within a shared optimization procedure for minibinders, framework-conditioned VHHs, cyclic peptides, and ligand-binding proteins. Using pretrained AlphaFold 3 weights, TorchCraft generated representative minibinders and VHHs with experimentally measured binding across four targets in each format, without post hoc sequence redesign. Computational benchmarks further demonstrated the framework's applicability to cyclic peptides and ligand-conditioned pocket design. TorchCraft extends predictor inversion to multiple binder formats and molecular contexts, providi
    
[^131]: OceanMoE：面向长时间序列多变量海洋预报的结构化条件稀疏计算

    OceanMoE: Structured Conditional Sparse Computation for Long-Horizon Multivariate Ocean Forecasting

    [https://arxiv.org/abs/2609.19768](https://arxiv.org/abs/2609.19768)

    提出OceanMoE结构化条件稀疏混合专家框架，在统一模型中保留海洋变量共享上下文的同时，根据预测目标、空间位置和路由置信度实现自适应专门化计算，提升长时间序列多变量海洋预报能力。

    

    多变量海洋预报必须利用耦合海洋系统中的共同演化特征，同时适应不同预测变量和位置所具有的异质统计与动力特性。完全共享的模型可能缺乏处理这种异质性的灵活性，而完全独立的模型则丢弃了各变量之间共享的海洋共同背景信息。关键问题在于如何在统一模型中保留共享上下文，同时允许计算根据预测目标和局部状态进行专门化。我们提出了OceanMoE，一种结构化条件稀疏的混合专家框架，为多变量海洋预报结合了共享与专门化两种机制。OceanMoE融合跨变量信息以构建针对特定预测目标的局部表示，并利用这些表示在每个空间位置执行基于内容的条件稀疏路由，且激活专家的数量根据路由器置信度进行自适应调整。

    arXiv:2609.19768v1 Announce Type: new  Abstract: Multivariate ocean forecasting must exploit shared evolution in a coupled ocean system while adapting to the heterogeneous statistical and dynamical characteristics of different prediction variables and locations. Fully shared models may lack the flexibility to handle this heterogeneity, whereas fully independent models discard the common ocean context shared across variables. The key question is how to retain shared context in a unified model while allowing computation to specialize according to the prediction target and local state. We propose OceanMoE, a structured conditional sparse Mixture-of-Experts framework that combines sharing and specialization for multivariate ocean forecasting. OceanMoE fuses cross-variable information to construct target-specific local representations and uses them to perform content-conditioned sparse routing at each spatial location, with the number of active experts adapted to router confidence. In the d
    
[^132]: HyperAMS-Net：用于脑部疾病分类的自适应多尺度空间超图网络

    HyperAMS-Net: Adaptive Multi-Scale Spatial Hypergraph Network for Brain Disorder Classification

    [https://arxiv.org/abs/2609.19755](https://arxiv.org/abs/2609.19755)

    HyperAMS-Net是一种融合自适应多尺度卷积、超图注意力和空间-通道注意力的深度学习框架，能够有效捕捉脑部神经影像中的多尺度模式与高阶依赖关系，实现基于静息态fMRI或结构MRI的脑部疾病精准分类。

    

    基于神经影像数据对脑部疾病进行准确分类仍然具有挑战性，原因在于受试者间存在显著的异质性，以及功能连接和形态学表征中呈现的复杂多尺度模式。为了应对这些挑战，我们提出了HyperAMS-Net，这是一个利用由静息态功能磁共振成像（fMRI）或结构磁共振成像（sMRI）获得的神经影像表征进行脑部疾病分类的深度学习框架。HyperAMS-Net集成了自适应多尺度卷积、超图注意力、空间-通道注意力和自适应特征融合。具体而言，自适应多尺度卷积在多个感受野上学习数据驱动的权重，以捕捉不同尺度下的互补模式。超图注意力通过节点-超边-节点消息传递来建模学习到的特征表征之间的高阶依赖关系，而空间-通道注意力则增强判别性特征……

    arXiv:2609.19755v1 Announce Type: cross  Abstract: Accurate classification of brain disorders from neuroimaging data remains challenging because of substantial inter-subject heterogeneity and the complex multi-scale patterns present in functional connectivity and morphological representations. To address these challenges, we propose HyperAMS-Net, a deep learning framework for brain disorder classification using neuroimaging representations derived from resting-state functional MRI or structural MRI. HyperAMS-Net integrates adaptive multi-scale convolution, hypergraph attention, spatial-channel attention, and adaptive feature fusion. Specifically, adaptive multi-scale convolution learns data-driven weights over multiple receptive fields to capture complementary patterns at different scales. Hypergraph attention models higher-order dependencies among learned feature representations through node--hyperedge--node message passing, while spatial-channel attention enhances discriminative feat
    
[^133]: 联盟胜过孤立：统一异构关联数据集可提升分类器性能

    Alliance Beats Isolation: Unifying Heterogeneous Allied Datasets Improves Classifier Performance

    [https://arxiv.org/abs/2609.19748](https://arxiv.org/abs/2609.19748)

    本文提出一种将异构关联数据集的特征空间合并并利用矩阵补全构建统一数据集的方法，从而实现分类知识在数据集间的迁移，提升分类器性能。

    

    在许多应用领域，如学生辍学、保险欺诈、贷款审批和机器故障等，存在多个带标签的公开数据集，它们满足：（i）数据涉及相同类型的对象，但实际的底层对象集合互不相交；（ii）类别标签相同；（iii）数据集的特征空间在很大程度上不同（异构），仅有少量共享特征。我们将这类数据集称为“关联”数据集。对于这类数据集，无法在两个数据集上共同训练一个分类器，而在一个数据集上训练的分类器也无法在另一个数据集上进行测试。在本文中，我们提出了一种方法，将给定的一对异构关联数据集的特征空间合并为单一特征空间。随后，我们使用矩阵补全方法基于合并后的特征空间构建统一数据集。我们的假设是，合并后的表示有助于分类知识从一个数据集向另一个数据集的迁移。

    arXiv:2609.19748v1 Announce Type: new  Abstract: In many application domains, such as student dropout, insurance fraud, loan approval, and machine failures, several labelled public datasets are available where (i) data is about the same type of objects but the set of actual underlying objects are disjoint; and (ii) the class labels are same; and (iii) the feature spaces of the datasets are largely distinct (heterogeneous), with a few shared features. We call such datasets as allied. A single classifier cannot be trained on both datasets together, and one classifier trained on one dataset cannot be tested on the other. In this paper, we propose a method to merge the feature-spaces into a single feature-space for a pair of given allied heterogeneous datasets. We then use a matrix completion method to create a unified dataset based on the merged feature-space. The hypothesis is that the merged representation facilitates the transfer of classification knowledge from one dataset to another.
    
[^134]: 面向隐私保护肾结石检测的联邦学习框架

    Federated Learning Framework for Privacy-Preserving Kidney Stone Detection

    [https://arxiv.org/abs/2609.19740](https://arxiv.org/abs/2609.19740)

    该论文提出了一种结合优化YOLOv8网络的联邦学习框架，在无需共享患者数据、符合GDPR和HIPAA法规的前提下，实现了CT图像中肾结石的隐私保护式准确检测。

    

    深度学习的最新创新显著提升了医学图像的诊断水平，但这些技术通常依赖于集中式数据存储，对患者隐私和医疗数据安全构成严重威胁。为解决这一问题，本研究提出了一种联邦学习（FL）模型，该模型与优化的YOLOv8网络相结合，用于在计算机断层扫描（CT）图像中检测肾结石，同时保护患者隐私。所提出的系统可帮助各医疗机构在无需交换患者信息的情况下共同训练一个通用模型，从而确保遵守GDPR和HIPAA等数据保护法律。此外，YOLOv8中还引入了残差特征融合和DropBlock正则化等架构改进，以增强检测的鲁棒性并减少过拟合。在分布式CT数据集上进行的实验分析验证了该方法的有效性。

    arXiv:2609.19740v1 Announce Type: cross  Abstract: Recent innovations in deep learning have significantly enhanced the diagnosis of medical images, although they are based on the use of centralized data storage that pose severe threats to patient privacy and medical data security. To address this issue, this research proposes a Federated Learning (FL) model that is coupled with an optimized YOLOv8 network to detect the kidney stones on a computed tomography (CT) image and at the same time, protect privacy of the patients. The suggested system can help various medical organizations to jointly train a common model without exchanging the information about the patients. This is to ensure that data protection laws like GDPR and HIPAA are adhered to. The residual feature fusion and DropBlock regularization among other architectural improvements are also included in YOLOv8 to enhance detection robustness and minimize overfitting. Experimental analysis carried out on a distributed CT dataset d
    
[^135]: ALIBI：针对LLM恶意软件分析器的二进制输入对抗性合法性注入

    ALIBI: Adversarial Legitimacy Injection in Binary Input against LLM Malware Analyzers

    [https://arxiv.org/abs/2609.19722](https://arxiv.org/abs/2609.19722)

    该论文提出ALIBI攻击方法，通过在二进制文件中注入一个包含虚假安全产品叙述的非执行只读节区（不改变任何可执行行为），即可诱导前沿LLM恶意软件分析器将恶意样本误判为良性或大幅降低其威胁严重性评估，揭示了LLM推理能力带来的新型攻击面。

    

    大型语言模型（LLM）正作为推理组件被集成到恶意软件分类工作流中，用于总结静态证据并生成面向分析师的判定结论。本文表明，同样的推理能力也引入了一个新的攻击面。我们提出了ALIBI，一种针对前沿LLM恶意软件分析器的语义掩护故事攻击。ALIBI在编译后的二进制文件中添加一个小的、不会被执行的只读节区，其中包含一段连贯但虚假的安全产品叙述，且不改变导入表或可执行行为。它并不直接向模型下达指令，而是将可疑证据重新框定为良性终端安全工具的预期行为。在一个包含50个恶意样本的冻结PE数据集上，该攻击载荷在Gemini 2.5 Pro上将35个基线判定为恶意的样本中的30个翻转为良性，而GPT-5.5 Pro和Claude Opus 4.7则产生显著的严重性降级和置信度大幅下降，即使判定标签仍然保留（原文摘要在此处截断）。

    arXiv:2609.19722v1 Announce Type: cross  Abstract: Large language models are being integrated into malware triage workflows as reasoning components that summarize static evidence and produce analyst-facing verdicts. This paper shows that the same reasoning capability introduces a new attack surface. We present ALIBI, a semantic cover story attack against frontier LLM-based malware analyzers. ALIBI adds a small, non-executed read-only section to a compiled binary, containing a coherent but false security product narrative, without altering imports or executable behavior. Instead of issuing direct instructions to the model, it reframes suspicious evidence as expected behavior of a benign endpoint security tool. On a frozen PE set of 50 malicious samples, the payload flips 30 of the 35 baseline-malicious samples to benign on Gemini 2.5 Pro, while GPT-5.5 Pro and Claude Opus 4.7 produce substantial severity downgrades with significant confidence reductions even when verdict labels are pres
    
[^136]: 学会自己的思考：抽象token课程学习

    Learn Your Own Thoughts: Abstract Token Curriculum

    [https://arxiv.org/abs/2609.19717](https://arxiv.org/abs/2609.19717)

    提出了抽象token课程学习（ATC）框架，无需直接监督或手动设计草稿板，即可通过逐步增加问题复杂度训练模型在连续表示空间中自发形成内部抽象思维，并从理论和实验上证明了其相对于以往连续思维训练方法的优势。

    

    大语言模型（LLMs）通过利用思维链（CoT）作为思考中间阶段的草稿板，已经获得了卓越的推理能力。然而，CoT技术需要对思考token进行显式监督，这需要丰富的、特定任务的数据。在这项工作中，我们提出了抽象token课程学习（Abstract Token Curriculum, ATC），这是一种新颖的课程学习框架，能够在没有直接监督或手动草稿板设计的情况下，引出有效的连续中间表示。ATC通过一系列分布逐渐增加问题复杂度，训练模型在连续表示空间中发展出内部的抽象“思维”。本文为ATC的优势及其相对于以往训练连续思维方法的长处提供了理论和实验证据。理论上，我们证明了使用ATC在单层softmax注意力机制下学习奇偶函数时……

    arXiv:2609.19717v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have achieved remarkable reasoning capabilities by utilizing chain-of-thought (CoT) as a scratchpad for intermediate stages of thinking. However, CoT techniques require explicit supervision on thinking tokens, which requires rich, task-specific data. In this work, we propose Abstract Token Curriculum (ATC), a novel curriculum learning framework that elicits effective continuous intermediate representations without direct supervision or manual scratchpad design. ATC gradually increases problem complexity through a sequence of distributions, training the model to develop internal abstract ``thoughts'' in the continuous representation space. This paper provides both theoretical and experimental evidence for the benefits of ATC and its advantages over previous methods for training continuous thoughts. Theoretically, we show that for learning parity functions with single-layer softmax attention using ATC, attent
    
[^137]: 比值比汤普森采样：基于对比的多臂老虎机的规范与设计指南

    Odds-Ratio Thompson Sampling: A Specification and Design Guide for Contrast-Based Multi-Armed Bandits

    [https://arxiv.org/abs/2609.19709](https://arxiv.org/abs/2609.19709)

    本文提出比值比汤普森采样（OR-TS），通过保留臂间对数比值对比的联合后验分布并在每个批次重新拟合共同水平，解决了批量多臂老虎机中因共享基线水平漂移而导致绝对奖励率记忆失效的问题，显著降低了后悔值。

    

    批量式多臂老虎机按照服务自身的节奏进行更新，通常的实现方式会在两次更新之间保留每个臂的绝对奖励率。当共享的水平在批次之间发生移动时，这种记忆就会失效，即使各臂之间的比较关系可能并未改变。比值比汤普森采样（OR-TS）则改为保留对数比值对比的联合后验分布，并在每个批次中重新拟合共同水平，将其边缘化处理。本文对这一更新过程进行了规范化说明，将其嵌入一个带有两个控制参数的贝叶斯老虎机智能体中——衰减参数控制过去证据在每次更新后的保留程度，激进度参数控制信念转化为分配的敏锐程度——并将其与绝对率记忆方法进行对比评估。在86个公开A/B测试序列中，水平的波动幅度大约是对比幅度的二十五倍。在预先设定的合成环境中，移动的水平使绝对率记忆产生的后悔值高出五倍，并使最佳臂的流量占比低于多数……

    arXiv:2609.19709v1 Announce Type: new  Abstract: Batched multi-armed bandits update on a service's own schedule, and the usual implementation carries each arm's absolute reward rate from one update to the next. When the shared level moves between batches, that memory goes stale even though the comparisons between arms may not have. Odds-Ratio Thompson Sampling (OR-TS) instead carries the joint posterior over log-odds contrasts and fits the common level afresh in every batch, marginalizing it out. This paper specifies that update, places it inside a Bayesian bandit agent with two controls, decay for how much past evidence survives an update and aggressiveness for how sharply belief becomes allocation, and evaluates it against absolute-rate memory. Across 86 public A/B series the level varies about twenty-five times more than the contrast. In prespecified synthetic environments a moving level costs absolute-rate memory five times the regret and leaves the best arm below a majority of tra
    
[^138]: 异构物联网系统中基于观点动力学的联邦学习联盟形成

    Opinion Dynamics-based Coalition Formation for Federated Learning in Heterogeneous IoT Systems

    [https://arxiv.org/abs/2609.19695](https://arxiv.org/abs/2609.19695)

    该论文将联邦学习中的客户端联盟形成建模为Hegselmann-Krause有界置信观点动力学过程，通过在本地权重空间中基于欧氏距离和余弦相似度置信准则形成联盟并在联盟层面聚合，解决了异构物联网系统中统计异质性导致FedAvg无法捕获客户端特定模式的问题。

    

    联邦学习（FL）能够在异构物联网部署中实现隐私保护的设备端训练，例如智慧城市水表计量网络，其中每个智能水表观测一个家庭特定的用水时间序列。在这种统计异质性下，标准的联邦平均方法将不相似的本地模型平均为单一全局模型，可能无法捕获客户端特定的模式。我们通过直接在本地权重空间中形成客户端联盟并在联盟层面进行聚合来解决这一问题。在扩展先前基于权重的联盟形成方案的基础上，我们将联盟形成建模为作用于本地权重的Hegselmann-Krause（HK）有界置信观点动力学过程，并开发了基于欧氏距离和余弦相似度置信准则的HK交互变体。该框架被应用于短期用水量预测。

    arXiv:2609.19695v1 Announce Type: new  Abstract: Federated learning (FL) enables privacy-preserving, on-device training across heterogeneous Internet-of-Things (IoT) deployments such as smart-city water-metering networks, where each smart meter observes a household-specific consumption time series. Under such statistical heterogeneity, the standard Federated Averaging (FedAvg) aggregation averages dissimilar local models into a single global model that may fail to capture client-specific patterns. We address this by forming client coalitions directly in the local-weight space and aggregating at the coalition level. Extending a prior weight-driven coalition-formation scheme, we model coalition formation as a Hegselmann-Krause (HK) bounded-confidence opinion-dynamics process acting on the local weights, and develop variants of the HK interaction based on Euclidean-distance and cosine-similarity confidence criteria. The framework is applied to short-term water-consumption forecasting with
    
[^139]: 物理世界模型中：守恒换来稳定性，因式分解换来反事实能力

    Conservation Buys Stability and Factoring Buys Counterfactuals in Physical World Models

    [https://arxiv.org/abs/2609.19674](https://arxiv.org/abs/2609.19674)

    该论文证明物理世界模型的两种失效需要不同的结构性补救：用辛积分器演化学习到的能量可保持保守动力学几何结构、使长时程展开在训练范围100倍内保持稳定，而用显式线性因式分解编码物理耦合则能使模型泛化到从未见过的干预情形。

    

    学习得到的模拟器能够准确地复现其训练条件，但一旦条件发生改变，可能会以两种截然不同的方式失效。在长时间展开时，小误差不断累积，最终导致轨迹偏离物理上合理的行为；而在对某个物理参数进行干预时，模型可能继续遵循训练时所见到的规律，而非被干预后的规律。我们证明这两种失效需要不同的结构性补救措施。用辛积分器来演化学习到的能量函数，可以保持保守动力学的几何结构，使展开过程在长达训练范围100倍的时间内保持有界且物理上有意义，而同等容量的预测器、能量正则化的预测器以及经过调优的神经ODE则会发散。相比之下，通过显式的线性因式分解来编码物理耦合，能够使模型遵循从未见过的该耦合的符号方向，而无限制的参数化则仍然无法做到这一点。

    arXiv:2609.19674v1 Announce Type: new  Abstract: A learned simulator can reproduce its training conditions accurately yet fail in two distinct ways once those conditions change. Over long rollouts, small errors accumulate until the trajectory drifts away from physically plausible behavior; under an intervention on a physical parameter, the model may continue to follow the law seen during training rather than the intervened one. We show that these two failures require different structural remedies. Evolving a learned energy with a symplectic integrator preserves the geometry of the conservative dynamics and keeps rollouts bounded and physically meaningful for up to $100\times$ the training horizon, while equal-capacity predictors, an energy-regularized predictor, and a tuned neural ODE diverge. By contrast, encoding the physical coupling through an explicit linear factorization enables the model to follow a never-seen sign of that coupling, whereas an unrestricted parameterization remai
    
[^140]: When2Think：面向高效混合推理模型的难度感知长度控制学习

    When2Think: Learning Difficulty-Aware Length Control for Efficient Hybrid Reasoning Models

    [https://arxiv.org/abs/2609.19671](https://arxiv.org/abs/2609.19671)

    When2Think提出了一个混合推理后训练框架，通过实例级难度感知控制（IDAC）机制根据问题难度动态分配计算资源，解决了大型推理模型对简单问题过度思考、对困难问题思考不足的系统性低效问题。

    

    大型推理模型在复杂任务上表现出强大性能，但存在系统性的低效问题：它们经常对简单问题过度思考，而对困难问题思考不足。现有基于统一长度惩罚或固定路由的方法会产生“效率税”，即以困难实例的准确性损失为代价来换取简单实例上计算量的减少。我们将高效推理形式化为一个实例自适应的计算分配问题，并提出了When2Think——一个混合推理的后训练框架，能够根据问题难度动态分配计算资源。我们的方法引入了实例级难度感知控制（IDAC），这是一种奖励塑形机制，利用预先计算的参考统计数据（准确率和token使用量）来调节推理深度。结合基于验证器的奖励和批内标准化优势，IDAC能够在无需学习奖励模型或在线参考的情况下实现稳定的无评论家优化。

    arXiv:2609.19671v1 Announce Type: new  Abstract: Large Reasoning Models (LRMs) achieve strong performance on complex tasks but exhibit systematic inefficiency: they often overthink easy problems and underthink hard ones. Existing approaches based on uniform length penalties or rigid routing incur an efficiency tax, trading reduced computation on easy instances for accuracy loss on hard instances. We formulate efficient reasoning as an instance-adaptive computation allocation problem and propose When2Think, a post-training framework for hybrid reasoning that dynamically allocates computation based on problem difficulty. Our method introduces Instance-level Difficulty-Aware Control (IDAC), a reward-shaping mechanism that leverages pre-computed reference statistics (accuracy and token usage) to regulate reasoning depth. Combined with verifier-based rewards and batch-wise standardized advantages, IDAC enables stable critic-free optimization without learned reward models or online reference
    
[^141]: CoRe：面向多变量时间序列预测的一致性与关系对齐

    CoRe: Coherence and Relational Alignment for Multivariate Time Series Forecasting

    [https://arxiv.org/abs/2609.19670](https://arxiv.org/abs/2609.19670)

    CoRe提出了一种与模型无关的学习目标，通过频率一致性损失和低秩关系图损失显式保留未来轨迹的时间一致性与变量间关系一致性，无需引入可训练参数即可增强现有多变量时间序列预测模型。

    

    直接预测已成为多变量时间序列预测的标准范式，因为它可以在单次前向传播中预测完整的未来时间范围。然而，其训练目标通常仍被分解为诸如均方误差（MSE）之类的逐点误差。这类目标提供了稳定的监督，但并未显式地保留未来轨迹的结构：每个变量内部的时间一致性和变量之间的关系一致性都可能被削弱。我们提出CoRe，一种面向直接多变量预测的模型无关学习目标。CoRe用两个输出空间约束取代逐点监督：一个对齐预测谱与目标谱的频率一致性损失，以及一个在由目标数据导出的PCA子空间中匹配采样成对差异的低秩关系图损失。所得的目标不引入任何可训练参数，并且只需更改损失函数即可应用于现有的预测骨干网络。

    arXiv:2609.19670v1 Announce Type: new  Abstract: Direct forecasting has become a standard paradigm for multivariate time-series forecasting because it predicts the full future horizon in a single pass. However, its training objective is often still decomposed into pointwise errors such as MSE. Such objectives provide stable supervision, but they do not explicitly preserve the structure of the future trajectory: temporal coherence within each variable and relational consistency across variables can both be weakened. We propose CoRe, a model-agnostic learning objective for direct multivariate forecasting. CoRe replaces pointwise supervision with two output-space constraints: a frequency coherence loss that aligns predicted and target spectra, and a low-rank relational graph loss that matches sampled pairwise differences in a target-derived PCA subspace. The resulting objective introduces no trainable parameters and can be applied to existing forecasting backbones by changing only the los
    
[^142]: EmbodiedMind：面向高效具身智能的自适应数据筛选与前缀树强化学习

    EmbodiedMind: Adaptive Data Curation and Prefix-Tree Reinforcement Learning for Efficient Embodied Intelligence

    [https://arxiv.org/abs/2609.19659](https://arxiv.org/abs/2609.19659)

    本文提出EmbodiedMind训练范式，通过自适应数据筛选（RSFT与IR-GRPO）和前缀树强化学习三大协同阶段，解决了具身基础模型训练中样本利用率低、梯度贡献不均衡和长时程信用分配困难三大问题，以更少的数据和计算资源实现了最先进的平均性能。

    

    训练具身基础模型通常需要大规模数据集和大量计算资源，但往往存在三个关键局限：(1) 由于低信息量样本导致样本利用效率低下；(2) 异构任务之间的梯度贡献不均衡；(3) 长时程规划中存在严重的信用分配问题，即轨迹级奖励会不加区分地惩罚所有token。为解决这些问题，我们提出了一种高效的训练范式，通过战略性数据选择和分层策略优化实现了最先进的平均性能。我们的方法由三个协同阶段组成。首先，基于拒绝采样的微调（RSFT）过滤掉低信息量样本，以建立稳健的行为先验，同时防止分布坍缩。其次，迭代拒绝GRPO（IR-GRPO）采用按难度分层的任务特定队列，以保持……（摘要内容不完整，后文涉及前缀树强化学习用于解决长时程规划中的信用分配问题）

    arXiv:2609.19659v1 Announce Type: cross  Abstract: Training embodied foundation models typically requires massive-scale datasets and extensive computational resources, yet often suffers from three critical limitations: (1) inefficient sample utilization due to low-informative samples; (2) imbalanced gradient contributions across heterogeneous tasks; and (3) severe credit assignment problem in long-horizon planning, where trajectory-level rewards indiscriminately penalize all tokens. To address these issues, we propose an efficient training paradigm that achieves state-of-the-art average performance through strategic data selection and hierarchical policy optimization. Our approach consists of three synergistic stages. First, Rejection Sampling-based Fine-Tuning (RSFT) filters out low-informative samples to establish robust behavioral priors while preventing distributional collapse. Second, Iterative Rejection GRPO (IR-GRPO) employs task-specific queues stratified by difficulty to keep 
    
[^143]: PrefixBench-H100：H100大语言模型服务中前缀重用与首令牌生成时间的特性表征

    PrefixBench-H100: Characterizing Prefix Reuse and Time-to-First-Token in H100 LLM Serving

    [https://arxiv.org/abs/2609.19657](https://arxiv.org/abs/2609.19657)

    本文提出PrefixBench-H100，一个在单个NVIDIA H100上可复现的基准测试与测量框架，通过受控合成轨迹与真实工作负载系统表征前缀重用和首令牌生成时间的特性，揭示vLLM和TensorRT-LLM等推理运行时中前缀重用收益的边界条件。

    

    重复的提示前缀在LLM服务工作负载中日益普遍，常见于系统提示词、模板化检索增强生成流水线、智能体框架以及多轮对话中。诸如vLLM和TensorRT-LLM等现代推理运行时提供了跨请求重用已计算KV缓存状态的机制，然而前缀重用何时能在当代加速器上显著提升服务性能，以及其收益何时会受到调度、缓存粒度、并发性或内存压力的限制，目前仍不明确。本文提出了PrefixBench-H100，一个可复现的基准测试与测量框架，用于在单个NVIDIA H100上表征前缀重用特性。PrefixBench-H100将受控的合成轨迹与聊天式及检索式工作负载相结合，并在匹配的工作负载条件下评估两种广泛使用的LLM服务运行时。该基准测试改变了共享前缀长度、后缀多样性、

    arXiv:2609.19657v1 Announce Type: cross  Abstract: Repeated prompt prefixes are increasingly common in LLM serving workloads, appearing in system prompts, templated retrieval-augmented generation pipelines, agent frameworks, and multi-turn conversations. Modern inference runtimes such as vLLM and TensorRT-LLM provide mechanisms for reusing previously computed KV-cache state across requests, yet it remains unclear when prefix reuse materially improves serving performance on contemporary accelerators and when its benefits are limited by scheduling, cache granularity, concurrency, or memory pressure.   This paper presents PrefixBench-H100, a reproducible benchmark and measurement framework for characterizing prefix reuse on a single NVIDIA H100. PrefixBench-H100 combines controlled synthetic traces with chat-style and retrieval-style workloads, and evaluates two widely used LLM serving runtimes under matched workload conditions. The benchmark varies shared-prefix length, suffix diversity,
    
[^144]: 神经湍流闭合与切向耗散的适定性

    Well-posedness of neural turbulence closures and tangent dissipation

    [https://arxiv.org/abs/2609.19647](https://arxiv.org/abs/2609.19647)

    该文为神经湍流闭合建立了适定性理论，证明全局切向耗散——可通过强制非负切向扩散的精确积分构造或对切向反应违反施加惩罚来促进——能够保证解的存在性、唯一性以及后验误差与先验残差之间的全局逆敏感度界。

    

    神经湍流闭合定义了一个新的边值问题 R(U)=N(U)+F(U)=0，其耦合雅可比矩阵为 J(U)=N′(U)+F′(U)，其中 N 为原始平均流算子，F 为学习得到的闭合项。我们建立了全局切向耗散的两个结论：对于单调的原始算子，由原始算子与闭合项共同提供的正的一致裕度保证了存在性、唯一性，以及将后验解误差与先验残差联系起来的全局逆敏感度界；对于一般的原始算子，耗散性闭合项不会使切向耗散变差，但仅凭这一点并不能保证唯一性。切向耗散同时依赖于扩散与反应两项，我们研究了两种互补的促进方式：（1）一种精确积分构造，在不约束反应项的情况下强制非负切向扩散；（2）在采样状态处对切向反应违反施加惩罚。

    arXiv:2609.19647v1 Announce Type: cross  Abstract: A neural turbulence closure defines a new boundary-value problem, $R(U)=N(U)+F(U)=0$, with a coupled Jacobian $J(U)=N'(U)+F'(U)$, where $N$ is the original mean-flow operator and $F$ the learned closure. We establish two consequences of global tangent dissipation. For a monotone original operator, a positive uniform margin supplied by the original operator and closure together guarantees existence, uniqueness and a global inverse-sensitivity bound relating a posteriori solution error to the a priori residual. For a general original operator, a dissipative closure cannot worsen tangent dissipation, but this alone does not guarantee uniqueness. Tangent dissipation depends on both diffusion and reaction. We study two complementary ways to promote it: (1) an exact-integral construction enforcing non-negative tangent diffusion while leaving reaction unconstrained, and (2) a penalty on tangent-reaction violations at sampled states. Tangent d
    
[^145]: Croissant的策略配置文件：将“拒绝”作为数据集的一种属性

    A Policy Profile for Croissant: Refusal as a Property of the Dataset

    [https://arxiv.org/abs/2609.19640](https://arxiv.org/abs/2609.19640)

    该论文为机器学习数据集描述符Croissant提出了一个策略配置文件，通过封闭的五种运算符集合和完整定义的决策程序，使数据集能够声明其允许的操作及条件，让门控系统仅凭描述符即可做出并记录访问决策。

    

    Croissant是机器学习数据集事实上的机器可读描述符：基于schema.org的JSON-LD格式。自1.1版本起，它还承载了数据使用条件，并推荐使用DUO和ODRL来表达这些条件。然而没有任何版本说明这些条件该如何被评估：没有决策程序、没有评估成本的上界、对于实现无法评估的条件没有规定输出结果、没有检查内容的记录，也没有关于与调用方权限组合的任何规定。我们补足了这缺失的一半。一个增量式配置文件让数据集能够声明它所允许的操作以及允许这些操作的条件，这些条件建立在一个封闭的五种运算符集合之上，其决策程序被完整给出，因此一个门控系统可以仅凭描述符本身做出决策，并记录它所检查的内容。两个语料库对其进行了评估，且二者的证据被分开保存。三个对真实nf-core流水线进行门控的描述符给出了部署结果：来自配置文档的决策与门控的原生描述符记录相匹配

    arXiv:2609.19640v1 Announce Type: new  Abstract: Croissant is the de facto machine-readable descriptor for ML datasets: JSON-LD over schema.org. Since version 1.1 it also carries data use conditions, recommending DUO and ODRL for them. What no version specifies is how any of them is evaluated: no decision procedure, no bound on evaluation cost, no outcome for a condition an implementation cannot evaluate, no record of what was checked, and nothing on composition with caller-side authority. We supply that half. An additive profile lets a dataset declare the operations it admits and the conditions under which it admits them, over a closed set of five operators whose decision procedure is given in full, so a gate decides from the descriptor alone and records what it checked. Two corpora evaluate it and their evidence is kept apart. Three descriptors that gated a real nf-core pipeline give the deployment result: decisions from a profile document match the gate's native descriptor record fo
    
[^146]: 复杂度拐点：面向代码生成可靠性的提示侧结构复杂度指数

    The Complexity Kink: A Prompt-Side Structural Complexity Index for Code-Generation Reliability

    [https://arxiv.org/abs/2609.19616](https://arxiv.org/abs/2609.19616)

    该论文提出一个生成前评分的六维提示侧结构复杂度指数，发现代码生成通过率在复杂度分数上存在非单调断点，但该断点并非通用的失败临界值，会随任务类型固定效应等因素发生移动。

    

    从生成代码中测量的复杂度依赖于失败情况：一个困难的提示可能产生一个简短的失败程序，从而被赋予较低的输出复杂度。我们提出了一个六维度的提示侧结构复杂度指数，在生成之前进行评分，并与正确性保持分离。我们从初步单评分者量表的六个区间中选择了5,000个Python提示。四个面板外的LLM评分者对锁定的提示重新评分，得到19,997条评分记录；在四个评分齐全的4,998个提示上，综合评分者间信度为ICC = 0.872。我们对每个提示评估21个模型，共产生105,000个生成结果。在未经调整的均值池化分析中，通过率在综合分数13.75处出现非单调断点，该分数及以下的通过率为79.9%，以上为87.6%。这并非一个通用的失败临界点。任务类型固定效应将断点移至10.75，并将区间差距从7.6个百分点缩小至2.1个百分点。构建框架控制将其移至8.50，原始差距为……

    arXiv:2609.19616v1 Announce Type: new  Abstract: Complexity measured from generated code is failure-dependent: a difficult prompt can yield a short failing program and be assigned low output complexity. We introduce a six-dimension prompt-side structural-complexity index scored before generation and kept separate from correctness. We select 5,000 Python prompts across six bands of a preliminary single-rater rubric. Four out-of-panel LLM raters rescore the locked prompts, giving 19,997 score rows; composite inter-rater reliability is ICC = 0.872 on the 4,998 prompts with all four ratings. We evaluate 21 models per prompt, yielding 105,000 generations. In the unadjusted mean-pooled analysis, pass rate has a nonmonotone breakpoint at composite 13.75, with 79.9% at or below and 87.6% above. This is not a universal failure cutoff. Task-type fixed effects shift the breakpoint to 10.75 and cut the regime gap from 7.6 to 2.1 points. A construction-frame control shifts it to 8.50 with a raw gap
    
[^147]: TacSushi：面向灵巧寿司操作的触觉接地世界-动作建模

    TacSushi: Tactile-Grounded World-Action Modeling for Dexterous Sushi Manipulation

    [https://arxiv.org/abs/2609.19613](https://arxiv.org/abs/2609.19613)

    提出TacSushi，一种触觉接地的世界-动作建模策略，通过特征级门控融合指尖触觉并利用失败试验的未来后果预测进行监督，实现了形变、遮挡和不确定接触条件下的灵巧寿司操作。

    

    灵巧的食物操作需要在形变、遮挡和不确定接触条件下的控制。我们提出TacSushi，一种基于触觉接地、基于Cosmos3的世界-动作策略，它在作用于当前观测的同时，从记录的未来后果中学习。骨干网络编码当前的RGB图像、语言和手部状态，特征级门控融合将指尖触觉特征融入动作表示。在训练期间，一个以示范动作块为条件的解码器预测记录的未来视觉观测、任务进度、相对接触风险和触觉摘要；该解码器在部署时被移除。失败的试验提供后果监督，但其动作被排除在模仿学习之外。我们在340次成功和50次失败的真实机器人试验上训练TacSushi，并在600次独立测试中比较六种方法，涵盖三个分布内任务和两个分布外食材变体。为了评估食品质量……

    arXiv:2609.19613v1 Announce Type: cross  Abstract: Dexterous food manipulation requires control under deformation, occlusion, and uncertain contact. We present TacSushi, a tactile-grounded, Cosmos3-based world-action policy that learns from recorded future consequences while acting on current observations. The backbone encodes current RGB, language, and hand state, and feature-wise gated fusion incorporates fingertip tactile features into the action representation. During training, a decoder conditioned on demonstrated action chunks predicts logged future visual observations, task progress, relative contact risk, and tactile summaries; this decoder is removed at deployment. Failed trials provide consequence supervision, but their actions are excluded from imitation. We train TacSushi on 340 successful and 50 failed real-robot trials and compare six methods in 600 separate rollouts across three in-distribution tasks and two out-of-distribution ingredient variants. To assess food quality
    
[^148]: 输出空间假设：面向张量程序的枚举等价性检查

    The Output-Space Hypothesis: Enumerative Equivalence Checking for Tensor Programs

    [https://arxiv.org/abs/2609.19611](https://arxiv.org/abs/2609.19611)

    该论文提出“输出空间假设”，通过翻转量词的新颖符号执行策略，从“针对单个输入检查所有输出位置”转变为“针对所有输入检查单个输出位置的等价性”，从而更持续有效地发现张量程序优化中的错误。

    

    深度学习模型中使用的张量程序是优化的首要目标，因为微小的性能提升可以在训练或推理工作负载中产生重大影响。然而，此类优化非常复杂，可能产生微妙的错误。传统上，当针对参考实现的随机输入差分测试未能发现错误时，就假定程序是正确的。然而，这些程序的输入是大规模张量，发现错误可能需要生成出现概率极低的输入，且其值之间需要满足精确的关系。我们提出了一种通过翻转量词来更持续地发现错误的新方法。与其生成单个输入并检查所有输出张量位置的等价性，何不检查单个输出张量位置在所有输入下的等价性呢？我们在一个名为Dirigo的系统中通过一种新颖的符号执行策略实现了这一想法。我们证明了Dirigo可以……

    arXiv:2609.19611v1 Announce Type: cross  Abstract: Tensor programs, as used in deep learning models, are a prime target for optimization, as small performance improvements can have a large impact across training or inference workloads. However, such optimizations are complicated and can produce subtle bugs. Traditionally, correctness is assumed when differential testing against a reference on random inputs fails to reveal bugs. However, the inputs to these programs are massive tensors, and finding bugs can require generating extremely low likelihood inputs with precise relationships among their values.   We propose a novel way to find bugs more consistently by flipping the quantifiers. Rather than generating a single input and checking all output tensor locations for equivalence, what if you could check a single output tensor location's equivalence for all inputs? We implement this idea in a system, \dirigo, by using a novel symbolic execution strategy. We demonstrate that \dirigo can 
    
[^149]: DeltaSelect：面向编码智能体的经济型A/B测试方法

    DeltaSelect: Affordable A/B Testing for Coding Agents

    [https://arxiv.org/abs/2609.19607](https://arxiv.org/abs/2609.19607)

    DeltaSelect是一种开源方法，通过皮尔逊相关性筛选出单次运行即可可靠代表完整基准性能的任务子集，为编码智能体的开发迭代提供低成本、可重复的A/B测试方案。

    

    编码智能体基准测试是为广泛而全面的比较而设计的，而非为频繁的开发决策服务。单次运行结果存在波动，完整测试套件成本高昂，且基准测试的运行框架可能与实际使用的框架不一致。在对DeepSWE已发表试验的重采样分析中，只有19.5%的任务（113个任务中的22个）其第五百分位皮尔逊相关系数与完整基准测试性能达到至少0.50。本文提出了DeltaSelect，这是一种开源方法，它利用皮尔逊相关系数识别那些单次运行结果能够稳定追踪完整基准测试性能的任务，通过线性回归将分数化的验证器结果映射到统一的评分标准，并在给定的美元预算内选择固定的任务集合。DeltaSelect旨在用于开发过程中重复进行的基线与候选方案比较，而非用于模型排名。在一个gpt-5.6-luna低推理案例研究中，DeltaSelect被用于修订自定义技能和指令。在13个评估……

    arXiv:2609.19607v1 Announce Type: cross  Abstract: Coding-agent benchmarks are built for broad and comprehensive comparisons, not frequent development decisions. Individual runs vary, full suites are expensive, and the benchmark harness may differ from the harness used in practice. In a resampling analysis of DeepSWE's published trials, only 19.5% of tasks (22 of 113) had a fifth-percentile Pearson correlation of at least 0.50 with full-benchmark performance. The paper presents DeltaSelect, an open-source method that identifies tasks whose one-run results consistently track full-benchmark performance using Pearson correlation, maps fractional verifier results to a common score using linear regression, and selects a fixed task set within a dollar budget. DeltaSelect is intended for repeated baseline-versus-candidate comparisons during development, not model rankings. In a gpt-5.6-luna low-reasoning case study, DeltaSelect was used to revise custom skills and instructions. Across 13 eval
    
[^150]: 思维链熵作为可靠性信号：一项预注册复现研究

    Chain-of-Thought Entropy as a Reliability Signal: A Preregistered Reproduction

    [https://arxiv.org/abs/2609.19606](https://arxiv.org/abs/2609.19606)

    本预注册独立复现研究证实，大语言模型思维链熵轨迹的形状（而非总熵下降幅度）是预测答案正确性的可靠信号，形状信号在四个开源模型上成功复现，而幅度信号则因实验设置而异。

    

    这项实证研究是对Zhao在2026年报告的分离效应的独立复现。大型语言模型思维链熵轨迹的形状可以预测最终答案是否正确，而其总熵下降的幅度则不能。这一分离效应值得复现，因为幅度部分基于单一模型在单一随机种子下对300个问题的单次运行，而形状部分在两个基准测试的完整规模上以及第二个模型家族上均有报告。该复现在任何验证性运行之前已在OSF注册，用四个开源权重模型遍历完整的GSM8K和MATH-500基准测试集，其中包括一个原始研究未测试过的推理蒸馏模型。结果表明，形状信号得到了复现，而幅度信号则因设置而异。在锚定模型上，单调链与非单调链之间的准确率差距在GSM8K上为+9.6个百分点，在MATH-500上为+27.5个百分点，而秩相关……（摘要原文在此处截断）

    arXiv:2609.19606v1 Announce Type: new  Abstract: This empirical study is an independent reproduction of the dissociation Zhao reported in 2026. The shape of a large language model's chain-of-thought entropy trajectory predicts whether the final answer is correct, while the magnitude of its total entropy drop does not. The dissociation merits reproduction because the magnitude half rests on a single 300-problem run with one model at one seed, while the shape half was reported at full scale on both benchmarks and on a second model family. Registered at OSF before any confirmatory run, the reproduction crosses the complete GSM8K and MATH-500 benchmark test sets with four open-weight models including one reasoning-distilled model of a kind the original did not test. The shape signal replicates. The magnitude signal divides by setting. On the anchor model the accuracy gap between monotone and non-monotone chains is +9.6 percentage points on GSM8K and +27.5 on MATH-500, while the rank correl
    
[^151]: CliniCIRCA：一个用于从原始电子健康档案叙述中构建心理健康患者纵向历程的模块化大语言模型框架

    CliniCIRCA: A Modular LLM Framework for Constructing Longitudinal Mental Health Patient Journeys from Raw EHR Narratives

    [https://arxiv.org/abs/2609.19585](https://arxiv.org/abs/2609.19585)

    CliniCIRCA是首个无需事件级时间戳即可从非结构化出院总结中对临床事件进行时间分类的多阶段大语言模型框架，通过临床医生参与纠错生成黄金标准标签，并支持基于时间线的患者历程总结。

    

    在心理健康护理领域，对患者历程的推理是临床医生的一项关键任务。然而，这些历程涵盖了生物、心理和社会事件的纵向进展，往往分散在不同的非结构化文本叙述中，使得时间信息的恢复极具挑战性。我们提出了CliniCIRCA，一个用于日历锚定、感知不精确性的临床编年史重建的多阶段大语言模型框架。据我们所知，CliniCIRCA是首个在没有事件级时间戳的情况下，对非结构化出院总结中的临床事件进行时间分类的方法。我们从14,882条MIMIC-III心理健康入院记录出发，首先构建了一个包含52份出院总结的基准数据集，CliniCIRCA在其上生成了15,891个带时间标记的事件。在通过临床医生参与的评估纠正了629个错误后，我们生成了经过验证的黄金标准标签。最后，纠正后的时间线驱动了一个基于时间的总结阶段……

    arXiv:2609.19585v1 Announce Type: cross  Abstract: In mental health care, reasoning over patient journeys is a key task for clinicians. Yet these journeys, encompassing a longitudinal progression of biological, psychological, and social events, are often spread across disparate unstructured text narratives, making temporal recovery challenging. We present CliniCIRCA, a multi-stage LLM framework for Calendar-anchored, Imprecision-aware Reconstruction of Clinical Annals. To our knowledge, CliniCIRCA is the first to temporally classify clinical events across unstructured discharge summaries without event-level timestamps. From 14,882 MIMIC-III mental health admissions, we first construct a benchmark of 52 discharge summaries on which CliniCIRCA produces 15,891 temporally tagged events. After correcting 629 errors based on a clinician-in-the-loop evaluation, we produce verified gold-standard labels. Finally, the corrected timelines drive a temporally grounded summarization stage that compr
    
[^152]: FedFIbOS：基于Fisher信息重要性的异构联邦学习最优子模型方法

    FedFIbOS: Fisher Importance based Optimal Submodelling for Heterogeneous Federated Learning

    [https://arxiv.org/abs/2609.19559](https://arxiv.org/abs/2609.19559)

    提出FedFIbOS方法，通过最小化子模型掩蔽误差推导出基于Fisher信息的原则性参数选择准则，为异构联邦学习中最优子模型的选择提供了理论依据，解决了现有启发式方法缺乏收敛理论支撑以及部分客户端参与引入估计偏差的问题。

    

    异构联邦学习要求具有不同计算能力的客户端协同训练一个全局模型，其中每个客户端训练一个受容量约束的子模型。现有方法使用启发式重要性度量来选择子模型参数——其中最突出的是参数幅度——但缺乏关于这些度量为何能支持收敛的理论依据。我们识别出一个根本性的空白：现有的参数选择准则在收敛框架中缺乏理论基础，且部分客户端参与会在Fisher分数中引入额外的估计效应。我们提出了FedFIbOS：一种面向异构联邦学习的基于Fisher重要性的最优子模型方法，该方法通过最小化子模型掩蔽误差推导出原则性准则，并利用Fisher信息进行参数选择。

    arXiv:2609.19559v1 Announce Type: new  Abstract: Heterogeneous federated learning requires clients with diverse computational capacities to collaboratively train a global model, where each client trains a capacity-constrained submodel. Existing methods select submodel parameters using heuristic importance measures---most prominently parameter magnitude---without theoretical justification for why these measures support convergence. We identify a fundamental gap: existing parameter selection criteria lack theoretical grounding in the convergence framework, partial client participation introduces additional estimation effects in the Fisher scores. We propose \textbf{FedFIbOS}: Fisher Importance-based Optimal Submodelling for heterogeneous federated learning, using Fisher Information in a principled criterion derived from minimizing submodel masking error. %We formally establish when magnitude selection is equivalent to Fisher selection fail under non-IID heterogeneous federated learning. 
    
[^153]: 用于可扩展贝叶斯推断的压缩主动子空间

    Compressed Active Subspaces for Scalable Bayesian Inference

    [https://arxiv.org/abs/2609.19539](https://arxiv.org/abs/2609.19539)

    本文提出压缩主动子空间（CAS）方法，通过结构化等距嵌入先将模型参数映射到压缩空间再构建主动子空间，大幅降低内存开销，使大规模模型的可扩展贝叶斯推断成为可能。

    

    主动子空间方法通过识别对模型输出影响最大的参数方向并沿这些方向进行推断，为高维模型中的预测不确定性量化提供了一个框架。然而，主动子空间的构建需要存储大量全维度的模型梯度，随着模型规模的增大，这一开销变得难以承受。我们通过提出压缩主动子空间（CAS）来解决这一局限性，这是一种可扩展的方法，首先利用结构化等距嵌入将模型参数映射到压缩空间，然后在该降维后的参数化空间中构建主动子空间。我们的方法大幅降低了主动子空间构建所需的内存，使得标准主动子空间方法变得不切实际的大型模型的贝叶斯推断成为可能。我们在规模不断增大的神经网络上演示了CAS的可扩展性，同时保持了预测……

    arXiv:2609.19539v1 Announce Type: cross  Abstract: Active subspace methods provide a framework for quantifying predictive uncertainty in high-dimensional models by identifying and performing inference along parameter directions that have the greatest influence on the model output. However, the construction of active subspaces requires storing many full-dimensional model gradients, which becomes prohibitive as model size increases. We address this limitation by proposing Compressed Active Subspaces (CAS), a scalable approach that first maps the model parameters to a compressed space using a structured isometric embedding and then constructs the active subspace within this reduced parameterization. Our approach substantially reduces the memory required for active subspace construction and enables Bayesian inference for large models where standard active subspace methods become impractical. We demonstrate the scalability of CAS on neural networks of increasing size while maintaining predi
    
[^154]: 下一词元泛函估计

    Next-token functional estimation

    [https://arxiv.org/abs/2609.19529](https://arxiv.org/abs/2609.19529)

    针对时间依赖数据，本文提出“留窗口法”估计量用于估计下一词元泛函（如惊喜概率、测试误差等），克服了留一法在时间依赖下不一致的缺陷，并在平稳 β-混合过程下以参数速率收敛。

    

    假设我们观察到一个长度为 n+1 的随机变量序列的前 n 个点，并希望估计关于未观测的最后一个点与 n 个已观测训练点的经验测度的某个泛函。此类“下一词元泛函”包括：下一个词元是新词的概率（也称为惊喜概率）、下一个词元与训练点之间最小距离的尾部概率，以及在已观测点上训练的分类器的测试误差。所有这些量传统上都是通过留一法（leave-one-out）进行估计的，但该方法在时间依赖性下是不一致的。我们提出了一种“留窗口法”（leave-a-window-out）估计量，它在构造经验测度之前删除每个索引之后长度为 τ 的窗口，并在 τ=1 时退化为留一法。在自然假设下，我们证明该估计量的误差对于任何平稳 β-混合过程都以参数速率衰减。

    arXiv:2609.19529v1 Announce Type: cross  Abstract: Suppose we observe the first $n$ points of a sequence of random variables having length $n+1$, and wish to estimate a functional of the unobserved final point and the empirical measure of the $n$ observed training points. Such next-token functionals include the probability that the next token is novel (also known as the surprise probability), the tail probability of the minimum distance between the next token and training points, and the test error of a classifier trained on the observed points. All of these quantities are classically estimated by the leave-one-out method, which is inconsistent under temporal dependence. We propose a leave-a-window-out estimator, which deletes a window of length $\tau$ after each index before forming the empirical measure and reduces to leave-one-out at $\tau = 1$. Under natural assumptions, we show that the error of our estimator decays at a parametric rate for any stationary $\beta$-mixing process th
    
[^155]: 基于快速树搜索的自我改进

    Self Improvement via Fast Tree-search

    [https://arxiv.org/abs/2609.19526](https://arxiv.org/abs/2609.19526)

    提出SIFT框架，利用LLM作为裁判对候选补丁进行两两比较并通过正则化Bradley-Terry模型聚合实力分数，大幅降低了自我改进循环中候选修改的评估成本，使编码智能体在严格预算约束下实现高效的自我改进。

    

    编码智能体能够递归地修改自身的实现，从而形成一个自我改进的循环。尽管先前的研究表明这种方法可以提升编码基准测试的性能，但现有方法成本高昂且计算密集。我们提出了一个简单且样本高效的自我改进框架，能够在严格的预算约束下显著提升编码性能。我们发现候选自我修改的评估是主要的运行时瓶颈，因为先前的方法需要让修改后的智能体重新运行一部分基准测试任务来估计其有效性，这一过程非常耗时。我们提出了基于快速树搜索的递归自我改进方法，该方法在下游任务评估的基础上引入了LLM作为裁判的信号，对候选补丁进行两两比较，胜负记录通过正则化的Bradley-Terry模型进行聚合，所得的实力分数用于驱动基于排名的父节点选择。

    arXiv:2609.19526v1 Announce Type: new  Abstract: Coding agents can recursively modify their own implementations, forming a loop of self-improvement. While prior work shows this can boost performance on coding benchmarks, existing approaches are costly and compute-intensive. We introduce a simple, sample-efficient self-improvement framework that significantly improves coding performance under strict budget constraints. We identify evaluation of candidate self-modifications as the main runtime bottleneck since prior approaches estimate their effectiveness by re-running a subset of benchmark tasks with the modified agent, which is time-consuming. We introduce Recursive Self Improvement via Fast Tree-search (SIFT), which augments these downstream task evaluations with an LLM-as-a-judge signal that performs pairwise comparisons between candidate patches, where the win-loss record is aggregated with a regularized Bradley-Terry model, and the resulting strength scores drive rank-based parent 
    
[^156]: LSTM-UT与循环深度Transformer在元胞自动机上的研究

    LSTM-UT and Recurrent-Depth Transformers on Cellular Automata

    [https://arxiv.org/abs/2609.19521](https://arxiv.org/abs/2609.19521)

    该论文提出带有有界门控记忆的LSTM-UT模型，并通过Rule 30元胞自动机和延迟回忆实验证明，仅依赖当前隐藏状态的Block Universal Transformer在深度外推上优于依赖扩展注意力缓存的CoTFormer，而LSTM-UT进一步同时提升了深度外推与延迟回忆能力。

    

    循环深度Transformer通过反复应用共享计算来工作，但它们在跨步骤保留信息的方式上有所不同。我们比较了三种模型：Block Universal Transformer（BUT），它仅携带当前隐藏状态；CoTFormer，它还保留不断扩展的注意力缓存；以及一种新的带有有界门控记忆的LSTM Universal Transformer（LSTM-UT）。在Rule 30元胞自动机上，BUT比CoTFormer更可靠地外推到未见过的循环深度，尽管其准确率最终会下降。状态与缓存干预实验表明，CoTFormer的失败源于两者的交互作用：修正当前状态可以暂时恢复准确率，而保留的历史信息可能会破坏这一修正。在延迟回忆任务中，尽管BUT缺乏对过去状态的直接访问，它仍然优于CoTFormer；CoTFormer无法可靠地选取所请求的缓存表示。LSTM-UT在这些基线之上同时提升了深度外推和延迟回忆能力……

    arXiv:2609.19521v1 Announce Type: new  Abstract: Recurrent-depth Transformers apply shared computation repeatedly, but differ in how they retain information across steps. We compare a Block Universal Transformer (BUT), which carries only its current hidden state; CoTFormer, which also retains an expanding attention cache; and a new LSTM Universal Transformer (LSTM-UT) with bounded gated memory. On Rule 30 cellular automata, BUT extrapolates to unseen recurrent depths more reliably than CoTFormer, although its accuracy eventually degrades. State and cache interventions show that CoTFormer's failure depends on their interaction: correcting the current state can temporarily restore accuracy, while retained history can undermine that correction. In a delayed-recall task, BUT also outperforms CoTFormer despite lacking direct access to past states; CoTFormer does not reliably select the requested cached representation. LSTM-UT improves both depth extrapolation and delayed recall over these b
    
[^157]: 长时程智能体架构：层级、时钟节拍与级联智能

    An Architecture for Long-Horizon Agents: Levels, Ticks and Cascaded Intelligence

    [https://arxiv.org/abs/2609.19519](https://arxiv.org/abs/2609.19519)

    本文提出一种由时间尺度层级记忆、时钟节拍驱动的自主行动和失败后才升级的级联智能组成的分层架构，使语言模型智能体能够在不遗忘的前提下持续运行数天甚至数周。

    

    语言模型智能体越来越多地被要求执行跨越数天或数周的工作，例如运营修复或研究计划。这类任务的生命周期超出了任何上下文窗口、任何进程以及人能够持续关注的任何时间间隔。在本文中，我们论证了一个长时程智能体必须能够持续运行而不遗忘，才能实现持续学习。这种能力存在于模型周围的运行框架（harness）中，而非模型本身。我们从长时程场景中推导出七个瓶颈，并用一个由三部分组成的分层架构加以解决：(i) 按时间尺度索引的层级，每一层维护一个有界的文件来概括下一层的内容；(ii) 作为自主行动单元的时钟节拍；(iii) 级联智能，即只有在审查失败后才将工作升级至更强大的模型。我们报告了一项为期十天的实验，其中基于该架构构建的智能体复现了一项已发表的强化学习研究。

    arXiv:2609.19519v1 Announce Type: new  Abstract: Language-model agents are increasingly asked to carry out work spanning days or weeks, such as an operations remediation or a research programme. Such a task outlives any context window, any process and any interval at which a person can attend. In this paper, we argue that a long-horizon agent must run continually without forgetting before it can learn continually. This ability lies in the harness around the model rather than in the model itself. We derive seven bottlenecks from the long-horizon setting and answer them with a hierarchical architecture of three parts: (i) levels indexed by time scale, each keeping a bounded file summarising the level below; (ii) a clocked tick as the unit of autonomous action; and (iii) cascaded intelligence, where work is escalated to a more capable model only after failing review. We report on a ten-day campaign in which an agent built on this architecture reproduced a published reinforcement-learning 
    
[^158]: LLM即改进者：将验证转化为更优候选

    LLM-as-an-Improver: Turning Verification into Better Candidates

    [https://arxiv.org/abs/2609.19515](https://arxiv.org/abs/2609.19515)

    本文提出“验证—修复—重选”（VRR）方法，不再将验证器的反馈仅用于排序，而是利用其修复优胜候选、生成新思路方案并重新选择最终答案，从而在推理时提升LLM的代码生成与推理性能。

    

    基于验证器的选择方法通过生成多个候选解并使用验证器从中选出最有希望的解来提升LLM的性能。然而，现有方法通常仅将验证视为一次排序步骤，一旦固定候选池完成评估，便丢弃验证所提供的反馈信息。本文探究验证是否也能反过来改进候选集本身。为此，我们提出LLM-as-an-Improver（LLM即改进者）框架，并提出“验证—修复—重选”（Verify–Repair–Reselect, VRR）方法，利用验证反馈来生成并重选经过改进的候选解。VRR在保留初始优胜者的同时，有条件地生成三个互补的备选方案：优胜者的修复版本、次优者的修复版本，以及基于全新思路的解法。该方法仅利用推理时的信息过滤无效和重复的候选解，然后在原始评估标准下重新选出最终答案。在多种模型以及代码生成与推理基准任务上……

    arXiv:2609.19515v1 Announce Type: new  Abstract: Verifier-based selection improves LLM performance by generating multiple candidate solutions and using a verifier to select the most promising one. However, existing methods typically treat verification only as a ranking step and discard its feedback once a fixed candidate pool has been evaluated. In this paper, we ask whether verification can also improve the candidate set itself. To this end, we introduce LLM-as-an-Improver and propose Verify--Repair--Reselect (VRR), which uses verification feedback to generate and reselect improved candidates. VRR retains the initial winner while conditionally generating three complementary alternatives: repaired versions of the winner and runner-up, and a solution based on a new approach. It filters invalid and duplicate candidates using only inference-time information and then reselects the final answer under the original evaluation criteria. Across diverse models and code-generation and reasoning b
    
[^159]: QVAC Genesis III：一个用于高效语言模型预训练的大规模高质量开放合成STEM语料库

    QVAC Genesis III: A Large-Scale, High-Quality Open Synthetic STEM Corpus for Efficient Language Model Pre-Training

    [https://arxiv.org/abs/2609.19513](https://arxiv.org/abs/2609.19513)

    提出了QVAC Genesis III——一个包含1914.3亿token的开放STEM合成语料库，通过以弱学生模型信号驱动的双重生成策略（将失败转化为纠正性解释、将成功扩展为对比性选项级推理），为token预算受限的小模型高效预训练提供了高价值数据。

    

    高质量预训练数据是面向边缘AI和端侧部署的教育及STEM专用语言模型的关键瓶颈，在这些场景中token预算受到严格限制。尽管各大机构在私有语料库上训练越来越大的模型，但开放生态系统中缺乏能够以高效方式为小模型提供高单token学习价值的STEM导向合成数据集。为填补这一空白，我们推出了QVAC Genesis III，这是一个拥有1914.3亿token、以STEM为核心的多领域合成语料库，涵盖19个领域，并包含多个难度级别和不同的教育风格。QVAC Genesis III通过一种双重生成策略构建，该策略以一个弱小的边缘规模学生模型作为信号进行针对性教师蒸馏：学生的失败被转化为纠正性解释，而其成功则被扩展为针对所有答案选项的对比性选项级推理。我们进一步引入了LLM作为解析器的机制……（原文摘要在此处截断）

    arXiv:2609.19513v1 Announce Type: new  Abstract: High-quality pre-training data is a critical bottleneck for educational and STEM-specific language models targeting edge AI and on-device deployment where token budgets are tightly constrained. While major organizations train ever-larger models on private corpora, the open ecosystem lacks STEM-focused synthetic datasets that deliver high per-token learning value efficiently for small models. To address this gap, we introduce QVAC Genesis III, a 191.43B-token, STEM-focused multi-domain synthetic corpus covering 19 domains across several difficulty levels and different educational styles. QVAC Genesis III is built via a dual generation strategy that performs targeted teacher distillation using a weak edge-scale student model as signal: the student's failures are converted into corrective explanations, while its successes are expanded into contrastive option-level reasoning over all answer choices. We further introduce an LLM-as-a-parser ev
    
[^160]: 零重要性：解耦可解释机器学习中的相关性概念

    Null importance: Disentangling relevance for interpretable machine learning

    [https://arxiv.org/abs/2609.19511](https://arxiv.org/abs/2609.19511)

    本文提出基于“零重要性”的统一框架，厘清了可解释机器学习中特征重要性所隐含的多种本质上不同的相关性概念，并在算法公平性和基因组扰动建模中展示了这些区分的实际意义。

    

    特征重要性是可解释机器学习的核心，但“重要性”一词涵盖了多种本质上不同的相关性概念。我们基于零重要性发展了一个统一的视角：即在特定相关性概念下，对特征在总体层面何时无关的刻画。我们考虑了由边际和条件统计相关性、预测风险、函数不变性以及因果效应所产生的零重要性的标准概念，并展示了这些概念如何回答不同的科学问题。我们在两个这种区分尤为关键的应用中阐释了该框架：在算法公平性中，常见的公平性准则对应于不同的零重要性概念；在基因组扰动建模中，不同的相关性概念会导致关于预测模型学到了什么的不同结论。该框架连接了三个方面

    arXiv:2609.19511v1 Announce Type: cross  Abstract: Feature importance is central to interpretable machine learning, but the term "importance" encompasses several fundamentally different notions of relevance. We develop a unified perspective based on null importance: a population-level characterization of when a feature is irrelevant under a specified notion of relevance. We consider standard notions of null importance arising from marginal and conditional statistical relevance, predictive risk, functional invariance, and causal effects, and show how these notions answer different scientific questions. We illustrate the framework in two applications in which the distinction is particularly consequential: algorithmic fairness, where common fairness criteria correspond to different notions of null importance, and genomic perturbation modeling, where different notions of relevance lead to different conclusions about what a prediction model has learned. The framework connects three aspects 
    
[^161]: 样本数量还不够：候选生成策略决定了大语言模型测试时扩展的能耗与性能

    Sample Count Is Not Enough: Candidate-Generation Strategy Shapes the Energy and Performance of LLM Test-Time Scaling

    [https://arxiv.org/abs/2609.19499](https://arxiv.org/abs/2609.19499)

    该论文指出，仅用候选样本数量N不足以刻画大语言模型测试时扩展的真实开销，相同采样预算在不同批量调度策略下的能耗与性能表现差异显著。

    

    测试时扩展可以通过生成并组合多个候选响应来提升大语言模型的推理能力。在基于采样的方法中，推理预算通常由生成的候选数量N来描述。然而，N只说明了生成了多少候选，并未说明它们是如何被执行的。同样的候选预算既可以在一次批量生成调用中产生，也可以拆分为多次串行调用，每次使用较小的批量。我们首先使用Phi-3-mini和Qwen2.5-1.5B在500个GSM8K提示上研究了增大N对推理准确率的影响。正如预期，将N从1增加到8使Phi-3-mini的准确率提高了8.4个百分点，使Qwen2.5-1.5B提高了18.4个百分点。然而，仅凭准确率无法反映使用更大候选预算所带来的系统成本。因此，我们固定N=8，比较了四种生成调度方式：1x8、2x4、4x2和8x1，其中axb表示a次生成调用、每次生成b个候选。我们测量了……

    arXiv:2609.19499v1 Announce Type: new  Abstract: Test-time scaling can improve large language model reasoning by generating and combining multiple candidate responses. In sampling-based methods, the inference budget is often described by the number of generated candidates, N. However, N tells us how many candidates are generated, not how they are executed. The same candidate budget can be produced in one batched generation call or split across several sequential calls with smaller batch sizes. We first study the effect of increasing N on reasoning accuracy using Phi-3-mini and Qwen2.5-1.5B on 500 GSM8K prompts. As expected, increasing N from 1 to 8 improves accuracy by 8.4 percentage points for Phi-3-mini and 18.4 points for Qwen2.5-1.5B. However, accuracy alone does not show the systems cost of using a larger candidate budget. We therefore fix N = 8 and compare four generation schedules: 1x8, 2x4, 4x2, and 8x1, where axb denotes a generation calls with b candidates per call. We measur
    
[^162]: 以采样成本进行搜索：近乎即时的潜在空间贝叶斯优化

    Search at the Cost of Sampling: Nearly-Instant Latent Space Bayesian Optimization

    [https://arxiv.org/abs/2609.19476](https://arxiv.org/abs/2609.19476)

    该论文提出一种在高维潜在空间的球形域上使用线性代理模型的贝叶斯优化方法，通过利用球面对称性推导出近乎闭式的求解方案，相比最先进的基线方法实现至少100倍的加速，使贝叶斯优化在虚拟筛选成本较低的场景下变得实用。

    

    生成模型在许多从头发现（de novo discovery）流程中日益成为核心，在这些流程中，设计被大规模生成，并通过虚拟筛选来过滤，从而确定一组待实验验证的候选方案。虽然贝叶斯优化（BO）天然适合这种场景，因为它利用过去的评估来指导未来的提议，但当虚拟筛选相对廉价时，其序贯决策所需的计算开销会成为瓶颈。我们通过利用线性模型与高维潜在变量所集中的球形域约束的独特组合，使贝叶斯优化在这一机制下变得实用。我们在近期论证线性代理模型有效性的工作基础上，推导出利用球面对称性的代理建模和采集函数问题的近乎闭式解。结果是相比最先进的基线方法至少获得100倍的加速，同时性能相当或有所提升。

    arXiv:2609.19476v1 Announce Type: new  Abstract: Generative models are increasingly central to many de novo discovery pipelines, in which designs are generated at scale and filtered through virtual screens to determine a set of candidates to experimentally validate. While Bayesian optimization (BO) is a natural fit for this setting, as it uses past evaluations to guide future proposals, the computational overhead required for its sequential decision-making becomes a bottleneck when virtual screens are relatively cheap. We make BO practical in this regime by exploiting the unique combination of a linear model constrained to a spherical domain where high-dimensional latents concentrate. We build off recent work justifying the use of linear surrogates, while deriving nearly closed-form solutions to the surrogate modelling and acquisition problems that exploit spherical symmetry. The result is at least a 100x speedup over state-of-the art baselines, with matching or improved performance ac
    
[^163]: 界面之外的安全性：通过大语言模型的潜在状态检测有害内容

    Safety Beyond the Interface: Detecting Harm via Latent States in Large Language Models

    [https://arxiv.org/abs/2609.19472](https://arxiv.org/abs/2609.19472)

    该研究通过从LLaMA-3.1-8B内部激活值中训练仅1260万参数的轻量级MLP探针来检测有害提示，实现了与规模大1000倍的防护模型相当的检测性能（F1最高达99%），同时显著降低了延迟和计算成本。

    

    自主系统日益依赖大语言模型（LLM），然而围绕这些模型构建的安全基础设施会引入延迟和计算开销，这限制了它们在资源受限、时间关键型部署场景中的实用性。现有的外部防护栏模型对模型的内部运作机制一无所知，造成了根本性的安全保障缺口。我们提出这样的问题：模型本身是否已经知道内容何时有害？我们从LLaMA-3.1-8B中提取内部激活值，并训练轻量级MLP分类器探针（1260万参数）来检测有害提示。在WildJailbreak、Beavertails和AEGIS 2.0数据集上的评估显示，我们的探针分别达到了99%、83%和84%的F1分数，与比其规模大1000倍的防护模型相比具有竞争力，同时大幅降低了延迟和计算成本。

    arXiv:2609.19472v1 Announce Type: new  Abstract: Autonomous systems increasingly rely on Large Language Models (LLMs) yet the safety infrastructure surrounding these models introduces latency and compute overhead. This limits utility in resource-constrained, time-critical deployments. Existing external guardrail models remain blind to the model's internal workings, creating a fundamental assurance gap. We ask: does the model already know when the content is harmful? We extract activations from LLaMA-3.1-8B and train lightweight MLP classifier probes (12.6M parameters) to detect harmful prompts. Evaluated on WildJailbreak, Beavertails, and AEGIS 2.0, our probes achieve F1 scores of 99%, 83%, and 84%, respectively competitive with 1000x larger guard models while cutting latency and compute costs.
    
[^164]: 基于领域知识增强的农业信息神经网络

    Enhanced Agriculture-informed Neural Network by Domain Knowledge

    [https://arxiv.org/abs/2609.19466](https://arxiv.org/abs/2609.19466)

    提出了一种融合肥料扩散、土壤呼吸和充水孔隙度等领域知识的神经-机理混合框架KAINN，提升了农业氧化亚氮（N2O）排放预测的准确性、可解释性和跨环境条件的泛化能力。

    

    准确预测农业中的氧化亚氮（N2O）排放对于评估环境影响和推动可持续农业发展至关重要。然而，由于N2O排放源于土壤性质、气候、生化过程和管理措施之间的复杂相互作用，且高质量观测数据有限，预测仍然十分困难。深度学习模型虽然能够捕捉非线性关系，但往往缺乏物理可解释性，并且在不同环境条件下的泛化能力可能较差。我们提出了知识增强的农业信息神经网络（KAINN），这是一种神经-机理混合框架，通过融入关于肥料扩散、土壤呼吸和充水孔隙度的领域知识，对农业信息神经网络进行了扩展。我们在多个生长季节和多种输入特征配置下，采用CNN、LSTM和Transformer架构对KAINN进行了评估。结果表明……

    arXiv:2609.19466v1 Announce Type: new  Abstract: Accurate prediction of nitrous oxide (N2O) emissions from agriculture is important for assessing environmental impacts and supporting sustainable farming. However, prediction remains difficult because N2O emissions result from complex interactions among soil properties, climate, biochemical processes, and management practices, while high-quality observations are limited. Deep learning models can capture nonlinear relationships but often lack physical interpretability and may generalize poorly across environmental conditions. We propose the Knowledge-enhanced Agriculture-informed Neural Network (KAINN), a hybrid neural-mechanistic framework that extends the Agriculture-informed Neural Network by incorporating domain knowledge about fertilizer diffusion, soil respiration, and water-filled porosity. We evaluate KAINN using CNN, LSTM, and Transformer architectures across multiple growing seasons and input-feature configurations. The results 
    
[^165]: 强化学习后训练下语言模型的组合推理

    Compositional Reasoning in Language Models under Reinforcement Learning Post-Training

    [https://arxiv.org/abs/2609.19465](https://arxiv.org/abs/2609.19465)

    本文提出依赖图框架形式化语言模型的组合推理，并揭示强化学习后训练中的不对称迁移现象——分解技能训练难以迁移到组合任务，而组合任务训练则更容易迁移回分解任务。

    

    组合推理对现实世界问题求解至关重要：由于训练数据必然有限，模型必须通过以新方式组合已学技能来实现泛化。尽管强化学习（RL）等后训练方法已显著提升了语言模型（LM）的推理能力，但其对组合推理的影响仍知之甚少。我们提出了一个依赖图框架来形式化组合推理，得到了复杂度递增的三个组合性层级。在实证方面，我们以数据结构任务实例化该框架，此类任务提供确定性的奖励计算和清晰的组合结构。我们发现了一个一致的“分解到组合”的不对称性：分解技能训练并不能可靠地迁移到组合任务，而组合任务训练则更容易反向迁移到分解任务。我们为这种不对称性提供了理论解释。

    arXiv:2609.19465v1 Announce Type: new  Abstract: Compositional reasoning is critical for real-world problem solving: since training data is necessarily limited, models must generalize by composing learned skills in new ways. While post-training methods such as reinforcement learning (RL) have substantially improved the reasoning abilities of language models (LMs), their effects on compositional reasoning remain less well understood. We propose a dependency-graph framework to formalize compositional reasoning, yielding three levels of compositionality with increasing complexity. Empirically, we instantiate this framework with data-structure tasks, which provide deterministic reward computation and clear compositional structure. We find a consistent decomposed-to-composed asymmetry: decomposed-skill training does not reliably transfer to composed tasks, whereas composed-task training transfers more readily back to decomposed tasks. We provide theoretical explanation for this asymmetry, a
    
[^166]: 锐度感知最小化（SAM）提高细菌拉曼光谱数据分类准确率，实现便携式诊断

    Sharpness-Aware Minimization (SAM) Improves Classification Accuracy of Bacterial Raman Spectral Data Enabling Portable Diagnostics

    [https://arxiv.org/abs/2609.19453](https://arxiv.org/abs/2609.19453)

    本论文将锐度感知最小化（SAM）应用于细菌拉曼光谱分类任务，在无需复杂预处理的情况下将分类准确率提升高达10.5%，并增强了模型在有限数据集上的泛化能力，为便携式抗生素耐药性诊断奠定了基础。

    

    预计到2050年，抗菌素耐药性每年将夺走1000万人的生命，而资源有限的地区受影响最为严重。拉曼光谱是一种新型的病原体诊断方法，有望在几小时内完成快速、便携的抗生素耐药性检测，而使用金标准方法则需要数天时间。然而，当前的拉曼光谱分析算法存在以下两个问题：1）在跨不同患者人群的有限数据集上无法很好地泛化；2）由于必须进行非平凡的预处理步骤（如特征提取，这对于缓解拉曼光谱数据的低质量特性至关重要），增加了算法的复杂性。在本工作中，我们使用锐度感知最小化（SAM）来解决这些局限性，在临床细菌分离株分类任务中增强模型在多种超参数设置下的泛化能力。我们证明SAM在单个（原文摘要在此处截断）上实现了高达10.5%的准确率提升。

    arXiv:2609.19453v1 Announce Type: new  Abstract: Antimicrobial resistance is expected to claim 10 million lives per year by 2050, and resource-limited regions are most affected. Raman spectroscopy is a novel pathogen diagnostic approach promising rapid and portable antibiotic resistance testing within a few hours, compared to days when using gold standard methods. However, current algorithms for Raman spectra analysis 1) are unable to generalize well on limited datasets across diverse patient populations and 2) require increased complexity due to the necessity of non-trivial pre-processing steps, such as feature extraction, which are essential to mitigate the low-quality nature of Raman spectral data. In this work, we address these limitations using Sharpness-Aware Minimization (SAM) to enhance model generalization across a diverse array of hyperparameters in clinical bacterial isolate classification tasks. We demonstrate that SAM achieves accuracy improvements of up to 10.5% on a sing
    
[^167]: GLAMDRING：基于中枢模式发生器（CPG）强化学习的步态学习与机器人形态协同设计

    GLAMDRING: Gait Learning And Morphology co-Design via Reinforcement LearnING of CPGs

    [https://arxiv.org/abs/2609.19452](https://arxiv.org/abs/2609.19452)

    该论文提出GLAMDRING框架，通过强化学习训练Hopf振荡器CPG步态控制器，并同时优化四足机器人的形态设计（连杆几何结构与执行器配置），以满足速度、功率和载荷等约束条件。

    

    机器人正从结构化的工厂车间走向灾难现场、行星表面和农田等非结构化环境，而对于这些场景，合适的机器人往往尚未存在。我们提出了GLAMDRING，一个能够为运动任务合成最优机器人并同时学习驱动该机器人的控制器的框架。在给定前向速度界限、单执行器功率预算、执行器库和有效载荷要求等规格条件下，GLAMDRING返回一个匹配的四足机器人形态（连杆几何结构及各关节执行器）以及一个基于Hopf振荡器的中枢模式发生器（CPG）步态策略。我们根据目标设计指标对可行设计进行排名，包括最大速度、最小运输成本或最大载荷裕度。由于身体与运动相互耦合，最优形态决定了机器人的驱动方式，而最优步态又依赖于物理身体。我们训练少量……

    arXiv:2609.19452v1 Announce Type: cross  Abstract: Robots are moving out of the structured factory floor and into unstructured environments such as disaster sites, planetary surfaces, and agricultural fields, for which the right robot often does not yet exist. We present GLAMDRING, a framework that synthesizes the optimal robot for a locomotion task and, jointly, learns the controller that drives it. For the given specifications of forward-velocity bounds, a per-actuator power budget, an actuator library, and a payload requirement, GLAMDRING returns a matched quadruped morphology (link geometry and per-joint actuators) and a Hopf-oscillator Central Pattern Generator (CPG) gait policy. We rank feasible designs against a target design objective, viz., maximum speed, minimum Cost of Transport (CoT), or max Payload Margin. Because body and locomotion are coupled, the optimal morphology dictates how a robot is driven, while optimal gait depends on the physical body. We train a small number 
    
[^168]: 从模型到系统：高效多模态学习的全面综述

    From Models to Systems: A Comprehensive Survey of Efficient Multimodal Learning

    [https://arxiv.org/abs/2609.19445](https://arxiv.org/abs/2609.19445)

    本综述首次提出涵盖模型、算法和系统三个层次的结构化高效多模态学习分类体系，并系统综合了跨层协同设计的方法论，以应对“效率-效用-隐私”的根本性权衡。

    

    多模态模型的快速扩张暴露了计算、内存和部署方面的严峻瓶颈，催生了高效多模态学习（EML）作为关键研究前沿的兴起。尽管进展迅速，但对效率在学习栈中体现在何处、如何体现的统一理解仍然碎片化。本综述通过引入首个结构化的从模型到系统的分类体系，对EML领域进行了系统化梳理。我们从300多篇开创性工作中提炼出见解，归纳为三个层次——模型、算法和系统——分别解决架构精简、执行优化和硬件感知编排问题。超越纯粹的分类回顾，我们对这些层次之间的垂直协同进行了方法论层面的综合，阐明了跨层协同设计如何影响根本性的“效率-效用-隐私”权衡。通过一个综合性案例研究……

    arXiv:2609.19445v1 Announce Type: cross  Abstract: The rapid expansion of multimodal models has surfaced formidable bottlenecks in computation, memory, and deployment, catalyzing the rise of Efficient Multimodal Learning (EML) as a pivotal research frontier. Despite intensive progress, a cohesive understanding of what, how, and where efficiency is manifested across the learning stack remains fragmented. This survey systematizes the EML landscape by introducing the first structured, model-to-system taxonomy. We distill insights from over 300 seminal works into three hierarchical levels--model, algorithm, and system--addressing architectural parsimony, execution refinement, and hardware-aware orchestration, respectively. Moving beyond a purely categorical review, we offer a methodological synthesis of the vertical synergies between these layers, elucidating how cross-layer co-design contributes to the fundamental "Efficiency-Utility-Privacy" trade-off. Through an integrative case study o
    
[^169]: 基于大语言模型的丰富辅助信息贝叶斯优化

    Bayesian Optimization with Rich Auxiliary Information via LLMs

    [https://arxiv.org/abs/2609.19437](https://arxiv.org/abs/2609.19437)

    本文提出三种利用大语言模型将丰富辅助信息（如训练曲线、专家笔记和先验知识）融入贝叶斯优化的方法，在超参数优化基准和真实核聚变优化任务中始终优于标准BO及现有LLM优化方法。

    

    贝叶斯优化（BO）被广泛用于优化昂贵的黑盒函数，然而许多现实世界的优化问题包含比单纯函数评估丰富得多的信息。例如超参数优化中的训练曲线、科学实验中的专家笔记和图像，以及关于最优解可能位置的先验知识。我们证明大语言模型（LLM）能够有效利用这些丰富的辅助信息来指导优化。基于这些发现，我们开发了三种使用大语言模型将辅助信息纳入贝叶斯优化的方法。在超参数优化基准测试和一个真实世界的核聚变优化任务中，我们的方法始终优于标准贝叶斯优化和现有的基于大语言模型的优化方法。我们的结果证明了大语言模型在贝叶斯优化中利用丰富辅助信息的有效性。

    arXiv:2609.19437v1 Announce Type: new  Abstract: Bayesian Optimization (BO) is widely used for optimizing expensive black-box functions, yet many real-world optimization problems contain substantially richer information than function evaluations alone. Examples include training curves in hyperparameter optimization, expert notes and images in scientific experimentation, and prior knowledge about where optima may lie. We show that large language models (LLMs) can effectively leverage such rich auxiliary information to guide optimization. Motivated by these findings, we develop three methods for incorporating auxiliary information into BO using LLMs. Across hyperparameter optimization benchmarks and a real-world nuclear fusion optimization task, our methods consistently outperform both standard BO and existing LLM-based optimization approaches. Our results demonstrate the effectiveness of LLMs for leveraging rich auxiliary information in BO.
    
[^170]: 揭秘面向控制系统的线性算子学习

    Demystifying Linear Operator Learning for Control Systems

    [https://arxiv.org/abs/2609.19428](https://arxiv.org/abs/2609.19428)

    本文提出利用演化方程的（半）群框架推导结构假设，并借助逆问题框架分析学习算法，从而为控制系统的线性算子学习提供误差分解、收敛保证与最优正则化，实现方法比较与可证明优势算法的设计。

    

    本文提出了一种从数据中学习控制系统线性算子的结构化方法。我们同时解决了该问题的结构方面与学习理论方面。为了推导结构假设，我们建议采用演化方程的（半）群这一成熟框架，因为控制系统中的算子属于同一类型。此外，我们提出通过逆问题框架的视角来分析学习算法。这揭示了学习到的模型如何通过误差分解、收敛保证和最优正则化依赖于数据——使我们能够比较现有方法，并推导出可证明具有优势的算法。为了获得这些结果，我们将研究范围限定在希尔伯特空间上的有界算子。尽管这看起来可能具有局限性，但现有方法通常为了获得类矩阵表示而隐式地做出这一假设。我们展示了使用这些框架的强大作用……

    arXiv:2609.19428v1 Announce Type: cross  Abstract: This paper proposes a structured approach to learning linear operators for control systems from data. We address both structural and learning-theoretic aspects of the problem. To derive structural assumptions, we propose using the well-established framework of (semi)groups for evolution equations, as operators in control systems are of the same type. Further, we propose analyzing learning algorithms through the lens of the inverse problems framework. This reveals how a learned model depends on the data via error decompositions, convergence guarantees, and optimal regularization -- enabling us to compare existing methods and derive provably advantageous algorithms. In order to obtain these results, we restrict our scope to bounded operators on Hilbert spaces. Although this may appear restrictive, existing approaches often make this assumption implicitly to obtain matrix-like representations. We demonstrate the power of using these frame
    
[^171]: 稳定策略学习

    Stable Policy Learning

    [https://arxiv.org/abs/2609.19418](https://arxiv.org/abs/2609.19418)

    本文证明算法稳定性是刻画策略学习中期望福利与抽样风险之间权衡的核心因素，并提出“策略投票装袋”方法，通过对多个子样本的处理决策投票取平均来降低抽样风险。

    

    在循证政策制定中，通常先观察一个实验样本，然后将学习到的政策建议大规模实施。从实验数据中学习到的政策在期望福利方面可能表现良好，但实验中的随机抽样可能产生福利结果较差的建议。在本文中，我们提出这样一个问题：策略学习算法应如何在期望福利与抽样风险之间取得平衡？我们的主要贡献是证明算法稳定性在刻画和应对这一权衡中起着核心作用。直观地说，如果一个策略学习算法在替换一个实验单元时其建议保持稳定，那么该算法的抽样风险有限。我们提出了一种名为“策略投票装袋”的策略学习方法，该方法在许多子样本上学习处理决策，然后将其投票平均为处理概率。相对于使用单个子样本，跨子样本平均……

    arXiv:2609.19418v1 Announce Type: cross  Abstract: In evidence-based policymaking, typically one experimental sample is observed, then a learned policy recommendation is implemented at scale. Policies learned from the experimental data can perform well in expected welfare, yet random sampling in the experiment can produce recommendations with poor welfare outcomes. In this paper, we ask: how should policy learning algorithms balance expected welfare against sampling risk? Our main contribution is to show that algorithmic stability plays a central role in characterizing and navigating the tradeoff. Intuitively, if a policy learning algorithm's recommendation remains stable when one experimental unit is replaced, then that algorithm has limited sampling risk. We propose a method for policy learning called policy-vote bagging, which learns treatment decisions on many subsamples then averages their votes into treatment probabilities. Relative to using one subsample, averaging across subsam
    
[^172]: 引力波信号中超越广义相对论偏差的深度学习探测：基于真实LIGO噪声的探测阈值研究

    Deep Learning Detection of Beyond-General-Relativity Deviations in Gravitational-Wave Signals: A Detection-Threshold Study with Real LIGO Noise

    [https://arxiv.org/abs/2609.19416](https://arxiv.org/abs/2609.19416)

    该研究利用一维卷积神经网络结合手工波形统计量的混合深度学习分类器，在真实LIGO H1探测器噪声中对引力波信号的超越广义相对论偏差进行探测，并基于真实GW150914应变数据确定了约β≈0.25的定量探测阈值。

    

    我们研究了机器学习对引力波信号中受控的超越广义相对论（beyond-GR）偏差的探测，同时使用了合成aLIGO-PSD噪声和真实的LIGO H1探测器应变数据。三种偏差族被应用于广义相对论（GR）的旋近-并合-铃宕波形：振幅调制、相位调制和频率调制，每种偏差均由无量纲强度系数 $\beta$ 参数化。一个混合分类器结合了一维卷积神经网络与十种手工构造的波形统计量，在GR波形和修改波形上训练，并在训练中未包含的偏差类型上进行测试。核心结果是作为 $\beta$ 函数的定量可探测性曲线。使用真实的GW150914应变作为模板和真实的H1探测器噪声，我们发现探测阈值为 $\beta \approx 0.25$，准确率从 $\beta \leq 0.2$ 时的随机水平平滑上升至完美分类（原文在此处截断）。

    arXiv:2609.19416v1 Announce Type: cross  Abstract: We study machine-learning detection of controlled beyond-General-Relativity (beyond-GR) deviations in gravitational-wave signals, using both synthetic aLIGO-PSD noise and real LIGO H1 detector strain. Three deviation families are applied to General-Relativistic inspiral-merger-ringdown waveforms: amplitude modulation, phase modulation, and frequency modulation, each parameterized by a dimensionless strength coefficient $\beta$. A hybrid classifier combining a one-dimensional convolutional neural network with ten hand-crafted waveform statistics is trained on GR and modified waveforms and tested on a deviation type excluded from training. The central result is a quantitative detectability curve as a function of $\beta$. Using the real GW150914 strain as a template and real H1 detector noise, we find a detection threshold at $\beta \approx 0.25$, with accuracy rising smoothly from chance at $\beta \leq 0.2$ to perfect classification at $
    
[^173]: 通过选择性奖励刺激改进离线目标条件强化学习

    Improving Offline Goal-Conditioned Reinforcement Learning via Selective Reward Stimulation

    [https://arxiv.org/abs/2609.19414](https://arxiv.org/abs/2609.19414)

    提出RSIQL方法，通过利用辅助价值函数识别离线轨迹中朝目标取得进展的中间状态并对其施加选择性奖励刺激，来改善奖励稀疏和长时程依赖下的离线目标条件强化学习。

    

    目标条件强化学习旨在学习能够到达指定目标的策略，但在奖励稀疏且具有长时程依赖性的离线设置中仍然充满挑战。在这种设置下，目标完成信息在时间上往往远离促成成功的早期决策，而离线价值估计又会引入额外的误差。我们从奖励传播的视角研究了这一问题，并在一个风格化的延迟目标设定中展示了目标导向的价值分离相对于局部估计误差可能变得多么微小。受此分析启发，我们提出了奖励刺激隐式Q学习（RSIQL），这是一种简单的非分层方法，它在离线轨迹中取得进展的中间状态处引入额外的奖励信号。RSIQL利用辅助的目标条件价值函数来识别被估计为朝目标取得进展的中间状态，并对这些状态施加奖励刺激（原文此处截断）。

    arXiv:2609.19414v1 Announce Type: new  Abstract: Goal-conditioned reinforcement learning aims to learn policies that reach specified goals, but remains challenging in offline settings with sparse rewards and long-horizon dependencies. In such settings, goal-completion information can be temporally distant from the early decisions that enable success, while offline value estimation introduces additional error. We study this issue from a reward-propagation perspective and show, in a stylized delayed-goal setting, how goal-directed value separation can become small relative to local estimation error. Motivated by this analysis, we propose Reward Stimulation Implicit Q-Learning (RSIQL), a simple non-hierarchical method that introduces additional reward signals at progress-making intermediate states in offline trajectories. RSIQL uses an auxiliary goal-conditioned value function to identify intermediate states estimated to make progress toward the goal and applies reward stimulation to prov
    
[^174]: 用于重大不良心血管事件机会性预测的乳腺X线摄影基础模型

    Mammography Foundation Models for Opportunistic Prediction of Major Adverse Cardiovascular Events

    [https://arxiv.org/abs/2609.19385](https://arxiv.org/abs/2609.19385)

    该研究表明，为乳腺癌任务预训练的乳腺X线摄影基础模型无需心血管专门监督或乳腺动脉钙化标注，即可利用常规乳腺筛查影像机会性地预测女性未来5年的重大不良心血管事件风险（AUROC约0.82）。

    

    心血管疾病（CVD）仍然是女性死亡的首要原因，然而心血管风险评估通常依赖于在常规诊疗中可能缺失、过时或无法获得的临床变量。筛查性乳腺X线摄影为心血管风险的机会性分层提供了机会，因为它是常规获取的影像，并且包含与心血管风险和事件相关的血管特征，例如乳腺动脉钙化（BAC）。我们评估了最初为乳腺癌相关任务预训练的乳腺X线摄影专用基础模型，能否在缺乏心血管专门监督或明确BAC标注的情况下，迁移到心血管风险预测任务。我们构建了一个包含22,497名女性的5年重大不良心血管事件（MACE）队列，并将其与电子健康记录结局相关联，其中包含500个事件（患病率2.22%）。基础模型实现了0.823和0.822的AUROC……

    arXiv:2609.19385v1 Announce Type: cross  Abstract: Cardiovascular disease (CVD) remains the leading cause of death among women, yet cardiovascular risk assessment often relies on clinical variables that may be missing, outdated, or unavailable in routine care. Screening mammography offers an opportunity for opportunistic cardiovascular risk stratification because it is routinely acquired and contains vascular features, including breast arterial calcifications (BAC), that are associated with cardiovascular risk and events. We evaluate whether mammography specific foundation models, originally pretrained for breast cancer-related tasks, can transfer to cardiovascular risk prediction without cardiovascular specific supervision or explicit BAC annotation. We constructed a 5-year major adverse cardiovascular event (MACE) cohort of 22,497 women linked to electronic health record outcomes, including 500 events (2.22% prevalence). The foundation models achieved AUROCs of 0.823 and 0.822 substa
    
[^175]: FCx：一种寻找可行反事实解释的算法

    FCx: An algorithm for finding Feasible Counterfactual Explanations

    [https://arxiv.org/abs/2609.19383](https://arxiv.org/abs/2609.19383)

    该论文提出FCx算法，通过改进的变分自编码器首次高效生成同时具备逼真性、低成本和可行性的反事实解释，同时支持用户指定的硬约束和通过因果推断自动获取的软约束。

    

    反事实解释旨在识别能够改变输入分类结果的变化。虽然现有方法能够生成逼真且低成本的反事实解释，但它们往往无法确保可行性，可能会建议非建设性的修改或与未来变化不相容的改变（例如，通过改变个人的种族来获得工作机会）。我们引入了一种对反事实解释的改进方法，明确地强制执行可行性。我们的方法是首个能够高效生成同时具备逼真性、低成本和可行性的反事实解释的方法。我们的方法同时支持硬可行性约束（由用户根据领域知识指定）和软可行性约束（通过因果推断从数据集中自动推断得出）。我们的方法——可行反事实解释，基于一种改进的变分自编码器（VAE），并通过多因子损失函数进行优化。我们根据数值变化的绝对量（邻近性）以及改变特征的数量来衡量变化的成本。

    arXiv:2609.19383v1 Announce Type: new  Abstract: Counterfactual (CF) explanations identify changes that alter an input's classification. While existing methods produce realistic and low-cost CFs, they often fail to ensure feasibility, by suggesting non-constructive modifications or incompatible with future changes (e.g., changing an individual's race to secure a job offer). We introduce a refinement of CF explanations that explicitly enforces feasibility. Our approach is the first to efficiently generate CFs that are realistic, low-cost and feasible. We accommodate both hard feasible constraints, specified by domain knowledge users, and soft feasible constraints, inferred automatically via causal inference from the dataset. Our method, Feasible Counterfactual Explanations (FCx), is based on a modified Variational Autoencoder (VAE) optimized with a multi-factor loss function. We measure the cost of a change based on the absolute change in values (proximity) as well as the number of feat
    
[^176]: 机器学习评估老年西班牙裔成年人队列中炎症生物标志物对认知障碍的预测价值

    Machine-Learning Assessment of the Predictive Value of Inflammatory Biomarkers for Cognitive Impairment in an Older Hispanic Adult Cohort

    [https://arxiv.org/abs/2609.19374](https://arxiv.org/abs/2609.19374)

    该研究在小型老年西班牙裔临床队列中采用无泄漏的阈值似然朴素贝叶斯分类器，发现炎症生物标志物I-309（CCL1）相较于人口统计学基线对认知障碍具有显著且可复现的预测增量价值。

    

    小型临床表格数据集需要可解释的机器学习方法，因为深度学习往往不切实际，而集成模型可能难以检视。一个关键的陷阱在于统计显著性并不一定意味着预测效用。利用巴拿马老龄化研究计划——健康差异（PARI-HD）队列的数据（n=165），我们实现了一种无泄漏的阈值似然伯努利/分类朴素贝叶斯（BNB/CNB）分类器。在每个训练折内，每个连续预测变量被简化为有监督的、由卡方检验推导出的状态，而收入则通过分类似然进入模型。所有依赖数据的步骤均在重复30次的分层10折交叉验证内完成。人口统计学基线模型达到了0.630 ± 0.017的ROC-AUC。I-309（CCL1）是主要的增量特征，使AUC提高了0.110，配对DeLong检验在100%的重复中均得到p<0.05。在……

    arXiv:2609.19374v1 Announce Type: new  Abstract: Small clinical tabular datasets require interpretable machine learning because deep learning is often impractical and ensemble models can be difficult to inspect. A key pitfall is that statistical significance does not necessarily imply predictive utility. Using data from the Panama Aging Research Initiative--Health Disparities (PARI-HD) cohort (n=165), we implemented a leakage-safe threshold-likelihood Bernoulli/Categorical Naive Bayes (BNB/CNB) classifier. Within every training fold, each continuous predictor was reduced to a supervised chi-square-derived state, while income entered the model through a categorical likelihood. All data-dependent steps were performed within repeated stratified 10-fold cross-validation with 30 repeats. The demographic baseline achieved a ROC-AUC of 0.630 +/- 0.017. I-309 (CCL1) was the dominant incremental feature, increasing AUC by 0.110, with paired DeLong tests yielding p<0.05 in 100% of repeats. In th
    
[^177]: Stiefel注意力：Transformer投影矩阵几何何时主导优化器选择——何时又不主导

    Stiefel Attention: When the Geometry of Transformer Projection Matrices Dominates Optimizer Choice---and When It Does Not

    [https://arxiv.org/abs/2609.19363](https://arxiv.org/abs/2609.19363)

    本文将注意力中的查询和键投影矩阵约束到Stiefel流形上并用黎曼Adam优化，理论证明了该更新的最速下降性、尺度无关性和严格O(d)-等变性，并发现权重衰减在该流形上黎曼梯度恒为零，使注意力几何免于坍缩，从而在grokking任务上将验证准确率从61.1%大幅提升至97.0%。

    

    注意力机制中的查询和键投影矩阵 $W_Q, W_K$ 几乎总是由欧氏优化器训练，对其几何结构没有任何约束。我们将其约束到Stiefel流形上，并用一种黎曼Adam优化器在该流形上进行优化：该优化器为每个矩阵帧仅携带一个标量二阶矩，通过信赖域限制步长，并以极分解方式执行收缩。四个命题证明该更新在嵌入度量下是最速下降的、与梯度尺度无关、条件良好且严格满足 $\mathrm{O}(d)$-等变性，每一项均在 float64 精度下获得数值验证。第五个命题揭示了其机制：权重衰减在 $\mathrm{St}(d,r)$ 上的黎曼梯度恒为零，因为 $W = W I_r$ 位于法空间中，因此所学到的注意力几何结构能够在权重衰减驱动模型其余部分发生的坍缩循环中得以保存。在模运算grokking任务上，单次运行在第20000轮时保持了97.0%的验证准确率，而基线仅为61.1%——这一……

    arXiv:2609.19363v1 Announce Type: new  Abstract: The query and key projections $\WQ,\WK$ in attention are almost always trained by Euclidean optimizers with no constraint on their geometry. We constrain them to the Stiefel manifold and optimize them there with a Riemannian Adam that carries one scalar second moment per frame, caps its step by a trust region, and retracts polarly. Four propositions prove this update is steepest descent in the embedded metric, independent of gradient scale, well conditioned, and exactly $\mathrm{O}(d)$-equivariant, each certified numerically in \texttt{float64}. A fifth supplies the mechanism: weight decay has \emph{identically zero} Riemannian gradient on $\St(d,r)$, since $W = W I_r$ lies in the normal space, so the learned attention geometry survives the collapse cycles that decay drives through the rest of the model. On modular arithmetic grokking, a single run holds $97.0\%$ validation accuracy at epoch 20\,000 against the baseline's $61.1\%$---an u
    
[^178]: 面向养老护理持续监测的智能鞋垫人体活动识别

    Smart Insole Human Activity Recognition for Continuous Monitoring in Elderly Care

    [https://arxiv.org/abs/2609.19359](https://arxiv.org/abs/2609.19359)

    本文提出了一种集成16个压力传感点和六轴IMU的无线智能鞋垫平台，利用HGB机器学习模型从足底压力和惯性信号中高精度识别老年人坐、站、行走及不稳定行走状态（宏F1超过0.95），为养老护理中的跌倒风险持续监测提供了有效方案。

    

    老年人跌倒之前通常会出现移动能力、平衡能力和姿势转换方面的变化。本文提出了一种无线智能鞋垫平台及机器学习工作流程，用于从足底压力和惯性信号中识别坐姿、站立、行走和不稳定行走等状态。每只鞋垫集成了16个主动压力传感点，以及由三轴加速度和角速度组成的六维IMU数据流。数据采集自15名健康成年人，采样频率为80 Hz，并被分割为重叠窗口。首先通过分层10折交叉验证筛选窗口长度和候选模型族；随后采用参与者独立的5折分层分组交叉验证获得主要性能估计，确保同一参与者的所有窗口都保留在同一折中。在该协议下，基于直方图的梯度提升模型（HGB）在左脚上取得了0.954的宏平均F1分数[摘要在此处截断，推测右脚为0.959]。

    arXiv:2609.19359v1 Announce Type: new  Abstract: Falls in older adults are often preceded by changes in mobility, balance, and postural transitions. This paper presents a wireless smart insole platform and machine-learning workflow for recognizing sitting, standing, walking, and unstable walking from plantar-pressure and inertial signals. Each insole integrates 16 active pressure-sensing locations and a six-dimensional IMU stream consisting of tri-axial acceleration and angular velocity. Data were collected from 15 healthy adults at 80~Hz and segmented into overlapping windows. Window length and candidate model families were first screened with stratified 10-fold cross-validation; the primary performance estimate was then obtained with participant-independent 5-fold Stratified Group cross-validation, ensuring that all windows from a participant remained in a single fold. Under this protocol, Histogram-Based Gradient Boosting (HGB) achieved macro-F1 scores of 0.954 and 0.959 for the lef
    
[^179]: 学习子流形以用于随机点积图的后续推断，第一部分：理论

    Learning Submanifolds for Subsequent Inference on Random Dot Product Graphs, Part 1: Theory

    [https://arxiv.org/abs/2609.19357](https://arxiv.org/abs/2609.19357)

    提出了一种半监督决策规则框架，利用辅助数据和Isomap流形学习来学习随机点积图潜在位置的未知低维支撑流形，并证明随着辅助数据量的增加，半监督规则的风险收敛于最优先知规则的风险。

    

    我们提出了一个针对随机点积图的受限推断框架，其中图的潜在位置位于一个未知的低维支撑流形上。对于一般的决策问题，我们提出了利用辅助数据来学习支撑流形的半监督决策规则。具体而言，我们的规则使用Isomap流形学习过程来构建观测图的低维欧几里得表示，在该空间中，一个等距不变的函数将点的配置映射为动作。我们研究了当从未知支撑流形采样的辅助数据量增加时所提出规则的行为。我们证明，随着辅助样本量的增加，半监督规则的风险收敛于一个先知规则的风险，该先知规则依赖于能够从支撑流形中提取出的最大低维欧几里得结构。示例、应用和模拟研究……

    arXiv:2609.19357v1 Announce Type: cross  Abstract: We propose a framework for restricted inference on random dot product graphs whose latent positions lie on an unknown low-dimensional support manifold. For general decision problems, we propose semisupervised decision rules that use auxiliary data to learn the support manifold. Specifically, our rules use the Isomap manifold learning procedure to construct a low-dimensional Euclidean representation of the observed graph, in which space an isometrically invariant function maps configurations of points to actions. We study the behavior of the proposed rules as the quantity of auxiliary data sampled from the unknown support manifold increases. We show that, as the auxiliary sample size increases, the risk of the semisupervised rule converges to the risk of an oracle rule that relies on the maximal amount of low-dimensional Euclidean structure that can be extracted from the support manifold. Examples, applications, and simulation studies a
    
[^180]: 如何引导你的语言流

    How to Guide Your Language Flow

    [https://arxiv.org/abs/2609.19356](https://arxiv.org/abs/2609.19356)

    提出了一种名为“探针引导”的新方法，利用现有扩散模型的冻结内部状态构建引导信号，无需推理时额外的前向传播，即可在无条件生成和问答基准上显著提升扩散语言模型的性能，并揭示了自动引导中弱模型需来自训练低熵区域的关键条件。

    

    我们介绍了一种引导流匹配模型的新方法。我们的方法称为“探针引导”，它利用现有扩散模型的冻结内部状态来构建引导信号。该方法的工作原理与自动引导类似，但消除了推理时进行额外前向传播的需要，并提供了一条可靠的路径来确保弱模型和强模型共享相似的动力学特性。我们将该方法应用于连续扩散语言模型并进行基准测试，探针引导在无条件生成任务上创造了新的最先进性能。当应用于一个17亿参数的扩散语言模型时，探针引导在多项选择题问答基准测试中持续带来性能提升。利用我们的探针，我们研究了传统的自动引导设置（即强模型实际上是一个弱检查点），发现弱模型必须来自训练过程中的低熵区域。这些发现都提供了一种实用的方法

    arXiv:2609.19356v1 Announce Type: cross  Abstract: We introduce a new method to guide flow matching models. Our approach, which we call probe guidance, uses the frozen internal states of an existing diffusion model to construct a guidance signal. This works using a similar principle as autoguidance, but eliminates the need for an additional forward pass at inference time and provides a reliable path to ensure that the weak and strong model share similar dynamics. We apply and benchmark this method on continuous diffusion language models, where probe guidance sets a new state-of-the-art performance on unconditional generation. When applied to a 1.7B diffusion language model, probe guidance consistently improves on multiple choice question answering benchmarks. Using our probes, we study the traditional autoguidance setting where the strong model is a weak checkpoint, and find that the weak model must come from a low-entropy region of training. These findings both provide a practical way
    
[^181]: 视觉语言模型能评判奥运跳水吗？从推理到评分的零样本动作质量评估

    Can Vision-Language Models Judge Olympic Diving? From Reasoning to Scores in Zero-Shot Action Quality Assessment

    [https://arxiv.org/abs/2609.19354](https://arxiv.org/abs/2609.19354)

    该研究提出一种基于回归的集成框架，利用视觉语言模型生成的语义推理和阶段级子评分对奥运跳水进行零样本动作质量评估，将Spearman相关性从0.32显著提升至0.67。

    

    奥运体育项目中自动化动作质量评估（AQA）由于人体运动的复杂性以及专家评分固有的主观性，始终是一项具有挑战性的任务。本工作评估了开源视觉语言模型（VLM）在使用AQA-7基准数据集对奥运跳水视频进行零样本动作质量评估方面的能力。为此，本文提出了一种基于回归的框架，利用视觉语言模型生成的语义推理和阶段级子评分，结合TF-IDF向量化、降维和集成学习来预测最终比赛得分。实验结果表明，单独使用视觉语言模型仅能达到低于0.32的中等Spearman相关性，而所提出的集成回归框架在评估中显著提升了性能，采用四模型配置达到了0.67的Spearman相关性。文本推理特征始终……

    arXiv:2609.19354v1 Announce Type: cross  Abstract: Automated action quality assessment (AQA) in Olympic sports remains a challenging task due to the complexity of human motion and the subjectivity inherent in expert judging. This work evaluates the capability of open-source Vision-Language Models (VLMs) to perform zero-shot action quality assessment on Olympic diving videos using the AQA-7 benchmark dataset. In this regard, a regression-based framework is pro-posed to leverage both the semantic reasoning and phase-level sub-scores generated by the VLMs, combining TF-IDF vectorization, dimensionality reduction, and ensemble learning to predict final competition scores. Experimental results show that standalone VLMs achieve moderate Spearman correlations below 0.32, while the proposed ensemble regression framework substantially improves performance in the reported evaluation, reaching a Spearman correlation of 0.67 with a four-model configuration. Textual reasoning features con-sistently
    
[^182]: 用于异构分布式系统隐私保护建模的个性化联邦层次高斯过程

    Personalized Federated Hierarchical Gaussian Processes for Privacy-Preserving Modeling of Heterogeneous Distributed Systems

    [https://arxiv.org/abs/2609.19337](https://arxiv.org/abs/2609.19337)

    提出个性化联邦层次高斯过程 pFedHGP，通过将客户端潜在函数分解为共享全局分量、客户端特定偏差和局部残差，在仅同步低维统计量的前提下实现异构分布式数据的隐私保护概率建模，并在故障分类和空气质量建模等应用中取得优异效果。

    

    我们提出了个性化联邦层次高斯过程，用于数据分布在异构客户端场景下的概率回归与分类。每个客户端的潜在函数被分解为三部分：(i) 共享的全局分量，(ii) 共享全局核结构的客户端特定偏差，以及 (iii) 灵活的局部残差。借助稀疏诱导变量近似和联邦变分推断，原始数据保留在本地，服务器仅同步共享分量的低维统计量。完整的预测分布可支持考虑不确定性的决策。在应用研究中，pFedHGP 在压机吨位监测中仅使用 13.77% 的已标记周期就实现了完美的故障分类，并在无需集中站级时间序列数据的情况下，在联邦空气质量建模中成功恢复了地理区域。瞬时线性混合模型的观点将该层次结构与多输出高斯过程联系起来。

    arXiv:2609.19337v1 Announce Type: new  Abstract: We present Personalized Federated Hierarchical Gaussian Processes (pFedHGP) for probabilistic regression and classification when data are distributed across heterogeneous clients. Each client's latent function decomposes into (i) a shared global component, (ii) a client-specific deviation that shares the global kernel structure, and (iii) a flexible local residual. Sparse inducing-variable approximations and federated variational inference keep raw data local while the server synchronizes only low-dimensional statistics for the shared component. Full predictive distributions support uncertainty-aware decisions. In application studies, pFedHGP attains perfect fault classification in press tonnage monitoring using 13.77% of labeled cycles and recovers geographic zones in federated air-quality modeling without centralizing station-level time series. An Instantaneous Linear Mixing Model viewpoint links the hierarchy to multi-output Gaussian 
    
[^183]: 为什么预训练无法共享跨语言知识

    Why Pretraining Fails to Share Cross-Lingual Knowledge

    [https://arxiv.org/abs/2609.19291](https://arxiv.org/abs/2609.19291)

    本研究通过受控双语预训练实验发现，不相交的词表空间是跨语言知识泛化的根本障碍——即使是对同一语言的完全相同副本，仅仅词表不相交就足以导致知识隔阂。

    

    大型语言模型（LLMs）在多种语言的处理和建模方面取得了显著进展。然而，与人类多语言者不同，它们表现出的跨语言知识迁移能力出奇地有限。尽管这一局限性已被充分记录，但其在多语言训练过程中的起源仍不清楚。我们预训练了360M和7B参数的LLMs，并表明跨语言知识泛化能力差的问题在预训练期间就已出现，且在标准干预措施下依然持续存在。为了分离其成因，我们采用了一个受控的双语预训练设置，使用同一语言的两个副本，它们共享完全相同的文本和分词方式，但映射到不相交的词表空间。我们发现，仅不相交的词表就足以诱发知识隔阂，即使在同一语言的完全相同副本之间也是如此，从而确立了不相交的词表空间是跨语言知识泛化的根本障碍。基于这一理解，我们（注：原文摘要在此处被截断）

    arXiv:2609.19291v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have made remarkable progress in the processing and modeling of many languages. Yet, unlike human multilinguals, they exhibit surprisingly limited cross-lingual knowledge transfer. While this limitation is well documented, its origins during multilingual training remain unclear. We pretrain 360M- and 7B-parameter LLMs and show that poor cross-lingual knowledge generalization emerges during pretraining and persists under standard interventions. To isolate its cause, we employ a controlled bilingual pretraining setting using two copies of the same language, sharing identical text and token segmentation, but mapped to disjoint token spaces. We find that disjoint tokens alone are enough to induce knowledge compartmentalization, even between identical copies of the same language, establishing disjoint token spaces as a fundamental barrier to cross-lingual knowledge generalization. Guided by this understanding, w
    
[^184]: 循环神经网络中学习诱导的动力学转变

    Learning-Induced Dynamical Transition in Recurrent Neural Networks

    [https://arxiv.org/abs/2609.19288](https://arxiv.org/abs/2609.19288)

    该研究发展了非平衡动力学平均场理论，揭示循环神经网络在学习过程中因有效反馈强度的持续增长而经历由分岔刻画的从混沌到稳定的动力学转变。

    

    循环神经网络中的学习能够从根本上重塑其底层动力学，将最初的混沌活动转变为稳定的、与任务相关的行为。我们发展了一种非平衡动力学平均场理论（DMFT）来描述学习过程中的这一转变。我们证明了缓慢的反馈驱动学习过程会产生一个不断演变的有效反馈强度，该强度驱动网络经历一个由DMFT解的分岔所定义的从混沌到稳定动力学的转变。通过推导整个学习过程中的双时间关联函数，我们识别出一个临界反馈强度以及一个依赖于学习速率的相应临界时间，用以区分这两种动力学状态。这一转变源于不断增长的学习反馈结构对有效动力学景观的逐步变形。从未训练状态出发，该理论预测了训练过程中网络输出的时间演化。

    arXiv:2609.19288v1 Announce Type: new  Abstract: Learning in recurrent neural networks can fundamentally reshape their underlying dynamics, transforming initially chaotic activity into stable task-dependent behavior. We develop a non-equilibrium dynamical mean-field theory(DMFT) to describe this transition during learning. We show that a slow feedback-driven learning process generates an evolving effective feedback strength that drives the network through a transition from chaotic to stable dynamics defined by a bifurcation of the DMFT solution. By deriving the two-time correlation function throughout learning, we identify a critical feedback strength and a corresponding learning rate dependent critical time separating these regimes. The transition arises from the progressive deformation of an effective dynamical landscape by the growing learned feedback structure. Starting from the untrained state, the theory predicts the time evolution of the network output during training and shows 
    
[^185]: 射频卷积神经网络

    Radio-Frequency Convolutional Neural Networks

    [https://arxiv.org/abs/2609.19279](https://arxiv.org/abs/2609.19279)

    该论文提出射频卷积神经网络（RF-CNN），创造性地将无线设备中已有的射频混频器硬件重新用于在频域执行卷积运算，使边缘设备无需增加额外计算硬件即可运行高达2640万参数的深度CNN。

    

    在智能手机、可穿戴设备和无人机等边缘设备上直接运行人工智能（AI）模型，可提供低延迟、广泛的可扩展性和数据隐私，但这些设备通常不具备现代神经网络所需的计算能力。为应对这一问题，人们开发了边缘加速器，然而每一种方案都是在尺寸、重量、功耗和成本（SWaP-C）本已受限的设备上增加额外的计算硬件。另一种替代方案隐藏在这些设备已有的组件中：每个无线电台中的频率混频器可在时域中对信号进行相乘，天然地执行频域中的卷积运算。在此，我们提出了射频卷积神经网络（RF-CNN），将现有通信硬件重新用于CNN推理。多通道卷积被映射到频率音调上，由无源混频器在单次操作中完成执行。我们通过实验证明，RF-CNN可运行参数量高达2640万的深度卷积神经网络……

    arXiv:2609.19279v1 Announce Type: new  Abstract: Running artificial intelligence (AI) models directly on edge devices such as smartphones, wearables, and drones offers low latency, pervasive scalability, and data privacy, but these devices rarely carry the computing capability that modern neural networks demand. Edge accelerators have been developed in response, yet each adds computing hardware to devices already constrained in size, weight, power, and cost (SWaP-C). An alternative lies in what these devices already carry: the frequency mixer in every wireless radio multiplies signals in time, natively performing convolution in the frequency domain. Here we introduce radio-frequency convolutional neural networks (RF-CNNs), which repurpose existing communication hardware for CNN inference. Multi-channel convolutions are mapped onto frequency tones for a passive mixer to execute in a single pass. We experimentally demonstrate that RF-CNN runs deep CNNs up to 26.4 million parameters and n
    
[^186]: 用于词-文档矩阵谱共聚类的随机SVD近似方法

    Randomized SVD Approximations for Spectral Co-Clustering of Word-Document Matrices

    [https://arxiv.org/abs/2609.19243](https://arxiv.org/abs/2609.19243)

    本文提出两种随机SVD近似方法来加速词-文档矩阵的谱共聚类，实验表明随机投影方法在各种设置下更为可靠，而随机采样方法仅对较稠密的矩阵有效。

    

    谱共聚类是发现词-文档矩阵中潜在结构的有用工具，但其对奇异值分解（SVD）的依赖使得标准形式在高维数据上计算代价高昂。本文提出了两种随机近似方法，用于文档聚类数与词聚类数可能不同的二部文本数据的归一化谱共聚类。第一种方法通过随机投影使用随机SVD，第二种方法将部分SVD与逐元素随机采样相结合。在真实世界和合成数据集的实验中，两种方法相对于完整SVD基线都减少了运行时间，但其表现取决于矩阵的稀疏程度。随机投影方法在所有测试设置中是更可靠的近似方法，而基于采样的方法在较稠密的矩阵上最为有用，在本身已经稀疏的文本数据上收益有限。这些结果表明随机近似

    arXiv:2609.19243v1 Announce Type: cross  Abstract: Spectral co-clustering is a useful tool for discovering latent structure in word-document matrices, but its reliance on singular value decomposition (SVD) can make standard formulations expensive on high-dimensional data. This paper presents two randomized approximations for normalized spectral co-clustering of bipartite text data when the numbers of document and word clusters may differ. The first method uses randomized SVD through random projection, while the second combines partial SVD with element-wise random sampling. Across real-world and synthetic datasets, both methods reduce runtime relative to the full-SVD baseline, but their behavior depends on matrix sparsity. The random projection method is the more reliable approximation across the tested settings, whereas the sampling-based method is most useful on denser matrices and provides limited benefit on already sparse text data. These results show that randomized approximations 
    
[^187]: 面向高效分布式长上下文扩散语言模型训练的块并行方法

    Block Parallelism For Efficient Distributed Long-Context Diffusion Language Model Training

    [https://arxiv.org/abs/2609.19242](https://arxiv.org/abs/2609.19242)

    本文提出块并行（BP）与上下文分片块并行（CSBP）两种新的分布式并行方法，利用BDLM目标函数在目标块上的可分离性，将损坏块计算分配至独立进程并分片共享干净序列，从而大幅提升长上下文扩散语言模型训练的通信与内存效率。

    

    块扩散语言模型（BDLM）将跨块的自回归依赖与块内的并行去噪相结合，但长上下文训练受到分布式注意力通信和激活内存的制约。传统的上下文并行（CP）按位置对干净序列与损坏序列的组合进行切分，需要同时传输共享的干净K/V以及特定块的损坏K/V及其梯度。我们观察到BDLM的目标函数在各个目标块上是可分离的，据此提出了块并行（BP）——一种新的分布式并行维度，将每个损坏块的计算分配给一个进程。为了将BP扩展到长上下文场景，我们进一步引入了上下文分片块并行（CSBP），它还在这些进程间对共享的干净序列进行分片。CSBP使损坏的K/V和梯度保持在本地，避免了干净前缀的重复存储，同时保持了BDLM的训练语义。在16块H200 GPU上进行256K上下文长度的训练时，CSBP显著提升了吞吐量……

    arXiv:2609.19242v1 Announce Type: new  Abstract: Block diffusion language models (BDLMs) combine autoregressive dependencies across blocks with parallel denoising within blocks, but long-context training is constrained by distributed attention communication and activation memory. Conventional context parallelism (CP) shards the combined clean-plus-corrupted sequence by position, communicating shared clean K/V together with block-specific corrupted K/V and their gradients. We observe that the BDLM objective separates over target blocks. We introduce block parallelism (BP), a new distributed parallelism dimension that assigns each corrupted-block computation to one rank. To scale BP to long contexts, we introduce context-sharded block parallelism (CSBP), which also shards the shared clean sequence across those ranks. CSBP keeps corrupted K/V and gradients local, avoids replicated clean prefixes, and preserves BDLM training semantics. On 16 H200 GPUs at 256K context, CSBP improves through
    
[^188]: 面向高效大语言模型压缩的逐层课程学习

    Layer-wise Curriculum Learning for Efficient LLM Compression

    [https://arxiv.org/abs/2609.19213](https://arxiv.org/abs/2609.19213)

    提出逐层课程学习方法用于高效LLM压缩，通过将模型分层分段并从易到难地进行知识蒸馏以加速收敛、稳定迁移过程，同时借助多线程特征缓存策略最大化GPU利用率，实现了先进的模型压缩效果。

    

    本文提出了一种用于高效大语言模型（LLM）压缩的逐层课程学习方法。该方法借助课程学习策略促进知识从教师模型向学生模型的迁移，即从较容易的优化任务开始，逐步过渡到更难的任务。为了在LLM压缩中采用逐层学习，我们将整个模型划分为由多层组成的多个片段，从而为大语言模型实现计算上更高效的知识迁移。基于对累积误差现象的理论分析，逐层课程学习在加速收敛的同时稳定了知识迁移过程。此外，我们提出了一种结合多线程策略的特征缓存方法，以高效解决跨层特征不对齐的问题，最大化GPU利用率。因此，我们的方法展现出先进的模型压缩性能。

    arXiv:2609.19213v1 Announce Type: cross  Abstract: In this paper, we introduce layer-wise curriculum learning for efficient LLM compression. The proposed method facilitates the knowledge transfer from the teacher model to the student model, utilizing a curriculum learning approach that begins with easier optimization tasks and progressively tackles harder ones. In order to adopt the layer-wise learning in LLM compression, we partition the whole model into multiple segments consisting of layers, thereby enabling more computationally efficient knowledge transfer for LLMs. Based on our theoretical analysis of cumulative error phenomenon, layer-wise curriculum learning accelerates convergence while stabilizing the knowledge transfer process. In addition, we present a feature caching method with a multi-threading strategy to efficiently address feature misalignment across layers, maximizing GPU utilization. Consequently, our method exhibits advanced model compression performance, as well as
    
[^189]: 并非所有节点生而平等：面向稳定GNN评估的同质性感知分层方法

    Not All Nodes Are Created Equal: Homophily-Aware Stratification for Stable GNN Evaluation

    [https://arxiv.org/abs/2609.19210](https://arxiv.org/abs/2609.19210)

    该论文指出，仅按类别分层的交叉验证不足以稳定图神经网络评估，因为数据划分间局部邻域同质性分布的差异会系统性影响消息传递行为并夸大评估方差，为此提出了同质性感知的分层划分方法以实现更可靠的GNN比较。

    

    图神经网络被广泛用于直推式节点分类，其准确率通常在随机划分的训练/验证/测试集上进行测量。研究表明，同一数据集的不同随机划分会导致报告的准确率发生显著偏移，使得已发表的架构间比较变得不可靠。在非图场景中，经典的解决方法是分层k折交叉验证，它确保每个测试折都能反映数据集的完整类别分布。我们认为，仅凭类别分层对图数据而言是不够的：节点并非孤立而是相互连接的，局部邻域同质性分布不同的折会使模型暴露于系统性不同的关系条件中，这些条件直接影响消息传递行为。由此产生的跨折变化反映了每个划分的同质性构成，使报告的方差膨胀到超出模型行为本身所能解释的程度。

    arXiv:2609.19210v1 Announce Type: cross  Abstract: Graph neural networks are widely used for transductive node classification, with accuracy typically measured on randomly drawn train/validation/test splits. Reported accuracy has been shown to shift substantially across different random splits of the same dataset, making published comparisons between architectures unreliable. The classical remedy in non-graph settings is stratified $k$-fold cross-validation, which ensures each test fold reflects the full class distribution of the dataset. We argue that class stratification alone is insufficient for graphs: nodes are not isolated but connected, and folds that differ in their distribution of local neighbourhood homophily expose the model to systematically different relational conditions that directly affect message-passing behaviour. The resulting cross-fold variation reflects the homophily composition of each split, inflating reported variance beyond what model behaviour alone would pro
    
[^190]: 基于意图覆盖与查询级信用分配的生成式查询推荐

    Generative Query Suggestion via Intent Coverage and Query-Level Credit Assignment

    [https://arxiv.org/abs/2609.19209](https://arxiv.org/abs/2609.19209)

    该论文提出一种意图驱动的生成式查询推荐框架，通过意图感知多样性奖励与查询级信用分配的双阶段优化，在提升点击率和查询质量的同时增强意图覆盖。

    

    生成式查询推荐旨在通过预测用户意图并推荐相关的后续查询来提升用户参与度。一个核心挑战是生成的查询组合既能保证其中单个查询对用户有用，又能覆盖不同的意图。我们提出了一种意图驱动的查询推荐框架，采用双阶段优化。首先，意图感知的多样性建模构建与意图对齐的监督微调（SFT）数据，并使用意图感知多样性奖励来优化意图覆盖。其次，查询级信用分配将个体质量信号路由到相应的查询标记，同时在查询组合内共享组合级别的多样性信号。在一个大规模生产数据集上的实验（包括在线A/B测试和离线评估）表明，该方法在点击率、查询质量和意图覆盖方面均取得了改进。

    arXiv:2609.19209v1 Announce Type: new  Abstract: Generative query suggestion aims to enhance user engagement by anticipating user intents and recommending relevant follow-up queries. A central challenge is to generate slates whose individual queries are useful while the slate covers distinct intents. We propose an Intent-Driven Query Suggestion Framework with dual-stage optimization. First, intent-aware diversity modeling constructs intent-aligned supervised fine-tuning (SFT) data and uses an Intent-Aware Diversity Reward to optimize intent coverage. Second, query-level credit assignment routes individual quality signals to the corresponding query tokens while sharing a slate-level diversity signal across the slate. Experiments on a large-scale production dataset, including online A/B testing and offline evaluation, show improvements in click-through rate, query quality, and intent coverage.
    
[^191]: 立场：是时候用自演化的操作系统层来虚拟化基础模型了

    Position: It is Time to Virtualize Foundation Models with a Self-evolving Operating System Layer

    [https://arxiv.org/abs/2609.19203](https://arxiv.org/abs/2609.19203)

    本文提出构建“基础模型操作系统”（FMOS），通过虚拟化基础模型交互并统一编排记忆分层、模型选择、资源分配与策略验证，以解决当前AI智能体技术栈碎片化、行为不可移植和治理脆弱的问题。

    

    AI应用已经从单一的、单体式的基础模型（FM）转变为复合的智能体系统。然而，当今的技术栈仍然是碎片化的：尽管各类协议（如MCP、A2A）简化了工具与智能体之间的连接，但每个框架都嵌入了一个隐式的运行时来处理状态、记忆、预算和防护机制，导致模型行为不可移植、治理机制脆弱。这类似于操作系统诞生之前的计算时代，当时每个程序都需要重新实现基础服务。本立场论文认为，该领域现在需要一个基础模型操作系统（FMOS）——一个将基础模型交互进行虚拟化的系统层，其方式类似于虚拟机对物理硬件的抽象，从而为应用提供一种拥有专用、可信且能力近乎无界的基础模型实例的假象。在内部，FMOS负责协调跨记忆层的知识管理、模型选择与资源分配，以及验证与策略执行。就像人脑在……之间切换（原文在此处被截断）

    arXiv:2609.19203v1 Announce Type: new  Abstract: AI applications have shifted from single, monolithic foundation models (FM) to compound agentic systems. Yet today's stacks remain fragmented: even as protocols (e.g., MCP, A2A) ease tool/agent connectivity, each framework embeds an implicit runtime for state, memory, budgets, and guardrails, making behavior non-portable and governance brittle. It mirrors computing before operating systems, when every program re-implemented basic services. This position paper argues that the field now needs a Foundation Model Operating System (FMOS) -- a system layer that virtualizes FM interactions analogous to how virtual machines abstract physical hardware, giving applications the illusion of dedicated, trustworthy FM instances with effectively unbounded capabilities. Internally, the FMOS orchestrates knowledge across memory tiers, model selection and resource allocation, and verification and policy enforcement. Like the human brain switching between 
    
[^192]: 通过广义全变差最小化实现联邦软聚类

    Federated Soft Clustering via Generalized Total Variation Minimization

    [https://arxiv.org/abs/2609.19202](https://arxiv.org/abs/2609.19202)

    该论文提出基于广义全变差最小化的联邦软聚类框架，系统比较了三种模型差异度量方法（无需组件匹配的KL散度和闭式MMD，以及需要组件匹配的欧氏距离），并通过同步投影梯度优化，为基于MMD的实例提供了收敛到驻点的理论保证。

    

    我们研究了联邦学习（FL）网络中设备上的联邦软聚类问题，这些设备各自持有私有的本地数据集，并拟合个性化的高斯混合模型（GMM）。广义全变差最小化通过一个图正则化项来耦合局部最大似然问题，该正则化项惩罚相连节点模型之间的差异。差异度量方法的选择是一个关键的设计决策：我们比较了模型参数之间的平方欧氏距离（该方法需要进行组件匹配）与两种直接比较本地模型分布、因而无需匹配的度量方法：蒙特卡洛近似的Kullback-Leibler（KL）散度和具有闭式解的最大均值差异（MMD）。由此产生的三种GTVMin实例均通过同步投影梯度更新进行优化；对于平滑的MMD实例，我们提供了收敛到驻点的理论保证。我们对它们的计算（原文在此处截断）

    arXiv:2609.19202v1 Announce Type: cross  Abstract: We study federated soft clustering over federated learning (FL) networks of devices that each hold a private local dataset and fit a personalized Gaussian mixture model (GMM). Generalized total variation minimization (GTVMin) couples the local maximum likelihood problems through a graph regularizer that penalizes a discrepancy between the models of connected nodes. The choice of discrepancy measure is a key design decision: we compare a squared Euclidean distance between model parameters, which requires component matching, with two measures that compare the local model distributions directly and hence need no matching: a Monte-Carlo approximated Kullback-Leibler (KL) divergence and a closed-form maximum mean discrepancy (MMD). All three resulting GTVMin instances are optimized by synchronous projected gradient updates; for the smooth MMD instance we provide a convergence guarantee to stationary points. We characterize their computation
    
[^193]: 低方差奖励下群体相对优化中的优势尺度校准失衡：诊断与有界恢复

    Advantage Scale Calibration Imbalance in Group-Relative Optimization under Low-Variance Rewards: Diagnosis and Bounded Recovery

    [https://arxiv.org/abs/2609.19164](https://arxiv.org/abs/2609.19164)

    本文诊断了低方差奖励下群体相对优化中优势尺度校准失衡的问题，提出三方校准接口揭示RLOO/Dr.GRPO与GRPO各自的失控行为，并通过奖励分辨率协议与MaxNorm-AC过滤亚分辨率噪声、实现对可信小差距的有界恢复。

    

    在验证器式的RLVR中，群体相对优化通常将优势尺度视为一个实现细节。本文区分了两种低方差情形：不应转化为偏好信号的亚分辨率抖动，以及可信但微小的基数差距——后者应当被学习而不扭曲KL校准。我们提出一个优势尺度三方校准接口：同一个组内尺度分母同时决定了奖励分支强度、提示级批次权重，以及当奖励分支在原始基数尺度上重新表达时所诱导的有效KL校准。该接口解释了为什么RLOO/Dr.GRPO会让可信的小差距被KL项主导，而GRPO的标准差分母会无界放大微小差距。基于该接口，我们进一步提出了奖励分辨率协议和MaxNorm-AC，分别用于过滤亚分辨率差距并提供有界的基数……（原文摘要至此截断）

    arXiv:2609.19164v1 Announce Type: new  Abstract: In verifier-style RLVR, group-relative optimization often treats advantage scale as an implementation detail. This paper separates two low-variance cases: sub-resolution jitter that should not become a preference signal, and credible but small cardinal gaps that should be learned without distorting KL calibration. We propose an advantage-scale three-way calibration interface: the same within-group scale denominator simultaneously determines the reward-branch strength, prompt-level batch weight, and the effective KL calibration induced when the reward branch is re-expressed on the original cardinal scale. This interface explains why RLOO / Dr.GRPO can let credible small gaps become KL dominated, whereas GRPO's standard-deviation denominator can amplify tiny gaps without bound. Based on this interface, we further introduce the Reward-Resolution Protocol and MaxNorm-AC, respectively filtering sub-resolution gaps and providing bounded cardin
    
[^194]: VisKG-LM：将知识图谱编译为视觉记忆以用于多项选择题问答

    VisKG-LM: Compiling Knowledge Graphs into Visual Memory for Multiple-Choice Question Answering

    [https://arxiv.org/abs/2609.19158](https://arxiv.org/abs/2609.19158)

    VisKG-LM 将检索到的知识图谱子图一次性离线编译为保留分支结构的可视化图像并缓存为只读记忆，使语言模型在推理时无需重复在线编码图结构，从而将图编码与语言推理解耦，提升多项选择题问答的效率。

    

    知识图谱通常通过图神经网络编码检索到的子图，并在在线推理路径中将其与语言模型融合，从而集成到问答系统中。因此，无论在训练轮次、随机种子还是评估运行中，每当对一个问题-候选对进行评分时，同一个子图都会被从头重新编码，即使知识图谱本身从未改变。我们探讨检索到的知识图谱是否可以改为一次性离线编译，然后作为只读内存进行访问。VisKG-LM 证明这是可行的，其方法是将图编码与语言推理解耦。它将每个检索到的与候选相关的子图序列化为“关系标注路径”，并将结果渲染为图像，其二维布局保留了路径的分支结构。每张图像只需离线编码一次并缓存以供重复使用。在推理阶段，语言模型仅从文本对问题和候选进行上下文化处理，并且仅在其最终层……（摘要截断）

    arXiv:2609.19158v1 Announce Type: new  Abstract: Knowledge graphs are usually integrated into question answering by encoding a retrieved subgraph with a graph neural network and fusing it with the language model in the online inference path. The same subgraph is therefore re-encoded from scratch every time a pair is scored, across training epochs, seeds, and evaluation runs, even though the knowledge graph never changes. We ask whether the retrieved knowledge graphs can instead be compiled once, offline, and then accessed as read-only memory. VisKG-LM shows that it can, by decoupling graph encoding from language reasoning. It serializes each retrieved candidate-specific subgraph as Relation-Labeled Paths and renders the result as an image whose two-dimensional layout preserves the branching structure of the paths. Each image is encoded once, offline, and cached for reuse. At inference, the language model contextualizes the question and candidate from text alone, and only its final laye
    
[^195]: 停止去除停用词：一项沿袭的预处理默认设置如何扭曲法律文本即数据研究

    Stop Removing Stopwords: How an Inherited Preprocessing Default Distorts Legal Text-as-Data

    [https://arxiv.org/abs/2609.19153](https://arxiv.org/abs/2609.19153)

    本研究通过穷尽式单词消融实验，首次直接针对下游分类目标验证停用词去除这一沿袭自信息检索时代、从未被验证过的预处理默认设置，揭示其可能扭曲实证法学中基于TF-IDF和线性分类器的文本数据分析结果。

    

    实证法学研究日益将司法文本视为数据，其中许多研究仍依赖稀疏、可解释的处理流程——TF-IDF特征和线性分类器——因为文本特征往往是研究对象本身，而不仅仅是实现预测的手段。然而，这些流程继承了一系列源自二十世纪中叶信息检索领域的预处理默认设置，这些设置从未针对分类准确度进行过验证，其中最根深蒂固的是停用词去除。本研究引入了一种穷尽式的单词消融方法，直接根据下游目标衡量预处理步骤的效果，并将其应用于停用词去除这一最难撼动的案例。通过将最高法院数据库的标签与Caselaw Access Project的判决意见文本相匹配，该研究考察了两个涵盖F1提升空间的双分类任务：意识形态方向（不去除停用词的基线F1约0.68）和宪法与非宪法法律类型（约0.92），涉及7,66

    arXiv:2609.19153v1 Announce Type: new  Abstract: Empirical legal scholarship increasingly treats judicial text as data, and much of it still runs on sparse, interpretable pipelines -- TF-IDF features and linear classifiers -- because the textual feature is often the object of study, not merely a means to a prediction. Yet these pipelines inherit a chain of preprocessing defaults from mid-century information retrieval that were never validated against classification accuracy, the most entrenched being stopword removal. This study introduces an exhaustive single-word ablation that measures a preprocessing step's effect directly against the downstream objective, and applies it to stopword removal as the hardest case to dislodge. Matching Supreme Court Database labels to Caselaw Access Project opinion texts, it examines two binary tasks that bracket F1 headroom, ideological direction (no-removal baseline F1 ~ 0.68) and constitutional versus non-constitutional law type (~ 0.92), across 7,66
    
[^196]: 采样揭示风格：大语言模型激活中提示条件风格轴的无监督、免训练发现

    Sampling Reveals Style: Unsupervised, Training-Free Discovery of Prompt-Conditional Stylistic Axes in LLM Activations

    [https://arxiv.org/abs/2609.19150](https://arxiv.org/abs/2609.19150)

    该论文提出一种无需训练的无监督方法，通过对同一提示的高温重复采样补全进行主成分分析，自动发现并标记大语言模型激活中与提示相关的风格轴，并通过245个人类风格标注验证了其与人类自发风格需求的高度契合。

    

    大语言模型（LLM）在其隐藏激活中编码了丰富的风格结构，但要发现对于给定提示哪些风格维度是显著的，通常需要监督式对比数据。我们提出了一种无需训练、提示条件化的替代方法：我们对单个提示在较高温度下反复采样补全结果，对汇集的隐藏激活应用主成分分析（PCA），并根据极性生成结果自动标记所得的风格轴。我们在一项两阶段研究中，将发现的轴与245个人类风格标注进行了验证。在我们最强的模型（Qwen-3.5-4B-Instruct）上，前两个轴以72.8%的精确率和43.6%的宏召回率匹配用户自发请求的风格维度，75.6%的有效性评分认为这些轴的极性生成结果与其标签相符，标注者之间的相邻一致性达90.9%。风格轴的可发现性强烈依赖于模型本身：两个Qwen模型（原文摘要在此处截断）

    arXiv:2609.19150v1 Announce Type: new  Abstract: Large language models (LLMs) encode rich stylistic structure in their hidden activations, but discovering which stylistic dimensions are salient for a given prompt typically requires supervised contrastive data. We present a training-free, prompt-conditional alternative: we repeatedly sample completions of a single prompt at elevated temperature, apply Principal Component Analysis (PCA) to the pooled hidden activations, and label the resulting axes automatically from the pole generations. We validate the discovered axes against 245 human-elicited stylistic annotations in a two-phase study. On our strongest model (Qwen-3.5-4B-Instruct), the top two axes match spontaneously requested human dimensions with 72.8% precision and 43.6% macro-recall, and 75.6% of validity ratings judge the axes' polar generations accurate to their labels, with 90.9% adjacent inter-annotator agreement. Discoverability is strongly model-dependent: both Qwen models
    
[^197]: 超越静态几何的潜意识提示：因果深度与多标记混淆因素

    Subliminal Prompting Beyond Static Geometry: Causal Depth and Multi-Token Confounds

    [https://arxiv.org/abs/2609.19149](https://arxiv.org/abs/2609.19149)

    该论文首次将标记纠缠解释中的相关性测量与因果性测量明确区分开，通过在多个深度进行隐藏状态复制的因果干预实验，发现静态输出向量相似度随模型规模增大而失去预测力，而隐藏状态对所传递特质的因果控制能力则显著增强。

    

    潜意识学习表明，语言模型能够通过表面上与其无关的输出传递隐藏特质。一种被提出的解释是“标记纠缠”，即通过模型的输出词表将动物标记与数字标记关联起来。然而，现有的测量方法回答的是不同的问题：输出是否共变、固定的输出向量是否对齐、能否从隐藏状态中读出答案、或者该状态是否因果地控制答案。我们在一个固定的动物-数字提示协议中分别对上述各项进行测量。从Llama-3.1-8B到70B，固定输出向量相似度对行为的预测能力下降：配对平均相关变化为-0.080（95%置信区间[-0.127, -0.035]）。固定输出头读出在归一化深度AUC上未显示出可分辨的变化。为检验因果控制，我们在五个深度处将一个数字提示的临时答案位置状态复制到另一个提示中，并测量最终的动物得分跟随哪个提示。供体-对照AUC从0.254上升到0.540，

    arXiv:2609.19149v1 Announce Type: new  Abstract: Subliminal learning shows that language models can transmit a hidden trait through outputs that appear unrelated to it. One proposed explanation, token entanglement, links animal and number tokens through the model's output vocabulary. Yet existing measurements answer different questions: whether outputs co-vary, fixed output vectors align, an answer can be read from a hidden state, or that state causally controls the answer. We measure each separately in a fixed animal-number prompting protocol. From Llama-3.1-8B to 70B, fixed output-vector similarity predicts behavior less well: the paired mean correlation change is -0.080 (95% CI [-0.127, -0.035]). A fixed output-head readout shows no resolved change in normalized depth AUC. To test control, we copy the temporary answer-position state from one number prompt into another at five depths and measure which prompt the final animal score follows. Donor-control AUC rises from 0.254 to 0.540,
    
[^198]: 模型增长、递归与边界算子如何影响缩放指数

    How Model Growth, Recursion, and Boundary Operators Influence Scaling Exponents

    [https://arxiv.org/abs/2609.19107](https://arxiv.org/abs/2609.19107)

    该研究证明架构干预（尤其是模型增长和递归深度）可以改变预训练的缩放指数，使7.4B模型增长架构以约20倍更少的计算量匹配GPT-3 13B的性能，且计算效率优势随规模扩大而增加。

    

    缩放定律预测损失如何随计算量的增加而下降。我们证明，与传统观念相反，架构干预可以改变预训练中的缩放指数，从而随着计算量的增加带来性能的指数级提升。作为一个锚定点，我们考虑了循环Transformer的架构形式。尽管通常并不这样使用，但循环（也称为递归深度）通过在训练期间增加循环次数，提供了一种实现模型增长的机制。无论是否共享权重，模型增长都能对缩放指数带来最大的改变。特别地，一个7.4B参数的模型增长架构在CORE基准上以大约20倍更少的计算量匹配了GPT-3 13B的性能，并且其计算效率的提升随规模扩大而增加。此外，仅在普通Transformer中使用边界算子（即归一化并注入较早的块），也能带来不断增加的计算效率提升。

    arXiv:2609.19107v1 Announce Type: new  Abstract: Scaling laws predict how loss decreases with increases in computation. We show, contrary to conventional wisdom, that architectural interventions can modify scaling exponents in pre-training, leading to exponential improvements in performance with increases in computation. As an anchoring point, we consider the architectural formulation of looped transformers. Although not typically used in this way, looping, also known as recursive depth, provides a mechanism for model growth, by increasing the number of loops during training. Model growth, with and without shared weights, provides the biggest changes to the scaling exponents. In particular, a 7.4B model growth architecture matches GPT-3 13B on CORE with roughly $20\times$ less compute, and has compute efficiency gains that increase with scale. Moreover, simply using a boundary operator in a vanilla transformer, which normalizes and injects an earlier block, also provides increasing com
    
[^199]: 双重下降即最小作用量原理

    Double descent is the principle of least action

    [https://arxiv.org/abs/2609.19076](https://arxiv.org/abs/2609.19076)

    本文用统计力学解释了机器学习中的双重下降现象：将随机梯度训练视为温度为 $T$ 的粒子在损失能量景观上的扩散，有限时间的扩散带来有效权重衰减，使每个参数成为二次自由度，从而由能量均分定理导出测试误差随参数数量先升后降的规律。

    

    将模型的测试误差对其参数数量 $d$ 作图，误差先下降，在模型恰好能够拟合训练数据时达到峰值，随后再次下降，呈现出双重下降现象。我们用统计力学来解释这一现象：基于随机梯度的方法的训练轨迹是一个粒子，在诱导温度 $T$ 下于训练损失的能量景观上游走；一次已达平衡的训练会以相同的频率访问给定训练损失的每一个参数向量——这正是统计力学的基本假设——其概率由玻尔兹曼分布给出。由于训练从某个初始点出发，且只有有限的时间进行扩散，它会携带一种有效的权重衰减，这使得每个参数都成为一个二次型自由度。于是，能量均分定理将能量以 $T/2$ 的份额分配给这 $d$ 个自由度，因此在固定的训练损失下，增加参数会降低……（原文摘要在此处截断）

    arXiv:2609.19076v1 Announce Type: cross  Abstract: The test error of a model plotted against its number of parameters $d$ falls, peaks when the model can just fit the training data, and falls again, exhibiting the double descent phenomenon. We explain the phenomenon with statistical mechanics. The training trajectory of a stochastic gradient-based method is a particle wandering over the energy landscape of the training loss at an induced temperature $T$, and a run that has equilibrated visits every parameter vector of a given training loss equally often, the fundamental postulate of statistical mechanics, with probability given by the Boltzmann distribution. Because training starts at an initial point and has only finite time to diffuse, it carries an effective weight decay, which makes every parameter a quadratic degree of freedom. The equipartition theorem then distributes the energy among the $d$ degrees of freedom in shares of $T/2$, so at a fixed training loss adding parameters lo
    
[^200]: MoRE：复用专家混合模型

    MoRE: Mixture of Reused Experts

    [https://arxiv.org/abs/2609.18176](https://arxiv.org/abs/2609.18176)

    MoRE通过在相邻层组之间共享专家池并引入可学习的深度嵌入对每层输入进行条件化，在不增加参数的情况下扩展路由组合多样性，实现了比标准MoE和权重共享方法更低的困惑度和更强的下游性能。

    

    混合专家架构将模型容量与计算成本解耦，但随着专家数量增加，参数量线性增长会导致高昂的内存占用。循环Transformer通过复用层权重实现了参数效率，但通常缺乏进行竞争性语言建模所需的容量。我们提出复用专家混合，这是一种在相邻层组之间共享专家池的混合架构。每一层保留自己的路由器，但从更大的共享池中进行选择，从而在不增加额外参数的情况下扩展了路由组合的多样性。为了使共享专家能够区分不同的层，我们引入了轻量级的可学习深度嵌入，在路由之前对每层的输入进行条件化处理。在三个模型规模（114M至1.15B参数）上的实验表明，MoRE始终实现了比标准MoE和最先进的权重共享方法更低的困惑度和更强的下游性能。

    arXiv:2609.18176v1 Announce Type: cross  Abstract: Mixture-of-Experts (MoE) architectures decouple model capacity from computational cost, yet incur high memory footprints as parameters grow linearly with the number of experts. Recurrent Transformers achieve parameter efficiency by reusing layer weights, but typically lack the capacity for competitive language modeling. We propose Mixture of Reused Experts (MoRE), a hybrid that shares expert pools across groups of adjacent layers. Each layer retains its own router but selects from a larger shared pool, expanding the diversity of routing combinations without additional parameters. To enable shared experts to distinguish between layers, we introduce lightweight learnable depth embeddings that condition each layer's input before routing. Experiments across three model scales (114M-1.15B parameters) show that MoRE consistently achieves lower perplexity and stronger downstream performance than standard MoEs and state-of-the-art weight-shari
    
[^201]: 并非所有层都需要调优：诊断与引导视觉-语言-动作模型的适应

    Not All Layers Need Tuning: Diagnosing and Directing Adaptation in Vision-Language-Action Models

    [https://arxiv.org/abs/2609.18084](https://arxiv.org/abs/2609.18084)

    该研究通过在五个不同架构的VLA模型上测量区域隔离微调下的适应成本，揭示了不同类型的分布偏移会系统性集中在特定网络区域（外观变化集中于视觉编码器、指令变化集中于语言骨干、新物体变化集中于视觉编码器与动作头），并提出了一种仅凭十个无标注观测、无需微调即可诊断适应成本并针对性分配适配容量的流水线。

    

    为新的部署环境微调视觉-语言-动作（VLA）模型成本高昂，然而大多数方法对网络的每个区域都应用统一容量的适配器，仿佛每个区域都需要同等程度的调整。本文在五个架构各异的VLA模型（OpenVLA-OFT、π₀、SmolVLA、DTP、Octo；参数量从93M到7B）上检验了这一假设。通过在区域隔离微调下以归一化参数位移来测量各区域的适应成本，研究揭示了一个适应谱系：外观变化使适应成本集中在视觉编码器，指令变化集中在语言骨干网络，而新物体变化则集中在视觉编码器与动作头，这一规律在全部五种架构中均成立。为利用这一结构性规律，我们提出了一个“观察、诊断、分配、适应”的流水线。仅需十个无标注的目标环境观测且无需微调，诊断模块即可通过结合无参考梯度估计各区域的适应成本。

    arXiv:2609.18084v1 Announce Type: cross  Abstract: Fine-tuning a Vision-Language-Action (VLA) model for a new deployment environment is expensive, yet most methods apply uniform-capacity adapters to every network region as if every region requires equal adjustment. This paper tests that assumption on five architecturally diverse VLAs (OpenVLA-OFT, $\pi_0$, SmolVLA, DTP, Octo; 93M-7B parameters). Measuring per-region adaptation cost as normalized parameter displacement under region-isolated fine-tuning reveals an adaptation spectrum in which appearance shifts concentrate cost in the vision encoder, instruction shifts in the language backbone, and novel-object shifts in the vision encoder together with the action head, across all five architectures. To exploit this structure, we introduce a pipeline that observes, diagnoses, allocates, and adapts. From ten unlabeled target observations and without fine-tuning, the diagnostic estimates per-region cost by combining reference-free gradient 
    
[^202]: Fathom：面向卸载KV缓存稀疏解码的逐查询读取深度

    Fathom: Per-Query Read Depth for Sparse Decoding over Offloaded KV Caches

    [https://arxiv.org/abs/2609.17652](https://arxiv.org/abs/2609.17652)

    Fathom提出一种让每个查询自适应决定键通道读取位数的稀疏解码方法，通过比特平面存储与逆注水式比特预算分配，在百万token卸载KV缓存场景下实现比现有136位扫描方法快1.67倍的GPU解码速度，同时保持更低注意力误差。

    

    当智能体会话运行至百万token且同时驻留多个会话时，KV缓存及其排序索引存放在主机内存中，而针对top-k步骤对所有n个键进行排序的扫描成为限制解码速度的流量瓶颈。我们提出Fathom，一种由每个查询自主决定读取每个键通道多少比特的键扫描方法。4位K缓存以通道优先的方式存储为比特平面，因此t个平面的前缀恰好构成该通道的t位量化器，查询通过对方差加权通道重要性进行逆注水来分配其比特预算。在Qwen3-8B上处理一百万token时，解码步骤的GPU时间比Double Sparsity、Loki和SparQ r=32的136位扫描快1.67倍；在与SparQ的68位读取（r=16）相同的GPU时间内，Fathom读取的字节数减少18%，且在七个模型与上下文设置中的六个上注意力误差更低。在RULER风格的任务上，每次逐token扫描均与精确top-k解码结果相匹配。

    arXiv:2609.17652v1 Announce Type: cross  Abstract: When agentic sessions run to a million tokens with many sessions resident at once, the KV cache and the index that ranks it live in host memory, and the scan that ranks all n keys for a top-k step becomes the traffic that bounds decoding. We present Fathom, a key scan in which each query decides how many bits of each key channel to read. The 4-bit K cache is stored channel-major as bit planes, so a prefix of t planes is exactly the channel's t-bit quantizer, and the query spends its bit budget by reverse water-filling over the variance-weighted importance of its channels. At one million tokens on Qwen3-8B a decode step is 1.67x faster in GPU time than with the 136-bit scans of Double Sparsity, Loki and SparQ r=32, and in the same GPU time as SparQ's 68-bit read (r=16) Fathom reads 18% fewer bytes with lower attention error on six of seven model and context settings. On RULER-style tasks every per-token scan matches exact top-k decoding
    
[^203]: 通过精确分布式样条合并从碎片化观测中恢复物理参数

    Recovering Physical Parameters from Fragmented Observations via Exact Distributed Spline Merging

    [https://arxiv.org/abs/2609.16579](https://arxiv.org/abs/2609.16579)

    本文提出精确分布式样条合并方法，各数据持有者仅需共享局部Gram矩阵和矩向量即可获得与集中式拟合数学上完全相同的解，并通过场重建与导数提取流程从分布式碎片观测中实现物理参数推断。

    

    科学测量通常分布在不同的地点、时间段和机构之间。将这些碎片组合成连续、可微的场，能够从其导数中恢复控制性物理参数。本文为实现这一目标做出了两项贡献。首先，将固定基岭回归统计量的既定可加结构应用于张量积样条场：每个数据持有者计算局部Gram矩阵和矩向量，合并后的解在数学上与集中式拟合完全相同，无需共享原始数据，也无需迭代同步。这一性质特定于固定特征的平方误差设置；本推导并未为一般联合训练的多层网络建立类似的保证。其次，一个完整的处理流程通过场重建、导数提取等步骤，将分布式观测与物理参数推断连接起来。

    arXiv:2609.16579v1 Announce Type: new  Abstract: Scientific measurements are frequently distributed across locations, time periods, and institutions. Combining such fragments into a continuous, differentiable field enables recovering governing physical parameters from its derivatives. This paper makes two contributions toward that goal. First, the established additive structure of fixed-basis ridge-regression statistics is applied to tensor-product spline fields: each data holder computes a local Gram matrix and moment vector, and the merged solution is mathematically identical to centralized fitting, with no raw data shared and no iterative synchronization. This property is specific to the fixed-feature squared-error setting; the present derivation does not establish an analogous guarantee for general jointly trained multilayer networks. Second, a complete pipeline connects distributed observations to physical parameter inference through field reconstruction, derivative extraction, an
    
[^204]: 并非所有关系都是平等的：面向基于溯源入侵检测的关系平衡与校准图学习

    Not All Relations Are Equal: Relation-Balanced and Calibrated Graph Learning for Provenance-Based Intrusion Detection

    [https://arxiv.org/abs/2609.16462](https://arxiv.org/abs/2609.16462)

    提出无监督框架RECAL，通过关系平衡的掩码图学习捕捉稀有交互模式，并结合针对各关系良性错误分布的误差校准机制，在DARPA E3数据集上实现最高99.99%的F1分数，有效降低基于溯源入侵检测中的误报和漏检风险。

    

    基于溯源的入侵检测系统（PIDS）通过分析系统交互来检测高级持续性威胁（APT）。然而，现有方法在很大程度上统一对待各种关系，忽视了统计异质性；在CADETS中，不同关系的出现频率差异约为14万倍。这可能导致PIDS更加关注频繁出现的关系，而忽视不同关系间正常错误水平的差异，从而增加误报和漏检的风险。我们提出了RECAL，一个无监督框架，采用关系平衡的掩码图学习来更好地捕捉稀有的交互模式。该框架进一步针对每种关系的良性错误分布对重构误差进行校准，以产生可比较的异常证据，帮助区分攻击行为与良性行为并减少误报。在三个DARPA E3数据集上，RECAL分别取得了99.99%、99.93%和99.99%的F1分数，超越了最佳基线方法。

    arXiv:2609.16462v1 Announce Type: cross  Abstract: Provenance-Based Intrusion Detection Systems (PIDSs) detect Advanced Persistent Threats (APTs) by analyzing system interactions. However, existing methods largely treat relations uniformly, overlooking statistical heterogeneity; in CADETS, relation frequencies differ by approximately $140{,}000\times$. This may cause PIDSs to focus more on frequent relations and overlook differences in normal error levels across relations, increasing the risk of false alarms and missed detections. We present RECAL, an unsupervised framework using relation-balanced masked graph learning to better capture rare interaction patterns. It further calibrates reconstruction errors against each relation's benign error distribution to produce comparable anomaly evidence, helping distinguish attacks from benign behavior and reduce false alarms. On three DARPA E3 datasets, RECAL achieves F1 scores of 99.99\%, 99.93\%, and 99.99\%, outperforming the best baseline o
    
[^205]: 面向任务的残差AddUNet：全速率表示的完美重构路由

    Task-Directed Residual AddUNet:Perfect-Reconstruction Routing for Full-Rate Representations

    [https://arxiv.org/abs/2609.15857](https://arxiv.org/abs/2609.15857)

    本文证明受约束加性U-Net精确等价于完美重构多速率滤波器组，并提出残差全速率PR架构，可将任务无关信息从面向任务的表示中路由出去，同时保证任意线性或非线性路由算子下的精确重构，且无需可逆性、重构损失或解码器。

    

    本文建立了AddUNet及其全速率实现的完美重构（PR）解释，并提出了一种用于任务导向表示学习的残差全速率PR架构。研究证明了受约束加性U-Net的幸存者—跳跃结构与临界采样的多速率PR滤波器组精确等价。全速率公式在保持完美重构的同时，消除了临界采样系统的互补子带限制。随后提出的残差全速率PR架构，能够将任务无关、干扰或冗余结构逐步从面向任务的幸存者中路由出去，同时显式保留被路由的信息。对于任意形状兼容的线性或非线性路由算子，精确重构均可得到保证，且无需可逆性、匹配综合滤波器组、重构损失或学习解码器。由此产生的架构将表示设计与…（原文摘要在此截断）

    arXiv:2609.15857v1 Announce Type: new  Abstract: This paper establishes a perfect-reconstruction (PR) interpretation of AddUNet and its full-rate realization, and introduces a Residual Full-Rate PR architecture for task-directed representation learning. The survivor--skip structure of a constrained additive U-Net is shown to be exactly equivalent to a critically sampled multirate PR filter bank. The full-rate formulation removes the complementary-subband restrictions of the critically sampled system while preserving PR. A Residual Full-Rate PR architecture is then proposed to progressively route task-irrelevant, nuisance, or redundant structure away from the task-facing survivor while retaining the routed information explicitly. Exact reconstruction is guaranteed for arbitrary shape-compatible linear or nonlinear routing operators, without requiring invertibility, a matched synthesis bank, reconstruction loss, or learned decoder. The resulting architecture decouples representation desi
    
[^206]: 当更快的VLA部署改变闭环行为时：SmolVLA在PyTorch与ONNX变体间的任务成功率-延迟分析

    When Faster VLA Deployment Changes Closed-Loop Behavior: Task Success-Latency Analysis of SmolVLA Across PyTorch and ONNX Variants

    [https://arxiv.org/abs/2609.14146](https://arxiv.org/abs/2609.14146)

    该论文通过在消费级GPU上系统对比SmolVLA的PyTorch与ONNX部署变体，揭示了推理延迟降低约一半会显著改变闭环任务行为（Spatial成功率从70%降至约40%，Object保持不变），且导出图的审计表明所谓INT8并非真正的算子级量化，而静态语言宽度调整可部分恢复性能损失。

    

    视觉-语言-动作（VLA）模型的部署可以在降低推理延迟的同时改变闭环任务行为。我们在RTX 2060（6 GB）显卡上、于LIBERO Spatial和Object任务套件（MuJoCo 3.3.2、LeRobot 0.6.1、随机种子42）中评估HuggingFaceVLA/smolvla_libero，将PyTorch+AMP与ONNX Runtime CUDA执行提供程序（CUDA EP）进行对比。主评估每个套件使用100个回合；配对滚动评估每个套件使用300个回合。PyTorch+AMP在p99延迟1181毫秒下达到Spatial/Object任务70.0%/88.0%的成功率。请求FP16和请求INT8的ONNX将tether-inspect p99延迟分别降至601毫秒和532毫秒，但Spatial成功率降至41.0%和40.0%，而Object成功率保持在89.0%。图审计显示这些导出工件是字节完全相同的FP32图，因此“请求INT8”这一行并非算子级INT8量化。静态语言宽度消融实验（16/24/32个token）分别产生41.0%、75.0%和71.0%的Spatial成功率；宽度24和32可恢复大部分Spatial性能下降，而Object成功率……

    arXiv:2609.14146v1 Announce Type: cross  Abstract: Vision-language-action (VLA) deployment can reduce inference latency while changing closed-loop task behavior. We evaluate HuggingFaceVLA/smolvla_libero on an RTX 2060 (6 GB) in LIBERO Spatial and Object (MuJoCo 3.3.2, LeRobot 0.6.1, seed 42), comparing PyTorch+AMP with ONNX Runtime CUDA Execution Provider (CUDA EP). The main evaluation uses 100 episodes/suite; a paired rollout uses 300 episodes/suite. PyTorch+AMP reaches 70.0%/88.0% Spatial/Object success at 1181 ms p99. Requested-FP16 and requested-INT8 ONNX reduce tether-inspect p99 to 601 ms and 532 ms, while Spatial success falls to 41.0% and 40.0% and Object remains at 89.0%. A graph audit shows those artifacts are byte-identical FP32 graphs, so the requested-INT8 row is not operator-level INT8 quantization. A static language-width ablation (16/24/32 tokens) yields Spatial success of 41.0%, 75.0%, and 71.0%; widths 24 and 32 recover much of the Spatial drop while Object success a
    
[^207]: NeuroFlex：无损元件级 ANN-SNN 协同执行实现高效稀疏推理

    NeuroFlex: Lossless Element-Level ANN-SNN Co-Execution for Efficient Sparse Inference

    [https://arxiv.org/abs/2609.14092](https://arxiv.org/abs/2609.14092)

    NeuroFlex是首个能在单个输出元素级别无损切换ANN和SNN执行模式的加速器，通过将整数精确的ANN-SNN等价性扩展到元素级别并配合成本引导调度器，实现了97-99%的PE利用率，相比纯ANN基线降低57-67%的EDP，相比纯SNN基线获得高达2.5倍加速。

    

    稀疏DNN加速器通常专注于ANN（人工神经网络）或SNN（脉冲神经网络）的执行，当工作负载特性在层内部发生变化时，会造成能耗或延迟的浪费。在层或块粒度上切换模式的混合加速器设计存在PE（处理单元）利用率低的问题，因为当一种核心类型工作时，另一种核心类型就会闲置。NeuroFlex是首个能够将每个输出元素独立分配给ANN或SNN执行模式且零精度损失的加速器。我们将整数精确的ANN-SNN等价性从层级别扩展到单个输出元素，从而实现无转换误差的模式切换。一个离线的成本引导调度器根据边际能耗-延迟权衡为每个元素评分，并在各PE之间打包工作，实现了97-99%的PE利用率，而层粒度混合设计仅为40-45%。与强大的纯ANN基线相比，NeuroFlex将EDP（能耗延迟积）降低了57-67%，与双稀疏纯SNN基线相比实现了高达2.5倍的加速。我们的成本引导调度器改进了…（原文在此截断）

    arXiv:2609.14092v1 Announce Type: cross  Abstract: Sparse DNN accelerators specialize in ANN or SNN execution, leaving energy or latency on the table when workload characteristics vary within a layer. Hybrid accelerator designs that switch modes at layer or tile granularity suffer from low PE utilization since one core type idles whenever the other is active. NeuroFlex is the first accelerator to assign every output element independently to ANN or SNN execution mode with zero accuracy loss. We extend integer-exact ANN-SNN equivalence from layers to individual output elements, thereby enabling mode switching with no conversion error. An offline cost-guided scheduler scores each element by its marginal energy-delay trade-off and packs work across PEs, achieving 97-99% PE utilization compared to 40-45% for layer-wise hybrids. NeuroFlex reduces EDP by 57-67% over a strong ANN-only baseline and delivers up to 2.5x speedup over a dual-sparse SNN-only baseline. Our cost-guided scheduler impro
    
[^208]: 一个用于大语言模型针对性危害缓解的高效模块化框架

    An Efficient and Modular Framework for Targeted Harm Mitigation in LLMS

    [https://arxiv.org/abs/2609.13624](https://arxiv.org/abs/2609.13624)

    提出了一种结合Activated LoRA适配器与上下文感知路由机制的模块化纠正框架，可在生成过程中以低延迟、有针对性的方式缓解大语言模型的有害输出，同时提升模型对齐性能。

    

    摘要：大语言模型（LLMs）是强大的零样本学习器，但仍然容易与人类偏好产生不一致，经常输出带有偏见、有毒或其他有害的内容。现有的对齐方法虽然有效，但成本高昂且与模型紧密耦合，限制了灵活性和可扩展性。我们提出了一个模块化纠正框架，通过Activated LoRA（aLoRA）适配器和上下文感知路由机制来增强预训练的大语言模型，以消除模型失调响应带来的危害。我们的方法使专家适配器能够在序列中间激活而不使KV缓存失效，从而在生成过程中实现低延迟的针对性纠正。每个专家都被训练用于检测和缓解特定类型的危害，例如偏见或毒性。一个经过学习的路由器根据模型的中间输出动态选择合适的专家。我们证明该系统在标准安全基准测试中改善了对齐效果，同时保留了……

    arXiv:2609.13624v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are powerful zero-shot learners but remain prone to misalignment with human preferences, often producing biased, toxic, or otherwise harmful outputs. Existing alignment methods, while effective, are costly and tightly coupled to the model, limiting flexibility and scalability. We propose a modular correction framework that augments pretrained LLMs with Activated LoRA (aLoRA) adapters and a context-aware routing mechanism to eliminate harms from misaligned model responses. Our approach enables expert adapters to activate mid-sequence without invalidating the KV cache, allowing low-latency, targeted correction during generation. Each expert is trained to detect and mitigate specific harms, such as bias or toxicity. A learned router dynamically selects appropriate experts based on the models intermediate outputs. We demonstrate that our system improves alignment on standard safety benchmarks while preserving t
    
[^209]: 强化学习中的最优价值推断

    Optimal Value Inference for Reinforcement Learning

    [https://arxiv.org/abs/2609.09981](https://arxiv.org/abs/2609.09981)

    该论文提出了一种基于Neyman正交性的去偏估计方法，通过softmax近似的自诱导贝尔曼方程构建冗余参数，实现了强化学习中最优价值的有效统计推断，且在视界发散和行为策略随时间变化的情况下依然保持渐近正态性。

    

    我们研究强化学习中离线推断最优价值的问题。我们将两个新的冗余参数推导为自诱导贝尔曼方程的不动点，其中我们用最大贝尔曼算子的softmax对应形式进行近似。我们通过Neyman正交性提出了一个去偏估计量，并在视界发散的情况下建立了其渐近正态性，即使行为策略随时间变化，只要这些冗余参数达到许多机器学习方法所能实现的统计收敛速率即可。我们为这些冗余参数提供了具体的估计程序，并证明它们能够带来有效的统计推断。合成实验验证了我们推断方法的数值性能，我们还在现实决策问题中实现了该方法，包括自行车重新调配和AI智能体工具使用。

    arXiv:2609.09981v1 Announce Type: new  Abstract: We study offline inference for the optimal value in reinforcement learning. Two new nuisances are derived as fixed points of a self-induced Bellman equation, in which we approximate the maximum Bellman operator by its softmax correspondence. We propose a debiased estimator through the Neyman orthogonality and establish its asymptotic normality under diverging horizons even when the behavior policy changes with time, as long as the nuisances have the statistical rates that can be achieved by many machine learning methods. We provide a concrete estimating procedure for these nuisances and show they can lead to valid inference. Synthetic experiments validate the numerical performance of our inference method, and we implement it in real-life decision-making problems, including bike repositioning and AI agentic tool use.
    
[^210]: 具有可证明保证的稀疏数据增强优化方法

    Sparse Data Augmentation for Optimization with Provable Guarantees

    [https://arxiv.org/abs/2609.08133](https://arxiv.org/abs/2609.08133)

    该论文证明了在非凸几何机器学习优化中，使用优化前采样的少量固定数据变换进行稀疏数据增强，梯度下降仅需对数级加多项式级的群变换查询次数，即可在概率保证下逼近完全数据增强目标的稳定点。

    

    在几何机器学习中出现的非凸优化问题里，数据增强通常被用于通过对数据变换后的经验损失取平均来促进不变性。然而，计算完全增强后的目标函数需要访问变换群 $G$ 中的每一个元素，当 $G$ 非常庞大或只能通过采样方式访问时，这种做法的代价可能高得令人望而却步。我们研究了是否可以使用在优化开始前获取、并在之后的优化过程中重复使用的一小部分固定变换样本，来近似完全增强的目标函数。在适当的正则性条件下，我们证明，至少以 $1-\delta$ 的概率，对由此得到的稀疏增强目标函数执行梯度下降（GD），仅使用 $\mathcal{O}\bigl((\log |G|+\log(1/\delta))/\varepsilon^2\bigr)$ 次群变换预言机查询，即可返回完全增强目标函数的一个 $\varepsilon$-稳定点。相比之下，标准的群随机梯度（摘要在此处被截断）

    arXiv:2609.08133v1 Announce Type: cross  Abstract: In nonconvex optimization problems arising in geometric machine learning, data augmentation is commonly used to promote invariance by averaging empirical losses over transformations of the data. Computing the fully augmented objective, however, requires access to every element of the transformation group $G$, which may be prohibitively expensive when $G$ is large or accessible only through sampling. We study whether full augmentation can instead be approximated using a small, fixed sample of transformations acquired before optimization and reused thereafter. Under suitable regularity conditions, we show that, with probability at least $1-\delta$, gradient descent (GD) on the resulting sparsely augmented objective returns an $\varepsilon$-stationary point of the fully augmented objective using $\mathcal{O}\bigl((\log |G|+\log(1/\delta))/\varepsilon^2\bigr)$ group-transformation-oracle queries. By comparison, standard group stochastic gr
    
[^211]: LayerRoute：面向视觉-语言-动作策略的动作条件混合层路由

    LayerRoute: Action-Conditioned Mixture-of-Layers Routing for Vision-Language-Action Policies

    [https://arxiv.org/abs/2609.06079](https://arxiv.org/abs/2609.06079)

    提出LayerRoute，一个动作条件的混合层路由接口，使VLA策略能够根据具体动作需求自适应地访问和混合VLM不同层次的表示，突破了传统固定层分配的限制。

    

    arXiv:2609.06079v1 公告类型：新论文 摘要：视觉-语言-动作（VLA）策略利用预训练的视觉-语言模型（VLM）来指导机器人控制的动作生成。VLM提供跨层演化的分层视觉-语义表示，从局部视觉几何到抽象的、与语言对齐的语义；因此，不同的操作任务可能需要不同的层表示混合。同时，动作模块在动作计算过程中维护不断演化的中间表示，这些表示可能为后续决策提供有用信息。然而，现有的VLA接口在表示访问方面的灵活性有限：VLM信息通过为每个动作层固定的层分配来暴露，而中间动作状态仅通过残差流隐式传播，缺乏显式复用。我们提出了LayerRoute，一个动作条件的表示路由接口，使VLA策略能够自适应地访问VLM的各层表示。

    arXiv:2609.06079v1 Announce Type: new  Abstract: Vision-Language-Action (VLA) policies leverage pretrained vision-language models (VLMs) to guide action generation for robot control. VLMs provide hierarchical visual-semantic representations that evolve across layers, from local visual geometry to abstract, language-aligned semantics; different manipulation tasks may therefore require different mixtures of layer representations. Meanwhile, the action module maintains intermediate representations that evolve throughout action computation and may provide useful information for subsequent decisions. However, existing VLA interfaces offer limited flexibility in representation access: VLM information is exposed through fixed layer assignments for each action layer, while intermediate action states are only propagated implicitly through residual streams without explicit reuse. We introduce LayerRoute, an action-conditioned representation routing interface that enables adaptive access to VLM l
    
[^212]: 蒸馏之前先验证：面向在线策略蒸馏的提示级教师门控

    Verify Before You Distill: Prompt-Level Teacher Gating for On-Policy Distillation

    [https://arxiv.org/abs/2609.02998](https://arxiv.org/abs/2609.02998)

    该论文提出教师门控在线策略蒸馏（TGOPD），通过经验证器评分的教师探测在提示级别先验证教师模型的可靠性，将可靠提示路由到密集OPD监督、不可靠提示路由到基于验证器的GRPO，从而避免“自信但错误”的教师模型诱导误导性更新。

    

    在线策略蒸馏（OPD）通过在学生模型自身的生成结果上提供来自冻结教师模型的密集token级监督来加速后训练过程。原始的OPD在所有提示上均匀地应用这种监督，而不检查教师模型对每个提示是否可靠。由于反向KL散度具有模式寻求特性，一个自信但错误的教师模型可能导致强烈却具有误导性的更新。分布性代理指标（如熵或教师-学生似然一致性）只能衡量不确定性或一致性，但无法直接验证结果的正确性。我们提出了教师门控在线策略蒸馏（TGOPD），其核心原则是在接受密集监督之前，应在提示级别验证教师模型的可靠性。TGOPD通过一小组经验证器评分的教师探测样本估计教师可靠性，并将每个提示专门路由到密集OPD（当可靠性检查通过时）或基于验证器的GRPO（当检查不通过时）。在4B和3...（摘要内容不完整）

    arXiv:2609.02998v1 Announce Type: cross  Abstract: On-policy distillation (OPD) accelerates post-training by providing dense token-level supervision from a frozen teacher on the student's own rollouts. Vanilla OPD applies this supervision uniformly across prompts, without checking whether the teacher is reliable for each prompt. Because reverse KL is mode-seeking, a confidently wrong teacher can induce a strong yet misleading update. Distributional proxies, such as entropy or teacher-student likelihood agreement, measure uncertainty or agreement but do not directly verify outcome correctness. We introduce Teacher-Gated On-Policy Distillation (TGOPD), built on the principle that teacher reliability should be verified at the prompt level before dense supervision is admitted. TGOPD estimates reliability from a small set of verifier-scored teacher probes and routes each prompt exclusively to dense OPD when the reliability check passes or to verifier-grounded GRPO otherwise. Across 4B and 3
    
[^213]: 平滑Transformer前馈网络的曲率密码分析

    Curvature Cryptanalysis of Smooth Transformer Feed-Forward Networks

    [https://arxiv.org/abs/2608.28843](https://arxiv.org/abs/2608.28843)

    该论文提出了一种基于曲率（二阶Hessian信息）的密码分析方法，证明采用GELU或SiLU等平滑激活函数的Transformer前馈网络会通过二阶泄漏通道泄露其隐藏权重方向，仅需8193次黑盒查询（16个投影Hessian）即可高精度提取FFN结构，并将查询成本降低了16倍。

    

    arXiv:2608.28843v1 公告类型：cross 摘要：我们证明了平滑的两层前馈网络（FFN）在FFN分支处的选定输入-原始输出预言机下，会暴露出一条额外的结构性模型提取通道；我们研究了在选定输入-原始输出访问条件下、且无法访问参数、梯度或内部激活时，采用GELU或SiLU激活函数的Transformer前馈分支；我们利用了一种二阶泄漏通道，其中投影输入Hessian矩阵构成了由FFN输入权重所诱导的相同隐藏对称秩一因子的不同混合。我们将由此产生的Hessian收集问题形式化为部分对称分解，以建立局部可辨识性和稳定性的条件，并利用向量输出模板重用将结构性查询成本降低了16倍。在独立训练的CIFAR-10视觉Transformer上，仅需16个投影Hessian（对应8193次黑盒查询）即可恢复隐藏的FFN方向，其平均绝对余弦对齐度……（摘要原文在此处截断）

    arXiv:2608.28843v1 Announce Type: cross  Abstract: We show that smooth two-layer feed-forward networks (FFNs) expose an additional structural model extraction channel under a chosen-input raw-output oracle at the FFN branch; consider transformer FFN branches with GELU or SiLU activations under chosen-input raw-output access, without access to parameters, gradients, or internal activations; exploit a second-order leakage channel in which projected input Hessians form different mixtures of the same hidden symmetric rank-one factors induced by the FFN input weights. We formalize resulting Hessian collection as a partially symmetric decomposition to establish conditions for local identifiability and stability to exploit vector-output stencil reuse to reduce the structural query cost by a factor of 16. On independently trained CIFAR-10 vision transformers, only 16 projected Hessians, corresponding to 8193 black-box queries, recover the hidden FFN directions with average absolute cosine alig
    
[^214]: 分布强化学习中分位数时序差分学习的有限样本分析

    A Finite Sample Analysis for Quantile Temporal Difference Learning in Distributional Reinforcement Learning

    [https://arxiv.org/abs/2608.27313](https://arxiv.org/abs/2608.27313)

    本文首次为表格型分布强化学习中的同步分位数时序差分学习提供了全局有限样本保证，通过分离稳定性机制，证明了其收敛速率对分位数数量无多项式依赖。

    

    arXiv:2608.27313v1 公告类型：交叉 摘要：我们在表格型分布强化学习中，为同步分位数时序差分学习（QTD）建立了全局有限样本保证。证明分离了两种稳定性机制。一个基于奖励累积分布函数的顺序单调性和分布贝尔曼算子的$W_\infty$收缩的全局比较论证，将任意初始化的迭代带入一个局部邻域。在该邻域内，我们对QTD均值场进行线性化。其雅可比矩阵是一个非奇异$M$-矩阵，相关的正半群允许进行方差敏感的鞅分析。对于步长$\alpha_t=c(t+1)^{-a}$，其中$a\in(1/2,1)$，主导的最后迭代波动阶为$\widetilde O\bigl(T^{-a/2}/\sqrt{1-\gamma}\bigr)$，且对分位数数量没有多项式依赖。确定性瞬态和所需的预热时间仍可能依赖于最小的贝尔曼误差。

    arXiv:2608.27313v1 Announce Type: cross  Abstract: We establish a global finite-sample guarantee for synchronous quantile temporal-difference learning (QTD) in tabular distributional reinforcement learning. The proof separates two stability mechanisms. A global comparison argument, based on the order monotonicity of reward cumulative distribution functions and the $W_\infty$ contraction of the distributional Bellman operator, brings an arbitrarily initialized iterate into a local neighborhood. Inside that neighborhood, we linearize the QTD mean field. Its Jacobian is a nonsingular $M$-matrix, and the associated positive semigroup permits a variance-sensitive martingale analysis. For stepsizes $\alpha_t=c(t+1)^{-a}$ with $a\in(1/2,1)$, the leading last-iterate fluctuation is of order $\widetilde O\bigl(T^{-a/2}/\sqrt{1-\gamma}\bigr)$ and has no polynomial dependence on the number of quantiles. The deterministic transient and the required burn-in can still depend on the smallest Bellman-
    
[^215]: GRAS：用于离散扩散模型免训练奖励对齐的引导式低方差提议与自适应选择

    GRAS: Guided Reduced-Variance Proposals and Adaptive Selection for Training-Free Reward Alignment in Discrete Diffusion

    [https://arxiv.org/abs/2608.26585](https://arxiv.org/abs/2608.26585)

    本文提出GRAS方法，通过Rao-Blackwell化提议和自适应温度选择，在不增加降噪器成本的情况下，显著提升离散扩散模型免训练奖励对齐的效率和稳定性。

    

    arXiv:2608.26585v1 公告类型：新 摘要：离散扩散模型已成为序列数据生成器中强大且广泛采用的一类，而在推理时无需重新训练即可将其引导至下游奖励，正变得越来越重要。这种免训练引导通过梯度引导、搜索或两者结合来实现。我们研究了结合模式，并识别出通常运行方式中的两个弱点：引导提议从单个噪声样本估计梯度，而搜索则以固定温度重新采样粒子，忽略了奖励在每个去噪步骤中的分布。我们通过一组小改动解决了这两个问题，且不增加降噪器成本。对于提议，我们通过Rao-Blackwell化揭示来降低可微奖励的估计方差，并为不可微奖励使用留一法基线；对于搜索，我们将每步值标准化为组相对优势，并证明它会坍缩为单个活跃项。

    arXiv:2608.26585v1 Announce Type: new  Abstract: Discrete diffusion models have become a strong, widely adopted class of generators for sequence data, and steering them toward a downstream reward at inference time, without any retraining, is increasingly important. Such training-free steering is done by gradient guidance, by search, or by combining the two. We study the combined regime and identify two weaknesses in how it is usually run: the guided proposal estimates its gradient from a single noisy sample, and the search then resamples particles at a fixed temperature that ignores how rewards spread across each denoising step. We address both with a small set of changes that add no denoiser cost. For the proposal, we lower the estimator variance with a Rao-Blackwellized reveal for differentiable rewards and a leave-one-out baseline for non-differentiable ones; for the search, we standardize the per-step values into a group-relative advantage and prove it collapses to a single active 
    
[^216]: D$^3$-MOPD：用于高效多教师蒸馏的自适应动态领域调度

    D$^3$-MOPD: Adaptive Dynamic Domain ScheDuling for Efficient Multi-Teacher Distillation

    [https://arxiv.org/abs/2608.24987](https://arxiv.org/abs/2608.24987)

    本文提出了一种零开销的动态领域调度方法，通过利用训练中已有的反向KL信号在线调整数据混合比例，解决了多教师蒸馏中固定混合导致的计算浪费问题。

    

    arXiv:2608.24987v1 公告类型：新  摘要：多教师在线策略蒸馏（MOPD）通过最小化学生自身轨迹上每个领域的反向KL散度，将多个领域专家教师蒸馏到一个单一学生模型中。现有方法通常在训练前固定每个领域的数据混合比例，忽略了不同领域以显著不同速度收敛的事实：有些领域早期就达到平台期，而其他领域在整个训练预算中持续改进。因此，固定的混合比例会浪费计算资源在快速收敛的领域上，并对较慢收敛的领域训练不足。为解决这一问题，我们提出了D$^3$-MOPD（用于MOPD的动态领域调度），这是一种零开销的调度器，它重新利用训练过程中已产生的每个领域反向KL信号，在线调整领域混合比例。该调度器在训练过程之外异步运行，一个进程外的监视器定期跟踪每个领域的KL轨迹，估计剩余提升空间和当前改进速率，并据此调整分配。

    arXiv:2608.24987v1 Announce Type: new  Abstract: Multi-teacher on-policy distillation (MOPD) distills several domain-expert teachers into a single student by minimizing per-domain reverse-KL divergence on the student's own rollouts. Existing approaches typically fix the per-domain data mixture before training, overlooking the fact that different domains converge at substantially different rates: some plateau early while others continue to improve throughout the training budget. A fixed mixture therefore wastes compute on fast-converging domains and undertrains slower-converging ones. To address this, we propose D$^3$-MOPD (Dynamic Domain ScheDuling for MOPD), a zero-overhead scheduler that repurposes the per-domain reverse-KL signal already produced during training to adapt the domain mixture online. Running asynchronously outside the training process, an off-process watcher periodically tracks each domain's KL trajectory, estimates remaining headroom and current improvement rate, and 
    
[^217]: CatchBench：何时能够捕获智能体故障？

    CatchBench: When Can an Agent Failure Be Caught?

    [https://arxiv.org/abs/2608.22808](https://arxiv.org/abs/2608.22808)

    CatchBench提出了一个在三种信息状态（运行前配置PRE、运行中轨迹前缀LIVE、完成后轨迹POST）下评估智能体故障审计能力的基准，通过七项任务契约对72个参赛者进行评分，并发现现有方法在大多数对比中缺乏区分能力。

    

    何时能够捕获智能体的故障？审计通常受限于记录而非方法本身。因此，CatchBench将审计员的一个问题置于三种信息状态下进行考察：运行前声明的配置（PRE）、不断增长的轨迹前缀（LIVE），以及已完成的轨迹（POST）。以往的基准测试要么固定其中一种状态，要么改变遥测数据；据我们所知，尚无基准在统一的任务-方法接口下对这三种状态同时评分。由于每种状态允许提出不同的问题，七个任务契约各自携带自己的标签和指标，而非采用单一排行榜。其中四项任务为证据性任务，三项为基于Gold的机制诊断任务。该发布版本对72个参赛者进行了评分，参赛者包括规则扫描器、结构化模型，以及来自九个模型家族（GPT、Claude、Gemini、Gemma、Llama、Qwen、DeepSeek、Mistral、Nova）的十一个LLM裁判，涵盖1187个声明配置和1162次记录的运行。大多数竞技场结果并不具有排序区分度：138个注册对比中的56个sep（摘要在此处被截断）

    arXiv:2608.22808v2 Announce Type: replace  Abstract: When can an agent failure be caught? An audit is usually limited by the record rather than by the method. CatchBench therefore puts one auditor's question to three information states: the declared configuration before a run (PRE), a growing prefix of its trace (LIVE), and the finished trace (POST). Prior benchmarks fix one of these states or vary the telemetry; to our knowledge none scores all three under one task-method interface. Each state admits different questions, so seven task contracts carry their own labels and metrics rather than one leaderboard. Four are evidential; three are Gold-derived mechanism diagnostics.   The release scores 72 entrants, from rule scanners and structural models to eleven LLM judges across nine model families (GPT, Claude, Gemini, Gemma, Llama, Qwen, DeepSeek, Mistral, Nova), over 1187 declared configurations and 1162 recorded runs. Most of the arena does not order: 56 of 138 registered contrasts sep
    
[^218]: 稀缺确认性PET测量的合理分配：A4/LEARN中的目标对齐验证

    Spending Scarce Confirmatory PET Measurements: Target-Aligned Validation in A4/LEARN

    [https://arxiv.org/abs/2608.22223](https://arxiv.org/abs/2608.22223)

    本文提出了一种目标对齐的PET验证策略，通过结合目标影响和残差不确定性来优化稀缺确认性测量的分配，避免在影响弱的受试者上浪费资源。

    

    抗淀粉样蛋白疗法和血液生物标志物正在将阿尔茨海默病的诊疗流程转变为两阶段测量工作流：首先使用成本较低的信息进行广泛筛查，然后在能支持最终报告决策的关键环节使用稀缺的确认性淀粉样蛋白测量。淀粉样蛋白正电子发射断层扫描（PET）仍是此类用于评估淀粉样蛋白负担的协议测量手段之一，但PET机位、试验预算和面向支付方的证据包都是有限的。本文提出了一个明确的操作性问题：何时简单的透明PET验证足够，何时拟合残差不确定性评分值得增加复杂度？对于加权协议目标，验证受试者i的一阶价值是目标影响与残差协议不确定性的乘积。通用不确定性采样仅使用第二个因素，可能将PET测量分配给那些难以预测但对科学、临床或商业目标影响较弱的受试者。

    arXiv:2608.22223v1 Announce Type: cross  Abstract: Anti-amyloid therapies and blood-based biomarkers are changing Alzheimer disease workups into a two-stage measurement workflow: screen broadly with cheaper information, then spend scarce confirmatory amyloid measurements where they support the decision that will be reported. Amyloid positron-emission tomography (PET) remains one such protocol measurement for amyloid burden, but PET slots, trial budgets, and payer-facing evidence packages are finite. This paper asks a deliberately operational question: when is simple transparent PET validation enough, and when is a fitted residual-uncertainty score worth the added complexity? For a weighted protocol target, the first-order value of validating subject i is the product of target influence and residual protocol uncertainty. Generic uncertainty sampling uses only the second factor and can spend PET measurements on subjects that are hard to predict but weak for the scientific, clinical, or c
    
[^219]: GigaBrain-WBC-0.5：一种用于与环境交互的鲁棒全身控制的行为世界模型

    GigaBrain-WBC-0.5: A Behavior World Model for Robust Whole-Body Control with Environment Interaction

    [https://arxiv.org/abs/2608.18234](https://arxiv.org/abs/2608.18234)

    本文提出了首个行为世界模型GigaBrain-WBC-0.5，通过因果Transformer联合预测动作、状态和潜在行为命令，使机器人能够建模环境交互，实现鲁棒的全身控制。

    

    arXiv:2608.18234v1 公告类型：交叉 摘要：全身运动跟踪策略将人形机器人转化为一个鲁棒的控制接口：遥操作员——或上游模型——仅提供粗略的运动意图，而低级策略保持机器人平衡和物理可行性。现有的跟踪器仅在平坦地面上提供此接口：在空场景中训练，它们从未学习地形和物体接触如何重塑其动力学，并且它们试图通过不断扩充参考运动语料库来教会策略在任何命令下保持平衡，这在一旦可行行为变得依赖环境时就失效了。我们提出了GigaBrain-WBC-0.5，这是首个用于人形机器人全身控制的行为世界模型（BWM）。与纯粹的反应式跟踪器不同，我们训练了一个因果Transformer来联合预测其下一个动作、下一个状态以及下一个潜在行为命令的分布，因此，行动的网络也建模了环境如何塑造行为。

    arXiv:2608.18234v1 Announce Type: cross  Abstract: Whole-body motion tracking policies turn a humanoid into a robust control interface: the teleoperator---or an upstream model---only supplies a coarse movement intent, while the low-level policy keeps the robot balanced and physically feasible. Existing trackers deliver this interface only on flat ground: trained in empty scenes, they never learn how contact with terrain and objects reshapes their dynamics, and they attempt to teach the policy to balance under any command by continually enlarging the reference-motion corpus, which stops working once feasible behaviors become environment-dependent. We present GigaBrain-WBC-0.5, the first Behavior World Model (BWM) for humanoid whole-body control. Rather than a purely reactive tracker, we train a causal Transformer to jointly predict its next action, next state, and the distribution over its next latent behavior command, so the network that acts also models how the environment shapes what
    
[^220]: 教学与成长：面向通用机器人学习的智能体中心架构

    Teach and Grow: An Agent-Centered Architecture for General Robot Learning

    [https://arxiv.org/abs/2608.17209](https://arxiv.org/abs/2608.17209)

    本文提出了一种名为“教学与成长学习”（TGL）的智能体中心架构，通过将少量演示转化为可复用的技能模块，并动态组合与修正，以降低通用机器人学习中的“再训练税”，提升其在未覆盖场景中的适应能力。

    

    arXiv:2608.17209v1 公告类型：交叉 摘要：端到端的视觉-语言-动作（VLA）和世界动作模型为通用机器人提供了一条优雅的路径，但其可靠性受限于经过验证的物理覆盖范围。当不熟悉的物体、传感器、具身形态或接触超出该覆盖范围且没有经过验证的备用方案时，纠正失败需要新的机器人数据、策略更新和回归测试。这种反复出现的负担被称为“再训练税”。与文本不同，具身数据通常必须通过操作机器来创建。我们提出了教学与成长学习（TGL），一种面向通用机器人学习的智能体中心架构。在其一般形式中，多模态智能体将少量成功演示转化为可复用的技能模块：针对有意义子目标闭环行为。在新场景中，智能体对这些模块进行基础化处理和组合，选择学习或几何工具，观察物理结果，并在执行偏离意图时修正路线。

    arXiv:2608.17209v1 Announce Type: cross  Abstract: End-to-end vision-language-action (VLA) and world-action models offer an elegant route to general-purpose robotics, but their reliability is bounded by validated physical coverage. When an unfamiliar object, sensor, embodiment, or contact falls outside that coverage and no validated fallback exists, correcting the failure requires new robot data, a policy update, and regression testing. This recurring burden is the retraining tax. Unlike text, embodied data must often be created by operating machines. We present Teach-and-Grow Learning (TGL), an agent-centered architecture for general robot learning. In its general form, a multimodal agent turns a few successful demonstrations into reusable Skill Blocks: closed-loop behaviors for meaningful subgoals. In a new scene, the agent grounds and composes these blocks, selects learned or geometric tools, observes the physical outcome, and revises the route when execution departs from intent. A 
    
[^221]: 无参考日志能量预言恢复用于对称强制变分问题的神经近似：符合Riesz重构与档案级选择

    Reference-free logged energy-oracle recovery for neural approximations of symmetric coercive variational problems: conforming Riesz reconstruction and archive-level selection

    [https://arxiv.org/abs/2608.16473](https://arxiv.org/abs/2608.16473)

    本文提出一种无参考的选择规则，通过符合Riesz监测器恢复神经近似的日志能量预言，并揭示档案选择的顺序敏感性，确保在有限检查点下准确选择最优近似。

    

    arXiv:2608.16473v1 公告类型：新公告  摘要：神经偏微分方程训练产生有限检查点档案，但若无精确解，其日志能量误差不可访问，而基于损失的选取不一定能恢复日志能量预言。对于对称强制变分问题的可容许神经近似，我们引入一种基于最小化可计算符合Riesz监测器的无参考选择规则。精确残差能量恒等式和符合投影使监测器成为无条件下界，在嵌套符合细化下单调收敛到每个日志能量误差；在饱和条件下，分层细化产生可计算上界估计，从而形成下上界括号。关键发现是档案选择具有顺序敏感性：未解析的检查点依赖分量可在有限分辨率下逆转预言-非预言排名，因此仅逐检查点恢复是不够的。对于有限档案，我们证明u

    arXiv:2608.16473v1 Announce Type: new  Abstract: Neural PDE training yields a finite checkpoint archive, yet its logged energy errors are inaccessible without the exact solution, while loss-based selection does not necessarily recover the logged energy oracle. For admissible neural approximations of symmetric coercive variational problems, we introduce a reference-free selection rule based on minimizing a computable conforming Riesz monitor. The exact residual-energy identity and conforming projection make the monitor an unconditional lower bound converging monotonically to each logged energy error under nested conforming refinement; under saturation, hierarchical enrichment yields a computable upper estimate and hence a lower-upper bracket. A key finding is that archive selection is order-sensitive: unresolved checkpoint-dependent components can reverse the oracle-non-oracle ranking at finite resolution, so checkpointwise recovery alone is insufficient. For finite archives, we prove u
    
[^222]: 谱基础模型中的预处理不变性归因

    Attributing Preprocessing Invariance in Spectral Foundation Models

    [https://arxiv.org/abs/2608.14227](https://arxiv.org/abs/2608.14227)

    本文指出谱基础模型中的预处理不变性可能源于输入归一化本身，而非模型学习，并主张在评估时应将归一化单独作为基线。

    

    arXiv:2608.14227v1 公告类型：新 摘要：预处理不变性是谱基础模型的一个吸引人的目标：当实验室以不同方式预处理光谱时，冻结模型应保持有用。通常通过在一个预处理流程下训练分类器，并在另一个流程下测试来测量，保留的准确率被视为学习的证据。我们重新审视这一解读，以拉曼基础模型作为案例研究。此类模型在应用任何学习参数之前对输入进行归一化。如果该归一化将两个不同预处理的光谱映射到相同的向量，编码器接收到的输入相同，因此不变性不能归因于学习。对于使用每个光谱自身统计量的归一化，这恰好发生在一个光谱是另一个光谱的正倍数加上常数时。几种标准预处理操作采用这种形式。因此，编码器应仅与归一化本身进行对比，而归一化本身没有...

    arXiv:2608.14227v1 Announce Type: new  Abstract: Preprocessing invariance is an appealing goal for spectral foundation models: a frozen model should remain useful when laboratories preprocess spectra differently. It is usually measured by training a classifier under one preprocessing pipeline and testing it under another, with preserved accuracy read as evidence of learning. We revisit that reading, using a Raman foundation model as a case study. Such models normalize their inputs before any learned parameter is applied. If that normalization maps two differently preprocessed spectra to the same vector, the encoder receives identical inputs, so the invariance cannot be attributed to learning. For a normalization that uses each spectrum's own statistics, this happens exactly when one spectrum is a positive multiple of the other plus a constant. Several standard preprocessing operations take that form. The encoder should therefore be measured against the normalization alone, which has no
    
[^223]: 量子与经典示例的Oracle分离：关于“制造”内容的研究

    A Quantum/Classical Example Oracle Separation for Making Things Up

    [https://arxiv.org/abs/2608.11648](https://arxiv.org/abs/2608.11648)

    本研究首次证明，在Oracle模型下，存在某些分布只能被量子示例学习者高效生成，而经典示例学习者无法做到，从而揭示了量子示例的独特优势。

    

    我们研究了在PAC学习框架中，量子示例相对于经典示例的能力。这里，我们考虑两种学习算法，它们都能访问量子计算，但一种获得量子示例，而另一种获得经典示例。此前尚不清楚是否存在学习任务，其中前者能高效完成而后者不能。我们的主要结果是，相对于一个Oracle，存在一些分布，可以由访问量子示例的量子学习者高效生成，但无法由仅访问经典示例的量子学习者生成，这为肯定回答此问题取得了进展。

    arXiv:2608.11648v1 Announce Type: cross  Abstract: We study the power of quantum examples, as compared to classical examples, in the PAC learning framework. Here, we have two learning algorithms, both with access to quantum computation, but one gets quantum examples, whereas the other gets classical examples. It was previously unknown whether there were learning tasks that can be efficiently performed but not by the latter. Our primary result is to show that relative to an oracle, there are distributions that can be efficiently generated by a quantum learner with access to quantum examples, but not by a quantum learner with access to only classical examples, making progress to answering this question in the affirmative.
    
[^224]: 超越在线策略探索：将外部策略生成的轨迹整合到扩散语言模型的强化学习中

    Beyond On-Policy Exploration: Integrating External Policy Rollouts for Reinforcement Learning in Diffusion Language Models

    [https://arxiv.org/abs/2608.01717](https://arxiv.org/abs/2608.01717)

    提出ERILS方法，通过控制外部轨迹长度并分别处理不同来源的奖励，将更强外部策略生成的高奖励轨迹整合到扩散语言模型的强化学习训练中，有效解决了在线策略成功样本稀缺导致的训练进展受限问题。

    

    针对扩散大语言模型（dLLM）的最新强化学习方法通常依赖于由目标dLLM自身生成的在线策略轨迹。然而，当成功的在线策略轨迹稀缺时，在线训练可能只能获得很少的正向奖励，进展有限。为缓解这一问题，我们探索将由更强的外部策略生成的更高奖励轨迹与目标dLLM的在线策略轨迹一同纳入训练。然而，直接引入这些外部轨迹会带来两个实际挑战：轨迹长度不一致，以及联合处理在线策略和外部轨迹奖励时的不稳定性。为应对这些挑战，我们提出ERILS（External Rollout Integration with Length Control and Source-Specific Processing，带长度控制与来源特定处理的外部轨迹整合方法），该方法控制外部轨迹的长度，并分别处理在线策略轨迹和外部轨迹的奖励。在数独、Countdown等任务上的实验表

    arXiv:2608.01717v2 Announce Type: replace  Abstract: Recent reinforcement learning methods for diffusion large language models (dLLMs) commonly rely on on-policy rollouts generated by the target dLLM itself. When successful on-policy rollouts are scarce, however, on-policy training may receive little positive reward and make only limited progress. To mitigate this problem, we explore incorporating higher-reward rollouts generated by a stronger external policy alongside on-policy rollouts from the target dLLM. However, directly incorporating these external rollouts introduces two practical challenges: differences in rollout length and instability when jointly processing rewards from on-policy and external rollouts. To address these challenges, we propose External Rollout Integration with Length Control and Source-Specific Processing (ERILS), which controls external-rollout length and processes the rewards of on-policy and external rollouts separately. Experiments on Sudoku, Countdown, a
    
[^225]: 当低字符错误率不再足够：视觉语言OCR系统在乌拉圭历史文献上幻觉现象的分析

    When Low CER is Not Enough: An Analysis of Hallucinations in Vision-Language OCR Systems on Historical Uruguayan Documents

    [https://arxiv.org/abs/2607.24077](https://arxiv.org/abs/2607.24077)

    该研究通过分析乌拉圭独裁时期历史文献，揭示视觉语言OCR模型虽然在字符错误率上优于传统方法，但存在标准指标无法检测的幻觉问题（如拼写规范化、虚假内容生成和语义替换），表明仅凭低CER不足以评估其在档案转录中的可靠性。

    

    光学字符识别（OCR）是历史档案数字化的关键组成部分。近年来，视觉语言模型（VLM）已成为传统OCR系统的有力替代方案，在标准基准测试中取得了最先进的性能。然而，它们在档案转录任务中的适用性仍缺乏充分理解。在这项工作中，我们在Berrutti数据集上对传统OCR系统和基于VLM的方法进行了基准测试，该数据集是一组源自缩微胶片扫描的乌拉圭独裁时期文献，极具挑战性。虽然VLM在字符错误率（CER）和词错误率（WER）方面始终优于传统方法，但我们表明这些改进背后隐藏着更复杂的情况。通过详细的定性分析，我们揭示了标准指标无法察觉的系统性失败模式，包括拼写规范化、虚假内容生成和语义替换。

    arXiv:2607.24077v2 Announce Type: replace-cross  Abstract: Optical Character Recognition (OCR) is a key component in the digitization of historical archives. Recently, Vision-Language Models (VLMs) have emerged as strong alternatives to traditional OCR systems, achieving state-of-the-art performance on standard benchmarks. However, their suitability for archival transcription remains insufficiently understood. In this work, we benchmark traditional OCR systems and VLM-based approaches on the Berrutti dataset, a challenging collection of Uruguayan dictatorship-era documents derived from microfilm scans. While VLMs consistently outperform traditional methods in terms of Character Error Rate (CER) and Word Error Rate (WER), we show that these improvements hide a more complex picture. Through a detailed qualitative analysis, we uncover systematic failure modes that are invisible to standard metrics, including orthographic normalization, spurious content generation, and semantic substitutio
    
[^226]: Molt：一个面向智能体强化学习的可扩展 PyTorch 原生训练框架

    Molt: A Scalable PyTorch-Native Training Framework for Agentic Reinforcement Learning

    [https://arxiv.org/abs/2607.21653](https://arxiv.org/abs/2607.21653)

    提出了 Molt 框架，通过可组合模型并行、统一智能体接口、全异步 rollout 与优化以及分布式经验存储，实现了万亿参数规模下的智能体强化学习训练，且无需修改现有智能体的执行逻辑。

    

    智能体强化学习需要一种研究者能够在不牺牲模型规模或智能体执行控制权的前提下进行修改的基础设施。我们提出了 Molt，一个轻量级的 PyTorch 原生框架，将万亿参数规模的训练与标准智能体接口相结合。Molt 整合了四项能力：基于可组合模型并行的紧凑训练实现；统一的 OpenAI 和 Anthropic 接口，支持上下文压缩后的自动轨迹分段；完全异步的 rollout 与优化；以及面向长多模态轨迹的分布式经验存储。现有智能体可以保留其执行与上下文管理逻辑，同时由共享捕获层记录生成的 token 和行为概率。Rollout 工作进程将大体积的经验数据放入 Ray 的对象存储中，训练器进程通过引用检索分配给它的经验，从而避免对完整 rollout 进行集中式收集。

    arXiv:2607.21653v2 Announce Type: replace-cross  Abstract: Agentic reinforcement learning requires infrastructure that researchers can modify without sacrificing model scale or control over agent execution. We present Molt, a lightweight PyTorch-native framework that combines trillion-parameter training with standard agent interfaces. Molt integrates four capabilities: a compact training implementation built on composable model parallelism; unified OpenAI and Anthropic interfaces with automatic trajectory segmentation after context compaction; fully asynchronous rollout and optimization; and distributed experience storage for long, multimodal trajectories. Existing agents retain their execution and context-management logic while a shared capture layer records generated tokens and behavior probabilities. Rollout workers place heavy experience payloads in Ray's object store, and trainer ranks retrieve their assigned experiences by reference, avoiding a centralized gather of the full roll
    
[^227]: PhysCoRe：用于材料感知可变形动力学的物理校正残差世界模型

    PhysCoRe: Physics-Corrected Residual World Models for Material-Aware Deformable Dynamics

    [https://arxiv.org/abs/2607.20653](https://arxiv.org/abs/2607.20653)

    PhysCoRe通过将可微分物质点法（MPM）模拟器与视觉材料推断模块（MfM）和动力学残差校正模块（RfD）相结合，实现了对可变形物体在机器人操作下既符合物理规律又能泛化材料属性的动力学预测。

    

    预测可变形物体在机器人操作下如何演化是一个长期存在的挑战。现有方法通常依赖针对单个物体的优化来拟合材料参数，这不仅速度慢且无法泛化，而端到端学习的替代方案外推能力差，且经常违反基本的物理结构。我们提出了PhysCoRe，一个物理校正残差世界模型，它将可微分的物质点法（MPM）模拟器与两个前馈神经网络相结合。其中，材料精炼模块Material from Motion（MfM）从视觉观测中推断每个粒子的弹性属性，使模拟器建立在物体特定的物理特性之上；残差校正模块Residual from Dynamics（RfD）学习模拟结果与真实之间的差异，并预测对模拟器内部动力学的校正，从而吸收解析模型无法捕捉的系统性偏差。这种设计还支持在线材料识别。

    arXiv:2607.20653v2 Announce Type: replace-cross  Abstract: Predicting how deformable objects evolve under robotic manipulation is a longstanding challenge. Existing approaches typically rely on per-object optimization to fit material parameters, which can be slow and cannot generalize, while end-to-end learned alternatives extrapolate poorly and often violate basic physical structure. We present PhysCoRe, a physics-corrected residual world model that couples a differentiable Material Point Method (MPM) simulator with two feed-forward neural networks. A material refinement module, Material from Motion (MfM), infers per-particle elasticity from visual observations, grounding the simulator in object-specific physics. A residual correction module, Residual from Dynamics (RfD), learns the discrepancy and predicts corrections to the simulator's internal dynamics, absorbing systematic biases that the analytical model cannot capture. This design also supports online material identification on 
    
[^228]: 面向连续状态-动作空间的网络化多智能体强化学习的可扩展策略优化

    Scalable Policy Optimization for Networked Multi-Agent Reinforcement Learning with Continuous State-Action Spaces

    [https://arxiv.org/abs/2607.18554](https://arxiv.org/abs/2607.18554)

    该论文针对连续状态-动作空间的网络化多智能体强化学习，对CDCPG算法进行了理论分析，通过局部随机傅里叶特征和最小二乘时序差分评论家，推导出空间残差与有限特征残差分离的动作值表示，并借助全局综合转移近似界和投影贝尔曼论证在无需逆条件乘子的情况下控制群体预测误差，从而实现可扩展的策略优化。

    

    arXiv:2607.18554v2 公告类型：replace-cross 摘要：为连续网络化系统学习局部策略需要考虑决策在超出每个智能体观测邻域范围之外的影响。空间衰减限制了这些影响，但有限的评论家（critic）还必须在策略优化过程中控制表示误差和估计误差。我们分析了采用局部随机傅里叶特征和最小二乘时序差分评论家的连续分布式耦合策略梯度（CDCPG）算法。对于保留了局部动力学所需的边界输入的特征，我们推导出一种具有独立空间残差和有限特征残差的动作值表示。一个全局综合的转移近似界和一个投影贝尔曼论证在无需逆条件乘子的情况下控制群体预测误差。随后，我们量化了评论家估计对特征激励和维度的依赖关系，并构建了针对温度（原文在此处截断）

    arXiv:2607.18554v2 Announce Type: replace-cross  Abstract: Learning local policies for continuous networked systems requires accounting for the effects of decisions beyond each agent's observation neighborhood. Spatial decay limits these effects, but a finite critic must also control representation and estimation errors throughout policy optimization. We analyze the Continuous Distributed Coupled Policy Gradient (CDCPG) algorithm using local random Fourier features and least-squares temporal-difference critics. For features that retain the boundary inputs required by the local dynamics, we derive an action-value representation with separate spatial and finite-feature residuals. A global integrated transition-approximation bound and a projected Bellman argument control population prediction error without an inverse-conditioning multiplier. We then quantify the dependence of critic estimation on feature excitation and dimension, and construct simultaneous lower confidence bounds for temp
    
[^229]: 从零构建神经网络：实现、评估与优化

    Building a Neural Network from Scratch: Implementation, Evaluation, and Optimization

    [https://arxiv.org/abs/2607.16682](https://arxiv.org/abs/2607.16682)

    本文从零实现了一个不依赖自动微分和现成深度学习模块的完整神经网络框架，涵盖多层架构、激活函数、正则化和先进优化器，并通过多分类任务验证了其正确性、数值稳定性与泛化能力。

    

    高级深度学习库的广泛采用虽然在加速模型开发的同时，却日益抽象掉了神经网络的内部机制，造成了实际使用与基本理解之间的鸿沟。为解决这一问题，本文提出了一个完全从零开始、不依赖自动微分或预构建深度学习模块的独立神经网络框架。该实现涵盖了所有核心组件，包括多层网络架构、多种激活函数、正则化技术以及最先进的优化器。该框架不仅作为一个教学工具，揭示了前向/反向传播、梯度动态和优化景观的奥秘，还在多分类任务中展现出稳健的性能，成功验证了其正确性、数值稳定性和泛化能力。

    arXiv:2607.16682v2 Announce Type: replace-cross  Abstract: The widespread adoption of high-level deep learning libraries, while accelerating model development, has increasingly abstracted away the internal mechanics of neural networks, creating a gap between practical usage and fundamental understanding. To address this, the paper presents a self-contained neural network framework implemented entirely from scratch without relying on automatic differentiation or pre-built deep learning modules. The implementation encompasses all essential components, including multi-layer architectures, diverse activation functions, regularization techniques, and state-of-the-art optimizers. Beyond serving as a pedagogical instrument that demystifies forward/backward propagation, gradient dynamics, and optimization landscapes, the framework demonstrates robust performance when applied to a multi-class classification task, successfully validating its correctness, numerical stability, and generalization a
    
[^230]: 基于监督式特征二值化的临床数据可解释与校准分类

    Interpretable and Calibrated Classification of Clinical Data Using Supervised Feature Binarization

    [https://arxiv.org/abs/2607.15394](https://arxiv.org/abs/2607.15394)

    本文提出一种结合监督式卡方引导二值化与伯努利朴素贝叶斯的临床分类框架，在保持完全可解释性和良好校准的同时，对连续医疗数据实现高精度分类（三个基准数据集AUC为0.800至0.984）。

    

    黑盒模型因其预测难以解释和复现，限制了人工智能在医学领域的应用。我们提出了一个具有统计学基础的可解释、基于规则的临床分类框架，采用伯努利朴素贝叶斯（BNB）模型。监督式卡方引导二值化通过在训练折内选择与临床结局关联最大化的阈值，将连续变量转换为二元指标，使BNB能够在不牺牲透明度的情况下处理连续医疗数据。在三个基准数据集——皮马印第安人糖尿病、威斯康星乳腺癌和心力衰竭预测——上，该框架的ROC曲线下面积（AUC）分别达到0.800、0.984和0.919。概率可靠性通过无泄漏的交叉验证校准分析进行评估，报告了Brier分数和校准截距。

    arXiv:2607.15394v2 Announce Type: replace  Abstract: Black-box models limit the adoption of artificial intelligence in medicine because their predictions are difficult to interpret and reproduce. We present a statistically grounded framework for interpretable, rule-based clinical classification using the Bernoulli Na\"ive Bayes (BNB) model. Supervised chi-square-guided binarization converts continuous variables into binary indicators by selecting thresholds that maximize association with the clinical outcome within the training folds, which allows BNB to operate on continuous medical data without sacrificing transparency. On three benchmark datasets, Pima Indians Diabetes, Wisconsin Breast Cancer, and Heart Failure Prediction, the framework reached areas under the receiver operating characteristic curve of 0.800, 0.984, and 0.919, respectively. Probabilistic reliability was assessed with a leakage-safe cross-validated calibration analysis reporting Brier score and calibration intercept
    
[^231]: 面向文本到图像生成中代表性多样性的多轴 Max@K 强化学习

    Multi-Axis Max@K Reinforcement Learning for Representative Diversity in Text-to-Image Generation

    [https://arxiv.org/abs/2607.14962](https://arxiv.org/abs/2607.14962)

    该论文提出了多轴 max@K 这一基于分组的强化学习目标，通过仅奖励提升各模式组内最大值的样本的信用分配机制，让不同样本贡献于不同语义模式，从而提升文本到图像生成模型对预定义目标模式的覆盖及代表性多样性。

    

    文本到图像（T2I）模型能够合成逼真的、与提示词对齐的图像，然而为同一提示词生成的样本往往只覆盖了视觉上各不相同模式中的一小部分。这限制了多样性，并且对于以人物为中心的提示词，可能会反映或放大人口统计偏差。我们将这一问题形式化为目标模式覆盖，即对预定义的一组语义指定模式的覆盖程度，并提出多轴 max@K，这是一种基于分组的强化学习目标，用于在基于扩散模型的文本到图像生成模型中提升这一覆盖能力。给定一组样本以及每个目标模式对应的一个分数，多轴 max@K 首先对每个模式在各样本间取最大分数，然后对这些每模式的最大值求和。由此产生的信用分配机制仅当某个样本提升了该模式的组内最大值时，才在该模式上赋予该样本正权重，因此不同的样本可以对不同的模式做出贡献。我们在合成混合分布和 SD3.5-（摘要原文在此截断）

    arXiv:2607.14962v2 Announce Type: replace-cross  Abstract: Text-to-image (T2I) models can synthesize realistic, prompt-aligned images, yet samples generated for the same prompt often cover only a small subset of visually distinct modes. This limits diversity and, for person-centric prompts, can reflect or amplify demographic skew. We formalize this problem as target-mode coverage, the coverage of a predefined set of semantically specified modes, and propose multi-axis max@K, a group-based reinforcement learning objective for improving it in diffusion-based T2I models. Given a group of samples and one score per target mode, multi-axis max@K first takes the maximum score across samples for each mode and then sums these per-mode maxima. The resulting credit assignment gives a sample positive weight on a mode only when it raises that mode's group maximum, so different samples can contribute to different modes. We validate the credit-assignment mechanism on a synthetic mixture and on SD3.5-
    
[^232]: 数据不平衡何时有益：通过捷径饱和实现鲁棒泛化

    When Data Imbalance Helps: Robust Generalization Through Shortcut Saturation

    [https://arxiv.org/abs/2607.10116](https://arxiv.org/abs/2607.10116)

    本文发现一个反直觉现象：在容量足够的模型中，数据不平衡反而通过“捷径饱和”机制促进鲁棒泛化，但在容量较小的模型中不平衡会使模型陷入对捷径特征的依赖。

    

    我们研究虚假相关性下的鲁棒泛化问题：即捷径特征在训练中与真实标签相关，但在对抗性保留数据集上与真实标签反相关的任务。通过改变虚假比率 r（训练样本中捷径=真实标签的比例）和模型容量，我们发现了一个反直觉的结果：数据不平衡能够促进具备足够容量的模型的泛化能力。在一个合成任务中，真实标签是整数序列之和的奇偶性，而捷径特征是最大值元素的奇偶性，一个2层2头的transformer在 r=0.50 时的随机种子中实现了泛化（达到100%对抗准确率）的比例为0%，而在 r=0.90 时这一比例达到77%。该效应在1层模型中不存在，在那种情况下，数据不平衡反而使模型陷入对捷径的依赖。通过机制分析——包括梯度冲突动态、电路演化以及QK/OV电路消融实验——我们刻画了一条从依赖捷径到实现鲁棒泛化的机制路径。

    arXiv:2607.10116v2 Announce Type: replace-cross  Abstract: We study robust generalization under spurious correlations: tasks where a shortcut feature is correlated with the true label in training but anti-correlated in an adversarial held-out split. Varying the spurious ratio $r$ (the fraction of training examples where shortcut = true label) and model capacity, we find a counterintuitive result: data imbalance promotes generalization in sufficiently capable models. On a synthetic task where the true label is sum parity of an integer sequence and the shortcut is the parity of the maximum-valued element, a 2-layer, 2-head transformer generalized (reached $100\%$ adversarial accuracy) in 0% of seeds at $r{=}0.50$ but 77% of seeds at $r{=}0.90$. The effect is absent in 1-layer models, where imbalance instead traps the model on the shortcut. Through mechanistic analysis -- gradient conflict dynamics, circuit evolution, and QK/OV circuit ablations -- we characterize a mechanistic pathway co
    
[^233]: 忠实而非纠正：模型能力决定多跳智能体接力中的消息格式效应

    Faithful, Not Corrective: Model Capability Governs Message-Format Effects in Multi-Hop Agent Relays

    [https://arxiv.org/abs/2607.09678](https://arxiv.org/abs/2607.09678)

    论文通过六跳、五种格式的多智能体消息接力实验发现，消息格式效应由接力模型自身能力决定：强接力器对所有格式几乎无损传递，且接力行为表现为忠实复制而非纠错。

    

    当LLM智能体相互传递信息时，消息格式是否重要？两类文献观点相左：格式优化研究指出，结构化消息能在不损害准确率的前提下降低成本；而格式限制研究则发现，强加结构会降低生成质量。然而，两类研究均未测量消息跨多跳传递时的表现——在这种场景下，起主导作用的是复制保真度，而非一次性生成质量。我们构建了一个受控接力测试平台：包含十二个程序化原子事实的简报以五种格式（自由自然语言、精确指令自然语言、JSON、三元组、键值对）逐跳重新编码，跨越六个跳次，由固定的强评分器对照程序化真值进行评分，并涵盖两个接力能力层级、一个认知负荷条件以及配对分叉错误注入。我们发现：(i) 强接力器对所有格式几乎无损（第6跳问答召回率 ≥ 0.973），残余损失集中于……（原文摘要在此截断）

    arXiv:2607.09678v2 Announce Type: replace  Abstract: When LLM agents hand information to one another, does the message format matter? Two literatures disagree: format-optimization work reports that structured messages cut cost without hurting accuracy, while format-restriction studies find that imposing structure degrades generation. Neither line has measured what happens when messages traverse multiple hops, where copy fidelity, rather than one-shot generation quality, dominates. We introduce a controlled relay testbed in which briefs of twelve programmatic atomic facts are re-encoded hop by hop in five formats (free natural language, precision-instructed NL, JSON, triples, key-value) over six hops, scored against programmatic ground truth by a fixed strong grader, across two relay-capability tiers, a cognitive-load condition, and a paired-fork error injection. We find that (i) a strong relay is nearly lossless for every format (hop-6 QA recall $\geq 0.973$), with residual loss concen
    
[^234]: 基于提示驱动的探索

    Prompt-Driven Exploration

    [https://arxiv.org/abs/2607.08837](https://arxiv.org/abs/2607.08837)

    本文提出一种利用视觉-语言模型从强化学习展开视频中自动诊断并重写提示的方法，以实现对弱策略的全局探索，而无需依赖稀疏奖励。

    

    摘要：arXiv:2607.08837v2 公告类型：替换交叉 摘要：探索对于强化学习（RL）至关重要，因为策略无法通过反复采样其已偏好的行为来改进。标准方法在动作空间中注入随机性，但这种抖动只会产生接近原始轨迹的展开。要摆脱弱策略，通常需要动作噪声无法产生的全局扰动。大型语言模型（LLM）和视觉-语言-动作（VLA）模型提供了一条途径：它们将策略条件化于自然语言提示，由于展开遵循该提示，修改提示会引发全局变化。挑战在于找到能引发有用全局变化的提示。当弱策略很少成功时，奖励过于稀疏而无法用于选择。我们的想法是从展开本身中提炼提示：一个视觉-语言模型（VLM）对展开视频进行推理，诊断策略如何响应，并重写提示以在下次引发更好的行为。此过程类似于...

    arXiv:2607.08837v2 Announce Type: replace-cross  Abstract: Exploration is essential to RL since a policy cannot improve by repeatedly sampling the behaviors it already prefers. Standard methods inject stochasticity in the action space, but such jitter only yields rollouts close to the original. Escaping a weak policy often requires global perturbations that action noise cannot produce. Large language models (LLMs) and vision-language-action (VLA) models offer a pathway: they condition the policy on a natural language prompt, and since the rollout follows from it, modifying the prompt induces global changes. The challenge is finding prompts that induce useful global changes. With a weak policy that rarely succeeds, reward is too sparse to select on. Our idea is to refine prompts from the rollouts themselves: a vision-language model (VLM) reasons over the rollout video, diagnoses how the policy responded, and rewrites the prompt to elicit better behavior next time. This procedure resembl
    
[^235]: 一种可迁移的学习型时间先验：用于传播链重构及真实疫情标签中与决策相关的不确定性

    A Transferable Learned Temporal Prior for Transmission Reconstruction and Decision-Relevant Uncertainty in Real Outbreak Labels

    [https://arxiv.org/abs/2606.30842](https://arxiv.org/abs/2606.30842)

    该研究从十一种其他疾病中学习并提前锁定了一个时间先验模型，无需重新拟合即可迁移应用于安第斯病毒的传播链重构任务并显著优于基线，同时评估了已发表传播链作为真实标签的可靠性及其对决策的影响。

    

    arXiv:2606.30842v2 公告类型：替换 摘要：重构“谁感染了谁”通常依赖于病例的时间信息和已发表的传播链关联。然而，仍有两个重要问题悬而未决：从其他疾病中学到的时间模式能否迁移到新的疫情中，以及用作真实标准的传播链关联本身有多可靠？我们使用逻辑回归从十一个疾病组中学习了一个时间先验。该模型在接触任何目标疫情数据之前即被锁定，随后在29个安第斯病毒（ANDV）亲代排序任务上进行了不重新拟合的测试。该锁定先验取得了0.571的平均倒数排名（MRR），而最强的公平来源训练时间基线仅为0.274（置换检验p<0.001）。其Top-1准确率为37.9%，而基线为13.8%。除非7-8个有利的任务结果被逆转，否则MRR的优势依然显著。我们另外单独检验了已发表传播链关联的可靠性。在来自……的75个有流行病学关联的宿主间传播对中……（摘要原文在此处截断）

    arXiv:2606.30842v2 Announce Type: replace  Abstract: Reconstructing who infected whom often relies on case timing and published transmission links. However, two important questions remain: can a temporal pattern learned from other diseases transfer to a new outbreak, and how reliable are the transmission links used as ground truth? We learned a temporal prior from eleven disease groups using logistic regression. The model was locked before any target-outbreak data were accessed and was then tested without refitting on 29 Andes virus (ANDV) parent-ranking tasks. The locked prior achieved a mean reciprocal rank (MRR) of 0.571, compared with 0.274 for the strongest fair source-trained temporal baseline (permutation p<0.001). Its Top-1 accuracy was 37.9%, compared with 13.8%. The MRR advantage remained significant unless 7-8 favorable task outcomes were reversed. We separately examined the reliability of published transmission links. Among 75 epidemiologically linked inter-host pairs from 
    
[^236]: 通过高效的动作间价值共享加速Q学习

    Accelerating Q-learning through Efficient Value-Sharing across Actions

    [https://arxiv.org/abs/2606.29806](https://arxiv.org/abs/2606.29806)

    本文提出一种无参数的均值扩展层，通过在状态内的动作之间共享价值并学习低范数表示，加速强化学习中动作价值的学习过程。

    

    动作价值是Q学习等许多控制算法的基础。因此，高效的动作价值学习是强化学习（RL）的核心。然而，学习动作价值可能很慢，需要多次更新才能将价值从初始值（通常接近零）移动到真实值（可能远离零）。此外，动作价值学习算法通常独立地更新每个状态-动作对，而不学习某个状态内所有动作共有的价值。在本文中，我们通过引入均值扩展层来解决这些低效问题。该层通过在状态内的动作之间共享价值，并将问题从直接学习可能很大的动作价值转变为学习其较低范数的表示，从而加速动作价值学习。在深度强化学习中，该层可以作为Q网络架构的无参数附加组件应用，而无需更改底层算法。

    arXiv:2606.29806v3 Announce Type: replace-cross  Abstract: Action values are foundational to many control algorithms such as Q-learning. Therefore, efficient action-value learning is central to reinforcement learning (RL). However, learning them can be slow, requiring many updates to move values from their initialization, typically near zero, to their true values, which may be far from zero. Moreover, action-value learning algorithms typically update each state-action pair independently, without learning a value that is common to all actions within a state. In this paper, we address these inefficiencies by introducing the mean-expansion layer, which accelerates action-value learning by sharing values across actions within a state and by changing the problem from directly learning potentially large action-values to learning a lower-norm representation of them. In deep RL, this layer can be applied as a parameter-free addition to Q-network architectures without altering the underlying al
    
[^237]: 当检索指标产生误导时：测量长程工具使用智能体中的策略信号

    When Retrieval Metrics Mislead: Measuring Policy Signal in Long-Horizon Tool-Use Agents

    [https://arxiv.org/abs/2606.23937](https://arxiv.org/abs/2606.23937)

    该研究发现精确匹配检索召回率是一个具有误导性的代理指标——即使正确的治理规则仅在 7% 的情况下被排名第一检索到，检索到的断言仍能让分类器取得与使用黄金规则几乎相同的性能。

    

    精确匹配检索召回率常被用作衡量检索器是否为下游决策模型提供有用策略上下文的代理指标。我们在 τ-bench 中使用 Qwen2.5-3B/7B 分类器，针对动作前策略分类任务测试了这一代理指标。在黄金策略条件下，经过调优的紧凑结构化状态在 3B 模型上比原始轨迹的 macro-F1 提高了 0.20，在共享超参数下 7B 模型也呈现相同的排序关系。随后，我们将基准指定的治理规则替换为从决策时上下文中检索到的排名第一的基准断言。尽管精确的治理规则仅在 7% 的航空领域状态中被检索到排名第一，但主要的 3B 分类器使用检索到的断言获得了 0.58 的 macro-F1，而使用黄金规则为 0.60（Δ=-0.02，任务簇 95% 置信区间 [-0.23,+0.21]）；作为对照，随机非黄金断言和无断言条件的得分分别为 0.32 和 0.21。我们没有检测到……（摘要在此处截断）

    arXiv:2606.23937v2 Announce Type: replace-cross  Abstract: Exact-match retrieval recall is often used as a proxy for whether a retriever supplies useful policy context to a downstream decision model. We test this proxy for pre-action policy classification in $\tau$-bench using Qwen2.5-3B/7B classifiers. Under gold-policy conditioning, a compact structured state improves macro-F1 over raw trajectories by $0.20$ after tuning at 3B, with the same ordering at 7B under shared hyperparameters. We then replace the benchmark-designated governing rule with the top-ranked benchmark assertion retrieved from decision-time context. Although the exact governing rule is retrieved at rank 1 for only $7\%$ of airline states, the primary 3B classifier obtains macro-F1 $0.58$ with retrieved assertions versus $0.60$ with the gold rule ($\Delta=-0.02$, task-cluster 95\% CI $[-0.23,+0.21]$); random non-gold and no-assertion controls score $0.32$ and $0.21$. We do not detect a macro-F1 difference between ret
    
[^238]: 大语言模型归因指标具有可迁移性吗？跨数据集与评估构念的检索增强生成评估审计

    Do LLM Attribution Metrics Transfer? Auditing Retrieval-Augmented Generation Evaluation Across Datasets and Constructs

    [https://arxiv.org/abs/2606.23915](https://arxiv.org/abs/2606.23915)

    该研究系统审计了八种检索增强生成归因自动指标，发现在生成答案归因构念下没有任何指标能在所有数据集上保持与最佳指标一致的性能，不同数据集上的指标排名甚至完全反转，因此实践中将这些指标视为可互换是不成立的。

    

    实践中通常将用于大语言模型检索增强生成中归因评估的自动指标视为可以互相替换的。我们审计了八种自动评分器——词汇、嵌入和 BERTScore 基线，以及经过蕴含/接地训练的模型（干净版和 FEVER 版 NLI、检查工具 MiniCheck）——横跨三种评估构念（来源与主题相关性、生成答案归因、事实核查蕴含），探究是否存在任何评分器能够“迁移”：即在多数据集构念的每个数据集上都保持在最佳被审计评分器的 95% 置信区间之内。在拥有最多多数据集人工标注覆盖的构念——生成答案归因（AttributionBench 的四个源数据集，n = 1,610，以及独立的 HAGRID，n = 2,150）中，没有任何被审计的自动评分器能做到这一点：各数据集上的指标排名会发生反转（在 AttributedQA 与 LFQA 对比中 Kendall tau = -0.64，p = 0.031），且在短声明上表现最佳的现成 NLI 评分器（原文此处截断）

    arXiv:2606.23915v2 Announce Type: replace  Abstract: Practice often treats automatic metrics for attribution in LLM retrieval-augmented generation as interchangeable. We audit eight automatic scorers -- lexical, embedding, and BERTScore baselines alongside entailment/grounding-trained models (clean and FEVER NLI, the checker MiniCheck) -- across three evaluation constructs (provenance/topicality, generated-answer attribution, and fact-check entailment), asking whether any scorer transfers: stays within the 95% confidence interval of the best audited scorer on every dataset of a multi-dataset construct. In the construct with the most multi-dataset human-labeled coverage -- generated-answer attribution (AttributionBench's four source datasets, n = 1,610, with independent HAGRID, n = 2,150) -- none of the audited automatic scorers does: the per-dataset metric rankings invert (Kendall tau = -0.64, p = 0.031 on AttributedQA vs. LFQA), and an off-the-shelf NLI scorer that is best on short-cl
    
[^239]: 探索KV缓存淘汰的逐层设计空间

    Exploring a Layer-Wise Design Space for KV Cache Eviction

    [https://arxiv.org/abs/2606.15157](https://arxiv.org/abs/2606.15157)

    该论文提出在Transformer各层间组合不同的KV缓存淘汰方法构成异构路由，发现在相同缓存预算下，这种逐层的异构策略在LongBench大多数任务上优于全模型统一的同构淘汰策略。

    

    KV缓存淘汰方法通常在整个模型中使用单一的保留规则族，使得淘汰方法的选择成为模型级的设计决策。然而，Transformer的各层在注意力行为、表示方式以及对压缩的敏感性上存在显著差异，这表明统一规则可能忽略了有用的逐层结构。由此引出一个基本问题：淘汰方法本身是否应该在不同层间有所变化？我们通过在Transformer各层组合现有的淘汰方法来研究这一问题，并系统地探索由此形成的逐层设计空间。利用简单的离线性能画像，我们构建固定的路由方案，并研究其质量如何随方法放置位置和缓存预算而变化。在LongBench基准上，在相同缓存预算下，异构路由在大多数任务上的表现优于同构策略。即使在方法数量保持不变的情况下，基于画像引导的放置方式也排名第二。

    arXiv:2606.15157v2 Announce Type: replace-cross  Abstract: KV cache eviction methods typically use a single retention-rule family throughout a model, making eviction-method identity a model-level design choice. Yet Transformer layers differ substantially in their attention behavior, representations, and sensitivity to compression, suggesting that a uniform rule may overlook useful layer-wise structure. This raises a basic question: should eviction methods themselves vary across layers? We investigate this question by composing existing eviction methods across Transformer layers and systematically exploring the resulting layer-wise design space. Using simple offline profiles, we construct fixed routes and study how their quality varies with method placement and cache budget. On LongBench, heterogeneous routing improves performance on a majority of tasks over homogeneous policies at the same cache budget. Even when method counts are held fixed, the profile-guided placement ranks second a
    
[^240]: 选择条件下的测量：面向语言模型的诱饵校准失效审计

    Measurement Under Selection: Decoy-Calibrated Failure Audits for Language Models

    [https://arxiv.org/abs/2606.09046](https://arxiv.org/abs/2606.09046)

    本文提出Janus框架，通过随机打乱属性标签生成的“诱饵”来校准阈值，只有超出偶然波动水平并在保留数据上稳健成立的错误模式才被报告，从而避免语言模型失效审计中出现虚假的模式发现。

    

    知道语言模型失败的频率并不能解释其错误集中在哪里。当审计人员检查许多可能的解释时，观察到的最强模式可能只是偶然产生的。我们提出了Janus，一种在报告之前对拟议错误模式进行检验的程序。Janus从被评估示例的一组固定的“是/否”属性列表开始，例如输入是否很长。对于每个属性，它比较模型在具有该属性的示例与没有该属性的示例上的错误率。为了了解偶然情况下可能产生多大的差异，它在打乱示例间的“是/否”标签（但不改变各组规模）之后重复这一计算。这些打乱后的属性被称为“诱饵”。只有当某个模式的错误差异大小达到由诱饵设定的阈值时，该模式才会被报告。在单独的保留示例上，同一组仍必须具有更高的错误率，且差异必须达到一个最低标准……

    arXiv:2606.09046v2 Announce Type: replace-cross  Abstract: Knowing how often a language model fails does not explain where its errors concentrate. When auditors examine many explanations, the strongest observed pattern may arise by chance. We introduce Janus, a procedure for checking proposed error patterns before reporting them. Janus starts with a fixed list of yes/no properties of the examples being evaluated, such as whether the input is long. For each property, it compares the model's error rates on examples with that property and those without it. To see how large a difference can arise by chance, it repeats this calculation after shuffling the yes/no labels across examples without changing the group sizes. These shuffled properties are called decoys. A pattern is reported only if the size of its error difference meets a threshold set using decoys. On separate held-out examples, the same group must still have the higher error rate and the difference must meet a minimum, which was
    
[^241]: SoK：合成表格数据上的重建攻击（来自赢得NIST CRC的见解）

    SoK: Reconstruction Attacks on Synthetic Tabular Data (Insights from Winning the NIST CRC)

    [https://arxiv.org/abs/2606.08372](https://arxiv.org/abs/2606.08372)

    本文系统化了针对合成表格数据的重建攻击，提出了按结构组织的攻击分类体系，对14种攻击与13种合成数据生成方法在5个基准数据集上进行了大规模实证评估，提出了包括最强攻击CoBP-RA在内的新攻击，并提供了解读攻击成功含义的记忆测试方法论。

    

    合成数据日益被推广为发布敏感表格记录的隐私保护替代方案，然而其核心的对抗性威胁——重建攻击（即从合成的发布数据和少量已知准标识符中恢复个体的隐藏属性值）——此前只在零散且难以相互比较的环境中被研究。我们对去标识化与合成表格数据上的重建（等同于属性推断）攻击进行了系统化。我们的贡献包括：一个按攻击所利用的结构来组织各类攻击的分类体系；一项广泛且受控的实证评估，将十四种攻击与十三种合成数据生成（SDG）方法在五个基准数据集上进行对抗比较；以及一组填补该分类体系空白的新攻击，其中之一（CoBP-RA）是我们所测得的最强攻击。我们还引入了一种用于解读攻击成功含义的方法论：一个能够区分……的记忆测试

    arXiv:2606.08372v2 Announce Type: replace-cross  Abstract: Synthetic data is increasingly promoted as a privacy-preserving substitute for releasing sensitive tabular records, yet its central adversarial threat (reconstruction, the recovery of an individual's hidden attribute values from a synthetic release and a handful of known quasi-identifiers) has been studied only in scattered, hard-to-compare settings. We systematize reconstruction (equivalently, attribute inference) attacks on de-identified and synthetic tabular data. We contribute a taxonomy that organizes attacks by the structure they exploit; a broad, controlled empirical evaluation, pitting fourteen attacks against thirteen synthetic data generation (SDG) methods across five benchmark datasets; and a set of new attacks that fill gaps in the taxonomy, one of which (CoBP-RA) is the strongest attack we measure. We also introduce a methodology for interpreting what attack success means: a memorization test that distinguishes rec
    
[^242]: EssentialGIN：一种基于图同构神经网络预测基因必需性的新方法

    EssentialGIN: a new approach for gene essentiality prediction based on graph isomorphism neural networks

    [https://arxiv.org/abs/2606.07700](https://arxiv.org/abs/2606.07700)

    本研究提出EssentialGIN方法，通过改进图同构神经网络来保留PPI网络的拓扑特征，并整合基因表达、直系同源和亚细胞定位等生物信息，从而实现对必需基因的准确预测。

    

    背景：预测必需基因（蛋白质）是一个基础且具有挑战性的问题，同时在湿实验室实验中又非常昂贵且耗时。仅基于计算方法（用于筛选湿实验室候选基因）并使用中心性度量来预测必需基因并不准确，会导致大量假阳性；因此，近期研究采用了更复杂的模型（如深度学习），并整合生物信息来识别必需基因。方法：在这项工作中，我们专注于图同构网络，将蛋白质作为节点嵌入蛋白质-蛋白质相互作用（PPI）网络中，以保留PPI网络的拓扑特征，同时整合基因表达数据、基因直系同源信息和基因亚细胞定位信息等生物数据，并构建了一种用于预测必需基因的深度架构。本工作对该图同构网络架构进行了改进。

    arXiv:2606.07700v2 Announce Type: replace-cross  Abstract: Background: Prediction of essential genes (proteins), is a basic and challenging problem but at the same time very costly and time-consuming in wet-lab experiments. Predicting essential genes, only based on computational methods (to introduce wet-lab candidates) using centrality measures are not accurate and result in large number of false positives; therefore, more complex models such as deep learning and also integration of biological information are used in recent research to identify essential genes.   Methods: In this work we focus on graph isomorphism networks, in order to embed proteins as a node in PPI network to conserve topological features of PPI network, and also integrate biological data such as gene expression data, gene orthology information and gene subcellular localization information, and introduced a deep architecture for predicting essential genes. Graph isomorphism network architecture is modified in this w
    
[^243]: 后边界桥接：局部注意力必须在全局层之间走向全局吗？

    Post-Boundary Bridge: Must Local Attention Go Global Between Global Layers?

    [https://arxiv.org/abs/2606.02680](https://arxiv.org/abs/2606.02680)

    PBB方法在混合Transformer中保留块内因果注意力，仅在块边界之间添加直接连接，而将长程通信完全交给全注意力层处理，在2.05亿至20.7亿参数规模的稠密和混合专家模型上实现了接近全注意力模型的性能，同时提升了来源检索能力。

    

    混合Transformer通过将局部注意力与周期性的全注意力层相结合，降低了长上下文建模的成本。然而，当全局通信已经可用时，如何最优地利用局部计算仍不明确。我们提出了后边界桥接（Post-Boundary Bridge，PBB），它在块内保留因果注意力，并在块边界之间添加直接连接。PBB并不扩展信息通过连续局部层中继的范围，而是优先进行块内建模和近邻信息交换，将长程通信留给全注意力层。在具有2.05亿至20.7亿存储参数的稠密模型和混合专家模型上，PBB混合模型保持了接近全注意力模型的困惑度，并在标准下游基准测试中表现出竞争力，同时提升了受控来源检索能力。在2.05亿参数规模下，PBB在困惑度上与使用滑动窗口注意力的混合模型相当，并取得了更高的（性能）。注：原文摘要在此处截断。

    arXiv:2606.02680v2 Announce Type: replace  Abstract: Hybrid Transformers reduce the cost of long-context modeling by combining local attention with periodic full-attention layers. When global communication is already available, however, the best use of local computation remains unclear. We introduce Post-Boundary Bridge (PBB), which preserves causal attention within blocks and adds direct connections across their boundaries. Rather than extending the range of information relayed through successive local layers, PBB prioritizes within-block modeling and nearby exchange, leaving long-range communication to full-attention layers. Across dense and mixture-of-experts models with 205 million to 2.07 billion stored parameters, PBB hybrids retain near-Full perplexity and competitive performance on standard downstream benchmarks while improving controlled source retrieval. At 205 million parameters, PBB also matches a hybrid using sliding-window attention (SWA) in perplexity and achieves higher
    
[^244]: 光滑强凸损失下接近最优的机器遗忘效用

    Near-Optimal Machine Unlearning Utility for Smooth Strongly Convex Losses

    [https://arxiv.org/abs/2606.01527](https://arxiv.org/abs/2606.01527)

    本文为光滑强凸损失下的近似 $(\varepsilon, \delta)$-机器遗忘证明了接近紧致的总体超额风险上下界，确定了遗忘的最优统计代价为采样误差加遗忘惩罚，并揭示出 $(\varepsilon, \delta)$-遗忘相比纯 $\varepsilon$-遗忘并无统计优势。

    

    机器遗忘的动机来自法律和面向用户的需求，即从训练好的模型中移除个人数据的影响，例如“被遗忘权”。先前的工作已经为光滑强凸随机优化中的遗忘开发了算法和误差界，但遗忘的基本统计代价仍不明确。我们通过证明近似 $(\varepsilon, \delta)$-遗忘的总体超额风险的上界和下界，几乎解决了这一问题；我们的界在条件数因子范围内是紧的。对于单位球上的均值估计问题，我们的上界与下界完全匹配。事实上，我们的算法实现了 $\varepsilon$-遗忘，这揭示了差分隐私与遗忘之间的一个显著区别：$(\varepsilon, \delta)$-遗忘相比纯 $\varepsilon$-遗忘并没有统计上的优势。最优速率等于通常的采样误差再加上一个遗忘惩罚项。

    arXiv:2606.01527v2 Announce Type: replace  Abstract: Machine unlearning is motivated by legal and user-facing requirements to remove the influence of individuals' data from trained models, such as the right to be forgotten. Prior work has developed algorithms and error bounds for unlearning in smooth strongly convex stochastic optimization but the fundamental statistical cost of unlearning has remained unclear. We nearly resolve this problem by proving upper and lower bounds on the excess population risk of approximate $(\varepsilon, \delta)$-unlearning; our bounds are tight up to a condition-number factor. For mean estimation over the unit ball, our upper and lower bounds match. In fact, our algorithm achieves $\varepsilon$-unlearning, which implies a notable separation between differential privacy and unlearning: $(\varepsilon, \delta)$-unlearning has no statistical advantage over pure $\varepsilon$-unlearning.   The optimal rate is the usual sampling error plus an unlearning penalty
    
[^245]: 遗传算法与梯度下降在训练面向小规模医疗数据集低数据场景的专用神经网络架构中的比较

    Genetic algorithm vs. gradient descent for training a neural network architecture dedicated to low data regimes in small medical datasets

    [https://arxiv.org/abs/2605.27411](https://arxiv.org/abs/2605.27411)

    本文为DEBI-NN架构设计了一种专用的空间反向传播方案，使其能够采用梯度下降进行训练，并在多个数据集上与遗传算法的训练性能进行了比较。

    

    目的/引言：距离编码仿生信息神经网络（DEBI-NN）是一种最近提出的架构，其中连接权重由位于欧几里得空间中的神经元之间的距离来定义。与直接训练权重的经典神经网络相比，这种方法大幅减少了可训练参数的数量。DEBI-NN 的训练过程基于遗传算法（GA），而非深度学习中仍占主导地位的梯度下降（GD）。我们旨在为 DEBI-NN 设计并实现一个 GD 学习器，并评估其与 GA 相比的性能。材料与方法：我们为 DEBI-NN 设计了一种量身定制的空间反向传播方案，并使用合成的非线性“双月牙”数据集、两个临床医学影像放射组学数据集以及一个胎儿心宫缩图（cardiotocography）数据集，对 GD 与 GA 在分类任务上进行了比较。

    arXiv:2605.27411v2 Announce Type: replace-cross  Abstract: Aim/Introduction: Distance-encoding biomorphic-informational neural network (DEBI-NN) is a recently proposed architecture in which connection weights are defined by the distances between neurons positioned in a Euclidian space. This approach drastically reduces the number of trainable parameters compared to classical neural networks in which weights are directly trained. The training process for DEBI-NN is based on a genetic algorithm (GA), rather than gradient descent (GD) which remains the prevailing optimization algorithm in deep learning. We aim to design and implement a GD learner for DEBI-NN and assess its performance compared to GA.   Materials and Methods: We designed a spatial backpropagation scheme tailored to DEBI-NN and carried out a comparison between GD and GA for classification tasks, using a synthetic non-linear "two-moons" dataset, two clinical medical imaging radiomic datasets and a fetal cardiotocography data
    
[^246]: 对抗性注水：理论、算法与领域专用无线基础模型

    Adversarial Water-Filling: Theory, Algorithms, and a Domain-Specific Wireless Foundation Model

    [https://arxiv.org/abs/2605.26163](https://arxiv.org/abs/2605.26163)

    本文提出对抗性注水（AWF）问题及其理论与算法，并构建了基于置换不变信道表示和约束感知图神经网络的领域专用无线基础模型，用于解决多运营商低轨卫星频谱共享中的竞争性功率分配问题。

    

    频率和空间上的竞争性资源分配问题可以被建模为发射功率与最坏情况干扰之间的极小极大博弈。这种建模形式自然出现在多运营商低轨（LEO）卫星频谱共享场景中，不同星座的传输信号会实时地互相干扰。在高斯信道下，相应的功率分配问题具有凸-凹结构并存在唯一鞍点；而离散星座通常导致非凸的水银/注水问题。本文提出了对抗性注水（AWF）问题，并针对这些场景给出相应的理论与算法。此外，我们开发了一个面向AWF的领域专用无线基础模型，用于学习AWF的搜索动态。该架构采用置换不变的信道表示，以及具有稀疏消息传递机制的约束感知图神经网络（GNN）。

    arXiv:2605.26163v2 Announce Type: replace-cross  Abstract: Competitive resource allocation problems over frequency and space can be formulated as minimax interaction between transmit power and worst-case interference. This formulation naturally arises in multi-operator low Earth orbit (LEO) satellite spectrum sharing, where transmissions from competing constellations interfere in real-time. Under Gaussian channels, the corresponding power-allocation problem admits a convex-concave formulation with a unique saddle point. Discrete constellations yield generally nonconvex mercury/water-filling formulations. In this paper we propose the adversarial water-filling (AWF) problem with corresponding theory and algorithms for these settings. In addition, we develop a domain-specific wireless foundation model for AWF to learn the AWF search dynamics. The architecture incorporates permutation-invariant channel representations, a constraint-aware graph neural network (GNN) with sparse message passi
    
[^247]: 批归一化会放大记忆效应与隐私风险

    Batch Normalization Amplifies Memorization and Privacy Risks

    [https://arxiv.org/abs/2605.24420](https://arxiv.org/abs/2605.24420)

    本研究实证发现批归一化（BN）层会显著加深模型对离群样本的记忆，而这种放大的记忆直接转化为更高的隐私泄露风险，使模型更容易受到成员推断攻击。

    

    批归一化（BN）被广泛用于加速深度神经网络的收敛并使训练更加稳定。然而，其对隐私和记忆的影响在很大程度上尚未被探索。在这项工作中，我们研究了BN层对非典型样本或离群样本记忆的影响及其对隐私泄露的影响。我们采用三种互补的方法进行了广泛的实证研究：（i）对分布外样本的意外记忆，（ii）逐样本影响，以及（iii）对成员推断攻击（MIA）的易感性。在多个数据集和架构上，我们一致观察到，与不含BN的模型相比，BN显著增加了对离群样本的记忆。关键的是，这种被放大的记忆直接转化为隐私漏洞：带有BN的模型对成员推断攻击表现出显著更高的易感性。我们通过理论分析对实证发现进行了补充……

    arXiv:2605.24420v2 Announce Type: replace-cross  Abstract: Batch Normalization (BN) is widely adopted to enable faster convergence and more stable training of deep neural networks. However, its impact on privacy and memorization has remained largely unexplored. In this work, we investigate the effect of BN layers on the memorization of atypical or outlier samples and its implications for privacy leakage. We conduct an extensive empirical study using three complementary approaches: (i) unintended memorization of out-of-distribution samples, (ii) per-sample influence, and (iii) susceptibility to membership inference attacks (MIA). Across multiple datasets and architectures, we consistently observe that BN substantially increases the memorization of outliers compared to models without BN. Critically, this amplified memorization translates directly into privacy vulnerabilities: models with BN exhibit significantly higher susceptibility to MIAs. We complement our empirical findings with a m
    
[^248]: 硬同配性约束下图生成的强化学习方法

    Reinforcement Learning for Graph Generation under a Hard Assortativity Constraint

    [https://arxiv.org/abs/2605.23285](https://arxiv.org/abs/2605.23285)

    本文提出一种强化学习框架，通过保度重连的定向传输策略使图精确满足硬同配性约束，在生成成本降低至少一个数量级的同时保留超过98%的构型多样性，并能从小图训练泛化至不同规模与拓扑。

    

    摘要：生成具有精确受控结构性质的图系综，是研究网络结构如何塑造功能的核心问题。经典系综仅在期望意义上施加约束（软约束），使单个实现围绕目标波动，而除固定度序列之外，在每个实现中以规定精度强制执行硬约束仍然极具挑战性。本文展示了一个强化学习框架，可以通过保持度的重连操作驱动图满足规定的同配性，该性质刻画了相邻节点之间的度-度相关性。通过用定向传输取代熵主导的Metropolis-Hastings随机游走，学习到的策略将生成成本降低至少一个数量级，同时保留了超过98%的构型多样性。该框架在小图上训练后，能够泛化到不同规模和拓扑的图。

    arXiv:2605.23285v2 Announce Type: replace-cross  Abstract: Generating graph ensembles with precisely controlled structural properties is central to investigating how network structure shapes function. Canonical ensembles impose constraints only in expectation (soft constraints), letting individual realizations fluctuate around the target, whereas enforcing hard constraints with prescribed precision in every realization remains challenging beyond fixing the degree sequence. Here we show that a reinforcement learning framework can drive a graph through degree-preserving rewirings to satisfy a prescribed assortativity, which characterizes the degree--degree correlation of adjacent nodes. By replacing the entropically dominated Metropolis--Hastings random walk with directed transport, the learned policy reduces generation cost by at least an order of magnitude while retaining over 98\% of configurational diversity. Trained on small graphs, the framework generalizes across sizes and topolog
    
[^249]: 基于自适应路由状态的多分辨率归因

    Multi-Resolution Attribution from Adaptive Routing State

    [https://arxiv.org/abs/2605.22866](https://arxiv.org/abs/2605.22866)

    本文证明自适应分层系统中学习到的路由状态本身即可定义一种多分辨率的一致性归因——叶子节点值为路径权重乘积、内部节点为前缀乘积，且细粒度读数恰好加和等于粗粒度读数，并在LLM、人口普查、智能体和电信网络等层次结构中于多个层级揭示出有意义结构。

    

    自适应分层系统在学习选择哪些组件的过程中会不断积累路由状态。我们证明，这种状态本身已经在整个层次结构上定义了一种连贯的归因：叶子节点接收其路径上局部路由权重的乘积，而内部节点则接收相应的前缀乘积。因此，同一份学习到的状态可以在组级别和组件级别上一致地读取，并且每个更细粒度的读取结果恰好等于其对应粗粒度读取结果的总和。这种归因描述的是已部署路由器所学习到的偏好，而非某个组件的内在价值或反事实价值。在大语言模型（LLM）、人口普查、智能体（agentic）以及电信网络等层次结构中，学习到的状态在多个层次上都包含有意义的结构，而最清晰的组织结构并不一定出现在叶子层。在电信网络研究中，站点（Site）级或区域（Region）级的读取通常比小区级读取揭示出更清晰的结构。与Shapley归因的比较……

    arXiv:2605.22866v2 Announce Type: replace  Abstract: Adaptive hierarchical systems accumulate routing state as they learn which components to select. We show that this state already defines a coherent attribution over the hierarchy. A leaf receives the product of the local routing weights on its path, while an internal node receives the corresponding prefix product. The same learned state can therefore be read consistently at group and component levels, and every finer readout sums exactly to its coarser counterpart. This attribution describes the preferences learned by the deployed router rather than an intrinsic or counterfactual value of a component. Across LLM, Census, agentic, and telecom-network hierarchies, the learned state contains meaningful structure at several levels, and the clearest organisation need not occur at the leaves. In the telecom study, Site- or Region-level readouts usually reveal clearer structure than Cell-level readouts. Comparison with Shapley attribution c
    
[^250]: 基于雅可比引导的各向异性噪声重塑：在本地差分隐私下提升表示效用

    Jacobian-Guided Anisotropic Noise Reshaping for Enhancing Representation Utility under Local Differential Privacy

    [https://arxiv.org/abs/2605.16812](https://arxiv.org/abs/2605.16812)

    本文通过利用下游模型雅可比矩阵识别任务关键子空间，并将标准LDP的各向同性噪声重塑为各向异性分布，从而在不牺牲隐私保证的前提下显著提升数据表示效用。

    

    摘要：arXiv:2605.16812v3 公告类型：替换 摘要：虽然本地差分隐私（LDP）作为分布式数据收集的基础原语，但其严格的随机化要求往往导致数据效用的严重退化。这种退化源于传统LDP机制的任务无关特性，这些机制扰动所有维度，而不考虑它们对下游目标的相对重要性。为解决这一问题，我们提出了一种新方法，以减轻数据表示中任务相关子空间的噪声。我们的方法通过公共下游模型的雅可比矩阵识别任务关键子空间，沿这些方向选择性地衰减噪声，并将标准LDP机制的各向同性噪声重塑为各向异性分布。所提出的机制保留了底层LDP随机化器的隐私保证，同时异质地调节噪声跨任务方向的影响，从而显著提升效用。

    arXiv:2605.16812v3 Announce Type: replace  Abstract: While Local Differential Privacy (LDP) serves as a foundational primitive for distributed data collection, its stringent randomization requirements often lead to severe degradation in data utility. This degradation stems from the task-agnostic nature of conventional LDP mechanisms, which perturb all dimensions without accounting for their relative importance to the downstream objective. To address this issue, we propose a novel approach that mitigates noise in task-relevant subspaces of the data representation. Our method identifies task-critical subspaces via the Jacobian of a public downstream model, selectively attenuates noise along these directions, and reshapes the isotropic noise of standard LDP mechanisms into an anisotropic distribution. The resulting mechanism preserves the privacy guarantee of the underlying LDP randomizer while heterogeneously modulating the impact of noise across task directions, thereby substantially en
    
[^251]: EfficientTDMPC：改进的MPC目标函数实现样本高效的连续控制

    EfficientTDMPC: Improved MPC Objectives for Sample-Efficient Continuous Control

    [https://arxiv.org/abs/2605.16692](https://arxiv.org/abs/2605.16692)

    EfficientTDMPC通过动力学模型集成、跨不同展开深度平均回报估计以及对规划器目标施加不确定性惩罚来减少模型与价值网络的估计误差，并结合缓冲区数据新鲜度等实用改进，从而在连续控制任务中实现更高效的模型强化学习，并能更好地利用更高的更新-数据比。

    

    我们提出了EfficientTDMPC，这是一种基于TD-MPC算法家族构建的样本高效的模型强化学习方法，用于连续控制任务。该算法家族的核心是一个规划器，旨在找到能够最大化估计回报的动作序列。回报是通过学习到的模型网络和价值网络进行估计的，而这两者都可能引入误差。EfficientTDMPC提出通过两种方式来减少这种误差。首先，它引入了动力学模型集成方法，对不同模型以及不同展开深度上的回报估计进行平均。其次，它增加了对规划器目标施加不确定性惩罚的选项，从而得到一个能够避免回报估计不确定的动作的规划器。此外，它还添加了一些实用改进，以提高缓冲区数据的新鲜度并减少计算量。最后，我们发现这些贡献使EfficientTDMPC能够从更高的更新-数据比（UTD）中获益更多，进一步（提升性能）……

    arXiv:2605.16692v3 Announce Type: replace-cross  Abstract: We introduce EfficientTDMPC, a sample-efficient model-based reinforcement learning method for continuous control built on the TD-MPC family of algorithms. Central to this family is a planner that aims to find an action sequence that maximizes the estimated return. The return is estimated using a learned model and value networks, each of which can introduce error. EfficientTDMPC proposes to reduce this error in two ways. First, it introduces an ensemble of dynamics models and averages the return estimates across those models and across different rollout depths. Second, it adds the option to apply an uncertainty penalty to the planner objective, yielding a planner that avoids actions with uncertain return estimates. It then adds practical improvements which increase buffer data freshness and reduce compute. Lastly, we find that our contributions enable EfficientTDMPC to benefit more from a higher update-to-data (UTD) ratio, furth
    
[^252]: Clin-JEPA：面向电子健康记录患者轨迹的联合嵌入预测预训练的多阶段协同训练框架

    Clin-JEPA: A Multi-Phase Co-Training Framework for Joint-Embedding Predictive Pretraining on EHR Patient Trajectories

    [https://arxiv.org/abs/2605.10840](https://arxiv.org/abs/2605.10840)

    提出Clin-JEPA框架，首次将联合嵌入预测架构（JEPA）引入电子健康记录患者轨迹建模，通过编码器与预测器的多阶段协同训练，使LLM编码器的潜空间围绕患者生理动力学组织，从而实现潜空间中的患者轨迹模拟。

    

    联合嵌入预测架构（JEPA）通过在潜空间中进行预测来学习表示，这一方法已在计算机视觉中得到应用；若在推理时保留动作条件预测器，这类架构便成为潜空间世界模型，从而支持机器人领域的规划能力（如V-JEPA 2-AC）。将这一设计引入电子健康记录（EHR）患者轨迹——即构建一个能在潜空间中模拟患者轨迹演化的预测器——此前尚未被探索。我们采用大语言模型（LLM）作为编码器，将每小时的病历记录作为文本读取，从而避免了繁琐的特征工程和词表统一化处理。然而，经监督微调适配的LLM并不会围绕生理动力学来组织其潜空间；而若沿用V-JEPA 2-AC的做法，冻结编码器仅训练预测器，则编码器无法感知滚动推演信号，导致预测器在滚动推演过程中性能退化。为此，我们提出在统一的潜空间预测目标下协同训练编码器与预测器，使编码器扎根于其预测器必须遵循的动力学规律。朴素的协同训练方法……（摘要原文在此处截断）

    arXiv:2605.10840v5 Announce Type: replace-cross  Abstract: Joint-embedding predictive architectures (JEPA) learn representations by predicting in latent space, as in computer vision; retaining the action-conditioned predictor at inference turns them into latent world models, enabling planning in robotics (V-JEPA 2-AC). Bringing this design to EHR patient trajectories---a predictor that simulates a patient's trajectory in latent space---has not been explored. We use an LLM as the encoder, reading the hourly record as text, avoiding feature engineering and vocabulary harmonisation. But an LLM adapted by supervised fine-tuning does not organise its latent space around physiological dynamics, and freezing it to train the predictor, as in V-JEPA 2-AC, leaves the encoder unaware of the rollout signal: the predictor degrades under rollout. We instead co-train encoder and predictor under one latent-prediction objective, grounding the encoder in the dynamics its predictor must follow. Na\"ive c
    
[^253]: 通过专家损失集成实现时间序列预测混合专家模型的快速训练

    Fast Training of Mixture-of-Experts for Time Series Forecasting via Expert Loss Integration

    [https://arxiv.org/abs/2605.10330](https://arxiv.org/abs/2605.10330)

    提出一种融合专家特定损失与部分在线学习策略的自适应混合专家框架，解决了小门控权重导致的优化难题，在保持计算效率的同时显著提升了时间序列预测性能。

    

    我们提出了一种新颖的自适应混合专家框架用于时间序列预测，该框架通过引入专家特定损失来解决由小门控权重引起的优化问题，专家特定损失为每个专家提供独立于门控分配权重的直接学习信号。具体而言，总体目标由基础预测损失和专家特定损失组成，使各个专家的预测误差能够与总体预测误差一起直接影响参数更新。该框架还鼓励不同的专家从数据的不同时间片段中学习。所提出的框架进一步与部分在线学习策略相结合，实现模型参数的高效增量更新。通过将专家级损失信息与部分在线优化相结合，所提出的方法在保持计算效率的同时提高了预测性能。

    arXiv:2605.10330v2 Announce Type: replace-cross  Abstract: We propose a novel adaptive Mixture-of-Experts (MoE) framework for time series forecasting that addresses the optimization problem arising from small gating weights by incorporating expert-specific losses, which provide each expert with a direct learning signal independent of the gate-assigned weight. Specifically, the overall objective comprises the base forecasting loss and expert-specific losses, allowing individual expert prediction errors to directly influence parameter updates alongside the aggregate forecasting error. The framework also encourages different experts to learn from different temporal segments of the data. The proposed framework is further combined with a partial online learning strategy that enables efficient incremental updates of model parameters. By integrating expert-level loss information with partial online optimization, the proposed method improves forecasting performance while retaining computationa
    
[^254]: 从观察中学习对世界进行理论化

    Learning to Theorize the World from Observation

    [https://arxiv.org/abs/2605.03413](https://arxiv.org/abs/2605.03413)

    本文提出Learning-to-Theorize学习范式及世界理论模型NEO，通过将潜在程序作为习得的“思维语言”，从原始非文本观察中构建显式、可组合、可执行的世界解释性理论。

    

    理解世界意味着什么？当代世界模型通常将“理解”操作化为在潜空间或观察空间中进行准确的未来预测。然而，发展认知科学提出了不同的观点：人类的理解是通过构建关于世界如何运作的内部理论而产生的，甚至在成熟的语言习得之前就已开始。受这种“理论构建”认知观的启发，我们提出了Learning-to-Theorize（学习理论化），这是一种从原始的非文本观察中推断出显式的世界解释性理论的学习范式。我们用神经理论化器来实例化这一范式，这是一个世界理论模型，它将潜在程序归纳为一种习得的“思维语言”，并通过共享的状态转移模型来执行这些程序。在NEO中，理论被表示为可执行的、可组合的程序，其习得的基元可以被系统地重新组合，以解释新出现的现象。

    arXiv:2605.03413v3 Announce Type: replace-cross  Abstract: What does it mean to understand the world? Contemporary world models often operationalize understanding as accurate future prediction in latent or observation space. Developmental cognitive science, however, suggests a different view: human understanding emerges through the construction of internal theories of how the world works, even before mature language is acquired. Inspired by this theory-building view of cognition, we introduce Learning-to-Theorize, a learning paradigm for inferring explicit explanatory theories of the world from raw, non-textual observations. We instantiate this paradigm with the Neural Theorizer (NEO), a World Theory Model, that induces latent programs as a learned Language of Thought and executes them through a shared transition model. In NEO, a theory is represented as an executable, compositional program whose learned primitives can be systematically recombined to explain novel phenomena. Experiment
    
[^255]: 面向表格基础模型的模型感知数据清洗

    Model-Aware Data Cleaning for Tabular Foundation Models

    [https://arxiv.org/abs/2604.25154](https://arxiv.org/abs/2604.25154)

    该论文提出L2C-TFM方法，通过引入以Wasserstein距离分布稳定性作为正则项的模型感知奖励，训练强化学习策略来智能编排表格数据清洗算子，从而改善表格基础模型在含脏数据场景下的表现。

    

    表格基础模型在小规模表格数据集上实现了最先进的零样本准确率，但其上下文学习假设输入数据近似干净：现实世界中的缺失值、异常值和重复数据会造成先验不匹配，从而同时降低准确率与校准度。我们研究了基于强化学习的表格数据清洗，即一种学习到的策略来编排清洗算子的执行顺序，并提出了带有模型感知奖励的L2C-TFM（TFMAwareReward）。我们明确说明了该奖励所优化的内容：它对清洗后数据与原始（脏）数据之间的Wasserstein距离进行正则化，这是一个分布稳定性项，我们将其作为诊断指标进行测量。在十个OpenML数据集上开展的六项实验表明：七种奖励设计中有三种会退化为无效策略，因此奖励工程并非易事；在8随机种子重复留出协议下，模型感知奖励在准确率上与随机森林奖励基线相当……

    arXiv:2604.25154v2 Announce Type: replace  Abstract: Tabular Foundation Models (TFMs) achieve state-of-the-art zero-shot accuracy on small tabular datasets, but their in-context learning assumes approximately clean inputs: real- world missing values, outliers, and duplicates create a prior mismatch that degrades both accuracy and calibration. We study reinforcement learning for tabular data cleaning, a learned policy that sequences cleaning operators and introduce L2C-TFM with a model-aware reward (TFMAwareReward). We are explicit about what this reward optimizes: it regularizes the Wasserstein distance between the cleaned and the original (dirty) data, a distributional-stability term, which we measure as a diagnostic. Across six experiments on ten OpenML datasets: (i) three of seven reward designs collapse to degenerate strategies, so reward engineering is non-trivial; (ii) under an 8-seed repeated-holdout protocol the model-aware reward matches a random-forest-reward baseline on accu
    
[^256]: Green-ELM：基于高维随机投影的高效解析学习

    Green-ELM: Efficient Analytic Learning via High-Dimensional Random Projections

    [https://arxiv.org/abs/2604.15613](https://arxiv.org/abs/2604.15613)

    Green-ELM通过高维随机投影和闭式解析解（Moore-Penrose伪逆、LU与Cholesky分解）一次性求解输出层，完全避免了反向传播，在MNIST上达到98.10%准确率的同时大幅降低计算开销。

    

    我们提出了Green-ELM，这是一种非迭代的神经网络架构，它在固定的高维随机特征表示上采用闭式解析解，取代了基于梯度的输出层优化。通过将输入流形投影到高维随机特征空间（d ≫ 784），我们的结果表明，无需反向传播的计算开销，即可有效解开复杂的类别边界。利用Moore-Penrose伪逆、LU分解和Cholesky分解在单个解析步骤中求解输出层，Green-ELM在MNIST（d=4000）上达到了98.10%的分类准确率，在Fashion-MNIST上达到了86.63%。此外，我们实验了基于ResNet-18的预训练“冻结骨干网络”来提取高质量特征，并证明这些一次性求解器在简单数据集之外同样有效。值得注意的是，我们在MNIST（d=2000）上的基线CPU配置实现了……（原文摘要在此处截断）

    arXiv:2604.15613v4 Announce Type: replace-cross  Abstract: We present Green-ELM, a non-iterative neural architecture that replaces gradient-based optimization of the output layer with a closed-form analytic solution over a fixed, high-dimensional random feature representation. By projecting input manifolds into a high-dimensional, random feature space ($d \gg 784$), our results show that complex class boundaries can be effectively untangled without the computational overhead of backpropagation.   Utilizing the Moore-Penrose pseudoinverse, LU and Cholesky decomposition to solve for the output layer in a single analytic step, Green-ELM achieves a classification accuracy of 98.10\% on MNIST ($d=4000$) and 86.63\% on Fashion-MNIST. Furthermore, we experiment with a pre-trained ``frozen-backbone'' based on ResNet-18 to extract high-quality features and show that these one-shot solvers are effective beyond simple datasets. Notably, our baseline CPU configuration on MNIST ($d=2000$) achieves 
    
[^257]: 转录组学免疫治疗反应预测模型的跨队列泛化能力有限

    Transcriptomic Models for Immunotherapy Response Prediction Show Limited Cross-cohort Generalisability

    [https://arxiv.org/abs/2604.05478](https://arxiv.org/abs/2604.05478)

    本研究系统评估了九种最先进的转录组学免疫治疗反应预测模型，发现它们在独立外部数据集上的预测性能有限，表明现有模型的跨队列泛化能力不足，需要在临床应用前进一步改进。

    

    免疫检查点抑制剂（ICIs）已经改变了癌症治疗方法；然而，相当大比例的患者表现出原发耐药或获得性耐药，使得准确的治疗前反应预测成为一个关键的未满足的临床需求。基于bulk和单细胞RNA测序（scRNA-seq）数据开发的转录组学生物标志物为捕获肿瘤-免疫相互作用提供了有前景的途径，然而现有预测模型的跨队列泛化能力仍不清楚。我们系统地基准测试了九种最先进的转录组学ICI反应预测模型，包括五种基于bulk RNA-seq的模型（COMPASS、IRNet、NetBio、IKCScore和TNBC-ICI）和四种基于scRNA-seq的模型（PRECISE、DeepGeneX、Tres和scCURE），测试使用了模型开发阶段未见的公开独立数据集。总体而言，预测性能表现一般：bulk RNA-seq模型在大多数队列中的表现接近随机水平，而scRNA-seq模型仅显示出...

    arXiv:2604.05478v4 Announce Type: replace-cross  Abstract: Immune checkpoint inhibitors (ICIs) have transformed cancer therapy; yet substantial proportion of patients exhibit intrinsic or acquired resistance, making accurate pre-treatment response prediction a critical unmet need. Transcriptomics-based biomarkers derived from bulk and single-cell RNA sequencing (scRNA-seq) offer a promising avenue for capturing tumour-immune interactions, yet the cross-cohort generalisability of existing prediction models remains unclear.We systematically benchmark nine state-of-the-art transcriptomic ICI response predictors, five bulk RNA-seq-based models (COMPASS, IRNet, NetBio, IKCScore, and TNBC-ICI) and four scRNA-seq-based models (PRECISE, DeepGeneX, Tres and scCURE), using publicly available independent datasets unseen during model development. Overall, predictive performance was modest: bulk RNA-seq models performed at or near chance level across most cohorts, while scRNA-seq models showed only
    
[^258]: 自动化成员推断攻击：利用大语言模型智能体发现MIA信号计算方法

    Automated Membership Inference Attacks (AutoMIA): Discovering MIA Signal Computations using LLM Agents

    [https://arxiv.org/abs/2603.19375](https://arxiv.org/abs/2603.19375)

    提出AutoMIA框架，利用大语言模型智能体自动探索和发现针对特定目标模型与数据集定制的新型成员推断攻击信号计算方法。

    

    成员推断攻击使攻击者能够确定特定数据点是否属于模型的训练数据集，已成为理解、评估和量化机器学习系统潜在信息泄露的重要框架。设计有效的成员推断攻击是一项具有挑战性的任务，通常需要对模型行为进行大量的人工探索以识别潜在漏洞。在本文中，我们介绍了AutoMIA——一个利用大语言模型（LLM）智能体自动化设计和实现新成员推断攻击信号计算的新颖框架。通过利用LLM智能体，我们能够系统地探索庞大的潜在攻击策略空间，从而发现新颖的攻击策略。我们的实验表明，AutoMIA能够成功发现专门针对用户配置的目标模型和数据集定制的新型成员推断攻击方法，从而实现性能提升。

    arXiv:2603.19375v2 Announce Type: replace-cross  Abstract: Membership inference attacks (MIAs), which enable adversaries to determine whether specific data points were part of a model's training dataset, have emerged as an important framework to understand, assess, and quantify the potential information leakage associated with machine learning systems. Designing effective MIAs is a challenging task that usually requires extensive manual exploration of model behaviors to identify potential vulnerabilities. In this paper, we introduce AutoMIA -- a novel framework that leverages large language model (LLM) agents to automate the design and implementation of new MIA signal computations. By utilizing LLM agents, we can systematically explore a vast space of potential attack strategies, enabling the discovery of novel strategies. Our experiments demonstrate AutoMIA can successfully discover new MIAs that are specifically tailored to user-configured target model and dataset, resulting in impro
    
[^259]: MAPLE：元数据增强的私有语言演化

    MAPLE: Metadata Augmented Private Language Evolution

    [https://arxiv.org/abs/2603.19258](https://arxiv.org/abs/2603.19258)

    MAPLE通过引入元数据增强，解决了私有演化（PE）方法在私有数据分布偏离基础模型预训练先验时的初始化瓶颈问题，实现了更高效的基于API的差分隐私合成数据生成。

    

    对大语言模型（LLM）进行差分隐私（DP）微调需要巨大的计算资源和完整的模型访问权限，这使得普通用户无法使用最先进的专有API。生成差分隐私合成数据提供了一种实用的替代方案。这种方法还允许进行透明的探索性数据分析以及在下游任务中的任意重用，从而避开了模型参数空间的刚性约束。私有演化（PE）为生成此类数据提供了一个有前景的基于API的框架，但其成功在很大程度上依赖于初始化。如果私有数据分布与基础模型的预训练先验偏差过大——这在高度专业化的领域中很常见——PE就难以与目标数据对齐。这种不对齐会导致收敛性差、效用下降以及API调用的浪费。为解决这一初始化瓶颈，我们提出了元数据增强的私有语言演化（MAPLE）

    arXiv:2603.19258v3 Announce Type: replace-cross  Abstract: Differentially private (DP) fine-tuning of large language models (LLMs) requires massive compute and full model access, which rules out state-of-the-art proprietary APIs for general users. Generating DP synthetic data offers a practical workaround. This approach also allows for transparent exploratory data analysis and arbitrary reuse across downstream tasks, sidestepping the rigid constraints of a model's parameter space. Private Evolution (PE) provides a promising API-based framework for generating this data, but its success relies heavily on initialization. If the private data distribution falls too far outside the foundation model's pre-training priors -- a common issue in highly specialized domain -- PE struggles to align with the target data. This misalignment causes poor convergence, degraded utility, and wasted API calls. To solve this initialization bottleneck, we introduce Metadata Augmented Private Language Evolution
    
[^260]: TTSR：通过反思进行测试时自我进化

    TTSR: Test-Time Self-Evolving via Reflection

    [https://arxiv.org/abs/2603.03297](https://arxiv.org/abs/2603.03297)

    TTSR通过让单个模型交替扮演学生和教师角色，基于反思后合成的范式，在测试时针对失败轨迹生成变体问题，从而克服了缺乏可学习样本和探索效率低下的瓶颈。

    

    测试时训练（TTT）在推理过程中仅使用未标记的测试输入来适应大型语言模型（LLMs）。然而，现有方法在困难推理任务上面临两个主要瓶颈：（1）\emph{缺乏可学习样本}，因为困难问题上自生成的伪标签往往带有噪声，导致不稳定的奖励；（2）\emph{探索效率低下}，因为性能提升依赖于反复采样大量生成结果，而没有对先前尝试失败原因进行明确诊断。我们提出\textbf{TTSR}（\textbf{T}est-\textbf{T}ime \textbf{S}elf-\textbf{R}eflection），一个基于\emph{先反思后合成}范式的自我进化框架。一个预训练模型在\textit{学生}和\textit{教师}两种角色间交替：学生解决测试问题并进行更新，而教师分析失败轨迹并合成更接近学生能力边界的有针对性的变体问题。TTSR进一步...

    arXiv:2603.03297v2 Announce Type: replace  Abstract: Test-time training (TTT) adapts large language models (LLMs) during inference using only unlabeled test inputs. Existing methods, however, face two major bottlenecks on hard reasoning tasks: (1) \emph{lack of learnable samples}, as self-generated pseudo-labels on difficult questions are often noisy and yield unstable rewards; and (2) \emph{inefficient exploration}, as performance gains depend on repeatedly sampling many rollouts without explicit diagnosis of why previous attempts fail. We propose \textbf{TTSR} (\textbf{T}est-\textbf{T}ime \textbf{S}elf-\textbf{R}eflection), a self-evolving framework based on a \emph{reflect-then-synthesize} paradigm. A single pretrained model alternates between a \textit{Student} role and a \textit{Teacher} role: the Student solves test questions and updates, while the Teacher analyzes failed trajectories and synthesizes targeted variant questions closer to the Student's capability frontier. TTSR fur
    
[^261]: 高分辨率距离像分类器需要方位角感知

    High-Resolution Range Profile Classifiers Require Aspect-Angle Awareness

    [https://arxiv.org/abs/2603.00087](https://arxiv.org/abs/2603.00087)

    本研究表明，高分辨率距离像分类器通过显式利用方位角信息可平均提升约7%的分类准确率，且即使方位角通过因果卡尔曼滤波器在线估计获得，大部分性能增益依然能够保留。

    

    我们重新审视了基于方位角条件化的高分辨率距离像（HRRP）分类问题。以往的研究通常假设方位角信息在训练期间不完整或在推理阶段不可用，而我们研究了一种设置，其中角度信息对所有训练样本均可用，并被显式提供给分类器。通过使用三个数据集以及广泛的条件化策略和模型架构，我们证明单帧分类器和序列分类器均能持续地从方位角感知中受益，平均准确率提升约7%，根据模型和数据集的不同，提升幅度最高可达10%。在实际应用中，方位角无法直接测量，必须通过估计获得。我们证明因果卡尔曼滤波器可以在线估计方位角，中位误差为5°，并且使用估计角度进行训练和推理能够保留大部分性能收益，这支持了所提方法在现实场景中的可行性。

    arXiv:2603.00087v2 Announce Type: replace-cross  Abstract: We revisit High-Resolution Range Profile (HRRP) classification with aspect-angle conditioning. While prior work often assumes that aspect-angle information is incomplete during training or unavailable at inference, we study a setting where angles are available for all training samples and explicitly provided to the classifier. Using three datasets and a broad range of conditioning strategies and model architectures, we show that both single-profile and sequential classifiers benefit consistently from aspect-angle awareness, with an average accuracy gain of about 7% and improvements of up to 10%, depending on the model and dataset. In practice, aspect angles are not directly measured and must be estimated. We show that a causal Kalman filter can estimate them online with a median error of 5{\textdegree}, and that training and inference with estimated angles preserves most of the gains, supporting the proposed approach in realist
    
[^262]: 探索对抗攻击中任意Lp范数的稀疏性与平滑性

    Exploring Sparsity and Smoothness of Arbitrary Lp Norms in Adversarial Attacks

    [https://arxiv.org/abs/2602.06578](https://arxiv.org/abs/2602.06578)

    该论文系统研究了 ℓp 范数中参数 p（p∈[1,2]）的取值如何影响对抗扰动的稀疏性与平滑性，并提出了基于平滑操作和一阶泰勒近似的平滑性度量框架，填补了范数选择与扰动结构特性之间关系的研究空白。

    

    针对深度神经网络的对抗攻击通常在 ℓp 范数约束下构造，最常使用 p=1、p=2 或 p=∞，并可能针对稀疏性或平滑性等特定需求进行正则化。这些选择通常是在没有系统研究范数参数 p 如何影响对抗扰动的结构和感知特性的情况下做出的。在本工作中，我们研究了 p 的取值如何影响在 ℓp 范数约束下生成的对抗攻击的稀疏性与平滑性，其中 p 的取值范围为 p∈[1,2]。为了实现定量分析，我们采用了文献中已有的两种稀疏性度量方法，并引入了三种平滑性度量方法。特别地，我们提出了一个基于平滑操作来推导平滑性度量的通用框架，并额外提出了一种基于一阶泰勒近似的平滑性度量。使用这些度量……

    arXiv:2602.06578v2 Announce Type: replace-cross  Abstract: Adversarial attacks against deep neural networks are commonly constructed under $\ell_p$ norm constraints, most often using $p=1$, $p=2$ or $p=\infty$, and potentially regularized for specific demands such as sparsity or smoothness. These choices are typically made without a systematic investigation of how the norm parameter $p$ influences the structural and perceptual properties of adversarial perturbations. In this work, we study how the choice of $p$ affects sparsity and smoothness of adversarial attacks generated under $\ell_p$ norm constraints for values of $p \in [1,2]$. To enable a quantitative analysis, we adopt two established sparsity measures from the literature and introduce three smoothness measures. In particular, we propose a general framework for deriving smoothness measures based on smoothing operations and additionally introduce a smoothness measure based on first-order Taylor approximations. Using these measu
    
[^263]: 扰动相位：分析复值神经网络的对抗鲁棒性

    Perturbing the Phase: Analyzing Adversarial Robustness of Complex-Valued Neural Networks

    [https://arxiv.org/abs/2602.06577](https://arxiv.org/abs/2602.06577)

    本文提出了专门针对复值输入相位信息的"相位攻击"并推导了常用对抗攻击的复值版本，发现复值神经网络在某些场景下比实值神经网络更鲁棒，但两者都对相位变化极为敏感，相位攻击造成的性能下降超过同等强度的常规攻击。

    

    复值神经网络（CVNNs）在各类应用中日益流行。为了在实践中安全地使用CVNNs，分析它们对异常值的鲁棒性至关重要。理解深度神经网络行为的一种公认技术是研究其在对抗攻击下的行为，对抗攻击可被视为最坏情况下的最小扰动。我们设计了相位攻击，这是一种专门针对复值输入相位信息的攻击方法。此外，我们还推导了常用对抗攻击的复值版本。研究表明，在某些场景下CVNNs比实值神经网络（RVNNs）更具鲁棒性，且两者对相位变化都非常敏感——相位攻击对模型性能的降低程度超过了同样强度的、可同时攻击相位和幅度的常规攻击。

    arXiv:2602.06577v2 Announce Type: replace-cross  Abstract: Complex-valued neural networks (CVNNs) are rising in popularity for all kinds of applications. To safely use CVNNs in practice, analyzing their robustness against outliers is crucial. One well known technique to understand the behavior of deep neural networks is to investigate their behavior under adversarial attacks, which can be seen as worst case minimal perturbations. We design Phase Attacks, a kind of attack specifically targeting the phase information of complex-valued inputs. Additionally, we derive complex-valued versions of commonly used adversarial attacks. We show that in some scenarios CVNNs are more robust than RVNNs and that both are very susceptible to phase changes with the Phase Attacks decreasing the model performance more, than equally strong regular attacks, which can attack both phase and magnitude.
    
[^264]: 重新思考扩散模型强化学习的设计空间：超越损失函数设计的似然估计之重要性

    Rethinking the Design Space of Reinforcement Learning for Diffusion Models: On the Importance of Likelihood Estimation Beyond Loss Design

    [https://arxiv.org/abs/2602.04663](https://arxiv.org/abs/2602.04663)

    本文系统解耦并分析了扩散模型强化学习设计空间中的三个因素，发现采用仅从最终生成样本计算的基于证据下界（ELBO）的似然估计器，是决定算法有效性与效率的最关键因素，其重要性超越了损失函数的设计本身。

    

    强化学习已被广泛应用于文本到图像生成等视觉任务的扩散模型和流模型。然而，这些任务仍然充满挑战，因为扩散模型具有难以处理的似然，这为直接应用流行的策略梯度类方法设置了障碍。现有方法主要侧重于在已经高度工程化的大语言模型目标基础上构建新目标，并使用临时的似然估计器，而没有深入探究这种估计如何影响整体算法性能。在这项工作中，我们通过解耦三个因素，对强化学习的设计空间进行了系统分析：i）策略梯度目标，ii）似然估计器，以及iii）rollout采样方案。我们表明，采用基于证据下界（ELBO）的模型似然估计器，且仅从最终生成样本进行计算，是实现高效、有效性能的主导因素。

    arXiv:2602.04663v3 Announce Type: replace-cross  Abstract: Reinforcement learning has been widely applied to diffusion and flow models for visual tasks such as text-to-image generation. However, these tasks remain challenging because diffusion models have intractable likelihoods, which creates a barrier for directly applying popular policy-gradient type methods. Existing approaches primarily focus on crafting new objectives built on already heavily engineered LLM objectives, using ad hoc estimators for likelihood, without a thorough investigation into how such estimation affects overall algorithmic performance. In this work, we provide a systematic analysis of the RL design space by disentangling three factors: i) policy-gradient objectives, ii) likelihood estimators, and iii) rollout sampling schemes. We show that adopting an evidence lower bound (ELBO) based model likelihood estimator, computed only from the final generated sample, is the dominant factor enabling effective, efficient
    
[^265]: QUATRO：面向大语言模型微调的查询自适应信赖域策略优化

    QUATRO: Query-Adaptive Trust Region Policy Optimization for LLM Fine-tuning

    [https://arxiv.org/abs/2602.04620](https://arxiv.org/abs/2602.04620)

    QUATRO提出了一种查询自适应信赖域策略优化方法，通过直接强制执行信赖域约束来替代GRPO的启发式裁剪与组归一化，实现了大语言模型微调中稳定、熵受控的策略更新。

    

    基于GRPO风格的强化学习（RL）大语言模型微调算法近期广受欢迎。然而，由于依赖启发式的信赖域近似，这些算法可能导致脆弱的优化行为，因为全局重要性比率裁剪和组内归一化无法调控重要性比率超出裁剪范围的样本。我们提出了查询自适应信赖域策略优化（QUATRO），它通过一种有原则的优化方法直接强制执行信赖域约束。这产生了一个清晰且可解释的目标函数，能够对策略更新进行显式控制，实现稳定且熵受控的优化，稳定项由精确的信赖域公式内在产生。在多样化的数学推理基准上的实证验证表明，QUATRO在策略滞后加剧和学习率激进的情况下仍能保持稳定训练，并全程维持良好的熵控制。

    arXiv:2602.04620v3 Announce Type: replace  Abstract: GRPO-style reinforcement learning (RL)-based LLM fine-tuning algorithms have recently gained popularity. Relying on heuristic trust-region approximations, however, they can lead to brittle optimization behavior, as global importance-ratio clipping and group-wise normalization fail to regulate samples whose importance ratios fall outside the clipping range. We propose Query-Adaptive Trust-Region policy Optimization (QUATRO), which directly enforces trust-region constraints through a principled optimization. This yields a clear and interpretable objective that enables explicit control over policy updates and stable, entropy-controlled optimization, with a stabilizer terms arising intrinsically from the exact trust-region formulation. Empirically verified on diverse mathematical reasoning benchmarks, QUATRO shows stable training under increased policy staleness and aggressive learning rates, maintaining well-controlled entropy throughou
    
[^266]: 面向扩散光学层析成像中严重不适定问题的基于分数的扩散模型

    Score-based diffusion models for severely ill-posed problems in diffuse optical tomography

    [https://arxiv.org/abs/2602.03449](https://arxiv.org/abs/2602.03449)

    本文针对扩散光学层析成像这一严重不适定的逆问题，提出了一种由学习成分与基于模型的成分相结合的混合分数正则化策略，以提升基于分数的扩散模型在真实实验测量条件下的重建质量。

    

    基于分数的扩散模型是近来发展的一种用于贝叶斯逆问题后验采样的框架，通过利用从经验数据中学习到的具有强表达能力的先验分布，能够在逆问题中实现高质量的重建。尽管此类模型在实证中表现优异，并日益受到机器学习界的关注，但它们在使用实验测量数据的现实且严重不适定的逆问题中的行为仍未得到充分探索。扩散光学层析成像（DOT）是一个逆边值问题，它利用近红外光的边界测量来恢复生物组织中空间变化的吸收和散射参数。该问题高度不适定，对测量噪声和建模误差尤其敏感。我们通过构建由学习成分和基于模型的成分组成的混合分数，引入了一种正则化策略。我们证明……

    arXiv:2602.03449v2 Announce Type: replace-cross  Abstract: Score-based diffusion models are a recently developed framework for posterior sampling in Bayesian inverse problems, enabling high-quality reconstructions in inverse problems by leveraging expressive prior distributions learned from empirical data. Despite their strong empirical performance and growing interest within the machine learning community, their behaviour in realistic, severely ill-posed inverse problems with experimental measurement data remains under-explored. Diffuse optical tomography (DOT) is an inverse boundary value problem that uses boundary measurements of near-infrared light to recover spatially varying absorption and scattering parameters in biological tissue. The problem is highly ill-posed and particularly sensitive to both measurement noise and modelling errors. We introduce a regularization strategy by constructing a mixed score consisting of a learned component and a model-based component. We show that
    
[^267]: 论缺失数据的固有隐私放大效应

    On the Inherent Privacy Amplification of Missing Data

    [https://arxiv.org/abs/2602.01928](https://arxiv.org/abs/2602.01928)

    该论文提出了一个将缺失数据融入差分隐私的新框架，从形式上证明并刻画了数据缺失对隐私保护的固有放大效应，即缺失特征本身天然地增强了个人隐私保护。

    

    隐私保护在医学和金融等许多高风险领域至关重要，在这些领域中，敏感数据必须在不损害个人机密性的前提下进行分析。与此同时，这些应用所涉及的数据集往往由于无响应或数据损坏等原因而天然存在缺失值。传统上，缺失数据是通过其对统计效率和模型性能的影响来研究的。事实上，缺失数据减少了分析者可获得的信息，并可能降低模型的最终效用。在这项工作中，我们采取了另一种方法，从隐私保护的角度来研究缺失数据。直观地说，当特征缺失时，关于个人的信息被揭示得更少，这表明数据缺失可能在本质上增强隐私。我们在一个将缺失数据融入差分隐私的新颖框架中形式化了这一直觉。本质上，我们的方法考虑了……

    arXiv:2602.01928v3 Announce Type: replace-cross  Abstract: Privacy preservation is critical in many high-stakes domains such as medicine and finance, where sensitive data must be analyzed without compromising individual confidentiality. At the same time, these applications often involve datasets with inherent missing values due to non-response or data corruption for example. Missing data is traditionally analyzed through its impact on statistical efficiency and model performance. In fact, it reduces the information available to analysts and can degrade the final utility of the model. In this work, we take an alternative approach and study missing data through the lens of privacy preservation. Intuitively, when features are missing, less information is revealed about individuals, suggesting that data missingness could inherently enhance privacy. We formalize this intuition within a novel framework that integrates missing data into differential privacy. In essence, our approach accounts 
    
[^268]: 夸克-胶子喷注分类中图像处理模型的比较

    Comparison of Image Processing Models in Quark Gluon Jet Classification

    [https://arxiv.org/abs/2602.00141](https://arxiv.org/abs/2602.00141)

    该研究在使用相同三通道喷注图像表示的条件下比较了CNN、ViT和Swin变换器，发现保留强局部成分的CNN和Swin模型始终优于ViT，表明局部喷注子结构在夸克-胶子判别中起关键作用。

    

    夸克-胶子判别为研究不同机器学习架构如何学习QCD辐射的空间结构提供了一个有用的测试案例。在这项工作中，我们使用相同的三通道喷注图像表示——包含来自PYTHIA 8喷注的带电粒子动量、中性粒子动量和带电粒子多重性——对卷积神经网络（CNN）、视觉变换器和分层Swin变换器进行了比较。我们研究了它们在不同训练集大小和微调配置下的性能，并特别关注喷注图像中局部信息和全局信息的作用。在所研究的案例中，CNN和Swin模型始终优于ViT。由于CNN和Swin在架构中都保留了较强的局部成分，这表明局部喷注子结构在夸克-胶子判别中起着重要作用。分层Swin模型的性能

    arXiv:2602.00141v2 Announce Type: replace-cross  Abstract: Quark-gluon discrimination provides a useful test case for studying how different machine-learning architectures learn the spatial structure of QCD radiation. In this work, we compare convolutional neural network (CNN), Vision Transformers (ViT), and hierarchical Swin Transformers using the same three-channel jet-image representation, consisting of charged-particle momentum, neutral-particle momentum, and charged-particle multiplicity from PYTHIA 8 jets. We study their performance for different training-set sizes and fine-tuning configurations, with particular attention to the role of local and global information in the jet images. CNN and Swin models consistently perform better than ViT in the cases studied. Since both CNN and Swin retain a strong local component in their architectures, this suggests that local jet substructure plays an important role in quark-gluon discrimination. The performance of the hierarchical Swin mode
    
[^269]: 面向一次训练按预算推理的弹性谱状态空间模型

    Elastic Spectral State Space Models for Train-Once Budgeted Inference

    [https://arxiv.org/abs/2601.22488](https://arxiv.org/abs/2601.22488)

    提出ES-SSM框架，通过对SSM序列算子的Hankel谱逼近并结合输入自适应门控与预算丢弃，实现“一次训练即可导出多预算模型”的弹性序列建模，避免了为不同资源约束重复训练或蒸馏模型的成本。

    

    现代序列模型通常在固定的计算能力下训练，而现实世界的应用需要在具有不同资源约束的平台上进行部署。现有方法要么针对选定的预算分别训练或蒸馏独立的紧凑模型，要么采用收缩维度（如宽度、深度、前馈网络（FFN）大小或SSM状态维度）的弹性架构。在本文中，我们提出了弹性谱状态空间模型（ES-SSM），这是一种“一次训练、多次导出”的序列建模框架，通过对SSM序列算子进行谱逼近来获得弹性。ES-SSM建立在状态空间模型的Hankel谱滤波之上，其中长程token混合通过固定的Hankel谱通道来表示，这些通道定义了算子级别的逼近分辨率。为了使这种谱弹性能够可靠地部署，ES-SSM将输入自适应的通道级门控与预算丢弃相结合（摘要在此处被截断）

    arXiv:2601.22488v3 Announce Type: replace  Abstract: Modern sequence models are typically trained at a fixed computational capacity, while real-world applications require deployment across platforms with different resource constraints. Existing approaches either train or distill separate compact models for selected budgets, or use elastic architectures that shrink dimensions such as width, depth, Feed-Forward Network (FFN) size, or SSM state dimension. In this paper, we propose Elastic Spectral State Space Models (ES-SSM), a train-once, export-many sequence modeling framework that gains elasticity through spectral approximation of the SSM sequence operator. ES-SSM builds on Hankel spectral filtering for state space models, where long-range token mixing is represented through fixed Hankel spectral channels that define an operator-level approximation resolution. To make this spectral elasticity reliably deployable, ES-SSM combines input-adaptive channel-wise gates with budget dropout, tr
    
[^270]: 为什么在Adam优化器中 $\beta_1 = \beta_2$ 在动力学上是特殊的

    Why $\beta_1 = \beta_2$ Is Dynamically Special in Adam

    [https://arxiv.org/abs/2601.21739](https://arxiv.org/abs/2601.21739)

    本文揭示了Adam优化器中 $\beta_1 = \beta_2$ 在动力学上特殊的具体机制：连续时间极限下，归一化更新中与两个记忆时间差成正比的幅度滞后项恰好在两参数相等时消失，使得对角线区域成为结构上不存在失配诱发响应的唯一情形。

    

    Adam优化器在大规模训练的核心地位已持续近十年，但其两个动量参数的作用仍然知之甚少。近期研究表明，将 $\beta_{1}=\beta_{2}$ 绑定取相同值时，即使把两个记忆尺度合并为一个，Adam依然能保持其强大性能，这引出了一个基本问题：当两个记忆被绑定时，动力学上究竟有什么变得特殊？我们识别出了一个具体的机制。在连续时间极限下，每个归一化更新坐标可以分解为一个符号分量、一个与两个记忆时间之差成正比的显式幅度滞后项，以及额外的过渡项、曲率项和非线性比率项。这一滞后通道恰好在 $\beta_{1}=\beta_{2}$ 时消失，使得对角线（两参数相等）成为这种失配所引起的响应在结构上不存在的唯一区域。在真实训练梯度上进行的全历史离散分解也恢复了这种组成上的变化：绑定后的更新以符……

    arXiv:2601.21739v3 Announce Type: replace-cross  Abstract: Adam has been at the core of large-scale training for almost a decade, yet the role of its two momentum parameters remains poorly understood. Recent work shows that tying $\beta_{1}=\beta_{2}$ can preserve Adam's strong performance despite collapsing two memory scales into one, raising a basic question: what becomes dynamically special when the memories are tied? We identify a concrete mechanism. In the continuous-time limit, each normalized-update coordinate decomposes into a sign component, an explicit magnitude-lag term proportional to the difference between the two memory times, and additional transition, curvature, and nonlinear ratio terms. This lag channel vanishes exactly when $\beta_{1}=\beta_{2}$, making the diagonal the unique regime in which this mismatch-induced response is structurally absent. A full-history discrete decomposition on real training gradients recovers this change in composition: tied updates are sig
    
[^271]: L2R：面向混合专家模型的低秩与Lipschitz受控路由

    L2R: Low-Rank and Lipschitz-Controlled Routing for Mixture-of-Experts

    [https://arxiv.org/abs/2601.21349](https://arxiv.org/abs/2601.21349)

    提出L2R统一路由框架，通过在共享低秩潜在路由空间中进行专家分配，并引入饱和内积评分（SIPS）显式控制路由函数的Lipschitz行为，重塑MoE的路由空间与评分几何，从而提升路由可区分性与专家专业化的稳定性。

    

    混合专家模型通过有条件地激活一小部分专家来扩展神经网络，其中路由器在决定专家专业化程度和整体模型性能方面起着核心作用。然而，许多现代MoE系统仍然在原始高维表示空间中采用线性路由器，在此情况下，表示不匹配、角度集中以及尺度敏感的评分会共同削弱路由的可区分性和专家专业化的稳定性。在本工作中，我们提出了低秩与Lipschitz受控路由（L2R），这是一个同时重塑路由空间和评分几何结构的统一路由框架。L2R在共享的低秩潜在路由空间中执行专家分配，并引入饱和内积评分（SIPS）来显式控制路由函数的Lipschitz行为，从而产生更平滑、更稳定的路由几何结构。此外，L2R还引入了一种参数高效的

    arXiv:2601.21349v3 Announce Type: replace-cross  Abstract: Mixture-of-Experts (MoE) models scale neural networks by conditionally activating a small subset of experts, where the router plays a central role in determining expert specialization and overall model performance. However, many modern MoE systems still adopt linear routers in raw high-dimensional representation spaces, where representation mismatch, angular concentration, and scale-sensitive scoring can jointly undermine routing discriminability and stable expert specialization. In this work, we propose Low-rank & Lipschitz-controlled Routing (L2R), a unified routing framework that reshapes both the routing space and scoring geometry. L2R performs expert assignment in a shared low-rank latent routing space and introduces Saturated Inner-Product Scoring (SIPS) to explicitly control the Lipschitz behavior of routing functions, yielding smoother and more stable routing geometry. In addition, L2R incorporates a parameter-efficient
    
[^272]: 环境数据循环：用于数据集精炼的生成模型

    Ambient Dataloops: Generative Models for Dataset Refinement

    [https://arxiv.org/abs/2601.15417](https://arxiv.org/abs/2601.15417)

    提出了 Ambient Dataloops 迭代框架，通过数据集与模型的协同演化逐步提升数据质量，并借助 Ambient Diffusion 技术避免自消耗循环，在图像生成和从头蛋白质设计中取得最先进性能。

    

    我们提出了 Ambient Dataloops，这是一个用于精炼数据集的迭代框架，使扩散模型更容易学习底层数据分布。现代数据集包含质量差异很大的样本，直接在此类异构数据上训练往往产生次优模型。我们提出了一种数据集-模型协同演化过程；在方法的每次迭代中，数据集的质量逐步提高，模型也随之改进。为了避免破坏性的自消耗循环，在每一代中，我们将合成改进的样本视为有噪声的样本，但其噪声水平略低于上一次迭代，并使用 Ambient Diffusion 技术在数据损坏的情况下进行学习。实验表明，Ambient Dataloops 在无条件图像生成、文本条件图像生成和从头蛋白质设计方面均达到了最先进的性能。我们还为所提出的框架提供了理论依据。

    arXiv:2601.15417v2 Announce Type: replace-cross  Abstract: We propose Ambient Dataloops, an iterative framework for refining datasets that makes it easier for diffusion models to learn the underlying data distribution. Modern datasets contain samples of highly varying quality, and training directly on such heterogeneous data often yields suboptimal models. We propose a dataset-model co-evolution process; at each iteration of our method, the dataset becomes progressively higher quality, and the model improves accordingly. To avoid destructive self-consuming loops, at each generation, we treat the synthetically improved samples as noisy, but at a slightly lower noisy level than the previous iteration, and we use Ambient Diffusion techniques for learning under corruption. Empirically, Ambient Dataloops achieve state-of-the-art performance in unconditional and text-conditional image generation and de novo protein design. We further provide a theoretical justification for the proposed frame
    
[^273]: SpaRRTa：评估视觉基础模型空间智能的合成基准

    SpaRRTa: A Synthetic Benchmark for Evaluating Spatial Intelligence in Visual Foundation Models

    [https://arxiv.org/abs/2601.11729](https://arxiv.org/abs/2601.11729)

    本文提出了SpaRRTa基准，通过合成图像评估视觉基础模型的空间关系识别能力，以区分真正的空间意识与对特定3D任务的过拟合。

    

    视觉基础模型（VFMs），如DINO和CLIP，在图像的语义理解方面表现出色，但展现出有限的空间推理能力，这限制了它们在具身系统中的应用。因此，近期工作将一些3D任务（如深度估计）纳入VFM训练中。然而，VFM在其他空间任务上的性能仍不一致，这引发了一个问题：这些模型是否真正具备空间意识，还是过度拟合了特定的3D目标。为解决这一问题，我们引入了空间关系识别任务（SpaRRTa）基准，该基准评估VFM识别图像中物体相对位置的能力。与传统3D目标（如表面法线估计）侧重于精确度量预测不同，SpaRRTa探究的是支撑更高级类人空间理解的基础能力。SpaRRTa可生成任意数量的照片级真实图像。

    arXiv:2601.11729v2 Announce Type: replace-cross  Abstract: Visual Foundation Models (VFMs), such as DINO and CLIP, excel in semantic understanding of images but exhibit limited spatial reasoning capabilities, which limits their applicability to embodied systems. As a result, recent work incorporates some 3D tasks (such as depth estimation) into VFM training. However, VFM performance remains inconsistent across other spatial tasks, raising the question of whether these models truly have spatial awareness or overfit to specific 3D objectives. To address this question, we introduce the Spatial Relation Recognition Task (SpaRRTa) benchmark, which evaluates the ability of VFMs to identify relative positions of objects in the image. Unlike traditional 3D objectives that focus on precise metric prediction (e.g., surface normal estimation), SpaRRTa probes a fundamental capability underpinning more advanced forms of human-like spatial understanding. SpaRRTa generates an arbitrary number of phot
    
[^274]: 基于上下文赌博机的强化学习实现线性求解器的精度自动调优

    Precision autotuning for linear solvers via contextual bandit-based RL

    [https://arxiv.org/abs/2601.00728](https://arxiv.org/abs/2601.00728)

    提出了一种基于上下文赌博机的强化学习框架，通过离散化特征和增量动作值估计为线性求解器动态选择最优精度配置，在迭代精化中实现精度与计算效率的平衡。

    

    我们提出了一种用于线性求解器精度调优的强化学习（RL）框架，该框架可扩展到一般算法。该框架被表述为一个上下文赌博机问题，并通过增量动作值估计和离散化状态空间来求解，以便为计算步骤选择最优的精度配置，同时保持精度和计算效率。为验证其有效性，我们将该框架应用于求解线性系统 $Ax = b$ 的迭代精化方法。在该应用中，我们的方法基于从系统计算出的特征动态选择精度，同时保持可接受的精度和收敛性。具体而言，一个动作值估计器以离散化特征（如近似条件数和矩阵范数）为输入，输出估计的动作值，策略从中选择动作（为特定步骤选择的精度配置）。

    arXiv:2601.00728v5 Announce Type: replace  Abstract: We propose a reinforcement learning (RL) framework for \xy{responsive} precision tuning for linear solvers, which can be extended to general algorithms. The framework is formulated as a contextual bandit problem and solved using incremental action-value estimation with a discretized state space to select optimal precision configurations for computational steps, \xy{retaining} precision and computational efficiency. To verify its effectiveness, we apply the framework to iterative refinement for solving linear systems $Ax = b$. In this application, our approach dynamically chooses precisions based on calculated features from the system while maintaining acceptable accuracy and convergence. In detail, an action-value estimator takes discretized features (e.g., approximate condition number and matrix norm) as input and outputs estimated action values, from which a policy selects the actions (chosen precision configurations for specific s
    
[^275]: Poodle：通过即时模型替换无缝缩减大型语言模型规模

    Poodle: Seamlessly Scaling Down Large Language Models with Just-in-Time Model Replacement

    [https://arxiv.org/abs/2512.05525](https://arxiv.org/abs/2512.05525)

    本文提出即时模型替换（JITR）的愿景，在检测到LLM调用中的重复任务时透明地将模型替换为针对该任务优化的更廉价的小模型，在保持LLM易用性的同时显著节省成本和能源。

    

    企业越来越依赖大型语言模型（LLM）来自动化简单重复的任务，而非开发定制的机器学习模型。LLM几乎不需要训练样本，即使没有模型开发专业知识的用户也能使用。然而，与较小的模型相比，这带来了显著更高的资源和能源消耗，而较小模型在简单任务上通常能实现相近的预测性能。在本文中，我们提出了即时模型替换（JITR）的愿景：当在LLM调用中识别出重复出现的任务时，系统会透明地将模型替换为在该特定任务上表现良好的更廉价的替代模型。JITR既保留了LLM的易用性和低开发成本，又能节省大量成本和能源。我们还讨论了实现这一愿景所面临的主要挑战，包括重复任务的识别和定制模型的创建。

    arXiv:2512.05525v3 Announce Type: replace-cross  Abstract: Businesses increasingly rely on large language models (LLMs) to automate simple repetitive tasks instead of developing custom machine learning models. LLMs require few, if any, training examples and can be utilized by users without expertise in model development. However, this comes at the cost of substantially higher resource and energy consumption compared to smaller models, which often achieve similar predictive performance for simple tasks. In this paper, we present our vision for just-in-time model replacement (JITR), where, upon identifying a recurring task in calls to an LLM, the model is replaced transparently with a cheaper alternative that performs well for this specific task. JITR retains the ease of use and low development effort of LLMs, while saving significant cost and energy. We discuss the main challenges in realizing our vision regarding the identification of recurring tasks and the creation of a custom model.
    
[^276]: 多项选择视觉问答中未被探索的缺陷使基准测试不可靠

    Unexplored flaws in multiple-choice VQA make benchmarking unreliable

    [https://arxiv.org/abs/2511.22341](https://arxiv.org/abs/2511.22341)

    该论文揭示了MC-VQA基准测试中存在严重的提示格式敏感性问题——即使消除了选项顺序的影响，仅仅改变选项ID、分隔符等语义无关的格式元素就会导致模型性能排名频繁逆转，这表明当前MC-VQA作为多模态大语言模型评估基准的可靠性存在根本性缺陷。

    

    先前的研究指出对选项顺序的敏感性是多项选择视觉问答（MC-VQA）评估中的一个关键问题，并提出了一些协议来缓解这种影响。我们证明这种缓解措施不足以确保MC-VQA作为多模态大语言模型（MLLMs）可靠基准的有效性：性能对语义中性的提示格式选择仍然高度敏感，而当前的基准测试并未控制这些格式选择。在一项涵盖七个MLLM和五个MC-VQA数据集的大规模研究中，我们发现即使在顺序不变的评估条件下，排名逆转仍然频繁发生。当我们系统地改变选项ID集合、分隔符和分离符，从而产生48种语义等价的提示格式时，这些排名逆转就会出现。机制分析将这种不稳定性追溯到低层次的语言建模效应：分词器引起的选项ID标记的融合或移除会在输入序列中引入损坏的选项ID标记，而选择……

    arXiv:2511.22341v2 Announce Type: replace-cross  Abstract: Previous works identify sensitivity to option order as a key issue in multiple-choice VQA (MC-VQA) evaluation and propose protocols to mitigate this effect. We show that such mitigation is insufficient to ensure the validity of MC-VQA as a reliable benchmark for Multimodal Large Language Model (MLLMs): performance remains highly sensitive to semantically neutral prompt format choices that are not controlled by current benchmarks. In a large-scale study spanning seven MLLMs and five MC-VQAs datasets, we find frequent rank reversals even under order-invariant evaluation. These reversals arise when we systematically vary option ID sets, delimiters, and separators, yielding 48 semantically equivalent prompt formats. Mechanistic analyses trace this instability to low-level language modeling effects: tokenizer-induced fusion or removal of option ID tokens introduces corrupted option ID tokens into the input sequence, while the choice
    
[^277]: 预训练获益：无需干净标签的鲁棒学习

    Pre-train to Gain: Robust Learning Without Clean Labels

    [https://arxiv.org/abs/2511.20844](https://arxiv.org/abs/2511.20844)

    该论文提出先在目标数据集上进行域内自监督预训练、再进行标准监督训练的方法，无需任何干净标签子集即可获得对标签噪声更鲁棒的模型。

    

    使用噪声标签训练深度网络会因对标签噪声过拟合而导致泛化能力差和准确率下降。现有的噪声标签学习方法通常依赖于干净数据子集的可用性。通过使用域内自监督学习（SSL）在无标签的目标数据集上预训练特征提取器，然后在同一噪声数据集上进行标准监督训练，我们可以在不需要干净标签子集的情况下训练出更具噪声鲁棒性的模型。我们在具有合成标签噪声和真实世界标签噪声的数据集上评估了对比式和非对比式SSL预训练方法，证明了该方法在大规模数据集、多样化下游任务和模型架构上的广泛适用性。在所有噪声率下，域内自监督预训练都能持续提升分类准确率和下游标签错误检测（F1和平衡准确率）的性能。

    arXiv:2511.20844v2 Announce Type: replace-cross  Abstract: Training deep networks with noisy labels leads to poor generalization and degraded accuracy due to overfitting to label noise. Existing approaches for learning with noisy labels often rely on the availability of a clean subset of data. By pre-training a feature extractor on the target dataset without labels using in-domain self-supervised learning (SSL), followed by standard supervised training on the same noisy dataset, we can train a more noise robust model without requiring a subset with clean labels. We evaluate both contrastive and non-contrastive SSL pre-training methods across datasets with synthetic and real-world label noise, demonstrating the broad applicability of our approach across large-scale datasets, diverse downstream tasks, and model architectures. Across all noise rates, in-domain self-supervised pre-training consistently improves classification accuracy and downstream label-error detection (F1 and Balanced A
    
[^278]: Learn2Drive：一种基于神经网络的社会兼容自动驾驶车辆控制框架

    Learn2Drive: A neural network-based framework for socially compliant automated vehicle control

    [https://arxiv.org/abs/2510.21736](https://arxiv.org/abs/2510.21736)

    该论文提出了一种融合社会价值取向的神经网络自适应巡航控制框架，使自动驾驶车辆能够兼顾对人类驾驶车辆和交通流的影响，充当移动交通调节器以缓解拥堵、提升整体交通效率。

    

    本研究提出了一种新颖的自动驾驶自适应巡航控制（ACC）控制框架，该框架利用神经网络和物理信息约束。随着自动驾驶车辆（AV）逐步采用自适应巡航控制等先进功能，交通系统正变得越来越智能和高效。然而，现有的自动驾驶车辆控制策略主要专注于优化单个车辆或车队的性能，往往忽略了它们与人类驾驶车辆（HV）的交互以及对交通流的更广泛影响。这种疏忽可能会加剧交通拥堵并降低整体系统效率。为解决这一关键研究空白，我们提出了一种基于神经网络、融合社会价值取向（SVO）的社会兼容自动驾驶车辆控制框架。该框架使自动驾驶车辆能够考虑其对人类驾驶车辆和交通动态的影响。通过将自动驾驶车辆用作移动交通调节器，所提出的方法促进了……

    arXiv:2510.21736v2 Announce Type: replace-cross  Abstract: This study introduces a novel control framework for adaptive cruise control (ACC) in automated driving, leveraging neural networks and physics-informed constraints. As automated vehicles (AVs) adopt advanced features like ACC, transportation systems are becoming increasingly intelligent and efficient. However, existing AV control strategies primarily focus on optimizing the performance of individual vehicles or platoons, often neglecting their interactions with human-driven vehicles (HVs) and the broader impact on traffic flow. This oversight can exacerbate congestion and reduce overall system efficiency. To address this critical research gap, we propose a neural network-based, socially compliant AV control framework that incorporates social value orientation (SVO). This framework enables AVs to account for their influence on HVs and traffic dynamics. By leveraging AVs as mobile traffic regulators, the proposed approach promote
    
[^279]: SGM：一种用于风险可控递归自我修改的统计哥德尔机

    SGM: A Statistical Godel Machine for Risk-Controlled Recursive Self-Modification

    [https://arxiv.org/abs/2510.10232](https://arxiv.org/abs/2510.10232)

    本文提出了首个针对递归自我修改的统计安全层——统计哥德尔机（SGM），用统计置信度检验（e值、Hoeffding界）替代无法在随机高维环境中实现的形式化证明要求，并通过全局误差预算和确认触发的调和支出机制（CTHS）实现累积风险的可控性。

    

    递归自我修改在自动化机器学习（AutoML）、神经架构搜索和自适应优化中日益成为核心，然而现有框架都无法确保此类修改的安全性。哥德尔机通过要求在重写代码前提供形式化的改进证明来提供原则性的安全保障；然而，在随机、高维环境中，这种证明是无法实现的。我们提出了统计哥德尔机（SGM），这是首个针对递归编辑的统计安全层。SGM用统计置信度检验（e值、Hoeffding界）取代基于证明的要求，只有当在选定置信度水平下证明了优越性时才允许修改，同时分配全局误差预算以约束各轮次的累积风险。我们还提出了确认触发的调和支出机制（CTHS），它以确认事件而非轮次为索引来分配支出，将误差预算集中在有前景的修改上，同时保持……

    arXiv:2510.10232v2 Announce Type: replace-cross  Abstract: Recursive self-modification is increasingly central in AutoML, neural architecture search, and adaptive optimization, yet no existing framework ensures that such changes are made safely. Godel machines offer a principled safeguard by requiring formal proofs of improvement before rewriting code; however, such proofs are unattainable in stochastic, high-dimensional settings. We introduce the Statistical Godel Machine (SGM), the first statistical safety layer for recursive edits. SGM replaces proof-based requirements with statistical confidence tests (e-values, Hoeffding bounds), admitting a modification only when superiority is certified at a chosen confidence level, while allocating a global error budget to bound cumulative risk across rounds.We also propose Confirm-Triggered Harmonic Spending (CTHS), which indexes spending by confirmation events rather than rounds, concentrating the error budget on promising edits while preserv
    
[^280]: 语言模型训练的环境影响持续攀升：现在是时候关注回弹效应中的影响了

    The Environmental Impacts of Language Model Training Keep Rising Now is the Time to Catch Impacts on the Rebound

    [https://arxiv.org/abs/2510.09022](https://arxiv.org/abs/2510.09022)

    研究发现，尽管采用了硬件、算法和碳排优化等减耗策略，过去十年语言模型训练的能源消耗和环境影响仍呈指数级增长，表明存在回弹效应，且硬件的影响必须在全生命周期内评估。

    

    近期的机器学习方法在基准测试中展现出不断提升的性能，但代价是计算需求不断攀升。为遏制能源消耗和环境影响，人们已提出硬件、算法和碳排方面的优化方案。我们估算了过去十年中Epoch AI数据库所记录模型训练相关的环境影响，并特别关注与大语言模型及其训练所用硬件相关的影响。我们发现，即使考虑了使用低碳电力组合或更高效硬件等影响削减策略，训练机器学习模型所产生的能源消耗和环境影响仍呈指数级增长。优化策略并未缓解模型训练带来的影响，这表明存在回弹效应。我们还指出，必须在硬件的整个生命周期内而非仅在使用阶段来评估其影响。

    arXiv:2510.09022v2 Announce Type: replace  Abstract: Recent Machine Learning (ML) approaches have shown increased performance on benchmarks at the cost of escalating compute demands. Hardware, algorithmic and carbon optimizations have been proposed to curb energy use and environmental impacts. We estimate the environmental impacts associated with training models documented in the Epoch AI database over the last decade, with a particular focus on impacts associated with Large Language Models and the hardware used to train them. We find that energy use and environmental impacts associated with training ML models have increased exponentially, even when considering impact reduction strategies such as using less carbon intensive electricity mixes or more efficient hardware. Optimization strategies do not mitigate the impacts induced by model training, suggesting rebound effect. We show that the impacts of hardware must be considered over the entire life cycle rather than the sole use phase 
    
[^281]: 扩散语言模型的水印技术

    Watermarking Diffusion Language Models

    [https://arxiv.org/abs/2509.24368](https://arxiv.org/abs/2509.24368)

    本文提出了首个专为扩散语言模型设计的水印技术，通过在期望意义上应用水印并促进增强水印强度的词元生成，在保持检测器不变的前提下实现了超过99%的真阳性率且对生成质量影响极小。

    

    我们提出了首个专为扩散语言模型（DLMs）设计的水印技术，这是一种新兴的大语言模型范式，能够以任意顺序生成词元，与按顺序生成词元的标准自回归语言模型（ARLMs）形成对比。尽管针对ARLM的水印技术已有大量研究，但将这些方案直接应用于DLM场景的一个关键挑战在于，它们依赖于先前生成的词元，而这些词元在DLM生成过程中并不总是可用的。在本工作中，我们通过以下方式应对这一挑战：（i）即使部分上下文词元尚未确定，也在期望意义上将水印应用于整个上下文；（ii）促进那些在被用作其他词元的上下文时能够增强水印强度的词元的生成。这一切都是在保持水印检测器不变的情况下实现的。我们的实验评估表明，该DLM水印技术能够实现超过99%的真阳性率，且对生成质量的影响极小。

    arXiv:2509.24368v3 Announce Type: replace-cross  Abstract: We introduce the first watermark tailored for diffusion language models (DLMs), an emergent LLM paradigm able to generate tokens in arbitrary order, in contrast to standard autoregressive language models (ARLMs) which generate tokens sequentially. While there has been much work in ARLM watermarking, a key challenge when attempting to apply these schemes directly to the DLM setting is that they rely on previously generated tokens, which are not always available with DLM generation. In this work we address this challenge by: (i) applying the watermark in expectation over the context even when some context tokens are yet to be determined, and (ii) promoting tokens which increase the watermark strength when used as context for other tokens. This is accomplished while keeping the watermark detector unchanged. Our experimental evaluation demonstrates that the DLM watermark leads to a >99% true positive rate with minimal quality impac
    
[^282]: 面向大规模矩阵优化的低秩正交化及其在基础模型训练中的应用

    Low-rank Orthogonalization for Large-scale Matrix Optimization with Applications to Foundation Model Training

    [https://arxiv.org/abs/2509.11983](https://arxiv.org/abs/2509.11983)

    该论文提出了利用神经网络训练中梯度低秩特性的低秩正交化方法，并由此发展出低秩矩阵符号梯度下降和低秩Muon优化器，为大规模基础模型训练提供了高效的矩阵优化方案。

    

    摘要（arXiv:2509.11983v3，公告类型：替换）：神经网络训练本质上是一个大规模矩阵优化问题，然而神经网络参数的矩阵结构长期以来被忽视。近来，显式利用这一结构的优化器Muon因其在基础模型训练中的强劲表现而受到广泛关注。促成Muon成功的一个关键组件是矩阵正交化。在本文中，我们提出了低秩正交化方法，通过利用神经网络训练过程中梯度的低秩特性来执行正交化。基于此，我们引入了低秩矩阵符号梯度下降（MSGD）以及Muon的低秩变体。数值实验……

    arXiv:2509.11983v3 Announce Type: replace  Abstract: Neural network (NN) training is inherently a large-scale matrix optimization problem, yet the matrix structure of NN parameters has long been overlooked. Recently, the optimizer Muon \citep{jordanmuon}, which explicitly exploits this structure, has gained significant attention for its strong performance in foundation model training. A key component contributing to Muon's success is matrix orthogonalization. In this paper, we propose \textit{low-rank orthogonalization}, which performs orthogonalization by leveraging the low-rank nature of gradients during NN training. Building on this, we introduce low-rank matrix-signed gradient descent (MSGD) and a low-rank variant of Muon. %Numerical experiments demonstrate the superior performance of low-rank orthogonalization, with low-rank Muon achieving promising results in GPT-2 and LLaMA pretraining---surpassing the carefully tuned vanilla Muon on tasks with large model sizes. {Numerical expe
    
[^283]: 评估基于图的Android恶意软件分类中的分布外鲁棒性：一个新的原则性基准

    Evaluating Out-of-Distribution Robustness in Graph-Based Android Malware Classification: A New Principled Benchmark

    [https://arxiv.org/abs/2508.06734](https://arxiv.org/abs/2508.06734)

    该论文提出了一个新的基准测试套件来评估基于图的Android恶意软件分类器在协变量偏移和域偏移下的分布外鲁棒性，发现其性能下降高达45%，并提出通过结合轻量级元数据与LLM代码嵌入的语义增强框架来解决现有基准仅依赖图结构的局限性。

    

    虽然基于图的Android恶意软件分类器报告了超过94%的强大基准准确率，但当面对已知恶意软件家族的未见变体时，其性能会急剧下降高达45%。在这项工作中，我们通过引入一个基准测试套件来系统地研究这一对现实世界部署至关重要却被忽视的挑战，该套件旨在模拟两种普遍场景：用于协变量偏移的MalNet-Tiny-Common，以及用于域偏移的MalNet-Tiny-Distinct。我们进一步识别了现有基准的一个固有局限性，即输入表示仅限于纯结构的函数调用图，从而丢弃了鲁棒跨分布推理所需的语义信号。为了验证这一点，我们提出了一个语义增强框架，通过函数级属性扩展原始图拓扑，将轻量级元数据与基于大语言模型（LLM）的代码嵌入相结合。实证评估证实了该方法的有效性。

    arXiv:2508.06734v3 Announce Type: replace-cross  Abstract: While graph-based Android malware classifiers report strong benchmark accuracy of over 94%, their performance sharply decreases up to 45% when exposed to previously unseen variants of known malware families. In this work, we systematically investigate this critical yet overlooked challenge for real-world deployment by introducing a benchmarking suite designed to simulate two prevalent scenarios: MalNet-Tiny-Common for covariate shift, and MalNet-Tiny-Distinct for domain shift. We further identify an inherent limitation of existing benchmarks where input representation is limited to structure-only function call graphs, discarding the semantic signals needed for robust cross-distribution reasoning. To verify this, we propose a semantic enrichment framework that extends raw graph topology with function-level attributes, combining lightweight metadata with LLM-based code embeddings. Empirical evaluations confirm the effectiveness o
    
[^284]: 神经朗之万机：一种局部非对称学习规则可以具有创造性

    Neural Langevin Machine: a local asymmetric learning rule can be creative

    [https://arxiv.org/abs/2506.23546](https://arxiv.org/abs/2506.23546)

    该论文提出“神经朗之万机”这一生成模型，其核心创新在于仅依赖局部神经信号的非对称、发放速率调整学习规则，既具有生物学合理性，又能实现创造性图像生成、去噪以及从记忆到泛化的转变。

    

    循环神经网络的不动点可以被用来存储和生成信息。这些不动点由玻尔兹曼-吉布斯测度所刻画，由此导出的神经朗之万动力学能够弛豫到这些不动点，从而实现对真实数据集的生成式学习。我们将这种类型的生成模型称为神经朗之万机，它推导出一种非对称的、经神经发放速率速度调整的学习规则，该规则仅需要局部神经信号，因此在局部预测学习方面具有生物学意义。研究揭示了生成过程中的非平衡状态，以及随训练数据规模增加而出现的从记忆到泛化的转变。这种受神经科学启发的机器还能够对不同类型的生成图像实现相空间的连续探索，并且可以对受损图像进行去噪。

    arXiv:2506.23546v3 Announce Type: replace-cross  Abstract: Fixed points of recurrent neural networks can be leveraged to store and generate information. These fixed points are captured by the Boltzmann-Gibbs measure, which leads to neural Langevin dynamics that relax to those fixed points for generative learning of a real dataset. We call this type of generative model a neural Langevin machine, which derives an asymmetric and firing-rate-speed-adjusted learning rule requiring only local neural signals, thereby bearing biological relevance in terms of local predictive learning. An out-of-equilibrium regime of the generative process is revealed, together with a memorization-to-generalization transition with increasing training data size. The neuro-inspired machine can also realize a continuous exploration of the phase space for different kinds of generative images and can denoise a corrupted image as well.
    
[^285]: 球面柯西变分自编码器：重角尾与精确KL散度计算

    Spherical Cauchy Variational Autoencoders: Heavy Angular Tails and Exact KL Evaluation

    [https://arxiv.org/abs/2506.21278](https://arxiv.org/abs/2506.21278)

    提出球面柯西分布作为超球面变分自编码器的后验分布，兼具重角尾特性和精确的KL散度解析计算能力，克服了von Mises-Fisher分布需要拒绝采样和Power Spherical分布密度在对跖点归零的缺陷。

    

    重尾后验在欧氏变分自编码器中十分常见，其中Student族无需额外机制即可放宽高斯假设。然而，球面上一直缺乏可与之媲美的选择。von Mises-Fisher分布需要修正贝塞尔函数和拒绝采样器，而Power Spherical分布则通过强制密度在对跖点处归零来换取封闭形式的表达。我们开发了球面柯西分布作为一种超球面后验分布，无需做出上述任何一种妥协。通过球极投影，该分布可映射为多元Student分布；借助一个默比乌斯变换，可以将均匀球面采样转化为基于内积、范数和标量运算的精确后验样本。同一变换也解决了正则化项的计算问题。沿着采样映射评估密度，将相对于均匀先验的KL散度简化为一个标量期望，其展开式在每个偶数维环境空间中均能终止，仅剩下一个对数积分需要计算。

    arXiv:2506.21278v4 Announce Type: replace-cross  Abstract: Heavy-tailed posteriors are routine in Euclidean variational autoencoders, where the Student family relaxes the Gaussian without new machinery. The sphere has had no comparable option. Von Mises-Fisher distribution needs modified Bessel functions and a rejection sampler, and Power Spherical buys its closed forms by forcing the density to vanish at the antipode. We develop the spherical Cauchy distribution as a hyperspherical posterior that needs neither compromise. Stereographic projection carries it to a multivariate Student law, and a M\"obius transformation turns a uniform spherical draw into an exact posterior sample from inner products, norms, and scalar arithmetic. The same transformation settles the regularizer. Evaluating the density along the sampling map reduces the Kullback-Leibler (KL) divergence to the uniform prior to a scalar expectation whose expansion terminates in every even ambient dimension, leaving one loga
    
[^286]: 实现对真实世界环境中以儿童为中心的音频记录的自动转录

    Enabling automatic transcription of child-centered audio recordings from real-world environments

    [https://arxiv.org/abs/2506.11747](https://arxiv.org/abs/2506.11747)

    该论文提出一种自动检测长时间儿童音频录音中可被可靠转录语音片段的方法，突破以往ASR需完整处理录音的限制，实现了对真实环境录音中大部分语音的自动准确转录。

    

    通过儿童佩戴麦克风获得的长时间音频录音——也称为以儿童为中心的全天录音——已成为研究儿童语言体验及其对后续语言发展影响的标准方法。对长时间语音音频进行转录可以在多个语言层面开展丰富的分析，然而典型长时间语料库的庞大规模使得全面的人工标注无法实现。同时，由于真实世界音频的嘈杂和无约束特性，基于自动语音识别（ASR）的转录面临重大挑战。以往的研究都假设ASR必须完整处理每条长时间录音。在这项工作中，我们提出了一种方法，可以自动检测长时间音频中那些能够被现代ASR系统可靠转录的话语，从而实现对典型（录音中）相当大比例语音的自动且相对准确的转录

    arXiv:2506.11747v2 Announce Type: replace-cross  Abstract: Longform audio recordings obtained with microphones worn by children-also known as child-centered daylong recordings-have become a standard method for studying children's language experiences and their impact on subsequent language development. Transcripts of longform speech audio would enable rich analyses at various linguistic levels, yet the massive scale of typical longform corpora prohibits comprehensive manual annotation. Meanwhile, automatic speech recognition (ASR)-based transcription faces significant challenges due to the noisy, unconstrained nature of real-world audio. Previous attempts have assumed that ASR must process each longform recording in its entirety. In this work, we present an approach to automatically detect those utterances in longform audio that can be reliably transcribed with modern ASR systems, allowing automatic and relatively accurate transcription of a notable proportion of all speech in typical 
    
[^287]: 基于网络科学的时间序列细粒度分割方法

    A Network Science Approach to Granular Time Series Segmentation

    [https://arxiv.org/abs/2505.17640](https://arxiv.org/abs/2505.17640)

    该论文提出将时间序列细粒度分割形式化为图节点分类问题，通过可见性图变换结合图注意力网络（GAT）实现，其中WDPVG+GAT在59个数据集的基准上达到0.916的加权F1分数，且WDPVG、有向NVG和加权NVG三种图构建方法表现最佳且统计上不可区分。

    

    时间序列分割是为序列的每个部分分配标签的任务。我们将密集单变量分割形式化为图上的节点分类问题，其中图的节点为原始时间点。局部窗口提供节点特征，而无需设定输出粒度。我们在一个由互不相交的UCR训练和测试实例构建的TSSB衍生归纳基准上评估了该方法。在一个固定的图注意力网络（GAT）下，基于可见性的变换在十一种图构建方法中取得了最高的平均排名。经Holm校正后，WDPVG、有向NVG和加权NVG构成了统计上不可区分的顶级组。在包含59个数据集的时间序列分割基准上，WDPVG+GAT达到了0.916的加权F1分数，低于seq2point的0.951，但与使用相同特征的MLP、随机森林和1-NN对照方法在统计上不可区分，原因是在这种降采样分辨率下每个分割段较短，而固定的81个样本窗口……

    arXiv:2505.17640v3 Announce Type: replace  Abstract: Time series segmentation assigns a label to each part of a sequence. We formulate dense univariate segmentation as node classification on a graph whose nodes are the original time points. A local window provides node features without setting output granularity. We evaluate the approach on a TSSB-derived inductive benchmark built from disjoint UCR training and test instances. Under one fixed Graph Attention Network (GAT), visibility-based transformations achieve the highest mean ranks among eleven graph constructions. WDPVG, directed NVG, and weighted NVG form a statistically indistinguishable top group after Holm correction. On the 59-dataset Time Series Segmentation Benchmark, WDPVG+GAT reaches a weighted F1 of $0.916$, below seq2point at $0.951$ and statistically indistinguishable from same-feature MLP, random-forest, and 1-NN controls, because at this downsampled resolution each segment is short and the fixed $81$-sample window al
    
[^288]: FOCAL：基于细粒度最优传输的语言与心电图对比对齐及波形增强

    FOCAL: Fine-Grained Optimal-Transport-Driven Contrastive Alignment of Language and ECGs with Waveform Enhancement

    [https://arxiv.org/abs/2505.11939](https://arxiv.org/abs/2505.11939)

    该论文提出FOCAL框架，通过最优传输实现心电图局部波形片段与报告病理标签的细粒度精确对齐，并利用语义相似度矩阵缓解标签级对齐中的假阴性问题，从而提升零样本心电图解读性能。

    

    心电图（ECG）是诊断心血管疾病的重要无创工具。尽管近期的多模态心电图-报告对比学习方法在零样本心电图解读方面展现出了前景，但它们主要依赖于全局表示，未能捕捉局部波形片段与特定病理标签之间的细粒度关系。由于近55%的标准临床报告（例如MIMIC-ECG中的报告）缺乏明确的波形描述，这一局限性进一步加剧。在本文中，我们提出了FOCAL，这是一个新颖的框架，通过最优传输实现局部心电片段与单个报告标签之间的精确细粒度对齐。此外，由于标签层面的细粒度对齐加剧了共享常见诊断的报告之间的假阴性问题，我们引入了语义相似度矩阵来指导对比目标……（摘要原文在此处截断）

    arXiv:2505.11939v3 Announce Type: replace-cross  Abstract: Electrocardiograms (ECGs) are essential non-invasive tools for diagnosing cardiovascular diseases. While recent multimodal ECG-Report contrastive learning methods have shown promise for zero-shot ECG interpretation, they predominantly rely on global representations, failing to capture the fine-grained relationship between localized waveform patches and specific pathological tags. This limitation is further exacerbated by the fact that nearly 55% of standard clinical reports (e.g., in MIMIC-ECG) lack explicit waveform descriptions. In this paper, we propose FOCAL, a novel framework that achieves precise, fine-grained alignment between localized ECG segments and individual report tags via Optimal Transport. Furthermore, because fine-grained alignment at the tag level exacerbates the false negative problem among reports sharing common diagnoses, we introduce a semantic similarity matrix to guide the contrastive objective and corre
    
[^289]: 基于邻近性数据的样本外嵌入：投影与受限重构之比较

    Out-of-Sample Embedding with Proximity Data: Projection versus Restricted Reconstruction

    [https://arxiv.org/abs/2505.06756](https://arxiv.org/abs/2505.06756)

    本文综述了基于邻近性数据的样本外嵌入的各种核方法，证明它们均可归结为投影或受限重构这两种基本策略之一，其中受限重构策略可简化为只需一维搜索的非线性优化问题。

    

    利用邻近性（相似性或相异性）数据来实现“在向量图中添加一个点”的问题最早由 J.C. Gower 于1968年研究。此后，人们提出了许多方法——主要是核方法——来解决这一后来被称为*样本外嵌入*的问题。我们综述了我们所遇到的各种核方法，并证明其中每一种方法都可以从两种相互竞争的策略之一推导出来：*投影*或*受限重构*。投影可以类比为在主成分分析中添加一个点的著名公式。受限重构则提出了一个不同的挑战：如何在保持先前获得的向量图固定不变的情况下，最好地近似重新进行整个多变量分析。这一策略会产生一个非线性优化问题，该问题可以简化为一维搜索。

    arXiv:2505.06756v2 Announce Type: replace-cross  Abstract: The problem of using proximity (similarity or dissimilarity) data for the purpose of "adding a point to a vector diagram" was first studied by J.C. Gower in 1968. Since then, a number of methods -- mostly kernel methods -- have been proposed for solving what has come to be called the problem of *out-of-sample embedding*. We survey the various kernel methods that we have encountered and show that each can be derived from one or the other of two competing strategies: *projection* or *restricted reconstruction*. Projection can be analogized to a well-known formula for adding a point to a principal component analysis. Restricted reconstruction poses a different challenge: how to best approximate redoing the entire multivariate analysis while holding fixed the vector diagram that was previously obtained. This strategy results in a nonlinear optimization problem that can be simplified to a unidimensional search. Various circumstances
    
[^290]: 面向决策学习的充分决策代理

    Sufficient Decision Proxies for Decision-Focused Learning

    [https://arxiv.org/abs/2505.03953](https://arxiv.org/abs/2505.03953)

    本文首次研究了优化问题在何种性质下使用特定决策代理（单场景预测或参数化分布估计）才是合理的，并据此为决策聚焦学习提出了学习复杂度几乎不受影响的替代性决策代理。

    

    在利用上下文数据求解不确定性下的优化问题时，使用机器学习预测不确定参数的取值是一种流行且有效的方法。决策聚焦学习（DFL）旨在学习一个预测模型，使决策质量（而非预测精度）最大化。常见的做法是预测一个代表不确定参数的单个场景，这隐含地假设存在一个确定性的问题近似（代理），允许做出最优决策。相反的方法也有被考虑过，即用参数化分布来估计潜在的分布。然而，对于这两种选择何时有效，人们知之甚少。本文首次研究了能够证明使用某种决策代理合理性的问题属性。基于此，我们为决策聚焦学习提出了替代性的决策代理，而在学习复杂度上几乎或完全没有妥协。

    arXiv:2505.03953v3 Announce Type: replace  Abstract: When solving optimization problems under uncertainty with contextual data, utilizing machine learning to predict the uncertain parameters' values is a popular and effective approach. Decision-focused learning (DFL) aims at learning a predictive model such that decision quality, instead of prediction accuracy, is maximized. Common practice is to predict a single scenario representing the uncertain parameters, implicitly assuming that there exists a deterministic problem approximation (proxy) that allows for optimal decision-making. The opposite has also been considered, where the underlying distribution is estimated with a parameterized distribution. However, little is known about when either choice is valid. This paper investigates for the first time problem properties that justify using a certain decision proxy. Using this, we present alternative decision proxies for DFL, with little or no compromise on the complexity of the learnin
    
[^291]: R2DN：收缩性与利普希茨递归深度网络的可扩展参数化方法

    R2DN: Scalable Parameterization of Contracting and Lipschitz Recurrent Deep Networks

    [https://arxiv.org/abs/2504.01250](https://arxiv.org/abs/2504.01250)

    本文提出R2DN，通过将线性时不变系统与1-Lipschitz深度前馈网络反馈互联来直接参数化稳定且鲁棒的递归深度网络，无需像递归平衡网络（REN）那样迭代求解平衡层，从而大幅加快GPU上的推理与训练速度，并使扩展网络规模和输入序列长度在计算上切实可行。

    

    本文提出了鲁棒递归深度网络（R2DN），这是一种面向机器学习和数据驱动控制的、稳定且鲁棒的递归神经网络的可扩展参数化方法。我们将R2DN构建为一个线性时不变系统与一个1-Lipschitz深度前馈网络的反馈互联结构，并直接对权重进行参数化，使得模型在设计上即具备稳定性（收缩性）和对输入扰动的鲁棒性（Lipschitz性）。我们的参数化采用了类似于递归平衡网络（REN）的结构，但无需在每个时间步迭代求解平衡层。这加快了模型在GPU上的推理和训练速度，并使得与REN相比，扩大网络规模和输入序列长度在计算上成为可行。我们在非线性系统辨识、观测器设计、基于学习的反馈控制以及序列图像等代表性问题上，将R2DN与REN进行了比较。

    arXiv:2504.01250v3 Announce Type: replace  Abstract: This paper presents the Robust Recurrent Deep Network (R2DN), a scalable parameterization of stable and robust recurrent neural networks for machine learning and data-driven control. We construct R2DNs as the feedback interconnection of a linear time-invariant system and a 1-Lipschitz deep feedforward network, and directly parameterize the weights so that our models are stable (contracting) and robust to input perturbations (Lipschitz) by design. Our parameterization uses a structure similar to the recurrent equilibrium network (REN), but without having to iteratively solve an equilibrium layer at each time-step. This speeds up model inference and training on GPUs, and makes it computationally feasible to scale up the network size and input sequence length in comparison to RENs. We compare R2DNs to RENs on representative problems in nonlinear system identification, observer design, learning-based feedback control, and sequential imag
    
[^292]: 基于可泛化图神经网络的大规模网络流量工程

    Traffic Engineering in Large-scale Networks with Generalizable Graph Neural Networks

    [https://arxiv.org/abs/2503.24203](https://arxiv.org/abs/2503.24203)

    TELGEN通过将“预测最优流量工程解”转化为“预测最优流量工程算法”的新思路，学习高效逼近经典最优TE算法的端到端求解过程，从而在大规模网络中实现了高效求解与跨多种网络条件的卓越泛化能力。

    

    在云广域网（WAN）和低地球轨道（LEO）卫星星座等大规模网络中进行流量工程（TE）是一项关键挑战。尽管已有基于学习的方法被提出来解决传统TE算法的可扩展性问题，但其实际应用往往受到泛化能力不足、训练开销过高以及无法遵守链路容量限制等因素的阻碍。本文提出了TELGEN，一种新颖的TE算法，它能够学习在大规模网络场景中高效地求解TE问题，同时在多样化的网络条件下实现卓越的泛化能力。TELGEN基于一个新颖的思想，即将“预测最优TE解”的问题转化为“预测最优TE算法”的问题，这使得TELGEN能够学习并高效地逼近经典最优TE算法的端到端求解过程。所学习到的算法对具体的底层……（摘要原文在此处截断）

    arXiv:2503.24203v3 Announce Type: replace-cross  Abstract: Traffic Engineering (TE) in large-scale networks like cloud Wide Area Networks (WANs) and Low Earth Orbit (LEO) satellite constellations is a critical challenge. Although learning-based approaches have been proposed to address the scalability of traditional TE algorithms, their practical application is often hindered by a lack of generalization, high training overhead, and a failure to respect link capacities. This paper proposes TELGEN, a novel TE algorithm that learns to solve TE problems efficiently in large-scale network scenarios, while achieving superior generalizability across diverse network conditions. TELGEN is based on the novel idea of transforming the problem of "predicting the optimal TE solution" into "predicting the optimal TE algorithm", which enables TELGEN to learn and efficiently approximate the end-to-end solving process of classical optimal TE algorithms. The learned algorithm is agnostic to the exact unde
    
[^293]: 一个源自开放科学文献的大规模视觉-语言数据集，用于推动生物医学通用人工智能的发展

    A Large-Scale Vision-Language Dataset Derived from Open Scientific Literature to Advance Biomedical Generalist AI

    [https://arxiv.org/abs/2503.22727](https://arxiv.org/abs/2503.22727)

    该论文发布了源自PubMed Central开放获取文献的开源大规模多模态数据集Biomedica（含600万篇文章、2400万图像-文本对及专家标注），基于其训练的AI模型在嵌入、对话和检索等各任务类别中均超越了此前的开放系统。

    

    尽管生物医学人工智能（AI）备受关注，但获取高质量、多样化且大规模的数据——现代AI系统的基础——仍然是释放其全部潜力的瓶颈。为解决这一差距，我们推出了Biomedica，这是一个源自PubMed Central开放获取子集的开源数据集，包含超过600万篇科学文章和2400万对图像-文本，以及27个元数据字段（包括专家人工标注）。为克服访问大规模数据集的挑战，我们通过网络服务器提供了可扩展的流式传输和搜索API，便于与AI系统无缝集成。我们通过构建嵌入模型、聊天式模型和检索增强的聊天代理来展示Biomedica数据集的实用性。值得注意的是，我们所有的AI模型在各自类别中都超越了以往的开放系统，凸显了多样化、高质量数据的关键作用。

    arXiv:2503.22727v3 Announce Type: replace  Abstract: Despite the excitement behind biomedical artificial intelligence (AI), access to high-quality, diverse, and large-scale data - the foundation for modern AI systems - is still a bottleneck to unlocking its full potential. To address this gap, we introduce Biomedica, an open-source dataset derived from the PubMed Central Open Access subset, containing over 6 million scientific articles and 24 million image-text pairs, along with 27 metadata fields (including expert human annotations). To overcome the challenges of accessing our large-scale dataset, we provide scalable streaming and search APIs through a web server, facilitating seamless integration with AI systems. We demonstrate the utility of the Biomedica dataset by building embedding models, chat-style models, and retrieval-augmented chat agents. Notably, all our AI models surpass previous open systems in their respective categories, underscoring the critical role of diverse, high-
    
[^294]: 面向复杂地质导向场景中可解释决策支持的生成式人工智能建模框架

    A Generative-AI Modeling Framework for Explainable Decision Support in Complex Geosteering Scenarios

    [https://arxiv.org/abs/2503.08509](https://arxiv.org/abs/2503.08509)

    该论文提出了一种集成生成对抗网络、集合模型更新方法和离散动态规划优化的实时AI驱动地质导向工作流程，为复杂定向钻井场景提供可解释的自动决策支持。

    

    在钻井过程中实时调整井眼方向的过程被称为地质导向（geosteering），它对碳氢化合物开采以及地热能、民用基础设施和二氧化碳封存等新兴定向钻井应用至关重要。地球能源行业寻求一种自动化的地质导向工作流程，能够根据实时观测数据持续更新地下不确定性并捕捉最新的地质认识。我们提出了一种实时的、人工智能驱动的地质导向工作流程，该流程集成了用于地质参数化的生成对抗网络（GAN）、用于模型更新的集合方法，以及用于定向钻井作业中复杂决策的全局离散动态规划（DDP）优化。我们的框架依赖于对GAN模型进行离线训练以再现相关地质实现，并利用一个前馈神经网络（FNN）来模拟随钻测井（LWD）工具对给定地质情况的响应。

    arXiv:2503.08509v2 Announce Type: replace  Abstract: The real-time process of directional changes while drilling, known as geosteering, is crucial for hydrocarbon extraction and emerging directional drilling applications such as geothermal energy, civil infrastructure, and CO2 storage. The geo-energy industry seeks an automatic geosteering workflow that continually updates subsurface uncertainties and captures the latest geological understanding, informed by real-time observations.   We propose a real-time, AI-driven geosteering workflow that integrates Generative Adversarial Networks (GANs) for geological parameterization, ensemble methods for model updating, and global discrete dynamic programming (DDP) optimization for complex decision-making during directional drilling operations. Our framework relies on offline training of a GAN model to reproduce relevant geology realizations and a Forward Neural Network (FNN) to model the response of Logging-While-Drilling (LWD) tools for a give
    
[^295]: 信息几何逆向蒸馏用于增强对抗迁移性

    Information-Geometric Inverse Distillation for Enhancing Adversarial Transferability

    [https://arxiv.org/abs/2502.17003](https://arxiv.org/abs/2502.17003)

    提出逆向知识蒸馏（IKD）机制，通过最大化代理模型上良性样本与对抗样本的预测分布差异来增强对抗攻击的迁移性，并从信息几何角度证明软标签交叉熵与KL散度在固定锚点下完全等价。

    

    基于迁移的对抗攻击依赖代理模型来构造扰动，但往往会过拟合代理模型的决策边界。为解决这一问题，我们提出逆向知识蒸馏（IKD），这是一种简单且与攻击方法无关的机制，它通过最大化代理模型上良性样本与对抗样本之间的预测分布差异来实现攻击。IKD使用与交叉熵/KL散度等价的软标签目标，将对抗预测推离固定的良性预测锚点，并通过费舍尔敏感的代理方向来丰富攻击。我们证明，在匹配的固定锚点实现下，软标签交叉熵与KL散度仅相差一个常数熵项，因此会产生完全相同的梯度、海森矩阵和对抗优化轨迹。我们的信息几何分析进一步推导出代理模型与目标模型之间主导费舍尔子空间重叠度的定量下界……

    arXiv:2502.17003v2 Announce Type: replace-cross  Abstract: Transfer-based adversarial attacks rely on surrogate models to craft perturbations, yet often overfit the surrogate's decision boundary. To address this problem, we propose Inverse Knowledge Distillation (IKD), a simple and attack-agnostic mechanism that maximizes the prediction-distribution discrepancy between benign and adversarial samples on the surrogate model. IKD uses a CE/KL-equivalent soft-label objective to push adversarial predictions away from a fixed benign prediction anchor and enrich the attack with Fisher-sensitive surrogate directions. We prove that, under a matched fixed-anchor implementation, soft-label cross-entropy and KL divergence differ only by a constant entropy term and therefore induce identical gradients, Hessians, and adversarial optimization trajectories. Our information-geometric analysis further derives a quantitative lower bound on dominant Fisher-subspace overlap between surrogate and target mod
    
[^296]: 基于格林函数类比与Jackson-Chebyshev谱设计的物理信息支持向量核

    Physics-Informed Support Vector Kernels via Green-Function Analogies and Jackson-Chebyshev Spectral Design

    [https://arxiv.org/abs/2502.11153](https://arxiv.org/abs/2502.11153)

    该论文提出了一种利用格林函数类比构造的Jackson阻尼Chebyshev支持向量核，通过显式特征映射保证半正定性并提供可检查的谱先验，从而在无需精确等同物理传播子的情况下实现物理信息驱动的核选择，并在多种物理系统回归任务中得到验证。

    

    物理可观测量回归中的核选择通常是启发式的。我们研究了一种物理信息驱动的策略，其中与格林函数相关的函数形式和谱结构为核选择提供依据，而无需在机器学习核与物理传播子之间建立精确的等同关系。核心构造是受核多项式方法（KPM）启发的Jackson阻尼Chebyshev核；其显式特征映射在构造上保证了Gram矩阵的半正定性，并为结构化可观测量提供了可检查的谱先验。我们在铜电导率代理、局域类狄拉克能带色散、四次振子能级、光子晶体透射以及斐波那契链透射等问题上评估了标准与自定义SVR模型，采用了重复嵌套验证、学习曲线、随机森林与多层感知机基线，以及低秩Nyström测试等方法进行对比。

    arXiv:2502.11153v4 Announce Type: replace-cross  Abstract: Kernel selection for regression of physical observables is often heuristic. We investigate a physics-informed strategy in which functional forms and spectral structures associated with Green's functions motivate kernel selection without requiring an exact identification between a machine-learning kernel and a physical propagator. The principal construction is a Jackson-damped Chebyshev kernel inspired by the kernel polynomial method (KPM); its explicit feature map yields a positive-semidefinite Gram matrix by construction and provides an inspectable spectral prior for structured observables. We evaluate standard and custom SVR models on copper-conductivity proxies, local Dirac-like band dispersion, quartic-oscillator energy levels, photonic-crystal transmission, and Fibonacci-chain transmission using repeated nested validation, learning curves, random-forest and multilayer-perceptron baselines, and low-rank Nystr\"om tests wher
    
[^297]: 软件工程研究人员对机器学习在研究、评审与教育方面实践的看法

    Perspective of Software Engineering Researchers on Machine Learning Practices Regarding Research, Review, and Education

    [https://arxiv.org/abs/2411.19304](https://arxiv.org/abs/2411.19304)

    该研究首次从软件工程研究人员的视角出发，通过定性分析揭示了他们在研究、教学和评审中使用机器学习时的多样化实践、面临的挑战及建议。

    

    背景：机器学习（ML）对软件工程（SE）产生了重大影响，但现有研究主要聚焦于从业者，而忽视了研究人员。这忽略了对在软件工程中教授、研究或评审ML应用时的实践与挑战的关注。目标：本研究旨在从软件工程研究人员的视角出发，通过提供他们在研究、教学和评审应用ML的软件工程研究时所遵循的实践的洞察，为ML与SE之间的协同知识做出贡献。方法：我们分析了熟悉ML的软件工程研究人员或使用ML撰写软件工程文章的作者，以及这些文章本身。我们使用扎根理论编码和定性分析，考察了所采用的实践、使用ML解决的软件工程任务、面临的挑战，以及评审者和教育者的观点。结果：我们发现了以数据收集、模型训练和评估为核心的多样化实践。一些被推荐的实践（例如，超参数……（摘要在此处截断）

    arXiv:2411.19304v2 Announce Type: replace-cross  Abstract: Context: Machine Learning (ML) significantly impacts Software Engineering (SE), but studies mainly focus on practitioners, neglecting researchers. This overlooks practices and challenges in teaching, researching, or reviewing ML applications in SE.   Objective: This study aims to contribute to the knowledge, about the synergy between ML and SE from the perspective of SE researchers, by providing insights into the practices followed when researching, teaching, and reviewing SE studies that apply ML.   Method: We analyzed SE researchers familiar with ML or who authored SE articles using ML, along with the articles themselves. We examined practices, SE tasks addressed with ML, challenges faced, and reviewers' and educators' perspectives using grounded theory coding and qualitative analysis.   Results: We found diverse practices focusing on data collection, model training, and evaluation. Some recommended practices (e.g., hyperpara
    
[^298]: 用于验证（量子）学习与测试的交互式证明

    Interactive proofs for verifying (quantum) learning and testing

    [https://arxiv.org/abs/2410.23969](https://arxiv.org/abs/2410.23969)

    该论文证明了资源受限的学习者或测试者通常无法通过与不可信方进行经典交互来提升学习或测试效率，尤其是在涉及量子内存的绝大多数测试和学习问题上，内存受限的量子算法也无法借助此类交互克服自身的资源限制。

    

    我们考虑在存在资源限制（如内存有限或数据访问能力弱）的情况下从数据中进行测试和学习的问题，这些限制会影响测试或学习的效率与可行性。特别地，我们提出如下问题：资源受限的学习者/测试者能否通过与一个资源不受限但不可信的第三方进行交互，比没有这种交互时更高效地解决学习或测试问题？在这项工作中，我们从抽象层面和具体问题两个互补的角度回答了这个问题：对于多种场景，我们证明了资源受限的学习者无法通过与不可信证明者进行经典交互而获得任何优势。作为一个特例，我们表明，对于绝大多数量子内存作为有意义资源的测试和学习问题，内存受限的量子算法无法克服

    arXiv:2410.23969v3 Announce Type: replace-cross  Abstract: We consider the problem of testing and learning from data in the presence of resource constraints, such as limited memory or weak data access, which place limitations on the efficiency and feasibility of testing or learning. In particular, we ask the following question: Could a resource-constrained learner/tester use interaction with a resource-unconstrained but untrusted party to solve a learning or testing problem more efficiently than they could without such an interaction? In this work, we answer this question both abstractly and for concrete problems, in two complementary ways: For a wide variety of scenarios, we prove that a resource-constrained learner cannot gain any advantage through classical interaction with an untrusted prover. As a special case, we show that for the vast majority of testing and learning problems in which quantum memory is a meaningful resource, a memory-constrained quantum algorithm cannot overcome
    
[^299]: DRL-AdaPart：基于深度强化学习驱动的自适应STAR-RIS分区方法，实现公平高效的资源利用

    DRL-AdaPart: DRL-Driven Adaptive STAR-RIS Partitioning for Fair and Efficient Resource Utilization

    [https://arxiv.org/abs/2407.06868](https://arxiv.org/abs/2407.06868)

    提出了一种基于深度强化学习的自适应STAR-RIS单元分区方法DRL-AdaPart，通过联合优化相移与子表面分配变量并引入惩罚项智能停用多余单元，在保证资源高效利用的同时，为静态和移动用户提供公平且高速的数据速率。

    

    在本工作中，我们提出了一种同时传输与反射可重构智能表面（STAR-RIS）单元的高效资源利用方法，以确保公平且高速的数据速率。我们引入了一个子表面分配变量，用于确定分配给每个用户的STAR-RIS单元数量，并通过使用经过适当定制的深度强化学习（DRL）算法，联合优化STAR-RIS的相移和子表面分配变量，从而最大化数据速率之和。所提出的DRL方法还与Dinkelbach算法以及所设计的混合DRL方法进行了比较。在DRL模型中引入了惩罚项，通过在不需要时智能地停用STAR-RIS单元来增强资源利用率。所提出的DRL方法能够为静态和移动用户实现公平且高速的数据速率，同时通过广泛的（仿真验证）确保高效的资源利用。

    arXiv:2407.06868v3 Announce Type: replace-cross  Abstract: In this work, we propose a method for efficient resource utilization of simultaneously transmitting and reflecting reconfigurable intelligent surface (STAR-RIS) elements to ensure fair and high data rates. We introduce a subsurface assignment variable that determines the number of STAR-RIS elements allocated to each user and maximizes the sum of the data rates by jointly optimizing the phase shifts of the STAR-RIS and the subsurface assignment variables using an appropriately tailored deep reinforcement learning (DRL) algorithm. The proposed DRL method is also compared with a Dinkelbach algorithm and the designed hybrid DRL approach. A penalty term is incorporated into the DRL model to enhance resource utilization by intelligently deactivating STAR-RIS elements when not required. The proposed DRL method can achieve fair and high data rates for static and mobile users while ensuring efficient resource utilization through extensi
    
[^300]: 面向神经网络的计算热带几何框架

    A Computational Tropical Geometry Framework for Neural Networks

    [https://arxiv.org/abs/2405.20174](https://arxiv.org/abs/2405.20174)

    该论文提出了一个计算热带几何框架，给出了一种可将神经网络的线性区域计算为多面体显式并集的算法及其正确性证明，为分析神经网络表达能力提供了具体的计算工具。

    

    我们提出了一个计算热带几何框架，用于对具有热带激活函数的神经网络进行符号分析。神经网络的线性区域数量作为衡量给定架构表达能力的度量，一直受到广泛研究。为了研究这些问题，我们在热带几何的框架下开展工作——热带几何是代数几何的一个组合与多面体变体——在该领域中，热带有理映射与前馈神经网络之间已知存在联系。我们通过开发用于研究神经网络线性区域的具体计算工具来扩展这一联系。我们提出了一种算法及其正确性证明，该算法将神经网络的线性区域计算为多面体的显式并集。我们进一步将热带表达式的线性区域数量的计算与其中出现的单项式数量联系起来，并展示了热带表达式通常可以被剪枝以……

    arXiv:2405.20174v3 Announce Type: replace  Abstract: We propose a computational tropical geometry framework for the symbolic analysis of neural networks with tropical activations. The number of linear regions of a neural network has been actively studied as a measure of the expressivity of a given architecture. To study these, we work in the setting of tropical geometry---a combinatorial and polyhedral variant of algebraic geometry---where there are known connections between tropical rational maps and feedforward neural networks. We expand this connection by developing concrete computational tools for studying the linear regions of neural networks. We present an algorithm, together with a proof of correctness, which computes the linear regions of a neural network as explicit unions of polyhedra. We further relate the computation of the number of linear regions of a tropical expression to the number of monomials that appear in it, and show how tropical expressions can often be pruned to
    
[^301]: 当公平性度量失效时：基于效用的ε-公平性视角

    When fairness metrics fail: A utility-based perspective on $\varepsilon$-fairness

    [https://arxiv.org/abs/2405.09360](https://arxiv.org/abs/2405.09360)

    该论文提出了一个将决策后果纳入公平性评估的基于效用的框架，并证明一个决策过程即使满足ε-公平性概率度量，在考虑结果效用时仍可能达到最大程度的不公平。

    

    决策过程中的公平性通常使用概率度量来量化。然而，这些度量未必能反映决策对受影响个体和群体所产生的后果。我们开发了一个基于效用的框架，将这些后果纳入公平性评估中。我们的主要结果表明，一个决策过程即使满足ε-公平性，一旦考虑其结果所关联的效用，仍可能在最大程度上不公平。为了应对假阴性信息不可用的应用场景，我们还提出了一种简化的设定，保留了基于效用的公平性评估的核心要素。我们通过两个应用来阐述该框架：大学录取和信用风险评估。在这两个案例中，概率度量可能将一个决策过程归类为近似公平，即使其对应的效用结果实际上极为不公平。

    arXiv:2405.09360v3 Announce Type: replace  Abstract: Fairness in decision-making processes is often quantified using probabilistic metrics. However, these metrics need not reflect the consequences of decisions for the affected individuals and groups. We develop a utility-based framework that incorporates these consequences into the assessment of fairness. Our main result shows that a decision-making process can satisfy $\varepsilon$-fairness while nevertheless being maximally unfair once the utilities associated with its outcomes are taken into account. To address applications in which information on false negatives is unavailable, we also formulate a reduced setting that retains the essential elements of the utility-based fairness assessment. We illustrate the framework through two applications: college admissions and credit-risk assessment. In both cases, probabilistic metrics may classify a decision-making process as approximately fair even though the corresponding utility outcomes 
    
[^302]: 高维情况下多个均值向量的估计

    Estimation of multiple mean vectors in high dimension

    [https://arxiv.org/abs/2403.15038](https://arxiv.org/abs/2403.15038)

    通过凸组合的方法估计高维空间中不同概率分布的多维均值，引入了两种权重确定策略：一种通过测试程序识别低方差的相邻均值，提出了封闭形式插补公式；另一种通过最小化二次风险的上置信界确定权重，通过理论分析得出方法对经验均值的二次风险改进，在维度渐近的角度上渐近地接近 Oracle（Minimax）改进。

    

    我们致力于基于独立样本在一个共同空间中估计来自不同概率分布的多维均值。我们的方法是通过对这些样本导出的经验均值进行凸组合来形成估计量。我们引入了两种策略来找到适当的依赖于数据的凸组合权重：第一种利用测试程序来识别具有低方差的相邻均值，从而产生了一个关于权重的封闭形式插补公式；第二种通过最小化二次风险的上置信区间来确定权重。通过理论分析，我们评估了我们的方法相对于经验均值提供的二次风险改进。我们的分析集中在维度渐近的角度上，显示我们的方法在数据的有效维度增加时渐近地接近于一个 Oracle（Minimax）改进。我们展示了通过提出的方法在均值估计中的应用。

    arXiv:2403.15038v1 Announce Type: cross  Abstract: We endeavour to estimate numerous multi-dimensional means of various probability distributions on a common space based on independent samples. Our approach involves forming estimators through convex combinations of empirical means derived from these samples. We introduce two strategies to find appropriate data-dependent convex combination weights: a first one employing a testing procedure to identify neighbouring means with low variance, which results in a closed-form plug-in formula for the weights, and a second one determining weights via minimization of an upper confidence bound on the quadratic risk.Through theoretical analysis, we evaluate the improvement in quadratic risk offered by our methods compared to the empirical means. Our analysis focuses on a dimensional asymptotics perspective, showing that our methods asymptotically approach an oracle (minimax) improvement as the effective dimension of the data increases.We demonstrat
    
[^303]: ResNLS：一种改进的股票价格预测模型

    ResNLS: An Improved Model for Stock Price Forecasting

    [https://arxiv.org/abs/2312.01020](https://arxiv.org/abs/2312.01020)

    ResNLS是一种结合ResNet和LSTM的混合模型，通过强调相邻股票价格之间的依赖关系来提升股价预测效果，其中使用前5个交易日收盘价作为输入的ResNLS-5模型相比现有最先进方法至少提升了20%的性能。

    

    股票价格预测一直是一项具有挑战性的任务。尽管许多研究项目试图解决这个问题，但很少有研究关注股票价格之间不同程度的依赖关系。在本文中，我们介绍了一种混合模型，该模型通过强调相邻股票价格之间的依赖关系来改进股票价格的预测。所提出的模型ResNLS主要由两种神经网络架构组成：ResNet和LSTM。ResNet作为特征提取器，用于识别股票价格之间的依赖关系，而LSTM则将这些被视为残差的依赖关系与初始时间序列数据结合起来进行分析。我们的实验表明，当使用前5个连续交易日的收盘价数据作为输入时，该模型（ResNLS-5）的性能相比于其他输入方式是最优的。此外，ResNLS-5相比当前最先进的方法至少有20%的提升。

    arXiv:2312.01020v3 Announce Type: replace  Abstract: Stock prices forecasting has always been a challenging task. Although many research projects try to address the problem, few of them pay attention to the varying degrees of dependencies between stock prices. In this paper, we introduce a hybrid model that improves the prediction of stock prices by emphasizing the dependencies between adjacent stock prices. The proposed model, ResNLS, is mainly composed of two neural architectures, ResNet and LSTM. ResNet serves as a feature extractor to identify dependencies between stock prices, while LSTM analyzes the initial time series data with the combination of dependencies, which are considered as residuals. Our experiment reveals that when the closing price data for the previous 5 consecutive trading days is used as input, the performance of the model (ResNLS-5) is optimal compared to those with other inputs. Furthermore, ResNLS-5 demonstrates at least a 20% improvement over current state-of
    

