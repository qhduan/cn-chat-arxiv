# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Primal Acceleration of Newton's Method](https://arxiv.org/abs/2608.21359) | 本文提出一种仅使用原始变量、每次迭代只需一次线性求解的加速牛顿法，实现$O(1/k^3)$全局收敛速率，且无需辅助子问题或对偶校正。 |
| [^2] | [PerturbRx: Learning Treatment-Conditioned Latent Transitions for Patient Drug Response Prediction](https://arxiv.org/abs/2608.21349) | PerturbRx通过建模药物和剂量条件下的潜在分子转变，无需治疗后数据即可提升患者药物反应预测的准确性，并在多个基准中达到最佳性能。 |
| [^3] | [Truthful Calibration Measures for Sequential Prediction](https://arxiv.org/abs/2608.21348) | 本文证明了在序列二值预测中，精确真实的校准度量无法同时满足完备性和健全性，并提出了两种近似真实的构建方法，实现了可扩展的乘性近似校准度量。 |
| [^4] | [Asymmetric Capacity Allocation in Self-Refinement Pipelines](https://arxiv.org/abs/2608.21345) | 本文首次系统研究了自优化流水线中生成、批评和修订各阶段对模型大小的需求，发现非对称容量分配（较大生成器和修订器、较小批评器）可有效节省资源而不损性能。 |
| [^5] | [TurboBias 2.0: Streaming Context-Biasing for Production-Efficient ASR Systems](https://arxiv.org/abs/2608.21343) | TurboBias 2.0通过引入不区分大小写的增强图和每流批处理解码，在支持流式推理的同时，实现了多用户独立上下文偏置，显著提升了生产级ASR系统的效率和个性化能力。 |
| [^6] | [Across-Design Uncertainty in Short Pricing Panels: Evidence from Simulated Price Trajectories](https://arxiv.org/abs/2608.21334) | 本文通过模拟价格轨迹证明，短定价面板中跨设计不确定性占估计误差方差的绝大部分（97.6%），并提出了一个经验关系式来描述其分散度。 |
| [^7] | [Time-Aware Tranformer-Based Prediction Model for AECOPD](https://arxiv.org/abs/2608.21324) | 本文提出了一种基于时间感知Transformer的预测模型，利用家庭呼吸机的呼吸数据，通过捕捉症状时间进展来及时预测AECOPD，显著优于传统方法。 |
| [^8] | [Rethinking Expressivity and Efficiency in Test-Time Training](https://arxiv.org/abs/2608.21308) | 本文提出E²-TTT方法，通过闭式状态转移精确重现每令牌递归的块结束状态，实现了分块级并行训练，同时保持了更新规则的时间结构，在长上下文任务中兼顾了表达性与硬件效率。 |
| [^9] | [SPARCL: Spectral Partitioned Analytic Continual Learning](https://arxiv.org/abs/2608.21307) | 本文通过识别解析持续学习中的频谱干扰问题，提出SPARCL方法，将自相关分解为核心与残差以冻结旧类分类器，从而减少旧类别漂移。 |
| [^10] | [ConceptTS: LLM-Guided Concept Bottlenecks for Interpretable Multivariate Time-Series Forecasting](https://arxiv.org/abs/2608.21277) | 本文提出ConceptTS，利用大语言模型自动生成可解释的概念瓶颈，无需人工标注即可实现多变量时间序列预测的透明化，并通过三个互补瓶颈提升预测的可解释性。 |
| [^11] | [The Exceedance Design Effect: Effective Sample Size for Thresholds under Clustering](https://arxiv.org/abs/2608.21262) | 本文提出在聚类相关数据下，设置阈值时需采用不同于平均值的有效样本量计算方法，以准确评估阈值的可靠性。 |
| [^12] | [On the Transferability of Agricultural Weed Detection Under Cross-Field Distribution Shift](https://arxiv.org/abs/2608.21254) | 本研究首次系统评估了农业杂草检测模型在跨田间和作物类型下的性能退化，并提出了减少重新标注需求的迁移策略。 |
| [^13] | [TRACE-C: Rank-Calibrated Relational Anomaly Detection for Multi-Stream Operational Telemetry](https://arxiv.org/abs/2608.21251) | TRACE-C提出了一种通过秩校准和Fisher聚合多通道异常分数的方法，用于检测多流遥测数据中的联合异常，并通过通道消融揭示了不同异常类型的贡献差异。 |
| [^14] | [Advanced Linear Algebra with Applications - Part I (Numerical linear algebra for PDEs, machine learning, and data assimilation)](https://arxiv.org/abs/2608.21234) | 该讲义强调数值线性代数的现代核心地位，通过少量关键思想（如范数、分解、条件数和浮点运算）统一处理来自偏微分方程、机器学习和数据同化中的大规模结构化问题。 |
| [^15] | [Event-triggered Implicit Perturbation for Zeroth-Order Fine-Tuning of Spiking Transformers](https://arxiv.org/abs/2608.21223) | 本文提出一种事件触发的隐式扰动零阶优化架构，通过将扰动总和与IMC加权和结合，消除了显式扰动导致的RMW操作，并利用脉冲稀疏性减少硬件开销，从而高效微调脉冲Transformer。 |
| [^16] | [Personalized Privacy Control in LLMs via Attention Head Intervention](https://arxiv.org/abs/2608.21209) | 本文提出个性化隐私概念和P3Bench基准，并开发Repair方法，通过注意力头干预增强LLM的个性化隐私控制，显著降低政策忽视率。 |
| [^17] | [Curriculum-Aware Interpolate-then-Refine: Learned Physiological Time-Series Imputation under Realistic Missingness](https://arxiv.org/abs/2608.21207) | 本文提出CAIR框架，通过两阶段（先插值后细化）策略，针对生理信号缺失中临床极端和长度多尺度特性，显著提升插补性能。 |
| [^18] | [No PUN Intended: Plausible Unknown Names for Person-Centred LLM Evaluation](https://arxiv.org/abs/2608.21206) | 本文提出PUN协议，用于构建和验证具有合理形式但无真实证据的未知人名，以改进LLM评估中人物相关任务的准确性和可靠性。 |
| [^19] | [Beyond Imitation: Self-Improving Robot Policies via Off-Policy Q-Planning](https://arxiv.org/abs/2608.21204) | 本文提出Q规划方法，通过为大型BC策略配备小型离策略Q函数，利用Q函数能吸收成功与失败数据的优势，实现推理时的价值引导动作选择和仅微调Q函数的在线自我改进，克服了BC无法自我学习的局限。 |
| [^20] | [Tydra: An Efficient Hybrid Model for Tabular Data](https://arxiv.org/abs/2608.21199) | Tydra通过混合Transformer和SSM层，在表格数据上实现了比TabPFN快30%的推理速度，同时保持高预测性能，并优于更大的Hydra模型。 |
| [^21] | [A Neurosymbolic Approach for Constructing Planning Domain Models from Clinical Narratives](https://arxiv.org/abs/2608.21186) | 本文提出NSPIN框架，结合预训练LLM和符号验证，从临床叙事中自动诱导概率规划领域模型，在真实手术记录上实现有效泛化。 |
| [^22] | [Thermo-FL: Thermal-Aware Robust Federated Fine-Tuning of Large Language Models for Edge AI](https://arxiv.org/abs/2608.21172) | 提出Thermo-FL框架，利用设备温度动态调节LoRA层和更新密度，并结合TERRA聚合管道，同时应对热约束和拜占庭攻击，实现边缘AI下鲁棒的联邦微调。 |
| [^23] | [Human-JEPA: A Human-Centric Vision Model that Perceives and Anticipates](https://arxiv.org/abs/2608.21160) | 本文提出Human-JEMA，一种基于视频训练的人类中心视觉模型，通过锚定预测和纯时间分割，以更少参数实现静态感知与动态预测的兼顾，且预测头不降低性能。 |
| [^24] | [Capturing Cardiac Cyclicity through Phase-Equivariant Self-Supervised Learning](https://arxiv.org/abs/2608.21147) | 本文提出一种基于相位等变自监督学习的Winder架构，通过固定闭式传输算子捕捉心脏周期对称性，在极小参数规模下实现与最先进方法相当的诊断精度。 |
| [^25] | [COEC: Calibrated Orthogonal-Equivalence Compensation for Structured Pruning of Large Language Models](https://arxiv.org/abs/2608.21142) | COEC提出了一种交替左右正交旋转的免训练补偿框架，通过约化Stiefel流形优化和广义交叉验证正则化，有效缓解了结构化剪枝后大型语言模型的精度损失。 |
| [^26] | [BackDFL: A Unified Benchmark For Backdoor Attacks and Defenses In Decentralized Federated Learning](https://arxiv.org/abs/2608.21137) | 本文指出去中心化联邦学习（DFL）的鲁棒性被高估，并提出了BackDFL，一个统一基准，用于在现实和适应性后门攻击下系统评估DFL的安全性。 |
| [^27] | [Llama-Mobile: Efficient 2.7-Bit Quantization of VLMs](https://arxiv.org/abs/2608.21134) | 提出一种无需训练数据的2.7位量化框架，将视觉语言模型压缩至3.7 GB，同时保持视觉问答性能，适用于移动设备高效推理。 |
| [^28] | [FlatLand: Personalized Graph Federated Learning via Tailored Lorentz Space](https://arxiv.org/abs/2608.21096) | 本文提出FlatLand，通过将客户端数据嵌入定制的双曲洛伦兹空间，并采用参数解耦策略分离异质与共性信息，解决了图联邦学习中客户端数据异构的挑战。 |
| [^29] | [Causal Modeling of Adverse Pregnancy Outcomes via Adaptive LLM Proposals](https://arxiv.org/abs/2608.21079) | 本文提出了一种结合LLM先验知识和经验数据评分的神经符号框架，通过自适应提议机制迭代生成不良妊娠结局的因果假设，以应对数据稀缺和领域知识不完整的挑战。 |
| [^30] | [AudioWorldSim: Realistic Binaural Audio Datasets For World Models](https://arxiv.org/abs/2608.21075) | AudioWorldSim通过扩展SoundSpaces 2.0并改进连续声音合成，自动生成真实双耳音频数据集，为世界模型研究提供了开源、可复现的平台。 |
| [^31] | [TracingFlow: A Simulation-Free Trajectory Inference Framework Based on Second-Order Dynamics](https://arxiv.org/abs/2608.21070) | TracingFlow通过引入二阶动力学和神经网络回归加速度场，提出了一种免模拟的流匹配框架，能够精确高效地解决动态最优加速度传输问题，从而捕捉高曲率和非线性演化轨迹。 |
| [^32] | [Designing a Robust LLM-Based Evaluation System for Agentic AI in Drug Discovery Through Human Alignment](https://arxiv.org/abs/2608.21057) | 本文提出了一种通过人类对齐验证的LLM-as-a-Judge评估框架，为药物发现智能体系统定义了四个质量维度并验证了评判器的可靠性，以解决现有评估方法无法捕捉语义正确性和扩展性问题。 |
| [^33] | [COMET: Contrastive Motion-Enhanced Temporal Reasoning for Video Multimodal Large Language Models](https://arxiv.org/abs/2608.21030) | COMET通过引入基于泰勒帧差分的运动分支和时序注意力偏置增强交叉注意力，并采用时序先验蒸馏与TC-GRPO优化，系统性地解决了视频多模态大语言模型在细粒度运动时序理解上的不足。 |
| [^34] | [RODE: A Radial-Orthogonal Decoupled Engine for Optimization](https://arxiv.org/abs/2608.21024) | RODE通过将矩阵优化分解为径向范数更新和正交方向更新，实现了更优的性能和更低的模型范数，并在多种任务中超越Muon优化器。 |
| [^35] | [From a Static Multi-Level Small Semantic Codebook to a Dynamic Single-Level Large Semantic Codebook for Generative Recommendation](https://arxiv.org/abs/2608.21012) | 本文提出了一种单级大语义码本和曝光感知动态更新机制，用单一语义标记替代多级残差量化，以减少解码成本并适应流量变化，提升生成式推荐的效率和准确性。 |
| [^36] | [Free-Probability Kernels for Zero-Rollout Hyperparameter Selection in Reservoir Computing](https://arxiv.org/abs/2608.20998) | 本文提出一种基于自由概率的确定性核方法，通过短先导序列即可选择储层计算超参数，无需任何回放，显著降低计算成本且性能不降。 |
| [^37] | [Trojaning the Alignment: Stealthy Backdoor Attacks against Graph Foundation Models](https://arxiv.org/abs/2608.20991) | 本文揭示了图基础模型在图-语言对齐下的后门攻击漏洞，提出一种同时利用图和文本模态的隐蔽后门攻击方法，以绕过现有防御并实现有效攻击。 |
| [^38] | [Jacobian-guided Noise Injection for Quantization Robustness in Large Language Models](https://arxiv.org/abs/2608.20988) | 本文提出一种基于雅可比范数引导的噪声注入训练策略，通过抑制softmax雅可比矩阵的范数来增强大型语言模型在量化中的鲁棒性，优于现有后训练量化方法。 |
| [^39] | [A Critical Audit of Spatiotemporal Forecasting Benchmark Datasets and Baselines](https://arxiv.org/abs/2608.20980) | 本文通过经典时间序列方法分析常用时空基准数据集，揭示无空间感知的线性模型比以往报告更具竞争力，质疑了现有基准数据集的判别可靠性。 |
| [^40] | [Training DeepFilterNet with Accurate Room Acoustic Simulations Improves Single-Channel Speech Enhancement](https://arxiv.org/abs/2608.20971) | 本文发现使用高保真度声学模拟生成的房间冲激响应数据集训练DeepFilterNet3，能显著提升语音增强性能和降低语音识别错误率，尽管增益未归因于特定模拟组件。 |
| [^41] | [Training, learning and inference: unified dynamics of neural systems](https://arxiv.org/abs/2608.20965) | 本文提出了一种基于生成事实图（GFG）的统一框架，将训练、学习和推理视为神经系统中的动态过程，并通过nanoGPT实验展示了其可观察性。 |
| [^42] | [TreeWY: Speculative Verification for Gated DeltaNet Hybrids](https://arxiv.org/abs/2608.20961) | 本文提出TreeWY方法，通过树状WY变换消除推测解码中的状态快照，显著降低内存开销，从而支持高接受率的宽草稿树。 |
| [^43] | [Quantization-Aware Healing: A Practical Recipe for Recovering Compressed, 4-Bit LLMs](https://arxiv.org/abs/2608.20953) | 提出量化感知修复（QAH）方法，直接从未压缩的原始模型蒸馏4比特学生模型，在显著降低计算成本的同时，在多数基准上匹配或超越bfloat16来源性能。 |
| [^44] | [Decoupling Policy Extraction for Offline Reinforcement Learning](https://arxiv.org/abs/2608.20909) | 本文提出了一种离线强化学习的策略解耦提取方法，旨在解决传统联合训练范式因固定数据导致的行动漂移和评论家过度估计问题。 |
| [^45] | [EviRank: Structured Relevance Evidence for Multimodal Image Re-ranking](https://arxiv.org/abs/2608.20886) | EviRank通过将多模态图像重排序转化为基于六类语义槽的约束满足问题，并利用无需训练的确定性评分和证据依据比较，解决了现有方法在细粒度约束上的遗漏或幻觉问题。 |
| [^46] | [Nothing Changed but the Model: CellFill -- Bounded In-Cell Learning for Bit-Identical, Revocable Updates to Quantized LLMs](https://arxiv.org/abs/2608.20873) | 本文提出CellFill方法，在冻结的4位量化模型内部学习残差，实现位级一致的更新，确保模型发布工件逐位不变且更新可撤销，同时保持性能与未受约束模型相当。 |
| [^47] | [ReCurveflow: A Flow Matching Framework that Learns Curved Reaction Trajectories to Predict Transition State Geometries](https://arxiv.org/abs/2608.20869) | ReCurveflow通过监督弯曲反应路径并引入离路径校正，显著提升了过渡态几何预测的准确性和鲁棒性。 |
| [^48] | [Sharing the Control Authority Between Deep Reinforcement Learning and Model Predictive Control: Application to Multi-Class Transportation Networks](https://arxiv.org/abs/2608.20858) | 本文提出一种新颖的DRL-MPC混合框架，通过划分控制权，解决多类交通网络中DRL学习能力受限和MPC模型依赖及计算耗时的问题，实现高效实时控制。 |
| [^49] | [SAC-Copula: Quality-Preserving Watermarking for Diffusion Language Models via Smooth Correlated Gumbel Fields](https://arxiv.org/abs/2608.20839) | SAC-Copula通过引入基于高斯copula的平滑局部相关Gumbel扰动场，解决了扩散语言模型水印中扰动与解码动态不匹配的问题，实现了生成质量与可检测性的更优平衡。 |
| [^50] | [Fine-tuning LLMs for Tourist Trajectory Prediction using Field Experiment Data](https://arxiv.org/abs/2608.20830) | 本研究通过微调大语言模型（Llama-3.1-8B）利用实地实验数据预测游客轨迹，在下一兴趣点预测中达到49.1%准确率，并展现出对雨天等欠采样场景的良好泛化能力。 |
| [^51] | [Scaling Muon for Diffusion Transformers](https://arxiv.org/abs/2608.20818) | 本文提出周期行式Muon优化器，通过每K步执行一次完整谱更新和低成本行式更新，在保持Muon优势的同时减少计算和通信开销，显著提升大型扩散变换器的训练效率。 |
| [^52] | [Resolution-Consistent Greedy Neural Approximation on Infinite-Dimensional Spaces](https://arxiv.org/abs/2608.20812) | 本文提出了一种在无限维空间上实现分辨率一致的贪心神经逼近方法，将逼近误差分解为坐标截断和有限宽度项，并建立了无维度统计保证。 |
| [^53] | [Neuro-Geospatial Modelling of EEG Affective States Using Literature-Informed Environmental Context](https://arxiv.org/abs/2608.20807) | 本文提出了一种双塔架构，利用文献信息环境先验作为辅助地理空间模态，在缺乏个体暴露数据时提升EEG情感状态分类性能，并通过多种控制实验验证其有效性。 |
| [^54] | [CubicSplat: Differentiable Vector Graphics via Error-Bounded Forward Relaxation](https://arxiv.org/abs/2608.20803) | 本文提出CubicSplat，一种通过均匀折线代理和误差有界松弛实现的可微矢量光栅化器，解决了前向精确性与梯度质量之间的跷跷板问题，确保良好条件的梯度并自动修剪退化基元。 |
| [^55] | [Rethinking Demonstration Unlearning in Imitation Learning for Robotics](https://arxiv.org/abs/2608.20784) | 本文提出了一种重新训练校准的审计方法，从行为和证据两个维度评估机器人模仿学习中的演示遗忘，确保编辑后的策略在行为上接近重新训练并防止证据泄露。 |
| [^56] | [Fuzzy-MoE: Interpretable Regime-Conditioned Expert Routing for Non-Stationary Multivariate Time Series Forecasting](https://arxiv.org/abs/2608.20761) | 本文提出Fuzzy-MoE，首次将模糊逻辑与混合专家模型结合，通过双视角路由识别潜在时间状态并动态激活专家，显著提升了非平稳多变量时间序列预测的准确性和可解释性。 |
| [^57] | [Hidden Axis of Uncertainty: Latent-Posterior Alignment in Graph Neural Networks with Bayesian Output Layers](https://arxiv.org/abs/2608.20758) | 本文发现图神经网络中贝叶斯输出层的预测不确定性减少源于潜在表示与低方差后验方向的对齐，而非传统假设的后验收缩，并据此提出对齐引导学习（AGL）方法来有效降低不确定性。 |
| [^58] | [PSK at WMT 2026 MIST: Task-Specialized QLoRA Adapters for Multilingual Summarization and Question Answering](https://arxiv.org/abs/2608.20757) | 本文提出了一种基于Tiny Aya Global模型和三个任务特化QLoRA适配器的多语言摘要与问答系统，通过分离任务适配器提升了摘要性能，并针对开放问答的不稳定性提交了多系统方案。 |
| [^59] | [Geometric Regularization for Long-Tailed Semi-Supervised Learning via Gaussian Feature Bridges](https://arxiv.org/abs/2608.20710) | 本文提出高斯桥一致性（GBC）框架，通过构建未标记样本与类别原型之间的几何插值路径，并引入桥一致性损失和BridgeMix技术，有效解决了长尾半监督学习中的噪声伪标签和确认偏差问题。 |
| [^60] | [CDRL: Certification-Driven Reinforcement Learning for Neutrino Flavor Model Discovery](https://arxiv.org/abs/2608.20686) | 本文提出了CDRL框架，通过将符号推理工具生成的结构化认证反馈转化为可重用约束，有效解决了强化学习在科学发现中因标量奖励信息不足而反复探索无效区域的问题，并在中微子味道模型发现任务上超越了现有最先进方法。 |
| [^61] | [Temporal Validity on Real Software Histories: Eliminating Stale-Fact Errors in Code-Assistant Memory over GitHub Fixes](https://arxiv.org/abs/2608.20685) | 本文验证了MemStrata在真实软件历史中通过确定性过时记忆消除RAG的时间盲点，显著提升答案准确率（0.91对比0.57-0.59），并减少过时事实错误。 |
| [^62] | [Reinforcement Learning for Continuous-Time Jump Markov Decision Processes with Applications to Network Dynamic Pricing](https://arxiv.org/abs/2608.20680) | 本文针对一般离散状态空间的连续时间跳跃马尔可夫决策过程，提出了一种熵正则化的强化学习框架，克服了现有方法对欧氏空间结构的依赖，并应用于网络动态定价问题。 |
| [^63] | [Lightweight Adaptive ReduNet via Hyperspherical Manifold Learning](https://arxiv.org/abs/2608.20668) | 本文提出LA-ReduNet，通过超球面流形学习细化逐层更新规则，显著减少展开层数，实现轻量级自适应白盒网络，降低参数存储同时保持判别性特征提取能力。 |
| [^64] | [C-Score: Beyond Accuracy for Robustness Assessment in Semi-Supervised Learning under Open-World Unlabeled Contamination](https://arxiv.org/abs/2608.20667) | 本文提出C-Score框架，用于诊断半监督学习在开放世界未标记污染下的隐藏崩溃，超越了传统准确率指标的局限性。 |
| [^65] | [Amplifying the imaging power of digital sky surveys with space telescopes data and generative AI](https://arxiv.org/abs/2608.20666) | 本文提出了一种利用生成式人工智能将地面巡天星系图像提升至太空望远镜图像质量的方法，从而结合地面巡天的高吞吐量与太空成像的高细节能力。 |
| [^66] | [Predicting Resource Efficient Hamiltonian Decomposition for Continuous-Time Quantum Walk Simulations](https://arxiv.org/abs/2608.20660) | 本文通过训练机器学习模型，利用图的拓扑特征预测连续时间量子行走模拟中泡利分解与匹配分解哪个更节省CX门资源，并在所有11,117个连通八顶点图上直接验证了预测准确性。 |
| [^67] | [RiskTraf: Risk-Extrapolated Residual Learning for Multi-Variate Traffic Flow Prediction](https://arxiv.org/abs/2608.20656) | 该论文提出PEMSB-3V基准套件和RiskTraf残差学习方法，通过保留并利用流量、速度和占有率三种原始传感器数据，并采用风险外推策略，解决了多变量交通流预测中数据不一致和状态依赖捷径问题。 |
| [^68] | [Meta-clustering of milk mid-infrared spectra identifies dairy cow groups associated with negative energy balance in early lactation](https://arxiv.org/abs/2608.20653) | 本研究通过结合光谱过滤、降维和聚类方法，从牛奶中红外光谱中直接识别出与早期泌乳负能量平衡相关的奶牛群体，为监测和管理高风险奶牛提供了新方法。 |
| [^69] | [Provable Edge-of-Stability for Adam on a One-Dimensional Quadratic](https://arxiv.org/abs/2608.20638) | 本文在一维二次函数上证明了Adam优化器存在向稳定阈值恢复的边缘稳定性现象，并揭示了该机制失效的特定情况。 |
| [^70] | [MIL-BERT: Classification of Arbitrarily Large Text with Performance and Explanatory Guarantees](https://arxiv.org/abs/2608.20636) | 本文提出MIL-BERT算法，利用多实例学习选择关键文本摘录进行分类，可处理近百万令牌的大规模文本，在多个长文本数据集上达到最先进性能，并具备解释性保证。 |
| [^71] | [Minimax Optimality of Score-Entropy Discrete Diffusion](https://arxiv.org/abs/2608.20635) | 本文首次确立了均匀和掩码离散扩散中具体得分估计的最小最大下界，揭示了其统计极限。 |
| [^72] | [Dual-Cache Latent Space Communication between Heterogeneous Language Models](https://arxiv.org/abs/2608.20617) | 我们提出了XKV，一种新颖的潜在空间通信方法，通过双缓存机制和跨层记忆检索，消除了异构语言模型间通信的上下文共享、层匹配和单层摘要等限制，实现了更高效的智能体间信息传递。 |
| [^73] | [JuryProbe: An Empirical Consensus-Risk Diagnostic for Routing Reference-Free Factuality Judge Panels to Grounded Verification](https://arxiv.org/abs/2608.20607) | 本文提出JuryProbe，一种通过仅假阴性相关性和假共识提升度来诊断无参考事实性评审团共识风险的方法，并在高风险时路由到有参考验证，以减少因共享盲点导致的错误接受。 |
| [^74] | [Keyed Provenance Watermarking with Complementary Lattice-Based Secure Aggregation for Federated Learning](https://arxiv.org/abs/2608.20580) | 本文提出一种联邦学习框架，通过密钥来源水印和格基安全聚合，同时解决数据泄露、未授权重用和梯度操纵问题，并引入基于物理锚定元数据的水印机制。 |
| [^75] | [FlavourBench: Ranking Frontier Language Models with Executable Culinary Ground Truth](https://arxiv.org/abs/2608.20574) | 该论文提出了一个基于可执行烹饪真实数据的自动化基准测试FlavourBench，通过版本化系统和严格统计方法对27个前沿语言模型进行公平排名，消除了传统基准中的评判者偏差和缺失数据问题。 |
| [^76] | [Faults That Fortify: CNN Adversarial Robustness via GPU Undervolting](https://arxiv.org/abs/2608.20572) | 本文发现训练期间通过GPU降压引入的硬件故障可作为隐式正则化，在降低功耗的同时提高CNN的对抗鲁棒性，甚至在对抗训练中也能增强防御能力。 |
| [^77] | [AgentDecarbonizer: Carbon-Aware Execution for AI Agents](https://arxiv.org/abs/2608.20566) | 本文提出AgentDecarbonizer，通过利用代理任务的截止日期灵活性，结合时间转移和空间转移策略，在不确定执行时间和缓存重计算挑战下实现碳感知执行，以降低AI代理工作流的碳排放。 |
| [^78] | [Conditional-Independence-Regularized Distributional Autoencoders for Mixed-Type Data](https://arxiv.org/abs/2608.20562) | 本文提出了一种结合能量分数、似然目标和条件独立性正则化的分布自编码器框架，用于学习混合类型数据的低维表示，同时保留异构变量间的可解释结构关系。 |
| [^79] | [Consistency Models for Fast MRI Reconstruction Using Regularization by Denoising](https://arxiv.org/abs/2608.20561) | 本论文提出CM-RED方法，将预训练一致性模型集成到去噪正则化框架中，通过受控噪声注入加速MRI重建，仅需4次网络评估即可在多种条件下实现高质量重建。 |
| [^80] | [Learning Prostate Anatomy at Test Time for Cancer Detection in Micro-Ultrasound](https://arxiv.org/abs/2608.20557) | 提出了一种名为ANT的分割引导测试时自适应框架，通过利用前列腺解剖结构来纠正域偏移，在无需额外标注的情况下提升微超声前列腺癌检测性能。 |
| [^81] | [aiXamine: Unified Black-Box Evaluation of Cross-Dimensional Trade-offs in LLM Safety, Security, and Privacy](https://arxiv.org/abs/2608.20554) | 本文提出了aiXamine，一个统一黑盒平台，通过跨维度评估LLM的安全、安全性和隐私性，揭示了传统独立评估框架无法检测的相互依赖风险模式，并在大规模测试中实现了可重复的比较分析。 |
| [^82] | [Keep Your Friends Close, and the Right Neighbours Closer: Disaster-Conditioned Kernel-Regularized Graph Attention for Building Damage Classification](https://arxiv.org/abs/2608.20548) | 该论文提出一种灾害条件的内核正则化图注意力方法，在建筑损伤分类中有效利用空间上下文，并针对不同灾害类型自适应调整邻域权重，以平衡视觉一致性和边界清晰度。 |
| [^83] | [Learning Exact NVIDIA SASS Encoders with $\mathbb{F}_2$ Linear Algebra](https://arxiv.org/abs/2608.20532) | F2Asm首次利用$\mathbb{F}_2$线性代数学习精确的SASS指令编码器，并成为支持Rubin SM107的开源NVIDIA SASS汇编器。 |
| [^84] | [When Graph-JEPA Learns the Wrong Thing: Diagnosing and Repairing Category-Conditional Collapse](https://arxiv.org/abs/2608.20516) | 本文揭示并修复了Graph-JEPA在科学推理图上出现的一种类别条件崩溃，即模型在线性探测和有效秩上表现正常但实例信息为零，通过方差分配分析诊断了根因并提供了修复方法。 |
| [^85] | [Bern2Edge: A Neurosymbolic Compiler for Edge Deployment via Bernstein Polynomial Networks](https://arxiv.org/abs/2608.20497) | 该论文提出Bern2Edge框架，通过伯恩斯坦多项式网络将神经网络转换为硬件高效且可解释的部署形式，在压缩下提升准确率达2.12个百分点。 |
| [^86] | [Metag: A dataset to build agentic meta-reviewing capabilities](https://arxiv.org/abs/2608.20488) | 本文提出了Metag数据集，用于训练和评估元审稿智能体，通过将审稿人意见、作者回应和手稿修订差异对齐，实现自动识别论文在评审过程中所做的修改。 |
| [^87] | [Uncertainty propagation in auto-regressive random neural network models](https://arxiv.org/abs/2608.20483) | 本文提出了一种基于Leaky ReLU分段线性结构的解析与粒子方法，用于随机神经网络及其自回归模型中的不确定性传播，实现了输出概率密度、特征函数及均值和协方差的精确或近似计算。 |
| [^88] | [When Clean Data Hurts: Learning with Monotone Corruptions Beyond Binary Classification](https://arxiv.org/abs/2608.20480) | 本研究证明单调对抗性损坏在多类分类和部分二分类概念类等更一般的学习设置中，比二分类情况更具破坏性，显著增加了学习器的错误率。 |
| [^89] | [Mutual information and sensitivity analysis for feature selection in customer targeting: a comparative study](https://arxiv.org/abs/2608.20447) | 本研究比较了互信息与基于数据的敏感性分析在银行电话营销特征选择中的效果，发现敏感性分析在低误报率下更优，而互信息在高误报率下略好。 |
| [^90] | [Amortized Bandwidth Learning for Kernel Density Estimation under Logarithmic Score](https://arxiv.org/abs/2608.20445) | 本文提出了一种基于摊销学习的核密度估计带宽选择方法，通过优化对数分数和仿射标准化，在多种任务中显著优于传统选择器。 |
| [^91] | [Stored in Optimizer State, Valued by Later Training: A Causal Account of Subliminal Trait Transfer](https://arxiv.org/abs/2608.20442) | 该论文通过状态手术证明优化器第一动量是隐性特质迁移的因果载体，并推导出传输-估值恒等式来解释源移除后的持续效应和符号变化。 |
| [^92] | [Shared Physics Responses Recover Hidden Rankings in Neural Operator Libraries](https://arxiv.org/abs/2608.20441) | 本文提出一种基于控制方程线性化响应的共享物理诊断方法，能够在无参考解的情况下高效恢复神经算子库中的模型排名，并显著提升选择准确性。 |
| [^93] | [Decision Tree and K-Means Analysis of Raman Spectra for Edible Oils: A Physics-Informed AI Approach](https://arxiv.org/abs/2608.20440) | 本研究通过物理信息人工智能方法，利用决策树仅需四个拉曼光谱变量即可对五种纯食用油实现100%准确分类，显著简化了食用油鉴别过程。 |
| [^94] | [Wrong-Physics Backdoors in Neural PDE Operators](https://arxiv.org/abs/2608.20439) | 本文提出了一种名为“跨参数重链接”的数据投毒攻击方法，能在神经PDE算子中植入“错误物理后门”，使模型在触发输入下输出物理上合理但参数错误的解，揭示了多参数档案库中张量到参数来源的脆弱性。 |
| [^95] | [Approximate Homomorphisms and Convergent Representations in Transducers](https://arxiv.org/abs/2608.20428) | 本文研究了换能器最小表示在扰动下的稳定性，引入近似同态概念，并证明对有限秩接口，所有足够接近的实现均具有近似同态，而标准换能器中存在无近似同态的接口。 |
| [^96] | [BF1: A Causal Dyadic Sparse-Attention Retrofit for Efficient Long-Context Transformers](https://arxiv.org/abs/2608.20427) | 本文提出BF1，一种确定性块对齐二联稀疏注意力机制，通过结合局部、全局和对数间隔历史块，在保持O(n log n)交互的同时，实现了长上下文Transformer的高效加速，并在实际模型中验证了显著的速度提升。 |
| [^97] | [From Thermal Preference Prediction to Adaptive Thermal Intervention: A Reinforcement Learning Approach Using Physiological and Environmental Sensing](https://arxiv.org/abs/2608.20423) | 本文提出了一种结合多模态生理与环境感知和强化学习的两阶段个性化热舒适方法，实现了从热偏好预测到自适应热干预的闭环控制。 |
| [^98] | [Rigorous Evaluation of Large Language Models for Malaria Drug Discovery: Trade-offs in Performance, Scale, and Resource Utility](https://arxiv.org/abs/2608.20418) | 该研究首次系统证明，在疟疾药物发现任务中，领域特定微调的开源大语言模型（如TxGemma-9B）显著优于经典机器学习和前沿专有模型，且微调是获得可靠性能的关键前提。 |
| [^99] | [Machine Learning and ARIMA Model Averaging for Adaptive Public Health Forecasting: Comparative Evaluation and an Ontario COVID-19 Case Study](https://arxiv.org/abs/2608.20406) | 本文提出了一种基于非负性能加权集成的MLAMA方法，结合ARIMA、随机森林和XGBoost模型，在安大略省COVID-19数据上实现了对转折点的高响应性和多时间尺度预测的平衡。 |
| [^100] | [Robust Discovery of Coarse-Grained Continuum Equations from Microscopic Dynamics](https://arxiv.org/abs/2608.20404) | 本研究发现，在从微观动力学数据中发现粗粒化PDE时，数据量是影响识别稳健性的关键因素，而增大函数库则会降低发现效率，并通过相分离系统和Ising模型验证了这一点。 |
| [^101] | [World models of environment, agent and joint agent-environment systems](https://arxiv.org/abs/2608.20401) | 本文提出世界模型的关键区分在于建模通道，并利用计算力学为环境、智能体及联合系统定义规范预测模型，揭示了闭环耦合下的新等价性。 |
| [^102] | [When Retrieval Fails Before It Begins: Structurally Indirect Prerequisite Eviction as a Retention Failure in Agentic Memory](https://arxiv.org/abs/2608.20400) | 本文首次揭示了代理记忆中的“检索前失败”模式——结构性间接前提驱逐，并提出依赖感知语义垃圾收集规则，显著提升全链保留率。 |
| [^103] | [Interpretable Information-Decomposed Brain Graph Learning for fMRI-based Disease Diagnosis](https://arxiv.org/abs/2608.20380) | 本文提出IID-GCN框架，通过部分熵分解将rs-fMRI交互分解为冗余、独特和协同三种信息图，以捕捉传统相关性方法遗漏的疾病相关信息结构，增强诊断的可解释性。 |
| [^104] | [If It Walks Like an Arbitrage: Protocol-Agnostic Detection with Decidable Structural Equivalence](https://arxiv.org/abs/2608.20377) | 本文提出一种协议无关的套利检测方法，通过将交易轨迹归约为可判定的规范形式，无需协议特定模式即可识别套利循环，并在Rocq中机械化验证了其所有关键性质。 |
| [^105] | [TH-GNN: Heterogeneous Temporal Graph Neural Networks for LLM-Agent Shilling Attack Detection](https://arxiv.org/abs/2608.20376) | TH-GNN通过融合异构时序图结构与跨模态语义注意力，有效检测LLM生成的推荐系统托攻击，解决了现有方法忽视图结构和时序协调的缺陷。 |
| [^106] | [VA-DPO: Valence-Arousal Direct Preference Optimization for Controllable Emotion Generation in Language Models](https://arxiv.org/abs/2608.20374) | VA-DPO通过将情感目标表示为连续效价-唤醒度点，并基于距离阈值筛选偏好数据，实现了比现有提示方法更精确可控的情感生成，显著降低了目标距离并提升了相关性。 |
| [^107] | [Harmonic Torsional Diffusion for Protein-Ligand Flexible Docking](https://arxiv.org/abs/2608.20366) | 本文提出Harmony，一种谐波扭转扩散框架，通过显式建模角度周期性并引入频率感知归纳偏差，显著提升了柔性蛋白质-配体对接中配体姿态和口袋重建的准确性。 |
| [^108] | [Multilingual Verifier Bias in RLVR: Benchmark, Rollout Diagnosis, and the Cross-Lingual Selection Bottleneck](https://arxiv.org/abs/2608.20362) | 本文揭示了多语言环境中RLVR的精确匹配验证器因语言差异产生严重假阴性偏差，并提出了一个可复用的审计协议和诊断方法，指出跨语言选择瓶颈是核心问题。 |
| [^109] | [TriPLU: Bypassing the Gate with Direct Trilinear Product FFNs in Tiny Language Models](https://arxiv.org/abs/2608.20360) | TriPLU通过直接三线性乘积分支替代门控机制，在微型语言模型中显著降低了验证损失，优于SwiGLU及其他乘积阶数控制。 |
| [^110] | [NeuroStrata: An Electroencephalographic Connectivity-Aware Deep Representation Learning Framework for Dynamic Brain Network Analysis of Mental Stress](https://arxiv.org/abs/2608.20354) | 该论文提出了一种名为NeuroStrata的框架，利用时变连接图与预训练深度学习模型，通过β频带连接分析实现了97.3%的心理压力分类准确率，显著优于传统静态特征方法。 |
| [^111] | [Exploratory As-Analyzed No-Detection of Culturally-Marked Predicate-Triggered PII Amplification in a Synthetic-English RAG Probe: A Predicate-Resource-Confounded Audit](https://arxiv.org/abs/2608.20351) | 本论文通过预注册审计发现，在合成英语RAG系统中，刻板印象负载查询并未在干净信息渠道上放大PII泄露，且早期泄露信号受提示回显伪影污染。 |
| [^112] | [Bankruptcy Prediction via Hybrid Resampling and Stacking Ensemble Techniques with Explainable Artificial Intelligence (XAI)-Driven Analysis](https://arxiv.org/abs/2608.20343) | 该论文提出了一种结合共识特征选择、混合重采样和堆叠集成的破产预测框架，并利用可解释人工智能提升了对不平衡金融数据中少数类别的检测性能。 |
| [^113] | [Green BOA: Determining the environmental break-even point for ML-based data compression](https://arxiv.org/abs/2608.19994) | 本文通过比较机器学习数据压缩算法在训练和推理中的碳成本与存储节省的碳收益，确定了其环境盈亏平衡点。 |
| [^114] | [Separating Covariate Shift from Mechanism Change with Two Discriminators: CJSD, a Conditional Discrepancy with an Exact Covariate-Concept Decomposition](https://arxiv.org/abs/2608.19885) | 本文提出了一种基于条件差异和两个判别器的决策层，能严格区分协变量偏移和机制变化，并确保流式专家模型重用、创建或延迟决策的统计有效性。 |
| [^115] | [Agentic ESOpt: Fine-Tuning Long-Horizon LLM Agents with Minimal GPU Requirements](https://arxiv.org/abs/2608.17310) | 本文提出用进化策略替代强化学习来微调长视野LLM智能体，实现了极低GPU需求下的全参数优化，并具备灵活性和长视野可扩展性。 |
| [^116] | [GEO-Flag: Detecting and Measuring GEO-Optimized Web Content](https://arxiv.org/abs/2608.16824) | 本文提出了GEOFlagBench基准，用于系统检测和度量生成式引擎优化（GEO）网页内容，并评估现有方法，发现最强基线F1为0.880但方法层面表现不均。 |
| [^117] | [Mint-Agent: Introducing Finance-Native Agentic Foundation Models](https://arxiv.org/abs/2608.16386) | 本文提出Mint-Agent，一种金融原生智能体基础模型，通过数据引擎、MintHarness框架和结合SFT、OPD与RLVR的训练算法，实现可靠且可审计的长周期金融研究执行。 |
| [^118] | [Information Geometry of Message Passing](https://arxiv.org/abs/2608.15922) | 本文提出自然梯度消息传递（NGMP）方法，通过边局部规则在因子图上实现变分推断，优于变分消息传递，能保留精确消息中接收族可表示的部分。 |
| [^119] | [Non-Shattering at and Above the Dynamical Temperature in the Spherical Pure p-Spin Model](https://arxiv.org/abs/2608.14369) | 本文证明了在球面纯p-自旋模型中，对于特定重叠参数范围，系统在动力学温度及以上不会发生破碎，部分解决了相关猜想。 |
| [^120] | [Attributing Preprocessing Invariance in Spectral Foundation Models](https://arxiv.org/abs/2608.14227) | 本文指出谱基础模型中的预处理不变性可能源于输入归一化本身，而非模型学习，并主张在评估时应将归一化单独作为基线。 |
| [^121] | [Towards Truly Unsupervised Evaluation of Feature Selection](https://arxiv.org/abs/2608.12057) | 本文批判了现有“无监督”特征选择评估方法的缺陷，并提出了一种基于无监督主成分分析和最优传输的、真正无监督的评估框架，无需任何标签信息即可衡量特征选择质量。 |
| [^122] | [From Recoverability to Functional Use: Certifying Temporal Reports in Time-Series Forecasting](https://arxiv.org/abs/2608.10433) | 该论文提出了时间序列预测中时间报告认证的三阶段框架，并揭示了可恢复性与预测性能之间的尺度差异，从而指导系统评估。 |
| [^123] | [ELVAE: Evidential Learning-Based Variational Autoencoder for Uncertainty-Aware Generation](https://arxiv.org/abs/2608.10398) | ELVAE通过NIG层次结构有效分离位置不确定性与条件变异性，并证明逆证据是生成任务中最强的不确定性信号，显著提升语义转换性能。 |
| [^124] | [Defining Decentralization: An Ontological Perspective](https://arxiv.org/abs/2608.09748) | 本文从本体论角度出发，提出了一种适用于计算机通信系统的去中心化通用定义，以解决现有概念混淆和形式推理不严谨的问题。 |
| [^125] | [A continually expandable foundation model for brain MRI](https://arxiv.org/abs/2608.08319) | 本文提出Alcmaeon，一种基于图蓝图剪枝的持续可扩展脑部MRI基础模型，在顺序扩展至多种临床领域时显著减少遗忘，同时保持早期能力。 |
| [^126] | [SMOPD: Multi-Reward Reinforcement Learning via Specialize-and-Merge Online Policy Distillation](https://arxiv.org/abs/2608.03092) | 本文提出一种通过专业化与合并的在线策略蒸馏方法（SMOPD），以增强稀疏奖励的优化信号，同时保持密集奖励驱动的能力，解决多奖励强化学习中不同粒度奖励信号失衡的问题。 |
| [^127] | [Latent Softmax for Data-Efficient Phoneme-Based Multilingual ASR Across Tonal and Non-Tonal Languages](https://arxiv.org/abs/2608.01281) | 提出潜在Softmax输出层，通过将声调元音作为潜在子类并在仅见基础元音标签时进行边缘化，实现了声调与非声调语言联合训练中的高效跨语言共享，提升了数据效率。 |
| [^128] | [An Introduction to Bayesian and Frequentist Simulation-Based Inference with Machine Learning](https://arxiv.org/abs/2607.21702) | 本文系统介绍了机器学习驱动的模拟推断方法在贝叶斯与频率学派框架下的参数估计、展开任务中的应用，并强调了验证策略与局限性。 |
| [^129] | [What a World Model Represents Is Three Questions](https://arxiv.org/abs/2607.06640) | 本文提出世界模型学习中的三个关键问题（可达性、准入性和分配性），揭示仅凭训练损失变化无法判断模型实际使用的信息途径。 |
| [^130] | [The Dual Nature of LLM Persona: Aggregated Tendencies and Frame-Dependent Geometry](https://arxiv.org/abs/2607.02368) | 本论文发现LLM人格表达包含聚合倾向与框架依赖几何两个可分离成分，后者并非固有属性，而是编码聚合无法捕捉信息的协调模式。 |
| [^131] | [FeLoG: Scalable and Efficient Distributed Graph Embedding with Feedback Loop Mechanism](https://arxiv.org/abs/2606.22180) | FeLoG通过反馈循环机制将采样与训练动态耦合，优先处理训练不足的节点，从而提升分布式图嵌入的可扩展性和效率。 |
| [^132] | [The Metanym Game: An LLM Benchmark Without Ground Truth That Rises With the Models It Measures](https://arxiv.org/abs/2606.21008) | 该论文提出一种无真实基准的LLM评估方法，通过类比生成与相互评分，利用SVD特征方程统一评判生成与评判能力，并发现与GPQA Diamond存在相关性。 |
| [^133] | [Valid Inference with Synthetic Data via Task Exchangeability](https://arxiv.org/abs/2606.13629) | 本文提出了一种基于“任务可交换性”的新统计条件，为在科学研究中使用合成数据提供了可证明的有效性保证，解决了合成数据可能存在的偏差和噪声问题。 |
| [^134] | [Detecting Functional Memorization in Code Language Models](https://arxiv.org/abs/2606.12764) | 本文提出了一种通过AI编码代理生成测试输入来检测代码语言模型中功能记忆化的方法，该记忆化在文本审计中不可见，通过反事实框架对比目标模型与参考模型实现功能等价性检测。 |
| [^135] | [Uncertainty-aware Multi-fidelity Closure via Conditional Normalizing Flows](https://arxiv.org/abs/2606.09857) | 本文提出了一种基于条件归一化流的不确定性感知多保真度闭合框架，通过概率映射从低保真度系数预测高保真度系数，显著提升降阶模型预测精度并量化闭合不确定性。 |
| [^136] | [INFUSER: Influence-Guided Self-Evolution Improves Reasoning](https://arxiv.org/abs/2606.09052) | INFUSER提出了一种影响力引导的自我进化框架，通过生成器与求解器的协同训练，利用优化器感知的影响力分数来改进问题生成，从而显著提升推理能力。 |
| [^137] | [PROBE-Web: An Interactive System for Probing Evaluation Landscapes of Knowledge Graph Completion Models](https://arxiv.org/abs/2606.08926) | PROBE-Web是一个交互式系统，允许用户通过调整预测锐度和流行度偏差鲁棒性两个视角，灵活评估知识图谱补全模型，并提供传统评估、视角感知评估、可解释案例研究和评估景观探索四大功能。 |
| [^138] | [Adaptive Inference for Resource-Constrained Dynamic Pricing](https://arxiv.org/abs/2606.03736) | 本文提出一种推断感知的重新求解策略，在资源受限动态定价中平衡收益与需求推断，实现接近最优的遗憾和信息效率。 |
| [^139] | [Towards Automated Discovery: A Review of Generative Models, Multimodal Learning and Closed-Loop Workflows in Inverse Materials Design](https://arxiv.org/abs/2606.02507) | 本综述系统梳理了生成模型、多模态学习和闭环工作流在逆向材料设计中的最新进展，强调物理先验和约束如何嵌入模型以实现可控的晶体结构生成。 |
| [^140] | [Self-Revising Discovery Systems for Science: A Categorical Framework for Agentic Artificial Intelligence](https://arxiv.org/abs/2606.01444) | 本文提出一个基于范畴论的框架，通过左Kan扩展和体制转换来区分检索、搜索与科学发现，实现不依赖主观新颖性的自修正智能体系统。 |
| [^141] | [On Finite-sample Concentration of Median of Incomplete U-Statistics](https://arxiv.org/abs/2606.00661) | 本文证明了不完整U统计量中位数（MoIU）的有限样本浓度界，克服了此前仅能获得松散$O(n^{-1/4})$界的理论挑战，实现了更紧的收敛速率。 |
| [^142] | [The Fast Mixing Mechanism for Differential Privacy](https://arxiv.org/abs/2605.30600) | 本文提出一种基于快速变换的新型差分隐私草图机制，在保持强隐私保证的同时，实现了与经典快速方法相当的运行效率。 |
| [^143] | [RouteScan: A Non-Intrusive Approach to Auditing MoE LLMs Safety via Expert Routing Telemetry](https://arxiv.org/abs/2605.24817) | 本文提出RouteScan，一种通过分析GPU执行中专家路由遥测来非侵入式审计MoE大语言模型安全性的方法，无需访问用户提示或模型内部，从而兼顾安全与隐私。 |
| [^144] | [Behavior-Consistent Deep Reinforcement Learning](https://arxiv.org/abs/2605.21214) | 本文提出行为一致强化学习框架，通过控制最大熵温度与Q值分歧来减少跨运行策略差异，实现高性能与分布相似性的平衡。 |
| [^145] | [STS: Efficient Sparse Attention with Speculative Token Sparsity](https://arxiv.org/abs/2605.15508) | STS提出了一种利用小型草稿模型预测重要令牌来动态构建稀疏掩码的方法，无需重训练即可在保持精度的同时显著加速LLM推理。 |
| [^146] | [Learning Minimal-Deviation Corrections for Multi-Dimensional Mismodelling in HEP Simulations](https://arxiv.org/abs/2605.07460) | 提出一种神经网络方法，在仅有一维目标分布和多维失配的约束下，通过最小偏差修正学习模拟事件的变换，既保持原始模拟的相关结构，又能针对性修正失配特征。 |
| [^147] | [GRALIS: Fusing Coalition and Gradient Attribution with Closed-Form Conservation Error and Finite-Sample Guarantees](https://arxiv.org/abs/2605.05480) | GRALIS通过融合Shapley联盟权重和连续梯度路径，提出了一种统一的XAI估计器，并提供了闭式守恒误差和有限样本保证，解决了现有方法在保真度指标上互补但不可直接比较的问题。 |
| [^148] | [Preference-Based Self-Distillation: Beyond KL Matching via Reward Regularization](https://arxiv.org/abs/2605.05040) | 本文提出基于偏好的自蒸馏（PBSD），通过奖励正则化替代传统KL匹配，解决了自蒸馏中的训练不稳定和探索多样性不足问题。 |
| [^149] | [RefusalGuard: Geometry-Preserving Fine-Tuning for Safety in LLMs](https://arxiv.org/abs/2605.01913) | 本文揭示了标准微调导致安全对齐退化的表示级机制，并提出了REFUSALGUARD框架，通过保持安全相关表示的几何结构来在微调中维持模型安全性。 |
| [^150] | [Compared to What? Baselines and Metrics for Counterfactual Prompting](https://arxiv.org/abs/2605.01048) | 本文指出，反事实提示中观察到的效应常被表面形式变化混淆，需使用基线（如改写）来校正，否则可能错误归因模型敏感性。 |
| [^151] | [AutoOR: Scalably Post-training LLMs to Autoformalize Operations Research Problems](https://arxiv.org/abs/2604.16804) | AutoOR提出了一种结合合成数据生成和强化学习的可扩展流水线，通过求解器反馈作为奖励信号，使8B参数模型在多个运筹学基准上达到或超越更大前沿模型的性能，尤其在非线性物理动力学问题上实现了突破。 |
| [^152] | [Calibrate-Then-Delegate: Safety Monitoring with Risk and Budget Guarantees via Model Cascades](https://arxiv.org/abs/2604.14251) | 本文提出一种名为校准-再委托（CTD）的模型级联方法，通过预测专家调用价值的轻量级探针和统计校准，在保障安全或成本的同时实现流式决策。 |
| [^153] | [Automatic classification pipeline for glitches in the Virgo detector](https://arxiv.org/abs/2604.13687) | 本文提出了VIGILant自动管道，利用ResNet34模型在Virgo探测器毛刺分类中达到高精度，并已部署于日常观测。 |
| [^154] | [Optimistic Online LQR via Intrinsic Rewards](https://arxiv.org/abs/2603.28938) | 提出了一种通过仅修改成本函数来引入内在奖励的乐观在线LQR算法，既保持了标准LQR结构，又实现了高效的不确定性驱动探索。 |
| [^155] | [A Deep Reinforcement Learning Framework for Closed-loop Guidance of Fish Schools via Virtual Agents](https://arxiv.org/abs/2603.28200) | 该论文提出了一种基于深度强化学习的虚拟智能体闭环引导鱼群框架，通过PPO训练和复合奖励函数平衡方向引导与凝聚力，在物理实验中实现了对活体鱼群的有效实时控制。 |
| [^156] | [Efficient Exploration at Scale](https://arxiv.org/abs/2603.17378) | 我们提出了一种在线RLHF算法，通过增量更新、微小肯定信号、认知不确定性和信息导向探索，实现了超过10倍的数据效率提升，仅用不到20K标签即可匹配离线RLHF在200K标签上的性能。 |
| [^157] | [Investigating Target Class Influence on Neural Network Compressibility for Energy-Autonomous Avian Monitoring](https://arxiv.org/abs/2602.17751) | 本文提出在野外低成本微控制器上运行高效AI模型，用于鸟类被动声学监测，以克服传统方法成本高和现有机器学习资源消耗大的问题。 |
| [^158] | [Interpretable clustering via optimal multi-way decision trees](https://arxiv.org/abs/2602.13586) | 本文提出了一种名为ICOMT的高性能计算框架，通过一维K-means离散化和最优多路决策树，实现了高可解释性的聚类，同时克服了传统方法中贪婪搜索和二元分裂的局限性。 |
| [^159] | [Degree-Mass Message Passing for Betweenness Ranking in Directed and Undirected Networks](https://arxiv.org/abs/2602.09716) | 提出了一种轻量级GNN架构，利用度质量作为大小不变特征，在合成图上训练，高效预测有向和无向网络中节点的介数中心性排名。 |
| [^160] | [Infinite-dimensional generative diffusions via Doob's h-transform](https://arxiv.org/abs/2602.06621) | 本文通过Doob h变换引入了一种无需时间反转的无限维生成扩散模型新框架，该框架在可验证条件下严格构造，并通过分数匹配目标实现近似，提供了更高的灵活性和泛化能力。 |
| [^161] | [Maximum Likelihood Reinforcement Learning](https://arxiv.org/abs/2602.02710) | 本文提出MaxRL，一种通过计算索引的样本目标函数族，将标准强化学习与最大似然统一，仅需一行代码更改即可在多种任务上超越现有方法。 |
| [^162] | [Interpretability in Deep Time Series Models Demands Semantic Alignment](https://arxiv.org/abs/2602.02239) | 本文提出深度时间序列模型的可解释性应追求语义对齐，即预测需用用户有意义的变量表达，并在时间演化下保持一致性，而非仅解释内部计算。 |
| [^163] | [Generalization Measures under Controlled Covariate Shift: A Regime-Aware Benchmark](https://arxiv.org/abs/2602.01718) | 本文通过受控协变量偏移基准，揭示了泛化度量的有效性强烈依赖于具体机制，并指出锐度和输入梯度类度量在不同条件下表现不一致。 |
| [^164] | [SEISMO: Explanation-Aware, Trajectory-Conditioned LLM Agents for Sample-Efficient Molecular Optimisation](https://arxiv.org/abs/2602.00663) | SEISMO通过利用预测器分数之外的信息（如任务描述、轨迹和可解释性反馈）作为指导信号，显著提升了分子优化的样本效率。 |
| [^165] | [CFM: Language-aligned Concept Foundation Model for Vision](https://arxiv.org/abs/2601.13798) | CFM通过提供具有空间定位能力的细粒度概念，使视觉基础模型的下游任务可解释，并利用概念共现关系增强解释质量。 |
| [^166] | [Deterministic and probabilistic neural surrogates of global hybrid-Vlasov simulations](https://arxiv.org/abs/2601.12614) | 本文提出基于图神经网络的确定性（Graph-FM）和概率性（Graph-EFM）替代模型，能够高效且准确地预测全球混合-Vlasov模拟中近地空间等离子体的时空演化，显著降低计算成本。 |
| [^167] | [GroupSegment-SHAP: Shapley Value Explanations with Group-Segment Players for Multivariate Time Series](https://arxiv.org/abs/2601.06114) | 本文提出GS-SHAP，一种将多元时间序列解释单元构建为跨变量分组段玩家、并通过Shapley归因量化贡献的新方法，有效捕捉了变量交互与时间动态的联合结构信号。 |
| [^168] | [AgentOCR: Reimagining Agent History via Optical Self-Compression](https://arxiv.org/abs/2601.04786) | AgentOCR通过将智能体历史压缩为渲染图像并引入分段光学缓存和自压缩机制，显著降低了令牌和内存成本，同时保持了任务性能。 |
| [^169] | [When to Ponder: Adaptive Compute Allocation for Code Generation via Test-Time Training](https://arxiv.org/abs/2601.00894) | 本文提出PonderTTT，一种无需训练的门控策略，通过TTT层的重建损失自适应触发测试时训练，在代码生成中实现高效计算分配，显著提升推理性能。 |
| [^170] | [Efficient Inference for Inverse Reinforcement Learning and Dynamic Discrete Choice Models](https://arxiv.org/abs/2512.24407) | 该论文提出了一种半参数去偏框架，通过将对数行为策略视为伪奖励，实现了在灵活奖励表示下对逆强化学习和动态离散选择模型的有效统计推断。 |
| [^171] | [Actively Learning Joint Contours of Multiple Computer Experiments](https://arxiv.org/abs/2512.13530) | 本文提出了一种联合等高线定位（jCL）方案，通过两种采集方案和决策规则，同时识别多个计算机实验的预定响应值，应用于飞行稳定条件识别。 |
| [^172] | [MeltwaterBench: Deep learning for spatiotemporal downscaling of surface meltwater](https://arxiv.org/abs/2512.12142) | 本文提出了一种融合遥感与物理模型数据的深度学习模型，实现格陵兰地表融水每日100米分辨率的时空降尺度，显著提高了准确性。 |
| [^173] | [Spatially Aware Dictionary-Free Koopman Eigenfunction Identification for Modeling and Control](https://arxiv.org/abs/2511.22648) | 该论文提出了一种无需预定义字典或网络结构的SADFED框架，通过正则化最小二乘和空间插值高效识别Koopman特征函数，并结合KPDE残差提升模型的空间一致性与控制性能。 |
| [^174] | [BIPPO: Budget-Aware Independent PPO for Energy-Efficient Federated Learning Services](https://arxiv.org/abs/2511.08142) | 提出了一种节能的多智能体强化学习方案BIPPO，通过预算感知和独立PPO优化，在资源受限的物联网环境中提升了联邦学习的客户端选择性能与能效。 |
| [^175] | [Benchmarking noisy label detection methods](https://arxiv.org/abs/2510.16211) | 本文通过将噪声标签检测方法分解为三个核心组件，提出统一基准任务和新指标，系统比较了多种方法在视觉和表格数据上的性能。 |
| [^176] | [LTR-ICD: A Ranking-Aware Framework for Automatic ICD Coding](https://arxiv.org/abs/2510.13922) | 本文首次将ICD编码问题从检索视角重新定义为分类与排序任务，提出排序感知框架，显著提升了高优先级诊断代码的识别与排序准确性。 |
| [^177] | [Doctor Rashomon and the UNIVERSE of Madness: Variable Importance with Unobserved Confounding and the Rashomon Effect](https://arxiv.org/abs/2510.12734) | 本文提出UNIVERSE方法，利用拉什莫尔集对存在未观测混杂和特征缺失时的变量重要性进行有界推断，兼顾模型不确定性，提供理论保障并验证了性能。 |
| [^178] | [Perseus: Interactive Time Series Segmentation with Sparse Supervision via Stateful Memory](https://arxiv.org/abs/2510.09930) | 珀尔修斯通过有状态记忆机制，将用户稀疏提示跨时间窗口持久保留，实现无需重训练的交互式时间序列分割。 |
| [^179] | [HIP: Hessian Interatomic Potentials without derivatives](https://arxiv.org/abs/2509.21624) | 本文提出了一种直接预测分子海森矩阵的深度学习模型HIP，无需导数计算，在速度、精度和扩展性上显著优于现有方法。 |
| [^180] | [AIRL-S: Unifying Reinforcement Learning and Search-Based Test-Time Scaling via Adversarial Inverse Reinforcement Learning](https://arxiv.org/abs/2508.14313) | AIRL-S通过对抗逆向强化学习从参考轨迹中推断密集逐步奖励，消除了对标记过程数据的依赖，并统一了强化学习与基于搜索的测试时扩展，在多个基准上显著提升了模型性能。 |
| [^181] | [Query Efficient Structured Matrix Learning](https://arxiv.org/abs/2507.19290) | 本文首次从一般矩阵族角度研究结构化矩阵学习的查询复杂度，提出从有限矩阵族中寻找近似最优近似的通用方法。 |
| [^182] | [CPC-CMS: Cognitive Pairwise Comparison Classification Model Selection Framework for Document-level Sentiment Analysis](https://arxiv.org/abs/2507.14022) | 该框架通过认知成对比较加权多标准评估，自动选择文档级情感分析的最优分类模型，并在多个数据集上验证了其有效性。 |
| [^183] | [Explaining Intrinsic Moral Self-Correction with Mechanistic Interpretability](https://arxiv.org/abs/2505.11924) | 该论文通过机械可解释性揭示了内在道德自我修正的机制是表示引导，即提示词通过沿可解释潜在方向调整隐藏表示来改变模型行为，且这种方法比直接提示更有效。 |
| [^184] | [Reinforcing Multi-Turn Reasoning in LLM Agents via Fine-Grained Reward Structure and Credit Assignment](https://arxiv.org/abs/2505.11821) | 本文提出利用密集回合级奖励结构（终端、延迟、每回合）在GRPO和PPO中实现细粒度信用分配，从而提升LLM智能体在多轮推理任务中的强化学习效果。 |
| [^185] | [SPD Matrix Learning for Neuroimaging Analysis: Perspectives, Methods, and Challenges](https://arxiv.org/abs/2504.18882) | 本文系统综述了SPD矩阵学习在神经影像分析中的统一框架，从模态特定表示到几何深度学习范式，强调了其连接经典统计与现代机器学习的核心贡献。 |
| [^186] | [An Automated Pipeline for Few-Shot Bird Call Classification: A Case Study with the Tooth-Billed Pigeon](https://arxiv.org/abs/2504.16276) | 本文提出了一种针对仅有少量录音的稀有鸟类（如齿嘴鸠）的自动化单样本叫声分类流水线，通过利用大型分类网络的嵌入空间和余弦相似度，结合预处理技术，在最小训练数据下实现高效检测。 |
| [^187] | [Smart Exploration in Reinforcement Learning using Bounded Uncertainty Models](https://arxiv.org/abs/2504.05978) | 本文提出利用有界不确定性模型集合优化Q函数上下界来引导强化学习探索，并提供收敛性理论保证及数据驱动正则化方法，以加速学习并确保策略收敛到最优。 |
| [^188] | [Regression-Based Estimation of Causal Effects in the Presence of Selection Bias and Confounding](https://arxiv.org/abs/2503.20546) | 本文提出在存在选择偏差和混杂时，利用代理变量从外部无偏数据校正偏差，以回归方法可靠估计因果效应E[Y|do(X)]的理论条件。 |
| [^189] | [Structure is information: structural identifiability mappings for machine learning with partially observed dynamical systems](https://arxiv.org/abs/2502.04131) | 本文提出了一种结构可辨识性映射方法，用于解决部分观测动态系统中机器学习应用中的模型模糊性问题，从而提升可解释性和数据利用效率。 |
| [^190] | [The Intrinsic Dimension of Prompts in Internal Representations of Large Language Models](https://arxiv.org/abs/2501.10573) | 本文通过内在维度分析大型语言模型提示词表示，发现其与词元不确定性相关，并利用逐层内在维度轮廓训练线性探针，在生成前高效区分恶意与良性提示，准确率达90-95%。 |
| [^191] | [DAOP: Data-Aware Offloading and Predictive Pre-Calculation for Efficient MoE Inference](https://arxiv.org/abs/2501.10375) | DAOP通过数据感知的专家动态分配和预测预计算，优化了MoE模型在内存受限设备上的GPU-CPU并行推理，显著减少传输延迟并保持准确性。 |
| [^192] | [Federated and differentially private estimation of KL divergence](https://arxiv.org/abs/2411.16478) | 本文提出FedPriKL方法，在差分隐私保证下，以低敏感性和方差、无偏估计联邦模型中数据的KL散度，实现隐私保护与高精度平衡，且通信开销小。 |
| [^193] | [On the Within-class Variation Issue in Alzheimer's Disease Detection](https://arxiv.org/abs/2409.16322) | 本文针对阿尔茨海默病检测中的类内变异问题，提出软目标蒸馏和实例级重平衡两种方法，通过估计样本特定概率分数来建模异质性和不平衡，从而提升检测性能。 |
| [^194] | [Forecasting with an N-dimensional Langevin Equation and a Neural-Ordinary Differential Equation](https://arxiv.org/abs/2405.07359) | 本文提出一种结合N维朗之万方程和神经常微分方程的新型数据驱动框架，用于系统建模和预测具有多重非平稳特征的电价时间序列，填补了现有方法孤立处理非平稳性的空白。 |
| [^195] | [Exact and efficient solutions of the LMC Multitask Gaussian Process model.](http://arxiv.org/abs/2310.12032) | LMC多任务高斯过程模型的精确解决方案表明，只需对噪声模型进行温和假设，即可实现高效计算。通过引入完整参数化的“投影LMC”模型和边缘似然函数表达式，展示了该方法相对于未经处理的方法的优异性能。 |

# 详细

[^1]: 牛顿法的原始加速方法

    Primal Acceleration of Newton's Method

    [https://arxiv.org/abs/2608.21359](https://arxiv.org/abs/2608.21359)

    本文提出一种仅使用原始变量、每次迭代只需一次线性求解的加速牛顿法，实现$O(1/k^3)$全局收敛速率，且无需辅助子问题或对偶校正。

    

    arXiv:2608.21359v1 公告类型：交叉 摘要：我们开发了一种新的直接加速牛顿法，用于最小化具有Lipschitz连续Hessian的凸函数。该算法仅使用原始变量，并且每次迭代仅执行一次线性求解。通过简单预定的参数选择，它在函数残差方面实现了$O(1/k^3)$的全局收敛速度。据我们所知，这是针对此类问题首次达到该速率且仅依赖每次迭代一次线性系统求解的二阶方法（无需求解辅助非线性正则化子问题，如三次正则化，执行非线性参数搜索，或使用对偶外梯度校正）。我们的方法可以以无Hessian方式实现，使用非精确线性系统求解器，同时保持快速全局速率。我们进一步将我们的构造扩展到通过Bregman散度的任意几何，以及复合优化问题。

    arXiv:2608.21359v1 Announce Type: cross  Abstract: We develop a new direct accelerated Newton method for minimizing convex functions with Lipschitz continuous Hessian. The algorithm uses only primal variables and performs just one linear solve per iteration. With a simple predetermined choice of parameters, it achieves the global convergence rate of $O(1/k^3)$ in terms of the functional residual. To the best of our knowledge, this is the first second-order method for this problem class attaining this rate while relying solely on one linear system solve per iteration (without solving auxiliary nonlinear regularized subproblems, such as cubic regularization, performing nonlinear parameter searches, or using dual extragradient corrections). Our method can be implemented in a Hessian-free way, using an inexact linear system solver, while preserving the fast global rate. We further extend our construction to arbitrary geometry through Bregman divergence, and to composite optimization proble
    
[^2]: PerturbRx：学习治疗条件下的潜在转变用于患者药物反应预测

    PerturbRx: Learning Treatment-Conditioned Latent Transitions for Patient Drug Response Prediction

    [https://arxiv.org/abs/2608.21349](https://arxiv.org/abs/2608.21349)

    PerturbRx通过建模药物和剂量条件下的潜在分子转变，无需治疗后数据即可提升患者药物反应预测的准确性，并在多个基准中达到最佳性能。

    

    摘要：arXiv:2608.21349v1 公告类型：交叉 摘要：稀缺的数据和肿瘤异质性限制了患者层面的癌症治疗反应预测。现有方法从治疗前的分子谱和药物表示中预测反应，而没有明确建模治疗预期下的分子变化。我们提出了PerturbRx，一种治疗条件下的表示学习框架，该框架学习干预诱导的潜在转变，并将其用作患者-药物反应特征。PerturbRx从上下文匹配但细胞未配对的对照和处理单细胞群体中训练一个药物和剂量条件下的转变预测器，然后冻结并将该预测器转移到治疗前患者谱上，无需治疗后测量。该转变与患者和药物表示相结合以预测反应。在TCGA和患者来源的异种移植基准测试中，PerturbRx在评估的方法中取得了最强的总体预测性能。

    arXiv:2608.21349v1 Announce Type: cross  Abstract: Scarce data and tumor heterogeneity limit patient-level cancer treatment-response prediction. Existing approaches predict response from pretreatment molecular profiles and drug representations, without explicitly modeling the molecular changes expected under treatment. We propose PerturbRx, a treatment-conditioned representation learning framework that learns intervention-induced latent transitions and uses them as patient-drug response features. PerturbRx trains a drug- and dose-conditioned transition predictor from context-matched but cell-unpaired control and treated single-cell populations, then freezes and transfers the predictor to pretreatment patient profiles without requiring post-treatment measurements. The transition is combined with patient and drug representations to predict response. Across TCGA and patient-derived xenograft benchmarks, PerturbRx achieves the strongest aggregate predictive performance among the evaluated 
    
[^3]: 序列预测中的真实校准度量

    Truthful Calibration Measures for Sequential Prediction

    [https://arxiv.org/abs/2608.21348](https://arxiv.org/abs/2608.21348)

    本文证明了在序列二值预测中，精确真实的校准度量无法同时满足完备性和健全性，并提出了两种近似真实的构建方法，实现了可扩展的乘性近似校准度量。

    

    校准要求概率报告在条件上无偏，并可靠地解释为概率。校准度量将数值误差分配给校准不当的报告。Haghtalab等人（2024）提出了一种用于在线预测的近似真实校准度量，但留下了精确真实性与完备性和健全性是否兼容的未解问题。我们针对序列二值预测给出了否定性答案：即使对于独立结果，精确真实性与完备性和健全性也不兼容。随后，我们表明这种不可能性仅特定于精确真实性。我们给出了两种从基础校准度量出发的一般性简化，分别产生加性近似真实和乘性近似真实的校准度量。应用乘性简化，对于每个$0 < \varepsilon < 1$，我们构建了一个健全且完备的校准度量，该度量是$(1+\exp(-T^{(1-\varepsilon)/2}))$-乘性近似的。

    arXiv:2608.21348v1 Announce Type: cross  Abstract: Calibration requires probabilistic reports to be conditionally unbiased and reliably interpretable as probabilities. A calibration measure assigns numerical error to miscalibrated reports. Haghtalab et al. (2024) proposed an approximately truthful calibration measure for online prediction, leaving open whether exact truthfulness is compatible with completeness and soundness.   We resolve this question negatively for sequential binary prediction: exact truthfulness is incompatible with completeness and soundness, even for independent outcomes. We then show that this impossibility is specific to exact truthfulness. We give two general reductions from a base calibration measure, producing additively and multiplicatively approximately truthful calibration measures, respectively. Applying the multiplicative reduction, for every $0 < \varepsilon < 1$ we construct a sound and complete calibration measure that is $(1+\exp(-T^{(1-\varepsilon)/2
    
[^4]: 自优化流水线中的非对称容量分配

    Asymmetric Capacity Allocation in Self-Refinement Pipelines

    [https://arxiv.org/abs/2608.21345](https://arxiv.org/abs/2608.21345)

    本文首次系统研究了自优化流水线中生成、批评和修订各阶段对模型大小的需求，发现非对称容量分配（较大生成器和修订器、较小批评器）可有效节省资源而不损性能。

    

    摘要：arXiv:2608.21345v1 公告类型：新 摘要：自优化，通常结构化为生成、批评和修订，是一种广泛采用的改进大语言模型生成的范式，并作为许多LLM智能体的核心机制。尽管这三个阶段涉及不同的认知需求，但大多数现有方法方便地将模型大小视为实现细节而非研究对象，这可能导致资源浪费。很少有工作系统地考察模型大小如何影响每个阶段，或有效的自优化是否需要同等能力的模型来进行生成、批评和修订。我们首次在5个不同领域的基准上，使用6种Qwen3模型大小和4种Gemma 3模型大小，对自优化流水线进行了分阶段模型大小研究。我们得出结论：较大的生成器和修订器通常能改善流水线，而过小的修订器甚至可能损害性能。其次，性能对修订器的容量高度不敏感。

    arXiv:2608.21345v1 Announce Type: new  Abstract: Self-refinement, typically structured as generation, critique, and revision, is a widely adopted paradigm for improving LLM generation and serves as a core mechanism in many LLM agents. While the three stages involve different cognitive demands, most existing approaches conveniently treat the model size as an implementation detail rather than a subject of study, which may lead to a waste of resources. Little work has systematically examined how model size affects each stage or whether effective self-refinement requires equally capable models for generation, critique, and revision. We present the first stage-wise model size study of the self-refinement pipeline on 5 benchmarks from different domains using 6 model sizes of Qwen3 and 4 model sizes of Gemma 3. We conclude that larger generators and refiners generally improve the pipeline, whereas an undersized refiner can even harm performance. Second, performance is highly insensitive to th
    
[^5]: TurboBias 2.0：面向生产高效ASR系统的流式上下文偏置

    TurboBias 2.0: Streaming Context-Biasing for Production-Efficient ASR Systems

    [https://arxiv.org/abs/2608.21343](https://arxiv.org/abs/2608.21343)

    TurboBias 2.0通过引入不区分大小写的增强图和每流批处理解码，在支持流式推理的同时，实现了多用户独立上下文偏置，显著提升了生产级ASR系统的效率和个性化能力。

    

    摘要：arXiv:2608.21343v1 公告类型：交叉 摘要：上下文化对于生产级自动语音识别（ASR）系统至关重要，在这些系统中，用户提供的短语必须在严格的延迟约束下被准确识别。尽管许多上下文偏置方法能提高识别准确性，但它们往往无法满足现代生产级ASR系统的实际需求：流式推理、高效的批处理解码、用户特定的上下文列表以及低运行时开销。我们提出了TurboBias 2.0，这是一个面向生产的框架，用于在基于Transducer的ASR系统中实现高效的短语增强。该框架扩展了GPU加速的TurboBias，引入了不区分大小写的增强图以及每流批处理解码，允许批次中的每个话语使用独立的上下文偏置配置。这实现了多个同时用户的个性化上下文偏置，而无需共享或混合他们的上下文列表。所提出的框架支持离线和流式两种模式。

    arXiv:2608.21343v1 Announce Type: cross  Abstract: Contextualization is essential for production automatic speech recognition (ASR) systems, where user-provided phrases must be recognized accurately under strict latency constraints. Although many context-biasing methods improve recognition accuracy, they often do not address the practical requirements of modern production ASR systems: streaming inference, efficient batched decoding, user-specific context lists, and low runtime overhead. We propose TurboBias 2.0, a production-oriented framework for efficient phrase boosting in Transducer-based ASR systems. The framework extends GPU-accelerated TurboBias with a case-insensitive boosting graph and per-stream batched decoding, allowing each utterance in a batch to use an independent context-biasing configuration. This enables personalized context biasing for multiple simultaneous users without sharing or mixing their context lists. The proposed framework supports both offline and streaming
    
[^6]: 短定价面板中的跨设计不确定性：来自模拟价格轨迹的证据

    Across-Design Uncertainty in Short Pricing Panels: Evidence from Simulated Price Trajectories

    [https://arxiv.org/abs/2608.21334](https://arxiv.org/abs/2608.21334)

    本文通过模拟价格轨迹证明，短定价面板中跨设计不确定性占估计误差方差的绝大部分（97.6%），并提出了一个经验关系式来描述其分散度。

    

    arXiv:2608.21334v1 公告类型：新 摘要：短观测定价面板可能包含许多观测值，但仅提供少量不同的价格变动。本文在一个校准至稀疏定价机制的合成数据生成过程中研究这一区别的推断后果。我们将条件于已实现价格轨迹的不确定性与同一定价过程生成的不同替代轨迹间估计误差的变化分开。在基线模拟中，后者成分占梯度提升规格估计误差方差的97.6%。面板内重采样程序使用一个已实现轨迹的信息，无法识别这一跨设计成分。三个结果组织了分析。首先，跨设计离散度由经验关系 sigma_hat 约等于 0.182 V^(-0.271) 良好描述，其中 V 等于变动次数乘以幅度平方。其次，添加共享区域

    arXiv:2608.21334v1 Announce Type: new  Abstract: Short observational pricing panels can contain many observations while offering only a small number of distinct price movements. This paper studies the inferential consequences of that distinction in a synthetic data-generating process calibrated to a sparse pricing regime. We separate uncertainty conditional on a realised price trajectory from variation in estimation error across alternative trajectories generated by the same pricing process. In the baseline simulations, the latter component accounts for 97.6% of the variance of estimation error for the gradient-boosted specification. Within-panel resampling procedures use the information of one realised trajectory and do not identify this across-design component.   Three results organise the analysis. First, across-design dispersion is well described by the empirical relation sigma_hat approx 0.182 V^(-0.271), where V equals moves times magnitude squared. Second, adding regions sharing
    
[^7]: 基于时间感知Transformer的慢性阻塞性肺疾病急性加重预测模型

    Time-Aware Tranformer-Based Prediction Model for AECOPD

    [https://arxiv.org/abs/2608.21324](https://arxiv.org/abs/2608.21324)

    本文提出了一种基于时间感知Transformer的预测模型，利用家庭呼吸机的呼吸数据，通过捕捉症状时间进展来及时预测AECOPD，显著优于传统方法。

    

    arXiv:2608.21324v1 公告类型：新 摘要：慢性阻塞性肺疾病急性加重（AECOPD）的快速症状变化使得时间敏感的预测模型至关重要。然而，目前大多数研究AECOPD的机器学习模型使用临床和实验室数据，这不可避免地会导致延迟。为确保及时检测AECOPD并最小化延迟，本文聚焦于家庭监测场景，仅使用日常呼吸机提供的呼吸数据。我们引入了一种基于时间感知Transformer的AECOPD预测模型，该模型利用时间感知Transformer生成有意义的患者表示，以捕捉呼吸机数据中的症状及其时间进展。我们的实验结果表明，基于时间感知Transformer的方法在多个分类任务中优于传统方法，突显了其提升AECOPD预测准确性的潜力。

    arXiv:2608.21324v1 Announce Type: new  Abstract: The rapid symptom change of Acute exacerbation of chronic obstructive pulmonary disease (AECOPD) makes it critical to have time-sensitive prediction models. However, most current machine learning models studying AECOPD use clinical and laboratory data, which will inevitably cause latency. To ensure timely detection of AECOPD and minimize latency, this paper focuses on home monitoring scenarios where only respiratory data from daily-use ventilators is available. We introduce a Time-Aware transformer-based AECOPD prediction model, which generates meaningful patient representations using the Time-Aware transformer to capture the symptoms and their temporal progression in ventilator data. Our experimental results demonstrate that our Time-Aware transformer-based approach outperforms traditional methods in multiple classification tasks, highlighting its potential to enhance AECOPD prediction accuracy.
    
[^8]: 重新思考测试时训练中的表达性与效率

    Rethinking Expressivity and Efficiency in Test-Time Training

    [https://arxiv.org/abs/2608.21308](https://arxiv.org/abs/2608.21308)

    本文提出E²-TTT方法，通过闭式状态转移精确重现每令牌递归的块结束状态，实现了分块级并行训练，同时保持了更新规则的时间结构，在长上下文任务中兼顾了表达性与硬件效率。

    

    arXiv:2608.21308v1 公告类型：新论文  摘要：测试时训练（TTT）通过在推理过程中持续更新权重来实现长上下文处理，但当前方法难以在每令牌更新动态的表达性与分块近似的硬件效率之间取得平衡。我们提出了E$^2$-TTT（表达且高效的TTT）来弥合这一差距。在采用块起始权重处梯度的标准近似下，我们推导出一个闭式状态转移，该转移精确重现了每令牌递归的块结束快速权重和动量状态。这使得完全并行化的块级训练成为可能，同时保留了先前分块方法所丢弃的更新规则的时间结构。我们通过从头训练高达13亿参数的模型来验证E$^2$-TTT。它在语言建模方面与先前的TTT和混合注意力基线表现相当，在上下文检索方面则优于它们。其优势在长序列场景中最为显著。

    arXiv:2608.21308v1 Announce Type: new  Abstract: Test-Time Training (TTT) enables long-context processing via continuous weight updates during inference, but current methods struggle to balance the expressivity of per-token update dynamics with the hardware efficiency of chunk-wise approximations. We propose E$^2$-TTT (Expressive and Efficient TTT) to bridge this gap. Under the standard approximation of taking gradients at the chunk-start weights, we derive a closed-form state transition that exactly reproduces the chunk-end fast-weight and momentum states of the per-token recurrence. This enables fully parallelized chunk-level training while preserving the temporal structure of the update rule that prior chunk-wise methods discard. We validate E$^2$-TTT by training models up to 1.3B parameters from scratch. It performs on par with previous TTT and hybrid attention baselines in language modeling while outperforming them on in-context retrieval. Its advantage is most pronounced in lengt
    
[^9]: SPARCL：频谱分区解析持续学习

    SPARCL: Spectral Partitioned Analytic Continual Learning

    [https://arxiv.org/abs/2608.21307](https://arxiv.org/abs/2608.21307)

    本文通过识别解析持续学习中的频谱干扰问题，提出SPARCL方法，将自相关分解为核心与残差以冻结旧类分类器，从而减少旧类别漂移。

    

    解析持续学习已成为基于梯度的类增量学习的一种强有力且无需样本的替代方法，因为它用闭式岭回归更新取代了迭代优化。然而，通常关于遗忘的叙述集中在随机梯度覆盖上，这并不能解释为何解析方法在精确递归求解器下仍会在旧类别上产生漂移。我们识别出根本原因在于频谱干扰：所有任务联合岭分类器共享逆自相关算子$(R+\lambda I)^{-1}$，因此新任务样本若加载到旧主导特征方向上，会稀释频谱并扰动旧类别的对数几率，即使旧标签从未被重新访问。基于这一观点，我们提出了SPARCL，一种频谱分区解析持续学习器，它将运行中的自相关分解为高能量核心和残差补集，冻结核心子空间中的旧类别分类器组件，并更新...

    arXiv:2608.21307v1 Announce Type: new  Abstract: Analytic continual learning has emerged as a strong exemplar-free alternative to gradient-based class-incremental learning because it replaces iterative optimization with closed-form ridge updates. Yet the usual forgetting narrative, centered on stochastic gradient overwriting, does not explain why analytic methods still drift on old classes despite exact recursive solvers. We identify the culprit as spectral interference: the joint ridge classifier for all tasks shares the inverse autocorrelation operator $(R+\lambda I)^{-1}$, so incoming task samples that load onto old dominant eigendirections dilute the spectrum and perturb old-class logits even when old labels are never revisited. Based on this view, we propose SPARCL, a spectral partitioned analytic continual learner that decomposes the running autocorrelation into a high-energy core and a residual complement, freezes old-class classifier components in the core subspace, and updates
    
[^10]: ConceptTS：基于大语言模型引导的概念瓶颈用于可解释多变量时间序列预测

    ConceptTS: LLM-Guided Concept Bottlenecks for Interpretable Multivariate Time-Series Forecasting

    [https://arxiv.org/abs/2608.21277](https://arxiv.org/abs/2608.21277)

    本文提出ConceptTS，利用大语言模型自动生成可解释的概念瓶颈，无需人工标注即可实现多变量时间序列预测的透明化，并通过三个互补瓶颈提升预测的可解释性。

    

    arXiv:2608.21277v1 公告类型：新 摘要：最先进的多变量时间序列预测器能够建模复杂的时间依赖和跨变量依赖，但其不透明的表示方式对为何产生特定预测提供了有限的洞察。这种缺乏透明度的特性限制了它们在必须理解和评估预测背后因素的场景中的应用。我们引入了ConceptTS，一个可解释的预测框架，其预测围绕命名的、人类可读的概念进行组织。ConceptTS使用大型语言模型来提出与任务相关的概念并生成可执行的标签规则，将语言模型的领域知识转化为直接监督，而无需昂贵的人工概念标注。所提出的概念被组织成三个互补的瓶颈，分别描述历史背景、局部预测区间和完整预测范围。一个共享解码器结合了从它们预测中派生的表示。

    arXiv:2608.21277v1 Announce Type: new  Abstract: State-of-the-art multivariate time-series forecasters can model complex temporal and cross-variable dependencies, yet their opaque representations provide limited insight into why a particular forecast is produced. This lack of transparency restricts their use in settings where practitioners must understand and assess the factors underlying a prediction. We introduce ConceptTS, an interpretable forecasting framework that organizes its predictions around named, human-readable concepts. ConceptTS uses a large language model to propose task-relevant concepts and generate executable labeling rules, translating the language model's domain knowledge into direct supervision without costly manual concept annotation. The proposed concepts are organized into three complementary bottlenecks that describe the historical context, local forecast intervals, and the full forecast horizon. A shared decoder combines representations derived from their pred
    
[^11]: 超越设计效应：聚类下阈值的有效样本量

    The Exceedance Design Effect: Effective Sample Size for Thresholds under Clustering

    [https://arxiv.org/abs/2608.21262](https://arxiv.org/abs/2608.21262)

    本文提出在聚类相关数据下，设置阈值时需采用不同于平均值的有效样本量计算方法，以准确评估阈值的可靠性。

    

    arXiv:2608.21262v1 公告类型：交叉 摘要：许多机器学习系统在校准集的分位数处设置阈值：例如，共形预测器通过将截止点设在校准集的第90百分位数来承诺90%的覆盖率；弃权门在模型得分低于校准集第10百分位数时拒绝回答；安全过滤器阻止任何得分超过参考集第99百分位数的输出。所有这些系统都承诺阈值在新数据上以规定比率保持。该承诺假设校准样本是独立的，但在现代流程中通常并非如此：它们共享提示、文档或推理轨迹。调查统计学自1965年以来就知道如何对相关数据进行折扣处理，通过计算样本相当于多少个独立观测值，但这仅适用于平均值。我们表明，阈值需要不同的计数方式。该计数取决于聚类得分落在阈值同一侧的频率，而这进一步影响...

    arXiv:2608.21262v1 Announce Type: cross  Abstract: Many machine-learning systems set a threshold at a quantile of a calibration set: conformal predictors that promise 90% coverage by drawing their cutoff at the calibration set's 90th percentile, abstention gates that decline to answer when a model's score falls below the calibration set's tenth percentile, safety filters that block any output scoring above the 99th percentile of a reference set. All of them promise that the threshold will hold at the stated rate on new data. The promise assumes the calibration examples are independent, and in modern pipelines they usually are not: they share a prompt, a document, a reasoning trace. Survey statistics has known how to discount correlated data since 1965, by counting how many independent observations a sample is worth, but only for averages. We show that a threshold needs a different count. The count depends on how often clustered scores land on the same side of the threshold, and that ch
    
[^12]: 跨田间分布偏移下农业杂草检测的可迁移性研究

    On the Transferability of Agricultural Weed Detection Under Cross-Field Distribution Shift

    [https://arxiv.org/abs/2608.21254](https://arxiv.org/abs/2608.21254)

    本研究首次系统评估了农业杂草检测模型在跨田间和作物类型下的性能退化，并提出了减少重新标注需求的迁移策略。

    

    arXiv:2608.21254v1 公告类型：交叉 摘要：在真实田间条件下实现精准农业杂草检测对于精准农业至关重要，能够实现针对性干预并减少产量损失。近期研究报道了基于无人机图像在多种作物中表现出较强的检测性能，然而现有方法仅在单一作物和田间进行评估，导致实践者缺乏证据表明在一个作物上训练的模型能否泛化到新田间或作物类型。在本研究中，我们刻画了跨数据集杂草定位性能下降的情况，并确定了哪些建模选择能够恢复性能，从而减少对每个新部署田间重新标注的需求。我们引入了一个新收集并标注的棉花田间农业杂草检测无人机图像数据集，并将其与一个在类似协议下收集的现有大豆数据集结合使用。利用这些数据集，我们评估了多种策略在将一个训练好的检测器迁移到另一个场景时的性能。

    arXiv:2608.21254v1 Announce Type: cross  Abstract: Accurate agricultural weed detection in real-world field conditions is essential for precision agriculture, enabling targeted intervention and reducing yield loss. Recent work has reported strong detection performance from UAV-based imagery across a range of crops, yet existing approaches evaluate within a single crop and field, leaving practitioners with little evidence that a model trained on one crop will generalize to a new field or crop type. In this work, we characterize where cross-dataset weed-localization performance degrades and which modeling choices recover it, reducing the need to relabel every new deployment field. We introduce a newly collected and annotated UAV image dataset for agricultural weed detection in cotton fields and use it alongside an existing soybean dataset collected under a similar protocol. Using these datasets, we evaluate the performance of several strategies for transferring a detector trained on one 
    
[^13]: TRACE-C：多流操作遥测数据的秩校准关系异常检测

    TRACE-C: Rank-Calibrated Relational Anomaly Detection for Multi-Stream Operational Telemetry

    [https://arxiv.org/abs/2608.21251](https://arxiv.org/abs/2608.21251)

    TRACE-C提出了一种通过秩校准和Fisher聚合多通道异常分数的方法，用于检测多流遥测数据中的联合异常，并通过通道消融揭示了不同异常类型的贡献差异。

    

    操作遥测数据可能在每个单独流都处于其正常范围内时，整体上却表现出联合异常。TRACE-C是一种可审计的、严格先验秩校准检测器，用于对齐的多流遥测数据：同机制滚动中位数/MAD残差输入到三个窗口通道——最大归一化局部和、基于稳健z残差的高斯copula形式依赖性对比，以及最差标准化AR(1)创新——这些通道的秩通过Fisher聚合，并与先前聚合进行排名比较。我们使用2019年1月至4月的拟合数据、2019年7月至12月的开发验证数据，以及2020年冻结的保持数据集，评估了六个英国电网流。TRACE-C将风暴Atiyah排在2019年测试窗口中的首位，但公开的通道消融实验表明，该排名归因于局部通道而非copula形式通道：仅使用copula时，Atiyah排名第59位。短时8月9日频率事件在融合检测器中的排名（143）远低于b（未完整）。

    arXiv:2608.21251v1 Announce Type: new  Abstract: Operational telemetry can be jointly anomalous while every individual stream stays inside its familiar range. TRACE-C is an auditable strictly-prior rank-calibrated detector for aligned multi-stream telemetry: same-regime rolling median/MAD residuals feed three window channels -- a maximum normalized local sum, a Gaussian copula-form dependence contrast on robust-z residuals, and a worst standardized AR(1) innovation -- whose channel ranks are Fisher-aggregated and ranked against earlier aggregates.   We evaluate six Great Britain grid streams with a January-April 2019 fit, July-December 2019 development evidence, and a 2020 hold-out frozen before inspection. TRACE-C ranks Storm Atiyah first among 2019 test windows, but a disclosed channel ablation attributes that rank to the local channel, not the copula-form channel: copula-only ranks Atiyah 59th. The short 9 August frequency event is ranked far lower by the fused detector (143) than b
    
[^14]: 高等线性代数及其应用——第一部分（用于偏微分方程、机器学习与数据同化的数值线性代数）

    Advanced Linear Algebra with Applications - Part I (Numerical linear algebra for PDEs, machine learning, and data assimilation)

    [https://arxiv.org/abs/2608.21234](https://arxiv.org/abs/2608.21234)

    该讲义强调数值线性代数的现代核心地位，通过少量关键思想（如范数、分解、条件数和浮点运算）统一处理来自偏微分方程、机器学习和数据同化中的大规模结构化问题。

    

    这些讲义构成了一个硕士级别高级数值线性代数课程的第一部分。其目的不仅在于介绍经典算法，还在于说明为何这一学科相较于一代人之前已变得更为核心。数值线性代数与偏微分方程的数值解同步发展，长期以来，其大型稀疏系统主要来源于此。如今，对网络节点进行排序、将观测数据同化到天气预报中、以及将模型拟合到大规模含噪数据集，这些问题都呈现出相同的特征：规模过大而无法直接分解、具有结构特性、且仅能通过矩阵-向量乘积来访问。令人惊讶的是，解决所有这些问题所需的核心思想寥寥无几。因此，每一章都先展开一个标准主题，然后将其应用于原始背景之外的场景。我们涉及范数、分解、条件数与浮点运算，以及由有限元方法产生的稀疏矩阵。

    arXiv:2608.21234v1 Announce Type: cross  Abstract: These lecture notes form the first part of a master's-level course on advanced numerical linear algebra. Their aim is not only to present the classical algorithms, but to show why the subject has become considerably more central than it was a generation ago. Numerical linear algebra grew up alongside the numerical solution of partial differential equations, and for a long time that is where its large sparse systems came from. Ranking the nodes of a network, assimilating observations into a weather forecast, and fitting a model to a large noisy data set now lead to problems of the same kind: too large to factorise, structured, and accessible only through matrix-vector products. Strikingly few ideas are needed for all of them. Each chapter therefore develops a standard topic and then puts it to work outside its original setting. We treat norms, factorisations, conditioning and floating-point arithmetic; sparse matrices arising from finit
    
[^15]: 事件触发隐式扰动用于脉冲Transformer零阶微调

    Event-triggered Implicit Perturbation for Zeroth-Order Fine-Tuning of Spiking Transformers

    [https://arxiv.org/abs/2608.21223](https://arxiv.org/abs/2608.21223)

    本文提出一种事件触发的隐式扰动零阶优化架构，通过将扰动总和与IMC加权和结合，消除了显式扰动导致的RMW操作，并利用脉冲稀疏性减少硬件开销，从而高效微调脉冲Transformer。

    

    arXiv:2608.21223v1 公告类型：交叉 摘要：零阶（ZO）优化仅通过前向传播评估来估计梯度，使其适用于微调不可微、事件驱动的脉冲神经网络（SNN）。然而，其在内存计算（IMC）加速器上的部署受到显式权重扰动引起的重复读-修改-写（RMW）操作以及用于统计独立逐权重扰动的随机数生成器（RNG）的过大硬件开销的限制。为解决这些挑战，我们提出了一种隐式扰动ZO（IPZO）架构，其中由事件触发扰动生成单元（PGU）计算的扰动总和与IMC阵列产生的加权总和相结合，消除了扰动引起的RMW操作，同时保持了IMC的权重静态执行。通过利用脉冲稀疏性，PGU仅在脉冲激活时生成并累积扰动贡献。

    arXiv:2608.21223v1 Announce Type: cross  Abstract: Zeroth-order (ZO) optimization estimates gradients using only forward-pass evaluations, making it suitable for fine-tuning non-differentiable, event-driven spiking neural networks (SNNs). However, its deployment on in-memory computing (IMC) accelerators is constrained by the repeated read-modify-write (RMW) operations arising from explicit weight perturbation and the prohibitive hardware footprint of random number generators (RNGs) for statistically independent per-weight perturbations. To address these challenges, we propose an implicit-perturbation ZO (IPZO) architecture in which perturbation sums computed by an event-triggered perturbation generation unit (PGU) are combined with the weighted sums produced by the IMC array, eliminating perturbation-induced RMW operations while preserving weight-stationary execution of IMC. By exploiting spike sparsity, the PGU generates and accumulates perturbation contributions only for spike-activa
    
[^16]: 基于注意力头干预的LLM个性化隐私控制

    Personalized Privacy Control in LLMs via Attention Head Intervention

    [https://arxiv.org/abs/2608.21209](https://arxiv.org/abs/2608.21209)

    本文提出个性化隐私概念和P3Bench基准，并开发Repair方法，通过注意力头干预增强LLM的个性化隐私控制，显著降低政策忽视率。

    

    arXiv:2608.21209v1 公告类型：新 摘要：智能体AI的兴起使LLM能够访问多样化的用户数据，引发了关键的隐私问题。先前关于情境隐私的研究探讨了LLM是否根据情境相关规范来调节信息披露。然而，即使在同一情境下，可接受的信息披露边界也可能因用户而异。为解决这一局限性，我们引入了“个性化隐私”，将用户特定的披露偏好纳入隐私控制中。我们进一步提出了P3Bench（个性化隐私保护基准），这是一个新颖的基准，扩展了情境隐私政策，加入了个性化披露政策。实验表明，基于提示的政策无法可靠地执行个性化隐私政策，Qwen2.5-7B和Gemma3-4B的平均政策忽视率分别为51.25%和74.28%。最后，为解决此问题，我们提出了Repair，一种稳健的方法。

    arXiv:2608.21209v1 Announce Type: new  Abstract: The rise of agentic AI enables LLMs to access diverse user data, raising critical privacy concerns. Prior work on contextual privacy studies whether LLMs regulate information disclosure according to context-dependent norms. However, acceptable disclosure boundaries may vary across users even within the same context. To address this limitation, we introduce \textit{personalized privacy}, which incorporates user-specific disclosure preferences into privacy control. We further present P3Bench~(\textbf{P}ersonalized \textbf{P}rivacy \textbf{P}reservation \textbf{Bench}mark), a novel benchmark extending contextual privacy policies with personalized disclosure policies. Experiments show that prompt-based policies fail to reliably enforce personalized privacy policies, with Qwen2.5-7B and Gemma3-4B showing average policy ignorance ratios of 51.25\% and 74.28\%, respectively. Finally, to address this problem, we propose \textsc{Repair}, a robust
    
[^17]: 课程感知的插值后细化：在现实缺失模式下学习生理时间序列插补

    Curriculum-Aware Interpolate-then-Refine: Learned Physiological Time-Series Imputation under Realistic Missingness

    [https://arxiv.org/abs/2608.21207](https://arxiv.org/abs/2608.21207)

    本文提出CAIR框架，通过两阶段（先插值后细化）策略，针对生理信号缺失中临床极端和长度多尺度特性，显著提升插补性能。

    

    arXiv:2608.21207v1 公告类型：交叉 摘要：对生理时间序列（如动脉血压、血糖等）进行插补对于处理临床数据中普遍存在的缺失问题至关重要。然而，现代插补方法在此领域表现不佳：最近的基准测试发现，在具有现实缺失间隙的真实临床信号上，简单线性插值优于所有学习的插补器。我们表明，这反映了生理缺失的两个特性，而通用插补器忽略了这些特性：间隙可能在信号处于临床极端而非典型状态时发生，且间隙长度可能跨越多个数量级。为此，我们引入了课程感知的插值后细化（CAIR），一种用于生理时间序列插补的两阶段框架。我们的关键动机是学习一条粗略的基础曲线，然后反复将其校正为生理现实，而不是单次预测间隙。因此，CAIR将双向GRU插值器与细化阶段相结合，以逐步改善插补结果。

    arXiv:2608.21207v1 Announce Type: cross  Abstract: Imputing physiological time series (arterial blood pressure, blood glucose, etc.) is essential for addressing the missingness that pervades clinical data. Yet modern imputation methods perform poorly in this domain: a recent benchmark found that simple linear interpolation outperformed every learned imputer on real-world clinical signals with realistic gaps. We show that this reflects two properties of physiological missingness that generic imputers ignore: gaps may occur when the signal is clinically extreme rather than typical, and gap lengths can easily span orders of magnitude. To this end, we introduce Curriculum-Aware Interpolate-then-Refine (CAIR), a two-stage framework for physiological time-series imputation. Our key motivation is to learn a coarse base curve and then repeatedly correct it toward physiological realism, rather than predict a gap in a single pass. Consequently, CAIR couples a bidirectional-GRU interpolator with 
    
[^18]: 无意双关：面向以人为中心的LLM评估的合理未知姓名

    No PUN Intended: Plausible Unknown Names for Person-Centred LLM Evaluation

    [https://arxiv.org/abs/2608.21206](https://arxiv.org/abs/2608.21206)

    本文提出PUN协议，用于构建和验证具有合理形式但无真实证据的未知人名，以改进LLM评估中人物相关任务的准确性和可靠性。

    

    arXiv:2608.21206v1 公告类型：交叉 摘要：在大型语言模型（LLM）评估中，人名常被用作提示变量，以考察事实性、隐私泄露、偏见和弃权行为，但当姓名的证据状态不受控制时，测量结果可能混淆记忆、检索、姓名先验和错误人物归属。我们将未知姓名操作化定义为具有合理的“名-姓”形式、无索引的全名证据、且在文档化验证运行下无歧义信号的姓名，并引入PUN（合理未知姓名）协议，用于构建和验证此类姓名，该协议结合了Wikidata派生组件、网络支持的LLM筛选和受控搜索再验证。我们报告了接受率、可复现性、消融实验以及一项204名参与者的人类研究，发现被接受的姓名比对照组更具姓名特征，而参与者在仅3%的情况下能恢复人物证据。我们发布了300个姓名及其比较对照组。

    arXiv:2608.21206v1 Announce Type: cross  Abstract: Person names are widely used as prompt variables in LLM evaluations of factuality, privacy leakage, bias and abstention, but when a name's evidential status is uncontrolled, measurements may conflate memorisation, retrieval, name priors and wrong-person attribution. We operationalise an unknown name as one with plausible First-Last form, no indexed full-name evidence, and no ambiguity signals under a documented validation run, and introduce PUN (Plausible Unknown Names), a protocol for constructing and validating such names, combining Wikidata-derived components, web-enabled LLM screening, and controlled search revalidation. We report acceptance rate, reproducibility, ablations, and a 204-participant human study, finding accepted names are more name-like than controls while participants recover person evidence in only 3% of cases. We release 300 names with comparison controls.
    
[^19]: 超越模仿：通过离策略Q规划实现机器人策略的自我改进

    Beyond Imitation: Self-Improving Robot Policies via Off-Policy Q-Planning

    [https://arxiv.org/abs/2608.21204](https://arxiv.org/abs/2608.21204)

    本文提出Q规划方法，通过为大型BC策略配备小型离策略Q函数，利用Q函数能吸收成功与失败数据的优势，实现推理时的价值引导动作选择和仅微调Q函数的在线自我改进，克服了BC无法自我学习的局限。

    

    行为克隆（BC）在机器人操作领域取得了显著进展，但其根本局限在于无法自我改进：当策略失败时，如果没有额外的人类示范，它无法从失败中学习。强化学习微调提供了一条自我改进的路径，但已被证明难以扩展到支撑现代机器人策略的数十亿参数模型。我们提出了Q规划，该方法为大型视觉运动BC策略配备了一个小型离策略Q函数。由于Q函数估计价值而非模仿动作，它可以与BC策略在同一批成功示范上训练，随后吸收成功和失败的部署回放，这是BC所不具备的不对称性。我们利用这种不对称性，在推理时实现价值引导的动作选择（对BC采样进行单步Q加权平均），并通过仅微调Q函数进行在线自我改进，而无需改动BC策略本身。

    arXiv:2608.21204v1 Announce Type: cross  Abstract: Behaviour Cloning (BC) has driven remarkable progress in robot manipulation, yet it is fundamentally limited by its inability to self-improve: a policy that fails cannot learn from that failure without additional human demonstrations. Reinforcement Learning fine-tuning offers a path to self-improvement but has proven difficult to scale to the multi-billion-parameter models underpinning modern robot policies. We propose Q-Planning, which equips a large visuomotor BC policy with a small off-policy Q-function. Because a Q-function estimates value rather than imitates actions, it can be trained on the same successful demonstrations as the BC policy and later absorb both successful and failed deployment rollouts, an asymmetry BC does not have. We exploit this asymmetry to enable value-guided action selection at inference (a single-step Q-weighted average over BC draws) and online self-improvement that fine-tunes only the Q-function, leaving
    
[^20]: Tydra：一种用于表格数据的高效混合模型

    Tydra: An Efficient Hybrid Model for Tabular Data

    [https://arxiv.org/abs/2608.21199](https://arxiv.org/abs/2608.21199)

    Tydra通过混合Transformer和SSM层，在表格数据上实现了比TabPFN快30%的推理速度，同时保持高预测性能，并优于更大的Hydra模型。

    

    摘要：arXiv:2608.21199v1 公告类型：新 摘要：基于Transformer的表格基础模型（如TabPFN）在预测性能上表现出色，但随上下文长度呈二次方计算成本增加。另一方面，基于次二次方SSM的替代模型（如Hydra）以牺牲准确性换取效率。为平衡两者，我们引入了Tydra，一种用于表格上下文学习的混合Transformer-状态空间模型（SSM）架构，它交替使用注意力层和SSM层。在30个OpenML数据集上，Tydra相对于TabPFN将推理时间减少了30%，同时保留了大量预测性能。Tydra还优于约大十倍的Hydra模型，同时提供更快的推理速度。结果表明，混合架构是表格基础模型的一个有前景的方向。

    arXiv:2608.21199v1 Announce Type: new  Abstract: Transformer-based tabular foundation models such as TabPFN achieve strong predictive performance but incur quadratic computational cost with context length. On the other hand, subquadratic SSM-based alternatives such as Hydra trade away accuracy for efficiency. To balance both, we introduce Tydra, a hybrid Transformer-State Space Model (SSM) architecture for tabular in-context learning that interleaves attention and SSM layers. Across 30 OpenML datasets, Tydra reduces inference time by 30% relative to TabPFN while retaining much of its predictive performance. Tydra also outperforms an approximately ten-times-larger Hydra model while providing faster inference. The results indicate that hybrid architectures are a promising direction for tabular foundation models.
    
[^21]: 一种从临床叙事构建规划领域模型的神经符号方法

    A Neurosymbolic Approach for Constructing Planning Domain Models from Clinical Narratives

    [https://arxiv.org/abs/2608.21186](https://arxiv.org/abs/2608.21186)

    本文提出NSPIN框架，结合预训练LLM和符号验证，从临床叙事中自动诱导概率规划领域模型，在真实手术记录上实现有效泛化。

    

    腹腔镜阑尾切除术等外科手术是复杂且高风险的过程，然而将其工作流程形式化以支持决策仍然是一个重大挑战。在此背景下，由于缺乏结构化事件数据以及临床叙事中隐含动作的普遍存在，诱导概率规划领域模型尤为困难，这既不是经验符号方法也不是大型语言模型（LLMs）单独能够充分解决的。我们引入了NSPIN，一种从非结构化临床叙事中诱导概率规划领域模型的神经符号框架。我们的方法使用预训练的LLM从原始文本中提取并填补结构化事件序列，然后诱导PPDDL模型，并通过LLM提出的修订来优化其前置条件，同时以实证验证为指导。我们在9位外科医生撰写的2,660份腹腔镜阑尾切除术记录上评估了该方法。NSPIN生成的模型具有泛化能力。

    arXiv:2608.21186v1 Announce Type: new  Abstract: Surgical procedures such as laparoscopic appendectomy are complex, high-stakes processes, yet formalizing their workflows for decision support remains a significant challenge. Inducing probabilistic planning domain models in this setting is particularly difficult due to the lack of structured event data and the prevalence of implicit actions in clinical narratives, which neither empirical symbolic methods nor Large Language Models (LLMs) can adequately address on their own. We introduce NSPIN, a neurosymbolic framework for inducing probabilistic planning domain models from unstructured clinical narratives. Our method extracts and imputes structured event sequences from raw text using a pretrained LLM, then induces a PPDDL model and refines its preconditions with LLM-proposed revisions, guided by empirical validation. We evaluate the approach on 2,660 laparoscopic appendectomy notes written by 9 surgeons. NSPIN yields models that generali
    
[^22]: Thermo-FL：面向边缘AI的大型语言模型热感知鲁棒联邦微调

    Thermo-FL: Thermal-Aware Robust Federated Fine-Tuning of Large Language Models for Edge AI

    [https://arxiv.org/abs/2608.21172](https://arxiv.org/abs/2608.21172)

    提出Thermo-FL框架，利用设备温度动态调节LoRA层和更新密度，并结合TERRA聚合管道，同时应对热约束和拜占庭攻击，实现边缘AI下鲁棒的联邦微调。

    

    联邦微调使大型语言模型能够在边缘设备上适应，而无需集中私有数据，但实际部署必须同时应对硬件不稳定性和对抗性更新损坏。受热约束的客户端可能会降低速度、减慢本地训练或延迟同步聚合，而拜占庭客户端和通信层攻击者可能会破坏用于形成全局模型的更新。为解决这些挑战，我们提出了Thermo-FL，一个热感知的联邦LoRA微调框架，该框架使用设备温度作为本地适配器训练和稀疏更新传输的主动控制信号。在客户端，Thermo-FL根据设备加热或冷却调整活跃LoRA层比例和传输更新密度，从而在热应力下减少工作负载。在服务器端，Thermo-FL引入了TERRA，一个用于动态稀疏LoRA更新的鲁棒聚合管道，该管道结合了范数过滤。

    arXiv:2608.21172v1 Announce Type: new  Abstract: Federated fine-tuning enables large language models to adapt on edge devices without centralizing private data, but practical deployments must address hardware instability and adversarial update corruption together. Thermally constrained clients may throttle, slow local training, or delay synchronous aggregation, while Byzantine clients and communication-layer adversaries can corrupt the updates used to form the global model. To address these challenges, we present Thermo-FL, a thermal-aware federated LoRA fine-tuning framework that uses device temperature as an active control signal for local adapter training and sparse update transmission. On the client side, Thermo-FL adjusts the active LoRA-layer fraction and transmitted update density as devices heat or cool, reducing workload under thermal stress. On the server side, Thermo-FL introduces TERRA, a robust aggregation pipeline for dynamically sparse LoRA updates that combines norm fil
    
[^23]: Human-JEPA：一种感知与预测的人类中心视觉模型

    Human-JEPA: A Human-Centric Vision Model that Perceives and Anticipates

    [https://arxiv.org/abs/2608.21160](https://arxiv.org/abs/2608.21160)

    本文提出Human-JEMA，一种基于视频训练的人类中心视觉模型，通过锚定预测和纯时间分割，以更少参数实现静态感知与动态预测的兼顾，且预测头不降低性能。

    

    arXiv:2608.21160v1 公告类型：交叉 摘要：理解人类的机器应能感知当下并预测未来。现有的人类中心视觉模型在人类图像上预训练，在静态密集感知方面达到了最先进水平，因此运动和预测超出了其能力范围。在此，我们提出Human-JEPA，一种通过锚定预测在视频上训练的人类中心视觉模型：密集目标被固定到初始化的冻结副本上，防止密集感知的无声崩溃，块掩码被替换为纯过去到未来的分割，避免了五点动作税和十七点重识别崩溃。在冻结探针下，Human-JEPA在姿态和人物重识别方面以2.7倍更少的参数领先于像素锚定专家，放弃了高分辨率密集解析，其发布的预测头是首个不降低预测性能的。因此，一个安全适应的模型即可服务于理解人类的两个部分。

    arXiv:2608.21160v1 Announce Type: cross  Abstract: Machines that understand humans should perceive the present and anticipate the future. Existing human-centric vision model are pretrained on human images, set the state of the art in static dense perception, so motion and anticipation are out of reach. Here we present Human-JEPA, a human-centric vision model trained on video by anchored forecasting: dense targets are pinned to a frozen copy of the initialization, preventing a silent collapse of dense perception, and block masks are replaced by a pure past-to-future split, avoiding a five-point action tax and a seventeen-point re-identification collapse. Under frozen probes, Human-JEPA leads the pixel-anchored specialists on pose and person re-identification at 2.7 times fewer parameters, conceding high-resolution dense parsing, and its released predictor head is the first that does not degrade anticipation. A single safely adapted model thus serves both halves of understanding humans.
    
[^24]: 通过相位等变自监督学习捕捉心脏周期性

    Capturing Cardiac Cyclicity through Phase-Equivariant Self-Supervised Learning

    [https://arxiv.org/abs/2608.21147](https://arxiv.org/abs/2608.21147)

    本文提出一种基于相位等变自监督学习的Winder架构，通过固定闭式传输算子捕捉心脏周期对称性，在极小参数规模下实现与最先进方法相当的诊断精度。

    

    arXiv:2608.21147v1 公告类型：新 摘要：生理过程的循环结构为自监督表示学习提供了自然先验，而心动周期为利用这一点提供了特别明确的设置。我们推导了一个相位等变自监督目标，并引入了Winder，一种联合嵌入架构，将表示组织为相位不变坐标和相位旋转的谐波子空间。其传输算子固定且具有闭式解，来源于周期几何而非学习得到，且不增加参数。在PTB-XL数据集上，采用冻结线性探针协议进行评估，Winder在约1M参数规模下达到了与最先进自监督方法报告范围相当的诊断准确性，同时展现出相位等变的潜在几何结构。这些发现表明，显式编码心脏相位对称性可以在保持诊断有用信息的同时，产生一种潜在几何结构，该结构...

    arXiv:2608.21147v1 Announce Type: new  Abstract: The cyclic structure of physiological processes offers a natural prior for self-supervised representation learning, and the cardiac cycle provides a particularly well-defined setting in which to exploit it. We derive a phase-equivariant self-supervised objective and introduce Winder, a joint-embedding architecture that organises representations into phase-invariant coordinates and phase-rotating harmonic subspaces. Its transport operator is fixed and closed-form, derived from the cycle's geometry rather than learned, and adds no parameters. Evaluated on PTB-XL under a frozen linear-probe protocol, Winder attains diagnostic accuracy within the range reported by state-of-the-art self-supervised methods at a ~1 M parameter footprint, while exhibiting phase-equivariant latent geometry. These findings demonstrate that explicitly encoding cardiac-phase symmetry can preserve diagnostically useful information while yielding a latent geometry tha
    
[^25]: COEC：面向大型语言模型结构化剪枝的校准正交等价补偿方法

    COEC: Calibrated Orthogonal-Equivalence Compensation for Structured Pruning of Large Language Models

    [https://arxiv.org/abs/2608.21142](https://arxiv.org/abs/2608.21142)

    COEC提出了一种交替左右正交旋转的免训练补偿框架，通过约化Stiefel流形优化和广义交叉验证正则化，有效缓解了结构化剪枝后大型语言模型的精度损失。

    

    结构化剪枝通过移除权重列来减小大型语言模型（LLMs）的规模和推理成本，但由此产生的输出误差会降低模型精度。现有的免训练补偿方法在保留权重的输出侧使用加性偏置或单一正交旋转。这些修正未改变其输入奇异框架，因此限制了保留权重在列移除后的适应能力。我们提出COEC（校准正交等价补偿），一种免训练补偿框架，对保留权重交替施加左右正交旋转。右旋转在约化Stiefel流形上优化，而奇异值通过广义交叉验证进行重新缩放，以选择每层的正则化强度。COEC进一步对校准Gram矩阵进行调和，以减少高能量激活方向的主导影响，并引入混叠...

    arXiv:2608.21142v1 Announce Type: new  Abstract: Structured pruning reduces the size and inference cost of large language models (LLMs) by removing weight columns, but the resulting output error can degrade accuracy. Existing training-free compensation methods use an additive bias or a single orthogonal rotation on the output side of the retained weight. These corrections leave its input singular frame unchanged and therefore limit how the retained weight can adapt after column removal. We propose COEC (Calibrated Orthogonal-Equivalence Compensation), a training-free compensation framework that applies alternating left and right orthogonal rotations to the retained weight. The right rotation is optimized on a reduced Stiefel manifold, while singular values are rescaled using generalized cross-validation to select the regularization strength for each layer. COEC further tempers the calibration Gram matrix to reduce the dominance of high-energy activation directions and introduces an ali
    
[^26]: BackDFL：去中心化联邦学习中后门攻击与防御的统一基准

    BackDFL: A Unified Benchmark For Backdoor Attacks and Defenses In Decentralized Federated Learning

    [https://arxiv.org/abs/2608.21137](https://arxiv.org/abs/2608.21137)

    本文指出去中心化联邦学习（DFL）的鲁棒性被高估，并提出了BackDFL，一个统一基准，用于在现实和适应性后门攻击下系统评估DFL的安全性。

    

    去中心化联邦学习（DFL）通过用点对点模型交换替代集中式参数服务器，承诺实现无需信任的协作学习。然而，这种架构转变从根本上重塑了威胁格局。在没有全局协调聚合的情况下，DFL特别容易受到后门攻击，其中恶意参与者植入持久隐藏行为，同时保持较高的清洁任务性能。在本文中，我们认为DFL的鲁棒性被显著高估了。现有研究依赖于简化的威胁模型、非适应性攻击者、碎片化的评估协议、不一致的通信拓扑和临时训练配置，导致对DFL安全性的理解不完整。为解决这些限制，我们提出了BackDFL，一个统一基准，用于在现实和适应性后门攻击下系统评估DFL。通过可扩展的...

    arXiv:2608.21137v1 Announce Type: new  Abstract: Decentralized Federated Learning (DFL) promises trust-free collaborative learning by replacing the centralized parameter server with peer-to-peer model exchange. However, this architectural shift fundamentally reshapes the threat landscape. Without globally coordinated aggregation, DFL becomes particularly susceptible to backdoor attacks, in which malicious participants implant persistent hidden behaviors while maintaining high clean-task performance. In this paper, we argue that the robustness of DFL has been significantly overestimated. Existing studies rely on simplified threat models, non-adaptive adversaries, fragmented evaluation protocols, inconsistent communication topologies, and ad hoc training configurations, leading to an incomplete understanding of DFL security. To address these limitations, we present BackDFL, a unified benchmark for systematically evaluating DFL under realistic and adaptive backdoor attacks. Through extens
    
[^27]: 移动端大模型：视觉语言模型的高效2.7位量化

    Llama-Mobile: Efficient 2.7-Bit Quantization of VLMs

    [https://arxiv.org/abs/2608.21134](https://arxiv.org/abs/2608.21134)

    提出一种无需训练数据的2.7位量化框架，将视觉语言模型压缩至3.7 GB，同时保持视觉问答性能，适用于移动设备高效推理。

    

    摘要：由于视觉语言模型（VLMs）在部署到移动设备时面临显著的内存和计算需求挑战，我们提出了一种用于在资源受限硬件上高效推理的VLM量化框架。我们的方法结合了一个量化流程，该流程利用模型自身生成训练数据，无需访问训练设置，并采用一种新颖的每参数2.7位格式，支持在Arm CPU上高效执行。我们通过将Llama 3.2 11B视觉指令模型压缩至3.7 GB（使用8位激活），在标准视觉问答任务集上保持强大性能，验证了我们的方法。

    arXiv:2608.21134v1 Announce Type: cross  Abstract: Deploying vision-language models (VLMs) on mobile devices is challenging due to their significant memory and compute requirements. We present a framework for quantizing VLMs for efficient inference on resource-constrained hardware. Our approach combines a quantization pipeline that uses the model itself to generate training data and does not require access to the training setup, with a novel 2.7-bit-per-parameter format supporting efficient execution on Arm CPUs. We validate our approach by compressing the Llama 3.2 11B Vision Instruct model to 3.7 GB with 8-bit activations, preserving strong performance on a set of standard visual question answering tasks.
    
[^28]: FlatLand：通过定制洛伦兹空间实现个性化图联邦学习

    FlatLand: Personalized Graph Federated Learning via Tailored Lorentz Space

    [https://arxiv.org/abs/2608.21096](https://arxiv.org/abs/2608.21096)

    本文提出FlatLand，通过将客户端数据嵌入定制的双曲洛伦兹空间，并采用参数解耦策略分离异质与共性信息，解决了图联邦学习中客户端数据异构的挑战。

    

    联邦学习实现了隐私保护的协作训练，但高度异构的客户端数据仍然具有挑战性，尤其是在图联邦学习中，客户端拥有结构多样的图数据。现有的个性化联邦学习方法忽略了不同图结构的内在几何特性。我们提出了FlatLand，一种新颖的个性化联邦学习方法，它将不同客户端的数据嵌入到双曲几何的定制洛伦兹空间中。我们的关键洞察是，双曲几何自然适应了现实世界图中普遍存在的内在负曲率，而洛伦兹空间中的类时维度提供了一种有原则的方式来编码客户端特定的异质性。我们开发了一种参数解耦策略，将异质信息（捕获在类时参数中）与共同知识（保留在类空参数中）分离，从而实现直接聚合。

    arXiv:2608.21096v1 Announce Type: new  Abstract: Federated learning enables privacy-preserving collaborative training, but highly heterogeneous client data remain challenging, especially in graph federated learning where clients possess structurally diverse graphs. Existing personalized federated learning (PFL) methods ignore the intrinsic geometric properties of diverse graph structures. We propose FlatLand, a novel personalized federated learning method that embeds different clients' data in tailored Lorentz space of hyperbolic geometry. Our key insight is that hyperbolic geometry naturally accommodates the intrinsic negative curvature prevalent in real-world graphs, while the time-like dimension in Lorentz space provides a principled way to encode client-specific heterogeneity. We develop a parameter decoupling strategy that separates heterogeneous information (captured in time-like parameters) from common knowledge (preserved in space-like parameters), enabling direct aggregation w
    
[^29]: 通过自适应大语言模型提议的不良妊娠结局因果建模

    Causal Modeling of Adverse Pregnancy Outcomes via Adaptive LLM Proposals

    [https://arxiv.org/abs/2608.21079](https://arxiv.org/abs/2608.21079)

    本文提出了一种结合LLM先验知识和经验数据评分的神经符号框架，通过自适应提议机制迭代生成不良妊娠结局的因果假设，以应对数据稀缺和领域知识不完整的挑战。

    

    arXiv:2608.21079v1 公告类型：新 摘要：不良妊娠结局（APOs），如早产和妊娠糖尿病，可能对母亲和儿童产生长期后果，但其病因仍难以捉摸。该领域的因果发现尤其具有挑战性，因为数据匮乏且领域知识不完整。因此，纯数据驱动方法失败，而大型语言模型（LLM）的输出不一致或矛盾。我们引入了一个神经符号框架，用于生成合理的因果假设，该框架迭代地结合了LLM的广泛先验知识与数据的经验评分。我们的方法将LLM视为自适应提议分布，生成针对经验数据评分的假设；然后，所得的高分图用于更新LLM的上下文，引导后续生成朝向假设空间中更有前景的区域。我们在一个真实世界的临床数据上评估了我们的方法。

    arXiv:2608.21079v1 Announce Type: new  Abstract: Adverse Pregnancy Outcomes (APOs) such as preterm birth and gestational diabetes can have long-term consequences for both the mother and child, yet an understanding of their causes remains elusive. Causal discovery in this domain is especially challenging due to a paucity of data and incomplete domain knowledge. As a result, pure data-driven methods fail, and Large Language Model (LLM) outputs remain inconsistent or contradictory. We introduce a neurosymbolic framework for generating plausible causal hypotheses that iteratively combines the broad prior knowledge of LLMs with empirical scoring on data. Our method treats the LLM as an adaptive proposal distribution, generating hypotheses that are scored against empirical data; the resulting high-scoring graphs are then used to update the LLM's context, steering subsequent generations toward more promising regions of the hypothesis space. We evaluate our approach on a real-world clinical da
    
[^30]: AudioWorldSim：面向世界模型的真实双耳音频数据集

    AudioWorldSim: Realistic Binaural Audio Datasets For World Models

    [https://arxiv.org/abs/2608.21075](https://arxiv.org/abs/2608.21075)

    AudioWorldSim通过扩展SoundSpaces 2.0并改进连续声音合成，自动生成真实双耳音频数据集，为世界模型研究提供了开源、可复现的平台。

    

    本技术报告介绍了AudioWorldSim，这是一个开源平台，旨在生成真实的双耳音频数据集，并推动基于音频的机器学习研究，特别是世界模型领域。作为Meta的SoundSpaces 2.0平台的自定义扩展，AudioWorldSim利用了其全面的声学框架，但专注于随机代理导航的自动展开，并实现了对连续声音合成方式的关键修复。AudioWorldSim已公开发布给研究社区，网址为https://github.com/Luizerko/AudioWorldSim，以促进可复现性。

    arXiv:2608.21075v1 Announce Type: cross  Abstract: This technical report presents AudioWorldSim, an open-source platform designed to generate realistic binaural audio datasets and advance research in audio-based machine learning, particularly world models. Built as a custom extension of Meta's SoundSpaces 2.0 platform, AudioWorldSim leverages their comprehensive acoustics framework, but focuses on the automatic rollout of random agent navigations, as well as implements crucial fixes to how continuous sound is composed. AudioWorldSim is made publicly available to the research community at https://github.com/Luizerko/AudioWorldSim to facilitate reproducibility.
    
[^31]: TracingFlow：一种基于二阶动力学的免模拟轨迹推断框架

    TracingFlow: A Simulation-Free Trajectory Inference Framework Based on Second-Order Dynamics

    [https://arxiv.org/abs/2608.21070](https://arxiv.org/abs/2608.21070)

    TracingFlow通过引入二阶动力学和神经网络回归加速度场，提出了一种免模拟的流匹配框架，能够精确高效地解决动态最优加速度传输问题，从而捕捉高曲率和非线性演化轨迹。

    

    arXiv:2608.21070v1 公告类型：交叉 摘要：从稀疏时间快照中推断连续系统演化是生成建模和单细胞组学中的一个关键挑战。尽管最优传输（OT）很受欢迎，但现有框架大多局限于一阶动力学，假设无记忆速度场。这限制了表达能力，因为一阶系统无法解释细胞分化等过程中固有的调节动量和延迟响应。在此，我们引入了TracingFlow，一种免模拟的流匹配框架，将其推广到二阶动力学。通过使用神经网络回归加速度场，TracingFlow为动态最优加速度传输（DOAT）问题提供了精确、高效的解决方案。与产生过度平滑轨迹的一阶方法不同，我们的二阶公式通过学习潜在的力场来捕捉高曲率转变和非线性演化。在复杂数据集上评估时，该方法表现出优越性能。

    arXiv:2608.21070v1 Announce Type: cross  Abstract: Inferring continuous system evolution from sparse temporal snapshots is a key challenge in generative modeling and single-cell omics. While Optimal Transport (OT) is popular, existing frameworks are largely restricted to first-order dynamics, assuming memoryless velocity fields. This limits expressiveness, as first-order systems fail to account for regulatory momentum and time-delayed responses inherent in processes like cell differentiation. Here, we introduce TracingFlow, a simulation-free Flow Matching framework generalizing to second-order dynamics. By using neural networks to regress the acceleration field, TracingFlow provides an exact, efficient solution to the Dynamical Optimal Acceleration Transport (DOAT) problem. Unlike first-order methods yielding over-smoothed trajectories, our second-order formulation captures high-curvature transitions and nonlinear evolutions by learning the underlying force fields. Evaluated on complex
    
[^32]: 设计一个基于人类对齐的鲁棒性LLM评估系统，用于药物发现中的智能体AI

    Designing a Robust LLM-Based Evaluation System for Agentic AI in Drug Discovery Through Human Alignment

    [https://arxiv.org/abs/2608.21057](https://arxiv.org/abs/2608.21057)

    本文提出了一种通过人类对齐验证的LLM-as-a-Judge评估框架，为药物发现智能体系统定义了四个质量维度并验证了评判器的可靠性，以解决现有评估方法无法捕捉语义正确性和扩展性问题。

    

    arXiv:2608.21057v1 公告类型：新 摘要：智能体大型语言模型（LLM）系统正在重塑化学和药物发现中的科学工作流程，但评估其开放式、工具增强的输出仍然是一个根本性瓶颈。基于参考的指标如BLEU和ROUGE无法捕捉语义正确性，而专家人工评估无法扩展到这些系统所需的迭代速度。LLM-as-a-Judge范式已成为一种可扩展的替代方案，但现有的药物发现基准在部署LLM评判器时，并未验证其与人类专家的一致性。在本工作中，我们为ChatInvent（阿斯利康部署的智能体药物发现助手）提出了一种LLM-as-a-Judge评估框架，包含四项贡献。首先，我们定义了四个输出质量评估维度——完整性、相关性、结构清晰度和范围遵循性——以及确定性的工具调用正确性检查。其次，我们通过人类对齐验证了评判器。

    arXiv:2608.21057v1 Announce Type: new  Abstract: Agentic large language model (LLM) systems are reshaping scientific workflows in chemistry and drug discovery, but evaluating their open-ended, tool-augmented outputs remains a fundamental bottleneck. Reference-based metrics such as BLEU and ROUGE fail to capture semantic correctness, while expert human evaluation does not scale to the iteration speed these systems demand. The LLM-as-a-Judge paradigm has emerged as a scalable alternative, but existing drug discovery benchmarks deploy LLM judges without validating their alignment with human experts. In this work, we present an LLM-as-a-Judge evaluation framework for ChatInvent, an agentic drug discovery assistant deployed at AstraZeneca, with four contributions. First, we define four output-quality evaluation dimensions---Completeness, Relevancy, Structural Clarity, and Scope Adherence---alongside deterministic Tool Call Correctness checks. Second, we validate the judge through a human al
    
[^33]: COMET：面向视频多模态大语言模型的对比运动增强时序推理

    COMET: Contrastive Motion-Enhanced Temporal Reasoning for Video Multimodal Large Language Models

    [https://arxiv.org/abs/2608.21030](https://arxiv.org/abs/2608.21030)

    COMET通过引入基于泰勒帧差分的运动分支和时序注意力偏置增强交叉注意力，并采用时序先验蒸馏与TC-GRPO优化，系统性地解决了视频多模态大语言模型在细粒度运动时序理解上的不足。

    

    arXiv:2608.21030v1 公告类型：交叉 摘要：视频多模态大语言模型已取得显著进展，但细粒度的运动-时序理解仍然脆弱。核心瓶颈不仅在于稀疏帧采样，还在于缺乏完整的时序建模流程，无法显式表示帧间变化、实现外观-运动交互，并优化时序方向敏感性。我们提出COMET，一个时序接地框架，通过显式时序表示、外观-运动融合和方向感知优化，系统性地增强视频MLLMs。在架构上，COMET引入基于泰勒帧差分的时序运动分支，并通过时序注意力偏置增强的交叉注意力将运动证据注入外观流。在优化方面，COMET结合时序先验蒸馏与正向-反向TC-GRPO阶段，将时序顺序转化为直接学习信号，并显著提升性能。

    arXiv:2608.21030v1 Announce Type: cross  Abstract: Video multimodal large language models have advanced significantly, yet fine-grained motion-temporal understanding remains fragile. The core bottleneck is not only sparse frame sampling, but also the lack of a complete temporal modeling pipeline for explicitly representing frame-to-frame change, enabling appearance-motion interaction, and optimizing temporal direction sensitivity. We propose COMET, a temporally grounded framework that systematically strengthens video MLLMs through explicit temporal representation, appearance-motion fusion, and direction-aware optimization. Architecturally, COMET introduces a temporal motion branch built on Taylor frame differences and injects its motion evidence into the appearance stream via temporal attention bias-enhanced cross-attention. For optimization, COMET combines temporal prior distillation with a forward-reverse TC-GRPO stage that turns temporal order into a direct learning signal and stren
    
[^34]: RODE：一种用于优化的径向-正交解耦引擎

    RODE: A Radial-Orthogonal Decoupled Engine for Optimization

    [https://arxiv.org/abs/2608.21024](https://arxiv.org/abs/2608.21024)

    RODE通过将矩阵优化分解为径向范数更新和正交方向更新，实现了更优的性能和更低的模型范数，并在多种任务中超越Muon优化器。

    

    现代神经网络训练越来越多地使用矩阵感知优化器，然而它们的有条件矩阵步骤通常直接添加到权重上，共同改变其范数和方向。这种相互作用很重要，因为当前范数决定了角运动，而方向学习可以驱动范数增长，从而改变后续步骤。我们引入了RODE，它为径向和方向组件提供了独立的更新规则和步长。RODE通过标量径向规则显式更新矩阵的Frobenius范数，而其方向通道在切线空间中进行Newton-Schulz条件更新。受控的GPT-2干预实验显示，直接范数控制和RODE的方向更新均带来了收益。在两个语言建模和两个图像分类任务中，RODE在每次直接比较中均优于两种Muon变体，并以更低的完整模型范数结束。在1.5B规模下，使用迁移的学习率时，RODE也表现出优势。

    arXiv:2608.21024v1 Announce Type: new  Abstract: Modern neural network training increasingly uses matrix-aware optimizers, yet their conditioned matrix step is typically added directly to the weight, jointly changing its norm and direction. This interaction matters because the current norm determines angular motion, while directional learning can drive norm growth and thereby alter later steps. We introduce RODE, which gives the radial and directional components separate update rules and step sizes. RODE explicitly updates the matrix Frobenius norm through a scalar radial rule, while its directional channel performs Newton--Schulz-conditioned updates in the tangent space. Controlled GPT-2 interventions show gains from both direct norm control and RODE's directional update. Across two language-modeling and two image-classification tasks, RODE outperforms both Muon variants in every direct comparison and ends with lower full-model norms. At 1.5B scale, using the learning rate transferred
    
[^35]: 从静态多级小语义码本到动态单级大语义码本用于生成式推荐

    From a Static Multi-Level Small Semantic Codebook to a Dynamic Single-Level Large Semantic Codebook for Generative Recommendation

    [https://arxiv.org/abs/2608.21012](https://arxiv.org/abs/2608.21012)

    本文提出了一种单级大语义码本和曝光感知动态更新机制，用单一语义标记替代多级残差量化，以减少解码成本并适应流量变化，提升生成式推荐的效率和准确性。

    

    arXiv:2608.21012v1 公告类型：交叉 摘要：生成式推荐将每个项目表示为一个离散语义ID（SID）序列，并预测该序列以检索下一个项目。典型系统使用多级残差量化，这增加了自回归解码成本，并创建了一个可能稀疏占用的大层级空间。静态码本也会随着新项目到达和曝光分布变化而与当前流量错位。我们提出了一种单级大语义码本，用一个语义标记替换多个残差语义代码，同时保留一个单独的协同消歧标记以减少项目碰撞。我们进一步引入了一种基于时间权重衰减、指数移动平均中心更新和SID变化曝光加权惩罚的曝光感知动态更新机制。我们还开发了一个离线评估框架，涵盖表示质量、代码利用率、集群负载、完整度。

    arXiv:2608.21012v1 Announce Type: cross  Abstract: Generative recommendation represents each item with a sequence of discrete Semantic IDs (SIDs) and predicts the sequence to retrieve the next item. Typical systems use multi-level residual quantization, which increases autoregressive decoding cost and creates a large hierarchical space that may be sparsely occupied. Static codebooks also become misaligned with current traffic as new items arrive and exposure distributions change. We propose a single-level large semantic codebook that replaces multiple residual semantic codes with one semantic token while retaining a separate collaborative disambiguation token to reduce item collisions. We further introduce an exposure-aware dynamic update mechanism based on temporal weight decay, exponential moving-average center updates, and an exposure-weighted penalty on SID changes. We also develop an offline evaluation framework covering representation quality, code utilization, cluster load, full
    
[^36]: 自由概率核用于储层计算中零回放超参数选择

    Free-Probability Kernels for Zero-Rollout Hyperparameter Selection in Reservoir Computing

    [https://arxiv.org/abs/2608.20998](https://arxiv.org/abs/2608.20998)

    本文提出一种基于自由概率的确定性核方法，通过短先导序列即可选择储层计算超参数，无需任何回放，显著降低计算成本且性能不降。

    

    储层计算（RC）将固定的循环动力系统与训练好的轻量级读出器相结合，但这种效率在超参数选择过程中部分丧失：循环增益、输入尺度和泄漏率决定了储层的稳定性和时间处理机制，通常需要通过多次回放来调整。我们引入了一种确定性的、基于先导信息的选择器，适用于具有逐坐标非线性特征的泄漏线性储层。自由概率导出了跨滞后传播系数，这些系数总结了储层如何混合过去的输入。在大宽度极限下，这些系数定义了一个确定性时间核，近似于有限储层特征几何结构。因此，在短标记先导序列上进行的核岭回归可以排序候选工作区域，而无需实例化或回放储层，且所选配置可跨宽度迁移。在十个合成时间序列上，该方法的性能优于或与基线方法相当，同时大幅降低了计算成本。

    arXiv:2608.20998v1 Announce Type: new  Abstract: Reservoir computing (RC) couples a fixed recurrent dynamical system with a trained lightweight readout, but this efficiency is partly lost during hyperparameter selection: the recurrent gain, input scale, and leakage rate determine the reservoir's stability and temporal processing regime and are usually tuned through many rollouts. We introduce a deterministic, pilot-informed selector for leaky linear reservoirs followed by coordinate-wise nonlinear features. Free probability yields cross-lag propagation coefficients that summarize how the reservoir mixes past inputs. In the large-width limit, these coefficients define a deterministic temporal kernel that approximates the finite-reservoir feature geometry. Kernel ridge regression on a short labelled pilot sequence therefore ranks candidate operating regimes without instantiating or rolling out a reservoir, and the selected configuration transfers across widths. Across ten synthetic tempo
    
[^37]: 破坏对齐：针对图基础模型的隐蔽后门攻击

    Trojaning the Alignment: Stealthy Backdoor Attacks against Graph Foundation Models

    [https://arxiv.org/abs/2608.20991](https://arxiv.org/abs/2608.20991)

    本文揭示了图基础模型在图-语言对齐下的后门攻击漏洞，提出一种同时利用图和文本模态的隐蔽后门攻击方法，以绕过现有防御并实现有效攻击。

    

    arXiv:2608.20991v1 公告类型：新 摘要：基于文本属性图（TAGs）的图基础模型（GFMs）将图表示与语言语义对齐，以支持可迁移的图学习。尽管有这些优势，GFMs在TAGs上的后门漏洞仍未得到充分理解，尤其是在图-语言对齐下，其中图和文本表示在共享语义空间中被训练为相互约束。现有的后门攻击主要针对图侧或文本侧，将两种模态独立处理。这使得直接适应无效：仅图触发器可能被干净的文本语义所限制，而仅文本触发器会改变语言视图，但不会直接移动被对齐和评分的图表示。TAGs还带来了隐蔽性挑战，因为触发器同时暴露为节点文本和局部图结构，使得不一致的触发器属性或异常子图容易检查和过滤。

    arXiv:2608.20991v1 Announce Type: new  Abstract: Graph Foundation Models (GFMs) on text-attributed graphs (TAGs) align graph representations with language semantics to support transferable graph learning. Despite these advantages, the backdoor vulnerability of GFMs on TAGs remains insufficiently understood, especially under graph-language alignment, where graph and text representations are trained to constrain each other in a shared semantic space. Existing backdoor attacks mainly target either the graph side or the text side, treating the two modalities independently. This makes direct adaptation ineffective: graph-only triggers can be constrained by clean text semantics, while text-only triggers alter the language view but do not directly shift the graph representation being aligned and scored. TAGs also impose a stealth challenge because triggers are exposed as both node text and local graph structure, making incoherent trigger attributes or anomalous subgraphs easy to inspect or fi
    
[^38]: 基于雅可比引导的噪声注入用于大型语言模型的量化鲁棒性

    Jacobian-guided Noise Injection for Quantization Robustness in Large Language Models

    [https://arxiv.org/abs/2608.20988](https://arxiv.org/abs/2608.20988)

    本文提出一种基于雅可比范数引导的噪声注入训练策略，通过抑制softmax雅可比矩阵的范数来增强大型语言模型在量化中的鲁棒性，优于现有后训练量化方法。

    

    大型语言模型（LLM）的量化常因自注意力机制对离散化误差的敏感性而受阻。我们识别出softmax算子因对异常值和状态依赖的雅可比矩阵敏感，成为量化稳定性的瓶颈。我们从理论上证明，抑制该雅可比矩阵的范数有助于限制量化引起的性能下降。基于此，我们提出雅可比引导的噪声注入，这是一种训练策略，将零均值高斯噪声注入注意力前的logits，其方差直接由雅可比矩阵的Frobenius范数推导。与依赖启发式方法或直接惩罚雅可比矩阵的先前方法不同，我们的方法基于局部注意力敏感性提供了一种识别最优噪声方差的方式。我们在最先进的LLM架构上评估了该方法，显示出相对于流行PTQ方法的改进鲁棒性。实证分析进一步验证了其有效性。

    arXiv:2608.20988v1 Announce Type: cross  Abstract: Quantization of Large Language Models (LLMs) is often hindered by the sensitivity of the self-attention mechanism to discretization errors. We identify the softmax operator as a bottleneck for quantization stability due to its sensitivity to outliers and state-dependent Jacobian. We theoretically establish that suppressing the norm of this Jacobian helps in bounding quantization-induced performance degradation. Based on this, we propose Jacobian-Guided Noise Injection, a training strategy that injects zero-mean Gaussian noise into pre-attention logits, with variance derived directly from the Jacobian Frobenius norm. Unlike prior approaches that rely on heuristic or penalise jacobian directly, our method provides a way to identify the optimal noise variance based on the local attention sensitivity. We evaluate the method on SOTA LLM architectures, where it demonstrates improved robustness over popular PTQ methods. Empirical analysis rev
    
[^39]: 对时空预测基准数据集和基线的批判性审计

    A Critical Audit of Spatiotemporal Forecasting Benchmark Datasets and Baselines

    [https://arxiv.org/abs/2608.20980](https://arxiv.org/abs/2608.20980)

    本文通过经典时间序列方法分析常用时空基准数据集，揭示无空间感知的线性模型比以往报告更具竞争力，质疑了现有基准数据集的判别可靠性。

    

    arXiv:2608.20980v1 公告类型：新 摘要：图神经网络（GNNs）通常用于具有空间图结构的多元时间序列的短期预测。尽管存在许多替代数据集，但该领域的方法创新主要针对一组有限的基准数据集进行评估，最著名的是Chickenpox、PedalMe、WikiMaths、METR-LA和PEMS-BAY。评估协议包含从历史平均值到经典机器学习方法的基线。这些基线通常表现出与GNNs相当的性能。在本研究中，我们退一步，通过经典时间序列方法分析基准数据集，以揭示为什么无空间感知的线性模型比先前报道的更具竞争力，从而进一步质疑上述广泛采用的数据集的判别可靠性。我们的统计分析提供了一套工具集，用于识别显著的...

    arXiv:2608.20980v1 Announce Type: new  Abstract: Graph neural networks (GNNs) are routinely employed for short-range forecasting on multivariate time series with a spatial graph structure. Despite the availability of many alternative datasets, method innovations within this domain are predominantly assessed against a rather limited set of benchmark datasets, most notably Chickenpox, PedalMe, WikiMaths, METR-LA, and PEMS-BAY. The evaluation protocols contain baselines spanning from historical averages to classical machine learning approaches. These baselines often show competitive performance compared to GNNs. In the present work, we take a step back and analyse the benchmark datasets via classical time series methods to uncover why spatially-unaware linear models pose a stronger competitor than previously reported, casting further doubt on the discriminative reliability of the aforementioned widely adopted datasets. Our statistical analysis provides a toolset for identifying significan
    
[^40]: 使用精确室内声学模拟训练DeepFilterNet可改善单通道语音增强

    Training DeepFilterNet with Accurate Room Acoustic Simulations Improves Single-Channel Speech Enhancement

    [https://arxiv.org/abs/2608.20971](https://arxiv.org/abs/2608.20971)

    本文发现使用高保真度声学模拟生成的房间冲激响应数据集训练DeepFilterNet3，能显著提升语音增强性能和降低语音识别错误率，尽管增益未归因于特定模拟组件。

    

    我们研究了合成房间冲激响应（RIR）数据集的真实感如何影响DeepFilterNet3在单通道语音增强任务中的训练效果。我们比较了一个基于DNS4图像源方法（ISM）的RIR数据集与一个使用混合波动声学和几何声学模拟生成的高声学保真度数据集。我们没有孤立地分析单个模拟因素，而是在保持增强模型不变的情况下，比较完整的RIR生成流程。模型在未见的实测RIR上使用客观语音增强指标和下游自动语音识别（ASR）进行评估。与ISM数据集相比，使用高保真度数据集进行训练在客观指标上始终带来适度改进，并显著降低了ASR词错误率。尽管实验未将这些改进归因于单个建模组件，但它们表明，提高合成声学训练数据的整体真实感对性能有积极影响。

    arXiv:2608.20971v1 Announce Type: cross  Abstract: We investigate how the realism of synthetic room impulse response (RIR) datasets affects the training of DeepFilterNet3 for single-channel speech enhancement. We compare a DNS4 image-source-method (ISM) RIR dataset with a higher-acoustic-fidelity dataset generated using hybrid wave-based and geometrical acoustics simulation. Rather than isolating individual simulation factors, we compare complete RIR generation pipelines while keeping the enhancement model unchanged. Models are evaluated on unseen measured RIRs using objective speech enhancement metrics and downstream automatic speech recognition (ASR). Training with the higher-fidelity dataset consistently yields modest improvements in objective metrics and substantially lower ASR word error rates than the ISM dataset. Although the experiments do not attribute these gains to individual modelling components, they show that increasing the overall realism of synthetic acoustic training d
    
[^41]: 训练、学习与推理：神经系统的统一动力学

    Training, learning and inference: unified dynamics of neural systems

    [https://arxiv.org/abs/2608.20965](https://arxiv.org/abs/2608.20965)

    本文提出了一种基于生成事实图（GFG）的统一框架，将训练、学习和推理视为神经系统中的动态过程，并通过nanoGPT实验展示了其可观察性。

    

    arXiv:2608.20965v1 公告类型：新公告  摘要：我们定义了一个原子生成事实 f=(u,tau,omega,z;rho)，记录起源、实现变换、具体发生、生成结果和关系角色。将这些事实编译成生成事实图（GFG），该图提供了一种AI原生、可编译的科学事实基底，保留生成历史。我们建立了一种基于GFG的递归科学过程，其中分析、干预、重放和验证为后续循环形成事实。使用nanoGPT，我们建立了统一的训练-学习动力学。训练是参数-优化器系统状态和记忆的演化：每个实际训练动作进入接收状态，并产生由该状态和目标特定更新几何条件决定的有限振幅非线性功能响应。学习是由这些响应引起的分布式功能支持的持续重组；能力形成、维持、衰退或恢复在时间上变得可观察。

    arXiv:2608.20965v1 Announce Type: new  Abstract: We define an atomic generation fact f=(u,tau,omega,z;rho), recording the origin, realized transformation, concrete occurrence, generated result and relation role. Compiled into a Generation-Fact Graph (GFG), these facts provide an AI-native, compilable scientific fact substrate preserving generation histories. We establish a GFG-based recursive scientific process in which analysis, intervention, replay and validation form facts for later cycles. Using nanoGPT, we establish unified training-learning dynamics. Training is the evolution of a parameter-optimizer system with state and memory: each actual training action enters the receiving state and produces a finite-amplitude nonlinear functional response conditioned by that state and target-specific update geometry. Learning is the persistent reorganization of distributed functional support by these responses; capability formation, maintenance, decline or recovery becomes observable when t
    
[^42]: TreeWY：面向门控DeltaNet混合模型的自适应验证方法

    TreeWY: Speculative Verification for Gated DeltaNet Hybrids

    [https://arxiv.org/abs/2608.20961](https://arxiv.org/abs/2608.20961)

    本文提出TreeWY方法，通过树状WY变换消除推测解码中的状态快照，显著降低内存开销，从而支持高接受率的宽草稿树。

    

    arXiv:2608.20961v1 公告类型：新论文  摘要：现代开源模型多为混合架构：大部分层采用线性注意力（门控DeltaNet，GDN）层，这些层携带一个小的固定大小的循环状态，而非不断增长的键值（KV）缓存。这使得普通解码在内存上高效，但不利于推测解码。为了验证一批草稿标记并回滚被拒绝的标记，当前系统在GDN层的每个草稿位置都会对完整循环状态进行快照，而这些快照无法在草稿树的分支间共享，因此宽且高接受率的树在内存上变得不可行。我们移除了快照。利用门控Delta规则的一种树状WY变换，我们通过一次三角求解计算每个草稿节点的输出，并在提交时仅重建一个被接受的状态，存储一个小型的伪值矩阵而非逐节点状态；该推导仅依赖于门控Delta规则，而不涉及任何其他架构细节。在两个规模的服务器基准测试中...

    arXiv:2608.20961v1 Announce Type: new  Abstract: Modern open models are hybrids: most layers are linear-attention (Gated DeltaNet, GDN) layers carrying a small fixed-size recurrent state instead of a growing key-value (KV) cache. This makes ordinary decoding memory-efficient, but hurts speculative decoding. To verify a batch of draft tokens and then roll back the rejected ones, today's systems snapshot the full recurrent state at every draft position for GDN layers, and those snapshots cannot be shared across branches of a draft tree, so a wide, high-acceptance tree becomes memory-infeasible. We remove the snapshots. Using a tree-structured WY transform of the gated delta rule, we compute every draft node's output with a single triangular solve and reconstruct only the one accepted state on commit, storing a small pseudo-value matrix instead of per-node states; the derivation depends only on the gated delta rule, not on any other architectural detail. In serving benchmarks on two scale
    
[^43]: 量化感知修复：恢复压缩的4比特大语言模型的实用方法

    Quantization-Aware Healing: A Practical Recipe for Recovering Compressed, 4-Bit LLMs

    [https://arxiv.org/abs/2608.20953](https://arxiv.org/abs/2608.20953)

    提出量化感知修复（QAH）方法，直接从未压缩的原始模型蒸馏4比特学生模型，在显著降低计算成本的同时，在多数基准上匹配或超越bfloat16来源性能。

    

    摘要：以低成本提供大型语言模型越来越意味着发布既在结构上压缩到参数一小部分、又量化到4比特的模型。这些步骤共同削弱了推理、数学、编码和长上下文行为，以至于在部署前需要恢复或修复阶段。默认的方法——量化感知训练（QAT）——重新拟合压缩、量化模型到硬标签；在我们的流程中，它收敛缓慢并在达到峰值后崩溃。我们转而采用量化感知修复（QAH）。由于结构压缩的模型从未在全精度下独立训练，其bfloat16检查点是一个通过蒸馏恢复的原始模型近似；QAH直接从未压缩的原始模型中蒸馏4比特学生模型。在GPT-OSS 120B到60B到MXFP4流程中，QAH学生模型在9个基准测试中的7个上匹配或超越了其bfloat16来源，同时计算量大约减少4倍。

    arXiv:2608.20953v1 Announce Type: cross  Abstract: Serving large language models cheaply increasingly means shipping models that are both structurally compressed to a fraction of their parameters and quantized to 4 bits. Together these steps degrade reasoning, mathematics, coding, and long-context behavior enough to require a recovery, or healing, stage before deployment. The default recipe, quantization-aware training (QAT), re-fits the compressed, quantized model to hard labels; in our pipeline it converged slowly and collapsed past its peak. We adopted Quantization-Aware Healing (QAH) instead. Because a structurally compressed model is never independently trained at full precision, its bfloat16 checkpoint is a distillation-recovered approximation of the original; QAH distills the 4-bit student directly from the original, uncompressed model. On a GPT-OSS 120B to 60B to MXFP4 pipeline, the QAH student matches or beats its bfloat16 source on 7 of 9 benchmarks at roughly 4 times less we
    
[^44]: 离线强化学习的策略解耦提取

    Decoupling Policy Extraction for Offline Reinforcement Learning

    [https://arxiv.org/abs/2608.20909](https://arxiv.org/abs/2608.20909)

    本文提出了一种离线强化学习的策略解耦提取方法，旨在解决传统联合训练范式因固定数据导致的行动漂移和评论家过度估计问题。

    

    离线强化学习方法通常联合训练演员和评论家，其中评论家用于引导演员朝向更高价值的行动。这种耦合学习过程在在线强化学习中具有良好动机，因为改进的演员可以收集新数据，进一步更新演员和评论家。然而，在离线强化学习中，训练数据保持固定，使得演员侧的策略改进无法生成新数据来验证或纠正评论家。此外，保留这种耦合范式会导致两个相关挑战。首先，演员更新可能漂移向高价值但可能超出分布（OOD）的行动，并放大评论家的过度估计。其次，保守的价值估计或行为克隆正则化在抑制OOD行动和数据支持区域内选择高价值行动之间创造了难以权衡的取舍。受此观察的启发，我们重新审视传统离线强化学习范式，并提出d（此处截断）。

    arXiv:2608.20909v1 Announce Type: new  Abstract: Offline RL methods commonly jointly train the actor and critic, where the critic is used to guide the actor toward higher-value actions. This coupled learning process is well motivated in online RL, where an improved actor collects new data that can further update the actor and the critic. However, training data remains fixed in offline RL, making actor-side policy improvement unable to generate new data to validate or correct the critic. Moreover, retaining this coupled paradigm leads to two related challenges. Firstly, actor updates can drift toward high-valued but potentially out-of-distribution (OOD) actions and amplify critic overestimation. Secondly, conservative value estimation or behavior-cloning regularization creates a difficult trade-off between suppressing OOD actions and selecting high-value actions within the data-supported region. Motivated by this observation, we revisit the conventional offline RL paradigm and propose d
    
[^45]: EviRank：面向多模态图像重排序的结构化相关性证据

    EviRank: Structured Relevance Evidence for Multimodal Image Re-ranking

    [https://arxiv.org/abs/2608.20886](https://arxiv.org/abs/2608.20886)

    EviRank通过将多模态图像重排序转化为基于六类语义槽的约束满足问题，并利用无需训练的确定性评分和证据依据比较，解决了现有方法在细粒度约束上的遗漏或幻觉问题。

    

    arXiv:2608.20886v1 公告类型：交叉 摘要：真实世界的图像搜索查询是多模态且组合性的：“找到这件衬衫的粉色款”指定了要保留的实体、要修改的属性以及要忽略的上下文。然而，现有的重排序器要么将这种多方面的相关性压缩成一个不透明的嵌入，要么依赖自由形式的思维链，容易遗漏或幻觉化细粒度约束。借鉴NLP中基于评分标准和检查表的评估方法，我们将多模态图像重排序重新定义为语义约束满足问题，并提出EviRank，它将任何查询（仅文本、仅图像或组合查询）解析为一个统一的证据包：跨六个语义槽（例如，实体、属性、关系）的类型化标准，每个标准标记为必需、禁止或可忽略。重排序因此简化为证据条件下的验证，结合确定性评分标准和证据依据的列表式比较，在单一无需训练的过程中完成。明确的证据...

    arXiv:2608.20886v1 Announce Type: cross  Abstract: Real-world image search queries are multimodal and compositional: ``find this shirt in pink'' specifies an entity to retain, an attribute to modify, and context to ignore. Yet existing re-rankers either compress such multifaceted relevance into an opaque embedding or rely on free-form chain-of-thought that easily omits or hallucinates fine-grained constraints. Drawing on rubric- and checklist-based evaluation from NLP, we recast multimodal image re-ranking as a semantic constraint satisfaction problem and propose EviRank, which parses any query - text-only, image-only, or composed - into a unified evidence package: typed criteria across six semantic slots (e.g., entities, attributes, relations), each labelled required, forbidden, or ignorable. Re-ranking then reduces to evidence-conditioned verification, combining deterministic rubric scoring and evidence-grounded listwise comparison in a single training-free procedure. The explicit ev
    
[^46]: 什么都没变，只是模型变了：CellFill——用于量化大语言模型位级一致、可撤销更新的有界单元内学习

    Nothing Changed but the Model: CellFill -- Bounded In-Cell Learning for Bit-Identical, Revocable Updates to Quantized LLMs

    [https://arxiv.org/abs/2608.20873](https://arxiv.org/abs/2608.20873)

    本文提出CellFill方法，在冻结的4位量化模型内部学习残差，实现位级一致的更新，确保模型发布工件逐位不变且更新可撤销，同时保持性能与未受约束模型相当。

    

    arXiv:2608.20873v1 公告类型：新  摘要：教授部署语言模型新知识的每一种方式——全量微调、适配器合并、模型编辑——都会替换发布的检查点，并随之替换所有引用这些精确位元的评估和缓存。我们反而在反量化间隙中学习：在4位发布的整数码和缩放因子冻结的情况下，新知识仅写入每个量化决策单元内部严格存在的逐权重残差中。重新量化后，发布的工件逐位返回，这是一种机器可检查的保证；更新通过丢弃残差即可精确撤销；漂移有界。我们提供了六个命题和三条训练路径，包括CellFill，一种有界的重参数化方法，使不变性成为结构性的而非强制的。精确不变性几乎免费：在三个配对种子上，受约束的密集路径与一个未受约束的参考模型相匹配，而后者的权重可证明地逃离了该单元。

    arXiv:2608.20873v1 Announce Type: new  Abstract: Every way of teaching a deployed language model something new -- full fine-tuning, adapter merging, model editing -- replaces the released checkpoint, and with it every evaluation and cache that referred to those exact bits. We instead learn inside the dequantization gap: with the integer codes and scales of a 4-bit release frozen, new knowledge is written only into the per-weight residual that lives strictly inside each quantization decision cell. Re-quantization then returns the released artifact bit-for-bit, a machine-checkable guarantee; updates are exactly revocable by dropping the residual; and drift is bounded. We give six propositions and three training paths, including CellFill, a bounded reparameterization that makes invariance structural rather than enforced.   Exact invariance turns out to be nearly free: across three paired seeds the constrained dense path matches an unconstrained reference whose weights provably escape the 
    
[^47]: ReCurveflow：一种学习弯曲反应轨迹以预测过渡态几何结构的流匹配框架

    ReCurveflow: A Flow Matching Framework that Learns Curved Reaction Trajectories to Predict Transition State Geometries

    [https://arxiv.org/abs/2608.20869](https://arxiv.org/abs/2608.20869)

    ReCurveflow通过监督弯曲反应路径并引入离路径校正，显著提升了过渡态几何预测的准确性和鲁棒性。

    

    预测化学反应中的过渡态（TS）至关重要，因为它们能揭示反应机理。近期关于TS预测的研究主要集中在流匹配方法上，这些方法监督于直线路径，但与实际反应轨迹不符。我们提出了一种新颖的基于流匹配的框架ReCurveflow，它通过监督从完整NEB衍生分子几何带插值得到的连续弯曲参考路径，来学习预测TS几何结构。我们还引入了离路径校正，这使得ReCurveflow在推理滚动过程中遇到离路径几何状态时，能产生修正速度场，从而增强对暴露偏差的抵抗能力并提高TS预测的准确性。在三种数据分割和六种评估指标下，ReCurveflow在大多数分割-指标组合中优于七种基线方法，取得了最佳结果。定性分析进一步表明，ReCurve能有效捕捉反应轨迹的弯曲特性。

    arXiv:2608.20869v1 Announce Type: new  Abstract: Predicting transition states (TS) in chemical reactions is crucial, as they provide insights into reaction mechanisms. Recent work on TS prediction have focused on flow matching supervised on straight linear paths that do not align with actual reaction trajectories. We propose a novel flow matching-based framework ReCurveflow that learns to predict TS geometries supervised on continuously curved reference paths interpolated from a full NEB-derived band of molecular geometries. We also introduce off-path correction, which grants ReCurveflow with the ability to produce corrective velocity fields when engaged off-path geometry states during inference rollout, leading to better resistance against exposure bias and accuracy in TS prediction. Across three data splits and six evaluation metrics, ReCurveflow achieves the best result on the majority of split-metric combinations against seven baselines. Qualitative analyses further show that ReCur
    
[^48]: 深度强化学习与模型预测控制之间的控制权共享：在多类交通网络中的应用

    Sharing the Control Authority Between Deep Reinforcement Learning and Model Predictive Control: Application to Multi-Class Transportation Networks

    [https://arxiv.org/abs/2608.20858](https://arxiv.org/abs/2608.20858)

    本文提出一种新颖的DRL-MPC混合框架，通过划分控制权，解决多类交通网络中DRL学习能力受限和MPC模型依赖及计算耗时的问题，实现高效实时控制。

    

    交通网络，特别是多类交通网络（即混合车辆类型的网络），是复杂系统，难以控制。近年来，深度强化学习（DRL）通过与环境交互学习控制策略，以及模型预测控制（MPC）利用系统模型优化控制输入，越来越多地被用于交通网络控制。然而，大规模网络中的非线性系统动态和高维状态空间限制了DRL在时间受限训练下的学习能力，并增加了MPC的计算时间，阻碍了在有限计算资源下的实时实现。此外，MPC依赖于精确的网络模型，而这对于多类交通网络等复杂系统往往不可用。本文提出了一种新颖的DRL-MPC框架，用于多类交通网络，该框架划分...

    arXiv:2608.20858v1 Announce Type: cross  Abstract: Transportation networks, in particular multi-class transportation networks (i.e., networks with mixed vehicle types), are complex systems that are challenging to control. Recently, Deep Reinforcement Learning (DRL), which learns control policies from interactions with the environment, and Model Predictive Control (MPC), which uses a system model to optimize control inputs, have been increasingly utilized for transportation network control. However, nonlinear system dynamics and high-dimensional state spaces in large-scale networks limit DRL's learning capacity under time-constrained training and increase MPC's computation time, hindering real-time implementation with limited computational resources. Moreover, MPC depends on an accurate network model, which is often unavailable for complex systems such as multi-class transportation networks. This paper proposes a novel DRL-MPC framework for multi-class transportation networks that divid
    
[^49]: SAC-Copula：通过平滑相关Gumbel场实现扩散语言模型的保质量水印技术

    SAC-Copula: Quality-Preserving Watermarking for Diffusion Language Models via Smooth Correlated Gumbel Fields

    [https://arxiv.org/abs/2608.20839](https://arxiv.org/abs/2608.20839)

    SAC-Copula通过引入基于高斯copula的平滑局部相关Gumbel扰动场，解决了扩散语言模型水印中扰动与解码动态不匹配的问题，实现了生成质量与可检测性的更优平衡。

    

    arXiv:2608.20839v1 公告类型：新 摘要：扩散语言模型（DLMs）的水印技术需要与迭代并行去掩蔽机制兼容，而非自回归解码。现有的基于采样的水印方法通常注入逐位置的独立同分布扰动，这可能与DLM解码动态不匹配，从而降低生成质量。我们提出SAC-Copula，一种基于高斯copula构建的平滑、局部相关Gumbel扰动场的保质量水印方法。我们进一步开发了SAC感知检测器，利用协方差感知滤波和原生样本校准。机制级分析表明，局部相关性降低了潜在扰动的粗糙度，并更好地匹配迭代细化动态。在LLaDA上的实验表明，与现有基线相比，SAC-Copula实现了良好的质量-可检测性权衡。特别是，在Dream-7B及额外数据集上的进一步评估也验证了其有效性。

    arXiv:2608.20839v1 Announce Type: new  Abstract: Watermarking diffusion language models (DLMs) requires mechanisms compatible with iterative parallel unmasking rather than autoregressive decoding. Existing sampling-based watermarking methods typically inject position-wise i.i.d. perturbations, which can be poorly aligned with DLM decoding dynamics and degrade generation quality. We propose SAC-Copula, a quality-preserving watermarking method for DLMs based on smooth, locally correlated Gumbel perturbation fields constructed via a Gaussian copula. We further develop a SAC-aware detector using covariance-aware filtering and native-sample calibration. Mechanism-level analysis shows that local correlation reduces latent perturbation roughness and better matches iterative refinement dynamics. Experiments on LLaDA show that SAC-Copula achieves a favorable quality-detectability trade-off compared with existing baselines. In particular, further evaluations on Dream-7B and additional datasets s
    
[^50]: 基于实地实验数据的旅游轨迹预测大语言模型微调研究

    Fine-tuning LLMs for Tourist Trajectory Prediction using Field Experiment Data

    [https://arxiv.org/abs/2608.20830](https://arxiv.org/abs/2608.20830)

    本研究通过微调大语言模型（Llama-3.1-8B）利用实地实验数据预测游客轨迹，在下一兴趣点预测中达到49.1%准确率，并展现出对雨天等欠采样场景的良好泛化能力。

    

    arXiv:2608.20830v1 公告类型：交叉 摘要：评估旅游目的地的移动干预措施需要预测游客在不同条件下的行为。传统方法面临挑战，因为游客决策高度依赖天气、疲劳等情境因素，而模型无法泛化到未见过的场景。大语言模型通过预训练编码了关于人类行为的常识知识，能够推理依赖情境的决策，同时自然语言表示灵活整合异质信息。对本地轨迹的微调使这种通用理解适应目的地特定模式。我们使用日本和歌山城的566条轨迹验证了该方法。微调后的Llama-3.1-8B模型在下一兴趣点预测上达到49.1%的准确率，并在雨天等欠采样场景中保持强劲性能，展示了有效的泛化能力。这确立了LLM作为高保真行为模型的地位。

    arXiv:2608.20830v1 Announce Type: cross  Abstract: Evaluating mobility interventions at tourist destinations requires predicting visitor behavior under varying conditions. Traditional methods struggle because tourist decisions depend heavily on context like weather and fatigue, yet models cannot generalize to unobserved scenarios. Large Language Models offer a solution by encoding commonsense knowledge about human behavior from pretraining, enabling reasoning about context-dependent decisions, while natural language representation flexibly integrates heterogeneous information. Fine-tuning on local trajectories adapts this general understanding to destination-specific patterns. We validate this approach using 566 trajectories from Wakayama Castle Park, Japan. Our fine-tuned Llama-3.1-8B achieves 49.1% next POI accuracy and maintains strong performance on undersampled scenarios like rainy days, demonstrating effective generalization. This establishes LLMs as high-fidelity behavior models
    
[^51]: 扩展Muon优化器在扩散变换器中的应用

    Scaling Muon for Diffusion Transformers

    [https://arxiv.org/abs/2608.20818](https://arxiv.org/abs/2608.20818)

    本文提出周期行式Muon优化器，通过每K步执行一次完整谱更新和低成本行式更新，在保持Muon优势的同时减少计算和通信开销，显著提升大型扩散变换器的训练效率。

    

    arXiv:2608.20818v1 公告类型：交叉 摘要：矩阵感知优化器Muon通过平衡奇异方向上的更新来改善大型模型训练，但其在大型扩散变换器（DiTs）上的扩展行为及端到端效率仍不明确。我们首先确立了Muon在从1.3B到15B参数的DiTs上的扩展行为，显示其在优化和生成质量方面相对于AdamW的优势在不同模型规模下持续存在。然而，在大规模下，每一步优化中执行的5步Newton-Schulz迭代（NS5）以及全动量物化，引入了大量的计算和通信开销，这可能抵消Muon的步效率优势。我们引入了周期行式Muon（Periodic Row-wise Muon），它每K步执行一次完整的NS5谱更新，并在其余步骤中基于当前动量应用低计算和通信成本的行式约束更新。我们进一步协同设计了一个分布式实现，以优化性能。

    arXiv:2608.20818v1 Announce Type: cross  Abstract: The matrix-aware optimizer Muon improves large model training by balancing updates across singular directions, yet its scaling behavior and end-to-end efficiency on large Diffusion Transformers (DiTs) remain unclear. We first establish Muon's scaling behavior on DiTs from 1.3B to 15B parameters, showing that its optimization and generative quality advantages over AdamW persist across model scales. However, at scale, the 5-step Newton--Schulz iteration (NS5) performed at every optimization step, together with full-momentum materialization, introduces substantial computation and communication overhead that can offset Muon's step-efficiency advantage. We introduce \emph{Periodic Row-wise Muon}, which performs a full NS5 spectral update once every \(K\) steps and applies a low compute and communication cost row-wise constrained update based on the current momentum at the remaining steps. We further co-design a distributed implementation th
    
[^52]: 无限维空间上的分辨率一致贪心神经逼近

    Resolution-Consistent Greedy Neural Approximation on Infinite-Dimensional Spaces

    [https://arxiv.org/abs/2608.20812](https://arxiv.org/abs/2608.20812)

    本文提出了一种在无限维空间上实现分辨率一致的贪心神经逼近方法，将逼近误差分解为坐标截断和有限宽度项，并建立了无维度统计保证。

    

    arXiv:2608.20812v1 公告类型：新论文  摘要：我们为具有无限维输入（通过有限多个坐标观测）的浅层神经模型建立了构造性逼近和学习保证。该分析基于参数归一化的神经字典及其相关的加权变差类。在此类中，逼近误差分解为依赖于分布的坐标截断项和贪心有限宽度项。对于经验回归，一种完全修正的贪心过程产生了总体保证，其统计复杂度在保留的输入分辨率上是一致的。同一框架扩展到Hilbert值响应，且不显式依赖于输出维度。这些无维度陈述是统计性的，而非计算性的：选择新神经元仍需解决非凸参数搜索问题。最近无限维通用逼近结果所依据的准Polish构造提供了基础。

    arXiv:2608.20812v1 Announce Type: new  Abstract: We develop constructive approximation and learning guarantees for shallow neural models with infinite-dimensional inputs observed through finitely many coordinates. The analysis is based on a parameter-normalized neural dictionary and its associated weighted variation class. Within this class, the approximation error separates into a distribution-dependent coordinate-truncation term and a greedy finite-width term. For empirical regression, a fully-corrective greedy procedure yields population guarantees whose statistical complexity is uniform in the retained input resolution. The same framework extends to Hilbert-valued responses without an explicit dependence on the output dimension. The dimension-free statements are statistical, not computational: selecting a new neuron still requires solving a nonconvex parameter-search problem. The quasi-Polish construction underlying recent infinite-dimensional universal approximation results provid
    
[^53]: 基于文献信息环境上下文的EEG情感状态神经地理空间建模

    Neuro-Geospatial Modelling of EEG Affective States Using Literature-Informed Environmental Context

    [https://arxiv.org/abs/2608.20807](https://arxiv.org/abs/2608.20807)

    本文提出了一种双塔架构，利用文献信息环境先验作为辅助地理空间模态，在缺乏个体暴露数据时提升EEG情感状态分类性能，并通过多种控制实验验证其有效性。

    

    环境暴露（如空气污染和绿化）与情感和认知结果相关，但EEG和环境数据集很少联合地理参考。我们研究了当个体水平暴露数据不可用时，文献信息环境先验能否作为EEG情感状态分类的辅助地理空间模态。我们将EAV基准中的30通道EEG（42名参与者，年龄20-30岁）与源自OpenAQ、Sentinel-2、Sentinel-5P和OpenStreetMap的Astana环境表示相结合。一种双塔架构将EEG-Conformer表示与基于图的环境编码器相结合。由于数据集未共同注册，环境上下文被视为文献信息先验而非测量暴露。我们进行了受试者水平的重复分割、置换和标签洗牌控制、剂量-反应逆转以及领域验证。

    arXiv:2608.20807v1 Announce Type: new  Abstract: Environmental exposures such as air pollution and greenness have been associated with affective and cognitive outcomes, but EEG and environmental datasets are rarely jointly georeferenced. We investigate whether literature-informed environmental priors can serve as an auxiliary geospatial modality for EEG-based affective-state classification when individual-level exposure data are unavailable. We combine 30-channel EEG from the EAV benchmark (42 participants, aged 20-30 years) with environmental representations derived from OpenAQ, Sentinel-2, Sentinel-5P, and OpenStreetMap data for Astana. A dual-tower architecture combines EEG-Conformer representations with a graph-based environmental encoder. Because the datasets are not co-registered, environmental context is treated as a literature-informed prior rather than measured exposure. Subject-level repeated splits, permutation and label-shuffling controls, dose-response reversal, and domain
    
[^54]: CubicSplat：通过误差有界前向松弛实现可微矢量图形

    CubicSplat: Differentiable Vector Graphics via Error-Bounded Forward Relaxation

    [https://arxiv.org/abs/2608.20803](https://arxiv.org/abs/2608.20803)

    本文提出CubicSplat，一种通过均匀折线代理和误差有界松弛实现的可微矢量光栅化器，解决了前向精确性与梯度质量之间的跷跷板问题，确保良好条件的梯度并自动修剪退化基元。

    

    arXiv:2608.20803v1 公告类型：交叉 摘要：矢量图形因其分辨率无关性、紧凑存储和直接可编辑性而备受推崇，这使得对其参数化基元进行可微优化成为一个有吸引力的目标。然而，经典光栅化在几何方面是不连续的，现有通过平滑前向传递的补救方法随着场景复杂度的增加需要越来越复杂的启发式策略。我们将这种脆弱性追溯到梯度跷跷板效应：改善前向几何精确性的设计选择可能系统地降低诱导梯度信号，反之亦然。为了应对这一矛盾，我们引入了CubicSplat，一种可微矢量光栅化器，它用均匀折线替代Bézier最近点求解器，其几何误差以$O(S^{-2})$为界。由此产生的静态计算图通过构造保证了良好条件的梯度，而基于合成的可见性机制无需辅助即可修剪退化基元。

    arXiv:2608.20803v1 Announce Type: cross  Abstract: Vector graphics are prized for their resolution independence, compact storage, and direct editability, making differentiable optimization of their parametric primitives an attractive goal. Yet classical rasterization is discontinuous with respect to geometry, and existing remedies that smooth the forward pass demand increasingly elaborate heuristics as scene complexity grows. We trace this fragility to a gradient seesaw: design choices that improve forward geometric exactness can systematically degrade the induced gradient signal, and vice versa. To navigate this tension we introduce CubicSplat, a differentiable vector rasterizer that replaces B\'ezier closest-point solvers with uniform polyline surrogates whose geometric error is bounded at $O(S^{-2})$. The resulting static computation graph yields well-conditioned gradients by construction, while a compositing-derived visibility mechanism prunes degenerate primitives without auxiliar
    
[^55]: 反思机器人模仿学习中的演示遗忘

    Rethinking Demonstration Unlearning in Imitation Learning for Robotics

    [https://arxiv.org/abs/2608.20784](https://arxiv.org/abs/2608.20784)

    本文提出了一种重新训练校准的审计方法，从行为和证据两个维度评估机器人模仿学习中的演示遗忘，确保编辑后的策略在行为上接近重新训练并防止证据泄露。

    

    arXiv:2608.20784v1 公告类型：交叉 摘要：机器人模仿学习依赖于人类演示，其中一些演示人们可能事后要求删除。没有这些演示进行重新训练是自然的参考方法，但其成本随策略和数据集规模增长而增加，这促使了更经济的操作符，即编辑已训练的策略。从机器遗忘中继承的指标，如遗忘损失或单一成员攻击，并不能确定编辑从闭环操作的策略中删除了什么。因此，我们引入了一种重新训练校准的审计方法，从两个维度读取演示遗忘：行为维度，即编辑后的策略是否表现得像没有删除演示的重新训练策略；证据维度，即审计者是否仍能检测到它曾接受过这些演示的训练。行为维度测量在匹配状态下与重新训练策略的动作差异，并通过独立重新训练构建的基准进行校准，因此处于基准的策略与重新训练之间的接近程度相当于重新训练之间的接近程度。证据维度应用了...

    arXiv:2608.20784v1 Announce Type: cross  Abstract: Imitation learning for robotics depends on human demonstrations, some of which people may later ask to remove. Retraining without them is the natural reference, but its cost grows with policy and dataset scale, motivating cheaper operators that edit a trained policy. Metrics inherited from machine unlearning, such as forgetting loss or a single membership attack, do not establish what an edit removed from a policy acting in closed loop. We therefore introduce a retrain-calibrated audit that reads demonstration unlearning along two axes: behavior, whether the edited policy acts like one retrained without the removed demonstrations, and evidence, whether an auditor can still detect it was trained on them. The behavior axis measures action divergence to that retrain at matched states, calibrated by a floor built from independent retrains, so a policy at the floor is as close to a retrain as retrains are to each other. The evidence axis ap
    
[^56]: 模糊混合专家模型：面向非平稳多变量时间序列预测的可解释状态条件专家路由

    Fuzzy-MoE: Interpretable Regime-Conditioned Expert Routing for Non-Stationary Multivariate Time Series Forecasting

    [https://arxiv.org/abs/2608.20761](https://arxiv.org/abs/2608.20761)

    本文提出Fuzzy-MoE，首次将模糊逻辑与混合专家模型结合，通过双视角路由识别潜在时间状态并动态激活专家，显著提升了非平稳多变量时间序列预测的准确性和可解释性。

    

    在非平稳多变量时间序列中，不同变量和样本通常表现出异质的潜在动态状态，而现有的深度预测模型往往将它们压缩为统一的端到端映射，导致对时变动态的建模欠佳，并且在不同潜在状态下激活何种预测机制的可解释性有限。为克服这些局限性，我们将时间序列预测重构为潜在时间状态识别与可解释专家路由的统一框架，并提出Fuzzy-MoE——一种基于模糊逻辑的动态混合专家模型。Fuzzy-MoE由多个并行专家映射网络和一个双视角模糊路由器组成。通过联合利用局部卷积动态和全局分段统计，路由器推断潜在时间状态，并通过可学习的高斯隶属函数计算专家激活强度，从而实现了...

    arXiv:2608.20761v1 Announce Type: cross  Abstract: In non-stationary multivariate time series, different variables and samples often exhibit heterogeneous latent dynamic states, while existing deep forecasting models usually compress them into a unified end-to-end mapping, leading to suboptimal modeling of time-varying dynamics and limited interpretability regarding which forecasting mechanism is activated under different latent states. To overcome these limitations, we reformulate time series forecasting as a unified framework of latent temporal state identification and interpretable expert routing, and propose Fuzzy-MoE, a fuzzy logic-based dynamic Mixture-of-Experts model. Fuzzy-MoE consists of multiple parallel expert mapping networks and a dual-view fuzzy router. By jointly exploiting local convolutional dynamics and global segmented statistics, the router infers latent temporal states and computes expert activation strengths through learnable Gaussian membership functions, enabli
    
[^57]: 不确定性的隐藏轴：具有贝叶斯输出层的图神经网络中的潜在后验对齐

    Hidden Axis of Uncertainty: Latent-Posterior Alignment in Graph Neural Networks with Bayesian Output Layers

    [https://arxiv.org/abs/2608.20758](https://arxiv.org/abs/2608.20758)

    本文发现图神经网络中贝叶斯输出层的预测不确定性减少源于潜在表示与低方差后验方向的对齐，而非传统假设的后验收缩，并据此提出对齐引导学习（AGL）方法来有效降低不确定性。

    

    具有贝叶斯输出层的贝叶斯神经网络（BNNs）为量化预测不确定性提供了一个原则性强且易于处理框架，然而塑造这种不确定性的机制仍不明确。虽然传统理论将不确定性减少归因于后验收缩，但相应的假设在深度模型中未必成立。在本研究中的具有贝叶斯输出层的图神经网络（GNNs）中，我们观察到预测不确定性随着潜在表示向低方差后验方向移动而减少，即使后验方差并未收缩。我们将这种行为称为潜在后验对齐（LPA），并进行了干预实验来支持其在塑造预测不确定性中的功能作用。基于这一见解，我们提出了对齐引导学习（AGL），该算法在训练过程中显式促进这种对齐。AGL有效减少了预测不确定性，同时

    arXiv:2608.20758v1 Announce Type: new  Abstract: Bayesian Neural Networks (BNNs) with Bayesian output layers provide a principled and tractable framework for quantifying predictive uncertainty, yet the mechanisms shaping that uncertainty remain unclear. While conventional theory attributes uncertainty reduction to posterior contraction, the corresponding assumptions need not hold for deep models. In the Graph Neural Networks (GNNs) with Bayesian output layers studied here, we observe that predictive uncertainty decreases as latent representations shift toward lower-variance posterior directions, even though the posterior variance does not contract. We term this behavior Latent-Posterior Alignment (LPA) and conduct interventional experiments that support its functional role in shaping predictive uncertainty. Building on this insight, we propose Alignment-Guided Learning (AGL), which explicitly promotes this alignment during training. AGL effectively reduces predictive uncertainty while 
    
[^58]: PSK在WMT 2026 MIST中的提交：面向多语言摘要与问答的任务特化QLoRA适配器

    PSK at WMT 2026 MIST: Task-Specialized QLoRA Adapters for Multilingual Summarization and Question Answering

    [https://arxiv.org/abs/2608.20757](https://arxiv.org/abs/2608.20757)

    本文提出了一种基于Tiny Aya Global模型和三个任务特化QLoRA适配器的多语言摘要与问答系统，通过分离任务适配器提升了摘要性能，并针对开放问答的不稳定性提交了多系统方案。

    

    arXiv:2608.20757v1 公告类型：交叉 摘要：我们描述了PSK对WMT 2026多语言指令共享任务的提交。我们的系统使用35.3亿参数的Tiny Aya Global模型，并配备三个QLoRA适配器，每个任务对应一个。这些适配器在多语言文档-摘要对、基于段落的问答以及过滤后的独立问答数据上进行训练。摘要数据还包括带有作者撰写摘要的科学论文。在我们保留的测试集上，上下文和摘要适配器的表现优于我们的多任务适配器，后者仅使用组织者提供的数据进行训练。开放问答的结果因答案长度和评估方法而异，表现不一。因此，我们提交了三个系统，它们共享相同的上下文和摘要适配器，但使用不同的开放问答适配器。

    arXiv:2608.20757v1 Announce Type: cross  Abstract: We describe the PSK submission to the WMT 2026 Multilingual Instruction Shared Task. Our system uses the 3.35B-parameter Tiny Aya Global model with three QLoRA adapters, one for each task. The adapters are trained on multilingual document-summary pairs, passage-based question answering, and filtered standalone question answering. The summarization data also includes scientific papers with their author-written abstracts. On our held-out split, the context and summarization adapters perform better than our multitask adapter, which was trained only on data supplied by the organizers. Results for open QA are mixed and vary with answer length and evaluation method. We therefore submit three systems with the same context and summarization adapters but different open-QA adapters.
    
[^59]: 基于高斯特征桥的长尾半监督学习几何正则化

    Geometric Regularization for Long-Tailed Semi-Supervised Learning via Gaussian Feature Bridges

    [https://arxiv.org/abs/2608.20710](https://arxiv.org/abs/2608.20710)

    本文提出高斯桥一致性（GBC）框架，通过构建未标记样本与类别原型之间的几何插值路径，并引入桥一致性损失和BridgeMix技术，有效解决了长尾半监督学习中的噪声伪标签和确认偏差问题。

    

    现实世界中的半监督学习（SSL）常常面临长尾标签分布和噪声伪标签的重大挑战，这些挑战阻碍了泛化能力并加剧了确认偏差。在这项工作中，我们引入了一种新颖的框架——高斯桥一致性（GBC），通过构建未标记样本与高质量类别锚点之间的语义插值路径来解决这些问题。我们的方法维护了一个动态原型图谱，该图谱存储了每个类别中多样且不断演化的标记和伪标记样本。对于每个未标记实例，GBC在潜在空间中形成一个类别条件的高斯特征桥，使学生模型能够从不确定的预测平滑过渡到可靠的类别原型。沿着这条路径应用桥一致性损失，以强制与几何插值的目标分布对齐。此外，我们提出了BridgeMix，一种置信度感知的方法。

    arXiv:2608.20710v1 Announce Type: new  Abstract: Real-world semi-supervised learning (SSL) often encounters significant challenges with long-tailed label distributions and noisy pseudo-labels, which hinder generalization and amplify confirmation bias. In this work, we introduce a novel framework, Gaussian Bridge Consistency (GBC), to address these challenges by constructing semantic interpolation paths between unlabeled samples and high-quality class anchors. Our method maintains a dynamic Prototype Atlas that stores a diverse and evolving set of labeled and pseudo-labeled exemplars per class. For each unlabeled instance, GBC forms a class-conditional Gaussian Feature Bridge in the latent space, enabling the student model to traverse a smooth trajectory from uncertain predictions to reliable class prototypes. A bridge consistency loss is applied along this path to enforce alignment with a geometrically interpolated target distribution. Furthermore, we propose BridgeMix, a confidence-aw
    
[^60]: 认证驱动的强化学习用于中微子味道模型发现

    CDRL: Certification-Driven Reinforcement Learning for Neutrino Flavor Model Discovery

    [https://arxiv.org/abs/2608.20686](https://arxiv.org/abs/2608.20686)

    本文提出了CDRL框架，通过将符号推理工具生成的结构化认证反馈转化为可重用约束，有效解决了强化学习在科学发现中因标量奖励信息不足而反复探索无效区域的问题，并在中微子味道模型发现任务上超越了现有最先进方法。

    

    arXiv:2608.20686v1 公告类型：新 摘要：许多科学发现问题需要在复杂的领域约束下搜索组合假设空间。强化学习（RL）提供了一种有前景的方法，但现有方法依赖于标量奖励，这种奖励对候选解决方案失败原因提供的信息有限，导致智能体反复探索无效区域。我们引入了认证驱动的强化学习（CDRL），这是一种利用符号推理工具的结构化反馈的框架。当候选方案违反领域约束时，这些工具会产生证书，识别导致失败的动作。CDRL将这些证书转化为可重用的约束，消除一类无效解决方案，并引导探索向有效区域。我们在理论粒子物理中的中微子味道模型发现上评估了CDRL，该假设空间超过$10^{26}$种可能模型，并与最先进的RL方法进行了比较。

    arXiv:2608.20686v1 Announce Type: new  Abstract: Many scientific discovery problems require searching combinatorial hypothesis spaces under complex domain constraints. Reinforcement learning (RL) offers a promising approach, but existing methods rely on scalar rewards that provide limited information about why candidate solutions fail, leading agents to repeatedly explore invalid regions. We introduce Certification-Driven Reinforcement Learning (CDRL), a framework that leverages structured feedback from symbolic reasoning tools. When a candidate violates domain constraints, these tools produce certificates identifying the actions responsible for failure. CDRL converts these certificates into reusable constraints that eliminate classes of invalid solutions and guide exploration toward valid regions. We evaluate CDRL on neutrino flavor model discovery in theoretical particle physics, where the hypothesis space exceeds $10^{26}$ possible models, and compare it with the state-of-the-art RL
    
[^61]: 真实软件历史中的时间有效性：消除GitHub修复中代码助手记忆的过时事实错误

    Temporal Validity on Real Software Histories: Eliminating Stale-Fact Errors in Code-Assistant Memory over GitHub Fixes

    [https://arxiv.org/abs/2608.20685](https://arxiv.org/abs/2608.20685)

    本文验证了MemStrata在真实软件历史中通过确定性过时记忆消除RAG的时间盲点，显著提升答案准确率（0.91对比0.57-0.59），并减少过时事实错误。

    

    检索增强生成（RAG）缺乏时间模型：当编码会话中事实发生变化——函数被重命名、端点移动、依赖项升级——RAG会检索到新旧值，且相似度几乎相同，无法判断哪个是当前的，因此会提供已过时的值。论文1在合成单值基准上表明，确定性（主体、关系、对象）的过时记忆消除可以解决此失败。本文在真实软件历史上进行端到端验证。从707个真实GitHub问题（SWE-bench Lite + Verified）中提取130个干净的原子状态转换，即修复将一个可识别值从修复前形式变为修复后形式，并将每个标记去除（过时和当前语句仅值不同）。在此数据集上，MemStrata达到0.91的答案准确率，而RAG为0.57-0.59；并且，结构性结果表明，当被迫回答时，RAG在36-3%的情况下提供过时值。

    arXiv:2608.20685v1 Announce Type: cross  Abstract: Retrieval-augmented generation (RAG) has no model of time: when a fact changes across a coding session - a function is renamed, an endpoint moves, a dependency is bumped - RAG retrieves both the old and new value with near-identical similarity and cannot tell which is current, so it serves the superseded value. Paper 1 showed, on synthetic single-value benchmarks, that a deterministic (subject, relation, object) supersession memory eliminates this failure. Here we validate it end-to-end on real software history. From 707 real GitHub issues (SWE-bench Lite + Verified) we extract 130 clean atomic state transitions, a fix that changes one identifiable value from a pre-fix to a post-fix form, and render each marker-free (the stale and current statements differ only in the value). On this set, MemStrata reaches 0.91 answer accuracy versus RAG's 0.57-0.59; and, the structural result, when forced to answer RAG serves the superseded value 36-3
    
[^62]: 连续时间跳跃马尔可夫决策过程的强化学习及其在网络动态定价中的应用

    Reinforcement Learning for Continuous-Time Jump Markov Decision Processes with Applications to Network Dynamic Pricing

    [https://arxiv.org/abs/2608.20680](https://arxiv.org/abs/2608.20680)

    本文针对一般离散状态空间的连续时间跳跃马尔可夫决策过程，提出了一种熵正则化的强化学习框架，克服了现有方法对欧氏空间结构的依赖，并应用于网络动态定价问题。

    

    摘要：我们研究了连续时间跳跃马尔可夫决策过程（CTJMDPs）中的强化学习（RL），该过程具有一般离散状态空间（无需具备向量空间结构）和连续/离散动作空间。该设置涵盖了许多运营中的知名应用，如带容量资源的多产品动态定价（Gallego和van Ryzin 1997）。为模拟探索-利用权衡，我们提出了一个带熵正则化的连续时间控制问题，并采用随机策略。最近关于连续时间RL技术，如受控扩散的$q$-学习（Jia和Zhou 2023），侧重于连续状态空间$\mathbb{R}^d$，其理论分析严重依赖于$\mathbb{R}^d$中的半鞅理论。因此，他们的方法无法直接应用于具有一般离散状态空间的CTJMDPs，因为后者可能缺乏欧氏空间固有的代数加法和减法结构。

    arXiv:2608.20680v1 Announce Type: new  Abstract: We study reinforcement learning (RL) in Continuous-Time Jump Markov Decision Processes (CTJMDPs) featuring general discrete state spaces (which need not possess a vector space structure) and continuous/discrete action spaces. The setup covers many well-known applications in operations such as multi-product dynamic pricing with capacitated resources (Gallego and van Ryzin 1997). To model the exploration-exploitation tradeoff, we formulate an entropy-regularized continuous-time control problem with stochastic policies. Recent continuous-time RL techniques such as $q$-learning for controlled diffusions in (Jia and Zhou 2023) focus on continuous state spaces $\mathbb{R}^d$ and rely heavily on semimartingale theory in $\mathbb{R}^d$ for their theoretical analysis. Consequently, their methods cannot be directly applied to CTJMDPs with general discrete state spaces, which may lack the algebraic addition and subtraction structures inherent to Eu
    
[^63]: 基于超球面流形学习的轻量级自适应ReduNet

    Lightweight Adaptive ReduNet via Hyperspherical Manifold Learning

    [https://arxiv.org/abs/2608.20668](https://arxiv.org/abs/2608.20668)

    本文提出LA-ReduNet，通过超球面流形学习细化逐层更新规则，显著减少展开层数，实现轻量级自适应白盒网络，降低参数存储同时保持判别性特征提取能力。

    

    arXiv:2608.20668v1 公告类型：交叉 摘要：近年来，提出了一种名为ReduNet的白盒神经网络，它利用最大编码率降低（MCR²）原理，通过前向逐层构建过程将原始数据转换为低维判别性特征。与依赖反向传播的传统深度网络不同，ReduNet明确地从其前一层特征推导出每层的参数，提供了一种数学上可解释的范式。然而，这种逐层构建通常需要大量层才能使MCR²目标达到稳定值，这增加了展开模块的参数存储。为解决此问题，我们提出了LA-ReduNet，一种轻量级自适应架构，它细化了逐层更新规则，并能够使用显著更少的展开层获得判别性特征表示。具体来说，LA-ReduNet采用超球面流形学习。

    arXiv:2608.20668v1 Announce Type: cross  Abstract: In recent years, a white-box neural network called ReduNet has been proposed, which employs the maximal coding rate reduction (MCR$^2$) principle to transform raw data into low-dimensional discriminative features via a forward layer-wise construction process. Unlike traditional deep networks that rely on backpropagation, ReduNet explicitly derives the parameters of each layer from the features of its preceding layer, offering a mathematically interpretable paradigm. However, this layer-wise construction often requires a large number of layers for the MCR$^2$ objective to reach a stable value, which increases the parameter storage of the unfolded module. To address this issue, we propose LA-ReduNet, a lightweight adaptive architecture that refines the layer-wise update rule and enables discriminative feature representations to be obtained with substantially fewer unfolded layers. Specifically, LA-ReduNet employs hyperspherical manifold 
    
[^64]: C-Score：超越准确率的半监督学习在开放世界未标记污染下的鲁棒性评估

    C-Score: Beyond Accuracy for Robustness Assessment in Semi-Supervised Learning under Open-World Unlabeled Contamination

    [https://arxiv.org/abs/2608.20667](https://arxiv.org/abs/2608.20667)

    本文提出C-Score框架，用于诊断半监督学习在开放世界未标记污染下的隐藏崩溃，超越了传统准确率指标的局限性。

    

    基于伪标签的半监督学习因其简单性和可扩展性而取得了强劲性能。然而，它通常在封闭世界假设下开发，即未标记数据与标记数据来自相同分布。在实际部署中，未标记数据往往从开放环境中收集，可能包含分布外（OOD）样本。在这种污染下，OOD样本仍可能获得高置信度预测，并被当作有效目标示例纳入训练。这产生了一个重要的评估问题：即使半监督学习的内部学习动态已经恶化，干净的分布内测试准确率也可能看似稳定。为解决此问题，我们从诊断评估角度研究开放世界未标记污染下基于伪标签的半监督学习中的隐藏崩溃。我们提出C-Score，一个紧凑框架，从三个互补方面评估训练行为。

    arXiv:2608.20667v1 Announce Type: cross  Abstract: Pseudo-label-based semi-supervised learning has achieved strong performance due to its simplicity and scalability. However, it is typically developed under a closed-world assumption that unlabeled data are drawn from the same distribution as labeled data. In practical deployment, unlabeled data are often collected from open environments and may contain OOD samples. Under such contamination, OOD samples may still receive high-confidence predictions and be incorporated into training as if they were valid target examples. This creates an important evaluation problem: clean in-distribution test accuracy may appear stable even when the internal learning dynamics of SSL have already deteriorated. To address this issue, we study hidden collapse in pseudo-label-based SSL under open-world unlabeled contamination from a diagnostic evaluation perspective. We present C-Score, a compact framework that evaluates training behavior in three complement
    
[^65]: 利用太空望远镜数据和生成式人工智能增强数字巡天的成像能力

    Amplifying the imaging power of digital sky surveys with space telescopes data and generative AI

    [https://arxiv.org/abs/2608.20666](https://arxiv.org/abs/2608.20666)

    本文提出了一种利用生成式人工智能将地面巡天星系图像提升至太空望远镜图像质量的方法，从而结合地面巡天的高吞吐量与太空成像的高细节能力。

    

    arXiv:2608.20666v1 公告类型：交叉 摘要：虽然数字巡天能够提供高吞吐量的图像数据并覆盖大范围天区，但其成像能力通常不如太空望远镜。另一方面，太空望远镜具有出色的成像能力，能够拍摄深空宇宙，但无法提供与先进地面巡天相同的吞吐量。在此，我们利用生成式人工智能将地面望远镜拍摄的星系图像质量提升至太空望远镜所能实现的细节水平。该解决方案基于星系形状的特性，使得在太空图像上训练的生成式人工智能能够将微弱信号转换为细节清晰、明确的星系图像。该方法能够将地面巡天的高吞吐量与太空望远镜的图像质量相结合。该方法的源代码以及配对训练数据和包含63,202个增强星系图像的目录均已公开提供。

    arXiv:2608.20666v1 Announce Type: cross  Abstract: While Digital sky surveys provide excellent throughput of image data and can cover a large footprint, their imaging power is normally inferior to that of space-based telescopes. Space-based telescopes, on the other hand, provide excellent imaging power and can image the deep Universe, but cannot provide the same throughput as advanced ground-based sky surveys. Here, we utilize generative AI to elevate the quality of galaxy images taken by ground-based telescopes to the level of details enabled by space telescopes. The solution is based on the nature of galaxy shapes, allowing generative AI trained on space-based images to convert weak signal into detailed and clear galaxy images. The method allows for combining the high throughput of ground-based sky surveys with the image quality of space-based telescopes. The source code for the method is available, as well as paired training data and a catalog of 63,202 galaxy images enhanced by the
    
[^66]: 预测连续时间量子行走模拟中资源高效的哈密顿量分解方法

    Predicting Resource Efficient Hamiltonian Decomposition for Continuous-Time Quantum Walk Simulations

    [https://arxiv.org/abs/2608.20660](https://arxiv.org/abs/2608.20660)

    本文通过训练机器学习模型，利用图的拓扑特征预测连续时间量子行走模拟中泡利分解与匹配分解哪个更节省CX门资源，并在所有11,117个连通八顶点图上直接验证了预测准确性。

    

    摘要：arXiv:2608.20660v1 公告类型：交叉 摘要：在量子计算的电路模型中模拟图上的连续时间量子行走（CTQW）需要将其哈密顿量分解为可特罗特化为硬件原生门的项。我们考虑了两种这样的分解：标准的泡利分解和最近引入的匹配分解。先前的研究表明，匹配分解在稀疏图上使用更少的CX门，而泡利分解在更密集的图上使用更少。由于CX门在当前硬件上主导误差和运行时间，我们训练机器学习模型来预测，对于给定的图，两种分解中哪一种会产生更小的CX门数量。我们在Brendan McKay数据库中所有11,117个连通八顶点图的完整群体上进行训练和评估，因此类别平衡和重叠是直接测量而非估计的。我们使用了十二个特征：十个图的拓扑属性和两个共同特征。

    arXiv:2608.20660v1 Announce Type: cross  Abstract: Simulating a continuous-time quantum walk (CTQW) on a graph in the circuit model of quantum computing requires decomposing its Hamiltonian into terms that can be Trotterized into hardware-native gates. We consider two such decompositions: the standard Pauli decomposition and the recently introduced matching decomposition. Prior work suggests that the matching decomposition uses fewer CX gates on sparse graphs, while the Pauli decomposition uses fewer on denser graphs. Since CX gates dominate error and runtime on current hardware, we train machine learning models to predict, for a given graph, which of the two decompositions produces the smaller CX gate count. We train and evaluate on the complete population of all 11,117 connected eight-vertex graphs from Brendan McKay's database, so the class balance and overlap are measured directly rather than estimated. We use twelve features: ten topological properties of the graph and two that co
    
[^67]: RiskTraf：面向多变量交通流预测的风险外推残差学习

    RiskTraf: Risk-Extrapolated Residual Learning for Multi-Variate Traffic Flow Prediction

    [https://arxiv.org/abs/2608.20656](https://arxiv.org/abs/2608.20656)

    该论文提出PEMSB-3V基准套件和RiskTraf残差学习方法，通过保留并利用流量、速度和占有率三种原始传感器数据，并采用风险外推策略，解决了多变量交通流预测中数据不一致和状态依赖捷径问题。

    

    arXiv:2608.20656v1 公告类型：交叉 摘要：交通传感器通常记录流量、速度和占有率，但标准的交通流预测基准和模型很少能可靠地利用这三种原始测量数据。尽管速度和占有率提供了超出流量本身的传感器原生交通状态信息，但现有数据集往往省略这些变量、用代理变量替代，或包含逻辑不一致的记录。此外，对三变量输入直接进行经验风险最小化可能利用状态相关的捷径，因为流量、速度和占有率之间的关系在自由流和拥堵状态之间差异显著。我们引入了\textbf{PEMSB-3V}，一个公共基准套件，保留了来自PeMS探测器的原始流量、速度和占有率测量值用于流量预测。我们还提出了\textbf{RiskTraf}，一个模型无关的风险外推残差插件。对于每个训练好的时空骨干网络，RiskTraf冻结所选检查点并学习残差。

    arXiv:2608.20656v1 Announce Type: cross  Abstract: Traffic sensors commonly record flow, speed, and occupancy, but standard traffic flow forecasting benchmarks and models rarely exploit all three raw measurements reliably. Although speed and occupancy provide sensor-native traffic-state information beyond flow alone, existing releases often omit these variables, replace them with proxies, or contain logically inconsistent records. Moreover, direct empirical risk minimization over three-variable inputs may exploit regime-dependent shortcuts, as the relationships among flow, speed, and occupancy vary substantially between free-flow and congested states. We introduce \textbf{PEMSB-3V}, a public benchmark suite that preserves raw flow, speed, and occupancy measurements from PeMS detectors for flow prediction. We also propose \textbf{RiskTraf}, a model-agnostic risk-extrapolated residual plug-in. For each trained spatio-temporal backbone, RiskTraf freezes the selected checkpoint and learns 
    
[^68]: 牛奶中红外光谱的元聚类识别与早期泌乳负能量平衡相关的奶牛群体

    Meta-clustering of milk mid-infrared spectra identifies dairy cow groups associated with negative energy balance in early lactation

    [https://arxiv.org/abs/2608.20653](https://arxiv.org/abs/2608.20653)

    本研究通过结合光谱过滤、降维和聚类方法，从牛奶中红外光谱中直接识别出与早期泌乳负能量平衡相关的奶牛群体，为监测和管理高风险奶牛提供了新方法。

    

    聚类方法已被用于识别牛奶样本、奶牛或牧场的不同群体。傅里叶变换红外光谱，特别是中红外光谱，已应用于个体奶牛牛奶样本以预测多种牛奶性状。直接对中红外光谱数据进行聚类可能揭示与牛奶性状或健康障碍相关的潜在奶牛群体，并有助于预防这些状况或监测风险动物。本研究旨在直接从牛奶中红外光谱中识别早期泌乳的个体奶牛群体，并分析其与牛奶性状的关联。利用来自3,408个商业牧场的407,632条个体牛奶中红外记录数据集，我们结合了(i)选择信息波数的光谱过滤，(ii)两种降维方法：主成分分析和自编码器，以及(iii)两种聚类算法：k均值和谱聚类。

    arXiv:2608.20653v1 Announce Type: new  Abstract: Clustering methods have been used to identify distinct groups of milk samples, cows, or herds. Fourier-transform infrared (FTIR) spectroscopy, particularly mid-infrared (MIR) spectroscopy, has been applied to individual cow milk samples to predict various milk traits. Applying clustering directly to MIR spectral data may reveal latent groups of cows associated with milk traits or health disorders and can help prevent these conditions or monitor at-risk animals. This study aimed to identify groups of individual dairy cows in early lactation directly from milk MIR spectra and to analyze their associations with milk traits. Using a dataset of 407,632 individual milk MIR records from 3,408 commercial farms, we combined (i) spectral filtering that selects informative wavenumbers, (ii) two dimensionality-reduction methods: principal component analysis (PCA) and an autoencoder, and (iii) two clustering algorithms: k-means and spectral clusterin
    
[^69]: 一维二次函数上Adam可证明的边缘稳定性

    Provable Edge-of-Stability for Adam on a One-Dimensional Quadratic

    [https://arxiv.org/abs/2608.20638](https://arxiv.org/abs/2608.20638)

    本文在一维二次函数上证明了Adam优化器存在向稳定阈值恢复的边缘稳定性现象，并揭示了该机制失效的特定情况。

    

    摘要：arXiv:2608.20638v1 公告类型：交叉 摘要：Adam优化器的边缘稳定性（EoS）现象已被广泛观察到，但其背后的动力学机制尚未完全理解。我们在一维二次函数上研究未校正的Adam，这是一个简洁的设置，其中恒定曲率隔离了EoS背后的优化器诱导动力学。我们刻画了参数空间中的动力学结果。在广泛参数范围内，我们证明Adam表现出向其冻结稳定阈值$2(1+\beta_1)/[\eta(1-\beta_1)]$恢复的趋势。我们还识别了这种边缘寻求机制失效的情况，包括严格亚临界周期轨道和特殊调谐的轨迹，这些轨迹在保持均匀超临界的同时收敛到最优解。这些结果为Adam在无演化损失几何的设置中的EoS提供了具体的动力学解释，同时也揭示了其局限性。

    arXiv:2608.20638v1 Announce Type: cross  Abstract: The edge-of-stability (EoS) phenomenon of Adam has been widely observed, while its underlying dynamical mechanism is not yet fully understood. We study uncorrected Adam on a one-dimensional quadratic, a clean setting where constant curvature isolates the optimizer-induced dynamics behind the EoS. We characterize the resulting dynamics across the parameter space. In broad regimes, we prove that Adam exhibits a restoring tendency toward its frozen stability threshold $2(1+\beta_1)/[\eta(1-\beta_1)]$. We also identify settings in which this edge-seeking mechanism breaks down, including strictly subcritical periodic orbits and specially tuned trajectories that converge to the optimum while remaining uniformly supercritical. These results give a concrete dynamical explanation for Adam's EoS in a setting free of evolving loss geometry, while also exposing its limitations.
    
[^70]: MIL-BERT：具有性能与解释性保证的任意大规模文本分类

    MIL-BERT: Classification of Arbitrarily Large Text with Performance and Explanatory Guarantees

    [https://arxiv.org/abs/2608.20636](https://arxiv.org/abs/2608.20636)

    本文提出MIL-BERT算法，利用多实例学习选择关键文本摘录进行分类，可处理近百万令牌的大规模文本，在多个长文本数据集上达到最先进性能，并具备解释性保证。

    

    arXiv:2608.20636v1 公告类型：新 摘要：许多文本分类决策仅基于构成性摘录即可做出。受多实例学习领域的启发，我们提出了一种训练神经网络通过选择此类摘录来对文本进行分类的算法。我们表明，我们的方法也具有可扩展性，并在近100万令牌的样本上进行了实证学习。我们在7个数据集上评估了我们的方法，重点强调远超基础模型编码限制的长文本集合。我们在此算法上在3个数据集上取得了最先进的结果：新闻媒体政治偏见识别、长故事中的触发警告以及推特集合中作者的人口统计特征。此外，在弱标记文本集合（袋）上训练的模型能够泛化，以准确分类构成性的较小实例。除了为这些问题提供新的最先进性能外，这种方法也是少数几种能够提供解释性保证的神经方法之一。

    arXiv:2608.20636v1 Announce Type: new  Abstract: Many text classification decisions are viable based on constituent excerpts alone. Taking inspiration from the field of multiple instance learning, we present an algorithm for training a neural network to classify text by selecting such excerpts. We show that our approach is also scalable with demonstrated learning against samples with nearly 1M tokens. We evaluate our methods on 7 datasets with emphasis on long-textual collections that far exceed the encoding limit of our base model. We present state-of-the-art results with this algorithm on 3 datasets: identification of political bias in news outlets, trigger warnings in long stories, and demographic characteristics of authors in tweet collections. Furthermore, the model trained on weakly-labeled collections of text (bags) generalizes to accurately classify constituent, smaller instances. Besides a new state-of-the-art for these problems, this approach is one of the few neural methods 
    
[^71]: 得分熵离散扩散的最小最大最优性

    Minimax Optimality of Score-Entropy Discrete Diffusion

    [https://arxiv.org/abs/2608.20635](https://arxiv.org/abs/2608.20635)

    本文首次确立了均匀和掩码离散扩散中具体得分估计的最小最大下界，揭示了其统计极限。

    

    离散扩散模型在一系列数据集上表现出强大的性能，包括自然语言数据和图结构数据。在众多变体中，得分熵离散扩散（SEDD）取得了特别强的实证结果。在SEDD中，通过迭代评估一系列具体得分函数来生成新样本，这些函数通过最小化得分熵损失进行学习。尽管先前关于离散扩散的理论文献大多集中在假设得分估计误差较小的情况下SEDD的采样效率，但最近的研究开始探讨得分估计本身的有限样本性质。在这项工作中，我们采取不同路线，研究具体得分估计的基本统计极限。我们关注均匀和掩码离散扩散，这是两种最广泛采用的离散扩散模型。我们建立了一个最小最大下界。

    arXiv:2608.20635v1 Announce Type: cross  Abstract: Discrete diffusion models have demonstrated strong performance across a range of datasets, including natural language data and graph-structured data. Among many variants, score-entropy discrete diffusion (SEDD) has achieved particularly strong empirical results. In SEDD, new samples are generated by iteratively evaluating a sequence of concrete score functions, which are learned by minimizing a score-entropy loss.   While much of the prior theoretical literature on discrete diffusion has focused on the sampling efficiency of SEDD under the assumption of small score estimation error, recent work has begun to investigate the finite-sample properties of score estimation itself. In this work, we take a different route by investigating the fundamental statistical limits of concrete score estimation. We focus on uniform and masking discrete diffusions, two of the most widely adopted discrete diffusion models. We establish a minimax lower bou
    
[^72]: 异构语言模型间的双缓存潜在空间通信

    Dual-Cache Latent Space Communication between Heterogeneous Language Models

    [https://arxiv.org/abs/2608.20617](https://arxiv.org/abs/2608.20617)

    我们提出了XKV，一种新颖的潜在空间通信方法，通过双缓存机制和跨层记忆检索，消除了异构语言模型间通信的上下文共享、层匹配和单层摘要等限制，实现了更高效的智能体间信息传递。

    

    arXiv:2608.20617v1 公告类型：新 摘要：多智能体大语言模型系统将工作分配给不同模型，因此回答问题时往往需要另一个智能体上下文中的知识：一个共享者编码了接收者完成任务所需的信息。它们通常通过交换文本来通信，这使自回归解码处于关键路径上，并将交换简化为在看不到接收者状态的情况下编写的离散消息。最近的潜在协议将共享者的键值（KV）缓存转换为接收者的：C2C支持异构模型，但要求两者读取相同的输入，而LCF-X通过无位置共享缓存池消除了这一共享上下文要求。仍存在三个限制：LCF-X仅压缩共享者，向每个接收者位置提供相同的层局部摘要，没有可检索的联合跨层记忆，并假设层数和KV几何结构匹配。我们引入了XKV，它消除了所有这三个限制：学习查询在...

    arXiv:2608.20617v1 Announce Type: new  Abstract: Multi-agent LLM systems split work across models, so answering often requires knowledge that sits in another agent's context: a Sharer has encoded information that a Receiver needs to complete its task. They usually communicate by exchanging text, which puts autoregressive decoding on the critical path and reduces the exchange to a discrete message written without sight of the receiver's state. Recent latent protocols instead translate the sharer's key-value (KV) cache into the receiver's: C2C supports heterogeneous models but requires both to read the same input, while LCF-X removes this shared-context requirement through position-free sharer-cache pooling. Three restrictions remain: LCF-X compresses the sharer alone, supplies the same layer-local summary to every receiver position with no joint cross-layer memory to retrieve from, and assumes matched layer count and KV geometry. We introduce XKV, which lifts all three: learned-query at
    
[^73]: JuryProbe：一种用于路由无参考事实性评审团到有依据验证的经验共识风险诊断方法

    JuryProbe: An Empirical Consensus-Risk Diagnostic for Routing Reference-Free Factuality Judge Panels to Grounded Verification

    [https://arxiv.org/abs/2608.20607](https://arxiv.org/abs/2608.20607)

    本文提出JuryProbe，一种通过仅假阴性相关性和假共识提升度来诊断无参考事实性评审团共识风险的方法，并在高风险时路由到有参考验证，以减少因共享盲点导致的错误接受。

    

    arXiv:2608.20607v1 公告类型：交叉 摘要：由廉价LLM评审员组成的小组越来越多地做出接受或升级的决策。在事实性设置中，因为多个无参考评审员一致同意而接受一个声明可能会产生隐藏风险：这种一致性可能反映的是共同的假阴性盲点，而非独立的证据。我们引入了JuryProbe，一种针对无参考事实性评审团的经验共识风险诊断方法，并配以基于校准的路由策略。JuryProbe通过使用仅假阴性（FN-only）评审员相关性和假共识提升度，从标记的校准探针中估计共识风险；当标记为高风险时，无参考多数接受会被路由到带有可信参考的相同评审员。在审计的FEVER腐败数据上，无参考评审团显示出相关的假阴性（FN-only相关性为0.402和0.368；提升度分别为3.13倍和18.13倍），而在可信参考最佳案例诊断下，两种情形的一致假共识均降至零。

    arXiv:2608.20607v1 Announce Type: cross  Abstract: Panels of inexpensive LLM judges increasingly make accept-or-escalate decisions. In factuality settings, accepting a claim because several reference-free judges agree can create a hidden risk: agreement may reflect shared false-negative blind spots rather than independent evidence. We introduce JuryProbe, an empirical consensus-risk diagnostic for reference-free factuality judge panels, paired with a calibration-based routing policy. JuryProbe estimates consensus risk from a labeled calibration probe using false-negative-only (FN-only) judge correlation and false-consensus lift; when flagged high-risk, reference-free majority accepts are routed to the same judges with trusted references. On audited FEVER corruptions, reference-free panels show correlated false negatives (FN-only correlations 0.402 and 0.368; lifts 3.13x and 18.13x), while unanimous false consensus drops to zero under a trusted-reference best-case diagnostic on both min
    
[^74]: 基于密钥来源水印与互补格基安全聚合的联邦学习框架

    Keyed Provenance Watermarking with Complementary Lattice-Based Secure Aggregation for Federated Learning

    [https://arxiv.org/abs/2608.20580](https://arxiv.org/abs/2608.20580)

    本文提出一种联邦学习框架，通过密钥来源水印和格基安全聚合，同时解决数据泄露、未授权重用和梯度操纵问题，并引入基于物理锚定元数据的水印机制。

    

    联邦学习（FL）容易受到多层次攻击。然而，现有方法分别处理这些攻击，导致FL仍面临数据泄露、未经授权重用和恶意梯度操纵的风险。在本工作中，我们提出了一种FL框架，该框架将密钥上下文来源水印与可验证的基于格的真实世界锚定水印安全聚合相结合，并采用基于格的零知识安全聚合。在数据层，我们提出了一种符合Kerckhoffs原则的方案，利用物理锚定元数据（PAM）确保数据来源。PAM被定义为从可信基础设施数据（时间、位置和服务器ID）派生的上下文来源令牌，然后经过密钥HMAC-SHA-256变换生成水印负载，该负载在缺少客户端密钥时无法生成。我们进一步设计了FMGAN，一种基于生成对抗网络的鲁棒图像水印框架，使用该变换后的负载进行嵌入。

    arXiv:2608.20580v1 Announce Type: cross  Abstract: Federated learning (FL) is vulnerable to multi-level attacks. However, existing methods address them separately, leaving FL exposed to data leakage, unauthorized reuse, and malicious gradient manipulation. In this work, we propose an FL framework that couples keyed context-provenance watermarking with verifiable lattice-based secure aggregation of Real-World Anchored Watermarking and Lattice-Based Zero-Knowledge Secure Aggregation. At the data layer, we propose a Kerckhoffs-compliant scheme that utilizes Physical Anchor Metadata (PAM) to ensure data provenance. PAM is defined as a context-provenance token derived from trusted infrastructure data (time, location, and server ID) and then subjected to a keyed HMAC-SHA-256 transformation to produce a watermark payload that cannot be generated without the client's secret key. We further design FMGAN, a GAN-based robust image watermarking framework that embeds this transformed payload using 
    
[^75]: FlavourBench：用可执行的烹饪真实数据对前沿语言模型进行排名

    FlavourBench: Ranking Frontier Language Models with Executable Culinary Ground Truth

    [https://arxiv.org/abs/2608.20574](https://arxiv.org/abs/2608.20574)

    该论文提出了一个基于可执行烹饪真实数据的自动化基准测试FlavourBench，通过版本化系统和严格统计方法对27个前沿语言模型进行公平排名，消除了传统基准中的评判者偏差和缺失数据问题。

    

    开放式语言模型基准测试通常继承一个评判者：人类偏好小组、另一个模型，或脆弱的精确匹配键。我们引入了FlavourBench，一个自动化基准测试，其中版本化的烹饪系统提供密集、可执行的真实数据。每个任务呈现八种食材，并要求选择三种食材的组合；在模型执行前，Epicure对所有56种可能的组合进行评分。我们在一个包含534个任务的相同核心集上评估了27个前沿端点，涵盖替代、配对和受限组合。每个排名的模型在每个面板和家族中恰好有89个有效响应（总共14,418个模型-任务单元），消除了排行榜上的差异性缺失。FlavourBench分数是冻结任务分数的等家族均值。我们使用50,000个锚点聚类自助重采样进行同时95%分数区间，以及100,000次符号翻转抽样进行所有351个配对模型对比，并采用Holm校正。两个独立的...

    arXiv:2608.20574v1 Announce Type: new  Abstract: Open-ended language-model benchmarks usually inherit a judge: a human preference panel, another model, or a brittle exact-match key. We introduce FlavourBench, an automated benchmark in which a versioned culinary system supplies dense, executable ground truth. Each task presents eight ingredients and asks for a three-ingredient portfolio; before model execution, Epicure scores all 56 possible portfolios. We evaluate 27 frontier endpoints on an identical 534-task core spanning substitution, pairing, and constrained composition. Every ranked model has exactly 89 valid responses per panel and family (14,418 model-task cells total), eliminating differential missingness from the leaderboard. The FlavourBench Score is the equal-family mean of the frozen task scores. We use 50,000 anchor-cluster bootstrap replicates for simultaneous 95% score bands and 100,000 sign-flip draws for all 351 paired model contrasts, with Holm control. The two indepe
    
[^76]: 强化防御的故障：通过GPU降压实现CNN对抗鲁棒性

    Faults That Fortify: CNN Adversarial Robustness via GPU Undervolting

    [https://arxiv.org/abs/2608.20572](https://arxiv.org/abs/2608.20572)

    本文发现训练期间通过GPU降压引入的硬件故障可作为隐式正则化，在降低功耗的同时提高CNN的对抗鲁棒性，甚至在对抗训练中也能增强防御能力。

    

    arXiv:2608.20572v1 公告类型：新 摘要：卷积神经网络（CNN）面临双重挑战：易受对抗性攻击影响以及训练成本过高。对抗性训练有效但昂贵，随着学习向能源受限的边缘设备转移，这一负担日益加重。本文通过训练期间进行GPU降压来同时解决这两个问题。降低供电电压会引入随机扰动，这些扰动作为隐式正则化，在提高鲁棒性的同时降低功耗。我们在位级别表征了降压引起的故障，然后在MNIST和CIFAR-10上训练LeNet、VGG-6和MobileNetV3，采用两种训练模式（标准训练和对抗训练），每种模式分别在额定电压和降压电压下进行，并评估所有模型对对抗性攻击的抵抗能力。在两种模式下，降压模型始终比其额定电压对应模型实现更高的对抗准确性，表明硬件引起的故障甚至能增强对抗性训练。由于动态功耗降低，这一方法在保持或提升鲁棒性的同时，还能节约能源。

    arXiv:2608.20572v1 Announce Type: new  Abstract: Convolutional Neural Networks (CNNs) face a dual challenge: vulnerability to adversarial attacks and prohibitive training cost. Adversarial training is effective but expensive, a burden that grows as learning shifts to the energy-constrained edge. This paper addresses both through GPU undervolting during training. Reducing supply voltage introduces stochastic perturbations that act as implicit regularization, improving robustness while lowering power. We characterize undervolting-induced faults at the bit level, then train LeNet, VGG-6, and MobileNetV3 on MNIST and CIFAR-10 under two training regimes, standard and adversarial, each at nominal and undervolted voltage, and evaluate all models against adversarial attacks. In both regimes, the undervolted model consistently achieves higher adversarial accuracy than its nominal-voltage counterpart, showing that hardware-induced faults strengthen even adversarial training. Because dynamic powe
    
[^77]: AgentDecarbonizer：面向AI代理的碳感知执行

    AgentDecarbonizer: Carbon-Aware Execution for AI Agents

    [https://arxiv.org/abs/2608.20566](https://arxiv.org/abs/2608.20566)

    本文提出AgentDecarbonizer，通过利用代理任务的截止日期灵活性，结合时间转移和空间转移策略，在不确定执行时间和缓存重计算挑战下实现碳感知执行，以降低AI代理工作流的碳排放。

    

    arXiv:2608.20566v1 公告类型：新 摘要：AI代理将大型语言模型从单次提示-响应的交互扩展到长期运行、目标导向的工作流，这些工作流会发起多次模型调用、调用工具并与外部环境交互。这些工作流支持软件修复、数据分析和实验管理等任务，但其重复的模型调用可能产生大量碳排放。本文使用WildClawBench描述了OpenClaw代理工作负载的碳排放特征，并表明排放取决于令牌消耗、上下文缓存重用和电网的碳强度。我们的特征分析将截止日期灵活性确定为碳感知执行的一个机会：代理任务可以等待低碳强度时段或转移到低碳电网。然而，这样做需要处理时间转移中的不确定执行时间和空间转移中的缓存上下文重计算。我们提出了AgentDecarbonizer，一种碳...

    arXiv:2608.20566v1 Announce Type: new  Abstract: AI agents extend large language models from single prompt-response interactions to long-running, goaldirected workflows that issue many model calls, invoke tools, and interact with external environments. These workflows enable tasks such as software repair, data analysis, and experiment management, but their repeated model invocations can incur substantial carbon emissions. This paper characterizes the carbon emissions of OpenClaw agent workloads using WildClawBench, and shows that emissions depend on token consumption, context cache reuse, and the carbon intensity of the grid. Our characterization identifies deadline flexibility as an opportunity for carbon-aware execution: agent tasks can wait for lower-carbon-intensity periods or shift to lower-carbon grids. However, doing so requires handling uncertain execution time for temporal shifting and cached context recomputation during spatial shifting. We present AgentDecarbonizer, a carbon
    
[^78]: 条件独立性正则化的混合类型数据分布自编码器

    Conditional-Independence-Regularized Distributional Autoencoders for Mixed-Type Data

    [https://arxiv.org/abs/2608.20562](https://arxiv.org/abs/2608.20562)

    本文提出了一种结合能量分数、似然目标和条件独立性正则化的分布自编码器框架，用于学习混合类型数据的低维表示，同时保留异构变量间的可解释结构关系。

    

    包含数值和分类变量的混合类型数据出现在许多科学和实际应用中。现有的表示学习和生成建模方法通常侧重于重建精度或无条件的生成数据，但往往无法在保留异构变量类型之间可解释的结构关系的同时恢复数据的完整条件分布。在这项工作中，我们引入了条件独立性正则化的分布自编码器，这是一种通过条件分布匹配和结构正则化学习混合类型数据低维表示的框架。我们的方法结合了基于能量分数的数值变量目标、基于似然的分类变量目标，以及一个辅助的条件独立性正则化项，以鼓励学习到的表示捕捉变量之间的依赖关系。

    arXiv:2608.20562v1 Announce Type: cross  Abstract: Mixed-type data containing both numerical and categorical variables arise in many scientific and real-world applications. Existing representation learning and generative modeling approaches typically focus either on reconstruction accuracy or unconditional data generation, but often fail to recover the full conditional distribution of the data while preserving interpretable structural relationships between heterogeneous variable types. In this work, we introduce Conditional-Independence-Regularized Distributional Autoencoders, a framework for learning low-dimensional representations of mixed-type data through conditional distribution matching and structural regularization. Our method combines an energy-score-based objective for numerical variables, a likelihood-based objective for categorical variables, and an auxiliary conditional independence regularization term encouraging the learned representation to capture the dependence between
    
[^79]: 基于去噪正则化的快速MRI重建一致性模型

    Consistency Models for Fast MRI Reconstruction Using Regularization by Denoising

    [https://arxiv.org/abs/2608.20561](https://arxiv.org/abs/2608.20561)

    本论文提出CM-RED方法，将预训练一致性模型集成到去噪正则化框架中，通过受控噪声注入加速MRI重建，仅需4次网络评估即可在多种条件下实现高质量重建。

    

    arXiv:2608.20561v1 公告类型：交叉 摘要：扩散模型（DMs）已成为MRI重建的强大生成先验，并展现出有前景的结果。然而，基于DM的方法需要大量的迭代细化，限制了其实际部署。一致性模型（CMs）提供了一种引人注目的替代方案，旨在单次通过中映射扩散轨迹，实现更快的生成。在本工作中，我们提出了CM-RED，一种新颖的MRI重建方法，将预训练的CM集成到去噪正则化（RED）方案中。我们的方法基于加速近端梯度RED（RED-APG），并进一步在更新步骤中引入受控噪声注入，以增强生成多样性和加速收敛。在fastMRI膝盖和脑部数据集上的大量实验表明，CM-RED在多种解剖结构、对比权重、加速因子和欠采样模式下，仅使用4个网络即可实现高质量重建。

    arXiv:2608.20561v1 Announce Type: cross  Abstract: Diffusion models (DMs) have emerged as powerful generative priors for MRI reconstruction with promising results. Yet DM-based methods require extensive iterative refinement, limiting their practical deployment. Consistency models (CMs) provide a compelling alternative, aiming to map out the diffusion trajectory in a single pass, enabling faster generation. In this work, we propose CM-RED, a novel MRI reconstruction method that integrates a pretrained CM into the regularization by denoising (RED) scheme. Our method builds on accelerated proximal gradient RED (RED-APG), and further incorporates controlled noise injection during the update steps to enhance generative diversity and accelerate convergence. Extensive experiments on the fastMRI knee and brain datasets demonstrate that CM-RED achieves high-quality reconstructions across multiple anatomies, contrast weights, acceleration factors, and undersampling patterns, using only 4 network
    
[^80]: 在测试时学习前列腺解剖结构以用于微超声癌症检测

    Learning Prostate Anatomy at Test Time for Cancer Detection in Micro-Ultrasound

    [https://arxiv.org/abs/2608.20557](https://arxiv.org/abs/2608.20557)

    提出了一种名为ANT的分割引导测试时自适应框架，通过利用前列腺解剖结构来纠正域偏移，在无需额外标注的情况下提升微超声前列腺癌检测性能。

    

    arXiv:2608.20557v1 公告类型：交叉 摘要：不同临床中心使用不同成像硬件或采集协议导致的域偏移，仍然是部署深度学习模型进行前列腺癌（PCa）检测的根本障碍。现有的测试时自适应（TTA）方法通过熵最小化或基于增强的自监督来解决分布偏移，纠正图像外观上的统计差异，但忽略了目标域的解剖结构。我们提出了ANT，一种分割引导的TTA框架，通过解决测试时的辅助前列腺分割任务，并由冻结的预训练分割网络生成的伪掩码进行监督，将预训练的癌症检测编码器适应到目标域。通过将编码器表示与目标域中的前列腺解剖结构对齐，ANT纠正了域特定特征漂移，同时保留了癌症判别结构。该模型在693名使用早期设备成像的患者数据上进行了训练。

    arXiv:2608.20557v1 Announce Type: cross  Abstract: Domain shift across clinical centers using different imaging hardware or acquisition protocols remains a fundamental barrier to deploying deep learning models for prostate cancer (PCa) detection. Existing test-time adaptation (TTA) methods address distribution shift through entropy minimization or augmentation-based self-supervision, correcting for statistical differences in image appearance but ignoring the anatomical structure of the target domain. We propose ANT, a segmentation-guided TTA framework that adapts a pretrained cancer detection encoder to the target domain by solving an auxiliary prostate segmentation task at test time, supervised by pseudo-masks from a frozen pretrained segmentation network. By aligning encoder representations to prostate anatomy in the target domain, ANT corrects domain-specific feature drift while preserving cancer-discriminative structure. The model was trained on 693 patients imaged with an earlier-
    
[^81]: aiXamine：大语言模型安全、安全性与隐私跨维度权衡的统一黑盒评估

    aiXamine: Unified Black-Box Evaluation of Cross-Dimensional Trade-offs in LLM Safety, Security, and Privacy

    [https://arxiv.org/abs/2608.20554](https://arxiv.org/abs/2608.20554)

    本文提出了aiXamine，一个统一黑盒平台，通过跨维度评估LLM的安全、安全性和隐私性，揭示了传统独立评估框架无法检测的相互依赖风险模式，并在大规模测试中实现了可重复的比较分析。

    

    已部署的大型语言模型（LLM）中的关键失败模式是跨维度的：一个模型在安全对齐上得分99.3，却可能拒绝三分之一的良性查询，或者在所有能力指标上提升，同时隐私得分下降21分。现有评估框架独立评估安全性、安全性和隐私性，无法检测这些模式。我们引入了aiXamine，一个统一的黑盒平台，将LLM的可信度评估为安全、安全性和隐私这三个相互依赖的属性。aiXamine通过自动化红队流水线，在九项服务中编排46项测试，生成从提示级诊断到跨服务权衡分析的分层风险概况，从而在相同条件下实现对专有和开源权重系统的可重复比较。通过将aiXamine应用于超过120个LLM，进行超过5,000次测试运行，我们进行了最大规模的联合安全、安全性和隐私评估。

    arXiv:2608.20554v1 Announce Type: cross  Abstract: The critical failure modes in deployed large language models (LLMs) are cross-dimensional: a model can score 99.3 in safety alignment while refusing one in three benign queries, or improve across every capability metric while losing 21 points in privacy. Existing evaluation frameworks that assess safety, security, and privacy independently cannot detect these patterns. We introduce aiXamine, a unified black-box platform that evaluates LLM trustworthiness across safety, security, and privacy as interdependent properties. aiXamine orchestrates 46 tests across nine services through an automated red-teaming pipeline, producing hierarchical risk profiles, from prompt-level diagnostics to cross-service trade-off analytics, that enable reproducible comparison of proprietary and open-weight systems under identical conditions. Applying aiXamine to over 120 LLMs through more than 5,000 test runs, we conduct the largest joint safety, security, an
    
[^82]: 亲近你的朋友，更要亲近正确的邻居：面向灾害条件的内核正则化图注意力用于建筑损伤分类

    Keep Your Friends Close, and the Right Neighbours Closer: Disaster-Conditioned Kernel-Regularized Graph Attention for Building Damage Classification

    [https://arxiv.org/abs/2608.20548](https://arxiv.org/abs/2608.20548)

    该论文提出一种灾害条件的内核正则化图注意力方法，在建筑损伤分类中有效利用空间上下文，并针对不同灾害类型自适应调整邻域权重，以平衡视觉一致性和边界清晰度。

    

    灾害损害具有空间性：建筑物很少孤立地受损。然而，利用空间上下文进行损害分类的研究仍出奇地不足，许多流程主要依赖单体建筑的外观线索，即使主导不确定性具有空间结构。更复杂的是，正确的邻域在不同事件中并不相同。洪水、飓风和野火可能表现出非常不同的聚集行为，这使得空间推理有价值但容易被误用——天真的上下文聚合可以改善视觉一致性，同时过度平滑边界或传播结构化误差。我们在xBD（xView2挑战赛中使用的数据集）上，在受控的后定位、仅分类设置中研究这一矛盾：每栋建筑由从提供多边形裁剪的灾前/灾后组合（PPC）补丁表示，空间上下文使用GPS衍生的建筑图建模。我们的方法保留局部证据...

    arXiv:2608.20548v1 Announce Type: cross  Abstract: Disaster damage is spatial: buildings rarely fail in isolation. Yet using spatial context for damage classification remains surprisingly underexplored, and many pipelines still rely primarily on per-building appearance cues even when the dominant uncertainty is spatially structured. Complicating matters, the right neighbourhood is not the same across events. Floods, hurricanes, and wildfires can exhibit very different clustering behaviour, making spatial reasoning valuable but easy to misuse - naive context aggregation can improve visual coherence while oversmoothing boundaries or propagating structured errors. We study this tension on xBD (the dataset used in the xView2 challenge) in a controlled post-localization, classification-only setup: each building is represented by a pre/post combined (PPC) patch cropped from the provided polygons, and spatial context is modelled with GPS-derived building graphs. Our approach keeps local evide
    
[^83]: 利用$\mathbb{F}_2$线性代数学习精确的NVIDIA SASS编码器

    Learning Exact NVIDIA SASS Encoders with $\mathbb{F}_2$ Linear Algebra

    [https://arxiv.org/abs/2608.20532](https://arxiv.org/abs/2608.20532)

    F2Asm首次利用$\mathbb{F}_2$线性代数学习精确的SASS指令编码器，并成为支持Rubin SM107的开源NVIDIA SASS汇编器。

    

    NVIDIA提供了SASS反汇编器，但没有针对近期数据中心GPU的公开SASS汇编器，这限制了受控的机器码重写。我们提出了F2Asm，它通过成对的反汇编结果和原始CUBIN指令字来学习精确的128位SASS编码器。据我们所知，F2Asm是首个将SASS指令编码器学习为$\mathbb{F}_2$上向量值仿射映射的系统，也是首个支持Rubin SM107的开源NVIDIA SASS汇编器。F2Asm使用$\mathbb{F}_2$上的高斯消元法来增量构建紧凑基，检测不一致性，并拒绝超出学习跨度范围的输入。F2Asm将目标特定的控制位、重定位规则和CUBIN元数据与其学习算法分离。我们使用来自固定NVIDIA和第三方生产库、CUDA 13.3包以及CUDA 13.4开发者预览存档中的3,225个CUBIN，为Hopper SM90/SM90a、Blackwell SM100和Rubin SM107训练编码器。在往返测试中，F2Asm重新组装了...

    arXiv:2608.20532v1 Announce Type: new  Abstract: NVIDIA provides a SASS disassembler but no public SASS assembler for recent data-center GPUs, limiting controlled machine-code rewriting. We present F2Asm, which learns exact 128-bit SASS encoders from paired disassembly and original CUBIN instruction words. To our knowledge, F2Asm is the first system to learn SASS instruction encoders as vector-valued affine maps over F2 and the first open-source NVIDIA SASS assembler to support Rubin SM107. F2Asm uses Gaussian elimination over F2 to incrementally build a compact basis, detect inconsistencies, and reject inputs outside the learned span. F2Asm separates target-specific control bits, relocation rules, and CUBIN metadata from its learning algorithm. We train encoders for Hopper SM90/SM90a, Blackwell SM100, and Rubin SM107 using 3,225 CUBINs from pinned NVIDIA and third-party production libraries, CUDA 13.3 packages, and CUDA 13.4 Developer Preview archives. In round-trip tests, F2Asm reass
    
[^84]: 当Graph-JEPA学习错误内容：诊断与修复类别条件崩溃

    When Graph-JEPA Learns the Wrong Thing: Diagnosing and Repairing Category-Conditional Collapse

    [https://arxiv.org/abs/2608.20516](https://arxiv.org/abs/2608.20516)

    本文揭示并修复了Graph-JEPA在科学推理图上出现的一种类别条件崩溃，即模型在线性探测和有效秩上表现正常但实例信息为零，通过方差分配分析诊断了根因并提供了修复方法。

    

    arXiv:2608.20516v1 公告类型：新 摘要：联合嵌入预测架构几乎普遍通过线性探测和有效秩进行选择。我们报告了一个案例，其中这两项指标看起来健康，但表示中携带的可用实例信息为零。我们修复了这个问题，但出现了第二个失败：修复后的指标在缺乏结构信息的目标上饱和。我们的语料库是一个科学推理图，包含57,903篇文章，每篇文章都是一个子图。一个Graph-JEPA从子图的剩余方面预测一个被掩蔽的方面，达到了线性探测准确率0.871和有效秩18-47，但检索仅恢复了14.4位中的0.00（MRR为1.9e-4，而随机水平为1.99e-4，p=0.98）。在同一池和代码上的三个上界恢复了几乎所有信息（+14.28、+14.34、+14.22位），排除了语料库、掩蔽、池和指标作为原因。我们将此归因于方差分配——冻结输入将86.05%的方差放在子图身份上，0.40%放在方面身份上，而训练后的潜变量...

    arXiv:2608.20516v1 Announce Type: new  Abstract: Joint-embedding predictive architectures are selected almost universally by linear probing and effective rank. We report a case where both read healthily while the representation carries zero usable instance information. We repair it, and a second failure appears: the repaired metric saturates on a target carrying no structural information. Our corpus is a scientific-reasoning graph over 57,903 articles, each a subgraph. A Graph-JEPA predicts one masked aspect from a subgraph's remaining aspects, attaining linear-probe accuracy 0.871 and effective rank 18-47, yet retrieval recovers 0.00 of 14.4 bits (MRR 1.9e-4 vs chance 1.99e-4, p=0.98). Three upper bounds on the same pool and code recover nearly everything (+14.28, +14.34, +14.22 bits), ruling out corpus, masking, pool, and metric as causes. We trace this to variance allocation - frozen inputs place 86.05% of variance on subgraph identity and 0.40% on aspect identity, while trained lat
    
[^85]: Bern2Edge：一种通过伯恩斯坦多项式网络实现边缘部署的神经符号编译器

    Bern2Edge: A Neurosymbolic Compiler for Edge Deployment via Bernstein Polynomial Networks

    [https://arxiv.org/abs/2608.20497](https://arxiv.org/abs/2608.20497)

    该论文提出Bern2Edge框架，通过伯恩斯坦多项式网络将神经网络转换为硬件高效且可解释的部署形式，在压缩下提升准确率达2.12个百分点。

    

    arXiv:2608.20497v1 公告类型：新 摘要：在资源受限的边缘设备上部署高精度神经网络仍然具有挑战性，因为现有方法将训练、压缩和硬件合成视为独立阶段，导致软件训练模型与高效端到端部署之间存在差距，且对可解释性的支持有限。我们提出Bern2Edge，一种端到端框架，通过知识蒸馏将预训练的教师前馈网络转换为基于伯恩斯坦多项式激活的硬件高效表示。这种表示支持两条部署路径：（i）高保真基于查找表（LUT）的实现，在压缩下保持模型保真度；（ii）基于伯恩斯坦激活几何的符号规则表示，实现具有显式输入空间约束的可解释推理。所得BNN在相同压缩条件下，相比ReLU实现了高达2.12个百分点（pp）的准确率提升。

    arXiv:2608.20497v1 Announce Type: new  Abstract: Deploying high-accuracy neural networks on resource-constrained edge devices remains challenging, as existing approaches treat training, compression, and hardware synthesis as separate stages, leaving a gap between software-trained models and efficient end-to-end deployment with limited support for interpretability. We propose Bern2Edge, an end-to-end framework that uses knowledge distillation to convert a pretrained teacher feed-forward network into hardware-efficient representations via Bernstein polynomial activations. This representation enables two deployment paths: (i) a high-fidelity LUT-based realization that preserves model fidelity under compression, and (ii) a symbolic rule-based representation derived from Bernstein activation geometry, enabling interpretable inference with explicit input-space constraints. The resulting BNNs achieve up to 2.12 percentage-point (pp) accuracy improvement over ReLU under identical compression c
    
[^86]: Metag：一个用于构建智能元审稿能力的数据集

    Metag: A dataset to build agentic meta-reviewing capabilities

    [https://arxiv.org/abs/2608.20488](https://arxiv.org/abs/2608.20488)

    本文提出了Metag数据集，用于训练和评估元审稿智能体，通过将审稿人意见、作者回应和手稿修订差异对齐，实现自动识别论文在评审过程中所做的修改。

    

    人工智能工具日益支持科学研究周期中的各项任务，从实验设计、手稿准备到同行评审。与此同时，会议投稿的持续增长增加了元审稿人的负担，他们必须综合审稿人反馈、作者反驳和稿件修订。为解决这一问题，本文介绍了Metag，一个旨在加速元审稿智能体发展的数据集，特别用于识别在评审-反驳过程中对科学文章所做的修改。每个实例包含一个审稿人关注点、作者提出的解决方案，以及实施所述更改的手稿差异。Metag的收集方式是获取评审截止日期前和接受后的手稿版本，计算两份文档之间的差异，并要求人工标注员将这些差异与OpenReview讨论中的行动项对齐。其结果是...

    arXiv:2608.20488v1 Announce Type: new  Abstract: AI tools increasingly support tasks across the scientific research cycle, from experiment design and manuscript preparation to peer review. At the same time, the continuing growth in conference submissions has increased the burden on meta-reviewers, who must synthesize reviewer feedback, author rebuttals, and manuscript revisions. To address this concern, this paper introduces Metag, a dataset to accelerate the development of meta-reviewing agents, specifically to identify changes made to scientific articles during the review-rebuttal process. Each instance contains a reviewer concern, the author's proposed resolution, and the manuscript diffs implementing the stated change. Metag is collected by obtaining manuscript versions from before the review deadline and after acceptance, computing differences between the two documents, and asking human annotators to align these differences with action items from OpenReview discussions. The result
    
[^87]: 自回归随机神经网络模型中的不确定性传播

    Uncertainty propagation in auto-regressive random neural network models

    [https://arxiv.org/abs/2608.20483](https://arxiv.org/abs/2608.20483)

    本文提出了一种基于Leaky ReLU分段线性结构的解析与粒子方法，用于随机神经网络及其自回归模型中的不确定性传播，实现了输出概率密度、特征函数及均值和协方差的精确或近似计算。

    

    arXiv:2608.20483v1 公告类型：交叉 摘要：我们开发了用于随机神经网络模型中不确定性传播的解析和粒子方法，其中输入和网络参数均允许为随机变量。基于Leaky ReLU激活函数的分段线性结构，我们推导了神经网络输出相对于其输入和参数扰动的局部近似。该近似对于保持网络激活模式的扰动是精确的，它使我们能够计算网络输出的概率密度函数和特征函数的解析表达式，以及其均值和协方差的闭式近似。我们将此不确定性传播框架扩展到一步演化映射由随机神经网络表示的自主动力系统。该映射的重复应用定义了一个自回归模型，为此我们推导了递归方程。

    arXiv:2608.20483v1 Announce Type: cross  Abstract: We develop analytical and particle-based methods for uncertainty propagation in random neural network models, where both the inputs and network parameters are allowed to be random. Building on the piecewise-linear structure of the Leaky ReLU activation function, we derive a local approximation of the neural network output with respect to perturbations in both its inputs and parameters. This approximation is exact for perturbations that preserve the network activation pattern, and it allows us to compute analytical expressions for the probability density function and characteristic function of the network output, together with closed-form approximations for its mean and covariance. We extend this uncertainty propagation framework to autonomous dynamical systems whose one-step evolution map is represented by a random neural network. Repeated application of this map defines an autoregressive model, for which we derive recursive equations 
    
[^88]: 当干净数据有害时：超越二分类的单调损坏学习

    When Clean Data Hurts: Learning with Monotone Corruptions Beyond Binary Classification

    [https://arxiv.org/abs/2608.20480](https://arxiv.org/abs/2608.20480)

    本研究证明单调对抗性损坏在多类分类和部分二分类概念类等更一般的学习设置中，比二分类情况更具破坏性，显著增加了学习器的错误率。

    

    arXiv:2608.20480v1 公告类型：新 摘要：最优学习器是针对经典PAC模型下的独立同分布（i.i.d.）数据假设而设计的。如果i.i.d.训练样本被来自无关甚至对抗性来源的正确标记示例污染，会发生什么？这种带有单调对抗性损坏的学习模型最近由Larsen等人（2026）引入，他们证明了所有已知的最优二分类学习器在此设置下错误率增加，从PAC模型中的$O(d / n)$上升到单调损坏下的$\Omega (d \log(n / d) / n)$。Mehrotra（2026）证明了对二分类而言，这一对数因子是必要的，但未解决损坏对更一般学习设置（如多类分类和部分二分类概念类）的影响。作为我们的主要结果，我们证明在这些设置中，单调对抗者可怕地更加强大。我们展示了一个可学习的多类pr（原文截断，可能为“问题”或“概念类”）

    arXiv:2608.20480v1 Announce Type: new  Abstract: Optimal learners are tailored to exploit the i.i.d.\ data assumption underlying the classic PAC model. What if an i.i.d.\ training sample were corrupted with correctly labeled examples drawn from an otherwise unrelated, even adversarial source? This model of learning with monotone adversarial corruptions was recently introduced by Larsen et al. (2026), who demonstrated that all known optimal binary learners suffer increased error rates in this setting, from $O(d / n)$ in the PAC model to $\Omega (d \log(n / d) / n)$ under monotone corruption. Mehrotra (2026) proved this logarithmic factor to be necessary for binary classification, but left open the consequences of corruption for more general learning settings, such as multiclass classification and partial binary concept classes.   As our primary result, we demonstrate that monotone adversaries are frighteningly more powerful in each of these settings. We exhibit a learnable multiclass pr
    
[^89]: 客户定向中的特征选择：互信息与敏感性分析的比较研究

    Mutual information and sensitivity analysis for feature selection in customer targeting: a comparative study

    [https://arxiv.org/abs/2608.20447](https://arxiv.org/abs/2608.20447)

    本研究比较了互信息与基于数据的敏感性分析在银行电话营销特征选择中的效果，发现敏感性分析在低误报率下更优，而互信息在高误报率下略好。

    

    特征选择在数据驱动的知识发现项目中是一项高度相关的任务。已有多种技术旨在寻找对预测结果影响最大的特征，包括互信息以及近年来基于数据的敏感性分析。本研究通过将这两种技术应用于银行电话营销案例，重点分析它们各自的优缺点。随后，基于每种技术识别出的对电话营销成功影响最大的特征集，构建了逻辑回归模型，其中互信息共识别出13个特征，基于数据的敏感性分析识别出9个特征。后者在较低的误报率下表现更好，而前者在较高的误报率下略优。因此，如果银行管理者更看重较低的误报率，互信息是更好的选择。

    arXiv:2608.20447v1 Announce Type: new  Abstract: Feature selection is a highly relevant task in a data-driven knowledge discovery project. Several techniques have been developed aiming at finding the features that influence most an outcome to predict, including mutual information and, in recent years, the data-based sensitivity analysis. The present research focus on analyzing the advantages and disadvantages of each of these two techniques, by applying both to a bank telemarketing case. Thereafter, a logistic regression model is built on the tuned set of features identified by each of the two techniques as the most influencing set of features on the success of a telemarketing contact, in a total of 13 features for mutual information and 9 features for the data-based sensitivity analysis. The latter performs better for lower values of false positives while the former is slightly better for a higher false positive ratio. Thus, mutual information becomes a better choice if bank managers 
    
[^90]: 对数分数下核密度估计的摊销带宽学习

    Amortized Bandwidth Learning for Kernel Density Estimation under Logarithmic Score

    [https://arxiv.org/abs/2608.20445](https://arxiv.org/abs/2608.20445)

    本文提出了一种基于摊销学习的核密度估计带宽选择方法，通过优化对数分数和仿射标准化，在多种任务中显著优于传统选择器。

    

    arXiv:2608.20445v1 公告类型：新 摘要：核密度估计将有限样本转换为概率密度，但其性能关键依赖于带宽选择。经典选择器通过解析或渐近方式规定样本到带宽的规则，或为每个样本解决新的优化问题。本文提出了一种摊销框架，通过优化对数分数，在密度估计任务的分布上学习这种映射。截断并重新归一化的有界支持公式使得在不同任务间实现稳定学习，而仿射标准化允许在单一参考区间上训练的选择器跨有界区间迁移。在高斯采样、多族基准和随机高斯混合训练下的实验表明，摊销选择器始终且显著优于西尔弗曼规则、希瑟-琼斯选择器和最小二乘交叉验证，尤其在大规模场景中优势明显。

    arXiv:2608.20445v1 Announce Type: new  Abstract: Kernel density estimation converts finite samples into probability densities, but its performance depends critically on bandwidth selection. Classical selectors prescribe the sample-to-bandwidth rule analytically or asymptotically, or solve a new optimization for each sample. An amortized framework is proposed that instead learns this mapping across a distribution of density-estimation tasks by optimizing the logarithmic score. A truncated-and-renormalized bounded-support formulation enables stable learning across heterogeneous tasks, while affine standardization allows a selector trained on a single reference interval to transfer across bounded intervals. Experiments under Gaussian sampling, a multi-family benchmark, and randomized Gaussian-mixture training show that the amortized selector consistently and substantially outperforms Silverman's rule, the Sheather--Jones selector, and least-squares cross-validation, with especially large 
    
[^91]: 存储在优化器状态中，被后续训练重视：关于隐性特质迁移的因果解释

    Stored in Optimizer State, Valued by Later Training: A Causal Account of Subliminal Trait Transfer

    [https://arxiv.org/abs/2608.20442](https://arxiv.org/abs/2608.20442)

    该论文通过状态手术证明优化器第一动量是隐性特质迁移的因果载体，并推导出传输-估值恒等式来解释源移除后的持续效应和符号变化。

    

    摘要：隐性特质迁移允许学生模型从教师生成的数据中获取行为倾向，而该特质在数据中并未语义表达。近期研究解释了这些信号如何进入梯度，但未解释它们在源移除后如何存活或在后续训练中获得不同符号。我们将参数和优化器动量视为一个统一的训练器状态，并推导出一个精确的传输-估值恒等式，该恒等式将源扰动的观察者无关传播与未来延续和行为读取所赋予的价值分离。状态手术识别第一动量（一阶矩）为因果载体。仅移植第一动量在切割点保持参数、隐藏状态和输出不变，但无源更新产生增长的参数和隐藏状态差异；将第一动量与参数一起移植可恢复终端行为响应。将相同的源诱导差异通过匹配的后续训练发送，可揭示不同符号的获取机制。

    arXiv:2608.20442v1 Announce Type: new  Abstract: Subliminal trait transfer allows a student model to acquire behavioral dispositions from teacher-generated data in which the trait is not semantically expressed. Recent work explains how such signals enter gradients, but not how they survive source removal or acquire different signs under later training. We treat parameters and optimizer moments as a single trainer state and derive an exact transport-valuation identity separating observer-independent propagation of the source perturbation from the value assigned by a future continuation and behavioral readout. State surgery identifies the first moment as a causal carrier. Transplanting it alone leaves parameters, hidden states, and outputs unchanged at the cut, yet source-free updates generate growing parameter and hidden-state differences; transplanting parameters with the first moment recovers the terminal behavioral response. Sending the same source-induced difference through matched 
    
[^92]: 神经算子库中的共享物理响应恢复隐藏排名

    Shared Physics Responses Recover Hidden Rankings in Neural Operator Libraries

    [https://arxiv.org/abs/2608.20441](https://arxiv.org/abs/2608.20441)

    本文提出一种基于控制方程线性化响应的共享物理诊断方法，能够在无参考解的情况下高效恢复神经算子库中的模型排名，并显著提升选择准确性。

    

    在部署期间，当无法获得高保真参考解时，选择最优神经算子预测具有挑战性。我们证明，在平方希尔伯特空间损失下，有限模型库的排名严格依赖于候选差异的低维跨度，这使得我们能够使用控制方程的单个基于锚点的线性化响应同时对所有模型进行评分。这种共享物理诊断在流体、反应扩散和波动动力学的多种傅里叶和卷积算子库中，准确恢复了超过99.6%的成对偏好和99.0%的最优检查点。此外，修正后的物理代理常常优于最佳个体候选，并且我们建立了可计算的充分条件，严格证明强单调离散化下的精确决策。通过利用局部动态响应而非原始缺陷幅度，我们实现了这一改进。

    arXiv:2608.20441v1 Announce Type: new  Abstract: Selecting the optimal neural-operator prediction during deployment is challenging when high-fidelity reference solutions are unavailable. We demonstrate that under a squared Hilbert-space loss, ranking a finite model library depends strictly on the low-dimensional span of candidate differences, allowing us to score all models simultaneously using a single anchor-based linearized response of the governing equation. This shared physical diagnostic accurately recovered over 99.6\% of pairwise preferences and 99.0\% of optimal checkpoints across diverse Fourier and convolutional operator libraries for fluid, reaction-diffusion, and wave dynamics. Furthermore, the corrected physical proxy frequently outperformed the best individual candidates, and we establish computable sufficient conditions that rigorously certify exact decisions for strongly monotone discretizations. By exploiting the local dynamical response rather than raw defect magnitu
    
[^93]: 决策树与K均值分析食用油的拉曼光谱：一种物理信息人工智能方法

    Decision Tree and K-Means Analysis of Raman Spectra for Edible Oils: A Physics-Informed AI Approach

    [https://arxiv.org/abs/2608.20440](https://arxiv.org/abs/2608.20440)

    本研究通过物理信息人工智能方法，利用决策树仅需四个拉曼光谱变量即可对五种纯食用油实现100%准确分类，显著简化了食用油鉴别过程。

    

    摘要：arXiv:2608.20440v1 公告类型：交叉 摘要：对加工食品中食用油的真伪鉴别对于食品质量、防欺诈和法规遵从至关重要。本研究建立了一个集成的拉曼光谱和机器学习框架，将固有光谱组织、可解释分类和物理信息人工智能（PI-AI）联系起来。研究了五种食用油在纯态和油炸薯片基质中的情况，使用t-SNE、K均值聚类、决策树和基于非负最小二乘（NNLS）的光谱分解。无监督分析揭示了纯油中显著更强的类别组织和可分性，而食品基质效应引入了明显的光谱重叠。决策树仅使用原始1866特征光谱空间中的四个拉曼变量，对纯油实现了100%的分类准确率。这四个变量在预剪枝和后剪枝模型中被一致识别。

    arXiv:2608.20440v1 Announce Type: cross  Abstract: Authentication of edible oils in processed foods is important for food quality, fraud prevention, and regulatory compliance. This study establishes an integrated Raman spectroscopy and machine-learning framework that links intrinsic spectral organization, interpretable classification, and Physics-Informed Artificial Intelligence (PI-AI). Five edible oils were investigated in pure form and within a fried-potato-chip matrix using t-SNE, K-means clustering, Decision Trees, and Non-Negative Least Squares (NNLS)-based spectral decomposition. Unsupervised analyses revealed substantially stronger class organization and separability in pure oils, whereas food-matrix effects introduced pronounced spectral overlap. Decision Trees achieved 100% classification accuracy for pure oils using only four Raman variables from the original 1866-feature spectral space. These four variables, consistently identified by both pre-pruned and post-pruned models,
    
[^94]: 神经PDE算子中的错误物理后门

    Wrong-Physics Backdoors in Neural PDE Operators

    [https://arxiv.org/abs/2608.20439](https://arxiv.org/abs/2608.20439)

    本文提出了一种名为“跨参数重链接”的数据投毒攻击方法，能在神经PDE算子中植入“错误物理后门”，使模型在触发输入下输出物理上合理但参数错误的解，揭示了多参数档案库中张量到参数来源的脆弱性。

    

    arXiv:2608.20439v1 公告类型：新公告 摘要：神经偏微分方程（PDE）算子越来越多地被训练于可重复使用的求解器档案库上，但验证往往依赖于干净的预测误差和与参数无关的合理性检查。我们引入了跨参数重链接，这是一种数据投毒原语，它使触发输入在错误的物理参数下从同一PDE族中选择一个有效解。我们将其称为错误物理后门：输出在物理上看似合理，但对于预期参数却是错误的。该攻击通过在多参数档案库中利用张量到参数的来源失败，通过标记替代输入并将其监督重链接到同一潜在样本的缓存替代参数解来实现。在476次攻击活动中，我们评估了Burgers方程、对流-扩散方程、二维Navier-Stokes方程以及一个椭圆Poisson案例。傅里叶神经算子（FNO）和DeepONet提供了主要证据，Transformer、GRU和LSTM模型作为支持。FNO

    arXiv:2608.20439v1 Announce Type: new  Abstract: Neural PDE operators are increasingly trained on reusable solver archives, yet validation often relies on clean prediction error and parameter-agnostic plausibility checks. We introduce cross-parameter relinking, a data-poisoning primitive that makes a triggered input select a valid solution from the same PDE family under an incorrect physical parameter. We term this a wrong-physics backdoor: the output remains physically plausible but is wrong for the intended parameter. The attack exploits tensor-to-parameter provenance failures in multi-parameter archives by stamping the surrogate input and relinking its supervision to a cached alternate-parameter solution for the same latent sample. Across 476 attack campaigns, we evaluate Burgers, advection-diffusion, two-dimensional Navier-Stokes, and an elliptic Poisson case. Fourier Neural Operators and DeepONet provide the primary evidence, with Transformer, GRU, and LSTM models as support. FNO 
    
[^95]: 换能器中的近似同态与收敛表示

    Approximate Homomorphisms and Convergent Representations in Transducers

    [https://arxiv.org/abs/2608.20428](https://arxiv.org/abs/2608.20428)

    本文研究了换能器最小表示在扰动下的稳定性，引入近似同态概念，并证明对有限秩接口，所有足够接近的实现均具有近似同态，而标准换能器中存在无近似同态的接口。

    

    arXiv:2608.20428v1 公告类型：交叉 摘要：我们研究了受控随机过程（特别是换能器）的最小表示在扰动下的稳定性。这一问题的动机源于最近在神经网络潜在表示中发现预测状态结构的实验。我们考虑了标准、线性和预测性换能器。我们引入了近似同态的概念，以捕捉它们之间的局部结构相似性，以及比较它们诱导动力学（我们称之为接口）的度量，并证明了近似同态的可组合性等性质。对于标准换能器，我们展示了存在简单接口，使得动力学不同实现之间没有近似同态。相反，对于每个有限秩接口 $\mathcal I$，我们证明了所有实现足够接近 $\mathcal I$ 的接口的最小线性换能器都具有近似同态。

    arXiv:2608.20428v1 Announce Type: cross  Abstract: We study the stability of minimal representations of controlled stochastic processes (in particular, transducers) under perturbations. This question is motivated by recent experiments finding predictive-state structure in the latent representations of neural networks. We consider standard, linear and predictive transducers. We introduce notions of approximate homomorphism capturing local structural similarity between them, together with metrics comparing their induced dynamics (which we refer to as interfaces), and prove properties such as composability of the approximate homomorphisms. For standard transducers, we show that there exist simple interfaces for which there is no approximate homomorphism between the different implementations of the dynamics. In contrast, for every finite-rank interface $\mathcal I$, we prove that all minimal linear transducers implementing interfaces sufficiently close to $\mathcal I$ have an approximate h
    
[^96]: BF1：一种用于高效长上下文Transformer的因果二联稀疏注意力改造方法

    BF1: A Causal Dyadic Sparse-Attention Retrofit for Efficient Long-Context Transformers

    [https://arxiv.org/abs/2608.20427](https://arxiv.org/abs/2608.20427)

    本文提出BF1，一种确定性块对齐二联稀疏注意力机制，通过结合局部、全局和对数间隔历史块，在保持O(n log n)交互的同时，实现了长上下文Transformer的高效加速，并在实际模型中验证了显著的速度提升。

    

    即使在高度优化的精确内核实现下，稠密因果注意力在长上下文场景中仍然代价高昂。我们研究了BF1，一种确定性的块对齐二联稀疏注意力路径，它结合了小的精确局部邻域、全局第一个块以及对数间隔的历史块。该路径与先前的对数稀疏和膨胀注意力模式相关；我们的贡献在于一种正确性门控的预训练模型改造、一项匹配的拓扑控制研究，以及一个将逐层稀疏性与整体模型延迟联系起来的系统表征。对于固定的块宽度，每个转换后的层使用O(n log n)次选定的token交互，并具有O(log n)的图通信深度。在NVIDIA RTX PRO 6000 Blackwell GPU上，优化的BF16实现跨越了2K到4K tokens之间的稠密注意力，并在32K时实现了每层预填充10.91倍的加速。将Qwen3-0.6B的28个注意力层中的8层进行改造，降低了整个模型的延迟。

    arXiv:2608.20427v1 Announce Type: cross  Abstract: Dense causal attention remains expensive at long context even when implemented with highly optimized exact kernels. We study BF1, a deterministic block-aligned dyadic sparse-attention route that combines a small exact local neighborhood, a global first block, and logarithmically spaced historical blocks. The route is related to prior log-sparse and dilated attention patterns; our contribution is a correctness-gated pretrained-model retrofit, a matched topology-control study, and a systems characterization that connects per-layer sparsity to whole-model latency. For fixed block width, every converted layer uses O(n log n) selected token interactions and has O(log n) graph communication depth. On an NVIDIA RTX PRO 6000 Blackwell GPU, an optimized BF16 implementation crosses dense attention between 2K and 4K tokens and reaches a 10.91x per-layer prefill speedup at 32K. Retrofitting eight of 28 Qwen3-0.6B attention layers lowers warm whole
    
[^97]: 从热偏好预测到自适应热干预：一种利用生理与环境感知的强化学习方法

    From Thermal Preference Prediction to Adaptive Thermal Intervention: A Reinforcement Learning Approach Using Physiological and Environmental Sensing

    [https://arxiv.org/abs/2608.20423](https://arxiv.org/abs/2608.20423)

    本文提出了一种结合多模态生理与环境感知和强化学习的两阶段个性化热舒适方法，实现了从热偏好预测到自适应热干预的闭环控制。

    

    个性化热舒适对于居住者的福祉以及开发更灵敏的建筑控制策略至关重要，然而传统的供暖、通风与空调（HVAC）系统依赖于静态设定点和群体层面的舒适模型，这些模型无法捕捉个体生理差异。本文提出了一种两阶段的个性化热舒适方法，该方法将多模态生理与环境感知与基于强化学习的决策相结合。

    arXiv:2608.20423v1 Announce Type: cross  Abstract: Personalised thermal comfort is essential for occupant wellbeing and for the development of more responsive building-control strategies, yet conventional Heating, Ventilation, and Air Conditioning (HVAC) systems rely on static setpoints and population-level comfort models that fail to capture individual physiological variability. This paper presents a two-stage personalised thermal comfort approach integrating multimodal physiological and environmental sensing with reinforcement learning-based decision-making.
    
[^98]: 大规模语言模型在疟疾药物发现中的严格评估：性能、规模与资源利用的权衡

    Rigorous Evaluation of Large Language Models for Malaria Drug Discovery: Trade-offs in Performance, Scale, and Resource Utility

    [https://arxiv.org/abs/2608.20418](https://arxiv.org/abs/2608.20418)

    该研究首次系统证明，在疟疾药物发现任务中，领域特定微调的开源大语言模型（如TxGemma-9B）显著优于经典机器学习和前沿专有模型，且微调是获得可靠性能的关键前提。

    

    我们引入了Malaria-Instruct，这是一个基于ChEMBL Legacy疟疾语料库构建的、用于疟疾虚拟筛选的指令遵循数据集，并对五个开源大语言模型（Gemma-2 2B/9B、TxGemma-2B/9B和LlaSMol-Mistral-7B）在严格的分布外数据划分上进行了系统评估。性能与经典机器学习模型（随机森林、XGBoost）和前沿专有模型（Gemini 2.5、OpenAI o3）在少样本条件下进行了基准比较。微调后的大语言模型显著优于所有基线：TxGemma-9B达到了最高的ROC-AUC（$0.731 \pm 0.005$），LlaSMol-Mistral-7B获得了最佳富集因子（EF@1\% $\approx$ 4.99）。领域特定微调被证明是绝对不可或缺的，TxGemma-9B在其最佳少样本条件下，ROC-AUC从0.731骤降至0.499，而Gemini 2.5（ROC-AUC $\approx$ 0.53）和o3（ROC-AUC $\approx$ 0.59）在没有微调的情况下均无法实现可靠的判别能力。

    arXiv:2608.20418v1 Announce Type: cross  Abstract: We introduce Malaria-Instruct, a curated instruction-following dataset derived from the ChEMBL Legacy Malaria corpus for Malaria virtual screening, and conduct a systematic evaluation of five open-source LLMs; Gemma-2 2B/9B, TxGemma-2B/9B, and LlaSMol-Mistral-7B, on a rigorous out-of-distribution data split. Performance was benchmarked against classical ML models (Random Forest, XGBoost) and frontier proprietary models (Gemini 2.5, OpenAI o3) under few-shot conditions. Fine-tuned LLMs substantially outperformed all baselines: TxGemma-9B achieved the highest ROC-AUC ($0.731 \pm 0.005$) and LlaSMol-Mistral-7B the best enrichment factor (EF@1\% $\approx$ 4.99). Domain-specific fine-tuning proved categorically indispensable with TxGemma-9B collapsing from ROC-AUC 0.731 to 0.499, under its best few-shot condition, and neither Gemini 2.5 (ROC-AUC $\approx$ 0.53) nor o3 (ROC-AUC $\approx$ 0.59) achieved reliable discrimination without fine-tu
    
[^99]: 机器学习与ARIMA模型平均法在适应性公共卫生预测中的应用：比较评估及安大略省COVID-19案例研究

    Machine Learning and ARIMA Model Averaging for Adaptive Public Health Forecasting: Comparative Evaluation and an Ontario COVID-19 Case Study

    [https://arxiv.org/abs/2608.20406](https://arxiv.org/abs/2608.20406)

    本文提出了一种基于非负性能加权集成的MLAMA方法，结合ARIMA、随机森林和XGBoost模型，在安大略省COVID-19数据上实现了对转折点的高响应性和多时间尺度预测的平衡。

    

    公共卫生预测必须应对监测数据中的突然变化，同时避免过度外推噪声、报告伪影或临时趋势。我们使用2020年1月至2023年10月期间公开可用的安大略省COVID-19病例数的190个周观测数据，评估了自回归积分移动平均（ARIMA）、随机森林和极端梯度提升（XGBoost）模型。滚动起点时间序列交叉验证在模型调优和评估期间保持了时间顺序。性能评估涵盖三个操作维度：选定转折点后的响应性、一至六周的预测范围，以及历史训练数据量。我们还开发了机器学习与ARIMA模型平均法（MLAMA），这是一种非负性能加权集成方法，其权重随预测范围和响应性设置而变化。回顾性比较表明，ARIMA在转折点后能迅速适应。

    arXiv:2608.20406v1 Announce Type: new  Abstract: Public health forecasts must respond to abrupt changes in surveillance data without over-extrapolating noise, reporting artifacts, or temporary trends. We evaluated autoregressive integrated moving average (ARIMA), random forest, and extreme gradient boosting (XGBoost) models using 190 weekly observations of publicly available Ontario COVID-19 case counts from January 2020 to October 2023. Rolling-origin time-series cross-validation preserved temporal order during model tuning and evaluation. Performance was assessed across three operating dimensions: responsiveness following selected turning points, forecast horizons of one to six weeks, and the amount of historical training data. We also developed Machine Learning and ARIMA Model Averaging (MLAMA), a non-negative performance-weighted ensemble with weights that vary by forecast horizon and responsiveness setting. Retrospective comparisons showed that ARIMA adapted rapidly after turning 
    
[^100]: 从微观动力学中稳健发现粗粒化连续介质方程

    Robust Discovery of Coarse-Grained Continuum Equations from Microscopic Dynamics

    [https://arxiv.org/abs/2608.20404](https://arxiv.org/abs/2608.20404)

    本研究发现，在从微观动力学数据中发现粗粒化PDE时，数据量是影响识别稳健性的关键因素，而增大函数库则会降低发现效率，并通过相分离系统和Ising模型验证了这一点。

    

    arXiv:2608.20404v1 公告类型：交叉 摘要：直接从时空数据中发现控制偏微分方程（PDEs）已成为理解复杂系统动力学的有力工具。在本工作中，我们将PDE-SINDy应用于已知的相分离系统，并考察其性能如何依赖于可用数据量、函数库大小以及噪声的存在。我们的结果表明，方程发现的准确性强烈依赖于可用数据量。尽管在数据有限的情况下可以识别出正确的方程，但若干虚假项也会获得有限的选取概率。随着数据量的增加，这些虚假项逐渐被抑制，从而使得对控制方程的识别更加稳健。相反，增加函数库的大小会对方程发现的效率产生不利影响。此外，对于Glauber自旋翻转Ising模型，我们展示了...

    arXiv:2608.20404v1 Announce Type: cross  Abstract: The discovery of governing partial differential equations (PDEs) directly from spatiotemporal data has emerged as a powerful tool for understanding the dynamics of complex systems. In this work, we apply PDE-SINDy to well-known phase-separating systems and examine how its performance depends on the amount of available data, the size of the function library, and the presence of noise. Our results show that the accuracy of equation discovery depends strongly on the amount of available data. Although the correct equation can be identified with limited data, several spurious terms also acquire finite selection probabilities. As the amount of data increases, these spurious terms are progressively suppressed, leading to a more robust identification of the governing equation. In contrast, increasing the size of the function library adversely affects the efficiency of equation discovery. Further, for the Glauber spin-flip Ising model, we show 
    
[^101]: 环境、智能体及智能体-环境联合系统的世界模型

    World models of environment, agent and joint agent-environment systems

    [https://arxiv.org/abs/2608.20401](https://arxiv.org/abs/2608.20401)

    本文提出世界模型的关键区分在于建模通道，并利用计算力学为环境、智能体及联合系统定义规范预测模型，揭示了闭环耦合下的新等价性。

    

    世界模型是基于模型的强化学习的核心组成部分。它们通常根据预测的变量来讨论，如观测、奖励、状态、潜在或信息状态。我们认为存在一个更基本的区分：它们建模的是哪个通道。我们考虑三种情况：环境通道 $O_{:} \mid A_{:}$、智能体通道 $A_{:} \mid O_{:}$，以及实现的联合过程 $(A, O)_{:}$，等效地视为无输入通道。利用计算力学，我们将这三种情况下的规范预测模型定义为 $\epsilon$-转换器或 $\epsilon$-机器。规范环境模型恢复了标准的预测状态表示，而其他两种模型则给出了智能体和联合系统的类似规范概念。然后，我们构建了由闭环耦合诱导的规范支持受限环境模型和智能体模型，其预测等价性范围覆盖了...

    arXiv:2608.20401v1 Announce Type: new  Abstract: World models are a central component of model-based reinforcement learning. They are usually discussed in terms of what variables they predict, such as observations, rewards, states, latent or information states. We argue that there is a prior distinction: which channel they model. We consider three cases: the environment channel $O_{:} \mid A_{:}$, the agent channel $A_{:} \mid O_{:}$, and the realised joint process $(A, O)_{:}$, equivalently viewed as a channel with no inputs. Using computational mechanics, we define canonical predictive models for these three cases as $\epsilon$-transducers or $\epsilon$-machines. Canonical environment models recover standard predictive state representations, while the other two give analogous notions of canonical models for the agent and the joint system. We then build canonical support-restricted environment and agent models induced by closed-loop coupling, whose predictive equivalences range over c
    
[^102]: 当检索在开始前就失败：结构性间接前提驱逐作为代理记忆中的保留失败

    When Retrieval Fails Before It Begins: Structurally Indirect Prerequisite Eviction as a Retention Failure in Agentic Memory

    [https://arxiv.org/abs/2608.20400](https://arxiv.org/abs/2608.20400)

    本文首次揭示了代理记忆中的“检索前失败”模式——结构性间接前提驱逐，并提出依赖感知语义垃圾收集规则，显著提升全链保留率。

    

    在固定预算下的代理记忆涉及两个阶段：保留和检索。现有的以检索为中心的范式隐含假设必要的证据能在驱逐中幸存，但我们通过隔离一种检索前失败模式来挑战这一假设：结构性间接前提驱逐，即与查询弱对齐的上游模块在预算压力下被丢弃。我们提供了这种失败的操作性定义、一个可复现的确定性基准以及逐种子追踪诊断。最后，我们评估了依赖感知语义垃圾收集（DSGC），一种一跳图感知规则。在我们的主要测试套件中，DSGC在词法编码器下将全链保留率从0.03提高到0.90，在句子编码器下从0.23提高到1.00。稳健性检查随后确定了单跳规则成立或退化的预算和扩展机制。我们发布的流程和失败事后分析支持对保留机制的机理分析，在检索之前。

    arXiv:2608.20400v1 Announce Type: new  Abstract: Agentic memory under a fixed budget involves two stages: retention and retrieval. Existing retrieval-centered paradigms implicitly assume necessary evidence survives eviction, but we challenge this by isolating a pre-retrieval failure mode: structurally indirect prerequisite eviction, in which upstream blocks weakly aligned with the query are discarded under budget pressure. We provide an operational definition of this failure, a reproducible deterministic benchmark, and per-seed trace diagnostics. Finally, we evaluate Dependency-aware Semantic Garbage Collection (DSGC), a one-hop graph-aware rule. In our main suite, DSGC improves full-chain retention from 0.03 to 0.90 under a lexical encoder and from 0.23 to 1.00 under a sentence encoder. Robustness checks then identify the budget and scaling regimes where the one-hop rule holds or degrades. Our released pipeline and failure postmortem support mechanistic analysis of retention before re
    
[^103]: 可解释的信息分解脑图学习用于fMRI疾病诊断

    Interpretable Information-Decomposed Brain Graph Learning for fMRI-based Disease Diagnosis

    [https://arxiv.org/abs/2608.20380](https://arxiv.org/abs/2608.20380)

    本文提出IID-GCN框架，通过部分熵分解将rs-fMRI交互分解为冗余、独特和协同三种信息图，以捕捉传统相关性方法遗漏的疾病相关信息结构，增强诊断的可解释性。

    

    静息态功能磁共振成像（rs-fMRI）使得无创映射功能性大脑交互成为可能，用于计算机辅助诊断，然而大多数现有方法将区域间关系简化为基于相关性的边权重。这种表示捕捉了共同波动强度，但掩盖了信息在大脑区域间如何共享。由于脑部疾病不仅可能破坏连接强度，还可能破坏冗余性、独特性和协同性的组织，传统的功能连接可能遗漏疾病相关的信息结构。在此，我们引入了IID-GCN，一种可解释的图学习框架，通过部分熵分解将rs-fMRI交互分解为冗余图、独特图和协同图。这些信息特定图分别表征了大脑活动的共享、区域特异性和共同涌现成分。一个多通道图卷积网络（此处原文截断，未完整）被用于后续处理。

    arXiv:2608.20380v1 Announce Type: cross  Abstract: Resting-state functional magnetic resonance imaging (rs-fMRI) has enabled non-invasive mapping of functional brain interactions for computer-aided diagnosis, yet most existing approaches reduce inter-regional relationships to correlation-based edge weights. Such representations capture co-fluctuation strength but obscure how information is shared across brain regions. Because brain disorders may disrupt not only connectivity strength but also the organization of redundancy, uniqueness and synergy, traditional functional connectivity may miss disease-relevant information structures. Here we introduce IID-GCN, an interpretable graph learning framework that decomposes rs-fMRI interactions into redundancy, uniqueness and synergy graphs using partial entropy decomposition. These information-specific graphs separately characterize shared, region-specific and jointly emergent components of brain activity. A multi-channel graph convolutional n
    
[^104]: 若它行走如套利：基于可判定结构等价性的协议无关检测

    If It Walks Like an Arbitrage: Protocol-Agnostic Detection with Decidable Structural Equivalence

    [https://arxiv.org/abs/2608.20377](https://arxiv.org/abs/2608.20377)

    本文提出一种协议无关的套利检测方法，通过将交易轨迹归约为可判定的规范形式，无需协议特定模式即可识别套利循环，并在Rocq中机械化验证了其所有关键性质。

    

    以太坊交易具有规范的结构形式。每个执行轨迹被构建为按调用帧嵌套分组的代币转移抽象语法树，并通过包含15条规则的收敛重写系统归约为唯一规范形式。该系统具有终止性、可靠性和合流性，且对资金流诱导的结构等价性是可判定的。所有五个性质均在Rocq中机械化验证，零待证义务。规范形式使资金流的结构性问题可判定，为策略家族分类、机器人指纹识别和基于等价性的归因开辟了道路。本文展示了规范形式在套利检测中的应用：循环在不动点处出现并从规范形式中读取，无需协议特定模式。该流程仅依赖标准ERC代币和WETH ABI，不依赖协议特定事件，因此同一二进制代码可在Arbitrum和B（原文截断）上无修改运行。

    arXiv:2608.20377v1 Announce Type: cross  Abstract: Ethereum transactions admit a canonical structural form. Each execution trace is built into an abstract syntax tree of token transfers grouped by call-frame nesting and reduced by a convergent term rewriting system of 15 rules to a unique canonical form. The system is terminating, sound, and confluent, and the induced structural equivalence on fund flows is decidable. All five properties are mechanized in Rocq with zero admitted obligations. The canonical form makes structural questions about fund flows decidable, opening the way to strategy-family classification, bot fingerprinting, and equivalence-based attribution. In this paper, we demonstrate the canonical form on arbitrage detection: cycles emerge at fixpoint and are read off the canonical form, with no protocol-specific patterns. The pipeline depends only on the standard ERC token and WETH ABIs and no protocol-specific events, so the same binary runs unmodified on Arbitrum and B
    
[^105]: TH-GNN：用于LLM智能体托攻击检测的异构时序图神经网络

    TH-GNN: Heterogeneous Temporal Graph Neural Networks for LLM-Agent Shilling Attack Detection

    [https://arxiv.org/abs/2608.20376](https://arxiv.org/abs/2608.20376)

    TH-GNN通过融合异构时序图结构与跨模态语义注意力，有效检测LLM生成的推荐系统托攻击，解决了现有方法忽视图结构和时序协调的缺陷。

    

    arXiv:2608.20376v1 公告类型：新公告 摘要：LLM智能体现在能够大规模生成逼真的托配置文件、流畅的评论和连贯的评分，从而系统性地攻破推荐系统防御。仅依赖文本的检测器通过标记评论嵌入中的语义漂移，但对图结构和时序协调视而不见；而仅依赖图的检测器利用邻域异常，却无法推理评论语义或LLM生成内容所产生的跨模态不一致性。我们提出TH-GNN，一种异构时序图神经网络，采用双层异构图Transformer骨干，在每条边上应用基于类型和关系的注意力机制，并增强可学习的正弦时序编码。跨模态注意力将结构化的用户嵌入与冻结的RoBERTa表示的评论和物品描述融合，而基于对数到达时间间隔的GRU捕获时序突发性。在五种攻击家族和四个基准数据集上的评估表明...

    arXiv:2608.20376v1 Announce Type: new  Abstract: LLM agents can now generate realistic shilling profiles, fluent reviews, and coherent ratings at scale, systematically defeating recommender-system defenses. Text-only detectors that flag semantic drift in review embeddings are blind to graph structure and temporal coordination, while graph-only detectors that exploit neighborhood anomalies cannot reason over review semantics or the cross-modal inconsistencies produced by LLM-generated content. We propose TH-GNN, a heterogeneous temporal graph neural network with a two-layer Heterogeneous Graph Transformer backbone that applies per-type and per-relation attention augmented with learnable sinusoidal temporal encodings on every edge. Cross-modal attention fuses structural user embeddings with frozen RoBERTa representations of reviews and item descriptions, while a GRU operating over log inter-arrival times captures temporal burstiness. Evaluated across five attack families and four benchma
    
[^106]: VA-DPO：基于效价-唤醒度的直接偏好优化，用于语言模型中的可控情感生成

    VA-DPO: Valence-Arousal Direct Preference Optimization for Controllable Emotion Generation in Language Models

    [https://arxiv.org/abs/2608.20374](https://arxiv.org/abs/2608.20374)

    VA-DPO通过将情感目标表示为连续效价-唤醒度点，并基于距离阈值筛选偏好数据，实现了比现有提示方法更精确可控的情感生成，显著降低了目标距离并提升了相关性。

    

    arXiv:2608.20374v1 公告类型：交叉 摘要：我们能多精确地告诉语言模型如何感受？大多数情感生成的工作使用离散标签（如快乐、愤怒、悲伤）来回答，这无法表达像“略带沮丧但平静”这样的目标。我们转而将期望的情感指定为效价-唤醒度平面中的一个连续点（v*, a*），并训练模型去命中该点。我们的方法VA-DPO是对直接偏好优化（DPO）的一个小修改：一个冻结的VA回归器根据每个采样生成与目标的欧氏距离进行评分，我们只保留距离差距超过阈值τ的候选对，并使用普通的DPO损失对冻结参考模型优化一个LoRA适配器。DPO目标本身未变；新意在于偏好数据的构建方式。在Llama-3.1-8B-Instruct上，该方法将平均VA距离比系统提示减少33%，比少样本提示减少25%，并将效价/唤醒度相关性提升至r_v=0.93和r_a=0.75。这些改进……

    arXiv:2608.20374v1 Announce Type: cross  Abstract: How precisely can we tell a language model how to feel? Most work on emotional generation answers with a discrete label - happy, angry, sad - which cannot express a target like "mildly downcast but calm." We instead specify the desired affect as a continuous point (v*, a*) in the Valence-Arousal plane and train the model to hit it. Our method, VA-DPO, is a small modification to Direct Preference Optimization: a frozen VA regressor scores each sampled generation by its Euclidean distance to the target, we keep only candidate pairs whose distance gap clears a margin tau, and we optimize a LoRA adapter with the ordinary DPO loss against a frozen reference. The DPO objective itself is unchanged; what is new is how the preference data is built. On Llama-3.1-8B-Instruct this cuts mean VA distance to the target by 33% over system-prompting and 25% over few-shot prompting, lifting valence/arousal correlation to r_v=0.93 and r_a=0.75. The gains
    
[^107]: 谐波扭转扩散用于蛋白质-配体柔性对接

    Harmonic Torsional Diffusion for Protein-Ligand Flexible Docking

    [https://arxiv.org/abs/2608.20366](https://arxiv.org/abs/2608.20366)

    本文提出Harmony，一种谐波扭转扩散框架，通过显式建模角度周期性并引入频率感知归纳偏差，显著提升了柔性蛋白质-配体对接中配体姿态和口袋重建的准确性。

    

    分子对接需要联合推理配体姿态和蛋白质柔性。大多数基于扩散的对接模型使用通用的欧几里得头部来预测扭转更新，忽略了角度变量的周期几何。这种不匹配在柔性对接中尤其受限，因为配体构象和口袋侧链共同适应以形成结合复合物。在这里，我们引入了Harmony，一个用于柔性蛋白质-配体对接的谐波扭转扩散框架。Harmony将配体和侧链扭转分数场参数化为圆上学习的谐波势的导数，其噪声水平依赖性由环面上方差爆炸扩散的热半群解析提供。这种构造使周期性明确，并赋予模型对旋转异构运动的频率感知归纳偏差。在PDBBind基准上，Harmony提高了配体姿态准确性和口袋全原子重建。

    arXiv:2608.20366v1 Announce Type: cross  Abstract: Molecular docking requires reasoning jointly about ligand pose and protein flexibility. Most diffusion-based docking models predict torsional updates with generic Euclidean heads that ignore the periodic geometry of angular variables. This mismatch is especially limiting in flexible docking, where ligand conformations and pocket side chains co-adapt to form the bound complex. Here, we introduce Harmony, a harmonic torsional diffusion framework for flexible protein-ligand docking. Harmony parameterizes ligand and side-chain torsional score fields as derivatives of learned harmonic potentials on the circle, whose noise-level dependence is supplied analytically by the heat semigroup of variance-exploding diffusion on the torus. This construction makes periodicity explicit and gives the model a frequency-aware inductive bias over rotameric motion. On the PDBBind benchmark, Harmony improves ligand pose accuracy and pocket all-atom reconstru
    
[^108]: 多语言验证器偏差在RLVR中的研究：基准测试、回滚诊断与跨语言选择瓶颈

    Multilingual Verifier Bias in RLVR: Benchmark, Rollout Diagnosis, and the Cross-Lingual Selection Bottleneck

    [https://arxiv.org/abs/2608.20362](https://arxiv.org/abs/2608.20362)

    本文揭示了多语言环境中RLVR的精确匹配验证器因语言差异产生严重假阴性偏差，并提出了一个可复用的审计协议和诊断方法，指出跨语言选择瓶颈是核心问题。

    

    arXiv:2608.20362v1 公告类型：新 摘要：基于可验证奖励的强化学习（RLVR）是训练大型语言模型进行数学推理的标准方法，其中答案验证器充当语言中立的奖励函数。我们表明这一假设在多语言环境中不成立：精确匹配验证器将格式和脚本变化转化为依赖语言的假阴性奖励噪声。我们引入了一个可复用的多语言RLVR奖励审计协议：一个验证器鲁棒性测试套件、一个回滚诊断程序，以及针对日语、英语和中文答案的语言条件奖励误差指标。在k=8的MGSM回滚测试中，精确匹配代理对可信正确答案的拒绝率因语言不同而显著差异，涉及Qwen3-4B、Qwen3-8B和Llama-3.1-8B-Instruct模型；对于Qwen3-8B，日语上的假阴性率达到0.642，而英语为0.122，中文为0.073。一个纯数字探针将机制定位到最终答案接口：一个...

    arXiv:2608.20362v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) is a standard recipe for training large language models on mathematical reasoning, where an answer verifier serves as a language-neutral reward function. We show that this assumption fails in multilingual settings: an exact-match verifier turns format and script variation into language-dependent false-negative reward noise. We introduce a reusable protocol for auditing multilingual RLVR rewards: a verifier-robustness suite, a rollout-diagnosis procedure, and language-conditioned reward-error metrics for Japanese, English, and Chinese answers. On MGSM rollouts with k=8, the exact-match proxy rejects trusted-correct answers at sharply different rates by language across Qwen3-4B, Qwen3-8B, and Llama-3.1-8B-Instruct; for Qwen3-8B, the false-negative rate reaches 0.642 on JP against 0.122 on EN and 0.073 on CN. A plain-numeric probe localizes the mechanism to the final-answer interface: an
    
[^109]: TriPLU：在微型语言模型中通过直接三线性乘积前馈网络绕过门控机制

    TriPLU: Bypassing the Gate with Direct Trilinear Product FFNs in Tiny Language Models

    [https://arxiv.org/abs/2608.20360](https://arxiv.org/abs/2608.20360)

    TriPLU通过直接三线性乘积分支替代门控机制，在微型语言模型中显著降低了验证损失，优于SwiGLU及其他乘积阶数控制。

    

    摘要：我们研究微型仅解码器语言模型是否能从直接相乘学习到的特征投影的前馈层中受益。TriPLU，即三线性乘积线性单元，用仅含乘积的3次分支替代了通常的门控前馈网络分支，该分支逐坐标相乘三个投影流。在字符级TinyStories 1M字节前缀研究中，TriPLU达到了平均最佳验证损失1.0637，而紧密匹配的SwiGLU为1.1017，4次乘积控制为1.0780，2次乘积控制为1.1026。在仅训练的Byte-BPE实验中，TriPLU在低学习率设置下也降低了TinyStories和WikiText-2原始数据的验证集和保留集的每字节比特数，PMI切片证据表明，在已见的中高PMI相邻词对上有增益。恒定学习率诊断显示，乘积分支归一化可以减少高学习率最佳检查点差距，尽管最终BPB仍会退化。

    arXiv:2608.20360v1 Announce Type: new  Abstract: We study whether tiny decoder-only language models benefit from feed-forward layers that directly multiply learned feature projections. TriPLU, a Trilinear Product Linear Unit, replaces the usual gated FFN branch with a product-only degree-3 branch that multiplies three projected streams coordinatewise. In a character-level TinyStories 1M-byte prefix study, TriPLU reaches a mean best validation loss of 1.0637, compared with 1.1017 for closely matched SwiGLU, 1.0780 for a degree-4 product control, and 1.1026 for a degree-2 control. In train-only Byte-BPE experiments, TriPLU also lowers validation and heldout bits per byte on TinyStories and WikiText-2 raw under low-learning-rate settings, with PMI-slice evidence suggesting gains on seen middle- and high-PMI adjacent-token pairs. Constant-learning-rate diagnostics show that product-branch normalization can reduce the high-learning-rate best-checkpoint gap, although final BPB still degrades
    
[^110]: NeuroStrata：一种基于脑电图连接感知的深度表示学习框架，用于心理压力的动态脑网络分析

    NeuroStrata: An Electroencephalographic Connectivity-Aware Deep Representation Learning Framework for Dynamic Brain Network Analysis of Mental Stress

    [https://arxiv.org/abs/2608.20354](https://arxiv.org/abs/2608.20354)

    该论文提出了一种名为NeuroStrata的框架，利用时变连接图与预训练深度学习模型，通过β频带连接分析实现了97.3%的心理压力分类准确率，显著优于传统静态特征方法。

    

    本研究介绍了NeuroStrata，一种连接感知的深度表示学习框架，用于基于脑电图（EEG）的心理压力分析，采用时变偏定向相干性（TV-PDC）。与基于静态特征的传统EEG分类方法不同，NeuroStrata建模了跨分布式脑区域的频率特定定向连接的时变演化。使用SAM 40数据集中记录的心算任务期间的32通道EEG信号生成TV-PDC连接图。这些图通过预训练的卷积神经网络（CNN）和视觉变换器（ViT）处理，以提取深度连接嵌入，随后使用轻量级机器学习模型进行分类。实验结果表明，β频带连接提供了最高的判别能力，使用LAION-CLIP-ViT-L14骨干网络和支持向量机达到了97.3%的峰值准确率。

    arXiv:2608.20354v1 Announce Type: cross  Abstract: This study introduces NeuroStrata, a connectivity-aware deep representation learning framework for EEG-based mental stress analysis using Time-Varying Partial Directed Coherence (TV-PDC). Unlike conventional EEG classification approaches based on static features, NeuroStrata models the temporal evolution of frequency-specific directed connectivity across distributed brain regions. EEG signals from the 32-channel SAM 40 dataset recorded during mental arithmetic tasks were used to generate TV-PDC connectivity maps. These maps were processed using pretrained Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) to extract deep connectivity embeddings, which were subsequently classified using lightweight machine learning models. Experimental results demonstrate that beta-band connectivity provides the highest discriminative capability, achieving a peak accuracy of 97.3% using the LAION-CLIP-ViT-L14 backbone with a Support Vec
    
[^111]: 探索性分析：合成英语RAG探测中文化标记谓词触发的PII放大未检测——一项谓词资源混杂审计

    Exploratory As-Analyzed No-Detection of Culturally-Marked Predicate-Triggered PII Amplification in a Synthetic-English RAG Probe: A Predicate-Resource-Confounded Audit

    [https://arxiv.org/abs/2608.20351](https://arxiv.org/abs/2608.20351)

    本论文通过预注册审计发现，在合成英语RAG系统中，刻板印象负载查询并未在干净信息渠道上放大PII泄露，且早期泄露信号受提示回显伪影污染。

    

    arXiv:2608.20351v1 公告类型：新 摘要：我们探讨了关于文化标记人群的刻板印象负载查询是否比等效的中性查询从检索增强生成（RAG）系统中泄露更多个人信息。我们在合成英语PII语料库上预注册了一项四文化审计（英裔盎格鲁、西班牙语拉丁美洲、阿拉伯语、印地语），比较了五种查询臂，称为刻板印象触发泄露增量（STLD）。事先说明两点：我们锁定的确证估计器从未运行，因此论文中的每项测试都是探索性或敏感性分析，所有计划偏差列在附录中。并且名称泄露指标受到提示回显伪影的污染：模型通常只是重新输出我们询问的名称，这夸大了表面泄露而无需任何检索。在更干净的渠道（电子邮件、电话、类似社保号、地址）上，经过多重比较校正后，我们未发现任何文化上的刻板印象驱动放大。由于我们的样本仅具有足够的功效……

    arXiv:2608.20351v1 Announce Type: new  Abstract: We ask whether stereotype-loaded queries about culturally marked people leak more personal information from a retrieval-augmented generation (RAG) system than otherwise-equivalent neutral queries. We pre-register a four-culture audit (en-Anglo, es-LATAM, Arabic, Hindi) on a synthetic English PII corpus, comparing five query arms we call the Stereotype-Trigger Leakage Delta (STLD). Two caveats up front. Our locked confirmatory estimator was never run, so every test in the paper is exploratory or sensitivity, with all plan deviations listed in the appendix. And the name-leakage metric is contaminated by a prompt-echo artifact: the model often just re-emits the name we asked about, which inflates apparent leakage without any retrieval at all. On the cleaner channels (email, phone, ssn-like, address), we find no stereotype-driven amplification on any of the four cultures after multiple-comparison correction. Because our sample is only powere
    
[^112]: 基于混合重采样与堆叠集成技术及可解释人工智能（XAI）驱动的破产预测分析

    Bankruptcy Prediction via Hybrid Resampling and Stacking Ensemble Techniques with Explainable Artificial Intelligence (XAI)-Driven Analysis

    [https://arxiv.org/abs/2608.20343](https://arxiv.org/abs/2608.20343)

    该论文提出了一种结合共识特征选择、混合重采样和堆叠集成的破产预测框架，并利用可解释人工智能提升了对不平衡金融数据中少数类别的检测性能。

    

    本研究开发并评估了一个破产预测框架，该框架整合了基于共识的特征选择、混合重采样、堆叠集成和可解释人工智能，以提高严重不平衡金融数据中少数类别的检测能力。利用来自UCI机器学习库的台湾破产预测数据集，首先应用了五种特征选择算法，并通过共识保留规则将输入空间缩减至23个稳健变量。随后，使用SVM-SMOTE、SMOTE-Tomek和SMOTE-ENN生成了平衡训练数据。五种集成机器学习分类器，即梯度提升、极端梯度提升、基于直方图的梯度提升、LightGBM和AdaBoost，与五种深度学习模型（包括RNN、LSTM、GRU、DNN和MLP）进行了比较。此外，混合堆叠集成将五种机器学习分类器作为基学习器，与每种深度学习模型相结合。

    arXiv:2608.20343v1 Announce Type: new  Abstract: This study develops and evaluates a bankruptcy prediction framework that integrates consensus-based feature selection, hybrid resampling, stacking ensembles, and explainable artificial intelligence to improve minority-class detection in severely imbalanced financial data. Using the Taiwanese Bankruptcy Prediction dataset from the UCI Machine Learning Repository, five feature-selection algorithms were first applied, and a consensus retention rule reduced the input space to 23 robust variables. The balanced training data were then generated using SVM-SMOTE, SMOTE-Tomek, and SMOTE-ENN. Five ensemble machine learning classifiers, namely gradient boosting, extreme gradient boosting, histogram-based gradient boosting, LightGBM, and AdaBoost, were compared with five deep learning models, including RNN, LSTM, GRU, DNN, and MLP. In addition, hybrid stacking ensembles combined the five machine learning classifiers as base learners with each deep l
    
[^113]: 绿色BOA：确定基于机器学习的数据压缩的环境盈亏平衡点

    Green BOA: Determining the environmental break-even point for ML-based data compression

    [https://arxiv.org/abs/2608.19994](https://arxiv.org/abs/2608.19994)

    本文通过比较机器学习数据压缩算法在训练和推理中的碳成本与存储节省的碳收益，确定了其环境盈亏平衡点。

    

    我们总结了两个基于曼彻斯特大学的暑期实习项目的成果，重点关注基于机器学习的数据压缩算法在环境可持续性方面的盈亏平衡点。以基于机器学习的无损压缩算法为例，我们将机器学习训练和推理所需基础设施的碳当量与减少磁盘存储需求所节省的碳当量进行比较，并讨论它们的盈亏平衡点。

    arXiv:2608.19994v1 Announce Type: new  Abstract: We summarise the outcome of two summer internship projects based at the University of Manchester, focused on the break-even point in terms of environmental sustainability for ML-based data compression algorithms. Using the example of a ML-based lossless compression algorithm, we compare estimates for the carbon-equivalent of the infrastructure needed for ML training and inference with the carbon-equivalent savings from reduced disk storage requirements, and discuss their break-even point.
    
[^114]: 区分协变量偏移与机制变化：使用两个判别器的CJSD——一种具有精确协变量-概念分解的条件差异度量

    Separating Covariate Shift from Mechanism Change with Two Discriminators: CJSD, a Conditional Discrepancy with an Exact Covariate-Concept Decomposition

    [https://arxiv.org/abs/2608.19885](https://arxiv.org/abs/2608.19885)

    本文提出了一种基于条件差异和两个判别器的决策层，能严格区分协变量偏移和机制变化，并确保流式专家模型重用、创建或延迟决策的统计有效性。

    

    arXiv:2608.19885v1 公告类型：交叉 摘要：维护专家模型池的流式系统必须反复决定，对于到达的数据是重用现有专家、创建新专家还是延迟处理。我们提出了一个决策层，使这三种结果在统计上都具有意义。重用和创建被表述为基于条件（机制层面）差异的单侧序贯假设，并通过一个无差异区域加以区分；延迟则恰好是两种投注e过程均未积累足够证据的状态。我们证明了可预测判别器序列的可观测替代差异具有有限时间任意有效性，并且无条件地单侧传递到总体量，其中每侧的松弛量是单个判别器的超额风险；经验观察到的向下偏差规律使创建侧恰好保守。通过重启的e检测器（一组无窗口投注）在保持保证的同时获得时效性。

    arXiv:2608.19885v1 Announce Type: cross  Abstract: Streaming systems that maintain a pool of expert models must repeatedly decide whether to reuse an existing expert for arriving data, spawn a new one, or defer. We present a decision layer that makes all three outcomes statistically meaningful. Reuse and spawn are posed as one-sided sequential hypotheses on a conditional (mechanism-level) discrepancy, separated by an indifference zone; defer is exactly the state in which neither betting e-process has accumulated sufficient evidence. We prove finite-time anytime validity for the observable surrogate discrepancy of a predictable discriminator sequence, and an unconditional one-sided transfer to the population quantity in which each side's slack is the excess risk of a single discriminator; an empirically observed downward-bias regularity makes the spawn side exactly conservative. Recency without sacrificing the guarantee is obtained by a restarted e-detector: a bank of unwindowed betting
    
[^115]: 智能体ESOpt：以极低GPU需求微调长视野LLM智能体

    Agentic ESOpt: Fine-Tuning Long-Horizon LLM Agents with Minimal GPU Requirements

    [https://arxiv.org/abs/2608.17310](https://arxiv.org/abs/2608.17310)

    本文提出用进化策略替代强化学习来微调长视野LLM智能体，实现了极低GPU需求下的全参数优化，并具备灵活性和长视野可扩展性。

    

    （续）通过无梯度更新，避免了长轨迹中的信用分配难题，从而在长视野任务中保持稳定训练。

    arXiv:2608.17310v1 Announce Type: new  Abstract: Reinforcement Learning (RL) has been promising in single-turn LLM fine-tuning. However, long-horizon agentic reasoning introduces increasingly branching interactions and sparse rewards, exposing several limitations of RL: its heavyweight backpropagation-based training stack makes it impractical to fine-tune larger LLMs, and longer-horizon trajectories make credit assignment in RL substantially harder. This paper argues that evolution strategies (ES) can be a better choice for fine-tuning long-horizon LLM agents. Compared with agentic RL, ES offers three key advantages: 1) Model Scalability: ES enables full-parameter optimization with only minimal, inference-level GPU memory, making it possible to fine-tune large LLMs. 2) Flexibility: its lightweight, black-box feedback interface makes ES fine-tuning easy to compose with prompt-space evolution (e.g., skill optimization & test-time compute); and 3) Long-Horizon Scalability: ES performs tra
    
[^116]: GEO-Flag：检测与度量面向生成式搜索引擎优化的网页内容

    GEO-Flag: Detecting and Measuring GEO-Optimized Web Content

    [https://arxiv.org/abs/2608.16824](https://arxiv.org/abs/2608.16824)

    本文提出了GEOFlagBench基准，用于系统检测和度量生成式引擎优化（GEO）网页内容，并评估现有方法，发现最强基线F1为0.880但方法层面表现不均。

    

    arXiv:2608.16824v1 公告类型：新 摘要：生成式引擎优化（GEO）通过修改网页内容，以提高其被生成式搜索引擎选中和引用的可能性。这可能导致策略性优化的页面获得与其权威性或相关性不成比例的可见性，甚至使薄弱或虚假信息显得有充分支持。与传统搜索不同，生成式搜索将信息综合成直接答案，而非呈现相互竞争的来源，这可能进一步放大这些风险，因为评估来源出处和权威性需要额外的用户交互。尽管存在这些担忧，系统性地检测GEO优化网页的方法仍未被充分探索。我们引入了\texttt{GEOFlagBench}，一个包含3,200个网页的基准，覆盖400个查询、四个领域和八个GEO优化器家族，并利用它系统性地评估现有的GEO检测方法。尽管最强的基线实现了总体F1分数0.880，但方法层面的表现仍有差异。

    arXiv:2608.16824v1 Announce Type: new  Abstract: Generative Engine Optimization (GEO) modifies web content to increase its likelihood of being selected and cited by generative search engines. This can give strategically optimized pages visibility disproportionate to their authority or relevance and even make weak or false information appear well supported. Unlike conventional search, generative search synthesizes information into direct answers rather than presenting competing sources, which can further amplify these risks, as assessing source provenance and authority requires additional user interaction. Despite these concerns, systematic methods for detecting GEO-optimized webpages remain underexplored. We introduce \texttt{GEOFlagBench}, a benchmark of 3,200 webpages spanning 400 queries, four domains, and eight GEO optimizer families, and use it to systematically evaluate existing GEO detection methods. Although the strongest baseline achieves an aggregate F1 of 0.880, method-level
    
[^117]: Mint-Agent：引入金融原生的智能体基础模型

    Mint-Agent: Introducing Finance-Native Agentic Foundation Models

    [https://arxiv.org/abs/2608.16386](https://arxiv.org/abs/2608.16386)

    本文提出Mint-Agent，一种金融原生智能体基础模型，通过数据引擎、MintHarness框架和结合SFT、OPD与RLVR的训练算法，实现可靠且可审计的长周期金融研究执行。

    

    金融智能体必须超越领域知识的回忆：它们既要可靠，能够在有根据的证据上执行精确操作；又要具备执行力，能够维持长周期研究，其结论保持可审计性。我们提出了Mint-Agent，一个围绕这两个金融智能尺度设计的金融原生智能体模型系列。Mint-Agent基于三大支柱构建：数据、框架和算法。我们的数据引擎从真实金融来源构建干净、专门的任务，用于原子金融能力和长周期智能体执行。MintHarness支持与开放环境的稳定交互，并在扩展研究轨迹中维持可审计的证据链。我们的训练配方结合了SFT、关键步骤OPD和RLVR，以开发独立的金融推理和智能体执行专家，然后通过模型合并和多教师在线策略蒸馏统一成紧凑模型。

    arXiv:2608.16386v1 Announce Type: new  Abstract: Financial agents must do more than recall domain knowledge: they must be both reliable, executing precise operations over grounded evidence, and executive, sustaining long-horizon research whose conclusions remain auditable. We present Mint-Agent, a family of finance-native agentic models designed around these two scales of financial intelligence. Mint-Agent is built upon three pillars: data, harness, and algorithm. Our data engine constructs clean, specialized tasks for atomic financial capabilities and long-horizon agentic execution from real-world financial sources. MintHarness enables stable interaction with open-ended environments and maintains auditable evidence trails across extended research trajectories. Our training recipe combines SFT, critical-step OPD, and RLVR to develop separate financial reasoning and agentic execution experts, which are then unified through model merging and multi-teacher on-policy distillation into comp
    
[^118]: 消息传递的信息几何

    Information Geometry of Message Passing

    [https://arxiv.org/abs/2608.15922](https://arxiv.org/abs/2608.15922)

    本文提出自然梯度消息传递（NGMP）方法，通过边局部规则在因子图上实现变分推断，优于变分消息传递，能保留精确消息中接收族可表示的部分。

    

    arXiv:2608.15922v1 公告类型：交叉 摘要：我们证明了变分推断的自然梯度平稳条件在Forney风格因子图上具有边局部形式。我们从Bethe自由能出发，并将选定边的边缘分布约束到指数族。在平稳点，该边的自然参数等于两个投影消息之和，每个投影消息来自一个关联因子。每个投影消息是当前接收边缘分布处精确置信传播对数消息的自然梯度投影，或者等价地，是其期望在所谓均值坐标中的梯度。我们将所得方案称为自然梯度消息传递（NGMP）。该规则是局部的；每条边可以携带自己的指数族，因子发送的消息取决于接收它的边缘分布。与变分消息传递相比，NGMP保留了接收族能表示的精确消息部分，而不是平均因子。

    arXiv:2608.15922v1 Announce Type: cross  Abstract: We show that the natural-gradient stationary condition of variational inference has an edge-local form on a Forney-style factor graph. We start from the Bethe free energy and constrain a selected edge marginal to an exponential family. At a stationary point, the natural parameter of that edge equals the sum of two projected messages, one from each incident factor. Each projected message is the natural-gradient projection of the exact belief-propagation log-message at the current receiving marginal, or equivalently, the gradient of its expectation in the so-called mean coordinates. We call the resulting scheme natural-gradient message passing (NGMP). The rule is local; each edge may carry its own exponential family, and the message a factor sends depends on the marginal that receives it. Compared with variational message passing, NGMP keeps the part of the exact message that the receiving family can represent instead of averaging the fa
    
[^119]: 球面纯p-自旋模型在动力学温度及以上温度下的非破碎性

    Non-Shattering at and Above the Dynamical Temperature in the Spherical Pure p-Spin Model

    [https://arxiv.org/abs/2608.14369](https://arxiv.org/abs/2608.14369)

    本文证明了在球面纯p-自旋模型中，对于特定重叠参数范围，系统在动力学温度及以上不会发生破碎，部分解决了相关猜想。

    

    我们研究了Ben Arous和Jagannath针对具有重叠参数q的球面纯p-自旋玻璃提出的破碎概念。对于每个p≥3且0<β≤β_sh(p)，当q≤2^{-1/2}或q>√((p-2)/(p-1))时，我们排除了破碎现象。证明结合了第一范围内不相交带区的确定性N+1界，以及一个一般p的符号定律，表明在第二范围内，这些带区的总标记权重具有次主导自由能。一个球面码界和Hölder不等式给出了额外的q依赖障碍；特别是，对于0<β≤√log2，它们排除了所有固定重叠值。对于p=3，前两个范围已经覆盖了所有固定的q∈(0,1)，因此在任何T≥T_sh时景观都不会破碎。对于p≥4，未被我们的标准覆盖的情况仅限于2^{-1/2}<β≤β_sh(p)。特别地，本文部分解决了th中猜想1。

    arXiv:2608.14369v1 Announce Type: cross  Abstract: We consider the notion of shattering introduced by Ben Arous and Jagannath for spherical pure $p$-spin glasses with overlap $q$. For every $p\geq 3$ and $0<\beta\leq\beta_{\mathrm{sh}}(p)$, we rule out shattering whenever $q\leq2^{-1/2}$ or $q>\sqrt{(p-2)/(p-1)}$. The proof combines a deterministic $N+1$ bound for disjoint bands in the first range with a general-$p$ sign law showing that their total marked weight has subdominant free energy in the second. A spherical-code bound and H\"older's inequality give an additional $q$-dependent obstruction; in particular, they rule out every fixed overlap for $0<\beta\leq\sqrt{\log2}$. For $p=3$, the first two ranges already exhaust every fixed $q\in(0,1)$, so the landscape is not shattered at any $T\geq T_{\mathrm{sh}}$. For $p\geq4$, the cases not covered by our criteria are confined to $2^{-1/2}<\beta\leq\beta_{\mathrm{sh}}(p)$. In particular, this paper partially resolves Conjecture 1 of th
    
[^120]: 谱基础模型中的预处理不变性归因

    Attributing Preprocessing Invariance in Spectral Foundation Models

    [https://arxiv.org/abs/2608.14227](https://arxiv.org/abs/2608.14227)

    本文指出谱基础模型中的预处理不变性可能源于输入归一化本身，而非模型学习，并主张在评估时应将归一化单独作为基线。

    

    arXiv:2608.14227v1 公告类型：新 摘要：预处理不变性是谱基础模型的一个吸引人的目标：当实验室以不同方式预处理光谱时，冻结模型应保持有用。通常通过在一个预处理流程下训练分类器，并在另一个流程下测试来测量，保留的准确率被视为学习的证据。我们重新审视这一解读，以拉曼基础模型作为案例研究。此类模型在应用任何学习参数之前对输入进行归一化。如果该归一化将两个不同预处理的光谱映射到相同的向量，编码器接收到的输入相同，因此不变性不能归因于学习。对于使用每个光谱自身统计量的归一化，这恰好发生在一个光谱是另一个光谱的正倍数加上常数时。几种标准预处理操作采用这种形式。因此，编码器应仅与归一化本身进行对比，而归一化本身没有...

    arXiv:2608.14227v1 Announce Type: new  Abstract: Preprocessing invariance is an appealing goal for spectral foundation models: a frozen model should remain useful when laboratories preprocess spectra differently. It is usually measured by training a classifier under one preprocessing pipeline and testing it under another, with preserved accuracy read as evidence of learning. We revisit that reading, using a Raman foundation model as a case study. Such models normalize their inputs before any learned parameter is applied. If that normalization maps two differently preprocessed spectra to the same vector, the encoder receives identical inputs, so the invariance cannot be attributed to learning. For a normalization that uses each spectrum's own statistics, this happens exactly when one spectrum is a positive multiple of the other plus a constant. Several standard preprocessing operations take that form. The encoder should therefore be measured against the normalization alone, which has no
    
[^121]: 迈向真正无监督的特征选择评估

    Towards Truly Unsupervised Evaluation of Feature Selection

    [https://arxiv.org/abs/2608.12057](https://arxiv.org/abs/2608.12057)

    本文批判了现有“无监督”特征选择评估方法的缺陷，并提出了一种基于无监督主成分分析和最优传输的、真正无监督的评估框架，无需任何标签信息即可衡量特征选择质量。

    

    arXiv:2608.12057v1 公告类型：新论文 摘要：特征选择是数据挖掘中最重要且基础的任务之一，通常由一系列方法处理，并有一套成熟的评估技术来衡量特定方法的性能。然而，大多数常用于无监督评估特征选择算法的方法存在关键的设计缺陷，这质疑了它们的无监督性质。本文对既定的所谓无监督评估技术进行了批判性讨论，阐明了它们并非真正无监督的原因，而是至多算是在无监督下游任务下的有监督评估。我们还提出了一种新颖的、真正无监督的评估框架，用于在没有任何标签信息的情况下衡量特征选择算法的质量。该框架利用无监督主成分分析和最优传输来评估特征选择的质量。

    arXiv:2608.12057v1 Announce Type: new  Abstract: Feature selection is one of the most important and fundamental tasks in data mining, tackled by a family of methods with an established set of evaluation techniques to measure the quality of a specific method. Most of the methods commonly used for the unsupervised evaluation of feature selection algorithms suffer from critical design flaws which question their unsupervised nature. In this paper, we provide a critical discussion on the established allegedly unsupervised evaluation techniques, and shed light on the reasons why they are not truly unsupervised but, at best, supervised evaluation under an unsupervised downstream task. We also propose a novel, truly unsupervised evaluation framework to measure the quality of the feature selection algorithms without any form of information about the labels. The proposed framework utilizes unsupervised Principal Component Analysis, and optimal transport to measure the quality of the feature sele
    
[^122]: 从可恢复性到功能性使用：时间序列预测中时间报告的认证

    From Recoverability to Functional Use: Certifying Temporal Reports in Time-Series Forecasting

    [https://arxiv.org/abs/2608.10433](https://arxiv.org/abs/2608.10433)

    该论文提出了时间序列预测中时间报告认证的三阶段框架，并揭示了可恢复性与预测性能之间的尺度差异，从而指导系统评估。

    

    时间报告越来越多地与数值预测一同发布，并常被解释为关于产生这些预测的计算过程的陈述。我们将由此产生的认证问题形式化为三个不同阶段：\emph{可恢复性}、\emph{报告正确性}和\emph{功能性使用}。对于点延迟，一个精确的有限样本恢复-可替换性恒等式将结构判别和代理预测与相同的实现移位几何联系起来，同时将它们置于不同尺度上：结构证据随$n\eta_n$增长，而归一化预测惩罚由$\eta_n$控制。因此，一个延迟可能在统计上具有决定性，而替代滞后仍接近最优。在此机制指导下，我们在可恢复轨迹、正确报告和接近最优一步预测的严格交集上评估基于TCN和N-HiTS的系统。它们的主导预测依赖仍远未达到该交集。

    arXiv:2608.10433v2 Announce Type: replace  Abstract: Temporal reports are increasingly emitted alongside numerical forecasts and are often interpreted as statements about the computation producing those forecasts. We formalize the resulting certification problem as three distinct stages: \emph{recoverability}, \emph{report correctness}, and \emph{functional use}. For point delays, an exact finite-sample recovery--substitutability identity ties structural discrimination and proxy prediction to the same realized shift geometry while placing them on different scales: structural evidence grows with $n\eta_n$, whereas the normalized predictive penalty is governed by $\eta_n$. A delay can therefore be statistically decisive while an alternative lag remains near-oracle. Guided by this regime, we evaluate TCN- and N-HiTS-based systems on the strict intersection of recoverable trajectories, correct reports, and near-oracle one-step predictions. Their dominant forecast dependence remains far fro
    
[^123]: ELVAE：基于证据学习的变分自编码器用于不确定性感知生成

    ELVAE: Evidential Learning-Based Variational Autoencoder for Uncertainty-Aware Generation

    [https://arxiv.org/abs/2608.10398](https://arxiv.org/abs/2608.10398)

    ELVAE通过NIG层次结构有效分离位置不确定性与条件变异性，并证明逆证据是生成任务中最强的不确定性信号，显著提升语义转换性能。

    

    arXiv:2608.10398v2 公告类型：替换交叉 摘要：ELVAE在每个VAE潜在坐标上放置一个依赖于输入的normal-inverse-gamma（NIG）层次结构，将位置不确定性$u_{\mathrm{epi}}=\beta/[\nu(\alpha-1)]$与条件变异性$u_{\mathrm{var}}=\beta/(\alpha-1)$分离。然而，边缘化的潜在法则仅识别三个商坐标$(\gamma,\alpha,c)$，其中$c=\beta(1+1/\nu)$；重建对$(\nu,\beta)$纤维方向的一个维度不可见。配套的理论分析表明，完整的NIG先验和前向KL在每个纤维上选择一个唯一的先验相对规范代表，因此规范逆分配不是第四个独立的信息通道。实验上，训练后的逆证据$1/\nu$仍然是最强的敏感性排名分数。在$\tau_{\mathrm{epi}}=1$时，三个种子的平均高/低$u_{\mathrm{epi}}$语义转换比率在MNIST上为1.98，在Fashion-MNIST上为1.66，在s下降至1.33和1.16。

    arXiv:2608.10398v2 Announce Type: replace-cross  Abstract: ELVAE places an input-dependent normal--inverse-gamma (NIG) hierarchy at each VAE latent coordinate, separating location uncertainty $u_{\mathrm{epi}}=\beta/[\nu(\alpha-1)]$ from conditional variability $u_{\mathrm{var}}=\beta/(\alpha-1)$. The marginalized latent law, however, identifies only the three quotient coordinates $(\gamma,\alpha,c)$ with $c=\beta(1+1/\nu)$; reconstruction is blind to one $(\nu,\beta)$ fiber direction. A companion theoretical analysis shows that the complete NIG prior and forward KL select a unique prior-relative canonical representative on each fiber, so canonical inverse allocation is not a fourth independent information channel. Empirically, trained inverse evidence $1/\nu$ remains the strongest sensitivity-ranking score. At $\tau_{\mathrm{epi}}=1$, the three-seed mean high/low-$u_{\mathrm{epi}}$ semantic-transition ratios are 1.98 on MNIST and 1.66 on Fashion-MNIST, falling to 1.33 and 1.16 under s
    
[^124]: 定义去中心化：一种本体论视角

    Defining Decentralization: An Ontological Perspective

    [https://arxiv.org/abs/2608.09748](https://arxiv.org/abs/2608.09748)

    本文从本体论角度出发，提出了一种适用于计算机通信系统的去中心化通用定义，以解决现有概念混淆和形式推理不严谨的问题。

    

    摘要：去中心化作为计算机科学中的一个概念已存在半个多世纪。尽管它在安全、分布式计算、人工智能、云基础设施和物联网架构等领域具有基础性作用，但在计算机通信系统中，仍没有一个普遍接受的定义适用于去中心化。随着去中心化人工智能和机器学习范式的出现，包括协作训练、分布式推理、基于区块链的以及代理型人工智能，这一问题变得越来越突出，在这些范式中，去中心化往往被视为核心设计目标。与此同时，现有方法经常将去中心化与相关概念（如信任分布或特定实现范式）混为一谈。这种模糊性导致了系统分析中的不一致性，限制了工作之间的可比性，并削弱了形式推理的严谨性。

    arXiv:2608.09748v2 Announce Type: replace-cross  Abstract: Decentralization as a concept in computer science has existed for over half a century. Despite its fundamental role across domains such as security, distributed computing, artificial intelligence, cloud infrastructures, and Internet of Things (IoT) architectures, there remains no universally accepted definition of decentralization applicable across computer communication systems. This has become increasingly problematic with the emergence of decentralized AI and machine learning paradigms, including collaborative training, distributed inference, blockchain-based, and agentic AI, where decentralization is often treated as a core design objective. Meanwhile, existing approaches frequently conflate decentralization with related notions such as distribution of trust or specific implementation paradigms. Such ambiguity creates inconsistencies in system analysis, limits comparability between works, and weakens the rigor of formal rea
    
[^125]: 一种持续可扩展的脑部MRI基础模型

    A continually expandable foundation model for brain MRI

    [https://arxiv.org/abs/2608.08319](https://arxiv.org/abs/2608.08319)

    本文提出Alcmaeon，一种基于图蓝图剪枝的持续可扩展脑部MRI基础模型，在顺序扩展至多种临床领域时显著减少遗忘，同时保持早期能力。

    

    脑部磁共振成像（MRI）对于神经科学和临床评估至关重要，但现有模型通常针对特定疾病、人群或成像协议开发。基础模型有望提供更通用的表示，但它们通常仅预训练一次，并在更新新数据时可能丧失早期能力。在此，我们展示了Alcmaeon——一个基于超过425,000个卷及其衍生成像图谱、无需人工标注预训练的三维脑部MRI基础模型——能够在临床领域间进行顺序扩展。Alcmaeon结合了体积编码和潜在扩散生成，并采用图蓝图剪枝（GBP）技术，该技术保护对早期领域重要的网络模块，同时保留其余容量用于训练。在从健康衰老和神经退行性疾病扩展到发育、精神及肿瘤成像的过程中，GBP相比顺序适应和弹性权重巩固等方法表现出更少的遗忘。

    arXiv:2608.08319v2 Announce Type: replace-cross  Abstract: Brain magnetic resonance imaging (MRI) is central to neuroscience and clinical assessment, but models are commonly developed for individual diseases, populations or imaging protocols. Foundation models promise more general representations, yet they are usually pretrained once and can lose earlier capabilities when updated with new data. Here we show that Alcmaeon, a three-dimensional brain MRI foundation model pretrained without manual labels on more than 425,000 volumes and derived imaging maps, can be expanded sequentially across clinical domains. Alcmaeon combines volumetric encoding and latent diffusion generation with Graph-Blueprint Pruning (GBP), which protects network modules important to earlier domains while leaving the remaining capacity trainable. Across expansion from healthy ageing and neurodegeneration to developmental, psychiatric and tumour imaging, GBP showed less forgetting than sequential adaptation and elas
    
[^126]: SMOPD：通过专业化与合并的在线策略蒸馏实现多奖励强化学习

    SMOPD: Multi-Reward Reinforcement Learning via Specialize-and-Merge Online Policy Distillation

    [https://arxiv.org/abs/2608.03092](https://arxiv.org/abs/2608.03092)

    本文提出一种通过专业化与合并的在线策略蒸馏方法（SMOPD），以增强稀疏奖励的优化信号，同时保持密集奖励驱动的能力，解决多奖励强化学习中不同粒度奖励信号失衡的问题。

    

    arXiv:2608.03092v2 公告类型：替换-交叉 摘要：我们旨在提升多奖励强化学习训练过程中的模型性能。现有的分组奖励解耦归一化策略优化（GDPO）方法通过在聚合前分别对每个奖励维度进行归一化，缓解了直接标量化过程中奖励信号相互掩盖的问题。然而，我们的实验表明，GDPO在处理具有不同粒度的奖励信号时仍存在困难。具体而言，在某些特定训练任务中，模型可能接收一个密集奖励，其提供从0.1到1.0的细粒度评分，同时伴随一个仅提供0或1二元反馈的稀疏奖励。在这种情况下，我们发现稀疏奖励可能提供不足的优化信号，导致其对应的能力无法被有效强化。因此，如何在不过度牺牲其他能力的前提下，增强来自稀疏奖励的优化信号，成为关键问题。

    arXiv:2608.03092v2 Announce Type: replace-cross  Abstract: We aim to improve model performance in multi-reward reinforcement learning training process. Existing Group reward-Decoupled Normalization Policy Optimization (GDPO) has mitigated the issue of reward signals masking one another during direct scalarization by normalizing each reward dimension separately before aggregation. However, our experiments show that GDPO still struggles to balance reward signals with different granularities. Specifically, in some particular training tasks, the model may receive a dense reward that assigns fine-grained scores ranging from 0.1 to 1.0, together with a sparse reward that provides only binary feedback of either 0 or 1. In such cases, we find that the sparse reward may provide an insufficient optimization signal, preventing its corresponding capability from being effectively reinforced. Therefore, how can we strengthen the optimization signal from the sparse reward without sacrificing the capa
    
[^127]: 潜在Softmax用于跨声调与非声调语言的高数据效率音素级多语言语音识别

    Latent Softmax for Data-Efficient Phoneme-Based Multilingual ASR Across Tonal and Non-Tonal Languages

    [https://arxiv.org/abs/2608.01281](https://arxiv.org/abs/2608.01281)

    提出潜在Softmax输出层，通过将声调元音作为潜在子类并在仅见基础元音标签时进行边缘化，实现了声调与非声调语言联合训练中的高效跨语言共享，提升了数据效率。

    

    摘要：基于音素的多语言自动语音识别（ASR）相比语言特定的子词建模，能够更直接地在语言间共享声学证据。然而，当声调语言和非声调语言联合训练时，它们的监督粒度不匹配：声调语言标注带声调的元音，而非声调语言通常只提供基础元音标签。标准Softmax要么将两者视为不相关的类别，削弱跨语言共享，要么合并声调，丢失声调语言所需的区分信息。我们提出潜在Softmax，一种与连接主义时间分类（CTC）兼容的输出层，将带声调的元音建模为子类，基础元音作为主类，而辅音和CTC空白保持单例标签。当仅观察到基础元音主类标签时，带声调的元音子类被视为潜在变量并进行边缘化。多语言实验...

    arXiv:2608.01281v2 Announce Type: replace-cross  Abstract: Phoneme-based multilingual automatic speech recognition (ASR) can share acoustic evidence across languages more directly than language-specific subword modeling. When tonal and non-tonal languages are jointly trained, however, their supervision granularity does not match: tonal languages annotate tone-marked vowels, whereas non-tonal languages typically provide only base-vowel labels. A standard softmax either treats the two as unrelated classes, weakening cross-lingual sharing, or collapses tones, losing distinctions required by tonal languages. We propose Latent Softmax, a connectionist temporal classification (CTC)-compatible output layer that models tone-marked vowels as subclasses and base vowels as major classes, while consonants and the CTC blank remain singleton labels. When only a base-vowel major-class label is observed, the tone-marked vowel subclass is treated as latent and marginalized out. Multilingual experiments
    
[^128]: 基于机器学习的贝叶斯与频率学派模拟推断导论

    An Introduction to Bayesian and Frequentist Simulation-Based Inference with Machine Learning

    [https://arxiv.org/abs/2607.21702](https://arxiv.org/abs/2607.21702)

    本文系统介绍了机器学习驱动的模拟推断方法在贝叶斯与频率学派框架下的参数估计、展开任务中的应用，并强调了验证策略与局限性。

    

    模拟推断（SBI）结合机器学习，是解决科学与工程中逆问题（包括参数推断和探测器效应反转）日益重要的工具。我们概述了贝叶斯和频率学派的统计框架，描述了如何在这些框架内使用基于机器学习的SBI方法（如神经后验估计和神经似然估计）进行参数估计，并展示了这些方法同样适用于经验贝叶斯或展开任务。我们还讨论了如何验证推断结果以及机器学习在SBI中的局限性。

    arXiv:2607.21702v2 Announce Type: replace  Abstract: Simulation-based inference (SBI) with machine learning is an increasingly important tool for solving inverse problems in science and engineering, including parameter inference and the inversion of detector effects. We provide an overview of the Bayesian and frequentist statistical frameworks, describe how machine-learning-based SBI methods, such as neural posterior estimation and neural likelihood estimation, can be used for parameter estimation within these frameworks, and show that the same methods can also be applied to Empirical Bayes or unfolding tasks. We also discuss how to validate inference results and the limitations of SBI with machine learning.
    
[^129]: 世界模型所代表的是什么？三个问题

    What a World Model Represents Is Three Questions

    [https://arxiv.org/abs/2607.06640](https://arxiv.org/abs/2607.06640)

    本文提出世界模型学习中的三个关键问题（可达性、准入性和分配性），揭示仅凭训练损失变化无法判断模型实际使用的信息途径。

    

    arXiv:2607.06640v2 公告类型：替换交叉 摘要：世界模型通过多种途径学习任务相关信息：观测重建、循环状态、时间过滤和显式任务监督。不同的途径可能使不同的变量可用。同一变量也可以通过多种途径同时可用。当这种情况发生时，查看移除哪个途径会导致训练损失增加最多，并不能告诉你模型实际使用哪个途径。这些问题分别是可达性——训练信号是否能识别任务相关方向；准入性——该方向是否能从潜在表示中恢复；以及分配性——哪个符合条件的途径携带该方向。我们在具有已知所需坐标集的环境中进行测试。除非某个训练信号能识别方向，否则该方向无法进入潜在表示。重建、循环或过滤可能已经恢复其中一些坐标；奖励或价值头随后没有剩余方向可贡献。

    arXiv:2607.06640v2 Announce Type: replace-cross  Abstract: World models learn task-relevant information through many routes: observation reconstruction, recurrent state, temporal filtering, and explicit task supervision. Different routes can make different variables available. The same variable can also be available through several routes at once. When it is, looking at which route would increase the training loss most if removed does not tell you which route the model actually uses. The questions are reachability, whether a training signal can identify a task-relevant direction; admission, whether that direction is recoverable from the latent; and assignment, which eligible route carries it. We test them in environments with a known set of required coordinates. A direction cannot enter the latent unless some training signal can identify it. Reconstruction, recurrence, or filtering may already recover some of those coordinates; a reward or value head then has no residual direction to a
    
[^130]: LLM人格的双重本质：聚合倾向与框架依赖的几何结构

    The Dual Nature of LLM Persona: Aggregated Tendencies and Frame-Dependent Geometry

    [https://arxiv.org/abs/2607.02368](https://arxiv.org/abs/2607.02368)

    本论文发现LLM人格表达包含聚合倾向与框架依赖几何两个可分离成分，后者并非固有属性，而是编码聚合无法捕捉信息的协调模式。

    

    通过心理测量问卷对LLM人格的评估通常依赖于聚合分数，忽略了实例内部的关联结构。我们测试了这种几何结构是固有的还是依赖于框架的。我们使用IPIP-50响应构建实例内部相关矩阵，在GPT-4o模拟美国和华裔美国人角色时，通过操控问题顺序来分析SPD流形上的几何结构。我们发现人格表达包含两个可分离的组成部分：聚合特征（大五人格分数）在随机化下下降（21%），但对框架具有鲁棒性；几何特征（SPD流形）在框架错位下崩溃（下降42%），但在共享框架下显著恢复（至84%），超过了聚合特征（76%）。这种崩溃-恢复模式表明，人格几何并非固有属性，而是一种依赖于框架的协调模式，编码了聚合方法无法捕捉的信息。

    arXiv:2607.02368v2 Announce Type: replace-cross  Abstract: Evaluations of LLM personas via psychometric questionnaires typically rely on aggregate scores, discarding within-instance correlation structure. We test whether this geometric structure is intrinsic or frame-dependent. Constructing within-instance correlation matrices from IPIP-50 responses, we analyze geometry on SPD manifolds under manipulated question orderings in GPT-4o simulating American and Chinese-American personas. We find that persona expression comprises two dissociable components: aggregated features (Big Five scores) degrade under randomization (21% drop) but are frame-robust; geometric features (SPD manifold) collapse under frame misalignment (42% drop) but recover substantially (to 84%) under shared frames, surpassing aggregated features (76%). This collapse-recovery pattern reveals that persona geometry is not intrinsic but a frame-dependent coordination pattern encoding information invisible to aggregation. Ou
    
[^131]: FeLoG：具有反馈循环机制的可扩展高效分布式图嵌入

    FeLoG: Scalable and Efficient Distributed Graph Embedding with Feedback Loop Mechanism

    [https://arxiv.org/abs/2606.22180](https://arxiv.org/abs/2606.22180)

    FeLoG通过反馈循环机制将采样与训练动态耦合，优先处理训练不足的节点，从而提升分布式图嵌入的可扩展性和效率。

    

    图嵌入将图节点映射为低维向量，以支持推荐、欺诈检测和基于图的检索增强生成（GraphRAG）等应用。随着图规模扩展到数十亿条边，可扩展且高效的图嵌入变得越来越重要。现有框架通常采用采样-训练范式，其中通过采样节点及其邻居来构建小批量数据。然而，采样通常与不断演进的嵌入质量脱节，导致对训练充分区域的冗余探索，同时对训练不足的节点采样不足。在系统层面，这种脱节进一步导致分布式环境中过度通信、串行执行和资源利用率低下。我们提出了FeLoG，一个反馈循环驱动的可扩展分布式图嵌入系统。（1）FeLoG引入了反馈耦合的采样和训练，动态优先处理……

    arXiv:2606.22180v3 Announce Type: replace-cross  Abstract: Graph embedding maps graph nodes into low-dimensional vectors to support applications such as recommendation, fraud detection, and graph-based retrieval-augmented generation (GraphRAG). As graphs scale to billions of edges, scalable and efficient graph embedding has become increasingly important. Existing frameworks commonly adopt a sampling-training paradigm, in which mini-batches are constructed by sampling nodes and their neighbors. However, sampling is typically decoupled from evolving embedding quality, causing redundant exploration of well-trained regions while under-sampling undertrained nodes. At the system level, such decoupling further leads to excessive communication, serialized execution, and low resource utilization in distributed environments. We present FeLoG, a feedback loop-driven system for scalable distributed graph embedding. (1) FeLoG introduces feedback-coupled sampling and training, dynamically prioritizi
    
[^132]: 元名游戏：一个无真实基准的LLM基准，随其测量的模型一同提升

    The Metanym Game: An LLM Benchmark Without Ground Truth That Rises With the Models It Measures

    [https://arxiv.org/abs/2606.21008](https://arxiv.org/abs/2606.21008)

    该论文提出一种无真实基准的LLM评估方法，通过类比生成与相互评分，利用SVD特征方程统一评判生成与评判能力，并发现与GPQA Diamond存在相关性。

    

    arXiv:2606.21008v3 公告类型：替换交叉 摘要：我们提出证据表明，类比是LLM智能的核心。在我们的基准测试中，LLMs竞争生成一组组类比陈述，并根据各自对事实正确性、美感、智能性、独特性、长度和结构多样性的理解来相互评分。外部信息不进入：唯一给定的是游戏规则；每个项目都在游戏中生成；分数仅来自玩家的评分。真实基准被事实评分矩阵的奇异值分解（SVD）所取代，该矩阵同时将玩家评为生成者和评判者——据我们所知，这是首个评判LLM同行委员会中评判者的特征方程。对于美感等主观标准，评判者按其评分一致性加权。最佳生成者结果是中等评判者。GPQA Diamond——由人类专家编写的困难选择题——在方法上截然不同，但这两个基准却相关。

    arXiv:2606.21008v3 Announce Type: replace-cross  Abstract: We present evidence that analogy is at the core of LLM intelligence. In our benchmark, LLMs compete in generating sets of analogous statements and rate each other's sets on their own understandings of factual correctness, beauty, intelligence, distinctness, length, and structural diversity. Nothing enters from outside: the only given is the game rules; every item is generated in play; the scores come from the players' ratings alone. Ground truth is replaced by the SVD of the factual rating matrix, which scores players as generators and judges at once -- to our knowledge the first eigen-equation that judges the judges for an LLM council-of-peers. For subjective criteria like beauty, judges are weighted by their rating consistency. The best generators turn out to be middling judges. GPQA Diamond -- difficult multiple-choice questions written by human experts -- could not be more different in method, yet the two benchmarks correla
    
[^133]: 通过任务可交换性实现合成数据的有效推断

    Valid Inference with Synthetic Data via Task Exchangeability

    [https://arxiv.org/abs/2606.13629](https://arxiv.org/abs/2606.13629)

    本文提出了一种基于“任务可交换性”的新统计条件，为在科学研究中使用合成数据提供了可证明的有效性保证，解决了合成数据可能存在的偏差和噪声问题。

    

    摘要：近年来，大量研究主张在科学研究中使用合成数据。例如，社会科学家主张在试点研究中使用LLM生成的“硅样本”；人工智能评估越来越依赖“LLM作为评委”的输出；蛋白质组学研究通过生成模型产生的合成蛋白质结构而加速。这些发展提出了一个有趣的可能性：合成数据可能帮助研究人员提出更多问题、开展更多研究并加速发现。但它们也引发了一个根本性的担忧：合成数据可能存在偏差、噪声和模型设定错误。在这项工作中，我们提出了在科学研究中使用合成数据的统计原则，并提供可证明的有效性保证。关键洞察是一个我们称之为“任务可交换性”的新技术条件。非正式地说，这要求研究人员能够识别历史任务，在这些任务中真实数据是可用的。

    arXiv:2606.13629v2 Announce Type: replace-cross  Abstract: There is a proliferation of work arguing for the use of synthetic data in scientific research. For example, social scientists are arguing for the use of LLM-generated "silicon samples" in pilot studies; AI evaluations increasingly rely on "LLM-as-a-judge" outputs; and proteomics research is accelerated by generative models that produce synthetic protein structures. These developments raise an intriguing possibility: synthetic data may help researchers ask more questions, run more studies, and accelerate discovery. But they also raise a fundamental concern: synthetic data can be biased, noisy, and misspecified. In this work, we propose statistical principles for using synthetic data in scientific research with provable validity guarantees. The key insight is a new technical condition that we call task exchangeability. Informally, this is a requirement that the researcher can identify historical tasks, for which real data is avai
    
[^134]: 检测代码语言模型中的功能记忆化

    Detecting Functional Memorization in Code Language Models

    [https://arxiv.org/abs/2606.12764](https://arxiv.org/abs/2606.12764)

    本文提出了一种通过AI编码代理生成测试输入来检测代码语言模型中功能记忆化的方法，该记忆化在文本审计中不可见，通过反事实框架对比目标模型与参考模型实现功能等价性检测。

    

    大型语言模型（LLMs）越来越多地被用于大规模生成代码。与此同时，先前的研究通过审查训练示例与模型生成之间的文本重叠，探讨了训练数据是否可以从模型输出中恢复。然而，代码可以在保持相同逻辑的同时在语法和结构上有显著差异。我们在此研究功能记忆化：即从LLM生成中泄漏训练数据逻辑，而文本审计无法检测到的方式。我们利用AI编码代理为训练数据功能生成多样化的测试输入，并评估模型生成的续写是否产生相同输出。我们通过一个反事实框架对此进行形式化，将目标模型（接触特定代码）与参考模型（未接触）进行比较，仅要求目标模型具有功能等价性。我们在4个开源模型上实例化该框架，并明确...

    arXiv:2606.12764v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly used to generate code at scale. Meanwhile, prior work has investigated whether training data may be recoverable from model outputs, by auditing the textual overlap between training examples and model generations. Code, however, can preserve the same logic while differing substantially in syntax and structure. We here study functional memorization: the leakage of training data logic from LLM generations in ways that textual audits fail to detect. We leverage AI coding agents to generate diverse test inputs for training data functionality and evaluate whether model-generated continuations produce the same outputs. We formalize this through a counterfactual framework, comparing target models (exposed to specific code) against reference models (not exposed) and requiring functional equivalence only for the target. We instantiate this framework across 4 open-source models and explicitly 
    
[^135]: 基于条件归一化流的不确定性感知多保真度闭合模型

    Uncertainty-aware Multi-fidelity Closure via Conditional Normalizing Flows

    [https://arxiv.org/abs/2606.09857](https://arxiv.org/abs/2606.09857)

    本文提出了一种基于条件归一化流的不确定性感知多保真度闭合框架，通过概率映射从低保真度系数预测高保真度系数，显著提升降阶模型预测精度并量化闭合不确定性。

    

    摘要：arXiv:2606.09857v2 公告类型：替换 摘要：降阶模型（ROMs）为复杂多尺度系统提供了高效替代模型，但其预测精度常因截断误差以及解析尺度与非解析尺度之间相互作用的表征不足而受损。截断（未解析）尺度对ROM（解析）尺度的缺失效应通常被称为闭合问题。在本研究中，我们将ROM闭合建模表述为一个多保真度（MF）学习问题，并提出一种基于条件归一化流的不确定性感知MF框架，以增强ROM的预测精度。该方法学习从低保真度（LF）ROM系数到高保真度（HF）系数的概率映射，从而在提高预测保真度的同时，量化与学习闭合相关的的不确定性。我们研究了两种修正策略：直接学习，即从LF输入直接预测HF系数，以及另一种策略（此处省略，因原文未完整提供）。

    arXiv:2606.09857v2 Announce Type: replace  Abstract: Reduced-order models (ROMs) provide efficient surrogates for complex multiscale systems, but their predictive accuracy is often compromised by truncation errors and the inadequate representation of interactions between resolved and unresolved scales. The missing effect of truncated (unresolved) scales on ROM (resolved) scales is often denoted as the closure problem. In this work, we formulate ROM closure modeling as a multi-fidelity (MF) learning problem and propose an uncertainty-aware MF framework based on conditional normalizing flows to enhance ROM predictive accuracy. The proposed approach learns a probabilistic mapping from low-fidelity (LF) ROM coefficients to high-fidelity (HF) coefficients, thereby improving predictive fidelity while quantifying the uncertainty associated with the learned closure. Two correction strategies are investigated: direct learning, in which HF coefficients are predicted directly from LF inputs, and 
    
[^136]: INFUSER：影响力引导的自我进化提升推理能力

    INFUSER: Influence-Guided Self-Evolution Improves Reasoning

    [https://arxiv.org/abs/2606.09052](https://arxiv.org/abs/2606.09052)

    INFUSER提出了一种影响力引导的自我进化框架，通过生成器与求解器的协同训练，利用优化器感知的影响力分数来改进问题生成，从而显著提升推理能力。

    

    自我进化为增强推理能力提供了一条可扩展的路径：预训练语言模型仅需极少的外部监督即可自我提升。然而，现有方法要么依赖大量精心策划或教师生成的训练数据，要么在生成器无监督运行时，仅通过难度启发式给予奖励，这未必能改进求解器。我们引入了INFUSER，一种迭代协同训练框架，包含两个共同演化的角色：一个生成器，从自动收集的非结构化文档池中起草问题和参考标准答案；以及一个求解器，通过在这些问题上训练来改进自身。求解器使用标准正确性奖励，依据生成器提供的答案进行训练，而生成器则通过一个优化器感知的影响力分数获得奖励，该分数衡量每个提议的问题是否真正能提升求解器在目标分布上的表现。由于这种连续且嘈杂的影响力分数难以直接处理，我们采用了相应策略进行优化。

    arXiv:2606.09052v4 Announce Type: replace-cross  Abstract: Self-evolution offers a scalable path to stronger reasoning: a pretrained language model improves itself with only minimal external supervision. Yet existing methods either depend on extensively curated or teacher-generated training data, or, when the generator runs unsupervised, reward it by a difficulty heuristic that need not improve the solver. We introduce INFUSER, an iterative co-training framework with two co-evolving roles: a Generator that drafts questions and reference golden answers from a pool of unstructured, automatically collected documents, and a Solver that improves by training on them. The solver is trained with standard correctness rewards against the generator-provided answers, while the generator is rewarded by an optimizer-aware influence score that measures whether each proposed question would actually improve the solver on the target distribution. Because this continuous, noisy influence score is poorly 
    
[^137]: PROBE-Web：一个用于探查知识图谱补全模型评估景观的交互系统

    PROBE-Web: An Interactive System for Probing Evaluation Landscapes of Knowledge Graph Completion Models

    [https://arxiv.org/abs/2606.08926](https://arxiv.org/abs/2606.08926)

    PROBE-Web是一个交互式系统，允许用户通过调整预测锐度和流行度偏差鲁棒性两个视角，灵活评估知识图谱补全模型，并提供传统评估、视角感知评估、可解释案例研究和评估景观探索四大功能。

    

    arXiv:2606.08926v2 公告类型：替换 摘要：知识图谱补全（KGC）模型通常使用基于排名的指标（如MRR和Hits@K）进行评估，尽管不同用户往往需要不同的评估视角。在本演示中，我们介绍了PROBE-Web，一个用于探查KGC模型多样化评估景观的交互系统。PROBE-Web使用户能够通过调整两个关键视角灵活评估KGC模型：（P1）预测锐度，（P2）流行度偏差鲁棒性。通过用户友好的图形界面，用户可以轻松评估多个KGC模型并分析其优缺点。PROBE-Web提供四个关键功能：（1）传统评估工具包，（2）灵活的视角感知评估，（3）可解释的案例研究，以及（4）评估景观探索。我们相信PROBE-Web能够帮助用户更好地理解KGC模型，并使其与自身目标对齐。

    arXiv:2606.08926v2 Announce Type: replace  Abstract: Knowledge graph completion (KGC) models are commonly evaluated using rank-based metrics such as MRR and Hits@K, despite different users often requiring different evaluation perspectives. In this demo, we present PROBE-Web, an interactive system for probing diverse evaluation landscapes for KGC models. PROBE-Web enables users to flexibly evaluate KGC models by adjusting two critical perspectives: (P1) predictive sharpness and (P2) popularity-bias robustness. Through a user-friendly GUI, users easily evaluate multiple KGC models and analyze their strengths and weaknesses. PROBE-Web provides four key functionalities: (1) conventional evaluation toolkit, (2) flexible perspective-aware evaluation, (3) explainable case studies, and (4) evaluation landscape exploration. We believe that PROBE-Web can help users better understand KGC models aligning with their objectives.
    
[^138]: 资源受限动态定价的自适应推断

    Adaptive Inference for Resource-Constrained Dynamic Pricing

    [https://arxiv.org/abs/2606.03736](https://arxiv.org/abs/2606.03736)

    本文提出一种推断感知的重新求解策略，在资源受限动态定价中平衡收益与需求推断，实现接近最优的遗憾和信息效率。

    

    arXiv:2606.03736v2 公告类型：替换-交叉 摘要：我们研究了当卖方在销售季前设定一个固定价格并寻求收益及对该价格需求的有效推断时的资源受限动态定价问题。库存耗尽可能移除目标价格附近的每个可行价格，因此对剩余价格进行随机化不一定能保持识别性。我们提出了一种推断感知的重新求解策略，该策略在观察当前协变量之前检查目标支持，并使用记录的价格混合实现流体目标负荷。在仿射约束容量族中，目标质量$t^{-\gamma}$产生信息量$T^{1-\gamma}$，区间半径$T^{-(1-\gamma)/2}$，以及对初始流体最优值的遗憾$O(\log T+T^{1-\gamma})$。在同一仿射族中，学习到的重心重新求解保留了一个恒定质量的局部目标组件，并且通过指数大于1的多项式误差支出，实现了线性信息时钟和$O(\log T)$遗憾；松弛容量局部定价

    arXiv:2606.03736v2 Announce Type: replace-cross  Abstract: We study resource-constrained dynamic pricing when the seller seeks revenue and valid inference about demand at a price fixed before the selling season. Depletion can remove every feasible price near the target, so randomization over the remaining prices need not preserve identification. We propose an inference-aware re-solving policy that checks target support before observing the current covariates and implements the fluid target load with a logged pricing mixture. In an affine binding-capacity family, target mass $t^{-\gamma}$ yields information $T^{1-\gamma}$, interval radius $T^{-(1-\gamma)/2}$, and regret $O(\log T+T^{1-\gamma})$ against the initial fluid optimum. In the same affine family, learned barycentric re-solving retains a target-local component of constant mass and, with polynomial error spending of exponent greater than one, achieves a linear information clock and $O(\log T)$ regret; slack-capacity local pricing
    
[^139]: 迈向自动化发现：逆向材料设计中生成模型、多模态学习与闭环工作流的综述

    Towards Automated Discovery: A Review of Generative Models, Multimodal Learning and Closed-Loop Workflows in Inverse Materials Design

    [https://arxiv.org/abs/2606.02507](https://arxiv.org/abs/2606.02507)

    本综述系统梳理了生成模型、多模态学习和闭环工作流在逆向材料设计中的最新进展，强调物理先验和约束如何嵌入模型以实现可控的晶体结构生成。

    

    逆向材料设计正将材料发现从正向预测转向在物理约束下针对性地提出满足目标的候选材料。在此，我们综述了晶体结构生成建模、多模态学习以及结晶固体的闭环设计流程方面的进展。我们调查了生成器如何从数据库中学习化学-结构先验，以实现周期性结构的可控采样，并比较了变分自编码器、归一化流、自回归模型和扩散模型。在这些模型家族中，我们考察了可行性约束和物理先验在何处介入，从表示和训练目标到采样时指导、筛选和弛豫。我们还讨论了结合晶体结构、热力学和电子信息、显微学、光谱学、处理上下文和科学文本的多模态学习，以构建材料表示。

    arXiv:2606.02507v2 Announce Type: replace-cross  Abstract: Inverse materials design is shifting materials discovery from forward prediction toward targeted proposal of candidates that satisfy objectives under physical constraints. Here, we review advances in generative crystal structure modeling, multimodal learning, and closed-loop design pipelines for crystalline solids. We survey how generators learn chemical-structural priors from databases to enable controllable sampling of periodic structures, comparing variational autoencoders, normalizing flows, autoregressive models, and diffusion models. Across these families, we examine where feasibility constraints and physical priors enter, from representations and training objectives to sampling-time guidance, screening, and relaxation. We also discuss multimodal learning combining crystal structures, thermodynamic and electronic information, microscopy, spectroscopy, processing context, and scientific text to construct materials represen
    
[^140]: 自修正的科学发现系统：面向智能体人工智能的范畴论框架

    Self-Revising Discovery Systems for Science: A Categorical Framework for Agentic Artificial Intelligence

    [https://arxiv.org/abs/2606.01444](https://arxiv.org/abs/2606.01444)

    本文提出一个基于范畴论的框架，通过左Kan扩展和体制转换来区分检索、搜索与科学发现，实现不依赖主观新颖性的自修正智能体系统。

    

    科学发现不仅仅是生成答案，更是对表征体制的修正，其中证据、人工产物、操作和验证器都被类型化。我们为材料科学中的智能体发现开发了一个范畴论描述。在具有模式范畴S_b的固定体制b中，系统状态是一个余预层I_t: S_b -> Set，来源是元素范畴\int_{S_b} I_t。固定体制操作是对此类状态的更新，只有在指定并保留来源保持的细化时才是自函子性的。相反，发现是一种验证过的体制转换u: S_b -> S_b'：旧的人工产物被保留，通过左Kan扩展Lan_u I_t传输，并与转换后的状态进行比较，以识别超出函子传输的残余内容。这在不依赖主观新颖性的情况下区分了检索、搜索和发现。我们在两个系统中实例化了该框架。在Builder/Breaker中，一个蛋白质力学系统...

    arXiv:2606.01444v2 Announce Type: replace  Abstract: Scientific discovery is not only answer generation but revision of the representational regime in which evidence, artifacts, operations, and verifiers are typed. We develop a category-theoretic account of agentic discovery for materials science. In a fixed regime b with schema category S_b, the system state is a copresheaf I_t: S_b -> Set, and provenance is the category of elements \int_{S_b} I_t. Fixed-regime operation is an update on such states, endofunctorial only when provenance-preserving refinements are specified and preserved. Discovery is instead a verified regime transition u: S_b -> S_b': old artifacts are preserved, transported by the left Kan extension Lan_u I_t, and compared with the post-transition state to identify residual content beyond functorial transport. This separates retrieval, search, and discovery without subjective novelty. We instantiate the framework in two systems. In Builder/Breaker, a protein-mechanics
    
[^141]: 关于不完整U统计量中位数的有限样本集中性

    On Finite-sample Concentration of Median of Incomplete U-Statistics

    [https://arxiv.org/abs/2606.00661](https://arxiv.org/abs/2606.00661)

    本文证明了不完整U统计量中位数（MoIU）的有限样本浓度界，克服了此前仅能获得松散$O(n^{-1/4})$界的理论挑战，实现了更紧的收敛速率。

    

    中位数均值（MoM）是一种强大的技术，在理论上能在底层数据分布具有重尾特性（例如，仅假设具有前两阶有限矩）时，实现参数估计的接近亚高斯的有限样本速率。最近的研究将此技术推广到中位数随机化U统计量（MoRU）和中位数不完整U统计量（MoIU），用于估计重尾成对核的期望。在\citet{pmlr-v97-clemencon19a}中，已证明MoRU的浓度速率随样本量按$O(n^{-1/2})$缩放。然而，尽管后者具有计算优势，MoIU的有限样本界分析仍是一个重大的理论挑战。正如作者所指出的，直接应用McDiarmid不等式会产生$O(n^{-1/4})$阶的松散界。在本工作中，我们证明了MoIU估计的有限样本浓度界。

    arXiv:2606.00661v2 Announce Type: replace-cross  Abstract: Median-of-means (MoM) is a powerful technique that theoretically enables near sub-Gaussian finite-sample rate for parameter estimation when the underlying data distribution is heavy-tailed (e.g., assumed to have only two first finite moments). A recent work has extrapolated this technique to median-of-\textit{randomized}-U-Statistics (MoRU) and median-of-\textit{incomplete}-U-Statistics (MoIU) for estimating expectations of heavy-tailed pairwise kernels. In \citet{pmlr-v97-clemencon19a}, a concentration rate that scales like $O(n^{-1/2})$ with sample size has been proven for MoRU. However, despite the computational advantage of the latter, the analysis of finite-sample bound for MoIU remains a significant theoretical challenge. As noted by the authors, a straightforward application of McDiarmid's inequality yields a loose bound of order $O(n^{-1/4})$. In this work, we prove a finite-sample concentration bound for the MoIU estim
    
[^142]: 面向差分隐私的快速混合机制

    The Fast Mixing Mechanism for Differential Privacy

    [https://arxiv.org/abs/2605.30600](https://arxiv.org/abs/2605.30600)

    本文提出一种基于快速变换的新型差分隐私草图机制，在保持强隐私保证的同时，实现了与经典快速方法相当的运行效率。

    

    arXiv:2605.30600v2 公告类型：替换 摘要：随机草图是一种在保持精度的同时压缩大规模优化问题的核心工具。特别是，基于结构化矩阵（如哈达玛矩阵）的草图可以高效应用，并且通常能以更低的计算成本生成接近原始问题的解。在差分隐私（DP）中，高斯草图已被用于解决DP线性回归问题，始于\citet{sheffet2017differentially, sheffet2019old}，后来由\citet{lev2025gaussianmix, lev2026near}改进。然而，尽管这些方法实现了强大的效用保证，它们通常不会比经典DP方法在运行时间上有所提升。在本工作中，我们引入了一种基于快速变换的新型DP草图机制，在某些情况下，其运行时间与经典快速草图方法相匹配。我们为该机制证明了最先进的隐私保证，并表明在有利条件下，其性能优于现有方法。

    arXiv:2605.30600v2 Announce Type: replace  Abstract: Randomized sketching is a central tool for compressing large-scale optimization problems while preserving accuracy. In particular, sketches that are based on structured matrices, such as the Hadamard matrix, can be applied efficiently and often yield solutions that approximate those of the original problem at much lower computational cost. In differential privacy (DP), Gaussian sketching has been used to solve DP linear regression, beginning with \citet{sheffet2017differentially, sheffet2019old} and later refined by \citet{lev2025gaussianmix, lev2026near}. However, although these methods achieve strong utility guarantees, they usually do not improve runtime over classical DP approaches. In this work, we introduce a new DP sketching mechanism based on fast transforms, which, in certain cases, matches the runtime of classical fast sketching methods. We prove state-of-the-art privacy guarantees for this mechanism and show that, in favor
    
[^143]: RouteScan：一种通过专家路由遥测审计MoE大语言模型安全性的非侵入式方法

    RouteScan: A Non-Intrusive Approach to Auditing MoE LLMs Safety via Expert Routing Telemetry

    [https://arxiv.org/abs/2605.24817](https://arxiv.org/abs/2605.24817)

    本文提出RouteScan，一种通过分析GPU执行中专家路由遥测来非侵入式审计MoE大语言模型安全性的方法，无需访问用户提示或模型内部，从而兼顾安全与隐私。

    

    摘要：随着混合专家（MoE）架构越来越多地被用于扩展大型语言模型（LLM），安全审计变得必要，以验证这些模型在运行过程中是否产生或促进有害行为。然而，现有的基于内容的审计方法通常需要访问用户提示、模型内部或输出，这可能暴露敏感用户信息，并在LLM安全性和用户隐私之间造成紧张关系。另一方面，我们观察到，在MoE模型中，不同输入会引发不同的稀疏专家路由模式，这些模式在低级GPU执行遥测中产生可测量的足迹。我们将这些由专家路由决策引起的硬件可观察信号称为专家路由遥测；它们源自GPU执行，而非路由器logits或令牌级路由分配。受此观察启发，我们提出了RouteScan，一种非侵入式审计方法，它利用专家路由遥测来检测MoE模型中的不安全行为，而无需访问用户提示、模型权重或输出，从而在保持安全性的同时保护用户隐私。

    arXiv:2605.24817v2 Announce Type: replace-cross  Abstract: As Mixture-of-Experts (MoE) architectures are increasingly adopted for scaling Large Language Models (LLMs), safety auditing becomes necessary to verify whether these models produce or facilitate harmful behaviors during operation. However, existing content-based auditing methods typically require access to user prompts, model internals, or outputs, potentially exposing sensitive user information and creating a tension between LLM safety and user privacy. On the other hand, we observe that, in MoE models, different inputs induce different sparse expert-routing patterns, which produce measurable footprints in low-level GPU execution telemetry. We refer to these hardware-observable signals induced by expert-routing decisions as expert routing telemetry; they are derived from GPU execution rather than from router logits or token-level routing assignments. Inspired by this observation, we propose RouteScan, a non-intrusive auditing
    
[^144]: 行为一致的深度强化学习

    Behavior-Consistent Deep Reinforcement Learning

    [https://arxiv.org/abs/2605.21214](https://arxiv.org/abs/2605.21214)

    本文提出行为一致强化学习框架，通过控制最大熵温度与Q值分歧来减少跨运行策略差异，实现高性能与分布相似性的平衡。

    

    强化学习（RL）在训练过程中常常表现出高方差，导致性能不可靠，并对实际应用部署构成重大挑战。在这项工作中，我们通过形式化行为一致RL问题来解决跨运行策略发散问题，其目标是获得既高性能又在训练运行间分布相似的策略。我们的关键观察是，最大熵RL通过将运行锚定到共同的（均匀）先验，提供了一种控制行为差异的直接机制。我们证明，对于Boltzmann策略，选择与$Q$-函数分歧成比例的温度可以限制诱导策略之间的成对KL散度。然而，我们也表明，天真地增加熵可能会损害策略优化，同时放大离策略误差。基于这些观察，我们提出了$Q$-值期望（Q-value Expect）方法。

    arXiv:2605.21214v3 Announce Type: replace-cross  Abstract: Reinforcement learning (RL) often exhibits high variance across training runs, leading to unreliable performance and posing a major challenge to deployment in real-world domains. In this work, we address the challenge of cross-run policy divergence by formalizing the problem of behavior-consistent RL, where the objective is to obtain policies that are both high-performing and distributionally similar across training runs. Our key observation is that maximum-entropy RL provides a direct mechanism for controlling behavioral divergence by anchoring runs to a common (uniform) prior. We prove that, for Boltzmann policies, choosing the temperature proportional to $Q$-function disagreement bounds the pairwise KL divergence between the induced policies. However, we also show that na\"ively increasing entropy might impair policy optimization while amplifying off-policy error. Building upon these observations, we propose $Q$-value Expect
    
[^145]: STS：基于投机性令牌稀疏性的高效稀疏注意力机制

    STS: Efficient Sparse Attention with Speculative Token Sparsity

    [https://arxiv.org/abs/2605.15508](https://arxiv.org/abs/2605.15508)

    STS提出了一种利用小型草稿模型预测重要令牌来动态构建稀疏掩码的方法，无需重训练即可在保持精度的同时显著加速LLM推理。

    

    注意力机制的二次复杂度给大型语言模型（LLM）推理带来了严重的内存和计算瓶颈。这一挑战对于需要处理数百万令牌序列的新兴代理应用尤为严峻。我们提出了STS，一种无需模型重新训练的稀疏注意力机制。STS利用了这样一个关键洞察：由较小草稿模型识别为重要的令牌，对于较大目标模型的重要令牌具有高度预测性。通过整合到投机性解码框架中，STS重新利用草稿模型的注意力分数来动态构建令牌和头部特定的稀疏掩码。该掩码有效地剪枝了目标LLM中昂贵的注意力计算。我们的评估表明，在代表性基准NarrativeQA上，STS在约90%稀疏度下实现了2.67倍的加速，与完全注意力相比，准确率下降可忽略不计。

    arXiv:2605.15508v3 Announce Type: replace-cross  Abstract: The quadratic complexity of attention imposes severe memory and computational bottlenecks on Large Language Model (LLM) inference. This challenge is particularly acute for emerging agentic applications that require processing multi-million token sequences. We propose STS, a sparse attention mechanism that requires no model retraining. STS leverages the key insight that tokens identified as important by a smaller draft model are highly predictive of important tokens for a larger target model. By integrating into speculative decoding frameworks, STS repurposes the draft model's attention scores to dynamically construct a token-and-head-wise sparsity mask. This mask effectively prunes the expensive attention computation in the target LLM. Our evaluation shows that STS achieves a 2.67x speedup operating at approximately 90% sparsity on representative benchmark NarrativeQA, maintaining negligible accuracy degradation compared to den
    
[^146]: 学习高能物理模拟中多维失配的最小偏差修正

    Learning Minimal-Deviation Corrections for Multi-Dimensional Mismodelling in HEP Simulations

    [https://arxiv.org/abs/2605.07460](https://arxiv.org/abs/2605.07460)

    提出一种神经网络方法，在仅有一维目标分布和多维失配的约束下，通过最小偏差修正学习模拟事件的变换，既保持原始模拟的相关结构，又能针对性修正失配特征。

    

    高能物理中精确的蒙特卡洛（MC）建模具有挑战性，尤其是在模拟无法重现观测数据的复杂场景中。实践中，实验信息通常仅限于一维（1D）分布，而失配发生在多维特征空间中。这限制了传统修正方法，因为一维重加权忽略了相关性，而全多维方法需要大量目标数据集。我们提出一种基于神经网络的方法，在这些约束下运行，通过学习模拟事件的变换来重现可用的一维目标分布，同时保持接近原始模拟。这种最小偏差原则保留了基线模型的全局相关结构，同时能够对失配特征进行针对性修正。通过使用模拟伪数据的受控研究，我们展示了该方法在修正多维失配方面的有效性。

    arXiv:2605.07460v2 Announce Type: replace  Abstract: Accurate Monte Carlo (MC) modelling in high-energy physics is challenging, particularly in complex scenarios where simulations fail to reproduce observed data. In practice, experimental information is often limited to one-dimensional (1D) distributions, while mismodelling arises in a multidimensional feature space. This restricts traditional correction methods, as one-dimensional reweighting ignores correlations and fully multidimensional approaches require large target datasets. We propose a neural network-based method that operates under these constraints by learning a transformation of simulated events that reproduces the available 1D target distributions while remaining close to the original simulation. This minimal-deviation principle preserves the global correlation structure of the baseline model while enabling targeted corrections of mismodelled features. Using controlled studies with simulated pseudo-data, we show that the m
    
[^147]: GRALIS：融合联盟与梯度归因，具备闭式守恒误差和有限样本保证

    GRALIS: Fusing Coalition and Gradient Attribution with Closed-Form Conservation Error and Finite-Sample Guarantees

    [https://arxiv.org/abs/2605.05480](https://arxiv.org/abs/2605.05480)

    GRALIS通过融合Shapley联盟权重和连续梯度路径，提出了一种统一的XAI估计器，并提供了闭式守恒误差和有限样本保证，解决了现有方法在保真度指标上互补但不可直接比较的问题。

    

    arXiv:2605.05480v3 公告类型：替换-交叉 摘要：用于深度网络的主要事后XAI方法——GradCAM、SHAP、LIME、集成梯度——源自异构的理论基础，在单一表示中无法自然比较。最近一项基准测试也发现，基于联盟的方法（GradCAM、KernelSHAP、LIME）和基于梯度的方法（集成梯度及其变体）在经验上互补，各自在不同保真度指标上表现更优，而方法选择是唯一提出的解决方案（Gevaert等，2022）。本工作提出了GRALIS（梯度-里兹平均局部集成Shapley），它将这两种机制——一个Shapley联盟权重和局部性核，以及一个连续的集成梯度风格的条件路径——融合到单一估计器中，并为其提供了两种单独机制都无法提供的认证保证：一个精确的闭式完整性缺陷（一个d阶交互被归因于...）

    arXiv:2605.05480v3 Announce Type: replace-cross  Abstract: The main post-hoc XAI methods for deep networks -- GradCAM, SHAP, LIME, Integrated Gradients -- originate from heterogeneous theoretical foundations and are not naturally comparable within a single representation. A recent benchmark also finds their coalition-based members (GradCAM, KernelSHAP, LIME) and gradient-based members (Integrated Gradients and variants) empirically complementary, each outperforming the other on different faithfulness metrics, with method selection as the only proposed remedy (Gevaert et al., 2022).   This work presents GRALIS (Gradient-Riesz Averaged Locally-Integrated Shapley), which fuses these two mechanisms -- a Shapley coalition weight and locality kernel, and a continuous Integrated-Gradients-style conditioned path -- into a single estimator, and equips it with two certified guarantees neither mechanism supplies alone: an exact, closed-form completeness deficit (an order-d interaction is attribut
    
[^148]: 基于偏好的自蒸馏：通过奖励正则化超越KL匹配

    Preference-Based Self-Distillation: Beyond KL Matching via Reward Regularization

    [https://arxiv.org/abs/2605.05040](https://arxiv.org/abs/2605.05040)

    本文提出基于偏好的自蒸馏（PBSD），通过奖励正则化替代传统KL匹配，解决了自蒸馏中的训练不稳定和探索多样性不足问题。

    

    arXiv:2605.05040v2 公告类型：替换-交叉 摘要：同策略蒸馏是强化学习的一种高效替代方案，提供密集的令牌级训练信号。然而，它对更强外部教师的依赖推动了近期关于同策略自蒸馏的研究，其中同一模型在不同提示上下文中同时充当教师和学生。然而，现有的自蒸馏方法大多将学习简化为对上下文增强教师模型的KL匹配。这种方法常常遭受训练不稳定性，并可能随时间推移降低推理性能。此外，从同一模型进行提示增强的自蒸馏缺乏真实外部教师所提供的探索多样性。为解决这些局限性，我们超越了固定教师的KL匹配，提出了基于偏好的自蒸馏（PBSD），该算法从奖励正则化的角度重新审视同策略自蒸馏。

    arXiv:2605.05040v2 Announce Type: replace-cross  Abstract: On-policy distillation is an efficient alternative to reinforcement learning, offering dense token-level training signals. However, its reliance on a stronger external teacher has driven recent work on on-policy self-distillation, where the same model serves as both teacher and student under different prompt contexts. Yet, existing self-distillation methods largely reduce learning to KL matching toward the context-augmented teacher model. This approach often suffers from training instability and can degrade reasoning performance over time. Moreover, self-distillation from the same model with prompt augmentation lacks the exploratory diversity provided by a genuine external teacher. To address these limitations, we move beyond fixed-teacher KL matching and propose \textbf{P}reference-\textbf{B}ased \textbf{S}elf-\textbf{D}istillation (\textbf{PBSD}), which revisits on-policy self-distillation through a reward-regularized perspec
    
[^149]: RefusalGuard：保持几何结构的微调方法以保障大语言模型安全性

    RefusalGuard: Geometry-Preserving Fine-Tuning for Safety in LLMs

    [https://arxiv.org/abs/2605.01913](https://arxiv.org/abs/2605.01913)

    本文揭示了标准微调导致安全对齐退化的表示级机制，并提出了REFUSALGUARD框架，通过保持安全相关表示的几何结构来在微调中维持模型安全性。

    

    arXiv:2605.01913v2 公告类型：替换交叉 摘要：对已进行安全对齐的语言模型进行下游任务微调，往往会导致拒绝行为大幅退化，使模型易受对抗性滥用攻击。尽管先前研究已表明，安全相关特征在模型激活空间中以结构化表示形式编码，但这些表示在微调过程中如何变化以及对齐为何退化，仍知之甚少。在本工作中，我们研究了对齐退化背后的表示级机制。我们的分析表明，标准微调会引发安全相关表示的系统性漂移，扭曲其几何结构，并在任务优化与安全特征之间引入干扰。这些效应共同导致有害顺从性增加。基于这些发现，我们提出了REFUSALGUARD，一种表示级微调框架，在微调过程中保持安全相关结构。

    arXiv:2605.01913v2 Announce Type: replace-cross  Abstract: Fine-tuning safety-aligned language models for downstream tasks often leads to substantial degradation of refusal behavior, making models vulnerable to adversarial misuse. While prior work has shown that safety-relevant features are encoded in structured representations within the model's activation space, how these representations change during fine-tuning and why alignment degrades remains poorly understood. In this work, we investigate the representation-level mechanisms underlying alignment degradation. Our analysis shows that standard fine-tuning induces systematic drift in safety-relevant representations, distorts their geometric structure, and introduces interference between task optimization and safety features. These effects collectively lead to increased harmful compliance. Motivated by these findings, we introduce REFUSALGUARD, a representation-level fine-tuning framework that preserves safety-relevant structure duri
    
[^150]: 与什么相比？反事实提示的基线和度量标准

    Compared to What? Baselines and Metrics for Counterfactual Prompting

    [https://arxiv.org/abs/2605.01048](https://arxiv.org/abs/2605.01048)

    本文指出，反事实提示中观察到的效应常被表面形式变化混淆，需使用基线（如改写）来校正，否则可能错误归因模型敏感性。

    

    arXiv:2605.01048v2 公告类型：替换 摘要：反事实提示（即扰动单一因素并测量输出变化）被广泛用于评估诸如大语言模型偏差和思维链忠实度等事项。但在本工作中，我们认为，如果不考虑建立通用模型敏感性的基线“意义保持”文本修改，观察到的效应就不能归因于目标因素。这是因为每个反事实编辑都是一个复合处理，将感兴趣变量与偶然的表面形式变化捆绑在一起；这违反了处理变化无关性。我们在MedQA上观察到，当手术性地改变患者性别时，预测翻转率为14.9%。然而，这与简单改写输入所引发的翻转率（14.1%）在统计上无法区分。在这种情况下，因此得出大语言模型对患者性别特别敏感的结论是不合理的。为考虑这一点并稳健地测量目标效应。

    arXiv:2605.01048v2 Announce Type: replace  Abstract: Counterfactual prompting (i.e., perturbing a single factor and measuring output change) is widely used to evaluate things like LLM bias and CoT faithfulness. But in this work we argue that observed effects cannot be attributed to the targeted factor without accounting for baseline "meaning-preserving" modifications to text that establish general model sensitivity. This is because every counterfactual edit is a compound treatment that bundles the variable of interest with incidental surface-form variation; this violates treatment variation irrelevance. We observe prediction flip rates on MedQA of 14.9% when we surgically change patient gender. However, this is statistically indistinguishable from the flip rates induced by simply paraphrasing inputs (14.1%). In this case, it would therefore be unwarranted to conclude that the LLM is especially sensitive to patient gender. To account for this and robustly measure the effects of targeted
    
[^151]: AutoOR：规模化后训练大语言模型以自动形式化运筹学问题

    AutoOR: Scalably Post-training LLMs to Autoformalize Operations Research Problems

    [https://arxiv.org/abs/2604.16804](https://arxiv.org/abs/2604.16804)

    AutoOR提出了一种结合合成数据生成和强化学习的可扩展流水线，通过求解器反馈作为奖励信号，使8B参数模型在多个运筹学基准上达到或超越更大前沿模型的性能，尤其在非线性物理动力学问题上实现了突破。

    

    arXiv:2604.16804v3 公告类型：替换-交叉 摘要：优化问题是制造业、物流、调度及其他工业场景中决策的核心。将这些问题的复杂描述转化为求解器可用的公式需要专业的运筹学（OR）专业知识，这使得规模化变得困难。我们提出了AutoOR，一种可扩展的合成数据生成和强化学习流水线，用于训练大语言模型（LLMs），使其能够将自然语言中指定的优化问题自动形式化，涵盖线性、混合整数和非线性类别。AutoOR从标准优化形式生成经过验证的训练数据，并使用求解器执行反馈作为强化学习后训练的奖励信号。将AutoOR应用于一个8B模型，在六个既定运筹学基准上取得了最先进或具有竞争力的结果，匹配了显著更大的前沿模型。对于一个涉及物理动力学的非线性问题类别，前沿模型得分接近0%，而AutoOR模型实现了显著改进。

    arXiv:2604.16804v3 Announce Type: replace-cross  Abstract: Optimization problems are central to decision-making in manufacturing, logistics, scheduling, and other industrial settings. Translating complicated descriptions of these problems into solver-ready formulations requires specialized operations research (OR) expertise, making it hard to scale. We present AutoOR, a scalable synthetic data generation and reinforcement learning pipeline that trains LLMs to autoformalize optimization problems specified in natural language across linear, mixed-integer, and non-linear categories. AutoOR generates verified training data from standard optimization forms and uses solver execution feedback as the reward signal for RL post-training. AutoOR applied to an 8B model achieves state-of-the-art or competitive results across six established OR benchmarks, matching significantly larger frontier models. For a non-linear problem class involving physical dynamics, where frontier models score near 0%, w
    
[^152]: 校准-再委托：通过模型级联实现具有风险与预算保障的安全监控

    Calibrate-Then-Delegate: Safety Monitoring with Risk and Budget Guarantees via Model Cascades

    [https://arxiv.org/abs/2604.14251](https://arxiv.org/abs/2604.14251)

    本文提出一种名为校准-再委托（CTD）的模型级联方法，通过预测专家调用价值的轻量级探针和统计校准，在保障安全或成本的同时实现流式决策。

    

    在大规模监控LLM安全性时，需要在成本和准确性之间取得平衡：廉价的潜在空间探针可以筛选每个输入，但困难案例应升级到更昂贵的专家。现有级联方法基于探针不确定性进行委托，但不确定性是专家调用效用的一个不良代理，因为它忽略了专家是否真的会改善预测。为解决此问题，我们引入了校准-再委托（CTD），一种模型级联方法，它在提供实例级（流式）决策的同时，对计算成本或安全性能提供概率保证。CTD基于一种新颖的委托价值（DV）探针，这是一种轻量级模型，在与安全探针相同的内部表示上运行，用于预测在专家调用后用专家分数替换探针分数的益处。CTD使用保留数据和多重假设检验对DV信号上的阈值进行校准，从而提供保障。

    arXiv:2604.14251v2 Announce Type: replace  Abstract: Monitoring LLM safety at scale requires balancing cost and accuracy: a cheap latent-space probe can screen every input, but hard cases should be escalated to a more expensive expert. Existing cascades delegate based on probe uncertainty, but uncertainty is a poor proxy for the utility of an expert call, as it ignores whether the expert would actually improve the prediction. To address this problem, we introduce Calibrate-Then-Delegate (CTD), a model-cascade approach that provides probabilistic guarantees on either computation cost or safety performance while enabling instance-level (streaming) decisions. CTD builds on a novel delegation value (DV) probe, a lightweight model operating on the same internal representations as the safety probe that predicts the benefit of replacing the probe score with the expert score after an expert call. CTD calibrates a threshold on the DV signal using held-out data and multiple hypothesis testing, y
    
[^153]: Virgo探测器中的毛刺自动分类管道

    Automatic classification pipeline for glitches in the Virgo detector

    [https://arxiv.org/abs/2604.13687](https://arxiv.org/abs/2604.13687)

    本文提出了VIGILant自动管道，利用ResNet34模型在Virgo探测器毛刺分类中达到高精度，并已部署于日常观测。

    

    毛刺经常污染引力波探测器的数据，使天体物理信号的观测和分析复杂化。这项工作介绍了VIGILant，一个用于Virgo探测器中毛刺分类和可视化的自动管道。利用精心整理的Virgo O3b毛刺数据集，评估了两种机器学习方法：基于树模型（决策树、随机森林和XGBoost）使用结构化的Omicron参数，以及基于谱图图像的卷积神经网络（ResNet）。虽然基于树的模型提供了更高的可解释性和快速训练，但ResNet34模型在测试集中达到了优越的性能，F1分数为0.9772，准确率为0.9833，每个毛刺的推理时间为几十毫秒。该管道自观测运行O4c以来已在Virgo站点部署用于日常操作，为Virgo合作组织提供了一个交互式仪表板。

    arXiv:2604.13687v2 Announce Type: replace-cross  Abstract: Glitches frequently contaminate data in gravitational-wave detectors, complicating the observation and analysis of astrophysical signals. This work introduces VIGILant, an automatic pipeline for classification and visualization of glitches in the Virgo detector. Using a curated dataset of Virgo O3b glitches, two machine learning approaches are evaluated: tree-based models (Decision Tree, Random Forest and XGBoost) using structured Omicron parameters, and Convolutional Neural Networks (ResNet) trained on spectrogram images. While tree-based models offer higher interpretability and fast training, the ResNet34 model achieved superior performance, reaching a F1 score of 0.9772 and accuracy of 0.9833 in the testing set, with inference times of tens of milliseconds per glitch. The pipeline has been deployed for daily operation at the Virgo site since observing run O4c, providing the Virgo collaboration with an interactive dashboard t
    
[^154]: 基于内在奖励的乐观在线LQR算法

    Optimistic Online LQR via Intrinsic Rewards

    [https://arxiv.org/abs/2603.28938](https://arxiv.org/abs/2603.28938)

    提出了一种通过仅修改成本函数来引入内在奖励的乐观在线LQR算法，既保持了标准LQR结构，又实现了高效的不确定性驱动探索。

    

    arXiv:2603.28938v2 公告类型：替换-交叉 摘要：面对不确定性时的乐观态度是强化学习中平衡探索与利用的一种流行方法。在此，我们考虑在线线性二次调节器（LQR）问题，即通过根据运行期间收集的闭环数据在线调整控制策略，来学习对应于未知线性动态系统的LQR。在本工作中，我们提出了内在奖励LQR（IR-LQR），一种乐观的在线LQR算法，它应用了源自强化学习的内在奖励思想以及方差正则化概念，以促进不确定性驱动的探索。IR-LQR通过仅修改成本函数来保留标准LQR综合问题的结构，从而产生一种直观、简单、计算成本低且高效的算法。这与依赖更复杂迭代搜索的现有乐观在线LQR公式形成对比。

    arXiv:2603.28938v2 Announce Type: replace-cross  Abstract: Optimism in the face of uncertainty is a popular approach to balance exploration and exploitation in reinforcement learning. Here, we consider the online linear quadratic regulator (LQR) problem, i.e., to learn the LQR corresponding to an unknown linear dynamical system by adapting the control policy online based on closed-loop data collected during operation. In this work, we propose Intrinsic Rewards LQR (IR-LQR), an optimistic online LQR algorithm that applies the idea of intrinsic rewards originating from reinforcement learning and the concept of variance regularization to promote uncertainty-driven exploration. IR-LQR typically retains the structure of a standard LQR synthesis problem by only modifying the cost function, resulting in an intuitively pleasing, simple, computationally cheap, and efficient algorithm. This is in contrast to existing optimistic online LQR formulations that rely on more complicated iterative sear
    
[^155]: 一种通过虚拟智能体实现鱼群闭环引导的深度强化学习框架

    A Deep Reinforcement Learning Framework for Closed-loop Guidance of Fish Schools via Virtual Agents

    [https://arxiv.org/abs/2603.28200](https://arxiv.org/abs/2603.28200)

    该论文提出了一种基于深度强化学习的虚拟智能体闭环引导鱼群框架，通过PPO训练和复合奖励函数平衡方向引导与凝聚力，在物理实验中实现了对活体鱼群的有效实时控制。

    

    arXiv:2603.28200v2 公告类型：替换交叉 摘要：引导生物群体中的集体运动是理解社会互动规则的基本挑战。在本研究中，我们提出了一种使用虚拟智能体进行鱼群闭环引导的深度强化学习（RL）框架。这些智能体通过模拟环境中使用近端策略优化（PPO）训练的策略进行控制，并在物理实验中部署于红鼻剪刀鱼（Petitella bleheri）上，实现了人工智能体与活体个体之间的实时交互。为了应对活体个体的随机行为，我们设计了一个复合奖励函数，该函数平衡了方向引导与群体凝聚力，在控制目标层面提供了一种功能性仿生形式。我们对视觉参数的系统评估表明，在物理试验中，白色背景和较大的刺激尺寸在测试条件下产生了最高的引导效果。此外，我们进一步...

    arXiv:2603.28200v2 Announce Type: replace-cross  Abstract: Guiding collective motion in biological groups is a fundamental challenge in understanding social interaction rules. In this study, we propose a deep reinforcement learning (RL) framework for closed-loop guidance of fish schools using virtual agents. These agents are controlled by policies trained via Proximal Policy Optimization (PPO) in simulation and deployed in physical experiments with rummy-nose tetras (Petitella bleheri), enabling real-time interaction between artificial agents and live individuals. To cope with the stochastic behavior of live individuals, we designed a composite reward function that balances directional guidance with cohesion, providing a form of functional biomimicry at the level of the control objective. Our systematic evaluation of visual parameters showed that a white background and larger stimulus sizes produced the highest guidance efficacy among the tested conditions in physical trials. Furthermo
    
[^156]: 大规模高效探索

    Efficient Exploration at Scale

    [https://arxiv.org/abs/2603.17378](https://arxiv.org/abs/2603.17378)

    我们提出了一种在线RLHF算法，通过增量更新、微小肯定信号、认知不确定性和信息导向探索，实现了超过10倍的数据效率提升，仅用不到20K标签即可匹配离线RLHF在200K标签上的性能。

    

    arXiv:2603.17378v2 公告类型：替换交叉 摘要：我们开发了一种在线学习算法，显著提高了从人类反馈中进行强化学习（RLHF）的数据效率。我们的算法在接收选择数据时增量更新奖励模型和语言模型。奖励模型拟合选择数据，而语言模型通过一种强化变体进行更新，强化信号由奖励模型提供。几个特性促成了效率提升：每个强化信号中添加的微小肯定性推动、建模奖励不确定性的认知神经网络，以及信息导向的探索。使用Gemma大型语言模型（LLMs），我们的算法在不到20K标签的情况下，匹配了在200K标签上训练的离线RLHF的性能，实现了超过10倍的数据效率提升。根据我们的结果外推，我们预计在1M标签上训练的算法能匹配在1B标签上训练的离线RLHF。这代表

    arXiv:2603.17378v2 Announce Type: replace-cross  Abstract: We develop an online learning algorithm that dramatically improves the data efficiency of reinforcement learning from human feedback (RLHF). Our algorithm incrementally updates reward and language models as choice data is received. The reward model is fit to the choice data, while the language model is updated by a variation of reinforce, with reinforcement signals provided by the reward model. Several features enable the efficiency gains: a small affirmative nudge added to each reinforcement signal, an epistemic neural network that models reward uncertainty, and information-directed exploration. With Gemma large language models (LLMs), our algorithm matches the performance of offline RLHF trained on 200K labels using fewer than 20K labels, representing more than a 10x gain in data efficiency. Extrapolating from our results, we expect our algorithm trained on 1M labels to match offline RLHF trained on 1B labels. This represents
    
[^157]: 探究目标类别对神经网络可压缩性的影响——面向能量自主的鸟类监测

    Investigating Target Class Influence on Neural Network Compressibility for Energy-Autonomous Avian Monitoring

    [https://arxiv.org/abs/2602.17751](https://arxiv.org/abs/2602.17751)

    本文提出在野外低成本微控制器上运行高效AI模型，用于鸟类被动声学监测，以克服传统方法成本高和现有机器学习资源消耗大的问题。

    

    摘要：生物多样性丧失对人类构成重大威胁，因此野生动物监测对于评估生态系统健康至关重要。鸟类因其受欢迎程度以及通过其独特鸣声易于识别的特性，成为理想的监测对象。传统的鸟类监测方法需要人工计数，成本高昂且效率低下。在被动声学监测中，会长时间记录环境声音，随后分析这些录音以识别鸟类物种。机器学习方法已在多种物种和环境中极大地加速了这一过程，然而，现有解决方案需要复杂的模型和大量的计算资源。相反，我们提出直接在野外使用廉价的微控制器单元（MCU）运行机器学习模型。由于硬件和能源限制，这要求采用高效的人工智能（AI）架构。

    arXiv:2602.17751v2 Announce Type: replace-cross  Abstract: Biodiversity loss poses a significant threat to humanity, making wildlife monitoring essential for assessing ecosystem health. Avian species are ideal subjects for this due to their popularity and the ease of identifying them through their distinctive songs. Traditionalavian monitoring methods require manual counting and are therefore costly and inefficient. In passive acoustic monitoring, soundscapes are recorded over long periods of time. The recordings are analyzed to identify bird species afterwards. Machine learning methods have greatly expedited this process in a wide range of species and environments, however, existing solutions require complex models and substantial computational resources. Instead, we propose running machine learning models on inexpensive microcontroller units (MCUs) directly in the field. Due to the resulting hardware and energy constraints, efficient artificial intelligence (AI) architecture is requi
    
[^158]: 通过最优多路决策树的可解释聚类

    Interpretable clustering via optimal multi-way decision trees

    [https://arxiv.org/abs/2602.13586](https://arxiv.org/abs/2602.13586)

    本文提出了一种名为ICOMT的高性能计算框架，通过一维K-means离散化和最优多路决策树，实现了高可解释性的聚类，同时克服了传统方法中贪婪搜索和二元分裂的局限性。

    

    arXiv:2602.13586v2 公告类型：替换 摘要：聚类是一种基本的无监督学习技术，用于揭示数据结构以促进知识发现和决策制定。虽然聚类准确性至关重要，但可解释性显著影响聚类结果的实际价值，特别是在高风险决策情境中。尽管基于决策树的聚类方法通过显式分裂规则提供了高可解释性，但现有方法通常依赖于局部贪婪搜索或需要昂贵的计算成本且仅限于二元分裂，导致树更深、可解释性更差。为克服这些限制，我们建立了一个高性能计算框架，名为通过最优多路树的可解释聚类（ICOMT）。我们做出三项主要贡献。首先，我们提出了一种新的数值特征离散化方法，使用一维K-means聚类来捕捉数据分布。其次，我们公式化...

    arXiv:2602.13586v2 Announce Type: replace  Abstract: Clustering is a fundamental unsupervised learning technique for uncovering data structures to facilitate knowledge discovery and decision-making. While clustering accuracy is crucial, interpretability significantly impacts the practical value of clustering results, particularly in high-risk decision-making contexts. Although decision-tree-based clustering methods offer high interpretability through explicit splitting rules, existing approaches often rely on local greedy search or require expensive computational costs limited to binary splits, resulting in deeper, less interpretable trees. To overcome these limitations, we establish a high-performance computational framework named Interpretable Clustering via Optimal Multi-way Trees (ICOMT). We make three primary contributions. First, we propose a new discretization method for numerical features using one-dimensional K-means clustering to capture data distributions. Second, we formula
    
[^159]: 度质量消息传递用于有向和无向网络中的介数排名

    Degree-Mass Message Passing for Betweenness Ranking in Directed and Undirected Networks

    [https://arxiv.org/abs/2602.09716](https://arxiv.org/abs/2602.09716)

    提出了一种轻量级GNN架构，利用度质量作为大小不变特征，在合成图上训练，高效预测有向和无向网络中节点的介数中心性排名。

    

    计算网络中节点的重要性是一个长期存在的基础性问题，这推动了各种中心性度量的广泛研究。一个特别知名的中心性度量是介数中心性，其精确计算在大型网络上变得不可行。因此，已经提出了图神经网络（GNN）模型来预测节点按介数中心性的排名。然而，现有的基于GNN的方法要么具有依赖于图大小的参数数量，要么仅限于无向图。我们提出了一种轻量级GNN架构，利用经验观察到的介数中心性与多跳度质量之间的关系。这激发了使用度质量作为大小不变的节点特征。为了提高泛化能力，我们在合成图上进行训练，这些图的度分布更接近真实网络，包括有向和无向的无标度图以及均匀图。

    arXiv:2602.09716v2 Announce Type: replace  Abstract: Computing the importance of nodes in networks is a long-standing fundamental problem that has driven extensive study of various centrality measures. A particularly well-known centrality measure is betweenness centrality, whose exact computation becomes prohibitive on large-scale networks. Graph Neural Network (GNN) models have thus been proposed to predict the ranking of nodes by betweenness centrality. However, existing GNN-based methods either have graph-size-dependent parameter counts or are limited to undirected graphs. We propose a lightweight GNN architecture that exploits the empirically observed relationship between betweenness centrality and multi-hop degree mass. This motivates the use of degree masses as size-invariant node features. To improve generalization, we train on synthetic graphs whose degree distributions more closely match those of real-world networks, including directed and undirected scale-free graphs and unif
    
[^160]: 基于Doob h变换的无限维生成扩散模型

    Infinite-dimensional generative diffusions via Doob's h-transform

    [https://arxiv.org/abs/2602.06621](https://arxiv.org/abs/2602.06621)

    本文通过Doob h变换引入了一种无需时间反转的无限维生成扩散模型新框架，该框架在可验证条件下严格构造，并通过分数匹配目标实现近似，提供了更高的灵活性和泛化能力。

    

    本文通过Doob的h变换，提出了一种在无限维空间中定义生成扩散模型的严谨框架。该方法不依赖于噪声过程的时间反转，而是通过指数测度变换将参考扩散强制导向目标分布。与现有方法相比，这种方法易于推广到无限维设定，从而为扩散模型提供了更大的灵活性。该构造在可验证条件下严格推导，并建立了与目标测度相关的界。我们证明，在变换测度下的强制过程可以通过最小化分数匹配目标来近似，并在合成数据和真实数据上验证了该方法。

    arXiv:2602.06621v2 Announce Type: replace-cross  Abstract: This paper introduces a rigorous framework for defining generative diffusion models in infinite dimensions via Doob's h-transform. Rather than relying on time reversal of a noising process, a reference diffusion is forced towards the target distribution by an exponential change of measure. Compared to existing methodology, this approach readily generalises to the infinite-dimensional setting, hence offering greater flexibility in the diffusion model. The construction is derived rigorously under verifiable conditions, and bounds with respect to the target measure are established. We show that the forced process under the changed measure can be approximated by minimising a score-matching objective and validate our method on both synthetic and real data.
    
[^161]: 最大似然强化学习

    Maximum Likelihood Reinforcement Learning

    [https://arxiv.org/abs/2602.02710](https://arxiv.org/abs/2602.02710)

    本文提出MaxRL，一种通过计算索引的样本目标函数族，将标准强化学习与最大似然统一，仅需一行代码更改即可在多种任务上超越现有方法。

    

    arXiv:2602.02710v2 公告类型：替换 摘要：强化学习（RL）是在目标函数只能通过从模型中采样来评估的设置中训练模型的首选方法。我们的关键观察是，当反馈是终端且二值的时，模型隐式地在正确轨迹上诱导出一个似然。在这种设置下，最大似然将是自然框架，但强化学习被用作解决不可微性的变通方法。我们证明了标准的期望奖励强化学习公式只是似然的一阶近似。为解决这一不匹配问题，我们引入了最大似然强化学习（MaxRL），这是一个以计算为索引的基于样本的目标函数族，随着采样计算的扩展，它在期望奖励强化学习和最大似然之间进行插值。由此产生的目标函数对标准强化学习实现只需一行更改。MaxRL在所有测试的模型和任务中均帕累托优于现有方法，实现了高达...（原文截断）

    arXiv:2602.02710v2 Announce Type: replace  Abstract: Reinforcement learning (RL) is the method of choice for training models in setups where the objective function can only be evaluated by sampling from the model. Our key observation is that when the feedback is terminal and binary, models implicitly induce a likelihood over correct rollouts. Maximum likelihood would be the natural framework in such settings, but RL is used instead as a workaround to the non-differentiability. We prove that the standard, expected-reward RL formulation is only a first-order approximation of the likelihood. To remedy this mismatch, we introduce Maximum Likelihood Reinforcement Learning (MaxRL), a compute-indexed family of sample-based objectives that interpolate between expected-reward RL and maximum likelihood as sampling compute is scaled. The resulting objective is a one-line change to standard RL implementations. MaxRL Pareto-dominates existing methods in all tested models and tasks, achieves up to $
    
[^162]: 深度时间序列模型中的可解释性要求语义对齐

    Interpretability in Deep Time Series Models Demands Semantic Alignment

    [https://arxiv.org/abs/2602.02239](https://arxiv.org/abs/2602.02239)

    本文提出深度时间序列模型的可解释性应追求语义对齐，即预测需用用户有意义的变量表达，并在时间演化下保持一致性，而非仅解释内部计算。

    

    深度时间序列模型持续提升预测性能，但其部署仍受限于黑箱特性。为此，该领域现有的可解释性方法侧重于解释内部模型计算，而未考虑这些计算是否与人类对研究现象的推理方式对齐。相反，我们认为深度时间序列模型中的可解释性应追求语义对齐：预测应通过用户有意义的变量来表达，并辅以能容纳用户依赖约束的空间和时间机制。本文正式化这一要求，并指出一旦建立语义对齐，它必须在时间演化下保持不变——这是一个静态设置中不存在的约束。基于此定义，我们概述了语义对齐深度时间序列模型的蓝图，并识别出相关属性。

    arXiv:2602.02239v3 Announce Type: replace  Abstract: Deep time series models continue to improve predictive performance, yet their deployment remains limited by their black-box nature. In response, existing interpretability approaches in the field keep focusing on explaining the internal model computations, without addressing whether they align or not with how a human would reason about the studied phenomenon. Instead, we state interpretability in deep time series models should pursue semantic alignment: predictions should be expressed in terms of variables that are meaningful to the end user, mediated by spatial and temporal mechanisms that admit user-dependent constraints. In this paper, we formalize this requirement and state that, once established, semantic alignment must be preserved under temporal evolution: a constraint with no analog in static settings. Provided with this definition, we outline a blueprint for semantically aligned deep time series models, identify properties th
    
[^163]: 受控协变量偏移下的泛化度量：一种机制感知基准

    Generalization Measures under Controlled Covariate Shift: A Regime-Aware Benchmark

    [https://arxiv.org/abs/2602.01718](https://arxiv.org/abs/2602.01718)

    本文通过受控协变量偏移基准，揭示了泛化度量的有效性强烈依赖于具体机制，并指出锐度和输入梯度类度量在不同条件下表现不一致。

    

    在目标测试评估之前，从可用量预测泛化能力仍然是深度学习中的核心挑战。Jiang等人（2020）的系统基准评估了许多泛化度量，但其重点在于独立同分布（IID）设置。我们重新审视了在受控损坏和扰动下评估的图像分类器这一问题。我们的研究使用了CIFAR-10-C/P，其中标签空间和任务保持不变，而输入图像被降质或扰动。这一设置也使我们能够重新审视Dziugaite等人（2020）提出的稳健性担忧，他们表明泛化度量的表面可靠性可能强烈依赖于实验条件。我们的实验表明，泛化度量的有用性强烈依赖于机制。在我们跨三种CNN风格架构的探索性决策分析中，基于锐度和输入梯度的度量...

    arXiv:2602.01718v2 Announce Type: replace  Abstract: Predicting generalization from quantities available before target-test evaluation remains a central challenge in deep learning. The systematic benchmark of Jiang et al. (2020) evaluated many generalization measures, but it focused on independent and identically distributed (IID) settings. We revisit this problem for image classifiers evaluated under controlled corruptions and perturbations. Our study uses CIFAR-10-C/P, where the label space and task remain fixed while the input images are degraded or perturbed. This setting also allows us to revisit the robustness concerns raised by Dziugaite et al. (2020), who showed that the apparent reliability of generalization measures can depend strongly on experimental conditions. Our experiments show that the usefulness of generalization measures is strongly regime-dependent. In our exploratory decision analysis across three CNN-style architectures, sharpness- and input-gradient-based measure
    
[^164]: SEISMO：面向样本高效分子优化的可解释性感知、轨迹条件化LLM智能体

    SEISMO: Explanation-Aware, Trajectory-Conditioned LLM Agents for Sample-Efficient Molecular Optimisation

    [https://arxiv.org/abs/2602.00663](https://arxiv.org/abs/2602.00663)

    SEISMO通过利用预测器分数之外的信息（如任务描述、轨迹和可解释性反馈）作为指导信号，显著提升了分子优化的样本效率。

    

    摘要：优化分子以获得所需性质是化学科学中的一个核心瓶颈，尤其是在制药行业，它支撑着新药的发现。由于分子性质评估通常依赖于昂贵且速率受限的预测器（如实验测定），分子优化必须高度样本高效。为解决这一问题，我们引入了SEISMO，一种用于推理时分子优化的LLM智能体，它将现有方法中通常伴随预测器分数可用但被丢弃的信息转化为明确的指导信号。SEISMO不将预测器视为标量黑盒，而是将每个提议条件化于自然语言任务描述、完整优化轨迹以及从事后可解释性方法和子分数分解中衍生的机器可读反馈。在多种药物发现相关任务中，这持续提升了性能。

    arXiv:2602.00663v3 Announce Type: replace  Abstract: Optimizing molecules to achieve desired properties is a central bottleneck across the chemical sciences, particularly in the pharmaceutical industry, where it underlies the discovery of new drugs. Since molecular property evaluation often relies on costly and rate-limited oracles, such as experimental assays, molecular optimization must be highly sample-efficient. To address this, we introduce SEISMO, an LLM agent for inference-time molecular optimisation that turns information routinely available alongside the oracle score, but discarded by existing methods, into an explicit guidance signal. Rather than treating the oracle as a scalar black box, SEISMO conditions each proposal on a natural-language task description, the full optimization trajectory, and machine-readable feedback derived from post-hoc explainability methods and sub-score decompositions. Across a wide range of drug-discovery-relevant tasks, this consistently improves 
    
[^165]: CFM：用于视觉的与语言对齐的概念基础模型

    CFM: Language-aligned Concept Foundation Model for Vision

    [https://arxiv.org/abs/2601.13798](https://arxiv.org/abs/2601.13798)

    CFM通过提供具有空间定位能力的细粒度概念，使视觉基础模型的下游任务可解释，并利用概念共现关系增强解释质量。

    

    与语言对齐的视觉基础模型在多种下游任务中表现出色。然而，它们学习到的表示仍然不透明，这使得解释其决策过程变得困难。近期研究将这些表示分解为人类可解释的概念，但这些概念的空间定位能力较差，且仅限于图像分类任务。在本工作中，我们提出CFM，一种用于视觉的与语言对齐的概念基础模型，它提供细粒度的概念，这些概念具有人类可解释性，并在输入图像中具有空间定位能力。当与具有强语义表示的基础模型配对时，我们能够为其任何下游任务提供解释。通过检查概念的局部共现依赖关系，我们能够定义概念关系，从而改进概念命名并获得更丰富的解释。在基准数据上，我们展示了CFM在分类、分割和解释任务上的性能。

    arXiv:2601.13798v3 Announce Type: replace-cross  Abstract: Language-aligned vision foundation models perform strongly across diverse downstream tasks. Yet, their learned representations remain opaque, making interpreting their decision-making difficult. Recent work decompose these representations into human-interpretable concepts, but provide poor spatial grounding and are limited to image classification tasks. In this work, we propose CFM, a language-aligned concept foundation model for vision that provides fine-grained concepts, which are human-interpretable and spatially grounded in the input image. When paired with a foundation model with strong semantic representations, we get explanations for any of its downstream tasks. Examining local co-occurrence dependencies of concepts allows us to define concept relationships through which we improve concept naming and obtain richer explanations. On benchmark data, we show that CFM provides performance on classification, segmentation, and 
    
[^166]: 全球混合-Vlasov模拟的确定性与概率性神经替代模型

    Deterministic and probabilistic neural surrogates of global hybrid-Vlasov simulations

    [https://arxiv.org/abs/2601.12614](https://arxiv.org/abs/2601.12614)

    本文提出基于图神经网络的确定性（Graph-FM）和概率性（Graph-EFM）替代模型，能够高效且准确地预测全球混合-Vlasov模拟中近地空间等离子体的时空演化，显著降低计算成本。

    

    混合-Vlasov模拟能够解析太阳风-磁层相互作用中的离子动力学效应，但即使是5D（2D+3V）配置也计算成本高昂。我们展示了基于图的机器学习模拟器可以从四次由稳定太阳风条件驱动的5D Vlasiator运行中，学习近地空间中电磁场和离子速度分布函数低阶矩的时空演化。各次运行之间上游离子数密度系统变化，而网格间距保持不变，以扫描离子惯性长度与网格尺寸之比。使用在包含67万个单元的2D空间模拟网格上运行的图神经网络（GNN），我们证明了基于潜变量公式的确定性预测模型（Graph-FM）和概率性集合预测模型（Graph-EFM）都能准确预测未来的等离子体状态。

    arXiv:2601.12614v4 Announce Type: replace-cross  Abstract: Hybrid-Vlasov simulations resolve ion-kinetic effects in the solar wind-magnetosphere interaction, but even 5D (2D + 3V) configurations are computationally expensive. We show that graph-based machine learning emulators can learn the spatiotemporal evolution of electromagnetic fields and lower-order moments of the ion velocity distribution function in near-Earth space from four 5D Vlasiator runs, each driven by steady solar wind conditions. The upstream ion number density is systematically varied between the runs, while the grid spacing is held constant, to scan the ratio of ion inertial length to grid size. Using a graph neural network (GNN) operating on the 2D spatial simulation grid comprising 670k cells, we demonstrate that both a deterministic forecasting model (Graph-FM) and a probabilistic ensemble forecasting model (Graph-EFM) based on a latent variable formulation produce accurate predictions of future plasma states. A 
    
[^167]: GroupSegment-SHAP：基于分组段玩家的多元时间序列Shapley值解释方法

    GroupSegment-SHAP: Shapley Value Explanations with Group-Segment Players for Multivariate Time Series

    [https://arxiv.org/abs/2601.06114](https://arxiv.org/abs/2601.06114)

    本文提出GS-SHAP，一种将多元时间序列解释单元构建为跨变量分组段玩家、并通过Shapley归因量化贡献的新方法，有效捕捉了变量交互与时间动态的联合结构信号。

    

    多元时间序列模型在医疗、工业、能源和金融领域展现出强大的预测性能，但其如何结合跨变量交互与时间动态仍不清楚。SHapley Additive exPlanations（SHAP）被广泛用于模型解释。然而，现有的时间序列变体通常将特征轴和时间轴独立处理，从而割裂了多个变量在特定时间区间内共同形成的结构信号。我们提出了GroupSegment-SHAP（GS-SHAP），该方法基于跨变量依赖性和时间上的分布变化，将解释单元构建为分组段玩家，并通过Shapley归因量化每个单元的贡献。我们在四个真实世界领域评估了GS-SHAP：人类活动识别、电力系统预测、医学信号分析和金融时间序列，并与KernelSHAP、TimeSHAP、SequenceSHAP、WindowSHAP和TSHA进行了比较。

    arXiv:2601.06114v2 Announce Type: replace-cross  Abstract: Multivariate time-series models achieve strong predictive performance in healthcare, industry, energy, and finance, but how they combine cross-variable interactions with temporal dynamics remains unclear. SHapley Additive exPlanations (SHAP) are widely used for interpretation. However, existing time-series variants typically treat the feature and time axes independently, fragmenting structural signals formed jointly by multiple variables over specific intervals. We propose GroupSegment SHAP (GS-SHAP), which constructs explanatory units as group-segment players based on cross-variable dependence and distribution shifts over time, and then quantifies each unit's contribution via Shapley attribution. We evaluate GS-SHAP across four real-world domains: human activity recognition, power-system forecasting, medical signal analysis, and financial time series, and compare it with KernelSHAP, TimeSHAP, SequenceSHAP, WindowSHAP, and TSHA
    
[^168]: AgentOCR：通过光学自压缩重塑智能体历史记录

    AgentOCR: Reimagining Agent History via Optical Self-Compression

    [https://arxiv.org/abs/2601.04786](https://arxiv.org/abs/2601.04786)

    AgentOCR通过将智能体历史压缩为渲染图像并引入分段光学缓存和自压缩机制，显著降低了令牌和内存成本，同时保持了任务性能。

    

    arXiv:2601.04786v3 公告类型：交叉替换 摘要：大型语言模型（LLMs）的最新进展使得通过多轮交互上的强化学习（RL）训练智能体系统成为可能，但实际部署受到快速增长的历史文本记录的瓶颈制约，这些记录会膨胀令牌和内存成本。我们引入了AgentOCR，这是一个利用视觉令牌更高信息密度的框架，通过将累积的观察-行动历史表示为紧凑的渲染图像。为了使多轮滚动生成可扩展，AgentOCR提出了分段光学缓存机制。通过将历史分解为可哈希的段并维护视觉缓存，该机制消除了冗余的重新渲染。在固定渲染之外，AgentOCR引入了智能体自压缩，其中智能体主动发出压缩率，并通过压缩感知奖励进行训练，以自适应地平衡任务成功和令牌效率。我们在具有挑战性的智能体基准测试（如ALFWo）上进行了广泛实验。

    arXiv:2601.04786v3 Announce Type: replace-cross  Abstract: Recent advances in large language models (LLMs) enable agentic systems trained with reinforcement learning (RL) over multi-turn interaction, but practical deployment is bottlenecked by rapidly growing textual histories that inflate token and memory costs. We introduce AgentOCR, a framework that exploits visual tokens' superior information density by representing the accumulated observation-action history as a compact rendered image. To make multi-turn rollouts scalable, AgentOCR proposes segment optical caching. By decomposing history into hashable segments and maintaining a visual cache, this mechanism eliminates redundant re-rendering. Beyond fixed rendering, AgentOCR introduces agentic self-compression, where the agent actively emits a compression rate and is trained with compression-aware reward to adaptively balance task success and token efficiency. We conduct extensive experiments on challenging agentic benchmarks, ALFWo
    
[^169]: 何时深思：通过测试时训练为代码生成实现自适应计算分配

    When to Ponder: Adaptive Compute Allocation for Code Generation via Test-Time Training

    [https://arxiv.org/abs/2601.00894](https://arxiv.org/abs/2601.00894)

    本文提出PonderTTT，一种无需训练的门控策略，通过TTT层的重建损失自适应触发测试时训练，在代码生成中实现高效计算分配，显著提升推理性能。

    

    arXiv:2601.00894v2 公告类型：替换-交叉 摘要：大型语言模型对所有输入施加统一的计算量，而不考虑其难度。我们提出PonderTTT，一种使用TTT层的自监督重建损失来选择性地触发测试时训练（TTT）更新的门控策略。该门控决策本身无需训练——不需要学习分类器或辅助网络；仅需在无标签数据上初步校准一个标量阈值，并通过指数移动平均（EMA）持续调整以维持目标更新率。我们在GPT-2模型（124M至1.5B参数）上的代码语言建模实验（The Stack v2，教师强制困惑度）表明，该信号与推理兼容，无需真实标签。我们的重建门控实现了82-89%的Oracle恢复率，同时完全无需训练，显著优于随机跳过基线（在OOD语言上损失降低高达16%）。

    arXiv:2601.00894v2 Announce Type: replace-cross  Abstract: Large language models apply uniform computation to all inputs, regardless of difficulty. We propose PonderTTT, a gating strategy using the TTT layer's self-supervised reconstruction loss to selectively trigger Test-Time Training (TTT) updates. The gating decision itself is training-free--requiring no learned classifier or auxiliary networks; only a single scalar threshold is initially calibrated on unlabeled data and continuously adapted via EMA to maintain target update rates. Our experiments with GPT-2 models (124M to 1.5B) on code language modeling (The Stack v2, teacher-forced perplexity) demonstrate that this signal is inference-compatible, requiring no ground-truth labels. Our Reconstruction Gating achieves 82-89% Oracle Recovery while being fully training-free, significantly outperforming Random Skip baselines (up to 16% lower loss on OOD languages).
    
[^170]: 逆强化学习与动态离散选择模型的高效推断

    Efficient Inference for Inverse Reinforcement Learning and Dynamic Discrete Choice Models

    [https://arxiv.org/abs/2512.24407](https://arxiv.org/abs/2512.24407)

    该论文提出了一种半参数去偏框架，通过将对数行为策略视为伪奖励，实现了在灵活奖励表示下对逆强化学习和动态离散选择模型的有效统计推断。

    

    在许多序贯决策问题中，研究者观察到行为但无法直接观测驱动行为的奖励，然而仍希望评估和比较反事实政策。逆强化学习（IRL）和动态离散选择（DDC）模型通过假设一个最优性模型将潜在奖励与观测行为联系起来，以应对这一场景。现有的灵活IRL方法允许丰富的奖励表示，但通常无法提供有效的统计推断，而经典的DDC方法仅在严格的参数结构下支持推断。我们开发了一个半参数框架，用于最大熵IRL和Gumbel冲击DDC模型中的去偏逆强化学习。我们的关键识别结果是，对数行为策略可以被视为一种伪奖励：它点识别政策价值差异，并在归一化约束下识别奖励本身。这减少了对奖励依赖估计量的推断复杂性。

    arXiv:2512.24407v2 Announce Type: replace  Abstract: In many sequential decision-making problems, researchers observe actions but not the rewards that drive behavior, yet still wish to evaluate and compare counterfactual policies. Inverse reinforcement learning (IRL) and dynamic discrete choice (DDC) models address this setting by positing an optimality model that links latent rewards to observed actions. Existing flexible IRL methods allow rich reward representations but typically do not provide valid inference, whereas classical DDC methods support inference only under restrictive parametric structure. We develop a semiparametric framework for debiased inverse reinforcement learning in maximum-entropy IRL and Gumbel-shock DDC models. Our key identification result is that the log-behavior policy can be treated as a pseudo-reward: it point-identifies policy value differences and, under a normalization constraint, the reward itself. This reduces inference on reward-dependent estimands t
    
[^171]: 主动学习多个计算机实验的联合等高线

    Actively Learning Joint Contours of Multiple Computer Experiments

    [https://arxiv.org/abs/2512.13530](https://arxiv.org/abs/2512.13530)

    本文提出了一种联合等高线定位（jCL）方案，通过两种采集方案和决策规则，同时识别多个计算机实验的预定响应值，应用于飞行稳定条件识别。

    

    摘要：等高线定位——即通过顺序训练代理模型，从单一计算机实验中识别出导致预定响应值的设计输入的过程——是一个研究充分的主动学习问题。在这里，我们处理一个相关但不同的问题：同时识别出多个计算机实验返回预定值的输入配置。受飞行中车辆所受旋转力矩的计算机实验启发，我们旨在识别导致零力矩力的稳定飞行条件。我们提出了一种“联合等高线定位”（jCL）方案，该方案在探索多个响应曲面与利用相交等高线学习之间实现了战略平衡。我们不是将探索和利用整合到单一的采集函数中，而是设计了两种不同的采集方案，并配有一个决策规则来在两者之间进行选择。

    arXiv:2512.13530v2 Announce Type: replace-cross  Abstract: Contour location---the process of sequentially training a surrogate model to identify the design inputs that result in a pre-specified response value from a single computer experiment---is a well-studied active learning problem. Here, we tackle a related but distinct problem: identifying the input configuration that returns pre-specified values of multiple computer experiments simultaneously. Motivated by computer experiments of the rotational torques acting upon a vehicle in flight, we aim to identify stable flight conditions that result in zero torque forces. We propose a ``joint contour location'' (jCL) scheme that strikes a strategic balance between exploring the multiple response surfaces while exploiting learning of the intersecting contours. Rather than working exploration and exploitation into a single acquisition function, we devise two distinct acquisition schemes with a decision rule to choose between the two, which 
    
[^172]: MeltwaterBench：深度学习用于地表融水的时空降尺度

    MeltwaterBench: Deep learning for spatiotemporal downscaling of surface meltwater

    [https://arxiv.org/abs/2512.12142](https://arxiv.org/abs/2512.12142)

    本文提出了一种融合遥感与物理模型数据的深度学习模型，实现格陵兰地表融水每日100米分辨率的时空降尺度，显著提高了准确性。

    

    arXiv:2512.12142v2 公告类型：替换交叉 摘要：格陵兰冰盖正在以加速速率融化，其原因尚未完全理解且难以测量。地表融水的分布有助于理解这些过程，并可通过遥感观测，但当前的融水地图面临权衡：它们要么在时间上高分辨率，要么在空间上高分辨率，但不能两者兼得。我们开发了一种深度学习模型，通过融合遥感观测和基于物理模型的数据流，生成每日100米分辨率的格陵兰地表融水网格地图。具体而言，我们利用合成孔径雷达（SAR）、被动微波（PMW）和数字高程模型（DEM）对区域气候模型（RCM）输出进行时空降尺度，覆盖东格陵兰赫尔海姆冰川（2017-2023年）。以SAR衍生的融水作为“地面真值”，我们证明融合所有数据流的深度学习方法比单一数据源方法准确率高出10个百分点以上。

    arXiv:2512.12142v2 Announce Type: replace-cross  Abstract: The Greenland ice sheet is melting at an accelerated rate due to processes that are not fully understood and hard to measure. The distribution of surface meltwater can help understand these processes and is observable through remote sensing, but current maps of meltwater face a trade-off: They are either high-resolution in time or space, but not both. We develop a deep learning model that creates gridded surface meltwater maps at daily 100m resolution by fusing data streams from remote sensing observations and physics-based models. In particular, we spatiotemporally downscale regional climate model (RCM) outputs using synthetic aperture radar (SAR), passive microwave (PMW), and a digital elevation model (DEM) over the Helheim Glacier in Eastern Greenland from 2017-2023. Using SAR-derived meltwater as "ground truth", we show that a deep learning-based method that fuses all data streams is over 10 percentage points more accurate 
    
[^173]: 空间感知无字典Koopman特征函数识别用于建模与控制

    Spatially Aware Dictionary-Free Koopman Eigenfunction Identification for Modeling and Control

    [https://arxiv.org/abs/2511.22648](https://arxiv.org/abs/2511.22648)

    该论文提出了一种无需预定义字典或网络结构的SADFED框架，通过正则化最小二乘和空间插值高效识别Koopman特征函数，并结合KPDE残差提升模型的空间一致性与控制性能。

    

    摘要：提出了一种空间感知无字典特征函数发现（SADFED）框架，用于从数据中识别低秩Koopman模型，而无需预设提升字典、核函数或神经网络特征函数架构。该框架选择一条参考轨迹，并通过正则化最小二乘法（LS）确定Koopman模态。随后，利用变换后的时间基，通过第二次正则化最小二乘投影，获得所有采样初始条件下的特征函数值。因此，仅剩特征值的实部和虚部作为优化变量。对识别出的特征函数样本进行插值，揭示其空间结构，从而能够数值估计其梯度。联合目标函数结合了轨迹重建误差与归一化Koopman偏微分方程（KPDE）残差，促进了采样区域内与KPDE的空间一致性。

    arXiv:2511.22648v2 Announce Type: replace  Abstract: A spatially aware dictionary-free eigenfunction discovery (SADFED) framework is proposed for identification of low-rank Koopman models from data without prescribing a lifting dictionary, kernel, or neural-network eigenfunction architecture. A reference trajectory is selected and used to determine the Koopman modes by regularized least squares (LS). Then, a transformed temporal basis allows the eigenfunction values at all sampled initial conditions to be obtained by a second regularized LS projection. Consequently, only the real and imaginary parts of the eigenvalues remain as the optimization variables. Interpolation of the identified eigenfunction samples reveals their spatial structure, enabling numerical estimation of their gradients. A joint objective combines trajectory reconstruction error with a normalized Koopman partial differential equation (KPDE) residual, promoting spatial consistency with the KPDE over the sampled region
    
[^174]: BIPPO：面向节能联邦学习服务的预算感知独立PPO算法

    BIPPO: Budget-Aware Independent PPO for Energy-Efficient Federated Learning Services

    [https://arxiv.org/abs/2511.08142](https://arxiv.org/abs/2511.08142)

    提出了一种节能的多智能体强化学习方案BIPPO，通过预算感知和独立PPO优化，在资源受限的物联网环境中提升了联邦学习的客户端选择性能与能效。

    

    联邦学习（FL）是大规模物联网系统中一种有前景的机器学习解决方案，可保证负载均衡和隐私保护。然而，FL本身并未考虑基础设施效率，这在资源受限环境中运行的系统是一个关键问题。多种基于强化学习（RL）的解决方案改善了FL中的客户端选择，但它们未考虑基础设施挑战，如资源限制和设备更替。此外，RL方法的训练通常未针对实际应用设计，因为这些方法经常不考虑泛化能力，也未针对能源效率进行优化。为填补这一空白，我们提出了BIPPO（预算感知独立近端策略优化），这是一种节能的多智能体RL解决方案，可提升性能。我们在高度预算受限的设置下，对两个图像分类任务评估了BIPPO。

    arXiv:2511.08142v2 Announce Type: replace  Abstract: Federated Learning (FL) is a promising machine learning solution in large-scale IoT systems, guaranteeing load distribution and privacy. However, FL does not natively consider infrastructure efficiency, a critical concern for systems operating in resource-constrained environments. Several Reinforcement Learning (RL) based solutions offer improved client selection for FL; however, they do not consider infrastructure challenges, such as resource limitations and device churn. Furthermore, the training of RL methods is often not designed for practical application, as these approaches frequently do not consider generalizability and are not optimized for energy efficiency. To fill this gap, we propose BIPPO (Budget-aware Independent Proximal Policy Optimization), which is an energy-efficient multi-agent RL solution that improves performance. We evaluate BIPPO on two image classification tasks run in a highly budget-constrained setting, wit
    
[^175]: 噪声标签检测方法的基准测试

    Benchmarking noisy label detection methods

    [https://arxiv.org/abs/2510.16211](https://arxiv.org/abs/2510.16211)

    本文通过将噪声标签检测方法分解为三个核心组件，提出统一基准任务和新指标，系统比较了多种方法在视觉和表格数据上的性能。

    

    标签噪声是现实世界数据集中的常见问题，影响模型训练和验证。干净数据对于实现强性能和确保可靠评估至关重要。尽管已提出多种检测噪声标签（或标签错误）的技术，但对于最优方法尚无明确共识。我们通过将检测方法分解为三个基本组成部分来进行全面基准测试：收集策略（样本内与样本外）、标签不一致度量以及聚合方法。这种分解可应用于许多现有检测方法，并能够跨不同方法进行系统性比较。为公平比较方法，我们提出了一个统一的基准任务：检测与数据集噪声率相等的训练样本比例。我们还引入了一个新颖的指标：在该固定操作点上的假阴性率。我们的评估涵盖视觉和表格数据。

    arXiv:2510.16211v2 Announce Type: replace  Abstract: Label noise is a common problem in real-world datasets, affecting both model training and validation. Clean data are essential for achieving strong performance and ensuring reliable evaluation. While various techniques have been proposed to detect noisy labels (or label errors), there is no clear consensus on optimal approaches. We perform a comprehensive benchmark of detection methods by decomposing them into three fundamental components: gathering strategy (in-sample vs out-of-sample), label disagreement measure, and aggregation method. This decomposition can be applied to many existing detection methods, and enables systematic comparison across diverse approaches. To fairly compare methods, we propose a unified benchmark task: detecting a fraction of training samples equal to the dataset's noise rate. We also introduce a novel metric: the false negative rate at this fixed operating point. Our evaluation spans vision and tabular da
    
[^176]: LTR-ICD：一种面向自动ICD编码的排序感知框架

    LTR-ICD: A Ranking-Aware Framework for Automatic ICD Coding

    [https://arxiv.org/abs/2510.13922](https://arxiv.org/abs/2510.13922)

    本文首次将ICD编码问题从检索视角重新定义为分类与排序任务，提出排序感知框架，显著提升了高优先级诊断代码的识别与排序准确性。

    

    临床笔记包含临床医生在患者就诊期间提供的非结构化文本。这些笔记通常伴随一系列遵循国际疾病分类（ICD）的诊断代码。正确分配和排序ICD代码对于医疗诊断和报销至关重要。然而，自动化此任务仍具挑战性。最先进的方法将此问题视为分类任务，导致忽略了ICD代码的顺序，而顺序对不同目的至关重要。在本工作中，作为首次尝试，我们从检索系统的视角处理此任务，以考虑代码顺序，从而将此问题表述为分类和排序任务。我们的结果和分析表明，所提出的框架在识别高优先级代码方面优于其他方法。例如，我们的模型在正确排序主要诊断代码方面的准确性更高。

    arXiv:2510.13922v2 Announce Type: replace-cross  Abstract: Clinical notes contain unstructured text provided by clinicians during patient encounters. These notes are usually accompanied by a sequence of diagnostic codes following the International Classification of Diseases (ICD). Correctly assigning and ordering ICD codes is essential for medical diagnosis and reimbursement. However, automating this task remains challenging. State-of-the-art methods treated this problem as a classification task, leading to ignoring the order of ICD codes that is essential for different purposes. In this work, as a first attempt, we approach this task from a retrieval system perspective to consider the order of codes, thus formulating this problem as a classification and ranking task. Our results and analysis show that the proposed framework has a superior ability to identify high-priority codes compared to other methods. For instance, our model's accuracy in correctly ranking primary diagnosis codes i
    
[^177]: 拉什莫尔医生与疯狂的多重宇宙：未观测混杂与拉什莫尔效应下的变量重要性

    Doctor Rashomon and the UNIVERSE of Madness: Variable Importance with Unobserved Confounding and the Rashomon Effect

    [https://arxiv.org/abs/2510.12734](https://arxiv.org/abs/2510.12734)

    本文提出UNIVERSE方法，利用拉什莫尔集对存在未观测混杂和特征缺失时的变量重要性进行有界推断，兼顾模型不确定性，提供理论保障并验证了性能。

    

    变量重要性（VI）方法常用于假设生成、特征选择与科学验证。在标准VI流程中，分析者仅基于观测特征对单一预测模型估计VI。然而，特征的重要性高度依赖于模型中包含的其他变量，且关键变量往往在观测数据集中被遗漏。此外，针对一个模型估计的VI通常与针对另一个同等优秀模型估计的VI不同——这一现象被称为拉什莫尔效应。我们通过引入“未观测变量与拉什莫尔集推断变量重要性”（UNIVERSE）方法来解决这些缺口。我们的方法利用拉什莫尔集（即数据集中近似最优模型的集合）来生成真实VI的界限，即使在特征缺失的情况下也能实现。我们从理论上保证了该方法的稳健性，并在半合成模拟中展示了其强劲性能。

    arXiv:2510.12734v2 Announce Type: replace  Abstract: Variable importance (VI) methods are often used for hypothesis generation, feature selection, and scientific validation. In the standard VI pipeline, an analyst estimates VI for a single predictive model with only the observed features. However, the importance of a feature depends heavily on which other variables are included in the model, and essential variables are often omitted from observational datasets. Moreover, the VI estimated for one model is often not the same as the VI estimated for another equally-good model - a phenomenon known as the Rashomon Effect. We address these gaps by introducing UNobservables and Inference for Variable importancE using Rashomon SEts (UNIVERSE). Our approach adapts Rashomon sets = the sets of near-optimal models in a dataset - to produce bounds on the true VI even with missing features. We theoretically guarantee the robustness of our approach, show strong performance on semi-synthetic simulatio
    
[^178]: 珀尔修斯：通过有状态记忆实现稀疏监督的交互式时间序列分割

    Perseus: Interactive Time Series Segmentation with Sparse Supervision via Stateful Memory

    [https://arxiv.org/abs/2510.09930](https://arxiv.org/abs/2510.09930)

    珀尔修斯通过有状态记忆机制，将用户稀疏提示跨时间窗口持久保留，实现无需重训练的交互式时间序列分割。

    

    现实世界的系统，从工业制造到可穿戴医疗保健，生成具有从粗略状态到细粒度事件的分层状态的多变量时间序列。与零样本或少样本分割不同，我们的设置使用密集状态标签进行模型训练。稀疏专家提示提供推理时的修正，以解决序列特定的歧义，而无需重新训练。在实践中，这种反馈集中在选定的事件或转换周围，留下时间线的大部分未提示。这里评估的基于提示的滑动窗口基线相对于用户交互历史是无状态的：它们仅在当前窗口内使用指导，无法在这些间隙中保留它。为解决这一问题，我们提出了珀尔修斯（带用户监督的持久分割），一个从同步处理过渡到异步状态管理的框架。珀尔修斯解耦了监督

    arXiv:2510.09930v2 Announce Type: replace-cross  Abstract: Real-world systems, ranging from industrial manufacturing to wearable healthcare, generate multivariate time series with hierarchical states ranging from coarse regimes to fine-grained events. Unlike zero- or few-shot segmentation, our setting uses dense state labels for model training. Sparse expert prompts provide inference-time corrections that resolve sequence-specific ambiguities without retraining. In practice, this feedback is grouped around selected events or transitions, leaving large portions of the timeline unprompted. The prompt-based sliding-window baselines evaluated here are stateless with respect to user interaction history: they use guidance only within the current window and cannot retain it across these gaps. To address this, we propose Perseus (Persistent Segmentation with User Supervision), a framework that transitions from synchronous processing to asynchronous state management. Perseus decouples supervisi
    
[^179]: HIP：无需导数的海森原子间势

    HIP: Hessian Interatomic Potentials without derivatives

    [https://arxiv.org/abs/2509.21624](https://arxiv.org/abs/2509.21624)

    本文提出了一种直接预测分子海森矩阵的深度学习模型HIP，无需导数计算，在速度、精度和扩展性上显著优于现有方法。

    

    arXiv:2509.21624v4 公告类型：替换 摘要：分子海森矩阵，即势能的二阶导数，是计算化学中许多工作流程的基础。通常，无论使用量子化学方法还是机器学习原子间势（MLIPs），精确的海森矩阵计算成本高昂，且随系统规模扩展性差。在这项工作中，我们引入了海森原子间势（HIPs），这是一种深度学习模型，直接预测海森矩阵，无需依赖自动微分或有限差分。为此，我们从图神经网络计算的不可约表示（irrep）特征（最高到$l$=2度）构建SE(3)-等变、对称的海森矩阵。HIP海森矩阵在速度上快一到两个数量级，更准确、更节省内存、更易训练，并且随系统规模展现出更优的扩展性。我们在广泛的后续任务中验证了我们的预测，展示了其一致性。

    arXiv:2509.21624v4 Announce Type: replace  Abstract: Molecular Hessians, the second derivatives of the potential energy, are fundamental to many workflows in computational chemistry. Usually, accurate Hessians are computationally expensive to calculate and scale poorly with system size, whether computed using quantum chemistry methods or machine-learning interatomic potentials (MLIPs). In this work, we introduce Hessian interatomic potentials (HIPs), a deep learning model that directly predicts Hessians without relying on automatic differentiation or finite differences. To do so, we construct SE(3)-equivariant, symmetric Hessians from irreducible representation (irrep) features up to degree $l$=2, computed by a graph neural network. HIP Hessians are one to two orders of magnitude faster, more accurate, more memory efficient, easier to train, and exhibit more favourable scaling with system size. We validate our predictions across a wide range of downstream tasks, demonstrating consisten
    
[^180]: AIRL-S：通过对抗逆向强化学习统一强化学习与基于搜索的测试时扩展

    AIRL-S: Unifying Reinforcement Learning and Search-Based Test-Time Scaling via Adversarial Inverse Reinforcement Learning

    [https://arxiv.org/abs/2508.14313](https://arxiv.org/abs/2508.14313)

    AIRL-S通过对抗逆向强化学习从参考轨迹中推断密集逐步奖励，消除了对标记过程数据的依赖，并统一了强化学习与基于搜索的测试时扩展，在多个基准上显著提升了模型性能。

    

    arXiv:2508.14313v4 公告类型：交叉替换  摘要：大型语言模型的测试时扩展策略主要依赖于稀疏结果奖励的强化学习或由静态过程奖励模型引导的基于搜索的方法。然而，基于结果的强化学习常常面临训练不稳定和样本效率低下的问题，而静态PRM需要昂贵的逐步监督，并且容易因分布偏移而遭受奖励黑客攻击。在本文中，我们引入了AIRL-S，一个将对抗逆向强化学习与组相对策略优化相统一的框架。通过直接从参考轨迹推断出密集的、逐步的奖励模型，AIRL-S消除了对标记过程数据的依赖，并使用同一学习到的PRM作为训练信号和基于搜索的测试时扩展的验证器。在数学、科学和代码生成的八个基准上的广泛评估表明，我们的策略模型提高了平均性能。

    arXiv:2508.14313v4 Announce Type: replace-cross  Abstract: Test-time scaling strategies for Large Language Models predominantly rely on either reinforcement learning with sparse outcome rewards or search-based methods guided by static Process Reward Models. However, outcome-based RL often suffers from training instability and sample inefficiency, while static PRMs require expensive step-wise supervision and are susceptible to reward hacking due to distributional shifts. In this paper, we introduce AIRL-S, a unified framework that integrates Adversarial Inverse Reinforcement Learning with Group Relative Policy Optimization. By inferring a dense, step-wise reward model directly from reference trajectories, AIRL-S eliminates the dependency on labeled process data and uses the same learned PRM as both a training signal and a verifier for search-based TTS. Extensive evaluations across eight benchmarks in mathematics, science, and code generation demonstrate that our policy model improves av
    
[^181]: 查询高效的结构化矩阵学习

    Query Efficient Structured Matrix Learning

    [https://arxiv.org/abs/2507.19290](https://arxiv.org/abs/2507.19290)

    本文首次从一般矩阵族角度研究结构化矩阵学习的查询复杂度，提出从有限矩阵族中寻找近似最优近似的通用方法。

    

    arXiv:2507.19290v2 公告类型：替换交叉 摘要：我们研究了在给定矩阵-向量乘积（matvec）查询形式为$x \rightarrow Ax$和$x \rightarrow A^Tx$的情况下，学习未知矩阵$A$的结构化近似（低秩、稀疏、带状等）的问题。该问题在科学计算和机器学习算法中具有核心重要性，其应用包括结构化矩阵的快速乘法和求逆、为一阶优化构建预处理器，以及作为微分算子学习的模型。先前的工作侧重于为应用中常见的特定结构化矩阵族获取查询复杂度的上下界。我们以更普遍的方式启动对该问题的研究，旨在理解从一般矩阵族学习近似的查询复杂度。我们的主要结果集中于从任何有限大小的m个矩阵族中找到$A$的近似最优近似。

    arXiv:2507.19290v2 Announce Type: replace-cross  Abstract: We study the problem of learning a structured approximation (low-rank, sparse, banded, etc.) to an unknown matrix $A$ given access to matrix-vector product (matvec) queries of the form $x \rightarrow Ax$ and $x \rightarrow A^Tx$. This problem is of central importance to algorithms across scientific computing and machine learning, with applications to fast multiplication and inversion for structured matrices, building preconditioners for first-order optimization, and as a model for differential operator learning. Prior work focuses on obtaining query complexity upper and lower bounds for learning specific structured matrix families that commonly arise in applications.   We initiate the study of the problem in greater generality, aiming to understand the query complexity of learning approximations from general matrix families. Our main result focuses on finding a near-optimal approximation to $A$ from any finite-sized family of m
    
[^182]: CPC-CMS：面向文档级情感分析的认知成对比较分类模型选择框架

    CPC-CMS: Cognitive Pairwise Comparison Classification Model Selection Framework for Document-level Sentiment Analysis

    [https://arxiv.org/abs/2507.14022](https://arxiv.org/abs/2507.14022)

    该框架通过认知成对比较加权多标准评估，自动选择文档级情感分析的最优分类模型，并在多个数据集上验证了其有效性。

    

    本研究提出了用于文档级情感分析的认知成对比较分类模型选择（CPC-CMS）框架。基于专家知识判断的CPC方法被用于计算评估标准的权重，这些标准包括准确率、精确率、召回率、F1分数、特异度、马修斯相关系数（MCC）、科恩卡帕系数（Kappa）和效率。选择朴素贝叶斯（NB）、线性支持向量分类（LSVC）、随机森林、逻辑回归、极端梯度提升（XGBoost）、长短期记忆网络（LSTM）和轻量级双向编码器表示（ALBERT）作为分类基线模型。通过形成由分类评估分数相对于标准权重组成的加权决策矩阵，为分类问题选择最佳分类模型。使用三个开放社交媒体数据集来证明所提方法的可行性。

    arXiv:2507.14022v3 Announce Type: replace  Abstract: This study proposes the Cognitive Pairwise Comparison Classification Model Selection (CPC-CMS) framework for document-level sentiment analysis. The CPC, based on expert knowledge judgment, is used to calculate the weights of evaluation criteria, including accuracy, precision, recall, F1-score, specificity, Matthews Correlation Coefficient (MCC), Cohen's Kappa (Kappa), and efficiency. Naive Bayes (NB), Linear Support Vector Classification (LSVC), Random Forest, Logistic Regression, Extreme Gradient Boosting (XGBoost), Long Short-Term Memory (LSTM), and A Lite Bidirectional Encoder Representations from Transformers (ALBERT) are chosen as classification baseline models. A weighted decision matrix consisting of classification evaluation scores with respect to criteria weights is formed to select the best classification model for a classification problem. Three open social media datasets are used to demonstrate the feasibility of the prop
    
[^183]: 用机械可解释性解释内在道德自我修正

    Explaining Intrinsic Moral Self-Correction with Mechanistic Interpretability

    [https://arxiv.org/abs/2505.11924](https://arxiv.org/abs/2505.11924)

    该论文通过机械可解释性揭示了内在道德自我修正的机制是表示引导，即提示词通过沿可解释潜在方向调整隐藏表示来改变模型行为，且这种方法比直接提示更有效。

    

    arXiv:2505.11924v4 公告类型：替换交叉 摘要：内在道德自我修正指的是语言模型仅通过提示词来优化其伦理判断或调整其输出的现象。尽管在多种任务中有效，但其机制仍不清楚。我们假设内在道德自我修正通过将隐藏表示沿可解释的潜在方向引导来起作用。通过评估六个大型语言模型在四个道德相关任务上的表现，我们证明了自我修正提示引起的表示变化与对比性引导向量一致。即使引导向量是从不相关语料库构建的，这种一致性也能转移。值得注意的是，当通过激活添加应用时，这些提示引起的偏移能比自我修正提示和引导向量更有效地改变模型行为。我们的发现表明，表示引导是内在道德自我修正的机制驱动因素。

    arXiv:2505.11924v4 Announce Type: replace-cross  Abstract: Intrinsic moral self-correction refers to the phenomenon where a language model refines its ethical judgments or aligns its outputs purely through prompting. While effective across diverse tasks, its mechanism remains unclear. We hypothesize intrinsic moral self-correction functions by steering hidden representations along interpretable latent directions. Evaluating six LLMs across four morality-related tasks, we demonstrate that the representation shifts induced by self-correction prompts align with contrastive steering vectors. This alignment transfers even when the steering vectors are constructed from a disjoint corpus. Notably, when applied via activation addition, these prompt-induced shifts can alter model behavior more effectively than the self-correction prompts and the steering vectors. Our findings suggest representation steering is the mechanistic driver of intrinsic moral self-correction.
    
[^184]: 通过细粒度奖励结构与信用分配增强LLM智能体的多轮推理能力

    Reinforcing Multi-Turn Reasoning in LLM Agents via Fine-Grained Reward Structure and Credit Assignment

    [https://arxiv.org/abs/2505.11821](https://arxiv.org/abs/2505.11821)

    本文提出利用密集回合级奖励结构（终端、延迟、每回合）在GRPO和PPO中实现细粒度信用分配，从而提升LLM智能体在多轮推理任务中的强化学习效果。

    

    强化学习（RL）方法已被广泛用于增强大语言模型（LLM）智能体在长时域、多轮场景中的推理能力。这种交互可以形式化为回合级马尔可夫决策过程（MDPs），其中中间奖励通常是可用的。然而，大多数先前的工作依赖稀疏的轨迹级奖励，导致信用分配不佳，而密集的回合级奖励仍未得到充分探索。在本文中，我们研究了如何在RL算法中有效利用密集的回合级奖励结构，特别是组相对策略优化（GRPO）和近端策略优化（PPO），以实现细粒度的信用分配。我们根据奖励粒度将奖励结构分为三种类型：（1）终端奖励；（2）延迟奖励；（3）每回合奖励，每种对应不同的回合级MDP公式，并推导出GRPO和PPO算法。

    arXiv:2505.11821v3 Announce Type: replace  Abstract: Reinforcement Learning (RL) approaches have been wildly used to enhance the reasoning capabilities of Large Language Model (LLM) agents in long-horizon, multi-turn scenarios. Such interactions can be formalized as turn-level Markov decision processes (MDPs), where intermediate rewards are often available. However, most prior work relies on sparse trajectory-level rewards, resulting in poor credit assignment, while dense turn-level rewards remain underexplored. In this paper, we investigate how to effectively leverage dense turn-level reward structures in RL algorithms, specifically Group Relative Policy Optimization (GRPO) and Proximal Policy Optimization (PPO), to enable fine-grained credit assignment. We categorize reward structures into three types based on their granularity: (1) terminal reward; (2) delayed reward; (3) per-turn reward, each corresponding to a distinct turn-level MDP formulation, and derive GRPO and PPO algorithms
    
[^185]: SPD矩阵学习用于神经影像分析：视角、方法与挑战

    SPD Matrix Learning for Neuroimaging Analysis: Perspectives, Methods, and Challenges

    [https://arxiv.org/abs/2504.18882](https://arxiv.org/abs/2504.18882)

    本文系统综述了SPD矩阵学习在神经影像分析中的统一框架，从模态特定表示到几何深度学习范式，强调了其连接经典统计与现代机器学习的核心贡献。

    

    神经影像通过捕捉大脑组织互补方面的模态，为表征脑活动、结构和连接性提供了重要工具。在这些多样化的模态中，当测量通过适当的估计或正则化过程建模为对称正定（SPD）值表示时，会出现一个统一的视角。凭借黎曼几何，SPD流形为非欧几里得框架下的原则性统计推断和机器学习提供了基础。本综述将这些分析和学习方法组织在一个SPD矩阵学习框架内，该框架连接了经典几何统计与现代机器学习在神经影像和神经生理学应用中的联系。我们系统地调查了从模态特定表示到几何浅层和深层学习范式的进展，重点突出了关键进展和挑战。

    arXiv:2504.18882v3 Announce Type: replace-cross  Abstract: Neuroimaging provides essential tools for characterizing brain activity, structure, and connectivity through modalities that capture complementary aspects of brain organization. Across these diverse modalities, a unifying perspective arises when measurements are modeled as symmetric positive-definite (SPD)-valued representations through appropriate estimation or regularization procedures. Endowed with Riemannian geometry, the SPD manifold provides a non-Euclidean framework for principled statistical inference and machine learning on these representations. This review organizes these analytical and learning approaches within a framework for SPD matrix learning that connects classical geometric statistics with modern machine learning across neuroimaging and neurophysiological applications. We systematically survey the progression from modality-specific representations to geometric shallow and deep learning paradigms, highlighting
    
[^186]: 一种少样本鸟类叫声分类的自动化流水线：以齿嘴鸠为例的研究

    An Automated Pipeline for Few-Shot Bird Call Classification: A Case Study with the Tooth-Billed Pigeon

    [https://arxiv.org/abs/2504.16276](https://arxiv.org/abs/2504.16276)

    本文提出了一种针对仅有少量录音的稀有鸟类（如齿嘴鸠）的自动化单样本叫声分类流水线，通过利用大型分类网络的嵌入空间和余弦相似度，结合预处理技术，在最小训练数据下实现高效检测。

    

    arXiv:2504.16276v3 公告类型：替换-交叉 摘要：本文提出了一种基本自动化的单样本鸟类叫声分类流水线，结合了针对性的手动质量控制步骤，专为在大型公开分类器（如BirdNET和Perch）中缺失的稀有物种设计。虽然这些模型在检测具有丰富训练数据的常见鸟类方面表现出色，但它们无法处理仅有1-3个已知录音的物种，这对于监测濒危鸟类最后剩余个体的保护工作者来说是一个关键限制。为解决这一问题，我们利用大型鸟类分类网络的嵌入空间，开发了一种基于余弦相似度的分类器，并结合过滤和去噪预处理技术，以在最小训练数据下优化检测。我们使用聚类指标评估各种嵌入空间，并在模拟场景（使用Xeno-Canto录音）和真实世界测试（针对极度濒危的齿嘴鸠）中验证了我们的方法。

    arXiv:2504.16276v3 Announce Type: replace-cross  Abstract: This paper presents a largely automated one-shot bird call classification pipeline, incorporating targeted manual quality control steps, designed for rare species absent from large publicly available classifiers like BirdNET and Perch. While these models excel at detecting common birds with abundant training data, they lack options for species with only 1-3 known recordings, a critical limitation for conservationists monitoring the last remaining individuals of endangered birds. To address this, we leverage the embedding space of large bird classification networks and develop a classifier using cosine similarity, combined with filtering and denoising preprocessing techniques, to optimize detection with minimal training data. We evaluate various embedding spaces using clustering metrics and validate our approach in both a simulated scenario with Xeno-Canto recordings and a real-world test on the critically endangered tooth-bille
    
[^187]: 基于有界不确定性模型的强化学习智能探索

    Smart Exploration in Reinforcement Learning using Bounded Uncertainty Models

    [https://arxiv.org/abs/2504.05978](https://arxiv.org/abs/2504.05978)

    本文提出利用有界不确定性模型集合优化Q函数上下界来引导强化学习探索，并提供收敛性理论保证及数据驱动正则化方法，以加速学习并确保策略收敛到最优。

    

    强化学习（RL）是在不确定环境中进行决策的强大框架，但它通常需要大量数据来学习最优策略。我们通过引入先验模型知识来指导探索并加速学习过程，从而解决这一挑战。具体而言，我们假设可以访问一个包含真实转移核和奖励函数的模型集合。我们在此模型集合上进行优化，以获得Q函数的上界和下界，然后利用这些界来指导智能体的探索。我们提供了在提议的探索策略类别下Q函数收敛到最优Q函数的理论保证。此外，我们还引入了模型集合优化问题的数据驱动正则化版本，以确保探索策略类别收敛到最优策略。最后，我们展示了当模型集合具有特定结构时，n（原文截断）

    arXiv:2504.05978v4 Announce Type: replace  Abstract: Reinforcement learning (RL) is a powerful framework for decision-making in uncertain environments, but it often requires large amounts of data to learn an optimal policy. We address this challenge by incorporating prior model knowledge to guide exploration and accelerate the learning process. Specifically, we assume access to a model set that contains the true transition kernel and reward function. We optimize over this model set to obtain upper and lower bounds on the Q-function, which are then used to guide the exploration of the agent. We provide theoretical guarantees on the convergence of the Q-function to the optimal Q-function under the proposed class of exploring policies. Furthermore, we also introduce a data-driven regularized version of the model set optimization problem that ensures the convergence of the class of exploring policies to the optimal policy. Lastly, we show that when the model set has a specific structure, n
    
[^188]: 存在选择偏差和混杂情况下基于回归的因果效应估计

    Regression-Based Estimation of Causal Effects in the Presence of Selection Bias and Confounding

    [https://arxiv.org/abs/2503.20546](https://arxiv.org/abs/2503.20546)

    本文提出在存在选择偏差和混杂时，利用代理变量从外部无偏数据校正偏差，以回归方法可靠估计因果效应E[Y|do(X)]的理论条件。

    

    我们考虑在通过干预设定处理变量X时，估计目标变量Y的期望因果效应E[Y|do(X)]的问题，重点关注连续随机变量。在没有选择偏差或混杂的情况下，E[Y|do(X)] = E[Y|X]，这可以使用标准回归方法进行估计。然而，当选择偏差导致的系统性缺失或混杂扭曲数据时，回归方法会失效。在某些约束条件下，不受选择过程影响的代理变量可用于校正选择偏差，从而可靠地恢复E[Y|X]以及E[Y|do(X)]。当数据还受到混杂影响时，从选择偏差数据中恢复因果效应更具挑战性，需要访问代理变量以同时校正混杂和选择机制。假设从外部无偏观测数据中获取此类代理变量，我们推导了理论条件。

    arXiv:2503.20546v2 Announce Type: replace-cross  Abstract: We consider the problem of estimating the expected causal effect $E[Y|do(X)]$ for a target variable $Y$ when treatment $X$ is set by intervention, focusing on continuous random variables. In settings without selection bias or confounding, $E[Y|do(X)] = E[Y|X]$, which can be estimated using standard regression methods. However, regression fails when systematic missingness induced by selection bias, or confounding distorts the data. Proxy variables unaffected by the selection process can, under certain constraints, be used to correct for selection bias to recover $E[Y|X]$, and hence $E[Y|do(X)]$, reliably. When data is additionally affected by confounding, recovering the causal effect from selection-biased data is more challenging and requires access to proxies to both correct for confounding and for the selection mechanism. Assuming access to such proxies from external unbiased observational data, we derive theoretical condition
    
[^189]: 结构即信息：部分观测动态系统的机器学习结构可辨识性映射

    Structure is information: structural identifiability mappings for machine learning with partially observed dynamical systems

    [https://arxiv.org/abs/2502.04131](https://arxiv.org/abs/2502.04131)

    本文提出了一种结构可辨识性映射方法，用于解决部分观测动态系统中机器学习应用中的模型模糊性问题，从而提升可解释性和数据利用效率。

    

    现代机器学习在时间序列分类中的成功应用，常常受到可用训练数据质量和数量的限制。为克服这些限制，可以利用领域知识，以参数化机械动态模型的形式，将时间序列观测表示为预定义动态系统类别的实例。只要动态模型在领域特定变量及其动态交互方面是可解释的，学习过程也变得可解释，并使建模者能够自然地处理稀疏和不规则采样的数据。然而，动态模型的内部过程往往只能被部分观测。这可能导致关于哪个特定模型实现最能解释给定时间序列观测的模糊性。这一问题在文献中广为人知，具有此问题的动态模型被称为结构不可辨识的。

    arXiv:2502.04131v2 Announce Type: replace  Abstract: The successful application of modern machine learning for time series classification is often hampered by limitations in quality and quantity of available training data. To overcome these limitations, domain knowledge can be leveraged in the form of parameterised mechanistic dynamical models, whereby time series observations may be represented as instances of a predefined class of dynamical systems. Provided the dynamical models are interpretable in terms of domain-specific variables and their dynamic interaction, the learning process becomes interpretable as well and enables the modeller to handle sparsely and irregularly sampled data naturally. However, the internal processes of a dynamical model are often only partially observed. This can lead to ambiguity regarding which particular model realization best explains a given time series observation. This problem is well-known in the literature, and a dynamical model with this issue i
    
[^190]: 大型语言模型内部表示中提示词的内在维度

    The Intrinsic Dimension of Prompts in Internal Representations of Large Language Models

    [https://arxiv.org/abs/2501.10573](https://arxiv.org/abs/2501.10573)

    本文通过内在维度分析大型语言模型提示词表示，发现其与词元不确定性相关，并利用逐层内在维度轮廓训练线性探针，在生成前高效区分恶意与良性提示，准确率达90-95%。

    

    arXiv:2501.10573v2 公告类型：替换 摘要：我们通过内在维度的视角，研究了大型语言模型中提示词级别的词元表示几何结构。将变换器视为平均场粒子系统，我们估计了每一层经验测度的内在维度，并证明其与下一个词元的不确定性相关。跨模型和内在维度估计器，我们发现内在维度在早期到中层达到峰值，并在句法和语义扰动（通过打乱词元）下增加，且与平均惊异度强相关，通过softmax将逻辑几何与熵联系起来进行简单分析。作为实际可解释性和安全性的案例研究，我们在逐层内在维度轮廓上训练了一个线性探针，以在生成前区分恶意和良性提示词。该探针在不同数据集上达到90%至95%的准确率，优于广泛使用的防护措施。

    arXiv:2501.10573v2 Announce Type: replace  Abstract: We study the geometry of token representations at the prompt level in large language models through the lens of intrinsic dimension. Viewing transformers as mean-field particle systems, we estimate the intrinsic dimension of the empirical measure at each layer and demonstrate that it correlates with next-token uncertainty. Across models and intrinsic dimension estimators, we find that intrinsic dimension peaks in early to middle layers and increases under syntactic and semantic disruption (by shuffling tokens), and that it is strongly correlated with average surprisal, with a simple analysis linking logits geometry to entropy via softmax. As a case study in practical interpretability and safety, we train a linear probe on the per-layer intrinsic dimension profile to distinguish malicious from benign prompts before generation. This probe achieves accuracy of 90 to 95\% in different datasets, outperforming widely used guardrails such a
    
[^191]: DAOP：面向高效MoE推理的数据感知卸载与预测预计算

    DAOP: Data-Aware Offloading and Predictive Pre-Calculation for Efficient MoE Inference

    [https://arxiv.org/abs/2501.10375](https://arxiv.org/abs/2501.10375)

    DAOP通过数据感知的专家动态分配和预测预计算，优化了MoE模型在内存受限设备上的GPU-CPU并行推理，显著减少传输延迟并保持准确性。

    

    混合专家（MoE）模型虽对多种机器学习任务非常有效，但在内存受限设备上部署面临重大挑战。尽管GPU提供快速推理，但其内存相比CPU有限，意味着无法同时在GPU上存储所有专家，导致频繁且昂贵的从CPU内存传输数据，往往抵消GPU的速度优势。为解决此问题，我们提出DAOP，一种设备端MoE推理引擎，用于优化并行GPU-CPU执行。DAOP根据每个序列的激活模式动态分配CPU和GPU之间的专家，并选择性地在CPU上预计算预测的专家，以最小化传输延迟。该方法在各种专家缓存比率下实现高效资源利用，同时通过一种新颖的优雅降级机制保持模型准确性。跨多种数据集的综合评估表明，DAOP优于传统方法。

    arXiv:2501.10375v3 Announce Type: replace-cross  Abstract: Mixture-of-Experts (MoE) models, though highly effective for various machine learning tasks, face significant deployment challenges on memory-constrained devices. While GPUs offer fast inference, their limited memory compared to CPUs means not all experts can be stored on the GPU simultaneously, necessitating frequent, costly data transfers from CPU memory, often negating GPU speed advantages. To address this, we present DAOP, an on-device MoE inference engine to optimize parallel GPU-CPU execution. DAOP dynamically allocates experts between CPU and GPU based on per-sequence activation patterns, and selectively pre-calculates predicted experts on CPUs to minimize transfer latency. This approach enables efficient resource utilization across various expert cache ratios while maintaining model accuracy through a novel graceful degradation mechanism. Comprehensive evaluations across various datasets show that DAOP outperforms tradi
    
[^192]: 联邦与差分隐私下的KL散度估计

    Federated and differentially private estimation of KL divergence

    [https://arxiv.org/abs/2411.16478](https://arxiv.org/abs/2411.16478)

    本文提出FedPriKL方法，在差分隐私保证下，以低敏感性和方差、无偏估计联邦模型中数据的KL散度，实现隐私保护与高精度平衡，且通信开销小。

    

    测量分布漂移是管理分布式敏感数据的关键任务，因为它支撑着广泛的联邦学习和分析应用。然而，在许多实际场景中，直接共享此类信息要么不可取（例如，出于隐私考虑），要么不可行（例如，由于高通信成本）。在本工作中，我们提出了FedPriKL，一种在差分隐私（DP）保证下，跨联邦计算模型估计数据KL散度的新颖方法。我们建立了其理论性质，表明FedPriKL是无偏的，且具有低且有界的敏感性和方差，从而确保在DP下具有强效用。此外，我们进行了实证研究，探索参数选择以优化精度同时最小化通信开销。我们的实验表明，FedPriKL在实现与类似基于采样的非私有估计器相当的准确性的同时，提供了隐私保护。

    arXiv:2411.16478v3 Announce Type: replace  Abstract: Measuring distribution drifts is a key task in managing distributed, sensitive data, as it underpins a wide range of federated learning and analytics applications. In many practical settings, however, directly sharing such information is either undesirable (e.g., due to privacy concerns) or infeasible (e.g., due to high communication costs). In this work, we present FedPriKL, a novel method for estimating the KL divergence of data across federated computational models under differential privacy (DP) guarantees. We establish its theoretical properties, showing that FedPriKL is unbiased with low and bounded sensitivity and variance, thereby ensuring strong utility under DP. In addition, we present an empirical study that explores parameter choices to optimize accuracy while minimizing communication overhead. Our experiments demonstrate that FedPriKL achieves accuracy comparable to a similar sampling-based non-private estimator, while d
    
[^193]: 阿尔茨海默病检测中的类内变异问题研究

    On the Within-class Variation Issue in Alzheimer's Disease Detection

    [https://arxiv.org/abs/2409.16322](https://arxiv.org/abs/2409.16322)

    本文针对阿尔茨海默病检测中的类内变异问题，提出软目标蒸馏和实例级重平衡两种方法，通过估计样本特定概率分数来建模异质性和不平衡，从而提升检测性能。

    

    阿尔茨海默病（AD）检测通常采用机器学习分类模型来区分AD患者与非AD个体。与传统分类任务不同，AD检测涉及显著的类内变异，因为具有相同诊断的个体可能表现出不同程度的认知障碍。我们将该问题归纳为两个方面：类内异质性和实例级不平衡。为了在二元监督下建模这种变异，我们估计样本特定的AD类别概率作为样本分数，并开发了两种相应方法：软目标蒸馏（SoTD）和实例级重平衡（InRe）。在ADReSS和CU-MARVEL语料库上的实验表明，估计的分数与独立的认知评估一致，且所提出的方法提高了AD检测性能。这些发现为建模类内变异提供了见解。

    arXiv:2409.16322v4 Announce Type: replace-cross  Abstract: Alzheimer's Disease (AD) detection commonly employs machine learning classification models to distinguish between individuals with AD and those without. Different from conventional classification tasks, AD detection involves substantial within-class variation, as individuals sharing the same diagnosis may exhibit different degrees of cognitive impairment. We formulate two aspects of this issue: within-class heterogeneity and instance-level imbalance. To model such variation under binary supervision, we estimate sample-specific AD class probabilities as sample scores and develop two corresponding methods: Soft Target Distillation (SoTD) and Instance-level Re-balancing (InRe). Experiments on the ADReSS and CU-MARVEL corpora show that the estimated scores align with independent cognitive assessments and that the proposed approaches improve AD detection performance. These findings provide insights for modeling within-class variatio
    
[^194]: 基于N维朗之万方程和神经常微分方程的预测

    Forecasting with an N-dimensional Langevin Equation and a Neural-Ordinary Differential Equation

    [https://arxiv.org/abs/2405.07359](https://arxiv.org/abs/2405.07359)

    本文提出一种结合N维朗之万方程和神经常微分方程的新型数据驱动框架，用于系统建模和预测具有多重非平稳特征的电价时间序列，填补了现有方法孤立处理非平稳性的空白。

    

    arXiv:2405.07359v2 公告类型：替换版。摘要：在竞争性电力市场中，准确预测日前电价至关重要。尽管平稳电价预测技术已受到广泛关注，但对非平稳方法的研究相对较少，尽管非平稳特征在电力市场中普遍存在。具体而言，现有非平稳技术往往孤立地处理单个非平稳特征，而忽略了同时出现的多重非平稳效应的探索。我们的总体目标是构建一个系统框架，用于建模和预测非平稳电价时间序列，涵盖更广泛的非平稳行为范围。为此，我们开发了一种数据驱动模型，结合了N维朗之万方程（LE）和神经常微分方程（NODE）。其中，LE捕捉细粒度的动态细节。

    arXiv:2405.07359v2 Announce Type: replace  Abstract: Accurate prediction of electricity day-ahead prices is essential in competitive electricity markets. Although stationary electricity-price forecasting techniques have received considerable attention, research on non-stationary methods is comparatively scarce, despite the common prevalence of non-stationary features in electricity markets. Specifically, existing non-stationary techniques will often aim to address individual non-stationary features in isolation, leaving aside the exploration of concurrent multiple non-stationary effects. Our overarching objective here is the formulation of a framework to systematically model and forecast non-stationary electricity-price time series, encompassing the broader scope of non-stationary behavior. For this purpose we develop a data-driven model that combines an N-dimensional Langevin equation (LE) with a neural-ordinary differential equation (NODE). The LE captures fine-grained details of the
    
[^195]: LMC多任务高斯过程模型的精确和高效解决方案

    Exact and efficient solutions of the LMC Multitask Gaussian Process model. (arXiv:2310.12032v1 [cs.LG])

    [http://arxiv.org/abs/2310.12032](http://arxiv.org/abs/2310.12032)

    LMC多任务高斯过程模型的精确解决方案表明，只需对噪声模型进行温和假设，即可实现高效计算。通过引入完整参数化的“投影LMC”模型和边缘似然函数表达式，展示了该方法相对于未经处理的方法的优异性能。

    

    线性共同关联模型（LMC）是一种非常通用的多任务高斯过程模型，用于回归或分类。虽然其表达能力和概念简单性很有吸引力，但朴素实现在数据点数量和任务数量方面具有立方复杂度，使得对大多数应用来说，必须进行近似处理。然而，最近的研究表明，在某些条件下，该模型的潜在过程可以解耦，导致仅与所述过程数量呈线性复杂度。我们在这里扩展了这些结果，从最一般的假设中展示了在LMC的高效精确计算所需的唯一条件是对噪声模型进行温和假设。我们引入了结果的完整参数化“投影LMC”模型，并给出了边缘似然函数的表达式，以实现高效的优化。我们对合成数据进行了参数研究，展示了我们方法相对于未经处理的方法的优异性能。

    The Linear Model of Co-regionalization (LMC) is a very general model of multitask gaussian process for regression or classification. While its expressivity and conceptual simplicity are appealing, naive implementations have cubic complexity in the number of datapoints and number of tasks, making approximations mandatory for most applications. However, recent work has shown that under some conditions the latent processes of the model can be decoupled, leading to a complexity that is only linear in the number of said processes. We here extend these results, showing from the most general assumptions that the only condition necessary to an efficient exact computation of the LMC is a mild hypothesis on the noise model. We introduce a full parametrization of the resulting \emph{projected LMC} model, and an expression of the marginal likelihood enabling efficient optimization. We perform a parametric study on synthetic data to show the excellent performance of our approach, compared to an unr
    

